# OpenVLA + LIBERO Stronger Motion Compensation Design

本文提出两种比当前 RGB-only global translation 更扎实的 motion-compensated
VLA-Cache 方案：

1. **Camera / robot trajectory geometry compensation**：根据机械臂或相机位姿变化，
   用平移和旋转矩阵做视角变换，再比较对齐后的 patch。
2. **Optical flow compensation**：用光流估计帧间运动矢量场，再聚合到 ViT patch
   level，决定哪些视觉 token 可以复用。

目标不是直接替换 OpenVLA policy，而是替换当前
`find_motion_compensated_static_patches(...)` 的 patch correspondence 估计器，
保持后续 VLA-Cache 接口尽量不变：

```text
current frame patch id -> previous frame source patch id
```

最终仍输出：

```python
PatchCorrespondence(
    target_patches=[...],   # 当前帧可复用 patch
    source_patches=[...],   # 对应上一帧 patch
    confidence=[...],
    similarity=[...],
    ...
)
```

## Why Current Scheme Is Weak

当前实现是 RGB-only global 2D translation：

```text
prev RGB, curr RGB
  -> exhaustive global shift search
  -> current patch votes source patch
  -> RGB cosine similarity
```

它的问题：

- 只能解释全局平移，不能解释 wrist camera 的旋转、视差和尺度变化。
- 不能区分相机自运动和物体运动。
- 没有利用 LIBERO/MuJoCo 中可获取的相机 pose、深度、机器人末端轨迹。
- 对真实 moving camera 的核心假设支撑不够强，更像 smoke-stage heuristic。

因此下一步应该把 motion compensation 拆成两条更有说服力的路线：一个是仿真中可控的
geometry/oracle 路线，一个是更接近真实数据可迁移的 optical flow 路线。

## Shared Interface

建议新增统一接口，替代现在的 `find_motion_compensated_static_patches(...)`：

```python
def find_motion_compensated_static_patches_v2(
    curr_image,
    prev_image,
    method: str,
    *,
    curr_depth=None,
    prev_depth=None,
    curr_camera_pose=None,
    prev_camera_pose=None,
    intrinsics=None,
    curr_eef_pose=None,
    prev_eef_pose=None,
    camera_extrinsic_in_eef=None,
    flow_model=None,
    patch_size=14,
    top_k=130,
    min_confidence=0.30,
    sim_threshold=0.70,
):
    ...
```

其中 `method` 可选：

```text
pose_depth
pose_planar
eef_pose
optical_flow
```

为了和现有 VLA-Cache 兼容，所有方法都返回同一种 `PatchCorrespondence`。

## Scheme 1: Camera / Robot Trajectory Geometry Compensation

### Core Idea

如果我们知道相机在连续两帧的位姿：

```text
T_WC_prev: previous camera -> world
T_WC_curr: current camera  -> world
```

并知道相机内参 `K`，就可以把上一帧像素反投影到 3D，再投影到当前帧：

```text
u_prev + depth_prev
  -> X_prev_camera
  -> X_world
  -> X_curr_camera
  -> u_curr
```

这样可以得到 **prev pixel -> curr pixel** 的几何对应关系。再聚合到 patch level：

```text
prev patch id -> curr patch id
```

为了复用 cache，我们需要的是当前 patch 从上一帧哪个 source patch 来：

```text
curr target patch -> prev source patch
```

可以通过 forward warp 后的 vote / z-buffer 得到。

### Preferred Variant: Pose + Depth Warp

这是仿真里最强的版本，也是最适合当 oracle upper bound 的版本。

需要输入：

| 数据 | 来源 |
|---|---|
| RGB | `obs["robot0_eye_in_hand_image"]` |
| depth | LIBERO / robosuite `camera_depths=True`，例如 `obs["robot0_eye_in_hand_depth"]` |
| camera pose | MuJoCo camera pose，或由 eef pose + camera extrinsic 推出 |
| intrinsics | camera fovy + image width/height 推出 |

局部公式：

```text
p_prev = [u_prev, v_prev, 1]^T
X_prev_C = depth_prev(u, v) * K^{-1} p_prev
X_W      = T_WC_prev * X_prev_C
X_curr_C = T_CW_curr * X_W
p_curr   = K * X_curr_C
u_curr   = p_curr.x / p_curr.z
v_curr   = p_curr.y / p_curr.z
```

其中：

```text
T_CW_curr = inverse(T_WC_curr)
```

对每个上一帧像素或 patch-sampled point 做 forward warp，落到当前帧有效区域后：

1. 记录 source patch id 和 target patch id。
2. 使用 depth 做 z-buffer，避免被遮挡点错误投票。
3. 对每个 current target patch 汇总来自 previous source patches 的投票。
4. 选票数最多的 source patch 作为 correspondence。
5. 用 RGB / feature similarity 做最后过滤。

Patch-level score 建议：

```text
confidence = valid_votes_for_best_source / total_samples_in_target_patch
similarity = cosine(curr_patch_rgb, prev_source_patch_rgb)
score = confidence * similarity * visibility_ratio
```

输出 top-k：

```text
target_patches = current frame reusable patches
source_patches = previous frame source patches
```

### Practical Data Access in LIBERO

LIBERO 已有 dataset creation 脚本支持：

```text
camera_names = ["robot0_eye_in_hand", "agentview"]
camera_depths = True
```

相关 observation key 预期包括：

```text
robot0_eye_in_hand_image
robot0_eye_in_hand_depth
agentview_image
agentview_depth
```

在线 rollout 里需要把 `get_libero_env(...)` 扩展为：

```python
env_args = {
    ...
    "camera_names": [camera_name],
    "camera_depths": use_depth,
}
```

相机 pose 获取有两条路线：

1. 直接从 MuJoCo camera state 查询 `cam_xpos / cam_xmat` 或 robosuite camera utils。
2. 对 wrist camera，用机器人末端 pose 推出：

```text
T_WC = T_WE * T_EC
```

其中：

- `T_WE`：end-effector pose in world，可由 `robot0_eef_pos` 和 `robot0_eef_quat` 构造。
- `T_EC`：eye-in-hand camera 相对 end-effector 的固定外参，可从模型 XML / MuJoCo site/camera
  pose 标定或初始化时求出。

优先级建议：

```text
先用 MuJoCo camera pose 直接读 T_WC
再实现 eef pose + fixed extrinsic fallback
```

### Planar Homography Fallback

如果暂时不启用 depth，可以先做一个平面假设版本：

```text
H = K * (R - t * n^T / d) * K^{-1}
```

其中：

- `R, t` 来自两帧相机相对运动。
- `n, d` 是假设工作台平面。

这可以比全局 2D 平移更强，能处理一定相机旋转和透视变化，但局限明显：

- 桌面上方的物体不满足同一平面。
- 拿起物体后误差会变大。
- 适合做中间 baseline，不适合作为最终 oracle。

### EEF Pose-Only Approximation

如果 camera pose 读取麻烦，可以先用末端 pose 近似 wrist camera pose：

```text
T_WC_t ≈ T_WE_t * T_EC
```

这条路线最适合快速把机械臂运动信息接进来。需要一次性估计或配置 `T_EC`。

风险：

- `robot0_eef_pos / quat` 的坐标系要和 camera/world 坐标系对齐。
- LIBERO observation 中的 eef quat 顺序和内部 MuJoCo quat 顺序需要确认。
- 如果 camera optical frame 和 eef frame 轴定义不一致，必须显式处理 axis convention。

### How Geometry Scheme Enters Cache

几何方案只替换 Step 1：

```text
stable_patches = geometry_compensated_correspondence(...)
```

后续保持：

```text
task_relevant_selection(prev_attn, image, stable_patches)
target_token_indices = target_patch_id + 1
source_token_indices = source_patch_id + 1
```

mask-only 模式仍只设置：

```python
vla.language_model.config.reusable_patches = target_token_indices
```

KV remap 模式则额外使用 `source_token_indices`。

### Evaluation Metrics

几何方案必须额外记录：

| 指标 | 含义 |
|---|---|
| `mean_reprojection_error_px` | warp 后像素/patch 对齐误差 |
| `valid_warp_ratio` | 有效投影比例 |
| `occlusion_ratio` | 被 z-buffer / depth gate 排除比例 |
| `pose_delta_translation` | 相机平移量 |
| `pose_delta_rotation_deg` | 相机旋转角 |
| `reuse_ratio` | 最终复用 token 比例 |
| `latency_ms` | 几何计算 + 模型总延迟 |
| `success_rate` | policy success，只在输入分布适配后作为主指标 |

### Pros / Cons

优点：

- 物理含义强，能明确解释 camera ego-motion。
- 在 LIBERO / MuJoCo 中可以拿 depth/pose，适合做 oracle upper bound。
- 能区分相机运动导致的像素位移和真正的场景变化。

缺点：

- 需要 depth、intrinsics、extrinsics，工程接入比 RGB heuristic 重。
- 深度边界、遮挡、动态物体会带来复杂 case。
- 真实数据上如果没有标定和深度，迁移难度较高。

## Scheme 2: Optical Flow Compensation

### Core Idea

用光流模型估计连续帧间的稠密运动场：

```text
flow_prev_to_curr[u, v] = (du, dv)
```

然后把像素级 flow 聚合到 patch level：

```text
current target patch -> previous source patch
```

这个方案不要求相机 pose / depth / calibration，更适合真实 wrist-camera 数据。但它更依赖视觉估计质量。

### Flow Direction

为了复用上一帧 cache，推荐估计两个方向：

```text
F_prev_to_curr
F_curr_to_prev
```

主路径使用 `F_curr_to_prev` 更直接：

```text
current pixel + F_curr_to_prev(current pixel) -> previous pixel
```

这样每个 current patch 内采样点可以直接投票到 previous source patch。

如果只计算 `F_prev_to_curr`，也可以 forward splat 到 current frame，但需要处理 holes 和 z-order；
因此实现上更推荐直接估计 current-to-previous flow，或者使用 forward-backward consistency。

### Candidate Flow Backends

按工程优先级：

| backend | 用途 | 优点 | 缺点 |
|---|---|---|---|
| OpenCV Farneback / TV-L1 | CPU smoke baseline | 无需大模型，接入快 | 对机器人场景和大位移可能不稳 |
| RAFT | 主力 RGB optical flow | 精度强，社区成熟 | 需要权重和 GPU，latency 较高 |
| GMFlow / UniMatch | 后续优化 | 对大位移和泛化较强 | 接入和依赖更复杂 |

权重应放在：

```text
/mnt/data0/zjh_data/Embodied_Proj/weights/optical_flow/<model_name>/
```

不要放入 repo。

### Patch Aggregation

对每个 current patch：

1. 在 patch 内采样 `N x N` 个点。
2. 对每个点 `(u_curr, v_curr)` 读取 flow：

```text
u_prev = u_curr + flow_curr_to_prev_x
v_prev = v_curr + flow_curr_to_prev_y
```

3. 如果 previous coordinate 在图像内，则转换为 previous patch id。
4. 对 previous patch id 投票。
5. 票数最多者作为 source patch。
6. 计算 confidence、flow variance、RGB similarity。

建议过滤条件：

```text
confidence >= min_confidence
flow_variance <= max_flow_variance
similarity >= sim_threshold
forward_backward_error <= fb_threshold
```

Patch-level score：

```text
score = confidence * similarity * exp(-flow_variance) * exp(-fb_error)
```

### Forward-Backward Consistency

为了减少遮挡和错误匹配，推荐同时估计：

```text
F_curr_to_prev
F_prev_to_curr
```

对 current pixel：

```text
p_prev = p_curr + F_curr_to_prev(p_curr)
p_roundtrip = p_prev + F_prev_to_curr(p_prev)
fb_error = ||p_roundtrip - p_curr||
```

如果 `fb_error` 太大，则该采样点不投票。

默认可先设：

```text
fb_threshold = 2.0 px
```

### Flow + Attention Fusion

光流只解决视觉对应关系，仍需要 task attention 过滤：

```text
flow-stable patches - task-relevant patches
```

最终复用 token：

```python
target_token_indices = target_patch_id + 1
source_token_indices = source_patch_id + 1
```

mask-only 先跑：

```text
--mc_enable_kv_remap False
```

确认稳定后再考虑 KV remap。

### Evaluation Metrics

光流方案建议记录：

| 指标 | 含义 |
|---|---|
| `mean_flow_magnitude` | 平均运动幅度 |
| `mean_flow_variance_per_patch` | patch 内 flow 一致性 |
| `fb_consistency_error` | forward-backward consistency |
| `valid_flow_ratio` | 有效 flow sample 比例 |
| `reuse_ratio` | 最终复用 token 比例 |
| `latency_ms_flow` | flow 估计耗时 |
| `latency_ms_total` | 端到端 action 耗时 |
| `cache_error_proxy` | 与 full recompute attention/action 的偏差 |

如果能离线跑 full recompute 对照，建议增加：

```text
action_l2_delta
action_cosine
attention_js_divergence
kv_reuse_rel_l2
```

### Pros / Cons

优点：

- 不依赖相机标定、深度和机器人模型。
- 更容易迁移到 DROID/RH20T/真实 wrist camera。
- 能处理一定非刚体、物体运动和局部运动。

缺点：

- 光流模型本身有误差，尤其是遮挡、反光、低纹理区域。
- GPU flow model latency 可能吃掉 cache 加速收益。
- 对 policy 输入分布没有直接帮助，只是 cache selection 更合理。

## Recommended Experiment Roadmap

### Phase 0: Keep Current Baseline

保留当前三组 baseline：

```text
full recompute
original same-grid VLA-Cache
current RGB global-translation MC
```

这能证明新方案不是只和一个弱 baseline 比。

### Phase 1: Geometry Oracle in LIBERO

优先实现：

```text
pose_depth
```

最小实验：

```text
LIBERO-Spatial
camera = robot0_eye_in_hand
tasks = 3
trials = 1
methods = original_grid, rgb_translation, pose_depth
```

目标：

- 验证几何 warp 后 candidate patch 更稳定。
- 观察相机旋转和平移较大时 reuse 是否比 same-grid 更合理。
- 记录 valid warp ratio 和 reprojection diagnostics。

### Phase 2: Optical Flow Baseline

先接 OpenCV flow 做 pipeline smoke：

```text
method = optical_flow_cv
```

然后接 RAFT / GMFlow：

```text
method = optical_flow_raft
```

目标：

- 在没有 depth/pose 的条件下，是否能接近 pose-depth oracle。
- 评估 flow latency 是否过高。
- 判断是否需要低分辨率 flow、patch center flow 或缓存 flow features。

### Phase 3: Hybrid Gate

最终可以融合两类信号：

```text
geometry correspondence if depth/pose is available
flow correspondence otherwise
RGB/feature similarity as final gate
task attention excludes important tokens
```

Hybrid score:

```text
score =
  w_pose * pose_confidence
  + w_flow * flow_confidence
  + w_rgb  * rgb_similarity
  - w_task * task_attention_score
```

在 LIBERO 中可以用 pose-depth 作为 teacher，对 flow-only 方案做 calibration。

## Implementation Plan

### New Files

建议新增：

```text
src/openvla/experiments/robot/camera_geometry.py
src/openvla/experiments/robot/flow_compensation.py
```

`camera_geometry.py`：

- camera intrinsics extraction
- camera pose extraction
- depth backprojection
- SE(3) warp
- z-buffer patch voting

`flow_compensation.py`：

- flow backend wrapper
- forward-backward consistency
- patch voting from flow
- flow diagnostics

### Modify Existing Files

`libero_utils.py`：

- add `use_depth`
- optionally expose raw camera pose / intrinsics helper

`run_libero_eval.py`：

- add `--mc_method`
- add `--use_depth`
- log method-specific diagnostics
- optionally write per-step CSV

`openvla_utils.py`：

- dispatch by `cfg.mc_method`
- keep output as `PatchCorrespondence`

### Suggested CLI

Geometry oracle:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --camera_name robot0_eye_in_hand \
  --use_vla_cache True \
  --use_motion_compensated_cache True \
  --mc_method pose_depth \
  --use_depth True \
  --mc_enable_kv_remap False
```

Optical flow:

```bash
python experiments/robot/libero/run_libero_eval.py \
  --camera_name robot0_eye_in_hand \
  --use_vla_cache True \
  --use_motion_compensated_cache True \
  --mc_method optical_flow \
  --flow_backend raft \
  --mc_enable_kv_remap False
```

## Success Criteria

短期成功标准：

- 两个新方案都能跑完整 closed-loop rollout。
- 输出 per-step diagnostics，没有明显 NaN/空 correspondence。
- 在 wrist-camera 下，reuse ratio 不低于 current RGB translation。
- 与 full recompute 的 action delta / attention divergence 小于 original grid cache。

中期成功标准：

- pose-depth oracle 在强相机运动任务中明显优于 same-grid cache。
- optical flow 接近 pose-depth oracle 的 reuse/action fidelity。
- 端到端 latency 可控，flow/geometry 预处理不完全吃掉 cache 收益。

长期成功标准：

- 在真实 wrist-camera 数据上，flow-only 或 hybrid 方案仍有效。
- 如果 policy 经过 wrist-camera 适配，成功率不显著下降，同时 latency/reuse 有收益。

## Recommendation

优先级建议：

1. **先做 pose + depth geometry oracle**：它最能证明“相机运动补偿”这个核心假设。
2. **再做 optical flow**：它是通向真实数据和无标定场景的路线。
3. **最后做 hybrid**：用 pose-depth oracle 校准 flow-only 方案，形成可解释又可迁移的版本。

当前最应该避免的是只继续优化 RGB global translation。它可以保留为 baseline，但不应作为主方案。

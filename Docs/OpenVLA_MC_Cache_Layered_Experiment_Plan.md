# OpenVLA MC-Cache Layered Experiment Plan

Date: 2026-06-02

本文给出 motion-compensated VLA-Cache 的分层实验计划。核心判断是：

```text
默认 LIBERO agentview 只能验证 OpenVLA + cache pipeline 不被破坏；
真正验证 moving-camera motion compensation，需要 wrist camera、depth/pose oracle、
系统性仿真环境，以及最终真实 wrist-camera 数据。
```

因此实验不应只看 task success，而要分阶段回答不同问题。

## Research Questions

本实验计划要回答五个问题：

1. 现有 OpenVLA + LIBERO 路径是否能在加入 MC-cache 逻辑后保持可用？
2. 切到 `robot0_eye_in_hand` 后，moving-camera 下 cache reuse / latency / stability 是否可测？
3. 使用 depth/pose geometry oracle 后，patch correspondence 是否明显优于 same-grid 和 RGB global translation？
4. 在更可控的仿真环境中，wrist-camera + RGB-D + pose oracle 是否能系统性验证方法上限？
5. 在真实 wrist-camera 数据中，几何/光流补偿是否仍有泛化价值？

## Core Hypotheses

| ID | 假设 | 主要验证层 |
|---|---|---|
| H1 | MC-cache 不破坏原始 OpenVLA + LIBERO evaluation pipeline | Layer 1 |
| H2 | wrist-camera 下 same-grid cache 的 patch 对齐会变差 | Layer 2 |
| H3 | depth/pose geometry oracle 能恢复更正确的 patch correspondence | Layer 3 |
| H4 | 在系统性仿真中，pose-depth oracle 是 motion compensation 的上限基准 | Layer 4 |
| H5 | optical flow / hybrid compensation 能迁移到真实 wrist-camera 数据 | Layer 5 |

## Method Variants

每层尽量比较同一组方法：

| 方法 | 说明 | 用途 |
|---|---|---|
| `full_recompute` | 不复用 cache | fidelity upper baseline |
| `original_grid_cache` | 原始 VLA-Cache，同位置 patch 比较 | same-grid baseline |
| `rgb_translation_mc` | 当前 RGB-only global 2D translation | weak MC baseline |
| `pose_depth_oracle` | depth + camera pose 3D warp | geometry oracle |
| `optical_flow_mc` | 光流估计 curr-to-prev correspondence | deployable moving-camera baseline |
| `hybrid_mc` | pose/flow/RGB/task attention 融合 | 后期主方案 |

其中 `pose_depth_oracle` 是方法验证上限，不一定是最终真实部署方案。

## Shared Metrics

所有层都记录：

| 指标 | 含义 |
|---|---|
| `success_rate` | task 成功率；仅在输入分布匹配时作为主指标 |
| `episode_completed` | 是否完整跑完 rollout，无 runtime exception |
| `reuse_ratio` | `reused_visual_tokens / total_visual_tokens` |
| `avg_candidates` | 每步可复用候选 patch 数 |
| `action_latency_ms` | 端到端 action 延迟 |
| `cuda_latency_ms` | 模型侧 GPU 推理延迟 |
| `action_l2_delta` | 相对 full recompute 的 action 偏差 |
| `attention_js_divergence` | 相对 full recompute 的 attention 分布偏差 |
| `patch_correspondence_confidence` | patch correspondence 置信度 |

针对 geometry oracle 额外记录：

| 指标 | 含义 |
|---|---|
| `valid_warp_ratio` | depth/pose warp 后仍在图像内的点比例 |
| `occlusion_ratio` | 被 z-buffer / depth gate 排除的比例 |
| `reprojection_error_px` | 重投影误差 |
| `camera_translation_delta` | 相机平移量 |
| `camera_rotation_delta_deg` | 相机旋转量 |

针对 optical flow 额外记录：

| 指标 | 含义 |
|---|---|
| `mean_flow_magnitude` | 平均光流幅值 |
| `flow_variance_per_patch` | patch 内 flow 一致性 |
| `forward_backward_error` | 前后向一致性误差 |
| `valid_flow_ratio` | 有效 flow sample 比例 |

## Layer 1: LIBERO Agentview Pipeline Sanity

### Purpose

验证加入 MC-cache 开关、日志、latency 统计后，不破坏原始 OpenVLA + LIBERO 路径。

这里 **不验证 moving-camera 方法优势**，因为默认 `agentview` 是固定第三人称视角。

### Setup

```text
Environment: LIBERO-Spatial
Camera: agentview
Checkpoint: openvla/openvla-7b-finetuned-libero-spatial
Trials: first smoke 1 task x 1 trial, then 10 tasks x 1 trial
```

### Methods

```text
full_recompute
original_grid_cache
rgb_translation_mc
```

`pose_depth_oracle` 可以暂不跑，因为 fixed agentview 下没有强相机运动。

### Main Metrics

```text
episode_completed
success_rate
reuse_ratio
action_latency_ms
cuda_latency_ms
action_l2_delta vs full_recompute
```

### Success Criteria

```text
所有方法能完整跑完 rollout
MC-cache 不显著降低原始 OpenVLA success
action delta 不出现异常爆炸
```

### Interpretation

如果 Layer 1 失败，说明 pipeline 接入有问题，不能进入后续层。

如果 Layer 1 成功，只能说明“不破坏原路径”，不能证明 motion compensation 有效。

## Layer 2: LIBERO Wrist-Camera Cache Stability

### Purpose

切到 `robot0_eye_in_hand`，验证 moving-camera 下 cache reuse、latency、stability 是否可测。

这里 **不主看 success rate**，因为现成 LIBERO OpenVLA checkpoint 主要适配默认
`agentview`，直接换 wrist-camera 会有输入分布偏移。

### Setup

```text
Environment: LIBERO-Spatial
Camera: robot0_eye_in_hand
Checkpoint: openvla/openvla-7b-finetuned-libero-spatial
Trials: 3 tasks x 1 trial -> 10 tasks x 1 trial
```

### Methods

```text
full_recompute
original_grid_cache
rgb_translation_mc
```

### Main Metrics

```text
episode_completed
reuse_ratio
action_latency_ms
cuda_latency_ms
action_l2_delta vs full_recompute
attention_js_divergence vs full_recompute
estimated_shift_l1_px
```

### Success Criteria

```text
wrist-camera observation path 稳定
MC-cache path 无中途 runtime exception
能观测到 same-grid 和 MC 的 reuse/latency 差异
action delta 可量化，不出现 NaN 或 cache 崩溃
```

### Known Risk

```text
success_rate 可能下降，不能直接归因于 cache 方法；
更可能是 agentview checkpoint 在 wrist-camera 输入上 OOD。
```

### Current Evidence

已有 smoke：

```text
Experiments/openvla_libero_mc_cache/outputs/libero_spatial_wrist_camera_smoke_results.md
```

结果显示两个 wrist-camera run 都能完整跑完 episode，但 success 为 `0/1`。

## Layer 3: LIBERO Wrist-Camera + Depth/Pose Oracle

### Purpose

验证 geometry compensation 是否真的比 same-grid / RGB translation 更准确。

这一层是 **方法验证关键层**。它仍然可以使用现成 OpenVLA checkpoint，因为
depth/pose 只用于 cache compensation，不输入 OpenVLA policy。

### Setup

```text
Environment: LIBERO-Spatial
Camera: robot0_eye_in_hand
RGB: robot0_eye_in_hand_image
Depth: robot0_eye_in_hand_depth
Pose: MuJoCo camera pose or eef pose + camera extrinsic
Checkpoint: openvla/openvla-7b-finetuned-libero-spatial
Trials: 3 tasks x 1 trial -> 10 tasks x 1 trial
```

### Required Implementation

需要扩展 LIBERO env：

```python
camera_names = ["robot0_eye_in_hand"]
camera_depths = True
```

需要新增或封装：

```text
camera intrinsics extraction
camera pose extraction
depth backprojection
SE(3) warp
z-buffer or visibility gate
patch-level voting
```

### Methods

```text
full_recompute
original_grid_cache
rgb_translation_mc
pose_depth_oracle
optional: pose_planar_homography
```

### Main Metrics

```text
valid_warp_ratio
reprojection_error_px
occlusion_ratio
reuse_ratio
action_l2_delta vs full_recompute
attention_js_divergence vs full_recompute
action_latency_ms
```

### Success Criteria

```text
pose_depth_oracle 的 correspondence confidence 高于 rgb_translation_mc
pose_depth_oracle 相对 full_recompute 的 action_l2_delta 更小
pose_depth_oracle 的 attention divergence 更低
在强 wrist motion 步骤中，pose_depth_oracle 比 same-grid 更合理
```

### Interpretation

如果 Layer 3 成功，就能证明：

```text
moving-camera 下，正确的几何补偿确实能改善 cache reuse fidelity。
```

如果 Layer 3 不成功，需要检查：

```text
camera pose 坐标系
depth 单位
intrinsics
图像翻转和 camera frame convention
遮挡处理
```

## Layer 4: RLBench / ManiSkill Systematic Simulation

### Purpose

用更适合 wrist-camera + RGB-D + pose oracle 的仿真环境，系统性验证方法上限。

Layer 4 不强求现成 OpenVLA checkpoint。这里的主目标是方法验证，而不是 OpenVLA
closed-loop success。

### Why RLBench / ManiSkill

| 环境 | 价值 |
|---|---|
| RLBench | eye-in-hand RGB-D、segmentation、camera pose，任务丰富 |
| ManiSkill | GPU 仿真、wrist camera、RGB-D、机器人状态，可系统扫相机运动强度 |

### Setup

```text
Inputs: wrist RGB, depth, camera pose, robot state
Policy options:
  - no-policy replay / offline frame-pair analysis
  - lightweight BC / Diffusion Policy baseline
  - optional VLA if checkpoint or fine-tuning path exists
```

### Methods

```text
original_grid_cache
rgb_translation_mc
pose_depth_oracle
optical_flow_mc
hybrid_mc
```

### Main Experiments

1. **Offline correspondence benchmark**

```text
Sample frame pairs with known camera motion
Evaluate patch correspondence quality
No policy needed
```

2. **Cache fidelity benchmark**

```text
Use model forward pass
Compare cache reuse output against full recompute
Measure action/logit/attention deltas
```

3. **Closed-loop policy benchmark**

```text
Only after a suitable policy checkpoint exists
Measure success / latency / reuse
```

### Main Metrics

```text
patch correspondence accuracy
valid warp ratio
flow consistency error
reuse ratio
action/logit delta
latency
closed-loop success if policy is available
```

### Success Criteria

```text
pose_depth_oracle 在 controlled camera motion 下显著优于 same-grid
optical_flow_mc 接近 pose_depth_oracle 的 correspondence quality
hybrid_mc 在遮挡/物体运动场景中更稳
```

### Interpretation

Layer 4 是论文方法论最扎实的一层：它能系统回答“补偿是否真的有效”，而不是被
LIBERO OpenVLA checkpoint 的视角分布限制住。

## Layer 5: DROID / RH20T Real Wrist-Camera Generalization

### Purpose

验证方法是否能迁移到真实 wrist-camera 数据。

这里优先验证 optical flow / hybrid，因为真实数据不一定有完美 depth/pose。

### Datasets

| 数据集 | 价值 |
|---|---|
| DROID | 大规模真实机器人，wrist-mounted camera，适合 VLA/flow 泛化 |
| RH20T | 多 RGB-D、in-hand camera、calibration，适合真实几何/深度验证 |

### Setup

```text
Inputs: wrist RGB sequence, optional depth/calibration/robot state
Policy:
  - offline cache fidelity first
  - VLA fine-tuning only after data format and action space aligned
```

### Methods

```text
original_grid_cache
rgb_translation_mc
optical_flow_mc
optional: calibrated geometry if depth/pose available
hybrid_mc
```

### Main Metrics

```text
flow consistency
patch reuse stability
action/logit delta vs full recompute
latency
if policy is fine-tuned: success / task progress
```

### Success Criteria

```text
optical_flow_mc 比 original_grid_cache 有更低 action/logit delta
reuse ratio 可控且不产生明显 cache corruption
latency overhead 不超过 cache savings
如果有 fine-tuned policy，success 不显著下降
```

### Interpretation

Layer 5 是最终外部有效性验证。没有它，方法可能只是仿真 oracle；有它，才能说明
真实 wrist-camera 场景下也有价值。

## Experiment Order

推荐执行顺序：

```text
1. Layer 1: agentview sanity
2. Layer 2: wrist-camera stability
3. Layer 3: LIBERO depth/pose oracle
4. Layer 4: RLBench / ManiSkill systematic sim
5. Layer 5: DROID / RH20T real data
```

不要跳过 Layer 1 和 Layer 2，因为它们能及时发现 OpenVLA evaluation、camera
observation、cache logging 的工程问题。

不要把 Layer 2 的 success rate 当主要结论，因为现成 OpenVLA LIBERO checkpoint
不是 wrist-camera checkpoint。

## Milestones

### Milestone A: Pipeline Baseline

输出：

```text
agentview sanity report
wrist-camera smoke report
full_recompute vs original_grid_cache vs rgb_translation_mc
```

通过标准：

```text
所有 run 完整结束
日志包含 success/reuse/latency/action_delta
```

### Milestone B: Geometry Oracle

输出：

```text
LIBERO robot0_eye_in_hand depth/pose oracle report
per-step warp diagnostics CSV
patch correspondence visualization
```

通过标准：

```text
pose_depth_oracle 在 correspondence/action fidelity 上优于 same-grid
```

### Milestone C: Optical Flow

输出：

```text
optical_flow_mc report
flow diagnostics CSV
flow vs pose-depth oracle comparison
```

通过标准：

```text
flow 方案接近 pose-depth oracle，且无需 simulator pose/depth
```

### Milestone D: Systematic Simulation

输出：

```text
RLBench or ManiSkill controlled camera-motion benchmark
ablation over camera motion magnitude
```

通过标准：

```text
方法收益随 camera motion 增大而体现
```

### Milestone E: Real Data

输出：

```text
DROID/RH20T wrist-camera offline cache-fidelity benchmark
optional fine-tuned policy evaluation
```

通过标准：

```text
真实数据上 optical_flow/hybrid 比 same-grid 更稳
```

## Data And Output Layout

遵守项目规则，数据和权重放在 `/mnt/data0/zjh_data`：

```text
/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_depth/
/mnt/data0/zjh_data/Embodied_Proj/datasets/rlbench/
/mnt/data0/zjh_data/Embodied_Proj/datasets/maniskill/
/mnt/data0/zjh_data/Embodied_Proj/datasets/droid/
/mnt/data0/zjh_data/Embodied_Proj/datasets/rh20t/

/mnt/data0/zjh_data/Embodied_Proj/weights/optical_flow/raft/
/mnt/data0/zjh_data/Embodied_Proj/weights/optical_flow/gmflow/
```

repo 内只放轻量结果和报告：

```text
vla-cache/Experiments/openvla_libero_mc_cache/outputs/
vla-cache/Docs/
```

## Reporting Template

每层实验报告都应包含：

```text
Setup
Methods
Command
Metrics table
Per-step diagnostics summary
Failure cases
Interpretation
Next action
```

最小 metrics table：

| method | camera | success | episode completed | reuse ratio | action delta | attention divergence | latency mean | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|

## Key Risks

| 风险 | 影响 | 缓解 |
|---|---|---|
| OpenVLA checkpoint 只适配 agentview | wrist-camera success 下降 | Layer 2 不主看 success；后续 fine-tune |
| depth/pose 坐标系错误 | geometry oracle 结论失效 | 先做 reprojection visualization |
| optical flow latency 太高 | cache 加速收益被抵消 | 先离线评估，再低分辨率/patch-level 优化 |
| dynamic objects / occlusion | correspondence 错误 | z-buffer、flow consistency、task attention gate |
| real data 缺标定 | geometry 方法难迁移 | 用 flow/hybrid 作为真实数据主线 |

## Final Recommendation

短期最该做：

```text
Layer 1 + Layer 2: 补齐 full recompute/action delta 对照
Layer 3: 实现 LIBERO wrist depth/pose oracle
```

中期主线：

```text
Layer 4: 用 RLBench/ManiSkill 做 controlled camera-motion benchmark
```

长期验证：

```text
Layer 5: 用 DROID/RH20T 做真实 wrist-camera 泛化
```

论文表达上，应避免声称默认 LIBERO agentview 能验证 moving-camera compensation。更准确的叙述是：

```text
LIBERO agentview verifies pipeline compatibility.
LIBERO wrist-camera + depth/pose verifies geometry compensation.
RLBench/ManiSkill provide systematic simulator validation.
DROID/RH20T test real wrist-camera generalization.
```

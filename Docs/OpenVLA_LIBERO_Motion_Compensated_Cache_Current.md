# OpenVLA + LIBERO Motion-Compensated Cache Current Implementation

本文记录当前仓库中 motion-compensated VLA-Cache 是如何实现和验证的。这里的
motion compensation 指的是 **RGB-only global 2D translation compensation**：
使用相邻两帧 RGB 图像估计一个全局平移，再把当前帧 ViT patch 映射到上一帧
patch，用于决定哪些视觉 token 可以复用 cache。

它目前不是 3D depth/pose oracle，也没有做逐物体或稠密光流对齐。

## Code Entry Points

主要涉及四个文件：

| 文件 | 作用 |
|---|---|
| `src/openvla/experiments/robot/libero/libero_utils.py` | 创建 LIBERO env、读取指定相机图像 |
| `src/openvla/experiments/robot/libero/run_libero_eval.py` | LIBERO rollout 入口，传入 `camera_name`，统计 reuse/latency |
| `src/openvla/experiments/robot/openvla_utils.py` | OpenVLA 推理入口，决定是否启用 MC cache |
| `src/openvla/experiments/robot/motion_compensation.py` | 当前 MC patch correspondence 的核心实现 |

相关 smoke 结果：

```text
Experiments/openvla_libero_mc_cache/outputs/libero_spatial_wrist_camera_smoke_results.md
```

## Rollout Data Flow

当前 closed-loop evaluation 的数据流如下：

```text
LIBERO OffScreenRenderEnv
  -> obs["<camera_name>_image"]
  -> get_libero_image(...)
  -> observation["full_image"], observation["prev_image"]
  -> get_vla_action(...)
  -> find_motion_compensated_static_patches(...)
  -> task_relevant_selection(...)
  -> vla.language_model.config.reusable_patches
  -> vla.predict_action(..., past_key_values=prompt_cache)
```

### 1. 支持移动相机输入

`run_libero_eval.py` 新增参数：

```text
--camera_name robot0_eye_in_hand
```

`libero_utils.get_libero_env(...)` 会把它传给 LIBERO:

```python
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": resolution,
    "camera_widths": resolution,
    "camera_names": [camera_name],
}
```

`libero_utils.get_libero_image(...)` 会读取：

```python
image_key = f"{camera_name}_image"
img = obs[image_key]
```

所以当 `camera_name=robot0_eye_in_hand` 时，实际进入 OpenVLA 的是：

```text
obs["robot0_eye_in_hand_image"]
```

图像之后仍沿用 OpenVLA/LIBERO 原有预处理：180 度翻转、resize 到 OpenVLA
需要的输入尺寸。

### 2. 保存当前帧和上一帧

每个控制步中，`run_libero_eval.py` 构造：

```python
observation = {
    "full_image": img,
    "prev_image": prev_img,
    "state": np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ),
}
```

OpenVLA 当前实际使用的是图像和语言指令；`state` 保留在 observation 里，但
OpenVLA 推理路径没有把 proprio state 作为模型输入。

第一步没有历史帧时，`prev_image = full_image`。从第二步开始，`prev_image`
使用上一控制步保存的图像。

## Motion Compensation Algorithm

核心函数：

```python
find_motion_compensated_static_patches(
    curr_image,
    prev_image,
    patch_size=14,
    top_k=130,
    search_radius=28,
    search_step=2,
    min_confidence=0.30,
    sim_threshold=0.70,
    samples_per_axis=5,
)
```

OpenVLA 图像尺寸是 `224 x 224`，patch size 是 `14`，所以视觉 token 对应：

```text
16 x 16 = 256 patches
```

### Step A: 估计全局 2D 平移

`estimate_global_translation(prev_image, curr_image, ...)` 做的是穷举搜索：

1. 把两帧 RGB 转成灰度。
2. 在 `[-search_radius, +search_radius]` 范围内按 `search_step` 枚举 `(dx, dy)`。
3. 对每个 shift，取上一帧和当前帧的重叠区域。
4. 用 mean squared error 作为匹配分数。
5. 选择 MSE 最低的 `(dx, dy)`。

默认参数：

```text
search_radius = 28 px
search_step   = 2 px
```

这里的 `(dx, dy)` 表示 **prev -> current** 的整数平移。

### Step B: 当前 patch 映射到上一帧 patch

对当前帧每个 patch，代码会在 patch 内均匀采样 `5 x 5` 个点：

```text
samples_per_axis = 5
```

对每个当前点 `(curr_x, curr_y)`，根据估计到的平移回投到上一帧：

```text
prev_x = curr_x - dx
prev_y = curr_y - dy
```

然后看这些采样点落入上一帧的哪个 patch。获得票数最多的 patch 作为
`source_patch`，置信度定义为：

```text
confidence = winning_votes / 25
```

如果没有合法 source patch，或者：

```text
confidence < min_confidence
```

则当前 patch 不参与复用候选。

默认：

```text
min_confidence = 0.30
```

### Step C: RGB patch 相似度过滤

对通过几何投票的 patch pair，计算当前 patch RGB 向量和上一帧 source patch RGB
向量的 cosine similarity：

```python
similarity = cosine(curr_patch_vector, prev_source_patch_vector)
```

如果：

```text
similarity < sim_threshold
```

则过滤掉。

默认：

```text
sim_threshold = 0.70
```

这个阈值比原始 same-grid VLA-Cache 的 `0.996` 低很多，因为 motion-compensated
patch 可能跨越旧 patch 边界，即使真实世界内容静止，patch 向量也不会完全一致。

### Step D: 选 top-k motion-compensated candidate patches

对剩余候选计算：

```text
combined_score = confidence * max(similarity, 0.0)
```

然后按 `combined_score` 降序排序，最多保留：

```text
top_k = 130
```

返回的结构是 `PatchCorrespondence`：

```python
target_patches      # 当前帧 patch ids
source_patches      # 对应上一帧 patch ids
confidence          # patch vote confidence
similarity          # RGB cosine similarity
shift_xy            # estimated (dx, dy)
score               # global translation MSE score
```

## How It Enters VLA-Cache

`openvla_utils.get_vla_action(...)` 中，只有满足以下条件时才会做复用选择：

```text
cfg.use_vla_cache == True
prompt_cache is not None
prev_attn is not None
```

第一步没有 `prompt_cache` 和 `prev_attn`，所以第一步是 full computation。之后每步：

1. 如果 `use_motion_compensated_cache=True`，调用
   `find_motion_compensated_static_patches(...)`。
2. 如果 `use_motion_compensated_cache=False`，走原始 same-grid
   `find_static_patches(...)`。
3. 得到 stable/significant patches 后，继续调用
   `task_relevant_selection(prev_attn, image, stable_patches, top_k=mc_task_top_k)`。

`task_relevant_selection(...)` 会从上一轮 attention 中估计 text token 对 vision
token 的关注度，取 top attention patches 作为任务相关区域。最终可复用的是：

```text
motion/static significant patches - top task-relevant patches
```

也就是说，高相似但被语言/任务强关注的视觉 patch 不复用，尽量重新计算。

默认：

```text
mc_task_top_k = 120
```

## Token Index Conversion

ViT patch id 是 `0..255`。进入语言模型 cache 的视觉 token index 从 `1` 开始：

```text
token_idx = patch_id + 1
```

MC 模式下，`task_relevant_selection(...)` 返回的是当前帧可复用 token index。
代码会把它转换回当前 patch id，再查对应上一帧 source patch：

```python
patch_id = int(token_idx) - 1
source_token_idx = source_by_target[patch_id] + 1
```

当前 mask-only smoke 中，真正写入模型配置的是：

```python
vla.language_model.config.reusable_patches = target_token_indices
vla.language_model.config.proportion_attn_var = get_layer_mask_schedule(prev_attn)
```

`reusable_patches` 告诉 modified transformers / VLA-Cache 路径哪些视觉 token 可复用。
`proportion_attn_var` 是按层的 reuse schedule，由上一轮 attention entropy 计算。

## KV Remap Branch

代码里有一个实验性分支：

```text
--mc_enable_kv_remap True
```

如果启用，会调用：

```python
remap_visual_kv_cache(prompt_cache, target_token_indices, source_token_indices)
```

它会尝试把上一帧 source token 的 KV cache copy 到当前帧 target token 位置。

但当前 wrist-camera smoke 使用的是：

```text
--mc_enable_kv_remap False
```

也就是 **mask-only MC cache**。原因是 KV remap 仍是 best-effort：

- 依赖 HuggingFace `DynamicCache` 暴露 `key_cache` 和 `value_cache`。
- 当前没有做 RoPE key correction。
- Python-side `index_copy_` 会增加额外开销。

所以目前主要验证的是：motion-compensated token selection 是否能稳定接入闭环
rollout，以及 reuse/latency 是否有可观测变化。

## Logged Metrics

`run_libero_eval.py` 对每个 episode 统计：

| 指标 | 含义 |
|---|---|
| `episode_mc_steps` | 有 cache metrics 的控制步数量 |
| `avg_reuse_ratio` | `reuse_tokens / (steps * 256)` |
| `avg_candidates` | 每步 MC/static 候选 patch 数 |
| `kv_remap_steps` | 应用了 KV remap 的步数 |
| `avg_shift_l1_px` | `abs(dx) + abs(dy)` 的 episode 平均 |
| `avg_shift_score` | 全局平移 MSE score 的 episode 平均 |
| `Action latency ms` | 包含图像处理、MC search、模型推理等的端到端 action 延迟 |

另外，模型内部也会在控制台打印 CUDA latency 和 TFLOPs。这个值更接近模型侧
GPU 推理时间，不包含所有 Python-side preprocessing。

## Smoke Command

当前 wrist-camera MC smoke 使用的命令形态是：

```bash
cd /home/zjh/Project/Embodied_Proj/vla-cache/src/openvla

env PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  MPLCONFIGDIR=/tmp/matplotlib \
  MUJOCO_GL=egl \
  CUDA_VISIBLE_DEVICES=0 \
  /mnt/data0/zjh_data/Embodied_Proj/envs/openvla/bin/python \
  experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --num_tasks_to_eval 1 \
  --camera_name robot0_eye_in_hand \
  --use_vla_cache True \
  --use_motion_compensated_cache True \
  --mc_enable_kv_remap False \
  --mc_search_radius 28 \
  --mc_search_step 2 \
  --mc_min_confidence 0.30 \
  --mc_similarity_threshold 0.70 \
  --mc_top_k 130 \
  --mc_task_top_k 120 \
  --run_id_note wrist-mc-cache-smoke \
  --use_wandb False
```

## Current Smoke Result

`libero_spatial`, first task, `robot0_eye_in_hand`, `1 trial`:

| method | success | avg reuse ratio | avg candidates | avg shift L1 px | action latency mean | final console avg CUDA latency |
|---|---:|---:|---:|---:|---:|---:|
| Original VLA-Cache | 0/1 | 0.369 | 130.0 | 0.00 | 410.0 ms | ~98.1 ms |
| MC mask-only | 0/1 | 0.391 | 130.0 | 0.70 | 453.3 ms | ~96.5 ms |

Interpretation:

- 两个 run 都完整跑完 closed-loop episode，没有 caught runtime exception。
- 成功率都是 0/1，主要因为 checkpoint 是按默认 `agentview` 分布微调，直接换
  `robot0_eye_in_hand` 有明显输入分布偏移。
- MC mask-only 的 reuse ratio 略高于原始 same-grid cache。
- 模型侧 CUDA latency 基本相当；端到端 action latency 增加，主要来自当前
  Python-side motion compensation search。
- 这个 first-task wrist motion 较弱，平均估计 shift 很小。后续需要扩大到更多
  LIBERO tasks，或者选 wrist motion 更强的场景。

## Current Limitations

1. 只估计全局 2D 平移，不能处理明显旋转、缩放、视差和非刚体/物体运动。
2. 没有使用 depth、相机 pose、机器人末端位姿或 segmentation。
3. RGB patch cosine 是粗粒度过滤；patch 边界跨越会降低 similarity。
4. 当前 MC search 是 Python/NumPy 实现，端到端 latency 不占优。
5. KV remap 还没有 RoPE correction，因此主实验目前使用 mask-only。
6. OpenVLA checkpoint 不是 wrist-camera fine-tuned，success rate 不应作为当前
   smoke 的主要结论。

## Next Improvements

推荐按以下顺序推进：

1. 跑 `libero_spatial` 多 task smoke，确认不同 wrist motion 强度下的 reuse/latency。
2. 保存每步 `(dx, dy, score, reuse_tokens)` 到 CSV，方便看时序稳定性。
3. 用 robot end-effector motion 或 simulator camera pose 做 oracle 2D/3D 对齐。
4. 把 global translation 替换为 optical flow 或 depth/pose warp。
5. 优化 MC search 到 torch/CUDA 或预计算 patch correspondence，降低 Python latency。
6. 如果要评估 success rate，需要 wrist-camera 分布的 policy adaptation 或微调。

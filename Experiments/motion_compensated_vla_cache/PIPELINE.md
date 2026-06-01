# Pipeline

## Goal

验证 Motion-Compensated VLA-Cache 的 MVP：

```text
RGB-D synthetic rollout
  -> full recompute teacher
  -> original same-grid reuse baseline
  -> oracle 3D patch correspondence
  -> MC-token / MC-KV-no-RoPE / MC-KV-RoPE
  -> action and KV deviation report
```

## Step 1: Synthetic RGB-D Rollout

脚本渲染几个简单 3D manipulation-like 场景：

- `static_cam`: 相机不动。
- `pan_tilt`: 相机有 yaw/pitch。
- `translate_xy`: 相机平移造成整幅画面在 grid 上移动。
- `wrist_like`: 模拟 wrist camera 的近距离视角变化。
- `dynamic_object`: 物体相对世界移动，用于测试 dynamic veto。

每帧包含：

- RGB image。
- depth。
- segmentation。
- camera pose。

## Step 2: Oracle 3D Patch Correspondence

对当前帧 patch 中的采样点做：

```text
current pixel + depth
  -> backproject to current camera
  -> transform to world
  -> transform to previous camera
  -> project to previous image
  -> depth consistency check
  -> vote source patch i for current patch j
```

输出 `prev_patch_i -> current_patch_j` 对应关系和置信度。

## Step 3: Reuse Candidates

- `original_grid`: 比较当前 patch `j` 和上一帧 patch `j` 的 RGB cosine similarity。
- `mc_*`: 比较当前 patch `j` 和 3D correspondence 给出的上一帧 patch `i`。
- attention/task veto 使用 action proxy 的 task attention top-k，模拟 VLA-Cache 中的 task-relevant exclusion。

## Step 4: Proxy Token/KV/Action

脚本构造轻量 teacher：

- patch token 由 RGB/depth/segmentation 特征投影得到。
- key 使用 RoPE-like 2D position rotation。
- value 不加位置旋转。
- action proxy 用 task query attend 到最后一层 KV，再线性读出 action。

方法对比：

- `full_recompute`: 当前帧全量 token/KV/action。
- `original_grid`: 复用上一帧相同 grid slot 的 KV。
- `mc_token`: 复用上一帧 correspondence token，再在当前位置重建 KV。
- `mc_kv_no_rope`: 直接把上一帧 source KV 搬到当前 target slot。
- `mc_kv_rope`: 先反旋到 source base key，再旋到 target position。

## Step 5: Metrics

- correspondence precision/recall proxy。
- reuse ratio。
- false reuse rate。
- action relative L2。
- reused key cosine。
- estimated saving fraction。

这个目录只做 MVP 验证，后续接真实 OpenVLA 时应把 `oracle_3d_patch_correspondence` 和 `remap_visual_kv_cache` 拆到 `experiments/robot/` 路径。

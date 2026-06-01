# OpenVLA + LIBERO-Spatial Motion-Compensated VLA-Cache

这个目录把 Motion-Compensated VLA-Cache 迁移到 `src/openvla` 的 LIBERO-Spatial 评测路径。

当前实现是 **OpenVLA 可运行的第一版迁移**：

- 保留原始 VLA-Cache 默认行为。
- 新增 `--use_motion_compensated_cache True`，用 RGB-only 2D 全局平移补偿替代 same-grid 静态 patch 检测。
- 新增可选 `--mc_enable_kv_remap True`，当运行时 `DynamicCache` 暴露 `key_cache/value_cache` 时，做 best-effort visual KV slot remap。
- 当前 LIBERO eval 路径没有传 depth/camera pose，因此 3D oracle 版本暂时预留接口，尚未默认启用。

## Files

```text
Experiments/openvla_libero_mc_cache/
  scripts/
    smoke_motion_compensation.py
    run_libero_spatial_original_cache.sh
    run_libero_spatial_mc_cache.sh
  outputs/
    smoke_motion/
```

相关代码改动：

```text
src/openvla/experiments/robot/motion_compensation.py
src/openvla/experiments/robot/openvla_utils.py
src/openvla/experiments/robot/libero/run_libero_eval.py
```

## CPU Smoke Test

不加载 OpenVLA，不需要 LIBERO/GPU：

```bash
python Experiments/openvla_libero_mc_cache/scripts/smoke_motion_compensation.py
```

输出：

```text
Experiments/openvla_libero_mc_cache/outputs/smoke_motion/
  prev.png
  curr_shifted.png
  mc_reuse_overlay.png
  smoke_report.md
```

## LIBERO-Spatial Smoke Runs

在 OpenVLA 环境中运行。由于要加载 OpenVLA-7B 和 GPU，按项目规则建议在沙箱外执行。

原始 VLA-Cache baseline：

```bash
CHECKPOINT=/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial \
TRIALS=1 TASKS=1 \
bash Experiments/openvla_libero_mc_cache/scripts/run_libero_spatial_original_cache.sh
```

Motion-compensated mask-only cache：

```bash
CHECKPOINT=/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial \
TRIALS=1 TASKS=1 \
bash Experiments/openvla_libero_mc_cache/scripts/run_libero_spatial_mc_cache.sh
```

Experimental KV remap：

```bash
CHECKPOINT=/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial \
TRIALS=1 TASKS=1 MC_KV_REMAP=True \
bash Experiments/openvla_libero_mc_cache/scripts/run_libero_spatial_mc_cache.sh
```

## Metrics

每个 episode 的 log 会新增：

```text
MC/VLA-Cache stats: steps=..., avg_reuse_ratio=..., avg_candidates=..., kv_remap_steps=...
```

含义：

- `avg_candidates`: motion compensation + RGB/confidence gate 之后、attention veto 之前的候选 patch 数。
- `avg_reuse_ratio`: attention veto 后实际交给 VLA-Cache pruning 的视觉 token 比例。
- `kv_remap_steps`: 成功执行 best-effort KV slot remap 的 step 数。

## Expected First Experiment

建议先跑最小矩阵：

| method | command | trials | tasks |
|---|---|---:|---:|
| Full recompute | `--use_vla_cache False` | 1 | 1 |
| Original VLA-Cache | `run_libero_spatial_original_cache.sh` | 1 | 1 |
| MC mask-only | `run_libero_spatial_mc_cache.sh` | 1 | 1 |
| MC KV remap | `MC_KV_REMAP=True run_libero_spatial_mc_cache.sh` | 1 | 1 |

如果 smoke 正常，再扩大到 `TASKS=10 TRIALS=5`。

## Current Limitation

这版是 RGB-only 2D motion compensation。它能验证 OpenVLA+LIBERO 接线和相机平移/转动下的候选恢复，但还不是最终 3D oracle 方案。下一步应从 LIBERO/MuJoCo renderer 取 agentview depth 和 camera pose，然后把 `motion_compensation.py` 的 translation estimator 替换成 3D patch projection。

## Current Smoke Results

已在 2026-05-31 运行两个 GPU smoke：

| method | task/trials | success | episode cache steps | avg reuse ratio | avg candidates | kv remap | final console avg CUDA latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original VLA-Cache | 1 task x 1 rollout | 1/1 | 79 | 0.319 | 130.0 | 0 | ~88.7 ms |
| MC mask-only | 1 task x 1 rollout | 1/1 | 89 | 0.303 | 130.0 | 0 | ~86.5 ms |
| MC KV-remap | 1 task x 1 rollout | 1/1 | 89 | 0.303 | 130.0 | 89 | ~99.7 ms |

日志文件：

```text
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_05_31-23_38_37--orig-grid-cache-smoke.txt
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_05_31-23_36_39--mc-cache-smoke.txt
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_05_31-23_42_13--mc-cache-kv-remap-smoke.txt
```

这只是接线 smoke，不代表统计显著结论。KV-remap 分支当前使用 Python 逐层 `index_copy_`，因此比 mask-only 慢，主要用于确认当前 cache 对象支持 visual slot remap。下一步需要在相同 initial states 上扩大到 `TASKS=10 TRIALS=5`，并加入 camera-motion perturbation 或 3D depth/pose oracle。

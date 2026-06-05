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
    run_kv_contextualization_study.py
    run_kv_contextualization_smoke.sh
    run_libero_spatial_semantic_static_matrix.sh
    summarize_libero_eval_matrix.py
    run_libero_spatial_original_cache.sh
    run_libero_spatial_mc_cache.sh
  outputs/
    smoke_motion/
    kv_study/
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

## Semantic-static Rollout Matrix

用于探究“像素几乎不动，但语义上重要”的场景下，VLA-Cache 是否影响成功率。默认 probe 是 LIBERO-Spatial task `0`：

```text
pick up the black bowl between the plate and the ramekin and place it on the plate
```

这个任务里 plate 和 ramekin 作为静态语义参照物，本身不会被操作，但它们决定 bowl 的初始空间关系，适合作为 first-pass semantic-static 成功率测试。

最小 smoke：

```bash
CHECKPOINT=/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial \
TRIALS=1 TASK_IDS=0 \
bash Experiments/openvla_libero_mc_cache/scripts/run_libero_spatial_semantic_static_matrix.sh
```

默认方法矩阵：

| method | meaning |
|---|---|
| `full_recompute` | `--use_vla_cache False`，全量重算 baseline |
| `original_cache` | 原始 VLA-Cache same-grid 静态 patch 复用 |

扩大到更像正式的 exact-task 成功率估计：

```bash
RUN_GROUP=semantic_static_bowl_between_e20 \
TRIALS=20 TASK_IDS=0 \
METHODS=full_recompute,original_cache \
bash Experiments/openvla_libero_mc_cache/scripts/run_libero_spatial_semantic_static_matrix.sh
```

加入相近空间关系对照任务：

```bash
RUN_GROUP=semantic_static_spatial_relation_contrast_e10 \
TRIALS=10 TASK_IDS=0,1,5,8 \
METHODS=full_recompute,original_cache \
bash Experiments/openvla_libero_mc_cache/scripts/run_libero_spatial_semantic_static_matrix.sh
```

这些 task 分别覆盖 `between plate and ramekin`、`next to ramekin`、`on ramekin`、`next to plate`。小规模 summary/logs 默认写到 repo 内：

```text
Experiments/openvla_libero_mc_cache/outputs/semantic_static/<run_group>/
  full_recompute_summary.json
  original_cache_summary.json
  matrix_summary.csv
  matrix_summary.md
  logs/
```

如需保存 rollout MP4：

```bash
SAVE_ROLLOUT_VIDEOS=True \
bash Experiments/openvla_libero_mc_cache/scripts/run_libero_spatial_semantic_static_matrix.sh
```

视频默认放在 `/mnt/data0/zjh_data/Embodied_Proj/experiments/vla_cache_semantic_static/<run_group>/rollouts/` 下，避免把大文件写进 repo。

## KV Contextualization Layer-wise Study

`Docs/OpenVLA_KV_Contextualization_Layerwise_Study_Design.md` 对应的新入口：

```bash
RUN_ID=kv_contextualization_model_smoke \
ROLLOUT_POLICY=model \
MAX_STEPS=1 SAMPLES=1 \
bash Experiments/openvla_libero_mc_cache/scripts/run_kv_contextualization_smoke.sh
```

默认输出：

```text
/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_kv_contextualization/<run_id>/layerwise_sim.csv
Experiments/openvla_libero_mc_cache/outputs/kv_study/<run_id>/
  kv_contextualization_study_report.md
  layerwise_kv_contextualization.png
```

已实现的 study 步骤：

- S0 self-frame sanity：同一帧重跑两次，验证 K/V/H 全层接近 1。
- S1 floor：同一帧内随机 patch pair，提供无对应下界。
- Grid A：`P0/P3 prompt x target/background patch_semantics`，逐层输出 K/V/H cosine。
- Grid B：`episode_phase x target/background patch_semantics`，复用同一套 oracle pair。
- 3D oracle pair：使用 LIBERO depth + robosuite camera matrices，把 current patch center 反投影到世界再投到 previous frame。

首个 GPU smoke（2026-06-03）：

| run | rollout | task | rows | S0 K/V/H | floor K mean | floor V mean | floor H mean | target pairs | background pairs |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `kv_contextualization_smoke_codex_20260603` | dummy | libero_spatial task 0 | 768 | 1.0 / 1.0 / 1.0 | 0.662 | 0.232 | 0.399 | 2 | 220 |
| `kv_contextualization_model_smoke_codex_20260603` | model | libero_spatial task 0 | 768 | 1.0 / 1.0 / 1.0 | 0.662 | 0.232 | 0.399 | 2 | 220 |
| `kv_contextualization_wrist_smoke_codex_20260603` | dummy | libero_spatial task 0, wrist camera | 768 | 1.0 / 1.0 / 1.0 | 0.656 | 0.223 | 0.365 | 9 | 182 |

这个 smoke 只证明 S0-S2 接线可用；target patch 数仍太少，不能用于结论。扩大采样时建议：

```bash
RUN_ID=kv_contextualization_spatial_agentview_phasefix_alltasks_e2 \
ROLLOUT_POLICY=model \
TASK_IDS=0-9 EPISODES=2 \
MAX_STEPS=160 SAMPLE_INTERVAL=10 SAMPLES=10 \
bash Experiments/openvla_libero_mc_cache/scripts/run_kv_contextualization_smoke.sh
```

phase 切分规则（2026-06-04 更新）：

- `reach`: eef-target 距离仍在阈值外。
- `grasp`: eef-target 距离首次进入阈值半径，及其后 `--phase-transition-window` 帧。
- `transport`: eef-target 距离在阈值内且夹爪未打开。
- `place`: episode 成功 `done` 后额外追加 terminal pair。
- `--phase-diagnostics-only True` 可只跑 rollout 并写 `phase_diagnostics.csv`，用于校准 phase 而不执行 K/V/H teacher forward。

已完成的 all-task agentview 结果（2026-06-04）：

| run | tasks | episodes/task | rows | S0 K/V/H | S1 floor mean | Grid B phase rows | zero target-pair rows |
|---|---:|---:|---:|---|---|---|---:|
| `kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604` | 10 | 2 | 116736 | 1.0 / 1.0 / 1.0 | K 0.669 / V 0.231 / H 0.402 | reach 16704 / grasp 2304 / transport 17856 / place 768 | A 8064 / B 4032 |

输出位置：

```text
/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_kv_contextualization/kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604/layerwise_sim.csv
Experiments/openvla_libero_mc_cache/outputs/kv_study/kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604/
  kv_contextualization_study_report.md
  layerwise_kv_contextualization.png
```

注意：`zero target-pair rows` 来自 agentview 下 target patch 不可见或 3D oracle depth consistency 过滤后没有有效对应。图和 summary 会跳过 NaN；最终统计应在分析脚本里显式过滤 `n_pairs == 0`。

扩大到完整 5-episode 版本时建议：

```bash
RUN_ID=kv_contextualization_spatial_agentview_phasefix_alltasks_e5 \
ROLLOUT_POLICY=model \
TASK_IDS=0-9 EPISODES=5 \
MAX_STEPS=220 SAMPLE_INTERVAL=10 SAMPLES=15 \
bash Experiments/openvla_libero_mc_cache/scripts/run_kv_contextualization_smoke.sh
```

再用 `CAMERA=robot0_eye_in_hand` 跑一份 wrist-camera 版本作为 motion magnitude 稳健性附录。

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

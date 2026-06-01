# OpenVLA + LIBERO-Spatial MC-VLA-Cache 迁移说明

## 已完成内容

本次迁移把 Motion-Compensated VLA-Cache 接入到 OpenVLA 的 LIBERO-Spatial 推理路径，保持原始 VLA-Cache 默认行为不变。

新增文件：

```text
src/openvla/experiments/robot/motion_compensation.py
Experiments/openvla_libero_mc_cache/
```

修改文件：

```text
src/openvla/experiments/robot/openvla_utils.py
src/openvla/experiments/robot/libero/run_libero_eval.py
```

## 方法路径

原始 VLA-Cache：

```text
prev RGB, curr RGB
  -> same-grid patch cosine
  -> attention veto
  -> reusable current token IDs
  -> LLM pruning / cache reuse
```

当前 MC-VLA-Cache 迁移版：

```text
prev RGB, curr RGB
  -> estimate global 2D prev->curr translation
  -> current patch j votes source patch i
  -> confidence + RGB similarity gate
  -> attention veto
  -> reusable current token IDs
  -> optional source i -> target j KV slot remap
  -> LLM pruning / cache reuse
```

## 新增参数

`run_libero_eval.py` 新增：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `use_motion_compensated_cache` | `False` | 是否启用 MC patch correspondence |
| `mc_enable_kv_remap` | `False` | 是否尝试 remap visual KV slots |
| `mc_patch_size` | `14` | OpenVLA ViT patch size |
| `mc_top_k` | `130` | motion compensation 后最多保留候选 patch 数 |
| `mc_task_top_k` | `120` | attention veto 的 top-k 任务相关 patch |
| `mc_search_radius` | `28` | 2D translation 搜索半径，单位 pixel |
| `mc_search_step` | `2` | translation 搜索步长 |
| `mc_min_confidence` | `0.30` | patch correspondence 采样投票置信度 |
| `mc_similarity_threshold` | `0.70` | MC source/target patch RGB similarity gate |
| `mc_samples_per_axis` | `5` | 每个 patch 的采样密度 |
| `num_tasks_to_eval` | `None` | smoke 时限制 task 数 |

## 已做验证

已运行 CPU smoke：

```bash
python Experiments/openvla_libero_mc_cache/scripts/smoke_motion_compensation.py
```

结果：

```text
Expected shift: (18, -12)
Estimated shift: (18, -12)
Selected patches: 130
Status: PASS
```

输出位于：

```text
Experiments/openvla_libero_mc_cache/outputs/smoke_motion/smoke_report.md
```

另外对新增/修改 Python 文件做了语法编译检查，并在 OpenVLA 环境中完成了 import/config smoke。

## OpenVLA + LIBERO-Spatial GPU Smoke

已在 2026-05-31 使用 `/mnt/data0/zjh_data/Embodied_Proj/envs/openvla` 和本地 checkpoint：

```text
/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial
```

运行两个 1 task x 1 rollout smoke：

| 方法 | 成功率 | episode cache steps | avg reuse ratio | avg candidates | kv remap steps | final console avg CUDA latency |
|---|---:|---:|---:|---:|---:|---:|
| Original VLA-Cache | 1/1 | 79 | 0.319 | 130.0 | 0 | ~88.7 ms |
| MC mask-only | 1/1 | 89 | 0.303 | 130.0 | 0 | ~86.5 ms |
| MC KV-remap | 1/1 | 89 | 0.303 | 130.0 | 89 | ~99.7 ms |

对应日志：

```text
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_05_31-23_38_37--orig-grid-cache-smoke.txt
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_05_31-23_36_39--mc-cache-smoke.txt
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_05_31-23_42_13--mc-cache-kv-remap-smoke.txt
```

观察：

- 新增 MC 分支能完整跑通 OpenVLA + LIBERO-Spatial closed-loop rollout。
- 同一 smoke task 上 MC mask-only 与原始 VLA-Cache 都成功。
- 当前 KV-remap 能执行，但 Python 逐层 remap 版本带来明显开销，且还没有 RoPE key correction，因此只作为 correctness smoke。
- 由于默认 LIBERO agentview camera 基本静止，MC 的平均 reuse ratio 与原始 VLA-Cache 接近；要验证本文核心假设，还需要加入相机扰动或 3D depth/pose oracle。

## 尚未完成的重型实验

当前只跑了 1 task x 1 rollout smoke，还没有跑统计规模实验。

推荐在 OpenVLA 环境中运行：

```bash
CHECKPOINT=/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial \
TRIALS=1 TASKS=1 \
bash Experiments/openvla_libero_mc_cache/scripts/run_libero_spatial_mc_cache.sh
```

如果要测试 best-effort KV remap：

```bash
CHECKPOINT=/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial \
TRIALS=1 TASKS=1 MC_KV_REMAP=True \
bash Experiments/openvla_libero_mc_cache/scripts/run_libero_spatial_mc_cache.sh
```

## 重要限制

这版是 RGB-only 2D motion compensation，不是最终 3D oracle 版本。它适合先验证 OpenVLA 接线、cache mask 行为和 2D 相机平移补偿；真正验证论文主张时，需要进一步从 LIBERO/MuJoCo 取 depth/camera pose，实现 3D patch projection。

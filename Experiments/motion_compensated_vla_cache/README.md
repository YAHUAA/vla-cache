# Motion-Compensated VLA-Cache MVP

这个独立实验目录用于验证 `Docs/Motion_Compensated_VLA_Cache_Experiment_Plan.md` 中的最小实验：

> 使用可控 RGB-D 合成相机运动，先验证 oracle 3D patch correspondence，再比较原始 grid-level cache、motion-compensated token reuse、motion-compensated KV reuse 和 RoPE 修正后的 KV reuse。

实验暂不加载 OpenVLA 大模型，而是用轻量 action proxy 模拟 VLA 的视觉 token、LLM KV 和动作输出。这样可以快速回答两个问题：

- 相机自运动时，3D motion compensation 是否能恢复跨帧 patch 对应关系。
- KV remap 是否比只按同一 grid 位置复用更接近 full recompute。

## Folder Layout

```text
Experiments/motion_compensated_vla_cache/
  configs/
    smoke.json
    default.json
  scripts/
    run_mc_vla_cache_mvp.py
  src/
    __init__.py
  outputs/
    smoke/
      metrics/
      figures/
      motion_compensated_vla_cache_report.md
  README.md
  PIPELINE.md
```

## Quick Run

```bash
python Experiments/motion_compensated_vla_cache/scripts/run_mc_vla_cache_mvp.py \
  --config Experiments/motion_compensated_vla_cache/configs/smoke.json
```

完整合成实验：

```bash
python Experiments/motion_compensated_vla_cache/scripts/run_mc_vla_cache_mvp.py \
  --config Experiments/motion_compensated_vla_cache/configs/default.json
```

## Outputs

- `metrics/pair_metrics.csv`: 每个场景、帧对、阈值和方法的细粒度指标。
- `metrics/summary_by_method.csv`: 按场景、阈值和方法聚合后的结果。
- `figures/debug_correspondence.png`: grid diff 与 3D MC correspondence 的可视化。
- `motion_compensated_vla_cache_report.md`: 自动生成的简要报告。

## How To Read Results

重点看三列：

- `reuse_ratio`: 被判定为可复用的 token 比例。
- `false_reuse_rate`: 复用 token 中几何上不应复用的比例。
- `action_rel_l2`: 相对 full recompute action proxy 的偏差。

预期现象：

- `pan_tilt` / `translate_xy` 下，`original_grid` 的可复用率或正确复用率会下降。
- `mc_kv_rope` 通常比 `mc_kv_no_rope` 有更高的 reused-key cosine 和更低 action 偏差。
- `mc_token` 主要验证 correspondence 质量，latency saving 只是估计上限，不代表真实 LLM KV 加速。

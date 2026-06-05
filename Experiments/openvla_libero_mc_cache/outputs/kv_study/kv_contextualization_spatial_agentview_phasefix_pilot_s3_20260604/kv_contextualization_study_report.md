# OpenVLA KV Contextualization Layer-wise Study

- Run ID: `kv_contextualization_spatial_agentview_phasefix_pilot_s3_20260604`
- Task suite: `libero_spatial`
- Task IDs: `0-2`
- Episodes per task: `2`
- Camera: `agentview`
- Rollout policy: `model`
- Full CSV: `/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_kv_contextualization/kv_contextualization_spatial_agentview_phasefix_pilot_s3_20260604/layerwise_sim.csv`
- Plot: `/home/zjh/Project/Embodied_Proj/vla-cache/Experiments/openvla_libero_mc_cache/outputs/kv_study/kv_contextualization_spatial_agentview_phasefix_pilot_s3_20260604/layerwise_kv_contextualization.png`

## Summary

```json
{
  "camera": "agentview",
  "center_crop": true,
  "csv_path": "/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_kv_contextualization/kv_contextualization_spatial_agentview_phasefix_pilot_s3_20260604/layerwise_sim.csv",
  "episodes": 2,
  "plot_path": "/home/zjh/Project/Embodied_Proj/vla-cache/Experiments/openvla_libero_mc_cache/outputs/kv_study/kv_contextualization_spatial_agentview_phasefix_pilot_s3_20260604/layerwise_kv_contextualization.png",
  "rollout_policy": "model",
  "run_id": "kv_contextualization_spatial_agentview_phasefix_pilot_s3_20260604",
  "summary": {
    "controls": {
      "S0_self/H": {
        "max": 1.0,
        "mean": 1.0,
        "min": 1.0
      },
      "S0_self/K": {
        "max": 1.0,
        "mean": 1.0,
        "min": 1.0
      },
      "S0_self/V": {
        "max": 1.0,
        "mean": 1.0,
        "min": 1.0
      },
      "S1_floor/H": {
        "max": 0.5108804106712341,
        "mean": 0.3955530446643631,
        "min": 0.25042325258255005
      },
      "S1_floor/K": {
        "max": 0.7436965703964233,
        "mean": 0.6635062911858162,
        "min": 0.5159628391265869
      },
      "S1_floor/V": {
        "max": 0.38003966212272644,
        "mean": 0.2256048982574915,
        "min": 0.12001873552799225
      }
    },
    "grid_pair_counts": {
      "A_prompt_patch/background": {
        "max": 221,
        "mean": 208.20833333333334,
        "min": 180
      },
      "A_prompt_patch/target": {
        "max": 4,
        "mean": 2.41025641025641,
        "min": 1
      },
      "B_phase_patch/background": {
        "max": 221,
        "mean": 208.20833333333334,
        "min": 180
      },
      "B_phase_patch/target": {
        "max": 4,
        "mean": 2.41025641025641,
        "min": 1
      }
    },
    "rows": 28800
  },
  "task_ids": "0-2",
  "task_suite": "libero_spatial"
}
```

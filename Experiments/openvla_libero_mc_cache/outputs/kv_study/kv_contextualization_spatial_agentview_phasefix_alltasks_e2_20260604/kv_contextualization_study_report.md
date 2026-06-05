# OpenVLA KV Contextualization Layer-wise Study

- Run ID: `kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604`
- Task suite: `libero_spatial`
- Task IDs: `0-9`
- Episodes per task: `2`
- Camera: `agentview`
- Rollout policy: `model`
- Full CSV: `/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_kv_contextualization/kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604/layerwise_sim.csv`
- Plot: `/home/zjh/Project/Embodied_Proj/vla-cache/Experiments/openvla_libero_mc_cache/outputs/kv_study/kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604/layerwise_kv_contextualization.png`

## Summary

```json
{
  "camera": "agentview",
  "center_crop": true,
  "csv_path": "/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_kv_contextualization/kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604/layerwise_sim.csv",
  "episodes": 2,
  "plot_path": "/home/zjh/Project/Embodied_Proj/vla-cache/Experiments/openvla_libero_mc_cache/outputs/kv_study/kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604/layerwise_kv_contextualization.png",
  "rollout_policy": "model",
  "run_id": "kv_contextualization_spatial_agentview_phasefix_alltasks_e2_20260604",
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
        "max": 0.5228983163833618,
        "mean": 0.4020378573331982,
        "min": 0.25042325258255005
      },
      "S1_floor/K": {
        "max": 0.7482996582984924,
        "mean": 0.6685543867759407,
        "min": 0.4985675811767578
      },
      "S1_floor/V": {
        "max": 0.41487234830856323,
        "mean": 0.23135932019213215,
        "min": 0.11563897877931595
      }
    },
    "grid_pair_counts": {
      "A_prompt_patch/background": {
        "max": 224,
        "mean": 201.10204081632654,
        "min": 176
      },
      "A_prompt_patch/target": {
        "max": 6,
        "mean": 2.6493506493506493,
        "min": 1
      },
      "B_phase_patch/background": {
        "max": 224,
        "mean": 201.10204081632654,
        "min": 176
      },
      "B_phase_patch/target": {
        "max": 6,
        "mean": 2.6493506493506493,
        "min": 1
      }
    },
    "phase_counts": {
      "grasp": 2304,
      "place": 768,
      "reach": 16704,
      "transport": 17856
    },
    "phase_reason_counts": {
      "eef_entered_target_radius": 960,
      "eef_far_from_target": 16704,
      "eef_near_target_and_gripper_closed": 17856,
      "env_done_after_action": 768,
      "within_grasp_transition_window_after_step_29": 192,
      "within_grasp_transition_window_after_step_37": 192,
      "within_grasp_transition_window_after_step_45": 192,
      "within_grasp_transition_window_after_step_46": 384,
      "within_grasp_transition_window_after_step_48": 192,
      "within_grasp_transition_window_after_step_65": 192
    },
    "rows": 116736,
    "zero_pair_rows": {
      "A_prompt_patch/target": 8064,
      "B_phase_patch/target": 4032
    }
  },
  "task_ids": "0-9",
  "task_suite": "libero_spatial"
}
```

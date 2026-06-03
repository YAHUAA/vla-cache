# LIBERO-Spatial Wrist-Camera MC-Cache Smoke Results

Date: 2026-06-01

Environment:

- Python env: `/mnt/data0/zjh_data/Embodied_Proj/envs/openvla`
- Checkpoint: `/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial`
- GPU: NVIDIA GeForce RTX 3090 24GB
- Task suite: `libero_spatial`
- Camera: `robot0_eye_in_hand`
- Smoke size: `num_tasks_to_eval=1`, `num_trials_per_task=1`

## Results

| method | run id note | success | cache steps | avg reuse ratio | avg candidates | avg shift L1 px | avg shift score | action latency mean / p50 / p90 | final console avg CUDA latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original VLA-Cache | `wrist-orig-grid-cache-smoke` | 0/1 | 219 | 0.369 | 130.0 | 0.00 | 0.000 | 410.0 / 406.4 / 409.9 ms | ~98.1 ms |
| MC mask-only | `wrist-mc-cache-smoke` | 0/1 | 219 | 0.391 | 130.0 | 0.70 | 0.002 | 453.3 / 434.4 / 500.9 ms | ~96.5 ms |

## Logs

```text
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_01-20_15_36--wrist-orig-grid-cache-smoke.txt
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_06_01-20_18_19--wrist-mc-cache-smoke.txt
```

## Interpretation

Both runs completed the full closed-loop episode without a caught runtime
exception, so the wrist-camera observation path and the MC-cache path are stable
for this minimal smoke. The post-exit EGL destructor warning appeared after each
completed run and did not interrupt evaluation.

The task success is 0/1 for both runs. This is expected for a first wrist-camera
probe because the checkpoint is fine-tuned for the default LIBERO view rather
than `robot0_eye_in_hand`.

The MC run reused slightly more visual tokens than same-grid reuse on this
moving-camera input (`0.391` vs `0.369`). End-to-end action latency increased
because the current MC correspondence search is Python-side preprocessing,
while model-side CUDA latency stayed comparable. The observed average
translation is small for this first task, so the next useful check is a larger
suite sample or a task with stronger wrist motion.

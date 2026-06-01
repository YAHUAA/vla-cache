# LIBERO-Spatial MC-Cache Smoke Results

Date: 2026-05-31

Environment:

- Python env: `/mnt/data0/zjh_data/Embodied_Proj/envs/openvla`
- Checkpoint: `/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial`
- GPU: NVIDIA GeForce RTX 3090 24GB
- Task suite: `libero_spatial`
- Smoke size: `num_tasks_to_eval=1`, `num_trials_per_task=1`

## Results

| method | run id note | success | cache steps | avg reuse ratio | avg candidates | kv remap steps | final console avg CUDA latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Original VLA-Cache | `orig-grid-cache-smoke` | 1/1 | 79 | 0.319 | 130.0 | 0 | ~88.7 ms |
| MC mask-only | `mc-cache-smoke` | 1/1 | 89 | 0.303 | 130.0 | 0 | ~86.5 ms |
| MC KV-remap | `mc-cache-kv-remap-smoke` | 1/1 | 89 | 0.303 | 130.0 | 89 | ~99.7 ms |

## Logs

```text
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_05_31-23_38_37--orig-grid-cache-smoke.txt
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_05_31-23_36_39--mc-cache-smoke.txt
src/openvla/experiments/logs/EVAL-libero_spatial-openvla-2026_05_31-23_42_13--mc-cache-kv-remap-smoke.txt
```

## Interpretation

This smoke confirms that the OpenVLA + LIBERO-Spatial inference path accepts
the new motion-compensated cache mode and can complete a closed-loop rollout.
It is not yet a statistical result: the default LIBERO agentview camera is
mostly static, so original same-grid reuse remains strong. The next experiment
should introduce camera perturbations or a depth/pose oracle to stress the
failure mode described in the proposal.

The KV-remap branch is currently a correctness smoke only. It performs Python
``index_copy_`` over every layer, so it is slower than mask-only reuse and does
not include RoPE key correction yet.

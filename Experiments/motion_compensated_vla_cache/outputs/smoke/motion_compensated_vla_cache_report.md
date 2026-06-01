# Motion-Compensated VLA-Cache MVP Report

- Run name: `smoke`
- Scenarios: `pan_tilt, translate_xy`
- Frames per scenario: `5`
- Image size: `112x112`
- Patch size: `14`
- Pair rows: `64`
- Runtime: `0.17s`

## Aggregate By Method

| method | reuse_ratio | false_reuse_rate | action_rel_l2 | reused_key_cosine | estimated_saving |
|---|---:|---:|---:|---:|---:|
| mc_kv_no_rope | 0.146 | 0.000 | 0.0046 | 0.590 | 0.122 |
| mc_kv_rope | 0.146 | 0.000 | 0.0046 | 0.997 | 0.122 |
| mc_token | 0.146 | 0.000 | 0.0046 | 0.997 | 0.015 |
| original_grid | 0.309 | 0.503 | 0.0037 | 1.000 | 0.257 |

## Notes

- `original_grid` reuses the previous KV at the same patch index, matching the failure mode of grid-level cache under camera ego-motion.
- `mc_token` remaps visual tokens but rebuilds KV at the current position; it is a correspondence-quality diagnostic rather than the main speed path.
- `mc_kv_no_rope` directly moves previous KV from source patch to target patch.
- `mc_kv_rope` additionally applies a RoPE-like key position correction from source slot to target slot.

Generated files:

- `metrics/pair_metrics.csv`
- `metrics/summary_by_method.csv`
- `figures/debug_correspondence.png`

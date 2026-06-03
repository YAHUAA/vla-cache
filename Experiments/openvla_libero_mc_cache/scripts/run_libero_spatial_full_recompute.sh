#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../src/openvla"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

CHECKPOINT="${CHECKPOINT:-/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial}"
TRIALS="${TRIALS:-1}"
TASKS="${TASKS:-1}"
RUN_ID_NOTE="${RUN_ID_NOTE:-full-recompute-smoke}"
CAMERA="${CAMERA:-agentview}"

# full_recompute baseline: VLA-Cache disabled, every step recomputes all visual tokens.
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_spatial \
  --camera_name "${CAMERA}" \
  --num_trials_per_task "${TRIALS}" \
  --num_tasks_to_eval "${TASKS}" \
  --use_vla_cache False \
  --run_id_note "${RUN_ID_NOTE}" \
  --use_wandb False

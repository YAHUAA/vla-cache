#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../src/openvla"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

CHECKPOINT="${CHECKPOINT:-/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial}"
TRIALS="${TRIALS:-1}"
TASKS="${TASKS:-1}"
RUN_ID_NOTE="${RUN_ID_NOTE:-mc-cache-smoke}"
MC_KV_REMAP="${MC_KV_REMAP:-False}"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name libero_spatial \
  --num_trials_per_task "${TRIALS}" \
  --num_tasks_to_eval "${TASKS}" \
  --use_vla_cache True \
  --use_motion_compensated_cache True \
  --mc_enable_kv_remap "${MC_KV_REMAP}" \
  --mc_search_radius 28 \
  --mc_search_step 2 \
  --mc_min_confidence 0.30 \
  --mc_similarity_threshold 0.70 \
  --mc_top_k 130 \
  --mc_task_top_k 120 \
  --run_id_note "${RUN_ID_NOTE}" \
  --use_wandb False

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/mnt/data0/zjh_data/Embodied_Proj/envs/openvla/bin/python}"
CHECKPOINT="${CHECKPOINT:-/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_IDS="${TASK_IDS:-0}"
EPISODES="${EPISODES:-1}"
CAMERA="${CAMERA:-agentview}"
ROLLOUT_POLICY="${ROLLOUT_POLICY:-dummy}"
MAX_STEPS="${MAX_STEPS:-1}"
SAMPLES="${SAMPLES:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"
CENTER_CROP="${CENTER_CROP:-True}"
RUN_CONTROLS="${RUN_CONTROLS:-True}"
INCLUDE_UNKNOWN_PHASE="${INCLUDE_UNKNOWN_PHASE:-False}"
PHASE_DIAGNOSTICS_ONLY="${PHASE_DIAGNOSTICS_ONLY:-False}"
PHASE_DIAGNOSTICS_PRINT_INTERVAL="${PHASE_DIAGNOSTICS_PRINT_INTERVAL:-10}"
PHASE_TRANSITION_WINDOW="${PHASE_TRANSITION_WINDOW:-5}"
SAMPLE_DONE_PHASE="${SAMPLE_DONE_PHASE:-True}"
RUN_ID="${RUN_ID:-kv_contextualization_smoke_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data0/zjh_data/Embodied_Proj/datasets/libero_kv_contextualization/${RUN_ID}}"
REPO_OUTPUT_DIR="${REPO_OUTPUT_DIR:-${REPO_ROOT}/Experiments/openvla_libero_mc_cache/outputs/kv_study/${RUN_ID}}"

cd "${REPO_ROOT}"

"${PYTHON}" Experiments/openvla_libero_mc_cache/scripts/run_kv_contextualization_study.py \
  --pretrained-checkpoint "${CHECKPOINT}" \
  --task-suite-name "${TASK_SUITE}" \
  --task-ids "${TASK_IDS}" \
  --episodes-per-task "${EPISODES}" \
  --camera-name "${CAMERA}" \
  --rollout-policy "${ROLLOUT_POLICY}" \
  --max-rollout-steps "${MAX_STEPS}" \
  --sample-interval "${SAMPLE_INTERVAL}" \
  --max-samples-per-episode "${SAMPLES}" \
  --center-crop "${CENTER_CROP}" \
  --run-controls "${RUN_CONTROLS}" \
  --include-unknown-phase "${INCLUDE_UNKNOWN_PHASE}" \
  --phase-diagnostics-only "${PHASE_DIAGNOSTICS_ONLY}" \
  --phase-diagnostics-print-interval "${PHASE_DIAGNOSTICS_PRINT_INTERVAL}" \
  --phase-transition-window "${PHASE_TRANSITION_WINDOW}" \
  --sample-done-phase "${SAMPLE_DONE_PHASE}" \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --repo-output-dir "${REPO_OUTPUT_DIR}"

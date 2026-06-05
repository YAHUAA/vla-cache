#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OPENVLA_ROOT="${REPO_ROOT}/src/openvla"
CUSTOM_TASK_DIR="${REPO_ROOT}/Experiments/openvla_libero_mc_cache/custom_tasks"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

CHECKPOINT="${CHECKPOINT:-/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial}"
CUSTOM_BDDL_FILE="${CUSTOM_BDDL_FILE:-${CUSTOM_TASK_DIR}/put_black_bowl_between_plate_and_ramekin.bddl}"
CUSTOM_INIT_STATES_FILE="${CUSTOM_INIT_STATES_FILE:-${CUSTOM_TASK_DIR}/init_states/put_black_bowl_between_plate_and_ramekin_from_next_to_ramekin.pruned_init}"
CUSTOM_TASK_NAME="${CUSTOM_TASK_NAME:-put_black_bowl_between_plate_and_ramekin}"
CUSTOM_MAX_STEPS="${CUSTOM_MAX_STEPS:-220}"
TRIALS="${TRIALS:-1}"
SEED="${SEED:-7}"
CAMERA="${CAMERA:-agentview}"
METHODS="${METHODS:-full_recompute,original_cache}"
SAVE_ROLLOUT_VIDEOS="${SAVE_ROLLOUT_VIDEOS:-True}"
RUN_GROUP="${RUN_GROUP:-custom_bowl_between_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_DIR="${SUMMARY_DIR:-${REPO_ROOT}/Experiments/openvla_libero_mc_cache/outputs/custom_bowl_between/${RUN_GROUP}}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SUMMARY_DIR}/logs}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${SUMMARY_DIR}/rollouts}"

mkdir -p "${SUMMARY_DIR}" "${LOCAL_LOG_DIR}"
if [[ "${SAVE_ROLLOUT_VIDEOS}" == "True" || "${SAVE_ROLLOUT_VIDEOS}" == "true" || "${SAVE_ROLLOUT_VIDEOS}" == "1" ]]; then
  mkdir -p "${ROLLOUT_DIR}"
fi

cd "${OPENVLA_ROOT}"

run_eval() {
  local method="$1"
  shift
  echo "[run] method=${method} custom_bddl=${CUSTOM_BDDL_FILE} trials=${TRIALS} camera=${CAMERA}"
  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --task_ids 0 \
    --custom_bddl_file "${CUSTOM_BDDL_FILE}" \
    --custom_init_states_file "${CUSTOM_INIT_STATES_FILE}" \
    --custom_task_name "${CUSTOM_TASK_NAME}" \
    --custom_max_steps "${CUSTOM_MAX_STEPS}" \
    --camera_name "${CAMERA}" \
    --num_trials_per_task "${TRIALS}" \
    --seed "${SEED}" \
    --run_id_note "${RUN_GROUP}-${method}" \
    --local_log_dir "${LOCAL_LOG_DIR}" \
    --save_rollout_videos "${SAVE_ROLLOUT_VIDEOS}" \
    --rollout_dir "${ROLLOUT_DIR}/${method}" \
    --summary_json_path "${SUMMARY_DIR}/${method}_summary.json" \
    --use_wandb False \
    "$@"
}

IFS=',' read -ra METHOD_LIST <<< "${METHODS}"
for raw_method in "${METHOD_LIST[@]}"; do
  method="${raw_method//[[:space:]]/}"
  case "${method}" in
    full_recompute)
      run_eval "${method}" \
        --use_vla_cache False
      ;;
    original_cache)
      run_eval "${method}" \
        --use_vla_cache True \
        --use_motion_compensated_cache False
      ;;
    *)
      echo "[error] unknown method: ${method}" >&2
      echo "[error] valid methods for current custom probe: full_recompute,original_cache" >&2
      exit 2
      ;;
  esac
done

python "${SCRIPT_DIR}/summarize_libero_eval_matrix.py" \
  --summary-dir "${SUMMARY_DIR}" \
  --output-md "${SUMMARY_DIR}/matrix_summary.md" \
  --output-csv "${SUMMARY_DIR}/matrix_summary.csv"

echo "[ok] custom bowl-between rollout matrix complete: ${SUMMARY_DIR}"

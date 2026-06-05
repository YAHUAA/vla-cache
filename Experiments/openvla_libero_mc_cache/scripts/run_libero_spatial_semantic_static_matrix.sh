#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OPENVLA_ROOT="${REPO_ROOT}/src/openvla"

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

CHECKPOINT="${CHECKPOINT:-/mnt/data0/zjh_data/Embodied_Proj/checkpoints/openvla-7b-finetuned-libero-spatial}"
TASK_IDS="${TASK_IDS:-0}"
TRIALS="${TRIALS:-5}"
SEED="${SEED:-7}"
CAMERA="${CAMERA:-agentview}"
METHODS="${METHODS:-full_recompute,original_cache}"
SAVE_ROLLOUT_VIDEOS="${SAVE_ROLLOUT_VIDEOS:-False}"
RUN_GROUP="${RUN_GROUP:-semantic_static_bowl_between_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_DIR="${SUMMARY_DIR:-${REPO_ROOT}/Experiments/openvla_libero_mc_cache/outputs/semantic_static/${RUN_GROUP}}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SUMMARY_DIR}/logs}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/mnt/data0/zjh_data/Embodied_Proj/experiments/vla_cache_semantic_static/${RUN_GROUP}/rollouts}"

MC_SEARCH_RADIUS="${MC_SEARCH_RADIUS:-28}"
MC_SEARCH_STEP="${MC_SEARCH_STEP:-2}"
MC_MIN_CONFIDENCE="${MC_MIN_CONFIDENCE:-0.30}"
MC_SIMILARITY_THRESHOLD="${MC_SIMILARITY_THRESHOLD:-0.70}"
MC_TOP_K="${MC_TOP_K:-130}"
MC_TASK_TOP_K="${MC_TASK_TOP_K:-120}"

mkdir -p "${SUMMARY_DIR}" "${LOCAL_LOG_DIR}"
if [[ "${SAVE_ROLLOUT_VIDEOS}" == "True" || "${SAVE_ROLLOUT_VIDEOS}" == "true" || "${SAVE_ROLLOUT_VIDEOS}" == "1" ]]; then
  mkdir -p "${ROLLOUT_DIR}"
fi

cd "${OPENVLA_ROOT}"

run_eval() {
  local method="$1"
  shift
  echo "[run] method=${method} task_ids=${TASK_IDS} trials=${TRIALS} camera=${CAMERA}"
  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name libero_spatial \
    --task_ids "${TASK_IDS}" \
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
    mc_cache)
      run_eval "${method}" \
        --use_vla_cache True \
        --use_motion_compensated_cache True \
        --mc_enable_kv_remap False \
        --mc_search_radius "${MC_SEARCH_RADIUS}" \
        --mc_search_step "${MC_SEARCH_STEP}" \
        --mc_min_confidence "${MC_MIN_CONFIDENCE}" \
        --mc_similarity_threshold "${MC_SIMILARITY_THRESHOLD}" \
        --mc_top_k "${MC_TOP_K}" \
        --mc_task_top_k "${MC_TASK_TOP_K}"
      ;;
    mc_kv_remap)
      run_eval "${method}" \
        --use_vla_cache True \
        --use_motion_compensated_cache True \
        --mc_enable_kv_remap True \
        --mc_search_radius "${MC_SEARCH_RADIUS}" \
        --mc_search_step "${MC_SEARCH_STEP}" \
        --mc_min_confidence "${MC_MIN_CONFIDENCE}" \
        --mc_similarity_threshold "${MC_SIMILARITY_THRESHOLD}" \
        --mc_top_k "${MC_TOP_K}" \
        --mc_task_top_k "${MC_TASK_TOP_K}"
      ;;
    *)
      echo "[error] unknown method: ${method}" >&2
      echo "[error] valid methods: full_recompute,original_cache,mc_cache,mc_kv_remap" >&2
      exit 2
      ;;
  esac
done

python "${SCRIPT_DIR}/summarize_libero_eval_matrix.py" \
  --summary-dir "${SUMMARY_DIR}" \
  --output-md "${SUMMARY_DIR}/matrix_summary.md" \
  --output-csv "${SUMMARY_DIR}/matrix_summary.csv"

echo "[ok] semantic-static rollout matrix complete: ${SUMMARY_DIR}"

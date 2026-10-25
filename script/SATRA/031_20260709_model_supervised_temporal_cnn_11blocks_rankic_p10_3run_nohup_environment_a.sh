#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"

cd "${PROJECT_ROOT}"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

if [[ "${1:-}" != "--worker" ]]; then
  run_stamp="$(date +%Y%m%d_%H%M%S)"
  run_dir="${RUN_DIR:-runs/${run_stamp}_temporal_cnn_11blocks_rankic_p10_3run}"
  mkdir -p "${run_dir}"

  log_file="${run_dir}/nohup.log"
  command_file="${run_dir}/command.txt"
  pid_file="${run_dir}/pid"

  {
    echo "log_file=${log_file}"
    echo "started_at=$(date +%Y-%m-%dT%H:%M:%S%z)"
    printf 'cd %q\n' "${PROJECT_ROOT}"
    printf 'setsid nohup bash %q --worker %q > %q 2>&1 < /dev/null &\n' "${SCRIPT_PATH}" "${run_dir}" "${log_file}"
  } > "${command_file}"

  setsid nohup bash "${SCRIPT_PATH}" --worker "${run_dir}" > "${log_file}" 2>&1 < /dev/null &
  pid="$!"
  echo "${pid}" > "${pid_file}"

  echo "[temporal-cnn-11blocks-3run] launched pid=${pid}"
  echo "[temporal-cnn-11blocks-3run] run_dir=${run_dir}"
  echo "[temporal-cnn-11blocks-3run] log=${log_file}"
  echo "[temporal-cnn-11blocks-3run] watch: tail -f \"${log_file}\""
  exit 0
fi

RUN_DIR="${2:?RUN_DIR is required for --worker}"
mkdir -p "${RUN_DIR}"
trap '' HUP

USE_GPU="${USE_GPU:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
ROOT_PATH="${ROOT_PATH:-data/processed}"
DEBUG="${DEBUG:-0}"
DEBUG_TRAIN_STEPS="${DEBUG_TRAIN_STEPS:-1}"
DEBUG_EVAL_STEPS="${DEBUG_EVAL_STEPS:-1}"
DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
METHODS="${METHODS:-temporal_cnn temporal_cnn_bottleneck_pool_moe_1x temporal_cnn_bottleneck_pool_moe_4x}"
NUM_RUNS="${NUM_RUNS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PATIENCE="${PATIENCE:-10}"
LR="${LR:-0.001}"
RESULTS_ROOT_BASE="${RESULTS_ROOT_BASE:-results/supervised_temporal_cnn_11blocks_rankic_p10_3run}"

status_file="${RUN_DIR}/status"
progress_file="${RUN_DIR}/progress.txt"
env_file="${RUN_DIR}/env.txt"

on_error() {
  local exit_code="$?"
  echo "failed exit_code=${exit_code}" > "${status_file}"
  echo "[temporal-cnn-11blocks-3run] worker failed exit_code=${exit_code} $(date +%Y%m%d_%H%M%S)" >&2
  exit "${exit_code}"
}

trap on_error ERR

dataset_arg() {
  local dataset="$1"
  case "${dataset}" in
    NASDAQ)
      printf '%s' "--market_num 20 --valid_index 871 --test_index 996 --data_start_index 0"
      ;;
    NYSE)
      printf '%s' "--market_num 25 --valid_index 871 --test_index 996 --data_start_index 0"
      ;;
    SP500)
      printf '%s' "--market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915"
      ;;
    *)
      echo "unknown dataset=${dataset}" >&2
      return 1
      ;;
  esac
}

method_label() {
  local method="$1"
  case "${method}" in
    temporal_cnn) printf '%s' "Temporal CNN 11 ConvBlocks" ;;
    temporal_cnn_bottleneck_pool_moe_1x) printf '%s' "Temporal CNN 11 ConvBlocks + 1x bottleneck pool MoE top-k=3" ;;
    temporal_cnn_bottleneck_pool_moe_4x) printf '%s' "Temporal CNN 11 ConvBlocks + 4x bottleneck pool MoE top-k=3" ;;
    *)
      echo "unknown method=${method}" >&2
      return 1
      ;;
  esac
}

method_model() {
  local method="$1"
  case "${method}" in
    temporal_cnn) printf '%s' "model_temporal_cnn" ;;
    temporal_cnn_bottleneck_pool_moe_1x) printf '%s' "model_temporal_cnn_bottleneck_pool_moe_1x" ;;
    temporal_cnn_bottleneck_pool_moe_4x) printf '%s' "model_temporal_cnn_bottleneck_pool_moe_4x" ;;
    *)
      echo "unknown method=${method}" >&2
      return 1
      ;;
  esac
}

method_moe_top_k() {
  local method="$1"
  case "${method}" in
    temporal_cnn) printf '%s' "0" ;;
    temporal_cnn_bottleneck_pool_moe_1x|temporal_cnn_bottleneck_pool_moe_4x) printf '%s' "3" ;;
    *)
      echo "unknown method=${method}" >&2
      return 1
      ;;
  esac
}

model_id_for() {
  local method="$1"
  local dataset="$2"
  local lower_dataset
  lower_dataset="$(printf '%s' "${dataset}" | tr '[:upper:]' '[:lower:]')"
  printf 'model_supervised_%s_11blocks_rankic_p10_dtw_%s_final%s_environment_a' \
    "${method}" "${lower_dataset}" "${NUM_RUNS}"
}

write_env() {
  {
    echo "started_at=$(date +%Y-%m-%dT%H:%M:%S%z)"
    echo "methods=${METHODS}"
    echo "datasets=${DATASETS}"
    echo "results_root_base=${RESULTS_ROOT_BASE}"
    echo "num_runs=${NUM_RUNS}"
    echo "train_epochs=${TRAIN_EPOCHS}"
    echo "patience=${PATIENCE}"
    echo "lr=${LR}"
    echo "batch_size=${BATCH_SIZE}"
    echo "use_gpu=${USE_GPU}"
    echo "pin_memory=${PIN_MEMORY}"
    echo "cnn_depth_convention=depth=10 plus one output ConvBlock, 11 ConvBlocks total"
  } > "${env_file}"
}

run_case() {
  local method="$1"
  local dataset="$2"
  local label
  local model_name
  local moe_top_k
  local model_id
  local split_args
  local results_root

  label="$(method_label "${method}")"
  model_name="$(method_model "${method}")"
  moe_top_k="$(method_moe_top_k "${method}")"
  model_id="$(model_id_for "${method}" "${dataset}")"
  split_args="$(dataset_arg "${dataset}")"
  results_root="${RESULTS_ROOT_BASE}/${method}"

  {
    echo "method=${method}"
    echo "label=${label}"
    echo "dataset=${dataset}"
    echo "model=${model_name}"
    echo "model_id=${model_id}"
    echo "results_root=${results_root}"
    echo "started_at=$(date +%Y-%m-%dT%H:%M:%S%z)"
  } > "${progress_file}"

  echo "[temporal-cnn-11blocks-3run] start method=${method} (${label}) dataset=${dataset} model_id=${model_id} $(date +%Y%m%d_%H%M%S)"
  # shellcheck disable=SC2086
  PYTHONPATH=src python run.py \
    --num_runs "${NUM_RUNS}" \
    --debug "${DEBUG}" \
    --debug_train_steps "${DEBUG_TRAIN_STEPS}" \
    --debug_eval_steps "${DEBUG_EVAL_STEPS}" \
    --is_training 1 \
    --model_id "${model_id}" \
    --model "${model_name}" \
    --data "${dataset}" \
    --root_path "${ROOT_PATH}" \
    --results "${results_root}" \
    --input_features all \
    --seq_len 16 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 0 \
    --use_gpu "${USE_GPU}" \
    --pin_memory "${PIN_MEMORY}" \
    --train_drop_last 1 \
    --valid_drop_last 1 \
    ${split_args} \
    --scale_factor 3 \
    --alpha 0.1 \
    --prediction_target return \
    --target_mode raw_return \
    --learning_rate "${LR}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --early_stop_metric RankIC \
    --early_stop_mode max \
    --lr_scheduler cosine_hard_restarts \
    --precision bf16_amp_tf32 \
    --torch_compile 0 \
    --max_grad_norm 1.0 \
    --moe_experts 6 \
    --moe_top_k "${moe_top_k}" \
    --stock_prior dtw \
    --stock_prior_cache_dir "${ROOT_PATH}/${dataset}_ohlcv/priors" \
    --stock_prior_feature close \
    --stock_prior_topk 16 \
    --stock_prior_candidate_k 48 \
    --stock_prior_downsample 64 \
    --stock_prior_band 6 \
    --stock_prior_gamma 2.0 \
    --stock_prior_scale 1.0
  echo "[temporal-cnn-11blocks-3run] done method=${method} (${label}) dataset=${dataset} model_id=${model_id} $(date +%Y%m%d_%H%M%S)"
}

write_env
echo "running" > "${status_file}"
echo "[temporal-cnn-11blocks-3run] worker start $(date +%Y%m%d_%H%M%S)"
echo "[temporal-cnn-11blocks-3run] methods=${METHODS}"
echo "[temporal-cnn-11blocks-3run] datasets=${DATASETS}"
echo "[temporal-cnn-11blocks-3run] results_root_base=${RESULTS_ROOT_BASE}"
echo "[temporal-cnn-11blocks-3run] num_runs=${NUM_RUNS} train_epochs=${TRAIN_EPOCHS} patience=${PATIENCE} lr=${LR}"

for method in ${METHODS}; do
  for dataset in ${DATASETS}; do
    run_case "${method}" "${dataset}"
  done
done

echo "complete" > "${status_file}"
echo "[temporal-cnn-11blocks-3run] worker complete $(date +%Y%m%d_%H%M%S)"

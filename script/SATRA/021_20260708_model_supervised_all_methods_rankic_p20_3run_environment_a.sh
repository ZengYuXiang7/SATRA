#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

USE_GPU="${USE_GPU:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
ROOT_PATH="${ROOT_PATH:-data/processed}"
DEBUG="${DEBUG:-0}"
DEBUG_TRAIN_STEPS="${DEBUG_TRAIN_STEPS:-1}"
DEBUG_EVAL_STEPS="${DEBUG_EVAL_STEPS:-1}"
DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
METHODS="${METHODS:-dense_moe topk2_moe plain_ffn bottleneck_pool_moe temporal_cnn temporal_cnn_bottleneck_pool_moe_1x temporal_cnn_bottleneck_pool_moe_4x}"
NUM_RUNS="${NUM_RUNS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PATIENCE="${PATIENCE:-20}"
LR="${LR:-0.001}"
RESULTS_ROOT_BASE="${RESULTS_ROOT_BASE:-results/supervised_all_methods_rankic_p20_3run}"

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
    dense_moe) printf '%s' "all-expert dense MoE" ;;
    topk2_moe) printf '%s' "MoE top-k=2" ;;
    plain_ffn) printf '%s' "Plain FFN" ;;
    bottleneck_pool_moe) printf '%s' "Bottleneck Pool MoE top-k=3" ;;
    temporal_cnn) printf '%s' "Temporal CNN" ;;
    temporal_cnn_bottleneck_pool_moe_1x) printf '%s' "Temporal CNN + 1x bottleneck pool MoE top-k=3" ;;
    temporal_cnn_bottleneck_pool_moe_4x) printf '%s' "Temporal CNN + 4x bottleneck pool MoE top-k=3" ;;
    *)
      echo "unknown method=${method}" >&2
      return 1
      ;;
  esac
}

method_model() {
  local method="$1"
  case "${method}" in
    dense_moe|topk2_moe) printf '%s' "model" ;;
    plain_ffn) printf '%s' "model_plain_ffn" ;;
    bottleneck_pool_moe) printf '%s' "model_bottleneck_pool_moe" ;;
    temporal_cnn) printf '%s' "model_temporal_cnn" ;;
    temporal_cnn_bottleneck_pool_moe_1x) printf '%s' "model_temporal_cnn_bottleneck_pool_moe_1x" ;;
    temporal_cnn_bottleneck_pool_moe_4x) printf '%s' "model_temporal_cnn_bottleneck_pool_moe_4x" ;;
    *)
      echo "unknown method=${method}" >&2
      return 1
      ;;
  esac
}

method_moe_experts() {
  local method="$1"
  case "${method}" in
    dense_moe|topk2_moe|plain_ffn|bottleneck_pool_moe|temporal_cnn|temporal_cnn_bottleneck_pool_moe_1x|temporal_cnn_bottleneck_pool_moe_4x) printf '%s' "6" ;;
    *)
      echo "unknown method=${method}" >&2
      return 1
      ;;
  esac
}

method_moe_top_k() {
  local method="$1"
  case "${method}" in
    dense_moe) printf '%s' "0" ;;
    topk2_moe|plain_ffn) printf '%s' "2" ;;
    temporal_cnn) printf '%s' "0" ;;
    bottleneck_pool_moe|temporal_cnn_bottleneck_pool_moe_1x|temporal_cnn_bottleneck_pool_moe_4x) printf '%s' "3" ;;
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
  printf 'model_supervised_all_methods_%s_rankic_p20_dtw_%s_final%s_environment_a' \
    "${method}" "${lower_dataset}" "${NUM_RUNS}"
}

run_case() {
  local method="$1"
  local dataset="$2"
  local label
  local model_name
  local moe_experts
  local moe_top_k
  local model_id
  local split_args
  local results_root

  label="$(method_label "${method}")"
  model_name="$(method_model "${method}")"
  moe_experts="$(method_moe_experts "${method}")"
  moe_top_k="$(method_moe_top_k "${method}")"
  model_id="$(model_id_for "${method}" "${dataset}")"
  split_args="$(dataset_arg "${dataset}")"
  results_root="${RESULTS_ROOT_BASE}/${method}"

  echo "[all-methods] start method=${method} (${label}) dataset=${dataset} model_id=${model_id} $(date +%Y%m%d_%H%M%S)"
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
    --moe_experts "${moe_experts}" \
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
  echo "[all-methods] done method=${method} (${label}) dataset=${dataset} model_id=${model_id} $(date +%Y%m%d_%H%M%S)"
}

echo "[all-methods] methods=${METHODS}"
echo "[all-methods] datasets=${DATASETS}"
echo "[all-methods] results_root_base=${RESULTS_ROOT_BASE}"
echo "[all-methods] num_runs=${NUM_RUNS} train_epochs=${TRAIN_EPOCHS} patience=${PATIENCE} lr=${LR}"

for method in ${METHODS}; do
  for dataset in ${DATASETS}; do
    run_case "${method}" "${dataset}"
  done
done

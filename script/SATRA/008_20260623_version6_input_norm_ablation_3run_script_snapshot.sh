#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
FINETUNE_RUNS="${FINETUNE_RUNS:-3}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100}"
USE_GPU="${USE_GPU:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION:-tanh}"
VARIANTS="${VARIANTS:-stock_feature_minmax global_feature_minmax stock_price_volume_minmax global_price_volume_minmax stock_feature_standard global_feature_standard stock_price_volume_standard global_price_volume_standard}"

variant_args() {
  local variant="$1"
  case "${variant}" in
    stock_feature_minmax)
      printf '%s' "--input_norm_method minmax --input_norm_scope stock --raw_norm_layout feature"
      ;;
    global_feature_minmax)
      printf '%s' "--input_norm_method minmax --input_norm_scope global --raw_norm_layout feature"
      ;;
    stock_price_volume_minmax)
      printf '%s' "--input_norm_method minmax --input_norm_scope stock --raw_norm_layout price_volume"
      ;;
    global_price_volume_minmax)
      printf '%s' "--input_norm_method minmax --input_norm_scope global --raw_norm_layout price_volume"
      ;;
    stock_feature_standard)
      printf '%s' "--input_norm_method standard --input_norm_scope stock --raw_norm_layout feature"
      ;;
    global_feature_standard)
      printf '%s' "--input_norm_method standard --input_norm_scope global --raw_norm_layout feature"
      ;;
    stock_price_volume_standard)
      printf '%s' "--input_norm_method standard --input_norm_scope stock --raw_norm_layout price_volume"
      ;;
    global_price_volume_standard)
      printf '%s' "--input_norm_method standard --input_norm_scope global --raw_norm_layout price_volume"
      ;;
    *)
      echo "unknown variant=${variant}" >&2
      return 1
      ;;
  esac
}

for variant in ${VARIANTS}; do
  data_root="dataset_norm/${variant}"
  results_root="results_norm/${variant}"
  echo "[norm-ablation] generate variant=${variant} data_root=${data_root} $(date +%Y%m%d_%H%M%S)"
  # shellcheck disable=SC2086
  PYTHONPATH=src python -m satra.data_provider.normalize_existing_data \
    --source-root dataset \
    --output-root "${data_root}" \
    --markets ${DATASETS} \
    --clean-output \
    $(variant_args "${variant}")

  for dataset in ${DATASETS}; do
    echo "[norm-ablation] start variant=${variant} dataset=${dataset} runs=${FINETUNE_RUNS} output_activation=${OUTPUT_ACTIVATION} $(date +%Y%m%d_%H%M%S)"
    DATASET="${dataset}" \
    ROOT_PATH="${data_root}" \
    RESULTS_ROOT="${results_root}" \
    FINETUNE_RUNS="${FINETUNE_RUNS}" \
    PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" \
    FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" \
    PRETRAIN_MASK_MODE="random_token" \
    PRETRAIN_TARGET_MODE="return_residual" \
    MAE_MASK_RATIO="0.3" \
    RET_FEATURE_INDEX="-1" \
    USE_GPU="${USE_GPU}" \
    PIN_MEMORY="${PIN_MEMORY}" \
    OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" \
    PYTHONPATH=src python -m stock.engine.version6_two_stage
    echo "[norm-ablation] done variant=${variant} dataset=${dataset} $(date +%Y%m%d_%H%M%S)"
  done
done

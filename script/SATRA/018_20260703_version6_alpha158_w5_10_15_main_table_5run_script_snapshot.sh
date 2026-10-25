#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
FINETUNE_RUNS="${FINETUNE_RUNS:-5}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100}"
USE_GPU="${USE_GPU:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION:-tanh}"
ALPHA158_WINDOWS="${ALPHA158_WINDOWS:-5 10 15}"
DATA_ROOT="${DATA_ROOT:-data/processed_alpha158_w5_10_15}"
RESULTS_ROOT="${RESULTS_ROOT:-results/main_table_5run_alpha158_w5_10_15}"

echo "[alpha158-w5-10-15-main-table-5run] generate data_root=${DATA_ROOT} windows=${ALPHA158_WINDOWS} $(date +%Y%m%d_%H%M%S)"
PYTHONPATH=src python -m satra.data_provider.generate_data \
  --predata-root dataset/raw_data \
  --output-root "${DATA_ROOT}" \
  --markets ${DATASETS} \
  --feature_set stock_market_aug \
  --use_alpha158_features 1 \
  --alpha158_windows ${ALPHA158_WINDOWS} \
  --normalization_scope stock \
  --impute_missing linear \
  --clean-output

for dataset in ${DATASETS}; do
  echo "[alpha158-w5-10-15-main-table-5run] start dataset=${dataset} runs=${FINETUNE_RUNS} output_activation=${OUTPUT_ACTIVATION} $(date +%Y%m%d_%H%M%S)"
  DATASET="${dataset}" \
  ROOT_PATH="${DATA_ROOT}" \
  RESULTS_ROOT="${RESULTS_ROOT}" \
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
  echo "[alpha158-w5-10-15-main-table-5run] done dataset=${dataset} $(date +%Y%m%d_%H%M%S)"
done

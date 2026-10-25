#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

USE_GPU="${USE_GPU:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
ROOT_PATH="${ROOT_PATH:-data/processed}"
DATASETS="${DATASETS:-NASDAQ}"
FINETUNE_RUNS="${FINETUNE_RUNS:-10}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100}"
MODEL_ID_PREFIX="${MODEL_ID_PREFIX:-}"

model_id_for() {
  local dataset="$1"
  local lower_dataset
  lower_dataset="$(printf '%s' "${dataset}" | tr '[:upper:]' '[:lower:]')"

  if [[ -n "${MODEL_ID_PREFIX}" ]]; then
    printf '%s_%s_final%s_environment_a' "${MODEL_ID_PREFIX}" "${lower_dataset}" "${FINETUNE_RUNS}"
  else
    printf 'model_random_token_dtw_%s_final%s_environment_a' "${lower_dataset}" "${FINETUNE_RUNS}"
  fi
}

run_case() {
  local dataset="$1"
  local model_id="$2"
  echo "[model-two-stage] start dataset=${dataset} model_id=${model_id} $(date +%Y%m%d_%H%M%S)"
  DATASET="${dataset}" \
  MODEL_ID="${model_id}" \
  FINETUNE_RUNS="${FINETUNE_RUNS}" \
  PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" \
  FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" \
  ROOT_PATH="${ROOT_PATH}" \
  USE_GPU="${USE_GPU}" \
  PIN_MEMORY="${PIN_MEMORY}" \
  PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
  echo "[model-two-stage] done dataset=${dataset} model_id=${model_id} $(date +%Y%m%d_%H%M%S)"
}

for dataset in ${DATASETS}; do
  run_case "${dataset}" "$(model_id_for "${dataset}")"
done

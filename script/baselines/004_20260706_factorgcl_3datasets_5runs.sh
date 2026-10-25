#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


if [ -n "${PYTHON:-}" ]; then
  RUNNER=("${PYTHON}")
else
  PYTHON_BIN="${PYTHON:-$(command -v python || true)}"
  if [ -z "${PYTHON_BIN}" ] && [ -x "${HOME}/.local/bin/python" ]; then
    PYTHON_BIN="${HOME}/.local/bin/python"
  fi
  RUNNER=("${PYTHON_BIN}")
fi

DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
NUM_RUNS="${NUM_RUNS:-5}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USE_GPU="${USE_GPU:-1}"
GPU="${GPU:-0}"

for dataset in ${DATASETS}; do
  "${RUNNER[@]}" -m factorgcl.run \
    --data "${dataset}" \
    --root_path "${ROOT_PATH:-data/processed}" \
    --relation_root "${RELATION_ROOT:-data/processed/relations}" \
    --relation_name "${RELATION_NAME:-auto}" \
    --prior_beta_name "${PRIOR_BETA_NAME:-factor_prior_beta}" \
    --results "${RESULTS:-results}" \
    --input_features "${INPUT_FEATURES:-stock_all}" \
    --num_runs "${NUM_RUNS}" \
    --debug "${DEBUG:-0}" \
    --seq_len "${SEQ_LEN:-16}" \
    --future_seq_len "${FUTURE_SEQ_LEN:-16}" \
    --steps "${STEPS:-1}" \
    --batch_size "${BATCH_SIZE}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE:-1e-3}" \
    --weight_decay "${WEIGHT_DECAY:-0.0}" \
    --hidden_dim "${HIDDEN_DIM:-32}" \
    --num_hidden_factors "${NUM_HIDDEN_FACTORS:-32}" \
    --num_gru_layers "${NUM_GRU_LAYERS:-2}" \
    --dropout "${DROPOUT:-0.1}" \
    --projection_dim "${PROJECTION_DIM:-0}" \
    --temperature "${TEMPERATURE:-0.1}" \
    --alpha "${ALPHA:-0.1}" \
    --gamma "${GAMMA:-0.1}" \
    --use_contrastive "${USE_CONTRASTIVE:-1}" \
    --patience "${PATIENCE:-10}" \
    --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
    --monitor_metric "${MONITOR_METRIC:-RankIC}" \
    --monitor_mode "${MONITOR_MODE:-max}" \
    --precision "${PRECISION:-bf16_amp_tf32}" \
    --torch_compile "${TORCH_COMPILE:-1}" \
    --compile_mode "${COMPILE_MODE:-reduce-overhead}" \
    --fused_optimizer "${FUSED_OPTIMIZER:-1}" \
    --use_gpu "${USE_GPU}" \
    --gpu "${GPU}"
done

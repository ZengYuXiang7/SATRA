#!/usr/bin/env bash
set -e
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

# Dataset split is FactorVAEData's current 7:1:2 project split, not the
# FactorVAE paper's original date periods.
DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
NUM_RUNS="${NUM_RUNS:-5}"
USE_GPU="${USE_GPU:-1}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
DEBUG="${DEBUG:-0}"
ROOT_PATH="${ROOT_PATH:-data/processed}"
INPUT_FEATURES="${INPUT_FEATURES:-factorvae_alpha20}"

# Source status:
# - Current final baseline protocol: seq_len=16, max_epochs=100 with early stopping.
# - Paper/core reproduction: NLL + gamma * KL objective.
# - Public-reproduction defaults, not confirmed as official best parameters:
#   num_factors=96, num_portfolios=128, hidden_size=64, lr=1e-4.
# - The paper text does not clearly expose the tuned gamma value; 1.0 is the
#   current implementation default.
for dataset in ${DATASETS}; do
    "${RUNNER[@]}" -m factorvae.run \
    --data "${dataset}" \
    --root_path "${ROOT_PATH}" \
    --results "${RESULTS:-results}" \
    --num_runs "${NUM_RUNS}" \
    --debug "${DEBUG}" \
    --seq_len "${SEQ_LEN:-16}" \
    --steps 1 \
    --input_features "${INPUT_FEATURES}" \
    --batch_size "${BATCH_SIZE}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --num_factors 96 \
    --num_portfolios 128 \
    --hidden_size 64 \
    --num_gru_layers 1 \
    --kl_weight 1.0 \
    --patience 10 \
    --max_grad_norm 1.0 \
    --monitor_metric "${MONITOR_METRIC:-RankIC}" \
    --monitor_mode "${MONITOR_MODE:-max}" \
    --precision "${PRECISION:-bf16_amp_tf32}" \
    --torch_compile "${TORCH_COMPILE:-1}" \
    --compile_mode "${COMPILE_MODE:-reduce-overhead}" \
    --fused_optimizer "${FUSED_OPTIMIZER:-1}" \
    --use_gpu "${USE_GPU}" \
    --gpu "${GPU}"
done

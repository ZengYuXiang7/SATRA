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
BATCH_SIZE="${BATCH_SIZE:-8}"
USE_GPU="${USE_GPU:-1}"
GPU="${GPU:-0}"

for dataset in ${DATASETS}; do
  "${RUNNER[@]}" -m tra.run \
    --data "${dataset}" \
    --root_path "${ROOT_PATH:-data/processed}" \
    --relation_root "${RELATION_ROOT:-data/processed/relations}" \
    --relation_name "${RELATION_NAME:-auto}" \
    --results "${RESULTS:-results}" \
    --input_features "${INPUT_FEATURES:-stock_all}" \
    --num_runs "${NUM_RUNS}" \
    --debug "${DEBUG:-0}" \
    --seq_len "${SEQ_LEN:-16}" \
    --steps "${STEPS:-1}" \
    --batch_size "${BATCH_SIZE}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE:-1e-3}" \
    --weight_decay "${WEIGHT_DECAY:-0.0}" \
    --hidden_size "${HIDDEN_SIZE:-64}" \
    --num_layers "${NUM_LAYERS:-1}" \
    --num_predictors "${NUM_PREDICTORS:-5}" \
    --router_hidden "${ROUTER_HIDDEN:-64}" \
    --dropout "${DROPOUT:-0.1}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --hard_routing "${HARD_ROUTING:-0}" \
    --alpha "${ALPHA:-0.1}" \
    --ot_weight "${OT_WEIGHT:-0.1}" \
    --sinkhorn_epsilon "${SINKHORN_EPSILON:-0.1}" \
    --sinkhorn_iters "${SINKHORN_ITERS:-20}" \
    --memory_decay "${MEMORY_DECAY:-0.9}" \
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

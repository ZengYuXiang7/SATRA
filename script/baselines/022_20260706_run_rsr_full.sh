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

DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
NUM_RUNS="${NUM_RUNS:-5}"
USE_GPU="${USE_GPU:-1}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
DEBUG="${DEBUG:-0}"
ROOT_PATH="${ROOT_PATH:-data/processed}"
RELATION_ROOT="${RELATION_ROOT:-data/processed/relations}"
INPUT_FEATURES="${INPUT_FEATURES:-stock_all}"

for dataset in ${DATASETS}; do
  "${RUNNER[@]}" -m rsr.run \
    --data "${dataset}" \
    --root_path "${ROOT_PATH}" \
    --relation_root "${RELATION_ROOT}" \
    --relation_name "${RELATION_NAME:-auto}" \
    --results "${RESULTS:-results}" \
    --input_features "${INPUT_FEATURES}" \
    --num_runs "${NUM_RUNS}" \
    --debug "${DEBUG}" \
    --seq_len "${SEQ_LEN:-16}" \
    --steps "${STEPS:-1}" \
    --batch_size "${BATCH_SIZE}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE:-1e-3}" \
    --weight_decay "${WEIGHT_DECAY:-0.0}" \
    --hidden_size "${HIDDEN_SIZE:-64}" \
    --num_layers "${NUM_LAYERS:-1}" \
    --dropout "${DROPOUT:-0.0}" \
    --flat_relation "${FLAT_RELATION:-0}" \
    --inner_product "${INNER_PRODUCT:-0}" \
    --softmax_axis "${SOFTMAX_AXIS:-source}" \
    --output_activation "${OUTPUT_ACTIVATION:-leaky_relu}" \
    --alpha "${ALPHA:-0.1}" \
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

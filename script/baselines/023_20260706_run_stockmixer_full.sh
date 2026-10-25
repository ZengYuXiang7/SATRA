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
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
USE_GPU="${USE_GPU:-1}"
GPU="${GPU:-0}"

for dataset in ${DATASETS}; do
  "${RUNNER[@]}" -m stockmixer.run \
    --data "${dataset}" \
    --root_path "${ROOT_PATH:-data/processed}" \
    --results "${RESULTS:-results}" \
    --input_features "${INPUT_FEATURES:-stockmixer_ma}" \
    --num_runs "${NUM_RUNS}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE:-0.001}" \
    --alpha "${ALPHA:-0.1}" \
    --seq_len "${SEQ_LEN:-16}" \
    --steps "${STEPS:-1}" \
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

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
TRAIN_EPOCHS="${TRAIN_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USE_GPU="${USE_GPU:-1}"
GPU="${GPU:-0}"

for dataset in ${DATASETS}; do
  "${RUNNER[@]}" -m master.run \
    --data "${dataset}" \
    --root_path "${ROOT_PATH:-data/processed}" \
    --results "${RESULTS:-results}" \
    --input_features "${INPUT_FEATURES:-master_stock_market}" \
    --num_runs "${NUM_RUNS}" \
    --debug "${DEBUG:-0}" \
    --seq_len "${SEQ_LEN:-16}" \
    --steps "${STEPS:-1}" \
    --batch_size "${BATCH_SIZE}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --weight_decay "${WEIGHT_DECAY:-0.0}" \
    --d_model "${D_MODEL:-256}" \
    --t_nhead "${T_NHEAD:-4}" \
    --s_nhead "${S_NHEAD:-2}" \
    --gate_beta "${GATE_BETA:-5.0}" \
    --dropout "${DROPOUT:-0.5}" \
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

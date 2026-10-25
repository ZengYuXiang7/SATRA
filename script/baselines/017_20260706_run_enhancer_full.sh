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
BATCH_SIZE="${BATCH_SIZE:-6}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
DEBUG="${DEBUG:-0}"
ROOT_PATH="${ROOT_PATH:-data/processed}"
INPUT_FEATURES="${INPUT_FEATURES:-enhancer_stock}"

for dataset in ${DATASETS}; do
  "${RUNNER[@]}" -m enhancer.run \
    --data "${dataset}" \
    --root_path "${ROOT_PATH}" \
    --results "${RESULTS:-results}" \
    --input_features "${INPUT_FEATURES}" \
    --num_runs "${NUM_RUNS}" \
    --debug "${DEBUG}" \
    --seq_len "${SEQ_LEN:-16}" \
    --steps "${STEPS:-1}" \
    --batch_size "${BATCH_SIZE}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE:-7e-4}" \
    --weight_decay "${WEIGHT_DECAY:-0.0}" \
    --hidden_dim "${HIDDEN_DIM:-30}" \
    --relation_dim "${RELATION_DIM:-30}" \
    --num_contexts "${NUM_CONTEXTS:-30}" \
    --polynomial_order "${POLYNOMIAL_ORDER:-3}" \
    --relation_topk "${RELATION_TOPK:-0}" \
    --sgc_steps "${SGC_STEPS:-2}" \
    --dropout "${DROPOUT:-0.1}" \
    --mse_weight "${MSE_WEIGHT:-1.0}" \
    --rank_weight "${RANK_WEIGHT:-4.0}" \
    --reg1_weight "${REG1_WEIGHT:-12.0}" \
    --reg2_weight "${REG2_WEIGHT:-1.0}" \
    --noise_scale "${NOISE_SCALE:-1e-3}" \
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

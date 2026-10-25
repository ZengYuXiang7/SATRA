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
HYPERGRAPH_NAME="${HYPERGRAPH_NAME:-auto}"

for dataset in ${DATASETS}; do
  "${RUNNER[@]}" -m ci_sthpan.run \
    --data "${dataset}" \
    --root_path "${ROOT_PATH}" \
    --relation_root "${RELATION_ROOT}" \
    --hypergraph_name "${HYPERGRAPH_NAME}" \
    --results "${RESULTS:-results}" \
    --input_features "${INPUT_FEATURES}" \
    --num_runs "${NUM_RUNS}" \
    --debug "${DEBUG}" \
    --seq_len "${SEQ_LEN:-16}" \
    --steps "${STEPS:-1}" \
    --batch_size "${BATCH_SIZE}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --weight_decay "${WEIGHT_DECAY:-0.0}" \
    --patch_len "${PATCH_LEN:-12}" \
    --stride "${STRIDE:-12}" \
    --d_model "${D_MODEL:-128}" \
    --n_heads "${N_HEADS:-16}" \
    --n_layers "${N_LAYERS:-3}" \
    --d_ff "${D_FF:-256}" \
    --dropout "${DROPOUT:-0.2}" \
    --use_revin "${USE_REVIN:-1}" \
    --use_channel_independence "${USE_CHANNEL_INDEPENDENCE:-1}" \
    --use_hypergraph "${USE_HYPERGRAPH:-1}" \
    --share_hypergraph_across_channels "${SHARE_HYPERGRAPH_ACROSS_CHANNELS:-1}" \
    --alpha "${ALPHA:-2.0}" \
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

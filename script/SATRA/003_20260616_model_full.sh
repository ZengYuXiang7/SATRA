#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

SATRA_PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
while [ ! -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] && [ "${SATRA_PROJECT_ROOT}" != "/" ]; do SATRA_PROJECT_ROOT="$(dirname "${SATRA_PROJECT_ROOT}")"; done
[ -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] || { echo "Unable to locate the project root." >&2; exit 1; }
cd "${SATRA_PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
if [[ -z "$PYTHON_BIN" && -x "$HOME/.local/bin/python" ]]; then
  PYTHON_BIN="$HOME/.local/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python was not found; set PYTHON_BIN to a Python interpreter." >&2
  exit 1
fi

USE_GPU="${USE_GPU:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"

"$PYTHON_BIN" run.py \
  --num_runs 3 \
  --is_training 1 \
  --model_id "nasdaq_model" \
  --model model \
  --data NASDAQ \
  --input_features all \
  --seq_len 16 \
  --batch_size 8 \
  --use_gpu "$USE_GPU" \
  --pin_memory "$PIN_MEMORY" \
  --train_drop_last 1 \
  --valid_drop_last 0 \
  --valid_index 756 \
  --test_index 1008 \
  --data_start_index 0 \
  --market_num 20 \
  --scale_factor 3 \
  --alpha 0.1 \
  --prediction_target return \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 20 \
  --early_stop_metric RankIC \
  --early_stop_mode max \
  --lr_scheduler cosine_hard_restarts

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
  if [ -z "${PYTHON_BIN}" ]; then
    echo "Python was not found. Set PYTHON to a Python interpreter." >&2
    exit 127
  fi
  RUNNER=("${PYTHON_BIN}")
fi

if [ -z "${MODEL:-}" ]; then
  echo "Set MODEL explicitly, for example MODEL=master." >&2
  exit 2
fi

MODEL="$(printf '%s' "${MODEL}" | tr '[:upper:]' '[:lower:]')"
DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
NUM_RUNS="${NUM_RUNS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
USE_GPU="${USE_GPU:-1}"

if [ -z "${ROOT_PATH:-}" ]; then
  ROOT_PATH="data/processed"
fi

if [ -z "${INPUT_FEATURES:-}" ] && [ "${MODEL}" = "master" ]; then
  INPUT_FEATURES="all"
fi

for dataset in ${DATASETS}; do
  CMD=(
    "${RUNNER[@]}" -m "${MODEL}.run"
    --data "${dataset}"
    --root_path "${ROOT_PATH}"
    --results results
    --num_runs "${NUM_RUNS}"
    --train_epochs "${TRAIN_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --use_gpu "${USE_GPU}"
  )
  if [ -n "${INPUT_FEATURES:-}" ]; then
    CMD+=(--input_features "${INPUT_FEATURES}")
  fi
  "${CMD[@]}"
done

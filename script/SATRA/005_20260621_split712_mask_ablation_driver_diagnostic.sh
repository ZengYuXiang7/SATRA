#!/usr/bin/env bash
set -u
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

SATRA_PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
while [ ! -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] && [ "${SATRA_PROJECT_ROOT}" != "/" ]; do
  SATRA_PROJECT_ROOT="$(dirname "${SATRA_PROJECT_ROOT}")"
done
[ -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] || {
  echo "Unable to locate the project root." >&2
  exit 97
}
RUN_ID='split712_mask_ablation_diagnostic'
LOG_DIR="$SATRA_PROJECT_ROOT/run_logs/diagnostic"
mkdir -p "$LOG_DIR"
cd "$SATRA_PROJECT_ROOT" || exit 97
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
START_MARKER="$LOG_DIR/${RUN_ID}.start"
touch "$START_MARKER"
echo "[diagnostic-driver] run_id=$RUN_ID"
echo "[diagnostic-driver] start=$(date '+%Y-%m-%d %H:%M:%S %z')"
TARGET_LAUNCHER="${TARGET_LAUNCHER:-}"
if [ -z "$TARGET_LAUNCHER" ] || [ ! -f "$TARGET_LAUNCHER" ]; then
  echo "[diagnostic-driver] set TARGET_LAUNCHER to the recovered historical launcher." >&2
  exit 98
fi
set +e
DATASETS='NASDAQ NYSE SP500' CASES='no_pretrain random_token cross_sectional_random' NUM_RUNS='5' PRETRAIN_EPOCHS='20' FINETUNE_EPOCHS='100' TRAIN_EPOCHS='100' PATIENCE='5' USE_GPU='1' PIN_MEMORY='1' CS_TIME_RATIO='0.3' CS_STOCK_RATIO='0.2' bash "$TARGET_LAUNCHER"
status=$?
set -e
echo "[diagnostic-driver] finish=$(date '+%Y-%m-%d %H:%M:%S %z') status=$status"
find results -path '*/summary_all_runs.json' -newer "$START_MARKER" -print 2>/dev/null | sort | sed 's#/summary_all_runs.json##' > "$LOG_DIR/${RUN_ID}.output_dirs"
echo "$status" > "$LOG_DIR/${RUN_ID}.exit"
exit "$status"

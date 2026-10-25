#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


export DATASETS="${DATASETS:-NASDAQ NYSE SP500}"
export NUM_RUNS="${NUM_RUNS:-5}"
export DEBUG="${DEBUG:-0}"
export USE_GPU="${USE_GPU:-1}"
export GPU="${GPU:-0}"
export ROOT_PATH="${ROOT_PATH:-data/processed}"
export RESULTS="${RESULTS:-results}"
export MONITOR_METRIC="${MONITOR_METRIC:-RankIC}"
export MONITOR_MODE="${MONITOR_MODE:-max}"
export PRECISION="${PRECISION:-bf16_amp_tf32}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
export FUSED_OPTIMIZER="${FUSED_OPTIMIZER:-1}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

echo "[queue] skipped completed: stockmixer factorvae master enhancer"
echo "[queue] running light-to-heavy: tra -> rsr -> hstockgat -> factorgcl -> ci_sthpan"

bash script/baselines/026_20260706_tra_3datasets_5runs.sh
bash script/baselines/013_20260706_rsr_3datasets_5runs.sh
bash script/baselines/007_20260706_hstockgat_3datasets_5runs.sh
bash script/baselines/004_20260706_factorgcl_3datasets_5runs.sh
bash script/baselines/001_20260706_ci_sthpan_3datasets_5runs.sh

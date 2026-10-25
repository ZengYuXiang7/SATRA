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
export RESULTS="${RESULTS:-results_rankic}"
export MONITOR_METRIC="RankIC"
export MONITOR_MODE="max"
export PRECISION="${PRECISION:-bf16_amp_tf32}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
export FUSED_OPTIMIZER="${FUSED_OPTIMIZER:-1}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

echo "[rankic] datasets=${DATASETS}"
echo "[rankic] num_runs=${NUM_RUNS}"
echo "[rankic] results=${RESULTS}"
echo "[rankic] monitor=${MONITOR_METRIC}/${MONITOR_MODE}"
echo "[rankic] queue: stockmixer -> factorvae -> master -> enhancer -> tra -> rsr -> hstockgat -> factorgcl -> ci_sthpan"

bash script/baselines/023_20260706_run_stockmixer_full.sh
bash script/baselines/019_20260706_run_factorvae_full.sh
bash script/baselines/021_20260706_run_master_full.sh
bash script/baselines/017_20260706_run_enhancer_full.sh
bash script/baselines/024_20260706_run_tra_full.sh
bash script/baselines/022_20260706_run_rsr_full.sh
bash script/baselines/020_20260706_run_hstockgat_full.sh
bash script/baselines/018_20260706_run_factorgcl_full.sh
bash script/baselines/016_20260706_run_ci_sthpan_full.sh

echo "[rankic] completed all queued full scripts"

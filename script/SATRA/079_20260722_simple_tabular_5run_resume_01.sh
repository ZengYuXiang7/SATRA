#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

SATRA_PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
while [ ! -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] && [ "${SATRA_PROJECT_ROOT}" != "/" ]; do SATRA_PROJECT_ROOT="$(dirname "${SATRA_PROJECT_ROOT}")"; done
[ -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] || { echo "Unable to locate the project root." >&2; exit 1; }
cd "${SATRA_PROJECT_ROOT}"

COMPLETED_TAG=20260722_113439
RESUME_TAG=20260722_115737
RESULTS_ROOT=results/simple_tabular_baselines_5run
EFFICIENCY_ROOT=results/efficiency_profile_${RESUME_TAG}/simple_tabular

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src

# Finish the two simple linear models on the two remaining markets first.
python -m baselines.simple_tabular.run \
  --model linear_regression --data NYSE \
  --root_path data/processed \
  --results "$RESULTS_ROOT" --exp_tag "${RESUME_TAG}_NYSE_linear_regression" \
  --seq_len 16 --steps 1 --input_features raw_ohlcv \
  --valid_index 871 --test_index 996 --data_start_index 0 \
  --num_runs 5 --random_seed 20260716 --workers 16

python -m baselines.simple_tabular.run \
  --model ridge --data NYSE \
  --root_path data/processed \
  --results "$RESULTS_ROOT" --exp_tag "${RESUME_TAG}_NYSE_ridge" \
  --seq_len 16 --steps 1 --input_features raw_ohlcv \
  --valid_index 871 --test_index 996 --data_start_index 0 \
  --ridge_alpha_candidates 0.1,1.0,10.0 \
  --num_runs 5 --random_seed 20260716 --workers 16

python -m baselines.simple_tabular.run \
  --model linear_regression --data SP500 \
  --root_path data/processed \
  --results "$RESULTS_ROOT" --exp_tag "${RESUME_TAG}_SP500_linear_regression" \
  --seq_len 16 --steps 1 --input_features raw_ohlcv \
  --valid_index 1127 --test_index 1288 --data_start_index 915 \
  --num_runs 5 --random_seed 20260716 --workers 16

python -m baselines.simple_tabular.run \
  --model ridge --data SP500 \
  --root_path data/processed \
  --results "$RESULTS_ROOT" --exp_tag "${RESUME_TAG}_SP500_ridge" \
  --seq_len 16 --steps 1 --input_features raw_ohlcv \
  --valid_index 1127 --test_index 1288 --data_start_index 915 \
  --ridge_alpha_candidates 0.1,1.0,10.0 \
  --num_runs 5 --random_seed 20260716 --workers 16

# Run all three tree-model units only after the simple models are complete.
python -m baselines.simple_tabular.run \
  --model lightgbm --data NASDAQ \
  --root_path data/processed \
  --results "$RESULTS_ROOT" --exp_tag "${RESUME_TAG}_NASDAQ_lightgbm" \
  --seq_len 16 --steps 1 --input_features all \
  --valid_index 871 --test_index 996 --data_start_index 0 \
  --lgbm_tree_candidates 100,200 \
  --num_runs 5 --random_seed 20260716 --workers 16

python -m baselines.simple_tabular.run \
  --model lightgbm --data NYSE \
  --root_path data/processed \
  --results "$RESULTS_ROOT" --exp_tag "${RESUME_TAG}_NYSE_lightgbm" \
  --seq_len 16 --steps 1 --input_features all \
  --valid_index 871 --test_index 996 --data_start_index 0 \
  --lgbm_tree_candidates 100,200 \
  --num_runs 5 --random_seed 20260716 --workers 16

python -m baselines.simple_tabular.run \
  --model lightgbm --data SP500 \
  --root_path data/processed \
  --results "$RESULTS_ROOT" --exp_tag "${RESUME_TAG}_SP500_lightgbm" \
  --seq_len 16 --steps 1 --input_features all \
  --valid_index 1127 --test_index 1288 --data_start_index 915 \
  --lgbm_tree_candidates 100,200 \
  --num_runs 5 --random_seed 20260716 --workers 16

python -m baselines.simple_tabular.profile \
  --summary "$RESULTS_ROOT/NASDAQ/linear_regression/${COMPLETED_TAG}_NASDAQ_linear_regression/summary_all_runs.json" \
  --output "$EFFICIENCY_ROOT/NASDAQ/linear_regression.json" \
  --workers 16 --latency_repeats 20 --warmup_repeats 2

python -m baselines.simple_tabular.profile \
  --summary "$RESULTS_ROOT/NASDAQ/ridge/${COMPLETED_TAG}_NASDAQ_ridge/summary_all_runs.json" \
  --output "$EFFICIENCY_ROOT/NASDAQ/ridge.json" \
  --workers 16 --latency_repeats 20 --warmup_repeats 2

python -m baselines.simple_tabular.profile \
  --summary "$RESULTS_ROOT/NASDAQ/lightgbm/${RESUME_TAG}_NASDAQ_lightgbm/summary_all_runs.json" \
  --output "$EFFICIENCY_ROOT/NASDAQ/lightgbm.json" \
  --workers 16 --latency_repeats 20 --warmup_repeats 2


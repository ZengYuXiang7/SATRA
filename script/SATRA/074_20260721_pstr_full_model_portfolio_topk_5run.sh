#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


# Top K: 5, 10, 15, 20, 25; reevaluate the existing five-run NASDAQ checkpoints for the complete model with MMR.
PYTHONPATH=src python script/SATRA/tool_evaluate_pstr_portfolio_topk.py --experiment-dir results/pstr_hparam/1g_original_vs_best_5run/cumulative_best/NASDAQ/model_b_spe_drt/20260717_1323 --output-json results/pstr_full_model_portfolio_topk_5run/NASDAQ/topk_return_metrics.json --topks 5 10 15 20 25 --num-runs 5

# Top K: 5, 10, 15, 20, 25; reevaluate the existing five-run NYSE checkpoints for the complete model with MMR.
PYTHONPATH=src python script/SATRA/tool_evaluate_pstr_portfolio_topk.py --experiment-dir results/pstr_final_best_nyse_sp500_5run/NYSE/model_b_spe_drt/20260718_2009 --output-json results/pstr_full_model_portfolio_topk_5run/NYSE/topk_return_metrics.json --topks 5 10 15 20 25 --num-runs 5

# Top K: 5, 10, 15, 20, 25; reevaluate the existing five-run SP500 checkpoints for the complete model with MMR.
PYTHONPATH=src python script/SATRA/tool_evaluate_pstr_portfolio_topk.py --experiment-dir results/pstr_final_best_nyse_sp500_5run/SP500/model_b_spe_drt/20260718_2159 --output-json results/pstr_full_model_portfolio_topk_5run/SP500/topk_return_metrics.json --topks 5 10 15 20 25 --num-runs 5

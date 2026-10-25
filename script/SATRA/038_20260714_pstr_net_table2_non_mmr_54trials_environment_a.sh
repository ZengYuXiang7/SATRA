#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


# An execution mirror may contain an older root-level package.
# Do not let the script directory shadow this checkout's ``src/satra`` package.
export PYTHONSAFEPATH=1
export PATH="${HOME}/.local/bin:${PATH}"

# The named source files currently provide 8 independent non-MMR model states
# per market.  Once Panel (c) No prior constraint becomes an unmasked DRT,
# it will provide the ninth state required for 54 two-seed trials.
# Panel (a) SPE and panel (b) SPE only share one temporal-only SPE run; panel
# (b) DRT and panel (c) DTW-based prior share full PSTR-Net; panel (c) No
# prior constraint currently shares Semantic tower only.  This is not yet the
# current Table 2 experiment contract.
# The current Table 2 header says Panel (c) is "within DRT". Its No prior
# constraint row needs redesign as an unmasked two-tower DRT before launch.
# MMR is deliberately excluded: its decoder, loss, and pretraining workflow are
# not implemented for PSTR-Net yet.
# Each --model identifier is directly registered in src/satra/exp/exp_main.py:
# the full model is src/satra/b_spe_drt.py and every non-full row is its own
# self-contained module in src/satra/ablation/. No dynamic import wrapper is used.
NUM_RUNS=2

# Keep this tanh-controlled formal batch separate from the earlier interrupted
# batch, whose prediction head did not yet consume --output_activation.
RESULTS=results/pstr_net_table2_non_mmr_tanh_54trials
CHECKPOINTS="${RESULTS}/checkpoints"
DATA_ROOT=data/processed

# Cross-stock attention is O(N^2).  Batch size 1 is intentional so the NYSE
# 1,737-stock relation towers fit reliably on the environment_a.
COMMON_ARGS=(
  --num_runs "${NUM_RUNS}"
  --debug 0
  --is_training 1
  --root_path "${DATA_ROOT}"
  --results "${RESULTS}"
  --checkpoints "${CHECKPOINTS}"
  --input_features all
  --use_grouped_inputs 0
  --seq_len 16
  --batch_size 1
  --num_workers 0
  --use_gpu 1
  --pin_memory 1
  --train_drop_last 1
  --valid_drop_last 1
  --prediction_target return
  --target_mode raw_return
  --output_activation tanh
  --learning_rate 0.001
  --train_epochs 100
  --patience 20
  --early_stop_metric RankIC
  --early_stop_mode max
  --lr_scheduler cosine_hard_restarts
  --precision bf16_amp_tf32
  --torch_compile 0
  --max_grad_norm 1.0
  --d_model 64
  --pstr_depth 10
  --pstr_kernel_size 3
  --pstr_num_heads 4
  --pstr_dropout 0.1
  --moe_top_k 3
  --stock_prior none
)

DTW_NASDAQ="${DATA_ROOT}/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz"
INDUSTRY_NASDAQ="${DATA_ROOT}/NASDAQ_ohlcv/priors/industry_binary_NASDAQ.npz"
EUCLIDEAN_NASDAQ="${DATA_ROOT}/NASDAQ_ohlcv/priors/euclidean_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_top32.npz"
DTW_NYSE="${DATA_ROOT}/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz"
INDUSTRY_NYSE="${DATA_ROOT}/NYSE_ohlcv/priors/industry_binary_NYSE.npz"
EUCLIDEAN_NYSE="${DATA_ROOT}/NYSE_ohlcv/priors/euclidean_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_top32.npz"
DTW_SP500="${DATA_ROOT}/SP500_ohlcv/priors/dtw_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_band6_cand48_top32.npz"
INDUSTRY_SP500="${DATA_ROOT}/SP500_ohlcv/priors/industry_binary_SP500.npz"
EUCLIDEAN_SP500="${DATA_ROOT}/SP500_ohlcv/priors/euclidean_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_top32.npz"

# NASDAQ: (a) temporal specialization, then (b) relation towers, then (c) priors.
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_shared_temporal_encoder_only_nasdaq_2run --model model_a_shared_temporal_encoder --data NASDAQ --market_num 20 --valid_index 871 --test_index 996 --data_start_index 0
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_homogeneous_experts_nasdaq_2run --model model_a_homogeneous_experts --data NASDAQ --market_num 20 --valid_index 871 --test_index 996 --data_start_index 0
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_spe_nasdaq_2run --model model_a_spe --data NASDAQ --market_num 20 --valid_index 871 --test_index 996 --data_start_index 0
# SPE only reuses pstr_table2_spe_nasdaq_2run.
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_semantic_tower_only_nasdaq_2run --model model_b_spe_semantic_tower --data NASDAQ --market_num 20 --valid_index 871 --test_index 996 --data_start_index 0
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_prior_guided_tower_only_nasdaq_2run --model model_b_spe_prior_guided_tower --data NASDAQ --market_num 20 --valid_index 871 --test_index 996 --data_start_index 0 --pstr_relation_path "${DTW_NASDAQ}"
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_drt_nasdaq_2run --model model_b_spe_drt --data NASDAQ --market_num 20 --valid_index 871 --test_index 996 --data_start_index 0 --pstr_relation_path "${DTW_NASDAQ}"
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_industry_relation_prior_nasdaq_2run --model model_c_industry_relation_prior --data NASDAQ --market_num 20 --valid_index 871 --test_index 996 --data_start_index 0 --pstr_relation_path "${INDUSTRY_NASDAQ}"
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_euclidean_distance_prior_nasdaq_2run --model model_c_euclidean_distance_prior --data NASDAQ --market_num 20 --valid_index 871 --test_index 996 --data_start_index 0 --pstr_relation_path "${EUCLIDEAN_NASDAQ}"

# NYSE: same Table 2 order and the same raw training interval [0, 871).
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_shared_temporal_encoder_only_nyse_2run --model model_a_shared_temporal_encoder --data NYSE --market_num 25 --valid_index 871 --test_index 996 --data_start_index 0
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_homogeneous_experts_nyse_2run --model model_a_homogeneous_experts --data NYSE --market_num 25 --valid_index 871 --test_index 996 --data_start_index 0
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_spe_nyse_2run --model model_a_spe --data NYSE --market_num 25 --valid_index 871 --test_index 996 --data_start_index 0
# SPE only reuses pstr_table2_spe_nyse_2run.
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_semantic_tower_only_nyse_2run --model model_b_spe_semantic_tower --data NYSE --market_num 25 --valid_index 871 --test_index 996 --data_start_index 0
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_prior_guided_tower_only_nyse_2run --model model_b_spe_prior_guided_tower --data NYSE --market_num 25 --valid_index 871 --test_index 996 --data_start_index 0 --pstr_relation_path "${DTW_NYSE}"
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_drt_nyse_2run --model model_b_spe_drt --data NYSE --market_num 25 --valid_index 871 --test_index 996 --data_start_index 0 --pstr_relation_path "${DTW_NYSE}"
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_industry_relation_prior_nyse_2run --model model_c_industry_relation_prior --data NYSE --market_num 25 --valid_index 871 --test_index 996 --data_start_index 0 --pstr_relation_path "${INDUSTRY_NYSE}"
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_euclidean_distance_prior_nyse_2run --model model_c_euclidean_distance_prior --data NYSE --market_num 25 --valid_index 871 --test_index 996 --data_start_index 0 --pstr_relation_path "${EUCLIDEAN_NYSE}"

# S&P 500: the cropped training tensor starts at raw date 915, so its exact
# relation construction interval is [915, 2042), not [0, 1127).
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_shared_temporal_encoder_only_sp500_2run --model model_a_shared_temporal_encoder --data SP500 --market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_homogeneous_experts_sp500_2run --model model_a_homogeneous_experts --data SP500 --market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_spe_sp500_2run --model model_a_spe --data SP500 --market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915
# SPE only reuses pstr_table2_spe_sp500_2run.
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_semantic_tower_only_sp500_2run --model model_b_spe_semantic_tower --data SP500 --market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_prior_guided_tower_only_sp500_2run --model model_b_spe_prior_guided_tower --data SP500 --market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915 --pstr_relation_path "${DTW_SP500}"
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_drt_sp500_2run --model model_b_spe_drt --data SP500 --market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915 --pstr_relation_path "${DTW_SP500}"
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_industry_relation_prior_sp500_2run --model model_c_industry_relation_prior --data SP500 --market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915 --pstr_relation_path "${INDUSTRY_SP500}"
PYTHONPATH=src python run.py "${COMMON_ARGS[@]}" --model_id pstr_table2_euclidean_distance_prior_sp500_2run --model model_c_euclidean_distance_prior --data SP500 --market_num 8 --valid_index 1127 --test_index 1288 --data_start_index 915 --pstr_relation_path "${EUCLIDEAN_SP500}"

echo "[pstr-table2] interim named plan complete: eight current states per market"

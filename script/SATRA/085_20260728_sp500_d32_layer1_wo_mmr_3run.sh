#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

project_root="${PACKAGE_PROJECT_ROOT}"
cd "$project_root"

exec env PYTHONPATH=src python -c \
  'import sys; sys.path.pop(0); from satra.main import main; main()' \
  --is_training 1 \
  --model_id astra_sp500_d32_layers1_wo_mmr_3run \
  --model model_b_spe_drt \
  --num_runs 3 \
  --random_seed 1 \
  --evaluate_test 0 \
  --portfolio_size 20 \
  --data SP500 \
  --root_path data/processed \
  --results results/astra_sp500_no_mmr_d32_layers1_3run_20260728 \
  --checkpoints results/astra_sp500_no_mmr_d32_layers1_3run_20260728/checkpoints \
  --seq_len 16 \
  --steps 1 \
  --input_features all \
  --use_grouped_inputs 0 \
  --batch_size 1 \
  --num_workers 0 \
  --pin_memory 1 \
  --train_drop_last 1 \
  --valid_drop_last 1 \
  --market_num 8 \
  --scale_factor 3 \
  --alpha 0.1 \
  --output_scale 0.2 \
  --output_activation tanh \
  --target_mode raw_return \
  --prediction_target return \
  --stock_prior dtw \
  --stock_prior_topk 16 \
  --stock_prior_candidate_k 48 \
  --stock_prior_downsample 64 \
  --stock_prior_band 6 \
  --adaptive_graph_topk 32 \
  --moe_top_k 3 \
  --moe_load_balance_weight 0.0 \
  --d_model 32 \
  --pstr_depth 10 \
  --pstr_kernel_size 3 \
  --pstr_num_heads 4 \
  --pstr_tower_layers 1 \
  --pstr_dropout 0.1 \
  --semantic_support global \
  --pstr_relation_path data/processed/SP500_ohlcv/priors/dtw_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_band6_cand48_top32.npz \
  --mae_mask_ratio 0.3 \
  --valid_index 1127 \
  --test_index 1288 \
  --data_start_index 915 \
  --train_epochs 50 \
  --learning_rate 0.001 \
  --patience 20 \
  --min_delta 0.0 \
  --min_epochs 0 \
  --min_updates 0 \
  --early_stop_metric RankIC \
  --early_stop_mode max \
  --lr_scheduler cosine \
  --warmup_ratio 0.1 \
  --num_cycles 1 \
  --min_ratio 0.01 \
  --min_lr 0.000001 \
  --use_gpu 1 \
  --gpu 0 \
  --precision bf16_amp_tf32 \
  --torch_compile 0 \
  --max_grad_norm 1.0

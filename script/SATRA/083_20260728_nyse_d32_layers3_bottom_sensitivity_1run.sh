#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


# NYSE supplementary sensitivity sweep requested on 2026-07-28.
# Fixed base: d_model=32, DRT layers=3, MMR mask ratio=0.3.
# Each candidate uses the same single seed (20260721). The checkpoint is selected
# by validation RankIC; test metrics are emitted only to populate the descriptive
# supplementary curves and must not be used to choose the final configuration.

formal_prior="data/processed/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz"
dtw_prior_root="data/processed/NYSE_ohlcv/priors"

test -f "$formal_prior"
test -f "$dtw_prior_root/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top8.npz"
test -f "$dtw_prior_root/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top16.npz"
test -f "$dtw_prior_root/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top32.npz"
test -f "$dtw_prior_root/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top64.npz"

# Panel (j): ACE routing Top-k. Fix DTW Top-32 and lambda_lb=1e-3.
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_ace_topk1_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/ace_topk/k1 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/ace_topk/k1/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 1 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path "$formal_prior" --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_ace_topk2_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/ace_topk/k2 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/ace_topk/k2/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 2 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path "$formal_prior" --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_ace_topk3_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/ace_topk/k3 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/ace_topk/k3/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path "$formal_prior" --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_ace_topk6_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/ace_topk/k6 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/ace_topk/k6/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 6 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path "$formal_prior" --valid_index 871 --test_index 996 --data_start_index 0

# Panel (k): DTW neighborhood size. All candidates use one common candidate-96 screen.
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_dtw_neighbors8_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/dtw_neighbors/k8 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/dtw_neighbors/k8/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path "$dtw_prior_root/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top8.npz" \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_dtw_neighbors16_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/dtw_neighbors/k16 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/dtw_neighbors/k16/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path "$dtw_prior_root/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top16.npz" \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_dtw_neighbors32_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/dtw_neighbors/k32 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/dtw_neighbors/k32/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path "$dtw_prior_root/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top32.npz" \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_dtw_neighbors64_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/dtw_neighbors/k64 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/dtw_neighbors/k64/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path "$dtw_prior_root/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand96_top64.npz" \
  --valid_index 871 --test_index 996 --data_start_index 0

# Panel (l): supervised ACE load-balancing coefficient. Fix Top-3 routing and formal DTW Top-32.
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_balance0_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/balance_weight/lb0 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/balance_weight/lb0/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 0 \
  --pstr_relation_path "$formal_prior" --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_balance1e-4_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/balance_weight/lb1e-4 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/balance_weight/lb1e-4/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 1e-4 \
  --pstr_relation_path "$formal_prior" --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_balance1e-3_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/balance_weight/lb1e-3 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/balance_weight/lb1e-3/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path "$formal_prior" --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id astra_nyse_d32_l3_balance1e-2_1run --model model_b_spe_drt \
  --num_runs 1 --random_seed 20260721 --evaluate_test 1 --portfolio_size 20 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/balance_weight/lb1e-2 \
  --checkpoints results/astra_nyse_d32_layers3_bottom_sensitivity_1run_20260728/balance_weight/lb1e-2/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max \
  --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 1e-2 \
  --pstr_relation_path "$formal_prior" --valid_index 871 --test_index 996 --data_start_index 0

#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


# NASDAQ supplementary sensitivity sweep.  Each explicit command executes five
# paired seeds (20260716--20260720) using the same MMR and supervised protocol.
# The controlled cand96 DTW priors are prepared by
# build_dtw_sensitivity_priors.py before this training script is launched.
test -f data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand96_top8.npz
test -f data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand96_top16.npz
test -f data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand96_top32.npz
test -f data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand96_top64.npz

# Panel (d): ACE routing Top-k, with the formal Top-32 DTW prior and lambda_lb=1e-3.
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_ace_topk1_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/ace_topk/k1 --checkpoints results/patra_sensitivity_20260721/ace_topk/k1/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 1 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_ace_topk2_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/ace_topk/k2 --checkpoints results/patra_sensitivity_20260721/ace_topk/k2/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 2 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_ace_topk3_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/ace_topk/k3 --checkpoints results/patra_sensitivity_20260721/ace_topk/k3/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_ace_topk6_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/ace_topk/k6 --checkpoints results/patra_sensitivity_20260721/ace_topk/k6/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 6 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

# Panel (e): DTW neighborhood size, with one fixed 96-stock candidate screen and formal ACE settings.
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_dtw_neighbors8_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/dtw_neighbors/k8 --checkpoints results/patra_sensitivity_20260721/dtw_neighbors/k8/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand96_top8.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_dtw_neighbors16_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/dtw_neighbors/k16 --checkpoints results/patra_sensitivity_20260721/dtw_neighbors/k16/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand96_top16.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_dtw_neighbors32_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/dtw_neighbors/k32 --checkpoints results/patra_sensitivity_20260721/dtw_neighbors/k32/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand96_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_dtw_neighbors64_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/dtw_neighbors/k64 --checkpoints results/patra_sensitivity_20260721/dtw_neighbors/k64/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand96_top64.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

# Panel (f): ACE load-balancing coefficient, with the formal Top-32 prior and Top-3 routing.
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_balance0_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/balance_weight/lb0 --checkpoints results/patra_sensitivity_20260721/balance_weight/lb0/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 0 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_balance1e-4_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/balance_weight/lb1e-4 --checkpoints results/patra_sensitivity_20260721/balance_weight/lb1e-4/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 1e-4 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_balance1e-3_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/balance_weight/lb1e-3 --checkpoints results/patra_sensitivity_20260721/balance_weight/lb1e-3/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 1e-3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id sensitivity_balance1e-2_nasdaq --model model_b_spe_drt --num_runs 5 --random_seed 20260716 \
  --data NASDAQ --market_num 20 --root_path data/processed \
  --results results/patra_sensitivity_20260721/balance_weight/lb1e-2 --checkpoints results/patra_sensitivity_20260721/balance_weight/lb1e-2/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 1e-2 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

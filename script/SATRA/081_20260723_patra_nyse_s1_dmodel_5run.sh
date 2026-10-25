#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


# Stage 1 of the NYSE sequential sensitivity screen.
# Only d_model varies. All four candidates share the paired seeds
# 20260721--20260725 and the frozen MMR Full model contract.
# The next stage is generated only after selecting the highest five-run mean
# validation RankIC from the completed structured summaries.

command -v python >/dev/null
test -f data/processed/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz

# hparam: d_model=16; NYSE paired 5-run evaluation
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id hparam_nyse_s1_dmodel_d16_5run --model model_b_spe_drt --num_runs 5 --random_seed 20260721 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/patra_hparam_nyse_sp500_20260723/NYSE/s1_d_model/d16 \
  --checkpoints results/patra_hparam_nyse_sp500_20260723/NYSE/s1_d_model/d16/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 16 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 0.0 \
  --pstr_relation_path data/processed/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

# hparam: d_model=32; NYSE paired 5-run evaluation
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id hparam_nyse_s1_dmodel_d32_5run --model model_b_spe_drt --num_runs 5 --random_seed 20260721 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/patra_hparam_nyse_sp500_20260723/NYSE/s1_d_model/d32 \
  --checkpoints results/patra_hparam_nyse_sp500_20260723/NYSE/s1_d_model/d32/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 0.0 \
  --pstr_relation_path data/processed/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

# hparam: d_model=64; NYSE paired 5-run evaluation
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id hparam_nyse_s1_dmodel_d64_5run --model model_b_spe_drt --num_runs 5 --random_seed 20260721 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/patra_hparam_nyse_sp500_20260723/NYSE/s1_d_model/d64 \
  --checkpoints results/patra_hparam_nyse_sp500_20260723/NYSE/s1_d_model/d64/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 0.0 \
  --pstr_relation_path data/processed/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

# hparam: d_model=128; NYSE paired 5-run evaluation
env PYTHONPATH=src python -c 'import sys; sys.path.pop(0); from satra.mmr_two_stage import main; main()' \
  --is_training 1 --model_id hparam_nyse_s1_dmodel_d128_5run --model model_b_spe_drt --num_runs 5 --random_seed 20260721 \
  --data NYSE --market_num 25 --root_path data/processed \
  --results results/patra_hparam_nyse_sp500_20260723/NYSE/s1_d_model/d128 \
  --checkpoints results/patra_hparam_nyse_sp500_20260723/NYSE/s1_d_model/d128/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 128 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 \
  --moe_top_k 3 --moe_load_balance_weight 0.0 \
  --pstr_relation_path data/processed/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0

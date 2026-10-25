#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

SATRA_PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
while [ ! -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] && [ "${SATRA_PROJECT_ROOT}" != "/" ]; do SATRA_PROJECT_ROOT="$(dirname "${SATRA_PROJECT_ROOT}")"; done
[ -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] || { echo "Unable to locate the project root." >&2; exit 1; }
cd "${SATRA_PROJECT_ROOT}"

# Stage 1a: search d_model; DTW Top-K remains 32.
# Run each candidate three times on NASDAQ.
# CHANGE: After this stage, copy the best d_model to BEST_D_MODEL in the Stage 1b script.

# hparam: d_model=16 market=NASDAQ
PYTHONPATH=src python -m satra.mmr_two_stage \
  --is_training 1 --model_id 1a_d_model_d16_nasdaq \
  --model model_b_spe_drt --data NASDAQ --market_num 20 \
  --root_path data/processed \
  --results results/pstr_hparam/1a_d_model/d16 --checkpoints results/pstr_hparam/1a_d_model/d16/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 100 --patience 20 \
  --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 16 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 1 --pstr_dropout 0.1 --moe_top_k 3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 3

# hparam: d_model=32 market=NASDAQ
PYTHONPATH=src python -m satra.mmr_two_stage \
  --is_training 1 --model_id 1a_d_model_d32_nasdaq \
  --model model_b_spe_drt --data NASDAQ --market_num 20 \
  --root_path data/processed \
  --results results/pstr_hparam/1a_d_model/d32 --checkpoints results/pstr_hparam/1a_d_model/d32/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 100 --patience 20 \
  --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 1 --pstr_dropout 0.1 --moe_top_k 3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 3

# hparam: d_model=64 market=NASDAQ
PYTHONPATH=src python -m satra.mmr_two_stage \
  --is_training 1 --model_id 1a_d_model_d64_nasdaq \
  --model model_b_spe_drt --data NASDAQ --market_num 20 \
  --root_path data/processed \
  --results results/pstr_hparam/1a_d_model/d64 --checkpoints results/pstr_hparam/1a_d_model/d64/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 100 --patience 20 \
  --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 1 --pstr_dropout 0.1 --moe_top_k 3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 3

# hparam: d_model=128 market=NASDAQ
PYTHONPATH=src python -m satra.mmr_two_stage \
  --is_training 1 --model_id 1a_d_model_d128_nasdaq \
  --model model_b_spe_drt --data NASDAQ --market_num 20 \
  --root_path data/processed \
  --results results/pstr_hparam/1a_d_model/d128 --checkpoints results/pstr_hparam/1a_d_model/d128/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 \
  --learning_rate 0.001 --train_epochs 100 --patience 20 \
  --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model 128 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers 1 --pstr_dropout 0.1 --moe_top_k 3 \
  --pstr_relation_path data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 3

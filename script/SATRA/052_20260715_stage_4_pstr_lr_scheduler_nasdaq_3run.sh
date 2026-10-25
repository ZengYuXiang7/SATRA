#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


# Stage 4: compare validation RankIC driven Plateau and epoch driven Cosine during supervised fine-tuning.
# Run each candidate three times on NASDAQ.
# MMR pretraining keeps the cosine scheduler fixed; only the supervised fine-tuning scheduler changes here.
# CHANGE: Set BEST_MASK_RATIO to the best value selected in Stage 2.
# CHANGE: Set BEST_TOWER_LAYERS to the best value selected in Stage 3.

# CHANGE: Set BEST_D_MODEL to the best value selected in Stage 1a.
BEST_D_MODEL=64
# CHANGE: Set DTW_NASDAQ to the best Top-K relation selected in Stage 1b.
DTW_NASDAQ=data/processed/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz

BEST_MASK_RATIO=0.3
BEST_TOWER_LAYERS=1

# hparam: fine-tuning lr_scheduler=plateau market=NASDAQ
PYTHONPATH=src python -m satra.mmr_two_stage \
  --is_training 1 --model_id 4_lr_scheduler_plateau_nasdaq \
  --model model_b_spe_drt --data NASDAQ --market_num 20 \
  --root_path data/processed \
  --results results/pstr_hparam/4_lr_scheduler/plateau --checkpoints results/pstr_hparam/4_lr_scheduler/plateau/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio ${BEST_MASK_RATIO} \
  --learning_rate 0.001 --train_epochs 100 --patience 20 \
  --early_stop_metric RankIC --early_stop_mode max --lr_scheduler plateau \
  --lr_factor 0.5 --lr_patience 2 --min_lr 0.000001 \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model ${BEST_D_MODEL} --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers ${BEST_TOWER_LAYERS} --pstr_dropout 0.1 --moe_top_k 3 \
  --pstr_relation_path ${DTW_NASDAQ} \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 3

# hparam: fine-tuning lr_scheduler=cosine market=NASDAQ
PYTHONPATH=src python -m satra.mmr_two_stage \
  --is_training 1 --model_id 4_lr_scheduler_cosine_nasdaq \
  --model model_b_spe_drt --data NASDAQ --market_num 20 \
  --root_path data/processed \
  --results results/pstr_hparam/4_lr_scheduler/cosine --checkpoints results/pstr_hparam/4_lr_scheduler/cosine/checkpoints \
  --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 \
  --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 \
  --prediction_target return --target_mode raw_return --output_activation tanh \
  --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio ${BEST_MASK_RATIO} \
  --learning_rate 0.001 --train_epochs 100 --patience 20 \
  --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine \
  --lr_factor 0.5 --lr_patience 2 --min_lr 0.000001 \
  --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 \
  --d_model ${BEST_D_MODEL} --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 \
  --pstr_tower_layers ${BEST_TOWER_LAYERS} --pstr_dropout 0.1 --moe_top_k 3 \
  --pstr_relation_path ${DTW_NASDAQ} \
  --valid_index 871 --test_index 996 --data_start_index 0 --num_runs 3


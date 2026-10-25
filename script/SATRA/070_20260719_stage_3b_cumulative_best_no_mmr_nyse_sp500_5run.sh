#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


# Resume the exact no-MMR cumulative-best protocol after the paired NYSE seed collapsed.
# Only the reproducible seed block changes to the user-approved independent seeds 1-5.
# All model, optimizer, scheduler, data, target, split, and fixed-structure values stay unchanged.
export PYTHONSAFEPATH=1

# NYSE recovery: seeds=1-5; immediate cosine schedule is retained for a strict MMR ablation.
PYTHONPATH=src python run.py --is_training 1 --model_id 3b_cumulative_best_no_mmr_5run_nyse --model model_b_spe_drt --num_runs 5 --random_seed 1 --data NYSE --market_num 25 --root_path data/processed --results results/pstr_final_best_no_mmr_nyse_sp500_5run_recovery --checkpoints results/pstr_final_best_no_mmr_nyse_sp500_5run_recovery/checkpoints --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 --prediction_target return --target_mode raw_return --output_activation tanh --pretrained_encoder_path "" --pretrained_backbone_path "" --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 --moe_load_balance_weight 0.0 --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --pstr_relation_path data/processed/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz --valid_index 871 --test_index 996 --data_start_index 0

# SP500 recovery: seeds=1-5; cropped start and raw-time-aligned DTW prior are preserved.
PYTHONPATH=src python run.py --is_training 1 --model_id 3b_cumulative_best_no_mmr_5run_sp500 --model model_b_spe_drt --num_runs 5 --random_seed 1 --data SP500 --market_num 8 --root_path data/processed --results results/pstr_final_best_no_mmr_nyse_sp500_5run_recovery --checkpoints results/pstr_final_best_no_mmr_nyse_sp500_5run_recovery/checkpoints --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 1 --num_workers 0 --use_gpu 1 --pin_memory 1 --train_drop_last 1 --valid_drop_last 1 --prediction_target return --target_mode raw_return --output_activation tanh --pretrained_encoder_path "" --pretrained_backbone_path "" --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 --moe_load_balance_weight 0.0 --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 3 --pstr_relation_path data/processed/SP500_ohlcv/priors/dtw_binary_imputed_inputstandard_SP500_train915-2042_close_ds64_band6_cand48_top32.npz --valid_index 1127 --test_index 1288 --data_start_index 915

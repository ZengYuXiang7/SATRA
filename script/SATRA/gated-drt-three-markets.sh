#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export UV_NO_PROGRESS=1

SATRA_DATA_ROOT="data/processed"
SATRA_RESULTS="results/gated-drt"
mkdir -p "${SATRA_RESULTS}"

# 三个市场各五次，种子为 20260716 至 20260720，按市场串行执行。
# 路由与两塔共同读取 ATPE 输出 H；两个训练阶段均加入 1e-3 的 DRT 均衡损失。

# NASDAQ：MMR 预训练后进行端到端监督微调。
python -P -m satra.mmr_two_stage --is_training 1 --model_id satra_pre_tower_routing_nasdaq_bs4_5run --model model_b_spe_drt_feature_aggregation --num_runs 5 --random_seed 20260716 --data NASDAQ --market_num 20 --root_path "${SATRA_DATA_ROOT}" --results "${SATRA_RESULTS}" --checkpoints "${SATRA_RESULTS}/checkpoints" --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 4 --num_workers 0 --use_gpu 1 --gpu 0 --pin_memory 1 --train_drop_last 0 --valid_drop_last 0 --prediction_target return --target_mode raw_return --output_activation tanh --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.3 --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 --d_model 64 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 3 --pstr_dropout 0.1 --moe_top_k 2 --moe_load_balance_weight 0.01 --semantic_support global --pstr_relation_path "${SATRA_DATA_ROOT}/NASDAQ_ohlcv/priors/dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top8.npz" --alpha 0.01 --evaluate_test 1 --portfolio_size 20 --valid_index 871 --test_index 996 --data_start_index 0 --drt_load_balance_weight 0.001

# NYSE：MMR 预训练后进行端到端监督微调。
python -P -m satra.mmr_two_stage --is_training 1 --model_id satra_pre_tower_routing_nyse_bs4_5run --model model_b_spe_drt_feature_aggregation --num_runs 5 --random_seed 20260716 --data NYSE --market_num 25 --root_path "${SATRA_DATA_ROOT}" --results "${SATRA_RESULTS}" --checkpoints "${SATRA_RESULTS}/checkpoints" --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 4 --num_workers 0 --use_gpu 1 --gpu 0 --pin_memory 1 --train_drop_last 0 --valid_drop_last 0 --prediction_target return --target_mode raw_return --output_activation tanh --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.1 --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 --d_model 32 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 2 --pstr_dropout 0.1 --moe_top_k 3 --moe_load_balance_weight 0.001 --semantic_support global --pstr_relation_path "${SATRA_DATA_ROOT}/NYSE_ohlcv/priors/dtw_binary_imputed_inputstandard_NYSE_train0-871_close_ds64_band6_cand48_top32.npz" --alpha 0.001 --evaluate_test 1 --portfolio_size 20 --valid_index 871 --test_index 996 --data_start_index 0 --drt_load_balance_weight 0.001

# SP500：MMR 预训练后进行端到端监督微调。
python -P -m satra.mmr_two_stage --is_training 1 --model_id satra_pre_tower_routing_sp500_bs4_5run --model model_b_spe_drt_feature_aggregation --num_runs 5 --random_seed 20260716 --data SP500 --market_num 8 --root_path "${SATRA_DATA_ROOT}" --results "${SATRA_RESULTS}" --checkpoints "${SATRA_RESULTS}/checkpoints" --input_features all --use_grouped_inputs 0 --seq_len 16 --batch_size 4 --num_workers 0 --use_gpu 1 --gpu 0 --pin_memory 1 --train_drop_last 0 --valid_drop_last 0 --prediction_target return --target_mode raw_return --output_activation tanh --pretrain_epochs 20 --pretrain_learning_rate 0.001 --pretrain_lr_scheduler cosine --mae_mask_ratio 0.2 --learning_rate 0.001 --train_epochs 50 --patience 20 --early_stop_metric RankIC --early_stop_mode max --lr_scheduler cosine --precision bf16_amp_tf32 --torch_compile 0 --max_grad_norm 1.0 --d_model 16 --pstr_depth 10 --pstr_kernel_size 3 --pstr_num_heads 4 --pstr_tower_layers 1 --pstr_dropout 0.1 --moe_top_k 2 --moe_load_balance_weight 0.001 --semantic_support global --pstr_relation_path "${SATRA_DATA_ROOT}/SP500_ohlcv/priors/sp500-dtw-top16.npz" --alpha 0.1 --evaluate_test 1 --portfolio_size 20 --valid_index 1127 --test_index 1288 --data_start_index 915 --drt_load_balance_weight 0.001

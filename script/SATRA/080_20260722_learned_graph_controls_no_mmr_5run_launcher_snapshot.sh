#!/usr/bin/env bash
set -euo pipefail

PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

# Two learned-relation controls only: static learned adjacency and a
# state-adaptive sparse graph.  They deliberately use the supervised run.py
# entrypoint with no pretrained encoder/backbone, so MMR is absent.

SATRA_PROJECT_ROOT="${PACKAGE_PROJECT_ROOT}"
while [ ! -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] && [ "${SATRA_PROJECT_ROOT}" != "/" ]; do SATRA_PROJECT_ROOT="$(dirname "${SATRA_PROJECT_ROOT}")"; done
[ -f "${SATRA_PROJECT_ROOT}/pyproject.toml" ] || { echo "Unable to locate the project root." >&2; exit 1; }
cd "${SATRA_PROJECT_ROOT}"

export PYTHONSAFEPATH=1

DATA_ROOT=data/processed
RESULTS=results/pstr_learned_graph_controls_no_mmr_5run
CHECKPOINTS="${RESULTS}/checkpoints"

run_case() {
  local model="$1"
  local variant="$2"
  local dataset="$3"
  local market_num="$4"
  local valid_index="$5"
  local test_index="$6"
  local data_start_index="$7"
  local random_seed="$8"

  echo "### START model=${model} variant=${variant} dataset=${dataset} runs=5 mmr=off"
  env -u PYTHONPATH python run.py \
    --is_training 1 \
    --model_id "learned_graph_${variant}_${dataset,,}_5run" \
    --model "${model}" \
    --num_runs 5 \
    --random_seed "${random_seed}" \
    --data "${dataset}" \
    --market_num "${market_num}" \
    --root_path "${DATA_ROOT}" \
    --results "${RESULTS}" \
    --checkpoints "${CHECKPOINTS}" \
    --input_features all \
    --use_grouped_inputs 0 \
    --seq_len 16 \
    --batch_size 1 \
    --num_workers 0 \
    --use_gpu 1 \
    --pin_memory 1 \
    --train_drop_last 1 \
    --valid_drop_last 1 \
    --prediction_target return \
    --target_mode raw_return \
    --output_activation tanh \
    --pretrained_encoder_path "" \
    --pretrained_backbone_path "" \
    --learning_rate 0.001 \
    --train_epochs 100 \
    --patience 20 \
    --early_stop_metric RankIC \
    --early_stop_mode max \
    --lr_scheduler cosine_hard_restarts \
    --precision bf16_amp_tf32 \
    --torch_compile 0 \
    --max_grad_norm 1.0 \
    --d_model 64 \
    --pstr_depth 10 \
    --pstr_kernel_size 3 \
    --pstr_num_heads 4 \
    --pstr_tower_layers 1 \
    --pstr_dropout 0.1 \
    --moe_top_k 3 \
    --moe_load_balance_weight 0.0 \
    --adaptive_graph_topk 32 \
    --semantic_support global \
    --stock_prior none \
    --valid_index "${valid_index}" \
    --test_index "${test_index}" \
    --data_start_index "${data_start_index}"
}

# Dataset-by-dataset order makes the two controls directly comparable as soon
# as each market finishes.
run_case model_c_learned_adjacency learned_adjacency NASDAQ 20 871 996 0 151567275
run_case model_c_adaptive_graph adaptive_graph NASDAQ 20 871 996 0 151567275
run_case model_c_learned_adjacency learned_adjacency NYSE 25 871 996 0 701943994
run_case model_c_adaptive_graph adaptive_graph NYSE 25 871 996 0 701943994
run_case model_c_learned_adjacency learned_adjacency SP500 8 1127 1288 915 761671755
run_case model_c_adaptive_graph adaptive_graph SP500 8 1127 1288 915 761671755

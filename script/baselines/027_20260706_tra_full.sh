#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


python -m tra.run --data NASDAQ --root_path data/processed --relation_root data/processed/relations --relation_name auto --results results --input_features stock_all --num_runs 5 --seq_len 16 --steps 1 --batch_size 8 --train_epochs 100 --learning_rate 1e-3 --hidden_size 64 --num_layers 1 --num_predictors 5 --router_hidden 64 --dropout 0.1 --alpha 0.1 --ot_weight 0.1 --use_gpu 1 --gpu 0

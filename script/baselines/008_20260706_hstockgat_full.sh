#!/bin/bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


python -m hstockgat.run --data NASDAQ --root_path data/processed --relation_root data/processed/relations --relation_name auto --results results --seq_len 16 --steps 1 --input_features stock_all --num_runs 5 --train_epochs 100 --batch_size 1 --learning_rate 4e-4 --weight_decay 1e-4 --hidden_size 16 --num_layers 2 --dropout 0.4 --curvature 1.0 --use_attention 1 --alpha 2.0 --use_gpu 1 --gpu 0

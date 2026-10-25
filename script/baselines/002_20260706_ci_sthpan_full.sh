#!/bin/bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


python -m ci_sthpan.run --data NASDAQ --root_path data/processed --relation_root data/processed/relations --hypergraph_name auto --results results --seq_len 16 --steps 1 --input_features stock_all --num_runs 5 --train_epochs 100 --batch_size 1 --learning_rate 1e-4 --patch_len 12 --stride 12 --d_model 128 --n_heads 16 --n_layers 3 --d_ff 256 --dropout 0.2 --alpha 2.0 --use_gpu 1 --gpu 0

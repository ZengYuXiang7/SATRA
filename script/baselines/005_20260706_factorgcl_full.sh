#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


python -m factorgcl.run --data NASDAQ --root_path data/processed --relation_root data/processed/relations --relation_name auto --prior_beta_name factor_prior_beta --results results --input_features stock_all --num_runs 5 --seq_len 16 --future_seq_len 16 --steps 1 --batch_size 1 --train_epochs 100 --learning_rate 1e-3 --hidden_dim 32 --num_hidden_factors 32 --num_gru_layers 2 --dropout 0.1 --alpha 0.1 --gamma 0.1 --use_gpu 1

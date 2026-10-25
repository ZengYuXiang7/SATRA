#!/bin/bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"


python -m rsr.run --data NASDAQ --root_path data/processed --relation_root data/processed/relations --relation_name auto --results results --input_features stock_all --num_runs 5 --seq_len 16 --steps 1 --batch_size 1 --train_epochs 100 --learning_rate 1e-3 --hidden_size 64 --num_layers 1 --dropout 0.0 --alpha 0.1 --patience 10 --max_grad_norm 1.0 --monitor_metric RankIC --monitor_mode max --precision bf16_amp_tf32 --torch_compile 1 --compile_mode reduce-overhead --fused_optimizer 1 --use_gpu 1 --gpu 0

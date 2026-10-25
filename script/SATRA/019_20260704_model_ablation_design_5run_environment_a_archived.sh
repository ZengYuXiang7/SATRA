#!/usr/bin/env bash
set -euo pipefail
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

FINETUNE_RUNS="${FINETUNE_RUNS:-5}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
USE_GPU="${USE_GPU:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION:-tanh}"
DATA_ROOT="${DATA_ROOT:-data/processed}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ablation_design_5run}"

echo "[ablation-design-5run] generate data_root=${DATA_ROOT} $(date +%Y%m%d_%H%M%S)"
PYTHONPATH=src "${PYTHON_BIN:-python}" -m satra.data_provider.normalize_existing_data \
  --source-root dataset \
  --output-root "${DATA_ROOT}" \
  --markets NASDAQ NYSE SP500 \
  --clean-output

echo "[ablation-design-5run] start variant=full dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"
DATASET="NASDAQ" MODEL_VARIANT="full" METHOD_NAME="model_ablation_full" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=full dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=full dataset=NYSE $(date +%Y%m%d_%H%M%S)"
DATASET="NYSE" MODEL_VARIANT="full" METHOD_NAME="model_ablation_full" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=full dataset=NYSE $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=full dataset=SP500 $(date +%Y%m%d_%H%M%S)"
DATASET="SP500" MODEL_VARIANT="full" METHOD_NAME="model_ablation_full" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=full dataset=SP500 $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=wo_relation_learning dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"
DATASET="NASDAQ" MODEL_VARIANT="wo_relation_learning" METHOD_NAME="model_ablation_wo_relation_learning" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="none" FINETUNE_STOCK_PRIOR="none" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=wo_relation_learning dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=wo_relation_learning dataset=NYSE $(date +%Y%m%d_%H%M%S)"
DATASET="NYSE" MODEL_VARIANT="wo_relation_learning" METHOD_NAME="model_ablation_wo_relation_learning" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="none" FINETUNE_STOCK_PRIOR="none" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=wo_relation_learning dataset=NYSE $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=wo_relation_learning dataset=SP500 $(date +%Y%m%d_%H%M%S)"
DATASET="SP500" MODEL_VARIANT="wo_relation_learning" METHOD_NAME="model_ablation_wo_relation_learning" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="none" FINETUNE_STOCK_PRIOR="none" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=wo_relation_learning dataset=SP500 $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=semantic_tower_only dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"
DATASET="NASDAQ" MODEL_VARIANT="semantic_tower_only" METHOD_NAME="model_ablation_semantic_tower_only" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="none" FINETUNE_STOCK_PRIOR="none" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=semantic_tower_only dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=semantic_tower_only dataset=NYSE $(date +%Y%m%d_%H%M%S)"
DATASET="NYSE" MODEL_VARIANT="semantic_tower_only" METHOD_NAME="model_ablation_semantic_tower_only" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="none" FINETUNE_STOCK_PRIOR="none" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=semantic_tower_only dataset=NYSE $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=semantic_tower_only dataset=SP500 $(date +%Y%m%d_%H%M%S)"
DATASET="SP500" MODEL_VARIANT="semantic_tower_only" METHOD_NAME="model_ablation_semantic_tower_only" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="none" FINETUNE_STOCK_PRIOR="none" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=semantic_tower_only dataset=SP500 $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=prior_tower_only dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"
DATASET="NASDAQ" MODEL_VARIANT="prior_tower_only" METHOD_NAME="model_ablation_prior_tower_only" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=prior_tower_only dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=prior_tower_only dataset=NYSE $(date +%Y%m%d_%H%M%S)"
DATASET="NYSE" MODEL_VARIANT="prior_tower_only" METHOD_NAME="model_ablation_prior_tower_only" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=prior_tower_only dataset=NYSE $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=prior_tower_only dataset=SP500 $(date +%Y%m%d_%H%M%S)"
DATASET="SP500" MODEL_VARIANT="prior_tower_only" METHOD_NAME="model_ablation_prior_tower_only" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=prior_tower_only dataset=SP500 $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=wo_mmr_init dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"
DATASET="NASDAQ" MODEL_VARIANT="wo_mmr_init" METHOD_NAME="model_ablation_wo_mmr_init" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="1" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=wo_mmr_init dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=wo_mmr_init dataset=NYSE $(date +%Y%m%d_%H%M%S)"
DATASET="NYSE" MODEL_VARIANT="wo_mmr_init" METHOD_NAME="model_ablation_wo_mmr_init" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="1" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=wo_mmr_init dataset=NYSE $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=wo_mmr_init dataset=SP500 $(date +%Y%m%d_%H%M%S)"
DATASET="SP500" MODEL_VARIANT="wo_mmr_init" METHOD_NAME="model_ablation_wo_mmr_init" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="1" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=wo_mmr_init dataset=SP500 $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=wo_temporal_experts dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"
DATASET="NASDAQ" MODEL_VARIANT="wo_temporal_experts" METHOD_NAME="model_ablation_wo_temporal_experts" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=wo_temporal_experts dataset=NASDAQ $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=wo_temporal_experts dataset=NYSE $(date +%Y%m%d_%H%M%S)"
DATASET="NYSE" MODEL_VARIANT="wo_temporal_experts" METHOD_NAME="model_ablation_wo_temporal_experts" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=wo_temporal_experts dataset=NYSE $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] start variant=wo_temporal_experts dataset=SP500 $(date +%Y%m%d_%H%M%S)"
DATASET="SP500" MODEL_VARIANT="wo_temporal_experts" METHOD_NAME="model_ablation_wo_temporal_experts" ROOT_PATH="${DATA_ROOT}" RESULTS_ROOT="${RESULTS_ROOT}" FINETUNE_RUNS="${FINETUNE_RUNS}" PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS}" FINETUNE_EPOCHS="${FINETUNE_EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PRETRAIN_TARGET_MODE="return_residual" MAE_MASK_RATIO="0.3" RET_FEATURE_INDEX="-1" PRETRAIN_STOCK_PRIOR="dtw" FINETUNE_STOCK_PRIOR="dtw" SKIP_PRETRAIN="0" USE_GPU="${USE_GPU}" PIN_MEMORY="${PIN_MEMORY}" OUTPUT_ACTIVATION="${OUTPUT_ACTIVATION}" PYTHONPATH=src "${PYTHON_BIN:-python}" -m stock.engine.model_two_stage
echo "[ablation-design-5run] done variant=wo_temporal_experts dataset=SP500 $(date +%Y%m%d_%H%M%S)"

echo "[ablation-design-5run] all done $(date +%Y%m%d_%H%M%S)"

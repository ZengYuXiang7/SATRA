#!/usr/bin/env bash
PACKAGE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PACKAGE_PROJECT_ROOT}"

# Comment-only hyperparameter preset sheet.
# Do not execute this file directly. It records paper/official-script defaults
# for later integration into the three current datasets.
#
# Current local dataset contract remembered for this project:
# - Datasets: NASDAQ_ohlcv, NYSE_ohlcv, SP500_ohlcv
# - Training pipeline usually batches as [B, stock, seq_len, feature].
# - Standalone model.py files here mostly consume one cross-section [stock, seq_len, feature].
# - Existing raw feature_dim is 24, while several papers originally use OHLCV/Alpha158/market-specific features.
#
# ==============================================================================
# MASTER: Market-Guided Stock Transformer for Stock Price Forecasting
# Source confidence: official repo + paper.
# Static source anchors:
# ==============================================================================
# MODEL=master
# model_file=src/master/model.py
# d_feat=158
# d_model=256
# t_nhead=4
# s_nhead=2
# t_dropout_rate=0.5
# s_dropout_rate=0.5
# gate_input_start_index=158
# gate_input_end_index=221
# beta=5      # CSI300-like setting in official main.py.
# beta=2      # CSI800-like larger-universe setting in official main.py.
# lookback_window=8
# prediction_interval=5
# label="Ref(close, -5) / Ref(close, -1) - 1"
# lr=1e-5
# epochs=40
# train_stop_loss_thred=0.95
# topk=30
# drop=30
#
# ==============================================================================
# StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting
# Source confidence: official repo + paper.
# Static source anchors:
# ==============================================================================
# MODEL=stockmixer
# model_file=src/stockmixer/model.py
# lookback_length=16
# fea_num=5          # Original paper/code: OHLCV-style features.
# feature_dim=24     # Local dataset candidate; adapter should map or select features explicitly.
# market_num=20
# scale_factor=3
# scale_dim=8        # Official src/model.py hard-codes scale_dim=8 for lookback_length=16.
# steps=1
# lr=1e-3
# epochs=100
# alpha=0.1          # Ranking-loss weight in official train.py.
# activation=GELU
#
# ==============================================================================
# FactorVAE: Probabilistic Dynamic Factor Model Based on VAE
# Source confidence: paper primary, public reproduction cross-check only.
# Static source anchors:
# ==============================================================================
# MODEL=factorvae
# model_file=src/factorvae/model.py
# seq_len=20
# num_features=20    # Paper: 20 Qlib-selected time-series features.
# feature_dim=24     # Local dataset candidate; adapter should map/select features explicitly.
# hidden_size=64
# num_factors=96
# num_portfolios=128
# lr=1e-4
# epochs=30
# batch_size=300     # Public reproduction utility default; paper batches by daily cross-section.
# kl_weight=1.0      # Paper gamma; exact tuned value not clearly specified in extracted text.
# train_period=2010-01-01:2017-12-31
# valid_period=2018-01-01:2018-12-31
# test_period=2019-01-01:2020-12-31
# topk=50
# drop=5
#
# ==============================================================================
# FactorGCL: Hypergraph-Based Factor Model with Temporal Residual Contrastive Learning
# Source confidence: PDF-derived. No usable official model source found.
# Static source anchors:
# ==============================================================================
# MODEL=factorgcl
# model_file=src/factorgcl/model.py
# input_dim=6         # Paper: OHLC, VWAP, volume.
# feature_dim=24      # Local dataset candidate; adapter should map/select features explicitly.
# seq_len=60
# future_seq_len=20
# horizons=1,5,10,20
# num_horizons=4
# prior_factors=83    # Secondary industry factors.
# num_hidden_factors=32
# hidden_dim=32          # Feature embedding dimension H.
# num_gru_layers=2       # Number of RNN layers L.
# temperature=0.1
# contrastive_gamma=0.1
# optimizer=Adam
# lr=1e-3
# epochs=100
# early_stopping_steps=20
# seed=0
# train_valid_test=5y:1y:2y_rolling
# test_period=2020-01-01:2023-06-30
#
# ==============================================================================
# Enhancer: Distribution-Aware Framework with Temporal-Relational Meta-Learning
# Source confidence: PDF-derived. Public appendix repo has Appendix.pdf only, no model source.
# Static source anchors:
# ==============================================================================
# MODEL=enhancer
# model_file=src/enhancer/model.py
# lookback_window=16       # Main-paper stock recommendation setting.
# feature_dim=5            # Original OHLCV-style setting.
# local_feature_dim=24     # Local dataset candidate; adapter should map/select features explicitly.
# downstream_predictor=Transformer  # Appendix says Transformer is strong for investment recommendation.
# num_contexts=30
# polynomial_order=3       # Ka=3.
# relation_dim=30          # F=30, shared hidden representation dimension in paper text.
# noise_terms=10           # T_N=10.
# integrate_steps=unreported  # K_z in Eq.(6) is not clearly exposed in extracted text.
# reg_beta=12              # Balancing parameter beta in Eq.(17).
# rpps_epsilon1=0.5
# rpps_epsilon2=0.5
# rpps_alpha=1
# topk_stocks=10           # Appendix Fig. 2 profitability uses Select top-k=10.
# transformer_hidden_size=30
# transformer_filter_size=256
# transformer_heads=4
# transformer_encoder_layers=2
# horizon=1
# epochs=20
# lr=0.0007
# batch_size=6
# stock_rank_zeta=4        # Searched in [1,2,3,4,5,6], finally set to 4.
# noise_scale=unreported   # Paper samples X_noise but does not expose a scalar noise std in extracted text.
# reg1_weight=12.0
# reg2_weight=1.0
#
# ==============================================================================
# CI-STHPAN: Channel-Independent Spatio-Temporal Hypergraph Attention Network
# Source confidence: official repo + paper, dependency-reduced into one file.
# Static source anchors:
# ==============================================================================
# MODEL=ci_sthpan
# model_file=src/ci_sthpan/model.py
# context_points=512       # Self-supervised pretrain/finetune scripts.
# supervised_context_points=336
# target_points=1
# patch_len=12             # Paper reports best return at patch length 12 in fine-tuning study.
# stride=12
# n_layers=3
# n_heads=16
# d_model=128
# d_ff=256
# dropout=0.2
# head_dropout=0
# graph=1
# ci=1
# revin=1
# rel_type=3               # DTW graph in the official fine-tune sensitivity script.
# k=20                     # NASDAQ DTW top-k candidate from official script/paper discussion.
# k=5                      # NYSE DTW top-k candidate from paper discussion.
# alpha=2                  # Paper discussion reports maximum return around alpha=2.
# lr=1e-4
# batch_size=1             # Each batch is a market cross-section in the official self-supervised scripts.
# pretrain_epochs=100
# finetune_epochs=20
# mask_ratio=0.4

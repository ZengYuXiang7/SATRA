# SATRA

| Baseline | Core implementation | Implemented components | Training entry | Input and relation contract | Formal launcher |
| --- | --- | --- | --- | --- | --- |
| StockMixer | `src/stockmixer/model.py` | causal temporal mixer, indicator mixer, multi-scale mixer, cross-stock mixer | `python -m stockmixer.run` | `stockmixer_ma`: MA5, MA10, MA20, MA30, close; no relation file | `script/baselines/025_20260706_stockmixer_3datasets_5runs.sh` |
| FactorVAE | `src/factorvae/model.py` | GRU feature extractor, posterior factor encoder, prior factor predictor, alpha and beta layers | `python -m factorvae.run` | `factorvae_alpha20`: 20 train-normalized factors; no relation file | `script/baselines/006_20260706_factorvae_3datasets_5runs.sh` |
| MASTER | `src/master/model.py`, `src/master/adapter.py` | temporal attention, cross-stock attention, market-guided gating, prediction decoder | `python -m master.run` | `master_stock_market`: stock and market-context features; no relation file | `script/baselines/009_20260706_master_3datasets_5runs.sh` |
| Enhancer | `src/enhancer/model.py` | temporal meta-learner, relational meta-learner, graph propagation, invariant and intervened heads | `python -m enhancer.run` | `enhancer_stock`: raw and stock-derived features; no external relation file | `script/baselines/003_20260706_enhancer_3datasets_5runs.sh` |
| RSR | `src/rsr/model.py` | LSTM stock encoder, typed relation scorer, relation-aware ranking head | `python -m rsr.run` | `stock_all` plus a relation tensor | `script/baselines/013_20260706_rsr_3datasets_5runs.sh` |
| HyperStockGAT | `src/hstockgat/model.py` | hyperbolic projection, hyperbolic graph convolution, temporal aggregation | `python -m hstockgat.run` | `stock_all` plus a relation tensor | `script/baselines/007_20260706_hstockgat_3datasets_5runs.sh` |
| CI-STHPAN | `src/ci_sthpan/model.py` | channel-independent patch encoder, temporal Transformer, hypergraph attention | `python -m ci_sthpan.run` | `stock_all` plus a hyperedge index | `script/baselines/001_20260706_ci_sthpan_3datasets_5runs.sh` |
| FactorGCL | `src/factorgcl/model.py` | GRU feature extractor, prior and hidden factor hypergraphs, temporal residual contrastive learning | `python -m factorgcl.run` | `stock_all`, relation tensor, and factor prior | `script/baselines/004_20260706_factorgcl_3datasets_5runs.sh` |
| TRA | `src/tra/model.py` | attention LSTM, multiple predictors, routing adaptor, causal error memory | `python -m tra.run` | `stock_all`; shared loader resolves a relation file | `script/baselines/026_20260706_tra_3datasets_5runs.sh` |

| Time-series baseline | Core implementation | Implemented components | Package entry |
| --- | --- | --- | --- |
| LSTNet | `src/timeseries/models/LSTNet.py` | temporal convolution, recurrent branch, skip recurrent branch, highway path | `load_model_class("LSTNet")` |
| xLSTM-Mixer | `src/timeseries/models/xLSTMMixer.py` | scalar-memory recurrence, temporal mixing, channel mixing | `load_model_class("xLSTMMixer")` |
| ModernTCN | `src/timeseries/models/ModernTCN.py` | patch stem, large-kernel depthwise convolution, grouped feed-forward blocks | `load_model_class("ModernTCN")` |
| TimeMixer | `src/timeseries/models/TimeMixer.py` | multiscale downsampling, seasonal mixing, trend mixing | `load_model_class("TimeMixer")` |
| WPMixer | `src/timeseries/models/WPMixer.py` | wavelet decomposition, multiresolution branches, token mixing | `load_model_class("WPMixer")` |
| PatchTST | `src/timeseries/models/PatchTST.py` | channel-independent patch embedding, Transformer encoder, flattening head | `load_model_class("PatchTST")` |
| iTransformer | `src/timeseries/models/iTransformer.py` | inverted variate tokens, attention encoder, forecasting projection | `load_model_class("iTransformer")` |
| Crossformer | `src/timeseries/models/Crossformer.py` | dimension-segment embedding, two-stage attention, hierarchical decoder | `load_model_class("Crossformer")` |
| xPatch | `src/timeseries/models/xPatch.py` | exponential decomposition, nonlinear patch stream, linear trend stream | `load_model_class("xPatch")` |
| DUET | `src/timeseries/models/DUET.py` | sparse temporal experts, channel Transformer, gated clustering | `load_model_class("DUET")` |

SATRA is an anonymous stock return prediction implementation that combines
temporal pattern specialization, relation-aware cross-stock reasoning, and
masked market reconstruction. The repository also includes the financial
and generic time-series baselines used in the evaluation.

Each runnable baseline is a self-contained package under `src/<baseline>/`.
`model.py` contains the neural implementation, `data.py` defines its view of
the shared processed arrays, and `run.py` contains training, validation
RankIC checkpointing, testing, and repeated-run reporting. Package-specific
losses or adapters are kept beside those files.

## Model

SATRA maps a market window with shape `[batch, stocks, time, features]` to one
return score per stock:

```text
market window
  -> feature projection
  -> Adaptive Temporal Pattern Encoder
       temporal convolution
       heterogeneous bottleneck experts
       sparse top-k routing
  -> Dual Relation Tower
       learned semantic cross-stock attention
       prior-guided cross-stock attention
       state-conditioned softmax gate over the two tower outputs
  -> prediction head
  -> stock return scores
```

The semantic tower learns cross-stock interactions from the current hidden
states. The prior-guided tower restricts attention with a precomputed relation
matrix. A two-way softmax gate computed from the shared ATPE output assigns
stock- and time-specific weights to the two towers. The gate is trained in
both stages and transferred with the MMR backbone checkpoint.
Masked Market Reconstruction optionally pretrains the same backbone by
reconstructing masked stock-time tokens before supervised adaptation.

All formal SATRA and baseline runs select checkpoints by validation RankIC,
maximized. MAE is reported as an evaluation metric but is not a checkpoint
selection criterion.

## Repository Layout

```text
.
├── prepare_data.py            # public data download and preprocessing entry
├── pyproject.toml
├── run.py
├── data/
│   └── priors/                # one small NASDAQ DTW relation file
├── src/
│   ├── satra/                 # SATRA model, training, data, and ablations
│   ├── metrics/               # shared baseline metric contract
│   ├── observability/         # shared experiment reporting
│   ├── timeseries/            # generic time-series models and shared layers
│   └── <baseline packages>/   # financial time-series baselines
└── script/
    ├── SATRA/                 # SATRA launchers ordered by execution date
    └── baselines/             # baseline launchers and evaluation tools
```

The submission package does not contain raw prices, processed tensors,
baseline relation arrays, checkpoints, results, logs, local environments, or
lock files. The only bundled data artifact is the basic NASDAQ Top-32 DTW graph
under `data/priors/`.

## Environment

SATRA requires Python 3.12.

```bash
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml`. No environment manager or
project-local virtual environment is required by the code.

## Download and Prepare Data

The public sources are identified by stable GitHub repository IDs rather than
personal account names:

- [Repository 157677083](https://api.github.com/repositories/157677083),
  named `Temporal_Relational_Stock_Ranking`, provides NASDAQ and NYSE
  end-of-day data, ticker lists, calendars, and optional relation metadata.
- [Repository 523661816](https://api.github.com/repositories/523661816),
  named `estimate`, provides `baseline_data_sp500.npy`, S&P 500 metadata, and
  optional lead-lag relations.

`prepare_data.py` resolves each repository ID through the public GitHub API,
checks the expected repository name, and then performs a shallow clone. No
personal account name is stored in this submission package.

The following command downloads both repositories and creates all three
model-ready datasets:

```bash
python prepare_data.py all --markets NASDAQ NYSE SP500
```

The command writes only to ignored working directories:

```text
data/raw/sources/            # shallow public source checkouts
data/raw/input/              # generated symlink staging area
data/processed/
├── NASDAQ_ohlcv/
├── NYSE_ohlcv/
└── SP500_ohlcv/
```

Each market directory contains `eod_data.pkl`, `gt_data.pkl`,
`mask_data.pkl`, `observed_mask_data.pkl`, `price_data.pkl`, and
`feature_config.json`. Features are normalized using training-period
statistics. The chronological validation and test periods are not used to fit
normalization statistics.

## Feature Construction

The default command builds the 24-dimensional input used by the formal SATRA
experiments. The feature groups are:

| Group | Count | Features |
| --- | ---: | --- |
| Raw stock observations | 5 | open, high, low, volume, close |
| Stock-derived features | 11 | return, price range, open-to-close return, volume change, overnight return, close-to-MA at 5/10/15 days, and return momentum at 5/10/15 days |
| Market context | 8 | market return, breadth, advance-decline indicator, return dispersion, up-volume ratio, volume surprise, 5-day market momentum, and 16-day market drawdown |

The implementation is in
`src/satra/data_provider/generate_data.py`. The same file constructs masks,
linearly imputes missing OHLCV values when requested, computes next-day raw
return targets, records observed-value masks, and standardizes inputs using
training-period statistics only.

Optional Alpha158-compatible features are also implemented. Because the public
OHLCV sources do not contain amount or true VWAP, `alpha158_vwap0` is omitted.
The remaining 157 factors consist of 12 candlestick features and 29 rolling
operators evaluated at 5, 10, 20, 30, and 60 days. Enabling them produces 181
input dimensions in total:

```bash
python prepare_data.py all \
  --markets NASDAQ NYSE SP500 \
  --alpha158
```

If the public sources have already been downloaded:

```bash
python prepare_data.py preprocess \
  --markets NASDAQ NYSE SP500 \
  --alpha158 \
  --clean-output
```

Baseline-specific transformations are generated from the same processed
arrays at load time:

- StockMixer constructs MA5, MA10, MA20, MA30, and close-price inputs in
  `src/satra/data_provider/stock_data.py`.
- FactorVAE constructs and train-normalizes its selected Alpha20 inputs in
  `src/satra/data_provider/alpha_features.py`.
- Other included baselines select documented raw, stock-derived, or market
  feature groups from `feature_config.json`.

After NASDAQ preprocessing, `prepare_data.py` copies the bundled basic DTW
graph to the path expected by the launchers:

```text
data/processed/NASDAQ_ohlcv/priors/
  dtw_binary_imputed_inputstandard_NASDAQ_train0-871_close_ds64_band6_cand48_top32.npz
```

To download first and preprocess later:

```bash
python prepare_data.py download --markets NASDAQ NYSE SP500
python prepare_data.py preprocess --markets NASDAQ NYSE SP500
```

To replace previously generated market directories:

```bash
python prepare_data.py preprocess --markets NASDAQ NYSE SP500 --clean-output
```

Raw source checkouts and generated tensors are substantially larger than the
submission limit. They remain ignored and must not be added to an anonymous
submission archive.

## Optional Baseline Relations

Most non-graph baselines use only `data/processed`. Graph and factor baselines
also require larger relation arrays. Generate them from the same public
sources only when needed:

```bash
python prepare_data.py all \
  --markets NASDAQ NYSE SP500 \
  --baseline-relations
```

This creates `data/processed/relations/`. These generated arrays are not part
of the submission package.

## Run SATRA

Show the supervised training options:

```bash
python run.py --help
```

Show the two-stage reconstruction and fine-tuning options:

```bash
PYTHONPATH=src python -m satra.mmr_two_stage --help
```

Run the formal three-market two-stage launcher after preparing the required
market data and priors:

```bash
bash script/SATRA/gated-drt-three-markets.sh
```

The gated model is selected with `--model model_b_spe_drt_feature_aggregation`;
`--drt_load_balance_weight` controls its two-tower balance loss in both stages.
Earlier numbered launchers and ablation modules retain their original model
identities and configurations; they are not the current gated-model launcher.
The supplied three-market launcher preserves the recovered experiment settings;
it is not a complete record of every later paper configuration.

The repository bundles only the NASDAQ Top-32 DTW prior. The gated launcher
requires the NASDAQ Top-8, NYSE Top-32, and S&P 500 Top-16 priors, which must
be generated separately from each prepared training split.

## Run Baselines

Baseline entry points can be inspected directly:

```bash
PYTHONPATH=src python -m stockmixer.run --help
PYTHONPATH=src python -m factorvae.run --help
PYTHONPATH=src python -m master.run --help
PYTHONPATH=src python -m enhancer.run --help
PYTHONPATH=src python -m rsr.run --help
PYTHONPATH=src python -m hstockgat.run --help
PYTHONPATH=src python -m ci_sthpan.run --help
PYTHONPATH=src python -m factorgcl.run --help
PYTHONPATH=src python -m tra.run --help
```

The consolidated RankIC-selected baseline queue is:

```bash
bash script/baselines/028_20260709_rankic_monitor_5run_full.sh
```

## Generic Time-Series Implementations

The ten generic forecasting architectures are grouped under
`src/timeseries/`. Their common interface maps an input tensor with shape
`[batch, history, features]` to a forecast with shape
`[batch, horizon, features]`. Models are loaded lazily:

```python
from timeseries import MODEL_MODULES, load_model_class

model_class = load_model_class("PatchTST")
```

Only the architecture implementations and required neural layers are included.
The package does not add a second checkpoint-monitoring workflow; formal SATRA
and financial-baseline experiment selection remains validation RankIC,
maximized. Required upstream license notices are retained in
`src/timeseries/THIRD_PARTY_LICENSES`; those notices identify third-party code
owners, not the anonymous submission authors.

## SATRA Code Guide

| File | Responsibility |
| --- | --- |
| `src/satra/b_spe_drt.py` | Adaptive Temporal Pattern Encoder, Dual Relation Tower, prediction head, and masked reconstruction model |
| `src/satra/mmr_two_stage.py` | masked reconstruction pretraining and supervised adaptation |
| `src/satra/main.py` | command-line contract and repeated-run orchestration |
| `src/satra/exp/exp_main.py` | training, validation RankIC checkpointing, and test evaluation |
| `src/satra/data_provider/stock_data.py` | chronological splits, rolling windows, and dataloaders |
| `src/satra/data_provider/generate_data.py` | raw-source conversion, feature construction, and train-only normalization |
| `src/satra/data_provider/stock_prior.py` | training-only DTW relation construction |
| `src/timeseries/` | ten generic forecasting models, shared attention, embedding, normalization, and wavelet layers |
| `src/satra/utils/metrics.py` | SATRA evaluation metrics |
| `src/satra/utils/stock_loss.py` | supervised stock prediction objective |
| `src/satra/ablation/` | component-level variants used by ablation launchers |

Some runner-facing class names retain historical implementation aliases to
preserve checkpoint and launcher compatibility. They refer to SATRA components
and do not change the model described above.

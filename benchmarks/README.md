# FeatCopilot Benchmarks

This directory contains comprehensive benchmarks demonstrating FeatCopilot's feature engineering capabilities.

## Headline Results

| Benchmark | Improvement | Description |
|-----------|-------------|-------------|
| **Spotify Classification** | **+12.37%** F1 | LLM + Text + Tabular engines with FLAML AutoML |
| **INRIA (Tabular)** | +0.40% avg, max +8.25% | Tabular only on 10 datasets with RF/Ridge |
| **INRIA (Tabular+LLM)** | +0.07% avg, max +2.46% | Tabular + LLM on 10 datasets with RF/Ridge |

## Structure

```
benchmarks/
├── datasets.py                              # Shared benchmark datasets
│
├── automl/                                  # AutoML integration benchmarks
│   ├── run_flaml_spotify_benchmark.py       # Spotify benchmark (LLM+Text+Tabular)
│   ├── run_flaml_realworld_benchmark.py     # 10 real-world datasets
│   ├── run_automl_benchmark.py              # Multi-framework (FLAML, AutoGluon, H2O)
│   ├── FLAML_SPOTIFY_CLASSIFICATION_REPORT.md  # Spotify result (+12.37%)
│   └── FLAML_REALWORLD_BENCHMARK_REPORT.md
│
├── feature_engineering/                     # Feature engineering benchmarks
│   ├── run_inria_basic_benchmark.py         # INRIA/OpenML datasets
│   ├── run_basic_models_benchmark.py        # Kaggle-style datasets
│   ├── INRIA_BASIC_MODELS_TABULAR.md        # INRIA results (Tabular only)
│   ├── INRIA_BASIC_MODELS_TABULAR_LLM.md    # INRIA results (Tabular+LLM)
│   ├── BASIC_MODELS_BENCHMARK_TABULAR.md    # Basic models (Tabular only)
│   └── BASIC_MODELS_BENCHMARK_TABULAR_LLM.md # Basic models (Tabular+LLM)
│
└── compare_tools/                           # Comparison with other FE tools
    ├── run_comparison_benchmark.py          # vs Featuretools, TSFresh, AutoFeat
    └── COMPARISON_BENCHMARK_REPORT.md
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[benchmark]"

# Run Spotify benchmark
python benchmarks/automl/run_flaml_spotify_benchmark.py

# Run INRIA benchmark suite
python benchmarks/feature_engineering/run_inria_basic_benchmark.py --engines tabular
python benchmarks/feature_engineering/run_inria_basic_benchmark.py --engines tabular llm

# Run tool comparison
python benchmarks/compare_tools/run_comparison_benchmark.py
```

## Benchmark Details

### 1. Spotify Genre Classification

Demonstrates FeatCopilot's full capabilities with LLM + Text + Tabular engines.

```bash
python benchmarks/automl/run_flaml_spotify_benchmark.py
```

**Result**: +12.37% F1 improvement (0.8243 → 0.9263)

### 2. INRIA Benchmark Suite

10 diverse datasets from OpenML testing Tabular and Tabular+LLM configurations with basic models (RandomForest, Ridge/LogisticRegression).

```bash
# Tabular only (10 datasets)
python benchmarks/feature_engineering/run_inria_basic_benchmark.py --medium --engines tabular

# Tabular + LLM (10 datasets)
python benchmarks/feature_engineering/run_inria_basic_benchmark.py --medium --engines tabular,llm
```

**Result**: +0.40% avg (Tabular), +0.07% avg (Tabular+LLM)

### 3. Tool Comparison

Compare FeatCopilot with Featuretools, TSFresh, and AutoFeat.

```bash
python benchmarks/compare_tools/run_comparison_benchmark.py
```

**Result**: Competitive accuracy with 1000x faster FE time than AutoFeat

### 4. AutoML Integration

Test FeatCopilot with FLAML, AutoGluon, and H2O.

```bash
python benchmarks/automl/run_automl_benchmark.py --frameworks flaml autogluon h2o
```

## Reports

| Report | Location | Description |
|--------|----------|-------------|
| **Spotify Classification** | `automl/FLAML_SPOTIFY_CLASSIFICATION_REPORT.md` | +12.37% F1 result |
| **INRIA (Tabular)** | `feature_engineering/INRIA_BASIC_MODELS_TABULAR.md` | 10 datasets, +0.40% avg |
| **INRIA (Tabular+LLM)** | `feature_engineering/INRIA_BASIC_MODELS_TABULAR_LLM.md` | 10 datasets, +0.07% avg |
| **Tool Comparison** | `compare_tools/COMPARISON_BENCHMARK_REPORT.md` | vs 4 other FE tools |
| **FLAML Real-World** | `automl/FLAML_REALWORLD_BENCHMARK_REPORT.md` | 10 real-world datasets |

## Dependencies

### Core (Required)

```bash
pip install -e ".[benchmark]"  # Includes FLAML
```

### Optional (For Tool Comparison)

```bash
pip install featuretools  # Deep Feature Synthesis
pip install tsfresh       # Time series features
pip install autofeat      # Automatic feature generation
```

### Optional (For AutoML)

```bash
pip install autogluon  # AutoGluon
pip install h2o        # H2O AutoML
```

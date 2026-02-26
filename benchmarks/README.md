# FeatCopilot Benchmarks

Comprehensive benchmarks demonstrating FeatCopilot's feature engineering capabilities across 63 datasets.

## Latest Results Summary

### Simple Models Benchmark (RandomForest, LogisticRegression/Ridge)

| Metric | Multi-Engine |
|--------|--------------|
| **Datasets** | 63 |
| **Improved** | 31 (49%) |
| **Avg Improvement** | **+7.52%** |
| **Best Improvement** | +144% (triple_interaction_regression) |

**Key Highlights:**
- **triple_interaction_regression**: +144% R² improvement
- **xor_regression**: +104% R² improvement
- **pairwise_product_regression**: +70% R² improvement
- **complex_classification**: +16.49% accuracy boost
- **xor_classification**: +16.67% accuracy boost

### AutoML Benchmark (FLAML + AutoGluon, 120s budget)

| Framework | Datasets | Improved | Avg Improvement |
|-----------|----------|----------|-----------------|
| **FLAML** | 10 | 9 (90%) | **+1.85%** |
| **AutoGluon** | 10 | 9 (90%) | **+1.55%** |

**Notable Results:**
- **complex_classification**: +6.67% (FLAML), +7.62% (AutoGluon)
- **xor_classification**: +5.62% (FLAML), +2.42% (AutoGluon)
- **polynomial_regression**: +2.99% (FLAML)
- **titanic**: +1.37% (both frameworks)

### FE Tools Comparison (FeatCopilot vs autofeat vs featuretools)

| Metric | FeatCopilot | autofeat | featuretools |
|--------|-------------|----------|--------------|
| **Win Rate** | **80%** 🏆 | 40% | 0% |
| **Avg Improvement** | **+1.89%** 🏆 | +1.46% | -2.71% |
| **Coverage** | **100%** 🏆 | 50% | 100% |
| **Composite Score** | **0.606** 🥇 | 0.351 🥉 | 0.397 🥈 |

## Structure

```
benchmarks/
├── datasets.py                    # Unified dataset API (63 datasets)
├── datasets.md                    # Dataset documentation
│
├── automl/                        # AutoML benchmarks
│   └── run_automl_benchmark.py    # FLAML/AutoGluon with FeatCopilot
│
├── simple_models/                 # Simple models benchmarks
│   └── run_simple_models_benchmark.py  # RF/LogReg/Ridge with FeatCopilot
│
└── compare_tools/                 # FE tools comparison
    └── run_fe_tools_comparison.py # FeatCopilot vs other FE frameworks
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[benchmark]"

# Run AutoML benchmark
python -m benchmarks.automl.run_automl_benchmark

# Run simple models benchmark
python -m benchmarks.simple_models.run_simple_models_benchmark

# Run FE tools comparison
python -m benchmarks.compare_tools.run_fe_tools_comparison
```

## Benchmark Scripts

### 1. AutoML Benchmark (`automl/run_automl_benchmark.py`)

Compares AutoML frameworks (FLAML, AutoGluon) with and without FeatCopilot.
FeatCopilot runs all applicable engines per dataset (tabular, relational) plus LLM when enabled.

```bash
# Quick benchmark (10 datasets)
python -m benchmarks.automl.run_automl_benchmark

# Specific datasets
python -m benchmarks.automl.run_automl_benchmark --datasets titanic,house_prices

# All classification datasets
python -m benchmarks.automl.run_automl_benchmark --category classification

# With LLM engine
python -m benchmarks.automl.run_automl_benchmark --with-llm

# Use AutoGluon
python -m benchmarks.automl.run_automl_benchmark --framework autogluon

# All frameworks
python -m benchmarks.automl.run_automl_benchmark --framework all
```

**Options:**
- `--datasets NAME,NAME` - Specific datasets
- `--category {classification,regression}` - All datasets in category
- `--all` - All datasets
- `--framework {flaml,autogluon,h2o,all}` - AutoML framework
- `--with-llm` - Enable LLM engine
- `--time-budget N` - AutoML time budget (default: 120s)

**Output:**
- Per-framework: `AUTOML_{FRAMEWORK}_BENCHMARK.md`
- Combined: `AUTOML_BENCHMARK.md` (when running multiple frameworks)

### 2. Simple Models Benchmark (`simple_models/run_simple_models_benchmark.py`)

Compares simple models with and without FeatCopilot.
FeatCopilot runs all applicable engines per dataset (tabular, timeseries, text, relational) plus LLM when enabled.

```bash
# Quick benchmark
python -m benchmarks.simple_models.run_simple_models_benchmark

# All regression datasets with LLM
python -m benchmarks.simple_models.run_simple_models_benchmark --category regression --with-llm
```

**Models:**
- Classification: RandomForest (n_estimators=200, max_depth=20), LogisticRegression
- Regression: RandomForest (n_estimators=200, max_depth=20), Ridge

**Output:**
- Without LLM: `SIMPLE_MODELS_BENCHMARK.md`
- With LLM: `SIMPLE_MODELS_BENCHMARK_LLM.md`

### 3. FE Tools Comparison (`compare_tools/run_fe_tools_comparison.py`)

Compares FeatCopilot with other feature engineering frameworks using FLAML (30s budget).

```bash
# Compare all available tools
python -m benchmarks.compare_tools.run_fe_tools_comparison

# Specific tools
python -m benchmarks.compare_tools.run_fe_tools_comparison --tools featcopilot featuretools autofeat
```

**Tools compared:**
- Baseline (no FE)
- FeatCopilot
- Featuretools
- autofeat

**Output:** `FE_TOOLS_COMPARISON.md`

## Datasets

63 datasets across 4 categories:

| Category | Count | Examples |
|----------|-------|----------|
| Classification | 26 | titanic, credit_risk, xor_classification, higgs |
| Regression | 30 | house_prices, complex_regression, diamonds |
| Forecasting | 3 | sensor_anomaly, retail_demand |
| Text | 4 | product_reviews, fake_news |

```python
from benchmarks.datasets import list_datasets, load_dataset

# List all datasets
list_datasets()

# List by category
list_datasets('classification')

# Load dataset
X, y, task, name = load_dataset('titanic')
```

See `datasets.md` for full documentation.

## Dependencies

### Core
```bash
pip install -e ".[benchmark]"  # Includes FLAML
```

### Optional (for tool comparison)
```bash
pip install featuretools  # Deep Feature Synthesis
pip install tsfresh       # Time series features
pip install autofeat      # Automatic feature generation
pip install openfe        # OpenFE
pip install caafe         # CAAFE (requires OpenAI API)
```

### Optional (for AutoML)
```bash
pip install autogluon     # AutoGluon
```

# FeatCopilot Benchmarks

Comprehensive benchmarks demonstrating FeatCopilot's feature engineering capabilities.

## Structure

```
benchmarks/
├── datasets.py                    # Unified dataset API (52 datasets)
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

```bash
# Quick benchmark (6 datasets)
python -m benchmarks.automl.run_automl_benchmark

# Specific datasets
python -m benchmarks.automl.run_automl_benchmark --datasets titanic,house_prices

# All classification datasets
python -m benchmarks.automl.run_automl_benchmark --category classification

# With LLM engine
python -m benchmarks.automl.run_automl_benchmark --with-llm

# Use AutoGluon
python -m benchmarks.automl.run_automl_benchmark --framework autogluon
```

**Options:**
- `--datasets NAME,NAME` - Specific datasets
- `--category {classification,regression}` - All datasets in category
- `--all` - All datasets
- `--framework {flaml,autogluon}` - AutoML framework
- `--with-llm` - Enable LLM engine
- `--time-budget N` - AutoML time budget (default: 60s)

**Output:** `AUTOML_BENCHMARK_YYYYMMDD.md`

### 2. Simple Models Benchmark (`simple_models/run_simple_models_benchmark.py`)

Compares simple models with and without FeatCopilot.

```bash
# Quick benchmark
python -m benchmarks.simple_models.run_simple_models_benchmark

# All regression datasets with LLM
python -m benchmarks.simple_models.run_simple_models_benchmark --category regression --with-llm
```

**Models:**
- Classification: RandomForest, LogisticRegression
- Regression: RandomForest, Ridge

**Output:** `SIMPLE_MODELS_BENCHMARK_YYYYMMDD.md`

### 3. FE Tools Comparison (`compare_tools/run_fe_tools_comparison.py`)

Compares FeatCopilot with other feature engineering frameworks using FLAML.

```bash
# Compare all available tools
python -m benchmarks.compare_tools.run_fe_tools_comparison

# Specific tools
python -m benchmarks.compare_tools.run_fe_tools_comparison --tools featcopilot featuretools openfe
```

**Tools compared:**
- Baseline (no FE)
- FeatCopilot
- Featuretools
- tsfresh
- autofeat
- OpenFE
- CAAFE

**Output:** `FE_TOOLS_COMPARISON_YYYYMMDD.md`

## Datasets

52 datasets across 4 categories:

| Category | Count | Examples |
|----------|-------|----------|
| Classification | 22 | titanic, credit_risk, higgs (INRIA) |
| Regression | 20 | house_prices, diamonds (INRIA) |
| Forecasting | 3 | sensor_anomaly, retail_demand |
| Text | 7 | product_reviews, fake_news |

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

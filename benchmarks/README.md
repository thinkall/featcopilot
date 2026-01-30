# FeatCopilot Benchmarks

This directory contains benchmarks for evaluating FeatCopilot's performance.

## Structure

```
benchmarks/
├── datasets.py                    # Shared benchmark datasets
├── feature_engineering/           # Feature engineering benchmarks
│   ├── run_benchmark.py          # Main benchmark runner
│   ├── BENCHMARK_REPORT.md       # Tabular engine results
│   └── LLM_BENCHMARK_REPORT.md   # LLM engine results
└── automl/                        # AutoML integration benchmarks
    └── run_automl_benchmark.py   # AutoML benchmark runner
```

## Feature Engineering Benchmarks

Evaluates FeatCopilot's feature engineering capabilities compared to baseline models.

```bash
# Run feature engineering benchmarks
python benchmarks/feature_engineering/run_benchmark.py
```

## AutoML Integration Benchmarks

Evaluates FeatCopilot's impact when combined with AutoML frameworks.

### Supported Frameworks

- **FLAML** (Microsoft) - Fast and lightweight AutoML
- **AutoGluon** (Amazon) - Easy-to-use AutoML with state-of-the-art results
- **Auto-sklearn** - Automated machine learning with scikit-learn
- **H2O AutoML** - Scalable AutoML platform

### Installation

Install the AutoML frameworks you want to test:

```bash
# FLAML (recommended - fastest)
pip install flaml

# AutoGluon
pip install autogluon

# Auto-sklearn (Linux only)
pip install auto-sklearn

# H2O
pip install h2o
```

### Running Benchmarks

```bash
# Run with all available frameworks
python benchmarks/automl/run_automl_benchmark.py

# Run with specific frameworks
python benchmarks/automl/run_automl_benchmark.py --frameworks flaml autogluon

# With longer time budget (better results)
python benchmarks/automl/run_automl_benchmark.py --time-budget 120

# Enable LLM-powered features (requires GitHub Copilot)
python benchmarks/automl/run_automl_benchmark.py --enable-llm

# Custom output path
python benchmarks/automl/run_automl_benchmark.py --output results.md
```

### Benchmark Methodology

1. **Baseline**: Run AutoML on raw dataset
2. **With FeatCopilot**: Apply feature engineering, then run AutoML
3. **Compare**: Measure improvement in primary metric (accuracy/R²)

### Datasets

Both benchmark suites use the same datasets from `datasets.py`:

| Dataset | Task | Description |
|---------|------|-------------|
| Titanic | Classification | Survival prediction |
| House Prices | Regression | Price prediction |
| Credit Card Fraud | Classification | Fraud detection |
| Bike Sharing | Regression | Demand forecasting |
| Employee Attrition | Classification | Churn prediction |

## Results

See the respective report files for detailed results:

- `feature_engineering/BENCHMARK_REPORT.md` - Tabular engine results
- `feature_engineering/LLM_BENCHMARK_REPORT.md` - LLM engine results
- `automl/AUTOML_BENCHMARK_REPORT.md` - AutoML integration results (generated after running)

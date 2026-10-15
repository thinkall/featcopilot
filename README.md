# FeatCopilot 🚀

**Next-Generation LLM-Powered Auto Feature Engineering Framework**

[![Tests](https://github.com/thinkall/featcopilot/actions/workflows/tests.yml/badge.svg)](https://github.com/thinkall/featcopilot/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/thinkall/featcopilot/graph/badge.svg)](https://codecov.io/gh/thinkall/featcopilot)

FeatCopilot automatically generates, selects, and explains predictive features using semantic understanding. It analyzes column meanings, applies domain-aware transformations, and provides human-readable explanations—turning raw data into ML-ready features in seconds.

## 🎬 Introduction Video

[![FeatCopilot Introduction](https://img.youtube.com/vi/H7m50TLGHFk/0.jpg)](https://www.youtube.com/watch?v=H7m50TLGHFk)

## 📊 Benchmarking

Our goal is strong, reproducible tabular feature engineering. **Current evidence
does not establish SOTA or guarantee that engineered features improve every
model.** The canonical benchmark uses a fixed, versioned real-data subset,
training-only preprocessing, inner-validation model tuning, and identical outer
folds for FeatCopilot, raw features, polynomial features, AutoFeat, and an explicitly
rowwise-restricted OpenFE variant. Later studies also include inductive OpenFE
and native-width AutoFeat references under separately recorded method identities.

Results are reported separately for linear models, RandomForest, and LightGBM,
and for classification, regression, and synthetic diagnostics. Failures,
timeouts, feature counts, runtime, and dataset-level uncertainty remain visible.

[Protocol and improvement task list](benchmarks/TABULAR_PROTOCOL.md) ·
[Running benchmarks](benchmarks/README.md) ·
[Measured development baseline](benchmarks/tabular-results/2026-09-16/ANALYSIS.md) ·
[Predictive-strategy follow-up](benchmarks/tabular-results/2026-09-16-predictive/ANALYSIS.md)

Older headline gains and composite rankings have been retired: the legacy
experiments mixed synthetic diagnostics into summaries and included test-fold
model selection or asymmetric preprocessing. Their historical reports remain
available for traceability, not as current comparative evidence.

Recent **opt-in development** improvements are supported by complete paired
measurements: [classifier averaging](benchmarks/tabular-results/2026-10-04-fixed-average-development/ANALYSIS.md)
reduced mean log loss by 6.82% versus its raw control on the three-seed panel.
[Regression replication](benchmarks/tabular-results/2026-10-08-regression-policy-replication/ANALYSIS.md)
reduced primary-panel mean RMSE by 23.79% for linear single FE and 2.97% for
LightGBM averaging. These exposed-panel results are not independent confirmation:
individual hardware folds still suffered severe harm. Experimental modes remain
explicit, legacy defaults are unchanged, and negative results stay published.

## Key Features

- 🔧 **Multi-Engine Architecture**: Tabular, time series, relational, and text feature engines
- 🤖 **LLM-Powered Intelligence**: Semantic feature discovery, domain-aware generation, and code synthesis
- 📊 **Intelligent Selection**: Statistical testing, importance ranking, and redundancy elimination
- 🔌 **Scikit-learn Compatible**: Drop-in replacement for sklearn transformers
- 📝 **Interpretable**: Every feature comes with human-readable explanations

## Installation

The core package supports Python 3.10+. The GitHub Copilot SDK extra
(`llm`, also included by `full` and `benchmark`) requires Python 3.11+
and SDK 0.3.0 or newer, matching the session APIs used by FeatCopilot.

```bash
# Basic installation
pip install featcopilot

# With LLM capabilities
pip install featcopilot[llm]

# Full installation
pip install featcopilot[full]
```

## Quick Start

### Fast Mode (Tabular Only)

```python
from featcopilot import AutoFeatureEngineer

# Feature engineering without LLM calls
engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50
)

X_transformed = engineer.fit_transform(X, y)
print(f"Features: {X.shape[1]} -> {X_transformed.shape[1]}")
```

### Predictive Tabular Selection (Opt-in)

Use an explicit task for integer-valued regression targets. The predictive
strategy compares feature subsets against a cloned learner on an internal
training-only repeated validation procedure; existing legacy defaults are unchanged.

```python
from sklearn.linear_model import Ridge
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=["tabular"],
    task="regression",
    selection_strategy="predictive",
    selection_estimator=Ridge(alpha=1.0),
    max_candidates=1000,
    max_features=50,
    stability_repeats=3,
    screening="auto",
)
X_train_fe = engineer.fit_transform(X_train, y_train)
X_test_fe = engineer.transform(X_test)
print(engineer.selection_report_)
```

The output preserves original categorical columns. Use appropriate downstream
encoding/imputation, and evaluate the entire pipeline inside cross-validation.
Internal validation scores are selection diagnostics, not a SOTA claim.

### Experimental engineered classifiers

`featcopilot.classification.FeatureEnsembleClassifier` couples feature
discovery with a supplied classifier using an explicit `single`,
`subsample85`, or fixed `raw_fe_average` prediction mode. Members retain their own feature
plans; predictions are averaged rather than exporting a fictitious shared
feature matrix. It does not import benchmark runners or change existing
`AutoFeatureEngineer` defaults.

See the [experimental classification guide](docs/user-guide/classification.md)
for matched raw controls, native estimator parameters, inference artifacts,
and the limited scope of current evidence.

### LLM Mode

```python
from featcopilot import AutoFeatureEngineer

# LLM-powered semantic features
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=50
)

X_transformed = engineer.fit_transform(
    X, y,
    column_descriptions={
        'age': 'Customer age in years',
        'income': 'Annual household income in USD',
        'tenure': 'Months as customer',
    },
    task_description="Predict customer churn"
)  # 30-60 seconds

# Get LLM-generated explanations
for feature, explanation in engineer.explain_features().items():
    print(f"{feature}: {explanation}")
```

## Command-Line Interface

FeatCopilot ships a `featcopilot` CLI for shell, scripting, and agentic
(LLM tool-use) workflows — no Python glue required. All subcommands accept
`--json` for machine-readable stdout; errors are written to stderr with a
non-zero exit code so agents can parse failures deterministically.

```bash
# Discover capabilities (engines, selection methods, I/O formats)
featcopilot info --json

# Run feature engineering on a CSV / JSON file
featcopilot transform \
    --input data.csv --target label --output features.csv \
    --engines tabular --max-features 50 --json

# Inspect generated features (name, explanation, code) as JSON for an LLM
featcopilot explain --input data.csv --target label

# Equivalent module form
python -m featcopilot info --json
```

Pass `--config config.json` to provide nested keys such as `llm_config`;
explicit CLI flags override values from the config file.

> **Parquet I/O.** FeatCopilot's base install does not pin a parquet engine.
> To use `--input file.parquet` / `--output file.parquet` (or the
> `parquet` value in `--input-format` / `--output-format`), install one of
> `pyarrow` or `fastparquet`. `featcopilot info --json` reports
> `"parquet_available": true` only when an engine is importable in the
> current environment.

## Engines

### Tabular Engine
Generates polynomial features, interaction terms, and mathematical transformations.

```python
from featcopilot.engines import TabularEngine

engine = TabularEngine(
    polynomial_degree=2,
    interaction_only=False,
    include_transforms=['log', 'sqrt', 'square']
)
```

### Time Series Engine
Extracts statistical, frequency, and temporal features from time series data.

```python
from featcopilot.engines import TimeSeriesEngine

engine = TimeSeriesEngine(
    features=['mean', 'std', 'skew', 'autocorr', 'fft_coefficients']
)
```

### LLM Engine
Uses GitHub Copilot SDK (default) or LiteLLM (100+ providers) for intelligent feature generation.

```python
from featcopilot.llm import SemanticEngine

# Default: GitHub Copilot SDK
engine = SemanticEngine(
    model='gpt-5.6-sol',
    max_suggestions=20,
    validate_features=True
)

# Alternative: LiteLLM backend
engine = SemanticEngine(
    model='gpt-4o',
    backend='litellm',
    max_suggestions=20
)
```

## Feature Selection

```python
from featcopilot.selection import FeatureSelector

selector = FeatureSelector(
    methods=['mutual_info', 'importance', 'correlation'],
    max_features=30,
    correlation_threshold=0.95
)

X_selected = selector.fit_transform(X, y)
```

## Comparison with Existing Libraries

| Feature | FeatCopilot | Featuretools | TSFresh | AutoFeat | OpenFE | CAAFE |
|---------|-------------|--------------|---------|----------|--------|-------|
| Tabular Features | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Time Series | ✅ | ⚠️ | ✅ | ❌ | ❌ | ❌ |
| Relational | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| LLM-Powered | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Semantic Understanding | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Code Generation | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Sklearn Compatible | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Interpretable | ✅ | ✅ | ⚠️ | ⚠️ | ❌ | ✅ |

## Documentation

📖 **Full Documentation**: [https://thinkall.github.io/featcopilot/](https://thinkall.github.io/featcopilot/)

## Requirements

- Python 3.10+
- NumPy, Pandas, Scikit-learn
- GitHub Copilot SDK (default) or LiteLLM (for 100+ LLM providers)

## License

MIT License

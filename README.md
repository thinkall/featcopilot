# FeatCopilot 🚀

**Next-Generation LLM-Powered Auto Feature Engineering with GitHub Copilot SDK**

FeatCopilot is a unified feature engineering framework that combines the best approaches from existing libraries (Featuretools, TSFresh, AutoFeat, OpenFE) with novel LLM-powered capabilities via GitHub Copilot SDK.

## 📊 Benchmark Highlights

### Tabular Engine (Fast Mode - <1s)

| Task Type | Average Improvement | Best Case |
|-----------|--------------------:|----------:|
| **Text Classification** | **+12.44%** | +49.02% (News Headlines) |
| Time Series | +1.51% | +12.12% (Retail Demand) |
| Classification | +0.54% | +4.35% |
| Regression | +0.65% | +5.57% |

### LLM Engine (With Copilot - 30-60s)

| Task Type | Average Improvement | Best Case |
|-----------|--------------------:|----------:|
| **Regression** | **+7.79%** | +19.66% (Retail Demand) |
| Classification | +2.38% | +2.87% |

- ✅ **12/12 wins** on text classification (tabular mode)
- 🧠 **+19.66% max improvement** with LLM-powered features
- ⚡ **<1 second** (tabular) or **30-60s** (with LLM) processing time
- 📈 Largest gains with simple models (LogisticRegression, Ridge)

[View Full Benchmark Results](https://thinkall.github.io/featcopilot/user-guide/benchmarks/)

## Key Features

- 🔧 **Multi-Engine Architecture**: Tabular, time series, relational, and text feature engines
- 🤖 **LLM-Powered Intelligence**: Semantic feature discovery, domain-aware generation, and code synthesis
- 📊 **Intelligent Selection**: Statistical testing, importance ranking, and redundancy elimination
- 🔌 **Scikit-learn Compatible**: Drop-in replacement for sklearn transformers
- 📝 **Interpretable**: Every feature comes with human-readable explanations

## Installation

```bash
# Basic installation
pip install featcopilot

# With LLM capabilities (requires GitHub Copilot)
pip install featcopilot[llm]

# Full installation
pip install featcopilot[full]
```

## Quick Start

### Fast Mode (Tabular Only)

```python
from featcopilot import AutoFeatureEngineer

# Sub-second feature engineering
engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50
)

X_transformed = engineer.fit_transform(X, y)  # <1 second
print(f"Features: {X.shape[1]} -> {X_transformed.shape[1]}")
```

### LLM Mode (With Copilot)

```python
from featcopilot import AutoFeatureEngineer

# LLM-powered semantic features (+19.66% max improvement)
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
Uses GitHub Copilot SDK for intelligent feature generation.

```python
from featcopilot.llm import SemanticEngine

engine = SemanticEngine(
    model='gpt-5',
    max_suggestions=20,
    validate_features=True
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
| Tabular Features | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Time Series | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Relational | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| LLM-Powered | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Semantic Understanding | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Code Generation | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ |
| Sklearn Compatible | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Interpretable | ✅ | ⚠️ | ⚠️ | ⚠️ | ❌ | ✅ |

## Documentation

📖 **Full Documentation**: [https://thinkall.github.io/featcopilot/](https://thinkall.github.io/featcopilot/)

## Requirements

- Python 3.9+
- NumPy, Pandas, Scikit-learn
- GitHub Copilot CLI (for LLM features)

## License

MIT License

# FeatCopilot 🚀

**Next-Generation LLM-Powered Auto Feature Engineering Framework**

FeatCopilot automatically generates, selects, and explains predictive features using semantic understanding. It analyzes column meanings, applies domain-aware transformations, and provides human-readable explanations—turning raw data into ML-ready features in seconds.

## 🎬 Introduction Video

[![FeatCopilot Introduction](https://img.youtube.com/vi/H7m50TLGHFk/0.jpg)](https://www.youtube.com/watch?v=H7m50TLGHFk)

## 📊 Benchmark Highlights

### Simple Models Benchmark (42 Datasets)

| Configuration | Improved | Avg Improvement | Best Improvement |
|---------------|----------|-----------------|------------------|
| **Tabular Engine** | 20 (48%) | +4.54% | +197% (delays_zurich) |
| **Tabular + LLM** | 23 (55%) | +6.12% | +420% (delays_zurich) |

Models: RandomForest (n_estimators=200, max_depth=20), LogisticRegression/Ridge

### AutoML Benchmark (FLAML, 120s budget)

| Metric | Value |
|--------|-------|
| **Datasets** | 41 |
| **Improved** | 19 (46%) |
| **Best Improvement** | +8.55% (abalone) |

### Key Results

- ✅ **+197% improvement** on delays_zurich (tabular only)
- 🧠 **+420% improvement** with LLM-enhanced features
- 📈 **+8.98%** on abalone regression task
- 🚀 **+5.68%** on complex_classification

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

# With LLM capabilities
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

### LLM Mode (With LiteLLM)

```python
from featcopilot import AutoFeatureEngineer

# LLM-powered semantic features (+420% max improvement)
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
Uses GitHub Copilot SDK (default) or LiteLLM (100+ providers) for intelligent feature generation.

```python
from featcopilot.llm import SemanticEngine

# Default: GitHub Copilot SDK
engine = SemanticEngine(
    model='gpt-5.2',
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
- GitHub Copilot SDK (default) or LiteLLM (for 100+ LLM providers)

## License

MIT License

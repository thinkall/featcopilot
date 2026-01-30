# FeatCopilot 🚀

**Next-Generation LLM-Powered Auto Feature Engineering with GitHub Copilot SDK**

FeatCopilot is a unified feature engineering framework that combines the best approaches from existing libraries (Featuretools, TSFresh, AutoFeat, OpenFE) with novel LLM-powered capabilities via GitHub Copilot SDK.

## 📊 Benchmark Highlights

| Task Type | Average Improvement | Best Case |
|-----------|--------------------:|----------:|
| **Text Classification** | **+12.44%** | +49.02% (News Headlines) |
| Classification | +0.54% | +4.35% |
| Regression | +0.65% | +5.57% |

- ✅ **12/12 wins** on text/semantic classification tasks
- ⚡ **<1 second** feature engineering time for most datasets
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

### Basic Usage

```python
from featcopilot import AutoFeatureEngineer

# Initialize the engineer
engineer = AutoFeatureEngineer(
    engines=['tabular', 'timeseries'],
    max_features=50
)

# Fit and transform
X_transformed = engineer.fit_transform(X, y)

# Get feature names and importances
print(engineer.get_feature_names())
print(engineer.feature_importances_)
```

### LLM-Powered Feature Engineering

```python
from featcopilot import AutoFeatureEngineer

# Enable LLM capabilities
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'enable_semantic': True,
        'enable_code_gen': True,
        'model': 'gpt-5'
    }
)

# Provide column descriptions for semantic understanding
X_transformed = engineer.fit_transform(
    X, y,
    column_descriptions={
        'age': 'Customer age in years',
        'income': 'Annual household income in USD',
        'tenure': 'Months as customer',
        'last_purchase': 'Date of most recent purchase'
    },
    task_description="Predict customer churn for telecom company"
)

# Get LLM-generated explanations
for feature, explanation in engineer.explain_features().items():
    print(f"{feature}: {explanation}")
```

### Generate Custom Features

```python
# Let LLM suggest domain-specific features
custom_features = engineer.generate_custom_features(
    domain="healthcare",
    task="Predict patient readmission risk",
    existing_columns=['age', 'diagnosis_code', 'length_of_stay', 'num_medications']
)

# Apply the generated features
X_with_custom = engineer.apply_custom_features(X, custom_features)
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

## Requirements

- Python 3.9+
- NumPy, Pandas, Scikit-learn
- GitHub Copilot CLI (for LLM features)

## License

MIT License

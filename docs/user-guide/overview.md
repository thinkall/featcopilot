# Overview

FeatCopilot provides a unified framework for automated feature engineering, combining multiple approaches into a single, easy-to-use API.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AutoFeatureEngineer                       │
│                   (Main Entry Point)                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Tabular  │  │TimeSeries│  │Relational│  │   LLM    │    │
│  │ Engine   │  │  Engine  │  │  Engine  │  │ Engine   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       └─────────────┴─────────────┴─────────────┘           │
│                          │                                   │
│                  Feature Generation                          │
│                          │                                   │
│              ┌───────────┴───────────┐                      │
│              │   Feature Selection   │                      │
│              │  (Statistical + ML)   │                      │
│              └───────────┬───────────┘                      │
│                          │                                   │
│                   Selected Features                          │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Engines

Engines are responsible for generating new features from input data:

| Engine | Purpose | Key Features |
|--------|---------|--------------|
| **TabularEngine** | Numeric feature transformation | Polynomial, interactions, math transforms |
| **TimeSeriesEngine** | Time series feature extraction | Statistics, autocorrelation, FFT |
| **RelationalEngine** | Multi-table aggregation | Joins, aggregations, group-by |
| **TextEngine** | Text feature extraction | Length stats, TF-IDF, embeddings |
| **SemanticEngine** | LLM-powered generation | Semantic understanding, code gen |

### 2. Feature Selection

After generation, features are ranked and selected:

- **Statistical Selection**: Mutual information, F-test, chi-square
- **Model-based Selection**: Random Forest importance, XGBoost
- **Redundancy Elimination**: Correlation-based filtering

### 3. Feature Representation

Every feature includes metadata:

```python
from featcopilot.core import Feature, FeatureType, FeatureOrigin

feature = Feature(
    name="age_income_ratio",
    dtype=FeatureType.NUMERIC,
    origin=FeatureOrigin.LLM_GENERATED,
    source_columns=["age", "income"],
    explanation="Ratio indicating financial maturity",
    code="result = df['age'] / (df['income'] + 1e-8)"
)
```

## Workflow

### Basic Workflow

```python
from featcopilot import AutoFeatureEngineer

# 1. Initialize
engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50
)

# 2. Fit (learns from data)
engineer.fit(X_train, y_train)

# 3. Transform (generates features)
X_train_fe = engineer.transform(X_train)
X_test_fe = engineer.transform(X_test)
```

### LLM-Enhanced Workflow

```python
# 1. Initialize with LLM
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={'model': 'gpt-5', 'domain': 'finance'}
)

# 2. Fit with context
engineer.fit(
    X_train, y_train,
    column_descriptions={...},
    task_description="Predict loan default"
)

# 3. Get explanations
explanations = engineer.explain_features()

# 4. Generate custom features
custom = engineer.generate_custom_features(
    prompt="Create risk indicators"
)
```

## Design Principles

### 1. Modularity

Each component can be used independently:

```python
# Use just the tabular engine
from featcopilot.engines import TabularEngine

engine = TabularEngine(polynomial_degree=2)
X_fe = engine.fit_transform(X)
```

### 2. Sklearn Compatibility

Works with sklearn pipelines:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('features', AutoFeatureEngineer(engines=['tabular'])),
    ('model', RandomForestClassifier())
])
```

### 3. Interpretability

Every feature has an explanation:

```python
for name, explanation in engineer.explain_features().items():
    print(f"{name}: {explanation}")
```

### 4. Graceful Degradation

LLM features fall back to heuristics when unavailable:

```python
# Works even without Copilot authentication
engineer = AutoFeatureEngineer(engines=['llm'])
# Warning: Using mock LLM responses
```

## Configuration

### Engine Configuration

```python
AutoFeatureEngineer(
    engines=['tabular', 'timeseries', 'llm'],
    max_features=100,
    selection_methods=['mutual_info', 'importance'],
    correlation_threshold=0.95,
    verbose=True
)
```

### LLM Configuration

```python
llm_config = {
    'model': 'gpt-5',           # Model to use
    'max_suggestions': 20,       # Features to suggest
    'domain': 'healthcare',      # Domain context
    'validate_features': True    # Validate generated code
}
```

## Next Steps

- [Learn about individual engines](engines.md)
- [Explore LLM features](llm-features.md)
- [Understand feature selection](selection.md)

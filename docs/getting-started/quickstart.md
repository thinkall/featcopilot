# Quick Start

Get up and running with FeatCopilot in 5 minutes.

## Basic Feature Engineering

```python
import pandas as pd
import numpy as np
from featcopilot import AutoFeatureEngineer

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.exponential(50000, 1000),
    'tenure': np.random.randint(1, 120, 1000),
})
y = (X['income'] > 50000).astype(int)

# Initialize feature engineer
engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=30
)

# Fit and transform
X_transformed = engineer.fit_transform(X, y)

print(f"Original features: {len(X.columns)}")
print(f"Transformed features: {len(X_transformed.columns)}")
```

**Output:**
```
Original features: 3
Transformed features: 30
```

## Understanding Generated Features

```python
# Get feature names
print(engineer.get_feature_names()[:10])
```

**Output:**
```
['age_pow2', 'income_pow2', 'tenure_pow2', 'age_x_income', 
 'age_x_tenure', 'income_x_tenure', 'age_log1p', 'income_log1p', 
 'tenure_log1p', 'age_sqrt']
```

## Feature Selection

Features are automatically selected based on importance:

```python
# Get feature importance scores
if engineer.feature_importances_:
    top_features = sorted(
        engineer.feature_importances_.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    for name, score in top_features:
        print(f"{name}: {score:.4f}")
```

## Using in Sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Create pipeline
pipeline = Pipeline([
    ('features', AutoFeatureEngineer(engines=['tabular'], max_features=20)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Note: For pipelines, use fit() then transform() separately
# or handle NaN values in the pipeline
engineer = AutoFeatureEngineer(engines=['tabular'], max_features=20)
X_fe = engineer.fit_transform(X, y).fillna(0)

# Evaluate
scores = cross_val_score(
    LogisticRegression(), 
    X_fe, y, 
    cv=5, 
    scoring='roc_auc'
)
print(f"ROC-AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

## Adding LLM-Powered Features

```python
from featcopilot import AutoFeatureEngineer

# Enable LLM engine
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=40,
    llm_config={
        'model': 'gpt-5',
        'max_suggestions': 10
    }
)

# Provide context for better features
X_transformed = engineer.fit_transform(
    X, y,
    column_descriptions={
        'age': 'Customer age in years',
        'income': 'Annual income in USD',
        'tenure': 'Months as customer'
    },
    task_description='Predict customer churn'
)

# Get explanations for features
explanations = engineer.explain_features()
for feat, expl in list(explanations.items())[:3]:
    print(f"{feat}: {expl}")
```

## Next Steps

- [Learn about different engines](../user-guide/engines.md)
- [Explore LLM features in depth](../user-guide/llm-features.md)
- [See more examples](../examples/basic.md)
- [Set up authentication for LLM](authentication.md)

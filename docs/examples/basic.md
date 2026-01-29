# Basic Usage Example

A complete example demonstrating basic feature engineering with FeatCopilot.

## The Problem

We have a customer churn dataset and want to:

1. Generate new features automatically
2. Select the most important features
3. Train a classifier

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

from featcopilot import AutoFeatureEngineer
```

## Create Sample Data

```python
def create_churn_data(n_samples=1000):
    """Create synthetic customer churn dataset."""
    np.random.seed(42)

    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'tenure_months': np.random.randint(1, 120, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.exponential(2000, n_samples),
        'contract_length': np.random.choice([1, 12, 24], n_samples),
        'num_products': np.random.randint(1, 6, n_samples),
        'support_tickets': np.random.poisson(2, n_samples),
    })

    # Create target with some signal
    churn_prob = (
        0.3
        - 0.002 * data['tenure_months']
        + 0.001 * data['monthly_charges']
        + 0.05 * data['support_tickets']
        - 0.01 * data['contract_length']
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    data['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)

    return data

# Create data
data = create_churn_data(1000)
X = data.drop('churn', axis=1)
y = data['churn']

print(f"Dataset shape: {X.shape}")
print(f"Churn rate: {y.mean():.2%}")
```

**Output:**
```
Dataset shape: (1000, 8)
Churn rate: 32.40%
```

## Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
```

## Baseline Model

```python
# Train without feature engineering
baseline = RandomForestClassifier(n_estimators=100, random_state=42)
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, baseline_pred)

print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
```

**Output:**
```
Baseline ROC-AUC: 0.6955
```

## Feature Engineering

```python
# Initialize feature engineer
engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50,
    selection_methods=['mutual_info', 'importance'],
    correlation_threshold=0.95,
    verbose=True
)

# Fit and transform training data
X_train_fe = engineer.fit_transform(X_train, y_train)

# Transform test data
X_test_fe = engineer.transform(X_test)

# Align columns (important!)
common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
X_train_fe = X_train_fe[common_cols].fillna(0)
X_test_fe = X_test_fe[common_cols].fillna(0)

print(f"\nOriginal features: {len(X_train.columns)}")
print(f"Engineered features: {len(X_train_fe.columns)}")
```

**Output:**
```
TabularEngine: Found 7 numeric columns
TabularEngine: Planned 91 features
TabularEngine: Generated 50 features
StatisticalSelector: Selected 58 features
ImportanceSelector: Selected 58 features
RedundancyEliminator: Removed 20 redundant features
FeatureSelector: Selected 38 features

Original features: 8
Engineered features: 38
```

## Model with Engineered Features

```python
# Train with engineered features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_fe, y_train)

pred = model.predict_proba(X_test_fe)[:, 1]
auc = roc_auc_score(y_test, pred)

print(f"Engineered ROC-AUC: {auc:.4f}")
print(f"Improvement: {(auc - baseline_auc) * 100:.2f}%")
```

## Feature Importance

```python
# Get feature importance from selector
if engineer.feature_importances_:
    importances = sorted(
        engineer.feature_importances_.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 10 Features:")
    for i, (name, score) in enumerate(importances[:10], 1):
        print(f"{i:2d}. {name}: {score:.4f}")
```

**Output:**
```
Top 10 Features:
 1. tenure_months_sqrt: 0.9144
 2. age_x_total_charges: 0.8987
 3. contract_length: 0.8240
 4. age_x_tenure_months: 0.6936
 5. total_charges_x_num_products: 0.6394
...
```

## Complete Script

```python
"""
Complete FeatCopilot Basic Example
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from featcopilot import AutoFeatureEngineer

# Create data
np.random.seed(42)
n = 1000
X = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'income': np.random.exponential(50000, n),
    'tenure': np.random.randint(1, 120, n),
    'charges': np.random.uniform(20, 150, n),
})
y = (X['income'] > 50000).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature engineering
engineer = AutoFeatureEngineer(engines=['tabular'], max_features=30)
X_train_fe = engineer.fit_transform(X_train, y_train).fillna(0)
X_test_fe = engineer.transform(X_test).fillna(0)

# Align columns
cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
X_train_fe, X_test_fe = X_train_fe[cols], X_test_fe[cols]

# Train and evaluate
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_fe, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test_fe)[:, 1])

print(f"ROC-AUC: {auc:.4f}")
```

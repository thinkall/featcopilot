# Feature Store Integration

FeatCopilot integrates with feature stores to enable feature reuse, versioning, and serving in production ML systems.

## Overview

Feature stores provide:

- **Feature Reuse**: Share engineered features across teams and projects
- **Online Serving**: Low-latency feature retrieval for real-time inference
- **Offline Storage**: Historical features for training and batch predictions
- **Feature Discovery**: Browse and search available features
- **Versioning**: Track feature definitions and values over time

## Supported Feature Stores

| Feature Store | Status | Install |
|--------------|--------|---------|
| [Feast](#feast) | ✅ Supported | `pip install featcopilot[feast]` |
| Tecton | 🔜 Planned | - |
| AWS SageMaker Feature Store | 🔜 Planned | - |
| Databricks Feature Store | 🔜 Planned | - |
| Vertex AI Feature Store | 🔜 Planned | - |

## Feast Integration

[Feast](https://feast.dev) is an open-source feature store that works with multiple backends.

### Installation

```bash
pip install featcopilot[feast]
```

### Quick Start

```python
from featcopilot import AutoFeatureEngineer
from featcopilot.stores import FeastFeatureStore

# 1. Generate features with FeatCopilot
engineer = AutoFeatureEngineer(engines=['tabular'])
X_transformed = engineer.fit_transform(X, y)

# 2. Add entity column and timestamp
X_transformed['customer_id'] = X['customer_id']
X_transformed['event_timestamp'] = datetime.now()

# 3. Save to Feast
store = FeastFeatureStore(
    repo_path='./feature_repo',
    entity_columns=['customer_id'],
    timestamp_column='event_timestamp'
)
store.initialize()

store.save_features(
    df=X_transformed,
    feature_view_name='customer_features',
    description='Customer churn prediction features'
)
```

### Configuration

```python
from featcopilot.stores import FeastFeatureStore

store = FeastFeatureStore(
    # Repository path (created if doesn't exist)
    repo_path='./feature_repo',

    # Feast project name
    project_name='my_project',

    # Entity columns (keys that identify each row)
    entity_columns=['customer_id'],

    # Timestamp column for point-in-time joins
    timestamp_column='event_timestamp',

    # Provider: 'local', 'gcp', 'aws'
    provider='local',

    # Online store type: 'sqlite', 'redis', 'dynamodb'
    online_store_type='sqlite',

    # Offline store type: 'file', 'bigquery', 'redshift'
    offline_store_type='file',

    # Feature time-to-live in days
    ttl_days=365,

    # Auto-sync to online store after save
    auto_materialize=True,

    # Tags for feature discovery
    tags={'team': 'ml', 'domain': 'churn'}
)
```

### Saving Features

```python
# Basic save
store.save_features(
    df=X_transformed,
    feature_view_name='customer_features'
)

# With metadata
store.save_features(
    df=X_transformed,
    feature_view_name='customer_features',
    description='Features for customer churn prediction',
    entity_columns=['customer_id'],  # Override config
    timestamp_column='event_timestamp'  # Override config
)
```

### Retrieving Features

#### Offline Store (Training)

Use the offline store for historical feature retrieval during training:

```python
# Entity DataFrame with timestamps for point-in-time join
entity_df = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'event_timestamp': [datetime(2024, 1, 1)] * 3
})

# Get features
features = store.get_features(
    entity_df=entity_df,
    feature_names=['age_income_ratio', 'tenure_months', 'total_purchases'],
    feature_view_name='customer_features',
    online=False  # Use offline store
)
```

#### Online Store (Inference)

Use the online store for low-latency feature retrieval during inference:

```python
# Real-time feature retrieval
features = store.get_online_features(
    entity_dict={'customer_id': [1, 2, 3]},
    feature_names=['age_income_ratio', 'tenure_months'],
    feature_view_name='customer_features'
)

# Returns dict: {'customer_id': [1, 2, 3], 'age_income_ratio': [...], ...}
```

### Pushing Real-Time Updates

For streaming scenarios, push new feature values directly to the online store:

```python
# New feature values
new_data = pd.DataFrame({
    'customer_id': [1001],
    'event_timestamp': [datetime.now()],
    'age_income_ratio': [0.45],
    'tenure_months': [24]
})

# Push to online store
store.push_features(new_data, feature_view_name='customer_features')
```

### Managing Feature Views

```python
# List all feature views
views = store.list_feature_views()
print(views)  # ['customer_features', 'product_features', ...]

# Get schema/metadata
schema = store.get_feature_view_schema('customer_features')
print(schema)
# {
#     'name': 'customer_features',
#     'entities': ['customer_id'],
#     'features': [{'name': 'age_income_ratio', 'dtype': 'DOUBLE'}, ...],
#     'ttl': '365 days',
#     'description': 'Features for customer churn prediction'
# }

# Delete a feature view
store.delete_feature_view('old_features')
```

### Production Setup

#### Redis Online Store

```python
store = FeastFeatureStore(
    repo_path='./feature_repo',
    entity_columns=['customer_id'],
    online_store_type='redis',
    # Set REDIS_CONNECTION_STRING env var or configure in feature_store.yaml
)
```

#### BigQuery Offline Store (GCP)

```python
store = FeastFeatureStore(
    repo_path='./feature_repo',
    entity_columns=['customer_id'],
    provider='gcp',
    offline_store_type='bigquery',
    # Set GCP credentials via GOOGLE_APPLICATION_CREDENTIALS
)
```

#### S3/Redshift (AWS)

```python
store = FeastFeatureStore(
    repo_path='./feature_repo',
    entity_columns=['customer_id'],
    provider='aws',
    offline_store_type='redshift',
    # Set AWS credentials via environment variables
)
```

## Complete Example

```python
"""
End-to-end example: Feature engineering + Feast
"""
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from featcopilot import AutoFeatureEngineer
from featcopilot.stores import FeastFeatureStore

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'customer_id': range(1, 1001),
    'event_timestamp': [datetime.now()] * 1000,
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.uniform(20000, 150000, 1000),
    'tenure_months': np.random.randint(1, 120, 1000),
})
data['churned'] = (np.random.random(1000) < 0.3).astype(int)

# Split data
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Generate features
feature_cols = ['age', 'income', 'tenure_months']
engineer = AutoFeatureEngineer(engines=['tabular'], max_features=20)

X_train_fe = engineer.fit_transform(train_df[feature_cols], train_df['churned'])
X_test_fe = engineer.transform(test_df[feature_cols])

# Add entity columns back
X_train_fe['customer_id'] = train_df['customer_id'].values
X_train_fe['event_timestamp'] = train_df['event_timestamp'].values

# Save to Feast
store = FeastFeatureStore(
    repo_path='./churn_feature_repo',
    entity_columns=['customer_id'],
    timestamp_column='event_timestamp'
)
store.initialize()

store.save_features(
    df=X_train_fe,
    feature_view_name='churn_features',
    description='Customer churn prediction features from FeatCopilot'
)

# Train model
feature_names = [c for c in X_train_fe.columns if c not in ['customer_id', 'event_timestamp']]
model = RandomForestClassifier(random_state=42)
model.fit(X_train_fe[feature_names], train_df['churned'])

# For inference, get features from online store
inference_features = store.get_online_features(
    entity_dict={'customer_id': [1, 2, 3]},
    feature_names=feature_names[:5],
    feature_view_name='churn_features'
)
print(f"Online features: {inference_features}")

# Cleanup
store.close()
```

## Best Practices

### 1. Use Meaningful Entity Keys

```python
# ✅ Good: Clear entity identification
entity_columns=['customer_id', 'product_id']

# ❌ Bad: Using row index
entity_columns=['index']
```

### 2. Include Event Timestamps

```python
# ✅ Good: Proper timestamp for point-in-time correctness
df['event_timestamp'] = df['transaction_date']

# ❌ Bad: Using current time for historical data
df['event_timestamp'] = datetime.now()
```

### 3. Set Appropriate TTL

```python
# Short-lived features (e.g., real-time signals)
ttl_days=7

# Long-lived features (e.g., customer demographics)
ttl_days=365
```

### 4. Use Tags for Discovery

```python
store = FeastFeatureStore(
    ...,
    tags={
        'team': 'data-science',
        'domain': 'customer-360',
        'model': 'churn-v2',
        'created_by': 'featcopilot'
    }
)
```

### 5. Materialize Before Inference

```python
# Ensure features are in online store before real-time inference
store.save_features(df, feature_view_name='features', auto_materialize=True)

# Or manually materialize
# feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

## Troubleshooting

### "Entity column not found"

Ensure entity columns exist in your DataFrame:

```python
# Check columns
print(df.columns)

# Add if missing
df['customer_id'] = your_ids
```

### "Feast not installed"

Install Feast:

```bash
pip install featcopilot[feast]
# or
pip install feast
```

### "Features not in online store"

Materialize features:

```python
store = FeastFeatureStore(..., auto_materialize=True)
# or run: feast materialize-incremental
```

### "Point-in-time join returns nulls"

Ensure timestamps align:

```python
# Entity timestamps should be >= feature timestamps
entity_df['event_timestamp'] = datetime.now()  # Current time
# Features should have earlier timestamps
```

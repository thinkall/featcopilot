# Benchmark Datasets for FeatCopilot

This module provides a unified API for loading benchmark datasets organized by category.

## Quick Start

```python
from benchmarks.datasets import (
    list_datasets,
    load_dataset,
    load_datasets,
    load_all_datasets,
    CATEGORY_CLASSIFICATION,
    CATEGORY_REGRESSION,
    CATEGORY_FORECASTING,
    CATEGORY_TEXT,
)

# List all available datasets
all_names = list_datasets()  # 52 datasets

# List by category
classification_names = list_datasets(CATEGORY_CLASSIFICATION)  # 22 datasets
regression_names = list_datasets(CATEGORY_REGRESSION)  # 20 datasets

# Load a single dataset by name
X, y, task, name = load_dataset('titanic')
X, y, task, name = load_dataset('diamonds')  # INRIA dataset

# Load all datasets in a category
datasets = load_datasets(CATEGORY_CLASSIFICATION)
for X, y, task, name in datasets:
    print(f"{name}: {X.shape}")

# Load ALL datasets
all_data = load_all_datasets()
```

---

## Dataset Categories

| Category | Count | Description |
|----------|-------|-------------|
| `classification` | 22 | Binary and multi-class classification |
| `regression` | 20 | Continuous target prediction |
| `forecasting` | 3 | Time series forecasting |
| `text` | 7 | Text/NLP datasets |

### Classification Datasets

```python
list_datasets('classification')
```

| Name | Description | Source |
|------|-------------|--------|
| titanic | Titanic survival | Synthetic |
| credit_card_fraud | Fraud detection | Synthetic |
| employee_attrition | Employee attrition | Synthetic |
| credit_risk | Credit risk assessment | Synthetic |
| medical_diagnosis | Medical diagnosis | Synthetic |
| customer_churn | Customer churn | Synthetic |
| higgs | Higgs boson detection | INRIA |
| covertype | Forest cover type | INRIA |
| electricity | Electricity price direction | INRIA |
| credit | Credit approval | INRIA |
| ... | +12 more INRIA datasets | INRIA |

### Regression Datasets

```python
list_datasets('regression')
```

| Name | Description | Source |
|------|-------------|--------|
| house_prices | House price prediction | Synthetic |
| bike_sharing | Bike sharing demand | Synthetic |
| insurance_claims | Insurance claim amount | Synthetic |
| spotify_tracks | Spotify popularity | HuggingFace |
| diamonds | Diamond price | INRIA |
| wine_quality | Wine quality score | INRIA |
| abalone | Abalone age | INRIA |
| ... | +13 more datasets | Various |

### Forecasting (Time Series) Datasets

```python
list_datasets('forecasting')
```

| Name | Description |
|------|-------------|
| sensor_anomaly | Sensor efficiency time series |
| retail_demand | Retail demand forecasting |
| server_latency | Server latency prediction |

### Text Datasets

```python
list_datasets('text')
```

| Name | Description |
|------|-------------|
| product_reviews | Product review sentiment |
| job_postings | Job posting salary prediction |
| news_classification | News headline classification |
| customer_support | Support ticket priority |
| medical_notes | Medical notes classification |
| ecommerce_product | E-commerce product sales |
| fake_news | Fake news detection (HuggingFace) |

---

## API Reference

### `list_datasets(category=None)`

List available dataset names.

```python
# All datasets
names = list_datasets()

# By category
names = list_datasets('classification')
names = list_datasets('regression')
names = list_datasets('forecasting')
names = list_datasets('text')
```

### `load_dataset(name, **kwargs)`

Load a single dataset by name.

```python
X, y, task, name = load_dataset('titanic')
X, y, task, name = load_dataset('diamonds')  # INRIA
X, y, task, name = load_dataset('spotify_tracks', max_samples=10000)
```

**Returns:** `(X, y, task_type, dataset_name)` tuple

### `load_datasets(category=None, **kwargs)`

Load all datasets in a category.

```python
# Load all classification datasets
clf_datasets = load_datasets('classification')

# Load all datasets
all_datasets = load_datasets()
```

**Returns:** List of `(X, y, task_type, dataset_name)` tuples

### `load_all_datasets(**kwargs)`

Load all available datasets.

```python
all_data = load_all_datasets()
print(f"Loaded {len(all_data)} datasets")
```

### `get_dataset_info(name)`

Get metadata about a dataset.

```python
info = get_dataset_info('customer_churn')
# {'name': 'customer_churn', 'category': 'classification',
#  'description': 'Customer churn prediction', 'loader': <function>}
```

### `get_category_summary()`

Get count of datasets per category.

```python
summary = get_category_summary()
# {'classification': 22, 'regression': 20, 'forecasting': 3, 'text': 7}
```

---

## INRIA-SODA Benchmark Datasets

30 real-world datasets from [inria-soda/tabular-benchmark](https://huggingface.co/datasets/inria-soda/tabular-benchmark):

```python
from benchmarks.datasets import INRIA_DATASETS, load_inria_dataset

# List available INRIA datasets
print(list(INRIA_DATASETS.keys()))

# Load specific INRIA dataset
X, y, task, name = load_dataset('diamonds')
X, y, task, name = load_dataset('higgs')
```

---

## Usage with FeatCopilot

```python
from benchmarks.datasets import load_dataset, load_datasets
from featcopilot import AutoFeatureEngineer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y, task, name = load_dataset('customer_churn')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply FeatCopilot
engineer = AutoFeatureEngineer(engines=['tabular'], max_features=50)
X_train_fe = engineer.fit_transform(X_train, y_train)
X_test_fe = engineer.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_fe, y_train)
print(f"Accuracy: {model.score(X_test_fe, y_test):.4f}")
```

### Batch Benchmarking

```python
from benchmarks.datasets import load_datasets

# Run on all classification datasets
for X, y, task, name in load_datasets('classification'):
    print(f"\n{name}:")
    # ... benchmark code
```

---

## Legacy API

For backward compatibility, these functions are still available:

```python
from benchmarks.datasets import (
    get_all_datasets,      # Returns list of loader functions
    get_timeseries_datasets,
    get_text_datasets,
    load_dataset_by_name,  # Alias for load_dataset()
)
```

# Best Practices

Guidelines for getting the most out of FeatCopilot.

## Data Preparation

### Keep Learned Preprocessing Inside the Pipeline

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from featcopilot import FeatureEngineerTransformer

pipeline = Pipeline([
    ('features', FeatureEngineerTransformer(
        engine='tabular',
        max_features=30,
        keep_original_categorical=False,
    )),
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000)),
])
```

Imputation, encoding, feature generation, and scaling should be fitted on training
rows only. Putting them in one pipeline preserves that rule during
cross-validation and grid search.

### Identify Column Types

```python
# Numeric columns are processed by TabularEngine
numeric_cols = X.select_dtypes(include=['number']).columns

# Text columns are processed by TextEngine
text_cols = X.select_dtypes(include=['object']).columns
```

## Feature Engineering

### Start Simple

```python
# Start with tabular engine only
engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=30
)

# Add complexity if needed
engineer = AutoFeatureEngineer(
    engines=['tabular', 'timeseries', 'llm'],
    max_features=50
)
```

### Limit Feature Explosion

```python
# Too many features can cause:
# - Overfitting
# - Slow training
# - Memory issues

engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50,  # Reasonable limit
    correlation_threshold=0.95  # Remove redundant
)
```

### Use Domain Knowledge

```python
# LLM features work best with context
engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config={
        'domain': 'healthcare',  # Specify domain
        'max_suggestions': 15
    }
)

X_fe = engineer.fit_transform(
    X, y,
    column_descriptions={...},  # Describe columns
    task_description="..."       # Describe task
)
```

## Model Integration

### Sklearn Pipeline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from featcopilot import FeatureEngineerTransformer

pipeline = Pipeline([
    ('features', FeatureEngineerTransformer(
        engine='tabular',
        max_features=30,
        keep_original_categorical=False,
    )),
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('model', RandomForestClassifier(random_state=42)),
])
```

`keep_original_categorical=False` prevents unencoded string columns from
reaching estimators that require numeric input. One-hot columns use categories
learned during `fit`; target-encoded training rows are cross-fitted so each row
is encoded without its own target.

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

from featcopilot import AutoFeatureEngineer

# Always split BEFORE feature engineering
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit on training data only
engineer = AutoFeatureEngineer(engines=['tabular'])
X_train_fe = engineer.fit_transform(X_train, y_train)

# Transform test data (no y needed)
X_test_fe = engineer.transform(X_test)
```

`fit_transform` may intentionally produce different target-encoded values than
`transform` on the same rows: training values are out-of-fold, while later
transforms use mappings learned from the full training set.

### Cross-Validation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from featcopilot import FeatureEngineerTransformer

pipeline = Pipeline([
    ('features', FeatureEngineerTransformer(
        engine='tabular',
        max_features=30,
        keep_original_categorical=False,
    )),
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('model', RandomForestClassifier(random_state=42)),
])

scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
```

## LLM Features

### Provide Rich Context

```python
# ❌ Poor context
engineer.fit_transform(X, y)

# ✅ Rich context
engineer.fit_transform(
    X, y,
    column_descriptions={
        'col1': 'Detailed description...',
        'col2': 'Another description...'
    },
    task_description="""
    Detailed task description including:
    - What we're predicting
    - Business context
    - Important considerations
    """
)
```

### Review Generated Code

```python
# Always review before production
for name, code in engineer.get_feature_code().items():
    print(f"# {name}")
    print(code)
    # Check for:
    # - Division by zero handling
    # - Edge cases
    # - Correct column references
```

### Validate Features

```python
# Enable validation (default)
llm_config = {'validate_features': True}

# Check for invalid features in verbose mode
engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config=llm_config,
    verbose=True  # See validation results
)
```

## Performance

### Large Datasets

```python
# For large datasets, limit features aggressively
engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=30,  # Keep it small
)

# Or sample for fitting
sample_idx = np.random.choice(len(X), 10000, replace=False)
engineer.fit(X.iloc[sample_idx], y.iloc[sample_idx])
X_fe = engineer.transform(X)  # Transform full data
```

### Memory Management

```python
# Process in batches for very large data
def batch_transform(engineer, X, batch_size=10000):
    results = []
    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i+batch_size]
        results.append(engineer.transform(batch))
    return pd.concat(results)
```

### Caching

```python
from featcopilot.utils import FeatureCache

# Cache expensive computations
cache = FeatureCache(cache_dir='.feature_cache')

cache_key = 'my_features'
if cache.has(cache_key):
    X_fe = cache.get(cache_key)
else:
    X_fe = engineer.fit_transform(X, y)
    cache.set(cache_key, X_fe)
```

## Debugging

### Verbose Mode

```python
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    verbose=True  # See detailed output
)
```

### Check Generated Features

```python
# See what was generated
print(f"Features generated: {len(engineer.get_feature_names())}")
print(engineer.get_feature_names()[:10])

# Check for issues
X_fe = engineer.transform(X)
print(f"NaN count: {X_fe.isna().sum().sum()}")
print(f"Inf count: {np.isinf(X_fe.select_dtypes('number')).sum().sum()}")
```

### Feature Statistics

```python
# Quick sanity check
X_fe = engineer.fit_transform(X, y)
print(X_fe.describe())
```

## Common Pitfalls

### 1. Data Leakage

```python
# ❌ Wrong: engineer once, then cross-validate the already transformed data
X_fe = engineer.fit_transform(X, y)
scores = cross_val_score(model, X_fe, y, cv=5)

# ✅ Correct: cross-validate the complete pipeline
scores = cross_val_score(pipeline, X, y, cv=5)
```

### 2. Feature Explosion

```python
# ❌ Too many features
engineer = AutoFeatureEngineer(
    polynomial_degree=4,  # 4th degree polynomials
    max_features=None     # No limit
)

# ✅ Controlled generation
engineer = AutoFeatureEngineer(
    polynomial_degree=2,
    max_features=50
)
```

### 3. Ignoring NaN Values

```python
# Generated features may have NaN
X_fe = engineer.fit_transform(X, y)

# Always handle before modeling
X_fe = X_fe.fillna(0)  # or
X_fe = X_fe.fillna(X_fe.median())
```

### 4. Not Validating LLM Features

```python
# LLM-generated code may have errors
# Always validate in development
llm_config = {'validate_features': True}
```

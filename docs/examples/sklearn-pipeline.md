# Sklearn Pipeline Example

Integrate FeatCopilot into scikit-learn pipelines.

## Basic Pipeline

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from featcopilot import FeatureEngineerTransformer

# Intermediate pipeline steps are cloned for each cross-validation fold.
pipeline = Pipeline([
    ('features', FeatureEngineerTransformer(
        engine='tabular',
        max_features=20,
        keep_original_categorical=False,
    )),
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000)),
])

scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
```

## Handling NaN Values

FeatCopilot may generate features with NaN values. Handle them in the pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from featcopilot import FeatureEngineerTransformer

# Pipeline with imputation
pipeline = Pipeline([
    ('features', FeatureEngineerTransformer(
        engine='tabular',
        max_features=30,
        keep_original_categorical=False,
    )),
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
])
```

Target-encoded columns use out-of-fold values during `fit_transform`, while
future `transform` calls use mappings learned from the full training set.
Unseen categories fall back to the training-target mean.

## Two-Stage Approach

For more control, separate feature engineering from modeling:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from featcopilot import AutoFeatureEngineer

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'a': np.random.randn(1000),
    'b': np.random.randn(1000),
    'c': np.random.randn(1000),
})
y = (X['a'] + X['b'] > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Stage 1: Feature Engineering
engineer = AutoFeatureEngineer(engines=['tabular'], max_features=20)
X_train_fe = engineer.fit_transform(X_train, y_train).fillna(0)
X_test_fe = engineer.transform(X_test).fillna(0)

# Stage 2: fit one final model for the held-out test set
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000)),
])

model_pipeline.fit(X_train_fe, y_train)
test_probability = model_pipeline.predict_proba(X_test_fe)[:, 1]
print(f"Test ROC-AUC: {roc_auc_score(y_test, test_probability):.4f}")
```

Do not run cross-validation on `X_train_fe` from this two-stage pattern: the
feature engineer has already seen all of `y_train`. Use the complete pipeline
from the previous section for cross-validation or hyperparameter search.

## Grid Search with Feature Engineering

```python
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from featcopilot import AutoFeatureEngineer

pipeline = Pipeline([
    ('features', AutoFeatureEngineer(engines=['tabular'])),
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000)),
])

param_grid = {
    'features__max_features': [20, 30, 50],
    'classifier__C': [0.1, 1, 10],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X, y)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

This example assumes numeric input. For mixed-type data, use the
`FeatureEngineerTransformer` pipeline above or a `ColumnTransformer`.

## Column Transformer Integration

Combine FeatCopilot with other transformers:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from featcopilot import FeatureEngineerTransformer

# Identify column types
numeric_features = ['age', 'income', 'tenure']
categorical_features = ['category', 'region']

# Create transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('features', FeatureEngineerTransformer(
                engine='tabular',
                polynomial_degree=2,
                encode_categorical=False,
            )),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

## FeatureEngineerTransformer

Use the built-in `FeatureEngineerTransformer` for single-engine pipelines:

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from featcopilot import FeatureEngineerTransformer

# Single engine transformer
pipeline = Pipeline([
    ('features', FeatureEngineerTransformer(
        engine='tabular',
        max_features=30,
        keep_original_categorical=False,
    )),
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000)),
])

# With time series engine
ts_pipeline = Pipeline([
    ('features', FeatureEngineerTransformer(
        engine='timeseries',
        window_sizes=[3, 7, 14],
        series_in_rows=True,
    )),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000)),
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Cross-Validation with Proper Leakage Prevention

```python
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring='roc_auc',
)
print(f"CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

Passing the complete pipeline to `cross_val_score` ensures every transformer is
refit inside each fold. Precomputing `X_fe = engineer.fit_transform(X, y)` before
cross-validation leaks learned encodings and selection decisions across folds.

# Sklearn Pipeline Example

Integrate FeatCopilot into scikit-learn pipelines.

## Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from featcopilot import AutoFeatureEngineer

# Create pipeline
pipeline = Pipeline([
    ('features', AutoFeatureEngineer(engines=['tabular'], max_features=20)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

## Handling NaN Values

FeatCopilot may generate features with NaN values. Handle them in the pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from featcopilot import AutoFeatureEngineer

# Pipeline with imputation
pipeline = Pipeline([
    ('features', AutoFeatureEngineer(engines=['tabular'], max_features=30)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])
```

## Two-Stage Approach

For more control, separate feature engineering from modeling:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
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

# Align columns
cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
X_train_fe = X_train_fe[cols]
X_test_fe = X_test_fe[cols]

# Stage 2: Modeling Pipeline
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Evaluate
scores = cross_val_score(model_pipeline, X_train_fe, y_train, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# Final model
model_pipeline.fit(X_train_fe, y_train)
test_score = model_pipeline.score(X_test_fe, y_test)
print(f"Test Accuracy: {test_score:.4f}")
```

## Grid Search with Feature Engineering

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Feature engineering first
engineer = AutoFeatureEngineer(engines=['tabular'], max_features=30)
X_fe = engineer.fit_transform(X, y).fillna(0)

# Grid search on modeling pipeline
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None],
}

grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_fe, y)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

## Column Transformer Integration

Combine FeatCopilot with other transformers:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from featcopilot.engines import TabularEngine

# Identify column types
numeric_features = ['age', 'income', 'tenure']
categorical_features = ['category', 'region']

# Create transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('features', TabularEngine(polynomial_degree=2)),
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

## Custom Transformer Wrapper

Create a fully sklearn-compatible transformer:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from featcopilot import AutoFeatureEngineer

class FeatCopilotTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible FeatCopilot wrapper."""

    def __init__(self, engines=None, max_features=50, fill_value=0):
        self.engines = engines or ['tabular']
        self.max_features = max_features
        self.fill_value = fill_value
        self.engineer_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.engineer_ = AutoFeatureEngineer(
            engines=self.engines,
            max_features=self.max_features
        )
        X_fe = self.engineer_.fit_transform(X, y)
        self.feature_names_ = list(X_fe.columns)
        return self

    def transform(self, X):
        X_fe = self.engineer_.transform(X)
        # Ensure consistent columns
        for col in self.feature_names_:
            if col not in X_fe.columns:
                X_fe[col] = self.fill_value
        X_fe = X_fe[self.feature_names_]
        return X_fe.fillna(self.fill_value).values

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

# Use in pipeline
pipeline = Pipeline([
    ('features', FeatCopilotTransformer(engines=['tabular'], max_features=30)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

## Cross-Validation with Proper Leakage Prevention

```python
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def cv_with_feature_engineering(X, y, n_splits=5):
    """Cross-validation with feature engineering inside each fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Feature engineering inside fold
        engineer = AutoFeatureEngineer(engines=['tabular'], max_features=30)
        X_train_fe = engineer.fit_transform(X_train, y_train).fillna(0)
        X_val_fe = engineer.transform(X_val).fillna(0)

        # Align columns
        cols = [c for c in X_train_fe.columns if c in X_val_fe.columns]

        # Train and evaluate
        model = LogisticRegression()
        model.fit(X_train_fe[cols], y_train)
        score = model.score(X_val_fe[cols], y_val)
        scores.append(score)

    return np.mean(scores), np.std(scores)

mean_score, std_score = cv_with_feature_engineering(X, y)
print(f"CV Score: {mean_score:.4f} (+/- {std_score*2:.4f})")
```

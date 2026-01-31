# LLM-Powered Example

Demonstrates FeatCopilot's unique LLM capabilities using LiteLLM (supports OpenAI, Azure, Anthropic, and more).

## Prerequisites

- LLM provider API key (e.g., OpenAI, Azure, Anthropic)
- `featcopilot[llm]` installed
- API key configured via environment variable

## The Problem

Build a diabetes risk prediction model with:

- Semantic feature understanding
- Domain-aware feature generation
- Human-readable explanations

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from featcopilot import AutoFeatureEngineer
```

## Create Healthcare Data

```python
def create_healthcare_data(n_samples=500):
    """Create synthetic healthcare dataset."""
    np.random.seed(42)

    data = pd.DataFrame({
        'age': np.random.randint(20, 90, n_samples),
        'bmi': np.random.normal(26, 5, n_samples),
        'blood_pressure_systolic': np.random.normal(120, 20, n_samples),
        'blood_pressure_diastolic': np.random.normal(80, 12, n_samples),
        'cholesterol_total': np.random.normal(200, 40, n_samples),
        'cholesterol_hdl': np.random.normal(55, 15, n_samples),
        'cholesterol_ldl': np.random.normal(120, 35, n_samples),
        'glucose_fasting': np.random.normal(100, 25, n_samples),
        'hba1c': np.random.normal(5.5, 1.2, n_samples),
        'smoking_years': np.random.exponential(5, n_samples),
        'exercise_hours_weekly': np.random.exponential(3, n_samples),
    })

    # Create diabetes risk target
    risk = (
        0.01 * (data['age'] - 40)
        + 0.02 * (data['bmi'] - 25)
        + 0.01 * data['glucose_fasting']
        + 0.1 * data['hba1c']
        + 0.01 * data['smoking_years']
        - 0.02 * data['exercise_hours_weekly']
    )
    risk = 1 / (1 + np.exp(-risk))
    data['diabetes_risk'] = (np.random.random(n_samples) < risk).astype(int)

    return data

data = create_healthcare_data(500)
X = data.drop('diabetes_risk', axis=1)
y = data['diabetes_risk']
```

## Define Column Descriptions

This is key for LLM understanding:

```python
column_descriptions = {
    'age': 'Patient age in years',
    'bmi': 'Body Mass Index (weight in kg / height in m squared)',
    'blood_pressure_systolic': 'Systolic blood pressure in mmHg',
    'blood_pressure_diastolic': 'Diastolic blood pressure in mmHg',
    'cholesterol_total': 'Total cholesterol level in mg/dL',
    'cholesterol_hdl': 'HDL (good) cholesterol in mg/dL',
    'cholesterol_ldl': 'LDL (bad) cholesterol in mg/dL',
    'glucose_fasting': 'Fasting blood glucose in mg/dL',
    'hba1c': 'Hemoglobin A1c percentage (3-month glucose average)',
    'smoking_years': 'Number of years patient has smoked',
    'exercise_hours_weekly': 'Average hours of exercise per week',
}
```

## Initialize with LLM

```python
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=40,
    llm_config={
        'model': 'gpt-5.2',
        'max_suggestions': 15,
        'domain': 'healthcare',
        'validate_features': True
    },
    verbose=True
)
```

## Feature Engineering

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_fe = engineer.fit_transform(
    X_train, y_train,
    column_descriptions=column_descriptions,
    task_description="Predict Type 2 diabetes risk based on patient health metrics"
)

X_test_fe = engineer.transform(X_test)

# Align and clean
common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
X_train_fe = X_train_fe[common_cols].fillna(0)
X_test_fe = X_test_fe[common_cols].fillna(0)

print(f"Features generated: {len(X_train_fe.columns)}")
```

## Get Feature Explanations

```python
explanations = engineer.explain_features()

print("Feature Explanations:")
print("=" * 50)

for name, explanation in list(explanations.items())[:5]:
    print(f"\n📊 {name}")
    print(f"   {explanation}")
```

**Example Output:**
```
Feature Explanations:
==================================================

📊 age_bmi_ratio
   Ratio of age to BMI, may indicate metabolic age vs chronological age

📊 glucose_hba1c_product
   Interaction between fasting glucose and HbA1c captures glucose control

📊 cholesterol_ratio
   Ratio of total to HDL cholesterol, key cardiovascular risk indicator

📊 blood_pressure_mean
   Mean arterial pressure approximation from systolic and diastolic

📊 lifestyle_score
   Combined score from exercise and inverse of smoking years
```

## Get Generated Code

```python
feature_code = engineer.get_feature_code()

print("Generated Feature Code:")
print("=" * 50)

for name, code in list(feature_code.items())[:3]:
    print(f"\n# {name}")
    print(code)
```

**Example Output:**
```python
# age_bmi_ratio
result = df['age'] / (df['bmi'] + 1e-8)

# glucose_hba1c_product
result = df['glucose_fasting'] * df['hba1c']

# cholesterol_ratio
result = df['cholesterol_total'] / (df['cholesterol_hdl'] + 1e-8)
```

## Train Model

```python
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train_fe, y_train)

pred = model.predict_proba(X_test_fe)[:, 1]
auc = roc_auc_score(y_test, pred)

print(f"\nROC-AUC: {auc:.4f}")
```

## Generate Custom Features

Request specific features:

```python
# Generate risk stratification features
custom_features = engineer.generate_custom_features(
    prompt="Create cardiac risk stratification features based on blood pressure and cholesterol",
    n_features=3
)

for feat in custom_features:
    print(f"\nFeature: {feat['name']}")
    print(f"Code: {feat['code']}")
    print(f"Explanation: {feat['explanation']}")
```

## Generate Feature Report

```python
from featcopilot.llm import FeatureExplainer

explainer = FeatureExplainer(model='gpt-5.2')

report = explainer.generate_feature_report(
    features=engineer._engine_instances['llm'].get_feature_set(),
    X=X_train_fe,
    column_descriptions=column_descriptions,
    task_description="Predict diabetes risk"
)

# Save report
with open('diabetes_feature_report.md', 'w') as f:
    f.write(report)

print("Report saved to diabetes_feature_report.md")
```

## Complete Script

```python
"""
FeatCopilot LLM-Powered Example
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from featcopilot import AutoFeatureEngineer

# Create sample healthcare data
np.random.seed(42)
n = 500
X = pd.DataFrame({
    'age': np.random.randint(20, 90, n),
    'bmi': np.random.normal(26, 5, n),
    'glucose': np.random.normal(100, 25, n),
    'hba1c': np.random.normal(5.5, 1.2, n),
})
y = ((X['glucose'] > 100) & (X['hba1c'] > 5.7)).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# LLM-powered feature engineering
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=30,
    llm_config={'model': 'gpt-5.2', 'domain': 'healthcare'}
)

X_train_fe = engineer.fit_transform(
    X_train, y_train,
    column_descriptions={
        'age': 'Patient age',
        'bmi': 'Body Mass Index',
        'glucose': 'Fasting glucose mg/dL',
        'hba1c': 'HbA1c percentage'
    },
    task_description="Predict diabetes risk"
).fillna(0)

X_test_fe = engineer.transform(X_test).fillna(0)
cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]

# Train and evaluate
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_fe[cols], y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test_fe[cols])[:, 1])

print(f"ROC-AUC: {auc:.4f}")

# Show explanations
for feat, expl in list(engineer.explain_features().items())[:3]:
    print(f"{feat}: {expl}")
```

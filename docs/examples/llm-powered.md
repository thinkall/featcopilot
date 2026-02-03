# LLM-Powered Example

Demonstrates FeatCopilot's unique LLM capabilities using LiteLLM (supports OpenAI, Azure, Anthropic, GitHub Models, GitHub Copilot, and more).

## Prerequisites

- LLM provider API key (e.g., OpenAI, Azure, Anthropic, GitHub)
- `featcopilot[litellm]` installed
- API key configured via environment variable

## LLM Backend Options

FeatCopilot supports multiple LLM backends:

```python
# Option 1: GitHub Copilot SDK (default)
llm_config = {'model': 'gpt-5.2'}

# Option 2: LiteLLM with OpenAI
llm_config = {'model': 'gpt-4o', 'backend': 'litellm'}

# Option 3: LiteLLM with Anthropic
llm_config = {'model': 'claude-3-opus', 'backend': 'litellm'}

# Option 4: LiteLLM with GitHub Marketplace Models
# Uses GITHUB_API_KEY environment variable
llm_config = {'model': 'github/gpt-4o', 'backend': 'litellm'}

# Option 5: LiteLLM with GitHub Copilot Chat API
# Uses OAuth device flow authentication (requires Copilot subscription)
llm_config = {'model': 'github_copilot/gpt-4', 'backend': 'litellm'}

# Option 6: LiteLLM with local Ollama
llm_config = {
    'model': 'ollama/llama2',
    'backend': 'litellm',
    'api_base': 'http://localhost:11434'
}
```

## The Problem

Build a diabetes risk prediction model with:

- Semantic feature understanding
- Domain-aware feature generation
- Human-readable explanations
- Reusable transform rules

## Setup

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from featcopilot import AutoFeatureEngineer
from featcopilot import TransformRule, TransformRuleStore, TransformRuleGenerator
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

### Using GitHub Copilot SDK (Default)

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

### Using LiteLLM with GitHub Copilot

There are two GitHub providers in LiteLLM:

#### GitHub Marketplace Models (`github/` prefix)

Access models from [GitHub Marketplace](https://github.com/marketplace/models) using the `github/` prefix.
Requires `GITHUB_API_KEY` environment variable.

```python
import os
os.environ['GITHUB_API_KEY'] = 'your-github-token'

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=40,
    llm_config={
        'model': 'github/gpt-4o',      # GitHub Marketplace GPT-4o
        'backend': 'litellm',
        'max_suggestions': 15,
        'domain': 'healthcare',
        'validate_features': True
    },
    verbose=True
)
```

Available GitHub Marketplace models:
- `github/gpt-4o` - GPT-4o
- `github/gpt-4o-mini` - Lighter, faster GPT-4o
- `github/Llama-3.2-11B-Vision-Instruct` - Llama 3.2 Vision
- `github/Llama-3.1-70b-Versatile` - Llama 3.1 70B
- `github/Phi-4` - Microsoft Phi-4
- `github/Mixtral-8x7b-32768` - Mixtral 8x7B

#### GitHub Copilot Chat API (`github_copilot/` prefix)

Access GitHub Copilot's Chat API using the `github_copilot/` prefix.
Uses OAuth device flow authentication (requires paid Copilot subscription).

```python
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=40,
    llm_config={
        'model': 'github_copilot/gpt-4',  # GitHub Copilot Chat API
        'backend': 'litellm',
        'max_suggestions': 15,
        'domain': 'healthcare',
        'validate_features': True
    },
    verbose=True
)
```

On first use, you'll be prompted to authenticate:
1. LiteLLM displays a device code and verification URL
2. Visit the URL and enter the code
3. Credentials are stored locally for future use

Available GitHub Copilot models:
- `github_copilot/gpt-4` - GPT-4
- `github_copilot/gpt-5.1-codex` - GPT-5.1 Codex

### Using LiteLLM with OpenAI

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-openai-key'

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=40,
    llm_config={
        'model': 'gpt-4o',
        'backend': 'litellm',
        'max_suggestions': 15,
        'domain': 'healthcare',
    },
    verbose=True
)
```

### Using LiteLLM with Anthropic Claude

```python
import os
os.environ['ANTHROPIC_API_KEY'] = 'your-anthropic-key'

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=40,
    llm_config={
        'model': 'claude-3-opus',
        'backend': 'litellm',
        'max_suggestions': 15,
        'domain': 'healthcare',
    },
    verbose=True
)
```

### Using LiteLLM with Local Ollama

```python
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    max_features=40,
    llm_config={
        'model': 'ollama/llama2',
        'backend': 'litellm',
        'api_base': 'http://localhost:11434',
        'max_suggestions': 15,
        'domain': 'healthcare',
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

## Transform Rules

Create reusable feature transformations from natural language that can be saved and applied across different datasets.

```python
from featcopilot import TransformRule, TransformRuleStore, TransformRuleGenerator

# Initialize store and generator
store = TransformRuleStore()  # Default: ~/.featcopilot/rules.json
generator = TransformRuleGenerator(store=store)

# Generate rule from natural language
rule = generator.generate_from_description(
    description="Calculate the ratio of glucose to HbA1c",
    columns={"glucose_fasting": "float", "hba1c": "float"},
    tags=["healthcare", "diabetes"],
    save=True
)

# Apply to data
df = pd.DataFrame({'glucose_fasting': [95, 110], 'hba1c': [5.4, 6.2]})
result = rule.apply(df)
```

### Reuse on Different Datasets

Rules automatically match columns with similar names:

```python
# New dataset with different column names
new_data = pd.DataFrame({
    'patient_glucose': [100, 120],
    'patient_hba1c': [5.8, 6.5]
})

# Find and apply matching rules
matches = store.find_matching_rules(columns=new_data.columns.tolist())
if matches:
    rule, mapping = matches[0]
    result = rule.apply(new_data, column_mapping=mapping)
```

### Manual Rules and Management

```python
# Create manual rule
manual_rule = TransformRule(
    name="bmi_calc",
    description="Calculate BMI",
    code="result = df['weight'] / (df['height'] ** 2 + 1e-8)",
    input_columns=["weight", "height"],
    column_patterns=[".*weight.*", ".*height.*"],
)
store.save_rule(manual_rule)

# Search, list, export
store.list_rules(tags=["healthcare"])
store.search_by_description("diabetes")
store.export_rules("rules.json")
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
from featcopilot import TransformRule, TransformRuleStore, TransformRuleGenerator

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

# ============================================================
# Transform Rules: Create reusable transformations
# ============================================================
store = TransformRuleStore()
generator = TransformRuleGenerator(store=store)

# Generate and save a reusable rule
rule = generator.generate_from_description(
    description="Calculate glucose-HbA1c product as diabetes indicator",
    columns={"glucose": "float", "hba1c": "float"},
    tags=["healthcare", "diabetes"],
    save=True
)

print(f"\nCreated reusable rule: {rule.name}")
print(f"Code: {rule.code}")
print(f"This rule can now be reused on any dataset with similar columns!")
```

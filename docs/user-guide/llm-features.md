# LLM-Powered Features

FeatCopilot's LLM integration via LiteLLM enables semantic understanding and intelligent feature generation with support for 100+ LLM providers including GitHub Models and GitHub Copilot.

## Overview

The LLM engine provides:

- **Semantic Feature Discovery**: Understands column meanings from names/descriptions
- **Domain-Aware Generation**: Tailored features for specific domains
- **Feature Explanations**: Human-readable descriptions for stakeholders
- **Code Generation**: Automatic Python code for custom features
- **Iterative Refinement**: Improve features based on feedback

## LLM Backend Options

FeatCopilot supports two LLM backends:

1. **GitHub Copilot SDK** (default): Native integration with GitHub Copilot
2. **LiteLLM**: Universal interface supporting 100+ providers including:
   - OpenAI (GPT-4o, GPT-4, GPT-3.5)
   - Anthropic (Claude 3 Opus, Sonnet, Haiku)
   - Azure OpenAI
   - Google (Gemini Pro, Gemini Ultra)
   - **GitHub Models** (via `github/` prefix) - Llama, Phi, Mixtral, etc.
   - **GitHub Copilot** (via `github_copilot/` prefix) - GPT-4, GPT-5.1-codex
   - AWS Bedrock
   - Ollama (local models)
   - And many more...

## SemanticEngine

The core LLM-powered engine:

```python
from featcopilot.llm import SemanticEngine

# Using GitHub Copilot SDK (default)
engine = SemanticEngine(
    model='gpt-5.2',
    max_suggestions=20,
    validate_features=True,
    domain='healthcare',
    verbose=True
)

# Using LiteLLM backend
engine = SemanticEngine(
    model='gpt-4o',
    backend='litellm',
    max_suggestions=20,
    domain='healthcare',
    verbose=True
)

X_features = engine.fit_transform(
    X, y,
    column_descriptions={
        'age': 'Patient age in years',
        'bmi': 'Body Mass Index',
        'glucose': 'Fasting blood glucose mg/dL',
        'hba1c': 'Hemoglobin A1c percentage'
    },
    task_description='Predict Type 2 diabetes diagnosis'
)
```

## GitHub Models via LiteLLM

Access models from [GitHub Marketplace Models](https://github.com/marketplace/models) using the `github/` prefix:

```python
import os
os.environ['GITHUB_API_KEY'] = 'your-github-token'

from featcopilot import AutoFeatureEngineer

# GitHub Marketplace Models via LiteLLM
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'github/gpt-4o',  # GitHub Marketplace GPT-4o
        'backend': 'litellm',
        'max_suggestions': 15,
        'domain': 'healthcare',
    }
)
```

### Available GitHub Marketplace Models

| Model | Usage |
|-------|-------|
| GPT-4o | `github/gpt-4o` |
| GPT-4o Mini | `github/gpt-4o-mini` |
| Llama 3.2 11B Vision | `github/Llama-3.2-11B-Vision-Instruct` |
| Llama 3.1 70B | `github/Llama-3.1-70b-Versatile` |
| Phi-4 | `github/Phi-4` |
| Mixtral 8x7B | `github/Mixtral-8x7b-32768` |

> **Note:** All GitHub Marketplace models are supported. Just use `github/<model-name>` prefix.

### Setting Up GitHub API Key

```bash
export GITHUB_API_KEY="your-github-personal-access-token"
```

## GitHub Copilot via LiteLLM

Access GitHub Copilot Chat API using the `github_copilot/` prefix. This uses OAuth device flow authentication:

```python
from featcopilot import AutoFeatureEngineer

# GitHub Copilot Chat API via LiteLLM
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'github_copilot/gpt-4',  # GitHub Copilot's GPT-4
        'backend': 'litellm',
        'max_suggestions': 15,
        'domain': 'healthcare',
    }
)
```

### Available GitHub Copilot Models

| Model | Usage |
|-------|-------|
| GPT-4 | `github_copilot/gpt-4` |
| GPT-5.1 Codex | `github_copilot/gpt-5.1-codex` |
| Text Embedding | `github_copilot/text-embedding-3-small` |

### Authentication

GitHub Copilot uses OAuth device flow:
1. On first use, LiteLLM displays a device code and verification URL
2. Visit the URL and enter the code to authenticate
3. Credentials are stored locally for future use

> **Note:** Requires a paid GitHub Copilot subscription.

## Configuration

### LLM Config Options

```python
llm_config = {
    'model': 'gpt-5.2',           # Model: gpt-5.2, github/gpt-4o, github_copilot/gpt-4, etc.
    'backend': 'copilot',         # Backend: 'copilot' or 'litellm'
    'max_suggestions': 20,        # Max features to suggest
    'domain': 'healthcare',       # Domain context
    'validate_features': True,    # Validate generated code
    'temperature': 0.3,           # Generation temperature
    'api_key': None,              # API key (for litellm backend)
    'api_base': None,             # Custom API base URL (for litellm)
}

engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config=llm_config
)
```

### Listing Available Models

Use LiteLLM to discover supported models:

```python
from featcopilot.utils import list_models, get_model_info, get_model_names

# List all available models
models = list_models()
for m in models:
    model_id = m.get('id') or m.get('name')
    print(f"{model_id}: {m.get('description', '')}")

# List with verbose output
list_models(verbose=True)

# Get just the model names/identifiers
model_names = get_model_names()
print(model_names)  # ['gpt-4', 'claude-3-sonnet', ...]

# Filter by provider
openai_models = list_models(provider='OpenAI')
anthropic_models = list_models(provider='Anthropic')

# Get info about a specific model
info = get_model_info('gpt-4')
if info:
    print(info)

# Check if a model is valid
from featcopilot.utils import is_valid_model
if is_valid_model('gpt-4'):
    print("Model is available")
```

### Supported Models

Common models supported through LiteLLM:

| Model | Provider | Description |
|-------|----------|-------------|
| `gpt-4o` | OpenAI | GPT-4o (recommended) |
| `gpt-4-turbo` | OpenAI | Faster GPT-4 variant |
| `gpt-3.5-turbo` | OpenAI | Fast and efficient |
| `azure/gpt-4` | Azure OpenAI | Azure-hosted GPT-4 |
| `github/gpt-4o` | GitHub Models | GPT-4o via GitHub Marketplace |
| `github/Llama-3.2-11B-Vision-Instruct` | GitHub Models | Llama 3.2 via GitHub |
| `github/Phi-4` | GitHub Models | Microsoft Phi-4 via GitHub |
| `github_copilot/gpt-4` | GitHub Copilot | GPT-4 via Copilot Chat API |
| `github_copilot/gpt-5.1-codex` | GitHub Copilot | GPT-5.1 Codex via Copilot |
| `claude-3-opus` | Anthropic | Premium quality |
| `claude-3-sonnet` | Anthropic | Balanced performance |
| `claude-3-haiku` | Anthropic | Fast and efficient |
| `gemini-pro` | Google | Google's Gemini Pro |
| `ollama/llama2` | Ollama (local) | Local LLaMA 2 |

> **Note:** See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for the full list of supported models and providers.

### Choosing a Model

```python
from featcopilot.utils import get_default_model, get_model_names

# Use the default model (gpt-5.2)
llm_config = {'model': get_default_model()}

# Or choose from available models
available = get_model_names()
print(f"Available models: {available}")

# Pick based on your needs:
# - gpt-5.2: Best all-around choice (default)
# - gpt-5.1-codex: Best for code generation tasks
# - claude-sonnet-4: Alternative with strong reasoning
# - gpt-4.1 / claude-haiku-4.5: Faster, lower cost
# - claude-opus-4.5: Premium quality (slower)
```

## Providing Context

### Column Descriptions

Help the LLM understand your data:

```python
column_descriptions = {
    'age': 'Customer age in years (18-100)',
    'income': 'Annual household income in USD',
    'tenure_months': 'Number of months as customer',
    'monthly_charges': 'Monthly subscription fee',
    'total_charges': 'Cumulative charges to date',
    'contract_type': 'Month-to-month, One year, or Two year',
    'support_tickets': 'Number of support tickets filed'
}
```

### Task Description

Describe what you're trying to predict:

```python
task_description = """
Predict customer churn for a telecommunications company.
Churn is defined as canceling service within the next 30 days.
Key business goals: reduce churn rate, identify at-risk customers early.
"""
```

### Domain Context

Specify the domain for relevant features:

```python
# Supported domains
domains = [
    'healthcare',    # Medical features, risk scores
    'finance',       # Financial ratios, risk metrics
    'retail',        # RFM, customer lifetime value
    'telecom',       # Usage patterns, churn indicators
    'manufacturing', # Quality metrics, efficiency
]

engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config={'domain': 'finance'}
)
```

## Feature Explanations

Get human-readable explanations:

```python
# After fit_transform
explanations = engineer.explain_features()

for feature, explanation in explanations.items():
    print(f"📊 {feature}")
    print(f"   {explanation}\n")
```

**Example Output:**
```
📊 age_bmi_ratio
   Ratio of age to BMI, may indicate metabolic age vs chronological age

📊 glucose_hba1c_interaction
   Interaction between fasting glucose and HbA1c captures glucose control

📊 cholesterol_risk_score
   Composite score from total, HDL, and LDL cholesterol levels
```

## Generated Code

Access the Python code for features:

```python
feature_code = engineer.get_feature_code()

for name, code in feature_code.items():
    print(f"# {name}")
    print(code)
    print()
```

**Example Output:**
```python
# age_bmi_ratio
result = df['age'] / (df['bmi'] + 1e-8)

# glucose_hba1c_interaction
result = df['glucose'] * df['hba1c']

# cholesterol_risk_score
result = (df['cholesterol_total'] - df['cholesterol_hdl']) / (df['cholesterol_ldl'] + 1)
```

## Custom Feature Generation

Request specific features:

```python
# Generate features for a specific focus area
custom_features = engineer.generate_custom_features(
    prompt="Create risk stratification features for cardiac patients",
    n_features=5
)

for feature in custom_features:
    print(f"Name: {feature['name']}")
    print(f"Code: {feature['code']}")
    print(f"Explanation: {feature['explanation']}\n")
```

## FeatureCodeGenerator

Generate features from natural language:

```python
from featcopilot.llm import FeatureCodeGenerator

generator = FeatureCodeGenerator(model='gpt-5.2')

# Generate single feature
feature = generator.generate(
    description="Calculate BMI from height in meters and weight in kg",
    columns={'height_m': 'float', 'weight_kg': 'float'}
)

print(feature.code)
# result = df['weight_kg'] / (df['height_m'] ** 2)

# Generate domain-specific features
healthcare_features = generator.generate_domain_features(
    domain='healthcare',
    columns={'age': 'int', 'bmi': 'float', 'glucose': 'float'},
    n_features=5
)
```

## FeatureExplainer

Generate reports:

```python
from featcopilot.llm import FeatureExplainer

explainer = FeatureExplainer(model='gpt-5.2')

# Generate comprehensive report
report = explainer.generate_feature_report(
    features=engineer._feature_set,
    X=X_transformed,
    column_descriptions=column_descriptions,
    task_description=task_description
)

# Save as markdown
with open('feature_report.md', 'w') as f:
    f.write(report)
```

## Best Practices

### 1. Provide Rich Context

```python
# ❌ Minimal context
engineer.fit_transform(X, y)

# ✅ Rich context
engineer.fit_transform(
    X, y,
    column_descriptions={...},
    task_description="...",
)
```

### 2. Validate Generated Features

```python
llm_config = {
    'validate_features': True  # Default
}
```

### 3. Review Generated Code

```python
# Always review LLM-generated code before production
for name, code in engineer.get_feature_code().items():
    print(f"Review: {name}")
    print(code)
```

### 4. Iterate and Refine

```python
# Generate more features in specific areas
more_features = engineer.generate_custom_features(
    prompt="Focus on interaction effects between age and metabolic markers"
)
```

## Fallback Behavior

When LLM is unavailable:

```python
# Without API key configured, mock responses are used
engineer = AutoFeatureEngineer(engines=['llm'])
# Warning: LLM not configured. Using mock responses.

# Mock generates context-aware features based on column names
# but without true semantic understanding
```

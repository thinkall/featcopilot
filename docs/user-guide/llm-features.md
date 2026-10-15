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

FeatCopilot supports three LLM backends:

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
3. **OpenAI SDK** (`backend='openai'`): Direct access to OpenAI-compatible endpoints.

The shared project default is `gpt-5.6-sol`. This is a requested model identifier,
not a guarantee that your provider or Copilot account supports it. Explicit model
overrides are preserved. Provider clients raise authentication, timeout and
empty-response errors; required general-feature and typed-plan requests propagate
failures. Production clients never substitute fabricated mock features or
silently select a different model. Legacy code-mode text suggestions are optional:
if that request fails, the engine logs a warning and can continue with explicitly
identified built-in `custom` text features. In a text-only frame, a successful
fit therefore does not by itself prove that the provider succeeded; inspect the
warning and feature origins. This does not apply to typed-plan mode.

## SemanticEngine

The core LLM-powered engine:

```python
from featcopilot.llm import SemanticEngine

# Using GitHub Copilot SDK (default)
engine = SemanticEngine(
    model='gpt-5.6-sol',
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

### Fit/inference contract and limitations

`fit` retains accepted feature code and names; `transform` executes those same rules
without an LLM request. Refitting clears old rules, and serialized engines exclude
live provider connections. With validation enabled, both text and general features
are checked for executable code, a `result`, compatible row shape/index, and usable
training output. Duplicate names and collisions with original columns are rejected.
Invalid suggestions are logged and omitted during fitting. An empty accepted set
is explicitly logged and adds no features. Inference execution errors raise rather
than silently dropping fitted columns.

Generated code runs in-process and should be reviewed before use; validation is
not a security sandbox. Pandas and NumPy are supplied as `pd` and `np`, including
inside lambdas and helper functions. Each rule must operate on original input
columns, not another generated feature. `validate_features=False` skips execution
checks during fitting, but does not suppress inference errors.

**Statistical limitation:** The engine persists code, not learned statistical
state. A generated `df['x'].mean()`, group mean, `qcut`, rank, or similar operation
is recomputed on the inference batch. That can make predictions batch-dependent
and introduce train/test leakage in evaluation. Prompts request row-wise rules,
but validation does not prove batch invariance. Review generated code with
`get_feature_code()` and use fitted tabular transformers for learned statistics;
do not use such generated rules for held-out evaluation without an explicit
training-only fit/transform implementation. Fit the LLM engine on training data
only, never on concatenated train/test data.

## Fitted statistical plans (opt-in)

`feature_format="plan"` requests a typed operator plan instead of executing
generated Python. The initial operator set is deliberately small:
`standardize`, `clip`, `frequency`, `group_mean`, `target_mean`,
`rare_category`, `multiply`, `divide`, and `log1p`. Each rule declares its name, operator, original source columns
and validated parameters. It is not a general Python compiler.

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=["llm"],
    task="regression",
    llm_config={
        "feature_format": "plan",
        "rules": [
            {"name": "amount_scaled", "operator": "standardize", "columns": ["amount"]},
            {"name": "group_frequency", "operator": "frequency", "columns": ["group"]},
        ],
    },
)
X_train_fe = engineer.fit_transform(X_train, y_train, apply_selection=False)
X_test_fe = engineer.transform(X_test)
fitted_state = engineer.get_fitted_rules()
```

Supplying `rules` uses no provider, including when the list is empty. Omit
`rules` to request a bounded plan from the configured LLM; the project default
remains `gpt-5.6-sol`. Plan mode does not run the legacy text/code feature
generation path. Invalid or unsupported plans fail explicitly.

Copilot plan requests use a fresh isolated temporary workspace and session,
with tools, MCPs, skills, custom agents and configuration discovery disabled
and permission requests rejected. The request timeout covers startup,
authentication, session creation and generation together. Each shutdown
step has a separate grace of at most five seconds (or the shorter request
timeout); cleanup failures are reported without replacing the original
request error. Cancellation is cooperative, so blocking or
cancellation-resistant SDK code cannot be given a hard process deadline by
the in-process client. This plan policy does not change legacy code-mode
session permissions.
If the OS refuses temporary-workspace cleanup, a warning is emitted and
the workspace may remain; the request result or original error is preserved.
Copilot and semantic-engine timeout settings must be positive finite numbers;
booleans are rejected rather than interpreted as one or zero seconds.

All means, scales, bounds, category counts and group mappings are fitted on
training data and frozen for inference. `target_mean` uses out-of-fold
training encodings and full-training mappings only for later `transform`.
Multiclass target means remain rejected rather than approximated by ordinal
means. Binary numeric/boolean targets preserve fixed 0/1 semantics even when
stored in object arrays; string labels such as `"1"` remain distinct labels.
Group-disjoint or chronological target cross-fitting accepts `groups`,
`timestamps`, and optional `label_end_times` at fit. No-history temporal prefix
rows use an explicit fixed-zero prior, with a warning and serialized
`cross_fit_report`; this is not a full-data-mean fallback. Invalid or fully
purged folds raise. Combined group/time constraints remain unsupported.

The three rowwise arithmetic operators require numeric inputs and empty
parameters. `divide` uses numerator/denominator order and returns missing for
zero denominators. `log1p` means `log1p(abs(x))`. Missing/nonfinite operands stay
missing; finite-input overflow follows the fitted-rule engine's documented
float64 saturation policy. No statistics are learned for these operations.

For provider-independent use, import `FittedRuleEngine` from
`featcopilot.engines`, or wrap it with
`FeatureEngineerTransformer(engine="fitted_rules", rules=[...])`.
`FittedRuleEngine.to_dict()` / `from_dict()` serialize and restore learned
state without a provider connection. Treat that state as a training-derived
artifact that may contain category values; do not publish it blindly.

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
    'model': 'gpt-5.6-sol',       # Explicit provider-supported overrides are allowed
    'backend': 'copilot',         # Backend: 'copilot', 'openai', or 'litellm'
    'max_suggestions': 20,        # Max features to suggest
    'domain': 'healthcare',       # Domain context
    'validate_features': True,    # Validate generated code
    'temperature': 0.3,           # OpenAI/LiteLLM generation temperature
    'timeout': 60.0,              # Request timeout in seconds
    'max_tokens': 4096,           # OpenAI/LiteLLM response token limit
    'api_key': None,              # API key (for litellm backend)
    'api_base': None,             # Custom API base URL (for litellm)
}

engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config=llm_config
)
```

The Copilot SDK controls generation parameters itself: its session API does not
expose temperature or response-token limits. `temperature` and `max_tokens` apply
to the OpenAI/LiteLLM backends, not Copilot. `timeout` applies to all backends.
Non-default unsupported provider settings emit a warning when the client starts.
`api_key` and `api_base` apply to OpenAI/LiteLLM, and `api_version` applies only
to the OpenAI backend; the Copilot backend uses its own authentication.
`enable_text_features`, `keep_text_columns`, and `text_feature_types` configure text
generation; the latter guides optional LLM suggestions, while built-in text
length/count features remain available.

### Listing Available Models

Query the authenticated Copilot SDK to discover available Copilot models:

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

# Use the project default model (gpt-5.6-sol)
llm_config = {'model': get_default_model()}

# Or choose from available models
available = get_model_names()
print(f"Available models: {available}")

# Pick based on your needs:
# - gpt-5.6-sol: Project default; verify account/provider availability
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

generator = FeatureCodeGenerator(model='gpt-5.6-sol')

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

explainer = FeatureExplainer(model='gpt-5.6-sol')

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

## Unavailable Providers

The LLM engine reports provider failures instead of inventing mock features.
For offline feature engineering, explicitly omit the LLM engine:

```python
engineer = AutoFeatureEngineer(engines=['tabular'])
```

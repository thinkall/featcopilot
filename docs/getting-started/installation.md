# Installation

## Requirements

- Python 3.9 or higher
- pip or conda package manager

## Basic Installation

Install FeatCopilot using pip:

```bash
pip install featcopilot
```

This installs the core package with support for:

- Tabular feature engineering
- Time series feature extraction
- Feature selection methods
- Scikit-learn integration

## Installation with LLM Features

FeatCopilot supports two LLM backends:

### Option 1: GitHub Copilot SDK (Default)

```bash
pip install featcopilot[llm]
```

This installs the GitHub Copilot SDK for native Copilot integration.

### Option 2: LiteLLM (100+ Providers)

```bash
pip install featcopilot[litellm]
```

This installs LiteLLM, supporting OpenAI, Azure, Anthropic, Google, GitHub Models, and 100+ other providers.

## Full Installation

For all features including both LLM backends and time series analysis:

```bash
pip install featcopilot[full]
```

This includes:

- GitHub Copilot SDK
- LiteLLM (100+ providers)
- `statsmodels` for advanced time series features

## Development Installation

For contributing to FeatCopilot:

```bash
# Clone the repository
git clone https://github.com/thinkall/featcopilot.git
cd featcopilot

# Install in development mode
pip install -e ".[dev]"
```

Development dependencies include:

- `pytest` - Testing framework
- `black` - Code formatting
- `ruff` - Linting
- `mypy` - Type checking

## Verify Installation

```python
import featcopilot
print(featcopilot.__version__)
# Output: 0.1.0

from featcopilot import AutoFeatureEngineer
print("Installation successful!")
```

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21.0 | Numerical operations |
| pandas | >=1.3.0 | Data manipulation |
| scipy | >=1.7.0 | Scientific computing |
| scikit-learn | >=1.0.0 | ML utilities & transformers |
| pydantic | >=2.0.0 | Configuration validation |
| joblib | >=1.1.0 | Parallel processing |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| github-copilot-sdk | >=0.1.0 | GitHub Copilot SDK (default LLM backend) |
| litellm | >=1.0.0 | LiteLLM (100+ LLM providers) |
| statsmodels | >=0.13.0 | Advanced time series |

## Troubleshooting

### ImportError: No module named 'featcopilot'

Ensure you've installed the package:

```bash
pip install featcopilot
```

### LLM features not working

1. Install LLM extras: `pip install featcopilot[llm]`
2. Set up your LLM provider API key (see [Authentication](authentication.md))
3. Ensure your API key is valid and has sufficient quota

### Dependency conflicts

Try creating a fresh virtual environment:

```bash
python -m venv featcopilot-env
source featcopilot-env/bin/activate  # Linux/Mac
# or
featcopilot-env\Scripts\activate  # Windows

pip install featcopilot
```

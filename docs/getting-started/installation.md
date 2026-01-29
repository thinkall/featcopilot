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

To use LLM-powered feature engineering with GitHub Copilot SDK:

```bash
pip install featcopilot[llm]
```

This additionally installs:

- `copilot-sdk` - GitHub Copilot Python SDK
- Dependencies for LLM communication

## Full Installation

For all features including time series analysis:

```bash
pip install featcopilot[full]
```

This includes:

- All LLM capabilities
- `statsmodels` for advanced time series features

## Development Installation

For contributing to FeatCopilot:

```bash
# Clone the repository
git clone https://github.com/yourusername/featcopilot.git
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
| copilot-sdk | >=0.1.0 | GitHub Copilot integration |
| statsmodels | >=0.13.0 | Advanced time series |

## Troubleshooting

### ImportError: No module named 'featcopilot'

Ensure you've installed the package:

```bash
pip install featcopilot
```

### LLM features not working

1. Install LLM extras: `pip install featcopilot[llm]`
2. Authenticate with GitHub Copilot CLI (see [Authentication](authentication.md))
3. Ensure you have an active GitHub Copilot subscription

### Dependency conflicts

Try creating a fresh virtual environment:

```bash
python -m venv featcopilot-env
source featcopilot-env/bin/activate  # Linux/Mac
# or
featcopilot-env\Scripts\activate  # Windows

pip install featcopilot
```

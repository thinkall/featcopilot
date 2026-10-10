# Installation

## Requirements

- Python 3.10 or higher for the core package
- Python 3.11 or higher for GitHub Copilot integration (`llm`, `full`, or `benchmark` extras)
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

This requires Python 3.11+ and installs GitHub Copilot SDK 0.3.0 or newer,
which provides the session APIs used by FeatCopilot. The older SDK available
on Python 3.10 does not provide those APIs; core feature engineering still
supports Python 3.10.

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
- `psutil` - Isolated worker and process-limit tests

On Python 3.11+, install `".[dev,llm]"` to also run the Copilot SDK contract
tests. These use mocked clients, not live provider calls. The Python 3.10
CI job runs the core suite without the unsupported SDK extra.

## Verify Installation

```python
import featcopilot
print(featcopilot.__version__)
# Output: 0.4.0

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
| pydantic | >=2.5.0 | Configuration validation |
| joblib | >=1.1.0 | Parallel processing |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| github-copilot-sdk | >=0.3.0 | GitHub Copilot SDK (Python 3.11+) |
| litellm | >=1.0.0 | LiteLLM (100+ LLM providers) |
| statsmodels | >=0.13.0 | Advanced time series |

## Troubleshooting

### Native optimizer crashes on Linux x86-64 with Python 3.10

The SciPy 1.15.0-1.15.3 wheels share an OpenBLAS 0.3.28 build with a
worker-buffer initialization regression. Growing the native thread pool
beyond its startup size can crash in a triangular-solve worker. The
upstream fix is documented in
[OpenMathLib/OpenBLAS#4879](https://github.com/OpenMathLib/OpenBLAS/pull/4879).

FeatCopilot excludes those four SciPy releases only on the reproduced
Python 3.10/Linux/x86-64 combination. The resolver can use SciPy 1.14.1
(compatible with NumPy 2.2.6) there, without changing the L-BFGS-B solver,
thread limits or test assertions. Other platforms and Python versions
are not constrained by this exclusion, and future patched releases are
not blocked. No library preload, CPU-kernel override or warmup is needed.

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

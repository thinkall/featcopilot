# Authentication

FeatCopilot supports two LLM backends:

1. **GitHub Copilot SDK** (default) - Native integration with GitHub Copilot
2. **LiteLLM** - Universal interface supporting 100+ providers

## GitHub Copilot SDK (Default)

The default backend uses GitHub Copilot SDK. If you have GitHub Copilot CLI installed and authenticated, it works automatically.

```python
from featcopilot import AutoFeatureEngineer

# Uses GitHub Copilot SDK by default
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={'model': 'gpt-5.2'}
)
```

## LiteLLM Backend

For access to 100+ LLM providers, use the LiteLLM backend.

### Supported Providers

| Provider | Environment Variable | Model Example |
|----------|---------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o`, `gpt-4` |
| Azure OpenAI | `AZURE_API_KEY`, `AZURE_API_BASE` | `azure/gpt-4` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-opus`, `claude-3-sonnet` |
| Google | `GOOGLE_API_KEY` | `gemini-pro` |
| GitHub Models | `GITHUB_API_KEY` | `github/gpt-4o`, `github/Llama-3.2-11B-Vision-Instruct` |
| GitHub Copilot | OAuth device flow | `github_copilot/gpt-4` |
| AWS Bedrock | AWS credentials | `bedrock/anthropic.claude-v2` |
| Ollama | (local) | `ollama/llama2` |

[See full list of supported providers](https://docs.litellm.ai/docs/providers)

### Step 1: Install LiteLLM

```bash
pip install featcopilot[litellm]
```

### Step 2: Set Up Authentication

#### Option A: Environment Variables (Recommended)

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# GitHub Models
export GITHUB_API_KEY="ghp_..."
```

#### Option B: Pass Directly in Code

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'gpt-4o',
        'backend': 'litellm',
        'api_key': 'sk-...',  # Not recommended for production
    }
)
```

### Step 3: Use LiteLLM Backend

```python
from featcopilot import AutoFeatureEngineer

# Specify backend='litellm' to use LiteLLM
engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config={
        'model': 'gpt-4o',
        'backend': 'litellm'
    }
)
```

## Provider-Specific Examples (LiteLLM)

### OpenAI

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'gpt-4o',
        'backend': 'litellm',
        # Uses OPENAI_API_KEY from environment
    }
)
```

### Azure OpenAI

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'azure/your-deployment-name',
        'backend': 'litellm',
        # Uses AZURE_API_KEY, AZURE_API_BASE from environment
    }
)
```

### Anthropic Claude

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'claude-3-sonnet-20240229',
        'backend': 'litellm',
        # Uses ANTHROPIC_API_KEY from environment
    }
)
```

### GitHub Marketplace Models

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'github/gpt-4o',  # or github/Llama-3.2-11B-Vision-Instruct
        'backend': 'litellm',
        # Uses GITHUB_API_KEY from environment
    }
)
```

### GitHub Copilot Chat API

```python
from featcopilot import AutoFeatureEngineer

# Uses OAuth device flow - you'll be prompted to authenticate
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'github_copilot/gpt-4',
        'backend': 'litellm',
    }
)
```

### Local Models (Ollama)

```python
from featcopilot import AutoFeatureEngineer

# Run Ollama locally first: ollama run llama2
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'ollama/llama2',
        'backend': 'litellm',
        'api_base': 'http://localhost:11434',
    }
)
```

## Environment Variables

Common environment variables for LiteLLM:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://..."
export AZURE_API_VERSION="2024-02-15-preview"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Custom proxy/base URL
export LITELLM_PROXY_BASE_URL="http://your-proxy:8000"
```

## Troubleshooting

### "API key not found"

Ensure your API key is set in the environment:

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set it if missing
export OPENAI_API_KEY="sk-..."
```

### "Rate limit exceeded"

LLM APIs have rate limits. If exceeded:

- Wait a few minutes before retrying
- Reduce `max_suggestions` in LLM config
- Use caching to avoid redundant requests

### "Invalid model"

Check that you're using the correct model name for your provider:

```python
# OpenAI models
'gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo'

# Azure models (use your deployment name)
'azure/my-gpt4-deployment'

# Anthropic models
'claude-3-opus-20240229', 'claude-3-sonnet-20240229'
```

### Using Behind Corporate Proxy

```bash
# Set proxy environment variables
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

## Security Best Practices

!!! warning "Security"

    - Never commit API keys to version control
    - Use environment variables for sensitive configuration
    - Regularly rotate API keys
    - Review your provider's data handling policies

## Offline/Mock Mode

If LLM is unavailable, FeatCopilot automatically falls back to mock responses:

```python
# This works without authentication (uses mock LLM)
engineer = AutoFeatureEngineer(engines=['tabular', 'llm'])

# You'll see this warning:
# "Warning: LLM not configured. Using mock responses."

# Features will still be generated using heuristics
X_transformed = engineer.fit_transform(X, y)
```

The mock mode generates context-aware features based on column names, but without the semantic understanding of a real LLM.

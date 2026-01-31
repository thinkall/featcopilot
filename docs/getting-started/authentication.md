# Authentication

FeatCopilot's LLM features use LiteLLM, which supports 100+ LLM providers including OpenAI, Azure OpenAI, Anthropic, and more.

## Prerequisites

- **Python 3.9+**
- **API Key** for your chosen LLM provider

## Supported Providers

LiteLLM supports many providers out of the box:

| Provider | Environment Variable | Model Example |
|----------|---------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4`, `gpt-3.5-turbo` |
| Azure OpenAI | `AZURE_API_KEY`, `AZURE_API_BASE` | `azure/gpt-4` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-opus`, `claude-3-sonnet` |
| Google | `GOOGLE_API_KEY` | `gemini-pro` |
| AWS Bedrock | AWS credentials | `bedrock/anthropic.claude-v2` |
| Ollama | (local) | `ollama/llama2` |

[See full list of supported providers](https://docs.litellm.ai/docs/providers)

## Step 1: Install LLM Dependencies

```bash
pip install featcopilot[llm]
```

This installs `litellm` and related dependencies.

## Step 2: Set Up Authentication

### Option A: Environment Variables (Recommended)

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Option B: Pass Directly in Code

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'gpt-4',
        'api_key': 'sk-...',  # Not recommended for production
    }
)
```

## Step 3: Test Your Setup

```python
from featcopilot import AutoFeatureEngineer

# Test with your configured provider
engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config={'model': 'gpt-4'}  # or your preferred model
)

# If authentication works, you'll see LLM-generated features
# If not, you'll see: "Warning: LLM not configured. Using mock responses."
```

## Provider-Specific Examples

### OpenAI

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'gpt-4',
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
        # Uses ANTHROPIC_API_KEY from environment
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

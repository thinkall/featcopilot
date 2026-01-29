# Authentication

FeatCopilot's LLM features use the GitHub Copilot SDK, which requires authentication with your GitHub account.

## Prerequisites

- **GitHub Account** with an active Copilot subscription
- **Node.js** (for CLI installation)
- **Python 3.9+**

## Step 1: Install GitHub Copilot CLI

### Option A: Using npm

```bash
npm install -g @github/copilot-cli
```

### Option B: Using GitHub CLI Extension

```bash
# Install GitHub CLI first (if not installed)
# See: https://cli.github.com/

# Then install Copilot extension
gh extension install github/gh-copilot
```

## Step 2: Authenticate

### Using Copilot CLI

```bash
copilot auth login
```

This will:

1. Open a browser window
2. Ask you to log in to GitHub
3. Authorize the Copilot CLI
4. Store credentials locally

### Using GitHub CLI

```bash
gh auth login
gh copilot auth
```

## Step 3: Verify Authentication

```bash
# Test CLI is working
copilot --version

# Test a simple query
copilot suggest "hello world in python"
```

## Step 4: Install Python SDK

```bash
pip install copilot-sdk
```

## Step 5: Test in FeatCopilot

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config={'model': 'gpt-5'}
)

# If authentication works, you'll see LLM-generated features
# If not, you'll see: "Warning: copilot-sdk not installed. Using mock LLM responses."
```

## Environment Variables

You can configure the SDK using environment variables:

```bash
# Custom CLI path
export COPILOT_CLI_PATH=/path/to/copilot

# Custom server URL (for enterprise)
export COPILOT_CLI_URL=https://copilot.enterprise.com
```

## Troubleshooting

### "copilot: command not found"

The CLI is not installed or not in PATH:

```bash
# Check if npm installed it globally
npm list -g @github/copilot-cli

# Add npm global bin to PATH
export PATH="$PATH:$(npm config get prefix)/bin"
```

### "Authentication required"

Re-authenticate:

```bash
copilot auth logout
copilot auth login
```

### "No active subscription"

Ensure your GitHub account has:

1. GitHub Copilot Individual, Business, or Enterprise subscription
2. The subscription is active (not expired)

Check at: [github.com/settings/copilot](https://github.com/settings/copilot)

### "Rate limit exceeded"

The Copilot API has rate limits. If exceeded:

- Wait a few minutes before retrying
- Reduce `max_suggestions` in LLM config
- Use caching to avoid redundant requests

### SDK Not Found

```python
# Error: Warning: copilot-sdk not installed
pip install copilot-sdk
```

### Using Behind Corporate Proxy

```bash
# Set proxy environment variables
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

## Security Best Practices

!!! warning "Security"

    - Never commit credentials to version control
    - Use environment variables for sensitive configuration
    - Regularly rotate authentication tokens
    - Review Copilot's data handling policies for your organization

## Offline/Mock Mode

If Copilot is unavailable, FeatCopilot automatically falls back to mock responses:

```python
# This works without authentication (uses mock LLM)
engineer = AutoFeatureEngineer(engines=['tabular', 'llm'])

# You'll see this warning:
# "Warning: copilot-sdk not installed. Using mock LLM responses."

# Features will still be generated using heuristics
X_transformed = engineer.fit_transform(X, y)
```

The mock mode generates context-aware features based on column names, but without the semantic understanding of a real LLM.

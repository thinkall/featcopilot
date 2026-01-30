"""Model utilities for Copilot client."""

from typing import Optional

from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)

# Supported models with their metadata
SUPPORTED_MODELS = {
    # OpenAI GPT models
    "gpt-5": {
        "provider": "OpenAI",
        "description": "Latest GPT-5 model with best overall performance",
        "speed": "fast",
        "quality": "excellent",
        "recommended": True,
    },
    "gpt-5-mini": {
        "provider": "OpenAI",
        "description": "Smaller, faster GPT-5 variant",
        "speed": "very fast",
        "quality": "very good",
        "recommended": False,
    },
    "gpt-5.1": {
        "provider": "OpenAI",
        "description": "GPT-5.1 with improved reasoning",
        "speed": "fast",
        "quality": "excellent",
        "recommended": False,
    },
    "gpt-5.1-codex": {
        "provider": "OpenAI",
        "description": "GPT-5.1 Codex optimized for code generation",
        "speed": "fast",
        "quality": "excellent",
        "recommended": True,
    },
    "gpt-5.1-codex-mini": {
        "provider": "OpenAI",
        "description": "Smaller GPT-5.1 Codex variant",
        "speed": "very fast",
        "quality": "good",
        "recommended": False,
    },
    "gpt-5.2": {
        "provider": "OpenAI",
        "description": "GPT-5.2 with enhanced capabilities",
        "speed": "fast",
        "quality": "excellent",
        "recommended": False,
    },
    "gpt-5.2-codex": {
        "provider": "OpenAI",
        "description": "GPT-5.2 Codex for advanced code tasks",
        "speed": "fast",
        "quality": "excellent",
        "recommended": False,
    },
    "gpt-4.1": {
        "provider": "OpenAI",
        "description": "GPT-4.1 - fast and efficient",
        "speed": "very fast",
        "quality": "good",
        "recommended": False,
    },
    # Anthropic Claude models
    "claude-sonnet-4": {
        "provider": "Anthropic",
        "description": "Claude Sonnet 4 - balanced performance",
        "speed": "medium",
        "quality": "excellent",
        "recommended": True,
    },
    "claude-sonnet-4.5": {
        "provider": "Anthropic",
        "description": "Claude Sonnet 4.5 - improved reasoning",
        "speed": "medium",
        "quality": "excellent",
        "recommended": False,
    },
    "claude-haiku-4.5": {
        "provider": "Anthropic",
        "description": "Claude Haiku 4.5 - fast and efficient",
        "speed": "very fast",
        "quality": "good",
        "recommended": False,
    },
    "claude-opus-4.5": {
        "provider": "Anthropic",
        "description": "Claude Opus 4.5 - premium quality",
        "speed": "slow",
        "quality": "premium",
        "recommended": False,
    },
    # Google Gemini models
    "gemini-3-pro-preview": {
        "provider": "Google",
        "description": "Gemini 3 Pro Preview",
        "speed": "medium",
        "quality": "excellent",
        "recommended": False,
    },
}

# Default model
DEFAULT_MODEL = "gpt-5"


def list_models(
    provider: Optional[str] = None,
    recommended_only: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """
    List all supported models for the Copilot client.

    Parameters
    ----------
    provider : str, optional
        Filter by provider ('OpenAI', 'Anthropic', 'Google')
    recommended_only : bool, default=False
        If True, only return recommended models
    verbose : bool, default=False
        If True, print model information

    Returns
    -------
    list[dict]
        List of model information dictionaries with keys:
        - name: model identifier
        - provider: model provider
        - description: model description
        - speed: speed rating
        - quality: quality rating
        - recommended: whether model is recommended

    Examples
    --------
    >>> from featcopilot.utils import list_models
    >>> models = list_models()
    >>> for m in models:
    ...     print(f"{m['name']}: {m['description']}")

    >>> # Get only recommended models
    >>> recommended = list_models(recommended_only=True)

    >>> # Filter by provider
    >>> claude_models = list_models(provider='Anthropic')
    """
    models = []

    for name, info in SUPPORTED_MODELS.items():
        # Apply filters
        if provider and info["provider"].lower() != provider.lower():
            continue
        if recommended_only and not info.get("recommended", False):
            continue

        model_info = {"name": name, **info}
        models.append(model_info)

    if verbose:
        _print_models_table(models)

    return models


def get_model_info(model_name: str) -> Optional[dict]:
    """
    Get information about a specific model.

    Parameters
    ----------
    model_name : str
        The model identifier

    Returns
    -------
    dict or None
        Model information if found, None otherwise

    Examples
    --------
    >>> from featcopilot.utils import get_model_info
    >>> info = get_model_info('gpt-5')
    >>> print(info['description'])
    """
    if model_name in SUPPORTED_MODELS:
        return {"name": model_name, **SUPPORTED_MODELS[model_name]}
    return None


def get_default_model() -> str:
    """
    Get the default model name.

    Returns
    -------
    str
        The default model identifier

    Examples
    --------
    >>> from featcopilot.utils import get_default_model
    >>> model = get_default_model()
    >>> print(model)  # 'gpt-5'
    """
    return DEFAULT_MODEL


def get_recommended_models() -> list[str]:
    """
    Get list of recommended model names.

    Returns
    -------
    list[str]
        List of recommended model identifiers

    Examples
    --------
    >>> from featcopilot.utils import get_recommended_models
    >>> models = get_recommended_models()
    >>> print(models)  # ['gpt-5', 'gpt-5.1-codex', 'claude-sonnet-4']
    """
    return [name for name, info in SUPPORTED_MODELS.items() if info.get("recommended", False)]


def is_valid_model(model_name: str) -> bool:
    """
    Check if a model name is valid/supported.

    Parameters
    ----------
    model_name : str
        The model identifier to check

    Returns
    -------
    bool
        True if model is supported, False otherwise

    Examples
    --------
    >>> from featcopilot.utils import is_valid_model
    >>> is_valid_model('gpt-5')  # True
    >>> is_valid_model('invalid-model')  # False
    """
    return model_name in SUPPORTED_MODELS


def _print_models_table(models: list[dict]) -> None:
    """Print models in a formatted table."""
    if not models:
        logger.info("No models found matching criteria.")
        return

    # Header
    header = f"{'Model':<25} {'Provider':<12} {'Speed':<12} {'Quality':<12} {'Recommended':<12}"
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for model in models:
        rec = "✓" if model.get("recommended") else ""
        line = f"{model['name']:<25} {model['provider']:<12} {model['speed']:<12} {model['quality']:<12} {rec:<12}"
        lines.append(line)

    lines.append(separator)

    for line in lines:
        logger.info(line)

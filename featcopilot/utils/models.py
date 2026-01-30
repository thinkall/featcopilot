"""Model utilities for Copilot client."""

import asyncio
from typing import Optional

from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)

# Cache for models fetched from Copilot
_cached_models: Optional[list[dict]] = None

# Default model
DEFAULT_MODEL = "gpt-5.2"


async def _fetch_models_from_copilot() -> list[dict]:
    """Fetch available models from Copilot SDK."""
    try:
        from copilot import CopilotClient

        client = CopilotClient()
        await client.start()

        # Get available models from Copilot
        models = await client.list_models()
        await client.stop()

        return models

    except ImportError:
        logger.warning("copilot-sdk not installed. Cannot fetch models from Copilot.")
        return []
    except Exception as e:
        logger.warning(f"Could not fetch models from Copilot: {e}")
        return []


def _get_event_loop():
    """Get or create an event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def fetch_models(force_refresh: bool = False) -> list[dict]:
    """
    Fetch available models from the Copilot client.

    This function queries the Copilot SDK to get the current list of
    supported models. Results are cached for subsequent calls.

    Parameters
    ----------
    force_refresh : bool, default=False
        If True, bypass cache and fetch fresh model list

    Returns
    -------
    list[dict]
        List of model information dictionaries. Each dict contains
        model metadata from the Copilot API.

    Examples
    --------
    >>> from featcopilot.utils import fetch_models
    >>> models = fetch_models()
    >>> for m in models:
    ...     print(m.get('id') or m.get('name'))
    """
    global _cached_models

    if _cached_models is not None and not force_refresh:
        return _cached_models

    loop = _get_event_loop()
    models = loop.run_until_complete(_fetch_models_from_copilot())

    if models:
        _cached_models = models

    return models


def list_models(
    provider: Optional[str] = None,
    verbose: bool = False,
    force_refresh: bool = False,
) -> list[dict]:
    """
    List all supported models from the Copilot client.

    Retrieves the current list of available models directly from
    the Copilot SDK.

    Parameters
    ----------
    provider : str, optional
        Filter by provider (e.g., 'OpenAI', 'Anthropic', 'Google')
    verbose : bool, default=False
        If True, print model information to logger
    force_refresh : bool, default=False
        If True, bypass cache and fetch fresh model list

    Returns
    -------
    list[dict]
        List of model information dictionaries from Copilot API

    Examples
    --------
    >>> from featcopilot.utils import list_models
    >>> models = list_models()
    >>> for m in models:
    ...     print(m)

    >>> # With verbose output
    >>> list_models(verbose=True)

    >>> # Filter by provider (if supported by returned data)
    >>> openai_models = list_models(provider='OpenAI')
    """
    models = fetch_models(force_refresh=force_refresh)

    # Apply provider filter if specified
    if provider and models:
        filtered = []
        for m in models:
            model_provider = m.get("provider", "") or m.get("vendor", "") or ""
            if provider.lower() in model_provider.lower():
                filtered.append(m)
        models = filtered

    if verbose:
        _print_models(models)

    return models


def get_model_info(model_name: str, force_refresh: bool = False) -> Optional[dict]:
    """
    Get information about a specific model from Copilot.

    Parameters
    ----------
    model_name : str
        The model identifier
    force_refresh : bool, default=False
        If True, bypass cache and fetch fresh model list

    Returns
    -------
    dict or None
        Model information if found, None otherwise

    Examples
    --------
    >>> from featcopilot.utils import get_model_info
    >>> info = get_model_info('gpt-5.2')
    >>> if info:
    ...     print(info)
    """
    models = fetch_models(force_refresh=force_refresh)

    for model in models:
        # Check various possible name fields
        if model.get("id") == model_name:
            return model
        if model.get("name") == model_name:
            return model
        if model.get("model") == model_name:
            return model

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
    >>> print(model)  # 'gpt-5.2'
    """
    return DEFAULT_MODEL


def get_model_names(force_refresh: bool = False) -> list[str]:
    """
    Get list of available model names/identifiers.

    Parameters
    ----------
    force_refresh : bool, default=False
        If True, bypass cache and fetch fresh model list

    Returns
    -------
    list[str]
        List of model identifiers

    Examples
    --------
    >>> from featcopilot.utils import get_model_names
    >>> names = get_model_names()
    >>> print(names)
    """
    models = fetch_models(force_refresh=force_refresh)
    names = []

    for m in models:
        name = m.get("id") or m.get("name") or m.get("model")
        if name:
            names.append(name)

    return names


def is_valid_model(model_name: str, force_refresh: bool = False) -> bool:
    """
    Check if a model name is valid/supported.

    Parameters
    ----------
    model_name : str
        The model identifier to check
    force_refresh : bool, default=False
        If True, bypass cache and fetch fresh model list

    Returns
    -------
    bool
        True if model is supported, False otherwise

    Examples
    --------
    >>> from featcopilot.utils import is_valid_model
    >>> is_valid_model('gpt-5.2')
    """
    models = fetch_models(force_refresh=force_refresh)

    if not models:
        # If we couldn't fetch models, allow any model name
        # (let the Copilot API validate it)
        logger.warning("Could not validate model - Copilot unavailable")
        return True

    return model_name in get_model_names(force_refresh=False)


def _print_models(models: list[dict]) -> None:
    """Print models information."""
    if not models:
        logger.info("No models found. Copilot SDK may not be available.")
        return

    logger.info("Available models from Copilot:")
    logger.info("-" * 60)

    for model in models:
        model_id = model.get("id") or model.get("name") or model.get("model") or "unknown"
        description = model.get("description", "")
        provider = model.get("provider") or model.get("vendor") or ""

        line = f"  {model_id}"
        if provider:
            line += f" ({provider})"
        if description:
            line += f" - {description[:50]}..."

        logger.info(line)

    logger.info("-" * 60)
    logger.info(f"Total: {len(models)} models")

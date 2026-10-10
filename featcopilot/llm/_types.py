"""Dependency-independent interfaces for optional LLM providers."""

from collections.abc import Sequence
from typing import Any, Protocol

from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class CompletionMessage(Protocol):
    """Text content returned by a chat completion."""

    @property
    def content(self) -> str | None: ...


class CompletionChoice(Protocol):
    """A single chat completion choice."""

    @property
    def message(self) -> CompletionMessage: ...


class CompletionResponse(Protocol):
    """Non-streaming chat completion response."""

    @property
    def choices(self) -> Sequence[CompletionChoice]: ...


class LiteLLMProvider(Protocol):
    """Async completion interface used from the optional LiteLLM module."""

    async def acompletion(self, **kwargs: Any) -> CompletionResponse: ...


class ChatCompletions(Protocol):
    """OpenAI-compatible async chat completions."""

    async def create(self, **kwargs: Any) -> CompletionResponse: ...


class ChatResource(Protocol):
    """Chat resource exposed by an OpenAI-compatible client."""

    @property
    def completions(self) -> ChatCompletions: ...


class OpenAIProvider(Protocol):
    """OpenAI-compatible client interface used by feature generation."""

    @property
    def chat(self) -> ChatResource: ...

    async def close(self) -> None: ...


class FeatureClient(Protocol):
    """Synchronous feature-generation interface shared by LLM backends."""

    def start(self) -> object: ...

    def stop(self) -> None: ...

    def send_prompt(self, prompt: str) -> str: ...

    def suggest_features(self, **kwargs: Any) -> list[dict[str, Any]]: ...

    def explain_feature(self, **kwargs: Any) -> str: ...

    def generate_feature_code(self, **kwargs: Any) -> str: ...

    def validate_feature_code(self, code: str, sample_data: dict[str, list] | None = None) -> dict[str, Any]: ...


def feature_suggestions(data: object) -> list[dict[str, Any]]:
    """Extract feature objects from a decoded provider response."""
    if not isinstance(data, dict):
        logger.warning("Expected a JSON object containing feature suggestions")
        return []
    features = data.get("features")
    if not isinstance(features, list):
        logger.warning("LLM response has no feature suggestions array")
        return []
    suggestions = [feature for feature in features if isinstance(feature, dict)]
    if len(suggestions) != len(features):
        logger.warning("Ignoring non-object feature suggestions")
    return suggestions

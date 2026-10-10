"""Runtime regressions covered by the package's typing boundaries."""

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from featcopilot.engines.text import TextEngine
from featcopilot.llm.copilot_client import CopilotFeatureClient, SyncCopilotFeatureClient
from featcopilot.llm.litellm_client import LiteLLMFeatureClient, SyncLiteLLMFeatureClient
from featcopilot.llm.openai_client import OpenAIFeatureClient, SyncOpenAIFeatureClient
from featcopilot.llm.semantic_engine import SemanticEngine
from featcopilot.stores.feast_store import FeastFeatureStore
from featcopilot.utils.cache import FeatureCache
from featcopilot.utils.models import _fetch_models_from_copilot

ASYNC_CLIENTS = [CopilotFeatureClient, OpenAIFeatureClient, LiteLLMFeatureClient]
SYNC_CLIENTS = [SyncCopilotFeatureClient, SyncOpenAIFeatureClient, SyncLiteLLMFeatureClient]


@pytest.mark.parametrize("client_type", ASYNC_CLIENTS)
@pytest.mark.parametrize("payload", [None, [], {"features": None}, {"features": {}}, {"features": [None, 1, "name"]}])
def test_suggestion_parsers_reject_non_feature_json(client_type, payload):
    """Valid JSON need not contain usable feature objects."""
    assert client_type()._parse_suggestions(json.dumps(payload)) == []


@pytest.mark.parametrize("client_type", ASYNC_CLIENTS)
def test_suggestion_parsers_keep_valid_objects(client_type):
    """Malformed entries must not discard the remaining feature objects."""
    feature = {"name": "ratio", "code": "result = df['a'] / df['b']"}
    payload = json.dumps({"features": [None, feature, 4]})
    assert client_type()._parse_suggestions(f"Here are features: {payload}") == [feature]


def test_text_feature_parser_validates_json_shape():
    """Semantic text suggestions use the same validated object boundary."""
    engine = SemanticEngine()
    assert engine._parse_text_features("null", "review") == []
    assert engine._parse_text_features('{"features": [null, {"name": "length"}]}', "review") == [
        {"name": "length", "source_columns": ["review"], "is_text_feature": True}
    ]


@pytest.mark.parametrize("client_type", SYNC_CLIENTS)
def test_sync_bridge_preserves_result_and_original_runtime_error(client_type):
    """Coroutine errors must propagate without retrying consumed coroutines."""
    client = client_type()
    result = {"features": [{"name": "ratio"}]}
    error = RuntimeError("provider failed")
    attempts = 0

    async def successful():
        return result

    async def failing():
        nonlocal attempts
        attempts += 1
        raise error

    try:
        assert client._run_async(successful()) is result
        with pytest.raises(RuntimeError) as caught:
            client._run_async(failing())
        assert caught.value is error
        assert attempts == 1
    finally:
        client.stop()


@pytest.mark.parametrize("client_type", SYNC_CLIENTS)
async def test_sync_bridge_in_running_loop_without_nest_asyncio(client_type, monkeypatch):
    """The thread fallback retains result types and original exceptions."""
    monkeypatch.setitem(sys.modules, "nest_asyncio", None)
    client = client_type()
    caller_loop = asyncio.get_running_loop()
    result = ["feature"]
    error = RuntimeError("provider failed")
    attempts = 0

    async def successful():
        assert asyncio.get_running_loop() is not caller_loop
        return result

    async def failing():
        nonlocal attempts
        attempts += 1
        raise error

    try:
        assert client._run_async(successful()) is result
        with pytest.raises(RuntimeError) as caught:
            client._run_async(failing())
        assert caught.value is error
        assert attempts == 1
    finally:
        client.stop()


@pytest.mark.parametrize("client_type", [OpenAIFeatureClient, LiteLLMFeatureClient])
async def test_chat_clients_reject_empty_content(client_type):
    """Null provider content must not masquerade as a successful generation."""
    client = client_type()
    client._is_started = True
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    completion = AsyncMock(return_value=response)
    if isinstance(client, OpenAIFeatureClient):
        client._openai_available = True
        client._async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    else:
        client._litellm_available = True
        client._litellm = SimpleNamespace(acompletion=completion)
    with pytest.raises(ValueError, match="empty"):
        await client.send_prompt("hello")
    completion.assert_awaited_once()


def test_litellm_sync_client_supports_direct_prompts():
    """Rule generation requires send_prompt on every synchronous backend."""
    client = SyncLiteLLMFeatureClient()
    client._async_client.send_prompt = AsyncMock(return_value="generated rule")
    assert client.send_prompt("generate", system_prompt="rules") == "generated rule"
    client._async_client.send_prompt.assert_awaited_once_with("generate", system_prompt="rules")


@pytest.mark.parametrize("legacy_dict", [False, True])
async def test_copilot_model_metadata_normalizes_sdk_responses(monkeypatch, legacy_dict):
    """Current SDK dataclasses and older dictionaries expose the same API."""

    @dataclass
    class Capabilities:
        vision: bool

    @dataclass
    class Model:
        id: str
        capabilities: Capabilities

    expected = {"id": "test-model", "capabilities": {"vision": True}}
    model = expected if legacy_dict else Model("test-model", Capabilities(True))
    client = SimpleNamespace(start=AsyncMock(), list_models=AsyncMock(return_value=[model]), stop=AsyncMock())
    monkeypatch.setitem(sys.modules, "copilot", SimpleNamespace(CopilotClient=lambda: client))
    assert await _fetch_models_from_copilot() == [expected]
    client.stop.assert_awaited_once()


async def test_copilot_model_fetch_stops_client_after_failure(monkeypatch):
    """A failed metadata request must still release the SDK client."""
    client = SimpleNamespace(
        start=AsyncMock(), list_models=AsyncMock(side_effect=RuntimeError("unavailable")), stop=AsyncMock()
    )
    monkeypatch.setitem(sys.modules, "copilot", SimpleNamespace(CopilotClient=lambda: client))
    assert await _fetch_models_from_copilot() == []
    client.stop.assert_awaited_once()


def test_cache_metadata_rejects_non_object_json(tmp_path):
    """Corrupt metadata falls back to the documented optional dictionary."""
    cache = FeatureCache(cache_dir=str(tmp_path))
    cache_key = cache._get_cache_key("feature", None)
    (tmp_path / f"{cache_key}.meta.json").write_text("[]")
    assert cache.get_metadata("feature") is None


def test_feast_requires_client_and_repository_after_initialization(monkeypatch):
    """An incomplete store initialization raises contextual runtime errors."""
    store = FeastFeatureStore()
    initialize = MagicMock()
    monkeypatch.setattr(store, "initialize", initialize)
    with pytest.raises(RuntimeError, match="did not create a client"):
        store.list_feature_views()
    initialize.assert_called_once()
    with pytest.raises(RuntimeError, match="repository path is not initialized"):
        store._get_repo_path()


def test_sentiment_pipeline_uses_canonical_task(monkeypatch):
    """The canonical text-classification task supports typed transformers."""
    pipeline = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    engine = TextEngine(features=["sentiment"])
    engine._load_sentiment()
    pipeline.assert_called_once_with(
        "text-classification", model=engine.config.sentiment_model, truncation=True, max_length=512
    )
    assert engine._sentiment_pipeline is pipeline.return_value


def test_optional_integrations_remain_importable_without_sdks():
    """Annotations must not introduce imports of optional runtime packages."""
    script = """
import builtins
original_import = builtins.__import__
blocked = {"copilot", "openai", "litellm", "feast", "transformers", "nest_asyncio"}
def base_install_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise ImportError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = base_install_import
import featcopilot
import featcopilot.llm.copilot_client
import featcopilot.llm.litellm_client
import featcopilot.llm.openai_client
import featcopilot.stores.feast_store
import featcopilot.engines.text
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

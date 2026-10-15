"""Runtime regressions covered by the package's typing boundaries."""

import asyncio
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from featcopilot.engines.point_in_time import PointInTimeRelationalEngine
from featcopilot.engines.text import TextEngine
from featcopilot.engines.timeseries import TimeSeriesEngine
from featcopilot.llm.copilot_client import CopilotFeatureClient, SyncCopilotFeatureClient
from featcopilot.llm.litellm_client import LiteLLMFeatureClient, SyncLiteLLMFeatureClient
from featcopilot.llm.openai_client import OpenAIFeatureClient, SyncOpenAIFeatureClient
from featcopilot.llm.semantic_engine import SemanticEngine
from featcopilot.selection.importance import ImportanceSelector
from featcopilot.stores.feast_store import FeastFeatureStore
from featcopilot.utils.cache import FeatureCache
from featcopilot.utils.models import _fetch_models_from_copilot

ASYNC_CLIENTS = [CopilotFeatureClient, OpenAIFeatureClient, LiteLLMFeatureClient]
SYNC_CLIENTS = [SyncCopilotFeatureClient, SyncOpenAIFeatureClient, SyncLiteLLMFeatureClient]


@pytest.mark.parametrize("client_type", [SyncOpenAIFeatureClient, SyncLiteLLMFeatureClient])
@pytest.mark.parametrize("nested_first", [False, True])
@pytest.mark.parametrize("nest_available", [False, True])
def test_pooled_sync_client_owns_one_loop_for_mixed_call_contexts(
    client_type, nested_first, nest_available, monkeypatch
):
    nest = SimpleNamespace(apply=MagicMock(side_effect=AssertionError("Do not patch caller loops")))
    monkeypatch.setitem(sys.modules, "nest_asyncio", nest if nest_available else None)
    client = client_type()
    owner = None
    seen = []
    failure = RuntimeError("original provider failure")

    async def start():
        nonlocal owner
        owner = (asyncio.get_running_loop(), threading.get_ident())
        seen.append("start")

    async def send(prompt, system_prompt=None):
        assert (asyncio.get_running_loop(), threading.get_ident()) == owner
        seen.append(prompt)
        if prompt == "failure":
            raise failure
        return prompt

    async def stop():
        assert (asyncio.get_running_loop(), threading.get_ident()) == owner
        seen.append("stop")

    client._async_client = SimpleNamespace(start=start, send_prompt=send, stop=stop)

    async def nested_call(method, *args):
        caller_loop = asyncio.get_running_loop()
        result = method(*args)
        assert not caller_loop.is_closed()
        return result

    def invoke(nested, method, *args):
        return asyncio.run(nested_call(method, *args)) if nested else method(*args)

    try:
        invoke(nested_first, client.start)
        assert invoke(not nested_first, client.send_prompt, "first") == "first"
        with pytest.raises(RuntimeError) as caught:
            invoke(nested_first, client.send_prompt, "failure")
        assert caught.value is failure
        assert invoke(not nested_first, client.send_prompt, "second") == "second"
    finally:
        invoke(not nested_first, client.stop)
    assert seen == ["start", "first", "failure", "second", "stop"]
    assert owner[0].is_closed()
    assert client._loop is None
    assert not any(thread.ident == owner[1] and thread.is_alive() for thread in threading.enumerate())
    if nest_available:
        nest.apply.assert_not_called()


@pytest.mark.parametrize("client_type", [SyncOpenAIFeatureClient, SyncLiteLLMFeatureClient])
def test_pooled_sync_client_closes_worker_on_stop_error_and_can_restart(client_type):
    client = client_type()
    original = RuntimeError("close failed")
    loops = []

    async def start():
        loops.append(asyncio.get_running_loop())

    client._async_client = SimpleNamespace(start=start, stop=AsyncMock(side_effect=original))
    client.start()
    with pytest.raises(RuntimeError) as caught:
        client.stop()
    assert caught.value is original
    assert loops[0].is_closed()
    assert client._loop is client._executor is None
    client._async_client.stop.side_effect = None
    try:
        client.start()
        assert loops[1] is not loops[0]
    finally:
        client.stop()
    assert loops[1].is_closed()


@pytest.mark.parametrize("client_type", [SyncOpenAIFeatureClient, SyncLiteLLMFeatureClient])
def test_concurrent_first_calls_create_one_provider_loop_and_worker(client_type, monkeypatch):
    client = client_type()
    competing = threading.Event()
    executors = []

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.count_lock = threading.Lock()
            self.attempts = 0

        def __enter__(self):
            with self.count_lock:
                self.attempts += 1
                if self.attempts == 2:
                    competing.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    def create_executor(**kwargs):
        executor = ThreadPoolExecutor(**kwargs)
        executors.append(executor)
        if len(executors) == 2:
            competing.set()
        assert competing.wait(timeout=2), "Second caller did not contend during initialization"
        return executor

    monkeypatch.setattr(client, "_executor_lock", ObservedLock())
    monkeypatch.setattr(sys.modules[client_type.__module__], "ThreadPoolExecutor", create_executor)

    async def identify_owner():
        return asyncio.get_running_loop(), threading.get_ident()

    coroutines = [identify_owner(), identify_owner()]
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            futures = [callers.submit(client._run_async, coroutine) for coroutine in coroutines]
            owners = [future.result(timeout=5) for future in futures]
        assert len(executors) == 1
        assert owners[0] == owners[1]
    finally:
        client.stop()
        for executor in executors:
            executor.shutdown(wait=True)
        for coroutine in coroutines:
            coroutine.close()
    assert owners[0][0].is_closed()
    assert not any(thread.ident == owners[0][1] and thread.is_alive() for thread in threading.enumerate())


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
blocked = {"copilot", "openai", "litellm", "feast", "transformers", "nest_asyncio", "xgboost"}
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
import featcopilot.selection.importance
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("is_classification", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_importance_optional_model_preserves_constructor_and_fallback(monkeypatch, is_classification, available):
    """Protocol annotations preserve optional learners and the dependency-free fallback."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    classifier = MagicMock()
    regressor = MagicMock()
    module = SimpleNamespace(XGBClassifier=classifier, XGBRegressor=regressor) if available else None
    monkeypatch.setitem(sys.modules, "xgboost", module)
    selector = ImportanceSelector(model="xgboost", n_estimators=7)

    model = selector._create_model(is_classification)

    if available:
        chosen, unused = (classifier, regressor) if is_classification else (regressor, classifier)
        assert model is chosen.return_value
        chosen.assert_called_once_with(n_estimators=7, random_state=42, n_jobs=-1)
        unused.assert_not_called()
    else:
        assert isinstance(model, RandomForestClassifier if is_classification else RandomForestRegressor)
        assert model.n_estimators == 7
        assert model.random_state == 42
        assert model.n_jobs is None


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_timeseries_numpy_index_scalars_preserve_feature_values(dtype):
    """Count ratios and extrema locations are real Python scalars across NumPy dtypes."""
    engine = TimeSeriesEngine()
    series = np.array([2, 0, 3, 0, 3, 1], dtype=dtype)
    features = engine._extract_counts(series, "value")
    features.update(engine._extract_complexity(series, "value"))
    features.update(engine._extract_fft(series, "value"))

    expected = {
        "value_first_loc_max": 2 / 6,
        "value_first_loc_min": 1 / 6,
        "value_last_loc_max": 4 / 6,
        "value_last_loc_min": 3 / 6,
        "value_pct_reoccurring_data_points": 4 / 6,
    }
    for name, value in expected.items():
        assert type(features[name]) is float
        assert features[name] == pytest.approx(value)
    magnitude = np.abs(np.fft.fft(series))
    assert type(features["value_dominant_freq_idx"]) is int
    assert features["value_dominant_freq_idx"] == int(np.argmax(magnitude[1:3])) + 1
    assert features["value_spectral_energy"] == pytest.approx(np.sum(magnitude**2))
    for coefficient in range(1, 4):
        assert features[f"value_fft_coeff_{coefficient}"] == pytest.approx(magnitude[coefficient])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_approximate_entropy_preserves_normalized_match_counts(dtype):
    """Explicit array typing must not change entropy normalization or short-series behavior."""
    engine = TimeSeriesEngine()
    series = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=dtype)
    expected = (4 * np.log(4 / 7 + 1e-10) + 3 * np.log(3 / 7 + 1e-10)) / 7 - np.log(0.5 + 1e-10)
    assert engine._approximate_entropy(series) == pytest.approx(expected)
    assert engine._approximate_entropy(series[:3]) == 0
    assert engine._approximate_entropy(np.ones(8, dtype=dtype)) == 0


@pytest.mark.parametrize("aggregation,empty", [("count", 0.0), ("sum", 0.0), ("mean", np.nan)])
def test_point_in_time_float_slices_preserve_unknown_and_known_entities(aggregation, empty):
    """Empty and populated float arrays share the aggregate boundary without changing cutoffs."""
    query = pd.DataFrame(
        {"entity": ["new", "known", "known"], "when": pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-01"])},
        index=[5, 2, 5],
    )
    history = pd.DataFrame(
        {
            "entity": ["known", "known", "known"],
            "available": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "value": pd.array([1.5, None, 99.0], dtype="Float64"),
        }
    )
    engine = PointInTimeRelationalEngine(
        [
            {
                "name": "past",
                "table": "events",
                "entity_column": "entity",
                "available_at": "available",
                "value_column": "value",
                "aggregation": aggregation,
            }
        ],
        entity_column="entity",
        prediction_time_column="when",
    )
    result = engine.fit_transform(query, related_tables={"events": history})
    pd.testing.assert_frame_equal(result[query.columns], query)
    np.testing.assert_allclose(result["past"], [empty, 1.0 if aggregation == "count" else 1.5, empty], equal_nan=True)

"""Deterministic LLM execution and provider failure regressions; no network calls."""

import asyncio
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from featcopilot.core.feature import FeatureOrigin
from featcopilot.llm.code_generator import FeatureCodeGenerator
from featcopilot.llm.copilot_client import CopilotConfig, CopilotFeatureClient
from featcopilot.llm.litellm_client import LiteLLMConfig, LiteLLMFeatureClient, SyncLiteLLMFeatureClient
from featcopilot.llm.openai_client import OpenAIClientConfig, OpenAIFeatureClient, SyncOpenAIFeatureClient
from featcopilot.llm.semantic_engine import SemanticEngine, SemanticEngineConfig
from featcopilot.llm.transform_rule_generator import TransformRuleGenerator
from featcopilot.utils.models import DEFAULT_MODEL

CLIENTS = [CopilotFeatureClient, OpenAIFeatureClient, LiteLLMFeatureClient]


def make_engine(suggestions, **kwargs):
    engine = SemanticEngine(**kwargs)
    engine._client = MagicMock()
    engine._client.suggest_features.return_value = suggestions
    engine._client.validate_feature_code.return_value = {"valid": True, "error": None, "warnings": []}
    return engine


@pytest.mark.parametrize("config_type", [CopilotConfig, OpenAIClientConfig, LiteLLMConfig, SemanticEngineConfig])
def test_shared_project_default_and_explicit_override(config_type):
    assert DEFAULT_MODEL == "gpt-5.6-sol"
    assert config_type().model == DEFAULT_MODEL
    assert config_type(model="custom/deployment").model == "custom/deployment"


@pytest.mark.parametrize("client_type", CLIENTS)
def test_validation_supports_nested_scope_and_intermediate_values(client_type):
    code = "scale = 2\nresult = df['x'].apply(lambda x: np.sqrt(x) * scale)"
    result = asyncio.run(client_type().validate_feature_code(code, {"x": [1, 4, 9]}))
    assert result["valid"], result["error"]


@pytest.mark.parametrize("client_type", CLIENTS)
@pytest.mark.parametrize(
    "code",
    ["", "x = 1", "result = None", "result = [1]", "result = df[['x', 'y']]", "result = df['x'] * np.nan"],
)
def test_validation_rejects_non_features(client_type, code):
    result = asyncio.run(client_type().validate_feature_code(code, {"x": [1, 2], "y": [3, 4]}))
    assert not result["valid"]
    assert result["error"]


@pytest.mark.parametrize("client_type", CLIENTS)
@pytest.mark.parametrize("response", ["", "```python\n```", "```\n \n```"])
def test_code_generation_rejects_empty_extracted_code(client_type, response):
    client = client_type()
    client.send_prompt = AsyncMock(return_value=response)
    with pytest.raises(ValueError, match="empty code"):
        asyncio.run(client.generate_feature_code("Synthetic square", {"x": "float"}))


def test_fit_transform_uses_same_scope_and_preserves_index():
    engine = make_engine([{"name": "root", "code": "scale = 2\nresult = df['x'].apply(lambda x: np.sqrt(x) * scale)"}])
    frame = pd.DataFrame({"x": [1, 4, 9]}, index=[8, 3, 2])
    result = engine.fit_transform(frame)
    pd.testing.assert_series_equal(result["root"], pd.Series([2.0, 4.0, 6.0], index=frame.index, name="root"))
    assert engine.get_feature_names() == ["root"]
    engine._client.suggest_features.assert_called_once()
    engine.transform(frame.iloc[:1])
    engine._client.suggest_features.assert_called_once()


def test_series_outputs_align_by_labels_not_position():
    engine = make_engine([{"name": "sorted_x", "code": "result = df['x'].sort_values()"}])
    frame = pd.DataFrame({"x": [3, 1, 2]}, index=[30, 10, 20])
    result = engine.fit_transform(frame)
    pd.testing.assert_series_equal(result["sorted_x"], frame["x"].rename("sorted_x"))


def test_serialized_rules_transform_without_a_provider_and_allow_missing_values():
    engine = make_engine([{"name": "squared", "code": "result = df['x'] ** 2"}])
    engine.fit(pd.DataFrame({"x": [1, 2]}))
    restored = pickle.loads(pickle.dumps(engine))
    assert restored._client is None
    assert restored.get_feature_names() == ["squared"]
    frame = pd.DataFrame({"x": [np.nan]}, index=[9])
    result = restored.transform(frame)
    assert list(result.columns) == ["x", "squared"]
    assert result["squared"].isna().all()


@pytest.mark.parametrize("client_type", [SyncOpenAIFeatureClient, SyncLiteLLMFeatureClient])
def test_sync_chat_requests_reuse_the_provider_event_loop(client_type):
    client = client_type()
    loops = []

    async def send_prompt(*args, **kwargs):
        loops.append(asyncio.get_running_loop())
        return "ok"

    client._async_client.send_prompt = send_prompt
    try:
        assert client.send_prompt("first") == client.send_prompt("second") == "ok"
        assert loops[0] is loops[1]
        assert not loops[0].is_closed()
    finally:
        client.stop()
    assert loops[0].is_closed()


@pytest.mark.parametrize(
    "backend,module,class_name",
    [
        ("copilot", "featcopilot.llm.copilot_client", "SyncCopilotFeatureClient"),
        ("openai", "featcopilot.llm.openai_client", "SyncOpenAIFeatureClient"),
        ("litellm", "featcopilot.llm.litellm_client", "SyncLiteLLMFeatureClient"),
    ],
)
def test_semantic_engine_forwards_supported_provider_configuration(backend, module, class_name, monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(f"{module}.{class_name}", factory)
    engine = SemanticEngine(
        backend=backend,
        model="custom-model",
        api_key="synthetic-key",
        api_base="http://localhost",
        temperature=0.7,
        timeout=12.0,
        max_tokens=321,
    )
    engine._ensure_client()
    expected = {"model": "custom-model", "timeout": 12.0}
    if backend != "copilot":
        expected.update(api_key="synthetic-key", api_base="http://localhost", temperature=0.7, max_tokens=321)
    if backend == "openai":
        expected["api_version"] = None
    factory.assert_called_once_with(**expected)


@pytest.mark.parametrize(
    "backend,module,class_name,kwargs",
    [
        ("copilot", "copilot_client", "SyncCopilotFeatureClient", {"temperature": 0.8, "max_tokens": 123}),
        ("litellm", "litellm_client", "SyncLiteLLMFeatureClient", {"api_version": "synthetic-version"}),
    ],
)
def test_semantic_engine_reports_unsupported_backend_settings(backend, module, class_name, kwargs, monkeypatch):
    factory = MagicMock()
    warning = MagicMock()
    monkeypatch.setattr(f"featcopilot.llm.{module}.{class_name}", factory)
    monkeypatch.setattr("featcopilot.llm.semantic_engine.logger.warning", warning)
    SemanticEngine(backend=backend, **kwargs)._ensure_client()
    warning.assert_called_once()
    assert "does not support" in warning.call_args.args[0]
    assert all(key in warning.call_args.args[0] for key in kwargs)


def test_fit_drops_invalid_and_duplicate_features_without_overwriting_inputs():
    engine = make_engine(
        [
            {"name": "x", "code": "result = df['x'] * 100"},
            {"name": "good", "code": "result = df['x'] * 2"},
            {"name": "good", "code": "result = df['x'] * 3"},
            {"name": "bad", "code": "result = [1]"},
            {"name": "empty", "code": ""},
            {"name": "missing", "code": "temporary = df['x']"},
        ]
    )
    frame = pd.DataFrame({"x": [1, 2, 3]})
    result = engine.fit_transform(frame)
    assert list(result.columns) == ["x", "good"]
    assert list(engine.get_feature_code()) == engine.get_feature_names() == ["good"]
    pd.testing.assert_series_equal(result["x"], frame["x"])


def test_transform_failure_raises_and_preserves_fitted_schema():
    engine = make_engine([{"name": "inverse", "code": "result = df['x'].apply(lambda x: 1 / x)"}])
    engine.fit(pd.DataFrame({"x": [1, 2]}))
    assert engine.get_feature_names() == ["inverse"]
    with pytest.raises(ValueError, match="inverse"):
        engine.transform(pd.DataFrame({"x": [0]}))
    assert engine.get_feature_names() == ["inverse"]


def test_refit_clears_text_features_and_failed_fit_invalidates_previous_state():
    engine = make_engine([])
    engine.fit(pd.DataFrame({"text": [f"A lengthy natural language sample number {i}." for i in range(12)]}))
    assert engine._text_features
    assert set(engine.get_feature_names()) == set(engine.get_feature_code())
    assert all(feature.origin == FeatureOrigin.CUSTOM for feature in engine.get_feature_set())
    engine.fit(pd.DataFrame({"x": [1, 2]}))
    assert engine._text_features == []
    assert engine.get_feature_names() == []
    engine._client.suggest_features.side_effect = RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        engine.fit(pd.DataFrame({"x": [3, 4]}))
    assert not engine.is_fitted
    with pytest.raises(RuntimeError, match="fitted"):
        engine.transform(pd.DataFrame({"x": [1]}))


def test_text_generated_code_is_validated_and_input_infinities_are_preserved():
    engine = make_engine([{"name": "broken", "code": "result = df['missing']"}])
    frame = pd.DataFrame({"text": [f"A lengthy text sample number {i}." for i in range(12)], "x": np.inf})
    result = engine.fit_transform(frame)
    assert "broken" not in result
    assert np.isinf(result["x"]).all()
    assert "text_avg_word_length" in result


@pytest.mark.parametrize(
    "client_type,module",
    [(CopilotFeatureClient, "copilot"), (OpenAIFeatureClient, "openai"), (LiteLLMFeatureClient, "litellm")],
)
def test_missing_provider_is_an_error_not_mock_features(client_type, module, monkeypatch):
    monkeypatch.setitem(sys.modules, module, None)
    client = client_type()
    with pytest.raises(ImportError):
        asyncio.run(client.start())
    assert not client._is_started


@pytest.mark.parametrize("client_type", [OpenAIFeatureClient, LiteLLMFeatureClient])
def test_chat_provider_errors_propagate(client_type):
    client = client_type()
    client._is_started = True
    completion = AsyncMock(side_effect=RuntimeError("provider rejected model"))
    if isinstance(client, OpenAIFeatureClient):
        client._openai_available = True
        client._async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    else:
        client._litellm_available = True
        client._litellm = SimpleNamespace(acompletion=completion)
    with pytest.raises(RuntimeError, match="provider rejected model"):
        asyncio.run(client.send_prompt("Synthetic numeric feature"))


@pytest.mark.parametrize("client_type", [OpenAIFeatureClient, LiteLLMFeatureClient])
def test_empty_chat_content_is_an_error(client_type):
    client = client_type()
    client._is_started = True
    completion = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    )
    if isinstance(client, OpenAIFeatureClient):
        client._openai_available = True
        client._async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    else:
        client._litellm_available = True
        client._litellm = SimpleNamespace(acompletion=completion)
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(client.send_prompt("Synthetic numeric feature"))


def test_litellm_passes_key_per_request_without_mutating_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    completion = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    )
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=completion))
    client = LiteLLMFeatureClient(api_key="explicit-key")
    asyncio.run(client.start())
    asyncio.run(client.send_prompt("Synthetic request"))
    import os

    assert os.environ["OPENAI_API_KEY"] == "existing-key"
    assert completion.call_args.kwargs["api_key"] == "explicit-key"


def test_generator_does_not_return_code_that_still_fails_validation():
    generator = FeatureCodeGenerator()
    generator._client = MagicMock()
    generator._client.generate_feature_code.return_value = "result = df['missing']"
    generator._client.validate_feature_code.return_value = {"valid": False, "error": "KeyError: missing"}
    with pytest.raises(ValueError, match="validation"):
        generator.generate("missing feature", {"x": "float"}, sample_data=pd.DataFrame({"x": [1]}))


@pytest.mark.parametrize("generator_type", [FeatureCodeGenerator, TransformRuleGenerator])
@pytest.mark.parametrize(
    "code,expected",
    [
        ("df['x'] == 2", [False, True]),
        ("df['result'] * 2", [6, 8]),
        ("intermediate = df['x'] * 2\noutput = intermediate + 1", [3, 5]),
    ],
)
def test_code_normalization_preserves_expressions_and_intermediate_names(generator_type, code, expected):
    from featcopilot.llm._execution import execute_feature_code

    cleaned = generator_type()._clean_code(code)
    frame = pd.DataFrame({"x": [1, 2], "result": [3, 4]})
    assert execute_feature_code(cleaned, frame).tolist() == expected


@pytest.mark.parametrize("response", ["", "No feature possible.", "{}", '{"name": "empty", "code": ""}'])
def test_rule_generator_rejects_empty_invalid_response(response):
    generator = TransformRuleGenerator()
    generator._client = MagicMock()
    generator._client.send_prompt.return_value = response
    with pytest.raises(ValueError):
        generator.generate_from_description("Synthetic ratio", {"x": "float"})

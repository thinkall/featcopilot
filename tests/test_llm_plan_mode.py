"""Provider-free tests of the opt-in typed SemanticEngine plan path."""

import asyncio
import errno
import inspect
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from featcopilot.core.feature import FeatureOrigin, FeatureType
from featcopilot.core.fitted_rule import FittedRuleSpec
from featcopilot.engines.fitted_rules import FittedRuleEngine
from featcopilot.llm.copilot_client import CopilotConfig, SyncCopilotFeatureClient
from featcopilot.llm.semantic_engine import SemanticEngine, SemanticEngineConfig
from featcopilot.utils.models import DEFAULT_MODEL


def frequency_plan():
    return {
        "rules": [
            {
                "name": "category_frequency",
                "operator": "frequency",
                "columns": ["category"],
                "parameters": {},
                "explanation": "Training prevalence of category",
            }
        ]
    }


def plan_engine(plan=None, **kwargs):
    engine = SemanticEngine(feature_format="plan", **kwargs)
    client = MagicMock()
    client.send_prompt.return_value = json.dumps(plan if plan is not None else frequency_plan())
    engine._client = client
    return engine, client


def test_plan_defaults_keep_shared_model_and_explicit_code_default():
    assert DEFAULT_MODEL == "gpt-5.6-sol"
    assert SemanticEngine().config.feature_format == "code"
    config = SemanticEngineConfig(feature_format="plan")
    assert config.model == DEFAULT_MODEL
    assert config.task == "auto"
    assert config.target_cv == 5
    assert config.random_state == 42
    with pytest.raises(ValidationError):
        SemanticEngine(feature_format="arbitrary")
    with pytest.raises(ValueError, match="require"):
        SemanticEngine(rules=[])


@pytest.mark.parametrize("config_type", [CopilotConfig, SemanticEngineConfig])
@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0, -1, True, False])
def test_timeout_configs_reject_invalid_numbers_and_booleans(config_type, timeout):
    with pytest.raises(ValidationError, match="timeout"):
        config_type(timeout=timeout)


@pytest.mark.parametrize("feature_format", ["code", "plan"])
@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0, -1, True, False])
def test_semantic_engine_rejects_invalid_timeout_at_construction(feature_format, timeout):
    with patch("featcopilot.llm.semantic_engine.SemanticEngine._ensure_client") as ensure_client:
        with pytest.raises(ValidationError, match="timeout"):
            SemanticEngine(feature_format=feature_format, timeout=timeout)
        ensure_client.assert_not_called()


@pytest.mark.parametrize("timeout", [19, 19.0, 0.03])
def test_timeout_configs_preserve_positive_integer_and_float_inputs(timeout):
    assert CopilotConfig(timeout=timeout).timeout == timeout
    assert SemanticEngineConfig(timeout=timeout).timeout == timeout
    assert SemanticEngine(feature_format="plan", timeout=timeout).config.timeout == timeout
    assert CopilotConfig().timeout == SemanticEngineConfig().timeout == SemanticEngine().config.timeout == 60.0


def test_plan_prompt_is_bounded_typed_and_contains_semantic_metadata_not_data_samples():
    X = pd.DataFrame({"category": ["PRIVATE_CATEGORY", "OTHER_CATEGORY"]})
    engine, client = plan_engine(max_suggestions=3, domain="retail")
    engine.fit(X, column_descriptions={"category": "Customer segment"}, task_description="Predict repeat purchases")
    prompt = client.send_prompt.call_args.args[0]
    for text in [
        "at most 3",
        "standardize",
        "clip",
        "frequency",
        "group_mean",
        "target_mean",
        "rare_category",
        "multiply",
        "divide",
        "log1p",
        "Customer segment",
        "Predict repeat purchases",
        "retail",
        '"target_available": false',
    ]:
        assert text in prompt
    assert "PRIVATE_CATEGORY" not in prompt
    client.send_prompt.assert_called_once()
    client.suggest_features.assert_not_called()
    client.validate_feature_code.assert_not_called()


def test_plan_transform_never_calls_provider_or_relearns_statistics():
    X = pd.DataFrame({"category": ["a", "a", "b"]}, index=[2, 2, 0])
    engine, client = plan_engine()
    training = engine.fit_transform(X)
    pd.testing.assert_frame_equal(training[X.columns], X)
    expected = pd.DataFrame(
        {"category": ["a", "unknown", None], "category_frequency": [2 / 3, 0.0, 0.0]}, index=[9, 3, 9]
    )
    with patch.object(engine, "_ensure_client", side_effect=AssertionError("No inference provider calls")):
        output = engine.transform(expected[["category"]])
        pd.testing.assert_frame_equal(output, expected)
        pd.testing.assert_frame_equal(
            output,
            pd.concat(
                [engine.transform(expected[["category"]].iloc[:1]), engine.transform(expected[["category"]].iloc[1:])]
            ),
        )
    client.send_prompt.assert_called_once()


def test_training_hook_uses_out_of_fold_target_statistics():
    plan = {
        "rules": [
            {"name": "encoded", "operator": "target_mean", "columns": ["category"], "parameters": {"smoothing": 0}}
        ]
    }
    X = pd.DataFrame({"category": ["a", "b", "c", None]}, index=[4, 4, 9, 2])
    y = pd.Series([0.0, 2.0, 4.0, 10.0], index=[8, 7, 6, 5])
    engine, client = plan_engine(plan, task="regression", target_cv=4)
    output = engine.fit_transform(X, y)
    np.testing.assert_allclose(output["encoded"], (y.sum() - y.to_numpy()) / 3)
    np.testing.assert_allclose(engine.transform(X)["encoded"], [0, 2, 4, 4])
    client.send_prompt.assert_called_once()


def test_feature_set_metadata_and_explanations_have_stable_plan_names():
    engine, _ = plan_engine()
    engine.fit(pd.DataFrame({"category": ["a", "b"]}))
    assert engine.get_feature_names() == ["category_frequency"]
    features = list(engine.get_feature_set())
    assert len(features) == 1
    assert features[0].name == "category_frequency"
    assert features[0].source_columns == ["category"]
    assert features[0].dtype == FeatureType.NUMERIC
    assert features[0].origin == FeatureOrigin.LLM_GENERATED
    assert features[0].transformation == "frequency"
    assert features[0].code is None
    assert features[0].metadata["feature_format"] == "plan"
    assert engine.get_feature_explanations() == {"category_frequency": "Training prevalence of category"}
    assert engine.get_feature_metadata()["category_frequency"]["operator"] == "frequency"
    assert engine.get_feature_code() == {}


def test_predeclared_rules_bypass_provider_and_record_categorical_feature_type():
    spec = FittedRuleSpec(
        name="collapsed",
        operator="rare_category",
        columns=["category"],
        parameters={"min_frequency": 0.4},
        explanation="Collapse rare strings",
    )
    engine = SemanticEngine(feature_format="plan", rules=[spec])
    with patch.object(engine, "_ensure_client", side_effect=AssertionError("No provider for supplied rules")):
        output = engine.fit_transform(pd.DataFrame({"category": ["a", "a", "b"]}))
    assert output["collapsed"].tolist() == ["a", "a", "__other__"]
    feature = list(engine.get_feature_set())[0]
    assert feature.dtype == FeatureType.CATEGORICAL
    assert feature.origin == FeatureOrigin.CUSTOM
    assert engine._client is None


def test_plan_skips_all_text_code_paths_and_preserves_original_text_columns():
    X = pd.DataFrame(
        {"category": ["a"] * 12, "text": [f"A lengthy text document with unique identifier {i}" for i in range(12)]}
    )
    engine, client = plan_engine(enable_text_features=True, keep_text_columns=False)
    with (
        patch.object(engine, "_generate_text_features", side_effect=AssertionError("No text code in plans")),
        patch("featcopilot.llm.semantic_engine.execute_feature_code", side_effect=AssertionError("No exec in plans")),
        patch("featcopilot.llm.semantic_engine.logger.info") as logged,
    ):
        output = engine.fit_transform(X)
    assert list(output.columns) == ["category", "text", "category_frequency"]
    pd.testing.assert_frame_equal(output[X.columns], X)
    assert any("plan mode" in call.args[0] for call in logged.call_args_list)
    client.suggest_features.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "",
        '{"features": [{"name": "f", "code": "result = 1"}]}',
        '{"rules": [{"name": "f", "operator": "execute", "columns": ["category"]}]}',
        '{"rules": [{"name": "f", "operator": "frequency", "columns": ["category"], "code": "result = 1"}]}',
        '{"rules": [{"name": "f", "operator": "frequency", "columns": ["category"], "parameters": {"unsafe": 1}}]}',
        '{"rules": "not a list"}',
        '```json\n{"rules": []}\n```',
        '{"rules": [null]}',
    ],
)
def test_malformed_plans_raise_without_mock_feature_fallback(response):
    engine, client = plan_engine()
    client.send_prompt.return_value = response
    with pytest.raises(ValueError, match="Invalid fitted feature plan"):
        engine.fit(pd.DataFrame({"category": ["a"]}))
    assert not engine.is_fitted
    assert engine.get_feature_names() == []
    assert len(engine.get_feature_set()) == 0
    client.suggest_features.assert_not_called()
    with pytest.raises(RuntimeError, match="fitted"):
        engine.transform(pd.DataFrame({"category": ["a"]}))


def test_provider_exceptions_propagate_unchanged():
    engine, client = plan_engine()
    error = TimeoutError("provider deadline")
    client.send_prompt.side_effect = error
    with pytest.raises(TimeoutError) as raised:
        engine.fit(pd.DataFrame({"category": ["a"]}))
    assert raised.value is error
    assert not engine.is_fitted


def test_plan_and_supplied_rule_budgets_are_enforced_without_truncation():
    plan = frequency_plan()
    plan["rules"].append({**plan["rules"][0], "name": "another"})
    engine, _ = plan_engine(plan, max_suggestions=1)
    with pytest.raises(ValueError, match="maximum is 1"):
        engine.fit(pd.DataFrame({"category": ["a"]}))
    engine = SemanticEngine(feature_format="plan", rules=plan["rules"], max_suggestions=1)
    with pytest.raises(ValueError, match="exceeds"):
        engine.fit(pd.DataFrame({"category": ["a"]}))


def test_empty_valid_plan_is_identity_not_a_fabricated_feature():
    engine, client = plan_engine({"rules": []})
    X = pd.DataFrame({"category": ["a", "b"]})
    pd.testing.assert_frame_equal(engine.fit_transform(X), X)
    assert engine.get_feature_names() == []
    client.send_prompt.assert_called_once()


def test_refit_resets_plan_state_feature_names_and_metadata_even_after_failed_refit():
    engine, client = plan_engine()
    engine.fit(pd.DataFrame({"category": ["a", "a"]}))
    updated = frequency_plan()
    updated["rules"][0]["name"] = "new_frequency"
    client.send_prompt.return_value = json.dumps(updated)
    engine.fit(pd.DataFrame({"category": ["b"]}))
    assert engine.get_feature_names() == ["new_frequency"]
    assert list(engine.get_feature_explanations()) == ["new_frequency"]
    assert list(engine.get_feature_metadata()) == ["new_frequency"]
    assert engine.transform(pd.DataFrame({"category": ["a"]}))["new_frequency"].iloc[0] == 0
    client.send_prompt.return_value = "invalid"
    with pytest.raises(ValueError):
        engine.fit(pd.DataFrame({"category": ["b"]}))
    assert not engine.is_fitted
    assert engine.get_feature_names() == []
    assert len(engine.get_feature_set()) == 0
    with pytest.raises(RuntimeError, match="fitted"):
        engine.get_fitted_rules()


@pytest.mark.parametrize(
    "backend,module,client_name",
    [
        ("copilot", "featcopilot.llm.copilot_client", "SyncCopilotFeatureClient"),
        ("openai", "featcopilot.llm.openai_client", "SyncOpenAIFeatureClient"),
        ("litellm", "featcopilot.llm.litellm_client", "SyncLiteLLMFeatureClient"),
    ],
)
def test_shared_sync_send_prompt_interface_for_each_backend(backend, module, client_name):
    with patch(f"{module}.{client_name}") as factory:
        client = factory.return_value
        client.send_prompt.return_value = json.dumps(frequency_plan())
        engine = SemanticEngine(feature_format="plan", backend=backend, timeout=9)
        engine.fit(pd.DataFrame({"category": ["a", "b"]}))
        assert factory.call_args.kwargs["model"] == DEFAULT_MODEL
        assert factory.call_args.kwargs["timeout"] == 9
        if backend == "copilot":
            assert factory.call_args.kwargs["isolated_plan"] is True
        client.start.assert_called_once()
        client.send_prompt.assert_called_once()
        client.suggest_features.assert_not_called()
        engine.transform(pd.DataFrame({"category": ["a"]}))
        client.send_prompt.assert_called_once()


@pytest.fixture
def isolated_copilot_sdk():
    pytest.importorskip("copilot")
    from copilot.generated.session_events import AssistantMessageData
    from copilot.session import CopilotSession

    with patch("copilot.CopilotClient", autospec=True) as factory:
        client = factory.return_value
        client.get_auth_status.return_value = SimpleNamespace(isAuthenticated=True)
        session = MagicMock(spec=CopilotSession)
        session.send_and_wait.return_value = SimpleNamespace(
            data=AssistantMessageData(content=json.dumps(frequency_plan()), message_id="mock")
        )
        client.create_session.return_value = session
        yield factory, client, session


def test_production_plan_uses_fresh_isolated_transport_and_schema_only_context(isolated_copilot_sdk):
    factory, client, session = isolated_copilot_sdk
    workspaces = []

    async def start():
        call = factory.call_args
        if call.args:
            config = call.args[0]
            assert call.kwargs == {"auto_start": False}
            assert config.use_logged_in_user is True
            assert config.use_stdio is True
            assert config.github_token is None
            assert config.log_level == "error"
            workspace, cli_args = config.cwd, config.cli_args
        else:
            options = call.kwargs
            assert options["use_logged_in_user"] is True
            assert options["log_level"] == "error"
            assert "github_token" not in options
            workspace, cli_args = options["working_directory"], options["connection"].args
        assert set(cli_args) == {
            "--no-custom-instructions",
            "--disable-builtin-mcps",
            "--no-ask-user",
            "--no-auto-update",
        }
        workspace = Path(workspace)
        assert workspace != Path.cwd()
        assert workspace.is_dir()
        assert list(workspace.iterdir()) == []
        workspaces.append(workspace)

    client.start.side_effect = start
    engine = SemanticEngine(feature_format="plan", timeout=17, model="chosen-model")
    try:
        engine._ensure_client()
        factory.assert_not_called()
        train = pd.DataFrame({"category": ["PRIVATE_TRAIN", "PRIVATE_TRAIN", "OTHER"]})
        test = pd.DataFrame({"category": ["PRIVATE_TEST"]})
        engine.fit(train, column_descriptions={"category": "Customer segment"})
        assert engine.transform(test)["category_frequency"].tolist() == [0.0]
        session.send_and_wait.assert_awaited_once()
        engine.fit(train, column_descriptions={"category": "Customer segment"})
        assert factory.call_count == client.start.await_count == client.get_auth_status.await_count == 2
        assert client.stop.await_count == session.disconnect.await_count == 2
        assert len(set(workspaces)) == 2
        assert all(not workspace.exists() for workspace in workspaces)
        for call in client.create_session.call_args_list:
            options = call.kwargs
            assert options["available_tools"] == options["tools"] == []
            assert options["skill_directories"] == options["custom_agents"] == []
            assert options["mcp_servers"] == {}
            assert options["enable_config_discovery"] is False
            assert options["infinite_sessions"] == {"enabled": False}
            assert options["streaming"] is False
            assert options["model"] == "chosen-model"
            assert options["on_permission_request"](None, {}).kind == "reject"
            assert options["working_directory"] in [str(workspace) for workspace in workspaces]
            assert "max_tokens" not in options and "temperature" not in options
        for call in session.send_and_wait.call_args_list:
            assert call.kwargs == {"timeout": 17}
            prompt = call.args[0]
            assert "Customer segment" in prompt
            assert "PRIVATE_TRAIN" not in prompt and "PRIVATE_TEST" not in prompt and "OTHER" not in prompt
        assert engine._client._loop is None
    finally:
        if engine._client is not None:
            engine._client.stop()


@pytest.mark.parametrize(
    "stage",
    [
        "start",
        "auth",
        "missing_login",
        "create_session",
        "send",
        "cancel",
        "parse",
        "empty",
        "tools",
        "disconnect",
        "stop",
    ],
)
def test_production_plan_cleanup_and_no_retry_on_errors(stage, isolated_copilot_sdk):
    factory, client, session = isolated_copilot_sdk
    if stage in ("start", "create_session", "stop"):
        getattr(client, stage).side_effect = RuntimeError(f"mock {stage} failure")
    elif stage == "auth":
        client.get_auth_status.side_effect = RuntimeError("mock auth failure")
    elif stage == "missing_login":
        client.get_auth_status.return_value = SimpleNamespace(isAuthenticated=False)
    elif stage == "send":
        session.send_and_wait.side_effect = TimeoutError("mock send failure")
    elif stage == "cancel":
        session.send_and_wait.side_effect = asyncio.CancelledError()
    elif stage == "parse":
        session.send_and_wait.return_value.data.content = "invalid JSON"
    elif stage == "empty":
        session.send_and_wait.return_value = None
    elif stage == "tools":
        session.send_and_wait.return_value.data.tool_requests = [object()]
    else:
        session.disconnect.side_effect = RuntimeError("mock disconnect failure")
    engine = SemanticEngine(feature_format="plan")
    try:
        with pytest.raises((RuntimeError, ValueError, TimeoutError, asyncio.CancelledError)):
            engine.fit(pd.DataFrame({"category": ["a"]}))
        assert not engine.is_fitted
        factory.assert_called_once()
        client.stop.assert_awaited_once()
        workspace = (
            factory.call_args.args[0].cwd if factory.call_args.args else factory.call_args.kwargs["working_directory"]
        )
        assert not Path(workspace).exists()
        sent = stage not in ("start", "auth", "missing_login", "create_session")
        assert session.send_and_wait.await_count == int(sent)
        assert session.disconnect.await_count == int(sent)
        assert engine._client._loop is None
    finally:
        if engine._client is not None:
            engine._client.stop()


def test_production_plan_requires_optional_sdk_but_predeclared_rules_do_not(monkeypatch):
    monkeypatch.setitem(sys.modules, "copilot", None)
    X = pd.DataFrame({"category": ["a"]})
    engine = SemanticEngine(feature_format="plan")
    with pytest.raises(ImportError):
        engine.fit(X)
    assert engine._client is None
    declared = SemanticEngine(feature_format="plan", rules=frequency_plan()["rules"])
    assert declared.fit_transform(X)["category_frequency"].tolist() == [1.0]
    assert declared._client is None


def test_missing_plan_sdk_logs_install_guidance_without_starting_client(isolated_copilot_sdk):
    factory, sdk_client, _ = isolated_copilot_sdk
    original = ModuleNotFoundError("PRIVATE_IMPORT_DETAIL")
    original.__cause__ = ImportError("original import cause")
    client = SyncCopilotFeatureClient(isolated_plan=True)
    try:
        with (
            patch("featcopilot.llm._plan_transport.import_module", side_effect=original),
            patch("featcopilot.llm._plan_transport.TemporaryDirectory") as workspace,
            patch("featcopilot.llm.copilot_client.logger.error") as error_log,
            pytest.raises(ModuleNotFoundError) as caught,
        ):
            client.start()
        assert caught.value is original
        assert str(caught.value.__cause__) == "original import cause"
        error_log.assert_called_once_with("Copilot SDK is unavailable. Install featcopilot[llm].")
        assert "PRIVATE" not in str(error_log.call_args_list)
        workspace.assert_not_called()
        factory.assert_not_called()
        sdk_client.start.assert_not_awaited()
        assert not client._async_client._is_started
        assert not client._async_client._copilot_available
    finally:
        client.stop()


@pytest.mark.parametrize("unsupported", ["client", "transport", "session"])
def test_production_plan_rejects_unsupported_sdk_without_permissive_fallback(
    unsupported, isolated_copilot_sdk, monkeypatch
):
    factory, _, _ = isolated_copilot_sdk
    if unsupported == "client":
        monkeypatch.setattr(factory, "__signature__", inspect.Signature())
    elif unsupported == "session":
        monkeypatch.setattr(factory.create_session, "__signature__", inspect.Signature())
    elif "connection" in inspect.signature(factory).parameters:
        from copilot import RuntimeConnection

        monkeypatch.setattr(RuntimeConnection, "for_stdio", lambda: None)
    else:
        import copilot.client as sdk_client

        monkeypatch.setattr(sdk_client, "SubprocessConfig", lambda: None)
    engine = SemanticEngine(feature_format="plan")
    with pytest.raises(RuntimeError, match=f"{unsupported}-isolation"):
        engine.fit(pd.DataFrame({"category": ["a"]}))
    assert engine._client is None
    factory.assert_not_called()


async def test_production_plan_in_active_event_loop_finishes_lifecycle_in_worker(isolated_copilot_sdk):
    _, client, session = isolated_copilot_sdk
    caller_loop = asyncio.get_running_loop()
    loops = []

    async def record_loop():
        loops.append(asyncio.get_running_loop())
        assert loops[-1] is not caller_loop

    client.start.side_effect = record_loop
    session.disconnect.side_effect = record_loop
    client.stop.side_effect = record_loop
    engine = SemanticEngine(feature_format="plan")
    try:
        engine.fit(pd.DataFrame({"category": ["a"]}))
        assert engine.is_fitted
        assert len(loops) == 3 and len(set(loops)) == 1
        assert loops[0].is_closed()
        assert not caller_loop.is_closed()
    finally:
        if engine._client is not None:
            engine._client.stop()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("stage", ["start", "get_auth_status", "create_session", "send_and_wait"])
def test_plan_deadline_covers_all_provider_phases(stage, nested, isolated_copilot_sdk):
    _, client, session = isolated_copilot_sdk
    cancelled = []

    async def hang(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(stage)

    target = session if stage == "send_and_wait" else client
    getattr(target, stage).side_effect = hang
    engine = SemanticEngine(feature_format="plan", timeout=0.03)

    async def nested_fit():
        engine.fit(pd.DataFrame({"category": ["a"]}))

    before = time.monotonic()
    try:
        with pytest.raises(TimeoutError) as caught:
            if nested:
                asyncio.run(nested_fit())
            else:
                engine.fit(pd.DataFrame({"category": ["a"]}))
        assert "0.03 seconds" in str(caught.value)
        assert "Copilot plan request" in str(caught.value)
        assert isinstance(caught.value.__cause__, TimeoutError)
        assert caught.value.__cause__.args == ()
        assert time.monotonic() - before < 2
        assert cancelled == [stage]
        client.stop.assert_awaited_once()
        assert session.disconnect.await_count == int(stage == "send_and_wait")
        assert not engine.is_fitted
    finally:
        if engine._client is not None:
            engine._client.stop()


async def test_isolated_sync_bridge_cancels_a_hung_coroutine():
    client = SyncCopilotFeatureClient(isolated_plan=True, timeout=0.03)
    cancelled = []

    async def hang():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    before = time.monotonic()
    with pytest.raises(TimeoutError) as caught:
        client._run_async(hang())
    assert "request timeout 0.03 seconds" in str(caught.value)
    assert "bridge" in str(caught.value)
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert time.monotonic() - before < 2
    assert cancelled == [True]


def test_plan_deadline_is_shared_across_startup_phases(isolated_copilot_sdk):
    _, client, session = isolated_copilot_sdk

    async def start():
        await asyncio.sleep(0.13)

    async def auth():
        await asyncio.sleep(0.13)
        return SimpleNamespace(isAuthenticated=True)

    client.start.side_effect = start
    client.get_auth_status.side_effect = auth
    engine = SemanticEngine(feature_format="plan", timeout=0.2)
    try:
        with pytest.raises(TimeoutError):
            engine.fit(pd.DataFrame({"category": ["a"]}))
        client.get_auth_status.assert_awaited_once()
        client.create_session.assert_not_awaited()
        client.stop.assert_awaited_once()
        session.send_and_wait.assert_not_awaited()
    finally:
        if engine._client is not None:
            engine._client.stop()


@pytest.mark.parametrize("phase", ["disconnect", "stop"])
def test_plan_shutdown_has_a_separate_bounded_grace(phase, isolated_copilot_sdk):
    _, client, session = isolated_copilot_sdk
    cancelled = []

    async def hang():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(phase)

    target = session if phase == "disconnect" else client
    getattr(target, phase).side_effect = hang
    engine = SemanticEngine(feature_format="plan", timeout=0.03)
    before = time.monotonic()
    try:
        with pytest.raises(TimeoutError) as caught, patch("featcopilot.llm._plan_transport.logger.warning") as warning:
            engine.fit(pd.DataFrame({"category": ["a"]}))
        assert f"{phase} cleanup" in str(caught.value)
        assert "0.03 seconds" in str(caught.value)
        assert isinstance(caught.value.__cause__, TimeoutError)
        assert time.monotonic() - before < 2
        assert cancelled == [phase]
        warning.assert_called_once()
        assert warning.call_args.args[1:] == (phase, "TimeoutError")
        session.disconnect.assert_awaited_once()
        client.stop.assert_awaited_once()
    finally:
        if engine._client is not None:
            engine._client.stop()


@pytest.mark.parametrize("request_timeout,cleanup_timeout", [(8, 5.0), (60.0, 5.0), (0.03, 0.03)])
@pytest.mark.parametrize("phase", ["disconnect", "stop"])
async def test_plan_cleanup_caps_actual_wait_for_argument(
    request_timeout, cleanup_timeout, phase, isolated_copilot_sdk
):
    from featcopilot.llm._plan_transport import PLAN_CLEANUP_TIMEOUT, IsolatedCopilotTransport

    assert PLAN_CLEANUP_TIMEOUT == 5.0
    _, client, session = isolated_copilot_sdk
    transport = IsolatedCopilotTransport()
    real_wait_for = asyncio.wait_for
    boundaries = []
    cancelled = []

    async def instrumented_wait_for(awaitable, timeout):
        boundaries.append(timeout)
        # Record the real policy argument, but exercise cooperative cleanup
        # cancellation with a short wait instead of spending five seconds.
        return await real_wait_for(awaitable, timeout=timeout if len(boundaries) == 1 else 0.01)

    async def hang():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(phase)

    target = session if phase == "disconnect" else client
    getattr(target, phase).side_effect = hang
    send = AsyncMock(return_value="successful plan")
    with patch("featcopilot.llm._plan_transport.asyncio.wait_for", side_effect=instrumented_wait_for):
        with pytest.raises(TimeoutError) as caught:
            await transport.request(DEFAULT_MODEL, request_timeout, send)
    assert f"{phase} cleanup" in str(caught.value)
    assert f"after {cleanup_timeout:g} seconds" in str(caught.value)
    assert f"request timeout {request_timeout:g} seconds" in str(caught.value)
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert boundaries == [request_timeout, cleanup_timeout, cleanup_timeout]
    assert cancelled == [phase]
    send.assert_awaited_once_with(session)
    session.disconnect.assert_awaited_once()
    client.stop.assert_awaited_once()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0, -1, True, False])
async def test_plan_transport_rejects_invalid_timeout_before_workspace_or_client(timeout, isolated_copilot_sdk):
    from featcopilot.llm._plan_transport import IsolatedCopilotTransport

    factory, client, _ = isolated_copilot_sdk
    transport = IsolatedCopilotTransport()
    send = AsyncMock()
    with patch("featcopilot.llm._plan_transport.TemporaryDirectory") as workspace:
        with pytest.raises(ValueError, match="timeout"):
            await transport.request(DEFAULT_MODEL, timeout, send)
        workspace.assert_not_called()
    factory.assert_not_called()
    client.start.assert_not_awaited()
    send.assert_not_awaited()


@pytest.mark.parametrize("failure_stage", ["start", "send"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError, TimeoutError])
def test_plan_shutdown_errors_are_observed_without_masking_original(failure_stage, error_type, isolated_copilot_sdk):
    _, client, session = isolated_copilot_sdk
    original = error_type("PRIVATE_REQUEST_FAILURE")
    if failure_stage == "start":
        client.start.side_effect = original
    else:
        session.send_and_wait.side_effect = original
    session.disconnect.side_effect = RuntimeError("PRIVATE_DISCONNECT_ERROR")
    client.stop.side_effect = RuntimeError("PRIVATE_STOP_ERROR")
    engine = SemanticEngine(feature_format="plan", timeout=0.03)
    try:
        with pytest.raises(error_type) as caught, patch("featcopilot.llm._plan_transport.logger.warning") as warning:
            engine.fit(pd.DataFrame({"category": ["a"]}))
        assert caught.value is original
        assert warning.call_count == (1 if failure_stage == "start" else 2)
        assert "PRIVATE" not in str(warning.call_args_list)
        client.stop.assert_awaited_once()
    finally:
        if engine._client is not None:
            engine._client.stop()


@pytest.mark.parametrize("args", [(), ("",)])
def test_empty_sdk_plan_timeout_gets_context_and_keeps_original_cause(args, isolated_copilot_sdk):
    _, client, session = isolated_copilot_sdk
    original = TimeoutError(*args)
    session.send_and_wait.side_effect = original
    engine = SemanticEngine(feature_format="plan", timeout=19)
    try:
        with pytest.raises(TimeoutError) as caught:
            engine.fit(pd.DataFrame({"category": ["a"]}))
        assert caught.value is not original
        assert type(caught.value) is type(original)
        assert caught.value.__cause__ is original
        assert "19 seconds" in str(caught.value)
        assert "send" in str(caught.value)
        session.disconnect.assert_awaited_once()
        client.stop.assert_awaited_once()
    finally:
        if engine._client is not None:
            engine._client.stop()


@pytest.mark.parametrize("outcome", ["timeout", "cancel", "isolation", "success"])
@pytest.mark.parametrize("cleanup_error_type", [PermissionError, OSError])
@pytest.mark.parametrize("include_errno", [False, True])
def test_workspace_cleanup_oserror_is_logged_without_replacing_request(
    outcome, cleanup_error_type, include_errno, isolated_copilot_sdk
):
    from featcopilot.llm._plan_transport import PlanIsolationError

    _, client, session = isolated_copilot_sdk
    original = {
        "timeout": TimeoutError("original deadline"),
        "cancel": asyncio.CancelledError("original cancellation"),
        "isolation": PlanIsolationError("original isolation failure"),
        "success": None,
    }[outcome]
    if outcome == "isolation":
        client.get_auth_status.side_effect = original
    else:
        session.send_and_wait.side_effect = original
    workspace = MagicMock()
    workspace.name = str(Path.cwd() / "featcopilot-plan-synthetic")
    workspace.__enter__.return_value = workspace.name
    cleanup_errno = (errno.EACCES if cleanup_error_type is PermissionError else errno.EIO) if include_errno else None
    workspace.cleanup.side_effect = (
        cleanup_error_type(cleanup_errno, "PRIVATE_BUSY_WORKSPACE", "PRIVATE_ERROR_FILENAME")
        if include_errno
        else cleanup_error_type("PRIVATE_BUSY_WORKSPACE")
    )
    workspace.__exit__.side_effect = lambda *args: workspace.cleanup()
    engine = SemanticEngine(feature_format="plan")
    try:
        with (
            patch("featcopilot.llm._plan_transport.TemporaryDirectory", return_value=workspace),
            patch("featcopilot.llm._plan_transport.logger.warning") as warning,
        ):
            X = pd.DataFrame({"category": ["a"]})
            if original is None:
                assert engine.fit_transform(X)["category_frequency"].tolist() == [1.0]
            else:
                with pytest.raises(type(original)) as caught:
                    engine.fit(X)
                assert caught.value is original
        assert engine.is_fitted is (original is None)
        client.stop.assert_awaited_once()
        workspace.cleanup.assert_called_once_with()
        workspace.__exit__.assert_not_called()
        warning.assert_called_once()
        assert "workspace cleanup" in warning.call_args.args[0]
        assert warning.call_args.args[1] == cleanup_error_type.__name__
        assert warning.call_args.args[2:] == (cleanup_errno, workspace.name)
        rendered = warning.call_args.args[0] % warning.call_args.args[1:]
        assert f"errno={cleanup_errno}" in rendered
        assert workspace.name in rendered
        assert "PRIVATE" not in str(warning.call_args_list)
    finally:
        if engine._client is not None:
            engine._client.stop()


def test_workspace_creation_failure_propagates_before_sdk_start(isolated_copilot_sdk):
    factory, client, _ = isolated_copilot_sdk
    original = PermissionError("workspace creation denied")
    engine = SemanticEngine(feature_format="plan")
    try:
        with (
            patch("featcopilot.llm._plan_transport.TemporaryDirectory", side_effect=original) as directory_factory,
            patch("featcopilot.llm._plan_transport.logger.warning") as warning,
            pytest.raises(PermissionError) as caught,
        ):
            engine.fit(pd.DataFrame({"category": ["a"]}))
        assert caught.value is original
        assert not engine.is_fitted
        directory_factory.assert_called_once_with(prefix="featcopilot-plan-")
        factory.assert_not_called()
        client.start.assert_not_awaited()
        warning.assert_not_called()
    finally:
        if engine._client is not None:
            engine._client.stop()


def test_benchmark_adapter_uses_the_shared_startup_deadline(isolated_copilot_sdk):
    from benchmarks.llm_evaluation import CopilotPlanProvider, PlanRequest

    _, client, session = isolated_copilot_sdk

    async def hang():
        await asyncio.Event().wait()

    client.start.side_effect = hang
    with pytest.raises(TimeoutError):
        CopilotPlanProvider(timeout=0.03).send_plan(PlanRequest("{}"), max_tokens=4096)
    client.stop.assert_awaited_once()
    session.send_and_wait.assert_not_awaited()


def test_pickle_and_json_export_exclude_live_provider_and_preserve_output():
    engine, client = plan_engine()
    client.do_not_pickle = lambda: None
    X = pd.DataFrame({"category": ["a", "a", "b"]})
    engine.fit(X)
    restored = pickle.loads(pickle.dumps(engine))
    assert restored._client is None
    pd.testing.assert_frame_equal(restored.transform(X), engine.transform(X))
    assert restored.get_feature_explanations() == engine.get_feature_explanations()
    data = json.loads(json.dumps(engine.get_fitted_rules(), allow_nan=False))
    rule_engine = FittedRuleEngine.from_dict(data)
    pd.testing.assert_frame_equal(rule_engine.transform(X), engine.transform(X))
    assert "_client" not in data


def test_combined_grouped_and_temporal_supervised_plan_is_rejected():
    rules = [{"name": "encoded", "operator": "target_mean", "columns": ["category"]}]
    engine = SemanticEngine(feature_format="plan", rules=rules)
    X = pd.DataFrame({"category": ["a", "a", "b", "b"]})
    with pytest.raises(ValueError, match="groups and timestamps"):
        engine.fit_transform(X, np.array([0, 1, 0, 1]), groups=np.array([1, 1, 2, 2]), timestamps=np.arange(4))


def test_input_and_rule_schema_errors_are_not_silently_dropped():
    engine, client = plan_engine()
    with pytest.raises(ValueError, match="missing source"):
        engine.fit(pd.DataFrame({"different": ["a"]}))
    assert not engine.is_fitted
    client.send_prompt.return_value = json.dumps(frequency_plan())
    engine.fit(pd.DataFrame({"category": ["a"]}))
    with pytest.raises(ValueError, match="schema"):
        engine.transform(pd.DataFrame({"category": ["a"], "new_column": [1]}))
    engine = SemanticEngine(
        feature_format="plan", rules=[{"name": "category", "operator": "frequency", "columns": ["category"]}]
    )
    with pytest.raises(ValueError, match="collide"):
        engine.fit(pd.DataFrame({"category": ["a"]}))


def test_code_generation_helpers_do_not_silently_mix_into_plan_mode():
    engine, client = plan_engine()
    with pytest.raises(ValueError, match="trusted code"):
        engine.suggest_more_features("ratios")
    with pytest.raises(ValueError, match="trusted code"):
        engine.generate_custom_feature("arbitrary Python")
    client.suggest_features.assert_not_called()
    client.generate_feature_code.assert_not_called()


def test_predeclared_grouped_target_plan_needs_no_provider_and_excludes_own_group():
    rules = [{"name": "encoded", "operator": "target_mean", "columns": ["category"], "parameters": {"smoothing": 0}}]
    X = pd.DataFrame({"category": ["a", "a", "a", None]}, index=[4, 4, 2, 9])
    groups = pd.Series(["first", "first", "second", "second"], index=[99, 98, 97, 96])
    y = np.array([2.0, 4.0, 8.0, 10.0])
    engine = SemanticEngine(feature_format="plan", rules=rules, task="regression", target_cv=2)
    with patch.object(engine, "_ensure_client", side_effect=AssertionError("No provider for predeclared plans")):
        output = engine.fit_transform(X, y, groups=groups)
    np.testing.assert_allclose(output["encoded"], [8, 8, 3, 3])
    assert engine.get_fitted_rules()["cross_fit_report"]["split_kind"] == "grouped"
    assert engine.get_fitted_rules()["cross_fit_report"]["prior_row_positions"] == []
    changed = y.copy()
    changed[:2] += 9000
    perturbed = SemanticEngine(feature_format="plan", rules=rules, task="regression", target_cv=2)
    after = perturbed.fit_transform(X, changed, groups=groups)
    np.testing.assert_allclose(output["encoded"].iloc[:2], after["encoded"].iloc[:2])
    pd.testing.assert_frame_equal(output[X.columns], X)
    assert engine._client is None


@pytest.mark.parametrize("predeclared", [False, True])
def test_temporal_plan_forwards_label_end_times_and_serializes_prior_marker(predeclared):
    rules = [{"name": "encoded", "operator": "target_mean", "columns": ["category"], "parameters": {"smoothing": 0}}]
    X = pd.DataFrame({"category": ["a"] * 8}, index=[9, 9, 4, 4, 0, 0, 2, 2])
    timestamps = np.arange(1, 9)
    label_ends = timestamps.copy()
    label_ends[0] = 7
    y = np.array([1000.0, 2, 3, 4, 5, 6, 7, 8])
    if predeclared:
        engine = SemanticEngine(feature_format="plan", rules=rules, task="regression", target_cv=2)
        client = None
    else:
        engine, client = plan_engine({"rules": rules}, task="regression", target_cv=2)
    output = engine.fit_transform(X, y, timestamps=timestamps, label_end_times=label_ends)
    np.testing.assert_allclose(output["encoded"], [0, 0, 0, 0, 3, 3, 4, 4])
    report = engine.get_fitted_rules()["cross_fit_report"]
    assert report["prior_policy"] == "fixed_zero"
    assert report["prior_row_positions"] == [0, 1, 2, 3]
    assert report["label_end_times_used"] is True
    with patch.object(engine, "_ensure_client", side_effect=AssertionError("No transform provider calls")):
        future = pd.DataFrame({"category": ["a", "unknown", None]})
        expected = engine.transform(future)
        np.testing.assert_allclose(expected["encoded"], y.mean())
        serialized = json.loads(json.dumps(engine.get_fitted_rules(), allow_nan=False))
        restored_rules = FittedRuleEngine.from_dict(serialized)
        pd.testing.assert_frame_equal(restored_rules.transform(future), expected)
        assert restored_rules.get_cross_fit_report() == report
    restored = pickle.loads(pickle.dumps(engine))
    pd.testing.assert_frame_equal(restored.transform(future), expected)
    if client is not None:
        client.send_prompt.assert_called_once()
    assert engine.get_feature_names() == ["encoded"]
    assert engine.get_feature_explanations() == {"encoded": ""}


def test_plan_label_availability_errors_propagate_and_clear_fitted_state():
    rules = [{"name": "encoded", "operator": "target_mean", "columns": ["category"]}]
    engine = SemanticEngine(feature_format="plan", rules=rules, task="regression", target_cv=2)
    X = pd.DataFrame({"category": ["a"] * 8})
    engine.fit(X, np.arange(8))
    with pytest.raises(ValueError, match="timestamps"):
        engine.fit(X, np.arange(8), label_end_times=np.arange(8))
    assert not engine.is_fitted
    assert engine.get_feature_names() == []
    with pytest.raises(RuntimeError, match="fitted"):
        engine.get_fitted_rules()


@pytest.mark.parametrize("predeclared", [False, True])
def test_rowwise_semantic_plans_execute_typed_rules_without_code(predeclared):
    rules = [
        {"name": "revenue", "operator": "multiply", "columns": ["price", "units"], "explanation": "Unit revenue"},
        {"name": "unit_ratio", "operator": "divide", "columns": ["price", "units"], "explanation": "Price per unit"},
        {"name": "log_price", "operator": "log1p", "columns": ["price"], "explanation": "Absolute log price"},
    ]
    if predeclared:
        engine = SemanticEngine(feature_format="plan", rules=rules)
        client = None
    else:
        engine, client = plan_engine({"rules": rules})
    X = pd.DataFrame({"price": [-4.0, 6, np.nan], "units": [2.0, 0, 3]}, index=[7, 7, 1])
    with patch("featcopilot.llm.semantic_engine.execute_feature_code", side_effect=AssertionError("No generated code")):
        output = engine.fit_transform(X)
        pd.testing.assert_frame_equal(engine.transform(X), output)
    np.testing.assert_allclose(output["revenue"], [-8, 0, np.nan], equal_nan=True)
    np.testing.assert_allclose(output["unit_ratio"], [-2, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(output["log_price"], [np.log1p(4), np.log1p(6), np.nan], equal_nan=True)
    assert engine.get_feature_names() == ["revenue", "unit_ratio", "log_price"]
    assert engine.get_feature_code() == {}
    assert engine.get_feature_explanations() == {spec["name"]: spec["explanation"] for spec in rules}
    assert [feature.transformation for feature in engine.get_feature_set()] == ["multiply", "divide", "log1p"]
    restored = FittedRuleEngine.from_dict(json.loads(json.dumps(engine.get_fitted_rules(), allow_nan=False)))
    pd.testing.assert_frame_equal(restored.transform(X), output)
    if client is not None:
        client.send_prompt.assert_called_once()
        client.suggest_features.assert_not_called()
        prompt = client.send_prompt.call_args.args[0]
        assert "Zero denominators produce NaN" in prompt
        assert "log(1 + abs(x))" in prompt
    else:
        assert engine._client is None

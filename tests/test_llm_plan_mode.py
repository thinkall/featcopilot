"""Provider-free tests of the opt-in typed SemanticEngine plan path."""

import json
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from featcopilot.core.feature import FeatureOrigin, FeatureType
from featcopilot.core.fitted_rule import FittedRuleSpec
from featcopilot.engines.fitted_rules import FittedRuleEngine
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
        client.start.assert_called_once()
        client.send_prompt.assert_called_once()
        client.suggest_features.assert_not_called()
        engine.transform(pd.DataFrame({"category": ["a"]}))
        client.send_prompt.assert_called_once()


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

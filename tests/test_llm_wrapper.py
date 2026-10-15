"""Main-API contracts for LLM configuration and output preservation."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from featcopilot import AutoFeatureEngineer
from featcopilot.utils.models import DEFAULT_MODEL


def test_llm_default_model_and_all_options_are_forwarded():
    config = {
        "validate_features": False,
        "temperature": 0.1,
        "timeout": 12.0,
        "max_tokens": 256,
        "enable_text_features": False,
        "keep_text_columns": False,
        "text_feature_types": ["readability"],
        "verbose": True,
    }
    with patch("featcopilot.llm.semantic_engine.SemanticEngine") as cls:
        AutoFeatureEngineer(engines=["llm"], llm_config=config)._create_engine("llm")
    assert DEFAULT_MODEL == "gpt-5.6-sol"
    assert cls.call_args.kwargs["model"] == DEFAULT_MODEL
    assert all(cls.call_args.kwargs[key] == value for key, value in config.items())


def test_explicit_llm_model_is_not_replaced():
    with patch("featcopilot.llm.semantic_engine.SemanticEngine") as cls:
        AutoFeatureEngineer(llm_config={"model": "explicit-model"})._create_engine("llm")
    assert cls.call_args.kwargs["model"] == "explicit-model"


@pytest.mark.parametrize("config", [["not-a-mapping"], {"enable_semantic": True}, {"timeot": 20}])
def test_unknown_llm_options_fail_instead_of_being_ignored(config):
    with pytest.raises(ValueError, match="llm_config"):
        AutoFeatureEngineer(llm_config=config)


@pytest.mark.parametrize("engines", [["llm", "tabular"], ["tabular", "llm"]])
def test_explicit_text_drop_is_honored_when_combining_engines(engines):
    X = pd.DataFrame({"description": ["hello", "world"], "price": [1.0, 2.0]})
    llm = MagicMock()
    llm.transform.return_value = X.drop(columns="description").assign(text_score=[0.1, 0.2])
    engineer = AutoFeatureEngineer(engines=engines, llm_config={"keep_text_columns": False})
    tabular = MagicMock()
    tabular.transform.return_value = X.assign(numeric_feature=[2.0, 4.0])
    engineer._engine_instances = {name: llm if name == "llm" else tabular for name in engines}
    engineer._is_fitted = True
    result = engineer.transform(X)
    assert "description" not in result
    assert result.columns.tolist()[0] == "price"
    assert {"text_score", "numeric_feature"} <= set(result.columns)


def test_selection_and_gate_cannot_restore_explicitly_dropped_text():
    X = pd.DataFrame({"description": ["hello", "world"], "price": [1.0, 2.0]})
    y = pd.Series([0, 1])
    transformed = X.drop(columns="description").assign(text_score=[0.1, 0.2])
    llm = MagicMock()
    llm._transform_fit_data.return_value = transformed
    llm.transform.return_value = transformed
    engineer = AutoFeatureEngineer(engines=["llm"], max_features=5, llm_config={"keep_text_columns": False})
    with (
        patch.object(engineer, "_create_engine", return_value=llm),
        patch("featcopilot.transformers.sklearn_compat.FeatureSelector") as selector,
        patch.object(engineer, "_do_no_harm_gate", side_effect=lambda frame, *args: frame) as gate,
    ):
        selector.return_value.fit_transform.side_effect = lambda frame, target: frame
        selector.return_value.get_selected_features.return_value = ["price", "text_score"]
        result = engineer.fit_transform(X, y)
        prediction = engineer.transform(X)
    assert selector.call_args.kwargs["original_features"] == {"price"}
    assert gate.call_args.args[1].columns.tolist() == ["price"]
    assert "description" not in result
    pd.testing.assert_frame_equal(result, prediction)


def test_provided_fitted_plans_work_without_a_provider():
    train = pd.DataFrame({"amount": [1.0, 3.0], "group": ["a", "b"]}, index=[10, 20])
    test = pd.DataFrame({"amount": [5.0, 7.0], "group": ["a", "new"]}, index=[30, 40])
    engineer = AutoFeatureEngineer(
        engines=["llm"],
        task="regression",
        llm_config={
            "feature_format": "plan",
            "rules": [
                {"name": "amount_standardized", "operator": "standardize", "columns": ["amount"]},
                {"name": "group_frequency", "operator": "frequency", "columns": ["group"]},
            ],
        },
    )
    with patch(
        "featcopilot.llm.copilot_client.SyncCopilotFeatureClient.start",
        side_effect=AssertionError("Unexpected provider"),
    ):
        engineer.fit(train)
        result = engineer.transform(test)
    assert result.index.equals(test.index)
    assert result.group_frequency.tolist() == [0.5, 0.0]
    assert isinstance(engineer.get_fitted_rules(), dict)
    batches = pd.concat([engineer.transform(test.iloc[:1]), engineer.transform(test.iloc[1:])])
    pd.testing.assert_frame_equal(result, batches)


def test_fitted_plans_cannot_use_random_legacy_selection_with_groups():
    engineer = AutoFeatureEngineer(
        engines=["llm"],
        max_features=5,
        llm_config={"feature_format": "plan", "rules": []},
    )
    with pytest.raises(ValueError, match="Legacy selection is not group/time-aware"):
        engineer.fit_transform(pd.DataFrame({"value": [1.0, 2.0]}), pd.Series([1.0, 2.0]), groups=["a", "b"])


@pytest.mark.parametrize(
    "metadata",
    [
        {"groups": [0, 0, 1, 1]},
        {"timestamps": [1, 2, 3, 4]},
        {"timestamps": [1, 2, 3, 4], "label_end_times": [1, 2, 3, 4]},
    ],
)
def test_unsupervised_plan_fit_transform_does_not_forward_training_boundaries_to_inference(metadata):
    X = pd.DataFrame({"category": ["a", "a", "b", "b"]}, index=[7, 3, 7, 2])
    engineer = AutoFeatureEngineer(
        engines=["llm"],
        llm_config={
            "feature_format": "plan",
            "rules": [{"name": "frequency", "operator": "frequency", "columns": ["category"]}],
        },
    )
    with patch(
        "featcopilot.llm.copilot_client.SyncCopilotFeatureClient.start",
        side_effect=AssertionError("Supplied rules must not call a provider"),
    ):
        training = engineer.fit_transform(X, **metadata)
    assert training["frequency"].tolist() == [0.5] * 4
    pd.testing.assert_frame_equal(training, engineer.transform(X))
    with pytest.raises(TypeError, match="transform options"):
        engineer.transform(X, **metadata)

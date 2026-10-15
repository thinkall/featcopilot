"""Regression tests for opt-in predictive selection and explicit task routing."""

import json
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge

from featcopilot import AutoFeatureEngineer, FeatureEngineerTransformer, cli
from featcopilot.engines.predictive import PredictiveTabularEngine


def example():
    rng = np.random.default_rng(12)
    X = pd.DataFrame({"a": rng.normal(size=250), "b": rng.normal(size=250)}, index=np.arange(250) + 500)
    y = (X.a * X.b + rng.normal(scale=0.1, size=250)).rename("response")
    return X, y


def test_legacy_configuration_is_still_default():
    engineer = AutoFeatureEngineer()
    assert engineer.task == "auto"
    assert engineer.selection_strategy == "legacy"
    assert engineer.selection_estimator is None
    assert engineer.include_frequency is False
    assert engineer.numeric_clip_quantiles is None
    assert engineer.importance_resamples == 1


@pytest.mark.parametrize(
    "params,message",
    [
        ({"task": "forecast"}, "task"),
        ({"selection_strategy": "unknown"}, "selection_strategy"),
        ({"selection_strategy": "predictive", "engines": ["text"]}, "engines"),
        ({"selection_estimator": Ridge()}, "selection_estimator"),
        ({"selection_strategy": "predictive", "selection_estimator": object()}, "sklearn"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"max_candidates": True}, "integer"),
        ({"random_state": -1}, "random_state"),
    ],
)
def test_configuration_rejects_unsupported_predictive_options(params, message):
    with pytest.raises(ValueError, match=message):
        AutoFeatureEngineer(**params)


def test_predictive_clone_and_nested_parameter_updates_are_atomic():
    estimator = Ridge(alpha=1.0)
    engineer = AutoFeatureEngineer(selection_strategy="predictive", selection_estimator=estimator, task="regression")
    cloned = clone(engineer)
    assert cloned.selection_estimator is not estimator
    assert cloned.get_params()["selection_estimator__alpha"] == 1.0
    engineer.set_params(selection_estimator__alpha=0.25)
    assert engineer.selection_estimator.alpha == 0.25
    assert estimator.alpha == 1.0
    previous = engineer.selection_estimator
    with pytest.raises(ValueError, match="task"):
        engineer.set_params(selection_estimator__alpha=5.0, task="bad")
    assert engineer.selection_estimator is previous
    assert engineer.selection_estimator.alpha == 0.25


def test_predictive_strategy_bypasses_legacy_selector_and_gate():
    X, y = example()
    engineer = AutoFeatureEngineer(
        selection_strategy="predictive",
        selection_estimator=Ridge(),
        task="regression",
        max_features=8,
        max_candidates=30,
    )
    with (
        patch("featcopilot.transformers.sklearn_compat.FeatureSelector") as selector,
        patch.object(engineer, "_do_no_harm_gate") as gate,
    ):
        result = engineer.fit_transform(X, y)
    selector.assert_not_called()
    gate.assert_not_called()
    assert isinstance(engineer.selection_report_, dict)
    assert result.index.equals(X.index)
    assert 2 <= result.shape[1] <= 8
    pd.testing.assert_frame_equal(result[X.columns], X)
    pd.testing.assert_frame_equal(engineer.transform(X), result)
    json.dumps(engineer.selection_report_, allow_nan=False)


def test_predictive_requires_target_and_cannot_silently_disable_selection():
    X, y = example()
    engineer = AutoFeatureEngineer(selection_strategy="predictive", task="regression")
    with pytest.raises(ValueError, match="target"):
        engineer.fit(X)
    with pytest.raises(ValueError, match="apply_selection"):
        engineer.fit_transform(X, y, apply_selection=False)


def test_predictive_wrapper_retains_fitted_order_for_downstream_predictions():
    X, y = example()
    engineer = AutoFeatureEngineer(
        selection_strategy="predictive",
        selection_estimator=Ridge(),
        task="regression",
        max_features=2,
        max_candidates=1,
        stability_repeats=1,
    )
    training = engineer.fit_transform(X, y)
    model = Ridge().fit(training, y)
    reordered = X[["b", "a"]]
    actual = engineer.transform(reordered)
    pd.testing.assert_frame_equal(actual, engineer._engine_instances["tabular"].transform(reordered))
    pd.testing.assert_frame_equal(actual, training)
    np.testing.assert_array_equal(model.predict(actual), model.predict(training))


def test_predictive_wrapper_preserves_original_nonfinite_values_during_fit_and_inference():
    X, y = example()
    X.loc[X.index[:3], "a"] = [np.inf, -np.inf, np.nan]
    engineer = AutoFeatureEngineer(
        selection_strategy="predictive",
        selection_estimator=Ridge(),
        task="regression",
        max_features=2,
        max_candidates=1,
        stability_repeats=1,
    )
    training = engineer.fit_transform(X, y)
    pd.testing.assert_frame_equal(training[X.columns], X)
    future = X.iloc[:4][["b", "a"]]
    pd.testing.assert_frame_equal(engineer.transform(future)[X.columns], X.iloc[:4])


@pytest.mark.parametrize("entry", ["engine", "wrapper"])
@pytest.mark.parametrize("labels", [["no", "yes"], [-7, 11]])
@pytest.mark.parametrize("weight_kind", ["full", "partial", "balanced", "none"])
def test_predictive_class_weights_follow_original_labels_without_mutating_caller(entry, labels, weight_kind):
    X, _ = example()
    codes = (X.a > 0).astype(int).to_numpy()
    y = np.asarray(labels)[codes]
    weights = {
        "full": {labels[0]: 0.5, labels[1]: 3.0},
        "partial": {labels[1]: 3.0},
        "balanced": "balanced",
        "none": None,
    }[weight_kind]
    expected = {"full": {0: 0.5, 1: 3.0}, "partial": {1: 3.0}, "balanced": "balanced", "none": None}[weight_kind]
    estimator = LogisticRegression(class_weight=weights, random_state=42, max_iter=300)
    before = pickle.dumps(estimator)

    def make(learner):
        options = {"task": "classification", "max_features": 4, "max_candidates": 5, "stability_repeats": 1}
        if entry == "engine":
            return PredictiveTabularEngine(estimator=learner, **options)
        return AutoFeatureEngineer(
            selection_strategy="predictive", selection_estimator=learner, gate_n_jobs=1, **options
        )

    actual = make(estimator)
    reference = make(LogisticRegression(class_weight=expected, random_state=42, max_iter=300))
    pd.testing.assert_frame_equal(actual.fit_transform(X, y), reference.fit_transform(X, codes))
    assert actual.selection_report_ == reference.selection_report_
    engine = actual if entry == "engine" else actual._engine_instances["tabular"]
    assert engine._learner("classification").get_params()["class_weight"] == expected
    assert pickle.dumps(estimator) == before


def test_predictive_unknown_class_weight_fails_before_fitting_and_refit_resets_vocabulary():
    X, _ = example()
    y = np.where(X.a > 0, "yes", "no")
    estimator = LogisticRegression(class_weight={"no": 1.0, "yes": 2.0})
    engine = PredictiveTabularEngine(
        estimator=estimator, task="classification", max_features=2, max_candidates=1, stability_repeats=1
    ).fit(X, y)
    with patch.object(LogisticRegression, "fit", side_effect=AssertionError("Invalid class weights reached fit")):
        with pytest.raises(ValueError, match="absent from the original training labels"):
            engine.fit(X, np.where(X.a > 0, "new", "other"))
    assert engine._is_fitted is False
    engine.fit(X, y)
    assert engine._learner("classification").class_weight == {0: 1.0, 1: 2.0}


def test_explicit_regression_reaches_legacy_gate_models():
    X, _ = example()
    y = np.arange(len(X)) % 7
    engineer = AutoFeatureEngineer(task="regression", max_features=8)
    derived = X.assign(candidate=X.a * X.b)
    model = MagicMock()
    model.score.return_value = 0.5
    with (
        patch("sklearn.ensemble.RandomForestClassifier", return_value=model) as classifier,
        patch("sklearn.ensemble.RandomForestRegressor", return_value=model) as regressor,
    ):
        engineer._do_no_harm_gate(derived, X, y, set(X.columns))
    classifier.assert_not_called()
    assert regressor.call_count == 10


def test_single_engine_predictive_wrapper_preserves_configuration():
    X, y = example()
    wrapper = clone(
        FeatureEngineerTransformer(
            engine="predictive",
            task="regression",
            estimator=Ridge(),
            max_candidates=20,
            max_features=6,
        )
    )
    result = wrapper.fit_transform(X, y)
    assert result.index.equals(X.index)
    assert wrapper.engine_kwargs["task"] == "regression"
    assert result.shape[1] <= 6


def test_single_engine_nested_estimator_parameters_can_be_tuned():
    original = Ridge(alpha=1.0)
    wrapper = FeatureEngineerTransformer(engine="predictive", task="regression", estimator=original)
    assert wrapper.get_params()["estimator__alpha"] == 1.0
    wrapper.set_params(estimator__alpha=0.25)
    assert wrapper.get_params()["estimator__alpha"] == 0.25
    assert original.alpha == 1.0
    with pytest.raises(ValueError):
        wrapper.set_params(estimator__unknown=1)
    assert wrapper.get_params()["estimator__alpha"] == 0.25


def test_fitted_rule_engine_wrapper_retains_training_statistics():
    transformer = clone(
        FeatureEngineerTransformer(
            engine="fitted_rules",
            rules=[{"name": "scaled", "operator": "standardize", "columns": ["value"]}],
        )
    )
    transformer.fit(pd.DataFrame({"value": [1.0, 3.0]}))
    data = pd.DataFrame({"value": [5.0, 7.0]}, index=[10, 20])
    full = transformer.transform(data)
    pieces = pd.concat([transformer.transform(data.iloc[:1]), transformer.transform(data.iloc[1:])])
    pd.testing.assert_frame_equal(full, pieces)
    assert "scaled" in full


def test_cli_predictive_options_and_report(tmp_path, capsys):
    X, y = example()
    source, target = tmp_path / "input.csv", tmp_path / "output.csv"
    X.assign(response=y).to_csv(source, index=False)
    code = cli.main(
        [
            "transform",
            "--input",
            str(source),
            "--output",
            str(target),
            "--target",
            "response",
            "--task",
            "regression",
            "--selection-strategy",
            "predictive",
            "--max-candidates",
            "20",
            "--max-features",
            "6",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    data = json.loads(captured.out)
    assert data["selection_strategy"] == "predictive"
    assert data["selection_applied"] is True
    assert isinstance(data["selection_report"], dict)
    assert len(pd.read_csv(target)) == len(X)


def test_cli_rejects_predictive_without_target(tmp_path, capsys):
    source = tmp_path / "input.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(source, index=False)
    code = cli.main(
        [
            "transform",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "out.csv"),
            "--selection-strategy",
            "predictive",
            "--json",
        ]
    )
    assert code == 2
    assert "--target" in capsys.readouterr().err


def test_label_availability_metadata_reaches_predictive_selection_only():
    X, y = example()
    engineer = AutoFeatureEngineer(
        selection_strategy="predictive",
        selection_estimator=Ridge(),
        task="regression",
        max_candidates=10,
        max_features=6,
        stability_repeats=1,
    )
    times = np.arange(len(X), dtype=float)
    result = engineer.fit_transform(X, y, timestamps=times, label_end_times=times + 0.5)
    assert engineer.selection_report_["label_end_times_supplied"] is True
    pd.testing.assert_frame_equal(result, engineer.transform(X))


def test_selection_only_ablation_keeps_selector_without_running_gate():
    X, y = example()
    engineer = AutoFeatureEngineer(task="regression", max_features=8)
    with patch.object(engineer, "_do_no_harm_gate", side_effect=AssertionError("Gate must be disabled")):
        result = engineer.fit_transform(X, y, apply_gate=False)
    assert engineer._selector is not None
    pd.testing.assert_frame_equal(result, engineer.transform(X))


def test_legacy_gate_flag_cannot_silently_override_predictive_validation():
    X, y = example()
    with pytest.raises(ValueError, match="legacy-only"):
        AutoFeatureEngineer(selection_strategy="predictive").fit_transform(X, y, apply_gate=False)


def test_extended_candidate_set_is_cloneable_and_reaches_engine():
    X, y = example()
    engineer = clone(
        AutoFeatureEngineer(
            selection_strategy="predictive",
            selection_estimator=Ridge(),
            task="regression",
            candidate_set="extended",
            selection_rule="paired_gain",
            max_candidates=30,
            max_features=8,
            stability_repeats=1,
        )
    )
    transformed = engineer.fit_transform(X, y)
    assert engineer.get_params()["candidate_set"] == "extended"
    assert engineer.selection_report_["candidate_set"] == "extended"
    pd.testing.assert_frame_equal(transformed, engineer.transform(X))


def test_importance_screening_requires_an_explicit_compatible_learner():
    with pytest.raises(ValueError, match="explicit compatible"):
        AutoFeatureEngineer(selection_strategy="predictive", screening="importance")
    X, y = example()
    engineer = clone(
        AutoFeatureEngineer(
            selection_strategy="predictive",
            selection_estimator=Ridge(),
            task="regression",
            screening="importance",
            candidate_set="extended",
            max_candidates=30,
            max_features=8,
        )
    )
    transformed = engineer.fit_transform(X, y)
    assert engineer.selection_report_["screening"] == "importance"
    assert "importance_ranking" in engineer.selection_report_
    pd.testing.assert_frame_equal(transformed, engineer.transform(X))


@pytest.mark.parametrize("value", [1, 0, "true", None, [], np.bool_(True)])
def test_frequency_wrapper_rejects_nonboolean_configuration(value):
    with pytest.raises(ValueError, match="include_frequency must be a boolean"):
        AutoFeatureEngineer(include_frequency=value)


@pytest.mark.parametrize(
    "options",
    [{}, {"selection_strategy": "predictive"}, {"selection_strategy": "predictive", "screening": "linear"}],
)
def test_frequency_wrapper_requires_predictive_importance(options):
    with pytest.raises(ValueError, match="include_frequency requires"):
        AutoFeatureEngineer(include_frequency=True, **options)


def test_frequency_wrapper_is_cloneable_forwards_option_and_validates_updates():
    X, y = example()
    X["category"] = np.where(np.arange(len(X)) % 3 == 0, "rare", "common")
    engineer = clone(
        AutoFeatureEngineer(
            selection_strategy="predictive",
            screening="importance",
            selection_estimator=Ridge(),
            include_frequency=True,
            task="regression",
            candidate_set="extended",
            max_candidates=30,
            max_features=8,
            stability_repeats=1,
        )
    )
    assert engineer.get_params()["include_frequency"] is True
    transformed = engineer.fit_transform(X, y)
    assert engineer._engine_instances["tabular"].config.include_frequency is True
    assert engineer.selection_report_["frequency_policy"]["operator"] == "fitted_frequency_v1"
    pd.testing.assert_frame_equal(transformed, engineer.transform(X))
    with pytest.raises(ValueError, match="include_frequency requires"):
        engineer.set_params(screening="auto")
    assert engineer.screening == "importance"
    assert engineer.include_frequency is True


def test_single_engine_frequency_wrapper_forwards_option():
    X, y = example()
    wrapper = clone(
        FeatureEngineerTransformer(
            engine="predictive",
            task="regression",
            estimator=Ridge(),
            screening="importance",
            include_frequency=True,
            max_candidates=20,
            max_features=6,
            stability_repeats=1,
        )
    )
    transformed = wrapper.fit_transform(X, y)
    assert wrapper.get_params()["include_frequency"] is True
    pd.testing.assert_frame_equal(transformed, wrapper.transform(X))


@pytest.mark.parametrize("quantiles", [(0.9, 0.1), (-0.1, 0.9), (0.1, np.inf), (True, 0.9), "clip"])
def test_numeric_clipping_wrapper_rejects_invalid_quantiles(quantiles):
    with pytest.raises(ValueError, match="quantile"):
        AutoFeatureEngineer(selection_strategy="predictive", numeric_clip_quantiles=quantiles)


def test_numeric_clipping_requires_predictive_and_preserves_raw_feature_outputs():
    quantiles = (0.01, 0.99)
    with pytest.raises(ValueError, match="requires selection_strategy"):
        AutoFeatureEngineer(numeric_clip_quantiles=quantiles)
    with pytest.raises(ValueError, match="requires screening='importance'"):
        AutoFeatureEngineer(selection_strategy="predictive", numeric_clip_quantiles=quantiles)
    X, y = example()
    engineer = clone(
        AutoFeatureEngineer(
            selection_strategy="predictive",
            selection_estimator=Ridge(),
            task="regression",
            screening="importance",
            numeric_clip_quantiles=quantiles,
            max_candidates=20,
            max_features=6,
            stability_repeats=1,
        )
    )
    output = engineer.fit_transform(X, y)
    assert engineer.get_params()["numeric_clip_quantiles"] == quantiles
    assert engineer._engine_instances["tabular"].config.numeric_clip_quantiles == quantiles
    pd.testing.assert_frame_equal(output[X.columns], X)
    pd.testing.assert_frame_equal(output, engineer.transform(X))
    with pytest.raises(ValueError, match="quantile"):
        engineer.set_params(numeric_clip_quantiles=(0.9, 0.1))
    assert engineer.numeric_clip_quantiles == quantiles


@pytest.mark.parametrize("value", [0, 2, 3, 6, True, 5.0, "5", None])
def test_importance_resampling_wrapper_rejects_undeclared_modes(value):
    with pytest.raises(ValueError, match="importance_resamples must be 1 or 5"):
        AutoFeatureEngineer(importance_resamples=value)


@pytest.mark.parametrize("options", [{}, {"selection_strategy": "predictive", "screening": "auto"}])
def test_importance_resampling_requires_predictive_importance(options):
    with pytest.raises(ValueError, match="requires predictive selection"):
        AutoFeatureEngineer(importance_resamples=5, **options)


def test_importance_resampling_wrapper_clones_forwards_and_preserves_originals():
    X, y = example()
    estimator = Ridge()
    engineer = clone(
        AutoFeatureEngineer(
            selection_strategy="predictive",
            selection_estimator=estimator,
            task="regression",
            screening="importance",
            numeric_clip_quantiles=(0.0, 1.0),
            importance_resamples=5,
            max_candidates=20,
            max_features=6,
            stability_repeats=1,
        )
    )
    output = engineer.fit_transform(X, y)
    assert engineer.get_params()["importance_resamples"] == 5
    assert engineer._engine_instances["tabular"].config.importance_resamples == 5
    pd.testing.assert_frame_equal(output[X.columns], X)
    pd.testing.assert_frame_equal(output, engineer.transform(X))
    assert not hasattr(estimator, "coef_")
    with pytest.raises(ValueError, match="requires predictive selection"):
        engineer.set_params(screening="auto", numeric_clip_quantiles=None)
    assert engineer.screening == "importance"
    assert engineer.importance_resamples == 5
    assert engineer.numeric_clip_quantiles == (0.0, 1.0)

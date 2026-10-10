"""Explicit task routing without changes to legacy selector defaults."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from featcopilot.selection import FeatureSelector, ImportanceSelector, StatisticalSelector
from featcopilot.utils.task import prepare_target, resolve_task, validate_task

SELECTORS = [StatisticalSelector, ImportanceSelector, FeatureSelector]


@pytest.fixture
def data():
    """Small numeric data with an integer-valued regression target."""
    rng = np.random.default_rng(42)
    y = np.tile([2, 10, 20], 20)
    X = pd.DataFrame({"signal": y + rng.normal(size=len(y)), "noise": rng.normal(size=len(y))})
    return X, y


@pytest.mark.parametrize("task", ["auto", "classification", "regression"])
def test_validate_task_accepts_supported_values(task):
    assert validate_task(task) is None


@pytest.mark.parametrize("task", ["binary", "Regression", "", None, 1])
def test_validate_task_rejects_invalid_values(task):
    with pytest.raises(ValueError, match="task must be"):
        validate_task(task)


@pytest.mark.parametrize(
    ("y", "task", "expected"),
    [
        ([0, 1, 0], "auto", "classification"),
        ([2, 10, 20], "auto", "classification"),
        ([0.0, 1.0, 0.0], "auto", "classification"),
        (["yes", "no", "yes"], "auto", "classification"),
        ([0.2, 0.5, 1.1], "auto", "regression"),
        ([2, 10, 20], "regression", "regression"),
        (np.array([2, 10, 20], dtype=object), "regression", "regression"),
        ([0.0, 1.0, 0.0], "classification", "classification"),
        (["yes", "no", "yes"], "classification", "classification"),
        (pd.Series([2, 10, 20], dtype="Int64"), "regression", "regression"),
    ],
)
def test_resolve_task(y, task, expected):
    assert resolve_task(y, task) == expected


@pytest.mark.parametrize(
    ("y", "task", "message"),
    [
        (["1", "2"], "regression", "numeric"),
        (["yes", "no"], "regression", "numeric"),
        (np.array([1, "2"], dtype=object), "regression", "numeric"),
        ([1 + 2j, 2 + 3j], "regression", "numeric"),
        ([0.2, 0.7], "classification", "discrete"),
        ([np.nan, 1.0], "regression", "missing"),
        ([np.inf, 1.0], "regression", "finite"),
        ([None, "yes"], "classification", "missing"),
        (pd.Series([1, pd.NA], dtype="Int64"), "regression", "missing"),
        ([], "auto", "non-empty"),
        ([[1, 2], [3, 4]], "auto", "one-dimensional"),
        ([[1, 2], [3, 4]], "regression", "one-dimensional"),
    ],
)
def test_resolve_task_rejects_incompatible_targets(y, task, message):
    with pytest.raises(ValueError, match=message):
        resolve_task(y, task)


def test_prepare_target_preserves_regression_values_instead_of_label_encoding():
    values = np.array([20, 2, 10, 2], dtype=object)
    prepared = prepare_target(values, "regression")
    np.testing.assert_array_equal(prepared, [20.0, 2.0, 10.0, 2.0])
    assert prepared.dtype.kind == "f"


@pytest.mark.parametrize("values", [["yes", "no", "yes"], [10, 2, 10], [1.0, 0.0, 1.0]])
def test_prepare_target_encodes_explicit_classification(values):
    np.testing.assert_array_equal(prepare_target(values, "classification"), [1, 0, 1])


def test_prepare_target_auto_does_not_encode_or_validate_legacy_targets():
    values = np.array([1, None], dtype=object)
    np.testing.assert_array_equal(prepare_target(values), values)


@pytest.mark.parametrize("selector_type", SELECTORS)
def test_task_is_last_positional_parameter_and_defaults_to_auto(selector_type):
    parameters = list(inspect.signature(selector_type).parameters.values())
    assert parameters[-2].name == "task"
    assert parameters[-2].default == "auto"
    assert parameters[-1].kind == inspect.Parameter.VAR_KEYWORD
    assert selector_type().task == "auto"
    with pytest.raises(ValueError, match="task must be"):
        selector_type(task="binary")


@pytest.mark.parametrize("method", ["mutual_info", "f_test"])
@pytest.mark.parametrize(
    ("task", "float_labels", "classification"),
    [
        ("auto", False, True),
        ("auto", True, False),
        ("regression", False, False),
        ("classification", True, True),
    ],
)
def test_statistical_scoring_routes_task(data, monkeypatch, method, task, float_labels, classification):
    X, y = data
    if float_labels:
        y = (y == 20).astype(float)
    scores = np.array([0.8, 0.2])
    result = scores if method == "mutual_info" else (scores, np.ones(2))
    classifier = MagicMock(return_value=result)
    regressor = MagicMock(return_value=result)
    prefix = "mutual_info" if method == "mutual_info" else "f"
    monkeypatch.setattr(f"sklearn.feature_selection.{prefix}_classif", classifier)
    monkeypatch.setattr(f"sklearn.feature_selection.{prefix}_regression", regressor)

    selector = StatisticalSelector(method=method, task=task).fit(X, y)

    chosen, unused = (classifier, regressor) if classification else (regressor, classifier)
    chosen.assert_called_once()
    unused.assert_not_called()
    assert selector.get_feature_scores() == {"signal": 0.8, "noise": 0.2}
    if task == "regression":
        np.testing.assert_array_equal(chosen.call_args.args[1], y)


@pytest.mark.parametrize("method", ["mutual_info", "f_test"])
def test_auto_statistical_routing_keeps_high_cardinality_integer_regression(data, monkeypatch, method):
    X, _ = data
    y = np.arange(len(X))
    scores = np.array([0.8, 0.2])
    result = scores if method == "mutual_info" else (scores, np.ones(2))
    regressor = MagicMock(return_value=result)
    prefix = "mutual_info" if method == "mutual_info" else "f"
    monkeypatch.setattr(f"sklearn.feature_selection.{prefix}_regression", regressor)
    StatisticalSelector(method=method).fit(X, y)
    regressor.assert_called_once()
    assert resolve_task(y) == "classification"


@pytest.mark.parametrize("method", ["mutual_info", "f_test", "chi2", "correlation"])
@pytest.mark.parametrize(
    ("task", "target", "message"),
    [
        ("regression", ["1", "2"], "numeric"),
        ("classification", [0.2, 0.7], "discrete"),
    ],
)
def test_statistical_methods_validate_explicit_targets(method, task, target, message):
    X = pd.DataFrame({"category": ["a", "b"]})
    with pytest.raises(ValueError, match=message):
        StatisticalSelector(method=method, task=task).fit(X, np.array(target))


@pytest.mark.parametrize(
    "selector",
    [StatisticalSelector(method="chi2", task="regression"), FeatureSelector(methods=["chi2"], task="regression")],
)
def test_chi2_rejects_explicit_regression(data, selector):
    X, y = data
    with pytest.raises(ValueError, match="chi2.*classification"):
        selector.fit(X, y)


def test_explicit_chi2_does_not_hide_scoring_failure(data, monkeypatch):
    X, y = data
    error = ValueError("chi-square failed")
    monkeypatch.setattr("sklearn.feature_selection.chi2", MagicMock(side_effect=error))
    fallback = MagicMock()
    monkeypatch.setattr(StatisticalSelector, "_compute_mutual_info", fallback)
    with pytest.raises(ValueError) as caught:
        StatisticalSelector(method="chi2", task="classification").fit(X, y)
    assert caught.value is error
    fallback.assert_not_called()


def test_regression_correlation_keeps_numeric_object_target_values(data):
    X, y = data
    selector = StatisticalSelector(method="correlation", task="regression").fit(X, y.astype(object))
    assert selector.get_feature_scores()["signal"] == pytest.approx(abs(np.corrcoef(X["signal"], y)[0, 1]))


@pytest.mark.parametrize(
    ("task", "float_labels", "model_type"),
    [
        ("auto", False, GradientBoostingClassifier),
        ("auto", True, GradientBoostingRegressor),
        ("regression", False, GradientBoostingRegressor),
        ("classification", True, GradientBoostingClassifier),
    ],
)
def test_importance_model_routes_task(data, task, float_labels, model_type):
    X, y = data
    if float_labels:
        y = (y == 20).astype(float)
    selector = ImportanceSelector(model="gradient_boosting", n_estimators=3, task=task).fit(X, y)
    assert isinstance(selector._model, model_type)
    assert selector.is_fitted


def test_auto_importance_keeps_high_cardinality_integer_regression(data):
    X, _ = data
    selector = ImportanceSelector(model="gradient_boosting", n_estimators=3).fit(X, np.arange(len(X)))
    assert isinstance(selector._model, GradientBoostingRegressor)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("model", ["random_forest", "gradient_boosting", "xgboost"])
def test_importance_task_controls_every_model_family(data, monkeypatch, task, model):
    import sys

    X, y = data
    fitted_model = MagicMock()
    fitted_model.feature_importances_ = np.array([0.8, 0.2])
    classifier = MagicMock(return_value=fitted_model)
    regressor = MagicMock(return_value=fitted_model)
    if model == "xgboost":
        monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBClassifier=classifier, XGBRegressor=regressor))
    else:
        prefix = "RandomForest" if model == "random_forest" else "GradientBoosting"
        monkeypatch.setattr(f"sklearn.ensemble.{prefix}Classifier", classifier)
        monkeypatch.setattr(f"sklearn.ensemble.{prefix}Regressor", regressor)

    ImportanceSelector(model=model, task=task).fit(X, y)
    chosen, unused = (classifier, regressor) if task == "classification" else (regressor, classifier)
    chosen.assert_called_once()
    unused.assert_not_called()
    expected = y.astype(float) if task == "regression" else np.tile([0, 1, 2], 20)
    np.testing.assert_array_equal(fitted_model.fit.call_args.args[1], expected)


@pytest.mark.parametrize("task", ["auto", "classification", "regression"])
@pytest.mark.parametrize("method", ["mutual_info", "f_test", "chi2", "correlation", "importance", "xgboost"])
def test_unified_children_receive_task(task, method):
    selector = FeatureSelector(task=task)
    child = selector._create_selector(method)
    assert child.task == task


@pytest.mark.parametrize(
    ("task", "labels", "classification"),
    [
        ("auto", [2, 10, 20], True),
        ("auto", [0.0, 1.0, 0.0], False),
        ("auto", ["a", "b", "a"], False),
        ("regression", [2, 10, 20], False),
        ("classification", [0.0, 1.0, 0.0], True),
        ("classification", ["a", "b", "a"], True),
    ],
)
def test_unified_refinement_routes_and_encodes_task(data, monkeypatch, task, labels, classification):
    X, _ = data
    y = np.tile(labels, 20)
    model = MagicMock()
    model.feature_importances_ = np.array([0.8, 0.2])
    classifier = MagicMock(return_value=model)
    regressor = MagicMock(return_value=model)
    monkeypatch.setattr("sklearn.ensemble.GradientBoostingClassifier", classifier)
    monkeypatch.setattr("sklearn.ensemble.GradientBoostingRegressor", regressor)
    selector = FeatureSelector(task=task)

    assert selector._l1_refine(X, y, ["signal", "noise"]) == ["signal"]
    chosen, unused = (classifier, regressor) if classification else (regressor, classifier)
    chosen.assert_called_once()
    unused.assert_not_called()
    fitted_target = model.fit.call_args.args[1]
    if task == "classification":
        assert fitted_target.dtype.kind in "iu"
        np.testing.assert_array_equal(fitted_target, prepare_target(y, task))
    else:
        np.testing.assert_array_equal(fitted_target, y)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_explicit_refinement_does_not_swallow_fit_errors(data, monkeypatch, task):
    X, y = data
    error = RuntimeError("model fit failed")
    model = MagicMock()
    model.fit.side_effect = error
    monkeypatch.setattr("sklearn.ensemble.GradientBoostingClassifier", lambda **kwargs: model)
    monkeypatch.setattr("sklearn.ensemble.GradientBoostingRegressor", lambda **kwargs: model)
    with pytest.raises(RuntimeError) as caught:
        FeatureSelector(task=task)._l1_refine(X, y, ["signal", "noise"])
    assert caught.value is error


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_unified_fits_children_and_refinement_with_explicit_targets(data, task):
    X, y = data
    if task == "classification":
        y = np.where(y == 20, "yes", "no")
    selector = FeatureSelector(methods=["mutual_info", "f_test"], task=task, correlation_threshold=1.0)
    result = selector.fit_transform(X, y)
    assert selector.is_fitted
    assert list(result.columns) == selector.get_selected_features()
    assert "signal" in result.columns
    assert all(child.task == task for child in selector._selectors.values())


@pytest.mark.parametrize("selector_type", SELECTORS)
def test_failed_task_refit_invalidates_previous_selection(data, selector_type):
    X, y = data
    selector = selector_type(task="regression")
    if isinstance(selector, ImportanceSelector):
        selector.model_type = "gradient_boosting"
        selector.n_estimators = 3
    if isinstance(selector, FeatureSelector):
        selector.methods = ["f_test"]
    selector.fit(X, y)
    assert selector.is_fitted

    selector.task = "classification"
    with pytest.raises(ValueError, match="discrete"):
        selector.fit(X, y + np.linspace(0.01, 0.99, len(y)))
    assert not selector.is_fitted
    assert selector.get_feature_scores() == {}
    assert selector.get_selected_features() == []
    with pytest.raises(RuntimeError, match="fitted"):
        selector.transform(X)


@pytest.mark.parametrize("selector_type", SELECTORS)
def test_selectors_revalidate_mutated_task(data, selector_type):
    X, y = data
    selector = selector_type()
    selector.task = "invalid"
    with pytest.raises(ValueError, match="task must be"):
        selector.fit(X, y)
    assert not selector.is_fitted


@pytest.mark.parametrize("selector_type", SELECTORS)
def test_regression_string_target_is_rejected_even_without_numeric_features(selector_type):
    X = pd.DataFrame({"category": ["a", "b", "a"]})
    with pytest.raises(ValueError, match="numeric"):
        selector_type(task="regression").fit(X, np.array(["1", "2", "1"]))


def test_importance_refit_replaces_regressor_with_classifier(data):
    X, y = data
    selector = ImportanceSelector(model="gradient_boosting", n_estimators=3, task="regression").fit(X, y)
    previous_model = selector._model
    selector.task = "classification"
    selector.fit(X, np.where(y == 20, "yes", "no"))
    assert isinstance(selector._model, GradientBoostingClassifier)
    assert selector._model is not previous_model
    assert selector.is_fitted


def test_unified_refit_replaces_task_specific_children(data):
    X, y = data
    selector = FeatureSelector(methods=["f_test", "correlation"], task="regression", correlation_threshold=1.0)
    selector.fit(X, y)
    old_child = selector._selectors["f_test"]
    selector.task = "classification"
    selector.methods = ["f_test"]
    selector.fit(X, (y == 20).astype(float))
    assert set(selector._selectors) == {"f_test"}
    assert set(selector.get_method_scores()) == {"f_test"}
    assert selector._selectors["f_test"] is not old_child
    assert selector._selectors["f_test"].task == "classification"

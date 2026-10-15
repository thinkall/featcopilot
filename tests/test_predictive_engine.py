"""Inductive selection and production-schema tests for the opt-in numeric engine."""

import json
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.engines.predictive import PredictiveEstimator, PredictiveTabularEngine


@pytest.fixture(autouse=True)
def limit_threads():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(731)
    X = pd.DataFrame(rng.normal(size=(320, 3)), columns=["a", "b", "linear"])
    X.index = pd.Index(np.arange(len(X)) * 7 + 13, name="sample")
    y = 50 * X["linear"] + 6 * X["a"] * X["b"] + rng.normal(scale=0.15, size=len(X))
    return X, y


def ridge_engine(**kwargs):
    kwargs.setdefault("stability_repeats", 1)
    return PredictiveTabularEngine(estimator=Ridge(alpha=0.01), task="regression", max_candidates=40, **kwargs)


class DuckRegressor:
    """Sklearn-compatible composition without BaseEstimator inheritance."""

    _estimator_type = "regressor"

    def __init__(self, alpha=0.01, n_jobs=1, random_state=None):
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.random_state = random_state

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "n_jobs": self.n_jobs, "random_state": self.random_state}

    def set_params(self, **params):
        for name, value in params.items():
            if name not in self.get_params():
                raise ValueError(f"Unknown parameter: {name}")
            setattr(self, name, value)
        return self

    def fit(self, X, y):
        self.model_ = Ridge(alpha=self.alpha).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def __sklearn_tags__(self):
        return Ridge().__sklearn_tags__()


def test_residual_signal_and_external_generalization(regression_data):
    X, y = regression_data
    engine = ridge_engine(max_features=4).fit(X, y)
    assert engine.get_feature_names() == ["a_x_b"]
    assert engine.selection_report_["candidate_ranking"][0] == "a_x_b"
    assert engine.selection_report_["selected_count"] == 1
    assert engine.selection_report_["baseline_loss"] > 1

    # This independent population and criterion are fixed before engine fitting.
    rng = np.random.default_rng(992)
    future = pd.DataFrame(rng.normal(size=(500, 3)), columns=X.columns)
    target = 50 * future["linear"] + 6 * future["a"] * future["b"]
    baseline = Ridge(alpha=0.01).fit(X, y)
    enhanced = Ridge(alpha=0.01).fit(engine.transform(X), y)
    assert mean_squared_error(target, enhanced.predict(engine.transform(future))) < (
        0.1 * mean_squared_error(target, baseline.predict(future))
    )


def test_logistic_nonlinear_external_generalization():
    rng = np.random.default_rng(72)
    X = pd.DataFrame(rng.normal(size=(600, 2)), columns=["a", "b"])
    probabilities = 1 / (1 + np.exp(-4 * X["a"] * X["b"]))
    y = rng.binomial(1, probabilities)
    engine = PredictiveTabularEngine(
        task="classification", estimator=LogisticRegression(max_iter=500), max_features=3, max_candidates=24
    ).fit(X, y)

    # Neither these independent rows nor their labels participate in selection.
    future_rng = np.random.default_rng(73)
    future = pd.DataFrame(future_rng.normal(size=(1000, 2)), columns=X.columns)
    future_probabilities = 1 / (1 + np.exp(-4 * future["a"] * future["b"]))
    future_y = future_rng.binomial(1, future_probabilities)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X, y)
    enhanced = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(engine.transform(X), y)
    assert log_loss(future_y, enhanced.predict_proba(engine.transform(future))) < (
        0.8 * log_loss(future_y, baseline.predict_proba(future))
    )


def test_default_histogram_learner_improves_nonlinear_signal():
    rng = np.random.default_rng(83)
    X = pd.DataFrame(rng.uniform(-2, 2, size=(600, 2)), columns=["a", "b"])
    y = X["a"] * X["b"] + rng.normal(scale=0.05, size=len(X))
    engine = PredictiveTabularEngine(task="regression", max_features=3, max_candidates=24).fit(X, y)
    assert engine.get_feature_names() == ["a_x_b"]
    report = engine.selection_report_
    assert report["validation_scores"][-1]["loss"] < report["baseline_loss"] * 0.85


def test_integer_regression_override(regression_data):
    X, y = regression_data
    engine = ridge_engine(max_features=4).fit(X, np.rint(y).astype(int))
    assert engine.selection_report_["task"] == "regression"
    assert engine.selection_report_["metric"] == "rmse"
    assert engine.get_feature_names() == ["a_x_b"]
    assert isinstance(engine, BaseEngine)
    assert isinstance(engine.config, EngineConfig)


@pytest.mark.parametrize("classes", [2, 3])
def test_string_classification_and_multiclass_residuals(classes):
    rng = np.random.default_rng(45)
    X = pd.DataFrame(rng.normal(size=(400, 2)), columns=["a", "b"])
    labels = pd.cut(X["a"] * X["b"], [-np.inf, *([0] if classes == 2 else [-0.4, 0.4]), np.inf], labels=False)
    y = np.array(["first", "second", "third"])[labels]
    engine = PredictiveTabularEngine(
        task="auto", estimator=LogisticRegression(max_iter=300), max_candidates=24, max_features=3
    ).fit(X, y)
    assert engine.selection_report_["task"] == "classification"
    assert engine.selection_report_["metric"] == "log_loss"
    assert len(engine.get_feature_names()) == 1
    assert engine.get_feature_names()[0] in {"a_x_b", "a_div_b", "b_div_a"}
    report = engine.selection_report_
    assert report["validation_scores"][1]["loss"] < report["baseline_loss"] * 0.999


def test_default_classifier_with_categories():
    X = pd.DataFrame({"category": pd.Categorical(["a", "b", "c"] * 40), "empty": [np.nan] * 120})
    y = np.array(["no", "yes", "yes"] * 40)
    engine = PredictiveTabularEngine(max_candidates=8).fit(X, y)
    assert engine.selection_report_["task"] == "classification"
    assert engine.selection_report_["candidate_count"] == 0
    assert_frame_equal(engine.transform(X), X)
    future = pd.DataFrame({"category": pd.Categorical(["unseen", None]), "empty": [np.nan, 1.0]})
    assert_frame_equal(engine.transform(future), future)


@pytest.mark.parametrize("dtype", ["object", "Int64", "Float64", "boolean"])
def test_numeric_classification_targets_in_extension_or_object_dtype(dtype):
    X = pd.DataFrame({"a": np.linspace(-2, 2, 80)})
    target = pd.Series([0, 1] * 40, dtype=dtype)
    engine = PredictiveTabularEngine(estimator=DummyClassifier(), max_candidates=1).fit(X, target)
    assert engine.selection_report_["task"] == "classification"


def test_small_class_partitions_require_nested_residual_coverage():
    X = pd.DataFrame({"a": np.arange(9, dtype=float)})
    y = np.array(["one", "two", "three"] * 3)
    with pytest.raises(ValueError, match="Classification CV"):
        PredictiveTabularEngine(estimator=DummyClassifier(), max_candidates=1, stability_repeats=1).fit(X, y)
    X = pd.DataFrame({"a": np.arange(12, dtype=float)})
    y = np.array(["one", "two", "three"] * 4)
    fitted = PredictiveTabularEngine(estimator=DummyClassifier(), max_candidates=1, stability_repeats=1).fit(X, y)
    assert fitted.selection_report_["residual_cv_folds"] == 3


def test_continuous_auto_task(regression_data):
    X, y = regression_data
    engine = PredictiveTabularEngine(estimator=Ridge(), max_candidates=1).fit(X, y)
    assert engine.selection_report_["task"] == "regression"


def test_estimator_is_cloned_without_mutation(regression_data):
    X, y = regression_data
    supplied = Ridge(alpha=0.1).fit(X, y)
    before = pickle.dumps(supplied)
    engine = PredictiveTabularEngine(task="regression", estimator=supplied, max_candidates=24).fit(X, y)
    assert pickle.dumps(supplied) == before
    assert engine.config.estimator is supplied
    unfitted = Ridge()
    PredictiveTabularEngine(task="regression", estimator=unfitted, max_candidates=8).fit(X, y)
    assert not hasattr(unfitted, "coef_")


def test_cloneable_duck_learner_preserves_fitted_caller_and_parameters(regression_data):
    X, y = regression_data
    supplied = DuckRegressor(n_jobs=2).fit(X, y)
    before = pickle.dumps(supplied)
    assert not isinstance(supplied, BaseEstimator)
    assert isinstance(supplied, PredictiveEstimator)
    engine = PredictiveTabularEngine(
        task="regression", estimator=supplied, max_features=4, max_candidates=12, stability_repeats=1
    ).fit(X, y)
    assert pickle.dumps(supplied) == before
    assert supplied.n_jobs == 2
    assert supplied.random_state is None
    assert engine.config.estimator is supplied
    assert engine.get_feature_names() == ["a_x_b"]
    assert_frame_equal(engine.transform(X)[X.columns], X)
    restored = pickle.loads(pickle.dumps(engine))
    assert_frame_equal(restored.transform(X), engine.transform(X))


def test_estimator_interface_requires_callable_methods():
    invalid = DuckRegressor()
    invalid.predict = 42
    with pytest.raises(ValueError, match="estimator"):
        PredictiveTabularEngine(estimator=invalid)


def test_estimator_must_be_cloneable_before_selection():
    class InvalidParameters(DuckRegressor):
        def get_params(self, deep=True):
            return {"not_a_constructor_parameter": 1}

    with pytest.raises(ValueError, match="cloneable"):
        PredictiveTabularEngine(estimator=InvalidParameters())


def test_clone_must_not_return_the_caller():
    supplied = DuckRegressor()
    with patch("featcopilot.engines.predictive.clone", side_effect=lambda estimator: estimator):
        with pytest.raises(ValueError, match="independent estimator"):
            PredictiveTabularEngine(estimator=supplied)
    assert not hasattr(supplied, "model_")


def test_cloneable_catboost_classifier_without_base_inheritance():
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(82)
    X = pd.DataFrame(rng.normal(size=(80, 2)), columns=["a", "b"])
    y = np.where(X["a"] * X["b"] > 0, "yes", "no")
    supplied = catboost.CatBoostClassifier(
        iterations=5, depth=2, thread_count=1, verbose=False, allow_writing_files=False
    )
    assert not isinstance(supplied, BaseEstimator)
    engine = PredictiveTabularEngine(
        task="classification", estimator=supplied, max_candidates=10, max_features=3, stability_repeats=1
    ).fit(X, y)
    assert engine.config.estimator is supplied
    assert not supplied.is_fitted()
    assert engine.selection_report_["task"] == "classification"
    assert np.isfinite(engine.selection_report_["baseline_loss"])
    assert_frame_equal(engine.transform(X)[X.columns], X)


def test_all_cpu_setting_resolves_native_limits_without_changing_selection(regression_data):
    X, y = regression_data
    expected = ridge_engine(max_features=4).fit(X, y)
    with (
        patch("featcopilot.engines.predictive.effective_n_jobs", return_value=1) as resolve_jobs,
        patch("featcopilot.engines.predictive.threadpool_limits", wraps=threadpool_limits) as native_limits,
    ):
        engine = ridge_engine(max_features=4, n_jobs=-1).fit(X, y)
    resolve_jobs.assert_called_once_with(-1)
    native_limits.assert_called_once_with(limits=1)
    assert engine.config.n_jobs == -1
    assert engine.selection_report_ == expected.selection_report_


def test_all_cpu_setting_applies_to_cloned_learner_only():
    supplied = RandomForestRegressor(n_estimators=1, n_jobs=2, random_state=7)
    engine = PredictiveTabularEngine(task="regression", estimator=supplied, n_jobs=-1)
    cloned = engine._learner("regression")
    assert cloned is not supplied
    assert cloned.n_jobs == -1
    assert cloned.random_state == 7
    assert supplied.n_jobs == 2


def test_holdout_labels_and_values_cannot_affect_ranking_or_imputer_fit(regression_data):
    X, y = regression_data
    train, validation = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
    X = X.copy()
    X.loc[X.index[::11], "a"] = np.nan
    X["category"] = pd.Series(["blue", "red"] * (len(X) // 2), index=X.index, dtype="string")
    observed_fit_indices = []
    original_fit = SimpleImputer.fit

    def record_fit(imputer, values, target=None, **kwargs):
        observed_fit_indices.append(set(values.index))
        return original_fit(imputer, values, target, **kwargs)

    with patch.object(SimpleImputer, "fit", record_fit):
        first = ridge_engine(max_features=5).fit(X, y)
    assert observed_fit_indices
    assert all(indices <= set(X.index[train]) for indices in observed_fit_indices)

    changed_X = X.copy()
    changed_X.loc[X.index[validation], "a"] = 1e6
    changed_X.loc[X.index[validation], "category"] = "validation-only-category"
    changed_y = y.copy()
    changed_y.iloc[validation] = np.arange(len(validation)) * -1234
    second = ridge_engine(max_features=5).fit(changed_X, changed_y)
    assert first.selection_report_["candidate_ranking"] == second.selection_report_["candidate_ranking"]
    assert first.selection_report_["candidate_scores"] == second.selection_report_["candidate_scores"]
    assert first.selection_report_["baseline_loss"] != second.selection_report_["baseline_loss"]


def test_batch_invariance_schema_indices_and_metadata(regression_data):
    X, y = regression_data
    X["category"] = pd.Categorical(["red", "blue"] * (len(X) // 2))
    original = X.copy(deep=True)
    engine = ridge_engine(max_features=5).fit(X, y)
    full = engine.transform(X)
    chunks = pd.concat([engine.transform(X.iloc[:37]), engine.transform(X.iloc[37:])])
    assert_frame_equal(full, chunks)
    assert_frame_equal(full, engine.transform(X[X.columns[::-1]]))
    assert_frame_equal(full.loc[:, X.columns], original)
    assert_frame_equal(X, original)
    assert full.columns.tolist() == X.columns.tolist() + engine.get_feature_names()
    assert engine.get_feature_set().get_names() == engine.get_feature_names()
    for feature in engine.get_feature_set():
        np.testing.assert_allclose(feature.compute(X), full[feature.name], equal_nan=True)
    assert engine.transform(X.iloc[:0]).columns.tolist() == full.columns.tolist()
    names = engine.get_feature_names()
    names.clear()
    assert engine.get_feature_names()


def test_duplicate_row_indices_and_defensive_feature_metadata(regression_data):
    X, y = regression_data
    X = X.copy()
    X.index = ["duplicate"] * len(X)
    X.columns.name = "measurements"
    engine = ridge_engine(max_features=4).fit(X, y.to_numpy())
    output = engine.transform(X)
    assert_frame_equal(output[X.columns], X)
    assert_frame_equal(output, pd.concat([engine.transform(X.iloc[:5]), engine.transform(X.iloc[5:])]))
    metadata = engine.get_feature_set()
    metadata[engine.get_feature_names()[0]].source_columns.clear()
    engine.get_feature_metadata()[engine.get_feature_names()[0]]["source_columns"].clear()
    assert_frame_equal(output, engine.transform(X))


def test_fit_transform_matches_fit_then_transform(regression_data):
    X, y = regression_data
    direct = ridge_engine(max_features=5).fit_transform(X, y)
    separate = ridge_engine(max_features=5).fit(X, y).transform(X)
    assert_frame_equal(direct, separate)


def test_permuted_fit_columns_have_identical_rank(regression_data):
    X, y = regression_data
    first = ridge_engine(max_features=5).fit(X, y)
    second = ridge_engine(max_features=5).fit(X[X.columns[::-1]], y)
    assert first.selection_report_["candidate_ranking"] == second.selection_report_["candidate_ranking"]
    assert first.selection_report_["candidate_scores"] == second.selection_report_["candidate_scores"]
    assert first.get_feature_names() == second.get_feature_names()


@pytest.mark.parametrize("maximum", [0, 1, 3, 4, None])
def test_max_features_is_total_and_never_drops_originals(regression_data, maximum):
    X, y = regression_data
    engine = ridge_engine(max_features=maximum)
    result = engine.fit_transform(X, y)
    assert_frame_equal(result[X.columns], X)
    if maximum is not None:
        assert result.shape[1] <= max(maximum, X.shape[1])
    assert result.shape[1] == X.shape[1] + len(engine.get_feature_names())
    assert engine.selection_report_["selected_count"] == len(engine.get_feature_names())
    budget = 50 if maximum is None else min(50, max(0, maximum - X.shape[1]))
    assert all(score["count"] <= budget for score in engine.selection_report_["validation_scores"])
    if budget == 0:
        report = engine.selection_report_
        assert report["candidate_count"] == report["screened_count"] == report["selected_count"] == 0
        assert report["selected_features"] == report["candidate_ranking"] == []
        assert [score["count"] for score in report["validation_scores"]] == [0]
        assert np.isfinite(report["baseline_loss"])
        assert report["train_rows"] + report["validation_rows"] == len(X)


def test_candidate_budget_interleaves_families_on_wide_input():
    rng = np.random.default_rng(16)
    X = pd.DataFrame(rng.normal(size=(100, 500)), columns=[f"x{i:04d}" for i in range(500)])
    engine = PredictiveTabularEngine(
        task="regression", estimator=DummyRegressor(), max_candidates=12, max_features=503
    ).fit(X, rng.normal(size=len(X)))
    report = engine.selection_report_
    assert report["candidate_count"] == 12
    assert report["screened_count"] <= 12
    assert len(report["candidate_ranking"]) <= 12
    names = report["candidate_ranking"]
    assert any("_x_" in name for name in names)
    assert any("_minus_" in name for name in names)
    assert "x0000_div_x0001" in names
    assert "x0001_div_x0000" in names
    assert "x0000_sq" in names
    assert not any("_pow" in name for name in names)


@pytest.mark.parametrize("budget", [0, 1, 2, 13])
def test_candidate_metadata_allocation_is_strictly_bounded(regression_data, budget):
    X, y = regression_data
    with patch.object(PredictiveTabularEngine, "_feature", wraps=PredictiveTabularEngine._feature) as feature_factory:
        engine = PredictiveTabularEngine(
            task="regression", estimator=DummyRegressor(), max_candidates=budget, stability_repeats=1
        ).fit(X, y)
    assert feature_factory.call_count == budget
    assert engine.selection_report_["candidate_count"] == budget


def test_binary_interactions_and_constants():
    rng = np.random.default_rng(65)
    X = pd.DataFrame({"a": rng.integers(0, 2, 240).astype(bool), "b": rng.normal(size=240), "constant": 1.0})
    y = 5 * X["a"] * X["b"]
    engine = ridge_engine(max_features=4).fit(X, y)
    assert engine.get_feature_names() == ["a_x_b"]
    assert all("constant" not in name for name in engine.selection_report_["candidate_ranking"])
    assert "a_sq" not in engine.selection_report_["candidate_ranking"]


@pytest.mark.parametrize("collision", ["a_sq", "a_x_b", "a_div_b"])
def test_generated_names_never_overwrite_originals(regression_data, collision):
    X, y = regression_data
    X = X.assign(**{collision: 0.0})
    with pytest.raises(ValueError, match="collision"):
        ridge_engine().fit(X, y)


def test_missing_nonfinite_and_nullable_values(regression_data):
    X, y = regression_data
    X = X.copy()
    X["a"] = X["a"].astype("Float64")
    X.iloc[0, 0] = pd.NA
    X.iloc[1, 1] = np.inf
    X.iloc[2, 1] = -np.inf
    X["all_missing"] = np.nan
    X["category"] = pd.Series(["a", None, 1, "1"] * 80, index=X.index, dtype=object)
    engine = ridge_engine(max_features=6).fit(X, y)
    output = engine.transform(X)
    assert_frame_equal(output[X.columns], X)
    assert engine.get_feature_names() == ["a_x_b"]
    assert output["a_x_b"].iloc[:3].isna().all()
    future = X.iloc[:3].copy()
    future.iloc[0, 0] = 1e300
    future.iloc[0, 1] = 1e300
    result = engine.transform(future)
    assert np.isnan(result["a_x_b"].iloc[0])
    assert result["a"].iloc[0] == 1e300
    assert_frame_equal(engine.transform(future.iloc[:1]), result.iloc[:1])


def test_all_candidate_code_matches_stateless_computation():
    engine = PredictiveTabularEngine()
    X = pd.DataFrame(
        {
            "a'quoted": pd.array([0, 1, -1e-8, pd.NA, np.inf, -2, 1e300], dtype="Float64"),
            "b": [0, -1, 2, 3, 4, -np.inf, 1e300],
        }
    )
    for feature in engine._candidate_stream(sorted(X.columns), sorted(X.columns)):
        np.testing.assert_allclose(feature.compute(X), engine._compute(feature, X), equal_nan=True)


def test_refit_and_failed_refit_clear_state(regression_data):
    X, y = regression_data
    engine = ridge_engine(max_features=4).fit(X, y)
    assert engine.get_feature_names()
    changed = pd.DataFrame({"constant": np.ones(len(X))})
    engine.fit(changed, y)
    assert engine.get_feature_names() == []
    assert len(engine.get_feature_set()) == 0
    assert engine.selection_report_["candidate_count"] == 0
    assert_frame_equal(engine.transform(changed), changed)
    with pytest.raises(ValueError, match="y is required"):
        engine.fit(X)
    assert not engine.is_fitted
    assert engine.get_feature_names() == []
    assert engine.selection_report_ == {}
    with pytest.raises(RuntimeError, match="fitted"):
        engine.transform(X)


def test_serialization_and_report_contains_no_training_state(regression_data):
    X, y = regression_data
    engine = ridge_engine(max_features=4).fit(X, y)
    restored = pickle.loads(pickle.dumps(engine))
    assert_frame_equal(engine.transform(X), restored.transform(X))
    assert engine.selection_report_ == restored.selection_report_
    json.dumps(engine.selection_report_, allow_nan=False)
    assert engine.selection_report_["internal_validation_only"] is True
    assert all(not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)) for value in vars(engine).values())
    assert all(not isinstance(value, BaseEstimator) for value in vars(engine).values())


def test_exact_fit_counts_and_transform_does_not_fit(regression_data):
    X, y = regression_data
    original_fit = Ridge.fit
    fit_widths = []

    def record_fit(estimator, values, target, **kwargs):
        fit_widths.append(values.shape[1])
        return original_fit(estimator, values, target, **kwargs)

    with patch.object(Ridge, "fit", record_fit):
        engine = ridge_engine(max_features=9).fit(X, y)
        count = len(fit_widths)
        engine.transform(X.iloc[:2])
        engine.transform(X.iloc[2:])
    report = engine.selection_report_
    assert count == report["learner_fit_count"] == report["residual_cv_folds"] * 2 + len(report["validation_scores"])
    assert len(fit_widths) == count
    assert [score["count"] for score in report["validation_scores"]] == [0, 1, 5, 6]
    assert max(fit_widths) <= 9


def test_no_gain_retains_only_originals(regression_data):
    X, y = regression_data
    engine = PredictiveTabularEngine(task="regression", estimator=DummyRegressor(), max_candidates=20).fit(X, y)
    assert engine.get_feature_names() == []
    assert_frame_equal(engine.transform(X), X)


def test_near_ties_prefer_smaller_subset(regression_data):
    X, y = regression_data
    losses = iter([10.0, 9.0, 8.995, 8.0])
    with patch.object(PredictiveTabularEngine, "_loss", side_effect=lambda *args: next(losses)):
        engine = ridge_engine(max_features=9).fit(X, y)
    assert engine.selection_report_["selected_count"] == 6
    losses = iter([10.0, 9.0, 8.995, 8.999])
    with patch.object(PredictiveTabularEngine, "_loss", side_effect=lambda *args: next(losses)):
        engine = ridge_engine(max_features=9).fit(X, y)
    assert engine.selection_report_["selected_count"] == 1


def test_minimum_relative_improvement_is_strict(regression_data):
    X, y = regression_data
    losses = iter([100.0, 99.9])
    with patch.object(PredictiveTabularEngine, "_loss", side_effect=lambda *args: next(losses)):
        engine = ridge_engine(max_features=4).fit(X, y)
    assert engine.get_feature_names() == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_candidates": -1}, "max_candidates"),
        ({"max_features": -1}, "max_features"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"n_jobs": -2}, "n_jobs"),
        ({"task": "forecasting"}, "task"),
    ],
)
def test_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PredictiveTabularEngine(**kwargs)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (None, "y is required"),
        (np.ones((20, 2)), "one-dimensional"),
        (np.ones(19), "same number"),
        (np.full(20, np.nan), "missing"),
        (np.full(20, np.inf), "finite"),
        (np.ones(20), "constant"),
        (np.arange(20) + 1j, "Complex"),
    ],
)
def test_invalid_targets(target, message):
    X = pd.DataFrame({"a": np.arange(20)})
    with pytest.raises(ValueError, match=message):
        PredictiveTabularEngine(task="regression").fit(X, target)


def test_shared_task_rules_reject_numeric_strings_for_regression():
    X = pd.DataFrame({"a": np.arange(20, dtype=float)})
    with pytest.raises(ValueError, match="real numeric targets"):
        ridge_engine().fit(X, np.arange(20).astype(str))


@pytest.mark.parametrize(
    ("X", "y", "kwargs"),
    [
        (pd.DataFrame({"a": range(4)}), np.arange(4), {"task": "regression"}),
        (pd.DataFrame({"a": range(20)}), np.array(["a"] * 19 + ["b"]), {"task": "classification"}),
        (pd.DataFrame({"a": [np.nan] * 20}), np.arange(20), {"task": "regression"}),
    ],
)
def test_small_or_degenerate_split_errors_are_explicit(X, y, kwargs):
    with pytest.raises(ValueError, match="(?i)(holdout|split|finite|samples)"):
        PredictiveTabularEngine(**kwargs).fit(X, y)


def test_constant_internal_training_target_is_rejected():
    X = pd.DataFrame({"a": np.arange(20, dtype=float)})
    train, validation = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
    target = np.ones(len(X))
    target[validation] = 2
    with pytest.raises(ValueError, match="training split has a constant target"):
        ridge_engine().fit(X, target)


def test_schema_validation_and_numpy_input(regression_data):
    X, y = regression_data
    engine = ridge_engine(max_features=4).fit(X, y)
    with pytest.raises(ValueError, match="schema"):
        engine.transform(X.drop(columns="b"))
    with pytest.raises(ValueError, match="schema"):
        engine.transform(X.assign(extra=1))
    with pytest.raises(TypeError, match="numeric"):
        engine.transform(X.assign(a="not-a-number"))
    with pytest.raises(ValueError, match="unique"):
        ridge_engine().fit(pd.DataFrame(np.ones((len(X), 2)), columns=["a", "a"]), y)
    with pytest.raises(ValueError, match="string"):
        ridge_engine().fit(pd.DataFrame(X.to_numpy()), y)
    with pytest.raises(ValueError, match="two-dimensional"):
        ridge_engine().fit(np.arange(len(X)), y)
    array_engine = ridge_engine(max_features=4).fit(X.to_numpy(), y.to_numpy())
    output = array_engine.transform(X.to_numpy())
    assert output.columns[:3].tolist() == ["feature_0", "feature_1", "feature_2"]
    np.testing.assert_array_equal(output.iloc[:, :3], X.to_numpy())


def test_wrong_estimator_task_and_unsupported_fit_arguments(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="regression"):
        PredictiveTabularEngine(task="regression", estimator=DummyClassifier()).fit(X, y)
    with pytest.raises(TypeError, match="Unsupported fit"):
        ridge_engine().fit(X, y, sample_weight=np.ones(len(X)))
    engine = ridge_engine(max_features=4).fit(X, y)
    with pytest.raises(TypeError, match="Unsupported transform"):
        engine.transform(X, ignored=True)


class FailingRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        raise ValueError("injected learner failed")

    def predict(self, X):
        return np.zeros(len(X))


def test_estimator_errors_are_not_silently_swallowed(regression_data):
    X, y = regression_data
    engine = PredictiveTabularEngine(task="regression", estimator=FailingRegressor())
    with pytest.raises(ValueError, match="injected learner failed"):
        engine.fit(X, y)
    assert not engine.is_fitted

"""Matched model preprocessing and metric coverage for stronger raw baselines."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from benchmarks.tabular import (
    NativeCategoricalFrame,
    candidate_parameters,
    generate_features,
    make_model,
    metrics,
    prepare_numeric,
)
from featcopilot import AutoFeatureEngineer


def test_native_categorical_preprocessing_preserves_unseen_values_and_schema():
    train = pd.DataFrame({"number": [1.0, np.nan], "category": ["a", None]})
    test = pd.DataFrame({"number": [3.0], "category": ["unseen"]})
    transformer = NativeCategoricalFrame().fit(train)
    transformed = transformer.transform(test)
    assert transformed.category.tolist() == ["value:unseen"]
    assert transformed.number.tolist() == [3.0]
    assert transformer.transform(train).category.tolist() == ["value:a", "missing:"]
    with pytest.raises(ValueError, match="schema"):
        transformer.transform(test[["category", "number"]])


def test_robust_boolean_passthrough_keeps_numeric_learner_encoding():
    from featcopilot.utils.numeric_clipping import NumericQuantileClipper

    train = pd.DataFrame({"number": [1.0, 2.0], "flag": [True, True], "cat": ["a", "b"]})
    test = pd.DataFrame({"number": [100.0], "flag": [False], "cat": ["new"]})
    clipped = NumericQuantileClipper(quantiles=(0.0, 1.0)).fit(train).transform(test)
    assert clipped.flag.dtype == bool
    assert clipped.flag.tolist() == [False]
    assert clipped.number.tolist() == [2.0]
    native = NativeCategoricalFrame(boolean_as_numeric=True).fit(train)
    assert native.categorical_ == ["cat"]
    assert native.transform(clipped).flag.tolist() == [0.0]
    model = make_model(
        train,
        "regression",
        "linear",
        {"regularization": 1.0},
        {"seed": 42, "threads": 1, "numeric_clip_quantiles": (0.0, 1.0)},
    )
    model.fit(train, [0.0, 1.0])
    assert model.steps[1][1].transformers_[0][2] == ["number", "flag"]


@pytest.mark.parametrize("learner", ["catboost", "xgboost"])
def test_stronger_learners_fit_mixed_data_and_emit_probability_metrics(learner):
    if importlib.util.find_spec(learner) is None:
        pytest.skip(f"{learner} is optional outside benchmark environment")
    rng = np.random.default_rng(44)
    train = pd.DataFrame({"a": rng.normal(size=80), "cat": ["one", "two"] * 40})
    test = pd.DataFrame({"a": rng.normal(size=20), "cat": ["two", "unseen"] * 10})
    train, test = prepare_numeric(train, test)
    y = pd.Series([0, 1] * 40)
    model = make_model(train, "classification", learner, candidate_parameters(learner)[0], {"seed": 42, "threads": 1})
    model.fit(train, y)
    result = metrics(model, test, pd.Series([0, 1] * 10), "classification")
    assert {"loss", "macro_f1", "average_precision", "roc_auc"} <= set(result)
    assert all(np.isfinite(value) for value in result.values())


@pytest.mark.parametrize("classes", [[0, 1], ["a", "b", "c"]])
def test_brier_diagnostic_uses_the_same_sum_over_classes_scale(classes):
    class CachedClassifier:
        classes_ = np.asarray(classes)

        def predict_proba(self, X):
            return np.full((len(X), len(classes)), 1.0 / len(classes))

        def predict(self, X):
            return np.full(len(X), classes[0])

    X = pd.DataFrame({"number": range(len(classes) * 2)})
    y = pd.Series(classes * 2)
    result = metrics(CachedClassifier(), X, y, "classification")
    assert result["brier_sum"] == pytest.approx(1.0 - 1.0 / len(classes))


def test_predictive_selection_accepts_cloneable_catboost_without_base_estimator_inheritance():
    pytest.importorskip("catboost")
    from catboost import CatBoostRegressor

    rng = np.random.default_rng(8)
    X = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    y = X.a * X.b
    estimator = CatBoostRegressor(iterations=5, depth=2, thread_count=1, verbose=False, allow_writing_files=False)
    engineer = AutoFeatureEngineer(
        selection_strategy="predictive",
        selection_estimator=estimator,
        task="regression",
        max_candidates=10,
        max_features=5,
        stability_repeats=1,
    )
    result = engineer.fit_transform(X, y)
    assert result.index.equals(X.index)
    assert not estimator.is_fitted()


def test_catboost_numeric_selection_adapter_omits_none_categorical_parameter(tmp_path):
    pytest.importorskip("catboost")
    from unittest.mock import patch

    from sklearn.base import clone

    X = pd.DataFrame({"value": np.arange(40.0), "category": ["a", "b"] * 20})
    y = pd.Series(np.arange(40.0))
    settings = {"selection_learner": "catboost", "seed": 42, "threads": 1, "max_added_features": 2}
    with patch("featcopilot.AutoFeatureEngineer") as constructor:
        constructor.return_value.fit_transform.return_value = X
        constructor.return_value.transform.return_value = X
        constructor.return_value.selection_report_ = None
        generate_features(X, X, y, "featcopilot-predictive", "regression", settings, tmp_path)
    estimator = constructor.call_args.kwargs["selection_estimator"]
    assert "cat_features" not in estimator.get_params()
    assert clone(estimator).get_params() == estimator.get_params()


@pytest.mark.parametrize("learner", ["linear", "lightgbm", "catboost", "xgboost"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize(
    "method",
    ["featcopilot-frequency-paired", "featcopilot-frequency-aligned-robust", "featcopilot-bagged-aligned-robust"],
)
def test_frequency_candidate_integrates_with_each_confirmation_learner(learner, task, method, tmp_path):
    if learner != "linear":
        pytest.importorskip(learner)
    rng = np.random.default_rng(2027)
    train = pd.DataFrame(
        {"number": rng.normal(size=90), "category": ["common", "common", "rare"] * 30},
        index=np.arange(90) + 100,
    )
    test = pd.DataFrame(
        {"number": rng.normal(size=18), "category": ["rare", "unseen", None] * 6},
        index=np.arange(18) + 300,
    )
    y = pd.Series(
        np.arange(90) % 3 if task == "classification" else train.number.to_numpy() ** 2,
        index=train.index,
    )
    settings = {
        "selection_learner": learner,
        "seed": 2027,
        "threads": 1,
        "max_added_features": 3,
        "max_candidates": 20,
        "stability_repeats": 1,
    }
    if method.endswith("-robust"):
        settings["numeric_clip_quantiles"] = (0.0, 1.0)
    output_train, output_test = generate_features(
        train,
        test,
        y,
        method,
        task,
        settings,
        tmp_path,
    )
    prepared_train, prepared_test = prepare_numeric(train, test)
    report = output_train.attrs["selection_report"]
    assert report["frequency_policy"]["operator"] == "fitted_frequency_v1"
    assert report["screening"] == "importance"
    assert report["selection_rule"] == "positive_paired_gain_and_smallest_within_paired_one_se"
    if method == "featcopilot-bagged-aligned-robust":
        assert report["importance_resampling"]["completed_bags"] == 5
    assert output_train.columns.equals(output_test.columns)
    assert output_train.index.equals(train.index)
    assert output_test.index.equals(test.index)
    pd.testing.assert_frame_equal(output_train[train.columns], prepared_train)
    pd.testing.assert_frame_equal(output_test[test.columns], prepared_test)
    assert output_train.shape[1] <= train.shape[1] + settings["max_added_features"]
    model = make_model(output_train, task, learner, candidate_parameters(learner)[1], settings)
    model.fit(output_train, y)
    predictions = model.predict_proba(output_test) if task == "classification" else model.predict(output_test)
    assert np.isfinite(predictions).all()
    if task == "classification":
        assert predictions.shape == (len(test), 3)
        assert ((predictions >= 0) & (predictions <= 1)).all()
        np.testing.assert_allclose(
            predictions.sum(axis=1), 1.0, rtol=0, atol=predictions.shape[1] * np.finfo(predictions.dtype).eps
        )


@pytest.mark.parametrize("learner", ["linear", "lightgbm", "catboost", "xgboost"])
def test_raw_and_engineered_models_fit_identical_caps_for_original_columns(learner):
    if learner != "linear":
        pytest.importorskip(learner)
    train = pd.DataFrame(
        {
            "number": np.r_[np.arange(39, dtype=float), np.nan],
            "category": pd.Categorical(["a", "b"] * 20),
        }
    )
    y = pd.Series(np.arange(40, dtype=float))
    test = pd.DataFrame({"number": [-1000.0, 1000.0, np.nan], "category": ["new", "a", None]})
    parameters = candidate_parameters(learner)[1]
    settings = {"seed": 42, "threads": 1, "numeric_clip_quantiles": (0.0, 1.0)}
    raw = make_model(train, "regression", learner, parameters, settings)
    engineered_train = train.assign(derived=np.arange(40, dtype=float) ** 2)
    engineered = make_model(engineered_train, "regression", learner, parameters, settings)
    raw.fit(train, y)
    engineered.fit(engineered_train, y)
    raw_values = raw.steps[0][1].transform(test)
    engineered_values = engineered.steps[0][1].transform(test.assign(derived=[-1e12, 1e12, np.nan]))
    pd.testing.assert_frame_equal(raw_values, engineered_values[train.columns])
    assert raw_values.number.iloc[:2].tolist() == [0.0, 38.0]
    assert pd.isna(raw_values.number.iloc[2])
    pd.testing.assert_series_equal(raw_values.category, test.category)
    pd.testing.assert_series_equal(train.category, pd.Series(pd.Categorical(["a", "b"] * 20), name="category"))


def test_deferred_imputation_does_not_disguise_all_missing_training_data(tmp_path):
    train = pd.DataFrame({"missing": np.full(40, np.nan)})
    test = pd.DataFrame({"missing": [np.nan, 3.0]}, index=[100, 101])
    y = pd.Series(np.arange(40, dtype=float))
    settings = {
        "selection_learner": "linear",
        "seed": 42,
        "threads": 1,
        "max_added_features": 3,
        "max_candidates": 20,
        "stability_repeats": 1,
    }
    with pytest.raises(ValueError, match="no finite numeric or categorical baseline features"):
        generate_features(train, test, y, "featcopilot-frequency-paired", "regression", settings, tmp_path)

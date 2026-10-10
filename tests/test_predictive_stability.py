"""Numerical parity, nested residual isolation and explicit split-boundary tests."""

import json
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from featcopilot.core.feature import FeatureSet
from featcopilot.engines.predictive import PredictiveTabularEngine
from featcopilot.utils.splitting import make_cv_splits, make_holdout_splits


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=1):
        yield


def signal_data(n=240):
    rng = np.random.default_rng(711)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "linear"])
    y = 4 * X["a"] * X["b"] + 3 * X["linear"] + rng.normal(scale=0.1, size=n)
    return X, y.to_numpy()


def engine(**kwargs):
    options = {"task": "regression", "estimator": Ridge(alpha=0.01), "max_candidates": 12, "max_features": 4}
    options.update(kwargs)
    return PredictiveTabularEngine(**options)


def test_default_stability_screening_and_report_contract():
    X, y = signal_data()
    fitted = engine().fit(X, y)
    report = fitted.selection_report_
    assert fitted.config.stability_repeats == report["stability_repeats"] == 3
    assert fitted.config.screening == "auto"
    assert report["screening"] == "linear"
    assert report["split_kind"] == "iid"
    assert report["ranking_scope"] == "all_fit_rows_first_repeat_candidates"
    assert report["numeric_policy"] == "finite_clip_1e30_nonfinite_to_nan_v1"
    assert report["selected_count"] == 1
    assert fitted.get_feature_names() == ["a_x_b"]
    assert len(report["count_by_repeat"]) == len(report["repeats"]) == 3
    assert report["validation_scores"][0]["count"] == 0
    assert report["baseline_losses"] == [repeat["baseline_loss"] for repeat in report["repeats"]]
    assert report["ranking_scored_rows"] == len(X)
    assert report["ranking_unscored_rows"] == 0
    for score in report["validation_scores"]:
        assert len(score["losses"]) == 3
        assert score["loss"] == pytest.approx(np.mean(score["losses"]))
        assert score["loss_std"] == pytest.approx(np.std(score["losses"], ddof=1))
        assert score["loss_se"] == pytest.approx(np.std(score["losses"], ddof=1) / np.sqrt(3))
    json.dumps(report, allow_nan=False)
    assert report["learner_fit_count"] <= 42


@pytest.mark.parametrize("screening", ["auto", "linear", "tree"])
def test_screening_ablation_modes_remain_deterministic(screening):
    X, y = signal_data()
    first = engine(screening=screening, stability_repeats=1).fit(X, y)
    second = engine(screening=screening, stability_repeats=1).fit(X, y)
    assert first.selection_report_ == second.selection_report_
    assert first.selection_report_["screening"] == ("linear" if screening == "auto" else screening)
    assert first.get_feature_names() == ["a_x_b"]


def test_auto_screening_recognizes_linear_subclasses_and_defaults_to_tree():
    class CustomRidge(Ridge):
        pass

    assert engine(estimator=CustomRidge())._resolve_screening(CustomRidge()) == "linear"
    assert engine()._resolve_screening(LogisticRegression()) == "linear"
    assert engine()._resolve_screening(DummyRegressor()) == "tree"
    default = PredictiveTabularEngine(task="regression")
    assert default._resolve_screening(default._learner("regression")) == "tree"


@pytest.mark.parametrize(
    "kwargs",
    [{"stability_repeats": 0}, {"stability_repeats": 11}, {"screening": "invalid"}, {"max_candidates": 1001}],
)
def test_stability_configuration_has_explicit_compute_bounds(kwargs):
    with pytest.raises(ValueError):
        engine(**kwargs)


def test_one_standard_error_prefers_small_count_and_allows_baseline():
    summarize = PredictiveTabularEngine._summarize_losses
    choose = PredictiveTabularEngine._choose_count
    scores = [
        summarize(0, [1.0, 1.0, 1.0]),
        summarize(1, [0.85, 0.85, 0.85]),
        summarize(5, [0.80, 0.90, 0.80]),
    ]
    assert scores[2]["loss"] < scores[1]["loss"]
    assert choose(scores) == 1
    assert choose([summarize(0, [1.0] * 3), summarize(1, [0.7, 1.1, 0.9])]) == 0
    assert choose([summarize(0, [100.0]), summarize(1, [99.9])]) == 0


def test_no_candidate_budget_still_produces_repeated_originals_only_report():
    X, y = signal_data()
    fitted = engine(max_candidates=0).fit(X, y)
    report = fitted.selection_report_
    assert report["selected_count"] == report["candidate_count"] == report["screened_count"] == 0
    assert report["count_by_repeat"] == [0, 0, 0]
    assert [score["count"] for score in report["validation_scores"]] == [0]
    assert report["learner_fit_count"] == 3
    assert_frame_equal(fitted.transform(X), X)


def test_candidate_and_model_allocations_are_bounded_per_repeat():
    X, y = signal_data()
    with patch.object(PredictiveTabularEngine, "_feature", wraps=PredictiveTabularEngine._feature) as create_feature:
        fitted = engine(max_candidates=7, max_features=9).fit(X, y)
    assert create_feature.call_count == 3 * 7
    assert all(repeat["candidate_count"] <= 7 for repeat in fitted.selection_report_["repeats"])
    assert fitted.selection_report_["learner_fit_count"] <= 42
    assert len(fitted.get_feature_set()) <= 6


def test_default_exact_upper_learner_fit_count_includes_zero_and_fifty():
    rng = np.random.default_rng(490)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
    y = rng.normal(size=len(X))
    fitted = engine(estimator=DummyRegressor(), max_candidates=60, max_features=None).fit(X, y)
    report = fitted.selection_report_
    assert [score["count"] for score in report["validation_scores"]] == [0, 1, 5, 10, 25, 50]
    assert report["learner_fit_count"] == 36
    assert report["ranking_scope"] == "baseline_retained_no_final_refit"


def test_wide_planning_allocates_exactly_the_thousand_candidate_limit():
    rng = np.random.default_rng(541)
    X = pd.DataFrame(rng.normal(size=(24, 2000)), columns=[f"x{i:04d}" for i in range(2000)])
    fitted = engine(max_candidates=1000)
    fitted._input_columns = X.columns.tolist()
    fitted._numeric_columns = X.columns.tolist()
    with patch.object(PredictiveTabularEngine, "_feature", wraps=PredictiveTabularEngine._feature) as factory:
        candidates, numeric_count = fitted._plan_candidates(X, budget=50)
    assert len(candidates) == factory.call_count == 1000
    assert numeric_count == 2000


def test_finite_policy_parity_in_all_generated_surfaces():
    X = pd.DataFrame(
        {
            "a": [1e20, -1e20, 1.0, 0.0, 1.0, 1e300, np.inf, np.nan, -1e-8],
            "b": [1e20, 1e20, 0.0, 0.0, 1e-40, 1e300, 1.0, 2.0, 1e-40],
        },
        index=pd.Index([9, 4, 9, 2, 8, 3, 6, 7, 0], name="custom"),
    )
    fitted = engine()
    features = list(fitted._candidate_stream(["a", "b"], ["a", "b"]))
    fitted._input_columns = X.columns.tolist()
    fitted._numeric_columns = X.columns.tolist()
    fitted._selected_features = features
    fitted._feature_names = [feature.name for feature in features]
    fitted._feature_set = FeatureSet(features)
    fitted._is_fitted = True
    transformed = fitted.transform(X)
    augmented = fitted._augment(fitted._model_frame(X), X, features)
    assert_frame_equal(transformed[X.columns], X)
    assert transformed["a_x_b"].iloc[0] == 1e30
    assert transformed["a_x_b"].iloc[1] == -1e30
    assert transformed["a_div_b"].iloc[4] == 1e30
    assert np.isnan(transformed["a_div_b"].iloc[2])
    assert np.isnan(transformed["a_div_b"].iloc[3])
    assert np.isnan(transformed["a_x_b"].iloc[5])
    assert transformed["a_div_b"].iloc[5] == 1.0
    assert np.isnan(transformed["a_recip"].iloc[-1])
    for feature in features:
        direct = fitted._compute(feature, X)
        np.testing.assert_array_equal(direct, transformed[feature.name])
        np.testing.assert_array_equal(direct, augmented[feature.name])
        np.testing.assert_array_equal(direct, feature.compute(X))
        assert np.nanmax(np.abs(direct)) <= 1e30
    restored = pickle.loads(pickle.dumps(fitted))
    assert_frame_equal(restored.transform(X), transformed)
    partitioned = pd.concat(
        [restored.transform(X.iloc[:2]), restored.transform(X.iloc[2:6]), restored.transform(X.iloc[6:])]
    )
    assert_frame_equal(partitioned, transformed)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_screening_training_residuals_never_depend_on_screening_test_labels(task):
    X, y = signal_data()
    if task == "classification":
        y = (y > np.median(y)).astype(int)
        model = LogisticRegression(max_iter=300)
        classes = 2
    else:
        model = Ridge(alpha=0.01)
        classes = 0
    fitted = engine(task=task, estimator=model, stability_repeats=1)
    prepared = fitted._model_frame(X)
    boundaries = fitted._validate_boundaries(None, None, len(X))
    original_splitter = fitted._cv_splits
    outer = original_splitter(y, task, boundaries, n_splits=3, seed=42)
    train, validation = outer[0]
    changed = y.copy()
    changed[validation] = 1 - changed[validation] if task == "classification" else changed[validation] + 12345

    def fixed_outer(target, requested_task, requested_boundaries, n_splits, seed):
        if len(target) == len(y):
            return outer
        return original_splitter(target, requested_task, requested_boundaries, n_splits, seed)

    with patch.object(fitted, "_cv_splits", side_effect=fixed_outer):
        first, _ = fitted._residual_folds(prepared, y, model, task, classes, boundaries, 42)
        second, _ = fitted._residual_folds(prepared, changed, model, task, classes, boundaries, 42)
    np.testing.assert_array_equal(first[0].train, second[0].train)
    np.testing.assert_array_equal(first[0].train_residuals, second[0].train_residuals)
    assert not np.array_equal(first[0].validation_residuals, second[0].validation_residuals)
    assert set(first[0].train).isdisjoint(validation)


def test_each_repeat_ranking_ignores_its_own_holdout_labels():
    X, y = signal_data()
    train, validation = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
    first = engine().fit(X, y)
    changed = y.copy()
    changed[validation] += 1000
    second = engine().fit(X, changed)
    first_repeat = first.selection_report_["repeats"][0]
    second_repeat = second.selection_report_["repeats"][0]
    assert first_repeat["candidate_ranking"] == second_repeat["candidate_ranking"]
    assert first_repeat["candidate_scores"] == second_repeat["candidate_scores"]
    assert first_repeat["baseline_loss"] != second_repeat["baseline_loss"]


@pytest.mark.parametrize("kind", ["grouped", "temporal"])
def test_every_baseline_prediction_respects_group_or_time_boundaries(kind):
    X, y = signal_data()
    X.index = np.arange(len(X)) * 11 + 5
    values = np.repeat(np.arange(30), 8)
    rng = np.random.default_rng(45)
    permutation = rng.permutation(len(X))
    X, y, values = X.iloc[permutation], y[permutation], values[permutation]
    lookup = pd.Series(values, index=X.index)
    # A deliberately unrelated metadata index verifies positional alignment.
    metadata = pd.Series(values, index=np.arange(len(X))[::-1])
    kwargs = {"groups" if kind == "grouped" else "timestamps": metadata}
    fitted = engine()
    original_fit = fitted._fit_learner
    original_prediction = fitted._prediction
    fit_rows = {}
    checked = []

    def record_fit(frame, target, template):
        model = original_fit(frame, target, template)
        fit_rows[id(model)] = frame.index.to_numpy()
        return model

    def audit_prediction(model, frame, task, n_classes):
        train_rows = fit_rows[id(model)]
        assert set(train_rows).isdisjoint(frame.index)
        training = lookup.loc[train_rows].to_numpy()
        validation = lookup.loc[frame.index].to_numpy()
        if kind == "grouped":
            assert set(training).isdisjoint(validation)
        else:
            assert training.max() < validation.min()
        checked.append((len(training), len(validation)))
        return original_prediction(model, frame, task, n_classes)

    with (
        patch.object(fitted, "_fit_learner", side_effect=record_fit),
        patch.object(fitted, "_prediction", side_effect=audit_prediction),
    ):
        result = fitted.fit_transform(X, y, **kwargs)
    assert checked
    assert_frame_equal(result[X.columns], X)
    report = fitted.selection_report_
    assert report["split_kind"] == kind
    assert report["chronological_prefix_unscored"] == (kind == "temporal")
    assert report["ranking_scored_rows"] + report["ranking_unscored_rows"] == len(X)
    if kind == "temporal":
        assert report["ranking_unscored_rows"] > 0
        assert all(repeat["screening_unscored_rows"] > 0 for repeat in report["repeats"])
    assert_frame_equal(pickle.loads(pickle.dumps(fitted)).transform(X), result)


def test_time_windows_are_disjoint_and_equal_timestamp_blocks_are_atomic():
    X, y = signal_data()
    timestamps = np.repeat(pd.date_range("2020-01-01", periods=30, tz="UTC").to_numpy(), 8)
    fitted = engine()
    boundaries = fitted._validate_boundaries(None, timestamps, len(X))
    partitions = fitted._holdouts(y, "regression", boundaries)
    tested = set()
    for train, validation in partitions:
        assert tested.isdisjoint(validation)
        tested.update(validation)
        assert timestamps[train].max() < timestamps[validation].min()
        for timestamp in np.unique(timestamps[validation]):
            assert set(np.flatnonzero(timestamps == timestamp)) <= set(validation)
    result = fitted.fit_transform(X, y, timestamps=timestamps)
    assert_frame_equal(result, fitted.transform(X))


@pytest.mark.parametrize("kind", ["grouped", "temporal"])
def test_multiclass_group_and_temporal_boundaries_with_valid_class_coverage(kind):
    X, _ = signal_data()
    target = np.tile(["first", "second", "third", "first", "second", "third", "first", "second"], 30)
    kwargs = {"groups" if kind == "grouped" else "timestamps": np.repeat(np.arange(30), 8)}
    fitted = engine(task="classification", estimator=LogisticRegression(max_iter=300)).fit(X, target, **kwargs)
    assert fitted.selection_report_["task"] == "classification"
    assert fitted.selection_report_["split_kind"] == kind
    json.dumps(fitted.selection_report_, allow_nan=False)
    assert_frame_equal(fitted.transform(X)[X.columns], X)


def test_repeat_rankings_are_invariant_to_input_column_permutation():
    X, y = signal_data()
    first = engine().fit(X, y)
    permuted = engine().fit(X[X.columns[::-1]], y)
    assert first.selection_report_ == permuted.selection_report_


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"groups": np.arange(240), "timestamps": np.arange(240)}, "together"),
        ({"groups": np.arange(239)}, "same number"),
        ({"timestamps": np.ones((240, 1))}, "one-dimensional"),
        ({"groups": np.full(240, np.nan)}, "missing"),
        ({"timestamps": np.full(240, np.inf)}, "finite"),
        ({"timestamps": ["invalid"] * 240}, "valid datetimes"),
        ({"groups": np.repeat(np.arange(3), 80)}, "groups"),
        ({"timestamps": np.repeat(np.arange(5), 48)}, "time blocks"),
    ],
)
def test_invalid_boundaries_fail_explicitly_without_random_fallback(kwargs, message):
    X, y = signal_data()
    with pytest.raises(ValueError, match=message):
        engine().fit(X, y, **kwargs)


def test_temporal_class_coverage_never_falls_back_to_random():
    X, _ = signal_data()
    timestamps = np.repeat(np.arange(30), 8)
    labels = np.array(["early"] * 200 + ["late"] * 40)
    with pytest.raises(ValueError, match="every class"):
        engine(task="classification", estimator=LogisticRegression()).fit(X, labels, timestamps=timestamps)


def test_refit_clears_boundary_specific_state():
    X, y = signal_data()
    fitted = engine().fit(X, y, groups=np.repeat(np.arange(30), 8))
    assert fitted.selection_report_["split_kind"] == "grouped"
    fitted.fit(X, y)
    assert fitted.selection_report_["split_kind"] == "iid"
    assert not any(isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)) for value in vars(fitted).values())
    with pytest.raises(ValueError, match="same number"):
        fitted.fit(X, y, timestamps=[1])
    assert fitted.selection_report_ == {}
    assert not fitted.is_fitted


def test_correlated_tail_noise_stress_preserves_raw_fallback():
    rng = np.random.default_rng(918)
    X = pd.DataFrame(rng.normal(size=(480, 3)), columns=["a", "b", "noise"])
    X["duplicate"] = X["a"] + rng.normal(scale=1e-7, size=len(X))
    X["tail"] = rng.standard_t(2, size=len(X))
    y = 3 * X["a"] + rng.normal(scale=0.2, size=len(X))
    fitted = engine(max_candidates=60, max_features=10).fit(X, y)
    json.dumps(fitted.selection_report_, allow_nan=False)
    future = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "noise"])
    future["duplicate"] = future["a"] + rng.normal(scale=1e-7, size=len(future))
    future["tail"] = rng.standard_t(2, size=len(future))
    future_y = 3 * future["a"] + rng.normal(scale=0.2, size=len(future))
    baseline = make_pipeline(StandardScaler(), Ridge(alpha=0.01)).fit(X, y)
    enhanced = make_pipeline(StandardScaler(), Ridge(alpha=0.01)).fit(fitted.transform(X), y)
    assert mean_squared_error(future_y, enhanced.predict(fitted.transform(future))) <= (
        1.1 * mean_squared_error(future_y, baseline.predict(future))
    )


@pytest.mark.parametrize("domain", ["numeric", "datetime"])
def test_label_horizons_reach_every_baseline_and_screening_boundary(domain):
    X, y = signal_data(400)
    X.index = np.arange(len(X)) * 13 + 5
    timestamps = np.repeat(np.arange(40, dtype=float), 10)
    ends = timestamps + 2.5
    ends[0] = 100
    if domain == "datetime":
        timestamps = (pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(timestamps, unit="D")).to_numpy()
        ends = (pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(ends, unit="D")).to_numpy()
    prediction_times = pd.Series(timestamps, index=X.index)
    end_times = pd.Series(ends, index=X.index)
    fitted = engine()
    original_fit = fitted._fit_learner
    original_prediction = fitted._prediction
    original_residual_folds = fitted._residual_folds
    model_rows = {}
    predictions_checked = []
    residuals_checked = []

    def assert_available(train_rows, validation_rows):
        cutoff = prediction_times.loc[validation_rows].min()
        assert prediction_times.loc[train_rows].max() < cutoff
        assert end_times.loc[train_rows].max() < cutoff

    def record_fit(frame, target, template):
        model = original_fit(frame, target, template)
        model_rows[id(model)] = frame.index.to_numpy()
        return model

    def audit_prediction(model, frame, task, n_classes):
        assert_available(model_rows[id(model)], frame.index)
        predictions_checked.append(len(frame))
        return original_prediction(model, frame, task, n_classes)

    def audit_residual_folds(prepared, *args, **kwargs):
        folds, fits = original_residual_folds(prepared, *args, **kwargs)
        for fold in folds:
            assert_available(prepared.index[fold.train], prepared.index[fold.validation])
            residuals_checked.append(len(fold.train))
        return folds, fits

    with (
        patch.object(fitted, "_fit_learner", side_effect=record_fit),
        patch.object(fitted, "_prediction", side_effect=audit_prediction),
        patch.object(fitted, "_residual_folds", side_effect=audit_residual_folds),
        patch("featcopilot.engines.predictive.make_cv_splits", wraps=make_cv_splits) as cv,
        patch("featcopilot.engines.predictive.make_holdout_splits", wraps=make_holdout_splits) as holdouts,
    ):
        # Series labels intentionally differ: all boundary metadata is positional.
        result = fitted.fit_transform(
            X, y, timestamps=pd.Series(timestamps, index=np.arange(len(X))[::-1]), label_end_times=ends
        )
    assert predictions_checked and residuals_checked
    for call in cv.call_args_list + holdouts.call_args_list:
        assert call.kwargs["label_end_times"] is not None
        assert call.kwargs["allow_constant_target"] is False
    report = fitted.selection_report_
    assert report["label_end_times_supplied"] is True
    assert report["label_availability_policy"] == "strictly_before_validation_start"
    assert all(count > 0 for count in report["purged_train_rows_by_repeat"])
    partitions = make_holdout_splits(y, "regression", 3, 42, timestamps=timestamps, label_end_times=ends)
    for repeat, (train, validation) in zip(report["repeats"], partitions, strict=True):
        expected = int(np.count_nonzero(timestamps < timestamps[validation].min())) - len(train)
        assert repeat["purged_train_rows"] == expected
    assert_frame_equal(result[X.columns], X)
    restored = pickle.loads(pickle.dumps(fitted))
    assert_frame_equal(restored.transform(X), result)
    assert_frame_equal(pd.concat([restored.transform(X.iloc[:77]), restored.transform(X.iloc[77:])]), result)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_unavailable_historical_targets_cannot_change_single_holdout_selection(task):
    X, target = signal_data(400)
    y = target if task == "regression" else (target > np.median(target)).astype(int)
    learner = Ridge(alpha=0.01) if task == "regression" else LogisticRegression(max_iter=300)
    timestamps = np.repeat(np.arange(40), 10)
    ends = timestamps + 2
    ends[0] = 100
    train, validation = make_holdout_splits(y, task, 1, 42, timestamps=timestamps, label_end_times=ends)[0]
    cutoff = timestamps[validation].min()
    purged = np.flatnonzero((timestamps < cutoff) & (ends >= cutoff))
    assert len(purged) and not set(purged) & set(train)
    changed = y.copy()
    changed[purged] = changed[purged] + 10000 if task == "regression" else 1 - changed[purged]
    first = engine(task=task, estimator=learner, stability_repeats=1).fit(
        X, y, timestamps=timestamps, label_end_times=ends
    )
    second = engine(task=task, estimator=learner, stability_repeats=1).fit(
        X, changed, timestamps=timestamps, label_end_times=ends
    )
    assert first.selection_report_ == second.selection_report_
    assert_frame_equal(first.transform(X), second.transform(X))


def test_unavailable_inner_labels_cannot_change_past_oof_predictions():
    X, y = signal_data(400)
    timestamps = np.repeat(np.arange(40), 10)
    ends = timestamps + 2
    fitted = engine()
    metadata = fitted._validate_boundaries(None, timestamps, len(X), label_end_times=ends)
    outer_train, _ = fitted._cv_splits(y, "regression", metadata, n_splits=3, seed=42)[0]
    _, inner_validation = fitted._cv_splits(
        y[outer_train], "regression", metadata.subset(outer_train), n_splits=2, seed=43
    )[0]
    cutoff = timestamps[outer_train[inner_validation]].min()
    unavailable = outer_train[(timestamps[outer_train] < cutoff) & (ends[outer_train] >= cutoff)]
    assert len(unavailable)
    changed = y.copy()
    changed[unavailable] += 10000
    prepared = fitted._model_frame(X)
    original_prediction = fitted._prediction
    recorded = []

    def record_prediction(model, frame, task, n_classes):
        prediction = original_prediction(model, frame, task, n_classes)
        recorded.append((frame.index.to_numpy(), prediction.copy()))
        return prediction

    with patch.object(fitted, "_prediction", side_effect=record_prediction):
        fitted._residual_folds(prepared, y, Ridge(alpha=0.01), "regression", 0, metadata, 42)
        before = recorded[0]
        recorded.clear()
        fitted._residual_folds(prepared, changed, Ridge(alpha=0.01), "regression", 0, metadata, 42)
        after = recorded[0]
    np.testing.assert_array_equal(before[0], outer_train[inner_validation])
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_horizon_refit_clears_metadata_and_does_not_change_iid_defaults():
    X, y = signal_data(400)
    timestamps = np.repeat(np.arange(40), 10)
    fitted = engine(stability_repeats=1).fit(X, y, timestamps=timestamps, label_end_times=timestamps + 1)
    assert fitted.selection_report_["label_end_times_supplied"]
    fitted.fit(X, y)
    default = engine(stability_repeats=1).fit(X, y, label_end_times=None)
    assert fitted.selection_report_ == default.selection_report_
    assert fitted.selection_report_["label_availability_policy"] == "not_requested"
    assert fitted.selection_report_["purged_train_rows_by_repeat"] == [0]
    assert not any(isinstance(value, (np.ndarray, pd.Series, pd.DataFrame)) for value in vars(fitted).values())
    with pytest.raises(ValueError, match="requires timestamps"):
        fitted.fit(X, y, label_end_times=timestamps)
    assert not fitted.is_fitted
    assert fitted.selection_report_ == {}


def test_predictive_engine_never_relaxes_constant_or_empty_horizon_training():
    X, y = signal_data()
    with pytest.raises(TypeError, match="Unsupported fit"):
        engine().fit(X, y, allow_constant_target=True)
    with pytest.raises(ValueError, match="constant"):
        engine().fit(X, np.ones(len(X)))
    timestamps = np.repeat(np.arange(30), 8)
    ends = np.full(len(X), 100)
    ends[0] = 0
    with pytest.raises(ValueError, match="constant target or fewer than two"):
        engine().fit(X, y, timestamps=timestamps, label_end_times=ends)
    fitted = engine()
    with patch.object(fitted, "_fit_learner") as fit_learner:
        with pytest.raises(ValueError, match="purging.*no.*training"):
            fitted.fit(X, y, timestamps=timestamps, label_end_times=np.full(len(X), 100))
    fit_learner.assert_not_called()
    assert not fitted.is_fitted
    assert fitted.selection_report_ == {}

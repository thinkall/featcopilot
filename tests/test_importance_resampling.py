"""Fixed five-bag mean-rank procedure tests using bounded synthetic/stub learners."""

import json
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from featcopilot.core.feature import Feature
from featcopilot.engines.predictive import PredictiveTabularEngine, _Ranking
from featcopilot.utils.frequency import fit_frequency_mapping
from featcopilot.utils.numeric_clipping import NumericQuantileClipper
from featcopilot.utils.splitting import make_holdout_splits


@pytest.fixture(autouse=True)
def single_native_thread():
    with threadpool_limits(limits=1):
        yield


class StubRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, zero=False):
        self.zero = zero

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros(X.shape[1]) if self.zero else np.arange(1, X.shape[1] + 1, dtype=float)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class StubClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.prior_ = counts / counts.sum()
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.arange(1, X.shape[1] + 1, dtype=float)
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))

    def predict(self, X):
        return np.full(len(X), self.classes_[np.argmax(self.prior_)])


def bagged(**kwargs):
    options = {
        "importance_resamples": 5,
        "screening": "importance",
        "task": "regression",
        "estimator": StubRegressor(),
        "stability_repeats": 1,
        "max_candidates": 20,
        "max_features": 4,
    }
    options.update(kwargs)
    return PredictiveTabularEngine(**options)


def numeric_data(n=120):
    rng = np.random.default_rng(346)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    return X, np.linspace(-2, 2, n)


@pytest.mark.parametrize("value", [True, False, 0, 2, 3, 4, 6, 5.0, "5", None, np.bool_(True)])
def test_only_strict_one_or_five_is_accepted(value):
    with pytest.raises(ValueError, match="importance_resamples"):
        bagged(importance_resamples=value)


@pytest.mark.parametrize("screening", ["auto", "linear", "tree"])
def test_resampling_requires_explicit_importance(screening):
    with pytest.raises(ValueError, match="explicit screening='importance'"):
        bagged(screening=screening)


@pytest.mark.parametrize(
    "metadata",
    [
        {"groups": np.repeat(np.arange(12), 10)},
        {"timestamps": np.arange(120)},
        {"timestamps": np.arange(120), "label_end_times": np.arange(120) + 1},
        {"label_end_times": np.arange(120)},
    ],
)
def test_metadata_rejected_before_fitting_and_failed_refit_clears_state(metadata):
    X, y = numeric_data()
    engine = bagged().fit(X, y)
    with patch.object(engine, "_fit_learner") as fit:
        with pytest.raises(ValueError, match="IID only"):
            engine.fit(X, y, **metadata)
    fit.assert_not_called()
    assert not engine.is_fitted
    assert engine.selection_report_ == {}


def test_default_one_keeps_full_scope_single_fit_scores_reports_and_metadata_support():
    X, y = numeric_data()
    options = {
        "screening": "importance",
        "estimator": StubRegressor(),
        "task": "regression",
        "stability_repeats": 1,
        "max_candidates": 12,
        "max_features": 4,
    }
    implicit = PredictiveTabularEngine(**options)
    explicit = PredictiveTabularEngine(**options, importance_resamples=1)
    with patch.object(implicit, "_rank_importance_resampled", side_effect=AssertionError("Opt-in only")):
        implicit.fit(X, y)
    explicit.fit(X, y)
    assert implicit.selection_report_ == explicit.selection_report_
    assert "importance_resampling" not in explicit.selection_report_
    assert "importance_resampling_selection" not in explicit.selection_report_
    assert all("importance_resampling" not in repeat for repeat in explicit.selection_report_["repeats"])
    assert_frame_equal(implicit.transform(X), explicit.transform(X))
    assert explicit.selection_report_["importance_ranking"]["learner_fits"] == 1
    explicit.fit(X, y, groups=np.repeat(np.arange(12), 10))
    assert explicit.selection_report_["split_kind"] == "grouped"


def test_mean_ranks_average_ties_instead_of_importance_magnitudes():
    X, y = numeric_data()
    engine = bagged()
    definitions = [Feature(name=name) for name in ["b", "d", "a", "c"]]
    values = [(100, 99), (1, 2), (1000, 1), (0.1, 0.2), (1, 1)]
    calls = []

    def one_bag(candidates, raw, prepared, target, template, *, fit_empty=False):
        assert fit_empty
        assert [feature.name for feature in candidates] == ["b", "d", "a", "c"]
        left, right = values[len(calls)]
        calls.append(raw.index.tolist())
        return _Ranking(
            [feature for feature in candidates if feature.name != "d"],
            {"a": left, "b": right, "c": 0.0},
            learner_fits=1,
            fit_rows=len(raw),
            discarded=[{"name": "d", "reason": "constant", "duplicate_of": None, "duplicate_kind": None}],
        )

    with patch.object(engine, "_rank_importance_single", side_effect=one_bag):
        result = engine._rank_importance_resampled(
            definitions, X, engine._model_frame(X), y, StubRegressor(), task="regression", seed=42
        )
    assert [feature.name for feature in result.features] == ["a", "b"]
    assert result.scores == {"a": -1.5, "b": -1.5}
    assert result.resampling["mean_ranks"] == {"a": 1.5, "b": 1.5, "c": 3.5, "d": 3.5}
    assert result.resampling["positive_counts"] == {"a": 5, "b": 5, "c": 0, "d": 0}
    assert result.resampling["never_positive_features"] == ["c", "d"]
    assert result.learner_fits == 5
    assert result.fit_rows == 5 * 102
    assert all(len(indices) == len(set(indices)) == 102 for indices in calls)
    np.testing.assert_array_equal(
        engine._candidate_average_ranks(np.array([-0.0, 0.0, -2.0, 3.0, 3.0])),
        [4.0, 4.0, 4.0, 1.5, 1.5],
    )
    with pytest.raises(ValueError, match="finite"):
        engine._candidate_average_ranks(np.array([1.0, np.nan]))


def test_one_positive_bag_is_eligible_without_a_majority_support_cutoff():
    X, y = numeric_data()
    engine = bagged()
    candidates = [Feature(name="rare"), Feature(name="common"), Feature(name="never")]
    count = 0

    def one_bag(candidates, raw, prepared, target, template, **kwargs):
        nonlocal count
        scores = {"rare": 2.0 if count == 0 else 0.0, "common": 1.0, "never": 0.0}
        count += 1
        return _Ranking(candidates, scores, learner_fits=1, fit_rows=len(raw))

    with patch.object(engine, "_rank_importance_single", side_effect=one_bag):
        ranked = engine._rank_importance_resampled(
            candidates, X, engine._model_frame(X), y, StubRegressor(), task="regression", seed=42
        )
    assert {feature.name for feature in ranked.features} == {"rare", "common"}
    assert ranked.resampling["positive_counts"] == {"common": 5, "never": 0, "rare": 1}


def test_bag_local_nan_constants_and_signed_zero_duplicates_map_back_without_refill():
    X = pd.DataFrame({"original": np.tile([0.0, -0.0, 1.0, np.nan, -2.0], 24)})
    y = np.arange(len(X), dtype=float)
    engine = bagged()
    candidates = [Feature(name=name) for name in ["alias", "all_nan", "novel", "copy_novel"]]

    def values(feature, frame):
        original = frame["original"].to_numpy()
        novel = np.sin(frame.index.to_numpy(dtype=float))
        return {
            "alias": original + 0.0,
            "all_nan": np.full(len(frame), np.nan),
            "novel": novel,
            "copy_novel": novel.copy(),
        }[feature.name]

    with (
        patch.object(engine, "_compute", side_effect=values),
        patch.object(engine, "_fit_learner", wraps=engine._fit_learner) as fits,
    ):
        ranked = engine._rank_importance_resampled(
            candidates, X, engine._model_frame(X), y, StubRegressor(), task="regression", seed=42
        )
    assert fits.call_count == ranked.learner_fits == 5
    assert [feature.name for feature in ranked.features] == ["novel"]
    for report in ranked.resampling["bags"]:
        assert report["constant_count"] == 1
        assert report["duplicate_count"] == 2
        assert report["retained_count"] == 1
    assert ranked.resampling["positive_counts"] == {"alias": 0, "all_nan": 0, "copy_novel": 0, "novel": 5}
    assert all("original" in call.args[0].columns for call in fits.call_args_list)


def test_bag_local_unusable_features_still_fit_original_context_and_are_counted():
    X = pd.DataFrame({"original_category": ["a", "b"] * 60})
    y = np.arange(len(X), dtype=float)
    engine = bagged()
    with (
        patch.object(engine, "_compute", side_effect=lambda feature, frame: (frame.index == 0).astype(float)),
        patch.object(engine, "_fit_learner", wraps=engine._fit_learner) as fit,
        patch(
            "featcopilot.engines.predictive.make_holdout_splits",
            return_value=[(np.arange(18, 120), np.arange(18))] * 5,
        ),
    ):
        ranked = engine._rank_importance([Feature(name="unusable")], X, engine._model_frame(X), y, StubRegressor())
    assert fit.call_count == ranked.learner_fits == 5
    assert ranked.features == []
    assert ranked.resampling["positive_counts"] == {"unusable": 0}
    assert all(call.args[0].columns.tolist() == ["original_category"] for call in fit.call_args_list)


def test_scope_deduplication_precedes_bags_and_empty_scope_needs_no_ranking_fits():
    X, y = numeric_data()
    engine = bagged()
    candidates = [Feature(name="same"), Feature(name="empty")]
    with (
        patch.object(
            engine,
            "_compute",
            side_effect=lambda feature, frame: (
                frame.a.to_numpy() if feature.name == "same" else np.full(len(frame), np.nan)
            ),
        ),
        patch.object(engine, "_fit_learner") as fit,
    ):
        ranked = engine._rank_importance(candidates, X, engine._model_frame(X), y, StubRegressor())
    fit.assert_not_called()
    assert ranked.features == []
    assert ranked.learner_fits == 0
    assert ranked.resampling["universe_count"] == ranked.resampling["completed_bags"] == 0
    assert {item["name"]: item["reason"] for item in ranked.discarded} == {"same": "duplicate", "empty": "constant"}


def test_duplicate_masks_are_reported_without_retries_and_seed_wrap_is_deterministic():
    X, y = numeric_data()
    engine = bagged(random_state=2**32 - 2)
    candidates = [engine._feature("a_sq", ["a"], "square", "x**2")]
    first = engine._rank_importance(candidates, X, engine._model_frame(X), y, StubRegressor())
    repeated = engine._rank_importance(candidates, X, engine._model_frame(X), y, StubRegressor())
    assert first.resampling == repeated.resampling
    assert [bag["seed"] for bag in first.resampling["bags"]] == [2**32 - 2, 2**32 - 1, 0, 1, 2]
    changed = engine._rank_importance(candidates, X, engine._model_frame(X), y, StubRegressor(), seed=23)
    assert [bag["position_mask_sha256"] for bag in first.resampling["bags"]] != [
        bag["position_mask_sha256"] for bag in changed.resampling["bags"]
    ]
    train, excluded = make_holdout_splits(y, "regression", 1, 12, test_size=0.15)[0]
    with patch("featcopilot.engines.predictive.make_holdout_splits", return_value=[(train, excluded)] * 5) as split:
        duplicated = engine._rank_importance(candidates, X, engine._model_frame(X), y, StubRegressor())
    split.assert_called_once()
    assert duplicated.resampling["duplicate_bag_mask_count"] == 4
    assert duplicated.learner_fits == 5


def test_invalid_bag_splits_and_missing_classes_raise_without_retry():
    X, _ = numeric_data(60)
    engine = bagged(task="classification", estimator=StubClassifier())
    candidate = engine._feature("a_sq", ["a"], "square", "x**2")
    labels = np.array([0] * 58 + [1, 2])
    with patch.object(engine, "_fit_learner") as fit:
        with pytest.raises(ValueError, match="holdout split"):
            engine._rank_importance(
                [candidate], X, engine._model_frame(X), labels, StubClassifier(), task="classification"
            )
    fit.assert_not_called()
    with patch("featcopilot.engines.predictive.make_holdout_splits", side_effect=ValueError("infeasible bag")) as split:
        with pytest.raises(ValueError, match="infeasible bag"):
            engine._rank_importance(
                [candidate], X, engine._model_frame(X), np.arange(len(X)), StubRegressor(), task="regression"
            )
    split.assert_called_once()


def test_constant_target_regression_bags_are_fitted_and_keep_zero_importance():
    X, _ = numeric_data(60)
    excluded = make_holdout_splits(np.arange(60), "regression", 1, 42, test_size=0.15)[0][1]
    y = np.zeros(len(X))
    y[excluded[0]] = 1.0
    engine = bagged(importance_resamples=np.int64(5))
    candidate = engine._feature("a_sq", ["a"], "square", "x**2")
    observed = []
    original = engine._rank_importance_single

    def one_bag(candidates, raw, prepared, target, template, **kwargs):
        constant = np.ptp(target) == 0
        observed.append(constant)
        return original(candidates, raw, prepared, target, StubRegressor(zero=constant), **kwargs)

    with patch.object(engine, "_rank_importance_single", side_effect=one_bag):
        ranked = engine._rank_importance([candidate], X, engine._model_frame(X), y, StubRegressor())
    assert observed[0]
    assert len(observed) == ranked.learner_fits == 5
    assert ranked.resampling["positive_counts"]["a_sq"] == 5 - sum(observed)


@pytest.mark.parametrize("repeats", [1, 3])
def test_positive_count_uses_same_five_bag_algorithm_for_final_fit_and_actual_prefixes(repeats):
    X, y = numeric_data()
    engine = bagged(stability_repeats=repeats, max_candidates=8)
    calls = []
    augmentations = []
    original_rank = engine._rank_importance_resampled
    original_augment = engine._augment

    def rank(*args, **kwargs):
        result = original_rank(*args, **kwargs)
        calls.append((set(args[1].index), result))
        return result

    def augment(prepared, raw, features):
        augmentations.append((set(raw.index), [feature.name for feature in features]))
        return original_augment(prepared, raw, features)

    with (
        patch.object(engine, "_rank_importance_resampled", side_effect=rank),
        patch.object(engine, "_augment", side_effect=augment),
        patch.object(engine, "_loss", side_effect=lambda model, frame, *args: 100.0 - (frame.shape[1] - X.shape[1])),
    ):
        result = engine.fit_transform(X, y)
    assert engine.selection_report_["selected_count"] == 1
    assert len(calls) == repeats + 1
    assert calls[-1][0] == set(X.index)
    assert all(ranking.learner_fits == 5 for _, ranking in calls)
    assert all(ranking.resampling["aggregation"] == "mean_candidate_rank" for _, ranking in calls)
    assert engine.selection_report_["learner_fit_count"] == repeats * 7 + 5
    assert engine.selection_report_["importance_resampling_selection"]["final_ranking_performed"]
    for rows, ranking in calls[:repeats]:
        assert (rows, [ranking.features[0].name]) in augmentations
    assert engine.selection_report_["ranking_scope"] == "all_fit_rows_resampled_importance"
    assert engine.get_feature_names() == [calls[-1][1].features[0].name]
    assert_frame_equal(result[X.columns], X)
    restored = pickle.loads(pickle.dumps(engine))
    assert_frame_equal(pd.concat([restored.transform(X.iloc[:37]), restored.transform(X.iloc[37:])]), result)
    for feature in engine.get_feature_set():
        np.testing.assert_array_equal(feature.compute(X), result[feature.name])


def test_zero_additions_skip_final_ranking_and_zero_budget_skips_all_bag_models():
    X, y = numeric_data()
    engine = bagged(estimator=StubRegressor(zero=True), stability_repeats=3)
    with patch.object(engine, "_rank_importance_resampled", wraps=engine._rank_importance_resampled) as rank:
        result = engine.fit_transform(X, y)
    assert rank.call_count == 3
    assert engine.selection_report_["learner_fit_count"] == 3 * (5 + 1)
    assert engine.selection_report_["selected_count"] == 0
    assert not engine.selection_report_["importance_resampling_selection"]["final_ranking_performed"]
    assert_frame_equal(result, X)
    empty = bagged(max_candidates=0, stability_repeats=3).fit(X, y)
    assert empty.selection_report_["learner_fit_count"] == 3
    assert empty.selection_report_["importance_resampling"]["completed_bags"] == 0
    assert empty.selection_report_["importance_resampling"]["universe_count"] == 0


def test_holdout_labels_and_values_never_reach_bag_fits_or_change_repeat_ranking():
    X, y = numeric_data()
    X["category"] = ["train_a", "train_b"] * 60
    train, holdout = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
    changed = X.copy()
    changed.loc[holdout, "a"] = 1e20
    changed.loc[holdout, "category"] = "holdout_only"
    modified_y = y.copy()
    modified_y[holdout] = 123456.0
    results = []
    for frame, target in [(X, y), (changed, modified_y)]:
        engine = bagged(max_features=5)
        original_fit = engine._fit_learner

        def guarded_fit(values, labels, template, original_fit=original_fit, **kwargs):
            assert set(values.index) <= set(train)
            assert not np.any(labels == 123456.0)
            return original_fit(values, labels, template, **kwargs)

        with (
            patch.object(engine, "_fit_learner", side_effect=guarded_fit),
            patch.object(engine, "_loss", return_value=100.0),
        ):
            engine.fit(frame, target)
        results.append(engine.selection_report_["repeats"][0])
    assert results[0]["candidate_ranking"] == results[1]["candidate_ranking"]
    assert results[0]["candidate_scores"] == results[1]["candidate_scores"]
    assert results[0]["importance_resampling"] == results[1]["importance_resampling"]


def test_bag_frequency_clipper_imputer_and_encoder_fit_only_the_current_bag():
    X, y = numeric_data()
    X["category"] = ["private_a", "private_a", "private_b"] * 40
    X.loc[::11, "a"] = np.nan
    engine = bagged(include_frequency=True, numeric_clip_quantiles=(0.0, 1.0), max_candidates=50, max_features=6)
    original_single = engine._rank_importance_single
    original_clip = NumericQuantileClipper.fit
    original_imputer = SimpleImputer.fit
    contexts = {"rows": None}
    seen = []
    snapshots = []

    def single(candidates, raw, prepared, target, template, **kwargs):
        contexts["rows"] = set(raw.index)
        for feature in candidates:
            if feature.transformation == "fitted_frequency_v1":
                state = feature.metadata["frequency_state"]
                assert state["fit_rows"] == len(raw)
                assert state["scope"] == "internal_train"
                assert state["mapping"] == fit_frequency_mapping(raw[feature.source_columns[0]])
                seen.append(feature.name)
        result = original_single(candidates, raw, prepared, target, template, **kwargs)
        contexts["rows"] = None
        return result

    def clip(estimator, frame, target=None):
        if contexts["rows"] is not None:
            assert set(frame.index) == contexts["rows"]
        return original_clip(estimator, frame, target)

    def imputer(estimator, frame, target=None, **kwargs):
        if contexts["rows"] is not None:
            assert set(frame.index) == contexts["rows"]
        return original_imputer(estimator, frame, target, **kwargs)

    original_rank = engine._rank_importance_resampled

    def rank(candidates, *args, **kwargs):
        before = [pickle.dumps(feature.metadata) for feature in candidates]
        result = original_rank(candidates, *args, **kwargs)
        assert [pickle.dumps(feature.metadata) for feature in candidates] == before
        snapshots.append(before)
        return result

    with (
        patch.object(engine, "_rank_importance_single", side_effect=single),
        patch.object(engine, "_rank_importance_resampled", side_effect=rank),
        patch.object(NumericQuantileClipper, "fit", clip),
        patch.object(SimpleImputer, "fit", imputer),
        patch.object(engine, "_loss", return_value=100.0),
    ):
        engine.fit(X, y)
    assert seen and snapshots
    assert "private_" not in json.dumps(engine.selection_report_, allow_nan=False)


@pytest.mark.parametrize("repeats", [1, 3])
def test_selected_frequency_keeps_full_fit_state_not_a_final_bag_mapping(repeats):
    X = pd.DataFrame(
        {
            "left": ["a", "a", "a", "b", "b", "c"] * 20,
            "right": ["u", "v", "u", "u"] * 30,
        }
    )
    y = np.arange(len(X), dtype=float)
    engine = bagged(include_frequency=True, stability_repeats=repeats, max_features=3)
    with patch.object(engine, "_loss", side_effect=lambda model, frame, *args: 100 - (frame.shape[1] - X.shape[1])):
        result = engine.fit_transform(X, y)
    assert engine.selection_report_["selected_count"] == 1
    feature = engine.get_feature_set()[engine.get_feature_names()[0]]
    assert feature.transformation == "fitted_frequency_v1"
    state = feature.metadata["frequency_state"]
    assert state["scope"] == "final_fit"
    assert state["fit_rows"] == len(X)
    assert state["mapping"] == fit_frequency_mapping(X[feature.source_columns[0]])
    assert engine.selection_report_["frequency_policy"]["final_refit_count"] == 2
    assert engine.selection_report_["importance_resampling"]["scope_rows"] == len(X)
    np.testing.assert_array_equal(feature.compute(X), result[feature.name])


def test_raw_thousand_candidate_budget_is_not_refilled_by_bag_filtering():
    X = pd.DataFrame({f"x{i:03d}": np.arange(1, 61, dtype=float) for i in range(100)})
    y = np.arange(len(X), dtype=float)
    engine = bagged(max_candidates=1000, max_features=None, candidate_set="extended")
    with (
        patch.object(PredictiveTabularEngine, "_feature", wraps=PredictiveTabularEngine._feature) as factory,
        patch.object(engine, "_loss", return_value=100.0),
    ):
        result = engine.fit_transform(X, y)
    assert factory.call_count == engine.selection_report_["candidate_count"] == 1000
    resampling = engine.selection_report_["importance_resampling"]
    assert 0 < resampling["universe_count"] < 100
    assert resampling["completed_bags"] == 5
    assert len(resampling["positive_counts"]) == resampling["universe_count"]
    assert len(engine.selection_report_["importance_ranking"]["discarded_candidates"]) > 900
    assert engine.selection_report_["selected_count"] == 0
    assert_frame_equal(result, X)


def test_stratified_multiclass_bags_and_caller_clone_are_preserved():
    X, _ = numeric_data()
    y = np.array([0] * 60 + [1] * 36 + [2] * 24)
    supplied = StubClassifier().fit(X, y)
    before = pickle.dumps(supplied)
    engine = bagged(task="classification", estimator=supplied)
    original_single = engine._rank_importance_single
    counts = []

    def single(candidates, raw, prepared, target, template, **kwargs):
        assert set(target) == {0, 1, 2}
        counts.append(len(target))
        return original_single(candidates, raw, prepared, target, template, **kwargs)

    with (
        patch.object(engine, "_rank_importance_single", side_effect=single),
        patch.object(engine, "_loss", return_value=100.0),
    ):
        engine.fit(X, y)
    assert counts == [76] * 5
    assert pickle.dumps(supplied) == before
    assert engine.config.estimator is supplied


def test_final_rank_eligibility_shrink_is_reported_without_postgate_consensus_or_retry():
    X, y = numeric_data()
    engine = bagged(max_candidates=2)
    original = engine._rank_importance_single

    def single(candidates, raw, prepared, target, template, **kwargs):
        template = StubRegressor(zero=len(raw) > 80)
        return original(candidates, raw, prepared, target, template, **kwargs)

    with (
        patch.object(engine, "_rank_importance_single", side_effect=single) as single_fit,
        patch.object(engine, "_loss", side_effect=lambda model, frame, *args: 100 - (frame.shape[1] - X.shape[1])),
    ):
        result = engine.fit_transform(X, y)
    assert single_fit.call_count == 10
    selection = engine.selection_report_["importance_resampling_selection"]
    assert selection == {
        "validated_selected_count": 1,
        "delivered_count": 0,
        "final_ranking_performed": True,
        "procedure_level_validation_only": True,
    }
    assert engine.selection_report_["ranking_scope"] == "baseline_retained_after_resampled_ranking"
    assert_frame_equal(result, X)

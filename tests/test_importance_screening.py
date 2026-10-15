"""Joint TRAIN-only importance, exact deduplication and encoded attribution."""

import json
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from featcopilot.core.feature import Feature
from featcopilot.engines.predictive import PredictiveTabularEngine


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=1):
        yield


def importance_engine(**kwargs):
    options = {
        "screening": "importance",
        "task": "regression",
        "estimator": Ridge(alpha=0.01),
        "stability_repeats": 1,
        "max_candidates": 32,
    }
    options.update(kwargs)
    return PredictiveTabularEngine(**options)


class IndexedImportanceRegressor(RegressorMixin, BaseEstimator):
    """Expose known per-encoded-position weights, independently of predictions."""

    def __init__(self, mode="native"):
        self.mode = mode

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.mean_ = float(np.mean(y))
        axis = np.arange(1, X.shape[1] + 1, dtype=float)
        if self.mode == "native":
            self.feature_importances_ = axis
        elif self.mode == "coef_multi":
            self.coef_ = np.vstack([axis, -2 * axis, 0.5 * axis])
        elif self.mode == "coef_vector":
            self.coef_ = axis * np.where(np.arange(len(axis)) % 2, -1, 1)
        elif self.mode == "coef_zero":
            self.coef_ = np.zeros_like(axis)
        elif self.mode == "negative":
            self.feature_importances_ = -axis
        elif self.mode == "nan":
            self.feature_importances_ = np.full_like(axis, np.nan)
        elif self.mode == "wrong_width":
            self.feature_importances_ = axis[:-1]
        elif self.mode == "native_matrix":
            self.feature_importances_ = axis.reshape(1, -1)
        elif self.mode == "complex":
            self.feature_importances_ = axis + 1j
        elif self.mode == "empty_coef":
            self.coef_ = np.empty((0, len(axis)))
        elif self.mode == "text":
            self.feature_importances_ = np.array(["unknown"] * len(axis))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def mixed_data():
    X = pd.DataFrame(
        {
            "all_missing": [np.nan] * 80,
            "category": ["candidate_a", "candidate_b", None, "numeric__candidate_a"] * 20,
            "original": np.linspace(0.1, 2.0, 80),
        },
        index=pd.Index(np.arange(80) * 9 + 5, name="sample"),
    )
    y = 2 * X["original"].to_numpy() ** 2 + np.sin(np.arange(len(X)))
    return X, y


def controlled_candidates():
    return [Feature(name="candidate_a"), Feature(name="candidate_b")]


@pytest.mark.parametrize("mode", ["native", "coef_vector", "coef_multi", "coef_zero"])
def test_encoded_column_attribution_uses_numeric_block_not_onehot_names(mode):
    X, y = mixed_data()
    engine = importance_engine(estimator=IndexedImportanceRegressor(mode))
    candidates = controlled_candidates()
    arrays = {
        "candidate_a": np.sin(np.arange(len(X))),
        "candidate_b": np.cos(np.arange(len(X))),
    }
    prepared = engine._model_frame(X)
    with (
        patch.object(engine, "_compute", side_effect=lambda feature, frame: arrays[feature.name]),
        patch.object(engine, "_fit_learner", wraps=engine._fit_learner) as fit,
    ):
        ranking = engine._rank_importance(candidates, X, prepared, y, engine._learner("regression"))
    fit.assert_called_once()
    assert ranking.learner_fits == 1
    assert ranking.fit_rows == len(X)
    # all_missing is dropped. Numeric output is original, candidate_a,
    # candidate_b, followed by four one-hot outputs whose names also match
    # candidate strings. Those later, larger weights must not be attributed.
    if mode == "native":
        assert ranking.scores == {"candidate_b": 3.0, "candidate_a": 2.0}
        assert ranking.importance_source == "feature_importances_"
    elif mode == "coef_zero":
        assert ranking.scores == {"candidate_a": 0.0, "candidate_b": 0.0}
    else:
        denominator = sum(range(1, 8))
        assert ranking.scores["candidate_a"] == pytest.approx(2 / denominator)
        assert ranking.scores["candidate_b"] == pytest.approx(3 / denominator)
        assert ranking.importance_source == "normalized_abs_coef"
    assert ranking.folds == ranking.scored_rows == 0
    assert ranking.unscored_rows == len(X)
    assert_frame_equal(X, mixed_data()[0])


def test_delivered_importance_explanation_does_not_claim_residual_screening():
    X = pd.DataFrame({"x": np.linspace(-2, 2, 128)})
    y = X["x"].to_numpy() ** 2
    engine = importance_engine(max_candidates=1, max_features=2).fit(X, y)
    assert engine.selection_report_["screening"] == "importance"
    features = list(engine.get_feature_set())
    assert len(features) == 1
    assert "internal screening and holdout loss" in features[0].explanation
    assert "residual" not in features[0].explanation


def test_exact_dedup_handles_signed_zero_nan_masks_and_collision_verification():
    X = pd.DataFrame({"a": [0.0, -0.0, 1.0, np.nan, -2.0], "b": [7, 8, 9, 10, 11]})
    engine = importance_engine()
    prepared = engine._model_frame(X)
    before = prepared["a"].to_numpy().copy()
    original = X["a"].to_numpy()
    duplicate = original.copy()
    duplicate[:2] = [-0.0, 0.0]
    duplicate.view(np.uint64)[3] = np.uint64(0x7FF8000000000042)
    novel = np.array([0.0, 1.0, 2.0, np.nan, 3.0])
    almost = original.copy()
    almost[2] += 1e-12
    arrays = {
        "original_alias": duplicate,
        "constant": np.array([0.0, -0.0, np.nan, 0.0, 0.0]),
        "all_nan": np.full(5, np.nan),
        "novel": novel,
        "earlier_alias": novel.copy(),
        "almost": almost,
        "almost_constant": np.array([1.0, 1.0, 1.0, 1.0, 1.0 + 1e-12]),
        "negated": -original,
        "different_missing": np.array([0.0, -0.0, np.nan, 1.0, -2.0]),
        "reversed_rows": original[::-1].copy(),
    }
    candidates = [Feature(name=name) for name in arrays]
    assert engine._numeric_fingerprint(original) == engine._numeric_fingerprint(duplicate)
    with (
        patch.object(engine, "_compute", side_effect=lambda feature, frame: arrays[feature.name]),
        patch.object(engine, "_numeric_fingerprint", return_value=b"forced collision"),
    ):
        retained, matrix, discarded = engine._deduplicate_candidates(candidates, X, prepared)
    assert [feature.name for feature in retained] == [
        "novel",
        "almost",
        "almost_constant",
        "negated",
        "different_missing",
        "reversed_rows",
    ]
    records = {item["name"]: item for item in discarded}
    assert records["original_alias"] == {
        "name": "original_alias",
        "reason": "duplicate",
        "duplicate_of": "a",
        "duplicate_kind": "original",
    }
    assert records["earlier_alias"]["duplicate_of"] == "novel"
    assert records["earlier_alias"]["duplicate_kind"] == "candidate"
    assert records["constant"]["reason"] == records["all_nan"]["reason"] == "constant"
    assert set(matrix) == {feature.name for feature in retained}
    np.testing.assert_array_equal(np.signbit(prepared["a"].to_numpy()), np.signbit(before))
    np.testing.assert_array_equal(prepared["a"], before)


def test_original_duplicate_comparison_uses_the_same_finite_policy():
    X = pd.DataFrame({"a": [np.nan, np.inf, -np.inf, 1e50, -1e50, -0.0]})
    engine = importance_engine()
    prepared = engine._model_frame(X)
    with patch.object(engine, "_compute", return_value=prepared["a"].to_numpy().copy()):
        retained, _, discarded = engine._deduplicate_candidates([Feature(name="finite_alias")], X, prepared)
    assert retained == []
    assert discarded[0]["duplicate_of"] == "a"
    assert discarded[0]["duplicate_kind"] == "original"
    assert np.isposinf(X["a"].iloc[1])
    assert np.isneginf(X["a"].iloc[2])
    assert X["a"].iloc[3] == 1e50


def test_degenerate_generated_pool_skips_joint_fit_without_replacing_originals():
    X = pd.DataFrame({"a": np.arange(10, dtype=float)})
    engine = importance_engine()
    candidates = [Feature(name="alias"), Feature(name="constant")]
    with (
        patch.object(
            engine,
            "_compute",
            side_effect=lambda feature, frame: X["a"].to_numpy() if feature.name == "alias" else np.zeros(10),
        ),
        patch.object(engine, "_fit_learner") as fit,
    ):
        ranking = engine._rank_importance(candidates, X, engine._model_frame(X), np.arange(10), Ridge())
    fit.assert_not_called()
    assert ranking.features == []
    assert ranking.learner_fits == ranking.fit_rows == 0
    assert len(ranking.discarded) == 2


def test_raw_candidate_budget_does_not_refill_after_exact_pruning():
    X = pd.DataFrame({f"x{i:03d}": np.arange(1, 81, dtype=float) for i in range(200)})
    y = np.random.default_rng(451).normal(size=len(X))
    engine = importance_engine(
        estimator=IndexedImportanceRegressor(), candidate_set="extended", max_candidates=1000, max_features=205
    )
    with patch.object(PredictiveTabularEngine, "_feature", wraps=PredictiveTabularEngine._feature) as factory:
        engine.fit(X, y)
    report = engine.selection_report_
    details = report["importance_ranking"]
    assert report["candidate_count"] == details["planned_count"] == factory.call_count == 1000
    assert 0 < details["retained_count"] < 20
    assert details["discarded_constant_count"] + details["discarded_duplicate_count"] > 980
    assert details["retained_count"] + len(details["discarded_candidates"]) == details["planned_count"]
    assert details["learner_fits"] == 1
    assert report["selected_count"] == 0
    assert report["ranking_scope"] == "baseline_retained_no_final_refit"
    assert_frame_equal(engine.transform(X), X)


def test_actual_joint_tree_sum_gain_on_an_independent_oblique_population():
    rng = np.random.default_rng(4512)
    X = pd.DataFrame(rng.uniform(-2, 2, size=(600, 2)), columns=["a", "b"])
    y = (X["a"] + X["b"] > 0).astype(int)
    source = DecisionTreeClassifier(max_depth=1, random_state=42)
    fitted = importance_engine(task="classification", estimator=source, candidate_set="extended", max_features=3).fit(
        X, y
    )
    assert fitted.get_feature_names() == ["a_plus_b"]
    assert not hasattr(source, "tree_")
    future_rng = np.random.default_rng(887)
    future = pd.DataFrame(future_rng.uniform(-2, 2, size=(500, 2)), columns=X.columns)
    future_y = (future["a"] + future["b"] > 0).astype(int)
    raw = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y)
    engineered = DecisionTreeClassifier(max_depth=1, random_state=42).fit(fitted.transform(X), y)
    baseline_accuracy = accuracy_score(future_y, raw.predict(future))
    improved_accuracy = accuracy_score(future_y, engineered.predict(fitted.transform(future)))
    assert improved_accuracy >= 0.97
    assert improved_accuracy - baseline_accuracy >= 0.12


def test_actual_joint_logistic_xor_gain_and_frozen_artifact_parity():
    rng = np.random.default_rng(683)
    flags = rng.integers(0, 2, size=(400, 2))
    X = pd.DataFrame({"left": np.where(flags[:, 0], "yes", "no"), "right": np.where(flags[:, 1], "on", "off")})
    X.index = pd.Index(np.arange(len(X)) * 3 + 1, name="custom")
    y = (flags[:, 0] != flags[:, 1]).astype(int)
    supplied = LogisticRegression(max_iter=500)
    fitted = importance_engine(task="classification", estimator=supplied, candidate_set="extended", max_features=3).fit(
        X, y
    )
    assert len(fitted.get_feature_names()) == 1
    assert fitted.get_feature_set()[fitted.get_feature_names()[0]].transformation == "category_pair"
    assert fitted.selection_report_["importance_ranking"]["importance_source"] == "normalized_abs_coef"
    future_flags = np.random.default_rng(951).integers(0, 2, size=(500, 2))
    future = pd.DataFrame(
        {
            "left": np.where(future_flags[:, 0], "yes", "no"),
            "right": np.where(future_flags[:, 1], "on", "off"),
        }
    )
    future_y = (future_flags[:, 0] != future_flags[:, 1]).astype(int)
    raw = fitted._fit_learner(fitted._model_frame(X), y, LogisticRegression(max_iter=500))
    enhanced = fitted._fit_learner(fitted._model_frame(fitted.transform(X)), y, LogisticRegression(max_iter=500))
    assert log_loss(future_y, enhanced.predict_proba(fitted._model_frame(fitted.transform(future)))) < (
        0.5 * log_loss(future_y, raw.predict_proba(fitted._model_frame(future)))
    )
    output = fitted.transform(X)
    assert_frame_equal(output[X.columns], X)
    restored = pickle.loads(pickle.dumps(fitted))
    assert_frame_equal(pd.concat([restored.transform(X.iloc[:71]), restored.transform(X.iloc[71:])]), output)
    assert_frame_equal(restored.transform(X[X.columns[::-1]]), output)
    for feature in fitted.get_feature_set():
        np.testing.assert_array_equal(feature.compute(X), output[feature.name])
    assert not hasattr(supplied, "coef_")
    json.dumps(fitted.selection_report_, allow_nan=False)


def test_real_multiclass_coefficients_are_combined_and_attributed_correctly():
    rng = np.random.default_rng(795)
    X = pd.DataFrame(rng.normal(size=(360, 2)), columns=["a", "b"])
    y = np.digitize(X["a"] * X["b"], [-0.4, 0.4])
    engine = importance_engine(task="classification", estimator=LogisticRegression(max_iter=500), max_features=3)
    original_fit = engine._fit_learner
    models = []

    def record_fit(frame, target, template):
        model = original_fit(frame, target, template)
        models.append(model)
        return model

    with patch.object(engine, "_fit_learner", side_effect=record_fit):
        engine.fit(X, y)
    first = models[0]
    assert first.named_steps["learner"].coef_.shape[0] == 3
    values = np.abs(first.named_steps["learner"].coef_).mean(axis=0)
    expected = values / values.sum()
    numeric = next(entry for entry in first.named_steps["preprocess"].transformers_ if entry[0] == "numeric")
    start = first.named_steps["preprocess"].output_indices_["numeric"].start
    positions = {name: start + index for index, name in enumerate(numeric[2])}
    for name, score in engine.selection_report_["candidate_scores"].items():
        assert score == pytest.approx(expected[positions[name]])
    assert engine.selection_report_["learner_fit_count"] == len(models)
    assert engine.selection_report_["importance_ranking"]["learner_fits"] == 1


def test_ranking_dedup_and_imputation_never_read_holdout_labels_or_statistics():
    rng = np.random.default_rng(315)
    X = pd.DataFrame({"a": rng.uniform(0.1, 2, 240), "negative": rng.uniform(-5, -4, 240)})
    X["category"] = ["a", "b", None] * 80
    y = X["a"].to_numpy() ** 2 + rng.normal(scale=0.01, size=len(X))
    train, validation = train_test_split(np.arange(len(X)), test_size=0.25, random_state=42)
    first = importance_engine(estimator=DecisionTreeRegressor(max_depth=3), candidate_set="extended", max_features=4)
    original_fit = first._fit_learner
    observed = []

    def record_fit(frame, target, template):
        observed.append(set(frame.index))
        return original_fit(frame, target, template)

    with patch.object(first, "_fit_learner", side_effect=record_fit):
        first.fit(X, y)
    assert observed and all(rows <= set(train) for rows in observed)
    changed_X, changed_y = X.copy(), y.copy()
    changed_X.loc[validation, "negative"] = 1000
    changed_X.loc[validation, "category"] = "validation-only"
    changed_y[validation] += 10000
    second = importance_engine(
        estimator=DecisionTreeRegressor(max_depth=3), candidate_set="extended", max_features=4
    ).fit(changed_X, changed_y)
    assert first.selection_report_["candidate_ranking"] == second.selection_report_["candidate_ranking"]
    assert first.selection_report_["candidate_scores"] == second.selection_report_["candidate_scores"]
    assert first.selection_report_["importance_ranking"] == second.selection_report_["importance_ranking"]
    aliases = {item["name"]: item for item in first.selection_report_["importance_ranking"]["discarded_candidates"]}
    assert aliases["a_min_negative"]["duplicate_of"] == "negative"
    assert aliases["a_max_negative"]["duplicate_of"] == "a"


@pytest.mark.parametrize("kind", ["grouped", "temporal"])
def test_joint_ranking_uses_only_available_repeat_train_rows_without_residual_cv(kind):
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["a", "b"])
    y = X["a"] * X["b"]
    engine = importance_engine(estimator=DecisionTreeRegressor(max_depth=2), max_features=3)
    if kind == "grouped":
        metadata = {"groups": np.repeat(np.arange(3), 40)}
    else:
        timestamps = np.repeat(np.arange(4), 30)
        metadata = {"timestamps": timestamps, "label_end_times": timestamps + 1}
    boundaries = engine._validate_boundaries(
        metadata.get("groups"), metadata.get("timestamps"), len(X), label_end_times=metadata.get("label_end_times")
    )
    train, validation = engine._holdouts(y.to_numpy(), "regression", boundaries)[0]
    original_fit = engine._fit_learner
    observed = []

    def record_fit(frame, target, template):
        observed.append(set(frame.index))
        return original_fit(frame, target, template)

    with (
        patch.object(engine, "_fit_learner", side_effect=record_fit),
        patch.object(engine, "_cv_splits", side_effect=AssertionError("No residual CV in importance mode")),
    ):
        engine.fit(X, y, **metadata)
    assert all(rows == set(train) for rows in observed)
    assert engine.selection_report_["importance_ranking"]["fit_rows"] == len(train)
    assert engine.selection_report_["residual_cv_folds"] == 0
    assert engine.selection_report_["chronological_prefix_unscored"] is False
    assert set(train).isdisjoint(validation)


def test_lightgbm_ranking_clone_uses_gain_without_mutating_the_caller():
    lightgbm = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(31)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=["a", "b"])
    y = X["a"] * X["b"]
    supplied = lightgbm.LGBMRegressor(
        n_estimators=5, num_leaves=4, min_child_samples=5, n_jobs=1, importance_type="split", verbosity=-1
    )
    before = supplied.get_params().copy()
    engine = importance_engine(estimator=supplied, candidate_set="extended", max_features=3)
    original_fit = engine._fit_learner
    types = []

    def record_fit(frame, target, template):
        types.append(template.get_params().get("importance_type"))
        return original_fit(frame, target, template)

    with patch.object(engine, "_fit_learner", side_effect=record_fit):
        engine.fit(X, y)
    assert types[0] == "gain"
    assert all(value == "split" for value in types[1:])
    assert supplied.get_params() == before
    assert not hasattr(supplied, "booster_")
    assert engine.selection_report_["importance_ranking"]["importance_source"] == "feature_importances_:gain"


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing", "feature_importances_ or coef_"),
        ("negative", "nonnegative"),
        ("nan", "finite"),
        ("wrong_width", "encoded feature width"),
        ("native_matrix", "one-dimensional"),
        ("complex", "real numeric"),
        ("text", "real numeric"),
        ("empty_coef", "nonempty"),
    ],
)
def test_invalid_learner_importance_attributes_raise_explicitly(mode, message):
    X, y = mixed_data()
    fitted = importance_engine(estimator=IndexedImportanceRegressor(mode), max_features=5)
    with pytest.raises(ValueError, match=message):
        fitted.fit(X, y)
    assert not fitted.is_fitted
    assert fitted.selection_report_ == {}


def test_unsupported_bare_learner_has_no_silent_importance_fallback():
    X, y = mixed_data()
    with pytest.raises(ValueError, match="feature_importances_ or coef_"):
        importance_engine(estimator=DummyRegressor()).fit(X, y)
    with pytest.raises(ValueError, match="feature_importances_ or coef_"):
        importance_engine(estimator=None, max_candidates=1).fit(X, y)


def test_preprocessing_attribution_mismatch_is_not_guessed():
    X, y = mixed_data()
    engine = importance_engine(estimator=IndexedImportanceRegressor())
    frame = engine._model_frame(X).assign(candidate_a=np.sin(np.arange(len(X))))
    model = engine._fit_learner(frame, y, IndexedImportanceRegressor())
    weights, _ = engine._importance_values(model)
    model.named_steps["preprocess"].output_indices_["numeric"] = slice(0, 3)
    with pytest.raises(ValueError, match="numeric preprocessing changed"):
        engine._candidate_output_indices(model, [Feature(name="candidate_a")], len(weights))


def test_refit_zero_budget_and_default_modes_remain_strictly_separate():
    X, y = mixed_data()
    zero = importance_engine(max_candidates=0, stability_repeats=3).fit(X, y)
    assert zero.selection_report_["learner_fit_count"] == 3
    assert zero.selection_report_["importance_ranking"]["learner_fits"] == 0
    assert zero.selection_report_["importance_ranking"]["importance_source"] is None
    assert_frame_equal(zero.transform(X), X)
    with pytest.raises(ValueError, match="y is required"):
        zero.fit(X)
    assert zero.selection_report_ == {}
    assert not zero.is_fitted
    for screening in ["auto", "linear", "tree"]:
        engine = importance_engine(screening=screening, max_candidates=4)
        with patch.object(engine, "_rank_importance", side_effect=AssertionError("Opt-in only")):
            engine.fit(X, y)
        assert "importance_ranking" not in engine.selection_report_
        assert all("importance_ranking" not in repeat for repeat in engine.selection_report_["repeats"])


def test_final_joint_rescreen_runs_only_after_positive_count_selection():
    rng = np.random.default_rng(62)
    X = pd.DataFrame(rng.normal(size=(320, 2)), columns=["a", "b"])
    y = X["a"] * X["b"] + rng.normal(scale=0.01, size=len(X))
    positive = importance_engine(max_features=3, stability_repeats=3, selection_rule="paired_gain")
    with patch.object(positive, "_rank_importance", wraps=positive._rank_importance) as joint:
        positive.fit(X, y)
    assert positive.selection_report_["selected_count"] == 1
    assert joint.call_count == 4
    assert positive.selection_report_["importance_ranking"]["fit_rows"] == len(X)
    assert positive.selection_report_["learner_fit_count"] == 4 + 3 * 2
    negative = importance_engine(estimator=IndexedImportanceRegressor(), stability_repeats=3, max_features=3)
    with patch.object(negative, "_rank_importance", wraps=negative._rank_importance) as joint:
        negative.fit(X, y)
    assert negative.selection_report_["selected_count"] == 0
    assert joint.call_count == 3
    assert negative.selection_report_["ranking_scope"] == "baseline_retained_no_final_refit"

"""Synthetic bounded/mocked checks; no external datasets or measurements."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from benchmarks import oof_blending as blend
from benchmarks import prediction_ensemble, tabular
from featcopilot import classification
from featcopilot.core.feature import Feature, FeatureSet
from featcopilot.engines.predictive import PredictiveTabularEngine
from featcopilot.utils.model_preprocessing import make_classification_pipeline

Data = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


def _data(n_classes: int = 2, *, mixed: bool = False, integers: bool = False) -> Data:
    rng = np.random.default_rng(197)
    labels = np.arange(150) % n_classes
    X = pd.DataFrame(
        {"a": labels + rng.normal(scale=0.8, size=len(labels)), "b": rng.normal(size=len(labels))},
        index=np.arange(len(labels)) * 3 + 17,
    )
    if mixed:
        X["category"] = pd.Categorical(["PRIVATE-LEVEL-A", "PRIVATE-LEVEL-B", None] * 50)
        X["flag"] = labels % 2 == 0
        X.loc[X.index[::13], "a"] = np.nan
    y = pd.Series(labels * 7 - 12 if integers else [f"class-{value}" for value in labels], index=X.index)
    test = X.iloc[120:].copy()
    if mixed:
        test["category"] = ["UNSEEN-TEST-LEVEL", None, "PRIVATE-LEVEL-A"] * 10
    return X.iloc[:120].copy(), y.iloc[:120].copy(), test, y.iloc[120:].copy()


class _SignalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, num_leaves: int = 7):
        self.num_leaves = num_leaves

    def fit(self, X: np.ndarray, y: np.ndarray) -> _SignalClassifier:
        self.classes_ = np.unique(y)
        self.centers_ = np.stack([np.mean(X[y == label], axis=0) for label in self.classes_])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        distances = np.mean((X[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2)
        logits = -distances * self.num_leaves / 15
        logits -= logits.max(axis=1, keepdims=True)
        values = np.exp(logits)
        return np.asarray(values / values.sum(axis=1, keepdims=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes_[np.argmax(self.predict_proba(X), axis=1)])


class _BoundedEngineer:
    def __init__(self, **kwargs: object):
        self.config = SimpleNamespace(**kwargs)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit_hash_ = hashlib.sha256(pd.util.hash_pandas_object(X).values.tobytes() + y.tobytes()).hexdigest()
        self.selection_report_ = {
            "selected_features": ["absolute_a"],
            "selected_count": 1,
            "candidate_count": 1,
            "stability_repeats": self.config.stability_repeats,
            "selection_rule": self.config.selection_rule,
        }
        return self.transform(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.assign(absolute_a=np.abs(X["a"]))

    def get_feature_set(self) -> FeatureSet:
        return FeatureSet(
            [
                Feature(
                    name="absolute_a",
                    source_columns=["a"],
                    transformation="abs",
                    code="PRIVATE-FORMULA",
                    metadata={"private_fitted_vocabulary": "PRIVATE-LEVEL-A"},
                )
            ]
        )


@pytest.fixture
def bounded(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Keep the real complete HPO/refit loop with cheap deterministic internals."""
    tracked: dict[str, list] = {"bases": [], "pipeline_rows": [], "events": [], "predictions": []}
    original_fit = prediction_ensemble.CompleteBaseProcedure.fit
    original_predict = prediction_ensemble.CompleteBaseProcedure.predict_proba
    original_pipeline_fit = Pipeline.fit

    def small_model(
        X: pd.DataFrame, task: str, learner: str, parameters: dict[str, object], config: dict[str, object]
    ) -> Pipeline:
        assert task == "classification" and learner == "lightgbm"
        assert config["numeric_clip_quantiles"] == (0.0, 1.0)
        leaves = parameters["num_leaves"]
        assert isinstance(leaves, int)
        return make_classification_pipeline(X, _SignalClassifier(num_leaves=leaves))

    def counted_fit(
        self: prediction_ensemble.CompleteBaseProcedure, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> prediction_ensemble.CompleteBaseProcedure:
        tracked["bases"].append(self)
        tracked["events"].append(("fit", len(X), self.engineered))
        return original_fit(self, X, y)

    def counted_predict(self: prediction_ensemble.CompleteBaseProcedure, X: pd.DataFrame) -> np.ndarray:
        tracked["predictions"].append((self, len(X)))
        return original_predict(self, X)

    def pipeline_fit(self: Pipeline, X: pd.DataFrame, y: np.ndarray, **kwargs: object) -> Pipeline:
        tracked["pipeline_rows"].append(len(X))
        return original_pipeline_fit(self, X, y, **kwargs)

    monkeypatch.setattr(tabular, "make_model", small_model)
    monkeypatch.setattr(classification, "PredictiveTabularEngine", _BoundedEngineer)
    monkeypatch.setattr(prediction_ensemble.CompleteBaseProcedure, "fit", counted_fit)
    monkeypatch.setattr(prediction_ensemble.CompleteBaseProcedure, "predict_proba", counted_predict)
    monkeypatch.setattr(Pipeline, "fit", pipeline_fit)
    return tracked


def _run(data: Data, config: dict[str, object] | None = None) -> blend.OOFBlendResult:
    return blend.fit_oof_blend_bundle_result(*data, learner="lightgbm", config={"threads": 1, **(config or {})})


def _run_fixed(data: Data, config: dict[str, object] | None = None) -> blend.FixedAverageResult:
    return blend.fit_fixed_average_bundle_result(*data, learner="lightgbm", config={"threads": 1, **(config or {})})


def _stable_diagnostics(base: prediction_ensemble.CompleteBaseProcedure) -> dict[str, object]:
    return {
        key: value
        for key, value in base.diagnostics().items()
        if key != "fit_identity_sha256" and not key.endswith("seconds")
    }


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("integers", [False, True])
def test_exact_coverage_fixed_seed_classes_and_complete_fit_budget(
    n_classes: int, integers: bool, bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data(n_classes, integers=integers, mixed=True)
    calibration = Mock(side_effect=AssertionError("No calibration construction allowed"))
    monkeypatch.setattr(prediction_ensemble, "CalibratedClassifierCV", calibration)
    result = _run(data)
    assert len(bounded["bases"]) == 12
    assert len(bounded["pipeline_rows"]) == 48
    assert bounded["pipeline_rows"] == [72, 72, 72, 96] * 10 + [90, 90, 90, 120] * 2
    assert [base.engineered for base in bounded["bases"]] == [False, True] * 6
    assert result.source.base_estimators == tuple(bounded["bases"][-2:])
    assert len(bounded["predictions"]) == 12
    assert [count for _, count in bounded["predictions"]] == [24] * 10 + [30, 30]
    assert result.diagnostics["n_oof_covered"] == 120
    assert result.diagnostics["oof_coverage_histogram"] == [0, 120]
    expected = list(StratifiedKFold(5, shuffle=True, random_state=1042).split(data[0], data[1]))
    actual = blend.make_oof_splits(data[1])
    counts = np.zeros(len(data[0]), dtype=int)
    for (a, b), (c, d) in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        counts[d] += 1
    np.testing.assert_array_equal(counts, 1)
    for base in result.source.base_estimators:
        assert base.random_state == 42 and base.n_jobs == 1
        assert base.max_candidates == 1000 and base.max_added_features == 50
        assert base.numeric_clip_quantiles == (0.0, 1.0)
        assert base.categorical_encoding == "one_hot"
        assert len(base.trials_) == 3
        if base.engineered:
            assert base.engineer_ is not None
            assert base.engineer_.config.importance_resamples == 1
            assert base.engineer_.config.stability_repeats == 3
            assert base.engineer_.config.screening == "importance"
            assert base.engineer_.config.selection_rule == "paired_gain"
            assert base.engineer_.config.candidate_set == "extended"
            assert base.engineer_.config.include_frequency is True
    assert all(base is not result.source.raw and base is not result.source.fe for base in bounded["bases"][:-2])
    calibration.assert_not_called()


def test_full_controls_match_direct_complete_procedure_state_and_probabilities(bounded: dict[str, list]) -> None:
    data = _data(3, mixed=True)
    result = _run(data)
    for arm, full in zip(("raw", "fe"), result.source.base_estimators, strict=True):
        direct = prediction_ensemble.CompleteBaseProcedure(
            learner="lightgbm", config={"seed": 42, "threads": 1, "stability_repeats": 3}, engineered=arm == "fe"
        ).fit(data[0], data[1])
        assert _stable_diagnostics(full) == _stable_diagnostics(direct)
        assert pickle.dumps(full.model_) == pickle.dumps(direct.model_)
        assert full.selected_parameters_ == direct.selected_parameters_
        assert full.trials_ == direct.trials_
        np.testing.assert_array_equal(result.test_probability[arm], direct.predict_proba(data[2]))
    np.testing.assert_array_equal(
        result.test_probability["average"], 0.5 * result.test_probability["raw"] + 0.5 * result.test_probability["fe"]
    )
    identities = result.diagnostics["full_fit_identity_sha256"]
    assert isinstance(identities, dict)
    assert identities["raw"] == [result.source.raw.fit_identity_sha256_]
    assert identities["fe"] == [result.source.fe.fit_identity_sha256_]
    assert identities["average"] == identities["blend"] == identities["raw"] + identities["fe"]
    assert result.to_report()["n_input_features"] == 4
    assert result.to_report()["n_output_features"] == 5


def test_fixed_holdout_perturbation_cannot_affect_corresponding_fold_fits(
    bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data(mixed=True)
    data[0]["category"] = data[0]["category"].astype(object)
    splits = blend.make_oof_splits(data[1])
    monkeypatch.setattr(blend, "make_oof_splits", lambda *args: splits)
    before = _run(data)
    left = bounded["bases"][:2]
    changed_X, changed_y = data[0].copy(), data[1].copy()
    held_out = splits[0][1]
    changed_X.iloc[held_out, 0] = np.arange(len(held_out)) * 1e8
    changed_X.iloc[held_out, 2] = "PRIVATE-HOLDOUT-ONLY"
    changed_y.iloc[held_out] = np.roll(changed_y.iloc[held_out].to_numpy(), 1)
    after = _run((changed_X, changed_y, data[2], data[3]))
    right = bounded["bases"][12:14]
    for first, second in zip(left, right, strict=True):
        assert _stable_diagnostics(first) == _stable_diagnostics(second)
        assert pickle.dumps(first.model_) == pickle.dumps(second.model_)
        np.testing.assert_array_equal(first.predict_proba(data[2]), second.predict_proba(data[2]))
    assert bounded["bases"][2].diagnostics()["fitted_state_sha256"] != (
        bounded["bases"][14].diagnostics()["fitted_state_sha256"]
    )


def test_outer_test_labels_and_features_do_not_affect_weight_or_any_fit(bounded: dict[str, list]) -> None:
    data = _data(3)
    before = _run(data)
    changed_X = data[2].assign(a=-1e12, b=1e13)
    changed_y = pd.Series(np.roll(data[3].to_numpy(), 1), index=data[3].index)
    after = _run((data[0], data[1], changed_X, changed_y))
    assert before.source.weight == after.source.weight
    assert before.diagnostics["optimization"] == after.diagnostics["optimization"]
    for first, second in zip(bounded["bases"][:12], bounded["bases"][12:], strict=True):
        assert _stable_diagnostics(first) == _stable_diagnostics(second)
    assert before.metrics != after.metrics


def test_weight_freezes_before_full_refits_and_test_access(
    bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data()
    original_optimize = blend.optimize_blend_weight
    calls = []

    def optimize(*args) -> blend.BlendWeight:
        assert len(bounded["bases"]) == 10
        calls.append("weight")
        return original_optimize(*args)

    original_frame = blend.validate_classification_frame
    original_target = blend.classification_target

    def frame(X: pd.DataFrame) -> pd.DataFrame:
        if X is data[2]:
            assert len(bounded["bases"]) == 12 and calls == ["weight"]
        return original_frame(X)

    def target(y: pd.Series | np.ndarray, rows: int) -> np.ndarray:
        if y is data[3]:
            assert len(bounded["bases"]) == 12 and calls == ["weight"]
        return original_target(y, rows)

    monkeypatch.setattr(blend, "optimize_blend_weight", optimize)
    monkeypatch.setattr(blend, "validate_classification_frame", frame)
    monkeypatch.setattr(blend, "classification_target", target)
    result = _run(data)
    assert calls == ["weight"]
    with pytest.raises(FrozenInstanceError):
        result.source.__setattr__("weight", 0.25)


@pytest.mark.parametrize("mode, expected", [("raw", 0.0), ("fe", 1.0), ("interior", 0.5), ("identical", 0.0)])
def test_objective_endpoints_interior_and_identical(mode: str, expected: float) -> None:
    y = np.array(["a", "b"])
    classes = np.array(["a", "b"])
    raw = np.array([[0.9, 0.1], [0.9, 0.1]])
    fe = np.array([[0.1, 0.9], [0.1, 0.9]])
    if mode == "raw":
        raw = np.array([[1.0, 0.0], [0.0, 1.0]])
    elif mode == "fe":
        fe = np.array([[1.0, 0.0], [0.0, 1.0]])
    elif mode == "identical":
        fe = raw.copy()
    result = blend.optimize_blend_weight(raw, fe, y, classes)
    assert result.weight == pytest.approx(expected, abs=1e-5)
    if mode != "interior":
        assert result.weight == expected
    for arm, probability in (("raw", raw), ("fe", fe), ("average", (raw + fe) / 2)):
        assert result.losses[arm] == pytest.approx(log_loss(y, probability, labels=classes), abs=1e-15)
    assert result.losses["blend"] <= min(result.losses["raw"], result.losses["fe"])


def test_flat_multiclass_objective_ties_choose_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    y = np.array([0, 1, 2])
    raw = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
    fe = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    optimizer = Mock(return_value=OptimizeResult(success=True, x=0.25, fun=np.log(2), nfev=1))
    monkeypatch.setattr(blend, "minimize_scalar", optimizer)
    result = blend.optimize_blend_weight(raw, fe, y, np.arange(3))
    assert result.weight == 0.0
    assert len(set(result.losses.values())) == 1
    optimizer.assert_not_called()


def test_nonflat_objective_uses_declared_optimizer_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer = Mock(wraps=blend.minimize_scalar)
    monkeypatch.setattr(blend, "minimize_scalar", optimizer)
    raw = np.array([[0.9, 0.1], [0.9, 0.1]])
    fe = np.array([[0.1, 0.9], [0.1, 0.9]])
    result = blend.optimize_blend_weight(raw, fe, np.arange(2), np.arange(2))
    assert result.weight == pytest.approx(0.5, abs=1e-5)
    assert optimizer.call_args.kwargs == {
        "method": "bounded",
        "bounds": (0.0, 1.0),
        "options": {"xatol": 1e-5, "maxiter": 500},
    }


def test_nondyadic_flat_true_class_objective_chooses_zero_with_real_optimizer() -> None:
    labels = np.tile(np.arange(3), 6)
    true_probability = np.array([0.9148780980665152, 0.28104290984116415, 0.8133594703611421])
    raw = np.zeros((len(labels), 3), dtype=np.float64)
    fe = raw.copy()
    for row, label in enumerate(labels):
        probability = true_probability[label]
        raw[row, label] = fe[row, label] = probability
        raw[row, (label + 1) % 3] = fe[row, (label + 2) % 3] = (1.0 - probability) / 4
        raw[row, (label + 2) % 3] = fe[row, (label + 1) % 3] = (1.0 - probability) * 3 / 4
    normalized_raw = classification.validate_probabilities(raw, len(labels), 3)
    normalized_fe = classification.validate_probabilities(fe, len(labels), 3)
    assert not np.array_equal(normalized_raw, normalized_fe)
    rows = np.arange(len(labels))
    np.testing.assert_array_equal(normalized_raw[rows, labels], normalized_fe[rows, labels])
    result = blend.optimize_blend_weight(raw, fe, labels, np.arange(3))
    assert result.weight == 0.0
    assert len(set(result.losses.values())) == 1
    assert result.optimization["status"] == "identical_probabilities"
    assert result.optimization["nfev"] == 0
    assert result.optimization["interior_candidate"] is None


def test_flat_clipped_true_class_probabilities_choose_zero() -> None:
    labels = np.arange(3)
    raw = np.array([[0.0, 0.25, 0.75], [0.75, 0.0, 0.25], [0.25, 0.75, 0.0]])
    fe = raw[:, ::-1].copy()
    for row, label in enumerate(labels):
        fe[row] = np.roll([np.finfo(np.float64).eps / 4, 0.75, 0.25], label)
    result = blend.optimize_blend_weight(raw, fe, labels, labels)
    assert result.weight == 0.0
    assert len(set(result.losses.values())) == 1
    assert result.losses["blend"] == -np.log(np.finfo(np.float64).eps)


@pytest.mark.parametrize(
    "success, x, fun",
    [(False, 0.5, 1.0), (True, np.nan, 1.0), (True, 0.5, np.inf), (True, -0.1, 1.0), (True, 1.0, 1.0)],
)
def test_optimizer_failures_raise_without_successful_raw_fallback(
    success: bool, x: float, fun: float, monkeypatch: pytest.MonkeyPatch, bounded: dict[str, list]
) -> None:
    monkeypatch.setattr(blend, "minimize_scalar", lambda *args, **kwargs: OptimizeResult(success=success, x=x, fun=fun))
    with pytest.raises(RuntimeError, match="optimization failed"):
        _run(_data())
    assert len(bounded["bases"]) == 10


@pytest.mark.parametrize(
    "probability",
    [
        np.zeros((2, 2)),
        np.array([[np.nan, 0.5], [0.4, 0.6]]),
        np.array([[np.inf, 0.5], [0.4, 0.6]]),
        np.array([[1.1, -0.1], [0.4, 0.6]]),
        np.array([[0.2, 0.2], [0.4, 0.6]]),
        np.ones((2, 1)),
        np.ones((1, 2)),
        np.ones((2, 2, 1)),
        np.array([[1, 0], [0, 1]], dtype=int),
    ],
)
def test_probability_failures_raise(probability: np.ndarray) -> None:
    with pytest.raises(ValueError):
        blend.optimize_blend_weight(probability, np.array([[0.5, 0.5], [0.4, 0.6]]), np.array([0, 1]), np.arange(2))


def test_float32_roundoff_validates_then_normalizes_in_float64() -> None:
    raw = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.1, 0.7, 0.2]], dtype=np.float32)
    fe = raw[:, ::-1].copy()
    y = np.arange(3)
    result = blend.optimize_blend_weight(raw, fe, y, y)
    normalized = raw.astype(np.float64) / raw.sum(axis=1, dtype=np.float64)[:, None]
    assert result.losses["raw"] == pytest.approx(log_loss(y, normalized, labels=y), abs=1e-15)
    with pytest.raises(ValueError, match="roundoff"):
        blend.optimize_blend_weight(raw.astype(np.float64), fe, y, y)


def test_native_float32_probabilities_use_complete_base_normalization(
    bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _SignalClassifier.predict_proba
    monkeypatch.setattr(_SignalClassifier, "predict_proba", lambda self, X: original(self, X).astype(np.float32))
    data = _data(3)
    result = _run(data)
    for arm, base in zip(("raw", "fe"), result.source.base_estimators, strict=True):
        probability = base.predict_proba(data[2])
        assert probability.dtype == np.float64
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=2e-15, rtol=0)
        np.testing.assert_array_equal(result.test_probability[arm], probability)


@pytest.mark.parametrize("reverse_engineered", [False, True])
def test_class_column_permutations_are_aligned_and_missing_classes_rejected(
    reverse_engineered: bool, bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data(3)
    before = _run(data)
    original_fit = prediction_ensemble.CompleteBaseProcedure.fit
    original_predict = prediction_ensemble.CompleteBaseProcedure.predict_proba

    def reversed_fit(self, X, y):
        original_fit(self, X, y)
        if self.engineered == reverse_engineered:
            self.classes_ = self.classes_[::-1]
        return self

    def reversed_predict(self, X):
        probability = original_predict(self, X)
        return probability[:, ::-1] if self.engineered == reverse_engineered else probability

    monkeypatch.setattr(prediction_ensemble.CompleteBaseProcedure, "fit", reversed_fit)
    monkeypatch.setattr(prediction_ensemble.CompleteBaseProcedure, "predict_proba", reversed_predict)
    after = _run(data)
    assert before.source.weight == after.source.weight
    for arm in blend.ARMS:
        np.testing.assert_array_equal(before.test_probability[arm], after.test_probability[arm])
        np.testing.assert_array_equal(before.test_probability[arm], after.source.predict_probabilities(data[2])[arm])
    after.source.fe.classes_[0] = "missing"
    with pytest.raises(ValueError, match="class alignment"):
        after.source.predict_probabilities(data[2])


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_endpoint_inference_returns_exact_native_probabilities_and_only_needed_model(
    weight: float, bounded: dict[str, list]
) -> None:
    data = _data()
    result = _run(data)
    source = blend.OOFBlendSource(result.source.raw, result.source.fe, weight)
    arm = "raw" if weight == 0 else "fe"
    count = len(bounded["predictions"])
    np.testing.assert_array_equal(source.predict_proba(data[2]), result.test_probability[arm])
    assert len(bounded["predictions"]) == count + 1
    probability = result.test_probability[arm]
    assert blend._convex_probability(probability, probability, weight) is probability


def test_raw_fe_identity_selects_zero_without_optimizer(
    bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_BoundedEngineer, "transform", lambda self, X: X)
    optimizer = Mock(side_effect=AssertionError("Identical OOF probabilities need no optimization"))
    monkeypatch.setattr(blend, "minimize_scalar", optimizer)
    result = _run(_data())
    assert result.source.weight == 0
    optimization = result.diagnostics["optimization"]
    assert isinstance(optimization, dict)
    assert optimization["status"] == "identical_probabilities"
    for arm in blend.ARMS:
        np.testing.assert_array_equal(result.test_probability[arm], result.test_probability["raw"])
    optimizer.assert_not_called()


def test_oof_improvement_can_harm_external_test_after_full_refit(
    bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data()
    labels = pd.concat([data[1], data[3]]).to_dict()

    def injected(self, X):
        target = np.array([labels[index] for index in X.index])
        is_oof = len(self.raw_schema_.columns) == 2 and self.inner_split_["train"]["size"] == 72
        correctness = (0.9 if is_oof else 0.1) if self.engineered else 0.7
        probabilities = np.full((len(X), 2), 1.0 - correctness)
        probabilities[np.arange(len(X)), np.searchsorted(self.classes_, target)] = correctness
        return probabilities

    monkeypatch.setattr(prediction_ensemble.CompleteBaseProcedure, "predict_proba", injected)
    result = _run(data)
    assert result.source.weight == 1
    losses = result.diagnostics["oof_losses"]
    policy = result.diagnostics["source_policy"]
    assert isinstance(losses, dict) and isinstance(policy, dict)
    assert losses["blend"] < losses["raw"]
    assert result.metrics["blend"]["loss"] > result.metrics["raw"]["loss"]
    assert "no_harm_guarantee" in policy["interpretation"]


@pytest.mark.parametrize(
    "config",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"seed": 1.2},
        {"threads": 0},
        {"threads": True},
        {"stability_repeats": 1},
        {"max_candidates": 999},
        {"max_candidates": 1000.0},
        {"max_added_features": 51},
        {"categorical_encoding": "target"},
        {"numeric_clip_quantiles": [0.01, 0.99]},
        {"task": "regression"},
        {"groups": [1]},
        {"timestamps": [1]},
        {"label_end_times": [1]},
        {"split_kind": "temporal"},
        {"split_strategy": "grouped"},
        {"metadata": {"groups": [1]}},
        {"metadata": {"metadata": {"label_end_times": [1]}}},
        {"metadata": []},
    ],
)
def test_invalid_config_rejected_before_any_fit(config: dict[str, object], bounded: dict[str, list]) -> None:
    with pytest.raises((ValueError, TypeError)):
        _run(_data(), config)
    assert bounded["bases"] == []


@pytest.mark.parametrize("learner", ["linear", "catboost", "xgboost", "auto", "LightGBM"])
def test_first_experiment_is_explicit_lightgbm_only(learner: str, bounded: dict[str, list]) -> None:
    with pytest.raises(ValueError, match="explicit LightGBM"):
        blend.fit_oof_blend_bundle(*_data(), learner=learner, config={})
    assert bounded["bases"] == []


def test_insufficient_class_support_and_invalid_train_raise_without_reduction(bounded: dict[str, list]) -> None:
    with pytest.raises(ValueError, match="five"):
        blend.make_oof_splits(np.array([0] * 30 + [1] * 4))
    data = _data()
    with pytest.raises(TypeError, match="DataFrame"):
        _run((data[0].to_numpy(), *data[1:]))
    bad_y = data[1].copy()
    bad_y.iloc[0] = None
    with pytest.raises(ValueError, match="nonmissing"):
        _run((data[0], bad_y, data[2], data[3]))
    assert bounded["bases"] == []


def test_existing_inner_fe_class_coverage_guard_propagates_without_retry(
    bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(classification, "PredictiveTabularEngine", PredictiveTabularEngine)
    labels = np.repeat(np.arange(20), 5)
    train = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.sin(np.arange(100))})
    # OOF TRAIN has four rows/class and HPO TRAIN three: FE's 25% holdout
    # cannot contain twenty classes. Do not reduce, redraw or skip the FE arm.
    with pytest.raises(ValueError, match="holdout"):
        _run((train, pd.Series(labels), train.iloc[:20], pd.Series(np.arange(20))))
    assert len(bounded["bases"]) == 2
    assert len(bounded["pipeline_rows"]) == 4


@pytest.mark.parametrize("stage", ["oof", "full"])
def test_malformed_base_predictions_abort_instead_of_fallback(
    stage: str, bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = prediction_ensemble.CompleteBaseProcedure.predict_proba

    def invalid(self, X):
        probability = original(self, X)
        if (stage == "oof" and len(X) == 24) or (stage == "full" and len(X) == 30):
            probability[:] = 0
        return probability

    monkeypatch.setattr(prediction_ensemble.CompleteBaseProcedure, "predict_proba", invalid)
    with pytest.raises(ValueError, match="sum to one"):
        _run(_data())
    assert len(bounded["bases"]) == (1 if stage == "oof" else 12)


def test_coverage_invariant_fails_instead_of_dropping_rows(
    bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data()
    splits = blend.make_oof_splits(data[1])
    splits[-1] = splits[0]
    monkeypatch.setattr(blend, "make_oof_splits", lambda *args: splits)
    with pytest.raises(ValueError, match="exactly one"):
        _run(data)
    assert len(bounded["bases"]) == 10


def test_json_metrics_privacy_widths_and_nonadditive_timing(bounded: dict[str, list]) -> None:
    data = _data(mixed=True)
    result = _run(data)
    report = result.to_report()
    serialized = json.dumps(report, allow_nan=False)
    for forbidden in ("PRIVATE-LEVEL", "UNSEEN-TEST-LEVEL", "PRIVATE-FORMULA", "private_fitted_vocabulary"):
        assert forbidden not in serialized
    assert all(key not in report for key in ("source", "oof_probability", "test_probability", "models"))
    assert report["shared_cost_not_additive"] is True
    assert report["n_input_features"] == 4 and report["n_output_features"] == 5
    metric_reports = (report["raw_metrics"], report["fe_metrics"], report["average_metrics"], report["blend_metrics"])
    for arm, metric_report in zip(blend.ARMS, metric_reports, strict=True):
        assert metric_report == result.metrics[arm]
        assert {"log_loss", "loss", "brier_sum", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc"} <= (
            result.metrics[arm].keys()
        )
        expected = tabular.metrics(
            blend._CachedPredictionView(result.source.classes_, result.test_probability[arm]),
            data[2],
            data[3],
            "classification",
        )
        assert result.metrics[arm] == {**expected, "log_loss": expected["loss"]}
    report["diagnostics"]["weight"] = -1
    assert result.diagnostics["weight"] == result.source.weight
    timing = result.timing
    assert all(np.isfinite(value) and value >= 0 for value in timing.values())
    both_fits = timing["full_raw_fit_seconds"] + timing["full_fe_fit_seconds"]
    assert timing["average_training_seconds"] == both_fits
    assert timing["blend_training_seconds"] == (
        both_fits + timing["split_seconds"] + timing["oof_seconds"] + timing["weight_optimization_seconds"]
    )
    for arm in blend.ARMS:
        assert timing[f"{arm}_standalone_seconds"] == timing[f"{arm}_training_seconds"] + timing[f"{arm}_score_seconds"]
        if arm in ("average", "blend"):
            assert timing[f"{arm}_score_seconds"] == (
                timing[f"{arm}_cached_score_seconds"] + timing["raw_predict_seconds"] + timing["fe_predict_seconds"]
            )
    physical_stages = (
        timing["split_seconds"]
        + timing["oof_fit_seconds"]
        + timing["oof_predict_seconds"]
        + timing["weight_optimization_seconds"]
        + both_fits
        + timing["test_score_seconds"]
    )
    assert physical_stages <= timing["physical_bundle_wall_seconds"]
    for model in result.source.base_estimators:
        assert model.fe_fit_seconds_ + model.model_fit_seconds_ <= model.fit_seconds_


@pytest.mark.parametrize("n_classes", [2, 3])
def test_dataframe_pickle_batch_and_private_artifact_boundaries(n_classes: int, bounded: dict[str, list]) -> None:
    data = _data(n_classes, mixed=True)
    train_copy = data[0].copy(deep=True)
    result = _run(data)
    restored = pickle.loads(pickle.dumps(result.source))
    assert restored.weight == result.source.weight
    assert [model.fit_identity_sha256_ for model in restored.base_estimators] == [
        model.fit_identity_sha256_ for model in result.source.base_estimators
    ]
    for arm, probability in restored.predict_probabilities(data[2]).items():
        np.testing.assert_array_equal(probability, result.test_probability[arm])
    batches = np.vstack(
        [
            restored.predict_proba(data[2].iloc[:1]),
            restored.predict_proba(data[2].iloc[1:7]),
            restored.predict_proba(data[2].iloc[7:]),
        ]
    )
    np.testing.assert_allclose(batches, result.test_probability["blend"], rtol=1e-14, atol=1e-15)
    assert restored.predict(data[2]).shape == (len(data[2]),)
    assert set(restored.predict(data[2])) <= set(data[1])
    with pytest.raises(ValueError, match="schema"):
        restored.predict_proba(data[2].iloc[:, ::-1])
    with pytest.raises(TypeError, match="DataFrame"):
        restored.predict_proba(data[2].to_numpy())
    with pytest.raises(ValueError, match="schema"):
        restored.predict_proba(data[2].iloc[:0])
    pd.testing.assert_frame_equal(data[0], train_copy)
    assert b"PRIVATE-LEVEL-A" in pickle.dumps(restored)
    assert "PRIVATE-LEVEL-A" not in json.dumps(result.to_report())


def test_public_entry_point_never_returns_private_result(bounded: dict[str, list]) -> None:
    report = blend.fit_oof_blend_bundle(*_data(), learner="lightgbm", config={"threads": 1})
    assert isinstance(report, dict)
    assert set(report) == {
        "raw_metrics",
        "fe_metrics",
        "average_metrics",
        "blend_metrics",
        "n_input_features",
        "n_output_features",
        "diagnostics",
        "timing",
        "shared_cost_not_additive",
    }
    json.dumps(report, allow_nan=False)
    assert len(bounded["bases"]) == 12


@pytest.mark.parametrize("n_classes", [2, 3])
def test_native_lightgbm_small_smoke(n_classes: int, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("lightgbm")
    original_make = tabular.make_model

    def small_model(X, task, learner, parameters, config):
        model = original_make(X, task, learner, parameters, config)
        model.steps[-1][1].set_params(n_estimators=2, min_child_samples=5)
        return model

    monkeypatch.setattr(tabular, "make_model", small_model)
    data = _data(n_classes, mixed=True)
    result = _run(data)
    assert len(result.source.base_estimators) == 2
    assert result.source.fe.engineer_ is not None
    assert result.source.fe.engineer_.config.max_candidates == 1000
    assert result.source.fe.engineer_.config.stability_repeats == 3
    assert 0 <= result.source.weight <= 1
    restored = pickle.loads(pickle.dumps(result.source))
    for arm, probability in restored.predict_probabilities(data[2]).items():
        np.testing.assert_array_equal(probability, result.test_probability[arm])
        assert np.isfinite(result.metrics[arm]["loss"])
    for full in result.source.base_estimators:
        direct = prediction_ensemble.CompleteBaseProcedure(
            learner="lightgbm", config={"threads": 1}, engineered=full.engineered
        ).fit(data[0], data[1])
        assert _stable_diagnostics(full) == _stable_diagnostics(direct)
        np.testing.assert_array_equal(full.predict_proba(data[2]), direct.predict_proba(data[2]))
    fixed = _run_fixed(data)
    for arm in blend.FIXED_AVERAGE_ARMS:
        assert fixed.metrics[arm] == result.metrics[arm]
        np.testing.assert_array_equal(fixed.test_probability[arm], result.test_probability[arm])
    for old, new in zip(result.source.base_estimators, fixed.source.base_estimators, strict=True):
        assert _stable_diagnostics(old) == _stable_diagnostics(new)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("integers", [False, True])
def test_fixed_average_exact_two_full_fits_and_no_learned_operations(
    n_classes: int, integers: bool, bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    forbidden = Mock(side_effect=AssertionError("Fixed average must not split OOF, optimize or calibrate"))
    monkeypatch.setattr(blend, "make_oof_splits", forbidden)
    monkeypatch.setattr(blend, "optimize_blend_weight", forbidden)
    monkeypatch.setattr(blend, "minimize_scalar", forbidden)
    monkeypatch.setattr(prediction_ensemble, "CalibratedClassifierCV", forbidden)
    data = _data(n_classes, mixed=True, integers=integers)
    result = _run_fixed(data)
    assert len(bounded["bases"]) == 2
    assert bounded["pipeline_rows"] == [90, 90, 90, 120] * 2
    assert bounded["events"] == [("fit", 120, False), ("fit", 120, True)]
    assert result.source.raw is bounded["bases"][0]
    assert result.source.fe is bounded["bases"][1]
    assert bounded["predictions"] == [(result.source.raw, 30), (result.source.fe, 30)]
    assert set(result.metrics) == set(result.test_probability) == {"raw", "fe", "average"}
    assert result.diagnostics["weight"] == 0.5
    assert result.diagnostics["weight_optimized"] is False
    assert result.diagnostics["oof_complete_procedure_fit_count"] == 0
    assert result.diagnostics["full_complete_procedure_fit_count"] == 2
    assert result.diagnostics["member_count"] == 0
    assert result.diagnostics["members"] == []
    assert not hasattr(result, "oof_probability")
    assert not hasattr(result.source, "weight")
    for base in result.source.base_estimators:
        assert base.numeric_clip_quantiles == (0.0, 1.0)
        assert base.categorical_encoding == "one_hot"
        assert base.max_candidates == 1000 and base.max_added_features == 50
        assert base.stability_repeats == 3 and len(base.trials_) == 3
        if base.engineered:
            assert base.engineer_ is not None
            assert base.engineer_.config.importance_resamples == 1
            assert base.engineer_.config.include_frequency is True
            assert base.engineer_.config.candidate_set == "extended"
            assert base.engineer_.config.selection_rule == "paired_gain"
            assert base.engineer_.config.screening == "importance"
    probabilities = result.source.predict_probabilities(data[2])
    assert set(probabilities) == {"raw", "fe", "average"}
    np.testing.assert_array_equal(probabilities["average"], (probabilities["raw"] + probabilities["fe"]) / 2)
    np.testing.assert_array_equal(result.source.predict_proba(data[2]), probabilities["average"])
    np.testing.assert_array_equal(
        result.source.predict(data[2]), result.source.classes_[np.argmax(probabilities["average"], axis=1)]
    )
    assert len(bounded["bases"]) == 2
    assert len(bounded["pipeline_rows"]) == 8
    forbidden.assert_not_called()


@pytest.mark.parametrize("seed", [42, 137, 2029])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_fixed_average_matches_oof_full_controls_exactly(seed: int, n_classes: int, bounded: dict[str, list]) -> None:
    data = _data(n_classes, mixed=True)
    oof = _run(data, {"seed": seed})
    fixed = _run_fixed(data, {"seed": seed})
    assert len(bounded["bases"]) == 14
    assert len(bounded["pipeline_rows"]) == 56
    for arm in blend.FIXED_AVERAGE_ARMS:
        assert fixed.metrics[arm] == oof.metrics[arm]
        np.testing.assert_array_equal(fixed.test_probability[arm], oof.test_probability[arm])
    for previous, current in zip(oof.source.base_estimators, fixed.source.base_estimators, strict=True):
        assert _stable_diagnostics(previous) == _stable_diagnostics(current)
        assert pickle.dumps(previous.model_) == pickle.dumps(current.model_)
        assert previous.selected_parameters_ == current.selected_parameters_
        assert previous.trials_ == current.trials_


def test_fixed_average_outer_labels_cannot_affect_fits_and_test_access_is_last(
    bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data(3)
    original_frame = blend.validate_classification_frame
    original_target = blend.classification_target

    def frame(X: pd.DataFrame) -> pd.DataFrame:
        if X is data[2]:
            assert len(bounded["bases"]) in (2, 4)
            assert len(bounded["pipeline_rows"]) in (8, 16)
        return original_frame(X)

    def target(y: pd.Series | np.ndarray, rows: int) -> np.ndarray:
        if y is data[3]:
            assert len(bounded["bases"]) == 2
            assert len(bounded["pipeline_rows"]) == 8
        return original_target(y, rows)

    monkeypatch.setattr(blend, "validate_classification_frame", frame)
    monkeypatch.setattr(blend, "classification_target", target)
    before = _run_fixed(data)
    changed_y = pd.Series(np.roll(data[3].to_numpy(), 1), index=data[3].index)
    after = _run_fixed((data[0], data[1], data[2], changed_y))
    assert before.diagnostics["weight"] == after.diagnostics["weight"] == 0.5
    for first, second in zip(before.source.base_estimators, after.source.base_estimators, strict=True):
        assert _stable_diagnostics(first) == _stable_diagnostics(second)
    for arm in blend.FIXED_AVERAGE_ARMS:
        np.testing.assert_array_equal(before.test_probability[arm], after.test_probability[arm])
    assert before.metrics != after.metrics


@pytest.mark.parametrize("reverse_engineered", [False, True])
def test_fixed_average_class_alignment_permutations(
    reverse_engineered: bool, bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data(3)
    before = _run_fixed(data)
    original_fit = prediction_ensemble.CompleteBaseProcedure.fit
    original_predict = prediction_ensemble.CompleteBaseProcedure.predict_proba

    def fit(self, X, y):
        original_fit(self, X, y)
        if self.engineered == reverse_engineered:
            self.classes_ = self.classes_[::-1]
        return self

    def predict(self, X):
        probability = original_predict(self, X)
        return probability[:, ::-1] if self.engineered == reverse_engineered else probability

    monkeypatch.setattr(prediction_ensemble.CompleteBaseProcedure, "fit", fit)
    monkeypatch.setattr(prediction_ensemble.CompleteBaseProcedure, "predict_proba", predict)
    after = _run_fixed(data)
    assert before.metrics == after.metrics
    for arm, probability in after.source.predict_probabilities(data[2]).items():
        np.testing.assert_array_equal(probability, before.test_probability[arm])


@pytest.mark.parametrize("kind", ["zero", "nan", "shape", "class"])
def test_fixed_average_malformed_outputs_are_errors(
    kind: str, bounded: dict[str, list], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = prediction_ensemble.CompleteBaseProcedure.predict_proba

    def invalid(self, X):
        probability = original(self, X)
        if kind == "zero":
            probability[:] = 0
        elif kind == "nan":
            probability[0, 0] = np.nan
        elif kind == "shape":
            probability = probability[:, :1]
        else:
            self.classes_[0] = "absent"
        return probability

    monkeypatch.setattr(prediction_ensemble.CompleteBaseProcedure, "predict_proba", invalid)
    with pytest.raises(ValueError):
        _run_fixed(_data())
    assert len(bounded["bases"]) == 2
    assert len(bounded["pipeline_rows"]) == 8


@pytest.mark.parametrize(
    "config",
    [
        {"seed": -1},
        {"threads": 0},
        {"groups": [1]},
        {"metadata": {"timestamps": [1]}},
        {"categorical_encoding": "target"},
        {"stability_repeats": 1},
        {"max_candidates": 999},
        {"max_added_features": 49},
        {"numeric_clip_quantiles": [0.01, 0.99]},
        {"task": "regression"},
    ],
)
def test_fixed_average_reuses_strict_configuration_guards(config: dict[str, object], bounded: dict[str, list]) -> None:
    with pytest.raises(ValueError):
        _run_fixed(_data(), config)
    assert bounded["bases"] == []


def test_fixed_average_non_lightgbm_is_rejected(bounded: dict[str, list]) -> None:
    with pytest.raises(ValueError, match="explicit LightGBM"):
        blend.fit_fixed_average_bundle(*_data(), learner="linear", config={})
    assert bounded["bases"] == []


def test_fixed_average_json_shared_identity_provenance_and_timing(bounded: dict[str, list]) -> None:
    data = _data(mixed=True)
    result = _run_fixed(data)
    report = result.to_report()
    assert set(report) == {
        "raw_metrics",
        "fe_metrics",
        "average_metrics",
        "n_input_features",
        "n_output_features",
        "diagnostics",
        "timing",
        "shared_cost_not_additive",
    }
    assert report["shared_cost_not_additive"] is True
    assert report["n_input_features"] == 4 and report["n_output_features"] == 5
    diagnostics = result.diagnostics
    assert diagnostics["protocol"] == "lightgbm_iid_fixed_average_v1"
    assert diagnostics["blend_mode"] == "fixed_average"
    assert diagnostics["criterion"] == "prespecified_probability_average_weight_0.5"
    identities = diagnostics["full_fit_identity_sha256"]
    models = diagnostics["full_models"]
    policy = diagnostics["source_policy"]
    assert isinstance(identities, dict) and isinstance(models, dict) and isinstance(policy, dict)
    raw_id, fe_id = (base.fit_identity_sha256_ for base in result.source.base_estimators)
    assert identities == {"raw": [raw_id], "fe": [fe_id], "average": [raw_id, fe_id]}
    assert diagnostics["exact_shared_fits"] is True
    assert models["raw"] == result.source.raw.diagnostics()
    assert models["fe"] == result.source.fe.diagnostics()
    assert policy["oof"] == "not_performed"
    assert "no_weight_search_no_causal_variance_or_no_harm_guarantee" in policy["interpretation"]
    for key in ("oof_losses", "oof_seed", "optimization", "n_oof_covered", "oof_coverage_histogram"):
        assert key not in diagnostics
    serialized = json.dumps(report, allow_nan=False)
    for forbidden in ("PRIVATE-LEVEL", "UNSEEN-TEST-LEVEL", "PRIVATE-FORMULA", "private_fitted_vocabulary"):
        assert forbidden not in serialized
    timing = result.timing
    assert set(timing) == {
        "full_raw_fit_seconds",
        "full_fe_fit_seconds",
        "fit_seconds",
        "test_score_seconds",
        "physical_bundle_wall_seconds",
        *(
            f"{arm}_{suffix}_seconds"
            for arm in blend.FIXED_AVERAGE_ARMS
            for suffix in ("predict", "cached_score", "training", "score", "standalone")
        ),
    }
    assert all(np.isfinite(value) and value >= 0 for value in timing.values())
    both = timing["full_raw_fit_seconds"] + timing["full_fe_fit_seconds"]
    assert timing["raw_training_seconds"] == timing["full_raw_fit_seconds"]
    assert timing["fe_training_seconds"] == timing["full_fe_fit_seconds"]
    assert timing["average_training_seconds"] == both
    assert timing["average_score_seconds"] == (
        timing["average_cached_score_seconds"] + timing["raw_predict_seconds"] + timing["fe_predict_seconds"]
    )
    assert both <= timing["fit_seconds"]
    assert timing["fit_seconds"] + timing["test_score_seconds"] <= timing["physical_bundle_wall_seconds"]
    for arm in blend.FIXED_AVERAGE_ARMS:
        assert timing[f"{arm}_standalone_seconds"] == timing[f"{arm}_training_seconds"] + timing[f"{arm}_score_seconds"]
    report["diagnostics"]["weight"] = -1
    report["raw_metrics"]["loss"] = -1
    assert result.diagnostics["weight"] == 0.5
    assert result.metrics["raw"]["loss"] >= 0


def test_fixed_average_public_result_has_exactly_three_arms(bounded: dict[str, list]) -> None:
    report = blend.fit_fixed_average_bundle(*_data(), learner="lightgbm", config={"threads": 1})
    assert isinstance(report, dict)
    assert {key for key in report if key.endswith("_metrics")} == {"raw_metrics", "fe_metrics", "average_metrics"}
    assert len(bounded["bases"]) == 2
    assert len(bounded["pipeline_rows"]) == 8
    json.dumps(report, allow_nan=False)


def test_fixed_average_pickle_batches_dataframe_and_trusted_artifact_boundaries(bounded: dict[str, list]) -> None:
    data = _data(3, mixed=True)
    original_train = data[0].copy(deep=True)
    result = _run_fixed(data)
    restored = pickle.loads(pickle.dumps(result.source))
    for original, current in zip(result.source.base_estimators, restored.base_estimators, strict=True):
        assert original.fit_identity_sha256_ == current.fit_identity_sha256_
    assert not hasattr(restored, "weight")
    for arm, probability in restored.predict_probabilities(data[2]).items():
        np.testing.assert_array_equal(probability, result.test_probability[arm])
    batch = np.vstack([restored.predict_proba(data[2].iloc[:1]), restored.predict_proba(data[2].iloc[1:])])
    np.testing.assert_allclose(batch, result.test_probability["average"], rtol=1e-14, atol=1e-15)
    with pytest.raises(ValueError, match="schema"):
        restored.predict_proba(data[2].iloc[:, ::-1])
    with pytest.raises(TypeError, match="DataFrame"):
        restored.predict_proba(data[2].to_numpy())
    with pytest.raises(FrozenInstanceError):
        restored.__setattr__("weight", 0.7)
    assert b"PRIVATE-LEVEL-A" in pickle.dumps(restored)
    assert "PRIVATE-LEVEL-A" not in json.dumps(result.to_report())
    pd.testing.assert_frame_equal(original_train, data[0])

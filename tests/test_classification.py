"""Public classification artifacts and exact frozen synthetic-reference parity."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import sklearn
from joblib import dump, load
from sklearn.base import clone, is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from benchmarks import oof_blending, tabular
from benchmarks import prediction_ensemble as benchmark
from featcopilot import classification
from featcopilot.classification import FeatureEnsembleClassifier, NestedBaseClassifier
from featcopilot.utils import model_preprocessing
from featcopilot.utils.model_preprocessing import CrossFittedTargetEncoder

FROZEN_ARCHIVE_SHA256 = "f579e22161f95f5685d9e598e5ecfaf3bf7f14e3304745c6157a596cd62e5380"
REFERENCE_CASES = [
    f"{learner}-{classes}-{mixed}-{engineered}"
    for learner in ("linear", "lightgbm")
    for classes in (2, 3)
    for mixed in (False, True)
    for engineered in (False, True)
] + ["catboost-3-True-True"]


def _data(classes: int = 2) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(2030)
    y = pd.Series(np.arange(120) % classes)
    X = pd.DataFrame(
        {
            "a": y + rng.normal(size=120),
            "b": rng.normal(size=120),
            "category": ["private-one", "private-two", None] * 40,
            "flag": [True, False] * 60,
        }
    )
    X.loc[::13, "a"] = np.nan
    return X, y


def _classifier(
    mode: Literal["single", "subsample85", "raw_fe_average"] = "single", engineered: bool = False
) -> FeatureEnsembleClassifier:
    return FeatureEnsembleClassifier(
        LogisticRegression(random_state=42, max_iter=2000),
        [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
        mode=mode,
        engineered=engineered,
        n_jobs=1,
        max_candidates=20,
        stability_repeats=1,
    )


@pytest.mark.parametrize("mode", ["single", "subsample85", "raw_fe_average"])
@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("classes", [2, 3])
def test_public_inference_pickle_and_independent_feature_plans(
    mode: Literal["single", "subsample85", "raw_fe_average"], engineered: bool, classes: int
) -> None:
    X, y = _data(classes)
    prototype = _classifier(mode, engineered)
    original_X = X.copy(deep=True)
    classifier = prototype.fit(X, y.map(lambda value: f"class-{value}"))
    test = X.iloc[:8].copy()
    test["category"] = ["unseen", None] * 4
    test["a"] = [1e300, -1e300, np.inf, -np.inf, np.nan, 0.0, 1.0, 2.0]
    probability = classifier.predict_proba(test)
    assert is_classifier(classifier)
    assert probability.shape == (len(test), classes)
    assert classifier.predict(test).shape == (len(test),)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    pd.testing.assert_frame_equal(X, original_X)
    assert not hasattr(classifier, "transform")
    assert not hasattr(classifier, "get_feature_names_out")
    expected_members = 3 if mode == "subsample85" else (2 if mode == "raw_fe_average" and engineered else 1)
    assert len(classifier.estimators_) == expected_members
    assert len(classifier.get_feature_plans()) == len(classifier.estimators_)
    assert len({id(member) for member in classifier.estimators_}) == len(classifier.estimators_)
    batched = np.vstack([classifier.predict_proba(test.iloc[:1]), classifier.predict_proba(test.iloc[1:])])
    np.testing.assert_allclose(batched, probability, atol=1e-15, rtol=1e-14)
    restored = pickle.loads(pickle.dumps(classifier))
    np.testing.assert_array_equal(restored.predict_proba(test), probability)
    assert not hasattr(classifier.estimator, "classes_")
    report = classifier.diagnostics()
    assert "private-one" not in json.dumps(report, allow_nan=False)
    assert "private-two" not in json.dumps(report, allow_nan=False)
    widths = report["member_output_feature_counts"]
    assert isinstance(widths, list)
    assert report["max_member_output_features"] == max(widths) <= X.shape[1] + 50


@pytest.mark.parametrize("mode", ["single", "subsample85"])
def test_public_plain_never_constructs_calibrator_and_has_exact_fit_count(
    mode: Literal["single", "subsample85"], monkeypatch: pytest.MonkeyPatch
) -> None:
    X, y = _data()
    calibrate = Mock(side_effect=AssertionError("Public plain fitting must not construct calibration"))
    monkeypatch.setattr(CalibratedClassifierCV, "__init__", calibrate)
    rows: list[int] = []
    original_fit = Pipeline.fit

    def counted_fit(self: Pipeline, train: pd.DataFrame, target: np.ndarray, **kwargs: object) -> Pipeline:
        rows.append(len(train))
        return original_fit(self, train, target, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", counted_fit)
    classifier = _classifier(mode).fit(X, y)
    expected = [90, 90, 90, 120] if mode == "single" else [76, 76, 76, 102] * 3
    assert rows == expected
    snapshots = [member.predict_proba(X.iloc[:9]) for member in classifier.estimators_]
    np.testing.assert_allclose(classifier.predict_proba(X.iloc[:9]), np.mean(snapshots, axis=0), atol=2e-16, rtol=0)
    classifier.predict(X.iloc[:9])
    assert rows == expected
    calibrate.assert_not_called()


def test_constructor_clone_get_set_params_and_caller_immutability() -> None:
    estimator = LogisticRegression(C=0.7, random_state=42)
    candidates = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    classifier = FeatureEnsembleClassifier(estimator, candidates, mode="single", engineered=False, n_jobs=1)
    assert classifier.estimator is estimator
    assert classifier.parameter_candidates is candidates
    copied = clone(classifier)
    assert copied.estimator is not estimator
    assert copied.parameter_candidates is not candidates
    assert copied.get_params()["estimator__C"] == 0.7
    classifier.set_params(estimator__C=0.9)
    assert classifier.estimator is not estimator
    assert estimator.C == 0.7
    assert classifier.estimator.C == 0.9
    before = pickle.dumps(candidates)
    X, y = _data()
    classifier.fit(X, y)
    assert pickle.dumps(candidates) == before
    classifier.set_params(mode="subsample85")
    assert not hasattr(classifier, "estimators_")
    classifier.fit(X, y)
    assert len(classifier.estimators_) == 3
    with pytest.raises(ValueError, match="two classes"):
        classifier.fit(X, np.zeros(len(X), dtype=int))
    with pytest.raises(NotFittedError):
        classifier.predict_proba(X)
    assert not hasattr(classifier, "member_reports_")


def test_failure_in_last_bag_does_not_publish_partial_or_old_model(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _data()
    classifier = _classifier("subsample85").fit(X, y)
    count = 0
    original_fit = NestedBaseClassifier.fit

    def fail_last(self: NestedBaseClassifier, X: pd.DataFrame, y: np.ndarray) -> NestedBaseClassifier:
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("synthetic final-bag failure")
        return original_fit(self, X, y)

    monkeypatch.setattr(NestedBaseClassifier, "fit", fail_last)
    with pytest.raises(RuntimeError, match="final-bag"):
        classifier.fit(X, y)
    assert not hasattr(classifier, "estimators_")
    assert not hasattr(classifier, "classes_")
    assert not hasattr(classifier, "member_reports_")


def test_joblib_artifact_preserves_predictions_and_member_plan_copies(tmp_path: Path) -> None:
    X, y = _data()
    classifier = _classifier("subsample85", engineered=True).fit(X, y)
    artifact = tmp_path / "classifier.joblib"
    dump(classifier, artifact)
    restored = load(artifact)
    np.testing.assert_array_equal(restored.predict_proba(X.iloc[:7]), classifier.predict_proba(X.iloc[:7]))
    plans = classifier.get_feature_plans()
    assert len({id(plan) for plan in plans}) == 3
    for member, plan in zip(classifier.estimators_, plans, strict=True):
        assert plan is not member.get_feature_plan()


@pytest.mark.parametrize("engineered", [False, True])
def test_public_bag_training_firewall_and_no_outer_evaluation_api(
    engineered: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    X, y = _data()
    splits = classification.make_subsample85_splits(y, 42)
    monkeypatch.setattr(classification, "make_subsample85_splits", lambda *args: splits)
    original = _classifier("subsample85", engineered).fit(X, y)
    changed_X, changed_y = X.copy(), y.copy()
    heldout = splits[0][1]
    changed_X.loc[heldout, "a"] = 1e28
    changed_X.loc[heldout, "category"] = "private-heldout-only-vocabulary"
    changed_y.iloc[heldout] = np.roll(changed_y.iloc[heldout].to_numpy(), 1)
    changed = _classifier("subsample85", engineered).fit(changed_X, changed_y)
    first, second = original.estimators_[0], changed.estimators_[0]
    assert first.trials_ == second.trials_
    assert first.selected_parameters_ == second.selected_parameters_
    assert first.feature_names_out_ == second.feature_names_out_
    assert first.diagnostics()["fitted_state_sha256"] == second.diagnostics()["fitted_state_sha256"]
    np.testing.assert_array_equal(first.predict_proba(X.iloc[:5]), second.predict_proba(X.iloc[:5]))
    before = original.diagnostics()
    original.predict_proba(changed_X.iloc[:9])
    assert original.diagnostics() == before


def test_public_schema_and_class_coverage_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _data()
    classifier = _classifier().fit(X, y)
    with pytest.raises(ValueError, match="schema"):
        classifier.predict_proba(X.drop(columns="b"))
    with pytest.raises(TypeError, match="numeric"):
        classifier.predict_proba(X.assign(a="bad"))
    with pytest.raises(TypeError, match="DataFrame"):
        classifier.predict_proba(X.to_numpy())
    missing = X.iloc[:5].copy()
    missing["a"] = np.nan
    missing["b"] = np.nan
    missing["category"] = None
    assert np.isfinite(classifier.predict_proba(missing)).all()
    fit = Mock(side_effect=AssertionError("Class validation precedes all fitting"))
    monkeypatch.setattr(Pipeline, "fit", fit)
    with pytest.raises(ValueError, match="class|holdout|inner"):
        _classifier("subsample85").fit(X.iloc[:4], np.array([0, 1, 2, 3]))
    with pytest.raises(ValueError):
        _classifier().fit(X, np.linspace(0.0, 1.0, len(X)))
    fit.assert_not_called()


@pytest.mark.parametrize("mode", ["single", "subsample85"])
@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_public_target_encoding_is_shared_fold_local_and_serializable(
    mode: Literal["single", "subsample85"], engineered: bool, n_classes: int
) -> None:
    X, y = _data(n_classes)
    original = X.copy()
    classifier = _classifier(mode, engineered).set_params(categorical_encoding="target").fit(X, y)
    test = X.iloc[:9].copy()
    test["category"] = ["private-inference-only", None, "private-one"] * 3
    prediction = classifier.predict_proba(test)
    assert prediction.shape == (len(test), n_classes)
    np.testing.assert_allclose(prediction.sum(axis=1), 1.0, atol=2e-15, rtol=0)
    restored = pickle.loads(pickle.dumps(classifier))
    np.testing.assert_array_equal(restored.predict_proba(test), prediction)
    pd.testing.assert_frame_equal(X, original)
    train_positions = (
        [np.arange(len(X))]
        if mode == "single"
        else [train for train, _ in classification.make_subsample85_splits(y, 42)]
    )
    for member, positions in zip(classifier.estimators_, train_positions, strict=True):
        report = member.diagnostics()
        encoding = report["encoding_report"]
        assert isinstance(encoding, dict)
        assert encoding["active"] is True
        assert encoding["target_type"] == ("binary" if n_classes == 2 else "multiclass")
        assert encoding["n_input_features"] == 1
        assert encoding["output_width"] == (1 if n_classes == 2 else n_classes)
        expected_rows = len(X) if mode == "single" else len(X) - int(np.ceil(len(X) * 0.15))
        assert encoding["n_train_rows"] == expected_rows
        assert encoding["class_counts"] == [int(np.count_nonzero(y.iloc[positions] == label)) for label in np.unique(y)]
        width = member.model_.steps[1][1].output_indices_["categorical"].stop
        assert report["model_input_width"] == width
        assert set(member.feature_names_out_).issuperset(X.columns)
        assert not any("target" in feature.transformation.lower() for feature in member.get_feature_plan())
        if engineered:
            assert member.engineer_ is not None
            assert member.engineer_.config.categorical_encoding == "target"
    serialized = json.dumps(classifier.diagnostics(), allow_nan=False)
    assert "private-one" not in serialized and "private-inference-only" not in serialized
    assert clone(classifier).categorical_encoding == "target"


@pytest.mark.parametrize("values", [(1, 2), (True, False)])
@pytest.mark.parametrize("dtype", ["object", "category"])
@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("mode", ["single", "subsample85"])
def test_public_target_preserves_category_types_before_engine_and_inference(
    values: tuple[object, object],
    dtype: str,
    engineered: bool,
    mode: Literal["single", "subsample85"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = values
    X = pd.DataFrame(
        {
            "category": pd.Series([first, second] * 60, dtype=dtype),
            "number": np.random.default_rng(42).normal(size=120),
        }
    )
    y = pd.Series([1, 0] * 60)
    original = X.copy(deep=True)
    observed_category_columns: list[pd.Series] = []
    original_fit_transform = CrossFittedTargetEncoder.fit_transform

    def capture(
        self: CrossFittedTargetEncoder, frame: pd.DataFrame, target: np.ndarray, **kwargs: object
    ) -> np.ndarray:
        observed_category_columns.append(frame["category"].copy())
        assert frame["category"].dtype == object or isinstance(frame["category"].dtype, pd.CategoricalDtype)
        assert not any(isinstance(value, str) for value in frame["category"])
        return original_fit_transform(self, frame, target, **kwargs)

    monkeypatch.setattr(CrossFittedTargetEncoder, "fit_transform", capture)
    classifier = _classifier(mode, engineered).set_params(categorical_encoding="target").fit(X, y)
    assert len(observed_category_columns) >= (4 if mode == "single" else 12)
    if engineered:
        assert len(observed_category_columns) > (4 if mode == "single" else 12)
    test = pd.DataFrame({"category": pd.Series([first, str(first)], dtype=object), "number": [0.0, 0.0]})
    test_original = test.copy(deep=True)
    probability = classifier.predict_proba(test)
    assert np.isfinite(probability).all()
    for member in classifier.estimators_:
        features = member._features(test)
        pd.testing.assert_series_equal(features["category"], test["category"])
        preprocessing = member.model_.steps[1][1]
        assert next(columns for name, _, columns in preprocessing.transformers_ if name == "categorical") == [
            "category"
        ]
        encoder = preprocessing.named_transformers_["categorical"]
        assert isinstance(encoder, CrossFittedTargetEncoder)
        encoded = encoder.transform(features[["category"]])
        full_prior = encoder.encoder_.target_mean_
        assert encoded[1, 0] == full_prior
        assert encoded[0, 0] > full_prior
        assert encoder.get_encoding_report()["n_input_features"] == 1
        if engineered:
            assert member.engineer_ is not None
            assert "category" not in member.engineer_._numeric_columns
            assert not any("target" in feature.transformation.lower() for feature in member.get_feature_plan())
    pd.testing.assert_frame_equal(X, original, check_exact=True)
    pd.testing.assert_frame_equal(test, test_original, check_exact=True)
    restored = pickle.loads(pickle.dumps(classifier))
    np.testing.assert_array_equal(restored.predict_proba(test), probability)


@pytest.mark.parametrize("mode", ["single", "subsample85"])
def test_lightgbm_target_encoded_feature_selection_keeps_numeric_attribution(
    mode: Literal["single", "subsample85"],
) -> None:
    lightgbm = pytest.importorskip("lightgbm")
    X, y = _data(3)
    classifier = FeatureEnsembleClassifier(
        lightgbm.LGBMClassifier(
            n_estimators=4, n_jobs=1, random_state=42, deterministic=True, force_col_wise=True, verbosity=-1
        ),
        [{"num_leaves": 7}, {"num_leaves": 15}, {"num_leaves": 31}],
        mode=mode,
        categorical_encoding="target",
        n_jobs=1,
        max_candidates=20,
        stability_repeats=1,
    ).fit(X, y)
    assert np.isfinite(classifier.predict_proba(X.iloc[:5])).all()
    for member in classifier.estimators_:
        assert member.engineer_ is not None
        assert member.engineer_.selection_report_["screening"] == "importance"
        assert member.model_.steps[1][1].output_indices_["categorical"].stop == len(member.feature_names_out_) - 1 + 3


def test_target_mode_forwards_seed_and_checks_each_actual_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _data(3)
    calls: list[tuple[int, int, int]] = []
    original = CrossFittedTargetEncoder.fit_transform

    def record(self: CrossFittedTargetEncoder, X: pd.DataFrame, y: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append((len(X), self.random_state, len(np.unique(y))))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(CrossFittedTargetEncoder, "fit_transform", record)
    _classifier(engineered=True).set_params(categorical_encoding="target", random_state=137).fit(X, y)
    assert len(calls) > 4
    assert all(seed == 137 and n_classes == 3 for _, seed, n_classes in calls)
    assert max(rows for rows, _, _ in calls) == len(X)
    assert min(rows for rows, _, _ in calls) < int(0.75 * len(X))
    assert not any(rows > len(X) for rows, _, _ in calls)


def test_numeric_only_target_is_explicitly_inactive_and_needs_no_new_sklearn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sklearn, "__version__", "1.0.2")
    factory = Mock(side_effect=AssertionError("No TargetEncoder is needed for numeric-only inputs"))
    monkeypatch.setattr(model_preprocessing, "_make_target_encoder", factory)
    X = pd.DataFrame({"number": np.arange(12, dtype=float)})
    y = pd.Series(np.arange(12) % 2)
    default = _classifier().fit(X, y)
    target = _classifier().set_params(categorical_encoding="target").fit(X, y)
    np.testing.assert_array_equal(default.predict_proba(X), target.predict_proba(X))
    report = target.estimators_[0].diagnostics()["encoding_report"]
    assert report == {
        "requested": "target",
        "active": False,
        "reason": "no_categorical_columns",
        "n_input_features": 0,
        "output_width": 0,
        "training_output_bytes": 0,
    }
    assert "encoding_report" not in default.estimators_[0].diagnostics()
    factory.assert_not_called()


@pytest.mark.parametrize("mode", [None, True, "bad", ["target"]])
def test_invalid_public_categorical_encoding_fails_before_fitting(
    mode: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = Mock(side_effect=AssertionError("Invalid encoding cannot reach a model fit"))
    monkeypatch.setattr(Pipeline, "fit", fit)
    with pytest.raises(ValueError, match="categorical_encoding"):
        _classifier().set_params(categorical_encoding=mode).fit(*_data())
    fit.assert_not_called()


def test_explicit_one_hot_public_option_preserves_default_model_and_reports() -> None:
    X, y = _data(3)
    default = _classifier().fit(X, y)
    explicit = _classifier().set_params(categorical_encoding="one_hot").fit(X, y)
    np.testing.assert_array_equal(default.predict_proba(X), explicit.predict_proba(X))
    assert default.estimators_[0].trials_ == explicit.estimators_[0].trials_
    assert (
        default.estimators_[0].diagnostics()["fitted_state_sha256"]
        == explicit.estimators_[0].diagnostics()["fitted_state_sha256"]
    )
    assert "encoding_report" not in default.estimators_[0].diagnostics()


def test_public_native_catboost_configuration_is_not_silently_one_hot_encoded() -> None:
    catboost = pytest.importorskip("catboost")
    estimator = catboost.CatBoostClassifier(cat_features=["category"], iterations=4, verbose=False)
    classifier = FeatureEnsembleClassifier(
        estimator, [{"depth": 3}, {"depth": 6}, {"depth": 9}], mode="single", n_jobs=1
    )
    with pytest.raises(ValueError, match="native cat_features"):
        classifier.fit(*_data())


@pytest.mark.parametrize(
    "parameters",
    [
        {"parameter_candidates": [{"C": 1}]},
        {"parameter_candidates": [{"C": 1}, {"C": 2}, {"regularization": 3}]},
        {"parameter_candidates": [{"C": 1}, {"C": 2}, None]},
        {"mode": "cv3"},
        {"mode": None},
        {"n_jobs": 0},
        {"n_jobs": True},
        {"random_state": "42"},
        {"random_state": -1},
        {"engineered": 1},
        {"numeric_clip_quantiles": (0.01, 0.99)},
        {"estimator": LinearRegression()},
        {"estimator": make_pipeline(StandardScaler(), LogisticRegression())},
    ],
)
def test_invalid_configuration_rejected_before_any_fit(
    parameters: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    classifier = _classifier()
    classifier.set_params(**parameters)
    fit = Mock(side_effect=AssertionError("Invalid configuration cannot reach learner fit"))
    monkeypatch.setattr(Pipeline, "fit", fit)
    with pytest.raises((ValueError, TypeError)):
        classifier.fit(*_data())
    fit.assert_not_called()


@pytest.mark.parametrize(
    "name", ["groups", "timestamps", "label_end_times", "sample_weight", "eval_set", "validation_data", "task"]
)
def test_iid_only_fit_metadata_rejected_before_any_fitting(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fit = Mock(side_effect=AssertionError("Metadata rejection must precede fitting"))
    monkeypatch.setattr(Pipeline, "fit", fit)
    with pytest.raises(TypeError, match="IID"):
        _classifier().fit(*_data(), **{name: [1, 2]})
    fit.assert_not_called()


def test_private_opaque_parameter_report_never_calls_repr() -> None:
    class Opaque:
        def __repr__(self) -> str:
            raise AssertionError("Do not reveal parameter repr")

    result = classification._report_parameters({"token": "sensitive", "opaque": Opaque(), "C": 1.0})
    assert result == {"token": {"type": "str"}, "opaque": {"type": "Opaque"}, "C": 1.0}
    assert "sensitive" not in json.dumps(result)


@pytest.mark.parametrize("mode", ["single", "subsample85", "raw_fe_average"])
@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("label_kind", ["string", "nonzero_integer"])
@pytest.mark.parametrize("candidate_override", [False, True])
def test_class_weight_original_labels_match_explicit_encoded_weights(
    mode: Literal["single", "subsample85", "raw_fe_average"],
    engineered: bool,
    label_kind: str,
    candidate_override: bool,
) -> None:
    X, encoded_y = _data(3)
    labels = (
        np.array(["negative", "positive", "third"])
        if label_kind == "string"
        else np.array([-7, 11, 40], dtype=np.int64)
    )
    y = pd.Series(labels[encoded_y.to_numpy()])
    weights = {labels[0]: 0.5, labels[2]: 3.0}
    override = {labels[0]: 2.0, labels[1]: 0.75, labels[2]: 1.25}
    candidates: list[dict[str, object]] = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    if candidate_override:
        candidates[1]["class_weight"] = override
        candidates[2]["class_weight"] = "balanced"
    original_candidates = pickle.dumps(candidates)
    original_weights = pickle.dumps(weights)
    estimator = LogisticRegression(random_state=42, max_iter=2000, class_weight=weights)
    classifier = FeatureEnsembleClassifier(
        estimator,
        candidates,
        mode=mode,
        engineered=engineered,
        n_jobs=1,
        max_candidates=20,
        stability_repeats=1,
    ).fit(X, y)
    encoded_candidates: list[dict[str, object]] = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    if candidate_override:
        encoded_candidates[1]["class_weight"] = {0: 2.0, 1: 0.75, 2: 1.25}
        encoded_candidates[2]["class_weight"] = "balanced"
    reference = FeatureEnsembleClassifier(
        LogisticRegression(random_state=42, max_iter=2000, class_weight={0: 0.5, 2: 3.0}),
        encoded_candidates,
        mode=mode,
        engineered=engineered,
        n_jobs=1,
        max_candidates=20,
        stability_repeats=1,
    ).fit(X, encoded_y)
    np.testing.assert_array_equal(classifier.predict_proba(X.iloc[:11]), reference.predict_proba(X.iloc[:11]))
    np.testing.assert_array_equal(classifier.classes_, labels)
    assert set(classifier.predict(X.iloc[:11])).issubset(set(labels))
    assert estimator.class_weight is weights
    assert pickle.dumps(weights) == original_weights
    assert pickle.dumps(candidates) == original_candidates
    for actual, expected in zip(classifier.estimators_, reference.estimators_, strict=True):
        assert [trial["loss"] for trial in actual.trials_] == [trial["loss"] for trial in expected.trials_]
        assert actual.model_.steps[-1][1].class_weight == expected.model_.steps[-1][1].class_weight
        assert actual.engineered == expected.engineered
        if actual.engineered:
            assert actual.engineer_ is not None and expected.engineer_ is not None
            assert actual.engineer_.config.estimator is not None and expected.engineer_.config.estimator is not None
            assert (
                actual.engineer_.config.estimator.get_params()["class_weight"]
                == expected.engineer_.config.estimator.get_params()["class_weight"]
            )


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_class_weight_none_and_balanced_are_preserved(class_weight: str | None) -> None:
    X, y = _data()
    classifier = _classifier()
    classifier.set_params(estimator__class_weight=class_weight)
    classifier.fit(X, y + 10)
    assert classifier.estimators_[0].model_.steps[-1][1].class_weight == class_weight


def test_class_weight_numpy_scalar_equality_and_candidate_override() -> None:
    classes = np.array([10, 20, 30], dtype=np.int64)
    weights = {np.float64(10.0): 2.0, np.int32(30): 3.0}
    parameters = classification._encoded_weight_parameters({}, {"class_weight": weights}, classes)
    assert parameters == {"class_weight": {0: 2.0, 2: 3.0}}
    assert list(weights) == [np.float64(10.0), np.int32(30)]
    assert classification._encoded_weight_parameters(
        {"class_weight": None}, {"class_weight": {"unused-unknown": 4.0}}, classes
    ) == {"class_weight": None}


@pytest.mark.parametrize("where", ["prototype", "last_candidate"])
def test_unknown_class_weight_keys_fail_before_any_learner_fit(where: str, monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _data()
    classifier = _classifier("subsample85", engineered=True)
    if where == "prototype":
        classifier.set_params(estimator__class_weight={10: 1.0, 11: 2.0, 999: 3.0})
    else:
        classifier.set_params(parameter_candidates=[{"C": 0.1}, {"C": 1.0}, {"C": 10.0, "class_weight": {999: 3.0}}])
    fit = Mock(side_effect=AssertionError("All class-weight keys must be checked before model fitting"))
    monkeypatch.setattr(LogisticRegression, "fit", fit)
    with pytest.raises(ValueError, match="class_weight.*original training labels"):
        classifier.fit(X, y + 10)
    fit.assert_not_called()
    assert not hasattr(classifier, "estimators_")


@pytest.mark.parametrize("mode", ["single", "subsample85"])
def test_nonpickleable_classifier_fits_in_memory_with_explicit_missing_hash(
    mode: Literal["single", "subsample85"], monkeypatch: pytest.MonkeyPatch
) -> None:
    X, y = _data()
    classifier = FeatureEnsembleClassifier(
        SVC(probability=True, kernel=lambda a, b: a @ b.T, random_state=42),
        [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
        mode=mode,
        engineered=False,
        n_jobs=1,
    )
    warning = Mock()
    monkeypatch.setattr(classification.logger, "warning", warning)
    classifier.fit(X, y)
    probability = classifier.predict_proba(X.iloc[:9])
    assert probability.shape == (9, 2)
    assert np.isfinite(probability).all()
    assert classifier.predict(X.iloc[:9]).shape == (9,)
    report = classifier.diagnostics()
    members = report["members"]
    assert isinstance(members, list)
    for member in members:
        assert member["fitted_state_sha256"] is None
        assert "not serializable by pickle" in member["fitted_state_unavailable_reason"]
    for base in classifier.estimators_:
        assert base.diagnostics()["fitted_state_sha256"] is None
        assert base.diagnostics()["fitted_state_unavailable_reason"]
    warning.assert_called_once()
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(classifier.predict_proba(X.iloc[:9]), probability)
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.dumps(classifier)


@pytest.mark.parametrize("failure", [pickle.PicklingError, TypeError, AttributeError])
def test_diagnostic_serialization_failures_are_explicit_and_warning_is_once(
    failure: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    X, y = _data()
    base = NestedBaseClassifier(
        LogisticRegression(random_state=42),
        [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
        engineered=False,
        n_jobs=1,
    ).fit(X, y)
    original_hash = classification._sha256

    def unavailable(value: object) -> str:
        if isinstance(value, tuple) and len(value) == 4 and isinstance(value[0], Pipeline):
            raise failure("synthetic serialization failure")
        return original_hash(value)

    warning = Mock()
    monkeypatch.setattr(classification, "_sha256", unavailable)
    monkeypatch.setattr(classification.logger, "warning", warning)
    assert base.diagnostics()["fitted_state_sha256"] is None
    assert base.diagnostics()["fitted_state_unavailable_reason"]
    warning.assert_called_once()
    assert np.isfinite(base.predict_proba(X.iloc[:3])).all()


def test_unrelated_fingerprint_failure_is_not_silently_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _classifier().fit(*_data()).estimators_[0]
    original_hash = classification._sha256

    def unexpected(value: object) -> str:
        if isinstance(value, tuple) and len(value) == 4 and isinstance(value[0], Pipeline):
            raise RuntimeError("unrelated failure")
        return original_hash(value)

    monkeypatch.setattr(classification, "_sha256", unexpected)
    with pytest.raises(RuntimeError, match="unrelated failure"):
        base.diagnostics()


@pytest.mark.parametrize("mode", ["subsample85", "raw_fe_average"])
def test_import_and_prediction_without_benchmark_or_optional_learner_packages(mode: str) -> None:
    script = """
import builtins
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'benchmarks', 'lightgbm', 'catboost', 'xgboost', 'openfe', 'autofeat', 'openml',
                              'copilot', 'litellm', 'openai', 'transformers'}:
        raise AssertionError('Unexpected optional/benchmark import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from featcopilot.classification import FeatureEnsembleClassifier
X = pd.DataFrame({'x': np.arange(80, dtype=float), 'cat': ['a', 'b'] * 40})
y = np.arange(80) % 2
classifier = FeatureEnsembleClassifier(LogisticRegression(random_state=42),
    [{'C': .1}, {'C': 1.}, {'C': 10.}], mode='subsample85', n_jobs=1,
    max_candidates=10, stability_repeats=1)
classifier.fit(X, y)
assert classifier.predict_proba(X.iloc[:2]).shape == (2, 2)
"""
    script = script.replace("mode='subsample85'", f"mode='{mode}'")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("engineered", [False, True])
def test_fixed_average_fit_counts_full_train_and_no_weight_learning(
    engineered: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbidden = Mock(side_effect=AssertionError("Fixed averaging cannot use blend OOF/calibration/optimization"))
    monkeypatch.setattr(classification, "make_subsample85_splits", forbidden)
    monkeypatch.setattr(benchmark, "make_ensemble_splits", forbidden)
    monkeypatch.setattr(oof_blending, "make_oof_splits", forbidden)
    monkeypatch.setattr(oof_blending, "optimize_blend_weight", forbidden)
    monkeypatch.setattr(oof_blending, "minimize_scalar", forbidden)
    monkeypatch.setattr(CalibratedClassifierCV, "__init__", forbidden)
    complete_fits: list[tuple[bool, int]] = []
    prediction_models: list[Pipeline] = []
    pipeline_fit_rows: list[int] = []
    original_fit = NestedBaseClassifier.fit
    original_model = NestedBaseClassifier._make_model
    original_pipeline_fit = Pipeline.fit

    def counted_base(self: NestedBaseClassifier, X: pd.DataFrame, y: np.ndarray) -> NestedBaseClassifier:
        complete_fits.append((self.engineered, len(X)))
        return original_fit(self, X, y)

    def tracked_model(self: NestedBaseClassifier, X: pd.DataFrame, parameters: dict[str, object]) -> Pipeline:
        model = original_model(self, X, parameters)
        prediction_models.append(model)
        return model

    def counted_pipeline(self: Pipeline, X: pd.DataFrame, y: np.ndarray, **kwargs: object) -> Pipeline:
        if any(self is model for model in prediction_models):
            pipeline_fit_rows.append(len(X))
        return original_pipeline_fit(self, X, y, **kwargs)

    monkeypatch.setattr(NestedBaseClassifier, "fit", counted_base)
    monkeypatch.setattr(NestedBaseClassifier, "_make_model", tracked_model)
    monkeypatch.setattr(Pipeline, "fit", counted_pipeline)
    X, y = _data()
    classifier = _classifier("raw_fe_average", engineered).fit(X, y)
    expected_roles = [False, True] if engineered else [False]
    assert complete_fits == [(role, len(X)) for role in expected_roles]
    assert pipeline_fit_rows == [90, 90, 90, 120] * len(expected_roles)
    assert len(prediction_models) == (8 if engineered else 4)
    assert classifier.member_roles_ == (("raw", "fe") if engineered else ("raw",))
    assert classifier.weights_ == ((0.5, 0.5) if engineered else (1.0,))
    assert classifier.estimators_[0].engineer_ is None
    assert len(classifier.get_feature_plans()[0]) == 0
    if engineered:
        assert classifier.estimators_[1].engineer_ is not None
        assert classifier.estimators_[0] is not classifier.estimators_[1]
        assert classifier.estimators_[0].model_ is not classifier.estimators_[1].model_
        assert classifier.estimators_[0].raw_schema_ is not classifier.estimators_[1].raw_schema_
    report = classifier.diagnostics()
    assert report["weights"] == list(classifier.weights_)
    assert report["member_roles"] == list(classifier.member_roles_)
    assert report["fe_fitted"] is engineered
    assert report["weight_optimized"] is False
    assert report["oof_weight_fit_count"] == 0
    assert [member["role"] for member in classifier.member_reports_] == list(classifier.member_roles_)
    assert [member["weight"] for member in classifier.member_reports_] == list(classifier.weights_)
    fe_seconds = report["base_fe_seconds"]
    assert isinstance(fe_seconds, float)
    assert fe_seconds > 0 if engineered else fe_seconds == 0
    classifier.predict_proba(X.iloc[:8])
    assert len(complete_fits) == len(expected_roles)
    forbidden.assert_not_called()


@pytest.mark.parametrize("learner", ["linear", "lightgbm"])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("mixed", [False, True])
def test_fixed_average_exact_benchmark_arithmetic_and_matched_raw_control(
    learner: str, n_classes: int, mixed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if learner == "lightgbm":
        pytest.importorskip("lightgbm")
    X, y = _data(n_classes)
    if not mixed:
        X = X[["a", "b"]]
    train, test = X.iloc[:90], X.iloc[90:]
    train_y, test_y = y.iloc[:90], y.iloc[90:]
    original_model = tabular.make_model

    def bounded_model(
        X: pd.DataFrame, task: str, learner: str, parameters: dict[str, object], config: dict[str, object]
    ) -> Pipeline:
        model = original_model(X, task, learner, parameters, config)
        if learner == "lightgbm":
            model.steps[-1][1].set_params(n_estimators=4)
        return model

    monkeypatch.setattr(tabular, "make_model", bounded_model)
    monkeypatch.setattr(benchmark, "MAX_CANDIDATES", 20)
    settings: dict[str, object] = {"seed": 42, "threads": 1}
    if learner == "lightgbm":
        reference = oof_blending.fit_fixed_average_bundle_result(
            train, train_y, test, test_y, learner=learner, config=settings
        ).source
    else:
        # The benchmark's public study helper intentionally permits LightGBM only.
        # Its same full-fit/source helpers also provide the linear arithmetic control.
        full, _ = oof_blending._fit_full_procedures(
            train, train_y.to_numpy(), learner, benchmark.PredictionBundleConfig(seed=42, threads=1)
        )
        reference = oof_blending.FixedAverageSource(full["raw"], full["fe"])
    prototype = bounded_model(
        train, "classification", learner, tabular.candidate_parameters(learner)[0], settings
    ).steps[-1][1]
    candidates = (
        [{"C": value} for value in (0.1, 1.0, 10.0)]
        if learner == "linear"
        else [{"num_leaves": value} for value in (7, 15, 31)]
    )
    classifier = FeatureEnsembleClassifier(
        prototype, candidates, mode="raw_fe_average", n_jobs=1, max_candidates=20, stability_repeats=3
    ).fit(train, train_y)
    expected = reference.predict_probabilities(test)
    np.testing.assert_array_equal(classifier.predict_proba(test), expected["average"])
    for actual, wanted in zip(classifier.estimators_, reference.base_estimators, strict=True):
        np.testing.assert_array_equal(actual.predict_proba(test), wanted.predict_proba(test))
        pd.testing.assert_frame_equal(actual._features(test), wanted._features(test), check_exact=True)
        assert actual.model_.steps[0][1].bounds_ == wanted.model_.steps[0][1].bounds_
        rename = {"regularization": "C"} if learner == "linear" else {}
        assert actual.selected_parameters_ == {
            rename.get(key, key): value for key, value in wanted.selected_parameters_.items()
        }
        assert [trial["loss"] for trial in actual.trials_] == [trial["loss"] for trial in wanted.trials_]
        assert actual.feature_names_out_ == wanted.feature_names_out_
        assert actual.inner_split_ == wanted.inner_split_
    raw = FeatureEnsembleClassifier(
        prototype,
        candidates,
        mode="raw_fe_average",
        engineered=False,
        n_jobs=1,
        max_candidates=20,
        stability_repeats=3,
    ).fit(train, train_y)
    np.testing.assert_array_equal(raw.predict_proba(test), expected["raw"])
    np.testing.assert_array_equal(raw.predict_proba(test), classifier.estimators_[0].predict_proba(test))


def test_fixed_average_class_alignment_and_no_extra_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _data(3)
    classifier = _classifier("raw_fe_average", True).fit(X, y)
    raw, fe = classifier.estimators_
    raw_probability = raw.predict_proba(X.iloc[:7])
    fe_probability = fe.predict_proba(X.iloc[:7])
    expected = 0.5 * raw_probability + 0.5 * fe_probability
    monkeypatch.setattr(fe, "classes_", fe.classes_[::-1])
    monkeypatch.setattr(fe, "predict_proba", lambda frame: fe_probability[:, ::-1])
    np.testing.assert_array_equal(classifier.predict_proba(X.iloc[:7]), expected)
    monkeypatch.setattr(fe, "predict_proba", lambda frame: np.full((len(frame), 3), 0.1))
    with pytest.raises(ValueError, match="sum to one"):
        classifier.predict_proba(X.iloc[:7])


def test_fixed_average_atomic_refit_set_params_and_caller_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _data()
    classifier = _classifier("raw_fe_average", True)
    prototype = classifier.estimator
    candidates = classifier.parameter_candidates
    prototype_parameters = prototype.get_params()
    original_candidates = pickle.dumps(candidates)
    cloned = clone(classifier)
    assert cloned.mode == "raw_fe_average" and cloned.estimator is not prototype
    classifier.fit(X, y)
    assert prototype.get_params() == prototype_parameters
    assert not hasattr(prototype, "classes_")
    assert pickle.dumps(candidates) == original_candidates
    original_fit = NestedBaseClassifier.fit

    def fail_fe(self: NestedBaseClassifier, X: pd.DataFrame, y: np.ndarray) -> NestedBaseClassifier:
        if self.engineered:
            raise RuntimeError("synthetic FE member failure")
        return original_fit(self, X, y)

    monkeypatch.setattr(NestedBaseClassifier, "fit", fail_fe)
    with pytest.raises(RuntimeError, match="FE member failure"):
        classifier.fit(X, y)
    for attribute in ("estimators_", "classes_", "weights_", "member_roles_", "member_reports_"):
        assert not hasattr(classifier, attribute)
    classifier.set_params(engineered=False)
    classifier.fit(X, y)
    assert classifier.weights_ == (1.0,) and classifier.member_roles_ == ("raw",)
    classifier.set_params(mode="single")
    assert not hasattr(classifier, "weights_")
    classifier.fit(X, y)
    assert not hasattr(classifier, "weights_")
    assert "weights" not in classifier.diagnostics()
    with pytest.raises(ValueError, match="Invalid parameter"):
        classifier.set_params(unknown_parameter=True)
    assert not hasattr(classifier, "estimators_")


@pytest.mark.parametrize("n_classes", [2, 3])
def test_fixed_average_target_encoding_remains_member_local(n_classes: int) -> None:
    X, y = _data(n_classes)
    classifier = _classifier("raw_fe_average", True).set_params(categorical_encoding="target").fit(X, y)
    assert [member.engineered for member in classifier.estimators_] == [False, True]
    encoders = [member.model_.steps[1][1].named_transformers_["categorical"] for member in classifier.estimators_]
    assert encoders[0] is not encoders[1]
    for encoder in encoders:
        assert isinstance(encoder, CrossFittedTargetEncoder)
        assert encoder.get_encoding_report()["n_train_rows"] == len(X)
    raw, fe = [member.predict_proba(X.iloc[:9]) for member in classifier.estimators_]
    np.testing.assert_array_equal(classifier.predict_proba(X.iloc[:9]), 0.5 * raw + 0.5 * fe)


@pytest.fixture
def frozen_references() -> dict:
    directory = os.environ.get("FEATCOPILOT_CLASSIFICATION_REFERENCE")
    if not directory:
        pytest.skip("Set FEATCOPILOT_CLASSIFICATION_REFERENCE to verified frozen synthetic references")
    root = Path(directory)
    manifest = json.loads((root / "reference_manifest.json").read_text())
    assert manifest["source_archive_sha256"] == FROZEN_ARCHIVE_SHA256
    if sklearn.__version__ != manifest["sklearn"]:
        pytest.skip("Frozen exact parity requires its pinned sklearn runtime")
    content = (root / "references.pkl").read_bytes()
    assert hashlib.sha256(content).hexdigest() == manifest["references_sha256"]
    references = pickle.loads(content)
    assert isinstance(references, dict)
    return references


def _assert_model_state(model: Pipeline, features: pd.DataFrame, expected: dict) -> None:
    pd.testing.assert_frame_equal(features, expected["features"], check_exact=True)
    np.testing.assert_array_equal(model.predict_proba(features), expected["probability"])
    np.testing.assert_array_equal(model.classes_, expected["classes"])
    assert model.steps[0][1].bounds_ == expected["bounds"]
    if "imputer" in expected:
        numeric = model.steps[1][1].named_transformers_["numeric"]
        np.testing.assert_array_equal(numeric.steps[0][1].statistics_, expected["imputer"])
        np.testing.assert_array_equal(numeric.steps[1][1].mean_, expected["mean"])
        np.testing.assert_array_equal(numeric.steps[1][1].scale_, expected["scale"])


def _assert_base_state(base: NestedBaseClassifier, X: pd.DataFrame, expected: dict, learner: str) -> None:
    _assert_model_state(base.model_, base._features(X), expected)
    rename = (
        {"regularization": "C"} if learner == "linear" and not isinstance(base, benchmark.CompleteBaseProcedure) else {}
    )
    expected_parameters = {rename.get(key, key): value for key, value in expected["parameters"].items()}
    assert base.selected_parameters_ == expected_parameters
    expected_trials = [
        {**trial, "parameters": {rename.get(key, key): value for key, value in trial["parameters"].items()}}
        for trial in expected["trials"]
    ]
    assert base.trials_ == expected_trials
    assert base.inner_split_ == expected["inner_split"]
    assert [
        (feature.name, feature.transformation, feature.source_columns, feature.metadata)
        for feature in base.get_feature_plan()
    ] == expected["feature_plan"]


@pytest.mark.parametrize("case", REFERENCE_CASES)
def test_exact_frozen_old_base_single_plain85_and_cv3_parity(
    case: str, frozen_references: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = frozen_references[case]
    learner, n_classes, mixed, engineered_text = case.split("-")
    engineered = engineered_text == "True"
    train, y, test, test_y = reference["data"]
    original_factory = tabular.make_model

    def bounded_model(X: pd.DataFrame, task: str, learner: str, parameters: dict, config: dict) -> Pipeline:
        model = original_factory(X, task, learner, parameters, config)
        if learner == "lightgbm":
            model.steps[-1][1].set_params(n_estimators=4)
        elif learner == "catboost":
            model.steps[-1][1].set_params(iterations=4)
        return model

    monkeypatch.setattr(tabular, "make_model", bounded_model)
    monkeypatch.setattr(benchmark, "MAX_CANDIDATES", 20)
    config = {"seed": 42, "threads": 1, "stability_repeats": 1}
    old_adapter = benchmark.CompleteBaseProcedure(learner=learner, config=config, engineered=engineered).fit(train, y)
    _assert_base_state(old_adapter, test, reference["base"], learner)
    for mode in ("cv3", "subsample85"):
        bundle = benchmark.fit_prediction_bundle_result(
            train,
            y,
            test,
            test_y,
            task="classification",
            learner=learner,
            config={**config, "ensemble_sampling": mode},
            engineered=engineered,
        )
        np.testing.assert_array_equal(bundle.source.predict_plain_proba(test), reference[mode]["plain"])
        np.testing.assert_array_equal(bundle.source.predict_sigmoid_proba(test), reference[mode]["calibrated"])
        for actual, expected in zip(bundle.source.calibrated_model.cv, reference[mode]["splits"], strict=True):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
        for member, expected in zip(bundle.source.base_estimators, reference[mode]["members"], strict=True):
            _assert_base_state(member, test, expected, learner)
    if learner == "catboost":
        return
    estimator = bounded_model(train, "classification", learner, tabular.candidate_parameters(learner)[0], config).steps[
        -1
    ][1]
    candidates = (
        [{"C": value} for value in (0.1, 1.0, 10.0)]
        if learner == "linear"
        else [{"num_leaves": value} for value in (7, 15, 31)]
    )
    single = FeatureEnsembleClassifier(
        estimator, candidates, mode="single", engineered=engineered, n_jobs=1, max_candidates=20, stability_repeats=1
    ).fit(train, y)
    single_expected = {
        **reference["base"],
        "trials": [
            {**trial, "loss": loss}
            for trial, loss in zip(reference["base"]["trials"], reference["single_trial_losses"], strict=True)
        ],
    }
    _assert_base_state(single.estimators_[0], test, single_expected, learner)
    _assert_model_state(single.estimators_[0].model_, single.estimators_[0]._features(test), reference["single"])
    np.testing.assert_array_equal(single.predict_proba(test), reference["single"]["probability"])
    assert [trial["loss"] for trial in single.estimators_[0].trials_] == reference["single_trial_losses"]
    bags = FeatureEnsembleClassifier(
        estimator,
        candidates,
        mode="subsample85",
        engineered=engineered,
        n_jobs=1,
        max_candidates=20,
        stability_repeats=1,
    ).fit(train, y)
    np.testing.assert_array_equal(bags.predict_proba(test), reference["subsample85"]["plain"])
    for member, expected in zip(bags.estimators_, reference["subsample85"]["members"], strict=True):
        _assert_base_state(member, test, expected, learner)

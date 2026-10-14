"""Synthetic-only checks for the shared-fit prediction-family benchmark."""

from __future__ import annotations

import hashlib
import json
import pickle
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from benchmarks import prediction_ensemble as ensemble
from benchmarks import tabular
from featcopilot import classification
from featcopilot.core.feature import Feature, FeatureSet
from featcopilot.engines.predictive import PredictiveTabularEngine, PredictiveTabularEngineConfig


def _data(n_classes: int = 2, mixed: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(391)
    labels = np.arange(180) % n_classes
    X = pd.DataFrame(
        {"a": labels + rng.normal(scale=0.7, size=180), "b": rng.normal(size=180)},
        index=np.arange(180) + 1000,
    )
    if mixed:
        X["category"] = pd.Categorical(["private-vocabulary-one", "private-vocabulary-two", None, "missing:"] * 45)
        X["flag"] = np.arange(180) % 2 == 0
        X.loc[X.index[::13], "a"] = np.nan
    y = pd.Series([f"class-{value}" for value in labels], index=X.index)
    train, test = X.iloc[:144].copy(), X.iloc[144:].copy()
    if mixed:
        test["category"] = ["unseen-category", None, "private-vocabulary-one"] * 12
        test.loc[test.index[:5], "a"] = [1e300, -1e300, np.nan, np.inf, -np.inf]
    return train, y.iloc[:144].copy(), test, y.iloc[144:].copy()


def _bundle(
    data: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    *,
    learner: str = "linear",
    engineered: bool = False,
    ensemble_sampling: ensemble.EnsembleSampling | None = None,
) -> ensemble.PredictionBundleResult:
    config: dict[str, object] = {"seed": 42, "threads": 1, "stability_repeats": 1}
    if ensemble_sampling is not None:
        config["ensemble_sampling"] = ensemble_sampling
    return ensemble.fit_prediction_bundle_result(
        *data,
        task="classification",
        learner=learner,
        config=config,
        engineered=engineered,
    )


@pytest.fixture
def bounded_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound integration cost while retaining real estimators and three settings."""
    original_model = tabular.make_model

    def small_model(
        train: pd.DataFrame, task: str, learner: str, parameters: dict[str, object], config: dict[str, object]
    ) -> Pipeline:
        model = original_model(train, task, learner, parameters, config)
        if learner == "catboost":
            model.steps[-1][1].set_params(iterations=4)
        elif learner in ("lightgbm", "xgboost"):
            model.steps[-1][1].set_params(n_estimators=4)
        return model

    monkeypatch.setattr(tabular, "make_model", small_model)
    monkeypatch.setattr(ensemble, "MAX_CANDIDATES", 20)


@pytest.mark.parametrize("learner", ensemble.LEARNERS)
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("ensemble_sampling", ["cv3", "subsample85"])
def test_real_classifiers_binary_multiclass_mixed_inference(
    learner: str,
    n_classes: int,
    engineered: bool,
    mixed: bool,
    ensemble_sampling: ensemble.EnsembleSampling,
    bounded_models: None,
) -> None:
    if learner != "linear":
        pytest.importorskip(learner)
    data = _data(n_classes=n_classes, mixed=mixed)
    original_train = data[0].copy(deep=True)
    result = _bundle(data, learner=learner, engineered=engineered, ensemble_sampling=ensemble_sampling)
    source = result.source
    test = data[2]
    pd.testing.assert_frame_equal(data[0], original_train)
    expected_response = "decision_function" if learner == "linear" else "predict_proba"
    for base in source.base_estimators:
        assert is_classifier(base)
        assert base.predict(test).shape == (len(test),)
        assert base.n_features_in_ == data[0].shape[1]
        assert base.raw_schema_.empty
        assert list(base.raw_schema_.columns) == list(data[0].columns)
        assert len(base.feature_names_out_) <= data[0].shape[1] + 50
        assert base.model_.n_features_in_ == len(base.feature_names_out_)
        assert hasattr(base, "decision_function") == (learner == "linear")
        assert base.diagnostics()["response_method"] == expected_response
        if learner == "linear":
            np.testing.assert_allclose(
                base.decision_function(test), base.model_.decision_function(base._features(test))
            )
        if learner == "catboost":
            native_prediction = base.model_.predict(base._features(test))
            assert native_prediction.shape in ((len(test),), (len(test), 1))
        if engineered:
            assert base.engineer_ is not None
            assert base.engineer_.config.include_frequency is True
            assert base.engineer_.config.importance_resamples == 1
            assert base.engineer_.config.selection_rule == "paired_gain"
    for predict in (source.predict_plain_proba, source.predict_sigmoid_proba):
        probability = predict(test)
        assert probability.shape == (len(test), n_classes)
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=2e-15, rtol=0)
        batch = np.vstack([predict(test.iloc[:1]), predict(test.iloc[1:9]), predict(test.iloc[9:])])
        np.testing.assert_allclose(batch, probability, atol=1e-12, rtol=1e-12)
    restored = pickle.loads(pickle.dumps(source))
    np.testing.assert_allclose(restored.predict_plain_proba(test), source.predict_plain_proba(test))
    np.testing.assert_allclose(restored.predict_sigmoid_proba(test), source.predict_sigmoid_proba(test))
    np.testing.assert_allclose(
        source.predict_sigmoid_proba(test), source.calibrated_model.predict_proba(test), atol=2e-15, rtol=1e-14
    )
    serialized = json.dumps(result.to_report(), allow_nan=False)
    for private_value in ("private-vocabulary-one", "private-vocabulary-two", "unseen-category", "frequency_state"):
        assert private_value not in serialized
    for arm in result.metrics.values():
        assert {"log_loss", "loss", "accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "brier_sum"} <= arm.keys()
        assert all(np.isfinite(value) for value in arm.values())


def test_fitted_frequency_vocabulary_and_formula_are_hash_only(
    bounded_models: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, y, _, _ = _data(mixed=True)
    base = ensemble.CompleteBaseProcedure(config={"threads": 1, "stability_repeats": 1}, engineered=True).fit(train, y)
    assert base.engineer_ is not None
    private_value = "PRIVATE-FREQUENCY-VOCABULARY"
    private_features = FeatureSet(
        [
            Feature(
                name="frequency_digest",
                source_columns=["category"],
                transformation="fitted_frequency_v1",
                code=f"frozen_mapping[{private_value!r}]",
                metadata={"frequency_state": {"mapping": {private_value: 0.125}}},
            )
        ]
    )
    monkeypatch.setattr(base.engineer_, "get_feature_set", lambda: private_features)
    first = base.diagnostics()
    assert private_value not in json.dumps(first, allow_nan=False)
    private_features["frequency_digest"].metadata["frequency_state"]["mapping"][private_value] = 0.25
    second = base.diagnostics()
    assert first["fitted_state_sha256"] != second["fitted_state_sha256"]
    assert first["formula_sha256"] == second["formula_sha256"]
    assert private_value not in json.dumps(second, allow_nan=False)


@pytest.mark.parametrize("ensemble_sampling", ["cv3", "subsample85"])
def test_exact_shared_fit_objects_mean_and_no_plain_refit(
    ensemble_sampling: ensemble.EnsembleSampling, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _data()
    fit_calls: list[ensemble.CompleteBaseProcedure] = []
    pipeline_rows: list[int] = []
    original_fit = ensemble.CompleteBaseProcedure.fit
    original_pipeline_fit = Pipeline.fit

    def counted_fit(
        self: ensemble.CompleteBaseProcedure, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> ensemble.CompleteBaseProcedure:
        fit_calls.append(self)
        return original_fit(self, X, y)

    def counted_pipeline_fit(self: Pipeline, X: pd.DataFrame, y: np.ndarray, **kwargs: object) -> Pipeline:
        pipeline_rows.append(len(X))
        return original_pipeline_fit(self, X, y, **kwargs)

    monkeypatch.setattr(ensemble.CompleteBaseProcedure, "fit", counted_fit)
    monkeypatch.setattr(Pipeline, "fit", counted_pipeline_fit)
    result = _bundle(data, ensemble_sampling=ensemble_sampling)
    source = result.source
    assert len(fit_calls) == 3
    assert len(pipeline_rows) == 12
    assert pipeline_rows == ([72, 72, 72, 96] if ensemble_sampling == "cv3" else [91, 91, 91, 122]) * 3
    assert len(data[0]) not in pipeline_rows
    for index, pair in enumerate(source.calibrated_model.calibrated_classifiers_):
        assert source.base_estimators[index] is pair.estimator is fit_calls[index]
    expected = np.mean([base.predict_proba(data[2]) for base in source.base_estimators], axis=0)
    np.testing.assert_allclose(source.predict_plain_proba(data[2]), expected, rtol=0, atol=2e-16)
    source.predict_plain_proba(data[2])
    source.predict_sigmoid_proba(data[2])
    assert len(fit_calls) == 3
    assert len(pipeline_rows) == 12
    identities = result.diagnostics["plain_fit_identity_sha256"]
    assert identities == result.diagnostics["sigmoid_fit_identity_sha256"]
    assert isinstance(identities, list) and len(set(identities)) == 3
    assert identities == [base.fit_identity_sha256_ for base in fit_calls]
    assert source.calibrated_model.ensemble is True
    assert source.calibrated_model.method == "sigmoid"
    assert source.calibrated_model.n_jobs == 1
    assert isinstance(source.calibrated_model.cv, list)


@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("ensemble_sampling", ["cv3", "subsample85"])
def test_fixed_partition_calibration_firewall(
    engineered: bool,
    ensemble_sampling: ensemble.EnsembleSampling,
    bounded_models: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, y, test, test_y = _data(mixed=True)
    train["category"] = train.category.astype(object)
    splits = ensemble.make_ensemble_splits(y, 42, ensemble_sampling=ensemble_sampling)
    monkeypatch.setattr(ensemble, "make_ensemble_splits", lambda *args, **kwargs: splits)
    before = _bundle((train, y, test, test_y), engineered=engineered, ensemble_sampling=ensemble_sampling)
    changed_X, changed_y = train.copy(), y.copy()
    calibration = splits[0][1]
    changed_X.iloc[calibration, 0] = np.arange(len(calibration)) * 1e20
    changed_X.iloc[calibration, 2] = "private-perturbed-calibration-vocabulary"
    changed_y.iloc[calibration] = np.roll(changed_y.iloc[calibration].to_numpy(), 1)
    after = _bundle((changed_X, changed_y, test, test_y), engineered=engineered, ensemble_sampling=ensemble_sampling)
    left, right = before.source.base_estimators[0], after.source.base_estimators[0]
    left_schema = left.diagnostics().copy()
    right_schema = right.diagnostics().copy()
    for key in (
        "fit_identity_sha256",
        "fit_seconds",
        "fe_fit_seconds",
        "model_fit_seconds",
        "tuning_and_orchestration_seconds",
    ):
        left_schema.pop(key)
        right_schema.pop(key)
    assert left_schema == right_schema
    np.testing.assert_array_equal(left.predict_proba(test), right.predict_proba(test))
    assert pickle.dumps(left.model_) == pickle.dumps(right.model_)
    assert left.selected_parameters_ == right.selected_parameters_
    assert left.trials_ == right.trials_
    assert (
        before.source.base_estimators[1].diagnostics()["fitted_state_sha256"]
        != after.source.base_estimators[1].diagnostics()["fitted_state_sha256"]
    )


@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("ensemble_sampling", ["cv3", "subsample85"])
def test_outer_test_features_and_labels_do_not_affect_any_fit(
    engineered: bool, ensemble_sampling: ensemble.EnsembleSampling, bounded_models: None
) -> None:
    train, y, test, test_y = _data(mixed=True)
    before = _bundle((train, y, test, test_y), engineered=engineered, ensemble_sampling=ensemble_sampling)
    changed_test = test.copy()
    changed_test["a"] = 1e200
    changed_labels = pd.Series(np.roll(test_y.to_numpy(), 1), index=test_y.index)
    after = _bundle(
        (train, y, changed_test, changed_labels), engineered=engineered, ensemble_sampling=ensemble_sampling
    )
    for left, right in zip(before.source.base_estimators, after.source.base_estimators, strict=True):
        assert left.diagnostics()["fitted_state_sha256"] == right.diagnostics()["fitted_state_sha256"]
        assert left.trials_ == right.trials_
        assert left.selected_parameters_ == right.selected_parameters_
    assert before.metrics != after.metrics


def test_clone_refit_and_failed_refit_clear_state() -> None:
    data = _data()
    settings = {"seed": 42, "threads": 1, "stability_repeats": 1}
    base = ensemble.CompleteBaseProcedure(config=settings)
    assert base.config is settings
    assert not hasattr(base, "decision_function")
    unfitted = clone(base)
    assert unfitted.get_params() == base.get_params()
    base.fit(data[0], data[1])
    assert hasattr(base, "decision_function")
    cloned = clone(base)
    assert not hasattr(cloned, "model_")
    with pytest.raises(NotFittedError):
        cloned.predict_proba(data[2])
    identity = base.fit_identity_sha256_
    renamed = data[0].rename(columns={"a": "renamed"})
    base.fit(renamed, data[1])
    assert base.fit_identity_sha256_ != identity
    assert base.feature_names_out_ == ["renamed", "b"]
    with pytest.raises(ValueError, match="schema"):
        base.predict_proba(data[2])
    with pytest.raises(ValueError, match="two classes"):
        base.fit(renamed, np.zeros(len(renamed), dtype=int))
    assert not hasattr(base, "classes_")
    assert not hasattr(base, "model_")
    assert not hasattr(base, "decision_function")


@pytest.mark.parametrize("stability_repeats", [None, 1, 2])
def test_declared_aligned_engine_parameters_and_independent_winning_refit(
    stability_repeats: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, y, test, test_y = _data(mixed=True)
    calls: list[PredictiveTabularEngineConfig] = []
    original_fit = PredictiveTabularEngine.fit

    def recorded_fit(
        self: PredictiveTabularEngine,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: object,
    ) -> PredictiveTabularEngine:
        assert all(value is None for value in kwargs.values())
        calls.append(self.config.model_copy())
        # Keep production declarations observable, but bound real test work.
        self.config = self.config.model_copy(update={"max_candidates": 20})
        return original_fit(self, X, y)

    monkeypatch.setattr(PredictiveTabularEngine, "fit", recorded_fit)
    settings: dict[str, object] = {"threads": 1}
    if stability_repeats is not None:
        settings["stability_repeats"] = stability_repeats
    base = ensemble.CompleteBaseProcedure(config=settings, engineered=True).fit(train, y)
    assert len(calls) == 4
    for call in calls:
        assert call.max_features == train.shape[1] + 50
        assert call.max_candidates == 1000
        assert call.candidate_set == "extended"
        assert call.include_frequency is True
        assert call.importance_resamples == 1
        assert call.screening == "importance"
        assert call.selection_rule == "paired_gain"
        assert call.numeric_clip_quantiles == (0.0, 1.0)
        assert call.random_state == 42
        assert call.stability_repeats == (3 if stability_repeats is None else stability_repeats)
        assert not isinstance(call.estimator, Pipeline)
    regularization = [call.estimator.get_params()["C"] for call in calls if call.estimator is not None]
    assert regularization[:3] == [0.1, 1.0, 10.0]
    assert regularization[3] == base.selected_parameters_["regularization"]
    assert len({id(call.estimator) for call in calls}) == 4
    assert base.raw_schema_.shape == (0, train.shape[1])


def test_catboost_selection_bare_learner_has_no_cat_features(
    bounded_models: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("catboost")
    original_fit = PredictiveTabularEngine.fit
    selection_parameters: list[dict[str, object]] = []

    def recorded_fit(
        self: PredictiveTabularEngine,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: object,
    ) -> PredictiveTabularEngine:
        assert all(value is None for value in kwargs.values())
        estimator = self.config.estimator
        assert estimator is not None
        selection_parameters.append(estimator.get_params())
        return original_fit(self, X, y)

    monkeypatch.setattr(PredictiveTabularEngine, "fit", recorded_fit)
    _bundle(_data(mixed=True), learner="catboost", engineered=True)
    assert len(selection_parameters) == 12
    assert all("cat_features" not in parameters for parameters in selection_parameters)


def test_deterministic_search_ties_choose_first_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    train, y, _, _ = _data()
    monkeypatch.setattr(classification, "log_loss", lambda *args, **kwargs: 0.5)
    base = ensemble.CompleteBaseProcedure(config={"threads": 1}).fit(train, y)
    assert base.selected_parameters_ == tabular.candidate_parameters("linear")[0]
    assert [trial["loss"] for trial in base.trials_] == [0.5, 0.5, 0.5]


@pytest.mark.parametrize("error_type", [pickle.PicklingError, ValueError, RuntimeError])
def test_benchmark_fitted_state_fingerprint_remains_strict(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    train, y, _, _ = _data()
    base = ensemble.CompleteBaseProcedure(config={"threads": 1}).fit(train, y)
    assert isinstance(base.diagnostics()["fitted_state_sha256"], str)
    original_dumps = classification.pickle.dumps
    error = error_type("synthetic unsupported state")

    def unavailable(value: object, **kwargs) -> bytes:
        if isinstance(value, tuple) and len(value) == 4 and isinstance(value[0], Pipeline):
            raise error
        return original_dumps(value, **kwargs)

    monkeypatch.setattr(classification.pickle, "dumps", unavailable)
    with pytest.raises(error_type, match="unsupported state") as caught:
        base.diagnostics()
    assert caught.value is error


@pytest.mark.parametrize("counts", [(2, 20), (3, 3), (3, 3, 3), (3, 3, 3, 3)])
def test_class_support_rejects_before_any_base_fit(counts: tuple[int, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    y = np.repeat(np.arange(len(counts)), counts)
    fit = Mock(side_effect=AssertionError("No fit may start before class support validation"))
    monkeypatch.setattr(ensemble.CompleteBaseProcedure, "fit", fit)
    with pytest.raises(ValueError, match="class|inner split"):
        ensemble.fit_prediction_bundle(
            pd.DataFrame({"a": np.arange(len(y), dtype=float)}),
            y,
            pd.DataFrame({"a": np.arange(len(y), dtype=float)}),
            y,
            task="classification",
            learner="linear",
            config={"threads": 1},
            engineered=False,
        )
    fit.assert_not_called()


def test_split_seed_hashes_and_every_partition_class_coverage() -> None:
    train, y, _, _ = _data(n_classes=3)
    first = ensemble.make_ensemble_splits(y, 42)
    second = ensemble.make_ensemble_splits(y, 42)
    from sklearn.model_selection import StratifiedKFold, train_test_split

    expected = list(StratifiedKFold(3, shuffle=True, random_state=42 + 32452843).split(train, y))
    for index, ((train_positions, calibration), repeated) in enumerate(zip(first, second, strict=True)):
        np.testing.assert_array_equal(train_positions, repeated[0])
        np.testing.assert_array_equal(calibration, repeated[1])
        np.testing.assert_array_equal(train_positions, expected[index][0])
        np.testing.assert_array_equal(calibration, expected[index][1])
        assert not set(train_positions) & set(calibration)
        assert len(set(train_positions) | set(calibration)) == len(y)
        inner_train, inner_valid = train_test_split(
            np.arange(len(train_positions)), test_size=0.25, stratify=y.iloc[train_positions], random_state=42
        )
        for values in (
            y.iloc[train_positions],
            y.iloc[calibration],
            y.iloc[train_positions[inner_train]],
            y.iloc[train_positions[inner_valid]],
        ):
            assert set(values) == set(y)
    wrapped = ensemble.make_ensemble_splits(y, 2**32 - 1)
    assert len(wrapped) == 3


@pytest.mark.parametrize("learner", ["linear", "lightgbm"])
@pytest.mark.parametrize("engineered", [False, True])
def test_default_and_explicit_cv3_match_legacy_control_exactly(
    learner: str, engineered: bool, bounded_models: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if learner != "linear":
        pytest.importorskip(learner)
    data = _data(mixed=True)
    legacy_splits = list(
        StratifiedKFold(3, shuffle=True, random_state=42 + 32452843).split(np.zeros(len(data[1])), data[1])
    )
    for sampling in ("cv3", "subsample85"):
        clone_config = ensemble.PredictionBundleConfig(threads=1, ensemble_sampling=sampling)
        estimator = ensemble.CompleteBaseProcedure(config=clone_config)
        assert clone(estimator).config == clone_config
    default = _bundle(data, learner=learner, engineered=engineered)
    explicit = _bundle(data, learner=learner, engineered=engineered, ensemble_sampling="cv3")
    monkeypatch.setattr(ensemble, "make_ensemble_splits", lambda *args, **kwargs: legacy_splits)
    legacy = _bundle(data, learner=learner, engineered=engineered)
    for result in (default, explicit):
        assert result.diagnostics["protocol"] == "shared_retained_sigmoid_3fold_v1"
        assert result.diagnostics["ensemble_sampling"] == "cv3"
        assert "sampling_design" not in result.diagnostics
        assert result.metrics == legacy.metrics
        for left, right in zip(result.source.base_estimators, legacy.source.base_estimators, strict=True):
            assert left.selected_parameters_ == right.selected_parameters_
            assert left.trials_ == right.trials_
            assert left.feature_names_out_ == right.feature_names_out_
            assert left.inner_split_ == right.inner_split_
            assert left.diagnostics()["fitted_state_sha256"] == right.diagnostics()["fitted_state_sha256"]
            np.testing.assert_array_equal(left.predict_proba(data[2]), right.predict_proba(data[2]))
        np.testing.assert_array_equal(
            result.source.predict_plain_proba(data[2]), legacy.source.predict_plain_proba(data[2])
        )
        np.testing.assert_array_equal(
            result.source.predict_sigmoid_proba(data[2]), legacy.source.predict_sigmoid_proba(data[2])
        )
        for actual, expected in zip(result.source.calibrated_model.cv, legacy_splits, strict=True):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("seed", [42, 137, 2029, 2**32 - 1, 2**32 - ensemble.ENSEMBLE_SEED_OFFSET - 1])
@pytest.mark.parametrize("n_rows,n_classes", [(144, 2), (149, 3), (41, 2)])
def test_subsample85_matches_holdout_helper_order_sizes_classes_and_seed_wrap(
    seed: int, n_rows: int, n_classes: int
) -> None:
    from sklearn.model_selection import train_test_split

    from featcopilot.utils.splitting import make_holdout_splits

    y = np.arange(n_rows) % n_classes
    expected = make_holdout_splits(
        y, "classification", 3, (seed + ensemble.ENSEMBLE_SEED_OFFSET) % 2**32, test_size=0.15
    )
    actual = ensemble.make_ensemble_splits(y, seed, ensemble_sampling="subsample85")
    repeated = ensemble.make_ensemble_splits(y, seed, ensemble_sampling="subsample85")
    for member, ((train, calibration), wanted, again) in enumerate(zip(actual, expected, repeated, strict=True)):
        consecutive = train_test_split(
            np.arange(n_rows),
            test_size=0.15,
            stratify=y,
            random_state=(seed + ensemble.ENSEMBLE_SEED_OFFSET + member) % 2**32,
        )
        for received, reference, duplicate, scheduled in zip(
            (train, calibration), wanted, again, consecutive, strict=True
        ):
            np.testing.assert_array_equal(received, reference)
            np.testing.assert_array_equal(received, duplicate)
            np.testing.assert_array_equal(received, scheduled)
            assert len(received) == len(set(received))
            assert set(y[received]) == set(y)
        assert len(calibration) == int(np.ceil(n_rows * 0.15))
        assert len(train) == n_rows - len(calibration)
        assert not np.array_equal(train, np.sort(train))
        assert not set(train) & set(calibration)
        assert set(train) | set(calibration) == set(range(n_rows))
        inner_train, inner_valid = ensemble._inner_split(y[train], seed, "test")
        assert set(y[train[inner_train]]) == set(y[train[inner_valid]]) == set(y)
    assert sum(len(calibration) for _, calibration in actual) < n_rows
    assert all(np.intersect1d(actual[left][0], actual[right][0]).size for left in range(3) for right in range(left))


@pytest.mark.parametrize("mode", ["cv", "subsample", "subsample80", "", 85, True, None, ["subsample85"]])
def test_invalid_sampling_mode_fails_before_fit(mode: object, monkeypatch: pytest.MonkeyPatch) -> None:
    fit = Mock(side_effect=AssertionError("Invalid mode must fail before fitting"))
    monkeypatch.setattr(ensemble.CalibratedClassifierCV, "fit", fit)
    with pytest.raises(ValueError, match="ensemble_sampling"):
        ensemble.fit_prediction_bundle(
            *_data(), task="classification", learner="linear", config={"ensemble_sampling": mode}, engineered=False
        )
    fit.assert_not_called()


@pytest.mark.parametrize("counts", [(1, 40), (2, 2, 2), (141, 3), (30, 1)])
def test_subsample85_insufficient_class_support_is_pre_fit(
    counts: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    y = np.repeat(np.arange(len(counts)), counts)
    X = pd.DataFrame({"a": np.arange(len(y), dtype=float)})
    fit = Mock(side_effect=AssertionError("Insufficient classes must fail before fitting"))
    monkeypatch.setattr(ensemble.CalibratedClassifierCV, "fit", fit)
    with pytest.raises(ValueError, match="class|holdout|inner"):
        ensemble.fit_prediction_bundle(
            X,
            y,
            X,
            y,
            task="classification",
            learner="linear",
            config={"ensemble_sampling": "subsample85"},
            engineered=False,
        )
    fit.assert_not_called()


@pytest.mark.parametrize(
    "malformation", ["duplicate", "overlap", "negative", "out_of_range", "float", "short", "two_pairs"]
)
def test_subsample85_rejects_invalid_positions_without_repair(
    malformation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, y, _, _ = _data()
    splits = ensemble.make_ensemble_splits(y, ensemble_sampling="subsample85")
    train, calibration = splits[0]
    if malformation == "duplicate":
        train[1] = train[0]
    elif malformation == "overlap":
        train[0] = calibration[0]
    elif malformation == "negative":
        train[0] = -1
    elif malformation == "out_of_range":
        train[0] = len(y)
    elif malformation == "float":
        splits[0] = (train.astype(float), calibration)
    elif malformation == "short":
        splits[0] = (train[:-1], calibration)
    else:
        splits.pop()
    helper = Mock(return_value=splits)
    monkeypatch.setattr(classification, "make_holdout_splits", helper)
    with pytest.raises(ValueError, match="positions|three"):
        ensemble.make_ensemble_splits(y, ensemble_sampling="subsample85")
    helper.assert_called_once()


@pytest.mark.parametrize("missing_side", ["base", "calibration", "inner"])
def test_subsample85_checks_all_class_sides_without_retries(missing_side: str, monkeypatch: pytest.MonkeyPatch) -> None:
    y = np.zeros(40, dtype=int)
    train = np.arange(34, dtype=np.int64)
    calibration = np.arange(34, 40, dtype=np.int64)
    if missing_side == "base":
        y[calibration] = 1
    elif missing_side == "calibration":
        y[train[:12]] = 1
    else:
        y[train[0]] = 1
        y[calibration[0]] = 1
    helper = Mock(return_value=[(train, calibration)] * 3)
    fit = Mock(side_effect=AssertionError("Every class check must precede fitting"))
    monkeypatch.setattr(classification, "make_holdout_splits", helper)
    monkeypatch.setattr(ensemble.CalibratedClassifierCV, "fit", fit)
    X = pd.DataFrame({"a": np.arange(len(y), dtype=float)})
    with pytest.raises(ValueError, match="base TRAIN|calibration HOLDOUT|inner split"):
        ensemble.fit_prediction_bundle(
            X,
            y,
            X,
            y,
            task="classification",
            learner="linear",
            config={"ensemble_sampling": "subsample85"},
            engineered=False,
        )
    helper.assert_called_once()
    fit.assert_not_called()


def test_subsample85_does_not_redraw_identical_bags_or_deduplicate_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    train, y, test, test_y = _data()
    train.loc[:, :] = 1.0
    train.index = pd.Index(np.zeros(len(train), dtype=int))
    pair = ensemble.make_ensemble_splits(y, ensemble_sampling="subsample85")[0]
    helper = Mock(return_value=[pair] * 3)
    monkeypatch.setattr(classification, "make_holdout_splits", helper)
    result = _bundle((train, y, test, test_y), ensemble_sampling="subsample85")
    helper.assert_called_once()
    assert result.diagnostics["distinct_train_bag_count"] == 1
    assert result.diagnostics["calibration_coverage_histogram"] == [122, 0, 0, 22]
    assert len({id(base) for base in result.source.base_estimators}) == 3
    for base in result.source.base_estimators:
        assert base.inner_split_["train"]["size"] == 91
        assert base.inner_split_["validation"]["size"] == 31
    members = result.diagnostics["members"]
    assert isinstance(members, list)
    assert all(member["base_train"]["size"] == 122 for member in members)


def test_subsample85_json_masks_fractions_coverage_and_overlap_are_truthful() -> None:
    data = _data(mixed=True)
    config = {
        "threads": 1,
        "ensemble_sampling": "subsample85",
        "seed": 2**32 - ensemble.ENSEMBLE_SEED_OFFSET - 1,
    }
    result = ensemble.fit_prediction_bundle_result(
        *data, task="classification", learner="linear", config=config, engineered=False
    )
    report = result.to_report()
    diagnostics = report["diagnostics"]
    assert diagnostics["protocol"] == "shared_retained_sigmoid_subsample85_v1"
    assert diagnostics["ensemble_sampling"] == "subsample85"
    assert diagnostics["sampling_design"] == "repeated_stratified_holdout"
    assert diagnostics["requested_train_fraction"] == 0.85
    assert diagnostics["requested_calibration_fraction"] == 0.15
    splits = result.source.calibrated_model.cv
    members = diagnostics["members"]
    assert isinstance(members, list)
    assert [member["sampling_seed"] for member in members] == [2**32 - 1, 0, 1]
    n_rows = len(data[0])
    coverage = np.zeros(n_rows, dtype=int)
    for member, (train, calibration) in zip(members, splits, strict=True):
        coverage[calibration] += 1
        for key, positions in (("base_train", train), ("calibration_validation", calibration)):
            component = member[key]
            mask = np.zeros(n_rows, dtype=np.uint8)
            mask[positions] = 1
            assert component["mask_sha256"] == hashlib.sha256(mask.tobytes()).hexdigest()
            assert (
                component["indices_sha256"] == hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest()
            )
            assert component["fraction"] == len(positions) / n_rows
            assert component["size"] == len(positions)
            assert sum(component["class_counts"]) == len(positions)
    histogram = diagnostics["calibration_coverage_histogram"]
    assert histogram == np.bincount(coverage, minlength=4).tolist()
    assert isinstance(histogram, list) and sum(histogram) == n_rows
    assert sum(index * count for index, count in enumerate(histogram)) == 3 * 22
    overlaps = diagnostics["pairwise_overlap"]
    assert isinstance(overlaps, list) and len(overlaps) == 3
    for overlap in overlaps:
        left, right = overlap["members"]
        assert overlap["base_train"] == len(set(splits[left][0]) & set(splits[right][0]))
        assert overlap["calibration_validation"] == len(set(splits[left][1]) & set(splits[right][1]))
    serialized = json.dumps(report, allow_nan=False)
    assert json.loads(serialized) == report
    for forbidden in ("private-vocabulary-one", "private-vocabulary-two", '"indices":', '"mask":', '"source":'):
        assert forbidden not in serialized
    widths = diagnostics["member_output_feature_counts"]
    assert isinstance(widths, list)
    assert diagnostics["max_member_output_features"] == max(widths)
    assert report["shared_cost_not_additive"] is True


@pytest.mark.parametrize(
    "config",
    [
        {"groups": [1, 2]},
        {"timestamps": [1, 2]},
        {"label_end_times": [1, 2]},
        {"group": "entity"},
        {"time": "timestamp"},
        {"label_end": "end"},
        {"split_kind": "group"},
        {"split_strategy": "temporal"},
        {"metadata": {"timestamps": [1, 2]}},
    ],
)
def test_non_iid_metadata_fails_explicitly(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="IID"):
        ensemble.fit_prediction_bundle(
            *_data(), task="classification", learner="linear", config=config, engineered=False
        )


@pytest.mark.parametrize("learner", ["linear", "catboost"])
@pytest.mark.parametrize("encoding", ["target", "unknown", None, True])
def test_calibration_categorical_encoding_override_is_rejected_before_fit(
    learner: str, encoding: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = Mock(side_effect=AssertionError("Unsupported categorical encoding must fail before calibration fitting"))
    monkeypatch.setattr(ensemble.CalibratedClassifierCV, "fit", fit)
    with pytest.raises(ValueError, match="categorical_encoding"):
        ensemble.fit_prediction_bundle(
            *_data(),
            task="classification",
            learner=learner,
            config={"categorical_encoding": encoding},
            engineered=False,
        )
    fit.assert_not_called()


def test_explicit_one_hot_calibration_config_preserves_default() -> None:
    data = _data()
    default = _bundle(data)
    explicit = ensemble.fit_prediction_bundle_result(
        *data,
        task="classification",
        learner="linear",
        config={"categorical_encoding": "one_hot", "threads": 1, "stability_repeats": 1},
        engineered=False,
    )
    assert default.metrics == explicit.metrics
    np.testing.assert_array_equal(
        default.source.predict_plain_proba(data[2]), explicit.source.predict_plain_proba(data[2])
    )
    np.testing.assert_array_equal(
        default.source.predict_sigmoid_proba(data[2]), explicit.source.predict_sigmoid_proba(data[2])
    )


def test_regression_rejected() -> None:
    with pytest.raises(ValueError, match="classification only"):
        ensemble.fit_prediction_bundle(
            *_data(), task="regression", learner="linear", config={"threads": 1}, engineered=False
        )


@pytest.mark.parametrize("key,fixed", [("max_candidates", 1000), ("max_added_features", 50)])
@pytest.mark.parametrize("kind", ["below", "above", "string", "float", "boolean", "none", "array"])
def test_public_api_rejects_budget_overrides_before_any_fit(
    key: str, fixed: int, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid: dict[str, object] = {
        "below": fixed - 1,
        "above": fixed + 1,
        "string": str(fixed),
        "float": float(fixed),
        "boolean": True,
        "none": None,
        "array": [fixed],
    }
    fit = Mock(side_effect=AssertionError("No real fitting may precede budget validation"))
    monkeypatch.setattr(ensemble.CalibratedClassifierCV, "fit", fit)
    with pytest.raises(ValueError, match=key):
        ensemble.fit_prediction_bundle(
            *_data(), task="classification", learner="linear", config={key: invalid[kind]}, engineered=False
        )
    fit.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", True),
        ("seed", 42.0),
        ("seed", "42"),
        ("seed", [42]),
        ("seed", -1),
        ("seed", 2**32),
        ("threads", True),
        ("threads", 1.0),
        ("threads", "1"),
        ("threads", [1]),
        ("threads", 0),
        ("threads", -1),
        ("stability_repeats", True),
        ("stability_repeats", 0),
        ("stability_repeats", 11),
        ("stability_repeats", 100),
        ("stability_repeats", "3"),
        ("stability_repeats", 3.5),
        ("stability_repeats", None),
    ],
)
def test_execution_settings_are_strict_before_any_fit(key: str, value: object, monkeypatch: pytest.MonkeyPatch) -> None:
    fit = Mock(side_effect=AssertionError("No real fitting may precede config validation"))
    monkeypatch.setattr(ensemble.CalibratedClassifierCV, "fit", fit)
    with pytest.raises(ValueError, match=key):
        ensemble.fit_prediction_bundle(
            *_data(), task="classification", learner="linear", config={key: value}, engineered=False
        )
    fit.assert_not_called()


def test_execution_settings_defaults_and_explicit_values() -> None:
    assert ensemble.PredictionBundleConfig().model_dump() == {
        "seed": 42,
        "threads": 8,
        "stability_repeats": 3,
        "split_kind": "iid",
        "ensemble_sampling": "cv3",
    }
    settings = ensemble._config({"seed": 2**32 - 1, "threads": 2, "stability_repeats": 4})
    assert settings.seed == 2**32 - 1
    assert settings.threads == 2
    assert settings.stability_repeats == 4
    assert ensemble.PredictionBundleConfig(stability_repeats=10).stability_repeats == 10


@pytest.mark.parametrize(
    "probability",
    [
        np.array([[0.2, 0.2]]),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 0.0]]),
        np.array([[-1e-16, 1.0]]),
        np.array([[1.0 + 1e-15, 0.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[0, 1]]),
        np.array([0.2, 0.8]),
        np.array([[0.2, 0.3, 0.5]]),
        np.array([[0.5, 0.5 + 1e-10]]),
    ],
)
def test_invalid_probabilities_never_become_success(probability: np.ndarray) -> None:
    with pytest.raises(ValueError, match="probabilit|Probabilit"):
        ensemble.validate_probabilities(probability, 1, 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_legal_roundoff_normalization_and_no_mutation(dtype: type[np.floating]) -> None:
    values = np.array([[0.2, 0.3, 0.5 + np.finfo(dtype).eps]], dtype=dtype)
    before = values.copy()
    actual = ensemble.validate_probabilities(values, 1, 3)
    np.testing.assert_allclose(actual.sum(axis=1), [1.0], atol=2e-16, rtol=0)
    np.testing.assert_array_equal(values, before)
    assert actual.dtype == np.float64


def test_native_probability_failure_is_not_replaced_by_calibrated_success(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data()
    result = _bundle(data)
    base = result.source.base_estimators[0]
    monkeypatch.setattr(base.model_, "predict_proba", lambda X: np.full((len(X), 2), 0.1))
    with pytest.raises(ValueError, match="sum to one"):
        result.source.predict_plain_proba(data[2])


def test_class_alignment_helper_and_adapter_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    aligned = ensemble._aligned_probability(np.array([[0.2, 0.8]]), np.array(["b", "a"]), np.array(["a", "b"]), 1)
    np.testing.assert_array_equal(aligned, [[0.8, 0.2]])
    with pytest.raises(ValueError, match="alignment"):
        ensemble._aligned_probability(np.array([[0.2, 0.8]]), np.array(["b", "c"]), np.array(["a", "b"]), 1)
    data = _data()
    source = _bundle(data).source
    pair = source.calibrated_model.calibrated_classifiers_[0]
    monkeypatch.setattr(pair.estimator, "classes_", np.array(["wrong", "classes"]))
    with pytest.raises(ValueError, match="alignment"):
        source.predict_plain_proba(data[2])
    monkeypatch.undo()
    monkeypatch.setattr(ensemble.sklearn, "__version__", "2.0.0")
    with pytest.raises(RuntimeError, match="adapter version"):
        source.predict_plain_proba(data[2])


def test_schema_guards_and_missing_values() -> None:
    train, y, test, _ = _data(mixed=True)
    base = ensemble.CompleteBaseProcedure(config={"threads": 1}).fit(train, y)
    for invalid in (test.drop(columns="b"), test.assign(extra=0), test[list(reversed(test.columns))]):
        with pytest.raises(ValueError, match="schema"):
            base.predict_proba(invalid)
    with pytest.raises(TypeError, match="numeric"):
        base.predict_proba(test.assign(a="not-numeric"))
    with pytest.raises(TypeError, match="boolean"):
        base.predict_proba(test.assign(flag=1))
    with pytest.raises(TypeError, match="DataFrame"):
        base.predict_proba(test.to_numpy())
    missing = test.copy()
    missing["a"] = np.nan
    missing["b"] = np.nan
    missing["category"] = None
    assert np.isfinite(base.predict_proba(missing)).all()


def test_report_member_widths_timing_and_binary_brier_scale() -> None:
    data = _data()
    result = _bundle(data)
    probability = result.source.predict_plain_proba(data[2])
    labels = np.asarray(result.source.calibrated_model.classes_)
    binary_target = np.asarray(data[3]) == labels[1]
    assert result.metrics["plain"]["brier_sum"] == pytest.approx(
        2 * float(np.mean((probability[:, 1] - binary_target) ** 2))
    )
    widths = [len(base.feature_names_out_) for base in result.source.base_estimators]
    assert result.diagnostics["member_feature_counts"] == widths
    assert result.diagnostics["max_member_feature_count"] == max(widths)
    assert "union" not in json.dumps(result.diagnostics)
    timing = result.timing
    assert timing["sum_base_fit_seconds"] == sum(base.fit_seconds_ for base in result.source.base_estimators)
    assert timing["fit_wall_seconds"] >= timing["sum_base_fit_seconds"]
    assert timing["calibration_orchestration_seconds"] == pytest.approx(
        timing["fit_wall_seconds"] - timing["sum_base_fit_seconds"]
    )
    assert timing["plain_training_estimate_seconds"] == timing["sum_base_fit_seconds"] + timing["split_seconds"]
    assert timing["calibrated_training_seconds"] == timing["fit_wall_seconds"]
    assert timing["base_fe_seconds"] == 0.0
    assert timing["base_model_fit_seconds"] > 0.0
    assert timing["base_fit_seconds"] == pytest.approx(
        timing["base_fe_seconds"] + timing["base_model_fit_seconds"] + timing["base_tuning_and_orchestration_seconds"]
    )
    assert timing["physical_bundle_wall_seconds"] >= (
        timing["fit_wall_seconds"] + timing["plain_score_seconds"] + timing["sigmoid_score_seconds"]
    )
    assert all(value >= 0 for value in timing.values())
    report = result.to_report()
    assert set(report) == {
        "plain_metrics",
        "calibrated_metrics",
        "diagnostics",
        "timing",
        "shared_cost_not_additive",
    }
    assert report["shared_cost_not_additive"] is True
    assert "source" not in json.dumps(report)
    members = result.diagnostics["members"]
    assert isinstance(members, list)
    assert len(members) == 3
    for member in members:
        assert member["base_train"]["size"] == 96
        assert member["calibration_validation"]["size"] == 48
        assert len(member["base_train"]["indices_sha256"]) == 64
        assert sum(member["base_train"]["class_counts"]) == 96
    report["plain_metrics"] = {}
    assert result.metrics


def test_public_supervisor_api_is_json_only_with_frozen_field_names() -> None:
    data = _data(mixed=True)
    report = ensemble.fit_prediction_bundle(
        *data,
        task="classification",
        learner="linear",
        config={"seed": 42, "threads": 1, "max_candidates": 1000, "max_added_features": 50},
        engineered=False,
    )
    assert isinstance(report, dict)
    assert set(report) == {
        "plain_metrics",
        "calibrated_metrics",
        "timing",
        "diagnostics",
        "shared_cost_not_additive",
    }
    serialized = json.dumps(report, allow_nan=False)
    assert json.loads(serialized) == report
    for forbidden in ("private-vocabulary-one", "private-vocabulary-two", "calibrated_model", "source"):
        assert forbidden not in serialized
    assert report["shared_cost_not_additive"] is True
    assert "brier_sum" in report["plain_metrics"]
    assert "brier_sum" in report["calibrated_metrics"]
    diagnostics = report["diagnostics"]
    assert diagnostics["n_input_features"] == data[0].shape[1]
    assert diagnostics["member_output_feature_counts"] == [data[0].shape[1]] * 3
    assert diagnostics["max_member_output_features"] == data[0].shape[1]
    timing = report["timing"]
    required = {
        "fit_seconds",
        "base_fit_seconds",
        "calibration_and_orchestration_seconds",
        "plain_predict_seconds",
        "calibrated_predict_seconds",
    }
    assert required <= timing.keys()
    assert timing["fit_seconds"] == timing["fit_wall_seconds"]
    assert timing["base_fit_seconds"] == timing["sum_base_fit_seconds"]
    assert timing["calibration_and_orchestration_seconds"] == pytest.approx(
        timing["fit_seconds"] - timing["base_fit_seconds"]
    )
    assert timing["plain_predict_seconds"] <= timing["plain_score_seconds"]
    assert timing["calibrated_predict_seconds"] <= timing["sigmoid_score_seconds"]


def test_fe_stage_timing_includes_all_search_trials_and_winning_refit(
    bounded_models: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    intervals: list[float] = []
    original_fit_transform = PredictiveTabularEngine.fit_transform

    def recorded_fit_transform(
        self: PredictiveTabularEngine,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: object,
    ) -> pd.DataFrame:
        assert all(value is None for value in kwargs.values())
        started = ensemble.time.perf_counter()
        features = original_fit_transform(self, X, y)
        intervals.append(ensemble.time.perf_counter() - started)
        return features

    monkeypatch.setattr(PredictiveTabularEngine, "fit_transform", recorded_fit_transform)
    result = _bundle(_data(mixed=True), engineered=True)
    assert len(intervals) == 3 * (3 + 1)
    timing = result.timing
    bases = result.source.base_estimators
    assert timing["base_fe_seconds"] == sum(base.fe_fit_seconds_ for base in bases)
    assert timing["base_model_fit_seconds"] == sum(base.model_fit_seconds_ for base in bases)
    assert timing["base_fe_seconds"] >= sum(intervals) > 0.0
    assert timing["base_fit_seconds"] == pytest.approx(
        timing["base_fe_seconds"] + timing["base_model_fit_seconds"] + timing["base_tuning_and_orchestration_seconds"]
    )
    for index, base in enumerate(bases):
        assert base.fe_fit_seconds_ >= sum(intervals[index * 4 : (index + 1) * 4]) > 0.0
        assert base.diagnostics()["fe_fit_seconds"] == base.fe_fit_seconds_
        assert base.fit_seconds_ >= base.fe_fit_seconds_ + base.model_fit_seconds_


def test_hpo_imputation_scaling_and_caps_are_fit_on_inner_train_only(monkeypatch: pytest.MonkeyPatch) -> None:
    train, y, _, _ = _data(mixed=True)
    fitted: list[tuple[pd.DataFrame, Pipeline]] = []
    original_fit = Pipeline.fit

    def record_fit(self: Pipeline, X: pd.DataFrame, target: np.ndarray, **kwargs: object) -> Pipeline:
        returned = original_fit(self, X, target, **kwargs)
        fitted.append((X.copy(), self))
        return returned

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    base = ensemble.CompleteBaseProcedure(config={"threads": 1}).fit(train, y)
    assert len(fitted) == 4
    for frame, pipeline in fitted[:3]:
        assert len(frame) == 108
        values = frame["a"]
        transformed = pipeline.steps[0][1].transform(pd.DataFrame(frame).assign(a=1e300))
        np.testing.assert_array_equal(transformed["a"], np.full(len(frame), values.max()))
        imputer = pipeline.steps[1][1].named_transformers_["numeric"].steps[0][1]
        assert imputer.statistics_[0] == pytest.approx(float(values.median()))
    assert len(fitted[-1][0]) == len(train)
    assert base.raw_schema_.dtypes.equals(train.dtypes)

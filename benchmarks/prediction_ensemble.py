"""Fold-local prediction ensembles sharing the exact retained calibrated bases.

``fit_prediction_bundle`` returns only a JSON-safe report. Its separate
``fit_prediction_bundle_result`` testing helper retains an in-memory ``source``.
Single-model arms remain supervised separately by ``benchmarks.tabular``.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
import sklearn
from numpy.typing import NDArray
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from benchmarks import tabular
from featcopilot.classification import (
    NestedBaseClassifier,
    aligned_probability,
    classification_target,
    inner_split,
    make_subsample85_splits,
    positions_report,
    require_classes,
)
from featcopilot.classification import validate_probabilities as validate_probabilities
from featcopilot.utils.model_preprocessing import validate_classification_frame

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]
EnsembleSampling = Literal["cv3", "subsample85"]
ENSEMBLE_SEED_OFFSET = 32452843
N_MEMBERS = 3
MAX_ADDED_FEATURES = 50
MAX_CANDIDATES = 1000
LEARNERS = ("linear", "lightgbm", "catboost", "xgboost")
_BOUNDARY_KEYS = (
    "groups",
    "group",
    "group_ids",
    "timestamps",
    "time",
    "times",
    "label_end",
    "label_end_times",
)


class PredictionBundleConfig(BaseModel):
    """Validated IID execution settings; the feature/search budgets are fixed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(default=42, ge=0, lt=2**32, strict=True)
    threads: int = Field(default=8, ge=1, strict=True)
    stability_repeats: int = Field(default=3, ge=1, le=10, strict=True)
    split_kind: Literal["iid"] = "iid"
    ensemble_sampling: EnsembleSampling = "cv3"


ConfigInput = PredictionBundleConfig | Mapping[str, object]


def _config(config: ConfigInput) -> PredictionBundleConfig:
    if isinstance(config, PredictionBundleConfig):
        return config
    if "categorical_encoding" in config and (
        not isinstance(config["categorical_encoding"], str) or config["categorical_encoding"] != "one_hot"
    ):
        raise ValueError(
            "Calibration bundles require categorical_encoding='one_hot'; target encoding is not wired for this study"
        )
    for key, fixed in (("max_candidates", MAX_CANDIDATES), ("max_added_features", MAX_ADDED_FEATURES)):
        if key in config:
            value = config[key]
            if not isinstance(value, int) or value != fixed:
                raise ValueError(f"Prediction ensemble protocol fixes {key}={fixed}; overrides are unsupported")
    for key in _BOUNDARY_KEYS:
        if config.get(key) is not None:
            raise ValueError(f"Prediction ensembles require strict IID data; {key} is unsupported")
    for key in ("split_kind", "split_strategy"):
        if key in config and config[key] not in ("iid", "stratified"):
            raise ValueError(f"Prediction ensembles require strict IID data; unsupported {key}")
    metadata = config.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        _config(metadata)
    # Runner-wide options do not alter this predeclared procedure.
    return PredictionBundleConfig.model_validate(
        {key: config[key] for key in ("seed", "threads", "stability_repeats", "ensemble_sampling") if key in config}
    )


def _frame(X: pd.DataFrame) -> pd.DataFrame:
    return validate_classification_frame(X)


def _target(y: pd.Series | np.ndarray, rows: int) -> np.ndarray:
    return classification_target(y, rows)


def _require_classes(target: np.ndarray, classes: np.ndarray, scope: str) -> None:
    require_classes(target, classes, scope)


def _inner_split(y: np.ndarray, seed: int, scope: str) -> tuple[IndexArray, IndexArray]:
    return inner_split(y, seed, scope)


def make_ensemble_splits(
    y: pd.Series | np.ndarray, seed: int = 42, *, ensemble_sampling: EnsembleSampling = "cv3"
) -> list[tuple[IndexArray, IndexArray]]:
    """Build three IID training/calibration pairs with complete class coverage.

    Parameters
    ----------
    y : pandas.Series or ndarray
        Classification labels in outer-training positional order.
    seed : int, default=42
        Inner seed. The ensemble seed adds ``ENSEMBLE_SEED_OFFSET`` modulo 2**32.
    ensemble_sampling : {"cv3", "subsample85"}, default="cv3"
        Original stratified three-fold partition, or three stratified 85% bags
        with complementary 15% calibration holdouts. Holdouts may overlap across
        bags and need not cover every row. Bag seeds increment modulo 2**32.

    Returns
    -------
    list[tuple[ndarray, ndarray]]
        Base-training and calibration positions, preserving splitter order.
        No retries, redraws, case deduplication, or adaptive fractions.
    """
    if ensemble_sampling not in ("cv3", "subsample85"):
        raise ValueError("ensemble_sampling must be 'cv3' or 'subsample85'")
    if ensemble_sampling == "subsample85":
        return make_subsample85_splits(y, seed)

    target = _target(y, len(y))
    classes, counts = np.unique(target, return_counts=True)
    if np.min(counts) < N_MEMBERS:
        raise ValueError("Every class needs at least three rows for three calibration folds")
    splitter = StratifiedKFold(n_splits=N_MEMBERS, shuffle=True, random_state=(seed + ENSEMBLE_SEED_OFFSET) % 2**32)
    splits = list(splitter.split(np.zeros(len(target)), target))
    for member, (train, valid) in enumerate(splits):
        _require_classes(target[train], classes, f"member {member} base TRAIN")
        _require_classes(target[valid], classes, f"member {member} calibration VAL")
        _inner_split(target[train], seed, f"member {member}")
    return splits


def _aligned_probability(
    probability: np.ndarray, actual_classes: np.ndarray, expected_classes: np.ndarray, rows: int
) -> FloatArray:
    return aligned_probability(probability, actual_classes, expected_classes, rows)


def _positions_report(
    positions: IndexArray, y: np.ndarray, classes: np.ndarray, *, membership: bool = False
) -> dict[str, object]:
    return positions_report(positions, y, classes, membership=membership)


def _settings(config: PredictionBundleConfig) -> dict[str, object]:
    return {"seed": config.seed, "threads": config.threads, "numeric_clip_quantiles": (0.0, 1.0)}


class CompleteBaseProcedure(NestedBaseClassifier):
    """Cloneable, fold-local three-setting search followed by one winning refit.

    Parameters
    ----------
    learner : str, default="linear"
        One of linear, lightgbm, catboost, or xgboost.
    config : PredictionBundleConfig or mapping or None, default=None
        Seed, thread count, and stability repeats. Constructor arguments are
        retained unchanged for sklearn cloning.
    engineered : bool, default=False
        Use the incumbent frequency/aligned/robust predictive feature procedure.
    """

    _strict_state_fingerprint = True

    def __init__(self, learner: str = "linear", config: ConfigInput | None = None, engineered: bool = False):
        self.learner = learner
        self.config = config
        self.engineered = engineered

    def _benchmark_parameters(self, parameters: dict[str, object]) -> dict[str, object]:
        rename = (
            {"C": "regularization"}
            if self.learner == "linear"
            else ({"max_depth": "depth"} if self.learner == "xgboost" else {})
        )
        return {rename.get(key, key): value for key, value in parameters.items()}

    def _make_model(self, X: pd.DataFrame, parameters: dict[str, object]) -> Pipeline:
        # Native CatBoost stays here; production never guesses learner representation.
        model = tabular.make_model(
            X,
            "classification",
            self.learner,
            self._benchmark_parameters(parameters),
            {"seed": self.random_state, "threads": self.n_jobs, "numeric_clip_quantiles": (0.0, 1.0)},
        )
        if not isinstance(model, Pipeline):
            raise TypeError("Matched learner must be a sklearn Pipeline")
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray, **fit_params: object) -> CompleteBaseProcedure:
        """Adapt frozen benchmark aliases/factories to the single production fit loop."""
        for name in list(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        config = _config(self.config if self.config is not None else PredictionBundleConfig())
        if self.learner not in LEARNERS:
            raise ValueError(f"Unsupported classifier learner {self.learner!r}")
        X = _frame(X)
        candidates = tabular.candidate_parameters(self.learner)
        if len(candidates) != 3:
            raise ValueError("The declared learner search must contain exactly three settings")
        bare = tabular.make_model(X, "classification", self.learner, candidates[0], _settings(config)).steps[-1][1]
        if self.learner == "catboost":
            bare = type(bare)(**{key: value for key, value in bare.get_params().items() if key != "cat_features"})
        rename = (
            {"regularization": "C"}
            if self.learner == "linear"
            else ({"depth": "max_depth"} if self.learner == "xgboost" else {})
        )
        self.estimator = bare
        self.parameter_candidates = [
            {rename.get(key, key): value for key, value in params.items()} for params in candidates
        ]
        self.random_state = config.seed
        self.n_jobs = config.threads
        self.stability_repeats = config.stability_repeats
        self.max_added_features = MAX_ADDED_FEATURES
        self.max_candidates = MAX_CANDIDATES
        self.numeric_clip_quantiles = (0.0, 1.0)
        self.categorical_encoding = "one_hot"
        super().fit(X, y, **fit_params)
        self.selected_parameters_ = self._benchmark_parameters(self.selected_parameters_)
        for trial in self.trials_:
            parameters = trial["parameters"]
            if not isinstance(parameters, dict):
                raise TypeError("Production trial parameters must be dictionaries")
            trial["parameters"] = self._benchmark_parameters(parameters)
        return self


def _retained_bases(model: CalibratedClassifierCV) -> tuple[CompleteBaseProcedure, ...]:
    # This is the only private sklearn layout boundary. Never reimplement Platt.
    version = Version(sklearn.__version__)
    if not Version("1.7.2") <= version < Version("1.10"):
        raise RuntimeError(f"Unsupported sklearn calibration adapter version {version}; expected 1.7.2/1.8/1.9")
    check_is_fitted(model, ["calibrated_classifiers_", "classes_"])
    pairs = model.calibrated_classifiers_
    if model.method != "sigmoid" or model.ensemble is not True or len(pairs) != N_MEMBERS:
        raise ValueError("Expected exactly three retained sigmoid ensemble pairs")
    estimators = []
    expected_calibrators = 1 if len(model.classes_) == 2 else len(model.classes_)
    for pair in pairs:
        estimator = getattr(pair, "estimator", None)
        if not isinstance(estimator, CompleteBaseProcedure):
            raise TypeError("Calibration pair does not retain a CompleteBaseProcedure")
        if len(pair.calibrators) != expected_calibrators or not callable(pair.predict_proba):
            raise ValueError("Unsupported retained sklearn calibration-pair layout")
        if not np.array_equal(pair.classes, model.classes_) or not np.array_equal(estimator.classes_, model.classes_):
            raise ValueError("Retained calibration pair class alignment is invalid")
        estimators.append(estimator)
    if len({id(estimator) for estimator in estimators}) != N_MEMBERS:
        raise ValueError("Calibration pairs must retain three distinct fitted base objects")
    return tuple(estimators)


@dataclass(frozen=True)
class RetainedPredictionSource:
    """In-memory predictions from the one fitted public sklearn calibrator."""

    calibrated_model: CalibratedClassifierCV

    @property
    def base_estimators(self) -> tuple[CompleteBaseProcedure, ...]:
        """Return the actual retained objects, not clones or independently fitted bases."""
        return _retained_bases(self.calibrated_model)

    def predict_plain_proba(self, X: pd.DataFrame) -> FloatArray:
        """Average the native probabilities of the exact retained fitted bases."""
        classes = np.asarray(self.calibrated_model.classes_)
        probabilities = [
            _aligned_probability(estimator.predict_proba(X), estimator.classes_, classes, len(X))
            for estimator in self.base_estimators
        ]
        return validate_probabilities(np.mean(probabilities, axis=0), len(X), len(classes))

    def predict_sigmoid_proba(self, X: pd.DataFrame) -> FloatArray:
        """Average sklearn's calibrated-pair predictions without any refitting."""
        _retained_bases(self.calibrated_model)
        classes = np.asarray(self.calibrated_model.classes_)
        probabilities = [
            _aligned_probability(np.asarray(pair.predict_proba(X)), np.asarray(pair.classes), classes, len(X))
            for pair in self.calibrated_model.calibrated_classifiers_
        ]
        return validate_probabilities(np.mean(probabilities, axis=0), len(X), len(classes))


@dataclass(frozen=True)
class _CachedPredictionView:
    classes_: np.ndarray
    probability: FloatArray

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.classes_[np.argmax(self.probability, axis=1)])

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        return self.probability


def _metrics(probability: FloatArray, X: pd.DataFrame, y: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    result = tabular.metrics(_CachedPredictionView(classes, probability), X, pd.Series(y), "classification")
    result["log_loss"] = result["loss"]
    return result


class PredictionBundleReport(TypedDict):
    """JSON-only supervisor contract; arm training estimates must not be added."""

    plain_metrics: dict[str, float]
    calibrated_metrics: dict[str, float]
    timing: dict[str, float]
    diagnostics: dict[str, object]
    shared_cost_not_additive: bool


@dataclass(frozen=True)
class PredictionBundleResult:
    """Paired metrics/report plus an explicitly nonserialized inference source."""

    metrics: dict[str, dict[str, float]]
    diagnostics: dict[str, object]
    timing: dict[str, float]
    source: RetainedPredictionSource = field(repr=False, compare=False)

    def to_report(self) -> PredictionBundleReport:
        """Return only JSON-safe metrics, diagnostics, and timing, never models."""
        report: PredictionBundleReport = {
            "plain_metrics": copy.deepcopy(self.metrics["plain"]),
            "calibrated_metrics": copy.deepcopy(self.metrics["sigmoid"]),
            "diagnostics": copy.deepcopy(self.diagnostics),
            "timing": self.timing.copy(),
            "shared_cost_not_additive": True,
        }
        json.dumps(report, allow_nan=False)
        return report


def fit_prediction_bundle_result(
    train_X: pd.DataFrame,
    train_y: pd.Series | np.ndarray,
    test_X: pd.DataFrame,
    test_y: pd.Series | np.ndarray,
    *,
    task: str,
    learner: str,
    config: ConfigInput,
    engineered: bool,
) -> PredictionBundleResult:
    """Fit a model-bearing bundle for inference/tests, not supervisor serialization.

    Parameters
    ----------
    train_X, train_y : pandas.DataFrame, pandas.Series or ndarray
        Original outer-training features and labels; no preprocessing is prefitted.
    test_X, test_y : pandas.DataFrame, pandas.Series or ndarray
        Untouched outer-test data, used only for scoring after all fits complete.
    task : str
        Must be ``"classification"``.
    learner : str
        One of ``linear``, ``lightgbm``, ``catboost``, ``xgboost``.
    config : PredictionBundleConfig or mapping
        Defaults: seed 42, threads 8, stability_repeats 3. Boundary metadata is
        rejected. Runner-wide unrelated keys are ignored; budgets remain fixed
        at 1000 candidates and 50 added features. Explicit budget keys must be
        integers equal to those fixed values; other values raise ValueError.
        ``ensemble_sampling="cv3"`` preserves the original partition design;
        ``"subsample85"`` opts into fixed repeated stratified 85% training bags.
    engineered : bool
        Whether each base's entire search/refit includes predictive engineering.

    Returns
    -------
    PredictionBundleResult
        Metrics keyed by ``plain`` and ``sigmoid``. Use ``to_report()`` for JSON.
        Timings count the physical bundle once; the plain standalone training
        estimate is the base-fit sum plus the separately measured split cost.

    Raises
    ------
    ValueError
        Unsupported tasks/boundaries, insufficient classes, invalid schemas or
        probabilities. There is no retry, adaptive fold count, or fallback arm.
    """
    started = time.perf_counter()
    if task != "classification":
        raise ValueError("Prediction ensembles support classification only")
    settings = _config(config)
    _frame(train_X)
    target = _target(train_y, len(train_X))
    if learner not in LEARNERS:
        raise ValueError(f"Unsupported classifier learner {learner!r}")
    fit_started = time.perf_counter()
    split_started = time.perf_counter()
    splits = make_ensemble_splits(target, settings.seed, ensemble_sampling=settings.ensemble_sampling)
    split_seconds = time.perf_counter() - split_started
    calibrated = CalibratedClassifierCV(
        estimator=CompleteBaseProcedure(learner=learner, config=settings, engineered=engineered),
        method="sigmoid",
        cv=splits,
        ensemble=True,
        n_jobs=1,
    )
    with threadpool_limits(limits=settings.threads):
        calibrated.fit(train_X, target)
        fit_seconds = time.perf_counter() - fit_started
        source = RetainedPredictionSource(calibrated)
        bases = source.base_estimators
        base_seconds = sum(base.fit_seconds_ for base in bases)
        remainder = fit_seconds - base_seconds
        if remainder < -max(1e-9, fit_seconds * 1e-12):
            raise RuntimeError("Sequential base-fit timings exceed total calibration fit wall time")
        _frame(test_X)
        test_target = _target(test_y, len(test_X))
        classes = np.asarray(calibrated.classes_)
        _require_classes(test_target, classes, "outer TEST evaluation")
        score_started = time.perf_counter()
        plain_probability = source.predict_plain_proba(test_X)
        plain_predict_seconds = time.perf_counter() - score_started
        plain = _metrics(plain_probability, test_X, test_target, classes)
        plain_seconds = time.perf_counter() - score_started
        score_started = time.perf_counter()
        sigmoid_probability = source.predict_sigmoid_proba(test_X)
        calibrated_predict_seconds = time.perf_counter() - score_started
        sigmoid = _metrics(sigmoid_probability, test_X, test_target, classes)
        sigmoid_seconds = time.perf_counter() - score_started
    subsample = settings.ensemble_sampling == "subsample85"
    members = [
        {
            "member": index,
            "base_train": _positions_report(train, target, classes, membership=subsample),
            "calibration_validation": _positions_report(valid, target, classes, membership=subsample),
            **({"sampling_seed": (settings.seed + ENSEMBLE_SEED_OFFSET + index) % 2**32} if subsample else {}),
            **base.diagnostics(),
        }
        for index, (base, (train, valid)) in enumerate(zip(bases, splits, strict=True))
    ]
    identities = [base.fit_identity_sha256_ for base in bases]
    widths = [len(base.feature_names_out_) for base in bases]
    diagnostics: dict[str, object] = {
        "protocol": "shared_retained_sigmoid_subsample85_v1" if subsample else "shared_retained_sigmoid_3fold_v1",
        "ensemble_sampling": settings.ensemble_sampling,
        "task": task,
        "split_kind": "iid",
        "learner": learner,
        "engineered": engineered,
        "sklearn_version": sklearn.__version__,
        "calibration_adapter": "retained_estimator_calibrators_1.7.2_1.8_1.9",
        "seed": settings.seed,
        "ensemble_seed": (settings.seed + ENSEMBLE_SEED_OFFSET) % 2**32,
        "classes": classes.tolist(),
        "class_counts": [int(np.count_nonzero(target == label)) for label in classes],
        "member_count": N_MEMBERS,
        "members": members,
        "member_feature_counts": widths,
        "max_member_feature_count": max(widths),
        "raw_feature_count": train_X.shape[1],
        "member_output_feature_counts": widths.copy(),
        "max_member_output_features": max(widths),
        "n_input_features": train_X.shape[1],
        "plain_fit_identity_sha256": identities,
        "sigmoid_fit_identity_sha256": identities.copy(),
        "exact_shared_fits": True,
    }
    if subsample:
        coverage = np.zeros(len(target), dtype=np.int64)
        for _, validation in splits:
            coverage[validation] += 1
        diagnostics.update(
            sampling_design="repeated_stratified_holdout",
            requested_train_fraction=0.85,
            requested_calibration_fraction=0.15,
            calibration_coverage_histogram=np.bincount(coverage, minlength=N_MEMBERS + 1).tolist(),
            pairwise_overlap=[
                {
                    "members": [left, right],
                    "base_train": int(np.intersect1d(splits[left][0], splits[right][0]).size),
                    "calibration_validation": int(np.intersect1d(splits[left][1], splits[right][1]).size),
                }
                for left in range(N_MEMBERS)
                for right in range(left + 1, N_MEMBERS)
            ],
            distinct_train_bag_count=len({frozenset(train.tolist()) for train, _ in splits}),
        )
    timing = {
        "fit_seconds": fit_seconds,
        "base_fit_seconds": base_seconds,
        "base_fe_seconds": sum(base.fe_fit_seconds_ for base in bases),
        "base_model_fit_seconds": sum(base.model_fit_seconds_ for base in bases),
        "base_tuning_and_orchestration_seconds": sum(base.tuning_and_orchestration_seconds_ for base in bases),
        "calibration_and_orchestration_seconds": max(0.0, remainder),
        "plain_predict_seconds": plain_predict_seconds,
        "calibrated_predict_seconds": calibrated_predict_seconds,
        "fit_wall_seconds": fit_seconds,
        "sum_base_fit_seconds": base_seconds,
        "split_seconds": split_seconds,
        "calibration_orchestration_seconds": max(0.0, remainder),
        "plain_training_estimate_seconds": base_seconds + split_seconds,
        "calibrated_training_seconds": fit_seconds,
        "plain_score_seconds": plain_seconds,
        "sigmoid_score_seconds": sigmoid_seconds,
        "physical_bundle_wall_seconds": time.perf_counter() - started,
    }
    result = PredictionBundleResult({"plain": plain, "sigmoid": sigmoid}, diagnostics, timing, source)
    result.to_report()
    return result


def fit_prediction_bundle(
    train_X: pd.DataFrame,
    train_y: pd.Series | np.ndarray,
    test_X: pd.DataFrame,
    test_y: pd.Series | np.ndarray,
    *,
    task: str,
    learner: str,
    config: ConfigInput,
    engineered: bool,
) -> PredictionBundleReport:
    """Fit and return the JSON-only shared-fit supervisor report.

    Parameters
    ----------
    train_X, train_y : pandas.DataFrame, pandas.Series or ndarray
        Original outer-training data. Every learned component is fitted locally.
    test_X, test_y : pandas.DataFrame, pandas.Series or ndarray
        Untouched outer-test data used only for scoring.
    task : str
        Must be ``"classification"`` with strict IID metadata.
    learner : str
        One of ``linear``, ``lightgbm``, ``catboost``, ``xgboost``.
    config : PredictionBundleConfig or mapping
        Defaults: seed 42, threads 8, stability_repeats 3.
        Explicit ``max_candidates``/``max_added_features`` must equal the fixed
        integer budgets 1000/50; other values are rejected before any fitting.
        ``ensemble_sampling`` is ``"cv3"`` by default or fixed ``"subsample85"``;
        all three members retain the same complete HPO/FE procedure in either mode.
    engineered : bool
        Enable the incumbent fold-local frequency/aligned/robust feature procedure.

    Returns
    -------
    PredictionBundleReport
        ``plain_metrics``, ``calibrated_metrics``, ``timing``, ``diagnostics``,
        and ``shared_cost_not_additive=True``. Prediction times exclude metric
        computation; score times include it. The plain standalone training
        estimate is base-fit time plus split overhead, while calibrated training
        uses full fit time. Charge ``physical_bundle_wall_seconds`` only once.
        ``base_fe_seconds`` measures FE construction/fit-transform in all search
        trials and winning refits, including selection's internal learner fits.
        ``base_model_fit_seconds`` measures matched prediction-pipeline fits;
        ``base_tuning_and_orchestration_seconds`` is remaining base fit time.
        These three components are already included in ``base_fit_seconds``.
        Subsample diagnostics report actual rounded fractions, consecutive seeds,
        positional membership hashes (N uint8 bytes), pairwise overlaps, and a
        four-bin calibration-coverage histogram for multiplicities zero to three.
        Repeated holdouts are not an out-of-fold partition or coverage guarantee.
        No models, vocabulary, raw rows, or prediction vectors are returned.
    """
    return fit_prediction_bundle_result(
        train_X, train_y, test_X, test_y, task=task, learner=learner, config=config, engineered=engineered
    ).to_report()

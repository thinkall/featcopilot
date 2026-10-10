"""Benchmark-only OOF blending and a separate prespecified 50:50 replication.

The OOF loss is a meta-training objective, not independent validation or a
no-harm guarantee. After freezing one global weight, both complete procedures
are fitted afresh on all TRAIN: their selected parameters and FE plans may
change. The external test is accessed only after these fits finish.

``fit_fixed_average_bundle`` fits just the same two full-TRAIN procedures, with
no OOF fits or weight optimization. This replicates the already-prespecified
50:50 arm, not a weight grid or a causal variance/no-harm claim. Neither entry
point changes production behavior or the other's defaults.

Only the explicit LightGBM/IID/one-hot experiment is supported. No calibration,
rank bagging, target encoding, provider access, or production policy is added.
The JSON entry point excludes models, rows, vocabularies and prediction vectors.
The separate testing helper retains private arrays and model artifacts; pickle
is for trusted artifacts only, and fitted preprocessing may contain private data.
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
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from benchmarks import tabular
from benchmarks.prediction_ensemble import CompleteBaseProcedure, PredictionBundleConfig
from featcopilot.classification import (
    aligned_probability,
    classification_target,
    inner_split,
    positions_report,
    require_classes,
    validate_probabilities,
)
from featcopilot.utils.model_preprocessing import validate_classification_frame

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]
Arm = Literal["raw", "fe", "average", "blend"]
ARMS: tuple[Arm, ...] = ("raw", "fe", "average", "blend")
FIXED_AVERAGE_ARMS: tuple[Arm, ...] = ("raw", "fe", "average")
OOF_FOLDS = 5
OOF_SEED_OFFSET = 1000
CRITERION = "float64_mean_log_loss_global_convex_weight"
_BOUNDARY_KEYS = ("groups", "group", "group_ids", "timestamps", "time", "times", "label_end", "label_end_times")


class OOFBlendConfig(BaseModel):
    """Fixed benchmark protocol with validated seed and thread count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(default=42, ge=0, lt=2**32, strict=True)
    threads: int = Field(default=8, ge=1, strict=True)
    stability_repeats: Literal[3] = 3
    split_kind: Literal["iid"] = "iid"
    categorical_encoding: Literal["one_hot"] = "one_hot"
    max_candidates: Literal[1000] = 1000
    max_added_features: Literal[50] = 50


ConfigInput = OOFBlendConfig | Mapping[str, object]


def _config(config: ConfigInput) -> OOFBlendConfig:
    if isinstance(config, OOFBlendConfig):
        return config
    for key in _BOUNDARY_KEYS:
        if config.get(key) is not None:
            raise ValueError(f"OOF blending requires strict IID data; {key} is unsupported")
    for key in ("split_kind", "split_strategy"):
        if key in config and config[key] not in ("iid", "stratified"):
            raise ValueError(f"OOF blending requires strict IID data; unsupported {key}")
    if config.get("task", "classification") != "classification":
        raise ValueError("OOF blending supports classification only")
    for key, fixed in (("stability_repeats", 3), ("max_candidates", 1000), ("max_added_features", 50)):
        if key in config and (type(config[key]) is not int or config[key] != fixed):
            raise ValueError(f"OOF blending fixes {key}={fixed}")
    if "numeric_clip_quantiles" in config:
        quantiles = config["numeric_clip_quantiles"]
        if not isinstance(quantiles, (list, tuple)) or tuple(quantiles) != (0.0, 1.0):
            raise ValueError("OOF blending fixes numeric_clip_quantiles=(0.0, 1.0)")
    metadata = config.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        _config(metadata)
    keys = ("seed", "threads", "stability_repeats", "categorical_encoding", "max_candidates", "max_added_features")
    return OOFBlendConfig.model_validate({key: config[key] for key in keys if key in config})


def make_oof_splits(y: pd.Series | np.ndarray, seed: int = 42) -> list[tuple[IndexArray, IndexArray]]:
    """Build exactly five stratified folds, validating all base inner partitions.

    Parameters
    ----------
    y : pandas.Series or ndarray
        Outer-training labels, in positional order.
    seed : int, default=42
        Base seed; the OOF splitter adds 1000 modulo 2**32.

    Returns
    -------
    list[tuple[ndarray, ndarray]]
        Complement TRAIN and OOF holdout positions. Insufficient class support
        raises; there is no fold reduction, redraw or row dropping.
    """
    settings = OOFBlendConfig(seed=seed)
    target = classification_target(y, len(y))
    classes, counts = np.unique(target, return_counts=True)
    if np.min(counts) < OOF_FOLDS:
        raise ValueError("Every class needs at least five rows for five OOF folds")
    splitter = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=(settings.seed + OOF_SEED_OFFSET) % 2**32)
    splits = list(splitter.split(np.zeros(len(target)), target))
    for member, (train, holdout) in enumerate(splits):
        require_classes(target[train], classes, f"OOF member {member} TRAIN")
        require_classes(target[holdout], classes, f"OOF member {member} holdout")
        inner_split(target[train], settings.seed, f"OOF member {member}")
    inner_split(target, settings.seed, "full TRAIN")
    return splits


def _convex_probability(raw: FloatArray, fe: FloatArray, weight: float) -> FloatArray:
    if weight == 0.0:
        return raw
    if weight == 1.0:
        return fe
    return validate_probabilities((1.0 - weight) * raw + weight * fe, len(raw), raw.shape[1], normalize=False)


@dataclass(frozen=True)
class BlendWeight:
    """Frozen global weight and compact meta-training optimization diagnostics."""

    weight: float
    losses: dict[str, float]
    optimization: dict[str, object]


def optimize_blend_weight(
    raw: np.ndarray, fe: np.ndarray, y: pd.Series | np.ndarray, classes: np.ndarray
) -> BlendWeight:
    """Minimize OOF float64 log loss, including both endpoints and exact ties.

    Parameters
    ----------
    raw, fe : ndarray
        Row-aligned OOF probabilities already in the supplied class order.
        Native float32 roundoff is validated before float64 row normalization.
    y : pandas.Series or ndarray
        Meta-training labels, never external test labels.
    classes : ndarray
        Unique, sorted class labels matching the probability columns.

    Returns
    -------
    BlendWeight
        One scalar in [0, 1], OOF losses, and optimizer/endpoint diagnostics.
        Identical clipped true-class probability vectors select zero, reported
        as ``identical_probabilities`` even if incorrect-class probabilities
        differ. Any optimizer or probability failure raises rather than becoming
        a successful raw fallback.
    """
    target = classification_target(y, len(y))
    require_classes(target, classes, "OOF objective")
    raw_probability = validate_probabilities(raw, len(target), len(classes))
    fe_probability = validate_probabilities(fe, len(target), len(classes))
    columns = np.searchsorted(classes, target)
    rows = np.arange(len(target))
    raw_true = raw_probability[rows, columns]
    fe_true = fe_probability[rows, columns]
    epsilon = np.finfo(np.float64).eps
    clipped_raw_true = np.clip(raw_true, epsilon, 1.0 - epsilon)
    identical = np.array_equal(clipped_raw_true, np.clip(fe_true, epsilon, 1.0 - epsilon))

    def objective(weight: float) -> float:
        # Equal effective targets make the objective flat; weighted arithmetic
        # must not manufacture an improvement from float64 roundoff.
        probability = clipped_raw_true if identical else (1.0 - weight) * raw_true + weight * fe_true
        return float(-np.mean(np.log(np.clip(probability, epsilon, 1.0 - epsilon)), dtype=np.float64))

    endpoints = [{"weight": weight, "loss": objective(weight)} for weight in (0.0, 1.0)]
    candidates = endpoints.copy()
    optimization: dict[str, object] = {
        "method": "bounded",
        "bounds": [0.0, 1.0],
        "xatol": 1e-5,
        "maxiter": 500,
        "tie_break": "smaller_weight_on_exact_equal_loss",
        "endpoint_candidates": endpoints,
        "epsilon": epsilon,
    }
    if identical:
        optimization.update(success=True, status="identical_probabilities", nfev=0, interior_candidate=None)
    else:
        optimum = minimize_scalar(
            objective, method="bounded", bounds=(0.0, 1.0), options={"xatol": 1e-5, "maxiter": 500}
        )
        if (
            not optimum.success
            or not np.isfinite(optimum.x)
            or not np.isfinite(optimum.fun)
            or not 0.0 < optimum.x < 1.0
        ):
            raise RuntimeError("OOF blend weight optimization failed or returned an invalid interior solution")
        interior = {"weight": float(optimum.x), "loss": objective(float(optimum.x))}
        candidates.append(interior)
        optimization.update(success=True, status="converged", nfev=int(optimum.nfev), interior_candidate=interior)
    winner = min(candidates, key=lambda candidate: (candidate["loss"], candidate["weight"]))
    return BlendWeight(
        winner["weight"],
        {"raw": endpoints[0]["loss"], "fe": endpoints[1]["loss"], "average": objective(0.5), "blend": winner["loss"]},
        optimization,
    )


def _base_probability(base: CompleteBaseProcedure, X: pd.DataFrame, classes: np.ndarray) -> FloatArray:
    # CompleteBaseProcedure already validates native dtype roundoff and normalizes
    # in float64. Do not renormalize a second time: endpoints preserve its output.
    return aligned_probability(base.predict_proba(X), base.classes_, classes, len(X), normalize=False)


@dataclass(frozen=True)
class FixedAverageSource:
    """Inference-only full-TRAIN models with a prespecified 50:50 average.

    These objects are shared by all three arms, not independently refitted.
    Serialization retains private fitted preprocessing; load only trusted pickle.
    """

    raw: CompleteBaseProcedure
    fe: CompleteBaseProcedure

    @property
    def classes_(self) -> np.ndarray:
        """Return the canonical sorted original-label probability order."""
        return np.sort(self.raw.classes_)

    @property
    def base_estimators(self) -> tuple[CompleteBaseProcedure, CompleteBaseProcedure]:
        """Return the two actual full-TRAIN fitted objects."""
        return self.raw, self.fe

    def predict_probabilities(self, X: pd.DataFrame) -> dict[str, FloatArray]:
        """Predict each full model once and return raw, FE and fixed average."""
        raw = _base_probability(self.raw, X, self.classes_)
        fe = _base_probability(self.fe, X, self.classes_)
        return {
            "raw": raw,
            "fe": fe,
            "average": _convex_probability(raw, fe, 0.5),
        }

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        """Predict the prespecified equal-weight probability average."""
        return self.predict_probabilities(X)["average"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict original string/integer classes using probability argmax."""
        return np.asarray(self.classes_[np.argmax(self.predict_proba(X), axis=1)])


@dataclass(frozen=True)
class OOFBlendSource(FixedAverageSource):
    """Inference-only full-TRAIN models and frozen OOF weight; no OOF children.

    The same two objects provide all four arms. Serialization retains private
    fitted preprocessing; load only trusted pickle.
    """

    weight: float

    def predict_probabilities(self, X: pd.DataFrame) -> dict[str, FloatArray]:
        """Predict each full model once and return all four probability arms."""
        probabilities = super().predict_probabilities(X)
        probabilities["blend"] = _convex_probability(probabilities["raw"], probabilities["fe"], self.weight)
        return probabilities

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        """Predict the frozen blend, preserving exact endpoint probabilities."""
        if self.weight == 0.0:
            return _base_probability(self.raw, X, self.classes_)
        if self.weight == 1.0:
            return _base_probability(self.fe, X, self.classes_)
        return self.predict_probabilities(X)["blend"]


@dataclass(frozen=True)
class _CachedPredictionView:
    classes_: np.ndarray
    probability: FloatArray

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.classes_[np.argmax(self.probability, axis=1)])

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        return self.probability


class FixedAverageReport(TypedDict):
    """JSON-only three-arm contract with no fabricated learned-blend outcome."""

    raw_metrics: dict[str, float]
    fe_metrics: dict[str, float]
    average_metrics: dict[str, float]
    n_input_features: int
    n_output_features: int
    diagnostics: dict[str, object]
    timing: dict[str, float]
    shared_cost_not_additive: bool


class OOFBlendReport(FixedAverageReport):
    """JSON-only controller contract; standalone arm costs are not additive."""

    blend_metrics: dict[str, float]


@dataclass(frozen=True)
class FixedAverageResult:
    """Testing-only three-arm result retaining full models and test probabilities."""

    metrics: dict[str, dict[str, float]]
    diagnostics: dict[str, object]
    timing: dict[str, float]
    source: FixedAverageSource = field(repr=False, compare=False)
    test_probability: dict[str, FloatArray] = field(repr=False, compare=False)

    def to_report(self) -> FixedAverageReport:
        """Return a defensive JSON report without private inference artifacts."""
        report: FixedAverageReport = {
            "raw_metrics": copy.deepcopy(self.metrics["raw"]),
            "fe_metrics": copy.deepcopy(self.metrics["fe"]),
            "average_metrics": copy.deepcopy(self.metrics["average"]),
            "n_input_features": self.source.raw.n_features_in_,
            "n_output_features": len(self.source.fe.feature_names_out_),
            "diagnostics": copy.deepcopy(self.diagnostics),
            "timing": self.timing.copy(),
            "shared_cost_not_additive": True,
        }
        json.dumps(report, allow_nan=False)
        return report


@dataclass(frozen=True)
class OOFBlendResult:
    """Testing result retaining models/arrays that are excluded from JSON."""

    metrics: dict[str, dict[str, float]]
    diagnostics: dict[str, object]
    timing: dict[str, float]
    source: OOFBlendSource = field(repr=False, compare=False)
    oof_probability: dict[str, FloatArray] = field(repr=False, compare=False)
    test_probability: dict[str, FloatArray] = field(repr=False, compare=False)

    def to_report(self) -> OOFBlendReport:
        """Return a defensive, JSON-safe report with no private model artifacts."""
        report: OOFBlendReport = {
            "raw_metrics": copy.deepcopy(self.metrics["raw"]),
            "fe_metrics": copy.deepcopy(self.metrics["fe"]),
            "average_metrics": copy.deepcopy(self.metrics["average"]),
            "blend_metrics": copy.deepcopy(self.metrics["blend"]),
            "n_input_features": self.source.raw.n_features_in_,
            "n_output_features": len(self.source.fe.feature_names_out_),
            "diagnostics": copy.deepcopy(self.diagnostics),
            "timing": self.timing.copy(),
            "shared_cost_not_additive": True,
        }
        json.dumps(report, allow_nan=False)
        return report


def _fit_full_procedures(
    train_X: pd.DataFrame, target: np.ndarray, learner: str, config: PredictionBundleConfig
) -> tuple[dict[str, CompleteBaseProcedure], dict[str, float]]:
    full: dict[str, CompleteBaseProcedure] = {}
    timing: dict[str, float] = {}
    for arm in ("raw", "fe"):
        stage_started = time.perf_counter()
        full[arm] = CompleteBaseProcedure(learner=learner, config=config, engineered=arm == "fe").fit(train_X, target)
        timing[f"full_{arm}_fit_seconds"] = time.perf_counter() - stage_started
    return full, timing


def _score_full_procedures(
    full: dict[str, CompleteBaseProcedure],
    test_X: pd.DataFrame,
    test_y: pd.Series | np.ndarray,
    classes: np.ndarray,
    arms: tuple[Arm, ...],
    weight: float,
) -> tuple[dict[str, FloatArray], dict[str, dict[str, float]], dict[str, float]]:
    # Call only after the weight and both full fits have frozen.
    score_started = time.perf_counter()
    validate_classification_frame(test_X)
    test_target = classification_target(test_y, len(test_X))
    require_classes(test_target, classes, "outer TEST evaluation")
    probabilities: dict[str, FloatArray] = {}
    scores: dict[str, dict[str, float]] = {}
    timing: dict[str, float] = {}
    for arm in arms:
        stage_started = time.perf_counter()
        if arm in ("raw", "fe"):
            probabilities[arm] = _base_probability(full[arm], test_X, classes)
        else:
            probabilities[arm] = _convex_probability(
                probabilities["raw"], probabilities["fe"], 0.5 if arm == "average" else weight
            )
        timing[f"{arm}_predict_seconds"] = time.perf_counter() - stage_started
        scores[arm] = tabular.metrics(
            _CachedPredictionView(classes, probabilities[arm]), test_X, pd.Series(test_target), "classification"
        )
        scores[arm]["log_loss"] = scores[arm]["loss"]
        timing[f"{arm}_cached_score_seconds"] = time.perf_counter() - stage_started
    timing["test_score_seconds"] = time.perf_counter() - score_started
    return probabilities, scores, timing


def _full_diagnostics(full: dict[str, CompleteBaseProcedure], arms: tuple[Arm, ...]) -> dict[str, object]:
    identities = {arm: base.fit_identity_sha256_ for arm, base in full.items()}
    widths = [len(full[arm].feature_names_out_) for arm in ("raw", "fe")]
    return {
        "full_models": {arm: base.diagnostics() for arm, base in full.items()},
        "full_fit_identity_sha256": {
            arm: [identities[arm]] if arm in ("raw", "fe") else [identities["raw"], identities["fe"]] for arm in arms
        },
        "exact_shared_fits": True,
        "full_complete_procedure_fit_count": 2,
        "n_input_features": full["raw"].n_features_in_,
        "n_output_features": widths[1],
        "full_output_feature_counts": widths,
    }


def _standalone_timings(
    timing: dict[str, float], arms: tuple[Arm, ...], *, learned_training_seconds: float = 0.0
) -> None:
    both_full_fits = timing["full_raw_fit_seconds"] + timing["full_fe_fit_seconds"]
    for arm in arms:
        training = timing[f"full_{arm}_fit_seconds"] if arm in ("raw", "fe") else both_full_fits
        if arm == "blend":
            training += learned_training_seconds
        scoring = timing[f"{arm}_cached_score_seconds"]
        if arm in ("average", "blend"):
            scoring += timing["raw_predict_seconds"] + timing["fe_predict_seconds"]
        timing[f"{arm}_training_seconds"] = training
        timing[f"{arm}_score_seconds"] = scoring
        timing[f"{arm}_standalone_seconds"] = training + scoring


def fit_oof_blend_bundle_result(
    train_X: pd.DataFrame,
    train_y: pd.Series | np.ndarray,
    test_X: pd.DataFrame,
    test_y: pd.Series | np.ndarray,
    *,
    learner: str,
    config: ConfigInput,
) -> OOFBlendResult:
    """Fit the fixed blend, retaining inference artifacts for synthetic tests.

    Parameters
    ----------
    train_X, train_y : pandas.DataFrame, pandas.Series or ndarray
        Original TRAIN data. Each OOF complement independently fits the whole
        three-setting HPO/FE/preprocessing procedure.
    test_X, test_y : pandas.DataFrame, pandas.Series or ndarray
        External test, first accessed after weight and full-model fits freeze.
    learner : str
        Must be ``"lightgbm"``.
    config : OOFBlendConfig or mapping
        Validated seed/threads. Fixed IID, one-hot, 1000 candidates, 50 additions,
        and three stability repeats; unsupported boundary metadata is rejected.
        Unrelated runner-wide keys do not change the declared procedure.

    Returns
    -------
    OOFBlendResult
        Four metrics arms and compact diagnostics with private inference state.
        Fold models are discarded. No failure is converted to a raw success.
    """
    started = time.perf_counter()
    settings = _config(config)
    if learner != "lightgbm":
        raise ValueError("The first OOF blend experiment supports explicit LightGBM classification only")
    validate_classification_frame(train_X)
    target = classification_target(train_y, len(train_X))
    classes = np.unique(target)
    split_started = time.perf_counter()
    splits = make_oof_splits(target, settings.seed)
    timing = {"split_seconds": time.perf_counter() - split_started, "oof_fit_seconds": 0.0, "oof_predict_seconds": 0.0}
    base_config = PredictionBundleConfig(seed=settings.seed, threads=settings.threads, stability_repeats=3)
    oof = {arm: np.empty((len(target), len(classes)), dtype=np.float64) for arm in ("raw", "fe")}
    coverage = np.zeros(len(target), dtype=np.int64)
    members: list[dict[str, object]] = []
    fit_started = time.perf_counter()
    with threadpool_limits(limits=settings.threads):
        oof_started = time.perf_counter()
        for member, (train, holdout) in enumerate(splits):
            report: dict[str, object] = {
                "member": member,
                "train": positions_report(train, target, classes),
                "holdout": positions_report(holdout, target, classes),
            }
            for arm in ("raw", "fe"):
                base = CompleteBaseProcedure(learner=learner, config=base_config, engineered=arm == "fe")
                stage_started = time.perf_counter()
                base.fit(train_X.iloc[train], target[train])
                timing["oof_fit_seconds"] += time.perf_counter() - stage_started
                stage_started = time.perf_counter()
                oof[arm][holdout] = _base_probability(base, train_X.iloc[holdout], classes)
                timing["oof_predict_seconds"] += time.perf_counter() - stage_started
                report[arm] = base.diagnostics()
                del base
            coverage[holdout] += 1
            members.append(report)
        if not np.all(coverage == 1):
            raise ValueError("Every TRAIN row must have exactly one OOF prediction per arm")
        timing["oof_seconds"] = time.perf_counter() - oof_started
        timing["oof_orchestration_seconds"] = (
            timing["oof_seconds"] - timing["oof_fit_seconds"] - timing["oof_predict_seconds"]
        )
        stage_started = time.perf_counter()
        weight = optimize_blend_weight(oof["raw"], oof["fe"], target, classes)
        timing["weight_optimization_seconds"] = time.perf_counter() - stage_started
        full, full_timing = _fit_full_procedures(train_X, target, learner, base_config)
        timing.update(full_timing)
        source = OOFBlendSource(full["raw"], full["fe"], weight.weight)
        timing["fit_seconds"] = time.perf_counter() - fit_started

        # This is the first access to external test features or labels.
        probabilities, scores, score_timing = _score_full_procedures(full, test_X, test_y, classes, ARMS, weight.weight)
        timing.update(score_timing)
    diagnostics: dict[str, object] = {
        "protocol": "lightgbm_iid_oof_convex_blend_5fold_v1",
        "task": "classification",
        "learner": learner,
        "split_kind": "iid",
        "seed": settings.seed,
        "oof_seed": (settings.seed + OOF_SEED_OFFSET) % 2**32,
        "sklearn_version": sklearn.__version__,
        "classes": classes.tolist(),
        "class_counts": [int(np.count_nonzero(target == label)) for label in classes],
        "n_oof_covered": int(np.count_nonzero(coverage == 1)),
        "oof_coverage_histogram": np.bincount(coverage, minlength=2).tolist(),
        "member_count": OOF_FOLDS,
        "members": members,
        **_full_diagnostics(full, ARMS),
        "oof_complete_procedure_fit_count": 2 * OOF_FOLDS,
        "weight": weight.weight,
        "oof_losses": weight.losses,
        "optimization": weight.optimization,
        "criterion": CRITERION,
        "source_policy": {
            "oof": "independent_complete_raw_and_fe_procedures_per_complement_no_whole_T_parameters",
            "inference": "fresh_complete_full_T_refits_after_weight_freeze_no_oof_children",
            "controls": "exact_same_full_T_objects_and_cached_probabilities_for_all_four_arms",
            "interpretation": "OOF_meta_training_loss_not_independent_validation_no_outer_test_no_harm_guarantee",
            "refit_shift": "standard_refit_full_stacking_parameters_and_feature_plans_may_change",
            "categorical_encoding": "one_hot",
            "numeric_clip_quantiles": [0.0, 1.0],
            "max_candidates": 1000,
            "max_added_features": 50,
            "stability_repeats": 3,
            "importance_resamples": 1,
            "selection_rule": "paired_gain",
            "candidate_set": "extended",
            "include_frequency": True,
        },
    }
    oof_cost = timing["split_seconds"] + timing["oof_seconds"]
    _standalone_timings(timing, ARMS, learned_training_seconds=oof_cost + timing["weight_optimization_seconds"])
    timing["physical_bundle_wall_seconds"] = time.perf_counter() - started
    result = OOFBlendResult(scores, diagnostics, timing, source, oof, probabilities)
    result.to_report()
    return result


def fit_oof_blend_bundle(
    train_X: pd.DataFrame,
    train_y: pd.Series | np.ndarray,
    test_X: pd.DataFrame,
    test_y: pd.Series | np.ndarray,
    *,
    learner: str,
    config: ConfigInput,
) -> OOFBlendReport:
    """Fit one physical bundle and return its JSON-only four-arm report.

    Parameters
    ----------
    train_X, train_y : pandas.DataFrame, pandas.Series or ndarray
        Outer TRAIN, with all preprocessing, HPO and feature discovery fold-local.
    test_X, test_y : pandas.DataFrame, pandas.Series or ndarray
        External test used only after the global weight and both full fits freeze.
    learner : str
        Explicitly ``"lightgbm"``; the experiment is classification/IID only.
    config : OOFBlendConfig or mapping
        Seed and threads, with one-hot and fixed 1000/50/3 feature budgets.

    Returns
    -------
    OOFBlendReport
        ``raw_metrics``, ``fe_metrics``, ``average_metrics``, ``blend_metrics``,
        final FE-plan widths, diagnostics and timing. Charge physical wall time
        once; standalone arm costs overlap and must not be summed. Cached score
        timings are disjoint physical stages; standalone average/blend scoring
        also includes both cached model prediction costs. Base diagnostics' FE
        and model stage times are subsets of complete fit time, never extra fits.
        The selected OOF loss is meta-training, not a no-harm test guarantee.
        No prediction vectors, rows, vocabularies or model objects are exported.
    """
    return fit_oof_blend_bundle_result(train_X, train_y, test_X, test_y, learner=learner, config=config).to_report()


def fit_fixed_average_bundle_result(
    train_X: pd.DataFrame,
    train_y: pd.Series | np.ndarray,
    test_X: pd.DataFrame,
    test_y: pd.Series | np.ndarray,
    *,
    learner: str,
    config: ConfigInput,
) -> FixedAverageResult:
    """Fit only the two full-TRAIN procedures for the prespecified 50:50 arm.

    Parameters
    ----------
    train_X, train_y : pandas.DataFrame, pandas.Series or ndarray
        Original outer TRAIN. Each complete procedure independently performs
        the same three-setting native HPO and winning full refit as the OOF
        bundle's full controls, with all FE/preprocessing local to its TRAIN.
    test_X, test_y : pandas.DataFrame, pandas.Series or ndarray
        External test, first accessed after both complete full fits finish.
    learner : str
        Must be ``"lightgbm"``.
    config : OOFBlendConfig or mapping
        Same IID/one-hot/1000-candidate/50-addition/three-repeat constraints,
        seed and threads as the OOF protocol. No weight is fitted or tunable.

    Returns
    -------
    FixedAverageResult
        Raw, FE and average metrics, diagnostics and private testing artifacts.
        Exactly two complete procedures (eight prediction-pipeline fits); no
        OOF fits, optimizer, calibration or fabricated learned-blend result.
        This is replication of a prespecified arm, not a causal variance claim
        or any guarantee against external-test harm.
    """
    started = time.perf_counter()
    settings = _config(config)
    if learner != "lightgbm":
        raise ValueError("Fixed averaging supports explicit LightGBM classification only")
    validate_classification_frame(train_X)
    target = classification_target(train_y, len(train_X))
    classes = np.unique(target)
    base_config = PredictionBundleConfig(seed=settings.seed, threads=settings.threads, stability_repeats=3)
    fit_started = time.perf_counter()
    with threadpool_limits(limits=settings.threads):
        full, timing = _fit_full_procedures(train_X, target, learner, base_config)
        source = FixedAverageSource(full["raw"], full["fe"])
        timing["fit_seconds"] = time.perf_counter() - fit_started
        probabilities, scores, score_timing = _score_full_procedures(
            full, test_X, test_y, classes, FIXED_AVERAGE_ARMS, 0.5
        )
        timing.update(score_timing)
    diagnostics: dict[str, object] = {
        "protocol": "lightgbm_iid_fixed_average_v1",
        "blend_mode": "fixed_average",
        "task": "classification",
        "learner": learner,
        "split_kind": "iid",
        "seed": settings.seed,
        "sklearn_version": sklearn.__version__,
        "classes": classes.tolist(),
        "class_counts": [int(np.count_nonzero(target == label)) for label in classes],
        "member_count": 0,
        "members": [],
        **_full_diagnostics(full, FIXED_AVERAGE_ARMS),
        "oof_complete_procedure_fit_count": 0,
        "weight": 0.5,
        "weight_optimized": False,
        "criterion": "prespecified_probability_average_weight_0.5",
        "source_policy": {
            "oof": "not_performed",
            "inference": "complete_full_T_raw_and_fe_procedures_no_oof",
            "controls": "exact_same_full_T_objects_and_cached_probabilities_for_all_three_arms",
            "interpretation": "replication_of_prespecified_50_50_arm_no_weight_search_no_causal_variance_or_no_harm_guarantee",
            "categorical_encoding": "one_hot",
            "numeric_clip_quantiles": [0.0, 1.0],
            "max_candidates": 1000,
            "max_added_features": 50,
            "stability_repeats": 3,
            "importance_resamples": 1,
            "selection_rule": "paired_gain",
            "candidate_set": "extended",
            "include_frequency": True,
        },
    }
    _standalone_timings(timing, FIXED_AVERAGE_ARMS)
    timing["physical_bundle_wall_seconds"] = time.perf_counter() - started
    result = FixedAverageResult(scores, diagnostics, timing, source, probabilities)
    result.to_report()
    return result


def fit_fixed_average_bundle(
    train_X: pd.DataFrame,
    train_y: pd.Series | np.ndarray,
    test_X: pd.DataFrame,
    test_y: pd.Series | np.ndarray,
    *,
    learner: str,
    config: ConfigInput,
) -> FixedAverageReport:
    """Return the benchmark-only prespecified 50:50 three-arm JSON report.

    Parameters
    ----------
    train_X, train_y : pandas.DataFrame, pandas.Series or ndarray
        Outer TRAIN for independent complete raw and FE HPO/full-refit procedures.
    test_X, test_y : pandas.DataFrame, pandas.Series or ndarray
        External test, accessed only after both full fits finish.
    learner : str
        Explicitly ``"lightgbm"`` with classification/IID data.
    config : OOFBlendConfig or mapping
        Same validated settings and fixed feature budgets as the OOF benchmark.

    Returns
    -------
    FixedAverageReport
        ``raw_metrics``, ``fe_metrics``, ``average_metrics``, full FE-plan widths,
        compact full-model diagnostics/shared identities and timing. The actual
        fixed weight is 0.5; no learned-blend arm or OOF statistics are fabricated.
        Charge physical wall time once, not summed overlapping standalone costs.
        FE/model timings are components of complete fits, never additional costs.
        No models, vectors, rows or vocabularies are exported. This replication
        does not imply causal variance reduction or a no-harm guarantee.
    """
    return fit_fixed_average_bundle_result(train_X, train_y, test_X, test_y, learner=learner, config=config).to_report()

"""Experimental learner-aware classification with independent fitted feature plans.

This opt-in API does not change AutoFeatureEngineer. CPU LogisticRegression and
LGBMClassifier are verified; other cloneable numeric/OHE probabilistic classifiers
must satisfy the same interface. There is no automatic learner/mode policy, native
CatBoost representation, calibration, external evaluation, or provider access.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from featcopilot.core.feature import FeatureSet
from featcopilot.engines.predictive import PredictiveTabularEngine
from featcopilot.utils.logger import get_logger
from featcopilot.utils.model_preprocessing import (
    CategoricalEncoding,
    CrossFittedTargetEncoder,
    make_classification_pipeline,
    normalize_model_frame,
    validate_classification_frame,
)
from featcopilot.utils.numeric_clipping import validate_clip_quantiles
from featcopilot.utils.splitting import make_holdout_splits

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]
ParameterCandidates = Sequence[Mapping[str, object]]
ENSEMBLE_SEED_OFFSET = 32452843


class _FitSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    random_state: int = Field(ge=0, lt=2**32, strict=True)
    n_jobs: int = Field(ge=1, strict=True)
    stability_repeats: int = Field(ge=1, le=10, strict=True)
    max_added_features: int = Field(ge=0, le=50, strict=True)
    max_candidates: int = Field(ge=1, le=1000, strict=True)
    engineered: bool = Field(strict=True)
    categorical_encoding: CategoricalEncoding = "one_hot"


def _reset_fitted_state(estimator: BaseEstimator) -> None:
    for name in list(vars(estimator)):
        if name.endswith("_"):
            delattr(estimator, name)


def _set_params(estimator: BaseEstimator, params: dict[str, object]) -> BaseEstimator:
    if params:
        _reset_fitted_state(estimator)
    if any(key.startswith("estimator__") for key in params):
        prototype = params.get("estimator", estimator.estimator)
        params = {**params, "estimator": clone(prototype)}
    return BaseEstimator.set_params(estimator, **params)


def classification_target(y: pd.Series | np.ndarray, rows: int) -> np.ndarray:
    """Require nonmissing one-dimensional classification labels aligned by position."""
    target = np.asarray(y)
    if target.ndim != 1 or len(target) != rows or pd.isna(target).any():
        raise ValueError("Classification targets must be nonmissing, one-dimensional, and row-aligned")
    check_classification_targets(target)
    if len(np.unique(target)) < 2:
        raise ValueError("Classification requires at least two classes")
    return target


def require_classes(target: np.ndarray, classes: np.ndarray, scope: str) -> None:
    """Reject partitions missing classes without reducing folds or retrying."""
    if not np.array_equal(np.unique(target), classes):
        raise ValueError(f"Every class must be present in {scope}; no fold reduction or retries are allowed")


def inner_split(y: np.ndarray, seed: int, scope: str) -> tuple[IndexArray, IndexArray]:
    """Construct the fixed stratified 75/25 split and validate both class sets."""
    classes = np.unique(y)
    try:
        train, valid = train_test_split(
            np.arange(len(y), dtype=np.int64), test_size=0.25, random_state=seed, stratify=y
        )
    except ValueError as error:
        raise ValueError(f"Insufficient class support for {scope} fixed 75/25 inner split: {error}") from error
    require_classes(y[train], classes, f"{scope} inner TRAIN")
    require_classes(y[valid], classes, f"{scope} inner VAL")
    return train, valid


def make_subsample85_splits(y: pd.Series | np.ndarray, seed: int = 42) -> list[tuple[IndexArray, IndexArray]]:
    """Return three fixed, ordered stratified 85% bags and their complements.

    Parameters
    ----------
    y : pandas.Series or ndarray
        Classification labels in original training order.
    seed : int, default=42
        Inner seed. Sampling starts at seed + 32452843 modulo 2**32, with
        consecutive seeds. All complements and inner splits must retain classes.

    Returns
    -------
    list[tuple[ndarray, ndarray]]
        Three training/complement position pairs. Overlap and repeated bags are
        allowed; there is no redraw, row deduplication, or OOF coverage promise.
    """
    target = classification_target(y, len(y))
    classes = np.unique(target)
    splits = make_holdout_splits(
        target, task="classification", n_splits=3, random_state=(seed + ENSEMBLE_SEED_OFFSET) % 2**32, test_size=0.15
    )
    if len(splits) != 3:
        raise ValueError("subsample85 requires exactly three training/calibration pairs")
    calibration_size = int(np.ceil(0.15 * len(target)))
    for member, (train, valid) in enumerate(splits):
        for positions, size in ((train, len(target) - calibration_size), (valid, calibration_size)):
            if (
                positions.ndim != 1
                or not np.issubdtype(positions.dtype, np.integer)
                or len(positions) != size
                or np.any(positions < 0)
                or np.any(positions >= len(target))
                or len(np.unique(positions)) != len(positions)
            ):
                raise ValueError(f"member {member} requires unique in-range positions and fixed subsample85 sizes")
        if np.intersect1d(train, valid).size:
            raise ValueError(f"member {member} training/calibration positions must be disjoint")
        require_classes(target[train], classes, f"member {member} base TRAIN")
        require_classes(target[valid], classes, f"member {member} calibration HOLDOUT")
        inner_split(target[train], seed, f"member {member}")
    return splits


def validate_probabilities(probability: np.ndarray, rows: int, n_classes: int, *, normalize: bool = True) -> FloatArray:
    """Validate probabilities; optionally preserve legal native single-model roundoff."""
    values = np.asarray(probability)
    if values.shape != (rows, n_classes) or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Invalid probability shape or nonfloating dtype")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    sums = values.sum(axis=1, dtype=np.float64)
    tolerance = 8 * n_classes * np.finfo(values.dtype).eps
    if np.any(np.abs(sums - 1.0) > tolerance) or np.any(sums <= 0):
        raise ValueError("Probability rows must sum to one within dtype roundoff")
    return np.asarray(values / sums[:, None] if normalize else values, dtype=np.float64)


def aligned_probability(
    probability: np.ndarray,
    actual_classes: np.ndarray,
    expected_classes: np.ndarray,
    rows: int,
    *,
    normalize: bool = True,
) -> FloatArray:
    """Validate and align class probabilities without fabricating missing classes."""
    if len(actual_classes) != len(expected_classes) or not np.array_equal(
        np.sort(actual_classes), np.sort(expected_classes)
    ):
        raise ValueError("Retained estimator class alignment differs from the ensemble")
    positions = [int(np.flatnonzero(actual_classes == label)[0]) for label in expected_classes]
    return validate_probabilities(probability, rows, len(expected_classes), normalize=normalize)[:, positions]


def _sha256(value: object) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def positions_report(
    positions: IndexArray, y: np.ndarray, classes: np.ndarray, *, membership: bool = False
) -> dict[str, object]:
    """Describe positional membership by counts and hashes, never raw indices."""
    report: dict[str, object] = {
        "size": len(positions),
        "indices_sha256": hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest(),
        "class_counts": [int(np.count_nonzero(y[positions] == label)) for label in classes],
    }
    if membership:
        mask = np.zeros(len(y), dtype=np.uint8)
        mask[positions] = 1
        report["mask_sha256"] = hashlib.sha256(mask.tobytes()).hexdigest()
        report["fraction"] = len(positions) / len(y)
    return report


def _compact_selection(engineer: PredictiveTabularEngine | None) -> dict[str, object] | None:
    if engineer is None:
        return None
    report: Mapping[str, object] = engineer.selection_report_
    keys = (
        "task",
        "metric",
        "original_count",
        "candidate_count",
        "candidate_set",
        "selected_count",
        "selected_features",
        "screening",
        "selection_rule",
        "stability_repeats",
        "split_kind",
        "learner_fit_count",
        "ranking_scope",
        "baseline_loss",
        "numeric_preprocessing",
    )
    result: dict[str, object] = {key: report[key] for key in keys if key in report}
    frequency = engineer.selection_report_.get("frequency_policy")
    if frequency is not None:
        result["frequency_policy"] = {key: value for key, value in frequency.items() if key != "final_discarded"}
    result["importance_resamples"] = 1
    return result


def _report_parameters(parameters: dict[str, object]) -> dict[str, object]:
    # Arbitrary estimator parameters can contain credentials or unsafe repr().
    return {
        key: value if value is None or type(value) in (bool, int, float) else {"type": type(value).__name__}
        for key, value in parameters.items()
    }


def _encoded_weight_parameters(
    parameters: Mapping[str, object], prototype: Mapping[str, object], classes: np.ndarray
) -> dict[str, object]:
    effective = copy.deepcopy(dict(parameters))
    weights = effective.get("class_weight", prototype.get("class_weight"))
    if isinstance(weights, dict):
        positions = {label: index for index, label in enumerate(classes)}
        encoded = {}
        for label, weight in weights.items():
            if label not in positions:
                raise ValueError("class_weight contains a key absent from the original training labels")
            encoded[positions[label]] = copy.deepcopy(weight)
        effective["class_weight"] = encoded
    return effective


def _has_decision_function(estimator: NestedBaseClassifier) -> bool:
    return hasattr(estimator, "model_") and hasattr(estimator.model_, "decision_function")


class NestedBaseClassifier(ClassifierMixin, BaseEstimator):
    """A reusable, cloneable complete inner search and winning refit procedure.

    Parameters
    ----------
    estimator : sklearn classifier
        Bare cloneable classifier with predict_proba, not a Pipeline or regressor.
        CPU LogisticRegression and LGBMClassifier are verified. Supply learner
        seed, thread/device settings and iteration limits explicitly.
    parameter_candidates : sequence of dict
        Exactly three dictionaries of native estimator parameters, evaluated in
        order. First minimum inner log loss wins. Inputs are never mutated.
        Class-weight dictionary keys refer to original labels; owned trial/refit
        copies map them to encoded classes. Unknown keys are rejected; partial
        dictionaries retain the learner's default weight for omitted classes.
    engineered : bool, default=True
        Enable extended/frequency candidates with importance screening,
        importance_resamples=1 and paired_gain selection.
    random_state : int, default=42
        Fixed split and feature-selection seed, not an override of learner seeds.
    n_jobs : int, default=8
        Positive FE job/threadpool limit. Base procedures are fitted sequentially.
    stability_repeats : int, default=3
        Feature-selection holdout repeats, from one to ten.
    max_added_features : int, default=50
        Per-member added-feature cap, from zero to fifty.
    max_candidates : int, default=1000
        Feature candidate cap, from one to one thousand.
    numeric_clip_quantiles : tuple[float, float], default=(0.0, 1.0)
        Shared fitted min/max policy; other quantile policies are not supported.
    categorical_encoding : {"one_hot", "target"}, default="one_hot"
        Fixed cross-fitted target encoding is opt-in and requires at least five
        rows per class in every actual categorical TRAIN fit. Numeric-only
        inputs require no target encoder. Feature definitions are unchanged.
        Target mode preserves raw category types before generation and encoding;
        stored categorical-pair literals can therefore differ from OHE's legacy
        string-normalized literals, without changing the operator catalog.

    Notes
    -----
    Only IID classification is supported. Any fit metadata, sample weights, or
    external validation arguments are rejected. Each fit owns a 75/25 inner
    split, three FE/model trials and an independent winning refit on all supplied
    TRAIN rows. Normalization is stateless; preprocessing is fitted locally.
    """

    model_: Pipeline
    engineer_: PredictiveTabularEngine | None
    raw_schema_: pd.DataFrame
    classes_: np.ndarray
    n_features_in_: int
    feature_names_in_: np.ndarray
    feature_names_out_: list[str]
    selected_parameters_: dict[str, object]
    trials_: list[dict[str, object]]
    inner_split_: dict[str, dict[str, object]]
    selection_report_: dict[str, object] | None
    fit_identity_sha256_: str
    fe_fit_seconds_: float
    model_fit_seconds_: float
    fit_seconds_: float
    tuning_and_orchestration_seconds_: float
    _normalize_inner_probabilities = True
    _strict_state_fingerprint = False

    def __init__(
        self,
        estimator: BaseEstimator,
        parameter_candidates: ParameterCandidates,
        *,
        engineered: bool = True,
        random_state: int = 42,
        n_jobs: int = 8,
        stability_repeats: int = 3,
        max_added_features: int = 50,
        max_candidates: int = 1000,
        numeric_clip_quantiles: tuple[float, float] = (0.0, 1.0),
        categorical_encoding: CategoricalEncoding = "one_hot",
    ):
        self.estimator = estimator
        self.parameter_candidates = parameter_candidates
        self.engineered = engineered
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.stability_repeats = stability_repeats
        self.max_added_features = max_added_features
        self.max_candidates = max_candidates
        self.numeric_clip_quantiles = numeric_clip_quantiles
        self.categorical_encoding = categorical_encoding

    def set_params(self, **params: object) -> NestedBaseClassifier:
        """Set parameters without mutating an externally supplied estimator."""
        _set_params(self, params)
        return self

    def _validate_configuration(self) -> _FitSettings:
        settings = _FitSettings(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            stability_repeats=self.stability_repeats,
            max_added_features=self.max_added_features,
            max_candidates=self.max_candidates,
            engineered=self.engineered,
            categorical_encoding=self.categorical_encoding,
        )
        if validate_clip_quantiles(self.numeric_clip_quantiles) != (0.0, 1.0):
            raise ValueError("Experimental classification requires numeric_clip_quantiles=(0.0, 1.0)")
        if isinstance(self.estimator, Pipeline) or not is_classifier(self.estimator):
            raise TypeError("estimator must be a bare classifier, not a Pipeline or regressor")
        if not callable(getattr(self.estimator, "predict_proba", None)):
            raise TypeError("The bare classifier must implement predict_proba")
        if not isinstance(self.parameter_candidates, (list, tuple)) or len(self.parameter_candidates) != 3:
            raise ValueError("parameter_candidates must contain exactly three native parameter dictionaries")
        available = self.estimator.get_params(deep=True)
        if available.get("cat_features") is not None:
            raise ValueError("Numeric/OHE preprocessing cannot honor native cat_features; supply a numeric classifier")
        for parameters in self.parameter_candidates:
            if not isinstance(parameters, dict) or any(
                not isinstance(key, str) or key not in available for key in parameters
            ):
                raise ValueError("Each candidate must be a dictionary of valid native estimator parameter keys")
            clone(self.estimator).set_params(**copy.deepcopy(parameters))
        return settings

    def _make_model(self, X: pd.DataFrame, parameters: dict[str, object]) -> Pipeline:
        return make_classification_pipeline(
            X,
            clone(self.estimator).set_params(**copy.deepcopy(parameters)),
            numeric_clip_quantiles=self.numeric_clip_quantiles,
            categorical_encoding=self.categorical_encoding,
            random_state=self.random_state,
        )

    def _fit_procedure(
        self, X: pd.DataFrame, y: np.ndarray, parameters: dict[str, object], config: _FitSettings
    ) -> tuple[PredictiveTabularEngine | None, Pipeline, pd.DataFrame, float, float]:
        engineer = None
        features = X
        fe_seconds = 0.0
        if self.engineered:
            fe_started = time.perf_counter()
            bare = clone(self.estimator).set_params(**copy.deepcopy(parameters))
            engineer = PredictiveTabularEngine(
                max_features=X.shape[1] + config.max_added_features,
                max_candidates=config.max_candidates,
                task="classification",
                estimator=bare,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                stability_repeats=config.stability_repeats,
                screening="importance",
                selection_rule="paired_gain",
                candidate_set="extended",
                include_frequency=True,
                numeric_clip_quantiles=self.numeric_clip_quantiles,
                importance_resamples=1,
                categorical_encoding=config.categorical_encoding,
            )
            features = engineer.fit_transform(X, y)
            fe_seconds = time.perf_counter() - fe_started
        if not isinstance(features, pd.DataFrame) or not features.index.equals(X.index):
            raise TypeError("Feature engineering must preserve the input DataFrame index")
        if not X.shape[1] <= features.shape[1] <= X.shape[1] + config.max_added_features:
            raise ValueError("Feature procedure violated the per-member added-feature budget")
        model_started = time.perf_counter()
        model = self._make_model(features, parameters)
        if not isinstance(model, Pipeline):
            raise TypeError("Matched learner must be a sklearn Pipeline")
        model.fit(features, y)
        model_seconds = time.perf_counter() - model_started
        return engineer, model, features, fe_seconds, model_seconds

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray, **fit_params: object) -> NestedBaseClassifier:
        """Fit on supplied TRAIN only; unsupported metadata fails before any fitting."""
        started = time.perf_counter()
        _reset_fitted_state(self)
        if fit_params:
            raise TypeError(f"Only IID classification without fit metadata is supported: {sorted(fit_params)}")
        config = self._validate_configuration()
        X = validate_classification_frame(X)
        target = classification_target(y, len(X))
        train, valid = inner_split(target, config.random_state, "base")
        encoder = LabelEncoder().fit(target[train])
        encoded = np.asarray(encoder.transform(target))
        schema = X.iloc[:0].copy()
        preserve_types = config.categorical_encoding == "target"
        inner_X = normalize_model_frame(
            X.iloc[train].iloc[:0], X.iloc[train], preserve_categorical_types=preserve_types
        )
        valid_X = normalize_model_frame(
            X.iloc[train].iloc[:0], X.iloc[valid], preserve_categorical_types=preserve_types
        )
        candidates = [copy.deepcopy(dict(parameters)) for parameters in self.parameter_candidates]
        fit_candidates = [
            _encoded_weight_parameters(parameters, self.estimator.get_params(deep=False), encoder.classes_)
            for parameters in candidates
        ]
        trials: list[dict[str, object]] = []
        losses: list[float] = []
        total_fe_seconds = 0.0
        total_model_seconds = 0.0
        with threadpool_limits(limits=config.n_jobs):
            for parameters, fit_parameters in zip(candidates, fit_candidates, strict=True):
                engineer, model, features, fe_seconds, model_seconds = self._fit_procedure(
                    inner_X, encoded[train], fit_parameters, config
                )
                total_fe_seconds += fe_seconds
                total_model_seconds += model_seconds
                validation = valid_X if engineer is None else engineer.transform(valid_X)
                probability = aligned_probability(
                    np.asarray(model.predict_proba(validation)),
                    np.asarray(model.classes_),
                    np.arange(len(encoder.classes_)),
                    len(valid),
                    normalize=self._normalize_inner_probabilities,
                )
                loss = float(log_loss(encoded[valid], probability, labels=np.arange(len(encoder.classes_))))
                if not np.isfinite(loss):
                    raise ValueError("Non-finite inner validation log loss")
                losses.append(loss)
                trials.append(
                    {
                        "parameters": dict(parameters),
                        "loss": loss,
                        "n_output_features": features.shape[1],
                        "selection_report": _compact_selection(engineer),
                    }
                )
            best = int(np.argmin(losses))
            parameters = dict(candidates[best])
            engineer, model, features, fe_seconds, model_seconds = self._fit_procedure(
                normalize_model_frame(schema, X, preserve_categorical_types=preserve_types),
                encoded,
                fit_candidates[best],
                config,
            )
            total_fe_seconds += fe_seconds
            total_model_seconds += model_seconds
        classes = np.asarray(encoder.classes_)
        inner_report = {
            "train": positions_report(train, target, classes),
            "validation": positions_report(valid, target, classes),
        }
        fit_seconds = time.perf_counter() - started
        remainder = fit_seconds - total_fe_seconds - total_model_seconds
        if remainder < -max(1e-9, fit_seconds * 1e-12):
            raise RuntimeError("Sequential FE/model stage timings exceed base fit wall time")
        self.__dict__.update(
            model_=model,
            engineer_=engineer,
            raw_schema_=schema,
            classes_=classes,
            n_features_in_=X.shape[1],
            feature_names_in_=np.asarray(X.columns, dtype=object),
            feature_names_out_=list(features.columns),
            selected_parameters_=parameters,
            trials_=trials,
            inner_split_=inner_report,
            selection_report_=_compact_selection(engineer),
            fit_identity_sha256_=_sha256((uuid.uuid4().hex, id(self))),
            fe_fit_seconds_=total_fe_seconds,
            model_fit_seconds_=total_model_seconds,
            fit_seconds_=fit_seconds,
            tuning_and_orchestration_seconds_=max(0.0, remainder),
        )
        return self

    def _features(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["model_", "raw_schema_", "classes_"])
        normalized = normalize_model_frame(
            self.raw_schema_, X, preserve_categorical_types=self.categorical_encoding == "target"
        )
        return normalized if self.engineer_ is None else self.engineer_.transform(normalized)

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        """Transform through this member's own FE plan and fitted preprocessing."""
        return self._probability(X, normalize=True)

    def _probability(self, X: pd.DataFrame, *, normalize: bool) -> FloatArray:
        features = self._features(X)
        return aligned_probability(
            np.asarray(self.model_.predict_proba(features)),
            np.asarray(self.model_.classes_),
            np.arange(len(self.classes_)),
            len(features),
            normalize=normalize,
        )

    @available_if(_has_decision_function)
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Expose native margins only if this member's fitted pipeline supports them."""
        values = np.asarray(self.model_.decision_function(self._features(X)))
        if not np.isfinite(values).all():
            raise ValueError("Non-finite native decision function")
        return values

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return one-dimensional original class labels by native probability argmax."""
        probability = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(probability, axis=1)])

    def get_feature_plan(self) -> FeatureSet:
        """Return this member's defensive fitted plan, including private fitted maps."""
        check_is_fitted(self, ["model_"])
        return FeatureSet() if self.engineer_ is None else self.engineer_.get_feature_set()

    def diagnostics(self, *, warn_on_unavailable: bool = True) -> dict[str, object]:
        """Return diagnostics with explicit unavailable hashes for unpickleable state.

        Parameters
        ----------
        warn_on_unavailable : bool, default=True
            Log once per fitted member if pickle cannot fingerprint its state.
            The enclosing public classifier suppresses member warnings and emits
            one aggregate warning per fit instead. Exporting the actual model
            still uses ordinary pickle/joblib behavior and may fail.
        """
        check_is_fitted(self, ["model_", "classes_"])
        features = list(self.get_feature_plan())
        formula = [(feature.name, feature.source_columns, feature.transformation, feature.code) for feature in features]
        schema = [(column, str(self.raw_schema_[column].dtype)) for column in self.raw_schema_]
        trials = []
        for trial in self.trials_:
            parameters = trial["parameters"]
            if not isinstance(parameters, dict):
                raise TypeError("Fitted trial parameters must be dictionaries")
            trials.append({**trial, "parameters": _report_parameters(parameters)})
        unavailable_reason = None
        try:
            fitted_state_hash = _sha256((self.model_, features, schema, self.classes_))
        except (pickle.PicklingError, TypeError, AttributeError) as error:
            if self._strict_state_fingerprint:
                raise
            fitted_state_hash = None
            unavailable_reason = (
                f"{type(error).__name__}: fitted state is not serializable by pickle; "
                "in-memory prediction remains available"
            )
            if not getattr(self, "_fingerprint_warning_emitted_", False):
                if warn_on_unavailable:
                    logger.warning("Fitted-state fingerprint unavailable: %s", unavailable_reason)
                self._fingerprint_warning_emitted_ = True
        report: dict[str, object] = {
            "fit_identity_sha256": self.fit_identity_sha256_,
            "selected_parameters": _report_parameters(self.selected_parameters_),
            "trials": copy.deepcopy(trials),
            "inner_split": copy.deepcopy(self.inner_split_),
            "feature_names": self.feature_names_out_.copy(),
            "n_output_features": len(self.feature_names_out_),
            "feature_names_sha256": _sha256(self.feature_names_out_),
            "schema_sha256": _sha256(schema),
            "formula_sha256": _sha256(formula),
            "fitted_state_sha256": fitted_state_hash,
            "selection_report": copy.deepcopy(self.selection_report_),
            "response_method": "decision_function" if hasattr(self, "decision_function") else "predict_proba",
            "fit_seconds": self.fit_seconds_,
            "fe_fit_seconds": self.fe_fit_seconds_,
            "model_fit_seconds": self.model_fit_seconds_,
            "tuning_and_orchestration_seconds": self.tuning_and_orchestration_seconds_,
        }
        if unavailable_reason is not None:
            report["fitted_state_unavailable_reason"] = unavailable_reason
        if self.categorical_encoding == "target":
            preprocessing = self.model_.steps[1][1]
            if not isinstance(preprocessing, ColumnTransformer):
                raise TypeError("Target-encoded model diagnostics require a fitted ColumnTransformer")
            categorical = preprocessing.named_transformers_["categorical"]
            columns = next(columns for name, _, columns in preprocessing.transformers_ if name == "categorical")
            if len(columns):
                if not isinstance(categorical, CrossFittedTargetEncoder):
                    raise TypeError("Target encoding requires the shared cross-fitted categorical transformer")
                encoding_report = categorical.get_encoding_report()
            else:
                encoding_report = {
                    "requested": "target",
                    "active": False,
                    "reason": "no_categorical_columns",
                    "n_input_features": 0,
                    "output_width": 0,
                    "training_output_bytes": 0,
                }
            report["encoding_report"] = encoding_report
            report["model_input_width"] = max(span.stop for span in preprocessing.output_indices_.values())
        return report


class _SingleBaseClassifier(NestedBaseClassifier):
    # The incumbent single control scores native probabilities; retained-ensemble
    # bases normalize legal roundoff before inner scoring. Keep both exact.
    _normalize_inner_probabilities = False


class FeatureEnsembleClassifier(ClassifierMixin, BaseEstimator):
    """Experimental prediction artifact with independent fitted feature plans.

    Parameters
    ----------
    estimator : sklearn classifier
        Bare cloneable probabilistic classifier. CPU LogisticRegression and
        LGBMClassifier are verified; supply native learner randomness/thread/device
        controls explicitly. No native categorical learner adapter is selected.
    parameter_candidates : sequence of dict
        Exactly three native-parameter dictionaries for each member's inner search.
        Class-weight dictionaries use original labels, including NumPy scalar
        keys; unknown class keys fail before model fitting.
    mode : {"single", "subsample85", "raw_fe_average"}
        Required explicit mode. Single refits the winner on all supplied training
        rows. Subsample85 fits three 85% bags and averages native probabilities;
        it never fits a calibrator or a subsequent all-training model.
        Raw_fe_average independently fits complete raw then FE procedures on all
        supplied TRAIN and uses a fixed 50:50 probability average. No blend-weight
        optimization, OOF weight fitting, or calibration is performed.
    engineered : bool, default=True
        Enable the fixed extended/frequency, importance, paired-gain FE procedure.
        False is the matched raw control. In raw_fe_average mode it fits only one
        raw full-TRAIN procedure, with weight one and no FE procedure.
    random_state : int, default=42
        Split and FE seed, not an override of the bare estimator's random state.
    n_jobs : int, default=8
        Positive FE/threadpool limit; members fit sequentially.
    stability_repeats : int, default=3
        Internal FE holdout repeats (one to ten).
    max_added_features : int, default=50
        Per-member added-feature cap (zero to fifty), never a feature union cap.
    max_candidates : int, default=1000
        FE candidate cap (one to one thousand).
    numeric_clip_quantiles : tuple[float, float], default=(0.0, 1.0)
        Shared post-generation fitted min/max clipping. Other policies are rejected.
    categorical_encoding : {"one_hot", "target"}, default="one_hot"
        Categorical model representation, shared by feature selection and final
        prediction. Target mode uses fixed five-fold cross-fitting, requires
        sklearn >=1.4 only for active categorical blocks, and adds no feature operators.
        It preserves raw category types before feature generation, unlike OHE's
        legacy string normalization. Fitted categorical literals may differ.

    Attributes
    ----------
    estimators_ : tuple[NestedBaseClassifier, ...]
        Actual fitted complete procedures, each with its own raw schema and FE plan.
    classes_ : ndarray
        Original class labels in probability-column order.
    member_reports_ : list[dict]
        Compact per-member selection, feature, timing and sampling diagnostics.
    weights_ : tuple[float, ...]
        Present only for raw_fe_average: (0.5, 0.5), or (1.0,) for its raw control.
    member_roles_ : tuple[str, ...]
        Present only for raw_fe_average, in fitted order: ("raw", "fe") or ("raw",).

    Notes
    -----
    DataFrame-only, IID classification. Group/time metadata, sample weights,
    external validation sets and all additional fit arguments are rejected.
    This is a prediction artifact: there is no common transform or union feature
    plan. Pickle/joblib artifacts contain private fitted maps and should be
    treated as trusted models; diagnostics() excludes vocabularies and raw rows.
    In-memory prediction does not require pickle support: nonserializable state
    has a null diagnostic fingerprint and an explicit unavailable reason.
    Raw_fe_average preserves normalized member probabilities and performs no
    additional normalization of their convex average. Its raw control uses that
    same member normalization, unlike single's native-probability convention.
    Development evidence for this mode is limited to one-hot LightGBM; it is not
    a SOTA, no-harm, or target-encoding improvement claim and changes no default.
    """

    estimators_: tuple[NestedBaseClassifier, ...]
    classes_: np.ndarray
    n_features_in_: int
    feature_names_in_: np.ndarray
    member_reports_: list[dict[str, object]]
    fit_seconds_: float
    weights_: tuple[float, ...]
    member_roles_: tuple[str, ...]

    def __init__(
        self,
        estimator: BaseEstimator,
        parameter_candidates: ParameterCandidates,
        *,
        mode: Literal["single", "subsample85", "raw_fe_average"],
        engineered: bool = True,
        random_state: int = 42,
        n_jobs: int = 8,
        stability_repeats: int = 3,
        max_added_features: int = 50,
        max_candidates: int = 1000,
        numeric_clip_quantiles: tuple[float, float] = (0.0, 1.0),
        categorical_encoding: CategoricalEncoding = "one_hot",
    ):
        self.estimator = estimator
        self.parameter_candidates = parameter_candidates
        self.mode = mode
        self.engineered = engineered
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.stability_repeats = stability_repeats
        self.max_added_features = max_added_features
        self.max_candidates = max_candidates
        self.numeric_clip_quantiles = numeric_clip_quantiles
        self.categorical_encoding = categorical_encoding

    def set_params(self, **params: object) -> FeatureEnsembleClassifier:
        """Set parameters without mutating the caller's estimator prototype."""
        _set_params(self, params)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray, **fit_params: object) -> FeatureEnsembleClassifier:
        """Fit independent complete TRAIN-local members, publishing state atomically."""
        started = time.perf_counter()
        _reset_fitted_state(self)
        if fit_params:
            raise TypeError(f"Only IID classification without fit metadata is supported: {sorted(fit_params)}")
        if not isinstance(self.mode, str) or self.mode not in ("single", "subsample85", "raw_fe_average"):
            raise ValueError("mode must explicitly be 'single', 'subsample85', or 'raw_fe_average'")
        base_type = _SingleBaseClassifier if self.mode == "single" else NestedBaseClassifier
        template = base_type(
            self.estimator,
            self.parameter_candidates,
            engineered=self.engineered,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            stability_repeats=self.stability_repeats,
            max_added_features=self.max_added_features,
            max_candidates=self.max_candidates,
            numeric_clip_quantiles=self.numeric_clip_quantiles,
            categorical_encoding=self.categorical_encoding,
        )
        template._validate_configuration()
        X = validate_classification_frame(X)
        target = classification_target(y, len(X))
        classes = np.unique(target)
        splits = (
            [(np.arange(len(X), dtype=np.int64), np.empty(0, dtype=np.int64))]
            if self.mode != "subsample85"
            else make_subsample85_splits(target, self.random_state)
        )
        fixed_average = self.mode == "raw_fe_average"
        if fixed_average and self.engineered:
            splits = splits * 2
        weights = (0.5, 0.5) if self.engineered else (1.0,)
        roles = ("raw", "fe") if self.engineered else ("raw",)
        members = []
        reports = []
        with threadpool_limits(limits=self.n_jobs):
            for index, (train, heldout) in enumerate(splits):
                member = clone(template)
                if fixed_average:
                    member.set_params(engineered=roles[index] == "fe")
                member.fit(X.iloc[train], target[train])
                if not np.array_equal(member.classes_, classes):
                    raise ValueError("Member classes differ from the complete training class set")
                members.append(member)
                report = {
                    "member": index,
                    **member.diagnostics(warn_on_unavailable=False),
                    "train": positions_report(train, target, classes, membership=True),
                }
                if self.mode == "subsample85":
                    report["heldout"] = positions_report(heldout, target, classes, membership=True)
                    report["sampling_seed"] = (self.random_state + ENSEMBLE_SEED_OFFSET + index) % 2**32
                if fixed_average:
                    report["role"] = roles[index]
                    report["weight"] = weights[index]
                reports.append(report)
        unavailable_members = sum(report["fitted_state_sha256"] is None for report in reports)
        if unavailable_members:
            logger.warning(
                "Fitted-state fingerprint unavailable for %d member(s); "
                "see fitted_state_unavailable_reason; in-memory prediction remains available",
                unavailable_members,
            )
        self.__dict__.update(
            estimators_=tuple(members),
            classes_=classes,
            n_features_in_=X.shape[1],
            feature_names_in_=np.asarray(X.columns, dtype=object),
            member_reports_=reports,
            fit_seconds_=time.perf_counter() - started,
            **({"weights_": weights, "member_roles_": roles} if fixed_average else {}),
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        """Average aligned probabilities of the actual retained fitted members."""
        check_is_fitted(self, ["estimators_", "classes_"])
        if hasattr(self, "weights_"):
            probability = [
                aligned_probability(member.predict_proba(X), member.classes_, self.classes_, len(X), normalize=False)
                for member in self.estimators_
            ]
            if len(probability) == 1:
                return probability[0]
            return validate_probabilities(
                0.5 * probability[0] + 0.5 * probability[1], len(X), len(self.classes_), normalize=False
            )
        if len(self.estimators_) == 1:
            member = self.estimators_[0]
            if not np.array_equal(member.classes_, self.classes_):
                raise ValueError("Member class alignment differs from the fitted classifier")
            return member._probability(X, normalize=False)
        probability = [
            aligned_probability(member.predict_proba(X), member.classes_, self.classes_, len(X))
            for member in self.estimators_
        ]
        return validate_probabilities(np.mean(probability, axis=0), len(X), len(self.classes_))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return one-dimensional original labels by mean-probability argmax."""
        probability = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(probability, axis=1)])

    def get_feature_plans(self) -> tuple[FeatureSet, ...]:
        """Return independent defensive member plans, never a fabricated union."""
        check_is_fitted(self, ["estimators_"])
        return tuple(member.get_feature_plan() for member in self.estimators_)

    def diagnostics(self) -> dict[str, object]:
        """Return a JSON-safe report, excluding models, vocabularies and predictions."""
        check_is_fitted(self, ["estimators_"])
        widths = [len(member.feature_names_out_) for member in self.estimators_]
        report: dict[str, object] = {
            "experimental": True,
            "mode": self.mode,
            "engineered": self.engineered,
            "split_kind": "iid",
            "member_count": len(self.estimators_),
            "n_input_features": self.n_features_in_,
            "member_output_feature_counts": widths,
            "max_member_output_features": max(widths),
            "members": copy.deepcopy(self.member_reports_),
            "fit_seconds": self.fit_seconds_,
            "base_fit_seconds": sum(member.fit_seconds_ for member in self.estimators_),
            "base_fe_seconds": sum(member.fe_fit_seconds_ for member in self.estimators_),
        }
        if hasattr(self, "weights_"):
            report.update(
                weights=list(self.weights_),
                member_roles=list(self.member_roles_),
                fe_fitted="fe" in self.member_roles_,
                weight_optimized=False,
                oof_weight_fit_count=0,
            )
        json.dumps(report, allow_nan=False)
        return report

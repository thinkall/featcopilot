"""Opt-in, inductive numeric feature discovery using residual screening.

The internal holdout selects a feature plan, not an unbiased performance estimate.
Production retains stateless arithmetic and, when explicitly enabled, frozen
categorical-pair definitions. Original columns remain available downstream.
"""

import hashlib
import json
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations, islice
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

import numpy as np
import pandas as pd
from joblib import effective_n_jobs
from numpy.typing import ArrayLike
from pydantic import ConfigDict, Field, field_validator, model_validator
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.engines.tabular import TabularEngine
from featcopilot.utils.frequency import apply_frequency_mapping, fit_frequency_mapping
from featcopilot.utils.logger import get_logger
from featcopilot.utils.model_preprocessing import encode_class_weights
from featcopilot.utils.numeric_clipping import NUMERIC_GUARD_LIMIT as _NUMERIC_LIMIT
from featcopilot.utils.numeric_clipping import NUMERIC_GUARD_POLICY as _NUMERIC_POLICY
from featcopilot.utils.numeric_clipping import (
    NumericQuantileClipper,
    guard_numeric_values,
    validate_clip_quantiles,
)
from featcopilot.utils.splitting import (
    SplitMetadata,
    make_cv_splits,
    make_holdout_splits,
    validate_split_metadata,
)
from featcopilot.utils.task import resolve_task

logger = get_logger(__name__)

_SUBSET_SIZES = (1, 5, 10, 25, 50)
_MIN_RELATIVE_IMPROVEMENT = 0.001
_IMPORTANCE_BAG_FRACTION = 0.85
_FREQUENCY_OPERATOR = "fitted_frequency_v1"
_FREQUENCY_MAX_QUOTA = 32
_FREQUENCY_QUOTA_DIVISOR = 10
_CATEGORY_MAX_LEVELS = 8
_CATEGORY_MAX_INDICATORS_PER_PAIR = 4
_CATEGORY_MIN_COUNT = 3
_CATEGORY_MIN_FREQUENCY = 0.01
_CATEGORY_MISSING_POLICY = "null_and_nonfinite_float_as_level"
_CategoryKind = Literal["missing", "str", "bool", "int", "float"]
_CategoryScalar = str | bool | int | float | None
_CategoryToken = tuple[_CategoryKind, _CategoryScalar]
_UNARY_EXPRESSIONS = {
    "square": "x**2",
    "log": "np.log1p(np.abs(x))",
    "sqrt": "np.sqrt(np.abs(x))",
    "cube": "x**3",
    "reciprocal": "1 / (x + 1e-8)",
    "tanh": "np.tanh(x)",
    "sin": "np.sin(x)",
    "cos": "np.cos(x)",
}


@runtime_checkable
class PredictiveEstimator(Protocol):
    """Cloneable numerical learner; inheriting sklearn BaseEstimator is not required."""

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return sklearn-compatible constructor parameters."""
        ...

    def set_params(self, **params: object) -> "PredictiveEstimator":
        """Update parameters on an independent selection clone."""
        ...

    def fit(self, X: np.ndarray, y: np.ndarray, /) -> object:
        """Fit preprocessed numerical rows and a one-dimensional target."""
        ...

    def predict(self, X: np.ndarray, /) -> ArrayLike:
        """Predict from preprocessed numerical rows."""
        ...


def _clone_estimator(estimator: PredictiveEstimator) -> PredictiveEstimator:
    methods = ("get_params", "set_params", "fit", "predict")
    if not all(callable(getattr(estimator, name, None)) for name in methods):
        raise ValueError("estimator must implement callable get_params, set_params, fit and predict methods")
    try:
        cloned: object = clone(estimator)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"estimator must be cloneable with sklearn.base.clone: {exc}") from exc
    if (
        cloned is estimator
        or not isinstance(cloned, PredictiveEstimator)
        or not all(callable(getattr(cloned, name, None)) for name in methods)
    ):
        raise ValueError("sklearn.base.clone must return an independent estimator implementing the learner interface")
    return cloned


class SubsetValidationScore(TypedDict):
    """Repeated internal losses for a predeclared ranked prefix, including zero."""

    count: int
    loss: float
    loss_std: float
    loss_se: float
    losses: list[float]


class DiscardedCandidate(TypedDict):
    """Train-only exact-filter provenance, with no row values."""

    name: str
    reason: Literal["constant", "duplicate", "never_positive"]
    duplicate_of: str | None
    duplicate_kind: Literal["original", "candidate"] | None


class ImportanceRankingReport(TypedDict):
    """Diagnostics for TRAIN-only importance ranking."""

    planned_count: int
    retained_count: int
    discarded_constant_count: int
    discarded_duplicate_count: int
    discarded_candidates: list[DiscardedCandidate]
    importance_source: str | None
    fit_rows: int
    learner_fits: int


class FrequencyPlanningReport(TypedDict):
    """Bounded frequency allocation for one internal TRAIN partition."""

    quota: int
    examined_columns: int
    planned_count: int
    constant_skipped: int
    base_planned_count: int
    fit_rows: int


class FrequencyPolicyReport(TypedDict):
    """Frequency policy and final refitting diagnostics, without vocabulary."""

    operator: str
    quota_max: int
    quota_divisor: int
    normalization: str
    missing_policy: str
    unseen_policy: str
    key_policy: str
    effective_output_cap: int
    validated_selected_count: int
    final_refit_count: int
    final_refit_rows: int
    selected_frequency_count: int
    final_discarded: list[DiscardedCandidate]


class NumericPreprocessingReport(TypedDict):
    """Explicit model-input clipping policy, without fitted bound values."""

    policy: str
    quantiles: list[float]
    numeric_guard: str
    guard_limit: float
    placement: str
    boolean_policy: str
    missing_policy: str
    all_missing_policy: str
    scope: str


class ImportanceBagReport(TypedDict):
    """One IID bag's counts and hashed positional membership, never its rows."""

    bag: int
    seed: int
    train_rows: int
    excluded_rows: int
    position_mask_sha256: str
    retained_count: int
    constant_count: int
    duplicate_count: int
    positive_count: int
    frequency_refit_count: int
    learner_fits: int


class ImportanceResamplingReport(TypedDict):
    """Descriptive mean-rank diagnostics, not stability-selection guarantees."""

    requested_bags: int
    completed_bags: int
    sample_fraction: float
    sampling: str
    aggregation: str
    rank_ties: str
    nonpositive_policy: str
    eligibility: str
    score_kind: str
    scope_rows: int
    universe_count: int
    duplicate_bag_mask_count: int
    bags: list[ImportanceBagReport]
    eligible_count: int
    never_positive_count: int
    positive_counts: dict[str, int]
    mean_ranks: dict[str, float]
    never_positive_features: list[str]


class ImportanceResamplingSelectionReport(TypedDict):
    """Final count-cap and reranking behavior of the opt-in procedure."""

    validated_selected_count: int
    delivered_count: int
    final_ranking_performed: bool
    procedure_level_validation_only: bool


class _ImportanceReportFields(TypedDict, total=False):
    importance_ranking: ImportanceRankingReport
    frequency_planning: FrequencyPlanningReport
    importance_resampling: ImportanceResamplingReport


class RepeatSelectionReport(_ImportanceReportFields):
    """One internal boundary's training-only ranking and heldout comparisons."""

    repeat: int
    train_rows: int
    purged_train_rows: int
    validation_rows: int
    candidate_count: int
    candidate_family_counts: dict[str, int]
    screened_count: int
    candidate_ranking: list[str]
    candidate_scores: dict[str, float]
    baseline_loss: float
    validation_scores: list[SubsetValidationScore]
    selected_count: int
    screening_scored_rows: int
    screening_unscored_rows: int


class SelectionReport(TypedDict, total=False):
    """JSON-compatible diagnostics, empty until fitting succeeds."""

    task: str
    metric: str
    original_count: int
    numeric_count: int
    train_rows: int
    validation_rows: int
    candidate_count: int
    candidate_set: str
    candidate_family_counts: dict[str, int]
    categorical_pair_policy: dict[str, int | float | str] | None
    screened_count: int
    candidate_ranking: list[str]
    candidate_scores: dict[str, float]
    residual_cv_folds: int
    baseline_loss: float
    validation_scores: list[SubsetValidationScore]
    selected_count: int
    selected_features: list[str]
    internal_validation_only: bool
    numeric_policy: str
    stability_repeats: int
    screening: str
    split_kind: str
    selection_rule: str
    uncertainty_note: str
    baseline_loss_std: float
    baseline_loss_se: float
    baseline_losses: list[float]
    count_by_repeat: list[int]
    repeats: list[RepeatSelectionReport]
    ranking_scope: str
    ranking_scored_rows: int
    ranking_unscored_rows: int
    chronological_prefix_unscored: bool
    learner_fit_count: int
    label_end_times_supplied: bool
    label_availability_policy: str
    purged_train_rows_by_repeat: list[int]
    importance_ranking: ImportanceRankingReport
    frequency_policy: FrequencyPolicyReport
    numeric_preprocessing: NumericPreprocessingReport
    categorical_preprocessing: dict[str, Any]
    importance_resampling: ImportanceResamplingReport
    importance_resampling_selection: ImportanceResamplingSelectionReport


@dataclass
class _ResidualFold:
    train: np.ndarray
    validation: np.ndarray
    train_residuals: np.ndarray
    validation_residuals: np.ndarray


@dataclass
class _Ranking:
    features: list[Feature]
    scores: dict[str, float]
    folds: int = 0
    scored_rows: int = 0
    unscored_rows: int = 0
    learner_fits: int = 0
    discarded: list[DiscardedCandidate] = field(default_factory=list)
    importance_source: str | None = None
    fit_rows: int = 0
    resampling: ImportanceResamplingReport | None = None


@dataclass
class _Repeat:
    train: np.ndarray
    validation: np.ndarray
    candidates: list[Feature]
    numeric_count: int
    ranking: _Ranking
    losses: dict[int, float]
    frequency_planning: FrequencyPlanningReport | None = None


def _finite_clip(values: np.ndarray) -> np.ndarray:
    """Apply the generated-value policy, leaving missing/undefined results missing."""
    return guard_numeric_values(values)


def _category_value(value: object, column: str) -> _CategoryToken:
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return "missing", None
    if isinstance(value, (str, np.str_)):
        return "str", str(value)
    if isinstance(value, (bool, np.bool_)):
        return "bool", bool(value)
    if isinstance(value, (int, np.integer)):
        return "int", int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return "missing", None
        return "float", numeric if numeric else 0.0
    raise pd.errors.ParserError(f"Column {column!r} contains unsupported categorical scalar values")


class PredictiveTabularEngineConfig(EngineConfig):
    """Validated configuration for opt-in predictive feature discovery."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = "PredictiveTabularEngine"
    max_features: int | None = Field(default=None, ge=0, description="Total output budget, retaining all originals")
    max_candidates: int = Field(default=1000, ge=0, le=1000, description="Maximum numeric candidates per repeat")
    task: Literal["auto", "classification", "regression"] = "auto"
    estimator: PredictiveEstimator | None = Field(
        default=None, description="Unfitted or fitted cloneable numerical learner with the sklearn parameter API"
    )
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)
    n_jobs: int = Field(default=1, ge=-1, description="Worker and native-thread limit; -1 uses all available CPUs")
    stability_repeats: int = Field(default=3, ge=1, le=10, description="Number of internal selection boundaries")
    screening: Literal["auto", "tree", "linear", "importance"] = "auto"
    selection_rule: Literal["one_se", "paired_gain"] = "one_se"
    candidate_set: Literal["standard", "extended"] = "standard"
    include_frequency: bool = Field(default=False, strict=True)
    numeric_clip_quantiles: tuple[float, float] | None = None
    importance_resamples: int = Field(default=1, strict=True)
    categorical_encoding: Literal["one_hot", "target"] = "one_hot"

    @model_validator(mode="after")
    def _validate_categorical_encoding(self) -> "PredictiveTabularEngineConfig":
        if self.categorical_encoding == "target":
            if self.screening != "importance":
                raise ValueError("categorical_encoding='target' requires explicit screening='importance'")
            if self.task == "regression":
                raise ValueError("categorical_encoding='target' supports classification only")
        return self

    @field_validator("importance_resamples", mode="before")
    @classmethod
    def _validate_importance_resamples(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value not in (1, 5):
            raise ValueError("importance_resamples must be 1 or 5")
        return int(value)

    @model_validator(mode="after")
    def _validate_resampled_importance(self) -> "PredictiveTabularEngineConfig":
        if self.importance_resamples != 1 and self.screening != "importance":
            raise ValueError("importance_resamples=5 requires explicit screening='importance'")
        return self

    @field_validator("numeric_clip_quantiles", mode="before")
    @classmethod
    def _validate_clip_quantiles(cls, value: object) -> tuple[float, float] | None:
        return validate_clip_quantiles(value)

    @model_validator(mode="after")
    def _validate_numeric_clipping(self) -> "PredictiveTabularEngineConfig":
        if self.numeric_clip_quantiles is not None and self.screening != "importance":
            raise ValueError("numeric_clip_quantiles requires explicit screening='importance'")
        return self

    @model_validator(mode="after")
    def _validate_frequency_screening(self) -> "PredictiveTabularEngineConfig":
        if self.include_frequency and self.screening != "importance":
            raise ValueError("include_frequency=True requires explicit screening='importance'")
        return self

    @field_validator("estimator")
    @classmethod
    def _validate_estimator(cls, value: PredictiveEstimator | None) -> PredictiveEstimator | None:
        if value is not None:
            _clone_estimator(value)
        return value

    @field_validator("n_jobs")
    @classmethod
    def _validate_n_jobs(cls, value: int) -> int:
        if value == 0:
            raise ValueError("n_jobs must be -1 or a positive integer")
        return value


class PredictiveTabularEngine(BaseEngine):
    """
    Select numeric features by incremental, training-only predictive signal.

    Candidate families are interleaved, not order-truncated by family. Each
    internal holdout has its own training-only candidate plan and ranking.
    Residual screening uses nested cross-fitting: no baseline producing a
    screening-training residual has seen that screening fold's test labels.
    Explicit importance screening instead fits one intended learner jointly on
    training originals and a bounded, exactly deduplicated candidate matrix.
    Classification residuals are one-hot targets minus class probabilities.
    Counts 0, 1, 5, 10, 25 and 50 (clamped to the common available budget) compete
    on mean internal loss. The smallest count within one standard error of the
    best is preferred, requiring >0.1% relative improvement over originals.
    These overlapping internal splits give descriptive uncertainty, not an
    unbiased external performance estimate or confidence interval.

    Parameters
    ----------
    max_features : int or None, default=None
        Total output-column limit. All original columns are retained even when
        they already exceed this limit. At most 50 generated features are selected.
    max_candidates : int, default=1000
        Hard cap on raw planned candidates per repeat, at most 1000. Residual
        screening materializes one candidate at a time; importance screening
        materializes the retained candidates jointly. Subset evaluation uses
        at most 50. Discards never trigger additional candidate generation.
    task : {"auto", "classification", "regression"}, default="auto"
        Explicit overrides are recommended for integer-valued regression targets.
        Auto follows sklearn's binary/multiclass/continuous target convention.
    estimator : PredictiveEstimator or None, default=None
        Bare sklearn-compatible learner, accepting preprocessed numeric matrices.
        Requires callable ``get_params``, ``set_params``, ``fit`` and ``predict``
        methods and an independent sklearn clone, not BaseEstimator inheritance.
        It is cloned for every fit and never mutated. Classifiers must implement
        ``predict_proba``. Data-bound, fixed-column preprocessing is not supported.
        Defaults to the task-appropriate histogram gradient boosting estimator.
    random_state : int, default=42
        Seed for splits, screening trees and unspecified learner random states.
    n_jobs : int, default=1
        Positive limit for native thread pools and learner ``n_jobs`` parameters.
        Use -1 for all available CPUs, resolved through joblib for native thread
        limits. When overriding worker settings, concrete LogisticRegression
        clones use one task worker with lbfgs/newton-cg and auto/deprecated/multinomial mode
        (including versions without multi_class), avoiding a process pool for a
        single binary/multinomial optimization. Explicit OVR, other solvers,
        subclasses and other estimators retain the usual worker policy.
        Candidate evaluation is sequential to bound memory.
    verbose : bool, default=False
        Log selection counts and losses.
    stability_repeats : int, default=3
        Number of independent seeded IID/group holdouts or disjoint forward
        validation windows, from 1 to 10. With default single-fit importance,
        one retains single-holdout selection and its training-partition ranking.
        Multiple repeats, or enabled importance resampling, refit a positive-count
        final ranking on all supplied training rows using the first repeat's
        bounded candidate pool after choosing the feature count.
    screening : {"auto", "tree", "linear", "importance"}, default="auto"
        Auto uses regularized linear residual screening for bare sklearn linear
        learners (including Ridge/LogisticRegression), shallow trees otherwise.
        Importance is explicitly opt-in: one cloned intended learner is fitted
        on the current TRAIN originals plus candidates, using native
        ``feature_importances_`` or normalized absolute coefficients (mean over
        classes). LightGBM ranking clones use gain when importance_type is
        supported; other fit parameters are unchanged. Unsupported or
        misaligned importance attributes raise. The independent holdout still
        makes the subset decision. Auto never selects importance implicitly.
        The default histogram learner lacks these attributes; supply a
        compatible intended learner when requesting importance screening.
    selection_rule : {"one_se", "paired_gain"}, default="one_se"
        Existing smaller-subset selection rule or paired-gain comparison.
    candidate_set : {"standard", "extended"}, default="standard"
        Standard preserves the original numeric candidate stream exactly.
        Extended adds rowwise sums/minima/maxima and lazily balanced numeric
        pairs. It also proposes categorical-pair indicators using only each
        internal holdout's TRAIN frequencies, never targets. At most eight
        frequent levels per categorical column and four indicators per column
        pair are retained, each with support >= max(3, ceil(0.01 * train_rows)).
        At most max_candidates categorical column pairs are inspected.
    include_frequency : bool, default=False
        Strict opt-in for unary training-frequency features, requiring explicit
        screening="importance". Available with either candidate set. Examine
        at most min(32, max_candidates // 10, supported_columns) deterministically
        spaced original columns; append nonconstant frequencies after the
        unchanged base prefix of length max_candidates - frequency_count.
        Unused quota returns to the base pool. Budgets below ten add no frequency
        candidates. Current subset selection still permits at most 50 additions.
    numeric_clip_quantiles : tuple[float, float] or None, default=None
        Explicit model-input policy requiring screening="importance". Each
        internal learner independently fits numeric, non-boolean TRAIN quantile
        bounds after the existing +/-1e30 finite guard, before imputation and
        scaling. Quantiles satisfy 0 <= low < high <= 1; no quantile search is
        performed. None preserves the existing pipeline exactly. Raw outputs,
        candidate generation and exported feature code remain unclipped by this
        policy. A downstream learner must use the same shared clipper explicitly.
    importance_resamples : {1, 5}, default=1
        One preserves the existing full-TRAIN importance fit exactly. Five
        requires explicit importance screening and IID input: five 85% samples
        without replacement (stratified for classification) use context seed
        plus bag index. Each bag refits its frequency maps and preprocessing,
        applies exact bag-local constant/duplicate rules, and returns candidate
        importances to the same planned definition universe. Average within-bag
        ranks use average ties; nonpositive/unusable entries tie last.
        Candidates never positive in any bag are excluded before count
        validation. Positive-count final ranking repeats the same five-bag
        procedure even with stability_repeats=1. No post-gate consensus cut is
        applied. This empirical ranking has no PFER/FDR or no-harm guarantee.
    categorical_encoding : {"one_hot", "target"}, default="one_hot"
        Model-input categorical representation. Target mode requires IID
        classification with importance screening, using fold-local five-fold
        cross-fitted target encoding. Raw outputs and candidate formulas remain
        unchanged; the downstream model must use the same encoding policy.
    **kwargs : dict
        Additional base configuration, such as ``enabled``.

    Attributes
    ----------
    selection_report_ : dict
        JSON-compatible counts, ordered ``candidate_ranking``, training-only
        ``candidate_scores`` (cross-validated relative residual MSE reduction,
        or native importance/normalized absolute coefficients in importance mode),
        ``baseline_loss``, ``validation_scores`` (including count zero, with
        ``loss``, ``loss_std``, ``loss_se`` and per-repeat ``losses``),
        ``selected_features``, ``selected_count``, ``task`` and ``metric``.
        ``repeats`` contains each boundary's training-only ranking, losses and
        scored/unscored row counts; ``count_by_repeat`` exposes count instability.
        ``baseline_loss`` is the mean across repeats, not the final refit's score.
        ``numeric_policy``, ``screening``, ``split_kind`` and ``ranking_scope``
        identify the representation and protocol. With the default three repeats
        there are at most 42 learner fits for residual screening, plus bounded
        candidate-screening fits, or 22 learner fits for single-fit joint
        importance. Five-bag ranking raises that bound to 38; with one holdout
        its positive-count bound is 16, including the same-policy final ranking.
        ``label_end_times_supplied``, ``label_availability_policy`` and
        ``purged_train_rows_by_repeat`` describe label-availability handling.
        ``candidate_set`` and ``candidate_family_counts`` identify pool coverage.
        ``categorical_pair_policy`` contains only bounds/policies, not levels.
        Enabled frequency features add ``frequency_policy`` and per-repeat
        ``frequency_planning`` diagnostics. A final map refit can make a
        frequency constant/duplicate; it is then removed, and actual output
        count can be below the validated cap without further learner fits.
        Enabled model-input clipping adds ``numeric_preprocessing`` describing
        the fixed policy and guard, not numerical bounds or raw values.
        Enabled resampling adds ``importance_resampling`` per repeat and at
        top level, plus ``importance_resampling_selection`` for the validated
        cap, delivered count and final-ranking status. ``candidate_scores``
        then means negative average candidate rank (higher is better), not
        importance magnitude. Ranking ``fit_rows`` counts repeated row
        presentations across bags; scope size is reported separately.
        Explicit importance mode adds ``importance_ranking`` at top level and
        per repeat: raw/retained counts, exact-discard provenance, importance
        source, fit rows and learner fits. Residual CV row counts are zero/
        unscored in that mode, not an in-sample performance claim.
        ``internal_validation_only`` is always true. No data, labels, indices,
        fitted preprocessors or fitted selection learners are retained, except
        the bounded category values in explicitly enabled frozen pair definitions.

    Notes
    -----
    Originals, including categorical columns, retain their values and dtypes.
    Generated infinities and undefined arithmetic become NaN, never batch-fitted
    imputations. Internal numeric preprocessing uses train-fitted median
    imputation and standard scaling; categoricals use constant missing-value
    imputation and train-fitted OHE with unknown categories ignored. Generated
    finite outputs are clipped to +/-1e30 in every stage, including exported code;
    overflowing arithmetic becomes NaN. Original values are never changed by
    transform. The internal learner's numeric preprocessing also bounds its
    original-column inputs for float32-based learner safety.
    Columns must have unique string names. Transform accepts reordered columns,
    but rejects missing/extra columns and returns the fitted output-column order.
    Products and differences use canonical column order; ratios include both
    directions. Reverse differences are omitted as exact sign aliases.

    Importance screening alone removes generated columns that are constant on
    their nonmissing TRAIN values (including entirely missing columns), or
    exactly equal to an original numeric/earlier retained candidate under the
    finite-value policy. Signed zeros compare equal; missing positions must
    match. Hash buckets are verified by equality, not trusted as proof.
    No approximate correlation pruning or original-column removal is performed.

    Extended categorical pairs support object/string/category/bool columns
    containing string, boolean, integer, float or missing scalars. Types remain
    distinct (e.g. 1, 1.0, True and "1"). Nulls and nonfinite floats share an
    explicit missing level; unseen pairs yield zero. Unsupported nonscalar
    values raise without including their contents in the error. Generated names
    are hashes, but model artifacts (Feature metadata and exported code) contain
    the learned category values and must be treated accordingly. Datetime-typed
    columns are retained, not expanded into categorical pairs.

    Frequency maps reuse fitted-rule key/serialization semantics: training
    counts divided by ALL training rows; unknown, missing and nonfinite keys
    return zero; strings/booleans/numbers differ, while numeric 1 and 1.0 share
    a key. Each repeat has independent TRAIN-only maps. After subset decisions,
    maps are refitted on the allowed full fit data (all pool frequency maps
    before final ranking, selected maps only in single-fit/single-holdout mode).
    Runtime and registered exported code use the same frozen lookup helper.
    Artifacts contain learned vocabulary, never raw training rows or targets.
    """

    def __init__(
        self,
        max_features: int | None = None,
        max_candidates: int = 1000,
        task: str = "auto",
        estimator: PredictiveEstimator | None = None,
        random_state: int = 42,
        n_jobs: int = 1,
        verbose: bool = False,
        stability_repeats: int = 3,
        screening: str = "auto",
        selection_rule: str = "one_se",
        candidate_set: str = "standard",
        include_frequency: bool = False,
        numeric_clip_quantiles: tuple[float, float] | None = None,
        importance_resamples: int = 1,
        categorical_encoding: str = "one_hot",
        **kwargs: Any,
    ):
        config = PredictiveTabularEngineConfig.model_validate(
            {
                **kwargs,
                "max_features": max_features,
                "max_candidates": max_candidates,
                "task": task,
                "estimator": estimator,
                "random_state": random_state,
                "n_jobs": n_jobs,
                "verbose": verbose,
                "stability_repeats": stability_repeats,
                "screening": screening,
                "selection_rule": selection_rule,
                "candidate_set": candidate_set,
                "include_frequency": include_frequency,
                "numeric_clip_quantiles": numeric_clip_quantiles,
                "importance_resamples": importance_resamples,
                "categorical_encoding": categorical_encoding,
            }
        )
        super().__init__(config=config)
        self.config: PredictiveTabularEngineConfig = config
        self._reset()

    @property
    def include_frequency(self) -> bool:
        """Whether training-fitted unary frequency candidates are enabled."""
        return self.config.include_frequency

    def _reset(self) -> None:
        self._is_fitted = False
        self._feature_names = []
        self._feature_metadata = {}
        self._input_columns: list[str] = []
        self._numeric_columns: list[str] = []
        self._selected_features: list[Feature] = []
        self._class_labels: np.ndarray | None = None
        self._feature_set = FeatureSet()
        self.selection_report_: SelectionReport = {}

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        *,
        groups: ArrayLike | None = None,
        timestamps: ArrayLike | None = None,
        label_end_times: ArrayLike | None = None,
        **kwargs: Any,
    ) -> "PredictiveTabularEngine":
        """
        Fit a stateless feature plan using training-only internal boundaries.

        Parameters
        ----------
        X : DataFrame or ndarray
            Original mixed features with at least eight rows.
        y : Series or ndarray
            Required one-dimensional, nonconstant target, aligned positionally.
        groups : array-like or None, default=None
            Positionally aligned entity labels. Every holdout and nested residual
            fold keeps entities disjoint. Residual screening needs at least four
            training groups in each holdout (normally six groups overall), with
            class coverage inside each training partition.
        timestamps : array-like or None, default=None
            Positionally aligned finite numeric or datetime timestamps. Equal
            timestamps stay together and every validation block is strictly
            after its training prefix. Residual screening needs at least six
            training time blocks per holdout (twelve overall with three repeats).
            The first prefix has no OOF residual and is explicitly
            unscored. Groups and timestamps together are not supported.
        label_end_times : array-like or None, default=None
            Positionally aligned label-availability times, requiring timestamps
            in the same numeric units or datetime domain. End times must be at
            least prediction timestamps. At EVERY holdout and nested residual
            fold, training labels must be available strictly before the earliest
            validation timestamp. Purged rows are not moved into validation.
            No random fallback is used if insufficient labeled history remains.
        **kwargs : dict
            Reserved; unsupported arguments raise TypeError rather than being
            silently ignored. Sample weights and external validation sets are
            not currently accepted.

        Returns
        -------
        self : PredictiveTabularEngine
            Fitted engine with selected feature metadata and selection report.

        Raises
        ------
        ValueError
            If targets, schema, generated names or the internal split are invalid.
        TypeError
            If input types or additional fit arguments are unsupported.
        """
        self._reset()
        if kwargs:
            raise TypeError(f"Unsupported fit arguments: {sorted(kwargs)}")
        if self.config.importance_resamples != 1 and any(
            value is not None for value in (groups, timestamps, label_end_times)
        ):
            raise ValueError(
                "importance_resamples=5 supports IID only; groups/timestamps/label_end_times are unsupported"
            )
        if self.config.categorical_encoding == "target" and any(
            value is not None for value in (groups, timestamps, label_end_times)
        ):
            raise ValueError("categorical_encoding='target' supports IID only; group/time metadata is unsupported")
        X = self._validate_frame(X)
        target, task = self._validate_target(y, len(X))
        if self.config.categorical_encoding == "target" and task != "classification":
            raise ValueError("categorical_encoding='target' supports classification only")
        boundaries = self._validate_boundaries(groups, timestamps, len(X), label_end_times=label_end_times)
        self._input_columns = X.columns.tolist()
        self._numeric_columns = sorted(column for column in X.columns if pd.api.types.is_numeric_dtype(X[column].dtype))
        n_classes = 0
        if task == "classification":
            encoder = LabelEncoder().fit(target)
            self._class_labels = encoder.classes_.copy()
            target = encoder.transform(target)
            n_classes = len(encoder.classes_)

        budget = _SUBSET_SIZES[-1]
        if self.config.max_features is not None:
            budget = min(budget, max(0, self.config.max_features - len(self._input_columns)))
        partitions = self._holdouts(target, task, boundaries)
        template = self._learner(task)
        screening = self._resolve_screening(template)
        repeats: list[_Repeat] = []
        learner_fits = 0

        with threadpool_limits(limits=effective_n_jobs(self.config.n_jobs)):
            for repeat_index, (train_indices, validation_indices) in enumerate(partitions):
                train = X.iloc[train_indices]
                candidates, numeric_count = self._plan_candidates(train, budget)
                ranking = self._rank_candidates(
                    candidates,
                    train,
                    self._model_frame(train),
                    target[train_indices],
                    template,
                    task,
                    n_classes,
                    boundaries=boundaries.subset(train_indices),
                    seed=self._seed(repeat_index),
                    screening=screening,
                )
                learner_fits += ranking.learner_fits
                frequency_planning = (
                    self._frequency_planning_report(train, budget, candidates)
                    if self.config.include_frequency
                    else None
                )
                repeats.append(
                    _Repeat(
                        train_indices, validation_indices, candidates, numeric_count, ranking, {}, frequency_planning
                    )
                )

            common_budget = min(budget, *(len(repeat.ranking.features) for repeat in repeats))
            sizes = sorted({0, *(min(size, common_budget) for size in _SUBSET_SIZES)})
            for repeat in repeats:
                train = X.iloc[repeat.train]
                validation = X.iloc[repeat.validation]
                prepared_train = self._model_frame(train)
                prepared_validation = self._model_frame(validation)
                for size in sizes:
                    features = repeat.ranking.features[:size]
                    augmented_train = self._augment(prepared_train, train, features) if size else prepared_train
                    augmented_validation = (
                        self._augment(prepared_validation, validation, features) if size else prepared_validation
                    )
                    learner = self._fit_selection_learner(augmented_train, target[repeat.train], template)
                    learner_fits += 1
                    repeat.losses[size] = self._loss(
                        learner, augmented_validation, target[repeat.validation], task, n_classes
                    )

            validation_scores = [
                self._summarize_losses(size, [repeat.losses[size] for repeat in repeats]) for size in sizes
            ]
            selected_count = self._choose_count(validation_scores, self.config.selection_rule)
            validated_selected_count = selected_count
            frequency_refit_count = 0
            final_frequency_discarded: list[DiscardedCandidate] = []
            final_ranking = repeats[0].ranking
            final_ranking_performed = selected_count > 0 and (
                self.config.stability_repeats > 1 or self.config.importance_resamples > 1
            )
            if final_ranking_performed:
                final_candidates = repeats[0].candidates
                if self.config.include_frequency:
                    final_candidates, frequency_refit_count = self._refit_frequency_features(final_candidates, X)
                final_ranking = self._rank_candidates(
                    final_candidates,
                    X,
                    self._model_frame(X),
                    target,
                    template,
                    task,
                    n_classes,
                    boundaries=boundaries,
                    seed=self._seed(self.config.stability_repeats),
                    screening=screening,
                )
                learner_fits += final_ranking.learner_fits
                if frequency_refit_count:
                    frequency_names = {
                        feature.name for feature in final_candidates if feature.transformation == _FREQUENCY_OPERATOR
                    }
                    final_frequency_discarded = [
                        item for item in final_ranking.discarded if item["name"] in frequency_names
                    ]
            if len(final_ranking.features) < selected_count:
                if frequency_refit_count or self.config.importance_resamples > 1:
                    selected_count = len(final_ranking.features)
                else:
                    raise ValueError("Final training ranking cannot supply the validated feature count")
            selected_features = final_ranking.features[:selected_count]
            if self.config.include_frequency and not final_ranking_performed and selected_count > 0:
                selected_features, frequency_refit_count = self._refit_frequency_features(selected_features, X)
                if frequency_refit_count:
                    selected_features, _, final_frequency_discarded = self._deduplicate_candidates(
                        selected_features, X, self._model_frame(X)
                    )
                    selected_count = len(selected_features)

        self._selected_features = selected_features
        self._feature_names = [feature.name for feature in self._selected_features]
        self._feature_set = FeatureSet(self._selected_features)
        self._feature_metadata = deepcopy({feature.name: feature.to_dict() for feature in self._selected_features})
        repeat_reports: list[RepeatSelectionReport] = []
        for index, repeat in enumerate(repeats):
            repeat_scores = [self._summarize_losses(size, [repeat.losses[size]]) for size in sizes]
            purged_train_rows = 0
            if boundaries.label_end_times is not None and boundaries.values is not None:
                cutoff = boundaries.values[repeat.validation].min()
                purged_train_rows = int(np.count_nonzero(boundaries.values < cutoff)) - len(repeat.train)
            repeat_reports.append(
                {
                    "repeat": index,
                    "train_rows": len(repeat.train),
                    "purged_train_rows": purged_train_rows,
                    "validation_rows": len(repeat.validation),
                    "candidate_count": len(repeat.candidates),
                    "candidate_family_counts": dict(Counter(feature.transformation for feature in repeat.candidates)),
                    "screened_count": len(repeat.ranking.features),
                    "candidate_ranking": [feature.name for feature in repeat.ranking.features],
                    "candidate_scores": repeat.ranking.scores,
                    "baseline_loss": repeat.losses[0],
                    "validation_scores": repeat_scores,
                    "selected_count": self._choose_count(repeat_scores, self.config.selection_rule),
                    "screening_scored_rows": repeat.ranking.scored_rows,
                    "screening_unscored_rows": repeat.ranking.unscored_rows,
                }
            )
            if screening == "importance":
                repeat_reports[-1]["importance_ranking"] = self._importance_report(
                    repeat.ranking, len(repeat.candidates)
                )
            if repeat.frequency_planning is not None:
                repeat_reports[-1]["frequency_planning"] = repeat.frequency_planning
            if repeat.ranking.resampling is not None:
                repeat_reports[-1]["importance_resampling"] = deepcopy(repeat.ranking.resampling)
        baseline_summary = validation_scores[0]
        self.selection_report_ = {
            "task": task,
            "metric": "log_loss" if task == "classification" else "rmse",
            "original_count": len(self._input_columns),
            "numeric_count": repeats[0].numeric_count,
            "train_rows": len(repeats[0].train),
            "validation_rows": len(repeats[0].validation),
            "candidate_count": len(repeats[0].candidates),
            "candidate_set": self.config.candidate_set,
            "candidate_family_counts": dict(Counter(feature.transformation for feature in repeats[0].candidates)),
            "categorical_pair_policy": (
                {
                    "max_levels_per_column": _CATEGORY_MAX_LEVELS,
                    "max_indicators_per_pair": _CATEGORY_MAX_INDICATORS_PER_PAIR,
                    "max_column_pairs": self.config.max_candidates,
                    "min_count": _CATEGORY_MIN_COUNT,
                    "min_frequency": _CATEGORY_MIN_FREQUENCY,
                    "missing_policy": _CATEGORY_MISSING_POLICY,
                    "unseen_policy": "zero",
                }
                if self.config.candidate_set == "extended"
                else None
            ),
            "screened_count": len(final_ranking.features),
            "candidate_ranking": [feature.name for feature in final_ranking.features],
            "candidate_scores": final_ranking.scores,
            "residual_cv_folds": final_ranking.folds,
            "baseline_loss": baseline_summary["loss"],
            "validation_scores": validation_scores,
            "selected_count": selected_count,
            "selected_features": self._feature_names.copy(),
            "internal_validation_only": True,
            "numeric_policy": _NUMERIC_POLICY,
            "stability_repeats": self.config.stability_repeats,
            "screening": screening,
            "split_kind": boundaries.kind,
            "selection_rule": (
                "smallest_within_one_se_and_0.1pct_relative_gain"
                if self.config.selection_rule == "one_se"
                else "positive_paired_gain_and_smallest_within_paired_one_se"
            ),
            "uncertainty_note": "Descriptive across overlapping internal training splits; not a confidence interval.",
            "baseline_loss_std": baseline_summary["loss_std"],
            "baseline_loss_se": baseline_summary["loss_se"],
            "baseline_losses": baseline_summary["losses"],
            "count_by_repeat": [report["selected_count"] for report in repeat_reports],
            "repeats": repeat_reports,
            "ranking_scope": (
                "baseline_retained_no_final_refit"
                if selected_count == 0
                else (
                    "single_holdout_train"
                    if self.config.stability_repeats == 1
                    else "all_fit_rows_first_repeat_candidates"
                )
            ),
            "ranking_scored_rows": final_ranking.scored_rows,
            "ranking_unscored_rows": final_ranking.unscored_rows,
            "chronological_prefix_unscored": boundaries.kind == "temporal" and screening != "importance",
            "learner_fit_count": learner_fits,
            "label_end_times_supplied": boundaries.label_end_times is not None,
            "label_availability_policy": (
                "strictly_before_validation_start" if boundaries.label_end_times is not None else "not_requested"
            ),
            "purged_train_rows_by_repeat": [report["purged_train_rows"] for report in repeat_reports],
        }
        if screening == "importance":
            self.selection_report_["importance_ranking"] = self._importance_report(
                final_ranking, len(repeats[0].candidates)
            )
        if self.config.include_frequency:
            self.selection_report_["frequency_policy"] = {
                "operator": _FREQUENCY_OPERATOR,
                "quota_max": _FREQUENCY_MAX_QUOTA,
                "quota_divisor": _FREQUENCY_QUOTA_DIVISOR,
                "normalization": "training_count_divided_by_all_training_rows",
                "missing_policy": "zero",
                "unseen_policy": "zero",
                "key_policy": "fitted_rule_v1_numeric_int_float_equivalence",
                "effective_output_cap": common_budget,
                "validated_selected_count": validated_selected_count,
                "final_refit_count": frequency_refit_count,
                "final_refit_rows": len(X) if frequency_refit_count else 0,
                "selected_frequency_count": sum(
                    feature.transformation == _FREQUENCY_OPERATOR for feature in self._selected_features
                ),
                "final_discarded": deepcopy(final_frequency_discarded),
            }
            if validated_selected_count > 0 and selected_count == 0:
                self.selection_report_["ranking_scope"] = "baseline_retained_after_frequency_refit"
        if self.config.importance_resamples > 1:
            if final_ranking.resampling is None:
                raise RuntimeError("Resampled importance ranking did not provide its diagnostics")
            self.selection_report_["importance_resampling"] = deepcopy(final_ranking.resampling)
            self.selection_report_["importance_resampling_selection"] = {
                "validated_selected_count": validated_selected_count,
                "delivered_count": selected_count,
                "final_ranking_performed": final_ranking_performed,
                "procedure_level_validation_only": True,
            }
            if final_ranking_performed:
                self.selection_report_["ranking_scope"] = (
                    "all_fit_rows_resampled_importance"
                    if selected_count > 0
                    else "baseline_retained_after_resampled_ranking"
                )
        if self.config.numeric_clip_quantiles is not None:
            self.selection_report_["numeric_preprocessing"] = {
                "policy": "train_quantile_clip_v1",
                "quantiles": list(self.config.numeric_clip_quantiles),
                "numeric_guard": _NUMERIC_POLICY,
                "guard_limit": _NUMERIC_LIMIT,
                "placement": "post_generation_before_imputation_and_scaling",
                "boolean_policy": "passthrough",
                "missing_policy": "preserve_nan",
                "all_missing_policy": "guard_only_no_fitted_bounds",
                "scope": "each_selection_learner_training_partition",
            }
        if self.config.categorical_encoding == "target":
            categorical_count = sum(not pd.api.types.is_numeric_dtype(X[column].dtype) for column in X)
            self.selection_report_["categorical_preprocessing"] = {
                "requested": "target",
                "active": categorical_count > 0,
                "categorical_input_columns": categorical_count,
                "categorical_output_columns": categorical_count * (n_classes if n_classes > 2 else 1),
                "target_type": "multiclass" if n_classes > 2 else "binary",
                "cv": 5,
                "smooth": "auto",
                "random_state": self.config.random_state,
                "training_matrix": "cross_fitted_fit_transform",
                "inference_matrix": "full_training_mapping_transform",
                "scope": "each_selection_learner_training_partition",
            }
        self._is_fitted = True
        if self.config.verbose:
            if selected_count != validated_selected_count:
                if self.config.importance_resamples > 1:
                    logger.info(
                        "PredictiveTabularEngine: delivered %d features after resampled ranking; validated cap was %d",
                        selected_count,
                        validated_selected_count,
                    )
                else:
                    logger.info(
                        "PredictiveTabularEngine: delivered %d features after frequency refit; validated count was %d",
                        selected_count,
                        validated_selected_count,
                    )
            else:
                logger.info(
                    "PredictiveTabularEngine: selected %d/%d candidates; internal %s %.6g -> %.6g",
                    selected_count,
                    len(repeats[0].candidates),
                    self.selection_report_["metric"],
                    baseline_summary["loss"],
                    next(score["loss"] for score in validation_scores if score["count"] == selected_count),
                )
        return self

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        *,
        groups: ArrayLike | None = None,
        timestamps: ArrayLike | None = None,
        label_end_times: ArrayLike | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit with positional boundary metadata and transform without reusing it."""
        self.fit(X, y, groups=groups, timestamps=timestamps, label_end_times=label_end_times, **kwargs)
        return self.transform(X)

    def transform(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> pd.DataFrame:
        """
        Return original columns plus selected, batch-invariant numeric features.

        Parameters
        ----------
        X : DataFrame or ndarray
            Rows with the fitted input schema, optionally reordered.
        **kwargs : dict
            Reserved; unsupported arguments raise TypeError.

        Returns
        -------
        DataFrame
            Exact input index, original values/dtypes in fitted order, then
            selected float columns. Missing generated values are not imputed.
        """
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")
        if kwargs:
            raise TypeError(f"Unsupported transform arguments: {sorted(kwargs)}")
        X = self._validate_frame(X)
        if set(X.columns) != set(self._input_columns):
            missing = sorted(set(self._input_columns) - set(X.columns))
            extra = sorted(set(X.columns) - set(self._input_columns))
            raise ValueError(f"Input schema differs from fit: missing={missing}, extra={extra}")
        for column in self._numeric_columns:
            if not pd.api.types.is_numeric_dtype(X[column].dtype):
                raise TypeError(f"Column {column!r} must remain numeric at transform time")
        result = X.loc[:, self._input_columns].copy()
        generated = {feature.name: self._compute(feature, X) for feature in self._selected_features}
        if generated:
            result = pd.concat([result, pd.DataFrame(generated, index=X.index)], axis=1)
            result.columns.name = X.columns.name
        return result

    def get_feature_set(self) -> FeatureSet:
        """Get a defensive copy of selected generated-feature metadata."""
        return deepcopy(self._feature_set)

    def _validate_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray) and X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        frame = self._validate_input(X)
        if not frame.columns.is_unique:
            raise ValueError("X must have unique column names")
        if not all(isinstance(column, str) for column in frame.columns):
            raise ValueError("X column names must be strings")
        if len(frame.columns) == 0:
            raise ValueError("X must contain at least one original feature")
        if any(pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes):
            raise TypeError("Complex numeric columns are not supported")
        if self.config.candidate_set == "extended":
            for column in frame.columns:
                if self._is_categorical_column(frame[column]):
                    for value in frame[column].astype(object):
                        _category_value(value, column)
        return frame

    def _validate_target(self, y: pd.Series | np.ndarray | None, n_rows: int) -> tuple[np.ndarray, str]:
        if y is None:
            raise ValueError("y is required to fit predictive feature selection")
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {target.shape}")
        if len(target) != n_rows:
            raise ValueError("X and y must contain the same number of rows")
        if pd.isna(target).any():
            raise ValueError("y must not contain missing values")
        target = pd.Series(target).infer_objects().to_numpy()
        if pd.api.types.is_complex_dtype(target.dtype):
            raise ValueError("Complex targets are not supported; y must contain real numbers or class labels")
        if pd.api.types.is_numeric_dtype(target.dtype) and not np.isfinite(target).all():
            raise ValueError("y must contain only finite values")
        if pd.Series(target).nunique() < 2:
            raise ValueError("y must not be constant")
        task = resolve_task(target, self.config.task)
        return target.astype(float) if task == "regression" else target, task

    _validate_boundaries = staticmethod(validate_split_metadata)

    def _seed(self, offset: int) -> int:
        return (self.config.random_state + offset) % (2**32)

    def _holdouts(
        self, target: np.ndarray, task: str, boundaries: SplitMetadata
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(target) < 8:
            raise ValueError("At least 8 samples are required for internal holdout selection")
        return make_holdout_splits(
            target,
            task,
            self.config.stability_repeats,
            self.config.random_state,
            groups=boundaries.groups,
            timestamps=boundaries.timestamps,
            label_end_times=boundaries.label_end_times,
            allow_constant_target=False,
        )

    def _cv_splits(
        self,
        target: np.ndarray,
        task: str,
        boundaries: SplitMetadata,
        n_splits: int,
        seed: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return make_cv_splits(
            target,
            task,
            n_splits,
            seed,
            groups=boundaries.groups,
            timestamps=boundaries.timestamps,
            label_end_times=boundaries.label_end_times,
            allow_constant_target=False,
        )

    @staticmethod
    def _summarize_losses(count: int, losses: list[float]) -> SubsetValidationScore:
        values = np.asarray(losses)
        scale = float(np.max(np.abs(values)))
        normalized = values / scale if scale else values
        mean = float(normalized.mean()) * scale
        std = float(normalized.std(ddof=1)) * scale if len(values) > 1 else 0.0
        return {"count": count, "loss": mean, "loss_std": std, "loss_se": std / np.sqrt(len(values)), "losses": losses}

    @staticmethod
    def _choose_count(scores: list[SubsetValidationScore], rule: str = "one_se") -> int:
        if rule == "paired_gain":
            baseline_score = next(score for score in scores if score["count"] == 0)
            baseline_values = np.asarray(baseline_score["losses"], dtype=float)
            minimum_gain = baseline_score["loss"] * _MIN_RELATIVE_IMPROVEMENT
            eligible_scores = []
            for score in scores:
                gains = baseline_values - np.asarray(score["losses"], dtype=float)
                summary = PredictiveTabularEngine._summarize_losses(score["count"], gains.tolist())
                if score["count"] > 0 and summary["loss"] > max(minimum_gain, summary["loss_se"]):
                    eligible_scores.append(score)
            if not eligible_scores:
                return 0
            best = min(eligible_scores, key=lambda score: (score["loss"], score["count"]))
            for score in sorted(eligible_scores, key=lambda score: score["count"]):
                gaps = np.asarray(score["losses"], dtype=float) - np.asarray(best["losses"], dtype=float)
                summary = PredictiveTabularEngine._summarize_losses(score["count"], gaps.tolist())
                if summary["loss"] <= max(summary["loss_se"], best["loss"] * _MIN_RELATIVE_IMPROVEMENT):
                    return score["count"]
            return best["count"]
        if rule != "one_se":
            raise ValueError(f"Unknown selection rule: {rule!r}")
        baseline = next(score["loss"] for score in scores if score["count"] == 0)
        best = min(scores, key=lambda score: (score["loss"], score["count"]))
        threshold = best["loss"] + max(best["loss_se"], best["loss"] * _MIN_RELATIVE_IMPROVEMENT)
        if baseline <= threshold:
            return 0
        eligible = [
            score["count"]
            for score in scores
            if score["loss"] <= threshold and score["loss"] < baseline * (1 - _MIN_RELATIVE_IMPROVEMENT)
        ]
        return min(eligible, default=0)

    def _resolve_screening(self, learner: PredictiveEstimator) -> str:
        if self.config.screening != "auto":
            return self.config.screening
        linear = any(cls.__module__.startswith("sklearn.linear_model.") for cls in type(learner).__mro__)
        return "linear" if linear else "tree"

    def _plan_candidates(self, train: pd.DataFrame, budget: int) -> tuple[list[Feature], int]:
        unique = {column: pd.Series(self._values(train[column])).nunique() for column in self._numeric_columns}
        eligible = [column for column, count in unique.items() if count > 1]
        continuous = [column for column, count in unique.items() if count > 2]
        frequency_features: list[Feature] = []
        if self.config.include_frequency:
            for column in self._frequency_columns(train, budget):
                mapping = fit_frequency_mapping(train[column])
                values = apply_frequency_mapping(train[column], mapping).to_numpy()
                if len(values) and not np.all(values == values[0]):
                    frequency_features.append(self._frequency_feature(column, mapping, len(train), "internal_train"))
        base_budget = self.config.max_candidates - len(frequency_features)
        candidates = (
            list(islice(self._candidate_stream(eligible, continuous, train=train), base_budget)) if budget else []
        )
        candidates.extend(frequency_features)
        self._check_collisions(candidates)
        return candidates, len(eligible)

    def _frequency_columns(self, train: pd.DataFrame, budget: int) -> list[str]:
        if not budget or self.config.max_candidates < _FREQUENCY_QUOTA_DIVISOR:
            return []
        supported = sorted(
            column
            for column in train.columns
            if pd.api.types.is_numeric_dtype(train[column].dtype) or self._is_categorical_column(train[column])
        )
        quota = min(_FREQUENCY_MAX_QUOTA, self.config.max_candidates // _FREQUENCY_QUOTA_DIVISOR, len(supported))
        if quota == 0:
            return []
        if quota == 1:
            return [supported[len(supported) // 2]]
        return [supported[index * (len(supported) - 1) // (quota - 1)] for index in range(quota)]

    def _frequency_planning_report(
        self, train: pd.DataFrame, budget: int, candidates: list[Feature]
    ) -> FrequencyPlanningReport:
        quota = len(self._frequency_columns(train, budget))
        planned = sum(feature.transformation == _FREQUENCY_OPERATOR for feature in candidates)
        return {
            "quota": quota,
            "examined_columns": quota,
            "planned_count": planned,
            "constant_skipped": quota - planned,
            "base_planned_count": len(candidates) - planned,
            "fit_rows": len(train),
        }

    @staticmethod
    def _frequency_feature(
        column: str, mapping: dict[str, float], fit_rows: int, scope: Literal["internal_train", "final_fit"]
    ) -> Feature:
        definition = json.dumps([_FREQUENCY_OPERATOR, column], ensure_ascii=True, separators=(",", ":"))
        name = f"frequency_{hashlib.sha256(definition.encode('utf-8')).hexdigest()[:24]}"
        frozen = dict(sorted(mapping.items()))
        return Feature(
            name=name,
            dtype=FeatureType.NUMERIC,
            origin=FeatureOrigin.AGGREGATION,
            source_columns=[column],
            transformation=_FREQUENCY_OPERATOR,
            explanation=f"Frozen training frequency of {column}; missing and unknown values return zero",
            code=f"result = {_FREQUENCY_OPERATOR}(df[{column!r}], {frozen!r})",
            metadata={
                "numeric_policy": _NUMERIC_POLICY,
                "operation": "frequency",
                "frequency_state": {"version": 1, "fit_rows": fit_rows, "scope": scope, "mapping": frozen},
            },
        )

    @staticmethod
    def _frequency_mapping(feature: Feature) -> dict[str, float]:
        state: object = feature.metadata.get("frequency_state")
        if not isinstance(state, dict):
            raise ValueError(f"Invalid frozen frequency state for {feature.name!r}")
        rows = state.get("fit_rows")
        if (
            type(state.get("version")) is not int
            or state["version"] != 1
            or type(rows) is not int
            or rows < 1
            or state.get("scope") not in ("internal_train", "final_fit")
        ):
            raise ValueError(f"Invalid frozen frequency state for {feature.name!r}")
        mapping = state.get("mapping")
        if not isinstance(mapping, dict):
            raise ValueError(f"Invalid frozen frequency mapping for {feature.name!r}")
        validated: dict[str, float] = {}
        for key, value in mapping.items():
            if not isinstance(key, str) or type(value) is not float or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid frozen frequency mapping for {feature.name!r}")
            validated[key] = value
        return validated

    def _refit_frequency_features(
        self,
        features: list[Feature],
        train: pd.DataFrame,
        *,
        scope: Literal["internal_train", "final_fit"] = "final_fit",
    ) -> tuple[list[Feature], int]:
        refitted: list[Feature] = []
        count = 0
        for feature in features:
            if feature.transformation == _FREQUENCY_OPERATOR:
                column = feature.source_columns[0]
                replacement = self._frequency_feature(column, fit_frequency_mapping(train[column]), len(train), scope)
                replacement.importance = feature.importance
                refitted.append(replacement)
                count += 1
            else:
                refitted.append(feature)
        return refitted, count

    @staticmethod
    def _values(column: pd.Series) -> np.ndarray:
        values = column.to_numpy(dtype=float, na_value=np.nan)
        return np.where(np.isfinite(values), values, np.nan)

    def _model_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        columns: dict[str, Any] = {}
        for column in sorted(frame.columns):
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series.dtype):
                if self.config.numeric_clip_quantiles is not None and pd.api.types.is_bool_dtype(series.dtype):
                    columns[column] = series.copy()
                else:
                    columns[column] = _finite_clip(self._values(series))
            else:
                if self.config.categorical_encoding == "target":
                    columns[column] = series.copy()
                    continue
                # Prefixing values distinguishes e.g. integer 1 from string "1"
                # and reserves the unprefixed imputer sentinel for missingness.
                columns[column] = series.astype(object).map(
                    lambda value: np.nan if pd.isna(value) else f"{type(value).__name__}:{value}"
                )
        return pd.DataFrame(columns, index=frame.index)

    def _learner(self, task: str) -> PredictiveEstimator:
        if self.config.estimator is None:
            cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
            default_learner: PredictiveEstimator = cls(
                max_iter=100, max_leaf_nodes=15, early_stopping=False, random_state=self.config.random_state
            )
            return default_learner
        learner = _clone_estimator(self.config.estimator)
        if (task == "classification" and is_regressor(learner)) or (task == "regression" and is_classifier(learner)):
            raise ValueError(f"The injected estimator is incompatible with task={task!r}")
        if task == "classification" and not callable(getattr(learner, "predict_proba", None)):
            raise ValueError("Classification estimators must implement predict_proba")
        parameters = learner.get_params(deep=True)
        overrides: dict[str, Any] = {
            name: self.config.n_jobs
            for name, value in parameters.items()
            if (name == "n_jobs" or name.endswith("__n_jobs")) and (value is not None or self.config.n_jobs != 1)
        }
        if (
            type(learner) is LogisticRegression
            and "n_jobs" in overrides
            and parameters.get("solver") in {"lbfgs", "newton-cg"}
            and parameters.get("multi_class", "auto") in {"auto", "deprecated", "multinomial"}
        ):
            # sklearn's process-preferring solvers dispatch one task for binary
            # or joint multinomial fits. Exclude newton-cholesky: older releases
            # used automatic OVR. Native thread limits stay independent.
            overrides["n_jobs"] = 1
        overrides.update(
            {
                name: self.config.random_state
                for name, value in parameters.items()
                if (name == "random_state" or name.endswith("__random_state")) and value is None
            }
        )
        if task == "classification" and self._class_labels is not None:
            for name, value in parameters.items():
                if (name == "class_weight" or name.endswith("__class_weight")) and isinstance(value, dict):
                    overrides[name] = encode_class_weights(value, self._class_labels)
        learner.set_params(**overrides)
        return learner

    def _fit_selection_learner(
        self, frame: pd.DataFrame, target: np.ndarray, template: PredictiveEstimator
    ) -> Pipeline:
        if self.config.categorical_encoding == "target":
            return self._fit_learner(
                frame,
                target,
                template,
                numeric_clip_quantiles=self.config.numeric_clip_quantiles,
                categorical_encoding="target",
                random_state=self.config.random_state,
            )
        if self.config.numeric_clip_quantiles is None:
            return self._fit_learner(frame, target, template)
        return self._fit_learner(frame, target, template, numeric_clip_quantiles=self.config.numeric_clip_quantiles)

    @staticmethod
    def _fit_learner(
        frame: pd.DataFrame,
        target: np.ndarray,
        template: PredictiveEstimator,
        *,
        numeric_clip_quantiles: tuple[float, float] | None = None,
        categorical_encoding: str = "one_hot",
        random_state: int = 42,
    ) -> Pipeline:
        if categorical_encoding not in {"one_hot", "target"}:
            raise ValueError("categorical_encoding must be 'one_hot' or 'target'")
        quantiles = validate_clip_quantiles(numeric_clip_quantiles)
        numeric = [
            column
            for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column].dtype) and frame[column].notna().any()
        ]
        categorical = [column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column].dtype)]
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                    numeric,
                )
            )
        if categorical:
            if categorical_encoding == "target":
                from featcopilot.utils.model_preprocessing import CrossFittedTargetEncoder

                categorical_pipeline = Pipeline([("target", CrossFittedTargetEncoder(random_state=random_state))])
            else:
                encoder_options: dict[str, Any] = {"handle_unknown": "ignore"}
                sparse_parameter = "sparse_output" if "sparse_output" in OneHotEncoder().get_params() else "sparse"
                encoder_options[sparse_parameter] = False
                imputer_options: dict[str, Any] = {"strategy": "constant", "fill_value": "__missing__"}
                if "keep_empty_features" in SimpleImputer().get_params():
                    imputer_options["keep_empty_features"] = True
                categorical_pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(**imputer_options)),
                        ("onehot", OneHotEncoder(**encoder_options)),
                    ]
                )
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical,
                )
            )
        if not transformers:
            raise ValueError("Internal training split contains no finite numeric or categorical baseline features")
        steps: list[tuple[str, object]] = []
        if quantiles is not None:
            steps.append(("numeric_clip", NumericQuantileClipper(quantiles)))
        steps.extend(
            [
                ("preprocess", ColumnTransformer(transformers, sparse_threshold=0.0)),
                ("learner", _clone_estimator(template)),
            ]
        )
        pipeline = Pipeline(steps)
        pipeline.fit(frame, target)
        return pipeline

    def _candidate_stream(
        self, eligible: list[str], continuous: list[str], *, train: pd.DataFrame | None = None
    ) -> Iterator[Feature]:
        families: deque[Iterator[Feature]]
        if self.config.candidate_set == "extended":
            operators = (
                ("add", False),
                ("multiply", False),
                ("subtract", False),
                ("minimum", False),
                ("maximum", False),
                ("divide", False),
                ("divide", True),
            )
            families = deque()
            for index, operation in enumerate(_UNARY_EXPRESSIONS):
                if index < len(operators):
                    pair_operation, reverse = operators[index]
                    families.append(
                        self._pair_family(
                            eligible,
                            pair_operation,
                            reverse,
                            pairs=self._balanced_family_pairs(eligible, index, len(operators)),
                        )
                    )
                if index == 0 and train is not None:
                    families.append(self._categorical_pair_family(train))
                families.append(
                    self._unary_family(
                        self._balanced_family_columns(continuous, index, len(_UNARY_EXPRESSIONS)), operation
                    )
                )
        else:
            families = deque(
                [
                    self._unary_family(continuous, "square"),
                    self._pair_family(eligible, "multiply"),
                    self._pair_family(eligible, "subtract"),
                    self._pair_family(eligible, "divide"),
                    self._pair_family(eligible, "divide", reverse=True),
                    *(
                        self._unary_family(continuous, operation)
                        for operation in _UNARY_EXPRESSIONS
                        if operation != "square"
                    ),
                ]
            )
        while families:
            family = families.popleft()
            feature = next(family, None)
            if feature is not None:
                families.append(family)
                if self.config.candidate_set == "extended":
                    feature.metadata.update({"candidate_set": "extended", "operation": feature.transformation})
                yield feature

    @staticmethod
    def _balanced_pairs(columns: list[str]) -> Iterator[tuple[str, str]]:
        size = len(columns) + len(columns) % 2
        if size < 2:
            return
        # Circle-method matchings, calculated by index rather than storing pairs
        # or rotating a length-d list for every operator family.
        for rotation in range(size - 1):
            for position in range(size // 2):
                left = 0 if position == 0 else 1 + (position - 1 - rotation) % (size - 1)
                right = 1 + (size - 2 - position - rotation) % (size - 1)
                if left < len(columns) and right < len(columns):
                    yield columns[min(left, right)], columns[max(left, right)]

    def _balanced_family_pairs(self, columns: list[str], slot: int, slots: int) -> Iterator[tuple[str, str]]:
        for sweep in range(slots):
            yield from islice(self._balanced_pairs(columns), (slot - sweep) % slots, None, slots)

    @staticmethod
    def _balanced_family_columns(columns: list[str], slot: int, slots: int) -> Iterator[str]:
        for sweep in range(slots):
            for index in range((slot - sweep) % slots, len(columns), slots):
                yield columns[index]

    def _unary_family(self, columns: Iterable[str], operation: str) -> Iterator[Feature]:
        suffix = TabularEngine.TRANSFORMATIONS[operation][0]
        for column in columns:
            yield self._feature(f"{column}_{suffix}", [column], operation, _UNARY_EXPRESSIONS[operation])

    def _pair_family(
        self,
        columns: list[str],
        operation: str,
        reverse: bool = False,
        *,
        pairs: Iterator[tuple[str, str]] | None = None,
    ) -> Iterator[Feature]:
        suffix, expression = {
            "multiply": ("x", "x * z"),
            "subtract": ("minus", "x - z"),
            "divide": ("div", "np.divide(x, z, out=np.full_like(x, np.nan), where=z != 0)"),
            "add": ("plus", "x + z"),
            "minimum": ("min", "np.minimum(x, z)"),
            "maximum": ("max", "np.maximum(x, z)"),
        }[operation]
        for left, right in combinations(columns, 2) if pairs is None else pairs:
            if reverse:
                left, right = right, left
            yield self._feature(f"{left}_{suffix}_{right}", [left, right], operation, expression)

    @staticmethod
    def _is_categorical_column(column: pd.Series) -> bool:
        dtype = column.dtype
        return bool(
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        )

    @staticmethod
    def _category_sort_key(token: _CategoryToken) -> str:
        return json.dumps(token, ensure_ascii=True, allow_nan=False, separators=(",", ":"))

    def _categorical_levels(self, train: pd.DataFrame, min_count: int) -> dict[str, tuple[_CategoryToken, ...]]:
        levels: dict[str, tuple[_CategoryToken, ...]] = {}
        for column in sorted(train.columns):
            if not self._is_categorical_column(train[column]):
                continue
            counts = Counter(_category_value(value, column) for value in train[column].astype(object))
            frequent = sorted(
                ((token, count) for token, count in counts.items() if count >= min_count),
                key=lambda item: (-item[1], self._category_sort_key(item[0])),
            )[:_CATEGORY_MAX_LEVELS]
            if len(frequent) > 1:
                levels[column] = tuple(token for token, _ in frequent)
        return levels

    def _categorical_pair_family(self, train: pd.DataFrame) -> Iterator[Feature]:
        min_count = max(_CATEGORY_MIN_COUNT, int(np.ceil(_CATEGORY_MIN_FREQUENCY * len(train))))
        levels = self._categorical_levels(train, min_count)
        plans: list[tuple[str, str, list[tuple[tuple[_CategoryToken, _CategoryToken], int]]]] = []
        for left, right in islice(self._balanced_pairs(sorted(levels)), self.config.max_candidates):
            allowed_left, allowed_right = set(levels[left]), set(levels[right])
            counts: Counter[tuple[_CategoryToken, _CategoryToken]] = Counter()
            for first, second in zip(train[left].astype(object), train[right].astype(object), strict=True):
                pair = _category_value(first, left), _category_value(second, right)
                if pair[0] in allowed_left and pair[1] in allowed_right:
                    counts[pair] += 1
            frequent = sorted(
                ((pair, count) for pair, count in counts.items() if min_count <= count < len(train)),
                key=lambda item: (
                    -item[1],
                    self._category_sort_key(item[0][0]),
                    self._category_sort_key(item[0][1]),
                ),
            )[:_CATEGORY_MAX_INDICATORS_PER_PAIR]
            if frequent:
                plans.append((left, right, frequent))
                pair, count = frequent[0]
                yield self._category_feature(left, right, pair, count)
        # Cover column pairs before spending the remaining budget on their
        # additional levels. Cached definitions are bounded by the scan cap.
        for level in range(1, _CATEGORY_MAX_INDICATORS_PER_PAIR):
            for left, right, frequent in plans:
                if level < len(frequent):
                    pair, count = frequent[level]
                    yield self._category_feature(left, right, pair, count)

    @staticmethod
    def _category_mask_expression(token: _CategoryToken) -> str:
        kind, value = token
        if kind == "missing":
            return (
                "pd.api.types.is_scalar(value) and (pd.isna(value) or "
                "(isinstance(value, (float, np.floating)) and not np.isfinite(float(value))))"
            )
        if kind == "str":
            return f"isinstance(value, (str, np.str_)) and str(value) == {value!r}"
        if kind == "bool":
            return f"isinstance(value, (bool, np.bool_)) and bool(value) == {value!r}"
        if kind == "int":
            return (
                "isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) "
                f"and int(value) == {value!r}"
            )
        if kind == "float":
            return (
                "isinstance(value, (float, np.floating)) and np.isfinite(float(value)) "
                f"and float(value) == {value!r}"
            )
        raise ValueError("Unsupported categorical token kind")

    def _category_feature(
        self,
        left: str,
        right: str,
        pair: tuple[_CategoryToken, _CategoryToken],
        count: int,
    ) -> Feature:
        signature = json.dumps([left, right, pair], ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        name = f"catpair_{hashlib.sha256(signature.encode('utf-8')).hexdigest()[:24]}"
        code: list[str] = []
        valid_scalar = (
            "isinstance(value, (str, np.str_, bool, np.bool_, int, np.integer, float, np.floating)) "
            "or (pd.api.types.is_scalar(value) and pd.isna(value))"
        )
        for variable, column, token in zip(("left", "right"), (left, right), pair, strict=True):
            message = f"Column {column!r} contains unsupported categorical scalar values"
            expression = self._category_mask_expression(token)
            code.extend(
                [
                    f"{variable}_values = df[{column!r}].astype('object')",
                    f"if not all({valid_scalar} for value in {variable}_values):",
                    f"    raise pd.errors.ParserError({message!r})",
                    f"{variable} = np.fromiter(({expression} for value in {variable}_values), "
                    "dtype=bool, count=len(df))",
                ]
            )
        code.extend(
            [
                "values = (left & right).astype(float)",
                f"result = pd.Series(np.clip(np.where(np.isfinite(values), values, np.nan), "
                f"{-_NUMERIC_LIMIT!r}, {_NUMERIC_LIMIT!r}), index=df.index)",
            ]
        )
        return Feature(
            name=name,
            dtype=FeatureType.NUMERIC,
            origin=FeatureOrigin.INTERACTION,
            source_columns=[left, right],
            transformation="category_pair",
            explanation=f"Frozen indicator for a frequent training-observed pair from {left} and {right}; no target statistics",
            code="\n".join(code),
            metadata={
                "numeric_policy": _NUMERIC_POLICY,
                "category_pair_values": [list(token) for token in pair],
                "training_pair_count": count,
                "missing_policy": _CATEGORY_MISSING_POLICY,
                "unseen_policy": "zero",
                "vocabulary_scope": "internal_training_only",
            },
        )

    @staticmethod
    def _category_pair_tokens(feature: Feature) -> tuple[_CategoryToken, _CategoryToken]:
        stored: object = feature.metadata.get("category_pair_values")
        if not isinstance(stored, (list, tuple)) or len(stored) != 2:
            raise ValueError(f"Invalid categorical-pair metadata for {feature.name!r}")
        tokens: list[_CategoryToken] = []
        for item in stored:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Invalid categorical-pair metadata for {feature.name!r}")
            token = _category_value(item[1], feature.name)
            if item[0] != token[0]:
                raise ValueError(f"Invalid categorical-pair metadata for {feature.name!r}")
            tokens.append(token)
        return tokens[0], tokens[1]

    @staticmethod
    def _feature(name: str, columns: list[str], operation: str, expression: str) -> Feature:
        code = []
        for variable, column in zip(("x", "z"), columns, strict=False):
            code.extend(
                [
                    f"{variable} = df[{column!r}].to_numpy(dtype=float, na_value=np.nan)",
                    f"{variable} = np.where(np.isfinite({variable}), {variable}, np.nan)",
                ]
            )
        code.extend(
            [
                'with np.errstate(divide="ignore", invalid="ignore", over="ignore"):',
                f"    values = {expression}",
                f"result = pd.Series(np.clip(np.where(np.isfinite(values), values, np.nan), "
                f"{-_NUMERIC_LIMIT!r}, {_NUMERIC_LIMIT!r}), index=df.index)",
            ]
        )
        origin = FeatureOrigin.INTERACTION if len(columns) == 2 else FeatureOrigin.CUSTOM
        if operation in {"square", "cube"}:
            origin = FeatureOrigin.POLYNOMIAL
        return Feature(
            name=name,
            dtype=FeatureType.NUMERIC,
            origin=origin,
            source_columns=columns,
            transformation=operation,
            explanation=f"{operation} of {', '.join(columns)}; selected by internal screening and holdout loss",
            code="\n".join(code),
            metadata={"numeric_policy": _NUMERIC_POLICY},
        )

    def _check_collisions(self, candidates: list[Feature]) -> None:
        names = set(self._input_columns)
        for feature in candidates:
            if feature.name in names:
                raise ValueError(f"Generated-name collision for {feature.name!r}; rename the conflicting input columns")
            names.add(feature.name)

    def _compute(self, feature: Feature, frame: pd.DataFrame) -> np.ndarray:
        operation = feature.transformation
        if operation == _FREQUENCY_OPERATOR:
            values = apply_frequency_mapping(
                frame[feature.source_columns[0]], self._frequency_mapping(feature)
            ).to_numpy()
            return _finite_clip(values)
        if operation == "category_pair":
            tokens = self._category_pair_tokens(feature)
            masks = [
                np.fromiter(
                    (_category_value(value, column) == token for value in frame[column].astype(object)),
                    dtype=bool,
                    count=len(frame),
                )
                for column, token in zip(feature.source_columns, tokens, strict=True)
            ]
            return _finite_clip((masks[0] & masks[1]).astype(float))
        left = self._values(frame[feature.source_columns[0]])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if len(feature.source_columns) == 1:
                values = TabularEngine.TRANSFORMATIONS[operation][1](left)
            else:
                right = self._values(frame[feature.source_columns[1]])
                if operation == "multiply":
                    values = left * right
                elif operation == "subtract":
                    values = left - right
                elif operation == "divide":
                    values = np.divide(left, right, out=np.full_like(left, np.nan), where=right != 0)
                elif operation == "add":
                    values = left + right
                elif operation == "minimum":
                    values = np.minimum(left, right)
                elif operation == "maximum":
                    values = np.maximum(left, right)
                else:
                    raise ValueError(f"Unknown feature operation: {operation!r}")
        return _finite_clip(values)

    def _augment(self, prepared: pd.DataFrame, raw: pd.DataFrame, features: list[Feature]) -> pd.DataFrame:
        generated = {feature.name: self._compute(feature, raw) for feature in features}
        return pd.concat([prepared, pd.DataFrame(generated, index=raw.index)], axis=1)

    @staticmethod
    def _prediction(model: Pipeline, frame: pd.DataFrame, task: str, n_classes: int) -> np.ndarray:
        if task == "regression":
            prediction = np.asarray(model.predict(frame), dtype=float)
            if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
                raise ValueError("Regression estimator must return finite one-dimensional predictions")
            return prediction
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        classes = np.asarray(model.classes_)
        if (
            probabilities.shape != (len(frame), n_classes)
            or set(classes) != set(range(n_classes))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0)
        ):
            raise ValueError("Classifier predict_proba must return finite probabilities for every training class")
        return probabilities[:, np.argsort(classes)]

    def _loss(self, model: Pipeline, frame: pd.DataFrame, target: np.ndarray, task: str, n_classes: int) -> float:
        prediction = self._prediction(model, frame, task, n_classes)
        loss = (
            float(log_loss(target, prediction, labels=np.arange(n_classes)))
            if task == "classification"
            else float(np.sqrt(mean_squared_error(target, prediction)))
        )
        if not np.isfinite(loss):
            raise ValueError("The internal validation loss is not finite; check target scale and learner predictions")
        return loss

    @staticmethod
    def _numeric_fingerprint(values: np.ndarray) -> bytes:
        normalized = np.ascontiguousarray(values, dtype=np.float64).copy()
        normalized[normalized == 0] = 0.0
        normalized[np.isnan(normalized)] = np.nan
        return hashlib.sha256(normalized.tobytes()).digest()

    def _deduplicate_candidates(
        self, candidates: list[Feature], raw: pd.DataFrame, prepared: pd.DataFrame
    ) -> tuple[list[Feature], dict[str, np.ndarray], list[DiscardedCandidate]]:
        buckets: dict[bytes, list[tuple[str, Literal["original", "candidate"], np.ndarray]]] = {}
        for column in sorted(raw.columns):
            if pd.api.types.is_numeric_dtype(raw[column].dtype):
                values = prepared[column].to_numpy(dtype=float, na_value=np.nan, copy=False)
                buckets.setdefault(self._numeric_fingerprint(values), []).append((column, "original", values))
        retained: list[Feature] = []
        matrix: dict[str, np.ndarray] = {}
        discarded: list[DiscardedCandidate] = []
        for feature in candidates:
            values = self._compute(feature, raw)
            if values.shape != (len(raw),):
                raise ValueError(f"Candidate {feature.name!r} must produce one numeric value per training row")
            finite = values[np.isfinite(values)]
            if not len(finite) or np.all(finite == finite[0]):
                discarded.append(
                    {"name": feature.name, "reason": "constant", "duplicate_of": None, "duplicate_kind": None}
                )
                continue
            fingerprint = self._numeric_fingerprint(values)
            duplicate = next(
                (
                    (name, kind)
                    for name, kind, reference in buckets.get(fingerprint, [])
                    if np.array_equal(values, reference, equal_nan=True)
                ),
                None,
            )
            if duplicate is not None:
                discarded.append(
                    {
                        "name": feature.name,
                        "reason": "duplicate",
                        "duplicate_of": duplicate[0],
                        "duplicate_kind": duplicate[1],
                    }
                )
                continue
            retained.append(feature)
            matrix[feature.name] = values
            buckets.setdefault(fingerprint, []).append((feature.name, "candidate", values))
        return retained, matrix, discarded

    @staticmethod
    def _importance_report(ranking: _Ranking, planned_count: int) -> ImportanceRankingReport:
        return {
            "planned_count": planned_count,
            "retained_count": len(ranking.features),
            "discarded_constant_count": sum(item["reason"] == "constant" for item in ranking.discarded),
            "discarded_duplicate_count": sum(item["reason"] == "duplicate" for item in ranking.discarded),
            "discarded_candidates": deepcopy(ranking.discarded),
            "importance_source": ranking.importance_source,
            "fit_rows": ranking.fit_rows,
            "learner_fits": ranking.learner_fits,
        }

    @staticmethod
    def _importance_values(model: Pipeline) -> tuple[np.ndarray, str]:
        learner = model.named_steps["learner"]
        native: object = getattr(learner, "feature_importances_", None)
        source = "feature_importances_" if native is not None else "normalized_abs_coef"
        supplied: object = native if native is not None else getattr(learner, "coef_", None)
        if supplied is None:
            raise ValueError(
                "screening='importance' requires the fitted intended learner to expose feature_importances_ or coef_"
            )
        try:
            values = np.asarray(supplied)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"screening='importance' received an invalid {source} array") from exc
        if values.dtype.kind not in "biuf":
            raise ValueError(f"screening='importance' requires real numeric {source} values")
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"screening='importance' requires finite {source} values")
        if native is not None:
            if values.ndim != 1 or (values < 0).any():
                raise ValueError("feature_importances_ must be a nonnegative one-dimensional encoded-feature vector")
            return values, source
        if values.ndim not in (1, 2) or not values.size:
            raise ValueError("coef_ must be a nonempty feature vector or class-by-feature matrix")
        absolute = np.abs(values)
        maximum = float(absolute.max())
        if maximum:
            absolute = absolute / maximum
        weights = absolute.mean(axis=0) if absolute.ndim == 2 else absolute
        total = float(weights.sum())
        if total:
            weights = weights / total
        return np.asarray(weights), source

    @staticmethod
    def _candidate_output_indices(model: Pipeline, features: list[Feature], importance_width: int) -> dict[str, int]:
        preprocessing = model.named_steps["preprocess"]
        output_indices: object = getattr(preprocessing, "output_indices_", None)
        if not isinstance(output_indices, dict):
            raise ValueError("Cannot attribute importances: fitted preprocessing lacks encoded output positions")
        spans: dict[str, tuple[int, int]] = {}
        for name, span in output_indices.items():
            if (
                not isinstance(name, str)
                or not isinstance(span, slice)
                or not isinstance(span.start, (int, np.integer))
                or not isinstance(span.stop, (int, np.integer))
                or span.step not in (None, 1)
                or span.start < 0
                or span.stop < span.start
            ):
                raise ValueError("Cannot attribute importances: invalid encoded preprocessing positions")
            spans[name] = (int(span.start), int(span.stop))
        width = max((stop for _, stop in spans.values()), default=0)
        if importance_width != width:
            raise ValueError(
                f"Importance width {importance_width} does not match the fitted encoded feature width {width}"
            )
        if not features:
            return {}
        numeric = next((entry for entry in preprocessing.transformers_ if entry[0] == "numeric"), None)
        if numeric is None or "numeric" not in spans:
            raise ValueError("Cannot attribute candidate importances: no fitted numeric preprocessing block")
        columns = numeric[2]
        start, stop = spans["numeric"]
        if (
            not isinstance(columns, (list, tuple))
            or not all(isinstance(column, str) for column in columns)
            or len(set(columns)) != len(columns)
            or stop - start != len(columns)
        ):
            raise ValueError("Cannot attribute importances: numeric preprocessing changed the candidate column width")
        positions = {column: start + offset for offset, column in enumerate(columns)}
        if any(feature.name not in positions for feature in features):
            raise ValueError(
                "Cannot attribute importances: a generated feature is missing from the fitted numeric block"
            )
        return {feature.name: positions[feature.name] for feature in features}

    def _rank_importance(
        self,
        candidates: list[Feature],
        raw: pd.DataFrame,
        prepared: pd.DataFrame,
        target: np.ndarray,
        template: PredictiveEstimator,
        *,
        task: str | None = None,
        seed: int | None = None,
    ) -> _Ranking:
        if self.config.importance_resamples == 1:
            return self._rank_importance_single(candidates, raw, prepared, target, template)
        retained, scope_values, discarded = self._deduplicate_candidates(candidates, raw, prepared)
        del scope_values
        ranking = self._rank_importance_resampled(
            retained,
            raw,
            prepared,
            target,
            template,
            task=resolve_task(target, self.config.task) if task is None else task,
            seed=self.config.random_state if seed is None else seed,
        )
        ranking.discarded = discarded + ranking.discarded
        return ranking

    def _rank_importance_single(
        self,
        candidates: list[Feature],
        raw: pd.DataFrame,
        prepared: pd.DataFrame,
        target: np.ndarray,
        template: PredictiveEstimator,
        *,
        fit_empty: bool = False,
    ) -> _Ranking:
        retained, values, discarded = self._deduplicate_candidates(candidates, raw, prepared)
        if not retained and not fit_empty:
            return _Ranking([], {}, unscored_rows=len(raw), discarded=discarded)
        importance_template = template
        uses_gain = any(cls.__module__.startswith("lightgbm.") for cls in type(template).__mro__) and (
            "importance_type" in template.get_params(deep=False)
        )
        if uses_gain:
            importance_template = _clone_estimator(template)
            importance_template.set_params(importance_type="gain")
        augmented = pd.concat([prepared, pd.DataFrame(values, index=raw.index)], axis=1)
        model = self._fit_selection_learner(augmented, target, importance_template)
        weights, source = self._importance_values(model)
        positions = self._candidate_output_indices(model, retained, len(weights))
        scores = {feature.name: float(weights[positions[feature.name]]) for feature in retained}
        ranked = sorted(retained, key=lambda feature: (-scores[feature.name], feature.name))
        for feature in ranked:
            feature.importance = scores[feature.name]
        return _Ranking(
            ranked,
            {feature.name: scores[feature.name] for feature in ranked},
            unscored_rows=len(raw),
            learner_fits=1,
            discarded=discarded,
            importance_source=f"{source}:gain" if uses_gain and source == "feature_importances_" else source,
            fit_rows=len(raw),
        )

    @staticmethod
    def _candidate_average_ranks(scores: np.ndarray) -> np.ndarray:
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("Importance resampling requires finite one-dimensional candidate scores")
        usable = np.where(scores > 0, scores, 0.0)
        return np.asarray(pd.Series(usable).rank(method="average", ascending=False).to_numpy(dtype=float))

    @staticmethod
    def _resampling_report(scope_rows: int, universe_count: int) -> ImportanceResamplingReport:
        return {
            "requested_bags": 5,
            "completed_bags": 0,
            "sample_fraction": _IMPORTANCE_BAG_FRACTION,
            "sampling": "iid_without_replacement",
            "aggregation": "mean_candidate_rank",
            "rank_ties": "average",
            "nonpositive_policy": "worst_tie",
            "eligibility": "positive_importance_in_any_bag",
            "score_kind": "negative_mean_candidate_rank",
            "scope_rows": scope_rows,
            "universe_count": universe_count,
            "duplicate_bag_mask_count": 0,
            "bags": [],
            "eligible_count": 0,
            "never_positive_count": 0,
            "positive_counts": {},
            "mean_ranks": {},
            "never_positive_features": [],
        }

    def _rank_importance_resampled(
        self,
        candidates: list[Feature],
        raw: pd.DataFrame,
        prepared: pd.DataFrame,
        target: np.ndarray,
        template: PredictiveEstimator,
        *,
        task: str,
        seed: int,
    ) -> _Ranking:
        report = self._resampling_report(len(raw), len(candidates))
        if not candidates:
            return _Ranking([], {}, unscored_rows=len(raw), resampling=report)
        names = sorted(feature.name for feature in candidates)
        if len(set(names)) != len(names):
            raise ValueError("Importance resampling requires unique candidate identities")
        positions = {name: index for index, name in enumerate(names)}
        splits = make_holdout_splits(
            target,
            task,
            self.config.importance_resamples,
            seed,
            test_size=0.15,
            allow_constant_target=task == "regression",
        )
        if len(splits) != self.config.importance_resamples:
            raise ValueError("Importance resampling could not produce the requested number of IID bags")
        rank_sum = np.zeros(len(names), dtype=float)
        positive_counts = np.zeros(len(names), dtype=int)
        seen_masks: set[bytes] = set()
        learner_fits = 0
        fit_rows = 0
        for bag, (train, excluded) in enumerate(splits):
            bag_raw = raw.iloc[train]
            refitted, frequency_count = self._refit_frequency_features(candidates, bag_raw, scope="internal_train")
            bag_candidates = [
                feature if feature.transformation == _FREQUENCY_OPERATOR else deepcopy(feature) for feature in refitted
            ]
            ranked = self._rank_importance_single(
                bag_candidates, bag_raw, prepared.iloc[train], target[train], template, fit_empty=True
            )
            retained_names = {feature.name for feature in ranked.features}
            discarded_names = {item["name"] for item in ranked.discarded}
            if (
                retained_names != set(ranked.scores)
                or retained_names & discarded_names
                or retained_names | discarded_names != set(names)
            ):
                raise RuntimeError("Importance bag results do not cover the fixed candidate-definition universe")
            scores = np.zeros(len(names), dtype=float)
            for name, score in ranked.scores.items():
                scores[positions[name]] = score
            rank_sum += self._candidate_average_ranks(scores)
            positive_counts += scores > 0
            membership = np.zeros(len(raw), dtype=bool)
            membership[train] = True
            mask = np.asarray([len(raw)], dtype="<i8").tobytes() + np.packbits(membership).tobytes()
            report["duplicate_bag_mask_count"] += int(mask in seen_masks)
            seen_masks.add(mask)
            report["bags"].append(
                {
                    "bag": bag,
                    "seed": (seed + bag) % (2**32),
                    "train_rows": len(train),
                    "excluded_rows": len(excluded),
                    "position_mask_sha256": hashlib.sha256(mask).hexdigest(),
                    "retained_count": len(ranked.features),
                    "constant_count": sum(item["reason"] == "constant" for item in ranked.discarded),
                    "duplicate_count": sum(item["reason"] == "duplicate" for item in ranked.discarded),
                    "positive_count": int(np.count_nonzero(scores > 0)),
                    "frequency_refit_count": frequency_count,
                    "learner_fits": ranked.learner_fits,
                }
            )
            learner_fits += ranked.learner_fits
            fit_rows += ranked.fit_rows
        mean_ranks = rank_sum / self.config.importance_resamples
        report["completed_bags"] = len(splits)
        report["positive_counts"] = {name: int(positive_counts[index]) for name, index in positions.items()}
        report["mean_ranks"] = {name: float(mean_ranks[index]) for name, index in positions.items()}
        report["never_positive_features"] = [name for name in names if report["positive_counts"][name] == 0]
        retained = [feature for feature in candidates if report["positive_counts"][feature.name] > 0]
        report["eligible_count"] = len(retained)
        report["never_positive_count"] = len(report["never_positive_features"])
        retained.sort(key=lambda feature: (report["mean_ranks"][feature.name], feature.name))
        scores_by_name = {feature.name: -report["mean_ranks"][feature.name] for feature in retained}
        for feature in retained:
            feature.importance = scores_by_name[feature.name]
        discarded: list[DiscardedCandidate] = [
            {"name": name, "reason": "never_positive", "duplicate_of": None, "duplicate_kind": None}
            for name in report["never_positive_features"]
        ]
        return _Ranking(
            retained,
            scores_by_name,
            unscored_rows=len(raw),
            learner_fits=learner_fits,
            discarded=discarded,
            importance_source="negative_mean_candidate_rank",
            fit_rows=fit_rows,
            resampling=report,
        )

    def _rank_candidates(
        self,
        candidates: list[Feature],
        raw: pd.DataFrame,
        prepared: pd.DataFrame,
        target: np.ndarray,
        template: PredictiveEstimator,
        task: str,
        n_classes: int,
        *,
        boundaries: SplitMetadata | None = None,
        seed: int | None = None,
        screening: str | None = None,
    ) -> _Ranking:
        if not candidates:
            ranking = _Ranking([], {}, unscored_rows=len(raw))
            if self.config.importance_resamples > 1:
                ranking.resampling = self._resampling_report(len(raw), 0)
            return ranking
        boundaries = boundaries or SplitMetadata()
        seed = self.config.random_state if seed is None else seed
        screening = screening or self._resolve_screening(template)
        if screening == "importance":
            if self.config.importance_resamples > 1:
                if boundaries.kind != "iid":
                    raise ValueError("importance_resamples=5 supports IID only")
                return self._rank_importance(candidates, raw, prepared, target, template, task=task, seed=seed)
            return self._rank_importance(candidates, raw, prepared, target, template)
        if boundaries.values is not None:
            count = len(np.unique(boundaries.values))
            if boundaries.kind == "grouped" and count < 4:
                raise ValueError(
                    "Grouped residual screening needs at least four training groups; "
                    "provide at least six groups before the holdout"
                )
            if boundaries.kind == "temporal" and count < 6:
                raise ValueError(
                    "Temporal residual screening needs at least six training time blocks; "
                    "provide at least twelve time blocks for repeated holdouts"
                )
        folds, learner_fits = self._residual_folds(prepared, target, template, task, n_classes, boundaries, seed)
        reference_error = sum(
            float(np.square(fold.validation_residuals - fold.train_residuals.mean(axis=0)).sum()) for fold in folds
        )
        if not np.isfinite(reference_error):
            raise ValueError("Residual screening loss is not finite; rescale the target or check the estimator")
        scores: dict[str, float] = {}
        for feature in candidates:
            values = self._compute(feature, raw)
            if pd.Series(values).nunique() < 2:
                continue
            error = 0.0
            for fold in folds:
                prediction = self._screen_prediction(
                    values[fold.train], values[fold.validation], fold.train_residuals, screening, seed
                )
                error += float(np.square(fold.validation_residuals - prediction).sum())
            if not np.isfinite(error):
                raise ValueError(f"Residual screening loss is not finite for candidate {feature.name!r}")
            scores[feature.name] = float((reference_error - error) / max(reference_error, np.finfo(float).eps))
        ranked = sorted(
            (feature for feature in candidates if feature.name in scores),
            key=lambda feature: (-scores[feature.name], feature.name),
        )
        for feature in ranked:
            feature.importance = scores[feature.name]
        scored_rows = sum(len(fold.validation) for fold in folds)
        return _Ranking(
            ranked,
            {feature.name: scores[feature.name] for feature in ranked},
            len(folds),
            scored_rows,
            len(raw) - scored_rows,
            learner_fits,
        )

    def _residual_folds(
        self,
        prepared: pd.DataFrame,
        target: np.ndarray,
        template: PredictiveEstimator,
        task: str,
        n_classes: int,
        boundaries: SplitMetadata,
        seed: int,
    ) -> tuple[list[_ResidualFold], int]:
        splits = self._cv_splits(target, task, boundaries, n_splits=3, seed=seed)
        observed = (
            np.eye(n_classes)[target.astype(int)] if task == "classification" else target.astype(float).reshape(-1, 1)
        )
        folds: list[_ResidualFold] = []
        learner_fits = 0
        for index, (train, validation) in enumerate(splits):
            residuals = np.full((len(train), observed.shape[1]), np.nan)
            scored = np.zeros(len(train), dtype=bool)
            validation_prediction = np.zeros((len(validation), observed.shape[1]))
            inner = self._cv_splits(
                target[train],
                task,
                boundaries.subset(train),
                n_splits=2,
                seed=(seed + index + 1) % (2**32),
            )
            # These baseline fits are entirely inside the screening-training
            # partition; none can depend on screening-validation labels.
            for inner_train, inner_validation in inner:
                fit_rows = train[inner_train]
                score_rows = train[inner_validation]
                baseline = self._fit_selection_learner(prepared.iloc[fit_rows], target[fit_rows], template)
                learner_fits += 1
                prediction = self._prediction(baseline, prepared.iloc[score_rows], task, n_classes)
                residuals[inner_validation] = observed[score_rows] - prediction.reshape(len(score_rows), -1)
                scored[inner_validation] = True
                prediction = self._prediction(baseline, prepared.iloc[validation], task, n_classes)
                validation_prediction += prediction.reshape(len(validation), -1) / len(inner)
            if scored.sum() < 2:
                raise ValueError(
                    "Nested residual screening needs at least two scored training rows after the unscored "
                    "chronological prefix; provide more time blocks or samples"
                )
            # Reuse the inner learners instead of a larger full-training learner:
            # otherwise training and validation residuals differ in model capacity.
            validation_residuals = observed[validation] - validation_prediction
            folds.append(_ResidualFold(train[scored], validation, residuals[scored], validation_residuals))
        return folds, learner_fits

    @staticmethod
    def _screen_prediction(
        train_values: np.ndarray,
        validation_values: np.ndarray,
        residuals: np.ndarray,
        screening: str,
        seed: int,
    ) -> np.ndarray:
        finite = train_values[np.isfinite(train_values)]
        median = float(np.median(finite)) if len(finite) else 0.0
        train = np.where(np.isfinite(train_values), train_values, median)
        validation = np.where(np.isfinite(validation_values), validation_values, median)
        if screening == "linear":
            center = float(train.mean())
            scale = float(train.std()) or 1.0
            normalized = (train - center) / scale
            intercept = residuals.mean(axis=0)
            # A fixed unit ridge penalty on train-standardized values prevents
            # fitting a perfect slope to tiny/near-constant screening samples.
            coefficient = normalized @ (residuals - intercept) / (float(normalized @ normalized) + 1.0)
            return np.asarray(intercept + ((validation - center) / scale)[:, None] * coefficient)
        if screening != "tree":
            raise ValueError(f"Unknown residual screening method: {screening!r}")
        tree = DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=max(2, min(20, len(train) // 20)),
            random_state=seed,
        )
        tree.fit(train.reshape(-1, 1), residuals)
        return np.asarray(tree.predict(validation.reshape(-1, 1))).reshape(len(validation), -1)

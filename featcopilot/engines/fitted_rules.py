"""Training-only statistics and leakage-safe target encodings for typed rules."""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from sklearn.preprocessing import LabelEncoder

from featcopilot.core.base import BaseEngine
from featcopilot.core.fitted_rule import FittedRulePlan, FittedRuleSpec
from featcopilot.utils.frequency import apply_frequency_mapping, fit_frequency_mapping
from featcopilot.utils.frequency import category_key as _category_key
from featcopilot.utils.frequency import category_keys as _keys
from featcopilot.utils.logger import get_logger
from featcopilot.utils.numeric_clipping import finite_quantile_bounds
from featcopilot.utils.splitting import make_cv_splits
from featcopilot.utils.task import resolve_task, validate_task

logger = get_logger(__name__)


class _RuleState(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rule: FittedRuleSpec
    mean: float = 0.0
    scale: float = Field(default=1.0, gt=0)
    lower: float = 0.0
    upper: float = 0.0
    mapping: dict[str, float] = Field(default_factory=dict)
    categories: list[str] = Field(default_factory=list)


class _CrossFitReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split_kind: Literal["iid", "grouped", "temporal"]
    n_splits: int = Field(ge=1)
    n_training_rows: int = Field(ge=2)
    prior_row_positions: list[int] = Field(default_factory=list)
    prior_policy: Literal["fixed_zero"] = "fixed_zero"
    label_end_times_used: bool = False


class _EngineState(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    version: Literal[1] = 1
    input_columns: list[str]
    rules: list[_RuleState]
    task: Literal["auto", "classification", "regression"]
    random_state: int
    target_cv: int = Field(ge=2)
    resolved_task: Literal["classification", "regression"] | None = None
    target_classes: list[JsonValue] = Field(default_factory=list)
    cross_fit_report: _CrossFitReport | None = None


def _numeric(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_complex_dtype(series.dtype):
        raise ValueError(f"Fitted numerical rule requires a real numeric source column, got {series.name!r}")
    if not pd.api.types.is_numeric_dtype(series.dtype) and series.notna().any():
        raise ValueError(f"Fitted numerical rule requires a numeric source column, got {series.name!r}")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    return np.where(np.isfinite(values), values, np.nan)


def _rowwise_values(rule: FittedRuleSpec, X: pd.DataFrame) -> np.ndarray:
    if any(pd.api.types.is_complex_dtype(X[column].dtype) for column in rule.columns):
        raise ValueError(f"Rowwise rule {rule.name!r} requires real numeric source columns")
    operands = [_numeric(X[column]) for column in rule.columns]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if rule.operator == "log1p":
            values = np.log1p(np.abs(operands[0]))
        elif rule.operator == "multiply":
            values = operands[0] * operands[1]
        else:
            values = np.divide(operands[0], operands[1], out=np.full_like(operands[0], np.nan), where=operands[1] != 0)
    return np.asarray(np.nan_to_num(values, nan=np.nan), dtype=float)


def _mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    magnitude = float(np.max(np.abs(values)))
    return float(np.mean(values / magnitude) * magnitude) if magnitude else 0.0


def _mean_mapping(keys: list[str | None], values: np.ndarray, smoothing: float = 0.0) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for key, value in zip(keys, values, strict=True):
        if key is not None and np.isfinite(value):
            grouped.setdefault(key, []).append(float(value))
    global_mean = _mean(values)
    result = {}
    for key, group in grouped.items():
        weight = len(group) / (len(group) + smoothing)
        result[key] = _mean(np.asarray(group)) * weight + global_mean * (1 - weight)
    return result


class FittedRuleEngine(BaseEngine):
    """
    Fit a bounded collection of statistical rules and freeze inference state.

    Parameters
    ----------
    rules : list[FittedRuleSpec or dict]
        Up to 100 validated rules, with distinct output names.
    task : {'auto', 'classification', 'regression'}, default='auto'
        Target interpretation. Target means support regression and binary
        classification only; auto uses sklearn's target-type inference.
        Explicit regression is required for integer regression targets.
    random_state : int, default=42
        Seed for shuffled, label-independent target cross-fitting.
    target_cv : int, default=5
        Cross-fitting folds, reduced to the training row count if necessary.

    Notes
    -----
    fit_transform returns out-of-fold target means; transform always uses
    full-training maps, even for the original training rows. Unknown/missing
    keys use a fold-local prior during cross-fitting. Splits do not depend on
    labels, and y is aligned positionally, never by pandas index.

    Original columns, order and index are preserved. Transform requires the
    exact fitted column names and order. Group folds keep entities disjoint;
    temporal folds use strictly earlier timestamp blocks, optionally purging
    labels unavailable at the validation cutoff. Combined group/time splitting
    is unsupported. The initial temporal prefix excluded from validation gets
    a fixed-zero no-history prior, never a full-training target mean. Its row
    positions are logged in get_cross_fit_report and serialized with the state.
    Inference maps still use all supplied training targets; callers must supply
    only labels available at their final fitting cutoff.
    """

    def __init__(
        self,
        rules: list[FittedRuleSpec | dict[str, JsonValue]],
        task: Literal["auto", "classification", "regression"] = "auto",
        random_state: int = 42,
        target_cv: int = 5,
    ) -> None:
        super().__init__()
        validate_task(task)
        if isinstance(target_cv, bool) or not isinstance(target_cv, int) or target_cv < 2:
            raise ValueError("target_cv must be an integer >= 2")
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise ValueError("random_state must be an integer")
        plan = FittedRulePlan.model_validate({"rules": rules})
        self.rules = [rule.model_copy(deep=True) for rule in plan.rules]
        self.task = task
        self.random_state = random_state
        self.target_cv = target_cv
        self._input_columns: list[str] = []
        self._states: list[_RuleState] = []
        self._training_target_features: dict[str, np.ndarray] = {}
        self._resolved_task: Literal["classification", "regression"] | None = None
        self._target_classes: list[JsonValue] = []
        self._cross_fit_report: _CrossFitReport | None = None

    def _validate_schema(self, X: pd.DataFrame, *, fitting: bool = False) -> None:
        if not X.columns.is_unique or any(not isinstance(column, str) for column in X.columns):
            raise ValueError("FittedRuleEngine requires unique string input column names")
        if not fitting and list(X.columns) != self._input_columns:
            raise ValueError(f"Input schema must match fitted columns in order: {self._input_columns}")
        rules = self.rules if fitting else [state.rule for state in self._states]
        names = [rule.name for rule in rules]
        if len(names) != len(set(names)) or set(names).intersection(X.columns):
            raise ValueError("Rule output names must be unique and must not collide with input columns")
        for rule in rules:
            missing = set(rule.columns).difference(X.columns)
            if missing:
                raise ValueError(f"Rule {rule.name!r} has missing source columns: {sorted(missing)}")

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        *,
        groups: pd.Series | np.ndarray | None = None,
        timestamps: pd.Series | np.ndarray | None = None,
        label_end_times: pd.Series | np.ndarray | None = None,
        **kwargs: object,
    ) -> "FittedRuleEngine":
        """
        Learn training statistics and prepare out-of-fold target means.

        Parameters
        ----------
        X : DataFrame or ndarray
            Non-empty training data with unique string column names.
        y : Series or ndarray, optional
            Positionally aligned target, required only for target_mean.
        groups : Series or ndarray, optional
            Positionally aligned entity labels for group-disjoint target means.
        timestamps : Series or ndarray, optional
            Positionally aligned timestamps for chronological target means,
            mutually exclusive with groups. Equal timestamps stay together.
        label_end_times : Series or ndarray, optional
            Label availability times, requiring timestamps. Fold training maps
            include only labels available strictly before the validation start.
            Split metadata is ignored when no target_mean rule is present.
        **kwargs : dict
            No additional fit options are supported.

        Returns
        -------
        FittedRuleEngine
            The fitted engine. Call fit_transform to obtain leakage-safe
            training target encodings; transform uses full-training maps.
        """
        self._is_fitted = False
        self._states = []
        self._training_target_features = {}
        self._feature_names = []
        self._feature_metadata = {}
        self._input_columns = []
        self._resolved_task = None
        self._target_classes = []
        self._cross_fit_report = None
        X = self._validate_input(X)
        self._validate_schema(X, fitting=True)
        if not len(X):
            raise ValueError("FittedRuleEngine requires non-empty training data")
        supervised = any(rule.operator == "target_mean" for rule in self.rules)
        if kwargs:
            raise TypeError(f"Unsupported fitted-rule fit options: {sorted(kwargs)}")
        target = self._prepare_target(y, len(X)) if supervised else None
        splits = self._target_splits(target, groups, timestamps, label_end_times) if target is not None else []
        self._input_columns = list(X.columns)
        for rule in self.rules:
            state = self._fit_rule(rule, X, target)
            self._states.append(state)
            if rule.operator == "target_mean":
                assert target is not None
                self._training_target_features[rule.name] = self._cross_fit(rule, X, target, splits)
        self._feature_names = [state.rule.name for state in self._states]
        self._feature_metadata = {state.rule.name: state.rule.model_dump(mode="json") for state in self._states}
        self._is_fitted = True
        return self

    def _prepare_target(self, y: pd.Series | np.ndarray | None, n_rows: int) -> np.ndarray:
        if y is None:
            raise ValueError("target_mean requires a target y")
        values = np.asarray(y)
        if values.ndim != 1 or len(values) != n_rows:
            raise ValueError("Target y must be one-dimensional and match X positionally")
        if n_rows < 2:
            raise ValueError("target_mean requires at least two rows for own-target exclusion")
        self._resolved_task = resolve_task(values, self.task)
        if self._resolved_task == "regression":
            return np.asarray(values, dtype=float)
        values = pd.Series(values).infer_objects().to_numpy()
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(values)
        if len(encoder.classes_) > 2:
            raise ValueError(
                "target_mean supports binary classification only, not ordinal averages of multiclass labels"
            )
        if values.dtype.kind in "biuf" and np.isin(values, [0, 1]).all():
            self._target_classes = [False, True] if values.dtype.kind == "b" else [0, 1]
            if len(encoder.classes_) == 1:
                logger.warning("FittedRuleEngine: one-class binary target; preserving the fixed 0/1 class encoding")
            return np.asarray(values, dtype=float)
        self._target_classes = encoder.classes_.tolist()
        if len(encoder.classes_) == 1:
            logger.warning("FittedRuleEngine: one-class target; binary target means are zero")
        return np.asarray(encoded, dtype=float)

    def _fit_rule(self, rule: FittedRuleSpec, X: pd.DataFrame, target: np.ndarray | None) -> _RuleState:
        state = _RuleState(rule=rule.model_copy(deep=True))
        source = X[rule.columns[0]]
        if rule.operator in ("multiply", "divide", "log1p"):
            _rowwise_values(rule, X)
            return state
        if rule.operator in ("standardize", "clip"):
            values = _numeric(source)
            finite = values[np.isfinite(values)]
            if not len(finite):
                logger.warning(f"FittedRuleEngine: {rule.name!r} has no finite training values; using zero statistics")
                return state
            state.mean = _mean(finite)
            if rule.operator == "standardize":
                magnitude = float(np.max(np.abs(finite))) or 1.0
                scale = float(np.std(finite / magnitude) * magnitude)
                if scale == 0:
                    logger.warning(f"FittedRuleEngine: {rule.name!r} is constant; using unit scale")
                state.scale = scale or 1.0
            else:
                lower = rule.parameters["lower_quantile"]
                upper = rule.parameters["upper_quantile"]
                assert isinstance(lower, float) and isinstance(upper, float)
                bounds = finite_quantile_bounds(finite, (lower, upper))
                if bounds is None:
                    raise RuntimeError("Finite fitted-rule clip input unexpectedly produced no quantile bounds")
                state.lower, state.upper = bounds
            return state
        if rule.operator == "frequency":
            state.mapping = fit_frequency_mapping(source)
            return state
        keys = _keys(source)
        if rule.operator == "rare_category":
            if any(not isinstance(value, str) for value in source if _category_key(value) is not None):
                raise ValueError("rare_category requires string categories")
            other_label = rule.parameters["other_label"]
            min_frequency = rule.parameters["min_frequency"]
            assert isinstance(other_label, str) and isinstance(min_frequency, float)
            if other_label in source.dropna().tolist():
                raise ValueError(f"rare_category other_label {other_label!r} collides with a training category")
            counts = source.value_counts(dropna=True)
            state.categories = [
                str(value) for value, count in counts.items() if count > 0 and count / len(X) >= min_frequency
            ]
        else:
            if rule.operator == "target_mean":
                assert target is not None
                values = target
            else:
                values = _numeric(X[rule.columns[1]])
            smoothing = rule.parameters.get("smoothing", 0.0)
            assert isinstance(smoothing, float)
            if not np.isfinite(values).any():
                logger.warning(f"FittedRuleEngine: {rule.name!r} has no finite training values; using global mean zero")
            state.mean = _mean(values)
            state.mapping = _mean_mapping(keys, values, smoothing)
        return state

    def _target_splits(
        self,
        target: np.ndarray,
        groups: pd.Series | np.ndarray | None,
        timestamps: pd.Series | np.ndarray | None,
        label_end_times: pd.Series | np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # Regression routing keeps even binary-target folds independent of the labels.
        splits = make_cv_splits(
            target,
            "regression",
            min(self.target_cv, len(target)),
            self.random_state,
            groups=groups,
            timestamps=timestamps,
            label_end_times=label_end_times,
            allow_constant_target=True,
        )
        if len(splits) != self.target_cv:
            logger.warning(f"FittedRuleEngine: reducing target_cv to {len(splits)} for the available split boundaries")
        covered = np.zeros(len(target), dtype=bool)
        for _, valid in splits:
            covered[valid] = True
        prior_positions = [int(position) for position in np.flatnonzero(~covered)]
        if prior_positions and timestamps is None:
            raise RuntimeError("Non-temporal target cross-fitting must cover every training row")
        self._cross_fit_report = _CrossFitReport(
            split_kind="temporal" if timestamps is not None else "grouped" if groups is not None else "iid",
            n_splits=len(splits),
            n_training_rows=len(target),
            prior_row_positions=prior_positions,
            label_end_times_used=label_end_times is not None,
        )
        if prior_positions:
            logger.warning(
                f"FittedRuleEngine: temporal target_mean uses a fixed-zero no-history prior for "
                f"{len(prior_positions)} initial-prefix rows excluded from cross-fitting; "
                "see get_cross_fit_report() for their positional indices"
            )
        return splits

    def _cross_fit(
        self, rule: FittedRuleSpec, X: pd.DataFrame, target: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        result = np.zeros(len(X), dtype=float)
        for train, valid in splits:
            state = self._fit_rule(rule, X.iloc[train], target[train])
            result[valid] = self._apply_rule(state, X.iloc[valid]).to_numpy()
        return result

    def _apply_rule(self, state: _RuleState, X: pd.DataFrame) -> pd.Series:
        rule = state.rule
        source = X[rule.columns[0]]
        if rule.operator in ("multiply", "divide", "log1p"):
            return pd.Series(_rowwise_values(rule, X), index=X.index, name=rule.name)
        if rule.operator in ("standardize", "clip"):
            values = _numeric(source)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if rule.operator == "standardize":
                    delta = values - state.mean
                    values = np.where(
                        np.isfinite(delta), delta / state.scale, values / state.scale - state.mean / state.scale
                    )
                else:
                    values = np.clip(values, state.lower, state.upper)
            values = np.nan_to_num(values, nan=0.0)
            return pd.Series(values, index=X.index, name=rule.name)
        if rule.operator == "rare_category":
            other_label = rule.parameters["other_label"]
            known = set(state.categories)
            values_out = [value if isinstance(value, str) and value in known else other_label for value in source]
            return pd.Series(values_out, index=X.index, name=rule.name, dtype=object)
        if rule.operator == "frequency":
            return apply_frequency_mapping(source, state.mapping).rename(rule.name)
        prior = state.mean
        return pd.Series(
            [state.mapping.get(key, prior) if key is not None else prior for key in _keys(source)],
            index=X.index,
            name=rule.name,
            dtype=float,
        )

    def transform(self, X: pd.DataFrame | np.ndarray, **kwargs: object) -> pd.DataFrame:
        """
        Apply frozen statistics without learning from the transform batch.

        Parameters
        ----------
        X : DataFrame or ndarray
            Input with the exact fitted column names and order.
        **kwargs : dict
            No additional transform options are supported.

        Returns
        -------
        DataFrame
            Original columns followed by the fitted features, retaining index.
        """
        if not self._is_fitted:
            raise RuntimeError("FittedRuleEngine must be fitted before transform")
        if kwargs:
            raise TypeError(f"Unsupported fitted-rule transform options: {sorted(kwargs)}")
        X = self._validate_input(X)
        self._validate_schema(X)
        result = X.copy()
        for state in self._states:
            result[state.rule.name] = self._apply_rule(state, X).to_numpy()
        return result

    def _transform_fit_data(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None, **kwargs: object
    ) -> pd.DataFrame:
        result = self.transform(X)
        for name, values in self._training_target_features.items():
            if len(values) != len(result):
                raise ValueError("Training target encodings must be used with the fitted training rows")
            result[name] = values.copy()
        return result

    def to_dict(self) -> dict[str, JsonValue]:
        """
        Export validated JSON-compatible inference state.

        Returns
        -------
        dict
            Rules, input schema, learned statistics, target-class metadata and
            cross-fitting diagnostics, including fixed-zero prior positions.
            Raw training rows and out-of-fold target values are excluded.
        """
        if not self._is_fitted:
            raise RuntimeError("FittedRuleEngine must be fitted before exporting state")
        state = _EngineState(
            input_columns=self._input_columns,
            rules=self._states,
            task=self.task,
            random_state=self.random_state,
            target_cv=self.target_cv,
            resolved_task=self._resolved_task,
            target_classes=self._target_classes,
            cross_fit_report=self._cross_fit_report,
        )
        return dict(state.model_dump(mode="json"))

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "FittedRuleEngine":
        """
        Restore fitted inference state without training data or a provider.

        Parameters
        ----------
        data : dict
            State returned by to_dict, optionally round-tripped through JSON.

        Returns
        -------
        FittedRuleEngine
            Fitted transformer with the same feature schema and statistics.
        """
        state = _EngineState.model_validate(data)
        engine = cls(
            rules=[rule_state.rule for rule_state in state.rules],
            task=state.task,
            random_state=state.random_state,
            target_cv=state.target_cv,
        )
        engine._validate_schema(pd.DataFrame(columns=state.input_columns), fitting=True)
        if any(rule.lower > rule.upper for rule in state.rules):
            raise ValueError("Serialized clipping lower bounds must not exceed upper bounds")
        engine._input_columns = state.input_columns.copy()
        engine._states = [rule.model_copy(deep=True) for rule in state.rules]
        engine._feature_names = [rule.rule.name for rule in state.rules]
        engine._feature_metadata = {rule.rule.name: rule.rule.model_dump(mode="json") for rule in state.rules}
        engine._resolved_task = state.resolved_task
        engine._target_classes = state.target_classes.copy()
        engine._cross_fit_report = state.cross_fit_report.model_copy(deep=True) if state.cross_fit_report else None
        engine._is_fitted = True
        return engine

    def get_cross_fit_report(self) -> dict[str, JsonValue]:
        """
        Get target cross-fitting diagnostics without raw target values.

        Returns
        -------
        dict
            Split kind/count, training row count, positional indices assigned
            the fixed-zero prior, and whether label availability was supplied.
            Empty when no target_mean was fitted or loading an older export
            without diagnostics. Inference never consumes this report.

        Raises
        ------
        RuntimeError
            If the engine has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("FittedRuleEngine must be fitted before reporting cross-fitting")
        return dict(self._cross_fit_report.model_dump(mode="json")) if self._cross_fit_report else {}

"""Point-in-time relational aggregates using explicit availability timestamps."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from featcopilot.core.base import BaseEngine, EngineConfig
from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet


class PointInTimeAggregation(BaseModel):
    """Declare a related-table aggregate available strictly before a prediction.

    Parameters
    ----------
    name : str
        Unique output column name.
    table : str
        Related table supplied at fit time.
    entity_column : str
        Entity key in the related table.
    available_at : str
        Related-table column recording when the value became available.
    aggregation : str
        One of count, sum, mean, min, max, std, or last.
    value_column : str, optional
        Numeric value to aggregate. Optional only for row counts.
    window : str, optional
        Trailing duration such as '30D'. The lower boundary is inclusive.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    table: str
    entity_column: str
    available_at: str
    aggregation: Literal["count", "sum", "mean", "min", "max", "std", "last"] = "mean"
    value_column: str | None = None
    window: str | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> PointInTimeAggregation:
        if not all(value.strip() for value in (self.name, self.table, self.entity_column, self.available_at)):
            raise ValueError("Aggregate names, table names and key/time columns must be nonempty")
        if self.aggregation != "count" and not self.value_column:
            raise ValueError("value_column is required except for row counts")
        if self.window is not None:
            duration = pd.Timedelta(self.window)
            if pd.isna(duration) or duration <= pd.Timedelta(0):
                raise ValueError("window must be a positive duration")
        return self


def _times(values: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(values):
        raise ValueError(f"{name} requires datetime values; convert numeric times with an explicit unit first")
    parsed = pd.to_datetime(values, errors="raise", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{name} cannot contain missing timestamps")
    converted = np.asarray(parsed.to_numpy(dtype="datetime64[ns]"))
    if not np.asarray(pd.to_datetime(converted, utc=True) == parsed).all():
        raise ValueError(f"{name} contains timestamps outside the supported nanosecond range")
    return converted.astype(np.int64)


def _entities(values: pd.Series, name: str) -> list[Hashable]:
    if values.isna().any():
        raise ValueError(f"{name} cannot contain missing entity keys")
    result: list[Hashable] = []
    for value in values:
        if not isinstance(value, Hashable):
            raise ValueError(f"{name} must contain scalar hashable entity keys")
        result.append(value)
    return result


class PointInTimeRelationalEngine(BaseEngine):
    """Generate relational history features without consuming future records.

    Related values are eligible only when their declared availability time is
    strictly earlier than the query's prediction timestamp. The engine retains
    a copy of the supplied history, never the target. Unknown entities have
    count/sum zero and missing mean/min/max/std/last values.

    Parameters
    ----------
    aggregations : list
        Validated aggregation definitions or dictionaries.
    entity_column : str
        Query-frame entity key.
    prediction_time_column : str
        Query-frame prediction timestamp.
    verbose : bool, default=False
        Standard engine verbosity.

    Notes
    -----
    Availability time is not necessarily event time: callers must account for
    publication delays. This engine enforces declared timestamps, not the truth
    of source metadata. History supplied at fit remains fixed; refit explicitly
    to include newly available observations. Original query columns, including
    identifiers/timestamps, are preserved; select appropriate model columns
    downstream. Serialized engines contain training-derived history.
    """

    def __init__(
        self,
        aggregations: list[PointInTimeAggregation | dict[str, Any]],
        entity_column: str,
        prediction_time_column: str,
        verbose: bool = False,
    ):
        super().__init__(config=EngineConfig(name="PointInTimeRelationalEngine", verbose=verbose))
        self.aggregations = [PointInTimeAggregation.model_validate(value) for value in aggregations]
        self.entity_column = entity_column
        self.prediction_time_column = prediction_time_column
        if not entity_column or not prediction_time_column or entity_column == prediction_time_column:
            raise ValueError("Distinct query entity and prediction-time columns are required")
        names = [spec.name for spec in self.aggregations]
        if len(names) != len(set(names)):
            raise ValueError("Aggregate output names must be unique")
        self._input_columns: list[str] = []
        self._history: dict[str, dict[Hashable, tuple[np.ndarray, np.ndarray]]] = {}
        self._feature_set = FeatureSet()

    def _query(self, X: pd.DataFrame | np.ndarray) -> tuple[pd.DataFrame, list[Hashable], np.ndarray]:
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise ValueError("Point-in-time queries require a DataFrame with unique column names")
        for column in (self.entity_column, self.prediction_time_column):
            if column not in X:
                raise ValueError(f"Missing query column {column!r}")
        return (
            X,
            _entities(X[self.entity_column], self.entity_column),
            _times(X[self.prediction_time_column], self.prediction_time_column),
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        *,
        related_tables: dict[str, pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> PointInTimeRelationalEngine:
        """Copy related history and validate declared keys, times and numeric values."""
        self._is_fitted = False
        self._history = {}
        self._feature_names = []
        self._feature_metadata = {}
        self._input_columns = []
        self._feature_set = FeatureSet()
        if kwargs:
            raise TypeError(f"Unsupported fit arguments: {sorted(kwargs)}")
        frame, _, _ = self._query(X)
        self._input_columns = frame.columns.tolist()
        tables = related_tables if related_tables is not None else {}
        for spec in self.aggregations:
            if spec.name in frame:
                raise ValueError(f"Aggregate output {spec.name!r} collides with an input column")
            if spec.table not in tables:
                raise ValueError(f"Missing related table {spec.table!r}")
            history = tables[spec.table]
            if not isinstance(history, pd.DataFrame) or history.columns.duplicated().any():
                raise ValueError(f"Related table {spec.table!r} must have unique columns")
            required = [spec.entity_column, spec.available_at]
            if spec.value_column is not None:
                required.append(spec.value_column)
            missing = [column for column in required if column not in history]
            if missing:
                raise ValueError(f"Related table {spec.table!r} lacks columns {missing}")
            entities = _entities(history[spec.entity_column], f"{spec.table}.{spec.entity_column}")
            times = _times(history[spec.available_at], f"{spec.table}.{spec.available_at}")
            if spec.value_column is None:
                values = np.ones(len(history), dtype=float)
            else:
                if not pd.api.types.is_numeric_dtype(history[spec.value_column]) or pd.api.types.is_complex_dtype(
                    history[spec.value_column]
                ):
                    raise ValueError(f"{spec.table}.{spec.value_column} must be real numeric")
                values = history[spec.value_column].to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(values).any():
                    raise ValueError(f"{spec.table}.{spec.value_column} contains infinite values")
            positions: dict[Hashable, list[int]] = {}
            for index, entity in enumerate(entities):
                positions.setdefault(entity, []).append(index)
            groups = {}
            for entity, indices in positions.items():
                order = np.asarray(indices)[np.argsort(times[indices], kind="stable")]
                groups[entity] = (times[order].copy(), values[order].copy())
            self._history[spec.name] = groups
            self._feature_set.add(
                Feature(
                    name=spec.name,
                    origin=FeatureOrigin.AGGREGATION,
                    source_columns=[self.entity_column, self.prediction_time_column],
                    transformation="point_in_time_" + spec.aggregation,
                    explanation=f"{spec.aggregation} from {spec.table}, available strictly before prediction time",
                    metadata={"point_in_time": spec.model_dump()},
                )
            )
        self._feature_names = [spec.name for spec in self.aggregations]
        self._feature_metadata = {feature.name: feature.to_dict() for feature in self._feature_set}
        self._is_fitted = True
        return self

    def _transform_fit_data(self, X, y=None, **kwargs) -> pd.DataFrame:
        return self.transform(X)

    @staticmethod
    def _aggregate(values: np.ndarray, method: str) -> float:
        finite = values[np.isfinite(values)]
        if method == "count":
            return float(len(finite))
        if method == "sum":
            return float(np.sum(finite))
        if not len(finite):
            return float("nan")
        if method == "mean":
            return float(np.mean(finite))
        if method == "min":
            return float(np.min(finite))
        if method == "max":
            return float(np.max(finite))
        if method == "std":
            return float(np.std(finite, ddof=0))
        if method == "last":
            return float(finite[-1])
        raise ValueError(f"Unknown aggregation {method!r}")

    def transform(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> pd.DataFrame:
        """Apply fixed history using per-row cutoffs, preserving index and row order."""
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform")
        if kwargs:
            raise TypeError("Transform uses frozen history; refit to change related tables or configuration")
        frame, entities, cutoffs = self._query(X)
        if set(frame.columns) != set(self._input_columns):
            raise ValueError("Query feature schema differs from fit")
        result = frame.loc[:, self._input_columns].copy()
        for spec in self.aggregations:
            window_ns = None if spec.window is None else pd.Timedelta(spec.window).value
            values = []
            for entity, cutoff in zip(entities, cutoffs, strict=True):
                history = self._history[spec.name].get(entity)
                eligible: NDArray[np.float64]
                if history is None:
                    eligible = np.empty(0)
                else:
                    times, observations = history
                    right = int(np.searchsorted(times, cutoff, side="left"))
                    left = 0 if window_ns is None else int(np.searchsorted(times, int(cutoff) - window_ns, side="left"))
                    eligible = observations[left:right]
                with np.errstate(over="raise", invalid="raise"):
                    try:
                        value = self._aggregate(eligible, spec.aggregation)
                    except FloatingPointError as exc:
                        raise ValueError(f"Aggregate {spec.name!r} overflowed; rescale its source values") from exc
                values.append(value)
            result[spec.name] = values
        return result

    def get_feature_set(self) -> FeatureSet:
        """Return metadata for the fitted point-in-time aggregates."""
        return self._feature_set

"""Shared positional IID, grouped and forward-only supervised split boundaries."""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold, train_test_split

from featcopilot.utils.task import resolve_task


@dataclass(frozen=True)
class SplitMetadata:
    """
    Validated positional entity or chronological block identifiers.

    Parameters
    ----------
    kind : {'iid', 'grouped', 'temporal'}, default='iid'
        Split protocol.
    values : ndarray or None, default=None
        Integer entity identifiers or chronologically ordered time-block codes.
        Construct these with ``validate_split_metadata`` rather than manually.
    label_end_times : ndarray or None, default=None
        Label-availability codes on the SAME ordered timeline as temporal
        ``values``. These are not durations or independently factorized codes.
    """

    kind: Literal["iid", "grouped", "temporal"] = "iid"
    values: np.ndarray | None = None
    label_end_times: np.ndarray | None = None

    @property
    def groups(self) -> np.ndarray | None:
        """Get entity identifiers for the public split helpers."""
        return self.values if self.kind == "grouped" else None

    @property
    def timestamps(self) -> np.ndarray | None:
        """Get ordered time-block identifiers for the public split helpers."""
        return self.values if self.kind == "temporal" else None

    def subset(self, indices: np.ndarray) -> "SplitMetadata":
        """Subset by row position without changing the protocol."""
        return SplitMetadata(
            self.kind,
            None if self.values is None else self.values[indices],
            None if self.label_end_times is None else self.label_end_times[indices],
        )


def validate_split_metadata(
    groups: ArrayLike | None,
    timestamps: ArrayLike | None,
    n_rows: int,
    *,
    label_end_times: ArrayLike | None = None,
) -> SplitMetadata:
    """
    Normalize split metadata without aligning pandas index labels.

    Parameters
    ----------
    groups : array-like or None
        One scalar, hashable entity identifier per row.
    timestamps : array-like or None
        One finite numeric or parseable datetime timestamp per row. Datetimes
        normalize to UTC; equal instants form one indivisible block.
    n_rows : int
        Expected positional length.
    label_end_times : array-like or None, default=None
        Positionally aligned label-availability times, requiring timestamps.
        Each must be at least its prediction timestamp. Both arrays must use
        the same time domain: finite real numbers in caller-specified matching
        units, or datetimes normalized to UTC. Numeric and datetime inputs
        cannot be mixed. Numeric units cannot be inferred from their values.

    Returns
    -------
    SplitMetadata
        Entity codes or chronologically ordered time-block codes.

    Raises
    ------
    ValueError
        For missing, malformed or mis-sized metadata, or simultaneous groups
        and timestamps. Joint group/temporal splitting is not implemented.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, Integral) or n_rows < 0:
        raise ValueError("n_rows must be a nonnegative integer")
    if groups is not None and timestamps is not None:
        raise ValueError("groups and timestamps together are not supported; choose one explicit split boundary")
    if label_end_times is not None and timestamps is None:
        raise ValueError("label_end_times requires timestamps in matching units")
    if groups is None and timestamps is None:
        return SplitMetadata()
    name = "groups" if groups is not None else "timestamps"
    values = _metadata_values(groups if groups is not None else timestamps, name, n_rows)
    if groups is not None:
        try:
            codes, _ = pd.factorize(values, sort=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("groups must contain scalar, hashable entity labels") from exc
        return SplitMetadata("grouped", np.asarray(codes))
    prediction, domain = _time_values(values, "timestamps")
    if label_end_times is None:
        codes, _ = pd.factorize(prediction, sort=True)
        return SplitMetadata("temporal", np.asarray(codes))
    ends, end_domain = _time_values(_metadata_values(label_end_times, "label_end_times", n_rows), "label_end_times")
    if domain != end_domain:
        raise ValueError("label_end_times and timestamps must use the same time domain and matching numeric units")
    # Joint object factorization preserves datetime resolutions and large integer
    # precision without coercing a mixed integer/float timeline to float64.
    timeline = pd.concat([prediction.astype(object), ends.astype(object)], ignore_index=True)
    codes, _ = pd.factorize(timeline, sort=True)
    prediction_codes, end_codes = np.asarray(codes[:n_rows]), np.asarray(codes[n_rows:])
    if (end_codes < prediction_codes).any():
        raise ValueError("label_end_times must be greater than or equal to timestamps for every row")
    return SplitMetadata("temporal", prediction_codes, end_codes)


def _metadata_values(values: ArrayLike | None, name: str, n_rows: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n_rows:
        raise ValueError(f"{name} must be one-dimensional and contain the same number of rows as the data")
    if pd.isna(array).any():
        raise ValueError(f"{name} must not contain missing values")
    return array


def _time_values(values: np.ndarray, name: str) -> tuple[pd.Series, str]:
    inferred = pd.Series(values).infer_objects()
    if pd.api.types.is_numeric_dtype(inferred.dtype):
        if pd.api.types.is_complex_dtype(inferred.dtype) or not np.isfinite(inferred.to_numpy()).all():
            raise ValueError(f"{name} must contain finite real numeric values or valid datetimes")
        return inferred, "numeric"
    if any(isinstance(value, Real) for value in inferred):
        raise ValueError(f"{name} must not mix numeric and datetime units")
    try:
        parsed = pd.to_datetime(inferred, errors="raise", utc=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numeric values or valid datetimes") from exc
    if parsed.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    return parsed, "datetime"


def make_cv_splits(
    y: ArrayLike,
    task: str,
    n_splits: int,
    random_state: int,
    *,
    groups: ArrayLike | None = None,
    timestamps: ArrayLike | None = None,
    label_end_times: ArrayLike | None = None,
    allow_constant_target: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build deterministic CV boundaries without entity or future-data overlap.

    Parameters
    ----------
    y : array-like
        One-dimensional supervised target, nonconstant unless regression
        explicitly opts in through ``allow_constant_target``.
    task : {'auto', 'classification', 'regression'}
        Explicit task or sklearn-compatible target inference.
    n_splits : int
        Requested maximum number of folds, at least two. Classification/group
        folds are capped by the smallest class count/number of groups. Temporal
        folds are capped by the available future blocks and may number one.
        Callers requiring an exact protocol count must check the returned length.
    random_state : int
        Nonnegative uint32 seed for IID folds. Group and temporal folds are
        deterministic; their partitions do not depend on the seed.
    groups : array-like or None, default=None
        Positionally aligned entity labels; every fold keeps groups disjoint.
    timestamps : array-like or None, default=None
        Positionally aligned timestamps, mutually exclusive with groups.
    label_end_times : array-like or None, default=None
        Label-availability times in the same domain/units as timestamps. Only
        training rows with end times STRICTLY BEFORE the earliest validation
        timestamp are retained. End times equal to that cutoff are purged.
        Validation blocks remain intact; purged rows are never moved to them.
    allow_constant_target : bool, default=False
        Explicit opt-in for regression-only statistics such as target means.
        Allows constant targets and constant/nonempty one-row training folds.
        Classification coverage checks are never relaxed. Use explicit
        ``task="regression"`` for label-independent target-mean folds.

    Returns
    -------
    list of (ndarray, ndarray)
        Training/validation integer row positions. IID/group validation folds
        cover all rows once. Temporal CV leaves an initial historical prefix
        unscored and partitions the remaining time blocks into disjoint windows.

    Raises
    ------
    ValueError
        For invalid options/metadata or infeasible supervised partitions.

    Notes
    -----
    Temporal CV reserves the first half of the time blocks for initial training,
    with a minimum of two blocks for two requested folds and four otherwise.
    Subsequent training rows come from expanding strictly earlier prefixes,
    filtered by label availability when supplied.
    There is no random fallback. This preserves the predictive engine's nested
    residual-split policy; these are not official provider-specific task folds.
    """
    _validate_options(n_splits, random_state, minimum=2, allow_constant_target=allow_constant_target)
    target, resolved_task = _validate_target(y, task, allow_constant_target)
    metadata = validate_split_metadata(groups, timestamps, len(target), label_end_times=label_end_times)
    values = metadata.values
    if metadata.kind == "temporal" and values is not None:
        blocks = np.unique(values)
        first = max(2 if n_splits == 2 else 4, (len(blocks) + 1) // 2)
        available = len(blocks) - first
        if available < 1:
            raise ValueError("Too few time blocks for forward cross-validation; provide more history")
        partitions = _temporal_partitions(
            values, list(np.array_split(blocks[first:], min(n_splits, available))), metadata.label_end_times
        )
    elif metadata.kind == "grouped" and values is not None:
        count = min(n_splits, len(np.unique(values)))
        if count < 2:
            raise ValueError("Too few groups for cross-validation")
        partitions = list(GroupKFold(n_splits=count).split(np.arange(len(target)), target, groups=values))
    elif resolved_task == "classification":
        count = min(n_splits, int(pd.Series(target).value_counts().min()))
        if count < 2:
            raise ValueError(
                "Classification CV requires at least two samples per class; provide more samples per class"
            )
        partitions = list(
            StratifiedKFold(n_splits=count, shuffle=True, random_state=random_state).split(target, target)
        )
    else:
        if len(target) < n_splits:
            raise ValueError("Too few samples for the requested cross-validation folds")
        partitions = list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(target))
    _check_partitions(target, resolved_task, partitions, metadata, "Cross-validation", allow_constant_target)
    return partitions


def make_holdout_splits(
    y: ArrayLike,
    task: str,
    n_splits: int,
    random_state: int,
    *,
    test_size: float = 0.25,
    groups: ArrayLike | None = None,
    timestamps: ArrayLike | None = None,
    label_end_times: ArrayLike | None = None,
    allow_constant_target: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build repeated IID/group holdouts or disjoint forward validation windows.

    Parameters
    ----------
    y : array-like
        One-dimensional supervised target, nonconstant unless regression
        explicitly opts in through ``allow_constant_target``.
    task : {'auto', 'classification', 'regression'}
        Explicit task or sklearn-compatible target inference.
    n_splits : int
        Exact number of holdouts, at least one.
    random_state : int
        Seed for the first IID/group holdout. Subsequent seeds increment modulo
        2**32. Temporal boundaries do not depend on random state.
    test_size : float, default=0.25
        Validation fraction in (0, 1), measured in IID rows or whole groups.
        For time blocks this is the approximate fraction of the first
        train-plus-validation prefix occupied by its validation window.
    groups : array-like or None, default=None
        Positionally aligned entity labels.
    timestamps : array-like or None, default=None
        Positionally aligned timestamps, mutually exclusive with groups.
    label_end_times : array-like or None, default=None
        Positionally aligned label-availability times requiring timestamps in
        the same domain/units. Purge training rows whose end time is at or after
        the earliest validation timestamp, without changing validation blocks.
    allow_constant_target : bool, default=False
        Allow constant regression targets and nonempty constant training folds.
        Classification checks remain strict. No mode allows empty training
        history after label-availability purging.

    Returns
    -------
    list of (ndarray, ndarray)
        Training/validation row positions. Temporal windows have equal numbers
        of complete time blocks, cover the final blocks without overlap, and
        have expanding strictly earlier training prefixes.

    Raises
    ------
    ValueError
        For invalid options/metadata or infeasible supervised partitions.

    Notes
    -----
    Temporal window width is floor(B * test_size /
    (1 + (n_splits - 1) * test_size)), at least one block, where B is the
    number of distinct time blocks. Rounding can change the achieved fraction.
    At least four blocks and a two-block initial prefix are required.
    Repeated random holdouts may overlap; none is an unbiased external test
    once used for feature/hyperparameter selection.
    """
    _validate_options(n_splits, random_state, minimum=1, allow_constant_target=allow_constant_target)
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, Real)
        or not np.isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be a finite fraction strictly between zero and one")
    target, resolved_task = _validate_target(y, task, allow_constant_target)
    metadata = validate_split_metadata(groups, timestamps, len(target), label_end_times=label_end_times)
    values = metadata.values
    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    if metadata.kind == "temporal" and values is not None:
        blocks = np.unique(values)
        if len(blocks) < 4:
            raise ValueError("Temporal holdouts need at least four distinct time blocks")
        width = max(1, int(len(blocks) * test_size / (1 + (n_splits - 1) * test_size)))
        first = len(blocks) - n_splits * width
        if first < 2:
            raise ValueError("Too few time blocks for the requested holdouts and a two-block historical prefix")
        windows = [blocks[first + repeat * width : first + (repeat + 1) * width] for repeat in range(n_splits)]
        partitions = _temporal_partitions(values, windows, metadata.label_end_times)
    else:
        for repeat in range(n_splits):
            seed = (int(random_state) + repeat) % (2**32)
            if metadata.kind == "grouped" and values is not None:
                if len(np.unique(values)) < 2:
                    raise ValueError("Grouped holdouts require at least two distinct groups")
                splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                try:
                    train, validation = next(splitter.split(np.arange(len(target)), target, groups=values))
                except ValueError as exc:
                    raise ValueError(f"Cannot create the grouped holdout split: {exc}") from exc
            else:
                try:
                    train, validation = train_test_split(
                        np.arange(len(target)),
                        test_size=test_size,
                        random_state=seed,
                        stratify=target if resolved_task == "classification" else None,
                    )
                except ValueError as exc:
                    raise ValueError(f"Cannot create the internal holdout split: {exc}") from exc
            partitions.append((np.asarray(train), np.asarray(validation)))
    _check_partitions(target, resolved_task, partitions, metadata, "Holdout", allow_constant_target)
    return partitions


def validate_split_boundaries(
    train: ArrayLike,
    test: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    timestamps: ArrayLike | None = None,
    label_end_times: ArrayLike | None = None,
    n_rows: int | None = None,
) -> None:
    """
    Validate an existing split without rewriting official row positions.

    Parameters
    ----------
    train : array-like
        Nonempty, unique integer training-row positions.
    test : array-like
        Nonempty, unique integer validation/test-row positions, disjoint from
        training. Input arrays, order and membership are never modified.
    groups : array-like or None, default=None
        Full-dataset positional entity labels; entities must be disjoint.
    timestamps : array-like or None, default=None
        Full-dataset prediction timestamps. Every training time must be
        strictly earlier than every test time; equal-time blocks cannot cross.
    label_end_times : array-like or None, default=None
        Full-dataset label-availability times requiring matching timestamps.
        Every supplied training label must end strictly before the earliest
        test timestamp. Unsafe rows raise rather than being silently purged.
    n_rows : int or None, default=None
        Dataset row count, otherwise inferred from metadata. With neither
        metadata nor a row count, only index structure/nonnegativity can be
        checked, not whether positions exceed the actual dataset length.

    Returns
    -------
    None
        Successful validation. This function does not generate a replacement
        split or check target constancy/class coverage.

    Raises
    ------
    ValueError
        For invalid/overlapping indices, invalid metadata or a violated split
        boundary. Combined group/time metadata remains explicitly unsupported.
    """
    train_indices = _provided_indices(train, "train")
    test_indices = _provided_indices(test, "test")
    if np.intersect1d(train_indices, test_indices).size:
        raise ValueError("Provided train and test rows must be disjoint")
    if n_rows is None:
        provided = next(
            (
                (name, values)
                for name, values in (
                    ("groups", groups),
                    ("timestamps", timestamps),
                    ("label_end_times", label_end_times),
                )
                if values is not None
            ),
            None,
        )
        if provided is None:
            n_rows = max(int(train_indices.max()), int(test_indices.max())) + 1
        else:
            name, values = provided
            array = np.asarray(values)
            if array.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            n_rows = len(array)
    metadata = validate_split_metadata(groups, timestamps, n_rows, label_end_times=label_end_times)
    if (train_indices >= n_rows).any() or (test_indices >= n_rows).any():
        raise ValueError("Provided train/test row positions exceed the dataset row count")
    _check_metadata_boundaries(train_indices, test_indices, metadata, "Provided split")


def _provided_indices(indices: ArrayLike, name: str) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must contain nonempty one-dimensional row positions")
    if values.dtype.kind not in "iu" or (values < 0).any() or (values > np.iinfo(np.intp).max).any():
        raise ValueError(f"{name} must contain nonnegative integer row positions in the platform index range")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} row positions must be unique")
    return values.astype(np.intp, copy=False)


def _validate_options(n_splits: int, random_state: int, minimum: int, allow_constant_target: bool = False) -> None:
    if not isinstance(allow_constant_target, bool):
        raise ValueError("allow_constant_target must be an explicit bool")
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or n_splits < minimum:
        raise ValueError(f"n_splits must be an integer of at least {minimum}")
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, Integral)
        or random_state < 0
        or random_state >= 2**32
    ):
        raise ValueError("random_state must be an integer between zero and 2**32 - 1")


def _validate_target(y: ArrayLike, task: str, allow_constant_target: bool = False) -> tuple[np.ndarray, str]:
    target = np.asarray(y)
    resolved_task = resolve_task(target, task)
    target = pd.Series(target).infer_objects().to_numpy()
    if pd.api.types.is_numeric_dtype(target.dtype) and not np.isfinite(target).all():
        raise ValueError("Target y must contain finite values")
    if pd.Series(target).nunique() < 2 and not (allow_constant_target and resolved_task == "regression"):
        raise ValueError("Target y must not be constant for supervised splitting")
    return target, resolved_task


def _temporal_partitions(
    values: np.ndarray, windows: list[np.ndarray], label_end_times: np.ndarray | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    partitions: list[tuple[np.ndarray, np.ndarray]] = []
    for window in windows:
        train = np.flatnonzero(values < window[0])
        validation = np.flatnonzero(np.isin(values, window))
        if label_end_times is not None:
            train = train[label_end_times[train] < window[0]]
            if not len(train):
                raise ValueError(
                    "label_end_times purging leaves no available training rows strictly before the validation cutoff; "
                    "provide earlier labeled history"
                )
        partitions.append((train, validation))
    return partitions


def _check_partitions(
    target: np.ndarray,
    task: str,
    partitions: list[tuple[np.ndarray, np.ndarray]],
    metadata: SplitMetadata,
    context: str,
    allow_constant_target: bool = False,
) -> None:
    for train, validation in partitions:
        if not len(train) or not len(validation) or np.intersect1d(train, validation).size:
            raise ValueError(f"{context} requires nonempty, disjoint training and validation rows")
        if task == "classification" and set(target[train]) != set(target):
            raise ValueError(f"{context} training split must contain every class; provide more samples per class")
        if (len(train) < 2 or pd.Series(target[train]).nunique() < 2) and not (
            allow_constant_target and task == "regression"
        ):
            raise ValueError(f"{context} training split has a constant target or fewer than two samples")
        _check_metadata_boundaries(train, validation, metadata, context)


def _check_metadata_boundaries(
    train: np.ndarray, validation: np.ndarray, metadata: SplitMetadata, context: str
) -> None:
    values = metadata.values
    if values is not None and metadata.kind == "grouped":
        if np.intersect1d(values[train], values[validation]).size:
            raise ValueError(f"{context} must not place the same group in training and validation")
    if values is not None and metadata.kind == "temporal":
        if values[train].max() >= values[validation].min():
            raise ValueError(f"{context} requires a strictly earlier training prefix without shared time blocks")
        if metadata.label_end_times is not None:
            if metadata.label_end_times[train].max() >= values[validation].min():
                raise ValueError(f"{context} training labels must be available strictly before validation starts")

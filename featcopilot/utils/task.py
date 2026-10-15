"""Shared target validation and explicit supervised-task routing."""

from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target


def validate_task(task: str) -> None:
    """
    Validate a supervised task option.

    Parameters
    ----------
    task : str
        One of 'auto', 'classification', or 'regression'.

    Raises
    ------
    ValueError
        If the task option is unsupported.
    """
    if task not in ("auto", "classification", "regression"):
        raise ValueError(f"task must be 'auto', 'classification', or 'regression'; got {task!r}")


def resolve_task(y: ArrayLike, task: str = "auto") -> Literal["classification", "regression"]:
    """
    Resolve and validate a single-output supervised task.

    Parameters
    ----------
    y : array-like
        Non-empty, one-dimensional target with no missing values.
    task : str, default='auto'
        Explicit task or sklearn target-type inference. Regression requires
        finite real numeric values; numeric strings are not coerced.
        Classification requires discrete binary or multiclass labels.

    Returns
    -------
    {'classification', 'regression'}
        Validated task. In auto mode, integer-valued targets are classified
        as discrete regardless of their cardinality.

    Raises
    ------
    ValueError
        If the task or target is unsupported or incompatible.

    Notes
    -----
    Existing selectors retain their own historical heuristics in auto mode;
    this helper does not replace those defaults.
    """
    validate_task(task)
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError("Target y must be one-dimensional; multi-output targets are not supported")
    if values.size == 0:
        raise ValueError("Target y must be non-empty")
    if pd.isna(values).any():
        raise ValueError("Target y must not contain missing values")
    if values.dtype.kind == "O" and all(isinstance(value, Real) for value in values):
        values = values.astype(float)

    if task == "regression":
        if values.dtype.kind not in "biuf":
            raise ValueError("task='regression' requires real numeric targets, not strings or categorical labels")
        if not np.isfinite(values).all():
            raise ValueError("task='regression' requires finite numeric targets")
        return "regression"

    try:
        target_type = type_of_target(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Target y is not valid for task={task!r}: {exc}") from exc
    if target_type in ("binary", "multiclass"):
        return "classification"
    if task == "auto" and target_type == "continuous":
        return "regression"
    if task == "classification":
        raise ValueError(f"task='classification' requires discrete binary or multiclass labels; got {target_type!r}")
    raise ValueError(f"Cannot infer a supported single-output task from target type {target_type!r}")


def prepare_target(y: ArrayLike, task: str = "auto") -> np.ndarray:
    """
    Prepare a target for explicit task routing without altering auto behavior.

    Parameters
    ----------
    y : array-like
        Target values.
    task : str, default='auto'
        Explicit task or legacy, unvalidated array conversion.

    Returns
    -------
    ndarray
        Unchanged values in auto mode, finite floats for regression, or
        contiguous integer-encoded labels for classification.

    Raises
    ------
    ValueError
        If the task is unsupported or an explicit target is incompatible.
    """
    validate_task(task)
    if task == "auto":
        return np.array(y)
    resolved = resolve_task(y, task)
    if resolved == "regression":
        return np.asarray(y, dtype=float)
    encoded: np.ndarray = LabelEncoder().fit_transform(y)
    return encoded

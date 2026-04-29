"""Shared split utilities for FeatCopilot benchmarks.

Centralizes the split policy so individual benchmark scripts share the same
realistic defaults: chronological splits for forecasting/timeseries tasks and
stratified random splits for classification tasks (when class counts allow).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_benchmark_data(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    random_state: int,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Split benchmark data with task-aware defaults.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix. Only its length is used; the function returns positional
        indices (use ``X.iloc[train_idx]`` / ``X.iloc[test_idx]`` to materialize
        the splits).
    y : pandas.Series
        Target values aligned with ``X``.
    task : str
        Task identifier. Substrings ``"forecast"`` / ``"timeseries"`` trigger a
        chronological split; ``"classification"`` triggers a stratified split
        when class counts allow it. Anything else falls back to a random split.
    random_state : int
        Random state for reproducible random splits.
    test_size : float, default=0.2
        Fraction of rows held out for the test split.

    Returns
    -------
    train_idx : numpy.ndarray
        Positional indices for the training rows.
    test_idx : numpy.ndarray
        Positional indices for the test rows.
    y_train : pandas.Series
        Target values for the training rows.
    y_test : pandas.Series
        Target values for the test rows.
    """
    indices = np.arange(len(X))

    if "forecast" in task or "timeseries" in task:
        split_idx = int(len(indices) * (1 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        return train_idx, test_idx, y_train, y_test

    stratify = None
    if "classification" in task:
        try:
            class_counts = pd.Series(y).value_counts(dropna=False)
            if len(class_counts) > 1 and class_counts.min() >= 2:
                stratify = y
        except Exception:
            stratify = None

    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_idx, test_idx, y_train, y_test


__all__ = ["split_benchmark_data"]

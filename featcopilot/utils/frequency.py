"""Shared typed keys and frozen unary-frequency mappings."""

import json
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import pandas as pd


def category_key(value: object) -> str | None:
    """
    Encode a scalar using the existing fitted-rule key convention.

    Parameters
    ----------
    value : object
        String, boolean, real number or missing scalar.

    Returns
    -------
    str or None
        JSON key distinguishing strings, booleans and numbers. Integral floats
        and equal integers share a key. Missing/nonfinite numbers return None.

    Raises
    ------
    ValueError
        If the value is not a supported scalar. Its contents are not logged.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, str):
        return json.dumps(["string", value], ensure_ascii=False)
    if isinstance(value, bool | np.bool_):
        return json.dumps(["boolean", bool(value)])
    if isinstance(value, Integral):
        return json.dumps(["number", str(value)])
    if isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return json.dumps(["number", str(int(numeric)) if numeric.is_integer() else repr(numeric)])
    raise ValueError("Categorical rules support only scalar strings, booleans, numbers, and missing values")


def category_keys(series: pd.Series) -> list[str | None]:
    """Encode series values positionally with the fitted-rule key convention."""
    return [category_key(value) for value in series]


def fit_frequency_mapping(series: pd.Series) -> dict[str, float]:
    """
    Fit unary frequencies using only the supplied training rows.

    Parameters
    ----------
    series : Series
        Nonempty training values. Missing values count in the denominator,
        but do not create mapping entries. No targets are accepted.

    Returns
    -------
    dict[str, float]
        Typed keys mapped to count / total training rows, preserving the
        existing fitted-rule serialization format.

    Raises
    ------
    ValueError
        If training is empty or contains unsupported scalar types.
    """
    if not len(series):
        raise ValueError("Frequency fitting requires nonempty training data")
    counts: dict[str, float] = {}
    for key in category_keys(series):
        if key is not None:
            counts[key] = counts.get(key, 0.0) + 1.0
    return {key: count / len(series) for key, count in counts.items()}


def apply_frequency_mapping(series: pd.Series, mapping: Mapping[str, float]) -> pd.Series:
    """
    Apply a frozen frequency mapping without learning from the current batch.

    Parameters
    ----------
    series : Series
        Values to transform, with any index or batch size.
    mapping : Mapping[str, float]
        Fitted mapping from ``fit_frequency_mapping`` or its JSON-restored
        equivalent. This argument is required; no implicit fitting is performed.

    Returns
    -------
    Series
        Float frequencies with the same index. Unknown and missing/nonfinite
        values return zero. Neither the input nor the mapping is modified.

    Raises
    ------
    ValueError
        If a value is not a supported scalar.
    """
    return pd.Series(
        [mapping.get(key, 0.0) if key is not None else 0.0 for key in category_keys(series)],
        index=series.index,
        dtype=float,
    )

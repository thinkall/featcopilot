"""Validation helpers for safer feature engineering workflows."""

import re
from typing import Any, Optional

DEFAULT_LEAKAGE_KEYWORDS = [
    "target",
    "label",
    "outcome",
    "ground_truth",
    "y_true",
    "future",
    "leak",
]


def _normalize_column_name(name: Any) -> str:
    """Normalize a column name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_potential_leakage_columns(
    columns: list[str],
    target_name: Optional[str] = None,
    keywords: Optional[list[str]] = None,
) -> list[str]:
    """
    Find suspicious columns that may leak label or future information.

    Parameters
    ----------
    columns : list[str]
        Input column names to inspect.
    target_name : str, optional
        Expected target/label column name. Related variants will be flagged.
    keywords : list[str], optional
        Additional suspicious keywords to match against normalized column names.

    Returns
    -------
    list[str]
        Column names that deserve manual review for leakage.
    """
    keywords = keywords or DEFAULT_LEAKAGE_KEYWORDS
    normalized_keywords = [_normalize_column_name(keyword) for keyword in keywords]
    normalized_target = _normalize_column_name(target_name) if target_name else None

    suspicious: list[str] = []
    for column in columns:
        normalized_column = _normalize_column_name(column)

        keyword_hit = any(keyword and keyword in normalized_column for keyword in normalized_keywords)
        target_hit = normalized_target is not None and (
            normalized_column == normalized_target
            or normalized_target in normalized_column
            or normalized_column in normalized_target
        )

        if keyword_hit or target_hit:
            suspicious.append(column)

    return suspicious

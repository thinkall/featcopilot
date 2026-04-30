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
    columns: list[Any],
    target_name: Optional[Any] = None,
    keywords: Optional[list[str]] = None,
) -> list[Any]:
    """
    Find suspicious columns that may leak label or future information.

    Parameters
    ----------
    columns : list
        Input column names or labels to inspect.
    target_name : optional
        Expected target/label column name or label. Related variants will be flagged.
    keywords : list[str], optional
        Additional suspicious keywords to match against normalized column names.

    Returns
    -------
    list
        Column names or labels that deserve manual review for leakage.

    Notes
    -----
    Target-name matching is intentionally fuzzy: labels are normalized and substring
    variants are flagged so derived names such as ``target_encoded`` are reviewed.

    Pass ``keywords=[]`` to opt out of keyword-based matching entirely (only
    the explicit ``target_name`` will be used). ``target_name`` is treated as
    absent only when it is ``None`` *or* normalizes to an empty string after
    stripping non-alphanumerics; this lets falsy-but-meaningful values such
    as ``0`` participate in matching while preventing ``target_name=""``
    from matching every column via the empty-substring trap.
    """
    # Use ``is None`` defaulting (rather than ``or``) so callers can pass an
    # empty list to explicitly disable keyword matching.
    if keywords is None:
        keywords = DEFAULT_LEAKAGE_KEYWORDS
    normalized_keywords = [_normalize_column_name(keyword) for keyword in keywords]

    # Use explicit ``is None`` so falsy but meaningful target labels (e.g. ``0``)
    # still participate in matching. After normalization, an empty string is
    # treated as "no target" so that ``target_name=""`` (or values like ``"---"``
    # that strip to nothing) cannot match every column via the
    # ``normalized_target in normalized_column`` substring check.
    if target_name is None:
        normalized_target: Optional[str] = None
    else:
        normalized = _normalize_column_name(target_name)
        normalized_target = normalized if normalized else None

    suspicious: list[Any] = []
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

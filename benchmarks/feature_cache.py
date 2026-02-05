"""Shared utilities for caching FeatCopilot-engineered features in benchmarks."""

from __future__ import annotations

import hashlib
import pickle
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

FEATURE_CACHE_DIR = Path("benchmarks/.feature_cache")
FEATURE_CACHE_VERSION = "v1"


def sanitize_feature_names(columns: list[str]) -> list[str]:
    """Sanitize feature names for compatibility with downstream models."""
    sanitized = []
    seen: dict[str, int] = {}
    for col in columns:
        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col)).strip("_")
        if not safe:
            safe = "feature"
        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}_{seen[safe]}"
        else:
            seen[safe] = 0
        sanitized.append(safe)
    return sanitized


def sanitize_feature_frames(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply consistent sanitized column names to train/test frames."""
    new_columns = sanitize_feature_names([str(col) for col in X_train.columns])
    if len(new_columns) != len(X_train.columns):
        raise ValueError(f"Sanitized column count mismatch: {len(new_columns)} vs {len(X_train.columns)}")
    X_train_sanitized = X_train.set_axis(new_columns, axis=1)
    X_test_sanitized = X_test.set_axis(new_columns, axis=1)
    if X_train_sanitized.columns.duplicated().any():
        raise ValueError("Duplicate feature names detected after sanitization.")
    return X_train_sanitized, X_test_sanitized


def get_feature_cache_key(
    dataset_name: str,
    max_features: int,
    with_llm: bool,
    engines: list[str],
    cache_version: str,
) -> str:
    """Generate a unique cache key for feature-engineered data."""
    engine_key = "-".join(engines)
    key_str = f"{cache_version}_{dataset_name}_{max_features}_{with_llm}_{engine_key}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def get_feature_cache_path(
    dataset_name: str,
    max_features: int,
    with_llm: bool,
    engines: list[str],
    cache_version: str,
) -> Path:
    """Get the path for cached feature-engineered data."""
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_feature_cache_key(dataset_name, max_features, with_llm, engines, cache_version)
    suffix = "_llm" if with_llm else "_tabular"
    return FEATURE_CACHE_DIR / f"{dataset_name}{suffix}_{cache_key}.pkl"


def save_feature_cache(
    cache_path: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    X_train_fe: pd.DataFrame,
    X_test_fe: pd.DataFrame,
    fe_time: float,
    task: str,
    n_original: int,
    engines: list[str],
) -> None:
    """Save feature-engineered data to cache."""
    cache_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_fe": X_train_fe,
        "X_test_fe": X_test_fe,
        "fe_time": fe_time,
        "task": task,
        "n_features_original": n_original,
        "n_features_fe": X_train_fe.shape[1],
        "engines": engines,
        "timestamp": datetime.now().isoformat(),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)


def load_feature_cache(cache_path: Path) -> dict | None:
    """Load feature-engineered data from cache."""
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    if "X_train" in cache_data and "X_test" in cache_data:
        cache_data["X_train"], cache_data["X_test"] = sanitize_feature_frames(
            cache_data["X_train"], cache_data["X_test"]
        )
    if "X_train_fe" in cache_data and "X_test_fe" in cache_data:
        cache_data["X_train_fe"], cache_data["X_test_fe"] = sanitize_feature_frames(
            cache_data["X_train_fe"], cache_data["X_test_fe"]
        )
    return cache_data

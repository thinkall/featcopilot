"""
Simple Models Benchmark for FeatCopilot.

Compares simple model performance with and without FeatCopilot feature engineering.

Comparison modes:
1. Baseline (no feature engineering)
2. FeatCopilot (multi-engine per dataset)
3. FeatCopilot + LLM (if --with-llm enabled)

Models:
- Classification: RandomForestClassifier, LogisticRegression
- Regression: RandomForestRegressor, Ridge

Statistical methodology:
- 5-fold stratified cross-validation (default)
- Multiple random seeds for robust estimation
- Reports mean ± std across folds
- Wilcoxon signed-rank test for significance

Usage:
    python -m benchmarks.simple_models.run_simple_models_benchmark [options]

Examples:
    # Quick benchmark with default settings
    python -m benchmarks.simple_models.run_simple_models_benchmark

    # Run on specific datasets
    python -m benchmarks.simple_models.run_simple_models_benchmark --datasets titanic,house_prices

    # Run on all classification datasets
    python -m benchmarks.simple_models.run_simple_models_benchmark --category classification

    # Run with LLM engine enabled
    python -m benchmarks.simple_models.run_simple_models_benchmark --with-llm

    # Run only real-world datasets
    python -m benchmarks.simple_models.run_simple_models_benchmark --real-world

    # Fast dev mode (3-fold, 1 seed)
    python -m benchmarks.simple_models.run_simple_models_benchmark --fast
"""

import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from benchmarks.datasets import (
    CATEGORY_CLASSIFICATION,
    CATEGORY_FORECASTING,
    CATEGORY_REGRESSION,
    CATEGORY_TEXT,
    is_real_world,
    list_datasets,
    list_real_world_datasets,
    load_dataset,
)
from benchmarks.feature_cache import (
    sanitize_feature_frames,
    sanitize_feature_names,
)
from featcopilot.utils.models import DEFAULT_MODEL

# Module logger for surfacing exceptions that were previously swallowed.
# We deliberately use the stdlib ``logging`` module here (rather than
# ``featcopilot.utils.logger.get_logger``) because the latter sets
# ``propagate=False`` on the ``featcopilot.*`` logger tree, which prevents
# benchmark output from reaching root-logger handlers configured by
# downstream consumers (CI runners, log aggregators, ``pytest --log-cli``).
# A vanilla ``logging.getLogger(__name__)`` here keeps the benchmark output
# routable through the consumer's normal logging configuration.
# ``logging.basicConfig`` is the consumer's responsibility.
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_FEATURES = 100
QUICK_DATASETS = ["titanic", "house_prices", "credit_risk", "bike_sharing", "customer_churn", "insurance_claims"]


# Exception types we expect to encounter during per-fold feature
# engineering. These come from sklearn / pandas / featcopilot validation
# code paths and represent recoverable user-input issues (bad columns,
# wrong dtypes, etc.). Anything else we surface via ``logger.exception``
# so genuine bugs (e.g. ``AttributeError`` from a refactor regression)
# don't get masked behind a benign-looking baseline-fallback.
_EXPECTED_FE_FAILURES: tuple[type[BaseException], ...] = (
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
    MemoryError,
    np.linalg.LinAlgError,
)


# Markers that indicate a loader returned synthetic data despite the
# registry tagging the dataset as real-world (e.g., a Kaggle/OpenML/HF
# loader that fell back to a synthesized dataset because the upstream
# fetch failed). Markers are matched case-insensitively against the
# loader_name suffix string (the 4th element of ``load_dataset`` output).
_SYNTHETIC_FALLBACK_MARKERS = ("synthetic", "-style")


def _infer_source(dataset_name: str, loader_name: str | None = None) -> str:
    """
    Infer the actual data source for a benchmark run.

    The dataset registry (``DATASET_SOURCE``) only knows whether a dataset
    is *intended* to be real-world or synthetic. Some loaders attempt to
    fetch real data and fall back to a synthesized dataset on failure
    (e.g., Kaggle/OpenML/HF loaders); in that case ``loader_name`` carries
    a marker like ``(synthetic)`` or ``(Kaggle-style)``. When such a
    marker is present we downgrade the source to ``synthetic`` so the
    real-world vs synthetic split in the report stays accurate even when
    upstream fetches fail.

    Parameters
    ----------
    dataset_name : str
        Registry key passed to ``load_dataset``.
    loader_name : str, optional
        The 4th element of ``load_dataset``'s return tuple (the loader's
        canonical label). When ``None`` the registry verdict is used.

    Returns
    -------
    str
        Either ``"real_world"`` or ``"synthetic"``.
    """
    if not is_real_world(dataset_name):
        return "synthetic"
    if loader_name:
        lname = loader_name.lower()
        if any(marker in lname for marker in _SYNTHETIC_FALLBACK_MARKERS):
            return "synthetic"
    return "real_world"


def _resolve_source(result: dict) -> str:
    """
    Return the source label for a (possibly legacy) cached result row.

    Newer runs always store ``source`` directly. Older cached results
    saved before the field existed don't have it; for those we fall back
    to ``is_real_world(result["dataset"])`` via the registry, which is
    strictly more accurate than the previous behaviour of treating every
    legacy row as synthetic. Note: this fallback can't detect a
    synthetic-fallback loader event (``loader_name`` isn't preserved in
    cached results), but for stable real-world datasets (INRIA,
    ``fake_news``) the registry verdict matches the historical run.
    """
    if "source" in result:
        return result["source"]
    dataset = result.get("dataset")
    if not isinstance(dataset, str):
        return "synthetic"
    try:
        return "real_world" if is_real_world(dataset) else "synthetic"
    except (KeyError, ValueError, TypeError):
        # ``is_real_world`` raises only ``KeyError`` for unknown datasets
        # and ``ValueError``/``TypeError`` for invalid input; anything else
        # is a genuine bug we want to surface rather than silently bucket
        # as "synthetic".
        return "synthetic"


# =============================================================================
# Models
# =============================================================================


def get_models(task: str) -> dict:
    """Get models for the task type."""
    if "classification" in task:
        return {
            "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        }
    else:
        return {
            "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            "Ridge": Ridge(alpha=1.0, random_state=42),
        }


# =============================================================================
# Evaluation
# =============================================================================


def evaluate(y_true, y_pred, y_prob, task: str) -> dict[str, float]:
    """Evaluate predictions."""
    if "classification" in task:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        }
        if y_prob is not None:
            try:
                if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            except (ValueError, IndexError):
                pass
        return metrics
    else:
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
        }


def get_primary_metric(task: str) -> str:
    """Get primary metric name for task."""
    return "accuracy" if "classification" in task else "r2"


# =============================================================================
# Data Preparation
# =============================================================================


def _safe_fillna_string(series: pd.Series, sentinel: str = "missing") -> pd.Series:
    """
    Replace NaN with ``sentinel`` and cast to string, safely for any dtype.

    Some non-numeric pandas dtypes (``Categorical``, ``datetime64``,
    ``timedelta64``, ``period``, etc.) raise on ``series.fillna("missing")``
    because the string sentinel is not a valid value for the dtype. Casting
    to object first preserves all values and lets ``fillna`` accept any
    sentinel uniformly.

    Order matters: ``fillna`` MUST run before ``astype(str)``. If the order is
    reversed, NaN becomes the literal string ``"nan"`` and lives in a separate
    code from real ``sentinel`` values — a subtle but real bug.

    Parameters
    ----------
    series : Series
        Any-dtype series. NaN/NA cells are replaced with ``sentinel``.
    sentinel : str, default ``"missing"``
        String used in place of missing values.

    Returns
    -------
    Series
        String-dtype series with no NaN.
    """
    # Cast through ``object`` for any dtype that doesn't natively accept a
    # string sentinel. ``object`` (already ``object``) and the pandas
    # extension string dtypes accept ``"missing"`` directly; everything else
    # (Categorical, datetime, timedelta, period, numeric, boolean, etc.) is
    # safer to widen first. The downstream consumer always calls
    # ``astype(str)`` on the result, so losing dtype metadata here is fine.
    if not (series.dtype == object or pd.api.types.is_string_dtype(series.dtype)):
        series = series.astype(object)
    return series.fillna(sentinel).astype(str)


def preprocess_target(y, task: str) -> np.ndarray:
    """
    Preprocess the target column.

    Target preprocessing is fold-independent — for classification we just
    label-encode the global vocabulary (which is by definition known at
    the dataset level, not learned from features), and for regression we
    cast to float. Doing this once outside the CV loop is leakage-free.

    Parameters
    ----------
    y : array-like
        Target values.
    task : str
        Task name. Substring "classification" triggers label encoding for
        any non-numeric, non-bool target dtype (object, category, pandas
        ``string``/``StringDtype``, etc.); otherwise the target is
        float-cast.

    Returns
    -------
    ndarray
        Numeric target array.
    """
    if "classification" in task:
        # Use the pandas dtype-introspection helpers so all non-numeric,
        # non-bool extension dtypes (object, category, string/StringDtype,
        # ArrowString, etc.) get label-encoded — not just the historical
        # ``object``/``category`` pair, which would otherwise leave a
        # pandas ``string``-typed series as a non-numeric ndarray and
        # break sklearn estimators downstream.
        if hasattr(y, "dtype"):
            is_numeric_or_bool = pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y)
            if not is_numeric_or_bool:
                le = LabelEncoder()
                return le.fit_transform(y.astype(str))
        return np.array(y)
    return np.array(y).astype(float)


def sanitize_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Rename columns via :func:`sanitize_feature_names`. Leakage-free (names only)."""
    column_map = dict(zip(X.columns, sanitize_feature_names(list(X.columns)), strict=True))
    return X.rename(columns=column_map)


def preprocess_X_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fold-local baseline preprocessing.

    Fits LabelEncoders (for non-numeric columns) and median-imputation values
    on **train only**, then applies the same transforms to test. This is the
    leakage-free analogue of the legacy ``preprocess_data`` (which fit on the
    full dataset before the CV split, leaking validation-fold information into
    training and biasing the paired Wilcoxon p-values).

    Numeric NaNs in the test fold use the **train** median; never compute
    medians from the test fold itself. If a numeric column is entirely
    missing in the train fold (so ``median()`` returns NaN), we fall back
    to ``0.0`` instead of leaving the NaN in place — otherwise
    ``fillna(NaN)`` would be a no-op and downstream model fitting would
    raise ``ValueError: Input X contains NaN``. Non-numeric encoding
    reuses :func:`_label_encode_non_numeric` (train-only fit + unknown
    bucket).

    Parameters
    ----------
    X_train, X_test : DataFrame
        Train/test frames for one CV fold.

    Returns
    -------
    tuple of DataFrame
        (X_train_processed, X_test_processed) — all numeric, no NaN.
    """
    # Encode non-numeric columns first (mirrors legacy ordering).
    X_train_proc, X_test_proc = _label_encode_non_numeric(X_train, X_test)

    # Median-impute numeric NaN using TRAIN medians only.
    for col in X_train_proc.select_dtypes(include=[np.number]).columns:
        train_median = X_train_proc[col].median()
        # All-missing columns produce a NaN median; ``fillna(NaN)`` is a
        # no-op and would leave NaNs for the model to choke on. Fall back
        # to 0.0 so the column becomes a constant (useless but safe).
        if pd.isna(train_median):
            train_median = 0.0
        X_train_proc[col] = X_train_proc[col].fillna(train_median)
        if col in X_test_proc.columns:
            X_test_proc[col] = X_test_proc[col].fillna(train_median)

    return X_train_proc, X_test_proc


def preprocess_data(X: pd.DataFrame, y, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Legacy single-shot preprocessor (kept for backwards-compat / smoke tests).

    .. warning::
        Fits LabelEncoders and numeric medians on the full dataset, which
        leaks validation-fold information into training when used inside
        a CV loop. Production benchmark code should call
        :func:`preprocess_target` + :func:`sanitize_columns` outside the
        fold loop and :func:`preprocess_X_train_test` inside it instead.
    """
    X_processed = X.copy()

    # Encode non-numeric columns. ``select_dtypes(exclude=[np.number, "bool",
    # "boolean"])`` mirrors the production path's ``_label_encode_non_numeric``
    # contract (numeric/bool stays as-is; everything else gets encoded). This
    # explicitly catches pandas ``string``/``StringDtype`` and other non-
    # numeric extension dtypes that an ``include=["object", "category"]``
    # filter would silently leave unencoded — leaving the caller with a
    # frame that breaks sklearn estimators expecting numeric input.
    # ``_safe_fillna_string`` handles pandas ``Categorical`` dtype (which
    # would otherwise raise on ``fillna("missing")`` if "missing" is not in
    # ``cat.categories``).
    for col in X_processed.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(_safe_fillna_string(X_processed[col]))

    # Fill numeric NaN
    for col in X_processed.select_dtypes(include=[np.number]).columns:
        X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    X_processed = sanitize_columns(X_processed)

    return X_processed, preprocess_target(y, task)


def get_featcopilot_engines(task: str, with_llm: bool) -> tuple[list[str], dict[str, Any] | None]:
    """Select FeatCopilot engines based on task type."""
    engines = ["tabular", "relational"]
    if "timeseries" in task:
        engines.append("timeseries")
    if "text" in task:
        engines.append("text")
    if with_llm:
        engines.append("llm")
        return engines, {"model": DEFAULT_MODEL, "max_suggestions": 20, "backend": "copilot"}
    return engines, None


def apply_featcopilot(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    task: str,
    max_features: int,
    with_llm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float, list[str]]:
    """Apply FeatCopilot feature engineering."""
    from featcopilot import AutoFeatureEngineer

    engines, llm_config = get_featcopilot_engines(task, with_llm)

    engineer = AutoFeatureEngineer(
        engines=engines,
        max_features=max_features,
        llm_config=llm_config,
        verbose=False,
    )

    start = time.time()
    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)
    fe_time = time.time() - start

    # Align columns
    for col in X_train_fe.columns:
        if col not in X_test_fe.columns:
            if pd.api.types.is_numeric_dtype(X_train_fe[col]):
                X_test_fe[col] = 0
            else:
                X_test_fe[col] = "missing"
    X_test_fe = X_test_fe[X_train_fe.columns]

    # Fill NaN by dtype
    numeric_cols = X_train_fe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X_train_fe[numeric_cols] = X_train_fe[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test_fe[numeric_cols] = X_test_fe[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    non_numeric_cols = [col for col in X_train_fe.columns if col not in numeric_cols]
    if non_numeric_cols:
        X_train_fe[non_numeric_cols] = X_train_fe[non_numeric_cols].astype("object").fillna("missing")
        X_test_fe[non_numeric_cols] = X_test_fe[non_numeric_cols].astype("object").fillna("missing")

    X_train_fe, X_test_fe = sanitize_feature_frames(X_train_fe, X_test_fe)

    return X_train_fe, X_test_fe, fe_time, engines


# =============================================================================
# Benchmark Runner
# =============================================================================


NON_NUMERIC_MODEL_NAMES = {"CatBoostClassifier", "CatBoostRegressor"}


def model_supports_non_numeric(model) -> bool:
    """Check whether a model supports non-numeric features."""
    return model.__class__.__name__ in NON_NUMERIC_MODEL_NAMES


def _label_encode_non_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Label-encode non-numeric columns for models that require numeric input.

    Encoder is fit on **train only** to avoid information leaking from the
    held-out fold. Unseen test categories are mapped to a sentinel code
    ``len(classes_)`` (treated as a single "unknown" bucket). This keeps the
    baseline-vs-FeatCopilot comparison apples-to-apples — without this helper,
    ``run_models`` would silently drop non-numeric columns and FeatCopilot would
    lose categorical signal that the label-encoded baseline keeps.

    Parameters
    ----------
    X_train, X_test : DataFrame
        Train/test frames. Numeric columns pass through unchanged.

    Returns
    -------
    tuple of DataFrame
        (X_train_encoded, X_test_encoded) with all columns numeric.
    """
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    for col in X_train.select_dtypes(exclude=[np.number]).columns:
        # ``_safe_fillna_string`` handles pandas ``Categorical`` dtype (which
        # would otherwise raise on ``fillna("missing")`` if "missing" is not
        # already a category).
        train_str = _safe_fillna_string(X_train[col])
        test_str = _safe_fillna_string(X_test[col])

        le = LabelEncoder()
        le.fit(train_str)
        mapping = {cls: i for i, cls in enumerate(le.classes_)}
        unknown_code = len(le.classes_)

        X_train_enc[col] = train_str.map(mapping).astype(int)
        X_test_enc[col] = test_str.map(mapping).fillna(unknown_code).astype(int)
    return X_train_enc, X_test_enc


def run_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test, task: str, label: str, quiet: bool = False
) -> dict[str, dict]:
    """Run all models and return metrics."""
    models = get_models(task)
    results = {}
    primary_metric = get_primary_metric(task)

    # Pre-compute encoded variants once. Lazily evaluated only if any model in
    # this batch requires numeric input (i.e. doesn't support non-numeric).
    encoded_cache: tuple[pd.DataFrame, pd.DataFrame] | None = None

    for name, model in models.items():
        non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0 and not model_supports_non_numeric(model):
            # Encode non-numeric columns instead of silently dropping them so the
            # comparison stays apples-to-apples with the baseline (which already
            # label-encodes during preprocess_data). Train-only fit + unknown
            # bucket avoids fold leakage.
            if encoded_cache is None:
                encoded_cache = _label_encode_non_numeric(X_train, X_test)
            X_train_model, X_test_model = encoded_cache
            if X_train_model.shape[1] == 0:
                raise ValueError(f"No features available for model '{name}' in {label}")
        else:
            X_train_model = X_train
            X_test_model = X_test

        start = time.time()
        model.fit(X_train_model, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test_model)
        y_prob = model.predict_proba(X_test_model) if hasattr(model, "predict_proba") else None

        metrics = evaluate(y_test, y_pred, y_prob, task)
        metrics["train_time"] = train_time

        results[name] = metrics
        if not quiet:
            print(f"   {name}: {primary_metric}={metrics[primary_metric]:.4f}, time={train_time:.2f}s")

    return results


def run_single_benchmark(
    dataset_name: str,
    max_features: int,
    with_llm: bool = False,
    n_folds: int = 5,
    n_seeds: int = 1,
) -> dict[str, Any] | None:
    """
    Run benchmark on a single dataset using k-fold cross-validation.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to benchmark.
    max_features : int
        Maximum number of features for FeatCopilot.
    with_llm : bool
        Whether to enable LLM engine.
    n_folds : int
        Number of cross-validation folds (default: 5).
    n_seeds : int
        Number of random seeds to average over (default: 1).

    Returns
    -------
    dict or None
        Benchmark results with mean ± std across folds.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    try:
        # ``load_dataset`` returns ``(X, y, task, loader_name)``. The loader's
        # canonical name carries source information (e.g. a Kaggle loader
        # may return "House Prices (Kaggle)" on success or
        # "House Prices (Kaggle-style)" when it falls back to synthetic
        # data on fetch failure). We propagate it through ``_infer_source``
        # so the real-world vs synthetic split stays accurate even when
        # the registry tags a dataset as real-world but the upstream fetch
        # actually fell back to a synthesized dataset.
        X, y, task, loader_name = load_dataset(dataset_name)
        source = _infer_source(dataset_name, loader_name)
        print(
            f"Task: {task}, Shape: {X.shape}, Source: {'real-world' if source == 'real_world' else 'synthetic'} ({loader_name})"
        )

        # Process target once (fold-independent — labels don't change between folds).
        # X is left raw at the dataset level; baseline preprocessing is performed
        # **inside** the fold loop on per-fold train/test splits to avoid leaking
        # validation-fold statistics (LabelEncoder vocab, numeric medians) into
        # training, which would bias the paired Wilcoxon p-values.
        y_processed = preprocess_target(y, task)
        X_renamed = sanitize_columns(X)

        primary_metric = get_primary_metric(task)
        baseline_fold_scores = []
        tabular_fold_scores = []
        fe_times = []
        n_features_generated = []
        engines_used: list[str] = []
        # Track per-fold FeatCopilot failures so the silent baseline
        # fallback is visible to consumers (previously the same broad
        # ``except Exception`` would mask the failure rate behind an
        # otherwise-healthy-looking results row).
        fe_failed_folds: list[dict[str, Any]] = []

        seeds = [42 + i * 7 for i in range(n_seeds)]
        if not seeds:
            print("  Skipping: n_seeds must be >= 1")
            return None

        # Time-series / forecasting tasks need chronological folds — using
        # KFold(shuffle=True) here would leak future information into training
        # folds and produce overly optimistic scores. We mirror the policy used
        # in benchmarks/splits.split_benchmark_data.
        is_time_series = "forecast" in task or "timeseries" in task

        # TimeSeriesSplit ignores random_state by design, so averaging across
        # multiple seeds would not vary the CV folds at all. Surface this once
        # and run a single pass instead of silently dropping seeds inside the
        # loop (which previously made results['n_seeds'] misleading).
        if is_time_series and n_seeds > 1:
            print(
                f"  Note: TimeSeriesSplit ignores random_state; --n-seeds={n_seeds} "
                f"has no effect for {task} task. Running 1 pass instead."
            )
            effective_seeds = [seeds[0]]
        else:
            effective_seeds = seeds
        effective_n_seeds = len(effective_seeds)

        for seed in effective_seeds:
            if is_time_series:
                kf = TimeSeriesSplit(n_splits=n_folds)
                split_iter = kf.split(X_renamed)
            elif "classification" in task and len(np.unique(y_processed)) < 50:
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                split_iter = kf.split(X_renamed, y_processed)
            else:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                split_iter = kf.split(X_renamed)

            for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
                # Fit baseline preprocessing on TRAIN ONLY, transform both halves.
                # Without this, LabelEncoder vocab and median imputation values
                # would leak from the validation fold into training.
                X_train, X_test = preprocess_X_train_test(X_renamed.iloc[train_idx], X_renamed.iloc[test_idx])
                y_train = y_processed[train_idx]
                y_test = y_processed[test_idx]
                X_train_raw = X.iloc[train_idx]
                X_test_raw = X.iloc[test_idx]

                # --- Baseline ---
                baseline_results = run_models(X_train, X_test, y_train, y_test, task, "Baseline", quiet=True)
                best_baseline = max(baseline_results.values(), key=lambda x: x[primary_metric])
                baseline_fold_scores.append(best_baseline[primary_metric])

                # --- FeatCopilot ---
                try:
                    X_train_fe, X_test_fe, fe_time, fold_engines = apply_featcopilot(
                        X_train_raw, X_test_raw, y_train, task, max_features, with_llm=with_llm
                    )
                    tabular_results = run_models(X_train_fe, X_test_fe, y_train, y_test, task, "Tabular", quiet=True)
                    best_tabular = max(tabular_results.values(), key=lambda x: x[primary_metric])
                    tabular_fold_scores.append(best_tabular[primary_metric])
                    fe_times.append(fe_time)
                    n_features_generated.append(X_train_fe.shape[1])
                    # Record the engine list once (it's identical across folds
                    # because we configure engines from ``task`` only).
                    # ``apply_featcopilot`` returns the engine identifiers
                    # already as strings (see ``get_featcopilot_engines``), so
                    # we record them directly — applying ``__class__.__name__``
                    # would just produce ``["str", "str", ...]``.
                    if not engines_used:
                        engines_used = list(fold_engines)
                except _EXPECTED_FE_FAILURES as e:
                    # Recoverable per-fold failure (bad columns, wrong dtypes,
                    # etc.). Fall back to baseline score and record the failure
                    # so it shows up in the results dict.
                    logger.warning(
                        "FeatCopilot recoverable error on dataset=%s seed=%s fold=%s: %s: %s",
                        dataset_name,
                        seed,
                        fold_idx,
                        type(e).__name__,
                        e,
                    )
                    fe_failed_folds.append(
                        {
                            "seed": seed,
                            "fold": fold_idx,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "expected": True,
                        }
                    )
                    tabular_fold_scores.append(best_baseline[primary_metric])
                    fe_times.append(0.0)
                    # Fall back to the (per-fold) baseline feature width since
                    # FeatCopilot didn't produce engineered features this fold.
                    n_features_generated.append(X_train.shape[1])
                except Exception as e:
                    # Unexpected error — surface the full traceback so genuine
                    # bugs (e.g. a refactor regression raising ``AttributeError``)
                    # don't get masked behind a silent baseline-fallback. We
                    # still continue to the next fold so a single bad fold
                    # doesn't poison the entire dataset run.
                    logger.exception(
                        "FeatCopilot UNEXPECTED error on dataset=%s seed=%s fold=%s",
                        dataset_name,
                        seed,
                        fold_idx,
                    )
                    fe_failed_folds.append(
                        {
                            "seed": seed,
                            "fold": fold_idx,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "expected": False,
                        }
                    )
                    tabular_fold_scores.append(best_baseline[primary_metric])
                    fe_times.append(0.0)
                    n_features_generated.append(X_train.shape[1])

        baseline_scores = np.array(baseline_fold_scores)
        tabular_scores = np.array(tabular_fold_scores)

        baseline_mean = float(np.mean(baseline_scores))
        baseline_std = float(np.std(baseline_scores))
        tabular_mean = float(np.mean(tabular_scores))
        tabular_std = float(np.std(tabular_scores))
        improvement_pct = (tabular_mean - baseline_mean) / max(abs(baseline_mean), 0.001) * 100

        # Wilcoxon signed-rank test (paired)
        p_value = 1.0
        if len(baseline_scores) >= 5 and not np.allclose(baseline_scores, tabular_scores):
            try:
                _, p_value = stats.wilcoxon(tabular_scores, baseline_scores, alternative="two-sided")
            except ValueError as e:
                # ``scipy.stats.wilcoxon`` raises ``ValueError`` when the input
                # contains all-zero differences or insufficient non-zero pairs.
                # Falling back to ``p_value = 1.0`` (no significance) is the
                # right behaviour, but log so it doesn't look like a real
                # null result. Anything other than ``ValueError`` is a bug
                # we want to surface.
                logger.warning(
                    "Wilcoxon test failed for %s (n=%d), reporting p_value=1.0: %s",
                    dataset_name,
                    len(baseline_scores),
                    e,
                )
                p_value = 1.0

        # Cast to native Python ``bool`` so the in-memory results dict is
        # immediately JSON-/consumer-friendly. ``p_value`` is a NumPy scalar
        # (returned by ``scipy.stats.wilcoxon``), so the bare comparison
        # would otherwise yield a ``numpy.bool_`` and force every consumer
        # — including ``save_cache``'s ``_to_native`` helper — to special-
        # case it. Casting at the source avoids that fragile dependency.
        significant = bool(p_value < 0.05)

        print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  Tabular:  {tabular_mean:.4f} ± {tabular_std:.4f}")
        print(f"  Improvement: {improvement_pct:+.2f}% (p={p_value:.4f}{'*' if significant else ''})")

        results = {
            "dataset": dataset_name,
            "task": task,
            "source": source,
            "loader_name": loader_name,
            "n_samples": len(X),
            "n_features_original": X.shape[1],
            "n_folds": n_folds,
            "n_seeds": effective_n_seeds,
            "with_llm": with_llm,
            "baseline_best_score": baseline_mean,
            "baseline_std": baseline_std,
            "tabular_best_score": tabular_mean,
            "tabular_std": tabular_std,
            "tabular_improvement_pct": improvement_pct,
            "p_value": float(p_value),
            "significant": significant,
            "n_features_tabular": int(np.mean(n_features_generated)),
            "fe_time_tabular": float(np.mean(fe_times)),
            "engines_used": engines_used,
            "baseline_fold_scores": baseline_scores.tolist(),
            "tabular_fold_scores": tabular_scores.tolist(),
            # Per-fold FeatCopilot failure log. Empty list means every fold
            # ran the engineered pipeline cleanly. Non-empty entries record
            # the seed/fold, exception class, message, and whether the
            # exception was an *expected* validation error (``expected=True``)
            # or an *unexpected* bug (``expected=False``) so reviewers /
            # report consumers can see at a glance whether the
            # ``tabular_best_score`` is a fair comparison.
            "fe_failed_folds": fe_failed_folds,
            "n_fe_failed_folds": len(fe_failed_folds),
        }

        return results

    except Exception as e:
        # Top-level safety net: keep the benchmark loop alive when a single
        # dataset fails so the rest of the suite still produces a report.
        # Surface the full traceback (``logger.exception``) so unexpected
        # failures don't look like a benign skip.
        logger.exception("Dataset run failed for %s: %s", dataset_name, e)
        return None


def generate_report(results: list[dict], with_llm: bool, output_path: Path) -> None:
    """Generate markdown report with statistical rigor."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Separate by source AND task category. Use ``_resolve_source`` so legacy
    # cached results without a ``source`` field (loaded via ``--report-only``)
    # get bucketed via the registry instead of being silently lumped into
    # "synthetic".
    real_world = [r for r in results if _resolve_source(r) == "real_world"]
    synthetic = [r for r in results if _resolve_source(r) != "real_world"]

    # Use substring matching for both classification AND regression (e.g.
    # "text_classification", "timeseries_regression", "text_regression"),
    # so non-canonical task names land in the correct bucket and pick up the
    # right metric label instead of being swept into the catch-all "Other"
    # table. Note: "regression" is a substring of "timeseries_regression",
    # so the substring check naturally subsumes both.
    real_clf = [r for r in real_world if "classification" in r["task"]]
    real_reg = [r for r in real_world if "regression" in r["task"]]
    real_other = [r for r in real_world if "classification" not in r["task"] and "regression" not in r["task"]]
    synth_clf = [r for r in synthetic if "classification" in r["task"]]
    synth_reg = [r for r in synthetic if "regression" in r["task"]]
    synth_other = [r for r in synthetic if "classification" not in r["task"] and "regression" not in r["task"]]

    # Compute summary stats
    def compute_summary(result_list: list[dict]) -> dict:
        if not result_list:
            return {}
        improvements = [r["tabular_improvement_pct"] for r in result_list]
        n_improved = sum(1 for imp in improvements if imp > 0.5)
        n_hurt = sum(1 for imp in improvements if imp < -0.5)
        n_tied = len(improvements) - n_improved - n_hurt
        n_sig_improved = sum(1 for r in result_list if r.get("significant") and r["tabular_improvement_pct"] > 0.5)
        return {
            "total": len(result_list),
            "improved": n_improved,
            "tied": n_tied,
            "hurt": n_hurt,
            "sig_improved": n_sig_improved,
            "mean_improvement": float(np.mean(improvements)),
            "median_improvement": float(np.median(improvements)),
            "max_regression": float(min(improvements)) if improvements else 0.0,
        }

    real_summary = compute_summary(real_world)
    synth_summary = compute_summary(synthetic)
    all_summary = compute_summary(results)

    # Compute n_folds / n_seeds across results. Most runs share a single
    # value, but ``run_single_benchmark`` collapses ``effective_n_seeds`` to
    # 1 for time-series tasks (TimeSeriesSplit ignores ``random_state``), so
    # the per-dataset values can differ. Surface a min–max range when they
    # do, with an explanatory note for time-series collapse.
    def _format_range(values: list[int], default: int) -> str:
        if not values:
            return str(default)
        unique = sorted({int(v) for v in values})
        if len(unique) == 1:
            return str(unique[0])
        return f"{unique[0]}–{unique[-1]}"

    n_folds_str = _format_range([r.get("n_folds", 5) for r in results], 5)
    n_seeds_values = [r.get("n_seeds", 1) for r in results]
    n_seeds_str = _format_range(n_seeds_values, 1)
    seeds_note = ""
    if results and len({int(v) for v in n_seeds_values}) > 1:
        seeds_note = (
            "  \n_Note: ``n_seeds`` varies across datasets — time-series tasks "
            "collapse to 1 seed because ``TimeSeriesSplit`` ignores "
            "``random_state``._"
        )

    report = f"""# Simple Models Benchmark Report

**Generated:** {timestamp}
**Models:** RandomForest, LogisticRegression/Ridge
**Cross-Validation:** {n_folds_str}-fold CV × {n_seeds_str} seed(s){seeds_note}
**LLM Enabled:** {with_llm}
**Datasets:** {len(results)} ({len(real_world)} real-world, {len(synthetic)} synthetic)

## Summary — Real-World Datasets (Primary)

| Metric | Value |
|--------|-------|
| Total Datasets | {real_summary.get('total', 0)} |
| Win / Tie / Loss | {real_summary.get('improved', 0)} / {real_summary.get('tied', 0)} / {real_summary.get('hurt', 0)} |
| Significant Wins (p<0.05) | {real_summary.get('sig_improved', 0)} |
| Mean Improvement | {real_summary.get('mean_improvement', 0):+.2f}% |
| Median Improvement | {real_summary.get('median_improvement', 0):+.2f}% |
| Max Regression | {real_summary.get('max_regression', 0):+.2f}% |

## Summary — Synthetic Datasets (Supplementary)

| Metric | Value |
|--------|-------|
| Total Datasets | {synth_summary.get('total', 0)} |
| Win / Tie / Loss | {synth_summary.get('improved', 0)} / {synth_summary.get('tied', 0)} / {synth_summary.get('hurt', 0)} |
| Mean Improvement | {synth_summary.get('mean_improvement', 0):+.2f}% |

## Summary — All Datasets

| Metric | Value |
|--------|-------|
| Total Datasets | {all_summary.get('total', 0)} |
| Win / Tie / Loss | {all_summary.get('improved', 0)} / {all_summary.get('tied', 0)} / {all_summary.get('hurt', 0)} |
| Significant Wins (p<0.05) | {all_summary.get('sig_improved', 0)} |
| Mean Improvement | {all_summary.get('mean_improvement', 0):+.2f}% |
| Median Improvement | {all_summary.get('median_improvement', 0):+.2f}% |

"""

    def add_results_table(section_results: list[dict], title: str, is_regression: bool = False) -> str:
        if not section_results:
            return ""
        section = f"## {title}\n\n"
        metric_label = "R²" if is_regression else "Score"
        section += (
            f"| Dataset | Baseline {metric_label} | FeatCopilot {metric_label} | Δ% | p-value | Sig | Features |\n"
        )
        section += f"|---------|{'--' * 8}|{'--' * 8}|-----|---------|-----|----------|\n"

        for r in sorted(section_results, key=lambda x: x["tabular_improvement_pct"], reverse=True):
            sig_marker = "✓" if r.get("significant") else ""
            imp = r["tabular_improvement_pct"]
            imp_str = f"{imp:+.2f}%"
            if imp > 0.5 and r.get("significant"):
                imp_str = f"**{imp_str}** 🟢"
            elif imp < -0.5:
                imp_str = f"{imp_str} 🔴"
            section += (
                f"| {r['dataset']} "
                f"| {r['baseline_best_score']:.4f}±{r.get('baseline_std', 0):.4f} "
                f"| {r['tabular_best_score']:.4f}±{r.get('tabular_std', 0):.4f} "
                f"| {imp_str} "
                f"| {r.get('p_value', 1.0):.3f} "
                f"| {sig_marker} "
                f"| {r['n_features_original']}→{r['n_features_tabular']} |\n"
            )
        return section + "\n"

    report += add_results_table(real_clf, "Real-World Classification", is_regression=False)
    report += add_results_table(real_reg, "Real-World Regression", is_regression=True)
    if real_other:
        report += add_results_table(real_other, "Real-World Other", is_regression=False)
    report += add_results_table(synth_clf, "Synthetic Classification (Supplementary)", is_regression=False)
    report += add_results_table(synth_reg, "Synthetic Regression (Supplementary)", is_regression=True)
    if synth_other:
        report += add_results_table(synth_other, "Other Datasets (Supplementary)", is_regression=False)

    # Write report
    llm_suffix = "_LLM" if with_llm else ""
    report_file = output_path / f"SIMPLE_MODELS_BENCHMARK{llm_suffix}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_file}")


def get_cache_file(output_path: Path, with_llm: bool) -> Path:
    """Get cache file path."""
    llm_suffix = "_LLM" if with_llm else ""
    return output_path / f"SIMPLE_MODELS_CACHE{llm_suffix}.json"


def save_cache(results: list[dict], output_path: Path, with_llm: bool) -> None:
    """Save benchmark results to cache file."""
    cache_file = get_cache_file(output_path, with_llm)

    # Convert numpy types to native Python types for JSON serialization.
    # ``np.bool_`` is intentionally checked BEFORE ``np.integer`` because on
    # NumPy < 2.0 ``np.bool_`` is registered as a subtype of ``np.generic``
    # but is **not** a subclass of Python ``bool`` for ``json``'s purposes,
    # so the default encoder raises ``TypeError``. (The ``significant``
    # field returned by ``p_value < 0.05`` is a ``np.bool_`` because
    # ``p_value`` comes from scipy as a numpy float.)
    # ``np.integer`` MUST convert via ``int(...)`` (not ``float(...)``);
    # otherwise count fields like ``n_samples`` / ``n_folds`` are written
    # as ``5.0`` and large ints lose precision past 2**53.
    def _to_native(v: Any) -> Any:
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    serializable_results = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, dict):
                sr[k] = {kk: _to_native(vv) for kk, vv in v.items()}
            else:
                sr[k] = _to_native(v)
        serializable_results.append(sr)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Cache saved: {cache_file}")


def load_cache(output_path: Path, with_llm: bool) -> list[dict] | None:
    """Load benchmark results from cache file."""
    cache_file = get_cache_file(output_path, with_llm)
    if not cache_file.exists():
        print(f"Cache file not found: {cache_file}")
        return None
    with open(cache_file, encoding="utf-8") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from cache: {cache_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Simple Models Benchmark for FeatCopilot")
    parser.add_argument("--datasets", type=str, help="Comma-separated dataset names")
    parser.add_argument("--category", type=str, choices=["classification", "regression", "forecasting", "text"])
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--real-world", action="store_true", help="Run only real-world datasets")
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM engine")
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--output", type=str, default="benchmarks/simple_models")
    parser.add_argument("--report-only", action="store_true", help="Only regenerate report from cache")
    parser.add_argument("--no-cache", action="store_true", help="Don't save results to cache")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of random seeds (default: 1)")
    parser.add_argument("--fast", action="store_true", help="Fast dev mode: 3 folds, 1 seed")

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    n_folds = 3 if args.fast else args.n_folds
    n_seeds = 1 if args.fast else args.n_seeds

    if n_folds < 2:
        parser.error(f"--n-folds must be >= 2 (got {n_folds}); cross-validation requires at least 2 folds.")
    if n_seeds < 1:
        parser.error(f"--n-seeds must be >= 1 (got {n_seeds}); at least one seed is required.")

    # Report-only mode: load from cache and regenerate report
    if args.report_only:
        results = load_cache(output_path, args.with_llm)
        if results:
            generate_report(results, args.with_llm, output_path)
        return

    # Determine datasets to run
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    elif args.real_world:
        dataset_names = list_real_world_datasets(args.category)
    elif args.category:
        dataset_names = list_datasets(args.category)
    elif args.all:
        dataset_names = (
            list_datasets(CATEGORY_CLASSIFICATION)
            + list_datasets(CATEGORY_REGRESSION)
            + list_datasets(CATEGORY_FORECASTING)
            + list_datasets(CATEGORY_TEXT)
        )
    else:
        dataset_names = QUICK_DATASETS

    print("Simple Models Benchmark")
    print("=======================")
    print("Models: RandomForest, LogisticRegression/Ridge")
    print(f"Cross-Validation: {n_folds}-fold × {n_seeds} seed(s)")
    print(f"LLM enabled: {args.with_llm}")
    print(f"Datasets: {len(dataset_names)}")

    # Run benchmarks
    results = []
    for name in dataset_names:
        result = run_single_benchmark(
            name,
            args.max_features,
            args.with_llm,
            n_folds=n_folds,
            n_seeds=n_seeds,
        )
        if result:
            results.append(result)

    # Save cache and generate report. Generate the report FIRST so that
    # a serialization failure in ``save_cache`` (e.g., a future numpy dtype
    # the converter doesn't yet handle) doesn't prevent the report from
    # being written — the in-memory results survive that path even if the
    # JSON cache doesn't.
    if results:
        generate_report(results, args.with_llm, output_path)
        if not args.no_cache:
            save_cache(results, output_path, args.with_llm)


if __name__ == "__main__":
    # Suppress benchmark-noise warnings only when this module is invoked
    # directly as a CLI entrypoint. ``main()`` itself does NOT mutate the
    # global warnings filter so that programmatic / test callers don't
    # have warnings silently suppressed for the rest of the process.
    warnings.filterwarnings("ignore")
    main()

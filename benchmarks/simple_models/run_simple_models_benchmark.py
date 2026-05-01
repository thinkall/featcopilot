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
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

warnings.filterwarnings("ignore")

# Default configuration
DEFAULT_MAX_FEATURES = 100
QUICK_DATASETS = ["titanic", "house_prices", "credit_risk", "bike_sharing", "customer_churn", "insurance_claims"]


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


def preprocess_data(X: pd.DataFrame, y, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Preprocess data for modeling."""
    X_processed = X.copy()

    # Encode categorical columns
    for col in X_processed.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col].astype(str).fillna("missing"))

    # Fill numeric NaN
    for col in X_processed.select_dtypes(include=[np.number]).columns:
        X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    column_map = dict(zip(X_processed.columns, sanitize_feature_names(list(X_processed.columns))))
    X_processed = X_processed.rename(columns=column_map)

    # Process target
    if "classification" in task:
        if hasattr(y, "dtype") and (y.dtype == "object" or y.dtype.name == "category"):
            le = LabelEncoder()
            y_processed = le.fit_transform(y.astype(str))
        else:
            y_processed = np.array(y)
    else:
        y_processed = np.array(y).astype(float)

    return X_processed, y_processed


def get_featcopilot_engines(task: str, with_llm: bool) -> tuple[list[str], Optional[dict[str, Any]]]:
    """Select FeatCopilot engines based on task type."""
    engines = ["tabular", "relational"]
    if "timeseries" in task:
        engines.append("timeseries")
    if "text" in task:
        engines.append("text")
    if with_llm:
        engines.append("llm")
        return engines, {"model": "gpt-5.2", "max_suggestions": 20, "backend": "copilot"}
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


def run_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test, task: str, label: str, quiet: bool = False
) -> dict[str, dict]:
    """Run all models and return metrics."""
    models = get_models(task)
    results = {}
    primary_metric = get_primary_metric(task)

    for name, model in models.items():
        non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0 and not model_supports_non_numeric(model):
            X_train_model = X_train.select_dtypes(include=[np.number])
            X_test_model = X_test.select_dtypes(include=[np.number])
            if X_train_model.shape[1] == 0:
                raise ValueError(f"No numeric features available for model '{name}' in {label}")
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
) -> Optional[dict[str, Any]]:
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
        X, y, task, name = load_dataset(dataset_name)
        print(f"Task: {task}, Shape: {X.shape}, Source: {'real-world' if is_real_world(dataset_name) else 'synthetic'}")

        X_processed, y_processed = preprocess_data(X, y, task)

        primary_metric = get_primary_metric(task)
        baseline_fold_scores = []
        tabular_fold_scores = []
        fe_times = []
        n_features_generated = []

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
                split_iter = kf.split(X_processed)
            elif "classification" in task and len(np.unique(y_processed)) < 50:
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                split_iter = kf.split(X_processed, y_processed)
            else:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                split_iter = kf.split(X_processed)

            for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
                X_train = X_processed.iloc[train_idx]
                X_test = X_processed.iloc[test_idx]
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
                    X_train_fe, X_test_fe, fe_time, engines_used = apply_featcopilot(
                        X_train_raw, X_test_raw, y_train, task, max_features, with_llm=with_llm
                    )
                    tabular_results = run_models(X_train_fe, X_test_fe, y_train, y_test, task, "Tabular", quiet=True)
                    best_tabular = max(tabular_results.values(), key=lambda x: x[primary_metric])
                    tabular_fold_scores.append(best_tabular[primary_metric])
                    fe_times.append(fe_time)
                    n_features_generated.append(X_train_fe.shape[1])
                except Exception as e:
                    print(f"   FeatCopilot error on fold {fold_idx}: {e}")
                    tabular_fold_scores.append(best_baseline[primary_metric])
                    fe_times.append(0.0)
                    n_features_generated.append(X_processed.shape[1])

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
            except ValueError:
                p_value = 1.0

        significant = p_value < 0.05

        print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
        print(f"  Tabular:  {tabular_mean:.4f} ± {tabular_std:.4f}")
        print(f"  Improvement: {improvement_pct:+.2f}% (p={p_value:.4f}{'*' if significant else ''})")

        results = {
            "dataset": dataset_name,
            "task": task,
            "source": "real_world" if is_real_world(dataset_name) else "synthetic",
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
            "baseline_fold_scores": baseline_scores.tolist(),
            "tabular_fold_scores": tabular_scores.tolist(),
        }

        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_report(results: list[dict], with_llm: bool, output_path: Path) -> None:
    """Generate markdown report with statistical rigor."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Separate by source AND task category
    real_world = [r for r in results if r.get("source") == "real_world"]
    synthetic = [r for r in results if r.get("source") != "real_world"]

    real_clf = [r for r in real_world if r["task"] == "classification"]
    real_reg = [r for r in real_world if r["task"] == "regression"]
    synth_clf = [r for r in synthetic if "classification" in r["task"]]
    synth_reg = [r for r in synthetic if r["task"] in ("regression", "timeseries_regression")]
    synth_other = [r for r in synthetic if r["task"] not in ("classification", "regression", "timeseries_regression")]

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

    n_folds = results[0].get("n_folds", 5) if results else 5
    n_seeds = results[0].get("n_seeds", 1) if results else 1

    report = f"""# Simple Models Benchmark Report

**Generated:** {timestamp}
**Models:** RandomForest, LogisticRegression/Ridge
**Cross-Validation:** {n_folds}-fold CV × {n_seeds} seed(s)
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
    # Convert numpy types to native Python types for JSON serialization
    serializable_results = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                sr[k] = float(v)
            elif isinstance(v, np.ndarray):
                sr[k] = v.tolist()
            elif isinstance(v, dict):
                sr[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
            else:
                sr[k] = v
        serializable_results.append(sr)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Cache saved: {cache_file}")


def load_cache(output_path: Path, with_llm: bool) -> Optional[list[dict]]:
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

    # Save cache and generate report
    if results:
        if not args.no_cache:
            save_cache(results, output_path, args.with_llm)
        generate_report(results, args.with_llm, output_path)


if __name__ == "__main__":
    main()

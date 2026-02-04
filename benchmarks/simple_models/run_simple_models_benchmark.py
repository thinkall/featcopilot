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
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, ".")  # noqa: E402

from benchmarks.datasets import (  # noqa: E402
    CATEGORY_CLASSIFICATION,
    CATEGORY_FORECASTING,
    CATEGORY_REGRESSION,
    CATEGORY_TEXT,
    list_datasets,
    load_dataset,
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
        return engines, {"model": "gpt-4o-mini", "max_suggestions": 10, "backend": "copilot"}
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

    return X_train_fe, X_test_fe, fe_time, engines


# =============================================================================
# Benchmark Runner
# =============================================================================


NON_NUMERIC_MODEL_NAMES = {"CatBoostClassifier", "CatBoostRegressor"}


def model_supports_non_numeric(model) -> bool:
    """Check whether a model supports non-numeric features."""
    return model.__class__.__name__ in NON_NUMERIC_MODEL_NAMES


def run_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test, task: str, label: str) -> dict[str, dict]:
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
        print(f"   {name}: {primary_metric}={metrics[primary_metric]:.4f}, time={train_time:.2f}s")

    return results


def run_single_benchmark(
    dataset_name: str,
    max_features: int,
    with_llm: bool = False,
) -> Optional[dict[str, Any]]:
    """Run benchmark on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    try:
        # Load dataset
        X, y, task, name = load_dataset(dataset_name)
        print(f"Task: {task}, Shape: {X.shape}")

        # Preprocess
        X_processed, y_processed = preprocess_data(X, y, task)

        # Split (keep raw and processed in sync)
        stratify = y_processed if "classification" in task and len(np.unique(y_processed)) < 50 else None
        indices = np.arange(len(X_processed))
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices, y_processed, test_size=0.2, random_state=42, stratify=stratify
        )
        X_train = X_processed.iloc[train_idx]
        X_test = X_processed.iloc[test_idx]
        X_train_raw = X.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]

        results = {
            "dataset": dataset_name,
            "task": task,
            "n_samples": len(X),
            "n_features_original": X.shape[1],
            "with_llm": with_llm,
        }
        primary_metric = get_primary_metric(task)

        # --- Baseline ---
        print("\n[1/3] Baseline (no FE)...")
        baseline_results = run_models(X_train, X_test, y_train, y_test, task, "Baseline")
        results["baseline"] = baseline_results

        # Best baseline score
        best_baseline = max(baseline_results.values(), key=lambda x: x[primary_metric])
        results["baseline_best_score"] = best_baseline[primary_metric]

        # --- FeatCopilot (multi-engine) ---
        X_train_fe, X_test_fe, fe_time, engines_used = apply_featcopilot(
            X_train_raw, X_test_raw, y_train, task, max_features, with_llm=False
        )
        results["n_features_tabular"] = X_train_fe.shape[1]
        results["fe_time_tabular"] = fe_time
        results["engines_tabular"] = engines_used
        print(f"\n[2/3] FeatCopilot ({', '.join(engines_used)})...")
        print(f"   Features: {X_train_raw.shape[1]} → {X_train_fe.shape[1]}, FE time: {fe_time:.2f}s")

        tabular_results = run_models(X_train_fe, X_test_fe, y_train, y_test, task, "Tabular")
        results["tabular"] = tabular_results

        best_tabular = max(tabular_results.values(), key=lambda x: x[primary_metric])
        results["tabular_best_score"] = best_tabular[primary_metric]
        results["tabular_improvement_pct"] = (
            (best_tabular[primary_metric] - best_baseline[primary_metric])
            / max(best_baseline[primary_metric], 0.001)
            * 100
        )

        # --- FeatCopilot + LLM (if enabled) ---
        if with_llm:
            X_train_llm, X_test_llm, fe_time_llm, engines_used = apply_featcopilot(
                X_train_raw, X_test_raw, y_train, task, max_features, with_llm=True
            )
            results["n_features_llm"] = X_train_llm.shape[1]
            results["fe_time_llm"] = fe_time_llm
            results["engines_llm"] = engines_used
            print(f"\n[3/3] FeatCopilot ({', '.join(engines_used)})...")
            print(f"   Features: {X_train_raw.shape[1]} → {X_train_llm.shape[1]}, FE time: {fe_time_llm:.2f}s")

            llm_results = run_models(X_train_llm, X_test_llm, y_train, y_test, task, "LLM")
            results["llm"] = llm_results

            best_llm = max(llm_results.values(), key=lambda x: x[primary_metric])
            results["llm_best_score"] = best_llm[primary_metric]
            results["llm_improvement_pct"] = (
                (best_llm[primary_metric] - best_baseline[primary_metric])
                / max(best_baseline[primary_metric], 0.001)
                * 100
            )
        else:
            print("\n[3/3] Skipped (--with-llm not enabled)")

        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_report(results: list[dict], with_llm: bool, output_path: Path) -> None:
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Separate by task category
    clf_results = [r for r in results if r["task"] == "classification"]
    reg_results = [r for r in results if r["task"] == "regression"]
    ts_results = [r for r in results if r["task"] == "timeseries_regression"]
    text_clf_results = [r for r in results if r["task"] == "text_classification"]
    text_reg_results = [r for r in results if r["task"] == "text_regression"]

    report = f"""# Simple Models Benchmark Report

**Generated:** {timestamp}
**Models:** RandomForest, LogisticRegression/Ridge
**LLM Enabled:** {with_llm}
**Datasets:** {len(results)}

## Summary

| Metric | Value |
|--------|-------|
| Total Datasets | {len(results)} |
| Classification | {len(clf_results)} |
| Regression | {len(reg_results)} |
| Forecasting | {len(ts_results)} |
| Text Classification | {len(text_clf_results)} |
| Text Regression | {len(text_reg_results)} |
| Improved ({"LLM" if with_llm else "Tabular"}) | {sum(1 for r in results if r.get('llm_improvement_pct' if with_llm else 'tabular_improvement_pct', 0) > 0)} |
| Avg Improvement | {np.mean([r.get('llm_improvement_pct' if with_llm else 'tabular_improvement_pct', 0) for r in results]):.2f}% |

"""

    def add_classification_table(section_results: list[dict], title: str) -> str:
        """Generate classification results table."""
        if not section_results:
            return ""
        section = f"## {title}\n\n"
        section += "| Dataset | Baseline | Tabular | Improvement |"
        if with_llm:
            section += " LLM | LLM Imp |"
        section += " Features |\n"
        section += "|---------|----------|---------|-------------|"
        if with_llm:
            section += "------|---------|"
        section += "----------|\n"

        for r in section_results:
            section += f"| {r['dataset']} | {r['baseline_best_score']:.4f} | {r['tabular_best_score']:.4f} | {r['tabular_improvement_pct']:+.2f}% |"
            if with_llm and "llm_best_score" in r:
                section += f" {r['llm_best_score']:.4f} | {r['llm_improvement_pct']:+.2f}% |"
            elif with_llm:
                section += " - | - |"
            section += f" {r['n_features_original']}→{r['n_features_tabular']} |\n"
        return section + "\n"

    def add_regression_table(section_results: list[dict], title: str) -> str:
        """Generate regression results table."""
        if not section_results:
            return ""
        section = f"## {title}\n\n"
        section += "| Dataset | Baseline R² | Tabular R² | Improvement |"
        if with_llm:
            section += " LLM R² | LLM Imp |"
        section += " Features |\n"
        section += "|---------|-------------|------------|-------------|"
        if with_llm:
            section += "--------|---------|"
        section += "----------|\n"

        for r in section_results:
            section += f"| {r['dataset']} | {r['baseline_best_score']:.4f} | {r['tabular_best_score']:.4f} | {r['tabular_improvement_pct']:+.2f}% |"
            if with_llm and "llm_best_score" in r:
                section += f" {r['llm_best_score']:.4f} | {r['llm_improvement_pct']:+.2f}% |"
            elif with_llm:
                section += " - | - |"
            section += f" {r['n_features_original']}→{r['n_features_tabular']} |\n"
        return section + "\n"

    # Add all category sections
    report += add_classification_table(clf_results, "Classification Results")
    report += add_regression_table(reg_results, "Regression Results")
    report += add_regression_table(ts_results, "Forecasting Results")
    report += add_classification_table(text_clf_results, "Text Classification Results")
    report += add_regression_table(text_reg_results, "Text Regression Results")

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
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM engine")
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--output", type=str, default="benchmarks/simple_models")
    parser.add_argument("--report-only", action="store_true", help="Only regenerate report from cache")
    parser.add_argument("--no-cache", action="store_true", help="Don't save results to cache")

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Report-only mode: load from cache and regenerate report
    if args.report_only:
        results = load_cache(output_path, args.with_llm)
        if results:
            generate_report(results, args.with_llm, output_path)
        return

    # Determine datasets to run
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
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
    print(f"LLM enabled: {args.with_llm}")
    print(f"Datasets: {len(dataset_names)}")

    # Run benchmarks
    results = []
    for name in dataset_names:
        result = run_single_benchmark(name, args.max_features, args.with_llm)
        if result:
            results.append(result)

    # Save cache and generate report
    if results:
        if not args.no_cache:
            save_cache(results, output_path, args.with_llm)
        generate_report(results, args.with_llm, output_path)


if __name__ == "__main__":
    main()

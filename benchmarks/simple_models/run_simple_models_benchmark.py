"""
Simple Models Benchmark for FeatCopilot.

Compares simple model performance with and without FeatCopilot feature engineering.

Comparison modes:
1. Baseline (no feature engineering)
2. FeatCopilot (tabular engine only)
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
    CATEGORY_REGRESSION,
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


def apply_featcopilot(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    max_features: int,
    with_llm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Apply FeatCopilot feature engineering."""
    from featcopilot import AutoFeatureEngineer

    engines = ["tabular"]
    llm_config = None

    if with_llm:
        engines.append("llm")
        llm_config = {"model": "gpt-4o-mini", "max_suggestions": 10, "backend": "copilot"}

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
            X_test_fe[col] = 0
    X_test_fe = X_test_fe[X_train_fe.columns]

    # Fill NaN
    X_train_fe = X_train_fe.fillna(0).replace([np.inf, -np.inf], 0)
    X_test_fe = X_test_fe.fillna(0).replace([np.inf, -np.inf], 0)

    return X_train_fe, X_test_fe, fe_time


# =============================================================================
# Benchmark Runner
# =============================================================================


def run_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, y_test, task: str, label: str) -> dict[str, dict]:
    """Run all models and return metrics."""
    models = get_models(task)
    results = {}
    primary_metric = get_primary_metric(task)

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

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

        # Split
        stratify = y_processed if "classification" in task and len(np.unique(y_processed)) < 50 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=stratify
        )

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

        # --- FeatCopilot (tabular only) ---
        print("\n[2/3] FeatCopilot (tabular)...")
        X_train_fe, X_test_fe, fe_time = apply_featcopilot(X_train, X_test, y_train, max_features, with_llm=False)
        results["n_features_tabular"] = X_train_fe.shape[1]
        results["fe_time_tabular"] = fe_time
        print(f"   Features: {X_train.shape[1]} → {X_train_fe.shape[1]}, FE time: {fe_time:.2f}s")

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
            print("\n[3/3] FeatCopilot (tabular + LLM)...")
            X_train_llm, X_test_llm, fe_time_llm = apply_featcopilot(
                X_train, X_test, y_train, max_features, with_llm=True
            )
            results["n_features_llm"] = X_train_llm.shape[1]
            results["fe_time_llm"] = fe_time_llm
            print(f"   Features: {X_train.shape[1]} → {X_train_llm.shape[1]}, FE time: {fe_time_llm:.2f}s")

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
    date_str = datetime.now().strftime("%Y%m%d")

    # Separate by task
    clf_results = [r for r in results if "classification" in r["task"]]
    reg_results = [r for r in results if "classification" not in r["task"]]

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
| Improved ({"LLM" if with_llm else "Tabular"}) | {sum(1 for r in results if r.get('llm_improvement_pct' if with_llm else 'tabular_improvement_pct', 0) > 0)} |
| Avg Improvement | {np.mean([r.get('llm_improvement_pct' if with_llm else 'tabular_improvement_pct', 0) for r in results]):.2f}% |

"""

    if clf_results:
        report += "## Classification Results\n\n"
        report += "| Dataset | Baseline | Tabular | Improvement |"
        if with_llm:
            report += " LLM | LLM Imp |"
        report += " Features |\n"
        report += "|---------|----------|---------|-------------|"
        if with_llm:
            report += "------|---------|"
        report += "----------|\n"

        for r in clf_results:
            report += f"| {r['dataset']} | {r['baseline_best_score']:.4f} | {r['tabular_best_score']:.4f} | {r['tabular_improvement_pct']:+.2f}% |"
            if with_llm and "llm_best_score" in r:
                report += f" {r['llm_best_score']:.4f} | {r['llm_improvement_pct']:+.2f}% |"
            elif with_llm:
                report += " - | - |"
            report += f" {r['n_features_original']}→{r['n_features_tabular']} |\n"

    if reg_results:
        report += "\n## Regression Results\n\n"
        report += "| Dataset | Baseline R² | Tabular R² | Improvement |"
        if with_llm:
            report += " LLM R² | LLM Imp |"
        report += " Features |\n"
        report += "|---------|-------------|------------|-------------|"
        if with_llm:
            report += "--------|---------|"
        report += "----------|\n"

        for r in reg_results:
            report += f"| {r['dataset']} | {r['baseline_best_score']:.4f} | {r['tabular_best_score']:.4f} | {r['tabular_improvement_pct']:+.2f}% |"
            if with_llm and "llm_best_score" in r:
                report += f" {r['llm_best_score']:.4f} | {r['llm_improvement_pct']:+.2f}% |"
            elif with_llm:
                report += " - | - |"
            report += f" {r['n_features_original']}→{r['n_features_tabular']} |\n"

    # Write report
    llm_suffix = "_LLM" if with_llm else ""
    report_file = output_path / f"SIMPLE_MODELS_BENCHMARK{llm_suffix}_{date_str}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Simple Models Benchmark for FeatCopilot")
    parser.add_argument("--datasets", type=str, help="Comma-separated dataset names")
    parser.add_argument("--category", type=str, choices=["classification", "regression", "forecasting"])
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM engine")
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--output", type=str, default="benchmarks/simple_models")

    args = parser.parse_args()

    # Determine datasets to run
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    elif args.category:
        dataset_names = list_datasets(args.category)
    elif args.all:
        dataset_names = list_datasets(CATEGORY_CLASSIFICATION) + list_datasets(CATEGORY_REGRESSION)
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

    # Generate report
    if results:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        generate_report(results, args.with_llm, output_path)


if __name__ == "__main__":
    main()

"""
FLAML Benchmark for INRIA-SODA Tabular Benchmark Datasets.

This benchmark uses the curated inria-soda/tabular-benchmark from HuggingFace,
which contains real-world datasets from OpenML specifically designed for
benchmarking tabular ML models.

These are challenging, real-world datasets where feature engineering
is known to make a significant difference.

Usage:
    python benchmarks/automl/run_flaml_inria_benchmark.py [--time-budget SECONDS]

Datasets include:
- Classification: Higgs, Covertype, Jannis, MiniBooNE, Electricity, etc.
- Regression: Diamonds, House Sales, Delays Zurich, etc.
"""

# ruff: noqa: E402

import argparse
import sys
import time
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")

from benchmarks.datasets import INRIA_DATASETS, load_inria_dataset
from featcopilot import AutoFeatureEngineer

warnings.filterwarnings("ignore")

# Default configuration
DEFAULT_TIME_BUDGET = 60
MAX_FEATURES = 100
MAX_SAMPLES = 50000  # Limit samples for reasonable benchmark time

# Quick benchmark (smaller datasets)
QUICK_DATASETS = [
    "electricity",
    "credit",
    "diamonds",
    "wine_quality",
    "bank_marketing",
    "abalone",
]

# Medium benchmark
MEDIUM_DATASETS = QUICK_DATASETS + [
    "covertype",
    "jannis",
    "house_sales",
    "bike_sharing_inria",
    "eye_movements",
    "superconduct",
]

# Full benchmark (all datasets)
FULL_DATASETS = list(INRIA_DATASETS.keys())


def preprocess_data(X: pd.DataFrame, y: pd.Series, task: str) -> tuple:
    """Preprocess data for FLAML."""
    X_processed = X.copy()

    # Handle missing values
    for col in X_processed.columns:
        if X_processed[col].dtype == "object" or X_processed[col].dtype.name == "category":
            X_processed[col] = X_processed[col].fillna("missing").astype(str)
        else:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    # Handle target
    if task == "classification":
        if y.dtype == "object" or y.dtype.name == "category":
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_processed = le.fit_transform(y.astype(str))
        else:
            y_processed = y.values
    else:
        y_processed = y.values.astype(float)

    return X_processed, y_processed


def run_flaml_benchmark(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str,
    time_budget: int,
    label: str = "Baseline",
    n_classes: int = 2,
) -> dict:
    """Run FLAML AutoML benchmark."""
    from flaml import AutoML

    print(f"\n--- {label} FLAML (time_budget={time_budget}s) ---")
    print(f"  Features: {X_train.shape[1]}")

    automl = AutoML()

    start_time = time.time()
    automl.fit(
        X_train,
        y_train,
        task=task,
        time_budget=time_budget,
        seed=42,
        verbose=0,
        force_cancel=True,
    )
    train_time = time.time() - start_time

    y_pred = automl.predict(X_test)

    results = {
        "label": label,
        "n_features": X_train.shape[1],
        "train_time": train_time,
        "best_model": automl.best_estimator,
    }

    if task == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        results.update(
            {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }
        )

        if n_classes == 2:
            try:
                y_prob = automl.predict_proba(X_test)[:, 1]
                results["roc_auc"] = roc_auc_score(y_test, y_prob)
            except Exception:
                results["roc_auc"] = None

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")

    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.update({"rmse": rmse, "mae": mae, "r2": r2})

        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")

    print(f"  Best model: {automl.best_estimator}")
    print(f"  Train time: {train_time:.1f}s")

    return results


def run_single_benchmark(
    dataset_name: str,
    time_budget: int = DEFAULT_TIME_BUDGET,
    max_features: int = MAX_FEATURES,
    max_samples: int = MAX_SAMPLES,
) -> Optional[dict]:
    """Run benchmark on a single dataset."""
    if dataset_name not in INRIA_DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        return None

    _config_name, task, description = INRIA_DATASETS[dataset_name]

    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name} - {description}")
    print("=" * 70)

    try:
        # Load dataset using centralized loader
        X, y, task, _name = load_inria_dataset(dataset_name, max_samples)
        print(f"Shape: {X.shape}")
        print(f"Task: {task}")

        # Preprocess
        X_processed, y_processed = preprocess_data(X, y, task)

        n_classes = len(np.unique(y_processed)) if task == "classification" else 0
        if task == "classification":
            print(f"Classes: {n_classes}")

        # Train/test split
        stratify = y_processed if task == "classification" and n_classes < 100 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=stratify
        )

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Baseline FLAML
        baseline_results = run_flaml_benchmark(
            X_train,
            X_test,
            y_train,
            y_test,
            task=task,
            time_budget=time_budget,
            label="Baseline",
            n_classes=n_classes,
        )

        # Apply FeatCopilot
        print("\n--- Applying FeatCopilot ---")
        fe_start = time.time()

        engineer = AutoFeatureEngineer(
            engines=["tabular"],
            max_features=max_features,
            verbose=True,
        )

        X_train_fe = engineer.fit_transform(X_train, y_train)
        X_test_fe = engineer.transform(X_test)

        fe_time = time.time() - fe_start

        # Align columns
        common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
        X_train_fe = X_train_fe[common_cols].copy()
        X_test_fe = X_test_fe[common_cols].copy()

        # Handle NaN
        for col in X_train_fe.columns:
            if X_train_fe[col].dtype == "object":
                X_train_fe[col] = X_train_fe[col].fillna("missing")
                X_test_fe[col] = X_test_fe[col].fillna("missing")
            else:
                med = X_train_fe[col].median()
                X_train_fe[col] = X_train_fe[col].fillna(med)
                X_test_fe[col] = X_test_fe[col].fillna(med)

        print(f"  Features: {X_train.shape[1]} -> {X_train_fe.shape[1]}")
        print(f"  FE Time: {fe_time:.2f}s")

        # FeatCopilot FLAML
        fc_results = run_flaml_benchmark(
            X_train_fe,
            X_test_fe,
            y_train,
            y_test,
            task=task,
            time_budget=time_budget,
            label="FeatCopilot",
            n_classes=n_classes,
        )

        # Calculate improvement
        if task == "classification":
            baseline_metric = baseline_results["f1_weighted"]
            fc_metric = fc_results["f1_weighted"]
            metric_name = "F1 (weighted)"
        else:
            baseline_metric = baseline_results["r2"]
            fc_metric = fc_results["r2"]
            metric_name = "R²"

        improvement = ((fc_metric - baseline_metric) / abs(baseline_metric) * 100) if baseline_metric != 0 else 0

        return {
            "dataset": dataset_name,
            "description": description,
            "task": task,
            "n_samples": len(X),
            "n_features_original": X_train.shape[1],
            "n_features_fe": X_train_fe.shape[1],
            "baseline": baseline_results,
            "featcopilot": fc_results,
            "fe_time": fe_time,
            "metric_name": metric_name,
            "baseline_metric": baseline_metric,
            "fc_metric": fc_metric,
            "improvement_pct": improvement,
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_report(results: list[dict], time_budget: int) -> str:
    """Generate markdown report."""
    report = []
    report.append("# FLAML INRIA-SODA Tabular Benchmark Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Time Budget:** {time_budget}s per FLAML run\n")
    report.append(f"**Max Samples:** {MAX_SAMPLES:,}\n\n")

    # Summary
    report.append("## Summary\n\n")
    report.append("| Dataset | Task | Samples | Features | Baseline | +FeatCopilot | Improvement | FE Time |\n")
    report.append("|---------|------|---------|----------|----------|--------------|-------------|--------|\n")

    improvements = []
    for r in results:
        if r is None:
            continue
        baseline_str = f"{r['baseline_metric']:.4f}"
        fc_str = f"{r['fc_metric']:.4f}"
        imp_str = f"{r['improvement_pct']:+.2f}%"
        fe_str = f"{r['n_features_original']}→{r['n_features_fe']}"

        report.append(
            f"| {r['dataset']} | {r['task'][:5]} | {r['n_samples']:,} | {fe_str} | "
            f"{baseline_str} | {fc_str} | {imp_str} | {r['fe_time']:.1f}s |\n"
        )
        improvements.append(r["improvement_pct"])

    # Statistics
    if improvements:
        report.append("\n## Overall Statistics\n\n")
        positive = [i for i in improvements if i > 0]
        negative = [i for i in improvements if i < 0]

        report.append(f"- **Datasets tested:** {len(improvements)}\n")
        report.append(f"- **FeatCopilot improved:** {len(positive)} datasets\n")
        report.append(f"- **Baseline better:** {len(negative)} datasets\n")
        report.append(f"- **Average improvement:** {np.mean(improvements):+.2f}%\n")
        report.append(f"- **Max improvement:** {max(improvements):+.2f}%\n")
        report.append(f"- **Min improvement:** {min(improvements):+.2f}%\n")

    # Detailed results
    report.append("\n## Detailed Results\n")

    for r in results:
        if r is None:
            continue

        report.append(f"\n### {r['dataset']} - {r['description']}\n")
        report.append(f"- **Task:** {r['task']}\n")
        report.append(f"- **Samples:** {r['n_samples']:,}\n")
        report.append(f"- **Features:** {r['n_features_original']} → {r['n_features_fe']}\n")
        report.append(f"- **FE Time:** {r['fe_time']:.1f}s\n\n")

        report.append("| Metric | Baseline | FeatCopilot |\n")
        report.append("|--------|----------|-------------|\n")

        if r["task"] == "classification":
            report.append(f"| Accuracy | {r['baseline']['accuracy']:.4f} | {r['featcopilot']['accuracy']:.4f} |\n")
            report.append(
                f"| F1 (weighted) | {r['baseline']['f1_weighted']:.4f} | {r['featcopilot']['f1_weighted']:.4f} |\n"
            )
        else:
            report.append(f"| RMSE | {r['baseline']['rmse']:.4f} | {r['featcopilot']['rmse']:.4f} |\n")
            report.append(f"| R² | {r['baseline']['r2']:.4f} | {r['featcopilot']['r2']:.4f} |\n")

        report.append(f"| Best Model | {r['baseline']['best_model']} | {r['featcopilot']['best_model']} |\n")

    return "".join(report)


def main():
    parser = argparse.ArgumentParser(description="FLAML Benchmark on INRIA-SODA Datasets")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names")
    parser.add_argument("--time-budget", type=int, default=DEFAULT_TIME_BUDGET, help="Time budget per run")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES, help="Max FeatCopilot features")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES, help="Max samples per dataset")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (6 datasets)")
    parser.add_argument("--medium", action="store_true", help="Run medium benchmark (12 datasets)")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--output", type=str, default="benchmarks/automl/FLAML_INRIA_BENCHMARK_REPORT.md")

    args = parser.parse_args()

    print("=" * 70)
    print("FLAML Benchmark - INRIA-SODA Tabular Datasets")
    print("=" * 70)

    # Check FLAML
    try:
        import flaml

        print(f"FLAML version: {flaml.__version__}")
    except ImportError:
        print("ERROR: FLAML not installed")
        sys.exit(1)

    # Determine datasets
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    elif args.all:
        dataset_names = FULL_DATASETS
    elif args.medium:
        dataset_names = MEDIUM_DATASETS
    else:
        dataset_names = QUICK_DATASETS

    print(f"Time budget: {args.time_budget}s")
    print(f"Max samples: {args.max_samples:,}")
    print(f"Datasets ({len(dataset_names)}): {dataset_names}")

    # Run benchmarks
    all_results = []
    total_start = time.time()

    for name in dataset_names:
        result = run_single_benchmark(
            name,
            time_budget=args.time_budget,
            max_features=args.max_features,
            max_samples=args.max_samples,
        )
        all_results.append(result)

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time / 60:.1f} minutes")

    successful = [r for r in all_results if r is not None]
    improved = [r for r in successful if r["improvement_pct"] > 0]

    print(f"\nDatasets tested: {len(successful)}/{len(dataset_names)}")
    print(f"FeatCopilot improved: {len(improved)}/{len(successful)}")

    if successful:
        improvements = [r["improvement_pct"] for r in successful]
        print(f"Average improvement: {np.mean(improvements):+.2f}%")

    # Generate report
    report = generate_report(all_results, args.time_budget)
    print("\n" + report)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {args.output}")

    return all_results


if __name__ == "__main__":
    main()

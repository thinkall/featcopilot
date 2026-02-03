"""
FLAML Benchmark for Real-World Datasets.

This benchmark evaluates FLAML AutoML performance on curated real-world datasets
comparing baseline performance vs. FeatCopilot-enhanced features.

Datasets:
- Classification: House Prices, Employee Attrition, Telco Churn, Adult Census,
                  Credit Card Fraud, Spaceship Titanic, Home Credit
- Regression: Bike Sharing, Medical Cost, Wine Quality, Life Expectancy, Store Sales

Usage:
    python benchmarks/automl/run_flaml_realworld_benchmark.py [--datasets NAMES] [--time-budget SECONDS]

Examples:
    # Run all datasets
    python benchmarks/automl/run_flaml_realworld_benchmark.py

    # Run specific datasets
    python benchmarks/automl/run_flaml_realworld_benchmark.py --datasets house_prices,medical_cost

    # Run with longer time budget
    python benchmarks/automl/run_flaml_realworld_benchmark.py --time-budget 120
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
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, ".")

from benchmarks.datasets import (
    load_kaggle_bike_sharing,
    load_kaggle_employee_attrition,
    load_kaggle_home_credit,
    load_kaggle_house_prices,
    load_kaggle_life_expectancy,
    load_kaggle_medical_cost,
    load_kaggle_spaceship_titanic,
    load_kaggle_store_sales,
    load_kaggle_telco_churn,
    load_openml_adult_census,
    load_openml_wine_quality,
)
from featcopilot import AutoFeatureEngineer

warnings.filterwarnings("ignore")

# Default configuration
DEFAULT_TIME_BUDGET = 60  # seconds per FLAML run
MAX_FEATURES = 100  # max features from FeatCopilot

# Dataset registry mapping names to loaders
DATASET_REGISTRY = {
    # Classification datasets
    "house_prices": load_kaggle_house_prices,
    "employee_attrition": load_kaggle_employee_attrition,
    "telco_churn": load_kaggle_telco_churn,
    "adult_census": load_openml_adult_census,
    "spaceship_titanic": load_kaggle_spaceship_titanic,
    "home_credit": load_kaggle_home_credit,
    # Regression datasets
    "bike_sharing": load_kaggle_bike_sharing,
    "medical_cost": load_kaggle_medical_cost,
    "wine_quality": load_openml_wine_quality,
    "life_expectancy": load_kaggle_life_expectancy,
    "store_sales": load_kaggle_store_sales,
}

# Quick benchmark datasets (smaller, faster)
QUICK_DATASETS = [
    "medical_cost",
    "employee_attrition",
    "wine_quality",
    "bike_sharing",
]

# Full benchmark datasets
FULL_DATASETS = list(DATASET_REGISTRY.keys())


def preprocess_data(X: pd.DataFrame, y: pd.Series, task: str) -> tuple:
    """
    Preprocess data for FLAML.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    task : str
        Task type ("classification" or "regression").

    Returns
    -------
    X_processed : pd.DataFrame
        Processed features.
    y_processed : np.ndarray
        Processed target.
    label_encoder : Optional[LabelEncoder]
        Label encoder for classification tasks.
    """
    X_processed = X.copy()

    # Handle missing values
    for col in X_processed.columns:
        if X_processed[col].dtype == "object":
            X_processed[col] = X_processed[col].fillna("missing")
        else:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    # Encode target for classification
    label_encoder = None
    if task == "classification":
        if y.dtype == "object" or y.dtype.name == "category":
            label_encoder = LabelEncoder()
            y_processed = label_encoder.fit_transform(y)
        else:
            y_processed = y.values
    else:
        y_processed = y.values

    return X_processed, y_processed, label_encoder


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
    """
    Run FLAML AutoML benchmark.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train and test feature matrices.
    y_train, y_test : np.ndarray
        Train and test targets.
    task : str
        Task type ("classification" or "regression").
    time_budget : int
        Time budget in seconds.
    label : str
        Label for this run.
    n_classes : int
        Number of classes for classification.

    Returns
    -------
    results : dict
        Dictionary with metrics and timing.
    """
    from flaml import AutoML

    print(f"\n--- {label} FLAML (time_budget={time_budget}s) ---")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Task: {task}")

    automl = AutoML()

    flaml_task = "classification" if task == "classification" else "regression"

    start_time = time.time()
    automl.fit(
        X_train,
        y_train,
        task=flaml_task,
        time_budget=time_budget,
        seed=42,
        verbose=0,
        force_cancel=True,
    )
    train_time = time.time() - start_time

    # Predictions
    y_pred = automl.predict(X_test)

    # Calculate metrics based on task
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

        # ROC-AUC for binary classification
        if n_classes == 2:
            try:
                y_prob = automl.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
                results["roc_auc"] = roc_auc
            except Exception:
                results["roc_auc"] = None

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        if results.get("roc_auc"):
            print(f"  ROC-AUC: {results['roc_auc']:.4f}")

    else:  # regression
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.update(
            {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")

    print(f"  Best model: {automl.best_estimator}")
    print(f"  Train time: {train_time:.1f}s")

    return results


def run_single_dataset_benchmark(
    dataset_name: str,
    loader_func,
    time_budget: int = DEFAULT_TIME_BUDGET,
    max_features: int = MAX_FEATURES,
) -> Optional[dict]:
    """
    Run benchmark on a single dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    loader_func : callable
        Function to load the dataset.
    time_budget : int
        Time budget for FLAML.
    max_features : int
        Maximum features from FeatCopilot.

    Returns
    -------
    results : dict or None
        Benchmark results or None if failed.
    """
    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)

    try:
        # Load dataset
        X, y, task, name = loader_func()
        print(f"Loaded: {name}")
        print(f"Shape: {X.shape}")
        print(f"Task: {task}")

        # Limit dataset size for reasonable benchmark time
        if len(X) > 50000:
            print(f"Sampling 50000 rows from {len(X)} for benchmark...")
            idx = np.random.RandomState(42).choice(len(X), 50000, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)

        # Preprocess
        X_processed, y_processed, label_encoder = preprocess_data(X, y, task)

        # Determine number of classes
        n_classes = len(np.unique(y_processed)) if task == "classification" else 0

        # Train/test split
        stratify = y_processed if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=stratify
        )

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        if task == "classification":
            print(f"Classes: {n_classes}")

        # Run baseline FLAML
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
        print("\n--- Applying FeatCopilot feature engineering ---")
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

        # Handle NaN in engineered features
        for col in X_train_fe.columns:
            if X_train_fe[col].dtype == "object":
                X_train_fe[col] = X_train_fe[col].fillna("missing")
                X_test_fe[col] = X_test_fe[col].fillna("missing")
            else:
                median_val = X_train_fe[col].median()
                X_train_fe[col] = X_train_fe[col].fillna(median_val)
                X_test_fe[col] = X_test_fe[col].fillna(median_val)

        print(f"  Features: {X_train.shape[1]} -> {X_train_fe.shape[1]}")
        print(f"  FE Time: {fe_time:.2f}s")

        # Run FLAML with FeatCopilot features
        featcopilot_results = run_flaml_benchmark(
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
            fc_metric = featcopilot_results["f1_weighted"]
            metric_name = "F1 (weighted)"
        else:
            baseline_metric = baseline_results["r2"]
            fc_metric = featcopilot_results["r2"]
            metric_name = "R²"

        if baseline_metric != 0:
            improvement = (fc_metric - baseline_metric) / abs(baseline_metric) * 100
        else:
            improvement = 0

        return {
            "dataset": dataset_name,
            "name": name,
            "task": task,
            "n_samples": len(X),
            "n_features_original": X_train.shape[1],
            "n_features_fe": X_train_fe.shape[1],
            "baseline": baseline_results,
            "featcopilot": featcopilot_results,
            "fe_time": fe_time,
            "metric_name": metric_name,
            "baseline_metric": baseline_metric,
            "fc_metric": fc_metric,
            "improvement_pct": improvement,
        }

    except Exception as e:
        print(f"ERROR: Failed to benchmark {dataset_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_summary_report(all_results: list[dict], time_budget: int) -> str:
    """Generate markdown summary report."""
    report = []
    report.append("# FLAML Real-World Datasets Benchmark Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Time Budget:** {time_budget}s per FLAML run\n\n")

    # Summary table
    report.append("## Summary\n\n")
    report.append("| Dataset | Task | Samples | Features | Baseline | +FeatCopilot | Improvement | FE Time |\n")
    report.append("|---------|------|---------|----------|----------|--------------|-------------|--------|\n")

    improvements = []
    for r in all_results:
        if r is None:
            continue

        baseline_str = f"{r['baseline_metric']:.4f}"
        fc_str = f"{r['fc_metric']:.4f}"
        imp_str = f"{r['improvement_pct']:+.2f}%"
        fe_features = f"{r['n_features_original']}→{r['n_features_fe']}"

        report.append(
            f"| {r['dataset']} | {r['task'][:5]} | {r['n_samples']:,} | {fe_features} | "
            f"{baseline_str} | {fc_str} | {imp_str} | {r['fe_time']:.1f}s |\n"
        )
        improvements.append(r["improvement_pct"])

    # Overall statistics
    if improvements:
        report.append("\n## Overall Statistics\n\n")
        positive_improvements = [i for i in improvements if i > 0]
        negative_improvements = [i for i in improvements if i < 0]

        report.append(f"- **Datasets tested:** {len(improvements)}\n")
        report.append(f"- **FeatCopilot improved:** {len(positive_improvements)} datasets\n")
        report.append(f"- **Baseline better:** {len(negative_improvements)} datasets\n")
        report.append(f"- **Average improvement:** {np.mean(improvements):+.2f}%\n")
        report.append(f"- **Max improvement:** {max(improvements):+.2f}%\n")
        report.append(f"- **Min improvement:** {min(improvements):+.2f}%\n")

    # Detailed results
    report.append("\n## Detailed Results\n")

    for r in all_results:
        if r is None:
            continue

        report.append(f"\n### {r['name']}\n")
        report.append(f"- **Task:** {r['task']}\n")
        report.append(f"- **Samples:** {r['n_samples']:,}\n")
        report.append(f"- **Original features:** {r['n_features_original']}\n")
        report.append(f"- **Engineered features:** {r['n_features_fe']}\n")
        report.append(f"- **Feature engineering time:** {r['fe_time']:.1f}s\n\n")

        report.append("| Metric | Baseline | FeatCopilot |\n")
        report.append("|--------|----------|-------------|\n")

        if r["task"] == "classification":
            report.append(f"| Accuracy | {r['baseline']['accuracy']:.4f} | {r['featcopilot']['accuracy']:.4f} |\n")
            report.append(f"| F1 (macro) | {r['baseline']['f1_macro']:.4f} | {r['featcopilot']['f1_macro']:.4f} |\n")
            report.append(
                f"| F1 (weighted) | {r['baseline']['f1_weighted']:.4f} | {r['featcopilot']['f1_weighted']:.4f} |\n"
            )
            if r["baseline"].get("roc_auc") and r["featcopilot"].get("roc_auc"):
                report.append(f"| ROC-AUC | {r['baseline']['roc_auc']:.4f} | {r['featcopilot']['roc_auc']:.4f} |\n")
        else:
            report.append(f"| RMSE | {r['baseline']['rmse']:.4f} | {r['featcopilot']['rmse']:.4f} |\n")
            report.append(f"| MAE | {r['baseline']['mae']:.4f} | {r['featcopilot']['mae']:.4f} |\n")
            report.append(f"| R² | {r['baseline']['r2']:.4f} | {r['featcopilot']['r2']:.4f} |\n")

        report.append(f"| Train Time | {r['baseline']['train_time']:.1f}s | {r['featcopilot']['train_time']:.1f}s |\n")
        report.append(f"| Best Model | {r['baseline']['best_model']} | {r['featcopilot']['best_model']} |\n")

    return "".join(report)


def main():
    """Run the FLAML real-world datasets benchmark."""
    parser = argparse.ArgumentParser(description="FLAML Benchmark for Real-World Datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to run (default: quick benchmark)",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=DEFAULT_TIME_BUDGET,
        help=f"Time budget per FLAML run in seconds (default: {DEFAULT_TIME_BUDGET})",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=MAX_FEATURES,
        help=f"Maximum features from FeatCopilot (default: {MAX_FEATURES})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all datasets (not just quick benchmark)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/automl/FLAML_REALWORLD_BENCHMARK_REPORT.md",
        help="Output report path",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FLAML Benchmark - Real-World Datasets")
    print("=" * 70)

    # Check FLAML availability
    try:
        import flaml

        print(f"FLAML version: {flaml.__version__}")
    except ImportError:
        print("ERROR: FLAML not installed. Run: pip install flaml")
        sys.exit(1)

    # Determine datasets to run
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        # Validate dataset names
        invalid = [d for d in dataset_names if d not in DATASET_REGISTRY]
        if invalid:
            print(f"ERROR: Unknown datasets: {invalid}")
            print(f"Available: {list(DATASET_REGISTRY.keys())}")
            sys.exit(1)
    elif args.all:
        dataset_names = FULL_DATASETS
    else:
        dataset_names = QUICK_DATASETS

    print(f"Time budget: {args.time_budget}s per run")
    print(f"Max features: {args.max_features}")
    print(f"Datasets: {dataset_names}")

    # Run benchmarks
    all_results = []
    total_start = time.time()

    for dataset_name in dataset_names:
        loader_func = DATASET_REGISTRY[dataset_name]
        result = run_single_dataset_benchmark(
            dataset_name=dataset_name,
            loader_func=loader_func,
            time_budget=args.time_budget,
            max_features=args.max_features,
        )
        all_results.append(result)

    total_time = time.time() - total_start

    # Generate and print summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time / 60:.1f} minutes")

    # Quick summary
    successful = [r for r in all_results if r is not None]
    improved = [r for r in successful if r["improvement_pct"] > 0]

    print(f"\nDatasets tested: {len(successful)}/{len(dataset_names)}")
    print(f"FeatCopilot improved: {len(improved)}/{len(successful)}")

    if successful:
        improvements = [r["improvement_pct"] for r in successful]
        print(f"Average improvement: {np.mean(improvements):+.2f}%")

    # Generate report
    report = generate_summary_report(all_results, args.time_budget)
    print("\n" + report)

    # Save report
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {args.output}")

    return all_results


if __name__ == "__main__":
    results = main()

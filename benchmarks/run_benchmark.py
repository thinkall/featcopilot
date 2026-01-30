"""
Benchmark runner for FeatCopilot evaluation.

Compares model performance:
1. Baseline (no feature engineering)
2. With FeatCopilot feature engineering
"""

# ruff: noqa: E402

import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")

from benchmarks.datasets import get_all_datasets, get_timeseries_datasets  # noqa: E402
from featcopilot import AutoFeatureEngineer  # noqa: E402

warnings.filterwarnings("ignore")


def evaluate_classification(y_true, y_pred, y_prob=None):
    """Evaluate classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def evaluate_regression(y_true, y_pred):
    """Evaluate regression metrics."""
    return {
        "r2_score": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def get_models(task: str):
    """Get models for benchmarking."""
    if "classification" in task:
        return [
            ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
            ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]
    else:
        return [
            ("Ridge", Ridge(random_state=42)),
            ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]


def run_benchmark_single(X, y, task: str, dataset_name: str, verbose: bool = True) -> dict:
    """Run benchmark on a single dataset."""
    results = {
        "dataset": dataset_name,
        "task": task,
        "n_samples": len(X),
        "n_features_original": X.shape[1],
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data for baseline (convert to numpy to avoid feature name issues)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Task: {task} | Samples: {len(X)} | Features: {X.shape[1]}")
        print("=" * 70)

    # Baseline results
    baseline_results = {}
    if verbose:
        print("\n--- Baseline (no feature engineering) ---")

    for model_name, model in get_models(task):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if "classification" in task:
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = evaluate_classification(y_test, y_pred, y_prob)
            baseline_results[model_name] = metrics
            if verbose:
                auc_str = f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") else "N/A"
                print(f"  {model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={auc_str}")
        else:
            metrics = evaluate_regression(y_test, y_pred)
            baseline_results[model_name] = metrics
            if verbose:
                print(
                    f"  {model_name}: R2={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}"
                )

    results["baseline"] = baseline_results

    # FeatCopilot results
    if verbose:
        print("\n--- With FeatCopilot ---")

    try:
        # Determine optimal settings based on task type
        is_regression = "regression" in task

        # For regression: be more conservative, prioritize keeping original features
        # For classification: be more aggressive with feature generation
        if is_regression:
            selection_methods = ["mutual_info"]
            max_features = max(X.shape[1], min(40, X.shape[1] * 3))  # At least keep original count
            correlation_threshold = 0.85  # More aggressive redundancy removal
        else:
            selection_methods = ["importance", "mutual_info"]
            max_features = min(50, X.shape[1] * 4)
            correlation_threshold = 0.95

        # Apply feature engineering
        start = time.time()
        engineer = AutoFeatureEngineer(
            engines=["tabular"],
            max_features=max_features,
            selection_methods=selection_methods,
            correlation_threshold=correlation_threshold,
            verbose=False,
        )
        X_train_fe = engineer.fit_transform(X_train, y_train)
        fe_time = time.time() - start
        X_test_fe = engineer.transform(X_test)

        # For regression, ensure we include original features if they're not in the output
        if is_regression:
            for col in X_train.columns:
                if col not in X_train_fe.columns:
                    X_train_fe[col] = X_train[col].values
                    X_test_fe[col] = X_test[col].values

        # Handle any NaN values
        X_train_fe = X_train_fe.fillna(0)
        X_test_fe = X_test_fe.fillna(0)

        results["n_features_engineered"] = X_train_fe.shape[1]
        results["fe_time"] = fe_time

        if verbose:
            print(f"  Features: {X.shape[1]} -> {X_train_fe.shape[1]} (generated in {fe_time:.2f}s)")

        # Scale engineered features (convert to numpy)
        scaler_fe = StandardScaler()
        X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe.values)
        X_test_fe_scaled = scaler_fe.transform(X_test_fe.values)

        featcopilot_results = {}
        for model_name, model in get_models(task):
            model.fit(X_train_fe_scaled, y_train)
            y_pred = model.predict(X_test_fe_scaled)

            if "classification" in task:
                y_prob = model.predict_proba(X_test_fe_scaled)[:, 1] if hasattr(model, "predict_proba") else None
                metrics = evaluate_classification(y_test, y_pred, y_prob)
                featcopilot_results[model_name] = metrics

                # Calculate improvement
                baseline_acc = baseline_results[model_name]["accuracy"]
                improvement = (metrics["accuracy"] - baseline_acc) / baseline_acc * 100
                auc_str = f"{metrics['roc_auc']:.4f}" if metrics.get("roc_auc") else "N/A"
                if verbose:
                    print(
                        f"  {model_name}: Acc={metrics['accuracy']:.4f} ({improvement:+.2f}%), F1={metrics['f1_score']:.4f}, AUC={auc_str}"
                    )
            else:
                metrics = evaluate_regression(y_test, y_pred)
                featcopilot_results[model_name] = metrics

                # Calculate improvement
                baseline_r2 = baseline_results[model_name]["r2_score"]
                if baseline_r2 > 0:
                    improvement = (metrics["r2_score"] - baseline_r2) / baseline_r2 * 100
                else:
                    improvement = (metrics["r2_score"] - baseline_r2) * 100
                if verbose:
                    print(
                        f"  {model_name}: R2={metrics['r2_score']:.4f} ({improvement:+.2f}%), RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}"
                    )

        results["featcopilot"] = featcopilot_results

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        results["featcopilot"] = None
        results["error"] = str(e)

    return results


def run_all_benchmarks(verbose: bool = True, include_timeseries: bool = True) -> list[dict]:
    """Run benchmarks on all datasets."""
    all_results = []

    # Standard datasets
    for loader in get_all_datasets():
        X, y, task, name = loader()
        result = run_benchmark_single(X, y, task, name, verbose=verbose)
        all_results.append(result)

    # Time series datasets
    if include_timeseries:
        if verbose:
            print("\n" + "=" * 70)
            print("TIME SERIES BENCHMARKS")
            print("=" * 70)

        for loader in get_timeseries_datasets():
            X, y, task, name = loader()
            result = run_benchmark_single(X, y, task, name, verbose=verbose)
            all_results.append(result)

    return all_results


def generate_report(results: list[dict]) -> str:
    """Generate a markdown benchmark report."""
    report = []
    report.append("# FeatCopilot Benchmark Report\n")
    report.append("## Summary\n")

    # Calculate overall improvements by task type
    classification_improvements = []
    regression_improvements = []
    ts_classification_improvements = []
    ts_regression_improvements = []

    for r in results:
        if r.get("featcopilot") is None:
            continue

        task = r["task"]
        for model_name in r["baseline"]:
            baseline = r["baseline"][model_name]
            featcopilot = r["featcopilot"][model_name]

            if "classification" in task:
                improvement = (featcopilot["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100
                if "timeseries" in task:
                    ts_classification_improvements.append(improvement)
                else:
                    classification_improvements.append(improvement)
            else:
                if baseline["r2_score"] > 0:
                    improvement = (featcopilot["r2_score"] - baseline["r2_score"]) / baseline["r2_score"] * 100
                else:
                    improvement = featcopilot["r2_score"] - baseline["r2_score"]
                if "timeseries" in task:
                    ts_regression_improvements.append(improvement)
                else:
                    regression_improvements.append(improvement)

    if classification_improvements:
        wins = sum(1 for x in classification_improvements if x > 0)
        report.append("**Classification Tasks:**")
        report.append(f"- Average Accuracy Improvement: **{np.mean(classification_improvements):+.2f}%**")
        report.append(f"- Max Improvement: {np.max(classification_improvements):+.2f}%")
        report.append(
            f"- Improvements > 0: {wins}/{len(classification_improvements)} ({100*wins/len(classification_improvements):.0f}%)\n"
        )

    if regression_improvements:
        wins = sum(1 for x in regression_improvements if x > 0)
        report.append("**Regression Tasks:**")
        report.append(f"- Average R² Improvement: **{np.mean(regression_improvements):+.2f}%**")
        report.append(f"- Max Improvement: {np.max(regression_improvements):+.2f}%")
        report.append(
            f"- Improvements > 0: {wins}/{len(regression_improvements)} ({100*wins/len(regression_improvements):.0f}%)\n"
        )

    if ts_classification_improvements or ts_regression_improvements:
        report.append("**Time Series Tasks:**")
        if ts_classification_improvements:
            wins = sum(1 for x in ts_classification_improvements if x > 0)
            report.append(
                f"- Classification Avg Improvement: **{np.mean(ts_classification_improvements):+.2f}%** ({wins}/{len(ts_classification_improvements)} wins)"
            )
        if ts_regression_improvements:
            wins = sum(1 for x in ts_regression_improvements if x > 0)
            report.append(
                f"- Regression Avg R² Improvement: **{np.mean(ts_regression_improvements):+.2f}%** ({wins}/{len(ts_regression_improvements)} wins)"
            )
        report.append("")

    # Detailed results table - Classification
    report.append("## Detailed Results\n")
    report.append("### Classification Datasets\n")
    report.append("| Dataset | Model | Baseline Acc | FeatCopilot Acc | Improvement | Baseline F1 | FeatCopilot F1 |")
    report.append("|---------|-------|--------------|-----------------|-------------|-------------|----------------|")

    for r in results:
        if "classification" not in r["task"] or r.get("featcopilot") is None:
            continue

        for model_name in r["baseline"]:
            baseline = r["baseline"][model_name]
            featcopilot = r["featcopilot"][model_name]
            improvement = (featcopilot["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100
            report.append(
                f"| {r['dataset']} | {model_name} | {baseline['accuracy']:.4f} | {featcopilot['accuracy']:.4f} | {improvement:+.2f}% | {baseline['f1_score']:.4f} | {featcopilot['f1_score']:.4f} |"
            )

    # Regression results
    report.append("\n### Regression Datasets\n")
    report.append("| Dataset | Model | Baseline R² | FeatCopilot R² | Improvement | Baseline RMSE | FeatCopilot RMSE |")
    report.append("|---------|-------|-------------|----------------|-------------|---------------|------------------|")

    for r in results:
        if "regression" not in r["task"] or r.get("featcopilot") is None:
            continue

        for model_name in r["baseline"]:
            baseline = r["baseline"][model_name]
            featcopilot = r["featcopilot"][model_name]
            if baseline["r2_score"] > 0:
                improvement = (featcopilot["r2_score"] - baseline["r2_score"]) / baseline["r2_score"] * 100
            else:
                improvement = featcopilot["r2_score"] - baseline["r2_score"]
            report.append(
                f"| {r['dataset']} | {model_name} | {baseline['r2_score']:.4f} | {featcopilot['r2_score']:.4f} | {improvement:+.2f}% | {baseline['rmse']:.2f} | {featcopilot['rmse']:.2f} |"
            )

    # Feature engineering stats
    report.append("\n## Feature Engineering Statistics\n")
    report.append("| Dataset | Original Features | Engineered Features | FE Time (s) |")
    report.append("|---------|-------------------|---------------------|-------------|")

    for r in results:
        if r.get("n_features_engineered"):
            report.append(
                f"| {r['dataset']} | {r['n_features_original']} | {r['n_features_engineered']} | {r.get('fe_time', 0):.2f} |"
            )

    report.append("\n## Methodology\n")
    report.append("- **Train/Test Split**: 80/20 with random_state=42")
    report.append("- **Feature Engineering**: FeatCopilot TabularEngine")
    report.append("  - Classification: importance + mutual_info selection, max 50 features")
    report.append("  - Regression: mutual_info selection, max 30 features, correlation_threshold=0.90")
    report.append("- **Preprocessing**: StandardScaler applied to all features")
    report.append("- **Models**: LogisticRegression/Ridge, RandomForest, GradientBoosting")
    report.append("- **Metrics**: Accuracy/R², F1-score/RMSE, ROC-AUC/MAE")

    report.append("\n## Datasets\n")
    report.append("### Real-world Datasets")
    report.append("- Diabetes (sklearn) - Medical regression")
    report.append("- Breast Cancer (sklearn) - Medical classification")
    report.append("- Titanic (Kaggle-style) - Survival classification")
    report.append("- House Prices (Kaggle-style) - Price regression")
    report.append("- Credit Card Fraud (Kaggle-style) - Imbalanced classification")
    report.append("- Bike Sharing (Kaggle-style) - Demand regression")
    report.append("- Employee Attrition (IBM HR) - HR classification")
    report.append("\n### Synthetic Datasets")
    report.append("- Credit Risk - Financial classification")
    report.append("- Medical Diagnosis - Healthcare classification")
    report.append("- Complex Regression - Non-linear regression")
    report.append("- Complex Classification - Imbalanced classification")
    report.append("\n### Time Series Datasets")
    report.append("- Energy Consumption - Forecasting regression")
    report.append("- Stock Price Direction - Movement classification")
    report.append("- Website Traffic - Traffic regression")

    return "\n".join(report)


if __name__ == "__main__":
    print("=" * 70)
    print("FeatCopilot Benchmark Suite")
    print("=" * 70)

    results = run_all_benchmarks(verbose=True, include_timeseries=True)

    # Generate and save report
    report = generate_report(results)

    with open("benchmarks/BENCHMARK_REPORT.md", "w") as f:
        f.write(report)

    print("\n" + "=" * 70)
    print("Benchmark complete! Report saved to benchmarks/BENCHMARK_REPORT.md")
    print("=" * 70)

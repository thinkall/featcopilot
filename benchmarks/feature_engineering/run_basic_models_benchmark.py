"""
Basic Models Benchmark for FeatCopilot Feature Engineering.

Compares baseline models vs models with FeatCopilot-enhanced features
using simple models (RandomForest, XGBoost, LogisticRegression, Ridge).

This benchmark focuses on demonstrating the value of feature engineering
rather than AutoML hyperparameter tuning.

Usage:
    python benchmarks/feature_engineering/run_basic_models_benchmark.py [options]

Options:
    --datasets NAMES     Comma-separated dataset names (default: quick set)
    --engines ENGINES    Comma-separated FeatCopilot engines (default: tabular)
    --all               Run on all datasets
    --quick             Run quick benchmark (4 datasets)
    --with-llm          Include LLM engine (requires API key)
    --max-features N    Max features to generate (default: 100)
    --output PATH       Output report path
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

sys.path.insert(0, ".")

warnings.filterwarnings("ignore")


# Dataset configurations
DATASETS = {
    # From datasets.py - synthetic but representative
    "titanic": ("load_titanic_dataset", "Titanic survival prediction"),
    "house_prices": ("load_house_prices_dataset", "House price prediction"),
    "credit_risk": ("create_credit_risk_dataset", "Credit risk classification"),
    "customer_churn": ("create_customer_churn_dataset", "Customer churn prediction"),
    "insurance": ("create_insurance_claims_dataset", "Insurance claim prediction"),
    # Real-world datasets
    "kaggle_house_prices": ("load_kaggle_house_prices", "Kaggle House Prices"),
    "employee_attrition": ("load_kaggle_employee_attrition", "IBM HR Attrition"),
    "telco_churn": ("load_kaggle_telco_churn", "Telco Customer Churn"),
    "adult_census": ("load_openml_adult_census", "Adult Census Income"),
    "spaceship_titanic": ("load_kaggle_spaceship_titanic", "Spaceship Titanic"),
    "bike_sharing": ("load_kaggle_bike_sharing", "Bike Sharing Demand"),
    "medical_cost": ("load_kaggle_medical_cost", "Medical Cost Prediction"),
    "wine_quality": ("load_openml_wine_quality", "Wine Quality"),
    "life_expectancy": ("load_kaggle_life_expectancy", "Life Expectancy"),
}

QUICK_DATASETS = ["titanic", "house_prices", "medical_cost", "wine_quality"]
MEDIUM_DATASETS = QUICK_DATASETS + ["employee_attrition", "bike_sharing", "credit_risk", "customer_churn"]
ALL_DATASETS = list(DATASETS.keys())


def load_dataset(name: str) -> tuple:
    """Load a dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    loader_name, description = DATASETS[name]

    # Import from datasets module
    from benchmarks import datasets

    loader = getattr(datasets, loader_name)
    result = loader()

    if len(result) == 4:
        X, y, task_type, _ = result
    else:
        X, y, task_type = result

    return X, y, task_type, description


def preprocess_data(X: pd.DataFrame, y, task: str) -> tuple:
    """Preprocess data for modeling."""
    X_processed = X.copy()

    # Handle categorical columns
    for col in X_processed.columns:
        if X_processed[col].dtype == "object" or X_processed[col].dtype.name == "category":
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str).fillna("missing"))
        else:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    # Handle target
    if task == "classification":
        if hasattr(y, "dtype") and (y.dtype == "object" or y.dtype.name == "category"):
            le = LabelEncoder()
            y_processed = le.fit_transform(y.astype(str))
        else:
            y_processed = np.array(y)
    else:
        y_processed = np.array(y).astype(float)

    # Replace infinities
    X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
    for col in X_processed.columns:
        if X_processed[col].isna().any():
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    return X_processed, y_processed


def get_models(task: str) -> dict:
    """Get models for the task type."""
    if task == "classification":
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        }
    else:
        return {
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            "Ridge": Ridge(alpha=1.0, random_state=42),
        }


def evaluate_model(model, X_train, X_test, y_train, y_test, task: str) -> dict:
    """Evaluate a model and return metrics."""
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)

    results = {"train_time": train_time}

    if task == "classification":
        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["f1_weighted"] = f1_score(y_test, y_pred, average="weighted")
        results["f1_macro"] = f1_score(y_test, y_pred, average="macro")

        n_classes = len(np.unique(y_test))
        if n_classes == 2 and hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                results["roc_auc"] = roc_auc_score(y_test, y_prob)
            except Exception:
                results["roc_auc"] = None
    else:
        results["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        results["mae"] = mean_absolute_error(y_test, y_pred)
        results["r2"] = r2_score(y_test, y_pred)

    return results


def apply_featcopilot(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    engines: list[str],
    max_features: int,
    task_description: str = "",
    column_descriptions: Optional[dict] = None,
) -> tuple:
    """Apply FeatCopilot feature engineering."""
    from featcopilot import AutoFeatureEngineer

    # Configure LLM if needed
    llm_config = {}
    if "llm" in engines:
        llm_config = {
            "model": "gpt-5.2",
            "max_suggestions": 30,
            "domain": "general",
            "backend": "copilot",
        }

    engineer = AutoFeatureEngineer(
        engines=engines,
        max_features=max_features,
        llm_config=llm_config,
        verbose=True,
    )

    # Fit and transform
    X_train_fe = engineer.fit_transform(
        X_train,
        y_train,
        column_descriptions=column_descriptions,
        task_description=task_description,
    )
    X_test_fe = engineer.transform(X_test)

    # Align columns
    common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
    X_train_fe = X_train_fe[common_cols].copy()
    X_test_fe = X_test_fe[common_cols].copy()

    # Handle NaN/inf
    X_train_fe = X_train_fe.replace([np.inf, -np.inf], np.nan)
    X_test_fe = X_test_fe.replace([np.inf, -np.inf], np.nan)

    for col in X_train_fe.columns:
        if X_train_fe[col].dtype == "object":
            X_train_fe[col] = X_train_fe[col].fillna("missing")
            X_test_fe[col] = X_test_fe[col].fillna("missing")
        else:
            med = X_train_fe[col].median()
            X_train_fe[col] = X_train_fe[col].fillna(med)
            X_test_fe[col] = X_test_fe[col].fillna(med)

    return X_train_fe, X_test_fe, engineer


def run_single_benchmark(
    dataset_name: str,
    engines: list[str] = None,
    max_features: int = 100,
) -> Optional[dict]:
    """Run benchmark on a single dataset."""
    engines = engines or ["tabular"]

    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Engines: {engines}")
    print("=" * 70)

    try:
        # Load dataset
        X, y, task, description = load_dataset(dataset_name)
        print(f"Description: {description}")
        print(f"Shape: {X.shape}")
        print(f"Task: {task}")

        # Preprocess
        X_processed, y_processed = preprocess_data(X, y, task)

        # Train/test split
        stratify = y_processed if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=stratify
        )

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Get models
        models = get_models(task)

        # Baseline results
        print("\n--- Baseline Models ---")
        baseline_results = {}
        for model_name, model in models.items():
            results = evaluate_model(model, X_train, X_test, y_train, y_test, task)
            baseline_results[model_name] = results

            if task == "classification":
                print(f"  {model_name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_weighted']:.4f}")
            else:
                print(f"  {model_name}: R²={results['r2']:.4f}, RMSE={results['rmse']:.4f}")

        # Apply FeatCopilot
        print(f"\n--- Applying FeatCopilot ({', '.join(engines)}) ---")
        fe_start = time.time()

        try:
            X_train_fe, X_test_fe, engineer = apply_featcopilot(
                X_train,
                X_test,
                y_train,
                engines=engines,
                max_features=max_features,
                task_description=f"{'Classify' if task == 'classification' else 'Predict'} {description}",
            )
            fe_time = time.time() - fe_start

            print(f"  Features: {X_train.shape[1]} -> {X_train_fe.shape[1]}")
            print(f"  FE Time: {fe_time:.2f}s")

            # FeatCopilot results
            print("\n--- FeatCopilot Models ---")
            fe_results = {}
            models = get_models(task)  # Fresh models
            for model_name, model in models.items():
                results = evaluate_model(model, X_train_fe, X_test_fe, y_train, y_test, task)
                fe_results[model_name] = results

                if task == "classification":
                    print(f"  {model_name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_weighted']:.4f}")
                else:
                    print(f"  {model_name}: R²={results['r2']:.4f}, RMSE={results['rmse']:.4f}")

        except Exception as e:
            print(f"  FeatCopilot ERROR: {e}")
            fe_results = None
            fe_time = 0
            X_train_fe = X_train

        # Calculate improvements
        improvements = {}
        if fe_results:
            for model_name in baseline_results:
                if task == "classification":
                    base = baseline_results[model_name]["f1_weighted"]
                    fc = fe_results[model_name]["f1_weighted"]
                else:
                    base = baseline_results[model_name]["r2"]
                    fc = fe_results[model_name]["r2"]

                if base != 0:
                    imp = ((fc - base) / abs(base)) * 100
                else:
                    imp = 0
                improvements[model_name] = imp

            print("\n--- Improvements ---")
            for model_name, imp in improvements.items():
                print(f"  {model_name}: {imp:+.2f}%")

        return {
            "dataset": dataset_name,
            "description": description,
            "task": task,
            "n_samples": len(X),
            "n_features_original": X_train.shape[1],
            "n_features_fe": X_train_fe.shape[1] if fe_results else X_train.shape[1],
            "engines": engines,
            "baseline": baseline_results,
            "featcopilot": fe_results,
            "improvements": improvements,
            "fe_time": fe_time,
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_report(results: list[dict], engines: list[str]) -> str:
    """Generate markdown report."""
    report = []
    report.append("# FeatCopilot Basic Models Benchmark Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Engines:** {', '.join(engines)}\n\n")

    # Summary table
    report.append("## Summary\n\n")
    report.append("| Dataset | Task | Samples | Features | Best Baseline | Best +FC | Improvement |\n")
    report.append("|---------|------|---------|----------|---------------|----------|-------------|\n")

    all_improvements = []
    for r in results:
        if r is None or not r.get("featcopilot"):
            continue

        # Find best model improvement
        best_imp = max(r["improvements"].values()) if r["improvements"] else 0

        # Best baseline metric
        if r["task"] == "classification":
            best_baseline = max(r["baseline"][m]["f1_weighted"] for m in r["baseline"])
            best_fc = max(r["featcopilot"][m]["f1_weighted"] for m in r["featcopilot"])
        else:
            best_baseline = max(r["baseline"][m]["r2"] for m in r["baseline"])
            best_fc = max(r["featcopilot"][m]["r2"] for m in r["featcopilot"])

        fe_str = f"{r['n_features_original']}→{r['n_features_fe']}"

        report.append(
            f"| {r['dataset']} | {r['task'][:5]} | {r['n_samples']:,} | {fe_str} | "
            f"{best_baseline:.4f} | {best_fc:.4f} | {best_imp:+.2f}% |\n"
        )
        all_improvements.append(best_imp)

    # Statistics
    if all_improvements:
        report.append("\n## Overall Statistics\n\n")
        positive = [i for i in all_improvements if i > 0]
        report.append(f"- **Datasets tested:** {len(all_improvements)}\n")
        report.append(
            f"- **FeatCopilot improved:** {len(positive)} datasets ({len(positive)/len(all_improvements)*100:.0f}%)\n"
        )
        report.append(f"- **Average improvement:** {np.mean(all_improvements):+.2f}%\n")
        report.append(f"- **Max improvement:** {max(all_improvements):+.2f}%\n")
        report.append(f"- **Min improvement:** {min(all_improvements):+.2f}%\n")

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

        if r["task"] == "classification":
            report.append("| Model | Baseline Acc | Baseline F1 | +FC Acc | +FC F1 | Improvement |\n")
            report.append("|-------|-------------|-------------|---------|--------|-------------|\n")
            for model_name in r["baseline"]:
                base = r["baseline"][model_name]
                fc = r["featcopilot"][model_name] if r["featcopilot"] else base
                imp = r["improvements"].get(model_name, 0)
                report.append(
                    f"| {model_name} | {base['accuracy']:.4f} | {base['f1_weighted']:.4f} | "
                    f"{fc['accuracy']:.4f} | {fc['f1_weighted']:.4f} | {imp:+.2f}% |\n"
                )
        else:
            report.append("| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |\n")
            report.append("|-------|------------|---------------|--------|----------|-------------|\n")
            for model_name in r["baseline"]:
                base = r["baseline"][model_name]
                fc = r["featcopilot"][model_name] if r["featcopilot"] else base
                imp = r["improvements"].get(model_name, 0)
                report.append(
                    f"| {model_name} | {base['r2']:.4f} | {base['rmse']:.4f} | "
                    f"{fc['r2']:.4f} | {fc['rmse']:.4f} | {imp:+.2f}% |\n"
                )

    return "".join(report)


def main():
    parser = argparse.ArgumentParser(description="Basic Models Benchmark for FeatCopilot")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names")
    parser.add_argument("--engines", type=str, default="tabular", help="Comma-separated engines")
    parser.add_argument("--max-features", type=int, default=100, help="Max features to generate")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--medium", action="store_true", help="Run medium benchmark")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--with-llm", action="store_true", help="Include LLM engine")
    parser.add_argument("--output", type=str, default=None, help="Output report path")

    args = parser.parse_args()

    print("=" * 70)
    print("FeatCopilot Basic Models Benchmark")
    print("=" * 70)

    # Determine datasets
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    elif args.all:
        dataset_names = ALL_DATASETS
    elif args.medium:
        dataset_names = MEDIUM_DATASETS
    else:
        dataset_names = QUICK_DATASETS

    # Determine engines
    engines = [e.strip() for e in args.engines.split(",")]
    if args.with_llm and "llm" not in engines:
        engines.append("llm")

    print(f"Datasets ({len(dataset_names)}): {dataset_names}")
    print(f"Engines: {engines}")
    print(f"Max features: {args.max_features}")

    # Run benchmarks
    all_results = []
    total_start = time.time()

    for name in dataset_names:
        result = run_single_benchmark(name, engines=engines, max_features=args.max_features)
        all_results.append(result)

    total_time = time.time() - total_start

    # Generate report
    report = generate_report(all_results, engines)
    print("\n" + report)

    # Save report
    output_path = args.output or f"benchmarks/feature_engineering/BASIC_MODELS_BENCHMARK_{'_'.join(engines).upper()}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")
    print(f"Total time: {total_time / 60:.1f} minutes")

    return all_results


if __name__ == "__main__":
    main()

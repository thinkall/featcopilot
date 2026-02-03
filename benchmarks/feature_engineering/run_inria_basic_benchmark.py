"""
INRIA Tabular Benchmark with Basic Models.

Uses simple models (RandomForest, Ridge/LogisticRegression) on INRIA-SODA
benchmark datasets to demonstrate FeatCopilot's feature engineering value.

These are challenging real-world datasets where feature engineering
can make a significant difference.
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
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, ".")

warnings.filterwarnings("ignore")

# INRIA-SODA datasets that showed best FeatCopilot improvements
INRIA_DATASETS = {
    # Regression - best for FE
    "abalone": ("reg_num_abalone", "regression", "Abalone age prediction"),
    "wine_quality": ("reg_num_wine_quality", "regression", "Wine quality score"),
    "diamonds": ("reg_num_diamonds", "regression", "Diamond price prediction"),
    "cpu_act": ("reg_num_cpu_act", "regression", "CPU activity prediction"),
    "houses": ("reg_num_houses", "regression", "House value prediction"),
    "bike_sharing": ("reg_num_Bike_Sharing_Demand", "regression", "Bike rental demand"),
    "miami_housing": ("reg_num_MiamiHousing2016", "regression", "Miami housing prices"),
    "superconduct": ("reg_num_superconduct", "regression", "Superconductor temperature"),
    # Classification
    "credit": ("clf_num_credit", "classification", "Credit approval"),
    "eye_movements": ("clf_cat_eye_movements", "classification", "Eye movement classification"),
    "jannis": ("clf_num_jannis", "classification", "Multi-class classification"),
    "higgs": ("clf_num_Higgs", "classification", "Higgs boson detection"),
    "bioresponse": ("clf_num_Bioresponse", "classification", "Biological response"),
    "diabetes": ("clf_num_Diabetes130US", "classification", "Diabetes readmission"),
}

QUICK_DATASETS = ["abalone", "wine_quality", "diamonds", "credit", "eye_movements"]
MEDIUM_DATASETS = QUICK_DATASETS + ["cpu_act", "houses", "bike_sharing", "jannis", "bioresponse"]
MAX_SAMPLES = 30000


def load_inria_dataset(config_name: str, max_samples: int = MAX_SAMPLES) -> tuple:
    """Load a dataset from inria-soda/tabular-benchmark."""
    from datasets import load_dataset

    ds = load_dataset("inria-soda/tabular-benchmark", config_name, split="train")
    df = ds.to_pandas()

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if len(X) > max_samples:
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    return X, y


def preprocess_data(X: pd.DataFrame, y, task: str) -> tuple:
    """Preprocess data."""
    X_processed = X.copy()

    for col in X_processed.columns:
        if X_processed[col].dtype == "object" or X_processed[col].dtype.name == "category":
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str).fillna("missing"))
        else:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    if task == "classification":
        if hasattr(y, "dtype") and (y.dtype == "object" or y.dtype.name == "category"):
            le = LabelEncoder()
            y_processed = le.fit_transform(y.astype(str))
        else:
            y_processed = np.array(y)
    else:
        y_processed = np.array(y).astype(float)

    X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
    for col in X_processed.columns:
        if X_processed[col].isna().any():
            X_processed[col] = X_processed[col].fillna(0)

    return X_processed, y_processed


def get_models(task: str) -> dict:
    """Get models."""
    if task == "classification":
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        }
    else:
        return {
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "Ridge": Ridge(alpha=1.0),
        }


def evaluate_model(model, X_train, X_test, y_train, y_test, task: str) -> dict:
    """Evaluate model."""
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    results = {"train_time": train_time}

    if task == "classification":
        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["f1_weighted"] = f1_score(y_test, y_pred, average="weighted")
    else:
        results["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred))
        results["r2"] = r2_score(y_test, y_pred)

    return results


def apply_featcopilot(X_train, X_test, y_train, engines: list, max_features: int) -> tuple:
    """Apply FeatCopilot."""
    from featcopilot import AutoFeatureEngineer

    llm_config = {}
    if "llm" in engines:
        llm_config = {"model": "gpt-5.2", "max_suggestions": 30, "backend": "copilot"}

    engineer = AutoFeatureEngineer(
        engines=engines,
        max_features=max_features,
        llm_config=llm_config,
        verbose=True,
    )

    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)

    common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
    X_train_fe = X_train_fe[common_cols].copy()
    X_test_fe = X_test_fe[common_cols].copy()

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

    return X_train_fe, X_test_fe


def run_single_benchmark(dataset_name: str, engines: list, max_features: int) -> Optional[dict]:
    """Run benchmark on single dataset."""
    if dataset_name not in INRIA_DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        return None

    config_name, task, description = INRIA_DATASETS[dataset_name]

    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name} - {description}")
    print(f"Engines: {engines}")
    print("=" * 70)

    try:
        X, y = load_inria_dataset(config_name)
        print(f"Shape: {X.shape}")
        print(f"Task: {task}")

        X_processed, y_processed = preprocess_data(X, y, task)

        stratify = y_processed if task == "classification" and len(np.unique(y_processed)) < 50 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42, stratify=stratify
        )

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Baseline
        print("\n--- Baseline Models ---")
        models = get_models(task)
        baseline_results = {}
        for name, model in models.items():
            results = evaluate_model(model, X_train, X_test, y_train, y_test, task)
            baseline_results[name] = results
            if task == "classification":
                print(f"  {name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_weighted']:.4f}")
            else:
                print(f"  {name}: R²={results['r2']:.4f}, RMSE={results['rmse']:.4f}")

        # FeatCopilot
        print(f"\n--- FeatCopilot ({', '.join(engines)}) ---")
        fe_start = time.time()

        try:
            X_train_fe, X_test_fe = apply_featcopilot(X_train, X_test, y_train, engines, max_features)
            fe_time = time.time() - fe_start
            print(f"  Features: {X_train.shape[1]} -> {X_train_fe.shape[1]}")
            print(f"  FE Time: {fe_time:.1f}s")

            print("\n--- FeatCopilot Models ---")
            models = get_models(task)
            fe_results = {}
            for name, model in models.items():
                results = evaluate_model(model, X_train_fe, X_test_fe, y_train, y_test, task)
                fe_results[name] = results
                if task == "classification":
                    print(f"  {name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_weighted']:.4f}")
                else:
                    print(f"  {name}: R²={results['r2']:.4f}, RMSE={results['rmse']:.4f}")

        except Exception as e:
            print(f"  FeatCopilot ERROR: {e}")
            import traceback

            traceback.print_exc()
            fe_results = None
            fe_time = 0
            X_train_fe = X_train

        # Calculate improvements
        improvements = {}
        if fe_results:
            for name in baseline_results:
                if task == "classification":
                    base = baseline_results[name]["f1_weighted"]
                    fc = fe_results[name]["f1_weighted"]
                else:
                    base = baseline_results[name]["r2"]
                    fc = fe_results[name]["r2"]
                imp = ((fc - base) / abs(base) * 100) if base != 0 else 0
                improvements[name] = imp

            print("\n--- Improvements ---")
            for name, imp in improvements.items():
                print(f"  {name}: {imp:+.2f}%")

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


def generate_report(results: list, engines: list) -> str:
    """Generate markdown report."""
    report = []
    report.append("# INRIA Basic Models Benchmark Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Engines:** {', '.join(engines)}\n")
    report.append(f"**Max Samples:** {MAX_SAMPLES:,}\n")
    report.append("**Models:** RandomForest, Ridge (regression) / LogisticRegression (classification)\n\n")

    report.append("## Summary\n\n")
    report.append("| Dataset | Task | Samples | Features | Best Baseline | Best +FC | Improvement |\n")
    report.append("|---------|------|---------|----------|---------------|----------|-------------|\n")

    all_improvements = []
    for r in results:
        if r is None or not r.get("featcopilot"):
            continue

        # Find the best performing model for baseline and use the SAME model for comparison
        if r["task"] == "classification":
            # Get best baseline model name and its score
            best_base_model = max(r["baseline"], key=lambda m: r["baseline"][m]["f1_weighted"])
            best_base = r["baseline"][best_base_model]["f1_weighted"]
            best_fc = r["featcopilot"][best_base_model]["f1_weighted"]
        else:
            best_base_model = max(r["baseline"], key=lambda m: r["baseline"][m]["r2"])
            best_base = r["baseline"][best_base_model]["r2"]
            best_fc = r["featcopilot"][best_base_model]["r2"]

        # Calculate improvement using the same model
        best_imp = ((best_fc - best_base) / abs(best_base) * 100) if best_base != 0 else 0

        fe_str = f"{r['n_features_original']}→{r['n_features_fe']}"
        report.append(
            f"| {r['dataset']} | {r['task'][:5]} | {r['n_samples']:,} | {fe_str} | "
            f"{best_base:.4f} | {best_fc:.4f} | {best_imp:+.2f}% |\n"
        )
        all_improvements.append(best_imp)

    if all_improvements:
        report.append("\n## Overall Statistics\n\n")
        positive = [i for i in all_improvements if i > 0]
        report.append(f"- **Datasets tested:** {len(all_improvements)}\n")
        report.append(f"- **FeatCopilot improved:** {len(positive)} ({len(positive)/len(all_improvements)*100:.0f}%)\n")
        report.append(f"- **Average improvement:** {np.mean(all_improvements):+.2f}%\n")
        report.append(f"- **Max improvement:** {max(all_improvements):+.2f}%\n")
        report.append(f"- **Min improvement:** {min(all_improvements):+.2f}%\n")

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
            for name in r["baseline"]:
                base = r["baseline"][name]
                fc = r["featcopilot"][name] if r["featcopilot"] else base
                imp = r["improvements"].get(name, 0)
                report.append(
                    f"| {name} | {base['accuracy']:.4f} | {base['f1_weighted']:.4f} | "
                    f"{fc['accuracy']:.4f} | {fc['f1_weighted']:.4f} | {imp:+.2f}% |\n"
                )
        else:
            report.append("| Model | Baseline R² | Baseline RMSE | +FC R² | +FC RMSE | Improvement |\n")
            report.append("|-------|------------|---------------|--------|----------|-------------|\n")
            for name in r["baseline"]:
                base = r["baseline"][name]
                fc = r["featcopilot"][name] if r["featcopilot"] else base
                imp = r["improvements"].get(name, 0)
                report.append(
                    f"| {name} | {base['r2']:.4f} | {base['rmse']:.4f} | "
                    f"{fc['r2']:.4f} | {fc['rmse']:.4f} | {imp:+.2f}% |\n"
                )

    return "".join(report)


def main():
    parser = argparse.ArgumentParser(description="INRIA Basic Models Benchmark")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--engines", type=str, default="tabular")
    parser.add_argument("--max-features", type=int, default=150)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    print("=" * 70)
    print("INRIA Basic Models Benchmark")
    print("=" * 70)

    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
    elif args.all:
        dataset_names = list(INRIA_DATASETS.keys())
    elif args.medium:
        dataset_names = MEDIUM_DATASETS
    else:
        dataset_names = QUICK_DATASETS

    engines = [e.strip() for e in args.engines.split(",")]
    if args.with_llm and "llm" not in engines:
        engines.append("llm")

    print(f"Datasets ({len(dataset_names)}): {dataset_names}")
    print(f"Engines: {engines}")

    all_results = []
    total_start = time.time()

    for name in dataset_names:
        result = run_single_benchmark(name, engines, args.max_features)
        all_results.append(result)

    total_time = time.time() - total_start

    report = generate_report(all_results, engines)
    print("\n" + report)

    output_path = args.output or f"benchmarks/feature_engineering/INRIA_BASIC_MODELS_{'_'.join(engines).upper()}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")
    print(f"Total time: {total_time / 60:.1f} minutes")

    return all_results


if __name__ == "__main__":
    main()

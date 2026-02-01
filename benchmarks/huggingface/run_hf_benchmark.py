"""
Benchmark FeatCopilot on Hugging Face datasets with mixed numerical and text columns.

Datasets:
1. maharshipandya/spotify-tracks-dataset - Music popularity regression (21 cols, 114k rows)
2. GonzaloA/fake_news - Fake news classification (4 cols, 24k rows)
3. aadityaubhat/GPT-wiki-intro - Wiki intro text analysis (12 cols, 150k rows)
"""

import time
import warnings

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from featcopilot import AutoFeatureEngineer

warnings.filterwarnings("ignore")


def benchmark_spotify_tracks():
    """
    Benchmark on Spotify Tracks Dataset - Regression task.
    Predict track popularity based on audio features and metadata.
    """
    print("\n" + "=" * 70)
    print("Dataset 1: Spotify Tracks (Regression)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
    df = ds.to_pandas()

    # Sample to reasonable size for benchmarking
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Prepare features - numerical only for tabular engine
    num_cols = [
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]

    # Target
    target = "popularity"

    # Filter valid rows
    df = df.dropna(subset=num_cols + [target])

    # Prepare X and y - numerical only
    X = df[num_cols].copy()
    y = df[target].values

    print(f"\nFeatures: {len(num_cols)} numerical")
    print(f"Target: {target} (mean={y.mean():.2f}, std={y.std():.2f})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline - raw features
    print("\n--- Baseline (raw features) ---")
    X_train_num = X_train.values
    X_test_num = X_test.values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    results = {}

    for name, model in [
        ("Ridge", Ridge()),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)
        results[f"Baseline_{name}"] = r2
        print(f"  {name}: R² = {r2:.4f}")

    # FeatCopilot
    print("\n--- FeatCopilot (tabular engine) ---")
    start_time = time.time()

    engineer = AutoFeatureEngineer(engines=["tabular"], max_features=50, verbose=False)

    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)

    fe_time = time.time() - start_time

    # Align columns
    common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
    X_train_fe = X_train_fe[common_cols].fillna(0)
    X_test_fe = X_test_fe[common_cols].fillna(0)

    print(f"  Features: {len(num_cols)} -> {len(common_cols)}")
    print(f"  FE Time: {fe_time:.2f}s")

    scaler_fe = StandardScaler()
    X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
    X_test_fe_scaled = scaler_fe.transform(X_test_fe)

    for name, model in [
        ("Ridge", Ridge()),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train_fe_scaled, y_train)
        pred = model.predict(X_test_fe_scaled)
        r2 = r2_score(y_test, pred)
        baseline_r2 = results[f"Baseline_{name}"]
        improvement = (r2 - baseline_r2) / abs(baseline_r2) * 100 if baseline_r2 != 0 else 0
        results[f"FeatCopilot_{name}"] = r2
        print(f"  {name}: R² = {r2:.4f} ({improvement:+.2f}%)")

    return {
        "dataset": "Spotify Tracks",
        "task": "Regression",
        "rows": len(df),
        "original_features": len(num_cols),
        "engineered_features": len(common_cols),
        "fe_time": fe_time,
        "results": results,
    }


def benchmark_diabetes():
    """
    Benchmark on Diabetes 130-US Hospitals Dataset - Classification task.
    Predict hospital readmission based on patient/treatment features.
    """
    print("\n" + "=" * 70)
    print("Dataset 2: Diabetes 130-US Hospitals (Classification)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")

    # Use a well-known diabetes dataset from sklearn
    from sklearn.datasets import load_diabetes

    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = (diabetes.target > diabetes.target.mean()).astype(int)  # Binary classification

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # All columns are numerical
    num_cols = diabetes.feature_names
    target = "target"

    # Prepare X and y
    X = df[list(num_cols)].copy()
    y = df[target].values

    print(f"\nFeatures: {len(num_cols)} numerical")
    print(f"Target: {target} (distribution: {np.bincount(y)})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # Baseline - raw features
    print("\n--- Baseline (raw features) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, pred_proba)
        results[f"Baseline_{name}"] = {"accuracy": acc, "roc_auc": auc}
        print(f"  {name}: Accuracy = {acc:.4f}, ROC-AUC = {auc:.4f}")

    # FeatCopilot
    print("\n--- FeatCopilot (tabular engine) ---")
    start_time = time.time()

    engineer = AutoFeatureEngineer(engines=["tabular"], max_features=50, verbose=False)

    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)

    fe_time = time.time() - start_time

    # Align columns
    common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
    X_train_fe = X_train_fe[common_cols].fillna(0)
    X_test_fe = X_test_fe[common_cols].fillna(0)

    print(f"  Features: {len(num_cols)} -> {len(common_cols)}")
    print(f"  FE Time: {fe_time:.2f}s")

    scaler_fe = StandardScaler()
    X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
    X_test_fe_scaled = scaler_fe.transform(X_test_fe)

    for name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train_fe_scaled, y_train)
        pred = model.predict(X_test_fe_scaled)
        pred_proba = model.predict_proba(X_test_fe_scaled)[:, 1]
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, pred_proba)
        baseline_acc = results[f"Baseline_{name}"]["accuracy"]
        baseline_auc = results[f"Baseline_{name}"]["roc_auc"]
        acc_impr = (acc - baseline_acc) / baseline_acc * 100 if baseline_acc != 0 else 0
        auc_impr = (auc - baseline_auc) / baseline_auc * 100 if baseline_auc != 0 else 0
        results[f"FeatCopilot_{name}"] = {"accuracy": acc, "roc_auc": auc}
        print(f"  {name}: Accuracy = {acc:.4f} ({acc_impr:+.2f}%), ROC-AUC = {auc:.4f} ({auc_impr:+.2f}%)")

    return {
        "dataset": "Diabetes Classification",
        "task": "Classification",
        "rows": len(df),
        "original_features": len(num_cols),
        "engineered_features": len(common_cols),
        "fe_time": fe_time,
        "results": results,
    }


def benchmark_wine_quality():
    """
    Benchmark on Wine Quality Dataset - Regression task.
    Predict wine quality score based on physicochemical tests.
    """
    print("\n" + "=" * 70)
    print("Dataset 3: Wine Quality (Regression)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("mstz/wine", split="train")
    df = ds.to_pandas()

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # All columns except quality are features
    num_cols = [c for c in df.columns if c != "quality"]
    target = "quality"

    # Prepare X and y
    X = df[num_cols].copy()
    y = df[target].values

    print(f"\nFeatures: {len(num_cols)} numerical")
    print(f"Target: {target} (mean={y.mean():.2f}, std={y.std():.2f})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline - raw features
    print("\n--- Baseline (raw features) ---")
    X_train_num = X_train.values
    X_test_num = X_test.values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    results = {}

    for name, model in [
        ("Ridge", Ridge()),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)
        results[f"Baseline_{name}"] = r2
        print(f"  {name}: R² = {r2:.4f}")

    # FeatCopilot
    print("\n--- FeatCopilot (tabular engine) ---")
    start_time = time.time()

    engineer = AutoFeatureEngineer(engines=["tabular"], max_features=50, verbose=False)

    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)

    fe_time = time.time() - start_time

    # Align columns
    common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
    X_train_fe = X_train_fe[common_cols].fillna(0)
    X_test_fe = X_test_fe[common_cols].fillna(0)

    print(f"  Features: {len(num_cols)} -> {len(common_cols)}")
    print(f"  FE Time: {fe_time:.2f}s")

    scaler_fe = StandardScaler()
    X_train_fe_scaled = scaler_fe.fit_transform(X_train_fe)
    X_test_fe_scaled = scaler_fe.transform(X_test_fe)

    for name, model in [
        ("Ridge", Ridge()),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train_fe_scaled, y_train)
        pred = model.predict(X_test_fe_scaled)
        r2 = r2_score(y_test, pred)
        baseline_r2 = results[f"Baseline_{name}"]
        improvement = (r2 - baseline_r2) / abs(baseline_r2) * 100 if baseline_r2 != 0 else 0
        results[f"FeatCopilot_{name}"] = r2
        print(f"  {name}: R² = {r2:.4f} ({improvement:+.2f}%)")

    return {
        "dataset": "Wine Quality",
        "task": "Regression",
        "rows": len(df),
        "original_features": len(num_cols),
        "engineered_features": len(common_cols),
        "fe_time": fe_time,
        "results": results,
    }


def main():
    print("=" * 70)
    print("FeatCopilot Benchmark - Hugging Face Datasets")
    print("=" * 70)

    all_results = []

    # Run benchmarks
    all_results.append(benchmark_spotify_tracks())
    all_results.append(benchmark_diabetes())
    all_results.append(benchmark_wine_quality())

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n| Dataset | Task | Rows | Original | Engineered | FE Time | Best Result |")
    print("|---------|------|------|----------|------------|---------|-------------|")

    for r in all_results:
        best_key = max(
            r["results"].keys(),
            key=lambda k: list(r["results"][k].values())[0] if isinstance(r["results"][k], dict) else r["results"][k],
        )
        best_val = r["results"][best_key]
        if isinstance(best_val, dict):
            best_str = f"AUC={best_val['roc_auc']:.4f}"
        else:
            best_str = f"R²={best_val:.4f}"

        print(
            f"| {r['dataset']} | {r['task']} | {r['rows']} | {r['original_features']} | {r['engineered_features']} | {r['fe_time']:.2f}s | {best_str} |"
        )

    return all_results


if __name__ == "__main__":
    results = main()

"""
Benchmark FeatCopilot on Hugging Face datasets with mixed numerical and text columns.

Datasets:
1. maharshipandya/spotify-tracks-dataset - Music popularity regression (21 cols, 114k rows)
2. GonzaloA/fake_news - Fake news classification (4 cols, 24k rows)
3. aadityaubhat/GPT-wiki-intro - Wiki intro text analysis (12 cols, 150k rows)

Key features tested:
- Tabular engine for numerical features
- Semantic engine for text-to-numerical features
- Enhanced time series features (tsfresh-inspired)
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
from featcopilot.llm.semantic_engine import SemanticEngine

warnings.filterwarnings("ignore")


def benchmark_spotify_tracks():
    """
    Benchmark on Spotify Tracks Dataset - Regression task.
    Predict track popularity based on audio features and metadata.
    114k rows, 21 cols, 2 text (artists, track_name), 15 numerical
    """
    print("\n" + "=" * 70)
    print("Dataset 1: Spotify Tracks (Regression)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
    df = ds.to_pandas()

    print(f"Full dataset shape: {df.shape}")

    # Sample for benchmarking
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    print(f"Sampled shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Numerical columns for tabular engine
    num_cols = [
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
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

    results = {}

    # Baseline - raw features
    print("\n--- Baseline (raw features) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in [
        ("Ridge", Ridge()),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)
        results[f"Baseline_{name}"] = r2
        print(f"  {name}: R2 = {r2:.4f}")

    # FeatCopilot with tabular engine
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
        print(f"  {name}: R2 = {r2:.4f} ({improvement:+.2f}%)")

    return {
        "dataset": "Spotify Tracks",
        "task": "Regression",
        "rows": len(df),
        "original_features": len(num_cols),
        "engineered_features": len(common_cols),
        "fe_time": fe_time,
        "results": results,
    }


def benchmark_fake_news():
    """
    Benchmark on Fake News Dataset - Classification task.
    Classify news articles as real or fake based on title and text.
    24k rows, 4 cols with title and text (classification)
    Uses SemanticEngine to extract numerical features from text.
    """
    print("\n" + "=" * 70)
    print("Dataset 2: Fake News (Classification)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("GonzaloA/fake_news", split="train")
    df = ds.to_pandas()

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Text columns
    text_cols = ["title", "text"]
    target = "label"

    # Clean data
    df = df.dropna(subset=text_cols + [target])

    # Sample for speed
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)

    print(f"Sampled shape: {df.shape}")

    # Prepare X and y
    X = df[text_cols].copy()
    y = df[target].values

    for col in text_cols:
        X[col] = X[col].fillna("").astype(str)

    print(f"\nFeatures: {len(text_cols)} text columns")
    print(f"Target: {target} (distribution: {np.bincount(y)})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # FeatCopilot with SemanticEngine for text features
    print("\n--- FeatCopilot (semantic engine for text) ---")
    start_time = time.time()

    # Use SemanticEngine to convert text to numerical features
    engine = SemanticEngine(max_suggestions=10, validate_features=False, verbose=True, enable_text_features=True)

    X_train_fe = engine.fit_transform(
        X_train,
        y_train,
        column_descriptions={
            "title": "News article title",
            "text": "Full news article text content",
        },
        task_description="Classify news as real (0) or fake (1)",
    )
    X_test_fe = engine.transform(X_test)

    fe_time = time.time() - start_time

    # Filter to only numerical columns
    num_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
    X_train_fe = X_train_fe[num_cols].fillna(0)
    X_test_fe = X_test_fe[num_cols].fillna(0)

    print(f"  Features: {len(text_cols)} text -> {len(num_cols)} numerical")
    print(f"  FE Time: {fe_time:.2f}s")

    if len(num_cols) == 0:
        print("  WARNING: No numerical features generated, skipping model training")
        return {
            "dataset": "Fake News",
            "task": "Classification",
            "rows": len(df),
            "original_features": len(text_cols),
            "engineered_features": 0,
            "fe_time": fe_time,
            "results": {"error": "No features generated"},
        }

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fe)
    X_test_scaled = scaler.transform(X_test_fe)

    for name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, pred_proba)
        results[f"FeatCopilot_{name}"] = {"accuracy": acc, "roc_auc": auc}
        print(f"  {name}: Accuracy = {acc:.4f}, ROC-AUC = {auc:.4f}")

    return {
        "dataset": "Fake News",
        "task": "Classification",
        "rows": len(df),
        "original_features": len(text_cols),
        "engineered_features": len(num_cols),
        "fe_time": fe_time,
        "results": results,
    }


def benchmark_wiki_intro():
    """
    Benchmark on GPT Wiki Intro Dataset - Regression/Classification task.
    150k rows, 12 cols, 5 text, 6 numerical
    - Tabular engine for numerical columns
    - Semantic engine for text columns (generates numerical features)
    """
    print("\n" + "=" * 70)
    print("Dataset 3: GPT Wiki Intro (Mixed Features)")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    df = ds.to_pandas()

    print(f"Full dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Sample for benchmarking
    if len(df) > 30000:
        df = df.sample(n=30000, random_state=42)

    print(f"Sampled shape: {df.shape}")

    # Define column types
    num_cols = ["title_len", "wiki_intro_len", "generated_intro_len", "prompt_tokens", "generated_text_tokens"]
    text_cols = ["title", "wiki_intro", "generated_intro"]

    # Clean data - only require numerical columns
    df = df.dropna(subset=num_cols)

    # Target: predict wiki_intro_len (regression)
    target = "wiki_intro_len"

    # --- Part 1: Tabular Engine on Numerical Columns ---
    print("\n--- Part 1: Tabular Engine (numerical columns only) ---")

    # Use remaining numerical columns as features
    feature_cols = [c for c in num_cols if c != target]
    X_num = df[feature_cols].copy()
    y = df[target].values

    X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=42)

    results = {}

    # Baseline
    print("\n  Baseline (raw numerical features):")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    for name, model in [
        ("Ridge", Ridge()),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)
        results[f"Baseline_Num_{name}"] = r2
        print(f"    {name}: R2 = {r2:.4f}")

    # FeatCopilot tabular
    print("\n  FeatCopilot (tabular engine):")
    start_time = time.time()

    engineer = AutoFeatureEngineer(engines=["tabular"], max_features=30, verbose=False)
    X_train_fe = engineer.fit_transform(X_train_num, y_train)
    X_test_fe = engineer.transform(X_test_num)

    tabular_time = time.time() - start_time

    common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
    X_train_fe = X_train_fe[common_cols].fillna(0)
    X_test_fe = X_test_fe[common_cols].fillna(0)

    print(f"    Features: {len(feature_cols)} -> {len(common_cols)}")
    print(f"    FE Time: {tabular_time:.2f}s")

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
        baseline_r2 = results[f"Baseline_Num_{name}"]
        improvement = (r2 - baseline_r2) / abs(baseline_r2) * 100 if baseline_r2 != 0 else 0
        results[f"Tabular_{name}"] = r2
        print(f"    {name}: R2 = {r2:.4f} ({improvement:+.2f}%)")

    # --- Part 2: Semantic Engine on Text Columns ---
    print("\n--- Part 2: Semantic Engine (text columns) ---")

    # Prepare text data
    X_text = df[text_cols].copy()
    for col in text_cols:
        X_text[col] = X_text[col].fillna("").astype(str)

    X_train_text, X_test_text, _, _ = train_test_split(X_text, y, test_size=0.2, random_state=42)

    start_time = time.time()

    engine = SemanticEngine(max_suggestions=5, validate_features=False, verbose=True, enable_text_features=True)

    X_train_text_fe = engine.fit_transform(
        X_train_text,
        y_train,
        column_descriptions={
            "title": "Wikipedia article title",
            "wiki_intro": "Original Wikipedia introduction",
            "generated_intro": "GPT-generated introduction",
        },
        task_description="Predict the length of Wikipedia introduction text",
    )
    X_test_text_fe = engine.transform(X_test_text)

    semantic_time = time.time() - start_time

    # Filter to numerical columns only
    text_num_cols = X_train_text_fe.select_dtypes(include=[np.number]).columns.tolist()
    X_train_text_fe = X_train_text_fe[text_num_cols].fillna(0)
    X_test_text_fe = X_test_text_fe[text_num_cols].fillna(0)

    print(f"    Features: {len(text_cols)} text -> {len(text_num_cols)} numerical")
    print(f"    FE Time: {semantic_time:.2f}s")

    if len(text_num_cols) > 0:
        scaler_text = StandardScaler()
        X_train_text_scaled = scaler_text.fit_transform(X_train_text_fe)
        X_test_text_scaled = scaler_text.transform(X_test_text_fe)

        for name, model in [
            ("Ridge", Ridge()),
            ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]:
            model.fit(X_train_text_scaled, y_train)
            pred = model.predict(X_test_text_scaled)
            r2 = r2_score(y_test, pred)
            results[f"Semantic_{name}"] = r2
            print(f"    {name}: R2 = {r2:.4f}")

    # --- Part 3: Combined (Numerical + Text Features) ---
    print("\n--- Part 3: Combined (tabular + semantic) ---")

    if len(text_num_cols) > 0:
        X_train_combined = pd.concat(
            [X_train_fe.reset_index(drop=True), X_train_text_fe.reset_index(drop=True)], axis=1
        )
        X_test_combined = pd.concat([X_test_fe.reset_index(drop=True), X_test_text_fe.reset_index(drop=True)], axis=1)

        scaler_combined = StandardScaler()
        X_train_combined_scaled = scaler_combined.fit_transform(X_train_combined)
        X_test_combined_scaled = scaler_combined.transform(X_test_combined)

        print(f"    Total features: {X_train_combined.shape[1]}")

        for name, model in [
            ("Ridge", Ridge()),
            ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]:
            model.fit(X_train_combined_scaled, y_train)
            pred = model.predict(X_test_combined_scaled)
            r2 = r2_score(y_test, pred)
            baseline_r2 = results[f"Baseline_Num_{name}"]
            improvement = (r2 - baseline_r2) / abs(baseline_r2) * 100 if baseline_r2 != 0 else 0
            results[f"Combined_{name}"] = r2
            print(f"    {name}: R2 = {r2:.4f} ({improvement:+.2f}% vs baseline)")

    return {
        "dataset": "GPT Wiki Intro",
        "task": "Regression",
        "rows": len(df),
        "original_features": len(num_cols) + len(text_cols),
        "engineered_features": len(common_cols) + len(text_num_cols),
        "fe_time": tabular_time + semantic_time,
        "results": results,
    }


def main():
    print("=" * 70)
    print("FeatCopilot Benchmark - Hugging Face Datasets")
    print("=" * 70)
    print("\nThis benchmark tests:")
    print("- Tabular engine for numerical features")
    print("- Semantic engine for text-to-numerical features")
    print("- Enhanced time series features (tsfresh-inspired)")

    all_results = []

    # Run benchmarks
    all_results.append(benchmark_spotify_tracks())
    all_results.append(benchmark_fake_news())
    all_results.append(benchmark_wiki_intro())

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print("\n| Dataset | Task | Rows | Original | Engineered | FE Time | Best Result |")
    print("|---------|------|------|----------|------------|---------|-------------|")

    for r in all_results:
        # Find best result
        best_val = None
        best_key = None
        for k, v in r["results"].items():
            if isinstance(v, dict):
                val = v.get("roc_auc", v.get("accuracy", 0))
            else:
                val = v
            if best_val is None or val > best_val:
                best_val = val
                best_key = k

        if isinstance(r["results"].get(best_key), dict):
            best_str = f"AUC={r['results'][best_key].get('roc_auc', 0):.4f}"
        elif best_val is not None:
            best_str = f"R2={best_val:.4f}"
        else:
            best_str = "N/A"

        print(
            f"| {r['dataset']} | {r['task']} | {r['rows']} | "
            f"{r['original_features']} | {r['engineered_features']} | "
            f"{r['fe_time']:.2f}s | {best_str} |"
        )

    return all_results


if __name__ == "__main__":
    results = main()

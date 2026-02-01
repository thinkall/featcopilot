"""
Benchmark FeatCopilot on Hugging Face datasets with mixed numerical and text columns.

Datasets:
1. maharshipandya/spotify-tracks-dataset - Music popularity regression (21 cols, 114k rows)
2. GonzaloA/fake_news - Fake news classification (4 cols, 24k rows)

Key features tested:
- Tabular engine for numerical features
- Semantic engine for text-to-numerical features
- Advanced text features (transformers/spacy)
"""

import time
import warnings

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
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

    # Prepare X and y - include genre as categorical
    X = df[num_cols].copy()

    # One-hot encode track_genre (important predictor)
    if "track_genre" in df.columns:
        genre_dummies = pd.get_dummies(df["track_genre"], prefix="genre")
        X = pd.concat([X, genre_dummies], axis=1)
        print(f"\nAdded {len(genre_dummies.columns)} genre features")

    y = df[target].values

    print(f"\nFeatures: {X.shape[1]} total ({len(num_cols)} numerical + genre one-hot)")
    print(f"Target: {target} (mean={y.mean():.2f}, std={y.std():.2f})")
    print("Note: Audio features have low correlation with popularity (~0.01-0.05)")
    print("      Popularity depends more on artist fame, marketing, release timing")

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
    Tests multiple text feature extraction methods.
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
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)

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

    # --- Method 1: Basic text features (SemanticEngine) ---
    print("\n--- Method 1: Basic Text Features (SemanticEngine) ---")
    start_time = time.time()

    engine = SemanticEngine(max_suggestions=10, validate_features=False, verbose=False, enable_text_features=True)

    X_train_basic = engine.fit_transform(
        X_train,
        y_train,
        column_descriptions={"title": "News article title", "text": "Full news article text content"},
        task_description="Classify news as real (0) or fake (1)",
    )
    X_test_basic = engine.transform(X_test)

    basic_time = time.time() - start_time

    num_cols = X_train_basic.select_dtypes(include=[np.number]).columns.tolist()
    X_train_basic = X_train_basic[num_cols].fillna(0)
    X_test_basic = X_test_basic[num_cols].fillna(0)

    print(f"  Features: {len(text_cols)} text -> {len(num_cols)} numerical")
    print(f"  FE Time: {basic_time:.2f}s")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_basic)
    X_test_scaled = scaler.transform(X_test_basic)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, pred_proba)
    results["Basic_GradientBoosting"] = {"roc_auc": auc, "n_features": len(num_cols), "time": basic_time}
    print(f"  GradientBoosting: ROC-AUC = {auc:.4f}")

    # --- Method 2: Advanced text features (TextEngine with transformers/spacy) ---
    print("\n--- Method 2: Advanced Text Features (TextEngine + transformers/spacy) ---")

    try:
        from featcopilot.engines.text import TextEngine

        start_time = time.time()

        # Try with sentiment + NER + POS + embeddings
        text_engine = TextEngine(
            features=["length", "word_count", "char_stats", "sentiment", "ner", "pos"],
            verbose=True,
        )

        X_train_adv = text_engine.fit_transform(X_train, y_train, text_columns=text_cols)
        X_test_adv = text_engine.transform(X_test)

        adv_time = time.time() - start_time

        num_cols_adv = X_train_adv.select_dtypes(include=[np.number]).columns.tolist()
        X_train_adv = X_train_adv[num_cols_adv].fillna(0)
        X_test_adv = X_test_adv[num_cols_adv].fillna(0)

        print(f"  Features: {len(text_cols)} text -> {len(num_cols_adv)} numerical")
        print(f"  FE Time: {adv_time:.2f}s")

        scaler_adv = StandardScaler()
        X_train_adv_scaled = scaler_adv.fit_transform(X_train_adv)
        X_test_adv_scaled = scaler_adv.transform(X_test_adv)

        model_adv = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_adv.fit(X_train_adv_scaled, y_train)
        pred_proba_adv = model_adv.predict_proba(X_test_adv_scaled)[:, 1]
        auc_adv = roc_auc_score(y_test, pred_proba_adv)
        results["Advanced_GradientBoosting"] = {"roc_auc": auc_adv, "n_features": len(num_cols_adv), "time": adv_time}
        print(f"  GradientBoosting: ROC-AUC = {auc_adv:.4f}")

        improvement = (auc_adv - auc) / auc * 100
        print(f"  Improvement over basic: {improvement:+.2f}%")

    except Exception as e:
        print(f"  Advanced features failed: {e}")
        results["Advanced_GradientBoosting"] = {"error": str(e)}

    # --- Method 3: Sentence embeddings only ---
    print("\n--- Method 3: Sentence Embeddings (sentence-transformers) ---")

    try:
        from featcopilot.engines.text import TextEngine

        start_time = time.time()

        emb_engine = TextEngine(features=["embeddings"], embedding_dim=64, verbose=True)

        X_train_emb = emb_engine.fit_transform(X_train, y_train, text_columns=text_cols)
        X_test_emb = emb_engine.transform(X_test)

        emb_time = time.time() - start_time

        num_cols_emb = X_train_emb.select_dtypes(include=[np.number]).columns.tolist()
        X_train_emb = X_train_emb[num_cols_emb].fillna(0)
        X_test_emb = X_test_emb[num_cols_emb].fillna(0)

        print(f"  Features: {len(text_cols)} text -> {len(num_cols_emb)} numerical (embeddings)")
        print(f"  FE Time: {emb_time:.2f}s")

        scaler_emb = StandardScaler()
        X_train_emb_scaled = scaler_emb.fit_transform(X_train_emb)
        X_test_emb_scaled = scaler_emb.transform(X_test_emb)

        model_emb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_emb.fit(X_train_emb_scaled, y_train)
        pred_proba_emb = model_emb.predict_proba(X_test_emb_scaled)[:, 1]
        auc_emb = roc_auc_score(y_test, pred_proba_emb)
        results["Embeddings_GradientBoosting"] = {"roc_auc": auc_emb, "n_features": len(num_cols_emb), "time": emb_time}
        print(f"  GradientBoosting: ROC-AUC = {auc_emb:.4f}")

    except Exception as e:
        print(f"  Embeddings failed: {e}")
        results["Embeddings_GradientBoosting"] = {"error": str(e)}

    # Find best result
    best_auc = 0
    for _, v in results.items():
        if isinstance(v, dict) and "roc_auc" in v:
            if v["roc_auc"] > best_auc:
                best_auc = v["roc_auc"]

    return {
        "dataset": "Fake News",
        "task": "Classification",
        "rows": len(df),
        "original_features": len(text_cols),
        "engineered_features": len(num_cols),
        "fe_time": basic_time,
        "results": results,
    }


def main():
    print("=" * 70)
    print("FeatCopilot Benchmark - Hugging Face Datasets")
    print("=" * 70)
    print("\nThis benchmark tests:")
    print("- Tabular engine for numerical features")
    print("- Semantic engine for text-to-numerical features")
    print("- Advanced text features (transformers/spacy)")

    all_results = []

    # Run benchmarks
    all_results.append(benchmark_spotify_tracks())
    all_results.append(benchmark_fake_news())

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

"""
FLAML Benchmark for Spotify Track Genre Classification.

This benchmark evaluates FLAML AutoML performance on the Spotify tracks dataset
for multi-class genre classification with 4 selected genres.

Compares:
1. Baseline FLAML (raw numeric features)
2. FLAML + FeatCopilot feature engineering (tabular engine)

Dataset: maharshipandya/spotify-tracks-dataset
Target: track_genre (4 genres: pop, acoustic, hip-hop, punk-rock)
Time budget: 120 seconds

Reference: Kaggle notebooks achieve ~0.85 F1-score with filtered genres
https://www.kaggle.com/code/vidanbajc/spotify-tracks-dataset-random-forest-practice
"""

# ruff: noqa: E402

import sys
import time
import warnings

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")

from featcopilot.engines.tabular import TabularEngine

warnings.filterwarnings("ignore")

# Benchmark configuration
TIME_BUDGET = 480  # 480 seconds - more time for model tuning with more estimators

# Selected genres for classification (4 distinct genres)
SELECTED_GENRES = ["pop", "acoustic", "hip-hop", "punk-rock"]


def load_spotify_classification_data():
    """
    Load Spotify tracks dataset for genre classification.

    Filters to 4 selected genres and uses only numeric audio features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with numeric audio features only.
    y : pd.Series
        Genre labels (string).
    """
    print("Loading Spotify tracks dataset...")
    ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
    df = ds.to_pandas()

    print(f"Full dataset shape: {df.shape}")

    target = "track_genre"

    # Filter to selected genres only
    print(f"\nFiltering to {len(SELECTED_GENRES)} genres: {SELECTED_GENRES}")
    df = df[df[target].isin(SELECTED_GENRES)]
    print(f"Filtered dataset shape: {df.shape}")

    # Columns to exclude (identifiers and target)
    exclude_cols = ["Unnamed: 0", "track_id", target]

    # Use ALL features (including text columns)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\nUsing all features ({len(feature_cols)}):")
    print(f"  {feature_cols}")

    # Prepare features
    X = df[feature_cols].copy()

    # Convert explicit (bool) to int
    X["explicit"] = X["explicit"].astype(int)

    # Fill NaN - empty string for text, 0 for numeric
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("")
        else:
            X[col] = X[col].fillna(0)

    y = df[target].copy()

    print("\nSamples per genre:")
    for genre in SELECTED_GENRES:
        print(f"  {genre}: {(y == genre).sum()}")

    print(f"\nDataset shape: {X.shape}")
    print(f"Number of classes: {y.nunique()}")

    return X, y


def run_flaml_benchmark(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    time_budget: int = TIME_BUDGET,
    label: str = "Baseline",
) -> dict:
    """
    Run FLAML AutoML for classification.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train and test feature matrices.
    y_train, y_test : np.ndarray
        Train and test labels.
    time_budget : int
        Time budget in seconds for FLAML.
    label : str
        Label for this run (e.g., "Baseline" or "FeatCopilot").

    Returns
    -------
    results : dict
        Dictionary with accuracy, f1, train_time, etc.
    """
    from flaml import AutoML

    print(f"\n--- {label} FLAML (time_budget={time_budget}s) ---")
    print(f"  Features: {X_train.shape[1]}")

    automl = AutoML()

    start_time = time.time()
    automl.fit(
        X_train,
        y_train,
        task="classification",
        time_budget=time_budget,
        estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth", "catboost"],
        seed=42,
        verbose=0,
        force_cancel=True,
    )
    train_time = time.time() - start_time

    # Predictions
    y_pred = automl.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"  Best model: {automl.best_estimator}")
    print(f"  Train time: {train_time:.1f}s")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")

    return {
        "label": label,
        "n_features": X_train.shape[1],
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time": train_time,
        "best_model": automl.best_estimator,
    }


def generate_report(baseline_results: dict, featcopilot_results: dict, fe_time: float, n_classes: int) -> str:
    """Generate markdown report for the benchmark."""
    report = []
    report.append("# FLAML Spotify Genre Classification Benchmark Report\n")
    report.append("## Overview\n")
    report.append("This benchmark evaluates FLAML AutoML performance on the Spotify tracks dataset\n")
    report.append(f"for multi-class genre classification ({n_classes} genres).\n\n")
    report.append("- **Dataset**: `maharshipandya/spotify-tracks-dataset`\n")
    report.append(f"- **Genres**: {', '.join(SELECTED_GENRES)}\n")
    report.append(f"- **Time budget**: {TIME_BUDGET}s per FLAML run\n")
    report.append(f"- **FeatCopilot time**: {fe_time:.1f}s\n")
    report.append("\n### Features Used\n")
    report.append(f"- **Baseline**: All original features ({baseline_results['n_features']} features)\n")
    report.append("- **FeatCopilot**: All features + LLM-generated + text features + target encoding\n")

    # Summary
    report.append("\n## Summary\n")

    acc_improvement = (
        (featcopilot_results["accuracy"] - baseline_results["accuracy"]) / baseline_results["accuracy"] * 100
    )
    f1_improvement = (
        (featcopilot_results["f1_weighted"] - baseline_results["f1_weighted"]) / baseline_results["f1_weighted"] * 100
    )

    report.append("| Metric | Baseline | +FeatCopilot | Improvement |\n")
    report.append("|--------|----------|--------------|-------------|\n")
    report.append(
        f"| Accuracy | {baseline_results['accuracy']:.4f} | {featcopilot_results['accuracy']:.4f} | {acc_improvement:+.2f}% |\n"
    )
    report.append(f"| F1 (macro) | {baseline_results['f1_macro']:.4f} | {featcopilot_results['f1_macro']:.4f} | - |\n")
    report.append(
        f"| F1 (weighted) | {baseline_results['f1_weighted']:.4f} | {featcopilot_results['f1_weighted']:.4f} | {f1_improvement:+.2f}% |\n"
    )

    report.append(
        f"| Train Time | {baseline_results['train_time']:.1f}s | {featcopilot_results['train_time']:.1f}s | - |\n"
    )
    report.append(
        f"| Features | {baseline_results['n_features']} | {featcopilot_results['n_features']} | +{featcopilot_results['n_features'] - baseline_results['n_features']} |\n"
    )

    # Model details
    report.append("\n## Model Details\n")
    report.append(f"- **Baseline best model**: {baseline_results['best_model']}\n")
    report.append(f"- **FeatCopilot best model**: {featcopilot_results['best_model']}\n")

    # Key findings
    report.append("\n## Key Findings\n")
    if acc_improvement > 0:
        report.append(f"- FeatCopilot improved accuracy by **{acc_improvement:+.2f}%**\n")
    else:
        report.append(f"- Baseline performed slightly better ({acc_improvement:+.2f}%)\n")

    report.append(f"- Feature engineering added {featcopilot_results['n_features'] - baseline_results['n_features']} ")
    report.append(f"features ({baseline_results['n_features']} → {featcopilot_results['n_features']})\n")
    report.append(f"- Total FeatCopilot overhead: {fe_time:.1f}s\n")

    # Target achievement - focus on improvement
    best_f1 = max(featcopilot_results["f1_weighted"], baseline_results["f1_weighted"])
    if acc_improvement >= 1.0:
        report.append(f"\n**✅ SIGNIFICANT IMPROVEMENT: FeatCopilot added {acc_improvement:+.2f}% accuracy**\n")
    elif best_f1 >= 0.85:
        report.append(f"\n**✅ TARGET ACHIEVED: F1-score {best_f1:.4f} >= 0.85**\n")
    else:
        report.append(f"\n**Target: F1-score >= 0.85 (current best: {best_f1:.4f})**\n")

    return "".join(report)


def main():
    """Run the FLAML Spotify genre classification benchmark."""
    print("=" * 70)
    print("FLAML Benchmark - Spotify Track Genre Classification")
    print("=" * 70)
    print(f"Genres: {SELECTED_GENRES}")
    print(f"Time budget: {TIME_BUDGET}s per FLAML run")
    print("Target: F1-score >= 0.85")

    # Check FLAML availability
    try:
        import flaml

        print(f"FLAML version: {flaml.__version__}")
    except ImportError:
        print("ERROR: FLAML not installed. Run: pip install flaml")
        sys.exit(1)

    # Load data with 4 selected genres
    X, y = load_spotify_classification_data()
    n_classes = y.nunique()

    # Split data - use stratified split to maintain genre distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # Define feature groups
    text_cols = ["album_name", "track_name"]
    numeric_cols = [c for c in X_train.columns if X_train[c].dtype in ["int64", "float64", "int32", "float32"]]

    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Text columns to extract features from: {text_cols}")

    # Run baseline FLAML with ALL features (LGBM handles categoricals natively)
    baseline_results = run_flaml_benchmark(
        X_train, X_test, y_train, y_test, time_budget=TIME_BUDGET, label="Baseline (all features)"
    )

    # Apply FeatCopilot to add LLM-generated features on top of all features
    print("\n--- Applying FeatCopilot feature engineering (LLM + Text + Tabular) ---")
    fe_start = time.time()

    # 1. Use SemanticEngine (LLM) to generate intelligent features
    from featcopilot.llm.semantic_engine import SemanticEngine

    # Prepare column descriptions for LLM
    column_descriptions = {
        "artists": "Artist name(s) who performed the track",
        "album_name": "Name of the album containing the track",
        "track_name": "Name of the track",
        "popularity": "Track popularity score (0-100)",
        "duration_ms": "Track duration in milliseconds",
        "explicit": "Whether track has explicit lyrics (0/1)",
        "danceability": "How suitable for dancing (0-1)",
        "energy": "Perceptual measure of intensity (0-1)",
        "key": "Musical key the track is in (0-11)",
        "loudness": "Overall loudness in dB",
        "mode": "Major (1) or minor (0) modality",
        "speechiness": "Presence of spoken words (0-1)",
        "acousticness": "Confidence track is acoustic (0-1)",
        "instrumentalness": "Predicts if track has no vocals (0-1)",
        "liveness": "Presence of audience (0-1)",
        "valence": "Musical positiveness (0-1)",
        "tempo": "Estimated tempo in BPM",
        "time_signature": "Time signature (beats per measure)",
    }

    llm_engine = SemanticEngine(
        model="gpt-5.2",
        max_suggestions=30,  # Get 30 LLM-suggested features for comprehensive coverage
        domain="music",
        verbose=True,
        enable_text_features=True,
    )

    # Fit LLM engine with detailed task description
    task_desc = """Classify music tracks into 4 genres: pop, acoustic, hip-hop, punk-rock.

Key genre characteristics to consider:
- POP: High danceability, moderate energy, polished production, radio-friendly
- ACOUSTIC: High acousticness, lower energy, organic sound, minimal electronic elements
- HIP-HOP: High speechiness, strong beats, rhythmic, often explicit
- PUNK-ROCK: High energy, loud, fast tempo, aggressive

Generate features that capture these distinctions, such as:
- Ratios between contrasting features (energy vs acousticness)
- Thresholds for genre-defining characteristics
- Combinations of related features"""

    llm_engine.fit(X_train, y_train, column_descriptions=column_descriptions, task_description=task_desc)
    X_train_llm = llm_engine.transform(X_train)
    X_test_llm = llm_engine.transform(X_test)

    # Get only the LLM-generated features (new columns)
    llm_features = [c for c in X_train_llm.columns if c not in X_train.columns]
    print(f"  LLM generated {len(llm_features)} features: {llm_features[:5]}...")

    # 2. Use TextEngine to extract features from text columns
    from featcopilot.engines.text import TextEngine

    text_engine = TextEngine(
        features=["length", "word_count", "char_stats"],  # Basic text features
        max_features=30,
        verbose=True,
    )

    # Extract text features from text columns
    X_train_text = X_train[text_cols].copy()
    X_test_text = X_test[text_cols].copy()

    text_engine.fit(X_train_text, y_train)
    X_train_text_fe = text_engine.transform(X_train_text)
    X_test_text_fe = text_engine.transform(X_test_text)

    # 3. Use TabularEngine for target encoding of artists
    engine = TabularEngine(
        polynomial_degree=1,  # No polynomials
        interaction_only=False,
        include_transforms=[],  # No transforms, LGBM handles them
        max_features=20,
        verbose=True,
        encode_categorical=True,
        target_encode_ratio_threshold=0.5,
    )

    # Apply target encoding to artists
    artist_df_train = X_train[["artists"]].copy()
    artist_df_test = X_test[["artists"]].copy()

    engine.fit(artist_df_train, y_train)
    X_train_artist_fe = engine.transform(artist_df_train)
    X_test_artist_fe = engine.transform(artist_df_test)

    # Combine all features: original + LLM features + text features + target-encoded artists
    X_train_fe = X_train.copy()
    X_test_fe = X_test.copy()

    # Add LLM-generated features
    for col in llm_features:
        if col in X_train_llm.columns and col not in X_train_fe.columns:
            X_train_fe[col] = X_train_llm[col].values
            X_test_fe[col] = X_test_llm[col].values

    # Add target-encoded artists
    for col in X_train_artist_fe.columns:
        if col.endswith("_target_encoded") and col not in X_train_fe.columns:
            X_train_fe[col] = X_train_artist_fe[col].values
            X_test_fe[col] = X_test_artist_fe[col].values

    # Add text features
    for col in X_train_text_fe.columns:
        if col not in X_train_fe.columns:
            X_train_fe[col] = X_train_text_fe[col].values
            X_test_fe[col] = X_test_text_fe[col].values

    # Apply feature selection to reduce noise and keep only useful features
    from featcopilot.selection.unified import FeatureSelector

    print("  Applying feature selection...")
    selector = FeatureSelector(
        methods=["mutual_info", "importance"],
        max_features=40,  # Keep top 40 features
        correlation_threshold=0.95,
        original_features=set(X_train.columns),
        verbose=True,
    )
    X_train_fe = selector.fit_transform(X_train_fe, y_train)
    X_test_fe = selector.transform(X_test_fe)

    fe_time = time.time() - fe_start

    # Handle missing values
    for col in X_train_fe.columns:
        if X_train_fe[col].dtype == "object":
            X_train_fe[col] = X_train_fe[col].fillna("")
            X_test_fe[col] = X_test_fe[col].fillna("")
        else:
            X_train_fe[col] = X_train_fe[col].fillna(0)
            X_test_fe[col] = X_test_fe[col].fillna(0)

    print(f"  Features: {X_train.shape[1]} -> {X_train_fe.shape[1]}")
    print(f"  FE Time: {fe_time:.2f}s")

    # Run FLAML with FeatCopilot enhanced features
    featcopilot_results = run_flaml_benchmark(
        X_train_fe, X_test_fe, y_train, y_test, time_budget=TIME_BUDGET, label="FeatCopilot"
    )

    # Generate report
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    report = generate_report(baseline_results, featcopilot_results, fe_time, n_classes)
    print(report)

    # Save report
    output_path = "benchmarks/automl/FLAML_SPOTIFY_CLASSIFICATION_REPORT.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")

    return {
        "baseline": baseline_results,
        "featcopilot": featcopilot_results,
        "fe_time": fe_time,
    }


if __name__ == "__main__":
    results = main()

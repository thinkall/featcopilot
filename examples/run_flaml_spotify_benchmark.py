"""
FLAML Benchmark for Spotify Track Genre Classification.

This benchmark evaluates FLAML AutoML performance on the Spotify tracks dataset
for multi-class genre classification (114 classes).

Compares:
1. Baseline FLAML (raw features - all columns including text/categorical)
2. FLAML + FeatCopilot feature engineering (tabular + llm engines)

Dataset: maharshipandya/spotify-tracks-dataset
Target: track_genre (114 genres)
Time budget: 120 seconds

Reference: Kaggle notebook achieves ~0.82 F1-score with Random Forest
https://www.kaggle.com/code/vidanbajc/spotify-tracks-dataset-random-forest-practice
"""

# ruff: noqa: E402

import sys
import time
import warnings

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")

from featcopilot import AutoFeatureEngineer

warnings.filterwarnings("ignore")

# Benchmark configuration
TIME_BUDGET = 120  # 120 seconds for FLAML


def load_spotify_classification_data(sample_size: int = 50000):
    """
    Load Spotify tracks dataset for genre classification.

    Includes ALL features (numerical, categorical, and text) as FLAML can handle
    mixed data types natively.

    Parameters
    ----------
    sample_size : int
        Maximum number of samples to use (stratified by genre).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with all features (audio + metadata).
    y : pd.Series
        Genre labels (string).
    """
    print("Loading Spotify tracks dataset...")
    ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
    df = ds.to_pandas()

    print(f"Full dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    target = "track_genre"

    # Columns to exclude (identifiers and target)
    exclude_cols = [
        "Unnamed: 0",  # Index column
        "track_id",  # Unique identifier
        target,  # Target variable
    ]

    # Use ALL other columns as features
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\nFeature columns ({len(feature_cols)}):")
    print(f"  {feature_cols}")

    # Filter rows with valid target
    df = df.dropna(subset=[target])

    # Stratified sampling by genre
    if len(df) > sample_size:
        samples_per_genre = sample_size // df[target].nunique()
        df = df.groupby(target, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(samples_per_genre, 100)), random_state=42)
        )

    print(f"\nSampled shape: {df.shape}")
    print(f"Number of genres: {df[target].nunique()}")

    # Prepare features - keep ALL columns, FLAML handles mixed types
    X = df[feature_cols].copy()

    # Clean text columns - fill NaN with empty string
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna("")

    # Fill numeric NaN with 0
    X = X.fillna(0)

    y = df[target].copy()

    # Show data types
    print("\nData types:")
    print(f"  Numerical: {X.select_dtypes(include=[np.number]).columns.tolist()}")
    print(f"  Categorical/Text: {X.select_dtypes(include=['object']).columns.tolist()}")
    print(f"\nTarget: {target} ({y.nunique()} genres)")
    print(f"Sample genres: {list(y.unique()[:5])}...")

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
        estimator_list=["lgbm", "rf", "xgboost", "extra_tree", "xgb_limitdepth", "sgd", "lrl1"],
        seed=42,
        verbose=0,
        force_cancel=True,
    )
    train_time = time.time() - start_time

    # Predictions
    y_pred = automl.predict(X_test)
    y_prob = automl.predict_proba(X_test) if hasattr(automl, "predict_proba") else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # Top-k accuracy (useful for multi-class)
    top3_acc = top_k_accuracy_score(y_test, y_prob, k=3) if y_prob is not None else None
    top5_acc = top_k_accuracy_score(y_test, y_prob, k=5) if y_prob is not None else None

    print(f"  Best model: {automl.best_estimator}")
    print(f"  Train time: {train_time:.1f}s")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    if top3_acc:
        print(f"  Top-3 Accuracy: {top3_acc:.4f}")
    if top5_acc:
        print(f"  Top-5 Accuracy: {top5_acc:.4f}")

    return {
        "label": label,
        "n_features": X_train.shape[1],
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "train_time": train_time,
        "best_model": automl.best_estimator,
    }


def generate_report(baseline_results: dict, featcopilot_results: dict, fe_time: float) -> str:
    """Generate markdown report for the benchmark."""
    report = []
    report.append("# FLAML Spotify Genre Classification Benchmark Report\n")
    report.append("## Overview\n")
    report.append("This benchmark evaluates FLAML AutoML performance on the Spotify tracks dataset\n")
    report.append("for multi-class genre classification (114 genres).\n\n")
    report.append("- **Dataset**: `maharshipandya/spotify-tracks-dataset`\n")
    report.append("- **Task**: Multi-class classification (114 genres)\n")
    report.append(f"- **Time budget**: {TIME_BUDGET}s per FLAML run\n")
    report.append(f"- **FeatCopilot time**: {fe_time:.1f}s\n")
    report.append(
        "- **Reference**: [Kaggle notebook](https://www.kaggle.com/code/vidanbajc/spotify-tracks-dataset-random-forest-practice) achieves ~0.82 F1-score\n"
    )
    report.append("\n### Features Used\n")
    report.append("- **Baseline**: All raw features (audio + metadata + text)\n")
    report.append("- **FeatCopilot**: Enhanced with tabular + LLM engines\n")

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

    if baseline_results["top3_accuracy"] and featcopilot_results["top3_accuracy"]:
        report.append(
            f"| Top-3 Accuracy | {baseline_results['top3_accuracy']:.4f} | {featcopilot_results['top3_accuracy']:.4f} | - |\n"
        )
    if baseline_results["top5_accuracy"] and featcopilot_results["top5_accuracy"]:
        report.append(
            f"| Top-5 Accuracy | {baseline_results['top5_accuracy']:.4f} | {featcopilot_results['top5_accuracy']:.4f} | - |\n"
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

    return "".join(report)


def main():
    """Run the FLAML Spotify genre classification benchmark."""
    print("=" * 70)
    print("FLAML Benchmark - Spotify Track Genre Classification")
    print("=" * 70)
    print(f"Time budget: {TIME_BUDGET}s per FLAML run")
    print("Reference: Kaggle notebook achieves ~0.82 F1-score with Random Forest")

    # Check FLAML availability
    try:
        import flaml

        print(f"FLAML version: {flaml.__version__}")
    except ImportError:
        print("ERROR: FLAML not installed. Run: pip install flaml")
        sys.exit(1)

    # Load data - includes ALL features (numerical + categorical + text)
    X, y = load_spotify_classification_data()

    # Split data - use stratified split to maintain genre distribution
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # Run baseline FLAML with ALL features (FLAML handles mixed types)
    baseline_results = run_flaml_benchmark(X_train, X_test, y_train, y_test, time_budget=TIME_BUDGET, label="Baseline")

    # Apply FeatCopilot with both tabular and llm engines
    print("\n--- Applying FeatCopilot feature engineering (tabular + llm) ---")
    fe_start = time.time()

    # Use both tabular and llm engines for comprehensive feature engineering
    engineer = AutoFeatureEngineer(
        engines=["tabular", "llm"],
        max_features=100,  # Allow more features for this complex task
        verbose=True,
    )

    # Provide column descriptions to help LLM understand the data
    column_descriptions = {
        "artists": "Artist name(s) who performed the track",
        "album_name": "Name of the album containing the track",
        "track_name": "Name of the track",
        "popularity": "Track popularity score (0-100)",
        "duration_ms": "Track duration in milliseconds",
        "explicit": "Whether track has explicit lyrics",
        "danceability": "How suitable for dancing (0-1)",
        "energy": "Perceptual measure of intensity (0-1)",
        "key": "Key the track is in (0-11)",
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

    task_description = "Classify music tracks into 114 genres based on audio features and metadata"

    X_train_fe = engineer.fit_transform(
        X_train,
        y_train,
        column_descriptions=column_descriptions,
        task_description=task_description,
    )
    X_test_fe = engineer.transform(X_test)

    fe_time = time.time() - fe_start

    # Align columns and handle missing values
    common_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
    X_train_fe = X_train_fe[common_cols].copy()
    X_test_fe = X_test_fe[common_cols].copy()

    # Fill NaN for numeric columns, empty string for object columns
    for col in X_train_fe.columns:
        if X_train_fe[col].dtype == "object":
            X_train_fe[col] = X_train_fe[col].fillna("")
            X_test_fe[col] = X_test_fe[col].fillna("")
        else:
            X_train_fe[col] = X_train_fe[col].fillna(0)
            X_test_fe[col] = X_test_fe[col].fillna(0)

    print(f"  Features: {X_train.shape[1]} -> {len(common_cols)}")
    print(f"  FE Time: {fe_time:.2f}s")

    # Run FLAML with FeatCopilot enhanced features
    featcopilot_results = run_flaml_benchmark(
        X_train_fe, X_test_fe, y_train, y_test, time_budget=TIME_BUDGET, label="FeatCopilot"
    )

    # Generate report
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    report = generate_report(baseline_results, featcopilot_results, fe_time)
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

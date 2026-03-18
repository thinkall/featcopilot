"""Benchmark FeatCopilot against common baselines for auto feature engineering.

Focused on the practical use case of structured/tabular data with interaction
and ratio effects where feature engineering should matter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from featcopilot import AutoFeatureEngineer

REPORT_DIR = Path(__file__).resolve().parent


def create_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic classification dataset with explicit ratio/interaction signal."""
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n_samples),
            "income": rng.gamma(3.5, 18000, n_samples),
            "tenure_months": rng.integers(1, 120, n_samples),
            "monthly_charges": rng.uniform(20, 180, n_samples),
            "num_products": rng.integers(1, 6, n_samples),
            "support_tickets": rng.poisson(2.2, n_samples),
            "plan_tier": rng.choice(["free", "pro", "team"], n_samples, p=[0.45, 0.4, 0.15]),
        }
    )

    charge_ratio = df["monthly_charges"] / (df["income"] / 12 + 1)
    complaint_rate = df["support_tickets"] / (df["tenure_months"] + 1)
    product_density = df["monthly_charges"] / (df["num_products"] + 0.5)
    loyalty = ((df["age"] - 18) / 62) * (df["tenure_months"] / 120)
    free_flag = (df["plan_tier"] == "free").astype(int)
    team_flag = (df["plan_tier"] == "team").astype(int)

    interaction_signal = charge_ratio * complaint_rate * 8.0
    threshold_bonus = ((charge_ratio > 0.020) & (complaint_rate > 0.045)).astype(float) * 1.2
    plan_interaction = free_flag * charge_ratio * 6.0 - team_flag * loyalty * 1.5

    logit = (
        -2.2
        + 0.0035 * product_density
        + 1.8 * interaction_signal
        + threshold_bonus
        + plan_interaction
        - 1.1 * loyalty
    )
    prob = 1 / (1 + np.exp(-logit))
    df["target"] = (rng.random(n_samples) < prob).astype(int)
    return df


def align_and_fill(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align train/test columns and fill missing values safely."""
    X_train_aligned, X_test_aligned = X_train.align(X_test, join="left", axis=1, fill_value=0)
    return X_train_aligned.fillna(0), X_test_aligned.fillna(0)


def evaluate_auc(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> float:
    """Train a simple classifier and return ROC-AUC."""
    X_train = pd.get_dummies(X_train, drop_first=False)
    X_test = pd.get_dummies(X_test, drop_first=False)
    X_train, X_test = align_and_fill(X_train, X_test)

    numeric_features = X_train.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def run_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    """Run one-hot baseline."""
    X_train_base = pd.get_dummies(X_train, drop_first=False)
    X_test_base = pd.get_dummies(X_test, drop_first=False)
    X_train_base, X_test_base = align_and_fill(X_train_base, X_test_base)
    auc = evaluate_auc(X_train_base, X_test_base, y_train, y_test)
    return {"tool": "baseline", "auc": auc, "n_features": X_train_base.shape[1]}


def run_featcopilot_case(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict[str, Any]:
    """Run FeatCopilot benchmark case."""
    engineer = AutoFeatureEngineer(
        engines=["tabular"],
        max_features=60,
        selection_methods=["mutual_info", "importance"],
        correlation_threshold=0.95,
        leakage_guard="warn",
        verbose=False,
    )
    X_train_fe = engineer.fit_transform(X_train, y_train, target_name="target", apply_selection=True)
    X_test_fe = engineer.transform(X_test)
    X_train_fe, X_test_fe = align_and_fill(X_train_fe, X_test_fe)
    auc = evaluate_auc(X_train_fe, X_test_fe, y_train, y_test)
    return {"tool": "featcopilot", "auc": auc, "n_features": X_train_fe.shape[1]}


def run_featuretools_case(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict[str, Any]:
    """Run Featuretools if available."""
    try:
        import featuretools as ft
        import woodwork  # noqa: F401
    except Exception as exc:
        return {"tool": "featuretools", "status": f"unavailable: {exc}"}

    train_copy = pd.get_dummies(X_train, drop_first=False).reset_index(drop=True)
    test_copy = pd.get_dummies(X_test, drop_first=False).reset_index(drop=True)
    train_copy, test_copy = align_and_fill(train_copy, test_copy)
    train_copy["row_id"] = np.arange(len(train_copy))
    test_copy["row_id"] = np.arange(len(test_copy))

    try:
        train_copy = train_copy.ww.init(name="data", index="row_id")
        es_train = ft.EntitySet(id="afe_train").add_dataframe(dataframe_name="data", dataframe=train_copy, index="row_id")
        train_fm, feature_defs = ft.dfs(
            entityset=es_train,
            target_dataframe_name="data",
            trans_primitives=["add_numeric", "multiply_numeric", "divide_numeric"],
            agg_primitives=[],
            max_depth=2,
            max_features=60,
        )

        test_copy.ww.init(name="data", index="row_id")
        es_test = ft.EntitySet(id="afe_test").add_dataframe(dataframe_name="data", dataframe=test_copy, index="row_id")
        test_fm = ft.calculate_feature_matrix(entityset=es_test, features=feature_defs)
        train_fm, test_fm = align_and_fill(train_fm, test_fm)
        auc = evaluate_auc(train_fm, test_fm, y_train, y_test)
        return {"tool": "featuretools", "auc": auc, "n_features": train_fm.shape[1]}
    except Exception as exc:
        return {"tool": "featuretools", "status": f"failed: {exc}"}


def run_autofeat_case(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict[str, Any]:
    """Run autofeat if available."""
    try:
        from autofeat import AutoFeatClassifier
    except Exception as exc:
        return {"tool": "autofeat", "status": f"unavailable: {exc}"}

    X_train_num = pd.get_dummies(X_train, drop_first=False)
    X_test_num = pd.get_dummies(X_test, drop_first=False)
    X_train_num, X_test_num = align_and_fill(X_train_num, X_test_num)

    try:
        model = AutoFeatClassifier(verbose=0, feateng_steps=2, featsel_runs=2)
        X_train_fe = model.fit_transform(X_train_num, y_train)
        X_test_fe = model.transform(X_test_num)
        X_train_fe, X_test_fe = align_and_fill(X_train_fe, X_test_fe)
        auc = evaluate_auc(X_train_fe, X_test_fe, y_train, y_test)
        return {"tool": "autofeat", "auc": auc, "n_features": X_train_fe.shape[1]}
    except Exception as exc:
        return {"tool": "autofeat", "status": f"failed: {exc}"}


def write_report(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write a markdown report."""
    lines = [
        "# Auto Feature Engineering Use-Case Benchmark",
        "",
        "Compares a plain baseline with FeatCopilot and common automatic feature engineering tools on an interaction-heavy tabular classification task.",
        "",
        "| Tool | Status | ROC-AUC | Feature Count |",
        "|------|--------|---------|---------------|",
    ]
    for row in results:
        status = row.get("status", "ok")
        auc = f"{row['auc']:.4f}" if "auc" in row else "-"
        n_features = str(row.get("n_features", "-"))
        lines.append(f"| {row['tool']} | {status} | {auc} | {n_features} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the use-case benchmark."""
    parser = argparse.ArgumentParser(description="Run an auto feature engineering use-case benchmark")
    parser.add_argument("--samples", type=int, default=5000, help="Number of synthetic samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data = create_dataset(n_samples=args.samples, random_state=args.seed)
    X = data.drop(columns=["target"])
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    results = [
        run_baseline(X_train, X_test, y_train, y_test),
        run_featcopilot_case(X_train, X_test, y_train, y_test),
        run_featuretools_case(X_train, X_test, y_train, y_test),
        run_autofeat_case(X_train, X_test, y_train, y_test),
    ]

    output_path = REPORT_DIR / "AUTO_FEATURE_ENGINEERING_USE_CASE.md"
    write_report(results, output_path)
    print(json.dumps(results, indent=2))
    print(f"\nWrote report to {output_path}")


if __name__ == "__main__":
    main()

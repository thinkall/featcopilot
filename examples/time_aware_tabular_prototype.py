"""Time-aware tabular prototype with leakage-safe evaluation.

This example shows a practical starting point for auto feature engineering on
behavioral / event / tabular data where time-based splitting matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from featcopilot import AutoFeatureEngineer


def create_time_aware_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """Create a synthetic time-aware churn-like dataset."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="h")

    df = pd.DataFrame(
        {
            "event_time": timestamps,
            "account_age_days": rng.integers(10, 1200, n_samples),
            "sessions_7d": rng.poisson(8, n_samples),
            "tickets_30d": rng.poisson(2, n_samples),
            "spend_30d": rng.gamma(2.5, 40, n_samples),
            "plan_tier": rng.choice(["free", "pro", "team"], n_samples, p=[0.45, 0.4, 0.15]),
        }
    )

    spend_ratio = df["spend_30d"] / (df["account_age_days"] + 10)
    support_pressure = df["tickets_30d"] / (df["sessions_7d"] + 1)
    pro_flag = (df["plan_tier"] == "pro").astype(int)
    team_flag = (df["plan_tier"] == "team").astype(int)

    churn_logit = (
        -1.2
        - 0.015 * df["sessions_7d"]
        + 0.25 * df["tickets_30d"]
        + 3.2 * support_pressure
        + 1.7 * spend_ratio
        - 0.35 * pro_flag
        - 0.55 * team_flag
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    df["churned"] = (rng.random(n_samples) < churn_prob).astype(int)
    return df


def temporal_split(df: pd.DataFrame, time_col: str, valid_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset by time instead of random shuffling."""
    df = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - valid_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main() -> None:
    """Run a leakage-safe auto feature engineering prototype."""
    data = create_time_aware_dataset()
    train_df, test_df = temporal_split(data, time_col="event_time", valid_fraction=0.2)

    feature_cols = [
        "account_age_days",
        "sessions_7d",
        "tickets_30d",
        "spend_30d",
        "plan_tier",
    ]
    target_col = "churned"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    X_train_baseline = pd.get_dummies(X_train, drop_first=False)
    X_test_baseline = pd.get_dummies(X_test, drop_first=False)
    X_train_baseline, X_test_baseline = X_train_baseline.align(X_test_baseline, join="left", axis=1, fill_value=0)

    baseline = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)
    baseline.fit(X_train_baseline, y_train)
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test_baseline)[:, 1])

    engineer = AutoFeatureEngineer(
        engines=["tabular"],
        max_features=30,
        selection_methods=["mutual_info", "importance"],
        correlation_threshold=0.9,
        leakage_guard="warn",
        verbose=True,
    )
    X_train_fe = engineer.fit_transform(X_train, y_train, target_name=target_col, apply_selection=True).fillna(0)
    X_test_fe = engineer.transform(X_test).fillna(0)

    common_cols = [col for col in X_train_fe.columns if col in X_test_fe.columns]
    X_train_fe = X_train_fe[common_cols]
    X_test_fe = X_test_fe[common_cols]

    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train_fe, y_train)
    engineered_auc = roc_auc_score(y_test, model.predict_proba(X_test_fe)[:, 1])

    print(f"Temporal baseline ROC-AUC: {baseline_auc:.4f}")
    print(f"Engineered ROC-AUC:       {engineered_auc:.4f}")
    print(f"Delta:                    {engineered_auc - baseline_auc:+.4f}")
    print(f"Selected features:        {len(common_cols)}")


if __name__ == "__main__":
    main()

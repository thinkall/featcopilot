"""
Example: Basic Feature Engineering with FeatCopilot

This example demonstrates basic usage of FeatCopilot for
automated feature engineering on tabular data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Import FeatCopilot
from featcopilot import AutoFeatureEngineer


def create_sample_data(n_samples=1000):
    """
    Create synthetic customer churn dataset with interaction effects.

    The target depends on feature interactions and ratios that
    benefit from feature engineering.
    """
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n_samples),
            "income": np.random.exponential(50000, n_samples) + 20000,
            "tenure_months": np.random.randint(1, 120, n_samples),
            "monthly_charges": np.random.uniform(20, 150, n_samples),
            "total_charges": np.random.exponential(2000, n_samples) + 100,
            "contract_length": np.random.choice([1, 12, 24], n_samples),
            "num_products": np.random.randint(1, 6, n_samples),
            "support_tickets": np.random.poisson(2, n_samples),
        }
    )

    # Create target based on MULTIPLICATIVE INTERACTIONS and RATIOS
    # Tree models struggle with these without explicit feature engineering

    # Key ratio: monthly charges as percentage of monthly income
    charge_ratio = data["monthly_charges"] / (data["income"] / 12 + 1)

    # Key ratio: average charge per product (value density)
    charge_per_product = data["monthly_charges"] / (data["num_products"] + 0.5)

    # Key interaction: tickets per month of tenure (complaint rate)
    complaint_rate = data["support_tickets"] / (data["tenure_months"] + 1)

    # Key interaction: product of normalized age and contract
    # (young people on short contracts churn more)
    age_norm = (data["age"] - 18) / (80 - 18)
    contract_norm = data["contract_length"] / 24
    loyalty_score = age_norm * contract_norm

    # Multiplicative interaction (hard for trees without engineering)
    # High charges + low tenure + tickets = very high churn
    risk_product = charge_ratio * complaint_rate * 10

    churn_prob = (
        0.1
        + 1.5 * charge_ratio  # High relative charges
        + 0.003 * charge_per_product  # Paying more per product
        + 2.0 * complaint_rate  # Complaints early = bad
        - 0.4 * loyalty_score  # Older + long contract = loyal
        + 0.8 * risk_product  # Multiplicative risk
        - 0.01 * data["tenure_months"]  # Tenure helps
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.98)
    data["churn"] = (np.random.random(n_samples) < churn_prob).astype(int)

    return data


def main():
    print("=" * 60)
    print("FeatCopilot Basic Example")
    print("=" * 60)

    # Create sample data
    print("\n1. Creating sample customer data...")
    data = create_sample_data(1000)
    X = data.drop("churn", axis=1)
    y = data["churn"]

    print(f"   - Samples: {len(X)}")
    print(f"   - Original features: {len(X.columns)}")
    print(f"   - Features: {list(X.columns)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline model (no feature engineering)
    print("\n2. Training baseline model (no feature engineering)...")
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_preds)
    print(f"   - Baseline ROC-AUC: {baseline_auc:.4f}")

    # Feature engineering with FeatCopilot
    print("\n3. Applying FeatCopilot feature engineering...")
    engineer = AutoFeatureEngineer(
        engines=["tabular"],
        max_features=50,
        selection_methods=["mutual_info", "importance"],
        correlation_threshold=0.95,
        verbose=True,
    )

    # Fit and transform training data
    X_train_fe = engineer.fit_transform(
        X_train,
        y_train,
        task_description="Predict customer churn",
        apply_selection=True,
    )

    # Transform test data
    X_test_fe = engineer.transform(X_test)

    # Select same features for test set
    selected_features = [c for c in X_train_fe.columns if c in X_test_fe.columns]
    X_test_fe = X_test_fe[selected_features]

    print(f"\n   - Generated features: {len(engineer.get_feature_names())}")
    print(f"   - Selected features: {len(X_train_fe.columns)}")

    # Handle any remaining NaN values
    X_train_fe = X_train_fe.fillna(0)
    X_test_fe = X_test_fe.fillna(0)

    # Model with engineered features
    print("\n4. Training model with engineered features...")
    fe_model = RandomForestClassifier(n_estimators=100, random_state=42)
    fe_model.fit(X_train_fe, y_train)
    fe_preds = fe_model.predict_proba(X_test_fe)[:, 1]
    fe_auc = roc_auc_score(y_test, fe_preds)
    print(f"   - Feature-Engineered ROC-AUC: {fe_auc:.4f}")

    # Results comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"Baseline ROC-AUC:            {baseline_auc:.4f}")
    print(f"Feature-Engineered ROC-AUC:  {fe_auc:.4f}")
    print(f"Improvement:                 {(fe_auc - baseline_auc) * 100:.2f}%")

    # Show top features
    if engineer.feature_importances_:
        print("\n5. Top 10 feature importance scores:")
        importances = sorted(engineer.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (name, score) in enumerate(importances, 1):
            print(f"   {i:2d}. {name}: {score:.4f}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

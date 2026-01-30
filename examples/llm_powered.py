"""
Example: LLM-Powered Feature Engineering with FeatCopilot

This example demonstrates the unique LLM-powered capabilities
of FeatCopilot using GitHub Copilot SDK.

NOTE: This requires the copilot-sdk package and GitHub Copilot
CLI to be installed and authenticated.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from featcopilot import AutoFeatureEngineer


def create_healthcare_data(n_samples=500):
    """Create synthetic healthcare dataset."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "age": np.random.randint(20, 90, n_samples),
            "bmi": np.random.normal(26, 5, n_samples),
            "blood_pressure_systolic": np.random.normal(120, 20, n_samples),
            "blood_pressure_diastolic": np.random.normal(80, 12, n_samples),
            "cholesterol_total": np.random.normal(200, 40, n_samples),
            "cholesterol_hdl": np.random.normal(55, 15, n_samples),
            "cholesterol_ldl": np.random.normal(120, 35, n_samples),
            "glucose_fasting": np.random.normal(100, 25, n_samples),
            "hba1c": np.random.normal(5.5, 1.2, n_samples),
            "smoking_years": np.random.exponential(5, n_samples),
            "exercise_hours_weekly": np.random.exponential(3, n_samples),
        }
    )

    # Create diabetes risk target
    risk = (
        0.01 * (data["age"] - 40)
        + 0.02 * (data["bmi"] - 25)
        + 0.01 * data["glucose_fasting"]
        + 0.1 * data["hba1c"]
        + 0.01 * data["smoking_years"]
        - 0.02 * data["exercise_hours_weekly"]
    )
    risk = 1 / (1 + np.exp(-risk))
    data["diabetes_risk"] = (np.random.random(n_samples) < risk).astype(int)

    return data


def main():
    print("=" * 70)
    print("FeatCopilot LLM-Powered Feature Engineering Example")
    print("=" * 70)

    # Create sample data
    print("\n1. Creating healthcare dataset...")
    data = create_healthcare_data(500)
    X = data.drop("diabetes_risk", axis=1)
    y = data["diabetes_risk"]

    print(f"   - Samples: {len(X)}")
    print(f"   - Features: {list(X.columns)}")

    # Define column descriptions for LLM understanding
    column_descriptions = {
        "age": "Patient age in years",
        "bmi": "Body Mass Index (weight in kg / height in m squared)",
        "blood_pressure_systolic": "Systolic blood pressure in mmHg",
        "blood_pressure_diastolic": "Diastolic blood pressure in mmHg",
        "cholesterol_total": "Total cholesterol level in mg/dL",
        "cholesterol_hdl": "HDL (good) cholesterol in mg/dL",
        "cholesterol_ldl": "LDL (bad) cholesterol in mg/dL",
        "glucose_fasting": "Fasting blood glucose in mg/dL",
        "hba1c": "Hemoglobin A1c percentage (3-month glucose average)",
        "smoking_years": "Number of years patient has smoked",
        "exercise_hours_weekly": "Average hours of exercise per week",
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline model (no feature engineering)
    print("\n2. Training baseline model (no feature engineering)...")
    baseline_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_preds)
    print(f"   - Baseline ROC-AUC: {baseline_auc:.4f}")

    # Feature engineering with LLM
    print("\n3. Applying LLM-powered feature engineering...")
    print("   (Note: If Copilot SDK not available, mock responses will be used)")

    engineer = AutoFeatureEngineer(
        engines=["tabular", "llm"],
        max_features=40,
        llm_config={
            "model": "gpt-5.2",
            "max_suggestions": 15,
            "domain": "healthcare",
        },
        verbose=True,
    )

    X_train_fe = engineer.fit_transform(
        X_train,
        y_train,
        column_descriptions=column_descriptions,
        task_description="Predict diabetes risk based on patient health metrics",
        apply_selection=True,
    )

    X_test_fe = engineer.transform(X_test)

    # Align features
    common_features = [c for c in X_train_fe.columns if c in X_test_fe.columns]
    X_train_fe = X_train_fe[common_features].fillna(0)
    X_test_fe = X_test_fe[common_features].fillna(0)

    print(f"\n   - Final features: {len(X_train_fe.columns)}")

    # Show LLM-generated explanations
    print("\n4. Feature explanations from LLM:")
    explanations = engineer.explain_features()
    for i, (name, explanation) in enumerate(list(explanations.items())[:5], 1):
        print(f"\n   {i}. {name}:")
        print(f"      {explanation[:100]}..." if len(explanation) > 100 else f"      {explanation}")

    # Train model with engineered features
    print("\n5. Training model with engineered features...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_fe, y_train)
    preds = model.predict_proba(X_test_fe)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"   - Engineered ROC-AUC: {auc:.4f}")
    print(f"   - Improvement: {(auc - baseline_auc) / baseline_auc * 100:+.2f}%")

    # Show generated feature code
    print("\n6. Sample generated feature code:")
    feature_code = engineer.get_feature_code()
    for i, (name, code) in enumerate(list(feature_code.items())[:3], 1):
        print(f"\n   {i}. {name}:")
        print(f"      {code}")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

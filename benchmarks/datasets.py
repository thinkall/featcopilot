"""
Benchmark datasets for FeatCopilot evaluation.

Includes both real-world datasets from sklearn and synthetic datasets
commonly used in feature engineering benchmarks.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    make_classification,
    make_regression,
)


def load_diabetes_dataset():
    """
    Diabetes dataset - regression task.
    Used by: AutoFeat, OpenFE, Featuretools benchmarks.
    """
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, "regression", "Diabetes (sklearn)"


def load_breast_cancer_dataset():
    """
    Breast cancer dataset - binary classification.
    Used by: AutoFeat, CAAFE, OpenFE benchmarks.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, "classification", "Breast Cancer (sklearn)"


def load_california_housing_dataset():
    """
    California housing dataset - regression task.
    Used by: Featuretools, OpenFE benchmarks.
    """
    try:
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        return X, y, "regression", "California Housing (sklearn)"
    except Exception:
        # Fallback: create synthetic housing data
        np.random.seed(42)
        n = 1000
        X = pd.DataFrame(
            {
                "MedInc": np.random.lognormal(1.5, 0.5, n),
                "HouseAge": np.random.uniform(1, 52, n),
                "AveRooms": np.random.uniform(2, 10, n),
                "AveBedrms": np.random.uniform(1, 5, n),
                "Population": np.random.lognormal(6, 1, n),
                "AveOccup": np.random.uniform(1, 6, n),
                "Latitude": np.random.uniform(32, 42, n),
                "Longitude": np.random.uniform(-124, -114, n),
            }
        )
        y = pd.Series(
            X["MedInc"] * 0.5 + X["AveRooms"] * 0.1 - X["HouseAge"] * 0.01 + np.random.normal(0, 0.5, n), name="target"
        )
        return X, y, "regression", "California Housing (synthetic fallback)"


def create_credit_risk_dataset(n_samples=2000, random_state=42):
    """
    Synthetic credit risk dataset - binary classification.
    Mimics datasets used in financial ML benchmarks.
    """
    np.random.seed(random_state)

    X = pd.DataFrame(
        {
            "age": np.random.randint(18, 70, n_samples),
            "income": np.random.lognormal(10.5, 0.8, n_samples),
            "debt": np.random.lognormal(9, 1.2, n_samples),
            "credit_history_months": np.random.randint(0, 360, n_samples),
            "num_credit_cards": np.random.poisson(3, n_samples),
            "num_loans": np.random.poisson(2, n_samples),
            "employment_years": np.random.exponential(5, n_samples).astype(int),
            "savings": np.random.lognormal(8, 1.5, n_samples),
            "monthly_expenses": np.random.lognormal(7.5, 0.5, n_samples),
            "num_dependents": np.random.poisson(1, n_samples),
        }
    )

    # Create target based on realistic credit risk factors
    debt_to_income = X["debt"] / (X["income"] + 1)
    credit_score = (
        0.3 * (X["credit_history_months"] / 360)
        + 0.25 * (1 - np.clip(debt_to_income, 0, 2) / 2)
        + 0.2 * (X["employment_years"] / 20)
        + 0.15 * (X["savings"] / X["income"])
        + 0.1 * (1 - X["num_loans"] / 10)
    )
    credit_score = np.clip(credit_score, 0, 1)
    noise = np.random.normal(0, 0.15, n_samples)
    y = pd.Series((credit_score + noise > 0.5).astype(int), name="target")

    return X, y, "classification", "Credit Risk (synthetic)"


def create_housing_price_dataset(n_samples=1500, random_state=42):
    """
    Synthetic housing price dataset - regression task.
    Used to test feature interactions and transformations.
    """
    np.random.seed(random_state)

    X = pd.DataFrame(
        {
            "square_feet": np.random.randint(500, 5000, n_samples),
            "bedrooms": np.random.randint(1, 6, n_samples),
            "bathrooms": np.random.randint(1, 4, n_samples),
            "lot_size": np.random.lognormal(8, 0.5, n_samples),
            "year_built": np.random.randint(1950, 2024, n_samples),
            "garage_spaces": np.random.randint(0, 4, n_samples),
            "distance_to_city": np.random.exponential(10, n_samples),
            "school_rating": np.random.randint(1, 11, n_samples),
            "crime_rate": np.random.exponential(5, n_samples),
            "property_tax_rate": np.random.uniform(0.5, 3.0, n_samples),
        }
    )

    # Create target with realistic interactions
    base_price = 50000
    price = (
        base_price
        + X["square_feet"] * 150
        + X["bedrooms"] * 15000
        + X["bathrooms"] * 20000
        + np.log1p(X["lot_size"]) * 5000
        + (2024 - X["year_built"]) * (-500)
        + X["garage_spaces"] * 10000
        - X["distance_to_city"] * 3000
        + X["school_rating"] * 8000
        - X["crime_rate"] * 2000
        - X["property_tax_rate"] * 10000
        + X["square_feet"] * X["school_rating"] * 5  # Interaction
    )
    noise = np.random.normal(0, 30000, n_samples)
    y = pd.Series(np.maximum(price + noise, 50000), name="target")

    return X, y, "regression", "Housing Price (synthetic)"


def create_customer_churn_dataset(n_samples=2000, random_state=42):
    """
    Synthetic customer churn dataset - binary classification.
    Common in ML benchmarks and Kaggle competitions.
    """
    np.random.seed(random_state)

    X = pd.DataFrame(
        {
            "tenure_months": np.random.randint(1, 72, n_samples),
            "monthly_charges": np.random.uniform(20, 120, n_samples),
            "total_charges": np.zeros(n_samples),  # Will be computed
            "num_products": np.random.randint(1, 6, n_samples),
            "num_support_tickets": np.random.poisson(2, n_samples),
            "avg_usage_hours": np.random.exponential(20, n_samples),
            "contract_length": np.random.choice([1, 12, 24], n_samples),
            "payment_delay_days": np.random.exponential(3, n_samples),
            "satisfaction_score": np.random.randint(1, 6, n_samples),
            "age": np.random.randint(18, 80, n_samples),
        }
    )

    # Compute total charges based on tenure and monthly charges
    X["total_charges"] = X["tenure_months"] * X["monthly_charges"] * np.random.uniform(0.9, 1.1, n_samples)

    # Create target based on churn factors
    churn_prob = (
        0.3 * (1 - X["tenure_months"] / 72)
        + 0.2 * (X["monthly_charges"] / 120)
        + 0.15 * (X["num_support_tickets"] / 10)
        - 0.15 * (X["num_products"] / 5)
        - 0.1 * (X["contract_length"] / 24)
        + 0.1 * (X["payment_delay_days"] / 30)
        - 0.1 * (X["satisfaction_score"] / 5)
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    y = pd.Series((np.random.random(n_samples) < churn_prob).astype(int), name="target")

    return X, y, "classification", "Customer Churn (synthetic)"


def create_medical_diagnosis_dataset(n_samples=1500, random_state=42):
    """
    Synthetic medical diagnosis dataset - binary classification.
    Tests domain-specific feature engineering.
    """
    np.random.seed(random_state)

    X = pd.DataFrame(
        {
            "age": np.random.randint(20, 85, n_samples),
            "bmi": np.random.normal(26, 5, n_samples),
            "blood_pressure_systolic": np.random.normal(120, 20, n_samples),
            "blood_pressure_diastolic": np.random.normal(80, 12, n_samples),
            "cholesterol_total": np.random.normal(200, 40, n_samples),
            "cholesterol_hdl": np.random.normal(55, 15, n_samples),
            "cholesterol_ldl": np.random.normal(120, 35, n_samples),
            "glucose_fasting": np.random.normal(100, 25, n_samples),
            "hba1c": np.random.normal(5.5, 1.0, n_samples),
            "heart_rate": np.random.normal(72, 12, n_samples),
            "smoking_years": np.maximum(np.random.exponential(5, n_samples), 0),
            "exercise_hours_weekly": np.random.exponential(3, n_samples),
        }
    )

    # Create target based on cardiovascular risk factors
    risk_score = (
        0.15 * (X["age"] / 85)
        + 0.15 * np.clip((X["bmi"] - 18.5) / 20, 0, 1)
        + 0.1 * np.clip((X["blood_pressure_systolic"] - 90) / 100, 0, 1)
        + 0.1 * np.clip((X["cholesterol_total"] - 150) / 150, 0, 1)
        - 0.1 * np.clip((X["cholesterol_hdl"] - 30) / 50, 0, 1)
        + 0.1 * np.clip((X["cholesterol_ldl"] - 70) / 130, 0, 1)
        + 0.1 * np.clip((X["glucose_fasting"] - 70) / 100, 0, 1)
        + 0.1 * np.clip((X["hba1c"] - 4) / 4, 0, 1)
        + 0.05 * np.clip(X["smoking_years"] / 30, 0, 1)
        - 0.05 * np.clip(X["exercise_hours_weekly"] / 10, 0, 1)
    )
    noise = np.random.normal(0, 0.1, n_samples)
    y = pd.Series((risk_score + noise > 0.4).astype(int), name="target")

    return X, y, "classification", "Medical Diagnosis (synthetic)"


def create_complex_regression_dataset(n_samples=2000, n_features=15, random_state=42):
    """
    Complex regression with non-linear relationships.
    Tests feature transformation capabilities.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        noise=10,
        random_state=random_state,
    )

    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")

    return X, y, "regression", "Complex Regression (synthetic)"


def create_complex_classification_dataset(n_samples=2000, n_features=15, random_state=42):
    """
    Complex classification with class imbalance.
    Tests feature selection and engineering.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        random_state=random_state,
    )

    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    y = pd.Series(y, name="target")

    return X, y, "classification", "Complex Classification (synthetic)"


def get_all_datasets():
    """Return all benchmark datasets."""
    return [
        load_diabetes_dataset,
        load_breast_cancer_dataset,
        load_california_housing_dataset,
        create_credit_risk_dataset,
        create_housing_price_dataset,
        create_customer_churn_dataset,
        create_medical_diagnosis_dataset,
        create_complex_regression_dataset,
        create_complex_classification_dataset,
    ]


def get_dataset_info():
    """Get information about all benchmark datasets."""
    info = []
    for loader in get_all_datasets():
        X, y, task, name = loader()
        info.append(
            {
                "name": name,
                "task": task,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "target_distribution": y.value_counts().to_dict() if task == "classification" else None,
            }
        )
    return info


if __name__ == "__main__":
    print("Benchmark Datasets Summary")
    print("=" * 60)
    for info in get_dataset_info():
        print(f"\n{info['name']}")
        print(f"  Task: {info['task']}")
        print(f"  Samples: {info['n_samples']}")
        print(f"  Features: {info['n_features']}")
        if info["target_distribution"]:
            print(f"  Target dist: {info['target_distribution']}")

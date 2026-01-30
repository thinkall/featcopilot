"""
Benchmark datasets for FeatCopilot evaluation.

Includes real-world datasets (sklearn, Kaggle-style), synthetic datasets,
and time series datasets for comprehensive benchmarking.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    make_classification,
    make_regression,
)

# =============================================================================
# Real-world Datasets (sklearn built-in)
# =============================================================================


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


# =============================================================================
# Kaggle-style Real-world Datasets (embedded data)
# =============================================================================


def load_titanic_dataset():
    """
    Titanic dataset - binary classification (survival prediction).
    Classic Kaggle competition dataset.
    """
    np.random.seed(42)
    n = 891  # Original Titanic size

    # Simulate realistic Titanic-like data
    pclass = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    sex = np.random.choice([0, 1], n, p=[0.35, 0.65])  # 0=female, 1=male
    age = np.clip(np.random.normal(30, 14, n), 0.5, 80)
    sibsp = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01])
    parch = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.76, 0.13, 0.08, 0.01, 0.01, 0.01])
    fare = np.where(
        pclass == 1,
        np.random.exponential(80, n),
        np.where(pclass == 2, np.random.exponential(25, n), np.random.exponential(10, n)),
    )
    embarked = np.random.choice([0, 1, 2], n, p=[0.72, 0.19, 0.09])  # S, C, Q

    X = pd.DataFrame(
        {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
        }
    )

    # Survival probability based on historical patterns
    survival_prob = (
        0.3
        - 0.15 * (pclass - 1) / 2
        - 0.35 * sex
        - 0.005 * np.clip(age - 10, 0, 60)
        + 0.05 * (embarked == 1)
        + 0.02 * (fare / 100)
    )
    survival_prob = np.clip(survival_prob, 0.05, 0.95)
    y = pd.Series((np.random.random(n) < survival_prob).astype(int), name="Survived")

    return X, y, "classification", "Titanic (Kaggle-style)"


def load_house_prices_dataset():
    """
    House Prices dataset - regression (price prediction).
    Based on Kaggle House Prices competition features.
    """
    np.random.seed(42)
    n = 1460  # Original competition size

    X = pd.DataFrame(
        {
            "OverallQual": np.random.randint(1, 11, n),
            "GrLivArea": np.random.randint(500, 4000, n),
            "GarageCars": np.random.choice([0, 1, 2, 3, 4], n, p=[0.05, 0.15, 0.45, 0.30, 0.05]),
            "TotalBsmtSF": np.random.randint(0, 3000, n),
            "FullBath": np.random.choice([0, 1, 2, 3], n, p=[0.01, 0.35, 0.55, 0.09]),
            "YearBuilt": np.random.randint(1900, 2010, n),
            "YearRemodAdd": np.random.randint(1950, 2010, n),
            "LotArea": np.random.lognormal(9.5, 0.5, n).astype(int),
            "Fireplaces": np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.40, 0.12, 0.01]),
            "TotRmsAbvGrd": np.random.randint(3, 12, n),
            "MasVnrArea": np.random.exponential(100, n),
            "BsmtFinSF1": np.random.exponential(400, n),
            "WoodDeckSF": np.random.exponential(80, n),
            "OpenPorchSF": np.random.exponential(40, n),
        }
    )

    # Price based on realistic factors
    price = (
        50000
        + X["OverallQual"] * 15000
        + X["GrLivArea"] * 50
        + X["GarageCars"] * 12000
        + X["TotalBsmtSF"] * 30
        + X["FullBath"] * 8000
        + (X["YearBuilt"] - 1900) * 200
        + X["LotArea"] * 0.5
        + X["Fireplaces"] * 5000
        + X["TotRmsAbvGrd"] * 3000
        + X["MasVnrArea"] * 50
    )
    noise = np.random.normal(0, 20000, n)
    y = pd.Series(np.maximum(price + noise, 50000), name="SalePrice")

    return X, y, "regression", "House Prices (Kaggle-style)"


def load_credit_card_fraud_dataset():
    """
    Credit Card Fraud dataset - binary classification (imbalanced).
    Based on Kaggle Credit Card Fraud Detection.
    """
    np.random.seed(42)
    n = 5000
    fraud_rate = 0.017  # ~1.7% fraud like original

    # PCA-like anonymized features (V1-V28)
    X = pd.DataFrame({f"V{i}": np.random.randn(n) for i in range(1, 29)})
    X["Amount"] = np.random.exponential(90, n)
    X["Time"] = np.cumsum(np.random.exponential(100, n))

    # Create fraud pattern - fraudulent transactions have different V patterns
    fraud_score = (
        0.1 * np.abs(X["V1"])
        + 0.1 * np.abs(X["V2"])
        - 0.1 * X["V3"]
        + 0.05 * X["V4"]
        - 0.15 * X["V14"]
        + 0.1 * (X["Amount"] > 200).astype(float)
    )
    fraud_prob = 1 / (1 + np.exp(-fraud_score + 2))  # Sigmoid
    fraud_prob = fraud_prob * (fraud_rate / fraud_prob.mean())  # Scale to target rate
    y = pd.Series((np.random.random(n) < fraud_prob).astype(int), name="Class")

    return X, y, "classification", "Credit Card Fraud (Kaggle-style)"


def load_bike_sharing_dataset():
    """
    Bike Sharing dataset - regression (demand prediction).
    Based on UCI/Kaggle Bike Sharing Demand.
    """
    np.random.seed(42)
    n = 2000

    hour = np.random.randint(0, 24, n)
    weekday = np.random.randint(0, 7, n)
    month = np.random.randint(1, 13, n)
    season = (month % 12 // 3) + 1
    holiday = np.random.choice([0, 1], n, p=[0.97, 0.03])
    workingday = ((weekday < 5) & (holiday == 0)).astype(int)
    weather = np.random.choice([1, 2, 3, 4], n, p=[0.65, 0.25, 0.09, 0.01])
    temp = 10 + 15 * np.sin(np.pi * (month - 1) / 6) + np.random.normal(0, 5, n)
    humidity = np.clip(np.random.normal(60, 20, n), 10, 100)
    windspeed = np.random.exponential(12, n)

    X = pd.DataFrame(
        {
            "hour": hour,
            "weekday": weekday,
            "month": month,
            "season": season,
            "holiday": holiday,
            "workingday": workingday,
            "weather": weather,
            "temp": temp,
            "humidity": humidity,
            "windspeed": windspeed,
        }
    )

    # Demand based on realistic patterns
    base_demand = 150
    demand = (
        base_demand
        + 50 * np.sin(np.pi * hour / 12)  # Peak at noon
        + 30 * (hour >= 7) * (hour <= 9) * workingday  # Morning commute
        + 40 * (hour >= 17) * (hour <= 19) * workingday  # Evening commute
        + 20 * ((weekday >= 5) & (hour >= 10) & (hour <= 16))  # Weekend midday
        + 5 * temp
        - 1 * humidity
        - 3 * windspeed
        - 30 * (weather >= 3)  # Bad weather
        - 50 * (weather == 4)  # Very bad weather
    )
    demand = np.maximum(demand + np.random.normal(0, 30, n), 0)
    y = pd.Series(demand.astype(int), name="count")

    return X, y, "regression", "Bike Sharing (Kaggle-style)"


def load_employee_attrition_dataset():
    """
    Employee Attrition dataset - binary classification.
    Based on IBM HR Analytics.
    """
    np.random.seed(42)
    n = 1470

    age = np.random.randint(18, 60, n)
    years_at_company = np.minimum(np.random.exponential(5, n), age - 18).astype(int)
    monthly_income = 2000 + age * 80 + years_at_company * 200 + np.random.normal(0, 1000, n)
    job_satisfaction = np.random.randint(1, 5, n)
    work_life_balance = np.random.randint(1, 5, n)
    overtime = np.random.choice([0, 1], n, p=[0.72, 0.28])
    distance_from_home = np.random.randint(1, 30, n)
    num_companies_worked = np.random.poisson(2, n)
    percent_salary_hike = np.random.randint(11, 25, n)
    training_times_last_year = np.random.randint(0, 7, n)
    years_since_last_promotion = np.minimum(np.random.exponential(2, n), years_at_company).astype(int)

    X = pd.DataFrame(
        {
            "Age": age,
            "YearsAtCompany": years_at_company,
            "MonthlyIncome": monthly_income,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "OverTime": overtime,
            "DistanceFromHome": distance_from_home,
            "NumCompaniesWorked": num_companies_worked,
            "PercentSalaryHike": percent_salary_hike,
            "TrainingTimesLastYear": training_times_last_year,
            "YearsSinceLastPromotion": years_since_last_promotion,
        }
    )

    # Attrition probability
    attrition_prob = (
        0.16  # Base rate
        - 0.003 * age
        - 0.01 * years_at_company
        - 0.03 * job_satisfaction
        - 0.02 * work_life_balance
        + 0.15 * overtime
        + 0.005 * distance_from_home
        + 0.02 * num_companies_worked
        - 0.005 * percent_salary_hike
        + 0.02 * years_since_last_promotion
    )
    attrition_prob = np.clip(attrition_prob, 0.02, 0.8)
    y = pd.Series((np.random.random(n) < attrition_prob).astype(int), name="Attrition")

    return X, y, "classification", "Employee Attrition (IBM HR)"


# =============================================================================
# Synthetic Datasets (for controlled benchmarking)
# =============================================================================


def create_credit_risk_dataset(n_samples=2000, random_state=42):
    """Synthetic credit risk dataset - binary classification."""
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


def create_medical_diagnosis_dataset(n_samples=1500, random_state=42):
    """Synthetic medical diagnosis dataset - binary classification."""
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
    """Complex regression with non-linear relationships."""
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
    """Complex classification with class imbalance."""
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


# =============================================================================
# Time Series Datasets
# =============================================================================


def create_energy_consumption_timeseries(n_samples=2000, random_state=42):
    """Energy consumption time series - regression (forecasting)."""
    np.random.seed(random_state)

    hour = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
    day_of_week = np.tile(np.repeat(np.arange(7), 24), n_samples // 168 + 1)[:n_samples]
    month = np.tile(np.repeat(np.arange(1, 13), 24 * 30), n_samples // (24 * 360) + 1)[:n_samples]

    temperature = 15 + 10 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 3, n_samples)
    humidity = np.clip(np.random.normal(60, 15, n_samples), 20, 95)
    is_holiday = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])

    lag_1 = np.random.normal(50, 10, n_samples)
    lag_24 = np.random.normal(50, 10, n_samples)
    lag_168 = np.random.normal(50, 10, n_samples)

    X = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "temperature": temperature,
            "humidity": humidity,
            "is_holiday": is_holiday,
            "lag_1h": lag_1,
            "lag_24h": lag_24,
            "lag_168h": lag_168,
        }
    )

    base_load = 30
    consumption = (
        base_load
        + 20 * np.sin(np.pi * hour / 12)
        + 10 * (hour >= 8) * (hour <= 20)
        + 15 * (hour >= 18) * (hour <= 21)
        - 5 * (day_of_week >= 5)
        + 10 * np.abs(temperature - 20) / 10
        - 0.1 * humidity
        - 10 * is_holiday
        + 0.3 * lag_1
        + 0.2 * lag_24
        + 0.1 * lag_168
    )
    consumption = np.maximum(consumption + np.random.normal(0, 5, n_samples), 5)
    y = pd.Series(consumption, name="energy_kwh")

    return X, y, "timeseries_regression", "Energy Consumption (time series)"


def create_stock_price_timeseries(n_samples=1500, random_state=42):
    """Stock price movement - binary classification (up/down)."""
    np.random.seed(random_state)

    returns = np.random.normal(0.0005, 0.02, n_samples)
    price = 100 * np.cumprod(1 + returns)

    sma_5 = pd.Series(price).rolling(5).mean().fillna(price[0]).values
    sma_20 = pd.Series(price).rolling(20).mean().fillna(price[0]).values
    ema_12 = pd.Series(price).ewm(span=12).mean().fillna(price[0]).values
    rsi = 50 + np.random.normal(0, 15, n_samples)
    volume = np.random.lognormal(15, 0.5, n_samples)
    volatility = pd.Series(returns).rolling(20).std().fillna(0.02).values

    X = pd.DataFrame(
        {
            "price": price,
            "sma_5": sma_5,
            "sma_20": sma_20,
            "ema_12": ema_12,
            "rsi": np.clip(rsi, 0, 100),
            "volume": volume,
            "volatility": volatility,
            "price_to_sma5": price / sma_5,
            "price_to_sma20": price / sma_20,
            "sma5_to_sma20": sma_5 / sma_20,
        }
    )

    future_returns = np.roll(returns, -1)
    future_returns[-1] = 0
    y = pd.Series((future_returns > 0).astype(int), name="direction")

    return X, y, "timeseries_classification", "Stock Price Direction (time series)"


def create_website_traffic_timeseries(n_samples=2000, random_state=42):
    """Website traffic prediction - regression."""
    np.random.seed(random_state)

    hour = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
    day_of_week = np.tile(np.repeat(np.arange(7), 24), n_samples // 168 + 1)[:n_samples]
    is_weekend = (day_of_week >= 5).astype(int)

    marketing_spend = np.random.exponential(100, n_samples)
    social_mentions = np.random.poisson(20, n_samples)
    competitor_traffic = np.random.lognormal(8, 0.3, n_samples)

    lag_1h = np.random.lognormal(7, 0.5, n_samples)
    lag_24h = np.random.lognormal(7, 0.5, n_samples)
    rolling_avg_7d = np.random.lognormal(7, 0.3, n_samples)

    X = pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "marketing_spend": marketing_spend,
            "social_mentions": social_mentions,
            "competitor_traffic": competitor_traffic,
            "lag_1h": lag_1h,
            "lag_24h": lag_24h,
            "rolling_avg_7d": rolling_avg_7d,
        }
    )

    base_traffic = 1000
    traffic = (
        base_traffic
        + 500 * np.sin(np.pi * hour / 12)
        + 300 * (hour >= 9) * (hour <= 18)
        - 200 * is_weekend
        + 2 * marketing_spend
        + 5 * social_mentions
        + 0.2 * lag_1h
        + 0.3 * lag_24h
        + 0.4 * rolling_avg_7d
    )
    traffic = np.maximum(traffic + np.random.normal(0, 100, n_samples), 50)
    y = pd.Series(traffic.astype(int), name="page_views")

    return X, y, "timeseries_regression", "Website Traffic (time series)"


# =============================================================================
# Dataset Registry
# =============================================================================


def get_all_datasets():
    """Return all benchmark datasets (excluding time series)."""
    return [
        # Real-world (sklearn)
        load_diabetes_dataset,
        load_breast_cancer_dataset,
        # Kaggle-style
        load_titanic_dataset,
        load_house_prices_dataset,
        load_credit_card_fraud_dataset,
        load_bike_sharing_dataset,
        load_employee_attrition_dataset,
        # Synthetic
        create_credit_risk_dataset,
        create_medical_diagnosis_dataset,
        create_complex_regression_dataset,
        create_complex_classification_dataset,
    ]


def get_timeseries_datasets():
    """Return time series benchmark datasets."""
    return [
        create_energy_consumption_timeseries,
        create_stock_price_timeseries,
        create_website_traffic_timeseries,
    ]


def get_dataset_info():
    """Get information about all benchmark datasets."""
    info = []
    for loader in get_all_datasets() + get_timeseries_datasets():
        X, y, task, name = loader()
        info.append(
            {
                "name": name,
                "task": task,
                "n_samples": len(X),
                "n_features": X.shape[1],
                "target_distribution": y.value_counts().to_dict() if "classification" in task else None,
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

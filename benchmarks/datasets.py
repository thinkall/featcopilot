"""
Benchmark datasets for FeatCopilot evaluation.

Includes real-world datasets (Kaggle-style), synthetic datasets,
time series datasets, and text/semantic datasets for comprehensive benchmarking.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification,
    make_regression,
)

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


def create_sensor_anomaly_timeseries(n_samples=2000, random_state=42):
    """
    Industrial sensor data - regression (predict equipment efficiency).
    Designed with polynomial and interaction effects that benefit from FE.
    """
    np.random.seed(random_state)

    # Time features
    hour = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
    shift = np.where(hour < 8, 0, np.where(hour < 16, 1, 2))  # 3 shifts

    # Sensor readings
    temperature = 50 + 20 * np.random.random(n_samples)  # 50-70°C
    pressure = 100 + 50 * np.random.random(n_samples)  # 100-150 PSI
    vibration = 0.1 + 0.4 * np.random.random(n_samples)  # 0.1-0.5 mm/s
    humidity = 30 + 40 * np.random.random(n_samples)  # 30-70%
    rpm = 1000 + 500 * np.random.random(n_samples)  # 1000-1500 RPM
    power_input = 50 + 30 * np.random.random(n_samples)  # 50-80 kW

    X = pd.DataFrame(
        {
            "hour": hour,
            "shift": shift,
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration,
            "humidity": humidity,
            "rpm": rpm,
            "power_input": power_input,
        }
    )

    # Efficiency depends on POLYNOMIAL and INTERACTION terms
    # This is designed to benefit from feature engineering!
    efficiency = (
        85  # Base efficiency
        - 0.5 * (temperature - 60) ** 2 / 100  # Quadratic temp effect (optimal at 60)
        - 0.3 * (pressure - 125) ** 2 / 100  # Quadratic pressure effect (optimal at 125)
        - 10 * vibration**2  # Quadratic vibration penalty
        + 0.01 * temperature * pressure / 100  # Temp-pressure interaction
        - 0.05 * vibration * rpm / 100  # Vibration-RPM interaction
        + 0.1 * np.sqrt(power_input)  # Sqrt transform benefit
        - 0.01 * humidity * temperature / 100  # Humidity-temp interaction
        + 2 * (shift == 1)  # Day shift bonus
    )
    efficiency = np.clip(efficiency + np.random.normal(0, 2, n_samples), 50, 100)
    y = pd.Series(efficiency, name="efficiency_pct")

    return X, y, "timeseries_regression", "Sensor Efficiency (time series)"


def create_retail_demand_timeseries(n_samples=2000, random_state=42):
    """
    Retail demand forecasting - regression.
    Has interaction and polynomial effects that benefit from FE.
    """
    np.random.seed(random_state)

    # Time features
    day_of_week = np.tile(np.arange(7), n_samples // 7 + 1)[:n_samples]
    week_of_year = np.tile(np.repeat(np.arange(1, 53), 7), n_samples // 364 + 1)[:n_samples]
    is_weekend = (day_of_week >= 5).astype(int)

    # Business features
    price = 10 + 20 * np.random.random(n_samples)  # $10-30
    promotion_discount = np.random.choice([0, 0.1, 0.2, 0.3], n_samples, p=[0.6, 0.2, 0.15, 0.05])
    competitor_price = 10 + 20 * np.random.random(n_samples)
    inventory_level = 50 + 150 * np.random.random(n_samples)
    marketing_score = np.random.randint(1, 11, n_samples)  # 1-10

    # Lag features
    lag_1d = 50 + 100 * np.random.random(n_samples)
    lag_7d = 50 + 100 * np.random.random(n_samples)

    X = pd.DataFrame(
        {
            "day_of_week": day_of_week,
            "week_of_year": week_of_year,
            "is_weekend": is_weekend,
            "price": price,
            "promotion_discount": promotion_discount,
            "competitor_price": competitor_price,
            "inventory_level": inventory_level,
            "marketing_score": marketing_score,
            "lag_1d": lag_1d,
            "lag_7d": lag_7d,
        }
    )

    # Demand with POLYNOMIAL and INTERACTION effects
    price_ratio = price / (competitor_price + 1)
    effective_price = price * (1 - promotion_discount)

    demand = (
        100  # Base demand
        - 5 * (effective_price - 15) ** 2 / 10  # Quadratic price sensitivity (optimal ~$15)
        + 30 * promotion_discount * marketing_score  # Promo × marketing interaction
        + 20 * (1 - price_ratio) * (price_ratio < 1)  # Price advantage interaction
        + 15 * is_weekend * (1 + promotion_discount)  # Weekend × promo interaction
        + 10 * np.sin(2 * np.pi * week_of_year / 52)  # Seasonality
        + 0.3 * lag_1d  # Lag effects
        + 0.2 * lag_7d
        + 0.05 * np.sqrt(inventory_level) * marketing_score  # Inventory × marketing
    )
    demand = np.maximum(demand + np.random.normal(0, 10, n_samples), 5)
    y = pd.Series(demand.astype(int), name="units_sold")

    return X, y, "timeseries_regression", "Retail Demand (time series)"


def create_server_latency_timeseries(n_samples=2000, random_state=42):
    """
    Server response latency prediction - regression.
    Designed with log/sqrt transforms and interactions that benefit from FE.
    """
    np.random.seed(random_state)

    # Time features
    hour = np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples]
    is_peak = ((hour >= 9) & (hour <= 11) | (hour >= 14) & (hour <= 16)).astype(int)

    # Server metrics
    cpu_usage = 20 + 60 * np.random.random(n_samples)  # 20-80%
    memory_usage = 30 + 50 * np.random.random(n_samples)  # 30-80%
    active_connections = np.random.poisson(100, n_samples)
    request_rate = 50 + 200 * np.random.random(n_samples)  # req/s
    db_connections = np.random.poisson(20, n_samples)
    cache_hit_rate = 0.5 + 0.4 * np.random.random(n_samples)  # 50-90%

    X = pd.DataFrame(
        {
            "hour": hour,
            "is_peak": is_peak,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "active_connections": active_connections,
            "request_rate": request_rate,
            "db_connections": db_connections,
            "cache_hit_rate": cache_hit_rate,
        }
    )

    # Latency with LOG, SQRT, and INTERACTION effects
    latency = (
        50  # Base latency (ms)
        + 2 * np.log1p(active_connections) * (1 - cache_hit_rate)  # Log connections × cache miss
        + 0.5 * cpu_usage * memory_usage / 100  # CPU × memory interaction
        + 3 * np.sqrt(request_rate) * is_peak  # Sqrt request × peak interaction
        + 5 * (cpu_usage / 100) ** 2 * 100  # Quadratic CPU penalty at high usage
        + 2 * db_connections * (1 - cache_hit_rate)  # DB × cache miss interaction
        - 20 * cache_hit_rate**2  # Quadratic cache benefit
        + 10 * is_peak * (cpu_usage > 60)  # Peak × high CPU interaction
    )
    latency = np.maximum(latency + np.random.normal(0, 5, n_samples), 10)
    y = pd.Series(latency, name="latency_ms")

    return X, y, "timeseries_regression", "Server Latency (time series)"


# =============================================================================
# Dataset Registry
# =============================================================================


def get_all_datasets():
    """Return all benchmark datasets (excluding time series)."""
    return [
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
    """Return time series benchmark datasets designed to benefit from FE."""
    return [
        create_sensor_anomaly_timeseries,
        create_retail_demand_timeseries,
        create_server_latency_timeseries,
    ]


# =============================================================================
# Text & Semantic Datasets
# =============================================================================


def create_product_reviews_dataset(n_samples=2000, random_state=42):
    """
    Product reviews dataset - sentiment classification with text.
    Tests text feature engineering capabilities.
    """
    np.random.seed(random_state)

    # Product categories
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"]
    category = np.random.choice(categories, n_samples)

    # Generate realistic review texts
    positive_phrases = [
        "excellent product",
        "highly recommend",
        "great quality",
        "amazing value",
        "works perfectly",
        "exceeded expectations",
        "love it",
        "best purchase",
        "fantastic",
        "wonderful",
        "impressed",
        "satisfied customer",
    ]
    negative_phrases = [
        "disappointed",
        "poor quality",
        "waste of money",
        "doesn't work",
        "broken",
        "terrible",
        "avoid",
        "worst purchase",
        "defective",
        "not as described",
        "returned it",
        "very unhappy",
    ]
    neutral_phrases = [
        "okay product",
        "average",
        "nothing special",
        "decent",
        "expected better",
        "it's fine",
        "does the job",
        "acceptable",
    ]

    reviews = []
    sentiments = []

    for i in range(n_samples):
        sentiment_score = np.random.random()
        if sentiment_score > 0.6:
            # Positive review
            n_phrases = np.random.randint(2, 5)
            phrases = np.random.choice(positive_phrases, n_phrases, replace=True)
            review = " ".join(phrases) + ". " + f"Great {category[i].lower()} product!"
            sentiments.append(1)
        elif sentiment_score < 0.3:
            # Negative review
            n_phrases = np.random.randint(2, 4)
            phrases = np.random.choice(negative_phrases, n_phrases, replace=True)
            review = " ".join(phrases) + ". " + f"Bad {category[i].lower()} experience."
            sentiments.append(0)
        else:
            # Neutral review
            n_phrases = np.random.randint(1, 3)
            phrases = np.random.choice(neutral_phrases, n_phrases, replace=True)
            review = " ".join(phrases) + "."
            sentiments.append(np.random.choice([0, 1]))

        reviews.append(review)

    # Numeric features
    price = np.random.lognormal(3.5, 1, n_samples)
    rating = np.clip(
        np.where(
            np.array(sentiments) == 1, np.random.normal(4.2, 0.5, n_samples), np.random.normal(2.5, 0.8, n_samples)
        ),
        1,
        5,
    )
    helpful_votes = np.random.poisson(5, n_samples)
    review_length = np.array([len(r.split()) for r in reviews])

    X = pd.DataFrame(
        {
            "review_text": reviews,
            "category": category,
            "price": price,
            "rating": rating,
            "helpful_votes": helpful_votes,
            "review_length": review_length,
        }
    )

    y = pd.Series(sentiments, name="sentiment")

    return X, y, "text_classification", "Product Reviews (text)"


def create_job_postings_dataset(n_samples=1500, random_state=42):
    """
    Job postings dataset - salary prediction with text descriptions.
    Tests semantic feature extraction from job descriptions.
    """
    np.random.seed(random_state)

    # Job titles and descriptions
    titles = [
        "Software Engineer",
        "Data Scientist",
        "Product Manager",
        "Marketing Manager",
        "Sales Representative",
        "HR Specialist",
        "Financial Analyst",
        "Operations Manager",
        "Customer Support",
        "UX Designer",
        "DevOps Engineer",
        "Business Analyst",
    ]

    skills_map = {
        "Software Engineer": ["Python", "Java", "JavaScript", "SQL", "AWS", "Docker"],
        "Data Scientist": ["Python", "Machine Learning", "SQL", "Statistics", "TensorFlow"],
        "Product Manager": ["Agile", "Roadmap", "Stakeholder Management", "Analytics"],
        "Marketing Manager": ["SEO", "Social Media", "Content Strategy", "Analytics"],
        "Sales Representative": ["CRM", "Negotiation", "Cold Calling", "Pipeline"],
        "HR Specialist": ["Recruiting", "Onboarding", "HRIS", "Employee Relations"],
        "Financial Analyst": ["Excel", "Financial Modeling", "SQL", "Forecasting"],
        "Operations Manager": ["Process Improvement", "Logistics", "KPIs", "Lean"],
        "Customer Support": ["Zendesk", "Communication", "Problem Solving", "Empathy"],
        "UX Designer": ["Figma", "User Research", "Prototyping", "Design Systems"],
        "DevOps Engineer": ["Kubernetes", "CI/CD", "AWS", "Terraform", "Linux"],
        "Business Analyst": ["Requirements", "SQL", "Process Mapping", "Stakeholders"],
    }

    experience_levels = ["Entry", "Mid", "Senior", "Lead", "Director"]
    locations = ["San Francisco", "New York", "Seattle", "Austin", "Remote", "Chicago"]
    company_sizes = ["Startup", "Small", "Medium", "Large", "Enterprise"]

    job_titles = []
    descriptions = []
    experience = []
    location = []
    company_size = []

    for _ in range(n_samples):
        title = np.random.choice(titles)
        exp_level = np.random.choice(experience_levels, p=[0.15, 0.30, 0.30, 0.15, 0.10])
        loc = np.random.choice(locations)
        size = np.random.choice(company_sizes)

        # Generate description
        skills = np.random.choice(skills_map[title], min(3, len(skills_map[title])), replace=False)
        desc = f"{exp_level} {title} position. Required skills: {', '.join(skills)}. "
        desc += f"Location: {loc}. Company size: {size}."

        job_titles.append(title)
        descriptions.append(desc)
        experience.append(experience_levels.index(exp_level))
        location.append(loc)
        company_size.append(size)

    # Calculate salary based on various factors
    base_salaries = {
        "Software Engineer": 120000,
        "Data Scientist": 130000,
        "Product Manager": 140000,
        "Marketing Manager": 100000,
        "Sales Representative": 70000,
        "HR Specialist": 65000,
        "Financial Analyst": 85000,
        "Operations Manager": 95000,
        "Customer Support": 50000,
        "UX Designer": 110000,
        "DevOps Engineer": 135000,
        "Business Analyst": 90000,
    }

    location_multipliers = {
        "San Francisco": 1.3,
        "New York": 1.25,
        "Seattle": 1.2,
        "Austin": 1.0,
        "Remote": 1.05,
        "Chicago": 1.05,
    }

    salaries = []
    for i in range(n_samples):
        base = base_salaries[job_titles[i]]
        loc_mult = location_multipliers[location[i]]
        exp_mult = 1 + experience[i] * 0.2
        size_mult = 1 + company_sizes.index(company_size[i]) * 0.05
        salary = base * loc_mult * exp_mult * size_mult
        salary += np.random.normal(0, salary * 0.1)
        salaries.append(max(salary, 40000))

    X = pd.DataFrame(
        {
            "job_title": job_titles,
            "description": descriptions,
            "experience_level": experience,
            "location": location,
            "company_size": company_size,
        }
    )

    y = pd.Series(salaries, name="salary")

    return X, y, "text_regression", "Job Postings (text)"


def create_news_classification_dataset(n_samples=2500, random_state=42):
    """
    News headlines classification - multi-class with text.
    Categories: Business, Technology, Sports, Entertainment, Politics.
    """
    np.random.seed(random_state)

    categories = ["Business", "Technology", "Sports", "Entertainment", "Politics"]

    headlines_templates = {
        "Business": [
            "Stock market {action} as {company} reports {result}",
            "{company} announces {number}B acquisition of {target}",
            "Fed {action} interest rates amid {condition}",
            "{company} CEO {action} after {event}",
            "Oil prices {action} on {reason}",
        ],
        "Technology": [
            "{company} launches new {product} with {feature}",
            "AI breakthrough: {achievement} announced",
            "{company} faces {issue} concerns over {topic}",
            "New {technology} promises to {benefit}",
            "{company} unveils {product} at {event}",
        ],
        "Sports": [
            "{team} defeats {opponent} {score} in {event}",
            "{player} signs {number}M contract with {team}",
            "{team} advances to {event} finals",
            "{player} breaks {record} record",
            "Injury update: {player} out for {duration}",
        ],
        "Entertainment": [
            "{movie} breaks box office records with {number}M opening",
            "{celebrity} announces {event}",
            "{show} renewed for {number} more seasons",
            "{celebrity} wins {award} at {event}",
            "New {genre} film from {director} premieres",
        ],
        "Politics": [
            "{politician} proposes new {policy} legislation",
            "Election results: {party} wins {location}",
            "{country} and {country2} sign {agreement}",
            "Debate: candidates clash over {topic}",
            "{politician} addresses {issue} in speech",
        ],
    }

    fill_values = {
        "action": ["rises", "falls", "surges", "drops", "stabilizes"],
        "company": ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta"],
        "result": ["strong earnings", "weak guidance", "record profits", "losses"],
        "number": ["1", "2", "5", "10", "50", "100"],
        "target": ["startup", "competitor", "tech firm"],
        "condition": ["inflation concerns", "economic uncertainty", "growth outlook"],
        "event": ["scandal", "announcement", "quarterly report"],
        "reason": ["supply concerns", "demand surge", "geopolitical tensions"],
        "product": ["smartphone", "AI assistant", "tablet", "smartwatch", "laptop"],
        "feature": ["revolutionary AI", "longer battery", "better camera"],
        "achievement": ["language model", "protein folding", "autonomous driving"],
        "issue": ["privacy", "antitrust", "security"],
        "topic": ["data collection", "market dominance", "user safety"],
        "technology": ["quantum computing", "blockchain", "AR glasses"],
        "benefit": ["revolutionize healthcare", "transform education", "enhance productivity"],
        "team": ["Lakers", "Yankees", "Patriots", "Warriors", "Chiefs"],
        "opponent": ["Celtics", "Red Sox", "Eagles", "Suns", "Ravens"],
        "score": ["3-2", "24-21", "105-98", "2-1"],
        "player": ["LeBron", "Brady", "Messi", "Curry", "Mahomes"],
        "record": ["scoring", "passing", "home run", "touchdown"],
        "duration": ["6 weeks", "season", "2 months"],
        "movie": ["Avatar 3", "Marvel film", "Pixar movie", "Nolan epic"],
        "celebrity": ["Taylor Swift", "Tom Hanks", "Beyoncé", "DiCaprio"],
        "show": ["Stranger Things", "The Crown", "Succession"],
        "award": ["Oscar", "Emmy", "Grammy", "Golden Globe"],
        "genre": ["sci-fi", "drama", "comedy", "action"],
        "director": ["Spielberg", "Nolan", "Scorsese", "Tarantino"],
        "politician": ["President", "Senator", "Governor", "Mayor"],
        "policy": ["healthcare", "tax", "climate", "immigration"],
        "party": ["Democrats", "Republicans"],
        "location": ["Senate seat", "key state", "local election"],
        "country": ["US", "China", "EU", "UK"],
        "country2": ["Japan", "India", "Canada", "Australia"],
        "agreement": ["trade deal", "climate accord", "security pact"],
    }

    headlines = []
    labels = []
    word_counts = []
    has_numbers = []

    for _ in range(n_samples):
        cat = np.random.choice(categories)
        template = np.random.choice(headlines_templates[cat])

        # Fill in template
        headline = template
        for key, values in fill_values.items():
            if "{" + key + "}" in headline:
                headline = headline.replace("{" + key + "}", np.random.choice(values), 1)

        headlines.append(headline)
        labels.append(categories.index(cat))
        word_counts.append(len(headline.split()))
        has_numbers.append(1 if any(c.isdigit() for c in headline) else 0)

    # Additional features
    hour_published = np.random.randint(0, 24, n_samples)
    source_credibility = np.random.uniform(0.5, 1.0, n_samples)

    X = pd.DataFrame(
        {
            "headline": headlines,
            "word_count": word_counts,
            "has_numbers": has_numbers,
            "hour_published": hour_published,
            "source_credibility": source_credibility,
        }
    )

    y = pd.Series(labels, name="category")

    return X, y, "text_classification", "News Headlines (text)"


def create_customer_support_dataset(n_samples=2000, random_state=42):
    """
    Customer support tickets - priority classification with text.
    Tests extraction of urgency and sentiment from support messages.
    """
    np.random.seed(random_state)

    priorities = ["Low", "Medium", "High", "Critical"]

    templates = {
        "Low": [
            "Question about {feature}",
            "How do I {action}?",
            "General inquiry about {topic}",
            "Feature request: {feature}",
        ],
        "Medium": [
            "Issue with {feature} not working correctly",
            "Need help with {action}",
            "Problem: {issue} happening sometimes",
            "Can't figure out how to {action}",
        ],
        "High": [
            "URGENT: {feature} is broken",
            "{feature} stopped working completely",
            "Critical issue: can't {action}",
            "Major problem with {issue}",
        ],
        "Critical": [
            "EMERGENCY: System down, can't access anything",
            "Data loss: {issue} caused problems",
            "Security breach detected in {feature}",
            "Production is DOWN - need immediate help",
        ],
    }

    fill_values = {
        "feature": ["login", "dashboard", "reports", "API", "billing", "notifications"],
        "action": ["export data", "reset password", "update settings", "integrate API"],
        "topic": ["pricing", "features", "billing", "account"],
        "issue": ["error messages", "slow loading", "crashes", "data sync"],
    }

    tickets = []
    priority_labels = []
    ticket_lengths = []
    contains_urgent = []
    customer_tiers = []

    for _ in range(n_samples):
        priority_idx = np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10])
        priority = priorities[priority_idx]
        template = np.random.choice(templates[priority])

        # Fill template
        ticket = template
        for key, values in fill_values.items():
            if "{" + key + "}" in ticket:
                ticket = ticket.replace("{" + key + "}", np.random.choice(values), 1)

        tickets.append(ticket)
        priority_labels.append(priority_idx)
        ticket_lengths.append(len(ticket.split()))
        contains_urgent.append(
            1 if any(w in ticket.upper() for w in ["URGENT", "EMERGENCY", "CRITICAL", "DOWN"]) else 0
        )
        customer_tiers.append(np.random.choice(["Free", "Basic", "Pro", "Enterprise"], p=[0.3, 0.3, 0.25, 0.15]))

    # Additional features
    response_time_hours = np.random.exponential(4, n_samples)
    previous_tickets = np.random.poisson(3, n_samples)

    X = pd.DataFrame(
        {
            "ticket_text": tickets,
            "ticket_length": ticket_lengths,
            "contains_urgent_words": contains_urgent,
            "customer_tier": customer_tiers,
            "response_time_hours": response_time_hours,
            "previous_tickets": previous_tickets,
        }
    )

    y = pd.Series(priority_labels, name="priority")

    return X, y, "text_classification", "Customer Support Tickets (text)"


def create_medical_notes_dataset(n_samples=1500, random_state=42):
    """
    Medical notes dataset - diagnosis prediction with clinical text.
    Tests domain-specific semantic understanding.
    """
    np.random.seed(random_state)

    conditions = ["Healthy", "Diabetes", "Hypertension", "Heart Disease", "Respiratory"]

    symptoms_map = {
        "Healthy": ["routine checkup", "no complaints", "feeling well", "annual physical"],
        "Diabetes": ["increased thirst", "frequent urination", "fatigue", "blurred vision", "high blood sugar"],
        "Hypertension": ["headaches", "high blood pressure", "dizziness", "chest discomfort", "shortness of breath"],
        "Heart Disease": ["chest pain", "irregular heartbeat", "fatigue", "swelling", "shortness of breath"],
        "Respiratory": ["coughing", "wheezing", "shortness of breath", "chest tightness", "mucus production"],
    }

    notes = []
    labels = []
    ages = []
    bmis = []
    systolic_bps = []
    glucose_levels = []

    for _ in range(n_samples):
        condition = np.random.choice(conditions, p=[0.30, 0.20, 0.20, 0.15, 0.15])
        symptoms = np.random.choice(symptoms_map[condition], np.random.randint(2, 4), replace=False)

        # Generate clinical note
        age = np.random.randint(25, 80)
        note = f"Patient presents with {', '.join(symptoms)}. "

        if condition == "Diabetes":
            glucose = np.random.randint(140, 250)
            note += f"Fasting glucose: {glucose} mg/dL. "
        elif condition == "Hypertension":
            bp = np.random.randint(150, 190)
            note += f"Blood pressure elevated: {bp}/{np.random.randint(90, 110)}. "
        elif condition == "Heart Disease":
            note += "ECG shows abnormalities. "
        elif condition == "Respiratory":
            note += "Lung sounds diminished. "
        else:
            note += "Vitals within normal limits. "

        notes.append(note)
        labels.append(conditions.index(condition))
        ages.append(age)

        # Generate correlated numeric features
        if condition == "Diabetes":
            bmis.append(np.random.normal(30, 5))
            glucose_levels.append(np.random.randint(140, 250))
            systolic_bps.append(np.random.randint(120, 150))
        elif condition == "Hypertension":
            bmis.append(np.random.normal(28, 4))
            glucose_levels.append(np.random.randint(90, 130))
            systolic_bps.append(np.random.randint(150, 190))
        elif condition == "Heart Disease":
            bmis.append(np.random.normal(29, 5))
            glucose_levels.append(np.random.randint(100, 150))
            systolic_bps.append(np.random.randint(130, 170))
        else:
            bmis.append(np.random.normal(25, 4))
            glucose_levels.append(np.random.randint(80, 110))
            systolic_bps.append(np.random.randint(110, 130))

    X = pd.DataFrame(
        {
            "clinical_notes": notes,
            "age": ages,
            "bmi": bmis,
            "systolic_bp": systolic_bps,
            "glucose_level": glucose_levels,
        }
    )

    y = pd.Series(labels, name="condition")

    return X, y, "text_classification", "Medical Notes (text)"


def create_ecommerce_product_dataset(n_samples=2000, random_state=42):
    """
    E-commerce product dataset - sales prediction with descriptions.
    Tests extraction of product attributes from text.
    """
    np.random.seed(random_state)

    categories = ["Electronics", "Clothing", "Home", "Sports", "Beauty"]

    product_templates = {
        "Electronics": [
            "{brand} {adjective} {product} with {feature}",
            "Premium {product} - {feature} technology",
            "{adjective} {product} for {use_case}",
        ],
        "Clothing": [
            "{brand} {adjective} {product} - {material}",
            "{adjective} {product} for {season}",
            "Designer {product} with {feature}",
        ],
        "Home": [
            "{brand} {adjective} {product} - {feature}",
            "{adjective} {product} for {room}",
            "Modern {product} with {material} finish",
        ],
        "Sports": [
            "{brand} {adjective} {product} for {activity}",
            "Professional {product} - {feature}",
            "{adjective} {product} with {technology}",
        ],
        "Beauty": [
            "{brand} {adjective} {product} - {benefit}",
            "Natural {product} with {ingredient}",
            "{adjective} {product} for {skin_type} skin",
        ],
    }

    fill_values = {
        "brand": ["Premium", "Elite", "Pro", "Ultra", "Essential"],
        "adjective": ["innovative", "sleek", "powerful", "compact", "advanced"],
        "product": ["device", "item", "accessory", "gear", "solution"],
        "feature": ["smart connectivity", "long battery", "HD display", "fast charging"],
        "use_case": ["everyday use", "professionals", "beginners", "experts"],
        "material": ["cotton", "leather", "synthetic", "organic", "recycled"],
        "season": ["summer", "winter", "all seasons", "spring"],
        "room": ["living room", "bedroom", "kitchen", "bathroom"],
        "activity": ["running", "training", "yoga", "outdoor sports"],
        "technology": ["moisture-wicking", "compression", "breathable"],
        "benefit": ["anti-aging", "moisturizing", "brightening", "hydrating"],
        "ingredient": ["vitamin C", "retinol", "hyaluronic acid", "niacinamide"],
        "skin_type": ["dry", "oily", "sensitive", "normal"],
    }

    descriptions = []
    cats = []
    prices = []
    ratings = []
    reviews_count = []

    base_prices = {"Electronics": 200, "Clothing": 50, "Home": 80, "Sports": 60, "Beauty": 40}

    for _ in range(n_samples):
        cat = np.random.choice(categories)
        template = np.random.choice(product_templates[cat])

        # Fill template
        desc = template
        for key, values in fill_values.items():
            if "{" + key + "}" in desc:
                desc = desc.replace("{" + key + "}", np.random.choice(values), 1)

        descriptions.append(desc)
        cats.append(cat)

        # Generate correlated features
        price = base_prices[cat] * np.random.lognormal(0, 0.5)
        prices.append(price)
        ratings.append(np.clip(np.random.normal(4.0, 0.7), 1, 5))
        reviews_count.append(np.random.poisson(50))

    # Calculate sales based on multiple factors
    sales = []
    for i in range(n_samples):
        base_sales = 100
        price_effect = -0.1 * (prices[i] / base_prices[cats[i]] - 1)
        rating_effect = 0.3 * (ratings[i] - 3)
        review_effect = 0.1 * np.log1p(reviews_count[i])

        sale = base_sales * (1 + price_effect + rating_effect + review_effect)
        sale = max(sale + np.random.normal(0, 20), 1)
        sales.append(int(sale))

    X = pd.DataFrame(
        {
            "description": descriptions,
            "category": cats,
            "price": prices,
            "rating": ratings,
            "reviews_count": reviews_count,
        }
    )

    y = pd.Series(sales, name="monthly_sales")

    return X, y, "text_regression", "E-commerce Products (text)"


def get_text_datasets():
    """Return text/semantic benchmark datasets."""
    return [
        create_product_reviews_dataset,
        create_job_postings_dataset,
        create_news_classification_dataset,
        create_customer_support_dataset,
        create_medical_notes_dataset,
        create_ecommerce_product_dataset,
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

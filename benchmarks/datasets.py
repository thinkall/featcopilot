"""
Benchmark datasets for FeatCopilot evaluation.

Includes real-world datasets (Kaggle-style), synthetic datasets,
time series datasets, and text/semantic datasets for comprehensive benchmarking.
"""

import numpy as np
import pandas as pd

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

    # Survival probability based on historical patterns with strong interaction effects
    family_size = sibsp + parch
    fare_per_class = fare / (pclass + 1)
    survival_prob = (
        0.35
        - 0.25 * sex * (pclass - 1) / 2  # Strong sex-class interaction
        - 0.005 * np.clip(age - 10, 0, 60)
        + 0.05 * (embarked == 1)
        + 0.03 * np.sqrt(fare_per_class / 20)  # Fare/class ratio + sqrt
        - 0.004 * age * sex  # Age-sex interaction
        + 0.08 * (family_size == 1)  # Family size effect (non-linear)
        - 0.06 * (family_size > 3)  # Large family penalty
        + 0.003 * fare * (1 - sex) / 50  # Fare helps female survival
        - 0.15 * sex * (age > 50) / 1  # Older males have low survival
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

    # Price dominated by interaction and polynomial effects
    price = (
        30000
        + X["OverallQual"] * 5000  # Modest linear
        + X["GrLivArea"] * 15  # Modest linear
        + X["GarageCars"] * 3000
        + X["OverallQual"] * X["GrLivArea"] * 25  # Strong quality-area interaction
        + X["OverallQual"] ** 2 * 2500  # Strong quadratic quality
        + np.log1p(X["LotArea"]) * 15000  # Strong log transform
        + X["GarageCars"] * X["OverallQual"] * 8000  # Strong garage-quality interaction
        + X["FullBath"] * X["TotRmsAbvGrd"] * 3000  # Strong bath-rooms interaction
        + X["OverallQual"] * X["FullBath"] * 5000  # Quality-bath interaction
        + np.sqrt(X["GrLivArea"]) * X["TotalBsmtSF"] * 0.5  # sqrt-area * basement
        + (X["GrLivArea"] / (X["TotRmsAbvGrd"] + 1)) * 30  # Area per room ratio
        + X["Fireplaces"] * X["OverallQual"] * 3000  # Fireplace-quality interaction
    )
    noise = np.random.normal(0, 12000, n)
    y = pd.Series(np.maximum(price + noise, 30000), name="SalePrice")

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

    # Demand dominated by interaction and polynomial effects
    base_demand = 80
    demand = (
        base_demand
        + 20 * np.sin(np.pi * hour / 12)  # Peak at noon (modest)
        + 15 * (hour >= 7) * (hour <= 9) * workingday  # Morning commute (modest)
        + 20 * (hour >= 17) * (hour <= 19) * workingday  # Evening commute (modest)
        # Dominant interaction/polynomial terms
        + 0.25 * temp**2  # Strong quadratic temperature
        - 0.15 * temp * humidity  # Strong temp-humidity interaction
        + 0.8 * temp * windspeed  # Strong temp-wind interaction
        - 0.08 * humidity * windspeed  # Humidity-wind interaction
        + 2.0 * temp * (1 - weather / 4)  # Temp-weather interaction
        - 0.003 * humidity**2  # Quadratic humidity
        + 0.5 * np.sqrt(np.maximum(temp, 0)) * (24 - np.abs(hour - 12))  # sqrt-temp * time
        + 0.02 * temp * hour * workingday  # Three-way interaction
        - 40 * (weather >= 3)  # Bad weather
    )
    demand = np.maximum(demand + np.random.normal(0, 20, n), 0)
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

    # Attrition probability dominated by interaction effects
    satisfaction_balance = job_satisfaction * work_life_balance
    attrition_prob = (
        0.15  # Base rate
        # Modest linear terms (25%)
        - 0.002 * age
        - 0.005 * years_at_company
        - 0.01 * job_satisfaction
        + 0.05 * overtime
        # Dominant interaction terms (65%)
        + 0.20 * overtime * (years_since_last_promotion / 8)  # Overtime-promotion interaction (strong)
        + 0.008 * distance_from_home * overtime  # Distance-overtime interaction
        - 0.003 * satisfaction_balance  # Satisfaction-balance synergy
        + 0.015 * np.sqrt(num_companies_worked * years_since_last_promotion)  # Job hopping risk
        - 0.00001 * monthly_income / (distance_from_home + 1)  # Income/distance ratio
        + 0.004 * num_companies_worked * (30 - years_at_company) / 30  # Job hopper + low tenure
        - 0.003 * percent_salary_hike * job_satisfaction / 4  # Hike-satisfaction interaction
        + 0.05 * overtime * (1 - work_life_balance / 4)  # Overtime when poor balance
        + 0.002 * distance_from_home * (5 - work_life_balance) / 5  # Distance + poor balance
        - 0.003 * training_times_last_year * job_satisfaction  # Training-satisfaction interaction
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
    savings_to_debt = X["savings"] / (X["debt"] + 1)
    expense_ratio = X["monthly_expenses"] / (X["income"] + 1)
    credit_score = (
        # Modest linear terms (30%)
        0.10 * (X["credit_history_months"] / 360)
        + 0.08 * (X["employment_years"] / 20)
        + 0.05 * (1 - X["num_loans"] / 10)
        # Dominant interaction/ratio terms (60%)
        + 0.20 * (1 - np.clip(debt_to_income, 0, 2) / 2)  # debt/income ratio
        + 0.15 * np.log1p(savings_to_debt)  # Log savings/debt ratio
        - 0.12 * np.sqrt(expense_ratio)  # sqrt expense ratio
        - 0.08 * (X["num_credit_cards"] * X["num_loans"] / 20)  # Card-loan interaction
        + 0.10 * (X["employment_years"] * X["credit_history_months"] / 3600)  # Stability interaction
        + 0.06 * np.log1p(X["savings"]) / 15  # Log savings transform
        - 0.04 * (X["num_dependents"] * X["monthly_expenses"] / 50000)  # Dependent-expense interaction
        + 0.05 * (X["income"] * X["employment_years"] / 500000)  # Income-employment interaction
    )
    credit_score = np.clip(credit_score, 0, 1)
    noise = np.random.normal(0, 0.08, n_samples)
    y = pd.Series((credit_score + noise > 0.45).astype(int), name="target")

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

    cholesterol_ratio = X["cholesterol_ldl"] / (X["cholesterol_hdl"] + 1)
    bp_ratio = X["blood_pressure_systolic"] / (X["blood_pressure_diastolic"] + 1)
    risk_score = (
        # Modest linear terms (25%)
        0.08 * (X["age"] / 85)
        + 0.05 * np.clip((X["bmi"] - 18.5) / 20, 0, 1)
        + 0.04 * np.clip((X["glucose_fasting"] - 70) / 100, 0, 1)
        + 0.04 * np.clip((X["hba1c"] - 4) / 4, 0, 1)
        + 0.04 * np.clip(X["smoking_years"] / 30, 0, 1)
        # Dominant interaction/ratio terms (65%)
        + 0.12 * np.clip(cholesterol_ratio / 5, 0, 1)  # LDL/HDL ratio
        + 0.10 * np.clip(bp_ratio / 2, 0, 1)  # Systolic/diastolic ratio
        + 0.10 * np.clip(X["bmi"] * X["blood_pressure_systolic"] / 4000, 0, 1)  # BMI-BP interaction
        + 0.08 * np.clip(X["glucose_fasting"] * X["hba1c"] / 1000, 0, 1)  # Glucose-HbA1c interaction
        + 0.06 * np.clip(X["age"] * X["smoking_years"] / 2500, 0, 1)  # Age-smoking interaction
        + 0.05 * np.clip(np.sqrt(X["bmi"] * X["glucose_fasting"]) / 60, 0, 1)  # sqrt BMI-glucose
        + 0.04 * np.clip(X["age"] * X["bmi"] / 4000, 0, 1)  # Age-BMI interaction
        + 0.04 * np.clip(X["heart_rate"] * X["blood_pressure_systolic"] / 12000, 0, 1)  # HR-BP interaction
        - 0.06 * np.clip(X["exercise_hours_weekly"] * X["cholesterol_hdl"] / 300, 0, 1)  # Exercise-HDL synergy
    )
    noise = np.random.normal(0, 0.06, n_samples)
    y = pd.Series((risk_score + noise > 0.35).astype(int), name="target")

    return X, y, "classification", "Medical Diagnosis (synthetic)"


def create_complex_regression_dataset(n_samples=2000, n_features=15, random_state=42):
    """Complex regression with non-linear relationships dominated by interactions."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)})

    # Target dominated by interactions, ratios, and polynomials
    y_val = (
        # Small linear terms (20%)
        2.0 * X["feature_0"]
        + 1.5 * X["feature_1"]
        + 1.0 * X["feature_2"]
        # Dominant interaction terms (70%)
        + 5.0 * X["feature_0"] * X["feature_1"]  # Product interaction
        + 4.0 * X["feature_2"] * X["feature_3"]  # Product interaction
        + 3.5 * X["feature_0"] ** 2  # Quadratic
        + 3.0 * X["feature_4"] ** 2  # Quadratic
        + 2.5 * X["feature_1"] * X["feature_5"]  # Product interaction
        + 2.0 * X["feature_3"] * X["feature_6"]  # Product interaction
        + 1.5 * X["feature_7"] * X["feature_8"]  # Product interaction
        + 1.0 * X["feature_2"] ** 2  # Quadratic
        + 0.8 * X["feature_0"] * X["feature_9"]  # Product interaction
    )
    noise = np.random.normal(0, 2.0, n_samples)
    y = pd.Series(y_val + noise, name="target")
    return X, y, "regression", "Complex Regression (synthetic)"


def create_complex_classification_dataset(n_samples=2000, n_features=15, random_state=42):
    """Complex classification with interaction-heavy decision boundary."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)})

    # Classification boundary dominated by interactions
    score = (
        # Small linear terms (20%)
        0.5 * X["feature_0"]
        + 0.3 * X["feature_1"]
        # Dominant interaction terms (70%)
        + 2.0 * X["feature_0"] * X["feature_1"]  # Product interaction
        + 1.5 * X["feature_2"] * X["feature_3"]  # Product interaction
        + 1.0 * X["feature_0"] ** 2  # Quadratic
        + 0.8 * X["feature_4"] * X["feature_5"]  # Product interaction
        + 0.6 * X["feature_6"] * X["feature_7"]  # Product interaction
        + 0.5 * X["feature_2"] ** 2  # Quadratic
        + 0.4 * X["feature_8"] * X["feature_9"]  # Product interaction
    )
    prob = 1 / (1 + np.exp(-score))
    y_val = (np.random.random(n_samples) < prob).astype(int)
    y = pd.Series(y_val, name="target")
    return X, y, "classification", "Complex Classification (synthetic)"


def create_polynomial_regression_dataset(n_samples=2000, n_features=12, random_state=43):
    """Regression where target depends on polynomial and sqrt transforms."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"x_{i}": np.random.randn(n_samples) for i in range(n_features)})

    y_val = (
        1.0 * X["x_0"]
        + 4.0 * X["x_0"] ** 2
        + 3.5 * X["x_1"] ** 2
        + 3.0 * X["x_2"] ** 2
        + 2.5 * X["x_3"] * X["x_4"]
        + 2.0 * X["x_5"] * X["x_6"]
        + 1.5 * X["x_0"] * X["x_7"]
        + 1.0 * X["x_1"] * X["x_8"]
    )
    y = pd.Series(y_val + np.random.normal(0, 1.5, n_samples), name="target")
    return X, y, "regression", "Polynomial Regression (synthetic)"


def create_ratio_regression_dataset(n_samples=2000, random_state=44):
    """Regression where target depends on feature ratios and products."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"r_{i}": np.random.randn(n_samples) + 3 for i in range(12)})

    y_val = (
        5.0 * X["r_0"] * X["r_1"]
        + 4.0 * X["r_2"] / (np.abs(X["r_3"]) + 1)
        + 3.5 * X["r_0"] ** 2
        + 3.0 * np.log(np.abs(X["r_4"]) + 1) * X["r_5"]
        + 2.5 * np.sqrt(np.abs(X["r_6"])) * X["r_7"]
        + 2.0 * X["r_8"] * X["r_9"]
        + 1.5 * X["r_10"] * X["r_11"]
        + 0.5 * X["r_0"]
    )
    y = pd.Series(y_val + np.random.normal(0, 2.0, n_samples), name="target")
    return X, y, "regression", "Ratio Regression (synthetic)"


def create_interaction_classification_dataset(n_samples=2000, n_features=12, random_state=45):
    """Classification where boundary depends on feature products."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"v_{i}": np.random.randn(n_samples) for i in range(n_features)})

    score = (
        0.3 * X["v_0"]
        + 2.5 * X["v_0"] * X["v_1"]
        + 2.0 * X["v_2"] * X["v_3"]
        + 1.5 * X["v_4"] ** 2
        + 1.0 * X["v_5"] * X["v_6"]
        + 0.8 * X["v_7"] * X["v_8"]
        + 0.5 * X["v_3"] ** 2
    )
    prob = 1 / (1 + np.exp(-score))
    y = pd.Series((np.random.random(n_samples) < prob).astype(int), name="target")
    return X, y, "classification", "Interaction Classification (synthetic)"


def create_nonlinear_regression_dataset(n_samples=2500, random_state=46):
    """Regression with non-linear transforms (log, sqrt, quadratic) dominating."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"n_{i}": np.random.randn(n_samples) for i in range(12)})

    y_val = (
        0.5 * X["n_0"]
        + 5.0 * X["n_0"] * X["n_1"]
        + 4.0 * X["n_2"] ** 2
        + 3.5 * X["n_3"] * X["n_4"]
        + 3.0 * X["n_5"] ** 2
        + 2.5 * X["n_6"] * X["n_7"]
        + 2.0 * X["n_8"] * X["n_9"]
        + 1.5 * X["n_10"] * X["n_11"]
        + 1.0 * X["n_1"] ** 2
    )
    y = pd.Series(y_val + np.random.normal(0, 1.5, n_samples), name="target")
    return X, y, "regression", "Nonlinear Regression (synthetic)"


def create_xor_regression_dataset(n_samples=2500, n_features=20, random_state=47):
    """Regression where target is purely products of feature pairs (XOR-like)."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"z_{i}": np.random.randn(n_samples) for i in range(n_features)})

    y_val = (
        6.0 * X["z_0"] * X["z_1"]
        + 5.0 * X["z_2"] * X["z_3"]
        + 4.0 * X["z_4"] * X["z_5"]
        + 3.5 * X["z_6"] * X["z_7"]
        + 3.0 * X["z_8"] * X["z_9"]
        + 2.5 * X["z_10"] * X["z_11"]
        + 2.0 * X["z_0"] * X["z_12"]
        + 1.5 * X["z_1"] * X["z_13"]
    )
    y = pd.Series(y_val + np.random.normal(0, 1.5, n_samples), name="target")
    return X, y, "regression", "XOR Regression (synthetic)"


def create_quadratic_heavy_regression_dataset(n_samples=2500, n_features=18, random_state=48):
    """Regression dominated by squared and cubic terms."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"q_{i}": np.random.randn(n_samples) for i in range(n_features)})

    y_val = (
        5.0 * X["q_0"] ** 2
        + 4.5 * X["q_1"] ** 2
        + 4.0 * X["q_2"] ** 2
        + 3.5 * X["q_3"] ** 2
        + 3.0 * X["q_4"] * X["q_5"]
        + 2.5 * X["q_6"] * X["q_7"]
        + 2.0 * X["q_8"] ** 2
        + 1.5 * X["q_0"] * X["q_9"]
        + 1.0 * X["q_1"] * X["q_10"]
    )
    y = pd.Series(y_val + np.random.normal(0, 1.8, n_samples), name="target")
    return X, y, "regression", "Quadratic Heavy Regression (synthetic)"


def create_pairwise_product_regression_dataset(n_samples=2000, n_features=16, random_state=49):
    """Regression where target is sum of many pairwise products."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"p_{i}": np.random.randn(n_samples) for i in range(n_features)})

    y_val = (
        4.0 * X["p_0"] * X["p_1"]
        + 3.5 * X["p_2"] * X["p_3"]
        + 3.0 * X["p_4"] * X["p_5"]
        + 2.5 * X["p_6"] * X["p_7"]
        + 2.0 * X["p_8"] * X["p_9"]
        + 1.5 * X["p_10"] * X["p_11"]
        + 1.0 * X["p_12"] * X["p_13"]
        + 0.8 * X["p_14"] * X["p_15"]
        + 3.0 * X["p_0"] ** 2
        + 2.5 * X["p_2"] ** 2
    )
    y = pd.Series(y_val + np.random.normal(0, 1.5, n_samples), name="target")
    return X, y, "regression", "Pairwise Product Regression (synthetic)"


def create_xor_classification_dataset(n_samples=2500, n_features=20, random_state=50):
    """Classification with pure interaction decision boundary."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"c_{i}": np.random.randn(n_samples) for i in range(n_features)})

    score = (
        3.0 * X["c_0"] * X["c_1"]
        + 2.5 * X["c_2"] * X["c_3"]
        + 2.0 * X["c_4"] * X["c_5"]
        + 1.5 * X["c_6"] * X["c_7"]
        + 1.0 * X["c_8"] * X["c_9"]
        + 0.8 * X["c_10"] ** 2
        + 0.6 * X["c_11"] * X["c_12"]
    )
    prob = 1 / (1 + np.exp(-score))
    y = pd.Series((np.random.random(n_samples) < prob).astype(int), name="target")
    return X, y, "classification", "XOR Classification (synthetic)"


def create_polynomial_classification_dataset(n_samples=2000, n_features=15, random_state=51):
    """Classification where boundary depends on polynomials and interactions."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"pc_{i}": np.random.randn(n_samples) for i in range(n_features)})

    score = (
        2.0 * X["pc_0"] ** 2
        + 1.8 * X["pc_1"] ** 2
        + 2.5 * X["pc_2"] * X["pc_3"]
        + 2.0 * X["pc_4"] * X["pc_5"]
        + 1.5 * X["pc_6"] * X["pc_7"]
        + 1.0 * X["pc_8"] ** 2
        + 0.8 * X["pc_9"] * X["pc_10"]
        - 3.0  # shift to center probabilities
    )
    prob = 1 / (1 + np.exp(-score))
    y = pd.Series((np.random.random(n_samples) < prob).astype(int), name="target")
    return X, y, "classification", "Polynomial Classification (synthetic)"


def create_sqrt_log_regression_dataset(n_samples=2500, n_features=15, random_state=52):
    """Regression with sqrt, log, and product transforms."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"s_{i}": np.abs(np.random.randn(n_samples)) + 0.5 for i in range(n_features)})

    y_val = (
        5.0 * np.sqrt(X["s_0"]) * np.sqrt(X["s_1"])
        + 4.0 * np.log(X["s_2"]) * X["s_3"]
        + 3.5 * X["s_4"] * X["s_5"]
        + 3.0 * np.sqrt(X["s_6"] * X["s_7"])
        + 2.5 * np.log(X["s_8"]) * np.log(X["s_9"])
        + 2.0 * X["s_10"] ** 2
        + 1.5 * X["s_11"] * X["s_12"]
        + 1.0 * np.sqrt(X["s_13"]) * X["s_14"]
    )
    y = pd.Series(y_val + np.random.normal(0, 1.5, n_samples), name="target")
    return X, y, "regression", "Sqrt Log Regression (synthetic)"


def create_triple_interaction_regression_dataset(n_samples=2000, n_features=18, random_state=53):
    """Regression with two-way and three-way interaction terms."""
    np.random.seed(random_state)
    X = pd.DataFrame({f"t_{i}": np.random.randn(n_samples) for i in range(n_features)})

    y_val = (
        5.0 * X["t_0"] * X["t_1"]
        + 4.0 * X["t_2"] * X["t_3"]
        + 3.5 * X["t_4"] ** 2
        + 3.0 * X["t_5"] * X["t_6"]
        + 2.5 * X["t_7"] * X["t_8"]
        + 2.0 * X["t_9"] ** 2
        + 1.5 * X["t_10"] * X["t_11"]
        + 1.0 * X["t_12"] * X["t_13"]
        + 0.8 * X["t_14"] * X["t_15"]
        + 0.5 * X["t_16"] * X["t_17"]
    )
    y = pd.Series(y_val + np.random.normal(0, 2.0, n_samples), name="target")
    return X, y, "regression", "Triple Interaction Regression (synthetic)"


# =============================================================================
# Real-World Datasets (from Kaggle, OpenML, HuggingFace)
# =============================================================================


def load_kaggle_house_prices():
    """
    Load House Prices dataset from Kaggle/HuggingFace.

    Predict sale prices of homes in Ames, Iowa based on 79 features.
    This is the gold standard for tabular feature engineering benchmarks.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (79 features).
    y : pd.Series
        Sale prices.
    task : str
        Task type ("regression").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("leomaurodesenv/house-prices-advanced-regression-techniques", split="train")
        df = ds.to_pandas()

        # Target column
        target = "SalePrice"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        # Drop ID column if present
        drop_cols = ["Id", "id"]
        X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns])
        y = df[target]

        return X, y, "regression", "House Prices (Kaggle)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Falling back to synthetic version...")
        return load_house_prices_dataset()


def load_kaggle_employee_attrition():
    """
    Load IBM HR Employee Attrition dataset.

    Predict employee attrition based on HR metrics.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (34 features).
    y : pd.Series
        Attrition labels (0/1).
    task : str
        Task type ("classification").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("jpmiller/hr-employee-attrition", split="train")
        df = ds.to_pandas()

        target = "Attrition"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        X = df.drop(columns=[target])
        y = df[target].map({"Yes": 1, "No": 0}) if df[target].dtype == object else df[target]

        return X, y, "classification", "Employee Attrition (Kaggle)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Falling back to synthetic version...")
        return load_employee_attrition_dataset()


def load_kaggle_telco_churn():
    """
    Load Telco Customer Churn dataset.

    Predict customer churn based on service usage and billing.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (20 features).
    y : pd.Series
        Churn labels (0/1).
    task : str
        Task type ("classification").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("aai510-group1/telco-customer-churn", split="train")
        df = ds.to_pandas()

        target = "Churn"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        # Drop customer ID
        drop_cols = ["customerID", "CustomerID"]
        X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns])
        y = df[target].map({"Yes": 1, "No": 0}) if df[target].dtype == object else df[target]

        return X, y, "classification", "Telco Customer Churn (Kaggle)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Returning synthetic churn dataset...")
        # Create synthetic churn dataset
        np.random.seed(42)
        n = 7000
        X = pd.DataFrame(
            {
                "tenure": np.random.randint(1, 72, n),
                "MonthlyCharges": np.random.uniform(20, 100, n),
                "TotalCharges": np.random.uniform(100, 8000, n),
                "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
                "PaymentMethod": np.random.choice(
                    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
                ),
                "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
                "OnlineSecurity": np.random.choice(["Yes", "No", "No internet"], n),
                "TechSupport": np.random.choice(["Yes", "No", "No internet"], n),
            }
        )
        churn_prob = 0.2 + 0.3 * (X["Contract"] == "Month-to-month") - 0.1 * (X["tenure"] / 72)
        y = pd.Series((np.random.random(n) < churn_prob).astype(int), name="Churn")
        return X, y, "classification", "Telco Churn (synthetic)"


def load_openml_adult_census():
    """
    Load Adult Census Income dataset from OpenML.

    Predict whether income exceeds $50K/year.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (14 features).
    y : pd.Series
        Income labels (0/1).
    task : str
        Task type ("classification").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("scikit-learn/adult-census-income", split="train")
        df = ds.to_pandas()

        target = "income"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        X = df.drop(columns=[target])
        y = df[target].map({">50K": 1, "<=50K": 0, ">50K.": 1, "<=50K.": 0})
        if y.isna().any():
            y = (df[target].str.contains(">50K")).astype(int)

        return X, y, "classification", "Adult Census Income (OpenML)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Returning synthetic census dataset...")
        np.random.seed(42)
        n = 10000
        X = pd.DataFrame(
            {
                "age": np.random.randint(18, 70, n),
                "workclass": np.random.choice(["Private", "Self-emp", "Gov", "Other"], n),
                "education": np.random.choice(["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"], n),
                "education-num": np.random.randint(8, 16, n),
                "marital-status": np.random.choice(["Married", "Never-married", "Divorced"], n),
                "occupation": np.random.choice(["Tech", "Sales", "Admin", "Service", "Exec"], n),
                "hours-per-week": np.random.randint(20, 60, n),
                "capital-gain": np.random.exponential(1000, n),
            }
        )
        income_prob = 0.2 + 0.02 * (X["education-num"] - 10) + 0.005 * (X["age"] - 30)
        y = pd.Series((np.random.random(n) < income_prob).astype(int), name="income")
        return X, y, "classification", "Adult Census (synthetic)"


def load_kaggle_credit_card_fraud():
    """
    Load Credit Card Fraud Detection dataset.

    Detect fraudulent transactions from anonymized features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (30 features: V1-V28, Amount, Time).
    y : pd.Series
        Fraud labels (0/1).
    task : str
        Task type ("classification").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("nelgiriyewithana/credit-card-fraud-detection-dataset-2023", split="train")
        df = ds.to_pandas()

        target = "Class"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        drop_cols = ["id", "Id"]
        X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns])
        y = df[target]

        return X, y, "classification", "Credit Card Fraud (Kaggle)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Falling back to synthetic version...")
        return load_credit_card_fraud_dataset()


def load_kaggle_spaceship_titanic():
    """
    Load Spaceship Titanic dataset from Kaggle.

    Predict which passengers were transported to an alternate dimension.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (~13 features).
    y : pd.Series
        Transported labels (0/1).
    task : str
        Task type ("classification").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("spaceship-titanic/spaceship-titanic", split="train")
        df = ds.to_pandas()

        target = "Transported"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        drop_cols = ["PassengerId", "Name"]
        X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns])
        y = df[target].astype(int) if df[target].dtype == bool else df[target]

        return X, y, "classification", "Spaceship Titanic (Kaggle)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Returning synthetic spaceship dataset...")
        np.random.seed(42)
        n = 8000
        X = pd.DataFrame(
            {
                "HomePlanet": np.random.choice(["Earth", "Europa", "Mars"], n),
                "CryoSleep": np.random.choice([True, False], n),
                "Cabin": [
                    f"{np.random.choice(['A','B','C','D','E','F'])}/{np.random.randint(1,1000)}/{np.random.choice(['P','S'])}"
                    for _ in range(n)
                ],
                "Destination": np.random.choice(["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"], n),
                "Age": np.random.uniform(0, 80, n),
                "VIP": np.random.choice([True, False], n, p=[0.05, 0.95]),
                "RoomService": np.random.exponential(200, n),
                "FoodCourt": np.random.exponential(400, n),
                "ShoppingMall": np.random.exponential(300, n),
                "Spa": np.random.exponential(300, n),
                "VRDeck": np.random.exponential(300, n),
            }
        )
        transported_prob = 0.5 + 0.2 * X["CryoSleep"] - 0.001 * X["Age"]
        y = pd.Series((np.random.random(n) < transported_prob).astype(int), name="Transported")
        return X, y, "classification", "Spaceship Titanic (synthetic)"


def load_kaggle_bike_sharing():
    """
    Load Bike Sharing Demand dataset.

    Predict hourly bike rental demand.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (12 features).
    y : pd.Series
        Rental count.
    task : str
        Task type ("regression").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("BrejBala/Bike_Sharing_Demand", split="train")
        df = ds.to_pandas()

        target = "count"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        drop_cols = ["datetime", "casual", "registered"]
        X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns])
        y = df[target]

        return X, y, "regression", "Bike Sharing Demand (Kaggle)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Falling back to synthetic version...")
        return load_bike_sharing_dataset()


def load_kaggle_medical_cost():
    """
    Load Medical Cost Personal dataset.

    Predict individual medical costs based on demographics and lifestyle.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (6 features).
    y : pd.Series
        Medical charges.
    task : str
        Task type ("regression").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("yashpandey02/medical-cost-insurance", split="train")
        df = ds.to_pandas()

        target = "charges"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        X = df.drop(columns=[target])
        y = df[target]

        return X, y, "regression", "Medical Cost (Kaggle)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Returning synthetic medical cost dataset...")
        np.random.seed(42)
        n = 1300
        X = pd.DataFrame(
            {
                "age": np.random.randint(18, 65, n),
                "sex": np.random.choice(["male", "female"], n),
                "bmi": np.random.normal(30, 6, n),
                "children": np.random.randint(0, 5, n),
                "smoker": np.random.choice(["yes", "no"], n, p=[0.2, 0.8]),
                "region": np.random.choice(["northeast", "northwest", "southeast", "southwest"], n),
            }
        )
        base_cost = 5000 + 250 * X["age"] + 300 * X["bmi"]
        smoker_effect = 20000 * (X["smoker"] == "yes")
        y = pd.Series(base_cost + smoker_effect + np.random.normal(0, 3000, n), name="charges")
        return X, y, "regression", "Medical Cost (synthetic)"


def load_openml_wine_quality():
    """
    Load Wine Quality dataset from OpenML.

    Predict wine quality score based on physicochemical properties.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (11 features).
    y : pd.Series
        Quality score (0-10).
    task : str
        Task type ("regression").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("codesignal/wine-quality", split="train")
        df = ds.to_pandas()

        target = "quality"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        drop_cols = ["type", "Id", "id"]
        X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns])
        y = df[target]

        return X, y, "regression", "Wine Quality (OpenML)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Returning synthetic wine dataset...")
        np.random.seed(42)
        n = 5000
        X = pd.DataFrame(
            {
                "fixed acidity": np.random.uniform(4, 16, n),
                "volatile acidity": np.random.uniform(0.1, 1.5, n),
                "citric acid": np.random.uniform(0, 1, n),
                "residual sugar": np.random.uniform(0.5, 20, n),
                "chlorides": np.random.uniform(0.01, 0.2, n),
                "free sulfur dioxide": np.random.uniform(1, 70, n),
                "total sulfur dioxide": np.random.uniform(5, 300, n),
                "density": np.random.uniform(0.99, 1.01, n),
                "pH": np.random.uniform(2.8, 4, n),
                "sulphates": np.random.uniform(0.3, 2, n),
                "alcohol": np.random.uniform(8, 15, n),
            }
        )
        quality = 5 + 0.3 * X["alcohol"] - 2 * X["volatile acidity"] + np.random.normal(0, 0.5, n)
        y = pd.Series(np.clip(quality, 3, 9).astype(int), name="quality")
        return X, y, "regression", "Wine Quality (synthetic)"


def load_kaggle_life_expectancy():
    """
    Load Life Expectancy dataset from WHO.

    Predict life expectancy based on health and economic factors.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (20+ features).
    y : pd.Series
        Life expectancy in years.
    task : str
        Task type ("regression").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("kumarajarshi/life-expectancy-who", split="train")
        df = ds.to_pandas()

        target = "Life expectancy "
        if target not in df.columns:
            target = "Life expectancy"
        if target not in df.columns:
            raise ValueError("Target column 'Life expectancy' not found")

        drop_cols = ["Country", "Year"]
        X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns])
        y = df[target]

        return X, y, "regression", "Life Expectancy (WHO)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Returning synthetic life expectancy dataset...")
        np.random.seed(42)
        n = 2500
        X = pd.DataFrame(
            {
                "Status": np.random.choice(["Developing", "Developed"], n, p=[0.8, 0.2]),
                "Adult Mortality": np.random.uniform(50, 400, n),
                "infant deaths": np.random.poisson(20, n),
                "Alcohol": np.random.uniform(0, 15, n),
                "percentage expenditure": np.random.exponential(500, n),
                "BMI": np.random.uniform(15, 50, n),
                "Polio": np.random.uniform(50, 99, n),
                "Diphtheria": np.random.uniform(50, 99, n),
                "HIV/AIDS": np.random.exponential(2, n),
                "GDP": np.random.exponential(10000, n),
                "Schooling": np.random.uniform(5, 20, n),
            }
        )
        life_exp = 65 + 5 * (X["Status"] == "Developed") - 0.02 * X["Adult Mortality"] + 0.5 * X["Schooling"]
        y = pd.Series(np.clip(life_exp + np.random.normal(0, 3, n), 40, 90), name="Life expectancy")
        return X, y, "regression", "Life Expectancy (synthetic)"


def load_kaggle_home_credit():
    """
    Load Home Credit Default Risk dataset (main application table).

    Predict loan default probability.

    Note: This loads only the main application table. The full competition
    includes multiple related tables for advanced feature engineering.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (100+ features).
    y : pd.Series
        Default labels (0/1).
    task : str
        Task type ("classification").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("inria-soda/tabular-benchmark", name="clf_cat_home-credit-default-risk-v2", split="train")
        df = ds.to_pandas()

        target = "TARGET"
        if target not in df.columns:
            target = "target"
        if target not in df.columns:
            raise ValueError("Target column not found")

        drop_cols = ["SK_ID_CURR", "index"]
        X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns])
        y = df[target]

        return X, y, "classification", "Home Credit Default (Kaggle)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Returning synthetic credit dataset...")
        return create_credit_risk_dataset(n_samples=5000)


def load_kaggle_store_sales():
    """
    Load Store Sales Time Series dataset.

    Predict daily sales for stores and product families.

    Note: This is a simplified version. The full competition includes
    multiple tables (stores, oil prices, holidays, transactions).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Sales values.
    task : str
        Task type ("regression").
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("t4tiana/store-sales-time-series-forecasting", split="train")
        df = ds.to_pandas()

        target = "sales"
        if target not in df.columns:
            raise ValueError("Target column 'sales' not found")

        drop_cols = ["id", "date"]
        X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns])
        y = df[target]

        # Limit size for benchmarking
        if len(X) > 100000:
            idx = np.random.RandomState(42).choice(len(X), 100000, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)

        return X, y, "regression", "Store Sales (Kaggle)"

    except Exception as e:
        print(f"Warning: Could not load real dataset: {e}")
        print("Falling back to synthetic retail demand...")
        return create_retail_demand_timeseries(n_samples=5000)


def get_real_world_datasets():
    """Return all real-world dataset loaders from Kaggle/OpenML/HuggingFace."""
    return [
        load_kaggle_house_prices,
        load_kaggle_employee_attrition,
        load_kaggle_telco_churn,
        load_openml_adult_census,
        load_kaggle_credit_card_fraud,
        load_kaggle_spaceship_titanic,
        load_kaggle_bike_sharing,
        load_kaggle_medical_cost,
        load_openml_wine_quality,
        load_kaggle_life_expectancy,
        load_kaggle_home_credit,
        load_kaggle_store_sales,
    ]


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

    # Efficiency dominated by interaction and polynomial terms
    efficiency = (
        80  # Base efficiency
        # Dominant interactions (80%)
        - 2.0 * (temperature - 60) ** 2 / 100  # Strong quadratic temp
        - 1.5 * (pressure - 125) ** 2 / 100  # Strong quadratic pressure
        - 30 * vibration**2  # Strong quadratic vibration
        + 0.05 * temperature * pressure / 100  # Temp-pressure interaction
        - 0.2 * vibration * rpm / 100  # Vibration-RPM interaction
        + 0.5 * np.sqrt(power_input)  # Sqrt transform
        - 0.05 * humidity * temperature / 100  # Humidity-temp interaction
        + 0.01 * rpm * power_input / 1000  # RPM-power interaction
        + 2 * (shift == 1)  # Day shift bonus
    )
    efficiency = np.clip(efficiency + np.random.normal(0, 1, n_samples), 50, 100)
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


def create_customer_churn_dataset(n_samples=2000, random_state=42):
    """
    Customer churn dataset - binary classification.

    Predicts whether customers will churn based on usage and engagement metrics.
    Designed with interaction effects that benefit from feature engineering.
    """
    np.random.seed(random_state)

    # Customer demographics
    tenure_months = np.random.exponential(24, n_samples).astype(int)
    age = np.random.randint(18, 70, n_samples)

    # Usage metrics
    monthly_charges = 20 + 80 * np.random.random(n_samples)
    total_charges = monthly_charges * tenure_months * (0.9 + 0.2 * np.random.random(n_samples))
    num_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05])
    support_tickets = np.random.poisson(2, n_samples)

    # Engagement
    login_frequency = np.random.exponential(10, n_samples)
    last_interaction_days = np.random.exponential(30, n_samples).astype(int)

    # Contract and payment
    contract_type = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])  # Month, Year, 2-Year
    payment_delay_count = np.random.poisson(1, n_samples)

    X = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "age": age,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_products": num_products,
            "support_tickets": support_tickets,
            "login_frequency": login_frequency,
            "last_interaction_days": last_interaction_days,
            "contract_type": contract_type,
            "payment_delay_count": payment_delay_count,
        }
    )

    # Churn probability purely from INTERACTION and RATIO effects
    charges_per_product = monthly_charges / (num_products + 1)
    engagement_score = login_frequency / (last_interaction_days + 1)
    cost_per_tenure = monthly_charges / (tenure_months + 1)

    churn_prob = (
        0.35  # Base churn rate
        # Pure interaction/ratio terms
        + 0.008 * charges_per_product  # Cost/product ratio
        - 0.25 * engagement_score  # Login/inactivity ratio
        + 0.005 * cost_per_tenure  # Cost/tenure ratio
        + 0.06 * payment_delay_count * (1 - contract_type / 2)  # Delay × short contract
        + 0.005 * last_interaction_days * (5 - num_products) / 5  # Inactive × few products
        - 0.0002 * tenure_months * num_products  # Long tenure × many products
        + 0.002 * support_tickets * monthly_charges / 10  # Tickets × high cost
        + 0.03 * np.sqrt(payment_delay_count * last_interaction_days + 1)  # sqrt delay-inactivity
        + 0.0006 * monthly_charges * (4 - contract_type)  # High cost × short contract
        - 0.001 * age * tenure_months / 50  # Age × tenure stability
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.8)
    y = pd.Series((np.random.random(n_samples) < churn_prob).astype(int), name="churn")

    return X, y, "classification", "Customer Churn (synthetic)"


def create_insurance_claims_dataset(n_samples=2000, random_state=42):
    """
    Insurance claims dataset - regression (claim amount prediction).

    Predicts insurance claim amounts based on policy and customer features.
    Has polynomial and interaction effects beneficial for feature engineering.
    """
    np.random.seed(random_state)

    # Policy features
    policy_age_years = np.random.exponential(5, n_samples)
    coverage_amount = np.random.choice([50000, 100000, 200000, 500000], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    deductible = np.random.choice([500, 1000, 2000, 5000], n_samples, p=[0.2, 0.4, 0.3, 0.1])
    premium_monthly = 50 + 0.001 * coverage_amount + np.random.normal(0, 20, n_samples)

    # Customer features
    age = np.random.randint(18, 75, n_samples)
    credit_score = np.random.normal(700, 80, n_samples).clip(300, 850)
    num_claims_history = np.random.poisson(1, n_samples)
    years_as_customer = np.minimum(np.random.exponential(7, n_samples), age - 18)

    # Risk factors
    risk_score = np.random.uniform(1, 10, n_samples)
    incident_severity = np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05])

    X = pd.DataFrame(
        {
            "policy_age_years": policy_age_years,
            "coverage_amount": coverage_amount,
            "deductible": deductible,
            "premium_monthly": premium_monthly,
            "age": age,
            "credit_score": credit_score,
            "num_claims_history": num_claims_history,
            "years_as_customer": years_as_customer,
            "risk_score": risk_score,
            "incident_severity": incident_severity,
        }
    )

    # Claim amount purely from interaction and polynomial effects
    claim_amount = (
        0.12 * coverage_amount * (incident_severity / 4)  # Coverage × severity (dominant)
        + 1000 * risk_score**1.5  # Polynomial risk (dominant)
        + 600 * num_claims_history * incident_severity  # History × severity
        + 100 * (age / 40) ** 2  # Quadratic age
        - 1.0 * deductible * incident_severity  # Deductible × severity
        + 0.003 * coverage_amount * risk_score / 10  # Coverage × risk
        + 150 * np.log1p(years_as_customer) * incident_severity  # Log-tenure × severity
        - 0.2 * credit_score * (1 - risk_score / 10)  # Credit-risk interaction
        + 80 * np.sqrt(policy_age_years) * incident_severity  # sqrt-policy × severity
    )
    claim_amount = np.maximum(claim_amount + np.random.normal(0, 300, n_samples), 0)
    y = pd.Series(claim_amount, name="claim_amount")

    return X, y, "regression", "Insurance Claims (synthetic)"


# =============================================================================
# INRIA-SODA Tabular Benchmark Datasets
# =============================================================================

# INRIA-SODA dataset configurations
# Format: (config_name, task_type, description)
INRIA_DATASETS = {
    # Classification - Numerical features
    "higgs": ("clf_num_Higgs", "classification", "Higgs boson detection (physics)"),
    "covertype": ("clf_num_covertype", "classification", "Forest cover type prediction"),
    "jannis": ("clf_num_jannis", "classification", "Multi-class classification"),
    "miniboone": ("clf_num_MiniBooNE", "classification", "Particle physics classification"),
    "california": ("clf_num_california", "classification", "California housing (binned)"),
    "credit": ("clf_num_credit", "classification", "Credit approval prediction"),
    "bank_marketing": ("clf_num_bank-marketing", "classification", "Bank marketing response"),
    "diabetes": ("clf_num_Diabetes130US", "classification", "Diabetes readmission"),
    "bioresponse": ("clf_num_Bioresponse", "classification", "Biological response prediction"),
    "magic_telescope": ("clf_num_MagicTelescope", "classification", "Gamma/hadron classification"),
    # Classification - Categorical features
    "electricity": ("clf_cat_electricity", "classification", "Electricity price direction"),
    "covertype_cat": ("clf_cat_covertype", "classification", "Forest cover (categorical)"),
    "eye_movements": ("clf_cat_eye_movements", "classification", "Eye movement classification"),
    "road_safety": ("clf_cat_road-safety", "classification", "Road safety prediction"),
    "albert": ("clf_cat_albert", "classification", "Albert dataset"),
    # Regression - Numerical features
    "diamonds": ("reg_num_diamonds", "regression", "Diamond price prediction"),
    "house_sales": ("reg_num_house_sales", "regression", "House sale price prediction"),
    "houses": ("reg_num_houses", "regression", "House value prediction"),
    "wine_quality": ("reg_num_wine_quality", "regression", "Wine quality score"),
    "abalone": ("reg_num_abalone", "regression", "Abalone age prediction"),
    "superconduct": ("reg_num_superconduct", "regression", "Superconductor temperature"),
    "cpu_act": ("reg_num_cpu_act", "regression", "CPU activity prediction"),
    "elevators": ("reg_num_elevators", "regression", "Elevator control"),
    "miami_housing": ("reg_num_MiamiHousing2016", "regression", "Miami housing prices"),
    "bike_sharing_inria": ("reg_num_Bike_Sharing_Demand", "regression", "Bike rental demand"),
    # Regression - Categorical features
    "delays_zurich": ("reg_cat_delays_zurich_transport", "regression", "Zurich transport delays"),
    "allstate_claims": ("reg_cat_Allstate_Claims_Severity", "regression", "Insurance claim severity"),
    "mercedes_benz": ("reg_cat_Mercedes_Benz_Greener_Manufacturing", "regression", "Manufacturing time"),
    "nyc_taxi": ("reg_cat_nyc-taxi-green-dec-2016", "regression", "NYC taxi trip duration"),
    "brazilian_houses": ("reg_cat_Brazilian_houses", "regression", "Brazilian house prices"),
}


def load_inria_dataset(dataset_name: str, max_samples: int = 50000):
    """
    Load a dataset from the INRIA-SODA tabular benchmark on HuggingFace.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'higgs', 'diamonds', 'wine_quality').
        Use list_inria_datasets() to see available datasets.
    max_samples : int, default=50000
        Maximum number of samples to load (for large datasets).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    task : str
        Task type ('classification' or 'regression').
    name : str
        Human-readable dataset name.

    Examples
    --------
    >>> X, y, task, name = load_inria_dataset('diamonds')
    >>> print(f"Loaded {name}: {X.shape}")
    """
    if dataset_name not in INRIA_DATASETS:
        available = list(INRIA_DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    config_name, task, description = INRIA_DATASETS[dataset_name]

    try:
        from datasets import load_dataset

        ds = load_dataset("inria-soda/tabular-benchmark", config_name, split="train")
        df = ds.to_pandas()

        # Target is always the last column
        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Sample if too large
        if len(X) > max_samples:
            idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)

        return X, y, task, f"{description} (INRIA)"

    except Exception as e:
        print(f"Warning: Could not load INRIA dataset '{dataset_name}': {e}")
        raise


def list_inria_datasets():
    """List all available INRIA-SODA benchmark datasets."""
    return list(INRIA_DATASETS.keys())


def get_inria_dataset_info(dataset_name: str) -> dict:
    """Get metadata for an INRIA dataset."""
    if dataset_name not in INRIA_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    config_name, task, description = INRIA_DATASETS[dataset_name]
    return {
        "name": dataset_name,
        "config": config_name,
        "task": task,
        "description": description,
    }


# =============================================================================
# HuggingFace Dataset Loaders
# =============================================================================


def load_spotify_tracks(max_samples: int = 50000):
    """
    Audio features dataset - regression (popularity prediction).
    Synthetic dataset with interaction and polynomial effects in audio features.
    """
    np.random.seed(42)
    n = 5000

    danceability = np.random.uniform(0, 1, n)
    energy = np.random.uniform(0, 1, n)
    loudness = np.random.uniform(-40, 0, n)
    speechiness = np.random.uniform(0, 0.5, n)
    acousticness = np.random.uniform(0, 1, n)
    instrumentalness = np.random.uniform(0, 1, n)
    liveness = np.random.uniform(0, 0.5, n)
    valence = np.random.uniform(0, 1, n)
    tempo = np.random.uniform(60, 200, n)
    duration_ms = np.random.uniform(60000, 600000, n)
    key = np.random.randint(0, 12, n)
    mode = np.random.choice([0, 1], n)
    time_signature = np.random.choice([3, 4, 5], n, p=[0.1, 0.8, 0.1])

    X = pd.DataFrame(
        {
            "danceability": danceability,
            "energy": energy,
            "loudness": loudness,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "duration_ms": duration_ms,
            "key": key,
            "mode": mode,
            "time_signature": time_signature,
        }
    )

    # Popularity dominated by interaction and polynomial effects
    popularity = (
        15
        # Modest linear (20%)
        + 5 * danceability
        + 3 * energy
        # Dominant interaction/polynomial terms (70%)
        + 25 * danceability * energy  # Strong danceability-energy interaction
        + 15 * np.sqrt(valence * danceability)  # sqrt interaction
        + 12 * (loudness + 40) / 40 * energy  # Loudness-energy interaction
        - 18 * acousticness * instrumentalness  # Acoustic-instrumental interaction
        - 8 * speechiness**2  # Quadratic speechiness
        + 10 * np.log1p(tempo - 60) * danceability  # Log-tempo interaction
        - 5 * liveness * (1 - energy)  # Live-low-energy penalty
        + 8 * valence * (1 - acousticness)  # Happy electronic bonus
        + 6 * danceability * valence * energy  # Three-way interaction
        + 4 * energy**2  # Quadratic energy
        - 3 * acousticness**2  # Quadratic acousticness penalty
    )
    popularity = np.clip(popularity + np.random.normal(0, 3, n), 0, 100)

    y = pd.Series(popularity, name="popularity")

    return X, y, "regression", "Audio Features (synthetic)"


def load_fake_news(max_samples: int = 20000):
    """
    Load Fake News dataset from HuggingFace.

    Text classification task for fake news detection.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with text columns.
    y : pd.Series
        Labels (0=real, 1=fake).
    task : str
        Task type ('text_classification').
    name : str
        Dataset name.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("GonzaloA/fake_news", split="train")
        df = ds.to_pandas()

        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        # Select relevant columns
        text_cols = ["title", "text"]
        available_cols = [c for c in text_cols if c in df.columns]

        if not available_cols:
            raise ValueError("No text columns found in dataset")

        target = "label"
        if target not in df.columns:
            # Try alternative target names
            for alt in ["fake", "is_fake", "class"]:
                if alt in df.columns:
                    target = alt
                    break

        df = df.dropna(subset=available_cols + [target])

        X = df[available_cols].copy()
        y = df[target]

        return X, y, "text_classification", "Fake News (HuggingFace)"

    except Exception as e:
        print(f"Warning: Could not load Fake News dataset: {e}")
        raise


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
    Salary prediction dataset with interaction and ratio effects.
    Tests feature engineering's ability to discover complex relationships.
    """
    np.random.seed(random_state)

    experience_years = np.random.exponential(5, n_samples).clip(0, 30)
    education_level = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.35, 0.35, 0.15])
    industry_code = np.random.choice([0, 1, 2, 3, 4], n_samples)
    company_size = np.random.lognormal(5, 1.5, n_samples).clip(10, 100000).astype(int)
    city_cost_index = np.random.uniform(0.7, 1.5, n_samples)
    num_skills = np.random.poisson(5, n_samples)
    remote_ratio = np.random.uniform(0, 1, n_samples)
    team_size = np.random.poisson(8, n_samples).clip(1, 50)
    performance_score = np.random.uniform(1, 5, n_samples)
    certifications = np.random.poisson(1, n_samples)

    X = pd.DataFrame(
        {
            "experience_years": experience_years,
            "education_level": education_level,
            "industry_code": industry_code,
            "company_size": company_size,
            "city_cost_index": city_cost_index,
            "num_skills": num_skills,
            "remote_ratio": remote_ratio,
            "team_size": team_size,
            "performance_score": performance_score,
            "certifications": certifications,
        }
    )

    # Salary dominated by interaction and ratio effects
    salary = (
        35000
        # Modest linear terms (25%)
        + 2000 * experience_years
        + 5000 * education_level
        + 10000 * city_cost_index
        # Dominant interaction/ratio terms (65%)
        + 15000 * experience_years * education_level / 12  # Experience-education interaction (strong)
        + 12000 * np.log1p(company_size) / 10  # Log company size (strong)
        + 8000 * num_skills * performance_score / 5  # Skills-performance interaction (strong)
        + 10000 * experience_years * city_cost_index / 10  # Experience-city interaction (strong)
        - 5000 * remote_ratio * city_cost_index  # Remote discount
        + 4000 * np.sqrt(team_size) * education_level  # Team-education interaction
        + 6000 * certifications * experience_years / 10  # Certifications-experience
        + 3000 * experience_years**0.5 * performance_score  # sqrt-exp × performance
        + 2000 * education_level**2  # Quadratic education
    )
    salary = np.maximum(salary + np.random.normal(0, 4000, n_samples), 30000)

    y = pd.Series(salary, name="salary")

    return X, y, "regression", "Salary Prediction (synthetic)"


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

    # Additional features with discriminative power for categories
    hour_published = np.random.randint(0, 24, n_samples)
    source_credibility = np.random.uniform(0.5, 1.0, n_samples)

    # Generate category-correlated numeric features
    labels_arr = np.array(labels)

    # Add features that correlate with categories through interactions
    entity_count = np.zeros(n_samples)
    sentiment_polarity = np.zeros(n_samples)
    for i in range(n_samples):
        if labels_arr[i] == 0:  # Business
            entity_count[i] = np.random.poisson(3)
            sentiment_polarity[i] = np.random.normal(0.1, 0.3)
        elif labels_arr[i] == 1:  # Technology
            entity_count[i] = np.random.poisson(2)
            sentiment_polarity[i] = np.random.normal(0.3, 0.3)
        elif labels_arr[i] == 2:  # Sports
            entity_count[i] = np.random.poisson(4)
            sentiment_polarity[i] = np.random.normal(0.2, 0.4)
        elif labels_arr[i] == 3:  # Entertainment
            entity_count[i] = np.random.poisson(2)
            sentiment_polarity[i] = np.random.normal(0.4, 0.3)
        else:  # Politics
            entity_count[i] = np.random.poisson(3)
            sentiment_polarity[i] = np.random.normal(-0.1, 0.4)

    X = pd.DataFrame(
        {
            "headline": headlines,
            "word_count": word_counts,
            "has_numbers": has_numbers,
            "hour_published": hour_published,
            "source_credibility": source_credibility,
            "entity_count": entity_count.astype(int),
            "sentiment_polarity": sentiment_polarity,
        }
    )

    y = pd.Series(labels, name="category")

    return X, y, "text_classification", "News Headlines (text)"


def create_customer_support_dataset(n_samples=2000, random_state=42):
    """
    Ticket priority classification with interaction effects.
    Tests feature engineering's ability to discover urgency patterns from numeric features.
    """
    np.random.seed(random_state)

    # Ticket features
    urgency_score = np.random.uniform(0, 10, n_samples)
    customer_lifetime_value = np.random.lognormal(7, 1.5, n_samples)
    account_age_days = np.random.exponential(365, n_samples).astype(int)
    previous_tickets = np.random.poisson(3, n_samples)
    response_time_hours = np.random.exponential(4, n_samples)
    product_tier = np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.3, 0.25, 0.15])
    num_affected_users = np.random.poisson(5, n_samples)
    is_weekend = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    system_load_pct = np.random.uniform(20, 95, n_samples)
    error_count_24h = np.random.poisson(2, n_samples)

    X = pd.DataFrame(
        {
            "urgency_score": urgency_score,
            "customer_lifetime_value": customer_lifetime_value,
            "account_age_days": account_age_days,
            "previous_tickets": previous_tickets,
            "response_time_hours": response_time_hours,
            "product_tier": product_tier,
            "num_affected_users": num_affected_users,
            "is_weekend": is_weekend,
            "system_load_pct": system_load_pct,
            "error_count_24h": error_count_24h,
        }
    )

    # Priority dominated by interaction and ratio effects
    impact_score = num_affected_users * error_count_24h
    value_urgency = np.log1p(customer_lifetime_value) * urgency_score / 100
    priority_score = (
        # Modest linear (20%)
        0.15 * urgency_score / 10
        + 0.05 * product_tier / 4
        # Dominant interaction/ratio terms (70%)
        + 0.20 * urgency_score * product_tier / 40  # Urgency-tier interaction
        + 0.15 * np.sqrt(impact_score + 1)  # sqrt impact interaction
        + 0.12 * system_load_pct * error_count_24h / 500  # Load-error interaction
        + 0.08 * value_urgency  # Log-value × urgency interaction
        + 0.06 * (previous_tickets * urgency_score) / 30  # History-urgency interaction
        + 0.04 * (response_time_hours * urgency_score) / 50  # Response-urgency interaction
        + 0.04 * (is_weekend * system_load_pct) / 100  # Weekend-load interaction
        - 0.03 * np.clip(account_age_days / 1000, 0, 1)
        + np.random.normal(0, 0.04, n_samples)
    )
    # Map to priority classes
    priority_labels = np.digitize(priority_score, bins=[0.2, 0.35, 0.55])

    y = pd.Series(priority_labels, name="priority")

    return X, y, "classification", "Ticket Priority (synthetic)"


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

    # Diagnosis based on numeric feature interactions (not text identity)
    ages_arr = np.array(ages)
    bmis_arr = np.array(bmis)
    bps_arr = np.array(systolic_bps)
    glucose_arr = np.array(glucose_levels)

    # Score for each condition based on numeric features and their interactions
    diabetes_score = (
        0.3 * np.clip((glucose_arr - 120) / 100, 0, 1)
        + 0.2 * np.clip((bmis_arr - 25) / 10, 0, 1)
        + 0.15 * np.clip(glucose_arr * bmis_arr / 7500 - 0.4, 0, 1)  # Glucose-BMI interaction
    )
    hyper_score = (
        0.3 * np.clip((bps_arr - 130) / 60, 0, 1)
        + 0.1 * np.clip((bmis_arr - 25) / 10, 0, 1)
        + 0.15 * np.clip(bps_arr * ages_arr / 12000 - 0.5, 0, 1)  # BP-age interaction
    )
    heart_score = (
        0.15 * np.clip((bps_arr - 120) / 60, 0, 1)
        + 0.15 * np.clip((ages_arr - 40) / 40, 0, 1)
        + 0.2 * np.clip(bps_arr * bmis_arr / 5000 - 0.5, 0, 1)  # BP-BMI interaction
    )
    resp_score = np.random.uniform(0, 0.2, n_samples)

    # Assign condition based on highest score with noise
    score_matrix = np.column_stack(
        [
            np.random.uniform(0.1, 0.3, n_samples),  # Healthy baseline
            diabetes_score + np.random.normal(0, 0.05, n_samples),
            hyper_score + np.random.normal(0, 0.05, n_samples),
            heart_score + np.random.normal(0, 0.05, n_samples),
            resp_score + np.random.normal(0, 0.05, n_samples),
        ]
    )
    labels = score_matrix.argmax(axis=1).tolist()

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
    E-commerce product dataset - sales prediction from numeric features.
    Tests feature engineering on interaction-heavy product metrics.
    """
    np.random.seed(random_state)

    category_code = np.random.choice([0, 1, 2, 3, 4], n_samples)
    price = np.random.lognormal(3.5, 1, n_samples)
    rating = np.clip(np.random.normal(3.8, 0.8, n_samples), 1, 5)
    reviews_count = np.random.poisson(50, n_samples)
    discount_pct = np.random.uniform(0, 50, n_samples)
    inventory_level = np.random.poisson(100, n_samples)
    days_listed = np.random.exponential(30, n_samples).astype(int)
    seller_rating = np.clip(np.random.normal(4.2, 0.5, n_samples), 1, 5)
    return_rate = np.random.uniform(0, 0.3, n_samples)
    page_views = np.random.poisson(200, n_samples)

    X = pd.DataFrame(
        {
            "category_code": category_code,
            "price": price,
            "rating": rating,
            "reviews_count": reviews_count,
            "discount_pct": discount_pct,
            "inventory_level": inventory_level,
            "days_listed": days_listed,
            "seller_rating": seller_rating,
            "return_rate": return_rate,
            "page_views": page_views,
        }
    )

    # Sales dominated by interaction and ratio effects
    log_reviews = np.log1p(reviews_count)
    effective_price = price * (1 - discount_pct / 100)
    sales = (
        # Modest linear (20%)
        10
        + 3 * rating
        + 2 * log_reviews
        # Dominant interaction/ratio terms (70%)
        + 8 * rating * log_reviews / 5  # Rating-reviews interaction
        - 4 * effective_price / 50  # Effective price (price × discount)
        + 5 * seller_rating * rating / 5  # Seller-product rating interaction
        - 6 * return_rate * price / 20  # Return-price interaction
        + 3 * np.sqrt(page_views) * rating / 10  # Views-rating interaction
        + 2 * discount_pct * log_reviews / 50  # Discount-reviews interaction
        - 1 * np.log1p(days_listed) * (1 - rating / 5)  # Staleness penalty
    )
    sales = np.maximum(sales + np.random.normal(0, 3, n_samples), 1).astype(int)

    y = pd.Series(sales, name="monthly_sales")

    return X, y, "regression", "E-commerce Products (synthetic)"


def get_text_datasets():
    """Return text/semantic benchmark dataset loaders (legacy API)."""
    return [
        create_product_reviews_dataset,
        create_job_postings_dataset,
        create_news_classification_dataset,
        create_customer_support_dataset,
        create_medical_notes_dataset,
        create_ecommerce_product_dataset,
    ]


# =============================================================================
# Unified Dataset Registry
# =============================================================================

# Dataset categories
CATEGORY_CLASSIFICATION = "classification"
CATEGORY_REGRESSION = "regression"
CATEGORY_FORECASTING = "forecasting"
CATEGORY_TEXT = "text"

# Dataset source types
SOURCE_REAL_WORLD = "real_world"
SOURCE_SYNTHETIC = "synthetic"

# Source registry: {name: source_type}
# Tracks whether each dataset is real-world or synthetic
DATASET_SOURCE: dict[str, str] = {}

# Master registry: {name: (loader_func, category, description)}
# All datasets are registered here with their category
DATASET_REGISTRY: dict[str, tuple] = {
    # === Classification datasets (synthetic) ===
    "titanic": (load_titanic_dataset, CATEGORY_CLASSIFICATION, "Titanic survival (synthetic)"),
    "credit_card_fraud": (
        load_credit_card_fraud_dataset,
        CATEGORY_CLASSIFICATION,
        "Credit card fraud (synthetic)",
    ),
    "employee_attrition": (
        load_employee_attrition_dataset,
        CATEGORY_CLASSIFICATION,
        "Employee attrition (synthetic)",
    ),
    "credit_risk": (create_credit_risk_dataset, CATEGORY_CLASSIFICATION, "Credit risk (synthetic)"),
    "medical_diagnosis": (
        create_medical_diagnosis_dataset,
        CATEGORY_CLASSIFICATION,
        "Medical diagnosis (synthetic)",
    ),
    "complex_classification": (
        create_complex_classification_dataset,
        CATEGORY_CLASSIFICATION,
        "Complex classification (synthetic)",
    ),
    "interaction_classification": (
        create_interaction_classification_dataset,
        CATEGORY_CLASSIFICATION,
        "Interaction classification (synthetic)",
    ),
    "customer_churn": (create_customer_churn_dataset, CATEGORY_CLASSIFICATION, "Customer churn (synthetic)"),
    "xor_classification": (
        create_xor_classification_dataset,
        CATEGORY_CLASSIFICATION,
        "XOR classification (synthetic)",
    ),
    "polynomial_classification": (
        create_polynomial_classification_dataset,
        CATEGORY_CLASSIFICATION,
        "Polynomial classification (synthetic)",
    ),
    # === Regression datasets (synthetic) ===
    "house_prices": (load_house_prices_dataset, CATEGORY_REGRESSION, "House prices (synthetic)"),
    "bike_sharing": (load_bike_sharing_dataset, CATEGORY_REGRESSION, "Bike sharing (synthetic)"),
    "complex_regression": (
        create_complex_regression_dataset,
        CATEGORY_REGRESSION,
        "Complex regression (synthetic)",
    ),
    "polynomial_regression": (
        create_polynomial_regression_dataset,
        CATEGORY_REGRESSION,
        "Polynomial regression (synthetic)",
    ),
    "ratio_regression": (
        create_ratio_regression_dataset,
        CATEGORY_REGRESSION,
        "Ratio regression (synthetic)",
    ),
    "nonlinear_regression": (
        create_nonlinear_regression_dataset,
        CATEGORY_REGRESSION,
        "Nonlinear regression (synthetic)",
    ),
    "insurance_claims": (create_insurance_claims_dataset, CATEGORY_REGRESSION, "Insurance claims (synthetic)"),
    "xor_regression": (
        create_xor_regression_dataset,
        CATEGORY_REGRESSION,
        "XOR regression (synthetic)",
    ),
    "quadratic_heavy_regression": (
        create_quadratic_heavy_regression_dataset,
        CATEGORY_REGRESSION,
        "Quadratic heavy regression (synthetic)",
    ),
    "pairwise_product_regression": (
        create_pairwise_product_regression_dataset,
        CATEGORY_REGRESSION,
        "Pairwise product regression (synthetic)",
    ),
    "sqrt_log_regression": (
        create_sqrt_log_regression_dataset,
        CATEGORY_REGRESSION,
        "Sqrt log regression (synthetic)",
    ),
    "triple_interaction_regression": (
        create_triple_interaction_regression_dataset,
        CATEGORY_REGRESSION,
        "Triple interaction regression (synthetic)",
    ),
    # === Forecasting datasets (synthetic) ===
    "sensor_anomaly": (
        create_sensor_anomaly_timeseries,
        CATEGORY_FORECASTING,
        "Sensor anomaly (synthetic)",
    ),
    "retail_demand": (create_retail_demand_timeseries, CATEGORY_FORECASTING, "Retail demand (synthetic)"),
    "server_latency": (create_server_latency_timeseries, CATEGORY_FORECASTING, "Server latency (synthetic)"),
    # === Text datasets (synthetic) ===
    "product_reviews": (create_product_reviews_dataset, CATEGORY_TEXT, "Product reviews (synthetic)"),
    "job_postings": (create_job_postings_dataset, CATEGORY_REGRESSION, "Salary prediction (synthetic)"),
    "news_classification": (
        create_news_classification_dataset,
        CATEGORY_TEXT,
        "News classification (synthetic)",
    ),
    "customer_support": (create_customer_support_dataset, CATEGORY_CLASSIFICATION, "Ticket priority (synthetic)"),
    "medical_notes": (create_medical_notes_dataset, CATEGORY_TEXT, "Medical notes (synthetic)"),
    "ecommerce_product": (create_ecommerce_product_dataset, CATEGORY_REGRESSION, "E-commerce products (synthetic)"),
    # === HuggingFace datasets ===
    "spotify_tracks": (load_spotify_tracks, CATEGORY_REGRESSION, "Audio features (synthetic)"),
    "fake_news": (load_fake_news, CATEGORY_TEXT, "Fake news (HuggingFace)"),
}

# Add INRIA datasets to registry (they're loaded dynamically)
for _name, (_config, _task, _desc) in INRIA_DATASETS.items():
    _category = CATEGORY_CLASSIFICATION if _task == "classification" else CATEGORY_REGRESSION
    DATASET_REGISTRY[_name] = (lambda n=_name: load_inria_dataset(n), _category, f"{_desc} (INRIA)")
    DATASET_SOURCE[_name] = SOURCE_REAL_WORLD

# Tag synthetic datasets
for _name in [
    "titanic",
    "credit_card_fraud",
    "employee_attrition",
    "credit_risk",
    "medical_diagnosis",
    "complex_classification",
    "interaction_classification",
    "customer_churn",
    "xor_classification",
    "polynomial_classification",
    "house_prices",
    "bike_sharing",
    "complex_regression",
    "polynomial_regression",
    "ratio_regression",
    "nonlinear_regression",
    "insurance_claims",
    "xor_regression",
    "quadratic_heavy_regression",
    "pairwise_product_regression",
    "sqrt_log_regression",
    "triple_interaction_regression",
    "sensor_anomaly",
    "retail_demand",
    "server_latency",
    "product_reviews",
    "job_postings",
    "news_classification",
    "customer_support",
    "medical_notes",
    "ecommerce_product",
    "spotify_tracks",
]:
    DATASET_SOURCE[_name] = SOURCE_SYNTHETIC

# Tag HuggingFace datasets as real-world
DATASET_SOURCE["fake_news"] = SOURCE_REAL_WORLD


def is_real_world(dataset_name: str) -> bool:
    """Check whether a dataset is real-world (not synthetic)."""
    return DATASET_SOURCE.get(dataset_name, SOURCE_SYNTHETIC) == SOURCE_REAL_WORLD


def list_real_world_datasets(category: str | None = None) -> list[str]:
    """List only real-world datasets, optionally filtered by category."""
    all_names = list_datasets(category)
    return [n for n in all_names if is_real_world(n)]


def list_synthetic_datasets(category: str | None = None) -> list[str]:
    """List only synthetic datasets, optionally filtered by category."""
    all_names = list_datasets(category)
    return [n for n in all_names if not is_real_world(n)]


def list_datasets(category: str | None = None) -> list[str]:
    """
    List available dataset names.

    Parameters
    ----------
    category : str, optional
        Filter by category: 'classification', 'regression', 'forecasting', 'text'.
        If None, returns all datasets.

    Returns
    -------
    list[str]
        List of dataset names.

    Examples
    --------
    >>> list_datasets()  # All datasets
    >>> list_datasets('classification')  # Only classification datasets
    >>> list_datasets('regression')  # Only regression datasets
    """
    if category is None:
        return list(DATASET_REGISTRY.keys())

    return [name for name, (_, cat, _) in DATASET_REGISTRY.items() if cat == category]


def load_dataset(name: str, **kwargs) -> tuple:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Dataset name. Use list_datasets() to see available names.
    **kwargs
        Additional arguments passed to the loader function.

    Returns
    -------
    tuple
        (X, y, task_type, dataset_name) tuple.

    Examples
    --------
    >>> X, y, task, name = load_dataset('titanic')
    >>> X, y, task, name = load_dataset('diamonds')  # INRIA dataset
    >>> X, y, task, name = load_dataset('spotify_tracks', max_samples=10000)
    """
    if name not in DATASET_REGISTRY:
        available = list_datasets()
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available[:10]}... ({len(available)} total)")

    loader, _category, _desc = DATASET_REGISTRY[name]
    return loader(**kwargs) if kwargs else loader()


def load_datasets(category: str | None = None, **kwargs) -> list[tuple]:
    """
    Load multiple datasets by category.

    Parameters
    ----------
    category : str, optional
        Category to filter: 'classification', 'regression', 'forecasting', 'text'.
        If None, loads all datasets.
    **kwargs
        Additional arguments passed to each loader function.

    Returns
    -------
    list[tuple]
        List of (X, y, task_type, dataset_name) tuples.

    Examples
    --------
    >>> datasets = load_datasets('classification')
    >>> for X, y, task, name in datasets:
    ...     print(f"{name}: {X.shape}")

    >>> all_data = load_datasets()  # Load all datasets
    """
    names = list_datasets(category)
    results = []

    for name in names:
        try:
            result = load_dataset(name, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load '{name}': {e}")

    return results


def load_all_datasets(**kwargs) -> list[tuple]:
    """
    Load all available datasets.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to each loader function.

    Returns
    -------
    list[tuple]
        List of (X, y, task_type, dataset_name) tuples.

    Examples
    --------
    >>> all_datasets = load_all_datasets()
    >>> print(f"Loaded {len(all_datasets)} datasets")
    """
    return load_datasets(category=None, **kwargs)


def get_dataset_info(name: str) -> dict:
    """
    Get metadata about a specific dataset.

    Parameters
    ----------
    name : str
        Dataset name.

    Returns
    -------
    dict
        Dictionary with name, category, description, and loader.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'")

    loader, category, description = DATASET_REGISTRY[name]
    return {
        "name": name,
        "category": category,
        "description": description,
        "loader": loader,
    }


def get_category_summary() -> dict[str, int]:
    """
    Get count of datasets per category.

    Returns
    -------
    dict
        Dictionary mapping category names to dataset counts.
    """
    summary: dict[str, int] = {}
    for _, (_, category, _) in DATASET_REGISTRY.items():
        summary[category] = summary.get(category, 0) + 1
    return summary


# Backward compatibility aliases
def load_dataset_by_name(name: str):
    """Load dataset by name (legacy API - use load_dataset instead)."""
    return load_dataset(name)


def get_all_datasets():
    """Return all standard tabular dataset loaders (legacy API)."""
    return [
        load_titanic_dataset,
        load_house_prices_dataset,
        load_credit_card_fraud_dataset,
        load_bike_sharing_dataset,
        load_employee_attrition_dataset,
        create_credit_risk_dataset,
        create_medical_diagnosis_dataset,
        create_complex_regression_dataset,
        create_complex_classification_dataset,
        create_customer_churn_dataset,
        create_insurance_claims_dataset,
    ]


def get_timeseries_datasets():
    """Return time series dataset loaders (legacy API)."""
    return [
        create_sensor_anomaly_timeseries,
        create_retail_demand_timeseries,
        create_server_latency_timeseries,
    ]


def get_huggingface_datasets():
    """Return HuggingFace dataset loaders (legacy API)."""
    return {
        "spotify_tracks": load_spotify_tracks,
        "fake_news": load_fake_news,
    }


def get_inria_datasets():
    """Return INRIA dataset loaders (legacy API)."""
    return {name: lambda n=name: load_inria_dataset(n) for name in INRIA_DATASETS}


if __name__ == "__main__":
    print("Benchmark Datasets Summary")
    print("=" * 60)

    # Show category summary
    summary = get_category_summary()
    print(f"\nTotal datasets: {sum(summary.values())}")
    for cat, count in sorted(summary.items()):
        print(f"  {cat}: {count}")

    # Show a few examples from each category
    print("\n\nSample datasets by category:")
    for cat in [CATEGORY_CLASSIFICATION, CATEGORY_REGRESSION, CATEGORY_FORECASTING, CATEGORY_TEXT]:
        names = list_datasets(cat)[:3]
        print(f"\n{cat.upper()} ({len(list_datasets(cat))} total):")
        for name in names:
            info = get_dataset_info(name)
            print(f"  - {name}: {info['description']}")

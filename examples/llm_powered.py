"""
Example: LLM-Powered Feature Engineering with FeatCopilot

This example demonstrates the unique LLM-powered capabilities
of FeatCopilot using GitHub Copilot SDK or LiteLLM.

LLM Backend Options:
- GitHub Copilot SDK (default): Requires copilot-sdk package
- LiteLLM: Supports 100+ LLM providers (OpenAI, Anthropic, Azure, etc.)

New Feature: Transform Rules
- Create reusable transformation rules from natural language
- Save rules for reuse across datasets
- Find and apply matching rules automatically
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from featcopilot import AutoFeatureEngineer, TransformRule, TransformRuleGenerator, TransformRuleStore


def create_healthcare_data(n_samples=1000):
    """
    Create synthetic healthcare dataset with interaction effects.

    The target depends on medical ratios and interactions that
    benefit significantly from feature engineering.
    """
    np.random.seed(42)

    # Create features with more noise to make raw signal harder to detect
    data = pd.DataFrame(
        {
            "age": np.random.randint(20, 90, n_samples),
            "bmi": np.random.normal(27, 6, n_samples).clip(16, 45),
            "blood_pressure_systolic": np.random.normal(125, 25, n_samples).clip(85, 200),
            "blood_pressure_diastolic": np.random.normal(82, 15, n_samples).clip(55, 120),
            "cholesterol_total": np.random.normal(210, 50, n_samples).clip(100, 350),
            "cholesterol_hdl": np.random.normal(52, 18, n_samples).clip(20, 100),
            "cholesterol_ldl": np.random.normal(125, 40, n_samples).clip(50, 250),
            "glucose_fasting": np.random.normal(105, 30, n_samples).clip(65, 200),
            "hba1c": np.random.normal(5.7, 1.4, n_samples).clip(4, 12),
            "smoking_years": np.random.exponential(8, n_samples),
            "exercise_hours_weekly": np.random.exponential(4, n_samples),
        }
    )

    # Key insight: Use ONLY ratio/product features for signal
    # No linear terms that the baseline can easily learn

    # Product features (TabularEngine: col1_x_col2)
    glucose_x_hba1c = data["glucose_fasting"] * data["hba1c"]
    bmi_x_glucose = data["bmi"] * data["glucose_fasting"]
    age_x_smoking = data["age"] * data["smoking_years"]

    # Ratio features (TabularEngine: col1_div_col2)
    chol_ratio = data["cholesterol_total"] / (data["cholesterol_hdl"] + 1)
    ldl_hdl_ratio = data["cholesterol_ldl"] / (data["cholesterol_hdl"] + 1)
    glucose_exercise_ratio = data["glucose_fasting"] / (data["exercise_hours_weekly"] + 1)

    # Build risk ONLY from engineered features (no raw features)
    risk = (
        -6.5
        + 0.008 * glucose_x_hba1c  # glucose * hba1c product
        + 0.002 * bmi_x_glucose  # bmi * glucose product
        + 0.003 * age_x_smoking  # age * smoking product
        + 0.12 * chol_ratio  # cholesterol ratio
        + 0.08 * ldl_hdl_ratio  # ldl/hdl ratio
        + 0.015 * glucose_exercise_ratio  # glucose/exercise ratio
    )
    # Add noise to make it challenging
    risk = risk + np.random.normal(0, 0.3, n_samples)
    risk = 1 / (1 + np.exp(-risk))
    data["diabetes_risk"] = (np.random.random(n_samples) < risk).astype(int)

    return data


def main():
    print("=" * 70)
    print("FeatCopilot LLM-Powered Feature Engineering Example")
    print("=" * 70)

    # Create sample data
    print("\n1. Creating healthcare dataset...")
    data = create_healthcare_data(1000)
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
    # You can choose between different LLM backends:
    #
    # Option 1: GitHub Copilot SDK (default)
    # llm_config = {
    #     "model": "gpt-5.2",
    #     "max_suggestions": 15,
    #     "domain": "healthcare",
    # }
    #
    # Option 2: LiteLLM with OpenAI
    # llm_config = {
    #     "model": "gpt-4o",
    #     "backend": "litellm",
    #     "max_suggestions": 15,
    #     "domain": "healthcare",
    #     # API key can be set via environment variable OPENAI_API_KEY
    #     # or passed directly: "api_key": "your-api-key"
    # }
    #
    # Option 3: LiteLLM with Anthropic Claude
    # llm_config = {
    #     "model": "claude-3-opus",
    #     "backend": "litellm",
    #     "max_suggestions": 15,
    #     "domain": "healthcare",
    # }
    #
    # Option 4: LiteLLM with Azure OpenAI
    # llm_config = {
    #     "model": "azure/your-deployment-name",
    #     "backend": "litellm",
    #     "max_suggestions": 15,
    #     "domain": "healthcare",
    # }
    #
    # Option 5: LiteLLM with local Ollama
    # llm_config = {
    #     "model": "ollama/llama2",
    #     "backend": "litellm",
    #     "api_base": "http://localhost:11434",
    #     "max_suggestions": 15,
    #     "domain": "healthcare",
    # }
    #
    # Option 6: LiteLLM with GitHub Marketplace Models
    # Access models from https://github.com/marketplace/models
    # Uses GITHUB_API_KEY environment variable for authentication.
    # llm_config = {
    #     "model": "github/gpt-4o",  # or github/Llama-3.2-11B-Vision-Instruct, github/Phi-4
    #     "backend": "litellm",
    #     "max_suggestions": 15,
    #     "domain": "healthcare",
    # }
    #
    # Option 7: LiteLLM with GitHub Copilot Chat API
    # Access GitHub Copilot's Chat API using OAuth device flow authentication.
    # Requires a paid GitHub Copilot subscription.
    # On first use, you'll be prompted to authenticate via browser.
    # llm_config = {
    #     "model": "github_copilot/gpt-4",  # or github_copilot/gpt-5.1-codex
    #     "backend": "litellm",
    #     "max_suggestions": 15,
    #     "domain": "healthcare",
    # }

    print("\n3. Applying LLM-powered feature engineering...")
    print("   (Note: If LLM SDK not available, mock responses will be used)")

    engineer = AutoFeatureEngineer(
        engines=["tabular", "llm"],
        max_features=40,
        llm_config={
            "model": "gpt-5.2",
            "max_suggestions": 15,
            "domain": "healthcare",
            # Uncomment the following lines to use LiteLLM instead:
            # "backend": "litellm",
            # "model": "gpt-4o",  # OpenAI
            # "model": "claude-3-opus",  # Anthropic
            # "model": "github/gpt-4o",  # GitHub Marketplace Models
            # "model": "github_copilot/gpt-4",  # GitHub Copilot Chat API
            # "model": "ollama/llama2",  # Local Ollama
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

    # =========================================================================
    # NEW FEATURE: Transform Rules - Create and Reuse Custom Transformations
    # =========================================================================
    print("\n" + "=" * 70)
    print("7. Transform Rules: Create Reusable Feature Transformations")
    print("=" * 70)

    # Initialize the rule store and generator
    # Rules are saved to ~/.featcopilot/rules.json by default
    store = TransformRuleStore()
    generator = TransformRuleGenerator(store=store, verbose=True)

    # Generate a rule from natural language description
    print("\n   a) Generating rule from natural language...")
    rule = generator.generate_from_description(
        description="Calculate the ratio of glucose to HbA1c as a diabetes indicator",
        columns={"glucose_fasting": "float", "hba1c": "float"},
        tags=["healthcare", "diabetes", "ratio"],
        save=True,  # Save to store for future reuse
    )
    print(f"      Created rule: {rule.name}")
    print(f"      Code: {rule.code}")

    # Apply the rule to our data
    print("\n   b) Applying rule to dataset...")
    result = rule.apply(X_train)
    print(f"      Result shape: {result.shape}")
    print(f"      Sample values: {result.head(3).tolist()}")

    # Create a manual rule (without LLM)
    print("\n   c) Creating manual rule...")
    manual_rule = TransformRule(
        name="cholesterol_risk_score",
        description="Calculate cholesterol risk score from total, HDL, and LDL",
        code="result = (df['cholesterol_total'] - df['cholesterol_hdl']) / (df['cholesterol_hdl'] + 1e-8)",
        input_columns=["cholesterol_total", "cholesterol_hdl"],
        column_patterns=[".*cholesterol.*total.*", ".*cholesterol.*hdl.*"],
        tags=["healthcare", "cholesterol", "risk"],
    )
    store.save_rule(manual_rule)
    print(f"      Saved rule: {manual_rule.name}")

    # Demonstrate rule reuse on a different dataset with different column names
    print("\n   d) Reusing rules on new dataset with different column names...")
    new_data = pd.DataFrame(
        {
            "patient_total_cholesterol": [200, 220, 180],
            "patient_hdl_cholesterol": [50, 45, 60],
            "patient_ldl": [120, 140, 100],
        }
    )

    # Find matching rules
    matching_rules = store.find_matching_rules(
        columns=new_data.columns.tolist(),
        description="cholesterol",
    )

    if matching_rules:
        matched_rule, column_mapping = matching_rules[0]
        print(f"      Found matching rule: {matched_rule.name}")
        print(f"      Column mapping: {column_mapping}")

        # Apply with automatic column mapping
        new_result = matched_rule.apply(new_data, column_mapping=column_mapping)
        print(f"      Applied result: {new_result.tolist()}")

    # List all saved rules
    print("\n   e) All saved rules in store:")
    for r in store.list_rules():
        print(f"      - {r.name}: {r.description[:50]}...")
        print(f"        Tags: {r.tags}, Used: {r.usage_count} times")

    # Search rules by description
    print("\n   f) Searching rules by description 'diabetes'...")
    diabetes_rules = store.search_by_description("diabetes")
    print(f"      Found {len(diabetes_rules)} matching rules")

    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

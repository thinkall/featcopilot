# Domain-Specific Examples

FeatCopilot's LLM engine can generate domain-aware features. Here are examples for common domains.

## Healthcare

### Diabetes Risk Prediction

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'model': 'gpt-5',
        'domain': 'healthcare',
        'max_suggestions': 15
    }
)

X_fe = engineer.fit_transform(
    X, y,
    column_descriptions={
        'age': 'Patient age in years',
        'bmi': 'Body Mass Index',
        'glucose_fasting': 'Fasting blood glucose mg/dL',
        'hba1c': 'Hemoglobin A1c percentage',
        'blood_pressure': 'Systolic blood pressure mmHg',
        'family_history': 'Family history of diabetes (0/1)',
    },
    task_description="Predict Type 2 diabetes risk within 5 years"
)
```

### Expected LLM-Generated Features

- `bmi_glucose_interaction`: BMI × glucose interaction
- `metabolic_age_score`: Combined metabolic risk indicators
- `prediabetes_indicator`: Based on glucose/HbA1c thresholds
- `cardiovascular_risk`: Blood pressure and metabolic markers

---

## Finance

### Credit Default Prediction

```python
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'domain': 'finance',
        'max_suggestions': 15
    }
)

X_fe = engineer.fit_transform(
    X, y,
    column_descriptions={
        'income': 'Annual income in USD',
        'debt': 'Total outstanding debt',
        'credit_score': 'FICO credit score (300-850)',
        'employment_years': 'Years at current employer',
        'loan_amount': 'Requested loan amount',
        'num_accounts': 'Number of credit accounts',
        'late_payments': 'Number of late payments in last 2 years',
    },
    task_description="Predict loan default probability"
)
```

### Expected LLM-Generated Features

- `debt_to_income`: Debt-to-income ratio
- `loan_to_income`: Loan amount relative to income
- `credit_utilization`: Estimated credit utilization
- `payment_reliability`: Based on late payment history
- `employment_stability`: Employment tenure score

---

## Retail / E-commerce

### Customer Churn Prediction

```python
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'domain': 'retail',
        'max_suggestions': 15
    }
)

X_fe = engineer.fit_transform(
    X, y,
    column_descriptions={
        'days_since_purchase': 'Days since last purchase',
        'total_orders': 'Total number of orders',
        'total_spend': 'Total amount spent',
        'avg_order_value': 'Average order value',
        'returns_count': 'Number of returns',
        'customer_tenure_days': 'Days since first purchase',
        'email_opens': 'Number of marketing emails opened',
    },
    task_description="Predict customer churn in next 90 days"
)
```

### Expected LLM-Generated Features

- `recency_score`: Based on days since purchase
- `frequency_score`: Orders per time period
- `monetary_score`: Spending patterns
- `rfm_combined`: Combined RFM score
- `engagement_rate`: Email open rate
- `return_rate`: Returns as percentage of orders
- `customer_lifetime_value`: Estimated CLV

---

## Telecom

### Service Churn Prediction

```python
engineer = AutoFeatureEngineer(
    engines=['tabular', 'llm'],
    llm_config={
        'domain': 'telecom',
        'max_suggestions': 15
    }
)

X_fe = engineer.fit_transform(
    X, y,
    column_descriptions={
        'tenure_months': 'Months as customer',
        'monthly_charges': 'Monthly bill amount',
        'total_charges': 'Total charges to date',
        'contract_type': 'Month-to-month, 1-year, or 2-year',
        'num_services': 'Number of subscribed services',
        'support_tickets': 'Support tickets in last 6 months',
        'data_usage_gb': 'Average monthly data usage',
    },
    task_description="Predict telecom customer churn"
)
```

### Expected LLM-Generated Features

- `charges_per_service`: Monthly charges per service
- `contract_risk`: Risk based on contract type
- `support_intensity`: Support tickets relative to tenure
- `usage_trend`: Data usage patterns
- `customer_value`: Revenue per customer metrics

---

## Manufacturing

### Equipment Failure Prediction

```python
engineer = AutoFeatureEngineer(
    engines=['tabular', 'timeseries', 'llm'],
    llm_config={
        'domain': 'manufacturing',
        'max_suggestions': 15
    }
)

X_fe = engineer.fit_transform(
    X, y,
    column_descriptions={
        'temperature': 'Operating temperature (°C)',
        'vibration': 'Vibration level (mm/s)',
        'pressure': 'Operating pressure (PSI)',
        'runtime_hours': 'Total runtime hours',
        'maintenance_days_ago': 'Days since last maintenance',
        'power_consumption': 'Power consumption (kW)',
        'error_count': 'Error events in last 24 hours',
    },
    task_description="Predict equipment failure within 7 days"
)
```

### Expected LLM-Generated Features

- `operating_stress`: Combined temp/pressure/vibration
- `maintenance_overdue`: Days past maintenance schedule
- `efficiency_degradation`: Power vs expected consumption
- `error_rate`: Errors per runtime hour
- `wear_indicator`: Based on runtime and maintenance

---

## Custom Domain

For domains not in the preset list:

```python
engineer = AutoFeatureEngineer(
    engines=['llm'],
    llm_config={
        'model': 'gpt-5',
        'max_suggestions': 20
    }
)

X_fe = engineer.fit_transform(
    X, y,
    column_descriptions={...},
    task_description="""
    [Detailed task description]
    
    Domain: [Your specific domain]
    
    Business Context:
    - [Key business objectives]
    - [Important domain knowledge]
    - [Relevant industry standards]
    
    Prediction Goal:
    - [What exactly to predict]
    - [Time horizon]
    - [Success metrics]
    """
)
```

## Best Practices for Domain Features

### 1. Be Specific in Descriptions

```python
# ❌ Vague
'revenue': 'Revenue'

# ✅ Specific
'revenue': 'Monthly recurring revenue in USD, includes all subscription tiers'
```

### 2. Include Units

```python
column_descriptions = {
    'temperature': 'Operating temperature in Celsius (normal range: 20-80)',
    'pressure': 'System pressure in PSI (max rated: 150)',
}
```

### 3. Note Constraints

```python
task_description = """
Predict customer churn.

Constraints:
- Features must be explainable to business stakeholders
- Avoid using sensitive demographic data
- Focus on behavioral indicators
"""
```

### 4. Iterate and Refine

```python
# Generate initial features
X_fe = engineer.fit_transform(X, y, ...)

# Request more specific features
additional = engineer.generate_custom_features(
    prompt="Generate features focusing on customer engagement patterns"
)
```

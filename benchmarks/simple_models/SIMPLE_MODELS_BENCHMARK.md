# Simple Models Benchmark Report

> **Scope: curated subset (26 of 63 datasets).** This snapshot covers a
> publication-ready selection — 15 of the 31 INRIA real-world datasets and
> 11 of the 32 synthetic datasets — focused on the most informative
> classification-leaning benchmarks. The full registry contains 63 datasets
> (31 real-world + 32 synthetic). To regenerate the full report, run:
>
> ```bash
> python -m benchmarks.simple_models.run_simple_models_benchmark --all --n-folds 5
> ```
>
> Or to reproduce just this curated subset (15 real-world + 11 synthetic
> datasets, in the order they appear below):
>
> ```bash
> python -m benchmarks.simple_models.run_simple_models_benchmark --n-folds 5 \
>   --datasets eye_movements,higgs,california,jannis,road_safety,covertype,bioresponse,bank_marketing,diabetes,miniboone,magic_telescope,albert,credit,electricity,covertype_cat,xor_classification,polynomial_classification,complex_classification,interaction_classification,credit_risk,customer_churn,customer_support,titanic,credit_card_fraud,employee_attrition,medical_diagnosis
> ```

**Generated:** 2026-04-16 19:14:39
**Models:** RandomForest, LogisticRegression/Ridge
**Cross-Validation:** 5-fold CV × 1 seed(s)
**LLM Enabled:** False
**Datasets:** 26 (15 real-world, 11 synthetic)

## Summary — Real-World Datasets (Primary)

| Metric | Value |
|--------|-------|
| Total Datasets | 15 |
| Win / Tie / Loss | 1 / 13 / 1 |
| Significant Wins (p<0.05) | 0 |
| Mean Improvement | +0.18% |
| Median Improvement | +0.02% |
| Max Regression | -1.14% |

## Summary — Synthetic Datasets (Supplementary)

| Metric | Value |
|--------|-------|
| Total Datasets | 11 |
| Win / Tie / Loss | 5 / 5 / 1 |
| Mean Improvement | +4.30% |

## Summary — All Datasets

| Metric | Value |
|--------|-------|
| Total Datasets | 26 |
| Win / Tie / Loss | 6 / 18 / 2 |
| Significant Wins (p<0.05) | 0 |
| Mean Improvement | +1.93% |
| Median Improvement | +0.12% |

## Real-World Classification

| Dataset | Baseline Score | FeatCopilot Score | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| eye_movements | 0.6442±0.0136 | 0.6676±0.0168 | +3.63% | 0.062 |  | 23→30 |
| higgs | 0.7164±0.0042 | 0.7196±0.0040 | +0.45% | 0.062 |  | 24→25 |
| california | 0.8965±0.0042 | 0.8991±0.0019 | +0.29% | 0.125 |  | 8→8 |
| jannis | 0.7843±0.0022 | 0.7859±0.0029 | +0.21% | 0.188 |  | 54→61 |
| road_safety | 0.7759±0.0043 | 0.7773±0.0031 | +0.18% | 0.500 |  | 32→36 |
| covertype | 0.8596±0.0044 | 0.8605±0.0046 | +0.11% | 0.438 |  | 10→10 |
| bioresponse | 0.7883±0.0105 | 0.7889±0.0108 | +0.07% | 0.875 |  | 419→419 |
| bank_marketing | 0.8012±0.0090 | 0.8014±0.0086 | +0.02% | 1.000 |  | 7→7 |
| diabetes | 0.6016±0.0027 | 0.6016±0.0028 | -0.01% | 1.000 |  | 7→7 |
| miniboone | 0.9309±0.0017 | 0.9301±0.0010 | -0.08% | 0.312 |  | 50→50 |
| magic_telescope | 0.8597±0.0054 | 0.8585±0.0038 | -0.15% | 0.500 |  | 10→10 |
| albert | 0.6541±0.0045 | 0.6527±0.0023 | -0.22% | 0.438 |  | 31→31 |
| credit | 0.7730±0.0055 | 0.7706±0.0073 | -0.31% | 0.188 |  | 10→10 |
| electricity | 0.8977±0.0018 | 0.8948±0.0022 | -0.32% | 0.062 |  | 8→10 |
| covertype_cat | 0.8734±0.0030 | 0.8634±0.0032 | -1.14% 🔴 | 0.062 |  | 54→58 |

## Synthetic Classification (Supplementary)

| Dataset | Baseline Score | FeatCopilot Score | Δ% | p-value | Sig | Features |
|---------|----------------|----------------|-----|---------|-----|----------|
| xor_classification | 0.6960±0.0180 | 0.8024±0.0054 | +15.29% | 0.062 |  | 20→24 |
| polynomial_classification | 0.7790±0.0142 | 0.8790±0.0120 | +12.84% | 0.062 |  | 15→21 |
| complex_classification | 0.7200±0.0123 | 0.7910±0.0174 | +9.86% | 0.062 |  | 15→19 |
| interaction_classification | 0.7570±0.0110 | 0.8240±0.0232 | +8.85% | 0.062 |  | 12→16 |
| credit_risk | 0.8530±0.0179 | 0.8575±0.0203 | +0.53% | 0.500 |  | 10→13 |
| customer_churn | 0.7510±0.0060 | 0.7530±0.0137 | +0.27% | 0.812 |  | 10→11 |
| customer_support | 0.8935±0.0162 | 0.8955±0.0086 | +0.22% | 1.000 |  | 10→13 |
| titanic | 0.8193±0.0116 | 0.8204±0.0119 | +0.14% | 1.000 |  | 7→7 |
| credit_card_fraud | 0.9842±0.0004 | 0.9842±0.0004 | +0.00% | 1.000 |  | 30→30 |
| employee_attrition | 0.9252±0.0030 | 0.9252±0.0030 | +0.00% | 1.000 |  | 11→11 |
| medical_diagnosis | 0.8200±0.0107 | 0.8147±0.0129 | -0.65% 🔴 | 0.375 |  | 12→15 |

# AutoML Benchmark Report

**Generated:** 2026-02-04 07:22:54
**Framework:** FLAML
**LLM Enabled:** False
**Datasets:** 41

## Summary

| Metric | Value |
|--------|-------|
| Total Datasets | 41 |
| Classification | 21 |
| Regression | 20 |
| Improved (Tabular) | 19 |
| Avg Improvement | -0.04% |

## Classification Results

| Dataset | Baseline | Tabular | Improvement | Features |
|---------|----------|---------|-------------|----------|
| titanic | 0.9218 | 0.9218 | +0.00% | 7→27 |
| credit_card_fraud | 0.9840 | 0.9840 | +0.00% | 30→100 |
| employee_attrition | 0.9558 | 0.9558 | +0.00% | 11→74 |
| credit_risk | 0.6925 | 0.7100 | +2.53% | 10→86 |
| medical_diagnosis | 0.8400 | 0.8467 | +0.79% | 12→72 |
| complex_classification | 0.9100 | 0.9225 | +1.37% | 15→100 |
| customer_churn | 0.7525 | 0.7500 | -0.33% | 10→82 |
| higgs | 0.7205 | 0.7228 | +0.32% | 24→100 |
| covertype | 0.8884 | 0.8763 | -1.36% | 10→54 |
| jannis | 0.7931 | 0.7934 | +0.04% | 54→100 |
| miniboone | 0.9434 | 0.9432 | -0.02% | 50→57 |
| california | 0.9026 | 0.8788 | -2.63% | 8→43 |
| credit | 0.7780 | 0.7831 | +0.65% | 10→67 |
| bank_marketing | 0.8100 | 0.8010 | -1.11% | 7→54 |
| diabetes | 0.6113 | 0.6082 | -0.51% | 7→66 |
| bioresponse | 0.7802 | 0.7875 | +0.93% | 419→419 |
| electricity | 0.9058 | 0.9033 | -0.27% | 8→50 |
| covertype_cat | 0.9213 | 0.9041 | -1.87% | 54→100 |
| eye_movements | 0.6649 | 0.6721 | +1.09% | 23→97 |
| road_safety | 0.7940 | 0.7951 | +0.14% | 32→89 |
| albert | 0.6586 | 0.6619 | +0.50% | 31→100 |

## Regression Results

| Dataset | Baseline R² | Tabular R² | Improvement | Features |
|---------|-------------|------------|-------------|----------|
| house_prices | 0.9242 | 0.9203 | -0.42% | 14→46 |
| bike_sharing | 0.8477 | 0.8390 | -1.03% | 10→48 |
| complex_regression | 0.9795 | 0.9731 | -0.66% | 15→79 |
| insurance_claims | 0.9909 | 0.9848 | -0.61% | 10→36 |
| spotify_tracks | 0.4688 | 0.4487 | -4.30% | 14→85 |
| diamonds | 0.9481 | 0.9487 | +0.06% | 6→19 |
| house_sales | 0.8923 | 0.8916 | -0.07% | 15→48 |
| houses | 0.8509 | 0.8555 | +0.54% | 8→40 |
| wine_quality | 0.5100 | 0.5014 | -1.69% | 11→45 |
| abalone | 0.5384 | 0.5844 | +8.55% | 7→30 |
| superconduct | 0.9290 | 0.9316 | +0.28% | 79→100 |
| cpu_act | 0.9829 | 0.9831 | +0.02% | 21→100 |
| elevators | 0.9010 | 0.8903 | -1.18% | 16→68 |
| miami_housing | 0.9303 | 0.9310 | +0.08% | 13→70 |
| bike_sharing_inria | 0.7026 | 0.7045 | +0.26% | 6→37 |
| delays_zurich | 0.0810 | 0.0828 | +2.24% | 11→57 |
| allstate_claims | 0.5456 | 0.5456 | -0.01% | 124→124 |
| mercedes_benz | 0.5763 | 0.5866 | +1.79% | 359→359 |
| nyc_taxi | 0.6696 | 0.6334 | -5.40% | 16→44 |
| brazilian_houses | 0.9986 | 0.9942 | -0.44% | 11→66 |

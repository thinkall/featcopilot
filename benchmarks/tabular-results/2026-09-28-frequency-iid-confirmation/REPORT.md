# Tabular development benchmark

**Benchmark measurement, not a SOTA claim.** Dataset identity, admission roles and split policies
are recorded in the manifest. Legacy manifests are not automatically rights-reviewed or confirmation data.

Primary loss: classification log loss; regression RMSE (lower is better).
Learners are tuned separately per method on inner validation, never selected using outer-test scores.
Synthetic diagnostics are reported separately. All requested failures count toward coverage.

OpenFE-stateless is an explicitly restricted variant, not full OpenFE. AutoFeat uses numeric generation.
OpenFE-inductive searches all supported operator families but freezes training statistics at inference;
its native variant retains all selected additions, while the width-matched variant retains the top-ranked budget.
AutoFeat-default uses native generation/selection defaults and output width with train-fitted categorical preprocessing.
All methods retain original categorical features for identical downstream one-hot preprocessing.
FeatCopilot-task changes only explicit task routing; FeatCopilot-predictive uses learner-aligned holdout selection.
FeatCopilot-paired is an opt-in development variant using paired-gain uncertainty instead of absolute-loss uncertainty.
Extended variants add balanced numeric pairs and frozen training-observed categorical pair indicators; defaults remain standard.
Importance variants fit a training-only joint learner after exact candidate deduplication; holdout subset validation remains independent.
Frequency-paired adds bounded frozen unary-frequency candidates to importance-paired; mappings never learn from validation or test rows.
Unselected and selection-only variants isolate the legacy stages; unselected output retains the generator's full planned width.
The predictive method fits a separate representation per learner with its predeclared middle search candidate.
Each protocol-v3 worker has one learner and an independent budget. Native CatBoost uses categorical data;
its internal predictive-selection estimator uses numeric preprocessing, a disclosed representation difference.

## Configuration

```json
{
  "datasets": null,
  "folds": 3,
  "seeds": [
    2027
  ],
  "max_rows": 0,
  "max_added_features": 50,
  "max_candidates": 1000,
  "threads": 8,
  "memory_limit_mb": 4096,
  "case_timeout": 900.0,
  "synthetic": false,
  "stability_repeats": 3,
  "screening": "auto",
  "allow_confirmation": true,
  "detailed_diagnostics": false,
  "live_llm": false,
  "llm_ledger": null,
  "allow_soft_llm_tokens": false,
  "methods": [
    "baseline",
    "featcopilot-frequency-paired",
    "openfe-inductive-native",
    "autofeat-native"
  ],
  "learners": [
    "linear",
    "lightgbm",
    "catboost",
    "xgboost"
  ],
  "suite": [
    "C:\\code\\featcopilot\\benchmarks\\suites\\frequency_fresh_confirmation_iid_20260928.json"
  ]
}
```

## Paired dataset-level results

Loss reduction is 1 - mean(method loss) / mean(baseline loss), first averaged within each dataset.
W/T/L uses a predeclared 1% relative-loss tolerance, not a significance test.
95% intervals bootstrap origin families (not dependent folds), are descriptive and not multiplicity-adjusted.
Rows with different complete-pair coverage must not be ranked against one another.

| Source / role | Task | Learner | Method | Paired datasets / families | Cells OK/all | W/T/L | Mean loss reduction | 95% CI |
|---|---|---|---|---|---|---|---|---|
| real_world / confirmation | classification | catboost | autofeat-native | 4/6 / 4/6 | 12/18 | 0/0/4 | -2.84% | [-4.01%, -1.67%] |
| real_world / confirmation | classification | catboost | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | catboost | featcopilot-frequency-paired | 6/6 / 6/6 | 18/18 | 1/2/3 | +5.25% | [-3.59%, +21.05%] |
| real_world / confirmation | classification | catboost | openfe-inductive-native | 5/6 / 5/6 | 15/18 | 1/1/3 | -0.52% | [-3.04%, +3.05%] |
| real_world / confirmation | classification | lightgbm | autofeat-native | 4/6 / 4/6 | 12/18 | 1/0/3 | -1.51% | [-3.19%, +0.58%] |
| real_world / confirmation | classification | lightgbm | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | lightgbm | featcopilot-frequency-paired | 6/6 / 6/6 | 18/18 | 1/3/2 | +0.18% | [-8.16%, +9.90%] |
| real_world / confirmation | classification | lightgbm | openfe-inductive-native | 6/6 / 6/6 | 18/18 | 3/1/2 | +9.02% | [-1.24%, +27.51%] |
| real_world / confirmation | classification | linear | autofeat-native | 4/6 / 4/6 | 12/18 | 2/1/1 | -0.47% | [-5.37%, +2.92%] |
| real_world / confirmation | classification | linear | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | linear | featcopilot-frequency-paired | 6/6 / 6/6 | 18/18 | 3/2/1 | +15.41% | [+0.53%, +40.65%] |
| real_world / confirmation | classification | linear | openfe-inductive-native | 6/6 / 6/6 | 18/18 | 2/2/2 | +7.92% | [-13.13%, +38.63%] |
| real_world / confirmation | classification | xgboost | autofeat-native | 4/6 / 4/6 | 12/18 | 2/1/1 | +0.97% | [-3.30%, +5.44%] |
| real_world / confirmation | classification | xgboost | baseline | 6/6 / 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | classification | xgboost | featcopilot-frequency-paired | 6/6 / 6/6 | 18/18 | 1/3/2 | -6.08% | [-16.61%, +0.45%] |
| real_world / confirmation | classification | xgboost | openfe-inductive-native | 6/6 / 6/6 | 18/18 | 1/2/3 | -0.83% | [-3.63%, +2.10%] |
| real_world / confirmation | regression | catboost | autofeat-native | 3/4 / 3/4 | 9/12 | 1/2/0 | +0.24% | [-0.28%, +1.01%] |
| real_world / confirmation | regression | catboost | baseline | 4/4 / 4/4 | 12/12 | 0/4/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | catboost | featcopilot-frequency-paired | 4/4 / 4/4 | 12/12 | 0/2/2 | -4.32% | [-9.83%, +0.10%] |
| real_world / confirmation | regression | catboost | openfe-inductive-native | 3/4 / 3/4 | 9/12 | 0/2/1 | -1.00% | [-4.19%, +0.92%] |
| real_world / confirmation | regression | lightgbm | autofeat-native | 3/4 / 3/4 | 9/12 | 0/2/1 | -1.76% | [-5.60%, +0.83%] |
| real_world / confirmation | regression | lightgbm | baseline | 4/4 / 4/4 | 12/12 | 0/4/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | lightgbm | featcopilot-frequency-paired | 4/4 / 4/4 | 12/12 | 1/2/1 | +0.00% | [-1.36%, +1.95%] |
| real_world / confirmation | regression | lightgbm | openfe-inductive-native | 4/4 / 4/4 | 12/12 | 2/1/1 | +0.50% | [-0.55%, +1.55%] |
| real_world / confirmation | regression | linear | autofeat-native | 3/4 / 3/4 | 9/12 | 0/0/3 | -73.46% | [-189.25%, -6.01%] |
| real_world / confirmation | regression | linear | baseline | 4/4 / 4/4 | 12/12 | 0/4/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | linear | featcopilot-frequency-paired | 4/4 / 4/4 | 12/12 | 0/2/2 | -26.68% | [-53.43%, +0.07%] |
| real_world / confirmation | regression | linear | openfe-inductive-native | 4/4 / 4/4 | 12/12 | 0/2/2 | -20.44% | [-59.21%, -0.03%] |
| real_world / confirmation | regression | xgboost | autofeat-native | 3/4 / 3/4 | 9/12 | 1/0/2 | -6.64% | [-18.89%, +1.91%] |
| real_world / confirmation | regression | xgboost | baseline | 4/4 / 4/4 | 12/12 | 0/4/0 | +0.00% | [+0.00%, +0.00%] |
| real_world / confirmation | regression | xgboost | featcopilot-frequency-paired | 4/4 / 4/4 | 12/12 | 0/2/2 | -10.20% | [-25.84%, -0.53%] |
| real_world / confirmation | regression | xgboost | openfe-inductive-native | 3/4 / 3/4 | 9/12 | 0/2/1 | -0.71% | [-2.24%, +0.12%] |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| australian-credit-approval | catboost | autofeat-native | 0.345826 | 144.37 | 214.68 | 69.3 |
| australian-credit-approval | catboost | baseline | 0.338445 | 0.03 | 71.04 | 15.0 |
| australian-credit-approval | catboost | featcopilot-frequency-paired | 0.339695 | 47.92 | 114.90 | 15.3 |
| australian-credit-approval | catboost | openfe-inductive-native | 0.351686 | 153.45 | 225.69 | 149.3 |
| australian-credit-approval | lightgbm | autofeat-native | 0.371560 | 144.16 | 145.43 | 67.7 |
| australian-credit-approval | lightgbm | baseline | 0.378230 | 0.03 | 1.43 | 15.0 |
| australian-credit-approval | lightgbm | featcopilot-frequency-paired | 0.395325 | 9.26 | 9.89 | 18.3 |
| australian-credit-approval | lightgbm | openfe-inductive-native | 0.365781 | 149.63 | 150.17 | 149.3 |
| australian-credit-approval | linear | autofeat-native | 0.340080 | 143.35 | 143.55 | 66.7 |
| australian-credit-approval | linear | baseline | 0.345194 | 0.03 | 0.21 | 15.0 |
| australian-credit-approval | linear | featcopilot-frequency-paired | 0.345194 | 5.29 | 5.63 | 15.0 |
| australian-credit-approval | linear | openfe-inductive-native | 0.398462 | 152.61 | 152.84 | 149.3 |
| australian-credit-approval | xgboost | autofeat-native | 0.368390 | 142.61 | 143.43 | 67.0 |
| australian-credit-approval | xgboost | baseline | 0.350881 | 0.03 | 0.67 | 15.0 |
| australian-credit-approval | xgboost | featcopilot-frequency-paired | 0.350881 | 13.39 | 14.14 | 15.0 |
| australian-credit-approval | xgboost | openfe-inductive-native | 0.373981 | 149.20 | 150.19 | 149.3 |
| communities-crime | catboost | baseline | 0.135750 | 0.31 | 14.00 | 122.0 |
| communities-crime | catboost | featcopilot-frequency-paired | 0.136468 | 129.84 | 149.25 | 128.7 |
| communities-crime | lightgbm | baseline | 0.138372 | 0.31 | 2.31 | 122.0 |
| communities-crime | lightgbm | featcopilot-frequency-paired | 0.138771 | 27.77 | 28.83 | 139.0 |
| communities-crime | lightgbm | openfe-inductive-native | 0.136260 | 406.24 | 435.36 | 5037.0 |
| communities-crime | linear | baseline | 0.139934 | 0.30 | 0.52 | 122.0 |
| communities-crime | linear | featcopilot-frequency-paired | 0.139733 | 9.88 | 10.03 | 124.0 |
| communities-crime | linear | openfe-inductive-native | 0.250184 | 405.54 | 409.79 | 5037.0 |
| communities-crime | xgboost | baseline | 0.139094 | 0.31 | 3.02 | 122.0 |
| communities-crime | xgboost | featcopilot-frequency-paired | 0.139193 | 69.40 | 71.91 | 122.3 |
| computer-hardware-performance | catboost | autofeat-native | 52.468431 | 62.24 | 65.14 | 13.3 |
| computer-hardware-performance | catboost | baseline | 52.320785 | 0.02 | 2.80 | 6.0 |
| computer-hardware-performance | catboost | featcopilot-frequency-paired | 54.683348 | 16.84 | 18.20 | 41.0 |
| computer-hardware-performance | catboost | openfe-inductive-native | 54.511763 | 146.33 | 148.91 | 7.0 |
| computer-hardware-performance | lightgbm | autofeat-native | 98.044786 | 60.56 | 61.42 | 13.3 |
| computer-hardware-performance | lightgbm | baseline | 92.848972 | 0.02 | 1.10 | 6.0 |
| computer-hardware-performance | lightgbm | featcopilot-frequency-paired | 90.159254 | 4.72 | 4.90 | 16.3 |
| computer-hardware-performance | lightgbm | openfe-inductive-native | 91.382165 | 146.42 | 146.59 | 7.0 |
| computer-hardware-performance | linear | autofeat-native | 217.342622 | 60.99 | 61.03 | 12.7 |
| computer-hardware-performance | linear | baseline | 75.139526 | 0.02 | 0.06 | 6.0 |
| computer-hardware-performance | linear | featcopilot-frequency-paired | 121.163480 | 2.82 | 2.87 | 17.7 |
| computer-hardware-performance | linear | openfe-inductive-native | 74.830311 | 146.34 | 146.38 | 7.0 |
| computer-hardware-performance | xgboost | autofeat-native | 70.587056 | 61.91 | 62.63 | 13.3 |
| computer-hardware-performance | xgboost | baseline | 59.372077 | 0.02 | 0.61 | 6.0 |
| computer-hardware-performance | xgboost | featcopilot-frequency-paired | 62.707627 | 8.60 | 9.04 | 6.3 |
| computer-hardware-performance | xgboost | openfe-inductive-native | 59.301900 | 145.29 | 145.89 | 7.0 |
| contraceptive-method-choice | catboost | autofeat-native | 0.914100 | 174.62 | 245.45 | 39.7 |
| contraceptive-method-choice | catboost | baseline | 0.903667 | 0.02 | 63.29 | 9.0 |
| contraceptive-method-choice | catboost | featcopilot-frequency-paired | 0.904504 | 40.88 | 103.71 | 9.3 |
| contraceptive-method-choice | catboost | openfe-inductive-native | 0.912413 | 156.00 | 241.88 | 84.0 |
| contraceptive-method-choice | lightgbm | autofeat-native | 0.957957 | 172.62 | 174.77 | 44.0 |
| contraceptive-method-choice | lightgbm | baseline | 0.944536 | 0.02 | 2.29 | 9.0 |
| contraceptive-method-choice | lightgbm | featcopilot-frequency-paired | 0.944536 | 21.73 | 23.15 | 9.0 |
| contraceptive-method-choice | lightgbm | openfe-inductive-native | 0.970049 | 155.70 | 157.49 | 84.0 |
| contraceptive-method-choice | linear | autofeat-native | 0.929869 | 171.43 | 171.66 | 40.3 |
| contraceptive-method-choice | linear | baseline | 0.965596 | 0.02 | 0.24 | 9.0 |
| contraceptive-method-choice | linear | featcopilot-frequency-paired | 0.911774 | 7.50 | 7.78 | 15.7 |
| contraceptive-method-choice | linear | openfe-inductive-native | 0.911502 | 158.09 | 158.63 | 84.0 |
| contraceptive-method-choice | xgboost | autofeat-native | 0.925641 | 173.01 | 174.56 | 42.3 |
| contraceptive-method-choice | xgboost | baseline | 0.923452 | 0.02 | 1.52 | 9.0 |
| contraceptive-method-choice | xgboost | featcopilot-frequency-paired | 0.922489 | 21.84 | 23.29 | 17.3 |
| contraceptive-method-choice | xgboost | openfe-inductive-native | 0.945654 | 153.53 | 156.01 | 84.0 |
| forest-fires-montesinho | catboost | autofeat-native | 61.334602 | 71.12 | 136.88 | 31.7 |
| forest-fires-montesinho | catboost | baseline | 61.326403 | 0.03 | 63.06 | 12.0 |
| forest-fires-montesinho | catboost | featcopilot-frequency-paired | 69.259485 | 52.25 | 115.62 | 45.7 |
| forest-fires-montesinho | catboost | openfe-inductive-native | 61.160111 | 146.16 | 210.90 | 13.0 |
| forest-fires-montesinho | lightgbm | autofeat-native | 64.002866 | 69.59 | 70.53 | 31.3 |
| forest-fires-montesinho | lightgbm | baseline | 63.680129 | 0.03 | 1.34 | 12.0 |
| forest-fires-montesinho | lightgbm | featcopilot-frequency-paired | 64.770969 | 7.77 | 8.17 | 13.7 |
| forest-fires-montesinho | lightgbm | openfe-inductive-native | 63.680129 | 147.25 | 147.55 | 13.0 |
| forest-fires-montesinho | linear | autofeat-native | 64.350123 | 66.24 | 66.31 | 32.0 |
| forest-fires-montesinho | linear | baseline | 60.703368 | 0.03 | 0.10 | 12.0 |
| forest-fires-montesinho | linear | featcopilot-frequency-paired | 60.703368 | 3.36 | 3.50 | 12.0 |
| forest-fires-montesinho | linear | openfe-inductive-native | 60.990763 | 146.98 | 147.04 | 13.0 |
| forest-fires-montesinho | xgboost | autofeat-native | 67.348634 | 69.22 | 70.05 | 31.3 |
| forest-fires-montesinho | xgboost | baseline | 65.412861 | 0.03 | 0.75 | 12.0 |
| forest-fires-montesinho | xgboost | featcopilot-frequency-paired | 87.738660 | 20.67 | 21.44 | 15.3 |
| forest-fires-montesinho | xgboost | openfe-inductive-native | 65.412861 | 147.88 | 148.65 | 13.0 |
| indian-liver-patients | catboost | autofeat-native | 0.550767 | 110.94 | 117.15 | 18.3 |
| indian-liver-patients | catboost | baseline | 0.530166 | 0.03 | 3.92 | 10.0 |
| indian-liver-patients | catboost | featcopilot-frequency-paired | 0.536949 | 50.16 | 52.93 | 26.7 |
| indian-liver-patients | catboost | openfe-inductive-native | 0.535755 | 147.02 | 150.73 | 11.0 |
| indian-liver-patients | lightgbm | autofeat-native | 0.614608 | 112.19 | 113.34 | 17.7 |
| indian-liver-patients | lightgbm | baseline | 0.594245 | 0.03 | 1.33 | 10.0 |
| indian-liver-patients | lightgbm | featcopilot-frequency-paired | 0.594245 | 8.03 | 8.44 | 10.0 |
| indian-liver-patients | lightgbm | openfe-inductive-native | 0.594245 | 147.82 | 148.20 | 11.0 |
| indian-liver-patients | linear | autofeat-native | 0.558344 | 109.57 | 109.71 | 21.0 |
| indian-liver-patients | linear | baseline | 0.518663 | 0.03 | 0.16 | 10.0 |
| indian-liver-patients | linear | featcopilot-frequency-paired | 0.518663 | 3.93 | 4.11 | 10.0 |
| indian-liver-patients | linear | openfe-inductive-native | 0.518740 | 145.27 | 145.40 | 11.0 |
| indian-liver-patients | xgboost | autofeat-native | 0.550811 | 116.59 | 117.30 | 19.7 |
| indian-liver-patients | xgboost | baseline | 0.560677 | 0.03 | 0.60 | 10.0 |
| indian-liver-patients | xgboost | featcopilot-frequency-paired | 0.553242 | 12.97 | 13.48 | 10.7 |
| indian-liver-patients | xgboost | openfe-inductive-native | 0.560677 | 145.53 | 146.01 | 11.0 |
| infrared-oral-temperature | catboost | autofeat-native | 0.253509 | 100.75 | 178.33 | 60.7 |
| infrared-oral-temperature | catboost | baseline | 0.256100 | 0.08 | 62.42 | 33.0 |
| infrared-oral-temperature | catboost | featcopilot-frequency-paired | 0.254257 | 178.00 | 244.18 | 45.0 |
| infrared-oral-temperature | catboost | openfe-inductive-native | 0.253739 | 166.24 | 534.31 | 1556.3 |
| infrared-oral-temperature | lightgbm | autofeat-native | 0.256004 | 99.84 | 101.14 | 61.0 |
| infrared-oral-temperature | lightgbm | baseline | 0.258151 | 0.08 | 1.63 | 33.0 |
| infrared-oral-temperature | lightgbm | featcopilot-frequency-paired | 0.260422 | 21.89 | 22.69 | 36.7 |
| infrared-oral-temperature | lightgbm | openfe-inductive-native | 0.261007 | 165.22 | 174.02 | 1556.3 |
| infrared-oral-temperature | linear | autofeat-native | 0.357736 | 99.09 | 99.18 | 59.7 |
| infrared-oral-temperature | linear | baseline | 0.285946 | 0.08 | 0.18 | 33.0 |
| infrared-oral-temperature | linear | featcopilot-frequency-paired | 0.416359 | 7.54 | 7.86 | 53.0 |
| infrared-oral-temperature | linear | openfe-inductive-native | 0.294325 | 167.96 | 168.68 | 1556.3 |
| infrared-oral-temperature | xgboost | autofeat-native | 0.248751 | 99.68 | 101.74 | 59.7 |
| infrared-oral-temperature | xgboost | baseline | 0.253606 | 0.08 | 1.73 | 33.0 |
| infrared-oral-temperature | xgboost | featcopilot-frequency-paired | 0.256095 | 81.96 | 83.88 | 36.7 |
| infrared-oral-temperature | xgboost | openfe-inductive-native | 0.259298 | 166.25 | 208.88 | 1556.3 |
| ionosphere-radar | catboost | baseline | 0.217674 | 0.08 | 7.61 | 34.0 |
| ionosphere-radar | catboost | featcopilot-frequency-paired | 0.227706 | 108.20 | 117.05 | 51.0 |
| ionosphere-radar | catboost | openfe-inductive-native | 0.203560 | 151.41 | 157.86 | 35.0 |
| ionosphere-radar | lightgbm | baseline | 0.220480 | 0.08 | 1.32 | 34.0 |
| ionosphere-radar | lightgbm | featcopilot-frequency-paired | 0.218458 | 8.62 | 8.97 | 34.0 |
| ionosphere-radar | lightgbm | openfe-inductive-native | 0.215873 | 150.50 | 150.79 | 35.0 |
| ionosphere-radar | linear | baseline | 0.325931 | 0.08 | 0.22 | 34.0 |
| ionosphere-radar | linear | featcopilot-frequency-paired | 0.285583 | 5.31 | 5.46 | 34.7 |
| ionosphere-radar | linear | openfe-inductive-native | 0.400686 | 152.00 | 152.11 | 35.0 |
| ionosphere-radar | xgboost | baseline | 0.228384 | 0.08 | 0.77 | 34.0 |
| ionosphere-radar | xgboost | featcopilot-frequency-paired | 0.228576 | 13.61 | 14.13 | 42.3 |
| ionosphere-radar | xgboost | openfe-inductive-native | 0.230675 | 151.19 | 151.70 | 35.0 |
| vertebral-column | catboost | autofeat-native | 0.372921 | 73.24 | 82.77 | 9.7 |
| vertebral-column | catboost | baseline | 0.358132 | 0.02 | 5.89 | 6.0 |
| vertebral-column | catboost | featcopilot-frequency-paired | 0.376742 | 65.84 | 72.43 | 14.3 |
| vertebral-column | catboost | openfe-inductive-native | 0.369466 | 145.62 | 151.34 | 7.0 |
| vertebral-column | lightgbm | autofeat-native | 0.488309 | 72.11 | 73.26 | 9.0 |
| vertebral-column | lightgbm | baseline | 0.474284 | 0.02 | 1.49 | 6.0 |
| vertebral-column | lightgbm | featcopilot-frequency-paired | 0.551664 | 8.28 | 8.84 | 26.0 |
| vertebral-column | lightgbm | openfe-inductive-native | 0.485067 | 145.60 | 146.16 | 7.0 |
| vertebral-column | linear | autofeat-native | 0.340058 | 74.68 | 74.80 | 10.0 |
| vertebral-column | linear | baseline | 0.342026 | 0.02 | 0.12 | 6.0 |
| vertebral-column | linear | featcopilot-frequency-paired | 0.346123 | 3.13 | 3.25 | 6.3 |
| vertebral-column | linear | openfe-inductive-native | 0.342589 | 146.80 | 146.91 | 7.0 |
| vertebral-column | xgboost | autofeat-native | 0.439182 | 72.29 | 73.43 | 9.0 |
| vertebral-column | xgboost | baseline | 0.473936 | 0.02 | 1.13 | 6.0 |
| vertebral-column | xgboost | featcopilot-frequency-paired | 0.506440 | 14.05 | 14.95 | 31.3 |
| vertebral-column | xgboost | openfe-inductive-native | 0.473936 | 146.33 | 147.31 | 7.0 |
| wall-following-robot | catboost | baseline | 0.020097 | 0.07 | 31.91 | 24.0 |
| wall-following-robot | catboost | featcopilot-frequency-paired | 0.011442 | 740.46 | 773.32 | 32.3 |
| wall-following-robot | lightgbm | baseline | 0.011059 | 0.07 | 2.94 | 24.0 |
| wall-following-robot | lightgbm | featcopilot-frequency-paired | 0.008737 | 59.22 | 60.95 | 50.7 |
| wall-following-robot | lightgbm | openfe-inductive-native | 0.005118 | 202.19 | 228.25 | 1817.0 |
| wall-following-robot | linear | baseline | 0.702359 | 0.07 | 0.37 | 24.0 |
| wall-following-robot | linear | featcopilot-frequency-paired | 0.170576 | 24.01 | 24.75 | 74.0 |
| wall-following-robot | linear | openfe-inductive-native | 0.137046 | 204.59 | 212.92 | 1817.0 |
| wall-following-robot | xgboost | baseline | 0.008297 | 0.07 | 2.72 | 24.0 |
| wall-following-robot | xgboost | featcopilot-frequency-paired | 0.010864 | 109.30 | 111.83 | 32.7 |
| wall-following-robot | xgboost | openfe-inductive-native | 0.007882 | 200.56 | 275.97 | 1817.0 |

## Failures / timeouts

- communities-crime / 2027 / fold 0 / autofeat-native / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- communities-crime / 2027 / fold 0 / openfe-inductive-native / catboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- ionosphere-radar / 2027 / fold 0 / autofeat-native / xgboost: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- wall-following-robot / 2027 / fold 0 / autofeat-native / catboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- ionosphere-radar / 2027 / fold 0 / autofeat-native / catboost: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- communities-crime / 2027 / fold 0 / openfe-inductive-native / xgboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- communities-crime / 2027 / fold 0 / autofeat-native / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 0 / autofeat-native / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 0 / autofeat-native / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 0 / openfe-inductive-native / catboost: timeout: Case exceeded 900.0s wall-clock limit
- ionosphere-radar / 2027 / fold 0 / autofeat-native / lightgbm: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- ionosphere-radar / 2027 / fold 0 / autofeat-native / linear: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- communities-crime / 2027 / fold 0 / autofeat-native / catboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 0 / autofeat-native / xgboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- communities-crime / 2027 / fold 0 / autofeat-native / xgboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 1 / autofeat-native / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 1 / openfe-inductive-native / catboost: timeout: Case exceeded 900.0s wall-clock limit
- communities-crime / 2027 / fold 1 / autofeat-native / xgboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- communities-crime / 2027 / fold 1 / autofeat-native / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 1 / autofeat-native / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- communities-crime / 2027 / fold 1 / autofeat-native / catboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- communities-crime / 2027 / fold 1 / openfe-inductive-native / catboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- ionosphere-radar / 2027 / fold 1 / autofeat-native / linear: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- ionosphere-radar / 2027 / fold 1 / autofeat-native / catboost: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- ionosphere-radar / 2027 / fold 1 / autofeat-native / lightgbm: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- communities-crime / 2027 / fold 1 / openfe-inductive-native / xgboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 1 / autofeat-native / catboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- communities-crime / 2027 / fold 1 / autofeat-native / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 1 / autofeat-native / xgboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- ionosphere-radar / 2027 / fold 1 / autofeat-native / xgboost: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- communities-crime / 2027 / fold 2 / autofeat-native / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- communities-crime / 2027 / fold 2 / openfe-inductive-native / xgboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- ionosphere-radar / 2027 / fold 2 / autofeat-native / linear: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- communities-crime / 2027 / fold 2 / autofeat-native / xgboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 2 / autofeat-native / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 2 / autofeat-native / linear: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 2 / openfe-inductive-native / catboost: timeout: Case exceeded 900.0s wall-clock limit
- communities-crime / 2027 / fold 2 / autofeat-native / catboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 2 / autofeat-native / catboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- ionosphere-radar / 2027 / fold 2 / autofeat-native / lightgbm: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- ionosphere-radar / 2027 / fold 2 / autofeat-native / xgboost: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 
- communities-crime / 2027 / fold 2 / openfe-inductive-native / catboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- wall-following-robot / 2027 / fold 2 / autofeat-native / xgboost: memory_limit: Worker tree exceeded the configured RSS memory limit
- communities-crime / 2027 / fold 2 / autofeat-native / lightgbm: memory_limit: Worker tree exceeded the configured RSS memory limit
- ionosphere-radar / 2027 / fold 2 / autofeat-native / catboost: error: ValueError: Input X contains NaN. LogisticRegressionCV does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is 

Raw cell metrics and provenance: local `results.json`. Completed runs also write `archive.json`,
which lists the authoritative lossless gzip parts and their uncompressed checksum.

# Tabular development benchmark

**Development measurement, not a SOTA claim.** This is a fixed subset of the INRIA suite,
using our own repeated CV splits and a documented row cap, not a reproduction of its published leaderboard.

Primary loss: classification log loss; regression RMSE (lower is better).
Learners are tuned separately per method on inner validation, never selected using outer-test scores.
Synthetic diagnostics are reported separately. All requested failures count toward coverage.

OpenFE-stateless is an explicitly restricted variant, not full OpenFE. AutoFeat uses numeric generation.
All methods retain original categorical features for identical downstream one-hot preprocessing.
FeatCopilot-task changes only explicit task routing; FeatCopilot-predictive uses learner-aligned holdout selection.
The predictive method fits a separate representation per learner with its predeclared middle search candidate.

## Configuration

```json
{
  "datasets": null,
  "methods": [
    "baseline",
    "featcopilot",
    "featcopilot-task",
    "featcopilot-predictive",
    "polynomial",
    "openfe-stateless",
    "autofeat"
  ],
  "learners": [
    "linear",
    "lightgbm"
  ],
  "folds": 3,
  "seeds": [
    42
  ],
  "max_rows": 5000,
  "max_added_features": 50,
  "max_candidates": 1000,
  "threads": 2,
  "memory_limit_mb": 4096,
  "case_timeout": 120.0,
  "total_seconds": 14400.0,
  "synthetic": true
}
```

## Paired dataset-level results

Loss reduction is 1 - mean(method loss) / mean(baseline loss), first averaged within each dataset.
W/T/L uses a predeclared 1% relative-loss tolerance, not a significance test.
95% intervals bootstrap datasets (not dependent folds), are descriptive and not multiplicity-adjusted.
Rows with different complete-pair coverage must not be ranked against one another.

| Source | Task | Learner | Method | Paired datasets | Folds OK/all | W/T/L | Mean loss reduction | 95% CI |
|---|---|---|---|---|---|---|---|---|
| real_world | classification | lightgbm | autofeat | 0/6 | 0/18 | 0/0/0 | n/a | n/a |
| real_world | classification | lightgbm | baseline | 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world | classification | lightgbm | featcopilot | 6/6 | 18/18 | 0/6/0 | +0.26% | [+0.02%, +0.54%] |
| real_world | classification | lightgbm | featcopilot-predictive | 6/6 | 18/18 | 1/5/0 | +0.49% | [-0.12%, +1.47%] |
| real_world | classification | lightgbm | featcopilot-task | 6/6 | 18/18 | 0/6/0 | +0.26% | [+0.02%, +0.54%] |
| real_world | classification | lightgbm | openfe-stateless | 6/6 | 18/18 | 1/5/0 | +0.98% | [-0.72%, +4.01%] |
| real_world | classification | lightgbm | polynomial | 6/6 | 18/18 | 0/6/0 | +0.11% | [-0.40%, +0.61%] |
| real_world | classification | linear | autofeat | 0/6 | 0/18 | 0/0/0 | n/a | n/a |
| real_world | classification | linear | baseline | 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world | classification | linear | featcopilot | 6/6 | 18/18 | 1/5/0 | +0.55% | [+0.06%, +1.31%] |
| real_world | classification | linear | featcopilot-predictive | 6/6 | 18/18 | 5/1/0 | +6.76% | [+2.22%, +12.88%] |
| real_world | classification | linear | featcopilot-task | 6/6 | 18/18 | 1/5/0 | +0.55% | [+0.06%, +1.31%] |
| real_world | classification | linear | openfe-stateless | 6/6 | 18/18 | 3/2/1 | +5.21% | [-0.05%, +12.52%] |
| real_world | classification | linear | polynomial | 6/6 | 18/18 | 4/1/1 | +4.84% | [-0.17%, +12.37%] |
| real_world | regression | lightgbm | autofeat | 5/6 | 16/18 | 5/0/0 | +3.80% | [+1.87%, +6.36%] |
| real_world | regression | lightgbm | baseline | 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world | regression | lightgbm | featcopilot | 6/6 | 18/18 | 0/6/0 | +0.08% | [-0.01%, +0.24%] |
| real_world | regression | lightgbm | featcopilot-predictive | 6/6 | 18/18 | 5/1/0 | +2.10% | [+0.85%, +3.45%] |
| real_world | regression | lightgbm | featcopilot-task | 6/6 | 18/18 | 0/6/0 | +0.02% | [-0.01%, +0.06%] |
| real_world | regression | lightgbm | openfe-stateless | 6/6 | 18/18 | 5/1/0 | +3.49% | [+1.40%, +5.77%] |
| real_world | regression | lightgbm | polynomial | 6/6 | 18/18 | 1/2/3 | -1.01% | [-2.30%, +0.61%] |
| real_world | regression | linear | autofeat | 5/6 | 16/18 | 3/0/2 | -15.65% | [-50.60%, +7.39%] |
| real_world | regression | linear | baseline | 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world | regression | linear | featcopilot | 6/6 | 18/18 | 0/6/0 | +0.00% | [-0.00%, +0.01%] |
| real_world | regression | linear | featcopilot-predictive | 6/6 | 18/18 | 3/1/2 | +8.67% | [-14.07%, +35.54%] |
| real_world | regression | linear | featcopilot-task | 6/6 | 18/18 | 0/6/0 | +0.02% | [-0.00%, +0.07%] |
| real_world | regression | linear | openfe-stateless | 6/6 | 18/18 | 4/2/0 | +14.18% | [+2.85%, +29.28%] |
| real_world | regression | linear | polynomial | 6/6 | 18/18 | 5/0/1 | +15.30% | [-0.70%, +36.81%] |
| synthetic | classification | lightgbm | autofeat | 0/1 | 0/3 | 0/0/0 | n/a | n/a |
| synthetic | classification | lightgbm | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | classification | lightgbm | featcopilot | 1/1 | 3/3 | 1/0/0 | +6.11% | n/a |
| synthetic | classification | lightgbm | featcopilot-predictive | 1/1 | 3/3 | 1/0/0 | +9.36% | n/a |
| synthetic | classification | lightgbm | featcopilot-task | 1/1 | 3/3 | 1/0/0 | +6.11% | n/a |
| synthetic | classification | lightgbm | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +27.97% | n/a |
| synthetic | classification | lightgbm | polynomial | 1/1 | 3/3 | 1/0/0 | +27.07% | n/a |
| synthetic | classification | linear | autofeat | 0/1 | 0/3 | 0/0/0 | n/a | n/a |
| synthetic | classification | linear | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | classification | linear | featcopilot | 1/1 | 3/3 | 1/0/0 | +28.70% | n/a |
| synthetic | classification | linear | featcopilot-predictive | 1/1 | 3/3 | 1/0/0 | +46.67% | n/a |
| synthetic | classification | linear | featcopilot-task | 1/1 | 3/3 | 1/0/0 | +28.70% | n/a |
| synthetic | classification | linear | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +46.09% | n/a |
| synthetic | classification | linear | polynomial | 1/1 | 3/3 | 1/0/0 | +54.23% | n/a |
| synthetic | regression | lightgbm | autofeat | 1/1 | 3/3 | 1/0/0 | +33.10% | n/a |
| synthetic | regression | lightgbm | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | regression | lightgbm | featcopilot | 1/1 | 3/3 | 1/0/0 | +20.54% | n/a |
| synthetic | regression | lightgbm | featcopilot-predictive | 1/1 | 3/3 | 1/0/0 | +29.21% | n/a |
| synthetic | regression | lightgbm | featcopilot-task | 1/1 | 3/3 | 1/0/0 | +20.54% | n/a |
| synthetic | regression | lightgbm | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +31.05% | n/a |
| synthetic | regression | lightgbm | polynomial | 1/1 | 3/3 | 1/0/0 | +31.26% | n/a |
| synthetic | regression | linear | autofeat | 1/1 | 3/3 | 1/0/0 | +83.27% | n/a |
| synthetic | regression | linear | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | regression | linear | featcopilot | 1/1 | 3/3 | 1/0/0 | +65.15% | n/a |
| synthetic | regression | linear | featcopilot-predictive | 1/1 | 3/3 | 1/0/0 | +58.66% | n/a |
| synthetic | regression | linear | featcopilot-task | 1/1 | 3/3 | 1/0/0 | +65.15% | n/a |
| synthetic | regression | linear | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +60.46% | n/a |
| synthetic | regression | linear | polynomial | 1/1 | 3/3 | 1/0/0 | +82.99% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| abalone | lightgbm | autofeat | 2.132829 | 23.05 | 24.24 | 22.0 |
| abalone | lightgbm | baseline | 2.175105 | 0.03 | 0.90 | 8.0 |
| abalone | lightgbm | featcopilot | 2.175105 | 6.18 | 7.09 | 8.0 |
| abalone | lightgbm | featcopilot-predictive | 2.134371 | 5.00 | 5.81 | 49.7 |
| abalone | lightgbm | featcopilot-task | 2.175105 | 11.86 | 12.67 | 8.0 |
| abalone | lightgbm | openfe-stateless | 2.130982 | 38.96 | 40.77 | 58.0 |
| abalone | lightgbm | polynomial | 2.189645 | 0.04 | 1.73 | 36.0 |
| abalone | linear | autofeat | 2.592417 | 23.05 | 23.12 | 22.0 |
| abalone | linear | baseline | 2.206077 | 0.03 | 0.10 | 8.0 |
| abalone | linear | featcopilot | 2.206077 | 6.18 | 6.24 | 8.0 |
| abalone | linear | featcopilot-predictive | 2.203284 | 2.22 | 2.29 | 31.3 |
| abalone | linear | featcopilot-task | 2.206077 | 11.86 | 11.92 | 8.0 |
| abalone | linear | openfe-stateless | 2.187064 | 38.96 | 39.07 | 58.0 |
| abalone | linear | polynomial | 2.462576 | 0.04 | 0.13 | 36.0 |
| bank | lightgbm | baseline | 0.429811 | 0.02 | 0.95 | 7.0 |
| bank | lightgbm | featcopilot | 0.429811 | 7.57 | 8.61 | 7.0 |
| bank | lightgbm | featcopilot-predictive | 0.430602 | 4.19 | 4.86 | 10.3 |
| bank | lightgbm | featcopilot-task | 0.429811 | 7.57 | 8.64 | 7.0 |
| bank | lightgbm | openfe-stateless | 0.433489 | 37.01 | 38.86 | 57.0 |
| bank | lightgbm | polynomial | 0.433064 | 0.04 | 1.63 | 35.0 |
| bank | linear | baseline | 0.520463 | 0.02 | 0.10 | 7.0 |
| bank | linear | featcopilot | 0.520463 | 7.57 | 7.65 | 7.0 |
| bank | linear | featcopilot-predictive | 0.467805 | 9.04 | 9.31 | 48.7 |
| bank | linear | featcopilot-task | 0.520463 | 7.57 | 7.65 | 7.0 |
| bank | linear | openfe-stateless | 0.472810 | 37.01 | 37.26 | 57.0 |
| bank | linear | polynomial | 0.494596 | 0.04 | 0.21 | 35.0 |
| cpu | lightgbm | baseline | 2.267884 | 0.06 | 1.42 | 21.0 |
| cpu | lightgbm | featcopilot | 2.267907 | 12.92 | 14.28 | 21.0 |
| cpu | lightgbm | featcopilot-predictive | 2.276232 | 13.70 | 15.62 | 21.7 |
| cpu | lightgbm | featcopilot-task | 2.267884 | 30.02 | 31.42 | 21.0 |
| cpu | lightgbm | openfe-stateless | 2.270341 | 66.38 | 69.07 | 71.0 |
| cpu | lightgbm | polynomial | 2.291728 | 0.09 | 3.36 | 71.0 |
| cpu | linear | baseline | 10.095029 | 0.06 | 0.14 | 21.0 |
| cpu | linear | featcopilot | 10.095029 | 12.92 | 12.98 | 21.0 |
| cpu | linear | featcopilot-predictive | 2.873454 | 9.15 | 9.27 | 62.7 |
| cpu | linear | featcopilot-task | 10.095029 | 30.02 | 30.08 | 21.0 |
| cpu | linear | openfe-stateless | 4.880648 | 66.38 | 66.51 | 71.0 |
| cpu | linear | polynomial | 3.417816 | 0.09 | 0.22 | 71.0 |
| credit | lightgbm | baseline | 0.486485 | 0.03 | 1.01 | 10.0 |
| credit | lightgbm | featcopilot | 0.486261 | 7.41 | 8.52 | 10.3 |
| credit | lightgbm | featcopilot-predictive | 0.485992 | 5.30 | 5.96 | 12.3 |
| credit | lightgbm | featcopilot-task | 0.486261 | 7.39 | 8.43 | 10.3 |
| credit | lightgbm | openfe-stateless | 0.486305 | 40.20 | 42.07 | 60.0 |
| credit | lightgbm | polynomial | 0.485342 | 0.05 | 2.02 | 60.0 |
| credit | linear | baseline | 0.544493 | 0.03 | 0.15 | 10.0 |
| credit | linear | featcopilot | 0.531913 | 7.41 | 7.52 | 10.3 |
| credit | linear | featcopilot-predictive | 0.533039 | 10.23 | 10.40 | 30.0 |
| credit | linear | featcopilot-task | 0.531913 | 7.39 | 7.50 | 10.3 |
| credit | linear | openfe-stateless | 0.554226 | 40.20 | 40.43 | 60.0 |
| credit | linear | polynomial | 0.562476 | 0.05 | 0.32 | 60.0 |
| credit_default | lightgbm | baseline | 0.575960 | 0.05 | 1.22 | 20.0 |
| credit_default | lightgbm | featcopilot | 0.576476 | 8.99 | 10.24 | 20.7 |
| credit_default | lightgbm | featcopilot-predictive | 0.574072 | 11.76 | 12.86 | 31.7 |
| credit_default | lightgbm | featcopilot-task | 0.576476 | 9.31 | 10.54 | 20.7 |
| credit_default | lightgbm | openfe-stateless | 0.578877 | 59.94 | 62.16 | 70.0 |
| credit_default | lightgbm | polynomial | 0.575883 | 0.10 | 2.43 | 70.0 |
| credit_default | linear | baseline | 0.616422 | 0.05 | 0.18 | 20.0 |
| credit_default | linear | featcopilot | 0.613089 | 8.99 | 9.10 | 20.7 |
| credit_default | linear | featcopilot-predictive | 0.585431 | 15.64 | 15.89 | 30.3 |
| credit_default | linear | featcopilot-task | 0.613089 | 9.31 | 9.42 | 20.7 |
| credit_default | linear | openfe-stateless | 0.610915 | 59.94 | 60.18 | 70.0 |
| credit_default | linear | polynomial | 0.599781 | 0.10 | 0.36 | 70.0 |
| diamonds | lightgbm | autofeat | 0.106689 | 16.44 | 17.68 | 14.0 |
| diamonds | lightgbm | baseline | 0.108669 | 0.04 | 1.17 | 9.0 |
| diamonds | lightgbm | featcopilot | 0.108673 | 10.95 | 11.97 | 9.0 |
| diamonds | lightgbm | featcopilot-predictive | 0.104953 | 5.30 | 6.32 | 26.3 |
| diamonds | lightgbm | featcopilot-task | 0.108673 | 10.84 | 11.90 | 9.0 |
| diamonds | lightgbm | openfe-stateless | 0.106435 | 39.62 | 41.94 | 59.0 |
| diamonds | lightgbm | polynomial | 0.109723 | 0.06 | 1.84 | 30.0 |
| diamonds | linear | autofeat | 0.130445 | 16.44 | 16.52 | 14.0 |
| diamonds | linear | baseline | 0.142347 | 0.04 | 0.13 | 9.0 |
| diamonds | linear | featcopilot | 0.142347 | 10.95 | 11.03 | 9.0 |
| diamonds | linear | featcopilot-predictive | 0.122099 | 2.14 | 2.29 | 50.7 |
| diamonds | linear | featcopilot-task | 0.142347 | 10.84 | 10.92 | 9.0 |
| diamonds | linear | openfe-stateless | 0.121424 | 39.62 | 39.76 | 59.0 |
| diamonds | linear | polynomial | 0.124001 | 0.06 | 0.16 | 30.0 |
| elevators | lightgbm | autofeat | 0.002342 | 17.93 | 19.16 | 28.0 |
| elevators | lightgbm | baseline | 0.002553 | 0.04 | 1.17 | 16.0 |
| elevators | lightgbm | featcopilot | 0.002554 | 18.17 | 19.29 | 16.0 |
| elevators | lightgbm | featcopilot-predictive | 0.002431 | 8.09 | 8.98 | 36.3 |
| elevators | lightgbm | featcopilot-task | 0.002554 | 17.71 | 18.79 | 16.0 |
| elevators | lightgbm | openfe-stateless | 0.002349 | 52.86 | 54.78 | 66.0 |
| elevators | lightgbm | polynomial | 0.002637 | 0.07 | 2.54 | 66.0 |
| elevators | linear | autofeat | 0.002654 | 17.93 | 18.00 | 28.0 |
| elevators | linear | baseline | 0.002885 | 0.04 | 0.11 | 16.0 |
| elevators | linear | featcopilot | 0.002885 | 18.17 | 18.22 | 16.0 |
| elevators | linear | featcopilot-predictive | 0.002471 | 5.52 | 5.62 | 66.0 |
| elevators | linear | featcopilot-task | 0.002885 | 17.71 | 17.76 | 16.0 |
| elevators | linear | openfe-stateless | 0.002434 | 52.86 | 52.98 | 66.0 |
| elevators | linear | polynomial | 0.002457 | 0.07 | 0.19 | 66.0 |
| eye_movements | lightgbm | baseline | 0.655070 | 0.07 | 1.48 | 23.0 |
| eye_movements | lightgbm | featcopilot | 0.649740 | 9.92 | 11.39 | 23.7 |
| eye_movements | lightgbm | featcopilot-predictive | 0.654011 | 13.40 | 14.69 | 35.0 |
| eye_movements | lightgbm | featcopilot-task | 0.649740 | 9.74 | 11.27 | 23.7 |
| eye_movements | lightgbm | openfe-stateless | 0.656264 | 52.79 | 55.20 | 73.0 |
| eye_movements | lightgbm | polynomial | 0.649673 | 0.12 | 2.73 | 73.0 |
| eye_movements | linear | baseline | 0.677233 | 0.07 | 0.26 | 23.0 |
| eye_movements | linear | featcopilot | 0.677071 | 9.92 | 10.08 | 23.7 |
| eye_movements | linear | featcopilot-predictive | 0.672022 | 16.42 | 16.65 | 41.7 |
| eye_movements | linear | featcopilot-task | 0.677071 | 9.74 | 9.90 | 23.7 |
| eye_movements | linear | openfe-stateless | 0.668506 | 52.79 | 53.07 | 73.0 |
| eye_movements | linear | polynomial | 0.670461 | 0.12 | 0.42 | 73.0 |
| heloc | lightgbm | baseline | 0.561693 | 0.06 | 1.18 | 22.0 |
| heloc | lightgbm | featcopilot | 0.560299 | 9.04 | 10.20 | 24.0 |
| heloc | lightgbm | featcopilot-predictive | 0.563400 | 11.73 | 12.88 | 25.3 |
| heloc | lightgbm | featcopilot-task | 0.560299 | 9.24 | 10.41 | 24.0 |
| heloc | lightgbm | openfe-stateless | 0.567016 | 63.36 | 65.23 | 72.0 |
| heloc | lightgbm | polynomial | 0.564817 | 0.11 | 2.53 | 72.0 |
| heloc | linear | baseline | 0.570431 | 0.06 | 0.21 | 22.0 |
| heloc | linear | featcopilot | 0.567891 | 9.04 | 9.17 | 24.0 |
| heloc | linear | featcopilot-predictive | 0.555904 | 15.96 | 16.22 | 50.3 |
| heloc | linear | featcopilot-task | 0.567891 | 9.24 | 9.38 | 24.0 |
| heloc | linear | openfe-stateless | 0.569361 | 63.36 | 63.65 | 72.0 |
| heloc | linear | polynomial | 0.560205 | 0.11 | 0.43 | 72.0 |
| houses | lightgbm | autofeat | 0.232209 | 28.39 | 30.39 | 35.0 |
| houses | lightgbm | baseline | 0.244681 | 0.02 | 1.11 | 8.0 |
| houses | lightgbm | featcopilot | 0.244681 | 24.34 | 25.53 | 8.0 |
| houses | lightgbm | featcopilot-predictive | 0.242089 | 5.56 | 6.68 | 13.3 |
| houses | lightgbm | featcopilot-task | 0.244681 | 23.41 | 24.59 | 8.0 |
| houses | lightgbm | openfe-stateless | 0.228722 | 39.61 | 42.11 | 58.0 |
| houses | lightgbm | polynomial | 0.250435 | 0.04 | 2.55 | 44.0 |
| houses | linear | autofeat | 0.613724 | 28.39 | 28.46 | 35.0 |
| houses | linear | baseline | 0.338541 | 0.02 | 0.08 | 8.0 |
| houses | linear | featcopilot | 0.338541 | 24.34 | 24.39 | 8.0 |
| houses | linear | featcopilot-predictive | 0.448000 | 2.55 | 2.64 | 58.0 |
| houses | linear | featcopilot-task | 0.338541 | 23.41 | 23.45 | 8.0 |
| houses | linear | openfe-stateless | 0.339693 | 39.61 | 39.72 | 58.0 |
| houses | linear | polynomial | 0.311700 | 0.04 | 0.14 | 44.0 |
| magic | lightgbm | baseline | 0.344359 | 0.03 | 1.14 | 10.0 |
| magic | lightgbm | featcopilot | 0.342473 | 10.30 | 11.56 | 10.0 |
| magic | lightgbm | featcopilot-predictive | 0.334665 | 7.29 | 9.07 | 28.3 |
| magic | lightgbm | featcopilot-task | 0.342473 | 10.07 | 11.27 | 10.0 |
| magic | lightgbm | openfe-stateless | 0.315621 | 43.47 | 45.80 | 60.0 |
| magic | lightgbm | polynomial | 0.341236 | 0.05 | 2.70 | 60.0 |
| magic | linear | baseline | 0.490530 | 0.03 | 0.14 | 10.0 |
| magic | linear | featcopilot | 0.490775 | 10.30 | 10.39 | 10.0 |
| magic | linear | featcopilot-predictive | 0.392392 | 10.72 | 10.96 | 51.7 |
| magic | linear | featcopilot-task | 0.490775 | 10.07 | 10.17 | 10.0 |
| magic | linear | openfe-stateless | 0.385100 | 43.47 | 43.74 | 60.0 |
| magic | linear | polynomial | 0.383191 | 0.05 | 0.36 | 60.0 |
| polynomial_regression | lightgbm | autofeat | 2.574925 | 29.84 | 31.15 | 24.3 |
| polynomial_regression | lightgbm | baseline | 3.848670 | 0.03 | 1.08 | 12.0 |
| polynomial_regression | lightgbm | featcopilot | 3.058279 | 13.72 | 14.89 | 16.0 |
| polynomial_regression | lightgbm | featcopilot-predictive | 2.724659 | 6.82 | 8.59 | 62.0 |
| polynomial_regression | lightgbm | featcopilot-task | 3.058279 | 14.37 | 15.55 | 16.0 |
| polynomial_regression | lightgbm | openfe-stateless | 2.653784 | 43.89 | 45.91 | 62.0 |
| polynomial_regression | lightgbm | polynomial | 2.645763 | 0.05 | 2.43 | 62.0 |
| polynomial_regression | linear | autofeat | 1.539797 | 29.84 | 29.89 | 24.3 |
| polynomial_regression | linear | baseline | 9.204904 | 0.03 | 0.09 | 12.0 |
| polynomial_regression | linear | featcopilot | 3.207920 | 13.72 | 13.77 | 16.0 |
| polynomial_regression | linear | featcopilot-predictive | 3.805596 | 3.37 | 3.44 | 62.0 |
| polynomial_regression | linear | featcopilot-task | 3.207920 | 14.37 | 14.41 | 16.0 |
| polynomial_regression | linear | openfe-stateless | 3.639468 | 43.89 | 43.96 | 62.0 |
| polynomial_regression | linear | polynomial | 1.565831 | 0.05 | 0.12 | 62.0 |
| wine | lightgbm | autofeat | 0.642147 | 36.72 | 38.83 | 44.0 |
| wine | lightgbm | baseline | 0.654417 | 0.03 | 1.09 | 11.0 |
| wine | lightgbm | featcopilot | 0.651329 | 11.11 | 12.07 | 12.3 |
| wine | lightgbm | featcopilot-predictive | 0.642346 | 6.67 | 8.61 | 44.3 |
| wine | lightgbm | featcopilot-task | 0.653585 | 26.63 | 27.77 | 15.0 |
| wine | lightgbm | openfe-stateless | 0.638328 | 42.50 | 45.08 | 61.0 |
| wine | lightgbm | polynomial | 0.639746 | 0.06 | 3.04 | 61.0 |
| wine | linear | autofeat | 0.706542 | 36.72 | 36.79 | 44.0 |
| wine | linear | baseline | 0.737508 | 0.03 | 0.09 | 11.0 |
| wine | linear | featcopilot | 0.737326 | 11.11 | 11.16 | 12.3 |
| wine | linear | featcopilot-predictive | 0.854650 | 3.70 | 3.82 | 52.7 |
| wine | linear | featcopilot-task | 0.736475 | 26.63 | 26.69 | 15.0 |
| wine | linear | openfe-stateless | 0.718771 | 42.50 | 42.62 | 61.0 |
| wine | linear | polynomial | 0.725725 | 0.06 | 0.17 | 61.0 |
| xor_classification | lightgbm | baseline | 0.482452 | 0.05 | 1.54 | 20.0 |
| xor_classification | lightgbm | featcopilot | 0.452956 | 8.12 | 9.50 | 23.3 |
| xor_classification | lightgbm | featcopilot-predictive | 0.437299 | 12.25 | 13.86 | 33.3 |
| xor_classification | lightgbm | featcopilot-task | 0.452956 | 8.20 | 9.59 | 23.3 |
| xor_classification | lightgbm | openfe-stateless | 0.347511 | 54.74 | 56.91 | 70.0 |
| xor_classification | lightgbm | polynomial | 0.351865 | 0.08 | 2.63 | 70.0 |
| xor_classification | linear | baseline | 0.685736 | 0.05 | 0.15 | 20.0 |
| xor_classification | linear | featcopilot | 0.488924 | 8.12 | 8.20 | 23.3 |
| xor_classification | linear | featcopilot-predictive | 0.365723 | 14.95 | 15.07 | 53.3 |
| xor_classification | linear | featcopilot-task | 0.488924 | 8.20 | 8.29 | 23.3 |
| xor_classification | linear | openfe-stateless | 0.369699 | 54.74 | 54.89 | 70.0 |
| xor_classification | linear | polynomial | 0.313876 | 0.08 | 0.20 | 70.0 |

## Failures / timeouts

- magic / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit

Raw fold metrics, hyperparameters, timings, split fingerprints, versions and data provenance are losslessly archived as `results.json.gz.part1`, `results.json.gz.part2`, and `results.json.gz.part3`. Concatenate these byte streams in order before gzip decompression; see [ANALYSIS.md](ANALYSIS.md) for a Python example. The uncompressed `results.json` is also available locally.

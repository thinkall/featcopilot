# Tabular development benchmark

**Development measurement, not a SOTA claim.** This is a fixed subset of the INRIA suite,
using our own repeated CV splits and a documented row cap, not a reproduction of its published leaderboard.

Primary loss: classification log loss; regression RMSE (lower is better).
Learners are tuned separately per method on inner validation, never selected using outer-test scores.
Synthetic diagnostics are reported separately. All requested failures count toward coverage.

OpenFE-stateless is an explicitly restricted variant, not full OpenFE. AutoFeat uses numeric generation.
All methods retain original categorical features for identical downstream one-hot preprocessing.

## Configuration

```json
{
  "datasets": null,
  "methods": [
    "baseline",
    "featcopilot",
    "polynomial",
    "openfe-stateless",
    "autofeat"
  ],
  "learners": [
    "linear",
    "random_forest",
    "lightgbm"
  ],
  "folds": 3,
  "seeds": [
    42
  ],
  "max_rows": 5000,
  "max_added_features": 50,
  "threads": 2,
  "memory_limit_mb": 4096,
  "case_timeout": 120.0,
  "total_seconds": 10800.0,
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
| real_world | classification | lightgbm | openfe-stateless | 6/6 | 18/18 | 1/5/0 | +0.98% | [-0.72%, +4.01%] |
| real_world | classification | lightgbm | polynomial | 6/6 | 18/18 | 0/6/0 | +0.11% | [-0.40%, +0.61%] |
| real_world | classification | linear | autofeat | 0/6 | 0/18 | 0/0/0 | n/a | n/a |
| real_world | classification | linear | baseline | 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world | classification | linear | featcopilot | 6/6 | 18/18 | 1/5/0 | +0.55% | [+0.06%, +1.31%] |
| real_world | classification | linear | openfe-stateless | 6/6 | 18/18 | 3/2/1 | +5.21% | [-0.05%, +12.52%] |
| real_world | classification | linear | polynomial | 6/6 | 18/18 | 4/1/1 | +4.84% | [-0.17%, +12.37%] |
| real_world | classification | random_forest | autofeat | 0/6 | 0/18 | 0/0/0 | n/a | n/a |
| real_world | classification | random_forest | baseline | 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world | classification | random_forest | featcopilot | 6/6 | 18/18 | 0/5/1 | -0.41% | [-1.34%, +0.18%] |
| real_world | classification | random_forest | openfe-stateless | 6/6 | 18/18 | 2/4/0 | +0.96% | [-0.13%, +2.48%] |
| real_world | classification | random_forest | polynomial | 6/6 | 18/18 | 1/5/0 | +0.36% | [-0.25%, +1.02%] |
| real_world | regression | lightgbm | autofeat | 5/6 | 15/18 | 5/0/0 | +3.63% | [+2.21%, +5.53%] |
| real_world | regression | lightgbm | baseline | 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world | regression | lightgbm | featcopilot | 6/6 | 18/18 | 0/6/0 | +0.08% | [-0.01%, +0.24%] |
| real_world | regression | lightgbm | openfe-stateless | 6/6 | 18/18 | 5/1/0 | +3.49% | [+1.40%, +5.77%] |
| real_world | regression | lightgbm | polynomial | 6/6 | 18/18 | 1/2/3 | -1.01% | [-2.30%, +0.61%] |
| real_world | regression | linear | autofeat | 5/6 | 15/18 | 3/0/2 | -16.06% | [-50.33%, +6.52%] |
| real_world | regression | linear | baseline | 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world | regression | linear | featcopilot | 6/6 | 18/18 | 0/6/0 | +0.00% | [-0.00%, +0.01%] |
| real_world | regression | linear | openfe-stateless | 6/6 | 18/18 | 4/2/0 | +14.18% | [+2.85%, +29.28%] |
| real_world | regression | linear | polynomial | 6/6 | 18/18 | 5/0/1 | +15.30% | [-0.70%, +36.81%] |
| real_world | regression | random_forest | autofeat | 5/6 | 15/18 | 3/2/0 | +3.41% | [+1.05%, +7.22%] |
| real_world | regression | random_forest | baseline | 6/6 | 18/18 | 0/6/0 | +0.00% | [+0.00%, +0.00%] |
| real_world | regression | random_forest | featcopilot | 6/6 | 18/18 | 0/6/0 | +0.03% | [-0.07%, +0.17%] |
| real_world | regression | random_forest | openfe-stateless | 6/6 | 18/18 | 5/0/1 | +3.35% | [-0.34%, +7.55%] |
| real_world | regression | random_forest | polynomial | 6/6 | 18/18 | 0/3/3 | -1.56% | [-2.95%, -0.24%] |
| synthetic | classification | lightgbm | autofeat | 0/1 | 0/3 | 0/0/0 | n/a | n/a |
| synthetic | classification | lightgbm | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | classification | lightgbm | featcopilot | 1/1 | 3/3 | 1/0/0 | +6.11% | n/a |
| synthetic | classification | lightgbm | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +27.97% | n/a |
| synthetic | classification | lightgbm | polynomial | 1/1 | 3/3 | 1/0/0 | +27.07% | n/a |
| synthetic | classification | linear | autofeat | 0/1 | 0/3 | 0/0/0 | n/a | n/a |
| synthetic | classification | linear | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | classification | linear | featcopilot | 1/1 | 3/3 | 1/0/0 | +28.70% | n/a |
| synthetic | classification | linear | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +46.09% | n/a |
| synthetic | classification | linear | polynomial | 1/1 | 3/3 | 1/0/0 | +54.23% | n/a |
| synthetic | classification | random_forest | autofeat | 0/1 | 0/3 | 0/0/0 | n/a | n/a |
| synthetic | classification | random_forest | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | classification | random_forest | featcopilot | 1/1 | 3/3 | 1/0/0 | +15.67% | n/a |
| synthetic | classification | random_forest | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +22.73% | n/a |
| synthetic | classification | random_forest | polynomial | 1/1 | 3/3 | 1/0/0 | +17.78% | n/a |
| synthetic | regression | lightgbm | autofeat | 1/1 | 3/3 | 1/0/0 | +33.03% | n/a |
| synthetic | regression | lightgbm | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | regression | lightgbm | featcopilot | 1/1 | 3/3 | 1/0/0 | +20.54% | n/a |
| synthetic | regression | lightgbm | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +31.05% | n/a |
| synthetic | regression | lightgbm | polynomial | 1/1 | 3/3 | 1/0/0 | +31.26% | n/a |
| synthetic | regression | linear | autofeat | 1/1 | 3/3 | 1/0/0 | +83.14% | n/a |
| synthetic | regression | linear | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | regression | linear | featcopilot | 1/1 | 3/3 | 1/0/0 | +65.15% | n/a |
| synthetic | regression | linear | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +60.46% | n/a |
| synthetic | regression | linear | polynomial | 1/1 | 3/3 | 1/0/0 | +82.99% | n/a |
| synthetic | regression | random_forest | autofeat | 1/1 | 3/3 | 1/0/0 | +26.19% | n/a |
| synthetic | regression | random_forest | baseline | 1/1 | 3/3 | 0/1/0 | +0.00% | n/a |
| synthetic | regression | random_forest | featcopilot | 1/1 | 3/3 | 1/0/0 | +17.78% | n/a |
| synthetic | regression | random_forest | openfe-stateless | 1/1 | 3/3 | 1/0/0 | +23.03% | n/a |
| synthetic | regression | random_forest | polynomial | 1/1 | 3/3 | 1/0/0 | +20.75% | n/a |

## Per-dataset means (complete folds only)

| Dataset | Learner | Method | Primary loss | FE seconds (inner + outer) | Pipeline seconds | Features |
|---|---|---|---|---|---|---|
| abalone | lightgbm | autofeat | 2.136438 | 23.55 | 24.79 | 24.3 |
| abalone | lightgbm | baseline | 2.175105 | 0.03 | 1.00 | 8.0 |
| abalone | lightgbm | featcopilot | 2.175105 | 6.29 | 7.29 | 8.0 |
| abalone | lightgbm | openfe-stateless | 2.130982 | 37.31 | 39.16 | 58.0 |
| abalone | lightgbm | polynomial | 2.189645 | 0.04 | 1.81 | 36.0 |
| abalone | linear | autofeat | 2.602416 | 23.55 | 23.62 | 24.3 |
| abalone | linear | baseline | 2.206077 | 0.03 | 0.10 | 8.0 |
| abalone | linear | featcopilot | 2.206077 | 6.29 | 6.35 | 8.0 |
| abalone | linear | openfe-stateless | 2.187064 | 37.31 | 37.43 | 58.0 |
| abalone | linear | polynomial | 2.462576 | 0.04 | 0.14 | 36.0 |
| abalone | random_forest | autofeat | 2.115958 | 23.55 | 29.59 | 24.3 |
| abalone | random_forest | baseline | 2.170321 | 0.03 | 2.15 | 8.0 |
| abalone | random_forest | featcopilot | 2.171956 | 6.29 | 8.39 | 8.0 |
| abalone | random_forest | openfe-stateless | 2.111774 | 37.31 | 50.75 | 58.0 |
| abalone | random_forest | polynomial | 2.177334 | 0.04 | 8.44 | 36.0 |
| bank | lightgbm | baseline | 0.429811 | 0.02 | 1.17 | 7.0 |
| bank | lightgbm | featcopilot | 0.429811 | 7.80 | 8.88 | 7.0 |
| bank | lightgbm | openfe-stateless | 0.433489 | 36.30 | 38.25 | 57.0 |
| bank | lightgbm | polynomial | 0.433064 | 0.04 | 1.81 | 35.0 |
| bank | linear | baseline | 0.520463 | 0.02 | 0.11 | 7.0 |
| bank | linear | featcopilot | 0.520463 | 7.80 | 7.88 | 7.0 |
| bank | linear | openfe-stateless | 0.472810 | 36.30 | 36.56 | 57.0 |
| bank | linear | polynomial | 0.494596 | 0.04 | 0.23 | 35.0 |
| bank | random_forest | baseline | 0.447738 | 0.02 | 1.55 | 7.0 |
| bank | random_forest | featcopilot | 0.447738 | 7.80 | 9.34 | 7.0 |
| bank | random_forest | openfe-stateless | 0.443329 | 36.30 | 39.17 | 57.0 |
| bank | random_forest | polynomial | 0.446570 | 0.04 | 2.20 | 35.0 |
| cpu | lightgbm | baseline | 2.267884 | 0.06 | 1.56 | 21.0 |
| cpu | lightgbm | featcopilot | 2.267907 | 13.33 | 14.72 | 21.0 |
| cpu | lightgbm | openfe-stateless | 2.270341 | 65.49 | 68.23 | 71.0 |
| cpu | lightgbm | polynomial | 2.291728 | 0.09 | 3.33 | 71.0 |
| cpu | linear | baseline | 10.095029 | 0.06 | 0.15 | 21.0 |
| cpu | linear | featcopilot | 10.095029 | 13.33 | 13.39 | 21.0 |
| cpu | linear | openfe-stateless | 4.880648 | 65.49 | 65.63 | 71.0 |
| cpu | linear | polynomial | 3.417816 | 0.09 | 0.22 | 71.0 |
| cpu | random_forest | baseline | 2.591104 | 0.06 | 6.17 | 21.0 |
| cpu | random_forest | featcopilot | 2.590787 | 13.33 | 19.39 | 21.0 |
| cpu | random_forest | openfe-stateless | 2.531531 | 65.49 | 88.65 | 71.0 |
| cpu | random_forest | polynomial | 2.586981 | 0.09 | 22.30 | 71.0 |
| credit | lightgbm | baseline | 0.486485 | 0.03 | 1.18 | 10.0 |
| credit | lightgbm | featcopilot | 0.486261 | 7.79 | 8.89 | 10.3 |
| credit | lightgbm | openfe-stateless | 0.486305 | 39.48 | 41.52 | 60.0 |
| credit | lightgbm | polynomial | 0.485342 | 0.06 | 2.29 | 60.0 |
| credit | linear | baseline | 0.544493 | 0.03 | 0.15 | 10.0 |
| credit | linear | featcopilot | 0.531913 | 7.79 | 7.90 | 10.3 |
| credit | linear | openfe-stateless | 0.554226 | 39.48 | 39.72 | 60.0 |
| credit | linear | polynomial | 0.562476 | 0.06 | 0.33 | 60.0 |
| credit | random_forest | baseline | 0.480040 | 0.03 | 1.69 | 10.0 |
| credit | random_forest | featcopilot | 0.481467 | 7.79 | 9.41 | 10.3 |
| credit | random_forest | openfe-stateless | 0.482884 | 39.48 | 42.28 | 60.0 |
| credit | random_forest | polynomial | 0.482065 | 0.06 | 2.25 | 60.0 |
| credit_default | lightgbm | baseline | 0.575960 | 0.06 | 1.37 | 20.0 |
| credit_default | lightgbm | featcopilot | 0.576476 | 8.97 | 10.29 | 20.7 |
| credit_default | lightgbm | openfe-stateless | 0.578877 | 55.28 | 57.59 | 70.0 |
| credit_default | lightgbm | polynomial | 0.575883 | 0.10 | 2.58 | 70.0 |
| credit_default | linear | baseline | 0.616422 | 0.06 | 0.19 | 20.0 |
| credit_default | linear | featcopilot | 0.613089 | 8.97 | 9.08 | 20.7 |
| credit_default | linear | openfe-stateless | 0.610915 | 55.28 | 55.52 | 70.0 |
| credit_default | linear | polynomial | 0.599781 | 0.10 | 0.36 | 70.0 |
| credit_default | random_forest | baseline | 0.573558 | 0.06 | 2.06 | 20.0 |
| credit_default | random_forest | featcopilot | 0.574405 | 8.97 | 10.93 | 20.7 |
| credit_default | random_forest | openfe-stateless | 0.565351 | 55.28 | 58.28 | 70.0 |
| credit_default | random_forest | polynomial | 0.568969 | 0.10 | 2.54 | 70.0 |
| diamonds | lightgbm | autofeat | 0.106006 | 15.89 | 17.19 | 13.7 |
| diamonds | lightgbm | baseline | 0.108669 | 0.04 | 1.35 | 9.0 |
| diamonds | lightgbm | featcopilot | 0.108673 | 12.05 | 13.09 | 9.0 |
| diamonds | lightgbm | openfe-stateless | 0.106435 | 37.61 | 40.00 | 59.0 |
| diamonds | lightgbm | polynomial | 0.109723 | 0.06 | 1.93 | 30.0 |
| diamonds | linear | autofeat | 0.131190 | 15.89 | 15.98 | 13.7 |
| diamonds | linear | baseline | 0.142347 | 0.04 | 0.13 | 9.0 |
| diamonds | linear | featcopilot | 0.142347 | 12.05 | 12.13 | 9.0 |
| diamonds | linear | openfe-stateless | 0.121424 | 37.61 | 37.77 | 59.0 |
| diamonds | linear | polynomial | 0.124001 | 0.06 | 0.16 | 30.0 |
| diamonds | random_forest | autofeat | 0.128964 | 15.89 | 20.52 | 13.7 |
| diamonds | random_forest | baseline | 0.129869 | 0.04 | 2.92 | 9.0 |
| diamonds | random_forest | featcopilot | 0.129823 | 12.05 | 15.01 | 9.0 |
| diamonds | random_forest | openfe-stateless | 0.135048 | 37.61 | 54.79 | 59.0 |
| diamonds | random_forest | polynomial | 0.134897 | 0.06 | 9.25 | 30.0 |
| elevators | lightgbm | autofeat | 0.002376 | 16.51 | 17.88 | 22.3 |
| elevators | lightgbm | baseline | 0.002553 | 0.05 | 1.37 | 16.0 |
| elevators | lightgbm | featcopilot | 0.002554 | 18.09 | 19.17 | 16.0 |
| elevators | lightgbm | openfe-stateless | 0.002349 | 51.39 | 53.36 | 66.0 |
| elevators | lightgbm | polynomial | 0.002637 | 0.08 | 2.80 | 66.0 |
| elevators | linear | autofeat | 0.002696 | 16.51 | 16.57 | 22.3 |
| elevators | linear | baseline | 0.002885 | 0.05 | 0.12 | 16.0 |
| elevators | linear | featcopilot | 0.002885 | 18.09 | 18.14 | 16.0 |
| elevators | linear | openfe-stateless | 0.002434 | 51.39 | 51.51 | 66.0 |
| elevators | linear | polynomial | 0.002457 | 0.08 | 0.20 | 66.0 |
| elevators | random_forest | autofeat | 0.002874 | 16.51 | 21.87 | 22.3 |
| elevators | random_forest | baseline | 0.003226 | 0.05 | 3.66 | 16.0 |
| elevators | random_forest | featcopilot | 0.003231 | 18.09 | 21.74 | 16.0 |
| elevators | random_forest | openfe-stateless | 0.002822 | 51.39 | 69.04 | 66.0 |
| elevators | random_forest | polynomial | 0.003301 | 0.08 | 15.71 | 66.0 |
| eye_movements | lightgbm | baseline | 0.655070 | 0.07 | 1.64 | 23.0 |
| eye_movements | lightgbm | featcopilot | 0.649740 | 9.60 | 11.08 | 23.7 |
| eye_movements | lightgbm | openfe-stateless | 0.656264 | 49.40 | 51.92 | 73.0 |
| eye_movements | lightgbm | polynomial | 0.649673 | 0.12 | 2.89 | 73.0 |
| eye_movements | linear | baseline | 0.677233 | 0.07 | 0.25 | 23.0 |
| eye_movements | linear | featcopilot | 0.677071 | 9.60 | 9.76 | 23.7 |
| eye_movements | linear | openfe-stateless | 0.668506 | 49.40 | 49.68 | 73.0 |
| eye_movements | linear | polynomial | 0.670461 | 0.12 | 0.42 | 73.0 |
| eye_movements | random_forest | baseline | 0.646804 | 0.07 | 2.34 | 23.0 |
| eye_movements | random_forest | featcopilot | 0.644243 | 9.60 | 11.86 | 23.7 |
| eye_movements | random_forest | openfe-stateless | 0.647158 | 49.40 | 52.82 | 73.0 |
| eye_movements | random_forest | polynomial | 0.643195 | 0.12 | 3.32 | 73.0 |
| heloc | lightgbm | baseline | 0.561693 | 0.06 | 1.27 | 22.0 |
| heloc | lightgbm | featcopilot | 0.560299 | 9.21 | 10.38 | 24.0 |
| heloc | lightgbm | openfe-stateless | 0.567016 | 61.42 | 63.43 | 72.0 |
| heloc | lightgbm | polynomial | 0.564817 | 0.11 | 2.69 | 72.0 |
| heloc | linear | baseline | 0.570431 | 0.06 | 0.22 | 22.0 |
| heloc | linear | featcopilot | 0.567891 | 9.21 | 9.35 | 24.0 |
| heloc | linear | openfe-stateless | 0.569361 | 61.42 | 61.72 | 72.0 |
| heloc | linear | polynomial | 0.560205 | 0.11 | 0.44 | 72.0 |
| heloc | random_forest | baseline | 0.557756 | 0.06 | 1.81 | 22.0 |
| heloc | random_forest | featcopilot | 0.556863 | 9.21 | 10.95 | 24.0 |
| heloc | random_forest | openfe-stateless | 0.559203 | 61.42 | 64.15 | 72.0 |
| heloc | random_forest | polynomial | 0.561788 | 0.11 | 2.90 | 72.0 |
| houses | lightgbm | autofeat | 0.234051 | 27.35 | 29.01 | 34.3 |
| houses | lightgbm | baseline | 0.244681 | 0.02 | 1.22 | 8.0 |
| houses | lightgbm | featcopilot | 0.244681 | 25.07 | 26.24 | 8.0 |
| houses | lightgbm | openfe-stateless | 0.228722 | 38.19 | 40.74 | 58.0 |
| houses | lightgbm | polynomial | 0.250435 | 0.05 | 2.60 | 44.0 |
| houses | linear | autofeat | 0.611115 | 27.35 | 27.41 | 34.3 |
| houses | linear | baseline | 0.338541 | 0.02 | 0.08 | 8.0 |
| houses | linear | featcopilot | 0.338541 | 25.07 | 25.12 | 8.0 |
| houses | linear | openfe-stateless | 0.339693 | 38.19 | 38.31 | 58.0 |
| houses | linear | polynomial | 0.311700 | 0.05 | 0.14 | 44.0 |
| houses | random_forest | autofeat | 0.255405 | 27.35 | 36.77 | 34.3 |
| houses | random_forest | baseline | 0.260752 | 0.02 | 3.34 | 8.0 |
| houses | random_forest | featcopilot | 0.260752 | 25.07 | 28.49 | 8.0 |
| houses | random_forest | openfe-stateless | 0.246897 | 38.19 | 60.36 | 58.0 |
| houses | random_forest | polynomial | 0.269778 | 0.05 | 15.72 | 44.0 |
| magic | lightgbm | baseline | 0.344359 | 0.03 | 1.32 | 10.0 |
| magic | lightgbm | featcopilot | 0.342473 | 10.06 | 11.31 | 10.0 |
| magic | lightgbm | openfe-stateless | 0.315621 | 38.52 | 40.96 | 60.0 |
| magic | lightgbm | polynomial | 0.341236 | 0.05 | 2.83 | 60.0 |
| magic | linear | baseline | 0.490530 | 0.03 | 0.14 | 10.0 |
| magic | linear | featcopilot | 0.490775 | 10.06 | 10.16 | 10.0 |
| magic | linear | openfe-stateless | 0.385100 | 38.52 | 38.77 | 60.0 |
| magic | linear | polynomial | 0.383191 | 0.05 | 0.36 | 60.0 |
| magic | random_forest | baseline | 0.357934 | 0.03 | 2.31 | 10.0 |
| magic | random_forest | featcopilot | 0.367177 | 10.06 | 12.27 | 10.0 |
| magic | random_forest | openfe-stateless | 0.342755 | 38.52 | 42.34 | 60.0 |
| magic | random_forest | polynomial | 0.351890 | 0.05 | 3.90 | 60.0 |
| polynomial_regression | lightgbm | autofeat | 2.577464 | 30.18 | 31.50 | 25.7 |
| polynomial_regression | lightgbm | baseline | 3.848670 | 0.03 | 1.19 | 12.0 |
| polynomial_regression | lightgbm | featcopilot | 3.058279 | 14.16 | 15.28 | 16.0 |
| polynomial_regression | lightgbm | openfe-stateless | 2.653784 | 43.40 | 45.45 | 62.0 |
| polynomial_regression | lightgbm | polynomial | 2.645763 | 0.05 | 2.53 | 62.0 |
| polynomial_regression | linear | autofeat | 1.551619 | 30.18 | 30.23 | 25.7 |
| polynomial_regression | linear | baseline | 9.204904 | 0.03 | 0.10 | 12.0 |
| polynomial_regression | linear | featcopilot | 3.207920 | 14.16 | 14.20 | 16.0 |
| polynomial_regression | linear | openfe-stateless | 3.639468 | 43.40 | 43.47 | 62.0 |
| polynomial_regression | linear | polynomial | 1.565831 | 0.05 | 0.13 | 62.0 |
| polynomial_regression | random_forest | autofeat | 3.475279 | 30.18 | 34.41 | 25.7 |
| polynomial_regression | random_forest | baseline | 4.708660 | 0.03 | 2.83 | 12.0 |
| polynomial_regression | random_forest | featcopilot | 3.871327 | 14.16 | 17.26 | 16.0 |
| polynomial_regression | random_forest | openfe-stateless | 3.624329 | 43.40 | 52.97 | 62.0 |
| polynomial_regression | random_forest | polynomial | 3.731550 | 0.05 | 9.76 | 62.0 |
| wine | lightgbm | autofeat | 0.637357 | 36.85 | 39.00 | 48.3 |
| wine | lightgbm | baseline | 0.654417 | 0.03 | 1.10 | 11.0 |
| wine | lightgbm | featcopilot | 0.651329 | 11.26 | 12.29 | 12.3 |
| wine | lightgbm | openfe-stateless | 0.638328 | 40.09 | 42.76 | 61.0 |
| wine | lightgbm | polynomial | 0.639746 | 0.06 | 3.14 | 61.0 |
| wine | linear | autofeat | 0.709686 | 36.85 | 36.94 | 48.3 |
| wine | linear | baseline | 0.737508 | 0.03 | 0.10 | 11.0 |
| wine | linear | featcopilot | 0.737326 | 11.26 | 11.32 | 12.3 |
| wine | linear | openfe-stateless | 0.718771 | 40.09 | 40.21 | 61.0 |
| wine | linear | polynomial | 0.725725 | 0.06 | 0.18 | 61.0 |
| wine | random_forest | autofeat | 0.632100 | 36.85 | 49.58 | 48.3 |
| wine | random_forest | baseline | 0.637791 | 0.03 | 3.25 | 11.0 |
| wine | random_forest | featcopilot | 0.635647 | 11.26 | 14.78 | 12.3 |
| wine | random_forest | openfe-stateless | 0.629847 | 40.09 | 59.43 | 61.0 |
| wine | random_forest | polynomial | 0.635064 | 0.06 | 19.68 | 61.0 |
| xor_classification | lightgbm | baseline | 0.482452 | 0.05 | 1.68 | 20.0 |
| xor_classification | lightgbm | featcopilot | 0.452956 | 8.12 | 9.50 | 23.3 |
| xor_classification | lightgbm | openfe-stateless | 0.347511 | 51.60 | 53.85 | 70.0 |
| xor_classification | lightgbm | polynomial | 0.351865 | 0.08 | 2.85 | 70.0 |
| xor_classification | linear | baseline | 0.685736 | 0.05 | 0.15 | 20.0 |
| xor_classification | linear | featcopilot | 0.488924 | 8.12 | 8.20 | 23.3 |
| xor_classification | linear | openfe-stateless | 0.369699 | 51.60 | 51.74 | 70.0 |
| xor_classification | linear | polynomial | 0.313876 | 0.08 | 0.20 | 70.0 |
| xor_classification | random_forest | baseline | 0.615276 | 0.05 | 1.98 | 20.0 |
| xor_classification | random_forest | featcopilot | 0.518887 | 8.12 | 9.87 | 23.3 |
| xor_classification | random_forest | openfe-stateless | 0.475431 | 51.60 | 53.97 | 70.0 |
| xor_classification | random_forest | polynomial | 0.505856 | 0.08 | 2.63 | 70.0 |

## Failures / timeouts

- magic / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 0 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 1 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 2 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- magic / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 0 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 1 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 2 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- bank / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 0 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 1 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 2 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- credit / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 0 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 1 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 2 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- heloc / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 0 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 1 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 2 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- credit_default / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 0 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 1 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 2 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- eye_movements / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 0 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 1 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 2 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- cpu / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 0 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 0 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 0 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 1 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 1 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 1 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 2 / autofeat / linear: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 2 / autofeat / random_forest: timeout: Case exceeded 120.0s wall-clock limit
- xor_classification / 42 / fold 2 / autofeat / lightgbm: timeout: Case exceeded 120.0s wall-clock limit

Raw fold metrics, hyperparameters, timings, split fingerprints, versions and data provenance: [results.json.gz](results.json.gz) (lossless archived JSON; `results.json` is also available locally).

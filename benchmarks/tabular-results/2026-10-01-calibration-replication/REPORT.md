# Robust classification calibration development study

Exposed development data only; not fresh confirmation. No source is silently dropped.
Cohort scope: canonical_nine; subset smoke results are not the full nine-family study.
Three-fold sigmoid calibration and plain prediction averaging share the exact same fitted bases.
Member feature counts / maximum width describe distinct fold-local plans, not a common feature union.
Brier-sum is mean(sum over classes of squared probability error), including both binary classes.

Bundle plain/calibrated outcomes reuse exactly the same three fitted base procedures. Row pipeline_seconds are standalone attributions with shared training, not additive measured cost. Physical-job dispatch wall time is recorded once per physical_job_id, including retries; active_seconds in manifest.json is cumulative whole-controller wall time, including preparation, reporting and archiving. The ten-second report reserve is conservative, not a hard bound; actual budget overshoot is recorded.

Planned: 432 physical jobs / 648 prediction rows.
Scored: 648 / 648 prediction rows.

## Relative to raw single-model baseline

| Learner | Method | Complete families / requested | Mean loss reduction |
|---|---|---:|---:|
| lightgbm | baseline-calibrated-robust | 9 / 9 | -0.216892 |
| lightgbm | baseline-ensemble-robust | 9 / 9 | -0.023835 |
| lightgbm | baseline-robust | 9 / 9 | 0.000000 |
| lightgbm | featcopilot-calibrated-robust | 9 / 9 | -0.156378 |
| lightgbm | featcopilot-ensemble-robust | 9 / 9 | 0.049016 |
| lightgbm | featcopilot-frequency-aligned-robust | 9 / 9 | 0.022662 |
| linear | baseline-calibrated-robust | 9 / 9 | -0.741304 |
| linear | baseline-ensemble-robust | 9 / 9 | -0.044727 |
| linear | baseline-robust | 9 / 9 | 0.000000 |
| linear | featcopilot-calibrated-robust | 9 / 9 | -0.544942 |
| linear | featcopilot-ensemble-robust | 9 / 9 | 0.136309 |
| linear | featcopilot-frequency-aligned-robust | 9 / 9 | 0.180586 |

## FE versus matched raw representation

| Learner | Architecture | Complete families / requested | Mean loss reduction |
|---|---|---:|---:|
| lightgbm | single | 9 / 9 | 0.022662 |
| linear | single | 9 / 9 | 0.180586 |
| lightgbm | plain-ensemble | 9 / 9 | 0.068853 |
| linear | plain-ensemble | 9 / 9 | 0.164584 |
| lightgbm | sigmoid-ensemble | 9 / 9 | 0.046904 |
| linear | sigmoid-ensemble | 9 / 9 | 0.124227 |

## Calibration operation: calibrated minus plain

Equal-weight complete-fold families; negative loss/Brier differences favor calibration.

| Learner | Representation | Complete / requested families | Loss difference | Brier-sum difference |
|---|---|---:|---:|---:|
| linear | raw | 9 / 9 | 0.072753 | 0.036616 |
| linear | fe | 9 / 9 | 0.068993 | 0.033385 |
| lightgbm | raw | 9 / 9 | 0.020897 | 0.003132 |
| lightgbm | fe | 9 / 9 | 0.020454 | 0.003124 |

Matched raw/FE architecture summaries, all errors and original dataset metadata: `results.json`.
Exact lossless gzip parts (each <=480 KiB): `archive.json`. Source allowlist: `manifest.json`.
Final cumulative controller time and any budget overshoot: `manifest.json`.

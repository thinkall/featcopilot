# Robust classification calibration development study

Exposed development data only; not fresh confirmation. No source is silently dropped.
Cohort scope: canonical_nine; subset smoke results are not the full nine-family study.
Three-fold sigmoid calibration and plain prediction averaging share the exact same fitted bases.
Member feature counts / maximum width describe distinct fold-local plans, not a common feature union.
Brier-sum is mean(sum over classes of squared probability error), including both binary classes.

Bundle plain/calibrated outcomes reuse exactly the same three fitted base procedures. Row pipeline_seconds are standalone attributions with shared training, not additive measured cost. Physical-job dispatch wall time is recorded once per physical_job_id, including retries; active_seconds in manifest.json is cumulative whole-controller wall time, including preparation, reporting and archiving. The ten-second report reserve is conservative, not a hard bound; actual budget overshoot is recorded.

Planned: 216 physical jobs / 324 prediction rows.
Scored: 324 / 324 prediction rows.

## Relative to raw single-model baseline

| Learner | Method | Complete families / requested | Mean loss reduction |
|---|---|---:|---:|
| lightgbm | baseline-calibrated-robust | 9 / 9 | -0.227685 |
| lightgbm | baseline-ensemble-robust | 9 / 9 | -0.030293 |
| lightgbm | baseline-robust | 9 / 9 | 0.000000 |
| lightgbm | featcopilot-calibrated-robust | 9 / 9 | -0.111168 |
| lightgbm | featcopilot-ensemble-robust | 9 / 9 | 0.073950 |
| lightgbm | featcopilot-frequency-aligned-robust | 9 / 9 | 0.081202 |
| linear | baseline-calibrated-robust | 9 / 9 | -0.605915 |
| linear | baseline-ensemble-robust | 9 / 9 | -0.045694 |
| linear | baseline-robust | 9 / 9 | 0.000000 |
| linear | featcopilot-calibrated-robust | 9 / 9 | -0.434835 |
| linear | featcopilot-ensemble-robust | 9 / 9 | 0.060355 |
| linear | featcopilot-frequency-aligned-robust | 9 / 9 | 0.161221 |

## FE versus matched raw representation

| Learner | Architecture | Complete families / requested | Mean loss reduction |
|---|---|---:|---:|
| lightgbm | single | 9 / 9 | 0.081202 |
| linear | single | 9 / 9 | 0.161221 |
| lightgbm | plain-ensemble | 9 / 9 | 0.087585 |
| linear | plain-ensemble | 9 / 9 | 0.112175 |
| lightgbm | sigmoid-ensemble | 9 / 9 | 0.072580 |
| linear | sigmoid-ensemble | 9 / 9 | 0.124333 |

## Calibration operation: calibrated minus plain

Equal-weight complete-fold families; negative loss/Brier differences favor calibration.

| Learner | Representation | Complete / requested families | Loss difference | Brier-sum difference |
|---|---|---:|---:|---:|
| linear | raw | 9 / 9 | 0.074469 | 0.037233 |
| linear | fe | 9 / 9 | 0.067944 | 0.032946 |
| lightgbm | raw | 9 / 9 | 0.022215 | 0.003799 |
| lightgbm | fe | 9 / 9 | 0.021757 | 0.003490 |

Matched raw/FE architecture summaries, all errors and original dataset metadata: `results.json`.
Exact lossless gzip parts (each <=480 KiB): `archive.json`. Source allowlist: `manifest.json`.
Final cumulative controller time and any budget overshoot: `manifest.json`.

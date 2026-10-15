# Prediction averaging and sigmoid calibration

**Plain prediction averaging is promising for LightGBM feature engineering;
sigmoid calibration is not a general improvement here.** The plain
LightGBM FE ensemble improves loss by 8.76% versus its matched raw ensemble
and has no greater-than-5% family harms on this panel. The same architecture
harms linear CKD, while sigmoid calibration worsens mean log loss and
Brier sum for both learners and both representations. Defaults stay
unchanged; these are exposed development families, not confirmation.

All **324 prediction outcomes succeeded**, from **216 physical jobs**, in
**4.77469 cumulative active hours**. The 108 single-model control losses
match earlier runs exactly. All 108 paired bundles were checked: plain and
sigmoid views share the exact three fitted bases, identities and per-member
widths agree, and measured timing components reconcile.

## Why this differs from feature-rank bagging

Each base procedure owns its complete three-setting hyperparameter search,
feature discovery, frequency state, numeric caps, imputation and learner
fit inside its own training fold. Calibration rows cannot influence those
steps. The three fitted bases are retained; there is no all-training base
refit after calibration and no union of selected features pretending to
be one shared feature plan.

The plain view averages native class probabilities. The sigmoid view uses
the same fitted bases with sklearn's fold-local sigmoid calibrators,
preferring native margins when available. Thus their paired difference
isolates the calibration operation on those bases. Single versus ensemble
also changes training size and per-child tuning, so it is not a pure
variance-only intervention.

The nine original classification families use three outer folds, seed 42,
linear/LightGBM, 5,000-row cap, 1,000 candidates, 50 additions and three
internal selection repeats. Calibration is fixed at three folds and
sigmoid; no method or threshold sweep occurred.

## Feature value within matched architectures

Mean log-loss reductions against the corresponding raw representation:

| Learner | Single FE vs single raw | Plain FE ensemble vs plain raw ensemble | Sigmoid FE ensemble vs sigmoid raw ensemble |
|---|---|---|---|
| Linear | **+16.12%** | +11.22% | +12.43% |
| LightGBM | +8.12% | **+8.76%** | +7.26% |

Matched gains alone are insufficient when the whole architecture degrades
the raw learner. Relative to the ordinary single raw baseline:

| Learner | Plain raw ensemble | Sigmoid raw ensemble | Plain FE ensemble | Sigmoid FE ensemble |
|---|---|---|---|---|
| Linear | -4.57% | **-60.59%** | +6.04% | **-43.48%** |
| LightGBM | -3.03% | **-22.77%** | +7.40% | **-11.12%** |

Relative losses are sensitive on low-loss tasks. Exact absolute losses and
proper-loss contrasts are preserved; none of these aggregate percentages
is a universal performance guarantee.

## Tail harms: a useful LightGBM result, not a universal winner

The plain LightGBM FE ensemble's worst family change is **0.40% worse**
than matched raw ensemble and **1.23% worse** than ordinary single raw.
All other family changes are within the declared 5% harm bound.

On Wisconsin/LightGBM, log loss changes from single raw **0.120432**
and single FE **0.147848** to plain raw ensemble **0.119365** and plain
FE ensemble **0.108319**. Thus the previous 22.76% single-FE harm is
absent in this experiment. Sigmoid FE scores 0.115767, worse than the
plain view using those same bases.

Linear ensembling is not uniformly beneficial. On CKD, single raw scores
**0.014859**, plain raw ensemble **0.021216**, and plain FE ensemble
**0.027152**: FE is **27.98% worse than its matched ensemble control**
and 82.74% worse than single raw. Do not switch architectures by dataset
to hide this result.

## Calibration does not automatically improve probability quality

Calibrated-minus-plain absolute losses, equally averaging complete
within-family fold means; positive values are worse:

| Learner / representation | Log-loss change | Brier-sum change |
|---|---|---|
| Linear / raw | +0.074469 | +0.037233 |
| Linear / FE | +0.067944 | +0.032946 |
| LightGBM / raw | +0.022215 | +0.003799 |
| LightGBM / FE | +0.021757 | +0.003490 |

Brier sum includes all class coordinates, including both binary coordinates.
Both proper losses reflect discrimination as well as calibration.
Sigmoid's restrictive mapping, limited calibration data and multiclass
normalization can be harmful; this experiment does not isolate the cause.
No post-result switch to isotonic or temperature scaling is applied.

## Cost and accounting

Mean standalone pipeline seconds per family/cell:

| Learner | Raw single | FE single | Raw plain ensemble | FE plain ensemble | FE sigmoid ensemble |
|---|---|---|---|---|---|
| Linear | 2.07 | 28.25 | 0.98 | 57.70 | 57.98 |
| LightGBM | 4.02 | 60.66 | 4.31 | 135.21 | 135.48 |

Standalone estimates and single-job timing boundaries differ slightly;
they are not the authority for physical budget consumption.
The plain and calibrated views share training and must not have their
row times added together. The controller's cumulative wall time is the
measurement authority. All jobs were sequential under the eight-thread
cap, with 900-second physical-job and 4,096 MiB RSS limits.

## Frozen follow-up, before additional outcomes

Retain a **learner-aware development policy**, not dataset-specific choices:
linear uses the existing single robust FE pipeline; LightGBM uses the
plain cross-fit FE ensemble. No policy is inferred for CatBoost or XGBoost.

Check this fixed policy on additional declared seeds **137 and 2029**
using the same nine families and full six-arm protocol. Calibration
remains a recorded diagnostic arm, not a candidate to switch to after
viewing the new results. The two-seed run is 648 prediction outcomes from
432 physical jobs, with a twelve-hour cumulative ceiling.
These additional splits are still development evidence, not new origins
or untouched confirmation. No library default is promoted.

## Evidence

All 324 outcome identities are unique and scored. Source remained:
`738d4fc39b57cb5fb3557070ed1cb7460c809114099656e9deb87e8a5262a40e`.
Exact source/result checksum and reconstruction checks passed.

- [REPORT.md](REPORT.md): paired architecture/calibration contrasts.
- [archive.json](archive.json): ordered lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): bounded source/manifest.
- [decision.json](decision.json): matched and ordinary-raw tail harms.

The current cycle has used **27.54499 of 72 hours**; **44.45501 hours
remain**. No additional live LLM calls were made.

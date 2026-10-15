# Measurement conclusions: 2026-09-16

**FeatCopilot is not SOTA on this development measurement.** It completed every
requested case, but usually returned the original feature set and delivered
little incremental real-data predictive value. The rowwise-restricted OpenFE
comparison demonstrates substantial remaining headroom, especially in regression.

This conclusion concerns the measured configurations, not every configuration
or all possible datasets. This is a development subset, not an independent
confirmation suite.

## What ran

- 12 real datasets (six classification, six regression) plus two separately
  reported synthetic diagnostics.
- Three outer folds, one seed (42), maximum 5,000 rows per real dataset.
- Five representations and three matched learners: 210 method/fold cases,
  producing 630 learner rows.
- Two-thread worker configuration; 120-second whole-case limit; 4 GiB sampled
  process-tree RSS limit; three-hour total budget.
- Completed in approximately 2 hours 17 minutes. Start/end source fingerprints
  match. There were no dataset-load failures or exhausted-total-budget cases.
- Environment: Python 3.13, pandas 2.3.3, sklearn 1.7.2, NumPy 2.4.4,
  LightGBM 4.6.0, OpenFE 0.0.12 and AutoFeat 2.1.1.

See [REPORT.md](REPORT.md) for all per-dataset summaries and [results.json.gz](results.json.gz)
for raw fold metrics, selected model hyperparameters, timings and provenance.
The archive is lossless gzip-compressed JSON; the uncompressed `results.json`
remains available locally but is ignored by Git.

## Real-data results

Values below are mean **relative primary-loss reductions versus the matched
raw-feature learner**, averaging folds within datasets first. Positive is better.
Each FeatCopilot/OpenFE/polynomial cell covers all six datasets of its task type.

| Learner | FeatCopilot classification | OpenFE-stateless classification | FeatCopilot regression | OpenFE-stateless regression |
|---|---|---|---|---|
| Linear | +0.55% | +5.21% | +0.00% | +14.18% |
| RandomForest | -0.41% | +0.96% | +0.03% | +3.35% |
| LightGBM | +0.26% | +0.98% | +0.08% | +3.49% |

Classification uses log loss; regression uses RMSE. These are not relative
R-squared improvements. Do not average the two task columns into a headline score.

FeatCopilot's classification W/T/L is 1/5/0 for linear models, 0/5/1 for RF,
and 0/6/0 for LightGBM. All three regression learners have 0/6/0.
The tie region is the predeclared 1% relative-loss tolerance, not statistical
equivalence. Dataset bootstrap intervals are in the report and are descriptive,
not multiplicity-adjusted proof of significance.

The polynomial control improved linear regression loss by +15.30% on average,
but worsened RF and LightGBM regression loss by -1.56% and -1.01%. This is
important evidence that useful features and safe selection depend on the
downstream learner.

## Reliability, runtime and failure interpretation

Baseline, FeatCopilot, polynomial and OpenFE-stateless each completed **42/42
method/fold cases**, including synthetic diagnostics.

AutoFeat completed 18/42 cases and timed out on 24/42. Those 24 timeouts account
for 72 unscored learner rows, which is why the runner exited with status 1.
The timeouts include every classification case and the CPU regression dataset;
they are not imports or fabricated baseline-equivalent results.

AutoFeat's real regression LightGBM reduction was +3.63%, but only over **five**
datasets. It cannot be ranked directly against a six-dataset aggregate.
Classification quality is unmeasured at this case budget; timeouts do not prove
that AutoFeat's eventual predictions would be worse.

On real tasks, mean combined inner-plus-outer FE time was approximately 11.63s
for FeatCopilot, 45.87s for OpenFE-stateless and 0.07s for the polynomial control.
These exclude downstream fitting and process startup. End-to-end and sampled
memory measurements are available in the raw records. Runtime is specific to
this shared Windows workstation and bounded configurations.

## Evidence-driven next tasks

1. **Make task type explicit.** CPU and wine worker logs contain gate failures
   caused by treating integer-valued regression targets as classification.
   Add a regression/classification override consistently across selectors and
   gate models, with integer-target regression tests. Do not "repair" this
   benchmark by silently perturbing targets.
2. **Separate generation, selection and gating ablations.** FeatCopilot emitted
   no added features in **29 of 36 real-data outer fits**. The current measurement
   does not identify how much each stage contributes. Compare unselected,
   selection-only and gate-enabled modes on development folds before changing
   thresholds.
3. **Align feature selection with learner and loss.** Controls show large linear
   gains that do not transfer to boosting. A shared RF-based gate should not be
   assumed to protect logistic/ridge/LightGBM metrics.
4. **Investigate the candidate budget and ordering.** Test whether caps prioritize
   simple powers over useful interactions, and eliminate duplicate candidates
   before spending selection budget. Use column-order invariance diagnostics.
5. **Broaden confirmation, not just tune this report.** Add repeated seeds,
   full-sized tasks and a held-out suite, stronger tuned boosting/AutoML
   references, an inductive full-operator OpenFE track, and budget curves that
   give slower comparators meaningful coverage.

No algorithm was retuned after viewing these outer-test results. Future
algorithm changes need a new source fingerprint and new evidence; this report
remains the preserved development baseline.

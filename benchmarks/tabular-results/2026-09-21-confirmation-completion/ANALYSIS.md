# Locked confirmation: complete six-family outcome

**The completion-only repair succeeded, and all 1,440 preregistered cells now
have scores across the original six families. Results support reliability
and a useful Airfoil regression gain, not a broad SOTA claim.**

## Repair lineage

The original run is preserved at
[`../2026-09-20-locked-confirmation`](../2026-09-20-locked-confirmation/ADMISSION_STATUS.md).
It scored 960 cells for four families and retained 480 admission failures for
COIL and Coupon. Both failed families had zero worker attempts.

The repair added only three source-declared STRING predictors to their
categorical declarations: COIL's `avgAge` and
`contributionPrivateThirdPartyInsurance`, and Coupon's `coupon`.
No dataset, target, feature membership/order, official split, learner search,
method or seed was replaced. The runtime Python files and project dependencies
were verified identical against the original source archive. No scores were
inspected to choose the metadata correction.

This separate run completed all 480 affected cells in **8.42 active hours**,
with unchanged source during execution. The combined outcome uses the original
four families' scored rows plus these two repaired families' rows. It does not
overwrite the original admission failures or call this an exact resume.

The reproducible combination rule, parent/repair run IDs, archive hashes and
combined summaries are in [combined_summary.json](combined_summary.json).
The original and repair result archives reconstruct their local JSON exactly,
and source archive checksums were verified.

## Preregistered cohort results

Primary-loss reduction is relative to the matched raw learner, averaging
official folds/repeats within each family before averaging families.
Classification uses log loss; regression uses RMSE.

| Learner | Classification: four families | Regression: two families |
|---|---|---|
| Linear | +0.02% | +9.71% |
| LightGBM | +0.00% | +3.26% |
| CatBoost | +0.00% | +5.79% |
| XGBoost | +0.01% | +0.93% |

At the preregistered 1% relative-loss tie tolerance, **every classification
family is a tie** for every learner. Regression has one win and one tie for
each learner. The classification result is essentially neutral; the positive
regression means should not hide that they are driven by one family.

### Airfoil versus QSAR fish toxicity

| Learner | Airfoil loss reduction | QSAR fish loss reduction |
|---|---|---|
| Linear | +19.36% | +0.06% |
| LightGBM | +6.78% | -0.27% |
| CatBoost | +11.66% | -0.09% |
| XGBoost | +2.10% | -0.24% |

Descriptive family-bootstrap intervals are in the combined summary. Only two
regression families and four classification families remain a small sample;
repeated folds do not create more independent datasets.

## Gate outcome and claim boundary

- Final scored-cell completion is 100%, **after a disclosed schema repair**.
  First-attempt admission completion was 66.7% and remains recorded.
- The worst family/learner aggregate primary-loss worsening was approximately
  0.27%, below the proposed 5% operational harm threshold on this cohort.
- Passing these operational checks is not statistical proof of no harm.
- There is no broad classification gain and no full-method SOTA comparison.
  This confirmation compared raw versus predictive FeatCopilot, not every
  unrestricted competing FE implementation.
- Legacy defaults remain unchanged. The result does not justify automatic
  default promotion.

These six families are now exposed. They must not be relabeled as fresh
confirmation after further algorithm changes.

Artifacts: [REPORT.md](REPORT.md) describes this two-family repair alone;
[manifest.json](manifest.json), [source.zip](source.zip), and
[archive.json](archive.json) retain its inputs and lossless result parts.

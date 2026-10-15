# Ten-family, four-learner frequency confirmation

**The frozen candidate fails the quality and harm gates. It completes every
cell and improves mean linear-classification loss, but causes substantial
regression harm and is not consistently competitive with strong references.
Do not promote it or claim SOTA.**

All **480 cells were attempted** in **17.36022 active hours**, including
preparation. **435 scored successfully**: raw and FeatCopilot each completed
120/120, OpenFE-inductive-native 111/120, and AutoFeat-native 84/120.
There were no total-budget skips. Exit code 1 reports the **45 explicit
reference failures**, not a truncated run. Failures have no substituted loss.

## Locked scope

Ten observational origin families: six classification and four regression.
The user approved **three-fold IID evaluation for the entire cohort**
before payload/model access, preserving the minimum of ten test examples
per class. Seed 2027, full rows, original sources/targets/features, matched
three-setting learner searches, and the 30/20/10 class-support margins
were fixed in advance. All actual admission checks passed.

Original OpenML task IDs, payload hashes and all official split identities
are retained as provenance. The thirty evaluation splits are new IID folds,
**not official-fold reproduction**. The earlier metadata-blocked proposal
remains in its audit and was not silently overwritten.

The single FeatCopilot candidate uniformly uses extended candidates,
importance screening, paired-gain selection and bounded frozen unary
frequencies: 1,000 candidates, 50 additions and three internal repeats.
References retain native output width. OpenFE is the documented inductive
adaptation, not unmodified transductive paper code. CatBoost's selection
learner sees numeric preprocessing while its final model uses native
categorical inputs; this representation difference remains explicit.

Every cell had eight threads, a 900-second deadline and a 4,096 MiB sampled
process-tree RSS limit, within a 24-hour total ceiling. No LLM calls occurred.

## Candidate performance on every requested family

Positive percentages mean lower mean log loss/RMSE than the matched raw
learner. Folds are averaged within each family before averaging families.

| Learner | Classification mean, 6 families | Classification median | Regression mean, 4 families | Regression median |
|---|---|---|---|---|
| Linear | +15.41% | +2.79% | **-26.68%** | -22.80% |
| LightGBM | +0.18% | 0.00% | +0.0039% | -0.58% |
| CatBoost | +5.25% | -0.82% | **-4.32%** | -2.52% |
| XGBoost | **-6.08%** | -0.04% | **-10.20%** | -3.30% |

The tiny LightGBM regression mean is essentially neutral, not a material
improvement. CatBoost's positive classification mean is driven by
Wall-Following Robot (+43.07%); its median is negative and W/T/L is 1/2/3.
Linear classification W/T/L is 3/2/1, with Wall-Following Robot contributing
a +75.71% relative reduction. All classification/regression uncertainty
intervals in [REPORT.md](REPORT.md) are descriptive, not universal
superiority guarantees.

## Material family-level harm

Nine of forty family/learner comparisons exceed the predeclared 5% harm
threshold. The largest examples:

| Family / learner | Raw primary loss | Candidate loss | Loss increase |
|---|---|---|---|
| Computer Hardware / linear | 75.139526 | 121.163480 | **61.25%** |
| Infrared Oral Temperature / linear | 0.285946 | 0.416359 | **45.61%** |
| Forest Fires / XGBoost | 65.412861 | 87.738660 | **34.13%** |
| Wall-Following Robot / XGBoost | 0.008297 | 0.010864 | **30.94%** |
| Vertebral Column / LightGBM | 0.474284 | 0.551664 | **16.32%** |
| Forest Fires / CatBoost | 61.326403 | 69.259485 | **12.94%** |

Other threshold violations are Vertebral Column with CatBoost/XGBoost and
Computer Hardware with XGBoost. Robot/XGBoost starts at a very low log loss;
both relative and absolute changes are shown to avoid exaggerating scale.
The hardware and temperature regressions remain substantial in their
respective RMSE units.

Internal selection is a heuristic, not a no-harm guarantee. This experiment
has no importance-only ablation on these families, so it cannot attribute
every failure specifically to frequency features.

## Compare references only on common complete families

Native-reference failures change coverage. Do not compare a six-family
candidate mean with a four- or five-family reference mean.

| Task / learner | Reference | Common families | Candidate mean reduction | Reference mean reduction |
|---|---|---|---|---|
| Classification / linear | OpenFE-native | 6 | +15.41% | +7.92% |
| Classification / linear | AutoFeat-native | 4 | +1.09% | -0.47% |
| Classification / LightGBM | OpenFE-native | 6 | +0.18% | +9.02% |
| Classification / LightGBM | AutoFeat-native | 4 | -5.21% | -1.51% |
| Classification / CatBoost | OpenFE-native | 5 | -2.31% | -0.52% |
| Classification / CatBoost | AutoFeat-native | 4 | -1.73% | -2.84% |
| Classification / XGBoost | OpenFE-native | 6 | -6.08% | -0.83% |
| Classification / XGBoost | AutoFeat-native | 4 | -1.36% | +0.97% |
| Regression / linear | OpenFE-native | 4 | -26.68% | -20.44% |
| Regression / linear | AutoFeat-native | 3 | -35.62% | -73.46% |
| Regression / LightGBM | OpenFE-native | 4 | +0.0039% | +0.50% |
| Regression / LightGBM | AutoFeat-native | 3 | +0.10% | -1.76% |
| Regression / CatBoost | OpenFE-native | 3 | -5.58% | -1.00% |
| Regression / CatBoost | AutoFeat-native | 3 | -5.58% | +0.24% |
| Regression / XGBoost | OpenFE-native | 3 | -13.58% | -0.71% |
| Regression / XGBoost | AutoFeat-native | 3 | -13.58% | -6.64% |

These are comparisons of fixed, bounded configurations. Poor reference
results do not turn a worse-than-raw candidate into a safe improvement.

## Frozen decision and failures

[decision.json](decision.json) preserves all exact gate values and
coverage-matched comparisons.

| Gate | Outcome |
|---|---|
| Candidate completes every cell | Pass: 120/120 |
| Positive mean versus raw in every task/learner stratum | **Fail:** four of eight strata have negative means |
| No observational family harm greater than 5% | **Fail:** nine of forty family/learner pairs |
| No worse mean than complete paired references in every stratum | **Fail** |
| Complete reference coverage for broad superiority claims | **Fail** |

AutoFeat-native exceeded RSS limits on Communities and Crime and
Wall-Following Robot, across all folds/learners: **24 failures**. Its
Ionosphere fits raised NaN-input errors in upstream LogisticRegressionCV:
**12 failures**. These remain errors; no upstream repair or baseline
fallback was applied.

OpenFE-inductive-native exceeded RSS limits on Communities and Crime with
CatBoost/XGBoost: **six failures**. Its Robot/CatBoost cells exceeded 900
seconds: **three timeouts**. No post-outcome retries, larger limits or
cohort substitutions were used.

## Integrity and next research boundary

All 480 cell identities are unique. Start/end/current source hashes matched
before further work:
`c30bfbfcacb2fc7d3c3398699d8b2ee67ace00807cecaf2de49c7c06f527a808`.
Exact result/source checksums and bounded archive reconstruction passed.

- [REPORT.md](REPORT.md): complete family metrics, timing and configuration.
- [archive.json](archive.json): ordered lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): bounded source/manifest parts.
- [decision.json](decision.json): promotion and harm gates.
- [Locked suite](../../suites/frequency_fresh_confirmation_iid_20260928.json)
  and [release audit](../../suites/frequency_fresh_confirmation_iid_release_20260928.json).

These ten origins are now exposed. Any diagnosis or revised algorithm using
their outcomes must be labeled retrospective development, not fresh
confirmation. Preserve this failed decision and keep defaults unchanged.
The next priority is training-selection reliability and the severe
regression cases, not another unmeasured expansion of the operator catalog.

Current-cycle dataset measurement totals **61.44467 hours**. After the
separate 0.25-hour operational allowance for synthetic timing probes,
**10.30533 hours remain** under the approved 72-hour ceiling. The shared
LLM ledger remains at 96 of 100 logical attempts.

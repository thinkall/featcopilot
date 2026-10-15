# Frozen independent native-reference confirmation

**The candidate did not pass the preregistered promotion gates. FeatCopilot
completed every requested cell and improved linear classification, but
remained behind the strong references with LightGBM. Defaults stay legacy;
SOTA is not demonstrated.**

The run attempted all **128 cells** in **4.26919 active hours**, including
preparation. **120 scored successfully**. Raw, importance-paired FeatCopilot
and OpenFE-inductive-native each completed **32/32**; AutoFeat-native
completed **24/32**, with six memory-limit failures and two timeouts.
Exit code 1 correctly reports those failures, not a crashed or incomplete
experiment. No failed cell received an invented loss or baseline substitute.

## Frozen scope

The twelve previously reserved families comprise **eleven observational**
and **one domain-simulation** origin. Nine observational classification
families include six binary and three multiclass tasks. Only **two**
observational regression families were eligible, limiting regression claims.

The independently locked protocol used full rows, seed 2026, matched
three-setting linear/LightGBM learner searches, and one uniform candidate:
extended candidates, joint importance, paired-gain selection, at most 1,000
candidates and 50 additions. No task-specific method switches were made.
The ten official tasks used exact repeat=0/fold=0/sample=0; the two UCI tasks
used three seeded IID folds. This is not the full repeated-split TabArena
leaderboard. All actual split-support admission checks passed.

Every cell had eight threads, a 900-second deadline and a 4,096 MiB sampled
process-tree RSS limit. The total ceiling was 38 hours, not the amount used.
OpenFE is the documented **inductive adaptation**, not an unmodified
transductive paper reproduction. Both references retained native output
width; OpenFE returned as many as 2,535 total columns, while FeatCopilot was
limited to 50 additions. This compares the declared bounded configurations,
not identical feature widths or unrestricted best-possible tuning.

## Observational primary-loss reductions

Positive percentages mean lower log loss/RMSE than the matched raw learner.
Average folds within each family first; never treat folds as independent
datasets.

| Task / learner | Families | FeatCopilot mean | FeatCopilot median | OpenFE-native mean |
|---|---|---|---|---|
| Classification / linear | 9 | +7.09% | +0.68% | -5.86% |
| Classification / LightGBM | 9 | +1.13% | 0.00% | +9.39% |
| Regression / linear | 2 | +0.34% | +0.34% | -1.33% |
| Regression / LightGBM | 2 | -0.95% | -0.95% | +2.17% |

FeatCopilot's classification W/T/L is **4/4/1** for linear and **2/7/0**
for LightGBM, using the predeclared 1% tolerance. Seven of nine boosted
classification families tie the raw learner exactly at the family level.
The full report retains descriptive family-bootstrap intervals; the small
cohort and single official split do not support a universal superiority claim.

AutoFeat classification coverage is only six of nine families. Comparing its
partial mean with FeatCopilot's nine-family mean would be misleading:

| Task / learner | Common families | FeatCopilot mean | AutoFeat-native mean |
|---|---|---|---|
| Classification / linear | 6 | +6.24% | +6.20% |
| Classification / LightGBM | 6 | +0.38% | +2.62% |
| Regression / linear | 2 | +0.34% | -7.31% |
| Regression / LightGBM | 2 | -0.95% | +3.24% |

The simulation remains separate: FeatCopilot improves RMSE by **7.60%**
with linear and **5.79%** with LightGBM; OpenFE improves by 7.53% and 5.08%.
Both AutoFeat simulation cells exceeded memory limits and remain unscored.

## Preregistered decision

[decision.json](decision.json) records the exact common-family gate values.

| Gate | Outcome |
|---|---|
| Candidate completes every requested cell | Pass: 32/32 |
| No observational family loss increase greater than 5% | Pass: worst increase 2.87% |
| Positive mean versus raw in every task/learner stratum | **Fail:** LightGBM regression -0.95% |
| No worse mean than complete paired strong references in every stratum | **Fail:** both LightGBM strata |
| Complete reference coverage for a broad superiority claim | **Fail:** eight AutoFeat failures |

The worst candidate classification regression was `tabarena-46913`/linear:
log loss **0.483081 to 0.496938**, a 2.87% increase. The worst candidate
regression change was `tabarena-46907`/LightGBM: RMSE **744.224731 to
761.601999**, a 2.33% increase. Passing the 5% threshold on this cohort
does not erase the larger regressions observed during development or create
a production no-harm guarantee.

## Cost and failure coverage

Mean measured pipeline seconds per family/cell, including inner and outer
feature fitting and learner tuning, but not all supervisor startup overhead:

| Task / learner | FeatCopilot | OpenFE-native |
|---|---|---|
| Classification / linear | 113.47 | 171.85 |
| Classification / LightGBM | 22.34 | 176.02 |
| Regression / linear | 4.94 | 124.94 |
| Regression / LightGBM | 12.67 | 125.99 |

AutoFeat-native removes the earlier nondefault internal memory-size hint.
This run had no recorded fitting/NaN errors, but still exceeded the external
RSS limit on `tabarena-46949`, `tabarena-46950`, and `tabarena-46962`, for both
learners. Both `tabarena-46960` cells exceeded 900 seconds. No retries,
raised limits, data substitutions or post-result cohort changes were applied.
Earlier hinted-reference failures remain unchanged.

## Evidence and next boundary

All 128 cell identities are unique. Start/end/current source hashes matched
at verification:
`886f18e9116e7dd7e22903b17d93218c739a24761927266d2df079e87e74422e`.
Source archive checksums and exact reconstruction of result, manifest and
source parts were verified before further development.

- [REPORT.md](REPORT.md): raw-family means, resource/configuration details.
- [archive.json](archive.json): ordered lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): bounded source/manifest parts.
- [decision.json](decision.json): coverage-aware promotion decision.
- [Locked suite](../../suites/final_reserved_confirmation_20260924.json) and
  [release evidence](../../suites/final_reserved_confirmation_release_20260924.json).

These twelve families are now exposed. A later algorithm must not reuse their
outcomes as untouched confirmation. Preserve this run and its failed cells;
return to explicitly labeled development work rather than altering the
frozen candidate retrospectively.

Current-cycle measurement totals **35.90575 of 72 hours**, leaving
**36.09425 hours**. No new LLM calls were made; 96 of 100 shared attempts
remain consumed.

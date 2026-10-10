# Joint learner-importance development comparison

**Joint importance closes much of the boosted-classification gap and improves
regression, but harmful classification regressions remain. Neither SOTA nor
default promotion is established.**

All **480 requested cells succeeded in 9.78235 active hours**. The comparison
retained the same 16 exposed families, three folds, seed 42, linear/LightGBM,
5,000-row cap, 1,000 candidates, 50 additions, eight threads and 600-second
cell limit. No reserved confirmation data or live LLM calls were used.

## Observational primary-loss reductions

Positive percentages mean lower loss than the matched raw learner, averaged
within family before averaging families. Classification uses log loss over
nine families; regression uses RMSE over six. These repeatedly examined
development tasks are not independent confirmation.

| Task / learner | Standard predictive | Importance + one-SE | Importance + paired | OpenFE-inductive |
|---|---|---|---|---|
| Classification / linear | +14.09% | +11.63% | +12.38% | +11.24% |
| Classification / LightGBM | -0.33% | +4.71% | +8.58% | +9.71% |
| Regression / linear | +10.65% | +10.97% | +13.37% | +8.28% |
| Regression / LightGBM | +1.62% | +1.66% | +2.06% | +2.56% |

Importance-paired medians are **+2.64%, 0.00%, +8.92%, and +1.65%** in the
same row order. Its boosted-classification W/T/L is **2/5/2** at the
predeclared 1% tolerance, compared with OpenFE's **3/1/5**. Large gains on a
few low-loss tasks must not be mistaken for uniformly better classification.
Descriptive family-bootstrap intervals for both importance-paired
classification means include zero; see [REPORT.md](REPORT.md).

Energy Efficiency is a separate simulation result: importance-paired reduces
RMSE by +76.99% with linear and +5.91% with LightGBM, compared with
OpenFE-inductive's +43.87% and +8.05%. It is excluded from the observational
means.

## Absolute losses and remaining harm

| Family / learner | Raw loss | Importance-paired loss | Relative reduction |
|---|---|---|---|
| Banknote / LightGBM | 0.015250 | 0.005345 | +64.95% |
| Chronic Kidney Disease / LightGBM | 0.035504 | 0.022914 | +35.46% |
| Wisconsin Diagnostic / LightGBM | 0.120432 | 0.146930 | **-22.00%** |
| Chronic Kidney Disease / linear | 0.016247 | 0.017848 | **-9.85%** |
| Wisconsin Diagnostic / linear | 0.089368 | 0.092795 | **-3.83%** |
| Houses / LightGBM | 0.244681 | 0.224733 | +8.15% |
| Concrete / linear | 10.559292 | 5.895346 | +44.17% |

The previous extended-paired CKD/linear regression of -46.28% is smaller,
not solved. Wisconsin's boosted regression is substantial even in absolute
log loss. Internal paired validation is a selection heuristic, not a
no-harm guarantee. No task-specific exception or post-hoc fallback is applied.

## Cost and training-only diagnosis

Mean successful pipeline seconds per family/cell, including inner and outer
feature fitting and model tuning:

| Task / learner | Standard | Importance-paired | OpenFE-inductive |
|---|---|---|---|
| Classification / linear | 104.57 | 106.46 | 130.26 |
| Classification / LightGBM | 31.44 | 29.26 | 132.38 |
| Regression / linear | 3.48 | 2.73 | 113.94 |
| Regression / LightGBM | 13.66 | 10.50 | 114.76 |

These are measured times for this environment and bounded configuration,
not asymptotic or universal speedup claims. Native-width references require
a separately declared comparison.

For the first houses/LightGBM outer fit, training-only deduplication removes
two constant and 22 exact duplicate candidates from 260 planned candidates,
including minima identical to original longitude. Candidate ranking fits
one joint learner on original and retained candidate columns; independent
internal holdouts still choose subset size. This confirms the intended
mechanism executes, not that deduplication alone caused the measured gain.

## Frozen next step

Stop choosing variants on this development panel. Freeze the existing
**importance-paired** configuration uniformly across tasks and learners for
a separately locked comparison on the 12 metadata-reserved families.
Do not select standard/importance variants family by family.

Compare raw features, this candidate, full-operator **OpenFE-inductive-native**
and **AutoFeat-native**. The latter removes the prior nondefault internal
memory-size hint; archived AutoFeat failures are not overwritten.
Use full rows, seed 2026, linear/LightGBM and unchanged three-setting learner
searches. Before any candidate modeling, lock exact official split
repeat=0/fold=0/sample=0 for OpenML sources and three seeded IID folds for
the two UCI sources. This is a bounded independent comparison, not the full
TabArena repeated-split leaderboard.

The planned run ceiling is 38 active hours, eight threads, 4,096 MiB sampled
process-tree RSS and 900 seconds/cell. Current-cycle consumption is
**31.63657 of 72 hours**, leaving 40.36343 hours before that run.
No LLM calls are scheduled; 96 of the shared 100 attempts remain consumed.
Reservations stay closed until the new protocol, audit and source snapshot
are locked. Defaults remain legacy regardless of experiment launch.

## Evidence integrity

All 480 cell identities are unique; failures would remain unscored.
Start/end/current source hashes matched at verification:
`6cdc6ab5021ca277e467b62183127f011b87229ffab8c540daf46d46d9986670`.
The source archive checksum matched the manifest, and exact reconstruction
of every archived artifact was verified before subsequent source work.

- [REPORT.md](REPORT.md): complete family means, uncertainty and configuration.
- [archive.json](archive.json): ordered, concatenated gzip result parts.
- [auxiliary_archive.json](auxiliary_archive.json): lossless bounded
  manifest/source parts and checksums.

Raw large artifacts remain local for exact resume; numbered public parts
preserve their bytes without exceeding the repository's per-file limit.

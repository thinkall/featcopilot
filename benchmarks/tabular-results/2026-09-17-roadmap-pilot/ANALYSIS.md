# Roadmap foundation pilot: 2026-09-17

**The foundation increment is implemented and measured. Stability improved
materially on the known regression cases, but these development results do
not establish SOTA or justify automatic default promotion.**

## Execution and evidence

The run completed 1,512 requested dataset/split/method/learner cells in
approximately **26.18 active hours**, under the approved 72-hour/eight-thread
ceiling. It used nine development families, three folds, seeds 42 and 137,
seven FE methods and four learners. The per-cell limit was 600 seconds with
a 4 GiB sampled worker-tree RSS cap.

Baseline, all three FeatCopilot variants, polynomial and OpenFE-stateless each
completed all 216 cells. AutoFeat completed 168 and timed out on 48. Those
timeouts remain unscored, accounting for exit code 1; there were no source
changes, dataset preparation failures, exhausted-budget cells, or confirmation
evaluations.

Evidence is retained in [REPORT.md](REPORT.md), [manifest.json](manifest.json),
[source.zip](source.zip), and the lossless result parts listed by
[archive.json](archive.json). The archive was checked byte-for-byte against
the local uncompressed results, including all 1,512 unique cell IDs. The source
archive checksum and start/end source hashes matched.

```python
import gzip
import json
from pathlib import Path

root = Path(".")
index = json.loads((root / "archive.json").read_text())
payload = b"".join((root / part).read_bytes() for part in index["parts"])
results = json.loads(gzip.decompress(payload))
```

## Cohort and comparability

Seven reviewed UCI development candidates were combined with cached houses
and wine diagnostics. Six UCI tasks are observational; Energy Efficiency is
simulation data and is reported separately.

The admitted variants intentionally exclude unavailable or inappropriate
predictors: Adult excludes the survey-weight field; Online Shoppers excludes
PageValues, durations and bounce/exit metrics; Steel Faults excludes all seven
target indicators; Energy uses heating load and excludes cooling load.
These reviewed variants are not unqualified reproductions of every published
score on similarly named datasets.

Protocol v3 gives each learner an independent budget and records source
snapshots, exact resume identity and family-aware outcomes. Its timeout rates
are not directly comparable with the old shared multi-learner case limit.
CatBoost's final learner uses native categoricals, but predictive selection
uses numeric preprocessing internally; that difference remains explicit.

## Main predictive-strategy results

Numbers are mean relative primary-loss reductions versus the matched raw
learner, first averaged within each dataset family. Positive is better.
Classification uses log loss and regression uses RMSE.

| Learner | Observational classification (5 families) | Observational regression (3 families) |
|---|---|---|
| Linear | +1.60% | +5.71% |
| LightGBM | +0.10% | +0.17% |
| CatBoost | +0.92% | +1.63% |
| XGBoost | +0.40% | +0.65% |

Uncertainty, W/T/L and comparator results are in the full report. The small
family counts and intervals do not support a broad superiority claim.
OpenFE-stateless, for example, has higher mean regression reductions for all
four learners on the common three-family observational cohort.

The separate Energy Efficiency simulation yielded +34.06% for linear
regression, -0.98% for LightGBM, -0.42% for CatBoost and +0.81% for XGBoost.
Do not blend those figures into an observational-data headline.

## Known regression cases: stability improved

On both measured seeds combined, predictive linear loss reduction is:

- **Houses: +7.43%**.
- **Wine: -1.05%**, still a small regression requiring disclosure.

Restricting the new run to seed 42 gives matching diagnostic splits and raw
baseline scores from the earlier run:

| Dataset / linear learner | Earlier predictive loss reduction | Current predictive loss reduction, seed 42 |
|---|---|---|
| Houses | -32.33% | +5.76% |
| Wine | -15.88% | +1.49% |

This is evidence that the combined numerical, screening and stability changes
addressed the observed severe failures. It is **not** an isolated causal test
of a particular change: several pieces changed together, and the known problem
families are development data.

## Cost and reliability tradeoffs

Mean inner-plus-outer predictive FE time on observational data was approximately:

| Learner | Predictive FE seconds | Legacy FE seconds | OpenFE-stateless FE seconds |
|---|---|---|---|
| Linear | 32.08 | 6.78 | 60.11 |
| LightGBM | 33.69 | 6.73 | 60.55 |
| CatBoost | 117.69 | 6.73 | 60.46 |
| XGBoost | 46.95 | 6.71 | 59.92 |

Repeated selection buys stability at meaningful cost. Especially for CatBoost,
small average quality gains do not yet establish a favorable accuracy/runtime
tradeoff. These times describe this machine, thread limit and workload, not
portable latency guarantees.

AutoFeat's classification coverage is three of five complete families for
each learner. Quality comparisons must use common completed families and
retain the missing cells; timeouts are not evidence of poor eventual
predictive quality.

## What this increment does and does not complete

Implemented foundations include catalog admission/provenance, seven new
development datasets, numerical consistency and repeated selection, fitted
statistical rules, per-learner budgets/resume, stronger raw learners and
reproducible artifacts. No paid LLM benchmarking occurred.

Remaining roadmap work includes broader rights/exposure-reviewed confirmation
admission, actual locked confirmation evaluation, label-horizon/embargo-aware
temporal protocols, supervised group/time target encodings, relational
evaluation, and separately budgeted live-LLM comparisons. Confirmation families
must be frozen before any tuning use; restricted sources remain blocked.

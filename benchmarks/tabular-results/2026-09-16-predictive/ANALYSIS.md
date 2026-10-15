# Predictive-strategy measurement: 2026-09-16

**The opt-in predictive strategy materially improves FeatCopilot, but this is
not a SOTA result.** It has the best mean linear-classification improvement
among the fully measured configurations in this development run. OpenFE-stateless
still has better mean regression and LightGBM-classification results, and the
new strategy has two substantial linear-regression regressions.

Legacy defaults remain unchanged, as requested. Use
`selection_strategy="predictive"` to opt in.
These scores measure tabular feature engineering without LLM calls. The separate
live `gpt-5.6-sol` synthetic generation check verifies LLM functionality, not
its predictive advantage on this suite.

## What ran

- All 12 real development datasets plus two separate synthetic diagnostics.
- Three folds, seed 42, maximum 5,000 real-data rows.
- Seven representations, each evaluated with linear models and LightGBM:
  294 method/fold cases and 588 learner rows.
- Same declared data, sampling and split policy as the preserved first
  measurement. Protocol version 2 adds explicit-task and predictive variants.
- Two threads, 120 seconds per method/fold, 4 GiB sampled process-tree RSS cap.
- Completed in approximately **2 hours 30 minutes**, within the approved
  four-hour measurement budget. Start, end and current source hashes matched
  when the evidence was verified.
- No dataset-load errors, source changes or total-budget exhaustion.

See [REPORT.md](REPORT.md), [manifest.json](manifest.json), and the lossless
raw JSON archive split into [part 1](results.json.gz.part1),
[part 2](results.json.gz.part2), and [part 3](results.json.gz.part3).
Each part stays below the repository's per-file size limit; no diagnostic
records were removed. The local uncompressed `results.json` is retained but
ignored by Git. To read the archive from this directory:

```python
import gzip
import json
from pathlib import Path

payload = b"".join(Path(f"results.json.gz.part{i}").read_bytes() for i in (1, 2, 3))
results = json.loads(gzip.decompress(payload))
```

## Real-data primary-loss reductions

Every cell below averages paired dataset-level loss reductions over all six
datasets of the corresponding task. Positive is better than the **same raw-feature
learner**. Classification uses log loss; regression uses RMSE.

| Task / learner | Legacy FeatCopilot | Explicit task only | Predictive strategy | OpenFE-stateless | Polynomial control |
|---|---|---|---|---|---|
| Classification / linear | +0.55% | +0.55% | **+6.76%** | +5.21% | +4.84% |
| Classification / LightGBM | +0.26% | +0.26% | +0.49% | **+0.98%** | +0.11% |
| Regression / linear | +0.00% | +0.02% | +8.67% | +14.18% | **+15.30%** |
| Regression / LightGBM | +0.08% | +0.02% | +2.10% | **+3.49%** | -1.01% |

Predictive W/T/L versus raw features, using the predeclared 1% relative-loss
tie tolerance:

| Task / learner | W/T/L | Descriptive dataset-bootstrap 95% interval |
|---|---|---|
| Classification / linear | 5 / 1 / 0 | +2.22% to +12.88% |
| Classification / LightGBM | 1 / 5 / 0 | -0.12% to +1.47% |
| Regression / linear | 3 / 1 / 2 | -14.07% to +35.54% |
| Regression / LightGBM | 5 / 1 / 0 | +0.85% to +3.45% |

These intervals are descriptive, not multiplicity-adjusted significance tests.
Single-seed results on six datasets per task do not establish broad superiority,
and the OpenFE comparator is an explicitly restricted variant.

## What improved

The predictive strategy no longer discards nearly every candidate: it added
features in **36/36 real-data linear-model fits** and **30/36 LightGBM fits**,
versus **7/36** with legacy selection.

Examples of predictive loss reductions against raw features:

- CPU / linear: **+71.54%** (RMSE 10.0950 to 2.8735).
- Magic Telescope / linear: **+20.01%**.
- Bank / linear: **+10.12%**.
- Credit default / linear: **+5.03%**.
- Elevators / LightGBM: **+4.79%**.
- Diamonds / LightGBM: **+3.42%**.

The task-only ablation shows that fixing integer-target routing is necessary
for correctness but **not sufficient** for meaningful performance gains.
The larger gains come from the broader candidate pool and different selection
procedure, although this run does not isolate their individual contributions.

Mean combined inner-plus-outer FE time per real-data representation was about
8.61 seconds for predictive linear selection and 8.16 seconds for predictive
LightGBM selection, versus 11.41 seconds for legacy FeatCopilot and 48.06 seconds
for OpenFE-stateless. Predictive representations are learner-specific and must
be fitted separately; shared representations can amortize their FE cost across
learners. The worker wall-clock fields capture that distinction.

## Regressions and limitations

Do not hide these behind the positive mean:

- **Houses / linear: -32.33%** loss reduction, meaning RMSE increased from
  0.3385 to 0.4480.
- **Wine / linear: -15.88%**, RMSE increased from 0.7375 to 0.8546.
- Houses selected 50 added features in every outer fit. Wine selected 50, 50
  and 25. Internal validation favored these subsets, but their external
  performance did not match the internal estimates.

The mechanism of these regressions is not established by aggregate scores.
Reciprocal/power-feature tails, collinearity, and single-holdout selection
stability need training-only diagnostics before choosing a remedy. Reducing
the feature cap after looking at these outer-test scores and reporting only
the better result would be test-set tuning, not confirmation.

All FeatCopilot, raw, polynomial and OpenFE-stateless cases completed.
AutoFeat timed out on **23/42 cases**, giving 46 unscored learner rows and
causing exit code 1. Its complete real-regression coverage is five datasets,
not six; the sixth has only partial fold coverage. Its +3.80% LightGBM
regression reduction must not be ranked directly against six-dataset means.

## Synthetic diagnostics

The predictive strategy improves XOR classification over legacy FeatCopilot,
but the polynomial control still has a larger linear gain. On synthetic
polynomial regression, predictive linear selection improves over raw features
by +58.66%, which is less than legacy's +65.15% and the polynomial control's
+82.99%. These diagnostics remain separate from real-data results.

## Next priorities

1. Investigate and constrain harmful linear-regression subsets with training-only
   stability checks and new synthetic stress tests for feature tails.
2. Compare residual-screening variants and candidate-family ablations while
   holding the final learner and resource budget fixed.
3. Improve conditional interactions for boosted models; their remaining gap
   to OpenFE-stateless is clear in this run.
4. Add fresh confirmation tasks, repeated seeds and larger-data tiers before
   promoting defaults or claiming SOTA.

No algorithm was changed while this measurement ran. This report preserves
both gains and regressions as development evidence.

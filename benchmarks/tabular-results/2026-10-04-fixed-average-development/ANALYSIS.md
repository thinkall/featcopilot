# Complete-panel fixed 50:50 raw/FE prediction replication

**Fixed averaging shows consistent development gains across all three
declared seeds, with less than 1% worst-family harm at each seed.**
This is a stronger tail result than the preceding learned-weight study,
but remains exposed development evidence—not fresh confirmation,
automatic default promotion, or SOTA.

All **243 outcomes succeeded** from **81 physical bundles** in **2.45070
active hours**. The complete nine-family panel includes Dry Bean, whose
two expensive OOF jobs timed out previously. This is a new, cheaper
algorithmic experiment; the OOF results and timeouts are unchanged.

## Fixed rule, no additional fitting

Each bundle fits the same full-training raw and FE procedures, including
their independent three-setting hyperparameter searches. It shares those
two fitted models across raw, FE and the fixed average:

`probability = 0.5 * raw_probability + 0.5 * FE_probability`.

There is no OOF fitting, optimizer, learned weight, calibration or
dataset-specific choice. Two complete procedures mean eight
prediction-pipeline fits per bundle, plus the existing FE-internal work.
All tuning and fitting remain inside the allowed outer-training rows.

## Replication results

Log-loss reduction is computed per family after averaging its relevant
fold losses. Positive percentages favor the method over matched raw.

| Seed | Fixed-average mean gain | Fixed-average worst family | Single-FE mean gain | Single-FE worst family |
|---|---|---|---|---|
| 42 | +7.68% | -0.92% | +8.12% | -22.76% |
| 137 | +6.12% | -0.006% | +1.15% | -19.06% |
| 2029 | +5.89% | -0.77% | +3.07% | -17.86% |
| All three | **+6.82%** | **-0.26%** | +4.75% | -1.73% |

The aggregated gains are not simple averages of the per-seed ratios.
Worst-family comparisons are retained separately for each seed so that
aggregation does not hide split-sensitive harms.

Across all three seeds, the fixed average gains 13.93% on CKD and 6.57%
on Wisconsin. At seed 42, Wisconsin's 22.76% single-FE harm is reduced to
0.92% harm; at seeds 137 and 2029, the fixed average improves that family
by 16.07% and 4.58%.

These results do not show that every future dataset will benefit.
Probability averaging can dilute a genuinely strong FE model and can
still harm raw performance. A fixed rule avoids meta-weight overfitting,
not all model or distribution-shift risks.

## Control and numerical parity

All 162 raw/FE control parameter choices and FE plan names match the
archived corresponding single-model runs. Discrimination metrics and
accuracy match exactly. Twelve control log losses differ by only one or
two float64 ULPs, and six Brier-sum values differ by one ULP.

The full-procedure prediction path validates and normalizes probabilities
before cached scoring, whereas canonical single controls score the native
probabilities directly. Those small arithmetic differences are explicitly
recorded in [control_parity.json](control_parity.json); they are not
misreported as bit-exact equality or removed by changing scores.

## Evidence and scope

All 243 identities are unique and scored. Source remained:
`a97b8300dc75352bf8b9b5f90b1abe8975ccb8366e76016c55817077ef3159db`.
Source/result checksums and exact bounded reconstruction passed.

- [REPORT.md](REPORT.md): complete panel, fixed policy and physical cost.
- [archive.json](archive.json): ordered lossless results.
- [auxiliary_archive.json](auxiliary_archive.json): manifest/source parts.
- [control_parity.json](control_parity.json): exact parameter/plan checks
  and all tiny metric differences.
- [decision.json](decision.json): per-seed gains and harm outcomes.

Keep existing library defaults unchanged. An explicitly selected
`raw_fe_average` prediction mode may expose this measured fixed rule,
with per-member plans and matched raw controls; it must not claim
independent confirmation or introduce a configurable weight search.

The current cycle has used **60.57826 of 72 measurement hours**,
leaving **11.42174 hours**. No new data sources or LLM calls were used.

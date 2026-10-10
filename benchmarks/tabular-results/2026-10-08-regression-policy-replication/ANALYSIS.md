# Frozen regression policy: two-seed replication

**All 340 cells completed, and all 136 component/control pairs matched
exactly.** The preregistered family-mean follow-up gates pass at both
seeds and pooled. Nevertheless, **severe individual-fold harms remain**,
so this does not establish robust general feature selection or justify
default promotion.

The frozen policy is single aligned FE for linear regression and fixed
50:50 raw/FE averaging for LightGBM. It was selected using the earlier
seed-42 development result, then replicated unchanged at seeds 137 and
2029. These are exposed development datasets, not fresh confirmation.

## Seven-family primary means

Positive percentages mean lower RMSE. Fold losses are averaged within
each family before calculating relative gains; families receive equal
weight. The pooled column combines the two replication seeds only,
not the development seed that selected the policy.

| Policy | Seed 137 vs raw | Seed 2029 vs raw | Pooled vs raw |
|---|---:|---:|---:|
| Linear: single FE | +24.58% | +23.00% | **+23.79%** |
| LightGBM: fixed average | +2.69% | +3.24% | **+2.97%** |
| LightGBM: single FE comparator | +0.46% | +2.48% | +1.48% |

LightGBM averaging improves directly over single FE by **2.12%**,
**0.75%** and **1.45%**, respectively. Those matched reductions are
computed from family losses, not differences between the raw-relative
percentages.

All pooled primary-family comparisons favor the frozen policies over
raw. At seed 2029, LightGBM averaging worsens the Energy family mean by
**0.162%**; this is below the predeclared 5% family-mean limit and remains
visible rather than being rounded into a universal win.

## Separate stress-family means

Neither selected stress family is included in the primary mean.

| Family | Policy | Seed 137 vs raw | Seed 2029 vs raw | Pooled vs raw |
|---|---|---:|---:|---:|
| Infrared oral temperature | Linear single FE | +3.66% | +3.71% | +3.69% |
| Hardware performance | Linear single FE | +16.10% | +18.51% | +17.30% |
| Infrared oral temperature | LightGBM average | +0.69% | +0.80% | +0.75% |
| Hardware performance | LightGBM average | +10.32% | +6.55% | +8.43% |

LightGBM averaging is **1.69% worse than single FE** on the pooled
hardware family mean, despite improving versus raw. It is not uniformly
better than both components.

Hardware reuses the same ten official outer splits at two different
estimator/tuning seeds. These are not twenty independent test datasets
or newly sampled official splits. Infrared uses three IID folds per seed
and retains its contemporaneous-measurement-only claim boundary.

## Individual-fold harms: the unresolved blocker

| Policy | Seed | Worst cell | RMSE increase vs matched raw |
|---|---:|---|---:|
| Linear single FE | 137 | Hardware fold 3 | **31.46%** |
| Linear single FE | 2029 | Hardware fold 6 | **245.61%** |
| LightGBM average | 137 | Auto MPG fold 1 | 1.97% |
| LightGBM average | 2029 | Hardware fold 2 | **15.63%** |

For context, LightGBM single FE's worst cell at seed 2029 is hardware
fold 0 at **75.04%** harm. Averaging reduces the largest observed harm
but does not eliminate it. The linear policy's excellent family means
do not excuse its severe tail failure.

The registered gate used **family-mean** harm, not a per-fold threshold.
Its passing result is therefore reported faithfully, but it is only a
necessary development follow-up condition. No safety guarantee,
production-default change, robust-selection acceptance or SOTA claim
follows. A train-only investigation of representation/refit instability
is more justified than another test-informed weight sweep.

### Read-only selection/refit audit

The hardware folds have identical ordered split hashes across all three
seeds. The worst linear cell contains 188 training and 21 test rows;
its raw RMSE is **54.6728**, versus **188.9567** for FE, an absolute
increase of **134.2839**. This is not a near-zero-denominator artifact.
The winning regularization is 0.1, and its winning inner plan grows
from ten additions to fifty at final discovery. Regularization 10 was
only 0.9398 RMSE worse on the inner tuning holdout, but its ten-feature
trial plan shared only two names with the winner's trial plan.

The same official fold at seeds 42 and 137 instead selects regularization
10, ends with twenty-five additions and produces identical final
prediction hashes with RMSE 35.4740. The seed-2029 final fifty-feature
plan looked favorable on its own train-only internal validation:
44.2729 mean RMSE versus raw 59.2539. Outer failure is therefore not
explained by ignoring an obviously negative saved inner mean.

Excessive width alone is not a sufficient explanation. The other
linear tail, seed-137/fold-3, **shrinks** from ten additions to five;
RMSE worsens from 60.4931 to 79.5255 while MAE improves from 42.5859
to 37.0493. Across thirty hardware linear-FE cells, 28 change ordered
selected names between winning inner fit and final discovery, and 20
change width, including successful cells.

These saved artifacts establish representation/refit instability, not
its numerical cause. They do not retain coefficients, scaled-design
conditioning, leverage or clipping-hit diagnostics. The next bounded
audit therefore instruments the unchanged procedure using **only
outer-TRAIN data**, across all ten hardware folds and all three existing
seeds, rather than tuning a rule on the selected bad test folds.

## Integrity and resources

All prior source-archive members, dependency versions and prepared data
bytes match the seed-42 study. The only new source-fingerprint member
is the prospective replication descriptor. Source remained unchanged
throughout all four stages:
`6e6815ee2dbd45d6179405ac5add14c5cabdb7725a68b9d50506a530b6283e5d`.

Verification reconstructed every IID and cached official split, checked
all 340 distinct requested identities and hash seeds, and reconstructed
all ordered result/source/manifest archives. There were no retries,
missing cells or fallback scores. All 136 raw/FE component pairs match
their independently fitted controls on every registered prediction hash,
feature-name hash, parameter, inner loss, width and metric.

Both methods retain identical train-only numeric support caps. The
average charges two complete procedures and keeps separate feature
plans; reported widths are not a union. Within-worker warmup differs
from separate-process controls, so component-stage timings do not
establish an algorithmic speedup.

The complete run used **20,046.4278946 seconds (5.56845 hours)**,
including startup, reference verification, preparation, workers and
initial publication. It remained below the seven-hour ceiling.
Current-cycle use is **69.96195 of 72 hours**, leaving **2.03805 hours**.
No LLM calls were made; the logical-attempt count remains 96/100.

- [analysis.json](analysis.json): full absolute family losses, seed-wise
  and pooled gains, exact parity checks, gates and fold tails.
- [analyze_study.py](analyze_study.py): reproducible verification.
- [protocol.json](protocol.json): unchanged pre-result policy and criteria.
- [controller.json](controller.json) and [launcher_timing.json](launcher_timing.json):
  source continuity, stage ceilings and exact resource accounting.
- [archive.json](archive.json) and [auxiliary_archive.json](auxiliary_archive.json):
  ordered, lossless bounded evidence.

Energy retains its simulation designation, and houses/wine retain
cached-only `legacy-not-reviewed` rights labels and original target
representations. No clinical, prospective or entity-disjoint infrared
claim is authorized. Prior confirmations remain unchanged.

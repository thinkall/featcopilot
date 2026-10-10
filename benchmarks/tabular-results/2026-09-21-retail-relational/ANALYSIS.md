# Point-in-time retail development evaluation

**Expanded historical aggregates improved both tested learners in this small
retail development task.** All 12 cells completed successfully.

## Task and provenance

The source is [UCI Online Retail, dataset 352](https://archive.ics.uci.edu/dataset/352/online+retail),
by Daqing Chen, under the official page's CC BY 4.0 notice.
Source bytes, schema, rights evidence and hashes were recorded. Raw transactions
remain in the ignored cache, not in these reports.

The fixed cohort contains 200 customers selected only from December 2010
history, before the evaluation periods. Forty-eight weekly cutoffs produce
9,600 queries; customers with zero future spend remain in the cohort.
The target is signed net line-item spend over the following seven days.
Refunds are retained. Missing-customer and invalid-price rows were removed
explicitly, while 5,225 exact duplicates were retained and documented.

Features use only history available before each cutoff. Outer and inner
forward splits purge labels that are not available before validation starts.
Equal-time/future records cannot enter point-in-time aggregates.

## Comparison

The six-feature simple-history baseline was compared with a fixed 22-feature
7/30/90-day aggregation library, using identical train-only model tuning.
This is **not** a comparison against an identifiers-only weak baseline.

| Learner | Simple-history mean RMSE | Expanded-history mean RMSE | Relative RMSE reduction |
|---|---|---|---|
| Linear | 431.74 | 408.57 | +5.37% |
| LightGBM | 471.96 | 399.41 | +15.37% |

The run used three purged forward folds and seed 42. Cumulative supervised
preparation/evaluation wall time was approximately **195 seconds**, below its
7,000-second ceiling. The recorded CPU affinity and thread policy were limited
to eight cores/threads; failures were not replaced with baseline scores.

## Limitations

Invoice generation time is assumed to equal information availability. This is
a same-customer future-period net-spend proxy, not unseen-customer transfer,
an official RelBench task, automatic multi-hop feature discovery, or a causal
treatment-effect estimate. One small cohort/seed is insufficient for broad
relational SOTA claims. Retained duplicates and heavy-tailed spend also limit
generalization.

The registration hash and all 12 successful, uniquely identified cells were
verified. See [protocol.json](protocol.json), [registration.json](registration.json)
and [report.json](report.json). The frozen runtime source is also retained in
the preceding [confirmation-completion source archive](../2026-09-21-confirmation-completion/source.zip);
both stages ran sequentially without source changes.

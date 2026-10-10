# Unary-frequency development ablation

**Frequency candidates improve linear classification and boosted regression
means, but do not improve every task/learner stratum or eliminate harmful
regressions. The feature remains opt-in; no default promotion or SOTA claim.**

All **384 cells succeeded in 8.03297 active hours** on the original sixteen
development families. Exact split/dependency identities match the prior
joint-importance experiment. All **288 control losses match that archived
run exactly**, despite the subsequent output-preserving LogisticRegression
worker optimization. This isolates the new named frequency variant rather
than silently replacing controls or historical results.

## Observational primary-loss reductions

Positive percentages mean lower log loss/RMSE than the matched raw learner.
Classification covers nine families; regression covers six.

| Task / learner | Importance-paired | Frequency-paired | OpenFE-inductive |
|---|---|---|---|
| Classification / linear | +12.38% | **+15.27%** | +11.24% |
| Classification / LightGBM | +8.58% | +8.31% | +9.71% |
| Regression / linear | **+13.37%** | +12.83% | +8.28% |
| Regression / LightGBM | +2.06% | **+2.89%** | +2.56% |

Frequency-paired medians in the same row order are **+2.11%, 0.00%,
+8.77%, and +2.33%**. Classification W/T/L is **7/2/0** for linear and
**2/6/1** for LightGBM, at the predeclared 1% tolerance. The boosted
classification mean remains below OpenFE and its median is still zero.
These repeatedly examined development families are not independent evidence.

Energy Efficiency remains a separate simulation: frequency-paired improves
RMSE by +82.07% with linear and +5.93% with LightGBM, versus +76.99%/+5.91%
for importance-paired and +43.87%/+8.05% for OpenFE. Do not fold this result
into the observational means.

## Useful changes and remaining harm

| Family / learner | Importance-paired reduction | Frequency-paired reduction |
|---|---|---|
| Chronic Kidney Disease / linear | -9.85% | +23.38% |
| Combined Cycle Power / LightGBM | +1.71% | +8.26% |
| Wisconsin Diagnostic / linear | -3.83% | 0.00% |
| Wisconsin Diagnostic / LightGBM | **-22.00%** | **-22.76%** |
| Concrete / LightGBM | -0.40% | -3.37% |
| Spambase / linear | +18.41% | +14.10% |
| Banknote / linear | +96.11% | +88.34% |

The low-loss Banknote comparison needs absolute context: raw log loss is
**0.029459**, importance-paired **0.001147**, and frequency-paired
**0.003435**. It remains much better than raw, but roughly triples loss
relative to the previous candidate. The improved classification mean cannot
erase individual tradeoffs. Wisconsin's substantial boosted harm remains
unresolved.

## Actual use, cost and fitting boundaries

At least one frequency feature is delivered in **31 of 96 outer fits**;
65 deliver none. Repeat-specific mappings use only eligible training rows.
Final training refits can produce constant/duplicate frequencies; removals
and the distinction between validated and delivered feature counts are
retained in the selection diagnostics. No validation score is fabricated
for a reduced subset.

This ablation reserves at most 32 candidates and at most 10% of the candidate
budget. It can displace the tail of a full base pool and change joint rankings
or subset choice even when a final fit retains no frequency feature.
Consequently, these comparisons do not isolate the causal contribution of
each retained frequency column.

Mean pipeline seconds per family/cell:

| Task / learner | Importance-paired | Frequency-paired | OpenFE-inductive |
|---|---|---|---|
| Classification / linear | 10.00 | 13.11 | 156.93 |
| Classification / LightGBM | 30.65 | 34.03 | 157.15 |
| Regression / linear | 3.04 | 4.58 | 140.08 |
| Regression / LightGBM | 10.89 | 12.92 | 142.89 |

These timings exclude some process/supervisor overhead and are specific to
the pinned environment. References here are width-matched, top-50
OpenFE-inductive, not native-width or unrestricted paper reproduction.

## Frozen next comparison

Stop selecting configurations on this panel. Freeze one uniform
**frequency-paired** candidate for the ten newly metadata-reserved origins,
subject to final exposure, rights, payload/schema and split-support admission.
Do not choose frequency only for favorable tasks or learners.

Compare against raw, OpenFE-inductive-native and AutoFeat-native with all
four supported strong-comparison learners: linear, LightGBM, CatBoost and
XGBoost. Use their existing matched three-setting grids, full rows,
seed 2027, three IID folds or the predeclared exact official split
repeat=0/fold=0/sample=0, and unchanged candidate/selection settings.
CatBoost's numeric selection versus native-categorical final-model
representation difference remains disclosed.

Ceiling: **24 active hours**, eight threads, 900 seconds/cell, 4,096 MiB
sampled process-tree RSS; no new LLM calls. Existing conservative gates
remain: positive mean versus raw in every task/learner stratum, no family
harm greater than 5%, competitiveness against complete paired references,
and complete coverage for broad claims. Independent evaluation is not an
automatic SOTA certificate, especially given the known development harms.

## Evidence

All 384 identities are unique and scored. Start/end/current source hashes
matched before later declaration work:
`f17435009ad2f106248a96c0747bf19a586ccfcb7e55bf4820886eaa9e23d865`.
Exact result/source checksum and bounded archive reconstruction checks passed.

- [REPORT.md](REPORT.md): configuration, family losses and descriptive intervals.
- [archive.json](archive.json): ordered lossless result parts.
- [auxiliary_archive.json](auxiliary_archive.json): bounded manifest/source parts.

Current-cycle dataset measurement totals **44.08445 hours**. After the
separate 0.25-hour operational allowance for synthetic timing probes,
**27.66555 hours** remain under the authorized 72-hour ceiling.

### Pre-acquisition amendment to the next experiment

The proposed official split for Vertebral Column has only six minority-class
test rows, below the locked minimum of ten. This was found from metadata,
before payload or model access. The user chose three-fold IID evaluation
for **all ten families**, preserving the class-support threshold and every
source/target/feature definition. The revised next experiment is 480 cells
under the unchanged 24-hour ceiling; the blocked proposal is retained in
its separate audit. This amendment does not alter any result above.

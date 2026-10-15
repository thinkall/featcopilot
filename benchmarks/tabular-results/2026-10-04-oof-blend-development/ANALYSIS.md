# OOF-learned raw/FE probability blending

**The learned OOF weight does not solve the external harm problem.**
Its mean gain is positive, but Wisconsin/LightGBM remains 22.92% worse
than raw. Fixed 50:50 averaging is the more promising control on the
completed families, with worst harm 0.92%, but the full-panel experiment
is incomplete and requires a separately declared replication.

The run recorded all **108 requested outcomes** from **27 physical
bundles** in **2.75401 active hours**. Twenty-five bundles succeeded
(100 scores); two Dry Bean bundles exceeded the unchanged 900-second
deadline. Each timeout leaves all four associated outcomes unscored.
No result was replaced with a baseline or silently omitted.

## Fixed procedure and exact controls

Five stratified OOF folds each fit complete raw and FE procedures,
including their own three-setting hyperparameter search and preprocessing.
One global convex weight minimizes OOF log loss, with explicit raw/FE
endpoints and raw preference for a flat objective. It is then frozen
before both complete procedures are refitted on all allowed training rows.

Raw, FE, fixed average and learned blend share those same full-training
models. All **50 successful raw/FE control losses match the archived
single-model results exactly**. OOF losses are meta-training scores,
not independent validation or a guarantee about the final refits.

## Complete-family results

All arms below use the same **eight complete families out of nine**:

| Method | Mean log-loss reduction vs raw | Worst family reduction |
|---|---|---|
| Single FE | +9.12% | **-22.76%** |
| Fixed 50:50 raw/FE average | +8.58% | **-0.92%** |
| OOF-learned blend | +9.45% | **-22.92%** |

Dry Bean is excluded only from complete-family aggregate comparisons;
its requested and failed outcomes remain in the coverage denominator.
Do not compare these eight-family means with a prior nine-family mean.

The learned blend's mean absolute log loss is **0.002464 higher** than
the fixed average, despite a +3.35% mean relative reduction when ratios
are computed separately by family. Low-loss tasks receive different
relative leverage. Both summaries are preserved; neither erases tail harm.

On Wisconsin fold 1, the OOF optimizer chooses the FE endpoint, where
its observed OOF log loss is lower than raw. Full-data refitting then
retains the known harmful external behavior. Keeping raw available in
the meta-objective does not force the algorithm to choose it before
unknown external errors occur.

## Cost and provenance

Each learned-blend bundle requires twelve complete base procedures:
ten OOF fits and two full refits, each containing three inner candidates
plus a winning refit. The fixed average needs no OOF models or optimized
weight, but this shared study charged their physical cost once to the
bundle. A direct fixed-average implementation can test that control
without repeating unnecessary OOF work.

Compact records retain all five OOF partitions, parameters, three trial
losses/widths, schema/formula/state identities, and full raw/FE plan names.
No raw rows, OOF/test prediction matrices or category vocabularies are
published. The two timeouts are resource failures, not evidence of absent
predictive value.

## Integrity and follow-up boundary

All 108 identities and 27 physical records are present. Source remained:
`dc177787d3979b876019f04902e9f36ded1ac131ae01bea23344aaebe9fc0153`.
Exact result/source checksum and bounded archive reconstruction passed.

- [REPORT.md](REPORT.md): all outcomes, costs and compact model provenance.
- [archive.json](archive.json): ordered lossless results.
- [auxiliary_archive.json](auxiliary_archive.json): bounded manifest/source.
- [decision.json](decision.json): no promotion and coverage-qualified comparisons.

Reject the learned weight as a general upgrade; do not tune its optimizer
or weight bounds to rescue these results. A separate fixed-average study
will retain the already-declared 50:50 rule, fit only the two full
procedures, and evaluate all nine families on seeds 42, 137 and 2029.
It will preserve these OOF timeouts rather than pretending to complete
the same experiment under a cheaper algorithm.

Current-cycle measurement totals **58.12757 of 72 hours**; **13.87243
hours remain**. No new data sources or LLM calls were used. This remains
exposed development evidence, not fresh confirmation or SOTA.

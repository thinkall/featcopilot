# Hardware regression: TRAIN-only numerical audit

**All 30 diagnostic jobs completed.** All sixty raw/FE component replays
exactly match their historical selected parameters, three inner losses,
final feature-name hashes and output widths. The audit instrumented
120 winning-inner/final fitted designs without changing selection rules.
It produced **no new outer-test scores**.

The largest previously observed failure coincides with a wider final
representation, weak regularization and the largest final coefficient
norm in this audit. However, counterexamples rule out treating width or
rank deficiency alone as a reliable failure detector.

## Same official fold, different final models

All three rows below use the same hardware fold-6 outer-TRAIN partition
(188 rows). Width includes the six original predictors.

| Seed | Chosen Ridge alpha | Winning inner → final width | Final design rank | Final coefficient L2 norm | Final training residual RMSE |
|---|---:|---:|---:|---:|---:|
| 42 | 10 | 16 → 31 | 31/31 | 66.34 | 26.6745 |
| 137 | 10 | 16 → 31 | 31/31 | 66.34 | 26.6745 |
| 2029 | 0.1 | 16 → 56 | **53/56** | **270.10** | **20.6312** |

The seed-2029 coefficient norm grows from 165.75 on the winning inner
fit to 270.10 on final refitting, approximately four times the final norm
at the other seeds. Its ten selected additions grow to fifty. Its lower
training residual does **not** establish better generalization.

The independently saved earlier evaluation recorded RMSE 188.9567 for
this seed-2029 FE fit versus 54.6728 for raw. Those numbers are historical
context only; this audit did not recompute or receive them in a worker.
It establishes the numerical/representation differences, not that any
one of them caused the earlier test failure.

## Counterexamples and diagnostic limits

- Seed 2029/fold 0 also ends with 56 features and a rank-deficient
  design (41/56), but uses alpha 10 and has a much smaller coefficient
  norm of 59.66. Its previously recorded test result improved over raw.
- The previously harmed seed-137/fold-3 model ends with only eleven
  features, full rank, design condition number 11.59 and alpha 10.
  Rank deficiency and excessive width therefore do not explain every
  observed failure.
- Across all thirty FE fits, **28** change their ordered selected
  feature names and **20** change width between the winning inner fit
  and final discovery. Four final designs and six winning-inner
  designs are rank deficient. These properties occur beyond the
  selected harmful cases.

The reported condition number describes the **unpenalized transformed
training design**, without an added intercept column. It is not the
condition number of Ridge's centered, regularized solve. A null value
with `rank_deficient` is not a Ridge solver failure: positive
regularization can still define a well-conditioned fit. Coefficient
norms also depend on scaling and representation, so this retrospective
contrast does not justify a numerical cutoff.

The fold-6 seed-2029 training boundary fraction is 11.06%, compared with
15.24% for the other two seeds. These are equality counts at fitted
training minima/maxima, with constant and undefined columns treated
separately. They are **not** counts of changed values or future clipping
events, and do not diagnose held-out extrapolation.

## What this changes

The benchmark now has a reusable, bounded training-only instrumentation
path that reproduces the original fitting decisions and makes
selection/refit drift measurable. It rejects mismatched replay fields
instead of reporting a successful-looking diagnostic.

The engineering conclusion is still **no default promotion**. A future
robustness experiment should validate complete representation-and-
regularization choices across training partitions, not impose a
post-hoc width, rank or coefficient threshold on these exposed failures.
Any such policy needs a prospective protocol, matched fitting-budget
controls and evidence across the full development panel before fresh
confirmation. This audit alone neither selects that policy nor proves
its effectiveness.

## Data boundary and verification

The controller extracts only the original ordered outer-TRAIN `X` and
`y` into ten checksummed payloads. Each worker receives one payload,
the seed and aggregate provenance, plus only these four historical
comparison fields: parameters, inner losses, final feature-name hash
and width. It receives no outer-test rows, labels, prediction vectors
or evaluation scores. The final transform probe is one training row
and is never scored.

Verification reconstructed all official split identities and compared
each payload exactly with the corresponding original TRAIN slice.
All sixty replay comparisons, 120 fitted-design records, clipping
count totals, source/dependency pins and ordered archives were checked.
Historical scored results, controllers and protocols remain unchanged.
There were no retries or failed jobs.

Source fingerprint:
`3ec8d9c2f8bfc02dd70ce35774d1a1cfd1dc43adca2cc793e3f8c5550a39b535`.

The run used **1,320.5106542 seconds (0.36681 hours)**, including startup,
reference verification, training extraction, workers and initial
archives. It stayed within the one-hour ceiling. Current-cycle use is
**70.32876 of 72 hours**, leaving **1.67124 hours**. No LLM calls were
made; the attempt count remains 96/100. Unused budget is not evidence
and is not spent on an unregistered threshold sweep.

- [analysis.json](analysis.json): all compact numerical diagnostics,
  integrity checks, scope limitations and resource accounting.
- [analyze_audit.py](analyze_audit.py): reproducible verification.
- [protocol.json](protocol.json): unchanged pre-result declaration.
- [controller.json](controller.json) and [launcher_timing.json](launcher_timing.json):
  training boundaries, source pins and elapsed time.
- [archive.json](archive.json) and [auxiliary_archive.json](auxiliary_archive.json):
  exact bounded diagnostic and source evidence.

These are retrospective training diagnostics on one exposed family,
not additional scored benchmark cells, fresh confirmation, a no-harm
guarantee or evidence of SOTA.

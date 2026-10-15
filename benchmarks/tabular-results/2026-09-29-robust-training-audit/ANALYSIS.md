# Robust training-only retrospective audit

**Retrospective development, not confirmation.** Each cell receives only its corresponding original
outer-training partition. Its audit holdout and nested tuning split exactly match the prior study.
The six audit partitions overlap; they are not six independent datasets.

Methods use current benchmark run_isolated/run_case without runtime modifications.
Aligned methods select among the fixed 0.1/1/10 alpha-specific pipelines using nested validation.
Robust methods use fixed TRAIN-min/max caps (0,1) under the shared finite guard.
No choice is made from audit-holdout outcomes.

| Family | Original fold | Raw RMSE | Robust raw RMSE | Aligned RMSE | Robust aligned RMSE |
|---|---:|---:|---:|---:|---:|
| computer-hardware-performance | 0 | 31.182808 | 31.182808 | 31.182808 | 29.638588 |
| computer-hardware-performance | 1 | 57.940178 | 58.518189 | 36.089309 | 47.325374 |
| computer-hardware-performance | 2 | 39.032058 | 39.032058 | 29.520326 | 32.308836 |
| infrared-oral-temperature | 0 | 0.25171067 | 0.25140148 | 0.23813339 | 0.23794734 |
| infrared-oral-temperature | 1 | 0.26159794 | 0.2615223 | 0.26159794 | 0.2604336 |
| infrared-oral-temperature | 2 | 1.5419319 | 0.26452105 | 2.1394429 | 0.25687365 |

## Coverage and interpretation

All **24 requested cells succeeded**, with no failure, timeout, memory-limit
termination or unfinished cell. The six original/audit/nested split and
isolated-data fingerprints matched the earlier studies before fitting.
All twelve nonrobust control cells reproduced prior aligned-audit losses
exactly, including chosen parameters and aligned selected names.

| Family | Mean raw RMSE | Mean robust raw | Mean aligned | Mean robust aligned | Robust FE reduction vs robust raw |
|---|---:|---:|---:|---:|---:|
| Hardware | 42.718348 | 42.911018 | 32.264148 | 36.424266 | +15.12% |
| IR-temperature | 0.685080 | 0.259148 | 0.879725 | 0.251752 | +2.85% |

The matched robust baseline is essential: IR robust raw alone improves by
62.17% relative to raw, while the additional FE reduction relative to robust
raw is 2.85%. Hardware robust FE is 12.89% worse than the old aligned FE mean,
despite improving on its own robust raw baseline. Thus this is not a uniform
improvement and does not establish no-harm, SOTA or default promotion.

For the existing IR fold-2 audit holdout, the original `x5` (Distance) support
diagnostic was computed using only that cell's audit-training rows:

- Training maximum absolute standardized value: **2.985894**.
- Audit maximum before the shared min/max clamp: **1174.086834**.
- Audit maximum after the clamp: **2.985894**, with one value above the
  training upper bound and none below the lower bound.

This uses the actual shared clipper and training-derived median/population
scale, not a new learner fit. It is an original-numeric-input diagnostic,
not fitted-model per-row contributions. No raw values, rows, category vocabulary
or prediction vectors are published.

On that audit split, raw RMSE changes from 1.541932 to robust raw 0.264521;
aligned FE changes from 2.139443 to robust aligned 0.256874. These observations
support the training-outlier/extrapolation diagnosis on this already exposed
partition. They do not isolate Distance as the sole cause: clipping also
changes other numeric inputs, internal selection, chosen alpha and the final
feature plan. No original outer-test partition was fitted or scored.

## Resources, safety and integrity

- **Complete external audit process: 911.234331 seconds (0.253121 hours)**,
  including interpreter startup, preparation, 24 requested cells, validation,
  initial summary and bounded archive writing.
- Hard whole-wall ceiling: 1,500 seconds, with 60 seconds reserved for reporting.
  Each cell used at most min(600 seconds, remaining model budget), eight native
  threads and a 4,096 MiB process-tree RSS limit.
- Supervisor time through cell execution: 896.042464 seconds; summed benchmark
  worker wall time: 725.709508 seconds; measured pipeline sum: 142.647582 seconds.
  These are nested intervals, not quantities to add to the external total.
- Four synthetic safety tests passed. Their complete process took
  **29.592559 seconds**, separate from the audit execution.
- Peak observed worker-tree RSS: **222.535 MiB**.
- Both previous audit directories remain byte-for-byte unchanged: 32 files
  in training-selection-audit and 72 in aligned-training-audit.
- Runtime source remained
  `f99759f0c84f668fddab71de44944d68f079604f719d4589946d4e190cce2f44`.
- Script SHA256:
  `5c43873b3e1193844936d50831f87b00c91727f6dc1d64ed229e8340a234ffc8`.
- Source ZIP SHA256:
  `d8243b1b81df4e6ca8cb7b0ba162a2cd1199c9d45db4f808b7f84236525b117c`.

Protocol, all 24 records, matched-robust summaries, aggregate support diagnostics,
timing records and exact bounded archives are preserved here. Each archive part
is at most 480 KiB; oversized originals remain local and ignored. No raw worker
input pickle remains. `artifact_hashes.json` contains final checksums.

No source/algorithm change, data acquisition, live LLM call, hyperparameter
search expansion, or post-outcome setting change was made. This final audit
is retrospective development only; the overlapping partitions do not create
independent confirmation. No further experiment is proposed.

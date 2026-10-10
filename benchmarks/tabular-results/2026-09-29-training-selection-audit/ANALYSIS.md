# Retrospective training-selection audit

**Retrospective development only. Each cell evaluates solely within its
corresponding original outer-training partition; its original outer-test
partition is excluded.** The three original training partitions overlap, so
these six diagnostics are not independent observations.

All six original outer-training partitions were isolated before fitting/statistics.
Audit holdout: 25%, seed 2028. Nested tuning: 25%, seed 2027; alpha grid 0.1, 1, 10.
The two engineered models use the same frozen final audit-training representation.
Their difference isolates alpha within that plan; inner/final set differences are descriptive.

| Family | Original fold | Raw tuned RMSE | Engineered alpha=1 | Engineered tuned alpha | Tuned RMSE | Inner/final additions |
|---|---:|---:|---:|---:|---:|---:|
| computer-hardware-performance | 0 | 31.1828 | 32.209 | 10 | 31.1828 | 1/0 |
| computer-hardware-performance | 1 | 57.9402 | 57.9402 | 1 | 57.9402 | 0/0 |
| computer-hardware-performance | 2 | 39.0321 | 24.802 | 10 | 26.149 | 0/10 |
| infrared-oral-temperature | 0 | 0.251711 | 0.25189 | 10 | 0.251423 | 5/1 |
| infrared-oral-temperature | 1 | 0.261598 | 0.261598 | 1 | 0.261598 | 0/0 |
| infrared-oral-temperature | 2 | 1.54193 | 1.88891 | 10 | 1.72875 | 5/5 |

## Findings and limits

All six cells succeeded. The same frozen final feature matrix and fitted
preprocessing state were used for each alpha=1 versus training-selected-alpha
comparison. Their preprocessing-state fingerprints matched exactly.

The clearest observed failure is the IR-temperature cell associated with
original fold 2. Original input `x5` (Distance) has maximum absolute standardized
value **2.9859** on audit training and **1174.0868** on the audit holdout: a
**393.21x** increase, with one audit row exceeding the training maximum.
Its maximum absolute prediction contribution is:

| Model on the same audit partition | Distance contribution max abs | Audit RMSE |
|---|---:|---:|
| Raw, training-selected alpha=10 | 19.6905 | 1.541932 |
| Engineered, alpha=1 | 24.2908 | 1.888906 |
| Same engineered plan, training-selected alpha=10 | 22.1374 | 1.728745 |

Changing alpha from 1 to the independently selected 10 reduces RMSE by
0.160161 **within this fixed plan**, but the engineered model still worsens
RMSE by 12.12% relative to the raw audit baseline. This is direct evidence
of extreme standardized-input leverage and regularization sensitivity in
this audit cell, not proof that the same row or mechanism caused the
previous confirmation failure. No raw patient values or row identifiers
are reported.

The hardware confirmation catastrophe was not reproduced on these different,
training-only audit holdouts. Two hardware cells retain originals only;
the third changes from zero inner additions to ten final additions. Both
alpha settings improve over its raw audit baseline, but alpha=10 is worse
than alpha=1 on that same fixed engineered plan (26.1490 versus 24.8020).
Thus "increase regularization" is not a universally supported remedy.

Representation changes remain descriptive: hardware additions change
1->0, 0->0, 0->10; IR changes 5->1, 0->0, 5->5. The final five IR features
in fold 2 share no names with the five used during its nested alpha tuning.
No final audit model selected a frequency feature. These observations
do not isolate the causal contribution of frequency to candidate ranking.

The parent preprocessing correction was active: predictive preparation
was asserted to use `impute=False`, leaving each engine/model to fit medians
on its own eligible training rows. This audit is **not** an old/new
imputation ablation and must not be interpreted as measuring that fix.
No alpha, feature rule, split or threshold was changed after audit outcomes.
Any subsequent algorithm changes remain retrospective development and
require a fresh independent cohort.

## Safety, resources and integrity

- All six reconstructed original split fingerprints matched the archived
  seed-2027 three-fold IID splits before fitting.
- Five synthetic safety tests passed in the pinned environment. During
  the real audit, 1,146 fit-boundary checks passed.
- Pinned runtime: sklearn 1.7.2, joblib 1.3.2, NumPy 2.4.4, pandas 2.3.3,
  SciPy 1.17.1, threadpoolctl 3.6.0; eight native threads.
- External launcher elapsed through process startup, all six child
  launches and machine-summary writing: **263.403183 seconds**
  (0.073168 hours). This is the whole-audit accounting figure.
- Child-process elapsed sum: **226.678432 seconds**. Worker-function
  elapsed sum: **88.752810 seconds**. Core six-partition fitting/scoring/
  diagnostics sum: **28.533908 seconds**. These are nested intervals,
  not quantities to add to the whole-audit figure.
- The supervisor's internal clock reports 255.027311 seconds; it excludes
  some interpreter startup/exit captured by the external launcher.
- Latest synthetic safety-test process: **24.276465 seconds**, separately
  measured and not included in the audit's 263.403183 seconds.
- No timeouts, failures or memory-limit terminations. Peak observed worker
  process-tree RSS was 247.735 MiB, below the 4,096 MiB limit.

| Family | Original fold | Whole child-process seconds |
|---|---:|---:|
| Hardware | 0 | 34.910738 |
| Hardware | 1 | 34.049377 |
| Hardware | 2 | 36.252540 |
| IR-temperature | 0 | 43.313526 |
| IR-temperature | 1 | 38.673639 |
| IR-temperature | 2 | 39.478612 |

Runtime source remained unchanged:
`8199702864295cba90f75f7742aea16f56b087b21a1cf607e2937d4e229fb0c3`.
Frozen audit script SHA256:
`8daf3dc2664b901e5cdb759b4a9797488e8621bb281bc8faf4dd2f1835ccc386`.
Frozen source ZIP SHA256:
`dffc2e41529e468ea9b51d939583bb05e1456940bc3bdf483fe7fb04e9b22180`.

Original confirmation manifest and prepared-cache checksums were unchanged.
The published JSON includes only metrics, selected-definition fingerprints,
aggregate standardized-feature/coefficient contributions, and hashed fit
row identities: no raw rows, category vocabulary, or prediction vectors.
See `launcher_timing.json` for whole-process timing and `artifact_hashes.json`
for final artifact checksums. Lossless bounded parts preserve the large
result/source files.

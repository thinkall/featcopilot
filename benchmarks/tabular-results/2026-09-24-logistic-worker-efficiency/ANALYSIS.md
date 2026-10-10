# Logistic selection-worker efficiency replay

**All 18 cells preserved their exact losses, selected features and learner
parameters. Mean measured FeatCopilot pipeline time fell from about
97-104 seconds to 2.4-7.4 seconds on the three development families.**
This is an output-preserving efficiency result, not an accuracy gain or SOTA
claim.

## Narrow change

Predictive selection previously assigned the native thread budget to
LogisticRegression's task-worker count. With sklearn 1.7.2 and joblib 1.3.2,
default LBFGS starts a loky process pool even though binary/multinomial
classification has one optimization task.

When the existing worker policy would override `n_jobs`, concrete
LogisticRegression selection clones now use one task worker for LBFGS and
Newton-CG with automatic/deprecated/multinomial mode, including versions
without `multi_class`. Native thread limits remain eight. Explicit OVR,
other solvers, subclasses, wrappers, CV and tree/custom estimators retain
their previous handling. Caller estimators and statistical settings are
unchanged. No new user option, candidate operator, subset rule, width
budget, or default strategy was introduced.

## Fixed replay and exact parity

The replay used original-development German Credit, Banknote Authentication
and Chronic Kidney Disease, raw and importance-paired representations,
linear learner, three folds, seed 42, a 5,000-row cap, 1,000 candidates,
50 additions, three stability repeats and eight threads. Dependency versions
match the archived joint-importance run exactly.

All **18/18 cells succeeded**, using **0.14573 active hours** (8.74 minutes).
Comparison with the archived pre-fix cells found **no differences** in:

- Split fingerprints, primary losses, chosen learner parameters or output width.
- Inner/outer selected feature names and counts.
- Recorded inner/outer baseline losses and subset-validation scores.
- Candidate set/count, screening rule, selection rule and ranking scope.

[parity.json](parity.json) contains per-cell checks and timing means, with
the reference result/source checksums. Numerical values were compared
exactly, not merely within a loose tolerance.

## Measured times

Means across the three folds for importance-paired:

| Family | First feature fit, before | First feature fit, after | Pipeline, before | Pipeline, after |
|---|---|---|---|---|
| Banknote | 96.58 s | 1.72 s | 97.39 s | 2.39 s |
| Chronic Kidney Disease | 98.38 s | 3.42 s | 102.09 s | 6.52 s |
| German Credit | 99.68 s | 3.84 s | 103.80 s | 7.41 s |

These are roughly **14-41 times lower measured pipeline times** on this
panel, not universal speedup estimates. Pipeline timing excludes some worker
startup/supervisor overhead; the complete 18-cell experiment still took
8.74 minutes. German Credit retained zero generated features in every fold,
confirming the old startup cost was not evidence of productive feature search.

Separate fresh-process synthetic checks preserved transformed matrices,
report numbers, selected names and predicted probabilities exactly for
binary and multiclass data. Their first feature fits fell from about
108-109 seconds to 0.25-0.29 seconds. Those small synthetic timings must
not replace the real-development measurements above.

## Provenance and limits

The initial attempt using the historical full development manifest failed
before creating a run or fitting models: unrelated families now have changed
exposure status. The
[new three-family replay suite](../../suites/efficiency_development_replay_20260924.json)
copies the selected dataset definitions unchanged. All three normalized
specifications match the originals; no historical manifest or exposure gate
was weakened.

All cell identities are unique. Start/end/current source hashes matched:
`480771e10faab397f544056737ec39a70828c4e4d1366527fd1a76d21a576ac0`.
Exact result/source checksums and bounded archive reconstruction were verified.
See [REPORT.md](REPORT.md), [archive.json](archive.json),
[auxiliary_archive.json](auxiliary_archive.json), and [parity.json](parity.json).

Current-cycle dataset measurement totals **36.05148 hours**. Synthetic
efficiency probes separately recorded **220.047783 seconds of timed fits**;
their untimed startup/check/shutdown overhead is unknown. An operational
0.25-hour allowance is reserved for those probes rather than labeling that
estimate measured time, leaving **35.69852 hours** available under the
72-hour ceiling. No LLM calls were made.

The independent confirmation's failed quality gates remain unchanged.
Faster execution does not turn those results into an accuracy improvement.

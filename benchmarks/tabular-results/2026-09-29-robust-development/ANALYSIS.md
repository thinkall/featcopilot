# Training-support clipping: matched four-arm development comparison

**Training-support caps improve the linear-model results on this development
panel, including gains beyond the identically capped raw learner. They do
not improve LightGBM scores or solve feature-selection stability. Keep the
option experimental; this is not fresh confirmation or a SOTA result.**

All **384 cells succeeded** after a source-identical completion-only resume,
using **5.51402 cumulative active hours**. All **192 uncapped control losses
match the preceding aligned-development run exactly**.

## What we learned from other packages

The intervention uses fitted training minima/maxima in original units,
followed by the existing imputer/scaler and learner. It draws on
[sklearn's separation of fitted extrema and clipping](https://github.com/scikit-learn/scikit-learn/blob/857849927da6e988d7d026b17aef214d43e5f26e/sklearn/preprocessing/_data.py#L529-L587)
and [Feature-engine's fit-on-training, transform-with-frozen-caps pattern](https://github.com/feature-engine/feature_engine/blob/d866312874a66364c6dee3c247e69476b478395c/feature_engine/outliers/base_outlier.py).
The implementation reuses FeatCopilot's existing bounds-estimation logic,
not copied upstream code.

Unlike swapping in RobustScaler, this bounds unseen marginal extremes.
Unlike changing to MinMaxScaler or QuantileTransformer, it retains the
existing scaling geometry. Unlike 1st/99th-percentile winsorization, the
fixed `(0, 1)` pair preserves ordinary finite training values.

The existing finite guard acts before fitting/clipping, so values beyond
`+/-1e30` are an explicit exception. Nonfinite values become missing.
Boolean values pass through capping and then retain numeric encoding;
categories, row identity and exported feature definitions are unchanged.
Every internal and final learner fits its own caps on its training rows,
after feature generation and before imputation/scaling.

## Matched attribution, not just improvement over a weaker baseline

The same sixteen exposed development families use three folds, seed 42,
5,000-row cap, linear/LightGBM, unchanged three-setting learner search,
1,000 candidates, 50 additions and three selection repeats.

Positive percentages mean primary-loss reduction. Classification covers
nine observational families; regression six.

| Task / learner | Uncapped FE vs raw | Capped raw vs raw | Capped FE vs raw | Capped FE vs capped raw |
|---|---|---|---|---|
| Classification / linear | +14.96% | +1.55% | +17.43% | **+16.12%** |
| Regression / linear | +12.70% | +0.10% | +15.06% | **+14.97%** |
| Classification / LightGBM | +8.12% | 0.00% | +8.12% | +8.12% |
| Regression / LightGBM | +1.39% | 0.00% | +1.39% | +1.39% |

Each column averages family-level ratios; subtracting aggregate percentages
does not reproduce a matched ratio. The last column, automatically emitted
as `matched_robust_summary`, is the appropriate incremental-FE comparison.
All four arms have complete coverage, so those means use identical families.

Every LightGBM primary loss is exactly unchanged by the cap, for both raw
and engineered representations. This is consistent with the limited value
of marginal support capping for these tree configurations, not evidence
that it will improve boosted models elsewhere.

The Energy Efficiency simulation remains separate: capped and uncapped FE
both reduce loss by +81.39% with linear and +6.32% with LightGBM.

## Benefits, harm and stability

Capped linear FE improves relative to its uncapped counterpart on several
families: CKD loss decreases 21.26%, Banknote 11.80%, wine 5.71%, houses
4.51%, and Concrete 4.17%. However, Wisconsin Diagnostic/linear becomes
2.54% worse than uncapped FE and **4.28% worse than the capped raw learner**.
The pre-existing Wisconsin/LightGBM loss increase of **22.76%** remains.

No new greater-than-5% family harm appears under the predeclared matched
comparison. But the separate stability gate does **not** pass:

| Task / learner | Uncapped mean selected-name Jaccard | Capped mean Jaccard | Selected-count standard deviation, uncapped to capped |
|---|---|---|---|
| Classification / linear | 0.460 | **0.389** | 3.85 to **6.13** |
| Regression / linear | 0.331 | **0.405** | 6.13 to **3.93** |
| Classification / LightGBM | 0.326 | 0.326 | 7.13 to 7.13 |
| Regression / LightGBM | 0.263 | 0.263 | 9.70 to 9.70 |

Jaccard averages the three fold pairs within each family, then families;
two empty sets score one. Count variability uses population standard
deviation. These name-based diagnostics do not imply identical fitted
frequency states or prove prediction stability. See [stability.json](stability.json)
for every family and exact predeclared gate outcome.

Thus the caps address a real linear-model failure mode, but do not justify
claiming generally stable selection or promoting a default.

## Cost and transparent completion

Mean measured pipeline seconds per family/cell:

| Task / learner | Uncapped FE | Capped FE |
|---|---|---|
| Classification / linear | 22.21 | 27.95 |
| Regression / linear | 7.44 | 9.92 |
| Classification / LightGBM | 55.46 | 60.32 |
| Regression / LightGBM | 23.58 | 25.70 |

The initial five-hour allowance yielded 351 successes, 32 budget skips and
one timeout shortened to 23.3 seconds by the remaining global allowance.
Completion/reporting overhead brought measured elapsed time to 5.06510
hours. Partial means had different family coverage and were not treated as
comparable performance evidence.

The exact first phase was archived before increasing **only the cumulative
operational ceiling to 5.6 hours**. All 33 unscored cells were retried;
the 351 successful records are byte-for-byte unchanged. Code, data, splits,
methods, hyperparameter grids, eight threads, 600-second per-cell deadline
and 4,096 MiB RSS limits did not change. The completion used 0.44892
additional hours. No failure or historical score was silently discarded.

## Evidence and claim boundary

Start/end/current source hashes matched:
`f99759f0c84f668fddab71de44944d68f079604f719d4589946d4e190cce2f44`.
All 384 identities are unique, all successful checkpoint records were
verified unchanged, and exact result/source reconstruction passed.

- [REPORT.md](REPORT.md): complete raw- and matched-baseline summaries.
- [archive.json](archive.json): ordered lossless final results.
- [auxiliary_archive.json](auxiliary_archive.json): bounded final manifest/source.
- [phase1_checkpoint.json](phase1_checkpoint.json): exact incomplete phase
  and documented operational completion amendment.
- [stability.json](stability.json): fold stability, matched gains and harm checks.

Training-support caps preserve bad extremes already in training, can remove
legitimate extrapolation, and cannot guarantee joint robustness or safe
symbolic arithmetic before capping. All datasets here are exposed
development data. Independent confirmation failures remain preserved.

Current-cycle measured execution totals **71.05739 hours**. After the
separate 0.25-hour operational probe allowance, **0.69261 hours remain**.
The predeclared 24-cell training-only outlier audit may use at most 1,500
seconds of that remainder. No additional LLM calls were made.

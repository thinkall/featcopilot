# Benchmarks

FeatCopilot's target is competitive, reliable automatic feature engineering,
starting with tabular data. **Current measurements are development evidence,
not a SOTA claim or a guarantee of improvement.**

## Canonical tabular evaluation

The protocol-v3 `benchmarks.tabular` runner compares explicitly selected
representations using identical outer splits and independently tuned learners.
Linear, RandomForest, LightGBM, CatBoost and XGBoost are supported; each
dataset/split/method/learner has its own recorded resource budget.

| Method | Scope |
|---|---|
| Raw-feature baseline | Identical train-only model preprocessing and tuning |
| FeatCopilot | Tabular engine with its default selection and do-no-harm heuristic |
| Predictive FeatCopilot variants | Opt-in learner-aligned feature selection; standard, paired, extended and joint-importance configurations are separately named |
| Polynomial | Degree-2 numeric features plus training-only univariate selection |
| OpenFE-stateless | Explicitly rowwise-restricted OpenFE variant; no joint train/test aggregates |
| AutoFeat | Budgeted two-step numeric generation and selection |
| OpenFE-inductive / native | Full operator search with frozen training statistics; width-matched or native output width, not an unmodified paper reproduction |
| AutoFeat-default / native | Native generation/selection and output width; declared internal memory-size hint or upstream no-hint setting, both externally resource-bounded |

The default remains a cached legacy 12-task development replay, with a
5,000-row cap. New `--suite` manifests add reviewed UCI/OpenML and approved
local sources, including a 50-origin development/stress inventory. Origins
are deduplicated and rights, exposure, schema and split policies are checked;
an inventory entry is not automatically fresh confirmation evidence.
Official/group/time tasks require uncapped rows and their declared boundaries.
Synthetic diagnostics and domain simulations are never merged into
observational headline results.

## Why this protocol is different

- Feature engineering, imputation and model tuning stay within training folds.
- Hyperparameters are chosen on inner validation; learners are **not** chosen
  using their outer-test scores.
- Classification log loss and regression RMSE are the primary losses. Secondary
  metrics, runtime, feature counts and memory estimates remain available.
- Failed/timed-out cases have no invented score. Coverage explicitly includes
  every requested case.
- Fold means are aggregated at the dataset level; descriptive bootstrap
  intervals resample datasets, not correlated folds.
- Every run records data/split/source hashes and software versions. A source
  change during evaluation invalidates the comparison.
- External worker deadlines trigger bounded, best-effort process-tree
  termination. OS-denied or incomplete cleanup is reported, not guaranteed
  to stop every child process.

## Run it

Use an isolated environment for comparator compatibility. The optional
`benchmark-tabular` extra pins pandas below 3, scikit-learn `>=1.7.2,<1.8`,
and AutoFeat `==2.1.1`. These constraints do not apply to the base FeatCopilot
install. Calibration adapters also support sklearn 1.8/1.9 in separately
managed environments, outside this reference-compatible extra.

```powershell
python -m venv .venv-benchmark
.\.venv-benchmark\Scripts\python.exe -m pip install -e ".[benchmark-tabular]"
.\.venv-benchmark\Scripts\python.exe -m benchmarks.tabular --output benchmarks\tabular-results\my-run
```

For a short functional smoke run:

```powershell
python -m benchmarks.tabular --datasets magic abalone --folds 2 --seeds 42 --max-rows 400 --output benchmarks\tabular-results\smoke
```

Each run writes `manifest.json`, `results.json`, a readable `REPORT.md`, and
worker logs. Output directories must be new unless `--resume` is explicitly
requested. Resume reuses only fingerprint-identical completed cells, preserving
previous attempts; it does not reuse arbitrary cached engineered features or
scores from another run. A nonzero exit status can mean partial failure, so
inspect the report for completed measurements and failed requests.

See the repository's
[full protocol and improvement task list](https://github.com/thinkall/featcopilot/blob/main/benchmarks/TABULAR_PROTOCOL.md)
and [benchmark instructions](https://github.com/thinkall/featcopilot/blob/main/benchmarks/README.md).

## First expanded measurement

The [2026-09-16 development analysis](https://github.com/thinkall/featcopilot/blob/main/benchmarks/tabular-results/2026-09-16/ANALYSIS.md)
covers 12 real datasets and two separate synthetic diagnostics, with three folds
and one seed. FeatCopilot completed all cases but added no features in 29 of the
36 real-data outer fits.

| Real-data primary-loss reduction vs matched raw features | FeatCopilot | OpenFE-stateless |
|---|---|---|
| Classification, LightGBM | +0.26% | +0.98% |
| Regression, LightGBM | +0.08% | +3.49% |

Each cell covers all six real datasets of its task type. These results show
remaining headroom, not superiority. AutoFeat timed out on 24 of 42 method/fold
cases; those outcomes remain explicit and unscored. The full report includes
linear/RF results, dataset-level uncertainty, time and memory estimates.

## Predictive-strategy follow-up

The [opt-in strategy follow-up](https://github.com/thinkall/featcopilot/blob/main/benchmarks/tabular-results/2026-09-16-predictive/ANALYSIS.md)
compares legacy behavior, explicit task routing and learner-aligned predictive
selection on the same 12 real development datasets:

| Mean primary-loss reduction vs matched raw features | Legacy | Predictive | OpenFE-stateless |
|---|---|---|---|
| Classification, linear | +0.55% | +6.76% | +5.21% |
| Classification, LightGBM | +0.26% | +0.49% | +0.98% |
| Regression, linear | +0.00% | +8.67% | +14.18% |
| Regression, LightGBM | +0.08% | +2.10% | +3.49% |

All FeatCopilot cases completed. These are meaningful gains, but **not SOTA**:
linear-regression RMSE increased by 32.33% on houses and 15.88% on wine.
The predictive strategy therefore remains opt-in; its next priority is robust
selection rather than automatic default promotion.

## Interpreting results

The [latest capped, aligned four-family confirmation](https://github.com/thinkall/featcopilot/blob/main/benchmarks/tabular-results/2026-09-30-robust-confirmation/ANALYSIS.md)
also fails promotion: FeatCopilot scores 45/48 requested cells, with
three CatBoost timeouts and thyroid harms of 13.76% (CatBoost) and
7.89% (XGBoost). Its small, mostly neutral results do not establish SOTA;
one regression origin cannot support broad regression claims.

The later [ten-family, four-learner confirmation](https://github.com/thinkall/featcopilot/blob/main/benchmarks/tabular-results/2026-09-28-frequency-iid-confirmation/ANALYSIS.md)
also **failed promotion** despite completing every FeatCopilot cell.
Linear-classification mean loss improved 15.41%, but linear-regression
loss worsened 26.68%, with hardware/linear RMSE up 61.25% and
temperature/linear up 45.61%. Nine family/learner comparisons exceeded
5% harm. Forty-five reference failures remain unscored and comparisons
use common complete families. Defaults are unchanged; these results are
evidence against claiming reliable general improvement or SOTA.

Follow-up training-only audits and a parameter-aligned pipeline experiment
also produced mixed outcomes. Alignment improves one diagnostic family but
worsens the other; it does not resolve extrapolation or feature-set
instability. The benchmark now defers predictive imputation until the
appropriate training split, but correcting that boundary does not
retroactively validate earlier performance claims.

The [12-family independent native-reference comparison](https://github.com/thinkall/featcopilot/blob/main/benchmarks/tabular-results/2026-09-24-frozen-native-confirmation/ANALYSIS.md)
completed every FeatCopilot cell, with +7.09% linear-classification mean loss
reduction. It nevertheless **failed the locked promotion gates**:
LightGBM classification improved by +1.13% versus OpenFE's +9.39%, while
LightGBM regression worsened by 0.95%. Eight AutoFeat resource failures remain
unscored. Default selection is unchanged, and these families are no longer
available as fresh confirmation for a revised algorithm.

The [joint-importance development comparison](https://github.com/thinkall/featcopilot/blob/main/benchmarks/tabular-results/2026-09-23-joint-importance-development/ANALYSIS.md)
completed 480 cells. Its paired variant narrowed the boosted-classification
mean-loss gap to OpenFE-inductive (+8.58% versus +9.71%) and improved linear
regression (+13.37% versus +8.28%). However, its boosted-classification median
was zero, with a 22.00% loss increase on Wisconsin Diagnostic/LightGBM.
The variant remains opt-in; these repeatedly examined development families
cannot establish independent superiority.

The [completed roadmap cycle](https://github.com/thinkall/featcopilot/blob/main/benchmarks/IMPROVEMENT_PLAN.md#evaluated-implementation-status)
adds audited confirmation and separate LLM, temporal and relational evaluations.
The six-family confirmation is complete after an explicitly recorded schema
repair; it is mostly neutral on classification, with an Airfoil-led regression
gain. It does not establish SOTA, and default promotion remains unwarranted.

Positive loss reduction means an improvement over the **same learner** on the
same outer folds. It does not mean the representation beats every other learner.
Do not rank methods measured on different complete-pair subsets. W/T/L uses a
predeclared 1% relative-loss tolerance, not a significance threshold.

The current subset uses our own CV splits; it is not a reproduction of the full
INRIA/OpenML leaderboard. A credible SOTA claim still requires wider datasets,
untouched confirmation tasks, repeated seeds, strong modern comparators,
competitive tuning, and multiple resource tiers. Temporal/grouped data need
separate split policies rather than shuffled tabular CV.

## Historical experiments

The previous simple-model, FLAML/AutoGluon, and feature-tool reports remain in
the repository for traceability. Their old "#1", composite-score, large
synthetic-gain and "do-no-harm guarantee" headlines are **retired**. They included
methodological limitations such as test-fold best-model selection, asymmetric
preprocessing, simulated competition-like datasets, and unenforced timeouts.

Do not combine those historical scores with results from the canonical
protocol or present them as current comparative evidence.

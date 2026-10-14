# Tabular benchmark protocol and improvement tasks

For the next engineering priorities and dataset-expansion recommendations, see
the [2026-09-17 improvement plan](IMPROVEMENT_PLAN.md). This file describes the
existing protocol and preserved measurements; proposed changes do not alter
those historical results.

## Implemented protocol v3

The [benchmark README](README.md#protocol-v3-catalog-admission-and-resumable-cells)
documents current execution. Version 3 adds reviewed catalogs, multiple
manifests, family/exposure gates, official and group/time split constraints,
per-learner cells, exact resume, stronger CatBoost/XGBoost baselines, and
source/result archives. Its budgets are **per learner**, so do not directly
compare v1/v2 shared-case timeout rates with v3 rates.

The preserved v1/v2 procedures and measurements below remain historical
references. The UCI development suite adds natural class imbalance, native
multiclass targets and a separately labeled simulation task; TabArena and
Seoul remain metadata inventories until their admission gates are satisfied.
Default feature engineering remains legacy, while predictive selection uses
the new stability and numerical policies only when explicitly selected.

Current development/reference evidence is linked from the
[benchmark README](README.md#measured-development-baseline), including the
16-family joint-importance comparison and its remaining harmful regressions.
Reviewed catalogs now include 50 development/stress origins, and separately
reserved confirmation families remain protected until a locked protocol
explicitly releases them. This does not supersede source-specific rights.

The methods, dataset counts and protocol descriptions in the following
v1/v2 audit sections describe those historical configurations, not every
method or admitted source now supported by v3.

### Locked mixed-source confirmation

A new confirmation registration may declare `split_plan`, keyed by every
selected origin family. An official entry names exact
`official_split_ids`, each with integer `repeat`, `fold` and `sample`;
an IID entry declares `folds`, matching the run configuration. All official
source dimensions are still acquired and validated before selecting the
registered identifiers. Missing/duplicate identifiers or positional
substitutions are rejected. IID uses the existing seeded, boundary-checked
splitter; class-support checks cover the actual selected outer folds.

The runner binds the complete family cohort, methods, learner searches,
feature-selection settings, seeds, row policy, declared memory limit,
case/total budgets and expected cell count before model execution. Source
identity binds the implementations of separately named methods; descriptive
`method_semantics` is documentation, not an executable override.
Metadata preserves all source split IDs and the selected IDs. Failed
preparation contributes the registered per-family number of unscored cells.
Historical registrations without `split_plan` retain their all-official
behavior; this does not retrospectively narrow earlier comparisons.

Selecting official repeat=0/fold=0/sample=0 is a resource-bounded protocol
choice that must precede outcomes, not a reproduction of every official
repeat. Report this scope explicitly. Active candidate reservations and
runtime family exposure remain authoritative; `--allow-confirmation` alone
cannot release a protected inventory.

An explicitly approved whole-cohort IID amendment uses
`preregistration.resplit_official_as_iid: true` with every `split_plan`
entry set to `{"policy": "iid", "folds": 3}`. Source OpenML policies,
dimensions, payload hashes and official split identities remain unchanged;
the experiment must be labeled as new IID evaluation, not official-fold
reproduction. The amendment requires uncapped rows and rejects group,
time or label-availability boundaries rather than discarding them.
Absent or false, the flag does not authorize replacement of official splits.
Actual class-support margins still apply to every newly generated fold.

## Goal and claim boundary

Measure the incremental value of **feature engineering**, holding the downstream
learner, data, outer splits, and search budget fixed. Evaluate linear models,
RandomForest, and LightGBM separately. Winning against a linear model alone does
not show superiority to strong tabular learning, and an internal "do-no-harm" gate
is a heuristic, not a statistical or production guarantee.

The current manifest is a **development subset**: six classification and six
regression tasks from a pinned INRIA-SODA CSV snapshot. It includes numeric and
mixed-type data, but not the complete suite, grouped/temporal evaluation,
very wide data, or a held-out final confirmation suite. We use our own seeded
CV splits, not the original OpenML task folds. The default uniform 5,000-row
subsample is recorded; it must not be described as a full-data result.

## Audit: why the legacy headline results were retired

| Finding | Consequence | Canonical protocol |
|---|---|---|
| Simple-model runner takes the maximum RF/linear score on each outer test fold | Test-set model-selection bias | Separate learners; choose hyperparameters on inner validation only |
| Tool-comparison runner label-encodes before splitting and gives tools different raw/encoded inputs | Asymmetric or transductive preprocessing | Common train-fitted preprocessing; originals retained for every method |
| Some runner exceptions return unchanged inputs as if FE succeeded | Inflated coverage and hidden implementation failures | Explicit failed rows without baseline scores |
| FE timeout is checked only after work finishes, and not uniformly | No actual resource limit | Kill only the isolated case process tree at the common deadline |
| Legacy feature-cache identity omits split/seed and source state | Stale, potentially incomparable results | Never cache engineered matrices or scores in this runner |
| Synthetic "Titanic"/"House Prices" simulations feature strong hand-written interactions | Not real competition-data evidence | Fixed real-data manifest; independent synthetic reporting |
| Relative R-squared changes and ad hoc weighted composite scores | Unstable or subjective headline rankings | Positive-loss reductions and coverage-qualified descriptive comparisons |
| Wilcoxon tests treat dependent folds as independent samples | Overstated significance | Average folds within datasets; dataset bootstrap intervals only |
| Public OpenFE `transform` combines training and test inputs for global operators | Not strict inductive deployment | Explicitly named rowwise-restricted OpenFE variant |

Legacy runners and caches remain for compatibility and historical inspection.
They are not inputs to the new reports.

## Data and reproducibility

- The canonical controller charges graceful preparation interruptions and
  report/archive publication to cumulative active time. Its final operational
  `run_status`, `active_seconds`, `publication_reserve_seconds` and
  `budget_overrun_seconds` are in `manifest.json`; score archives are publication
  snapshots, not a substitute for final operational status. Publication reserves
  are estimates, and an overrun is reported with a nonzero exit rather than hidden.
  Per-cell state checkpoints provide crash recovery; the full report and
  descriptive bootstrap are published at finalization, not after every cell.
  Resumes preserve completed cells and prepared provenance; a rejected identity
  does not consume the stored budget, and unchanged score evidence is not republished.
  Invalid controller protocols or malformed worker result contracts record a
  terminal `protocol_error`; resume is rejected without rewriting that checkpoint.
  Ordinary failed evaluation cells retain their explicit retry behavior.
- Worker supervision cleans up the owned process tree and input payload on
  exceptional exits as well as explicit limits. Termination waits are bounded;
  OS-denied cleanup is best-effort and reported, not a guaranteed hard kill.
  Malformed result files are protocol errors; operational monitoring failures
  remain visible retryable failures.
- New result archives use immutable generation-specific part names and switch
  the authoritative index only after the new parts are complete. Readers must
  use the index, never a glob. Older generations are retained intentionally so
  an interrupted publication or concurrent reader does not lose prior evidence.
- Actual confirmation runs transactionally admit a family before acquisition,
  preventing competing runs from both passing a separate check before opening
  its data. Reservations are conservatively retained after acquisition failures;
  they are not silently released as supposedly unexposed. The explicit
  prepare-only non-exposure policy remains unchanged.
- Source ZIP snapshots are also published atomically. A failed initialization
  records recoverable bootstrap identity/configuration and consumed time;
  `--resume` can finish that bootstrap only when the identity still matches and
  no prepared, attempted or scored evidence exists. Missing bootstrap state or
  identity drift fails explicitly rather than rebuilding a populated run.
- Dataset names, source paths, targets, categorical columns and corresponding
  transformed OpenML dataset IDs are in `tabular_suite.json`.
- Source revision: `8d0ff9103525b7e3579b180230fddb3186258301`. Data downloads fail
  explicitly; no loader substitutes synthetic data.
- Each result records downloaded-byte SHA-256, row sampling fingerprint, original
  and sampled sizes, column names/types, split fingerprints, seeds, library
  versions, OS/Python, Git commit, and current source hash (including dirty code).
- Source is hashed again at completion. A changed fingerprint invalidates the
  comparison, rather than claiming measurements represent a single revision.
  New fingerprints use POSIX relative path names, matching ZIP members and
  source allowlists, so separators alone do not change source identity.
  File bytes are still hashed exactly; historical fingerprints are not regenerated.
  Source discovery prunes generated evidence and cache directories before
  descending into them. Identity checks still rehash included bytes rather than
  relying on file size or modification time.
- New runs through the shared subprocess launcher set `PYTHONHASHSEED` to the
  cell's integer seed (0 through 4294967295) **before Python starts**, overriding
  inherited hash randomization without changing the parent environment. The
  supervisor records `worker_python_hash_seed` on successful, timed-out,
  memory-limited and failed worker outcomes; calibration/blending normalization
  preserves it and rejects mismatches. This fixes one source of string-set
  ordering variability, not all numerical or parallel nondeterminism. Historical
  records without this field have an unknown hash seed, not an inferred value.
  Replay them with their archived source; the new source-bound policy cannot
  silently replace old measurements or resume their identities.
- Run directories are new and immutable with respect to reuse. Reports are
  checkpointed after every case. Public data stay in an ignored download cache;
  worker copies of input payloads are removed on completion.
- Git preserves exact bytes for published evidence, suite JSON, the family
  registry and the two legacy `simple_models/SIMPLE_MODELS_CACHE*.json`
  descriptors referenced by exposure proofs using `.gitattributes`.
  Mutating pre-commit formatters exclude these
  frozen records (and archived runner copies); the large-file and conflict checks
  still apply. Verify ordered archive parts against their recorded SHA-256
  indexes rather than reformatting historical evidence. Active benchmark Python
  outside result directories remains subject to Black and Ruff.
- Do not redistribute dataset files without checking their original licenses.
  The HF dataset card does not supply a blanket license. Results contain metrics
  and provenance, not the underlying data.

## Evaluation boundaries

For each outer fold and method:

1. Stratified K-fold for classification; shuffled K-fold for regression.
2. Split outer training into 75% inner training and 25% inner validation.
3. Fit numeric median imputation and FE only on inner training. Transform inner
   validation without its labels. Fit model-specific preprocessing on inner
   training and evaluate the three fixed learner candidates on inner validation.
4. Independently refit FE on all outer training rows, then refit each chosen
   learner. Evaluate the untouched outer test fold once.
5. Persist every learner's scores and chosen hyperparameters. Never take the
   maximum across learners on outer-test scores.

The inner holdout selects only learner hyperparameters; FE settings are fixed
in advance. Reusing these development results to tune FeatCopilot makes the
datasets part of development, not an untouched final test set. A later SOTA
claim needs independent confirmation tasks.

### Predictive internal-imputation correction

After the ten-family frequency confirmation, a boundary audit found that
benchmark-level median filling occurred before predictive selection created
its own internal holdouts. Internal-validation feature values could therefore
influence that earlier median, although outer-test rows/labels did not.

New predictive, typed-plan and hybrid runs preserve numeric missingness
(normalizing infinities to missing) at engine entry. The engine and final
learner fit their own preprocessing on the applicable training partition.
Categorical marker normalization is unchanged and learns no distribution.
Legacy/reference preprocessing remains as declared; upstream reference
internal-validation protocols are not silently replaced.

This correction can change missing-data results and is source-fingerprinted;
historical archives are not rewritten or resumed under changed code.
An internal training partition with no usable baseline feature now reaches
the engine's explicit error rather than being disguised by early zero fills.
The correction is not evidence that imputation caused the severe hardware
regression: that dataset has no recorded missing values.

## Methods and controlled differences

All methods see identical outer rows, preserve original columns, share train-only
numeric imputation, and use identical final numeric scaling / categorical
one-hot preprocessing (`handle_unknown="ignore"`). Categorical metadata comes
from declared source schemas, not guessed test-set cardinality.

| Method | Fixed configuration |
|---|---|
| Baseline | Raw inputs with the same downstream preprocessing and model tuning |
| FeatCopilot | `AutoFeatureEngineer(engines=["tabular"])`, default selection/gate, maximum total width = originals + 50 |
| FeatCopilot-task | Same legacy configuration with explicit classification/regression routing from the dataset declaration |
| FeatCopilot-predictive | Opt-in predictive strategy, up to 1,000 candidates; learner-aligned internal selection; maximum total width = originals + 50 |
| Polynomial | Degree-2 powers/interactions of numeric columns; training-only F-test selection of at most 50 additions |
| OpenFE-stateless | OpenFE 0.0.12, order-1 numeric candidates restricted to rowwise arithmetic/transforms; up to 50 additions |
| AutoFeat | Two generation steps, three selection runs, `1/`, `log`, `sqrt`, `^2`; numeric generation; at most 50 additions |

OpenFE's frequency/group/combined operators are **not** used in this variant.
Its own per-feature predictive evaluation and stage-two ranking remain unchanged.
A pickleable adapter replaces the removed sklearn `squared=False` RMSE call with
its mathematically identical square-root-of-MSE expression. This result must
never be labeled "full OpenFE".

AutoFeat keeps the first 50 additions in its returned selected-column order.
Its generation and selection budgets are documented deviations from unrestricted
defaults. Both comparator adapters retain original categorical columns for the
common learner, but do not synthesize categorical interactions.

Classifier targets use a fixed label vocabulary only for metric/class alignment;
no feature statistics or target means are learned across outer folds.
Integer-valued regression targets are deliberately not disguised as continuous:
if FeatCopilot's automatic task heuristic chooses classification, its behavior
and logged gate fallback remain part of the `featcopilot` measurement.
The explicit-task and predictive variants use the declared task instead.

Protocol version 2 adds these variants; it does not rewrite the version-1
baseline. The predictive method fits a separate representation per learner,
using the bare learner's fixed middle search candidate as its selection
estimator. Its internal holdout is entirely within the current fit data.
Final model hyperparameters are still selected on the benchmark's inner
validation data, never the outer test fold. Its extra feature-selection work
counts against the same method/fold deadline.

### Learner search

| Learner | Search (same three candidates for every method) |
|---|---|
| Linear | Logistic regression C / Ridge alpha: 0.1, 1, 10; logistic max_iter=2000 |
| RandomForest | 100 trees; max_depth: 8, 16, unlimited |
| LightGBM | 300 trees, learning_rate=0.05, L2=1; num_leaves: 7, 15, 31 |

These are equal-search **budgeted configurations**, not exhaustive AutoML or
best-possible comparator tuning. Report FE costs separately from model search
and fitting. The case-wide timeout includes both FE refits and all three learners,
so failed cases make no claim about unfinished learners' scores.

## Metrics, aggregation and reliability

- Primary: **log loss** for classification, **RMSE** for regression (lower is
  better). Also record accuracy, balanced accuracy, ROC-AUC, MAE and R-squared
  where applicable.
- Pair on dataset, learner, seed, fold **and split fingerprint**.
- A dataset enters a method/baseline paired summary only when all requested
  folds/seeds succeeded on both sides. Display complete-pair coverage and all
  requested failures. Never fill a failed score with a baseline score.
- First average primary losses within each dataset. Report
  `1 - mean(method loss) / mean(baseline loss)` across datasets. Keep task types
  and synthetic diagnostics separate. Baseline loss zero is excluded.
- W/T/L uses an a priori 1% relative-loss tolerance. It is not a significance
  test. Bootstrap datasets with seed 2026 (5,000 draws) for descriptive 95%
  intervals, not folds; intervals are omitted with fewer than two datasets.
  There is no multiplicity-adjusted superiority claim.
- Do not compare method rankings across different complete-pair subsets.
  Small-suite CIs and repeated-fold dependence do not establish equivalence or
  a no-harm guarantee.
- Workers execute sequentially. All inherit bounded native/joblib threads and
  explicit learner thread limits. No LLM, GPU, paid API, or external service is
  used for feature generation.
- Timeouts terminate the worker and its descendants, not unrelated processes.
  Sampled process-tree RSS includes shared pages more than once and is only a
  conservative memory estimate, not private memory or total system load.

## Task list toward competitive tabular FE

- [x] Clear the full package mypy baseline and add a CI gate.
- [x] Introduce a fixed, provenance-recorded real-data development manifest.
- [x] Add nested training boundaries, matched tuned learners and a polynomial control.
- [x] Add functional OpenFE/AutoFeat adapters with honest variant labels.
- [x] Enforce worker time/memory limits and retain failed requests in coverage.
- [x] Test leakage boundaries, split pairing, source separation and timeouts.
- [x] Use measured per-dataset failures/regressions to prioritize algorithm work.
- [x] Add an explicit task override so integer regression targets cannot select classifiers.
- [ ] Compare no-selection, selection-only and gate-enabled ablations using development folds.
- [x] Add an opt-in selection strategy aligned with the intended learner and primary loss.
- [ ] Diagnose and constrain the new linear-regression regressions with training-only stability checks.
- [ ] Benchmark column-order sensitivity, generation caps and redundant candidates.
- [ ] Add fully inductive learned categorical/group mappings and unrestricted comparator tracks.
- [ ] Expand to full OpenML suites, task-provided folds, multiple seeds, and larger-data/resource tiers.
- [ ] Reserve untouched confirmation datasets before tuning; review newer FE methods and tuned CatBoost/XGBoost/AutoML baselines.
- [ ] Extend separate protocols to temporal/grouped, relational and text tasks only after tabular evidence is stable.

Changes motivated by these measurements should get a new source hash and a new
run. Do not retune repeatedly on this report and call the best observed result
an unbiased estimate.

## First expanded measurement

The [2026-09-16 analysis](tabular-results/2026-09-16/ANALYSIS.md) records a completed
three-fold, one-seed measurement across all 12 real datasets and two separate
synthetic diagnostics. FeatCopilot completed all cases but emitted no additions
in 29/36 real-data outer fits. Its mean LightGBM loss reductions were +0.26%
classification and +0.08% regression, compared with +0.98% and +3.49% for
OpenFE-stateless on the same six datasets per task. This establishes a candid
development baseline and identifies task inference, selection/gate ablations
and learner-aligned scoring as priorities; it does not establish SOTA.

## Predictive-strategy follow-up

The [protocol-v2 follow-up](tabular-results/2026-09-16-predictive/ANALYSIS.md)
completed the same 12 real tasks plus diagnostics, comparing legacy,
explicit-task and learner-aligned predictive selection. Mean linear
classification loss reduction increased from +0.55% to +6.76%, and LightGBM
regression from +0.08% to +2.10%. All FeatCopilot cases completed. Substantial
linear-regression losses on houses and wine remain visible, and the strategy
stays opt-in. The next priority is robust selection, not default promotion.

## Primary references

- [INRIA dataset card and provenance](https://huggingface.co/datasets/inria-soda/tabular-benchmark/blob/8d0ff9103525b7e3579b180230fddb3186258301/README.md)
- [Grinsztajn et al., NeurIPS 2022](https://arxiv.org/abs/2207.08815);
  OpenML suites [337](https://www.openml.org/search?type=study&id=337) / [336](https://www.openml.org/search?type=study&id=336)
  (numerical), 334 / 335 (mixed)
- [OpenFE, ICML 2023](https://proceedings.mlr.press/v202/zhang23ay.html);
  [global transform implementation](https://github.com/IIIS-Li-Group/OpenFE/blob/master/openfe/utils.py)
- [AutoFeat implementation](https://github.com/cod3licious/autofeat)
- [Cawley and Talbot: selection bias](https://jmlr.org/papers/v11/cawley10a.html)
- [Demsar: comparisons across datasets](https://jmlr.org/papers/v7/demsar06a.html)
- [AMLB: whole-system benchmark methodology](https://jmlr.org/papers/v25/22-0493.html)

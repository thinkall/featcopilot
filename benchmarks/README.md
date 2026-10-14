# FeatCopilot Benchmarks

## Canonical tabular benchmark

Use `python -m benchmarks.tabular` for new comparisons. The old scripts below are
retained for compatibility and exploratory work, not for SOTA claims.

### Prediction-ensemble calibration study

`benchmarks.calibration_evaluation` is a separate classification-development
protocol: raw/FE crossed with single models, plain three-fold prediction
ensembles, and sigmoid-calibrated ensembles. It is not a new default feature
transformer or an implicit extension of earlier confirmation runs.

Each retained base procedure performs its entire three-setting search,
feature discovery and preprocessing inside its own training fold.
Calibration rows cannot affect those steps. The plain and calibrated
prediction views use the **same fitted base objects**, while each child
keeps its own feature plan and mapping state. There is no feature union
pretending to be one deployable schema, no model refit on all data after
calibration, and no calibration-method or threshold search.

Single controls are independently supervised jobs. Plain/calibrated views
share one physical bundle; training time is counted once toward the run
budget, but standalone method-cost estimates include the shared base cost
for each view. A failed bundle leaves both paired views explicitly unscored.
Reports distinguish original-raw, architecture-matched FE, ensemble, and
same-base calibration contrasts using complete matching folds/families.
`brier_sum` always sums squared error over all class coordinates, including
both binary coordinates; it is diagnostic, not a replacement primary metric.

The `benchmark-tabular` extra installs scikit-learn `>=1.7.2,<1.8` and
AutoFeat `==2.1.1`, matching the calibration adapter's floor and native
reference constraints. The retained-base adapter also supports sklearn
1.8/1.9 in separately managed environments, not the AutoFeat-compatible
extra. The safe-scale diagnostic retains its additional exact runtime pins.

```powershell
python -m benchmarks.calibration_evaluation --prepare-only --output benchmarks\tabular-results\calibration-study
python -m benchmarks.calibration_evaluation --resume --output benchmarks\tabular-results\calibration-study
```

Defaults select the nine original development classification families,
linear/LightGBM, three folds, seed 42, 5,000 rows, eight threads, 900 seconds
per physical job and a twelve-hour cumulative ceiling. They produce 216
physical jobs and 324 prediction outcomes. Feature budgets are fixed at
1,000 candidates and 50 additions for this study; incompatible overrides
fail before acquisition. Preparation fits no models. Exact source/data/
dependency/configuration identity is required for resume, and completed
single jobs or paired bundles are not silently rerun.

This first harness is restricted to those nine reviewed families and their
canonical source/schema definitions. Explicit subsets are allowed for
pre-measurement smoke work; a different suite cannot introduce new families
or altered definitions under familiar names. Source-payload checksum pins
are revalidated on resume in addition to prepared artifacts.

An explicitly locked `--allow-confirmation` path supports only the separate
Mammographic Mass/Haberman standard-IID cohort. It requires the complete
registered methods, source fingerprints, `subsample85` mode, restricted
claim metadata and unchanged class-support margins before acquisition.
Every output retains `standard_iid_unverified_entity`: it cannot be pooled
with verified-entity evidence or described as clinical/patient-independent
generalization. The additional native-reference arms remain single-model,
native-width, robust-preprocessed comparisons, not ensemble reproductions.

Qualified runs additionally emit an authoritative `confirmation_decision`.
It retains both registered families and all four policy comparators, with
family-level absolute losses, complete-fold coverage, harm and native-
reference gates. Missing/failed outcomes, nonfinite metrics and undefined
relative gains against zero-loss references block acceptance instead of
shrinking the decision denominator. Even a scoped pass permits no automatic
default promotion or general SOTA/clinical claim.

The [two-source qualified result](tabular-results/2026-10-03-qualified-policy-confirmation/ANALYSIS.md)
completed all 96 outcomes. LightGBM's fixed FE85 policy passed its scoped
comparisons, including +2.25% versus raw single and +0.22% versus raw85.
The overall locked decision still **fails** because linear single FE
trails native AutoFeat by 1.28% on the two-family mean. Tier restrictions
and defaults remain unchanged; no diagnostic method is substituted afterward.

The [first complete six-arm result](tabular-results/2026-10-01-calibration-development/ANALYSIS.md)
scores all 324 outcomes. Plain LightGBM FE averaging gains 8.76% over its
matched raw ensemble, with worst family harm 0.40%; linear averaging
instead harms CKD by 27.98% against its matched control. Sigmoid worsens
mean log loss and Brier sum in both representations. The fixed follow-up
keeps single FE for linear and plain FE averaging for LightGBM; additional
seeds remain development evidence, not default promotion.

The [fixed-policy replication](tabular-results/2026-10-01-calibration-replication/ANALYSIS.md)
completed all 648 outcomes on seeds 137/2029. Linear single FE gains 18.06%
with worst family harm 1.00%; LightGBM plain FE averaging gains 6.89%
versus matched raw averaging with worst harm 0.59%. However, it is 23.66%
worse on CKD than the existing single raw model. The stronger-baseline
tail criterion therefore remains unmet, without dataset-specific switches.

`--ensemble-sampling subsample85` is a separately labeled development
hypothesis: three fixed stratified 85% training bags with complementary
15% calibration holdouts. The default `cv3` retains the original partition
design. In `subsample85`, holdouts can overlap or leave rows uncovered;
reports retain actual fractions, membership hashes, coverage counts and
pairwise overlaps rather than claiming pooled OOF coverage. Each base still
performs all tuning and feature fitting inside its own training side.
Sampling mode is part of resume identity and every result row, and the
controller rejects a mode change or mismatched worker response.

The [three-seed higher-retention study](tabular-results/2026-10-03-subsample85-development/ANALYSIS.md)
completed all 486 outcomes. LightGBM FE85 gains 7.60% versus raw85 and
7.31% versus single raw across all seeds, but CKD still has individual
seed harms up to 28.97% versus raw85. The combined worst-family value
does not imply every split is safe; the option remains experimental.

### OOF-trained raw/FE probability blend

`benchmarks.blending_evaluation` is a separate, fixed classification
development experiment. Five fold-local raw/FE learning procedures produce
OOF probabilities; one convex weight minimizes OOF log loss, including raw
and FE endpoints. All HPO, feature discovery and preprocessing stay inside
each OOF training complement. The weight is then frozen while both complete
procedures are refitted on all allowed training rows.

Raw, FE, fixed 50:50 average and learned blend share those exact full-training
models. Physical training is counted once, with OOF work included in the
learned blend's standalone cost. A failed bundle leaves all four outcomes
explicitly unscored. The OOF optimum is not independent validation, and
including a raw endpoint does not guarantee raw-or-better outer-test loss.
This protocol keeps one-hot encoding and is distinct from sigmoid
calibration or prediction bagging.

```powershell
python -m benchmarks.blending_evaluation --prepare-only --output benchmarks\tabular-results\blend-study
python -m benchmarks.blending_evaluation --resume --output benchmarks\tabular-results\blend-study
```

The fixed default is LightGBM, nine original classification families,
three outer folds and seed 42: 27 shared physical jobs and 108 outcomes.
Its six-hour cumulative ceiling, eight threads, 900-second physical-job
limit and 4 GiB RSS cap include OOF construction and final refits.
Preparation fits no models; successful jobs are reusable only under
unchanged source, data, dependencies and semantic configuration.

The [first OOF blend study](tabular-results/2026-10-04-oof-blend-development/ANALYSIS.md)
scored 100/108 outcomes; two Dry Bean bundles timed out. On eight complete
families, the learned blend gained 9.45% on average but retained 22.92%
worst-family harm. Fixed 50:50 averaging had 8.58% mean gain and 0.92%
worst harm. These are matched incomplete-panel comparisons, not grounds
for promoting the learned weight or silently dropping the failed family.

The explicit `--blend-mode fixed_average` follow-up fits only the two
complete full-training raw/FE procedures and emits three shared views:
raw, FE, and their prespecified 50:50 probability average. It performs no
OOF fits, optimization or calibration. Use `--seeds 42 137 2029` for the
fixed multi-seed replication; `--seed` and `--seeds` are mutually exclusive.
Mode and seed list are part of resume identity, so this cheaper protocol
cannot silently resume or replace the original OOF experiment.

The [complete three-seed replication](tabular-results/2026-10-04-fixed-average-development/ANALYSIS.md)
scored all 243 outcomes. Fixed averaging gained 6.82% versus raw overall,
with worst family harm below 1% at each declared seed. All control
parameters/plans match prior runs; tiny multiclass metric-roundoff
differences are retained explicitly. This is promising development
evidence, not SOTA or automatic default promotion.

An explicit locked `--allow-confirmation` path supports only the separately
reserved Thoracic Surgery source and qualified standard-IID claim tier.
It evaluates the fixed average, raw/FE components and two native robust
references in nine physical jobs/fifteen outcomes. The source pin, full
cohort, exact methods, seed, class-support margins and claim boundary must
match before acquisition. Zero-reference or incomplete comparisons block
the authoritative decision; even a scoped pass grants no clinical,
entity-independent, broad SOTA or automatic default-promotion claim.

The [Thoracic result](tabular-results/2026-10-04-thoracic-fixed-average-confirmation/ANALYSIS.md)
is **blocked**, not passed: fixed averaging slightly improves all-fold
log loss versus raw (0.508215 versus 0.510180), but one AutoFeat fold
fails with an upstream NaN-input error. The missing native comparison is
not replaced by its successful-fold average, and the qualified single-
origin result does not justify changing defaults.

The separately named [AutoFeat safe-scale diagnostic](tabular-results/2026-10-04-autofeat-safe-scale-diagnostic/ANALYSIS.md)
completed all three archived folds, including the native failure. Its
mean log loss is 0.503197, better than the fixed average's 0.508215.
Previously successful native folds also differ in scores and selected
widths, so output parity is not claimed. These retrospective results
neither replace the original native outcomes nor unblock confirmation.
The follow-up audit found no invocation mismatch, but identified
upstream hash-order-sensitive selection and an unrecorded historical
Python hash seed. New shared workers pin that seed before startup and
record it in outcomes. This removes one reproducibility weakness; it
does not establish the cause of the historical score differences.

### Experimental fixed averaging for regression

`featcopilot-regression-average-robust` combines predictions from two
complete, independently tuned procedures: `baseline-robust` and
`featcopilot-frequency-aligned-robust`, at fixed weights of 0.5 each.
It averages predictions before scoring RMSE, not the components' scores.
Both procedures retain their own fitted preprocessing and feature plans;
there is no shared feature union, optimized mixture weight or new public
regression API. Runtime includes both procedures.

This benchmark-only method supports single-output regression with linear
or LightGBM learners, no group/time metadata, identity targets and
one-hot categorical preprocessing. It preserves the matched train-only
numeric caps. Final component prediction and feature-name hashes,
parameters and inner losses allow comparison with independently executed
controls; actual test predictions are kept in memory, not published.

The [fixed development protocol](suites/regression_average_protocol_20261008.json)
declares 204 cells under one four-hour ceiling. Seven primary families
are reported separately from infrared and hardware stress cases.
Hardware retains all ten official folds without a row cap; it is not a
replay of the historical three-fold IID result. All origins are exposed.
Complete coverage, matched-control parity, mean gains and worst-family
harms determine eligibility for further study, not automatic promotion.

The [completed regression study](tabular-results/2026-10-08-regression-average-development/ANALYSIS.md)
scored all 204 cells with exact component/control parity. LightGBM
averaging improves primary mean RMSE by 3.36% versus raw and 1.20%
versus single FE. For linear regression, single FE is better on every
family mean. A hardware LightGBM fold still suffers 8.97% harm with
averaging, despite favorable family means. The next candidate is
therefore a fixed learner-specific policy, not universal averaging or
a no-harm claim.

The [fixed-policy replication protocol](suites/regression_policy_replication_20261008.json)
adds seeds 137 and 2029 without changing prior model/data source members.
It declares 340 cells under one seven-hour maximum: linear raw/single-FE
controls, and LightGBM raw/single-FE/fixed-average comparisons. Each
seed and the pooled result must satisfy the family-level follow-up
criteria; the same exposed primary/stress scopes remain separate.

The [completed replication](tabular-results/2026-10-08-regression-policy-replication/ANALYSIS.md)
scored all 340 cells with exact component/control parity. Across the
two replication seeds, linear single FE reduces primary mean RMSE by
23.79% versus raw; LightGBM averaging reduces it by 2.97% versus raw
and 1.45% versus single FE. Family-mean gates pass, but hardware has
individual-fold harms of **245.61% for linear FE** and **15.63% for
LightGBM averaging**. These exposed-data results do not resolve the
robust-selection blocker or authorize default promotion.

The [TRAIN-only numerical audit](tabular-results/2026-10-08-regression-training-audit/ANALYSIS.md)
completed all 30 hardware fold/seed jobs and reproduced sixty raw/FE
component decisions exactly on the registered fields. The worst
historical linear failure has a 16-to-56-feature refit expansion,
alpha 0.1 and coefficient norm 270.10, versus 66.34 on the same fold
at the other seeds. But rank-deficient successful fits and full-rank
harmful fits rule out a simple width/conditioning safety rule.
`benchmarks.regression_training_audit` provides isolated training-only
instrumentation; its diagnostics are not outer-test scores.

### Protocol v3: catalog admission and resumable cells

```powershell
# Metadata only: pending inventories are inspectable but cannot acquire data.
python -m benchmarks.tabular --suite benchmarks\suites\uci_development.json --list-datasets
python -m benchmarks.tabular --suite benchmarks\suites\tabarena_v01_inventory.json --list-datasets

# Admitted development run: naturally imbalanced/multiclass and regression tasks.
python -m benchmarks.tabular --suite benchmarks\suites\uci_development.json --methods baseline featcopilot-predictive polynomial openfe-stateless --learners linear lightgbm catboost xgboost --folds 3 --seeds 42 137 --threads 8 --case-timeout 600 --total-seconds 259200 --output benchmarks\tabular-results\uci-development

# Resume matching cells; successful learner results are not rerun.
python -m benchmarks.tabular --output benchmarks\tabular-results\uci-development --resume
# Explicitly retry failures without changing data/code/model budgets.
python -m benchmarks.tabular --output benchmarks\tabular-results\uci-development --resume --retry-failed
```

The 72-hour/eight-thread command is an example of the approved development
ceiling, not a default or a prediction of runtime. Use smaller settings for smoke
work. Default experiment controls remain two threads and four hours.

`--prepare-only` admits, acquires and validates data/splits without fitting
models. It can be followed by `--resume` under unchanged source and semantic
configuration. `--list-datasets` does not acquire data.

`--suite` accepts multiple manifests, allowing admitted UCI data and cached
legacy regression diagnostics in one bounded run. Dataset names and origin
families must remain unique across the selected primary cohort.

The catalog supports pinned HF CSVs, reviewed UCI schemas, official OpenML
tasks, and approved local CSV/Kaggle files. `families.json` records canonical
origins, aliases and historical exposure. Pending/restricted sources cannot
download, and confirmation requires explicit admission plus
`--allow-confirmation`. Runtime family exposure is recorded separately; a
previously evaluated family cannot silently become a new confirmation family.

`suites/development_expansion_20260922.json` contains 50 admitted
development/stress origins, with source/rights/schema and CV-support evidence
in its companion audit. These are not 50 fresh confirmation datasets, nor
are all of them appropriate for one shuffled-IID scoreboard. Official,
grouped, temporal, rare-class and homogeneous-feature stress tasks retain their
declared limitations. The previously evaluated six-family confirmation is exposed.

Raw caches are versioned by normalized source/parser identity under each
dataset name. Compatible historical bytes may be reused only after actual
checksum verification; new manifest versions never overwrite historical pins,
and corruption in a namespace fails closed.

Predictive, typed-plan and hybrid methods now defer numeric imputation to
their own training-local preprocessing, preserving missingness before
internal holdouts. Earlier archived measurements used a benchmark-level
median before those nested splits; the correction is documented in the
[protocol](TABULAR_PROTOCOL.md#predictive-internal-imputation-correction).
It does not retroactively alter failed confirmation outcomes.

The old `tabular_suite.json` is preserved as a **local cached-data replay**
interface, with rights marked unreviewed. It cannot acquire new data through
the v3 driver and must never be treated as a clean confirmation manifest.

Every worker now evaluates **one dataset/split/method/learner cell**. Its
timeout covers FE, tuning, refitting and prediction. An unsuccessful learner
does not discard another learner's completed result. OpenML official task
indices and group/time sources require uncapped data; infeasible requested
outer-fold counts fail rather than silently shrinking the protocol.

Resume checks source, suite, dependency versions, configuration, dataset and
split identity. The total active-time budget is cumulative across resumes;
only that operational ceiling may be increased without defining a new
experiment. Changing threads, memory/case limits, methods or other semantic
settings requires a new run. Source changes stop new work and invalidate the run.
Each run also retains an allowlisted `source.zip` with the exact Python source,
project/declaration manifests and resolved suite. Its checksum is verified on
resume; a dirty working-tree hash alone is no longer the only reproduction artifact.

Cell attempts and content-checked operational checkpoints remain local.
Completed reports automatically create lossless gzip parts and `archive.json`;
use the index's listed parts, not a wildcard that could include older parts.
Standard selection reports omit full candidate rankings/scores; opt into
`--detailed-diagnostics` before a run when that extra evidence is required.

CatBoost uses native categorical inputs in final model evaluation; XGBoost
uses the matched numeric/one-hot preprocessing path. CatBoost's predictive
selection estimator currently sees the engine's numeric preprocessing, so that
representation difference is explicitly reported rather than claimed equivalent.
Binary average precision and macro-F1 complement the existing primary losses.
Summaries separate roles/source kinds and average variants within origin families.

### Protected confirmation and temporal evaluation

Confirmation manifests require a locked preregistration. The runner binds the
complete family cohort, methods, learner search, seeds, row policy and resource
limits before opening payloads; it rechecks the runtime exposure ledger.
Official repeat/fold/sample identities and class-support margins are validated,
and failed admission remains in the requested-cell denominator. Do not alter a
cohort after inspecting results.

New locked registrations can use per-family `split_plan` entries: exact
official repeat/fold/sample IDs or a fixed IID fold count. Official sources
still undergo full source-dimension validation; the selected subset is
recorded, never silently substituted. Mixed-source admission failures retain
their registered cell counts, and declared memory limits are bound. Historical
registrations without this field still evaluate all official splits.
A bounded official-split subset is not a full repeated-split leaderboard.

The [12-family native-reference confirmation suite](suites/final_reserved_confirmation_20260924.json)
locks one importance-paired candidate against raw, OpenFE-inductive-native
and AutoFeat-native, using linear/LightGBM and full rows. Its 128 cells use
ten official split-0 tasks and two three-fold UCI tasks, with a 38-hour total
ceiling. The [release audit](suites/final_reserved_confirmation_release_20260924.json)
preserves eligibility and the pre-outcome decision gates; launch or successful
admission alone is not performance evidence.

The [completed outcome](tabular-results/2026-09-24-frozen-native-confirmation/ANALYSIS.md)
contains 120 scored cells and eight explicit AutoFeat resource failures.
FeatCopilot completed 32/32, with +7.09% linear-classification mean loss
reduction, but failed the LightGBM quality gates. The candidate is not promoted;
these families are now exposed and cannot be reused as fresh confirmation.

The later [ten-family frequency confirmation](suites/frequency_fresh_confirmation_iid_20260928.json)
uses linear, LightGBM, CatBoost and XGBoost with raw/frequency-paired/native
reference representations: 480 cells under a 24-hour ceiling. Before any
payload or result access, the user approved whole-cohort three-fold IID
evaluation to retain the minimum test-class support. Original OpenML split
provenance is preserved; this is explicitly not official-fold reproduction.
See the [release audit](suites/frequency_fresh_confirmation_iid_release_20260928.json).

The [completed ten-family outcome](tabular-results/2026-09-28-frequency-iid-confirmation/ANALYSIS.md)
**fails promotion**: all 120 FeatCopilot cells complete, but nine of forty
family/learner comparisons exceed 5% harm. Linear classification improves
15.41% on average while linear regression worsens 26.68%; hardware and
temperature linear RMSE increase 61.25% and 45.61%. Forty-five reference
failures remain unscored. These origins are now exposed, defaults remain
unchanged, and SOTA is not established.

`label_end_column` supplies explicit label-availability metadata for temporal
splits; it never becomes a model feature. Supplied official indices are checked
against group/time/availability constraints, not silently rewritten.

### Explicitly budgeted live LLM methods

`llm-plan` and `llm-hybrid` are opt-in and never part of the default methods.
They require `--live-llm` and a shared durable `--llm-ledger` directory across
runs and retries. The approved allowance is 100 logical generation attempts;
failed attempts count. Hard output-token enforcement is the default preflight
requirement. For the current Copilot transport, use
`--allow-soft-llm-tokens` only with explicit user approval: 4,096 output tokens
then becomes a best-effort request, **not** a guaranteed cap.

Prompts use training schema and approved descriptions, never held-out rows or
labels. Semantic names are passed through an explicit column mapping. Stored
results retain proposal definitions and usage diagnostics but not learned
category maps or raw prompts. Unreported provider usage remains unknown.
SDK-internal inference/retry counts are not guaranteed by the logical-call ledger.
The library default remains `gpt-5.6-sol`, with no silent model substitution.

```powershell
python -m venv .venv-benchmark
.\.venv-benchmark\Scripts\python.exe -m pip install -e ".[benchmark-tabular]"
.\.venv-benchmark\Scripts\python.exe -m benchmarks.tabular --output benchmarks\tabular-results\my-run
```

Use a separate environment: AutoFeat 2.1.1 is incompatible with pandas 3
(`Series.ravel`) and sklearn 1.8 (`force_all_finite`). The `benchmark-tabular`
extra pins compatible ranges without constraining the base FeatCopilot install.

Without `--suite`, the default selects the 12-task legacy local replay manifest,
a fixed 5,000-row cap, 3 folds and 2 seeds, seven methods and five learner types.
Each dataset/split/method/learner worker has a 120-second whole-pipeline limit,
a 4 GiB sampled process-tree RSS limit, and two
threads. Cases execute sequentially. A four-hour total wall-clock budget marks
unfinished requests explicitly rather than dropping them from the denominator.

```powershell
# Small functional check, not a performance claim
python -m benchmarks.tabular --datasets magic abalone --folds 2 --seeds 42 --max-rows 400 --output benchmarks\tabular-results\smoke
# Expanded development measurement with separate synthetic diagnostics
python -m benchmarks.tabular --folds 3 --seeds 42 --synthetic --total-seconds 10800 --output benchmarks\tabular-results\development
# More repeated splits, if the budget permits
python -m benchmarks.tabular --folds 5 --seeds 42 137 --max-rows 0 --total-seconds 14400 --output benchmarks\tabular-results\expanded
# Compare the opt-in predictive strategy, task routing and legacy defaults
python -m benchmarks.tabular --methods baseline featcopilot featcopilot-task featcopilot-predictive polynomial openfe-stateless --learners linear lightgbm --folds 3 --seeds 42 --synthetic --output benchmarks\tabular-results\predictive
```

`featcopilot` retains automatic task inference. `featcopilot-task` changes only
explicit task routing. `featcopilot-predictive` uses learner-aligned selection;
it fits separate representations for each requested learner using that learner's
predeclared middle hyperparameter candidate, without looking at outer-test scores.
An independent per-learner cell time limit includes this work. Inner and outer
selection diagnostics are recorded in the raw result rows.

### Stronger opt-in reference and composition variants

The following methods are explicit choices, not automatic additions to the
default run:

| Method | Declared policy |
|---|---|
| `baseline-robust` | Raw features with training-min/max numeric model-input caps before the existing preprocessing |
| `baseline-target-robust` | The same capped raw classification inputs with fixed cross-fitted categorical target encoding |
| `baseline-discovery-robust` | Capped raw model whose three-setting tuning uses discovery rows only, matched to the fixed-plan protocol |
| `featcopilot-paired` | Experimental paired-gain selection; legacy and standard predictive defaults remain unchanged |
| `featcopilot-extended` | Extended numeric/categorical candidate pool with the existing one-SE rule |
| `featcopilot-extended-paired` | The same extended pool with paired-gain selection |
| `featcopilot-importance` | Extended pool, exact training-only deduplication, joint learner-importance ranking and one-SE subset selection |
| `featcopilot-importance-paired` | The same joint ranking with paired-gain subset selection |
| `featcopilot-frequency-paired` | Importance-paired plus bounded training-fitted unary frequencies; defaults unchanged |
| `featcopilot-frequency-aligned` | Frequency-paired with separately fitted, parameter-matched feature selection for each of the same three learner-search settings |
| `featcopilot-frequency-aligned-robust` | The same aligned method with independently fitted numeric caps in every internal and final learner |
| `featcopilot-frequency-aligned-target-robust` | The same capped aligned classification procedure with matched cross-fitted categorical encoding throughout selection and final fitting |
| `featcopilot-bagged-aligned-robust` | The same capped aligned configuration with five 85% IID training subsamples and mean candidate-importance ranks |
| `featcopilot-plan-gated-robust` | Discovery-only pipeline search, one independent fixed-plan gate, then identity-preserving state/model refitting |
| `openfe-inductive` | Full numeric/categorical order-1 operator search; frozen training statistics at inference; top-ranked additions limited by `--max-added-features` |
| `openfe-inductive-native` | The same full operator search, retaining all selected additions |
| `openfe-inductive-native-robust` | Native-width inductive OpenFE generation with the same final numeric-support caps as robust FeatCopilot |
| `autofeat-default` | Native generation/selection defaults and returned output width; no arbitrary first-50 truncation |
| `autofeat-native` | The same native algorithms/output width, with upstream `max_gb=None` rather than the 1-GiB generation-size hint |
| `autofeat-native-robust` | The same native generation followed by matched final numeric-support caps |
| `autofeat-native-safe-scale-v1-robust` | Separately pinned AutoFeat 2.1.1 compatibility variant: finite, exactly constant feature/noise columns use a unit scale; native results are not overwritten |
| `llm-hybrid-namespaced` | Hybrid with a training-defined `llm__` namespace and deterministic suffixes; no feature source overwritten or dropped |

OpenFE-inductive supports all operators from the pinned release, including
composed rules and learned frequency/group/combination statistics. It is still
an **adapted inductive reference**, not an unmodified transductive paper
reproduction: search retains the upstream train-internal protocol, while
inference uses frozen mappings and empirical group distributions. Unknown
frequencies become zero; unknown group aggregates/ranks are missing. Typed
combination keys distinguish delimiter/type collisions.

AutoFeat-default uses train-fitted categorical one-hot preprocessing because
the upstream categorical helper otherwise refits at inference. Original
categorical columns remain available to the matched downstream learner.
Its `max_gb=1` setting is a declared resource preference, not the external
worker's hard memory limit. Native output width differs from width-matched
methods and is reported explicitly.
`autofeat-native` removes that nondefault subsampling hint but keeps the same
external deadline/RSS limits. It does not repair upstream NaNs or drop constant
indicators, and it is not a replacement for the archived hinted-reference results.

Robust reference variants preserve upstream generation
inputs and search behavior, including training-median fill where previously
declared. They restore original numeric missingness before the final
learner fits its caps/imputer, avoiding mismatched original-column bounds.
Their generated features are capped only by the final matched learner;
this does not claim that upstream discovery uses FeatCopilot's clipping.

The namespaced hybrid freezes its aliases at fit time; test batches cannot
change feature names. The old `llm-hybrid` retains its explicit collision-error
behavior for reproducibility. Invalid plans still fail—namespacing does not
repair model semantics or reset the call ledger.

Frequency-paired fits each mapping only on the current internal training
partition, then freezes it for validation/inference. Missing and unseen values
map to zero. It reserves at most 32 candidates and at most 10% of the candidate
budget; unused capacity returns to the existing pool. This is a separate
development ablation, not a retrospective amendment to confirmation results.

Frequency-aligned is a benchmark-only pipeline ablation. It regenerates
features on the same inner-training partition for each declared learner
setting, using that setting for both internal feature selection and the
downstream learner. Inner-validation loss chooses the configuration; outer
refitting uses its same parameters. All additional feature fits count toward
runtime limits. `inner_pipeline_trials` retains each trial's loss, feature
count and selection report; ordinary methods still perform their original
two feature fits. Alignment does not ensure identical feature sets across
training partitions or protection from extrapolation.

The two `-robust` methods form a separately named preprocessing ablation:
fit numeric bounds at each applicable training minimum/maximum, cap model
inputs after feature generation, then retain existing imputation/scaling.
Bounds never learn from validation/test rows. The numeric safety guard is
shared with internal selection; categories and exported feature definitions
are unchanged. Raw robust inputs likewise retain missingness until model
preprocessing, avoiding an asymmetric pre-imputation step.

Robust preparation preserves Boolean dtype until the clipper so Boolean
fields are not capped to an observed Boolean value. The subsequent learner
preprocessor retains their numeric encoding, matching predictive selection
and the historical prepared inputs, including native CatBoost. The shared
finite guard acts before bounds fitting. For values beyond that guard,
comparisons with unguarded pipelines include guarding as well as min/max
saturation. Both matched robust arms use the same guard and Boolean treatment.

Reports include `matched_robust_summary` against `baseline-robust` on
complete matching splits, in addition to the usual unclipped-raw summary.
Only the matched comparison estimates incremental FE value beyond robust
preprocessing. Training-support caps can suppress legitimate extrapolation
and preserve bad training extremes; they are not a no-harm guarantee.

An all-robust comparison may declare `baseline-robust` as its sole paired
control. Mixed or nonrobust comparisons still require `baseline`. Reports
choose their reference from the declared methods, never by substituting a
successful control after another fails.

Target-encoding methods are an IID classification-only preprocessing
ablation with fixed five-fold cross-fitting and automatic smoothing.
The numeric candidate catalogue, caps and learner search do not change.
Final fitted encoder widths and training-array diagnostics are distinct
from the original-plus-generated feature width.
`matched_target_summary` compares FE with `baseline-target-robust`;
improvement due to changing the raw encoder is not credited as FE gain.
No precomputed target encoding is reused across tuning or selection splits.

Target-mode preparation preserves categorical scalar types until the shared
encoder handles them; the historical one-hot path retains its string
normalization. Fitted categorical-pair literals and frequency mappings may
therefore reflect those raw types, but target-encoded values never enter
the candidate formula pool. This type-preservation correction is part of
the declared target-mode pipeline, not an unreported change to OHE controls.

The fixed-plan protocol has an additional matched control:
`baseline-discovery-robust`. It reserves the same 25% gate rows from
hyperparameter tuning, tunes raw and engineered branches independently
within discovery, and compares the frozen winning pair once on the gate.
Acceptance requires strictly more than 1% additive-loss reduction
(log loss for classification, MSE for regression); paired SE is diagnostic,
not another cutoff. Invalid inputs/predictions remain errors. Rejection
reproduces the discovery-matched raw branch, not necessarily the ordinary
raw model tuned with more data.

After acceptance, only unsupervised frequency state may be refitted:
formula identities, operands, constants, ordering and output schema stay
fixed. No selector/search reruns or post-gate feature removal occurs.
Final model coefficients and model preprocessing are refitted normally.
This IID-only, benchmark-level protocol must execute through `run_case`;
calling only `generate_features` cannot reproduce it and is rejected.
Reports preserve `matched_plan_summary` separately from both the ordinary
baseline comparison and matched-clipping comparison. Frozen identities
reduce one refit mismatch, not every mismatch or the possibility of harm.

The [first fixed-plan experiment](tabular-results/2026-09-30-fixed-plan-development/ANALYSIS.md)
completed 384/384 cells and verified all gate/rejection/refit identities,
but reduced every observational mean gain versus the incumbent and
increased classification tail harm. Accepted gates did not ensure better
outer-test performance. The protocol remains experimental, with no
post-result margin tuning or default promotion.

Bagged ranking is a separate empirical ablation, not a statistical
false-selection guarantee. Each internal training partition constructs its
candidate universe, refits learned frequency/preprocessing state within
five deterministic training-only subsamples, and averages candidate-level
importance ranks with tied average ranks. Independent validation evaluates
the resulting prefixes. Positive-count final ranking repeats the same
procedure; there is no post-validation majority-vote filter.
The initial five-fit option is IID-only and rejects group/time metadata
explicitly. The existing single-fit method retains its prior behavior.

Its [first 288-cell measurement](tabular-results/2026-09-29-bagged-development/ANALYSIS.md)
did not support promotion: three mean-gain strata declined, boosted
classification's worst family harm increased to 32.90%, and measured
pipeline cost rose roughly 2.7-3.2 times. Boosted regression improved, but
feature-set stability remained mixed. The option is experimental.

The subsequent [frozen training-only audit](tabular-results/2026-09-29-aligned-training-audit/ANALYSIS.md)
found mixed results: alignment improved hardware mean RMSE by 16.03%
versus paired but worsened temperature by 17.73%. All 18 cells and
control-parity checks completed. This is retrospective diagnostic evidence,
not a fix for the independent confirmation failures or a reason to promote
the method.

The final audit's three oversized individual records are preserved through
its [supplementary archive index](tabular-results/2026-09-29-aligned-training-audit/supplementary_archive.json).
Reconstruct each listed gzip stream to its original filename before checking
the unchanged audit checksum index. The complete result archive also remains
available; raw local records are not deleted or modified.

Every new run creates a new output directory with `manifest.json`, `results.json`,
`REPORT.md`, and worker logs. Explicit `--resume` reuses matching cell results;
previous attempts are never overwritten.
Raw dataset downloads alone are cached by immutable source revision.
Nonzero exit status means at least one requested case failed/timed out or source
code changed while measuring; inspect the report instead of assuming total failure.
For version control, archive large `results.json` files losslessly as
`results.json.gz` and update the report's evidence link. Uncompressed results
and per-case scratch directories are ignored; run-level manifests, reports,
analysis and compressed evidence are retained.
If full selection diagnostics make an archive exceed the per-file limit, retain
the complete gzip stream in numbered `results.json.gz.partN` files and document
byte concatenation before decompression, as in the predictive-strategy report.

See [the protocol, audit findings, and improvement task list](TABULAR_PROTOCOL.md).
The [further improvement and dataset-expansion plan](IMPROVEMENT_PLAN.md) covers
selection reliability, broader dataset sources, confirmation-set isolation and
the work required before scaling this benchmark.
These are development measurements, not a reproduction of the full published
INRIA leaderboard or evidence that FeatCopilot is SOTA.

### Measured development baseline

The [cross-fitted target-encoding ablation](tabular-results/2026-10-03-target-encoding-development/ANALYSIS.md)
completed 216/216 cells and exactly reproduced all 108 OHE controls.
It did not justify replacing OHE: linear target FE gained 15.52% versus
its target-encoded raw comparator but only 0.89% versus OHE raw, with
substantial CKD harm. Matched preprocessing controls must be reported
alongside the stronger existing baseline.

The [public-classifier extraction replay](tabular-results/2026-10-03-classifier-extraction-parity/ANALYSIS.md)
completed all 162 outcomes and exactly reproduced prior metrics,
hyperparameters and member feature plans. The useful prediction procedure
is now available from `featcopilot.classification` without benchmark or
calibration dependencies. This is migration correctness, not new SOTA evidence.

The [latest four-family independent comparison](tabular-results/2026-09-30-robust-confirmation/ANALYSIS.md)
does not support promotion: the incumbent completed 45/48 cells, had mostly
neutral gains, and increased thyroid loss by 13.76% with CatBoost and
7.89% with XGBoost. Twelve AutoFeat and three incumbent cells timed out;
all remain unscored. The single regression family limits claim scope,
and all four families are now exposed.

The [matched robustness comparison](tabular-results/2026-09-29-robust-development/ANALYSIS.md)
completed 384/384 cells after a disclosed completion-only resume. Training
min/max caps raised linear FE gains to +16.12% classification and +14.97%
regression **against the identically capped raw learner**. All LightGBM
scores were unchanged, and classification selected-feature stability declined.
This is useful development evidence, not SOTA or default-promotion evidence.

The [training-only outlier audit](tabular-results/2026-09-29-robust-training-audit/ANALYSIS.md)
confirms the intended mechanism on its extreme temperature partition:
capped raw RMSE is 0.2645 versus 1.5419 raw, with capped FE at 0.2569.
Across the six audit partitions, matched FE gains are 15.12% for hardware
and 2.85% for temperature, but capped hardware FE is worse than uncapped
aligned FE. These are retrospective diagnostics, not fresh confirmation.

The [parameter-aligned pipeline ablation](tabular-results/2026-09-29-aligned-development/ANALYSIS.md)
completed 288/288 cells. It improved linear-classification mean loss reduction
from -0.30% to +14.96% against the corrected same-run control, but reduced
three other task/learner means and retained 22.76% Wisconsin/LightGBM harm.
It costs roughly 1.6-1.8 times as much and remains benchmark-only.
Historical missing-data scores differ after the documented imputation fix.

The [unary-frequency ablation](tabular-results/2026-09-24-frequency-development/ANALYSIS.md)
completed 384/384 cells. Frequency-paired improved linear-classification mean
loss reduction to +15.27% and boosted regression to +2.89%, while boosted
classification (+8.31%) and linear regression (+12.83%) were slightly worse
than the preceding candidate. Wisconsin Diagnostic/LightGBM still worsened
by 22.76%. All 288 control losses reproduced exactly; no default promotion.

The [LogisticRegression worker replay](tabular-results/2026-09-24-logistic-worker-efficiency/ANALYSIS.md)
preserved exact losses, selected features and learner parameters in 18
development cells. Avoiding a needless single-task process pool reduced
measured FeatCopilot pipeline time from about 97-104 seconds to 2.4-7.4
seconds on three families. Native thread caps and statistical settings are
unchanged; this is an efficiency result, not an accuracy or SOTA claim.

The [joint-importance comparison](tabular-results/2026-09-23-joint-importance-development/ANALYSIS.md)
completed 480/480 cells in 9.78 hours. Importance-paired reached +8.58% mean
boosted-classification loss reduction (OpenFE-inductive: +9.71%) and +13.37%
linear-regression reduction (+8.28%). Its boosted-classification median was
zero, Wisconsin Diagnostic/LightGBM worsened by 22.00%, and CKD/linear by
9.85%. It remains experimental; positive averages do not establish SOTA.
Exact archives, absolute losses and measured runtime comparisons are retained.

The [full-operator reference comparison](tabular-results/2026-09-22-full-reference-development/ANALYSIS.md)
adds 16-family development evidence. Paired FeatCopilot reached +14.18% linear
classification and +11.37% linear regression mean loss reduction, but
OpenFE-inductive retained a clear boosted-classification advantage. The
paired rule did not improve every regression case and remains experimental.
Failures, width policies, absolute-loss context and exact archives are retained.

The [six-family confirmation outcome](tabular-results/2026-09-21-confirmation-completion/ANALYSIS.md)
now contains all 1,440 scored cells after a transparent two-family categorical
schema repair. Classification was essentially neutral; regression gains were
driven by Airfoil. This is reliability evidence, not a broad SOTA result or a
reason to promote defaults automatically.

Separate development reports cover the
[budgeted typed-LLM pilot](tabular-results/2026-09-20-llm-pilot/ANALYSIS.md),
[Seoul temporal evaluation](tabular-results/2026-09-20-seoul-temporal/ANALYSIS.md),
and [retail point-in-time aggregates](tabular-results/2026-09-21-retail-relational/ANALYSIS.md).
Their populations and comparison types differ and must not be pooled into one
headline score. Total active measurement time across this roadmap cycle was
approximately 46.65 hours, with 96 logical LLM attempts under the approved limits.

The [protocol-v3 foundation pilot](tabular-results/2026-09-17-roadmap-pilot/ANALYSIS.md)
completed 1,512 per-learner cells across seven reviewed UCI tasks and cached
houses/wine diagnostics. All FeatCopilot cells succeeded. On matching seed-42
diagnostic splits, the earlier houses/wine linear regressions changed from
-32.33%/-15.88% loss reduction to +5.76%/+1.49%. Across both seeds, wine still
shows a -1.05% regression; stability improved, not a universal no-harm guarantee.
The report separates observational tasks from Energy Efficiency simulation
and includes CatBoost/XGBoost, cost and incomplete comparator coverage.

The [predictive-strategy follow-up](tabular-results/2026-09-16-predictive/ANALYSIS.md)
measures the new opt-in path against legacy behavior and competitors on the same
development suite. Mean loss reductions rose from +0.55% to +6.76% for linear
classification, and from +0.08% to +2.10% for LightGBM regression.
However, linear regression worsened materially on houses and wine, and
OpenFE-stateless still leads several aggregates. Defaults therefore remain
legacy; this is progress, not a SOTA claim.

The [2026-09-16 analysis](tabular-results/2026-09-16/ANALYSIS.md) and
[full report](tabular-results/2026-09-16/REPORT.md) cover all 12 real datasets plus
two separate synthetic diagnostics (three folds, one seed). FeatCopilot completed
all cases, with mean LightGBM loss reductions of +0.26% for classification and
+0.08% for regression; OpenFE-stateless achieved +0.98% and +3.49% on the same
datasets. AutoFeat's 24 case timeouts remain unscored and visible.
The main improvement targets are now explicit task inference and generation /
selection / gating ablations, not stronger marketing claims.

## Legacy experiments (archived evidence)

Historical reports are preserved, but their mixed synthetic/real summaries,
test-fold "best model" scores, composite rankings, and "do-no-harm guarantee"
language must not be used to justify superiority. The following commands describe
the legacy interfaces; they do not implement the new protocol.

## Structure

```
benchmarks/
├── datasets.py                    # Unified dataset API (63 datasets)
├── datasets.md                    # Dataset documentation
│
├── automl/                        # AutoML benchmarks
│   └── run_automl_benchmark.py    # FLAML/AutoGluon with FeatCopilot
│
├── simple_models/                 # Simple models benchmarks
│   └── run_simple_models_benchmark.py  # RF/LogReg/Ridge with FeatCopilot
│
└── compare_tools/                 # FE tools comparison
    └── run_fe_tools_comparison.py # FeatCopilot vs other FE frameworks
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[benchmark]"

# Run AutoML benchmark
python -m benchmarks.automl.run_automl_benchmark

# Run simple models benchmark
python -m benchmarks.simple_models.run_simple_models_benchmark

# Run FE tools comparison
python -m benchmarks.compare_tools.run_fe_tools_comparison
```

## Benchmark Scripts

### 1. AutoML Benchmark (`automl/run_automl_benchmark.py`)

Compares AutoML frameworks (FLAML, AutoGluon) with and without FeatCopilot.
FeatCopilot runs all applicable engines per dataset (tabular, relational) plus LLM when enabled.

```bash
# Quick benchmark (10 datasets)
python -m benchmarks.automl.run_automl_benchmark

# Specific datasets
python -m benchmarks.automl.run_automl_benchmark --datasets titanic,house_prices

# All classification datasets
python -m benchmarks.automl.run_automl_benchmark --category classification

# With LLM engine
python -m benchmarks.automl.run_automl_benchmark --with-llm

# Use AutoGluon
python -m benchmarks.automl.run_automl_benchmark --framework autogluon

# All frameworks
python -m benchmarks.automl.run_automl_benchmark --framework all
```

**Options:**
- `--datasets NAME,NAME` - Specific datasets
- `--category {classification,regression}` - All datasets in category
- `--all` - All datasets
- `--framework {flaml,autogluon,h2o,all}` - AutoML framework
- `--with-llm` - Enable LLM engine
- `--time-budget N` - AutoML time budget (default: 120s)

**Output:**
- Per-framework: `AUTOML_{FRAMEWORK}_BENCHMARK.md`
- Combined: `AUTOML_BENCHMARK.md` (when running multiple frameworks)

### 2. Simple Models Benchmark (`simple_models/run_simple_models_benchmark.py`)

Compares simple models with and without FeatCopilot.
FeatCopilot runs all applicable engines per dataset (tabular, timeseries, text, relational) plus LLM when enabled.

```bash
# Quick benchmark
python -m benchmarks.simple_models.run_simple_models_benchmark

# All regression datasets with LLM
python -m benchmarks.simple_models.run_simple_models_benchmark --category regression --with-llm
```

**Models:**
- Classification: RandomForest (n_estimators=200, max_depth=20), LogisticRegression
- Regression: RandomForest (n_estimators=200, max_depth=20), Ridge

**Output:**
- Without LLM: `SIMPLE_MODELS_BENCHMARK.md`
- With LLM: `SIMPLE_MODELS_BENCHMARK_LLM.md`

### 3. FE Tools Comparison (`compare_tools/run_fe_tools_comparison.py`)

Compares FeatCopilot with other feature engineering frameworks using FLAML (30s budget).

```bash
# Compare all available tools
python -m benchmarks.compare_tools.run_fe_tools_comparison

# Specific tools
python -m benchmarks.compare_tools.run_fe_tools_comparison --tools featcopilot featuretools autofeat
```

**Tools compared:**
- Baseline (no FE)
- FeatCopilot
- Featuretools
- autofeat

**Output:** `FE_TOOLS_COMPARISON.md`

## Datasets

63 datasets across 4 categories:

| Category | Count | Examples |
|----------|-------|----------|
| Classification | 26 | titanic, credit_risk, xor_classification, higgs |
| Regression | 30 | house_prices, complex_regression, diamonds |
| Forecasting | 3 | sensor_anomaly, retail_demand |
| Text | 4 | product_reviews, fake_news |

```python
from benchmarks.datasets import list_datasets, load_dataset

# List all datasets
list_datasets()

# List by category
list_datasets('classification')

# Load dataset
X, y, task, name = load_dataset('titanic')
```

See `datasets.md` for full documentation.

## Dependencies

### Core
```bash
pip install -e ".[benchmark]"  # Includes FLAML
```

### Optional (for tool comparison)
```bash
pip install featuretools  # Deep Feature Synthesis
pip install tsfresh       # Time series features
pip install autofeat      # Automatic feature generation
pip install openfe        # OpenFE
pip install caafe         # CAAFE (requires OpenAI API)
```

### Optional (for AutoML)
```bash
pip install autogluon     # AutoGluon
```

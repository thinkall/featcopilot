# FeatCopilot improvement and benchmark-expansion plan

**Date:** 2026-09-17. **Status:** approved for incremental implementation.

### 2026-10-08: regression robustness before more operator search

The next bounded hypothesis transfers the already fixed classification
averaging rule to regression: fit the complete robust raw and aligned-FE
procedures independently, then average their predictions **50:50**.
This follows the prediction-level combination used by
[scikit-learn VotingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html),
not an average of reported RMSE scores. Averaging can still be worse than
the better component; it is not a no-harm guarantee.

1. **Implement a benchmark-only regression arm.** Reuse canonical inner
   tuning, train-only preprocessing/selection and final refitting rather
   than copying the fitting algorithm. Preserve separate component
   parameters, plan-name identities, prediction hashes and costs. Use linear and
   LightGBM learners only, IID regression, identity targets and the
   existing fixed train-min/max caps. No optimized weights, new
   operators, target encoding or production-default changes.
2. **Freeze a four-hour development comparison before outcomes.** Use
   the seven original regression families (Concrete, Energy, Combined
   Cycle Power, Real Estate, Auto MPG, Wine and California Housing),
   plus separately reported, already exposed hardware/infrared stress
   cases under their existing rights/split constraints. The metadata
   audit requires hardware to use **all ten official folds, uncapped**;
   this is not a replay of its earlier three-fold IID failure.
   Compare robust raw, aligned FE and the fixed average at seed 42.
   The seven primary families plus infrared use three IID folds and
   at most 5,000 rows. All use eight threads, 600 seconds/cell and
   4,096 MiB RSS. There are **204 cells**, with stage maxima of three
   hours for the IID panel and one hour for hardware, including a
   shared four-hour total cap. Every physical fit counts. The remaining cycle
   allowance is 10.94712 hours; this does not create a new allocation.
3. **Check the intended outcome, not just a mean.** Require exact
   component/control prediction and plan-name hashes, parameters and
   inner losses on matching cells. Retain every failure and report
   equal-family RMSE gains, absolute losses, worst-family harm and
   runtime per learner. Keep stress cases separate from primary means.
   Positive primary mean gain versus raw and no greater-than-5% raw
   harm in either declared cohort are necessary for follow-up, not
   sufficient for promotion. Report the gain/tail tradeoff versus
   single FE even when it favors the existing method.
4. **Use the result to choose the next bounded step.** A complete,
   promising result merits fixed-code multi-seed replication, then
   explicitly registered native-reference and fresh-origin comparisons.
   Otherwise preserve the negative result and investigate training-only
   error complementarity/refit instability; do not sweep weights or
   quietly change the dataset panel. Public regression API extraction
   remains conditional on evidence, not a prerequisite for the study.

The existing blocked confirmation and all older archives remain
unchanged. No live LLM calls or new data-rights exceptions are authorized.

The [completed study](tabular-results/2026-10-08-regression-average-development/ANALYSIS.md)
scored **204/204 cells** in **3.34062 hours**, with all 136 matched
component/control pairs exact on the declared hashes, parameters,
trial losses and metrics. LightGBM averaging gains **3.36% versus raw**
and **1.20% versus single FE** across the seven primary families.
Linear single FE gains 24.46% versus raw, compared with averaging's
16.35%, and beats averaging on all nine family means. Thus retain
**linear single FE / LightGBM fixed average** as the development-derived
policy for frozen multi-seed replication. Both average variants pass
the necessary family-mean follow-up gates, but LightGBM hardware fold 0
still has **8.97% harm**, versus 73.94% for single FE. This is not a
per-fold safety guarantee or a SOTA result. Current-cycle use is
**64.39350 hours**, leaving **7.60650**; no new LLM calls were made.

The [frozen policy replication](suites/regression_policy_replication_20261008.json)
uses seeds **137 and 2029** with the unchanged model/data implementation.
Keep raw and single FE for linear; retain raw, single FE and fixed
averaging for LightGBM. This yields **340 cells**, with the same seven
primary families and two separately reported stress families. All old
source members, dependencies and prepared payloads must match the
seed-42 study; only the prospective protocol descriptor is added.
Use one **seven-hour maximum**, including controller time, within the
remaining 7.60650-hour allocation. Apply complete coverage, parity,
positive primary mean gain and the 5% family-harm limit at **each seed
and pooled**, while retaining individual-fold harms. The policy was
selected on development evidence; this is replication, not fresh
confirmation, a default change or an authorization for new LLM calls.

The [replication completed](tabular-results/2026-10-08-regression-policy-replication/ANALYSIS.md)
**340/340 cells** in **5.56845 hours**, with all 136 component/control
pairs exact. Both learners pass their necessary family-mean gates at
each seed and pooled: linear single FE gains **23.79%** versus raw;
LightGBM averaging gains **2.97%** versus raw and **1.45%** versus
single FE on the pooled primary panel. However, linear hardware
seed-2029/fold-6 has **245.61%** harm and LightGBM averaging's worst
hardware fold has **15.63%** harm. The policy is not robustly accepted.
Freeze these results and investigate training-only selection/refit
instability before considering a public regression API or fresh
confirmation. No weight sweep is justified by these exposed tails.
Current-cycle use is **69.96195 hours**, leaving **2.03805**; the LLM
count remains 96/100.

The read-only tail audit finds genuine, seed-dependent representation
changes on identical hardware folds, not a denominator artifact:
seed-2029/fold-6 linear RMSE increases from 54.6728 to 188.9567 after
an inner ten-addition plan becomes fifty additions at final refit.
However, another harmed fold shrinks its plan, and 28/30 hardware cells
change selected names, so width/turnover alone is not a failure rule.
The [predeclared train-only numerical audit](suites/regression_training_audit_20261008.json)
instruments unchanged raw/aligned-FE fitting across **all ten folds and
all three existing seeds**, with workers receiving only original
outer-TRAIN data. Record plan turnover, scaled-design conditioning,
coefficient norms, in-sample residuals and training clipping boundaries;
require exact original parameters, inner losses and final feature-name
hashes. One **one-hour ceiling**, 180 seconds/job and eight threads,
comes from the remaining 2.03805 hours. This is diagnostic evidence,
not a new test result or permission to fit case-specific thresholds.

The [TRAIN-only audit completed](tabular-results/2026-10-08-regression-training-audit/ANALYSIS.md)
all **30 jobs** in **0.36681 hours**, with exact historical parity on
the four registered fields for all sixty component replays. The worst
prior failure's final design is rank 53/56, alpha 0.1 and coefficient
L2 norm 270.10 versus 66.34 at the other two seeds on the same official
fold. Its lower training residual does not explain generalization.
Four final designs are rank deficient; successful rank-deficient and
harmful full-rank fits prevent a simple rank/width cutoff. Condition
numbers describe the unpenalized transformed design, not the centered
regularized Ridge solve. No algorithm, default or previous scored
result changed. Current-cycle use is **70.32876 hours**, leaving
**1.67124**; 96/100 LLM attempts remain consumed. The next defensible
experiment is training-partition validation of the complete
representation/regularization procedure with matched budget controls,
not a threshold fitted to these exposed tails. Do not spend the
remaining allowance on an arbitrary operator or mixture-weight sweep.

### New stability-research cycle: separately authorized

After the robustness audit, the user explicitly authorized **72 additional
measurement hours**, still at eight threads. This is a new allocation,
not permission to relabel or discard prior failures. The total live-LLM
ceiling remains 100 logical attempts, with 96 already consumed.
The preceding cycle closes at 71.32057 measured hours plus its separate
0.25-hour operational allowance; its unused remainder is not rolled into
this new allocation.

The next hypothesis is **selection-frequency or importance aggregation**
from training-only resamples, informed by AutoFeat and primary stability-
selection literature. This differs from the already implemented input-value
frequency features. Aggregation must occur inside each selection-training
partition, so the independent validation scores the actual proposed
representation. A final post-validation consensus filter would not satisfy
that requirement.

The primary-source review recommends **five-bag mean-rank aggregation**
as the smallest next experiment, not a hard selection-frequency threshold.
[AutoFeat](https://github.com/cod3licious/autofeat/blob/639793a6e620f2b8da21f32620a1c94d1d7c50bc/src/autofeat/featsel.py#L179-L276)
uses five repeated selections on 85% subsamples, supporting the resampling
idea but not proving this different rank-averaging procedure effective.
Its pre-subsampling scaling is not copied: each learner fit must retain
training-local preprocessing.

For the first ablation, fix five 85%-without-replacement bags per joint
ranking, using deterministic seeds derived from the configured random seed.
Average generated-candidate importance ranks, with average ranks for ties;
exclude candidates never receiving positive importance and break aggregate
ties by canonical identity. Keep all originals, existing importance
definitions, candidate budgets and subset proposals. Validate those ranked
prefixes inside the existing holdouts, and use the same aggregation
procedure during final refitting. A single-fit default remains unchanged.

This is **empirical resampled importance ranking**, not theoretical
stability selection. [Meinshausen--Buehlmann](https://arxiv.org/pdf/0809.2932v2)
requires specific noise-selection assumptions for false-selection bounds;
[Shah--Samworth CPSS](https://arxiv.org/pdf/1105.5578v2) uses complementary
half-samples and distinguishes low-selection-probability bounds from
false-positive claims. Five overlapping 85% bags do not justify PFER/FDR
guarantees. Correlated alternatives can split votes or change names without
equivalent prediction changes, so a hard frequency filter and correlation-
group representative selection are deliberately deferred.

The first implementation uses `importance_resamples=1|5`: one preserves
the existing full-training fit, while five uses fixed 85% IID resamples.
The initial five-fit mode explicitly rejects group/time metadata rather
than adding unverified resampling policies. Existing single-fit group/time
behavior remains unchanged.

Predeclare **288 development cells** on the original sixteen-family panel:
`baseline-robust`, `featcopilot-frequency-aligned-robust`, and
`featcopilot-bagged-aligned-robust`; linear/LightGBM, three folds, seed 42,
5,000-row cap, 1,000 candidates, 50 additions, three validation repeats,
unchanged three-setting learner grids and fixed `(0,1)` numeric caps.
Ceiling: **12 active hours**, eight threads, 600 seconds per cell and
4,096 MiB sampled RSS. Every additional ranking fit counts toward runtime;
no three-versus-five or fraction sweep is part of this experiment.
Use the automatic matched robust-baseline summaries, full failure coverage,
tail harms, selected-count dispersion and fold-name Jaccard. Keep the
pending new cohort closed until a candidate and confirmation protocol
are frozen.

The [five-bag experiment](tabular-results/2026-09-29-bagged-development/ANALYSIS.md)
completed 288/288 cells in 6.36447 hours; all 192 control losses reproduced
exactly. Bagging improved boosted-regression mean gain from 1.39% to 2.60%
but reduced the other three observational means, increased the largest
boosted-classification harm to 32.90%, and cost 2.7-3.2 times as much.
Stability moved in mixed directions. Promotion is rejected; do not sweep
bag fractions/counts or consume the fresh cohort to rescue this result.
The new cycle has **65.63553 measurement hours remaining**.

The next bounded research question is independent validation of a fixed
feature plan, with validation rows withheld from the entire search and
ranking process and no reranking after acceptance. This targets selection
optimism and plan/refit mismatch rather than adding operators or another
ranking ensemble. Its precise criterion and implementation must be frozen
before measurement; no accuracy or no-harm claim is made yet.

**Next fixed protocol:** within each allowed outer-training set, reserve
25% as an IID gate using seed `(seed + 104729) mod 2**32`. Complete raw
model tuning and feature-pipeline discovery using only the other 75%,
including a separate discovery-only inner split. Freeze one winning
feature plan and both model configurations before evaluating the gate.
The raw and engineered models use the same three-setting search spaces,
but each chooses its own settings inside discovery; a stronger independently
tuned raw comparator is preferable to forcing the raw model to use the
engineered model's settings.

Accept the plan only when its mean additive gate loss is **strictly below
99% of raw loss**, with positive raw loss. Classification uses additive
log loss to match the existing primary objective; regression uses MSE,
so a 1% MSE threshold is not a 1% RMSE threshold. Paired per-example SE
is descriptive only. This fixed engineering margin is not statistically
calibrated or searched. Invalid predictions, nonfinite metrics, infeasible
splits or malformed plans remain errors, not successful raw fallbacks.
Valid rejection and a valid zero-feature proposal choose the matched raw
branch explicitly.

After acceptance, refit only unsupervised unary-frequency state on all
allowed training rows, preserving formulas, operands, fixed constants,
column order and schema. Do not rerank, filter or refill the plan; final
model preprocessing and coefficients may be refitted normally.
The matching `baseline-discovery-robust` uses the identical discovery-only
raw tuning and all-training refit, so a rejected candidate can reproduce
that control exactly. Also retain the ordinary robust baseline: a win over
a control tuned on less data is not automatically a win over the stronger
existing pipeline.

This follows the leakage/procedure distinction in
[sklearn nested CV](https://github.com/scikit-learn/scikit-learn/blob/c9021ac077341eef99fe4c1756597f52d3b2a870/examples/model_selection/plot_nested_cross_validation_iris.py)
and [Cawley--Talbot's model-selection analysis](https://jmlr.org/papers/v11/cawley10a.html).
It is a new empirical protocol, not a package reproduction or reusable-
holdout/no-harm theorem. The inspected AutoFeat/OpenFE paths use validation
inside discovery, not this independent fixed-plan gate.

Predeclare 384 development cells on the same original sixteen families:
ordinary robust raw, current single-fit aligned robust FE,
discovery-matched robust raw, and `featcopilot-plan-gated-robust`;
linear/LightGBM, three folds, seed 42, existing row/feature/search limits.
Ceiling: 12 hours, eight threads, 600 seconds/cell and 4,096 MiB RSS.
No original outer-test values or labels inform gate decisions. Keep all
four reserved families closed; no parameter or margin sweep follows.

The [fixed-plan comparison](tabular-results/2026-09-30-fixed-plan-development/ANALYSIS.md)
completed 384/384 cells in 5.63226 hours, with all 192 incumbent/control
losses reproduced exactly. Fifty proposals passed the gate; 35 had no
features and eleven failed the fixed improvement margin. All rejection
control identities and accepted formula hashes checked out, but the
candidate underperformed the incumbent in every observational mean and
worsened Wisconsin classification tail harm. The empirical gate hypothesis
is not promoted; do not sweep its margin to rescue the result.

The cycle now has **60.00326 hours remaining**. Freeze the existing
single-fit, capped, frequency-aligned incumbent—not either unsuccessful
new variant—for an independent four-family comparison. Use ordinary
robust raw, the incumbent, native-width inductive OpenFE and native
AutoFeat with the same final numeric clipping policy. Reference feature
generation retains its declared upstream preprocessing/search; clipping
is applied to the matched final learner, not claimed to be part of the
upstream algorithm.

Predeclare all four previously reserved families, four learners
(linear/LightGBM/CatBoost/XGBoost), three IID folds, seed 2028, full rows,
the existing three-setting grids and feature budgets: **192 cells**,
at most twelve hours, eight threads, 900 seconds/cell and 4,096 MiB RSS.
Release only after current exposure, rights, schema and timing checks.
Positive means, no greater-than-5% family harm and complete paired
reference coverage remain necessary but not sufficient for any promotion.
One regression origin and three classification origins cannot support
a broad SOTA conclusion.

The [four-family executable suite](suites/cycle2_robust_confirmation_20260930.json)
and [release audit](suites/cycle2_robust_confirmation_release_20260930.json)
now bind that exact comparison. All four sources passed preparation and
actual three-fold IID/class-support checks, with twelve splits and
192 planned cells. No models were fit during preparation. Runtime source
is frozen at
`2c56b1386d67e0a812324482b57ba4f345e5d404f32c3b2cf437765456036e15`;
source/reference changes require a separately declared experiment.

The [four-family confirmation outcome](tabular-results/2026-09-30-robust-confirmation/ANALYSIS.md)
is complete: 192 attempted, 177 scored, in 10.77356 hours.
The incumbent completes 45/48 cells; three CatBoost/personality cells
and twelve AutoFeat/personality cells time out at 900 seconds.
Thyroid loss increases 13.76% for CatBoost and 7.89% for XGBoost.
Other gains are mostly neutral, with one-family Facebook/XGBoost +3.82%.
The frozen coverage/quality/harm gates fail, and the four origins are
now exposed. Defaults remain unchanged.

Current-cycle measurement is **22.77030 hours**, leaving **49.22970**.
The next bounded public-prior investigation concerns prediction-level
cross-fit ensembles and sigmoid calibration, with raw controls using the
same architecture. Any calibration fold must contain the entire
feature-discovery/preprocessing/learner fit inside its training side;
precomputing selected features on the full calibration input would violate
that boundary. Calibration cannot be assumed to fix ranking/discrimination
errors, and it must not be credited as FE gain without matched controls.

**Next fixed classifier-family study:** raw versus the unchanged single-fit
robust aligned FE procedure, each evaluated as (1) a single model,
(2) a three-fold cross-fit prediction ensemble, and (3) the same fitted
ensemble with sigmoid calibration. Use the nine original classification
families, linear/LightGBM, three outer folds, seed 42 and the existing
row/feature/search limits: **324 prediction-result cells**, with a
twelve-hour ceiling, eight threads, 900 seconds per physical job and
4,096 MiB sampled RSS.

Strict calibration independence requires hyperparameter tuning inside
each base-training fold as well. The complete base learner therefore owns
its three-setting inner search, all FE discovery, frequency state, caps,
imputation and scaling. No calibration-fold features or labels enter
those fits. Retain `CalibratedClassifierCV(ensemble=True, method="sigmoid",
cv=3, n_jobs=1)` and do not refit a base estimator on all training rows
after calibration.

The plain and calibrated views share the **exact fitted base objects**,
not merely matching seeds. Single controls are separate jobs so a bundle
failure does not erase their completed measurements. The study has 108
single jobs and 108 paired-view jobs: **216 physical jobs**. Physical
training time is counted once; standalone method costs attribute the
shared base-training cost to each prediction view explicitly.

Use fixed log loss as primary loss and consistently scaled Brier sum
(sum of classwise squared errors, including both binary coordinates),
accuracy and discrimination metrics as diagnostics. Separate FE effect
within an architecture, cross-fit-ensemble effect, and calibration effect
on identical base models. The ensemble comparison changes training size
as well as variance, and Brier/log loss do not isolate calibration alone.
Child plans may differ; report per-child widths and schema hashes rather
than pretending a union of features is one deployed plan.

The design follows the pinned
[sklearn calibration implementation](https://github.com/scikit-learn/scikit-learn/blob/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/calibration.py)
and [small-sample calibration guidance](https://github.com/scikit-learn/scikit-learn/blob/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/doc/modules/calibration.rst),
with [AutoGluon prediction averaging](https://github.com/autogluon/autogluon/blob/2d7d603c7c26234f578d3a593cf466165b649fa1/core/src/autogluon/core/models/ensemble/bagged_ensemble_model.py)
as additional prior art. It does not copy AutoGluon's use of held-out fold
data during child fitting. Research source versions are not runtime
dependency upgrades. Sigmoid, three folds and equal prediction weights
are fixed; no calibration-method, threshold or learned-stacking sweep is
part of this experiment.

The shared-fit core and controller are implemented. Review-driven fixes
cover interrupted preparation, revalidation of original source checksums
on resume, and exact nine-family/source admission. All nine sources
prepared successfully with 27 outer splits matching the earlier runs:
216 physical jobs and 324 prediction outcomes. No models were fit during
preparation. Source is frozen at
`738d4fc39b57cb5fb3557070ed1cb7460c809114099656e9deb87e8a5262a40e`.

The [complete six-arm outcome](tabular-results/2026-10-01-calibration-development/ANALYSIS.md)
has 324/324 scores from 216 physical jobs in 4.77469 hours, with all
108 old single controls reproduced exactly. Sigmoid calibration worsens
mean log loss/Brier sum and is rejected as a general upgrade.
Plain LightGBM FE averaging gains 8.76% over matched raw averaging,
with worst family harm only 0.40% (1.23% versus ordinary single raw).
Linear averaging instead worsens CKD by 27.98% versus its matched raw
ensemble. No per-dataset architecture switching is allowed.

Freeze a learner-aware **development-only** policy: existing single robust
FE for linear, plain cross-fit robust FE ensemble for LightGBM. Do not
generalize it to unmeasured learners. Replicate the unchanged six-arm
study on seeds **137 and 2029**, fixed before their outcomes, with the
same nine origins: 648 prediction outcomes/432 physical jobs, twelve-hour
ceiling, eight threads, 900 seconds/job and 4,096 MiB RSS. Sigmoid remains
diagnostic, not a post-hoc fallback candidate. These are new splits of
exposed origins, never fresh confirmation.

Current-cycle measurement totals 27.54499 hours; 44.45501 remain.
A separate bounded metadata review may examine Mammographic Masses,
Blood Transfusion Service Center, Haberman Survival and HTRU2, excluding
all known aliases/exposures and keeping source, entity and timing criteria
unchanged. None is acquired or modeled before explicit admission and freeze.

The four-source review added no reservation: Blood Transfusion is already
exposed under a TabArena alias; Mammographic Mass/Haberman lack sufficiently
documented case-to-patient/index-operation independence; HTRU2 has unresolved
candidate/source grouping and conflicting primary-source license metadata.
No acquisition or eligibility relaxation followed.

The seed-137/2029 replication is prepared under the **same runtime source
and dependency identity** as seed 42, with nine admitted families, 54 outer
splits, 432 physical jobs and 648 prediction outcomes. The learner-aware
policy remains fixed before scoring. No new source, architecture or
calibration setting is selected from the replication outcomes.

The [replication](tabular-results/2026-10-01-calibration-replication/ANALYSIS.md)
completed 648/648 outcomes in 9.95263 hours. The fixed linear-single policy
gains 18.06% over raw with worst family harm 1.00%. LightGBM plain FE
averaging gains 6.89% over matched raw averaging with worst harm 0.59%,
but remains 23.66% worse on CKD than the full-training single raw learner.
That predeclared stronger-baseline harm gate fails; do not loosen it or
switch architectures by dataset. Current-cycle use is 37.49762 hours,
with 34.50238 remaining.

The next single intervention is **three 85%-retention stratified training
bags**, replacing the two-thirds training partitions only in an opt-in
`ensemble_sampling="subsample85"` mode. The `cv3` default is unchanged.
The larger training share is a hypothesis, not a demonstrated cause of
CKD's outcome; it also increases model overlap and reduces diversity.

Use `make_holdout_splits(test_size=0.15, n_splits=3)` with the existing
ensemble seed offset and consecutive seeds, preserving positional order.
Each member's training/calibration sides remain disjoint and require every
class. Across members, calibration holdouts may overlap or leave rows
uncovered: report those facts and do not call this a three-fold partition
or pooled OOF predictions. Complete hyperparameter search, FE and
preprocessing remain inside each member's training side. No fraction
sweep, retries for inconvenient classes, or new feature rules.

Measure **LightGBM only**, leaving the frozen linear-single policy alone,
on the same nine families and seeds **42, 137 and 2029**: 486 prediction
outcomes from 324 physical jobs, with a sixteen-hour cumulative ceiling,
eight threads, 900 seconds/job and 4,096 MiB RSS. Compare against archived
`cv3` results on identical source/splits and both matched raw85 and the
ordinary full-training raw learner. Plain/sigmoid views still share exact
base fits; sigmoid remains diagnostic rather than another candidate.
No fresh-confirmation claim follows from this exposed-panel experiment.

The higher-retention core/controller are frozen and all nine datasets
passed preparation. The 81 outer split identities match the archived
seed-42/137/2029 studies exactly, with unchanged dependencies and
324 physical jobs/486 prediction outcomes. Source:
`ec3d1f9c93d39ff9e6667085571be2194cfcf73197d274ca6c653095f5757ef5`.
No models were fit during preparation.

The [qualified two-source protocol](suites/standard_iid_policy_confirmation_20261003.json)
and [release audit](suites/standard_iid_policy_confirmation_release_20261003.json)
are now locked. Both sources passed actual preparation/class-support
checks, yielding six IID splits, 72 physical jobs and 96 prediction
outcomes. Source freeze:
`d8952d7715d9d5aaf786708822f32b4f554e9dca21c7b2d7f90dfe0bce02d826`.
All outputs retain the unverified-entity standard-IID tier. The
confirmation-specific decision keeps both families, absolute losses and
all four comparator gates; missing or undefined comparisons explicitly
block acceptance. No automatic SOTA, patient-independent or clinical
generalization claim follows even if the scoped gates pass.

The [qualified confirmation](tabular-results/2026-10-03-qualified-policy-confirmation/ANALYSIS.md)
completed 96/96 outcomes from 72 physical jobs in 1.74079 hours.
The overall locked decision fails only the linear-versus-AutoFeat mean
comparison (-1.28%). LightGBM FE85 passes its scoped comparisons:
+2.25% versus raw single, +0.22% versus matched raw85, +7.33% versus
native OpenFE and +7.51% versus native AutoFeat. All raw-family harms
remain below 5%, but the restricted two-family tier cannot establish
clinical/entity-independent or broad SOTA performance. No policy or
threshold is changed after these results.

Current-cycle measurement is 48.36633 hours; 23.63367 remain.
The next engineering task is to design an explicit experimental
sklearn-compatible prediction artifact for the useful plain ensemble
procedure without depending on benchmark modules. It must retain separate
member feature plans rather than expose a fictitious union of features,
and keep every existing `AutoFeatureEngineer` default unchanged.

The approved extraction introduces an explicitly experimental
`featcopilot.classification.FeatureEnsembleClassifier` with user-supplied
bare classifier, three native parameter dictionaries, and required
`single`/`subsample85` mode. A reusable production base procedure owns
inner search, FE and final preprocessing without importing benchmark or
calibration code. The benchmark retains its alias/model-family adapters,
including native CatBoost representation, and delegates shared fitting
instead of duplicating a production algorithm.

Before accepting the extraction, require frozen synthetic parity for
trial losses, chosen native parameters, formula order, fitted numeric/
frequency state and probabilities; old class-location or pickle byte hashes
are not a valid semantic comparison. Public plain85 must not construct
calibrators. Installed-package import and inference must work with benchmark
modules unavailable, and the base dependency minimum is not silently
raised. Any real-data parity measurement gets a separate bounded run;
the qualified confirmation records remain immutable.

The extraction is implemented and passes exact frozen synthetic parity
checks. Review identified and corrected original-label class-weight
translation and mandatory diagnostic-pickling failures. A built wheel
imports and performs three-member fitting/prediction with benchmark,
optional-learner/provider imports and calibrator construction blocked;
weighted string-label and pickle-inference behavior also pass.

Before declaring the artifact parity-verified on real data, run a separate
bounded replay of the canonical nine development classifiers, LightGBM,
seed 42 and `subsample85`, retaining all six diagnostic arms: 108 physical
jobs/162 outcomes, eight-hour ceiling, unchanged source data, splits,
budgets and learner settings. Compare primary losses, parameters and
member plans to the archived pre-extraction study, without selecting a
different method or changing thresholds. This is migration verification,
not a new performance or confirmation claim.

The [real-data extraction replay](tabular-results/2026-10-03-classifier-extraction-parity/ANALYSIS.md)
completed all 108 physical jobs/162 prediction outcomes in 3.01883 hours.
All primary and secondary metrics, chosen parameters, inner losses and
member feature plans match the archived pre-extraction results exactly.
The experimental public prediction artifact is therefore parity-verified
on this fixed replay, without changing any earlier performance conclusion.
The cycle now has **20.61484 hours** remaining after 51.38516 measured hours.

The next bounded prior/integration investigation is cross-fitted target
encoding as an explicit **model-input categorical representation**,
with identical raw and FE controls. It is not a new target-derived feature
formula or a license to fit categorical mappings before validation splits.
The smoothing, cross-fitting and dependency-version contract must be
fixed from primary sources before implementation or measurement; existing
one-hot defaults stay unchanged.

Primary-source review fixes the proposed target-encoding policy to
`TargetEncoder(categories="auto", smooth="auto")`, explicit binary or
multiclass targets, and five shuffled stratified folds at the existing
integer seed. The encoder is a model-input transform, never a candidate
formula operand. Multiclass mode requires sklearn 1.4 or later; lazy,
feature-local version checks must leave the base one-hot path importable.
For 1.9+, pass the equivalent explicit splitter instead of deprecated
encoder-level shuffle/seed arguments.

The learner must train on the cross-fitted matrix returned by
`ColumnTransformer.fit_transform(X_train, y_train)`. Calling `fit` then
`transform(X_train)`, or even calling `transform` again after a correct
fit-transform, would use the full-fit target map and defeat cross-fitting.
This applies independently inside every tuning, ranking, validation,
ensemble and calibration training partition. Unknown categories use the
applicable fitted prior; stored full-training maps are for inference,
not a replacement training matrix.

References: [sklearn target-encoding implementation](https://github.com/scikit-learn/scikit-learn/blob/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/preprocessing/_target_encoder.py),
[cross-fitting guidance](https://github.com/scikit-learn/scikit-learn/blob/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/doc/modules/preprocessing.rst),
and [CatBoost ordered target-statistics paper](https://papers.nips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf).
Sklearn's fold encoding is not CatBoost's ordered-statistics/boosting
algorithm and does not inherit its claims. Automatic smoothing alone
does not prevent unique-ID leakage without cross-fitting.

Before measurement, require matrix-provenance spies, fixed-fold target
perturbations, unique-ID complementary-prior checks, exact class/width/
missingness/batch/serialization behavior, and explicit failure for
infeasible five-fold or group/time inputs. No cardinality-based switching,
smoothing grid or fallback encoder is allowed.

The first target-encoding implementation is restricted to IID
classification with explicit importance screening. It changes only the
categorical model-input block; generated operators, raw columns, numeric
caps and learner-search settings stay unchanged. Native CatBoost and
calibration-bundle target mode remain explicitly unsupported rather than
silently changing representation.

Predeclare the four single-model arms on the original nine classification
families: robust raw/OHE, robust raw/target, aligned robust FE/OHE and
aligned robust FE/target; linear/LightGBM, three folds, seed 42, 5,000-row
cap, 1,000 candidates, 50 additions and three selection repeats.
These are **216 cells under a six-hour ceiling**, eight threads,
600 seconds/cell and 4,096 MiB RSS. Match FE to raw within each encoder,
retain failed cells, and report encoded widths separately from feature
plan width. This is development evidence, not a fresh confirmation.

The [target-encoding comparison](tabular-results/2026-10-03-target-encoding-development/ANALYSIS.md)
completed all 216 cells in 3.98840 hours, with all 108 OHE control losses
unchanged. Linear target raw is 16.73% worse than OHE raw on average;
target FE gains 15.52% over that weakened control but only 0.89% over OHE
raw. CKD target FE is 116.86% worse than OHE raw. LightGBM target FE is
essentially unchanged versus its OHE FE counterpart. One-hot stays default.

The cycle now has **16.62644 hours remaining** after 55.37356 measured
hours. The next bounded public-prior question is an OOF-trained convex
blend of raw and FE single-model probabilities, with the entire
hyperparameter/feature procedure inside every OOF training fold.
This is not another sigmoid calibration or target-encoding sweep.
Its exact loss, fitting boundaries and refit-full limitations must be
fixed before implementation or measurement.

The next fixed blend study follows the OOF-meta-fitting/refit-full
distinction used by
[sklearn stacking](https://github.com/scikit-learn/scikit-learn/blob/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/sklearn/ensemble/_stacking.py)
and [AutoGluon weighted ensembles](https://github.com/autogluon/autogluon/blob/2d7d603c7c26234f578d3a593cf466165b649fa1/core/src/autogluon/core/models/greedy_ensemble/ensemble_selection.py).
It is not a reproduction of their meta-model or greedy selection algorithms.

Within each outer training set, use five shuffled stratified folds at
seed `(seed + 1000) mod 2**32`. Each raw/FE fold procedure performs its
entire existing three-setting search and preprocessing inside the fold's
training complement. Each training row receives exactly one OOF
probability vector from each procedure.

Fit one global convex weight `w` by minimizing OOF log loss of
`(1-w) * raw + w * FE`, using bounded scalar optimization on `[0,1]`
with fixed `xatol=1e-5` and `maxiter=500`. Compare both endpoints explicitly,
prefer smaller weights on exact ties, and choose zero for identical
predictions. Invalid probabilities or optimization failure are errors,
not raw-model success fallbacks. The OOF optimum is a meta-training
objective, not independent performance or a no-harm certificate.

Freeze the weight, refit both complete procedures on all outer-training
rows, and score four views of those same full-training models: raw, FE,
fixed 50:50 average and learned blend. Full refitting can change feature
plans, hyperparameters and probability distributions; that mismatch is
explicit rather than denied. No weight, seed or threshold is chosen from
outer-test outcomes.

The first study is **LightGBM classification only**, original nine
development families, three outer folds, seed 42: **27 physical bundles
and 108 prediction outcomes** under a six-hour cumulative ceiling,
eight threads, 900 seconds/bundle and 4,096 MiB RSS. Each bundle includes
48 complete-pipeline fits (ten OOF procedures plus two full refits, each
with three trials and one winning refit); fixed averaging adds no model
fit. All additional work is charged once physically and attributed
explicitly in standalone method costs. One-hot encoding, numeric caps,
feature vocabulary, and learner search stay unchanged.

The [OOF blend study](tabular-results/2026-10-04-oof-blend-development/ANALYSIS.md)
recorded 108 outcomes in 2.75401 hours: 100 scored and eight unscored from
two Dry Bean 900-second bundle timeouts. On the same eight complete
families, learned blending gains 9.45% versus raw but retains 22.92%
worst harm; fixed 50:50 averaging gains 8.58% with worst harm 0.92%.
All successful raw/FE control losses reproduce exactly. No optimizer
or weight-bound tuning follows this failed general-safety hypothesis.

The next fixed study removes unnecessary OOF work only for the
already-declared **50:50 control**, fitting the two full raw/FE procedures
and sharing them across raw, FE and fixed-average views. The old OOF
algorithm, archives and timeouts remain unchanged. Use all nine original
LightGBM classification families, three folds and seeds 42/137/2029:
**81 physical bundles / 243 outcomes**, six-hour ceiling, eight threads,
900 seconds/bundle and 4,096 MiB RSS. No learned weight, calibration,
source change or dataset-specific choice is introduced. This is a new
complete-panel development replication, not a repair of the OOF run.
Current-cycle use is 58.12757 hours, leaving **13.87243 hours**.

The [fixed-average replication](tabular-results/2026-10-04-fixed-average-development/ANALYSIS.md)
completed all 243 outcomes from 81 bundles in 2.45070 hours.
Mean gains versus raw are 7.68%, 6.12% and 5.89% at seeds 42, 137 and
2029; worst-family harm is below 1% in each. The pooled family comparison
gains 6.82% with worst harm 0.26%. Parameters and feature plans reproduce
the old controls; twelve log-loss and six Brier values differ by only
one or two float64 ULPs because of explicit probability validation.
Those differences are preserved, not relabeled exact equality.

Expose this fixed rule only as an explicit experimental public
`raw_fe_average` mode: fit full-training raw and FE member procedures
and average their probabilities with fixed equal weights. Matched
`engineered=False` fits one raw procedure. No OOF, calibration,
weight optimization, implicit model-family policy or default change is
introduced. Preserve old single/subsample85 behavior and verify exact
public-versus-benchmark fixed-average parity before accepting the API.
Current-cycle use is **60.57826 hours**, leaving **11.42174**.

The bounded two-origin metadata follow-up reserved only Thoracic Surgery
(UCI277/OpenML1506/task9969), with 470 records, sixteen reviewed predictors
and the original one-year binary outcome. QSAR Biodegradation's
cross-source compound-overlap hold remains unresolved; it is not declared
fresh. Thoracic remains in the explicitly qualified standard-IID,
unverified-entity tier.

Freeze a small Thoracic-only confirmation of the **same fixed 50:50
rule**, pending exact source/rights/exposure and controller binding:
LightGBM, seed 2031, full rows, three explicitly registered IID folds.
Preserve original official split metadata while validating actual 30/20/10
class-support margins. Compare raw, FE, fixed average, native robust
OpenFE and native robust AutoFeat: nine physical jobs/fifteen outcomes,
four-hour ceiling, eight threads, 900 seconds/job and 4,096 MiB RSS.
Require complete finite coverage, defined relative comparisons, positive
gain versus raw, no greater-than-5% raw harm, and no worse mean than the
other three fixed references. Even passing all scoped criteria would not
authorize clinical/entity-independent, broad SOTA or automatic default
promotion claims on this single origin.

The [Thoracic confirmation](tabular-results/2026-10-04-thoracic-fixed-average-confirmation/ANALYSIS.md)
recorded all fifteen outcomes in 0.34997 hours: fourteen scores and one
AutoFeat NaN-input failure. Fixed averaging scores 0.508215 versus
raw/OpenFE 0.510180 and single FE 0.520143, but the locked decision is
**blocked by incomplete AutoFeat coverage**. Preserve that failure and
do not report a passed confirmation from the remaining methods.

A bounded local dependency investigation may determine whether the
AutoFeat failure is a narrow numerical compatibility defect. Any remedy
must be separately named and validated, without silently altering the
archived reference or promoting the candidate from an incomplete result.
Current-cycle use is **60.92824 hours**, leaving **11.07176**.

The compatibility adapter is implemented with preserved upstream MIT
notices and pinned source/dependency checks. A focused review found no
significant issues; constant/nonconstant/nonfinite scaling, unchanged
target scaling, private bindings and spawned-worker behavior are tested.
Run all three archived Thoracic splits once with the separately named
reference, within a **one-hour diagnostic ceiling**, preserving the
original prepared/source/result checksums and restricted tier.
This is retrospective compatibility measurement, not a new confirmation
or replacement for the original blocked decision.

The dependency diagnosis confirms that the failing AutoFeat selection
frame already contains all-one original indicators **before** its 85%
resamples. Native feature standardization divides those zero-variance
columns by zero. Resampling an already finite standardized frame was
not the source of NaNs in this case.

Implement only a separately named
`autofeat-native-safe-scale-v1-robust` compatibility reference: preserve
the pinned feature generation, transformations, sampling, selection and
output width, but use denominator one for wholly finite, actually constant
feature/noise columns with exactly zero standard deviation. No epsilon,
feature dropping, target-scaling change or masking of unrelated NaN/inf
errors. A private pinned adapter must retain upstream MIT/author notices
and avoid global monkeypatches.

Any three-fold comparison on the existing Thoracic payload is a
retrospective compatibility diagnostic, not new confirmation. The native
failure, original 14/15 result and authoritative blocked decision remain
unchanged; repaired-reference outcomes cannot be silently substituted.

The [separate compatibility diagnostic](tabular-results/2026-10-04-autofeat-safe-scale-diagnostic/ANALYSIS.md)
completed **3/3 folds** in **0.124644 whole-process hours**. Exact
prepared/split identities, original artifacts, source/dependency pins
and bounded archive reconstruction were verified. Safe-scale AutoFeat
scores **0.503197 mean log loss**, versus the archived fixed average's
**0.508215**; this comparison does not establish superiority for
FeatCopilot. Scores and widths on native-successful folds differ, so
neither exact output parity nor a causal accuracy effect is claimed.
The original **14/15, BLOCKED** decision remains unchanged. This is
retrospective qualified-tier evidence, not fresh confirmation.
Current-cycle use is **61.05288 hours**, leaving **10.94712**; no new
LLM calls were made. Do not consume that remainder on a post-result
threshold sweep or reclassify this exposed origin as confirmation.

The read-only discrepancy audit found matched seeds, preprocessing,
dependencies and search settings, but different pre-learner selections.
AutoFeat's preserved set-to-list operations expose a concrete Python
hash-order dependence; historical launchers neither pinned nor recorded
that seed. No evidence isolates it from guard effects in the observed
loss differences. The shared future-run launcher now pins and records
`PYTHONHASHSEED` from each cell seed, preserving it through calibration
and blending outcomes and rejecting mismatched metadata. Regression
coverage verifies actual fresh-interpreter hashes/order, inherited-
environment isolation, seed bounds and unscored worker failures.
All 680 affected launcher/controller/resume tests passed, as did the
additional real-worker error-provenance check and package typing.
This is an infrastructure correction, not a change to the archived
compatibility/native algorithms or scores. No additional real-data
benchmark fits or LLM calls were used.

The [higher-retention study](tabular-results/2026-10-03-subsample85-development/ANALYSIS.md)
completed 486/486 outcomes in 9.12791 hours, with all 162 single controls
reproduced exactly. Across three seeds, FE85 gains 7.60% over raw85 and
7.31% over single raw, with combined worst-family harms 2.59% and 4.95%.
Per-seed CKD harms nevertheless reach 28.97% versus raw85 and 22.50%
versus single raw; no safety guarantee or default promotion follows.
Current-cycle use is **46.62553 hours**, leaving **25.37447**.

Freeze the learner-aware policy for qualified confirmation: linear single
robust FE and LightGBM plain FE85. Mammographic Mass and Haberman remain
the entire predeclared two-origin cohort in the explicitly approved
standard-IID/unverified-entity tier. Use seed 2030, three IID folds,
full rows, existing search/feature limits, eight threads, 900 seconds/job
and 4,096 MiB RSS, with a ten-hour cumulative ceiling.
Include matched single/ensemble raw controls and unchanged native-width
OpenFE/AutoFeat robust single references; calibrated views are diagnostic,
not candidates. This is 72 physical jobs and 96 prediction outcomes.
Release only after binding the complete cohort, source identities,
sampling mode and restricted tier in every result/report. Never pool this
tier into verified-entity or clinical-generalization evidence.

Before any additional source acquisition, the user explicitly approved
a **separate standard-IID benchmark tier** for public datasets lacking
verifiable patient/entity identifiers. This does not establish patient/
entity-independent, prospective clinical or deployment generalization.
Primary licenses, target exclusions and train/test leakage safeguards
remain unchanged. Known available group/time boundaries are not silently
ignored; sources with conflicting rights remain blocked. Old strict-tier
audits are preserved, and the tiers must not be pooled into a stronger
claim. Metadata-only reconsideration is limited initially to Mammographic
Mass and Haberman under that explicitly qualified scope.

The [qualified-tier reservation](suites/standard_iid_entity_reservation_20261003.json)
and [audit](suites/standard_iid_entity_reservation_audit_20261003.json)
now contain those two sources as active metadata-only inventory.
Mammographic Mass excludes BI-RADS and retains its four reviewed
predictors; Haberman keeps its original three predictors and five-year
endpoint. Both retain primary CC-BY-4.0 evidence and exposure `unknown`.
No payload/model access or confirmation release has occurred.
Future admission and reporting must explicitly enforce this tier's
restricted claims; merely preserving an `evaluation_tier` field in catalog
metadata is not sufficient to claim report isolation.

Implementation details preserve the fixed proposal boundary: remove
scope-level exact constants/duplicates before resampling, then retain that
definition universe without regeneration or budget refill. Bag-local
constant/duplicate/absent candidates contribute zero and tied worst ranks;
their identities are mapped back to the shared universe. Constant-target
regression bags remain legitimate fits, while infeasible classification
bags fail explicitly. Single-fit behavior and global target validation
remain unchanged.

Research and integration design come first. Keep a single fixed resampling
configuration, the existing candidate vocabulary, raw-feature preservation,
and explicit matched robust baselines. Do not claim formal false-selection
guarantees without satisfying their assumptions. Evaluate loss, tail harm,
coverage and cost alongside feature-set stability; improved Jaccard alone
is not sufficient.

A new metadata-only reservation targets twelve genuinely distinct
observational origins (six regression, three binary, three multiclass),
with explicit rights, timing, schema and historical-exposure checks.
Eligibility is fixed before acquisition; unsupported candidates remain
pending rather than being force-admitted. No reserved family may be modeled
until the final algorithm and comparison protocol are frozen.

The bounded 30-origin metadata audit produced a
[four-family reservation](suites/cycle2_origin_reservation_20260929.json)
and [audit](suites/cycle2_origin_reservation_audit_20260929.json):
Facebook Cosmetics Posts (regression), Differentiated Thyroid Recurrence
(binary), Assam Medical Entrance and Personality Drug Consumption
(multiclass). All have primary UCI CC-BY-4.0 evidence and remain
active/inventory-only, with exposure unknown pending formal release.

There was no quota redistribution: shortfalls are five regression, two
binary and one multiclass origin. Unresolved entity/time boundaries,
schemas/support, simulations, size limits and restrictive source terms
remain blockers. A narrow follow-up may resolve documentation-only blockers
for at most five already-reviewed regression origins; it must not weaken
the eligibility rules or reinterpret a restricted license to fill the quota.
This smaller reservation cannot support a broad regression-SOTA claim.

### Renewed robustness work: 2026-09-29

The user requested continued improvement and learning from other packages.
The next fixed hypothesis is **training-support capping of numeric model
inputs**, not another feature-family expansion. Existing defaults stay
unchanged and the remaining **6.20662 measurement hours/eight-thread limit**
remain binding; no additional live LLM calls are planned.

Primary-source review distinguishes several superficially similar approaches:

| Upstream mechanism | Relevant lesson | Why not copy its whole pipeline |
|---|---|---|
| [Feature-engine fitted caps](https://github.com/feature-engine/feature_engine/blob/d866312874a66364c6dee3c247e69476b478395c/feature_engine/outliers/base_outlier.py) | Learn numerical bounds during fit and reuse them at transform | Quantile winsorization also modifies training tails; missing/constant policies differ |
| [sklearn MinMaxScaler clipping](https://github.com/scikit-learn/scikit-learn/blob/857849927da6e988d7d026b17aef214d43e5f26e/sklearn/preprocessing/_data.py#L529-L587) | Separate learned training support from inference clipping | Replacing StandardScaler with min-max scaling would change Ridge regularization geometry |
| [sklearn robust/quantile transforms](https://github.com/scikit-learn/scikit-learn/blob/857849927da6e988d7d026b17aef214d43e5f26e/sklearn/preprocessing/_data.py) | RobustScaler does not bound unseen extremes; QuantileTransformer does | Quantile mapping changes interior geometry, a larger intervention |
| [AutoFeat repeated selection](https://github.com/cod3licious/autofeat/blob/639793a6e620f2b8da21f32620a1c94d1d7c50bc/src/autofeat/featsel.py) | Selection-frequency accounting can describe stability | Its preprocessing precedes internal resampling; do not copy that boundary |
| [OpenFE transform](https://github.com/IIIS-Li-Group/OpenFE/blob/3e3bdf9f5b972e214e018c22f6f9f456a47b416e/openfe/utils.py) | Replacing infinities does not constrain very large finite predictors | Public global-operator transforms concatenate train/test; preserve strict inductive behavior here |

The inspected Feature-engine/sklearn snapshots are BSD-3-Clause;
AutoFeat/OpenFE are MIT. This work reuses FeatCopilot's existing bounds
estimation mathematics rather than copying upstream code.

Choose **min/max bounds (`quantiles=(0, 1)`)** for the first ablation.
This preserves training/interior values in ordinary finite ranges and
isolates out-of-training-support saturation, rather than also clipping
the 1%/99% training tails. It retains the existing numeric safety guard,
missing-value policy and downstream scaler. Each internal training fit
and final learner fits a separate clipper after candidate generation;
categorical columns and feature definitions remain unchanged.

The robust arms preserve Boolean dtype through capping, then retain numeric
Boolean encoding in the learner, consistently with predictive selection.
They share the selection engine's finite guard before bounds estimation.
For values beyond that guard, raw-to-robust changes include guarding as well
as support saturation; only robust-FE versus identically robust-raw provides
the intended matched FE attribution.

The fixed comparison is a 2-by-2 design on the original sixteen development
families: raw, capped raw, frequency-aligned, and capped frequency-aligned.
Use linear/LightGBM, three folds, seed 42, a 5,000-row cap, the existing
three-setting learner grids, 1,000 candidates/50 additions/three repeats,
eight threads, 600 seconds per cell and 4,096 MiB sampled RSS.
These are **384 cells with a five-hour cumulative ceiling**.

After that run, a separately bounded **24-cell, half-hour** retrospective
check may use the exact six pre-existing hardware/temperature training-only
audit partitions with the same four methods. No original outer-test data
enter fitting or scoring. Its configuration is fixed now, not chosen from
the new development outcomes. Runs are sequential to preserve the shared
eight-thread ceiling; together their maximum 5.5 hours fit the remaining
measurement allowance.

Report both loss reduction against original raw and incremental FE value
against **capped raw on identical complete splits**. Also report family-level
harm, coverage, cost, selected-count dispersion and pairwise selected-name
Jaccard across folds. A preprocessing-only gain must not be relabeled FE
superiority. Before considering another confirmation, require positive
matched-baseline mean gains, no new greater-than-5% family harms, and no
aggregate stability deterioration. This development panel cannot establish
SOTA or authorize default promotion.

For stability, compute selected-generated-name Jaccard for each of the
three outer-fold pairs (two empty sets score one), then average within each
family before the task/learner mean. Report population standard deviation
of selected counts by family as well. Compare capped aligned FE with
uncapped aligned FE; record any task/learner mean Jaccard decline explicitly.
These name-based diagnostics do not measure equality of learned mapping
state or prove prediction stability.

Train min/max caps preserve contaminated training extremes and suppress
legitimate extrapolation. Marginal bounds do not guarantee joint robustness,
stable feature identities, safe symbolic arithmetic before clipping, or
bounded prediction error. Those remain explicit limitations.

**Operational completion amendment:** the initial five-hour run produced
351 successful cells, 32 budget-exhausted cells and one timeout shortened
by the remaining global allowance. Final reporting brought measured elapsed
time to 5.06510 hours. Partial means have different family coverage and are
inconclusive; they are not grounds for choosing a winner.
The exact first-phase results, manifest, state and report are preserved in
`2026-09-29-robust-development/phase1_checkpoint.json` and its lossless parts.
Increase only the cumulative operational ceiling to **5.6 hours** and retry
all 33 unscored cells under unchanged source, splits, methods, learner grids,
600-second case caps, RSS and thread limits. Successful cells are not rerun.
The optional audit is limited to 1,500 seconds and may start only after
reconciling actual remaining time against the user's 72-hour ceiling.

The [completed four-arm comparison](tabular-results/2026-09-29-robust-development/ANALYSIS.md)
has 384/384 successful cells in 5.51402 cumulative hours. All 351 initial
successes were reused unchanged, and all 192 uncapped control losses match
the preceding run exactly. Capped FE reduces loss versus capped raw by
**16.12% linear classification and 14.97% linear regression**. Every
LightGBM primary loss remains exactly unchanged. No new greater-than-5%
family harm appears, but linear-classification selected-name Jaccard falls
from 0.460 to 0.389, failing the predeclared stability criterion.

Keep the shared capper opt-in; no default promotion or new confirmation is
justified yet. Actual cumulative measurement is 71.05739 hours, leaving
0.69261 hours after the probe allowance. The predeclared training-only
outlier audit is authorized for at most 1,500 seconds, without any code or
method changes.

The [robust training-only audit](tabular-results/2026-09-29-robust-training-audit/ANALYSIS.md)
completed 24/24 cells in 947.45 measured seconds including publication.
All prior audit partitions and twelve nonrobust controls reproduce exactly.
The extreme temperature audit fold improves from raw RMSE **1.541932** to
capped-raw **0.264521**, with capped FE **0.256874**. Its original Distance
predictor is capped from standardized magnitude 1174.0868 to the training
maximum 2.9859, confirming the intended mechanism on that audit partition.
Most of this benefit is preprocessing, not FE.

Against matched capped raw, family-mean FE gains are **15.12% hardware**
and **2.85% temperature**. Capped hardware FE nevertheless worsens 12.89%
versus the uncapped aligned representation. Keep this tradeoff explicit;
the overlapping training-only audit is not fresh confirmation.

Cumulative measured execution is now **71.32057 hours**; with the separate
0.25-hour operational probe allowance, **0.42943 hours remain**.
No further measurement is scheduled within this cycle. The next candidate
research priority is training-only selection-frequency/consensus methods
inspired by AutoFeat, preserving honest validation of the final proposed
representation rather than merely filtering features after subset scoring.
This is a research hypothesis, not an implemented fix or a guarantee.
Any substantial next comparison needs an explicit new resource allocation
and an independently reserved cohort before confirmation.

### Continuing development cycle: 2026-09-22

The user authorized **another 72 measurement hours at eight CPU threads**.
The total live-LLM allowance remains **100 attempts**, with 96 already used;
no new live LLM comparison is scheduled in this cycle.

The new admission work provides 50 development/stress origins (48
observational, two domain-simulation), including mixed/imbalanced, multiclass,
wide, grouped and temporal tasks. This is not a claim of 50 fresh confirmation
datasets. The six previously evaluated confirmation families are exposed.
Unresolved inventory and source-specific resource holds remain explicit.

Implemented next-cycle changes:

- Full-operator inductive OpenFE and native-default AutoFeat adapters, with
  explicit learned-statistic, categorical and output-width policies.
- Training-defined hybrid namespaces, preserving both feature sources and
  leaving malformed-plan failures and the call ledger intact.
- An opt-in paired-gain selection rule. Internal development diagnostics
  motivate the hypothesis; they are not external performance evidence.
- An output-preserving fast path that skips final all-data rescreening when
  selection keeps no generated features.

The first development comparison is fixed before inspecting its outcomes:
the seven original UCI tasks, Wisconsin Diagnostic, Spambase, Banknote,
Combined Cycle Power, Sindian Real Estate, Auto MPG, Chronic Kidney Disease,
and cached houses/wine diagnostics. Use three folds, seed 42, a 5,000-row cap,
linear/LightGBM learners, and raw, standard predictive, paired predictive,
full-operator width-matched OpenFE and native-default AutoFeat representations.
These are 480 paired evaluation cells. Reserve at most 36 hours for this
comparison; later studies remain within the new 72-hour allowance.

This development panel deliberately excludes unsupported temporal/grouped
comparators and does not cherry-pick successful folds. Native AutoFeat width
and width-matched methods are labeled separately. Legacy/default selection
is not promoted, and new independent confirmation must be reserved only after
the next candidate is frozen.

The first 480-cell comparison is preserved in
[its development analysis](tabular-results/2026-09-22-full-reference-development/ANALYSIS.md).
It motivates the next fixed comparison on the **same 16 exposed development
families**: raw, standard, paired, extended, extended-paired and OpenFE-inductive;
linear/LightGBM, three folds, seed 42, unchanged row/output/search caps. These
are 576 cells with a 30-hour ceiling, leaving further room within the newly
approved 72 hours. No source-specific formulas or performance-selected family
substitutions are introduced. The separate metadata-only candidate reservation
remains closed to models.

After that comparison, cumulative measurement time is 21.85 hours. Extended
candidates alone did not close the boosted-model gap and worsened Chronic
Kidney Disease/linear substantially; those results remain preserved.

The next preregistered **development-only** comparison keeps the identical
16-family panel, three folds, seed 42, row/feature/search limits and
linear/LightGBM learners. Compare raw, standard predictive, joint-importance,
joint-importance-paired and OpenFE-inductive: 480 cells, at most 20 hours, eight
threads and 600 seconds/cell. The joint variants use the extended vocabulary
with exact training-only deduplication. All candidate tuning remains inside
training partitions. No new LLM calls or reserved confirmation outcomes are
used to select this experiment.

That [joint-importance comparison](tabular-results/2026-09-23-joint-importance-development/ANALYSIS.md)
completed 480/480 cells in 9.78235 hours. Cumulative current-cycle measurement
is **31.63657 hours**, leaving **40.36343 hours**. Importance-paired narrows the
boosted-classification gap and improves regression means, but Wisconsin
Diagnostic/LightGBM worsens by 22.00% and CKD/linear by 9.85%. No default changes.

The next step freezes **one importance-paired configuration**, without
task/family-specific switching, against raw features, OpenFE-inductive-native
and AutoFeat-native on all 12 reserved families. The separately locked protocol
uses full rows, seed 2026, linear/LightGBM, three fixed learner candidates,
1,000 feature candidates and 50 additions. Select official repeat=0/fold=0/sample=0
before outcomes for the ten OpenML sources, plus three IID folds for each of
the two UCI sources: 128 cells. Ceiling: 38 active hours, eight threads,
4,096 MiB sampled RSS and 900 seconds/cell. No new LLM calls.

This narrower split scope is declared for resource feasibility, not presented
as a full TabArena reproduction. Require positive mean loss reduction versus
raw, no observational family harm greater than 5%, and no worse mean than
complete-pair strong references in each task/learner stratum before considering
promotion. Missing comparator coverage prevents a broad superiority claim;
family-bootstrap intervals remain descriptive. Independent confirmation is
not an automatic SOTA certificate, even if those gates pass.

The [executable confirmation suite](suites/final_reserved_confirmation_20260924.json)
and [release audit](suites/final_reserved_confirmation_release_20260924.json)
now lock that protocol. All 12 families passed preparation, including actual
selected-split class-support checks, with 16 splits and 128 planned cells.
No model calls occurred during preparation. Runtime source is frozen at
`886f18e9116e7dd7e22903b17d93218c739a24761927266d2df079e87e74422e`;
results will be recorded separately, not used to revise this candidate.
The previous metadata-only draft and reservation history remain intact.

The [independent outcome](tabular-results/2026-09-24-frozen-native-confirmation/ANALYSIS.md)
is complete: 128 attempted, 120 scored, all candidate/raw/OpenFE cells successful.
AutoFeat-native had six RSS failures and two timeouts; none was substituted.
Candidate linear classification improved by +7.09%, but boosted classification
(+1.13%) remained behind OpenFE (+9.39%), and observational boosted regression
worsened by -0.95%. The preregistered quality/coverage gates **failed**.
Default promotion and SOTA claims remain blocked.

This run used 4.26919 hours, bringing current-cycle consumption to **35.90575
hours**, with **36.09425 hours** left. The twelve families are now exposed.
Further work returns to development: diagnose missing candidate operators,
selection-budget coverage and classification fitting cost on the original
16 development families; choose one bounded next change, not a bundle of
confirmation-specific fixes. Any subsequent independent measurement needs a
new metadata-audited, origin-disjoint reservation.

The first post-confirmation change is an **efficiency-only** ablation, not
another accuracy variant. Original development records show mean inner
feature-fit time of 100.58 seconds for linear classification versus 1.88
seconds for linear regression; even German Credit's zero-addition fits
incurred the classification cost. The benchmark's LogisticRegression starts
with `n_jobs=None`, but predictive selection overwrites it with eight.
In the recorded sklearn 1.7.2/joblib 1.3.2 combination, default LBFGS
dispatches its single binary/multinomial optimization task through loky.
Avoiding that unnecessary process pool must preserve the caller, solver and
all statistical settings, native-thread cap, and explicitly requested
one-vs-rest behavior.

Verify the narrow fix with fresh-process synthetic parity plus a fixed
18-cell replay on original-development German Credit, Banknote and Chronic
Kidney Disease: raw and importance-paired, linear only, three folds, seed 42,
5,000-row cap, original feature budgets and eight threads. Compare losses
and selected feature names with the archived joint-importance run before
claiming output preservation or speedup. Do not combine this change with
new operators, output-width changes or default promotion.

A [new metadata-only reservation](suites/fresh_origin_reservation_20260924.json)
and [source audit](suites/fresh_origin_reservation_audit_20260924.json) identify
ten additional observational origins, without quota redistribution:
four regression (Communities and Crime, Computer Hardware PRP, Forest Fires,
Infrared Oral Temperature), three binary (Australian Credit, Indian Liver
Patients, Ionosphere), and three multiclass (Contraceptive Method Choice,
Vertebral Column, Wall-Following Robot). Canonical-origin checks distinguish
Hardware PRP from previously exposed CPU activity and consolidate Australian
credit variants. Nine sources have primary UCI CC-BY-4.0 grants; temperature
uses its original PhysioNet CC0 grant.

These remain **active reservations, not executable confirmation**. Registry
exposure is `unknown`, not an unsupported unseen certification. No payloads
or models were acquired; payload integrity, actual fold class support and
final exposure/protocol checks remain required before release. This inventory
does not authorize changing a frozen experiment or claiming ten successful
benchmark tasks.

The [18-cell efficiency replay](tabular-results/2026-09-24-logistic-worker-efficiency/ANALYSIS.md)
is complete: every loss, chosen learner setting and inner/outer selected
feature list matched the archived cells exactly. Mean pipeline time fell
from 97-104 seconds to 2.4-7.4 seconds, excluding some supervisor startup
overhead. This narrow worker fix is retained without changing feature
algorithms or defaults. Dataset measurement now totals 36.05148 hours;
after a separate 0.25-hour operational allowance for synthetic timing probes
(220.05 seconds of timed fits, other overhead unknown), 35.69852 hours remain.

The next accuracy hypothesis adds **only bounded unary-frequency candidates**
to extended/importance/paired selection. Original-development OpenFE records
retained frequency definitions 63 times across eleven families; retention
alone does not establish causal value. Each mapping must be fit on the
current internal training partition, remain frozen at validation/inference,
and share the fitted-rule engine's key/missing/serialization semantics.
Pair frequencies, group statistics, selection thresholds and the effective
50-addition limit are deliberately unchanged.

Predeclare 384 development cells on the same original 16-family panel:
raw, importance-paired, frequency-paired and width-matched OpenFE-inductive;
linear/LightGBM, three folds, seed 42, 5,000-row cap, 1,000 candidates,
50 additions and three internal repeats. Limit the run to **12 hours**,
eight threads, 600 seconds/cell and 4,096 MiB sampled process-tree RSS.
Use a new reviewed replay manifest for the original fourteen UCI definitions
plus cached houses/wine, not the now-stale whole expansion manifest.
All ten newly reserved origins remain closed to data/model acquisition.
Freeze code before running; retain harms and failures rather than selecting
datasets or changing modes from these outcomes.

The [frequency ablation](tabular-results/2026-09-24-frequency-development/ANALYSIS.md)
completed 384/384 cells in 8.03297 hours; all 288 control losses reproduced
exactly. Frequency-paired raises observational mean loss reduction to
15.27% for linear classification and 2.89% for boosted regression, while
slightly reducing boosted-classification and linear-regression means.
Wisconsin Diagnostic/LightGBM harm remains 22.76%, so default promotion is
still blocked. Dataset measurements total 44.08445 hours; 27.66555 hours
remain after the separate synthetic-probe allowance.

Freeze this single configuration, without task-specific switches, for the
ten new reserved origins after final metadata/exposure admission. The next
locked comparison uses raw, frequency-paired, OpenFE-inductive-native and
AutoFeat-native with **linear, LightGBM, CatBoost and XGBoost**, matched
existing three-setting grids, full rows and seed 2027. Use three IID folds
or exact official repeat=0/fold=0/sample=0, as declared before outcomes.
Ceiling: 24 hours, eight threads, 900 seconds/cell and 4,096 MiB sampled RSS.
No LLM calls. Preserve the same conservative quality/harm/coverage gates;
do not treat another independent run as automatic evidence of SOTA.

**Pre-acquisition split amendment:** metadata revealed that Vertebral Column's
official ten-fold task supplies only six minority-class test rows, below the
locked minimum of ten. The
[blocked-protocol audit](suites/frequency_fresh_confirmation_blocked_audit_20260924.json)
is preserved. Before any payload/model access, the user explicitly chose
**three-fold IID evaluation for the entire ten-family cohort**, retaining
the ten-example minimum rather than weakening it.

All sources, features, targets, methods and learner grids remain unchanged.
The revised plan has **30 splits and 480 cells**, seed 2027, under the same
24-hour ceiling. It is not an official-split leaderboard reproduction.
Official source identities and provided split metadata remain recorded,
while actual evaluation uses explicitly registered, training-only IID folds.
The total budget can still leave cells unscored; report those failures or
budget skips without substituting families or reducing the protocol after
results are known.

The [amended executable suite](suites/frequency_fresh_confirmation_iid_20260928.json)
and [release audit](suites/frequency_fresh_confirmation_iid_release_20260928.json)
are now frozen. All ten sources passed preparation and actual class-support
checks for the thirty generated IID splits, with 480 planned cells.
The eight OpenML sources retain all ten original official split identities
as provenance, but none is mislabeled as an evaluation split.
Runtime source:
`c30bfbfcacb2fc7d3c3398699d8b2ee67ace00807cecaf2de49c7c06f527a808`.
No model calls occurred during preparation; results will be evaluated against
the frozen gates rather than used to revise this candidate.

The [ten-family outcome](tabular-results/2026-09-28-frequency-iid-confirmation/ANALYSIS.md)
is complete after 17.36022 hours: all 480 cells attempted, 435 scored.
Raw/candidate each complete 120/120; references have 45 retained failures.
**Promotion gates fail materially**: four of eight candidate task/learner
means are negative, and nine of forty family/learner losses worsen more
than 5%. Hardware/linear RMSE rises 61.25%, temperature/linear 45.61%.
Positive linear-classification mean (+15.41%) does not offset those failures.

All ten families are now exposed. Stop expanding the feature catalog until
a bounded retrospective diagnosis distinguishes execution/numerical bugs
from extrapolation, selection optimism and refit/tuning mismatch. Use
archived training-selection evidence first; do not invent task-specific
thresholds or reuse this cohort as fresh confirmation for a fix.
The current cycle has 10.30533 hours left after 61.44467 dataset-measurement
hours and the separate 0.25-hour probe allowance. Defaults remain legacy;
the broader SOTA goal is explicitly **not achieved**.

The retrospective archive trace found no stale-frequency execution bug:
the two catastrophic final linear models contain no frequency features.
It does establish representation/regularization changes between tuning and
refitting, plus unstable internal losses; the causal contribution of each
mechanism is still unmeasured.

A separate confirmed benchmark defect is corrected: predictive/plan/hybrid
inputs no longer receive a whole-input median before their internal splits.
Numeric missingness reaches the training-local preprocessors. This cannot
explain hardware's failure, which has no recorded missing values, and no
historical score is amended.

Run one bounded **retrospective training-only audit** on all three original
outer-training partitions of hardware and temperature. Original outer-test
data remain outside every fit, tuning step and audit score. Predeclare an
audit holdout inside each training partition, compare fixed Ridge alpha=1
with the training-selected alpha on the same final engineered
representation, and record schema/state changes, coefficient norms,
standardized-feature extremes and aggregate prediction contributions.
Use unchanged frequency/selection settings and pinned dependencies, eight
threads, no new data/LLM calls, and a one-hour audit ceiling. This diagnoses
failure mechanisms; it neither tunes a new winner nor supplies fresh
confirmation.

The [training-only audit](tabular-results/2026-09-29-training-selection-audit/ANALYSIS.md)
completed all six cells in 263.40 seconds, without scoring original
outer-test rows. Hardware's catastrophe was not reproduced. Temperature's
third audit holdout contained an original Distance value with standardized
magnitude 1174.09 versus a training maximum of 2.99. On the same engineered
representation, alpha=10 improved RMSE relative to alpha=1 but remained
12.12% worse than raw. This supports investigating extrapolation and
pipeline alignment, not claiming a causal fix for the original failures.

The final bounded development ablation in this allowance will **align
selection-estimator parameters with the learner being tuned**. For each of
the same three learner settings, refit feature engineering on the same
inner-training rows using that setting, score its matched learner on inner
validation, and refit the chosen configuration on outer training. No
outer-label choices or new search values; all extra feature fits count
toward the budget. Existing methods and defaults remain unchanged.

Compare raw, frequency-paired and the separately named
`featcopilot-frequency-aligned` on the original sixteen development families:
linear/LightGBM, three folds, seed 42, unchanged row/candidate/width/repeat
limits, eight threads, 600 seconds/cell and 4,096 MiB RSS. These are
288 cells with an **eight-hour ceiling**. No fresh-family confirmation
or SOTA claim is authorized by this development result. After the audit
and separate probe allowance, 10.23216 hours remain in the current cycle.

The [aligned ablation](tabular-results/2026-09-29-aligned-development/ANALYSIS.md)
completed 288/288 cells in 3.85811 hours. Linear classification improves
from -0.30% paired to +14.96% aligned, driven partly by correcting a large
CKD/linear failure after training-local imputation. Three other aggregate
strata worsen slightly, and Wisconsin/LightGBM harm remains 22.76%.
All raw controls reproduce exactly; historical predictive missing-data
scores cannot be treated as unchanged controls after the imputation fix.

Measurements total 65.37595 hours, leaving 6.37405 hours after the separate
probe allowance. The final bounded retrospective check uses the exact six
hardware/temperature training-only audit partitions to compare
raw/paired/aligned, with a one-hour ceiling. It does not touch original
outer-test rows or confer fresh-confirmation status. No additional operator
search, task-specific thresholds, default promotion or SOTA claim follows
from this development experiment.

The [final training-only comparison](tabular-results/2026-09-29-aligned-training-audit/ANALYSIS.md)
completed 18/18 cells in 602.74 seconds. Every audit split matched the
previous study, and raw/paired controls reproduced their prior losses,
parameters and selected names exactly. Alignment improves hardware mean
RMSE by 16.03% versus paired but worsens temperature by 17.73%; the outlier
audit fold expands from five to fifty additions while retaining alpha=10.
This rejects alignment as a general remedy for the observed regression
instability. Neither audit evaluates original outer-test rows.

### End-of-cycle evidence and unresolved work

The current authorized cycle used **65.54338 actual measurement hours**.
Including the separate 0.25-hour operational allowance for synthetic timing
probes, **6.20662 hours remain unused**; no further experiment is scheduled.
The shared live-LLM ledger remains at **96/100 attempts**. All run-level
failures and failed promotion decisions are preserved.

Implemented improvements include typed fitted features, explicit task and
split handling, expanded rights/exposure-aware catalogs, reproducible
resource-bounded comparisons, training-local predictive preprocessing,
the measured LogisticRegression worker optimization, and opt-in importance,
frequency and aligned-pipeline experiments. These engineering improvements
do **not** establish a state-of-the-art feature-selection algorithm.

Remaining acceptance requirements are substantive:

- A general, training-defined strategy for extrapolation and selection/refit
  instability, without dataset-specific exceptions or post-test fallbacks.
- A new origin-disjoint, predeclared confirmation cohort for any revised
  algorithm; every previously measured confirmation family is now exposed.
- Competitive quality, tail harm, completion and cost on stronger learners,
  not only a favorable linear-classification mean.
- Separately scoped full paper/reference reproductions and unresolved
  source-rights/domain work; inductive adaptations and bounded native-width
  results must not be relabeled unrestricted SOTA comparisons.

Defaults remain legacy. The SOTA goal and those acceptance requirements
remain **unmet**, rather than being marked complete because the allotted
experiments finished.

### First implementation increment

The following foundation is implemented; measured pilot outcomes are tracked
separately and do not imply completion of the full roadmap:

- Catalog/provider admission with family aliases, exposure and rights gates;
  seven reviewed UCI development candidates; blocked Seoul/TabArena inventories.
- Numerical parity, repeated stability-aware predictive selection, linear/tree
  screening and shared group/time split constraints.
- Typed fitted statistical rules and an opt-in LLM plan path; provided plans
  need no provider and keep learned statistics fixed at inference.
- Protocol-v3 per-learner cells, strict resume fingerprints, cumulative active
  budgets, corruption-checked checkpoints, runtime exposure tracking, automatic
  archival, family/role-aware summaries, CatBoost and XGBoost baselines.

The [foundation pilot](tabular-results/2026-09-17-roadmap-pilot/ANALYSIS.md)
completed all 1,512 requested cells in 26.18 active hours; only AutoFeat's 48
timeouts were unscored. The known severe linear regressions improved materially
on matching diagnostic folds. Defaults remain legacy, and the broader
confirmation/domain/LLM phases below are still outstanding.

Still outstanding: broad confirmation admission/evaluation, complete temporal
label-horizon/embargo policies, supervised grouped/temporal target encoding,
relational benchmark integration, and separately approved costed live-LLM
comparisons. Pending data rights are not overridden by the implementation approval.

### Evaluated implementation status

The subsequent increment implements label-availability purging, supervised
group/temporal target-mean cross-fitting, a frozen-history point-in-time
relational engine, and nine-operator typed LLM plans with durable call budgets.
The linked results preserve both useful gains and limitations:

- [Confirmation](tabular-results/2026-09-21-confirmation-completion/ANALYSIS.md):
  six families, all 1,440 scored cells after a disclosed categorical-schema
  completion repair; classification essentially neutral, regression gains
  concentrated on Airfoil. Defaults remain legacy.
- [LLM development](tabular-results/2026-09-20-llm-pilot/ANALYSIS.md):
  useful Concrete/linear gains, no general boosted-model advantage, two hybrid
  failures retained; 96 logical attempts including readiness checks.
- [Temporal development](tabular-results/2026-09-20-seoul-temporal/ANALYSIS.md):
  all 18 cells complete; retrospective observed-weather estimation only.
- [Relational development](tabular-results/2026-09-21-retail-relational/ANALYSIS.md):
  expanded point-in-time history reduced mean RMSE by 5.37% for linear models
  and 15.37% for LightGBM in the fixed retail cohort.

The cycle used about 46.65 active measurement hours, within the 72-hour ceiling.
The paragraph above records what was outstanding after the first increment,
not a claim that these later capabilities are still absent.

**Not completed or claimed:** a fully rights/schema-cleared 40-60-family
benchmark, unrestricted FE/LLM competitor reproductions, official RelBench
evaluation, joint group-and-time constraints, or SOTA/default promotion.
The 51-entry TabArena inventory still has unresolved admissions; these cannot
be converted into approved confirmation by changing a flag. Remaining
algorithm research includes efficiency/stability tradeoffs and robust hybrid
feature-name composition. These stay explicit follow-up work, not hidden
behind the completed bounded evaluations.

### Next evaluation increment

The first pilot consumed 26.18 of the approved 72 active measurement hours.
Before inspecting any new outcomes, the remaining allocation is:

| Track | Frozen cohort and comparison | Ceiling |
|---|---|---|
| LLM development | Online Shoppers and Concrete; 3 folds, seeds 42/137; linear/LightGBM; raw, predictive, typed LLM and hybrid | 3 hours; at most 100 logical generation attempts across smoke/retries/runs |
| Temporal development | Reviewed Seoul day-bucket variant; 3 forward splits, seed 42; linear/LightGBM; raw, predictive, polynomial | 6 hours |
| Locked confirmation | Six audited TabArena task families; all 180 official splits, estimator seed 2026; four learners; raw vs predictive | 36 hours; 600 seconds/cell |

The user explicitly relaxed the hard 4,096-output-token requirement after
provider-capability review. Copilot requests retain 4,096 as a best-effort
request, and diagnostics must say `token_cap_enforced=False`. The durable
100-attempt ceiling remains hard. Two counted readiness attempts preceded the
LLM pilot (one authentication-configuration failure, one successful synthetic
typed-plan request); the planned pilot requires at most 96 more attempts.
No retries or extra calls are authorized beyond the shared ledger's allowance.

The confirmation resource amendment was made before payload/performance
inspection, using development-only evidence that CatBoost predictive cells
average more than 120 seconds. Its cohort, method/search settings and seeds
were not changed. Preregistration binding, payload/class-support checks, live
exposure checks and exact budget/role reporting are mandatory. Failed or
unfinished cases must not be replaced by a different cohort.

## Recommendation

Prioritize **reliable selection and stronger evidence before a larger feature
catalog**. Keep the predictive strategy opt-in until harmful regressions are
understood and performance is confirmed on previously unexamined dataset
families. Build the tabular evaluation foundation first; use separate protocols
for temporal, relational, and text/LLM feature engineering later.

The [preserved predictive measurement](tabular-results/2026-09-16-predictive/ANALYSIS.md)
shows real progress: linear-classification loss reduction improved from +0.55%
to +6.76%, and LightGBM regression from +0.08% to +2.10%. However, linear
regression RMSE increased by 32.33% on houses and 15.88% on wine. Neither the
positive mean nor one successful LLM-generated feature establishes SOTA.

Two benchmark findings materially change the expansion plan:

- **All six current classification source datasets are exactly balanced binary
  tasks.** Their public CSV class counts were checked locally. Multiclass and
  naturally imbalanced classification are missing from the current core.
- **Historical exposure extends beyond the current 12 tasks.** The legacy
  `simple_models/SIMPLE_MODELS_CACHE_LLM.json` contains real-world results for
  all 30 INRIA registry variants plus fake news. Another INRIA variant or mirror
  is not automatically a fresh confirmation dataset.

## 1. Prioritized engineering work

| Priority | Work | Deliverable and acceptance criteria |
|---|---|---|
| P0 | Diagnose the houses/wine regressions | Reproduce archived folds; log per-family candidate counts, selected-subset instability, feature tails, conditioning, and internal-versus-outer loss. Test hypotheses using training data and synthetic stress cases; do not assert a cause from aggregate scores. |
| P0 | Unify numerical behavior across selection and inference | One numerical policy for candidate scoring, subset validation, `transform`, and exported feature code. Tests cover near-zero denominators, large finite inputs, overflow, missing values, batch partitioning, and serialization. No stage-specific representation change. |
| P0 | Make subset selection stability-aware | Compare repeated training-only holdouts, a smaller-subset/uncertainty-aware rule, and regularization jointly with subset choice. Audit dependencies introduced by reusing OOF residuals across screening folds. Predeclare the comparison before measuring it; retain raw-feature fallback as a valid outcome. |
| P1 | Align screening with the downstream learner | Ablate current shallow-tree residual screening versus linear/residual-correlation screening for linear learners and conditional-importance screening for boosting. Preserve original columns; measure mean, median, worst-tail harm, completion and cost rather than only the best score. |
| P1 | Improve candidate efficiency and coverage | Compare generation-only, selection-only, and gate-enabled modes; ablate powers, ratios, products and differences. Detect constant/duplicate or redundant candidate families; measure permutation/renaming sensitivity and coverage under wide-data budgets. Do not increase candidate count without a demonstrated accuracy/cost benefit. |
| P1 | Add fitted categorical/group primitives | Training-only rare-category handling, cross-fitted target statistics, frequency/group aggregates, and unknown-category behavior. Frozen mappings, batch-invariant inference, schema checks, and group/time-aware cross-fitting are required. |
| P2 | Measure LLM value, not just connectivity | Keep `gpt-5.6-sol` as the requested default. Compare no LLM, LLM-only and hybrid proposals with matched selection/learners. Record valid/accepted/selected counts, incremental loss, tokens, latency, failures and cost. Preserve semantic names/descriptions separately from machine-safe column IDs. |
| P2 | Replace learned-statistic code snippets with fitted rules | Start with a small typed operator plan: operator, source columns, parameters and required fitted state. Reject or explicitly route unsupported batch aggregates; do not attempt to compile arbitrary Python into a leak-proof transformer. Keep existing code execution labeled as trusted-code functionality. |
| P3 | Generalize beyond IID tabular tasks | First add explicit group/time metadata and evaluation support, then relational feature plans and tabular-plus-text tasks. Separate scoreboards, baselines and budgets; do not pool forecasting, ranking and IID tabular metrics. |

**Planning-baseline code evidence, not an attribution of the observed regressions:**
At planning time, [`predictive.py`](../featcopilot/engines/predictive.py) used one internal
holdout and fixed subset sizes `(1, 5, 10, 25, 50)`. `_augment` clipped candidate
values to `[-1e30, 1e30]`, whereas production `transform` used `_compute` without
that same finite-value clipping. `_rank_candidates` reused its OOF residual
array across screening folds. These were concrete places to inspect and test;
they do not by themselves prove the cause of houses/wine harm.

## 2. Establish dataset identity before adding rows to a manifest

Use an **origin/family registry**, not only dataset names or hosting URLs:

| Required field | Why it matters |
|---|---|
| `family_id`, aliases, upstream origin | California/houses, wine variants, hosted mirrors and transformed versions must not count as independent evidence. |
| Source/provider ID, dataset/task version, revision and checksum | Pin OpenML tasks, Hugging Face commits, UCI records or locally supplied Kaggle files. Reject unexpected content changes. |
| Explicit target, target transform, feature schema and prediction-time availability | A derived target or post-outcome feature can invalidate a comparison. Record original and curated versions separately. |
| Prior exposure and assigned role | Distinguish canonical runs, legacy results, loader-only references and metadata-only inspection. Assign development/confirmation roles at the family level before running models. |
| License, terms, access requirements and approval status | A hosting site's availability or license badge does not automatically cover upstream data or redistribution. Unknown terms block admission, not the entire investigation. |
| Split policy and official split identifiers | IID, grouped and temporal tasks require different evaluators. Record entity/group/time columns and embargo/prediction cutoff where applicable. |
| Original/sample counts, classes, prevalence, missingness and cardinality | Expose effects of curation and row caps. Preserve natural imbalance; ensure split feasibility without silently substituting a different task. |
| Observational, domain-simulation or synthetic-diagnostic origin | A useful engineering simulation is not an observational dataset; keep these distinctions visible in reporting. |

**Do not allocate fresh confirmation from the remaining INRIA names by default.**
Legacy exposure includes Albert, Allstate claims, Bike Sharing, Bioresponse,
Brazilian houses, Covertype, Zurich delays, Diabetes130US, electricity, Higgs,
house sales, Jannis, Mercedes-Benz, Miami housing, MiniBooNE, NYC taxi,
road safety and superconductivity, in addition to the current families.
Treat ambiguous history conservatively.

Conversely, the old `titanic` and `house_prices` cache records explicitly describe
synthetic Kaggle-style data. Actual Titanic and Ames are different data, but
their separate loader/history exposure must still be checked before calling
them unseen. Public model pretraining exposure is a further limitation for
LLM/foundation-model evaluation; project-level holdout status does not resolve it.

## 3. Benchmark infrastructure changes before scaling

1. **Add configurable manifests and small provider adapters.** Keep the existing
   pinned suite unchanged; add OpenML-task, pinned-HF, UCI and approved local
   Kaggle inputs. Separate metadata admission/download validation from model
   execution. No synthetic fallback on a real-data load failure.
2. **Make the primary evaluation unit dataset × split × method × learner.**
   The current shared method/fold deadline couples learners, while predictive
   selection fits a separate representation for each learner. Isolate failures
   and budget accounting per learner; report shared-FE amortization separately.
3. **Preserve partial progress and support exact resume.** Reuse only completed
   cells with identical data/split/code/dependency/configuration fingerprints.
   Never reuse an engineered matrix across changed folds, labels or parameters.
4. **Support official, grouped and temporal splits.** Use task-provided splits
   for reproduction tracks. For entity or time tasks, every internal selection
   split must obey the outer group/time restriction too.
5. **Strengthen learner baselines fairly.** Add CatBoost and XGBoost; allow native
   categorical handling where supported. Fairness means identical preprocessing
   for raw versus engineered inputs of a given learner, not forcing every
   learner through dense one-hot encoding.
6. **Separate comparison tracks.** Direct FE tools with matched learners;
   complete AutoML/foundation-model pipelines as contextual baselines; and
   LLM-assisted FE as a separately costed track. Keep restricted comparator
   variants labeled, and add unrestricted/reference tracks when feasible.
7. **Keep reliable reporting.** Dataset-family-level pairing, complete-pair
   coverage, explicit timeout/error/OOM outcomes, performance/resource curves,
   and family-blocked uncertainty. Add PR-AUC for rare positives and suitable
   multiclass metrics; declare regression target transforms and losses up front.
8. **Make diagnostics manageable.** Archive reports and raw records
   automatically; keep detailed selection diagnostics separate from compact
   metrics. Local prediction diagnostics may aid debugging, but do not publish
   raw rows or targets unless terms permit.

The current loader hardcodes one HF snapshot and renames columns to `x0`, `x1`,
etc. That is adequate for the existing numerical comparison, but an LLM track
must receive the original semantic context through an explicit mapping.
See [`benchmarks/tabular.py`](tabular.py) and the
[LLM statistical-state limitation](../docs/user-guide/llm-features.md#fitinference-contract-and-limitations).

## 4. Proposed suites and execution tiers

Target **40-60 unique primary dataset families**, after overlap and admission
checks, rather than maximizing an unreviewed dataset count. Allocate approximately
two-thirds to development and one-third to locked confirmation. Existing
problem cases remain development data. Stress slices and multiple targets or
variants from one origin are useful diagnostics, not extra independent families.

| Tier | Purpose | Suggested scope |
|---|---|---|
| Fast regression suite | Catch numerical, schema, leakage and stability bugs | Synthetic invariants plus a small admitted development subset; cheap enough for routine checks. |
| Development comparison | Select engineering approaches, not advertise SOTA | Start with the current 12 plus a small coverage-driven expansion; capped data, fixed seeds, matched strong learners and resource limits. |
| Broad comparative run | Measure heterogeneity and accuracy/cost tradeoffs | 40-60 deduplicated families; binary, multiclass and regression; mixed/numeric, missing/high-cardinality, small/wide/large strata; multiple seeds and budget tiers. |
| Locked confirmation | Evaluate a preregistered release candidate | Previously unexamined families, fixed protocol and comparator configurations; no iterative tuning after results are revealed. |
| Future-domain suites | Extend beyond IID tabular engineering | Temporal/grouped, relational and mixed text/tabular tracks, each with its own legitimate prediction-time features and baselines. |

**Cost reality:** the preserved 14-task, 3-fold, 1-seed, 7-method, 2-learner run
took 150.21 minutes on the bounded local configuration. Simple linear scaling
at the same row caps and workloads gives roughly **10.7 hours for 60 tasks**,
or **53.6 hours for 60 tasks × 5 folds × 3 seeds**. These are coarse planning
extrapolations, not runtime guarantees; full datasets, larger searches, new
evaluation units and different timeout rates can change them substantially.
A four-hour pilot cannot honestly stand in for that full comparison.

## 5. Promotion and claim gates

Agree on and freeze thresholds before confirmation. Recommended initial gates:

- All train/inference numerical, batch-invariance, schema, serialization and
  split-boundary tests pass; full package mypy and standard checks remain clean.
- Preserve the known regression cases and report them all. As a proposed
  operational target, no confirmation dataset/learner's aggregate primary loss
  should worsen by more than 5%; do not interpret that target as a statistical
  guarantee or delete hard cases to satisfy it.
- Target at least 95% completion in the declared supported tier, with failures
  retained in the denominator. Compare quality on common completed families,
  and do not conflate coverage with predictive quality.
- Report mean, median, worst-tail harm, uncertainty and cost against strong raw
  learners and credible FE references. Use family-level analysis and appropriate
  multiple-comparison treatment for confirmatory superiority claims.
- A successful development experiment may justify an opt-in release, not a
  universal "#1" claim. Default promotion needs independent confirmation.
  Any SOTA statement must name the dataset suite, task/learner scope, comparator
  versions and resource budget.

The 5% and 95% values are proposed acceptance policies, not measured properties
of the current implementation.

## 6. Investigated benchmark ecosystems and papers

The following identities were verified from primary papers, official metadata
or source manifests. Counts describe the cited release/metadata, not necessarily
independent origins or fresh confirmation families.

| Resource | Verified scope | Recommended use and important limitation |
|---|---|---|
| **TabArena-v0.1** | [OpenML suite 457](https://www.openml.org/api/v1/json/study/457): **51 datasets**, 30 binary, 8 multiclass, 13 regression | Recommended backbone for broader IID comparisons. `lite` keeps split 0 on the **same datasets**; it is not a smaller dataset collection. Pin the official task manifest and implementation revision. |
| **OpenML-CC18** | [Suite 99](https://www.openml.org/api/v1/json/study/99): **72 classification tasks** | Fill classification gaps and reproduce established evaluations. Its class-balance criteria exclude some extreme rare-event problems; it is not a regression suite. |
| **AutoML Benchmark / AMLB** | [271](https://www.openml.org/api/v1/json/study/271): **71 classification**; [269](https://www.openml.org/api/v1/json/study/269): **33 regression** | Broader regression, reliability and system-level context. Do not add these counts to TabArena/CC18 without family deduplication, or confuse whole-AutoML rankings with isolated FE benefit. |
| **Grinsztajn transformed suites** | [334](https://www.openml.org/api/v1/json/study/334), [335](https://www.openml.org/api/v1/json/study/335), [336](https://www.openml.org/api/v1/json/study/336), [337](https://www.openml.org/api/v1/json/study/337): **7 / 17 / 19 / 16 tasks**, respectively mixed classification, mixed regression, numeric regression, numeric classification | Useful controlled panels, but most relevant families are already exposed in this project. These January-2023 transformed task collections are not 59 independent fresh datasets. |
| **OpenFE paper/reproduction** | [ICML 2023 paper](https://proceedings.mlr.press/v202/zhang23ay.html), [reproduction repository](https://github.com/ZhangTP1996/OpenFE_reproduce): **10 benchmark datasets plus 2 separate competition cases** | Add a labeled replication panel, not a handpicked replacement for the core. The ten include California, Microsoft, Medical, Diabetes, Nomao, Vehicle, Broken Machine, Telecom, Jannis and Covertype; source identity must be resolved before mapping names. Paper “Vehicle” is not automatically the small UCI Vehicle dataset. |
| **CAAFE** | [Paper](https://arxiv.org/abs/2305.03403), [official data loader](https://github.com/automl/CAAFE/blob/bb65b4d517bd86056352ad2ee98be6655f2d5aec/caafe/data.py): 10 OpenML IDs plus 4 other declared entries; actual loader filters/caps the panel | Good small-data semantic-FE replication. The inspected loader rejects regression, filters tasks at a 25-feature limit and caps rows at 2,000; declared entries are not proof all were successfully evaluated. |
| **LLM-FE** | [Revised paper v3, May 2026](https://arxiv.org/html/2503.14434v3), [official repository](https://github.com/nikhilsab/LLMFE): **19 classification + 10 regression datasets** in the paper | A newer direct LLM-FE comparator worth a bounded reproduction pilot. The inspected launch script covers less than the revised paper's panel; reconcile versions, source identities and caps before calling it reproduced. |
| **TabRepo** | [Official historical documentation](https://github.com/autogluon/tabarena/blob/f87f22d301b5985554c037c1bb1429ae6635dded/tabrepo.md): **1,530 configurations on 211 successful datasets** | Useful portfolio/ensemble reference artifacts, not a shortcut to scoring new FE inputs. The `D244_F3_C1530` name does not mean 244 successful datasets. Full prediction caches are very large; fetch small metadata/metrics first. |
| **BeyondArena** | [2026 paper](https://arxiv.org/abs/2606.30410), [official HF collection](https://huggingface.co/datasets/TabArena/BeyondArena), [data tooling](https://github.com/TabArena/data-foundry): **142 datasets** across IID, temporal and grouped regimes | Later distribution/split stress coverage. This is separate from TabArena-v0.1, not its “full” 142-dataset mode. Individual licenses and origin overlap still need admission review. |

TabArena metadata/source was inspected at
[`f87f22d301b5985554c037c1bb1429ae6635dded`](https://github.com/autogluon/tabarena/tree/f87f22d301b5985554c037c1bb1429ae6635dded).
The [curated manifest](https://github.com/autogluon/tabarena/blob/f87f22d301b5985554c037c1bb1429ae6635dded/packages/tabarena/src/tabarena/benchmark/task/metadata/data/curated_tabarena_dataset_metadata.csv)
is preferable to manually copying dataset names from a leaderboard.
Its official evaluation has nontrivial inner bagging/repeated outer evaluation:
a capped custom run on its datasets must be called **TabArena-derived**, not an
official reproduced TabArena score. Its binary primary metric is ROC-AUC;
our current binary primary metric is log loss. Preserve both tracks rather
than comparing incompatible published numbers.

Useful additional, audit-pending TabArena entries have stable dataset/task IDs:

| Candidate | OpenML dataset / task | Coverage |
|---|---|---|
| Airfoil self-noise | **46904 / 363612** | Small scientific regression |
| Anneal | **46906 / 363614** | Small multiclass manufacturing data |
| APSFailure | **46908 / 363616** | Larger failure-detection classification and missingness |
| In-vehicle coupon recommendation | **46937 / 363681** | Mixed categorical recommendation |
| NATICUSdroid | **46969 / 363689** | Wider classification |
| QSAR fish toxicity | **46954 / 363698** | Small scientific regression |
| Student dropout/academic success | **46960 / 363704** | Multiclass mixed data |

The curated manifest declares CC BY 4.0 for these entries. This is evidence of
a declaration, not a completed upstream-rights or historical-exposure audit.
Concrete, German Credit and Online Shoppers also occur in TabArena; do not count
their UCI and OpenML versions twice.

## 7. First concrete dataset-admission shortlist

Start by admitting **seven non-temporal candidates**, plus one **conditional
temporal candidate**. Do not run every candidate and then retrospectively label
the best ones as confirmation: assign roles after metadata/exposure review and
before model evaluation.

The official UCI pages below display **CC BY 4.0** and public acquisition links.
Sizes are source-reported; verify the actual payload, schema and row counts
when acquisition is approved.

| Family / source | Reported size and target | Why add it | Required evaluation guard |
|---|---|---|---|
| [Adult, UCI 2](https://archive.ics.uci.edu/dataset/2/adult) | 48,842 rows, 14 features; income class | Naturally imbalanced, mixed types and missing values | Preserve source train/test boundary; parse `?`, audit duplicate overlap and explicitly handle the survey-weight field. |
| [Online Shoppers, UCI 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | 12,330 × 17; `Revenue` | Imbalanced session classification and meaningful feature context | Declare prediction timing. Audit `PageValues`, durations and exit metrics; month alone cannot support precise chronological forecasting claims. |
| [German Credit, UCI 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | 1,000 × 20; 1=good, 2=bad | Low-N categorical robustness and variance | Repeated stratification; preserve category meanings. Numeric recoding is the same family. |
| [Dry Bean, UCI 602](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | 13,611 × 16; seven classes | Native multiclass and correlated measurements | Stratified multiclass splits; macro/per-class reporting and duplicate checks. |
| [Steel Plates Faults, UCI 198](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) | 1,941 rows, 27 predictors; seven fault indicators | Low-N multiclass industrial task | Verify exactly-one-label semantics before converting indicators to one target. Exclude **all seven** target columns from predictors. |
| [Concrete Strength, UCI 165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) | 1,030 × 8; compressive strength | Nonlinear, units-aware, low-N regression | Separate random interpolation from mixture-composition-grouped evaluation across curing ages. |
| [Energy Efficiency, UCI 242](https://archive.ics.uci.edu/dataset/242/energy+efficiency) | 768 × 8; heating/cooling loads | Low-N engineering regression with meaningful interactions | This is simulation data. Choose one primary target and exclude the other; geometry-grouped evaluation is separate from interpolation. Both targets remain one family. |
| [Seoul Bike Sharing, UCI 560](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) | 8,760 rows, 13 reported features; intended hourly rented-bike count | Integer regression and a small temporal proving ground | **Conditional on temporal support and target review.** Use forward/day-blocked splits; define weather/operating-status availability. |

**Metadata trap found during research:** the
[Seoul UCI API](https://archive.ics.uci.edu/api/dataset?id=560) labels
`Functioning Day` as the target and `Rented Bike Count` as a feature, while the
dataset narrative describes hourly demand prediction. A generic automatic
target loader would construct the wrong task. Require a reviewed explicit
target mapping; if unresolved, start with seven rather than silently replacing
it with already-exposed Capital Bikeshare.

None of these eight is certified fresh confirmation by this research alone.
Adult has an existing loader in the repository; loader presence is not proof
of execution, but its history must be checked.

## 8. Kaggle, Hugging Face and later-domain panels

### Kaggle: useful task coverage, conditional admission

Use labeled training data with internal tuning and a locked holdout. Do not
tune on leaderboards, treat sample submissions/pseudo-labels as truth, or
recover hidden competition test labels from upstream mirrors.

| Candidate / official rules | Value | Admission or leakage issue |
|---|---|---|
| [Real Ames House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules) | `SalePrice`, 79 explanatory variables; structural missingness and domain semantics | Conditional first addition after exposure and terms review. Generic competition metadata saying “MIT” is not a sufficient data-use grant; the inspected rules include academic/non-commercial conditions. |
| [Porto Seguro](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/rules) | Naturally imbalanced mixed-data claim prediction | Preserve `_cat`/`_bin` meanings and the `-1` missing sentinel; exclude ID from modeling. Verify approved acquisition and redistribution terms. |
| [Santander Value](https://www.kaggle.com/competitions/santander-value-prediction-challenge/rules) | Proposed wide, low-N anonymous regression stress test | Verify actual shape first; fold-local constant/duplicate selection, no leaderboard-derived target reconstruction. Poor semantic-LLM test, useful numerical/resource test. |
| [IEEE-CIS Fraud](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) | Wide, missing/high-cardinality transaction and identity tables | Forward evaluation by elapsed `TransactionDT`; safe `TransactionID` joins and recurring-entity checks. Not a drop-in shuffled IID task. |
| [Rossmann](https://www.kaggle.com/competitions/rossmann-store-sales/rules) | Store-level forecasting, promotions and metadata | **Hold for terms review/permission:** inspected rules restrict competition purpose/duration. If admitted, use shared forward cutoffs and past-only lags; future same-day `Customers` is not an available feature by default. |
| [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk/rules) | Application prediction with one-to-many history tables | **Hold for terms review/permission:** inspected competition-specific clauses restrict data use. Split applications/customers, not child rows; enforce pre-application information cutoffs. |

Official data/rules descriptions were inspected, but authenticated file-manifest
endpoints prevented complete shape verification. Do not put familiar but
unverified Kaggle row/feature counts into the registry. Terms notes identify
review requirements, not legal advice.

### Hugging Face: CARTE for a semantic-rich LLM panel

Verified official resource:
[`inria-soda/carte-benchmark`](https://huggingface.co/datasets/inria-soda/carte-benchmark),
revision **`8278e97aac2690ad7e780ff78f77e932e1c59cab`**, with
[`soda-inria/carte`](https://github.com/soda-inria/carte) and its
[2024 paper](https://arxiv.org/html/2402.16785v2).

The paper and inspected repository contain **51 task configurations: 40
regression and 11 classification**. This is not proof of 51 independent
families. The resource is public/ungated, but its card declares
**`license: unknown`**; review original-source rights before admission.

Start a small, rights-cleared semantic pilot rather than importing everything:

| Exact task identifier | Verified semantic field / target | Why investigate |
|---|---|---|
| `chocolate_bar_ratings` | `Specific_Bean_Origin_or_Bar_Name` / `Rating` | Product/origin semantics plus numerical attributes |
| `coffee_ratings` | `name` / `rating` | Named-product context; inspect the provided target transformation |
| `clear_corpus` | `Title` / `BT_Easiness` | Readability-related text/tabular regression |
| `prescription_drugs` | `Drug_Product_Description` / `WAC_at_Introduction` | Product descriptions, dates and numeric information |

Pinned task configs follow
[`data_carte/<identifier>/config_data.json`](https://huggingface.co/datasets/inria-soda/carte-benchmark/tree/8278e97aac2690ad7e780ff78f77e932e1c59cab/data_carte).
Some classification targets are binarized, so CARTE does not automatically fill
native-multiclass or natural-imbalance gaps. The inspected configs use
`repeated: false`; their loader does not thereby establish entity-disjoint
evaluation. Review entity keys rather than trusting metadata blindly.

Compare original names/descriptions against anonymized names on the **same
data and splits**. Include CatBoost, a strong string/high-cardinality encoder
such as a suitable `skrub.TableVectorizer` pipeline, and text-representation
baselines. CARTE as a model is a whole-model comparator, not itself a direct
FE tool. Do not send sensitive personal rows to hosted LLMs; schema-only or
approved, non-sensitive training context should be the default.

For LLM-FE/CAAFE comparisons, hold the underlying language model, prompt-call
budget and downstream learner constant where the APIs permit. Otherwise label
the result as a whole model/provider/method configuration comparison; a newer
LLM must not be mistaken for evidence that the FE search algorithm alone is
better. Keep hosted-model cost and request failures separate from CPU FE time.

**HF provenance trap:** [`mstz/covertype`](https://huggingface.co/datasets/mstz/covertype)
is a UCI-derived mirror, not new confirmation evidence. Its pinned loader still
downloads an external payload, so a code revision alone does not pin the data.
Retain upstream identity and payload checksums; a vague host license tag is
not enough.

### RelBench and BeyondArena after group/time-aware support

Verified RelBench host:
[`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1),
revision **`d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600`**. The shorter
`stanford-star/relbench` alias redirects, so record the explicit version.

- **First relational proving ground:** `rel-f1` / `driver-position`.
  The [task manifest](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/rel-f1/tasks/driver-position/manifest.yaml)
  predicts mean finishing position over the next 60 days. Official metadata
  describes 9 tables / 97,606 database rows and 7,453 / 499 / 760
  training/validation/test task rows. Use point-in-time joins and official
  temporal cutoffs; an average finishing position is not an integer target.
- **Later scale test:** `rel-amazon` / `user-churn`.
  The [manifest](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600/rel-amazon/tasks/user-churn/manifest.yaml)
  defines absence of reviews over 91 days for recently active users. Database
  size and task-label count are different; this is a large workload.
  Review inactivity is a proxy, not verified purchase churn. Never aggregate
  future reviews into prediction-time features.

The host reports CC BY-SA 4.0; upstream Ergast/Amazon data rights remain
review-pending. These are later capability tracks, not immediate additions to
the current shuffled-row runner. BeyondArena can provide analogous explicit
IID/group/time stress tasks once individual manifests and terms are audited.

## 9. First implementation batch, after approval

1. Create the family/exposure/rights registry seeded from all historical results.
2. Add metadata validation and explicit targets for the seven UCI IID candidates;
   resolve Seoul separately. Assign development/confirmation roles before runs.
3. Add `--suite`-style manifest selection and OpenML task support; import a
   pinned TabArena manifest as an inventory, not an automatically clean holdout.
4. Fix and test selection/inference numerical parity, then diagnose stability
   on existing development regressions.
5. Add per-learner budgeting/resume and stronger matched raw-model baselines.
6. Run a bounded development pilot, preserve all failures, and use it to choose
   the next engineering experiment. Do not open the confirmation cohort yet.

This planning pass inspected existing repository evidence and public source
metadata. It did **not** add dataset loaders/manifests, download new full datasets,
run model comparisons, or change feature-engineering algorithms.

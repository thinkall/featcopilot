# Feature Selection

After generating features, FeatCopilot automatically selects the most important ones to prevent overfitting and reduce dimensionality.

## Overview

The default, legacy selection pipeline:

1. **Statistical Selection**: Filter by statistical significance
2. **Model-based Selection**: Rank by ML model importance
3. **Redundancy Elimination**: Remove highly correlated features
4. **Final Selection**: Keep top N features

## Explicit task routing

`task="auto"` preserves legacy inference. For regression targets that happen to
be integers (counts, ratings, CPU activity), specify `task="regression"` rather
than relying on label-cardinality heuristics:

```python
from featcopilot import AutoFeatureEngineer
from featcopilot.selection import FeatureSelector, ImportanceSelector, StatisticalSelector

engineer = AutoFeatureEngineer(task="regression", max_features=50)
selector = FeatureSelector(task="regression", max_features=50)
statistical = StatisticalSelector(method="f_test", task="regression")
importance = ImportanceSelector(task="regression")
```

The explicit task reaches child selectors, refinement models and the legacy
gate. Explicit classification validates discrete labels; explicit regression
requires finite numeric targets. Chi-square is not a regression selector.

## Predictive tabular strategy (opt-in)

For an experimental artifact that also fits a downstream classifier and
can retain several independent member plans, see
[engineered classification](classification.md). `AutoFeatureEngineer`
remains the feature-transformer API and does not expose an ensemble union.

```python
from sklearn.linear_model import Ridge
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=["tabular"],
    task="regression",
    selection_strategy="predictive",
    selection_estimator=Ridge(alpha=1.0),
    max_candidates=1000,
    max_features=50,
    random_state=42,
    stability_repeats=3,
    screening="auto",
)
X_train_fe = engineer.fit_transform(X_train, y_train)
X_test_fe = engineer.transform(X_test)
print(engineer.selection_report_)
```

This strategy replaces, rather than stacks on top of, the legacy selector and
RF gate. It generates a bounded pool of numeric transformations and
interactions, ranks incremental signal, then compares candidate subsets with
the raw-feature baseline on repeated training-only holdouts. A smaller-subset,
one-standard-error rule favors stability over a single best holdout. All original columns
are retained. A predictor suited to the downstream task can be supplied; the
default uses sklearn histogram gradient boosting.

The predictive wrapper returns originals in their fitted column order even
when inference columns arrive reordered. Original values, including infinities
and missing values, are preserved in the output; learned numeric preprocessing
belongs inside the downstream model pipeline. Generated features follow the
predictive engine's separate finite-value policy.

For injected classifiers, dictionary `class_weight` keys use the original
training labels, including string or nonzero integer labels. The engine maps
them on its owned estimator clone when encoding targets; it does not mutate
the supplied estimator or weights. Partial dictionaries, `None` and
`"balanced"` retain their native meanings. Unknown label keys are rejected.

`screening="auto"` uses linear residual screening for supported bare linear
estimators and shallow trees otherwise. Explicit `"linear"` and `"tree"` modes
support ablations; `stability_repeats=1` provides a single-holdout mode.
These are not exact reproductions of older engine revisions: numerical
consistency and screening-boundary fixes apply in every mode.

### Experimental paired-gain rule

`selection_rule="paired_gain"` is an opt-in development alternative; the default
remains `"one_se"`. It measures each subset's gain against the raw baseline on
the **same** internal repeats. A positive mean gain must clear both the existing
0.1% relative threshold and its paired standard-error margin. Smaller subsets
are then preferred using paired loss gaps to the best eligible subset.

This removes shared fold difficulty from the uncertainty calculation rather
than treating absolute loss variation as uncertainty about incremental value.
Overlapping holdouts still do not provide a formal confidence interval. The
rule is a hypothesis to evaluate on development data, not a claimed improvement
on untouched data.

```python
engineer = AutoFeatureEngineer(
    selection_strategy="predictive",
    selection_rule="paired_gain",
    task="regression",
    max_features=50,
)
```

When either rule retains raw features, the engine now skips the unused final
all-data candidate rescreen. This preserves output values and records
`ranking_scope="baseline_retained_no_final_refit"` rather than presenting the
earlier ranking as a fresh all-data fit.

### Extended candidate coverage

`candidate_set="extended"` is a separate opt-in vocabulary. It adds rowwise
sums, minima and maxima, balances bounded pair enumeration across columns,
and plans categorical-pair indicators from each internal training partition.
The default `"standard"` pool and ordering are preserved.

Categorical planning retains at most eight levels per column and four
indicators per column pair, with support at least
`max(3, ceil(0.01 * training_rows))`. It uses no target statistics or validation
vocabulary. Missing/nonfinite categorical values have an explicit missing
level; unseen pairs produce zero. Candidate planning and pair scanning remain
bounded by the configured maximum of 1,000.

Names are hashed rather than exposing category values in report names. Fitted
feature/model artifacts and exported code do contain the learned values and
must be treated as training-derived artifacts.

```python
engineer = AutoFeatureEngineer(
    selection_strategy="predictive",
    candidate_set="extended",
    selection_rule="paired_gain",
    task="classification",
    max_features=50,
)
```

This mode addresses measured coverage gaps; synthetic checks do not establish
superiority on real tasks. It must be compared with the standard vocabulary on
the same development folds before any independent confirmation.

### Joint importance screening (opt-in)

`screening="importance"` ranks candidates with one cloned learner fitted
jointly on originals and candidate columns from the current internal training
partition. It first removes constant candidates and exact aliases of original
numeric or earlier candidate columns, verifying equality after hash matches.
It does not apply correlation-based pruning or refill the candidate budget.

The ranking uses native feature importance (gain for supported LightGBM
ranking clones), or normalized absolute coefficients averaged across classes.
Independent holdout losses still determine the final subset; fitted training
importance alone never establishes a feature's external value.

Supply an explicitly compatible `selection_estimator`, such as Ridge,
LogisticRegression, RandomForest or LightGBM. The default histogram-boosting
estimator does not expose the required importance API. Unsupported estimators
or inconsistent transformed-column dimensions raise rather than invent scores.
`auto`, `linear` and `tree` keep their existing behavior.

The `importance_ranking` report records planned/retained candidates, exact
discard reasons, importance source, fit rows and learner-fit counts. Matching
a candidate to an original on training values does not prove universal
algebraic equivalence, but it cannot use validation/test values in that decision.

Candidate scoring, subset evaluation, production transforms and exported
feature code share the same numeric policy: finite generated outputs are
clipped to `[-1e30, 1e30]`; undefined/overflowing results become missing values.
Original columns are not modified. Internal uncertainty estimates remain
descriptive because repeated holdouts are dependent.

`selection_estimator` must be a cloneable bare sklearn learner. The strategy
fits its own train-only mixed-data preprocessing, so do not supply a
column-bound preprocessor that silently ignores newly generated features.
The supplied estimator is cloned, not fitted in place.

When the selection worker policy overrides `n_jobs`, concrete
`LogisticRegression` clones using LBFGS or Newton-CG with automatic or
multinomial classification use one task worker. These solvers have one
binary/multinomial optimization task, so spawning a process pool adds overhead
without parallelizing independent fits. Native thread limits remain unchanged.
Explicit one-vs-rest configurations, other solvers, wrapped/subclassed
estimators, and tree/custom estimators retain their existing worker handling.
The original estimator and all statistical solver settings are preserved.

### Resampled importance ranking (opt-in)

`importance_resamples=5` replaces each single joint-importance ranking with
the mean candidate ranks from five deterministic 85% IID training
subsamples. The default `1` keeps the existing behavior; this first
experimental API accepts only `1` or `5` and requires explicit
`screening="importance"`.

Every bag retains all original features as model context and uses the same
candidate-definition universe. Learned frequency mappings, clipping,
imputation and encoding are fitted on the bag's training rows. Ranks use
the existing learner importance definition, average ties, and canonical
candidate identities for final tie-breaking. Candidates never receiving
positive importance are ineligible; there is no majority-vote threshold.

Exact scope-level constants and duplicates are removed before resampling.
Bag-local unusable or duplicate candidates receive zero importance and
worst tied ranks in that same universe; no replacement features are
generated. Constant-target regression bags remain valid, while infeasible
classification bags raise explicitly.

Aggregation occurs inside each internal training partition, before its
validation loss is computed. The final positive-count ranking uses the
same procedure, including with one validation repeat. The gate evaluates
a selection/refit procedure, not a guarantee that the exact final feature
names were seen in an earlier validation fit.

This initial mode rejects group, timestamp and label-availability metadata;
the single-fit mode retains existing support. It is empirical resampling,
not formal stability selection: no false-positive, FDR, or no-harm guarantee
is claimed. More consistent names alone do not establish better predictions.

Enabled `importance_resampling` diagnostics record bag counts, deterministic
seeds and hashed positional masks, actual learner fits, eligible counts and
rank semantics. Per-candidate ranks/support are available in detailed
benchmark diagnostics; ordinary reports keep compact counts and selected
names rather than repeating the full score arrays.

The first fixed development comparison did not establish a general benefit:
three task/learner mean gains declined, tail harm increased, and runtime
roughly tripled. The setting remains experimental and off by default.

### Cross-fitted categorical model inputs (opt-in)

`categorical_encoding="target"` is a separate model-input representation
for IID classification with explicit importance screening. The default
`"one_hot"` is unchanged. This option does not generate target-encoded
`Feature` objects, alter raw output columns, or insert encoded values into
candidate formulas.

The shared encoder uses sklearn's `TargetEncoder` with automatic smoothing,
explicit binary/multiclass target type, and five shuffled stratified folds
at the configured seed. It requires sklearn 1.4 or later only when a
categorical target-encoding block is active; older one-hot installations
remain unaffected. Numeric-only frames do not construct a target encoder.

The learner must receive the cross-fitted matrix returned by
`fit_transform(X_train, y_train)`. A later `transform(X_train)` uses the
full-training mapping and is **not** an interchangeable training matrix.
This boundary is enforced independently for each internal ranking model,
subset evaluation and final model. External validation/test rows use only
the fitted `transform`, without their labels.

Binary encoding produces one column per categorical input; multiclass
produces one per input/class pair. Unknown values use the applicable
training prior, and missingness remains distinct from literal sentinel-like
strings. Target mode preserves categorical scalar types until the shared
adapter normalizes them, so integer `1`, string `"1"`, Boolean `True` and
string `"True"` are not silently merged. The one-hot path keeps its
existing normalization. Learned category maps and priors belong to the fitted model;
diagnostics report only policy, counts, widths and resource information.

Use the experimental [classifier API](classification.md) to apply the same
policy consistently to feature selection and the final learner. A bare
`AutoFeatureEngineer` still returns unencoded features, so its downstream
model must independently use the matching encoding pipeline.
Target mode rejects regression, group/time metadata and insufficient
five-fold class support rather than reducing folds or falling back to OHE.
Native CatBoost preprocessing and calibrated benchmark bundles are not
silently switched to this mode.

Compare FE against an identically target-encoded raw learner. Encoding
alone can change model geometry and dimensionality, and cross-fitting
does not guarantee improved prediction or eliminate train/inference
representation shift.

The first fixed development study did not justify changing the default.
Linear target FE improved versus its target-encoded raw control but was
substantially worse on CKD than the original OHE pipeline. Always compare
the whole pipeline with the existing baseline, not just FE versus a
potentially weakened encoder-matched control.

### Training-support numeric clipping (opt-in)

`numeric_clip_quantiles=(0.0, 1.0)` enables independently fitted numeric
model-input caps during predictive selection. `None` preserves the existing
default. This initial option requires explicit `screening="importance"`.
The endpoint pair caps new values to training minima/maxima while
preserving ordinary finite training values and the existing downstream
scaler. It does not replace the original columns or change generated
`Feature.code`; applications must apply the same clipper in the final
learner pipeline.

For numeric-only inputs, keep every learned step inside the model pipeline:

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from featcopilot.utils.numeric_clipping import NumericQuantileClipper

model = make_pipeline(
    AutoFeatureEngineer(
        selection_strategy="predictive",
        selection_estimator=Ridge(alpha=1.0),
        task="regression",
        screening="importance",
        numeric_clip_quantiles=(0.0, 1.0),
    ),
    NumericQuantileClipper(quantiles=(0.0, 1.0)),
    SimpleImputer(strategy="median"),
    StandardScaler(),
    Ridge(alpha=1.0),
)
model.fit(X_train, y_train)
```

For mixed inputs, use an appropriate `ColumnTransformer` or native-categorical
learner after the clipper. Numeric bounds are fit before imputation from
finite training observations; missing values remain missing. Categories,
row order and schema are preserved. Numeric infinities become missing and
the existing finite safety range of `[-1e30, 1e30]` is applied consistently
before bounds estimation and clipping. This guard is an explicit exception
to unchanged finite training values beyond that range.

Other quantile pairs can cap training tails, but the first benchmark
ablation fixes `(0, 1)` rather than searching thresholds. The application CLI
does not expose this model-input option because it exports features, not a
paired downstream learner pipeline. Compare against the same capped
raw-feature learner: repairing raw outliers alone is not evidence of
better feature engineering. Caps can remove useful extrapolation and do
not guarantee stable feature selection or bounded prediction error.

The first matched development comparison improved linear-model FE results
beyond capped raw inputs, while every LightGBM score stayed unchanged and
linear-classification feature-set stability declined. This option therefore
remains experimental, not a general robustness guarantee.

On a fixed training-only temperature audit split, capping reduced raw RMSE
from 1.5419 to 0.2645 and capped FE reached 0.2569: most of the gain was
preprocessing. A hardware audit worsened relative to uncapped FE, illustrating
the legitimate-extrapolation tradeoff. These diagnostics do not justify
enabling clipping blindly for every dataset.

### Training-fitted frequency candidates (opt-in)

`include_frequency=True` adds a bounded unary-frequency ablation to
predictive selection. It requires `screening="importance"` and an explicit
compatible `selection_estimator`; the default is `False`. This is available
through the Python APIs and the benchmark's `featcopilot-frequency-paired`
method, not the application CLI, which cannot supply an estimator object.

```python
from sklearn.linear_model import Ridge

engineer = AutoFeatureEngineer(
    selection_strategy="predictive",
    selection_estimator=Ridge(),
    task="regression",
    screening="importance",
    selection_rule="paired_gain",
    candidate_set="extended",
    include_frequency=True,
)
```

Each internal training partition fits its own value-to-frequency mapping;
frequency means training count divided by total training rows. Missing,
nonfinite numeric and unseen values map to zero. Strings, booleans and
numbers have distinct keys; numeric `1` and `1.0` intentionally share a key.
Mappings remain frozen during validation and inference, including across
batches. The final fitted representation learns only from its allowed
training rows, never from held-out rows or labels.

For candidate budget `C`, at most `min(32, C // 10, supported_columns)`
columns are inspected at evenly spaced positions in canonical column order.
Constant frequency candidates are skipped without searching an unbounded
replacement pool. The unchanged base candidate prefix receives all remaining
capacity; total candidates stay within `C`. Budgets below ten allocate no
frequency candidates. This policy preserves at least 90% of base capacity,
not every candidate from a previously full pool.

Fitted artifacts and exported definitions contain the learned vocabulary.
Treat them accordingly; diagnostic logs do not disclose those values.
There are no pair-frequency or target-frequency operators in this ablation.
More candidate coverage is not a performance guarantee, and the current
effective limit of 50 selected additions is unchanged.

The frozen frequency-paired configuration subsequently failed independent
promotion gates, including substantial linear-regression harms. Keep this
mode experimental and validate on task-appropriate held-out data; internal
selection does not guarantee improvement over raw features.

When enabled, `selection_report_["frequency_policy"]` records bounds,
semantics and final refit/discard counts. Each repeat's `frequency_planning`
records the inspected columns and its training-row count without exposing
the learned vocabulary. A final training refit can make a selected frequency
constant or duplicate; those removals are reported explicitly. `selected_count`
then records delivered additions, while `validated_selected_count` preserves
the earlier subset decision. Validation scores are not rewritten to pretend
the reduced subset was separately evaluated.

Predictive selection currently requires `engines=["tabular"]` and a target.
`apply_selection=False` is incompatible: use the legacy strategy for raw
generation. `max_candidates` bounds the candidate search; `max_features`
limits the final total width without discarding original columns.

The exposed `selection_report_` describes internal selection decisions. It is
not an unbiased external score or a guarantee of improvement. Put feature
engineering inside the complete cross-validation pipeline and use untouched
test data for final evaluation.

### Group and time constraints

```python
engineer.fit(X_train, y_train, groups=training_customer_ids)
# Or, for ordered observations:
engineer.fit(X_train, y_train, timestamps=training_timestamps)
# If labels become available after their prediction timestamps:
engineer.fit(
    X_train, y_train,
    timestamps=training_timestamps,
    label_end_times=training_label_availability,
)
```

The same restriction applies to internal holdouts, ranking and residual fitting.
Groups stay disjoint; temporal validation uses strictly later timestamp blocks
without splitting equal timestamps. When `label_end_times` are supplied, every
internal training partition excludes rows whose labels are not available
strictly before validation starts; equality is purged. End times must be at or
after prediction times and use matching numeric units or valid datetimes.
Combined group/time restrictions and infeasible/purged-empty partitions raise
explicitly, never falling back to random splits. Availability metadata must
reflect the real prediction problem; timestamp ordering alone cannot infer it.

Legacy selection/gating does not support these constraints. Fitted LLM plans
with metadata cannot silently invoke the legacy selector; use
`apply_selection=False` when applying supported plans under caller-controlled
splits. Fitted target-mean plans support group-disjoint or chronological
cross-fitting with explicit coverage diagnostics; combined group/time metadata
remains unsupported.

## FeatureSelector

The unified selector combines multiple methods:

```python
from featcopilot.selection import FeatureSelector

selector = FeatureSelector(
    methods=['mutual_info', 'importance'],
    max_features=50,
    correlation_threshold=0.95,
    verbose=True
)

X_selected = selector.fit_transform(X, y)

# Get results
print(f"Selected: {len(selector.get_selected_features())} features")
print(f"Top features: {selector.get_ranking()[:5]}")
```

## Selection Methods

### Statistical Selection

```python
from featcopilot.selection import StatisticalSelector

# Mutual Information
selector = StatisticalSelector(
    method='mutual_info',
    max_features=30
)

# F-test (ANOVA)
selector = StatisticalSelector(
    method='f_test',
    max_features=30
)

# Chi-square (for categorical targets)
selector = StatisticalSelector(
    method='chi2',
    max_features=30
)

# Correlation with target
selector = StatisticalSelector(
    method='correlation',
    threshold=0.1  # Minimum correlation
)
```

### Model-based Selection

```python
from featcopilot.selection import ImportanceSelector

# Random Forest importance
selector = ImportanceSelector(
    model='random_forest',
    max_features=30,
    n_estimators=100
)

# Gradient Boosting
selector = ImportanceSelector(
    model='gradient_boosting',
    max_features=30
)

# XGBoost (if installed)
selector = ImportanceSelector(
    model='xgboost',
    max_features=30
)
```

### Redundancy Elimination

```python
from featcopilot.selection import RedundancyEliminator

eliminator = RedundancyEliminator(
    correlation_threshold=0.95,  # Remove if corr > 0.95
    method='pearson',            # pearson, spearman, kendall
    importance_scores=None       # Optional: keep more important feature
)

X_reduced = eliminator.fit_transform(X)

# See what was removed
print(f"Removed: {eliminator.get_removed_features()}")
```

## Combined Selection

### In AutoFeatureEngineer

```python
from featcopilot import AutoFeatureEngineer

engineer = AutoFeatureEngineer(
    engines=['tabular'],
    max_features=50,
    selection_methods=['mutual_info', 'importance'],
    correlation_threshold=0.95
)

# Selection happens automatically in fit_transform
X_selected = engineer.fit_transform(X, y)
```

### Manual Pipeline

```python
from featcopilot.engines import TabularEngine
from featcopilot.selection import FeatureSelector

# Generate features
engine = TabularEngine(polynomial_degree=2)
X_features = engine.fit_transform(X)

# Select best features
selector = FeatureSelector(
    methods=['mutual_info', 'f_test', 'importance'],
    max_features=30,
    correlation_threshold=0.95
)
X_selected = selector.fit_transform(X_features, y)
```

## Configuration

### Method Comparison

| Method | Best For | Speed | Handles Non-linear |
|--------|----------|-------|-------------------|
| `mutual_info` | General | Medium | ✅ Yes |
| `f_test` | Linear relationships | Fast | ❌ No |
| `chi2` | Categorical features | Fast | ❌ No |
| `correlation` | Quick filtering | Fast | ❌ No |
| `importance` | Complex patterns | Slow | ✅ Yes |

### Combining Methods

```python
# Union: feature selected by ANY method
selector = FeatureSelector(
    methods=['mutual_info', 'importance'],
    combination='union'
)

# Intersection: feature selected by ALL methods
selector = FeatureSelector(
    methods=['mutual_info', 'importance'],
    combination='intersection'
)
```

## Accessing Results

### Feature Scores

```python
# Combined scores (normalized 0-1)
scores = selector.get_feature_scores()

# Per-method scores
method_scores = selector.get_method_scores()
print(method_scores['mutual_info'])
print(method_scores['importance'])
```

### Feature Ranking

```python
# Sorted list of (feature, score) tuples
ranking = selector.get_ranking()

for i, (feature, score) in enumerate(ranking[:10], 1):
    print(f"{i}. {feature}: {score:.4f}")
```

### Selected Features

```python
selected = selector.get_selected_features()
print(f"Selected {len(selected)} features:")
print(selected)
```

## Best Practices

### 1. Use Multiple Methods

```python
# More robust selection
selector = FeatureSelector(
    methods=['mutual_info', 'f_test', 'importance']
)
```

### 2. Set Appropriate Thresholds

```python
# Conservative: fewer, stronger features
selector = FeatureSelector(
    max_features=20,
    correlation_threshold=0.90
)

# Liberal: more features
selector = FeatureSelector(
    max_features=100,
    correlation_threshold=0.99
)
```

### 3. Consider Task Type

```python
# Classification
selector = StatisticalSelector(method='mutual_info')

# Regression
selector = StatisticalSelector(method='f_test')
```

### 4. Handle Imbalanced Data

```python
# For imbalanced classification, importance-based selection
# often works better than statistical methods
selector = ImportanceSelector(
    model='random_forest',
    n_estimators=200
)
```

## Correlation Matrix

Visualize feature correlations:

```python
eliminator = RedundancyEliminator(correlation_threshold=0.95)
eliminator.fit(X)

# Get correlation matrix
corr_matrix = eliminator.get_correlation_matrix()

# Plot (requires matplotlib/seaborn)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png')
```

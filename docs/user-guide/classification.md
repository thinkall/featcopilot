# Experimental engineered classification

`FeatureEnsembleClassifier` is an explicitly selected prediction artifact:
it fits feature discovery, preprocessing and a classifier together. It does
not change `AutoFeatureEngineer` defaults or establish SOTA.

The extracted procedure reproduced all 162 outcomes of a fixed real-data
benchmark replay exactly, including member plans and selected parameters.
That verifies implementation parity, not superiority over competitors.

Unlike a feature transformer, an ensemble retains separate member plans.
Use `predict_proba` or `predict`; there is no shared transformed feature
matrix or union of member features.

## Single-model example

Supply a bare probabilistic classifier and exactly three dictionaries of
its **native** hyperparameters. The mode is explicit rather than inferred
from the estimator's class.

```python
from sklearn.linear_model import LogisticRegression

from featcopilot.classification import FeatureEnsembleClassifier

classifier = FeatureEnsembleClassifier(
    estimator=LogisticRegression(max_iter=2000, random_state=42),
    parameter_candidates=[{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    mode="single",
    engineered=True,
    random_state=42,
    n_jobs=2,
    numeric_clip_quantiles=(0.0, 1.0),
)
classifier.fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)
predictions = classifier.predict(X_test)
```

The wrapper performs its fixed inner holdout search only on supplied
training rows. Each hyperparameter candidate fits its own feature discovery
and preprocessing; the winning procedure is then refitted on those
training rows. Place the **whole classifier** inside outer cross-validation
when measuring generalization.

`engineered=False` supplies the matched raw control with the same tuning
and prediction architecture. Comparing only against an untuned or
differently preprocessed raw learner would confound feature value.

### Optional categorical target encoding

`categorical_encoding="target"` selects fixed five-fold cross-fitted
categorical preprocessing for both feature-selection models and the final
classifier. `"one_hot"` remains the default. The learner trains on the
encoder's cross-fitted `fit_transform` output, while inference uses its
full-training mapping. Do not encode the full dataset before splitting.

Active categorical target encoding requires sklearn 1.4+, adequate class
support in every actual training partition, and IID classification.
It does not reproduce CatBoost's ordered target-statistics algorithm.
Use the same setting with `engineered=False` for a matched raw comparison;
see [the encoding contract](selection.md#cross-fitted-categorical-model-inputs-opt-in).

## Three-member prediction ensemble

For the fixed higher-retention mode, use `mode="subsample85"`. The following
example requires a separately installed LightGBM package; importing
`featcopilot.classification` does not require it.

```python
from lightgbm import LGBMClassifier

from featcopilot.classification import FeatureEnsembleClassifier

classifier = FeatureEnsembleClassifier(
    estimator=LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        reg_lambda=1.0,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        random_state=42,
        n_jobs=2,
    ),
    parameter_candidates=[{"num_leaves": 7}, {"num_leaves": 15}, {"num_leaves": 31}],
    mode="subsample85",
    engineered=True,
    random_state=42,
    n_jobs=2,
    numeric_clip_quantiles=(0.0, 1.0),
)
classifier.fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)
```

Each of three deterministic stratified bags contains approximately 85% of
the input training rows, without replacement within a bag. Bags may overlap.
Each member performs its entire search and fitted preprocessing within its
own bag; inference averages aligned native probabilities. There is no
calibrator and no subsequent all-training refit of an ensemble member.

The supplied estimator controls its own solver, iteration count, device
and explicit randomness/thread settings. Set those deliberately along with
the wrapper's split/FE seed and native-thread budget. Arbitrary supplied
estimators are not thereby guaranteed deterministic.

## Fixed raw/FE probability average

`mode="raw_fe_average"` fits one full-training raw procedure and one
full-training FE procedure, then averages their probabilities with fixed
weights `[0.5, 0.5]`. It does not learn a blending weight, construct
calibrators, or perform additional prediction-level OOF fitting.
The two members tune independently using the same three supplied
hyperparameter candidates.

```python
from sklearn.base import clone

fixed_average = clone(classifier).set_params(
    mode="raw_fe_average",
    engineered=True,
)
fixed_average.fit(X_train, y_train)
probabilities = fixed_average.predict_proba(X_test)

raw_control = clone(fixed_average).set_params(engineered=False)
raw_control.fit(X_train, y_train)
```

With `engineered=False`, this mode fits one raw member with weight `[1.0]`;
it does not train duplicate raw models. `member_roles_` and `weights_`
identify the actual deployed members. This rule showed consistent
LightGBM/OHE gains on the measured development panel, but it is not a
general recommendation for every estimator or categorical encoding.
Existing `single` and `subsample85` behavior remains unchanged.

## Feature plans and serialization

The fitted `estimators_` are the complete member procedures: one for
`single`, three for `subsample85`, or raw/FE members for the explicit
fixed-average mode. Each retains its own selected parameters,
feature definitions and learned preprocessing state. There is intentionally
no ensemble `transform()` or single `get_feature_names_out()` implying that
different plans share one feature space.

`get_feature_plans()` returns independent copies of member feature metadata;
`diagnostics()` returns compact member counts, selected settings, hashes
and timing information without learned category vocabularies or row data.
Diagnostics are not a substitute for the fitted prediction artifact.

Serialize the complete classifier rather than exporting a union of features.
Its trusted pickle/joblib artifact includes fitted category/frequency
vocabularies and must be handled as model data. Compact diagnostics must not
be confused with that complete inference artifact.

Native `class_weight` dictionaries may use the original class labels;
the wrapper translates them on owned copies without modifying your
estimator or candidate dictionaries. If a supplied classifier is usable
in memory but cannot be pickled, fitting and prediction can still succeed:
diagnostics mark its fitted-state hash unavailable and emit a warning.
Actual pickle/joblib export retains its normal serialization requirements.

## Scope and limitations

- This first API is classification-only and IID-only. Group/time metadata,
  sample weights and external validation inputs are not silently ignored.
- The tested numeric/OHE prediction path is distinct from native CatBoost
  benchmark preprocessing. Do not assume equivalent native-category behavior.
- Exact migration parity and compatibility are verified on sklearn 1.7.2
  and 1.8. Broader historical dependency ranges are not a tested support
  matrix for this new experimental API.
- The feature policy is fixed to the tested extended, frequency-enabled,
  single-importance, paired-selection procedure. It is not a new automatic
  architecture or feature-policy search.
- Training-support caps can suppress valid extrapolation. Internal selection
  and prediction averaging do not guarantee improvement over raw features.
- The favorable qualified LightGBM evidence covers only two public IID
  datasets with unverified entity mapping. The overall locked comparison
  still failed its linear-versus-AutoFeat criterion.

Existing regression/feature-transformer workflows continue to use
`AutoFeatureEngineer`. No clinical, patient-independent, universal
no-harm or SOTA claim is implied by this experimental classifier.

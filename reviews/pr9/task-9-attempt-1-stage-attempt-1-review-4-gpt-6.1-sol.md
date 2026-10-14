## Review Summary
- **Round**: 4
- **Theme**: Detailed correctness
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol / max; explicitly assigned two-model override
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-4-gpt-6.1-sol.md
- **Scope**: Full PR9, base `bf119812162af87c29bc8d09bb3c705576b94f27` through the current worktree, including uncommitted fixes and `featcopilot\llm\_plan_transport.py`. Observed HEAD: `62cbb1cf4f434a6bbfd956b34c2615e761441e46`.
- **Stage Attempt**: 1
- **Issues Found**: 2 newly introduced issues: 1 Medium, 1 Low
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read the assigned prompt's first 100 lines and scope index, then inspected actual source bodies rather than treating the trimmed map as a complete diff. Used read-only base comparisons; `git diff --diff-filter=A --name-only <base> -- featcopilot benchmarks` confirms both finding paths and their supporting classifier modules are additions relative to the PR base.
- [x] Traced positional IID/group/temporal splitting, joint timestamp/end-time normalization, purging, target cross-fitting, typed frequency keys, and fitted-state export in `featcopilot\utils\splitting.py`, `task.py`, `frequency.py`, and `featcopilot\engines\fitted_rules.py`.
- [x] Read predictive candidate planning/computation, nested residual folds, importance output-position attribution, frequency refitting, bag aggregation, and subset selection in `featcopilot\engines\predictive.py`; followed classifier normalization, label/class-weight encoding, trial selection, final refit, and probability alignment through `classification.py` and `utils\model_preprocessing.py`.
- [x] On the established benchmark interpreter, ran a bounded, provider-free check with duplicate/nonmonotonic DataFrame indices, nullable booleans, numeric categorical values, and string labels. Both one-hot and target-encoded raw single classifiers returned finite `(3, 2)` probabilities in the original class order and valid diagnostics.
- [x] Inspected catalog target/class encoding, source identity/cache pins, official/positional splits, dataset hydration, worker validation, and calibration/blending normalization in `benchmarks\catalog.py`, `execution.py`, `tabular.py`, `calibration_evaluation.py`, and `blending_evaluation.py`. Also traced the prediction/OOF/fixed-average, frozen-plan, regression-average/audit, relational, and inductive-reference producers.
- [x] Reviewed immutable archive generation, atomic index publication, source snapshots, record hashes, and resume state in `benchmarks\run_state.py`. Did not read or regenerate frozen result archive bodies.
- [x] Inspected provider output/error contracts and the new isolated transport, including `featcopilot\llm\_execution.py`, `_plan_transport.py`, `copilot_client.py`, `openai_client.py`, `litellm_client.py`, `semantic_engine.py`, and `benchmarks\llm_evaluation.py`. Offline injected timeouts propagated from both chat clients without fallback; generated-code checks rejected two-dimensional/all-infinite outputs and correctly realigned a reversed uniquely indexed Series.
- [x] Executed the established benchmark Python with `-B -m pytest -q -p no:cacheprovider tests\test_splitting.py tests\test_fitted_rules.py tests\test_model_preprocessing.py`, with plugin autoload disabled: **285 passed, 3 warnings, 8.80 seconds**. These tests do not exercise the two counterexamples below.
- [x] Independently reproduced both findings using small in-memory examples on NumPy **2.4.4**, pandas **2.3.3**, and scikit-learn **1.7.2**. No dataset acquisition, provider request, installation, nested agent, source edit, or git write was performed.

## Issues

### Issue 1: Object-backed positive binary targets silently become zero target means
- **Severity**: Medium
- **File**: `featcopilot\engines\fitted_rules.py`
- **Line(s)**: 248–270, especially 262–270
- **Classification**: Newly introduced module/behavior; this engine does not exist at the PR base.
- **Description**: `resolve_task` deliberately recognizes object arrays containing real numbers or booleans, but `_prepare_target` subsequently tests the original array's storage dtype to decide whether to preserve fixed binary 0/1 semantics. Consequently, an all-positive target stored as `object` bypasses the fixed-code branch and is passed through the one-class `LabelEncoder` result, which is entirely zero. The same values in native integer/boolean storage produce entirely one. This contradicts the engine/specification's documented fixed 0/1 policy for one-class numeric/boolean targets.
- **Reproduction**:

  ```python
  import numpy as np
  import pandas as pd
  from featcopilot.engines.fitted_rules import FittedRuleEngine

  X = pd.DataFrame({"category": ["a"] * 10})
  rules = [{
      "name": "te", "operator": "target_mean",
      "columns": ["category"], "parameters": {"smoothing": 0.0},
  }]
  for dtype in (int, object):
      engine = FittedRuleEngine(rules, task="classification")
      training = engine.fit_transform(X, np.array([1] * 10, dtype=dtype))
      print(dtype, training["te"].unique(), engine.transform(X)["te"].iloc[0],
            engine.to_dict()["target_classes"])
  ```

  Observed native integer output: training `[1.]`, inference `1.0`, classes `[0, 1]`.
  Observed object-integer output: training `[0.]`, inference `0.0`, classes `[1]`.
  Replacing the values with `True` reproduces the same discrepancy: native booleans preserve `[False, True]`, while object booleans produce zero and store `[True]`. Both inputs pass explicit classification task validation.
- **Risk**: A storage-dtype change reverses the meaning of the learned binary feature, including its inference mapping and serialized class vocabulary. This also reaches typed semantic plans and the benchmark plan evaluator because they use this engine.
- **Suggested Fix**: Normalize validated numeric/boolean object targets before the fixed-binary check, or recognize their actual scalar types rather than only `dtype.kind`. Preserve the distinction from string labels such as `"1"`. Add equivalence tests for native/object integer and boolean all-positive targets across cross-fitted output, inference, and JSON state round-trips.

### Issue 2: Calibration preflight admits repeat counts that its worker classifiers cannot fit
- **Severity**: Low
- **File**: `benchmarks\calibration_evaluation.py`
- **Line(s)**: 184–198, specifically 190
- **Supporting Locations**: `benchmarks\prediction_ensemble.py:69,222–250`; `featcopilot\classification.py:62,352–364`
- **Classification**: Newly introduced controller/worker configuration mismatch; these modules are additions relative to the PR base.
- **Description**: The study accepts `stability_repeats` in `[1, 100]`, and `PredictionBundleConfig` has no upper bound. Every retained base subsequently enters `NestedBaseClassifier`, whose `_FitSettings` requires at most ten repeats. Thus counts 11–100 pass controller validation, source preparation, and job planning but are guaranteed to fail the bundle workers. This affects raw bundles as well: classifier configuration is validated before the `engineered` branch.
- **Reproduction**:

  ```python
  import numpy as np
  import pandas as pd
  from benchmarks import calibration_evaluation as study
  from benchmarks.prediction_ensemble import CompleteBaseProcedure

  config = {**study.DEFAULTS, "threads": 1, "stability_repeats": 11}
  study.validate_config(config, 1000.0)  # Succeeds.
  X = pd.DataFrame({"x": np.arange(60, dtype=float)})
  y = np.array([0, 1] * 30)
  CompleteBaseProcedure(
      learner="linear",
      config={"seed": 42, "threads": 1, "stability_repeats": 11},
      engineered=False,
  ).fit(X, y)
  ```

  Observed: `_FitSettings` raises a `ValidationError` for `stability_repeats`. The same failure occurs with `engineered=True`, before model fitting.
- **Risk**: A controller-approved CLI configuration unnecessarily acquires/prepares sources and launches doomed workers, yielding incomplete study comparisons rather than a useful preflight configuration error.
- **Suggested Fix**: Match the controller and `PredictionBundleConfig` upper bounds to the supported classifier/predictive limit of ten. Add boundary checks demonstrating that ten is admissible and eleven fails before acquisition/output-state creation.

## Scope and Limitations

- Findings are based on current source, not previous reviewer artifacts or driver notes. Earlier-stage findings/resolutions were not consulted.
- The scope includes the full PR, but execution was deliberately bounded to the listed synthetic/unit checks. No full suite, real benchmark, live provider, SDK end-to-end session, archive regeneration, or cross-version installation was attempted. Generated evidence was assessed through producers and validation code, not a byte-by-byte audit of archived results.
- The benchmark interpreter was `C:\Users\lijiang1\.copilot\session-state\fc6747ca-1d37-46b8-a1c4-fb650ac71116\files\benchmark-env\Scripts\python.exe`.
- Independently verified **pre-existing limitations, not counted as new PR findings**:
  - With pandas 2.3.3, a categorical source `pd.Categorical(["a"] * 6 + ["b"] * 6 + ["c"] * 6)` and numeric targets repeating `0, 10, 30` six times each can raise `TypeError: Cannot setitem on a Categorical with a new category` during target-map `fillna`. Both the base `TabularEngine` and current `fit(...).transform(...)` reproduce it; current cross-fitting also encounters it at `featcopilot\engines\tabular.py:321`. The inference expression at line 548 retains the original limitation. A separate correction would convert mapped numerical values to a noncategorical numerical Series before filling.
  - A singleton object-category input with `onehot_ratio_threshold=0`, `target_encode_ratio_threshold=1`, and `min_samples_per_category=1` still emits its own target as its training encoding. Base and current both emit `3.0` for target `3.0` and `91.0` for target `91.0`; the new fallback at `featcopilot\engines\tabular.py:302–307` preserves rather than introduces that behavior. Rejecting infeasible cross-fitting or using an explicit target-independent prior would close this retained exception to own-target exclusion.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: Pending; reviewer made no source changes.
- **Why**: Object-backed numeric/boolean targets must obey the same fixed binary semantics as native targets.
- **How verified**: Independent native/object integer and boolean counterexamples; current mappings and exported class metadata differ as documented above.

### Issue 2
- **Status**: Open
- **What changed**: Pending; reviewer made no source changes.
- **Why**: Controller admission and the actual worker's supported repeat range must agree.
- **How verified**: Controller validation accepts eleven; both raw and engineered complete base procedures reject it through `_FitSettings` before model fitting.

### Driver resolution update

- **Issue 1 - Fixed, pending independent retry**: Validated object-backed
  targets are inferred to their numeric/boolean storage representation before
  applying the fixed 0/1 policy. Added native/object integer, float and boolean
  equivalence checks for all-positive/all-negative targets under auto and
  explicit classification, including cross-fit output, inference and JSON state.
  A separate string `"1"` check preserves label semantics.
- **Issue 2 - Fixed, pending independent retry**: The calibration controller
  and `PredictionBundleConfig` now enforce the same 1-10 repeat range as
  their worker classifiers. Ten is admitted; eleven and one hundred are
  rejected before acquisition or model fitting.
- **How verified**: Sixteen newly added dtype/state and bound checks failed
  before the changes. Broader affected-suite validation and build are in
  progress before both reviewers' second detailed-correctness attempt.

The broader validation completed successfully: 566 affected tests passed,
package mypy passed 52 files, and the wheel built. Existing upstream
feature-name/probability-roundoff warnings remain; no scores or benchmark
defaults were changed.

## Review Summary
- **Round**: 6
- **Theme**: Polish & hardening
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-6-gpt-6.1-sol.md
- **Issues Found**: 5
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read the supplied stage-6/attempt-2 contract and scope inventory, plus repository `AGENTS.md`. Reviewed against base `bf119812162af87c29bc8d09bb3c705576b94f27` and the current worktree, not merely HEAD. Explicitly inspected the complete untracked `featcopilot\llm\_plan_transport.py`.
- [x] Read complete relevant implementation files: `featcopilot\engines\predictive.py`, `featcopilot\classification.py`, `featcopilot\utils\model_preprocessing.py`, `featcopilot\utils\numeric_clipping.py`, `featcopilot\engines\fitted_rules.py`, and `featcopilot\engines\point_in_time.py`. Traced wrapper routing, semantic-engine execution/validation, Copilot request/cleanup policy, and corresponding user guides.
- [x] Examined benchmark producers rather than archived result bodies: complete `benchmarks\run_state.py`; controller preparation, dispatch, publication and failure handling in `benchmarks\execution.py`; and supervision, aggregation, reporting and source fingerprinting in `benchmarks\tabular.py`. Read the protocol and relevant benchmark README narrative, plus deterministic archive/publication and summary test implementations.
- [x] Independently checked introduction with read-only `git cat-file -e <base>:<path>`: `point_in_time.py`, `predictive.py`, `classification.py`, and `_plan_transport.py` do not exist at the base. Independently read base versions of the best-practices guide and wrapper constructor with `git show`; Issue 5 is demonstrably pre-existing.
- [x] Ran a deliberately small provider-free test selection with the installed benchmark interpreter: `python -B -m pytest -q -p no:cacheprovider tests\test_point_in_time.py tests\test_predictive_logistic_workers.py tests\test_benchmark_tabular.py::test_source_fingerprint_uses_zip_names_even_for_windows_relative_paths tests\test_benchmark_tabular.py::test_summary_uses_dataset_means_and_excludes_incomplete_pairs tests\test_benchmark_tabular.py::test_summary_never_pairs_different_splits_or_sources tests\test_classification.py::test_private_opaque_parameter_report_never_calls_repr`. Result: **54 passed in 4.01 seconds**. Bytecode and pytest caching were disabled. Disabling plugin autoload produced one expected `asyncio_mode` configuration warning; no async tests were selected.
- [x] Executed bounded, in-memory synthetic probes for prefix-scan work, candidate-array lifetime, native-NaN parameter reporting, and explanation provenance. Results and reproduction details are recorded below. Probe environment: Python 3.13.13, NumPy 2.4.4, pandas 2.3.3, sklearn 1.7.2, Pydantic 2.13.3; the native-parameter probe used installed XGBoost 3.3.0.
- [ ] Did not run the full suite or duplicate the parent's full-coverage workload. Did not make provider requests, download/install dependencies, run real-data measurements, open frozen result/archive bodies, read other current-run review feedback/driver metadata, or modify source/evidence.

## Scope and Limitations

This is the independently assigned GPT slot of the human-authorized two-model High override, at maximum reasoning, stage 6 attempt 2. No nested review or additional attempt was launched.

The findings comprise **four introduced concerns and one explicitly pre-existing documentation concern**. The performance observations below establish allocation lifetime and operation counts, not a measured production latency or real-data benchmark regression. Historical numerical headlines were not independently regenerated or authenticated from raw archives; their conservative claim boundaries and relevant producer/test paths were assessed instead. This focused hardening review is not an exhaustive re-derivation of every catalog loader, reference adapter, or test in the large PR.

## Issues

### Issue 1: Point-in-time row counts repeatedly scan and copy the entire eligible history
- **Severity**: Medium
- **File**: `featcopilot\engines\point_in_time.py`
- **Line(s)**: 217–220, 250–262
- **Classification**: Introduced; the point-in-time engine is absent at the base.
- **Description**: Binary searches locate the eligible interval, but `_aggregate` immediately evaluates `np.isfinite` over that whole interval and creates a filtered array for every query. This happens even for `count` without a value column, where fit stored only ones and the answer is already `right - left`. For increasing prediction times on one entity, counting N historical events for N queries therefore performs quadratic history scans and repeated prefix-sized allocations. `last` also scans/copies the complete prefix to return one value.
- **Risk**: Ordinary long entity histories can make prediction-time feature extraction disproportionately expensive despite the apparent indexed lookup. Multiple aggregates multiply this work; splitting queries into batches does not remove it.
- **Reproduction / Evidence**: Built one entity with second-spaced availability timestamps and queries immediately after each successive event, using one `count` aggregation without `value_column`. Instrumenting `_aggregate` to sum `len(values)` while delegating to its unchanged implementation produced:

  | Queries / history rows | History elements passed through the finite filter | Correct output |
  |---|---:|---|
  | 256 | 32,896 | counts 1 through 256 |
  | 512 | 131,328 | counts 1 through 512 |

  Doubling both input sizes nearly quadruples actual scanned elements. This is an operation-count probe, not an elapsed-time extrapolation.
- **Suggested Fix**: Handle row counts directly from the search boundaries. For finite-value counts and other suitable aggregates, cache training-history prefix statistics; for `last`, retain/index finite observations instead of filtering each prefix. Preserve strict cutoff, inclusive lower-window, missing-value and tied-order semantics, and add a deterministic work-bound test. Document any deliberately scan-based aggregate limitations.

### Issue 2: Five-bag importance ranking retains an unused full-scope candidate matrix throughout resampling
- **Severity**: Medium
- **File**: `featcopilot\engines\predictive.py`
- **Line(s)**: 2066–2076
- **Classification**: Introduced; this predictive/resampling implementation is absent at the base.
- **Description**: `retained, _, discarded = self._deduplicate_candidates(...)` binds `_` to the dictionary containing every retained full-scope candidate array. An underscore is still a strong local reference. That matrix is never used by `_rank_importance_resampled`, but remains alive for its entire call while each bag separately computes its own candidate matrix and preprocessing allocations.
- **Risk**: The opt-in memory-intensive path unnecessarily keeps another approximately `8 * scope_rows * retained_candidates` bytes resident during all five learner fits. For illustration, 50,000 rows and 1,000 retained candidates add approximately 381 MiB solely for this unused matrix, in addition to required bag/model allocations. This can turn otherwise feasible resource-limited jobs into memory failures.
- **Reproduction / Evidence**: Used 512 synthetic rows, six numeric originals, 96 planned/retained candidates, `screening="importance"` and `importance_resamples=5`. Wrapped `_deduplicate_candidates` to retain only weak references to the returned candidate arrays and replaced `_rank_importance_resampled` with a provider-free inspection callback. At callback entry, **96 unused scope arrays / 393,216 bytes were still alive**; all became collectible immediately after `_rank_importance` returned. No learner fit or large allocation was needed to establish the lifetime.
- **Suggested Fix**: Explicitly release the returned value dictionary before entering resampling, or provide a deduplication mode that does not return the matrix when only definitions/provenance are needed. Add a lifetime/resource regression test without changing ranking, candidate identities, or frozen evidence.

### Issue 3: A legal native NaN estimator parameter breaks the advertised JSON-safe diagnostics
- **Severity**: Low
- **File**: `featcopilot\classification.py`
- **Line(s)**: 240–245, 879–906
- **Classification**: Introduced; the public classifier/reporting implementation is absent at the base.
- **Description**: `_report_parameters` passes all built-in floats through unchanged, including NaN/infinities. These values are copied into selected-parameter and trial diagnostics, but public `diagnostics()` ultimately calls `json.dumps(report, allow_nan=False)`. Thus a classifier that fits and predicts successfully can have unusable diagnostics for a legitimate supplied native parameter, rather than a JSON-safe representation of that setting.
- **Risk**: Observability/export integrations fail precisely when requesting the supposedly JSON-safe report. This is not a training or prediction failure and should not be represented as one.
- **Reproduction / Evidence**: The following bounded synthetic case fitted and predicted successfully, then raised `ValueError: Out of range float values are not JSON compliant: nan`:

  ```python
  X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 64)})
  y = (X.x > 0).astype(int)
  fitted = FeatureEnsembleClassifier(
      XGBClassifier(n_estimators=2, max_depth=1, n_jobs=1, random_state=42),
      [{"missing": float("nan")}] * 3,
      mode="single",
      engineered=False,
      n_jobs=1,
  ).fit(X, y)
  fitted.predict_proba(X.iloc[:3])  # succeeds
  fitted.diagnostics()             # raises
  ```

  The retained member report contained a float NaN at `selected_parameters["missing"]`. XGBoost's NaN missing-value sentinel is valid native configuration, not malformed input.
- **Suggested Fix**: Convert nonfinite numeric parameter values to an explicit JSON-safe sentinel/type description in the diagnostic normalizer, while preserving the actual supplied learner parameter and existing privacy redaction. Test successful fitting/prediction followed by strict JSON serialization of this report.

### Issue 4: Importance-selected feature explanations incorrectly claim residual screening
- **Severity**: Low
- **File**: `featcopilot\engines\predictive.py`
- **Line(s)**: 1822
- **Classification**: Introduced; the shared predictive feature-construction code is absent at the base.
- **Description**: Numeric candidate explanations unconditionally say “selected by internal residual screening and holdout loss.” Importance mode does not perform residual screening; it fits a joint intended learner and ranks native importances/coefficients. The incorrect explanation survives into the delivered feature plan and public explanation APIs.
- **Risk**: Users inspecting an engineered model receive misleading selection provenance even though `selection_report_["screening"]` correctly reports importance. This undermines the interpretability/audit purpose of these explanations.
- **Reproduction / Evidence**: On 128 synthetic rows with `y = x**2 + small_noise`, fitted `PredictiveTabularEngine(estimator=Ridge(alpha=1.0), task="regression", screening="importance", stability_repeats=1, max_candidates=1, max_features=2)`. The report recorded `screening="importance"` and selected one feature, while its delivered explanation was:

  ```text
  x_sq: square of x; selected by internal residual screening and holdout loss
  ```

- **Suggested Fix**: Use screening-neutral wording in the shared constructor, or annotate delivered explanations with the actual resolved screening policy. Test the explanation alongside the existing importance selection report.

### Issue 5: The existing feature-explosion examples pass unsupported wrapper arguments
- **Severity**: Low
- **File**: `docs\user-guide\best-practices.md`
- **Line(s)**: 309–321, particularly 313 and 319
- **Classification**: **Pre-existing, not introduced by this PR.** At base `bf119812162af87c29bc8d09bb3c705576b94f27`, the same unsupported `polynomial_degree=4` and `polynomial_degree=2` calls appear at lines 297 and 303. The base `AutoFeatureEngineer.__init__` signature at line 139 likewise lacks this parameter.
- **Description**: Both the “too many features” and “controlled generation” examples construct `AutoFeatureEngineer(polynomial_degree=...)`, but that option belongs to the tabular engine/single-engine wrapper, not this constructor.
- **Risk**: Copying the documented recommended example immediately fails rather than demonstrating controlled feature generation. This concern should remain visible, but must not be labeled a new regression or evidence against the new predictive API.
- **Reproduction / Evidence**: `AutoFeatureEngineer(polynomial_degree=2, max_features=50)` raises `TypeError: AutoFeatureEngineer.__init__() got an unexpected keyword argument 'polynomial_degree'`. Independently inspected both relevant base files before classifying the issue.
- **Suggested Fix**: In a suitably scoped documentation follow-up, use `FeatureEngineerTransformer(engine="tabular", polynomial_degree=2, max_features=50)` or the direct `TabularEngine` API, consistent with the corrected pipeline examples elsewhere in this PR.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: None; reviewer remained read-only.
- **Why**: New prediction-time implementation has verified avoidable quadratic work for row counts.
- **How verified**: Current-code synthetic operation-count reproduction above; driver fix and targeted verification pending.

### Issue 2
- **Status**: Open
- **What changed**: None; reviewer remained read-only.
- **Why**: Unused full-scope candidate arrays demonstrably remain resident throughout resampling.
- **How verified**: Current-code weak-reference lifetime probe above; driver fix and targeted verification pending.

### Issue 3
- **Status**: Open
- **What changed**: None; reviewer remained read-only.
- **Why**: A valid fitted classifier must not lose its JSON-safe diagnostics solely because a native parameter uses NaN.
- **How verified**: Installed-XGBoost synthetic fit/predict succeeded and diagnostics failed as shown; driver fix and targeted verification pending.

### Issue 4
- **Status**: Open
- **What changed**: None; reviewer remained read-only.
- **Why**: User-visible feature explanations should describe the actual selection procedure.
- **How verified**: Current-code synthetic importance fit delivered the contradictory explanation above; driver fix and targeted verification pending.

### Issue 5
- **Status**: Deferred
- **What changed**: None; independently recorded as an existing documentation concern.
- **Why**: This exact constructor/example mismatch predates the reviewed base and is not an introduced PR regression.
- **How verified**: Read-only base-file inspection plus the current constructor reproduction above. Any follow-up remains the driver's scope decision.

## Driver resolution update

- **Issue 1 - Fixed**: Count/last histories retain only finite observations at
  fit, preserving stable tied order. Queries use indexed boundaries and a
  constant-time length or final-element lookup, not a copied/filtered prefix.
  Other aggregations retain their numerical implementation and are explicitly
  documented as interval scans; no broader prefix-sum redesign was needed.
  Deterministic tests reject the scanning helper at both 256 and 512 rows and
  preserve row/finite counts, strict cutoffs, inclusive windows, missing-only
  entities, ties, pickle restoration and batch invariance.
- **Issue 2 - Fixed**: The unused scope-value dictionary is explicitly released
  before resampling. A weak-reference regression confirms arrays are already
  collectible when resampling begins; ranking definitions and scores are
  unchanged.
- **Issue 3 - Fixed**: Nonfinite built-in float parameters become explicit
  JSON-safe type/value descriptions in diagnostics only. Actual native learner
  parameters remain unchanged, and opaque parameter redaction is preserved.
  Regressions cover NaN/infinities and a fitted XGBoost NaN missing-value
  sentinel, successful prediction and strict JSON serialization.
- **Issue 4 - Fixed**: Shared numeric feature explanations use screening-neutral
  wording. The selection report remains the authoritative resolved policy;
  an actual importance fit verifies the delivered explanation.
- **Issue 5 - Deferred, pre-existing**: The unsupported legacy documentation
  example remains a separate follow-up, not an introduced regression.
- **How verified**: Twelve regression cases failed before the source repairs;
  all 18 new cases passed afterward. Package mypy passed 52 files, Black and
  Ruff passed. The complete affected classification, importance and
  point-in-time modules passed 236 tests (17 optional skips), and the wheel
  built successfully. Final-tree validation follows the remaining controller
  repairs.
- **Stage outcome**: This second attempt remains `ISSUES_FOUND`, and the stage
  advances with unresolved findings at the High two-attempt cap. Later fixes
  are not an independent clean verdict; no third attempt will be run.

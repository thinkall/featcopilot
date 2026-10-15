## Review Summary
- **Round**: 1
- **Theme**: Broad sweep
- **Mode**: resolved-plan; High, stage attempt 2
- **Model**: gpt-6.1-sol; reasoning=max, as assigned by the driving human
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-1-gpt-6.1-sol.md
- **Issues Found**: 4
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read the generated prompt's first 100 lines and the review-code evidence contract before reviewing. Used the supplied human two-model override; this is one independent reviewer result, not a stage-gate decision.
- [x] Mapped the full 720-file base-to-worktree scope using the adjacent `scope.json`, `git diff --name-only`, and `git diff --numstat`. Reviewed the current working-tree implementations, not just `git diff HEAD` or the trimmed prompt.
- [x] Independently compared against base `bf119812162af87c29bc8d09bb3c705576b94f27`. `git diff --summary` confirms that `benchmarks\execution.py`, `benchmarks\run_state.py`, `benchmarks\regression_averaging.py`, and `featcopilot\engines\predictive.py` are new in this PR. The findings below are in these new paths.
- [x] Reviewed the public wrapper and predictive engine's fit/transform, candidate planning, nested residual fitting, joint importance, frequency-state refitting, target encoding, and selection paths. A synthetic wrapper probe passed for fitted original-column order, unchanged infinity values, and duplicated inference indexes.
- [x] Traced canonical admission/preparation, official/IID/group/time splits, confirmation binding, job construction, isolated dispatch, checkpoint reuse, interruption handling, report generation, and final archiving through `benchmarks\catalog.py`, `execution.py`, `run_state.py`, and `tabular.py`.
- [x] Extended the broad sweep to the new classification procedures, fitted rules, point-in-time engine, reference adapters, plan gate, calibration/prediction bundles, OOF/fixed averaging, regression instrumentation, LLM plan/ledger paths, and the smaller legacy selector/engine/CLI/persistence/dependency diffs. Checked associated tests and protocol documentation where relevant to the findings.
- [x] Ran provider-free synthetic probes with the established `benchmark-env\Scripts\python.exe -B -`: reproduced the class-weight failure, the finalization accounting defect, the official-preparation coverage defect, and successful regression-average execution despite a nonidentity declared target transform. Controller probes used in-memory path/state substitutes or virtual checkpoint files only; no physical experiment directories were created.
- [x] Checked frozen evidence through its producers and bounded integrity checks: six archive indexes matched their supplied scope SHA-256 values; three sampled indexes referenced existing basename-only parts and declared uncompressed hashes. Read and hashed their compressed parts without opening result bodies. `git diff --check HEAD` passed for the pending execution/state/wrapper fixes.
- [x] Did not read prior review artifacts or rely on previous verdicts. Independently checked suspected pre-existing behavior instead: the generic Copilot client's permissive session setup and generated-only feature-name reporting already exist at base. A one-row legacy target-encoding probe also returns the same full-target value at base; these are not counted as newly introduced findings here.
- [ ] Did not rerun the reported 356-test/mypy/wheel validation, invoke providers, acquire real datasets, install dependencies, or reproduce historical measurements. Synthetic tests and source traces do not verify every optional numerical backend or archived empirical claim.

**Reviewed snapshot:** HEAD `46c7db92165cb4078699c3f4cc00ae9855f3c002` plus the pending execution/state/wrapper/documentation/test fixes. Current runner source fingerprint: `04457e6b43ea26b13ff2c3a68c4484ad170d3d2ffa88bd545ec72897f9035a9a`. Pending `git diff --binary HEAD -- benchmarks featcopilot tests docs` SHA-256: `943b63c05623a03bc0d4eace41c438959643310d90d91ea0035751bc8adbddd9`.

**Validation environment:** Python 3.13.13, scikit-learn 1.7.2, pandas 2.3.3, NumPy 2.4.4. This is a broad correctness review, not a claim of line-by-line inspection of every generated evidence file. Only this assigned review artifact was written.

## Issues

### Issue 1: Finalization is outside cumulative active-time accounting and can exceed the ceiling while returning success
- **Severity**: Medium
- **File**: `benchmarks\execution.py`
- **Line(s)**: 859-869; 871-876
- **Classification**: Introduced in the new canonical execution controller.
- **Description**: The last update/save of `state.elapsed_seconds` happens before the final source fingerprint, manifest write, complete report rendering, and result compression/archiving. None of that finalization time is subsequently charged to state or manifest. The success predicate also never checks whether cumulative active time exceeded `budget`. Consequently, a run can publish success beyond its ceiling, and subsequent resumes recover less consumed time than actually elapsed. This conflicts with the documented cumulative active-execution budget (`benchmarks\README.md:310-313` and the CLI help).
- **Risk**: Large evidence sets incur real serialization/compression costs that disappear from the durable budget ledger. Resumes can spend that uncharged time again, and a supposedly bounded experiment can be reported as successful after exceeding its authorized total.
- **Reproduction**: Executed `execute()` with two synthetic baseline/linear cells, a 30-second budget, virtual filesystem/checkpoints, and a controlled clock. Preparation consumed 1 second, each worker 1 second, each of the three report calls 10 seconds, and final archiving 7 seconds. The observed result was:

  ```json
  {
    "budget_seconds": 30.0,
    "active_clock_seconds": 40.0,
    "persisted_elapsed_seconds": 23.0,
    "manifest_active_seconds": 23.0,
    "exit_code": 0
  }
  ```

  The earlier reports are accounted for because they precede another clock update; the last report/archive are not. The probe performed no physical writes or actual waiting.
- **Suggested Fix**: Include final rendering/archiving in cumulative accounting, persist the final charged elapsed time, and make budget overrun explicit in the manifest and return status. Reserve cleanup/report time before dispatch rather than treating finalization as free. Add a controlled-clock test that advances time inside the final report and archive calls, then verifies persisted time and resume's remaining allowance.

### Issue 2: Failed preparation invents an IID-sized cell grid for declared official development splits
- **Severity**: Medium
- **File**: `benchmarks\execution.py`
- **Line(s)**: 672-688
- **Classification**: Introduced in the new official-split execution/preparation path.
- **Description**: The predeclared fallback split grid uses `config["folds"]` unless there is a confirmation preregistration. For an ordinary development dataset with `split_policy="official"`, the runner ignores the declared `source.official_split_dimensions`. Successful preparation later replaces this grid with all official splits, but a preparation error, interruption before preparation, or pre-preparation budget exhaustion leaves the invented IID-sized grid in place. `plan_jobs()` then publishes this smaller grid as `planned_cells` and as each row's `expected_cells`.
- **Risk**: Coverage and requested-cell counts depend on whether acquisition/preparation succeeded. For example, an official ten-fold source failing preparation can contribute only three failures under the default configuration instead of ten. The resulting failure denominator understates the requested protocol even though every emitted row is explicitly marked failed.
- **Reproduction**: Executed the controller with an in-memory development spec declaring official dimensions `repeats=2, folds=3, samples=1`, one seed/method/learner, and CLI `folds=2`. Made preparation raise a synthetic `ValueError`, before any worker could dispatch. The persisted manifest and state reported:

  ```json
  {
    "exit": 1,
    "declared_splits": 6,
    "planned_cells": 2,
    "recorded_cells": 2
  }
  ```

  The same declared official source would request six cells if its payload prepared successfully. This is not limited to malformed custom metadata: admitted official development sources in `benchmarks\suites\development_expansion_20260922.json` declare these dimensions.
- **Suggested Fix**: Derive the pre-preparation official grid from its locked source dimensions, with the existing explicit confirmation split-plan selection taking precedence. Never substitute the IID fold count for a declared official design. If official dimensions are genuinely unavailable, represent that uncertainty explicitly rather than claiming a complete requested grid. Add development-official tests for preparation error, interruption, and exhausted preparation budget.

### Issue 3: Dispatch drops target-transform policy, bypassing regression averaging's explicit rejection
- **Severity**: Medium
- **File**: `benchmarks\execution.py`
- **Line(s)**: 726-741
- **Classification**: Introduced integration defect between the new catalog, dispatcher, and regression-average method.
- **Description**: The catalog supports and applies `target_transform="log1p"` (`benchmarks\catalog.py:1112-1117`). Its metadata retains that policy, but the prepared data dictionary and per-cell payload do not carry `target_transform` or equivalent dataset policy metadata. `regression_averaging._validate_policy()` therefore sees its default `"identity"` rather than the real policy. The explicitly identity-only method can execute successfully on transformed targets, and its result then hardcodes `"target_transform": "identity"` (`benchmarks\regression_averaging.py:51-52,327`).
- **Risk**: A supported catalog/custom-suite configuration silently bypasses a declared method restriction and publishes false provenance. Those scores measure the transformed target scale, not the identity-target protocol advertised by the regression-average result. This finding does not assert that the currently published identity-target measurements are wrong.
- **Reproduction**: Ran the canonical controller with a synthetic regression spec and preparation metadata both declaring `target_transform="log1p"`. The in-memory prepared target was `log1p` of positive raw values. Used the actual `tabular.run_case()` and actual regression averaging/linear procedures, with two folds, one seed, `max_added_features=1`, `max_candidates=5`, and one stability repeat; only filesystem/controller infrastructure was virtualized. Both averaging cells returned success:

  ```json
  {
    "exit": 0,
    "average_cells": 2,
    "status": ["success", "success"],
    "claimed_transforms": ["identity", "identity"]
  }
  ```

  By contrast, directly supplying `payload["target_transform"]="log1p"` makes the method reject as intended; the existing direct-input tests cover that narrower case, not dispatch's omission.
- **Suggested Fix**: Preserve authoritative dataset policy in the hydrated worker payload and validate method restrictions against it. Since the transform is already declared before acquisition, reject this method/dataset combination before running models; retain worker-side validation as defense in depth. Add an end-to-end synthetic catalog-to-dispatch test proving the nonidentity transform cannot disappear. Supporting transformed targets instead would require an explicitly different protocol and truthful provenance, not the current identity fallback.

### Issue 4: Predictive selection encodes labels without remapping the injected estimator's class weights
- **Severity**: Medium
- **File**: `featcopilot\engines\predictive.py`
- **Line(s)**: 781-790
- **Classification**: Introduced in the new predictive public-engine/wrapper API.
- **Description**: Classification labels are replaced with contiguous `LabelEncoder` codes, but `_learner()` clones the supplied estimator with its original `class_weight` dictionary intact. A valid native classifier whose weight keys are original string labels or nonzero-based numeric labels is then fitted with incompatible encoded labels. This breaks the supported injected-learner path even if the output budget allows zero additions. The separate new classification API already handles this mapping in `_encoded_weight_parameters()`, but the directly exposed predictive engine does not.
- **Risk**: Class-weighted predictive feature selection fails on otherwise valid, common imbalanced-classification configurations; depending on the learner, label-specific weighting can also be misinterpreted rather than preserved. Users must not have to guess the engine's private encoded vocabulary to configure their downstream estimator.
- **Reproduction**: Executed this provider-free synthetic case in the established environment:

  ```python
  import numpy as np
  import pandas as pd
  from sklearn.linear_model import LogisticRegression
  from featcopilot import AutoFeatureEngineer

  rng = np.random.default_rng(8)
  X = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
  y = np.where(X.a > 0, "yes", "no")
  learner = LogisticRegression(class_weight={"no": 1.0, "yes": 3.0})
  learner.fit(X, y)  # succeeds with original labels

  AutoFeatureEngineer(
      selection_strategy="predictive",
      task="classification",
      selection_estimator=learner,
      max_features=2,
      max_candidates=5,
      stability_repeats=1,
      screening="importance",
      leakage_guard="off",
      gate_n_jobs=1,
  ).fit_transform(X, y)
  # ValueError: The classes, [0, 1], are not in class_weight
  ```

- **Suggested Fix**: Remap original-label class-weight dictionaries on the owned estimator clone using the same fitted label vocabulary, preserving partial dictionaries and `"balanced"` semantics without mutating the caller's estimator. Reuse/share the classification API's policy where appropriate. Test string and nonzero-based numeric labels, unchanged prototypes, and both direct `PredictiveTabularEngine` and `AutoFeatureEngineer` entry points.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: Pending; reviewer made no source edits.
- **Why**: Finalization time must remain part of the cumulative authorized budget.
- **How verified**: Controlled-clock, memory-only controller reproduction above.

### Issue 2
- **Status**: Open
- **What changed**: Pending; reviewer made no source edits.
- **Why**: Requested official coverage must not shrink when preparation fails.
- **How verified**: Synthetic official development preparation failure persisted two cells for a declared six-split source.

### Issue 3
- **Status**: Open
- **What changed**: Pending; reviewer made no source edits.
- **Why**: A missing dispatch field must not authorize a forbidden target policy or fabricate identity-target provenance.
- **How verified**: Actual synthetic regression-average procedures completed through the virtualized controller with nonidentity metadata and false identity claims.

### Issue 4
- **Status**: Open
- **What changed**: Pending; reviewer made no source edits.
- **Why**: Private label encoding must preserve the semantics of supported injected learners.
- **How verified**: Direct native classifier fit succeeded; the predictive public wrapper failed on the same original-label class weights.

### Driver resolution update: Issue 4

- **Status**: Fixed after the stage's two-attempt cap; no claim that stage 1 was clean.
- **What changed**: Extracted the existing classifier label-weight translation into `utils/model_preprocessing.py`, reused it from `classification.py`, and applied it to owned predictive estimator parameters using the current fitted label vocabulary. The vocabulary resets before every fit; caller estimators and dictionaries remain unchanged.
- **Why**: Label encoding is internal implementation detail and must not alter caller-supplied class-weight semantics. A shared helper avoids divergence between the two public classification surfaces.
- **How verified**: Added direct-engine and public-wrapper tests for string/nonzero integer labels, full/partial dictionaries, `"balanced"`/`None`, encoded-label equivalence, caller immutability, unknown-key rejection before fitting and failed-refit recovery. The selected predictive/classifier/worker-policy suite passed 56 tests; package mypy passed all 51 files.

### Driver resolution update: Issues 1-3

- **Status**: Fixed after the stage's attempt cap; retained as findings in the original reviewer text.
- **What changed**: The execution controller now reserves publication time, charges final report/archive work and records any overrun. Failed official preparation uses declared source dimensions with confirmation selection taking precedence. Authoritative target-transform policy reaches worker payloads, and unsupported regression averaging is rejected before model execution.
- **Why**: The complete controller lifecycle must honor cumulative budgets, invariant requested denominators and truthful dataset policy. No historical result or protocol was rewritten.
- **How verified**: Controlled-clock finalization/overrun/resume tests, official development failure/interruption/budget tests and end-to-end target-policy dispatch regressions are included in the execution/state suite. All 221 targeted tests passed with the pinned interpreter; Black, Ruff and whitespace checks passed.
- **Limit**: Publication reserves are estimates, not hard real-time deadlines. Overruns are charged and surfaced rather than treated as successful bounded execution.

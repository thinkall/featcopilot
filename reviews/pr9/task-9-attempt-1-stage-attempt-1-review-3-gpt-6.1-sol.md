## Review Summary
- **Round**: 3
- **Theme**: Edge cases & robustness
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Reasoning**: max
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-3-gpt-6.1-sol.md
- **Issues Found**: 3
- **Verdict**: ISSUES_FOUND
- **Scope**: Base `bf119812162af87c29bc8d09bb3c705576b94f27` to the current worktree, including uncommitted source/test fixes and untracked `featcopilot\llm\_plan_transport.py`. HEAD was `62cbb1cf4f434a6bbfd956b34c2615e761441e46`; this was not a HEAD-only review.

## Evidence Checklist
- [x] Read the assigned generated prompt's first 100 lines, the substantive path inventory in `scope.json`, and `AGENTS.md`. Used exact-base diffs and `git ls-tree` to check introduction independently. Neither prior reviewer findings nor driver-progress artifacts were consulted.
- [x] Inspected the predictive engine's input validation, fit/reset/transform, candidate computation, frequency refitting, nested splitting and learner-fitting paths; fitted-rule and point-in-time engines; classification preprocessing and classifier publication/invalidation; and both sklearn wrappers. Checked the shared splitting, frequency and numeric-clipping helpers where those paths depend on them.
- [x] Inspected the untracked isolated-plan transport, production plan generator, semantic-engine fit/training/inference hooks, provider startup/send/cleanup paths, and evaluation-ledger reservation/finalization. An entirely in-memory cancellation probe of `IsolatedCopilotTransport.session` confirmed one session disconnect and one client stop, with tools disabled. It did not create a workspace or start an SDK client.
- [x] Inspected canonical benchmark preparation, dispatch, interruption, resume, terminal-protocol and publication paths in `benchmarks\execution.py`, persistence/archive producers in `benchmarks\run_state.py`, the shared process supervisor in `benchmarks\tabular.py`, and calibration/blending controller failure/resume paths.
- [x] Ran **13 focused, provider-free synthetic tests: 13 passed in 7.37 seconds**. Used the established `files\benchmark-env\Scripts\python.exe`, `-B`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and `-p no:cacheprovider`. Selected tests did not use file-writing fixtures. The sole warning was the expected unknown `asyncio_mode` setting with plugin autoload disabled. Exact selectors:
  ```text
  tests\test_predictive_engine.py::test_refit_and_failed_refit_clear_state
  tests\test_predictive_engine.py::test_batch_invariance_schema_indices_and_metadata
  tests\test_predictive_engine.py::test_wrong_estimator_task_and_unsupported_fit_arguments
  tests\test_fitted_rules.py::test_inference_schema_errors_and_failed_refit_clear_state
  tests\test_fitted_rules.py::test_rowwise_missing_nonfinite_and_overflow_policy
  tests\test_classification.py::test_failure_in_last_bag_does_not_publish_partial_or_old_model
  tests\test_classification.py::test_fixed_average_atomic_refit_set_params_and_caller_configuration
  tests\test_classification.py::test_public_schema_and_class_coverage_fail_closed
  tests\test_llm_plan_mode.py::test_empty_valid_plan_is_identity_not_a_fabricated_feature
  tests\test_llm_plan_mode.py::test_predeclared_grouped_target_plan_needs_no_provider_and_excludes_own_group
  tests\test_llm_plan_mode.py::test_temporal_plan_forwards_label_end_times_and_serializes_prior_marker
  tests\test_llm_plan_mode.py::test_plan_label_availability_errors_propagate_and_clear_fitted_state
  ```
- [x] Independently reproduced each issue below with small synthetic data or in-memory filesystem/process doubles. Also exercised the actual canonical `execute` function with the actual launcher and mocked persistence/acquisition to confirm Issue 3's missing terminal checkpoint.
- [x] Respected documented experimental constraints: unsupported group/time target encoding, insufficient per-class support, fixed split/member policies, and absence of live-provider/token-limit guarantees were not reported as defects.
- [x] Reviewed source identities: `featcopilot\transformers\sklearn_compat.py` SHA-256 `f3b994ebef7c0670f8e3ea221b48b0dad3482fb62d29ad10b58b3e10c1baa956`; `benchmarks\tabular.py` `821438e74564c293403afa13d1d2a59744b214b16a9e489db20ef146059de2c1`; `benchmarks\execution.py` `693e2181aa360f67d8d94a454c1e3d94dc34214fd2b8493d009215d1ea33ce9f`. Runtime: Windows, Python 3.13.13, NumPy 2.4.4, pandas 2.3.3, sklearn 1.7.2.
- [ ] No full-suite rerun, live provider, downloads, installs, real-data benchmark, or generated archive-body inspection was performed. Process failure/cancellation probes used doubles rather than real child termination. This is targeted robustness coverage of the paths listed above, not an exhaustive line-by-line audit of every substantive path or every dependency/platform combination.

## Issues

### Issue 1: Unsupervised fitted plans forward fit-only boundary metadata into inference
- **Severity**: Medium
- **File**: C:\code\featcopilot\featcopilot\transformers\sklearn_compat.py
- **Line(s)**: 668–671; call site 773
- **Description**: `fit_transform` identifies the training path solely by whether `y` is non-`None`. With a valid unsupervised LLM plan, it therefore calls `engine.transform(X, **fit_params)` instead of the training hook. The newly supported `groups`, `timestamps` and `label_end_times` fit options reach `FittedRuleEngine.transform`, which rejects them. This is not an unsupported supervised split: `AutoFeatureEngineer.fit` explicitly permits boundary metadata for the fitted-plan path, and fitted-rule fitting explicitly ignores it when no `target_mean` rule is present.
- **Risk**: A provider-free frequency/rowwise plan fails through the public wrapper in workflows that consistently supply boundary metadata, although the same plan and metadata work directly through `SemanticEngine`. Failure occurs after the wrapper has marked itself fitted, so `fit_transform` raises while subsequent plain `transform` succeeds.
- **Introduction**: The training/inference helper and the fitted-plan/boundary-metadata route are additions relative to the exact base. The defective branch is in the added helper, not an inherited limitation of an existing base plan API.
- **Reproduction**:
  ```python
  import pandas as pd
  from featcopilot import AutoFeatureEngineer
  from featcopilot.llm.semantic_engine import SemanticEngine

  X = pd.DataFrame({"category": ["a", "a", "b", "b"]})
  groups = [0, 0, 1, 1]
  rules = [{"name": "frequency", "operator": "frequency", "columns": ["category"]}]
  options = {"feature_format": "plan", "rules": rules}

  direct = SemanticEngine(**options).fit_transform(X, groups=groups)
  assert direct["frequency"].tolist() == [0.5, 0.5, 0.5, 0.5]

  wrapper = AutoFeatureEngineer(engines=["llm"], llm_config=options)
  wrapper.fit_transform(X, groups=groups)
  # TypeError: Unsupported fitted-rule transform options: ['groups']
  ```
  Independently observed afterward: `wrapper._is_fitted` was `True`, and `wrapper.transform(X)["frequency"]` returned the expected four `0.5` values.
- **Suggested Fix**: Distinguish training transformation explicitly from the presence of a target, and invoke the training hook from `fit_transform` even when `y=None`. Retain normal inference rejection of fit-only options and preserve other engines' transform-relevant kwargs. Add provider-free wrapper tests for an unsupervised plan with boundary metadata.

### Issue 2: A process-monitoring exception leaves the benchmark worker unsupervised
- **Severity**: Medium
- **File**: C:\code\featcopilot\benchmarks\tabular.py
- **Line(s)**: 1127–1155
- **Description**: After `Popen` succeeds, cleanup is only entered for explicit time/memory-limit decisions or `KeyboardInterrupt`. An operational error such as `psutil.AccessDenied` from `memory_info()` escapes the loop without stopping or waiting for the worker tree. Closing the log context does not terminate the subprocess. The input-pickle cleanup after the context is also bypassed.
- **Risk**: A case can continue consuming CPU/memory and writing results after its supervisor has failed; its advertised resource limits are no longer enforced. Controllers that catch the exception can then start another case while the previous worker is still running. The per-attempt input payload also remains on disk.
- **Introduction**: `benchmarks\tabular.py`, including this process supervisor, does not exist at the exact base. This is an introduced resource-cleanup failure in the new benchmark infrastructure.
- **Reproduction**: The following in-memory path double is also used for Issue 3. It deliberately performs no filesystem operations:
  ```python
  import contextlib
  import io
  from unittest.mock import Mock, patch
  import psutil
  from benchmarks import tabular

  class MemoryPath:
      def __init__(self, name="case", document="[]"):
          self.name, self.document = name, document
      def __truediv__(self, name):
          return MemoryPath(name, self.document)
      def mkdir(self, **kwargs): pass
      def write_bytes(self, value): pass
      def open(self, *args, **kwargs):
          return contextlib.nullcontext(io.StringIO())
      def exists(self): return True
      def read_text(self, **kwargs): return self.document
      def unlink(self, **kwargs): pass
      def __str__(self): return self.name

  payload = {"seed": 42, "config": {"threads": 1, "learners": ["linear"]}}
  process = Mock(pid=424242)
  process.poll.return_value = None
  parent = Mock()
  parent.children.return_value = []
  parent.memory_info.side_effect = psutil.AccessDenied(pid=424242)

  with (
      patch.object(tabular.subprocess, "Popen", return_value=process),
      patch.object(psutil, "Process", return_value=parent),
      patch.object(tabular, "stop_process_tree") as cleanup,
  ):
      try:
          tabular.run_isolated(payload, MemoryPath(), 10)
      except psutil.AccessDenied:
          pass
      assert process.poll() is None
      cleanup.assert_not_called()
      process.kill.assert_not_called()
  ```
  Observed: `AccessDenied` escaped, tree-cleanup calls `0`, worker-kill calls `0`, simulated worker still running.
- **Suggested Fix**: Enclose the successfully launched process's entire lifetime in unconditional cleanup: on every exceptional exit, terminate/wait for any still-running owned worker tree, and remove the input payload in `finally`. Preserve the original failure and make cleanup best-effort across individual descendant failures so one inaccessible child cannot prevent cleanup of the owned parent. Add a monitor-error regression test alongside timeout/interruption tests.

### Issue 3: Malformed worker files bypass the canonical terminal-protocol handler
- **Severity**: Medium
- **File**: C:\code\featcopilot\benchmarks\tabular.py
- **Line(s)**: 1159–1167; caller C:\code\featcopilot\benchmarks\execution.py:967–968, 983–988
- **Description**: The launcher parses and decorates worker output before `_validated_worker_row` can validate the result contract. A decoded `null`, a list containing a scalar, or malformed JSON raises `TypeError`/`JSONDecodeError` inside `run_isolated`, rather than `BenchmarkProtocolError`. The canonical controller only catches the latter as terminal, so these failures exit through its state-save `finally` without executing terminal-status/publication logic. The documented contract says malformed worker result contracts become terminal `protocol_error`, with subsequent resume rejected.
- **Risk**: Invalid worker evidence can leave a new run without any final `run_status` or `protocol_error`, rather than a durable terminal failure. A resume is not blocked by the existing terminal-protocol guard and can reattempt the malformed cell. For JSON parsing failures, the launcher's later input-payload cleanup is skipped too.
- **Introduction**: Both the shared launcher and canonical controller are new relative to the exact base. The current worktree's terminal-protocol validation does not cover errors raised before its validator receives a result.
- **Reproduction**: Using `MemoryPath` and `payload` from Issue 2:
  ```python
  from benchmarks import execution

  for document in ("[42]", "null", "{"):
      process = Mock(pid=424243, returncode=0)
      process.poll.return_value = 0
      with patch.object(tabular.subprocess, "Popen", return_value=process):
          try:
              result = tabular.run_isolated(payload, MemoryPath(document=document), 10)
              execution._validated_worker_row(result, payload)
          except Exception as error:
              print(document, type(error).__name__,
                    isinstance(error, execution.BenchmarkProtocolError))
  ```
  Observed:
  ```text
  [42] TypeError False
  null TypeError False
  { JSONDecodeError False
  ```
  An additional fully in-memory probe called the actual `execution.execute` with the actual `run_isolated`, a completed fake process returning `[42]`, synthetic preparation, an in-memory `RunState`, and intercepted path/persistence operations. It raised `TypeError`; only the initial manifest was written, `run_status` and `protocol_error` were absent, and no cell was recorded. No real process or file was used.
- **Suggested Fix**: Validate decoded result structure before decorating rows, and route JSON/worker-contract decoding failures into the canonical terminal-protocol exception/checkpoint path. Preserve legitimate operational worker failures as retryable failures rather than treating every launcher exception as a protocol violation. Test through the real launcher boundary with malformed JSON, `null` and non-object rows; mocking `run_isolated` to return malformed data misses this failure.

## Driver resolution update: Issue 1

- **Status**: Fixed, pending independent stage retry.
- **What changed**: `_transform_fitted_engines` now receives an explicit training flag from `fit_transform`; it invokes training hooks even when no target is supplied. Inference continues to reject fit-only metadata.
- **Why**: Supervised target presence is not a reliable indicator that a transform is operating on training rows.
- **How verified**: Three provider-free grouped/timestamped/label-availability plan regressions failed before the change and passed afterward. Duplicate indexes and ordinary inference behavior remain covered. The combined engine/wrapper/predictive suite passed 291 tests with two optional skips.

## Driver resolution update: Issues 2-3

- **Status**: Fixed, pending independent retry.
- **What changed**: The shared supervisor performs unconditional bounded
  cleanup and input removal after worker launch. Monitoring failures remain
  explicit operational failures. Worker JSON and row structure are validated
  before decoration and raise the shared terminal protocol error on malformed
  content, reaching the canonical terminal-status handler.
- **Why**: Resource ownership cannot depend on which exception escapes, and
  protocol validation must precede use of untrusted worker output.
- **How verified**: Monitor/cleanup exception doubles, malformed real-file
  boundary cases, terminal resume rejection and existing worker-limit cases
  are covered in the 456-test supervisor/execution/state suite. Black, Ruff
  and whitespace checks passed. OS-denied termination remains best-effort.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: No production changes; this read-only review records the finding.
- **Why**: Training-path selection must not depend on whether the plan needs a target.
- **How verified**: Provider-free direct-engine/wrapper reproduction above; direct output succeeds and wrapper raises after fitting. No fix verification performed.

### Issue 2
- **Status**: Open
- **What changed**: No production changes; this read-only review records the finding.
- **Why**: Resource guarantees require cleanup on operational exceptions as well as configured limit/cancellation branches.
- **How verified**: In-memory `psutil.AccessDenied` probe left the simulated process running with zero cleanup calls. No real child was spawned and no fix verification performed.

### Issue 3
- **Status**: Open
- **What changed**: No production changes; this read-only review records the finding.
- **Why**: Malformed result files must reach the documented durable terminal-protocol path before resume is possible.
- **How verified**: Three malformed-output probes through the actual launcher, plus an actual-controller/in-memory-persistence probe confirming absent terminal metadata. No fix verification performed.

## Review Summary
- **Round**: 5
- **Theme**: Testing & coverage
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-5-grok-4.7.md
- **Issues Found**: 3
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read the stage-5 attempt-2 input header and `review-pr9/scope.json` (721 paths, base `bf119812162af87c29bc8d09bb3c705576b94f27`, HEAD `62cbb1cf4f434a6bbfd956b34c2615e761441e46`). Assessed current worktree source and tests, including untracked `featcopilot/llm/_plan_transport.py`. Did not read prior review artifacts or driver progress.
- [x] Allowlist separator contract is tested and passed: `tests/test_benchmark_run_state.py:300` and `tests/test_calibration_evaluation.py:189` force `PureWindowsPath`, assert `pkg\mod.py`, then require `source_allowlist == ["pkg/mod.py"]` equal to zip members on init and resume.
- [x] Exhausted-resume fingerprint checks are meaningful. `test_confirmation_exhausted_resume_rejects_retained_input_drift` logged `Cannot resume: inputs or split changed` for both metadata and split tampering and did not open the ledger. `test_retained_input_drift_is_rejected_before_any_resume_write` also asserts that message. Blending `test_exhausted_resume_still_refingerprints_cached_raw_without_rerunning_successes` asserts a second acquisition and no extra worker calls.
- [x] Fake/real SDK boundary is the real transport plus a mocked client. `isolated_copilot_sdk` / `copilot_sdk` patch `copilot.CopilotClient` (autospec against installed github-copilot-sdk 0.3.0) and assert isolation kwargs, one send, stop/disconnect, and no raw row text. Deadline, cleanup, and workspace OSError tests call `IsolatedCopilotTransport.request`, not a replaced transport.
- [x] Numeric dtype and index assertions that were run are value- and type-level, not existence checks: `test_timeseries_numpy_index_scalars_preserve_feature_values` requires builtin `float`/`int`; `test_dataframe_schema_index_categories_and_boolean_values_are_preserved` uses `Index.equals` on a duplicate named index; fitted-rule and preprocessing index/dtype tests use `assert_frame_equal` / native-vs-object equality. Complex numeric sources raise `real numeric`.
- [x] Focused pytest, not the parent 611-test run: 43 passed in 15.63s (`test_initialization_source_allowlist_matches_zip_names_with_windows_separators`, calibration allowlist, exhausted resume fingerprint tests, plan deadline/grace/adapter deadline, numeric role rejection, timeseries index scalars, clipping index, fitted-rule index).
- [x] Direct probes, not suite evidence: live `source_files()` returned 121 paths with cache/result dirs excluded and backslash prefixes; `source_fingerprint()` is 64 hex chars. Transport cleanup with `timeout=8` and a hung `stop` elapsed 5.008s (`PLAN_CLEANUP_TIMEOUT`). `SemanticEngine(timeout=inf)` constructs; `CopilotPlanProvider(timeout=inf)` rejects; `send_plan` inside a running loop raises the documented `RuntimeError`. `rg` found zero test references to `PLAN_CLEANUP_TIMEOUT` and zero test calls of `source_fingerprint()`.
- [ ] Full-suite, branch coverage, and mutation score not run. Parent 611 affected tests, mypy, and wheel success were not treated as proof. Optional `importorskip("copilot")` and the Python 3.10 SDK install skip are not coverage defects. Historical archive bytes were not revalidated.

## Scope and limits
Reviewed the full PR worktree from base `bf11981` through pending fixes, not only the latest test edits. Excluded `reviews/`, `submissions/`, session files, and raw frozen benchmark archives. No source edits, git writes, package installs, provider calls, or real dataset benchmarks. Established interpreter: `C:\Users\lijiang1\Miniforge3\envs\flaml313\python.exe` with `github-copilot-sdk` 0.3.0 (legacy `config`/`auto_start` constructor). Real Copilot subprocesses were not started; isolation is asserted on the fake client boundary.

## Issues

### Issue 1: Production source identity helpers are never executed and embed OS separators
- **Severity**: Medium
- **File**: benchmarks/tabular.py
- **Line(s)**: 1530-1550
- **Description**: `source_files()` and `source_fingerprint()` are new resume-identity inputs, but no test calls either. Study fixtures replace both (`tests/test_benchmark_execution.py:34`, `tests/test_blending_evaluation.py:49`, `tests/test_calibration_evaluation.py:53`). The allowlist tests only prove `as_posix()` when a path list is injected. On this Windows worktree the live helper hashed `str(relative)` prefixes such as `featcopilot\__init__.py` (`has_backslash` true, 121 selected paths, caches excluded). The same bytes on Linux would hash `featcopilot/__init__.py`. That disagrees with the tested allowlist contract (`pkg/mod.py`) and cannot be seen by ubuntu-only CI or by the stubbed study tests.
- **Risk**: A selector regression (include `__pycache__` / `tabular-results`, drop `featcopilot` or `pyproject.toml`) or a separator-only difference changes `source_sha256` without a red test. Windows and Linux recomputations of the same tree will not match, so published source identity is not reproducible across the developer OS and CI. This is not a request to rewrite historical result archives; those keep their recorded hashes.
- **Suggested Fix**: Add a regression that calls the real helpers (or the same functions with `ROOT` pointed at a temp tree) and asserts cache/result exclusion plus inclusion of `pyproject.toml` and the suite file. Build the fingerprint preimage with `relative.as_posix()` and assert equality with `source_fingerprint()`. That assertion fails today on Windows until line 1549 stops using `str(path.relative_to(ROOT))`. Do not regenerate frozen archives.

### Issue 2: Shutdown grace cap is not what the deadline tests measure
- **Severity**: Medium
- **File**: featcopilot/llm/_plan_transport.py
- **Line(s)**: 22, 99-101, 171
- **Description**: Cleanup uses `min(timeout, PLAN_CLEANUP_TIMEOUT)` and the sync watchdog repeats that cap (`featcopilot/llm/copilot_client.py:562`). Every hanging cleanup or startup test uses `timeout=0.03` (`tests/test_llm_plan_mode.py:497`, `520`, `574`, `601`, `695`). At that value `min(timeout, 5)` equals `timeout`, so the tests cannot see the 5-second cap. A direct probe of `IsolatedCopilotTransport.request(..., timeout=8)` with a hung `stop` elapsed 5.008s and still invoked stop once. No test mentions `PLAN_CLEANUP_TIMEOUT`. The `< 2` elapsed checks do catch an unconditional 5-second wait when the request timeout is 0.03; they do not catch `wait_for(close(), timeout=timeout)` once the request timeout is the default 60s or larger.
- **Risk**: Replacing the cap with the full request timeout keeps the current suite green and can block a caller for the whole request deadline on each hung `disconnect` and `stop`. Default plan timeout is 60s; benchmark case timeouts are much larger.
- **Suggested Fix**: Add a provider-free regression with request timeout 8 seconds, a successful send, and a hung stop. Assert elapsed time is above 4.5s and below 6.5s, disconnect and stop are both awaited, and the successful plan is not returned. Keep the existing 0.03s tests for the shared startup deadline. Do not use a multi-minute timeout in CI.

### Issue 3: Non-finite timeout and running-loop rejection have no tests
- **Severity**: Low
- **File**: featcopilot/llm/_plan_transport.py
- **Line(s)**: 105-106
- **Description**: The transport rejects non-finite or non-positive timeouts before creating a workspace, but nothing in `tests/` calls that guard. Probes: `nan`/`inf`/`0`/`-1`/`False` raise `ValueError`; `True` is accepted because `math.isfinite(True)` is true. `SemanticEngine(timeout=inf)` and `CopilotConfig(timeout=inf)` construct (`featcopilot/llm/semantic_engine.py:34`, `featcopilot/llm/copilot_client.py:37`, pydantic `gt=0` allows infinity). `CopilotPlanProvider` rejects infinity, bools, and non-numbers (`benchmarks/llm_evaluation.py:320-326`) and `send_plan` inside a running loop raises `RuntimeError: Run synchronous Copilot evaluation outside an active asyncio loop...`. Neither provider guard has a test. `True` is coerced to `1.0` by the engine config before it reaches the transport, so the engine path hides the transport bool hole.
- **Risk**: Deleting the transport finite check or the provider running-loop guard would not fail CI. An engine constructed with `timeout=inf` would then hang in `asyncio.wait_for`. The two public entry points also disagree, and no test records which contract is intended.
- **Suggested Fix**: Parametrize `nan`, `inf`, `0`, `-1`, `True`, and `False` for `CopilotPlanProvider` and assert `ValueError` before `CopilotClient` is called. Lock one engine contract: reject non-finite timeouts at config construction, or at `fit` before `client.start`. From a running loop, `send_plan` must raise the documented `RuntimeError` and must not start the SDK.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 2
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 3
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Driver resolution update: Issue 1

- **Status**: Fixed after the stage's second-attempt cap.
- **What changed**: Source fingerprint preimages use `.as_posix()` names.
  New tests call the real source selection/fingerprinting functions against
  a temporary source tree, checking required files, excluded generated paths,
  exact preimage and dirty-source detection. A simulated Windows relative
  path test runs on every host.
- **Why**: Path-separator differences should not change the identity of
  otherwise identical source bytes, and source-selection regressions need
  executable tests rather than only mocked controller fixtures.
- **How verified**: Both source-hash regressions failed before the fix on
  Windows. Historical sources and archived fingerprints were not regenerated.

### Driver resolution update: Issues 2-3

- **Status**: Fixed after the testing stage's second-attempt cap.
- **What changed**: Instrumented cooperative tests assert the actual cleanup
  boundary receives five seconds for request timeouts of eight/sixty seconds,
  and the shorter request timeout for 0.03 seconds. Transport/provider guards
  reject invalid numeric and boolean deadlines before workspace/client startup.
  Copilot/semantic timeout configuration rejects nonfinite/nonpositive/boolean
  inputs while preserving normal integer and float timeouts. The benchmark
  provider's running-loop rejection is now covered without an SDK call.
- **Why**: The old short-timeout tests could not detect loss of the cleanup
  cap, and invalid configuration guards need direct no-side-effect assertions.
- **How verified**: The 457-test provider/plan suite passed on SDK 0.3.0 and
  1.0.16; the driver reran all 457 tests, package mypy (52 files) and the wheel
  build successfully. No real provider or benchmark measurements were used.

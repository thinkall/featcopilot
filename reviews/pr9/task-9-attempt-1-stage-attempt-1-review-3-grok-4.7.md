## Review Summary
- **Round**: 3
- **Theme**: Edge cases & robustness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-3-grok-4.7.md
- **Issues Found**: 7
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Compared current worktree, including untracked `featcopilot/llm/_plan_transport.py` and uncommitted review fixes, against base `bf119812162af87c29bc8d09bb3c705576b94f27`. HEAD `62cbb1cf4f434a6bbfd956b34c2615e761441e46`.
- [x] Read supervisor control flow in `benchmarks/execution.py` (`execute`, preparation, interrupt, publication) and all of `benchmarks/run_state.py`.
- [x] Read worker lifecycle in `benchmarks/tabular.py` `stop_process_tree` / `run_isolated` (lines 1064-1188). Callers also include calibration, blending, and regression audit supervisors.
- [x] Read fit/reset/transform boundaries in `featcopilot/engines/tabular.py`, `fitted_rules.py`, `predictive.py` (`_reset`, `fit`, `transform`, `_learner`), `featcopilot/llm/semantic_engine.py`, `featcopilot/transformers/sklearn_compat.py`, and `featcopilot/classification.py`. Predictive/wrapper/classifier/fitted-rule/semantic paths clear fitted state before mutation; tabular does not.
- [x] Read provider lifecycle in `featcopilot/llm/_plan_transport.py`, `copilot_client.py`, `plan_generator.py`, `openai_client.py`, and `litellm_client.py`.
- [x] Synthetic probe, flaml313 Python, no provider/network/install: failed `TabularEngine` refit left `_is_fitted True` and dropped one-hot columns on the next `transform` (`['cat', 'num']` instead of the fitted one-hot schema).
- [x] Synthetic probe: `run_isolated` after an injected monitor exception left PID 7736 (`python -c time.sleep(60)`) running and `input.pkl` on disk. Process was then killed. Production `except` only handles `KeyboardInterrupt`.
- [x] Synthetic probe: second `archive_results` shrank to `results.json.gz.part1` but left parts 2-26 on disk.
- [x] Synthetic probe: two threads both passed `ExposureLedger.check` before either `admit`; one admit then raised `already exposed`.
- [x] Synthetic probe: `SyncCopilotFeatureClient(isolated_plan=True, timeout=0.2)._run_async` was still blocked after 1.0s. Nested `SyncOpenAIFeatureClient.send_prompt` ran on a different event-loop id than `start`.
- [x] Empty categorical `TabularEngine.fit` raises `ZeroDivisionError`. That division is unchanged from base, so it is pre-existing.
- [ ] Full pytest suite not rerun. Coverage above is source inspection plus the listed probes.
- [ ] Live Copilot/OpenAI/LiteLLM, downloads, package installs, and real-data benchmarks were not run.
- [ ] Generated benchmark archive bodies were not reviewed as inputs. `predictive.py` candidate generators were not line-audited in full.

## Issues

### Issue 1: Isolated worker monitor leaks the process and case payload
- **Severity**: High
- **Classification**: Introduced (`benchmarks/tabular.py` worker supervisor; absent at base)
- **File**: benchmarks/tabular.py
- **Line(s)**: 1064-1081, 1111-1165; compounded by benchmarks/execution.py 967-988
- **Description**: `run_isolated` starts the worker, then monitors it. Cleanup is not in `finally`. `stop_process_tree` and `input_path.unlink` run for `KeyboardInterrupt` and for timeout/memory status, but any other exception in the monitor loop skips both. `json.loads` of a successful worker result is also outside that handler, so a malformed result skips the unlink. `stop_process_tree` then calls `process.wait()` with no timeout, so a worker that does not reap after `kill` also stalls the case cap. `execute` only catches `ResumeMismatchError`, `BenchmarkProtocolError`, and `KeyboardInterrupt` around `run_isolated`, so the same exception skips manifest finalization.
- **Risk**: A psutil/OS error, bad memory-limit comparison, or result-parse error leaves the evaluation process running and retains `input.pkl`. Resume does not see a recorded cell, so it can start a second worker beside the orphan. The wall-clock cap is not a hard bound while `wait()` blocks.
- **Suggested Fix**: After `Popen`, use `try/finally` to stop the process if it is still running and to unlink `input.pkl` on every exit. Give `process.wait` a short timeout and kill any remaining descendants again. In `execute`, catch cell exceptions, record an error row, and still publish `run_status`.
- **Repro**: Patch `subprocess.Popen` to launch `python -c "import time; time.sleep(60)"` and `psutil.Process` to raise `RuntimeError` on the first monitor call. `run_isolated` raised `RuntimeError`; PID 7736 was still alive and `input.pkl` existed. The sleeper was killed after the probe.

### Issue 2: Failed tabular refit leaves a fitted engine with wiped encodings
- **Severity**: High
- **Classification**: Introduced on the changed fit path. Base already omitted an up-front fitted-state reset; new `_encode_target` now raises after that path clears encodings.
- **File**: featcopilot/engines/tabular.py
- **Line(s)**: 132-171, 175-199, 254
- **Description**: `fit` overwrites `_numeric_columns` and, when categorical encoding is on, clears one-hot/target maps and the label encoder before validation can fail. `_is_fitted` is set only at the end and is not cleared at the start. `_encode_target` raises `ValueError` when `y` length differs from `X` (line 254) after those maps are already empty. A caller that catches the refit error still sees `is_fitted` and `transform` uses the torn state.
- **Risk**: A failed refit silently changes the feature schema. The probe fit produced one-hot columns; after the caught length error, `transform` returned only `['cat', 'num']`. Downstream models trained on the original fitted schema then see missing columns, or a later successful-looking transform drops target encodings.
- **Suggested Fix**: Set `_is_fitted = False` and build the next categorical/numeric plan locally. Publish maps, column lists, and `_is_fitted` only after fit succeeds, as `PredictiveTabularEngine._reset`, `FittedRuleEngine.fit`, and `NestedBaseClassifier.fit` already do.
- **Repro**: `TabularEngine(encode_categorical=True, min_samples_per_category=1, onehot_ratio_threshold=1.0)` fit on 4 rows with a categorical column and `y` of length 4, then `fit` again with `y` of length 2. Result: `ValueError` length mismatch, `_is_fitted is True`, `_onehot_columns == []`, and `transform` dropped the one-hot features.

### Issue 3: Isolated plan startup ignores timeout and skips shutdown on hang
- **Severity**: High
- **Classification**: Introduced with isolated plan transport
- **File**: featcopilot/llm/copilot_client.py
- **Line(s)**: 177-182, 550-560
- **File 2**: featcopilot/llm/_plan_transport.py
- **Line(s)**: 89-136
- **Description**: Plan mode starts a fresh SDK client on every prompt. `session()` awaits `client.start()`, `get_auth_status()`, and `create_session()` with no `wait_for`. `config.timeout` is applied only to `send_and_wait` after startup succeeds. `client.stop()` is in `finally` after `start()`, so a hung start never disconnects or stops. `SyncCopilotFeatureClient._run_async` for `isolated_plan=True` uses `asyncio.run` or `executor.submit(...).result()` with no timeout and no cancellation of that task. The legacy nested path at least passes `timeout=120`; the new branch does not.
- **Risk**: Semantic plan fitting can block forever despite `timeout=0.2` or `60`. The Copilot subprocess, event-loop thread, and `TemporaryDirectory` stay alive, and the "always disconnect and stop" contract does not run. A benchmark case timeout may eventually kill the parent, but library callers have no bound.
- **Suggested Fix**: Wrap start, auth, session creation, and send in one `asyncio.wait_for(..., timeout=self.config.timeout)`. On timeout or cancellation, run `session.disconnect()` and `client.stop()` in `finally`, including when `start()` is cancelled. Pass that same timeout to `future.result` and cancel the task before executor shutdown.
- **Repro**: `SyncCopilotFeatureClient(isolated_plan=True, timeout=0.2)._run_async` of a coroutine that waits on `asyncio.Event()` was still running after 1.0 second (`box` empty, thread alive).

### Issue 4: Result archive rewrite is not atomic or idempotent
- **Severity**: Medium
- **Classification**: Introduced (`benchmarks/run_state.py`)
- **File**: benchmarks/run_state.py
- **Line(s)**: 33-48
- **Description**: `archive_results` writes gzip parts with `Path.write_bytes` at stable names, then updates `archive.json`. Part replacement truncates the published part in place. Parts that are no longer listed are not deleted. `archive.json` is the only authoritative name list, and it is written after the part bytes change.
- **Risk**: A crash or cancellation during re-publication leaves `archive.json` pointing at truncated parts, so the concatenated archive no longer matches `uncompressed_sha256`. A later smaller archive leaves stale `results.json.gz.partN` files beside the new index. Globbing or committing every part file reconstructs a different payload than the index.
- **Suggested Fix**: Write parts and the index to temporary names, fsync, replace the index last, and delete part files not listed in the new index. Keep the previous part set intact until the new index is in place.
- **Repro**: `archive_results` on 2000 random bytes with `part_bytes=80`, then again on 20 random bytes. Index listed only `results.json.gz.part1`; the directory still contained parts 2-26.

### Issue 5: Confirmation exposure check and admit are not one critical section
- **Severity**: Medium
- **Classification**: Introduced. `prepare_only` skipping admit is an existing documented non-exposure path and is not this finding.
- **File**: benchmarks/execution.py
- **Line(s)**: 892-909
- **File 2**: benchmarks/run_state.py
- **Line(s)**: 86-105
- **Description**: `execute` calls `ExposureLedger.check`, then loads and validates the dataset, and only then calls `admit`. `check` and `admit` use separate connections. `admit` is transactional, but nothing reserves the family before `_prepare_dataset` opens the payload. Two confirmation runs can both observe no prior exposure, both open the data, and only the later `admit` fails.
- **Risk**: The ledger is the cross-run guard against opening a confirmation family twice. Concurrent runs defeat it for the whole preparation window. The loser records `dataset_error` only after the payload has already been read. `BEGIN IMMEDIATE` does not close that window.
- **Suggested Fix**: Reserve the confirmation family in the same immediate transaction that performs the exposure check, and do that before opening the payload. Roll the reservation back only for failures that did not read protected data.
- **Repro**: Two threads called `check('family', run_id, 'confirmation')`, waited on a barrier, then `admit`. Both checks passed. One admit succeeded and the other raised `Confirmation family 'family' was already exposed in another run`.

### Issue 6: Pooled OpenAI and LiteLLM clients are used on another event loop
- **Severity**: Medium
- **Classification**: Introduced by the rewritten sync wrappers. Non-nested calls correctly share one loop; the nested branch does not.
- **File**: featcopilot/llm/openai_client.py
- **Line(s)**: 533-552
- **File 2**: featcopilot/llm/litellm_client.py
- **Line(s)**: 495-514
- **Description**: `_get_or_create_loop` keeps the async SDK client on one loop, but a call made while an event loop is already running submits `asyncio.run` on a worker thread. That creates and closes a different loop. `start()` from a normal thread binds `AsyncOpenAI` / LiteLLM to the persistent loop; a later nested `send_prompt` does not use that loop. `nest_asyncio` is absent in the probed environment, so this is the active fallback.
- **Risk**: Jupyter or any caller already inside an event loop can fail the next request with a closed-loop or cross-loop client error, or leave the pooled HTTP client unclosed. `stop()` closes only `self._loop`, not the temporary `asyncio.run` loop that issued the request.
- **Suggested Fix**: Submit nested work with `asyncio.run_coroutine_threadsafe` on the persistent loop, or create and close the SDK client inside the same loop that sends. Do not mix a long-lived async client with `asyncio.run`.
- **Repro**: `SyncOpenAIFeatureClient.start` and a non-nested `send_prompt` shared loop id `1875133595728`. `asyncio.run` of another `send_prompt` used loop id `1875133596688`. LiteLLM's `_run_async` has the same `executor.submit(asyncio.run, coro)` branch.

### Issue 7: Empty categorical tabular fit raises ZeroDivisionError
- **Severity**: Low
- **Classification**: Pre-existing. `n_unique / n_rows` is unchanged from base `bf119812`; this is not a new regression.
- **File**: featcopilot/engines/tabular.py
- **Line(s)**: 191-199
- **Description**: With categorical columns and zero rows, `_fit_categorical_encoding` computes `n_unique / n_rows` after the empty-category early return is skipped. An empty object column still enters the loop.
- **Risk**: An empty training frame fails with `ZeroDivisionError` instead of a validation error. Callers that handle `ValueError` do not catch it, and no fitted-state cleanup runs because the exception is raised mid-mutation.
- **Suggested Fix**: Reject `len(X) == 0` before cardinality ratios, with the same explicit error used by the other engines.
- **Repro**: `TabularEngine().fit(pd.DataFrame({'cat': pd.Series(dtype='object')}))` raises `ZeroDivisionError: division by zero`.

## Driver resolution update

- **Issue 2 - Fixed, pending independent retry**: `TabularEngine` resets fitted state, feature metadata and learned categorical/numeric state before validating a new fit. New mismatched-target and invalid-input refit tests failed before the repair and passed afterward, requiring unfitted state, cleared names/features, rejected transform and successful recovery. The combined engine/wrapper/predictive suite passed 291 tests with two optional skips. Successful fitting behavior is unchanged.
- **Issue 7 - Deferred, pre-existing**: The legacy zero-row cardinality division predates this PR. No unrelated exception-type policy change is made; the new reset does ensure such a failed fit cannot retain prior fitted state. The Low concern remains visible rather than being labeled fixed.

## Driver resolution update: Issues 1, 4 and 5

- **Issue 1 - Fixed, pending retry**: Owned worker cleanup and input removal
  are unconditional; reaping waits are bounded and inaccessible descendants
  do not prevent attempting cleanup of the owned parent. Operational errors
  remain explicit retryable failures, distinct from malformed result contracts.
- **Issue 4 - Fixed, pending retry**: Archive parts now use immutable
  generation-specific names, with an atomic authoritative-index switch only
  after all new parts are complete. Previous generation files are intentionally
  retained rather than deleted under concurrent readers; index-driven
  reconstruction remains the contract, not globbing arbitrary parts.
- **Issue 5 - Fixed, pending retry**: Actual confirmation admission is
  transactional before acquisition. Competing admission losers cannot open
  protected data. Reservations are conservatively retained on failures; the
  documented prepare-only non-exposure policy is unchanged.
- **How verified**: The implementation passed 456 supervisor/execution/state
  tests, including interruption/partial publication, concurrent admission and
  cleanup failure cases. Black, Ruff and whitespace checks passed. No historical
  archives or providers were touched.

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

### Issue 4
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 5
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 6
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Issue 7
- **Status**: Open
- **What changed**: pending
- **Why**: pending
- **How verified**: pending

### Driver resolution update: Issues 3 and 6

- **Issue 3 - Fixed, pending retry**: One shared plan-request deadline now
  covers startup, authentication, session creation and sending. Separate
  bounded cleanup grace attempts shutdown and preserves the original
  failure. The synchronous bridge accounts for the request deadline and
  cleanup; cancellation remains cooperative, not an OS-level guarantee
  against blocking or cancellation-resistant SDK code.
- **Issue 6 - Fixed as a coupled completion of the PR's loop-ownership
  contract**: OpenAI and LiteLLM synchronous clients now use one owned
  worker/event loop for construction, ordinary/nested requests and cleanup.
  Independent base inspection found `executor.submit(asyncio.run, coro)`
  already present, so the broader cross-loop problem is not accurately
  classified as wholly introduced. The PR's persistent-client changes
  nevertheless require consistent ownership on both paths.
- **How verified**: The implementation's 471 targeted tests passed with
  both SDK 0.3.0 and 1.0.16 using local fakes, including timeout/cleanup and
  loop-bound client assertions. Ruff, targeted mypy and changed-scope
  Black passed. No live provider or real-data benchmark calls occurred.

The driver also requested a deterministic first-call contention check.
It reproduced two lazily created executors per client. A per-client lock
now protects executor creation and submission, with waiting outside the
lock. The regression verifies one executor, loop and worker under gated
contention; 302 lifecycle/provider tests passed afterward. Concurrent
`stop()` while other calls are in flight remains explicitly unsupported.

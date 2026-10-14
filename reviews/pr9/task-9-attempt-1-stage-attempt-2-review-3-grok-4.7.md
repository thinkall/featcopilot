## Review Summary
- **Round**: 3
- **Theme**: Edge cases & robustness
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-3-grok-4.7.md
- **Issues Found**: 3
- **Verdict**: ISSUES_FOUND

Independent review of worktree changes from `bf119812162af87c29bc8d09bb3c705576b94f27` through current sources, including uncommitted fixes and `featcopilot/llm/_plan_transport.py`. Parent test/mypy/wheel success was not treated as proof.

## Evidence Checklist
- [x] Read the stage-3 prompt contract and `review-pr9/scope.json`; inspected current sources rather than the trimmed map. Compared admission, archive, and plan-transport behavior with `git diff bf119812162af87c29bc8d09bb3c705576b94f27` and the uncommitted worktree diff.
- [x] Confirmation admission: `benchmarks/execution.py` admits before `_prepare_dataset` (lines 890-892). `benchmarks/calibration_evaluation.py` lines 1331-1335 then 1371, and `benchmarks/blending_evaluation.py` lines 1817-1832 then 1906-1908, only `check()` before acquisition. Protocol requirement is `benchmarks/TABULAR_PROTOCOL.md` lines 126-130. Synthetic check-then-open race: both runs passed `ExposureLedger.check` and opened before one `admit` rejected the other (`reviews/pr9` probe removed after the check).
- [x] Plan deadline/cleanup: `featcopilot/llm/_plan_transport.py` lines 99 and 155-174 preserve `primary_error` inside the `try`/`finally`, but `TemporaryDirectory` cleanup sits outside that guard. Documented contract: `docs/user-guide/llm-features.md` line 139. Windows probe: a held file in the temporary directory replaced `TimeoutError("original request failure")` with `PermissionError` `[WinError 32]`.
- [x] Source archive ordering: `benchmarks/execution.py` lines 699-705 create `RunState` (`benchmarks/run_state.py` line 186, `exist_ok=False`) before `archive_source` (`benchmarks/run_state.py` lines 102-110) truncates `source.zip` in place. The first manifest write is line 733, after that call returns.
- [x] Worker cleanup, result protocol, fit invalidation, and provider loop ownership were inspected in current sources (`benchmarks/tabular.py` `stop_process_tree`/`run_isolated`, `benchmarks/execution.py` `_validated_worker_row`, engine `fit`/`_reset` paths, OpenAI/LiteLLM sync wrappers). Findings below are limited to defects those contracts do not already accept.

## Limitations
- OS-denied `process.kill` / `psutil` failures and a shared five-second reap budget are documented best-effort cleanup (`benchmarks/TABULAR_PROTOCOL.md` lines 118-119; `stop_process_tree`). They are not reported as a missing hard kill. Cached descendant PIDs can theoretically be reused after exit; that is a residual of best-effort PID cleanup, not a new guarantee.
- Blocking or cancellation-resistant Copilot SDK calls cannot be given a hard process deadline by this in-process adapter. The nested-loop bridge can return after its watchdog and abandon the worker; the normal `asyncio.run` path cannot interrupt a call that never yields. That matches the documented cooperative limit and is not reported as a defect.
- `SyncOpenAIFeatureClient` / `SyncLiteLLMFeatureClient` document that `stop` must not overlap calls. No concurrent-stop contract was inferred. Concurrent sends are serialized on the owned worker.
- Fit invalidation was checked for `TabularEngine`, `PredictiveTabularEngine`, `FittedRuleEngine`, `PointInTimeRelationalEngine`, `AutoFeatureEngineer`, and the classification estimators. Current `fit` paths clear fitted flags before work and publish fitted attributes only after success. No introduced stale-transform defect was found.
- Canonical confirmation (not prepare-only) now admits before acquisition and keeps that reservation if preparation fails. Prepare-only non-exposure is explicit and was not treated as a defect. A failed **development** acquisition still opens data before `admit` (`benchmarks/execution.py` lines 890-909). That does not by itself violate the confirmation-run sentence, but a later confirmation of that same `family_id` would not see the failed open. Residual exposure gap, not counted above.
- Prepared dataset pickles in the canonical controller are still written with `Path.write_bytes` rather than the atomic JSON helper. A crash can leave a partial pickle; resume then fails closed on hash/load rather than publishing scores.

## Issues

### Issue 1: Qualified confirmation runners open protected data before transactional admission
- **Severity**: High
- **File**: benchmarks/calibration_evaluation.py; benchmarks/blending_evaluation.py
- **Line(s)**: calibration_evaluation.py 1331-1335, 1371, 1561; blending_evaluation.py 1502, 1817-1832, 1906-1908
- **Description**: `benchmarks/TABULAR_PROTOCOL.md` lines 126-130 require an actual confirmation run to `admit` a family inside the exposure transaction before acquisition, and to keep that reservation if acquisition fails. The canonical controller does this (`benchmarks/execution.py` lines 890-892 call `ledger.admit` before `_prepare_dataset`). The qualified confirmation paths do not. Both call `ledger.check`, then `_prepare_dataset` / `_prepare`, and call `admit` only later if a physical job is dispatched. `check` is not the reservation: a synthetic pair of `ExposureLedger.check` calls both succeeded, both proceeded to the open step, and only the later `admit` rejected the loser. A preparation exception in these runners never reaches `admit`, so the opened family remains unmarked. Their tests currently lock in that weaker order (`test_confirmation_checks_exposure_before_prepare_and_prepare_only_does_not_admit`, `test_confirmation_validates_original_ids_and_full_payload_before_iid_execution`). This is new versus base `bf11981`, which has none of these runners. The pending canonical-controller fix was not applied here.
- **Risk**: Two concurrent calibration or blending confirmation controllers can both pass the check and both download/open the same protected family. A failed or crashed acquisition does not block a later confirmation, so the family can be treated as unexposed after it was opened.
- **Suggested Fix**: On non-prepare-only confirmation, call `ExposureLedger.admit` (not only `check`) immediately before `_prepare_dataset` / `_prepare`, and do not delete that row if preparation fails. Keep prepare-only as a read-only check. Update the confirmation tests that currently assert no `admit` event after a failed acquisition. `execution.py` lines 890-892 are the reference order.

### Issue 2: Temporary workspace cleanup can replace the original plan error or a successful plan
- **Severity**: High
- **File**: featcopilot/llm/_plan_transport.py
- **Line(s)**: 99, 155-174
- **Description**: `request` preserves the original failure inside its `try`/`finally`: cleanup errors are logged, and `cleanup_error` is raised only when `primary_error` is None (lines 166-174). That matches `docs/user-guide/llm-features.md` line 139 ("cleanup failures are reported without replacing the original request error"). The `with TemporaryDirectory(...)` at line 99 is outside that guard and does not pass `ignore_cleanup_errors=True` (available on Python 3.10+). The CLI is started with that directory as `working_directory` / `cwd`. If `stop` times out or returns before the process releases the directory — the documented cooperative-cancellation case — Windows directory removal raises `PermissionError` from `TemporaryDirectory.__exit__`. A local probe of that exit path surfaced `PermissionError` `[WinError 32]` instead of the in-flight `TimeoutError`, and `surfaced is original` was false. The same exit also suppresses a successful `return` if removal fails after generation. Base has no `_plan_transport.py`; the pending refactor moved the evaluation provider onto this helper and added the production `isolated_plan` path, so both callers inherit the hole.
- **Risk**: Authentication, isolation, and deadline failures become filesystem errors and bypass callers that handle `TimeoutError` or `PlanIsolationError`. A plan that was already generated can be discarded because the workspace is still busy. This is not a request for a hard process kill; it is the context manager replacing an error the adapter already decided to keep.
- **Suggested Fix**: Construct the directory with `ignore_cleanup_errors=True`, or delete it in a logged `except OSError` that never replaces `primary_error` or a successful result. Leave the cooperative `stop`/`disconnect` grace as it is.

### Issue 3: Source archive is truncated in place after the run directory is committed
- **Severity**: Medium
- **File**: benchmarks/run_state.py; benchmarks/execution.py
- **Line(s)**: run_state.py 102-110 and 186; execution.py 699-705, 733
- **Description**: Result parts and `archive.json` are published by temp file, fsync, and `os.replace`, and the index switches only after parts exist. `archive_source` does not. It opens `directory / "source.zip"` with `ZipFile(..., "w")`, which truncates the destination, then writes members and hashes the result. `execute` constructs `RunState` first (`mkdir(exist_ok=False)` plus `state.json`) and only then calls `archive_source`. `manifest.json` is not written until line 733, after that call returns. A crash or `KeyboardInterrupt` during the zip write leaves a partial or missing `source.zip`, a durable `state.json`, and no manifest. Resume then fails opening `manifest.json`; a fresh run in the same directory fails because the directory already exists. Base has no `archive_source`. Result-archive atomicity in the same module shows the intended failure behavior and was not applied here.
- **Risk**: One interrupted source snapshot bricks the output directory. The run can neither resume nor be recreated in place, and a partial zip is not distinguishable from a completed archive until a later hash check that this crash window never reaches.
- **Suggested Fix**: Write the zip to a unique temporary name in the same directory, fsync, and `os.replace` it onto `source.zip` only after the archive is complete. On resume, if `manifest.json` is absent and `source.zip` is missing or truncated, either rebuild that snapshot before identity checks or fail with an explicit incomplete-publication error that does not require deleting the directory by hand.

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

### Driver resolution update: Issue 2

- **Status**: Fixed after the stage's second-attempt cap.
- **What changed**: Workspace cleanup now explicitly catches and logs
  `OSError`, preserving a successful result or the original timeout,
  cancellation or isolation failure. Directory-creation errors still
  propagate; SDK disconnect/stop grace is unchanged.
- **Why**: A busy temporary directory must not replace the request's
  authoritative outcome. Cleanup remains observable rather than silently ignored.
- **How verified**: Deterministic failing-cleanup fakes reproduced the masking
  defect and covered success, timeout, cancellation, isolation and creation
  errors. All 180 targeted tests passed with SDK 0.3.0 and 1.0.16; Ruff,
  Black, mypy and whitespace checks passed.
- **Limit**: A busy workspace can remain after a logged cleanup warning;
  no unrelated files or directories are removed.

### Driver resolution update: Issues 1 and 3

- **Status**: Fixed after the stage's second-attempt cap.
- **What changed**: Both qualified confirmation controllers now reserve
  families transactionally before raw or cached payload reads, retaining
  reservations on failure and keeping prepare-only check-only. Source ZIPs
  publish atomically, preserving existing snapshots if publication fails.
  All three controllers persist guarded incomplete-bootstrap identity,
  configuration and time so eligible initialization failures can resume
  without deleting their output directories.
- **Why**: Shared safety contracts must apply to every controller, not only
  the canonical runner. Recovery must not silently rebuild populated evidence.
- **How verified**: The implementation's 787 relevant tests passed; guarded
  recovery rejects identity drift and prepared/attempted/result evidence.
  The driver's focused initialization/admission/workspace/compiler checks
  passed 55 tests, package mypy passed 52 files and the wheel built.
- **Limits**: Recovery requires a valid bootstrap record. No historical
  snapshots or scored evidence were rewritten.

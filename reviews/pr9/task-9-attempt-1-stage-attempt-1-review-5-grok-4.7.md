## Review Summary
- **Round**: 5
- **Theme**: Testing & coverage
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-5-grok-4.7.md
- **Issues Found**: 2
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Scope map `review-pr9/scope.json` (721 paths; tests and non-frozen Python inspected) plus worktree vs HEAD `62cbb1cf` pending diffs, including untracked `featcopilot/llm/_plan_transport.py`
- [x] Controller/worker contract read at `benchmarks/execution.py` (`_validated_worker_row`, preparation skip, reusable bypass) and `benchmarks/tabular.py` (`failure_rows`, `run_isolated`); a real `baseline` `run_case` row passed `execution._validated_worker_row` and `json.dumps(..., allow_nan=False)`
- [x] Synthetic resume probe: completed 2-cell run, tampered `state.json` dataset `sha256` to `tampered-dataset`, set `elapsed_seconds` to the 30s budget, resumed. Exhausted resume exited 1 with `run_status=budget_overrun`, `protocol_error=None`, tamper retained, `results.json` unchanged. Same tamper with remaining budget logged `Cannot resume: prepared dataset identity changed` and exited 2
- [x] Windows separator probe: `archive_source` stored `pkg/mod.py`; `str(path.relative_to(root))` was `pkg\mod.py`
- [x] Targeted pytest, not the full suite: source-initialization tests 9 passed; pooled sync-client tests 12 passed; `test_resume_without_new_evidence_preserves_prepared_provenance_and_archives` 6 passed
- [x] SDK 1.0.21 `create_session` / `RuntimeConnection.for_stdio` / `PermissionDecisionReject.kind == "reject"` inspected statically; plan-mode tests call the real `IsolatedCopilotTransport` helper rather than only a success mock
- [ ] Full `pytest tests/` and coverage regeneration not run; parent final gate owns that. Prior 93%/green CI was not treated as proof (`.github/workflows/tests.yml` covers only `featcopilot` on `ubuntu-latest`)
- [ ] Both Copilot SDK contracts were not executed here. The local `copilot` import is 0.1.0, not `github-copilot-sdk`; no SDK install or provider call was made

## Issues

### Issue 1: Exhausted resume skips dataset identity checks and is mislabeled as a budget failure
- **Severity**: Medium
- **File**: benchmarks/execution.py
- **Line(s)**: 880-885, 928-936
- **Description**: If the active-time deadline is already exhausted, preparation `continue`s before `validate_dataset` and ledger admission. `plan_jobs` still rebuilds cell hashes from checkpoint metadata and prior split hashes, but the reusable map forces `True` whenever that preparation status is `budget_exhausted` and the cell already exists, without calling `RunState.reusable`. A funded resume of the same tampered checkpoint reaches `validate_dataset` and exits 2 with `Cannot resume: prepared dataset identity changed for 'test-data'`. An exhausted resume does not. It returns 1, persists the tampered dataset identity, leaves prior success rows in place, and sets `run_status` to `budget_overrun` with `protocol_error` null. `test_resume_without_new_evidence_preserves_prepared_provenance_and_archives` only resumes unchanged provenance, so it still passes. `test_rejected_resume_leaves_durable_checkpoint_untouched` only covers the funded path.
- **Risk**: An identity mismatch is reported as a time problem. Increasing the budget later is what finally fails closed. Until then the checkpoint and prior scores remain associated with a dataset identity that no longer matches the recorded cells. Confirmation and development resumes are both affected because the skip happens before role-specific admission.
- **Suggested Fix**: Even when preparation is skipped for budget, recompute the cell hash from the stored dataset identity and split hash and call `state.reusable` (or an equivalent fingerprint check) before treating any existing cell as reusable. On mismatch, raise `ResumeMismatchError` before `save`, manifest rewrite, or dispatch. Add a regression that completes a mocked-worker run, tampers `state["datasets"][name]` or a stored split hash, sets elapsed to the budget, resumes, and expects exit 2, an identity error, and byte-stable `results.json` / record files. Keep the existing unchanged exhausted-resume test; it does not cover this failure.

### Issue 2: Source allowlist tests never lock POSIX zip names, and CI cannot see the Windows mismatch
- **Severity**: Low
- **File**: benchmarks/run_state.py
- **Line(s)**: 114, 294
- **Description**: `archive_source` writes `path.relative_to(root).as_posix()` into `source.zip`, but `RunState.initialize_source` records `source_allowlist` with `str(path.relative_to(root))`. `benchmarks/calibration_evaluation.py` line 1282 does the same before initialization. On this Windows worktree a nested file was archived as `pkg/mod.py` and stringified as `pkg\mod.py`. No test mentions `source_allowlist`. `.github/workflows/tests.yml` runs only `ubuntu-latest`, so the separator split is invisible to CI. Intentional `importorskip` / Python 3.10 SDK skips are separate and are not this gap.
- **Risk**: Published reproduction manifests can disagree with the archive member names on Windows. A later consumer comparing the allowlist to the zip will fail only outside the current CI matrix.
- **Suggested Fix**: Record allowlist entries with `.as_posix()` in both writers. Add a unit test that archives a nested relative path and asserts every allowlist entry is a forward-slash zip namelist entry. Do not rely on a Windows CI job for that assertion.

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

### Driver resolution update

- **Issue 1 - Fixed, pending retry**: Retained cell fingerprints and split-plan
  consistency are checked before any resume persistence, including exhausted
  and interrupted paths. Metadata/split drift is rejected without rewriting
  checkpoints or score archives; unchanged exhausted resumes and genuine
  unprepared placeholders remain supported.
- **Issue 2 - Fixed, pending retry**: Both source-allowlist writers use
  `.as_posix()`. Nested-path tests simulate Windows separator behavior on any
  host and require allowlist entries to match the ZIP member names.
- **How verified**: Forty targeted regressions passed, with Black, Ruff and
  whitespace checks passing. The driver's broader execution/state/calibration
  suite and package build are running before the independent second attempt.
- **Evidence preservation**: Historical benchmark artifacts were not edited.

The broader prerequisite gate completed: 611 execution/state/calibration tests
passed, package mypy passed 52 files and the wheel built.

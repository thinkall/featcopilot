## Review Summary
- **Round**: 1
- **Theme**: Broad sweep
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-2-review-1-grok-4.7.md
- **Issues Found**: 3
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read the stage-attempt-2 prompt header and evidence contract; did not read prior review artifacts or driver verdicts.
- [x] Compared the worktree to HEAD `46c7db9` (`git diff HEAD`) and the PR file set to base `bf119812` (`git diff --name-status`). Inspected `benchmarks/execution.py` `execute`/`plan_jobs`, `benchmarks/run_state.py` `reusable`, and finalization in `benchmarks/tabular.py` `write_report`/`archive_results`.
- [x] Inspected the public predictive wrapper short-circuit in `featcopilot/transformers/sklearn_compat.py` `_transform_fitted_engines` (lines 654-661) against `PredictiveTabularEngine.transform` (fitted column order, original values preserved, generated values through `_finite_clip`) and `tests/test_predictive_integration.py` reorder/nonfinite tests. No new output-contract defect found.
- [x] Synthetic provider-free repros with the session `benchmark-env` interpreter: exhausted-budget resume of a successful run; interrupt before re-acquisition; interrupt after a success checkpoint; rejected dataset-identity resume. Scratch outputs were deleted.
- [x] Sampled sibling resume/finalization in `benchmarks/calibration_evaluation.py` and `benchmarks/blending_evaluation.py`, plus LLM plan rejection of fabricated fallback in `featcopilot/llm/plan_generator.py`. Not a line-by-line review of all 720 PR paths.
- [ ] Did not re-run the reported 356-test/mypy/wheel suite; those passes are not treated as proof of correctness. Did not read frozen archive bodies or call providers.

## Issues

### Issue 1: Retained prepared runs are republished as unprepared
- **Severity**: High
- **File**: benchmarks/execution.py
- **Line(s)**: 705-708, 770-772, 808-809, 850-852, 866-876
- **Description**: If a dataset was already prepared and checkpointed, a later resume that does not re-acquire inputs still rebuilds the published manifest from an empty `data_path`. `plan_jobs` then stamps `preparation_state="unprepared"` and, on budget exhaustion before preparation, also sets `error` to "Experiment budget exhausted before preparation". Cell rows and `state.datasets` keep the real prepared identity and split hashes, but `manifest.json`, `results.json`, and `archive.json` are rewritten. A fully successful budget-consumed run resumed with no remaining time returns 1 even though every cell is still `success`. An interrupt during re-preparation has the same false `preparation_state` while preserving success rows. This is new in the uncommitted finalization path: committed HEAD raised on changed fingerprints before replacing that manifest.
- **Risk**: Authoritative benchmark evidence contradicts itself. A confirmation or development archive can claim the inputs were unprepared and failed while still publishing successful scores. `REPORT.md` can say there are no failures. Re-running `--resume` after the budget is consumed replaces the previous archive checksum.
- **Suggested Fix**: When retaining a previously prepared dataset without acquiring new inputs, publish the stored dataset provenance and prior split hashes unchanged. Do not add `preparation_state="unprepared"` or a preparation `error` to that dataset. Record the no-budget or interrupted resume as run-level status only. Skip `write_report`/`archive_results` when cell evidence did not change. Keep the cell-row skip at lines 808-809.

### Issue 2: Interrupt handler overwrites a completed success checkpoint
- **Severity**: High
- **File**: benchmarks/execution.py
- **Line(s)**: 833-845, 850-858
- **Description**: `active_cell` stays set until after `state.record` returns. The `KeyboardInterrupt` handler records an interrupted row whenever `cell_id == active_cell`, even if that cell was already checkpointed as `success`. A synthetic interrupt raised after `RunState.record` persisted a success row (`loss=0.25`) left the success JSON orphaned and moved the authoritative `records` pointer, `results.json`, and report to `status="interrupted"` with the loss removed. The new handler introduced this downgrade; it is not the pre-fix behavior.
- **Risk**: Ctrl+C during checkpoint I/O, or any `KeyboardInterrupt` after a successful record and before `active_cell = None`, destroys a completed measurement. That violates immutable evidence and the requirement to retain successes. Resume cannot recover the score from the published checkpoint.
- **Suggested Fix**: Never replace a stored `success` row with `interrupted`. Clear `active_cell` only after a durable record, and in the handler synthesize an interrupted row only when the cell is absent or its stored status is still unfinished. Leave an already recorded success untouched, including its record pointer.

### Issue 3: A rejected resume still consumes checkpoint elapsed time
- **Severity**: Low
- **File**: benchmarks/execution.py
- **Line(s)**: 859-861
- **Description**: `finally` always assigns `state.elapsed_seconds` and calls `state.save()`, including when preparation raises `ResumeMismatchError`. A synthetic resume whose prepared sha256 changed returned 2 and left `manifest.json` bytes unchanged, but advanced `state.json` `elapsed_seconds` from 0.0 to 7.0. Committed HEAD did not save elapsed on that rejection path.
- **Risk**: An explicitly rejected resume burns active budget. A later valid resume of the same output can hit budget exhaustion earlier and then take the false-unprepared publication path in Issue 1.
- **Suggested Fix**: On identity rejection, do not update elapsed time or rewrite `state.json`. Persist elapsed only after an accepted preparation or an explicit unfinished cell record.

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

### Driver resolution update: Issues 1-3

- **Status**: Fixed after the stage's two-attempt cap; the stage was not relabeled clean.
- **What changed**: Retained prepared datasets now keep stored provenance and split identities when a resume cannot prepare again. Score reports and archives are not republished when cell evidence is unchanged. Interruption handling never downgrades a durably checkpointed success. Rejected identity resumes leave checkpoint elapsed time and durable state unchanged.
- **Why**: Graceful cancellation or a rejected resume must neither contradict prior successful evidence nor replace or consume it.
- **How verified**: Added synthetic no-budget/interrupted prepared-resume tests, a post-success-checkpoint interruption test, archive-byte preservation tests and rejected-resume state-byte assertions. The complete execution/state suite passed 221 tests, with Black, Ruff and whitespace checks passing. No real data acquisition or model measurement was used.

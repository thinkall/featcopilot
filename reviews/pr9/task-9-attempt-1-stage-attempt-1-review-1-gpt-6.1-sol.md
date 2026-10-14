## Review Summary
- **Round**: 1
- **Theme**: Broad sweep
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol, reasoning=max
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-1-gpt-6.1-sol.md
- **Issues Found**: 2
- **Verdict**: ISSUES_FOUND
- **Reviewed range**: `bf119812162af87c29bc8d09bb3c705576b94f27` → `46c7db92165cb4078699c3f4cc00ae9855f3c002`, including the full-PR additions rather than only the last commit.

## Evidence Checklist
- [x] Read the generated stage prompt's first 112 lines, the supplied `scope.json` inventory, and repository `AGENTS.md`. Worked independently without subagents or other reviewers' feedback.
- [x] Read the full predictive implementation, including candidate planning, nested residual cross-fitting, joint importance, five-bag ranking, final refitting, and inference. Also read `classification.py`, fitted-rule specifications/engine, point-in-time engine, and shared splitting, frequency, numeric-clipping, and classification-preprocessing implementations.
- [x] Compared legacy tabular, core execution, task routing, selector, sklearn-wrapper, SDK-client, utility, CLI, and packaging changes directly against the PR base. Checked selected associated tests, including predictive integration/stability, classifier ownership/serialization, preprocessing, point-in-time, LLM-plan, and benchmark-execution contracts.
- [x] Read benchmark producers and execution logic: `tabular.py`, `execution.py`, `catalog.py`, `run_state.py`, `reference_methods.py`, `prediction_ensemble.py`, `oof_blending.py`, `plan_validation.py`, and `regression_averaging.py`. Inspected calibration/blending dispatch, preparation/resume, output normalization, and confirmation-decision logic, plus relational preparation/supervision and training-audit code.
- [x] Ran five bounded, in-memory sanity checks successfully: fitted-rule own-fold target exclusion and JSON restoration; group-disjoint and purged temporal splits; strict PIT cutoffs/frozen history/pickle; target-encoder and raw/FE classifier clone/pickle/probabilities; and five-bag predictive frequency selection/batch invariance.
- [x] Executed provider-free reproductions of both findings. The preparation-interruption reproduction mocked filesystem operations, dataset preparation, state, source archiving, and the clock; it performed no acquisition or benchmark measurement.
- [x] Verified introduction: the base has no predictive strategy, while the current wrapper newly bypasses legacy selection for that strategy; `benchmarks\execution.py` is entirely added in this PR. Independently confirmed that Copilot's unrestricted permission handler already existed at the base and did not report it as an introduced finding.
- [ ] This is a broad, prioritized review, not exhaustive line-by-line coverage of every substantive file or every test. Large frozen result/archive bodies, historical review evidence, and current-run review artifacts were not used as review input.
- [ ] No full CI rerun, real-data benchmark, download, package installation, or provider invocation was performed. Local synthetic probes used NumPy 2.4.4, pandas 3.0.2, and sklearn 1.8.0, not the optional pinned benchmark interpreter. No SOTA or general no-harm conclusion is inferred.

## Issues

### Issue 1: Preserve the predictive engine's fitted output schema in the public wrapper
- **Severity**: Medium
- **File**: `featcopilot\transformers\sklearn_compat.py`
- **Line(s)**: 654–681, especially 658–665 and 672–675; related selection bypass at 779
- **Description**: The new predictive path computes a correctly ordered `transformed` frame, but then discards its original-column layout: the helper starts from the incoming `X.copy()` and copies only newly generated columns. Because predictive mode also bypasses the legacy selector, nothing subsequently restores the fitted original-column order. `PredictiveTabularEngine.transform()` explicitly accepts reordered inputs and returns fitted order, but `AutoFeatureEngineer(selection_strategy="predictive")` silently returns the reordered input layout instead. The same routing also applies legacy infinity cleanup to originals that the predictive engine deliberately preserves.
- **Reproduction**: Executed on a 40-row synthetic two-column regression problem with `Ridge`, `max_features=2`, `max_candidates=1`, and one stability repeat. Training output was `["a", "b"]`; transforming `X[["b", "a"]]` through the public wrapper returned `["b", "a"]`, whereas its underlying predictive engine returned `["a", "b"]`. A downstream fitted `Ridge` rejected the public output with “Feature names must be in the same order as they were in fit.”

  ```python
  rng = np.random.default_rng(7)
  X = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
  y = 2 * X["a"].to_numpy()
  fe = AutoFeatureEngineer(
      selection_strategy="predictive", task="regression",
      selection_estimator=Ridge(), max_features=2,
      max_candidates=1, stability_repeats=1, leakage_guard="off",
  )
  training = fe.fit_transform(X, y)
  Ridge().fit(training, y).predict(fe.transform(X[["b", "a"]]))
  # ValueError: fitted feature-name order differs.
  ```

  A separate executed probe with original values `[+inf, -inf]` returned `[NaN, NaN]` through the wrapper, but `[+inf, -inf]` through the fitted predictive engine.
- **Risk**: A valid input accepted by the predictive engine breaks sklearn prediction pipelines. Positional downstream consumers can instead silently apply coefficients/splits to the wrong original columns. The public API also fails to preserve the predictive engine's raw-output semantics.
- **Suggested Fix**: Since predictive mode requires exactly the tabular engine, return that engine's authoritative output directly after removing fit-only boundary metadata, or explicitly reconstruct its entire fitted output schema without legacy cleanup. Keep existing merging/cleanup behavior for legacy mode. Add public-wrapper tests for reordered inputs, downstream prediction, and original nonfinite values.
- **Introduction evidence**: The merge existed previously, but this PR introduces predictive routing through it and changes the selector condition to legacy-only. The new engine's schema/value guarantees therefore are not carried through its newly added public integration.

### Issue 2: Preparation interrupts lose cumulative benchmark time and resumable outcomes
- **Severity**: Medium
- **File**: `benchmarks\execution.py`
- **Line(s)**: 661–704; elapsed-time persistence is deferred until 795–808
- **Description**: Dataset acquisition/preparation runs before the dispatch loop, with only `except Exception` around it. `KeyboardInterrupt` escapes both this block and `tabular.main()`. There is no enclosing finalization to charge elapsed preparation time or checkpoint interrupted/unprepared cells. Consequently, the initially saved `state.json` can retain `elapsed_seconds=0`, empty records, and a manifest with no completed preparation metadata after substantial work. Resuming under the unchanged identity grants that preparation time back, contrary to the documented cumulative active-time allowance.
- **Reproduction**: Executed the real `execute()` control flow with an in-memory state and mocked I/O. Started the clock at zero, used a 100-second budget, and replaced `_prepare_dataset` with a function that advanced the clock to 60 seconds and raised `KeyboardInterrupt`. Observed:

  ```text
  KeyboardInterrupt propagated
  durable elapsed_seconds: 0.0
  checkpointed outcomes: 0
  latest published manifest datasets: []
  latest published manifest active_seconds: absent
  ```

  An equivalent regression can use the existing `experiment` fixture in `tests\test_benchmark_execution.py`: inject the interrupt into `_prepare_dataset`, catch it around `tabular.main(arguments)`, and inspect the persisted state. `RunState` initially saves zero elapsed time; the interrupted preparation never reaches either later elapsed-time update.
- **Risk**: Normal cancellation during acquisition, parsing, or split preparation loses operational evidence and bypasses cumulative resource accounting on resume. Repeated interruptions can exceed the declared total allowance without being reflected in persisted active time. Unlike the dedicated calibration/blending controllers, this controller does not retain explicit interrupted preparation outcomes.
- **Suggested Fix**: Add controller-level interruption handling and finalization spanning preparation and dispatch. Persist cumulative elapsed time on interruption, retain requested unfinished cells with resumable preparation markers, and preserve available preparation metadata. Test interrupt → resume with a controlled clock, asserting that consumed preparation time reduces the remaining allowance and that unfinished cells stay represented.
- **Introduction evidence**: `benchmarks\execution.py`, including this preparation/dispatch lifecycle, is new relative to the PR base.

## Resolution Log
_Updated by the driving agent as findings are addressed._

### Issue 1
- **Status**: Open
- **What changed**: Pending; review only, no source or test changes made.
- **Why**: The driver owns fixes and should preserve legacy behavior while repairing predictive integration.
- **How verified**: Reproduced public-wrapper/underlying-engine column-order divergence, downstream sklearn rejection, and original nonfinite-value divergence using small synthetic inputs.

### Issue 2
- **Status**: Open
- **What changed**: Pending; review only, no source or historical-evidence changes made.
- **Why**: The driver owns interruption/resume accounting and coverage fixes.
- **How verified**: Executed I/O-mocked `execute()` with a 60-second preparation interruption and confirmed zero persisted elapsed time, no outcomes, and an unfinished initial manifest.

## Driver: effective review decision

This section records orchestration, not the reviewer's independent verdict.

| Field | Effective decision |
|---|---|
| Tier | High, estimated by the primary agent; user did not prescribe a tier. |
| Change size | Very large: 720 PR files; excluding dated evidence, suites, registry and legacy cache bodies still leaves 127 substantive files across public APIs, controllers, tests and documentation. |
| Cyclomatic complexity | Qualitatively high: nested fitting/selection boundaries, fitted-state transitions, resumable execution, subprocess supervision, optional-provider failure paths and wrapper interactions. No numerical complexity score was computed. |
| Rationale | Breadth and interacting state/failure paths warrant six ordered stages rather than one lightweight pass. Generated archive volume alone did not determine the tier. |
| Primary model | `gpt-6-astra`. |
| Required reviewers | `gpt-6.1-sol` at `max`; `grok-4.7` at `xhigh`, resolved from the live task catalog. |
| Human override | User selected `high-two-model-override`: explicitly waive unavailable Anthropic Opus; retain both available reviewers and all six stages. |
| Repository policy | `AGENTS.md`: style, mandatory full pre-commit before commits, explicit staging and preservation of unrelated work. No repository reviewer-model override was found. |
| Stage gate | Both required reviewers run independently per attempt. Advance clean when both are clean; otherwise fix and validate, then allow at most one second attempt. Any surviving second-attempt findings stay unresolved and visible. |
| Safety and evidence | No live provider calls or real-data benchmark measurements. Preserve checksum-bound historical evidence; review its producers and deterministic validation. Do not include current review artifacts in subsequent reviewer inputs. |
| Scope | PR #9 base `bf119812162af87c29bc8d09bb3c705576b94f27` to starting head `46c7db92165cb4078699c3f4cc00ae9855f3c002`, including subsequent review fixes. |

### Driver stage outcomes

- Stage 1, attempt 1: both reviews completed with findings. GPT's two introduced defects are being fixed. Grok's two provider-policy concerns were independently verified as present at the PR base and remain visible as deferred pre-existing issues; they are not silently counted as fixed.
- Stage 1, attempt 2: both required reviewers returned `ISSUES_FOUND`. GPT reported four Medium findings; Grok reported two High and one Low finding in resume handling. The stage advances **with unresolved findings**, not clean, after the High contract's two-attempt limit. Introduced findings are being repaired before stage 2; no third broad-sweep attempt will be dispatched.
- Stage 2, attempt 1: both required reviewers returned findings. Concrete API, typed-plan transport and label-provenance issues were repaired; architectural factory/cycle redesign concerns remain explicitly deferred with evidence.
- Stage 2, attempt 2: GPT returned one Low complex-source validation issue; Grok returned one Medium controller protocol-error finalization issue. The stage advances **with unresolved findings**, not clean, at its second-attempt cap. The Low issue is fixed and controller handling is being repaired before stage 3.
- Stage 2 follow-up: both introduced second-attempt findings now have tested fixes. Validation before stage 3 passed 465 tests, package mypy (52 files) and a wheel build. Architectural/deferred legacy concerns remain visible; stage 2 is not retroactively called clean.
- Stage 3, attempt 1: both required reviewers returned robustness findings.
  Introduced defects were repaired with resolution notes in both artifacts.
  The combined supervisor/controller/provider suite passed 1,033 tests,
  package mypy passed 52 files and the wheel built before the independent retry.
  The Low pre-existing empty-categorical input error remains deferred.
- Stage 3, attempt 2: both required reviewers returned findings. GPT found
  one Low compiler-validation regression; Grok found qualified-controller
  admission, temporary-workspace cleanup and source-archive initialization
  failures. The stage advances **with unresolved findings**, not clean, at
  its second-attempt cap. Compiler validation is fixed; the remaining
  introduced defects are being repaired before stage 4.
- Stage 3 follow-up: all introduced second-attempt findings have tested fixes.
  The backend implementation passed 787 relevant tests; the driver verified
  55 focused repaired contracts, package mypy (52 files) and the wheel build
  before stage 4. The stage remains historically advanced-with-unresolved.
- Stage 4, attempt 1: GPT returned one Medium and one Low finding; Grok
  returned CLEAN. Object-backed binary target semantics and repeat-count
  preflight were repaired. Validation passed 566 tests, package mypy
  (52 files) and a wheel build before attempt 2.
- Stage 4, attempt 2: both required reviewers returned findings; the stage
  advances **with unresolved findings**, not clean, after its second attempt.
  Categorical predictor/target semantics and optional stage propagation were
  repaired. The retained singleton target-encoding leak was also closed as a
  directly coupled cross-fit contract correction, with its pre-existing origin
  explicitly retained. Validation passed 555 tests (two optional skips), package
  mypy (52 files) and a wheel build.
- Stage 5, attempt 1: GPT returned CLEAN; Grok identified an exhausted-resume
  identity-check gap and a Windows/ZIP allowlist mismatch. Both were fixed
  with deterministic regressions. Prerequisite validation passed 611 tests,
  package mypy (52 files) and a wheel build.
- Stage 5, attempt 2: both required reviewers returned findings. The stage
  advances **with unresolved findings**, not clean, at its second-attempt cap.
  Impossible worker domains, portable source fingerprints and deadline/cleanup
  test gaps were repaired. Validation passed 473 controller tests and 457
  provider tests, package mypy (52 files) and the wheel build before stage 6.
- Stage 6, attempt 1: both required reviewers returned findings. Serialization
  diagnostics, benchmark dependency constraints, stale provider documentation
  and safe plan-failure context were repaired. Before the independent retry,
  the driver passed 453 relevant tests (17 optional skips), package mypy
  (52 files) and a wheel build. Transport-specific validation also passed
  462 provider-free tests on each of SDK 0.3.0 and 1.0.16.
- Stage 6, attempt 2: dispatching both required reviewers against frozen
  current sources. A fresh full-suite coverage run covers the latest repairs;
  the earlier 4,178-pass/93.44% run predates these final-hardening fixes.
- Stage 6, attempt 2 outcome: both reviewers returned `ISSUES_FOUND`. The stage
  advances **with unresolved findings** at the High contract's second-attempt
  cap, completing all six stages without a third attempt. Eight introduced
  performance, diagnostics and documentation findings are being repaired;
  two independently verified pre-existing findings remain deferred. The
  concurrently running suite was stopped at approximately two-thirds complete
  before final repairs to avoid reporting mixed-source coverage. It had no
  observed failures, but is not a completed final gate.

Stage-1 follow-up repair validation completed before stage 2: 417 tests passed,
17 optional tests skipped, package mypy passed 51 files, and the wheel built.
All introduced findings from both attempts have implementation/test resolutions
recorded in their original artifacts. Stage 1 remains historically
**advanced-with-unresolved**, not retroactively clean; two independently
verified pre-existing provider-policy concerns remain deferred.

### Driver resolution update: Issue 1

- **Status**: Fixed, pending the independent second attempt.
- **What changed**: `featcopilot\transformers\sklearn_compat.py` now returns the sole predictive engine's authoritative frame, retaining fitted order and original values instead of applying the legacy merge/cleanup path. Training-only boundary metadata is still removed for the training transform. Legacy merging is unchanged.
- **Why**: Predictive configuration already requires exactly the tabular engine; rebuilding its originals duplicated and violated that engine's output contract.
- **How verified**: Two new public-wrapper regressions failed before the fix and passed afterward, including a downstream Ridge prediction on reordered input and preservation of original infinities/NaNs. The combined predictive-integration, sklearn-wrapper, fitted-plan-wrapper and splitting suite passed 307 tests; package mypy passed 51 files.

### Driver resolution update: Issue 2

- **Status**: Fixed, pending the independent second attempt.
- **What changed**: `benchmarks\execution.py` finalizes cumulative elapsed time across preparation and dispatch interruptions and retains resumable unfinished cells. `benchmarks\run_state.py` distinguishes unprepared placeholders from validated dataset/split identities so continuation can recover without weakening completed-cell checks.
- **Why**: Preparation consumes the same cumulative allowance as model dispatch. A graceful interruption must neither grant that time back nor erase prior successes.
- **How verified**: New controlled-clock tests cover new/resumed interruptions, prepare-only runs, exhausted budgets, continuation, retained successes and exact locked 128/480-cell plans. The implementation's execution/state suite passed 191 tests. The driver's combined execution/state/predictive/wrapper suite then passed 356 tests; package mypy passed 51 files and the wheel built successfully.
- **Limit**: Graceful exceptions/interruption are covered; forced process termination cannot execute Python finalization.

### Driver: surviving scope limits and deferred concerns

All six stages completed two independent attempts per required reviewer:
24 artifacts total. Every stage reached its High two-attempt cap with findings;
none is retroactively labeled clean after driver repairs.

- Legacy code-mode permission auto-approval and malformed-suggestion
  empty-success behavior were independently verified at the base and remain
  deferred. New typed-plan requests use the separately hardened isolated path.
- The broader preprocessing-factory and benchmark dependency-cycle redesigns
  remain architectural follow-ups rather than demonstrated current import
  failures. The existing `gate_n_jobs` role was documented, not replaced with
  another API knob.
- Empty categorical zero-row legacy tabular fitting still has a pre-existing
  error-path limitation. Failed refits now reliably clear fitted state.
- The unsupported `polynomial_degree` examples in the legacy best-practices
  guide and legacy dataset-loader `print()` warnings were independently
  verified at the base and remain separate follow-ups.
- The singleton target-encoding issue was initially classified as pre-existing
  but was subsequently fixed because it directly conflicted with the new
  own-target-exclusion contract. It is not a surviving deferred defect.
- Provider cancellation is cooperative, process-tree cleanup is bounded
  best-effort, and uncooperative providers or OS-denied termination remain
  explicit limits. No live provider invocation or real-data measurement was
  performed during these review repairs.

Final validation and publication status are recorded below.

### Driver final repair completion

All eight introduced findings from the final capped attempt now have targeted
repairs and resolution notes in their original artifacts. Learner/PIT modules
passed 236 tests (17 optional skips); controller/source-discovery repairs passed
512 tests; plan-mode cleanup passed 133 tests on each SDK 0.3.0 and 1.0.16.
The next gate freezes formatted sources for fresh full-suite coverage, package
types, wheel isolation, mandatory pre-commit and explicit-path publication.

### Driver final validation

- Fresh full-suite execution on frozen production sources completed with
  **4,217 passed, 19 skipped and one failed stale test injection**; production
  coverage was **93.42%**, above the required 85%.
- The sole failure patched a superseded combined hash helper instead of the
  separated serializer. The test-only correction now verifies three exception
  types and original error identity. All **18 related diagnostic/fingerprint
  tests passed** afterward. Exact hashes confirm all production, benchmark and
  documentation bytes still match the full-suite snapshot; only that test
  changed. This is full-suite evidence plus a targeted correction, not a claim
  that the initial full-suite invocation exited successfully.
- CI-environment mypy passed **52 source files**. The final wheel built and
  isolated imports loaded exclusively from it without benchmark/test packages;
  the new shared plan transport is included.
- Mandatory `pre-commit run --all-files` passed after explicit staging of
  **79 owned paths**, including all **24 review artifacts**. It is rerun after
  the final test/driver-note update before commit.
- Frozen result/suite/family evidence is absent from the staged repair diff.
  Both legacy cache-descriptor SHA256 values match their immutable expected
  bytes. Unrelated `submissions/` remains untouched and excluded.
- Publication and a fresh remote Python 3.10-3.13 CI matrix follow this local
  gate. No merge, default promotion, new benchmark measurement or SOTA claim
  is part of this review completion.

### Driver published CI follow-up

The first published matrix, run `38614776818` on `08317c7`, exposed two
cross-environment issues not reproduced by the pinned local full-suite run:

- Python 3.10 intentionally has no Copilot SDK. Mypy consequently sees the
  imported SDK message type as `Any`; the isolated callback now gives its
  already SDK-validated text an explicit local `str` annotation. No SDK
  dependency, runtime coercion, fallback or ignored type error was added.
  A missing-SDK type-check configuration reproduced the exact error before
  the fix and passed all 52 package files afterward; installed-SDK typing
  also passed all 52 files.
- Linux BLAS batch-versus-singleton prediction arithmetic differed by at most
  `1.11e-16` in the new missing-only target-encoding test. The test now asserts
  **exact equality of encoded model inputs**, and applies the already existing
  public probability tolerances (`atol=1e-15`, `rtol=1e-14`) after the learner.
  This targets encoding invariance without requiring bit-identical matrix
  multiplication. Production prediction and preprocessing code are unchanged.

The follow-up passed 150 relevant plan/inference tests in the SDK 0.3
environment and 133 plan tests in the SDK 1.0 environment; 16 inference/pickle
cases also passed with CI-style dependencies. The wheel rebuilt successfully.
These are CI repairs within the completed capped review, not a third review
attempt. A new published matrix remains the final remote gate.

## Review Summary
- **Round**: 6
- **Theme**: Polish & hardening
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-6-grok-4.7.md
- **Issues Found**: 3
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Compared `bf119812162af87c29bc8d09bb3c705576b94f27` with the current worktree, including uncommitted controller/SDK fixes and untracked `featcopilot/llm/_plan_transport.py`. Base `featcopilot/llm/copilot_client.py` logged and returned mock responses; current `start()` raises. `docs/getting-started/authentication.md` and `docs/user-guide/llm-features.md` were updated; `docs/user-guide/overview.md` was not.
- [x] Inspected bounded publication, worker cleanup, and archive producers in `benchmarks/execution.py`, `benchmarks/run_state.py`, and `benchmarks/tabular.py` against `benchmarks/TABULAR_PROTOCOL.md`. New archives are index-addressed `concatenated-gzip-v1` parts; historical frozen bodies were not opened.
- [x] Checked package/API constraints in `pyproject.toml`, `docs/getting-started/installation.md`, `.github/workflows/tests.yml`, and the local already-installed `github-copilot-sdk==0.3.0` (`Requires-Python >=3.11`; legacy `SubprocessConfig` has the isolation parameters the plan transport requires). No install, download, or live provider call.
- [x] Synthetic version check: `benchmark-tabular`'s `scikit-learn>=1.6,<1.8` accepts 1.6.0, 1.6.1, 1.7.0, and 1.7.1, while `benchmarks/prediction_ensemble.py` `_retained_bases` accepts only `[1.7.2, 1.10)`.
- [x] Synthetic target-encoding check: with encoding activated (`target_encode_ratio_threshold=1.0`, `min_samples_per_category=1`), one row raises `ValueError: Target encoding requires at least two rows for own-target exclusion`; two rows encode. This matches `docs/user-guide/engines.md`.
- [x] Synthetic timeout text check: `asyncio.wait_for` raises `TimeoutError` with empty `args`. That is the exception isolated plan requests surface, unlike the legacy client message.
- [x] Narrative-only claim check: README 6.82% and 23.79% match the cited ANALYSIS tables; those narratives and the updated public guides disclaim SOTA, no-harm guarantees, and default promotion. `AutoFeatureEngineer.selection_strategy` remains `"legacy"`.
- [ ] Raw frozen result archives, current review artifacts, and driver progress were not read, by assignment.

## Limitations
Reviewed the named polish surfaces and their producers, not all 127 substantive files line by line. Historical numbers were checked through ANALYSIS/REPORT narrative and public docs, not raw result bodies. No full test suite, live Copilot call, package install, or benchmark run. Local SDK inspection covers the installed 0.3.0 contract only.

## Issues

### Issue 1: Published overview still promises mock LLM fallback
- **Severity**: Medium
- **File**: docs/user-guide/overview.md
- **Line(s)**: 140-147
- **Description**: The published Overview, still linked from `mkdocs.yml` line 78, says LLM features fall back to heuristics and that `AutoFeatureEngineer(engines=['llm'])` works without Copilot authentication, with a mock-response warning. That was true at base: `copilot_client.py` logged `Using mock LLM responses` and returned `_mock_response`. This change removes that path. Current `CopilotFeatureClient.start()` raises on a missing or failed SDK, and `docs/getting-started/authentication.md` lines 26-27 and 286-289 now say missing SDKs are errors and mock mode is not entered. Overview was not updated, so the public page contradicts the implementation and the updated authentication guide. Related stale agent text remains in `docs/user-guide/cli.md` line 257, which still lists mock-mode notices as a `warnings` source. This is an incomplete doc update caused by the behavior change, not a pre-existing false claim.
- **Risk**: Readers of the published overview expect unauthenticated LLM engineering to return heuristic features. They now get a hard failure and no features, and the two public guides disagree.
- **Repro**: Follow the Overview snippet `AutoFeatureEngineer(engines=['llm'])` without the SDK extra or an authenticated CLI. `start()` raises instead of logging a mock warning and returning features. Compare with `docs/getting-started/authentication.md` "Offline Feature Engineering".
- **Suggested Fix**: Replace the Graceful Degradation section with the explicit non-LLM fallback already documented in authentication.md. Remove the mock-mode notice from the CLI agent contract, or state that missing providers are stderr errors rather than JSON warnings.

### Issue 2: benchmark-tabular allows sklearn versions the calibration adapter rejects
- **Severity**: Medium
- **File**: pyproject.toml
- **Line(s)**: 78
- **Description**: `benchmark-tabular` pins `scikit-learn>=1.6,<1.8`. `benchmarks/prediction_ensemble.py` lines 266-267 reject every sklearn outside `[1.7.2, 1.10)` and the artifact identity at line 478 is `retained_estimator_calibrators_1.7.2_1.8_1.9`. `benchmarks/README.md` lines 33-34 and 402-407 tell users to install that extra and run `benchmarks.calibration_evaluation`. A resolver outcome the extra explicitly allows (1.6.x or 1.7.0/1.7.1, including an already-installed 1.6 that satisfies `>=1.6`) fails only when a calibration fit reaches `_retained_bases`. The upper bound `<1.8` is a documented AutoFeat constraint and is not itself the defect. Both the extra and the adapter are introduced by this change.
- **Risk**: The documented benchmark install does not guarantee the documented calibration path. The failure is a runtime `RuntimeError` after environment setup, and the artifact adapter name claims a 1.7.2/1.8/1.9 identity the extra cannot uniformly provide.
- **Repro**: In an environment whose installed scikit-learn is 1.6.1 or 1.7.1, `pip install -e ".[benchmark-tabular]"` leaves that version in place. Importing and fitting the retained sigmoid bundle then raises `Unsupported sklearn calibration adapter version ...; expected 1.7.2/1.8/1.9`. Version membership was checked locally without installing or fitting.
- **Suggested Fix**: Raise the extra floor to `scikit-learn>=1.7.2,<1.8` if AutoFeat accepts 1.7.2, and state that floor next to the calibration command. Keep the artifact adapter label aligned with the versions that extra can actually install, or document that 1.8/1.9 adapter support requires an environment outside `benchmark-tabular`.

### Issue 3: Isolated plan SDK failures lose the legacy error text
- **Severity**: Medium
- **File**: featcopilot/llm/copilot_client.py
- **Line(s)**: 98-102, 180-188, 212
- **Description**: Code-mode `start()` logs `Copilot SDK is unavailable. Install featcopilot[llm].` before raising, and code-mode `send_prompt` rewrites timeouts as `Copilot request timed out after {timeout} seconds`. Isolated plan `start()` calls `IsolatedCopilotTransport()` and returns on success without that handler, so a missing SDK is a raw `ImportError` from `import_module("copilot")`. Isolated `send_prompt` delegates to `_plan_transport.request`, whose `asyncio.wait_for` ( `_plan_transport.py` lines 159-161) raises `TimeoutError` with empty args. A local `wait_for` probe produced `TimeoutError.args == ()`. `docs/getting-started/authentication.md` lines 26-27 say missing SDKs and timeouts are reported as errors. The CLI prints one stderr line, `featcopilot: unexpected error: TimeoutError: `, with no duration or phase. Cleanup warnings at `_plan_transport.py` lines 173 and 181-184 record only `type(exc).__name__`, so a cleanup failure that accompanies a primary error drops the SDK message from the log while preserving the primary exception.
- **Risk**: Plan-mode and benchmark Copilot failures, which now share this transport, are not diagnosable from the operator log or the CLI's single stderr line. Timeout and missing-extra failures look empty or like an unrelated import name (`copilot` versus `featcopilot[llm]`).
- **Repro**: Call the isolated plan client with the SDK uninstalled, or expire `request(..., timeout=0.01)` around a slow `generate()`. Observe `ImportError: No module named 'copilot'` or `TimeoutError` with an empty string, not the code-mode messages at lines 126 and 212.
- **Suggested Fix**: Catch `ImportError` on the isolated start path and log/re-raise the same install error used by code mode. Translate plan `TimeoutError` into the duration-qualified `TimeoutError` before it reaches the CLI. Include the cleanup exception message in the warning without replacing `primary_error`.

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

### Driver resolution update: Issues 1-2

- **Issue 1 - Documentation fixed**: The overview now recommends explicit
  non-LLM engines or supplied rules for offline operation instead of promising
  mock generation. The CLI warning contract no longer lists mock-mode notices.
  Legacy optional built-in text degradation is documented separately.
- **Issue 2 - Fixed, pending retry**: The benchmark extra's sklearn floor is
  1.7.2; its existing `<1.8` cap remains aligned with AutoFeat. The README
  distinguishes broader adapter support in separately managed environments
  from the reference-compatible extra. AutoFeat itself is pinned to 2.1.1.
- **How verified**: Dependency metadata tests assert accepted and rejected
  release boundaries; related reference/serialization checks passed 16 tests.
  No benchmark version gates or frozen results were weakened or regenerated.

### Driver resolution update: Issue 3

- **Status**: Fixed, pending the independent second attempt.
- **What changed**: Isolated startup now logs the actionable
  `featcopilot[llm]` installation guidance while preserving the original import
  exception. Empty deadline errors carry request phase and configured duration,
  chaining the original exception; nonempty SDK timeout objects retain identity.
- **Why**: Operator-facing context should identify the failed operation without
  leaking provider messages, prompts or credentials. The suggestion to log raw
  cleanup exception text was intentionally not adopted. Cleanup warnings retain
  safe phase/error-class context and cannot replace the primary error.
- **How verified**: The transport implementation's provider-free suite passed
  462 tests on each of SDK 0.3.0 and 1.0.16, with Ruff, Black, mypy and whitespace
  checks. Regressions cover missing-SDK startup avoidance, duration/phase text,
  chained causes, nonempty timeout identity and cleanup-message privacy.
- **Limit**: Timeout cancellation remains cooperative; this does not guarantee
  termination of arbitrary uncooperative provider code.

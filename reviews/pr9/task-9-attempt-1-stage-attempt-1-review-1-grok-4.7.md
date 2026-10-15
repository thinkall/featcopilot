## Review Summary
- **Round**: 1
- **Theme**: Broad sweep
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-1-grok-4.7.md
- **Issues Found**: 2
- **Verdict**: ISSUES_FOUND

Inspected the full PR range `bf119812162af87c29bc8d09bb3c705576b94f27..46c7db92165cb4078699c3f4cc00ae9855f3c002`, not only the last commit. Legacy constructor defaults stay `selection_strategy="legacy"` and predictive/ensemble/plan behavior is opt-in. No SOTA claim is treated as established. Limitations: no live provider calls, no real-data benchmarks, no archive-byte review, and not every study script was line-audited.

## Evidence Checklist
- [x] Scope map `review-pr9/scope.json` (720 paths; substantive engines, classification, LLM clients, benchmark controllers, and tests) plus `git diff --stat bf119812162af87c29bc8d09bb3c705576b94f27 HEAD`
- [x] Train-only selection and transforms: `featcopilot/engines/predictive.py` fit/holdout/final ranking (`705-1101`), residual folds (`2290-2348`), importance bags (`2140-2238`); `featcopilot/engines/fitted_rules.py` cross-fit (`250-360`); `featcopilot/engines/point_in_time.py` strict cutoff (`247-268`); `featcopilot/classification.py` member fit and ensemble average (`400-980`); `featcopilot/utils/splitting.py` holdout/purge
- [x] Benchmark controllers: `benchmarks/execution.py` preregistration and confirmation splits (`200-750`); `benchmarks/catalog.py` official split alignment and row-cap rejection; `benchmarks/tabular.py` train-only `generate_features` / `run_isolated` RSS poll (`1064-1169`); `benchmarks/llm_evaluation.py` isolated session (`413-442`)
- [x] SDK contrast: product session `featcopilot/llm/copilot_client.py:100-104` uses `PermissionHandler.approve_all` and omits tool/cwd lockdown; installed SDK `copilot/session.py:246` returns `approve-once`; `copilot/client.py` only sends `availableTools` when explicitly set
- [x] Synthetic repro, no network: `CopilotFeatureClient/OpenAIFeatureClient/LiteLLMFeatureClient._parse_suggestions("Sorry, I cannot comply.")` each return `[]` after a warning; `request_fitted_plan` on the same text raises `ValueError`. Predictive native `_compute` matched `Feature.compute` on a 40-row extended/frequency fit (0 mismatches)
- [ ] Frozen benchmark archives and `benchmarks/families.json` raw bytes were not reviewed; producers and admission checks were
- [ ] Live Copilot/OpenAI/LiteLLM sessions were not started, so tool execution was verified from SDK source and call sites only

## Issues

### Issue 1: Public Copilot client auto-approves tools
- **Severity**: High
- **File**: featcopilot/llm/copilot_client.py
- **Line(s)**: 90-104
- **Description**: `CopilotFeatureClient.start` creates a long-lived session with `PermissionHandler.approve_all` and does not pass `available_tools=[]`, `tools=[]`, `enable_config_discovery=False`, empty MCP/skill lists, or an isolated `working_directory`. The installed SDK documents `approve_all` as allowing permissions (`approve-once`) and only includes `availableTools` in the session payload when the caller sets it, so CLI default tools and the process working directory remain in play. Column names, descriptions, and task text are interpolated into prompts and sent on that session. The same PR's benchmark provider does the opposite: it rejects permissions, clears tools, disables config discovery, and uses an empty workspace (`benchmarks/llm_evaluation.py:413-442`).
- **Risk**: A prompt-injected column/description, or a tool-using model turn, can run local Copilot tools with the user's credentials and repository CWD (file read, shell, MCP). That is a credential and data-exposure path on the default `backend="copilot"` feature client, not only on an explicit benchmark opt-in.
- **Suggested Fix**: Match the benchmark lockdown for every library session: deny permission requests, set `available_tools=[]` and `tools=[]`, disable MCP/skills/config discovery, and use a fresh empty working directory. Do not use `approve_all` for non-interactive feature generation.

### Issue 2: Code-mode provider failures are treated as empty success
- **Severity**: Medium
- **File**: featcopilot/llm/copilot_client.py
- **Line(s)**: 384-385
- **Description**: Invalid suggestion JSON is logged and returned as `[]` in all three code-mode clients (`copilot_client.py:384-385`, `openai_client.py:396-397`, `litellm_client.py:358-359`). `SemanticEngine.fit` then stores that empty list and sets `_is_fitted = True` (`semantic_engine.py:355-376`). Text-column suggestion errors are also swallowed, leaving only built-in length/count features (`semantic_engine.py:496-501`). `standardize_categories` returns `{}` on any exception and logs only when verbose (`semantic_engine.py:990-992`). Plan mode does not do this: `request_fitted_plan` raises `ValueError` and documents that provider errors propagate without a fabricated plan. Reproduced without a provider: each `_parse_suggestions("Sorry, I cannot comply.")` returned `[]`; the same text raised from `request_fitted_plan`.
- **Risk**: An outage, refusal, or non-JSON response becomes a successful fit with no LLM features, or with only deterministic text features. Callers cannot distinguish "the model proposed nothing" from "the provider failed" except by reading logs. That violates the retain-failures / no fabricated-fallback contract and can hide a broken backend behind a green fit.
- **Suggested Fix**: Raise a provider-specific error when suggestion JSON is invalid or a required provider call fails. Do not mark the engine fitted on that path. If built-in text features must remain, gate them with an explicit flag and record the provider failure on the result instead of returning success.

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

### Driver triage: independently verified pre-existing concerns

The original findings above are preserved unchanged. Both describe real
legacy behavior, but independent base-commit inspection shows they were not
introduced by this PR:

- **Issue 1 - Deferred, pre-existing**: At base
  `bf119812162af87c29bc8d09bb3c705576b94f27`,
  `featcopilot/llm/copilot_client.py:78-99` already constructed
  `CopilotClient()` without an isolated workspace and called
  `create_session(on_permission_request=PermissionHandler.approve_all,
  model=..., streaming=...)`. The stricter new benchmark provider is a
  separate API; its existence does not establish that the public client's
  unchanged permission policy is an introduced regression. The risk remains
  visible and is not claimed fixed. Changing this legacy permission policy
  is outside the surgical fixes for the current PR review.
- **Issue 2 - Deferred, pre-existing**: The same base's code-mode parsers
  already returned `[]` after invalid JSON:
  `copilot_client.py:334-359`, `openai_client.py:351-373`, and
  `litellm_client.py:336-358`. Its `semantic_engine.py:236-248` swallowed
  general suggestion failures, `279-300` swallowed text-suggestion failures,
  and category standardization already used its legacy fallback path.
  The new plan mode's strict error contract does not retroactively change
  these existing code-mode semantics. A broader legacy error-policy change
  remains deferred rather than being reported as resolved.

Verification used `git show <base>:<path>` and exact surrounding source,
not another reviewer's conclusion. No provider calls were made and no
source change was applied for either deferred concern.

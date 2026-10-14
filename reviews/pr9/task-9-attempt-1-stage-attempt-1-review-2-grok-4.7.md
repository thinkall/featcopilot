## Review Summary
- **Round**: 2
- **Theme**: Architecture & patterns
- **Mode**: resolved-plan
- **Model**: grok-4.7
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-2-grok-4.7.md
- **Issues Found**: 4
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Scope inventory: `review-pr9/scope.json` has 720 paths, 126 non-frozen; reviewed production ownership in `featcopilot/classification.py`, `featcopilot/engines/predictive.py`, `featcopilot/utils/model_preprocessing.py`, `featcopilot/transformers/sklearn_compat.py`, and `featcopilot/llm/{_types,plan_generator,semantic_engine,copilot_client,openai_client}.py`
- [x] Dependency direction from an AST import walk of `featcopilot/` and `benchmarks/`: production does not import `benchmarks`; `benchmarks.execution` and `benchmarks.tabular` import each other; `benchmarks.regression_averaging` imports both at module level
- [x] Base check `git cat-file -e bf119812162af87c29bc8d09bb3c705576b94f27:<path>`: `classification.py`, `model_preprocessing.py`, `plan_generator.py`, and `benchmarks/execution.py` are absent at base. `approve_all` and `gate_n_jobs` coupling exist at base in `copilot_client.py` / `sklearn_compat.py`
- [x] Synthetic label-vocabulary check with pinned `files/benchmark-env/Scripts/python.exe`: a second `LabelEncoder` on already encoded `["no","yes"]` labels reports classes `['0','1']`
- [x] Train-only helpers inspected: `CrossFittedTargetEncoder.fit_transform` vs `transform`, `PredictiveTabularEngine.fit_transform` (`fit` then `transform`), `FittedRuleEngine` shared by `SemanticEngine._fit_plan` and `evaluate_plan_fit`
- [ ] Generated archive bodies and omitted large added-source dumps in the input map were not re-read; producers and current full sources were reviewed instead
- [ ] No provider calls, downloads, package installs, real benchmarks, or other review artifacts

## Scope and Limitations
Reviewed design consistency, dependency direction, abstraction ownership, and API contracts from base `bf119812162af87c29bc8d09bb3c705576b94f27` through worktree `46c7db9` plus uncommitted review fixes. Focus was production engine/classifier/wrapper ownership, train-only preprocessing boundaries, benchmark controller/state separation, and optional provider APIs. Benchmark study scripts were traced only where they own those boundaries. Style, naming preference, and missing docstring nits are not reported. Introduction was checked against base before a finding is called new.

## Issues

### Issue 1: Typed plan generation and plan evaluation do not share a provider contract
- **Severity**: High
- **File**: featcopilot/llm/semantic_engine.py
- **Line(s)**: 217-267, 379-404
- **Description**: New production plan mode and the new benchmark plan adapter are two APIs for one `FittedRulePlan` operation. `SemanticEngine._fit_plan` calls `request_fitted_plan` (`featcopilot/llm/plan_generator.py:11-90`) on the shared synchronous `FeatureClient`. The default backend is Copilot, and `CopilotFeatureClient.start` (`featcopilot/llm/copilot_client.py:100-104`) opens one reusable session with `PermissionHandler.approve_all`, without disabling tools, isolating the working directory, or rejecting permissions. `benchmarks/llm_evaluation.py` does not use that client or prompt. `PlanRequest.prompt` (`benchmarks/llm_evaluation.py:224-250`) asks for `FittedRulePlan.model_json_schema()` text, and `CopilotPlanProvider` (`benchmarks/llm_evaluation.py:280-433`) uses a fresh isolated cwd, denies permissions, disables tools, and checks an existing login. `evaluate_plan_fit` then correctly reuses `FittedRuleEngine`, so only the request/transport half is forked. The permissive Copilot session itself is pre-existing; routing the new no-code plan API through it, while adding a safer adapter that production does not call, is new.
- **Risk**: A benchmark that passes the isolated adapter does not certify `SemanticEngine(feature_format="plan")`. The default production transport can approve tool use and read the process working directory during a request that both prompts describe as schema-only and free of external data. That breaks the train-only plan boundary: workspace files outside the declared training schema can influence the returned rules. OpenAI and LiteLLM `send_prompt` are ordinary tool-free chat completions, so the shared `FeatureClient` protocol cannot even express the isolation contract the benchmark requires.
- **Suggested Fix**: Give typed plans one prompt builder and one transport. Production plan mode should use the isolated, tool-disabled, permission-rejecting adapter (or an equivalent `FeatureClient` policy), not the feature-code session. Keep `FittedRuleEngine` as the only fit/transform owner.

### Issue 2: Feature screening and the deployed classifier do not share a preprocessor
- **Severity**: Medium
- **File**: featcopilot/classification.py
- **Line(s)**: 386-420
- **Description**: `NestedBaseClassifier._fit_procedure` selects features with `PredictiveTabularEngine(screening="importance")`, whose private `_fit_learner` (`featcopilot/engines/predictive.py:1472-1537`) builds its own `ColumnTransformer`: it drops all-missing numeric columns, imputes categoricals with `"__missing__"`, and does not call `make_classification_pipeline`. The trial and winning models then go through `_make_model`, which calls `make_classification_pipeline` (`featcopilot/utils/model_preprocessing.py:248-291`). `CompleteBaseProcedure` (`benchmarks/prediction_ensemble.py:209-236`) overrides only that private scorer. For CatBoost it strips `cat_features` before `super().fit`, so production's native-categorical guard passes, importance screening sees an OHE numeric clone, and `_make_model` restores native `cat_features` through `tabular.make_model`. The same split exists for the public wrapper: `AutoFeatureEngineer` can request `categorical_encoding="target"` for screening, but `transform` does not own or return the encoder the downstream model must reuse. Surrounding pattern that should have been followed: `NumericQuantileClipper` and `CrossFittedTargetEncoder` were extracted so selection and deployment share one fitted policy.
- **Risk**: Published inner loss and the refit model can be native CatBoost, LightGBM, or a target-encoded pipeline while the feature plan was chosen under a different matrix. Changing missing-value or all-missing handling in one factory silently desynchronizes screening from deployment. Callers of the wrapper can set target encoding and still receive raw categoricals, with no fitted encoder to reuse.
- **Suggested Fix**: Inject one preprocessor factory into both importance screening and `_make_model`. Do not strip `cat_features` and continue; reject native categorical screening until that factory can represent it. If target encoding stays screening-only on `AutoFeatureEngineer`, do not accept it there unless the wrapper also exposes the fitted encoder.

### Issue 3: Benchmark controller is not separated from the method module
- **Severity**: Medium
- **File**: benchmarks/execution.py
- **Line(s)**: 117-140, 543-555, 1549
- **Description**: `run_state.py` is a real boundary: it persists cells and exposure and imports neither `tabular` nor `execution`. The new controller does not match that pattern. `execute` lazily imports `benchmarks.tabular` for dataset loading, `digest`, `DEFAULT_METHODS`, `LEARNERS`, `PROTOCOL_VERSION`, `source_fingerprint`, `failure_rows`, `run_isolated`, and `write_report`. `benchmarks/tabular.py:1549` imports `execute` back for the CLI. `tabular.py:838` and `:844` lazily import `regression_averaging` and `plan_validation`, while those modules import `tabular` again (`plan_validation.py` inside helpers; `regression_averaging.py:27-28` at module level, including `benchmarks.execution`). Label vocabulary has the same split ownership: `catalog.py:1103-1147` encodes targets and stores original `class_labels`; `tabular.load_real_dataset` encodes legacy targets at line 190; `execution._prepare_dataset` encodes every classification target again at lines 155-157 and writes `evaluation_classes` from that second encoder.
- **Risk**: The cycle currently survives only because the `execution`/`tabular` imports are function-local. Moving either to module level, which `regression_averaging.py` already does, raises `ImportError` during CLI or study import. The controller also cannot prepare, fingerprint, run, or record a cell without the method god-module, so protocol changes and method changes are the same edit. The second encoder is identity on `0..k-1` codes, but `evaluation_classes` becomes `["0","1"]` while catalog `class_labels` still holds the original labels. Nothing reads `evaluation_classes` today; the next consumer of that controller field will publish the wrong vocabulary.
- **Suggested Fix**: Make `execution` depend on `run_state` plus a small dataset/protocol interface, not on `tabular`. Keep method registries and `make_model` behind that interface. Encode labels once, in the loader that knows the original classes, and have the controller copy that vocabulary instead of fitting `LabelEncoder` again.

### Issue 4: Predictive engine parallelism is still controlled by the legacy gate knob
- **Severity**: Low
- **File**: featcopilot/transformers/sklearn_compat.py
- **Line(s)**: 160-165, 592-608
- **Description**: Pre-existing, not introduced by this diff: `AutoFeatureEngineer` documents `gate_n_jobs` as the do-no-harm gate's RandomForest worker count, but `_create_engine` passes `n_jobs=self.gate_n_jobs` into `PredictiveTabularEngine`. The new classifier API does not follow that shortcut: `NestedBaseClassifier` has an explicit `n_jobs` and uses it for both the engine and the thread pool. New benchmark construction still uses the old knob (`benchmarks/tabular.py` passes `gate_n_jobs=config["threads"]` into predictive `AutoFeatureEngineer`).
- **Risk**: Setting `gate_n_jobs` to change only the unused legacy gate also changes predictive screening thread limits and learner `n_jobs` overrides, which can change selected features. Users of the new classifier and the public wrapper now have two different parallelism contracts for the same engine.
- **Suggested Fix**: Add an explicit `n_jobs` on `AutoFeatureEngineer` for predictive selection, keep `gate_n_jobs` for the legacy gate only, and map benchmark `threads` to `n_jobs`.

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

### Driver triage and resolution update

The independent findings above remain intact.

- **Issue 1 - Fix in progress**: The new typed-plan path is being given an
  isolated, tool-disabled Copilot transport without altering the legacy
  code-mode permission default. Sharing the same learned-rule engine does
  not by itself enforce a transport boundary. Distinct production versus
  benchmark prompt budgets remain explicitly different protocols; prompt
  identity alone is not being imposed as an unrelated behavior change.
- **Issue 2 - Deferred architectural concern; claimed current contract
  violation not established**: The raw-feature wrapper deliberately does not
  return a fitted target encoder. `docs/user-guide/selection.md:265-269`
  states that it returns unencoded features and requires downstream
  train-only encoding. Returning its internal encoder would expose
  selection-partition state as a deployment transform, contrary to that
  boundary. The native CatBoost/OHE selection distinction is also expressly
  disclosed in `benchmarks/README.md:325-328`; the public classifier excludes
  native CatBoost (`docs/user-guide/classification.md:168`).
  Factory duplication can drift and remains a maintainability concern,
  but no concrete newly failing input or undisclosed equivalence claim was
  established here. A broad preprocessor rewrite or removal of supported
  experimental modes is deferred rather than called a completed fix.
- **Issue 3 - Partially addressed / partially deferred**: The original-label
  provenance error is being corrected with a focused loader/controller
  test. The broader cycle-breaking refactor is deferred: both
  `execution -> tabular -> regression_averaging` and the reverse import
  order succeeded in independent fresh processes. Hypothetical movement
  of lazy imports to module scope is not a demonstrated current import
  failure. Dependency coupling remains recorded as architectural debt.
- **Issue 4 - Documentation fixed; API redesign deferred**: The public
  `gate_n_jobs` docstring now explicitly states its predictive-engine
  thread/worker role. Existing wiring and defaults are preserved instead
  of introducing another parallelism parameter in this review fix.

### Driver artifact handling

This reviewer committed only its own artifact in `62cbb1c` before the
driver's final review check-in. No source fixes or other pending work were
included. That commit is retained without rewriting history; current-run
review artifacts remain explicitly excluded from subsequent review inputs.

### Driver verification update: Issue 3 label provenance

- **Status**: Concrete provenance defect fixed; broader dependency refactor remains deferred.
- **What changed**: `benchmarks/execution.py` maps observed encoded classes through the loader's original `class_labels`, including a correctly ordered subset when sampling drops a class. Prepared numeric targets and the original vocabulary are unchanged. Invalid vocabularies/code indices fail explicitly; raw-label diagnostic inputs retain their existing encoding path.
- **How verified**: Fifty-one focused metadata cases and the complete 250-test execution suite passed with the pinned interpreter; Black, Ruff and whitespace checks passed. No row-level raw targets or historical archive bytes were published or modified.

### Driver verification update: Issue 1 typed-plan isolation

- **Status**: Fixed for the newly introduced production typed-plan path.
- **What changed**: Added a production-owned `_plan_transport.py` shared with the benchmark provider. Plan mode uses a fresh isolated temporary workspace/session, denies permissions, disables tools/MCPs/skills/custom agents/config discovery, verifies existing authentication, and cleans up on normal and exceptional exits. Supplied rules remain provider-free. Legacy code-mode session defaults and the distinct production/benchmark prompts are unchanged.
- **Why**: Transport isolation belongs in the reusable production layer, not in a benchmark-only adapter or prompt wording. System temporary storage avoids adding a writable-project-directory requirement.
- **How verified**: The implementation's 427-test suite passed with both SDK 0.3.0 and 1.0.16 using mocked clients. The driver's combined plan/provider/controller/preprocessing/task suite passed 789 tests, package mypy passed 52 files and the wheel built. Cleanup, rejected permissions, isolation settings, event-loop behavior, invalid responses and supplied-rule bypass are covered without provider calls.

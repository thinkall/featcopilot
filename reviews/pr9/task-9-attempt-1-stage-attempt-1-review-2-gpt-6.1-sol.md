## Review Summary
- **Round**: 2
- **Theme**: Architecture & patterns
- **Mode**: resolved-plan
- **Model**: gpt-6.1-sol
- **Reasoning**: max
- **Artifact**: C:\code\featcopilot\reviews\pr9\task-9-attempt-1-stage-attempt-1-review-2-gpt-6.1-sol.md
- **Issues Found**: 3
- **Verdict**: ISSUES_FOUND

## Evidence Checklist
- [x] Read the assigned stage contract, `scope.json`, repository `AGENTS.md`, and the review skill's architecture/evidence requirements. Applied the supplied High/two-required-reviewer human override without dispatching another reviewer.
- [x] Examined production ownership and dependency direction in `featcopilot\classification.py`, `featcopilot\engines\predictive.py`, `featcopilot\transformers\sklearn_compat.py`, and their shared preprocessing/splitting utilities. Traced native-parameter cloning, member-local schemas/plans, training-only internal partitions, final refits, and inference paths.
- [x] Examined training-versus-inference hooks in `featcopilot\core\base.py`, `featcopilot\engines\tabular.py`, `featcopilot\engines\fitted_rules.py`, and `featcopilot\llm\semantic_engine.py`; compared their contracts with `featcopilot\utils\task.py` and related integration tests.
- [x] Traced benchmark/controller boundaries through `benchmarks\execution.py`, `benchmarks\run_state.py`, `benchmarks\tabular.py`, the calibration/blending controllers, and `benchmarks\prediction_ensemble.py`. Inspected checkpoint reuse, identity hashing, source allowlisting/archiving, and benchmark-to-production adaptation. Production sources do not import the benchmark package.
- [x] Inspected optional-provider contracts in `featcopilot\llm\_types.py`, `plan_generator.py`, `_execution.py`, the Copilot/OpenAI/LiteLLM clients, and `SemanticEngine._ensure_client`. No provider was started or called.
- [x] Independently checked introduction against `bf119812162af87c29bc8d09bb3c705576b94f27`: executed that revision's `TabularEngine` source in memory for Issues 2–3; verified `utils\model_preprocessing.py` and `classification.py` are absent from that base and added by this PR for Issue 1. Also compared existing public positional signatures across changed production modules.
- [x] Ran provider-free, in-memory synthetic checks with `C:\Users\lijiang1\.copilot\session-state\fc6747ca-1d37-46b8-a1c4-fb650ac71116\files\benchmark-env\Scripts\python.exe -B -` (`PYTHONDONTWRITEBYTECODE=1`, sklearn 1.7.2). Confirmed all three findings below. One-hot fixed-average ownership, batched prediction, unchanged fitted state after prediction, and pickle round-trip checks passed. Typed-plan provider-free fit/export/inference and distinct cross-fitted training outputs passed. In-memory `RunState.reusable` terminal-identity rejection and unfinished-preparation checks passed.
- [ ] Real-data benchmarks, downloads, live providers, package installation, full test suites, and generated archive-byte/checksum validation were not performed, as required by this assignment.

## Scope and Limitations

- Snapshot: base `bf119812162af87c29bc8d09bb3c705576b94f27` to HEAD `46c7db92165cb4078699c3f4cc00ae9855f3c002`, including the existing uncommitted review fixes. The supplied scope contains 720 paths, of which 126 are marked non-frozen/non-generated. This is a focused architecture/API review of the relevant full sources and surrounding patterns, not a claim to have executed or inspected every branch of that entire scope.
- Read actual source bodies rather than treating the trimmed input map as a complete diff. Generated archives were considered through their producers; their contents were not expanded.
- No current reviewer feedback, earlier findings, or driver-progress artifacts were consulted. No nested agents were launched.
- Existing imperfections were not promoted to introduced defects merely because these files changed. In particular, the wrappers' generated-only feature-name behavior already exists in the base and is not reported as a new regression.
- Source hashes at review: `featcopilot\engines\tabular.py` SHA-256 `4a1caa090bd35c90c71f18f00976f47cfcdcd92cb0c7f43b36eec106e0ed6b0e`; `featcopilot\utils\model_preprocessing.py` SHA-256 `2543f1a1aa2bd1989afd38369cc5e03f094c6a96b09630655327bdb9f6eb391f`.
- Only this assigned artifact was written. The driving agent owns code fixes, regression tests, subsequent gates, and inclusion of the artifact in the requested commit.

## Issues

### Issue 1: Shared target-category normalization changes dtype with the inference batch
- **Severity**: Medium
- **File**: `featcopilot\utils\model_preprocessing.py`
- **Line(s)**: 56–59; consumed at 163–171
- **Description**: `normalize_categorical_frame` casts to object **before** mapping, but pandas infers the mapped result's dtype again. A mixed training column containing strings and missing values normalizes to object; a valid inference batch containing only missing values normalizes to float64. On the pinned supported sklearn 1.7.2 runtime, `TargetEncoder.transform` then takes the numeric unknown-category path against its object-valued fitted categories and raises `TypeError: ufunc 'isnan' not supported`. Thus the shared preprocessing abstraction is not a stable categorical representation: a row can predict successfully in a mixed batch and fail when predicted alone.
- **Risk**: Both predictive selection models and the public classifier reuse this adapter. A single-row request or a shard with an entirely missing categorical column fails despite the documented missing-category and batch-invariant inference contracts. Ordinary mixed-batch tests do not establish the single-row API contract.
- **Surrounding Pattern / Contract**: `CrossFittedTargetEncoder` explicitly supports missing values as a category and applies frozen full-TRAIN maps at inference; `normalize_model_frame` explicitly preserves categorical roles. `tests\test_model_preprocessing.py:225–258` tests batch invariance, but its missing row remains grouped with a nonmissing row and therefore never creates the failing all-missing normalized column.
- **Base Verification**: The adapter and its production classifier consumers are newly added; `git ls-tree bf119812162af87c29bc8d09bb3c705576b94f27 -- featcopilot/utils/model_preprocessing.py featcopilot/classification.py` returns no entries, while the base-to-worktree name-status marks both added. This is an introduced API defect, not a claim about pre-existing target encoding.
- **Reproduction / Verification**:
  ```python
  X = pd.DataFrame({"category": ["a", "b", None] * 10})
  encoder = CrossFittedTargetEncoder().fit(X, np.arange(30) % 2)
  query = pd.DataFrame({"category": ["a", None]})
  encoder.transform(query)           # succeeds
  encoder.transform(query.iloc[1:])  # TypeError on sklearn 1.7.2
  ```
  The normalized dtypes are respectively object and float64. Separately reproduced through `FeatureEnsembleClassifier(mode="single", engineered=False, categorical_encoding="target")`: a three-row mixed batch succeeds, its missing-only singleton fails. An in-memory experiment forcing object dtype **after** normalization makes the isolated row match its whole-batch prediction; repository source was not modified.
- **Suggested Fix**: Make the normalized categorical output dtype stable after mapping, including entirely missing columns, while retaining actual missing values and the typed nonmissing tokens. Add adapter and public-classifier regression tests that compare every singleton to the same row in a mixed batch, including missing-only columns and multiclass encoding.

### Issue 2: Legacy tabular target preparation disagrees with the explicit regression API
- **Severity**: Medium
- **File**: `featcopilot\engines\tabular.py`
- **Line(s)**: 255–263
- **Description**: `_encode_target` now calls `is_string_dtype(y_series.dtype)`. For object dtype this is true even when every element is a real number, so valid object-backed numeric regression targets are converted to strings and label-encoded. The base used the value-aware `is_string_dtype(y_series)` check and retained these numeric target values. The new shared task utility correctly accepts them as regression, but `AutoFeatureEngineer(task="regression").fit(...)` merely validates that interpretation and passes the unchanged target to this separate, inconsistent encoder.
- **Risk**: The legacy target-mean feature can silently cease to represent the regression target. This affects both frozen inference maps and the newly cross-fitted training encodings; merely choosing `task="regression"` does not protect callers. The regression signal can even collapse to a constant, rather than just undergo harmless rescaling.
- **Surrounding Pattern / Contract**: `featcopilot\utils\task.py:101–120` preserves numeric regression magnitudes, and `tests\test_task_routing.py:79–83` explicitly verifies object-backed numeric targets are not label-encoded. The main wrapper exposes a task override precisely to avoid inconsistent target heuristics across its stages.
- **Base Verification**: Executed the exact base `TabularEngine` module from `git show bf119812162af87c29bc8d09bb3c705576b94f27:featcopilot/engines/tabular.py` in memory with the same current dependencies and inputs. The base yields category means `{"a": 5.0, "b": 200.0}`; the worktree yields `{"a": 1.5, "b": 1.5}`. The regression is the newly extracted dtype-only check, not the base's handling of these values.
- **Reproduction / Verification**:
  ```python
  X = pd.DataFrame({"category": ["a", "a", "b", "b"]})
  y = pd.Series([1.0, 9.0, 100.0, 300.0], dtype=object)
  engine = TabularEngine(
      include_transforms=[], polynomial_degree=1,
      onehot_ratio_threshold=0.0, target_encode_ratio_threshold=1.0,
      min_samples_per_category=1,
  ).fit(X, y)
  ```
  The current `category_target_encoded` output is `[1.5, 1.5, 1.5, 1.5]` instead of the base's `[5.0, 5.0, 200.0, 200.0]`. Also reproduced with the unmodified default `AutoFeatureEngineer(task="regression")` using three copies of each numeric target: all 12 inference values become 1.5 although shared task preparation preserves the original magnitudes.
- **Suggested Fix**: Preserve value-aware numeric-versus-label interpretation when preparing legacy targets; infer numeric objects or use the value-aware string test before label encoding. Keep explicit task semantics consistent across wrapper and engine without silently broadening accepted numeric-string targets. Add direct-engine and explicit-regression wrapper tests comparing object-backed numeric targets with equivalent numeric-dtype targets, for both cross-fitted training output and frozen inference maps.

### Issue 3: Inserting the CV option rebinds an existing public positional argument
- **Severity**: Medium
- **File**: `featcopilot\engines\tabular.py`
- **Line(s)**: 100–102
- **Description**: The new `target_encode_cv` argument is inserted before the existing positional `min_samples_per_category`. Consequently an unchanged nine-argument `TabularEngine(...)` call no longer sets the minimum category support. A former minimum of 1 now fails the CV minimum validation; values such as 2 or 4 silently configure the number of folds and leave category support at 3.
- **Risk**: Existing callers can fail at construction or change which categories are encoded without changing their configuration. This is a public constructor compatibility break, not a preference for keyword arguments.
- **Surrounding Pattern / Contract**: The PR preserves legacy defaults and appends its new `task` parameter after established selector positional parameters; `tests\test_task_routing.py:97–100` explicitly checks that compatibility pattern. The established `TabularEngine` category-support parameter should retain its positional slot as well.
- **Base Verification**: Inspected and executed the exact base constructor. Its ninth non-self positional parameter is `min_samples_per_category`; the worktree's is `target_encode_cv`. A source-level comparison of common public positional signatures across changed production modules identified this insertion.
- **Reproduction / Verification**:
  ```python
  TabularEngine(2, False, ["log"], None, False, True, 0.05, 0.5, 1)
  ```
  Base: constructs with `min_samples_per_category=1`. Worktree: raises Pydantic `ValidationError` for `target_encode_cv`. Substituting 2 or 4 constructs successfully but leaves `min_samples_per_category=3`, unlike the base.
- **Suggested Fix**: Keep `min_samples_per_category` in its established position and add `target_encode_cv` afterward, preferably as a new keyword-only option if that fits the API. Add a constructor compatibility regression test covering both previously valid minimum 1 and a nondefault minimum that would otherwise silently rebind.

## Resolution Log
_Updated by the driving agent as findings are addressed. Original findings above must remain intact._

### Issue 1
- **Status**: Open
- **What changed**: Pending; no production source was changed by this reviewer.
- **Why**: The assigned independent review is artifact-only; the driving agent owns fixes.
- **How verified**: Adapter and public-classifier missing-only inference reproductions on pinned sklearn 1.7.2; post-normalization object-dtype experiment passed in memory. No checked-in fix has been verified.

### Issue 2
- **Status**: Open
- **What changed**: Pending; no production source was changed by this reviewer.
- **Why**: Preserve legacy regression target semantics and align the engine with the explicit task API.
- **How verified**: Same-input execution of exact base versus worktree source, plus default-wrapper explicit-regression reproduction. No checked-in fix has been verified.

### Issue 3
- **Status**: Open
- **What changed**: Pending; no production source was changed by this reviewer.
- **Why**: Preserve the existing public constructor's positional contract.
- **How verified**: Exact-base/worktree construction with ninth arguments 1, 2, and 4; source-level positional signature comparison. No checked-in fix has been verified.

### Driver resolution update

- **Issue 1 - Fixed**: Categorical normalization now explicitly retains object dtype after token mapping. Added public-classifier binary/multiclass tests with and without engineering and strengthened the adapter check to compare every singleton, including an all-missing row, with the same mixed-batch result.
- **Issue 2 - Fixed**: Restored value-aware string detection in the legacy target encoder. Direct-engine and explicit-regression wrapper tests require object-backed numeric targets to match numeric-dtype cross-fitted training output and frozen inference means.
- **Issue 3 - Fixed**: Restored `min_samples_per_category` to its existing positional slot; the new `target_encode_cv` follows it as a keyword-only option. Tests cover positional support values 1, 2 and 4 plus explicit CV selection.
- **Why**: These are concrete batch/schema, target-value and compatibility regressions, not changes to statistical defaults.
- **How verified**: All ten added/strengthened reproductions failed before the fixes and passed afterward on the pinned environment. Package mypy passed 51 files; Black and Ruff passed. The broader architecture-stage validation and independent retry follow after all stage repairs are ready.

"""Mock-only coverage for bounded, training-only LLM plan evaluation."""

import json
import pickle
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from benchmarks.llm_evaluation import (
    BudgetExceededError,
    CopilotPlanProvider,
    EvaluationBudget,
    EvaluationLedger,
    HardCapViolationError,
    PlanProvider,
    PlanRequest,
    PlanResponse,
    PreflightBlockedError,
    ProviderCapabilities,
    build_plan_request,
    evaluate_plan_fit,
    preflight,
)
from featcopilot.core.fitted_rule import FittedRulePlan
from featcopilot.utils.models import DEFAULT_MODEL


def frequency_plan() -> FittedRulePlan:
    return FittedRulePlan.model_validate(
        {"rules": [{"name": "segment_frequency", "operator": "frequency", "columns": ["Customer segment"]}]}
    )


class MockProvider:
    capabilities = ProviderCapabilities(
        backend="mock",
        model=DEFAULT_MODEL,
        hard_output_cap=True,
        token_parameter="max_tokens",
        single_attempt=True,
        configured=True,
        evidence="In-process mock, no network; one deterministic response.",
        is_mock=True,
    )

    def __init__(self, response: str | None = None, error: Exception | None = None) -> None:
        self.response = frequency_plan().model_dump_json() if response is None else response
        self.error = error
        self.calls: list[tuple[PlanRequest, int]] = []
        self.output_tokens: int | None = 30
        self.before_call = None

    def send_plan(self, request: PlanRequest, *, max_tokens: int) -> PlanResponse:
        self.calls.append((request, max_tokens))
        if self.before_call is not None:
            self.before_call()
        if self.error is not None:
            raise self.error
        return PlanResponse(self.response, input_tokens=20, output_tokens=self.output_tokens)


@pytest.fixture
def frames():
    train = pd.DataFrame(
        {
            "Customer segment": ["PRIVATE_TRAIN_A", "PRIVATE_TRAIN_A", "PRIVATE_TRAIN_B"],
            "Annual income": [1.0, 3.0, 5.0],
        },
        index=[7, 7, 2],
    )
    test = pd.DataFrame(
        {"Customer segment": ["PRIVATE_TEST_C"], "Annual income": [900123.0]}, index=["PRIVATE_TEST_INDEX"]
    )
    y = pd.Series([10101.0, 20202.0, 30303.0], name="PRIVATE_TARGET", index=[20, 30, 40])
    return train, y, test


def run_fit(frames, tmp_path, provider: PlanProvider, **kwargs):
    return evaluate_plan_fit(
        *frames,
        provider=provider,
        ledger=EvaluationLedger(tmp_path),
        expected_backend=provider.capabilities.backend,
        **kwargs,
    )


def _process_reservation(args: tuple[str, PlanRequest]) -> int | None:
    output_dir, request = args
    try:
        return EvaluationLedger(output_dir, EvaluationBudget(max_calls=8)).reserve(request, MockProvider.capabilities)
    except BudgetExceededError:
        return None


@pytest.mark.parametrize("value", [0, -1, 4097, True, 1.5, "4096", None])
def test_token_budget_rejects_nonpositive_over_limit_and_nonintegers(value):
    with pytest.raises(ValueError, match="max_tokens"):
        EvaluationBudget(max_tokens=value)


@pytest.mark.parametrize("value", [0, -1, 101, True, 1.5, "100", None])
def test_call_budget_cannot_exceed_authorization(value):
    with pytest.raises(ValueError, match="max_calls"):
        EvaluationBudget(max_calls=value)


def test_budget_accepts_exact_authorized_boundaries():
    assert EvaluationBudget().max_calls == 100
    assert EvaluationBudget().max_tokens == 4096
    assert EvaluationBudget(max_calls=1, max_tokens=1).max_tokens == 1


def test_ledger_transactions_close_connections_after_commit_and_rollback(tmp_path):
    ledger = EvaluationLedger(tmp_path)
    with ledger._transaction() as committed:
        committed.execute("UPDATE budget SET halted=1 WHERE id=1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        committed.execute("SELECT 1")
    assert ledger.snapshot().halted

    with pytest.raises(RuntimeError, match="force rollback"):
        with ledger._transaction() as rolled_back:
            rolled_back.execute("UPDATE budget SET halted=0 WHERE id=1")
            raise RuntimeError("force rollback")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        rolled_back.execute("SELECT 1")
    assert ledger.snapshot().halted


def test_100_attempt_ceiling_across_new_ledger_instances(tmp_path, frames):
    request = build_plan_request(frames[0])
    for _ in range(100):
        EvaluationLedger(tmp_path).reserve(request, MockProvider.capabilities)
    with pytest.raises(BudgetExceededError):
        EvaluationLedger(tmp_path).reserve(request, MockProvider.capabilities)
    assert EvaluationLedger(tmp_path).snapshot().attempts == 100
    assert EvaluationLedger(tmp_path).snapshot().remaining == 0


def test_100_attempt_ceiling_with_concurrent_new_workers(tmp_path, frames):
    request = build_plan_request(frames[0])

    def reserve(_):
        try:
            return EvaluationLedger(tmp_path).reserve(request, MockProvider.capabilities)
        except BudgetExceededError:
            return None

    with ThreadPoolExecutor(max_workers=12) as executor:
        attempts = list(executor.map(reserve, range(130)))
    successful = [attempt for attempt in attempts if attempt is not None]
    assert len(successful) == len(set(successful)) == 100
    assert EvaluationLedger(tmp_path).snapshot().attempts == 100


def test_reservations_are_shared_across_separate_processes(tmp_path, frames):
    request = build_plan_request(frames[0])
    with ProcessPoolExecutor(max_workers=3) as executor:
        attempts = list(executor.map(_process_reservation, [(str(tmp_path), request)] * 16))
    successful = [attempt for attempt in attempts if attempt is not None]
    assert len(successful) == len(set(successful)) == 8
    snapshot = EvaluationLedger(tmp_path, EvaluationBudget(max_calls=8)).snapshot()
    assert snapshot.attempts == snapshot.pending == 8
    assert snapshot.remaining == 0


def test_resume_cannot_reset_or_raise_persisted_limits(tmp_path, frames):
    ledger = EvaluationLedger(tmp_path, EvaluationBudget(max_calls=2, max_tokens=100))
    ledger.reserve(build_plan_request(frames[0]), MockProvider.capabilities)
    with pytest.raises(ValueError, match="persisted budget"):
        EvaluationLedger(tmp_path)
    assert EvaluationLedger(tmp_path, ledger.budget).snapshot().attempts == 1


def test_failed_and_explicit_retry_attempts_are_durable_before_provider_call(tmp_path, frames):
    ledger = EvaluationLedger(tmp_path, EvaluationBudget(max_calls=2))
    provider = MockProvider(error=TimeoutError("FAKE_CREDENTIAL_DO_NOT_LOG"))

    def check_reservation():
        snapshot = EvaluationLedger(tmp_path, ledger.budget).snapshot()
        assert snapshot.attempts == len(provider.calls)
        assert snapshot.pending == 1

    provider.before_call = check_reservation
    for _ in range(2):
        with pytest.raises(TimeoutError):
            evaluate_plan_fit(*frames, provider=provider, ledger=ledger, expected_backend="mock")
    with pytest.raises(BudgetExceededError):
        evaluate_plan_fit(*frames, provider=provider, ledger=ledger, expected_backend="mock")
    assert len(provider.calls) == 2
    assert ledger.snapshot().failed == 2
    assert ledger.snapshot().pending == 0


@pytest.mark.parametrize(
    "change",
    [
        {"hard_output_cap": False},
        {"single_attempt": False},
        {"configured": False},
        {"evidence": ""},
        {"token_parameter": None},
        {"is_mock": False},
        {"model": "unrequested-model"},
    ],
)
def test_blocked_preflight_makes_zero_calls_and_no_reservations(change, tmp_path, frames):
    provider = MockProvider()
    provider.capabilities = replace(provider.capabilities, **change)
    ledger = EvaluationLedger(tmp_path)
    with pytest.raises(PreflightBlockedError):
        evaluate_plan_fit(*frames, provider=provider, ledger=ledger, expected_backend="mock")
    assert provider.calls == []
    assert ledger.snapshot().attempts == 0


def test_default_copilot_preflight_is_explicitly_blocked_even_with_live_approval():
    report = preflight(live_approved=True)
    assert not report.allowed
    assert report.backend == "copilot"
    assert report.model == DEFAULT_MODEL
    assert "hard_output_cap_unverified" in report.blockers


def test_real_provider_requires_explicit_review_and_matching_backend():
    provider = MockProvider()
    provider.capabilities = replace(provider.capabilities, is_mock=False)
    assert not preflight(provider, expected_backend="mock").allowed
    assert not preflight(provider, live_approved=True).allowed
    assert preflight(provider, expected_backend="mock", live_approved=True).allowed


@pytest.mark.parametrize(
    ("hard_cap_verified", "auth_verified", "blocker"),
    [
        (False, True, "hard_output_cap_unverified"),
        (True, False, "configuration_unverified"),
    ],
)
def test_github_copilot_mapping_alone_cannot_enable_live_calls(
    hard_cap_verified, auth_verified, blocker, tmp_path, frames
):
    provider = MockProvider()
    provider.capabilities = ProviderCapabilities(
        backend="github_copilot",
        model=DEFAULT_MODEL,
        token_parameter="max_completion_tokens",
        hard_output_cap=hard_cap_verified,
        single_attempt=True,
        configured=auth_verified,
        evidence="Static upstream GPT-5 parameter mapping only; no live enforcement or account probe.",
    )
    report = preflight(provider, expected_backend="github_copilot", live_approved=True)
    assert report.blockers == (blocker,)
    assert not report.allowed
    assert report.model == "gpt-5.6-sol"
    ledger = EvaluationLedger(tmp_path)
    with pytest.raises(PreflightBlockedError, match=blocker):
        evaluate_plan_fit(
            *frames,
            provider=provider,
            ledger=ledger,
            expected_backend="github_copilot",
            live_approved=True,
        )
    assert provider.calls == []
    assert ledger.snapshot().attempts == 0


def test_train_only_context_preserves_headers_without_samples_targets_or_test_metadata(frames):
    train, y, test = frames
    request = build_plan_request(
        train,
        target_available=True,
        column_descriptions={"Customer segment": "Public business segment description"},
        approved_aggregates=("row_count", "missing_fraction", "numeric_mean"),
    )
    context = json.loads(request.context_json)
    assert list(context["columns"]) == ["Customer segment", "Annual income"]
    assert context["column_descriptions"]["Customer segment"] == "Public business segment description"
    assert context["train_aggregates"]["row_count"] == 3
    assert context["train_aggregates"]["numeric_mean"]["Annual income"] == 3.0
    for private in ["PRIVATE_TRAIN_A", "PRIVATE_TRAIN_B", "PRIVATE_TEST_C", str(test.iloc[0, 1]), y.name, "10101"]:
        assert private not in request.prompt
    assert "PRIVATE_TEST_INDEX" not in request.prompt
    assert "train_aggregates" not in json.loads(build_plan_request(train).context_json)


def test_plan_prompt_matches_all_nine_core_operators_and_rowwise_contract(frames):
    prompt = build_plan_request(frames[0]).prompt
    schema = json.loads(prompt.splitlines()[1])
    operators = schema["$defs"]["FittedRuleSpec"]["properties"]["operator"]["enum"]
    assert set(operators) == {
        "standardize",
        "clip",
        "frequency",
        "group_mean",
        "target_mean",
        "rare_category",
        "multiply",
        "divide",
        "log1p",
    }
    contract = prompt.split("Allowed operators:", 1)[1].split("Only the named parameters", 1)[0]
    for operator in operators:
        assert operator in contract
    for instruction in (
        "multiply (two distinct numeric columns, parameters {})",
        "divide (two distinct numeric columns [numerator, denominator], parameters {})",
        "log1p (one numeric column, computes log1p(abs(x)), parameters {})",
        "rowwise and learn no statistics",
        "missing/nonfinite operands produce NaN",
        "NaN for zero denominators",
        "Finite-input overflow saturates to float64 bounds",
    ):
        assert instruction in contract


@pytest.mark.parametrize("aggregate", ["sample_rows", "target_mean", "category_values", "arbitrary"])
def test_only_explicit_allowlisted_train_aggregates_can_be_exported(frames, aggregate):
    with pytest.raises(ValueError, match="aggregate"):
        build_plan_request(frames[0], approved_aggregates=(aggregate,))


def test_one_typed_plan_per_fit_usage_diagnostics_and_no_transform_calls(tmp_path, frames):
    provider = MockProvider()
    result = run_fit(frames, tmp_path, provider, column_descriptions={"Annual income": "Income in public units"})
    assert len(provider.calls) == 1
    assert provider.calls[0][1] == 4096
    pd.testing.assert_frame_equal(result.train_X[frames[0].columns], frames[0])
    pd.testing.assert_frame_equal(result.test_X[frames[2].columns], frames[2])
    np.testing.assert_allclose(result.train_X["segment_frequency"], [2 / 3, 2 / 3, 1 / 3])
    assert result.test_X["segment_frequency"].tolist() == [0.0]
    pd.testing.assert_frame_equal(result.transform(frames[2]), result.test_X)
    assert len(provider.calls) == 1
    assert result.diagnostics.calls == 1
    assert result.diagnostics.accepted_rules == 1
    assert result.diagnostics.input_tokens == 20
    assert result.diagnostics.output_tokens == 30
    assert result.diagnostics.total_attempts == 1


def test_smaller_budget_cap_is_passed_exactly_and_unknown_usage_is_not_zero(tmp_path, frames):
    provider = MockProvider()
    provider.output_tokens = None
    ledger = EvaluationLedger(tmp_path, EvaluationBudget(max_tokens=42))
    result = evaluate_plan_fit(*frames, ledger=ledger, provider=provider, expected_backend="mock")
    assert provider.calls[0][1] == 42
    assert result.diagnostics.output_tokens is None


def test_changing_heldout_rows_and_target_values_cannot_change_exported_context(tmp_path, frames):
    train, y, test = frames
    provider = MockProvider()
    run_fit(frames, tmp_path, provider, approved_aggregates=("numeric_mean",))
    changed_test = test.copy()
    changed_test["Annual income"] = -888888.0
    changed_test["Customer segment"] = "HELDOUT_CHANGED_SECRET"
    changed_y = y + 111111.0
    run_fit((train, changed_y, changed_test), tmp_path, provider, approved_aggregates=("numeric_mean",))
    assert len(provider.calls) == 2
    assert provider.calls[0][0].context_json == provider.calls[1][0].context_json
    assert provider.calls[0][0].context_sha256 == provider.calls[1][0].context_sha256


def test_provided_plan_and_no_llm_are_offline_and_do_not_require_ledger(frames):
    provider = MockProvider(error=AssertionError("Must not call provider"))
    result = evaluate_plan_fit(*frames, plan=frequency_plan(), provider=provider)
    assert result.diagnostics.calls == 0
    assert result.diagnostics.source == "provided"
    pd.testing.assert_frame_equal(result.transform(frames[2]), result.test_X)
    baseline = evaluate_plan_fit(*frames, mode="no_llm", provider=provider)
    pd.testing.assert_frame_equal(baseline.train_X, frames[0])
    pd.testing.assert_frame_equal(baseline.test_X, frames[2])
    assert provider.calls == []


@pytest.mark.parametrize(
    "response",
    [
        "not JSON FAKE_CREDENTIAL",
        '{"rules":[{"name":"code","operator":"execute","columns":["Customer segment"]}]}',
        '{"rules":[{"name":"f","operator":"frequency","columns":["unknown"]}]}',
        '{"rules":[{"name":"Annual income","operator":"frequency","columns":["Customer segment"]}]}',
        '{"rules":[{"name":"f","operator":"frequency","columns":["Customer segment"],"parameters":{"code":"run"}}]}',
        '{"rules":[],"python":"unsafe"}',
    ],
)
def test_invalid_plans_fail_without_fallback_and_are_audited(response, tmp_path, frames):
    provider = MockProvider(response)
    with pytest.raises(ValueError):
        run_fit(frames, tmp_path, provider)
    assert len(provider.calls) == 1
    assert EvaluationLedger(tmp_path).snapshot().failed == 1


def test_plan_rule_budget_is_validated_not_truncated(tmp_path, frames):
    plan = frequency_plan().model_dump()
    plan["rules"].append({**plan["rules"][0], "name": "another_frequency"})
    with pytest.raises(ValueError, match="rule"):
        run_fit(frames, tmp_path, MockProvider(json.dumps(plan)), max_rules=1)


def test_empty_plan_is_accepted_identity_not_fabricated(tmp_path, frames):
    result = run_fit(frames, tmp_path, MockProvider('{"rules":[]}'))
    assert result.diagnostics.accepted_rules == 0
    pd.testing.assert_frame_equal(result.train_X, frames[0])


def test_target_rules_use_core_crossfitting_and_frozen_test_state():
    train = pd.DataFrame({"key": ["a", "b", "c", "d"]}, index=[9, 9, 1, 3])
    y = pd.Series([0.0, 2.0, 4.0, 10.0])
    test = pd.DataFrame({"key": ["a", "unknown"]})
    plan = FittedRulePlan.model_validate(
        {"rules": [{"name": "encoded", "operator": "target_mean", "columns": ["key"], "parameters": {"smoothing": 0}}]}
    )
    result = evaluate_plan_fit(train, y, test, plan=plan, task="regression", target_cv=4)
    np.testing.assert_allclose(result.train_X["encoded"], (y.sum() - y.to_numpy()) / 3)
    np.testing.assert_allclose(result.test_X["encoded"], [0.0, 4.0])
    grouped = evaluate_plan_fit(
        train, y, test, plan=plan, task="regression", target_cv=2, groups=np.array([1, 1, 2, 2])
    )
    np.testing.assert_allclose(grouped.train_X["encoded"], [7.0, 7.0, 1.0, 1.0])


def test_boundary_metadata_stays_local_and_reaches_both_hybrid_engines(tmp_path, frames):
    train, _, test = frames
    plan = {
        "rules": [
            {
                "name": "encoded",
                "operator": "target_mean",
                "columns": ["Customer segment"],
                "parameters": {"smoothing": 0},
            }
        ]
    }
    provider = MockProvider(json.dumps(plan))
    # Group members share a feature value, so exclusion must use groups, not rows.
    extended = pd.concat([train.iloc[:2], train.iloc[2:], train.iloc[2:]])
    groups = np.array(["PRIVATE_GROUP_A", "PRIVATE_GROUP_A", "PRIVATE_GROUP_B", "PRIVATE_GROUP_B"])
    y = pd.Series([0.0, 2.0, 4.0, 10.0])
    result = run_fit(
        (extended, y, test),
        tmp_path,
        provider,
        mode="hybrid",
        task="regression",
        target_cv=2,
        groups=groups,
        deterministic_factory=TrainingMeanFeatures,
    )
    np.testing.assert_allclose(result.train_X["encoded"], [7.0, 7.0, 1.0, 1.0])
    np.testing.assert_array_equal(result.deterministic_engine.kwargs["groups"], groups)
    assert "PRIVATE_GROUP" not in provider.calls[0][0].prompt


class TrainingMeanFeatures:
    fitted_batches: ClassVar[list[pd.DataFrame]] = []

    def fit_transform(self, X, y=None, **kwargs):
        self.fitted_batches.append(X.copy())
        self.mean = X["Annual income"].mean()
        self.y = y
        self.kwargs = kwargs
        return self.transform(X)

    def transform(self, X):
        return X.assign(centered=X["Annual income"] - self.mean)


def test_hybrid_fits_each_engine_on_same_original_training_fold_and_merges(frames):
    TrainingMeanFeatures.fitted_batches = []
    result = evaluate_plan_fit(
        *frames, mode="hybrid", plan=frequency_plan(), deterministic_factory=TrainingMeanFeatures
    )
    assert len(TrainingMeanFeatures.fitted_batches) == 1
    pd.testing.assert_frame_equal(TrainingMeanFeatures.fitted_batches[0], frames[0])
    assert list(result.train_X) == ["Customer segment", "Annual income", "centered", "segment_frequency"]
    assert result.test_X["centered"].tolist() == [900120.0]
    pd.testing.assert_frame_equal(result.transform(frames[2]), result.test_X)
    pd.testing.assert_series_equal(result.deterministic_engine.y, frames[1])


def test_hybrid_refuses_name_collisions_instead_of_silently_overwriting(frames):
    plan = frequency_plan().model_dump()
    plan["rules"][0]["name"] = "centered"
    with pytest.raises(ValueError, match="collid"):
        evaluate_plan_fit(
            *frames, mode="hybrid", plan=FittedRulePlan.model_validate(plan), deterministic_factory=TrainingMeanFeatures
        )


def test_namespace_preserves_both_colliding_features_and_records_safe_provenance(frames):
    raw_plan = frequency_plan().model_dump()
    raw_plan["rules"][0]["name"] = "centered"
    raw_plan["rules"][0]["explanation"] = "PRIVATE_EXPLANATION_NOT_FOR_DIAGNOSTICS"
    plan = FittedRulePlan.model_validate(raw_plan)
    original_plan = plan.model_dump()
    result = evaluate_plan_fit(
        *frames,
        mode="hybrid",
        plan=plan,
        deterministic_factory=TrainingMeanFeatures,
        composition_policy="namespace",
    )
    assert list(result.train_X) == ["Customer segment", "Annual income", "centered", "llm__centered"]
    np.testing.assert_allclose(result.train_X["centered"], [-2, 0, 2])
    np.testing.assert_allclose(result.train_X["llm__centered"], [2 / 3, 2 / 3, 1 / 3])
    assert result.test_X["centered"].tolist() == [900120.0]
    assert result.test_X["llm__centered"].tolist() == [0.0]
    pd.testing.assert_frame_equal(result.train_X[frames[0].columns], frames[0])
    pd.testing.assert_frame_equal(result.test_X[frames[2].columns], frames[2])
    assert result.plan_column_mapping == {"centered": "llm__centered"}
    assert result.diagnostics.composition_policy == "namespace"
    assert result.diagnostics.calls == 0
    assert result.diagnostics.accepted_rules == 1
    aliases = result.diagnostics.plan_feature_aliases
    assert len(aliases) == 1
    assert asdict(aliases[0]) == {
        "plan_name": "centered",
        "output_name": "llm__centered",
        "operator": "frequency",
        "source_columns": ("Customer segment",),
    }
    with pytest.raises(FrozenInstanceError):
        aliases[0].output_name = "changed"
    external_mapping = result.plan_column_mapping
    external_mapping["centered"] = "changed"
    assert result.plan_column_mapping == {"centered": "llm__centered"}
    serialized = json.dumps(asdict(result.diagnostics))
    assert "PRIVATE_" not in serialized
    assert "context_json" not in serialized and '"mapping"' not in serialized
    assert plan.model_dump() == original_plan
    assert result.plan_engine.rules[0].name == "centered"


def test_namespace_resolves_raw_prefix_deterministic_and_repeated_alias_collisions(frames):
    class PrefixFeatures(TrainingMeanFeatures):
        def transform(self, X):
            return super().transform(X).assign(llm__centered__3=123.0)

    train, y, test = frames
    train = train.assign(llm__centered=11, llm__centered__2=22, llm__llm__other=33)
    test = test.assign(llm__centered=44, llm__centered__2=55, llm__llm__other=66)
    plan = FittedRulePlan.model_validate(
        {
            "rules": [
                {"name": "centered", "operator": "frequency", "columns": ["Customer segment"]},
                {"name": "centered__3", "operator": "standardize", "columns": ["Annual income"]},
                {"name": "centered__4", "operator": "log1p", "columns": ["Annual income"]},
                {"name": "llm__other", "operator": "clip", "columns": ["Annual income"]},
            ]
        }
    )
    result = evaluate_plan_fit(
        train,
        y,
        test,
        mode="hybrid",
        plan=plan,
        deterministic_factory=PrefixFeatures,
        composition_policy="namespace",
    )
    assert result.plan_column_mapping == {
        "centered": "llm__centered__4",
        "centered__3": "llm__centered__3__2",
        "centered__4": "llm__centered__4__2",
        "llm__other": "llm__llm__other__2",
    }
    assert result.train_X.columns.is_unique
    assert result.train_X.shape[1] == len(train.columns) + 2 + len(plan.rules)
    pd.testing.assert_frame_equal(result.train_X[train.columns], train)
    pd.testing.assert_frame_equal(result.test_X[test.columns], test)
    assert result.train_X["llm__centered__3"].tolist() == [123.0] * len(train)
    reference = evaluate_plan_fit(train, y, test, plan=plan)
    for source_name, output_name in result.plan_column_mapping.items():
        pd.testing.assert_series_equal(result.train_X[output_name], reference.train_X[source_name], check_names=False)
        pd.testing.assert_series_equal(result.test_X[output_name], reference.test_X[source_name], check_names=False)


def test_namespace_does_not_cascade_renames_when_another_rule_uses_the_prefix(frames):
    plan = FittedRulePlan.model_validate(
        {
            "rules": [
                {"name": "centered", "operator": "frequency", "columns": ["Customer segment"]},
                {"name": "llm__centered", "operator": "log1p", "columns": ["Annual income"]},
            ]
        }
    )
    result = evaluate_plan_fit(
        *frames,
        mode="hybrid",
        plan=plan,
        deterministic_factory=TrainingMeanFeatures,
        composition_policy="namespace",
    )
    assert result.plan_column_mapping == {"centered": "llm__centered", "llm__centered": "llm__llm__centered"}
    np.testing.assert_allclose(result.train_X["llm__centered"], [2 / 3, 2 / 3, 1 / 3])
    np.testing.assert_allclose(result.train_X["llm__llm__centered"], np.log1p(frames[0]["Annual income"]))
    assert result.train_X.columns.is_unique


def test_namespace_preserves_cross_fitted_training_features_instead_of_retransforming(frames):
    plan = FittedRulePlan.model_validate(
        {
            "rules": [
                {
                    "name": "centered",
                    "operator": "target_mean",
                    "columns": ["Customer segment"],
                    "parameters": {"smoothing": 0},
                }
            ]
        }
    )
    reference = evaluate_plan_fit(*frames, plan=plan, task="regression", target_cv=3)
    result = evaluate_plan_fit(
        *frames,
        mode="hybrid",
        plan=plan,
        task="regression",
        target_cv=3,
        deterministic_factory=TrainingMeanFeatures,
        composition_policy="namespace",
    )
    pd.testing.assert_series_equal(result.train_X["llm__centered"], reference.train_X["centered"], check_names=False)
    pd.testing.assert_series_equal(result.test_X["llm__centered"], reference.test_X["centered"], check_names=False)
    assert not np.array_equal(result.train_X["llm__centered"], result.plan_engine.transform(frames[0])["centered"])


def test_namespace_is_frozen_for_pickle_batches_row_indices_and_changed_heldout_values(frames):
    train, y, test = frames
    raw_plan = frequency_plan().model_dump()
    raw_plan["rules"][0]["name"] = "centered"
    plan = FittedRulePlan.model_validate(raw_plan)
    test = pd.concat([train.iloc[[1, 0]], test, train.iloc[[2]]])
    result = evaluate_plan_fit(
        train,
        y,
        test,
        mode="hybrid",
        plan=plan,
        deterministic_factory=TrainingMeanFeatures,
        composition_policy="namespace",
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.plan_column_mapping == result.plan_column_mapping
    assert restored.diagnostics.plan_feature_aliases == result.diagnostics.plan_feature_aliases
    pd.testing.assert_frame_equal(restored.train_X, result.train_X)
    pd.testing.assert_frame_equal(restored.transform(test), result.test_X)
    batches = pd.concat(
        [restored.transform(test.iloc[:1]), restored.transform(test.iloc[1:3]), restored.transform(test.iloc[3:])]
    )
    pd.testing.assert_frame_equal(batches, result.test_X)
    pd.testing.assert_frame_equal(restored.transform(test.iloc[:0]), result.test_X.iloc[:0])
    changed_test = test.iloc[::-1].copy()
    changed_test["Annual income"] = -9e10
    changed_test["Customer segment"] = "NEW_HELDOUT_VALUE"
    repeated = evaluate_plan_fit(
        train,
        y,
        changed_test,
        mode="hybrid",
        plan=plan,
        deterministic_factory=TrainingMeanFeatures,
        composition_policy="namespace",
    )
    assert repeated.plan_column_mapping == restored.plan_column_mapping
    assert repeated.train_X.columns.equals(restored.train_X.columns)
    pd.testing.assert_frame_equal(repeated.test_X, restored.transform(changed_test))


@pytest.mark.parametrize("change", ["add", "drop", "reorder"])
def test_namespace_rejects_future_deterministic_schema_drift_instead_of_reallocating(frames, change):
    class ChangingFeatures(TrainingMeanFeatures):
        schema_change = None

        def transform(self, X):
            result = super().transform(X).assign(second=1.0)
            if self.schema_change == "add":
                return result.assign(llm__centered=-1.0)
            if self.schema_change == "drop":
                return result.drop(columns=["centered"])
            if self.schema_change == "reorder":
                return result[list(X.columns) + ["second", "centered"]]
            return result

    raw_plan = frequency_plan().model_dump()
    raw_plan["rules"][0]["name"] = "centered"
    result = evaluate_plan_fit(
        *frames,
        mode="hybrid",
        plan=FittedRulePlan.model_validate(raw_plan),
        deterministic_factory=ChangingFeatures,
        composition_policy="namespace",
    )
    mapping = result.plan_column_mapping
    result.deterministic_engine.schema_change = change
    with pytest.raises(ValueError, match="schema"):
        result.transform(frames[2])
    assert result.plan_column_mapping == mapping


@pytest.mark.parametrize("change", ["add", "drop", "reorder"])
def test_namespace_rejects_changed_raw_schema(frames, change):
    result = evaluate_plan_fit(*frames, plan=frequency_plan(), composition_policy="namespace")
    test = frames[2]
    if change == "add":
        test = test.assign(llm__segment_frequency=1)
    elif change == "drop":
        test = test.drop(columns=["Annual income"])
    else:
        test = test[list(test.columns)[::-1]]
    with pytest.raises(ValueError, match="schema"):
        result.transform(test)


@pytest.mark.parametrize("mode", ["typed_llm", "hybrid"])
def test_namespace_preserves_prior_rejection_of_plan_names_colliding_with_originals(frames, mode):
    raw_plan = frequency_plan().model_dump()
    raw_plan["rules"][0]["name"] = "Annual income"
    with pytest.raises(ValueError, match="collid"):
        evaluate_plan_fit(
            *frames,
            mode=mode,
            plan=FittedRulePlan.model_validate(raw_plan),
            deterministic_factory=TrainingMeanFeatures if mode == "hybrid" else None,
            composition_policy="namespace",
        )


def test_namespace_is_opt_in_and_does_not_rename_no_llm_features(frames):
    legacy = evaluate_plan_fit(*frames, plan=frequency_plan())
    assert legacy.diagnostics.composition_policy == "raise"
    assert legacy.plan_column_mapping == {"segment_frequency": "segment_frequency"}
    assert legacy.train_X.columns[-1] == "segment_frequency"
    typed = evaluate_plan_fit(*frames, plan=frequency_plan(), composition_policy="namespace")
    assert typed.train_X.columns[-1] == "llm__segment_frequency"
    for factory in (None, TrainingMeanFeatures):
        original = evaluate_plan_fit(*frames, mode="no_llm", deterministic_factory=factory)
        namespaced = evaluate_plan_fit(
            *frames, mode="no_llm", deterministic_factory=factory, composition_policy="namespace"
        )
        pd.testing.assert_frame_equal(original.train_X, namespaced.train_X)
        pd.testing.assert_frame_equal(original.test_X, namespaced.test_X)
        pd.testing.assert_frame_equal(namespaced.transform(frames[2]), original.test_X)
        assert namespaced.plan_column_mapping == {}
        assert namespaced.diagnostics.plan_feature_aliases == ()


def test_namespace_accepts_empty_plan_without_fabricated_aliases(frames):
    result = evaluate_plan_fit(*frames, plan=FittedRulePlan(rules=[]), composition_policy="namespace")
    assert result.plan_column_mapping == {}
    pd.testing.assert_frame_equal(result.train_X, frames[0])
    pd.testing.assert_frame_equal(result.transform(frames[2]), frames[2])


@pytest.mark.parametrize("invalid", ["duplicate", "operator", "parameters"])
def test_namespace_does_not_repair_malformed_provider_plans_or_retry(invalid, tmp_path, frames):
    raw_plan = frequency_plan().model_dump()
    raw_plan["rules"][0]["name"] = "centered"
    if invalid == "duplicate":
        raw_plan["rules"].append(dict(raw_plan["rules"][0]))
    elif invalid == "operator":
        raw_plan["rules"][0]["operator"] = "execute"
    else:
        raw_plan["rules"][0]["parameters"] = {"unexpected": 1}
    provider = MockProvider(json.dumps(raw_plan))
    with pytest.raises(ValueError, match="Invalid fitted rule plan"):
        run_fit(
            frames,
            tmp_path,
            provider,
            mode="hybrid",
            deterministic_factory=TrainingMeanFeatures,
            composition_policy="namespace",
        )
    assert len(provider.calls) == 1
    assert EvaluationLedger(tmp_path).snapshot().failed == 1


def test_namespace_mock_proposal_is_accepted_with_one_attempt_and_no_transform_calls(tmp_path, frames):
    raw_plan = frequency_plan().model_dump()
    raw_plan["rules"][0]["name"] = "centered"
    provider = MockProvider(json.dumps(raw_plan))
    result = run_fit(
        frames,
        tmp_path,
        provider,
        mode="hybrid",
        deterministic_factory=TrainingMeanFeatures,
        composition_policy="namespace",
    )
    assert result.diagnostics.accepted_rules == 1
    assert result.diagnostics.calls == 1
    assert result.plan_column_mapping == {"centered": "llm__centered"}
    result.transform(frames[2])
    assert len(provider.calls) == 1
    assert EvaluationLedger(tmp_path).snapshot().succeeded == 1


def test_invalid_composition_policy_fails_before_any_provider_call(tmp_path, frames):
    provider = MockProvider()
    with pytest.raises(ValueError, match="composition_policy"):
        run_fit(frames, tmp_path, provider, composition_policy="guess")
    assert provider.calls == []
    assert EvaluationLedger(tmp_path).snapshot().attempts == 0


def test_deterministic_boundary_rejects_dropped_rows_or_modified_originals(frames):
    class BadTransformer(TrainingMeanFeatures):
        def transform(self, X):
            return X.iloc[:1].copy()

    with pytest.raises(ValueError, match="preserve"):
        evaluate_plan_fit(*frames, mode="no_llm", deterministic_factory=BadTransformer)


def test_observed_cap_violation_halts_future_calls_durably(tmp_path, frames):
    provider = MockProvider()
    provider.output_tokens = 4097
    with pytest.raises(HardCapViolationError):
        run_fit(frames, tmp_path, provider)
    provider.output_tokens = 1
    with pytest.raises(PreflightBlockedError, match="halted"):
        run_fit(frames, tmp_path, provider)
    assert len(provider.calls) == 1
    assert EvaluationLedger(tmp_path).snapshot().halted


@pytest.mark.parametrize("usage", [-1, True, "30", 1.5])
def test_invalid_provider_usage_is_counted_without_success_fallback(usage, tmp_path, frames):
    provider = MockProvider()
    provider.output_tokens = usage
    with pytest.raises(ValueError, match="usage"):
        run_fit(frames, tmp_path, provider)
    assert EvaluationLedger(tmp_path).snapshot().failed == 1


def test_success_audit_stores_hashes_and_usage_but_not_plan_or_context(tmp_path, frames):
    provider = MockProvider()
    result = run_fit(frames, tmp_path, provider)
    ledger = EvaluationLedger(tmp_path)
    with closing(sqlite3.connect(ledger.path)) as connection, connection:
        row = connection.execute(
            "SELECT context_sha256,max_tokens,status,input_tokens,output_tokens,accepted_rules,error_code FROM attempts"
        ).fetchone()
        stored = "\n".join(connection.iterdump())
    assert row == (result.diagnostics.context_sha256, 4096, "succeeded", 20, 30, 1, None)
    assert "segment_frequency" not in stored
    assert "Customer segment" not in stored
    assert "PRIVATE_TRAIN" not in stored


def test_audit_and_logs_never_store_prompt_plan_raw_exceptions_or_credentials(tmp_path, frames, caplog):
    provider = MockProvider(error=TimeoutError("FAKE_CREDENTIAL_VALUE"))
    with pytest.raises(TimeoutError):
        run_fit(
            frames,
            tmp_path,
            provider,
            column_descriptions={"Annual income": "PRIVATE_DESCRIPTION"},
            fit_id="PRIVATE_FIT_IDENTIFIER",
        )
    ledger = EvaluationLedger(tmp_path)
    with closing(sqlite3.connect(ledger.path)) as connection, connection:
        stored = "\n".join(connection.iterdump())
    for private in [
        "PRIVATE_DESCRIPTION",
        "PRIVATE_FIT_IDENTIFIER",
        "FAKE_CREDENTIAL_VALUE",
        "PRIVATE_TRAIN",
        "Annual income",
    ]:
        assert private not in stored
        assert private not in caplog.text
    assert "provider_error" in stored
    assert "context_sha256" in stored


def test_schema_errors_fail_before_provider_attempt(tmp_path, frames):
    train, y, test = frames
    provider = MockProvider()
    with pytest.raises(ValueError, match="schema"):
        run_fit((train, y, test.rename(columns={"Annual income": "different"})), tmp_path, provider)
    assert provider.calls == []


@pytest.fixture
def copilot_sdk():
    from copilot.generated.session_events import AssistantMessageData
    from copilot.session import CopilotSession

    with patch("copilot.CopilotClient", autospec=True) as factory:
        client = factory.return_value
        client.get_auth_status.return_value = SimpleNamespace(isAuthenticated=True)
        session = MagicMock(spec=CopilotSession)
        session.send_and_wait.return_value = SimpleNamespace(
            data=AssistantMessageData(content=frequency_plan().model_dump_json(), message_id="mock")
        )
        client.create_session.return_value = session
        yield factory, client, session


def test_soft_policy_requires_an_explicit_boolean():
    assert EvaluationBudget().require_hard_token_cap is True
    for invalid in (None, 0, 1, "false"):
        with pytest.raises(ValueError, match="require_hard_token_cap"):
            EvaluationBudget(require_hard_token_cap=invalid)


def test_copilot_soft_policy_preflight_is_honest_and_never_starts_sdk(copilot_sdk):
    factory, _, _ = copilot_sdk
    provider = CopilotPlanProvider()
    report = preflight(provider, live_approved=True)
    assert not report.allowed
    assert "hard_output_cap_unverified" in report.blockers
    assert not provider.capabilities.hard_output_cap
    assert provider.capabilities.token_parameter is None
    assert provider.capabilities.attempt_scope == "logical_api_call"
    budget = EvaluationBudget(require_hard_token_cap=False)
    assert not preflight(provider, budget=budget).allowed
    soft = preflight(provider, budget=budget, live_approved=True)
    assert soft.allowed
    assert not soft.require_hard_token_cap
    assert not soft.token_cap_enforced
    factory.assert_not_called()


def test_copilot_hard_default_cannot_make_calls_even_with_live_approval(copilot_sdk, tmp_path, frames):
    factory, _, _ = copilot_sdk
    ledger = EvaluationLedger(tmp_path)
    with pytest.raises(PreflightBlockedError, match="hard_output_cap_unverified"):
        evaluate_plan_fit(*frames, provider=CopilotPlanProvider(), ledger=ledger, live_approved=True)
    factory.assert_not_called()
    assert ledger.snapshot().attempts == 0


def test_copilot_adapter_keeps_model_and_sends_one_isolated_typed_request(copilot_sdk, tmp_path, frames):
    factory, client, session = copilot_sdk
    budget = EvaluationBudget(require_hard_token_cap=False)
    ledger = EvaluationLedger(tmp_path, budget)

    async def start():
        snapshot = EvaluationLedger(tmp_path, budget).snapshot()
        assert snapshot.attempts == snapshot.pending == 1

    client.start.side_effect = start
    result = evaluate_plan_fit(
        *frames, provider=CopilotPlanProvider(timeout=19), ledger=ledger, live_approved=True, fit_id="pilot/fold/0"
    )
    client.start.assert_awaited_once()
    client.get_auth_status.assert_awaited_once()
    client.stop.assert_awaited_once()
    session.send_and_wait.assert_awaited_once()
    session.on.return_value.assert_called_once()
    config = factory.call_args.args[0]
    assert config.use_logged_in_user is True
    assert config.github_token is None
    assert "--no-custom-instructions" in config.cli_args
    assert "--no-auto-login" not in config.cli_args
    assert "--disable-builtin-mcps" in config.cli_args
    assert config.cwd != str(tmp_path)
    options = client.create_session.call_args.kwargs
    assert options["working_directory"] == config.cwd
    assert options["available_tools"] == options["tools"] == []
    assert options["mcp_servers"] == {}
    assert options["enable_config_discovery"] is False
    assert options["infinite_sessions"] == {"enabled": False}
    assert options["model"] == "gpt-5.6-sol"
    assert options["on_permission_request"](None, {}).kind == "reject"
    assert "max_tokens" not in options
    assert "temperature" not in options
    prompt = session.send_and_wait.call_args.args[0]
    assert session.send_and_wait.call_args.kwargs == {"timeout": 19}
    assert "4096" in prompt and "best-effort" in prompt
    assert "Customer segment" in prompt and "Annual income" in prompt
    assert "FittedRuleSpec" in prompt
    assert "PRIVATE_TRAIN" not in prompt and "PRIVATE_TEST" not in prompt and "10101" not in prompt
    assert result.diagnostics.calls == 1
    assert result.diagnostics.attempt_scope == "logical_api_call"
    assert result.diagnostics.token_cap_enforced is False
    assert result.diagnostics.require_hard_token_cap is False
    assert result.diagnostics.input_tokens is None
    assert result.diagnostics.output_tokens is None
    assert result.diagnostics.usage_scope == "unknown"
    assert result.diagnostics.response_chars == len(frequency_plan().model_dump_json())
    pd.testing.assert_frame_equal(result.transform(frames[2]), result.test_X)
    session.send_and_wait.assert_awaited_once()
    with closing(sqlite3.connect(ledger.path)) as connection, connection:
        row = connection.execute(
            "SELECT require_hard_token_cap,token_cap_enforced,attempt_scope,response_chars,usage_scope FROM attempts"
        ).fetchone()
    assert row == (0, 0, "logical_api_call", result.diagnostics.response_chars, "unknown")


@pytest.mark.parametrize("stage", ["start", "auth", "create_session", "send", "parse"])
def test_copilot_adapter_stops_on_failure_without_retry(stage, copilot_sdk, tmp_path, frames):
    _, client, session = copilot_sdk
    if stage == "start":
        client.start.side_effect = TimeoutError("mock startup failure")
    elif stage == "auth":
        client.get_auth_status.side_effect = TimeoutError("mock auth status failure")
    elif stage == "create_session":
        client.create_session.side_effect = TimeoutError("mock session failure")
    elif stage == "send":
        session.send_and_wait.side_effect = TimeoutError("mock send failure")
    else:
        session.send_and_wait.return_value.data.content = "invalid typed JSON"
    ledger = EvaluationLedger(tmp_path, EvaluationBudget(require_hard_token_cap=False))
    with pytest.raises((TimeoutError, ValueError)):
        evaluate_plan_fit(*frames, provider=CopilotPlanProvider(), ledger=ledger, live_approved=True)
    client.stop.assert_awaited_once()
    assert session.send_and_wait.await_count == (1 if stage in ("send", "parse") else 0)
    assert ledger.snapshot().attempts == ledger.snapshot().failed == 1


def test_missing_copilot_login_fails_before_session_or_generation(copilot_sdk, tmp_path, frames):
    _, client, session = copilot_sdk
    client.get_auth_status.return_value = SimpleNamespace(isAuthenticated=False)
    ledger = EvaluationLedger(tmp_path, EvaluationBudget(require_hard_token_cap=False))
    with pytest.raises(PreflightBlockedError, match="authenticate outside"):
        evaluate_plan_fit(*frames, provider=CopilotPlanProvider(), ledger=ledger, live_approved=True)
    client.get_auth_status.assert_awaited_once()
    client.create_session.assert_not_awaited()
    session.send_and_wait.assert_not_awaited()
    client.stop.assert_awaited_once()
    assert ledger.snapshot().attempts == ledger.snapshot().failed == 1


def test_explicit_soft_retry_reserves_another_attempt_and_stops_again(copilot_sdk, tmp_path, frames):
    _, client, session = copilot_sdk
    session.send_and_wait.side_effect = TimeoutError("mock failure")
    ledger = EvaluationLedger(tmp_path, EvaluationBudget(max_calls=2, require_hard_token_cap=False))
    provider = CopilotPlanProvider()
    for _ in range(2):
        with pytest.raises(TimeoutError):
            evaluate_plan_fit(*frames, provider=provider, ledger=ledger, live_approved=True)
    with pytest.raises(BudgetExceededError):
        evaluate_plan_fit(*frames, provider=provider, ledger=ledger, live_approved=True)
    assert session.send_and_wait.await_count == client.stop.await_count == 2
    assert ledger.snapshot().failed == 2


def test_copilot_records_observed_usage_without_treating_soft_size_as_hard(copilot_sdk, tmp_path, frames):
    from copilot.generated.session_events import AssistantUsageData

    _, _, session = copilot_sdk
    response = session.send_and_wait.return_value

    async def send(prompt, *, timeout):
        callback = session.on.call_args.args[0]
        callback(SimpleNamespace(data=AssistantUsageData(model=DEFAULT_MODEL, input_tokens=11.0, output_tokens=5000.0)))
        return response

    session.send_and_wait.side_effect = send
    ledger = EvaluationLedger(tmp_path, EvaluationBudget(require_hard_token_cap=False))
    result = evaluate_plan_fit(*frames, provider=CopilotPlanProvider(), ledger=ledger, live_approved=True)
    assert result.diagnostics.input_tokens == 11
    assert result.diagnostics.output_tokens == 5000
    assert result.diagnostics.usage_scope == "sdk_usage_events"
    assert result.diagnostics.usage_events == 1
    assert result.diagnostics.token_cap_enforced is False
    assert not ledger.snapshot().halted


def test_copilot_retains_message_usage_when_no_usage_event_is_reported(copilot_sdk, tmp_path, frames):
    _, _, session = copilot_sdk
    session.send_and_wait.return_value.data.output_tokens = 23.0
    ledger = EvaluationLedger(tmp_path, EvaluationBudget(require_hard_token_cap=False))
    result = evaluate_plan_fit(*frames, provider=CopilotPlanProvider(), ledger=ledger, live_approved=True)
    assert result.diagnostics.input_tokens is None
    assert result.diagnostics.output_tokens == 23
    assert result.diagnostics.usage_scope == "assistant_message"


def test_soft_policy_never_changes_model_or_copies_usage_from_another_model(copilot_sdk, tmp_path, frames):
    from copilot.generated.session_events import AssistantUsageData

    with pytest.raises(ValueError, match="model"):
        CopilotPlanProvider(model="another-model")
    _, client, session = copilot_sdk
    response = session.send_and_wait.return_value

    async def send(prompt, *, timeout):
        session.on.call_args.args[0](SimpleNamespace(data=AssistantUsageData(model="another-model", output_tokens=1.0)))
        return response

    session.send_and_wait.side_effect = send
    ledger = EvaluationLedger(tmp_path, EvaluationBudget(require_hard_token_cap=False))
    with pytest.raises(ValueError, match="model"):
        evaluate_plan_fit(*frames, provider=CopilotPlanProvider(), ledger=ledger, live_approved=True)
    client.stop.assert_awaited_once()
    session.send_and_wait.assert_awaited_once()


def test_explicit_soft_policy_transition_preserves_all_prior_attempts(tmp_path, frames):
    request = build_plan_request(frames[0])
    hard = EvaluationLedger(tmp_path)
    for _ in range(99):
        hard.reserve(request, MockProvider.capabilities)
    soft_budget = EvaluationBudget(require_hard_token_cap=False)
    soft = EvaluationLedger(tmp_path, soft_budget)
    assert soft.snapshot().attempts == 99
    with pytest.raises(ValueError, match="persisted budget"):
        hard.reserve(request, MockProvider.capabilities)
    with pytest.raises(ValueError, match="persisted budget"):
        EvaluationLedger(tmp_path)
    soft.reserve(request, MockProvider.capabilities)
    with pytest.raises(BudgetExceededError):
        EvaluationLedger(tmp_path, soft_budget).reserve(request, MockProvider.capabilities)
    with closing(sqlite3.connect(soft.path)) as connection, connection:
        assert connection.execute("SELECT require_hard_token_cap FROM budget").fetchone() == (0,)
        assert connection.execute(
            "SELECT require_hard_token_cap,COUNT(*) FROM attempts GROUP BY require_hard_token_cap"
        ).fetchall() == [(0, 1), (1, 99)]


def test_v1_ledger_migration_preserves_count_and_unknown_legacy_enforcement(tmp_path, frames):
    with closing(sqlite3.connect(tmp_path / "llm_evaluation.sqlite3")) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE budget(id INTEGER PRIMARY KEY,version INTEGER,max_calls INTEGER,max_tokens INTEGER,halted INTEGER);
            INSERT INTO budget VALUES(1,1,100,4096,0);
            CREATE TABLE attempts(
                id INTEGER PRIMARY KEY AUTOINCREMENT,context_sha256 TEXT,request_sha256 TEXT,provider_sha256 TEXT,
                fit_sha256 TEXT,max_tokens INTEGER,status TEXT,created_at TEXT,finished_at TEXT,
                input_tokens INTEGER,output_tokens INTEGER,accepted_rules INTEGER,error_code TEXT
            );
            INSERT INTO attempts(context_sha256,max_tokens,status) VALUES('legacy-hash',4096,'failed');
        """
        )
    budget = EvaluationBudget(require_hard_token_cap=False)
    ledger = EvaluationLedger(tmp_path, budget)
    assert ledger.snapshot().attempts == ledger.snapshot().failed == 1
    ledger.reserve(build_plan_request(frames[0]), MockProvider.capabilities)
    with closing(sqlite3.connect(ledger.path)) as connection, connection:
        assert connection.execute("SELECT version,require_hard_token_cap FROM budget").fetchone() == (2, 0)
        assert connection.execute(
            "SELECT token_cap_enforced,require_hard_token_cap FROM attempts WHERE id=1"
        ).fetchone() == (None, 1)
    assert EvaluationLedger(tmp_path, budget).snapshot().attempts == 2


def test_soft_policy_keeps_100_ceiling_across_concurrent_workers(tmp_path, frames):
    request = build_plan_request(frames[0])
    budget = EvaluationBudget(require_hard_token_cap=False)

    def reserve(_):
        try:
            return EvaluationLedger(tmp_path, budget).reserve(request, MockProvider.capabilities)
        except BudgetExceededError:
            return None

    with ThreadPoolExecutor(max_workers=12) as executor:
        attempts = list(executor.map(reserve, range(125)))
    assert len({attempt for attempt in attempts if attempt is not None}) == 100
    assert EvaluationLedger(tmp_path, budget).snapshot().remaining == 0

"""Comprehensive tests for LLM modules.

Tests FeatureCodeGenerator, FeatureExplainer, OpenAI/Copilot clients,
SemanticEngine, and TransformRuleGenerator using mocking (no real API calls).
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from featcopilot.core.feature import Feature, FeatureOrigin, FeatureSet, FeatureType
from featcopilot.core.transform_rule import TransformRule
from featcopilot.llm.code_generator import FeatureCodeGenerator
from featcopilot.llm.copilot_client import CopilotConfig, CopilotFeatureClient, SyncCopilotFeatureClient
from featcopilot.llm.explainer import FeatureExplainer
from featcopilot.llm.litellm_client import LiteLLMConfig, LiteLLMFeatureClient, SyncLiteLLMFeatureClient
from featcopilot.llm.openai_client import OpenAIClientConfig, OpenAIFeatureClient, SyncOpenAIFeatureClient
from featcopilot.llm.semantic_engine import SemanticEngine, SemanticEngineConfig
from featcopilot.llm.transform_rule_generator import TransformRuleGenerator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """Small DataFrame used across tests."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 15),
            "income": np.random.uniform(20000, 150000, 15).round(2),
            "score": np.random.uniform(0, 100, 15).round(2),
            "category": np.random.choice(["A", "B", "C"], 15),
        }
    )


@pytest.fixture
def column_info():
    return {"age": "integer", "income": "float", "score": "float", "category": "string"}


@pytest.fixture
def mock_client():
    """Pre-configured MagicMock acting as a sync LLM client."""
    client = MagicMock()
    client.start.return_value = client
    client.stop.return_value = None
    client.generate_feature_code.return_value = "result = df['age'] / (df['income'] + 1e-8)"
    client.validate_feature_code.return_value = {"valid": True, "error": None, "warnings": []}
    client.explain_feature.return_value = "This feature captures the relationship between age and income."
    client.suggest_features.return_value = [
        {
            "name": "age_income_ratio",
            "code": "result = df['age'] / (df['income'] + 1e-8)",
            "explanation": "Ratio of age to income",
            "source_columns": ["age", "income"],
        },
        {
            "name": "score_zscore",
            "code": "result = (df['score'] - df['score'].mean()) / (df['score'].std() + 1e-8)",
            "explanation": "Z-score of score",
            "source_columns": ["score"],
        },
    ]
    client.send_prompt.return_value = "mock prompt response"
    return client


# ===========================================================================
# 1. FeatureCodeGenerator
# ===========================================================================


class TestFeatureCodeGenerator:
    """Tests for FeatureCodeGenerator."""

    def test_init_copilot_backend(self):
        gen = FeatureCodeGenerator(backend="copilot")
        assert gen.backend == "copilot"
        assert gen._client is None

    def test_init_openai_backend(self):
        gen = FeatureCodeGenerator(backend="openai", api_key="test-key")
        assert gen.backend == "openai"
        assert gen.api_key == "test-key"

    def test_init_litellm_backend(self):
        gen = FeatureCodeGenerator(backend="litellm", api_base="http://localhost:8000")
        assert gen.backend == "litellm"
        assert gen.api_base == "http://localhost:8000"

    def test_init_defaults(self):
        gen = FeatureCodeGenerator()
        assert gen.model == "gpt-5.2"
        assert gen.validate is True
        assert gen.verbose is False

    # ---- _clean_code ---------------------------------------------------------

    def test_clean_code_removes_markdown(self):
        gen = FeatureCodeGenerator()
        code = "```python\nresult = df['a'] + df['b']\n```"
        assert gen._clean_code(code) == "result = df['a'] + df['b']"

    def test_clean_code_removes_comments(self):
        gen = FeatureCodeGenerator()
        code = "# compute ratio\nresult = df['a'] / df['b']"
        assert gen._clean_code(code) == "result = df['a'] / df['b']"

    def test_clean_code_ensures_result_assignment(self):
        gen = FeatureCodeGenerator()
        code = "df['a'] + df['b']"
        cleaned = gen._clean_code(code)
        assert cleaned.startswith("result = ")

    def test_clean_code_replaces_variable_name(self):
        gen = FeatureCodeGenerator()
        code = "output = df['a'] * 2"
        cleaned = gen._clean_code(code)
        assert cleaned.startswith("result =")

    def test_clean_code_keeps_existing_result(self):
        gen = FeatureCodeGenerator()
        code = "result = df['a'] * 2"
        assert gen._clean_code(code) == "result = df['a'] * 2"

    # ---- _generate_name ------------------------------------------------------

    def test_generate_name_basic(self):
        gen = FeatureCodeGenerator()
        name = gen._generate_name("Calculate BMI from height and weight")
        assert "bmi" in name
        assert "_" in name or name == "bmi"

    def test_generate_name_special_chars(self):
        gen = FeatureCodeGenerator()
        name = gen._generate_name("price/quantity ratio!! #1")
        # Should only contain a-z, 0-9, _
        assert all(c.isalnum() or c == "_" for c in name)

    def test_generate_name_stop_words_filtered(self):
        gen = FeatureCodeGenerator()
        name = gen._generate_name("calculate the ratio from income")
        # "calculate", "the", "from" are stop words
        assert "calculate" not in name
        assert "the" not in name

    def test_generate_name_empty_fallback(self):
        gen = FeatureCodeGenerator()
        name = gen._generate_name("the and for")
        assert name == "custom_feature"

    # ---- _detect_source_columns ----------------------------------------------

    def test_detect_source_columns_single_quote(self):
        gen = FeatureCodeGenerator()
        code = "result = df['age'] / df['income']"
        cols = gen._detect_source_columns(code, ["age", "income", "score"])
        assert "age" in cols
        assert "income" in cols
        assert "score" not in cols

    def test_detect_source_columns_double_quote(self):
        gen = FeatureCodeGenerator()
        code = 'result = df["age"] + df["score"]'
        cols = gen._detect_source_columns(code, ["age", "income", "score"])
        assert set(cols) == {"age", "score"}

    def test_detect_source_columns_dot_access(self):
        gen = FeatureCodeGenerator()
        code = "result = df.age * df.income"
        cols = gen._detect_source_columns(code, ["age", "income", "score"])
        assert set(cols) == {"age", "income"}

    def test_detect_source_columns_none_found(self):
        gen = FeatureCodeGenerator()
        code = "result = 42"
        cols = gen._detect_source_columns(code, ["age", "income"])
        assert cols == []

    # ---- _fix_common_issues --------------------------------------------------

    def test_fix_division_by_zero(self):
        gen = FeatureCodeGenerator()
        code = "result = df['a'] / df['b']"
        fixed = gen._fix_common_issues(code, "ZeroDivisionError: division by zero")
        assert "1e-8" in fixed

    def test_fix_keyerror_unchanged(self):
        gen = FeatureCodeGenerator()
        code = "result = df['missing']"
        fixed = gen._fix_common_issues(code, "KeyError: 'missing' not found")
        assert fixed == code

    def test_fix_syntax_returns_code(self):
        gen = FeatureCodeGenerator()
        code = "result = df['age'] + df['income']"
        fixed = gen._fix_common_issues(code, "SyntaxError: invalid syntax")
        # Method attempts quote normalization; code should be returned intact for valid quotes
        assert "result" in fixed
        assert "df" in fixed

    # ---- generate (mocked) ---------------------------------------------------

    def test_generate_with_mock_client(self, mock_client, column_info, sample_df):
        gen = FeatureCodeGenerator(validate=False)
        gen._client = mock_client

        feature = gen.generate("ratio of age to income", column_info, sample_data=sample_df)

        assert isinstance(feature, Feature)
        assert feature.origin == FeatureOrigin.LLM_GENERATED
        assert feature.code is not None
        mock_client.generate_feature_code.assert_called_once()

    def test_generate_with_validation(self, mock_client, column_info, sample_df):
        gen = FeatureCodeGenerator(validate=True)
        gen._client = mock_client

        feature = gen.generate("ratio of age to income", column_info, sample_data=sample_df)

        assert isinstance(feature, Feature)
        mock_client.validate_feature_code.assert_called_once()

    def test_generate_validation_failure_triggers_fix(self, mock_client, column_info, sample_df):
        mock_client.validate_feature_code.return_value = {
            "valid": False,
            "error": "division by zero",
            "warnings": [],
        }
        gen = FeatureCodeGenerator(validate=True, verbose=True)
        gen._client = mock_client

        feature = gen.generate("ratio of age to income", column_info, sample_data=sample_df)
        assert isinstance(feature, Feature)

    # ---- generate_batch ------------------------------------------------------

    def test_generate_batch_success(self, mock_client, column_info):
        gen = FeatureCodeGenerator(validate=False)
        gen._client = mock_client

        features = gen.generate_batch(["ratio of age to income", "score squared"], column_info)

        assert len(features) == 2
        assert all(isinstance(f, Feature) for f in features)

    def test_generate_batch_one_fails(self, mock_client, column_info):
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("LLM error")
            return "result = df['age'] * 2"

        mock_client.generate_feature_code.side_effect = side_effect

        gen = FeatureCodeGenerator(validate=False, verbose=True)
        gen._client = mock_client

        features = gen.generate_batch(["ok feature", "bad feature", "another ok"], column_info)
        assert len(features) == 2

    # ---- generate_domain_features --------------------------------------------

    def test_generate_domain_features_healthcare(self, mock_client, column_info):
        gen = FeatureCodeGenerator(validate=False)
        gen._client = mock_client

        features = gen.generate_domain_features("healthcare", column_info, n_features=3)
        assert len(features) <= 3
        assert all(isinstance(f, Feature) for f in features)

    def test_generate_domain_features_finance(self, mock_client, column_info):
        gen = FeatureCodeGenerator(validate=False)
        gen._client = mock_client

        features = gen.generate_domain_features("finance", column_info, n_features=2)
        assert len(features) <= 2

    def test_generate_domain_features_unknown_domain(self, mock_client, column_info):
        gen = FeatureCodeGenerator(validate=False)
        gen._client = mock_client

        features = gen.generate_domain_features("astrology", column_info, n_features=2)
        # Unknown domain still returns features from fallback prompts
        assert isinstance(features, list)


# ===========================================================================
# 2. FeatureExplainer
# ===========================================================================


class TestFeatureExplainer:
    """Tests for FeatureExplainer."""

    def test_init_copilot(self):
        exp = FeatureExplainer(backend="copilot")
        assert exp.backend == "copilot"

    def test_init_openai(self):
        exp = FeatureExplainer(backend="openai", api_key="k")
        assert exp.backend == "openai"

    def test_init_litellm(self):
        exp = FeatureExplainer(backend="litellm")
        assert exp.backend == "litellm"

    # ---- explain_feature -----------------------------------------------------

    def test_explain_feature(self, mock_client):
        exp = FeatureExplainer()
        exp._client = mock_client

        feature = Feature(
            name="age_income_ratio",
            code="result = df['age'] / df['income']",
            source_columns=["age", "income"],
        )

        explanation = exp.explain_feature(feature, task_description="predict churn")

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        mock_client.explain_feature.assert_called_once()

    # ---- explain_features (FeatureSet) ---------------------------------------

    def test_explain_features_skips_existing(self, mock_client):
        exp = FeatureExplainer()
        exp._client = mock_client

        f1 = Feature(name="f1", source_columns=["a"], explanation="already explained")
        f2 = Feature(name="f2", source_columns=["b"], code="result = df['b'] * 2")
        fs = FeatureSet(features=[f1, f2])

        explanations = exp.explain_features(fs, task_description="test")

        assert explanations["f1"] == "already explained"
        assert "f2" in explanations
        # explain_feature should be called only for f2
        mock_client.explain_feature.assert_called_once()

    def test_explain_features_exception(self, mock_client):
        mock_client.explain_feature.side_effect = RuntimeError("LLM down")

        exp = FeatureExplainer(verbose=True)
        exp._client = mock_client

        f1 = Feature(name="f1", source_columns=["a", "b"], code="result = df['a']")
        fs = FeatureSet(features=[f1])

        explanations = exp.explain_features(fs)

        assert "f1" in explanations
        # Fallback explanation should mention source columns
        assert "a" in explanations["f1"]

    # ---- generate_feature_report ---------------------------------------------

    def test_generate_feature_report_structure(self, mock_client, sample_df):
        exp = FeatureExplainer()
        exp._client = mock_client

        f1 = Feature(
            name="age",
            dtype=FeatureType.NUMERIC,
            origin=FeatureOrigin.ORIGINAL,
            source_columns=["age"],
            code="result = df['age']",
        )
        f2 = Feature(
            name="score",
            dtype=FeatureType.NUMERIC,
            origin=FeatureOrigin.LLM_GENERATED,
            source_columns=["score"],
            code="result = df['score']",
        )
        fs = FeatureSet(features=[f1, f2])

        report = exp.generate_feature_report(fs, sample_df, task_description="predict churn")

        assert "# Feature Engineering Report" in report
        assert "predict churn" in report
        assert "## Features by Origin" in report
        assert "## Feature Details" in report
        assert "### age" in report
        assert "### score" in report
        assert "```python" in report

    def test_generate_feature_report_features_by_origin(self, mock_client, sample_df):
        exp = FeatureExplainer()
        exp._client = mock_client

        f1 = Feature(name="age", origin=FeatureOrigin.ORIGINAL, source_columns=["age"])
        f2 = Feature(name="score", origin=FeatureOrigin.ORIGINAL, source_columns=["score"])
        f3 = Feature(name="ratio", origin=FeatureOrigin.LLM_GENERATED, source_columns=["age", "score"])
        fs = FeatureSet(features=[f1, f2, f3])

        report = exp.generate_feature_report(fs, sample_df)

        assert "original: 2" in report
        assert "llm_generated: 1" in report


# ===========================================================================
# 3. OpenAIFeatureClient
# ===========================================================================


class TestOpenAIClientConfig:
    """Tests for OpenAIClientConfig."""

    def test_defaults(self):
        config = OpenAIClientConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096
        assert config.timeout == 60.0
        assert config.api_key is None
        assert config.api_base is None
        assert config.api_version is None

    def test_custom_values(self):
        config = OpenAIClientConfig(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2048,
            api_key="sk-test",
            api_base="https://custom.endpoint/v1",
            api_version="2024-12-01-preview",
        )
        assert config.model == "gpt-4-turbo"
        assert config.temperature == 0.7
        assert config.api_key == "sk-test"
        assert config.api_version == "2024-12-01-preview"


class TestSyncOpenAIFeatureClient:
    """Tests for SyncOpenAIFeatureClient (mock mode)."""

    def test_creation(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        assert client._async_client is not None
        assert client._async_client._openai_available is False

    def test_start_stop_lifecycle(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()
        assert client._async_client._is_started is True
        client.stop()
        assert client._async_client._is_started is False

    def test_suggest_features_mock_mode(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()

        suggestions = client.suggest_features(
            column_info={"age": "int", "income": "float"},
            task_description="predict churn",
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert "name" in suggestions[0]
        client.stop()

    def test_explain_feature_mock_mode(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()

        explanation = client.explain_feature(
            feature_name="age_ratio",
            feature_code="result = df['age'] / df['income']",
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        client.stop()

    def test_generate_feature_code_mock_mode(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()

        code = client.generate_feature_code(
            description="ratio of age to income",
            column_info={"age": "int", "income": "float"},
        )

        assert isinstance(code, str)
        assert len(code) > 0
        client.stop()

    def test_validate_feature_code_mock_mode(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()

        result = client.validate_feature_code(
            code="result = df['a'] + df['b']",
            sample_data={"a": [1, 2, 3], "b": [4, 5, 6]},
        )

        assert result["valid"] is True
        assert result["error"] is None
        client.stop()

    def test_validate_feature_code_syntax_error(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()

        result = client.validate_feature_code(code="result = (")

        assert result["valid"] is False
        assert "Syntax error" in result["error"]
        client.stop()


# ===========================================================================
# 4. CopilotFeatureClient
# ===========================================================================


class TestCopilotConfig:
    """Tests for CopilotConfig."""

    def test_defaults(self):
        config = CopilotConfig()
        assert config.model == "gpt-5.2"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096
        assert config.timeout == 60.0
        assert config.streaming is False

    def test_custom_values(self):
        config = CopilotConfig(model="gpt-4o", temperature=0.5, max_tokens=8192, streaming=True)
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192
        assert config.streaming is True


class TestSyncCopilotFeatureClient:
    """Tests for SyncCopilotFeatureClient (mock mode)."""

    def test_creation(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        assert client._async_client is not None
        assert client._async_client._copilot_available is False

    def test_start_stop_lifecycle(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()
        assert client._async_client._is_started is True
        client.stop()
        assert client._async_client._is_started is False

    def test_suggest_features_mock_mode(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()

        suggestions = client.suggest_features(
            column_info={"age": "int", "income": "float", "score": "float"},
            task_description="predict churn",
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert "name" in suggestions[0]
        client.stop()

    def test_generate_feature_code_mock_mode(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()

        code = client.generate_feature_code(
            description="ratio of age to income",
            column_info={"age": "int", "income": "float"},
        )

        assert isinstance(code, str)
        assert len(code) > 0
        client.stop()

    def test_explain_feature_mock_mode(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()

        explanation = client.explain_feature(
            feature_name="age_ratio",
            feature_code="result = df['age'] / df['income']",
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        client.stop()

    def test_validate_feature_code_valid(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()

        result = client.validate_feature_code(
            code="result = df['x'] + df['y']",
            sample_data={"x": [1, 2], "y": [3, 4]},
        )

        assert result["valid"] is True
        client.stop()

    def test_validate_feature_code_runtime_error(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()

        result = client.validate_feature_code(
            code="result = df['nonexistent']",
            sample_data={"x": [1, 2]},
        )

        assert result["valid"] is False
        assert "Runtime error" in result["error"]
        client.stop()


# ===========================================================================
# 5. SemanticEngine
# ===========================================================================


class TestSemanticEngineConfig:
    """Tests for SemanticEngineConfig."""

    def test_defaults(self):
        config = SemanticEngineConfig()
        assert config.name == "SemanticEngine"
        assert config.model == "gpt-5.2"
        assert config.max_suggestions == 20
        assert config.validate_features is True
        assert config.domain is None
        assert config.backend == "copilot"
        assert config.enable_text_features is True

    def test_custom(self):
        config = SemanticEngineConfig(
            model="gpt-4o",
            max_suggestions=10,
            domain="finance",
            backend="openai",
            api_key="sk-test",
        )
        assert config.model == "gpt-4o"
        assert config.max_suggestions == 10
        assert config.domain == "finance"
        assert config.backend == "openai"


class TestSemanticEngine:
    """Tests for SemanticEngine."""

    def test_creation_copilot(self):
        engine = SemanticEngine(backend="copilot")
        assert engine.config.backend == "copilot"

    def test_creation_openai(self):
        engine = SemanticEngine(backend="openai", api_key="sk-test")
        assert engine.config.backend == "openai"

    def test_creation_litellm(self):
        engine = SemanticEngine(backend="litellm")
        assert engine.config.backend == "litellm"

    def test_fit_with_mock_client(self, mock_client, sample_df):
        engine = SemanticEngine(validate_features=False, verbose=True)
        engine._client = mock_client

        engine.fit(sample_df, task_description="predict outcome")

        assert engine._is_fitted is True
        mock_client.suggest_features.assert_called()
        assert len(engine._suggested_features) == 2

    def test_transform_with_prebuilt_features(self, sample_df):
        engine = SemanticEngine(validate_features=False)
        engine._is_fitted = True
        engine._text_features = []
        engine._text_columns = []
        engine._suggested_features = [
            {
                "name": "age_doubled",
                "code": "result = df['age'] * 2",
                "explanation": "Age times two",
                "source_columns": ["age"],
            }
        ]

        result = engine.transform(sample_df)

        assert "age_doubled" in result.columns
        assert (result["age_doubled"] == sample_df["age"] * 2).all()

    def test_transform_not_fitted_raises(self, sample_df):
        engine = SemanticEngine()
        with pytest.raises(RuntimeError, match="fitted"):
            engine.transform(sample_df)

    def test_get_feature_set(self, mock_client, sample_df):
        engine = SemanticEngine(validate_features=False)
        engine._client = mock_client
        engine.fit(sample_df, task_description="test")

        fs = engine.get_feature_set()

        assert isinstance(fs, FeatureSet)
        assert len(fs) >= 2

    def test_get_feature_explanations(self, mock_client, sample_df):
        engine = SemanticEngine(validate_features=False)
        engine._client = mock_client
        engine.fit(sample_df, task_description="test")

        explanations = engine.get_feature_explanations()

        assert isinstance(explanations, dict)
        assert "age_income_ratio" in explanations

    def test_get_feature_code(self, mock_client, sample_df):
        engine = SemanticEngine(validate_features=False)
        engine._client = mock_client
        engine.fit(sample_df, task_description="test")

        code = engine.get_feature_code()

        assert isinstance(code, dict)
        assert "age_income_ratio" in code
        assert "result" in code["age_income_ratio"]

    def test_standardize_categories(self, mock_client):
        engine = SemanticEngine()
        engine._client = mock_client
        mock_client.send_prompt.return_value = (
            '{"mapping": {"software eng": "Software Engineer", "swe": "Software Engineer"}}'
        )

        df = pd.DataFrame({"job": ["software eng", "swe", "Data Scientist", "Data Scientist"]})

        mapping = engine.standardize_categories(df, "job", context="tech job titles")

        assert isinstance(mapping, dict)

    def test_standardize_categories_missing_column(self):
        engine = SemanticEngine()
        df = pd.DataFrame({"a": [1, 2]})

        with pytest.raises(ValueError, match="not found"):
            engine.standardize_categories(df, "nonexistent")

    def test_apply_category_mapping(self):
        engine = SemanticEngine()
        df = pd.DataFrame({"job": ["swe", "ds", "swe", "pm"]})
        mapping = {"swe": "Software Engineer", "ds": "Data Scientist"}

        result = engine.apply_category_mapping(df, "job", mapping)

        assert result["job"].tolist() == ["Software Engineer", "Data Scientist", "Software Engineer", "pm"]
        # Original df unchanged (not inplace)
        assert df["job"].iloc[0] == "swe"

    def test_apply_category_mapping_inplace(self):
        engine = SemanticEngine()
        df = pd.DataFrame({"job": ["swe", "ds"]})
        mapping = {"swe": "Software Engineer"}

        result = engine.apply_category_mapping(df, "job", mapping, inplace=True)

        assert result is df
        assert df["job"].iloc[0] == "Software Engineer"

    def test_apply_category_mapping_missing_column(self):
        engine = SemanticEngine()
        df = pd.DataFrame({"a": [1]})

        with pytest.raises(ValueError, match="not found"):
            engine.apply_category_mapping(df, "missing", {})


# ===========================================================================
# 6. TransformRuleGenerator
# ===========================================================================


class TestTransformRuleGenerator:
    """Tests for TransformRuleGenerator."""

    def test_init_defaults(self):
        gen = TransformRuleGenerator()
        assert gen.model == "gpt-5.2"
        assert gen.validate is True
        assert gen.backend == "copilot"
        assert gen.store is None

    def test_build_generation_prompt(self):
        gen = TransformRuleGenerator()
        prompt = gen._build_generation_prompt("Calculate ratio", {"price": "float", "qty": "int"})

        assert "Calculate ratio" in prompt
        assert "price (float)" in prompt
        assert "qty (int)" in prompt
        assert "result" in prompt
        assert "JSON" in prompt

    def test_parse_rule_response_valid_json(self):
        gen = TransformRuleGenerator()
        response = (
            '{"name": "price_ratio", "code": "result = df[\'price\'] / (df[\'qty\'] + 1e-8)", '
            '"input_columns": ["price", "qty"], "output_type": "numeric", '
            '"column_patterns": [".*price.*"], "explanation": "Ratio of price to qty"}'
        )

        rule = gen._parse_rule_response(response, "ratio calc", {"price": "float", "qty": "int"}, tags=["test"])

        assert isinstance(rule, TransformRule)
        assert rule.name == "price_ratio"
        assert "result" in rule.code
        assert rule.tags == ["test"]

    def test_parse_rule_response_with_markdown(self):
        gen = TransformRuleGenerator()
        response = (
            "```json\n"
            '{"name": "test_rule", "code": "result = df[\'a\'] + df[\'b\']", '
            '"input_columns": ["a", "b"], "output_type": "numeric", "explanation": "sum"}\n'
            "```"
        )

        rule = gen._parse_rule_response(response, "sum", {"a": "int", "b": "int"})
        assert isinstance(rule, TransformRule)

    def test_parse_rule_response_invalid_json_fallback(self):
        gen = TransformRuleGenerator()
        response = "Here is some code:\nresult = df['x'] * 2\nEnjoy!"

        rule = gen._parse_rule_response(response, "double x", {"x": "float"})

        assert isinstance(rule, TransformRule)
        assert "result" in rule.code

    def test_validate_and_fix(self, mock_client, sample_df):
        gen = TransformRuleGenerator(validate=True)
        gen._client = mock_client

        rule = TransformRule(
            name="test",
            description="test",
            code="result = df['age'] / df['income']",
            input_columns=["age", "income"],
        )

        result = gen._validate_and_fix(rule, sample_df)
        assert isinstance(result, TransformRule)
        mock_client.validate_feature_code.assert_called()

    def test_validate_and_fix_with_failure(self, mock_client, sample_df):
        mock_client.validate_feature_code.side_effect = [
            {"valid": False, "error": "division by zero", "warnings": []},
            {"valid": True, "error": None, "warnings": []},
        ]

        gen = TransformRuleGenerator(validate=True, verbose=True)
        gen._client = mock_client

        rule = TransformRule(
            name="test",
            description="test",
            code="result = df['age'] / df['income']",
            input_columns=["age", "income"],
        )

        result = gen._validate_and_fix(rule, sample_df)
        assert isinstance(result, TransformRule)
        assert mock_client.validate_feature_code.call_count == 2

    def test_clean_code_removes_markdown(self):
        gen = TransformRuleGenerator()
        code = "```python\nresult = df['a'] + 1\n```"
        cleaned = gen._clean_code(code)
        assert "```" not in cleaned
        assert "result" in cleaned

    def test_clean_code_ensures_result(self):
        gen = TransformRuleGenerator()
        code = "output = df['a'] * 2"
        cleaned = gen._clean_code(code)
        assert cleaned.startswith("result =")

    def test_clean_code_raw_expression(self):
        gen = TransformRuleGenerator()
        code = "df['a'] + df['b']"
        cleaned = gen._clean_code(code)
        assert cleaned.startswith("result = ")

    def test_fix_common_issues_division(self):
        gen = TransformRuleGenerator()
        code = "result = df['a'] / df['b']"
        fixed = gen._fix_common_issues(code, "division by zero")
        assert "1e-8" in fixed

    def test_fix_common_issues_syntax(self):
        gen = TransformRuleGenerator()
        code = "result = df['col'] + 1"
        fixed = gen._fix_common_issues(code, "SyntaxError: invalid syntax")
        # Method attempts quote normalization; code should be returned intact for valid quotes
        assert "result" in fixed
        assert "df" in fixed

    def test_suggest_rules_no_store(self):
        gen = TransformRuleGenerator(store=None)
        result = gen.suggest_rules({"a": "int"})
        assert result == []

    def test_save_rule_no_store_raises(self):
        gen = TransformRuleGenerator(store=None)
        rule = TransformRule(name="r", description="d", code="result = 1", input_columns=["a"])

        with pytest.raises(ValueError, match="No rule store"):
            gen.save_rule(rule)

    def test_generate_and_suggest_no_existing(self, mock_client):
        gen = TransformRuleGenerator(store=None, validate=False)
        gen._client = mock_client
        mock_client.send_prompt.return_value = (
            '{"name": "new_rule", "code": "result = df[\'a\'] + 1", '
            '"input_columns": ["a"], "output_type": "numeric", "explanation": "add 1"}'
        )

        new_rule, existing = gen.generate_and_suggest("add 1", {"a": "int"})

        assert new_rule is not None
        assert existing == []
        assert isinstance(new_rule, TransformRule)


# ===========================================================================
# 7. Additional Coverage — CopilotFeatureClient
# ===========================================================================


class TestCopilotClientParseSuggestions:
    """Tests for CopilotFeatureClient._parse_suggestions edge cases."""

    def _client(self):
        return CopilotFeatureClient()

    def test_parse_valid_json(self):
        client = self._client()
        response = '{"features": [{"name": "f1", "code": "result = 1"}]}'
        result = client._parse_suggestions(response)
        assert len(result) == 1
        assert result[0]["name"] == "f1"

    def test_parse_markdown_wrapped_json(self):
        client = self._client()
        response = '```json\n{"features": [{"name": "f1", "code": "result = 1"}]}\n```'
        result = client._parse_suggestions(response)
        assert len(result) == 1
        assert result[0]["name"] == "f1"

    def test_parse_invalid_json_returns_empty(self):
        client = self._client()
        result = client._parse_suggestions("this is not json at all!!!")
        assert result == []

    def test_parse_json_embedded_in_text(self):
        client = self._client()
        response = 'Here are suggestions:\n{"features": [{"name": "f2", "code": "result = 2"}]}\nEnd.'
        result = client._parse_suggestions(response)
        assert len(result) == 1
        assert result[0]["name"] == "f2"

    def test_parse_empty_features_key(self):
        client = self._client()
        response = '{"features": []}'
        result = client._parse_suggestions(response)
        assert result == []

    def test_parse_no_features_key(self):
        client = self._client()
        response = '{"other_key": "value"}'
        result = client._parse_suggestions(response)
        assert result == []

    def test_parse_nested_json_extraction_fallback_invalid(self):
        """Embedded JSON that itself is invalid triggers the inner except."""
        client = self._client()
        response = "prefix {invalid json!} suffix"
        result = client._parse_suggestions(response)
        assert result == []


class TestCopilotClientBuildSuggestionPrompt:
    """Tests for CopilotFeatureClient._build_suggestion_prompt."""

    def _client(self):
        return CopilotFeatureClient()

    def test_basic_prompt(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"age": "int", "income": "float"},
            task_description="predict churn",
        )
        assert "predict churn" in prompt
        assert "age (int)" in prompt
        assert "income (float)" in prompt

    def test_prompt_with_domain(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"age": "int"},
            task_description="test",
            domain="healthcare",
        )
        assert "healthcare" in prompt

    def test_prompt_with_column_descriptions(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"age": "int"},
            task_description="test",
            column_descriptions={"age": "Patient age in years"},
        )
        assert "Patient age in years" in prompt

    def test_prompt_max_suggestions(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"x": "float"},
            task_description="test",
            max_suggestions=5,
        )
        assert "5" in prompt


class TestCopilotClientMockResponse:
    """Tests for CopilotFeatureClient._mock_response branches."""

    def _client(self):
        return CopilotFeatureClient()

    def test_mock_response_with_columns_suggest(self):
        client = self._client()
        prompt = "Please suggest features:\n- age (int): age\n- income (float): inc"
        result = client._mock_response(prompt)
        assert "features" in result
        import json

        data = json.loads(result)
        assert len(data["features"]) >= 2

    def test_mock_response_with_three_columns(self):
        client = self._client()
        prompt = "suggest features:\n- a (int): x\n- b (float): y\n- c (str): z"
        result = client._mock_response(prompt)
        import json

        data = json.loads(result)
        assert len(data["features"]) >= 3  # ratio, product, normalized, zscore

    def test_mock_response_suggest_no_columns(self):
        client = self._client()
        result = client._mock_response("suggest some features for the dataset")
        import json

        data = json.loads(result)
        assert "features" in data

    def test_mock_response_explain(self):
        client = self._client()
        result = client._mock_response("explain this concept in detail")
        assert "relationship" in result.lower()

    def test_mock_response_code(self):
        client = self._client()
        result = client._mock_response("generate code for this transformation only")
        assert "result" in result

    def test_mock_response_other(self):
        client = self._client()
        result = client._mock_response("hello world")
        assert result.startswith("Mock response for:")

    def test_mock_response_single_column(self):
        client = self._client()
        prompt = "suggest feature:\n- age (int): patient age"
        result = client._mock_response(prompt)
        import json

        data = json.loads(result)
        names = [f["name"] for f in data["features"]]
        assert any("zscore" in n for n in names)


class TestCopilotClientSendPromptMock:
    """Tests for CopilotFeatureClient.send_prompt in mock mode."""

    def test_send_prompt_auto_starts(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        # Do NOT call start() — send_prompt should auto-start
        result = client.send_prompt("explain this feature")
        assert isinstance(result, str)
        assert len(result) > 0
        client.stop()

    def test_send_prompt_mock_mode(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()
        result = client.send_prompt("hello world")
        # Should return a non-empty string (either mock or real response)
        assert isinstance(result, str)
        assert len(result) > 0
        client.stop()


class TestCopilotClientExplainFeatureWithDescriptions:
    """Tests for CopilotFeatureClient.explain_feature with optional params."""

    def test_explain_feature_with_column_descriptions(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()
        result = client.explain_feature(
            feature_name="bmi",
            feature_code="result = df['weight'] / (df['height'] ** 2)",
            column_descriptions={"weight": "in kg", "height": "in meters"},
            task_description="predict diabetes",
        )
        assert isinstance(result, str)
        assert len(result) > 0
        client.stop()

    def test_explain_feature_no_optional_params(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()
        result = client.explain_feature(
            feature_name="ratio",
            feature_code="result = df['a'] / df['b']",
        )
        assert isinstance(result, str)
        client.stop()


class TestCopilotClientGenerateCodeWithConstraints:
    """Tests for CopilotFeatureClient.generate_feature_code with constraints."""

    def test_generate_code_with_constraints(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()
        code = client.generate_feature_code(
            description="compute ratio",
            column_info={"a": "float", "b": "float"},
            constraints=["handle division by zero", "no NaN output"],
        )
        assert isinstance(code, str)
        assert len(code) > 0
        client.stop()


class TestCopilotClientValidateEdgeCases:
    """Additional validate_feature_code tests for CopilotFeatureClient."""

    def test_validate_no_result_assignment(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()
        result = client.validate_feature_code(
            code="x = 42",
            sample_data={"a": [1, 2]},
        )
        assert result["valid"] is True
        assert any("result" in w for w in result["warnings"])
        client.stop()

    def test_validate_without_sample_data(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")
        client.start()
        result = client.validate_feature_code(code="result = 1 + 2")
        assert result["valid"] is True
        assert result["error"] is None
        client.stop()


class TestSyncCopilotRunInNewLoop:
    """Tests for SyncCopilotFeatureClient._run_in_new_loop."""

    def test_run_in_new_loop_directly(self):
        client = SyncCopilotFeatureClient(model="gpt-5.2")

        async def coro():
            return 42

        result = client._run_in_new_loop(coro())
        assert result == 42


# ===========================================================================
# 8. Additional Coverage — OpenAIFeatureClient
# ===========================================================================


class TestOpenAIClientParseSuggestions:
    """Tests for OpenAIFeatureClient._parse_suggestions edge cases."""

    def _client(self):
        return OpenAIFeatureClient()

    def test_parse_valid_json(self):
        client = self._client()
        response = '{"features": [{"name": "f1", "code": "result = 1"}]}'
        result = client._parse_suggestions(response)
        assert len(result) == 1

    def test_parse_markdown_wrapped(self):
        client = self._client()
        response = '```json\n{"features": [{"name": "f1", "code": "result = 1"}]}\n```'
        result = client._parse_suggestions(response)
        assert len(result) == 1

    def test_parse_invalid_json(self):
        client = self._client()
        result = client._parse_suggestions("not json content!!! $$")
        assert result == []

    def test_parse_json_in_text(self):
        client = self._client()
        response = 'Some text {"features": [{"name": "x"}]} more text'
        result = client._parse_suggestions(response)
        assert len(result) == 1

    def test_parse_nested_invalid_fallback(self):
        client = self._client()
        result = client._parse_suggestions("prefix {not: valid json!} suffix")
        assert result == []


class TestOpenAIClientBuildSuggestionPrompt:
    """Tests for OpenAIFeatureClient._build_suggestion_prompt."""

    def _client(self):
        return OpenAIFeatureClient()

    def test_basic(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"age": "int"},
            task_description="classify",
        )
        assert "classify" in prompt
        assert "age (int)" in prompt

    def test_with_domain(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"x": "float"},
            task_description="test",
            domain="finance",
        )
        assert "finance" in prompt

    def test_with_column_descriptions(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"age": "int"},
            task_description="test",
            column_descriptions={"age": "Years old"},
        )
        assert "Years old" in prompt


class TestOpenAIClientMockResponse:
    """Tests for OpenAIFeatureClient._mock_response branches."""

    def _client(self):
        return OpenAIFeatureClient()

    def test_mock_with_columns(self):
        client = self._client()
        prompt = "suggest features:\n- age (int): x\n- income (float): y"
        result = client._mock_response(prompt)
        import json

        data = json.loads(result)
        assert len(data["features"]) >= 2

    def test_mock_with_three_columns(self):
        client = self._client()
        prompt = "suggest:\n- a (int): x\n- b (float): y\n- c (str): z"
        result = client._mock_response(prompt)
        import json

        data = json.loads(result)
        assert len(data["features"]) >= 3

    def test_mock_suggest_no_columns(self):
        client = self._client()
        result = client._mock_response("suggest some features")
        import json

        data = json.loads(result)
        assert "features" in data

    def test_mock_explain(self):
        client = self._client()
        result = client._mock_response("explain this concept")
        assert "relationship" in result.lower()

    def test_mock_code(self):
        client = self._client()
        result = client._mock_response("generate code only")
        assert "result" in result

    def test_mock_other(self):
        client = self._client()
        result = client._mock_response("hello")
        assert result.startswith("Mock response for:")

    def test_mock_single_column(self):
        client = self._client()
        prompt = "suggest some data:\n- score (float): test"
        result = client._mock_response(prompt)
        import json

        data = json.loads(result)
        assert any("zscore" in f["name"] for f in data["features"])


class TestSyncOpenAIClientAllMethods:
    """Full mock-mode integration tests for SyncOpenAIFeatureClient."""

    def test_send_prompt_auto_start(self):
        """send_prompt when not started triggers auto-start."""
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        result = client.send_prompt("explain this feature")
        assert isinstance(result, str)
        assert len(result) > 0
        client.stop()

    def test_suggest_features_with_domain(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()
        result = client.suggest_features(
            column_info={"age": "int", "income": "float", "score": "float"},
            task_description="predict churn",
            domain="telecom",
            column_descriptions={"age": "Customer age"},
        )
        assert isinstance(result, list)
        assert len(result) > 0
        client.stop()

    def test_explain_feature_with_all_params(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()
        result = client.explain_feature(
            feature_name="bmi",
            feature_code="result = df['w'] / (df['h'] ** 2)",
            column_descriptions={"w": "weight", "h": "height"},
            task_description="predict outcome",
        )
        assert isinstance(result, str)
        client.stop()

    def test_generate_feature_code_with_constraints(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()
        code = client.generate_feature_code(
            description="compute BMI",
            column_info={"w": "float", "h": "float"},
            constraints=["handle zeros"],
        )
        assert isinstance(code, str)
        client.stop()

    def test_validate_no_result_warning(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()
        result = client.validate_feature_code(
            code="x = 42",
            sample_data={"a": [1, 2, 3]},
        )
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        client.stop()

    def test_validate_runtime_error(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()
        result = client.validate_feature_code(
            code="result = df['missing_col']",
            sample_data={"a": [1, 2]},
        )
        assert result["valid"] is False
        assert "Runtime error" in result["error"]
        client.stop()

    def test_validate_no_sample_data(self):
        client = SyncOpenAIFeatureClient(model="gpt-4o")
        client.start()
        result = client.validate_feature_code(code="result = 42")
        assert result["valid"] is True
        assert result["error"] is None
        client.stop()


# ===========================================================================
# 9. Additional Coverage — LiteLLMFeatureClient
# ===========================================================================


class TestLiteLLMConfig:
    """Tests for LiteLLMConfig."""

    def test_defaults(self):
        config = LiteLLMConfig()
        assert config.model == "gpt-4o"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096
        assert config.timeout == 60.0
        assert config.api_key is None
        assert config.api_base is None

    def test_custom_values(self):
        config = LiteLLMConfig(model="claude-3-opus", api_key="key", api_base="http://localhost:8000")
        assert config.model == "claude-3-opus"
        assert config.api_key == "key"
        assert config.api_base == "http://localhost:8000"


class TestLiteLLMClientParseSuggestions:
    """Tests for LiteLLMFeatureClient._parse_suggestions."""

    def _client(self):
        return LiteLLMFeatureClient()

    def test_parse_valid_json(self):
        client = self._client()
        response = '{"features": [{"name": "f1", "code": "result = 1"}]}'
        result = client._parse_suggestions(response)
        assert len(result) == 1

    def test_parse_markdown_wrapped(self):
        client = self._client()
        response = '```json\n{"features": [{"name": "f1"}]}\n```'
        result = client._parse_suggestions(response)
        assert len(result) == 1

    def test_parse_invalid_json(self):
        client = self._client()
        result = client._parse_suggestions("completely invalid!!!")
        assert result == []

    def test_parse_json_in_text(self):
        client = self._client()
        response = 'Blah {"features": [{"name": "x"}]} blah'
        result = client._parse_suggestions(response)
        assert len(result) == 1

    def test_parse_nested_invalid_fallback(self):
        client = self._client()
        result = client._parse_suggestions("prefix {broken json} suffix")
        assert result == []


class TestLiteLLMClientBuildSuggestionPrompt:
    """Tests for LiteLLMFeatureClient._build_suggestion_prompt."""

    def _client(self):
        return LiteLLMFeatureClient()

    def test_basic(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"x": "float"},
            task_description="classify",
        )
        assert "classify" in prompt
        assert "x (float)" in prompt

    def test_with_domain(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"x": "float"},
            task_description="test",
            domain="retail",
        )
        assert "retail" in prompt

    def test_with_column_descriptions(self):
        client = self._client()
        prompt = client._build_suggestion_prompt(
            column_info={"price": "float"},
            task_description="test",
            column_descriptions={"price": "Product price in USD"},
        )
        assert "Product price in USD" in prompt


class TestLiteLLMClientMockResponse:
    """Tests for LiteLLMFeatureClient._mock_response branches."""

    def _client(self):
        return LiteLLMFeatureClient()

    def test_mock_with_columns(self):
        client = self._client()
        prompt = "suggest features:\n- age (int): x\n- income (float): y"
        import json

        result = client._mock_response(prompt)
        data = json.loads(result)
        assert len(data["features"]) >= 2

    def test_mock_with_three_columns(self):
        client = self._client()
        prompt = "suggest:\n- a (int): x\n- b (float): y\n- c (str): z"
        import json

        result = client._mock_response(prompt)
        data = json.loads(result)
        assert len(data["features"]) >= 3

    def test_mock_suggest_no_columns(self):
        client = self._client()
        import json

        result = client._mock_response("suggest features please")
        data = json.loads(result)
        assert "features" in data

    def test_mock_explain(self):
        client = self._client()
        result = client._mock_response("explain this concept")
        assert "relationship" in result.lower()

    def test_mock_code(self):
        client = self._client()
        result = client._mock_response("generate code only")
        assert "result" in result

    def test_mock_other(self):
        client = self._client()
        result = client._mock_response("hello")
        assert result.startswith("Mock response for:")


class TestSyncLiteLLMFeatureClient:
    """Full mock-mode tests for SyncLiteLLMFeatureClient."""

    def test_creation(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        assert client._async_client is not None

    def test_start_stop(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        assert client._async_client._is_started is True
        client.stop()
        assert client._async_client._is_started is False

    def test_suggest_features_mock(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        result = client.suggest_features(
            column_info={"age": "int", "income": "float"},
            task_description="predict churn",
        )
        assert isinstance(result, list)
        assert len(result) > 0
        client.stop()

    def test_explain_feature_mock(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        result = client.explain_feature(
            feature_name="ratio",
            feature_code="result = df['a'] / df['b']",
        )
        assert isinstance(result, str)
        assert len(result) > 0
        client.stop()

    def test_generate_feature_code_mock(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        code = client.generate_feature_code(
            description="compute ratio",
            column_info={"a": "float", "b": "float"},
        )
        assert isinstance(code, str)
        assert len(code) > 0
        client.stop()

    def test_validate_feature_code_valid(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        result = client.validate_feature_code(
            code="result = df['x'] + 1",
            sample_data={"x": [1, 2, 3]},
        )
        assert result["valid"] is True
        client.stop()

    def test_validate_feature_code_syntax_error(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        result = client.validate_feature_code(code="result = (")
        assert result["valid"] is False
        assert "Syntax error" in result["error"]
        client.stop()

    def test_validate_feature_code_runtime_error(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        result = client.validate_feature_code(
            code="result = df['nonexistent']",
            sample_data={"a": [1, 2]},
        )
        assert result["valid"] is False
        assert "Runtime error" in result["error"]
        client.stop()

    def test_validate_no_result_warning(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        result = client.validate_feature_code(
            code="x = 42",
            sample_data={"a": [1]},
        )
        assert result["valid"] is True
        assert any("result" in w for w in result["warnings"])
        client.stop()

    def test_send_prompt_auto_start(self):
        """explain_feature without calling start() triggers auto-start via send_prompt."""
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        # Do NOT call start — explain_feature calls send_prompt internally which auto-starts
        result = client.explain_feature(
            feature_name="ratio",
            feature_code="result = df['a'] / df['b']",
        )
        assert isinstance(result, str)
        assert len(result) > 0
        client.stop()

    def test_suggest_with_domain_and_descriptions(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        result = client.suggest_features(
            column_info={"age": "int", "income": "float", "score": "float"},
            task_description="test",
            domain="healthcare",
            column_descriptions={"age": "Patient age"},
        )
        assert isinstance(result, list)
        client.stop()

    def test_explain_with_all_params(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        result = client.explain_feature(
            feature_name="bmi",
            feature_code="result = df['w'] / (df['h'] ** 2)",
            column_descriptions={"w": "weight", "h": "height"},
            task_description="predict outcome",
        )
        assert isinstance(result, str)
        client.stop()

    def test_generate_code_with_constraints(self):
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        code = client.generate_feature_code(
            description="compute BMI",
            column_info={"w": "float", "h": "float"},
            constraints=["handle zeros"],
        )
        assert isinstance(code, str)
        client.stop()


# ===========================================================================
# 10. Additional Coverage — SemanticEngine
# ===========================================================================


class TestSemanticEngineExtended:
    """Extended tests for SemanticEngine to increase coverage."""

    def test_fit_transform_with_mocked_client(self, sample_df):
        """Test full fit → transform cycle with mocked client."""
        engine = SemanticEngine(model="gpt-5.2", validate_features=False)
        mock_cl = MagicMock()
        mock_cl.suggest_features.return_value = [
            {
                "name": "age_squared",
                "code": "result = df['age'] ** 2",
                "explanation": "Squared age",
                "source_columns": ["age"],
            }
        ]
        engine._client = mock_cl

        y = pd.Series(np.random.randint(0, 2, len(sample_df)))
        engine.fit(sample_df, y, task_description="classify")

        assert engine._is_fitted is True

        result = engine.transform(sample_df)
        assert "age_squared" in result.columns
        assert (result["age_squared"] == sample_df["age"] ** 2).all()

    def test_transform_feature_fails_gracefully(self, sample_df):
        """Features with broken code should be silently skipped."""
        engine = SemanticEngine(validate_features=False, verbose=True)
        engine._is_fitted = True
        engine._text_features = []
        engine._text_columns = []
        engine._suggested_features = [
            {
                "name": "good_feature",
                "code": "result = df['age'] * 2",
                "explanation": "doubled",
                "source_columns": ["age"],
            },
            {
                "name": "bad_feature",
                "code": "result = df['nonexistent_col'] / 0",
                "explanation": "will fail",
                "source_columns": ["nonexistent_col"],
            },
        ]
        result = engine.transform(sample_df)
        assert "good_feature" in result.columns
        assert "bad_feature" not in result.columns

    def test_transform_feature_no_code_skipped(self, sample_df):
        """Features without code should be skipped."""
        engine = SemanticEngine(validate_features=False)
        engine._is_fitted = True
        engine._text_features = []
        engine._text_columns = []
        engine._suggested_features = [
            {"name": "no_code_feat", "code": "", "explanation": "empty", "source_columns": ["age"]},
        ]
        result = engine.transform(sample_df)
        assert "no_code_feat" not in result.columns

    def test_fit_with_validation(self, sample_df):
        """Test fit with validate_features=True."""
        engine = SemanticEngine(validate_features=True, verbose=True)
        mock_cl = MagicMock()
        mock_cl.suggest_features.return_value = [
            {
                "name": "valid_feat",
                "code": "result = df['age'] + 1",
                "explanation": "test",
                "source_columns": ["age"],
            },
            {
                "name": "invalid_feat",
                "code": "result = df['nonexistent']",
                "explanation": "bad",
                "source_columns": ["nonexistent"],
            },
        ]
        mock_cl.validate_feature_code.side_effect = [
            {"valid": True, "error": None, "warnings": []},
            {"valid": False, "error": "KeyError", "warnings": []},
        ]
        engine._client = mock_cl

        engine.fit(sample_df, task_description="test")
        assert len(engine._suggested_features) == 1
        assert engine._suggested_features[0]["name"] == "valid_feat"

    def test_suggest_more_features(self, sample_df):
        """Test suggest_more_features method."""
        engine = SemanticEngine(validate_features=False)
        mock_cl = MagicMock()
        mock_cl.suggest_features.return_value = [
            {
                "name": "age_income_ratio",
                "code": "result = df['age'] / df['income']",
                "explanation": "Ratio",
                "source_columns": ["age", "income"],
            }
        ]
        engine._client = mock_cl
        engine._column_info = {"age": "integer", "income": "float"}
        engine._column_descriptions = {}
        engine._task_description = "classify"

        suggestions = engine.suggest_more_features("interactions", n_features=3)
        assert isinstance(suggestions, list)
        assert len(suggestions) == 1
        mock_cl.suggest_features.assert_called_once()

    def test_generate_custom_feature(self, sample_df):
        """Test generate_custom_feature method."""
        engine = SemanticEngine()
        mock_cl = MagicMock()
        mock_cl.generate_feature_code.return_value = "result = df['age'] ** 2"
        engine._client = mock_cl
        engine._column_info = {"age": "integer"}

        result = engine.generate_custom_feature("age squared", constraints=["positive values only"])
        assert isinstance(result, dict)
        assert "name" in result
        assert "code" in result
        assert result["code"] == "result = df['age'] ** 2"
        mock_cl.generate_feature_code.assert_called_once()

    def test_generate_custom_feature_name_generation(self):
        """Test name generation from description."""
        engine = SemanticEngine()
        mock_cl = MagicMock()
        mock_cl.generate_feature_code.return_value = "result = 1"
        engine._client = mock_cl
        engine._column_info = {}

        result = engine.generate_custom_feature("calculate-the BMI ratio!")
        name = result["name"]
        # Should be snake_case, alphanumeric + underscore only
        assert all(c.isalnum() or c == "_" for c in name)

    def test_ensure_client_copilot(self):
        """Test _ensure_client creates copilot client by default."""
        engine = SemanticEngine(backend="copilot")
        engine._ensure_client()
        assert engine._client is not None

    def test_ensure_client_openai(self):
        """Test _ensure_client creates openai client."""
        engine = SemanticEngine(backend="openai")
        engine._ensure_client()
        assert engine._client is not None

    def test_ensure_client_litellm(self):
        """Test _ensure_client creates litellm client."""
        engine = SemanticEngine(backend="litellm")
        engine._ensure_client()
        assert engine._client is not None

    def test_ensure_client_only_once(self):
        """_ensure_client should not recreate if already set."""
        engine = SemanticEngine(backend="copilot")
        mock_cl = MagicMock()
        engine._client = mock_cl
        engine._ensure_client()
        assert engine._client is mock_cl

    def test_fit_with_text_columns(self):
        """Test fit with text columns detected."""
        # Create DataFrame with a text column (long strings, high cardinality)
        texts = [f"This is a long text description number {i} with lots of words and content" for i in range(20)]
        df = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 20),
                "description": texts,
            }
        )
        engine = SemanticEngine(validate_features=False, verbose=True, enable_text_features=True)
        mock_cl = MagicMock()
        mock_cl.suggest_features.return_value = []
        engine._client = mock_cl

        engine.fit(df, task_description="classify")

        assert "description" in engine._text_columns
        assert len(engine._text_features) > 0

    def test_transform_text_features(self):
        """Test transform applies text features correctly."""
        engine = SemanticEngine(validate_features=False, keep_text_columns=True)
        engine._is_fitted = True
        engine._text_columns = ["text"]
        engine._suggested_features = []
        engine._text_features = [
            {
                "name": "text_char_length",
                "code": "result = df['text'].fillna('').astype(str).str.len()",
                "explanation": "char length",
                "source_columns": ["text"],
                "is_text_feature": True,
            },
        ]
        df = pd.DataFrame({"text": ["hello", "world!!", "test"], "num": [1, 2, 3]})
        result = engine.transform(df)
        assert "text_char_length" in result.columns
        assert result["text_char_length"].iloc[0] == 5

    def test_transform_drop_text_columns(self):
        """Test transform drops text columns when keep_text_columns=False."""
        engine = SemanticEngine(validate_features=False, keep_text_columns=False, verbose=True)
        engine.config.keep_text_columns = False
        engine._is_fitted = True
        engine._text_columns = ["text"]
        engine._suggested_features = []
        engine._text_features = [
            {
                "name": "text_len",
                "code": "result = df['text'].fillna('').astype(str).str.len()",
                "explanation": "length",
                "source_columns": ["text"],
                "is_text_feature": True,
            },
        ]
        df = pd.DataFrame({"text": ["hello", "world"], "num": [1, 2]})
        result = engine.transform(df)
        assert "text" not in result.columns
        assert "text_len" in result.columns

    def test_transform_text_feature_fails_gracefully(self):
        """Text features with broken code are skipped."""
        engine = SemanticEngine(validate_features=False, verbose=True)
        engine._is_fitted = True
        engine._text_columns = []
        engine._suggested_features = []
        engine._text_features = [
            {
                "name": "broken_text",
                "code": "result = df['nonexistent_text_col']",
                "explanation": "will fail",
                "source_columns": ["text"],
                "is_text_feature": True,
            },
        ]
        df = pd.DataFrame({"num": [1, 2, 3]})
        result = engine.transform(df)
        assert "broken_text" not in result.columns

    def test_transform_text_feature_no_code_skipped(self):
        """Text features without code are skipped."""
        engine = SemanticEngine(validate_features=False)
        engine._is_fitted = True
        engine._text_columns = []
        engine._suggested_features = []
        engine._text_features = [
            {"name": "empty_code", "code": "", "explanation": "no code", "source_columns": [], "is_text_feature": True},
        ]
        df = pd.DataFrame({"a": [1, 2]})
        result = engine.transform(df)
        assert "empty_code" not in result.columns

    def test_fit_suggest_features_exception_handled(self, sample_df):
        """When suggest_features raises, fit still completes."""
        engine = SemanticEngine(validate_features=False, verbose=True)
        mock_cl = MagicMock()
        mock_cl.suggest_features.side_effect = RuntimeError("LLM failed")
        engine._client = mock_cl

        engine.fit(sample_df, task_description="test")
        assert engine._is_fitted is True
        assert engine._suggested_features == []

    def test_fit_with_column_descriptions(self, sample_df):
        """Test fit with column_descriptions parameter."""
        engine = SemanticEngine(validate_features=False)
        mock_cl = MagicMock()
        mock_cl.suggest_features.return_value = []
        engine._client = mock_cl

        engine.fit(
            sample_df,
            task_description="test",
            column_descriptions={"age": "Customer age", "income": "Annual income"},
        )
        assert engine._column_descriptions == {"age": "Customer age", "income": "Annual income"}

    def test_standardize_categories_empty_column(self):
        """Standardize with all-NaN column returns empty mapping."""
        engine = SemanticEngine(verbose=True)
        mock_cl = MagicMock()
        engine._client = mock_cl

        df = pd.DataFrame({"job": [None, None, None]})
        mapping = engine.standardize_categories(df, "job")
        assert mapping == {}

    def test_standardize_categories_many_unique_values(self):
        """Test truncation to max_categories."""
        engine = SemanticEngine(verbose=True)
        mock_cl = MagicMock()
        mock_cl.send_prompt.return_value = '{"mapping": {}}'
        engine._client = mock_cl

        df = pd.DataFrame({"cat": [f"val_{i}" for i in range(100)]})
        mapping = engine.standardize_categories(df, "cat", max_categories=10)
        assert isinstance(mapping, dict)

    def test_standardize_categories_exception_handled(self):
        """When send_prompt raises, return empty mapping."""
        engine = SemanticEngine(verbose=True)
        mock_cl = MagicMock()
        mock_cl.send_prompt.side_effect = RuntimeError("LLM error")
        engine._client = mock_cl

        df = pd.DataFrame({"job": ["swe", "ds", "pm"]})
        mapping = engine.standardize_categories(df, "job")
        assert mapping == {}

    def test_standardize_multiple_columns(self):
        """Test standardize_multiple_columns."""
        engine = SemanticEngine(verbose=True)
        mock_cl = MagicMock()
        mock_cl.send_prompt.return_value = '{"mapping": {"swe": "Software Engineer"}}'
        engine._client = mock_cl

        df = pd.DataFrame(
            {
                "job": ["swe", "ds", "swe"],
                "dept": ["eng", "data", "eng"],
            }
        )
        result_df, all_mappings = engine.standardize_multiple_columns(
            df,
            columns=["job", "dept"],
            contexts={"job": "tech jobs", "dept": "departments"},
        )
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(all_mappings, dict)
        assert "job" in all_mappings
        assert "dept" in all_mappings

    def test_standardize_multiple_columns_missing_col(self):
        """Missing columns in standardize_multiple_columns are skipped."""
        engine = SemanticEngine(verbose=True)
        mock_cl = MagicMock()
        mock_cl.send_prompt.return_value = '{"mapping": {}}'
        engine._client = mock_cl

        df = pd.DataFrame({"job": ["swe", "ds"]})
        result_df, all_mappings = engine.standardize_multiple_columns(
            df,
            columns=["job", "nonexistent"],
        )
        assert "job" in all_mappings
        assert "nonexistent" not in all_mappings

    def test_parse_category_mapping_valid_json(self):
        """Test _parse_category_mapping with valid JSON."""
        engine = SemanticEngine()
        response = '{"mapping": {"swe": "Software Engineer", "ds": "Data Scientist"}}'
        mapping = engine._parse_category_mapping(response, ["swe", "ds", "pm"])
        assert mapping == {"swe": "Software Engineer", "ds": "Data Scientist"}

    def test_parse_category_mapping_groups_format(self):
        """Test _parse_category_mapping with groups format."""
        engine = SemanticEngine()
        response = '{"groups": [{"canonical": "SWE", "members": ["swe", "sw eng"]}]}'
        mapping = engine._parse_category_mapping(response, ["swe", "sw eng"])
        assert "swe" in mapping
        assert mapping["swe"] == "SWE"

    def test_parse_category_mapping_direct_dict(self):
        """Test _parse_category_mapping when response is a flat dict."""
        engine = SemanticEngine()
        response = '{"swe": "Software Engineer"}'
        mapping = engine._parse_category_mapping(response, ["swe"])
        assert mapping["swe"] == "Software Engineer"

    def test_parse_category_mapping_markdown_wrapped(self):
        """Test _parse_category_mapping with markdown code block."""
        engine = SemanticEngine()
        response = '```json\n{"mapping": {"swe": "Software Engineer"}}\n```'
        mapping = engine._parse_category_mapping(response, ["swe"])
        assert mapping == {"swe": "Software Engineer"}

    def test_parse_category_mapping_invalid_json(self):
        """Test _parse_category_mapping with invalid JSON."""
        engine = SemanticEngine(verbose=True)
        mapping = engine._parse_category_mapping("not json at all!!!", ["swe"])
        assert mapping == {}

    def test_parse_category_mapping_case_insensitive_match(self):
        """Test _parse_category_mapping matches case-insensitively."""
        engine = SemanticEngine()
        response = '{"mapping": {"SWE": "Software Engineer"}}'
        mapping = engine._parse_category_mapping(response, ["swe"])
        assert "swe" in mapping

    def test_parse_category_mapping_json_embedded_in_text(self):
        """Test _parse_category_mapping extracts JSON from surrounding text."""
        engine = SemanticEngine()
        response = 'Here is the result:\n{"mapping": {"swe": "SWE"}}\nDone.'
        mapping = engine._parse_category_mapping(response, ["swe"])
        assert mapping == {"swe": "SWE"}

    def test_parse_category_mapping_non_dict_json(self):
        """Test _parse_category_mapping with non-dict top-level JSON."""
        engine = SemanticEngine()
        response = '["swe", "ds"]'
        mapping = engine._parse_category_mapping(response, ["swe", "ds"])
        assert mapping == {}

    def test_build_category_standardization_prompt(self):
        """Test _build_category_standardization_prompt."""
        engine = SemanticEngine()
        prompt = engine._build_category_standardization_prompt(
            column="job",
            unique_values=["swe", "ds"],
            target_categories=["Software Engineer", "Data Scientist"],
            context="tech industry",
        )
        assert "job" in prompt
        assert "swe" in prompt
        assert "Software Engineer" in prompt
        assert "tech industry" in prompt

    def test_build_category_standardization_prompt_no_targets(self):
        """Test prompt without target categories or context."""
        engine = SemanticEngine()
        prompt = engine._build_category_standardization_prompt(
            column="dept",
            unique_values=["eng", "sales"],
        )
        assert "dept" in prompt
        assert "eng" in prompt
        assert "Target Categories" not in prompt

    def test_parse_text_features_valid(self):
        """Test _parse_text_features with valid JSON."""
        engine = SemanticEngine()
        response = '{"features": [{"name": "len", "code": "result = df[\'t\'].str.len()"}]}'
        features = engine._parse_text_features(response, "t")
        assert len(features) == 1
        assert features[0]["source_columns"] == ["t"]
        assert features[0]["is_text_feature"] is True

    def test_parse_text_features_markdown(self):
        """Test _parse_text_features with markdown-wrapped JSON."""
        engine = SemanticEngine()
        response = '```json\n{"features": [{"name": "len", "code": "result = 1"}]}\n```'
        features = engine._parse_text_features(response, "t")
        assert len(features) == 1

    def test_parse_text_features_invalid_json(self):
        """Test _parse_text_features with invalid JSON."""
        engine = SemanticEngine()
        features = engine._parse_text_features("not json!!!", "col")
        assert features == []

    def test_parse_text_features_json_in_text(self):
        """Test _parse_text_features with JSON embedded in text."""
        engine = SemanticEngine()
        response = 'blah {"features": [{"name": "x"}]} blah'
        features = engine._parse_text_features(response, "col")
        assert len(features) == 1
        assert features[0]["source_columns"] == ["col"]

    def test_parse_text_features_nested_invalid(self):
        """_parse_text_features with invalid nested JSON extraction."""
        engine = SemanticEngine()
        features = engine._parse_text_features("prefix {broken!!} suffix", "col")
        assert features == []

    def test_get_fallback_text_features(self):
        """Test _get_fallback_text_features returns correct list."""
        engine = SemanticEngine()
        features = engine._get_fallback_text_features("review")
        assert len(features) == 11
        for f in features:
            assert f["source_columns"] == ["review"]
            assert f["is_text_feature"] is True
            assert "review" in f["name"]

    def test_build_text_feature_prompt(self):
        """Test _build_text_feature_prompt."""
        engine = SemanticEngine()
        engine._task_description = "sentiment analysis"
        prompt = engine._build_text_feature_prompt(
            col="review",
            samples=["Great product!", "Terrible service.", "OK."],
            description="Customer review",
        )
        assert "review" in prompt
        assert "Customer review" in prompt
        assert "sentiment analysis" in prompt

    def test_validate_suggestions(self, sample_df):
        """Test _validate_suggestions filters invalid features."""
        engine = SemanticEngine(verbose=True)
        mock_cl = MagicMock()
        mock_cl.validate_feature_code.side_effect = [
            {"valid": True, "error": None, "warnings": []},
            {"valid": False, "error": "error", "warnings": []},
        ]
        engine._client = mock_cl
        engine._suggested_features = [
            {"name": "good", "code": "result = df['age']", "source_columns": ["age"]},
            {"name": "bad", "code": "result = broken", "source_columns": []},
        ]
        engine._validate_suggestions(sample_df)
        assert len(engine._suggested_features) == 1
        assert engine._suggested_features[0]["name"] == "good"

    def test_validate_suggestions_skips_empty_code(self, sample_df):
        """Features with no code are skipped during validation."""
        engine = SemanticEngine(verbose=True)
        mock_cl = MagicMock()
        engine._client = mock_cl
        engine._suggested_features = [
            {"name": "no_code", "code": "", "source_columns": []},
        ]
        engine._validate_suggestions(sample_df)
        assert engine._suggested_features == []
        mock_cl.validate_feature_code.assert_not_called()

    def test_build_feature_set(self):
        """Test _build_feature_set creates proper Feature objects."""
        engine = SemanticEngine()
        engine._text_features = [
            {
                "name": "text_len",
                "code": "result = df['t'].str.len()",
                "explanation": "length",
                "source_columns": ["t"],
                "is_text_feature": True,
            },
        ]
        engine._suggested_features = [
            {
                "name": "age_sq",
                "code": "result = df['age'] ** 2",
                "explanation": "squared",
                "source_columns": ["age"],
            },
        ]
        engine._build_feature_set()
        fs = engine._feature_set
        assert len(fs) == 2
        names = list(fs._features.keys())
        assert "text_len" in names
        assert "age_sq" in names

    def test_del_cleanup(self):
        """Test __del__ calls stop on client."""
        engine = SemanticEngine()
        mock_cl = MagicMock()
        engine._client = mock_cl
        engine.__del__()
        mock_cl.stop.assert_called_once()

    def test_del_cleanup_no_client(self):
        """Test __del__ with no client does not raise."""
        engine = SemanticEngine()
        engine._client = None
        engine.__del__()  # Should not raise

    def test_del_cleanup_stop_raises(self):
        """Test __del__ handles exception from stop()."""
        engine = SemanticEngine()
        mock_cl = MagicMock()
        mock_cl.stop.side_effect = RuntimeError("stop failed")
        engine._client = mock_cl
        engine.__del__()  # Should not raise

    def test_standardize_categories_no_send_prompt_fallback(self):
        """Test standardize_categories fallback when client has no send_prompt."""
        engine = SemanticEngine()
        mock_cl = MagicMock(spec=["suggest_features", "start", "stop", "validate_feature_code"])
        mock_cl.suggest_features.return_value = [{"mapping": {"swe": "Software Engineer"}}]
        engine._client = mock_cl

        df = pd.DataFrame({"job": ["swe", "ds"]})
        mapping = engine.standardize_categories(df, "job")
        assert isinstance(mapping, dict)

    def test_standardize_categories_no_send_prompt_empty_response(self):
        """Test standardize_categories fallback with empty response list."""
        engine = SemanticEngine()
        mock_cl = MagicMock(spec=["suggest_features", "start", "stop", "validate_feature_code"])
        mock_cl.suggest_features.return_value = []
        engine._client = mock_cl

        df = pd.DataFrame({"job": ["swe", "ds"]})
        mapping = engine.standardize_categories(df, "job")
        assert isinstance(mapping, dict)

    def test_transform_replaces_inf_with_nan(self, sample_df):
        """Transform should replace inf/-inf with NaN."""
        engine = SemanticEngine(validate_features=False)
        engine._is_fitted = True
        engine._text_features = []
        engine._text_columns = []
        engine._suggested_features = [
            {
                "name": "inf_feat",
                "code": "result = df['age'] / 0.0",
                "explanation": "will produce inf",
                "source_columns": ["age"],
            },
        ]
        # This will produce inf but exec inside transform catches it
        # and the inf→nan replacement cleans it up
        result = engine.transform(sample_df)
        # The feature may or may not be created depending on runtime error
        # but inf values in the DataFrame should be replaced with NaN
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()

    def test_fit_only_text_columns(self):
        """Test fit when all columns are text (no non-text for general suggestions)."""
        texts = [f"Long description text content number {i} for testing purposes only" for i in range(20)]
        df = pd.DataFrame({"desc": texts})
        engine = SemanticEngine(validate_features=False, verbose=True)
        mock_cl = MagicMock()
        mock_cl.suggest_features.return_value = []
        engine._client = mock_cl

        engine.fit(df, task_description="classify")
        assert engine._is_fitted is True
        # non_text_column_info is empty so suggest_features for general features
        # shouldn't be called (the text-feature-level call may still happen)
        assert engine._suggested_features == []

    def test_generate_text_features_with_llm_suggestions(self):
        """Test _generate_text_features when LLM returns extra text features."""
        engine = SemanticEngine(validate_features=False, verbose=True)
        engine._text_columns = ["review"]
        engine._column_descriptions = {"review": "Customer review"}
        engine._task_description = "sentiment"

        mock_cl = MagicMock()
        mock_cl.suggest_features.return_value = [
            {"name": "review_sentiment", "code": "result = df['review'].str.len()", "explanation": "sentiment proxy"},
        ]
        engine._client = mock_cl

        texts = pd.DataFrame({"review": ["Great product!", "Bad service", "OK"]})
        features = engine._generate_text_features(texts)

        # Should have fallback features + LLM-suggested ones
        assert len(features) > 11
        llm_feats = [f for f in features if f.get("name") == "review_sentiment"]
        assert len(llm_feats) == 1
        assert llm_feats[0]["source_columns"] == ["review"]
        assert llm_feats[0]["is_text_feature"] is True

    def test_generate_text_features_llm_fails(self):
        """Test _generate_text_features when LLM call fails."""
        engine = SemanticEngine(validate_features=False, verbose=True)
        engine._text_columns = ["review"]
        engine._column_descriptions = {}
        engine._task_description = "classify"

        mock_cl = MagicMock()
        mock_cl.suggest_features.side_effect = RuntimeError("LLM down")
        engine._client = mock_cl

        texts = pd.DataFrame({"review": ["hello world"]})
        features = engine._generate_text_features(texts)

        # Should still have fallback features
        assert len(features) == 11

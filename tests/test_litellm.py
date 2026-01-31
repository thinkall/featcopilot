"""Tests for LiteLLM client."""

import pytest

from featcopilot.llm.litellm_client import (
    LiteLLMConfig,
    LiteLLMFeatureClient,
    SyncLiteLLMFeatureClient,
)


class TestLiteLLMConfig:
    """Tests for LiteLLMConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LiteLLMConfig()

        assert config.model == "gpt-4o"
        assert config.temperature == 0.3
        assert config.max_tokens == 4096
        assert config.timeout == 60.0
        assert config.api_key is None
        assert config.api_base is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LiteLLMConfig(
            model="claude-3-opus",
            temperature=0.5,
            max_tokens=8192,
            timeout=120.0,
            api_key="test-key",
            api_base="https://custom.api.com",
        )

        assert config.model == "claude-3-opus"
        assert config.temperature == 0.5
        assert config.max_tokens == 8192
        assert config.timeout == 120.0
        assert config.api_key == "test-key"
        assert config.api_base == "https://custom.api.com"


class TestSyncLiteLLMFeatureClient:
    """Tests for SyncLiteLLMFeatureClient (mock mode)."""

    @pytest.fixture
    def client(self):
        """Create client in mock mode."""
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        client.start()
        return client

    @pytest.fixture
    def sample_columns(self):
        """Sample column info for testing."""
        return {
            "age": "integer",
            "income": "float",
            "education_years": "integer",
            "job_tenure": "float",
        }

    def test_client_start_stop(self, client):
        """Test client start and stop."""
        # Client should be started by fixture
        assert client._async_client._is_started

        # Stop client
        client.stop()
        assert not client._async_client._is_started

    def test_suggest_features_mock(self, client, sample_columns):
        """Test feature suggestions in mock mode."""
        suggestions = client.suggest_features(
            column_info=sample_columns,
            task_description="Predict customer churn",
            max_suggestions=5,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        # Check suggestion structure
        for suggestion in suggestions:
            assert "name" in suggestion
            assert "code" in suggestion
            assert "explanation" in suggestion

    def test_suggest_features_with_descriptions(self, client, sample_columns):
        """Test feature suggestions with column descriptions."""
        column_descriptions = {
            "age": "Customer age in years",
            "income": "Annual income in USD",
            "education_years": "Years of formal education",
            "job_tenure": "Years at current job",
        }

        suggestions = client.suggest_features(
            column_info=sample_columns,
            task_description="Predict loan default",
            column_descriptions=column_descriptions,
            domain="finance",
            max_suggestions=5,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_validate_feature_code_valid(self, client):
        """Test validation of valid feature code."""
        code = "result = df['age'] / (df['income'] + 1e-8)"
        sample_data = {"age": [25, 30, 35], "income": [50000, 60000, 70000]}

        result = client.validate_feature_code(code, sample_data)

        assert result["valid"] is True
        assert result["error"] is None

    def test_validate_feature_code_syntax_error(self, client):
        """Test validation of code with syntax error."""
        code = "result = df['age' / df['income']"  # Missing bracket

        result = client.validate_feature_code(code, None)

        assert result["valid"] is False
        assert "Syntax error" in result["error"]

    def test_validate_feature_code_runtime_error(self, client):
        """Test validation of code with runtime error."""
        code = "result = df['nonexistent_column']"
        sample_data = {"age": [25, 30, 35], "income": [50000, 60000, 70000]}

        result = client.validate_feature_code(code, sample_data)

        assert result["valid"] is False
        assert "Runtime error" in result["error"]


class TestLiteLLMFeatureClientAsync:
    """Tests for async LiteLLMFeatureClient."""

    @pytest.fixture
    def column_info(self):
        """Sample column info."""
        return {
            "age": "integer",
            "income": "float",
            "tenure": "integer",
        }

    @pytest.mark.asyncio
    async def test_async_client_start_stop(self):
        """Test async client start and stop."""
        client = LiteLLMFeatureClient(model="gpt-4o")

        await client.start()
        assert client._is_started

        await client.stop()
        assert not client._is_started

    @pytest.mark.asyncio
    async def test_async_suggest_features(self, column_info):
        """Test async feature suggestions."""
        client = LiteLLMFeatureClient(model="gpt-4o")
        await client.start()

        suggestions = await client.suggest_features(
            column_info=column_info,
            task_description="Predict churn",
            max_suggestions=3,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        await client.stop()

    @pytest.mark.asyncio
    async def test_async_explain_feature(self, column_info):
        """Test async feature explanation."""
        client = LiteLLMFeatureClient(model="gpt-4o")
        await client.start()

        explanation = await client.explain_feature(
            feature_name="age_income_ratio",
            feature_code="result = df['age'] / df['income']",
            column_descriptions={"age": "Customer age", "income": "Annual income"},
            task_description="Predict churn",
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0

        await client.stop()

    @pytest.mark.asyncio
    async def test_async_generate_feature_code(self, column_info):
        """Test async code generation."""
        client = LiteLLMFeatureClient(model="gpt-4o")
        await client.start()

        code = await client.generate_feature_code(
            description="Calculate the ratio of age to income",
            column_info=column_info,
        )

        assert isinstance(code, str)
        assert len(code) > 0

        await client.stop()


class TestLiteLLMModels:
    """Tests for different model configurations."""

    def test_openai_model_config(self):
        """Test OpenAI model configuration."""
        client = SyncLiteLLMFeatureClient(model="gpt-4o")
        assert client._async_client.config.model == "gpt-4o"

    def test_anthropic_model_config(self):
        """Test Anthropic model configuration."""
        client = SyncLiteLLMFeatureClient(model="claude-3-opus")
        assert client._async_client.config.model == "claude-3-opus"

    def test_azure_model_config(self):
        """Test Azure model configuration."""
        client = SyncLiteLLMFeatureClient(model="azure/my-deployment")
        assert client._async_client.config.model == "azure/my-deployment"

    def test_ollama_model_config(self):
        """Test Ollama model configuration."""
        client = SyncLiteLLMFeatureClient(
            model="ollama/llama2",
            api_base="http://localhost:11434",
        )
        assert client._async_client.config.model == "ollama/llama2"
        assert client._async_client.config.api_base == "http://localhost:11434"

    def test_custom_api_base(self):
        """Test custom API base configuration."""
        client = SyncLiteLLMFeatureClient(
            model="gpt-4",
            api_base="https://my-proxy.example.com/v1",
        )
        assert client._async_client.config.api_base == "https://my-proxy.example.com/v1"

"""LiteLLM client wrapper for feature engineering.

Provides a unified interface to 100+ LLM providers through LiteLLM,
enabling flexible model selection without vendor lock-in.
"""

import asyncio
import json
from collections.abc import Coroutine
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from featcopilot.llm._execution import validate_feature_code
from featcopilot.llm._types import LiteLLMProvider, feature_suggestions
from featcopilot.utils.logger import get_logger
from featcopilot.utils.models import DEFAULT_MODEL

logger = get_logger(__name__)
_T = TypeVar("_T")


class LiteLLMConfig(BaseModel):
    """Configuration for LiteLLM client."""

    model: str = Field(default=DEFAULT_MODEL, description="Model identifier")
    temperature: float = Field(default=0.3, ge=0, le=2, description="Temperature for generation")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens in response")
    timeout: float = Field(default=60.0, gt=0, description="Timeout in seconds")
    api_key: str | None = Field(default=None, description="API key (uses env var if not provided)")
    api_base: str | None = Field(default=None, description="Custom API base URL")


class LiteLLMFeatureClient:
    """
    LiteLLM client wrapper for feature engineering.

    Provides a unified interface to 100+ LLM providers through LiteLLM,
    supporting OpenAI, Anthropic, Azure, Google, Cohere, and many more.

    Parameters
    ----------
    config : LiteLLMConfig, optional
        Configuration for the client
    model : str, optional
        Model to use for generation. Defaults to the shared DEFAULT_MODEL;
        specify a provider-supported identifier when using another backend.
    api_key : str, optional
        API key for the provider (uses environment variable if not provided)
    api_base : str, optional
        Custom API base URL for self-hosted models

    Examples
    --------
    >>> client = LiteLLMFeatureClient(model='gpt-4o')
    >>> await client.start()
    >>> suggestions = await client.suggest_features(
    ...     column_info={'age': 'int', 'income': 'float'},
    ...     task='predict churn'
    ... )
    >>> await client.stop()

    Notes
    -----
    Supported model prefixes:
    - OpenAI: gpt-4, gpt-4o, gpt-3.5-turbo
    - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
    - Azure: azure/deployment-name
    - Google: gemini-pro, gemini-ultra
    - AWS Bedrock: bedrock/model-id
    - Ollama: ollama/llama2, ollama/mistral
    - And many more...
    """

    def __init__(
        self,
        config: LiteLLMConfig | None = None,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ):
        self.config = config or LiteLLMConfig(model=model, api_key=api_key, api_base=api_base, **kwargs)
        self._is_started = False
        self._litellm_available = False
        self._litellm: LiteLLMProvider | None = None

    async def start(self) -> "LiteLLMFeatureClient":
        """
        Start the LiteLLM client.

        Returns
        -------
        self : LiteLLMFeatureClient
        """
        try:
            import litellm

            self._litellm = litellm
            self._litellm_available = True
            self._is_started = True

            logger.info(f"LiteLLM client started with model: {self.config.model}")

        except ImportError:
            self._litellm_available = False
            self._is_started = False
            logger.error("LiteLLM is unavailable. Install featcopilot[litellm].")
            raise

        return self

    async def stop(self) -> None:
        """Stop the LiteLLM client."""
        self._is_started = False
        self._litellm_available = False
        self._litellm = None

    async def send_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Send a prompt and get a response.

        Parameters
        ----------
        prompt : str
            The prompt to send
        system_prompt : str, optional
            System prompt for the model

        Returns
        -------
        response : str
            The model's response
        """
        if not self._is_started:
            await self.start()

        if not self._litellm_available or self._litellm is None:
            raise RuntimeError("LiteLLM client is unavailable")

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Build kwargs for litellm
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout": self.config.timeout,
            }

            if self.config.api_base:
                kwargs["api_base"] = self.config.api_base
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key

            # Use async completion
            response = await self._litellm.acompletion(**kwargs)

            content = response.choices[0].message.content if response.choices else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("LiteLLM returned an empty or non-text response")
            return content

        except Exception as e:
            logger.error(f"LiteLLM request failed: {e}")
            raise

    def _mock_response(self, prompt: str) -> str:
        """Generate synthetic responses for explicit test doubles only."""
        import re

        columns = re.findall(r"- (\w+) \(", prompt)

        if ("suggest" in prompt.lower() or "feature" in prompt.lower()) and columns:
            features = []
            if len(columns) >= 2:
                col1, col2 = columns[0], columns[1]
                features.append(
                    {
                        "name": f"{col1}_{col2}_ratio",
                        "code": f"result = df['{col1}'] / (df['{col2}'] + 1e-8)",
                        "explanation": f"Ratio of {col1} to {col2}, captures relative relationship",
                        "source_columns": [col1, col2],
                    }
                )
                features.append(
                    {
                        "name": f"{col1}_{col2}_product",
                        "code": f"result = df['{col1}'] * df['{col2}']",
                        "explanation": f"Interaction between {col1} and {col2}",
                        "source_columns": [col1, col2],
                    }
                )
            if len(columns) >= 3:
                col3 = columns[2]
                features.append(
                    {
                        "name": f"{columns[0]}_normalized_by_{col3}",
                        "code": f"result = (df['{columns[0]}'] - df['{columns[0]}'].mean()) / (df['{col3}'] + 1e-8)",
                        "explanation": f"Normalized {columns[0]} adjusted by {col3}",
                        "source_columns": [columns[0], col3],
                    }
                )
            if len(columns) >= 1:
                features.append(
                    {
                        "name": f"{columns[0]}_zscore",
                        "code": f"result = (df['{columns[0]}'] - df['{columns[0]}'].mean()) / (df['{columns[0]}'].std() + 1e-8)",
                        "explanation": f"Z-score normalization of {columns[0]}",
                        "source_columns": [columns[0]],
                    }
                )
            return json.dumps({"features": features})
        elif "suggest" in prompt.lower() or "feature" in prompt.lower():
            return json.dumps(
                {
                    "features": [
                        {
                            "name": "feature_interaction",
                            "code": "result = df.iloc[:, 0] * df.iloc[:, 1]",
                            "explanation": "Interaction between first two features",
                        }
                    ]
                }
            )
        elif "explain" in prompt.lower():
            return "This feature captures the relationship between the input variables."
        elif "code" in prompt.lower():
            return "result = df.iloc[:, 0] * df.iloc[:, 1]"
        else:
            return "Mock response for: " + prompt[:100]

    async def suggest_features(
        self,
        column_info: dict[str, str],
        task_description: str,
        column_descriptions: dict[str, str] | None = None,
        domain: str | None = None,
        max_suggestions: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get LLM suggestions for new features.

        Parameters
        ----------
        column_info : dict
            Dictionary mapping column names to data types
        task_description : str
            Description of the ML task
        column_descriptions : dict, optional
            Human-readable descriptions of columns
        domain : str, optional
            Domain context (e.g., 'healthcare', 'finance')
        max_suggestions : int, default=10
            Maximum number of feature suggestions

        Returns
        -------
        suggestions : list
            List of feature suggestions with code and explanations
        """
        prompt = self._build_suggestion_prompt(
            column_info, task_description, column_descriptions, domain, max_suggestions
        )

        system_prompt = (
            "You are an expert data scientist specializing in feature engineering. "
            "Always respond with valid JSON only."
        )

        response = await self.send_prompt(prompt, system_prompt=system_prompt)
        return self._parse_suggestions(response)

    def _build_suggestion_prompt(
        self,
        column_info: dict[str, str],
        task_description: str,
        column_descriptions: dict[str, str] | None = None,
        domain: str | None = None,
        max_suggestions: int = 10,
    ) -> str:
        """Build the prompt for feature suggestions."""
        prompt = f"""Suggest {max_suggestions} new features for the following machine learning task.

## ML Task
{task_description}

## Available Columns
"""
        for col, dtype in column_info.items():
            desc = column_descriptions.get(col, "") if column_descriptions else ""
            prompt += f"- {col} ({dtype}): {desc}\n"

        if domain:
            prompt += f"\n## Domain Context\nThis is a {domain} problem.\n"

        prompt += """
## Requirements
1. Suggest features that would be predictive for this task
2. Provide Python code using pandas (assume df is the DataFrame)
3. Explain why each feature might be useful
4. Consider interactions, ratios, and domain-specific transformations
5. Assign a one-dimensional result to `result`, retaining the input row index
6. Use row-wise transformations only: do not recompute dataset/group statistics,
   quantile bins, ranks, or learned mappings from the inference batch
7. Do not mutate df; np and pd are available, without imports

## Output Format
Return a JSON object with a "features" array, each element having:
- "name": feature name (snake_case)
- "code": Python code to compute the feature (single line, result assigned to variable)
- "explanation": why this feature might be predictive
- "source_columns": list of column names used

Example:
{
  "features": [
    {
      "name": "age_income_ratio",
      "code": "result = df['age'] / (df['income'] + 1)",
      "explanation": "Ratio of age to income may indicate life stage and financial maturity",
      "source_columns": ["age", "income"]
    }
  ]
}

Return ONLY the JSON object, no other text.
"""
        return prompt

    def _parse_suggestions(self, response: str) -> list[dict[str, Any]]:
        """Parse feature suggestions from LLM response."""
        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)
            return feature_suggestions(data)

        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return feature_suggestions(data)
                except json.JSONDecodeError:
                    pass

            logger.warning("LiteLLM returned invalid feature suggestion JSON")
            return []

    async def explain_feature(
        self,
        feature_name: str,
        feature_code: str,
        column_descriptions: dict[str, str] | None = None,
        task_description: str | None = None,
    ) -> str:
        """
        Get a human-readable explanation of a feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature
        feature_code : str
            Code that generates the feature
        column_descriptions : dict, optional
            Descriptions of source columns
        task_description : str, optional
            Description of the ML task

        Returns
        -------
        explanation : str
            Human-readable explanation
        """
        prompt = f"""Explain this feature in simple terms for a business stakeholder:

Feature Name: {feature_name}
Code: {feature_code}
"""
        if column_descriptions:
            prompt += "\nColumn Descriptions:\n"
            for col, desc in column_descriptions.items():
                prompt += f"- {col}: {desc}\n"

        if task_description:
            prompt += f"\nML Task: {task_description}\n"

        prompt += """
Provide a 2-3 sentence explanation of:
1. What this feature represents
2. Why it might be predictive for the task
"""
        return await self.send_prompt(prompt)

    async def generate_feature_code(
        self, description: str, column_info: dict[str, str], constraints: list[str] | None = None
    ) -> str:
        """
        Generate Python code for a described feature.

        Parameters
        ----------
        description : str
            Natural language description of desired feature
        column_info : dict
            Available columns and their types
        constraints : list, optional
            Constraints on the generated code

        Returns
        -------
        code : str
            Python code to generate the feature
        """
        prompt = f"""Generate Python code to create this feature:

Description: {description}

Available Columns:
"""
        for col, dtype in column_info.items():
            prompt += f"- {col} ({dtype})\n"

        if constraints:
            prompt += "\nConstraints:\n"
            for c in constraints:
                prompt += f"- {c}\n"

        prompt += """
Requirements:
1. Use pandas operations (assume df is the DataFrame)
2. Assign the result to a variable called 'result'
3. Handle edge cases (division by zero, missing values)
4. Return ONLY the code, no explanations

Example output:
result = df['col1'] / (df['col2'] + 1e-8)
"""
        response = await self.send_prompt(prompt)

        code = response.strip()
        if "```" in code:
            lines = code.split("\n")
            code_lines = []
            in_code_block = False
            for line in lines:
                if line.startswith("```"):
                    in_code_block = not in_code_block
                elif in_code_block:
                    code_lines.append(line)
            code = "\n".join(code_lines)

        if not code.strip():
            raise ValueError("LiteLLM returned an empty code response")
        return code

    async def validate_feature_code(self, code: str, sample_data: dict[str, list] | None = None) -> dict[str, Any]:
        """
        Validate generated feature code.

        Parameters
        ----------
        code : str
            Feature code to validate
        sample_data : dict, optional
            Sample data for testing

        Returns
        -------
        result : dict
            Validation result with 'valid', 'error', and 'warnings' keys
        """
        return validate_feature_code(code, sample_data)


class SyncLiteLLMFeatureClient:
    """Synchronous wrapper for LiteLLMFeatureClient."""

    def __init__(self, **kwargs) -> None:
        self._async_client = LiteLLMFeatureClient(**kwargs)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Keep pooled provider connections on one event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_async(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run an async coroutine, handling nested event loops (e.g., Jupyter)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return self._get_or_create_loop().run_until_complete(coro)

        try:
            import nest_asyncio
        except ImportError:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()

        nest_asyncio.apply()
        return loop.run_until_complete(coro)

    def start(self) -> "LiteLLMFeatureClient":
        """Start the client."""
        return self._run_async(self._async_client.start())

    def stop(self) -> None:
        """Stop the client."""
        try:
            self._run_async(self._async_client.stop())
        finally:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.close()
                self._loop = None

    def send_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send a prompt and get a response."""
        return self._run_async(self._async_client.send_prompt(prompt, system_prompt=system_prompt))

    def suggest_features(self, **kwargs) -> list[dict[str, Any]]:
        """Get feature suggestions."""
        return self._run_async(self._async_client.suggest_features(**kwargs))

    def explain_feature(self, **kwargs) -> str:
        """Explain a feature."""
        return self._run_async(self._async_client.explain_feature(**kwargs))

    def generate_feature_code(self, **kwargs) -> str:
        """Generate feature code."""
        return self._run_async(self._async_client.generate_feature_code(**kwargs))

    def validate_feature_code(self, code: str, sample_data: dict[str, list] | None = None) -> dict[str, Any]:
        """Validate feature code."""
        return self._run_async(self._async_client.validate_feature_code(code=code, sample_data=sample_data))

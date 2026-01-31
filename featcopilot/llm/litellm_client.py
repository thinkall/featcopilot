"""LiteLLM client wrapper for feature engineering.

Provides a unified interface to 100+ LLM providers through LiteLLM,
enabling flexible model selection without vendor lock-in.
"""

import asyncio
import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class LiteLLMConfig(BaseModel):
    """Configuration for LiteLLM client."""

    model: str = Field(default="gpt-4o", description="Model identifier (e.g., gpt-4o, claude-3-opus)")
    temperature: float = Field(default=0.3, ge=0, le=2, description="Temperature for generation")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    timeout: float = Field(default=60.0, description="Timeout in seconds")
    api_key: Optional[str] = Field(default=None, description="API key (uses env var if not provided)")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")


class LiteLLMFeatureClient:
    """
    LiteLLM client wrapper for feature engineering.

    Provides a unified interface to 100+ LLM providers through LiteLLM,
    supporting OpenAI, Anthropic, Azure, Google, Cohere, and many more.

    Parameters
    ----------
    config : LiteLLMConfig, optional
        Configuration for the client
    model : str, default='gpt-4o'
        Model to use for generation (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-pro')
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
        config: Optional[LiteLLMConfig] = None,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        self.config = config or LiteLLMConfig(model=model, api_key=api_key, api_base=api_base, **kwargs)
        self._is_started = False
        self._litellm_available = False
        self._litellm = None

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

            # Configure litellm settings
            if self.config.api_key:
                # Set API key based on model provider
                model_lower = self.config.model.lower()
                if "gpt" in model_lower or "openai" in model_lower:
                    import os

                    os.environ["OPENAI_API_KEY"] = self.config.api_key
                elif "claude" in model_lower or "anthropic" in model_lower:
                    import os

                    os.environ["ANTHROPIC_API_KEY"] = self.config.api_key

            logger.info(f"LiteLLM client started with model: {self.config.model}")

        except ImportError:
            self._litellm_available = False
            self._is_started = True
            logger.warning("litellm not installed. Using mock LLM responses. Install with: pip install litellm")

        except Exception as e:
            self._litellm_available = False
            self._is_started = True
            logger.warning(f"Could not initialize LiteLLM: {e}. Using mock LLM responses.")

        return self

    async def stop(self) -> None:
        """Stop the LiteLLM client."""
        self._is_started = False

    async def send_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
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

        if not self._litellm_available:
            return self._mock_response(prompt)

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

            # Use async completion
            response = await self._litellm.acompletion(**kwargs)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LiteLLM request failed: {e}")
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response when LiteLLM is unavailable."""
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
        column_descriptions: Optional[dict[str, str]] = None,
        domain: Optional[str] = None,
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
        column_descriptions: Optional[dict[str, str]] = None,
        domain: Optional[str] = None,
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
            return data.get("features", [])

        except json.JSONDecodeError:
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return data.get("features", [])
                except json.JSONDecodeError:
                    pass

            return []

    async def explain_feature(
        self,
        feature_name: str,
        feature_code: str,
        column_descriptions: Optional[dict[str, str]] = None,
        task_description: Optional[str] = None,
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
        self, description: str, column_info: dict[str, str], constraints: Optional[list[str]] = None
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

        return code

    async def validate_feature_code(self, code: str, sample_data: Optional[dict[str, list]] = None) -> dict[str, Any]:
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
        import numpy as np
        import pandas as pd

        result: dict[str, Any] = {"valid": True, "error": None, "warnings": []}

        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            result["valid"] = False
            result["error"] = f"Syntax error: {e}"
            return result

        if sample_data:
            try:
                df = pd.DataFrame(sample_data)
                local_vars: dict[str, Any] = {"df": df, "np": np, "pd": pd}
                exec(
                    code,
                    {
                        "__builtins__": {
                            "len": len,
                            "sum": sum,
                            "max": max,
                            "min": min,
                            "int": int,
                            "float": float,
                            "str": str,
                            "bool": bool,
                            "abs": abs,
                            "round": round,
                            "pow": pow,
                            "range": range,
                            "list": list,
                            "dict": dict,
                            "set": set,
                            "tuple": tuple,
                            "sorted": sorted,
                            "reversed": reversed,
                            "enumerate": enumerate,
                            "zip": zip,
                            "any": any,
                            "all": all,
                            "map": map,
                            "filter": filter,
                            "isinstance": isinstance,
                            "hasattr": hasattr,
                            "getattr": getattr,
                        }
                    },
                    local_vars,
                )

                if "result" not in local_vars:
                    result["warnings"].append("Code does not assign to 'result' variable")

            except Exception as e:
                result["valid"] = False
                result["error"] = f"Runtime error: {e}"

        return result


class SyncLiteLLMFeatureClient:
    """Synchronous wrapper for LiteLLMFeatureClient."""

    def __init__(self, **kwargs):
        self._async_client = LiteLLMFeatureClient(**kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def start(self) -> "LiteLLMFeatureClient":
        """Start the client."""
        return self._get_loop().run_until_complete(self._async_client.start())

    def stop(self) -> None:
        """Stop the client."""
        return self._get_loop().run_until_complete(self._async_client.stop())

    def suggest_features(self, **kwargs) -> list[dict[str, Any]]:
        """Get feature suggestions."""
        return self._get_loop().run_until_complete(self._async_client.suggest_features(**kwargs))

    def explain_feature(self, **kwargs) -> str:
        """Explain a feature."""
        return self._get_loop().run_until_complete(self._async_client.explain_feature(**kwargs))

    def generate_feature_code(self, **kwargs) -> str:
        """Generate feature code."""
        return self._get_loop().run_until_complete(self._async_client.generate_feature_code(**kwargs))

    def validate_feature_code(self, code: str, sample_data: Optional[dict[str, list]] = None) -> dict[str, Any]:
        """Validate feature code."""
        return self._get_loop().run_until_complete(
            self._async_client.validate_feature_code(code=code, sample_data=sample_data)
        )

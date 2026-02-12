"""OpenAI SDK client wrapper for feature engineering.

Provides a direct interface to OpenAI-compatible endpoints using the
official openai Python SDK, ideal for customized internal endpoints.
"""

import asyncio
import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIClientConfig(BaseModel):
    """Configuration for OpenAI SDK client."""

    model: str = Field(default="gpt-4o", description="Model identifier")
    temperature: float = Field(default=0.3, ge=0, le=2, description="Temperature for generation")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    timeout: float = Field(default=60.0, description="Timeout in seconds")
    api_key: Optional[str] = Field(default=None, description="API key (uses OPENAI_API_KEY env var if not provided)")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL for internal endpoints")


class OpenAIFeatureClient:
    """
    OpenAI SDK client wrapper for feature engineering.

    Uses the official openai Python SDK directly, supporting any
    OpenAI-compatible endpoint including customized internal endpoints.

    Parameters
    ----------
    config : OpenAIClientConfig, optional
        Configuration for the client
    model : str, default='gpt-4o'
        Model to use for generation
    api_key : str, optional
        API key (uses OPENAI_API_KEY env var if not provided)
    api_base : str, optional
        Custom API base URL for internal endpoints

    Examples
    --------
    >>> client = OpenAIFeatureClient(model='gpt-4o', api_base='https://internal.endpoint/v1')
    >>> await client.start()
    >>> suggestions = await client.suggest_features(
    ...     column_info={'age': 'int', 'income': 'float'},
    ...     task='predict churn'
    ... )
    >>> await client.stop()
    """

    def __init__(
        self,
        config: Optional[OpenAIClientConfig] = None,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        self.config = config or OpenAIClientConfig(model=model, api_key=api_key, api_base=api_base, **kwargs)
        self._is_started = False
        self._openai_available = False
        self._async_client = None
        self._openai_module = None
        self._use_module_api = True

    async def start(self) -> "OpenAIFeatureClient":
        """
        Start the OpenAI client.

        Returns
        -------
        self : OpenAIFeatureClient
        """
        try:
            import openai

            self._openai_module = openai

            if self.config.api_key or self.config.api_base:
                # Custom config provided — create explicit client instances
                client_kwargs: dict[str, Any] = {}
                if self.config.api_key:
                    client_kwargs["api_key"] = self.config.api_key
                if self.config.api_base:
                    client_kwargs["base_url"] = self.config.api_base
                client_kwargs["timeout"] = self.config.timeout

                self._async_client = openai.AsyncOpenAI(**client_kwargs)
                self._use_module_api = False
            else:
                # No custom config — use module-level API to inherit
                # environment auth (Azure AD, managed identity, etc.)
                self._async_client = None
                self._use_module_api = True

            self._openai_available = True
            self._is_started = True

            logger.info(f"OpenAI client started with model: {self.config.model}")

        except ImportError:
            self._openai_available = False
            self._is_started = True
            logger.warning("openai not installed. Using mock LLM responses. Install with: pip install openai")

        except Exception as e:
            self._openai_available = False
            self._is_started = True
            logger.warning(f"Could not initialize OpenAI client: {e}. Using mock LLM responses.")

        return self

    async def stop(self) -> None:
        """Stop the OpenAI client."""
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

        if not self._openai_available:
            return self._mock_response(prompt)

        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            call_kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            if self._use_module_api:
                # Use module-level API to inherit environment auth
                response = self._openai_module.chat.completions.create(**call_kwargs)
            else:
                response = await self._async_client.chat.completions.create(**call_kwargs)

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response when OpenAI is unavailable."""
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
                        "code": (
                            f"result = (df['{columns[0]}'] - df['{columns[0]}'].mean()) "
                            f"/ (df['{columns[0]}'].std() + 1e-8)"
                        ),
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


class SyncOpenAIFeatureClient:
    """Synchronous wrapper for OpenAIFeatureClient."""

    def __init__(self, **kwargs):
        self._async_client = OpenAIFeatureClient(**kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _run_async(self, coro):
        """Run an async coroutine, handling nested event loops (e.g., Jupyter)."""
        try:
            loop = asyncio.get_running_loop()
            try:
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            except ImportError:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    def start(self) -> "OpenAIFeatureClient":
        """Start the client."""
        return self._run_async(self._async_client.start())

    def stop(self) -> None:
        """Stop the client."""
        return self._run_async(self._async_client.stop())

    def send_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
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

    def validate_feature_code(self, code: str, sample_data: Optional[dict[str, list]] = None) -> dict[str, Any]:
        """Validate feature code."""
        return self._run_async(self._async_client.validate_feature_code(code=code, sample_data=sample_data))

"""GitHub Copilot SDK client wrapper for feature engineering.

Provides a simplified interface to the Copilot SDK specifically
designed for feature engineering tasks.
"""

import asyncio
import json
from typing import Any, Optional

from pydantic import BaseModel, Field


class CopilotConfig(BaseModel):
    """Configuration for Copilot client."""

    model: str = Field(default="gpt-5", description="Model to use")
    temperature: float = Field(default=0.3, ge=0, le=1, description="Temperature for generation")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    timeout: float = Field(default=60.0, description="Timeout in seconds")
    streaming: bool = Field(default=False, description="Enable streaming responses")


class CopilotFeatureClient:
    """
    GitHub Copilot SDK client wrapper for feature engineering.

    Provides high-level methods for:
    - Generating feature suggestions
    - Explaining features
    - Generating feature code
    - Validating features

    Parameters
    ----------
    config : CopilotConfig, optional
        Configuration for the client
    model : str, default='gpt-5'
        Model to use for generation

    Examples
    --------
    >>> client = CopilotFeatureClient(model='gpt-5')
    >>> await client.start()
    >>> suggestions = await client.suggest_features(
    ...     column_info={'age': 'int', 'income': 'float'},
    ...     task='predict churn'
    ... )
    >>> await client.stop()
    """

    def __init__(self, config: Optional[CopilotConfig] = None, model: str = "gpt-5", **kwargs):
        self.config = config or CopilotConfig(model=model, **kwargs)
        self._client = None
        self._session = None
        self._is_started = False
        self._copilot_available = False

    async def start(self) -> "CopilotFeatureClient":
        """
        Start the Copilot client.

        Returns
        -------
        self : CopilotFeatureClient
        """
        try:
            from copilot import CopilotClient

            self._client = CopilotClient()
            await self._client.start()
            self._session = await self._client.create_session(
                {
                    "model": self.config.model,
                    "streaming": self.config.streaming,
                }
            )
            self._is_started = True
            self._copilot_available = True

        except ImportError:
            # Copilot SDK not installed - use mock mode
            self._copilot_available = False
            self._is_started = True
            print("Warning: copilot-sdk not installed. Using mock LLM responses.")

        except Exception as e:
            # Copilot not available - use mock mode
            self._copilot_available = False
            self._is_started = True
            print(f"Warning: Could not connect to Copilot: {e}. Using mock LLM responses.")

        return self

    async def stop(self) -> None:
        """Stop the Copilot client."""
        if self._session and self._copilot_available:
            await self._session.destroy()
        if self._client and self._copilot_available:
            await self._client.stop()
        self._is_started = False

    async def send_prompt(self, prompt: str) -> str:
        """
        Send a prompt and get a response.

        Parameters
        ----------
        prompt : str
            The prompt to send

        Returns
        -------
        response : str
            The model's response
        """
        if not self._is_started:
            await self.start()

        if not self._copilot_available:
            return self._mock_response(prompt)

        # Use asyncio.Event to wait for completion
        done = asyncio.Event()
        response_content = []

        def on_event(event):
            if event.type.value == "assistant.message":
                response_content.append(event.data.content)
            elif event.type.value == "session.idle":
                done.set()

        self._session.on(on_event)
        await self._session.send({"prompt": prompt})

        # Wait with timeout
        try:
            await asyncio.wait_for(done.wait(), timeout=self.config.timeout)
        except asyncio.TimeoutError:
            return "Error: Request timed out"

        return response_content[-1] if response_content else ""

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response when Copilot is unavailable."""
        # Extract column names from prompt if available
        import re

        columns = re.findall(r"- (\w+) \(", prompt)

        if ("suggest" in prompt.lower() or "feature" in prompt.lower()) and columns:
            # Generate context-aware mock features based on actual columns
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
                        "name": f"{col1}_normalized_by_{col3}",
                        "code": f"result = (df['{col1}'] - df['{col1}'].mean()) / (df['{col3}'] + 1e-8)",
                        "explanation": f"Normalized {col1} adjusted by {col3}",
                        "source_columns": [col1, col3],
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

        response = await self.send_prompt(prompt)
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
        prompt = f"""You are an expert data scientist specializing in feature engineering.

TASK: Suggest {max_suggestions} new features for the following machine learning task.

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
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)
            return data.get("features", [])

        except json.JSONDecodeError:
            # Try to extract JSON substring
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

        # Extract code from response
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

        result = {"valid": True, "error": None, "warnings": []}

        # Syntax check
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            result["valid"] = False
            result["error"] = f"Syntax error: {e}"
            return result

        # Runtime check with sample data
        if sample_data:
            try:
                df = pd.DataFrame(sample_data)
                local_vars = {"df": df, "np": np, "pd": pd}
                exec(
                    code,
                    {"__builtins__": {"len": len, "sum": sum, "max": max, "min": min}},
                    local_vars,
                )

                if "result" not in local_vars:
                    result["warnings"].append("Code does not assign to 'result' variable")

            except Exception as e:
                result["valid"] = False
                result["error"] = f"Runtime error: {e}"

        return result


# Synchronous wrapper for non-async contexts
class SyncCopilotFeatureClient:
    """Synchronous wrapper for CopilotFeatureClient."""

    def __init__(self, **kwargs):
        self._async_client = CopilotFeatureClient(**kwargs)
        self._loop = None

    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def start(self):
        return self._get_loop().run_until_complete(self._async_client.start())

    def stop(self):
        return self._get_loop().run_until_complete(self._async_client.stop())

    def suggest_features(self, **kwargs):
        return self._get_loop().run_until_complete(self._async_client.suggest_features(**kwargs))

    def explain_feature(self, **kwargs):
        return self._get_loop().run_until_complete(self._async_client.explain_feature(**kwargs))

    def generate_feature_code(self, **kwargs):
        return self._get_loop().run_until_complete(self._async_client.generate_feature_code(**kwargs))

    def validate_feature_code(self, code: str, sample_data=None):
        return self._get_loop().run_until_complete(
            self._async_client.validate_feature_code(code=code, sample_data=sample_data)
        )

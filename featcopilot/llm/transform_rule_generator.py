"""LLM-powered transform rule generator.

Generates reusable transform rules from natural language descriptions
using GitHub Copilot SDK.
"""

import json
import re
from typing import Any, Literal, Optional

import pandas as pd

from featcopilot.core.transform_rule import TransformRule
from featcopilot.stores.rule_store import TransformRuleStore
from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class TransformRuleGenerator:
    """
    Generate reusable transform rules from natural language descriptions.

    Uses LLM to understand transformation requirements and generate
    reusable Python code that can be applied across different datasets.

    Parameters
    ----------
    model : str, default='gpt-5.2'
        LLM model to use
    store : TransformRuleStore, optional
        Rule store for saving and retrieving rules
    validate : bool, default=True
        Whether to validate generated code
    backend : str, default='copilot'
        LLM backend to use: 'copilot', 'openai', or 'litellm'
    api_key : str, optional
        API key for openai/litellm backend
    api_base : str, optional
        Custom API base URL for openai/litellm backend

    Examples
    --------
    >>> generator = TransformRuleGenerator()
    >>> rule = generator.generate_from_description(
    ...     description="Calculate the ratio of price to quantity",
    ...     columns={"price": "float", "quantity": "int"}
    ... )
    >>> generator.save_rule(rule)
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        store: Optional[TransformRuleStore] = None,
        validate: bool = True,
        verbose: bool = False,
        backend: Literal["copilot", "litellm", "openai"] = "copilot",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model = model
        self.store = store
        self.validate = validate
        self.verbose = verbose
        self.backend = backend
        self.api_key = api_key
        self.api_base = api_base
        self._client: Optional[Any] = None

    def _ensure_client(self) -> None:
        """Ensure LLM client is initialized."""
        if self._client is None:
            if self.backend == "openai":
                from featcopilot.llm.openai_client import SyncOpenAIFeatureClient

                self._client = SyncOpenAIFeatureClient(model=self.model, api_key=self.api_key, api_base=self.api_base)
            elif self.backend == "litellm":
                from featcopilot.llm.litellm_client import SyncLiteLLMFeatureClient

                self._client = SyncLiteLLMFeatureClient(model=self.model, api_key=self.api_key, api_base=self.api_base)
            else:
                from featcopilot.llm.copilot_client import SyncCopilotFeatureClient

                self._client = SyncCopilotFeatureClient(model=self.model)
            self._client.start()

    def generate_from_description(
        self,
        description: str,
        columns: dict[str, str],
        sample_data: Optional[pd.DataFrame] = None,
        tags: Optional[list[str]] = None,
        save: bool = False,
    ) -> TransformRule:
        """
        Generate a transform rule from natural language description.

        Parameters
        ----------
        description : str
            Natural language description of the transformation
        columns : dict
            Available columns and their types (e.g., {"price": "float"})
        sample_data : DataFrame, optional
            Sample data for validation
        tags : list[str], optional
            Tags to add to the rule
        save : bool, default=False
            Whether to save the rule to the store

        Returns
        -------
        TransformRule
            Generated transform rule

        Examples
        --------
        >>> rule = generator.generate_from_description(
        ...     description="Calculate BMI from height in meters and weight in kg",
        ...     columns={"height_m": "float", "weight_kg": "float"},
        ...     tags=["healthcare", "bmi"]
        ... )
        """
        self._ensure_client()

        # Build prompt for rule generation
        prompt = self._build_generation_prompt(description, columns)

        # Get LLM response
        response = self._client.send_prompt(prompt)

        # Parse response into rule
        rule = self._parse_rule_response(response, description, columns, tags)

        # Validate if enabled
        if self.validate and sample_data is not None:
            rule = self._validate_and_fix(rule, sample_data)

        # Save if requested
        if save and self.store is not None:
            self.store.save_rule(rule)

        return rule

    def _build_generation_prompt(self, description: str, columns: dict[str, str]) -> str:
        """Build prompt for rule generation."""
        column_list = "\n".join(f"- {col} ({dtype})" for col, dtype in columns.items())

        return f"""You are an expert data scientist creating a REUSABLE feature transformation rule.

## Task
Create a reusable transformation rule based on this description:
"{description}"

## Available Columns
{column_list}

## Requirements
1. Generate Python code using pandas (assume `df` is the DataFrame)
2. Make the code REUSABLE by using the actual column names that can be substituted later
3. Assign the result to a variable called `result`
4. Handle edge cases (division by zero, missing values)
5. The rule should be generalizable to similar columns in other datasets

## Output Format
Return a JSON object with these fields:
- "name": short snake_case name for the rule (e.g., "ratio_calculation")
- "code": Python code that computes the transformation (single expression or multiple lines)
- "input_columns": list of column names used as inputs
- "output_type": "numeric", "categorical", or "boolean"
- "column_patterns": list of regex patterns to match similar columns (e.g., [".*price.*", ".*amount.*"])
- "explanation": brief explanation of what the rule does

Example output:
{{
  "name": "ratio_calculation",
  "code": "result = df['price'] / (df['quantity'] + 1e-8)",
  "input_columns": ["price", "quantity"],
  "output_type": "numeric",
  "column_patterns": [".*price.*", ".*quantity.*"],
  "explanation": "Calculates the ratio of price to quantity, handling division by zero"
}}

Return ONLY the JSON object, no other text.
"""

    def _parse_rule_response(
        self,
        response: str,
        description: str,
        columns: dict[str, str],
        tags: Optional[list[str]] = None,
    ) -> TransformRule:
        """Parse LLM response into a TransformRule."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(line for line in lines if not line.startswith("```"))

            data = json.loads(response)

            return TransformRule(
                name=data.get("name", "custom_rule"),
                description=description,
                code=self._clean_code(data.get("code", "")),
                input_columns=data.get("input_columns", list(columns.keys())),
                output_type=data.get("output_type", "numeric"),
                column_patterns=data.get("column_patterns", []),
                tags=tags or [],
                metadata={"original_columns": columns, "explanation": data.get("explanation", "")},
            )

        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return self._parse_rule_response(json_match.group(), description, columns, tags)
                except json.JSONDecodeError:
                    pass

            # Fallback: create basic rule from response
            logger.warning("Could not parse JSON response, creating basic rule")
            return TransformRule(
                name=self._generate_name(description),
                description=description,
                code=self._extract_code(response),
                input_columns=list(columns.keys()),
                tags=tags or [],
            )

    def _clean_code(self, code: str) -> str:
        """Clean and normalize generated code."""
        code = code.strip()

        # Remove markdown code blocks
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(line for line in lines if not line.startswith("```"))

        # Ensure result assignment
        if "result" not in code and "=" in code:
            code = re.sub(r"^(\w+)\s*=", "result =", code, count=1)
        elif "result" not in code:
            code = f"result = {code}"

        return code.strip()

    def _extract_code(self, response: str) -> str:
        """Extract code from a response that isn't valid JSON."""
        # Look for code patterns
        code_patterns = [
            r"result\s*=\s*[^\n]+",
            r"df\[['\"][^\n]+",
        ]

        for pattern in code_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group()

        # Fallback
        return "result = df.iloc[:, 0]"

    def _generate_name(self, description: str) -> str:
        """Generate rule name from description."""
        words = description.lower().split()
        significant = [w for w in words if len(w) > 2 and w not in {"the", "and", "for", "from", "with", "calculate"}][
            :3
        ]
        name = "_".join(significant)
        name = re.sub(r"[^a-z0-9_]", "", name)
        return name or "custom_rule"

    def _validate_and_fix(self, rule: TransformRule, sample_data: pd.DataFrame) -> TransformRule:
        """Validate rule code and attempt to fix issues."""
        validation = self._client.validate_feature_code(
            rule.code, {col: sample_data[col].tolist() for col in sample_data.columns}
        )

        if not validation["valid"]:
            if self.verbose:
                logger.warning(f"Rule validation failed: {validation['error']}")

            # Try to fix common issues
            fixed_code = self._fix_common_issues(rule.code, validation["error"])

            # Re-validate
            validation = self._client.validate_feature_code(
                fixed_code, {col: sample_data[col].tolist() for col in sample_data.columns}
            )

            if validation["valid"]:
                rule.code = fixed_code
            else:
                logger.warning(f"Could not fix rule code: {validation['error']}")

        return rule

    def _fix_common_issues(self, code: str, error: str) -> str:
        """Attempt to fix common code issues."""
        if "division by zero" in error.lower():
            code = re.sub(r"/\s*\(([^)]+)\)", r"/ (\1 + 1e-8)", code)
            code = re.sub(r"/\s*df\['([^']+)'\]", r"/ (df['\1'] + 1e-8)", code)

        if "syntax" in error.lower():
            code = code.replace("'", "'").replace("'", "'")
            code = code.replace(""", '"').replace(""", '"')

        return code

    def suggest_rules(
        self,
        columns: dict[str, str],
        task_description: Optional[str] = None,
        limit: int = 5,
    ) -> list[tuple[TransformRule, dict[str, str]]]:
        """
        Suggest applicable rules from the store for given columns.

        Parameters
        ----------
        columns : dict
            Available columns and their types
        task_description : str, optional
            Description of the ML task for better matching
        limit : int, default=5
            Maximum number of suggestions

        Returns
        -------
        list[tuple[TransformRule, dict]]
            List of (rule, column_mapping) tuples
        """
        if self.store is None:
            logger.warning("No rule store configured")
            return []

        column_names = list(columns.keys())
        matching = self.store.find_matching_rules(columns=column_names, description=task_description)

        return matching[:limit]

    def generate_and_suggest(
        self,
        description: str,
        columns: dict[str, str],
        sample_data: Optional[pd.DataFrame] = None,
        tags: Optional[list[str]] = None,
    ) -> tuple[Optional[TransformRule], list[tuple[TransformRule, dict[str, str]]]]:
        """
        Find existing matching rules or generate a new one.

        First searches for existing rules that match the description and columns.
        If no good matches found, generates a new rule.

        Parameters
        ----------
        description : str
            Natural language description
        columns : dict
            Available columns and their types
        sample_data : DataFrame, optional
            Sample data for validation
        tags : list[str], optional
            Tags for the new rule

        Returns
        -------
        new_rule : TransformRule or None
            Newly generated rule (None if existing rules found)
        existing_rules : list
            List of matching existing rules with column mappings
        """
        # Search for existing rules
        existing = self.suggest_rules(columns, description, limit=3)

        if existing:
            if self.verbose:
                logger.info(f"Found {len(existing)} existing matching rules")
            return None, existing

        # Generate new rule
        if self.verbose:
            logger.info("No matching rules found, generating new rule")

        new_rule = self.generate_from_description(
            description=description,
            columns=columns,
            sample_data=sample_data,
            tags=tags,
            save=False,
        )

        return new_rule, []

    def save_rule(self, rule: TransformRule) -> str:
        """
        Save a rule to the store.

        Parameters
        ----------
        rule : TransformRule
            Rule to save

        Returns
        -------
        str
            Rule ID
        """
        if self.store is None:
            raise ValueError("No rule store configured")
        return self.store.save_rule(rule)

    def __del__(self):
        """Clean up client."""
        if self._client:
            try:
                self._client.stop()
            except Exception:
                pass

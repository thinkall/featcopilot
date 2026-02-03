"""Transform rule model for reusable feature transformations.

Defines TransformRule - a reusable transformation that can be created from
natural language descriptions and applied across different datasets.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from featcopilot.utils.logger import get_logger

logger = get_logger(__name__)


class TransformRule(BaseModel):
    """
    A reusable feature transformation rule.

    Transform rules capture feature engineering logic that can be generated
    from natural language descriptions and reused across different datasets.

    Parameters
    ----------
    id : str, optional
        Unique identifier for the rule
    name : str
        Human-readable name for the rule
    description : str
        Natural language description of what the rule does
    code : str
        Python code that implements the transformation
    input_columns : list[str]
        Column names or patterns this rule expects as input
    output_name : str, optional
        Name for the output feature (default: derived from rule name)
    output_type : str
        Expected output data type ('numeric', 'categorical', 'boolean')
    tags : list[str]
        Tags for categorization and search
    column_patterns : list[str]
        Regex patterns for matching columns (e.g., 'price.*', '.*_amount')
    usage_count : int
        Number of times this rule has been applied
    created_at : str
        ISO timestamp of rule creation
    metadata : dict
        Additional metadata

    Examples
    --------
    >>> rule = TransformRule(
    ...     name="ratio_calculation",
    ...     description="Calculate ratio of two numeric columns",
    ...     code="result = df['{col1}'] / (df['{col2}'] + 1e-8)",
    ...     input_columns=["col1", "col2"],
    ...     tags=["ratio", "numeric"]
    ... )
    >>> result = rule.apply(df, column_mapping={"col1": "price", "col2": "quantity"})
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique rule identifier")
    name: str = Field(description="Human-readable rule name")
    description: str = Field(description="Natural language description of the transformation")
    code: str = Field(description="Python code implementing the transformation")
    input_columns: list[str] = Field(default_factory=list, description="Expected input column names or placeholders")
    output_name: Optional[str] = Field(default=None, description="Output feature name")
    output_type: str = Field(default="numeric", description="Output data type")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    column_patterns: list[str] = Field(default_factory=list, description="Regex patterns for column matching")
    usage_count: int = Field(default=0, description="Number of times applied")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Creation timestamp"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def get_output_name(self, column_mapping: Optional[dict[str, str]] = None) -> str:
        """
        Get the output feature name.

        Parameters
        ----------
        column_mapping : dict, optional
            Mapping from placeholder columns to actual column names

        Returns
        -------
        str
            Output feature name
        """
        if self.output_name:
            return self.output_name

        # Generate name from input columns
        if column_mapping and self.input_columns:
            cols = [column_mapping.get(c, c) for c in self.input_columns[:2]]
            return f"{'_'.join(cols)}_{self.name}"

        return f"rule_{self.name}"

    def matches_columns(self, columns: list[str]) -> tuple[bool, dict[str, str]]:
        """
        Check if this rule can be applied to the given columns.

        Parameters
        ----------
        columns : list[str]
            Available column names

        Returns
        -------
        matches : bool
            Whether the rule can be applied
        mapping : dict
            Suggested mapping from rule's input_columns to actual columns
        """
        if not self.input_columns:
            return True, {}

        mapping = {}

        for input_col in self.input_columns:
            # Try exact match first
            if input_col in columns:
                mapping[input_col] = input_col
                continue

            # Try pattern matching
            matched = False
            for pattern in self.column_patterns:
                regex = re.compile(pattern, re.IGNORECASE)
                for col in columns:
                    if regex.match(col) and col not in mapping.values():
                        mapping[input_col] = col
                        matched = True
                        break
                if matched:
                    break

            # Try fuzzy matching by checking if input_col is substring
            if not matched:
                for col in columns:
                    if input_col.lower() in col.lower() and col not in mapping.values():
                        mapping[input_col] = col
                        matched = True
                        break

            if not matched:
                return False, {}

        return len(mapping) == len(self.input_columns), mapping

    def apply(
        self,
        df: pd.DataFrame,
        column_mapping: Optional[dict[str, str]] = None,
        validate: bool = True,
    ) -> pd.Series:
        """
        Apply the transformation rule to a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input data
        column_mapping : dict, optional
            Mapping from rule's input_columns to actual column names
        validate : bool, default=True
            Whether to validate before execution

        Returns
        -------
        Series
            Transformed feature values

        Raises
        ------
        ValueError
            If required columns are missing or code execution fails
        """
        column_mapping = column_mapping or {}

        # Prepare the code with actual column names
        code = self._prepare_code(column_mapping)

        if validate:
            # Check required columns exist
            for input_col in self.input_columns:
                actual_col = column_mapping.get(input_col, input_col)
                if actual_col not in df.columns:
                    raise ValueError(f"Required column '{actual_col}' not found in DataFrame")

        # Execute the code in a restricted environment
        local_vars: dict[str, Any] = {"df": df, "np": np, "pd": pd}
        try:
            exec(self._get_safe_code(code), {"__builtins__": self._get_safe_builtins()}, local_vars)

            if "result" not in local_vars:
                raise ValueError("Code did not produce a 'result' variable")

            result = local_vars["result"]

            # Increment usage count
            self.usage_count += 1

            return result

        except Exception as e:
            logger.error(f"Failed to apply rule '{self.name}': {e}")
            raise ValueError(f"Rule execution failed: {e}") from e

    def _prepare_code(self, column_mapping: dict[str, str]) -> str:
        """Substitute column placeholders with actual column names."""
        code = self.code

        # Replace {col} style placeholders
        for placeholder, actual in column_mapping.items():
            code = code.replace(f"{{{{ '{placeholder}' }}}}", f"'{actual}'")
            code = code.replace(f"{{{placeholder}}}", actual)
            code = code.replace(f"df['{placeholder}']", f"df['{actual}']")
            code = code.replace(f'df["{placeholder}"]', f'df["{actual}"]')

        return code

    def _get_safe_code(self, code: str) -> str:
        """Wrap code for safe execution."""
        return code

    def _get_safe_builtins(self) -> dict[str, Any]:
        """Get restricted builtins for safe code execution."""
        return {
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

    def to_dict(self) -> dict[str, Any]:
        """Convert rule to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformRule":
        """Create rule from dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        return f"TransformRule(name='{self.name}', description='{self.description[:50]}...')"

"""LLM-powered feature code generator.

Generates Python code for custom features based on natural language descriptions.
"""

import re
from typing import Optional

import pandas as pd

from featcopilot.core.feature import Feature, FeatureOrigin, FeatureType
from featcopilot.llm.copilot_client import SyncCopilotFeatureClient


class FeatureCodeGenerator:
    """
    Generate Python code for features from natural language descriptions.

    Uses LLM to understand feature requirements and generate
    working pandas code.

    Parameters
    ----------
    model : str, default='gpt-5'
        LLM model to use
    validate : bool, default=True
        Whether to validate generated code

    Examples
    --------
    >>> generator = FeatureCodeGenerator()
    >>> feature = generator.generate(
    ...     description="Calculate BMI from height and weight",
    ...     columns={'height_m': 'float', 'weight_kg': 'float'}
    ... )
    """

    def __init__(self, model: str = "gpt-5", validate: bool = True, verbose: bool = False):
        self.model = model
        self.validate = validate
        self.verbose = verbose
        self._client: Optional[SyncCopilotFeatureClient] = None

    def _ensure_client(self) -> None:
        """Ensure client is initialized."""
        if self._client is None:
            self._client = SyncCopilotFeatureClient(model=self.model)
            self._client.start()

    def generate(
        self,
        description: str,
        columns: dict[str, str],
        constraints: Optional[list[str]] = None,
        sample_data: Optional[pd.DataFrame] = None,
    ) -> Feature:
        """
        Generate a feature from natural language description.

        Parameters
        ----------
        description : str
            Natural language description of the feature
        columns : dict
            Available columns and their types
        constraints : list, optional
            Code constraints (e.g., "avoid division by zero")
        sample_data : DataFrame, optional
            Sample data for validation

        Returns
        -------
        feature : Feature
            Generated feature with code
        """
        self._ensure_client()

        # Generate code
        code = self._client.generate_feature_code(
            description=description,
            column_info=columns,
            constraints=constraints,
        )

        # Clean code
        code = self._clean_code(code)

        # Generate feature name
        name = self._generate_name(description)

        # Detect source columns
        source_columns = self._detect_source_columns(code, list(columns.keys()))

        # Validate if enabled
        if self.validate and sample_data is not None:
            validation = self._client.validate_feature_code(
                code, {col: sample_data[col].tolist() for col in sample_data.columns}
            )
            if not validation["valid"]:
                if self.verbose:
                    print(f"Code validation failed: {validation['error']}")
                # Try to fix common issues
                code = self._fix_common_issues(code, validation["error"])

        feature = Feature(
            name=name,
            dtype=FeatureType.NUMERIC,
            origin=FeatureOrigin.LLM_GENERATED,
            source_columns=source_columns,
            transformation="custom",
            explanation=description,
            code=code,
        )

        return feature

    def generate_batch(
        self,
        descriptions: list[str],
        columns: dict[str, str],
        sample_data: Optional[pd.DataFrame] = None,
    ) -> list[Feature]:
        """
        Generate multiple features from descriptions.

        Parameters
        ----------
        descriptions : list
            List of feature descriptions
        columns : dict
            Available columns and their types
        sample_data : DataFrame, optional
            Sample data for validation

        Returns
        -------
        features : list
            List of generated features
        """
        features = []
        for desc in descriptions:
            try:
                feature = self.generate(desc, columns, sample_data=sample_data)
                features.append(feature)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to generate feature for '{desc}': {e}")

        return features

    def _clean_code(self, code: str) -> str:
        """Clean and normalize generated code."""
        # Remove markdown code blocks
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(line for line in lines if not line.startswith("```"))

        # Remove comments
        lines = []
        for line in code.split("\n"):
            if not line.strip().startswith("#"):
                lines.append(line)
        code = "\n".join(lines).strip()

        # Ensure result assignment
        if "result" not in code:
            # Try to extract the expression and wrap it
            if "=" in code:
                # Already has an assignment, replace variable name
                code = re.sub(r"^(\w+)\s*=", "result =", code)
            else:
                # Raw expression
                code = f"result = {code}"

        return code

    def _generate_name(self, description: str) -> str:
        """Generate a feature name from description."""
        # Take first few significant words
        words = description.lower().split()
        significant = [
            w for w in words if len(w) > 2 and w not in {"the", "and", "for", "from", "with", "calculate", "compute"}
        ][:4]

        name = "_".join(significant)
        # Clean up
        name = re.sub(r"[^a-z0-9_]", "", name)
        name = re.sub(r"_+", "_", name)

        return name or "custom_feature"

    def _detect_source_columns(self, code: str, available_columns: list[str]) -> list[str]:
        """Detect which columns are used in the code."""
        sources = []
        for col in available_columns:
            # Check for df['col'] or df["col"] or df.col patterns
            patterns = [
                f"df['{col}']",
                f'df["{col}"]',
                f"df.{col}",
            ]
            if any(pattern in code for pattern in patterns):
                sources.append(col)

        return sources

    def _fix_common_issues(self, code: str, error: str) -> str:
        """Try to fix common code issues."""
        if "division by zero" in error.lower():
            # Add small epsilon to divisors
            code = re.sub(r"/\s*\(([^)]+)\)", r"/ (\1 + 1e-8)", code)
            code = re.sub(r"/\s*df\['([^']+)'\]", r"/ (df['\1'] + 1e-8)", code)

        if "keyerror" in error.lower() or "not found" in error.lower():
            # Can't fix missing columns
            pass

        if "syntax" in error.lower():
            # Try removing problematic characters
            code = code.replace("'", "'").replace("'", "'")
            code = code.replace(""", '"').replace(""", '"')

        return code

    def generate_domain_features(self, domain: str, columns: dict[str, str], n_features: int = 5) -> list[Feature]:
        """
        Generate domain-specific features.

        Parameters
        ----------
        domain : str
            Domain name (e.g., 'healthcare', 'finance', 'retail')
        columns : dict
            Available columns and their types
        n_features : int, default=5
            Number of features to generate

        Returns
        -------
        features : list
            Generated domain-specific features
        """
        domain_prompts = {
            "healthcare": [
                "Calculate BMI if height and weight columns exist",
                "Create age group categories (pediatric, adult, elderly)",
                "Calculate medication count normalized by age",
                "Create comorbidity score from diagnosis codes",
                "Calculate length of stay relative to average",
            ],
            "finance": [
                "Calculate debt-to-income ratio",
                "Create credit utilization percentage",
                "Calculate payment-to-income ratio",
                "Create account age in years",
                "Calculate average transaction amount",
            ],
            "retail": [
                "Calculate average order value",
                "Create recency score (days since last purchase)",
                "Calculate purchase frequency per month",
                "Create customer lifetime value estimate",
                "Calculate category diversity score",
            ],
            "telecom": [
                "Calculate average monthly charges",
                "Create contract length in months",
                "Calculate service usage intensity",
                "Create support ticket frequency",
                "Calculate revenue per service",
            ],
        }

        prompts = domain_prompts.get(
            domain.lower(),
            [
                f"Create a useful feature for {domain} analytics",
                f"Calculate a key metric for {domain}",
                f"Create an interaction feature relevant to {domain}",
            ],
        )

        # Select prompts based on available columns
        applicable_prompts = prompts[:n_features]

        return self.generate_batch(applicable_prompts, columns)

    def __del__(self):
        """Clean up client."""
        if self._client:
            try:
                self._client.stop()
            except Exception:
                pass

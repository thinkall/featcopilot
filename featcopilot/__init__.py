"""
FeatCopilot - Next-Generation LLM-Powered Auto Feature Engineering

A unified feature engineering framework combining traditional approaches
with novel LLM-powered capabilities via GitHub Copilot SDK.
"""

__version__ = "0.1.0"
__author__ = "FeatCopilot Contributors"

from featcopilot.core.base import BaseEngine, BaseSelector
from featcopilot.core.feature import Feature, FeatureSet
from featcopilot.transformers.sklearn_compat import (
    AutoFeatureEngineer,
    FeatureEngineerTransformer,
)

__all__ = [
    # Core
    "BaseEngine",
    "BaseSelector",
    "Feature",
    "FeatureSet",
    # Main API
    "AutoFeatureEngineer",
    "FeatureEngineerTransformer",
    # Version
    "__version__",
]

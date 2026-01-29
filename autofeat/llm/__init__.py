"""LLM-powered feature engineering module.

Uses GitHub Copilot SDK for intelligent feature generation.
"""

from autofeat.llm.copilot_client import CopilotFeatureClient
from autofeat.llm.semantic_engine import SemanticEngine
from autofeat.llm.explainer import FeatureExplainer
from autofeat.llm.code_generator import FeatureCodeGenerator

__all__ = [
    "CopilotFeatureClient",
    "SemanticEngine",
    "FeatureExplainer",
    "FeatureCodeGenerator",
]

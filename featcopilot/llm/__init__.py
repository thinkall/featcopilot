"""LLM-powered feature engineering module.

Uses GitHub Copilot SDK or LiteLLM for intelligent feature generation.
"""

from featcopilot.llm.code_generator import FeatureCodeGenerator
from featcopilot.llm.copilot_client import CopilotFeatureClient
from featcopilot.llm.explainer import FeatureExplainer
from featcopilot.llm.litellm_client import LiteLLMFeatureClient, SyncLiteLLMFeatureClient
from featcopilot.llm.semantic_engine import SemanticEngine

__all__ = [
    "CopilotFeatureClient",
    "LiteLLMFeatureClient",
    "SyncLiteLLMFeatureClient",
    "SemanticEngine",
    "FeatureExplainer",
    "FeatureCodeGenerator",
]

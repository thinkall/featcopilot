"""LLM-powered feature engineering module.

Uses OpenAI SDK, LiteLLM, or GitHub Copilot SDK for intelligent feature generation.
"""

from featcopilot.llm.code_generator import FeatureCodeGenerator
from featcopilot.llm.copilot_client import CopilotFeatureClient
from featcopilot.llm.explainer import FeatureExplainer
from featcopilot.llm.litellm_client import LiteLLMFeatureClient, SyncLiteLLMFeatureClient
from featcopilot.llm.openai_client import OpenAIFeatureClient, SyncOpenAIFeatureClient
from featcopilot.llm.semantic_engine import SemanticEngine
from featcopilot.llm.transform_rule_generator import TransformRuleGenerator

__all__ = [
    "CopilotFeatureClient",
    "LiteLLMFeatureClient",
    "SyncLiteLLMFeatureClient",
    "OpenAIFeatureClient",
    "SyncOpenAIFeatureClient",
    "SemanticEngine",
    "FeatureExplainer",
    "FeatureCodeGenerator",
    "TransformRuleGenerator",
]

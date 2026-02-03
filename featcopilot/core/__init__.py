"""Core module containing base classes and interfaces."""

from featcopilot.core.base import BaseEngine, BaseSelector
from featcopilot.core.feature import Feature, FeatureSet
from featcopilot.core.registry import FeatureRegistry
from featcopilot.core.transform_rule import TransformRule

__all__ = [
    "BaseEngine",
    "BaseSelector",
    "Feature",
    "FeatureSet",
    "FeatureRegistry",
    "TransformRule",
]

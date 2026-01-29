"""Core module containing base classes and interfaces."""

from featcopilot.core.base import BaseEngine, BaseSelector
from featcopilot.core.feature import Feature, FeatureSet
from featcopilot.core.registry import FeatureRegistry

__all__ = [
    "BaseEngine",
    "BaseSelector",
    "Feature",
    "FeatureSet",
    "FeatureRegistry",
]

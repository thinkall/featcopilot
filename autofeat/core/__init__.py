"""Core module containing base classes and interfaces."""

from autofeat.core.base import BaseEngine, BaseSelector
from autofeat.core.feature import Feature, FeatureSet
from autofeat.core.registry import FeatureRegistry

__all__ = [
    "BaseEngine",
    "BaseSelector",
    "Feature",
    "FeatureSet",
    "FeatureRegistry",
]

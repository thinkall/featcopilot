"""Feature selection module."""

from autofeat.selection.statistical import StatisticalSelector
from autofeat.selection.importance import ImportanceSelector
from autofeat.selection.redundancy import RedundancyEliminator
from autofeat.selection.unified import FeatureSelector

__all__ = [
    "StatisticalSelector",
    "ImportanceSelector",
    "RedundancyEliminator",
    "FeatureSelector",
]

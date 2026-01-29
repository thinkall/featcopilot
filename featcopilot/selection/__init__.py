"""Feature selection module."""

from featcopilot.selection.importance import ImportanceSelector
from featcopilot.selection.redundancy import RedundancyEliminator
from featcopilot.selection.statistical import StatisticalSelector
from featcopilot.selection.unified import FeatureSelector

__all__ = [
    "StatisticalSelector",
    "ImportanceSelector",
    "RedundancyEliminator",
    "FeatureSelector",
]

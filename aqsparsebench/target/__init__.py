"""Marginal target-station selection after sparse-network design."""

from aqsparsebench.config import TargetSelectionConfig, TargetSelectionWeights
from aqsparsebench.target.filters import filter_target_candidates, merge_eligibility
from aqsparsebench.target.scoring import compute_target_scores
from aqsparsebench.target.select import select_target_station

__all__ = [
    "TargetSelectionConfig",
    "TargetSelectionWeights",
    "compute_target_scores",
    "filter_target_candidates",
    "merge_eligibility",
    "select_target_station",
]

"""Pluggable sparse-network generation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class GenerationContext:
    """Inputs passed to a :class:`SparseGenerationStrategy`."""

    station_scores: dict[str, float]
    retention_level: float
    n_candidates: int
    region_id: str
    random_state: int
    station_df: "pd.DataFrame | None" = None
    """Optional columns ``station_id`` (or index), ``latitude``, ``longitude`` for spatial diversity."""
    distance_matrix_km: "pd.DataFrame | None" = None
    """Optional square matrix indexed by ``station_id``; used if ``station_df`` is absent."""
    min_pairwise_km: float | None = None
    diversity_penalty: bool = True
    score_power: float = 1.0
    """Raise scores to this power before softmax (>= 1 favors high-utility stations more)."""
    spatial_diversity_scale_km: float = 25.0
    """Larger values down-weight proximity less aggressively in sequential picks."""
    max_resample_attempts: int = 200


@runtime_checkable
class SparseGenerationStrategy(Protocol):
    """Implement ``generate`` to define how retained station sets are sampled."""

    @property
    def strategy_id(self) -> str: ...

    def generate(self, ctx: GenerationContext) -> list[list[str]]:
        """
        Return ``n_candidates`` (or fewer if impossible) lists of retained ``station_id`` strings.

        Each inner list must have length ``retained_station_count(len(scores), retention)``.
        """

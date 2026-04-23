"""Sparse monitoring-network candidate generation."""

from __future__ import annotations

import uuid

import pandas as pd

from aqsparsebench.benchmark.retention import retained_station_count, validate_station_scores
from aqsparsebench.benchmark.strategies.base import GenerationContext, SparseGenerationStrategy
from aqsparsebench.benchmark.strategies.registry import get_sparse_strategy
from aqsparsebench.config import SparseGenerationSettings
from aqsparsebench.types import SparseNetworkCandidate


def generate_sparse_candidates(
    station_scores: dict[str, float],
    retention_level: float,
    n_candidates: int,
    *,
    region_id: str = "region",
    random_state: int | None = None,
    station_df: "pd.DataFrame | None" = None,
    distance_matrix_km: "pd.DataFrame | None" = None,
    min_pairwise_km: float | None = None,
    diversity_penalty: bool = True,
    strategy: str | SparseGenerationStrategy | None = None,
    settings: SparseGenerationSettings | None = None,
) -> list[SparseNetworkCandidate]:
    """
    Sample ``n_candidates`` sparse retained-station sets at ``retention_level``.

    * **EPA-friendly default** — pass ``station_df`` with ``station_id``, ``latitude``,
      ``longitude`` to enable spatial diversity and ``min_pairwise_km`` checks.
    * **Custom rules** — pass ``strategy`` as a :class:`SparseGenerationStrategy` instance, or
      register one under a string id via :func:`~aqsparsebench.benchmark.strategies.registry.register_sparse_strategy`.
    """
    validate_station_scores(station_scores)
    rs = int(random_state if random_state is not None else 0)

    if isinstance(strategy, str):
        strat = get_sparse_strategy(strategy)
    elif strategy is not None:
        strat = strategy
    else:
        sid = settings.strategy if settings is not None else "default_weighted"
        strat = get_sparse_strategy(sid)

    min_km = min_pairwise_km
    if min_km is None and settings is not None:
        min_km = settings.min_pairwise_km

    ctx = GenerationContext(
        station_scores=dict(station_scores),
        retention_level=retention_level,
        n_candidates=n_candidates,
        region_id=region_id,
        random_state=rs,
        station_df=station_df,
        distance_matrix_km=distance_matrix_km,
        min_pairwise_km=min_km,
        diversity_penalty=diversity_penalty,
        score_power=settings.score_power if settings is not None else 1.0,
        spatial_diversity_scale_km=settings.spatial_diversity_scale_km if settings is not None else 25.0,
        max_resample_attempts=settings.max_resample_attempts if settings is not None else 200,
    )

    raw_sets = strat.generate(ctx)
    k = retained_station_count(len(station_scores), retention_level)
    out: list[SparseNetworkCandidate] = []
    for subset in raw_sets:
        if len(subset) != k:
            continue
        sub_scores = {s: float(station_scores[s]) for s in subset}
        out.append(
            SparseNetworkCandidate(
                candidate_id=str(uuid.uuid4()),
                region_id=region_id,
                retention_level=retention_level,
                station_ids=sorted(subset),
                station_scores=sub_scores,
                candidate_feature_vector=None,
            )
        )
    return out

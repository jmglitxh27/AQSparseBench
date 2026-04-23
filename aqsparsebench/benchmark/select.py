"""Pick medoid sparse networks (one representative per cluster)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aqsparsebench.types import RepresentativeSparseNetwork, SparseNetworkCandidate


def select_representative_networks(
    clustered_candidates: Sequence[SparseNetworkCandidate],
    *,
    network_id_prefix: str = "network",
) -> list[RepresentativeSparseNetwork]:
    """
    Emit one :class:`~aqsparsebench.types.RepresentativeSparseNetwork` per medoid candidate.

    Medoids must already be flagged with ``is_medoid`` (see :func:`~aqsparsebench.benchmark.cluster.cluster_candidates`).
    """
    medoids = [c for c in clustered_candidates if c.is_medoid]
    medoids.sort(key=lambda c: (c.cluster_id if c.cluster_id is not None else -1, c.candidate_id))
    out: list[RepresentativeSparseNetwork] = []
    for i, c in enumerate(medoids, start=1):
        nid = f"{network_id_prefix}_{i:04d}"
        vals = list(c.station_scores.values())
        summary = {
            "mean_utility_retained": float(np.mean(vals)) if vals else float("nan"),
            "n_retained": len(c.station_ids),
            "retention_level": c.retention_level,
            "cluster_id": c.cluster_id,
        }
        out.append(
            RepresentativeSparseNetwork(
                network_id=nid,
                retention_level=c.retention_level,
                cluster_id=int(c.cluster_id) if c.cluster_id is not None else -1,
                candidate_id=c.candidate_id,
                station_ids=list(c.station_ids),
                summary=summary,
            )
        )
    return out

import numpy as np
import pandas as pd

from aqsparsebench.benchmark.cluster import cluster_candidates
from aqsparsebench.benchmark.represent import attach_candidate_vectors
from aqsparsebench.types import SparseNetworkCandidate


def test_cluster_candidates_fallback_without_sklearn_extra(monkeypatch) -> None:
    # Force fallback path
    import aqsparsebench.benchmark.cluster as cl

    monkeypatch.setattr(cl, "KMedoids", None)

    station_feature_table = pd.DataFrame(
        {
            "station_id": ["a", "b", "c", "d"],
            "latitude": [0.0, 0.0, 1.0, 1.0],
            "longitude": [0.0, 1.0, 0.0, 1.0],
            "J_s": [0.1, 0.2, 0.9, 0.8],
            "C_s": [0.1, 0.2, 0.9, 0.8],
            "P_s": [0.5, 0.5, 0.5, 0.5],
            "V_s": [0.2, 0.2, 0.3, 0.3],
        }
    )
    cands = [
        SparseNetworkCandidate("c1", "r", 0.1, ["a", "b"], {"a": 0.1, "b": 0.2}),
        SparseNetworkCandidate("c2", "r", 0.1, ["a", "c"], {"a": 0.1, "c": 0.9}),
        SparseNetworkCandidate("c3", "r", 0.1, ["c", "d"], {"c": 0.9, "d": 0.8}),
        SparseNetworkCandidate("c4", "r", 0.1, ["b", "d"], {"b": 0.2, "d": 0.8}),
    ]
    cands = attach_candidate_vectors(cands, station_feature_table)
    out = cluster_candidates(cands, auto_k=True, random_state=0)

    assert all(c.cluster_id is not None for c in out)
    assert any(c.is_medoid for c in out)

    # exactly one medoid per cluster label
    labels = np.array([c.cluster_id for c in out], dtype=int)
    n_clusters = len(set(labels.tolist()))
    assert sum(1 for c in out if c.is_medoid) == n_clusters


"""k-medoids clustering of sparse-network candidates **within** one retention level."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from aqsparsebench.types import SparseNetworkCandidate

try:
    from sklearn_extra.cluster import KMedoids
except ImportError as e:  # pragma: no cover
    KMedoids = None  # type: ignore[misc, assignment]
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _require_kmedoids() -> None:
    if KMedoids is None:
        raise ImportError(
            "sklearn-extra is required for k-medoids clustering. Install scikit-learn-extra."
        ) from _IMPORT_ERR


def _pairwise_sqeuclidean(X: np.ndarray) -> np.ndarray:
    """Dense squared Euclidean distances for small n (O(n^2))."""
    # (x-y)^2 = x^2 + y^2 - 2xy
    g = X @ X.T
    sq = np.sum(X * X, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * g
    return np.maximum(D2, 0.0)


def _choose_medoids_from_labels(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return medoid indices, one per unique label, minimizing within-cluster distance sums."""
    medoids: list[int] = []
    for lab in sorted(set(int(x) for x in labels.tolist())):
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        if idx.size == 1:
            medoids.append(int(idx[0]))
            continue
        sub = X[idx]
        D2 = _pairwise_sqeuclidean(sub)
        # medoid = argmin sum distances
        best = int(idx[int(np.argmin(np.sum(D2, axis=1)))])
        medoids.append(best)
    return np.array(medoids, dtype=int)


def _auto_k_kmeans(X: np.ndarray, *, max_k: int, random_state: int) -> int:
    """Pick k for KMeans with silhouette, using Euclidean distance."""
    n = X.shape[0]
    if n < 3:
        return max(2, min(n, 2)) if n >= 2 else 1
    hi = min(max_k, n - 1, max(2, n - 1))
    lo = 2
    best_k, best_s = lo, -1.0
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(X)
        if len(np.unique(km.labels_)) < 2:
            continue
        try:
            s = float(silhouette_score(X, km.labels_, metric="euclidean"))
        except Exception:
            continue
        if s > best_s:
            best_s, best_k = s, k
    return best_k if best_s > -0.5 else min(lo, n - 1)


def _auto_k(
    X: np.ndarray,
    *,
    max_k: int,
    random_state: int,
    metric: str,
) -> int:
    """Pick ``k`` in ``[2, min(max_k, n-1)]`` with highest silhouette (if possible)."""
    _require_kmedoids()
    n = X.shape[0]
    if n < 3:
        return max(2, min(n, 2)) if n >= 2 else 1
    hi = min(max_k, n - 1, max(2, n - 1))
    lo = 2
    best_k, best_s = lo, -1.0
    for k in range(lo, hi + 1):
        km = KMedoids(n_clusters=k, metric=metric, random_state=random_state).fit(X)
        if len(np.unique(km.labels_)) < 2:
            continue
        try:
            s = float(silhouette_score(X, km.labels_, metric=metric))
        except Exception:
            continue
        if s > best_s:
            best_s, best_k = s, k
    return best_k if best_s > -0.5 else min(lo, n - 1)


def cluster_candidates(
    candidates: Sequence[SparseNetworkCandidate],
    *,
    n_clusters: int | None = None,
    auto_k: bool = True,
    method: str = "kmedoids",
    metric: str = "euclidean",
    random_state: int = 0,
    max_clusters: int = 12,
) -> list[SparseNetworkCandidate]:
    """
    Cluster candidates that share the same ``retention_level`` using Euclidean k-medoids.

    Updates ``cluster_id`` and ``is_medoid`` (``True`` exactly for medoid rows). Raises if
    vectors are missing or fewer than two candidates.
    """
    if method.lower() != "kmedoids":
        raise ValueError(f"Only 'kmedoids' is supported, got {method!r}")

    cand_list = list(candidates)
    if len(cand_list) < 2:
        raise ValueError("Need at least two candidates for clustering")
    if any(c.candidate_feature_vector is None for c in cand_list):
        raise ValueError("All candidates must have candidate_feature_vector set (use represent.attach_candidate_vectors)")
    levels = {c.retention_level for c in cand_list}
    if len(levels) != 1:
        raise ValueError("All candidates in one clustering call must share the same retention_level")

    X = np.array([list(c.candidate_feature_vector or []) for c in cand_list], dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Invalid feature matrix assembled from candidates")

    n = X.shape[0]
    using_fallback = KMedoids is None

    if not using_fallback:
        # Primary path: sklearn-extra k-medoids
        if n == 2:
            km = KMedoids(n_clusters=1, metric=metric, random_state=random_state).fit(X)
            labels = np.zeros(n, dtype=int)
            med_idx = np.array([int(km.medoid_indices_[0])], dtype=int)
        else:
            if auto_k or n_clusters is None:
                k = _auto_k(X, max_k=max_clusters, random_state=random_state, metric=metric)
            else:
                k = int(n_clusters)
            k = max(2, min(k, n - 1))
            km = KMedoids(n_clusters=k, metric=metric, random_state=random_state).fit(X)
            labels = km.labels_.astype(int)
            med_idx = np.array(km.medoid_indices_, dtype=int)
    else:
        # Fallback: KMeans labels + true medoids picked by within-cluster distances.
        # This keeps the "medoid representatives" idea without sklearn-extra binaries.
        if n == 2:
            labels = np.zeros(n, dtype=int)
            med_idx = np.array([0], dtype=int)
        else:
            if auto_k or n_clusters is None:
                k = _auto_k_kmeans(X, max_k=max_clusters, random_state=random_state)
            else:
                k = int(n_clusters)
            k = max(2, min(k, n - 1))
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(X)
            labels = km.labels_.astype(int)
            med_idx = _choose_medoids_from_labels(X, labels)
    out: list[SparseNetworkCandidate] = []
    medoids_set = set(med_idx.tolist())
    for i, c in enumerate(cand_list):
        out.append(
            replace(
                c,
                cluster_id=int(labels[i]),
                is_medoid=i in medoids_set,
            )
        )
    return out

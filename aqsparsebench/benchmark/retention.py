"""Retention level helpers (how many stations stay active in a sparse realization)."""

from __future__ import annotations

import math


def retained_station_count(n_stations: int, retention_level: float) -> int:
    """
    Number of stations to retain for ``retention_level`` in ``(0, 1]``.

    Uses ``max(1, round(retention * n))`` so the dense baseline ``1.0`` keeps all sites.
    """
    if n_stations <= 0:
        return 0
    if not (0.0 < retention_level <= 1.0):
        raise ValueError(f"retention_level must be in (0, 1], got {retention_level}")
    k = int(round(retention_level * n_stations))
    return max(1, min(n_stations, k))


def validate_station_scores(station_scores: dict[str, float]) -> None:
    if not station_scores:
        raise ValueError("station_scores must be non-empty")
    for sid, v in station_scores.items():
        x = float(v)
        if not math.isfinite(x) or x < 0.0:
            raise ValueError(f"Invalid score for {sid!r}: {v}")

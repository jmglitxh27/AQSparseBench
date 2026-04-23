"""Great-circle distances between stations (WGS84 degrees in, kilometers out)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometers."""
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    h = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return float(2 * r * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def pairwise_distance_matrix(
    station_df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Symmetric distance matrix (km) indexed and columned by ``station_df`` row order.

    For a keyed matrix, set ``station_df`` index to ``station_id`` before calling.
    """
    if station_df.empty:
        return pd.DataFrame()
    lat = np.radians(pd.to_numeric(station_df[lat_col], errors="coerce").to_numpy(dtype=float))
    lon = np.radians(pd.to_numeric(station_df[lon_col], errors="coerce").to_numpy(dtype=float))
    n = len(lat)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        dphi = lat - lat[i]
        dlmb = lon - lon[i]
        h = np.sin(dphi / 2.0) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlmb / 2.0) ** 2
        d[i, :] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))
    return pd.DataFrame(d, index=station_df.index, columns=station_df.index)

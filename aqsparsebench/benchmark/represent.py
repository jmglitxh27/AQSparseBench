"""Map each sparse-network candidate to a fixed-length feature vector for clustering."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from aqsparsebench.preprocess.geo import haversine_km
from aqsparsebench.types import SparseNetworkCandidate


def _feature_row(table: pd.DataFrame, sid: str, station_col: str) -> pd.Series | None:
    if station_col not in table.columns:
        table = table.reset_index().rename(columns={"index": station_col})
    m = table[table[station_col].astype(str) == str(sid)]
    if m.empty:
        return None
    return m.iloc[0]


def _pairwise_stats_km(
    station_ids: Sequence[str],
    station_feature_table: pd.DataFrame,
    *,
    station_col: str = "station_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> tuple[float, float]:
    """Return (mean pairwise km, min pairwise km) over unordered distinct pairs."""
    coords: dict[str, tuple[float, float]] = {}
    for sid in station_ids:
        row = _feature_row(station_feature_table, str(sid), station_col)
        if row is None or lat_col not in row.index or lon_col not in row.index:
            continue
        la = float(row[lat_col])
        lo = float(row[lon_col])
        if math.isfinite(la) and math.isfinite(lo):
            coords[str(sid)] = (la, lo)
    ids = [s for s in station_ids if str(s) in coords]
    if len(ids) < 2:
        return 0.0, 0.0
    dists: list[float] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = coords[ids[i]], coords[ids[j]]
            dists.append(haversine_km(a[0], a[1], b[0], b[1]))
    return float(np.mean(dists)), float(np.min(dists))


def _score_entropy(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    counts, _ = np.histogram(values, bins=min(10, max(3, values.size)))
    counts = counts.astype(float) + 1e-9
    return float(scipy_entropy(counts))


def candidate_to_vector(
    candidate: SparseNetworkCandidate,
    station_feature_table: pd.DataFrame,
    *,
    station_col: str = "station_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> list[float]:
    """
    Fixed 10-D summary (stable column order for Parquet / clustering):

    #. mean ``J_s`` among retained stations
    #. std / min / max ``J_s``
    #. mean / min pairwise great-circle distance (km) when lat/lon exist
    #. entropy of ``J_s`` histogram across retained sites
    #. mean ``C_s``, ``P_s``, ``V_s`` among retained sites (concentration / population / variability)
    """
    ids = list(candidate.station_ids)
    js: list[float] = []
    cs: list[float] = []
    ps: list[float] = []
    vs: list[float] = []
    for sid in ids:
        row = _feature_row(station_feature_table, sid, station_col)
        if row is None:
            continue
        if "J_s" in row.index and pd.notna(row.get("J_s")):
            js.append(float(row["J_s"]))
        if "C_s" in row.index and pd.notna(row.get("C_s")):
            cs.append(float(row["C_s"]))
        if "P_s" in row.index and pd.notna(row.get("P_s")):
            ps.append(float(row["P_s"]))
        if "V_s" in row.index and pd.notna(row.get("V_s")):
            vs.append(float(row["V_s"]))

    arr = np.array(js, dtype=float)
    mean_j = float(arr.mean()) if arr.size else 0.0
    std_j = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    min_j = float(arr.min()) if arr.size else 0.0
    max_j = float(arr.max()) if arr.size else 0.0
    ent = _score_entropy(arr)
    mean_d, min_d = _pairwise_stats_km(ids, station_feature_table, station_col=station_col, lat_col=lat_col, lon_col=lon_col)
    mean_c = float(np.mean(cs)) if cs else 0.0
    mean_p = float(np.mean(ps)) if ps else 0.0
    mean_v = float(np.mean(vs)) if vs else 0.0
    return [mean_j, std_j, min_j, max_j, mean_d, min_d, ent, mean_c, mean_p, mean_v]


def attach_candidate_vectors(
    candidates: Sequence[SparseNetworkCandidate],
    station_feature_table: pd.DataFrame,
    *,
    station_col: str = "station_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> list[SparseNetworkCandidate]:
    """Return new candidates with ``candidate_feature_vector`` populated."""
    from dataclasses import replace

    out: list[SparseNetworkCandidate] = []
    for c in candidates:
        vec = candidate_to_vector(
            c,
            station_feature_table,
            station_col=station_col,
            lat_col=lat_col,
            lon_col=lon_col,
        )
        out.append(replace(c, candidate_feature_vector=vec))
    return out

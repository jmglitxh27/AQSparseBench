"""Marginal target suitability scores ``T_s`` relative to a retained sparse network."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from aqsparsebench.config import TargetSelectionConfig
from aqsparsebench.features._common import minmax_series
from aqsparsebench.target.filters import filter_target_candidates, merge_eligibility

FEATURE_COLS = ("C_s", "P_s", "W_s", "V_s", "B_s")


def _table_indexed(df: pd.DataFrame, station_col: str) -> pd.DataFrame:
    out = df.copy()
    if station_col not in out.columns:
        out = out.reset_index().rename(columns={"index": station_col})
    out[station_col] = out[station_col].astype(str)
    return out.set_index(station_col)


def compute_target_scores(
    retained_station_ids: Sequence[str],
    candidate_station_ids: Sequence[str],
    station_feature_table: pd.DataFrame,
    distance_matrix_km: pd.DataFrame,
    config: TargetSelectionConfig,
    *,
    station_col: str = "station_id",
    apply_filters: bool = True,
) -> pd.DataFrame:
    """
    One row per **candidate** (station not in the retained set) with ``J_s``, ``G_s``, ``U_s``,
    ``E_s``, ``T_s``, and eligibility columns.

    * ``G_s_given_network`` — gap coverage from nearest retained geographic distance (min–max
      scaled across candidates so larger empty space scores higher).
    * ``U_s_given_network`` — underrepresentation via minimum Euclidean distance in z-scored
      feature space to any retained station.
    * ``E_s_given_network`` — v1 blend ``(G_s + U_s) / 2`` after both are on ``[0, 1]``.
    """
    retained = frozenset(str(s) for s in retained_station_ids)
    candidates = [str(c) for c in candidate_station_ids if str(c) not in retained]
    if not candidates:
        return pd.DataFrame(
            columns=[
                station_col,
                "J_s",
                "G_s_given_network",
                "U_s_given_network",
                "E_s_given_network",
                "T_s_given_network",
                "eligible",
                "eligibility_reason",
            ]
        )

    feat = _table_indexed(station_feature_table, station_col)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    if not cols:
        raise ValueError(f"station_feature_table needs at least one of {FEATURE_COLS}")

    z = feat[cols].apply(pd.to_numeric, errors="coerce")
    mu = z.mean(axis=0)
    sigma = z.std(axis=0, ddof=1).replace(0.0, np.nan)
    znorm = ((z - mu) / sigma).fillna(0.0)

    dist = distance_matrix_km.copy()
    dist.index = dist.index.astype(str)
    dist.columns = dist.columns.astype(str)
    ret_list = [s for s in retained if s in dist.index and s in dist.columns]
    z_ret = znorm.loc[[s for s in retained if s in znorm.index]]

    g_raw: dict[str, float] = {}
    u_raw: dict[str, float] = {}
    j_map: dict[str, float] = {}

    for sid in candidates:
        if sid not in feat.index:
            continue
        if ret_list and sid in dist.index:
            g_raw[sid] = float(dist.loc[sid, ret_list].min())
        else:
            g_raw[sid] = float("nan")

        if not z_ret.empty and sid in znorm.index:
            v_c = znorm.loc[sid, cols].to_numpy(dtype=float)
            dmin = math.inf
            for r in z_ret.index:
                v_r = z_ret.loc[r, cols].to_numpy(dtype=float)
                dmin = min(dmin, float(np.linalg.norm(v_c - v_r)))
            u_raw[sid] = dmin if math.isfinite(dmin) else float("nan")
        else:
            u_raw[sid] = float("nan")

        if "J_s" in feat.columns:
            j_map[sid] = float(pd.to_numeric(feat.loc[sid, "J_s"], errors="coerce"))
        else:
            j_map[sid] = float("nan")

    g_series = pd.Series({k: v for k, v in g_raw.items() if math.isfinite(v)})
    u_series = pd.Series({k: v for k, v in u_raw.items() if math.isfinite(v)})

    g_score = minmax_series(g_series, feature_low=0.0, feature_high=1.0) if len(g_series) else pd.Series(dtype=float)
    u_score = minmax_series(u_series, feature_low=0.0, feature_high=1.0) if len(u_series) else pd.Series(dtype=float)

    rows = []
    for sid in candidates:
        if sid not in feat.index:
            continue
        G = float(g_score.loc[sid]) if sid in g_score.index else 0.5
        U = float(u_score.loc[sid]) if sid in u_score.index else 0.5
        E = 0.5 * (G + U)
        Jv = j_map.get(sid, float("nan"))
        if not math.isfinite(Jv):
            Jv = 0.5
        w = config.weights
        T = w.lambda_J * Jv + w.lambda_G * G + w.lambda_U * U + w.lambda_E * E
        rows.append(
            {
                station_col: sid,
                "J_s": Jv,
                "G_s_given_network": G,
                "U_s_given_network": U,
                "E_s_given_network": E,
                "T_s_given_network": T,
            }
        )

    out = pd.DataFrame(rows)
    if apply_filters:
        universe = frozenset(retained) | frozenset(candidates)
        _, elig = filter_target_candidates(
            retained,
            universe,
            station_feature_table,
            distance_matrix_km,
            min_days=config.min_days,
            min_completeness=config.min_completeness,
            min_feature_distance=config.min_feature_distance,
            min_geo_distance_km=config.min_geo_distance_km,
            station_col=station_col,
        )
        out = merge_eligibility(out, elig, station_col=station_col)
    else:
        out["eligible"] = True
        out["eligibility_reason"] = "not_filtered"
    return out.sort_values("T_s_given_network", ascending=False).reset_index(drop=True)

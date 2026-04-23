"""Concentration relevance score from daily PM (or other pollutant) statistics."""

from __future__ import annotations

import pandas as pd

from aqsparsebench.config import ScoringConfig
from aqsparsebench.features._common import minmax_series
from aqsparsebench.preprocess.canonical import COL_CONCENTRATION, COL_STATION_ID


def station_concentration_aggregates(
    daily_df: pd.DataFrame,
    *,
    station_col: str = COL_STATION_ID,
    value_col: str = COL_CONCENTRATION,
) -> pd.DataFrame:
    """Per-station mean, 95th percentile, and maximum of daily concentrations."""
    if daily_df.empty:
        return pd.DataFrame(columns=[station_col, "conc_mean", "conc_q95", "conc_max"])

    def q95(x: pd.Series) -> float:
        return float(x.quantile(0.95))

    g = daily_df.groupby(station_col, sort=False)[value_col]
    out = g.agg(conc_mean="mean", conc_q95=q95, conc_max="max").reset_index()
    return out


def compute_concentration_score(
    daily_df: pd.DataFrame,
    config: ScoringConfig,
    *,
    station_col: str = COL_STATION_ID,
    value_col: str = COL_CONCENTRATION,
) -> pd.Series:
    """
    Return ``C_s`` indexed by ``station_id``: weighted mix of min–max normalized
    mean / 95th percentile / maximum daily concentration across stations in ``daily_df``.
    """
    if daily_df.empty:
        return pd.Series(dtype=float)
    lo, hi = config.normalize_range
    agg = station_concentration_aggregates(daily_df, station_col=station_col, value_col=value_col)
    w = config.concentration
    nm = minmax_series(agg["conc_mean"], feature_low=lo, feature_high=hi).to_numpy()
    nq = minmax_series(agg["conc_q95"], feature_low=lo, feature_high=hi).to_numpy()
    nx = minmax_series(agg["conc_max"], feature_low=lo, feature_high=hi).to_numpy()
    vals = w.lambda_mean * nm + w.lambda_q95 * nq + w.lambda_max * nx
    return pd.Series(vals, index=agg[station_col].astype(str).values, name="C_s")

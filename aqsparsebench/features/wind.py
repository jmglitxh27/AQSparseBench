"""Wind / transport relevance from merged daily weather columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqsparsebench.config import ScoringConfig
from aqsparsebench.features._common import minmax_series


def _mean_resultant_length(deg: np.ndarray) -> float:
    """R in [0, 1] for directional vectors; 1 = perfectly aligned, ~0 = diffuse."""
    rad = np.deg2rad(deg[~np.isnan(deg)])
    if rad.size == 0:
        return float("nan")
    c = np.cos(rad).mean()
    s = np.sin(rad).mean()
    return float(np.hypot(c, s))


def compute_wind_score(
    weather_daily_df: pd.DataFrame,
    config: ScoringConfig,
    *,
    station_col: str = "station_id",
    speed_col: str = "wind_speed_10m_max",
    direction_col: str = "wind_direction_10m_dominant",
    station_ids: pd.Index | None = None,
) -> pd.Series:
    """
    ``W_s`` from mean wind speed and directional dispersion ``1 - R`` (resultant length).

    If required columns are missing or ``weather_daily_df`` is empty, returns **0.5**
    for every ``station_id`` in ``station_ids`` (neutral), or an empty series if
    ``station_ids`` is None.
    """
    lo, hi = config.normalize_range
    mid = lo + (hi - lo) / 2.0
    need = {station_col, speed_col, direction_col}
    if weather_daily_df.empty or not need.issubset(weather_daily_df.columns):
        if station_ids is None:
            return pd.Series(dtype=float)
        return pd.Series(mid, index=station_ids.astype(str), dtype=float).rename("W_s")

    rows: list[dict[str, float | str]] = []
    for sid, g in weather_daily_df.groupby(station_col, sort=False):
        spd = pd.to_numeric(g[speed_col], errors="coerce").dropna()
        dire = pd.to_numeric(g[direction_col], errors="coerce").dropna()
        mean_speed = float(spd.mean()) if not spd.empty else float("nan")
        R = _mean_resultant_length(dire.to_numpy(dtype=float)) if not dire.empty else float("nan")
        disp = float(1.0 - R) if not np.isnan(R) else float("nan")
        rows.append({"station_id": str(sid), "mean_speed": mean_speed, "dispersion": disp})

    agg = pd.DataFrame(rows)
    w = config.wind
    ns = minmax_series(agg["mean_speed"], feature_low=lo, feature_high=hi).to_numpy()
    nd = minmax_series(agg["dispersion"], feature_low=lo, feature_high=hi).to_numpy()
    vals = w.omega_speed * ns + w.omega_directional_variability * nd
    out = pd.Series(vals, index=agg["station_id"].values, name="W_s")

    if station_ids is not None:
        out = out.reindex(station_ids.astype(str))
        out = out.fillna(mid)
    return out

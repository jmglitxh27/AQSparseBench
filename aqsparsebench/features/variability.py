"""Temporal variability score from daily concentration dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqsparsebench.config import ScoringConfig
from aqsparsebench.features._common import minmax_series
from aqsparsebench.preprocess.canonical import COL_CONCENTRATION, COL_DATE, COL_STATION_ID


def _seasonal_amplitude(dates: pd.Series, values: pd.Series) -> float:
    """Peak-to-trough of monthly mean concentrations (simple annual amplitude proxy)."""
    t = pd.to_datetime(dates)
    m = pd.to_numeric(values, errors="coerce")
    df = pd.DataFrame({"month": t.dt.month, "y": m}).dropna()
    if df.empty:
        return float("nan")
    monthly = df.groupby("month", sort=True)["y"].mean()
    if monthly.empty:
        return float("nan")
    return float(monthly.max() - monthly.min())


def compute_variability_score(
    daily_df: pd.DataFrame,
    config: ScoringConfig,
    *,
    station_col: str = COL_STATION_ID,
    date_col: str = COL_DATE,
    value_col: str = COL_CONCENTRATION,
) -> pd.Series:
    """
    ``V_s`` from std, IQR, and a simple seasonal amplitude of daily concentrations.
    """
    if daily_df.empty:
        return pd.Series(dtype=float)
    lo, hi = config.normalize_range
    rows: list[dict[str, float | str]] = []
    for sid, g in daily_df.groupby(station_col, sort=False):
        x = pd.to_numeric(g[value_col], errors="coerce").dropna()
        if x.empty:
            rows.append({"station_id": str(sid), "std": np.nan, "iqr": np.nan, "seasonal": np.nan})
            continue
        q75, q25 = x.quantile(0.75), x.quantile(0.25)
        amp = _seasonal_amplitude(g[date_col], g[value_col])
        rows.append(
            {
                "station_id": str(sid),
                "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "iqr": float(q75 - q25),
                "seasonal": float(amp) if not np.isnan(amp) else 0.0,
            }
        )
    agg = pd.DataFrame(rows)
    w = config.variability
    ns = minmax_series(agg["std"], feature_low=lo, feature_high=hi).to_numpy()
    ni = minmax_series(agg["iqr"], feature_low=lo, feature_high=hi).to_numpy()
    nz = minmax_series(agg["seasonal"], feature_low=lo, feature_high=hi).to_numpy()
    vals = w.rho_std * ns + w.rho_iqr * ni + w.rho_seasonal * nz
    return pd.Series(vals, index=agg["station_id"].values, name="V_s")

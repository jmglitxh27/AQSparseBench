"""Regional background / representativeness score."""

from __future__ import annotations

import pandas as pd

from aqsparsebench.config import ScoringConfig
from aqsparsebench.features._common import minmax_series
from aqsparsebench.preprocess.canonical import COL_CONCENTRATION, COL_STATION_ID


def compute_background_score(
    daily_df: pd.DataFrame,
    config: ScoringConfig,
    *,
    station_col: str = COL_STATION_ID,
    value_col: str = COL_CONCENTRATION,
) -> pd.Series:
    """
    ``B_s``: higher when the station's overall mean concentration is close to the
    regional mean (average of per-station means).
    """
    if daily_df.empty:
        return pd.Series(dtype=float)
    lo, hi = config.normalize_range
    means = daily_df.groupby(station_col, sort=False)[value_col].mean()
    region = float(means.mean())
    dev = (means - region).abs()
    # Smaller deviation -> higher score after min–max flip
    inv = -dev
    b = minmax_series(inv, feature_low=lo, feature_high=hi)
    b.index = means.index.astype(str)
    return b.rename("B_s")

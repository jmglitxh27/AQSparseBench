"""Shared normalization helpers for station-level score components."""

from __future__ import annotations

import pandas as pd


def minmax_series(
    s: pd.Series,
    *,
    feature_low: float = 0.0,
    feature_high: float = 1.0,
) -> pd.Series:
    """
    Per-column min–max to ``[feature_low, feature_high]``.

    Constant or all-NaN inputs map to the midpoint of the feature range.
    """
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = feature_low, feature_high
    mid = lo + (hi - lo) / 2.0
    xmin, xmax = x.min(), x.max()
    if pd.isna(xmin) or pd.isna(xmax) or xmin == xmax:
        return pd.Series(mid, index=x.index, dtype=float)
    return lo + (x - xmin) * (hi - lo) / (xmax - xmin)

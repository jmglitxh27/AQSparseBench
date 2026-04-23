"""Column scaling helpers (used later by feature modules)."""

from __future__ import annotations

import pandas as pd


def minmax_scale_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Min-max scale selected numeric columns in-place on a **copy**."""
    if df.empty:
        return df.copy()
    lo, hi = feature_range
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        xmin, xmax = x.min(), x.max()
        if pd.isna(xmin) or pd.isna(xmax) or xmax == xmin:
            out[c] = lo
        else:
            out[c] = lo + (x - xmin) * (hi - lo) / (xmax - xmin)
    return out

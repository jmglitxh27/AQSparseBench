"""Gap filling strategies for daily aligned panels."""

from __future__ import annotations

import pandas as pd

from aqsparsebench.preprocess.canonical import COL_CONCENTRATION, COL_DATE, COL_STATION_ID


def interpolate_time(
    df: pd.DataFrame,
    *,
    station_col: str = COL_STATION_ID,
    date_col: str = COL_DATE,
    value_col: str = COL_CONCENTRATION,
    limit_direction: str = "both",
) -> pd.DataFrame:
    """
    Per-station time-based interpolation on a sorted daily index.

    Requires contiguous calendar rows (e.g. output of :func:`~aqsparsebench.preprocess.align.align_daily`).
    Preserves non-target columns (e.g. merged weather) row-aligned within each station.
    """
    if df.empty:
        return df.copy()
    out_parts: list[pd.DataFrame] = []
    for _, g in df.groupby(station_col, sort=False):
        block = g.sort_values(date_col).copy()
        t = pd.to_datetime(block[date_col])
        filled = (
            pd.Series(block[value_col].to_numpy(), index=t)
            .interpolate(method="time", limit_direction=limit_direction)
            .to_numpy()
        )
        block[value_col] = filled
        out_parts.append(block)
    return pd.concat(out_parts, ignore_index=True)


def seasonal_mean_fill(
    df: pd.DataFrame,
    *,
    station_col: str = COL_STATION_ID,
    date_col: str = COL_DATE,
    value_col: str = COL_CONCENTRATION,
) -> pd.DataFrame:
    """
    Fill missing ``value_col`` with the station's mean for that calendar month-day across years.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["_md"] = out[date_col].dt.month * 100 + out[date_col].dt.day
    fill_vals = (
        out.dropna(subset=[value_col])
        .groupby([station_col, "_md"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "_seasonal_mean"})
    )
    out = out.merge(fill_vals, on=[station_col, "_md"], how="left")
    m = out[value_col].isna() & out["_seasonal_mean"].notna()
    out.loc[m, value_col] = out.loc[m, "_seasonal_mean"]
    out = out.drop(columns=["_md", "_seasonal_mean"])
    return out

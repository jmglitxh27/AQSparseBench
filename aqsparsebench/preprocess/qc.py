"""Lightweight quality checks on station tables and panels."""

from __future__ import annotations

import pandas as pd

from aqsparsebench.preprocess.canonical import (
    COL_CONCENTRATION,
    COL_DATE,
    COL_LATITUDE,
    COL_LONGITUDE,
    COL_STATION_ID,
)


def drop_invalid_coordinates(
    df: pd.DataFrame,
    *,
    lat_col: str = COL_LATITUDE,
    lon_col: str = COL_LONGITUDE,
) -> pd.DataFrame:
    """Remove rows with missing or out-of-range lat/lon."""
    if df.empty:
        return df.copy()
    out = df.copy()
    lat = pd.to_numeric(out[lat_col], errors="coerce")
    lon = pd.to_numeric(out[lon_col], errors="coerce")
    ok = lat.between(-90, 90) & lon.between(-180, 180) & lat.notna() & lon.notna()
    return out.loc[ok].reset_index(drop=True)


def drop_short_station_series(
    daily_df: pd.DataFrame,
    *,
    station_col: str = COL_STATION_ID,
    date_col: str = COL_DATE,
    value_col: str = COL_CONCENTRATION,
    min_valid_days: int = 30,
) -> pd.DataFrame:
    """Drop stations with fewer than ``min_valid_days`` non-null measurements."""
    if daily_df.empty:
        return daily_df.copy()
    counts = daily_df.groupby(station_col)[value_col].count()
    keep = counts[counts >= min_valid_days].index
    return daily_df[daily_df[station_col].isin(keep)].reset_index(drop=True)


def drop_duplicate_station_ids(monitors_df: pd.DataFrame, *, station_col: str = COL_STATION_ID) -> pd.DataFrame:
    return monitors_df.drop_duplicates(subset=[station_col], keep="last").reset_index(drop=True)

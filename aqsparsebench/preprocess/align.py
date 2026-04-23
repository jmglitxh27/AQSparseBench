"""Daily time alignment and joining exogenous (e.g. weather) series."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from aqsparsebench.preprocess.canonical import COL_CONCENTRATION, COL_DATE, COL_STATION_ID


def align_daily(
    df: pd.DataFrame,
    *,
    station_col: str = COL_STATION_ID,
    date_col: str = COL_DATE,
    value_col: str = COL_CONCENTRATION,
    calendar: Literal["union", "intersection"] = "union",
) -> pd.DataFrame:
    """
    Expand to one row per calendar day per station.

    * ``union`` — every station uses the same global date range (min to max in ``df``).
    * ``intersection`` — keep only dates where **every** station has at least one row in ``df``.
    """
    if df.empty:
        return df.copy()
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col]).dt.normalize()

    if calendar == "union":
        dmin, dmax = work[date_col].min(), work[date_col].max()
        all_dates = pd.date_range(dmin, dmax, freq="D")
        parts: list[pd.DataFrame] = []
        for sid, g in work.groupby(station_col, sort=False):
            g = g.drop_duplicates(subset=[date_col], keep="last").set_index(date_col)[value_col]
            s = g.reindex(all_dates)
            block = s.reset_index().rename(columns={"index": date_col, value_col: value_col})
            block[station_col] = sid
            block["missing_flag"] = block[value_col].isna()
            parts.append(block)
        return pd.concat(parts, ignore_index=True)

    # intersection: dates present for all stations
    by_day = work.groupby(date_col)[station_col].nunique()
    n_stations = work[station_col].nunique()
    common_days = by_day[by_day == n_stations].index
    if len(common_days) == 0:
        return pd.DataFrame(columns=[station_col, date_col, value_col, "missing_flag"])
    trimmed = work[work[date_col].isin(common_days)].copy()
    trimmed["missing_flag"] = trimmed[value_col].isna()
    return trimmed.sort_values([station_col, date_col]).reset_index(drop=True)


def merge_exogenous(
    aq_df: pd.DataFrame,
    exo_df: pd.DataFrame,
    *,
    station_col: str = COL_STATION_ID,
    date_col: str = COL_DATE,
    how: Literal["left", "inner", "outer"] = "left",
    exo_station_col: str | None = None,
    exo_date_col: str | None = None,
) -> pd.DataFrame:
    """
    Left-join auxiliary columns (weather, population, etc.) on ``station_id`` + ``date``.

    ``exo_df`` must include the same ``station_col`` / ``date_col`` names unless remapped.
    """
    if aq_df.empty:
        return aq_df.copy()
    left = aq_df.copy()
    left[date_col] = pd.to_datetime(left[date_col]).dt.normalize()
    right = exo_df.copy()
    esc = exo_station_col or station_col
    edc = exo_date_col or date_col
    right[edc] = pd.to_datetime(right[edc]).dt.normalize()
    rename_r: dict[str, str] = {}
    if esc != station_col:
        rename_r[esc] = station_col
    if edc != date_col:
        rename_r[edc] = date_col
    if rename_r:
        right = right.rename(columns=rename_r)
    # avoid duplicate column names on merge (keep left AQ value col as-is)
    overlap = [c for c in right.columns if c in left.columns and c not in (station_col, date_col)]
    if overlap:
        right = right.drop(columns=overlap)
    return left.merge(right, on=[station_col, date_col], how=how, suffixes=("", "_exo"))

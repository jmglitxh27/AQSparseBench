"""Filter and subsample AQS-style monitor catalogs before normalization."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from aqsparsebench.preprocess.canonical import COL_STATION_ID
from aqsparsebench.preprocess.epa_normalize import _col, derive_aqs_station_id_series


def _as_timestamp(d: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(d, pd.Timestamp):
        return pd.Timestamp(d).normalize()
    ts = pd.to_datetime(d, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date value: {d!r}")
    return pd.Timestamp(ts).normalize()


def filter_monitors_operational_span(
    monitors_df: pd.DataFrame,
    period_start: str | pd.Timestamp,
    period_end: str | pd.Timestamp,
    *,
    mode: Literal["continuous", "overlap"] = "continuous",
) -> pd.DataFrame:
    """
    Keep monitors using EPA ``open_date`` / ``close_date`` when those columns exist.

    * ``continuous`` — site was open on ``period_start`` and not closed before ``period_end``
      (inclusive calendar-day interpretation on parsed dates).
    * ``overlap`` — site's lifetime intersects ``[period_start, period_end]`` at all.

    Rows without a parsable ``open_date`` are dropped when filtering is applied. If neither
    column exists, returns ``monitors_df`` unchanged and callers should not rely on filtering.
    """
    if monitors_df.empty:
        return monitors_df
    open_c = _col(monitors_df, "open_date")
    close_c = _col(monitors_df, "close_date")
    if open_c is None and close_c is None:
        return monitors_df

    ps = _as_timestamp(period_start)
    pe = _as_timestamp(period_end)
    if ps > pe:
        raise ValueError(f"period_start {period_start!r} must be <= period_end {period_end!r}")

    work = monitors_df.copy()
    open_dt = pd.to_datetime(work[open_c], errors="coerce") if open_c else pd.Series(pd.NaT, index=work.index)
    close_dt = (
        pd.to_datetime(work[close_c], errors="coerce") if close_c else pd.Series(pd.NaT, index=work.index)
    )

    if mode == "continuous":
        if open_c is None:
            return monitors_df
        mask_open = open_dt.notna() & (open_dt <= ps)
        mask_still = close_dt.isna() | (close_dt >= pe)
        mask = mask_open & mask_still
    else:
        # overlap with [ps, pe]
        if open_c is None:
            return monitors_df
        open_ok = open_dt.isna() | (open_dt <= pe)
        close_ok = close_dt.isna() | (close_dt >= ps)
        mask = open_ok & close_ok

    return work.loc[mask].reset_index(drop=True)


def subsample_monitors_to_max_stations(
    monitors_df: pd.DataFrame,
    max_stations: int,
    *,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Keep at most ``max_stations`` distinct sites (uniform over ``station_id``)."""
    if monitors_df.empty:
        return monitors_df.copy()
    if max_stations <= 0:
        return monitors_df.copy()
    sids = derive_aqs_station_id_series(monitors_df)
    unique = pd.unique(sids.astype(str))
    if len(unique) <= max_stations:
        return monitors_df.copy()
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(unique, size=max_stations, replace=False)
    keep = set(chosen.tolist())
    return monitors_df.loc[sids.astype(str).isin(keep)].reset_index(drop=True)


def restrict_daily_to_station_ids(daily_df: pd.DataFrame, station_ids: set[str] | frozenset[str]) -> pd.DataFrame:
    """Drop daily rows whose ``station_id`` is not in ``station_ids``."""
    if daily_df.empty:
        return daily_df.copy()
    if not station_ids:
        return daily_df.iloc[0:0].copy()
    if COL_STATION_ID not in daily_df.columns:
        raise ValueError(f"daily_df must contain {COL_STATION_ID!r}")
    s = daily_df[COL_STATION_ID].astype(str)
    return daily_df.loc[s.isin(station_ids)].reset_index(drop=True)

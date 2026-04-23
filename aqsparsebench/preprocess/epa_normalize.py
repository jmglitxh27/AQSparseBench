"""Normalize EPA AQS API data frames to :mod:`aqsparsebench.preprocess.canonical` columns."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from aqsparsebench.preprocess.canonical import (
    COL_CONCENTRATION,
    COL_DATE,
    COL_LATITUDE,
    COL_LONGITUDE,
    COL_STATION_ID,
)


def _lower_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).lower(): str(c) for c in columns}


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    m = _lower_map(df.columns)
    for cand in candidates:
        key = cand.lower()
        if key in m:
            return m[key]
    return None


def looks_like_aqs_monitors(df: pd.DataFrame) -> bool:
    """Heuristic: AQS monitor listings use state/county/site codes or EPA-style site ids."""
    if df.empty:
        return False
    cols = {str(c).lower() for c in df.columns}
    if {"state_code", "county_code", "site_number"}.issubset(cols):
        return True
    if "state_code" in cols and "latitude" in cols:
        return True
    if COL_STATION_ID in cols and any(
        re.match(r"^\d{2}_\d{3}_\d{4}$", str(x)) for x in df[COL_STATION_ID].dropna().head(5)
    ):
        return True
    return False


def looks_like_aqs_daily(df: pd.DataFrame) -> bool:
    """Heuristic: AQS daily summaries expose ``date_local`` and ``arithmetic_mean`` style fields."""
    if df.empty:
        return False
    cols = {str(c).lower() for c in df.columns}
    if "date_local" in cols and ("arithmetic_mean" in cols or "arithmetic mean" in cols):
        return True
    if "date_local" in cols and "sample_measurement" in cols:
        return True
    return False


def normalize_aqs_monitors_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build ``station_id``, ``latitude``, ``longitude`` from typical AQS monitor columns."""
    if df.empty:
        return pd.DataFrame(columns=[COL_STATION_ID, COL_LATITUDE, COL_LONGITUDE])
    out = df.copy()
    lat_c = _col(out, "latitude")
    lon_c = _col(out, "longitude")
    if lat_c is None or lon_c is None:
        raise ValueError("AQS monitors DataFrame must include latitude/longitude columns")

    sid_c = _col(out, COL_STATION_ID, "station_id")
    if sid_c is not None:
        station_series = out[sid_c].astype(str)
    else:
        sc = _col(out, "state_code")
        cc = _col(out, "county_code")
        sn = _col(out, "site_number")
        if not (sc and cc and sn):
            raise ValueError("Cannot derive station_id: need station_id or state/county/site columns")
        station_series = (
            out[sc].astype(str).str.zfill(2)
            + "_"
            + out[cc].astype(str).str.zfill(3)
            + "_"
            + out[sn].astype(str).str.zfill(4)
        )

    name_c = _col(out, "site_name", "local_site_name")
    canon = pd.DataFrame(
        {
            COL_STATION_ID: station_series,
            COL_LATITUDE: pd.to_numeric(out[lat_c], errors="coerce"),
            COL_LONGITUDE: pd.to_numeric(out[lon_c], errors="coerce"),
        }
    )
    if name_c:
        canon[name_c] = out[name_c].values
    return canon.drop_duplicates(subset=[COL_STATION_ID], keep="last").reset_index(drop=True)


def normalize_aqs_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build long daily panel: ``station_id``, ``date``, ``concentration`` (from arithmetic mean / sample).

    Multiple POCs or methods for the same site-day are averaged.
    """
    if df.empty:
        return pd.DataFrame(columns=[COL_STATION_ID, COL_DATE, COL_CONCENTRATION])

    out = df.copy()
    date_c = _col(out, "date_local", "date", COL_DATE)
    if date_c is None:
        raise ValueError("AQS daily DataFrame needs date_local (or date) column")

    val_c = _col(out, "arithmetic_mean", "arithmetic mean", "sample_measurement", "first_max_value")
    if val_c is None:
        raise ValueError(
            "AQS daily DataFrame needs arithmetic_mean, sample_measurement, or first_max_value"
        )

    sid_c = _col(out, COL_STATION_ID, "station_id")
    if sid_c is not None:
        out[COL_STATION_ID] = out[sid_c].astype(str)
    else:
        sc = _col(out, "state_code")
        cc = _col(out, "county_code")
        sn = _col(out, "site_number")
        if not (sc and cc and sn):
            raise ValueError("Cannot derive station_id for daily rows")
        out[COL_STATION_ID] = (
            out[sc].astype(str).str.zfill(2)
            + "_"
            + out[cc].astype(str).str.zfill(3)
            + "_"
            + out[sn].astype(str).str.zfill(4)
        )

    out[COL_DATE] = pd.to_datetime(out[date_c]).dt.normalize()
    out[COL_CONCENTRATION] = pd.to_numeric(out[val_c], errors="coerce")

    slim = out[[COL_STATION_ID, COL_DATE, COL_CONCENTRATION]].copy()
    slim = (
        slim.groupby([COL_STATION_ID, COL_DATE], as_index=False)[COL_CONCENTRATION]
        .mean()
        .sort_values([COL_STATION_ID, COL_DATE])
        .reset_index(drop=True)
    )
    return slim


def coerce_custom_daily_to_canonical(
    df: pd.DataFrame,
    *,
    station_col: str,
    date_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Map arbitrary column names to canonical ``station_id`` / ``date`` / ``concentration``."""
    if df.empty:
        return pd.DataFrame(columns=[COL_STATION_ID, COL_DATE, COL_CONCENTRATION])
    out = df[[station_col, date_col, value_col]].rename(
        columns={station_col: COL_STATION_ID, date_col: COL_DATE, value_col: COL_CONCENTRATION}
    )
    out[COL_DATE] = pd.to_datetime(out[COL_DATE]).dt.normalize()
    out[COL_STATION_ID] = out[COL_STATION_ID].astype(str)
    out[COL_CONCENTRATION] = pd.to_numeric(out[COL_CONCENTRATION], errors="coerce")
    return out.groupby([COL_STATION_ID, COL_DATE], as_index=False)[COL_CONCENTRATION].mean()


def coerce_custom_monitors_to_canonical(
    df: pd.DataFrame,
    *,
    station_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    out = df[[station_col, lat_col, lon_col]].rename(
        columns={station_col: COL_STATION_ID, lat_col: COL_LATITUDE, lon_col: COL_LONGITUDE}
    )
    out[COL_STATION_ID] = out[COL_STATION_ID].astype(str)
    out[COL_LATITUDE] = pd.to_numeric(out[COL_LATITUDE], errors="coerce")
    out[COL_LONGITUDE] = pd.to_numeric(out[COL_LONGITUDE], errors="coerce")
    return out.drop_duplicates(subset=[COL_STATION_ID], keep="last").reset_index(drop=True)

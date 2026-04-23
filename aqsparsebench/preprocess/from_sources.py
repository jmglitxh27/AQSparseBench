"""
High-level loaders that prefer **US EPA AQS** ergonomics but accept **custom tables**.

Typical US workflow::

    from aqsparsebench.io import DataSources
    from aqsparsebench.preprocess.from_sources import load_air_quality_for_preprocess

    sources = DataSources.from_us_epa_defaults(api)
    monitors, daily = load_air_quality_for_preprocess(
        sources, region, pollutant=\"pm25\", years=[2020, 2021]
    )

Custom / non-US workflow — pass your own frames and column names::

    monitors, daily = load_air_quality_for_preprocess(
        sources,
        region,
        pollutant=\"pm25\",
        years=[2020],
        monitors_df=my_mon,
        daily_df=my_day,
        normalization=\"none\",
        custom_monitor_columns=(\"id\", \"lat\", \"lon\"),
        custom_daily_columns=(\"id\", \"day\", \"pm25\"),
    )
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from aqsparsebench.io.protocols import AirQualitySource
from aqsparsebench.io.sources import DataSources
from aqsparsebench.preprocess.canonical import (
    COL_CONCENTRATION,
    COL_DATE,
    COL_LATITUDE,
    COL_LONGITUDE,
    COL_STATION_ID,
    REQUIRED_DAILY_COLUMNS,
    REQUIRED_MONITOR_COLUMNS,
)
from aqsparsebench.preprocess.epa_normalize import (
    coerce_custom_daily_to_canonical,
    coerce_custom_monitors_to_canonical,
    looks_like_aqs_daily,
    looks_like_aqs_monitors,
    normalize_aqs_daily_df,
    normalize_aqs_monitors_df,
)
from aqsparsebench.types import RegionSpec


def _validate_canonical_daily(df: pd.DataFrame) -> None:
    missing = REQUIRED_DAILY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Daily table missing columns {sorted(missing)}. "
            f"Expected at least {sorted(REQUIRED_DAILY_COLUMNS)}."
        )


def _validate_canonical_monitors(df: pd.DataFrame) -> None:
    missing = REQUIRED_MONITOR_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Monitor table missing columns {sorted(missing)}. "
            f"Expected at least {sorted(REQUIRED_MONITOR_COLUMNS)}."
        )


def resolve_air_quality_normalization(
    air_quality: AirQualitySource,
    mode: Literal["auto", "epa_aqs", "none"],
) -> Literal["epa_aqs", "none"]:
    """Resolve ``auto`` using the registered air-quality client; does not inspect DataFrame shape."""
    if mode == "epa_aqs":
        return "epa_aqs"
    if mode == "none":
        return "none"
    sid = getattr(air_quality, "source_id", "")
    if sid == "epa_aqs":
        return "epa_aqs"
    try:
        from aqsparsebench.io.aqs_api import AQSClient

        if isinstance(air_quality, AQSClient):
            return "epa_aqs"
    except Exception:
        pass
    return "none"


def load_air_quality_for_preprocess(
    sources: DataSources,
    region: RegionSpec,
    *,
    pollutant: str,
    years: list[int],
    param: str | None = None,
    normalization: Literal["auto", "epa_aqs", "none"] = "auto",
    monitors_df: pd.DataFrame | None = None,
    daily_df: pd.DataFrame | None = None,
    custom_monitor_columns: tuple[str, str, str] | None = None,
    custom_daily_columns: tuple[str, str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return canonical monitor + daily long tables.

    * **EPA-first** — ``normalization=\"auto\"`` applies AQS normalization when the active
      source is :class:`~aqsparsebench.io.aqs_api.AQSClient` (or ``source_id == \"epa_aqs\"``),
      or when the fetched frames look like raw AQS (``date_local``, ``state_code``, …).
    * **Bring-your-own-data** — pass ``monitors_df`` / ``daily_df`` to skip fetches, and either
      set ``normalization=\"none\"`` with ``custom_*_columns``, or pre-shape columns to the
      canonical names.
    """
    aq = sources.air_quality
    mon = monitors_df if monitors_df is not None else aq.fetch_monitor_catalog(region, pollutant=pollutant, years=years, param=param)
    day = daily_df if daily_df is not None else aq.fetch_daily_air_quality(region, pollutant=pollutant, years=years, param=param)

    if custom_daily_columns is not None:
        sc, dc, vc = custom_daily_columns
        day = coerce_custom_daily_to_canonical(day, station_col=sc, date_col=dc, value_col=vc)
    if custom_monitor_columns is not None:
        sc, latc, lonc = custom_monitor_columns
        mon = coerce_custom_monitors_to_canonical(mon, station_col=sc, lat_col=latc, lon_col=lonc)

    if custom_monitor_columns is None:
        epa_from_client = False
        if normalization == "epa_aqs":
            epa_from_client = True
        elif normalization == "auto":
            epa_from_client = resolve_air_quality_normalization(aq, "auto") == "epa_aqs"
        if epa_from_client or (normalization == "auto" and looks_like_aqs_monitors(mon)):
            mon = normalize_aqs_monitors_df(mon)

    if custom_daily_columns is None:
        epa_from_client = False
        if normalization == "epa_aqs":
            epa_from_client = True
        elif normalization == "auto":
            epa_from_client = resolve_air_quality_normalization(aq, "auto") == "epa_aqs"
        if epa_from_client or (normalization == "auto" and looks_like_aqs_daily(day)):
            day = normalize_aqs_daily_df(day)

    _validate_canonical_monitors(mon)
    _validate_canonical_daily(day)
    return mon, day


def load_weather_for_monitors(
    sources: DataSources,
    monitors_df: pd.DataFrame,
    *,
    years: list[int],
    lat_col: str = COL_LATITUDE,
    lon_col: str = COL_LONGITUDE,
    site_id_col: str | None = COL_STATION_ID,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Pull Open-Meteo (or any :class:`~aqsparsebench.io.protocols.WeatherArchiveSource`) for monitor coordinates."""
    if monitors_df.empty:
        return pd.DataFrame()
    return sources.weather.fetch_daily_meteorology_for_sites(
        monitors_df,
        lat_col=lat_col,
        lon_col=lon_col,
        site_id_col=site_id_col,
        years=years,
        variables=variables,
    )


def build_merged_daily_panel(
    daily_canonical: pd.DataFrame,
    weather_long: pd.DataFrame,
) -> pd.DataFrame:
    """Merge AQ daily with weather on ``station_id`` + ``date`` (UTC-normalized)."""
    from aqsparsebench.preprocess.align import merge_exogenous

    return merge_exogenous(daily_canonical, weather_long, how="left")

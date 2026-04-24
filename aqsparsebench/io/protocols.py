"""
Pluggable IO contracts.

Swap **air-quality** backends (EPA AQS, EEA, OpenAQ exports, CSV fixtures) and keep
**weather** on Open-Meteo or another :class:`WeatherArchiveSource` implementation.
:class:`~aqsparsebench.types.RegionSpec` bbox mode is WGS84 and is not US-specific; only
EPA-specific query shapes live in :class:`~aqsparsebench.io.aqs_api.AQSClient`.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

from aqsparsebench.types import RegionSpec


@runtime_checkable
class AirQualitySource(Protocol):
    """Any provider that returns monitor metadata and daily concentration-style tables."""

    @property
    def source_id(self) -> str:
        """Stable short id, e.g. ``\"epa_aqs\"`` or ``\"csv_fixture\"``."""

    def fetch_monitor_catalog(
        self,
        region: RegionSpec,
        *,
        pollutant: str,
        years: list[int],
        param: str | None = None,
        bdate: str | None = None,
        edate: str | None = None,
    ) -> pd.DataFrame:
        """Columns depend on provider; include ``latitude``, ``longitude``, ``station_id`` when possible."""

    def fetch_daily_air_quality(
        self,
        region: RegionSpec,
        *,
        pollutant: str,
        years: list[int],
        param: str | None = None,
        bdate: str | None = None,
        edate: str | None = None,
    ) -> pd.DataFrame:
        """Daily (or finest available) measurements keyed by site + local date."""


@runtime_checkable
class WeatherArchiveSource(Protocol):
    """Historical daily meteorology (global grid; coordinates in WGS84)."""

    @property
    def source_id(self) -> str:
        """Stable short id, e.g. ``\"open_meteo_archive\"``."""

    def fetch_daily_meteorology(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """One row per calendar day; includes at least ``date`` plus requested variables."""

    def fetch_daily_meteorology_for_sites(
        self,
        sites: pd.DataFrame,
        *,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        site_id_col: str | None = "station_id",
        years: list[int],
        start_date: str | None = None,
        end_date: str | None = None,
        variables: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Long-form table: ``station_id``, ``latitude``, ``longitude``, ``date``, variables."""


@runtime_checkable
class PopulationContextSource(Protocol):
    """Optional exposure / population context (Census, WorldPop, static rasters, etc.)."""

    @property
    def source_id(self) -> str: ...

    def fetch_population_context(
        self,
        sites: pd.DataFrame,
        *,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        site_id_col: str | None = "station_id",
    ) -> pd.DataFrame:
        """Site-level population or density features (implementation-defined columns)."""

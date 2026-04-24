"""In-memory :class:`~aqsparsebench.io.protocols.AirQualitySource` for tests and non-US fixtures."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aqsparsebench.types import RegionSpec


@dataclass
class DataFrameAirQualitySource:
    """
    Wrap pre-built monitor and daily tables (e.g. CSV/Parquet loaded outside the library).

    ``fetch_*`` methods ignore ``region`` / ``pollutant`` / ``years`` unless you subclass
    and add filtering—intended for **controlled benchmarks** and unit tests.
    """

    monitors_df: pd.DataFrame
    daily_df: pd.DataFrame
    provider_label: str = "dataframe_air_quality"

    @property
    def source_id(self) -> str:
        return self.provider_label

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
        return self.monitors_df.copy()

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
        return self.daily_df.copy()


class NullPopulationSource:
    """Placeholder population backend (minimal ids only) until Census / raster adapters land."""

    @property
    def source_id(self) -> str:
        return "null_population"

    def fetch_population_context(
        self,
        sites: pd.DataFrame,
        *,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        site_id_col: str | None = "station_id",
    ) -> pd.DataFrame:
        if sites.empty:
            return pd.DataFrame()
        idc = site_id_col if site_id_col and site_id_col in sites.columns else None
        if idc:
            return pd.DataFrame({idc: sites[idc].astype(str)})
        return pd.DataFrame(index=sites.index)


class DataFramePopulationSource:
    """In-memory population context for notebooks/tests (bring-your-own population table)."""

    def __init__(self, population_df: pd.DataFrame, *, provider_label: str = "dataframe_population") -> None:
        self.population_df = population_df.copy()
        self.provider_label = provider_label

    @property
    def source_id(self) -> str:
        return self.provider_label

    def fetch_population_context(
        self,
        sites: pd.DataFrame,
        *,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        site_id_col: str | None = "station_id",
    ) -> pd.DataFrame:
        # Expected: station_id column exists in population_df.
        if self.population_df.empty:
            return pd.DataFrame()
        if site_id_col and site_id_col in sites.columns and site_id_col in self.population_df.columns:
            keep = set(sites[site_id_col].astype(str).unique())
            return self.population_df[self.population_df[site_id_col].astype(str).isin(keep)].copy()
        return self.population_df.copy()

"""Compose pluggable air-quality, weather, and (optional) population backends."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Literal

import pandas as pd

from aqsparsebench.config import ApiConfig
from aqsparsebench.io.protocols import AirQualitySource, PopulationContextSource, WeatherArchiveSource
from aqsparsebench.types import RegionSpec


@dataclass
class DataSources:
    """
    Bundle used by the benchmark builder so IO stays swappable.

    Use :meth:`from_us_epa_defaults` for EPA AQS + global Open-Meteo, or :meth:`from_clients`
    with a custom :class:`DataFrameAirQualitySource` for non-US / proprietary AQ tables.
    """

    air_quality: AirQualitySource
    weather: WeatherArchiveSource
    population: PopulationContextSource | None = None

    def uses_epa_aqs_client(self) -> bool:
        """True when the bundled air-quality client is :class:`~aqsparsebench.io.aqs_api.AQSClient`."""
        from aqsparsebench.io.aqs_api import AQSClient

        return isinstance(self.air_quality, AQSClient)

    def load_air_quality_for_preprocess(
        self,
        region: RegionSpec,
        *,
        pollutant: str,
        years: list[int],
        normalization: Literal["auto", "epa_aqs", "none"] | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Shorthand for :func:`aqsparsebench.preprocess.from_sources.load_air_quality_for_preprocess`.

        When ``normalization`` is omitted, defaults to ``\"auto\"`` — EPA-friendly for
        :meth:`from_us_epa_defaults` bundles, heuristic for custom sources.
        """
        from aqsparsebench.preprocess.from_sources import load_air_quality_for_preprocess

        norm = normalization if normalization is not None else "auto"
        return load_air_quality_for_preprocess(
            self,
            region,
            pollutant=pollutant,
            years=years,
            normalization=norm,
            **kwargs,
        )

    @staticmethod
    def from_us_epa_defaults(
        api: ApiConfig,
        *,
        aqs_base_url: str | None = None,
    ) -> DataSources:
        """EPA AQS (US regulatory AQ) + Open-Meteo archive (global weather)."""
        from aqsparsebench.io.aqs_api import AQSClient, DEFAULT_AQS_BASE
        from aqsparsebench.io.memory_sources import NullPopulationSource
        from aqsparsebench.io.weather_api import OpenMeteoClient

        aq = AQSClient(api, base_url=aqs_base_url or DEFAULT_AQS_BASE)
        wx = OpenMeteoClient(api)
        return DataSources(air_quality=aq, weather=wx, population=NullPopulationSource())

    @staticmethod
    def from_clients(
        *,
        air_quality: AirQualitySource,
        weather: WeatherArchiveSource,
        population: PopulationContextSource | None = None,
    ) -> DataSources:
        """Wire arbitrary implementations (OpenAQ adapter, CSV fixtures, enterprise weather API, …)."""
        return DataSources(air_quality=air_quality, weather=weather, population=population)

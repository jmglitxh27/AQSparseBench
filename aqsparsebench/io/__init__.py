"""API clients, protocols, and on-disk cache."""

from aqsparsebench.io.aqs_api import (
    AQSAPIError,
    AQSClient,
    DEFAULT_AQS_BASE,
    DEFAULT_POLLUTANT_PARAMS,
    is_available,
    resolve_aqs_param_codes,
)
from aqsparsebench.io.cache import LocalCache, cache_key_from_request
from aqsparsebench.io.memory_sources import DataFrameAirQualitySource, NullPopulationSource
from aqsparsebench.io.protocols import AirQualitySource, PopulationContextSource, WeatherArchiveSource
from aqsparsebench.io.sources import DataSources
from aqsparsebench.io.weather_api import (
    DEFAULT_ARCHIVE_DAILY_VARIABLES,
    DEFAULT_OPEN_METEO_ARCHIVE_URL,
    OpenMeteoAPIError,
    OpenMeteoClient,
)

__all__ = [
    "AQSAPIError",
    "AQSClient",
    "AirQualitySource",
    "DEFAULT_AQS_BASE",
    "DEFAULT_ARCHIVE_DAILY_VARIABLES",
    "DEFAULT_OPEN_METEO_ARCHIVE_URL",
    "DEFAULT_POLLUTANT_PARAMS",
    "DataFrameAirQualitySource",
    "DataSources",
    "LocalCache",
    "NullPopulationSource",
    "OpenMeteoAPIError",
    "OpenMeteoClient",
    "PopulationContextSource",
    "WeatherArchiveSource",
    "cache_key_from_request",
    "is_available",
    "resolve_aqs_param_codes",
]

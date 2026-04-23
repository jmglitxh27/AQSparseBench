import pandas as pd

from aqsparsebench.config import ApiConfig
from aqsparsebench.io.aqs_api import AQSClient
from aqsparsebench.io.memory_sources import DataFrameAirQualitySource, NullPopulationSource
from aqsparsebench.io.protocols import AirQualitySource, PopulationContextSource, WeatherArchiveSource
from aqsparsebench.io.sources import DataSources
from aqsparsebench.io.weather_api import OpenMeteoClient
from aqsparsebench.types import BoundingBox, RegionSpec


def test_protocol_isinstance_aqs_openmeteo() -> None:
    api = ApiConfig(aqs_email="a@b.c", aqs_key="k", aqs_request_sleep_seconds=0.0)
    aq = AQSClient(api)
    wx = OpenMeteoClient(api)
    assert isinstance(aq, AirQualitySource)
    assert isinstance(wx, WeatherArchiveSource)


def test_dataframe_air_quality_source_protocol() -> None:
    mon = pd.DataFrame({"station_id": ["x"], "latitude": [1.0], "longitude": [2.0]})
    day = pd.DataFrame({"date": ["2020-01-01"], "station_id": ["x"], "value": [3.0]})
    src = DataFrameAirQualitySource(mon, day, provider_label="fixture_eu")
    assert isinstance(src, AirQualitySource)
    assert src.source_id == "fixture_eu"
    region = RegionSpec.from_bbox("r", BoundingBox(-1.0, 1.0, -1.0, 1.0))
    assert len(src.fetch_monitor_catalog(region, pollutant="pm25", years=[2020])) == 1


def test_null_population_protocol() -> None:
    p = NullPopulationSource()
    assert isinstance(p, PopulationContextSource)


def test_data_sources_from_us_epa_defaults() -> None:
    api = ApiConfig(aqs_email="a@b.c", aqs_key="k", aqs_request_sleep_seconds=0.0)
    ds = DataSources.from_us_epa_defaults(api)
    assert isinstance(ds.air_quality, AQSClient)
    assert isinstance(ds.weather, OpenMeteoClient)
    assert ds.population is not None
    assert ds.uses_epa_aqs_client() is True


def test_data_sources_from_clients() -> None:
    api = ApiConfig()
    mon = pd.DataFrame()
    day = pd.DataFrame()
    aq = DataFrameAirQualitySource(mon, day)
    wx = OpenMeteoClient(api)
    ds = DataSources.from_clients(air_quality=aq, weather=wx, population=None)
    assert ds.population is None
    assert ds.uses_epa_aqs_client() is False

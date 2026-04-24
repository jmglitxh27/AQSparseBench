import pandas as pd
import pytest

from aqsparsebench.preprocess.from_sources import load_weather_for_monitors
from aqsparsebench.preprocess.canonical import COL_LATITUDE, COL_LONGITUDE, COL_STATION_ID
from aqsparsebench.io.sources import DataSources
from aqsparsebench.io.memory_sources import DataFrameAirQualitySource


class SpyWeather:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str | None, tuple[int, ...]]] = []

    @property
    def source_id(self) -> str:
        return "spy_weather"

    def fetch_daily_meteorology(self, *a, **k):
        raise NotImplementedError

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
        variables=None,
    ) -> pd.DataFrame:
        self.calls.append((start_date, end_date, tuple(years)))
        if sites.empty:
            return pd.DataFrame()
        # minimal long output
        return pd.DataFrame(
            {
                COL_STATION_ID: [str(sites.iloc[0][site_id_col])] if site_id_col else ["site_0"],
                COL_LATITUDE: [float(sites.iloc[0][lat_col])],
                COL_LONGITUDE: [float(sites.iloc[0][lon_col])],
                "date": [pd.Timestamp("2020-01-01")],
                "temperature_2m_mean": [1.0],
            }
        )


def test_load_weather_for_monitors_accepts_explicit_window() -> None:
    mon = pd.DataFrame(
        [{COL_STATION_ID: "a", COL_LATITUDE: 1.0, COL_LONGITUDE: -1.0}]
    )
    src = DataFrameAirQualitySource(pd.DataFrame(), pd.DataFrame(), provider_label="fixture")
    spy = SpyWeather()
    ds = DataSources.from_clients(air_quality=src, weather=spy)

    out = load_weather_for_monitors(ds, mon, years=[], start_date="2020-01-01", end_date="2020-01-31")
    assert not out.empty
    assert spy.calls == [("2020-01-01", "2020-01-31", tuple())]


def test_load_weather_for_monitors_requires_both_dates() -> None:
    mon = pd.DataFrame([{COL_STATION_ID: "a", COL_LATITUDE: 1.0, COL_LONGITUDE: -1.0}])
    src = DataFrameAirQualitySource(pd.DataFrame(), pd.DataFrame(), provider_label="fixture")
    spy = SpyWeather()
    ds = DataSources.from_clients(air_quality=src, weather=spy)

    with pytest.raises(ValueError):
        load_weather_for_monitors(ds, mon, years=[], start_date="2020-01-01", end_date=None)


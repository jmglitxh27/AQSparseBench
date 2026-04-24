import pandas as pd

from aqsparsebench.io.memory_sources import DataFrameAirQualitySource
from aqsparsebench.io.sources import DataSources
from aqsparsebench.preprocess.canonical import COL_DATE, COL_STATION_ID, COL_CONCENTRATION
from aqsparsebench.preprocess.from_sources import load_air_quality_for_preprocess
from aqsparsebench.preprocess.monitor_select import (
    filter_monitors_operational_span,
    restrict_daily_to_station_ids,
    subsample_monitors_to_max_stations,
)
from aqsparsebench.types import BoundingBox, RegionSpec


def test_filter_continuous_keeps_open_through_period() -> None:
    mon = pd.DataFrame(
        [
            {"state_code": "06", "county_code": "075", "site_number": "0001", "latitude": 1.0, "longitude": -1.0, "open_date": "1990-01-01", "close_date": None},
            {"state_code": "06", "county_code": "075", "site_number": "0002", "latitude": 2.0, "longitude": -2.0, "open_date": "2023-01-01", "close_date": None},
            {"state_code": "06", "county_code": "075", "site_number": "0003", "latitude": 3.0, "longitude": -3.0, "open_date": "1990-01-01", "close_date": "2021-06-01"},
        ]
    )
    out = filter_monitors_operational_span(mon, "2020-01-01", "2024-12-31", mode="continuous")
    sids = out["state_code"].astype(str).str.zfill(2) + "_" + out["county_code"].astype(str).str.zfill(3) + "_" + out["site_number"].astype(str).str.zfill(4)
    assert set(sids) == {"06_075_0001"}


def test_filter_overlap() -> None:
    mon = pd.DataFrame(
        [
            {"state_code": "06", "county_code": "075", "site_number": "0001", "latitude": 1.0, "longitude": -1.0, "open_date": "2023-06-01", "close_date": "2023-08-01"},
            {"state_code": "06", "county_code": "075", "site_number": "0002", "latitude": 2.0, "longitude": -2.0, "open_date": "1990-01-01", "close_date": "2019-01-01"},
        ]
    )
    out = filter_monitors_operational_span(mon, "2020-01-01", "2024-12-31", mode="overlap")
    sids = out["site_number"].astype(str)
    assert list(sids) == ["0001"]


def test_subsample_max_stations() -> None:
    mon = pd.DataFrame(
        [
            {"state_code": "06", "county_code": "075", "site_number": f"{i:04d}", "latitude": float(i), "longitude": -1.0}
            for i in range(10)
        ]
    )
    out = subsample_monitors_to_max_stations(mon, 3, random_state=42)
    assert out["site_number"].nunique() == 3


def test_restrict_daily() -> None:
    day = pd.DataFrame(
        {
            COL_STATION_ID: ["a", "b", "a"],
            COL_DATE: pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            COL_CONCENTRATION: [1.0, 2.0, 3.0],
        }
    )
    out = restrict_daily_to_station_ids(day, frozenset({"a"}))
    assert len(out) == 2
    assert set(out[COL_STATION_ID]) == {"a"}


def test_load_air_quality_restricts_daily_with_operational_filter() -> None:
    mon = pd.DataFrame(
        [
            {
                "state_code": "06",
                "county_code": "075",
                "site_number": "0001",
                "latitude": 1.0,
                "longitude": -1.0,
                "open_date": "1990-01-01",
                "close_date": None,
            },
            {
                "state_code": "06",
                "county_code": "075",
                "site_number": "0002",
                "latitude": 2.0,
                "longitude": -2.0,
                "open_date": "2023-01-01",
                "close_date": None,
            },
        ]
    )
    day = pd.DataFrame(
        [
            {"date_local": "2020-06-01", "state_code": "06", "county_code": "075", "site_number": "0001", "arithmetic_mean": 5.0},
            {"date_local": "2020-06-01", "state_code": "06", "county_code": "075", "site_number": "0002", "arithmetic_mean": 9.0},
        ]
    )
    region = RegionSpec.from_bbox("x", BoundingBox(0.0, 1.0, -2.0, -1.0))
    src = DataFrameAirQualitySource(mon, day, provider_label="fixture")
    ds = DataSources.from_clients(air_quality=src, weather=MagicWeather())
    m, d = load_air_quality_for_preprocess(
        ds,
        region,
        pollutant="pm25",
        years=[2020],
        monitors_operational_start="2020-01-01",
        monitors_operational_end="2024-12-31",
        monitor_operational_mode="continuous",
    )
    assert len(m) == 1
    assert m.iloc[0][COL_STATION_ID] == "06_075_0001"
    assert len(d) == 1
    assert d.iloc[0][COL_STATION_ID] == "06_075_0001"


class MagicWeather:
    @property
    def source_id(self) -> str:
        return "magic"

    def fetch_daily_meteorology(self, *a, **k):
        raise NotImplementedError

    def fetch_daily_meteorology_for_sites(self, *a, **k):
        return pd.DataFrame()

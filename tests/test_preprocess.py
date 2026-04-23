import numpy as np
import pandas as pd

from aqsparsebench.config import ApiConfig
from aqsparsebench.io.memory_sources import DataFrameAirQualitySource
from aqsparsebench.io.sources import DataSources
from aqsparsebench.io.weather_api import OpenMeteoClient
from aqsparsebench.preprocess.align import align_daily, merge_exogenous
from aqsparsebench.preprocess.canonical import COL_CONCENTRATION, COL_DATE, COL_STATION_ID
from aqsparsebench.preprocess.epa_normalize import normalize_aqs_daily_df, normalize_aqs_monitors_df
from aqsparsebench.preprocess.from_sources import load_air_quality_for_preprocess
from aqsparsebench.preprocess.geo import haversine_km, pairwise_distance_matrix
from aqsparsebench.preprocess.impute import interpolate_time
from aqsparsebench.preprocess.qc import drop_invalid_coordinates
from aqsparsebench.types import BoundingBox, RegionSpec


def test_haversine_paris_berlin_order_of_magnitude() -> None:
    km = haversine_km(48.8566, 2.3522, 52.5200, 13.4050)
    assert 600 < km < 1100


def test_pairwise_distance_matrix() -> None:
    df = pd.DataFrame({"latitude": [0.0, 0.0], "longitude": [0.0, 0.001]})
    m = pairwise_distance_matrix(df.reset_index(drop=True))
    assert m.shape == (2, 2)
    assert m.iloc[0, 1] > 0


def test_normalize_aqs_daily_and_monitors() -> None:
    mon = pd.DataFrame(
        {
            "state_code": ["36"],
            "county_code": ["061"],
            "site_number": ["0123"],
            "latitude": [40.7],
            "longitude": [-74.0],
        }
    )
    m2 = normalize_aqs_monitors_df(mon)
    assert m2.iloc[0]["station_id"] == "36_061_0123"

    day = pd.DataFrame(
        {
            "state_code": ["36"],
            "county_code": ["061"],
            "site_number": ["0123"],
            "date_local": ["2020-01-01"],
            "arithmetic_mean": [12.0],
        }
    )
    d2 = normalize_aqs_daily_df(day)
    assert len(d2) == 1
    assert d2.iloc[0][COL_CONCENTRATION] == 12.0


def test_align_daily_union() -> None:
    df = pd.DataFrame(
        {
            COL_STATION_ID: ["a", "a"],
            COL_DATE: pd.to_datetime(["2020-01-01", "2020-01-03"]),
            COL_CONCENTRATION: [1.0, 3.0],
        }
    )
    out = align_daily(df, calendar="union")
    assert len(out) == 3
    assert out["missing_flag"].sum() == 1


def test_merge_exogenous() -> None:
    aq = pd.DataFrame(
        {
            COL_STATION_ID: ["x"],
            COL_DATE: pd.to_datetime(["2020-01-01"]),
            COL_CONCENTRATION: [1.0],
        }
    )
    wx = pd.DataFrame(
        {
            COL_STATION_ID: ["x"],
            COL_DATE: pd.to_datetime(["2020-01-01"]),
            "temperature_2m_mean": [5.0],
        }
    )
    m = merge_exogenous(aq, wx)
    assert "temperature_2m_mean" in m.columns


def test_interpolate_and_seasonal() -> None:
    df = pd.DataFrame(
        {
            COL_STATION_ID: ["s"] * 3,
            COL_DATE: pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            COL_CONCENTRATION: [1.0, np.nan, 3.0],
        }
    )
    filled = interpolate_time(df)
    assert filled[COL_CONCENTRATION].isna().sum() == 0


def test_load_air_quality_dataframe_source_custom_columns() -> None:
    api = ApiConfig()
    mon = pd.DataFrame({"id": ["eu1"], "lat": [50.0], "lon": [10.0]})
    day = pd.DataFrame({"id": ["eu1"], "day": ["2020-06-01"], "pm25": [8.0]})
    src = DataFrameAirQualitySource(mon, day, provider_label="eu_fixture")
    sources = DataSources.from_clients(air_quality=src, weather=OpenMeteoClient(api))
    region = RegionSpec.from_bbox("eu", BoundingBox(49.0, 51.0, 9.0, 11.0))
    m, d = load_air_quality_for_preprocess(
        sources,
        region,
        pollutant="pm25",
        years=[2020],
        normalization="none",
        custom_monitor_columns=("id", "lat", "lon"),
        custom_daily_columns=("id", "day", "pm25"),
    )
    assert m.iloc[0]["station_id"] == "eu1"
    assert d.iloc[0][COL_CONCENTRATION] == 8.0


def test_data_sources_loader_method() -> None:
    api = ApiConfig()
    mon = pd.DataFrame({"id": ["eu1"], "lat": [50.0], "lon": [10.0]})
    day = pd.DataFrame({"id": ["eu1"], "day": ["2020-06-01"], "pm25": [8.0]})
    src = DataFrameAirQualitySource(mon, day)
    ds = DataSources.from_clients(air_quality=src, weather=OpenMeteoClient(api))
    region = RegionSpec.from_bbox("eu", BoundingBox(49.0, 51.0, 9.0, 11.0))
    m, d = ds.load_air_quality_for_preprocess(
        region,
        pollutant="pm25",
        years=[2020],
        normalization="none",
        custom_monitor_columns=("id", "lat", "lon"),
        custom_daily_columns=("id", "day", "pm25"),
    )
    assert not m.empty and not d.empty


def test_drop_invalid_coords() -> None:
    df = pd.DataFrame({"latitude": [40.0, 999.0], "longitude": [-74.0, -74.0]})
    out = drop_invalid_coordinates(df)
    assert len(out) == 1

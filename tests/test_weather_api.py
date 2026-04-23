from unittest.mock import MagicMock

import pandas as pd
import pytest

from aqsparsebench.config import ApiConfig
from aqsparsebench.io.weather_api import (
    OpenMeteoAPIError,
    OpenMeteoClient,
    _archive_response_to_daily_df,
)


def test_archive_response_to_df() -> None:
    payload = {
        "daily": {
            "time": ["2020-01-01", "2020-01-02"],
            "temperature_2m_mean": [1.0, 2.0],
            "wind_speed_10m_max": [3.0, 4.0],
        }
    }
    df = _archive_response_to_daily_df(payload)
    assert len(df) == 2
    assert "date" in df.columns


def test_get_archive_raw_error_payload() -> None:
    api = ApiConfig(cache_dir=None, open_meteo_request_sleep_seconds=0.0)
    c = OpenMeteoClient(api)
    mock_get = MagicMock(
        return_value=MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={"error": True, "reason": "bad variable"}),
        )
    )
    c.session.get = mock_get
    with pytest.raises(OpenMeteoAPIError):
        c.get_archive_raw({"latitude": 0, "longitude": 0, "start_date": "2020-01-01", "end_date": "2020-01-02", "daily": "temperature_2m_mean"})


def test_fetch_daily_meteorology_for_sites_dedupes_coords() -> None:
    api = ApiConfig(cache_dir=None, open_meteo_request_sleep_seconds=0.0)
    c = OpenMeteoClient(api)
    payload = {
        "daily": {
            "time": ["2020-01-01"],
            "temperature_2m_mean": [5.0],
        }
    }

    def fake_get_archive_raw(params, use_cache=True, throttle=True):
        return payload

    c.get_archive_raw = fake_get_archive_raw  # type: ignore[method-assign]

    sites = pd.DataFrame(
        {
            "station_id": ["a", "b"],
            "latitude": [48.8566, 48.8566],
            "longitude": [2.3522, 2.3522],
        }
    )
    out = c.fetch_daily_meteorology_for_sites(sites, years=[2020])
    assert len(out) == 2
    assert set(out["station_id"]) == {"a", "b"}
    assert (out["temperature_2m_mean"] == 5.0).all()

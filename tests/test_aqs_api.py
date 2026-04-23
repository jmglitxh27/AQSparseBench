from unittest.mock import MagicMock

import pandas as pd
import pytest

from aqsparsebench.config import ApiConfig
from aqsparsebench.io.aqs_api import (
    AQSAPIError,
    AQSClient,
    _extract_rows,
    _header_ok,
    resolve_aqs_param_codes,
)
from aqsparsebench.io.cache import LocalCache, cache_key_from_request
from aqsparsebench.types import BoundingBox, RegionSpec


def test_resolve_aqs_param_codes() -> None:
    assert resolve_aqs_param_codes("pm25") == "88101"
    assert resolve_aqs_param_codes("PM2.5") == "88101"
    assert resolve_aqs_param_codes("pm25", extra_param=" 88502 ") == "88502"


def test_resolve_unknown_pollutant() -> None:
    with pytest.raises(ValueError):
        resolve_aqs_param_codes("so2")


def test_extract_rows_data_and_body() -> None:
    assert _extract_rows({"Data": [{"a": 1}], "Header": []}) == [{"a": 1}]
    assert _extract_rows({"Body": [{"b": 2}]}) == [{"b": 2}]
    assert _extract_rows({}) == []


def test_header_ok() -> None:
    assert _header_ok("success")
    assert _header_ok("SUCCESS")
    assert _header_ok("No data matched your selection")
    assert not _header_ok("Failed")


def test_cache_key_stable() -> None:
    k1 = cache_key_from_request("https://aqs.epa.gov/data/api", "dailyData/byBox", {"a": 1, "b": 2})
    k2 = cache_key_from_request("https://aqs.epa.gov/data/api", "dailyData/byBox", {"b": 2, "a": 1})
    assert k1 == k2
    k3 = cache_key_from_request(
        "https://aqs.epa.gov/data/api",
        "dailyData/byBox",
        {"email": "x", "key": "y", "a": 1, "b": 2},
    )
    assert k1 == k3
    k4 = cache_key_from_request(
        "https://aqs.epa.gov/data/api",
        "dailyData/byBox",
        {"a": 1, "b": 2, "apikey": "one"},
    )
    k5 = cache_key_from_request(
        "https://aqs.epa.gov/data/api",
        "dailyData/byBox",
        {"a": 1, "b": 2, "apikey": "two"},
    )
    assert k4 == k5


def test_local_cache_roundtrip(tmp_path) -> None:
    c = LocalCache(tmp_path)
    c.set_json("abc", {"x": 1})
    assert c.get_json("abc") == {"x": 1}


def test_get_raw_uses_cache(tmp_path) -> None:
    api = ApiConfig(aqs_email="a@b.c", aqs_key="k", cache_dir=str(tmp_path), aqs_request_sleep_seconds=0.0)
    client = AQSClient(api)
    payload = {
        "Header": [{"status": "success", "rows": 1}],
        "Data": [{"state_code": "01"}],
    }
    mock_get = MagicMock(
        return_value=MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value=payload),
        )
    )
    client.session.get = mock_get
    p1 = client.get_raw("list/states", {})
    p2 = client.get_raw("list/states", {})
    assert p1 == payload == p2
    mock_get.assert_called_once()


def test_assert_header_failed() -> None:
    api = ApiConfig(aqs_email="a@b.c", aqs_key="k", aqs_request_sleep_seconds=0.0)
    client = AQSClient(api)
    bad = {"Header": [{"status": "Failed", "error": ["bad param"]}], "Data": []}
    with pytest.raises(AQSAPIError):
        client._assert_header(bad)


def test_fetch_monitors_df_mocked_states(tmp_path) -> None:
    api = ApiConfig(aqs_email="a@b.c", aqs_key="k", cache_dir=str(tmp_path), aqs_request_sleep_seconds=0.0)
    client = AQSClient(api)
    row = {
        "state_code": "36",
        "county_code": "061",
        "site_number": "0123",
        "latitude": 40.7,
        "longitude": -74.0,
        "site_name": "Test",
    }
    payload = {"Header": [{"status": "success", "rows": 1}], "Data": [row]}

    def fake_get_raw(*args, **kwargs):
        endpoint = args[0] if args else ""
        assert "monitors" in str(endpoint)
        return payload

    client.get_raw = fake_get_raw  # type: ignore[method-assign]

    region = RegionSpec.from_states("ny", ("36",))
    df = client.fetch_monitors_df(region, pollutant="pm25", years=[2020])
    assert len(df) == 1
    assert df.iloc[0]["station_id"] == "36_061_0123"


def test_monitors_to_station_records() -> None:
    api = ApiConfig(aqs_email="a@b.c", aqs_key="k", aqs_request_sleep_seconds=0.0)
    client = AQSClient(api)
    df = pd.DataFrame(
        [
            {
                "station_id": "36_061_0123",
                "state_code": "36",
                "county_code": "061",
                "site_number": "0123",
                "latitude": 40.0,
                "longitude": -74.0,
                "site_name": "S",
            }
        ]
    )
    recs = client.monitors_to_station_records(df, region_id="ny")
    assert len(recs) == 1
    assert recs[0].station_id == "36_061_0123"
    assert recs[0].region_id == "ny"

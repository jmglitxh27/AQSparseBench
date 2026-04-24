"""Open-Meteo Historical Weather API (global archive, WGS84)."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Sequence

import pandas as pd
import requests

from aqsparsebench.config import ApiConfig
from aqsparsebench.io.cache import LocalCache, cache_key_from_request

DEFAULT_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Matches aqsparsebench wind/concentration feature needs; all verified against the public archive API.
DEFAULT_ARCHIVE_DAILY_VARIABLES: tuple[str, ...] = (
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
)


class OpenMeteoAPIError(RuntimeError):
    """Raised when Open-Meteo returns an error payload or invalid JSON."""


def _archive_response_to_daily_df(payload: dict[str, Any]) -> pd.DataFrame:
    daily = payload.get("daily")
    if not isinstance(daily, dict):
        return pd.DataFrame()
    times = daily.get("time")
    if not isinstance(times, list) or not times:
        return pd.DataFrame()
    n = len(times)
    cols: dict[str, list[Any]] = {"date": pd.to_datetime(times)}
    for k, v in daily.items():
        if k == "time":
            continue
        if isinstance(v, list) and len(v) == n:
            cols[k] = v
    return pd.DataFrame(cols)


def _year_range_bounds(years: list[int]) -> tuple[str, str]:
    y0, y1 = min(years), max(years)
    return f"{y0:04d}-01-01", f"{y1:04d}-12-31"


def _validate_ymd_dash(token: str, *, name: str) -> None:
    try:
        ts = pd.to_datetime(token, format="%Y-%m-%d", errors="raise")
    except Exception as e:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be YYYY-MM-DD, got {token!r}") from e
    if pd.isna(ts):
        raise ValueError(f"{name} must be YYYY-MM-DD, got {token!r}")


class OpenMeteoClient:
    """
    Historical daily weather via `Open-Meteo archive <https://open-meteo.com/en/docs/historical-weather-api>`_.

    Works **worldwide** (same client for US and non-US benchmarks). Optional
    ``open_meteo_api_key`` in :class:`~aqsparsebench.config.ApiConfig` is sent as ``apikey``
    for `customer API <https://open-meteo.com/en/pricing>`_ hosts.
    """

    def __init__(
        self,
        api: ApiConfig,
        *,
        base_url: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.api = api
        self.base_url = (base_url or api.open_meteo_base_url or DEFAULT_OPEN_METEO_ARCHIVE_URL).rstrip(
            "/"
        )
        self.session = session or requests.Session()
        self.cache = LocalCache(api.cache_dir)

    @property
    def source_id(self) -> str:
        return "open_meteo_archive"

    def get_archive_raw(
        self,
        params: dict[str, Any],
        *,
        use_cache: bool = True,
        throttle: bool = True,
    ) -> dict[str, Any]:
        """GET archive endpoint with optional disk cache and throttling."""
        merged = dict(params)
        if self.api.open_meteo_api_key:
            merged.setdefault("apikey", self.api.open_meteo_api_key)

        url = self.base_url
        cache_path = "GET"

        cache_key: str | None = None
        if use_cache and self.cache.enabled:
            cache_key = cache_key_from_request(url, cache_path, merged)
            hit = self.cache.get_json(cache_key, service="open_meteo")
            if hit is not None:
                return hit

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = self.session.get(
                    url, params=merged, timeout=self.api.open_meteo_read_timeout_seconds
                )
                if r.status_code in (429, 503) and attempt < 2:
                    time.sleep(2.0 ** (attempt + 1) + self.api.open_meteo_request_sleep_seconds)
                    continue
                r.raise_for_status()
                payload: dict[str, Any] = r.json()
                break
            except (requests.RequestException, json.JSONDecodeError) as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(2.0 ** attempt + self.api.open_meteo_request_sleep_seconds)
                    continue
                raise
        else:
            assert last_exc is not None
            raise last_exc

        if payload.get("error") is True:
            reason = payload.get("reason", payload)
            raise OpenMeteoAPIError(str(reason))

        if use_cache and cache_key is not None:
            self.cache.set_json(cache_key, payload, service="open_meteo")

        if throttle and self.api.open_meteo_request_sleep_seconds > 0:
            time.sleep(self.api.open_meteo_request_sleep_seconds)

        return payload

    def fetch_daily_meteorology(
        self,
        *,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """One row per day for a single coordinate pair (``YYYY-MM-DD`` bounds)."""
        vars_ = tuple(variables) if variables is not None else DEFAULT_ARCHIVE_DAILY_VARIABLES
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(vars_),
            "timezone": "UTC",
        }
        raw = self.get_archive_raw(params)
        return _archive_response_to_daily_df(raw)

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
        """
        Batch weather for many sites: **deduplicates** rounded (lat, lon) to limit HTTP calls,
        then expands rows so each site id receives the same series when collocated.
        """
        if sites.empty:
            return pd.DataFrame()
        if (start_date is None) ^ (end_date is None):
            raise ValueError("start_date and end_date must both be set or both omitted")
        if start_date is not None and end_date is not None:
            _validate_ymd_dash(start_date, name="start_date")
            _validate_ymd_dash(end_date, name="end_date")
            if start_date > end_date:
                raise ValueError(f"start_date {start_date!r} must be <= end_date {end_date!r}")
            s0, s1 = start_date, end_date
        else:
            if not years:
                raise ValueError("years must be non-empty when start_date/end_date are omitted")
            s0, s1 = _year_range_bounds(years)
        df = sites.reset_index(drop=True)
        groups: dict[tuple[float, float], list[str]] = defaultdict(list)
        for i, row in df.iterrows():
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            key = (round(lat, 5), round(lon, 5))
            if site_id_col and site_id_col in df.columns and pd.notna(row.get(site_id_col)):
                sid = str(row[site_id_col])
            else:
                sid = f"site_{i}"
            groups[key].append(sid)

        frames: list[pd.DataFrame] = []
        for (lat, lon), sids in groups.items():
            base = self.fetch_daily_meteorology(
                latitude=lat,
                longitude=lon,
                start_date=s0,
                end_date=s1,
                variables=variables,
            )
            if base.empty:
                continue
            for sid in sids:
                part = base.copy()
                part.insert(0, "station_id", sid)
                part.insert(1, "latitude", lat)
                part.insert(2, "longitude", lon)
                frames.append(part)
        return pd.concat(frames, ignore_index=True)

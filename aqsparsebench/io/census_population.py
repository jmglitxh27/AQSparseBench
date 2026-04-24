"""U.S. Census-backed population context for monitor coordinates.

This module provides a PopulationContextSource implementation that:

1) Uses the Census Geocoder API to map each (lat, lon) to a tract and county GEOID.
2) Fetches total population from the ACS 5-year API.
3) Fetches land area (``AREALAND``) via TIGERweb so density (people / km^2) can be computed.

It is intended as a pragmatic starting point for 1 km station-centric population context.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import requests

from aqsparsebench.config import ApiConfig
from aqsparsebench.io.cache import LocalCache, cache_key_from_request

DEFAULT_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
DEFAULT_ACS_BASE_URL = "https://api.census.gov/data"
DEFAULT_TIGERWEB_URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb"


class CensusAPIError(RuntimeError):
    """Raised when Census APIs return an unexpected payload."""


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"sites is missing required columns: {missing}")


def _json_get(
    session: requests.Session,
    cache: LocalCache,
    *,
    url: str,
    params: dict[str, Any],
    service: str,
    timeout_s: float,
    sleep_s: float,
) -> Any:
    cache_key = None
    if cache.enabled:
        cache_key = cache_key_from_request(url, "GET", params)
        hit = cache.get_json(cache_key, service=service)
        if hit is not None:
            return hit

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            r = session.get(url, params=params, timeout=timeout_s)
            if r.status_code in (429, 503) and attempt < 2:
                time.sleep(2.0 ** (attempt + 1) + sleep_s)
                continue
            r.raise_for_status()
            payload = r.json()
            if cache_key is not None:
                cache.set_json(cache_key, payload, service=service)
            if sleep_s > 0:
                time.sleep(sleep_s)
            return payload
        except (requests.RequestException, json.JSONDecodeError) as e:
            last_exc = e
            if attempt < 2:
                time.sleep(2.0 ** attempt + sleep_s)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _extract_geographies(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """
    Return (tract_geoid, county_geoid) from a Census geocoder response.

    tract_geoid is 11-digit (state+county+tract), county_geoid is 5-digit (state+county).
    """
    try:
        results = payload["result"]["geographies"]
    except Exception as e:  # pragma: no cover
        raise CensusAPIError("Invalid geocoder payload") from e
    # Prefer 2020 vintage keys if present, else any Census Tracts.
    tracts = (
        results.get("Census Tracts", [])
        or results.get("Census Tracts, 2020", [])
        or results.get("Census Tracts, 2010", [])
    )
    counties = results.get("Counties", []) or results.get("Counties, 2020", []) or results.get("Counties, 2010", [])
    tract_geoid = None
    county_geoid = None
    if isinstance(tracts, list) and tracts and isinstance(tracts[0], dict):
        tract_geoid = str(tracts[0].get("GEOID") or "").strip() or None
    if isinstance(counties, list) and counties and isinstance(counties[0], dict):
        county_geoid = str(counties[0].get("GEOID") or "").strip() or None
    return tract_geoid, county_geoid


def _acs_total_population(
    payload: Any,
    *,
    expect_field: str = "B01003_001E",
) -> int | None:
    # ACS responses are [["NAME","B01003_001E",...],[...values...]]
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    header = payload[0]
    row = payload[1]
    if not isinstance(header, list) or not isinstance(row, list):
        return None
    try:
        idx = header.index(expect_field)
    except ValueError:
        return None
    try:
        v = row[idx]
        if v is None:
            return None
        return int(float(v))
    except Exception:
        return None


def _tigerweb_arealand_m2(payload: Any) -> float | None:
    """Return AREALAND in square meters from a TIGERweb query payload."""
    if not isinstance(payload, dict):
        return None
    feats = payload.get("features")
    if not isinstance(feats, list) or not feats:
        return None
    attrs = feats[0].get("attributes") if isinstance(feats[0], dict) else None
    if not isinstance(attrs, dict):
        return None
    raw = attrs.get("AREALAND", attrs.get("ALAND"))
    if raw is None:
        return None
    try:
        s = str(raw).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


@dataclass
class CensusPopulationSource:
    """
    Population context for US sites using Census APIs (tract-first, county fallback).

    Outputs per station:
    - tract_geoid (if found)
    - county_geoid (if found)
    - population_total
    - land_area_km2
    - population_density_per_km2
    - pop_level: "tract" | "county" | "none"
    """

    api: ApiConfig
    acs_year: int = 2022
    geocoder_url: str = DEFAULT_GEOCODER_URL
    acs_base_url: str = DEFAULT_ACS_BASE_URL
    tigerweb_url: str = DEFAULT_TIGERWEB_URL
    request_sleep_seconds: float = 0.2
    read_timeout_seconds: float = 60.0
    session: requests.Session | None = None

    @property
    def source_id(self) -> str:
        return "us_census_acs5_tigerweb"

    def _sess(self) -> requests.Session:
        return self.session or requests.Session()

    def _cache(self) -> LocalCache:
        return LocalCache(self.api.cache_dir)

    def _geocode(self, *, lat: float, lon: float) -> tuple[str | None, str | None]:
        params: dict[str, Any] = {
            "x": lon,
            "y": lat,
            "benchmark": "Public_AR_Current",
            "vintage": "Current_Current",
            "format": "json",
        }
        payload = _json_get(
            self._sess(),
            self._cache(),
            url=self.geocoder_url,
            params=params,
            service="census_geocoder",
            timeout_s=self.read_timeout_seconds,
            sleep_s=self.request_sleep_seconds,
        )
        if not isinstance(payload, dict):
            raise CensusAPIError("Geocoder did not return JSON object")
        return _extract_geographies(payload)

    def _acs_pop_tract(self, tract_geoid: str) -> int | None:
        st, co, tr = tract_geoid[:2], tract_geoid[2:5], tract_geoid[5:]
        params: dict[str, Any] = {
            "get": "B01003_001E",
            "for": f"tract:{tr}",
            "in": f"state:{st} county:{co}",
        }
        if self.api.census_api_key:
            params["key"] = self.api.census_api_key
        url = f"{self.acs_base_url}/{self.acs_year}/acs/acs5"
        payload = _json_get(
            self._sess(),
            self._cache(),
            url=url,
            params=params,
            service="census_acs",
            timeout_s=self.read_timeout_seconds,
            sleep_s=self.request_sleep_seconds,
        )
        return _acs_total_population(payload)

    def _acs_pop_county(self, county_geoid: str) -> int | None:
        st, co = county_geoid[:2], county_geoid[2:5]
        params: dict[str, Any] = {
            "get": "B01003_001E",
            "for": f"county:{co}",
            "in": f"state:{st}",
        }
        if self.api.census_api_key:
            params["key"] = self.api.census_api_key
        url = f"{self.acs_base_url}/{self.acs_year}/acs/acs5"
        payload = _json_get(
            self._sess(),
            self._cache(),
            url=url,
            params=params,
            service="census_acs",
            timeout_s=self.read_timeout_seconds,
            sleep_s=self.request_sleep_seconds,
        )
        return _acs_total_population(payload)

    def _aland_tract_m2(self, tract_geoid: str) -> float | None:
        # TIGERweb/Tracts_Blocks: layer 0 is Census Tracts (see service layer directory).
        url = f"{self.tigerweb_url}/Tracts_Blocks/MapServer/0/query"
        params = {
            "where": f"GEOID='{tract_geoid}'",
            "outFields": "AREALAND,GEOID",
            "returnGeometry": "false",
            "f": "json",
        }
        payload = _json_get(
            self._sess(),
            self._cache(),
            url=url,
            params=params,
            service="tigerweb",
            timeout_s=self.read_timeout_seconds,
            sleep_s=self.request_sleep_seconds,
        )
        return _tigerweb_arealand_m2(payload)

    def _aland_county_m2(self, county_geoid: str) -> float | None:
        # TIGERweb/State_County: layer 1 is Counties (5-digit county GEOID).
        url = f"{self.tigerweb_url}/State_County/MapServer/1/query"
        params = {
            "where": f"GEOID='{county_geoid}'",
            "outFields": "AREALAND,GEOID",
            "returnGeometry": "false",
            "f": "json",
        }
        payload = _json_get(
            self._sess(),
            self._cache(),
            url=url,
            params=params,
            service="tigerweb",
            timeout_s=self.read_timeout_seconds,
            sleep_s=self.request_sleep_seconds,
        )
        return _tigerweb_arealand_m2(payload)

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
        _require_cols(sites, [lat_col, lon_col])
        df = sites.reset_index(drop=True).copy()

        out_rows: list[dict[str, Any]] = []
        for i, row in df.iterrows():
            sid = str(row[site_id_col]) if site_id_col and site_id_col in df.columns else f"site_{i}"
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            tract, county = self._geocode(lat=lat, lon=lon)

            pop_level = "none"
            pop = None
            aland_m2 = None

            if tract:
                pop = self._acs_pop_tract(tract)
                aland_m2 = self._aland_tract_m2(tract)
                if pop is not None and aland_m2 is not None and aland_m2 > 0:
                    pop_level = "tract"
            if pop_level != "tract" and county:
                pop2 = self._acs_pop_county(county)
                aland2 = self._aland_county_m2(county)
                if pop2 is not None and aland2 is not None and aland2 > 0:
                    pop_level = "county"
                    pop = pop2
                    aland_m2 = aland2

            land_km2 = (float(aland_m2) / 1_000_000.0) if aland_m2 else None
            density = (float(pop) / land_km2) if (pop is not None and land_km2 and land_km2 > 0) else None

            out_rows.append(
                {
                    "station_id": sid,
                    "tract_geoid": tract,
                    "county_geoid": county,
                    "population_total": pop,
                    "land_area_km2": land_km2,
                    "population_density_per_km2": density,
                    "pop_level": pop_level,
                }
            )
        return pd.DataFrame(out_rows)


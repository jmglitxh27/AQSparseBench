"""EPA AQS Data API client: metadata, monitors, daily summaries, and parameter mapping."""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import requests

from aqsparsebench.config import ApiConfig
from aqsparsebench.io.cache import LocalCache, cache_key_from_request
from aqsparsebench.types import BoundingBox, RegionSpec

DEFAULT_AQS_BASE = "https://aqs.epa.gov/data/api"

# Logical pollutant id -> default AQS 5-digit parameter codes (CSV for API).
# 88101: PM2.5 Local Conditions (FRM/FEM regulatory). 88502: acceptable PM2.5 / AQI mass (non-regulatory).
DEFAULT_POLLUTANT_PARAMS: dict[str, str] = {
    "pm25": "88101",
    "pm2.5": "88101",
    "pm2_5": "88101",
}


class AQSAPIError(RuntimeError):
    """Raised when the AQS API returns a failure header or unexpected payload."""


def resolve_aqs_param_codes(pollutant: str, extra_param: str | None = None) -> str:
    """
    Map a logical pollutant name to AQS ``param`` query value (comma-separated codes).

    If ``extra_param`` is set (e.g. ``\"88502\"`` or ``\"88101,88502\"``), it overrides defaults.
    """
    if extra_param:
        return extra_param.strip()
    key = pollutant.strip().lower().replace(" ", "").replace(".", "")
    if key not in DEFAULT_POLLUTANT_PARAMS:
        known = ", ".join(sorted(set(DEFAULT_POLLUTANT_PARAMS)))
        raise ValueError(f"Unknown pollutant {pollutant!r}. Built-in defaults: {known}")
    return DEFAULT_POLLUTANT_PARAMS[key]


def _year_bounds(y: int) -> tuple[str, str]:
    return f"{y:04d}0101", f"{y:04d}1231"


def _validate_aqs_ymd(token: str, *, name: str) -> None:
    if len(token) != 8 or not token.isdigit():
        raise ValueError(f"{name} must be an 8-digit AQS date string YYYYMMDD, got {token!r}")


def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """AQS responses use ``Data`` or ``Body`` for row arrays depending on service/version."""
    for k in ("Data", "Body"):
        rows = payload.get(k)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    return []


def _parse_header(payload: dict[str, Any]) -> dict[str, Any]:
    hdr = payload.get("Header")
    if isinstance(hdr, list) and hdr and isinstance(hdr[0], dict):
        return hdr[0]
    return {}


def _header_ok(status: str) -> bool:
    s = (status or "").strip().casefold()
    if s == "success":
        return True
    if "no data matched" in s:
        return True
    return False


class AQSClient:
    """
    Low-level GET helpers plus high-level DataFrame builders for monitors and daily data.

    Credentials come from :class:`~aqsparsebench.config.ApiConfig`. Requests are spaced
    with ``aqs_request_sleep_seconds`` and optionally cached under ``cache_dir``.
    """

    def __init__(
        self,
        api: ApiConfig,
        *,
        base_url: str = DEFAULT_AQS_BASE,
        session: requests.Session | None = None,
    ) -> None:
        self.api = api
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.cache = LocalCache(api.cache_dir)

    def _require_credentials(self) -> tuple[str, str]:
        if not self.api.aqs_email or not self.api.aqs_key:
            raise ValueError("ApiConfig.aqs_email and ApiConfig.aqs_key are required for AQS requests")
        return self.api.aqs_email, self.api.aqs_key

    def get_raw(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        use_cache: bool = True,
        throttle: bool = True,
    ) -> dict[str, Any]:
        """
        Perform a GET request to ``{base_url}/{endpoint}`` with ``email`` and ``key`` injected.

        ``endpoint`` examples: ``\"list/states\"``, ``\"monitors/byBox\"``, ``\"dailyData/byState\"``.
        """
        email, key = self._require_credentials()
        path = endpoint.lstrip("/")
        merged: dict[str, Any] = {"email": email, "key": key, **params}
        cache_key: str | None = None
        if use_cache and self.cache.enabled:
            cache_key = cache_key_from_request(self.base_url, path, merged)
            hit = self.cache.get_json(cache_key, service="aqs")
            if hit is not None:
                return hit

        url = f"{self.base_url}/{path}"
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = self.session.get(url, params=merged, timeout=self.api.aqs_read_timeout_seconds)
                if r.status_code in (429, 503) and attempt < 2:
                    time.sleep(2.0 ** (attempt + 1) + self.api.aqs_request_sleep_seconds)
                    continue
                r.raise_for_status()
                payload: dict[str, Any] = r.json()
                break
            except (requests.RequestException, json.JSONDecodeError) as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(2.0 ** attempt + self.api.aqs_request_sleep_seconds)
                    continue
                raise
        else:
            assert last_exc is not None
            raise last_exc

        if use_cache and cache_key is not None:
            self.cache.set_json(cache_key, payload, service="aqs")

        if throttle and self.api.aqs_request_sleep_seconds > 0:
            time.sleep(self.api.aqs_request_sleep_seconds)

        return payload

    def _assert_header(self, payload: dict[str, Any]) -> dict[str, Any]:
        h = _parse_header(payload)
        status = str(h.get("status", ""))
        if not _header_ok(status):
            errs = h.get("error") or h.get("errors") or []
            msg = f"AQS status={status!r} errors={errs!r}"
            raise AQSAPIError(msg)
        return h

    def list_states_raw(self) -> dict[str, Any]:
        return self.get_raw("list/states", {})

    def list_parameters_by_class_raw(self, pc: str = "CRITERIA") -> dict[str, Any]:
        return self.get_raw("list/parametersByClass", {"pc": pc})

    def list_states_df(self) -> pd.DataFrame:
        p = self.list_states_raw()
        self._assert_header(p)
        return pd.DataFrame(_extract_rows(p))

    def list_parameters_by_class_df(self, pc: str = "CRITERIA") -> pd.DataFrame:
        p = self.list_parameters_by_class_raw(pc)
        self._assert_header(p)
        return pd.DataFrame(_extract_rows(p))

    def monitors_by_box_raw(
        self,
        *,
        param: str,
        bdate: str,
        edate: str,
        bbox: BoundingBox,
    ) -> dict[str, Any]:
        return self.get_raw(
            "monitors/byBox",
            {
                "param": param,
                "bdate": bdate,
                "edate": edate,
                "minlat": bbox.min_lat,
                "maxlat": bbox.max_lat,
                "minlon": bbox.min_lon,
                "maxlon": bbox.max_lon,
            },
        )

    def monitors_by_state_raw(
        self,
        *,
        param: str,
        bdate: str,
        edate: str,
        state: str,
    ) -> dict[str, Any]:
        st = f"{int(state):02d}" if state.isdigit() else state
        return self.get_raw(
            "monitors/byState",
            {"param": param, "bdate": bdate, "edate": edate, "state": st},
        )

    def daily_data_by_box_raw(
        self,
        *,
        param: str,
        bdate: str,
        edate: str,
        bbox: BoundingBox,
    ) -> dict[str, Any]:
        return self.get_raw(
            "dailyData/byBox",
            {
                "param": param,
                "bdate": bdate,
                "edate": edate,
                "minlat": bbox.min_lat,
                "maxlat": bbox.max_lat,
                "minlon": bbox.min_lon,
                "maxlon": bbox.max_lon,
            },
        )

    def daily_data_by_state_raw(
        self,
        *,
        param: str,
        bdate: str,
        edate: str,
        state: str,
    ) -> dict[str, Any]:
        st = f"{int(state):02d}" if state.isdigit() else state
        return self.get_raw(
            "dailyData/byState",
            {"param": param, "bdate": bdate, "edate": edate, "state": st},
        )

    def fetch_monitors_df(
        self,
        region: RegionSpec,
        *,
        pollutant: str,
        years: list[int],
        param: str | None = None,
        bdate: str | None = None,
        edate: str | None = None,
    ) -> pd.DataFrame:
        """
        Monitors operating in each calendar year (one API call per year × geography chunk).

        ``param`` overrides the default code list for ``pollutant``.

        When ``bdate`` and ``edate`` are both set (``YYYYMMDD``), only that **single** date
        window is queried (one pass per geography chunk); ``years`` is ignored. Use this for
        short test pulls (e.g. one week) to avoid huge byBox payloads.
        """
        codes = resolve_aqs_param_codes(pollutant, param)
        frames: list[pd.DataFrame] = []
        bbox = region.resolved_bbox()
        if (bdate is None) ^ (edate is None):
            raise ValueError("bdate and edate must both be set or both omitted")
        if bdate is not None and edate is not None:
            _validate_aqs_ymd(bdate, name="bdate")
            _validate_aqs_ymd(edate, name="edate")
            if bdate > edate:
                raise ValueError(f"bdate {bdate!r} must be <= edate {edate!r}")
            windows: list[tuple[str, str]] = [(bdate, edate)]
        else:
            if not years:
                raise ValueError("years must be non-empty when bdate/edate are omitted")
            windows = [_year_bounds(y) for y in years]

        for bd, ed in windows:
            if region.mode == "states" and region.state_fips:
                for st in region.state_fips:
                    p = self.monitors_by_state_raw(param=codes, bdate=bd, edate=ed, state=st)
                    self._assert_header(p)
                    frames.append(pd.DataFrame(_extract_rows(p)))
            elif bbox is not None:
                p = self.monitors_by_box_raw(param=codes, bdate=bd, edate=ed, bbox=bbox)
                self._assert_header(p)
                frames.append(pd.DataFrame(_extract_rows(p)))
            else:
                raise ValueError("RegionSpec has no bbox and no state_fips for AQS monitor query")

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if out.empty:
            return out
        # Stable site key for downstream joins
        if "state_code" in out.columns and "county_code" in out.columns and "site_number" in out.columns:
            out = out.copy()
            out["station_id"] = (
                out["state_code"].astype(str).str.zfill(2)
                + "_"
                + out["county_code"].astype(str).str.zfill(3)
                + "_"
                + out["site_number"].astype(str).str.zfill(4)
            )
        if "station_id" in out.columns:
            return out.drop_duplicates(subset=["station_id"], ignore_index=True)
        return out.drop_duplicates(ignore_index=True)

    def fetch_daily_summary_df(
        self,
        region: RegionSpec,
        *,
        pollutant: str,
        years: list[int],
        param: str | None = None,
        bdate: str | None = None,
        edate: str | None = None,
    ) -> pd.DataFrame:
        """Daily summary rows (one call per calendar chunk × state or box).

        Pass ``bdate`` / ``edate`` (``YYYYMMDD``) together to use a **single** window instead
        of full calendar years from ``years`` (``years`` is then ignored).
        """
        codes = resolve_aqs_param_codes(pollutant, param)
        frames: list[pd.DataFrame] = []
        bbox = region.resolved_bbox()
        if (bdate is None) ^ (edate is None):
            raise ValueError("bdate and edate must both be set or both omitted")
        if bdate is not None and edate is not None:
            _validate_aqs_ymd(bdate, name="bdate")
            _validate_aqs_ymd(edate, name="edate")
            if bdate > edate:
                raise ValueError(f"bdate {bdate!r} must be <= edate {edate!r}")
            windows = [(bdate, edate)]
        else:
            if not years:
                raise ValueError("years must be non-empty when bdate/edate are omitted")
            windows = [_year_bounds(y) for y in years]

        for bd, ed in windows:
            if region.mode == "states" and region.state_fips:
                for st in region.state_fips:
                    p = self.daily_data_by_state_raw(param=codes, bdate=bd, edate=ed, state=st)
                    self._assert_header(p)
                    frames.append(pd.DataFrame(_extract_rows(p)))
            elif bbox is not None:
                p = self.daily_data_by_box_raw(param=codes, bdate=bd, edate=ed, bbox=bbox)
                self._assert_header(p)
                frames.append(pd.DataFrame(_extract_rows(p)))
            else:
                raise ValueError("RegionSpec has no bbox and no state_fips for AQS daily query")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @property
    def source_id(self) -> str:
        return "epa_aqs"

    def fetch_monitor_catalog(
        self,
        region: RegionSpec,
        *,
        pollutant: str,
        years: list[int],
        param: str | None = None,
        bdate: str | None = None,
        edate: str | None = None,
    ) -> pd.DataFrame:
        """Protocol alias for :meth:`fetch_monitors_df` (:class:`~aqsparsebench.io.protocols.AirQualitySource`)."""
        return self.fetch_monitors_df(
            region, pollutant=pollutant, years=years, param=param, bdate=bdate, edate=edate
        )

    def fetch_daily_air_quality(
        self,
        region: RegionSpec,
        *,
        pollutant: str,
        years: list[int],
        param: str | None = None,
        bdate: str | None = None,
        edate: str | None = None,
    ) -> pd.DataFrame:
        """Protocol alias for :meth:`fetch_daily_summary_df`."""
        return self.fetch_daily_summary_df(
            region, pollutant=pollutant, years=years, param=param, bdate=bdate, edate=edate
        )

    def monitors_to_station_records(
        self,
        monitors_df: pd.DataFrame,
        *,
        region_id: str,
    ) -> list:
        """Convert a monitors DataFrame into :class:`~aqsparsebench.types.StationRecord` objects."""
        from aqsparsebench.types import StationRecord

        if monitors_df.empty:
            return []
        rows: list[StationRecord] = []
        core = {
            "station_id",
            "latitude",
            "longitude",
            "state_code",
            "county_code",
            "site_number",
            "site_name",
            "local_site_name",
        }
        for _, r in monitors_df.iterrows():
            sid = str(r.get("station_id", ""))
            if not sid and all(k in r for k in ("state_code", "county_code", "site_number")):
                sid = (
                    str(r["state_code"]).zfill(2)
                    + "_"
                    + str(r["county_code"]).zfill(3)
                    + "_"
                    + str(r["site_number"]).zfill(4)
                )
            lat = float(r["latitude"]) if pd.notna(r.get("latitude")) else float("nan")
            lon = float(r["longitude"]) if pd.notna(r.get("longitude")) else float("nan")
            name = r.get("site_name") or r.get("local_site_name")
            extra = {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in r.items()
                if k not in core and pd.notna(v)
            }
            rows.append(
                StationRecord(
                    station_id=sid,
                    region_id=region_id,
                    lat=lat,
                    lon=lon,
                    state_code=str(r.get("state_code", "")) or None,
                    county_code=str(r.get("county_code", "")) or None,
                    site_number=str(r.get("site_number", "")) or None,
                    name=str(name) if name is not None and pd.notna(name) else None,
                    extra=extra,
                )
            )
        return rows


def is_available(api: ApiConfig, session: requests.Session | None = None) -> bool:
    """Return True if ``metaData/isAvailable`` reports the service is up."""
    c = AQSClient(api, session=session)
    try:
        p = c.get_raw("metaData/isAvailable", {}, use_cache=False, throttle=False)
    except Exception:
        return False
    h = _parse_header(p)
    return str(h.get("status", "")).strip().casefold() == "success"

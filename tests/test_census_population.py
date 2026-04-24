import pandas as pd

from aqsparsebench.io.census_population import CensusPopulationSource, _tigerweb_arealand_m2


def test_tigerweb_arealand_parses_string() -> None:
    payload = {"features": [{"attributes": {"AREALAND": "1234567"}}]}
    assert _tigerweb_arealand_m2(payload) == 1234567.0


def test_fetch_population_context_sets_tract_density(monkeypatch) -> None:
    api = type("Api", (), {"cache_dir": None, "census_api_key": None})()

    src = CensusPopulationSource(api)  # type: ignore[arg-type]

    monkeypatch.setattr(src, "_geocode", lambda lat, lon: ("35001003740", "35001"))
    monkeypatch.setattr(src, "_acs_pop_tract", lambda tract: 5598)
    monkeypatch.setattr(src, "_aland_tract_m2", lambda tract: 1_000_000.0)  # 1 km^2

    sites = pd.DataFrame([{"station_id": "35_001_0023", "latitude": 32.0, "longitude": -96.0}])
    out = src.fetch_population_context(sites)
    assert out.iloc[0]["pop_level"] == "tract"
    assert out.iloc[0]["land_area_km2"] == 1.0
    assert out.iloc[0]["population_density_per_km2"] == 5598.0

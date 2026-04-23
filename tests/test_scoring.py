import pandas as pd

from aqsparsebench.config import ScoringConfig
from aqsparsebench.features.concentration import compute_concentration_score, station_concentration_aggregates
from aqsparsebench.features.utility import build_station_component_table, compute_utility_score
from aqsparsebench.features.variability import compute_variability_score
from aqsparsebench.features.wind import compute_wind_score
from aqsparsebench.preprocess.canonical import COL_CONCENTRATION, COL_DATE, COL_STATION_ID


def _tiny_daily() -> pd.DataFrame:
    rng = pd.date_range("2020-01-01", periods=40, freq="D")
    rows = []
    for sid, base in [("s1", 10.0), ("s2", 20.0)]:
        for i, t in enumerate(rng):
            rows.append({COL_STATION_ID: sid, COL_DATE: t, COL_CONCENTRATION: base + 0.1 * (i % 5)})
    return pd.DataFrame(rows)


def test_concentration_aggregates_and_score() -> None:
    df = pd.DataFrame(
        {
            COL_STATION_ID: ["a", "a", "b"],
            COL_DATE: pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
            COL_CONCENTRATION: [1.0, 3.0, 2.0],
        }
    )
    agg = station_concentration_aggregates(df)
    assert set(agg["station_id"]) == {"a", "b"}
    cfg = ScoringConfig()
    s = compute_concentration_score(df, cfg)
    assert len(s) == 2
    assert (s >= cfg.normalize_range[0]).all() and (s <= cfg.normalize_range[1]).all()


def test_variability_score_bounds() -> None:
    df = _tiny_daily()
    cfg = ScoringConfig()
    v = compute_variability_score(df, cfg)
    assert len(v) == 2
    assert v.notna().all()


def test_wind_score_neutral_without_weather() -> None:
    cfg = ScoringConfig()
    w = compute_wind_score(pd.DataFrame(), cfg, station_ids=pd.Index(["s1", "s2"]))
    assert len(w) == 2
    assert (w == 0.5).all()


def test_build_station_component_table_end_to_end() -> None:
    daily = pd.DataFrame(
        {
            COL_STATION_ID: ["x", "x", "y", "y"],
            COL_DATE: pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"]),
            COL_CONCENTRATION: [5.0, 15.0, 8.0, 12.0],
        }
    )
    wx = pd.DataFrame(
        {
            COL_STATION_ID: ["x", "x", "y", "y"],
            COL_DATE: pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"]),
            "wind_speed_10m_max": [4.0, 5.0, 3.0, 6.0],
            "wind_direction_10m_dominant": [180.0, 190.0, 90.0, 95.0],
        }
    )
    pop = pd.DataFrame({COL_STATION_ID: ["x", "y"], "population_proxy": [1000.0, 500.0]})
    meta = pd.DataFrame(
        {
            COL_STATION_ID: ["x", "y"],
            "latitude": [40.0, 41.0],
            "longitude": [-74.0, -75.0],
        }
    )
    cfg = ScoringConfig()
    tab = build_station_component_table(
        daily,
        cfg,
        weather_daily_df=wx,
        population_df=pop,
        station_meta_df=meta,
    )
    assert list(tab.columns) == [COL_STATION_ID, "C_s", "P_s", "W_s", "V_s", "B_s", "J_s"]
    assert len(tab) == 2
    j = compute_utility_score(tab, cfg)
    assert j.between(cfg.normalize_range[0], cfg.normalize_range[1]).all()

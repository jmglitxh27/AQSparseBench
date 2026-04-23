"""Aggregate station utility ``J_s`` from normalized score components."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from aqsparsebench.config import ScoringConfig
from aqsparsebench.types import StationFeatureRecord


def compute_utility_score(
    components: pd.DataFrame,
    config: ScoringConfig,
) -> pd.Series:
    """
    ``J_s = alpha*C_s + beta*P_s + gamma*W_s + delta*V_s + eta*B_s``.

    ``components`` must include columns ``C_s``, ``P_s``, ``W_s``, ``V_s``, ``B_s``
    (NaN filled with the midpoint of ``normalize_range`` before combining).
    """
    lo, hi = config.normalize_range
    mid = lo + (hi - lo) / 2.0
    need = ["C_s", "P_s", "W_s", "V_s", "B_s"]
    for c in need:
        if c not in components.columns:
            components = components.copy()
            components[c] = mid
    block = components[need].apply(pd.to_numeric, errors="coerce").fillna(mid)
    w = config.utility
    j = (
        w.alpha * block["C_s"]
        + w.beta * block["P_s"]
        + w.gamma * block["W_s"]
        + w.delta * block["V_s"]
        + w.eta * block["B_s"]
    )
    j.name = "J_s"
    return j


def build_station_component_table(
    daily_df: pd.DataFrame,
    config: ScoringConfig,
    *,
    weather_daily_df: pd.DataFrame | None = None,
    population_df: pd.DataFrame | None = None,
    station_meta_df: pd.DataFrame | None = None,
    population_col: str = "population_proxy",
) -> pd.DataFrame:
    """
    One row per station with ``C_s`` … ``B_s`` and ``J_s``.

    ``station_meta_df`` (optional) lists stations to score; defaults to stations
    present in ``daily_df``. Weather and population are optional; missing inputs
    yield neutral ``W_s`` / ``P_s``.
    """
    from aqsparsebench.preprocess.canonical import COL_STATION_ID

    from aqsparsebench.features.background import compute_background_score
    from aqsparsebench.features.concentration import compute_concentration_score
    from aqsparsebench.features.population import compute_population_score
    from aqsparsebench.features.variability import compute_variability_score
    from aqsparsebench.features.wind import compute_wind_score

    if daily_df.empty:
        return pd.DataFrame(columns=[COL_STATION_ID, "C_s", "P_s", "W_s", "V_s", "B_s", "J_s"])

    if station_meta_df is None:
        station_meta_df = pd.DataFrame({COL_STATION_ID: daily_df[COL_STATION_ID].unique()})

    sids = station_meta_df[COL_STATION_ID].astype(str)

    c = compute_concentration_score(daily_df, config)
    v = compute_variability_score(daily_df, config)
    b = compute_background_score(daily_df, config)

    wnd = compute_wind_score(
        weather_daily_df if weather_daily_df is not None else pd.DataFrame(),
        config,
        station_ids=sids,
    )
    pop = compute_population_score(
        station_meta_df,
        population_df,
        config,
        station_col=COL_STATION_ID,
        population_col=population_col,
    )

    def align(s: pd.Series) -> pd.Series:
        s = s.reindex(sids.values)
        lo, hi = config.normalize_range
        return s.fillna(lo + (hi - lo) / 2.0)

    tab = pd.DataFrame(
        {
            COL_STATION_ID: sids.values,
            "C_s": align(c),
            "P_s": align(pop),
            "W_s": align(wnd),
            "V_s": align(v),
            "B_s": align(b),
        }
    )
    tab["J_s"] = compute_utility_score(tab, config).values
    return tab


def component_table_to_feature_records(
    table: pd.DataFrame,
    *,
    region_id: str,
    station_meta: Mapping[str, tuple[float, float]] | pd.DataFrame | None = None,
) -> list[StationFeatureRecord]:
    """Materialize :class:`~aqsparsebench.types.StationFeatureRecord` rows."""
    from aqsparsebench.preprocess.canonical import COL_STATION_ID

    meta_df: pd.DataFrame | None
    if station_meta is None:
        meta_df = None
    elif isinstance(station_meta, pd.DataFrame):
        meta_df = station_meta.copy()
        if COL_STATION_ID not in meta_df.columns:
            raise ValueError(f"station_meta DataFrame must include {COL_STATION_ID!r}")
        meta_df[COL_STATION_ID] = meta_df[COL_STATION_ID].astype(str)
        meta_df = meta_df.set_index(COL_STATION_ID)[["latitude", "longitude"]]
    else:
        meta_df = pd.DataFrame.from_dict(
            {k: {"latitude": v[0], "longitude": v[1]} for k, v in station_meta.items()},
            orient="index",
        )

    out: list[StationFeatureRecord] = []
    for _, row in table.iterrows():
        sid = str(row[COL_STATION_ID])
        lat = lon = float("nan")
        if meta_df is not None and sid in meta_df.index:
            lat = float(meta_df.loc[sid, "latitude"])
            lon = float(meta_df.loc[sid, "longitude"])
        out.append(
            StationFeatureRecord(
                station_id=sid,
                region_id=region_id,
                lat=lat,
                lon=lon,
                C_s=float(row["C_s"]),
                P_s=float(row["P_s"]),
                W_s=float(row["W_s"]),
                V_s=float(row["V_s"]),
                B_s=float(row["B_s"]),
                J_s=float(row["J_s"]),
            )
        )
    return out

"""Population / exposure proxy score (optional until Census wiring is complete)."""

from __future__ import annotations

import pandas as pd

from aqsparsebench.config import ScoringConfig
from aqsparsebench.features._common import minmax_series


def compute_population_score(
    station_df: pd.DataFrame,
    population_df: pd.DataFrame | None,
    config: ScoringConfig,
    *,
    station_col: str = "station_id",
    population_col: str = "population_proxy",
) -> pd.Series:
    """
    ``P_s`` by min–max scaling a ``population_proxy`` joined on ``station_id``.

    If ``population_df`` is None or missing the merge key / column, returns a **neutral**
    score at the midpoint of ``config.normalize_range`` for every station.
    """
    lo, hi = config.normalize_range
    mid = lo + (hi - lo) / 2.0
    ids = station_df[station_col].astype(str)
    if population_df is None or population_df.empty or population_col not in population_df.columns:
        return pd.Series(mid, index=ids, dtype=float).rename("P_s")

    pop = population_df[[station_col, population_col]].copy()
    pop[station_col] = pop[station_col].astype(str)
    pop = pop.drop_duplicates(subset=[station_col], keep="last").set_index(station_col)[population_col]
    aligned = ids.map(pop)
    score = minmax_series(aligned, feature_low=lo, feature_high=hi)
    score.index = ids.values
    return score.rename("P_s")

"""Eligibility filters for candidate target stations (post–sparse-network)."""

from __future__ import annotations

import pandas as pd


def filter_target_candidates(
    retained_station_ids: set[str] | frozenset[str],
    all_station_ids: set[str] | frozenset[str],
    station_feature_table: pd.DataFrame,
    distance_matrix_km: pd.DataFrame,
    *,
    min_days: int = 365,
    min_completeness: float = 0.8,
    min_feature_distance: float | None = None,
    min_geo_distance_km: float | None = None,
    station_col: str = "station_id",
) -> tuple[list[str], pd.DataFrame]:
    """
    Return ``(eligible_ids, eligibility_table)`` with one row per outside station.

    Optional columns in ``station_feature_table``:

    * ``n_valid_days`` — counts toward ``min_days``
    * ``data_completeness`` — fraction in ``[0, 1]`` compared to ``min_completeness``

    If those columns are absent, the corresponding checks are skipped.
    """
    retained = frozenset(str(s) for s in retained_station_ids)
    universe = frozenset(str(s) for s in all_station_ids)
    candidates = sorted(universe - retained)
    rows: list[dict[str, object]] = []

    feat = station_feature_table.copy()
    if station_col not in feat.columns:
        feat = feat.reset_index().rename(columns={"index": station_col})
    feat[station_col] = feat[station_col].astype(str)

    # Pre-compute min geo and min feature distance to retained for redundancy checks
    for sid in candidates:
        reasons: list[str] = []
        ok = True
        if sid not in feat[station_col].values:
            rows.append(
                {
                    station_col: sid,
                    "eligible": False,
                    "eligibility_reason": "missing_from_feature_table",
                }
            )
            continue
        row = feat.loc[feat[station_col] == sid].iloc[0]
        if "n_valid_days" in feat.columns:
            nd = int(row["n_valid_days"])
            if nd < min_days:
                ok = False
                reasons.append(f"n_valid_days={nd}<{min_days}")
        if "data_completeness" in feat.columns:
            comp = float(row["data_completeness"])
            if comp < min_completeness:
                ok = False
                reasons.append(f"completeness={comp:.3f}<{min_completeness}")
        if min_geo_distance_km is not None and sid in distance_matrix_km.index:
            d_geo = float(distance_matrix_km.loc[sid, list(retained)].min())
            if d_geo < min_geo_distance_km:
                ok = False
                reasons.append(f"geo_dmin={d_geo:.3f}km<{min_geo_distance_km}")
        if min_feature_distance is not None:
            # caller should pass precomputed column ``feature_dmin_to_retained`` if needed
            if "feature_dmin_to_retained" in feat.columns:
                fd = float(row["feature_dmin_to_retained"])
                if fd < min_feature_distance:
                    ok = False
                    reasons.append(f"feature_dmin={fd:.4f}<{min_feature_distance}")
        rows.append(
            {
                station_col: sid,
                "eligible": ok,
                "eligibility_reason": ";".join(reasons) if reasons else "ok",
            }
        )

    tab = pd.DataFrame(rows)
    elig = tab.loc[tab["eligible"], station_col].astype(str).tolist()
    return elig, tab


def merge_eligibility(
    scores_df: pd.DataFrame,
    eligibility_table: pd.DataFrame,
    *,
    station_col: str = "station_id",
) -> pd.DataFrame:
    """Attach ``eligible`` / ``eligibility_reason`` columns onto a scores table."""
    return scores_df.merge(eligibility_table, on=station_col, how="left")

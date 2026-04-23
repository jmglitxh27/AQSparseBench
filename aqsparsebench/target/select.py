"""Choose target station(s) after a retained sparse network is fixed."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from aqsparsebench.config import TargetSelectionConfig
from aqsparsebench.target.scoring import compute_target_scores


def select_target_station(
    retained_station_ids: list[str] | set[str],
    candidate_station_ids: list[str] | set[str],
    station_feature_table: pd.DataFrame,
    distance_matrix_km: pd.DataFrame,
    config: TargetSelectionConfig,
    *,
    strategy: Literal["argmax", "topk_random", "stratified"] = "argmax",
    random_state: int | None = None,
    network_id: str = "network",
    retention_level: float = 1.0,
    topk_pool: int = 5,
    n_strata: int = 3,
    station_col: str = "station_id",
) -> dict[str, Any]:
    """
    Pick a target outside the retained set using marginal suitability ``T_s``.

    Strategies:

    * ``argmax`` — single best eligible station by ``T_s``.
    * ``topk_random`` — uniform draw among the ``topk_pool`` highest ``T_s`` rows.
    * ``stratified`` — ``pd.qcut`` on ``T_s`` (eligible only), then one random pick per stratum
      (up to ``n_strata`` stations), returned as ``alternatives`` with the first as primary.
    """
    rng = np.random.default_rng(random_state)
    scores = compute_target_scores(
        retained_station_ids,
        candidate_station_ids,
        station_feature_table,
        distance_matrix_km,
        config,
        station_col=station_col,
        apply_filters=True,
    )
    elig = scores[scores["eligible"]].copy()
    if elig.empty:
        return {
            "target_station_id": None,
            "target_score": float("nan"),
            "retained_network_id": network_id,
            "retention_level": retention_level,
            "score_breakdown": {},
            "selection_strategy": strategy,
            "alternatives": [],
            "candidates_evaluated": int(len(scores)),
        }

    if strategy == "argmax":
        row = elig.iloc[0]
        tid = str(row[station_col])
        breakdown = row[["J_s", "G_s_given_network", "U_s_given_network", "E_s_given_network", "T_s_given_network"]].to_dict()
        return {
            "target_station_id": tid,
            "target_score": float(row["T_s_given_network"]),
            "retained_network_id": network_id,
            "retention_level": retention_level,
            "score_breakdown": breakdown,
            "selection_strategy": strategy,
            "alternatives": [],
            "candidates_evaluated": int(len(scores)),
        }

    if strategy == "topk_random":
        pool = elig.nlargest(min(topk_pool, len(elig)), "T_s_given_network")
        pick = pool.sample(1, random_state=rng).iloc[0]
        tid = str(pick[station_col])
        others = pool[pool[station_col].astype(str) != tid]
        alts = [
            {
                "target_station_id": str(r[station_col]),
                "target_score": float(r["T_s_given_network"]),
                "score_breakdown": r[
                    ["J_s", "G_s_given_network", "U_s_given_network", "E_s_given_network", "T_s_given_network"]
                ].to_dict(),
            }
            for _, r in others.iterrows()
        ]
        breakdown = pick[
            ["J_s", "G_s_given_network", "U_s_given_network", "E_s_given_network", "T_s_given_network"]
        ].to_dict()
        return {
            "target_station_id": tid,
            "target_score": float(pick["T_s_given_network"]),
            "retained_network_id": network_id,
            "retention_level": retention_level,
            "score_breakdown": breakdown,
            "selection_strategy": strategy,
            "alternatives": alts,
            "candidates_evaluated": int(len(scores)),
        }

    # stratified
    elig = elig.sort_values("T_s_given_network", ascending=False).reset_index(drop=True)
    try:
        elig["stratum"] = pd.qcut(
            elig["T_s_given_network"],
            q=min(n_strata, len(elig)),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        elig["stratum"] = 0
    picks: list[dict[str, Any]] = []
    for _, grp in elig.groupby("stratum", sort=False):
        r = grp.sample(1, random_state=rng).iloc[0]
        picks.append(
            {
                "target_station_id": str(r[station_col]),
                "target_score": float(r["T_s_given_network"]),
                "score_breakdown": r[
                    ["J_s", "G_s_given_network", "U_s_given_network", "E_s_given_network", "T_s_given_network"]
                ].to_dict(),
            }
        )
    primary = picks[0] if picks else {}
    return {
        "target_station_id": primary.get("target_station_id"),
        "target_score": float(primary.get("target_score", float("nan"))),
        "retained_network_id": network_id,
        "retention_level": retention_level,
        "score_breakdown": primary.get("score_breakdown", {}),
        "selection_strategy": strategy,
        "alternatives": picks[1:],
        "candidates_evaluated": int(len(scores)),
    }

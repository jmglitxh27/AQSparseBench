"""Default weighted sampling with optional spatial diversity and pairwise distance constraints."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from aqsparsebench.benchmark.retention import retained_station_count
from aqsparsebench.benchmark.strategies.base import GenerationContext, SparseGenerationStrategy
from aqsparsebench.preprocess.geo import haversine_km


def _pairwise_min_km(ids: Sequence[str], station_df: pd.DataFrame) -> float:
    """Minimum great-circle distance (km) over unordered distinct pairs."""
    rows = station_df.set_index("station_id") if "station_id" in station_df.columns else station_df
    best = math.inf
    ids = list(ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if a not in rows.index or b not in rows.index:
                continue
            la, lo = float(rows.loc[a, "latitude"]), float(rows.loc[a, "longitude"])
            lb, lo2 = float(rows.loc[b, "latitude"]), float(rows.loc[b, "longitude"])
            best = min(best, haversine_km(la, lo, lb, lo2))
    return best if math.isfinite(best) else 0.0


def _pairwise_min_from_matrix(ids: Sequence[str], dist: pd.DataFrame) -> float:
    sub = dist.loc[list(ids), list(ids)]
    m = float("inf")
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            m = min(m, float(sub.iloc[i, j]))
    return m if math.isfinite(m) else 0.0


class DefaultWeightedSparseStrategy:
    """
    Sequential risk-weighted picks:

    * probability ∝ ``score**power`` for the first station;
    * later picks multiply by ``exp(-penalty * sum_j 1/(d_ij + eps)))`` when coordinates exist.

    Rejects whole candidates failing ``min_pairwise_km`` until ``max_resample_attempts``.
    """

    @property
    def strategy_id(self) -> str:
        return "default_weighted"

    def generate(self, ctx: GenerationContext) -> list[list[str]]:
        scores = dict(ctx.station_scores)
        ids = list(scores.keys())
        n = len(ids)
        rng = np.random.default_rng(ctx.random_state)
        k = retained_station_count(n, ctx.retention_level)
        if k == n:
            return [ids]

        # Prepare coordinates lookup
        coord_df: pd.DataFrame | None = None
        if ctx.station_df is not None and not ctx.station_df.empty:
            coord_df = ctx.station_df.copy()
            if "station_id" not in coord_df.columns:
                coord_df = coord_df.reset_index().rename(columns={"index": "station_id"})
            coord_df["station_id"] = coord_df["station_id"].astype(str)
            coord_df = coord_df.set_index("station_id")
        dist_m = ctx.distance_matrix_km

        def pick_one_candidate() -> list[str] | None:
            weights = np.array([max(0.0, float(scores[s])) ** ctx.score_power for s in ids], dtype=float)
            if weights.sum() <= 0:
                weights = np.ones(n, dtype=float) / n
            else:
                weights = weights / weights.sum()

            chosen: list[str] = []
            avail = set(ids)
            for _ in range(k):
                cand_list = sorted(avail)
                w = np.array([weights[ids.index(s)] for s in cand_list], dtype=float)
                if ctx.diversity_penalty and chosen and (coord_df is not None or dist_m is not None):
                    pen = np.ones(len(cand_list), dtype=float)
                    for ii, s in enumerate(cand_list):
                        acc = 0.0
                        for t in chosen:
                            if coord_df is not None and s in coord_df.index and t in coord_df.index:
                                d = haversine_km(
                                    float(coord_df.loc[s, "latitude"]),
                                    float(coord_df.loc[s, "longitude"]),
                                    float(coord_df.loc[t, "latitude"]),
                                    float(coord_df.loc[t, "longitude"]),
                                )
                            elif dist_m is not None and s in dist_m.index and t in dist_m.index:
                                d = float(dist_m.loc[s, t])
                            else:
                                d = 1e6
                            acc += 1.0 / (d + ctx.spatial_diversity_scale_km)
                        pen[ii] = math.exp(-acc)
                    w = w * pen
                if w.sum() <= 0 or not math.isfinite(w.sum()):
                    w = np.ones(len(cand_list), dtype=float)
                w = w / w.sum()
                pick = rng.choice(cand_list, p=w)
                chosen.append(str(pick))
                avail.remove(pick)

            if ctx.min_pairwise_km is not None and len(chosen) > 1:
                if coord_df is not None:
                    dmin = _pairwise_min_km(chosen, coord_df.reset_index())
                elif dist_m is not None:
                    dmin = _pairwise_min_from_matrix(chosen, dist_m)
                else:
                    dmin = None
                if dmin is not None and dmin < ctx.min_pairwise_km:
                    return None
            return chosen

        out: list[list[str]] = []
        attempts = 0
        while len(out) < ctx.n_candidates and attempts < ctx.max_resample_attempts:
            attempts += 1
            cand = pick_one_candidate()
            if cand is None:
                continue
            out.append(sorted(cand))

        # dedupe identical sets while preserving order
        seen: set[tuple[str, ...]] = set()
        uniq: list[list[str]] = []
        for c in out:
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                uniq.append(list(t))
        return uniq[: ctx.n_candidates]

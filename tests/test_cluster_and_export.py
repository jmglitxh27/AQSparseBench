import json
from pathlib import Path

import pandas as pd

from aqsparsebench.benchmark.cluster import cluster_candidates
from aqsparsebench.benchmark.generate import generate_sparse_candidates
from aqsparsebench.benchmark.represent import attach_candidate_vectors
from aqsparsebench.benchmark.select import select_representative_networks
from aqsparsebench.export.parquet_export import get_training_jobs, write_benchmark_bundle
from aqsparsebench.preprocess.canonical import COL_CONCENTRATION, COL_DATE, COL_STATION_ID


def _feature_table_and_daily():
    stations = [f"s{i}" for i in range(6)]
    feat = []
    lat, lon = 40.0, -74.0
    for i, sid in enumerate(stations):
        feat.append(
            {
                COL_STATION_ID: sid,
                "latitude": lat + 0.02 * i,
                "longitude": lon - 0.02 * i,
                "J_s": 0.2 + 0.05 * i,
                "C_s": 0.3 + 0.02 * i,
                "P_s": 0.4,
                "V_s": 0.35,
                "B_s": 0.25,
            }
        )
    feat_df = pd.DataFrame(feat)
    daily = []
    for sid in stations:
        for d in range(5):
            daily.append(
                {
                    COL_STATION_ID: sid,
                    COL_DATE: pd.Timestamp("2020-01-01") + pd.Timedelta(days=d),
                    COL_CONCENTRATION: 10.0 + hash(sid) % 3 + 0.1 * d,
                }
            )
    daily_df = pd.DataFrame(daily)
    return feat_df, daily_df


def test_represent_cluster_select_pipeline(tmp_path: Path) -> None:
    feat_df, daily_df = _feature_table_and_daily()
    scores = {row[COL_STATION_ID]: float(row["J_s"]) for _, row in feat_df.iterrows()}
    cands = generate_sparse_candidates(
        scores,
        retention_level=0.5,
        n_candidates=25,
        random_state=42,
        station_df=feat_df[[COL_STATION_ID, "latitude", "longitude"]],
        region_id="test_region",
    )
    assert len(cands) >= 2
    enriched = attach_candidate_vectors(cands, feat_df)
    clustered = cluster_candidates(enriched, auto_k=True, random_state=0, max_clusters=5)
    assert all(c.cluster_id is not None for c in clustered)
    assert sum(1 for c in clustered if c.is_medoid) >= 1
    reps = select_representative_networks(clustered, network_id_prefix="network")
    assert len(reps) >= 1

    manifest = write_benchmark_bundle(
        tmp_path,
        station_features=feat_df,
        candidates=clustered,
        representatives=reps,
        daily_panel=daily_df,
        target_by_network={reps[0].network_id: str(feat_df.iloc[-1][COL_STATION_ID])},
        target_selection_meta={
            reps[0].network_id: {
                "target_selection_strategy": "argmax",
                "target_score": 0.9,
                "target_base_score_J": 0.5,
                "target_gap_score_G": 0.2,
                "target_underrep_score_U": 0.15,
                "target_expansion_score_E": 0.05,
                "target_rank_within_network": 1,
            }
        },
    )
    assert (tmp_path / "station_features.parquet").is_file()
    assert (tmp_path / "training_manifest.parquet").is_file()
    ndir = tmp_path / "networks" / reps[0].network_id
    assert (ndir / "source_data.parquet").is_file()
    assert (ndir / "metadata.json").is_file()
    meta = json.loads((ndir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["network_id"] == reps[0].network_id

    jobs = get_training_jobs(manifest.training_manifest_path)
    assert len(jobs) == len(reps)
    jobs2 = manifest.get_training_jobs()
    assert len(jobs2) == len(jobs)

"""Parquet manifest writers and ``get_training_jobs`` helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from aqsparsebench.benchmark.manifests import (
    CANDIDATE_MANIFEST_FILE,
    REPRESENTATIVE_NETWORKS_FILE,
    STATION_FEATURES_FILE,
    TRAINING_MANIFEST_FILE,
)
from aqsparsebench.export.split_export import write_all_network_bundles
from aqsparsebench.types import BenchmarkManifest, RepresentativeSparseNetwork, SparseNetworkCandidate


def _candidates_dataframe(candidates: Sequence[SparseNetworkCandidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        rows.append(
            {
                "candidate_id": c.candidate_id,
                "region_id": c.region_id,
                "retention_level": c.retention_level,
                "station_ids_json": json.dumps(sorted(c.station_ids)),
                "feature_vector_json": json.dumps(c.candidate_feature_vector or []),
                "cluster_id": c.cluster_id,
                "is_medoid": c.is_medoid,
            }
        )
    return pd.DataFrame(rows)


def _representatives_dataframe(reps: Sequence[RepresentativeSparseNetwork]) -> pd.DataFrame:
    rows = []
    for r in reps:
        rows.append(
            {
                "network_id": r.network_id,
                "retention_level": r.retention_level,
                "cluster_id": r.cluster_id,
                "candidate_id": r.candidate_id,
                "station_ids_json": json.dumps(sorted(r.station_ids)),
                "summary_json": json.dumps(r.summary, default=str),
            }
        )
    return pd.DataFrame(rows)


def _training_manifest_rows(
    representatives: Sequence[RepresentativeSparseNetwork],
    *,
    target_by_network: dict[str, str | None] | None,
    target_selection_meta: dict[str, dict[str, Any]] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, rep in enumerate(representatives):
        tid = None
        if target_by_network and rep.network_id in target_by_network:
            tid = target_by_network[rep.network_id]
        rel_path = f"networks/{rep.network_id}/"
        exp_id = f"exp_{rep.network_id}_{i:04d}"
        row: dict[str, Any] = {
            "experiment_id": exp_id,
            "network_id": rep.network_id,
            "retention_level": rep.retention_level,
            "target_station_id": tid if tid is not None else "",
            "source_station_ids_json": json.dumps(sorted(rep.station_ids)),
            "data_path": rel_path,
            "cluster_id": rep.cluster_id,
            "candidate_id": rep.candidate_id,
        }
        if target_selection_meta and rep.network_id in target_selection_meta:
            meta = target_selection_meta[rep.network_id]
            for k, v in meta.items():
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def write_benchmark_bundle(
    output_dir: str | Path,
    *,
    station_features: pd.DataFrame,
    candidates: Sequence[SparseNetworkCandidate],
    representatives: Sequence[RepresentativeSparseNetwork],
    daily_panel: pd.DataFrame,
    station_col: str = "station_id",
    date_col: str = "date",
    target_by_network: dict[str, str | None] | None = None,
    target_selection_meta: dict[str, dict[str, Any]] | None = None,
    extra_metadata_by_network: dict[str, dict] | None = None,
    bundle_metadata: dict[str, Any] | None = None,
) -> BenchmarkManifest:
    """
    Write the standard benchmark layout under ``output_dir``:

    * ``station_features.parquet``
    * ``candidate_manifest.parquet``
    * ``representative_networks.parquet``
    * ``training_manifest.parquet``
    * ``networks/<network_id>/{source_data,target_data,metadata}.…``
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    station_features.to_parquet(root / STATION_FEATURES_FILE, index=False)
    _candidates_dataframe(candidates).to_parquet(root / CANDIDATE_MANIFEST_FILE, index=False)
    _representatives_dataframe(representatives).to_parquet(root / REPRESENTATIVE_NETWORKS_FILE, index=False)

    write_all_network_bundles(
        root,
        representatives,
        daily_panel,
        station_col=station_col,
        date_col=date_col,
        target_by_network=target_by_network,
        extra_metadata_by_network=extra_metadata_by_network,
    )

    train = _training_manifest_rows(
        representatives,
        target_by_network=target_by_network,
        target_selection_meta=target_selection_meta,
    )
    train.to_parquet(root / TRAINING_MANIFEST_FILE, index=False)

    meta = bundle_metadata or {}
    meta.setdefault("n_networks", len(representatives))
    meta.setdefault("n_candidates", len(candidates))

    return BenchmarkManifest(
        output_dir=str(root.resolve()),
        station_features_path=str((root / STATION_FEATURES_FILE).resolve()),
        candidate_manifest_path=str((root / CANDIDATE_MANIFEST_FILE).resolve()),
        representative_networks_path=str((root / REPRESENTATIVE_NETWORKS_FILE).resolve()),
        training_manifest_path=str((root / TRAINING_MANIFEST_FILE).resolve()),
        metadata=meta,
    )


def get_training_jobs(training_manifest_path: str | Path) -> pd.DataFrame:
    """Load ``training_manifest.parquet`` for Colab / joblib fan-out."""
    return pd.read_parquet(Path(training_manifest_path))

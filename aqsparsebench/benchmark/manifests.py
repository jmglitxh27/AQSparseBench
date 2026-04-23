"""Helpers for benchmark manifest schemas (paths + column names)."""

from __future__ import annotations

from pathlib import Path

STATION_FEATURES_FILE = "station_features.parquet"
CANDIDATE_MANIFEST_FILE = "candidate_manifest.parquet"
REPRESENTATIVE_NETWORKS_FILE = "representative_networks.parquet"
TRAINING_MANIFEST_FILE = "training_manifest.parquet"
NETWORKS_SUBDIR = "networks"


def network_dir(output_dir: Path | str, network_id: str) -> Path:
    """``networks/<network_id>/`` under the bundle root."""
    root = Path(output_dir)
    return root / NETWORKS_SUBDIR / network_id

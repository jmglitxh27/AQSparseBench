"""Per-network ``source_data`` / ``target_data`` Parquet slices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from aqsparsebench.types import RepresentativeSparseNetwork


def write_network_parquet_bundle(
    bundle_root: Path | str,
    rep: RepresentativeSparseNetwork,
    daily_panel: pd.DataFrame,
    *,
    station_col: str = "station_id",
    date_col: str = "date",
    target_station_id: str | None = None,
    extra_metadata: dict | None = None,
) -> Path:
    """
    Write ``networks/<network_id>/source_data.parquet``, ``target_data.parquet``, ``metadata.json``.
    """
    root = Path(bundle_root)
    ndir = root / "networks" / rep.network_id
    ndir.mkdir(parents=True, exist_ok=True)

    src = daily_panel[daily_panel[station_col].astype(str).isin(map(str, rep.station_ids))].copy()
    src.to_parquet(ndir / "source_data.parquet", index=False)

    if target_station_id:
        tgt = daily_panel[daily_panel[station_col].astype(str) == str(target_station_id)].copy()
    else:
        tgt = daily_panel.iloc[0:0].copy()
    tgt.to_parquet(ndir / "target_data.parquet", index=False)

    meta = {
        "network_id": rep.network_id,
        "retention_level": rep.retention_level,
        "cluster_id": rep.cluster_id,
        "candidate_id": rep.candidate_id,
        "source_station_ids": list(rep.station_ids),
        "target_station_id": target_station_id,
        "summary": rep.summary,
    }
    if extra_metadata:
        meta.update(extra_metadata)
    (ndir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return ndir


def write_all_network_bundles(
    bundle_root: Path | str,
    representatives: Sequence[RepresentativeSparseNetwork],
    daily_panel: pd.DataFrame,
    *,
    station_col: str = "station_id",
    date_col: str = "date",
    target_by_network: dict[str, str | None] | None = None,
    extra_metadata_by_network: dict[str, dict] | None = None,
) -> list[Path]:
    """Export every representative network under ``bundle_root/networks/``."""
    paths: list[Path] = []
    for rep in representatives:
        tid = None
        if target_by_network and rep.network_id in target_by_network:
            tid = target_by_network[rep.network_id]
        extra = extra_metadata_by_network.get(rep.network_id) if extra_metadata_by_network else None
        paths.append(
            write_network_parquet_bundle(
                bundle_root,
                rep,
                daily_panel,
                station_col=station_col,
                date_col=date_col,
                target_station_id=tid,
                extra_metadata=extra,
            )
        )
    return paths

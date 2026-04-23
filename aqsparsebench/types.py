"""Typed records for stations, candidates, manifests, and geographic scope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class BoundingBox:
    """Latitude/longitude box in decimal degrees (EPA AQS: north/east positive)."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __post_init__(self) -> None:
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be < max_lat")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be < max_lon (note: western longitudes are negative)")


@dataclass(frozen=True)
class RegionSpec:
    """
    Geographic scope for **discovering** monitors.

    ``bbox`` / ``preset`` modes use WGS84 (EPSG:4326) and are **not** US-specific; any
    air-quality backend may interpret them. The ``states`` mode uses U.S. two-digit state
    FIPS codes and matches EPA AQS state-based listing.
    """

    region_id: str
    mode: Literal["bbox", "states", "preset"]
    bbox: BoundingBox | None = None
    """Used when mode is 'bbox' or resolved from 'preset'."""
    state_fips: tuple[str, ...] | None = None
    """Two-digit state FIPS codes, zero-padded (e.g. ``('09', '36')`` for CT, NY)."""
    preset: str | None = None
    """Named preset key from ``aqsparsebench.regions.REGION_PRESETS`` when mode is 'preset'."""

    @staticmethod
    def from_bbox(region_id: str, bbox: BoundingBox) -> RegionSpec:
        return RegionSpec(region_id=region_id, mode="bbox", bbox=bbox)

    @staticmethod
    def from_states(region_id: str, state_fips: tuple[str, ...]) -> RegionSpec:
        if not state_fips:
            raise ValueError("state_fips must be non-empty")
        normalized = tuple(f"{int(s):02d}" for s in state_fips)
        return RegionSpec(region_id=region_id, mode="states", state_fips=normalized)

    @staticmethod
    def from_preset(region_id: str, preset: str) -> RegionSpec:
        from aqsparsebench.regions import REGION_PRESETS

        if preset not in REGION_PRESETS:
            keys = ", ".join(sorted(REGION_PRESETS))
            raise ValueError(f"Unknown preset {preset!r}. Available: {keys}")
        return RegionSpec(
            region_id=region_id,
            mode="preset",
            preset=preset,
            bbox=REGION_PRESETS[preset],
        )

    def resolved_bbox(self) -> BoundingBox | None:
        """Bounding box for AQS queries, if applicable."""
        if self.mode == "bbox":
            return self.bbox
        if self.mode == "preset":
            return self.bbox
        return None


@dataclass
class StationRecord:
    station_id: str
    region_id: str
    lat: float
    lon: float
    state_code: str | None = None
    county_code: str | None = None
    site_number: str | None = None
    name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StationFeatureRecord:
    station_id: str
    region_id: str
    lat: float
    lon: float
    C_s: float | None = None
    P_s: float | None = None
    W_s: float | None = None
    V_s: float | None = None
    B_s: float | None = None
    J_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SparseNetworkCandidate:
    candidate_id: str
    region_id: str
    retention_level: float
    station_ids: list[str]
    station_scores: dict[str, float]
    candidate_feature_vector: list[float] | None = None
    cluster_id: int | None = None
    is_medoid: bool = False


@dataclass
class RepresentativeSparseNetwork:
    network_id: str
    retention_level: float
    cluster_id: int
    candidate_id: str
    station_ids: list[str]
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkManifest:
    """Paths and checksums for an exported benchmark bundle."""

    output_dir: str
    station_features_path: str
    candidate_manifest_path: str
    representative_networks_path: str
    training_manifest_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_training_jobs(self) -> Any:
        """Load ``training_manifest.parquet`` as a DataFrame (requires pandas)."""
        import pandas as pd

        return pd.read_parquet(self.training_manifest_path)

"""AQSparseBench: sparse AQM network benchmarks for transfer-learning experiments."""

from aqsparsebench.config import (
    ApiConfig,
    BenchmarkRunConfig,
    ConcentrationWeights,
    ScoringConfig,
    SparseGenerationSettings,
    TargetSelectionConfig,
    TargetSelectionWeights,
    UtilityWeights,
    VariabilityWeights,
    WindWeights,
)
from aqsparsebench.benchmark import (
    attach_candidate_vectors,
    cluster_candidates,
    generate_sparse_candidates,
    register_sparse_strategy,
    select_representative_networks,
)
from aqsparsebench.features import (
    build_station_component_table,
    compute_concentration_score,
    compute_utility_score,
)
from aqsparsebench.preprocess import (
    align_daily,
    haversine_km,
    load_air_quality_for_preprocess,
    merge_exogenous,
    normalize_aqs_daily_df,
    normalize_aqs_monitors_df,
)
from aqsparsebench.io import (
    AQSAPIError,
    AQSClient,
    AirQualitySource,
    DataFrameAirQualitySource,
    DataSources,
    LocalCache,
    OpenMeteoClient,
    WeatherArchiveSource,
    resolve_aqs_param_codes,
)
from aqsparsebench.export import get_training_jobs, write_benchmark_bundle
from aqsparsebench.target import compute_target_scores, select_target_station
from aqsparsebench.types import (
    BenchmarkManifest,
    BoundingBox,
    RegionSpec,
    RepresentativeSparseNetwork,
    SparseNetworkCandidate,
    StationFeatureRecord,
    StationRecord,
)

__all__ = [
    "AQSAPIError",
    "AQSClient",
    "AirQualitySource",
    "align_daily",
    "ApiConfig",
    "attach_candidate_vectors",
    "build_station_component_table",
    "BenchmarkManifest",
    "BenchmarkRunConfig",
    "BoundingBox",
    "compute_concentration_score",
    "cluster_candidates",
    "compute_target_scores",
    "compute_utility_score",
    "ConcentrationWeights",
    "DataFrameAirQualitySource",
    "DataSources",
    "generate_sparse_candidates",
    "get_training_jobs",
    "haversine_km",
    "load_air_quality_for_preprocess",
    "LocalCache",
    "merge_exogenous",
    "normalize_aqs_daily_df",
    "normalize_aqs_monitors_df",
    "OpenMeteoClient",
    "RegionSpec",
    "register_sparse_strategy",
    "select_representative_networks",
    "RepresentativeSparseNetwork",
    "ScoringConfig",
    "select_target_station",
    "write_benchmark_bundle",
    "SparseGenerationSettings",
    "SparseNetworkCandidate",
    "StationFeatureRecord",
    "StationRecord",
    "TargetSelectionConfig",
    "TargetSelectionWeights",
    "WeatherArchiveSource",
    "resolve_aqs_param_codes",
    "UtilityWeights",
    "VariabilityWeights",
    "WindWeights",
]

__version__ = "0.1.0"

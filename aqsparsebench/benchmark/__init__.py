"""Sparse candidate generation, clustering, and manifest helpers."""

from aqsparsebench.benchmark.cluster import cluster_candidates
from aqsparsebench.benchmark.generate import generate_sparse_candidates
from aqsparsebench.benchmark.manifests import (
    CANDIDATE_MANIFEST_FILE,
    REPRESENTATIVE_NETWORKS_FILE,
    STATION_FEATURES_FILE,
    TRAINING_MANIFEST_FILE,
    network_dir,
)
from aqsparsebench.benchmark.represent import attach_candidate_vectors, candidate_to_vector
from aqsparsebench.benchmark.retention import retained_station_count, validate_station_scores
from aqsparsebench.benchmark.select import select_representative_networks
from aqsparsebench.benchmark.strategies import (
    DefaultWeightedSparseStrategy,
    GenerationContext,
    SparseGenerationStrategy,
    get_sparse_strategy,
    register_sparse_strategy,
)

__all__ = [
    "CANDIDATE_MANIFEST_FILE",
    "DefaultWeightedSparseStrategy",
    "GenerationContext",
    "REPRESENTATIVE_NETWORKS_FILE",
    "STATION_FEATURES_FILE",
    "SparseGenerationStrategy",
    "TRAINING_MANIFEST_FILE",
    "attach_candidate_vectors",
    "candidate_to_vector",
    "cluster_candidates",
    "generate_sparse_candidates",
    "get_sparse_strategy",
    "network_dir",
    "register_sparse_strategy",
    "retained_station_count",
    "select_representative_networks",
    "validate_station_scores",
]

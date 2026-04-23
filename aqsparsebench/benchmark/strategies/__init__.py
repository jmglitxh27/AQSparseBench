"""Sparse candidate generation strategies."""

from aqsparsebench.benchmark.strategies.base import GenerationContext, SparseGenerationStrategy
from aqsparsebench.benchmark.strategies.registry import get_sparse_strategy, register_sparse_strategy
from aqsparsebench.benchmark.strategies.weighted_sampling import DefaultWeightedSparseStrategy

__all__ = [
    "DefaultWeightedSparseStrategy",
    "GenerationContext",
    "SparseGenerationStrategy",
    "get_sparse_strategy",
    "register_sparse_strategy",
]

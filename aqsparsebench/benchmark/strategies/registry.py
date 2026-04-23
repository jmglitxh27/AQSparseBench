"""Built-in :class:`SparseGenerationStrategy` registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aqsparsebench.benchmark.strategies.weighted_sampling import DefaultWeightedSparseStrategy

if TYPE_CHECKING:
    from aqsparsebench.benchmark.strategies.base import SparseGenerationStrategy

_default_weighted = DefaultWeightedSparseStrategy()
_BUILTIN: dict[str, SparseGenerationStrategy] = {_default_weighted.strategy_id: _default_weighted}


def get_sparse_strategy(strategy_id: str) -> SparseGenerationStrategy:
    if strategy_id not in _BUILTIN:
        known = ", ".join(sorted(_BUILTIN))
        raise ValueError(f"Unknown sparse generation strategy {strategy_id!r}. Built-ins: {known}")
    return _BUILTIN[strategy_id]


def register_sparse_strategy(strategy_id: str, strategy: SparseGenerationStrategy) -> None:
    """Register a custom strategy (e.g. plug-in retention rule) at runtime."""
    _BUILTIN[strategy_id] = strategy

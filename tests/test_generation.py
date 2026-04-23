import pandas as pd

from aqsparsebench.benchmark.generate import generate_sparse_candidates
from aqsparsebench.benchmark.retention import retained_station_count
from aqsparsebench.benchmark.strategies.base import GenerationContext, SparseGenerationStrategy
from aqsparsebench.benchmark.strategies.registry import register_sparse_strategy


def test_retained_station_count() -> None:
    assert retained_station_count(10, 1.0) == 10
    assert retained_station_count(10, 0.5) == 5
    assert retained_station_count(3, 0.33) >= 1


def test_generate_dense_baseline() -> None:
    scores = {f"s{i}": float(i) for i in range(5)}
    cands = generate_sparse_candidates(scores, 1.0, n_candidates=10, random_state=0)
    assert len(cands) == 1
    assert len(cands[0].station_ids) == 5


def test_generate_partial_retention() -> None:
    scores = {f"s{i}": float(i + 1) for i in range(8)}
    df = pd.DataFrame(
        {
            "station_id": list(scores.keys()),
            "latitude": [40.0 + 0.01 * i for i in range(8)],
            "longitude": [-74.0 - 0.01 * i for i in range(8)],
        }
    )
    cands = generate_sparse_candidates(
        scores,
        0.5,
        n_candidates=20,
        random_state=1,
        station_df=df,
        diversity_penalty=True,
    )
    k = retained_station_count(8, 0.5)
    assert all(len(c.station_ids) == k for c in cands)
    assert len(cands) >= 1


def test_custom_strategy_registration() -> None:
    class FixedStrategy:
        @property
        def strategy_id(self) -> str:
            return "fixed_test"

        def generate(self, ctx: GenerationContext) -> list[list[str]]:
            ids = sorted(ctx.station_scores.keys())
            k = retained_station_count(len(ids), ctx.retention_level)
            return [ids[:k]] * min(3, ctx.n_candidates)

    register_sparse_strategy("fixed_test", FixedStrategy())  # type: ignore[arg-type]
    scores = {"a": 1.0, "b": 2.0, "c": 3.0}
    out = generate_sparse_candidates(scores, 0.66, n_candidates=5, strategy="fixed_test", random_state=0)
    assert len(out) >= 1

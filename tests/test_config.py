import pytest

from aqsparsebench.config import ScoringConfig, UtilityWeights
from aqsparsebench.types import BoundingBox, RegionSpec


def test_utility_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        UtilityWeights(alpha=0.6, beta=0.6, gamma=0.0, delta=0.0, eta=0.0)


def test_scoring_config_partial_merge() -> None:
    cfg = ScoringConfig.from_partial(utility={"alpha": 0.25, "beta": 0.25, "gamma": 0.20, "delta": 0.20, "eta": 0.10})
    assert cfg.utility.alpha == 0.25


def test_region_bbox_and_preset() -> None:
    bb = BoundingBox(min_lat=40.0, max_lat=41.0, min_lon=-75.0, max_lon=-74.0)
    r = RegionSpec.from_bbox("philly_box", bb)
    assert r.mode == "bbox"
    assert r.resolved_bbox() == bb

    r2 = RegionSpec.from_preset("ne", "northeast")
    assert r2.mode == "preset"
    assert r2.resolved_bbox() is not None

    r3 = RegionSpec.from_preset("ny", "new_york")
    assert r3.mode == "preset"
    assert r3.resolved_bbox() is not None


def test_region_states() -> None:
    r = RegionSpec.from_states("ny_ct", ("36", "09"))
    assert r.state_fips == ("36", "09")

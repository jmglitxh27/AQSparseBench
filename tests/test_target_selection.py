import numpy as np
import pandas as pd

from aqsparsebench.config import TargetSelectionConfig
from aqsparsebench.target.scoring import compute_target_scores
from aqsparsebench.target.select import select_target_station


def _tiny_network():
    stations = [f"s{i}" for i in range(6)]
    feat = []
    for i, sid in enumerate(stations):
        feat.append(
            {
                "station_id": sid,
                "C_s": 0.2 + 0.1 * i,
                "P_s": 0.3,
                "W_s": 0.4,
                "V_s": 0.35,
                "B_s": 0.25,
                "J_s": 0.3 + 0.05 * i,
            }
        )
    lat = np.array([40.0, 40.1, 40.2, 40.3, 40.4, 40.5])
    lon = np.array([-74.0, -74.1, -74.2, -74.3, -74.4, -74.5])
    dist = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i != j:
                # rough km proxy
                dist[i, j] = 10.0 * abs(i - j) + 1.0
    ddf = pd.DataFrame(dist, index=stations, columns=stations)
    return pd.DataFrame(feat), ddf


def test_compute_target_scores_columns() -> None:
    feat, dist = _tiny_network()
    retained = ["s0", "s1"]
    candidates = [f"s{i}" for i in range(6)]
    cfg = TargetSelectionConfig(min_geo_distance_km=None, min_feature_distance=None)
    out = compute_target_scores(retained, candidates, feat, dist, cfg, apply_filters=False)
    assert set(out.columns) >= {
        "station_id",
        "J_s",
        "G_s_given_network",
        "U_s_given_network",
        "E_s_given_network",
        "T_s_given_network",
    }
    assert len(out) == 4


def test_select_target_argmax() -> None:
    feat, dist = _tiny_network()
    retained = ["s0", "s1"]
    candidates = [f"s{i}" for i in range(6)]
    cfg = TargetSelectionConfig()
    res = select_target_station(
        retained,
        candidates,
        feat,
        dist,
        cfg,
        strategy="argmax",
        random_state=0,
        network_id="net1",
        retention_level=0.5,
    )
    assert res["target_station_id"] in {"s2", "s3", "s4", "s5"}
    assert res["selection_strategy"] == "argmax"
    assert "T_s_given_network" in res["score_breakdown"]

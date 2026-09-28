"""Error-slice counting contract: one shared gallery and explicit denominators."""

import numpy as np
import pandas as pd

from audit_joint_errors import summarize_queries


def test_known_and_unknown_errors_share_fixed_gallery():
    frame = pd.DataFrame({
        "image_id": ["q0", "q1", *[f"g{i}" for i in range(10)]],
        "vehicle_id": [1, 2, *range(10, 20)], "camera_id": [0] * 12,
        "min_side": [200, 400, *([300] * 10)],
        "brightness": [40, 100, *([100] * 10)],
        "sharpness_128": [50, 200, *([200] * 10)],
    })
    frame.loc[2, "vehicle_id"] = 1
    pack = {
        "query_indices": np.array([0, 1]), "gallery_indices": np.arange(2, 12),
        "ranking": np.tile(np.arange(10), (2, 1)),
        "known": np.array([True, False]), "truth": np.array([[True] + [False] * 9, [False] * 10]),
        "scores": np.ones((2, 10), dtype=np.float32) * 0.5,
    }
    decisions = np.zeros((2, 10), dtype=bool)
    decisions[1, 0] = True
    rows, slices = summarize_queries(pack, frame, decisions)
    assert rows.error_type.tolist() == ["false_reject_known", "false_accept_unknown"]
    assert sum(row["queries"] for row in slices if row["slice"] == "resolution_slice") == 2

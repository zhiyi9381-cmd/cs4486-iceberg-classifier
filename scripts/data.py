# -*- coding: utf-8 -*-
"""
Data loading: convert train.json / test.json into numpy arrays.
- img: (n, 2, 75, 75) float32  (band_1, band_2)
- angle: (n,) float32 (missing = np.nan, following the P0 convention)
- y: (n,) int64 (train only)
- ids: list[str]
The 75x75 dual-band image shape was confirmed by the P0 audit.
"""
import numpy as np

import common as C

H, W = 75, 75


def _to_arrays(records, has_label):
    n = len(records)
    img = np.zeros((n, 2, H, W), dtype=np.float32)
    angle = np.zeros(n, dtype=np.float32)
    y = np.zeros(n, dtype=np.int64) if has_label else None
    ids = []
    for i, r in enumerate(records):
        img[i, 0] = np.asarray(r["band_1"], dtype=np.float32).reshape(H, W)
        img[i, 1] = np.asarray(r["band_2"], dtype=np.float32).reshape(H, W)
        v = r["inc_angle"]
        angle[i] = np.nan if v == C.MISSING_SENTINEL_RAW else float(v)
        if has_label:
            y[i] = int(r["is_iceberg"])
        ids.append(r["id"])
    return img, angle, y, ids


def load_train():
    return _to_arrays(C.load_json(C.TRAIN_JSON), has_label=True)


def load_test():
    img, angle, _, ids = _to_arrays(C.load_json(C.TEST_JSON), has_label=False)
    return img, angle, ids

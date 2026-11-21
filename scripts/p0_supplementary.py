# -*- coding: utf-8 -*-
"""
P0 supplementary statistics: inc_angle distribution / full image dimension consistency / pixel value range
Writes logs/p0_audit_supplementary_{run_id}.json (versioned naming)
"""

import json
import os
import datetime
import platform
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
MISSING_SENTINEL_RAW = "na"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    out = {
        "run_id": "p0supp_" + t0.strftime("%Y%m%d_%H%M%S"),
        "timestamp_utc": t0.isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }

    train = load_json(os.path.join(DATA, "train.json"))
    test = load_json(os.path.join(DATA, "test.json"))

    # full image dimension consistency
    def check_dims(records, name):
        bad = 0
        for r in records:
            if len(r["band_1"]) != 5625 or len(r["band_2"]) != 5625:
                bad += 1
        out[f"{name}_records_with_non_5625_bands"] = bad
        return bad

    check_dims(train, "train")
    check_dims(test, "test")

    # inc_angle distribution (non-missing only)
    def angle_stats(records):
        vals = []
        for r in records:
            v = r.get("inc_angle")
            if v != MISSING_SENTINEL_RAW:
                vals.append(float(v))
        vals = np.asarray(vals, dtype=float)
        return {
            "n": int(len(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std_ddof0": float(vals.std(ddof=0)),
            "median": float(np.median(vals)),
        }

    out["train_inc_angle_stats_nonmissing"] = angle_stats(train)
    out["test_inc_angle_stats_nonmissing"] = angle_stats(test)

    # pixel value range (all band_1+band_2, using a sample subset + full min/max aggregation to avoid running out of memory)
    def pixel_stats(records):
        gmin = np.inf
        gmax = -np.inf
        gsum = 0.0
        gcnt = 0
        for r in records:
            for b in ("band_1", "band_2"):
                arr = np.asarray(r[b], dtype=np.float32)
                gmin = min(gmin, float(arr.min()))
                gmax = max(gmax, float(arr.max()))
                gsum += float(arr.sum())
                gcnt += arr.size
        return {"min": float(gmin), "max": float(gmax), "mean": float(gsum / gcnt)}

    out["train_pixel_stats"] = pixel_stats(train)
    out["test_pixel_stats"] = pixel_stats(test)

    os.makedirs(LOGS, exist_ok=True)
    p = os.path.join(LOGS, f"p0_audit_supplementary_{out['run_id']}.json")
    if os.path.exists(p):
        raise FileExistsError(f"Target file already exists, refusing to overwrite: {p}")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[P0-supp] written to: {p}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
P0 data audit script (CS4486 Mini Project / Statoil Iceberg Classifier Challenge)

Responsibilities:
  1. Compute SHA-256 of train.json / test.json (streaming, without loading fully into memory)
  2. Count samples, missing inc_angle, class distribution, image dimensions
  3. Confirm test has no labels (no is_iceberg field)
  4. Verify sample_submission.csv format (columns, row count, id order)
  5. Write all results to logs/p0_audit_{run_id}.json (with run_id / timestamp / environment, versioned naming)

Missing angle convention (consistent everywhere, fixed):
  In the raw JSON a missing inc_angle is represented by the string "na".
  This convention maps it to numpy.nan (float).
  All subsequent imputation / binning / feature engineering uses np.nan as the sole representation of "missing".
"""

import json
import hashlib
import os
import sys
import platform
import datetime
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")

TRAIN_JSON = os.path.join(DATA, "train.json")
TEST_JSON = os.path.join(DATA, "test.json")
SAMPLE_SUB = os.path.join(DATA, "sample_submission.csv")

MISSING_SENTINEL_RAW = "na"   # missing string in the raw JSON
MISSING_VALUE = np.nan         # unified missing representation in this project


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    run_id = t0.strftime("p0_%Y%m%d_%H%M%S")

    audit = {
        "run_id": run_id,
        "timestamp_utc": t0.isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "missing_angle_convention": {
            "raw_sentinel": MISSING_SENTINEL_RAW,
            "mapped_value": "numpy.nan (float)",
        },
    }

    # ---- 1. SHA-256 ----
    audit["train_sha256"] = sha256_of_file(TRAIN_JSON)
    audit["test_sha256"] = sha256_of_file(TEST_JSON)
    audit["train_bytes"] = os.path.getsize(TRAIN_JSON)
    audit["test_bytes"] = os.path.getsize(TEST_JSON)

    # ---- 2. train stats ----
    train = load_json(TRAIN_JSON)
    n_train = len(train)
    keys_train = sorted(train[0].keys())
    audit["n_train"] = n_train
    audit["train_record_keys"] = keys_train

    band1 = np.asarray(train[0]["band_1"], dtype=float)
    band2 = np.asarray(train[0]["band_2"], dtype=float)
    audit["band_1_len"] = int(len(band1))
    audit["band_2_len"] = int(len(band2))
    audit["image_dim"] = f"{int(np.sqrt(len(band1)))}x{int(np.sqrt(len(band2)))}"
    audit["band_1_dtype"] = str(band1.dtype)

    inc_angles_train = []
    for r in train:
        v = r.get("inc_angle")
        inc_angles_train.append(np.nan if v == MISSING_SENTINEL_RAW else float(v))
    inc_angles_train = np.asarray(inc_angles_train, dtype=float)
    audit["train_inc_angle_missing"] = int(np.isnan(inc_angles_train).sum())
    audit["train_inc_angle_nonmissing"] = int((~np.isnan(inc_angles_train)).sum())

    labels = np.asarray([int(r["is_iceberg"]) for r in train], dtype=int)
    uniq, cnt = np.unique(labels, return_counts=True)
    audit["train_class_distribution"] = {str(int(k)): int(v) for k, v in zip(uniq, cnt)}
    audit["train_class_imbalance_ratio_pos"] = float((labels == 1).mean())

    # ---- 3. test stats ----
    test = load_json(TEST_JSON)
    n_test = len(test)
    keys_test = sorted(test[0].keys())
    audit["n_test"] = n_test
    audit["test_record_keys"] = keys_test
    audit["test_has_label_field"] = "is_iceberg" in keys_test

    inc_angles_test = []
    for r in test:
        v = r.get("inc_angle")
        inc_angles_test.append(np.nan if v == MISSING_SENTINEL_RAW else float(v))
    inc_angles_test = np.asarray(inc_angles_test, dtype=float)
    audit["test_inc_angle_missing"] = int(np.isnan(inc_angles_test).sum())

    # test image dimension validation (first record)
    tb1 = np.asarray(test[0]["band_1"], dtype=float)
    audit["test_band_1_len"] = int(len(tb1))

    # ---- 4. sample_submission.csv format ----
    with open(SAMPLE_SUB, "r", encoding="utf-8") as f:
        sub_lines = f.read().strip().splitlines()
    header = sub_lines[0]
    n_sub_rows = len(sub_lines) - 1
    audit["sample_submission_header"] = header
    audit["sample_submission_n_rows"] = n_sub_rows
    audit["sample_submission_first_lines"] = sub_lines[:3]

    # id order check (train / test / submission)
    sub_ids = [ln.split(",")[0] for ln in sub_lines[1:]]
    test_ids = [r["id"] for r in test]
    train_ids = [r["id"] for r in train]
    audit["submission_id_order_matches_test"] = (sub_ids == test_ids)
    audit["test_id_unique"] = (len(set(test_ids)) == len(test_ids))
    audit["train_id_unique"] = (len(set(train_ids)) == len(train_ids))

    # ---- 5. write log (run_id versioned naming + refuse overwrite) ----
    os.makedirs(LOGS, exist_ok=True)
    out_path = os.path.join(LOGS, f"p0_audit_{run_id}.json")
    if os.path.exists(out_path):
        raise FileExistsError(f"Target file already exists, refusing to overwrite: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\n[P0] audit written to: {out_path}")


if __name__ == "__main__":
    main()

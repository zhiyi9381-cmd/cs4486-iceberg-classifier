# -*- coding: utf-8 -*-
"""
Loading and validation of the fold indices (the P1 base; all subsequent experiments
read only this fold file and must not re-split on the fly).

load_folds enforces mandatory checks on every load (any script that loads it is
forced to validate; a failure raises AssertionError):
  M1  key meta fields (n_splits=5, random_state=42, shuffle=True, sklearn_version, train_sha256)
  M5  train_sha256 matches the on-disk data/train.json SHA (prevents silent misalignment if the data is replaced)
  M2  labels length and labels_sha256 are self-consistent
  M3  fold_idx_sha256 is self-consistent (a hash of the fold indices themselves, tamper-evident)
  M4  all validate_folds V1-V5 invariants
This turns "loading wrong / modified folds" into an error rather than a silent pass-through.

validate_folds returns (ok, report); every check in report carries evidence.
"""
import hashlib
import json
import os
import glob
import numpy as np

import common as C

FOLDS_V2 = os.path.join(C.FOLDS, "folds_v2.json")   # current authoritative fold base (full metadata)
FOLDS_V1 = os.path.join(C.FOLDS, "folds_v1.json")   # historical (incomplete metadata, superseded by v2)


def labels_sha256(labels):
    h = hashlib.sha256()
    h.update(np.asarray(labels, dtype=int).tobytes())
    return h.hexdigest()


def fold_indices_sha256(folds):
    """Hash of the fold indices themselves: SHA-256 over a stable serialization of (fold, train_idx, val_idx)."""
    payload = json.dumps(
        [{"fold": f["fold"], "train_idx": f["train_idx"], "val_idx": f["val_idx"]}
         for f in folds],
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_folds(folds, n_samples=None, labels=None):
    """
    Validate the fold-index invariants:
      V1  the number of folds is correct and equal to 5
      V2  each fold's train/val union is the full set and disjoint (train_idx ∩ val_idx = ∅)
      V3  validation folds are pairwise disjoint and cover everything (each sample is in exactly one val fold)
      V4  index ranges are valid
      V5  stratification: per-fold positive count deviates from the global-proportion expectation by at most 1
         (StratifiedKFold allocates by class count, an integer-count upper bound, not a "rate difference" test)
    Arguments:
      folds: dict returned by load_folds
      n_samples: total number of samples (if omitted, taken from meta)
      labels: label array (if omitted, taken from folds["labels"])
    """
    r = {"passed": True, "checks": {}}

    fs = folds["folds"]
    meta = folds["meta"]
    n = n_samples if n_samples is not None else meta["n_train"]
    y = np.asarray(labels if labels is not None else folds["labels"], dtype=int)

    # V1 number of folds
    r["checks"]["V1_n_folds"] = {"ok": len(fs) == C.N_SPLITS, "value": len(fs),
                                 "expect": C.N_SPLITS}

    # V2 per-fold union/intersection
    v2_ok = True
    v2_detail = []
    for f in fs:
        tr = set(f["train_idx"]); va = set(f["val_idx"])
        inter = tr & va
        union_ok = (tr | va) == set(range(n))
        sub_ok = (len(inter) == 0)
        v2_ok = v2_ok and sub_ok and union_ok
        v2_detail.append({"fold": f["fold"], "train_n": len(tr), "val_n": len(va),
                          "intersection": len(inter), "union_full": union_ok})
    r["checks"]["V2_per_fold_disjoint_and_full"] = {"ok": v2_ok, "detail": v2_detail}

    # V3 validation folds pairwise disjoint and covering everything
    val_sets = [set(f["val_idx"]) for f in fs]
    pairwise_disjoint = all(
        len(val_sets[i] & val_sets[j]) == 0
        for i in range(len(val_sets)) for j in range(i + 1, len(val_sets))
    )
    union_val = set().union(*val_sets)
    full_cover = (union_val == set(range(n)))
    r["checks"]["V3_val_disjoint_and_cover"] = {
        "ok": pairwise_disjoint and full_cover,
        "pairwise_disjoint": pairwise_disjoint,
        "full_coverage": full_cover,
        "val_sizes": [len(s) for s in val_sets],
    }

    # V4 index range
    all_idx = []
    for f in fs:
        all_idx += list(f["train_idx"]) + list(f["val_idx"])
    all_idx = np.asarray(all_idx, dtype=int)
    in_range = bool((all_idx >= 0).all() and (all_idx < n).all())
    r["checks"]["V4_index_range"] = {"ok": in_range, "min": int(all_idx.min()),
                                     "max": int(all_idx.max()), "n": n}

    # V5 stratification: per-fold positive count deviates from the global-proportion expectation by at most 1
    global_pos = float((y == 1).mean())
    v5_ok = True
    v5_detail = []
    for f in fs:
        va = np.asarray(f["val_idx"], dtype=int)
        pos_cnt = int((y[va] == 1).sum())
        exp = global_pos * len(va)
        dev = abs(pos_cnt - exp)
        close = dev <= 1 + 1e-9
        v5_ok = v5_ok and close
        v5_detail.append({"fold": f["fold"], "val_size": int(len(va)),
                          "pos_count": pos_cnt,
                          "expected_pos_count": round(exp, 3),
                          "pos_rate": pos_cnt / len(va),
                          "global_pos_rate": global_pos,
                          "count_deviation": round(dev, 3)})
    r["checks"]["V5_stratification"] = {"ok": v5_ok, "global_pos_rate": global_pos,
                                        "tolerance": "|pos_count - expected| <= 1",
                                        "detail": v5_detail}

    r["passed"] = all(c["ok"] for c in r["checks"].values())
    return r["passed"], r


def load_folds(path=FOLDS_V2):
    """Read the fold indices and enforce validation (the single entry point). Any failed check raises AssertionError."""
    with open(path, "r", encoding="utf-8") as f:
        folds = json.load(f)

    meta = folds["meta"]
    n = meta["n_train"]
    y = np.asarray(folds["labels"], dtype=int)

    # M1 key meta fields
    assert meta.get("n_splits") == C.N_SPLITS, "n_splits must be 5"
    assert meta.get("random_state") == C.SEED, "random_state must be 42"
    assert meta.get("shuffle") is True, "shuffle must be True"
    assert meta.get("sklearn_version"), "meta missing sklearn_version"
    assert meta.get("train_sha256"), "meta missing train_sha256"

    # M5 verify train_sha256 against the actual on-disk file (prevents silent misalignment if the data is replaced)
    actual_train_sha = C.sha256_of_file(C.TRAIN_JSON)
    assert actual_train_sha == meta["train_sha256"], (
        f"train.json on-disk SHA differs from folds meta.train_sha256; data may have been replaced: "
        f"disk={actual_train_sha[:16]}... meta={meta['train_sha256'][:16]}..."
    )

    # M2 labels length and hash self-consistency
    assert len(y) == n, f"labels length {len(y)} != n_train {n}"
    assert folds.get("labels_sha256") == labels_sha256(y), "labels_sha256 does not match labels"

    # M3 fold_idx_sha256 self-consistency (tamper-evident)
    assert folds.get("fold_idx_sha256") == fold_indices_sha256(folds["folds"]), \
        "fold_idx_sha256 does not match the fold indices (the fold indices may have been tampered with)"

    # M4 V1-V5 invariants
    ok, report = validate_folds(folds, n_samples=n, labels=y)
    assert ok, "fold validation failed: " + json.dumps(report, ensure_ascii=False)

    return folds


def _latest_log(pattern):
    """Return the latest matching log by filename (run_id embeds a timestamp, so lexicographic order is chronological)."""
    files = glob.glob(os.path.join(C.LOGS, pattern))
    if not files:
        raise FileNotFoundError(f"no matching log found: {pattern}")
    return sorted(files)[-1]


def verify_data_sha256():
    """Check that the SHA-256 of the current data/train.json|test.json matches the P0 audit record (data source not replaced)."""
    audit_path = _latest_log("p0_audit_p0_*.json")
    audit = json.load(open(audit_path, encoding="utf-8"))
    checks = {}
    for name, key in [("train", "train_sha256"), ("test", "test_sha256")]:
        actual = C.sha256_of_file(os.path.join(C.DATA, name + ".json"))
        recorded = audit[key]
        checks[name] = {"actual": actual, "recorded": recorded, "match": actual == recorded}
    return all(c["match"] for c in checks.values()), checks

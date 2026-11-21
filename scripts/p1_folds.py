# -*- coding: utf-8 -*-
"""
P1 fold split base: generate folds/folds_v2.json (full metadata version)
  - compute train.json SHA-256 at runtime (no longer hard-coded)
  - record sklearn version
  - record fold_idx_sha256 (hash of the fold indices themselves)
  - self-consistent labels_sha256
After generation it is written to the log only after passing load_folds mandatory validation.
The fold indices are bit-identical to folds_v1.json (same seed, same split), only metadata is added.
"""
import json
import os
import sys
import platform
import numpy as np
from sklearn.model_selection import StratifiedKFold

import common as C
import folds as F


def main():
    # 1. read labels and ids (order is the train.json array order)
    train = C.load_json(C.TRAIN_JSON)
    ids = [r["id"] for r in train]
    y = np.asarray([int(r["is_iceberg"]) for r in train], dtype=int)
    n = len(ids)
    assert n == 1604, f"unexpected number of train samples: {n}"

    # 2. generate fold indices
    skf = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.SEED)
    folds = []
    for fold, (tr, va) in enumerate(skf.split(np.zeros(n), y)):
        folds.append({
            "fold": int(fold),
            "train_idx": [int(i) for i in tr],
            "val_idx": [int(i) for i in va],
        })

    # 3. compute train SHA-256 at runtime (no hard-coded reuse; falls back to the audit log for consistency)
    train_sha = C.sha256_of_file(C.TRAIN_JSON)

    doc = {
        "meta": {
            "run_id": C.run_id("p1"),
            "timestamp_utc": C.now_utc().isoformat(),
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn_version": C.sklearn_version(),
            "n_train": n,
            "n_splits": C.N_SPLITS,
            "shuffle": True,
            "random_state": C.SEED,
            "splitter": "sklearn.model_selection.StratifiedKFold",
            "stratify_on": "is_iceberg",
            "train_sha256": train_sha,
            "note": "Continues folds_v1.json: fold indices are bit-identical to v1, only metadata is added (sklearn_version / fold_idx_sha256 / runtime train_sha256).",
        },
        "labels": [int(v) for v in y],
        "labels_sha256": F.labels_sha256(y),
        "ids": ids,
        "folds": folds,
    }
    doc["fold_idx_sha256"] = F.fold_indices_sha256(folds)

    out_path = F.FOLDS_V2
    C.save_json_no_overwrite(doc, out_path)

    # 4. mandatory validation via load_folds (no exception means pass)
    F.load_folds(out_path)
    _, report = F.validate_folds(doc, n_samples=n, labels=y)

    # 5. write log (run_id suffix, refuse overwrite)
    log = {
        "run_id": doc["meta"]["run_id"],
        "timestamp_utc": C.now_utc().isoformat(),
        "folds_path": out_path,
        "n_train": n,
        "n_splits": C.N_SPLITS,
        "shuffle": True,
        "random_state": C.SEED,
        "sklearn_version": doc["meta"]["sklearn_version"],
        "train_sha256": train_sha,
        "labels_sha256": doc["labels_sha256"],
        "fold_idx_sha256": doc["fold_idx_sha256"],
        "global_pos_rate": float((y == 1).mean()),
        "per_fold_val_sizes": [len(f["val_idx"]) for f in folds],
        "per_fold_val_pos_count": [
            int((y[np.asarray(f["val_idx"], dtype=int)] == 1).sum()) for f in folds
        ],
        "validate_passed": True,
        "validate_report": report,
        "note": "This log continues logs/p1_folds.json (folds_v1, where the first run's V5 criterion of 1e-6 was a bug causing false negatives) and logs/p1_folds_v2.json (folds_v1 validation passed). This version is the generation log for folds_v2.json.",
    }
    log_path = os.path.join(C.LOGS, f"p1_folds_{doc['meta']['run_id']}.json")
    C.save_json_no_overwrite(log, log_path)

    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P1] folds -> {out_path}")
    print(f"[P1] log   -> {log_path}")


if __name__ == "__main__":
    main()

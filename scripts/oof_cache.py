# -*- coding: utf-8 -*-
"""
OOF cache (source provenance): save/load outer 5-fold OOF predictions + strict source validation.
Cache artifacts include:
  - fold_idx_sha256 / labels_sha256 / train_sha256 (provenance)
  - model config / seeds / code version
  - val_fold_assignment (which fold's val prediction each sample comes from, used for coverage validation).
On load it auto-validates: matching hashes, matching config, complete and non-overlapping val coverage; any failure raises an exception and rejects loading.
"""
import json
import os
import hashlib
import numpy as np

import common as C
import folds as F


def _code_hash(files):
    """Compute SHA-256 over the given script file contents (code version provenance)."""
    h = hashlib.sha256()
    for f in files:
        p = os.path.join(C.SCRIPTS, f)
        h.update(open(p, "rb").read())
        h.update(b"\x00")
    return h.hexdigest()


def save_oof(npz_path, oof_pred, val_fold_assignment, meta):
    """
    oof_pred: (n,) float, predicted probabilities in original sample order
    val_fold_assignment: (n,) int, which fold's val prediction each sample comes from
    meta: dict, must contain fold_idx_sha256/labels_sha256/train_sha256/model_cfg/seeds/code_version/run_id
    """
    if os.path.exists(npz_path):
        raise FileExistsError(f"Refusing to overwrite: {npz_path}")
    np.savez_compressed(npz_path, oof_pred=np.asarray(oof_pred, dtype=np.float32),
                        val_fold_assignment=np.asarray(val_fold_assignment, dtype=np.int8))
    meta_path = npz_path + ".meta.json"
    if os.path.exists(meta_path):
        raise FileExistsError(f"Refusing to overwrite: {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return npz_path, meta_path


def load_oof(npz_path, current_folds=None, expected_model_cfg=None, expected_seeds=None):
    """
    Load OOF and auto-validate; raises AssertionError on any failure.
    current_folds: the current folds_v2 (defaults to loading F.FOLDS_V2)
    """
    if current_folds is None:
        current_folds = F.load_folds(F.FOLDS_V2)
    meta = json.load(open(npz_path + ".meta.json", encoding="utf-8"))

    # source hashes match
    assert meta["fold_idx_sha256"] == current_folds["fold_idx_sha256"], \
        "fold_idx_sha256 mismatch (fold base changed)"
    assert meta["labels_sha256"] == current_folds["labels_sha256"], \
        "labels_sha256 mismatch (labels changed)"
    actual_train_sha = C.sha256_of_file(C.TRAIN_JSON)
    assert meta["train_sha256"] == actual_train_sha, \
        "train_sha256 mismatch (train.json may have been replaced)"

    # config matches
    if expected_model_cfg is not None:
        assert meta["model_cfg"] == expected_model_cfg, "model config mismatch"
    if expected_seeds is not None:
        assert meta["seeds"] == expected_seeds, "seeds mismatch"

    z = np.load(npz_path)
    oof = z["oof_pred"].astype(np.float64)
    vfa = z["val_fold_assignment"].astype(int)

    # val coverage complete and non-overlapping: each sample is predicted by exactly one fold, consistent with the fold index
    n = current_folds["meta"]["n_train"]
    assert len(oof) == n, f"oof length {len(oof)} != n_train {n}"
    expected_vfa = np.full(n, -1, dtype=int)
    for f in current_folds["folds"]:
        va = np.asarray(f["val_idx"], dtype=int)
        assert (vfa[va] == f["fold"]).all(), f"fold {f['fold']} val assignment mismatch"
        expected_vfa[va] = f["fold"]
    assert (vfa == expected_vfa).all(), "val_fold_assignment inconsistent with fold indices (incomplete coverage)"

    return oof, meta

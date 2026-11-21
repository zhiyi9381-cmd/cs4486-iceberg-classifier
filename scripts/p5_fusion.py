# -*- coding: utf-8 -*-
"""
P5 fusion and weight selection (route B: compromise disclosure)
- logit fusion: p = σ(w·logit(p_meta) + (1−w)·logit(p_nn))
- p_meta: LightGBM full-feature (plus_KNN) OOF; p_nn: strictly nested nn_pred (i.e. CNN prediction)
- weight selection nested within fold: for each fold tr_k, use inner OOF to select w_k (outer validation fold does not participate)
  disclosed re-nested indirect dependency: meta inner CV reuses P4 nn_pred (CNN not retrained within itr), magnitude about 1/1026
- also report: descriptive comparison (select w after outer OOF, not an unbiased test) + fixed-weight reference
"""
import json
import os
import glob
import platform
import sys
import numpy as np
import lightgbm as lgb

import common as C
import folds as F
import data as D
import metrics as M
import oof_cache
from p4_meta import build_features, LGBM_CFG

EPS = C.LOGLOSS_EPS


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def fuse(w, p_meta, p_nn):
    return sigmoid(w * logit(p_meta) + (1 - w) * logit(p_nn))


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img, angle, y, ids = D.load_train()
    n = len(y)

    # read strictly nested nn_pred (P4)
    npz = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    z = np.load(npz)
    nn_pred_folds = z["nn_pred_folds"].astype(np.float64)  # (5, n)
    nmeta = json.load(open(npz + ".meta.json", encoding="utf-8"))
    inner_folds = nmeta["inner_folds"]

    # outer_val_fold[i] = the outer val fold of sample i
    outer_val_fold = np.full(n, -1, dtype=int)
    for k, f in enumerate(folds["folds"]):
        outer_val_fold[np.asarray(f["val_idx"], dtype=int)] = k

    run_id = C.run_id("p5")
    code_version = oof_cache._code_hash(["common.py", "folds.py", "data.py", "metrics.py",
                                         "p4_meta.py", "p5_fusion.py"])

    # ---- p_meta outer OOF (each fold meta trains and predicts val_k) + p_nn outer OOF ----
    p_meta_oof = np.zeros(n, dtype=np.float64)
    p_nn_oof = np.zeros(n, dtype=np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        X_tr, X_va, feat_names = build_features(nn_pred_folds[k], angle, y, tr, va)
        m = lgb.LGBMClassifier(**LGBM_CFG)
        m.fit(X_tr, y[tr])
        p_meta_oof[va] = m.predict_proba(X_va)[:, 1]
        p_nn_oof[va] = nn_pred_folds[k, va]

    # ---- nested within-fold selection of w_k ----
    w_candidates = np.round(np.arange(0.0, 1.01, 0.05), 3)
    w_fold = []
    val_fuse = np.zeros(n, dtype=np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        # meta inner CV (reuse P4 inner folds): produce meta inner OOF for tr_k
        p_meta_inner = np.full(len(tr), np.nan)
        tr_pos = {int(idx): pos for pos, idx in enumerate(tr)}
        for inf in inner_folds:
            if inf["outer"] != k:
                continue
            itr = np.asarray(inf["itr"], dtype=int)
            iva = np.asarray(inf["iva"], dtype=int)
            X_it, X_iv, _ = build_features(nn_pred_folds[k], angle, y, itr, iva)
            m = lgb.LGBMClassifier(**LGBM_CFG)
            m.fit(X_it, y[itr])
            pred = m.predict_proba(X_iv)[:, 1]
            # map iva (global index) to position in tr
            pos = [tr_pos[i] for i in iva]
            p_meta_inner[pos] = pred
        assert not np.isnan(p_meta_inner).any(), "meta inner OOF does not cover tr_k"
        p_nn_inner = nn_pred_folds[k, tr]
        # scan w to select the best (on tr_k inner OOF)
        best_w, best_ll = None, np.inf
        for w in w_candidates:
            fused = fuse(w, p_meta_inner, p_nn_inner)
            ll = M.logloss(y[tr], fused)
            if ll < best_ll:
                best_w, best_ll = w, ll
        w_fold.append({"fold": k, "w": float(best_w), "inner_logloss": round(best_ll, 6)})
        # fuse val_k
        val_fuse[va] = fuse(best_w, p_meta_oof[va], p_nn_oof[va])
        print(f"fold {k}: w={best_w:.3f} inner_ll={best_ll:.6f}", flush=True)

    ll_fold_nested = M.logloss(y, val_fuse)

    # ---- descriptive comparison: select w after outer OOF (not an unbiased test) ----
    best_w_descr, best_ll_descr = None, np.inf
    for w in w_candidates:
        fused = fuse(w, p_meta_oof, p_nn_oof)
        ll = M.logloss(y, fused)
        if ll < best_ll_descr:
            best_w_descr, best_ll_descr = w, ll

    # ---- fixed-weight reference ----
    fixed_ref = {}
    for w in [0.5, 0.7, 0.9]:
        fixed_ref[f"w={w}"] = round(M.logloss(y, fuse(w, p_meta_oof, p_nn_oof)), 6)

    # ---- baseline reference ----
    meta_ll = M.logloss(y, p_meta_oof)
    nn_ll = M.logloss(y, p_nn_oof)

    log = {
        "run_id": run_id,
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
        "code_version": code_version,
        "nn_pred_source": npz,
        "seed": C.SEED,
        "logloss_eps": EPS,
        "p_meta_oof_logloss": round(meta_ll, 6),
        "p_nn_oof_logloss": round(nn_ll, 6),
        "w_candidates_step": 0.05,
        "fold_nested_w": w_fold,
        "fold_nested_fusion_logloss": round(ll_fold_nested, 6),
        "descriptive_w": float(best_w_descr),
        "descriptive_fusion_logloss": round(best_ll_descr, 6),
        "fixed_weight_ref": fixed_ref,
        "disclosure": (
            "nested within-fold selection of w (outer validation fold does not participate); "
            "meta inner CV reuses P4 strictly nested nn_pred, CNN not retrained within itr, "
            "there is slight indirect dependency (magnitude about 1/1026), making only the w selection slightly optimistic; the val fusion prediction has no indirect dependency and the CV score is honest."
        ),
    }
    out = os.path.join(C.LOGS, f"p5_fusion_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P5] -> {out}")


if __name__ == "__main__":
    main()

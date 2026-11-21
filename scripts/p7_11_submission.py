# -*- coding: utf-8 -*-
"""
P7.11 v5 submission: multi-seed nn_pred + exact grouping + label_prior(k=2) + handcrafted features (15 dims) + soft_leaky(w=0.5)
handcrafted features (an image information pipeline orthogonal to CNN) improved CV by 0.010 (0.1425 -> 0.1325), but did not transfer to the leaderboard, so it was abandoned
"""
import os
import json
import glob
import hashlib
import platform
import sys
from collections import defaultdict
import numpy as np
import lightgbm as lgb

import common as C
import folds as F
import data as D
import oof_cache
from p4_meta import LGBM_CFG
from p7_5_submission_v2 import build_tr_va_exact, build_te_exact
from p7_7_labelprior import build_label_prior
from p7_11_handcrafted import extract_features

LGBM_SEEDS = [42, 7, 2024]
PRIOR_HI = 0.99
PRIOR_LO = 0.01
LP_K = 2
W_SOFT = 0.5


def build_label_prior_test(angle_train, y_train, angle_test, k):
    a_tr = np.where(np.isnan(angle_train), np.nanmedian(angle_train), angle_train).astype(np.float64)
    a_te = np.asarray(angle_test, dtype=np.float64)
    out = np.zeros(len(a_te), dtype=np.float64)
    for t in range(len(a_te)):
        d = np.abs(a_tr - a_te[t])
        kk = min(k, len(a_tr))
        out[t] = y_train[np.argpartition(d, kk - 1)[:kk]].mean()
    return out


def leaky_soft_test(angle_train, y_train, angle_test, pred, w):
    groups = defaultdict(list)
    for i, a in enumerate(angle_train):
        if not np.isnan(a):
            groups[round(float(a), 4)].append(int(y_train[i]))
    pred = np.asarray(pred, dtype=np.float64).copy()
    n = 0
    for t in range(len(angle_test)):
        a = round(float(angle_test[t]), 4)
        if a in groups and len(set(groups[a])) == 1:
            prior = PRIOR_HI if groups[a][0] == 1 else PRIOR_LO
            pred[t] = w * pred[t] + (1 - w) * prior
            n += 1
    return pred, n


def fit_multi_lgbm(X_tr, y_tr, X_pr):
    logits = []
    for s in LGBM_SEEDS:
        m = lgb.LGBMClassifier(**dict(LGBM_CFG, random_state=s)).fit(X_tr, y_tr)
        p = np.clip(m.predict_proba(X_pr)[:, 1], 1e-7, 1 - 1e-7)
        logits.append(np.log(p / (1 - p)))
    return 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img_tr, angle_tr, y, ids_tr = D.load_train()
    img_te, angle_te, ids_te = D.load_test()
    n_test = len(ids_te)

    # handcrafted features (train + test)
    print("extracting train handcrafted features...", flush=True)
    H_tr = extract_features(img_tr.astype(np.float32))
    print("extracting test handcrafted features...", flush=True)
    H_te = extract_features(img_te.astype(np.float32))
    n_hand = H_tr.shape[1]

    npz_ms = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_multiseed_*.npz")))[-1]
    z = np.load(npz_ms)
    nn_pred_folds = z["nn_pred_folds"].astype(np.float64)
    test_nn_pred = z["test_nn_pred"].astype(np.float64)
    mmeta = json.load(open(npz_ms + ".meta.json", encoding="utf-8"))
    assert mmeta["fold_idx_sha256"] == folds["fold_idx_sha256"]

    npz_nested = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    nn_nested = np.load(npz_nested)["nn_pred_folds"].astype(np.float64)
    nan_mask = np.isnan(nn_pred_folds)
    nn_pred_folds[nan_mask] = nn_nested[nan_mask]
    n_nan_fixed = int(nan_mask.sum())

    run_id = C.run_id("p711")
    code_version = oof_cache._code_hash(["common.py", "folds.py", "data.py", "metrics.py",
                                         "p4_meta.py", "p7_5_submission_v2.py", "p7_7_labelprior.py",
                                         "p7_11_handcrafted.py", "p7_11_submission.py"])

    # CV evaluation
    from metrics import logloss as ll, mean_std
    oof = np.zeros(len(y), dtype=np.float64)
    for kf, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int); va = np.asarray(f["val_idx"], dtype=int)
        X_tr, X_va = build_tr_va_exact(nn_pred_folds[kf], angle_tr, y, tr, va)
        lp_tr = build_label_prior(angle_tr, y, tr, tr, LP_K)
        lp_va = build_label_prior(angle_tr, y, tr, va, LP_K)
        X_tr = np.column_stack([X_tr, lp_tr, H_tr[tr]])
        X_va = np.column_stack([X_va, lp_va, H_tr[va]])
        oof[va] = fit_multi_lgbm(X_tr, y[tr], X_va)
    mm, ss = mean_std([ll(y[np.asarray(f["val_idx"], dtype=int)], oof[np.asarray(f["val_idx"], dtype=int)]) for f in folds["folds"]])
    print(f"CV meta+label_prior+handcrafted features: {mm:.6f} ± {ss:.6f}", flush=True)
    # soft leaky
    p_soft = np.zeros(len(y))
    for f in folds["folds"]:
        tr = np.asarray(f["train_idx"], dtype=int); va = np.asarray(f["val_idx"], dtype=int)
        groups = defaultdict(list)
        for i in tr:
            if not np.isnan(angle_tr[i]): groups[round(float(angle_tr[i]), 4)].append(int(y[i]))
        pred = oof[va].copy()
        for pos, i in enumerate(va):
            a = round(float(angle_tr[i]), 4)
            if a in groups and len(set(groups[a])) == 1:
                prior = PRIOR_HI if groups[a][0] == 1 else PRIOR_LO
                pred[pos] = W_SOFT * pred[pos] + (1 - W_SOFT) * prior
        p_soft[va] = pred
    mm2, ss2 = mean_std([ll(y[np.asarray(f["val_idx"], dtype=int)], p_soft[np.asarray(f["val_idx"], dtype=int)]) for f in folds["folds"]])
    print(f"CV + soft_leaky: {mm2:.6f} ± {ss2:.6f}", flush=True)

    # v5 submission
    test_preds = np.zeros((C.N_SPLITS, n_test), dtype=np.float64)
    for kf, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int); va = np.asarray(f["val_idx"], dtype=int)
        X_tr, _ = build_tr_va_exact(nn_pred_folds[kf], angle_tr, y, tr, va)
        lp_tr = build_label_prior(angle_tr, y, tr, tr, LP_K)
        X_tr = np.column_stack([X_tr, lp_tr, H_tr[tr]])
        X_te = build_te_exact(nn_pred_folds[kf], angle_tr, test_nn_pred[kf], angle_te, tr)
        lp_te = build_label_prior_test(angle_tr, y, angle_te, LP_K)
        X_te = np.column_stack([X_te, lp_te, H_te])
        test_preds[kf] = fit_multi_lgbm(X_tr, y[tr], X_te)
        print(f"fold {kf} test done", flush=True)
    test_mean = test_preds.mean(axis=0)
    test_final, n_ovr = leaky_soft_test(angle_tr, y, angle_te, test_mean, W_SOFT)

    checks = {
        "pred_in_01": bool((test_final > 0).all() and (test_final < 1).all()),
        "pred_min": float(test_final.min()), "pred_max": float(test_final.max()),
        "n_override": int(n_ovr), "n_test": n_test,
        "id_unique": bool(len(set(ids_te)) == n_test),
    }
    pred_npz = os.path.join(C.ROOT, "oof", f"test_preds_5fold_v5_{run_id}.npz")
    np.savez_compressed(pred_npz, test_preds=test_preds, test_final=test_final)
    sub_dir = os.path.join(C.ROOT, "submissions"); os.makedirs(sub_dir, exist_ok=True)
    csv_path = os.path.join(sub_dir, f"submission_v5_{run_id}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,is_iceberg\n")
        for i in range(n_test):
            f.write(f"{ids_te[i]},{test_final[i]:.6f}\n")
    csv_sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()

    log = {
        "run_id": run_id, "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(), "python": sys.version.split()[0], "lightgbm": lgb.__version__,
        "code_version": code_version,
        "model": "multiseed_DeepCNN + exact_angle_group + label_prior_k2 + handcrafted_15 + multi_LGBM + soft_leaky_w0.5",
        "nn_pred_source": npz_ms, "lgbm_seeds": LGBM_SEEDS,
        "label_prior_k": LP_K, "soft_leaky_w": W_SOFT, "n_handcrafted": n_hand,
        "cv_meta_handcrafted": round(mm, 6),
        "cv_meta_handcrafted_softleaky": round(mm2, 6),
        "nan_fix": n_nan_fixed,
        "pred_checks": checks, "csv_path": csv_path, "csv_sha256": csv_sha, "pred_npz": pred_npz,
        "disclosure": "handcrafted features are a new image-side information source (orthogonal to CNN); label_prior/soft_leaky are based on train-label patterns (no leaderboard feedback).",
    }
    out = os.path.join(C.LOGS, f"p7_11_submission_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P7.11] csv -> {csv_path}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
P7.6 meta upgrade + post-processing + v3 submission (corrected version)
- input: multi-seed nn_pred (produced by p7_6_bagging.py): nn_pred_folds(5,n) + test_nn_pred(5,n_test)
- meta features: exact inc_angle grouping (4 decimal places) + KNN (reusing p7_5's build_*_exact)
- multiple LightGBM variants: 3 LGBMs with different random_state, logit-averaged
- post-processing: soft_leaky soft fusion (the quick experiment already proved sharpening is harmful and soft fusion beats hard clipping), auto-select the best w
"""
import json
import os
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

LGBM_SEEDS = [42, 7, 2024]
PRIOR_HI = 0.99
PRIOR_LO = 0.01
W_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7]   # soft leaky soft-fusion weight candidates (0=hard clipping)


def leaky_soft(angle, y, tr, idx, pred, w):
    """Soft fusion: pure-group sample p = w*p + (1-w)*prior. w=0 means hard clipping. prior is determined by pure-group labels within tr."""
    groups = defaultdict(list)
    for i in tr:
        if not np.isnan(angle[i]):
            groups[round(float(angle[i]), 4)].append(int(y[i]))
    pred = np.asarray(pred, dtype=np.float64).copy()
    n_ovr = 0
    for pos, i in enumerate(idx):
        a = round(float(angle[i]), 4)
        if a in groups and len(set(groups[a])) == 1:
            prior = PRIOR_HI if groups[a][0] == 1 else PRIOR_LO
            pred[pos] = w * pred[pos] + (1 - w) * prior
            n_ovr += 1
    return pred, n_ovr


def fit_predict_lgbm(X_tr, y_tr, X_pr):
    """3 LGBM variants logit-averaged."""
    logits = []
    for s in LGBM_SEEDS:
        m = lgb.LGBMClassifier(**dict(LGBM_CFG, random_state=s)).fit(X_tr, y_tr)
        p = np.clip(m.predict_proba(X_pr)[:, 1], 1e-7, 1 - 1e-7)
        logits.append(np.log(p / (1 - p)))
    return 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))


def leaky_override_test_soft(angle_train, y_train, angle_test, pred, w):
    """test-side soft fusion (using the full train pure groups)."""
    groups = defaultdict(list)
    for i, a in enumerate(angle_train):
        if not np.isnan(a):
            groups[round(float(a), 4)].append(int(y_train[i]))
    pred = np.asarray(pred, dtype=np.float64).copy()
    n_ovr = 0
    for t in range(len(angle_test)):
        a = round(float(angle_test[t]), 4)
        if a in groups and len(set(groups[a])) == 1:
            prior = PRIOR_HI if groups[a][0] == 1 else PRIOR_LO
            pred[t] = w * pred[t] + (1 - w) * prior
            n_ovr += 1
    return pred, n_ovr


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img_tr, angle_tr, y, ids_tr = D.load_train()
    img_te, angle_te, ids_te = D.load_test()
    n_test = len(ids_te)

    npz_ms = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_multiseed_*.npz")))[-1]
    z = np.load(npz_ms)
    nn_pred_folds = z["nn_pred_folds"].astype(np.float64)
    test_nn_pred = z["test_nn_pred"].astype(np.float64)
    mmeta = json.load(open(npz_ms + ".meta.json", encoding="utf-8"))
    assert mmeta["fold_idx_sha256"] == folds["fold_idx_sha256"]

    # fix: p7_6_bagging's train_predict did not fill missing val angles, causing 133 NaNs in the val part.
    # fill these positions with P4 single-seed nn_pred_nested (angle correctly filled) (direction correct, no need to retrain).
    npz_nested = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    nn_nested = np.load(npz_nested)["nn_pred_folds"].astype(np.float64)
    nan_mask = np.isnan(nn_pred_folds)
    n_nan_fixed = int(nan_mask.sum())
    assert np.isnan(nn_nested).sum() == 0, "P4 nn_pred_nested should not contain NaN"
    nn_pred_folds[nan_mask] = nn_nested[nan_mask]
    assert not np.isnan(nn_pred_folds).any(), "nn_pred_folds still contains NaN after fix"
    print(f"[fix] filled {n_nan_fixed} val NaNs with P4 single-seed nn_pred (train_predict angle_pr not filled bug)", flush=True)

    run_id = C.run_id("p76meta")
    code_version = oof_cache._code_hash(["common.py", "folds.py", "data.py", "metrics.py",
                                         "p4_meta.py", "p7_5_submission_v2.py", "p7_6_meta.py"])

    # ---- CV: multi-seed nn_pred + exact grouping + multi-LGBM OOF ----
    oof = np.zeros(len(y), dtype=np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        X_tr, X_va = build_tr_va_exact(nn_pred_folds[k], angle_tr, y, tr, va)
        oof[va] = fit_predict_lgbm(X_tr, y[tr], X_va)
        print(f"fold {k} meta OOF done", flush=True)

    from metrics import logloss as ll, mean_std
    schemes = {}
    def cv_of(p):
        lls = [ll(y[np.asarray(f["val_idx"], dtype=int)], p[np.asarray(f["val_idx"], dtype=int)]) for f in folds["folds"]]
        return mean_std(lls)

    mm, ss = cv_of(oof)
    schemes["meta_base"] = {"mean": round(mm, 6), "std": round(ss, 6)}
    print(f"meta_base: {mm:.6f} ± {ss:.6f}", flush=True)

    best_w = None
    for w in W_CANDIDATES:
        p = np.zeros(len(y), dtype=np.float64)
        for f in folds["folds"]:
            tr = np.asarray(f["train_idx"], dtype=int); va = np.asarray(f["val_idx"], dtype=int)
            p[va], _ = leaky_soft(angle_tr, y, tr, va, oof[va], w)
        mm, ss = cv_of(p)
        schemes[f"softleaky_w{w}"] = {"mean": round(mm, 6), "std": round(ss, 6)}
        print(f"softleaky_w{w}: {mm:.6f} ± {ss:.6f}", flush=True)
        if best_w is None or mm < schemes[f"softleaky_w{best_w}"]["mean"]:
            best_w = w

    # hard clipping reference
    p = np.zeros(len(y), dtype=np.float64)
    for f in folds["folds"]:
        tr = np.asarray(f["train_idx"], dtype=int); va = np.asarray(f["val_idx"], dtype=int)
        p[va], _ = leaky_soft(angle_tr, y, tr, va, oof[va], 0.0)
    mm, ss = cv_of(p)
    schemes["hardleaky_w0"] = {"mean": round(mm, 6), "std": round(ss, 6)}
    print(f"hardleaky_w0: {mm:.6f} ± {ss:.6f}", flush=True)

    print(f"\nbest soft leaky w = {best_w} (CV={schemes[f'softleaky_w{best_w}']['mean']})", flush=True)

    # ---- v3 submission: multi-seed nn_pred + exact grouping + multi-LGBM + soft leaky (best w) ----
    test_preds = np.zeros((C.N_SPLITS, n_test), dtype=np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        X_tr, _ = build_tr_va_exact(nn_pred_folds[k], angle_tr, y, tr, va)
        X_te = build_te_exact(nn_pred_folds[k], angle_tr, test_nn_pred[k], angle_te, tr)
        test_preds[k] = fit_predict_lgbm(X_tr, y[tr], X_te)
        print(f"fold {k} test prediction done", flush=True)
    test_mean = test_preds.mean(axis=0)
    test_final, n_ovr = leaky_override_test_soft(angle_tr, y, angle_te, test_mean, best_w)

    checks = {
        "pred_in_01": bool((test_final > 0).all() and (test_final < 1).all()),
        "pred_min": float(test_final.min()), "pred_max": float(test_final.max()),
        "n_override": int(n_ovr), "n_test": n_test,
        "id_unique": bool(len(set(ids_te)) == n_test),
    }

    pred_npz = os.path.join(C.ROOT, "oof", f"test_preds_5fold_v3_{run_id}.npz")
    if os.path.exists(pred_npz):
        raise FileExistsError(f"Refuse to overwrite (R8): {pred_npz}")
    np.savez_compressed(pred_npz, test_preds=test_preds, test_final=test_final)

    sub_dir = os.path.join(C.ROOT, "submissions"); os.makedirs(sub_dir, exist_ok=True)
    csv_path = os.path.join(sub_dir, f"submission_v3_{run_id}.csv")
    if os.path.exists(csv_path):
        raise FileExistsError(f"Refuse to overwrite (R8): {csv_path}")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,is_iceberg\n")
        for i in range(n_test):
            f.write(f"{ids_te[i]},{test_final[i]:.6f}\n")
    csv_sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()

    log = {
        "run_id": run_id,
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
        "code_version": code_version,
        "model": "multiseed_bagging_DeepCNN + exact_angle_group + multi_LGBM + soft_leaky(best w)",
        "nn_pred_source": npz_ms,
        "lgbm_seeds": LGBM_SEEDS,
        "cv_schemes": schemes,
        "cv_best": f"softleaky_w{best_w}",
        "best_w": best_w,
        "nan_fix": f"the {n_nan_fixed} NaNs in val nn_pred (angle-missing samples) were filled with P4 single-seed nn_pred",
        "sharpening": "not used (quick experiment p7_6_experiment confirmed sharpening 1.15 raised CV from 0.1586 to 0.1666, harmful)",
        "pred_checks": checks,
        "csv_path": csv_path,
        "csv_sha256": csv_sha,
        "pred_npz": pred_npz,
        "disclosure": "multi-seed bagging nn_pred's val/test use 5 seeds+TTA, tr_k reuses P4 single-seed inner OOF; soft_leaky soft fusion is based on train-label patterns (no leaderboard feedback), and Chapter 5 of the report must reference the inc_angle association discussion.",
    }
    out = os.path.join(C.LOGS, f"p7_6_meta_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P7.6-meta] csv -> {csv_path}")
    print(f"[P7.6-meta] log -> {out}")


if __name__ == "__main__":
    main()

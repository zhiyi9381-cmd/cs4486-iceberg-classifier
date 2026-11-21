# -*- coding: utf-8 -*-
"""
P7 submission generation (final model = pure meta plus_KNN, P5 conclusion)
- average the five-fold predictions on the test set:
    each fold k: CNN_k trained on tr_k -> predicts test (as the nn_pred feature for meta)
            meta_k trained on tr_k (features = nn_pred_folds[k]+angle+bin+KNN) -> predicts test
- output CSV (id,is_iceberg 8424 rows + header) + per-fold test prediction npz + CSV SHA-256
- R8 versioned naming, refuse to overwrite
- checks: prediction range (0,1), no all-0.5, id order consistent with test.json
"""
import json
import os
import glob
import platform
import sys
import hashlib
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

import common as C
import folds as F
import data as D
import oof_cache
from deepcnn import DeepCNN
from p4_meta import LGBM_CFG, N_BINS, K

EPOCHS = 60
BATCH = 32
LR = 1e-3
WD = 1e-4
SEED = 42
DROPOUT = 0.4


def train_cnn_predict_test(img_tr, angle_tr, y_tr, img_pr, angle_pr):
    """Train DeepCNN (cosine+dropout0.4) on a train subset, predict independent test data."""
    tr = np.asarray(img_tr)
    med = float(np.nanmedian(angle_tr))
    a = np.where(np.isnan(angle_tr), med, angle_tr).astype(np.float32)
    mu = float(a.mean()); sd = float(a.std(ddof=0)) or 1.0
    a_tr = ((a - mu) / sd).astype(np.float32)
    a_pr = ((np.asarray(angle_pr, dtype=np.float32) - mu) / sd).astype(np.float32)

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = DeepCNN(dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.BCEWithLogitsLoss()

    nt = len(tr)
    Xtr = torch.from_numpy(tr); Atr = torch.from_numpy(a_tr[:, None])
    yt = torch.from_numpy(np.asarray(y_tr, dtype=np.float32))[:, None]
    Xpr = torch.from_numpy(img_pr); Apr = torch.from_numpy(a_pr[:, None])

    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(nt)
        for i in range(0, nt, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx], Atr[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xpr, Apr)).numpy().reshape(-1)
    return prob


def build_test_features(nn_pred_train, angle_train, nn_pred_test, angle_test, tr):
    """Build test meta features (6 cols), bin/KNN computed from tr samples' nn_pred."""
    med = float(np.nanmedian(angle_train[tr]))
    a_fill = np.where(np.isnan(angle_train), med, angle_train).astype(np.float64)
    mu = float(a_fill[tr].mean()); sd = float(a_fill[tr].std(ddof=0)) or 1.0
    angle_test_norm = ((np.asarray(angle_test, dtype=np.float64) - mu) / sd).astype(np.float32)

    qs = np.quantile(a_fill[tr], np.linspace(0, 1, N_BINS + 1))
    qs[0] = -np.inf; qs[-1] = np.inf
    bins_tr = np.clip(np.digitize(a_fill[tr], qs) - 1, 0, N_BINS - 1)
    bins_te = np.clip(np.digitize(np.asarray(angle_test, dtype=np.float64), qs) - 1, 0, N_BINS - 1)
    nn_tr = nn_pred_train[tr]

    nt = len(angle_test)
    b_med = np.zeros(nt, dtype=np.float32); b_mean = np.zeros(nt, dtype=np.float32)
    b_cnt = np.zeros(nt, dtype=np.float32); kpr = np.zeros(nt, dtype=np.float32)
    a_tr = a_fill[tr]
    for t in range(nt):
        mask = bins_tr == bins_te[t]
        grp = nn_tr[mask]
        b_med[t] = np.median(grp) if len(grp) else 0.5
        b_mean[t] = grp.mean() if len(grp) else 0.5
        b_cnt[t] = np.log1p(len(grp))
        dist = np.abs(a_tr - angle_test[t])
        kk = min(K, len(tr))
        nn_idx = np.argpartition(dist, kk - 1)[:kk]
        kpr[t] = nn_tr[nn_idx].mean()

    return np.column_stack([np.asarray(nn_pred_test, dtype=np.float32), angle_test_norm,
                            b_med, b_mean, b_cnt, kpr])


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img_tr, angle_tr, y, ids_tr = D.load_train()
    img_te, angle_te, ids_te = D.load_test()
    img_tr = img_tr.astype(np.float32); img_te = img_te.astype(np.float32)
    n_test = len(ids_te)

    # read strictly nested nn_pred (P4)
    npz = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    z = np.load(npz)
    nn_pred_folds = z["nn_pred_folds"].astype(np.float64)
    nmeta = json.load(open(npz + ".meta.json", encoding="utf-8"))
    assert nmeta["fold_idx_sha256"] == folds["fold_idx_sha256"]

    run_id = C.run_id("p7")
    code_version = oof_cache._code_hash(["common.py", "folds.py", "data.py", "metrics.py",
                                         "deepcnn.py", "p4_meta.py", "p7_submission.py"])

    test_preds = np.zeros((C.N_SPLITS, n_test), dtype=np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        # CNN_k: train on tr -> predict test (nn_pred feature for meta)
        cnn_test = train_cnn_predict_test(img_tr[tr], angle_tr[tr], y[tr], img_te, angle_te)
        # meta_k: train on tr -> predict test
        from p4_meta import build_features
        X_tr, _, feat_names = build_features(nn_pred_folds[k], angle_tr, y, tr, va)
        m = lgb.LGBMClassifier(**LGBM_CFG)
        m.fit(X_tr, y[tr])
        X_te = build_test_features(nn_pred_folds[k], angle_tr, cnn_test, angle_te, tr)
        test_preds[k] = m.predict_proba(X_te)[:, 1]
        print(f"fold {k}: test prediction done (mean={test_preds[k].mean():.4f})", flush=True)

    test_mean = test_preds.mean(axis=0)

    # checks (ids_te returned by D.load_test in test.json order, CSV written in the same order)
    checks = {
        "pred_in_01": bool((test_mean > 0).all() and (test_mean < 1).all()),
        "pred_min": float(test_mean.min()),
        "pred_max": float(test_mean.max()),
        "not_all_05": bool(np.abs(test_mean - 0.5).max() > 1e-6),
        "n_test": n_test,
        "id_unique": bool(len(set(ids_te)) == n_test),
    }

    # save per-fold test prediction npz
    pred_npz = os.path.join(C.ROOT, "oof", f"test_preds_5fold_{run_id}.npz")
    if os.path.exists(pred_npz):
        raise FileExistsError(f"Refuse to overwrite (R8): {pred_npz}")
    np.savez_compressed(pred_npz, test_preds=test_preds)

    # output CSV (R8 versioned naming)
    sub_dir = os.path.join(C.ROOT, "submissions")
    os.makedirs(sub_dir, exist_ok=True)
    csv_path = os.path.join(sub_dir, f"submission_v1_{run_id}.csv")
    if os.path.exists(csv_path):
        raise FileExistsError(f"Refuse to overwrite (R8): {csv_path}")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,is_iceberg\n")
        for i in range(n_test):
            f.write(f"{ids_te[i]},{test_mean[i]:.6f}\n")

    # CSV SHA-256
    csv_sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()

    log = {
        "run_id": run_id,
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "lightgbm": lgb.__version__,
        "code_version": code_version,
        "final_model": "pure_meta_plus_KNN (P5 conclusion, no CNN fusion)",
        "nn_pred_source": npz,
        "n_test": n_test,
        "pred_checks": checks,
        "csv_path": csv_path,
        "csv_sha256": csv_sha,
        "pred_npz": pred_npz,
        "test_pred_mean_overall": float(test_mean.mean()),
    }
    out = os.path.join(C.LOGS, f"p7_submission_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P7] csv -> {csv_path}")
    print(f"[P7] log -> {out}")


if __name__ == "__main__":
    main()

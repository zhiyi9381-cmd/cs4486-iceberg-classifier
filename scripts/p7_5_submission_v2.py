# -*- coding: utf-8 -*-
"""
P7.5 v2 submission: exact inc_angle grouping + soft-leaky post-processing (clip 0.99)
- exact angle grouping (4 decimal places) replaces the 30 equal-frequency bins (more precise group statistics, exploiting the "same angle -> same label" pattern)
- soft-leaky post-processing: test samples whose angle is in a train pure group (single label) are overridden with the clip probability (0.99/0.01)
- strict: post-processing uses only train-label patterns (no leaderboard feedback); the inc_angle association feature must be disclosed in the report
"""
import json
import os
import glob
import hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

import common as C
import folds as F
import data as D
import oof_cache
from deepcnn import DeepCNN
from p4_meta import LGBM_CFG, K

EPOCHS = 60; BATCH = 32; LR = 1e-3; WD = 1e-4; SEED = 42; DROPOUT = 0.4
CLIP_HI = 0.99; CLIP_LO = 0.01


def train_cnn_predict_test(img_tr, angle_tr, y_tr, img_pr, angle_pr):
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
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xpr, Apr)).numpy().reshape(-1)


def build_tr_va_exact(nn_pred, angle, y, tr, va):
    med = float(np.nanmedian(angle[tr]))
    a_fill = np.where(np.isnan(angle), med, angle).astype(np.float64)
    mu = float(a_fill[tr].mean()); sd = float(a_fill[tr].std(ddof=0)) or 1.0
    angle_norm = ((a_fill - mu) / sd).astype(np.float32)
    key_tr = np.round(a_fill[tr], 4)

    def gs(idx_all, excl):
        bm = np.zeros(len(idx_all), np.float32); bn = np.zeros(len(idx_all), np.float32)
        bc = np.zeros(len(idx_all), np.float32)
        for p, i in enumerate(idx_all):
            m = key_tr == round(float(a_fill[i]), 4)
            if excl:
                m &= (tr != i)
            g = nn_pred[tr[m]]
            bm[p] = np.median(g) if len(g) else 0.5
            bn[p] = g.mean() if len(g) else 0.5
            bc[p] = np.log1p(len(g))
        return bm, bn, bc

    def ks(idx_all, excl):
        kp = np.zeros(len(idx_all), np.float32)
        for p, i in enumerate(idx_all):
            d = np.abs(a_fill[tr] - a_fill[i]).copy()
            if excl:
                d[np.where(tr == i)[0]] = np.inf
            kp[p] = nn_pred[tr[np.argpartition(d, min(K, len(tr)) - 1)[:K]]].mean()
        return kp

    def asm(idx_all, excl):
        return np.column_stack([nn_pred[idx_all], angle_norm[idx_all], *gs(idx_all, excl), ks(idx_all, excl)])
    return asm(tr, True), asm(va, False)


def build_te_exact(nn_pred_train, angle_train, nn_pred_test, angle_test, tr):
    med = float(np.nanmedian(angle_train[tr]))
    a_fill = np.where(np.isnan(angle_train), med, angle_train).astype(np.float64)
    mu = float(a_fill[tr].mean()); sd = float(a_fill[tr].std(ddof=0)) or 1.0
    angle_norm = ((np.asarray(angle_test, np.float64) - mu) / sd).astype(np.float32)
    key_tr = np.round(a_fill[tr], 4)
    nn_tr = nn_pred_train[tr]
    nt = len(angle_test)
    bm = np.zeros(nt, np.float32); bn = np.zeros(nt, np.float32); bc = np.zeros(nt, np.float32)
    kp = np.zeros(nt, np.float32)
    a_tr = a_fill[tr]
    for t in range(nt):
        m = key_tr == round(float(angle_test[t]), 4)
        g = nn_tr[m]
        bm[t] = np.median(g) if len(g) else 0.5
        bn[t] = g.mean() if len(g) else 0.5
        bc[t] = np.log1p(len(g))
        d = np.abs(a_tr - angle_test[t])
        kp[t] = nn_tr[np.argpartition(d, min(K, len(tr)) - 1)[:K]].mean()
    return np.column_stack([np.asarray(nn_pred_test, np.float32), angle_norm, bm, bn, bc, kp])


def leaky_override_test(angle_train, y_train, angle_test, pred):
    groups = defaultdict(list)
    for i, a in enumerate(angle_train):
        if not np.isnan(a):
            groups[round(float(a), 4)].append(int(y_train[i]))
    pred = np.asarray(pred, np.float64).copy()
    n_ovr = 0
    for t in range(len(angle_test)):
        a = round(float(angle_test[t]), 4)
        if a in groups and len(set(groups[a])) == 1:
            pred[t] = CLIP_HI if groups[a][0] == 1 else CLIP_LO
            n_ovr += 1
    return pred, n_ovr


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img_tr, angle_tr, y, ids_tr = D.load_train()
    img_te, angle_te, ids_te = D.load_test()
    img_tr = img_tr.astype(np.float32); img_te = img_te.astype(np.float32)
    n_test = len(ids_te)
    npz = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    nn_pred_folds = np.load(npz)["nn_pred_folds"].astype(np.float64)

    run_id = C.run_id("p75v2")
    code_version = oof_cache._code_hash(["common.py", "folds.py", "data.py", "metrics.py",
                                         "deepcnn.py", "p4_meta.py", "p7_5_submission_v2.py"])

    test_preds = np.zeros((C.N_SPLITS, n_test), np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], int); va = np.asarray(f["val_idx"], int)
        cnn_test = train_cnn_predict_test(img_tr[tr], angle_tr[tr], y[tr], img_te, angle_te)
        X_tr, _ = build_tr_va_exact(nn_pred_folds[k], angle_tr, y, tr, va)
        m = lgb.LGBMClassifier(**LGBM_CFG).fit(X_tr, y[tr])
        X_te = build_te_exact(nn_pred_folds[k], angle_tr, cnn_test, angle_te, tr)
        test_preds[k] = m.predict_proba(X_te)[:, 1]
        print(f"fold {k} done", flush=True)
    test_mean = test_preds.mean(axis=0)

    # soft-leaky post-processing (full train pure groups)
    test_final, n_ovr = leaky_override_test(angle_tr, y, angle_te, test_mean)

    # checks
    checks = {
        "pred_in_01": bool((test_final > 0).all() and (test_final < 1).all()),
        "pred_min": float(test_final.min()), "pred_max": float(test_final.max()),
        "n_override": int(n_ovr), "n_test": n_test,
        "id_unique": bool(len(set(ids_te)) == n_test),
    }

    pred_npz = os.path.join(C.ROOT, "oof", f"test_preds_5fold_v2_{run_id}.npz")
    np.savez_compressed(pred_npz, test_preds=test_preds, test_final=test_final)
    sub_dir = os.path.join(C.ROOT, "submissions"); os.makedirs(sub_dir, exist_ok=True)
    csv_path = os.path.join(sub_dir, f"submission_v2_{run_id}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,is_iceberg\n")
        for i in range(n_test):
            f.write(f"{ids_te[i]},{test_final[i]:.6f}\n")
    csv_sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()

    log = {"run_id": run_id, "timestamp_utc": C.now_utc().isoformat(),
           "code_version": code_version, "model": "meta_plusKNN + exact_angle_group + leaky_clip0.99",
           "cv_logloss_reference": 0.154778,
           "clip_hi": CLIP_HI, "clip_lo": CLIP_LO,
           "pred_checks": checks, "csv_path": csv_path, "csv_sha256": csv_sha, "pred_npz": pred_npz,
           "disclosure": "soft-leaky post-processing exploits the competition's incident-angle association (samples sharing the same inc_angle tend to share the same label), derived solely from training-label statistics (no leaderboard feedback). Referenced in Section V of the report."}
    out = os.path.join(C.LOGS, f"p7_5_v2_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P7.5-v2] csv -> {csv_path}")


if __name__ == "__main__":
    main()

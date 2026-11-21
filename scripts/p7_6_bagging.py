# -*- coding: utf-8 -*-
"""
P7.6 multi-seed DeepCNN bagging + TTA to generate nn_pred features
- 5 seeds [42, 7, 2024, 1234, 0] × 5 folds = 25 trainings (reusing the cosine+dropout0.4 config validated in P3/P6)
- per fold, do 4-way flip TTA (original/horizontal/vertical/180) on val_k and test, averaged over 5 seeds
- output nn_pred's val part (5,n) and test part (5,n_test) for meta use
- tr_k samples' nn_pred reuses P4 strictly nested single-seed inner OOF (avoid cross-layer label indirect dependency)
"""
import json
import os
import glob
import platform
import sys
import numpy as np
import torch
import torch.nn as nn

import common as C
import folds as F
import data as D
import oof_cache
from deepcnn import DeepCNN, count_params

EPOCHS = 60
BATCH = 32
LR = 1e-3
WD = 1e-4
DROPOUT = 0.4
SEEDS = [42, 7, 2024, 1234, 0]


def predict_tta(model, X, A):
    """4-way flip TTA (original / horizontal / vertical / 180), average predicted probabilities. X: (B,C,H,W) tensor."""
    preds = []
    for flip_h in (False, True):
        for flip_v in (False, True):
            Xt = X
            if flip_h:
                Xt = torch.flip(Xt, dims=[3])   # horizontal (width)
            if flip_v:
                Xt = torch.flip(Xt, dims=[2])   # vertical (height)
            with torch.no_grad():
                p = torch.sigmoid(model(Xt, A)).numpy().reshape(-1)
            preds.append(p)
    return np.mean(preds, axis=0)


def train_predict(img_tr, angle_tr, y_tr, img_pr, angle_pr, seed):
    """Train one seeded CNN on tr (cosine+dropout0.4), and do TTA prediction on pr."""
    tr = np.asarray(img_tr)
    med = float(np.nanmedian(angle_tr))
    a = np.where(np.isnan(angle_tr), med, angle_tr).astype(np.float32)
    mu = float(a.mean()); sd = float(a.std(ddof=0)) or 1.0
    a_tr = ((a - mu) / sd).astype(np.float32)
    a_pr = ((np.asarray(angle_pr, dtype=np.float32) - mu) / sd).astype(np.float32)

    torch.manual_seed(seed); np.random.seed(seed)
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
    return predict_tta(model, Xpr, Apr)


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img_tr, angle_tr, y, ids_tr = D.load_train()
    img_te, angle_te, ids_te = D.load_test()
    img_tr = img_tr.astype(np.float32); img_te = img_te.astype(np.float32)
    n = len(y); n_test = len(ids_te)

    # reuse P4 strictly nested single-seed nn_pred (inner OOF of the tr_k part)
    npz_nested = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    z_nested = np.load(npz_nested)
    nn_pred_nested = z_nested["nn_pred_folds"].astype(np.float64)   # (5, n) strictly nested single-seed
    assert json.load(open(npz_nested + ".meta.json", encoding="utf-8"))["fold_idx_sha256"] == folds["fold_idx_sha256"]

    per_layer, total = count_params(DeepCNN())
    run_id = C.run_id("p76cnn")
    code_version = oof_cache._code_hash(["common.py", "folds.py", "data.py", "metrics.py",
                                         "deepcnn.py", "p7_6_bagging.py"])

    nn_pred_folds = np.zeros((C.N_SPLITS, n), dtype=np.float64)     # val_k part = multi-seed average
    test_nn_pred = np.zeros((C.N_SPLITS, n_test), dtype=np.float64)  # test multi-seed average
    per_fold_val_logloss = []

    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        # tr_k part reuses P4 inner OOF (single seed)
        nn_pred_folds[k, tr] = nn_pred_nested[k, tr]
        # val_k and test: 5-seed bagging + TTA
        val_acc = np.zeros(len(va), dtype=np.float64)
        test_acc = np.zeros(n_test, dtype=np.float64)
        for s in SEEDS:
            val_acc += train_predict(img_tr[tr], angle_tr[tr], y[tr], img_tr[va], angle_tr[va], s)
            test_acc += train_predict(img_tr[tr], angle_tr[tr], y[tr], img_te, angle_te, s)
            print(f"  fold {k} seed {s} done", flush=True)
        val_acc /= len(SEEDS); test_acc /= len(SEEDS)
        nn_pred_folds[k, va] = val_acc
        test_nn_pred[k] = test_acc
        from metrics import logloss
        per_fold_val_logloss.append(logloss(y[va], val_acc))
        print(f"fold {k}: multi-seed val nn_pred logloss={per_fold_val_logloss[-1]:.6f}", flush=True)

    # save
    out_npz = os.path.join(C.ROOT, "oof", f"nn_pred_multiseed_{run_id}.npz")
    if os.path.exists(out_npz):
        raise FileExistsError(f"Refuse to overwrite (R8): {out_npz}")
    np.savez_compressed(out_npz, nn_pred_folds=nn_pred_folds, test_nn_pred=test_nn_pred)

    meta = {
        "run_id": run_id,
        "timestamp_utc": C.now_utc().isoformat(),
        "code_version": code_version,
        "fold_idx_sha256": folds["fold_idx_sha256"],
        "labels_sha256": folds["labels_sha256"],
        "train_sha256": folds["meta"]["train_sha256"],
        "seeds": SEEDS,
        "tta": "4-flip (orig/h/v/180)",
        "train_cfg": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD, "dropout": DROPOUT,
                       "schedule": "cosine(T_max=60)"},
        "param_count_total": total,
        "nested_source": npz_nested,
        "note": "val_k/test use 5-seed bagging+TTA; tr_k reuses P4 strictly nested single-seed inner OOF (R1 prevent leakage)",
    }
    with open(out_npz + ".meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)

    log = dict(meta)
    log["per_fold_val_nn_pred_logloss"] = [round(v, 6) for v in per_fold_val_logloss]
    log["nn_pred_val_mean_logloss"] = round(float(np.mean(per_fold_val_logloss)), 6)
    out = os.path.join(C.LOGS, f"p7_6_bagging_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P7.6] nn_pred npz -> {out_npz}")
    print(f"[P7.6] log -> {out}")


if __name__ == "__main__":
    main()

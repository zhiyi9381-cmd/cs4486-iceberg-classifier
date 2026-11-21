# -*- coding: utf-8 -*-
"""
P4 strictly nested CNN OOF
Correct structure: nn_pred is (5, n), for each fold k:
  - val_k samples: predicted by outer CNN_k (trained on tr_k)
  - tr_k samples: inner CNN OOF of the 5 inner folds within tr_k
(samples have different nn_pred values across folds, cannot be overwritten by a single array)
Audit (asserted at generation time):
  A1 every CNN train_idx ∩ predict_idx = ∅ (OOF property)
  A2 outer CNN train_idx = tr_k, predict_idx = val_k, and tr_k ∩ val_k = ∅
  A3 inner CNN train_idx ⊆ tr_k, predict_idx ⊆ tr_k (no overlap with val_k)
Save nn_pred_folds (5,n) + inner fold indices (for reconstructing source_train_indices)
"""
import json
import os
import platform
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

import common as C
import folds as F
import data as D
from deepcnn import DeepCNN
import oof_cache


MODEL_CFG = {"name": "DeepCNN", "dropout": 0.4}
TRAIN_CFG = {"epochs": 60, "batch": 32, "lr": 1e-3, "optimizer": "Adam",
             "weight_decay": 1e-4, "lr_schedule": "cosine(T_max=60)", "loss": "BCEWithLogitsLoss"}
SEED = 42


def train_cnn_predict(img, angle, y, train_idx, predict_idx, seed, epochs, batch, lr):
    tr = np.asarray(train_idx, dtype=int)
    pr = np.asarray(predict_idx, dtype=int)
    med = float(np.nanmedian(angle[tr]))
    a = np.where(np.isnan(angle), med, angle).astype(np.float32)
    mu = float(a[tr].mean()); sd = float(a[tr].std(ddof=0)) or 1.0
    a_norm = ((a - mu) / sd).astype(np.float32)

    torch.manual_seed(seed); np.random.seed(seed)
    model = DeepCNN(dropout=MODEL_CFG["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TRAIN_CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    nt = len(tr)
    Xtr = torch.from_numpy(img[tr]); Atr = torch.from_numpy(a_norm[tr][:, None])
    yt = torch.from_numpy(y[tr].astype(np.float32))[:, None]
    Xpr = torch.from_numpy(img[pr]); Apr = torch.from_numpy(a_norm[pr][:, None])

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(nt)
        for i in range(0, nt, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx], Atr[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xpr, Apr)).numpy().reshape(-1)
    return prob


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img, angle, y, ids = D.load_train()
    n = len(y)
    img = img.astype(np.float32)

    run_id = C.run_id("p4nested")
    code_version = oof_cache._code_hash(
        ["common.py", "folds.py", "data.py", "metrics.py", "deepcnn.py", "p4_nested_oof.py"])

    nn_pred_folds = np.zeros((C.N_SPLITS, n), dtype=np.float32)
    inner_folds = []   # record inner fold indices (rebuild source)
    audits = {"A1_disjoint": True, "A2_outer": True, "A3_inner": True}
    cnt = 0

    for outer, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)

        # outer CNN: tr -> va
        prob_va = train_cnn_predict(img, angle, y, tr, va, SEED,
                                    TRAIN_CFG["epochs"], TRAIN_CFG["batch"], TRAIN_CFG["lr"])
        nn_pred_folds[outer, va] = prob_va
        # A1 + A2 audit
        audits["A1_disjoint"] &= bool(len(set(tr) & set(va)) == 0)
        audits["A2_outer"] &= bool(set(tr) == set(f["train_idx"]) and set(va) == set(f["val_idx"]))
        cnt += 1
        print(f"[outer {outer}] CNN(tr={len(tr)}) -> val={len(va)}", flush=True)

        # inner 5 folds
        skf = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.SEED)
        for inner, (itr_pos, iva_pos) in enumerate(skf.split(np.zeros(len(tr)), y[tr])):
            itr = tr[itr_pos]; iva = tr[iva_pos]
            prob_iva = train_cnn_predict(img, angle, y, itr, iva, SEED,
                                         TRAIN_CFG["epochs"], TRAIN_CFG["batch"], TRAIN_CFG["lr"])
            nn_pred_folds[outer, iva] = prob_iva
            # A1 + A3 audit
            audits["A1_disjoint"] &= bool(len(set(itr) & set(iva)) == 0)
            audits["A3_inner"] &= bool(set(itr).issubset(set(tr)) and set(iva).issubset(set(tr)))
            inner_folds.append({"outer": outer, "inner": inner,
                                "itr": [int(x) for x in itr], "iva": [int(x) for x in iva]})
            cnt += 1
            print(f"  [outer {outer} inner {inner}] CNN(tr={len(itr)}) -> val={len(iva)}", flush=True)

    assert audits["A1_disjoint"], "A1 violation: CNN train ∩ predict ≠ ∅"
    assert audits["A2_outer"], "A2 violation: outer CNN train/predict indices mismatch"
    assert audits["A3_inner"], "A3 violation: inner CNN training set exceeds tr_k"

    # save
    oof_dir = os.path.join(C.ROOT, "oof")
    os.makedirs(oof_dir, exist_ok=True)
    npz = os.path.join(oof_dir, f"nn_pred_nested_{run_id}.npz")
    if os.path.exists(npz):
        raise FileExistsError(f"Refuse to overwrite: {npz}")
    np.savez_compressed(npz, nn_pred_folds=nn_pred_folds)

    meta = {
        "run_id": run_id,
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "seed": SEED,
        "n_train": n,
        "folds_path": F.FOLDS_V2,
        "fold_idx_sha256": folds["fold_idx_sha256"],
        "labels_sha256": folds["labels_sha256"],
        "train_sha256": folds["meta"]["train_sha256"],
        "model_cfg": MODEL_CFG,
        "train_cfg": TRAIN_CFG,
        "code_version": code_version,
        "n_cnn_trains": cnt,
        "audit": audits,
        "inner_folds": inner_folds,
    }
    with open(npz + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log = {k: v for k, v in meta.items() if k != "inner_folds"}
    log_path = os.path.join(C.LOGS, f"p4_nested_{run_id}.json")
    C.save_json_no_overwrite(log, log_path)

    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P4-nested] nn_pred_folds -> {npz}")
    print(f"[P4-nested] log -> {log_path}")


if __name__ == "__main__":
    main()

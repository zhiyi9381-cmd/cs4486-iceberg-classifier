# -*- coding: utf-8 -*-
"""
P3 DeepCNN main model + outer 5-fold OOF generation
- single seed seed=42 training -> OOF
- dual-seed bagging seeds=[42, 7] (explicitly declared) -> average of two OOF probabilities
- OOF saved as npz + meta (R7 source provenance), auto-validated on load
- scoring: metrics.logloss (eps=1e-15)
Results written to logs/p3_deepcnn_{run_id}.json
"""
import json
import os
import platform
import sys
import numpy as np
import torch
import torch.nn as nn

import common as C
import folds as F
import data as D
import metrics as M
from deepcnn import DeepCNN, count_params
import oof_cache


MODEL_CFG = {
    "name": "DeepCNN",
    "dropout": 0.4,
    "arch": "Conv(2->16)+BN+ReLU+MP2 -> Conv(16->32)+BN+ReLU+MP2 -> Conv(32->64)+BN+ReLU+MP2 -> Conv(64->72)+BN+ReLU -> GAP -> Dropout(0.4) -> cat(angle) -> Linear(73->1)",
}
TRAIN_CFG = {"epochs": 60, "batch": 32, "lr": 1e-3, "optimizer": "Adam", "weight_decay": 1e-4, "lr_schedule": "cosine(T_max=60)", "loss": "BCEWithLogitsLoss"}
SEEDS_SINGLE = [42]
SEEDS_BAG2 = [42, 7]


def train_fold_oof(folds, img, angle, y, seed, epochs, batch, lr):
    """Per-fold independent: angle median imputation + standardization fitted only on tr (R3), reinitialize the model (seed) for training, predict val."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float32)
    per_fold = []
    loss_fn = nn.BCEWithLogitsLoss()
    for f in folds["folds"]:
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        # imputation + standardization fitted only on tr
        med = float(np.nanmedian(angle[tr]))
        a = np.where(np.isnan(angle), med, angle).astype(np.float32)
        mu = float(a[tr].mean()); sd = float(a[tr].std(ddof=0)) or 1.0
        a_norm = ((a - mu) / sd).astype(np.float32)

        torch.manual_seed(seed); np.random.seed(seed)
        model = DeepCNN(dropout=MODEL_CFG["dropout"])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TRAIN_CFG["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        nt = len(tr)
        Xtr = torch.from_numpy(img[tr]); Atr = torch.from_numpy(a_norm[tr][:, None])
        yt = torch.from_numpy(y[tr].astype(np.float32))[:, None]
        Xva = torch.from_numpy(img[va]); Ava = torch.from_numpy(a_norm[va][:, None])

        model.train()
        for ep in range(epochs):
            perm = torch.randperm(nt)
            for i in range(0, nt, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                out = model(Xtr[idx], Atr[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(Xva, Ava)).numpy().reshape(-1)
        oof[va] = prob
        ll = M.logloss(y[va], prob)
        per_fold.append(ll)
        print(f"  seed {seed} fold {f['fold']}: logloss={ll:.6f}", flush=True)
    return oof, per_fold


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img, angle, y, ids = D.load_train()
    n = len(y)
    img = img.astype(np.float32)

    per_layer, total = count_params(DeepCNN())
    run_id = C.run_id("p3cnn")
    code_version = oof_cache._code_hash(["common.py", "folds.py", "data.py", "metrics.py", "deepcnn.py", "oof_cache.py", "p3_deepcnn.py"])

    log = {
        "run_id": run_id,
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "seed": C.SEED,
        "n_train": n,
        "folds_path": F.FOLDS_V2,
        "fold_idx_sha256": folds["fold_idx_sha256"],
        "labels_sha256": folds["labels_sha256"],
        "train_sha256": folds["meta"]["train_sha256"],
        "logloss_eps": C.LOGLOSS_EPS,
        "model_cfg": MODEL_CFG,
        "train_cfg": TRAIN_CFG,
        "code_version": code_version,
        "param_count_total": total,
        "param_count_per_layer": per_layer,
        "seeds_single": SEEDS_SINGLE,
        "seeds_bag2": SEEDS_BAG2,
    }

    # single seed
    print("[P3] single seed seed=42", flush=True)
    oof_single, pf_single = train_fold_oof(folds, img, angle, y, 42,
                                            TRAIN_CFG["epochs"], TRAIN_CFG["batch"], TRAIN_CFG["lr"])

    # dual-seed bagging
    print("[P3] dual seed seed=7", flush=True)
    oof_seed7, pf_seed7 = train_fold_oof(folds, img, angle, y, 7,
                                           TRAIN_CFG["epochs"], TRAIN_CFG["batch"], TRAIN_CFG["lr"])
    oof_bag2 = 0.5 * (oof_single + oof_seed7)

    vfa = np.full(n, -1, dtype=np.int8)
    for f in folds["folds"]:
        vfa[np.asarray(f["val_idx"], dtype=int)] = f["fold"]

    meta_base = {
        "fold_idx_sha256": folds["fold_idx_sha256"],
        "labels_sha256": folds["labels_sha256"],
        "train_sha256": folds["meta"]["train_sha256"],
        "model_cfg": MODEL_CFG,
        "train_cfg": TRAIN_CFG,
        "code_version": code_version,
        "run_id": run_id,
        "timestamp_utc": C.now_utc().isoformat(),
    }
    oof_dir = os.path.join(C.ROOT, "oof")
    os.makedirs(oof_dir, exist_ok=True)

    paths = {}
    for name, arr, seeds in [("single_seed42", oof_single, SEEDS_SINGLE),
                             ("seed7", oof_seed7, [7]),
                             ("bag2_seed42_7", oof_bag2, SEEDS_BAG2)]:
        m = dict(meta_base); m["seeds"] = seeds
        npz = os.path.join(oof_dir, f"oof_{name}_{run_id}.npz")
        oof_cache.save_oof(npz, arr, vfa, m)
        paths[name] = npz

    # validation (auto-validated R7 via load_oof)
    for name in paths:
        oof_cache.load_oof(paths[name], current_folds=folds, expected_model_cfg=MODEL_CFG)

    ll_single = M.logloss(y, oof_single)
    ll_bag2 = M.logloss(y, oof_bag2)
    log["oof_logloss_single_seed42"] = round(ll_single, 6)
    log["oof_logloss_bag2_seed42_7"] = round(ll_bag2, 6)
    log["per_fold_logloss_single_seed42"] = [round(v, 6) for v in pf_single]
    log["per_fold_logloss_seed7"] = [round(v, 6) for v in pf_seed7]
    log["oof_paths"] = paths

    out = os.path.join(C.LOGS, f"p3_deepcnn_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P3] -> {out}")


if __name__ == "__main__":
    main()

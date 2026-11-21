# -*- coding: utf-8 -*-
"""
P6 single-variable controlled experiments E1/E2
- E1 learning-rate schedule comparison: fixed lr (no schedule) vs cosine (everything else matched, Dropout=0.3)
- E2 Dropout comparison: 0 vs 0.3 (everything else matched, cosine schedule)
- configurations matched item by item (R6), single seed=42
- each experiment runs 2 independent runs (R9, two complete logs)
- each experiment: DeepCNN five-fold CV (outer folds_v2), LogLoss mean±std(ddof=0)
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
import oof_cache
from deepcnn import DeepCNN

EPOCHS = 60
BATCH = 32
LR = 1e-3
WD = 1e-4
SEED = 42

# 3 experiments (E1B and E2B share the same config: cosine + dropout0.3)
EXPERIMENTS = {
    "E1A_fixed_lr_dropout0.3": {"use_cosine": False, "dropout": 0.3},
    "E1B_cosine_dropout0.3": {"use_cosine": True, "dropout": 0.3},
    "E2A_cosine_dropout0": {"use_cosine": True, "dropout": 0.0},
}


def train_predict(img, angle, y, train_idx, predict_idx, dropout, use_cosine):
    tr = np.asarray(train_idx, dtype=int)
    pr = np.asarray(predict_idx, dtype=int)
    med = float(np.nanmedian(angle[tr]))
    a = np.where(np.isnan(angle), med, angle).astype(np.float32)
    mu = float(a[tr].mean()); sd = float(a[tr].std(ddof=0)) or 1.0
    a_norm = ((a - mu) / sd).astype(np.float32)

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = DeepCNN(dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if use_cosine else None
    loss_fn = nn.BCEWithLogitsLoss()

    nt = len(tr)
    Xtr = torch.from_numpy(img[tr]); Atr = torch.from_numpy(a_norm[tr][:, None])
    yt = torch.from_numpy(y[tr].astype(np.float32))[:, None]
    Xpr = torch.from_numpy(img[pr]); Apr = torch.from_numpy(a_norm[pr][:, None])

    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(nt)
        for i in range(0, nt, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx], Atr[idx]), yt[idx])
            loss.backward()
            opt.step()
        if sched is not None:
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
    code_version = oof_cache._code_hash(["common.py", "folds.py", "data.py", "metrics.py",
                                         "deepcnn.py", "p6_ablation.py"])

    for name, cfg in EXPERIMENTS.items():
        for run_no in [1, 2]:
            lls = []
            for k, f in enumerate(folds["folds"]):
                tr = np.asarray(f["train_idx"], dtype=int)
                va = np.asarray(f["val_idx"], dtype=int)
                prob = train_predict(img, angle, y, tr, va, cfg["dropout"], cfg["use_cosine"])
                lls.append(M.logloss(y[va], prob))
                print(f"{name} run{run_no} fold {k}: {lls[-1]:.6f}", flush=True)
            m, s = M.mean_std(lls)
            log = {
                "run_id": C.run_id(f"p6_{name}_r{run_no}"),
                "timestamp_utc": C.now_utc().isoformat(),
                "os": platform.platform(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": torch.__version__,
                "code_version": code_version,
                "experiment": name,
                "run_no": run_no,
                "config": {
                    "use_cosine": cfg["use_cosine"],
                    "dropout": cfg["dropout"],
                    "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                    "weight_decay": WD, "seed": SEED, "model": "DeepCNN(65426)",
                },
                "folds_logloss": [round(v, 6) for v in lls],
                "mean": round(m, 6),
                "std_ddof0": round(s, 6),
            }
            out = os.path.join(C.LOGS, f"p6_{name}_run{run_no}_{log['run_id']}.json")
            C.save_json_no_overwrite(log, out)
            print(f"  {name} run{run_no}: mean={m:.6f} ± {s:.6f}", flush=True)

    print("[P6] all experiments done", flush=True)


if __name__ == "__main__":
    main()

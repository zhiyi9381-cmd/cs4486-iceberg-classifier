# -*- coding: utf-8 -*-
"""
P2 baseline ShallowCNN (~7-8k parameters, with inc_angle bypass) 5-fold CV LogLoss
- input: 2x75x75 dual-band image + inc_angle (scalar bypass)
- architecture: 3 groups of Conv+BN+ReLU+MaxPool -> GlobalAvgPool -> concat angle -> FC -> logit
- per-layer parameter counts are computed and written to the log
- inc_angle median imputation + standardization fitted only on the current fold's training subset (avoiding preprocessing leakage)
- seed=42 (torch.manual_seed + numpy)
- scoring: metrics.logloss (eps=1e-15)
CPU training, batch=32, epochs=60, Adam lr=1e-3 (no cosine, fixed lr)
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


class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),   nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),  nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 40, 3, padding=1), nn.BatchNorm2d(40), nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(40 + 1, 1)   # 40-dim GAP output + 1-dim angle bypass

    def forward(self, x_img, x_angle):
        h = self.features(x_img)                 # (B, 40, 18, 18)
        h = h.mean(dim=(2, 3))                   # GlobalAvgPool -> (B, 40)
        h = self.dropout(h)
        h = torch.cat([h, x_angle], dim=1)       # (B, 41)
        return self.head(h)                       # (B, 1) logit


def count_params(model):
    per_layer = [(name, p.numel()) for name, p in model.named_parameters()]
    total = sum(n for _, n in per_layer)
    return per_layer, total


def train_one_fold(model, Xtr_img, Xtr_ang, ytr, Xva_img, Xva_ang, yva, epochs=60, batch=32, lr=1e-3, seed=C.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(ytr)
    Xtr_img = torch.from_numpy(Xtr_img)
    Xtr_ang = torch.from_numpy(Xtr_ang[:, None].astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))[:, None]

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xtr_img[idx], Xtr_ang[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Xva_img = torch.from_numpy(Xva_img)
        Xva_ang = torch.from_numpy(Xva_ang[:, None].astype(np.float32))
        logit = model(Xva_img, Xva_ang)
        prob = torch.sigmoid(logit).numpy().reshape(-1)
    return prob


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img, angle, y, ids = D.load_train()
    n = len(y)
    img = img.astype(np.float32)

    run_id = C.run_id("p2cnn")
    per_layer, total = count_params(ShallowCNN())

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
        "logloss_eps": C.LOGLOSS_EPS,
        "model": "ShallowCNN (2x75x75 + inc_angle bypass)",
        "arch": "Conv(2->8,3x3,pad1)+BN+ReLU+MP2 -> Conv(8->16)+BN+ReLU+MP2 -> Conv(16->40)+BN+ReLU -> GAP -> cat(angle) -> Linear(41->1)",
        "train_cfg": {"epochs": 60, "batch": 32, "lr": 1e-3, "optimizer": "Adam", "weight_decay": 1e-4, "dropout": 0.3, "loss": "BCEWithLogitsLoss"},
        "param_count_total": total,
        "param_count_per_layer": per_layer,
        "folds_logloss": [],
    }

    lls = []
    for f in folds["folds"]:
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        # angle imputation + standardization fitted only on tr
        med = float(np.nanmedian(angle[tr]))
        a = np.where(np.isnan(angle), med, angle).astype(np.float32)
        mu = float(a[tr].mean()); sd = float(a[tr].std(ddof=0))
        if sd == 0:
            sd = 1.0
        a_norm = ((a - mu) / sd).astype(np.float32)

        model = ShallowCNN()
        prob = train_one_fold(model, img[tr], a_norm[tr], y[tr],
                              img[va], a_norm[va], y[va],
                              epochs=60, batch=32, lr=1e-3, seed=C.SEED)
        ll = M.logloss(y[va], prob)
        lls.append(ll)
        print(f"fold {f['fold']}: logloss={ll:.6f}", flush=True)

    m, s = M.mean_std(lls)
    log["folds_logloss"] = [round(v, 6) for v in lls]
    log["mean"] = round(m, 6)
    log["std_ddof0"] = round(s, 6)

    out = os.path.join(C.LOGS, f"p2_shallowcnn_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P2-cnn] -> {out}")


if __name__ == "__main__":
    main()

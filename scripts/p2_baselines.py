# -*- coding: utf-8 -*-
"""
P2 baselines (Dummy / LogisticRegression) 5-fold CV LogLoss
- read folds_v2.json (load_folds mandatory validation)
- Dummy: per-fold training subset positive rate as a constant probability (no features, no leakage)
- LR: flattened image (2x75x75 -> 11250) + inc_angle
- inc_angle median imputation + StandardScaler fitted only on the current fold's training subset, then transform train/val
- scoring: metrics.logloss (eps=1e-15)
Results written to logs/p2_baselines_{run_id}.json (versioned naming, does not overwrite old files)
"""
import json
import os
import platform
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common as C
import folds as F
import data as D
import metrics as M


def main():
    folds = F.load_folds(F.FOLDS_V2)          # mandatory validation
    img, angle, y, ids = D.load_train()
    n = len(y)
    X_flat = img.reshape(n, -1).astype(np.float32)   # (n, 11250)

    run_id = C.run_id("p2base")
    log = {
        "run_id": run_id,
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn_version": C.sklearn_version(),
        "seed": C.SEED,
        "n_train": n,
        "folds_path": F.FOLDS_V2,
        "folds_meta_run_id": folds["meta"]["run_id"],
        "logloss_eps": C.LOGLOSS_EPS,
        "models": {},
    }

    # ---------- Dummy: per-fold training subset positive rate ----------
    dummy_lls = []
    for f in folds["folds"]:
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        p = float((y[tr] == 1).mean())
        pred = np.full(len(va), p)
        dummy_lls.append(M.logloss(y[va], pred))
    dm, ds = M.mean_std(dummy_lls)
    log["models"]["dummy"] = {
        "type": "constant_probability (per-fold train prior)",
        "folds_logloss": [round(v, 6) for v in dummy_lls],
        "mean": round(dm, 6),
        "std_ddof0": round(ds, 6),
    }

    # ---------- LogisticRegression ----------
    lr_cfg = {"C": 0.001, "solver": "liblinear", "max_iter": 1000, "random_state": C.SEED}
    lr_lls = []
    for f in folds["folds"]:
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        # median imputation + scaler fitted only on tr
        med = float(np.nanmedian(angle[tr]))
        def build(idx):
            X = np.hstack([X_flat[idx], angle[idx][:, None].astype(np.float32)])
            X[:, -1] = np.nan_to_num(X[:, -1], nan=med)
            return X
        X_tr = build(tr); X_va = build(va)
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr); X_va = scaler.transform(X_va)
        model = LogisticRegression(**lr_cfg).fit(X_tr, y[tr])
        pred = model.predict_proba(X_va)[:, 1]
        lr_lls.append(M.logloss(y[va], pred))
    lm, ls = M.mean_std(lr_lls)
    log["models"]["logistic_regression"] = {
        "type": "LogisticRegression (flatten 2x75x75 + inc_angle, StandardScaler)",
        "config": lr_cfg,
        "folds_logloss": [round(v, 6) for v in lr_lls],
        "mean": round(lm, 6),
        "std_ddof0": round(ls, 6),
    }

    out = os.path.join(C.LOGS, f"p2_baselines_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P2-base] -> {out}")


if __name__ == "__main__":
    main()

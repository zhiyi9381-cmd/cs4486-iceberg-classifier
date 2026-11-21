# -*- coding: utf-8 -*-
"""
P4 meta feature construction + LightGBM per-group ablation
nn_pred source: p4_nested's nn_pred_folds (5, n), each fold k uses nn_pred_folds[k]
(tr_k part = inner OOF, val_k part = outer CNN_k prediction)
"""
import json
import os
import glob
import platform
import sys
import numpy as np
import lightgbm as lgb

import common as C
import folds as F
import data as D
import metrics as M
import oof_cache


LGBM_CFG = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 8,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": C.SEED,
    "verbosity": -1,
}
N_BINS = 30
K = 15


def build_features(nn_pred, angle, y, tr, va):
    """Build meta features for one outer fold. nn_pred is the nn_pred feature for that fold (length n).
    Angle filling/normalization/bin boundaries are fit on tr only; bin/KNN features use leave-one-out to exclude self."""
    med = float(np.nanmedian(angle[tr]))
    a_fill = np.where(np.isnan(angle), med, angle).astype(np.float64)
    mu = float(a_fill[tr].mean()); sd = float(a_fill[tr].std(ddof=0)) or 1.0
    angle_norm = ((a_fill - mu) / sd).astype(np.float32)

    qs = np.quantile(a_fill[tr], np.linspace(0, 1, N_BINS + 1))
    qs[0] = -np.inf; qs[-1] = np.inf
    bins = np.clip(np.digitize(a_fill, qs) - 1, 0, N_BINS - 1)

    def group_stats(idx_all, exclude):
        n_i = len(idx_all)
        b_med = np.zeros(n_i, dtype=np.float32)
        b_mean = np.zeros(n_i, dtype=np.float32)
        b_cnt = np.zeros(n_i, dtype=np.float32)
        for pos, i in enumerate(idx_all):
            mask = (bins[tr] == bins[i])
            if exclude:
                mask &= (tr != i)
            grp = nn_pred[tr[mask]]
            if len(grp) == 0:
                b_med[pos] = 0.5; b_mean[pos] = 0.5
            else:
                b_med[pos] = np.median(grp); b_mean[pos] = grp.mean()
            b_cnt[pos] = np.log1p(len(grp))
        return b_med, b_mean, b_cnt

    def knn_stats(idx_all, exclude):
        n_i = len(idx_all)
        kpr = np.zeros(n_i, dtype=np.float32)
        for pos, i in enumerate(idx_all):
            dist = np.abs(a_fill[tr] - a_fill[i]).copy()
            if exclude:
                dist[np.where(tr == i)[0]] = np.inf
            kk = min(K, len(tr))
            nn_idx = np.argpartition(dist, kk - 1)[:kk]
            kpr[pos] = nn_pred[tr[nn_idx]].mean()
        return kpr

    def assemble(idx_all, exclude):
        return np.column_stack([
            nn_pred[idx_all], angle_norm[idx_all],
            *group_stats(idx_all, exclude), knn_stats(idx_all, exclude),
        ])

    X_tr = assemble(tr, exclude=True)
    X_va = assemble(va, exclude=False)
    feat_names = ["nn_pred", "inc_angle", "bin_median", "bin_mean", "bin_logcount", "knn_prior"]
    return X_tr, X_va, feat_names


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img, angle, y, ids = D.load_train()
    n = len(y)

    nested_files = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))
    if not nested_files:
        raise FileNotFoundError("Missing nn_pred_nested npz, run p4_nested_oof.py first")
    npz_path = nested_files[-1]
    z = np.load(npz_path)
    nn_pred_folds = z["nn_pred_folds"].astype(np.float64)  # (5, n)
    meta = json.load(open(npz_path + ".meta.json", encoding="utf-8"))
    assert meta["fold_idx_sha256"] == folds["fold_idx_sha256"], "nn_pred fold base mismatch"
    assert meta["labels_sha256"] == folds["labels_sha256"], "nn_pred labels mismatch"
    assert all(meta["audit"].values()), f"nn_pred audit failed: {meta['audit']}"

    run_id = C.run_id("p4meta")
    log = {
        "run_id": run_id,
        "code_version": oof_cache._code_hash(["common.py", "folds.py", "data.py", "metrics.py", "p4_meta.py"]),
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "lightgbm": lgb.__version__,
        "nn_pred_source": npz_path,
        "n_bins": N_BINS,
        "knn_k": K,
        "lgbm_cfg": LGBM_CFG,
        "seed": C.SEED,
        "ablation": {},
    }

    feature_groups = {
        "CNN_only": ["nn_pred"],
        "plus_angle": ["nn_pred", "inc_angle"],
        "plus_bin": ["nn_pred", "inc_angle", "bin_median", "bin_mean", "bin_logcount"],
        "plus_KNN": ["nn_pred", "inc_angle", "bin_median", "bin_mean", "bin_logcount", "knn_prior"],
    }
    col_idx = {"nn_pred": 0, "inc_angle": 1, "bin_median": 2, "bin_mean": 3,
               "bin_logcount": 4, "knn_prior": 5}

    for name, cols in feature_groups.items():
        lls = []
        for k, f in enumerate(folds["folds"]):
            tr = np.asarray(f["train_idx"], dtype=int)
            va = np.asarray(f["val_idx"], dtype=int)
            nn_pred = nn_pred_folds[k]  # strictly nested nn_pred for this fold
            X_tr, X_va, feat_names = build_features(nn_pred, angle, y, tr, va)
            sel = [col_idx[c] for c in cols]
            m = lgb.LGBMClassifier(**LGBM_CFG)
            m.fit(X_tr[:, sel], y[tr])
            prob = m.predict_proba(X_va[:, sel])[:, 1]
            lls.append(M.logloss(y[va], prob))
        mm, ss = M.mean_std(lls)
        log["ablation"][name] = {
            "features": cols,
            "folds_logloss": [round(v, 6) for v in lls],
            "mean": round(mm, 6),
            "std_ddof0": round(ss, 6),
        }
        print(f"{name}: mean={mm:.6f} ± {ss:.6f}  folds={[round(v,4) for v in lls]}", flush=True)

    out = os.path.join(C.LOGS, f"p4_meta_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P4-meta] -> {out}")


if __name__ == "__main__":
    main()

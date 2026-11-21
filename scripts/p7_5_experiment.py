# -*- coding: utf-8 -*-
"""
P7.5 iteration experiments: leaky inc_angle post-processing + exact angle grouping (strict CV, tr info only)
Evaluate each variant's LogLoss under the unified five-fold CV, compared against baseline (existing plus_KNN)
"""
import json
import os
import glob
from collections import defaultdict
import numpy as np
import lightgbm as lgb

import common as C
import folds as F
import data as D
import metrics as M
from p4_meta import build_features, LGBM_CFG


def leaky_override(angle, y, tr, va, pred_va, clip_hi, clip_lo):
    """For val samples, if angle is in a tr pure group (single label), override the prediction with the clip probability (tr info only)."""
    groups = defaultdict(list)
    for i in tr:
        if not np.isnan(angle[i]):
            groups[round(float(angle[i]), 4)].append(int(y[i]))
    pred = np.asarray(pred_va, dtype=np.float64).copy()
    n_ovr = 0
    for pos, i in enumerate(va):
        a = round(float(angle[i]), 4)
        if a in groups and len(set(groups[a])) == 1:
            pred[pos] = clip_hi if groups[a][0] == 1 else clip_lo
            n_ovr += 1
    return pred, n_ovr


def build_features_exact(nn_pred, angle, y, tr, va):
    """Exact angle grouping (4 decimal places) replaces the 30 equal-frequency bins' bin statistics; the rest is the same as build_features."""
    med = float(np.nanmedian(angle[tr]))
    a_fill = np.where(np.isnan(angle), med, angle).astype(np.float64)
    mu = float(a_fill[tr].mean()); sd = float(a_fill[tr].std(ddof=0)) or 1.0
    angle_norm = ((a_fill - mu) / sd).astype(np.float32)

    # exact grouping key
    key_tr = np.round(a_fill[tr], 4)

    def group_stats(idx_all, exclude):
        n_i = len(idx_all)
        b_med = np.zeros(n_i, dtype=np.float32); b_mean = np.zeros(n_i, dtype=np.float32)
        b_cnt = np.zeros(n_i, dtype=np.float32)
        for pos, i in enumerate(idx_all):
            mask = key_tr == round(float(a_fill[i]), 4)
            if exclude:
                mask &= (tr != i)
            grp = nn_pred[tr[mask]]
            b_med[pos] = np.median(grp) if len(grp) else 0.5
            b_mean[pos] = grp.mean() if len(grp) else 0.5
            b_cnt[pos] = np.log1p(len(grp))
        return b_med, b_mean, b_cnt

    def knn_stats(idx_all, exclude):
        n_i = len(idx_all)
        kpr = np.zeros(n_i, dtype=np.float32)
        for pos, i in enumerate(idx_all):
            dist = np.abs(a_fill[tr] - a_fill[i]).copy()
            if exclude:
                dist[np.where(tr == i)[0]] = np.inf
            nn_idx = np.argpartition(dist, 14)[:15]
            kpr[pos] = nn_pred[tr[nn_idx]].mean()
        return kpr

    def assemble(idx_all, exclude):
        return np.column_stack([nn_pred[idx_all], angle_norm[idx_all],
                                *group_stats(idx_all, exclude), knn_stats(idx_all, exclude)])
    return assemble(tr, exclude=True), assemble(va, exclude=False), None


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img, angle, y, ids = D.load_train()
    n = len(y)
    npz = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    nn_pred_folds = np.load(npz)["nn_pred_folds"].astype(np.float64)

    results = {}

    def run_cv(build_fn, use_leaky, clip_hi, clip_lo, label):
        lls = []; ovr_total = 0
        for k, f in enumerate(folds["folds"]):
            tr = np.asarray(f["train_idx"], dtype=int)
            va = np.asarray(f["val_idx"], dtype=int)
            X_tr, X_va, _ = build_fn(nn_pred_folds[k], angle, y, tr, va)
            m = lgb.LGBMClassifier(**LGBM_CFG)
            m.fit(X_tr, y[tr])
            pred = m.predict_proba(X_va)[:, 1]
            ovr = 0
            if use_leaky:
                pred, ovr = leaky_override(angle, y, tr, va, pred, clip_hi, clip_lo)
                ovr_total += ovr
            lls.append(M.logloss(y[va], pred))
        mm, ss = M.mean_std(lls)
        results[label] = {"mean": round(mm, 6), "std": round(ss, 6),
                          "folds": [round(v, 6) for v in lls],
                          "override_count": int(ovr_total)}
        print(f"{label}: mean={mm:.6f} ± {ss:.6f} override={ovr_total}", flush=True)

    run_cv(build_features, False, 0.999, 0.001, "baseline_plusKNN")
    run_cv(build_features, True, 0.999, 0.001, "baseline_leaky_0.999")
    run_cv(build_features, True, 0.995, 0.005, "baseline_leaky_0.995")
    run_cv(build_features, True, 0.99, 0.01, "baseline_leaky_0.99")
    run_cv(build_features_exact, False, 0.999, 0.001, "exact_group_no_leaky")
    run_cv(build_features_exact, True, 0.995, 0.005, "exact_group_leaky_0.995")
    run_cv(build_features_exact, True, 0.99, 0.01, "exact_group_leaky_0.99")

    log = {"run_id": C.run_id("p75exp"), "results": results}
    out = os.path.join(C.LOGS, f"p7_5_experiment_{log['run_id']}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P7.5-exp] -> {out}")


if __name__ == "__main__":
    main()

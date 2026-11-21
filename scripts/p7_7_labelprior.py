# -*- coding: utf-8 -*-
"""
P7.7 quick experiment: validate the gain of angle-neighborhood label prior (label_prior)
- data exploration found: after sorting by angle, adjacent samples share the same label 86.94%, i.e. labels change continuously with angle
- existing knn_prior uses KNN over nn_pred (model prediction), weak; label_prior directly uses KNN over true labels, strong
- comparison: baseline (existing meta) vs +label_prior vs label_prior replacing knn_prior
- strict LOO: tr samples exclude their own label, val uses all tr (prevent leakage)
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
from p4_meta import LGBM_CFG
from p7_5_submission_v2 import build_tr_va_exact


def build_label_prior(angle, y, tr, idx_all, k):
    """True-label ratio of the K nearest neighbors in the angle neighborhood (LOO: tr samples exclude self)."""
    a_fill = np.where(np.isnan(angle), np.nanmedian(angle[tr]), angle).astype(np.float64)
    a_tr = a_fill[tr]
    out = np.zeros(len(idx_all), dtype=np.float64)
    for pos, i in enumerate(idx_all):
        d = np.abs(a_tr - a_fill[i]).copy()
        if i in set(tr):
            d[np.where(tr == i)[0]] = np.inf   # LOO exclude self
        kk = min(k, len(tr))
        nn_idx = np.argpartition(d, kk - 1)[:kk]
        out[pos] = y[tr[nn_idx]].mean()
    return out


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img, angle, y, ids = D.load_train()
    npz = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    nn_pred_folds = np.load(npz)["nn_pred_folds"].astype(np.float64)

    # OOF of existing meta features (6 cols), then stack label_prior on top
    def cv_with_extra(extra_name, k_val=None):
        lls = []
        for kf, f in enumerate(folds["folds"]):
            tr = np.asarray(f["train_idx"], dtype=int)
            va = np.asarray(f["val_idx"], dtype=int)
            X_tr, X_va = build_tr_va_exact(nn_pred_folds[kf], angle, y, tr, va)
            if extra_name == "label_prior":
                lp_tr = build_label_prior(angle, y, tr, tr, k_val)
                lp_va = build_label_prior(angle, y, tr, va, k_val)
                X_tr = np.column_stack([X_tr, lp_tr])
                X_va = np.column_stack([X_va, lp_va])
            m = lgb.LGBMClassifier(**LGBM_CFG).fit(X_tr, y[tr])
            lls.append(M.logloss(y[va], m.predict_proba(X_va)[:, 1]))
        mm, ss = M.mean_std(lls)
        print(f"{extra_name}: mean={mm:.6f} ± {ss:.6f}", flush=True)
        return mm

    print("=== baseline vs label_prior feature ===", flush=True)
    base = cv_with_extra("baseline")
    for k in (5, 15, 30):
        cv_with_extra("label_prior", k)

    print("\n=== label_prior as post-processing soft prior (all samples) ===", flush=True)
    # for each val sample of OOF, soft-fuse with label_prior
    for w in (0.3, 0.5, 0.7):
        lls = []
        for kf, f in enumerate(folds["folds"]):
            tr = np.asarray(f["train_idx"], dtype=int)
            va = np.asarray(f["val_idx"], dtype=int)
            X_tr, X_va = build_tr_va_exact(nn_pred_folds[kf], angle, y, tr, va)
            m = lgb.LGBMClassifier(**LGBM_CFG).fit(X_tr, y[tr])
            p = m.predict_proba(X_va)[:, 1]
            lp_va = build_label_prior(angle, y, tr, va, 15)
            p = w * p + (1 - w) * lp_va
            lls.append(M.logloss(y[va], p))
        mm, ss = M.mean_std(lls)
        print(f"soft_blend_labelprior_w{w}: mean={mm:.6f} ± {ss:.6f}", flush=True)


if __name__ == "__main__":
    main()

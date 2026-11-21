# -*- coding: utf-8 -*-
"""
P7.11 handcrafted morphological/statistical feature validation
ship = compact bright spot, iceberg = diffuse texture. Extract classic image features for meta, to see if they contribute new information (orthogonal to CNN)
"""
import os
import numpy as np
import glob
from collections import defaultdict
import lightgbm as lgb
from scipy import ndimage

import common as C
import folds as F
import data as D
import metrics as M
from p4_meta import LGBM_CFG
from p7_5_submission_v2 import build_tr_va_exact, build_te_exact
from p7_7_labelprior import build_label_prior


def extract_features(img):
    """img: (n, 2, 75, 75) -> handcrafted features (n, F)."""
    b1 = img[:, 0, :, :]
    b2 = img[:, 1, :, :]
    n = len(img)
    feats = []
    # statistical features (vectorized)
    feats.append(b1.max(axis=(1, 2)))                       # band1 max
    feats.append(b1.mean(axis=(1, 2)))                      # band1 mean
    feats.append(b1.std(axis=(1, 2)))                       # band1 std
    feats.append(b1.max(axis=(1, 2)) / (b1.mean(axis=(1, 2)) + 1e-6))  # peak ratio
    feats.append(b2.max(axis=(1, 2)))
    feats.append(b2.mean(axis=(1, 2)))
    feats.append(b2.std(axis=(1, 2)))
    # dual-band relationship
    ratio = b1 / (b2 + 1e-6)
    feats.append(ratio.mean(axis=(1, 2)))                   # band1/band2 mean
    feats.append(ratio.std(axis=(1, 2)))
    diff = b1 - b2
    feats.append(diff.mean(axis=(1, 2)))
    feats.append(diff.std(axis=(1, 2)))
    # top-1% pixel mean (bright-spot intensity)
    b1_flat = b1.reshape(n, -1)
    k = max(1, b1_flat.shape[1] // 100)
    top1 = np.sort(b1_flat, axis=1)[:, -k:]
    feats.append(top1.mean(axis=1))
    feats.append(top1.mean(axis=1) / (b1_flat.mean(axis=1) + 1e-6))
    # morphology: threshold segmentation + largest connected-component area + component count (per sample)
    area_max = np.zeros(n); n_comp = np.zeros(n)
    for i in range(n):
        thresh = b1[i] > (b1[i].mean() + 2 * b1[i].std())
        lab, num = ndimage.label(thresh)
        if num > 0:
            sizes = ndimage.sum(thresh, lab, range(1, num + 1))
            area_max[i] = sizes.max()
            n_comp[i] = num
    feats.append(area_max)
    feats.append(n_comp)
    return np.column_stack([f.astype(np.float32) for f in feats])


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img, angle, y, ids = D.load_train()
    img = img.astype(np.float32)

    # handcrafted features (full train)
    H = extract_features(img)
    print(f"handcrafted feature dim: {H.shape[1]}", flush=True)

    npz = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_multiseed_*.npz")))[-1]
    z = np.load(npz)
    nn = z["nn_pred_folds"].astype(np.float64)
    nnn = np.load(sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1])["nn_pred_folds"].astype(np.float64)
    nn[np.isnan(nn)] = nnn[np.isnan(nn)]

    def cv(with_handcrafted):
        oof = np.zeros(len(y))
        for kf, f in enumerate(folds["folds"]):
            tr = np.asarray(f["train_idx"], int); va = np.asarray(f["val_idx"], int)
            X_tr, X_va = build_tr_va_exact(nn[kf], angle, y, tr, va)
            lp_tr = build_label_prior(angle, y, tr, tr, 2); lp_va = build_label_prior(angle, y, tr, va, 2)
            X_tr = np.column_stack([X_tr, lp_tr]); X_va = np.column_stack([X_va, lp_va])
            if with_handcrafted:
                X_tr = np.column_stack([X_tr, H[tr]]); X_va = np.column_stack([X_va, H[va]])
            m = lgb.LGBMClassifier(**LGBM_CFG).fit(X_tr, y[tr])
            oof[va] = m.predict_proba(X_va)[:, 1]
        # soft leaky
        for f in folds["folds"]:
            tr = np.asarray(f["train_idx"], int); va = np.asarray(f["val_idx"], int)
            groups = defaultdict(list)
            for i in tr:
                if not np.isnan(angle[i]): groups[round(float(angle[i]), 4)].append(int(y[i]))
            for pos, i in enumerate(va):
                a = round(float(angle[i]), 4)
                if a in groups and len(set(groups[a])) == 1:
                    prior = 0.99 if groups[a][0] == 1 else 0.01
                    oof[i] = 0.5 * oof[i] + 0.5 * prior
        lls = [M.logloss(y[np.asarray(f["val_idx"], int)], oof[np.asarray(f["val_idx"], int)]) for f in folds["folds"]]
        return M.mean_std(lls)

    mm, ss = cv(False)
    print(f"baseline (label_prior k=2 + soft_leaky): {mm:.6f} ± {ss:.6f}", flush=True)
    mm2, ss2 = cv(True)
    print(f"+ handcrafted features: {mm2:.6f} ± {ss2:.6f}", flush=True)
    print(f"improvement: {mm - mm2:.6f}", flush=True)


if __name__ == "__main__":
    main()

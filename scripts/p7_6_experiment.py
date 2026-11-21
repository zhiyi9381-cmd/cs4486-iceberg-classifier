# -*- coding: utf-8 -*-
"""
P7.6 quick experiment: use existing single-seed nn_pred_nested + exact grouping features to validate the marginal gain of post-processing/ensembling improvements
- comparison: multi-LGBM (3 variants logit-averaged) / sharpening (1.15) / soft_leaky (soft fusion) / hard_leaky (v2 reference)
- purpose: while multi-seed bagging is training, confirm early whether the sharpening+soft_leaky direction is correct
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

LGBM_SEEDS = [42, 7, 2024]
PRIOR_HI = 0.99
PRIOR_LO = 0.01


def sharpen(p, alpha):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-alpha * np.log(p / (1 - p))))


def leaky_soft(angle, y, tr, va, pred, w):
    groups = defaultdict(list)
    for i in tr:
        if not np.isnan(angle[i]):
            groups[round(float(angle[i]), 4)].append(int(y[i]))
    pred = np.asarray(pred, dtype=np.float64).copy()
    n = 0
    for pos, i in enumerate(va):
        a = round(float(angle[i]), 4)
        if a in groups and len(set(groups[a])) == 1:
            prior = PRIOR_HI if groups[a][0] == 1 else PRIOR_LO
            pred[pos] = w * pred[pos] + (1 - w) * prior
            n += 1
    return pred, n


def fit_multi_lgbm(X_tr, y_tr, X_va):
    logits = []
    for s in LGBM_SEEDS:
        m = lgb.LGBMClassifier(**dict(LGBM_CFG, random_state=s)).fit(X_tr, y_tr)
        p = np.clip(m.predict_proba(X_va)[:, 1], 1e-7, 1 - 1e-7)
        logits.append(np.log(p / (1 - p)))
    return 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))


def main():
    folds = F.load_folds(F.FOLDS_V2)
    img, angle, y, ids = D.load_train()
    npz = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    nn_pred_folds = np.load(npz)["nn_pred_folds"].astype(np.float64)

    # first compute single-LGBM OOF (baseline), to compare with multi-LGBM
    oof_single = np.zeros(len(y), dtype=np.float64)
    oof_multi = np.zeros(len(y), dtype=np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        X_tr, X_va = build_tr_va_exact(nn_pred_folds[k], angle, y, tr, va)
        m = lgb.LGBMClassifier(**LGBM_CFG).fit(X_tr, y[tr])
        oof_single[va] = m.predict_proba(X_va)[:, 1]
        oof_multi[va] = fit_multi_lgbm(X_tr, y[tr], X_va)

    def cv(p):
        lls = [M.logloss(y[np.asarray(f["val_idx"], dtype=int)], p[np.asarray(f["val_idx"], dtype=int)]) for f in folds["folds"]]
        mm, ss = M.mean_std(lls)
        return mm, ss, [round(v, 6) for v in lls]

    results = {}

    def add(label, p):
        mm, ss, fl = cv(p)
        results[label] = {"mean": round(mm, 6), "std": round(ss, 6), "folds": fl}
        print(f"{label}: mean={mm:.6f} ± {ss:.6f}", flush=True)

    add("single_lgbm", oof_single)
    add("multi_lgbm_3", oof_multi)
    add("single_sharp_1.15", sharpen(oof_single, 1.15))
    add("multi_sharp_1.15", sharpen(oof_multi, 1.15))

    # soft leaky at various weights (based on multi)
    for w in (0.5, 0.7, 0.85):
        p = np.zeros(len(y), dtype=np.float64)
        for f in folds["folds"]:
            tr = np.asarray(f["train_idx"], dtype=int); va = np.asarray(f["val_idx"], dtype=int)
            p[va], _ = leaky_soft(angle, y, tr, va, oof_multi[va], w)
        add(f"multi_softleaky_w{w}", p)

    # hard leaky (v2 reference)
    p = np.zeros(len(y), dtype=np.float64)
    for f in folds["folds"]:
        tr = np.asarray(f["train_idx"], dtype=int); va = np.asarray(f["val_idx"], dtype=int)
        p[va], _ = leaky_soft(angle, y, tr, va, oof_multi[va], 0.0)
    add("multi_hardleaky_w0", p)

    # sharp + soft leaky combination
    p_sharp = sharpen(oof_multi, 1.15)
    for w in (0.7, 0.85):
        p = np.zeros(len(y), dtype=np.float64)
        for f in folds["folds"]:
            tr = np.asarray(f["train_idx"], dtype=int); va = np.asarray(f["val_idx"], dtype=int)
            p[va], _ = leaky_soft(angle, y, tr, va, p_sharp[va], w)
        add(f"multi_sharp1.15_softleaky_w{w}", p)

    log = {"run_id": C.run_id("p76exp"), "nn_pred_source": npz, "results": results}
    out = os.path.join(C.LOGS, f"p7_6_experiment_{log['run_id']}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P7.6-exp] -> {out}")


if __name__ == "__main__":
    main()

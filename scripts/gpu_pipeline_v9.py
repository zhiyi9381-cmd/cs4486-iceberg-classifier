# -*- coding: utf-8 -*-
"""
GPU v9 pipeline: self-supervised pretraining + multi-round pseudo-label iteration (snowball) + VGG9 multiseed + TTA + meta + submission
On top of v8: two pseudo-label iterations (each round relabels with a stronger model, thresholds gradually relaxed), aiming to approach the top-5 threshold of 0.08883
Usage: python gpu_pipeline_v9.py
"""
import os
import json
import glob
import hashlib
import platform
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb

import common as C
import folds as F
import data as D
import oof_cache
from p4_meta import LGBM_CFG
from p7_5_submission_v2 import build_tr_va_exact, build_te_exact
from p7_7_labelprior import build_label_prior
from vgg9 import VGG9

SEEDS = [42, 7, 2024, 1234, 0]
PRETRAIN_EPOCHS = 15
PRETRAIN_NOISE = 0.15
EPOCHS = 60
BATCH = 32
LR = 1e-3
WD = 5e-4
DROPOUT = 0.4
SMOOTHING = 0.05
PSEUDO_HI1 = 0.9
PSEUDO_LO1 = 0.1
PSEUDO_HI2 = 0.85
PSEUDO_LO2 = 0.15
LP_K = 2
W_SOFT = 0.5
LGBM_SEEDS = [42, 7, 2024]
PRIOR_HI = 0.99
PRIOR_LO = 0.01


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1),
        )
        self.upsample = nn.Upsample(size=(75, 75), mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.upsample(self.dec(x))


def pretrain_denoise(all_img, device):
    vgg = VGG9().to(device)
    encoder = vgg.features
    decoder = Decoder().to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()
    n = len(all_img)
    X = torch.from_numpy(all_img).to(device)
    for ep in range(PRETRAIN_EPOCHS):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            noisy = X[idx] + torch.randn_like(X[idx]) * PRETRAIN_NOISE
            opt.zero_grad()
            recon = decoder(encoder(noisy))
            loss = loss_fn(recon, X[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0:
            print(f"  pretrain ep {ep+1}: mse={tot/n:.5f}", flush=True)
    return encoder.state_dict()


def smooth_bce(logits, targets, smoothing=SMOOTHING):
    y = targets * (1 - smoothing) + 0.5 * smoothing
    return nn.BCEWithLogitsLoss()(logits, y)


def predict_tta(model, X, A, device):
    preds = []
    for flip_h in (False, True):
        for flip_v in (False, True):
            Xt = X
            if flip_h:
                Xt = torch.flip(Xt, dims=[3])
            if flip_v:
                Xt = torch.flip(Xt, dims=[2])
            with torch.no_grad():
                p = torch.sigmoid(model(Xt, A)).cpu().numpy().reshape(-1)
            preds.append(p)
    return np.mean(preds, axis=0)


def train_predict(img_tr, angle_tr, y_tr, img_pr, angle_pr, seed, pretrained_enc, device):
    tr = np.asarray(img_tr)
    med = float(np.nanmedian(angle_tr))
    a = np.where(np.isnan(angle_tr), med, angle_tr).astype(np.float32)
    mu = float(a.mean()); sd = float(a.std(ddof=0)) or 1.0
    a_tr = ((a - mu) / sd).astype(np.float32)
    a_pr = ((np.where(np.isnan(angle_pr), med, angle_pr).astype(np.float32) - mu) / sd).astype(np.float32)

    torch.manual_seed(seed); np.random.seed(seed)
    model = VGG9(dropout=DROPOUT).to(device)
    if pretrained_enc is not None:
        model.features.load_state_dict(pretrained_enc)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    nt = len(tr)
    Xtr = torch.from_numpy(tr).to(device); Atr = torch.from_numpy(a_tr[:, None]).to(device)
    yt = torch.from_numpy(np.asarray(y_tr, dtype=np.float32))[:, None].to(device)
    Xpr = torch.from_numpy(img_pr).to(device); Apr = torch.from_numpy(a_pr[:, None]).to(device)

    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(nt, device=device)
        for i in range(0, nt, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = smooth_bce(model(Xtr[idx], Atr[idx]), yt[idx])
            loss.backward(); opt.step()
        sched.step()
    model.eval()
    return predict_tta(model, Xpr, Apr, device)


def predict_test_5fold(img_tr, angle_tr, y, img_te, angle_te, extra_img, extra_angle, extra_y, seeds, pretrained_enc, device):
    """5-fold (each fold's training set = tr + extra) prediction on test, averaged over seeds. Returns the test prediction."""
    folds = F.load_folds(F.FOLDS_V2)
    pred = np.zeros(len(angle_te), dtype=np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        img_ext = np.concatenate([img_tr[tr], extra_img], axis=0) if len(extra_img) else img_tr[tr]
        angle_ext = np.concatenate([angle_tr[tr], extra_angle], axis=0) if len(extra_angle) else angle_tr[tr]
        y_ext = np.concatenate([y[tr], extra_y], axis=0) if len(extra_y) else y[tr]
        for s in seeds:
            pred += train_predict(img_ext, angle_ext, y_ext, img_te, angle_te, s, pretrained_enc, device)
    return pred / (C.N_SPLITS * len(seeds))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    folds = F.load_folds(F.FOLDS_V2)
    img_tr, angle_tr, y, ids_tr = D.load_train()
    img_te, angle_te, ids_te = D.load_test()
    img_tr = img_tr.astype(np.float32); img_te = img_te.astype(np.float32)
    n = len(y); n_test = len(ids_te)

    # ---- Self-supervised pretraining ----
    print("Self-supervised pretraining...", flush=True)
    all_img = np.concatenate([img_tr, img_te], axis=0)
    mu = all_img.mean(axis=(2, 3), keepdims=True)
    sd = all_img.std(axis=(2, 3), keepdims=True) + 1e-6
    all_img_norm = ((all_img - mu) / sd).astype(np.float32)
    pretrained_enc = pretrain_denoise(all_img_norm, device)
    del all_img, all_img_norm, mu, sd

    # ---- Round 1 pseudo-labels: pure CNN (real) labels ----
    print("Round 1 pseudo-label generation...", flush=True)
    test_pred_r1 = predict_test_5fold(img_tr, angle_tr, y, img_te, angle_te, [], [], [], SEEDS, pretrained_enc, device)
    m1 = (test_pred_r1 > PSEUDO_HI1) | (test_pred_r1 < PSEUDO_LO1)
    print(f"Round 1 pseudo-label count: {int(m1.sum())} ({m1.mean()*100:.1f}%)", flush=True)

    # ---- Round 2 pseudo-labels: retrain on real + pseudo1, relabel (relaxed threshold) ----
    print("Round 2 pseudo-label generation...", flush=True)
    p1_img = img_te[m1]; p1_angle = angle_te[m1]; p1_label = (test_pred_r1[m1] > 0.5).astype(np.int64)
    test_pred_r2 = predict_test_5fold(img_tr, angle_tr, y, img_te, angle_te, p1_img, p1_angle, p1_label, SEEDS, pretrained_enc, device)
    m2 = m1 | ((test_pred_r2 > PSEUDO_HI2) | (test_pred_r2 < PSEUDO_LO2))
    print(f"Round 2 cumulative pseudo-label count: {int(m2.sum())} ({m2.mean()*100:.1f}%)", flush=True)

    # ---- Round 3 (final): retrain on real + pseudo2, produce nn_pred + test prediction ----
    print("Round 3 (final) training...", flush=True)
    p2_img = img_te[m2]; p2_angle = angle_te[m2]; p2_label = (test_pred_r2[m2] > 0.5).astype(np.int64)
    npz_nested = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    nn_nested = np.load(npz_nested)["nn_pred_folds"].astype(np.float64)

    run_id = C.run_id("vggv9")
    nn_pred_folds = np.zeros((C.N_SPLITS, n), dtype=np.float64)
    test_nn_pred = np.zeros((C.N_SPLITS, n_test), dtype=np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        va = np.asarray(f["val_idx"], dtype=int)
        nn_pred_folds[k, tr] = nn_nested[k, tr]
        img_ext = np.concatenate([img_tr[tr], p2_img], axis=0)
        angle_ext = np.concatenate([angle_tr[tr], p2_angle], axis=0)
        y_ext = np.concatenate([y[tr], p2_label], axis=0)
        val_acc = np.zeros(len(va), dtype=np.float64)
        test_acc = np.zeros(n_test, dtype=np.float64)
        for s in SEEDS:
            val_acc += train_predict(img_ext, angle_ext, y_ext, img_tr[va], angle_tr[va], s, pretrained_enc, device)
            test_acc += train_predict(img_ext, angle_ext, y_ext, img_te, angle_te, s, pretrained_enc, device)
            print(f"  final fold {k} seed {s} done", flush=True)
        nn_pred_folds[k, va] = val_acc / len(SEEDS)
        test_nn_pred[k] = test_acc / len(SEEDS)
        print(f"final fold {k} done", flush=True)

    # ---- meta + submission ----
    def fit_multi_lgbm(X_tr, y_tr, X_pr):
        logits = []
        for s in LGBM_SEEDS:
            m = lgb.LGBMClassifier(**dict(LGBM_CFG, random_state=s)).fit(X_tr, y_tr)
            p = np.clip(m.predict_proba(X_pr)[:, 1], 1e-7, 1 - 1e-7)
            logits.append(np.log(p / (1 - p)))
        return 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))

    def label_prior_test(angle_train, y_train, angle_test, k):
        a_tr = np.where(np.isnan(angle_train), np.nanmedian(angle_train), angle_train).astype(np.float64)
        a_te = np.asarray(angle_test, dtype=np.float64)
        out = np.zeros(len(a_te), dtype=np.float64)
        for t in range(len(a_te)):
            d = np.abs(a_tr - a_te[t])
            kk = min(k, len(a_tr))
            out[t] = y_train[np.argpartition(d, kk - 1)[:kk]].mean()
        return out

    def leaky_soft(angle_train, y_train, angle_test, pred, w):
        groups = defaultdict(list)
        for i, a in enumerate(angle_train):
            if not np.isnan(a):
                groups[round(float(a), 4)].append(int(y_train[i]))
        pred = np.asarray(pred, dtype=np.float64).copy()
        n_ovr = 0
        for t in range(len(angle_test)):
            a = round(float(angle_test[t]), 4)
            if a in groups and len(set(groups[a])) == 1:
                prior = PRIOR_HI if groups[a][0] == 1 else PRIOR_LO
                pred[t] = w * pred[t] + (1 - w) * prior
                n_ovr += 1
        return pred, n_ovr

    test_preds = np.zeros((C.N_SPLITS, n_test), dtype=np.float64)
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int); va = np.asarray(f["val_idx"], dtype=int)
        X_tr, _ = build_tr_va_exact(nn_pred_folds[k], angle_tr, y, tr, va)
        lp_tr = build_label_prior(angle_tr, y, tr, tr, LP_K)
        X_tr = np.column_stack([X_tr, lp_tr])
        X_te = build_te_exact(nn_pred_folds[k], angle_tr, test_nn_pred[k], angle_te, tr)
        lp_te = label_prior_test(angle_tr, y, angle_te, LP_K)
        X_te = np.column_stack([X_te, lp_te])
        test_preds[k] = fit_multi_lgbm(X_tr, y[tr], X_te)
        print(f"meta fold {k} done", flush=True)
    test_mean = test_preds.mean(axis=0)
    test_final, n_ovr = leaky_soft(angle_tr, y, angle_te, test_mean, W_SOFT)

    checks = {
        "pred_in_01": bool((test_final > 0).all() and (test_final < 1).all()),
        "pred_min": float(test_final.min()), "pred_max": float(test_final.max()),
        "n_override": int(n_ovr), "n_test": n_test,
        "id_unique": bool(len(set(ids_te)) == n_test),
    }
    sub_dir = os.path.join(C.ROOT, "submissions"); os.makedirs(sub_dir, exist_ok=True)
    csv_path = os.path.join(sub_dir, f"submission_v9_{run_id}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,is_iceberg\n")
        for i in range(n_test):
            f.write(f"{ids_te[i]},{test_final[i]:.6f}\n")
    csv_sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()

    log = {
        "run_id": run_id, "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(), "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda": torch.cuda.is_available(), "lightgbm": lgb.__version__,
        "model": "Self-supervised + 2-round pseudo-label iteration + VGG9 (1.76M) multiseed+TTA + exact_angle_group + label_prior_k2 + multi_LGBM + soft_leaky",
        "seeds": SEEDS, "lgbm_seeds": LGBM_SEEDS, "label_prior_k": LP_K, "soft_leaky_w": W_SOFT,
        "pretrain": {"epochs": PRETRAIN_EPOCHS, "noise": PRETRAIN_NOISE},
        "pseudo_r1": {"hi": PSEUDO_HI1, "lo": PSEUDO_LO1, "n": int(m1.sum())},
        "pseudo_r2": {"hi": PSEUDO_HI2, "lo": PSEUDO_LO2, "n": int(m2.sum())},
        "train_cfg": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD, "dropout": DROPOUT, "label_smoothing": SMOOTHING},
        "pred_checks": checks, "csv_path": csv_path, "csv_sha256": csv_sha,
        "disclosure": "Self-supervised pretraining uses all images (unlabeled only); pseudo-labels are produced by a pure CNN (without angle priors) to avoid self-reinforcement; label_prior/soft_leaky are derived from training-label statistics.",
    }
    out = os.path.join(C.LOGS, f"gpu_pipeline_v9_{run_id}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[GPU v9] csv -> {csv_path}")


if __name__ == "__main__":
    main()

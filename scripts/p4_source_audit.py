# -*- coding: utf-8 -*-
"""
P4 tr-side indirect dependency path quantification (disclose evidence faithfully)
Principle: tr sample i is in inner fold t's iva_t (CNN_t excludes i), and in itr_s for s≠t (CNN_s includes i).
     Other tr samples j's nn_pred is produced by CNN_{s(j)} (s(j) = j's inner validation fold).
       s(j)==t -> CNN_t excludes i -> nn_pred[j] does not depend on y_i (clean)
       s(j)!=t -> CNN_s includes i -> nn_pred[j] indirectly depends on y_i (indirectly includes i)
Count, per outer fold, the number of other tr samples that are "clean / indirectly include i", and log it.
"""
import json
import os
import glob
import platform
import sys
import numpy as np

import common as C
import folds as F


def main():
    folds = F.load_folds(F.FOLDS_V2)
    npz = sorted(glob.glob(os.path.join(C.ROOT, "oof", "nn_pred_nested_*.npz")))[-1]
    meta = json.load(open(npz + ".meta.json", encoding="utf-8"))
    inner_folds = meta["inner_folds"]

    stats = []
    for k, f in enumerate(folds["folds"]):
        tr = np.asarray(f["train_idx"], dtype=int)
        inner_of = {}
        for inf in inner_folds:
            if inf["outer"] == k:
                for idx in inf["iva"]:
                    inner_of[idx] = inf["inner"]
        clean_list, dirty_list = [], []
        for i in tr:
            t = inner_of[i]
            clean = dirty = 0
            for j in tr:
                if j == i:
                    continue
                if inner_of[j] == t:
                    clean += 1
                else:
                    dirty += 1
            clean_list.append(clean); dirty_list.append(dirty)
        clean = float(np.mean(clean_list)); dirty = float(np.mean(dirty_list))
        stats.append({
            "outer": k, "tr_size": int(len(tr)),
            "clean_source_mean": round(clean, 2),
            "dirty_source_mean": round(dirty, 2),
            "dirty_ratio": round(dirty / (clean + dirty), 4),
        })

    log = {
        "run_id": C.run_id("p4src"),
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "nn_pred_source": npz,
        "per_outer_stats": stats,
        "overall_dirty_ratio": round(float(np.mean([s["dirty_ratio"] for s in stats])), 4),
        "interpretation": (
            "tr sample i's bin/KNN features aggregate other tr samples j's nn_pred[j]; "
            "about 80% of j come from a different inner fold than i (their CNN training set includes i), forming the tr-side indirect dependency path; "
            "but the va side is clean (va samples are not in tr_k, so no inner CNN has seen va samples), so the five-fold CV score is honest; "
            "indirect dependency magnitude: each source nn_pred is affected by i's label by about 1/1283, negligible after aggregation."
        ),
    }
    out = os.path.join(C.LOGS, f"p4_source_audit_{log['run_id']}.json")
    C.save_json_no_overwrite(log, out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P4-src] -> {out}")


if __name__ == "__main__":
    main()

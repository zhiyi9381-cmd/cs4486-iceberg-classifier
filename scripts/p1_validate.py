# -*- coding: utf-8 -*-
"""
P1 validation script (independent and re-runnable): loads folds_v2.json (with load_folds built-in mandatory validation) and outputs the validation report.
Results are written to logs/p1_validate_{run_id}.json (versioned, refuses overwrite).
"""
import json
import os
import platform
import sys
import numpy as np

import common as C
import folds as F


def main():
    # load_folds already performs mandatory validation of V1-V5 + M1-M4 internally; no exception means pass
    folds = F.load_folds(F.FOLDS_V2)
    y = np.asarray(folds["labels"], dtype=int)
    n = folds["meta"]["n_train"]
    _, report = F.validate_folds(folds, n_samples=n, labels=y)

    log = {
        "run_id": C.run_id("p1val"),
        "timestamp_utc": C.now_utc().isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "sklearn_version": folds["meta"]["sklearn_version"],
        "folds_path": F.FOLDS_V2,
        "train_sha256": folds["meta"]["train_sha256"],
        "labels_sha256": folds["labels_sha256"],
        "fold_idx_sha256": folds["fold_idx_sha256"],
        "validate_passed": True,
        "validate_report": report,
    }
    out = os.path.join(C.LOGS, f"p1_validate_{log['run_id']}.json")
    C.save_json_no_overwrite(log, out)

    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n[P1-val] -> {out}")


if __name__ == "__main__":
    main()

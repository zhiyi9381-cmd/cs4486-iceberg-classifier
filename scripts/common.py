# -*- coding: utf-8 -*-
"""
Shared constants and utilities (project-wide, no per-script scattered definitions).
- Random seed SEED=42 (project-wide)
- Scoring function uses a fixed eps=1e-15
- Missing-angle convention (set in P0)
- Versioned output helper (refuses to overwrite existing files)
"""
import json
import os
import hashlib
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
FOLDS = os.path.join(ROOT, "folds")
REPORTS = os.path.join(ROOT, "reports")
SCRIPTS = os.path.join(ROOT, "scripts")

TRAIN_JSON = os.path.join(DATA, "train.json")
TEST_JSON = os.path.join(DATA, "test.json")
SAMPLE_SUB = os.path.join(DATA, "sample_submission.csv")

SEED = 42                      # project-wide random seed
N_SPLITS = 5                   # number of folds (uniform 5-fold CV)
LOGLOSS_EPS = 1e-15            # fixed clipping lower bound for scoring
MISSING_SENTINEL_RAW = "na"    # missing string in the raw JSON
MISSING_VALUE = float("nan")   # missing value mapped to np.nan


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def run_id(prefix):
    return prefix + "_" + now_utc().strftime("%Y%m%d_%H%M%S")


def save_json_no_overwrite(obj, path):
    """Write JSON, raising an error if the target already exists (never overwrite)."""
    if os.path.exists(path):
        raise FileExistsError(f"target file already exists, refusing to overwrite: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_of_file(path):
    """Compute the SHA-256 of a file in a streaming fashion (without loading it fully into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sklearn_version():
    """Lazily import and return the scikit-learn version (for log/meta provenance)."""
    import sklearn
    return sklearn.__version__

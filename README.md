# CS4486 AI Course — Mini Project: Statoil/C-CORE Iceberg Classifier Challenge

City University of Hong Kong (Dongguan) — CS4486 (Artificial Intelligence) Mini Project, Group 28.

**Members:** Zhou Zhiyi (Group Leader, 72400776), Huang Yingjie (72400819), Yang Waner (72400322).

---

## Task Overview

Binary classification for the Kaggle [Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge): given a 75×75 dual-polarization SAR image (HH/HV bands) plus an auxiliary incident angle `inc_angle`, predict whether the target is an iceberg (`is_iceberg=1`) or a ship (`is_iceberg=0`). The evaluation metric is binary logarithmic loss (LogLoss).

- Training set: 1,604 images (753 icebergs / 851 ships)
- Test set: 8,424 images (unlabeled)
- 133 training samples have a missing `inc_angle`

## Final Results

| Version | Description | Private LogLoss | Public LogLoss |
|---------|-------------|-----------------|----------------|
| **v8 (final)** | Self-supervised denoising pretraining + one round of pseudo-labeling + multi-seed VGG9 with TTA + angle-feature meta-learner | **0.09955** | 0.12293 |

Compared with the historical competition leaderboard, the Top-5 private-score threshold is about **0.08883**; our 0.09955 is 0.0107 away (late submission does not update the ranking, so we compare against the historical scores).

The full submission trajectory is in `logs/submission_history.json`.

## Solution Architecture

Our solution is a multi-stage "CNN image model + LightGBM meta-learner" pipeline:

1. **CNN image model**: from ShallowCNN (7,290 parameters) → DeepCNN (65,426) → VGG9 (1.76M). We finally adopt VGG9 with multi-seed bagging and 4-way flip TTA.
2. **Meta-learner (LightGBM)**: uses the CNN probability `nn_pred` as a feature, augmented with incident-angle-derived features (exact angle grouping, label prior k=2, soft-leaky post-processing).
3. **Regularization**: self-supervised denoising pretraining (on all 10,028 images) and one round of pseudo-labeling (3,992 high-confidence samples).

Key observation: in this dataset, the incident angle (`inc_angle`) is strongly associated with the label (exact-angle groups share the same label in 99.47% of pairwise cases), making it the single most important information source of the solution.

## Directory Structure

```
method2/
├── scripts/            # Python scripts (data processing, baselines, CNN, meta-learner, GPU pipeline)
├── data/               # Raw data (train/test.json must be downloaded from Kaggle)
├── folds/              # Fixed 5-fold split folds_v2.json (with SHA-256 checksum)
├── logs/               # Per-stage experiment logs + submission_history.json
├── oof/                # Intermediate prediction caches (NPZ, regenerable from scripts)
├── submissions/        # Final submission CSVs (v1-v11)
├── reports/            # Report source (final LaTeX + figures + compiled PDF)
├── requirements.txt    # CPU dependencies (version-locked)
└── requirements_gpu.txt # GPU dependencies
```

## Reproduction Guide

### 1. Obtain the Data

Download the raw data from Kaggle and place it under `data/`:

```bash
data/train.json        # training set
data/test.json         # test set
data/sample_submission.csv
```

> The raw JSON files are too large (test.json ≈ 1.5 GB) to be included in this repository.

### 2. Install Dependencies

CPU development environment:

```bash
pip install -r requirements.txt
```

GPU environment (PyTorch must be installed separately according to the CUDA version; see `requirements_gpu.txt`).

### 3. Run the Pipeline

```bash
# Data audit + fold split
python scripts/p0_audit.py
python scripts/p1_folds.py

# Baseline models
python scripts/p2_baselines.py
python scripts/p2_shallowcnn.py

# DeepCNN + nested OOF + meta-learner
python scripts/p3_deepcnn.py
python scripts/p4_nested_oof.py
python scripts/p4_meta.py

# Final GPU pipeline (requires CUDA)
python scripts/gpu_pipeline_v8.py
```

### 4. Report

Final report source: `reports/Mini_Project_Final_Report.tex` (compiles to PDF).

## Environment

- **CPU development**: Python 3.13.14, numpy 2.5.1, scikit-learn 1.9.0, PyTorch 2.15.0+cpu, LightGBM 4.7.0
- **GPU training (v8)**: Python 3.13.14, PyTorch 2.6.0+cu124, LightGBM 4.7.0

## Notes

- This repository is the CS4486 coursework; the code and data are from the public Kaggle competition.
- Detailed methodology, ablation studies, reproducibility, and reflection are in the final report.

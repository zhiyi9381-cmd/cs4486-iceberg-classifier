# -*- coding: utf-8 -*-
"""
Scoring function (Logarithmic Loss / binary cross-entropy).
- eps is fixed at 1e-15; predicted probabilities are clipped to [eps, 1-eps]
- Formula: -1/n * sum [y*log(p) + (1-y)*log(1-p)]
"""
import numpy as np

import common as C


def logloss(y_true, p_pred, eps=C.LOGLOSS_EPS):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mean_std(values, ddof=0):
    """mean ± std (std uses the population standard deviation ddof=0, as required)."""
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=ddof))

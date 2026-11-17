"""Forecast-accuracy metrics, computed by hand from first principles and
exposed alongside their scikit-learn equivalents. tests/test_metrics.py
cross-checks the two agree, the same "derive it, then check it against
the library" pattern used throughout this project.
"""
from __future__ import annotations

import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """n_features = number of predictors, NOT counting the intercept."""
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    denom = n - n_features - 1
    if denom <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / denom


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, as a fraction (0.05 = 5%). Excludes
    any points where y_true == 0 to avoid a division-by-zero blowup, tbilrate touches ~0 in 2008-09, so this matters for this dataset."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

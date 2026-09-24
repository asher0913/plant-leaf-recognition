"""Classification metrics and across-split summary statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Two-sided 95% Student-t critical values for small numbers of repeats.
_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Top-1 accuracy plus weighted and macro F1."""
    return {
        "top1": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def per_class_accuracy(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    support = matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(support > 0, np.diag(matrix) / support, np.nan)


def _t_critical(dof: int) -> float:
    if dof in _T975:
        return _T975[dof]
    smaller = [k for k in _T975 if k < dof]
    return _T975[max(smaller)] if dof <= 30 else 1.96


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample standard deviation, range and a t-based 95% interval.

    The interval treats splits as independent. Repeated random splits share
    most of their training data, so it is a descriptive band rather than a
    rigorous confidence interval on generalisation error.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarise an empty sequence")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    half_width = _t_critical(array.size - 1) * std / math.sqrt(array.size) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }

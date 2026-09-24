import math

import numpy as np
import pytest

from leafnet.metrics import classification_metrics, per_class_accuracy, summarize


def test_classification_metrics_perfect_and_partial():
    assert classification_metrics([0, 1, 2], [0, 1, 2]) == {
        "top1": 1.0,
        "f1_weighted": 1.0,
        "f1_macro": 1.0,
    }
    partial = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert partial["top1"] == pytest.approx(0.75)
    assert 0 < partial["f1_macro"] < 1


def test_per_class_accuracy_marks_absent_classes_nan():
    accuracy = per_class_accuracy([0, 0, 1], [0, 1, 1], num_classes=3)
    assert accuracy[0] == pytest.approx(0.5)
    assert accuracy[1] == pytest.approx(1.0)
    assert math.isnan(accuracy[2])


def test_summarize_uses_sample_std_and_t_interval():
    values = [0.97, 0.98, 0.99, 1.00]
    summary = summarize(values)
    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(0.985)
    assert summary["std"] == pytest.approx(np.std(values, ddof=1))
    half = 3.182 * summary["std"] / 2
    assert summary["ci95_low"] == pytest.approx(0.985 - half)
    assert summary["min"] == 0.97 and summary["max"] == 1.00


def test_summarize_single_value_and_empty():
    assert summarize([0.5])["std"] == 0.0
    with pytest.raises(ValueError):
        summarize([])

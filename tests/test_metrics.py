import math

from health_misinfo.evaluation.bootstrap import bootstrap_intervals
from health_misinfo.evaluation.metrics import classification_metrics, confusion_matrix_frame


def test_classification_metrics_on_toy_case():
    y_true = ["reliable", "reliable", "unreliable", "unreliable"]
    y_pred = ["reliable", "unreliable", "unreliable", "unreliable"]
    scores = [0.1, 0.8, 0.7, 0.9]
    metrics = classification_metrics(y_true, y_pred, scores)
    assert metrics["n"] == 4
    assert math.isclose(metrics["reliable_recall"], 0.5)
    assert math.isclose(metrics["unreliable_recall"], 1.0)
    assert "unreliable_pr_auc" in metrics


def test_confusion_matrix_shape():
    cm = confusion_matrix_frame(["reliable", "unreliable"], ["reliable", "reliable"])
    assert list(cm.index) == ["true_reliable", "true_unreliable"]
    assert list(cm.columns) == ["pred_reliable", "pred_unreliable"]


def test_bootstrap_intervals_returns_expected_metrics():
    y_true = ["reliable", "reliable", "unreliable", "unreliable"]
    y_pred = ["reliable", "unreliable", "unreliable", "unreliable"]
    scores = [0.1, 0.8, 0.7, 0.9]
    intervals = bootstrap_intervals(y_true, y_pred, scores, iterations=5, seed=1)
    assert set(intervals["metric"]) == {"macro_f1", "unreliable_f1", "unreliable_pr_auc"}


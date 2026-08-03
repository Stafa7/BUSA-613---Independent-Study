import numpy as np

from health_misinfo.evaluation.calibration import (
    calibration_metrics,
    fit_calibrator,
    selective_result,
    threshold_for_coverage,
)


def test_platt_calibrator_returns_probabilities():
    scores = np.array([-2.0, -1.0, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1])
    calibrator = fit_calibrator("platt", scores, labels)
    probabilities = calibrator.predict(scores)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert probabilities[-1] > probabilities[0]


def test_calibration_and_selective_outputs_are_bounded():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = calibration_metrics(labels, probabilities)
    threshold = threshold_for_coverage(probabilities, 0.5)
    selective = selective_result(labels, probabilities, threshold)
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["ece_10"] <= 1
    assert 0 <= metrics["adaptive_ece_10"] <= 1
    assert 0 <= selective["coverage"] <= 1


def test_selective_result_uses_declared_decision_threshold():
    labels = np.array([0, 1])
    probabilities = np.array([0.55, 0.65])
    result = selective_result(
        labels,
        probabilities,
        confidence_threshold=0,
        decision_threshold=0.6,
    )
    assert result["selective_risk"] == 0

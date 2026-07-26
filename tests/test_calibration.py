import numpy as np

from health_misinfo.evaluation.calibration import (
    calibration_metrics,
    decision_confidence,
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


def test_decision_confidence_matches_class_probability_at_default_threshold():
    probabilities = np.array([0.0, 0.1, 0.5, 0.8, 1.0])
    confidence = decision_confidence(probabilities)
    np.testing.assert_allclose(confidence, np.maximum(probabilities, 1 - probabilities))


def test_decision_confidence_is_normalized_on_both_sides_of_nondefault_threshold():
    threshold = 0.25
    probabilities = np.array([0.0, 0.125, 0.25, 0.625, 1.0])
    confidence = decision_confidence(probabilities, decision_threshold=threshold)

    np.testing.assert_allclose(confidence, [1.0, 0.75, 0.5, 0.75, 1.0])
    assert confidence[0] > confidence[1] > confidence[2]
    assert confidence[2] < confidence[3] < confidence[4]
    assert confidence[1] == confidence[3]


def test_nondefault_boundary_is_least_confident_even_when_probability_exceeds_half():
    probabilities = np.array([0.1, 0.6, 0.8])
    confidence = decision_confidence(probabilities, decision_threshold=0.6)

    assert confidence[1] == 0.5
    assert confidence[1] < confidence[0]
    assert confidence[1] < confidence[2]


def test_threshold_for_coverage_uses_declared_decision_boundary():
    probabilities = np.array([0.0, 0.1, 0.2, 0.4, 0.8])
    confidence_threshold = threshold_for_coverage(
        probabilities,
        coverage=0.4,
        decision_threshold=0.2,
    )
    confidence = decision_confidence(probabilities, decision_threshold=0.2)

    assert confidence_threshold == 0.875
    np.testing.assert_array_equal(
        confidence >= confidence_threshold,
        [True, False, False, False, True],
    )


def test_threshold_for_coverage_does_not_undershoot_discrete_target():
    probabilities = np.array([0.0, 0.1, 0.2, 0.4, 0.8])
    target_coverage = 0.7
    confidence_threshold = threshold_for_coverage(
        probabilities,
        coverage=target_coverage,
        decision_threshold=0.2,
    )
    achieved_coverage = (
        decision_confidence(probabilities, decision_threshold=0.2)
        >= confidence_threshold
    ).mean()

    assert achieved_coverage == 0.8
    assert achieved_coverage >= target_coverage


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


def test_selective_result_defers_near_boundary_cases_on_both_predicted_sides():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.2, 0.3, 1.0])
    result = selective_result(
        labels,
        probabilities,
        confidence_threshold=0.75,
        decision_threshold=0.25,
    )

    assert result["coverage"] == 0.5
    assert result["automated_records"] == 2
    assert result["deferred_records"] == 2
    assert result["reliable_side_coverage"] == 0.5
    assert result["unreliable_side_coverage"] == 0.5
    assert result["reliable_probability_ceiling"] == 0.125
    assert result["unreliable_probability_floor"] == 0.625


def test_threshold_relative_confidence_rejects_invalid_inputs():
    with np.testing.assert_raises(ValueError):
        decision_confidence([0.2, np.nan])
    with np.testing.assert_raises(ValueError):
        decision_confidence([-0.1, 0.2])
    with np.testing.assert_raises(ValueError):
        decision_confidence([0.2], decision_threshold=1.1)
    with np.testing.assert_raises(ValueError):
        threshold_for_coverage([0.2], coverage=0)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from health_misinfo.evaluation.metrics import expected_calibration_error


@dataclass
class ScoreCalibrator:
    method: str
    model: object | None = None

    def predict(self, scores) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if self.method == "uncalibrated":
            return np.clip(values, 0.0, 1.0)
        if self.method == "platt":
            return self.model.predict_proba(values.reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return np.clip(self.model.predict(values), 0.0, 1.0)
        raise ValueError(f"Unknown calibration method: {self.method}")


def fit_calibrator(method: str, scores, labels) -> ScoreCalibrator:
    values = np.asarray(scores, dtype=float)
    binary = np.asarray(labels, dtype=int)
    if method == "uncalibrated":
        return ScoreCalibrator(method)
    if method == "platt":
        model = LogisticRegression(random_state=613)
        model.fit(values.reshape(-1, 1), binary)
        return ScoreCalibrator(method, model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(values, binary)
        return ScoreCalibrator(method, model)
    raise ValueError(f"Unknown calibration method: {method}")


def calibration_metrics(labels, probabilities) -> dict[str, float]:
    binary = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    metrics = {
        "brier_score": float(brier_score_loss(binary, probability)),
        "ece_10": expected_calibration_error(binary, probability, bins=10),
        "adaptive_ece_10": adaptive_calibration_error(binary, probability, bins=10),
    }
    if len(np.unique(binary)) == 2 and len(np.unique(probability)) > 1:
        logit = np.log(probability / (1 - probability)).reshape(-1, 1)
        calibration_model = LogisticRegression(C=1e6, max_iter=2_000)
        calibration_model.fit(logit, binary)
        metrics["calibration_intercept"] = float(calibration_model.intercept_[0])
        metrics["calibration_slope"] = float(calibration_model.coef_[0, 0])
    else:
        metrics["calibration_intercept"] = np.nan
        metrics["calibration_slope"] = np.nan
    return metrics


def adaptive_calibration_error(labels, probabilities, bins: int = 10) -> float:
    binary = np.asarray(labels, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if len(binary) == 0:
        return float("nan")
    order = np.argsort(probability)
    groups = np.array_split(order, min(bins, len(order)))
    return float(
        sum(
            len(group)
            / len(binary)
            * abs(float(binary[group].mean()) - float(probability[group].mean()))
            for group in groups
            if len(group)
        )
    )


def reliability_table(labels, probabilities, bins: int = 10) -> pd.DataFrame:
    binary = np.asarray(labels, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for index in range(bins):
        mask = bucket == index
        rows.append(
            {
                "bin": index + 1,
                "lower": edges[index],
                "upper": edges[index + 1],
                "records": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()) if mask.any() else np.nan,
                "observed_unreliable_rate": float(binary[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def decision_confidence(
    probabilities,
    decision_threshold: float = 0.5,
) -> np.ndarray:
    """Return confidence in a thresholded decision, normalized within each side.

    A probability exactly on the decision boundary has confidence 0.5. Confidence
    then increases linearly to 1.0 at the endpoint of either predicted class:

    - below the boundary, distance is scaled by ``decision_threshold``;
    - at or above the boundary, distance is scaled by ``1 - decision_threshold``.

    This equals ``max(p, 1 - p)`` when the decision threshold is 0.5, while
    remaining consistent with classifiers whose selected operating threshold is
    not 0.5.
    """

    probability = np.asarray(probabilities, dtype=float)
    threshold = float(decision_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("decision_threshold must be finite and between 0 and 1")
    if np.any(~np.isfinite(probability)):
        raise ValueError("probabilities must all be finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must all be between 0 and 1")

    relative_margin = np.zeros_like(probability, dtype=float)
    reliable_side = probability < threshold
    unreliable_side = ~reliable_side
    if threshold > 0.0:
        relative_margin[reliable_side] = (
            threshold - probability[reliable_side]
        ) / threshold
    if threshold < 1.0:
        relative_margin[unreliable_side] = (
            probability[unreliable_side] - threshold
        ) / (1.0 - threshold)
    return 0.5 + 0.5 * relative_margin


def threshold_for_coverage(
    probabilities,
    coverage: float,
    decision_threshold: float = 0.5,
) -> float:
    if not np.isfinite(coverage) or not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be finite and in (0, 1]")
    confidence = decision_confidence(probabilities, decision_threshold)
    if confidence.size == 0:
        raise ValueError("probabilities must contain at least one value")
    if coverage >= 1:
        return 0.0
    retained_target = int(np.ceil(coverage * confidence.size))
    cutoff_index = confidence.size - retained_target
    return float(np.partition(confidence, cutoff_index)[cutoff_index])


def selective_result(
    labels,
    probabilities,
    confidence_threshold: float,
    decision_threshold: float = 0.5,
) -> dict[str, float | int]:
    binary = np.asarray(labels, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if binary.shape != probability.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if not np.isfinite(confidence_threshold):
        raise ValueError("confidence_threshold must be finite")
    confidence = decision_confidence(probability, decision_threshold)
    retained = confidence >= confidence_threshold
    prediction = (probability >= decision_threshold).astype(int)
    if retained.any():
        selective_risk = float((prediction[retained] != binary[retained]).mean())
        macro_f1 = float(f1_score(binary[retained], prediction[retained], average="macro", zero_division=0))
    else:
        selective_risk = np.nan
        macro_f1 = np.nan
    deferred = ~retained
    predicted_reliable = prediction == 0
    predicted_unreliable = prediction == 1
    normalized_margin = float(np.clip(2.0 * confidence_threshold - 1.0, 0.0, 1.0))
    return {
        "confidence_threshold": confidence_threshold,
        "decision_threshold": decision_threshold,
        "reliable_probability_ceiling": decision_threshold * (1.0 - normalized_margin),
        "unreliable_probability_floor": decision_threshold
        + (1.0 - decision_threshold) * normalized_margin,
        "coverage": float(retained.mean()),
        "automated_records": int(retained.sum()),
        "deferred_records": int(deferred.sum()),
        "reliable_side_coverage": (
            float(retained[predicted_reliable].mean())
            if predicted_reliable.any()
            else np.nan
        ),
        "unreliable_side_coverage": (
            float(retained[predicted_unreliable].mean())
            if predicted_unreliable.any()
            else np.nan
        ),
        "selective_risk": selective_risk,
        "macro_f1": macro_f1,
        "deferred_error_rate": float((prediction[deferred] != binary[deferred]).mean()) if deferred.any() else np.nan,
        "deferred_unreliable_fraction": float(binary[deferred].mean()) if deferred.any() else np.nan,
    }


def choose_operating_thresholds(labels, probabilities) -> dict[str, float]:
    binary = np.asarray(labels, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probability)))
    rows = []
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(binary, prediction, zero_division=0),
                "recall": recall_score(binary, prediction, zero_division=0),
                "macro_f1": f1_score(binary, prediction, average="macro", zero_division=0),
            }
        )
    frame = pd.DataFrame(rows)
    recall_eligible = frame[frame["recall"] >= 0.90]
    recall_threshold = (
        float(recall_eligible.sort_values(["precision", "threshold"], ascending=False).iloc[0]["threshold"])
        if len(recall_eligible)
        else 0.5
    )
    precision_eligible = frame[frame["precision"] >= 0.80]
    precision_threshold = (
        float(precision_eligible.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
        if len(precision_eligible)
        else 0.5
    )
    return {
        "default": 0.5,
        "macro_f1_validation_optimum": float(
            frame.sort_values(["macro_f1", "threshold"], ascending=[False, True]).iloc[0]["threshold"]
        ),
        "recall_oriented_validation_target_0.90": recall_threshold,
        "precision_oriented_validation_target_0.80": precision_threshold,
    }


def operating_point_result(labels, probabilities, threshold: float) -> dict[str, float | int]:
    binary = np.asarray(labels, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    prediction = (probability >= threshold).astype(int)
    return {
        "threshold": threshold,
        "records": len(binary),
        "precision_unreliable": float(precision_score(binary, prediction, zero_division=0)),
        "recall_unreliable": float(recall_score(binary, prediction, zero_division=0)),
        "macro_f1": float(f1_score(binary, prediction, average="macro", zero_division=0)),
        "predicted_unreliable": int(prediction.sum()),
    }

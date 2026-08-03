from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


LABELS = ["reliable", "unreliable"]
POSITIVE_LABEL = "unreliable"


def _positive_scores(y_score: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(y_score)
    if arr.ndim == 2:
        return arr[:, 1]
    return arr.astype(float)


def expected_calibration_error(y_true_binary, probabilities, bins: int = 10) -> float:
    true = np.asarray(y_true_binary, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            error += mask.mean() * abs(true[mask].mean() - probability[mask].mean())
    return float(error)


def classification_metrics(
    y_true: list[str] | pd.Series,
    y_pred: list[str] | pd.Series,
    y_score=None,
    probability_scores: bool = False,
) -> dict[str, float | int]:
    true = pd.Series(y_true).astype(str)
    pred = pd.Series(y_pred).astype(str)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=LABELS, zero_division=0
    )
    metrics: dict[str, float | int] = {
        "n": int(len(true)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=LABELS, average="weighted", zero_division=0)),
    }
    for i, label in enumerate(LABELS):
        metrics[f"{label}_precision"] = float(precision[i])
        metrics[f"{label}_recall"] = float(recall[i])
        metrics[f"{label}_f1"] = float(f1[i])
        metrics[f"{label}_support"] = int(support[i])
    if y_score is not None:
        binary = (true == POSITIVE_LABEL).astype(int).to_numpy()
        scores = _positive_scores(y_score)
        if len(np.unique(binary)) == 2:
            metrics["unreliable_pr_auc"] = float(average_precision_score(binary, scores))
            metrics["roc_auc"] = float(roc_auc_score(binary, scores))
        else:
            metrics["unreliable_pr_auc"] = float("nan")
            metrics["roc_auc"] = float("nan")
        if probability_scores:
            probabilities = np.clip(scores, 0.0, 1.0)
            metrics["brier_score"] = float(brier_score_loss(binary, probabilities))
            metrics["ece_10"] = expected_calibration_error(binary, probabilities, bins=10)
    return metrics


def confusion_matrix_frame(y_true: list[str] | pd.Series, y_pred: list[str] | pd.Series) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return pd.DataFrame(cm, index=[f"true_{label}" for label in LABELS], columns=[f"pred_{label}" for label in LABELS])


def metrics_table(metrics: dict[str, float | int]) -> pd.DataFrame:
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])


def write_metrics(metrics: dict[str, float | int], out_dir: Path) -> None:
    serializable = {k: (None if pd.isna(v) else v) for k, v in metrics.items()}
    (out_dir / "metrics.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    metrics_table(metrics).to_csv(out_dir / "metrics_table.csv", index=False)

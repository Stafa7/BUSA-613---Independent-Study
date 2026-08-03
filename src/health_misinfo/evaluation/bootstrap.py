from __future__ import annotations

import numpy as np
import pandas as pd

from health_misinfo.evaluation.metrics import classification_metrics


CORE_BOOTSTRAP_METRICS = (
    "macro_f1",
    "unreliable_f1",
    "unreliable_pr_auc",
    "roc_auc",
)


def _prepare_group_positions(groups=None) -> list[np.ndarray] | None:
    if groups is None:
        return None
    group_values = np.asarray(groups)
    return [np.flatnonzero(group_values == group) for group in pd.unique(group_values)]


def _resampled_indices(
    rng: np.random.Generator,
    n: int,
    group_positions: list[np.ndarray] | None = None,
) -> np.ndarray:
    if group_positions is None:
        return rng.integers(0, n, size=n)
    sampled = rng.integers(0, len(group_positions), size=len(group_positions))
    return np.concatenate([group_positions[index] for index in sampled])


def bootstrap_intervals(
    y_true,
    y_pred,
    y_score=None,
    iterations: int = 1_000,
    seed: int = 613,
    groups=None,
    probability_scores: bool = False,
) -> pd.DataFrame:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    score = None if y_score is None else np.asarray(y_score)
    group_positions = _prepare_group_positions(groups)
    rng = np.random.default_rng(seed)
    rows = []
    n = len(true)
    for _ in range(iterations):
        idx = _resampled_indices(rng, n, group_positions)
        sample_score = None if score is None else score[idx]
        metrics = classification_metrics(
            true[idx],
            pred[idx],
            sample_score,
            probability_scores=probability_scores,
        )
        rows.append({metric: metrics.get(metric, np.nan) for metric in CORE_BOOTSTRAP_METRICS})
    frame = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": frame[metric].mean(),
                "ci_low": frame[metric].quantile(0.025),
                "ci_high": frame[metric].quantile(0.975),
                "iterations": iterations,
                "resampling_unit": "group" if groups is not None else "record",
            }
            for metric in CORE_BOOTSTRAP_METRICS
        ]
    )


def paired_bootstrap_difference(
    y_true,
    pred_a,
    pred_b,
    score_a=None,
    score_b=None,
    iterations: int = 1_000,
    seed: int = 613,
    groups=None,
) -> pd.DataFrame:
    true = np.asarray(y_true)
    prediction_a = np.asarray(pred_a)
    prediction_b = np.asarray(pred_b)
    scores_a = None if score_a is None else np.asarray(score_a)
    scores_b = None if score_b is None else np.asarray(score_b)
    group_positions = _prepare_group_positions(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(iterations):
        idx = _resampled_indices(rng, len(true), group_positions)
        metrics_a = classification_metrics(
            true[idx],
            prediction_a[idx],
            None if scores_a is None else scores_a[idx],
        )
        metrics_b = classification_metrics(
            true[idx],
            prediction_b[idx],
            None if scores_b is None else scores_b[idx],
        )
        rows.append(
            {
                metric: metrics_a.get(metric, np.nan) - metrics_b.get(metric, np.nan)
                for metric in CORE_BOOTSTRAP_METRICS
            }
        )
    frame = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean_difference_a_minus_b": frame[metric].mean(),
                "ci_low": frame[metric].quantile(0.025),
                "ci_high": frame[metric].quantile(0.975),
                "iterations": iterations,
                "resampling_unit": "group" if groups is not None else "record",
            }
            for metric in CORE_BOOTSTRAP_METRICS
        ]
    )

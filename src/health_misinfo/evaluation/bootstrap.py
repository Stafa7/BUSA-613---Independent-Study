from __future__ import annotations

import numpy as np
import pandas as pd

from health_misinfo.evaluation.metrics import classification_metrics


def bootstrap_intervals(
    y_true,
    y_pred,
    y_score=None,
    iterations: int = 200,
    seed: int = 613,
) -> pd.DataFrame:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    score = None if y_score is None else np.asarray(y_score)
    rng = np.random.default_rng(seed)
    rows = []
    n = len(true)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        sample_score = None if score is None else score[idx]
        metrics = classification_metrics(true[idx], pred[idx], sample_score)
        rows.append(
            {
                "macro_f1": metrics["macro_f1"],
                "unreliable_f1": metrics["unreliable_f1"],
                "unreliable_pr_auc": metrics.get("unreliable_pr_auc", np.nan),
            }
        )
    frame = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "metric": col,
                "mean": frame[col].mean(),
                "ci_low": frame[col].quantile(0.025),
                "ci_high": frame[col].quantile(0.975),
            }
            for col in frame.columns
        ]
    )


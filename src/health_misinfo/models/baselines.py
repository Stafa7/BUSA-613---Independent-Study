from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    estimator: object


def baseline_specs() -> list[BaselineSpec]:
    tfidf = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2, strip_accents="unicode")
    return [
        BaselineSpec("majority", DummyClassifier(strategy="most_frequent")),
        BaselineSpec(
            "logistic_regression",
            Pipeline(
                [
                    ("tfidf", tfidf),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            ),
        ),
        BaselineSpec(
            "linear_svm",
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2, strip_accents="unicode")),
                    ("clf", LinearSVC(class_weight="balanced")),
                ]
            ),
        ),
    ]


def fit_predict(spec: BaselineSpec, train_text, train_y, test_text) -> tuple[np.ndarray, np.ndarray]:
    model = spec.estimator
    model.fit(train_text, train_y)
    pred = model.predict(test_text)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(test_text)
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(test_text)
        score = raw if raw.ndim == 1 else raw[:, 1]
    else:
        score = np.array([1.0 if value == "unreliable" else 0.0 for value in pred])
    return pred, score


def top_features(model: object, limit: int = 50) -> pd.DataFrame:
    if not isinstance(model, Pipeline) or "tfidf" not in model.named_steps:
        return pd.DataFrame(columns=["feature", "weight", "direction"])
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame(columns=["feature", "weight", "direction"])
    names = np.asarray(vectorizer.get_feature_names_out())
    weights = clf.coef_[0]
    top_pos = np.argsort(weights)[-limit:][::-1]
    top_neg = np.argsort(weights)[:limit]
    rows = [
        {"feature": names[i], "weight": float(weights[i]), "direction": "unreliable"} for i in top_pos
    ] + [
        {"feature": names[i], "weight": float(weights[i]), "direction": "reliable"} for i in top_neg
    ]
    return pd.DataFrame(rows)


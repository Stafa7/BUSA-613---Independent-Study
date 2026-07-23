from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    hyperparameters: dict[str, Any]
    probability_scores: bool


def _vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    tfidf = config["models"]["tfidf"]
    return TfidfVectorizer(
        max_features=int(tfidf["max_features"]),
        ngram_range=tuple(tfidf["ngram_range"]),
        min_df=int(tfidf["min_df"]),
        strip_accents="unicode",
    )


def baseline_candidates(config: dict[str, Any]) -> dict[str, list[BaselineSpec]]:
    seed = int(config["random_seed"])
    logistic = config["models"]["logistic_regression"]
    svm = config["models"]["linear_svm"]
    candidates: dict[str, list[BaselineSpec]] = {
        "majority": [
            BaselineSpec(
                "majority",
                DummyClassifier(strategy="most_frequent"),
                {"strategy": "most_frequent"},
                True,
            )
        ],
        "logistic_regression": [],
        "linear_svm": [],
    }
    for c_value in logistic["c_grid"]:
        candidates["logistic_regression"].append(
            BaselineSpec(
                "logistic_regression",
                Pipeline(
                    [
                        ("tfidf", _vectorizer(config)),
                        (
                            "clf",
                            LogisticRegression(
                                C=float(c_value),
                                max_iter=int(logistic["max_iter"]),
                                class_weight=logistic["class_weight"],
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
                {"C": float(c_value), "max_iter": int(logistic["max_iter"])},
                True,
            )
        )
    for c_value in svm["c_grid"]:
        candidates["linear_svm"].append(
            BaselineSpec(
                "linear_svm",
                Pipeline(
                    [
                        ("tfidf", _vectorizer(config)),
                        ("clf", LinearSVC(C=float(c_value), class_weight=svm["class_weight"])),
                    ]
                ),
                {"C": float(c_value)},
                False,
            )
        )
    return candidates


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

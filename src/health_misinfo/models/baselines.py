from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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


def tfidf_parameter_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen representation grid in deterministic tie-break order."""

    tfidf = config["models"]["tfidf"]
    max_features = tfidf.get("max_features_grid", [tfidf["max_features"]])
    minimum_document_frequencies = tfidf.get("min_df_grid", [tfidf["min_df"]])
    sublinear_values = tfidf.get("sublinear_tf_grid", [tfidf.get("sublinear_tf", False)])
    ngram_range = tuple(int(value) for value in tfidf["ngram_range"])
    return [
        {
            "tfidf_max_features": int(features),
            "tfidf_ngram_min": int(ngram_range[0]),
            "tfidf_ngram_max": int(ngram_range[1]),
            "tfidf_min_df": int(minimum_frequency),
            "tfidf_sublinear_tf": bool(sublinear),
        }
        for features, minimum_frequency, sublinear in product(
            max_features,
            minimum_document_frequencies,
            sublinear_values,
        )
    ]


def _vectorizer(parameters: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=int(parameters["tfidf_max_features"]),
        ngram_range=(
            int(parameters["tfidf_ngram_min"]),
            int(parameters["tfidf_ngram_max"]),
        ),
        min_df=int(parameters["tfidf_min_df"]),
        sublinear_tf=bool(parameters["tfidf_sublinear_tf"]),
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
    representation_grid = tfidf_parameter_grid(config)
    for representation in representation_grid:
        for c_value in logistic["c_grid"]:
            hyperparameters = {
                **representation,
                "C": float(c_value),
                "max_iter": int(logistic["max_iter"]),
            }
            candidates["logistic_regression"].append(
                BaselineSpec(
                    "logistic_regression",
                    Pipeline(
                        [
                            ("tfidf", _vectorizer(representation)),
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
                    hyperparameters,
                    True,
                )
            )
    for representation in representation_grid:
        for c_value in svm["c_grid"]:
            candidates["linear_svm"].append(
                BaselineSpec(
                    "linear_svm",
                    Pipeline(
                        [
                            ("tfidf", _vectorizer(representation)),
                            (
                                "clf",
                                LinearSVC(
                                    C=float(c_value),
                                    class_weight=svm["class_weight"],
                                    random_state=seed,
                                ),
                            ),
                        ]
                    ),
                    {**representation, "C": float(c_value)},
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

#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_experiments, load_paths
from health_misinfo.evaluation.bootstrap import (
    bootstrap_intervals,
    paired_bootstrap_difference,
)
from health_misinfo.evaluation.metrics import classification_metrics


NUMERIC_COLUMNS = ["tweet_count", "reply_count", "retweet_count"]


def _estimator(config: dict, feature_set: str, c_value: float) -> Pipeline:
    logistic = config["models"]["logistic_regression"]
    numeric = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    if feature_set == "engagement_only":
        features = ColumnTransformer([("engagement", numeric, NUMERIC_COLUMNS)])
    elif feature_set == "text_only":
        tfidf = config["models"]["tfidf"]
        features = ColumnTransformer(
            [
                (
                    "text",
                    TfidfVectorizer(
                        max_features=int(tfidf["max_features"]),
                        ngram_range=tuple(tfidf["ngram_range"]),
                        min_df=int(tfidf["min_df"]),
                        strip_accents="unicode",
                    ),
                    "model_text",
                )
            ]
        )
    elif feature_set == "text_plus_engagement":
        tfidf = config["models"]["tfidf"]
        features = ColumnTransformer(
            [
                (
                    "text",
                    TfidfVectorizer(
                        max_features=int(tfidf["max_features"]),
                        ngram_range=tuple(tfidf["ngram_range"]),
                        min_df=int(tfidf["min_df"]),
                        strip_accents="unicode",
                    ),
                    "model_text",
                ),
                ("engagement", numeric, NUMERIC_COLUMNS),
            ]
        )
    else:
        raise ValueError(feature_set)
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=int(logistic["max_iter"]),
                    class_weight=logistic["class_weight"],
                    random_state=int(config["random_seed"]),
                ),
            ),
        ]
    )


def _predict(estimator: Pipeline, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    prediction = estimator.predict(frame)
    probabilities = estimator.predict_proba(frame)[:, 1]
    return prediction, probabilities


def main() -> int:
    paths = load_paths()
    config = load_experiments()
    latest = json.loads(
        (paths["experiments"] / "latest_run.json").read_text(encoding="utf-8")
    )
    run_root = Path(latest["run_path"])
    out_root = run_root / "engagement_ablation"
    out_root.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[
        data["dataset"].eq("fakehealth")
        & data["primary_eligible"]
        & data["engagement_available"]
    ].copy()
    iterations = int(config["metrics"]["bootstrap_iterations"])
    seed = int(config["random_seed"])
    summary_rows = []
    prediction_sets: dict[tuple[str, str], pd.DataFrame] = {}

    for split_name, manifest_name in (
        ("standard", "standard_split_manifest.csv"),
        ("controlled", "controlled_split_manifest.csv"),
    ):
        manifest = pd.read_csv(paths["splits"] / manifest_name)
        manifest = manifest[manifest["dataset"].eq("fakehealth")]
        joined = data.merge(
            manifest[["record_id", "split", "group_field"]],
            on="record_id",
            how="inner",
        )
        train = joined[joined["split"].eq("train")]
        validation = joined[joined["split"].eq("validation")]
        test = joined[joined["split"].eq("test")]
        group_field = str(test["group_field"].iloc[0])

        for feature_set in ("text_only", "engagement_only", "text_plus_engagement"):
            candidates = []
            for c_value in config["models"]["logistic_regression"]["c_grid"]:
                estimator = _estimator(config, feature_set, float(c_value))
                estimator.fit(train, train["harmonized_label"])
                prediction, probability = _predict(estimator, validation)
                metrics = classification_metrics(
                    validation["harmonized_label"],
                    prediction,
                    probability,
                    probability_scores=True,
                )
                candidates.append((estimator, float(c_value), metrics))
            selected, selected_c, validation_metrics = max(
                candidates,
                key=lambda item: item[2]["macro_f1"],
            )
            selected.fit(train, train["harmonized_label"])
            prediction, probability = _predict(selected, test)
            metrics = classification_metrics(
                test["harmonized_label"],
                prediction,
                probability,
                probability_scores=True,
            )
            predictions = test[
                [
                    "record_id",
                    "harmonized_label",
                    "text_unit",
                    "publisher_id",
                    "family_group",
                ]
            ].copy()
            predictions["prediction"] = prediction
            predictions["score_unreliable"] = probability
            predictions["bootstrap_group"] = test[group_field].astype(str).to_numpy()
            prediction_sets[(split_name, feature_set)] = predictions
            experiment_out = out_root / f"fakehealth_{split_name}_{feature_set}"
            experiment_out.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(experiment_out / "predictions.csv", index=False)
            bootstrap_intervals(
                predictions["harmonized_label"],
                predictions["prediction"],
                predictions["score_unreliable"],
                iterations=iterations,
                seed=seed,
                groups=predictions["bootstrap_group"],
                probability_scores=True,
            ).to_csv(experiment_out / "bootstrap_intervals.csv", index=False)
            summary_rows.append(
                {
                    "dataset": "fakehealth",
                    "split": split_name,
                    "feature_set": feature_set,
                    "selected_c": selected_c,
                    "train_records": len(train),
                    "validation_records": len(validation),
                    "test_records": len(test),
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    **metrics,
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        out_root / "engagement_ablation_summary.csv",
        index=False,
    )
    paired_rows = []
    for split_name in ("standard", "controlled"):
        for left_name, right_name in combinations(
            ("text_only", "engagement_only", "text_plus_engagement"),
            2,
        ):
            left = prediction_sets[(split_name, left_name)]
            right = prediction_sets[(split_name, right_name)]
            joined = left.merge(
                right[["record_id", "prediction", "score_unreliable"]],
                on="record_id",
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            difference = paired_bootstrap_difference(
                joined["harmonized_label"],
                joined["prediction_left"],
                joined["prediction_right"],
                joined["score_unreliable_left"],
                joined["score_unreliable_right"],
                iterations=iterations,
                seed=seed,
                groups=joined["bootstrap_group"],
            )
            difference.insert(0, "right_feature_set", right_name)
            difference.insert(0, "left_feature_set", left_name)
            difference.insert(0, "split", split_name)
            paired_rows.append(difference)
    pd.concat(paired_rows, ignore_index=True).to_csv(
        out_root / "paired_feature_differences.csv",
        index=False,
    )
    (out_root / "README.md").write_text(
        "# Engagement ablation\n\n"
        "FakeHealth-only complete-case comparison using unique engagement-ID counts. "
        "Counts are cross-sectional volume measures without exposure-time normalization; "
        "results are associational and do not identify diffusion or causal engagement effects.\n",
        encoding="utf-8",
    )
    print(f"Wrote FakeHealth engagement ablations to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from health_misinfo.evaluation.explanations import (
    compare_standard_controlled_explanations,
    explain_linear_tfidf,
)


@pytest.mark.parametrize(
    "classifier",
    [
        LogisticRegression(C=10, random_state=613),
        LinearSVC(C=1),
    ],
)
def test_linear_tfidf_contributions_reconstruct_unreliable_score(classifier):
    train_text = [
        "evidence clinical study",
        "clinical evidence research",
        "miracle hoax secret",
        "secret miracle conspiracy",
    ]
    labels = ["reliable", "reliable", "unreliable", "unreliable"]
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", classifier),
        ]
    ).fit(train_text, labels)

    summaries, contributions = explain_linear_tfidf(
        model,
        ["miracle clinical", "evidence secret"],
        record_ids=["a", "b"],
        top_per_direction=None,
    )

    np.testing.assert_allclose(
        summaries["decision_score_unreliable"],
        summaries["reconstructed_decision_score"],
        atol=1e-10,
    )
    assert summaries["reconstruction_error"].max() <= 1e-10
    assert set(contributions["direction"]) == {
        "toward_reliable",
        "toward_unreliable",
    }
    np.testing.assert_allclose(
        contributions["contribution"],
        contributions["tfidf_value"] * contributions["coefficient_unreliable"],
    )


def test_local_explanation_order_and_output_are_deterministic():
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(C=10, random_state=613)),
        ]
    ).fit(
        ["evidence study", "research evidence", "miracle hoax", "secret miracle"],
        ["reliable", "reliable", "unreliable", "unreliable"],
    )

    first_summary, first = explain_linear_tfidf(
        model,
        ["miracle evidence secret"],
        record_ids=["case"],
        top_per_direction=2,
    )
    second_summary, second = explain_linear_tfidf(
        model,
        ["miracle evidence secret"],
        record_ids=["case"],
        top_per_direction=2,
    )

    pd.testing.assert_frame_equal(first_summary, second_summary)
    pd.testing.assert_frame_equal(first, second)
    for _, direction in first.groupby("direction"):
        magnitudes = direction.sort_values("direction_rank")["contribution"].abs()
        assert magnitudes.is_monotonic_decreasing
    assert len(first) <= 4


def test_standard_controlled_stability_compares_same_record_and_signed_features():
    summaries = pd.DataFrame(
        [
            {
                "dataset": "sample",
                "model": "logistic_regression",
                "split_name": "standard",
                "record_id": "a",
                "model_prediction": "unreliable",
                "decision_score_unreliable": 1.0,
            },
            {
                "dataset": "sample",
                "model": "logistic_regression",
                "split_name": "controlled",
                "record_id": "a",
                "model_prediction": "reliable",
                "decision_score_unreliable": -0.2,
            },
        ]
    )
    contributions = pd.DataFrame(
        [
            {
                "dataset": "sample",
                "model": "logistic_regression",
                "split_name": "standard",
                "record_id": "a",
                "feature": "shared",
                "contribution": 2.0,
            },
            {
                "dataset": "sample",
                "model": "logistic_regression",
                "split_name": "standard",
                "record_id": "a",
                "feature": "standard_only",
                "contribution": -1.0,
            },
            {
                "dataset": "sample",
                "model": "logistic_regression",
                "split_name": "controlled",
                "record_id": "a",
                "feature": "shared",
                "contribution": -3.0,
            },
            {
                "dataset": "sample",
                "model": "logistic_regression",
                "split_name": "controlled",
                "record_id": "a",
                "feature": "controlled_only",
                "contribution": 1.0,
            },
        ]
    )

    result = compare_standard_controlled_explanations(
        contributions,
        summaries,
        top_k=2,
    ).iloc[0]

    assert result["feature_jaccard"] == pytest.approx(1 / 3)
    assert result["overlap_sign_agreement"] == 0
    assert result["top_contribution_cosine"] < 0
    assert not result["prediction_agreement"]


def test_explanation_rejects_nonpositive_feature_limit():
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(random_state=613)),
        ]
    ).fit(
        ["evidence study", "miracle hoax"],
        ["reliable", "unreliable"],
    )
    with pytest.raises(ValueError, match="top_per_direction"):
        explain_linear_tfidf(model, ["evidence"], top_per_direction=0)

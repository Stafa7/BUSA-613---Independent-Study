import numpy as np
import pandas as pd
import pytest

from health_misinfo.evaluation.narratives import (
    NarrativeThresholds,
    assigned_topic_probabilities,
    candidate_topic_table,
    discovery_eligibility,
    finalize_human_narratives,
    narrative_metadata_profile,
    narrative_performance_table,
    representative_review_template,
    topic_review_template,
)


def _candidate(records: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "sample",
                "topic_id": 0,
                "records": records,
                "mean_assignment_probability": 0.8,
                "minimum_topic_documents": 30,
                "eligible_topic_size": records >= 30,
                "machine_topic_name": "0_machine_words",
            }
        ]
    )


def _completed_topic_review(artifact_pattern: str = "none") -> pd.DataFrame:
    review = topic_review_template(_candidate())
    review.loc[0, "human_topic_coherence"] = "coherent"
    review.loc[0, "human_topic_stance"] = "endorsement"
    review.loc[0, "human_artifact_pattern"] = artifact_pattern
    review.loc[0, "human_review_confirmation"] = "human_reviewed"
    if artifact_pattern == "none":
        review.loc[0, "human_narrative_id"] = "narrative_01"
        review.loc[0, "human_narrative_label"] = "Human supplied label"
        review.loc[0, "human_label_source"] = "reviewer_authored"
    return review


def _completed_representatives(count: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "sample",
                "topic_id": 0,
                "record_id": f"r{index}",
                "representative_rank": index + 1,
                "human_supports_topic": "yes",
                "human_stance": "endorsement",
                "human_artifact_pattern": "none",
                "human_review_confirmation": "human_reviewed",
            }
            for index in range(count)
        ]
    )


def _expected_representatives(count: int = 5) -> pd.DataFrame:
    return _completed_representatives(count)[
        ["dataset", "topic_id", "record_id", "representative_rank"]
    ].copy()


def test_fit_and_topic_size_gates_use_protocol_boundaries():
    thresholds = NarrativeThresholds()
    below = pd.DataFrame({"harmonized_label": ["unreliable"] * 199})
    boundary = pd.DataFrame({"harmonized_label": ["unreliable"] * 200})

    assert not discovery_eligibility(below, thresholds)["eligible"]
    assert discovery_eligibility(boundary, thresholds)["eligible"]

    assignments = pd.DataFrame(
        {
            "dataset": ["sample"] * 60,
            "record_id": [f"r{index}" for index in range(60)],
            "topic_id": [0] * 30 + [1] * 29 + [-1],
            "assigned_probability": np.linspace(0.5, 0.9, 60),
        }
    )
    candidates = candidate_topic_table(assignments, thresholds)
    eligibility = candidates.set_index("topic_id")["eligible_topic_size"].to_dict()
    assert eligibility == {0: True, 1: False}


def test_validation_templates_leave_all_human_judgments_blank():
    candidates = _candidate()
    assignments = pd.DataFrame(
        {
            "dataset": ["sample"] * 10,
            "record_id": [f"r{index}" for index in range(10)],
            "topic_id": [0] * 10,
            "assigned_probability": np.linspace(1.0, 0.1, 10),
        }
    )
    documents = pd.DataFrame(
        {
            "dataset": ["sample"] * 10,
            "record_id": [f"r{index}" for index in range(10)],
            "title": [f"title {index}" for index in range(10)],
            "model_text": [f"text {index}" for index in range(10)],
        }
    )

    topics = topic_review_template(candidates)
    representatives = representative_review_template(
        candidates,
        assignments,
        documents,
    )

    assert topics["human_narrative_label"].eq("").all()
    assert topics["human_topic_coherence"].eq("").all()
    assert topics["human_label_source"].eq("").all()
    assert representatives["human_supports_topic"].eq("").all()
    assert representatives["human_stance"].eq("").all()
    assert representatives["human_artifact_pattern"].eq("").all()
    assert representatives["human_review_confirmation"].eq("").all()


def test_finalizer_stops_when_human_validation_is_blank():
    candidates = _candidate()
    result = finalize_human_narratives(
        candidates,
        topic_review_template(candidates),
        _completed_representatives(),
        NarrativeThresholds(),
        expected_representatives=_expected_representatives(),
    )

    assert result.status == "awaiting_human_validation"
    assert result.narratives.empty
    assert result.topic_mapping.empty
    assert any("human_topic_coherence" in issue for issue in result.issues)


def test_finalizer_reports_malformed_human_sheet_as_a_gate():
    result = finalize_human_narratives(
        _candidate(),
        pd.DataFrame({"topic_id": [0]}),
        pd.DataFrame({"record_id": ["r0"]}),
        NarrativeThresholds(),
        expected_representatives=_expected_representatives(),
    )

    assert result.status == "awaiting_human_validation"
    assert result.narratives.empty
    assert any("missing required columns" in issue for issue in result.issues)


def test_finalizer_requires_explicit_single_reviewer_confirmation():
    representatives = _completed_representatives()
    representatives["human_review_confirmation"] = ""

    result = finalize_human_narratives(
        _candidate(),
        _completed_topic_review(),
        representatives,
        NarrativeThresholds(),
        expected_representatives=_expected_representatives(),
    )

    assert result.status == "awaiting_human_validation"
    assert result.topic_mapping.empty
    assert any(
        "fewer than 5 human-reviewed representatives" in issue
        for issue in result.issues
    )


def test_finalizer_rejects_substituted_representative_record():
    representatives = _completed_representatives()
    representatives.loc[0, "record_id"] = "unselected_record"

    result = finalize_human_narratives(
        _candidate(),
        _completed_topic_review(),
        representatives,
        NarrativeThresholds(),
        expected_representatives=_expected_representatives(),
    )

    assert result.status == "awaiting_human_validation"
    assert any(
        "representative review roster differs" in issue
        for issue in result.issues
    )


@pytest.mark.parametrize("invalid_topic_id", ["", "not_a_topic", 0.5])
def test_finalizer_gates_malformed_human_topic_ids(invalid_topic_id):
    topic_review = _completed_topic_review()
    topic_review["topic_id"] = topic_review["topic_id"].astype("object")
    topic_review.loc[0, "topic_id"] = invalid_topic_id

    result = finalize_human_narratives(
        _candidate(),
        topic_review,
        _completed_representatives(),
        NarrativeThresholds(),
        expected_representatives=_expected_representatives(),
    )

    assert result.status == "awaiting_human_validation"
    assert any(
        "blank or non-integer topic_id" in issue
        for issue in result.issues
    )


def test_finalizer_retains_only_human_labeled_topic_with_five_supporting_examples():
    result = finalize_human_narratives(
        _candidate(),
        _completed_topic_review("none"),
        _completed_representatives(),
        NarrativeThresholds(),
        expected_representatives=_expected_representatives(),
    )

    assert result.status == "completed"
    assert result.issues == []
    assert result.narratives.iloc[0]["retained"]
    assert result.narratives.iloc[0]["representatives_supporting"] == 5
    assert result.topic_mapping.iloc[0]["candidate_narrative_label"] == (
        "Human supplied label"
    )


def test_publisher_dataset_or_generic_pattern_is_not_retained_as_narrative():
    result = finalize_human_narratives(
        _candidate(),
        _completed_topic_review(artifact_pattern="publisher"),
        _completed_representatives(),
        NarrativeThresholds(),
        expected_representatives=_expected_representatives(),
    )

    assert result.status == "completed"
    assert result.narratives.empty
    assert result.topic_mapping.empty
    assert result.topic_outcomes.iloc[0]["retention_reason"] == (
        "rejected_publisher_artifact_pattern"
    )


def test_machine_topic_name_cannot_be_substituted_for_human_label():
    topic_review = _completed_topic_review("none")
    topic_review.loc[0, "human_narrative_label"] = "0 machine words"

    result = finalize_human_narratives(
        _candidate(),
        topic_review,
        _completed_representatives(),
        NarrativeThresholds(),
        expected_representatives=_expected_representatives(),
    )

    assert result.status == "awaiting_human_validation"
    assert result.topic_mapping.empty
    assert any("unchanged machine topic name" in issue for issue in result.issues)


def test_finalizer_handles_dataset_with_no_size_eligible_topics():
    candidates = _candidate(records=29)

    result = finalize_human_narratives(
        candidates,
        topic_review_template(candidates),
        pd.DataFrame(
            columns=["dataset", "topic_id", "record_id", "representative_rank"]
        ),
        NarrativeThresholds(),
        expected_representatives=pd.DataFrame(
            columns=["dataset", "topic_id", "record_id", "representative_rank"]
        ),
    )

    assert result.status == "completed"
    assert result.issues == []
    assert result.narratives.empty
    assert result.topic_mapping.empty
    assert result.topic_outcomes.empty


def test_narrative_performance_gate_requires_twenty_records_and_five_per_class():
    mapping = pd.DataFrame(
        [
            {
                "dataset": "sample",
                "topic_id": 0,
                "final_narrative_id": "n0",
                "candidate_narrative_label": "Narrative zero",
            },
            {
                "dataset": "sample",
                "topic_id": 1,
                "final_narrative_id": "n1",
                "candidate_narrative_label": "Narrative one",
            },
        ]
    )
    records = [f"a{index}" for index in range(20)] + [
        f"b{index}" for index in range(19)
    ]
    assignments = pd.DataFrame(
        {
            "dataset": ["sample"] * 39,
            "record_id": records,
            "topic_id": [0] * 20 + [1] * 19,
        }
    )
    truth = (
        ["reliable"] * 5
        + ["unreliable"] * 15
        + ["reliable"] * 5
        + ["unreliable"] * 14
    )
    predictions = pd.DataFrame(
        {
            "dataset": ["sample"] * 39,
            "record_id": records,
            "harmonized_label": truth,
            "prediction": truth,
            "score_unreliable": [
                0.1 if label == "reliable" else 0.9
                for label in truth
            ],
            "split_name": ["standard"] * 39,
            "model": ["logistic_regression"] * 39,
        }
    )

    result = narrative_performance_table(
        predictions,
        assignments,
        mapping,
        NarrativeThresholds(),
    ).set_index("final_narrative_id")

    assert result.loc["n0", "performance_eligible"]
    assert result.loc["n0", "macro_f1"] == 1
    assert not result.loc["n1", "performance_eligible"]
    assert pd.isna(result.loc["n1", "macro_f1"])


def test_metadata_gate_requires_thirty_complete_records_per_narrative():
    mapping = pd.DataFrame(
        [
            {
                "dataset": "sample",
                "topic_id": 0,
                "final_narrative_id": "n0",
                "candidate_narrative_label": "Narrative zero",
            },
            {
                "dataset": "sample",
                "topic_id": 1,
                "final_narrative_id": "n1",
                "candidate_narrative_label": "Narrative one",
            },
        ]
    )
    records = [f"a{index}" for index in range(30)] + [
        f"b{index}" for index in range(29)
    ]
    assignments = pd.DataFrame(
        {
            "dataset": ["sample"] * 59,
            "record_id": records,
            "topic_id": [0] * 30 + [1] * 29,
        }
    )
    data = pd.DataFrame(
        {
            "dataset": ["sample"] * 59,
            "record_id": records,
            "engagement_available": [True] * 59,
            "tweet_count": np.arange(59),
        }
    )

    result = narrative_metadata_profile(
        data,
        assignments,
        mapping,
        ["tweet_count"],
        NarrativeThresholds(),
        availability_column="engagement_available",
    ).set_index("final_narrative_id")

    assert result.loc["n0", "metadata_comparison_eligible"]
    assert not result.loc["n1", "metadata_comparison_eligible"]
    assert pd.isna(result.loc["n1", "mean"])


def test_assigned_probability_handles_topic_matrix_and_outliers():
    probabilities = np.array([[0.8, 0.1], [0.2, 0.7], [0.4, 0.3]])
    assigned = assigned_topic_probabilities([0, 1, -1], probabilities)
    np.testing.assert_allclose(assigned, [0.8, 0.7, 0.6])


def test_thresholds_reject_nonpositive_values():
    with pytest.raises(ValueError, match="minimum_topic_documents"):
        NarrativeThresholds(minimum_topic_documents=0)

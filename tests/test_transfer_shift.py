import math

import numpy as np
import pandas as pd

from health_misinfo.evaluation.transfer_shift import (
    SOURCE_COHORT,
    TARGET_COHORT,
    baseline_comparator_spec,
    build_transfer_shift_artifacts,
    categorical_shift_diagnostics,
    document_length_diagnostics,
    jensen_shannon_divergence,
    render_degradation_markdown,
    topic_proxy_diagnostics,
    total_variation_distance,
    transfer_degradation_rows,
    vocabulary_shift_diagnostics,
)


def test_distribution_distances_have_expected_bounds_and_symmetry():
    identical = [3, 1]
    left = [1, 0]
    right = [0, 1]

    assert jensen_shannon_divergence(identical, identical) == 0
    assert total_variation_distance(identical, identical) == 0
    assert math.isclose(jensen_shannon_divergence(left, right), 1.0)
    assert math.isclose(total_variation_distance(left, right), 1.0)
    assert math.isclose(
        jensen_shannon_divergence([2, 1], [1, 2]),
        jensen_shannon_divergence([1, 2], [2, 1]),
    )


def test_categorical_shift_aligns_categories_and_reports_prevalence():
    distribution, summary = categorical_shift_diagnostics(
        ["reliable", "reliable", "unreliable", None],
        ["unreliable", "unreliable", "reliable", "reliable"],
        field="harmonized_label",
    )

    assert set(distribution["category"]) == {
        "(missing)",
        "reliable",
        "unreliable",
    }
    assert math.isclose(summary["source_unreliable_prevalence"], 0.25)
    assert math.isclose(summary["target_unreliable_prevalence"], 0.50)
    assert math.isclose(
        summary["target_minus_source_unreliable_prevalence"], 0.25
    )
    assert 0 <= summary["total_variation_distance"] <= 1
    assert 0 <= summary["jensen_shannon_divergence_bits"] <= 1


def test_vocabulary_shift_reports_type_overlap_token_coverage_and_terms():
    terms, summary = vocabulary_shift_diagnostics(
        ["apple apple care", "apple health"],
        ["banana banana risk", "banana health"],
        max_features=100,
        minimum_term_count=1,
        top_terms_per_direction=2,
    )

    assert summary["status"] == "ok"
    assert math.isclose(summary["active_type_jaccard"], 1 / 5)
    assert math.isclose(
        summary["target_token_coverage_by_source_active_types"], 1 / 5
    )
    assert summary[
        "jensen_shannon_divergence_bits_retained_vocabulary"
    ] > 0
    assert set(terms["enrichment"]) == {
        "source_enriched",
        "target_enriched",
    }
    assert "apple" in set(
        terms.loc[terms["enrichment"].eq("source_enriched"), "term"]
    )
    assert "banana" in set(
        terms.loc[terms["enrichment"].eq("target_enriched"), "term"]
    )


def test_document_length_shift_uses_the_actual_text_representation():
    profiles, shift = document_length_diagnostics(
        ["one two", "three four"],
        ["one two three four", "five six seven eight"],
    )

    assert set(profiles["cohort"]) == {SOURCE_COHORT, TARGET_COHORT}
    assert math.isclose(shift["target_to_source_mean_ratio"], 2.0)
    assert math.isclose(shift["kolmogorov_smirnov_statistic"], 1.0)


def test_topic_proxy_is_deterministic_and_marks_itself_as_a_proxy():
    source = [
        "heart cardiac artery treatment",
        "cardiac heart surgery artery",
        "heart disease cardiac care",
    ]
    target = [
        "vaccine immune dose trial",
        "immune vaccine booster dose",
        "vaccine trial immune response",
    ]
    first = topic_proxy_diagnostics(
        source,
        target,
        random_seed=7,
        requested_topics=2,
        max_features=100,
        top_terms_per_topic=3,
    )
    second = topic_proxy_diagnostics(
        source,
        target,
        random_seed=7,
        requested_topics=2,
        max_features=100,
        top_terms_per_topic=3,
    )

    first_terms, first_distribution, assignments, first_summary = first
    _, second_distribution, _, second_summary = second
    assert first_summary["method"] == "pooled_tfidf_nmf"
    assert first_summary["fitted_topics"] == 2
    assert len(assignments) == 6
    assert {"heart", "vaccine"} <= set(first_terms["term"])
    pd.testing.assert_frame_equal(first_distribution, second_distribution)
    assert (
        first_summary[
            "jensen_shannon_divergence_bits_mean_topic_weights"
        ]
        == second_summary[
            "jensen_shannon_divergence_bits_mean_topic_weights"
        ]
    )
    for cohort in (SOURCE_COHORT, TARGET_COHORT):
        mixture = first_distribution[
            first_distribution["cohort"].eq(cohort)
        ]["mean_weight_share"]
        assert np.isclose(mixture.sum(), 1.0)


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "record_id",
            "harmonized_label",
            "text_unit",
            "publisher_id",
            "source_domain",
            "model_text",
            "title",
        ],
    )


def test_transfer_artifact_builder_records_exclusion_and_all_shift_facets():
    source = _frame(
        [
            (
                "s1",
                "reliable",
                "article",
                "publisher_a",
                "source-a.test",
                "heart cardiac artery health",
                "Heart care",
            ),
            (
                "s2",
                "unreliable",
                "article",
                "publisher_b",
                "source-b.test",
                "miracle cure cardiac claim",
                "Miracle cure",
            ),
            (
                "s3",
                "reliable",
                "article",
                "publisher_a",
                "source-a.test",
                "clinical heart treatment evidence",
                "Clinical evidence",
            ),
            (
                "s4",
                "unreliable",
                "article",
                "publisher_b",
                "source-b.test",
                "unsupported cure health claim",
                "Unsupported claim",
            ),
        ]
    )
    target_before = _frame(
        [
            (
                "t0",
                "reliable",
                "story",
                "publisher_a",
                "source-a.test",
                "overlapping heart story",
                "Overlap",
            ),
            (
                "t1",
                "unreliable",
                "story",
                "publisher_c",
                "target-c.test",
                "vaccine immune booster dose",
                "Vaccine dose",
            ),
            (
                "t2",
                "reliable",
                "news_release",
                "publisher_d",
                "target-d.test",
                "immune vaccine clinical trial",
                "Vaccine trial",
            ),
        ]
    )
    target_after = target_before[target_before["record_id"].ne("t0")].copy()

    artifacts = build_transfer_shift_artifacts(
        source,
        target_before,
        target_after,
        source_dataset="source",
        target_dataset="target",
        exclusion_source_domains={"source-a.test", "source-b.test"},
        random_seed=11,
        vocabulary_max_features=100,
        vocabulary_minimum_term_count=1,
        topic_requested_components=2,
        topic_max_features=100,
        topic_top_terms=3,
    )

    expected_tables = {
        "cohort_profiles",
        "domain_exclusion_summary",
        "vocabulary_summary",
        "document_length_shift",
        "categorical_shift_summary",
        "topic_proxy_summary",
        "shift_summary",
    }
    assert expected_tables <= set(artifacts.tables)
    exclusion = artifacts.tables["domain_exclusion_summary"].iloc[0]
    assert exclusion["target_records_removed"] == 1
    assert exclusion["remaining_target_records_on_exclusion_domains"] == 0
    dimensions = set(artifacts.tables["shift_summary"]["dimension"])
    assert {
        "vocabulary",
        "document_length",
        "composition",
        "topic_proxy",
        "domain_exclusion",
    } <= dimensions
    assert artifacts.metadata["interpretation"]["causal_attribution"] is False
    assert "not validated narratives" in artifacts.markdown


def _baseline_summary(experiment_id, **metrics):
    return pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "n": 3,
                "macro_f1": metrics.get("macro_f1", 0.9),
                "unreliable_f1": metrics.get("unreliable_f1", 0.8),
                "unreliable_pr_auc": metrics.get(
                    "unreliable_pr_auc", 0.85
                ),
            }
        ]
    )


def _baseline_config(spec):
    return {
        "experiment_id": spec.baseline_experiment_id,
        "dataset": spec.comparator_dataset,
        "split_name": "standard",
        "model": spec.baseline_model,
        "text_column": spec.text_column,
    }


def _prediction_frame():
    return pd.DataFrame(
        {
            "record_id": ["t1", "t2"],
            "harmonized_label": ["reliable", "unreliable"],
            "prediction": ["reliable", "reliable"],
            "score_unreliable": [0.9, 0.4],
        }
    )


def test_comparator_resolution_is_exact_and_rejects_title_svm_mismatch():
    full = baseline_comparator_spec(
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="full_text",
        transfer_model="linear_svm",
        comparator_kind="target_in_domain_same_records",
    )
    title_logistic = baseline_comparator_spec(
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="title_only",
        transfer_model="logistic_regression",
        comparator_kind="source_in_domain_standard_test",
    )
    title_svm = baseline_comparator_spec(
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="title_only",
        transfer_model="linear_svm",
        comparator_kind="target_in_domain_same_records",
    )

    assert (
        full.baseline_experiment_id
        == "fakehealth_standard_linear_svm_model_text"
    )
    assert (
        title_logistic.baseline_experiment_id
        == "coaid_standard_title_only_logistic_title"
    )
    assert title_logistic.baseline_model == "title_only_logistic"
    assert title_svm.baseline_experiment_id is None
    assert (
        title_svm.unavailable_reason
        == "same_family_title_only_baseline_not_in_baseline_matrix"
    )


def test_target_comparator_recomputes_metrics_on_exact_transfer_records():
    comparator = baseline_comparator_spec(
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="full_text",
        transfer_model="logistic_regression",
        comparator_kind="target_in_domain_same_records",
    )
    baseline_predictions = pd.DataFrame(
        {
            "record_id": ["t0", "t1", "t2"],
            "harmonized_label": [
                "reliable",
                "reliable",
                "unreliable",
            ],
            "prediction": ["reliable", "reliable", "unreliable"],
            "score_unreliable": [0.2, 0.1, 0.8],
        }
    )
    degradation = transfer_degradation_rows(
        transfer_experiment_id="coaid_to_fakehealth_full_text_logistic",
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="full_text",
        transfer_model="logistic_regression",
        transfer_predictions=_prediction_frame(),
        comparator=comparator,
        baseline_summary=_baseline_summary(
            comparator.baseline_experiment_id,
            macro_f1=0.91,
        ),
        baseline_config=_baseline_config(comparator),
        baseline_predictions=baseline_predictions,
    )

    macro = degradation[degradation["metric"].eq("macro_f1")].iloc[0]
    pr_auc = degradation[
        degradation["metric"].eq("unreliable_pr_auc")
    ].iloc[0]
    assert set(degradation["comparison_status"]) == {"available"}
    assert macro["comparison_records"] == 2
    assert macro["baseline_full_standard_value"] == 0.91
    assert macro["in_domain_comparator_value"] == 1.0
    assert math.isclose(macro["transfer_value"], 1 / 3)
    assert math.isclose(
        macro["absolute_delta_transfer_minus_in_domain"], -2 / 3
    )
    assert pr_auc["in_domain_comparator_value"] == 1.0
    assert pr_auc["transfer_value"] == 0.5


def test_source_comparator_uses_summary_and_marks_zero_relative_delta_unavailable():
    comparator = baseline_comparator_spec(
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="full_text",
        transfer_model="linear_svm",
        comparator_kind="source_in_domain_standard_test",
    )
    degradation = transfer_degradation_rows(
        transfer_experiment_id="coaid_to_fakehealth_full_text_svm",
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="full_text",
        transfer_model="linear_svm",
        transfer_predictions=_prediction_frame(),
        comparator=comparator,
        baseline_summary=_baseline_summary(
            comparator.baseline_experiment_id,
            macro_f1=0.8,
            unreliable_f1=0.0,
            unreliable_pr_auc=0.9,
        ),
        baseline_config=_baseline_config(comparator),
        baseline_predictions=None,
        baseline_predictions_unavailable_reason=(
            "baseline_prediction_artifact_missing"
        ),
    )

    macro = degradation[degradation["metric"].eq("macro_f1")].iloc[0]
    unreliable_f1 = degradation[
        degradation["metric"].eq("unreliable_f1")
    ].iloc[0]
    assert macro["comparison_status"] == "available"
    assert macro["same_evaluation_records"] == np.False_
    assert macro["in_domain_comparator_value"] == 0.8
    assert unreliable_f1["comparison_status"] == "available"
    assert unreliable_f1["relative_delta_status"] == "unavailable"
    assert (
        unreliable_f1["relative_delta_unavailable_reason"]
        == "zero_in_domain_comparator_metric"
    )


def test_degradation_records_unavailable_reason_instead_of_model_or_record_mismatch():
    title_svm = baseline_comparator_spec(
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="title_only",
        transfer_model="linear_svm",
        comparator_kind="target_in_domain_same_records",
    )
    unavailable = transfer_degradation_rows(
        transfer_experiment_id="title_svm_transfer",
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="title_only",
        transfer_model="linear_svm",
        transfer_predictions=_prediction_frame(),
        comparator=title_svm,
        baseline_summary=pd.DataFrame(),
        baseline_config=None,
        baseline_predictions=None,
    )
    assert set(unavailable["comparison_status"]) == {"unavailable"}
    assert set(unavailable["unavailable_reason"]) == {
        "same_family_title_only_baseline_not_in_baseline_matrix"
    }

    target = baseline_comparator_spec(
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="full_text",
        transfer_model="logistic_regression",
        comparator_kind="target_in_domain_same_records",
    )
    uncovered = transfer_degradation_rows(
        transfer_experiment_id="full_text_transfer",
        source_dataset="coaid",
        target_dataset="fakehealth",
        text_representation="full_text",
        transfer_model="logistic_regression",
        transfer_predictions=_prediction_frame(),
        comparator=target,
        baseline_summary=_baseline_summary(target.baseline_experiment_id),
        baseline_config=_baseline_config(target),
        baseline_predictions=pd.DataFrame(
            {
                "record_id": ["t1"],
                "harmonized_label": ["reliable"],
                "prediction": ["reliable"],
                "score_unreliable": [0.1],
            }
        ),
    )
    assert set(uncovered["unavailable_reason"]) == {
        "baseline_predictions_do_not_cover_transfer_record_set"
    }
    markdown = render_degradation_markdown(
        pd.concat([unavailable, uncovered], ignore_index=True)
    )
    assert "unavailable:" in markdown

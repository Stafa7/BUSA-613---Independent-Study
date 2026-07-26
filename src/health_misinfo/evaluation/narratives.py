from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from health_misinfo.evaluation.metrics import classification_metrics


@dataclass(frozen=True)
class NarrativeThresholds:
    minimum_fit_documents: int = 200
    minimum_topic_documents: int = 30
    minimum_representatives: int = 5
    minimum_performance_records: int = 20
    minimum_performance_per_class: int = 5
    minimum_metadata_complete_per_group: int = 30

    def __post_init__(self) -> None:
        for field, value in self.__dict__.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")


@dataclass
class NarrativeFinalization:
    status: str
    issues: list[str]
    narratives: pd.DataFrame
    topic_mapping: pd.DataFrame
    topic_outcomes: pd.DataFrame


def discovery_eligibility(
    frame: pd.DataFrame,
    thresholds: NarrativeThresholds,
    label_column: str = "harmonized_label",
) -> dict[str, object]:
    unreliable_records = int(frame[label_column].eq("unreliable").sum())
    return {
        "records": len(frame),
        "unreliable_records": unreliable_records,
        "minimum_fit_documents": thresholds.minimum_fit_documents,
        "eligible": unreliable_records >= thresholds.minimum_fit_documents,
        "gate_reason": (
            "eligible"
            if unreliable_records >= thresholds.minimum_fit_documents
            else "fewer_than_minimum_unreliable_documents"
        ),
    }


def assigned_topic_probabilities(topics, probabilities) -> np.ndarray:
    """Return each document's assigned-topic probability from BERTopic outputs."""

    topic_values = np.asarray(topics, dtype=int)
    if probabilities is None:
        return np.full(len(topic_values), np.nan)
    probability_values = np.asarray(probabilities, dtype=float)
    if probability_values.ndim == 1:
        if len(probability_values) != len(topic_values):
            raise ValueError("topics and probabilities must have the same length")
        return probability_values
    if probability_values.ndim != 2 or probability_values.shape[0] != len(topic_values):
        raise ValueError("probabilities must have one row per topic assignment")
    assigned = np.full(len(topic_values), np.nan)
    for index, topic in enumerate(topic_values):
        if topic == -1:
            assigned[index] = 1.0 - float(np.max(probability_values[index]))
        elif 0 <= topic < probability_values.shape[1]:
            assigned[index] = float(probability_values[index, topic])
        else:
            assigned[index] = float(np.max(probability_values[index]))
    return assigned


def candidate_topic_table(
    assignments: pd.DataFrame,
    thresholds: NarrativeThresholds,
) -> pd.DataFrame:
    required = {"dataset", "record_id", "topic_id"}
    if missing := required.difference(assignments.columns):
        raise ValueError(f"assignments are missing required columns: {sorted(missing)}")
    candidates = (
        assignments[assignments["topic_id"].ne(-1)]
        .groupby(["dataset", "topic_id"])
        .agg(
            records=("record_id", "nunique"),
            mean_assignment_probability=("assigned_probability", "mean"),
        )
        .reset_index()
    )
    candidates["minimum_topic_documents"] = thresholds.minimum_topic_documents
    candidates["eligible_topic_size"] = (
        candidates["records"] >= thresholds.minimum_topic_documents
    )
    return candidates.sort_values(
        ["dataset", "records", "topic_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def topic_review_template(candidates: pd.DataFrame) -> pd.DataFrame:
    eligible = candidates[candidates["eligible_topic_size"]].copy()
    for column in (
        "human_topic_coherence",
        "human_topic_stance",
        "human_artifact_pattern",
        "human_narrative_id",
        "human_narrative_label",
        "human_label_source",
        "human_notes",
        "human_review_confirmation",
    ):
        eligible[column] = ""
    return eligible


def representative_review_template(
    candidates: pd.DataFrame,
    assignments: pd.DataFrame,
    documents: pd.DataFrame,
    representatives_per_topic: int = 10,
) -> pd.DataFrame:
    if representatives_per_topic <= 0:
        raise ValueError("representatives_per_topic must be positive")
    eligible_keys = candidates.loc[
        candidates["eligible_topic_size"],
        ["dataset", "topic_id"],
    ]
    selected = assignments.merge(
        eligible_keys,
        on=["dataset", "topic_id"],
        how="inner",
        validate="many_to_one",
    )
    selected = selected.sort_values(
        ["dataset", "topic_id", "assigned_probability", "record_id"],
        ascending=[True, True, False, True],
    )
    selected["representative_rank"] = (
        selected.groupby(["dataset", "topic_id"]).cumcount() + 1
    )
    selected = selected[
        selected["representative_rank"] <= representatives_per_topic
    ].copy()
    document_columns = [
        column
        for column in (
            "dataset",
            "record_id",
            "title",
            "model_text",
            "source_name",
            "publisher_id",
        )
        if column in documents.columns
    ]
    selected = selected.merge(
        documents[document_columns].drop_duplicates(["dataset", "record_id"]),
        on=["dataset", "record_id"],
        how="left",
        validate="one_to_one",
    )
    if "model_text" in selected:
        selected["text_excerpt"] = (
            selected["model_text"].fillna("").astype(str).str.slice(0, 800)
        )
        selected = selected.drop(columns="model_text")
    for column in (
        "human_supports_topic",
        "human_stance",
        "human_artifact_pattern",
        "human_notes",
        "human_review_confirmation",
    ):
        selected[column] = ""
    return selected


def _normalized_text(value: object) -> str:
    return str(value).strip().lower() if not pd.isna(value) else ""


def _descriptor_signature(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _normalized_text(value)))


def finalize_human_narratives(
    candidates: pd.DataFrame,
    topic_reviews: pd.DataFrame,
    representative_reviews: pd.DataFrame,
    thresholds: NarrativeThresholds,
    *,
    expected_representatives: pd.DataFrame,
    required_confirmation: str = "human_reviewed",
) -> NarrativeFinalization:
    """Apply single-reviewer protocol gates without inferring human judgments."""

    normalized_confirmation = _normalized_text(required_confirmation)
    if not normalized_confirmation:
        raise ValueError("required_confirmation must not be blank")

    empty_narratives = pd.DataFrame(
        columns=[
            "dataset",
            "final_narrative_id",
            "candidate_narrative_label",
            "records",
            "topics",
            "representatives_reviewed",
            "representatives_supporting",
            "retained",
            "retention_reason",
        ]
    )
    empty_mapping = pd.DataFrame(
        columns=[
            "dataset",
            "topic_id",
            "final_narrative_id",
            "candidate_narrative_label",
        ]
    )
    issues: list[str] = []
    key_columns = ["dataset", "topic_id"]
    required_columns = (
        (
            "candidate topic table",
            candidates,
            {*key_columns, "records", "eligible_topic_size"},
        ),
        ("topic review", topic_reviews, set(key_columns)),
        (
            "representative review",
            representative_reviews,
            {*key_columns, "record_id", "representative_rank"},
        ),
        (
            "expected representative roster",
            expected_representatives,
            {*key_columns, "record_id", "representative_rank"},
        ),
    )
    for name, frame, required in required_columns:
        if missing := required.difference(frame.columns):
            issues.append(f"{name} is missing required columns: {sorted(missing)}")
    if issues:
        return NarrativeFinalization(
            "awaiting_human_validation",
            issues,
            empty_narratives,
            empty_mapping,
            pd.DataFrame(),
        )

    normalized_frames: dict[str, pd.DataFrame] = {}
    for name, frame in (
        ("candidate topic table", candidates),
        ("topic review", topic_reviews),
        ("representative review", representative_reviews),
        ("expected representative roster", expected_representatives),
    ):
        normalized = frame.copy()
        normalized["dataset"] = (
            normalized["dataset"].astype("string").fillna("").str.strip()
        )
        numeric_topic_ids = pd.to_numeric(
            normalized["topic_id"],
            errors="coerce",
        )
        invalid_topic_ids = numeric_topic_ids.isna() | (
            numeric_topic_ids.notna() & numeric_topic_ids.mod(1).ne(0)
        )
        if invalid_topic_ids.any():
            issues.append(
                f"{name} contains blank or non-integer topic_id values"
            )
            continue
        normalized["topic_id"] = numeric_topic_ids.astype("int64")
        if "record_id" in normalized:
            normalized["record_id"] = (
                normalized["record_id"].astype("string").fillna("").str.strip()
            )
            if normalized["record_id"].eq("").any():
                issues.append(f"{name} contains blank record_id values")
            numeric_ranks = pd.to_numeric(
                normalized["representative_rank"],
                errors="coerce",
            )
            invalid_ranks = numeric_ranks.isna() | (
                numeric_ranks.notna() & numeric_ranks.mod(1).ne(0)
            )
            if invalid_ranks.any():
                issues.append(
                    f"{name} contains blank or non-integer representative_rank values"
                )
                continue
            normalized["representative_rank"] = numeric_ranks.astype("int64")
        normalized_frames[name] = normalized
    if issues:
        return NarrativeFinalization(
            "awaiting_human_validation",
            issues,
            empty_narratives,
            empty_mapping,
            pd.DataFrame(),
        )

    candidates = normalized_frames["candidate topic table"]
    topic_reviews = normalized_frames["topic review"]
    representative_reviews = normalized_frames["representative review"]
    expected_representatives = normalized_frames[
        "expected representative roster"
    ]

    eligible_flag = candidates["eligible_topic_size"]
    if not pd.api.types.is_bool_dtype(eligible_flag):
        eligible_flag = eligible_flag.map(_normalized_text).isin({"true", "yes", "1"})
    eligible = candidates[eligible_flag].copy()
    if topic_reviews.duplicated(key_columns).any():
        issues.append("topic review contains duplicate dataset/topic_id rows")
    if representative_reviews.duplicated([*key_columns, "record_id"]).any():
        issues.append("representative review contains duplicate document rows")
    if expected_representatives.duplicated([*key_columns, "record_id"]).any():
        issues.append("expected representative roster contains duplicate document rows")

    def context_matches(
        submitted: pd.DataFrame,
        expected: pd.DataFrame,
        frozen_columns: list[str],
        sort_columns: list[str],
    ) -> bool:
        if missing := set(frozen_columns).difference(submitted.columns):
            issues.append(
                "review sheet is missing frozen context columns: "
                f"{sorted(missing)}"
            )
            return False
        submitted_context = submitted[frozen_columns].copy()
        expected_context = expected[frozen_columns].copy()
        for column in frozen_columns:
            submitted_context[column] = (
                submitted_context[column].astype("string").fillna("")
            )
            expected_context[column] = (
                expected_context[column].astype("string").fillna("")
            )
        submitted_context = submitted_context.sort_values(
            sort_columns,
            kind="mergesort",
        ).reset_index(drop=True)
        expected_context = expected_context.sort_values(
            sort_columns,
            kind="mergesort",
        ).reset_index(drop=True)
        return submitted_context.equals(expected_context)

    eligible_topic_keys = set(
        eligible[key_columns].itertuples(index=False, name=None)
    )
    submitted_topic_keys = set(
        topic_reviews[key_columns].itertuples(index=False, name=None)
    )
    if submitted_topic_keys != eligible_topic_keys:
        issues.append(
            "topic review roster differs from the generated eligible-topic roster"
        )
    eligible_context_columns = list(candidates.columns)
    expected_topic_context = eligible[eligible_context_columns]
    if not context_matches(
        topic_reviews,
        expected_topic_context,
        eligible_context_columns,
        key_columns,
    ):
        issues.append(
            "topic review context differs from the generated candidate-topic sheet"
        )

    representative_key_columns = [
        *key_columns,
        "record_id",
        "representative_rank",
    ]
    expected_representative_keys = set(
        expected_representatives[representative_key_columns].itertuples(
            index=False,
            name=None,
        )
    )
    submitted_representative_keys = set(
        representative_reviews[representative_key_columns].itertuples(
            index=False,
            name=None,
        )
    )
    if submitted_representative_keys != expected_representative_keys:
        issues.append(
            "representative review roster differs from the generated document roster"
        )
    representative_context_columns = [
        column
        for column in expected_representatives.columns
        if not column.startswith("human_")
    ]
    if not context_matches(
        representative_reviews,
        expected_representatives,
        representative_context_columns,
        representative_key_columns,
    ):
        issues.append(
            "representative review context differs from the generated document sheet"
        )
    if issues:
        return NarrativeFinalization(
            "awaiting_human_validation",
            issues,
            empty_narratives,
            empty_mapping,
            pd.DataFrame(),
        )

    reviews = eligible[key_columns].merge(
        topic_reviews,
        on=key_columns,
        how="left",
        suffixes=("", "_review"),
        validate="one_to_one",
    )
    allowed_coherence = {"coherent", "mixed", "incoherent"}
    allowed_support = {"yes", "no", "unclear"}
    allowed_stance = {
        "endorsement",
        "discussion",
        "debunking",
        "mixed",
        "unclear",
    }
    allowed_artifact_patterns = {
        "none",
        "publisher",
        "dataset",
        "generic",
        "mixed",
        "unclear",
    }

    review_rows: list[dict[str, object]] = []
    for candidate in eligible.itertuples(index=False):
        dataset = str(candidate.dataset)
        topic_id = int(candidate.topic_id)
        matching = reviews[
            reviews["dataset"].astype(str).eq(dataset)
            & reviews["topic_id"].astype(int).eq(topic_id)
        ]
        if matching.empty:
            issues.append(f"{dataset}/topic_{topic_id}: missing topic review")
            continue
        review = matching.iloc[0]
        coherence = _normalized_text(review.get("human_topic_coherence", ""))
        topic_stance = _normalized_text(review.get("human_topic_stance", ""))
        artifact_pattern = _normalized_text(
            review.get("human_artifact_pattern", "")
        )
        topic_confirmation = _normalized_text(
            review.get("human_review_confirmation", "")
        )
        if coherence not in allowed_coherence:
            issues.append(
                f"{dataset}/topic_{topic_id}: human_topic_coherence "
                "must be coherent, mixed, or incoherent"
            )
        if topic_stance not in allowed_stance:
            issues.append(
                f"{dataset}/topic_{topic_id}: human_topic_stance must be "
                "endorsement, discussion, debunking, mixed, or unclear"
            )
        if artifact_pattern not in allowed_artifact_patterns:
            issues.append(
                f"{dataset}/topic_{topic_id}: human_artifact_pattern must be "
                "none, publisher, dataset, generic, mixed, or unclear"
            )
        if topic_confirmation != normalized_confirmation:
            issues.append(
                f"{dataset}/topic_{topic_id}: human_review_confirmation must be "
                f"{required_confirmation!r}"
            )
        narrative_id = str(review.get("human_narrative_id", "")).strip()
        narrative_label = str(review.get("human_narrative_label", "")).strip()
        label_source = _normalized_text(review.get("human_label_source", ""))
        if coherence == "coherent" and artifact_pattern == "none":
            if not narrative_id:
                issues.append(
                    f"{dataset}/topic_{topic_id}: coherent topic lacks "
                    "human_narrative_id"
                )
            if not narrative_label:
                issues.append(
                    f"{dataset}/topic_{topic_id}: coherent topic lacks a "
                    "human narrative label"
                )
            if label_source != "reviewer_authored":
                issues.append(
                    f"{dataset}/topic_{topic_id}: human_label_source must be "
                    "'reviewer_authored'; machine topic names are not validated labels"
                )
            machine_descriptors = {
                _descriptor_signature(review.get("machine_topic_name", "")),
                _descriptor_signature(review.get("machine_topic_keywords", "")),
            }
            if _descriptor_signature(narrative_label) in machine_descriptors:
                issues.append(
                    f"{dataset}/topic_{topic_id}: human_narrative_label must not "
                    "be an unchanged machine topic name or keyword string"
                )

        representatives = representative_reviews[
            representative_reviews["dataset"].astype(str).eq(dataset)
            & representative_reviews["topic_id"].astype(int).eq(topic_id)
        ].copy()
        supports = representatives.get(
            "human_supports_topic",
            pd.Series("", index=representatives.index),
        ).map(_normalized_text)
        stances = representatives.get(
            "human_stance",
            pd.Series("", index=representatives.index),
        ).map(_normalized_text)
        artifacts = representatives.get(
            "human_artifact_pattern",
            pd.Series("", index=representatives.index),
        ).map(_normalized_text)
        confirmations = representatives.get(
            "human_review_confirmation",
            pd.Series("", index=representatives.index),
        ).map(_normalized_text)
        valid_review = (
            supports.isin(allowed_support)
            & stances.isin(allowed_stance)
            & artifacts.isin(allowed_artifact_patterns)
            & confirmations.eq(normalized_confirmation)
        )
        reviewed_representatives = int(valid_review.sum())
        supporting_representatives = int((supports[valid_review] == "yes").sum())
        if reviewed_representatives < thresholds.minimum_representatives:
            issues.append(
                f"{dataset}/topic_{topic_id}: fewer than "
                f"{thresholds.minimum_representatives} human-reviewed representatives "
                "with support, stance, artifact, and confirmation fields completed"
            )
        review_rows.append(
            {
                "dataset": dataset,
                "topic_id": topic_id,
                "records": int(candidate.records),
                "human_topic_coherence": coherence,
                "human_topic_stance": topic_stance,
                "human_artifact_pattern": artifact_pattern,
                "final_narrative_id": narrative_id,
                "candidate_narrative_label": narrative_label,
                "representatives_reviewed": reviewed_representatives,
                "representatives_supporting": supporting_representatives,
            }
        )

    topic_outcomes = pd.DataFrame(
        review_rows,
        columns=[
            "dataset",
            "topic_id",
            "records",
            "human_topic_coherence",
            "human_topic_stance",
            "human_artifact_pattern",
            "final_narrative_id",
            "candidate_narrative_label",
            "representatives_reviewed",
            "representatives_supporting",
        ],
    )
    if issues:
        return NarrativeFinalization(
            "awaiting_human_validation",
            issues,
            empty_narratives,
            empty_mapping,
            topic_outcomes,
        )

    coherent = topic_outcomes[
        topic_outcomes["human_topic_coherence"].eq("coherent")
        & topic_outcomes["human_artifact_pattern"].eq("none")
    ].copy()
    narrative_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    for (dataset, narrative_id), group in coherent.groupby(
        ["dataset", "final_narrative_id"],
        sort=True,
    ):
        labels = set(group["candidate_narrative_label"])
        if len(labels) != 1:
            issues.append(
                f"{dataset}/{narrative_id}: merged topics have inconsistent "
                "human labels"
            )
            continue
        records = int(group["records"].sum())
        representatives_reviewed = int(group["representatives_reviewed"].sum())
        representatives_supporting = int(group["representatives_supporting"].sum())
        retained = (
            records >= thresholds.minimum_topic_documents
            and representatives_supporting >= thresholds.minimum_representatives
        )
        if records < thresholds.minimum_topic_documents:
            retention_reason = "fewer_than_minimum_documents"
        elif representatives_supporting < thresholds.minimum_representatives:
            retention_reason = "fewer_than_minimum_supporting_representatives"
        else:
            retention_reason = "retained_after_human_validation"
        narrative_rows.append(
            {
                "dataset": dataset,
                "final_narrative_id": narrative_id,
                "candidate_narrative_label": next(iter(labels)),
                "records": records,
                "topics": "|".join(str(value) for value in sorted(group["topic_id"])),
                "representatives_reviewed": representatives_reviewed,
                "representatives_supporting": representatives_supporting,
                "retained": retained,
                "retention_reason": retention_reason,
            }
        )
        if retained:
            for topic_id in group["topic_id"]:
                mapping_rows.append(
                    {
                        "dataset": dataset,
                        "topic_id": int(topic_id),
                        "final_narrative_id": narrative_id,
                        "candidate_narrative_label": next(iter(labels)),
                    }
                )

    narratives = pd.DataFrame(narrative_rows, columns=empty_narratives.columns)
    mapping = pd.DataFrame(mapping_rows, columns=empty_mapping.columns)
    if issues:
        return NarrativeFinalization(
            "awaiting_human_validation",
            issues,
            empty_narratives,
            empty_mapping,
            topic_outcomes,
        )

    retained_topics = set(zip(mapping["dataset"], mapping["topic_id"]))
    topic_outcomes["retained"] = [
        (row.dataset, row.topic_id) in retained_topics
        for row in topic_outcomes.itertuples(index=False)
    ]
    topic_outcomes["retention_reason"] = [
        (
            "retained_after_human_validation"
            if row.retained
            else (
                f"rejected_{row.human_artifact_pattern}_artifact_pattern"
                if row.human_artifact_pattern != "none"
                else f"rejected_{row.human_topic_coherence}"
                if row.human_topic_coherence != "coherent"
                else "rejected_by_narrative_retention_gate"
            )
        )
        for row in topic_outcomes.itertuples(index=False)
    ]
    return NarrativeFinalization(
        "completed",
        [],
        narratives,
        mapping,
        topic_outcomes,
    )


def narrative_prevalence_table(
    unreliable_assignments: pd.DataFrame,
    topic_mapping: pd.DataFrame,
) -> pd.DataFrame:
    mapped = unreliable_assignments.merge(
        topic_mapping,
        on=["dataset", "topic_id"],
        how="inner",
        validate="many_to_one",
    )
    denominators = unreliable_assignments.groupby("dataset")["record_id"].nunique()
    prevalence = (
        mapped.groupby(
            ["dataset", "final_narrative_id", "candidate_narrative_label"]
        )["record_id"]
        .nunique()
        .reset_index(name="unreliable_records")
    )
    prevalence["all_unreliable_records"] = prevalence["dataset"].map(denominators)
    prevalence["unreliable_prevalence"] = (
        prevalence["unreliable_records"] / prevalence["all_unreliable_records"]
    )
    return prevalence


def narrative_performance_table(
    predictions: pd.DataFrame,
    all_document_assignments: pd.DataFrame,
    topic_mapping: pd.DataFrame,
    thresholds: NarrativeThresholds,
) -> pd.DataFrame:
    assigned = all_document_assignments.merge(
        topic_mapping,
        on=["dataset", "topic_id"],
        how="inner",
        validate="many_to_one",
    )
    joined = predictions.merge(
        assigned[
            [
                "dataset",
                "record_id",
                "final_narrative_id",
                "candidate_narrative_label",
            ]
        ],
        on=["dataset", "record_id"],
        how="inner",
        validate="many_to_one",
    )
    group_columns = [
        column
        for column in ("dataset", "split_name", "model", "experiment_id")
        if column in joined.columns
    ] + ["final_narrative_id", "candidate_narrative_label"]
    rows: list[dict[str, object]] = []
    for keys, group in joined.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        class_counts = group["harmonized_label"].value_counts()
        records = len(group)
        reliable_records = int(class_counts.get("reliable", 0))
        unreliable_records = int(class_counts.get("unreliable", 0))
        eligible = (
            records >= thresholds.minimum_performance_records
            and reliable_records >= thresholds.minimum_performance_per_class
            and unreliable_records >= thresholds.minimum_performance_per_class
        )
        row.update(
            {
                "records": records,
                "reliable_records": reliable_records,
                "unreliable_records": unreliable_records,
                "performance_eligible": eligible,
                "performance_gate_reason": (
                    "eligible"
                    if eligible
                    else "requires_minimum_records_and_both_class_counts"
                ),
            }
        )
        if eligible:
            row.update(
                classification_metrics(
                    group["harmonized_label"],
                    group["prediction"],
                    group["score_unreliable"],
                    probability_scores=False,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def narrative_metadata_profile(
    data: pd.DataFrame,
    all_document_assignments: pd.DataFrame,
    topic_mapping: pd.DataFrame,
    metadata_columns: list[str],
    thresholds: NarrativeThresholds,
    availability_column: str | None = None,
) -> pd.DataFrame:
    assigned = all_document_assignments.merge(
        topic_mapping,
        on=["dataset", "topic_id"],
        how="inner",
        validate="many_to_one",
    )
    joined = assigned.merge(
        data[["dataset", "record_id", *metadata_columns, *(
            [availability_column] if availability_column else []
        )]],
        on=["dataset", "record_id"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for (dataset, narrative_id, label), group in joined.groupby(
        ["dataset", "final_narrative_id", "candidate_narrative_label"],
        sort=True,
    ):
        for column in metadata_columns:
            available = group[column].notna()
            if availability_column:
                available &= group[availability_column].fillna(False).astype(bool)
            values = pd.to_numeric(
                group.loc[available, column],
                errors="coerce",
            ).dropna()
            eligible = (
                len(values) >= thresholds.minimum_metadata_complete_per_group
            )
            rows.append(
                {
                    "dataset": dataset,
                    "final_narrative_id": narrative_id,
                    "candidate_narrative_label": label,
                    "metadata_field": column,
                    "complete_records": len(values),
                    "metadata_comparison_eligible": eligible,
                    "metadata_gate_reason": (
                        "eligible"
                        if eligible
                        else "fewer_than_minimum_metadata_complete_records"
                    ),
                    "mean": float(values.mean()) if eligible else np.nan,
                    "median": float(values.median()) if eligible else np.nan,
                    "standard_deviation": (
                        float(values.std(ddof=1)) if eligible else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)

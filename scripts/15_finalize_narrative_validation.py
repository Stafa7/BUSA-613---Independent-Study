#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_paths, load_yaml
from health_misinfo.evaluation.narratives import (
    NarrativeThresholds,
    finalize_human_narratives,
    narrative_metadata_profile,
    narrative_performance_table,
    narrative_prevalence_table,
    representative_review_template,
)


MODELS = ("logistic_regression", "linear_svm")
SPLITS = ("standard", "controlled")
PREVALENCE_COLUMNS = [
    "dataset",
    "final_narrative_id",
    "candidate_narrative_label",
    "unreliable_records",
    "all_unreliable_records",
    "unreliable_prevalence",
]
PERFORMANCE_COLUMNS = [
    "dataset",
    "split_name",
    "model",
    "experiment_id",
    "final_narrative_id",
    "candidate_narrative_label",
    "records",
    "reliable_records",
    "unreliable_records",
    "performance_eligible",
    "performance_gate_reason",
    "n",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "reliable_precision",
    "reliable_recall",
    "reliable_f1",
    "reliable_support",
    "unreliable_precision",
    "unreliable_recall",
    "unreliable_f1",
    "unreliable_support",
    "unreliable_pr_auc",
    "roc_auc",
]
METADATA_PROFILE_COLUMNS = [
    "dataset",
    "final_narrative_id",
    "candidate_narrative_label",
    "metadata_field",
    "complete_records",
    "metadata_comparison_eligible",
    "metadata_gate_reason",
    "mean",
    "median",
    "standard_deviation",
]


def _write_status(out: Path, status: str, **details: object) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    (out / "human_validation_status.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _thresholds(config: dict) -> NarrativeThresholds:
    return NarrativeThresholds(**{
        key: int(value)
        for key, value in config["thresholds"].items()
    })


def _prediction_table(run_root: Path, datasets: set[str]) -> pd.DataFrame:
    rows = []
    for dataset in sorted(datasets):
        for split_name in SPLITS:
            for model in MODELS:
                experiment_id = f"{dataset}_{split_name}_{model}_model_text"
                path = run_root / experiment_id / "predictions.csv"
                if not path.exists():
                    continue
                predictions = pd.read_csv(path, dtype={"record_id": str})
                predictions["dataset"] = dataset
                predictions["split_name"] = split_name
                predictions["model"] = model
                predictions["experiment_id"] = experiment_id
                rows.append(predictions)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_prevalence_chart(prevalence: pd.DataFrame, path: Path) -> None:
    chart = prevalence.sort_values(
        ["dataset", "unreliable_prevalence", "candidate_narrative_label"],
        ascending=[True, True, True],
    ).copy()
    figure_height = max(4.0, 0.45 * max(len(chart), 1) + 1.5)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    if chart.empty:
        axis.text(
            0.5,
            0.5,
            "No candidate narratives passed the frozen human-validation gates.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    else:
        chart["display_label"] = (
            chart["dataset"].astype(str)
            + " — "
            + chart["candidate_narrative_label"].astype(str)
        )
        axis.barh(
            chart["display_label"],
            chart["unreliable_prevalence"] * 100,
            color="#3b6ea8",
        )
        axis.set_xlabel("Share of dataset-defined unreliable records (%)")
        axis.set_ylabel("Human-validated candidate narrative")
        axis.set_title("Candidate-narrative prevalence by dataset")
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    paths = load_paths()
    config = load_yaml("configs/narratives.yaml")
    thresholds = _thresholds(config)
    human_review_config = config["human_review"]
    if int(human_review_config["reviewer_count"]) != 1:
        raise ValueError("Narrative protocol requires exactly one human reviewer")
    review_files = human_review_config["files"]
    latest_path = paths["experiments"] / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            "No baseline latest_run.json is available; run script 05 first"
        )
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    run_root = Path(latest["run_path"])
    discovery_root = run_root / "narrative_discovery"
    final_root = run_root / "narrative_validation"
    final_root.mkdir(parents=True, exist_ok=True)
    if not discovery_root.exists():
        _write_status(
            final_root,
            "gated_no_discovery_artifacts",
            reason="Run script 13 before finalizing human narrative validation.",
        )
        return 2

    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[data["primary_eligible"]].copy()
    narrative_frames = []
    mapping_frames = []
    outcome_frames = []
    unreliable_assignment_frames = []
    all_assignment_frames = []
    issues: list[str] = []
    discovered_datasets: set[str] = set()
    dataset_dirs = sorted(
        path for path in discovery_root.iterdir() if path.is_dir()
    )
    for dataset_dir in dataset_dirs:
        required = {
            "candidates": dataset_dir / "candidate_topics.csv",
            "topics": dataset_dir / str(review_files["topic_decisions"]),
            "representatives": (
                dataset_dir / str(review_files["representative_documents"])
            ),
            "unreliable_assignments": (
                dataset_dir / "unreliable_document_topic_assignments.csv"
            ),
            "all_assignments": dataset_dir / "all_document_topic_assignments.csv",
        }
        missing_files = [
            f"{name}={path.name}"
            for name, path in required.items()
            if not path.exists()
        ]
        if missing_files:
            discovered_datasets.add(dataset_dir.name)
            issues.append(
                f"{dataset_dir.name}: missing required handoff artifacts: "
                f"{', '.join(missing_files)}"
            )
            continue
        dataset = dataset_dir.name
        discovered_datasets.add(dataset)
        candidates = pd.read_csv(required["candidates"])
        topic_reviews = pd.read_csv(
            required["topics"],
            keep_default_na=False,
        )
        representative_reviews = pd.read_csv(
            required["representatives"],
            keep_default_na=False,
            dtype={"record_id": str},
        )
        unreliable_assignments = pd.read_csv(
            required["unreliable_assignments"],
            dtype={"record_id": str},
        )
        all_assignments = pd.read_csv(
            required["all_assignments"],
            dtype={"record_id": str},
        )
        expected_representatives = representative_review_template(
            candidates,
            unreliable_assignments,
            data[data["dataset"].eq(dataset)],
            representatives_per_topic=int(config["representatives_per_topic"]),
        )
        finalization = finalize_human_narratives(
            candidates,
            topic_reviews,
            representative_reviews,
            thresholds,
            expected_representatives=expected_representatives,
            required_confirmation=str(
                human_review_config["required_confirmation"]
            ),
        )
        if len(finalization.topic_outcomes):
            finalization.topic_outcomes.to_csv(
                final_root / f"{dataset}_topic_validation_outcomes.csv",
                index=False,
            )
        if finalization.status != "completed":
            issues.extend(
                f"{dataset}: {issue}" for issue in finalization.issues
            )
            continue
        narrative_frames.append(finalization.narratives)
        mapping_frames.append(finalization.topic_mapping)
        outcome_frames.append(finalization.topic_outcomes)
        unreliable_assignment_frames.append(unreliable_assignments)
        all_assignment_frames.append(all_assignments)

    if not discovered_datasets:
        _write_status(
            final_root,
            "gated_no_candidate_topics",
            reason=(
                "Narrative discovery produced no dataset-level candidate artifacts; "
                "inspect narrative_discovery/status.json."
            ),
        )
        return 2
    if issues:
        _write_status(
            final_root,
            "awaiting_human_validation",
            issues=issues,
            human_labels_created_by_finalizer=False,
            thresholds=config["thresholds"],
            review_files=review_files,
        )
        return 2

    narratives = pd.concat(narrative_frames, ignore_index=True)
    topic_mapping = pd.concat(mapping_frames, ignore_index=True)
    topic_outcomes = pd.concat(outcome_frames, ignore_index=True)
    unreliable_assignments = pd.concat(
        unreliable_assignment_frames,
        ignore_index=True,
    )
    all_assignments = pd.concat(all_assignment_frames, ignore_index=True)
    narratives.to_csv(final_root / "validated_narratives.csv", index=False)
    topic_mapping.to_csv(final_root / "retained_topic_to_narrative.csv", index=False)
    topic_outcomes.to_csv(final_root / "topic_validation_outcomes.csv", index=False)

    retained_assignments = all_assignments.merge(
        topic_mapping,
        on=["dataset", "topic_id"],
        how="inner",
        validate="many_to_one",
    )
    retained_assignments.to_csv(
        final_root / "all_document_narrative_assignments.csv",
        index=False,
    )
    prevalence = narrative_prevalence_table(
        unreliable_assignments,
        topic_mapping,
    ).reindex(columns=PREVALENCE_COLUMNS)
    prevalence.to_csv(final_root / "narrative_prevalence.csv", index=False)
    _write_prevalence_chart(
        prevalence,
        final_root / "narrative_prevalence.png",
    )

    predictions = _prediction_table(run_root, discovered_datasets)
    if len(predictions) and len(topic_mapping):
        performance = narrative_performance_table(
            predictions,
            all_assignments,
            topic_mapping,
            thresholds,
        )
        performance_status = (
            "evaluated_with_group_specific_gates"
            if len(performance)
            else "unavailable_no_assigned_test_records"
        )
    elif not len(topic_mapping):
        performance = pd.DataFrame()
        performance_status = "unavailable_no_retained_narratives"
    else:
        performance = pd.DataFrame()
        performance_status = "unavailable_no_baseline_predictions"
    performance = performance.reindex(columns=PERFORMANCE_COLUMNS)
    performance.to_csv(final_root / "narrative_performance.csv", index=False)

    if len(topic_mapping):
        engagement = narrative_metadata_profile(
            data,
            all_assignments,
            topic_mapping,
            ["tweet_count", "reply_count", "retweet_count"],
            thresholds,
            availability_column="engagement_available",
        )
        engagement_status = (
            "evaluated_with_group_specific_gates"
            if len(engagement)
            else "unavailable_no_assigned_metadata_records"
        )
    else:
        engagement = pd.DataFrame()
        engagement_status = "unavailable_no_retained_narratives"
    engagement = engagement.reindex(columns=METADATA_PROFILE_COLUMNS)
    engagement.to_csv(
        final_root / "narrative_engagement_profiles.csv",
        index=False,
    )
    gates = pd.DataFrame(
        [
            {
                "analysis": "narrative_performance",
                "minimum_records": thresholds.minimum_performance_records,
                "minimum_per_class": thresholds.minimum_performance_per_class,
                "eligible_groups": (
                    int(performance["performance_eligible"].sum())
                    if len(performance)
                    else 0
                ),
                "status": performance_status,
            },
            {
                "analysis": "narrative_engagement",
                "minimum_complete_per_group": (
                    thresholds.minimum_metadata_complete_per_group
                ),
                "eligible_groups": (
                    int(engagement["metadata_comparison_eligible"].sum())
                    if len(engagement)
                    else 0
                ),
                "status": engagement_status,
            },
            {
                "analysis": "narrative_emotion",
                "eligible_groups": 0,
                "status": "unavailable_no_frozen_emotion_features",
            },
        ]
    )
    gates.to_csv(final_root / "conditional_analysis_gates.csv", index=False)
    retained_count = int(narratives["retained"].sum()) if len(narratives) else 0
    _write_status(
        final_root,
        "completed" if retained_count else "completed_no_retained_narratives",
        discovered_datasets=sorted(discovered_datasets),
        retained_narratives=retained_count,
        human_reviewer_count=int(human_review_config["reviewer_count"]),
        human_labels_created_by_finalizer=False,
        thresholds=config["thresholds"],
        review_files=review_files,
    )
    print(f"Wrote human-validated narrative outputs to {final_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

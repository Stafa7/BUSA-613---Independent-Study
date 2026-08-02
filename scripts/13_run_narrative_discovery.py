#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_paths, load_yaml
from health_misinfo.evaluation.narratives import (
    NarrativeThresholds,
    assigned_topic_probabilities,
    candidate_topic_table,
    discovery_eligibility,
    representative_review_template,
    topic_review_template,
)


OPTIONAL_DEPENDENCIES = {
    "bertopic": "bertopic",
    "sentence_transformers": "sentence-transformers",
    "umap": "umap-learn",
    "hdbscan": "hdbscan",
}


def _write_status(out: Path, status: str, **details: object) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    (out / "status.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _thresholds(config: dict) -> NarrativeThresholds:
    return NarrativeThresholds(**{
        key: int(value)
        for key, value in config["thresholds"].items()
    })


def _topic_descriptors(topic_model) -> pd.DataFrame:
    info = topic_model.get_topic_info().rename(columns={"Topic": "topic_id"})
    descriptor_columns = ["topic_id"]
    rename = {}
    for source, destination in (
        ("Count", "model_topic_records"),
        ("Name", "machine_topic_name"),
        ("Representation", "machine_topic_keywords"),
    ):
        if source in info:
            descriptor_columns.append(source)
            rename[source] = destination
    descriptors = info[descriptor_columns].rename(columns=rename)
    if "machine_topic_keywords" in descriptors:
        descriptors["machine_topic_keywords"] = descriptors[
            "machine_topic_keywords"
        ].map(
            lambda value: "|".join(map(str, value))
            if isinstance(value, (list, tuple))
            else str(value)
        )
    return descriptors


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
    out = run_root / "narrative_discovery"
    out.mkdir(parents=True, exist_ok=True)
    (out / "narrative_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[data["primary_eligible"]].copy()
    eligibility_rows = []
    for dataset, frame in data.groupby("dataset"):
        eligibility_rows.append(
            {
                "dataset": dataset,
                **discovery_eligibility(frame, thresholds),
            }
        )
    eligibility = pd.DataFrame(eligibility_rows)
    eligibility.to_csv(out / "discovery_eligibility.csv", index=False)
    eligible_datasets = eligibility.loc[eligibility["eligible"], "dataset"].tolist()
    if not eligible_datasets:
        _write_status(
            out,
            "gated_insufficient_unreliable_documents",
            thresholds=config["thresholds"],
            eligibility=eligibility_rows,
        )
        return 2

    missing_dependencies = [
        package
        for module, package in OPTIONAL_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing_dependencies:
        _write_status(
            out,
            "gated_optional_dependencies_unavailable",
            reason=(
                "BERTopic discovery is optional and was not approximated with a "
                "different method. Provision and freeze a reviewed optional stack "
                "in an approved environment before rerunning this phase."
            ),
            missing_dependencies=missing_dependencies,
            eligible_datasets=eligible_datasets,
            planned_embedding_model=config["embedding_model"],
            thresholds=config["thresholds"],
        )
        return 2

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    seed = int(config["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    embedding_config = config["embedding_model"]
    try:
        embedding_model = SentenceTransformer(
            str(embedding_config["name"]),
            revision=str(embedding_config["revision"]),
            device="cpu",
        )
    except Exception as exc:
        _write_status(
            out,
            "gated_embedding_model_unavailable",
            reason=f"{type(exc).__name__}: {exc}",
            embedding_model=embedding_config,
            eligible_datasets=eligible_datasets,
        )
        return 2

    completed: list[str] = []
    failures: list[dict[str, str]] = []
    for dataset in eligible_datasets:
        dataset_out = out / str(dataset)
        dataset_out.mkdir(parents=True, exist_ok=True)
        dataset_frame = data[data["dataset"].eq(dataset)].copy()
        fit_frame = dataset_frame[
            dataset_frame["harmonized_label"].eq("unreliable")
        ].copy()
        try:
            umap_config = config["bertopic"]["umap"]
            hdbscan_config = config["bertopic"]["hdbscan"]
            umap_model = UMAP(
                n_neighbors=int(umap_config["n_neighbors"]),
                n_components=int(umap_config["n_components"]),
                min_dist=float(umap_config["min_dist"]),
                metric=str(umap_config["metric"]),
                random_state=seed,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=thresholds.minimum_topic_documents,
                metric=str(hdbscan_config["metric"]),
                cluster_selection_method=str(
                    hdbscan_config["cluster_selection_method"]
                ),
                prediction_data=True,
            )
            ngram_range = tuple(
                int(value)
                for value in config["bertopic"]["ngram_range"]
            )
            vectorizer_model = CountVectorizer(
                stop_words="english",
                ngram_range=ngram_range,
            )
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                top_n_words=int(config["bertopic"]["top_n_words"]),
                calculate_probabilities=True,
                verbose=True,
            )
            fit_topics, fit_probabilities = topic_model.fit_transform(
                fit_frame["model_text"].fillna("").astype(str).tolist()
            )
            fit_assignments = fit_frame[
                ["dataset", "record_id", "harmonized_label"]
            ].copy()
            fit_assignments["topic_id"] = np.asarray(fit_topics, dtype=int)
            fit_assignments["assigned_probability"] = assigned_topic_probabilities(
                fit_topics,
                fit_probabilities,
            )
            fit_assignments.to_csv(
                dataset_out / "unreliable_document_topic_assignments.csv",
                index=False,
            )

            non_fit_frame = dataset_frame[
                ~dataset_frame["record_id"].isin(fit_frame["record_id"])
            ].copy()
            if len(non_fit_frame):
                # HDBSCAN's full out-of-sample membership matrix can fail for
                # otherwise valid condensed trees. We only consume the assigned
                # topic strength here, so use BERTopic's approximate-prediction
                # path and retain full probabilities for the fitted documents.
                topic_model.calculate_probabilities = False
                try:
                    non_fit_topics, non_fit_probabilities = topic_model.transform(
                        non_fit_frame["model_text"].fillna("").astype(str).tolist()
                    )
                finally:
                    topic_model.calculate_probabilities = True
                non_fit_assignments = non_fit_frame[
                    ["dataset", "record_id", "harmonized_label"]
                ].copy()
                non_fit_assignments["topic_id"] = np.asarray(
                    non_fit_topics,
                    dtype=int,
                )
                non_fit_assignments[
                    "assigned_probability"
                ] = assigned_topic_probabilities(
                    non_fit_topics,
                    non_fit_probabilities,
                )
                all_assignments = pd.concat(
                    [fit_assignments, non_fit_assignments],
                    ignore_index=True,
                )
            else:
                all_assignments = fit_assignments.copy()
            all_assignments.to_csv(
                dataset_out / "all_document_topic_assignments.csv",
                index=False,
            )

            candidates = candidate_topic_table(fit_assignments, thresholds)
            candidates = candidates.merge(
                _topic_descriptors(topic_model),
                on="topic_id",
                how="left",
                validate="one_to_one",
            )
            candidates.to_csv(dataset_out / "candidate_topics.csv", index=False)
            topic_review_topic = topic_review_template(candidates)
            topic_review_topic.to_csv(
                dataset_out / str(review_files["topic_decisions"]),
                index=False,
            )
            representatives = representative_review_template(
                candidates,
                fit_assignments,
                dataset_frame,
                representatives_per_topic=int(config["representatives_per_topic"]),
            )
            representatives.to_csv(
                dataset_out / str(review_files["representative_documents"]),
                index=False,
            )
            metadata = {
                "dataset": dataset,
                "fit_records": len(fit_frame),
                "all_records_transformed": len(dataset_frame),
                "candidate_topics": len(candidates),
                "size_eligible_topics": int(candidates["eligible_topic_size"].sum()),
                "embedding_model": embedding_config,
                "runtime": {
                    "python": platform.python_version(),
                    **{
                        package: importlib.metadata.version(distribution)
                        for package, distribution in OPTIONAL_DEPENDENCIES.items()
                    },
                },
            }
            (dataset_out / "topic_model_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            completed.append(str(dataset))
        except Exception as exc:
            failures.append(
                {
                    "dataset": str(dataset),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    instructions = f"""# Start here: single-reviewer narrative validation

Machine topic names and keywords are navigation aids, not validated narrative
findings. One human reviewer completes these files inside each dataset folder.

## 1. Review representative documents

Open `{review_files["representative_documents"]}` first. Inspect at least
{thresholds.minimum_representatives} documents per topic and complete:

- `human_supports_topic`: `yes`, `no`, or `unclear`
- `human_stance`: `endorsement`, `discussion`, `debunking`, `mixed`, or `unclear`
- `human_artifact_pattern`: `none`, `publisher`, `dataset`, `generic`, `mixed`,
  or `unclear`
- `human_notes`: optional rationale
- `human_review_confirmation`: `{human_review_config["required_confirmation"]}`

## 2. Make the topic decision and supply the label

Then open `{review_files["topic_decisions"]}` and complete:

- `human_topic_coherence`: `coherent`, `mixed`, or `incoherent`
- `human_topic_stance`: the same stance values listed above
- `human_artifact_pattern`: the same artifact values listed above
- `human_narrative_id` and `human_narrative_label` for a coherent, non-artifact
  topic; topics may share an ID only when the reviewer intentionally merges them
- `human_label_source`: `reviewer_authored` for each retained candidate
- `human_notes`: optional rationale
- `human_review_confirmation`: `{human_review_config["required_confirmation"]}`

Write a descriptive label in your own words. Do not copy `machine_topic_name` or
`machine_topic_keywords`; the finalizer does not generate, repair, or validate a
machine label. Publisher-, dataset-, and generic-language patterns are rejected
as narratives.

After every eligible topic is complete, run:

`.venv/bin/python scripts/15_finalize_narrative_validation.py`

The finalizer applies the frozen size, representative, performance, and metadata
gates. It will stop and list incomplete fields instead of filling them.
"""
    (out / str(review_files["instructions"])).write_text(
        instructions,
        encoding="utf-8",
    )
    status = (
        "awaiting_human_validation"
        if completed and not failures
        else "partial_compute_failure"
        if completed
        else "gated_compute_failure"
    )
    _write_status(
        out,
        status,
        completed_datasets=completed,
        failures=failures,
        thresholds=config["thresholds"],
        human_review=human_review_config,
        human_labels_created=False,
    )
    print(f"Wrote candidate narrative-discovery artifacts to {out}")
    return 0 if completed and not failures else 1 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

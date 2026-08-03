#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_experiments, load_paths  # noqa: E402
from health_misinfo.evaluation.bootstrap import bootstrap_intervals  # noqa: E402
from health_misinfo.evaluation.metrics import classification_metrics  # noqa: E402
from health_misinfo.evaluation.transfer_shift import (  # noqa: E402
    BaselineComparatorSpec,
    TransferShiftArtifacts,
    baseline_comparator_spec,
    build_transfer_shift_artifacts,
    render_degradation_markdown,
    transfer_degradation_rows,
)
from health_misinfo.models.baselines import (  # noqa: E402
    baseline_candidates,
    fit_predict,
)


def _score_vector(score, prediction):
    if hasattr(score, "ndim") and score.ndim == 2:
        return score[:, 1]
    return score


def _json_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a frame to strict JSON-compatible records, including nulls."""

    return json.loads(
        frame.to_json(orient="records", double_precision=15, date_format="iso")
    )


def _write_shift_artifacts(
    diagnostics_root: Path,
    artifacts: TransferShiftArtifacts,
) -> None:
    direction_root = diagnostics_root / artifacts.metadata["direction"]
    direction_root.mkdir(parents=True, exist_ok=True)
    for table_name, frame in artifacts.tables.items():
        frame.to_csv(direction_root / f"{table_name}.csv", index=False)

    summary_names = (
        "cohort_profiles",
        "domain_exclusion_summary",
        "vocabulary_summary",
        "document_length_shift",
        "categorical_shift_summary",
        "topic_proxy_summary",
        "shift_summary",
    )
    payload = {
        "metadata": artifacts.metadata,
        "summaries": {
            name: _json_records(artifacts.tables[name])
            for name in summary_names
        },
        "detail_tables": {
            name: f"{name}.csv"
            for name in artifacts.tables
            if name not in summary_names
        },
    }
    (direction_root / "shift_diagnostics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (direction_root / "shift_diagnostics.md").write_text(
        artifacts.markdown,
        encoding="utf-8",
    )


def _demote_markdown_headings(markdown: str) -> str:
    return "\n".join(
        f"#{line}" if line.startswith("#") else line
        for line in markdown.splitlines()
    )


def _load_baseline_artifacts(
    run_root: Path,
    comparator: BaselineComparatorSpec,
) -> tuple[dict | None, pd.DataFrame | None, str, str]:
    if not comparator.baseline_experiment_id:
        return None, None, "", ""
    baseline_root = run_root / comparator.baseline_experiment_id
    config_path = baseline_root / "config.yaml"
    config = None
    config_reason = ""
    if not config_path.exists():
        config_reason = "baseline_config_artifact_missing"
    else:
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeError, yaml.YAMLError):
            config_reason = "baseline_config_artifact_unreadable"

    predictions = None
    prediction_reason = ""
    prediction_paths = (
        baseline_root / "predictions.parquet",
        baseline_root / "predictions.csv",
    )
    existing_prediction_paths = [
        path for path in prediction_paths if path.exists()
    ]
    for prediction_path in existing_prediction_paths:
        try:
            predictions = (
                pd.read_parquet(prediction_path)
                if prediction_path.suffix == ".parquet"
                else pd.read_csv(prediction_path)
            )
            break
        except (OSError, ValueError, ImportError):
            continue
    if predictions is None:
        prediction_reason = (
            "baseline_prediction_artifact_unreadable"
            if existing_prediction_paths
            else "baseline_prediction_artifact_missing"
        )
    return config, predictions, config_reason, prediction_reason


def main() -> int:
    paths = load_paths()
    config = load_experiments()
    latest = json.loads(
        (paths["experiments"] / "latest_run.json").read_text(encoding="utf-8")
    )
    run_root = Path(latest["run_path"])
    out_root = run_root / "cross_dataset_transfer"
    out_root.mkdir(parents=True, exist_ok=True)
    baseline_summary_path = run_root / "text_baseline_summary.csv"
    baseline_summary = (
        pd.read_csv(baseline_summary_path)
        if baseline_summary_path.exists()
        else pd.DataFrame()
    )
    baseline_artifact_cache: dict[
        str,
        tuple[dict | None, pd.DataFrame | None, str, str],
    ] = {}
    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[data["primary_eligible"]].copy()
    manifest = pd.read_csv(paths["splits"] / "standard_split_manifest.csv")
    candidates = baseline_candidates(config)
    iterations = int(config["metrics"]["bootstrap_iterations"])
    seed = int(config["random_seed"])
    rows = []
    degradation_frames: list[pd.DataFrame] = []
    shift_artifacts: list[TransferShiftArtifacts] = []
    diagnostics_root = out_root / "shift_diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    for source_dataset, target_dataset in (
        ("coaid", "fakehealth"),
        ("fakehealth", "coaid"),
    ):
        source = data[data["dataset"].eq(source_dataset)].merge(
            manifest[manifest["dataset"].eq(source_dataset)][["record_id", "split"]],
            on="record_id",
            how="inner",
        )
        target = data[data["dataset"].eq(target_dataset)].merge(
            manifest[manifest["dataset"].eq(target_dataset)][
                ["record_id", "split", "family_group"]
            ],
            on=["record_id", "family_group"],
            how="inner",
        )
        source_train = source[source["split"].eq("train")]
        source_validation = source[source["split"].eq("validation")]
        target_test_before_domain_exclusion = target[
            target["split"].eq("test")
        ].copy()
        source_domains = set(
            source[source["split"].isin(["train", "validation"])]
            ["source_domain"]
            .fillna("")
            .loc[lambda values: values.ne("")]
        )
        target_test = target_test_before_domain_exclusion.copy()
        target_test["overlapping_source_domain"] = target_test["source_domain"].isin(
            source_domains
        )
        target_test = target_test[~target_test["overlapping_source_domain"]].copy()

        diagnostics = build_transfer_shift_artifacts(
            source_train,
            target_test_before_domain_exclusion,
            target_test,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            exclusion_source_domains=source_domains,
            random_seed=seed,
            vocabulary_max_features=int(
                config["models"]["tfidf"]["max_features"]
            ),
            vocabulary_minimum_term_count=2,
            topic_requested_components=10,
            topic_max_features=min(
                5_000,
                int(config["models"]["tfidf"]["max_features"]),
            ),
            topic_top_terms=10,
        )
        _write_shift_artifacts(diagnostics_root, diagnostics)
        shift_artifacts.append(diagnostics)

        for text_name, text_col in (("full_text", "model_text"), ("title_only", "title")):
            for model_name in ("logistic_regression", "linear_svm"):
                fitted = []
                for spec in candidates[model_name]:
                    validation_prediction, validation_score = fit_predict(
                        spec,
                        source_train[text_col].fillna(""),
                        source_train["harmonized_label"],
                        source_validation[text_col].fillna(""),
                    )
                    validation_metrics = classification_metrics(
                        source_validation["harmonized_label"],
                        validation_prediction,
                        _score_vector(validation_score, validation_prediction),
                        probability_scores=spec.probability_scores,
                    )
                    fitted.append((spec, validation_metrics))
                selected, validation_metrics = max(
                    fitted,
                    key=lambda item: item[1]["macro_f1"],
                )
                prediction, score = fit_predict(
                    selected,
                    source_train[text_col].fillna(""),
                    source_train["harmonized_label"],
                    target_test[text_col].fillna(""),
                )
                score_vector = _score_vector(score, prediction)
                metrics = classification_metrics(
                    target_test["harmonized_label"],
                    prediction,
                    score_vector,
                    probability_scores=selected.probability_scores,
                )
                experiment_id = (
                    f"{source_dataset}_to_{target_dataset}_{text_name}_{model_name}"
                )
                experiment_out = out_root / experiment_id
                experiment_out.mkdir(parents=True, exist_ok=True)
                predictions = target_test[
                    [
                        "dataset",
                        "record_id",
                        "harmonized_label",
                        "text_unit",
                        "publisher_id",
                        "source_domain",
                        "family_group",
                    ]
                ].copy()
                predictions["prediction"] = prediction
                predictions["score_unreliable"] = score_vector
                for comparator_kind in (
                    "target_in_domain_same_records",
                    "source_in_domain_standard_test",
                ):
                    comparator = baseline_comparator_spec(
                        source_dataset=source_dataset,
                        target_dataset=target_dataset,
                        text_representation=text_name,
                        transfer_model=model_name,
                        comparator_kind=comparator_kind,
                    )
                    cache_key = comparator.baseline_experiment_id or (
                        f"unavailable:{text_name}:{model_name}"
                    )
                    if cache_key not in baseline_artifact_cache:
                        baseline_artifact_cache[cache_key] = (
                            _load_baseline_artifacts(run_root, comparator)
                        )
                    (
                        baseline_config,
                        baseline_predictions,
                        baseline_config_reason,
                        baseline_predictions_reason,
                    ) = baseline_artifact_cache[cache_key]
                    degradation_frames.append(
                        transfer_degradation_rows(
                            transfer_experiment_id=experiment_id,
                            source_dataset=source_dataset,
                            target_dataset=target_dataset,
                            text_representation=text_name,
                            transfer_model=model_name,
                            transfer_predictions=predictions,
                            comparator=comparator,
                            baseline_summary=baseline_summary,
                            baseline_config=baseline_config,
                            baseline_predictions=baseline_predictions,
                            baseline_config_unavailable_reason=(
                                baseline_config_reason
                            ),
                            baseline_predictions_unavailable_reason=(
                                baseline_predictions_reason
                            ),
                        )
                    )
                predictions.to_csv(experiment_out / "predictions.csv", index=False)
                bootstrap_intervals(
                    predictions["harmonized_label"],
                    predictions["prediction"],
                    predictions["score_unreliable"],
                    iterations=iterations,
                    seed=seed,
                    groups=predictions["family_group"],
                    probability_scores=selected.probability_scores,
                ).to_csv(experiment_out / "bootstrap_intervals.csv", index=False)
                (experiment_out / "config.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "experiment_id": experiment_id,
                            "source_dataset": source_dataset,
                            "target_dataset": target_dataset,
                            "source_train_records": len(source_train),
                            "source_validation_records": len(source_validation),
                            "target_test_records_after_domain_exclusion": len(target_test),
                            "text_representation": text_name,
                            "model": model_name,
                            "selected_hyperparameters": selected.hyperparameters,
                            "validation_macro_f1": validation_metrics["macro_f1"],
                            "interpretation": (
                                "Construct/domain-shift diagnostic; dataset labels are not "
                                "assumed to be equivalent ground truth."
                            ),
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "text_representation": text_name,
                        "model": model_name,
                        "target_test_records": len(target_test),
                        **metrics,
                    }
                )
    pd.DataFrame(rows).to_csv(out_root / "transfer_summary.csv", index=False)
    degradation = pd.concat(degradation_frames, ignore_index=True)
    degradation.insert(
        1,
        "baseline_run_id",
        latest.get("run_id", run_root.name),
    )
    degradation.to_csv(
        out_root / "in_domain_vs_transfer_degradation.csv",
        index=False,
    )
    degradation_payload = {
        "baseline_run_id": latest.get("run_id", run_root.name),
        "baseline_summary": "../text_baseline_summary.csv",
        "primary_comparator": "target_in_domain_same_records",
        "secondary_comparator": "source_in_domain_standard_test",
        "delta_definition": {
            "absolute": "transfer minus in-domain",
            "relative": "(transfer minus in-domain) divided by in-domain",
            "degradation": "in-domain minus transfer",
        },
        "rows": _json_records(degradation),
    }
    (out_root / "in_domain_vs_transfer_degradation.json").write_text(
        json.dumps(degradation_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out_root / "in_domain_vs_transfer_degradation.md").write_text(
        render_degradation_markdown(degradation),
        encoding="utf-8",
    )
    pd.concat(
        [artifacts.tables["shift_summary"] for artifacts in shift_artifacts],
        ignore_index=True,
    ).to_csv(diagnostics_root / "shift_summary.csv", index=False)
    aggregate_payload = {
        "method": (
            "Post-hoc descriptive construct/covariate-shift diagnostics; "
            "no causal attribution and no assumption of equivalent labels."
        ),
        "related_performance_artifact": (
            "../in_domain_vs_transfer_degradation.csv"
        ),
        "directions": [
            {
                "metadata": artifacts.metadata,
                "summary": _json_records(artifacts.tables["shift_summary"]),
            }
            for artifacts in shift_artifacts
        ],
    }
    (diagnostics_root / "shift_diagnostics.json").write_text(
        json.dumps(aggregate_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    combined_markdown = (
        "# Cross-dataset transfer shift diagnostics\n\n"
        "The direction-specific sections compare each source training cohort "
        "with the post-domain-exclusion target test cohort. Like-for-like "
        "performance deltas are reported separately in "
        "[the degradation table](../in_domain_vs_transfer_degradation.md).\n\n"
        + "\n\n---\n\n".join(
            _demote_markdown_headings(artifacts.markdown)
            for artifacts in shift_artifacts
        )
        + "\n"
    )
    (diagnostics_root / "shift_diagnostics.md").write_text(
        combined_markdown,
        encoding="utf-8",
    )
    print(
        f"Wrote {len(rows)} cross-dataset transfer experiments and "
        f"{len(shift_artifacts)} shift-diagnostic bundles with "
        f"{len(degradation)} degradation rows to {out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

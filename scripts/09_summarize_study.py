#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_experiments, load_paths
from health_misinfo.evaluation.bootstrap import paired_bootstrap_difference
from health_misinfo.evaluation.metrics import classification_metrics


MODELS = ("logistic_regression", "linear_svm")
CONTROLS = ("publisher_only_logistic", "length_only_logistic")


def _predictions(run_root: Path, experiment_id: str) -> pd.DataFrame:
    path = run_root / experiment_id / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _paired_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    joined = left.merge(
        right[
            ["record_id", "harmonized_label", "prediction", "score_unreliable"]
        ],
        on="record_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError("Paired comparison inputs do not contain identical record IDs")
    if "harmonized_label_right" in joined and not joined[
        "harmonized_label_left"
    ].equals(joined["harmonized_label_right"]):
        raise ValueError("Paired comparison inputs disagree on harmonized labels")
    true_column = (
        "harmonized_label_left"
        if "harmonized_label_left" in joined
        else "harmonized_label"
    )
    difference = paired_bootstrap_difference(
        joined[true_column],
        joined["prediction_left"],
        joined["prediction_right"],
        joined["score_unreliable_left"],
        joined["score_unreliable_right"],
        iterations=iterations,
        seed=seed,
        groups=joined["bootstrap_group"],
    )
    left_metrics = classification_metrics(
        joined[true_column],
        joined["prediction_left"],
        joined["score_unreliable_left"],
    )
    right_metrics = classification_metrics(
        joined[true_column],
        joined["prediction_right"],
        joined["score_unreliable_right"],
    )
    difference["observed_difference_left_minus_right"] = difference["metric"].map(
        lambda metric: left_metrics.get(metric, float("nan"))
        - right_metrics.get(metric, float("nan"))
    )
    return difference


def _equivalence_summary(results: pd.DataFrame) -> pd.DataFrame:
    macro = results[results["metric"].eq("macro_f1")].copy()
    rows = []
    for (dataset, model), frame in macro.groupby(["dataset", "model"], sort=True):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "runs": len(frame),
                "seeds": ",".join(
                    str(int(value))
                    for value in frame["seed"].dropna().sort_values().unique()
                ),
                "mean_observed_difference_standard_minus_masked": frame[
                    "observed_difference_left_minus_right"
                ].mean(),
                "minimum_observed_difference": frame[
                    "observed_difference_left_minus_right"
                ].min(),
                "maximum_observed_difference": frame[
                    "observed_difference_left_minus_right"
                ].max(),
                "equivalence_margin": frame["equivalence_margin"].iloc[0],
                "runs_supporting_practical_equivalence": int(
                    frame["supports_practical_equivalence"].sum()
                ),
                "all_runs_support_practical_equivalence": bool(
                    frame["supports_practical_equivalence"].all()
                ),
                "decision_rule": (
                    "all available run-specific 95% grouped-bootstrap macro-F1 "
                    "difference intervals (seed-specific for transformers) must lie "
                    "strictly inside the declared margin"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    paths = load_paths()
    config = load_experiments()
    latest = json.loads(
        (paths["experiments"] / "latest_run.json").read_text(encoding="utf-8")
    )
    run_root = Path(latest["run_path"])
    iterations = int(config["metrics"]["bootstrap_iterations"])
    seed = int(config["random_seed"])
    margin = float(config["metrics"]["artifact_equivalence_margin"])

    artifact_rows = []
    for dataset in ("coaid", "fakehealth"):
        for model in MODELS:
            standard_id = f"{dataset}_standard_{model}_model_text"
            masked_id = f"{dataset}_masked_{model}_model_text_masked"
            difference = _paired_difference(
                _predictions(run_root, standard_id),
                _predictions(run_root, masked_id),
                iterations,
                seed,
            )
            difference["equivalence_margin"] = margin
            difference["supports_practical_equivalence"] = (
                difference["metric"].eq("macro_f1")
                & difference["ci_low"].gt(-margin)
                & difference["ci_high"].lt(margin)
            )
            difference.insert(0, "test_records", len(_predictions(run_root, standard_id)))
            difference.insert(0, "seed", pd.NA)
            difference.insert(0, "comparison", "standard_minus_artifact_masked")
            difference.insert(0, "model", model)
            difference.insert(0, "dataset", dataset)
            artifact_rows.append(difference)

    transformer_root = run_root / "transformers"
    transformer_seeds = [
        int(value) for value in config["models"]["transformer_training"]["seeds"]
    ]
    for transformer in config["models"]["transformers"]:
        model = str(transformer["experiment_name"])
        model_root = transformer_root / model
        for dataset in ("coaid", "fakehealth"):
            for transformer_seed in transformer_seeds:
                standard_id = (
                    f"{dataset}_standard_{model}_model_text_seed{transformer_seed}"
                )
                masked_id = (
                    f"{dataset}_masked_{model}_model_text_masked_seed{transformer_seed}"
                )
                standard_predictions = _predictions(model_root, standard_id)
                difference = _paired_difference(
                    standard_predictions,
                    _predictions(model_root, masked_id),
                    iterations,
                    seed,
                )
                difference["equivalence_margin"] = margin
                difference["supports_practical_equivalence"] = (
                    difference["metric"].eq("macro_f1")
                    & difference["ci_low"].gt(-margin)
                    & difference["ci_high"].lt(margin)
                )
                difference.insert(0, "test_records", len(standard_predictions))
                difference.insert(0, "seed", transformer_seed)
                difference.insert(0, "comparison", "standard_minus_artifact_masked")
                difference.insert(0, "model", model)
                difference.insert(0, "dataset", dataset)
                artifact_rows.append(difference)

    artifact_results = pd.concat(artifact_rows, ignore_index=True)
    artifact_results.to_csv(
        run_root / "artifact_masking_paired_differences.csv",
        index=False,
    )
    _equivalence_summary(artifact_results).to_csv(
        run_root / "artifact_masking_equivalence_summary.csv",
        index=False,
    )

    control_rows = []
    controlled_evaluations = (
        ("coaid", "standard"),
        ("coaid", "controlled"),
        ("fakehealth", "standard"),
        ("fakehealth", "controlled"),
        ("fakehealth", "hosting_domain_sensitivity"),
    )
    for dataset, split in controlled_evaluations:
        experiment_id = f"{dataset}_{split}_logistic_regression_model_text"
        if not (run_root / experiment_id / "predictions.csv").exists():
            continue
        text = _predictions(run_root, experiment_id)
        text_config = yaml.safe_load(
            (run_root / experiment_id / "config.yaml").read_text(encoding="utf-8")
        )
        for control in CONTROLS:
            control_predictions = _predictions(
                run_root,
                f"{dataset}_{split}_{control}_"
                f"{'diagnostic_publisher' if control.startswith('publisher') else 'diagnostic_length'}",
            )
            difference = _paired_difference(
                text,
                control_predictions,
                iterations,
                seed,
            )
            difference.insert(0, "comparison", f"text_logistic_minus_{control}")
            difference.insert(
                0,
                "hosting_domain_disjoint_sensitivity",
                text_config["split_type"] == "hosting_domain_disjoint_sensitivity",
            )
            difference.insert(
                0,
                "canonical_publisher_disjoint",
                text_config["split_type"] == "canonical_publisher_disjoint",
            )
            difference.insert(0, "split_type", text_config["split_type"])
            difference.insert(0, "split", split)
            difference.insert(0, "dataset", dataset)
            control_rows.append(difference)
    pd.concat(control_rows, ignore_index=True).to_csv(
        run_root / "confirmatory_control_differences.csv",
        index=False,
    )
    print(f"Wrote paired study summaries to {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

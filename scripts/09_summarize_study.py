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
        right[["record_id", "prediction", "score_unreliable"]],
        on="record_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    return paired_bootstrap_difference(
        joined["harmonized_label"],
        joined["prediction_left"],
        joined["prediction_right"],
        joined["score_unreliable_left"],
        joined["score_unreliable_right"],
        iterations=iterations,
        seed=seed,
        groups=joined["bootstrap_group"],
    )


def main() -> int:
    paths = load_paths()
    config = load_experiments()
    latest = json.loads(
        (paths["experiments"] / "latest_run.json").read_text(encoding="utf-8")
    )
    run_root = Path(latest["run_path"])
    iterations = int(config["metrics"]["bootstrap_iterations"])
    seed = int(config["random_seed"])

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
            difference.insert(0, "comparison", "standard_minus_masked")
            difference.insert(0, "model", model)
            difference.insert(0, "dataset", dataset)
            artifact_rows.append(difference)
    pd.concat(artifact_rows, ignore_index=True).to_csv(
        run_root / "artifact_masking_paired_differences.csv",
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

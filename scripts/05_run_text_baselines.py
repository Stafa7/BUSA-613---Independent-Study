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
from health_misinfo.evaluation.bootstrap import bootstrap_intervals
from health_misinfo.evaluation.metrics import classification_metrics, confusion_matrix_frame, write_metrics
from health_misinfo.models.baselines import baseline_specs, fit_predict, top_features
from health_misinfo.paths import ensure_project_dirs


SPLITS = {
    "standard": "standard_split_manifest.csv",
    "controlled": "source_disjoint_split_manifest.csv",
    "masked": "topic_or_fallback_split_manifest.csv",
}


def _score_vector(score, pred) -> list[float]:
    if hasattr(score, "ndim") and score.ndim == 2:
        return score[:, 1].tolist()
    if score is None:
        return [1.0 if value == "unreliable" else 0.0 for value in pred]
    return list(score)


def _run_one(dataset: str, split_name: str, text_col: str, spec, data: pd.DataFrame, manifest: pd.DataFrame, out_root: Path, iterations: int, seed: int) -> None:
    joined = data.merge(manifest[["record_id", "split"]], on="record_id", how="inner")
    train = joined[joined["split"] == "train"]
    test = joined[joined["split"] == "test"]
    if train["harmonized_label"].nunique() < 2 or test["harmonized_label"].nunique() < 2:
        return
    pred, score = fit_predict(spec, train[text_col].fillna(""), train["harmonized_label"], test[text_col].fillna(""))
    score_vector = _score_vector(score, pred)
    metrics = classification_metrics(test["harmonized_label"], pred, score_vector)
    experiment_id = f"{dataset}_{split_name}_{spec.name}_{text_col}"
    out = out_root / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    predictions = test[["dataset", "record_id", "harmonized_label"]].copy()
    predictions["prediction"] = pred
    predictions["score_unreliable"] = score_vector
    predictions.to_parquet(out / "predictions.parquet", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    write_metrics(metrics, out)
    confusion_matrix_frame(test["harmonized_label"], pred).to_csv(out / "confusion_matrix.csv")
    bootstrap_intervals(
        test["harmonized_label"].to_numpy(),
        pred,
        score_vector,
        iterations=iterations,
        seed=seed,
    ).to_csv(out / "bootstrap_intervals.csv", index=False)
    top_features(spec.estimator).to_csv(out / "top_features.csv", index=False)
    config = {
        "dataset": dataset,
        "split": split_name,
        "model": spec.name,
        "text_column": text_col,
        "train_records": int(len(train)),
        "test_records": int(len(test)),
    }
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out / "notes.md").write_text(
        "# Experiment notes\n\nWithin-dataset text baseline using frozen split manifest. Labels are dataset-defined reliability mappings.\n",
        encoding="utf-8",
    )


def main() -> int:
    ensure_project_dirs()
    paths = load_paths()
    config = load_experiments()
    iterations = int(config["metrics"]["bootstrap_iterations"])
    seed = int(config["random_seed"])
    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[data["exclusion_reason"].fillna("") == ""].copy()
    summary_rows = []
    for split_name, file_name in SPLITS.items():
        manifest = pd.read_csv(paths["splits"] / file_name)
        text_col = "model_text_masked" if split_name == "masked" else "model_text"
        for dataset in sorted(data["dataset"].unique()):
            dataset_data = data[data["dataset"] == dataset]
            dataset_manifest = manifest[manifest["dataset"] == dataset]
            for spec in baseline_specs():
                _run_one(dataset, split_name, text_col, spec, dataset_data, dataset_manifest, paths["experiments"], iterations, seed)
    for metrics_path in sorted(paths["experiments"].glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        summary_rows.append({"experiment_id": metrics_path.parent.name, **metrics})
    pd.DataFrame(summary_rows).to_csv(paths["experiments"] / "text_baseline_summary.csv", index=False)
    print(f"Wrote text baseline experiments to {paths['experiments']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


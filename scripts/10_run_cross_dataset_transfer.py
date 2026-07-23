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
from health_misinfo.evaluation.metrics import classification_metrics
from health_misinfo.models.baselines import baseline_candidates, fit_predict


def _score_vector(score, prediction):
    if hasattr(score, "ndim") and score.ndim == 2:
        return score[:, 1]
    return score


def main() -> int:
    paths = load_paths()
    config = load_experiments()
    latest = json.loads(
        (paths["experiments"] / "latest_run.json").read_text(encoding="utf-8")
    )
    run_root = Path(latest["run_path"])
    out_root = run_root / "cross_dataset_transfer"
    out_root.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[data["primary_eligible"]].copy()
    manifest = pd.read_csv(paths["splits"] / "standard_split_manifest.csv")
    candidates = baseline_candidates(config)
    iterations = int(config["metrics"]["bootstrap_iterations"])
    seed = int(config["random_seed"])
    rows = []

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
        target_test = target[target["split"].eq("test")].copy()
        source_domains = set(
            source[source["split"].isin(["train", "validation"])]
            ["source_domain"]
            .fillna("")
            .loc[lambda values: values.ne("")]
        )
        target_test["overlapping_source_domain"] = target_test["source_domain"].isin(
            source_domains
        )
        target_test = target_test[~target_test["overlapping_source_domain"]].copy()

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
    print(f"Wrote {len(rows)} cross-dataset transfer diagnostics to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

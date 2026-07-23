#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_experiments, load_paths
from health_misinfo.evaluation.bootstrap import bootstrap_intervals
from health_misinfo.evaluation.metrics import classification_metrics, confusion_matrix_frame, write_metrics
from health_misinfo.paths import ensure_project_dirs


SPLITS = {
    "standard": ("standard_split_manifest.csv", "model_text"),
    "controlled": ("controlled_split_manifest.csv", "model_text"),
    "masked": ("artifact_masked_split_manifest.csv", "model_text_masked"),
}


def _write_stop_note(out: Path, reason: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    note = f"""# Transformer compute stop note

Generated: {datetime.now().isoformat(timespec="seconds")}

## Attempted environment

- Python: {platform.python_version()}
- Platform: {platform.platform()}
- Working directory: `{ROOT}`

## Stop reason

{reason}

## Study implication

The requested transformer comparison did not complete in this run. Traditional
text baselines remain available. Transformer results must not be inferred or
reported until this script completes on the frozen corrected manifests.
"""
    (out / "compute_stop_note.md").write_text(note, encoding="utf-8")
    (out / "status.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": reason,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ensure_project_dirs()
    paths = load_paths()
    config = load_experiments()
    latest_path = paths["experiments"] / "latest_run.json"
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        run_root = Path(latest["run_path"]) / "transformers"
    else:
        run_root = (
            paths["experiments"]
            / "transformer_runs"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
    stop_out = run_root / "compute_stop"
    try:
        import numpy as np
        from datasets import Dataset
        from sklearn.metrics import f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        _write_stop_note(stop_out, f"Required transformer dependency is unavailable: `{type(exc).__name__}: {exc}`.")
        print(f"Wrote transformer compute stop note to {stop_out}")
        return 2

    training = config["models"]["transformer_training"]
    model_specs = config["models"]["transformers"]
    seeds = [int(seed) for seed in training["seeds"]]
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    learning_rate = float(training["learning_rate"])
    patience = int(training["early_stopping_patience"])
    iterations = int(config["metrics"]["bootstrap_iterations"])
    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[data["primary_eligible"]].copy()
    labels = {"reliable": 0, "unreliable": 1}
    id_to_label = {0: "reliable", 1: "unreliable"}
    summary_rows: list[dict[str, object]] = []
    failures = 0

    for model_spec in model_specs:
        experiment_name = str(model_spec["experiment_name"])
        model_name = str(model_spec["model_name"])
        requested_revision = str(model_spec["revision"])
        max_length = int(model_spec["max_length"])
        model_root = run_root / experiment_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=requested_revision,
            )
            resolved_revision = getattr(tokenizer, "_commit_hash", None) or requested_revision

            for split_name, (manifest_file, text_col) in SPLITS.items():
                manifest_path = paths["splits"] / manifest_file
                if not manifest_path.exists():
                    continue
                manifest = pd.read_csv(manifest_path)
                if manifest.empty:
                    continue
                for dataset in sorted(manifest["dataset"].unique()):
                    joined = data[data["dataset"] == dataset].merge(
                        manifest[manifest["dataset"] == dataset][
                            ["record_id", "split", "split_type", "group_field"]
                        ],
                        on="record_id",
                        how="inner",
                    )
                    if joined.empty:
                        continue

                    def make_dataset(split: str) -> Dataset:
                        frame = joined[joined["split"] == split][
                            [
                                "record_id",
                                text_col,
                                "harmonized_label",
                                "source_name",
                                "publisher_id",
                                "family_group",
                                "text_unit",
                            ]
                        ].copy()
                        frame["label"] = frame["harmonized_label"].map(labels)
                        return Dataset.from_pandas(frame, preserve_index=False)

                    raw_train = make_dataset("train")
                    raw_validation = make_dataset("validation")
                    raw_test = make_dataset("test")

                    def tokenize(batch):
                        return tokenizer(
                            batch[text_col],
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                        )

                    train_ds = raw_train.map(tokenize, batched=True)
                    validation_ds = raw_validation.map(tokenize, batched=True)
                    test_ds = raw_test.map(tokenize, batched=True)
                    token_lengths = [
                        len(ids)
                        for ids in tokenizer(
                            joined[text_col].fillna("").tolist(),
                            add_special_tokens=True,
                            truncation=False,
                        )["input_ids"]
                    ]
                    truncation = {
                        "records": len(token_lengths),
                        "maximum_length": max_length,
                        "records_exceeding_maximum": int(
                            sum(length > max_length for length in token_lengths)
                        ),
                        "fraction_exceeding_maximum": float(
                            sum(length > max_length for length in token_lengths)
                            / len(token_lengths)
                        ),
                        "median_tokens": float(pd.Series(token_lengths).median()),
                        "p95_tokens": float(pd.Series(token_lengths).quantile(0.95)),
                    }

                    def compute_metrics(eval_pred):
                        logits, y_true = eval_pred
                        y_pred = np.argmax(logits, axis=1)
                        return {"macro_f1": f1_score(y_true, y_pred, average="macro")}

                    for seed in seeds:
                        experiment_id = (
                            f"{dataset}_{split_name}_{experiment_name}_{text_col}_seed{seed}"
                        )
                        experiment_out = model_root / experiment_id
                        experiment_out.mkdir(parents=True, exist_ok=False)
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_name,
                            revision=requested_revision,
                            num_labels=2,
                        )
                        model_revision = (
                            getattr(model.config, "_commit_hash", None) or resolved_revision
                        )
                        args = TrainingArguments(
                            output_dir=str(experiment_out / "checkpoints"),
                            num_train_epochs=epochs,
                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            learning_rate=learning_rate,
                            seed=seed,
                            data_seed=seed,
                            eval_strategy="epoch",
                            save_strategy="epoch",
                            load_best_model_at_end=True,
                            metric_for_best_model="macro_f1",
                            greater_is_better=True,
                            save_total_limit=1,
                            report_to=[],
                        )
                        trainer = Trainer(
                            model=model,
                            args=args,
                            train_dataset=train_ds,
                            eval_dataset=validation_ds,
                            processing_class=tokenizer,
                            compute_metrics=compute_metrics,
                            callbacks=[
                                EarlyStoppingCallback(
                                    early_stopping_patience=patience
                                )
                            ],
                        )
                        trainer.train()

                        def prediction_frame(raw_dataset, prediction_output) -> pd.DataFrame:
                            stable_logits = (
                                prediction_output.predictions
                                - prediction_output.predictions.max(axis=1, keepdims=True)
                            )
                            exponentiated = np.exp(stable_logits)
                            probabilities = exponentiated / exponentiated.sum(
                                axis=1,
                                keepdims=True,
                            )
                            predicted_ids = probabilities.argmax(axis=1)
                            frame = pd.DataFrame(
                                {
                                    "dataset": dataset,
                                    "record_id": raw_dataset["record_id"],
                                    "harmonized_label": raw_dataset["harmonized_label"],
                                    "source_name": raw_dataset["source_name"],
                                    "publisher_id": raw_dataset["publisher_id"],
                                    "family_group": raw_dataset["family_group"],
                                    "text_unit": raw_dataset["text_unit"],
                                    "prediction": [
                                        id_to_label[int(value)] for value in predicted_ids
                                    ],
                                    "score_unreliable": probabilities[:, 1],
                                }
                            )
                            return frame

                        validation_predictions = prediction_frame(
                            raw_validation,
                            trainer.predict(validation_ds),
                        )
                        validation_predictions.to_csv(
                            experiment_out / "validation_predictions.csv",
                            index=False,
                        )
                        predictions = prediction_frame(raw_test, trainer.predict(test_ds))
                        group_field = str(joined["group_field"].iloc[0])
                        group_lookup = joined.set_index("record_id")[group_field].astype(str)
                        predictions["bootstrap_group"] = predictions["record_id"].map(
                            group_lookup
                        )
                        predictions.to_parquet(
                            experiment_out / "predictions.parquet",
                            index=False,
                        )
                        predictions.to_csv(experiment_out / "predictions.csv", index=False)
                        metrics = classification_metrics(
                            predictions["harmonized_label"],
                            predictions["prediction"],
                            predictions["score_unreliable"],
                            probability_scores=True,
                        )
                        write_metrics(metrics, experiment_out)
                        confusion_matrix_frame(
                            predictions["harmonized_label"],
                            predictions["prediction"],
                        ).to_csv(experiment_out / "confusion_matrix.csv")
                        bootstrap_intervals(
                            predictions["harmonized_label"].to_numpy(),
                            predictions["prediction"].to_numpy(),
                            predictions["score_unreliable"].to_numpy(),
                            iterations=iterations,
                            seed=seed,
                            groups=predictions["bootstrap_group"].to_numpy(),
                            probability_scores=True,
                        ).to_csv(experiment_out / "bootstrap_intervals.csv", index=False)
                        run_config = {
                            "experiment_id": experiment_id,
                            "dataset": dataset,
                            "split_name": split_name,
                            "split_type": str(joined["split_type"].iloc[0]),
                            "group_field": group_field,
                            "experiment_name": experiment_name,
                            "model_name": model_name,
                            "requested_model_revision": requested_revision,
                            "resolved_model_revision": model_revision,
                            "seed": seed,
                            "maximum_length": max_length,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "early_stopping_patience": patience,
                            "train_records": len(raw_train),
                            "validation_records": len(raw_validation),
                            "test_records": len(raw_test),
                            "truncation_audit": truncation,
                            "runtime": {
                                "python": platform.python_version(),
                                "platform": platform.platform(),
                            },
                        }
                        (experiment_out / "config.yaml").write_text(
                            yaml.safe_dump(run_config, sort_keys=False),
                            encoding="utf-8",
                        )
                        pd.DataFrame(trainer.state.log_history).to_csv(
                            experiment_out / "training_curve.csv",
                            index=False,
                        )
                        (experiment_out / "status.json").write_text(
                            json.dumps({"status": "completed"}, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        summary_rows.append(
                            {
                                "experiment_id": experiment_id,
                                "experiment_name": experiment_name,
                                "dataset": dataset,
                                "split": split_name,
                                "seed": seed,
                                **metrics,
                            }
                        )
        except Exception as exc:
            failures += 1
            failure_out = model_root / "failed"
            _write_stop_note(
                failure_out,
                f"{experiment_name} failed: `{type(exc).__name__}: {exc}`.",
            )
            print(f"Wrote transformer failure status to {failure_out}")

    run_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(run_root / "transformer_summary.csv", index=False)
    (run_root / "status.json").write_text(
        json.dumps(
            {
                "status": "completed" if failures == 0 else "partial_failure",
                "models_requested": len(model_specs),
                "models_failed": failures,
                "experiments_completed": len(summary_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summary_rows)} transformer experiments to {run_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

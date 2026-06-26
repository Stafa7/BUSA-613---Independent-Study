#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_experiments, load_paths
from health_misinfo.evaluation.bootstrap import bootstrap_intervals
from health_misinfo.evaluation.metrics import classification_metrics, confusion_matrix_frame, write_metrics
from health_misinfo.paths import ensure_project_dirs


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

Phase 6 DistilBERT training did not complete locally in this run. Phase 5 traditional text baselines remain the completed first-half analytical milestone. The repository contains a runnable transformer entrypoint and dependency specification; rerun this script after installing `torch`, `transformers`, `datasets`, and `evaluate`.
"""
    (out / "compute_stop_note.md").write_text(note, encoding="utf-8")


def main() -> int:
    ensure_project_dirs()
    paths = load_paths()
    config = load_experiments()
    out = paths["experiments"] / "transformer_compute_stop"
    try:
        import numpy as np
        from datasets import Dataset
        from sklearn.metrics import f1_score
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:
        _write_stop_note(out, f"Required transformer dependency is unavailable: `{type(exc).__name__}: {exc}`.")
        print(f"Wrote transformer compute stop note to {out}")
        return 0
    try:
        model_name = config["models"]["transformer"]["model_name"]
        seed = int(config["models"]["transformer"]["primary_seed"])
        max_length = int(config["models"]["transformer"]["max_length"])
        epochs = int(config["models"]["transformer"]["epochs"])
        batch_size = int(config["models"]["transformer"]["batch_size"])
        iterations = int(config["metrics"]["bootstrap_iterations"])
        data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
        data = data[data["exclusion_reason"].fillna("") == ""].copy()
        manifest = pd.read_csv(paths["splits"] / "standard_split_manifest.csv")
        labels = {"reliable": 0, "unreliable": 1}
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for dataset in sorted(data["dataset"].unique()):
            joined = data[data["dataset"] == dataset].merge(
                manifest[manifest["dataset"] == dataset][["record_id", "split"]],
                on="record_id",
                how="inner",
            )
            if joined.empty:
                continue
            exp_id = f"{dataset}_standard_distilbert_model_text_seed{seed}"
            exp_out = paths["experiments"] / exp_id
            exp_out.mkdir(parents=True, exist_ok=True)

            def make_dataset(split: str) -> Dataset:
                frame = joined[joined["split"] == split][["record_id", "model_text", "harmonized_label"]].copy()
                frame["label"] = frame["harmonized_label"].map(labels)
                return Dataset.from_pandas(frame, preserve_index=False)

            train_ds = make_dataset("train")
            val_ds = make_dataset("validation")
            test_ds = make_dataset("test")

            def tokenize(batch):
                return tokenizer(batch["model_text"], padding="max_length", truncation=True, max_length=max_length)

            train_ds = train_ds.map(tokenize, batched=True)
            val_ds = val_ds.map(tokenize, batched=True)
            test_ds = test_ds.map(tokenize, batched=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

            def compute_metrics(eval_pred):
                logits, y_true = eval_pred
                y_pred = np.argmax(logits, axis=1)
                return {"macro_f1": f1_score(y_true, y_pred, average="macro")}

            args = TrainingArguments(
                output_dir=str(exp_out / "checkpoints"),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=2e-5,
                seed=seed,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="macro_f1",
                report_to=[],
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            raw = trainer.predict(test_ds)
            probabilities = np.exp(raw.predictions) / np.exp(raw.predictions).sum(axis=1, keepdims=True)
            pred_ids = probabilities.argmax(axis=1)
            id_to_label = {0: "reliable", 1: "unreliable"}
            predictions = pd.DataFrame(
                {
                    "dataset": dataset,
                    "record_id": test_ds["record_id"],
                    "harmonized_label": test_ds["harmonized_label"],
                    "prediction": [id_to_label[int(value)] for value in pred_ids],
                    "score_unreliable": probabilities[:, 1],
                }
            )
            predictions.to_parquet(exp_out / "predictions.parquet", index=False)
            predictions.to_csv(exp_out / "predictions.csv", index=False)
            metrics = classification_metrics(
                predictions["harmonized_label"], predictions["prediction"], predictions["score_unreliable"]
            )
            write_metrics(metrics, exp_out)
            confusion_matrix_frame(predictions["harmonized_label"], predictions["prediction"]).to_csv(
                exp_out / "confusion_matrix.csv"
            )
            bootstrap_intervals(
                predictions["harmonized_label"].to_numpy(),
                predictions["prediction"].to_numpy(),
                predictions["score_unreliable"].to_numpy(),
                iterations=iterations,
                seed=seed,
            ).to_csv(exp_out / "bootstrap_intervals.csv", index=False)
            (exp_out / "compute_note.md").write_text(
                f"# Transformer compute note\n\nModel `{model_name}` trained locally for dataset `{dataset}` using primary seed `{seed}`.\n",
                encoding="utf-8",
            )
            pd.DataFrame(trainer.state.log_history).to_csv(exp_out / "training_curve.csv", index=False)
        print(f"Wrote transformer experiments to {paths['experiments']}")
    except Exception as exc:
        _write_stop_note(out, f"DistilBERT training could not complete locally: `{type(exc).__name__}: {exc}`.")
        print(f"Wrote transformer compute stop note to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

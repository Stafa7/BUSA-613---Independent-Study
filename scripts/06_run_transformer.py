#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    "hosting_domain_sensitivity": (
        "hosting_domain_disjoint_sensitivity_manifest.csv",
        "model_text",
    ),
    "masked": ("artifact_masked_split_manifest.csv", "model_text_masked"),
}


def _write_stop_note(
    out: Path,
    reason: str,
    *,
    gate: str,
    details: dict[str, object] | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    details = details or {}
    detail_lines = "\n".join(
        f"- {key.replace('_', ' ').title()}: {value}" for key, value in details.items()
    )
    if detail_lines:
        detail_lines = f"\n## Gate details\n\n{detail_lines}\n"
    note = f"""# Transformer compute stop note

Generated: {datetime.now().isoformat(timespec="seconds")}

## Attempted environment

- Python: {platform.python_version()}
- Platform: {platform.platform()}
- Working directory: `{ROOT}`

## Stop reason

{reason}
{detail_lines}

## Study implication

The requested transformer comparison did not complete in this run. Traditional
text baselines remain available. Transformer results must not be inferred or
reported until this script completes on the frozen corrected manifests.
"""
    (out / "compute_stop_note.md").write_text(note, encoding="utf-8")
    (out / "status.json").write_text(
        json.dumps(
            {
                "status": "stopped",
                "gate": gate,
                "reason": reason,
                "details": details,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _planned_experiment_count(
    paths: dict[str, Path],
    model_specs: list[dict[str, object]],
    seeds: list[int],
    requested_splits: set[str] | None = None,
    requested_datasets: set[str] | None = None,
) -> tuple[int, list[dict[str, object]]]:
    """Count requested model/split/dataset/seed runs without loading model weights."""

    manifest_plan: list[dict[str, object]] = []
    dataset_split_count = 0
    for split_name, (manifest_file, _) in SPLITS.items():
        if requested_splits is not None and split_name not in requested_splits:
            continue
        manifest_path = paths["splits"] / manifest_file
        datasets: list[str] = []
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path, usecols=["dataset"])
            datasets = sorted(manifest["dataset"].dropna().astype(str).unique().tolist())
            if requested_datasets is not None:
                datasets = [value for value in datasets if value in requested_datasets]
        dataset_split_count += len(datasets)
        manifest_plan.append(
            {
                "split": split_name,
                "manifest": manifest_file,
                "datasets": datasets,
            }
        )
    return len(model_specs) * len(seeds) * dataset_split_count, manifest_plan


def _accelerator_status(
    torch_module,
    preference: list[str] | tuple[str, ...] = ("cuda", "mps"),
) -> dict[str, object]:
    """Detect accelerators and select the first available configured backend."""

    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(torch_module.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    mps_built = bool(mps_backend and mps_backend.is_built())
    availability = {"cuda": cuda_available, "mps": mps_available}
    normalized_preference = [str(value).strip().lower() for value in preference]
    if not normalized_preference or any(
        value not in availability for value in normalized_preference
    ):
        raise ValueError(
            "accelerator_preference must contain only 'cuda' and/or 'mps'"
        )
    backend = next(
        (value for value in normalized_preference if availability[value]),
        "cpu",
    )
    return {
        "backend": backend,
        "device": "cuda:0" if backend == "cuda" else backend,
        "accelerator_preference": normalized_preference,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch_module.cuda.device_count()) if cuda_available else 0,
        "mps_available": mps_available,
        "mps_built": mps_built,
    }


def _training_device_options(accelerator: dict[str, object]) -> dict[str, bool]:
    """Return TrainingArguments options that preserve the selected backend."""

    backend = str(accelerator["backend"])
    return {
        "use_cpu": backend == "cpu",
        # Pinned host-memory transfers benefit CUDA but are not useful on MPS.
        "dataloader_pin_memory": backend == "cuda",
    }


def _verify_trainer_device(trainer, expected_backend: str) -> str:
    """Fail before training if Transformers silently selected another device."""

    device = trainer.args.device
    actual_backend = str(getattr(device, "type", device)).split(":", 1)[0]
    if actual_backend != expected_backend:
        raise RuntimeError(
            "Trainer device mismatch: "
            f"detected {expected_backend!r}, but Transformers selected "
            f"{str(device)!r}"
        )
    return str(device)


def _run_transformer_tuning(
    *,
    run_root: Path,
    paths: dict[str, Path],
    config: dict,
    accelerator: dict[str, object],
    dependencies: dict[str, object],
) -> dict[str, object]:
    """Select one shared learning rate without consulting either test partition."""

    tuning = config["models"].get("transformer_tuning", {})
    training = config["models"]["transformer_training"]
    if not tuning.get("enabled", False):
        return {
            "learning_rate": float(training["learning_rate"]),
            "selection_source": "fixed_transformer_training_config",
        }

    tuning_root = run_root / "transformer_tuning"
    tuning_root.mkdir(parents=True, exist_ok=True)
    selection_path = tuning_root / "selected_hyperparameters.yaml"
    if selection_path.exists():
        selected = yaml.safe_load(selection_path.read_text(encoding="utf-8")) or {}
        if "learning_rate" not in selected:
            raise ValueError(f"Invalid transformer tuning artifact: {selection_path}")
        return selected

    np = dependencies["np"]
    torch = dependencies["torch"]
    Dataset = dependencies["Dataset"]
    f1_score = dependencies["f1_score"]
    AutoModelForSequenceClassification = dependencies[
        "AutoModelForSequenceClassification"
    ]
    AutoTokenizer = dependencies["AutoTokenizer"]
    DataCollatorWithPadding = dependencies["DataCollatorWithPadding"]
    EarlyStoppingCallback = dependencies["EarlyStoppingCallback"]
    Trainer = dependencies["Trainer"]
    TrainingArguments = dependencies["TrainingArguments"]

    reference_name = str(tuning["reference_model"])
    model_specs = {
        str(spec["experiment_name"]): spec
        for spec in config["models"]["transformers"]
    }
    if reference_name not in model_specs:
        raise ValueError(f"Unknown transformer_tuning reference_model: {reference_name}")
    model_spec = model_specs[reference_name]
    model_name = str(model_spec["model_name"])
    revision = str(model_spec["revision"])
    max_length = int(model_spec["max_length"])
    split_name = str(tuning["split"])
    if split_name not in SPLITS:
        raise ValueError(f"Unknown transformer tuning split: {split_name}")
    manifest_name, text_col = SPLITS[split_name]
    manifest = pd.read_csv(paths["splits"] / manifest_name)
    requested_datasets = [str(value) for value in tuning["datasets"]]
    seed = int(tuning["seed"])
    learning_rates = [float(value) for value in tuning["learning_rate_grid"]]
    if not learning_rates:
        raise ValueError("transformer_tuning.learning_rate_grid cannot be empty")
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    patience = int(training["early_stopping_patience"])
    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[data["primary_eligible"]].copy()
    labels = {"reliable": 0, "unreliable": 1}
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trial_rows: list[dict[str, object]] = []

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(y_true, y_pred, average="macro")}

    for learning_rate in learning_rates:
        for dataset in requested_datasets:
            trial_id = f"{reference_name}_{dataset}_lr{learning_rate:.0e}"
            trial_out = tuning_root / "trials" / trial_id
            metrics_path = trial_out / "validation_metrics.json"
            if metrics_path.exists():
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                trial_rows.append(saved)
                continue
            dataset_manifest = manifest[manifest["dataset"].eq(dataset)]
            joined = data[data["dataset"].eq(dataset)].merge(
                dataset_manifest[["record_id", "split"]],
                on="record_id",
                how="inner",
            )
            if joined.empty:
                raise ValueError(f"No records for transformer tuning dataset {dataset}")

            def make_dataset(split: str):
                frame = joined[joined["split"].eq(split)][
                    ["record_id", text_col, "harmonized_label"]
                ].copy()
                frame[text_col] = frame[text_col].fillna("")
                frame["label"] = frame["harmonized_label"].map(labels)
                return Dataset.from_pandas(frame, preserve_index=False)

            train_ds = make_dataset("train")
            validation_ds = make_dataset("validation")

            def tokenize(batch):
                return tokenizer(
                    batch[text_col],
                    truncation=True,
                    max_length=max_length,
                )

            train_ds = train_ds.map(tokenize, batched=True)
            validation_ds = validation_ds.map(tokenize, batched=True)
            trial_out.mkdir(parents=True, exist_ok=True)
            checkpoint_attempt = datetime.now(timezone.utc).strftime(
                "checkpoints_%Y%m%dT%H%M%SZ"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                revision=revision,
                num_labels=2,
            )
            args = TrainingArguments(
                output_dir=str(trial_out / checkpoint_attempt),
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
                save_only_model=True,
                report_to=[],
                disable_tqdm=True,
                **_training_device_options(accelerator),
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=validation_ds,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
            )
            trainer_device = _verify_trainer_device(
                trainer,
                expected_backend=str(accelerator["backend"]),
            )
            trainer.train()
            evaluation = trainer.evaluate(validation_ds)
            best_checkpoint = trainer.state.best_model_checkpoint
            best_step = (
                int(Path(best_checkpoint).name.removeprefix("checkpoint-"))
                if best_checkpoint
                else None
            )
            best_epoch_logs = [
                entry
                for entry in trainer.state.log_history
                if entry.get("step") == best_step
                and "eval_macro_f1" in entry
            ]
            best_epoch = (
                float(best_epoch_logs[0]["epoch"])
                if best_epoch_logs
                else float(trainer.state.epoch or epochs)
            )
            row = {
                "trial_id": trial_id,
                "reference_model": reference_name,
                "dataset": dataset,
                "split": split_name,
                "seed": seed,
                "learning_rate": learning_rate,
                "validation_macro_f1": float(evaluation["eval_macro_f1"]),
                "best_epoch": best_epoch,
                "trainer_device": trainer_device,
                "test_records_examined": 0,
            }
            metrics_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            pd.DataFrame(trainer.state.log_history).to_csv(
                trial_out / "training_curve.csv",
                index=False,
            )
            trial_rows.append(row)
            del trainer, model
            if accelerator["backend"] == "cuda":
                torch.cuda.empty_cache()
            elif accelerator["backend"] == "mps":
                torch.mps.empty_cache()

    trial_frame = pd.DataFrame(trial_rows)
    trial_frame.to_csv(tuning_root / "validation_trials.csv", index=False)
    aggregate = (
        trial_frame.groupby("learning_rate", as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            mean_validation_macro_f1=("validation_macro_f1", "mean"),
            minimum_validation_macro_f1=("validation_macro_f1", "min"),
        )
    )
    aggregate.to_csv(tuning_root / "learning_rate_summary.csv", index=False)
    mean_scores = aggregate.set_index("learning_rate")[
        "mean_validation_macro_f1"
    ].to_dict()
    selected_rate = max(learning_rates, key=lambda value: mean_scores[value])
    selected = {
        "learning_rate": selected_rate,
        "selection_source": "validation_only_transformer_tuning",
        "reference_model": reference_name,
        "split": split_name,
        "datasets": requested_datasets,
        "seed": seed,
        "selection_metric": str(tuning["selection_metric"]),
        "selected_mean_validation_macro_f1": float(mean_scores[selected_rate]),
        "candidate_learning_rates": learning_rates,
        "test_records_examined": 0,
    }
    selection_path.write_text(
        yaml.safe_dump(selected, sort_keys=False),
        encoding="utf-8",
    )
    (tuning_root / "status.json").write_text(
        json.dumps({"status": "completed", **selected}, indent=2) + "\n",
        encoding="utf-8",
    )
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the pinned transformer comparison on the frozen manifests."
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Explicitly permit the full transformer grid without CUDA or MPS. "
            "This can be very slow and is disabled by default."
        ),
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Run or reuse validation-only transformer tuning, then stop before test evaluation.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional transformer experiment names to run (for resumable chunks).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLITS),
        help="Optional split names to run (for resumable chunks).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset names to run (for resumable chunks).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional configured seeds to run (for resumable chunks).",
    )
    args = parser.parse_args(argv)

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
        import torch
        from datasets import Dataset
        from sklearn.metrics import f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        _write_stop_note(
            stop_out,
            f"Required transformer dependency is unavailable: `{type(exc).__name__}: {exc}`.",
            gate="dependency_import",
            details={"python": platform.python_version()},
        )
        print(f"Wrote transformer compute stop note to {stop_out}")
        return 2

    training = config["models"]["transformer_training"]
    model_specs = config["models"]["transformers"]
    seeds = [int(seed) for seed in training["seeds"]]
    if args.models:
        requested_models = set(args.models)
        model_specs = [
            spec
            for spec in model_specs
            if str(spec["experiment_name"]) in requested_models
        ]
        missing_models = requested_models - {
            str(spec["experiment_name"]) for spec in model_specs
        }
        if missing_models:
            raise ValueError(f"Unknown transformer models: {sorted(missing_models)}")
    if args.seeds:
        unknown_seeds = set(args.seeds) - set(seeds)
        if unknown_seeds:
            raise ValueError(f"Seeds are not in the frozen config: {sorted(unknown_seeds)}")
        seeds = list(args.seeds)
    requested_splits = set(args.splits or SPLITS)
    requested_datasets = set(args.datasets) if args.datasets else None
    planned_experiments, manifest_plan = _planned_experiment_count(
        paths,
        model_specs,
        seeds,
        requested_splits=requested_splits,
        requested_datasets=requested_datasets,
    )
    accelerator = _accelerator_status(
        torch,
        preference=training.get("accelerator_preference", ["cuda", "mps"]),
    )
    allow_cpu = bool(training.get("allow_cpu", False) or args.allow_cpu)
    if accelerator["backend"] == "cpu" and not allow_cpu:
        _write_stop_note(
            stop_out,
            (
                "No CUDA or Apple MPS accelerator is available. The full pinned "
                "transformer grid is intentionally blocked on CPU; rerun in a suitable "
                "accelerated environment, or pass `--allow-cpu` to accept the cost."
            ),
            gate="accelerator_required",
            details={
                **accelerator,
                "planned_experiments": planned_experiments,
                "expected_experiments": planned_experiments,
                "completed_experiments": 0,
                "models": len(model_specs),
                "seeds": len(seeds),
                "manifest_plan": manifest_plan,
                "cpu_override_used": False,
            },
        )
        print(f"Wrote transformer compute stop note to {stop_out}")
        return 3

    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    patience = int(training["early_stopping_patience"])
    iterations = int(config["metrics"]["bootstrap_iterations"])
    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[data["primary_eligible"]].copy()
    labels = {"reliable": 0, "unreliable": 1}
    id_to_label = {0: "reliable", 1: "unreliable"}
    summary_rows: list[dict[str, object]] = []
    failures = 0

    selected_tuning = _run_transformer_tuning(
        run_root=run_root,
        paths=paths,
        config=config,
        accelerator=accelerator,
        dependencies={
            "np": np,
            "torch": torch,
            "Dataset": Dataset,
            "f1_score": f1_score,
            "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
            "AutoTokenizer": AutoTokenizer,
            "DataCollatorWithPadding": DataCollatorWithPadding,
            "EarlyStoppingCallback": EarlyStoppingCallback,
            "Trainer": Trainer,
            "TrainingArguments": TrainingArguments,
        },
    )
    learning_rate = float(selected_tuning["learning_rate"])
    if args.tune_only:
        print(
            "Transformer tuning completed; selected learning rate "
            f"{learning_rate:g} in {run_root / 'transformer_tuning'}"
        )
        return 0

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
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            resolved_revision = getattr(tokenizer, "_commit_hash", None) or requested_revision

            for split_name, (manifest_file, text_col) in SPLITS.items():
                if split_name not in requested_splits:
                    continue
                manifest_path = paths["splits"] / manifest_file
                if not manifest_path.exists():
                    continue
                manifest = pd.read_csv(manifest_path)
                if manifest.empty:
                    continue
                for dataset in sorted(manifest["dataset"].unique()):
                    if requested_datasets is not None and dataset not in requested_datasets:
                        continue
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
                        frame[text_col] = frame[text_col].fillna("")
                        frame["label"] = frame["harmonized_label"].map(labels)
                        return Dataset.from_pandas(frame, preserve_index=False)

                    raw_train = make_dataset("train")
                    raw_validation = make_dataset("validation")
                    raw_test = make_dataset("test")

                    def tokenize(batch):
                        return tokenizer(
                            batch[text_col],
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
                        status_path = experiment_out / "status.json"
                        metrics_path = experiment_out / "metrics.json"
                        if status_path.exists() and metrics_path.exists():
                            status = json.loads(status_path.read_text(encoding="utf-8"))
                            if status.get("status") == "completed":
                                summary_rows.append(
                                    {
                                        "experiment_id": experiment_id,
                                        "experiment_name": experiment_name,
                                        "dataset": dataset,
                                        "split": split_name,
                                        "seed": seed,
                                        **json.loads(metrics_path.read_text(encoding="utf-8")),
                                    }
                                )
                                continue
                        experiment_out.mkdir(parents=True, exist_ok=True)
                        checkpoint_attempt = datetime.now(timezone.utc).strftime(
                            "checkpoints_%Y%m%dT%H%M%SZ"
                        )
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_name,
                            revision=requested_revision,
                            num_labels=2,
                        )
                        model_revision = (
                            getattr(model.config, "_commit_hash", None) or resolved_revision
                        )
                        args = TrainingArguments(
                            output_dir=str(experiment_out / checkpoint_attempt),
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
                            save_only_model=True,
                            report_to=[],
                            disable_tqdm=True,
                            **_training_device_options(accelerator),
                        )
                        trainer = Trainer(
                            model=model,
                            args=args,
                            train_dataset=train_ds,
                            eval_dataset=validation_ds,
                            processing_class=tokenizer,
                            data_collator=data_collator,
                            compute_metrics=compute_metrics,
                            callbacks=[
                                EarlyStoppingCallback(
                                    early_stopping_patience=patience
                                )
                            ],
                        )
                        trainer_device = _verify_trainer_device(
                            trainer,
                            expected_backend=str(accelerator["backend"]),
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
                            "hyperparameter_selection": selected_tuning,
                            "early_stopping_patience": patience,
                            "train_records": len(raw_train),
                            "validation_records": len(raw_validation),
                            "test_records": len(raw_test),
                            "truncation_audit": truncation,
                            "runtime": {
                                "python": platform.python_version(),
                                "platform": platform.platform(),
                                "accelerator": accelerator,
                                "trainer_device": trainer_device,
                                "cpu_override_used": bool(
                                    accelerator["backend"] == "cpu" and allow_cpu
                                ),
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
                gate="model_execution",
                details={
                    "experiment_name": experiment_name,
                    "accelerator": accelerator,
                },
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
                "accelerator": accelerator,
                "cpu_override_used": bool(
                    accelerator["backend"] == "cpu" and allow_cpu
                ),
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

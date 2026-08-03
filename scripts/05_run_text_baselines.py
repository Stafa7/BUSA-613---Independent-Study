#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd
import sklearn
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_experiments, load_paths
from health_misinfo.evaluation.bootstrap import bootstrap_intervals, paired_bootstrap_difference
from health_misinfo.evaluation.metrics import classification_metrics, confusion_matrix_frame, write_metrics
from health_misinfo.features.leakage import is_artifact_feature
from health_misinfo.models.baselines import baseline_candidates, fit_predict, top_features
from health_misinfo.paths import ensure_project_dirs


SPLITS = {
    "standard": "standard_split_manifest.csv",
    "controlled": "controlled_split_manifest.csv",
    "hosting_domain_sensitivity": "hosting_domain_disjoint_sensitivity_manifest.csv",
    "temporal": "temporal_split_manifest.csv",
    "masked": "artifact_masked_split_manifest.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return f"{revision}-dirty" if dirty else revision
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _source_state_hash() -> str:
    digest = hashlib.sha256()
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        digest.update(diff)
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        for raw_path in sorted(path for path in untracked if path):
            path = ROOT / raw_path.decode("utf-8")
            digest.update(raw_path)
            if path.is_file():
                digest.update(path.read_bytes())
        return digest.hexdigest()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _score_vector(score, pred) -> list[float]:
    if hasattr(score, "ndim") and score.ndim == 2:
        return score[:, 1].tolist()
    if score is None:
        return [1.0 if value == "unreliable" else 0.0 for value in pred]
    return list(score)


def _fit_and_score(spec, train, evaluation, text_col: str) -> tuple[object, list[float], dict]:
    prediction, score = fit_predict(
        spec,
        train[text_col].fillna(""),
        train["harmonized_label"],
        evaluation[text_col].fillna(""),
    )
    score_vector = _score_vector(score, prediction)
    metrics = classification_metrics(
        evaluation["harmonized_label"],
        prediction,
        score_vector,
        probability_scores=spec.probability_scores,
    )
    return prediction, score_vector, metrics


def _run_one(
    dataset: str,
    split_name: str,
    text_col: str,
    model_name: str,
    specs,
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    out_root: Path,
    iterations: int,
    seed: int,
    processed_hash: str,
    code_revision: str,
    source_state_hash: str,
) -> tuple[dict, pd.DataFrame]:
    joined = data.merge(
        manifest[
            [
                "record_id",
                "split",
                "split_type",
                "group_field",
            ]
        ],
        on="record_id",
        how="inner",
    )
    train = joined[joined["split"] == "train"].copy()
    validation = joined[joined["split"] == "validation"].copy()
    test = joined[joined["split"] == "test"].copy()
    if any(frame["harmonized_label"].nunique() < 2 for frame in (train, validation, test)):
        raise ValueError(f"{dataset}/{split_name}/{model_name} lacks both classes in a required split")

    validation_rows = []
    candidate_evaluations = Parallel(
        n_jobs=min(4, len(specs)),
        prefer="processes",
    )(
        delayed(_fit_and_score)(spec, train, validation, text_col)
        for spec in specs
    )
    fitted = []
    for candidate_index, (spec, evaluation) in enumerate(
        zip(specs, candidate_evaluations, strict=True)
    ):
        validation_prediction, validation_score, validation_metrics = evaluation
        validation_rows.append(
            {
                "candidate_index": candidate_index,
                **spec.hyperparameters,
                **validation_metrics,
            }
        )
        fitted.append((spec, validation_prediction, validation_score, validation_metrics))
    selected_index = max(
        range(len(fitted)),
        key=lambda index: (fitted[index][3]["macro_f1"], -index),
    )
    selected, validation_prediction, validation_score, validation_metrics = fitted[selected_index]

    test_prediction, test_score = fit_predict(
        selected,
        train[text_col].fillna(""),
        train["harmonized_label"],
        test[text_col].fillna(""),
    )
    test_score_vector = _score_vector(test_score, test_prediction)
    metrics = classification_metrics(
        test["harmonized_label"],
        test_prediction,
        test_score_vector,
        probability_scores=selected.probability_scores,
    )

    experiment_id = f"{dataset}_{split_name}_{model_name}_{text_col}"
    out = out_root / experiment_id
    out.mkdir(parents=True, exist_ok=True)

    group_field = str(test["group_field"].iloc[0])
    prediction_columns = [
        "dataset",
        "record_id",
        "harmonized_label",
        "source_name",
        "publisher_id",
        "family_group",
        "duplicate_group",
        "text_unit",
    ]
    predictions = test[prediction_columns].copy()
    predictions["prediction"] = test_prediction
    predictions["score_unreliable"] = test_score_vector
    predictions["bootstrap_group"] = test[group_field].astype(str).to_numpy()
    predictions["analysis_cohort"] = dataset
    predictions.to_parquet(out / "predictions.parquet", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)

    validation_predictions = validation[prediction_columns].copy()
    if group_field not in validation_predictions.columns:
        validation_predictions[group_field] = validation[group_field].astype(str).to_numpy()
    validation_predictions["prediction"] = validation_prediction
    validation_predictions["score_unreliable"] = validation_score
    validation_predictions["analysis_cohort"] = dataset
    validation_predictions.to_csv(out / "validation_predictions.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(out / "validation_tuning.csv", index=False)

    subgroup_rows = []
    for text_unit, subgroup in predictions.groupby("text_unit"):
        subgroup_metrics = classification_metrics(
            subgroup["harmonized_label"],
            subgroup["prediction"],
            subgroup["score_unreliable"],
            probability_scores=selected.probability_scores,
        )
        subgroup_rows.append(
            {
                "subgroup_field": "text_unit",
                "subgroup_value": text_unit,
                "both_classes_present": subgroup["harmonized_label"].nunique() == 2,
                **subgroup_metrics,
            }
        )
    pd.DataFrame(subgroup_rows).to_csv(out / "subgroup_metrics.csv", index=False)

    write_metrics(metrics, out)
    confusion_matrix_frame(test["harmonized_label"], test_prediction).to_csv(out / "confusion_matrix.csv")
    bootstrap_intervals(
        test["harmonized_label"].to_numpy(),
        test_prediction,
        test_score_vector,
        iterations=iterations,
        seed=seed,
        groups=predictions["bootstrap_group"].to_numpy(),
        probability_scores=selected.probability_scores,
    ).to_csv(out / "bootstrap_intervals.csv", index=False)

    features = top_features(selected.estimator)
    if len(features):
        features["artifact_flag"] = features["feature"].map(is_artifact_feature)
    else:
        features["artifact_flag"] = pd.Series(dtype=bool)
    features.to_csv(out / "top_features.csv", index=False)
    artifact_share = float(features["artifact_flag"].mean()) if len(features) else 0.0
    pd.DataFrame(
        [
            {
                "top_features_reviewed": len(features),
                "artifact_flagged_features": int(features["artifact_flag"].sum()) if len(features) else 0,
                "artifact_feature_share": artifact_share,
                "artifact_warning": artifact_share >= 0.20,
            }
        ]
    ).to_csv(out / "artifact_feature_audit.csv", index=False)

    run_config = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "split_name": split_name,
        "split_type": str(test["split_type"].iloc[0]),
        "group_field": group_field,
        "model": model_name,
        "selected_hyperparameters": selected.hyperparameters,
        "text_column": text_col,
        "random_seed": seed,
        "train_records": int(len(train)),
        "validation_records": int(len(validation)),
        "test_records": int(len(test)),
        "validation_macro_f1": float(validation_metrics["macro_f1"]),
        "bootstrap_iterations": iterations,
        "processed_data_sha256": processed_hash,
        "code_revision": code_revision,
        "source_state_sha256": source_state_hash,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (out / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    (out / "notes.md").write_text(
        "# Experiment notes\n\n"
        "Within-dataset text baseline using a frozen split manifest and validation-selected "
        "hyperparameters. Labels are dataset-defined reliability mappings. Confidence "
        "intervals resample the manifest's highest independent grouping unit.\n",
        encoding="utf-8",
    )
    return {"experiment_id": experiment_id, **metrics}, predictions


def _run_controlled_repeat_sensitivity(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    inputs = {
        "text_logistic": "model_text",
        "publisher_only_logistic": "diagnostic_publisher",
        "length_only_logistic": "diagnostic_length",
    }

    def run_repeat(dataset, repeat_id, repeat_manifest) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        logistic_specs = baseline_candidates(config)["logistic_regression"]
        joined = data[data["dataset"].eq(dataset)].merge(
            repeat_manifest[["record_id", "split", "split_seed"]],
            on="record_id",
            how="inner",
        )
        train = joined[joined["split"].eq("train")]
        validation = joined[joined["split"].eq("validation")]
        test = joined[joined["split"].eq("test")]
        for model_name, text_col in inputs.items():
            fitted = [
                (
                    spec,
                    _fit_and_score(spec, train, validation, text_col)[2],
                )
                for spec in logistic_specs
            ]
            selected, validation_metrics = max(
                fitted,
                key=lambda item: item[1]["macro_f1"],
            )
            test_prediction, test_score = fit_predict(
                selected,
                train[text_col].fillna(""),
                train["harmonized_label"],
                test[text_col].fillna(""),
            )
            metrics = classification_metrics(
                test["harmonized_label"],
                test_prediction,
                _score_vector(test_score, test_prediction),
                probability_scores=True,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "repeat_id": int(repeat_id),
                    "split_seed": int(repeat_manifest["split_seed"].iloc[0]),
                    "model": model_name,
                    "train_records": len(train),
                    "validation_records": len(validation),
                    "test_records": len(test),
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    **metrics,
                }
            )
        return rows

    tasks = [
        (dataset, repeat_id, repeat_manifest.copy())
        for (dataset, repeat_id), repeat_manifest in manifest.groupby(["dataset", "repeat_id"])
    ]
    completed = Parallel(n_jobs=min(4, len(tasks)), prefer="threads")(
        delayed(run_repeat)(dataset, repeat_id, repeat_manifest)
        for dataset, repeat_id, repeat_manifest in tasks
    )
    return pd.DataFrame([row for repeat_rows in completed for row in repeat_rows])


def _prepare_analysis_data(processed_path: Path) -> pd.DataFrame:
    data = pd.read_parquet(processed_path)
    data = data[data["primary_eligible"]].copy()
    data["diagnostic_publisher"] = (
        "publisher_" + data["publisher_id"].fillna("missing").astype(str)
    )
    data["diagnostic_unit"] = "unit_" + data["text_unit"].fillna("missing").astype(str)
    word_count = data["body_word_count"].fillna(0).astype(int).clip(lower=0)
    data["diagnostic_length"] = "length_bin_" + (
        (word_count + 1).map(lambda value: int(value).bit_length() - 1).astype(str)
    )
    return data


def _write_repeat_sensitivity(
    data: pd.DataFrame,
    paths: dict[str, Path],
    config: dict,
    run_root: Path,
) -> None:
    repeats_path = paths["splits"] / "controlled_split_repeats.csv"
    if not repeats_path.exists():
        return
    repeated_manifest = pd.read_csv(repeats_path)
    if repeated_manifest.empty:
        return
    repeat_results = _run_controlled_repeat_sensitivity(
        data,
        repeated_manifest,
        config,
    )
    repeat_results.to_csv(
        run_root / "controlled_repeat_sensitivity.csv",
        index=False,
    )
    (
        repeat_results.groupby(["dataset", "model"])
        .agg(
            repeats=("repeat_id", "nunique"),
            unique_test_assignments=("split_seed", "nunique"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            minimum_macro_f1=("macro_f1", "min"),
            maximum_macro_f1=("macro_f1", "max"),
            mean_unreliable_f1=("unreliable_f1", "mean"),
        )
        .reset_index()
        .to_csv(
            run_root / "controlled_repeat_sensitivity_summary.csv",
            index=False,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeats-only",
        action="store_true",
        help="Regenerate only repeated publisher-disjoint sensitivity outputs for the latest run.",
    )
    args = parser.parse_args()
    ensure_project_dirs()
    paths = load_paths()
    config = load_experiments()
    iterations = int(config["metrics"]["bootstrap_iterations"])
    seed = int(config["random_seed"])
    processed_path = paths["data_processed"] / "combined_items.parquet"
    processed_hash = _sha256(processed_path)
    data = _prepare_analysis_data(processed_path)
    if args.repeats_only:
        latest_path = paths["experiments"] / "latest_run.json"
        if not latest_path.exists():
            raise FileNotFoundError("No latest baseline run exists")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        run_root = Path(latest["run_path"])
        _write_repeat_sensitivity(data, paths, config, run_root)
        print(f"Regenerated controlled repeat sensitivity outputs in {run_root}")
        return 0
    revision = _code_revision()
    source_state_hash = _source_state_hash()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{processed_hash[:8]}_{source_state_hash[:8]}"
    )
    run_root = paths["experiments"] / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    candidates = baseline_candidates(config)
    summary_rows = []
    prediction_sets: dict[tuple[str, str, str], pd.DataFrame] = {}

    for split_name, file_name in SPLITS.items():
        manifest_path = paths["splits"] / file_name
        if not manifest_path.exists():
            continue
        manifest = pd.read_csv(manifest_path)
        if manifest.empty:
            continue
        for dataset in sorted(manifest["dataset"].unique()):
            if split_name == "temporal":
                controlled_manifest = pd.read_csv(
                    paths["splits"] / "controlled_split_manifest.csv"
                )
                selected = controlled_manifest[
                    controlled_manifest["dataset"].eq(dataset)
                ]
                if (
                    len(selected)
                    and selected["split_type"].iloc[0] == "temporal_family_disjoint"
                ):
                    continue
            dataset_data = data[data["dataset"] == dataset]
            dataset_manifest = manifest[manifest["dataset"] == dataset]
            cohorts = {dataset: dataset_data}
            if dataset == "fakehealth" and split_name in {"standard", "controlled", "temporal"}:
                cohorts["fakehealth_rating3_excluded"] = dataset_data[
                    dataset_data["native_label"] != "rating_3"
                ].copy()
            for cohort_name, cohort_data in cohorts.items():
                experiment_plan = [
                    ("majority", "model_text", candidates["majority"]),
                    (
                        "logistic_regression",
                        "model_text_masked" if split_name == "masked" else "model_text",
                        candidates["logistic_regression"],
                    ),
                    (
                        "linear_svm",
                        "model_text_masked" if split_name == "masked" else "model_text",
                        candidates["linear_svm"],
                    ),
                ]
                if split_name != "masked":
                    experiment_plan.extend(
                        [
                            ("publisher_only_logistic", "diagnostic_publisher", candidates["logistic_regression"]),
                            ("unit_only_logistic", "diagnostic_unit", candidates["logistic_regression"]),
                            ("length_only_logistic", "diagnostic_length", candidates["logistic_regression"]),
                            ("title_only_logistic", "title", candidates["logistic_regression"]),
                            ("body_only_logistic", "body_text", candidates["logistic_regression"]),
                        ]
                    )
                for model_name, text_col, specs in experiment_plan:
                    summary, predictions = _run_one(
                        cohort_name,
                        split_name,
                        text_col,
                        model_name,
                        specs,
                        cohort_data,
                        dataset_manifest,
                        run_root,
                        iterations,
                        seed,
                        processed_hash,
                        revision,
                        source_state_hash,
                    )
                    summary_rows.append(summary)
                    prediction_sets[(cohort_name, split_name, model_name)] = predictions

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(run_root / "text_baseline_summary.csv", index=False)

    paired_rows = []
    analysis_cohorts = sorted({key[0] for key in prediction_sets})
    for dataset in analysis_cohorts:
        for split_name in SPLITS:
            available = {
                key[2]: predictions
                for key, predictions in prediction_sets.items()
                if key[0] == dataset and key[1] == split_name
            }
            for model_a, model_b in combinations(sorted(available), 2):
                a = available[model_a]
                b = available[model_b]
                joined = a.merge(
                    b[["record_id", "prediction", "score_unreliable"]],
                    on="record_id",
                    suffixes=("_a", "_b"),
                    validate="one_to_one",
                )
                difference = paired_bootstrap_difference(
                    joined["harmonized_label"],
                    joined["prediction_a"],
                    joined["prediction_b"],
                    joined["score_unreliable_a"],
                    joined["score_unreliable_b"],
                    iterations=iterations,
                    seed=seed,
                    groups=joined["bootstrap_group"],
                )
                difference.insert(0, "model_b", model_b)
                difference.insert(0, "model_a", model_a)
                difference.insert(0, "split", split_name)
                difference.insert(0, "dataset", dataset)
                paired_rows.append(difference)
    if paired_rows:
        pd.concat(paired_rows, ignore_index=True).to_csv(
            run_root / "paired_model_differences.csv",
            index=False,
        )

    rating3_rows = []
    rating3_difference_rows = []
    non_rating3_records = set(
        data.loc[
            data["dataset"].eq("fakehealth") & data["native_label"].ne("rating_3"),
            "record_id",
        ]
    )
    for split_name in ("standard", "controlled"):
        for model_name in ("logistic_regression", "linear_svm"):
            primary_key = ("fakehealth", split_name, model_name)
            retrained_key = ("fakehealth_rating3_excluded", split_name, model_name)
            if primary_key not in prediction_sets or retrained_key not in prediction_sets:
                continue
            primary = prediction_sets[primary_key]
            primary = primary[primary["record_id"].isin(non_rating3_records)]
            retrained = prediction_sets[retrained_key]
            common = primary.merge(
                retrained[["record_id", "prediction", "score_unreliable"]],
                on="record_id",
                suffixes=("_primary", "_retrained"),
                validate="one_to_one",
            )
            for training_condition, prediction_col, score_col in (
                ("rating3_in_training", "prediction_primary", "score_unreliable_primary"),
                ("rating3_excluded_from_training", "prediction_retrained", "score_unreliable_retrained"),
            ):
                rating3_rows.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "training_condition": training_condition,
                        "common_test_records": len(common),
                        **classification_metrics(
                            common["harmonized_label"],
                            common[prediction_col],
                            common[score_col],
                            probability_scores=model_name == "logistic_regression",
                        ),
                    }
                )
            difference = paired_bootstrap_difference(
                common["harmonized_label"],
                common["prediction_primary"],
                common["prediction_retrained"],
                common["score_unreliable_primary"],
                common["score_unreliable_retrained"],
                iterations=iterations,
                seed=seed,
                groups=common["bootstrap_group"],
            )
            difference.insert(0, "comparison", "rating3_in_training_minus_excluded")
            difference.insert(0, "model", model_name)
            difference.insert(0, "split", split_name)
            rating3_difference_rows.append(difference)
    pd.DataFrame(rating3_rows).to_csv(
        run_root / "rating3_common_test_sensitivity.csv",
        index=False,
    )
    if rating3_difference_rows:
        pd.concat(rating3_difference_rows, ignore_index=True).to_csv(
            run_root / "rating3_common_test_paired_differences.csv",
            index=False,
        )

    _write_repeat_sensitivity(data, paths, config, run_root)

    run_metadata = {
        "run_id": run_id,
        "processed_data_sha256": processed_hash,
        "code_revision": revision,
        "source_state_sha256": source_state_hash,
        "bootstrap_iterations": iterations,
        "experiments_completed": len(summary_frame),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (run_root / "baseline_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (paths["experiments"] / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "run_path": str(run_root)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summary_frame)} corrected text baseline experiments to {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

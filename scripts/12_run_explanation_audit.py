#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_experiments, load_paths
from health_misinfo.evaluation.explanations import (
    compare_standard_controlled_explanations,
    explain_linear_tfidf,
)
from health_misinfo.models.baselines import baseline_candidates


MANIFESTS = {
    "standard": "standard_split_manifest.csv",
    "controlled": "controlled_split_manifest.csv",
}
MODELS = ("logistic_regression", "linear_svm")
REFIT_SCORE_ATOL = 1e-5


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


def _selected_estimator(config: dict, model_name: str, run_config: dict):
    selected_hyperparameters = run_config["selected_hyperparameters"]
    candidates = baseline_candidates(config)[model_name]
    matching = [
        candidate
        for candidate in candidates
        if candidate.hyperparameters == selected_hyperparameters
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Could not uniquely reconstruct {model_name} with "
            f"hyperparameters={selected_hyperparameters}"
        )
    return clone(matching[0].estimator)


def _unreliable_score(estimator, texts: pd.Series) -> np.ndarray:
    classes = list(estimator.named_steps["clf"].classes_)
    unreliable_index = classes.index("unreliable")
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(texts)[:, unreliable_index], dtype=float)
    score = np.asarray(estimator.decision_function(texts), dtype=float).reshape(-1)
    return score if unreliable_index == 1 else -score


def main() -> int:
    paths = load_paths()
    config = load_experiments()
    latest_path = paths["experiments"] / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError("No baseline latest_run.json is available; run script 05 first")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    run_root = Path(latest["run_path"])
    out = run_root / "explanation_audit"
    out.mkdir(parents=True, exist_ok=True)
    audit_path = run_root / "error_audit" / "qualitative_error_audit_template.csv"
    if not audit_path.exists():
        _write_status(
            out,
            "gated_missing_audit_sample",
            reason="Run scripts 07 and 08 before generating local explanations.",
        )
        return 2

    audit = pd.read_csv(audit_path, dtype={"record_id": str, "case_id": str})
    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet")
    data = data[data["primary_eligible"]].copy()
    audit_context_columns = [
        column
        for column in ("case_id", "record_id", "selection_reasons", "outcomes")
        if column in audit.columns
    ]
    audit_context = audit[audit_context_columns].drop_duplicates("record_id")

    summaries: list[pd.DataFrame] = []
    contributions: list[pd.DataFrame] = []
    verification_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for dataset in sorted(audit["dataset"].dropna().astype(str).unique()):
        cases = audit_context.merge(
            data[data["dataset"].eq(dataset)],
            on="record_id",
            how="inner",
            validate="one_to_one",
        )
        for split_name, manifest_name in MANIFESTS.items():
            manifest = pd.read_csv(
                paths["splits"] / manifest_name,
                dtype={"record_id": str},
            )
            manifest = manifest[manifest["dataset"].eq(dataset)].copy()
            if manifest.empty:
                failures.append(f"{dataset}/{split_name}: manifest unavailable")
                continue
            membership = manifest.set_index("record_id")["split"].astype(str)
            train_ids = set(manifest.loc[manifest["split"].eq("train"), "record_id"])
            train = data[
                data["dataset"].eq(dataset) & data["record_id"].isin(train_ids)
            ].copy()
            for model_name in MODELS:
                experiment_id = f"{dataset}_{split_name}_{model_name}_model_text"
                experiment_dir = run_root / experiment_id
                run_config_path = experiment_dir / "config.yaml"
                predictions_path = experiment_dir / "predictions.csv"
                if not (run_config_path.exists() and predictions_path.exists()):
                    failures.append(f"{experiment_id}: baseline artifacts unavailable")
                    continue
                run_config = yaml.safe_load(
                    run_config_path.read_text(encoding="utf-8")
                )
                text_column = str(run_config["text_column"])
                estimator = _selected_estimator(config, model_name, run_config)
                estimator.fit(
                    train[text_column].fillna(""),
                    train["harmonized_label"],
                )

                saved = pd.read_csv(predictions_path, dtype={"record_id": str})
                verification = saved.merge(
                    data[data["dataset"].eq(dataset)][["record_id", text_column]],
                    on="record_id",
                    how="left",
                    validate="one_to_one",
                )
                refit_prediction = estimator.predict(
                    verification[text_column].fillna("")
                )
                refit_score = _unreliable_score(
                    estimator,
                    verification[text_column].fillna(""),
                )
                prediction_agreement = float(
                    np.mean(refit_prediction == verification["prediction"].to_numpy())
                )
                maximum_score_difference = float(
                    np.max(
                        np.abs(
                            refit_score
                            - verification["score_unreliable"].to_numpy(dtype=float)
                        )
                    )
                )
                verified = (
                    prediction_agreement == 1.0
                    and maximum_score_difference <= REFIT_SCORE_ATOL
                )
                verification_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset": dataset,
                        "split_name": split_name,
                        "model": model_name,
                        "records": len(verification),
                        "prediction_agreement": prediction_agreement,
                        "maximum_score_difference": maximum_score_difference,
                        "verified": verified,
                    }
                )
                if not verified:
                    failures.append(
                        f"{experiment_id}: refit did not reproduce frozen predictions"
                    )
                    continue

                case_summary, case_contributions = explain_linear_tfidf(
                    estimator,
                    cases[text_column],
                    record_ids=cases["record_id"],
                    top_per_direction=20,
                )
                context = cases.set_index("record_id")
                for frame in (case_summary, case_contributions):
                    frame.insert(0, "experiment_id", experiment_id)
                    frame.insert(1, "dataset", dataset)
                    frame.insert(2, "split_name", split_name)
                    frame.insert(3, "split_type", str(manifest["split_type"].iloc[0]))
                    frame.insert(4, "model", model_name)
                    frame["case_id"] = frame["record_id"].map(context["case_id"])
                    if "selection_reasons" in context:
                        frame["selection_reasons"] = frame["record_id"].map(
                            context["selection_reasons"]
                        )
                    frame["record_manifest_split"] = frame["record_id"].map(membership)
                case_summary["harmonized_label"] = case_summary["record_id"].map(
                    context["harmonized_label"]
                )
                summaries.append(case_summary)
                contributions.append(case_contributions)

    verification_frame = pd.DataFrame(verification_rows)
    verification_frame.to_csv(out / "refit_verification.csv", index=False)
    if not summaries:
        _write_status(
            out,
            "gated_no_verified_models",
            failures=failures,
            experiments_attempted=len(verification_rows),
        )
        return 2

    summary_frame = pd.concat(summaries, ignore_index=True)
    contribution_frame = pd.concat(contributions, ignore_index=True)
    summary_frame.to_csv(out / "case_model_explanations.csv", index=False)
    contribution_frame.to_csv(
        out / "local_feature_contributions.csv",
        index=False,
    )
    stability = compare_standard_controlled_explanations(
        contribution_frame,
        summary_frame,
        top_k=10,
    )
    stability.to_csv(out / "standard_controlled_stability.csv", index=False)
    if len(stability):
        stability_summary_rows = []
        for (dataset, model), group in stability.groupby(["dataset", "model"]):
            held_out = group[
                group.get(
                    "both_models_held_out",
                    pd.Series(False, index=group.index),
                ).fillna(False)
            ]
            stability_summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "records_compared_descriptively": group["record_id"].nunique(),
                    "both_models_held_out_records": held_out["record_id"].nunique(),
                    "held_out_mean_feature_jaccard": (
                        held_out["feature_jaccard"].mean()
                        if len(held_out)
                        else np.nan
                    ),
                    "held_out_median_feature_jaccard": (
                        held_out["feature_jaccard"].median()
                        if len(held_out)
                        else np.nan
                    ),
                    "held_out_mean_overlap_sign_agreement": (
                        held_out["overlap_sign_agreement"].mean()
                        if len(held_out)
                        else np.nan
                    ),
                    "held_out_mean_top_contribution_cosine": (
                        held_out["top_contribution_cosine"].mean()
                        if len(held_out)
                        else np.nan
                    ),
                    "held_out_prediction_agreement": (
                        held_out["prediction_agreement"].mean()
                        if len(held_out)
                        else np.nan
                    ),
                }
            )
        stability_summary = pd.DataFrame(stability_summary_rows)
    else:
        stability_summary = pd.DataFrame()
    stability_summary.to_csv(out / "stability_summary.csv", index=False)
    _write_status(
        out,
        "completed" if not failures else "partial_failure",
        failures=failures,
        verified_models=int(verification_frame["verified"].sum()),
        cases=int(summary_frame["record_id"].nunique()),
        local_explanation_rows=len(contribution_frame),
        stability_records=len(stability),
        contribution_definition="signed TF-IDF value multiplied by the unreliable-oriented linear coefficient",
        refit_verification=(
            "All frozen predictions must agree. Decision scores may differ by at most "
            f"{REFIT_SCORE_ATOL:g} to allow floating-point variation in linear solvers."
        ),
        stability_interpretation=(
            "Descriptive same-record comparison only; no single local explanation "
            "is treated as stable or causal evidence."
        ),
    )
    print(f"Wrote linear-model explanation audit to {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

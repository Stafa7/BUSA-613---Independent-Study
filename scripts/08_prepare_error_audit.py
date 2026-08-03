#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_paths


def _outcome(row: pd.Series) -> str:
    truth = row["harmonized_label"]
    prediction = row["calibrated_prediction"]
    if truth == "unreliable" and prediction == "unreliable":
        return "true_positive"
    if truth == "reliable" and prediction == "reliable":
        return "true_negative"
    if truth == "reliable" and prediction == "unreliable":
        return "false_positive"
    return "false_negative"


def _select_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["outcome"] = predictions.apply(_outcome, axis=1)
    selections = []
    errors = predictions[predictions["calibrated_error"]].sort_values("confidence", ascending=False).head(10).copy()
    errors["selection_reason"] = "high_confidence_error"
    selections.append(errors)
    uncertain = predictions.sort_values("confidence").head(10).copy()
    uncertain["selection_reason"] = "lowest_confidence_case"
    selections.append(uncertain)
    for outcome, group in predictions.groupby("outcome"):
        sample = group.sort_values("confidence", ascending=False).head(5).copy()
        sample["selection_reason"] = f"stratified_{outcome}"
        selections.append(sample)
        random_sample = group.sample(n=min(10, len(group)), random_state=613).copy()
        random_sample["selection_reason"] = f"random_{outcome}"
        selections.append(random_sample)
    return pd.concat(selections, ignore_index=True)


def main() -> int:
    paths = load_paths()
    data = pd.read_parquet(paths["data_processed"] / "combined_items.parquet").set_index("record_id")
    latest_path = paths["experiments"] / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError("No baseline latest_run.json is available; run scripts 05 and 07 first")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    run_root = Path(latest["run_path"])
    out = run_root / "error_audit"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        config_path = experiment_dir / "config.yaml"
        calibrated_path = experiment_dir / "calibration_selective" / "test_calibrated_predictions.csv"
        if not (config_path.exists() and calibrated_path.exists()):
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if "rating3_excluded" in config.get("experiment_id", ""):
            continue
        predictions = pd.read_csv(calibrated_path)
        selected = _select_cases(predictions)
        selected["experiment_id"] = config["experiment_id"]
        rows.append(selected)
    if not rows:
        raise RuntimeError("No calibrated predictions are available for audit selection")
    raw_audit = pd.concat(rows, ignore_index=True)
    aggregation = (
        raw_audit.groupby("record_id")
        .agg(
            experiment_ids=("experiment_id", lambda values: "|".join(sorted(set(values)))),
            selection_reasons=("selection_reason", lambda values: "|".join(sorted(set(values)))),
            outcomes=("outcome", lambda values: "|".join(sorted(set(values)))),
        )
    )
    audit = (
        raw_audit.sort_values(["confidence", "record_id"], ascending=[False, True])
        .drop_duplicates("record_id")
        .drop(columns=["experiment_id", "selection_reason", "outcome"])
        .join(aggregation, on="record_id")
    )
    audit["case_id"] = [f"case_{index:04d}" for index in range(1, len(audit) + 1)]
    audit["title"] = audit["record_id"].map(data["title"])
    audit["text_excerpt"] = audit["record_id"].map(data["model_text"]).fillna("").str.slice(0, 600)
    for reviewer in ("reviewer_a", "reviewer_b", "adjudicated"):
        audit[f"{reviewer}_error_code"] = ""
        audit[f"{reviewer}_review_notes"] = ""
        audit[f"{reviewer}_health_relevant_signal"] = ""
        audit[f"{reviewer}_source_or_dataset_artifact"] = ""
        audit[f"{reviewer}_label_ambiguity"] = ""
    columns = [
        "case_id",
        "experiment_ids",
        "selection_reasons",
        "outcomes",
        "dataset",
        "record_id",
        "text_unit",
        "source_name",
        "harmonized_label",
        "calibrated_prediction",
        "calibrated_probability_unreliable",
        "confidence",
        "title",
        "text_excerpt",
        "reviewer_a_error_code",
        "reviewer_a_review_notes",
        "reviewer_a_health_relevant_signal",
        "reviewer_a_source_or_dataset_artifact",
        "reviewer_a_label_ambiguity",
        "reviewer_b_error_code",
        "reviewer_b_review_notes",
        "reviewer_b_health_relevant_signal",
        "reviewer_b_source_or_dataset_artifact",
        "reviewer_b_label_ambiguity",
        "adjudicated_error_code",
        "adjudicated_review_notes",
        "adjudicated_health_relevant_signal",
        "adjudicated_source_or_dataset_artifact",
        "adjudicated_label_ambiguity",
    ]
    audit[columns].to_csv(out / "qualitative_error_audit_template.csv", index=False)
    blinded_columns = [
        "case_id",
        "dataset",
        "text_unit",
        "source_name",
        "title",
        "text_excerpt",
        *[column for column in columns if column.startswith(("reviewer_", "adjudicated_"))],
    ]
    audit[blinded_columns].to_csv(
        out / "qualitative_error_audit_blinded.csv",
        index=False,
    )
    audit[
        [
            "case_id",
            "record_id",
            "experiment_ids",
            "selection_reasons",
            "outcomes",
            "harmonized_label",
            "calibrated_prediction",
        ]
    ].to_csv(out / "qualitative_error_audit_key.csv", index=False)
    instructions = """# Qualitative error-audit instructions

The tables include a purposive failure-discovery sample and a reproducible
outcome-stratified random sample. They are not a completed qualitative analysis.
Two reviewers should independently code `qualitative_error_audit_blinded.csv`,
then calculate agreement and adjudicate disagreements before opening the key.
Do not infer medical truth solely from the dataset label or model output.

Required categories to consider:

- source or publisher artifact;
- topic shortcut;
- temporal or event-specific cue;
- label ambiguity;
- text-unit mismatch;
- insufficient context;
- discussion or debunking mistaken for endorsement;
- medical terminology misunderstanding;
- sensational or emotional framing;
- truncation or preprocessing failure;
- engagement proxy or missingness artifact; and
- unexplained residual error.
"""
    (out / "README.md").write_text(instructions, encoding="utf-8")
    print(f"Wrote {len(audit)} qualitative audit candidates to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from health_misinfo.config import load_paths
from health_misinfo.evaluation.calibration import decision_confidence
from health_misinfo.evaluation.qualitative import (
    AUDIT_PROTOCOL_VERSION,
    AUDIT_QUOTAS,
    AUDIT_RANDOM_SEED,
    AUDIT_ROSTER_FIELD,
    ERROR_CODES,
    STANCE_VALUES,
    select_frozen_audit_sample,
)


TRANSFER_PATTERN = re.compile(
    r"^(coaid|fakehealth)_to_(coaid|fakehealth)_(.+)$"
)


def _outcome(truth: pd.Series, prediction: pd.Series) -> pd.Series:
    values = pd.Series("false_negative", index=truth.index, dtype="string")
    values.loc[truth.eq("unreliable") & prediction.eq("unreliable")] = (
        "true_positive"
    )
    values.loc[truth.eq("reliable") & prediction.eq("reliable")] = "true_negative"
    values.loc[truth.eq("reliable") & prediction.eq("unreliable")] = (
        "false_positive"
    )
    return values


def _coverage_threshold(experiment_dir: Path) -> float | None:
    coverage_path = (
        experiment_dir / "calibration_selective" / "fixed_coverage_results.csv"
    )
    if not coverage_path.exists():
        return None
    coverage = pd.read_csv(coverage_path)
    required = {"target_validation_coverage", "confidence_threshold"}
    if not required <= set(coverage.columns):
        return None
    target = coverage[
        coverage["target_validation_coverage"].sub(0.8).abs().lt(1e-9)
    ]
    if target.empty:
        return None
    return float(target.iloc[0]["confidence_threshold"])


def _baseline_candidates(run_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    experiment_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    for experiment_dir in experiment_dirs:
        config_path = experiment_dir / "config.yaml"
        calibrated_path = (
            experiment_dir
            / "calibration_selective"
            / "test_calibrated_predictions.csv"
        )
        calibration_selection_path = (
            experiment_dir
            / "calibration_selective"
            / "selected_calibration.json"
        )
        if not (
            config_path.exists()
            and calibrated_path.exists()
            and calibration_selection_path.exists()
        ):
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        experiment_id = str(config.get("experiment_id", experiment_dir.name))
        if "rating3_excluded" in experiment_id:
            continue
        predictions = pd.read_csv(
            calibrated_path,
            dtype={"record_id": str},
        )
        calibration_selection = json.loads(
            calibration_selection_path.read_text(encoding="utf-8")
        )
        decision_threshold = float(
            calibration_selection["primary_decision_threshold"]
        )
        confidence = decision_confidence(
            predictions["calibrated_probability_unreliable"].to_numpy(),
            decision_threshold=decision_threshold,
        )
        predictions["decision_threshold"] = decision_threshold
        predictions["decision_confidence"] = confidence
        predictions["confidence"] = confidence
        predictions["calibrated_error"] = predictions[
            "calibrated_prediction"
        ].ne(predictions["harmonized_label"])
        predictions["anchor_source"] = "calibrated_baseline"
        predictions["anchor_experiment_id"] = experiment_id
        predictions["anchor_prediction"] = predictions["calibrated_prediction"]
        predictions["anchor_outcome"] = _outcome(
            predictions["harmonized_label"],
            predictions["anchor_prediction"],
        )
        predictions["anchor_score_unreliable"] = predictions.get(
            "score_unreliable",
            pd.Series(float("nan"), index=predictions.index),
        )
        predictions["audit_split_name"] = str(config.get("split_name", ""))
        predictions["audit_model"] = str(config.get("model", ""))
        threshold = _coverage_threshold(experiment_dir)
        predictions["deferral_confidence_threshold"] = threshold
        predictions["deferred_at_80pct_target"] = (
            predictions["decision_confidence"].lt(threshold)
            if threshold is not None
            else False
        )
        rows.append(predictions)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _transfer_candidates(run_root: Path) -> pd.DataFrame:
    transfer_root = run_root / "cross_dataset_transfer"
    if not transfer_root.exists():
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    experiment_dirs = sorted(path for path in transfer_root.iterdir() if path.is_dir())
    for experiment_dir in experiment_dirs:
        match = TRANSFER_PATTERN.match(experiment_dir.name)
        predictions_path = experiment_dir / "predictions.csv"
        if match is None or not predictions_path.exists():
            continue
        source_dataset, target_dataset, _ = match.groups()
        predictions = pd.read_csv(predictions_path, dtype={"record_id": str})
        if not {"harmonized_label", "prediction"} <= set(predictions.columns):
            continue
        predictions = predictions[
            predictions["prediction"].ne(predictions["harmonized_label"])
        ].copy()
        if predictions.empty:
            continue
        predictions["anchor_source"] = "cross_dataset_transfer"
        predictions["anchor_experiment_id"] = experiment_dir.name
        predictions["anchor_prediction"] = predictions["prediction"]
        predictions["anchor_outcome"] = _outcome(
            predictions["harmonized_label"],
            predictions["anchor_prediction"],
        )
        predictions["anchor_score_unreliable"] = predictions.get(
            "score_unreliable",
            pd.Series(float("nan"), index=predictions.index),
        )
        predictions["transfer_source_dataset"] = source_dataset
        predictions["transfer_target_dataset"] = target_dataset
        predictions["decision_threshold"] = float("nan")
        predictions["decision_confidence"] = float("nan")
        predictions["confidence"] = float("nan")
        predictions["calibrated_probability_unreliable"] = float("nan")
        predictions["calibrated_error"] = True
        predictions["deferred_at_80pct_target"] = False
        rows.append(predictions)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _narrative_candidates(
    run_root: Path,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    assignments_path = (
        run_root
        / "narrative_validation"
        / "all_document_narrative_assignments.csv"
    )
    if not assignments_path.exists() or baseline.empty:
        return pd.DataFrame()
    assignments = pd.read_csv(assignments_path, dtype={"record_id": str})
    required = {"dataset", "record_id", "final_narrative_id"}
    if not required <= set(assignments.columns):
        return pd.DataFrame()
    narrative_columns = [
        column
        for column in (
            "dataset",
            "record_id",
            "final_narrative_id",
            "candidate_narrative_label",
            "assigned_probability",
        )
        if column in assignments
    ]
    assignments = assignments[narrative_columns].drop_duplicates(
        ["dataset", "record_id", "final_narrative_id"]
    )
    candidates = baseline.merge(
        assignments,
        on=["dataset", "record_id"],
        how="inner",
        validate="many_to_many",
    )
    if candidates.empty:
        return candidates
    assigned_probability = (
        candidates["assigned_probability"]
        if "assigned_probability" in candidates
        else pd.Series(0, index=candidates.index)
    )
    candidates["sampling_priority"] = -pd.to_numeric(
        assigned_probability,
        errors="coerce",
    ).fillna(0)
    candidates["balance_group"] = (
        candidates["dataset"].astype(str)
        + "|"
        + candidates["final_narrative_id"].astype(str)
    )
    return candidates


def _sampling_strata(
    baseline: pd.DataFrame,
    transfer: pd.DataFrame,
    narrative: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    strata: dict[str, pd.DataFrame] = {}
    for outcome in (
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ):
        sample = baseline[baseline["anchor_outcome"].eq(outcome)].copy()
        sample["sampling_priority"] = 0
        sample["balance_group"] = sample["dataset"].astype(str)
        strata[f"outcome_{outcome}"] = sample

    high_confidence = baseline[baseline["calibrated_error"]].copy()
    high_confidence["sampling_priority"] = -high_confidence[
        "decision_confidence"
    ]
    high_confidence["balance_group"] = high_confidence["dataset"].astype(str)
    strata["high_confidence_error"] = high_confidence

    deferred = baseline[baseline["deferred_at_80pct_target"]].copy()
    deferred["sampling_priority"] = deferred["decision_confidence"]
    deferred["balance_group"] = deferred["dataset"].astype(str)
    strata["deferred_at_80pct_target"] = deferred

    for source_dataset, target_dataset in (
        ("coaid", "fakehealth"),
        ("fakehealth", "coaid"),
    ):
        direction = transfer[
            transfer["transfer_source_dataset"].eq(source_dataset)
            & transfer["transfer_target_dataset"].eq(target_dataset)
        ].copy()
        direction["sampling_priority"] = 0
        direction["balance_group"] = direction["dataset"].astype(str)
        strata[
            f"transfer_error_{source_dataset}_to_{target_dataset}"
        ] = direction

    strata["retained_narrative_example"] = narrative
    return strata


def _aggregate_contexts(
    baseline: pd.DataFrame,
    transfer: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        frame[
            ["record_id", "anchor_experiment_id", "anchor_outcome"]
        ]
        for frame in (baseline, transfer)
        if not frame.empty
    ]
    if not frames:
        return pd.DataFrame(
            columns=["record_id", "experiment_ids", "outcomes"]
        )
    contexts = pd.concat(frames, ignore_index=True)
    return (
        contexts.groupby("record_id", as_index=False)
        .agg(
            experiment_ids=(
                "anchor_experiment_id",
                lambda values: "|".join(sorted(set(values))),
            ),
            outcomes=(
                "anchor_outcome",
                lambda values: "|".join(sorted(set(values))),
            ),
        )
    )


def _roster_hash(audit: pd.DataFrame, *, run_id: str) -> str:
    frozen_columns = [
        "case_id",
        "record_id",
        "selection_reason",
        "anchor_source",
        "anchor_experiment_id",
        "dataset",
        "harmonized_label",
        "anchor_prediction",
        "anchor_outcome",
        "anchor_score_unreliable",
        "decision_threshold",
        "decision_confidence",
        "title",
        "text_excerpt",
        "final_narrative_id",
        "candidate_narrative_label",
    ]
    rows = []
    for row in audit.reindex(columns=frozen_columns).itertuples(
        index=False,
        name=None,
    ):
        rows.append(
            {
                column: None if pd.isna(value) else str(value)
                for column, value in zip(frozen_columns, row, strict=True)
            }
        )
    payload = json.dumps(
        {"run_id": run_id, "rows": rows},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> int:
    paths = load_paths()
    latest_path = paths["experiments"] / "latest_run.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            "No baseline latest_run.json is available; run scripts 05 and 07 first"
        )
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    run_root = Path(latest["run_path"])
    run_id = str(latest.get("run_id", run_root.name))
    out = run_root / "error_audit"
    out.mkdir(parents=True, exist_ok=True)

    baseline = _baseline_candidates(run_root)
    if baseline.empty:
        raise RuntimeError(
            "No calibrated predictions are available for audit selection"
        )
    transfer = _transfer_candidates(run_root)
    narrative = _narrative_candidates(run_root, baseline)
    selection = select_frozen_audit_sample(
        _sampling_strata(baseline, transfer, narrative),
        quotas=AUDIT_QUOTAS,
        seed=AUDIT_RANDOM_SEED,
    )
    if selection.selected.empty:
        raise RuntimeError("The frozen sampling plan selected no audit cases")

    data = pd.read_parquet(
        paths["data_processed"] / "combined_items.parquet"
    ).set_index("record_id")
    audit = selection.selected.copy()
    audit = audit.merge(
        _aggregate_contexts(baseline, transfer),
        on="record_id",
        how="left",
        validate="many_to_one",
    )
    audit["case_id"] = [
        f"case_{index:04d}" for index in range(1, len(audit) + 1)
    ]
    audit["audit_protocol_version"] = AUDIT_PROTOCOL_VERSION
    # Preserve the plural context field consumed by the explanation-audit
    # workflow. Each frozen case has exactly one primary selection stratum.
    audit["selection_reasons"] = audit["selection_reason"]
    source_name = audit.get(
        "source_name",
        pd.Series("", index=audit.index),
    ).astype("string")
    missing_source = source_name.fillna("").str.strip().eq("")
    audit["source_name"] = source_name.mask(
        missing_source,
        audit["record_id"].map(data["source_name"]),
    )
    audit["title"] = audit["record_id"].map(data["title"])
    audit["text_excerpt"] = (
        audit["record_id"].map(data["model_text"]).fillna("").str.slice(0, 600)
    )
    for column in (
        "final_narrative_id",
        "candidate_narrative_label",
    ):
        if column not in audit:
            audit[column] = ""
    roster_sha256 = _roster_hash(audit, run_id=run_id)
    audit[AUDIT_ROSTER_FIELD] = roster_sha256
    for column in (
        "stage_a_health_relevant_signal",
        "stage_a_source_or_dataset_artifact",
        "stage_a_label_ambiguity",
        "stage_a_stance",
        "stage_a_notes",
        "stage_b_error_code",
        "stage_b_review_notes",
    ):
        audit[column] = ""

    template_columns = [
        "case_id",
        "audit_protocol_version",
        AUDIT_ROSTER_FIELD,
        "selection_order",
        "selection_reason",
        "selection_reasons",
        "anchor_source",
        "anchor_experiment_id",
        "experiment_ids",
        "outcomes",
        "dataset",
        "record_id",
        "text_unit",
        "source_name",
        "harmonized_label",
        "anchor_prediction",
        "anchor_outcome",
        "anchor_score_unreliable",
        "calibrated_probability_unreliable",
        "decision_threshold",
        "decision_confidence",
        "deferral_confidence_threshold",
        "deferred_at_80pct_target",
        "transfer_source_dataset",
        "transfer_target_dataset",
        "final_narrative_id",
        "candidate_narrative_label",
        "title",
        "text_excerpt",
        "stage_a_health_relevant_signal",
        "stage_a_source_or_dataset_artifact",
        "stage_a_label_ambiguity",
        "stage_a_stance",
        "stage_a_notes",
        "stage_b_error_code",
        "stage_b_review_notes",
    ]
    for column in template_columns:
        if column not in audit:
            audit[column] = ""
    audit[template_columns].to_csv(
        out / "qualitative_error_audit_template.csv",
        index=False,
    )

    stage_a_columns = [
        "case_id",
        "audit_protocol_version",
        AUDIT_ROSTER_FIELD,
        "dataset",
        "text_unit",
        "source_name",
        "title",
        "text_excerpt",
        "stage_a_health_relevant_signal",
        "stage_a_source_or_dataset_artifact",
        "stage_a_label_ambiguity",
        "stage_a_stance",
        "stage_a_notes",
    ]
    audit[stage_a_columns].to_csv(
        out / "qualitative_error_audit_stage_a_blinded.csv",
        index=False,
    )
    stage_b_columns = [
        "case_id",
        "audit_protocol_version",
        AUDIT_ROSTER_FIELD,
        "selection_reason",
        "dataset",
        "record_id",
        "title",
        "text_excerpt",
        "anchor_source",
        "anchor_experiment_id",
        "experiment_ids",
        "outcomes",
        "harmonized_label",
        "anchor_prediction",
        "anchor_outcome",
        "anchor_score_unreliable",
        "calibrated_probability_unreliable",
        "decision_threshold",
        "decision_confidence",
        "deferral_confidence_threshold",
        "deferred_at_80pct_target",
        "transfer_source_dataset",
        "transfer_target_dataset",
        "final_narrative_id",
        "candidate_narrative_label",
        "stage_b_error_code",
        "stage_b_review_notes",
    ]
    audit[stage_b_columns].to_csv(
        out / "qualitative_error_audit_stage_b_revealed.csv",
        index=False,
    )
    selection.quota_manifest.to_csv(
        out / "qualitative_error_audit_sampling_manifest.csv",
        index=False,
    )
    freeze = {
        "audit_protocol_version": AUDIT_PROTOCOL_VERSION,
        "run_id": run_id,
        "random_seed": AUDIT_RANDOM_SEED,
        "quotas": AUDIT_QUOTAS,
        "maximum_cases": sum(AUDIT_QUOTAS.values()),
        "selected_cases": len(audit),
        "roster_sha256": roster_sha256,
        "narrative_examples_available": not narrative.empty,
        "transfer_predictions_available": not transfer.empty,
        "sampling_manifest": "qualitative_error_audit_sampling_manifest.csv",
        "frozen_template": "qualitative_error_audit_template.csv",
        "stage_a": "qualitative_error_audit_stage_a_blinded.csv",
        "stage_b": "qualitative_error_audit_stage_b_revealed.csv",
    }
    (out / "qualitative_error_audit_freeze.json").write_text(
        json.dumps(freeze, indent=2) + "\n",
        encoding="utf-8",
    )
    instructions = f"""# Single-reviewer qualitative error audit

This is a frozen, deterministic sample (seed `{AUDIT_RANDOM_SEED}`; at most
`{sum(AUDIT_QUOTAS.values())}` unique records). The sampling manifest records
every quota and any shortfall. It is an audit instrument, not a completed
qualitative analysis, and it must not be populated with generated judgments.
Keep `qualitative_error_audit_template.csv` unchanged; the explanation workflow
and finalizer use it as the frozen context source.

## Review order

1. Open only `qualitative_error_audit_stage_a_blinded.csv`.
2. Complete every Stage A field and save a frozen copy before opening Stage B.
3. Open `qualitative_error_audit_stage_b_revealed.csv`; use the revealed
   dataset label, anchor prediction, selection reason, and available explanation
   artifacts to complete both Stage B fields.
4. Run `python scripts/14_finalize_qualitative_audit.py` only after both stages
   are complete. The finalizer validates and combines the two files.

Stage A shows dataset and source context so provenance artifacts can be coded,
but withholds the label, prediction, outcome, confidence, and selection reason.
The two sheets are generated together for a local workflow, so blinding is
procedural: do not inspect Stage B until Stage A has been saved.

## Stage A vocabulary

- the three signal/artifact/ambiguity fields: `yes`, `no`, or `unclear`;
- stance: `{'`, `'.join(sorted(STANCE_VALUES))}`; and
- notes: a case-specific rationale, never blank.

Do not infer medical truth from style or source alone. `label_ambiguity` asks
whether the supplied text could support more than one reasonable
dataset-reliability label, not whether the reviewer can clinically adjudicate
the underlying claim.

## Stage B vocabulary

Use exactly one error code:

{chr(10).join(f"- `{code}`" for code in sorted(ERROR_CODES))}

Use `no_model_error` for a correctly classified anchor case. Notes must explain
the code using the case, revealed key, and explanation output where available.
Dataset labels remain operational benchmark labels, not independent medical
ground truth. A single local explanation is illustrative, not causal evidence.
"""
    (out / "README.md").write_text(instructions, encoding="utf-8")
    print(
        f"Wrote {len(audit)} frozen single-reviewer audit cases to {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

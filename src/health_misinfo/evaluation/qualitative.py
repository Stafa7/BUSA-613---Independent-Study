from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


AUDIT_PROTOCOL_VERSION = "single_reviewer_two_stage_v1"
AUDIT_RANDOM_SEED = 613
AUDIT_ROSTER_FIELD = "audit_roster_sha256"
AUDIT_QUOTAS = {
    "outcome_true_positive": 8,
    "outcome_true_negative": 8,
    "outcome_false_positive": 8,
    "outcome_false_negative": 8,
    "high_confidence_error": 12,
    "deferred_at_80pct_target": 12,
    "transfer_error_coaid_to_fakehealth": 8,
    "transfer_error_fakehealth_to_coaid": 8,
    "retained_narrative_example": 8,
}
STAGE_A_TRISTATE_FIELDS = (
    "stage_a_health_relevant_signal",
    "stage_a_source_or_dataset_artifact",
    "stage_a_label_ambiguity",
)
STAGE_A_STANCE_FIELD = "stage_a_stance"
STAGE_A_NOTES_FIELD = "stage_a_notes"
STAGE_B_ERROR_FIELD = "stage_b_error_code"
STAGE_B_NOTES_FIELD = "stage_b_review_notes"
TRISTATE_VALUES = {"yes", "no", "unclear"}
STANCE_VALUES = {
    "endorsement",
    "discussion",
    "debunking",
    "mixed",
    "unclear",
    "not_applicable",
}
ERROR_CODES = {
    "no_model_error",
    "source_or_publisher_artifact",
    "topic_shortcut",
    "temporal_or_event_specific_cue",
    "label_ambiguity",
    "text_unit_mismatch",
    "insufficient_context",
    "discussion_or_debunking_mistaken_for_endorsement",
    "medical_terminology_misunderstanding",
    "sensational_or_emotional_framing",
    "truncation_or_preprocessing_failure",
    "engagement_proxy_or_missingness_artifact",
    "unexplained_residual_error",
}


@dataclass(frozen=True)
class FrozenAuditSample:
    selected: pd.DataFrame
    quota_manifest: pd.DataFrame


@dataclass(frozen=True)
class QualitativeAuditResult:
    finalized: pd.DataFrame
    code_summary: pd.DataFrame


def _normalized(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower()


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _validate_case_ids(frame: pd.DataFrame, label: str) -> pd.Series:
    if frame.empty:
        raise ValueError(f"{label} contains no cases")
    case_ids = frame["case_id"].astype("string").fillna("").str.strip()
    if case_ids.eq("").any():
        raise ValueError(f"{label} contains blank case_id values")
    duplicates = sorted(case_ids[case_ids.duplicated(keep=False)].unique().tolist())
    if duplicates:
        raise ValueError(f"{label} contains duplicate case_id values: {duplicates[:5]}")
    return case_ids


def _stable_hash(value: str, *, seed: int, stratum: str) -> str:
    payload = f"{seed}|{stratum}|{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_frozen_audit_sample(
    strata: Mapping[str, pd.DataFrame],
    *,
    quotas: Mapping[str, int] = AUDIT_QUOTAS,
    seed: int = AUDIT_RANDOM_SEED,
) -> FrozenAuditSample:
    """Select a deterministic, globally unique audit roster.

    Each input frame is already restricted to one eligible stratum. An optional
    numeric ``sampling_priority`` column is sorted ascending before a stable
    seeded hash; an optional ``balance_group`` column is round-robined so a
    single dataset or narrative cannot fill a quota first. Records selected for
    an earlier stratum cannot be selected again.
    """

    selected_ids: set[str] = set()
    selected_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, int | str]] = []
    next_order = 1

    for stratum, raw_quota in quotas.items():
        quota = int(raw_quota)
        if quota < 0:
            raise ValueError(f"quota for {stratum} must be non-negative")
        candidates = strata.get(stratum, pd.DataFrame()).copy()
        if candidates.empty:
            manifest_rows.append(
                {
                    "stratum": stratum,
                    "quota": quota,
                    "eligible_unique_records": 0,
                    "available_after_prior_selections": 0,
                    "selected_records": 0,
                    "shortfall": quota,
                    "status": "unavailable",
                }
            )
            continue

        _require_columns(candidates, {"record_id"}, f"{stratum} candidates")
        candidates["record_id"] = (
            candidates["record_id"].astype("string").fillna("").str.strip()
        )
        if candidates["record_id"].eq("").any():
            raise ValueError(f"{stratum} candidates contain blank record_id values")
        sampling_priority = (
            candidates["sampling_priority"]
            if "sampling_priority" in candidates
            else pd.Series(0, index=candidates.index)
        )
        candidates["sampling_priority"] = pd.to_numeric(
            sampling_priority,
            errors="coerce",
        ).fillna(float("inf"))
        balance_group = (
            candidates["balance_group"]
            if "balance_group" in candidates
            else pd.Series("all", index=candidates.index)
        )
        candidates["balance_group"] = (
            pd.Series(balance_group, index=candidates.index)
            .astype("string")
            .fillna("unavailable")
            .str.strip()
            .replace("", "unavailable")
        )
        anchor = candidates.get(
            "anchor_experiment_id",
            pd.Series("", index=candidates.index),
        ).astype("string")
        candidates["_stable_tie_breaker"] = [
            _stable_hash(
                f"{record_id}|{experiment_id}",
                seed=seed,
                stratum=stratum,
            )
            for record_id, experiment_id in zip(
                candidates["record_id"],
                anchor,
                strict=True,
            )
        ]
        candidates = (
            candidates.sort_values(
                ["sampling_priority", "_stable_tie_breaker"],
                kind="mergesort",
            )
            .drop_duplicates("record_id", keep="first")
            .copy()
        )
        eligible_unique = len(candidates)
        candidates = candidates[
            ~candidates["record_id"].isin(selected_ids)
        ].copy()
        available = len(candidates)
        candidates["_balance_rank"] = candidates.groupby(
            "balance_group",
            sort=True,
        ).cumcount()
        candidates["_balance_tie_breaker"] = candidates["balance_group"].map(
            lambda value: _stable_hash(
                str(value),
                seed=seed,
                stratum=f"{stratum}:balance",
            )
        )
        candidates = candidates.sort_values(
            [
                "_balance_rank",
                "_balance_tie_breaker",
                "sampling_priority",
                "_stable_tie_breaker",
            ],
            kind="mergesort",
        )
        chosen = candidates.head(quota).copy()
        chosen["selection_reason"] = stratum
        chosen["selection_order"] = range(next_order, next_order + len(chosen))
        next_order += len(chosen)
        selected_ids.update(chosen["record_id"])
        selected_frames.append(
            chosen.drop(
                columns=[
                    "sampling_priority",
                    "balance_group",
                    "_stable_tie_breaker",
                    "_balance_rank",
                    "_balance_tie_breaker",
                ],
                errors="ignore",
            )
        )
        shortfall = max(0, quota - len(chosen))
        manifest_rows.append(
            {
                "stratum": stratum,
                "quota": quota,
                "eligible_unique_records": eligible_unique,
                "available_after_prior_selections": available,
                "selected_records": len(chosen),
                "shortfall": shortfall,
                "status": "complete" if shortfall == 0 else "shortfall",
            }
        )

    selected = (
        pd.concat(selected_frames, ignore_index=True, sort=False)
        .sort_values("selection_order")
        .reset_index(drop=True)
        if selected_frames
        else pd.DataFrame()
    )
    return FrozenAuditSample(
        selected=selected,
        quota_manifest=pd.DataFrame(manifest_rows),
    )


def finalize_qualitative_audit(
    stage_a_audit: pd.DataFrame,
    stage_b_audit: pd.DataFrame,
    frozen_template: pd.DataFrame,
) -> QualitativeAuditResult:
    """Validate and combine one reviewer's outcome-blinded and revealed stages."""

    stage_a_columns = {
        "case_id",
        "audit_protocol_version",
        AUDIT_ROSTER_FIELD,
        *STAGE_A_TRISTATE_FIELDS,
        STAGE_A_STANCE_FIELD,
        STAGE_A_NOTES_FIELD,
    }
    stage_b_columns = {
        "case_id",
        "audit_protocol_version",
        AUDIT_ROSTER_FIELD,
        "record_id",
        "selection_reason",
        "harmonized_label",
        "anchor_prediction",
        "anchor_outcome",
        STAGE_B_ERROR_FIELD,
        STAGE_B_NOTES_FIELD,
    }
    _require_columns(stage_a_audit, stage_a_columns, "Stage A audit")
    _require_columns(stage_b_audit, stage_b_columns, "Stage B audit")
    _require_columns(
        frozen_template,
        stage_a_columns | stage_b_columns,
        "Frozen audit template",
    )

    stage_a = stage_a_audit.copy()
    stage_b = stage_b_audit.copy()
    template = frozen_template.copy()
    stage_a["case_id"] = _validate_case_ids(stage_a, "Stage A audit")
    stage_b["case_id"] = _validate_case_ids(stage_b, "Stage B audit")
    template["case_id"] = _validate_case_ids(
        template,
        "Frozen audit template",
    )
    stage_a_ids = set(stage_a["case_id"])
    stage_b_ids = set(stage_b["case_id"])
    template_ids = set(template["case_id"])
    if stage_a_ids != stage_b_ids or stage_a_ids != template_ids:
        missing_from_stage_a = sorted(stage_b_ids - stage_a_ids)
        missing_from_stage_b = sorted(stage_a_ids - stage_b_ids)
        raise ValueError(
            "Stage A, Stage B, and frozen-template case rosters differ; "
            f"missing_from_stage_a={missing_from_stage_a[:5]}, "
            f"missing_from_stage_b={missing_from_stage_b[:5]}"
        )

    metadata_values: dict[tuple[str, str], str] = {}
    for label, frame in (("Stage A audit", stage_a), ("Stage B audit", stage_b)):
        for field in ("audit_protocol_version", AUDIT_ROSTER_FIELD):
            values = set(_normalized(frame[field]))
            if "" in values or len(values) != 1:
                raise ValueError(
                    f"{label} must contain one nonblank {field} value"
                )
            metadata_values[(label, field)] = next(iter(values))

    stage_a_protocol = metadata_values[
        ("Stage A audit", "audit_protocol_version")
    ]
    stage_b_protocol = metadata_values[
        ("Stage B audit", "audit_protocol_version")
    ]
    if stage_a_protocol != stage_b_protocol:
        raise ValueError("Stage A and Stage B audit protocol versions differ")
    if stage_a_protocol != AUDIT_PROTOCOL_VERSION:
        raise ValueError(
            "audit_protocol_version must be "
            f"{AUDIT_PROTOCOL_VERSION}; found {stage_a_protocol}"
        )
    if (
        metadata_values[("Stage A audit", AUDIT_ROSTER_FIELD)]
        != metadata_values[("Stage B audit", AUDIT_ROSTER_FIELD)]
    ):
        raise ValueError("Stage A and Stage B roster hashes differ")

    coding_columns = {
        *STAGE_A_TRISTATE_FIELDS,
        STAGE_A_STANCE_FIELD,
        STAGE_A_NOTES_FIELD,
        STAGE_B_ERROR_FIELD,
        STAGE_B_NOTES_FIELD,
    }

    def validate_frozen_context(frame: pd.DataFrame, label: str) -> None:
        frozen_columns = [
            column
            for column in frame.columns
            if column not in coding_columns
        ]
        if missing := set(frozen_columns).difference(template.columns):
            raise ValueError(
                f"{label} contains context columns absent from the frozen "
                f"template: {sorted(missing)}"
            )
        submitted = frame[frozen_columns].copy()
        expected = template[frozen_columns].copy()
        for column in frozen_columns:
            submitted[column] = submitted[column].astype("string").fillna("")
            expected[column] = expected[column].astype("string").fillna("")
        submitted = submitted.sort_values("case_id").reset_index(drop=True)
        expected = expected.sort_values("case_id").reset_index(drop=True)
        if not submitted.equals(expected):
            raise ValueError(
                f"{label} frozen context differs from the generated template"
            )

    validate_frozen_context(stage_a, "Stage A audit")
    validate_frozen_context(stage_b, "Stage B audit")

    required_coding_columns = (
        *STAGE_A_TRISTATE_FIELDS,
        STAGE_A_STANCE_FIELD,
        STAGE_A_NOTES_FIELD,
        STAGE_B_ERROR_FIELD,
        STAGE_B_NOTES_FIELD,
    )
    frames_by_column = {
        **{column: stage_a for column in stage_a_columns if column != "case_id"},
        **{column: stage_b for column in stage_b_columns if column != "case_id"},
    }
    for column in required_coding_columns:
        frame = frames_by_column[column]
        blank = _normalized(frame[column]).eq("")
        if blank.any():
            incomplete = frame.loc[blank, "case_id"].tolist()
            raise ValueError(
                f"{column} is incomplete for {len(incomplete)} cases: {incomplete[:5]}"
            )

    for field in STAGE_A_TRISTATE_FIELDS:
        invalid = sorted(set(_normalized(stage_a[field])) - TRISTATE_VALUES)
        if invalid:
            raise ValueError(
                f"{field} must use yes, no, or unclear; invalid values: {invalid[:5]}"
            )
        stage_a[field] = _normalized(stage_a[field])

    stance = _normalized(stage_a[STAGE_A_STANCE_FIELD])
    invalid_stance = sorted(set(stance) - STANCE_VALUES)
    if invalid_stance:
        raise ValueError(
            f"{STAGE_A_STANCE_FIELD} contains values outside the frozen "
            f"vocabulary: {invalid_stance[:5]}"
        )
    stage_a[STAGE_A_STANCE_FIELD] = stance

    error_codes = _normalized(stage_b[STAGE_B_ERROR_FIELD])
    invalid_error_codes = sorted(set(error_codes) - ERROR_CODES)
    if invalid_error_codes:
        raise ValueError(
            f"{STAGE_B_ERROR_FIELD} contains codes outside the frozen taxonomy: "
            f"{invalid_error_codes[:5]}"
        )
    stage_b[STAGE_B_ERROR_FIELD] = error_codes
    stage_a[STAGE_A_NOTES_FIELD] = (
        stage_a[STAGE_A_NOTES_FIELD].astype("string").str.strip()
    )
    stage_b[STAGE_B_NOTES_FIELD] = (
        stage_b[STAGE_B_NOTES_FIELD].astype("string").str.strip()
    )

    finalized = stage_a.merge(
        stage_b,
        on="case_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_stage_a_context", ""),
    )
    summary_rows = []
    for field in (*STAGE_A_TRISTATE_FIELDS, STAGE_A_STANCE_FIELD, STAGE_B_ERROR_FIELD):
        counts = finalized[field].value_counts(dropna=False)
        for value, records in counts.items():
            summary_rows.append(
                {
                    "field": field,
                    "value": value,
                    "records": int(records),
                    "fraction": float(records / len(finalized)),
                }
            )
    return QualitativeAuditResult(
        finalized=finalized,
        code_summary=pd.DataFrame(summary_rows),
    )

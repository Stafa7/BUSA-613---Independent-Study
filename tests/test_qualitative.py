import pandas as pd
import pytest

from health_misinfo.evaluation.qualitative import (
    AUDIT_PROTOCOL_VERSION,
    AUDIT_QUOTAS,
    finalize_qualitative_audit,
    select_frozen_audit_sample,
)


def _stage_a_audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["case_0001", "case_0002"],
            "audit_protocol_version": [AUDIT_PROTOCOL_VERSION] * 2,
            "audit_roster_sha256": ["a" * 64] * 2,
            "dataset": ["coaid", "fakehealth"],
            "title": ["Item one", "Item two"],
            "text_excerpt": ["Excerpt one", "Excerpt two"],
            "stage_a_health_relevant_signal": ["yes", "unclear"],
            "stage_a_source_or_dataset_artifact": ["no", "yes"],
            "stage_a_label_ambiguity": ["no", "yes"],
            "stage_a_stance": ["endorsement", "unclear"],
            "stage_a_notes": ["clear health claim", "insufficient context"],
        }
    )


def _stage_b_audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["case_0001", "case_0002"],
            "audit_protocol_version": [AUDIT_PROTOCOL_VERSION] * 2,
            "audit_roster_sha256": ["a" * 64] * 2,
            "record_id": ["record_1", "record_2"],
            "selection_reason": [
                "outcome_false_positive",
                "outcome_false_negative",
            ],
            "harmonized_label": ["reliable", "unreliable"],
            "anchor_prediction": ["unreliable", "reliable"],
            "anchor_outcome": ["false_positive", "false_negative"],
            "stage_b_error_code": ["topic_shortcut", "insufficient_context"],
            "stage_b_review_notes": [
                "topic cue dominated",
                "needed the full article",
            ],
        }
    )


def _frozen_template() -> pd.DataFrame:
    template = _stage_b_audit().copy()
    stage_a = _stage_a_audit()
    for column in ("dataset", "title", "text_excerpt"):
        template[column] = stage_a[column]
    for column in (
        "stage_a_health_relevant_signal",
        "stage_a_source_or_dataset_artifact",
        "stage_a_label_ambiguity",
        "stage_a_stance",
        "stage_a_notes",
        "stage_b_error_code",
        "stage_b_review_notes",
    ):
        template[column] = ""
    return template


def test_finalize_single_reviewer_audit_combines_and_summarizes():
    result = finalize_qualitative_audit(
        _stage_a_audit(),
        _stage_b_audit(),
        _frozen_template(),
    )

    assert len(result.finalized) == 2
    assert result.finalized.loc[1, "record_id"] == "record_2"
    assert result.finalized.loc[1, "stage_b_error_code"] == "insufficient_context"
    assert result.code_summary["records"].sum() == 10
    assert set(result.code_summary["field"]) == {
        "stage_a_health_relevant_signal",
        "stage_a_source_or_dataset_artifact",
        "stage_a_label_ambiguity",
        "stage_a_stance",
        "stage_b_error_code",
    }


def test_finalize_single_reviewer_audit_rejects_incomplete_or_mismatched_inputs():
    stage_a = _stage_a_audit()
    stage_a.loc[0, "stage_a_notes"] = ""
    with pytest.raises(ValueError, match="stage_a_notes is incomplete"):
        finalize_qualitative_audit(
            stage_a,
            _stage_b_audit(),
            _frozen_template(),
        )

    mismatched_stage_b = _stage_b_audit()
    mismatched_stage_b.loc[1, "case_id"] = "case_9999"
    with pytest.raises(ValueError, match="case rosters differ"):
        finalize_qualitative_audit(
            _stage_a_audit(),
            mismatched_stage_b,
            _frozen_template(),
        )

    mismatched_roster = _stage_b_audit()
    mismatched_roster["audit_roster_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="roster hashes differ"):
        finalize_qualitative_audit(
            _stage_a_audit(),
            mismatched_roster,
            _frozen_template(),
        )

    altered_context = _stage_b_audit()
    altered_context.loc[0, "record_id"] = "substituted_record"
    with pytest.raises(ValueError, match="frozen context differs"):
        finalize_qualitative_audit(
            _stage_a_audit(),
            altered_context,
            _frozen_template(),
        )


def test_finalize_single_reviewer_audit_enforces_frozen_vocabularies():
    stage_a = _stage_a_audit()
    stage_a.loc[0, "stage_a_label_ambiguity"] = "maybe"
    with pytest.raises(ValueError, match="must use yes, no, or unclear"):
        finalize_qualitative_audit(
            stage_a,
            _stage_b_audit(),
            _frozen_template(),
        )

    stage_a = _stage_a_audit()
    stage_a.loc[0, "stage_a_stance"] = "supports"
    with pytest.raises(ValueError, match="outside the frozen vocabulary"):
        finalize_qualitative_audit(
            stage_a,
            _stage_b_audit(),
            _frozen_template(),
        )

    stage_b = _stage_b_audit()
    stage_b.loc[0, "stage_b_error_code"] = "invented_code"
    with pytest.raises(ValueError, match="outside the frozen taxonomy"):
        finalize_qualitative_audit(
            _stage_a_audit(),
            stage_b,
            _frozen_template(),
        )

    stage_b = _stage_b_audit()
    stage_b.loc[0, "audit_protocol_version"] = "old_protocol"
    with pytest.raises(ValueError, match="one nonblank audit_protocol_version"):
        finalize_qualitative_audit(
            _stage_a_audit(),
            stage_b,
            _frozen_template(),
        )


def test_frozen_sampler_is_deterministic_unique_and_records_shortfalls():
    strata = {
        "first": pd.DataFrame(
            {
                "record_id": ["r1", "r2", "r3"],
                "anchor_experiment_id": ["e1", "e1", "e1"],
                "sampling_priority": [0, 1, 2],
                "balance_group": ["coaid", "fakehealth", "coaid"],
            }
        ),
        "second": pd.DataFrame(
            {
                "record_id": ["r1", "r4"],
                "anchor_experiment_id": ["e2", "e2"],
                "sampling_priority": [0, 1],
                "balance_group": ["coaid", "fakehealth"],
            }
        ),
    }
    quotas = {"first": 2, "second": 2, "unavailable": 1}

    first = select_frozen_audit_sample(strata, quotas=quotas, seed=613)
    shuffled = select_frozen_audit_sample(
        {name: frame.sample(frac=1, random_state=99) for name, frame in strata.items()},
        quotas=quotas,
        seed=613,
    )

    assert first.selected["record_id"].tolist() == shuffled.selected[
        "record_id"
    ].tolist()
    assert not first.selected["record_id"].duplicated().any()
    assert len(first.selected) == 3
    manifest = first.quota_manifest.set_index("stratum")
    assert manifest.loc["second", "shortfall"] == 1
    assert manifest.loc["unavailable", "status"] == "unavailable"


def test_frozen_sampler_balances_groups_and_caps_original_scope_at_80():
    candidates = pd.DataFrame(
        {
            "record_id": [f"coaid_{index}" for index in range(6)]
            + [f"fakehealth_{index}" for index in range(2)],
            "anchor_experiment_id": ["experiment"] * 8,
            "sampling_priority": [0] * 8,
            "balance_group": ["coaid"] * 6 + ["fakehealth"] * 2,
        }
    )

    result = select_frozen_audit_sample(
        {"balanced": candidates},
        quotas={"balanced": 4},
        seed=613,
    )

    assert set(result.selected.head(2)["record_id"].str.split("_").str[0]) == {
        "coaid",
        "fakehealth",
    }
    assert sum(AUDIT_QUOTAS.values()) == 80

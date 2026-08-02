# Manual review guide

This guide preserves the original study scope with one disclosed human reviewer.
It covers two limited tasks: a staged qualitative error audit and validation of
candidate BERTopic narratives. It does not ask the reviewer to establish medical
truth.

## Before reviewing anything

Always resolve the review location from:

`outputs/experiments/latest_run.json`

Treat its `run_path` value as `RUN_ROOT` throughout this guide. Confirm that its
`run_id` is not the stale pre-overhaul run
`20260723T221313Z_33100ceb_b83d97b4` before opening any review sheet. Never copy
judgments from an older run into the current sheets.

The future error-audit roster is frozen deterministically with seed `613` and a
maximum of 80 unique cases. Do not add, remove, reorder, substitute, or
hand-select cases after seeing their content. The audit is a purposive,
stratified diagnostic sample—not an estimate of error prevalence.

For a complete original-scope handoff, the order is:

1. Complete the baseline, calibration, and bidirectional-transfer computation.
2. Run `.venv/bin/python scripts/13_run_narrative_discovery.py`, perform the
   representative and topic reviews below, and finalize the retained narratives.
3. Generate the error audit with
   `.venv/bin/python scripts/08_prepare_error_audit.py` only after transfer and
   retained-narrative outputs are available.
4. Complete Error Stage A.
5. Run `.venv/bin/python scripts/12_run_explanation_audit.py`, without opening
   its outputs before Stage A is saved.
6. Complete Error Stage B and run its finalizer.

This order lets the frozen error roster include transfer errors and examples
from retained narratives. If either upstream analysis is unavailable, the
sampling manifest must show the corresponding shortfall rather than silently
replacing it with hand-picked cases.

## Candidate-narrative review

Generate the review sheets from the repository root:

```bash
.venv/bin/python scripts/13_run_narrative_discovery.py
```

Start with:

`RUN_ROOT/narrative_discovery/START_HERE_HUMAN_REVIEW.md`

Then, inside each generated dataset directory under
`RUN_ROOT/narrative_discovery/`, complete the two numbered files in order.

### 1 — representative documents

Open:

`01_HUMAN_REVIEW_representative_documents.csv`

For at least five representative documents per eligible topic, complete:

| Field | Allowed values |
|---|---|
| `human_supports_topic` | `yes`, `no`, `unclear` |
| `human_stance` | `endorsement`, `discussion`, `debunking`, `mixed`, `unclear` |
| `human_artifact_pattern` | `none`, `publisher`, `dataset`, `generic`, `mixed`, `unclear` |
| `human_notes` | Optional rationale |
| `human_review_confirmation` | Exactly `human_reviewed` |

### 2 — topic decisions

After reviewing the representatives, open:

`02_HUMAN_REVIEW_topic_decisions.csv`

Complete every eligible topic:

| Field | Allowed value or rule |
|---|---|
| `human_topic_coherence` | `coherent`, `mixed`, `incoherent` |
| `human_topic_stance` | `endorsement`, `discussion`, `debunking`, `mixed`, `unclear` |
| `human_artifact_pattern` | `none`, `publisher`, `dataset`, `generic`, `mixed`, `unclear` |
| `human_narrative_id` | Required for a coherent, non-artifact topic |
| `human_narrative_label` | Required, reviewer-authored descriptive label for a coherent, non-artifact topic |
| `human_label_source` | Exactly `reviewer_authored` for a retained candidate |
| `human_notes` | Optional rationale |
| `human_review_confirmation` | Exactly `human_reviewed` |

Topics may share a narrative ID only when intentionally merged; merged topics
must use the same reviewer-authored label. Do not copy
`machine_topic_name` or `machine_topic_keywords` unchanged as a validated label.
A publisher, dataset, or generic-language pattern is not a narrative.

After every eligible topic is complete, run:

```bash
.venv/bin/python scripts/15_finalize_narrative_validation.py
```

Review the gate result in:

`RUN_ROOT/narrative_validation/human_validation_status.json`

If validation completes, the principal outputs include
`validated_narratives.csv`, `topic_validation_outcomes.csv`,
`retained_topic_to_narrative.csv`, `narrative_prevalence.csv`,
`narrative_prevalence.png`, and `narrative_performance.csv` in the same
directory. The finalizer validates and summarizes the human entries; it never
supplies missing judgments or labels.

## Error review

Start in:

`RUN_ROOT/error_audit/`

Keep these generated provenance files unchanged:

- `qualitative_error_audit_freeze.json`
- `qualitative_error_audit_sampling_manifest.csv`
- `qualitative_error_audit_template.csv` (internal frozen context; do not review
  or edit it)

They document the frozen roster and sampling coverage.

The frozen quotas are:

| Stratum | Maximum cases |
|---|---:|
| True positive | 8 |
| True negative | 8 |
| False positive | 8 |
| False negative | 8 |
| High-confidence error | 12 |
| Deferred at the 80% validation-coverage target | 12 |
| CoAID-to-FakeHealth transfer error | 8 |
| FakeHealth-to-CoAID transfer error | 8 |
| Retained-narrative example | 8 |

Selection is globally deduplicated. Consequently, a later stratum can have a
document shortfall when its eligible records were already selected for an
earlier stratum; the sampling manifest records this explicitly.

### Stage A — label/prediction-blinded content review

Open and edit only:

`qualitative_error_audit_stage_a_blinded.csv`

Do not open the Stage B sheet, model explanations, labels, predictions, outcomes,
or selection reasons until every Stage A row is complete. Do not alter any
generated context or identifier column.

Stage A still shows dataset and source context because those are needed to code
possible provenance artifacts; it withholds the benchmark label, prediction,
outcome, confidence, and selection reason. Both stage sheets are generated
together in this local workflow, so this is procedural rather than
cryptographically enforced blinding. Save Stage A before the reveal and do not
revise its judgments after opening Stage B.

Complete these fields for every case:

| Field | Allowed values |
|---|---|
| `stage_a_health_relevant_signal` | `yes`, `no`, `unclear` |
| `stage_a_source_or_dataset_artifact` | `yes`, `no`, `unclear` |
| `stage_a_label_ambiguity` | `yes`, `no`, `unclear` |
| `stage_a_stance` | `endorsement`, `discussion`, `debunking`, `mixed`, `unclear`, `not_applicable` |
| `stage_a_notes` | Required case-specific rationale |

Judge only what the supplied title and excerpt support. Use `unclear` or
`not_applicable` when appropriate; do not fill gaps from assumptions.

### Stage B — revealed model-error review

Only after saving a complete Stage A sheet, open:

`qualitative_error_audit_stage_b_revealed.csv`

You may now consult the revealed dataset label, model prediction, confidence,
outcome, and selection reason. When available, use these explanation files as
descriptive aids:

- `RUN_ROOT/explanation_audit/case_model_explanations.csv`
- `RUN_ROOT/explanation_audit/local_feature_contributions.csv`
- `RUN_ROOT/explanation_audit/refit_verification.csv`
- `RUN_ROOT/explanation_audit/standard_controlled_stability.csv`

Complete `stage_b_error_code` with exactly one code below and provide a
case-specific rationale in the required `stage_b_review_notes` field:

- `no_model_error`
- `source_or_publisher_artifact`
- `topic_shortcut`
- `temporal_or_event_specific_cue`
- `label_ambiguity`
- `text_unit_mismatch`
- `insufficient_context`
- `discussion_or_debunking_mistaken_for_endorsement`
- `medical_terminology_misunderstanding`
- `sensational_or_emotional_framing`
- `truncation_or_preprocessing_failure`
- `engagement_proxy_or_missingness_artifact`
- `unexplained_residual_error`

Use `no_model_error` for a correctly classified control. When more than one
failure pattern seems plausible, select the best-supported primary code and
record alternatives and uncertainty in the notes.

After both sheets are complete, run from the repository root:

```bash
.venv/bin/python scripts/14_finalize_qualitative_audit.py
```

The command uses the new `latest_run.json` by default. Its expected outputs are
under `RUN_ROOT/error_audit/finalized/`, including
`finalized_qualitative_audit.csv`, `qualitative_code_summary.csv`,
`summary.json`, and `status.json`. A one-reviewer study must not report
inter-rater agreement or adjudication.

After both review finalizers complete, regenerate the study summary with:

```bash
.venv/bin/python scripts/09_summarize_study.py
```

## Interpretation boundaries

- Dataset labels mean dataset-defined reliability status, not independently
  verified medical truth.
- A document that mentions a claim may discuss or debunk it rather than endorse
  it.
- A short excerpt may be insufficient; do not reconstruct absent context.
- Feature contributions and topic assignments are associations, not causal
  explanations or evidence of model reasoning.
- A suspected source, publisher, topic, or dataset artifact is a coded
  diagnostic judgment, not proof of causation.
- The capped error sample cannot support prevalence estimates.
- A machine topic becomes a reported candidate narrative only after the human
  evidence and frozen retention gates pass.
- Narrative prevalence is prevalence within the analyzed dataset-defined
  corpus, not prevalence in the public or on the internet.
- Do not make clinical-accuracy, medical-fact-checking, or population-level
  claims from either review.
- Disclose that one reviewer performed the work; do not claim independent
  replication, inter-rater reliability, adjudication, or qualitative saturation.

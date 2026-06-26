# Independent Study Working Log

Related context note: [[independent_study_project_context]]

Use this file as the session-by-session working log for the project. Update it each time you make progress so the main context file stays focused on stable project context.

## Session Template

Copy this block for each new work session:

```md
## YYYY-MM-DD

- Focus:
- Changes made:
- Files updated:
- Decisions:
- Questions or blockers:
- Next step:
```

## Baseline Carry-Forward Notes

### Supervisor feedback

- TBD

### Dataset access notes

- CoAID
  - Status: Official repository downloaded and verified
  - Downloaded: 2026-06-25 to `data/raw/coaid/`
  - Issues: Data profiling and label-provenance audit remain outstanding
- FakeHealth
  - Status: Official repository downloaded and verified
  - Downloaded: 2026-06-25 to `data/raw/fakehealth/`
  - Issues: Engagement files provide platform IDs rather than hydrated post objects; API rehydration is not required
- KInIT
  - Status: Optional
  - Downloaded: TBD
  - Issues: TBD

### Methodology decisions

- Main baseline model: TBD
- Main transformer model: TBD
- Topic modeling method: BERTopic likely
- Explainability method: SHAP/LIME likely
- Dashboard: Optional

### Important project decisions

- TBD

### Questions for supervisor

1. Is the scope appropriate for a 3-credit independent study?
2. Should the emphasis be more methodological, managerial, or public-health oriented?
3. Should FakeHealth be required or optional?
4. Is a dashboard expected or is a reproducible pipeline/report enough?
5. How much technical depth is expected in transformer modeling?
6. What should be the expected final report length?
7. How often should progress meetings occur?

## 2026-05-27

- Focus: Restructure project documentation so the main context note does not carry the running work log.
- Changes made: Moved the working notes/update log out of the main context file into this dedicated note.
- Files updated: `docs/independent_study_project_context.md`, `docs/independent_study_working_log.md`
- Decisions: Keep long-lived project context in the context file and track session-level changes here.
- Questions or blockers: None.
- Next step: Add a new dated section here at the end of each work session.

## 2026-06-06
- Focus: Begin Phase 1 literature review for the health misinformation NLP/social engagement analytics project.
- Changes made: Identified core literature, converted downloaded PDFs into Markdown, created detailed per-paper review notes from the PDFs, and drafted a first-pass literature-review synthesis.
- Additional progress: Created a final literature brainstorming document summarizing dataset provenance, methodologies, findings, successes, risks, and possible implications for re-evaluating the project design.
- Files updated:
  - `docs/literature_mds/`
  - `docs/literature_review/`
  - `docs/independent_study_working_log.md`
- Decisions: Keep raw PDF-to-Markdown conversions in `docs/literature_mds/` and cleaned analytical notes in `docs/literature_review/`; use the synthesis note as the starting point for the formal report literature review.
- Questions or blockers: The downloaded PDF set does not currently include the CoAID paper, so CoAID is not represented in the converted Markdown batch.
- Next step: Use the final brainstorming document to decide whether the current project outline should be refined before moving into data acquisition/profiling.

## 2026-06-25

- Focus: Convert the broad project outline into a decision-complete research execution protocol.
- Changes made: Created `docs/independent_study_execution_plan.md` with five locked research questions, sequential phases, experiment definitions, evaluation requirements, feasibility gates, completion criteria, deliverable checklists, a risk register, a decision log, and final-report mapping.
- Files updated:
  - `docs/independent_study_execution_plan.md`
  - `docs/independent_study_working_log.md`
- Decisions: Frame the contribution as a robustness audit; use CoAID as the primary dataset and FakeHealth as the secondary dataset; require artifact-controlled evaluation, calibration, selective prediction, and explanation auditing; exclude AI-generated content and dependence on live social-media collection.
- Questions or blockers: Dataset access, field availability, and cross-dataset compatibility remain to be tested during the acquisition and provenance phases.
- Next step: Begin Phase 0 of the execution plan, then acquire and inventory CoAID and FakeHealth.

## 2026-06-25 — Dataset acquisition

- Focus: Acquire the two required datasets from their official repositories.
- Changes made: Downloaded CoAID and FakeHealth into `data/raw/`, verified clean repository states and exact Git revisions, inspected required content/review/engagement files, and created a dataset acquisition manifest.
- Files updated:
  - `data/raw/coaid/`
  - `data/raw/fakehealth/`
  - `docs/dataset_acquisition_manifest.md`
  - `docs/independent_study_working_log.md`
- Decisions: Do not download the optional 2.2 GB FakeHealth follower/following archive at this stage; do not make the project dependent on X API rehydration.
- Questions or blockers: The usable engagement features and cross-dataset label compatibility still require formal profiling.
- Next step: Execute Phase 2 and Phase 3 audits: provenance, label mapping, field inventory, missingness, duplicates, and engagement feasibility.

## 2026-06-25 — Repository documentation

- Focus: Create a comprehensive repository entry point and audit the current project structure.
- Changes made: Added a root `README.md` covering the project purpose, contribution, locked questions, dataset sources and revisions, terminology, workflow, planned methods, evaluation framework, reproducibility requirements, documentation map, status, ethical boundaries, storage policy, and citations.
- Files updated:
  - `README.md`
  - `docs/independent_study_working_log.md`
- Decisions: Document the repository as it currently exists and obtain approval before relocating or renaming existing files.
- Questions or blockers: The project still needs an approved directory reorganization, a root Git repository decision, environment files, and analysis-code scaffolding.
- Next step: Review and approve the proposed repository reorganization before any files are moved.

## 2026-06-26 — Optional FakeHealth archive and data tracking policy

- Focus: Preserve the optional FakeHealth Zenodo user-network archive locally while preventing raw data from being committed to GitHub.
- Changes made: Added `scripts/download_datasets.py` as the canonical dataset acquisition script, expanded `.gitignore` to exclude raw/intermediate/processed data and generated artifacts, removed raw dataset entries and local OS files from Git tracking without deleting them locally, downloaded and extracted the optional FakeHealth Zenodo archive, and verified the archive MD5 checksum.
- Files updated:
  - `.gitignore`
  - `README.md`
  - `docs/dataset_acquisition_manifest.md`
  - `docs/independent_study_working_log.md`
  - `scripts/download_datasets.py`
- Local data added:
  - `data/raw/fakehealth_zenodo/FakeHealth.zip`
  - `data/raw/fakehealth_zenodo/extracted/`
- Verification: `FakeHealth.zip` size `2,205,934,463` bytes; MD5 `1dd710f663694096ba604144ad7e4930`; extracted user-network files include `255,235` follower JSON files and `255,229` following JSON files.
- Decisions: Dataset acquisition and preprocessing should live in `.py` scripts as the reproducible source of truth; notebooks can be used later for exploratory analysis and presentation.
- Questions or blockers: A commit and push are still required for GitHub to remove the previously tracked raw-data gitlinks from the main branch.
- Next step: Commit and push the `.gitignore`, documentation, acquisition script, and Git index cleanup.

## 2026-06-26 — Phase 0–6 first-half implementation

- Focus: Implement the first half of the study through reproducible data preparation, frozen splits, traditional text baselines, and a transformer compute gate.
- Changes made: Added package/config/test scaffolding, implemented Phase 1–6 scripts, generated local inventory/provenance/profile/split/baseline outputs, and created a transformer compute-stop note because local transformer dependencies are unavailable.
- Files updated:
  - `.gitignore`
  - `README.md`
  - `configs/`
  - `scripts/01_inventory.py` through `scripts/06_run_transformer.py`
  - `src/health_misinfo/`
  - `tests/`
- Local outputs generated:
  - `outputs/inventory/`
  - `outputs/provenance/`
  - `data/processed/`
  - `outputs/profiles/`
  - `outputs/splits/`
  - `outputs/experiments/`
- Verification: Processed `8,271` included records; generated `18` traditional baseline experiment outputs; `python -m pytest` passed with `10` tests.
- Decisions: CoAID source-disjoint evaluation falls back to duplicate-group stratified evaluation because the source-disjoint split violated duplicate-family separation; FakeHealth source-disjoint evaluation passes. DistilBERT is deferred until `torch`, `transformers`, `datasets`, and `evaluate` are installed.
- Questions or blockers: None for Phase 0–5. Phase 6 model training is compute/dependency blocked locally, with a documented stop note.
- Next step: Install transformer dependencies or use an approved compute environment, then rerun `scripts/06_run_transformer.py` for the DistilBERT comparison.

# Health Misinformation Detection, Robustness, and Decision Support

Independent study project for **BUSA 613 at McGill University**.

This project examines whether natural-language-processing models for health-information reliability learn robust and useful signals or primarily exploit dataset-specific artifacts. It combines text classification, social-engagement analysis, cross-dataset transfer, narrative discovery, confidence calibration, selective prediction, explainability, and qualitative error analysis.

The study uses **CoAID** as its primary dataset and **FakeHealth** as its secondary validation and robustness dataset.

> **Project status:** The archival computational rerun is complete in the locked
> Python 3.11 environment, including validation-only hyperparameter tuning, 63
> CUDA/MPS-aware transformer runs, calibration, transfer, engagement, and
> candidate-narrative discovery. The narrative and staged qualitative audits
> remain open at their required human-review gates. See the
> [study execution and results record](docs/study_execution_and_results.md).

---

## Table of contents

- [Research purpose](#research-purpose)
- [Research contribution](#research-contribution)
- [Locked research questions](#locked-research-questions)
- [Datasets](#datasets)
- [Dataset terminology and limitations](#dataset-terminology-and-limitations)
- [Repository structure](#repository-structure)
- [Research workflow](#research-workflow)
- [Study quality audit](#study-quality-audit)
- [Planned models and analyses](#planned-models-and-analyses)
- [Evaluation framework](#evaluation-framework)
- [Reproducibility rules](#reproducibility-rules)
- [Documentation guide](#documentation-guide)
- [Current project status](#current-project-status)
- [Environment and execution](#environment-and-execution)
- [Ethical and practical boundaries](#ethical-and-practical-boundaries)
- [Data storage and version-control policy](#data-storage-and-version-control-policy)
- [References and dataset citations](#references-and-dataset-citations)

---

## Research purpose

Health-misinformation detection is often evaluated as a standard binary-classification problem. A model is trained on a labeled dataset, its aggregate performance is reported, and the resulting score is treated as evidence that the model can detect misinformation.

That evaluation can be misleading. A model may perform well because it recognizes:

- publishers or domains associated with a label;
- COVID-specific words that do not generalize to other health topics;
- duplicated or closely related records across data splits;
- differences in how datasets were constructed;
- engagement-data availability rather than engagement behavior;
- stylistic shortcuts instead of meaningful health-information quality signals.

This study therefore evaluates more than predictive performance. It asks whether model behavior remains credible under stricter tests involving source and topic controls, cross-dataset transfer, confidence calibration, abstention, subgroup analysis, explanations, and qualitative review.

The intended result is not an autonomous medical fact-checker. It is an evidence-based assessment of how health-information classification might support human review, public-health communication, and digital-risk monitoring—and where it should not be trusted.

---

## Research contribution

The project is framed as a **robustness audit**, not as the invention of a new neural architecture.

Its contribution is the controlled integration of:

1. traditional and transformer-based text models;
2. standard and artifact-controlled evaluation;
3. text-only, engagement-only, and combined feature experiments;
4. bidirectional transfer between datasets with different label provenance;
5. narrative-level model and engagement analysis;
6. confidence calibration and selective prediction;
7. explanation-based auditing and qualitative error analysis.

The central proposition being tested is:

> Health-information classifiers should be judged not only by aggregate predictive performance, but also by their robustness to dataset artifacts, transferability, calibration, ability to defer uncertain cases, narrative-level behavior, and explanation quality.

Negative findings remain meaningful. Poor transfer, inconsistent engagement benefits, unstable explanations, or a need for substantial human review would all provide evidence about the limitations of current health-misinformation datasets and models.

---

## Locked research questions

1. **How do traditional machine-learning and transformer-based models compare when classifying dataset-defined unreliable health information under standard and artifact-controlled within-dataset evaluation?**

2. **Do available social-engagement features add consistent predictive value over text-only models under the same evaluation controls?**

3. **To what extent do models transfer between CoAID and FakeHealth, and which differences in topic, text unit, or label provenance explain any performance degradation?**

4. **What recurring narratives characterize dataset-defined unreliable health information, and—where metadata permit—how do model performance, engagement, and emotional profiles vary across those narratives?**

5. **Can confidence calibration, selective prediction, and explanation-based auditing identify cases in which automated classifications should not be trusted?**

Changes to these questions require a documented rationale in the local study protocol.

---

## Datasets

The datasets were obtained from the repositories maintained by their original authors. Exact source revisions, checksums, observed file counts, and acquisition decisions are recorded in the [dataset acquisition manifest](docs/dataset_acquisition_manifest.md).

### CoAID

**CoAID: COVID-19 Healthcare Misinformation Dataset**

- Official repository: <https://github.com/cuilimeng/CoAID>
- Local location: `data/raw/coaid/`
- Download date: June 25, 2026
- Pinned commit: `d238224346781255e1e7e6ed8bc410c2b2e6329e`
- Nearest tagged release: `V0.4-2-gd238224`
- Role in this study: primary dataset

The downloaded repository contains four dated collections covering reliable and unreliable COVID-19 news and claims, together with mappings to related posts and replies.

Observed raw CSV records, before parsing and deduplication:

| Record type | Count |
|---|---:|
| Reliable news | 4,532 |
| Unreliable news | 925 |
| Reliable claims | 490 |
| Unreliable claims | 28 |
| Post-ID links | 160,875 |
| Reply-ID links | 135,877 |

These are acquisition-level observations. The validity-overhaul primary cohort
contains 3,055 substantive CoAID articles; claims remain available only for
audit and sensitivity work.

### FakeHealth

**FakeHealth: HealthStory and HealthRelease**

- Official repository: <https://github.com/EnyanDai/FakeHealth>
- Archived dataset record: <https://zenodo.org/records/3862989>
- Local location: `data/raw/fakehealth/`
- Download date: June 25, 2026
- Pinned commit: `ec9379de8f8f13af8c436dd6dd9bfaddacd2df30`
- Role in this study: secondary validation and robustness dataset

The downloaded repository contains:

- article content and metadata;
- expert health-news reviews;
- numerical ratings;
- judgments against health-reporting criteria;
- review summaries and explanations;
- mappings from reviewed articles to post, repost, and reply identifiers.

Observed repository contents:

| Component | HealthStory | HealthRelease |
|---|---:|---:|
| Downloaded article-content files | 1,638 | 599 |
| Expert review records | 1,690 | 606 |
| Items with engagement-ID mappings | 1,690 | 606 |
| Post IDs | 384,073 | 47,338 |
| Repost IDs | 120,709 | 16,959 |
| Reply IDs | 27,601 | 1,575 |

The optional 2.2 GB Zenodo user-network archive is stored locally under `data/raw/fakehealth_zenodo/`. It primarily supplies follower and following identifiers. These files are useful for documenting the available FakeHealth network metadata, but the core study still does not depend on live X API access or the successful rehydration of historical social-media records.

Observed optional Zenodo archive contents:

| Component | Count |
|---|---:|
| Downloaded archive size | 2.1 GB |
| Extracted directory size | 4.9 GB |
| User follower JSON files | 255,235 |
| User following JSON files | 255,229 |
| Total user-network JSON files | 510,464 |

### Why these datasets are used together

CoAID and FakeHealth are complementary rather than equivalent:

- **CoAID** focuses on COVID-19 healthcare news and claims.
- **FakeHealth** covers broader health reporting and links article quality to expert-review criteria.
- Their different topics, source structures, text units, and labeling processes make them useful for testing robustness.
- Those same differences make direct transfer difficult and require explicit compatibility checks before cross-dataset results can be interpreted.

No third dataset is required for the core study. Additional datasets may be considered only after all required analyses are complete.

---

## Dataset terminology and limitations

This repository uses **dataset-defined unreliable health information** as the umbrella term when discussing CoAID and FakeHealth together.

This wording is deliberate:

- the datasets do not use identical definitions of misinformation;
- some labels originate from source selection or credibility judgments;
- FakeHealth uses expert health-news ratings and reporting criteria;
- a dataset label is not independent proof that every claim in an item is false;
- a source-level label should not be represented as a claim-level fact check;
- harmonized labels used in this study are analytical mappings, not new ground-truth annotations.

The study will preserve each dataset's native labels and document every transformation used to create a common analytical target.

Poor cross-dataset performance may reflect:

- label-provenance differences;
- different text units;
- topic shift;
- source shift;
- temporal shift;
- class-prevalence differences;
- model limitations.

These explanations must be investigated rather than attributing all degradation to the model.

---

## Repository structure

Current top-level structure:

```text
.
├── README.md
├── Approval and Application/
│   └── BUSA 613 independent-study forms
├── data/
│   └── raw/
│       ├── coaid/
│       └── fakehealth/
└── docs/
    ├── dataset_acquisition_manifest.md
    ├── final_literature_brainstorming_document.md
    ├── independent_study_evaluation_metrics.md
    ├── independent_study_project_context.md
    ├── literature_mds/
    ├── literature_pdfs/
    └── literature_review/
```

### Directory responsibilities

| Path | Purpose |
|---|---|
| `data/raw/` | Immutable copies of source datasets |
| `docs/` | Research plans, logs, manifests, and synthesized documentation |
| `docs/literature_pdfs/` | Original literature PDFs used in the review |
| `docs/literature_mds/` | Machine-readable conversions of literature PDFs |
| `docs/literature_review/` | Clean per-paper analytical notes and synthesis |
| `scripts/` | Reproducible project scripts, including dataset acquisition |
| `Approval and Application/` | Local administrative course forms, ignored by Git |

The raw dataset directories retain their upstream Git metadata so the exact downloaded revisions can be verified. They should not be edited directly.

Raw data is intentionally excluded from Git. The canonical acquisition command is:

```bash
python scripts/download_datasets.py --all --extract
```

Use the script as the source of truth for where datasets come from. Notebooks may be added later for exploration, but dataset acquisition and preprocessing should live in `.py` files so the workflow is reproducible from a clean clone.

Processed data and generated results are produced locally and ignored by Git.
The strict Python environment specification is tracked in `.python-version`,
`pyproject.toml`, `uv.lock`, `requirements.txt`, and
`requirements.lock.txt`.

---

## Research workflow

The authoritative protocol and its decision log are maintained locally.

The study proceeds through the following stages:

1. **Protocol and workspace setup**
   - Freeze research questions.
   - Establish environment, naming, and artifact conventions.

2. **Dataset acquisition and inventory**
   - Preserve official source files.
   - Record revisions, checksums, licenses or terms, and file inventories.

3. **Provenance and compatibility audit**
   - Document dataset construction and label origin.
   - Define any harmonized label mapping.
   - Decide whether cross-dataset transfer is defensible.

4. **Cleaning and profiling**
   - Parse text and metadata.
   - Quantify empty-string-aware missingness, class balance, text lengths, sources, dates, duplicates, and engagement coverage.
   - Exclude scraper/interface pages, non-substantive bodies, contradictory exact duplicates, and redundant exact copies using coded reasons.
   - Keep CoAID claims and title-only items outside the primary cohort.

5. **Leakage audit and controlled splits**
   - Canonicalize publisher aliases, domains, and archived URLs.
   - Construct connected URL/title/body/text families.
   - Freeze unit-by-label family-safe manifests.
   - Use canonical-publisher-disjoint evaluation only where the validity gates pass; otherwise use the declared controlled-design fallback without relabeling it.

6. **Traditional text baselines**
   - Train and evaluate TF-IDF Logistic Regression and Linear SVM.

7. **Transformer evaluation**
   - Compare pinned DistilBERT, ModernBERT, and BiomedBERT systems at 512 tokens and three seeds.

8. **Engagement ablations**
   - Compare text-only, engagement-only, and combined models where metadata are sufficient.

9. **Cross-dataset transfer**
   - Diagnose CoAID-to-FakeHealth and FakeHealth-to-CoAID transfer after filtering overlapping source domains.

10. **Narrative discovery**
    - Use BERTopic and qualitative validation to identify candidate narratives.

11. **Calibration and selective prediction**
    - Evaluate confidence quality and human-review deferral strategies.

12. **Explanation and error audit**
    - Inspect influential features, local explanations, high-confidence errors, and dataset artifacts.

13. **Synthesis and reporting**
    - Answer every research question and translate findings into practical recommendations.

Each stage has prerequisites, required artifacts, decision gates, and an objective definition of done in the local study protocol.

## Study quality audit

A local-only study-quality audit distinguishes repaired pipeline defects from remaining scientific limits. The current repository now enforces substantive-text filtering, deterministic exact-deduplication, parent-family split protection, honest controlled-split naming, validation-based baseline tuning, group-aware uncertainty, and experiment metadata. Remaining limits are stated in the current-status section and local study records.

## Planned models and analyses

### Required models

| Role | Planned method |
|---|---|
| Sanity-check baseline | Majority-class prediction |
| Primary traditional baseline | Word/bigram TF-IDF with Logistic Regression |
| Secondary traditional baseline | Word/bigram TF-IDF with Linear SVM |
| Nonsemantic controls | Publisher-only, unit-only, and length-only Logistic Regression |
| Text ablations | Title-only and body-only Logistic Regression |
| Transformer benchmarks | Pinned DistilBERT, ModernBERT, and BiomedBERT |
| Engagement-only model | Logistic Regression on structured engagement features |
| Combined model | Text-model scores plus standardized engagement features |

Additional models are optional and may not replace the required comparisons.

### Artifact controls

The evaluation will compare a standard stratified split against the strongest feasible controlled design:

1. canonical-publisher-disjoint split;
2. topic-held-out split;
3. temporal split;
4. artifact-masking sensitivity analysis when grouped splits are not feasible.

Publisher names, URLs, explicit source identifiers, review scores, verdict text, and other outcome-bearing fields are excluded from core text-model inputs.

The current primary cohort supports a canonical-publisher-disjoint split for
FakeHealth, with ten distinct valid allocations used to quantify split
sensitivity. CoAID cannot meet the frozen publisher-separation, prevalence,
concentration, and sample-size gates; its controlled condition is explicitly a
temporal family-disjoint fallback. Artifact removal deletes spans without
inserting learnable placeholder tokens; it remains a sensitivity analysis
rather than proof that the remaining language is causally meaningful.

### Engagement analysis

Potential features include:

- numbers of posts, reposts, replies, and participating users;
- response sentiment or emotion;
- temporal rates and burstiness;
- aggregate user or network indicators when already available and ethically usable.

Engagement modeling is conditional. It proceeds only if coverage and per-class sample-size requirements in the local study protocol are met. The project will not purchase API access merely to force this analysis.

FakeHealth passes the coverage gate and its matched complete-case ablation has
been run. Engagement-ID counts alone underperform text, and adding them to text
does not produce a reliable paired macro-F1 improvement. CoAID remains gated
because engagement availability differs substantially by label.

### Narrative analysis

BERTopic is the planned topic-modeling method. Computational topics will be treated as candidate narratives only after representative documents are reviewed manually.

A topic must not be interpreted as support for misinformation merely because it contains misinformation-related words. Documents may quote, discuss, reject, or debunk the same narrative.

### Explainability and error analysis

Planned methods include:

- signed Logistic Regression coefficients;
- signed Linear SVM feature weights;
- SHAP explanations;
- LIME where appropriate;
- stratified qualitative case review;
- a coded error taxonomy;
- comparisons of explanations across models, splits, and datasets.

Attention weights alone will not be treated as explanations.

---

## Evaluation framework

### Classification performance

The primary aggregate metric is **macro F1**.

Every required classifier will also report:

- precision, recall, and F1 for each class;
- PR-AUC for the dataset-defined unreliable class;
- confusion matrix;
- accuracy and weighted F1 as secondary metrics;
- ROC-AUC when valid probabilities or calibrated scores exist.

The dataset-defined unreliable class is treated as the positive class.

### Statistical uncertainty

Key metrics and paired model differences will use:

- 1,000 paired bootstrap resamples;
- 95% percentile confidence intervals;
- group-level resampling when article, source, or claim groupings define the independent units.

A small numerical increase will not automatically be described as a meaningful improvement.

### Calibration

Confidence quality will be evaluated using:

- Brier score;
- expected calibration error;
- reliability diagrams;
- confidence histograms;
- pre- and post-calibration comparisons.

The validation set is divided into calibration-fitting and method/threshold-selection halves. Platt scaling is the default learned calibrator; isotonic regression is conditional on sufficient calibration-fit data.

### Selective prediction

The project will test whether models can safely defer uncertain cases for human review.

Reported outputs will include:

- coverage;
- selective risk;
- risk-coverage curves;
- performance at fixed coverage levels;
- characteristics and error rates of deferred cases.

### Operating thresholds

The study will compare:

1. the default classification threshold;
2. a recall-oriented threshold intended to miss fewer unreliable items;
3. a precision-oriented threshold intended to reduce false flags against reliable material.

No universal threshold will be recommended without considering the operational context.

### Subgroup evaluation

Where sample sizes permit, performance will be reported by:

- dataset;
- source;
- candidate narrative;
- health topic;
- engagement availability;
- standard versus artifact-controlled setting.

Small subgroups will be described without unsupported inferential claims.

---

## Reproducibility rules

Every experiment must record:

- dataset revision and processed-data version;
- unit of analysis;
- label mapping;
- split manifest;
- random seed;
- preprocessing configuration;
- feature set;
- model and hyperparameters;
- calibration method;
- metrics;
- predictions and confidence scores;
- runtime environment.

The fixed project seed is `613`. Transformer experiments should use seeds `613`, `614`, and `615` when computationally feasible.

All learned transformations must be fit using training and validation data only. This includes:

- vocabulary construction;
- imputation;
- scaling;
- feature selection;
- probability calibration;
- confidence and decision thresholds.

The test labels must not be used to choose preprocessing, models, topic configurations, calibration methods, or thresholds.

Raw data under `data/raw/` must remain unchanged. Derived files should eventually be stored under `data/interim/` or `data/processed/`, with scripts that reproduce them.

---

## Documentation guide

| Document | Purpose |
|---|---|
| [Project context](docs/independent_study_project_context.md) | Broad project background, motivation, methodology options, timeline, and report planning |
| [Literature synthesis](docs/literature_review/literature_review_synthesis.md) | Thematic synthesis supporting the research design |
| [Literature brainstorming](docs/final_literature_brainstorming_document.md) | Dataset, method, contribution, and scope implications drawn from the reviewed literature |
| [Evaluation metrics](docs/independent_study_evaluation_metrics.md) | Compact record of the original model-evaluation criteria |
| [Dataset manifest](docs/dataset_acquisition_manifest.md) | Official sources, revisions, checksums, file counts, and acquisition decisions |
| [Study execution and results](docs/study_execution_and_results.md) | Detailed data attrition, pipeline logic, experiment rationale, computational findings, and validity limits |
| [Hyperparameter protocol](docs/hyperparameter_tuning_protocol.md) | Frozen validation-only search spaces and model-selection rules |
| [Manual review guide](docs/manual_review_guide.md) | Ordered narrative and qualitative-review gates that require human judgment |
| `docs/literature_review/` | Per-paper analytical review notes |
| `docs/literature_mds/` | Converted source text for local searching and reference |
| `docs/literature_pdfs/` | Original article PDFs |

The execution protocol, detailed decision log, working log, study-quality audit,
future-model plan, and draft final report are maintained locally and are not
published in the repository. For public documentation conflicts, use the
following precedence:

1. dataset acquisition manifest for source revisions and acquisition facts;
2. project context for broader background and earlier planning material;
3. this README for current public-facing status.

---

## Current project status

### Completed

- [x] Project topic and integrated pipeline framing established.
- [x] Core literature collected and converted for local review.
- [x] Per-paper review notes created.
- [x] Literature synthesis completed.
- [x] Robustness-audit contribution defined.
- [x] Five research questions approved and locked.
- [x] Detailed execution protocol created.
- [x] CoAID downloaded from the official repository.
- [x] FakeHealth downloaded from the official repository.
- [x] Dataset acquisition script added.
- [x] Optional FakeHealth Zenodo user-network archive downloaded and verified locally.
- [x] Exact dataset revisions and checksums recorded.
- [x] Analysis package, configs, phase scripts, and tests created.
- [x] Field-level inventory generated locally.
- [x] Label provenance and harmonized mappings generated locally.
- [x] Processed analysis tables and profile reports generated locally.
- [x] Primary cohort restricted to substantive CoAID articles and substantive FakeHealth stories/releases.
- [x] FakeHealth epoch dates normalized and CoAID publication dates prioritized over collection dates.
- [x] Publisher aliases and archived target domains canonicalized.
- [x] Scraper/interface pages and near-empty CoAID content excluded with coded reasons.
- [x] Contradictory and redundant exact duplicates excluded deterministically.
- [x] Empty strings incorporated into missingness reporting.
- [x] Connected-family standard manifests generated locally for both datasets.
- [x] FakeHealth canonical-publisher-disjoint manifest and ten distinct valid allocations generated locally.
- [x] CoAID publisher-disjoint infeasibility recorded under frozen gates; temporal family-disjoint fallback generated without overstating the control.
- [x] Traditional text baselines tuned on validation data and rerun locally.
- [x] Publisher-, unit-, length-, title-, and body-only controls completed locally.
- [x] FakeHealth rating-3 common-test sensitivity completed locally.
- [x] Group-aware bootstrap intervals and paired model differences completed locally.
- [x] Calibration fit and method/threshold selection separated within validation.
- [x] Calibration, reliability, operating-point, and selective-prediction outputs completed for Logistic Regression and Linear SVM.
- [x] Bidirectional cross-dataset transfer diagnostics completed after source-domain overlap filtering.
- [x] FakeHealth matched complete-case text, engagement, and combined ablations completed.
- [x] Artifact-feature and text-unit subgroup audit outputs generated locally.
- [x] Frozen, deterministic single-reviewer error-audit workflow implemented.
- [x] DistilBERT, ModernBERT, and BiomedBERT checkpoints pinned in configuration.
- [x] Transformer execution now returns a nonzero status and machine-readable failure record when incomplete.
- [x] CUDA/MPS acceleration auto-detection and Trainer-device verification implemented.
- [x] Traditional-model and shared transformer hyperparameter searches frozen and completed using validation data only.
- [x] All 63 final transformer runs completed across three architectures, three seeds, and the declared robustness splits on Apple MPS.
- [x] Standard-versus-artifact-masked paired equivalence analysis completed with the frozen ±0.03 margin.
- [x] Calibration/selective prediction, bidirectional transfer, and matched engagement ablation rerun on the archival cohort.
- [x] Candidate narrative discovery completed; CoAID produced no size-eligible candidate and FakeHealth produced four awaiting human validation.

### Next

- [ ] Review at least five representatives for each of the four FakeHealth
  candidate topics and complete the topic-decision sheet.
- [ ] Finalize retained narratives, then generate the frozen qualitative error roster.
- [ ] Complete blinded Error Stage A before generating or opening explanation artifacts.
- [ ] Complete Error Stage B and run the qualitative finalizer using
  [`docs/manual_review_guide.md`](docs/manual_review_guide.md).

### Later

- [ ] Produce the final report and recommendations.

---

## Environment and execution

The executable first-half analysis pipeline is implemented as Python package code plus phase scripts. Notebooks may be used for review, but the reproducible source of truth is the script pipeline.

The strict runtime is Python `3.11.15`, recorded in `.python-version`. The exact
resolved dependency graph is committed in `uv.lock`; hash-checked pip exports
are committed in `requirements.lock.txt` for the core study and
`requirements-narratives.lock.txt` for the optional narrative-analysis stack.

Preferred strict setup:

```bash
uv sync --frozen
```

Narrative discovery is optional and has its heavier BERTopic stack isolated in
the `narratives` dependency group:

```bash
uv sync --frozen --extra narratives
```

Pip-only fallback:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock.txt
python -m pip install -e .
```

For a pip-only environment that will run narrative discovery, replace
`requirements.lock.txt` above with `requirements-narratives.lock.txt`.

`requirements.txt` pins only the direct project dependencies for readability.
Use `uv.lock` or the matching core/narrative hash-locked export for strict
reproduction.

Run the unattended computational prerequisites after activating the
environment:

```bash
python scripts/01_inventory.py
python scripts/02_provenance_labels.py
python scripts/03_build_processed_tables.py
python scripts/04_make_splits.py
python scripts/05_run_text_baselines.py
python scripts/06_run_transformer.py
python scripts/07_run_calibration_selective.py
python scripts/10_run_cross_dataset_transfer.py
python scripts/11_run_engagement_ablation.py
python scripts/09_summarize_study.py
python -m pytest -q
```

Transformer execution reads `accelerator_preference` from
`configs/experiments.yaml` and selects the first available backend in the
declared order (`cuda`, then `mps`). CPU execution remains disabled unless
`--allow-cpu` is passed explicitly. Before training, the script verifies that
Hugging Face Trainer resolved the same backend and records the detected
capabilities and actual device in each run configuration.

On Apple silicon, verify MPS from the same execution context that will launch
training:

```bash
python -c "import torch; print(torch.backends.mps.is_available(), torch.mps.device_count())"
```

Sandboxed processes may not be able to see the Metal device even when it is
available to the same environment from a normal terminal.

Do not generate the final audit roster before transfer and narrative outputs are
settled. The optional narrative and staged single-reviewer phases are
deliberately not a single unattended command. Follow
[`docs/manual_review_guide.md`](docs/manual_review_guide.md), beginning with:

```bash
.venv/bin/python scripts/13_run_narrative_discovery.py
```

Current archival execution status:

- Phase 3 parsed 8,271 records and retained 5,192 primary records: 3,055 CoAID articles and 2,137 FakeHealth items.
- Phase 4 produced family-safe standard splits, a FakeHealth canonical-publisher-disjoint split with ten valid repetitions, and temporal family-disjoint manifests. CoAID correctly falls back to temporal evaluation because no publisher-disjoint allocation passes the frozen gates.
- Phase 5 produced 78 traditional-model, diagnostic-control, and sensitivity experiments with complete validation-tuning ledgers.
- Phase 6 completed six validation-only tuning trials and 63 final transformer runs on verified Apple MPS with zero failures or missing artifacts.
- Calibration/selective prediction completed for 22 eligible sparse-text experiments; transfer produced eight experiments and two shift bundles; engagement produced six matched ablations.
- Candidate discovery produced no size-eligible CoAID topic and four FakeHealth machine candidates. The human finalizer correctly reports `awaiting_human_validation` and created no human labels.

Generated results are local and keyed by `outputs/experiments/latest_run.json`.
Use the [study execution and results record](docs/study_execution_and_results.md)
for current findings and limitations; do not review superseded instruments from
older run directories.

Generated outputs are intentionally ignored by Git. The committed repository contains code, configs, tests, and documentation; it does not contain raw data, processed tables, model outputs, or predictions.

---

## Ethical and practical boundaries

This project analyzes publicly released research datasets. It does not collect new private health information and does not attempt to identify or profile individual users.

Required boundaries:

- Do not characterize dataset labels as unquestionable medical truth.
- Do not expose unnecessary user identifiers in reports or public artifacts.
- Do not use raw social-media identifiers as examples when paraphrased or anonymized cases are sufficient.
- Do not claim that model explanations establish causality or medical reasoning.
- Do not present model outputs as clinical advice.
- Do not recommend automated removal or censorship solely from classifier output.
- Treat uncertain, ambiguous, and high-risk cases as candidates for qualified human review.
- Report false-positive risks against reliable health information.
- Report false-negative risks where unreliable information is missed.

The practical target is decision support and triage, not autonomous adjudication.

---

## Data storage and version-control policy

### Raw data

- Keep original source files under `data/raw/`.
- Do not edit upstream files in place.
- Record source revisions and checksums.
- Do not commit derived files inside the upstream dataset repositories.

### Derived data

Derived data are separated into:

- `data/interim/` for parsed or partially transformed records;
- `data/processed/` for analysis-ready tables;
- `outputs/` for predictions, model artifacts, explanations, and topic models;
- `reports/` for final figures and tables.

Every derived artifact should have a reproducible generating script or notebook.

### Large and sensitive artifacts

Before publishing this project to a remote Git host:

- review dataset redistribution terms;
- avoid committing large model checkpoints;
- avoid committing credentials, API keys, or local environment files;
- decide whether raw datasets should be ignored, attached as releases, managed with Git LFS, or reproduced through acquisition scripts;
- remove OS and application artifacts such as `.DS_Store` and `.Rhistory`;
- review nested `.git` directories under `data/raw/`.

No repository-wide software or content license has yet been declared. Dataset and article licenses remain governed by their respective sources.

---

## References and dataset citations

### CoAID

Cui, L., & Lee, D. (2020). *CoAID: COVID-19 Healthcare Misinformation Dataset*. arXiv:2006.00885.  
Dataset: <https://github.com/cuilimeng/CoAID>

```bibtex
@misc{cui2020coaid,
  title        = {CoAID: COVID-19 Healthcare Misinformation Dataset},
  author       = {Limeng Cui and Dongwon Lee},
  year         = {2020},
  eprint       = {2006.00885},
  archivePrefix = {arXiv},
  primaryClass = {cs.SI}
}
```

### FakeHealth

Dai, E., Sun, Y., & Wang, S. (2020). *Ginger Cannot Cure Cancer: Battling Fake Health News with a Comprehensive Data Repository*. arXiv:2002.00837.  
Dataset repository: <https://github.com/EnyanDai/FakeHealth>  
Dataset archive: <https://zenodo.org/records/3862989>

```bibtex
@article{dai2020ginger,
  title   = {Ginger Cannot Cure Cancer: Battling Fake Health News with a Comprehensive Data Repository},
  author  = {Dai, Enyan and Sun, Yiwei and Wang, Suhang},
  journal = {arXiv preprint arXiv:2002.00837},
  year    = {2020}
}
```

Additional literature used to motivate the methodology and robustness framework is documented in `docs/literature_review/`.

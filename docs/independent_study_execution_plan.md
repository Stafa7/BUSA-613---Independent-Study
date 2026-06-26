# Independent Study Execution Plan

## Document purpose

This document is the operational protocol for completing the independent study on health misinformation detection, robustness, and decision support. It converts the broader project context into an ordered sequence of tasks, experiments, decision gates, outputs, and completion criteria.

Use this document as the source of truth for execution. Record deviations and their rationale in the decision log at the end of this file and summarize each work session in `docs/independent_study_working_log.md`.

Related documents:

- `docs/independent_study_project_context.md`
- `docs/independent_study_evaluation_metrics.md`
- `docs/literature_review/literature_review_synthesis.md`
- `docs/final_literature_brainstorming_document.md`
- `docs/independent_study_working_log.md`

---

## 1. Study contribution

The study is a robustness audit of health-information classification systems. It does not claim to introduce a new model architecture. Its contribution is a controlled evaluation of whether text models and engagement-informed models learn signals that are:

- effective within a dataset,
- robust to source and topic controls,
- transferable across datasets,
- calibrated enough to support confidence-based deferral,
- interpretable as health-information signals rather than dataset artifacts, and
- useful for narrative-level analysis and human decision support.

The study combines these tests in one reproducible pipeline using two datasets with different topical coverage and label provenance.

### Central claim to evaluate

> Health-information classifiers should be judged not only by aggregate predictive performance, but also by their robustness to dataset artifacts, transferability, calibration, ability to abstain on uncertain cases, narrative-level behavior, and explanation quality.

The study remains valid if advanced models fail to outperform baselines, engagement adds no consistent value, transfer performance is poor, or abstention is required for safe use. These are substantive findings rather than project failures.

---

## 2. Locked research questions

These questions are final. Any change requires a dated entry in the decision log and supervisor approval.

1. **How do traditional machine-learning and transformer-based models compare when classifying dataset-defined unreliable health information under standard and artifact-controlled within-dataset evaluation?**
2. **Do available social-engagement features add consistent predictive value over text-only models under the same evaluation controls?**
3. **To what extent do models transfer between CoAID and FakeHealth, and which differences in topic, text unit, or label provenance explain any performance degradation?**
4. **What recurring narratives characterize dataset-defined unreliable health information, and—where metadata permit—how do model performance, engagement, and emotional profiles vary across those narratives?**
5. **Can confidence calibration, selective prediction, and explanation-based auditing identify cases in which automated classifications should not be trusted?**

---

## 3. Terminology and interpretation rules

### 3.1 Required terminology

Use **dataset-defined unreliable health information** as the umbrella term when discussing CoAID and FakeHealth together.

Use each dataset's native terminology when describing its labels and document exactly how those labels were constructed. Do not refer to every positive-class item as verified claim-level misinformation.

### 3.2 Label interpretation

- A dataset label is evidence of the dataset's labeling process, not independent proof that every claim in an item is false.
- Source- or publisher-derived labels must not be presented as claim-level fact checks.
- FakeHealth review scores must be described as expert health-news quality assessments.
- Harmonized binary labels are analytical mappings created for this study and must be reported as such.
- Poor transfer may result from label mismatch, text-unit mismatch, topic shift, source shift, or model weakness. The analysis must consider all five explanations.

### 3.3 Unit of analysis

The preferred unit is an article or claim with a usable text field and one harmonized reliability label. Tweets, replies, and retweets are engagement records or auxiliary text unless a later documented decision explicitly makes them classification targets.

---

## 4. Scope

### 4.1 Required scope

- CoAID is the primary dataset.
- FakeHealth is the secondary validation and robustness dataset.
- Complete dataset and label-provenance audits.
- Train traditional text baselines.
- Train at least one transformer model.
- Compare standard and artifact-controlled evaluation.
- Test engagement features when the metadata gate passes.
- Attempt cross-dataset transfer only when the compatibility gate passes.
- Conduct narrative discovery with qualitative validation.
- Evaluate calibration and selective prediction.
- Perform explanation-based auditing and qualitative error analysis.
- Translate results into public-health communication and digital-risk-monitoring implications.

### 4.2 Explicit exclusions

- No live social-media scraping or required API hydration.
- No AI-generated misinformation dataset or synthetic-content research question.
- No claim that the system performs automated medical fact-checking.
- No new neural architecture is required.
- No full propagation-network reconstruction unless already available in the downloaded data.
- No KInIT extension, dashboard, multimodal model, multilingual model, or multiple-transformer benchmark until all required analyses are complete.

### 4.3 Priority order

If time or data limitations require reductions, preserve work in this order:

1. Provenance, leakage, and split audit.
2. Traditional baselines and DistilBERT.
3. Calibration, abstention, and error analysis.
4. Engagement ablations.
5. Cross-dataset transfer.
6. Narrative-level analysis.
7. Optional extensions.

Do not remove a locked research question silently. If a gate prevents an analysis, report the gate result as the answer to feasibility and document the limitation.

---

## 5. Reproducibility and experiment rules

### 5.1 Required experiment record

Every model run must record:

- dataset and version or download date,
- analysis unit and included fields,
- label mapping,
- split identifier,
- random seed,
- preprocessing version,
- feature set,
- model and hyperparameters,
- calibration method,
- evaluation metrics,
- saved predictions and probabilities,
- code or notebook version, and
- runtime environment.

Use a fixed project seed of `613` unless a method requires multiple seeds. Transformer results must use three seeds (`613`, `614`, and `615`) when computationally feasible; otherwise use seed `613` and disclose the limitation.

### 5.2 Data separation

- Fit vocabulary, imputation, scaling, feature selection, calibration, and threshold selection using training and validation data only.
- Never tune models, thresholds, topics, or calibration against the final test labels.
- Remove exact duplicates before splitting.
- Keep near-duplicates and records from the same article or claim in the same split.
- Preserve an immutable split manifest containing record identifiers and split assignments.

### 5.3 Comparison rule

Compare models on identical test records whenever possible. Report absolute results and paired differences. Do not describe a small numerical difference as an improvement unless its bootstrap confidence interval and practical magnitude support that interpretation.

---

## 6. Locked analytical design

### 6.1 Required model families

| Role | Required implementation | Purpose |
|---|---|---|
| Majority baseline | Most frequent training label | Minimum sanity check |
| Primary traditional model | TF-IDF + Logistic Regression | Interpretable probabilistic baseline |
| Secondary traditional model | TF-IDF + Linear SVM | Strong margin-based text baseline |
| Required transformer | DistilBERT fine-tuned for sequence classification | Feasible contextual-language benchmark |
| Engagement-only model | Logistic Regression on structured engagement features | Measure engagement signal without text |
| Combined model | Logistic Regression using text decision scores plus standardized engagement features | Controlled text-plus-engagement comparison |

If Linear SVM probabilities are needed, calibrate its decision scores on the validation set. Additional models are optional and may not replace the required set.

### 6.2 Text representations

- Logistic Regression and Linear SVM: word and bigram TF-IDF.
- DistilBERT: title plus body when both are available, truncated using a documented rule.
- Cross-dataset tests: use only text fields with defensible equivalents in both datasets.
- Source names, URLs, publisher names, author names, label-bearing fields, review scores, and review explanations are excluded from core text input.

### 6.3 Engagement feature groups

Use only fields present in the downloaded datasets. Candidate groups are:

- volume: tweet, share, retweet, reply, and user counts;
- response: reply sentiment or emotion summaries;
- temporal: time to first engagement, duration, burstiness, or rate;
- user/network: aggregate user or bot indicators, if already provided and ethically usable.

Create separate ablations by feature group before combining all available engagement features. Apply `log1p` to highly skewed nonnegative counts and standardize using training data only.

### 6.4 Narrative method

Use BERTopic as the primary topic-modeling method. Fit it to dataset-defined unreliable items after classification experiments are frozen. Manually inspect representative documents and assign descriptive narrative labels. Topic identifiers are not automatically treated as coherent misinformation narratives.

### 6.5 Explanation methods

- Logistic Regression: signed feature coefficients and local SHAP explanations.
- Linear SVM: signed feature weights and calibrated local inspection where available.
- DistilBERT: SHAP or LIME token-level explanations for a stratified sample of cases.
- All models: error taxonomy and comparison of explanations across standard and artifact-controlled settings.

Attention weights alone are not accepted as explanations.

---

## 7. Evaluation protocol

### 7.1 Core classification metrics

Report the following for every required classifier:

- macro F1 as the primary aggregate metric,
- positive-class precision, recall, and F1,
- negative-class precision, recall, and F1,
- PR-AUC for the unreliable class,
- confusion matrix,
- accuracy and weighted F1 as secondary descriptive metrics, and
- ROC-AUC only when probabilities or calibrated scores are available and both classes are represented.

The dataset-defined unreliable class is the positive class. Explain the asymmetric consequences of false negatives and false positives.

### 7.2 Statistical uncertainty

- Use 1,000 paired bootstrap resamples of test predictions.
- Report 95% percentile confidence intervals for core metrics and paired metric differences.
- Use the same resampled record indices for paired model comparisons.
- If grouping exists by article, source, or claim, bootstrap at the highest independent grouping level available.
- Treat statistical intervals as uncertainty estimates, not automatic declarations of scientific importance.

### 7.3 Calibration

For models that output probabilities or calibratable scores, report:

- Brier score,
- expected calibration error using 10 equal-width bins,
- reliability diagram,
- confidence histogram, and
- pre- and post-calibration results.

Fit Platt scaling on the validation set as the default calibration method. Test isotonic regression only when the validation set contains at least 1,000 records and at least 200 records per class. Select the method using validation Brier score; evaluate the selected method once on the test set.

### 7.4 Selective prediction

Evaluate confidence-based abstention using:

- coverage: proportion of records receiving a single automated prediction,
- selective risk: error rate among non-abstained predictions,
- risk-coverage curve,
- macro F1 at 100%, 90%, 80%, and 70% coverage, and
- composition and error rate of the deferred set.

Choose confidence thresholds on validation data. The analysis must state whether lower coverage produces a meaningful reduction in test error and which groups or narratives are disproportionately deferred.

### 7.5 Cost-sensitive thresholds

Report at least three validation-selected operating points:

1. default probability threshold,
2. threshold prioritizing recall of unreliable content, and
3. threshold prioritizing precision to reduce false flags against reliable content.

Do not prescribe one universal threshold. Explain the operational trade-off for public-health monitoring or human-review triage.

### 7.6 Optional conformal prediction

Conformal prediction is optional and may begin only after all core calibration and selective-prediction outputs are complete. If used, document the method, exchangeability assumptions, calibration set, target coverage, empirical coverage, and average prediction-set size. Do not claim unconditional guarantees under dataset shift.

---

## 8. Controlled split strategy

### 8.1 Standard split

Create a stratified 70/15/15 train/validation/test split at the highest available grouping level. The same record or parent article may not occur in multiple splits.

### 8.2 Artifact-controlled split

Use the following precedence:

1. **Source-disjoint split:** no publisher, domain, or source appears in more than one split.
2. **Topic-held-out split:** if source identifiers are unavailable or the source gate fails, hold coherent topic groups out of training.
3. **Temporal split:** if neither source nor defensible topic groups are available but reliable timestamps exist, train on earlier records and test on later records.
4. **Unavailable:** if none is feasible, document why and perform source/topic masking plus sensitivity analysis.

### 8.3 Source-disjoint gate

Proceed only if:

- at least five distinct usable sources exist in each class,
- every split can contain both classes,
- the test split contains at least 100 records and at least 30 records per class, and
- no single source must be divided to satisfy the class requirements.

If the gate fails, record the reason and move to the next split strategy.

### 8.4 Topic-held-out gate

Proceed only if:

- topic groups can be assigned without using test labels to tune the classifier,
- at least four coherent groups contain both labels or permit a meaningful held-out comparison,
- the held-out test data contain at least 100 records and at least 30 records per class, and
- the grouping procedure and representative examples can be reported.

### 8.5 Artifact-masking sensitivity test

For each dataset, compare core text models:

- with all permissible text,
- with explicit publisher, domain, URL, and source strings masked, and
- with any discovered label-bearing tokens removed.

A large performance drop is evidence that artifacts contributed to performance; it is not proof that all remaining features are causal or medically meaningful.

---

## 9. Sequential execution phases

### Phase 0 — Protocol freeze and workspace setup

**Prerequisites**

- Supervisor-approved study topic.
- This execution plan accepted as the working protocol.

**Tasks**

1. Create the planned data, notebook, source, output, table, and figure directories.
2. Create an environment specification and record package versions.
3. Establish naming conventions for experiments and artifacts.
4. Copy the five locked research questions into the proposal and report outline.
5. Start the decision log and record any supervisor-directed changes.

**Artifacts**

- Environment file.
- Project directory structure.
- Experiment naming guide.
- Initial decision-log entry.

**Research questions addressed**

- All, through protocol governance.

**Definition of done**

- A clean environment can import the required libraries.
- Every planned output location exists.
- The research questions in all active planning documents are consistent.

---

### Phase 1 — Data acquisition and immutable inventory

**Prerequisites**

- Phase 0 complete.
- Stable public download locations identified.

**Tasks**

1. Download CoAID and FakeHealth without relying on live platform APIs.
2. Preserve original archives and files unchanged in `data/raw/`.
3. Record source URL, access date, license or terms, file hashes, and reported dataset version.
4. Inventory all files, tables, fields, record counts, and join keys.
5. Identify which engagement records are already available locally.

**Artifacts**

- Raw-data manifest.
- File-hash table.
- Dataset inventory.
- Access and licensing note.

**Research questions addressed**

- RQ1–RQ5 by establishing feasible evidence.

**Definition of done**

- Both datasets have an inventory, or FakeHealth's access failure is documented with attempted sources and a supervisor-approved fallback.
- Raw files are preserved unchanged.
- Every later analysis field can be traced to an original file.

---

### Phase 2 — Provenance, label, and compatibility audit

**Prerequisites**

- Phase 1 complete.

**Tasks**

1. Document the creator, collection period, platform, health scope, text unit, source selection, and labeling method for each dataset.
2. Create a native-label dictionary.
3. Define the harmonized reliable/unreliable mapping without modifying raw labels.
4. Identify source-level, item-level, rating-based, or claim-level label construction.
5. Document label ambiguity, excluded categories, and possible proxy labels.
6. Complete the cross-dataset compatibility checklist.

**Cross-dataset compatibility checklist**

- Comparable binary construct can be defined.
- Comparable text unit can be extracted.
- Positive and negative classes exist in both datasets.
- The mapping does not require using outcome-bearing review text as model input.
- At least 100 usable records and 30 records per class remain in each test direction.
- Label differences can be stated and retained as an interpretation limitation.

**Artifacts**

- Dataset provenance table.
- Native and harmonized label dictionaries.
- Cross-dataset compatibility memo.
- Inclusion/exclusion flow table.

**Research questions addressed**

- RQ1, RQ3, and RQ5.

**Decision gate**

- **Proceed with RQ3 transfer** only if all compatibility checklist items pass.
- **Stop transfer modeling** if the construct or text unit cannot be mapped defensibly. Replace numerical transfer with a documented incompatibility analysis; do not force a common label.

**Definition of done**

- Every analytical label has a traceable origin and mapping.
- The transfer decision is recorded before model training.
- The report language does not equate dataset labels with verified claim-level truth.

---

### Phase 3 — Cleaning, profiling, and metadata feasibility

**Prerequisites**

- Phase 2 complete.

**Tasks**

1. Parse and normalize text without removing medically meaningful language.
2. Remove exact duplicates and group near-duplicates.
3. Quantify missingness, class balance, text length, language, dates, sources, topics, and engagement coverage.
4. Identify malformed records and document exclusions.
5. Profile engagement distributions and missingness by class and dataset.
6. Create data dictionaries and analysis-ready tables.

**Artifacts**

- Data-cleaning report.
- Dataset profile tables and figures.
- Missingness report.
- Duplicate and near-duplicate report.
- Engagement feasibility memo.
- Processed-data version and data dictionary.

**Research questions addressed**

- RQ1–RQ5.

**Engagement gate**

Proceed with predictive engagement experiments for a dataset only if:

- at least one meaningful engagement feature exists,
- feature coverage is at least 60% overall,
- at least 100 records per class have usable text and engagement data, and
- missingness does not deterministically reveal the label.

If the gate fails, stop predictive engagement modeling for that dataset. Retain descriptive analysis when at least 30 usable records per class exist and clearly label it exploratory.

**Definition of done**

- Record counts reconcile from raw to processed data.
- Every exclusion has a coded reason.
- The engagement decision is documented separately for CoAID and FakeHealth.

---

### Phase 4 — Leakage audit and frozen split manifests

**Prerequisites**

- Phase 3 complete.

**Tasks**

1. Search text and metadata for label names, review scores, verdict text, publisher identifiers, URLs, duplicate stories, and other outcome-bearing fields.
2. Build parent-item groups for related articles, claims, tweets, replies, or versions.
3. Create the standard stratified split.
4. Apply the artifact-controlled split precedence and gates in Section 8.
5. Create masked-text variants.
6. Freeze split manifests before model tuning.

**Artifacts**

- Leakage audit.
- Standard split manifest.
- Artifact-controlled split manifest or infeasibility memo.
- Masked-field specification.
- Split-distribution table.

**Research questions addressed**

- RQ1, RQ2, RQ3, and RQ5.

**Definition of done**

- No record family crosses splits.
- No prohibited outcome-bearing field remains in model input.
- Standard and controlled evaluations use frozen, reproducible manifests.

---

### Phase 5 — Traditional text baselines

**Prerequisites**

- Phase 4 complete.

**Tasks**

1. Train the majority baseline.
2. Tune TF-IDF Logistic Regression using training and validation data.
3. Tune TF-IDF Linear SVM using training and validation data.
4. Evaluate both models on standard and artifact-controlled tests.
5. Run artifact-masking sensitivity tests.
6. Save predictions, scores, coefficients, metrics, and bootstrap intervals.

**Artifacts**

- Baseline experiment table.
- Saved predictions and probabilities or scores.
- Confusion matrices.
- Bootstrap comparison table.
- Top positive and negative text features.
- Standard-versus-controlled performance table.

**Research questions addressed**

- RQ1 and RQ5.

**Definition of done**

- Both required traditional models have reproducible results on every feasible split.
- Performance changes under control and masking are quantified.
- Errors and influential features have been inspected before transformer training.

---

### Phase 6 — Transformer evaluation

**Prerequisites**

- Phase 5 complete.
- Adequate compute available locally or through an approved notebook environment.

**Tasks**

1. Fine-tune DistilBERT using the frozen training and validation records.
2. Use early stopping based on validation macro F1.
3. Run three seeds where feasible and report mean, range, and the predefined primary seed.
4. Evaluate on the same standard and controlled test records as the baselines.
5. Save probabilities, predictions, training curves, configuration, and checkpoints.
6. Compare DistilBERT against Logistic Regression and Linear SVM using paired bootstrap intervals.

**Artifacts**

- Transformer training record.
- Test predictions and probabilities.
- Learning curves.
- Model-comparison table.
- Compute and reproducibility note.

**Research questions addressed**

- RQ1 and RQ5.

**Decision gate**

If DistilBERT cannot be trained after one documented local attempt and one approved fallback-environment attempt, stop further transformer tuning. Report the failure and retain the traditional-model robustness study; do not spend the project schedule on infrastructure.

**Definition of done**

- DistilBERT is evaluated on all feasible frozen test sets, or the stop condition is documented.
- Comparisons use identical records and uncertainty intervals.

---

### Phase 7 — Engagement ablation experiments

**Prerequisites**

- Phase 3 engagement gate passed for the relevant dataset.
- Phase 5 text baseline complete.

**Tasks**

1. Create engagement-only feature sets by group.
2. Train engagement-only Logistic Regression models.
3. Create combined models using frozen text-model scores plus engagement features.
4. Compare text-only, engagement-only, and combined models on identical complete-case records.
5. Repeat comparisons under the artifact-controlled split when feasible.
6. Run ablations removing one engagement group at a time.
7. Examine whether missingness, age, or source acts as a proxy for the label.

**Artifacts**

- Engagement feature dictionary.
- Complete-case cohort table.
- Text-only versus engagement-only versus combined results.
- Ablation table.
- Engagement missingness and proxy-risk analysis.

**Research questions addressed**

- RQ2, RQ4, and RQ5.

**Interpretation rule**

Engagement adds **consistent predictive value** only if the combined model improves over the text-only model in the same direction on macro F1 or unreliable-class PR-AUC across both standard and controlled settings, with paired confidence intervals and no obvious leakage explanation. Otherwise report the result as inconsistent, dataset-specific, or unsupported.

**Definition of done**

- All feasible feature groups have been ablated.
- Comparisons use the same records and splits.
- Any apparent gain has been checked for missingness and source proxies.

---

### Phase 8 — Cross-dataset transfer

**Prerequisites**

- Phase 2 transfer gate passed.
- Phases 5 and 6 complete for the transferable text representation.

**Tasks**

1. Restrict both datasets to the harmonized labels and comparable text unit.
2. Apply the same minimum text preprocessing and prohibit dataset-specific source fields.
3. Train on CoAID and test on FakeHealth.
4. Train on FakeHealth and test on CoAID.
5. Compare transfer performance with each target dataset's in-domain result.
6. Quantify vocabulary, topic, text-length, source, class-balance, and label-provenance differences.
7. Inspect transfer errors by source and narrative where sample sizes permit.

**Artifacts**

- Harmonized transfer cohort.
- Bidirectional transfer results.
- In-domain versus out-of-domain degradation table.
- Domain-difference profile.
- Transfer error analysis.

**Research questions addressed**

- RQ3 and RQ5.

**Interpretation rule**

Do not attribute transfer degradation solely to model failure. Discuss text-unit mismatch, topic shift, source shift, prevalence shift, and label-provenance differences. Do not pool the datasets into a single performance number unless dataset identity remains visible in reporting.

**Definition of done**

- Both directions are evaluated, or a direction-specific sample-size failure is documented.
- Performance degradation and plausible dataset differences are quantified.
- Conclusions remain conditional on the harmonization assumptions.

---

### Phase 9 — Narrative discovery and subgroup evaluation

**Prerequisites**

- Classification test results frozen.
- At least 200 dataset-defined unreliable documents available for topic modeling in a dataset or pooled analysis.

**Tasks**

1. Fit BERTopic to dataset-defined unreliable content without using test metrics to choose a favorable model.
2. Review representative documents for each retained topic.
3. Merge incoherent duplicates and label coherent topics as candidate narratives.
4. Record at least five representative documents per retained narrative.
5. Mark mixed-stance or incoherent topics explicitly.
6. Evaluate model performance by narrative.
7. Compare engagement and emotional profiles by narrative when metadata gates pass.

**Artifacts**

- Topic-model configuration.
- Topic and candidate-narrative table.
- Representative-document audit sheet.
- Narrative prevalence chart.
- Narrative-level performance table.
- Conditional engagement and emotion comparisons.

**Research questions addressed**

- RQ4 and RQ5.

**Narrative retention gate**

Retain a candidate narrative only if:

- it contains at least 30 documents,
- at least five representative documents support a coherent interpretation, and
- the label does not merely repeat a publisher, dataset, or generic stopword pattern.

For performance comparisons, require at least 20 test records in the narrative and at least five examples of each relevant true class. Otherwise report examples and prevalence without inferential comparison.

For engagement or emotion comparisons, require at least 30 metadata-complete records in each compared group. Otherwise mark the comparison unavailable.

**Definition of done**

- Every reported narrative has qualitative evidence.
- Topic mention is not interpreted as endorsement without supporting examples.
- Small subgroups are clearly separated from evaluated subgroups.

---

### Phase 10 — Calibration, selective prediction, and operating points

**Prerequisites**

- Final candidate models and frozen validation/test predictions available.

**Tasks**

1. Measure uncalibrated confidence quality.
2. Fit Platt scaling and, when eligible, isotonic regression on validation data.
3. Select the calibration method using validation Brier score.
4. Evaluate Brier score, expected calibration error, and reliability on test data.
5. Produce risk-coverage curves and fixed-coverage results.
6. Analyze deferred cases by class, source, narrative, dataset, and error type.
7. Evaluate recall-oriented and precision-oriented thresholds.
8. Optionally run conformal prediction after all core outputs are complete.

**Artifacts**

- Calibration table and reliability plots.
- Risk-coverage curves.
- Fixed-coverage results.
- Deferred-case profile.
- Operating-point comparison.
- Optional conformal-prediction appendix.

**Research questions addressed**

- RQ5, with supporting evidence for RQ1–RQ4.

**Trust rule**

A model is not described as trustworthy merely because it is accurate. The report must jointly discuss discrimination, calibration, controlled-split robustness, deferred-case behavior, and explanations.

**Definition of done**

- Every final model has a calibration assessment.
- At least one abstention strategy is evaluated.
- The report identifies which cases should be routed to human review and the cost in coverage.

---

### Phase 11 — Explanation audit and qualitative error analysis

**Prerequisites**

- Phases 5–10 complete.

**Tasks**

1. Inspect global feature importance for traditional models.
2. Select a stratified audit sample containing:
   - true positives,
   - true negatives,
   - false positives,
   - false negatives,
   - high-confidence errors,
   - low-confidence deferred cases,
   - transfer errors, and
   - examples from retained narratives.
3. Generate local explanations for the selected cases.
4. Compare explanations across standard and artifact-controlled models.
5. Code errors using the taxonomy below.
6. Assess whether explanations rely on health-relevant language, emotional framing, source artifacts, topic shortcuts, or annotation ambiguity.

**Required error taxonomy**

- source or publisher artifact,
- topic shortcut,
- temporal or event-specific cue,
- label ambiguity,
- text-unit mismatch,
- insufficient context,
- discussion or debunking mistaken for endorsement,
- medical terminology misunderstanding,
- sensational or emotional framing,
- truncation or preprocessing failure,
- engagement proxy or missingness artifact, and
- unexplained residual error.

**Artifacts**

- Explanation audit table.
- Error taxonomy with counts and examples.
- High-confidence-error case studies.
- Standard-versus-controlled explanation comparison.
- Human-review recommendation criteria.

**Research questions addressed**

- RQ5 and explanatory evidence for RQ1–RQ4.

**Explanation stability rule**

Treat an explanation pattern as stable only when it recurs across multiple examples and remains visible across at least two relevant settings, such as models, seeds, splits, or datasets. Single local explanations are illustrative, not general evidence.

**Definition of done**

- All required error categories have been considered.
- High-confidence errors and deferred cases are explicitly analyzed.
- Explanations are not presented as causal proof.

---

### Phase 12 — Synthesis, recommendations, and final report

**Prerequisites**

- Required analytical phases complete or stopped through documented gates.

**Tasks**

1. Build the final results-to-research-question matrix.
2. Separate confirmatory, robustness, and exploratory findings.
3. State answers to each research question with evidence and limitations.
4. Translate findings into a human-review workflow.
5. Document which outputs are suitable for public-health communication or risk monitoring.
6. Write the final report and reproducibility appendix.
7. Archive final tables, figures, predictions, configurations, and decision logs.

**Artifacts**

- Final results matrix.
- Final report.
- Reproducibility appendix.
- Practical recommendation summary.
- Archived analysis package.

**Research questions addressed**

- RQ1–RQ5.

**Definition of done**

- Every research question has a direct, evidence-based answer.
- All unavailable analyses cite their failed gate.
- Claims use dataset-appropriate terminology.
- Another researcher can reconstruct the analytical sequence from the archived materials.

---

## 10. Results-to-research-question matrix

| Experiment or analysis | RQ1 | RQ2 | RQ3 | RQ4 | RQ5 |
|---|:---:|:---:|:---:|:---:|:---:|
| Provenance and label audit | ✓ |  | ✓ |  | ✓ |
| Standard within-dataset baselines | ✓ |  |  |  | ✓ |
| Source-disjoint/topic-held-out evaluation | ✓ | ✓ |  |  | ✓ |
| Artifact masking | ✓ | ✓ |  |  | ✓ |
| DistilBERT comparison | ✓ |  |  |  | ✓ |
| Engagement-only and combined ablations |  | ✓ |  | ✓ | ✓ |
| Bidirectional cross-dataset transfer |  |  | ✓ |  | ✓ |
| Domain and provenance difference analysis |  |  | ✓ | ✓ | ✓ |
| BERTopic and qualitative narrative validation |  |  |  | ✓ | ✓ |
| Narrative-level model evaluation | ✓ | ✓ | ✓ | ✓ | ✓ |
| Calibration and reliability analysis | ✓ | ✓ | ✓ | ✓ | ✓ |
| Selective prediction and deferred-case analysis | ✓ | ✓ | ✓ | ✓ | ✓ |
| Explanation audit and error taxonomy | ✓ | ✓ | ✓ | ✓ | ✓ |

Every reported experiment must reference at least one row in this matrix. Any new experiment must be added to the matrix before execution.

---

## 11. Deliverable checklist

### Proposal deliverable

- [ ] Locked research questions.
- [ ] Robustness-audit contribution statement.
- [ ] Dataset descriptions and label-provenance caveat.
- [ ] Required and conditional methods.
- [ ] Evaluation protocol.
- [ ] Feasibility gates and risks.
- [ ] Semester timeline.

### Literature review and preliminary findings deliverable

- [ ] Literature synthesis.
- [ ] Industry and public-health relevance.
- [ ] Dataset inventories and provenance tables.
- [ ] Label harmonization decision.
- [ ] Dataset profiles and cleaning flow.
- [ ] Leakage audit and frozen split design.
- [ ] Traditional baseline results.
- [ ] Preliminary controlled-split comparison.
- [ ] Engagement and transfer gate decisions.
- [ ] Updated risk register.

### Final deliverable

- [ ] Traditional and transformer model comparison.
- [ ] Standard and artifact-controlled evaluation.
- [ ] Engagement ablations where feasible.
- [ ] Cross-dataset transfer or documented stop decision.
- [ ] Narrative discovery and qualitative validation.
- [ ] Narrative-level evaluation where sample sizes permit.
- [ ] Calibration and reliability plots.
- [ ] Selective-prediction analysis.
- [ ] Explanation audit and error taxonomy.
- [ ] Bootstrap confidence intervals.
- [ ] Direct answers to all five research questions.
- [ ] Practical recommendations and limitations.
- [ ] Reproducibility appendix.

---

## 12. Risk register

| Risk | Trigger or evidence | Required response |
|---|---|---|
| Dataset unavailable | Stable public files cannot be downloaded | Document attempts; keep CoAID core and request approval before substituting FakeHealth |
| Label incompatibility | Harmonized mapping fails Phase 2 checklist | Stop numerical transfer; report incompatibility as a methodological result |
| Engagement data insufficient | Phase 3 coverage or sample gate fails | Stop predictive engagement analysis; retain eligible descriptive analysis |
| Source-disjoint split infeasible | Phase 4 source gate fails | Use topic-held-out, temporal, or masking sensitivity in the required order |
| Strong source leakage | Large controlled-split or masking performance drop | Center the finding in RQ1/RQ5; avoid claim-level interpretation |
| Severe class imbalance | Minority class too small for stable splits | Use class weights, PR-AUC, grouped resampling, and disclose uncertainty; do not create synthetic text |
| Transformer compute failure | Local and fallback attempts fail | Stop transformer tuning and complete the robustness study with traditional models |
| Cross-dataset collapse | Transfer approaches chance performance | Treat as evidence of limited transportability; investigate provenance and domain shift |
| Noisy topics | Topics fail coherence and representative-document checks | Reduce topic count, merge carefully, or report that reliable narratives were not recovered |
| Small narrative groups | Retained group fails subgroup gate | Report prevalence/examples only; do not make performance claims |
| Poor calibration | Reliability and Brier results remain weak after calibration | Recommend mandatory human review or non-deployment; quantify abstention behavior |
| Explanations dominated by artifacts | Source/topic cues recur across audit cases | Report the model as artifact-sensitive; do not frame explanations as medical reasoning |
| Scope expansion | Optional work begins before required outputs finish | Stop optional work and return to the priority order in Section 4.3 |

---

## 13. Final report mapping

| Final report section | Required execution-plan evidence |
|---|---|
| Introduction | Problem, robustness-audit contribution, and locked questions |
| Background and literature review | Dataset heterogeneity, engagement, narratives, transfer, and explainability evidence |
| Industry analysis | Human-review, communication, and digital-risk-monitoring use cases |
| Data | Inventory, provenance, label mapping, cleaning flow, profiles, and limitations |
| Methodology | Frozen splits, models, ablations, transfer gate, topic method, calibration, abstention, and explanations |
| Results | Core metrics, confidence intervals, controlled tests, transfer, narratives, calibration, and errors |
| Findings | Direct answers to RQ1–RQ5 |
| Recommendations | Operating points, deferral workflow, appropriate and inappropriate uses |
| Discussion | Robustness, artifacts, label limitations, external validity, and ethical implications |
| Conclusion | What generalizes, what does not, and what requires human judgment |
| Appendix | Hyperparameters, split manifests, extra tables, reproducibility details, and decision log |

---

## 14. Decision log

Add an entry before implementing any material deviation from this protocol.

| Date | Decision | Reason and evidence | Impact on research questions | Approved by |
|---|---|---|---|---|
| 2026-06-25 | Lock the five research questions in Section 2 | Questions were refined against the reviewed literature and approved before implementation | Establishes final RQ1–RQ5 | Student |
| 2026-06-25 | Frame the contribution as a robustness audit | Individual methods already exist; the contribution is their controlled integration across datasets | Centers artifact controls, transfer, calibration, abstention, and explanation auditing | Student |
| 2026-06-25 | Exclude AI-generated content | It would materially change the study and dilute the existing design | No impact; keeps all five questions focused on CoAID and FakeHealth | Student |
| 2026-06-25 | Make calibration and selective prediction core | Confidence-aware deferral provides a measurable trust and human-review analysis | Strengthens RQ5 | Student |

### Blank entry template

| Date | Decision | Reason and evidence | Impact on research questions | Approved by |
|---|---|---|---|---|
| YYYY-MM-DD |  |  |  |  |

---

## 15. Final completion test

The study is complete only when all statements below are true:

- [ ] The five locked research questions appear verbatim in the final report.
- [ ] Every executed experiment maps to at least one research question.
- [ ] Every phase is complete or has a documented stop decision tied to an explicit gate.
- [ ] CoAID and FakeHealth labels are described according to their provenance.
- [ ] Standard and artifact-controlled evaluation results are reported where feasible.
- [ ] Traditional baselines and DistilBERT are compared, or the transformer stop condition is documented.
- [ ] Engagement claims are based on matched-record ablations and pass the metadata gate.
- [ ] Transfer claims pass the compatibility gate and retain label-provenance caveats.
- [ ] Every reported narrative has qualitative validation and sufficient records.
- [ ] Final models have calibration, reliability, and selective-prediction results.
- [ ] High-confidence errors and deferred cases are audited.
- [ ] Confidence intervals accompany key performance comparisons.
- [ ] Explanations are used for auditing and illustration, not causal proof.
- [ ] Recommendations distinguish automated triage from verified medical fact-checking.
- [ ] All important deviations are recorded in the decision log.
- [ ] Final tables, figures, predictions, configurations, and split manifests are archived.


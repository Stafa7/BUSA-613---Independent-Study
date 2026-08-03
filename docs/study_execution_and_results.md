# Study execution, data audit, and computational results

This document is the plain-language technical record for the archival study
run. It explains what the datasets contain, why records leave the primary
cohort, what each experiment is intended to test, and how to interpret the
results without treating dataset labels as independently verified medical
truth. Generated predictions and model artifacts remain local under
`outputs/`; the code, frozen configuration, and interpretation rules are
tracked in Git.

## Big picture

The study is a robustness audit of health-information classifiers. A high
score on a random-looking train/test split is not sufficient evidence that a
model recognizes unreliable health information. It may instead recognize a
publisher, duplicated article family, collection period, platform phrase, or
other dataset construction artifact. The experimental sequence therefore asks
progressively harder questions:

1. Can conventional and transformer text models predict each dataset's label?
2. Does performance survive family-, publisher-, time-, domain-, and artifact
   controls?
3. Does a model trained on one dataset transfer to the other?
4. Does engagement add information beyond text on the records where engagement
   is available?
5. Are scores calibrated, and can abstention reduce the error rate among cases
   the system elects to classify?
6. Do discovered narratives and model explanations remain meaningful under
   blinded human review?

The first five questions are computational. The sixth deliberately has human
gates and cannot be completed by generating plausible-looking annotations.

## What CoAID and FakeHealth contain

### CoAID

CoAID is distributed as dated CSV collections. Its content units are news
articles and short claims, grouped natively as Real or Fake, plus CSV mappings
to related social posts and replies. The repository contains 5,975 content
rows: 5,457 article rows and 518 claim rows. Article bodies are separate from
some metadata records and are not available for every row. The harmonized
target retains the native label provenance and maps it to `reliable` or
`unreliable`; it does not turn a source/dataset label into a new claim-level
fact check.

### FakeHealth

FakeHealth consists of HealthStory and HealthRelease. It combines article JSON
content with expert HealthNewsReview records, criteria, explanations, and
social-post identifier mappings. This study parses 1,690 story reviews and 606
news-release reviews. Outcome-bearing expert-review text and criteria are never
used as classifier input. The numerical review rating is preserved, then
mapped to the common analytical target; this is a reporting-quality/reliability
proxy, not the same labeling construct as CoAID.

### Completeness of the local acquisition

| Dataset/unit | Parsed metadata rows | Body files available | Primary eligible |
|---|---:|---:|---:|
| CoAID article | 5,457 | 4,470 (81.9%) | 3,055 (56.0% of article rows) |
| CoAID claim | 518 | 0 | 0 |
| FakeHealth story | 1,690 | 1,638 (96.9%) | 1,544 (91.4%) |
| FakeHealth news release | 606 | 599 (98.8%) | 593 (97.9%) |

Overall, the primary cohort retains 3,055/5,975 CoAID rows (51.1%) and
2,137/2,296 FakeHealth rows (93.1%). Those percentages describe fitness for
this study's substantive-body article analysis, not whether the upstream
datasets are defective or incomplete for every possible use.

## What happens to every record

The pipeline is deterministic and retains excluded rows in the processed table
with a coded reason. It does not silently discard them.

1. `01_inventory.py` inventories source files, table structures, fields, sizes,
   and hashes.
2. `02_provenance_labels.py` records native labels, the harmonized mapping, and
   the incompatibilities between labeling constructs.
3. `03_build_processed_tables.py` parses metadata and body files; normalizes
   dates, publishers, and hosting domains; constructs text and artifact-masked
   text; links engagement counts; forms exact-duplicate and connected-family
   identifiers; applies exclusions; and writes all parsed rows plus profile
   audits.
4. `04_make_splits.py` freezes standard and robustness manifests, then checks
   family/publisher/domain leakage, class balance, record counts, and effective
   group counts.
5. Later scripts fit only on `train`, choose model/calibration/threshold settings
   only on `validation`, and evaluate the locked choice once on `test`.

### Exclusive primary-cohort attrition

| Dataset | Reason | Records | Why excluded |
|---|---|---:|---|
| CoAID | access-denied page | 1,085 | Retrieved body is an access-denial page, not article content. |
| CoAID | insufficient body text | 619 | Fewer than the frozen 30 body-word minimum. |
| CoAID | non-primary text unit | 460 | Claims differ materially from full articles and have no body files. |
| CoAID | near-empty text | 411 | Parsed page has too little substantive content. |
| CoAID | exact duplicate | 178 | A deterministic representative is retained to prevent duplication. |
| CoAID | forbidden page | 91 | Retrieved body is a forbidden-response page. |
| CoAID | login/signup page | 41 | Retrieved body is an interface/authentication page. |
| CoAID | JavaScript-disabled page | 27 | Retrieved body is a browser placeholder. |
| CoAID | conflicting exact-duplicate labels | 8 | Identical text carries contradictory labels, so neither version is trusted. |
| FakeHealth | insufficient body text | 137 | Fewer than the frozen 30 body-word minimum. |
| FakeHealth | JavaScript-disabled page | 22 | Retrieved body is a browser placeholder. |

The main CoAID loss is therefore unavailable or non-substantive web content,
not an arbitrary class-balancing operation. The study never downsamples the
primary cohort merely to equalize labels. Class imbalance is handled with
balanced linear-model weights, macro F1, per-class metrics, PR-AUC, and grouped
bootstrap uncertainty.

## Where the audits live

The reproducible local audit trail is separated by purpose:

- `outputs/inventory/`: file, table, field, and hash inventories;
- `outputs/provenance/`: native labels, harmonized mappings, dataset provenance,
  and the transfer-compatibility memo;
- `outputs/profiles/`: data dictionary, missingness, parse-issue ledger,
  content-quality audit, exclusive inclusion/exclusion flows, duplicate and
  connected-family reports, date audit, language/Unicode audit, and publisher
  canonicalization audit;
- `outputs/splits/`: frozen manifests, split decisions, leakage audit, split
  integrity, class/record balance, and effective bootstrap-group counts;
- `outputs/experiments/runs/<run_id>/`: per-experiment configuration, validation
  tuning, predictions, metrics, uncertainty intervals, calibration, transfer,
  and later human-review instruments.

These generated files are ignored by Git because they include large derived
data, predictions, and model artifacts. Their generators and schemas are
tracked, and `outputs/experiments/latest_run.json` identifies the active run.

## Frozen split designs and why each exists

| Design | Purpose | Important interpretation |
|---|---|---|
| Standard family-grouped | Estimates ordinary within-dataset performance while keeping duplicates/related families together. | Still permits publishers and domains to appear across partitions. |
| Controlled | Tests a stronger shortcut control. FakeHealth is publisher-disjoint; CoAID uses the declared temporal family-disjoint fallback because no publisher-disjoint allocation met the frozen feasibility gates. | The two datasets' controlled splits answer related but non-identical questions. |
| Repeated controlled | Repeats valid FakeHealth publisher-disjoint allocations. | Shows whether a conclusion depends on one favorable publisher allocation. |
| Hosting-domain sensitivity | Holds FakeHealth hosting domains out of training/validation. | A demanding domain-shift stress test with few effective domain groups; uncertainty must be emphasized. |
| Temporal | Trains on earlier and evaluates on later family groups where dates permit. | Measures temporal shift and can change class prevalence. |
| Artifact-masked | Uses the standard records but removes URLs, source names, platform/review/release phrases, and explicit label terms. | Near-equivalence supports, but cannot prove, reduced dependence on the named artifacts. |
| Rating-3 excluded | Removes FakeHealth's boundary rating on a common test cohort. | Tests whether the binary mapping is driven by ambiguous borderline items. |

All manifests keep connected `family_group` values together. Integrity audits
verify a maximum cross-split count of one for the governing group. Group-aware
bootstrap intervals resample the appropriate family, publisher, or domain unit
rather than pretending correlated records are independent.

## Hyperparameter tuning

Tuning is needed, but broad optimization would be counterproductive on these
modest validation sets. The frozen search is deliberately limited and uses no
test labels.

For TF-IDF Logistic Regression and Linear SVM, every dataset/split/text cohort
compares 24 configurations: 10k/20k maximum features, minimum document
frequency 5/2, ordinary/sublinear term frequency, and `C` 0.1/1/10. Word
unigrams and bigrams are fixed. Selection is by validation macro F1, the winner
is refit on training data, and test data are used once.

For transformers, DistilBERT is the protocol-calibration model. Learning rates
`1e-5`, `2e-5`, and `3e-5` are compared on the standard CoAID and FakeHealth
validation partitions using seed 613. Their mean validation macro F1 values
were 0.7737, 0.7664, and 0.7593, respectively, so `1e-5` was locked for all
three architectures. The audit records six tuning fits and zero test records.
This common rate makes the architecture comparison fairer, though it may not be
the individually optimal rate for ModernBERT or BiomedBERT.

## Experiment-by-experiment logic

### Traditional text baselines and diagnostic controls

- Majority class establishes the no-text floor.
- TF-IDF Logistic Regression provides an interpretable probabilistic linear
  baseline.
- TF-IDF Linear SVM tests a strong margin-based sparse-text classifier.
- Full text, title only, and body only show where predictive signal resides.
- Publisher only, text-unit only, and document-length only quantify obvious
  non-semantic shortcuts rather than competing as deployable models.
- Standard versus controlled, temporal, domain-disjoint, masked, repeated, and
  rating-boundary sensitivities test whether conclusions survive specific
  threats to validity.

### Transformer comparison

DistilBERT is a compact general-domain reference; ModernBERT tests a newer
long-context-era encoder at the same frozen 512-token input; BiomedBERT tests
whether biomedical pretraining helps. Each architecture is evaluated with
seeds 613--615 on standard, controlled, masked, and applicable domain-shift
cohorts. The three seeds measure optimization variability; grouped bootstrap
intervals measure test-sample uncertainty. They answer different questions and
are both retained.

Input is dynamically padded and truncated at 512 tokens. Truncation is minor
for CoAID but material for FakeHealth: 1,749/2,137 FakeHealth primary records
(81.8%) exceed 512 tokens, with median length 858 and 95th percentile 1,743
tokens. FakeHealth results are therefore results for the leading 512-token
window, not the entire article. Comparing ModernBERT at a longer context would
be a justified future experiment, but doing so inside the current architecture
comparison would change two factors at once.

### Calibration and selective prediction

Raw validation predictions are divided into a calibration-fit half and a
method/threshold-selection half, preferably group-disjoint. Platt scaling is
always considered; uncalibrated probabilities are considered for Logistic
Regression; isotonic regression is allowed only with enough fit records and
both classes. Lowest validation Brier score selects the method. Validation then
sets macro-F1-, recall-, precision-, and coverage-oriented operating points.
Only after those choices are frozen are Brier score, expected calibration
error, reliability, operating-point performance, and risk-versus-coverage
reported on test data.

This serves a practical purpose: a review tool needs probabilities that mean
something and a defensible rule for deferring uncertain cases. Calibration
does not make predictions medically correct, and selective-prediction gains
must be weighed against the cases left for humans.

### Bidirectional cross-dataset transfer

The study trains tuned TF-IDF models on one dataset's training data and tests
them on the other dataset's held-out test data after removing target domains
seen in the source training/validation cohort. Both CoAID-to-FakeHealth and
FakeHealth-to-CoAID are run with full text/title and Logistic Regression/SVM.
Vocabulary overlap, document length, source/domain composition, categorical
shift, and topic proxies are recorded alongside performance degradation.

Transfer is a stringent external-validity test, but it confounds legitimate
differences in topic, unit, time, and label provenance. Poor transfer means the
learned decision rule is not portable as configured; it does not by itself
identify which dataset or model is wrong.

### FakeHealth engagement ablation

On FakeHealth records with available engagement IDs, the same complete cases
and split are used for text only, engagement counts only, and text plus
engagement. Engagement counts are unique post/reply/repost-ID volumes, log
transformed and standardized within training. Validation selects `C`; paired
group bootstrap differences compare test predictions.

Matching the cohort is essential: otherwise a difference could be caused by
which records have engagement metadata. Even with matching, this is
associational. Exposure time and reach are unavailable, and engagement after
publication may be a consequence rather than a cause of content visibility.

### Narrative discovery and human validation

BERTopic proposes candidate clusters among dataset-defined unreliable records.
Machines may supply representative documents and keywords, but a single human
reviewer must decide whether a cluster is coherent, distinguish endorsement
from discussion/debunking, reject publisher/dataset/generic artifacts, and
author the retained narrative label. CoAID can legitimately yield no eligible
topics under the frozen size/stability gates. This phase is exploratory and
descriptive, not a prevalence estimate of real-world misinformation.

### Explanation and qualitative error audit

A deterministic, stratified roster samples ordinary correct/error cells,
high-confidence errors, deferred cases, transfer errors, and retained-narrative
examples. Stage A blinds labels, predictions, and explanations so the reviewer
judges content/ambiguity first. Only after Stage A is finalized may explanation
artifacts be generated and Stage B opened. The audit examines failure modes and
whether highlighted features are useful to a reviewer; feature attribution is
not causal reasoning or a factual justification.

## Computational results

The unattended archival computation completed on August 2, 2026 under run
`20260802T034237Z_cf102ec1_891ec46a`. The run contains 78 traditional baseline
and diagnostic experiments, 63 final transformer runs, six transformer-tuning
trials, 22 calibration/selective-prediction analyses, eight transfer
experiments, six engagement ablations, two narrative-discovery fits, and all
declared paired summaries. The qualitative stages remain open because they
require actual human judgments.

### Compute and tuning audit

The transformer script detected `cuda_available=false`, `mps_available=true`,
selected `mps`, and verified that Hugging Face Trainer also resolved `mps`
before training. The archival host reports an Apple M4 Pro with a 20-core GPU
and 48 GB of unified memory; the environment uses ARM64 Python 3.11.15 and an
MPS-enabled PyTorch 2.12.1 build. All 63 final run directories have a completed status,
configuration, validation and test predictions, metrics, grouped bootstrap
intervals, training curve, and checkpoint provenance. All 63 configurations
record `trainer_device: mps`; there are no failed models, missing artifacts, or
CPU overrides.

The transformer learning-rate audit contains all six DistilBERT
dataset-by-rate validation fits, the locked `1e-5` selection, the actual MPS
device, and zero test records. Every non-majority sparse-text experiment stores
24 validation candidates and the selected full TF-IDF/model configuration.

### Traditional text models

| Dataset/split | Logistic macro F1 (95% grouped bootstrap CI) | SVM macro F1 | Logistic unreliable F1 | Logistic unreliable PR-AUC |
|---|---:|---:|---:|---:|
| CoAID standard | 0.857 (0.790–0.912) | 0.846 | 0.735 | 0.765 |
| CoAID controlled/temporal | 0.675 (0.589–0.765) | 0.719 | 0.375 | 0.461 |
| FakeHealth standard | 0.677 (0.620–0.728) | 0.672 | 0.580 | 0.625 |
| FakeHealth controlled/publisher-disjoint | 0.660 (0.594–0.717) | 0.643 | 0.550 | 0.566 |
| FakeHealth hosting-domain-disjoint | 0.516 (0.266–0.774) | 0.537 | 0.388 | 0.301 |

The CoAID standard score is useful but highly vulnerable to source identity:
publisher-only Logistic Regression reaches macro F1 0.978 on standard and
0.938 on the temporal controlled split. In paired group bootstraps, full text
is worse than publisher-only by 0.121 (95% CI 0.069–0.184 worse) on standard
and 0.263 (0.173–0.356 worse) on controlled. CoAID's controlled split cannot
hold publishers apart, so its improvement over length-only (0.583, 95% CI
0.498–0.679) shows information beyond length but does not eliminate publisher
shortcut concerns.

FakeHealth provides the stronger source-control result. On its
publisher-disjoint test, full text exceeds publisher-only by 0.434 macro F1
(95% CI 0.323–0.535). Across ten distinct valid publisher-disjoint allocations,
text Logistic Regression averages 0.651 (SD 0.062; range 0.501–0.741), showing
that the exact publisher allocation still matters. On the domain-disjoint
stress test, its wide 0.266–0.774 interval reflects only about five effective
domain groups, not high certainty around a 0.516 point score.

FakeHealth's rating-3 boundary matters. On the same 211 non-rating-3 standard
test records, excluding rating-3 examples from training raises Logistic macro
F1 from 0.735 to 0.791. The paired primary-minus-excluded difference is -0.056
(95% CI -0.112 to -0.003). This supports reporting the boundary-label
sensitivity rather than treating the binary mapping as unambiguous.

### Transformer comparison

The table reports mean macro F1 ± sample SD across seeds 613–615.

| Dataset/split | DistilBERT | ModernBERT | BiomedBERT |
|---|---:|---:|---:|
| CoAID standard | 0.908 ± 0.016 | 0.908 ± 0.018 | **0.914 ± 0.015** |
| CoAID controlled/temporal | 0.836 ± 0.022 | 0.863 ± 0.029 | **0.888 ± 0.009** |
| CoAID artifact-masked | 0.904 ± 0.011 | 0.905 ± 0.003 | **0.912 ± 0.015** |
| FakeHealth standard | 0.651 ± 0.031 | **0.674 ± 0.010** | 0.658 ± 0.011 |
| FakeHealth controlled/publisher-disjoint | 0.638 ± 0.048 | **0.649 ± 0.021** | 0.591 ± 0.016 |
| FakeHealth hosting-domain-disjoint | 0.517 ± 0.005 | 0.479 ± 0.023 | **0.628 ± 0.024** |
| FakeHealth artifact-masked | 0.641 ± 0.030 | **0.658 ± 0.018** | 0.651 ± 0.061 |

The architecture ranking is conditional, not universal. BiomedBERT is strongest
on CoAID standard, CoAID temporal, and FakeHealth domain-disjoint tests, but it
is weakest on FakeHealth publisher-disjoint evaluation. ModernBERT is best on
FakeHealth standard and controlled point means but does not fix domain shift.
DistilBERT is competitive on several ordinary splits at lower compute cost.

Transformers materially exceed the sparse models on CoAID point estimates,
especially BiomedBERT under temporal control. On FakeHealth standard,
ModernBERT's 0.674 mean is essentially level with Logistic Regression's 0.677;
neural complexity does not supply a clear aggregate gain. Minority-class F1
and PR-AUC remain lower than the headline CoAID accuracy might suggest, and the
FakeHealth 512-token truncation limitation applies to every encoder here.

### Artifact-masking equivalence

The paired analysis defines the difference as standard minus masked macro F1.
Small means indicate similar aggregate point performance, but the 95% grouped
bootstrap interval must lie wholly inside ±0.03 to support the predeclared
practical-equivalence claim.

| Dataset/model | Mean observed difference | Run intervals inside margin | All-run equivalence conclusion |
|---|---:|---:|---|
| CoAID Logistic Regression | -0.011 | 0/1 | Not established |
| CoAID Linear SVM | -0.007 | 1/1 | Supported |
| CoAID DistilBERT | 0.003 | 2/3 | Not established |
| CoAID ModernBERT | 0.003 | 0/3 | Not established |
| CoAID BiomedBERT | 0.002 | 2/3 | Not established |
| FakeHealth Logistic Regression | 0.021 | 0/1 | Not established |
| FakeHealth Linear SVM | 0.019 | 0/1 | Not established |
| FakeHealth DistilBERT | 0.010 | 1/3 | Not established |
| FakeHealth ModernBERT | 0.017 | 0/3 | Not established |
| FakeHealth BiomedBERT | 0.007 | 0/3 | Not established |

The distinction matters: CoAID transformer point differences are tiny, but
their intervals are generally too wide for equivalence. FakeHealth BiomedBERT
is especially unstable: individual standard-minus-masked differences range
from -0.038 (masking helps) to +0.061 (masking hurts). The study therefore
cannot claim that named-artifact removal has no meaningful transformer effect.

### Calibration and deferral

CoAID standard Logistic Regression selects Platt calibration and obtains test
Brier score 0.041 and ECE 0.068. At the validation-selected 80% target, actual
test coverage is 84.5% and selective risk falls from 3.9% to 0.5%. That result
is not a balanced automation policy: unreliable-side coverage is zero, meaning
the retained automated cases are almost entirely on the reliable side.

FakeHealth standard Logistic Regression selects its uncalibrated probabilities
over Platt on the held-out validation selection half. Its test Brier score is
0.202 and ECE 0.135. At the 80% target, actual coverage is 81.6% and risk falls
from 30.0% to 24.5%, with reliable- and unreliable-side coverage of 84.4% and
75.5%. Deferral helps, but the retained risk remains too high for autonomous
medical-information adjudication.

### Cross-dataset transfer

Hosting domains seen in the source train/validation cohort are removed from
the target test cohort before transfer scoring.

| Direction/representation/model | Target records | Transfer macro F1 | Same-record target-trained macro F1 | Absolute degradation |
|---|---:|---:|---:|---:|
| CoAID→FakeHealth full-text Logistic | 263 | 0.421 | 0.686 | 0.264 |
| CoAID→FakeHealth full-text SVM | 263 | 0.424 | 0.681 | 0.257 |
| CoAID→FakeHealth title Logistic | 263 | 0.553 | 0.558 | 0.005 |
| FakeHealth→CoAID full-text Logistic | 161 | 0.309 | 0.874 | 0.565 |
| FakeHealth→CoAID full-text SVM | 161 | 0.309 | 0.822 | 0.513 |
| FakeHealth→CoAID title Logistic | 161 | 0.539 | 0.836 | 0.297 |

Full-text transfer fails badly in both directions. CoAID-to-FakeHealth title
transfer degrades little only because the target-trained title baseline is
itself modest. The vocabulary, length, source, category, and topic-proxy
diagnostics are stored with each direction. These results reject a claim that
the within-dataset decision rule is portable without adaptation; they cannot
separate model failure from topic, text-unit, and label-construct shift by
themselves.

### Engagement ablation

| Split | Text only | Engagement only | Text + engagement |
|---|---:|---:|---:|
| FakeHealth standard | 0.667 | 0.498 | 0.677 |
| FakeHealth publisher-disjoint | 0.687 | 0.569 | 0.700 |

On the matched complete-case cohort, combined features improve point macro F1
over text by only about 0.010 standard and 0.013 controlled. The paired
bootstrap intervals for text minus combined include zero in both settings
(-0.056 to 0.032 standard; -0.075 to 0.012 controlled). Engagement-only is
clearly worse than combined in both splits. The available counts therefore do
not establish a reliable incremental benefit beyond text.

### Narrative and qualitative status

Both datasets meet the 200-unreliable-document fit gate. CoAID's 233 unreliable
articles yield no candidate topic meeting the 30-document minimum. FakeHealth's
720 unreliable items yield four size-eligible machine topics containing 173,
138, 83, and 65 records. Candidate keywords concern cancer treatment,
brain/Alzheimer/diet, women's symptoms/depression/menopause, and heart/pain/
surgery. These phrases are machine summaries, not validated narratives.

The narrative finalizer records `awaiting_human_validation`, enumerates every
missing review field, and confirms that it created no human labels. The user
must review at least five representatives per FakeHealth topic and complete the
four topic decisions. Error-roster generation, label/prediction-blinded Stage
A, explanation generation, Stage B, and qualitative finalization remain
intentionally unrun so the declared ordering and blinding are preserved.

## Validity and functional value

The methods are valid for the study's stated purpose: auditing discrimination,
shortcut sensitivity, transfer, calibration, abstention, and review support
under dataset-defined labels. Their strongest features are frozen manifests,
train/validation/test separation, validation-only tuning, family/group leakage
controls, paired and grouped uncertainty, explicit label provenance, multiple
stress tests, and staged blinded review.

They are not sufficient to validate an autonomous medical fact-checker. The
main limits are construct mismatch between datasets, source-selected labels,
CoAID's small unreliable class, low effective group counts in some disjoint
splits, substantial FakeHealth truncation, incomplete/observational engagement
metadata, only three neural seeds, and one human reviewer. Going forward, the
pipeline is functionally useful as a reproducible model-audit and human-triage
research framework. Deployment claims would require independent contemporary
data, medical-domain annotation, external multi-reviewer validation, monitoring
for drift, and a defined escalation policy.

## Interpretation rules

- Macro F1 gives both classes equal weight; unreliable-class precision/recall/F1
  must still be shown because macro averages can hide the error type.
- PR-AUC is especially important when unreliable examples are rare.
- A confidence interval describes uncertainty from the declared grouped
  resampling scheme; it is not proof of external generalization.
- The 0.03 artifact-equivalence margin is a predeclared practical threshold. A
  paired interval entirely inside `[-0.03, 0.03]` supports practical
  equivalence for that comparison, not proof that no artifact remains.
- No binary “statistically significant” label is reported without a declared
  hypothesis test. If primary-comparison p-values are later introduced, the
  frozen configuration requires Holm correction.
- Negative results are substantive: they identify where the model or dataset
  does not support a portability or automation claim.

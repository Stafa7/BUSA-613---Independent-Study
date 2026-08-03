# Independent Study Project Context File  
**Project working title:** Detecting and Explaining Health Misinformation Narratives Using NLP and Social Engagement Analytics  
**Student:** Mustafa Yousuf  
**Course:** BUSA 613 Independent Study, MMA Program  
**Credit option:** 3 credits  
**Project type:** Research-focused / applied analytics project  
**Core methods:** NLP, social media analytics, misinformation detection, topic modeling, explainability, engagement analytics  
**Primary domain:** Health misinformation  
**Secondary management relevance:** Public-health communication, digital risk monitoring, platform governance, healthcare reputation management  

---

# 1. Project Overview

This independent study will examine how **natural language processing (NLP)** and **social media analytics** can be used to detect, explain, and monitor health misinformation narratives online.

The core idea is to move beyond a simple **fake-versus-real classification task**. The project should instead investigate:

1. How accurately different NLP models can detect health misinformation.
2. Whether social engagement signals improve misinformation detection.
3. What misinformation narratives appear in unreliable health content.
4. How misinformation narratives differ from reliable health content.
5. How transferable models are across datasets.
6. What explanations and model failure patterns reveal.
7. What practical recommendations can be made for public-health communication and digital risk monitoring.

The intended final output is both:

- a **research report**, and  
- a **reproducible analytical pipeline** that performs classification, narrative analysis, engagement analysis, and explainability.

The project is meant to be academically rigorous, methodologically interesting, and practically useful.

---

# 2. Refined Project Title Options

## Main title

**Detecting and Explaining Health Misinformation Narratives Using NLP and Social Engagement Analytics**

## Alternative title options

1. **From Detection to Explanation: NLP-Based Analysis of Health Misinformation Narratives Online**
2. **Interpretable NLP for Health Misinformation Detection and Narrative Discovery**
3. **Health Misinformation Analytics: Classification, Engagement, and Explainability in Online Health Content**
4. **Detecting, Interpreting, and Monitoring Health Misinformation Narratives Using NLP**
5. **A Social Media Analytics Framework for Detecting and Explaining Health Misinformation**

The current best title is:

> **Detecting and Explaining Health Misinformation Narratives Using NLP and Social Engagement Analytics**

This title is strong because it includes:
- the technical method: NLP,
- the data context: social engagement/social media analytics,
- the domain: health misinformation,
- and the contribution: detection plus explanation.

---

# 3. Short Form Description

This independent study will examine how natural language processing and social media analytics can be used to detect, explain, and monitor health misinformation narratives online. Using publicly available datasets such as CoAID and FakeHealth, the project will compare traditional machine learning models, transformer-based language models, and hybrid models incorporating social engagement signals. Beyond binary misinformation classification, the study will use topic modeling, sentiment/emotion analysis, and explainability techniques to identify recurring misinformation narratives, evaluate model failure patterns, and assess whether engagement features improve detection. The final output will be a reproducible analytical pipeline and research report with practical implications for public-health communication, platform governance, and digital risk monitoring.

---

# 4. Longer Proposal Description

This independent study will investigate how NLP and social media analytics can be used to detect, explain, and monitor health misinformation narratives online. Health misinformation is an important digital risk for public-health agencies, healthcare organizations, pharmaceutical firms, social media platforms, and the public. Misleading health claims can spread rapidly through online platforms, often amplified by emotionally charged language, distrust narratives, personal anecdotes, and selective or distorted references to scientific evidence.

Existing misinformation detection approaches often focus on binary classification: whether a piece of content is fake or real. However, this is limited from a management and practical decision-making perspective. Decision-makers need to understand not only whether content is unreliable, but also what kind of misinformation narrative is spreading, how audiences engage with it, and why a model flags it as problematic.

This project will therefore develop and evaluate an interpretable NLP pipeline for health misinformation analytics. The pipeline will use publicly available health misinformation datasets, primarily CoAID and FakeHealth. The methodology will compare traditional machine learning models, transformer-based models, topic modeling, sentiment or emotion analysis, social engagement features, and explainability techniques.

The final report will evaluate the effectiveness and limitations of different NLP approaches and present useful findings about misinformation themes, engagement patterns, model limitations, and practical implications for public-health communication and digital risk monitoring.

---

# 5. Course Alignment

The independent study syllabus expects the project to:

1. Formulate a problem that advances current knowledge in a management discipline or provides a solution that meets industry needs.
2. Conduct appropriate research such as literature review or industry analysis.
3. Propose and deploy quantitative or qualitative methods of analyzing data, including text mining.
4. Discuss current solutions to the problem.
5. Support why the proposed solution represents an improvement.
6. Communicate the problem formulation, background, methods, findings, and implications in a final report.
7. Produce at least three deliverables over the semester:
   - Week 4: Proposal
   - Week 8: Literature review/industry analysis and preliminary findings
   - Week 13: Final report

This project fits the course because it is:
- a research-focused applied analytics project,
- centered on text mining and NLP,
- supported by real datasets,
- tied to public-health and digital risk management needs,
- able to produce a practical analytical solution,
- and suitable for a final report with clear findings and implications.

---

# 6. Management and Industry Relevance

Although the project is technical, it should be framed as a **management analytics** and **digital risk monitoring** problem.

The key industry/management problem is:

> Organizations need scalable and interpretable ways to detect, understand, and respond to health misinformation circulating online.

Relevant stakeholder groups include:

- public-health agencies,
- healthcare organizations,
- pharmaceutical companies,
- social media platforms,
- trust and safety teams,
- communications teams,
- policy analysts,
- digital risk and reputation management teams.

The project should be positioned as a decision-support framework rather than merely a classifier.

A simple classifier only says:
> “This content is fake or real.”

The proposed framework should aim to answer:
> “Is this content likely unreliable? What narrative does it belong to? What language patterns drive the classification? How much engagement does it receive? How could decision-makers respond?”

This makes the project much more valuable and aligned with the course.

---

# 7. Main Research Problem

## Core problem statement

Health misinformation detection systems often focus on binary classification and model performance, but decision-makers require more interpretable and actionable insights. This project asks whether NLP and social media analytics can be combined to detect misinformation, identify recurring health misinformation narratives, analyze engagement patterns, and explain model predictions in a way that supports public-health communication and digital risk monitoring.

## Short version

> How can NLP and social media analytics be used to detect, explain, and interpret health misinformation narratives in a way that is useful for public-health communication and digital risk monitoring?

---

# 8. Research Questions

The project should be guided by 4–5 main research questions.

## Primary research questions

1. **How accurately can traditional machine learning and transformer-based NLP models detect health misinformation?**

2. **Do social engagement features improve misinformation detection compared with text-only models?**

3. **What recurring narratives, topics, or linguistic patterns characterize unreliable health information?**

4. **How transferable are misinformation detection models across datasets such as CoAID and FakeHealth?**

5. **What model explanations and error patterns are most useful for public-health or platform decision-makers?**

## Optional additional research questions

6. **Do unreliable health articles use more emotional or sensational language than reliable health articles?**

7. **Are certain misinformation topics associated with higher engagement?**

8. **Do models rely on meaningful content signals, or do they overfit to dataset/source artifacts?**

9. **Can topic modeling help identify actionable misinformation themes beyond binary labels?**

10. **Which cases are most ambiguous or difficult for automated misinformation detection?**

---

# 9. Project Objectives

## Objective 1: Evaluate misinformation detection performance

Compare traditional machine learning and transformer-based NLP models for detecting health misinformation.

## Objective 2: Test whether social engagement signals add predictive value

Determine whether social engagement metadata improves misinformation classification beyond text-only models.

## Objective 3: Identify dominant health misinformation narratives

Use topic modeling and narrative clustering to uncover recurring misinformation themes.

## Objective 4: Assess interpretability and model limitations

Use explainability methods and qualitative error analysis to understand what linguistic or semantic patterns drive model predictions.

## Objective 5: Translate findings into practical recommendations

Develop recommendations for public-health communication, platform governance, digital risk monitoring, and healthcare reputation management.

---

# 10. Proposed Solution

The proposed solution is a **multi-stage NLP and social media analytics pipeline**.

The pipeline should:

1. Ingest publicly available health misinformation datasets.
2. Clean and standardize text and labels.
3. Profile datasets and identify data quality issues.
4. Train traditional ML baseline models.
5. Train or fine-tune transformer-based classifiers.
6. Evaluate whether social engagement signals improve performance.
7. Apply topic modeling to identify recurring narratives.
8. Apply sentiment/emotion analysis to characterize rhetorical patterns.
9. Use explainability tools to interpret model decisions.
10. Conduct qualitative error analysis.
11. Summarize findings into practical implications and recommendations.

The proposed improvement over a simple fake-news classifier is the integration of:

- classification,
- social engagement analysis,
- narrative discovery,
- explainability,
- and management implications.

---

# 11. Dataset Plan

## Core datasets

### 1. CoAID

**Full name:** COVID-19 Healthcare Misinformation Dataset  
**Role:** Primary dataset  
**Use:** Main classification, engagement analysis, topic modeling, and explainability experiments.

CoAID is suitable because it includes:
- healthcare misinformation content,
- fake/real labels,
- social platform posts,
- user engagement information,
- COVID-19/health-related news items.

Possible use cases:
- text-only misinformation classification,
- text plus engagement modeling,
- topic modeling of fake and real content,
- analysis of engagement patterns by misinformation category.

Strength:
- strong fit for the project because it supports both NLP and social engagement analysis.

Potential limitation:
- COVID-specific, so generalizability should be discussed carefully.

---

### 2. FakeHealth

**Role:** Secondary validation dataset  
**Use:** Robustness testing, external validation, narrative comparison, possible cross-dataset evaluation.

FakeHealth is suitable because it was created specifically for fake health news detection and includes:
- health news content,
- labels,
- reviews/explanations,
- social engagement,
- user network information.

Possible use cases:
- train on CoAID and test/fine-tune on FakeHealth,
- compare misinformation themes across datasets,
- evaluate whether model performance transfers,
- analyze explainability using a dataset designed with health fake-news detection in mind.

Strength:
- supports health-specific fake-news detection.
- peer-reviewed dataset paper exists.
- strong complement to CoAID.

Potential limitation:
- setup may require careful data preparation.
- dataset structure may differ from CoAID, making harmonization necessary.

---

## Optional dataset

### 3. KInIT Medical Misinformation Dataset

**Role:** Optional extension only  
**Use:** Large-scale source credibility/narrative analysis if time permits.

Possible use cases:
- large-scale topic modeling,
- source-level credibility modeling,
- temporal analysis of medical misinformation trends.

Strength:
- much larger than CoAID/FakeHealth.
- can support richer narrative discovery.

Potential limitation:
- may be too large/complex for core project scope.
- labels may be source-level rather than article-level.
- should only be used as a stretch goal.

---

# 12. Dataset Strategy

## Core feasible strategy

Use:

1. **CoAID** as the main dataset.
2. **FakeHealth** as a secondary dataset for robustness and validation.

This is the recommended scope.

## Stretch strategy

Add KInIT only if the core analysis is completed early.

## Important scope decision

The project should **not** depend on live scraping or current social media API access. It should use stable, publicly available datasets first.

This reduces risk and makes the proposal more credible.

---

# 13. Methodology Overview

The methodology should proceed in phases:

1. Literature review and industry analysis.
2. Data acquisition and profiling.
3. Text cleaning and preprocessing.
4. Baseline classification models.
5. Transformer-based classification.
6. Social engagement feature modeling.
7. Topic modeling and narrative discovery.
8. Sentiment/emotion analysis.
9. Explainability and error analysis.
10. Practical recommendations and final reporting.

---

# 14. Phase 1: Literature Review and Industry Analysis

## Literature review themes

The literature review should cover:

1. Health misinformation prevalence and spread.
2. Public-health impact of misinformation.
3. Health misinformation datasets.
4. Automated misinformation detection.
5. Transformer models for misinformation and health NLP.
6. Social media analytics and engagement patterns.
7. Topic modeling for narrative discovery.
8. Explainable AI for misinformation detection.
9. Limitations, ethics, and risks of automated misinformation detection.

## Industry analysis themes

The industry analysis should cover:

1. Why public-health agencies need misinformation monitoring.
2. Why platforms need scalable detection and content-risk triage.
3. Why healthcare and pharmaceutical firms care about misinformation and public trust.
4. Why binary fake/real classification is insufficient for decision-making.
5. How explainable analytics could support communication response.

---

# 15. Phase 2: Data Acquisition and Profiling

## Tasks

- Download CoAID and FakeHealth.
- Inspect available files and fields.
- Standardize label definitions.
- Identify text fields.
- Identify engagement metadata.
- Identify timestamps if available.
- Remove duplicates.
- Check missing values.
- Analyze class balance.
- Analyze text length.
- Analyze vocabulary.
- Detect potential data leakage, such as source names or labels embedded in fields.
- Create train/validation/test splits.

## Outputs

- dataset dictionary,
- dataset profiling table,
- missingness report,
- class distribution chart,
- text-length distribution,
- engagement feature summary,
- data limitations section.

## Final dataset profile

| Dataset | Number of documents | Text fields | Labels | Engagement features | Time fields | Notes |
|---|---:|---|---|---|---|---|
| CoAID | 3,055 primary articles | Title and article body | Native credibility labels mapped to reliable/unreliable | Incomplete and label-dependent | Publication date | COVID-specific, source-structured dataset |
| FakeHealth | 2,137 stories and news releases | Title and article body | HealthNewsReview ratings mapped to reliable/unreliable | Available for matched complete-case analysis | Publication and review metadata | Broader health reporting with expert-review criteria |

---

# 16. Phase 3: Baseline Classification Models

Baseline models are important because they help determine whether complex methods actually improve performance.

## Baseline model candidates

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. TF-IDF + Naive Bayes
4. TF-IDF + Random Forest
5. TF-IDF/text embeddings + XGBoost

## Why baselines matter

The project should not assume transformer models are automatically better. A strong report will show:

- how well simple models perform,
- where simple models fail,
- whether transformers justify their extra complexity,
- and whether engagement signals improve performance.

## Evaluation metrics

Detailed evaluation guidance now lives in [[independent_study_evaluation_metrics]].

Path: `docs/independent_study_evaluation_metrics.md`

---

# 17. Phase 4: Transformer-Based Classification

## Candidate transformer models

1. DistilBERT  
   - efficient, feasible, good first transformer model.

2. BERT-base  
   - standard benchmark.

3. RoBERTa-base  
   - often strong for text classification.

4. BioBERT / PubMedBERT / SciBERT  
   - optional, if health/scientific language makes domain-specific pretraining useful.

## Recommended sequence

Start with:

1. TF-IDF + Logistic Regression
2. TF-IDF + SVM
3. DistilBERT
4. RoBERTa or BERT

Then add domain-specific models only if time permits.

## Transformer experiments

Possible experiments:

1. Fine-tune on CoAID.
2. Evaluate on CoAID test set.
3. Evaluate generalization to FakeHealth.
4. Fine-tune on combined CoAID + FakeHealth.
5. Test limited-data performance.
6. Compare text-only versus text-plus-engagement setups.

## Important feasibility note

For a 3-credit independent study, it is not necessary to fine-tune many large models. One or two transformer models plus strong baselines are enough if the analysis is deep.

---

# 18. Phase 5: Social Engagement Feature Analysis

This is one of the key features that makes the project more interesting than standard text classification.

## Potential engagement features

Depending on dataset availability:

- shares,
- likes,
- comments,
- replies,
- retweets,
- quote tweets,
- user engagement counts,
- source engagement,
- network features,
- review counts,
- engagement velocity if timestamps are available.

## Model comparison structure

| Model type | Purpose |
|---|---|
| Text-only model | Measures linguistic/content signal |
| Engagement-only model | Measures social/diffusion signal |
| Text + engagement model | Tests whether hybrid features improve performance |
| Transformer embedding + engagement model | More advanced hybrid approach |

## Possible modeling approaches

1. Logistic Regression with TF-IDF text features.
2. Logistic Regression or XGBoost with engagement-only features.
3. XGBoost with TF-IDF/SBERT embeddings plus engagement features.
4. Neural/transformer text embeddings concatenated with structured metadata.

## Expected analysis

The report should evaluate:

- whether engagement features improve F1-score,
- whether engagement helps recall misinformation,
- whether engagement creates risks of popularity bias,
- whether fake and real content have different engagement patterns,
- whether some misinformation topics receive more engagement than others.

## Important caution

If engagement features are missing or inconsistent, the project can still be successful using text-only models and topic modeling. Engagement analysis should be core if available but not a single point of failure.

---

# 19. Phase 6: Topic Modeling and Narrative Discovery

This is crucial for maximizing insights.

## Purpose

Topic modeling will move the project beyond “Which model performed best?” toward:

- what misinformation themes exist,
- which themes are most common,
- which themes are most engaging,
- how misinformation differs from reliable content,
- and what public-health communication gaps may exist.

## Methods

1. BERTopic  
   - main narrative discovery method.

2. LDA  
   - optional traditional topic modeling baseline.

3. Sentence embeddings + clustering  
   - possible alternative or supporting method.

4. Manual topic labeling  
   - review representative documents and assign interpretable labels.

## Possible misinformation narrative categories

Potential categories may include:

- vaccine safety concerns,
- false treatment claims,
- miracle cure narratives,
- natural remedy narratives,
- conspiracy/institutional distrust narratives,
- distorted scientific uncertainty,
- exaggerated prevention claims,
- anecdotal health claims,
- anti-pharmaceutical narratives,
- public-health policy resistance,
- fear-based outbreak narratives.

## Topic modeling outputs

The final report should include:

- topic labels,
- representative documents,
- top keywords,
- topic prevalence by label,
- topic prevalence by dataset,
- topic engagement patterns,
- possibly topic evolution over time if timestamps allow.

## Strong insight opportunities

Possible findings:

- misinformation may cluster around distrust and treatment claims,
- reliable content may cluster around official guidance and epidemiological updates,
- high-engagement misinformation may be more emotional or personal,
- some ambiguous topics may confuse models,
- certain narratives may be more likely to produce false positives/false negatives.

---

# 20. Phase 7: Sentiment and Emotion Analysis

Sentiment/emotion analysis is optional but valuable.

## Purpose

To understand whether misinformation content differs in emotional tone from reliable content.

## Possible features

- polarity,
- subjectivity,
- anger,
- fear,
- disgust,
- sadness,
- trust,
- uncertainty,
- moral/emotional language,
- sensationalism indicators.

## Methods

Possible tools/models:

- VADER for social-media-style sentiment,
- TextBlob as a simple baseline,
- transformer-based emotion classifier,
- GoEmotions-style model if appropriate.

## Research questions supported

- Does misinformation use more negative or emotionally charged language?
- Are fear or anger more common in misinformation?
- Are emotional posts more likely to receive engagement?
- Do emotional features improve classification?

## Caution

Sentiment analysis should be used as a supporting analysis, not the central contribution.

---

# 21. Phase 8: Explainability and Error Analysis

Explainability is one of the most important components for an A+ project.

## Purpose

The goal is to understand what the models are learning and where they fail.

## Methods

1. SHAP  
   - useful for Logistic Regression, XGBoost, and possibly text features.

2. LIME  
   - useful for local explanations on individual examples.

3. Feature importance  
   - for linear and tree-based models.

4. Token attribution  
   - optional for transformers.

5. Qualitative error analysis  
   - manually review false positives and false negatives.

## Error taxonomy

Possible error categories:

| Error type | Description |
|---|---|
| Anecdotal language | Personal experience mistaken for misinformation |
| Scientific-sounding misinformation | Misleading content using technical terms |
| Sensational reliable content | Reliable article written dramatically |
| Ambiguous questions | Content asking questions rather than making claims |
| Dataset/source leakage | Model relies on source rather than content |
| Topic overfitting | Model associates a topic with misinformation too strongly |
| Sarcasm/satire | Tone difficult for model to interpret |
| Mixed content | Article contains both factual and misleading claims |
| Label ambiguity | Dataset label may not capture nuance |

## Explainability outputs

The final report should include:

- examples of correctly classified misinformation,
- examples of false positives,
- examples of false negatives,
- SHAP/LIME explanation visuals,
- discussion of whether explanations are meaningful,
- implications for human-in-the-loop systems.

## Key practical point

For health misinformation, fully automated decisions can be risky. A better recommendation is likely:

> Use the model as a triage and monitoring tool, not as a final authority.

---

# 22. Cross-Dataset Generalization

This is a high-value research angle.

## Why it matters

A model that performs well on one dataset may fail on another because it learns dataset-specific artifacts.

## Possible experiments

1. Train on CoAID, test on CoAID.
2. Train on FakeHealth, test on FakeHealth.
3. Train on CoAID, test on FakeHealth.
4. Train on FakeHealth, test on CoAID.
5. Train on combined data, test separately.
6. Fine-tune on one dataset and evaluate transfer.

## Expected result

Cross-dataset performance may drop.

This is not a failure. It can become one of the most important findings:

> Health misinformation detection models may be dataset-sensitive, which limits real-world deployment unless models are evaluated across domains and sources.

## Possible interpretation

If cross-dataset performance drops, discuss:

- differences in topic distribution,
- differences in source types,
- COVID-specific versus general health misinformation,
- differences in writing style,
- label definition mismatch,
- dataset collection artifacts.

---

# 23. Expected Findings

The project may find:

## Model findings

- Transformer models likely outperform traditional baselines on context-dependent cases.
- TF-IDF + Logistic Regression/SVM may still perform surprisingly well.
- RoBERTa or BERT may perform best, but at higher computational cost.
- Domain-specific models may help if language is medically technical.
- Cross-dataset performance may be weaker than within-dataset performance.
- Hybrid text-plus-engagement models may improve performance if engagement data is reliable.

## Narrative findings

- Misinformation may cluster around recurring themes such as distrust, false cures, vaccine concerns, prevention myths, or distorted scientific uncertainty.
- Reliable content may focus more on official guidance, public-health recommendations, or evidence-based updates.
- Some misinformation narratives may be more emotionally charged.
- Some topics may be more likely to generate high engagement.

## Practical findings

- Binary fake/real classification is insufficient for organizational use.
- Narrative labels and explanations make outputs more actionable.
- Engagement metrics may help prioritize which narratives require attention.
- Models should be used with human review because false positives and false negatives have real consequences.
- Dataset generalizability is a major challenge.

---

# 24. Contribution and Originality

The project’s originality comes from combining multiple elements:

1. Health misinformation detection.
2. Social engagement analytics.
3. Topic/narrative discovery.
4. Explainability.
5. Cross-dataset robustness testing.
6. Practical recommendations for decision-makers.

The key contribution is not simply training a classifier.

The contribution should be framed as:

> An interpretable NLP and social media analytics framework that helps decision-makers understand not only whether health content is unreliable, but also what narrative it belongs to, how audiences engage with it, why it is flagged, and where automated systems fail.

This framing makes the project stronger and more aligned with an independent study.

---

# 25. Practical Implications

The final report should discuss implications for several groups.

## Public-health agencies

They could use narrative detection to:
- identify emerging misinformation themes,
- prioritize communication responses,
- understand public concerns,
- target educational campaigns.

## Healthcare organizations

They could use the system to:
- monitor misinformation affecting patient behavior,
- identify reputational risks,
- understand recurring misconceptions,
- improve communication strategies.

## Pharmaceutical firms

They could use similar analytics to:
- monitor drug-related misinformation,
- understand public trust issues,
- identify high-risk narratives related to treatment, vaccines, or side effects.

## Social media platforms

They could use explainable models to:
- triage content for review,
- prioritize high-engagement misinformation,
- support trust and safety workflows,
- reduce reliance on opaque black-box systems.

## Researchers

The project can contribute by:
- comparing model types,
- testing cross-dataset robustness,
- identifying dataset limitations,
- showing the value of narrative-focused misinformation analysis.

---

# 26. Ethical Considerations

This project deals with health-related misinformation, so ethical framing matters.

## Key ethical concerns

1. **False positives**
   - Reliable content could be wrongly flagged as misinformation.
   - This could harm legitimate discussion or scientific debate.

2. **False negatives**
   - Harmful misinformation could be missed.
   - This could reduce the value of the monitoring system.

3. **Over-censorship**
   - Automated misinformation detection could be misused for excessive moderation.

4. **Ambiguous health claims**
   - Not all uncertain or anecdotal content is misinformation.

5. **Dataset bias**
   - Models may learn source-specific or collection-specific artifacts.

6. **Public trust**
   - Health communication is sensitive, and automated tools must be transparent.

7. **Privacy**
   - Use public, pre-existing datasets.
   - Avoid attempting to identify users.
   - Avoid collecting unnecessary personal data.

## Recommended ethical framing

The project should explicitly state:

> The proposed system is intended as a decision-support and monitoring tool, not as a replacement for expert review or medical judgment.

---

# 27. Scope Control

This project should be complex, but it must remain feasible.

## Core project requirements

The project must complete:

1. CoAID data preparation.
2. FakeHealth inspection or partial validation.
3. Dataset profiling.
4. Baseline ML classification.
5. At least one transformer model.
6. Topic modeling.
7. Explainability or error analysis.
8. Final interpretation and recommendations.

## Strong but feasible target

The ideal completed version includes:

1. CoAID + FakeHealth.
2. TF-IDF baselines.
3. DistilBERT or RoBERTa.
4. Text-only versus text-plus-engagement comparison.
5. BERTopic narrative discovery.
6. SHAP/LIME or equivalent explanations.
7. Cross-dataset robustness test.
8. Practical recommendations.

## Stretch goals

Only attempt these after the core is complete:

1. KInIT extension.
2. Multiple transformer models.
3. Domain-specific transformer models.
4. Full interactive dashboard.
5. Temporal topic evolution.
6. Advanced LLM-assisted topic labeling.
7. Hybrid neural model combining text embeddings and structured engagement features.

---

# 28. Suggested Technical Stack

## Programming language

- Python

## Environment

- Jupyter Notebook or VS Code notebooks
- Conda environment
- Optional: Google Colab if GPU is needed

## Core libraries

### Data handling
- pandas
- numpy
- pyarrow

### Visualization
- matplotlib
- seaborn
- plotly

### Machine learning
- scikit-learn
- xgboost

### NLP
- nltk
- spaCy
- transformers
- datasets
- sentence-transformers
- BERTopic

### Explainability
- shap
- lime

### Evaluation
- scikit-learn metrics
- imbalanced-learn if needed

### Dashboard, optional
- Streamlit
- Plotly Dash
- Power BI if preferred for final visualization

---

# 29. Reproducible Pipeline Structure

Possible folder structure:

```text
health-misinformation-nlp/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/
│   │   ├── coaid/
│   │   ├── fakehealth/
│   │   └── kinit_optional/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_transformer_models.ipynb
│   ├── 04_engagement_analysis.ipynb
│   ├── 05_topic_modeling.ipynb
│   ├── 06_explainability_error_analysis.ipynb
│   └── 07_final_results.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_coaid.py
│   │   ├── load_fakehealth.py
│   │   └── preprocessing.py
│   ├── features/
│   │   ├── text_features.py
│   │   └── engagement_features.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── transformer.py
│   │   └── evaluation.py
│   ├── topics/
│   │   └── bertopic_pipeline.py
│   └── explainability/
│       └── shap_lime.py
│
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── proposal/
│   ├── week8_preliminary_findings/
│   └── final_report/
│
└── outputs/
    ├── models/
    ├── predictions/
    ├── topic_models/
    └── explanations/
```

---

# 30. Deliverable Timeline

The syllabus requires at least three deliverables:

1. Week 4 proposal.
2. Week 8 literature review/industry analysis and preliminary findings.
3. Week 13 final report.

## Week 1

### Goal
Confirm scope with supervisor.

### Tasks
- Discuss topic and project feasibility.
- Confirm supervisor expectations.
- Confirm datasets.
- Confirm communication frequency.
- Finalize brief description for independent study form.

### Output
- Approved project scope.

---

## Weeks 2–3

### Goal
Prepare proposal and confirm data feasibility.

### Tasks
- Download/inspect CoAID.
- Download/inspect FakeHealth.
- Conduct preliminary literature search.
- Create initial dataset summary.
- Draft proposal.
- Define exact research questions and methodology.

### Output
- Proposal draft.

---

## Week 4 Deliverable: Proposal

### Contents
- Project title.
- Problem statement.
- Research motivation.
- Research questions.
- Dataset descriptions.
- Proposed methodology.
- Expected findings.
- Feasibility and risks.
- Timeline.
- Preliminary references.

---

## Weeks 5–7

### Goal
Literature review, data preparation, and baseline modeling.

### Tasks
- Complete literature review draft.
- Complete industry analysis.
- Clean and standardize main dataset.
- Conduct exploratory data analysis.
- Train baseline models.
- Begin topic modeling.
- Incorporate supervisor feedback.

### Outputs
- Literature review draft.
- Dataset profiling results.
- Baseline model results.
- Preliminary topic clusters.

---

## Week 8 Deliverable: Literature Review / Industry Analysis and Preliminary Findings

### Contents
- Literature review.
- Industry analysis.
- Dataset profiling.
- Data preprocessing methodology.
- Baseline results.
- Preliminary topic modeling.
- Early interpretation.
- Updated plan based on feedback.

---

## Weeks 9–12

### Goal
Advanced modeling, engagement analysis, explainability, and final synthesis.

### Tasks
- Fine-tune transformer model.
- Compare models.
- Test engagement features.
- Conduct cross-dataset evaluation if feasible.
- Finalize topic modeling.
- Run explainability analysis.
- Conduct qualitative error analysis.
- Draft final recommendations.
- Prepare final report.

### Outputs
- Final model results.
- Topic/narrative analysis.
- Explainability results.
- Error taxonomy.
- Practical recommendations.

---

## Week 13 Deliverable: Final Report

### Required sections
1. Introduction / problem description and proposed solution.
2. Background / literature review / industry analysis.
3. Data analysis / testing of proposed solution.
4. Findings.
5. Recommendations / implications for practice.
6. Discussion.
7. Conclusion.

---

# 31. Final Report Structure

## 1. Introduction

Include:
- health misinformation problem,
- public-health and organizational relevance,
- project motivation,
- research questions,
- overview of proposed solution.

## 2. Background and Literature Review

Include:
- prevalence/spread of health misinformation,
- NLP misinformation detection,
- datasets,
- transformer models,
- social engagement signals,
- topic modeling,
- explainability,
- current gaps.

## 3. Industry Analysis

Include:
- public-health communication needs,
- platform governance challenges,
- healthcare/pharma reputation risk,
- limitations of current binary detection tools,
- need for interpretable monitoring.

## 4. Data

Include:
- CoAID description,
- FakeHealth description,
- optional KInIT description,
- data cleaning,
- label standardization,
- engagement variables,
- limitations.

## 5. Methodology

Include:
- preprocessing,
- baseline models,
- transformer models,
- engagement feature modeling,
- topic modeling,
- sentiment/emotion analysis,
- explainability,
- evaluation metrics.

## 6. Results

Include:
- dataset profiling,
- baseline model results,
- transformer results,
- text-only vs hybrid model comparison,
- cross-dataset performance,
- topic modeling results,
- sentiment/emotion findings,
- explainability examples,
- error analysis.

## 7. Findings

Include answers to research questions:
- which models performed best,
- whether engagement helped,
- what narratives appeared,
- where models failed,
- what findings matter practically.

## 8. Recommendations and Implications

Include:
- recommendations for public-health agencies,
- recommendations for platforms,
- recommendations for healthcare organizations,
- human-in-the-loop monitoring,
- communication strategy implications.

## 9. Discussion

Include:
- limitations,
- dataset bias,
- model generalizability,
- ethical concerns,
- technical constraints,
- future work.

## 10. Conclusion

Include:
- project summary,
- final contribution,
- main findings,
- future extensions.

---

# 32. Literature Base

The project should be grounded in peer-reviewed and authoritative sources.

## Literature bucket 1: Health misinformation as a public-health problem

### Suarez-Lledo & Alvarez-Galvez, 2021
**Title:** Prevalence of Health Misinformation on Social Media: Systematic Review  
**Venue:** Journal of Medical Internet Research  
**Use in project:** Supports the claim that health misinformation is widespread across social platforms and health domains.

### Wang et al., 2019
**Title:** Systematic Literature Review on the Spread of Health-related Misinformation on Social Media  
**Venue:** Social Science & Medicine  
**Use in project:** Supports framing misinformation as a social-media and public-health communication problem.

### do Nascimento et al., 2022
**Title:** Infodemics and Health Misinformation: A Systematic Review of Reviews  
**Use in project:** Supports discussion of infodemics, trust, public behavior, and public-health consequences.

---

## Literature bucket 2: Dataset papers

### Cui & Lee, 2020
**Title:** CoAID: COVID-19 Healthcare Misinformation Dataset  
**Use in project:** Dataset paper for the primary dataset.

### Dai, Sun & Wang, 2020
**Title:** Ginger Cannot Cure Cancer: Battling Fake Health News with a Comprehensive Data Repository  
**Venue:** International AAAI Conference on Web and Social Media  
**Use in project:** Dataset paper for FakeHealth. Also supports explainability emphasis.

---

## Literature bucket 3: Automated health misinformation detection

### Schlicht et al., 2023
**Title:** Automatic Detection of Health Misinformation: A Systematic Review  
**Use in project:** Supports methodology and discussion of existing automated detection approaches.

### Feng et al., 2025
**Title:** Health Misinformation Detection: Approaches, Challenges, and Future Directions  
**Use in project:** Useful for recent challenges, datasets, methods, and future directions.

### Kumari et al., 2021
**Title:** Debunking Health Fake News with Domain Specific Pre-trained Model  
**Use in project:** Supports comparing general transformer models with health/scientific domain models.

### Abdelminaam et al., 2021
**Title:** CoAID-DEEP: An Optimized Intelligent Framework for Automated Detecting COVID-19 Misleading Information on Twitter  
**Use in project:** Example of deep learning applied to CoAID/COVID misinformation detection.

---

## Literature bucket 4: Topic modeling and narrative discovery

### Grootendorst, 2022
**Title:** BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure  
**Use in project:** Main methodological source for BERTopic.

### Blei, Ng & Jordan, 2003
**Title:** Latent Dirichlet Allocation  
**Venue:** Journal of Machine Learning Research  
**Use in project:** Traditional topic modeling baseline.

---

## Literature bucket 5: Explainability

Potential sources to include:
- SHAP original paper by Lundberg and Lee.
- LIME original paper by Ribeiro, Singh, and Guestrin.
- FakeHealth paper for the importance of explainability in health fake-news detection.
- XAI literature related to text classification and content moderation.

---

# 33. Final One-Paragraph Project Summary

This independent study developed and evaluated an interpretable NLP and social media analytics pipeline for detecting and explaining health misinformation narratives online. Using CoAID and FakeHealth, the project compared traditional machine-learning models, transformer-based classifiers, and hybrid models incorporating social-engagement signals. Topic modeling, explanation methods, calibration, transfer testing, and qualitative error analysis identified recurring narratives, model failure patterns, and practical limits. The completed work extends beyond binary classification by showing what themes appear in the datasets, why models flag records, where results fail to generalize, and how the system can support human decision-making.

---

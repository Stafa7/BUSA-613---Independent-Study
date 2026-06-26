# Athira, Madhu Kumar, and Chacko 2023 - Explainable AI for Fake News Detection

## Citation

Athira, A. B., Madhu Kumar, S. D., & Chacko, A. M. (2023). *A systematic survey on explainable AI applied to fake news detection*. Engineering Applications of Artificial Intelligence, 122, 106087. https://doi.org/10.1016/j.engappai.2023.106087

## Source Files

- PDF: `docs/literature_pdfs/A systematic survey on explainable AI applied to fake news detection.pdf`
- Converted Markdown: `docs/literature_mds/a_systematic_survey_on_explainable_ai_applied_to_fake_news_detection.md`

## Purpose

This survey reviews explainable AI methods applied to fake news detection. The authors argue that many automatic fake-news detection systems achieve promising classification results but do not explain why a prediction was made. Explainability is important because it can reveal bias and discrimination in detection algorithms, support trust in classification decisions, and help developers identify model failure modes.

The paper is not health-specific, but its explainability framework is directly relevant to a health misinformation project because health misinformation detection is high-stakes and explanations need to be usable by decision-makers.

## Background and Scope

The paper discusses the difficulty of defining fake news and distinguishes broad definitions such as "fake news is false news" from narrower definitions involving intentional falsehood published by a news outlet. It reviews automatic fake news detection as a binary classification task and distinguishes single-modal from multimodal detection.

Single-modal features include:

- user features,
- news content features,
- social context features.

Multimodal fake-news detection often combines textual and visual features, or user-profile and news-content features. The authors note that combining auxiliary information can improve detection when a single feature type is insufficient.

## Explainability Framework

The survey analyzes explainable fake-news detection from four perspectives:

- features used for classification,
- type of explanation,
- type of explainee,
- metric used to evaluate explainability.

The authors find that most explainable fake-news systems focus on textual content and attention-based explanations. Some use news content plus user comments, social engagements, network patterns, propagation features, or knowledge graphs. The survey discusses models such as dEFEND, GCAN, MVAN, knowledge graph frameworks, Tsetlin machines, and propagation-based models.

## Key Observations

The authors identify disagreement in the literature over whether attention weights should be treated as explanations. They distinguish several explanation types, including visual explanation, mimic learning or simplification, perturbation-based explanations, ablation explanations, feature relevance explanations, attention-based explanations, and relational explanations.

They argue that different explanation types cover different aspects of explainability, so combining multiple explanation approaches may produce more robust explanations. They also note that interactive explanation systems are rare, even though interaction between explainer and explainee could improve system usefulness.

From the explainee perspective, the survey finds that most existing approaches prioritize data experts or AI experts. End users are not well served by current approaches. This is important for health misinformation because useful explanations may need to be understandable to public-health communicators, platform analysts, or nontechnical stakeholders.

From the evaluation perspective, the survey finds no common benchmark for comparing explainable fake-news models. Some studies use Amazon Mechanical Turk human evaluation, visualization, or case studies, but explanation evaluation remains unresolved.

## Research Directions

The paper recommends several future directions:

- explainable fake-news detection using visual features,
- models combining text and visual features with correlation analysis,
- ensemble approaches that combine models trained on different feature types,
- multi-class explainable fake-news detection rather than only binary classification,
- better consideration of explainee type,
- stronger evaluation metrics and benchmarks for explanation quality.

## Relevance to This Independent Study

This paper supports the explainability and error-analysis phase. It justifies using interpretability not only to explain predictions but also to detect model bias, dataset artifacts, and overreliance on superficial cues.

For this project, the most useful takeaway is that explanations should be designed around the stakeholder. A public-health communication team may need narrative-level and claim-level explanations, while a model developer needs feature-level and error-analysis explanations. The paper also supports treating misinformation as multi-class or narrative-based rather than purely binary.

# Baris-Schlicht, Fernandez, Chulvi, and Rosso 2023 - Automatic Detection of Health Misinformation

## Citation

Baris-Schlicht, I., Fernandez, E., Chulvi, B., & Rosso, P. (2023). *Automatic detection of health misinformation: a systematic review*. Journal of Ambient Intelligence and Humanized Computing, 15, 2009-2021. https://doi.org/10.1007/s12652-023-04619-4

## Source Files

- PDF: `docs/literature_pdfs/Automatic detection of health misinformation a systematic review.pdf`
- Converted Markdown: `docs/literature_mds/automatic_detection_of_health_misinformation_a_systematic_review.md`

## Purpose

This systematic review focuses specifically on computer science research for automatic health misinformation detection. The authors argue that health misinformation can produce public-health harms such as vaccine hesitancy, use of unproven treatments, hostility toward health workers, and hate toward ethnic groups. Because manual fact-checking is slow and requires health expertise, automated detection methods are needed.

The review is particularly relevant because it surveys text mining, machine learning, deep learning, datasets, and open challenges in the exact domain of this independent study.

## Definition and Scope

The paper uses two related definitions of health misinformation:

- health-related claims that contradict current scientific knowledge,
- exaggeration or misrepresentation of scientific findings that changes the original value of the research.

This is important because health misinformation is not always fabricated from nothing. A claim can be misleading by exaggerating preliminary findings, overstating causal relationships, or stripping scientific findings from context.

## Methodology

The authors follow systematic-review guidance from Kitchenham and Charters. Their search query combined health/medical terms, misinformation-related terms, and machine-learning/text-mining terms. The review includes computer science literature, including peer-reviewed work and influential preprints cited by peer-reviewed papers. The final review covers 43 computer science papers, 19 of which present publicly available datasets.

The authors introduce a taxonomy with six dimensions:

- input,
- source,
- topic,
- misinformation type,
- task,
- method.

They also compare COVID-19 datasets with other health-related misinformation datasets through content-based analysis.

## Main Findings

Around 53% of reviewed papers evaluated models for COVID-19 misinformation. Most datasets are English, and most health misinformation datasets focus on COVID-19. Few studies address other health topics such as noncommunicable diseases. Many datasets are small, which limits deep learning.

Methodologically, the review finds that recent studies are dominated by deep learning and transformers, but traditional machine learning remains important. Reported method categories include deep learning, traditional machine learning, and mixed approaches. Traditional models remain valuable because they are easier to implement and provide strong baselines. Transformers often perform well on short informal social media texts, but they have limitations: token-length constraints, computational cost, need for fine-tuning, and potential inherited bias from pretraining data.

The paper identifies models and methods used in the reviewed literature, including SVM, random forest, naive Bayes, logistic regression, CNNs, recurrent networks, BERT, RoBERTa, SciBERT, XLM-R, MARBERT, sentence transformers, and T5-style models. It also discusses models augmented with user/post interaction features.

The dataset evaluation compares COVID-19 datasets with other health-related datasets. The authors find that the corpora are separable, with the best classifier receiving AUC 0.883 when distinguishing COVID samples from other health misinformation samples. Linguistic, affective, and biomedical features differed across corpora. COVID samples were more unpredictable, while other health topics contained more varied biomedical terms.

## Open Challenges

The paper identifies several challenges directly relevant to this independent study:

- data scarcity outside COVID-19,
- lack of high-quality datasets for non-COVID health domains,
- limited generalization across health topics,
- cross-platform transfer problems,
- need for multilingual datasets and methods,
- difficulty of bridging scientific evidence and public explanations,
- need for early detection before harmful information spreads widely.

The authors emphasize that health misinformation labeling requires medical knowledge, which makes dataset creation expensive and time-consuming. They also point to few-shot and zero-shot learning as possible ways to address data scarcity.

## Relevance to This Independent Study

This should be a backbone source for the technical literature review. It supports using both traditional baselines and transformer models. It also supports explicit cross-dataset evaluation between CoAID and FakeHealth because COVID and non-COVID health misinformation datasets differ in language, topical scope, affective features, and biomedical vocabulary.

The paper also supports a cautious interpretation of high model performance: models may overfit to dataset/topic artifacts, especially in COVID-specific corpora. This aligns with the project's planned explainability and error-analysis components.

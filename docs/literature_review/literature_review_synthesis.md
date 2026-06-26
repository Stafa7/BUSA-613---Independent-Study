# Literature Review Synthesis - Health Misinformation Detection and Narrative Analytics

## Purpose

This synthesis organizes the reviewed literature around the independent study's central question: how NLP and social media analytics can detect, explain, and interpret health misinformation narratives in a way that supports public-health communication and digital risk monitoring.

The review suggests that the project should not be framed as a simple fake-versus-real classifier. The stronger framing is a decision-support pipeline that combines misinformation detection, narrative discovery, engagement analysis, transferability testing, and explainability.

## Source Base

The current literature base covers four types of sources:

- broad health misinformation reviews,
- computational health misinformation detection reviews,
- dataset papers,
- misinformation diffusion, narrative, and explainability studies.

The most important broad health misinformation sources are Suarez-Lledo and Alvarez-Galvez (2021) and Wang et al. (2019). Suarez-Lledo and Alvarez-Galvez establish prevalence across health topics and social platforms, while Wang et al. explain spread mechanisms through agents, messages, interpreters, and network dynamics.

The most important detection-method source is Baris-Schlicht et al. (2023), which directly reviews automatic health misinformation detection, datasets, methods, and open challenges. Di Sotto and Viviani (2022) provide a feature-oriented data science approach that is useful for designing the pipeline.

For datasets, Dai et al. (2020) is central because FakeHealth supports text, reviews, explanations, engagement, and user-network information. COCO is useful as a model for fine-grained narrative and stance annotation, even if it is not a core dataset. CoAID remains important as the primary project dataset, but the downloaded PDF set currently does not include the CoAID paper.

For social engagement and narrative analysis, Vosoughi et al. (2018) provides foundational evidence that false news diffuses differently from true news. Chen et al. (2022) shows how topic modeling, domain classification, and retweet-network analysis can identify COVID-19 misinformation narratives and echo chambers. Athira et al. (2023) supports the explainability component.

## Theme 1: Health Misinformation Is Topic-Specific and Platform-Specific

Suarez-Lledo and Alvarez-Galvez show that health misinformation is not evenly distributed. The reviewed literature clusters around vaccines, drugs/smoking, noncommunicable diseases, pandemics, eating disorders, and medical treatments. Prevalence differs sharply by topic and platform. Twitter is heavily represented, both because it is an important platform and because it is easier for researchers to access.

This matters for the project because a binary classifier can hide important variation. A false vaccine claim, a misleading cancer treatment article, and a COVID prevention conspiracy may all be misinformation, but they differ in language, risk, audience, and response strategy.

## Theme 2: Misinformation Spread Requires More Than Text Analysis

Wang et al. argue that misinformation spread involves an agent, a message, and an interpreter. This framework is useful because most NLP systems focus on the message only. Social engagement analytics helps approximate how interpreters and platform communities respond to content.

Vosoughi et al. show that false news spreads farther, faster, deeper, and more broadly than true news on Twitter. The study also finds that novelty and emotion matter, and that human sharing behavior, not only bots, explains the advantage of falsehood. Chen et al. extend this type of thinking into COVID-19 misinfodemics by showing that misinformation narratives are linked to dense retweet communities and political alignment.

For the independent study, this supports testing text-only models against engagement-informed models.

## Theme 3: Automated Detection Needs Strong Baselines and Careful Dataset Evaluation

Baris-Schlicht et al. emphasize that transformers dominate newer work, but traditional machine learning models remain important because they are easier to implement, easier to compare, and strong baselines. This supports the planned sequence: TF-IDF plus logistic regression/SVM, then transformer-based models.

The literature also shows that dataset differences are a major methodological risk. COVID-specific datasets differ from broader health datasets in vocabulary, affective language, biomedical terminology, and source structure. Therefore, a model trained on CoAID may not transfer well to FakeHealth. This makes cross-dataset evaluation one of the most academically interesting parts of the project.

## Theme 4: Health Misinformation Detection Requires Domain-Specific Features

Di Sotto and Viviani identify feature groups that are directly useful for the pipeline:

- textual representation features,
- linguistic/stylistic features,
- emotional features,
- medical/domain-specific features,
- propagation/network features,
- user-profile features.

FakeHealth also shows that health-specific review criteria can explain why a story is unreliable. A health article may be misleading because it fails to quantify harms, exaggerates benefits, ignores costs, relies on a press release, or uses unjustified sensational language. These are more specific than general fake-news cues.

## Theme 5: Narrative Discovery Is Necessary for Practical Decision Support

Chen et al. identify COVID-19 misinformation narratives around hydroxychloroquine and alternative medicines, US officials and governing agencies, and prevention measures. COCO shows that misinformation can be organized into fine-grained conspiracy categories and that stance matters: a tweet can mention a conspiracy without endorsing it.

This is central to the project. Public-health decision-makers do not only need to know whether a post is unreliable. They need to know what narrative it belongs to, how it is framed, and whether it is gaining engagement.

## Theme 6: Explainability Is a Model-Validation and Decision-Support Requirement

Athira et al. argue that fake-news detection models often lack explanations, which limits trust and makes bias difficult to identify. Their survey groups explainability by features, explanation type, explainee type, and evaluation metric. This is useful because the independent study has multiple explainability audiences:

- the researcher needs to know whether the model learned meaningful signals,
- public-health communicators need to understand narratives and rhetorical cues,
- platform or risk-monitoring teams need explanations that support triage.

Explainability should therefore be used both for model interpretation and error analysis. It should help identify false positives, false negatives, source leakage, topic leakage, and cases where a model confuses discussion of misinformation with endorsement of misinformation.

## Implications for the Project Design

The literature supports the current project design:

- Use CoAID as the primary dataset and FakeHealth as a validation or robustness dataset.
- Start with traditional ML baselines before transformers.
- Compare text-only, engagement-only, and text-plus-engagement models.
- Conduct cross-dataset evaluation to test transferability.
- Use topic modeling for narrative discovery, but validate topics qualitatively.
- Include sentiment/emotion analysis because affect and novelty are linked to misinformation spread.
- Use explainability and error analysis to test whether models rely on meaningful cues or artifacts.

## Key Literature Gap This Project Can Address

The strongest gap is the integration gap. Prior work often studies one piece at a time: prevalence, spread, detection, datasets, narratives, or explainability. This project can contribute by combining these pieces into a single applied analytics pipeline for health misinformation:

1. Detect unreliable health content.
2. Test whether engagement improves detection.
3. Identify misinformation narratives.
4. Compare reliable and unreliable content.
5. Evaluate transfer across datasets.
6. Explain model decisions and failure cases.
7. Translate results into management and public-health communication recommendations.

That integrated framing is stronger than presenting the project as another misinformation classifier.

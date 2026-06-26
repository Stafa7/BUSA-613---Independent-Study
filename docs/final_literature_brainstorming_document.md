# Final Literature Brainstorming Document - Health Misinformation NLP and Social Engagement Analytics

## Purpose

This document summarizes the current literature review as a brainstorming aid for re-evaluating the independent study design. It focuses on what the reviewed articles collectively say about datasets, data provenance, methodologies, findings, and successful research patterns.

The key takeaway is that this project should not be treated as only a binary fake-versus-real classification task. The stronger framing is an applied analytics pipeline for health misinformation: detect questionable health information, explain model decisions, identify recurring narratives, evaluate engagement behavior, and test whether models generalize across datasets and health topics.

## Article Set Reviewed

The reviewed literature covers four types of work:

- health misinformation prevalence and spread reviews,
- automatic health misinformation detection reviews,
- dataset and benchmark papers,
- diffusion, narrative, and explainability studies.

The main articles are:

- Suarez-Lledo and Alvarez-Galvez (2021), a systematic review of health misinformation prevalence on social media.
- Wang, McKee, Torbica, and Stuckler (2019), a systematic review of how health-related misinformation spreads on social media.
- Baris-Schlicht, Fernandez, Chulvi, and Rosso (2023), a systematic review of automatic health misinformation detection.
- Dai, Sun, and Wang (2020), the FakeHealth dataset paper.
- Di Sotto and Viviani (2022), a health misinformation data science paper using CoAID, ReCOVery, and FakeHealth.
- Chen, Jiang, Chang, Muric, and Ferrara (2022), a large-scale COVID-19 misinfodemic study on Twitter.
- Vosoughi, Roy, and Aral (2018), a foundational study of true and false news diffusion on Twitter.
- Langguth et al. (2023), the COCO COVID-19 conspiracy theory dataset paper.
- Athira, Madhu Kumar, and Chacko (2023), a systematic survey of explainable AI for fake news detection.

## Dataset Map

| Article | Dataset or evidence base | Where the data came from | Labels or classification source | Data types available | Usefulness for this project |
|---|---|---|---|---|---|
| Suarez-Lledo and Alvarez-Galvez (2021) | 69 empirical studies of health misinformation on social media | Academic databases: PubMed, MEDLINE, Scopus, and Web of Science; English articles before March 2019 | Labels varied by study; the review coded topic, platform, method, prevalence, and study quality | Review-level evidence across social media platforms and health topics | Best for background, motivation, topic categories, and prevalence patterns |
| Wang et al. (2019) | 57 empirical and methodological studies on health misinformation spread | PubMed, Cochrane, Web of Science, Scopus, and Google; articles from 2012 to 2018 | Labels and definitions varied by study; the review synthesized mechanisms of spread | Review-level evidence across disciplines, platforms, and health topics | Best for conceptual framing around agents, messages, interpreters, networks, and interventions |
| Baris-Schlicht et al. (2023) | 43 computer science papers, including 19 public datasets | Computer science literature using health, misinformation, and machine learning search criteria | Varied by dataset; includes manual labels, fact-check labels, source labels, and task-specific annotations | Review-level evidence on datasets, features, models, tasks, and health topics | Best for technical framing, model choices, dataset limitations, and cross-dataset risks |
| Dai et al. (2020) | FakeHealth: HealthStory and HealthRelease | HealthNewsReview.org expert reviews from 2005 to 2018; Twitter engagement data collected around reviewed items | HealthNewsReview ratings and criteria; ratings below 3 treated as fake in the dataset paper | Article text, metadata, expert review ratings, review explanations, tweets, replies, retweets, user profiles, timelines, followers, followings | Best secondary dataset because it includes text, social engagement, expert health criteria, and explanations |
| Di Sotto and Viviani (2022) | CoAID, ReCOVery, and FakeHealth | CoAID from COVID health news/claims and social media; ReCOVery from COVID news websites and tweets; FakeHealth from HealthNewsReview and Twitter | CoAID credible/not credible sources and claims; ReCOVery source reliability from NewsGuard and Media Bias/Fact Check; FakeHealth expert review ratings | Textual content, tweet IDs, retweets, replies where available, user IDs, textual, emotional, medical, propagation, and user-profile features | Best methodological guide for comparing text, emotion, medical, propagation, and user features across datasets |
| Chen et al. (2022) | Large-scale COVID-19 Twitter dataset and misinformation subset | Public COVID-19 Twitter dataset, release v2.45, covering Jan. 21, 2020 to Apr. 30, 2021; collected through Twitter streaming API using COVID keywords | Misinformation subset defined by URLs from Media Bias/Fact Check questionable or conspiracy-pseudoscience domains | Tweets, URLs, domains, retweet networks, inferred political leaning, topics, source-domain categories | Best model for narrative discovery, source-domain analysis, and networked misinfodemic analysis |
| Vosoughi et al. (2018) | 126,000 Twitter rumor cascades from 2006 to 2017 | Twitter cascades connected to fact-checked stories | Six fact-checking organizations: Snopes, PolitiFact, FactCheck.org, TruthOrFiction, Hoax-Slayer, and Urban Legends | Cascades, retweets, diffusion metrics, novelty measures, emotional replies, bot analysis | Best foundational evidence that false information spreads differently from true information |
| Langguth et al. (2023) | COCO: 3,495 annotated COVID-19 conspiracy tweets | Twitter COVID keyword collection from Jan. 17, 2020 to Jun. 30, 2021; candidate tweets filtered by 54 conspiracy-related keywords | Expert annotation with majority vote across 12 conspiracy categories and stance labels | Tweets, conspiracy category labels, stance labels: supporting/promoting, related but not supporting, unrelated | Best guide for narrative and stance design, especially avoiding confusion between mentioning and endorsing misinformation |
| Athira et al. (2023) | Survey of explainable AI approaches for fake news detection | Research literature on explainable fake news detection | Not a single dataset; synthesizes explanation methods and evaluation patterns | Features, explanation types, explainee types, evaluation methods | Best source for explainability design and stakeholder-aware explanations |

## Dataset Details and Implications

### FakeHealth

FakeHealth is the most directly useful dataset for a health misinformation project with social engagement analytics. It is built from HealthNewsReview.org, which reviewed health news stories and press releases from 2005 to 2018. The dataset has two subsets:

- HealthStory, based on health news stories.
- HealthRelease, based on health-related press releases.

FakeHealth includes article text, metadata, health review ratings, review explanations, social engagements, and user-network information. Dai et al. report collecting approximately 500,000 tweets, 29,000 replies, 14,000 retweets, and 27,000 user profiles with timelines and friend lists. This is valuable because it supports both content modeling and engagement modeling.

Its strongest feature is the expert review structure. HealthNewsReview evaluated stories using health-specific criteria such as whether the article quantified benefits, quantified harms, discussed costs, compared alternatives, explained evidence quality, established novelty, avoided disease-mongering, avoided unjustified sensational language, and disclosed conflicts of interest. These criteria give the dataset a clearer health-information-quality basis than generic fake-news labels.

For this project, FakeHealth can support:

- binary classification using health review ratings,
- feature experiments using article text and metadata,
- explainability based on expert review criteria,
- engagement comparison through tweets, replies, and retweets,
- discussion of why health misinformation is often misleading rather than entirely fabricated.

The main caution is source leakage. If a model learns that certain publishers are usually low-quality or high-quality, performance may improve without the model learning the content-level health reasoning we care about. Source features can be useful, but they should be interpreted as credibility signals rather than proof of content understanding.

### CoAID

The downloaded PDF set does not include the original CoAID paper, but CoAID is discussed in Di Sotto and Viviani (2022). In that article, CoAID is described as an English COVID-19 dataset containing health news and claims with social media data.

CoAID's credible news sources include Healthline, ScienceDaily, NIH, MedicalNewsToday, Mayo Clinic, Cleveland Clinic, WebMD, WHO, and CDC. Its not-credible content comes from fact-checking and false-myth sources. Its claims are also sourced from official or curated COVID-19 myth and Q&A content, including WHO and MedicalNewsToday material.

Di Sotto and Viviani report the original CoAID dataset as containing:

- 3,555 textual contents,
- 151,964 tweet IDs,
- 122,150 reply IDs.

After data gathering and cleaning, they report:

- 1,820 textual contents,
- 74,722 tweet IDs,
- 65,464 retweet IDs,
- 29,969 reply IDs,
- 164,891 user IDs.

For this project, CoAID remains valuable as a COVID-specific health misinformation dataset. It can support experiments involving article/claim text, tweet engagement, retweets, replies, and user-level information if the data can be hydrated or is already locally available.

The main caution is that COVID datasets may contain strong topic, time-period, and source artifacts. A model might learn "COVID-era content from certain sources" rather than general health misinformation patterns. That is why cross-dataset evaluation with FakeHealth is important.

### ReCOVery

ReCOVery is discussed in Di Sotto and Viviani (2022). It contains COVID-19 English news from about 60 news websites. Reliability labels are based on source trustworthiness, using NewsGuard and Media Bias/Fact Check thresholds. This makes it a source-reliability dataset rather than a direct expert judgment of each individual article.

Di Sotto and Viviani report the original ReCOVery dataset as containing:

- 2,029 textual contents,
- 140,820 tweet IDs.

After data gathering and cleaning, they report:

- 1,910 textual contents,
- 42,153 tweet IDs,
- 43,024 retweet IDs,
- 58,495 user IDs.

ReCOVery could be useful as an optional robustness dataset, but it is less central than CoAID and FakeHealth because its labels are source-based. It may be better for discussing source credibility and engagement patterns than for claim-level health misinformation classification.

### COCO

COCO is not a general health misinformation dataset. It is a COVID-19 conspiracy theory dataset from Twitter. The raw collection began with approximately two billion Twitter statuses collected between Jan. 17, 2020 and Jun. 30, 2021 using COVID-related keywords. After removing retweets, quotes, and replies, the authors had more than 400 million tweets. They then used 54 conspiracy-related keywords to find candidates, removed tweets with links, resolved user location, selected longer tweets, and produced a candidate pool of around 100,000 tweets.

The final dataset contains 3,495 tweets. Each tweet is labeled across 12 conspiracy categories, producing 41,940 category-level labels. Labels were assigned by three expert annotators using majority vote.

COCO's most important contribution is stance-aware labeling. It distinguishes:

- tweets that support or promote a conspiracy,
- tweets that are related but do not support it,
- tweets that are unrelated.

This is important for this project because a classifier or topic model can easily confuse topic mention with endorsement. A tweet or article debunking misinformation may use the same words as the misinformation itself. The project should treat narrative discovery as qualitative interpretation, not just clustering keywords.

### Chen et al. COVID-19 Twitter Misinfodemic Data

Chen et al. use a very large public COVID-19 Twitter dataset. The analysis uses release v2.45, covering Jan. 21, 2020 to Apr. 30, 2021, with 1,443,871,621 tweets. The dataset was collected in real time through Twitter's streaming API with manually curated COVID-related keywords.

Their misinformation subset contains 12,689,165 tweets that include URLs from domains classified by Media Bias/Fact Check as questionable or conspiracy-pseudoscience sources.

This data source is different from FakeHealth or CoAID because the label is not attached to a claim or article by direct health experts. Instead, misinformation exposure is approximated through domain quality. This makes it powerful for large-scale network analysis, but weaker for article-level truth assessment.

The paper is still very useful because it shows how to combine:

- topic modeling,
- domain reliability categories,
- political leaning inferred from shared domains,
- retweet networks,
- community structure,
- narrative interpretation.

For this project, Chen et al. provide a strong example of how to translate NLP topics into public-health-relevant misinformation narratives.

### Vosoughi et al. Twitter Rumor Cascades

Vosoughi et al. analyze around 126,000 rumor cascades on Twitter from 2006 to 2017. These stories were tweeted by about 3 million people more than 4.5 million times. Labels came from six fact-checking organizations: Snopes, PolitiFact, FactCheck.org, TruthOrFiction, Hoax-Slayer, and Urban Legends. The fact-checking organizations showed high agreement, around 95% to 98%.

The dataset is not health-specific. Its value is methodological and conceptual. It demonstrates that falsehood can be studied as a diffusion phenomenon, not just a text classification label.

Useful variables include:

- cascade depth,
- cascade size,
- maximum breadth,
- structural virality,
- speed of diffusion,
- novelty,
- emotional responses in replies,
- bot involvement.

For this project, this supports including social engagement and emotion features. Even if the local datasets do not allow full cascade reconstruction, the study justifies analyzing engagement counts, reply sentiment, retweets, and temporal patterns where available.

## Methodological Patterns Across the Literature

### 1. Systematic reviews establish scope, definitions, and gaps

Suarez-Lledo and Alvarez-Galvez, Wang et al., Baris-Schlicht et al., and Athira et al. all use review methods to organize fragmented research. Their value is not a single model result, but a defensible map of the field.

The reviews show that health misinformation varies by topic, platform, label definition, data source, and research method. This matters because it prevents overclaiming. A model that performs well on COVID tweets may not work on cancer treatment articles, vaccine videos, or press releases.

### 2. Labeling strategies differ substantially

The reviewed articles use different ways to decide whether content is misinformation:

- expert health review ratings, as in FakeHealth,
- fact-checking organization labels, as in Vosoughi et al.,
- source/domain credibility labels, as in Chen et al. and ReCOVery,
- curated credible and not-credible source sets, as in CoAID,
- expert stance and narrative annotation, as in COCO,
- study-specific content coding, as in the prevalence and spread reviews.

This is one of the biggest project-design issues. These label sources are not interchangeable. Expert article reviews, fact-check labels, source credibility, and stance annotations all measure different constructs. The final project should explicitly say what each dataset's label means.

### 3. Traditional machine learning remains useful

The literature supports starting with traditional baselines. Baris-Schlicht et al. emphasize that transformers dominate newer work, but traditional models such as logistic regression, SVM, random forest, naive Bayes, and gradient boosting remain important baselines. Di Sotto and Viviani also use classical models alongside neural models.

For this project, traditional baselines are useful because:

- they are easier to interpret,
- they establish whether the problem is learnable with simple features,
- they reveal dataset artifacts through feature weights,
- they provide fair comparisons before adding transformers.

### 4. Deep learning and transformers are useful but not automatically better

The automatic detection literature uses CNNs, recurrent networks, BERT, RoBERTa, SciBERT, XLM-R, MARBERT, sentence-transformer models, and T5-style models. Transformers are especially useful for short, informal social media text and for capturing semantic context.

However, the literature repeatedly warns about:

- token-length limits,
- compute requirements,
- bias inherited from pretraining,
- fine-tuning instability with small datasets,
- poor transfer across topics and platforms,
- high performance caused by source or topic artifacts.

This means transformer experiments should be included if feasible, but the project should not depend entirely on them. The strongest report would compare traditional baselines, transformer results, and explainability/error analysis.

### 5. Feature engineering is especially important in health misinformation

Di Sotto and Viviani provide the clearest feature taxonomy. They organize useful features into:

- textual representation features,
- linguistic and stylistic features,
- emotional features,
- medical and domain-specific features,
- propagation or network features,
- user-profile features.

This is highly compatible with the independent study. It supports a modular experimental design:

- text-only classification,
- text plus sentiment/emotion,
- text plus medical/domain features,
- engagement-only classification where feasible,
- text plus engagement,
- optional source or user features with caution.

### 6. Narrative analysis should be qualitative as well as computational

Chen et al. and COCO both show that misinformation narratives are not captured well by binary labels alone. Chen et al. identify COVID-19 misinformation narratives around hydroxychloroquine and alternative medicines, US officials and governing agencies, and prevention measures. COCO shows that conspiracy content requires topic and stance labels, because mentioning a conspiracy does not mean supporting it.

For this project, topic modeling should not be presented as automatically discovering "truth." Topic models should be used to generate candidate themes, followed by qualitative interpretation and example-based validation.

### 7. Explainability is necessary for trust and error analysis

Athira et al. show that explainability in fake-news detection is still underdeveloped. Many systems focus on attention weights, but attention is contested as an explanation. Other explanation methods include perturbation, ablation, feature relevance, mimic learning, visual explanations, and relational explanations.

For this project, explainability can serve two purposes:

- stakeholder communication: explain what textual, emotional, medical, or engagement cues make an item suspicious,
- model auditing: identify source leakage, topic leakage, overreliance on sensational words, and failures on debunking or neutral discussion.

The stakeholder point matters. A public-health communication team does not need the same explanation as a machine-learning researcher. Public-health users need narrative-level and claim-level interpretation; model developers need feature-level diagnostics.

## Major Findings Across the Articles

### Health misinformation is topic-specific

Suarez-Lledo and Alvarez-Galvez find that the reviewed literature clusters around vaccines, drugs and smoking, noncommunicable diseases, pandemics, eating disorders, and medical treatments. The reported prevalence of misinformation differs by topic. Drugs and smoking products had especially high misinformation prevalence in some studies, while vaccine misinformation was also common.

This means the project should avoid saying "health misinformation behaves one way." It is better to ask whether the patterns differ between COVID, general health news, cancer-related claims, treatment claims, vaccine claims, and other subtopics.

### Platform access shapes research

Twitter appears repeatedly because it has historically been easier to access through APIs and public datasets. This creates a research bias. Findings about Twitter do not automatically generalize to YouTube, Facebook, TikTok, Reddit, web articles, or messaging apps.

This matters for the project because CoAID, FakeHealth, Chen et al., Vosoughi et al., and COCO all involve Twitter in some way. The final report should state that the project studies misinformation in article/social-sharing environments rather than the entire online health information ecosystem.

### False information can spread differently from true information

Vosoughi et al. show that false news spreads farther, faster, deeper, and more broadly than true news on Twitter. False stories were also more novel and produced different emotional reactions, with more fear, disgust, and surprise. Bots accelerated both true and false news similarly, so human sharing behavior was central.

This supports the social engagement analytics part of the project. If false or unreliable health information behaves differently in engagement patterns, those signals may help detection and risk prioritization.

### Engagement features can help, but they are not universally reliable

FakeHealth shows that engagement differences exist but are not uniform. In HealthStory, real stories had larger tweet and retweet counts than fake stories, while HealthRelease did not show the same significant difference. Reply sentiment and temporal engagement patterns may be more informative in some settings.

This is important because the project should not assume that "more engagement equals misinformation." Instead, the project should test whether engagement features add predictive or interpretive value by dataset and by feature type.

### COVID misinformation is strongly narrative-based

Chen et al. show that COVID misinformation on Twitter clustered around narratives such as hydroxychloroquine and alternative medicines, US officials and governing agencies, and prevention measures. These narratives were linked to political alignment and dense sharing communities.

COCO reinforces this point by treating conspiracy theories as categories with stance labels. This suggests that the project's narrative component can be valuable even if classification performance is modest.

### Dataset differences are a major risk and research opportunity

Baris-Schlicht et al. show that COVID datasets differ from broader health misinformation datasets in linguistic, affective, and biomedical patterns. Their review also notes that many health misinformation datasets are small, English-language, COVID-focused, and hard to generalize from.

This makes cross-dataset evaluation one of the strongest parts of the independent study. Testing a model trained on CoAID against FakeHealth, or vice versa, can reveal whether the model learns general misinformation signals or dataset-specific artifacts.

### Health-specific explanation is more valuable than generic fake-news explanation

FakeHealth's review criteria show that a health article may be misleading for specific reasons:

- it fails to quantify harms,
- it exaggerates benefits,
- it ignores costs,
- it does not compare alternatives,
- it overstates novelty,
- it lacks independent sources,
- it relies too heavily on a news release,
- it uses sensational language,
- it does not explain evidence quality.

This suggests that the explainability component should try to connect model explanations to health communication quality, not only generic words associated with fake news.

## Methodological Successes Worth Reusing

### FakeHealth: expert criteria plus social context

FakeHealth succeeds because it combines expert health-review ratings with article content and social engagement. It is not just a scrape of random social posts. It has a health-quality rationale behind the labels.

This is useful for the project because it can support a richer discussion of explainability. A false label can be connected to concrete reasons, such as not explaining harms or overstating benefits.

### Di Sotto and Viviani: modular feature groups

Di Sotto and Viviani succeed methodologically by treating misinformation detection as a multi-feature problem. Their feature groups are useful for turning the project into controlled experiments.

A strong project design could follow this style:

- Model 1: text representation only.
- Model 2: text plus sentiment/emotion.
- Model 3: text plus medical/domain features.
- Model 4: engagement features only.
- Model 5: text plus engagement.
- Model 6: transformer text model.
- Model 7: transformer plus selected metadata or engagement features, if feasible.

The value is not only accuracy. The experiment tells us which information types matter.

### Chen et al.: topic modeling plus network interpretation

Chen et al. succeed by not stopping at topic modeling. They connect topics to domains, political leaning, and retweet communities. This turns topics into interpretable misinformation narratives.

For this project, a smaller-scale version could be:

- extract topics from misinformation and reliable content separately,
- label topics qualitatively,
- compare engagement across topics,
- identify which misinformation themes appear most amplified,
- connect topics to sentiment and model explanations.

### COCO: stance-aware narrative annotation

COCO succeeds because it separates topic from stance. This is a major conceptual point. A post can mention a false claim in order to criticize, debunk, quote, joke about, or support it.

For the project, this means topic modeling results should be interpreted cautiously. If a topic includes "vaccine," "microchip," and "Bill Gates," that does not prove every item in the topic endorses a conspiracy. Manual examples and stance awareness are needed.

### Vosoughi et al.: diffusion metrics

Vosoughi et al. succeed by operationalizing spread. They use cascade size, depth, breadth, structural virality, and time to reach audience thresholds. Even if this project cannot reconstruct full cascades, it can borrow the logic:

- engagement volume matters,
- engagement speed matters if timestamps are available,
- retweets and replies may behave differently,
- emotional responses are part of spread.

### Athira et al.: explainee-aware XAI

Athira et al. succeed by framing explainability around the user of the explanation. This is useful for the final report because the project has multiple audiences:

- technical evaluator: wants model diagnostics,
- public-health communicator: wants narrative and risk interpretation,
- management audience: wants workflow and decision-support implications.

The project should present explanations in a way that serves the stakeholder, not only the model developer.

## Caveats and Risks for the Project

### Label heterogeneity

The datasets do not use the same definition of misinformation. FakeHealth uses health-review ratings. CoAID uses credible and not-credible sources and claims. ReCOVery uses source reliability. Chen et al. use domain-level Media Bias/Fact Check categories. COCO uses conspiracy topic and stance annotation. Vosoughi et al. use fact-checking organizations.

This means cross-dataset results must be interpreted carefully. Poor transfer may reflect label mismatch, not only model weakness.

### Source leakage

Several datasets are vulnerable to source leakage. If labels are based on source reliability, then models may learn publishers or domain terms rather than health claim quality. Even FakeHealth can have publisher patterns.

Potential safeguards:

- compare models with and without source features,
- inspect top features,
- perform source-aware error analysis if metadata allows,
- avoid claiming that source-based performance proves claim-level understanding.

### COVID overrepresentation

COVID dominates recent health misinformation research. This is understandable but limiting. Models trained on COVID content may not generalize to broader health domains such as cancer, nutrition, chronic disease, vaccines outside COVID, smoking, or medical treatments.

This supports using FakeHealth as a broader health-news comparison dataset.

### Twitter-centric evidence

Most of the data sources involve Twitter. Twitter data is useful for engagement and diffusion analysis, but the project should avoid generalizing too broadly to all social media or all public information environments.

### API decay and missing social data

Datasets that provide tweet IDs often require hydration. Some tweets, replies, or user profiles may no longer be available due to deletion, suspension, privacy changes, or platform access restrictions. Di Sotto and Viviani's cleaned counts show that hydrated datasets can be smaller than original reported counts.

The project should plan for missing engagement data and treat this as a limitation, not a failure.

### Binary classification may be too blunt

The literature suggests that a binary label is often not enough. Health misinformation can be exaggerated, incomplete, misleading, unsupported, sensationalized, or source-distorted. COCO also shows that stance matters.

Binary classification is still useful as a first task, but the final project should include narrative, explanation, and qualitative error analysis to avoid oversimplification.

### Explanation quality is hard to evaluate

Athira et al. show that explainability lacks standard evaluation benchmarks. This means the project should not overclaim explanation quality. It can use explanations for model auditing and illustrative stakeholder interpretation, but should be clear about limitations.

## Brainstorming Implications for Re-Evaluating the Current Project

### 1. Keep the integrated pipeline framing

The strongest project framing is not "build the best fake-news classifier." A better framing is:

> Can NLP and social engagement analytics help detect, explain, and monitor health misinformation narratives across different datasets?

This framing fits the literature better because it includes detection, narrative analysis, engagement behavior, explainability, and transferability.

### 2. Make dataset provenance a central part of the methodology

The final report should explicitly describe where labels come from:

- FakeHealth: expert review ratings from HealthNewsReview.
- CoAID: credible and not-credible sources/claims around COVID-19.
- ReCOVery, if used: source reliability from NewsGuard and Media Bias/Fact Check.
- Any Twitter engagement: hydrated or provided tweet, reply, and retweet metadata.

This will make the project more rigorous because the reader can see that "misinformation" means different things in different datasets.

### 3. Treat CoAID and FakeHealth as complementary, not equivalent

CoAID is COVID-specific and likely useful for pandemic-era misinformation. FakeHealth is broader health news and includes expert review criteria. They should not be treated as interchangeable datasets.

A useful comparison is:

- CoAID tests COVID-19 misinformation detection.
- FakeHealth tests broader health-news reliability.
- Cross-dataset testing asks whether health misinformation signals generalize.

### 4. Use a layered experimental design

The literature supports a staged design:

1. Dataset profiling: labels, class balance, topics, sources, missing engagement data.
2. Text-only baselines: TF-IDF with logistic regression, SVM, or random forest.
3. Transformer model: BERT/RoBERTa-style classifier if feasible.
4. Engagement-feature models: tweet counts, retweets, replies, sentiment in replies, temporal features if available.
5. Combined models: text plus engagement.
6. Cross-dataset transfer: train on one dataset and test on another where label compatibility permits.
7. Narrative analysis: topic modeling plus qualitative labeling.
8. Explainability: feature importance, SHAP/LIME or model-specific explanations, and error analysis.

### 5. Put narrative discovery after classification, not before it

Classification can identify broad reliable/unreliable categories, but narrative analysis can explain what misinformation themes are present. A practical order might be:

- run classification baselines,
- identify false positives and false negatives,
- perform topic modeling on misinformation-labeled content and high-confidence model predictions,
- manually interpret the most important topics,
- compare narrative topics against engagement and sentiment patterns.

This order lets narrative analysis support both research findings and model interpretation.

### 6. Use engagement as a separate research question

The literature does not say engagement is always predictive. It says engagement can reveal diffusion, community response, novelty, emotional reaction, and risk.

The research question should be phrased carefully:

> Do social engagement features improve detection or interpretation of health misinformation compared with text-only models?

This is better than assuming engagement will improve accuracy. Even if engagement does not improve classification, it may still help prioritize high-risk misinformation narratives.

### 7. Include a transferability question

Baris-Schlicht et al. make cross-dataset generalization a major issue. This project can contribute by explicitly testing transfer:

- train on CoAID, test on FakeHealth,
- train on FakeHealth, test on CoAID,
- compare in-domain and out-of-domain performance,
- inspect which errors appear during transfer.

If transfer performance is weak, that is still a meaningful finding. It would support the claim that health misinformation detection is highly domain-specific.

### 8. Make explainability concrete

The project should not only say "we used explainable AI." It should specify what explanations are for:

- model debugging: Are top features meaningful or artifacts?
- stakeholder interpretation: What narrative or language cues made content risky?
- dataset critique: Does the model rely on source names, COVID terms, or sensational language?
- public-health use: What claims or themes would communication teams need to respond to?

For traditional models, feature weights and SHAP-style explanations may be enough. For transformers, local explanations can be used selectively for representative cases.

## Possible Revised Research Questions

These are brainstorming options, not changes to the current plan.

1. How accurately can traditional machine learning and transformer-based NLP models detect health misinformation in CoAID and FakeHealth?
2. Do social engagement features improve misinformation detection compared with text-only models?
3. Which linguistic, emotional, medical, and engagement features are most associated with unreliable health information?
4. How well do models trained on one health misinformation dataset transfer to another dataset with different topics and label construction?
5. What recurring misinformation narratives appear in the datasets, and how do these narratives differ in engagement or emotional tone?
6. Can model explanations reveal meaningful health misinformation cues, or do they expose dataset artifacts such as source leakage and topic leakage?

## Possible Revised Contribution Statement

The study can be framed as follows:

> This project develops and evaluates an interpretable health misinformation analytics pipeline that combines NLP classification, social engagement features, narrative discovery, cross-dataset validation, and explainability. Rather than treating misinformation detection as a standalone binary classification problem, the project examines how dataset provenance, social response, and narrative structure shape the detection and interpretation of health misinformation.

This contribution is realistic for an independent study because it does not require inventing a new deep learning architecture. Its value comes from integration, evaluation, and interpretation.

## Practical Design Choices Suggested by the Literature

| Design choice | Literature support | Why it matters |
|---|---|---|
| Use traditional ML baselines before transformers | Baris-Schlicht et al.; Di Sotto and Viviani; Dai et al. | Establishes interpretable, reproducible benchmarks |
| Use FakeHealth as a secondary or robustness dataset | Dai et al.; Di Sotto and Viviani | Provides expert health review labels and broader health-topic coverage |
| Use CoAID as the main COVID misinformation dataset | Di Sotto and Viviani | Supports COVID-specific health misinformation detection and engagement analysis |
| Compare text-only and text-plus-engagement models | Dai et al.; Di Sotto and Viviani; Vosoughi et al. | Tests whether social behavior adds value beyond language |
| Include sentiment and emotion features | Vosoughi et al.; Di Sotto and Viviani | False information may produce distinctive emotional patterns |
| Use topic modeling with qualitative validation | Chen et al.; COCO | Supports narrative discovery without overclaiming automated interpretation |
| Include cross-dataset transfer | Baris-Schlicht et al. | Tests generalization across health topics and label types |
| Use explainability for auditing | Athira et al.; Dai et al. | Helps identify meaningful cues, source leakage, and model failure modes |

## Management and Public-Health Implications

The literature suggests several applied implications for the final project:

- Public-health teams need narrative monitoring, not only misinformation counts.
- Engagement features may help prioritize which misinformation narratives deserve response.
- Explanations should translate model outputs into reasons that humans can act on.
- Different health topics require different communication strategies.
- Source credibility is useful but should not replace claim-level analysis.
- Early detection matters because some misinformation grows through rapid engagement spikes.
- Misinformation interventions should account for audiences, communities, and prior beliefs.

The final report can connect this to business analytics by positioning the pipeline as a risk-monitoring and decision-support system. The output is not only "this item is fake." A more useful output is:

- what the content is about,
- why it appears unreliable,
- how much engagement it receives,
- what emotion or framing it uses,
- what narrative it belongs to,
- whether the model is confident,
- whether the explanation appears meaningful or artifact-driven.

## Most Important Open Questions Before Revising the Project Plan

1. Are CoAID and FakeHealth both fully accessible locally, including enough social engagement metadata for analysis?
2. Will the project classify only article/news content, or also tweets and replies?
3. Should the main classification label be "fake/real," "credible/not credible," "reliable/unreliable," or another term that better matches dataset provenance?
4. How much cross-dataset transfer is feasible given label differences?
5. Should engagement features be used for prediction, risk prioritization, or both?
6. Should narrative analysis focus on all content, only misinformation-labeled content, or model errors?
7. What explanation format is most useful for the intended audience: feature weights, local examples, narrative summaries, or health-quality criteria?

## Bottom-Line Direction

The literature supports the current broad plan, but it also suggests refining the project around three central ideas:

1. Dataset provenance matters. Each dataset's labels are created differently, so the project should explain and test those differences.
2. Engagement matters, but not always as a simple predictor. It may be most useful for understanding reach, risk, emotion, and prioritization.
3. Explainability and narrative analysis are not optional extras. They are what turn a classifier into a useful health misinformation decision-support tool.

The strongest version of the independent study is therefore an integrated, interpretable, cross-dataset health misinformation analytics project rather than a single-model performance report.

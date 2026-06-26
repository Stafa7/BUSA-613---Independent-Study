# Dai, Sun, and Wang 2020 - FakeHealth Dataset

## Citation

Dai, E., Sun, Y., & Wang, S. (2020). *Ginger Cannot Cure Cancer: Battling Fake Health News with a Comprehensive Data Repository*. Proceedings of the International AAAI Conference on Web and Social Media, 14(1), 853-862. https://doi.org/10.1609/icwsm.v14i1.7350

## Source Files

- PDF: `docs/literature_pdfs/Ginger Cannot Cure Cancer- Battling Fake Health News with a Comprehensive Data Repository.pdf`
- Converted Markdown: `docs/literature_mds/ginger_cannot_cure_cancer_battling_fake_health_news_with_a_comprehensive_data_repository.md`

## Purpose

This paper introduces FakeHealth, a comprehensive data repository for fake health news detection. The authors argue that fake health news differs from general fake news because health claims can use subtle distortions such as confusing association with causation, mixing absolute and relative risk, or misrepresenting medical evidence. Health misinformation is also harder for ordinary users to judge because it requires specialized knowledge.

The authors position explainability as essential in fake health news detection. Explanations can help users trust detection results, stop propagating harmful claims, and help researchers diagnose model bias or training-data problems.

## Dataset Design

FakeHealth consists of two datasets: HealthStory and HealthRelease. Both are built from HealthNewsReview.org, which reviewed health news stories and press releases from 2005 to 2018. HealthNewsReview used reviewers from journalism, medicine, health services research, public health, and patient communities, and assessed stories with a standard rating system.

FakeHealth includes four major information categories:

- news contents: URL, title, keywords, text, images, tags, authors, date,
- news reviews: ratings, evaluation criteria, explanations, tags, category, title, summary, source,
- social engagements: tweets, replies, retweets,
- user network: user profiles, timelines, followers, and followings.

The authors report collecting approximately 500,000 tweets, 29,000 replies, 14,000 retweets, and 27,000 user profiles with timelines and friend lists. They classify news items with ratings below 3 as fake, following prior fake-news dataset practice.

## Health Evaluation Criteria

A major strength of FakeHealth is that it includes ten review criteria. Eight criteria are common across HealthStory and HealthRelease, including whether the news compares the new approach with alternatives, explains harms, evaluates evidence quality, quantifies benefits, establishes novelty, discusses availability, avoids disease-mongering, and discusses costs. HealthStory has two extra criteria about independent sources and reliance on news releases. HealthRelease has two extra criteria about funding/conflicts of interest and unjustified sensational language.

These criteria make the dataset valuable for explainable health misinformation detection because a model could potentially identify not only whether content is unreliable, but also which health-news standard it fails to meet.

## Exploratory Findings

The authors examine labels, topics, sources, social engagements, replies, user credibility, temporal engagement patterns, and bot activity.

The topic analysis suggests that real and fake health news share many topics, which is important because a dataset should not simply allow models to separate fake and real news by topic alone. However, there are marginal differences in topic distributions, and some topic features such as tags and keywords may still provide useful predictive information.

The source analysis finds that source quality differs across publishers. For example, in HealthStory, some news agencies have significantly different average ratings. This suggests that source credibility can be a useful feature but also raises the risk that models may overfit to source artifacts.

The engagement analysis finds that social engagement patterns differ across real and fake content, but not uniformly across both subsets. In HealthStory, tweet and retweet counts for real stories are larger than fake stories, while HealthRelease does not show significant differences in tweet and retweet counts. Replies may help identify fake health news because sentiment patterns differ in HealthStory. The authors also find that users who propagate fake news are slightly more likely to be bots, but both real and fake health news are spread by normal users and bots.

Temporal engagement patterns are also informative. Some fake health news exhibits spike-like temporal patterns, including repeated or coordinated bursts, while real news may grow more steadily.

## Baseline Models and Results

The authors evaluate several baselines:

- logistic regression with unigram features,
- unigram plus news source,
- unigram plus tags,
- SVM,
- random forest,
- CNN,
- bidirectional GRU,
- social context-based SAF model.

On HealthStory, CNN obtains AUC 0.814 and SAF obtains F1 0.756. On HealthRelease, SAF performs best with accuracy 0.810, F1 0.802, and AUC 0.809. The results show that text, source/tag features, and social-context features can all be useful, but performance remains limited enough to motivate new approaches.

## Future Research Directions

The paper identifies several directions:

- explainable fake health news detection using the ten review criteria,
- knowledge-based detection using health knowledge graphs,
- credibility-based detection using topic, source, and user credibility,
- multimodal detection using article images, tweet images, and user profile images,
- early detection before misinformation spreads widely,
- propagation analysis to identify high-impact fake health news.

## Relevance to This Independent Study

FakeHealth is directly relevant as the secondary dataset. This paper supports the project's emphasis on text plus engagement, explainability, and cross-dataset validation. It also provides a methodological warning: if source features improve prediction, the report must distinguish useful credibility signals from leakage or source overfitting.

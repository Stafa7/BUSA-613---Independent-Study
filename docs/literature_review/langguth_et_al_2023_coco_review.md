# Langguth et al. 2023 - COCO COVID-19 Conspiracy Theory Dataset

## Citation

Langguth, J., Schroeder, D. T., Filkukova, P., Brenner, S., Phillips, J., & Pogorelov, K. (2023). *COCO: an annotated Twitter dataset of COVID-19 conspiracy theories*. Journal of Computational Social Science. https://doi.org/10.1007/s42001-023-00200-3

## Source Files

- PDF: `docs/literature_pdfs/COCO- an annotated Twitter dataset of COVID-19 conspiracy theories.pdf`
- Converted Markdown: `docs/literature_mds/coco_an_annotated_twitter_dataset_of_covid_19_conspiracy_theories.md`

## Purpose

This paper presents COCO, a manually annotated Twitter dataset for COVID-19 conspiracy theory narratives. The authors argue that COVID-19 misinformation included many competing narratives and that true/false labels alone are insufficient for studying or detecting those narratives. The dataset is designed to support topic and stance detection, individually or jointly.

The paper defines conspiracy theories as narratives involving disproved or unproven accusations against powerful individuals or groups, used to explain major events through clandestine malevolent schemes. The authors focus on conspiracy theories because they were among the most prominent misinformation phenomena during the pandemic.

## Dataset Creation

The raw data were collected from Twitter between January 17, 2020 and June 30, 2021 using COVID-related keywords. The broader raw dataset contained approximately two billion statuses. The authors removed retweets, quotes, and replies, leaving more than 400 million tweets.

Because conspiracy tweets are rare, the authors did not randomly sample. Instead, they used a list of 54 conspiracy-related keywords to find candidate tweets. This reduced the pool to slightly more than one million tweets. They then removed tweets with hyperlinks to focus on language understanding rather than link analysis, attempted to resolve user location, removed tweets without resolvable location, and selected longer tweets because very short tweets were difficult to interpret. This produced a candidate set of about 100,000 tweets.

The final dataset contains 3,495 tweets. Each tweet is labeled with respect to 12 conspiracy categories, producing 41,940 category-level labels. Each tweet-category label is determined by majority vote among three expert annotators.

## Annotation Design

The dataset labels both topic and stance. This is important because a tweet can mention a conspiracy theory without endorsing it. The annotation scheme distinguishes between tweets that support/promote a conspiracy, tweets that are related but do not support it, and tweets that are unrelated to the category.

This stance-aware design is a major contribution because many misinformation detection systems confuse topic mention with endorsement. For example, a tweet debunking a conspiracy could contain the same keywords as a tweet promoting it.

## Main Contributions

COCO contributes:

- a quality-controlled manually annotated dataset,
- 12 COVID conspiracy-topic categories,
- stance labels for each category,
- qualitative and quantitative descriptions of conspiracy narratives,
- evidence that BERT can be used successfully for the combined topic/stance task.

The dataset can be used for classifier training, prevalence analysis, narrative analysis, and evaluation of NLP systems' ability to handle multi-category misinformation content.

## Limitations

COCO is COVID-specific and Twitter-specific. It focuses on conspiracy theories, which are only one form of health misinformation. The candidate set is keyword-selected, so it is not designed to estimate general misinformation prevalence. The removal of tweets with links supports language-focused modeling but excludes link-sharing behavior that may matter for misinformation spread.

## Relevance to This Independent Study

This paper is useful for the narrative-discovery and qualitative interpretation phases. It shows that misinformation should be treated as a set of narratives and stances, not only a binary label. It also warns that unsupervised topic modeling must be interpreted carefully because topic mention is not the same as endorsement.

For the independent study, COCO can guide how discovered CoAID or FakeHealth topics are labeled and interpreted, even if COCO itself is not used as a core dataset.

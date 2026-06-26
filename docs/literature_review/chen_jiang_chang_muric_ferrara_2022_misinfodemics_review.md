# Chen, Jiang, Chang, Muric, and Ferrara 2022 - COVID-19 Misinfodemic Narratives

## Citation

Chen, E., Jiang, J., Chang, H.-C. H., Muric, G., & Ferrara, E. (2022). *Charting the Information and Misinformation Landscape to Characterize Misinfodemics on Social Media: COVID-19 Infodemiology Study at a Planetary Scale*. JMIR Infodemiology, 2(1), e32378. https://doi.org/10.2196/32378

## Source Files

- PDF: `docs/literature_pdfs/Charting the Information and Misinformation Landscape to Characterize Misinfodemics on Social Media- COVID-19 Infodemiology Study at a Planetary Scale.pdf`
- Converted Markdown: `docs/literature_mds/charting_the_information_and_misinformation_landscape_to_characterize_misinfodemics_on_social_media_covid_19_infodemiolo.md`

## Purpose

This paper analyzes COVID-19 discourse and misinformation narratives on Twitter at very large scale. The authors argue that lockdowns and social distancing moved communication and information consumption online, while health authorities also shifted public-health messaging to social platforms. This created opportunities for timely information sharing but also opened the gate for misinformation and misinfodemics.

The paper aims to identify misinformation narratives, describe the audiences engaging with them, and analyze network mechanisms that drive misinformation spread.

## Data and Methods

The study uses a public COVID-19 Twitter dataset containing more than one billion tweets. For the paper's main analysis, the authors use release v2.45, covering January 21, 2020 to April 30, 2021, with 1,443,871,621 tweets. Tweets were collected in real time through Twitter's streaming API using manually curated COVID-related keywords.

To identify misinformation-related discourse, the authors isolate tweets containing URLs from domains classified by Media Bias/Fact Check as questionable or conspiracy-pseudoscience sources. This produced a misinformation subset of 12,689,165 tweets.

The methods include:

- topic modeling with LDA,
- clustering methods for discussion topics,
- domain/source analysis,
- political leaning inference based on shared domains,
- retweet network analysis,
- comparison of conspiracy, questionable, and random source networks.

## Main Findings

The paper identifies 11 major general COVID-19 discussion topics that align with pandemic events over time. For misinformation-specific discourse, the paper identifies three major narratives:

- hydroxychloroquine and alternative medicines,
- US officials and governing agencies,
- COVID-19 prevention measures.

The hydroxychloroquine narrative involved hashtags and discourse around HCQ, early treatment, claims of efficacy, opposition to Fauci, censorship claims, and political interpretations of medical decisions. The US officials/governing-agencies narrative involved political distrust, claims about Fauci, Bill Gates, vaccines, Pfizer, and government action. The prevention-measures narrative involved masks, lockdowns, public-health guidance, and claims that prevention measures were ineffective or politically motivated.

The authors find that misinformation narratives are not only topical but also political and networked. Users sharing questionable or conspiracy-related content form dense communities. Political alignment is associated with the narratives users engage with. The paper finds a concentration of hydroxychloroquine-related discourse around right-leaning users, while other narratives have different engagement structures.

The paper also finds increasing sharing from conspiracy and questionable sources over time. Domain-sharing networks for conspiracy and questionable sources are much more tightly clustered than random-source networks, suggesting that users who share unreliable sources often share multiple unreliable domains.

## Limitations

The authors note several limitations. Their keyword list was English and manually selected, which shapes the dataset. The use of Media Bias/Fact Check domains is a proxy for misinformation exposure, not a claim that every tweet or every article from those domains is misinformation. Political leaning is inferred from domain sharing, which is useful but imperfect.

## Relevance to This Independent Study

This paper is one of the strongest models for the narrative-analysis component. It shows how topic modeling and source-domain analysis can identify practical misinformation narratives rather than simply producing abstract topics.

It supports the project's plan to use topic modeling for narrative discovery, analyze engagement patterns, and translate findings into public-health monitoring recommendations. It also reinforces the need to interpret topics qualitatively and consider audience/community structure when available.

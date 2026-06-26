# Wang, McKee, Torbica, and Stuckler 2019 - Spread of Health-Related Misinformation on Social Media

## Citation

Wang, Y., McKee, M., Torbica, A., & Stuckler, D. (2019). *Systematic Literature Review on the Spread of Health-related Misinformation on Social Media*. Social Science & Medicine, 240, 112552. https://doi.org/10.1016/j.socscimed.2019.112552

## Source Files

- PDF: `docs/literature_pdfs/Systematic Literature Review on the Spread of Health-related Misinformation on Social Media.pdf`
- Converted Markdown: `docs/literature_mds/systematic_literature_review_on_the_spread_of_health_related_misinformation_on_social_media.md`

## Purpose

This review studies how health-related misinformation spreads on social media and what mechanisms drive its circulation. The authors situate health misinformation in a longer history of rumor and false information, but argue that the internet and social media have changed the speed, reach, and amplification of misinformation. In health, these dynamics are especially serious because misinformation can delay effective care, discourage prevention, and threaten lives.

The paper avoids the term "fake news" because it is politically overloaded and conceptually imprecise. Instead, it uses misinformation as an umbrella term for false information related to health, especially when the intent behind sharing cannot be established.

## Conceptual Framework

A key conceptual contribution is the agent-message-interpreter framing. Misinformation spread involves:

- the agent or actor who creates or distributes the message,
- the message itself, including its content, framing, ambiguity, and durability,
- the interpreter or receiver, whose beliefs, susceptibility, and sharing decisions reproduce the misinformation.

At the micro level, individuals judge believability based on source, narrative, and context. At the macro level, misinformation appears through cascades, network patterns, and community structures. This framework is useful for distinguishing content analysis from engagement and propagation analysis.

## Methodology

The authors searched PubMed, Cochrane, Web of Science, Scopus, and Google for empirical and methodological articles published between 2012 and 2018. The final review included 57 articles for full-text analysis. The review covers multiple disciplines, including public health, communication, psychology, epidemiology, and computational science.

## Main Findings

The review finds that health misinformation research increased sharply during the period studied. The most extensively studied misinformation topics were vaccination, Ebola, and Zika virus. Other topics included nutrition, cancer, water fluoridation, smoking, diabetes, heart failure, hypertension, and cancer screening.

The reviewed studies commonly used:

- content analysis,
- social network analysis,
- experiments,
- epidemiological or simulation-style models.

The theoretical foundations were fragmented. Some studies drew on psychology, especially rumor theory, credibility, confirmation bias, and belief systems. Others drew on network science, including social influence, social learning, contagion, homophily, and polarization. The authors identify weak integration across fields and argue that more interdisciplinary research is needed.

Several findings are especially important for the independent study. The review shows that misinformation is often spread by individuals who are highly active in influencing opinions. It also highlights that rumor sharing is shaped by fear, dread, ambiguity, source credibility, and prior beliefs. Studies on Zika and Ebola show that public-health emergencies generate misinformation narratives quickly, and corrective interventions can vary in effectiveness depending on context and audience.

## Discussion and Implications

The authors argue that misinformation thrives when health topics are important but ambiguous. Emerging infectious diseases are especially vulnerable because scientific uncertainty is high, guidance changes over time, and people seek immediate explanations. This matters for COVID-19 misinformation, where uncertainty and changing evidence were central features of the information environment.

The paper also argues that interventions should target both individuals and networks. Correcting misperceptions can work in some cases, but misinformation spread is also shaped by platform structure, user communities, and message framing. This is directly relevant to social engagement analytics.

## Limitations

The review predates COVID-19 and modern transformer-based NLP. Its value is conceptual and interdisciplinary rather than technical implementation guidance. The paper does not provide a computational pipeline, and it does not evaluate specific classifiers.

## Relevance to This Independent Study

This paper is important for framing the management and public-health problem. It supports the project's decision to analyze misinformation as more than a classification label. The agent-message-interpreter framework can be used to position the proposed pipeline: text models analyze the message, engagement features approximate interpreter/platform response, and source/user features may be added when available.

The paper also motivates research questions about whether engagement features improve detection and whether certain narratives generate stronger engagement.

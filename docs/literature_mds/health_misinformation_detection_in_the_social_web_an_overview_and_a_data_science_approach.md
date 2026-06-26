# Health Misinformation Detection in the Social Web- An Overview and a Data Science Approach

## Source PDF

`docs/literature_pdfs/Health Misinformation Detection in the Social Web- An Overview and a Data Science Approach.pdf`

## Extraction Notes

- Converted from PDF using Ghostscript `txtwrite`.
- This is a working Markdown conversion for analysis; tables, figures, equations, and multi-column layout may need manual cleanup against the original PDF.
- The text below is extracted from the downloaded PDF, not independently rewritten.

## Extracted Text

International Journal of
 Environmental Research
 and Public Health
 Article
 HealthMisinformationDetectionintheSocialWeb:
 AnOverviewandaDataScienceApproach
 StefanoDiSottoandMarcoViviani*
 DepartmentofInformatics,Systems,andCommunication(DISCo),UniversityofMilano-Bicocca,
 Ediﬁcio U14-ABACUS,VialeSarca,336,20126Milan,Italy;s.disotto@campus.unimib.it
 * Correspondence: marco.viviani@unimib.it
 Abstract: The increasing availability of online content these days raises several questions about
 effective access to information. In particular, the possibility for almost everyone to generate content
 withnotraditionalintermediary,ifontheonehandledtoaprocessof“informationdemocratization”,
 ontheotherhand,hasnegativelyaffected the genuineness of the information disseminated. This
 issue is particularly relevant when accessing health information, which impacts both the individual
 and societal level. Often, laypersons do not have sufﬁcient health literacy when faced with the
 decision to rely or not rely on this information, and expert users cannot cope with such a large
 amountofcontent. Forthesereasons,thereisaneedtodevelopautomatedsolutionsthatcanassist
 bothexpertsandnon-expertsindiscerningbetweengenuineandnon-genuinehealthinformation. To
 makeacontributioninthisarea,inthispaperweproceedtothestudyandanalysisofdistinctgroups
 of features and machine learning techniques that can be effective to assess misinformation in online
 health-related content, whether in the form of Web pages or social media content. To this aim, and for
 evaluation purposes, we consider several publicly available datasets that have only recently been
 generated for the assessment of health misinformation under different perspectives.
  Keywords: health misinformation; information disorder; consumer health; information access; social
 
 Citation: Di Sotto, S.; Viviani, M. Web;machinelearning;deeplearning;datascience
 HealthMisinformationDetectionin
 the Social Web: An Overview and a
 DataScienceApproach. Int. J. 1. Introduction
 Environ. Res. Public Health 2022, 19,
 2173. https://doi.org/10.3390/ In contemporary society, access to information plays a crucial role, inﬂuencing choices
 ijerph19042173 andbehaviorsbothatthelevelofindividualsandcommunities. Web2.0technologieshave
 AcademicEditor: PaulB.Tchounwou enabledanyonetoplayanactiveroleineverystageoftheinformationlifecycle,fromits
 generation to its dissemination, especially through social media platforms. In this context,
 Received: 29 December 2021 characterized by “disintermediation” [1,2], it is essential to be able to distinguish what
 Accepted: 8 February 2022 is genuine information from what is not; this need is ampliﬁed, in particular, for those
 Published: 15 February 2022 contents that may be particularly delicate and sensitive, because they could have extremely
 Publisher’sNote: MDPIstaysneutral negative social repercussions, such as those related to health.
 with regard to jurisdictional claims in In this scenario, health misinformation can be deﬁned as “a health-related claim of fact
 publishedmapsandinstitutionalafﬁl- that is currently false due to a lack of scientiﬁc evidence” [3]. In most cases, people who are
 iations. not an expert in the ﬁeld are unable to properly assess the genuineness of such claims, both,
 in general, due to their limited cognitive capacities [4,5] and, more speciﬁcally, due to their
 insufﬁcient level of health literacy [6]. This latter concept was included in the glossary of the
 WorldHealthOrganization(WHO)in1988[7],andindicates“theabilityofacitizentoobtain,
 Copyright: © 2022 by the authors. process, and understand basic health information in order to make informed choices”.
 Licensee MDPI, Basel, Switzerland. Hence, the above-mentioned limitations, combined with the exponential increase in the
 This article is an open access article amountofuser-generated content, makes it necessary to develop methods that keep up with
 distributed under the terms and this growing trend and, at the same time, are able to reliably provide accurate predictions
 conditions of the Creative Commons aboutthepresenceofhealthmisinformation. This is because, in the vast majority of cases,
 Attribution (CC BY) license (https:// laypersons are called upon to play an active role in managing their own health and that of
 creativecommons.org/licenses/by/ others, contributing to social good [8].
 4.0/).
 Int. J. Environ. Res. Public Health 2022, 19, 2173. https://doi.org/10.3390/ijerph19042173 https://www.mdpi.com/journal/ijerph
 Int. J. Environ. Res. Public Health 2022, 19, 2173 2of20
 Despite the recent technological and methodological improvements in addressing
 the problem of identifying distinct forms of non-genuine information—a.k.a. information
 disorder [9]—in several domains [10], the studies focusing on health misinformation de-
 tection are still quite limited. This is found both at the level of proposing domain-speciﬁc
 solutions, and in terms of the availability of appropriately labeled data with respect to the
 genuineness of such kind of information. In the last period, also concerning the COVID-19
 pandemicandthegeneralinterestinaccesstohealthinformationonline,thissecondprob-
 lemhasbeensomehowtakenintoaccount;infact,someusefuldatasetsforthispurpose
 havebeenrecently madeavailable. This makes it possible, in this work, to study health
 misinformation features and machine learning solutions that can be employed for the
 problemunderconsideration,takingintoaccountbothWebandsocialmediacontentfrom
 the Twitter microblogging platform.
 Therefore, the main contributions of this work include, both from a theoretical and
 technological point of view, the following:
 • SummarizingthecharacteristicsoftheWeb,socialmediaplatforms,andhealth-related
 content being disseminated online, by considering factors of information genuineness
 in the health domain;
 • Identifying key features (both “general-purpose” and domain-speciﬁc) that may be
 useful for detecting health misinformation in Web pages and social media content;
 this involves both synthesizing features that have been used in the literature for this
 purpose and investigating additional features that may be useful for the purpose
 considered, but employed to date for different research tasks;
 • Studying the impact of such features when used in association with supervised learn-
 ing techniques; this objective requires a comparison between different approaches that
 havebeenusedsofartosolvetheproblemconsidered,especiallyconcerning“general”
 information (i.e., not health-related). This comparison involves the use of classical
 machinelearningalgorithmsandtheimplementationofConvolutionalNeuralNetworks,
 Bidirectional Long Short-Term Memory networks, and Hierarchical Attention Networks,
 whichhavebeenusedintheliteraturetotackledocumentclassiﬁcationproblemsand
 are used in this work as an additional baseline for comparison;
 • Evaluating the obtained results on publicly available datasets, which consider health
 misinformation in various communication media, in distinct forms, and with respect
 to various health-related topics.
 2. An OverviewonHealthMisinformationDetection
 Traditionally, the channel through which health-related content has been disseminated
 has always been an expert in the ﬁeld, such as a physician or a recognized health organiza-
 tion, in an “extremely intermediate” way [11]. With the advent of modern communication
 channels such as newspapers, radio and TV, the intermediation aspect through experts has
 beensomewhatloosened,aswithtimethesechannelshavealsobeguntodealwiththis
 type of information. However, it is with the advent of the Web that the expert–layperson
 relationship has totally changed, providing anyone with a powerful means of getting health-
 related content without going through direct contact with the expert. In such a scenario,
 different solutions assessing health misinformation in Web pages and social media content
 havebeenproposedinthelastfewyears.
 Before introducing these solutions, it is necessary to underline that various concepts
 that are totally or partially superimposable to that of information genuineness (which,
 in general, we employ in this article) have been used. In the state-of-the-art works detailed
 in the next section, it is possible to ﬁnd a reference, among others, to the concepts of
 reliability, truthfulness, trustworthiness, credibility, veracity, etc., which however can have
 speciﬁc meaningsbasedonwhethertheyrefertothesourceofinformation,theinformation
 itself, the communication medium through which information is propagated, or other
 theoretical aspects. Furthermore, reference was also made in the literature to the con-
 cepts of disinformation, understood as false information propagated with malicious intent,
 Int. J. Environ. Res. Public Health 2022, 19, 2173 3of20
 misinformation, understood as false information generated without malicious intentions,
 and malinformation, understood as genuine information shared to cause damage, often
 moving information destined to remain private in the public sphere [9]. In this article,
 as stated in the Introduction, we refer to the deﬁnition of health misinformation provided
 in [3], which makes no distinction between false information generated with the intent or
 not to cause harm.
 2.1. Manual or Pseudo-Automated Approaches
 In an attempt to understand the speciﬁc determinants related to online health mis-
 information, several studies were conducted starting from the early 2000s, especially in
 the ﬁelds of psychology and sociology, through the administration of questionnaires and
 bymeansofsurveyanalyses. Themainﬁndingsoftheseinitialstudiesconcernedaspects
 connectedtothetrustworthiness (and related concepts) of Web sites (in fact, social media
 wasinitsinfancy)andthecredibility (and related concepts) of the information published
 therein. They considered, among others, the Web site accessibility, the (perceived) quality
 of the content published, some of the users’ own characteristics, and the familiarity and
 habit in using the Web and content online [12]. The study presented in [13], in particular,
 reports and summarizes the following outcomes:
 • TheWebischaracterizedbythepresenceofaconsiderableamountofincompleteor
 poor-quality information, accompanied, however, by the presence of some excellent
 content and sources;
 • Usingkeywordsearches(searchbyquery),insteadofreferring to known URLs(naviga-
 tional search), increases the likelihood of stumbling upon less than credible
 health information;
 • Most users make a hasty reading of content, without doing research related to the
 organization, author, or source of the health information being disseminated;
 • When the source is taken into account, however, institutional sites are perceived
 as more credible than others whose source is unknown or not authoritative [14].
 Furthermore, users tend to trust content produced, sponsored, or published by health
 care institutions and physicians (e.g., if the content link back to, or cite, professionals
 in the ﬁeld). Conversely, “paid links” and broken links can reduce the credibility of a
 site or an article. However, there is no unequivocal agreement on a particular source,
 butit appears that differences in judgments are attributable to demographic factors
 andindividualcircumstances[15].
 In particular, in [16], the relationship between source expertise and knowledge of the do-
 mainhasbeenstudied. Thestudyfoundthatinthecaseofunfamiliarcontent,theexpertise
 ofthesourcecansomewhatinﬂuenceusers’perceptionsofthecredibilityoftheinformation,
 while in the case of known content (or perceived as such), there is no signiﬁcant interaction.
 2.1.1. Centralized Approaches
 Thesearemethodsbasedonevaluationscarriedoutbyscientiﬁcauthoritiesorexpert
 organizations that are able to properly assess the information and thus ensure quality
 results. One of the main approaches that fall under this type of methodology is the one in
 whichdomainexpertsmanuallyevaluateWebpages(andpossiblysocialcontent)and,if
 somequality criteria are met, issue “badges” that prove the informational goodness of the
 content [17].
 Alternatively, health-related content can be placed in a central collection, where users
 canbeinformedsafely. This provides a knowledge base that can be employed for inferring
 newinformation,bymodelinggenuineinformationknownbymeansofaknowledgegraph
 against which the truthfulness of new information, possibly expressed in RDF triples, can
 be evaluated. In this way it would be a centralized and pseudo-automated method, as,
 for example, the one illustrated in [18] in the context of general misinformation detection.
 The advantage of such methods is that the results in terms of accuracy, quality, and,
 in general, genuineness of information are very high; however, relying on a manual process,
 Int. J. Environ. Res. Public Health 2022, 19, 2173 4of20
 they are particularly expensive and time-consuming. In addition, they do not scale with
 respect to the speed at which content grows and even how it is updated.
 2.1.2. Distributed Approaches
 These are methods based on the use of tools designed to allow any user to be able
 to distinguish information from misinformation even without prior medical knowledge.
 Amongthevarious tools proposed is DISCERN [19], a system designed by the English
 national health system, whose accuracy has been tested on several occasions [20]. The
 advantageofusingthisapproachisthatitallowsanyonetoindependentlyassessmedical
 information regarding health treatments, based on “objective” criteria. These criteria are
 illustrated in Figure 1.
 Figure 1. The DISCERN questionnaire. Graphic elaboration of the 16 questions extracted from
 http://www.discern.org.uk/discern_instrument.php (accessed on 3 February 2022).
 Disadvantages include the reluctance of users to carve out time to proceed with ﬁlling
 out the questionnaire, the fact that the process must be iterated over every piece of health-
 related content, and the low uptake of the method among the population. It should also be
 notedthat this tool is designed for rather long texts that characterize Web pages; it has not
 beentested in the context of health misinformation detection in social media content.
 2.2. Automated Approaches
 Thesearemethodsthataimtoassessthegenuinenessofhealthinformationautomati-
 cally, without the direct intervention of users. A number of solutions have been proposed
 in recent years for misinformation detection in “general” online content for different do-
 mainsandtasks,rangingfromopinionspamtofakenewsdetection[10,21–23]. However, only
 recently studies have been carried out in the health domain, also due to the growing impact
 of Consumer Health Search (CHS) [24,25].
 Theapproachproposedin[17],consideringWebcontentintheformofarticlesabout
 health treatments, therapies, etc., aims to automate (a reduced version of) DISCERN. A
 Hierarchical Encoder Attention-based model is implemented, which exploits BERT [26] and
 BioBERT[27]toassignscorestoWebpageswithrespecttothecompletenessandaccuracyof
 ﬁveaspectsformtheabbreviatedDISCERN.Themodel,despiteitslimitationsrelatedto
 the speciﬁcity of the medical information considered, obtains high performance, also by
 virtue of exploiting the semantics of the texts.
 Int. J. Environ. Res. Public Health 2022, 19, 2173 5of20
 Anotherapproachfocusingoncontentanalysishasbeenproposedin[28]. Inaﬁrst
 method,twogroupsofmedicalWebpagesareselected,i.e.,highly-credible and non-credible.
 ThroughtheTAGMEapplication[29],themostrepresentativetopicsareextractedfromthe
 twogroupsofpages. Toevaluatethecredibility of a new topic extracted from an unjudged
 Web page, the Jaccard similarity between the topic just extracted and those belonging
 to the two reference groups is computed. The method returns good values, but it has
 several limitations including dependence on the choice of the two reference groups. In a
 second proposed method a Markov model is trained using textual features, considered
 discriminating in identifying health misinformation. These include the use of business
 terms, certain speciﬁc punctuation elements, and certain parts of speech. The results
 obtained showthattheidentiﬁedfeatures are suitable for modeling the problem at hand.
 Recently, the graph-based DETERRENTmodel[30]hasbeenproposedtodetectmisin-
 formation in online health-related articles by leveraging on additional information from
 a Medical Knowledge Graph and on a propagation network built on such graph and on
 anArticle-Entity Bipartite Graph. Empirical validation using two disease-speciﬁc datasets,
 i.e., related to diabetes and cancer, demonstrates the effectiveness of the model. Another
 recent solution, proposed in [8], aims at considering both the structure and the content of
 Webpagestoperformabinaryclassiﬁcationbetweenhealthinformationandmisinforma-
 tion, by adding domain-speciﬁcity to the Web2Vec model developed for phishing Web page
 detection [31]. Features are extracted by means of a CNN-BiLSTM network; in particular,
 aWord2Veclayerispre-trainedonPubMedforcontent-basedfeatures,andanattention
 mechanismisappliedtofeatureembeddingrepresentations. The results show, on three
 publicly available datasets, the effectiveness of considering health domain characteristics
 in the proposed approach.
 Oneoftheﬁrststudiesexplicitly addressing health misinformation in social media is
 the one proposed in [32]. The study presents a Markov Random Field (MRF) model, which
 jointly considers the reliability of the user (based on the engagement within the community
 and other information such as socio-demographic factors), of the provided statements
 (with regard to the medicines to be taken and the potential side effects, as reported in
 the Mayo Clinic dataset [33]), and their linguistic objectivity (identiﬁed based on the extrac-
 tion of linguistic-stylistic but also linguistic-emotional features). The results obtained on
 15,000 users and their 2,800,000 posts about the side effects of drugs and health products
 fromhealthboards [34] (one of the largest health-focused online communities) reach a good
 accuracy, even if, also in this case, the method is very disease-speciﬁc. Another disease-
 speciﬁc work examines social media users promoting cancer treatments that have been
 showntobeineffective[35]. Theauthorsstudyaround4000Twitteruserspostingabout
 such“treatments”, and compare them to a baseline of users generally interested in cancer,
 byconsidering features capturing user attributes, writing style, and sentiment. Such misin-
 formation detection strategy presents an accuracy of over 90% for the considered disease.
 In the last couple of years, some work based on the use of multiple features in con-
 junction with machine learning approaches has been proposed for health misinformation
 detection in social media. In [36], the authors describe (in addition to various baselines
 considered) an approach based on a traditional statistical learner such as SVM, in asso-
 ciation with the Linguistic Inquiry and Word Count (LIWC) psycho-linguistic lexicon [37],
 from which 93 textual features are extracted. Other content-based approaches are illus-
 trated in [36,38,39], where Convolutional Neural Networks (CNNs) are employed to classify
 health information with respect to its genuineness based on textual features. In the study
 presented in [40], the authors incorporate in supervised classiﬁcation models central-level
 features (including topic features) and peripheral-level features (including linguistic, senti-
 ment,anduserbehavioralfeatures). The considered dataset (around 5000 items) has been
 collected from a real online health community focusing on autism. The results show that,
 in this case, behavioral features are more informative than linguistic features in detecting
 healthmisinformation. Anotherrecentworkaimstoexplorevariousfeaturesotherthantex-
 tual to identify health misinformation in social media [41]. Empirical data including around
 1200pieces of health information have been collected and manually labeled. The ﬁndings
 Int. J. Environ. Res. Public Health 2022, 19, 2173 6of20
 showthattherearesigniﬁcantdifferences in the features of health misinformation between
 different topics.
 In this article, as illustrated in the next section, we propose the use of supervised
 learning in association with a large number of different features, not all of which were
 considered together in previous works, for health misinformation detection. To do this,
 wetesttheeffectiveness of such features on three publicly available datasets with a higher
 number of observations than those used so far in the literature, also to consider at the
 sametimedifferenttypesofcontent(bothWebnews/reviewsandsocialmediacontent)
 anddifferent types of health-related topics, to avoid disease-speciﬁcity. Finally, we also
 consider other baselines that have been used in the literature for “general” misinformation
 detection. For example, Hierarchical Propagation Networks (HPNs) proposed in [42] for the
 identiﬁcation of fake news, and Bidirectional Long-Short Term Memory (Bi-LSTM) networks
 employedfortheidentiﬁcationofbothfakenewsin[43]andrumorsin[44],andalsointhe
 health domain for the identiﬁcation of fake content related to COVID-19 [45].
 3. Materials and Methods
 This section initially presents the datasets that were considered in this paper and
 the health misinformation features extracted from such datasets, and then describes several
 classiﬁers used and tested to discern between information and misinformation.
 3.1. Datasets
 Following extensive research against resources available on the Web and in the litera-
 ture, a low number of publicly available datasets containing health information labeled
 with respect to its genuineness were found. Among the best we have identiﬁed include the
 CoAIDdataset[38],theReCOVerydataset[36],andtheFakeHealthdataset[39]. WhileCoAID
 andReCOVerycontain“informal”health-relatedcontentabouttheCOVID-19pandemic,
 FakeHealth contains expert-generated health information on various medical topics.
 3.1.1. CoAID
 It is a collection of news and claims written in English, having the COVID-19 pandemic
 as the main subject of discussion. In detail, by “news” the authors mean classic articles
 abouteventsand/orpublic-interest information, while by “claim” they mean short texts
 that are generally composed of a maximum of two sentences. Depending on the type of
 information considered, authors performed a different data collection ﬂow.
 • News: are considered as “credible” news those extracted from nine highly-reliable
 sources including public health institutions or authorities, such as: Healthline [46], Sci-
 enceDaily[47],theNationalInstitutesofHealth(NIH)[48],MedicalNewsToday(MNT) [49],
 the Mayo Clinic [50], the Cleveland Clinic [51], WebMD [52], the World Health Organiza-
 tion (WHO)[53], and the Centers for Disease Control and Prevention (CDC) [54]. As for
 “not credible” news, the authors include those reported by various sites involved in
 fact-checking (e.g., WHO and MNT) as false myths.
 • Claims: they are sourced from the ofﬁcial Web site and the ofﬁcial Twitter account of
 the WHOandfromtheMNTWebsite. Theseclaimsareinformationexpressedinthe
 formofanswerstocommonquestionsorfalsemythsrelatedtoCOVID-19.
 Foreachpieceofcollectedcontent,whetherintheformofnewsorclaims,thedatasetis
 providedwiththeIDsofthetweetsthattheauthorscollectedthatrefertothem. Withrespect
 to such tweets, the IDs of the corresponding replies are also provided.
 3.1.2. ReCOVery
 It is a collection of news items written in English related to COVID-19, collected from
 around 60 different news Web sites. The “reliability” of news items is assessed by an
 automated approach that does not involve the use of domain experts. Speciﬁcally, it is
 derived based on the trustworthiness of its source. To assess such trustworthiness, both
 NewsGuard[55], and Media Bias/Fact Check (MBFC) [56], two popular fact-checking sites,
 Int. J. Environ. Res. Public Health 2022, 19, 2173 7of20
 are employed. Both sites produce reliability ratings: a news item is labeled as reliable if it
 exceeds certain thresholds for both sites, as unreliable otherwise. As in the cases of CoAID,
 also for ReCOVery the IDs of the tweets that refer to the news items taken into consideration
 are provided, but the IDs of the corresponding retweets and replies are not made available.
 3.1.3. FakeHealth
 It is a collection of reviews generated by experts regarding health-related content
 aboutmedicalinterventions, wellness, therapies, etc., written in English. Such reviews are
 publishedontheHealthNewsReviewWebsite[57],whichistheWebinterfaceofaproject
 active from 2005 to 2018 and supported by the Informed Medical Decisions Foundation [58].
 This project involved at least two independent reviewers with years of experience to
 evaluate, according to 10 criteria, the “reliability” of the more than 2500 total health-related
 contents analyzed. As for CoAID, also in FakeHealth the various contents are distinguished
 accordingtothetypeofsource: wespeakofStoryifthesourceisamedia,orofReleaseifthe
 source is a health institution, a university, etc. The reliability assessment is performed by
 considering10differentcriteriadetailedin[39],concerningforexamplethecompletenessof
 informationabouttreatments,thepresenceofdisease-mongering,theusageofindependent
 sources, etc. Additionally, in this case, for each piece of collected content, the dataset is
 providedwiththeIDsofthetweetsrelatedtoit. Withrespecttosuchtweets,theIDsofthe
 corresponding retweets and replies are also provided.
 3.1.4. Data Gathering and Cleaning
 Table 1 summarizes the numerosity of the data made available in each dataset, with re-
 spect to their nature.
 Table1. Dimensionality of the original datasets.
 Data CoAID ReCOVery FakeHealth(Release) FakeHealth(Story)
 Textual contents 3555 2029 606 1690
 TweetIDs 151,964 140,820 47,338 384,073
 RetweetIDs - - 16,959 92,758
 ReplyIDs 122,150 - 1575 20,644
 Toextract the features related to the propagation network (see Section 3.2.5) and user
 proﬁles (see Section 3.2.6), it was necessary to proceed with a further data gathering phase,
 bymeansoftheTwitterAPIstartingfromtheavailabletweetIDs. Hence,foreachtweet
 ID, the following data has been gathered: message text, date and time of the message, number
 of retweets, number of favorites, user ID (i.e., the author of the tweet), and the associated user
 proﬁle description, number of followers, and number of followees. The same data were gathered
 for retweets and replies associated with the considered tweet ID. Because the data in the
 CoAID,ReCOVery,andFakeHealthdatasetswerecollectedsometimepriortothiswork,it
 occurredthatnotalloftheoriginaltweetIDswerestillaccessibleduringthedatagathering
 phase. This is because, in the meantime, messages have been removed from the platform
 as not compliant with Twitter policies, some users have deleted their proﬁles, or have
 beensuspended. It was thus possible to verify that the number of inaccessible tweet IDs is
 signiﬁcant for each dataset. At a general level, FakeHealth (Release) and FakeHealth (Story)
 are those incurring in the lower reduction compared to CoAID and ReCOVery. Furthermore,
 the health-related content collected within the FakeHealth (Story) dataset is, among the
 considered datasets, the one that has generated more activity on Twitter.
 Following the data gathering phase, a data cleaning phase was also carried out to
 eliminate duplicates and remove HTML codes and undocumentedcharactersinthetexts
 andlinks pointing to images, which are not considered in this work. The numerosity of the
 data used in this work following the data gathering and cleaning phases on the considered
 datasets is shown in Table 2.
 Int. J. Environ. Res. Public Health 2022, 19, 2173 8of20
 Table2. Dimensionality of the gathered and cleaned datasets.
 Data CoAID ReCOVery FakeHealth(Release) FakeHealth(Story)
 Textual contents 1820 1910 594 1498
 TweetIDs 74,722 42,153 44,547 315,709
 RetweetIDs 65,464 43,024 16,070 99,971
 Replies IDs 29,969 - 1253 14,472
 UserIDs 164,891 58,495 28,893 206,798
 Further details about the employed datasets, together with the code developed to
 gather and clean data, to extract the considered features, and to implement the proposed
 classiﬁers, as well as further documentation about the extracted features, are publicly
 accessible at the following link: https://github.com/ikr3-lab/health-misinformation/
 (accessed on 3 February 2022).
 3.2. Health Misinformation Features
 Based on the literature and on a classiﬁcation work performed in this article, we
 identify six classes of health misinformation features: (i) textual representation features, i.e., re-
 lating to different possible formal representations of the text, (ii) linguistic-stylistic features,
 i.e., taking into account the presence of different stylistic aspects of the text, (iii) linguistic-
 emotional features, i.e., identifying aspects of emotional character that transpire from the
 text, (iv) linguistic-medical features, i.e., related to the presence of speciﬁc medical terms
 within the text, (v) propagation-network features, i.e., taking into account the social network
 andthewayinformationispropagatedonit,and(vi)user-proﬁlefeatures,i.e.,related to
 various metadata connected to user proﬁles.
 3.2.1. Textual Representation Features
 BoththeBag-of-Words(BoW)andthewordembeddingrepresentationmodelsareem-
 ployedtoextract such features, which underlie a variety of approaches to detect misinfor-
 mation, both general and health-related [10,32,59,60]. Bag-of-Words features are constituted
 bytheuniquetermsthatappearwithinthedocuments(i.e.,health-relatedtextual content).
 In the case under consideration, two different weighting schemes are considered: (i) binary
 weighting, where the presence or absence of the term within the document is indicated
 withabinaryvalueassociatedwiththeterm,and(ii)TermFrequency—InverseDocument
 Frequency (TF-IDF) weighting, where the frequency of a term within the document related to
 the number of occurrences of the term in the document collection is associated with the
 term. Word embedding features are constituted by real-valued vectors encoding the meaning
 of each word, such that words that are closest in vector space are expected to be similar
 in meaning [61]. In this work, the GloVe model has been used to perform embedding of
 words[62]. Speciﬁcally, the model was pre-trained on Wikipedia 2014 + English Gigaword
 Fifth Edition corpora [63], having sizes of 50, 100, and 200. The total number of features
 obtained by means of the above text representation models depends on the dataset under
 consideration. This information is, therefore, provided in the shared documentation.
 3.2.2. Linguistic-Stylistic Features
 These features capture the set of stylistic traits that characterize a text. They have
 already been employed in assessing the credibility of online health information, such
 as in [28,32]. In the context of generic medical-themed Web pages, the study presented
 in [28] noticed that many sentences that provide credible information are expressed in the
 passive form. With reference to health-related social media content, in [32] the authors
 have associated constructs expressing uncertainty with non-credible information. Such
 constructs are identiﬁed in strong modals, possibility adverbs (e.g., maybe, probably, etc.),
 conditional and question particles (e.g., who, what, when, etc.). Conversely, credible
 information is associated with the use of constructs that provide some degree of certainty
 suchas: weakmodals,demonstrativeadjectives/pronouns,anddeclarativeconjunctions
 Int. J. Environ. Res. Public Health 2022, 19, 2173 9of20
 (e.g., therefore). Table 3 illustrates the list of the 25 features considered in this paper,
 extracted using Part-of-Speech (PoS) tagging by means of the NLTK toolkit [64].
 Table3. List of the linguistic-stylistic features considered.
 Features Examples/Explanations
 Strong modals might, could, can, would, may
 Weakmodals should, ought, need, shall, will
 Conditionals if
 Negations no, not, neither, nor, never
 Conclusive conjunctions therefore, thus, furthermore
 Subordinating conjunctions until, despite, in spite, though
 Following conjunctions but, however, otherwise, yet
 Deﬁnitedeterminers the, this, that, those, these
 Personal pronouns I, you
 First person I, we, me, my, mine, us, our
 Secondperson you, your, yours
 Third person he, she, him, her, his, it, its
 Questionparticles why,what,when,which,who
 Adjectives correct, extreme, long, visible
 Adverbs maybe,about,probably,much
 Propernouns namesofplaces,things,etc.
 Othernouns other nouns
 Tobeform be, am, is are, was, were, been
 Tohaveform have, has, had, having
 Past tense verb past tense verb
 Gerund gerund
 Participle verb past or present participle verb
 Superlatives superlative adjectives or adverbs
 Exclamation exclamation mark
 Other other terms
 3.2.3. Linguistic-Emotional Features
 Thesefeatures concern the possible emotional aspect expressed by a text. An example
 of the usage of such features for health misinformation detection can be found in [32,40].
 Such features capture the degree of objectivity/subjectivity of a text, its polarity, and the
 emotions it conveys. An objective text is one in which the information is expressed from a
 general point of view, is supported by objective data and there is no personal or emotional
 involvementoftheauthor. In contrast, a subjective text is characterized by the expression
 of informationthroughtheauthor’sviews,beliefs,andemotions. Thepolarityofatext,cap-
 turedbysentimentanalysistechniques[65],istheexpressionthatdeterminesthesentimental
 aspect of an opinion; in the case of an objective text, it is by deﬁnition neutral, while, in the
 presence of a subjective text, it can be positive or negative, with different levels of intensity.
 The presence of polarity, especially when the intensity is high, is likely to indicate bias
 and, therefore, health information of questionable credibility [32]. Similarly, the presence
 of particular emotions, which can be extracted using emotion analysis techniques [65], can
 beassociated with health misinformation. For example, texts expressing emotions such as
 anxiety and uncertainty are a symptom of information that is not always credible, while
 texts with little emotional involvement seem to be more credible [32].
 In this work, to compute the objectivity, subjectivity, and polarity features, we used
 the algorithm proposed in the TextBlob library [66], which is based on a general-purpose
 lexicon (in the absence of an established literature on medical lexicons used in sentiment
 analysis techniques and semantic approaches pre-trained on such lexicons). To extract
 emotions, we employed two distinct solutions. The ﬁrst solution makes use of the NRC
 Emoticon Lexicon (NRC) proposed in [67], which allows identifying the percentage of terms
 in the text that express a given emotion among the following ones: anger, fear, anticipation,
 trust, surprise, sadness, joy, and disgust. The second solution is based on the text2emotion
 library [68], which provides, for each text, a score in the [0, 1] interval with respect to the
 Int. J. Environ. Res. Public Health 2022, 19, 2173 10of20
 intensity detected for each of the following emotions: happiness, anger, sadness, surprise, and
 fear. In total, for this feature class, we extracted 38 features that, again, are better detailed in
 the shared documentation.
 3.2.4. Linguistic-Medical Features
 Theyaimtocapturestatistics related to the appearance of speciﬁc medical (or health-
 related) terms that impact health information genuineness. The 4 features considered for
 this category include: normalized count of medical terms, normalized count of unique medical
 terms, hyperlink count, and normalized count of commercial terms. The idea of using such
 features for this purpose was partly inspired by the outcomes illustrated in Section 2 with
 respect to the Web content scenario, in particular concerning the count of hyperlinks and
 commercialterms. Details about these features are provided below.
 • Normalized count of medical terms: this is a count of the medical terms present in a
 given text normalized by the total number of words. Extracting this feature required
 the use of a Named-Entity Recognition (NER) model specially trained on medical
 information, namely the spaCy library [69]. Such a model is particularly suitable since
 it is trained on MedMentions [70], a collection of 4392 titles and abstracts published
 in PubMed [71], manually annotated by a team of experts. The main limitation of
 using this application lies in the fact that it is not able to recognize terms introduced
 in the medical-scientiﬁc language after the training operation (in the speciﬁc case that
 occurred in 2018). To overcome this, the output of the model has been supplemented
 withmanualextractionofalltermsrelatedtoCOVID-19andnotpresentintheoriginal
 dictionary. This list of words is the same as that used to generate the COVID-19 stream
 madeavailablebyTwitterforresearchpurposesandaccessibleatthefollowinglink:
 https://developer.twitter.com/en/docs/labs/covid19-stream/overview/ (accessed
 on3February2022).
 • Normalized count of unique medical terms: in this case, the (normalized) unique count of
 medical terms is considered. The rationale behind this choice is because we expect
 that a high number of distinct medical terms corresponds to a higher mastery of the
 speciﬁc language and therefore of the speciﬁc medical domain.
 • Hyperlinkcount: thepresenceofexternalhyperlinkscanbeassociatedwithmisinforma-
 tionwhensuchlinkspointtomisleadingand/oradvertisingcontent. Thecomputation
 of such feature was done by counting the number of hyperlinks, extracted using appro-
 priate regular expressions.
 • Normalized count of commercial terms: As illustrated in the literature [13], the higher the
 numberofcommercialterms,thelesscredibleisperceivedtherelated information,
 due to the for-proﬁt purpose of such information. At a practical level, a list of 45
 commercialtermstakenfrom[72](suchas“sale”,“deal”,“ad”, etc.) has been compiled.
 The frequency of such terms in the health-related content has been computed and
 normalizedonthebasisofthetotalnumberoftermspresentintheconsideredtext.
 3.2.5. Propagation-Network Features
 Recentworkshaveidentiﬁedsomecharacteristicpatternsthatdifferbetweeninfor-
 mationpropagationrelatedtogenuine(general)newsandthoserelatedtofake(general)
 news[42]. Making explicit reference to Twitter (i.e., the social media platform on which
 some social media features have been extracted and evaluated, as later illustrated in
 Section 4), gives the possibility to reason on different kinds of relationships, even oriented.
 Speciﬁcally, so-called macro interactions are those made explicit by the “retweeted by” rela-
 tionship, and micro interactions are represented by the “reply to” relationship. In this work,
 by referring to the features proposed in [42] and additional features, we considered the
 following four categories.
 • Structural features: variables designed to capture aspects of network structure and
 topology. They refer to characteristics such as depth, breadth, and out-degree (in this case
 representing a measure of popularity) at both the global and cascade network levels;
 Int. J. Environ. Res. Public Health 2022, 19, 2173 11of20
 • Temporal features: variables whose goal is to capture temporal aspects related to infor-
 mationdissemination. They refer to characteristics such as duration of dissemination,
 average speed of dissemination, and average speed of response at both the global network
 andcascadelevels;
 • Linguistic features: variables designed to capture the linguistic aspects of messages
 that interact with information dissemination. These variables are only applied within
 the micro-network, since at the retweet level, and to a large extent at the tweet level,
 the recorded messages are the same. This group of features relates to the sentiment
 analysis of the above texts both globally and at the cascade level. In this case, to assess
 the sentiment related to such short texts, we employed VADER [73], a lexicon and
 rule-based sentiment analysis tool that is particularly suitable for social media content;
 • Engagementfeatures: variables that assess the level of appreciation received by nodes
 expressed in the form of “favorites”.
 Giventheirhighnumerosity,the36consideredfeaturesbelongingtotheabovefour
 categories are described in detail in the documentation shared along with the data and
 code, at the link previously provided.
 3.2.6. User-Proﬁle Features
 Thisclassoffeatures(whichmayhavebeenreferredtobyadifferentnameinotherworks)
 have been widely used in assessing the genuineness of “general” information[10,21,23],
 andalsoproposedinrecentworksforhealthmisinformationdetection[40,41]. Atageneral
 level, information about user characteristics can be derived from any social media; in this
 paper, we refer to the Twitter platform. Therefore, the features considered are the number of
 followers and followees, the number of medical terms in the proﬁle description of users, and the
 active contribution given by each user in the development of the network of information
 propagation. The expected behavior on the number of followers is that higher numbers
 correspond to higher authority and consequently more genuine information diffused.
 Instead, being a followee of many proﬁles can indicate not being a bot (bots often have many
 followers but follow few other accounts) [74]. User proﬁle descriptions were analyzed
 to search within them for medical-scientiﬁc terms (again, extracted from PubMed, as in
 the case of linguistic-medical features). The hypothesis is that a greater number of such
 terms in the description corresponds to a greater knowledge of the health care domain and,
 therefore, more reliable information is expected. Regarding the individual contribution to
 the development of the propagation network, it is deﬁned as active contribution, and can
 be measured in terms of tweet posting, retweets, and replies by each user in the virtual
 community. Excessive contribution by the individual user (especially in a short period of
 time) could be associated with a malicious behavior of a bot. Again, due to their rather high
 numerosityandspeciﬁcity, the 16 user proﬁle features considered as part of the proposed
 studyaredetailed in the shared documentation.
 3.3. Health Misinformation Detection
 In total, 119 different features were considered in this work, which have been used
 within binary classiﬁers to distinguish health information from misinformation. Some
 of them have been used in the literature in the context of both “general” and health
 misinformation detection, as illustrated in Section 2, and others have been employed for
 general-purpose document classiﬁcation. In particular, a number of “classical” machine
 learning algorithms such as Gradient Boosting, Logistic Regression, Naïve Bayes, and Random
 Forests have been considered; furthermore, in order to exploit at best the features that
 rely on onerous and complex representation models such as word embedding, advanced
 algorithmic solutions have been implemented, such as those pertaining to the world of
 DeepLearning (DL), which are explained in detail in the following.
 Int. J. Environ. Res. Public Health 2022, 19, 2173 12of20
 3.3.1. Convolutional Neural Networks
 AConvolutional Neural Network (CNN) classiﬁer, whose architecture is illustrated in
 Figure 2, was implemented.
 Figure 2. Architecture of the Convolutional Neural Network.
 Asitemergesfromtheﬁgure,thenetworktakesasinputanembeddinglayercreated
 using the pre-trained vectors from GloVe. These are inserted into a Conv1D layer, whose
 high-levelpurposeistotrainanumberof128neurons(whichrepresentsthedimensionality
 of the output space). This layer has a kernel size of 5. The output of the layer is taken as
 input by a MaxPooling1D layer, whose kernel size is set, again, to 5 (the choice of such
 parametersis consistent with those that are usually employed in such architectures in the
 context of text classiﬁcation). The introduction of such a layer allows us to reduce the
 outputcomplexity. AnadditionalConv1DlayerandMaxPooling1Dlayerarethencreated
 according to the same settings in order to train features of a higher level. These layers are
 then followed by a Flatten layer whose output is processed within a dense layer consisting
 of 128 neurons. The output of such layer is concatenated to the features considered in the
 proposedapproachandﬁnallyprocessedinthedenseoutputlayerformedby2neuronsto
 generate the binary classiﬁcation. From a technical point of view, the Adam optimizer [75]
 wasemployed,whiletheconsideredlossfunctionwasthebinarycross-entropyfunction[76].
 Anothertuningoperationconcernedthechoiceofthevectorsizeforthewordembedding
 representation; in the current study, vector sizes of 50, 100, and 200 elements were chosen.
 Regardingthenumberofepochsconsidered,thecallbacks were set to save the trained model
 at the epoch in which the loss on the test data was minimized and stop the training in case
 of no improvementinthenext5epochs.
 3.3.2. Bidirectional Long-Short Term Memory
 ABidirectional Long-Short Term Memory (Bi-LSTM) classiﬁer was implemented accord-
 ing to the architecture shown in Figure 3.
 Int. J. Environ. Res. Public Health 2022, 19, 2173 13of20
 Figure 3. Architecture of the Bidirectional LSTM network.
 It consists of an initial embedding layer containing GloVe’s word representations fol-
 lowedbyaBidirectionalLSTMlayerconstitutedby128neurons. Tolimitoverﬁtting,adropout
 of 0.3 and an l2 regulator were set. The output of the aforementioned layer is concatenated to
 the other features considered in this study, and then with a dense output layer consisting of
 2neuronstogeneratethebinaryclassiﬁcation. Additionally,inthiscase,theAdamoptimizer
 wasemployed,andthelossfunctionwasthebinarycross-entropyfunction. Thesameword
 embeddingsizeswereconsidered,i.e., vectors of 50, 100, and 200 elements. Regarding the
 numberofepochs,callbacks were set as in the case of the CNN architecture.
 4. Results
 Inthissection,wepresenttheresultsobtainedbyemployingthehealthmisinformation
 features illustrated in Section 3.2 in association with the binary classiﬁers detailed above.
 In particular, we introduce evaluation metrics and technical details, the effectiveness of the
 different classiﬁers with respect to different feature conﬁgurations, and the effectiveness of
 speciﬁc feature classes in health misinformation detection.
 4.1. Evaluation Metrics and Technical Details
 In the context of evaluating the effectiveness of the considered features and machine
 learning approaches in performing binary classiﬁcation, genuine information (as labeled in
 the considered datasets) has been treated as the positive class and, conversely, not genuine
 information has been considered the negative class. As evaluation metrics, we considered
 the Area Under the ROC Curve (AUC) measure and the f-measure [77]. Stratiﬁed 5-fold cross-
 validation has been used to perform all assessments, by employing the scikit-learn library
 implementation [78]. Furthermore, to prevent issues related to possible redundancy and
 multicollinearity, the Correlation-based Feature Selection (CFS) approach proposed in [79] has
 beenappliedduringcross-validation.
 4.2. Global Evaluation Results
 First, “classical” ML algorithms, such as Gradient Boosting, Logistic Regression, Naïve
 Bayes, and Random Forests are considered. Since textual representation features are based
 onthreedifferent representation models, the following three classiﬁcation conﬁgurations
 are assessed:
 • ML(BoW-binary+all): MLalgorithmsinassociationwithtextualrepresentationfea-
 tures (Bag-of-Words with binary weights) + all other features;
 • ML(BoW-TF-IDF+all): MLalgorithmsinassociationwithtextualrepresentationfea-
 tures (Bag-of-Words with TF-IDF weights) + all other features;
 • ML(WE+all): MLalgorithmsinassociationwithtextualrepresentationfeatures(word
 embeddings)+allotherfeatures.
 AsforMLandtheotherbinaryclassifiersconsideredinthispaper,theyarefirstevalu-
 atedagainstusingbaselinearchitecturesandfeaturesasdoneintheliterature[36,38,39,42–45].
 In this case, they are denoted as:
 Int. J. Environ. Res. Public Health 2022, 19, 2173 14of20
 • Bi-LSTM(WE):BidirectionalLong-ShortTermMemoryclassiﬁerinassociationwith
 only textual representation features (word embeddings);
 • CNN(WE):ConvolutionalNeuralNetworkclassiﬁerinassociationwithonlytextual
 representation features (word embeddings);
 • HPN:HierarchicalPropagationNetworksinassociationwiththepropagation-network
 features, as proposed in [42];
 • ML(LIWC):MLalgorithmsemployedinassociationwiththeLIWCfeaturesproposed
 in [37].
 Finally, CNN and Bi-LSTM classiﬁers are tested with respect to the use of word
 embeddingfeaturestogetherwithalltheotherfeaturesconsidered:
 • CNN(WE+all): ConvolutionalNeuralNetworkclassiﬁerinassociationwithtextual
 representation features (word embeddings) + all other features;
 • Bi-LSTM(WE+all): Bidirectional Long-Short Term Memory classiﬁer in association
 withtextual representation features (word embeddings) + all other features.
 In Tables 4–6, the values obtained as a result of the global evaluation are reported for
 each considered dataset. The best AUC and f-measure scores are shown in bold. The classi-
 ﬁcation conﬁgurations are ordered according to their AUC scores. Result differences are
 statistically signiﬁcant according to the t-test employed [80]. For ML classiﬁers, only the
 best performance obtained among the four considered algorithms is shown in the tables.
 Table4. Global evaluation results for the CoAID dataset.
 Dataset Classiﬁer AUC f-Measure
 CNN(WE) 0.973 0.953
 CNN(WE+all) 0.962 0.943
 ML(WE+all) 0.925 0.914
 CoAID ML(BoW-TF-IDF+all) 0.898 0.865
 ML(BoW-binary+all) 0.892 0.863
 Bi-LSTM(WE+all) 0.849 0.859
 Bi-LSTM(WE) 0.848 0.857
 HPN 0.844 0.858
 ML(LIWC) 0.669 0.789
 Table5. Global evaluation results for the ReCOVery dataset.
 Dataset Classiﬁer AUC f-Measure
 ML(WE+all) 0.921 0.848
 ML(BoW-TF-IDF+all) 0.915 0.771
 CNN(WE) 0.913 0.850
 ReCOVery ML(BoW-binary+all) 0.903 0.709
 CNN(WE+all) 0.896 0.828
 ML(LIWC) 0.817 0.743
 Bi-LSTM(WE+all) 0.741 0.655
 Bi-LSTM(WE) 0.734 0.673
 HPN 0.716 0.694
 Int. J. Environ. Res. Public Health 2022, 19, 2173 15of20
 Table6. Global evaluation results for the FakeHealth dataset.
 Dataset Classiﬁer AUC f-Measure
 ML(BoW-TF-IDF+all) 0.693 0.653
 ML(WE+all) 0.687 0.658
 ML(BoW-binary+all) 0.675 0.641
 CNN(WE) 0.661 0.602
 FakeHealth (Release) CNN(WE+all) 0.645 0.597
 ML(LIWC) 0.608 0.598
 Bi-LSTM(WE) 0.583 0.574
 Bi-LSTM(WE+all) 0.563 0.539
 HPN 0.581 0.593
 ML(BoW-TF-IDF+all) 0.717 0.627
 CNN(WE) 0.700 0.624
 CNN(WE+all) 0.698 0.655
 ML(LIWC) 0.694 0.704
 FakeHealth (Story) ML(BoW-binary+all) 0.679 0.609
 ML(WE+all) 0.657 0.706
 Bi-LSTM(WE+all) 0.656 0.602
 Bi-LSTM(WE) 0.654 0.602
 HPN 0.563 0.660
 4.3. Feature Class Evaluation Results
 This section illustrates the effectiveness of each class of features in health misinfor-
 mation detection, except for textual representation features. In fact, as it emerges from
 Tables 4–6, and as further detailed in Section 5, textual representation features are of funda-
 mentalimportanceinallapproachesconsidered,soweaimatassessingthediscriminating
 power of the other feature classes. To do so, each distinct feature group is tested using
 the same “classical” ML algorithms, with the same default hyper-parameters of the sklearn
 library. Table 7 shows, for each dataset, the best performance obtained by each class of
 features in terms of AUCs. In the end, the best performing ML classiﬁer for all datasets
 turned out to be the one based on Random Forests. From the table, it emerges that there
 is no class of features able to return, uniquely for each dataset, a higher AUC value than
 the others.
 Table7. Evaluation of effectiveness by feature class.
 AUC CoAID ReCOVery FakeHealth(Release) FakeHealth(Story)
 Linguistic-emotional 0.624 0.708 0.576 0.630
 Linguistic-stylistic 0.601 0.774 0.625 0.532
 Linguistic-medical 0.610 0.612 0.595 0.633
 Propagation-network 0.729 0.886 0.525 0.548
 User-proﬁle 0.847 0.795 0.602 0.563
 5. Discussion
 Regarding the problem of identifying online health misinformation, through the
 features considered in this work and applied in the context of supervised machine learning,
 several considerations can be made about the results presented in the previous section.
 5.1. Global Evaluation
 Asfortheresultsoftheglobalevaluation, we can afﬁrm that, for all datasets considered,
 there is no superior classiﬁer-feature conﬁguration compared to all the others. However, we
 canstate that, at the level of the distinct datasets:
 • CoAID:theCNN(WE)andCNN(WE+all)conﬁgurationsaresuperioroneverymetric
 comparedtoallotherconﬁgurations. ML(WE+all)issuperiortoML(BoW)onboth
 AUCandf-measure;
 Int. J. Environ. Res. Public Health 2022, 19, 2173 16of20
 • ReCOVery: the ML(WE+all)conﬁgurationissuperiorintermsofAUCandf-measureto
 ML(BoW),anditiscomparabletoCNN(WE)intermsofbothmetrics;
 • FakeHealth (Release): ML conﬁgurations (with both WE and BoW representations
 together with all the other features) are superior in terms of both AUC and f-measure
 to all other conﬁgurations;
 • FakeHealth(Story): in terms of AUC,theML(BoW-TF-IDF+all),CNN(WE),CNN(WE+all),
 andML(LIWC)conﬁgurationsturnouttobenotsuperiortoeachother. ML(WE+all)
 andML(LIWC)arebothsuperiortoallotherconﬁgurationsconsideringthef-measure.
 This is the only dataset for which ML(LIWC) has proven to be effective.
 Fromtheseandotherobservationsoneachdatasetandeachconﬁgurationconsidered
 (see Tables 4–6) we can say that, in general, the inclusion of all features in the CNN classiﬁers
 does not seem to lead to improvements compared to the use of only textual representa-
 tion features (word embeddings). There is some improvement, instead, in considering all
 the features together with the Bi-LSTM classiﬁers, which, however, have always a lower
 effectiveness than using the CNN- and ML-based conﬁgurations. Such solutions are there-
 fore always—except for the FakeHealth (Story) dataset—signiﬁcantly superior to the other
 conﬁgurations. On a statistical level, on the FakeHealth (Release) dataset ML > CNN, while
 ontheCoAIDdatasetCNN>ML;ontheremainingdatasets,nostatisticallysigniﬁcant
 difference in results are observed. ML(LIWC) and HPN always have suboptimal perfor-
 mance, except in the FakeHealth (Story) dataset where ML(LIWC) is effective as the best
 classiﬁers. This aspect may be related to the peculiarity of the FakeHealth (Story) dataset of
 havinggeneratedtheabsolutelargestnumberofsocialreactions(seeTable2),whichare
 often highly polarized. In a case like this, the use of LIWC-based features, which counts
 wordsinpsychologicallymeaningfulcategories, could be particularly effective. This aspect
 deserves a deﬁnite investigation in the future.
 5.2. Feature Class Evaluation
 Regarding the evaluation of the effectiveness of different feature classes (except for
 textual representation features) per dataset in detecting online health misinformation:
 • CoAID: user-proﬁle and propagation-network features are particularly effective to
 tackle the problem under consideration; conversely, textual features such as linguistic-
 emotional, linguistic-stylistic, and linguistic-medical features are less performing;
 • ReCOVery: also, in this case, user-proﬁle and propagation-network features are those
 mostsuitable for the problem at hand, even if linguistic-stylistic features also show
 goodeffectiveness;
 • FakeHealth (Release): linguistic-stylistic features show the best performance, followed,
 respectively, by user-proﬁle and linguistic-medical features;
 • FakeHealth (Story): linguistic-medical and linguistic-emotional features are those pre-
 senting the best effectiveness.
 Theseobservationsallowustosurmisethat,forsimilartypesofhealth-relatedcontent,
 the feature classes that are most effective in detecting health misinformation are similar,
 and may depend on some intrinsic characteristics of both the topic and the style (i.e.,
 the linguistic register and vocabulary employed) of the content under consideration. In fact,
 both ReCOVeryandCoAIDaremadeupbycontentmainlyintheformofnewsandclaims
 related to COVID-19, whose genuineness is more effectively evaluated by the propagation-
 network anduser-proﬁle features. Such content is often intended for the general public, and in
 most cases uses vocabulary that is not too formal. On the contrary, FakeHealth (Release)
 and(Story) contain health-related content aimed at describing medical therapies, health
 treatments, surgeries, etc., which arguably uses a more complex linguistic register and a
 complexmedical-scientiﬁcvocabulary. Forsuchtypesofmedicalinformation,thelinguistic-
 stylistic and linguistic-medical features are the ones best suited to the problem addressed
 by this article. As also illustrated above, the FakeHealth (Story) dataset is the one that
 generated the most reactions on Twitter, which is perhaps why the linguistic-emotional
 feature category is particularly effective in this particular case.
 Int. J. Environ. Res. Public Health 2022, 19, 2173 17of20
 6. Conclusions and Future Research
 In the context of online health misinformation detection, this article carried out an
 overview and a study on how to succeed in assessing the genuineness of health-related
 content by means of supervised learning in association with suitable health misinformation
 features. Despite the fact that several studies have been carried out to tackle the same
 issue in the context of “general” misinformation detection, and some recent studies have
 focused on the health domain, in this article we have investigated and evaluated some
 theoretical andtechnologicalaspectsthatstilldeservetobeconsidered,especiallyregarding
 the speciﬁc characteristics of health information and the recent availability of some publicly
 available datasets, to avoid the disease-speciﬁcity of some recently proposed approaches.
 The obtained results show that deep learning solutions are effective when using
 word embedding features obtained from appropriate training on medical vocabulary,
 without the need to use other types of features. However, when “classical” machine
 learning classiﬁers are used, the importance of considering other types of features increases.
 In particular, it has been observed that for health-related content that uses a more informal
 language, propagation-network features and user-proﬁle features are particularly effective;
 whendealingwithmoreformalmedicalcontent,linguistic-stylistic and linguistic-medical
 features are the most suitable. In particular, when content generates a high volume of social
 reactions, linguistic-emotional features can also make an important contribution.
 In spite of these interesting ﬁndings, this work needs to be followed by further in-
 vestigations, which primarily concern the type of data taken into consideration and the
 respective labeling methods. In fact, it is essential to remember that datasets have only
 recently been developed for the analysis of the genuineness of medical information; these
 collections have been generated in a heterogeneous way, without common guidelines,
 applyingdifferent labeling processes of the information, which potentially refer to concepts
 of genuineness that are not always totally superimposable. This can have an effect on
 the quality of the results obtained in this work. Furthermore, another aspect of in-depth
 analysis concerns the effectiveness of particular classes of features with respect to both the
 useofparticular classiﬁers and with respect to speciﬁc types of content related to health.
 In particular, given the observation of the effectiveness of textual representation features
 (particularly word embedding features) in association with CNNs, we estimate that it will
 beofgreatimportancetofurthertestsemanticandpossiblycontext-awarerepresentation
 modelsforhealthmisinformation detection, such as BERT appropriately pre-trained on
 medical terms, or other transformer-based models.
 AuthorContributions: Conceptualization, S.D.S. and M.V.; methodology, S.D.S. and M.V.; software,
 S.D.S.; validation, S.D.S. and M.V.; formal analysis, S.D.S. and M.V.; investigation, S.D.S.; resources,
 S.D.S.; data curation, S.D.S.; writing—original draft preparation, S.D.S. and M.V.; writing—review
 andediting, M.V.; visualization, S.D.S. and M.V.; supervision, M.V.; project administration, S.D.S. and
 M.V.Allauthorshavereadandagreedtothepublishedversionofthemanuscript.
 Funding: This research received no external funding.
 Institutional Review Board Statement: Not applicable.
 InformedConsentStatement: Notapplicable.
 Data Availability Statement: The employed datasets, together with the code developed to ex-
 tract health misinformation features and implement the classiﬁers, and more details about the
 considered features, are publicly accessible in the GitHub repository of the Information and Knowl-
 edge Representation, Retrieval and Reasoning Laboratory (IKR3 Lab), at the following link: https:
 //github.com/ikr3-lab/health-misinformation/ (accessed on 3 February 2022).
 ConﬂictsofInterest: The authors declare no conﬂict of interest.
 Int. J. Environ. Res. Public Health 2022, 19, 2173 18of20
 References
 1. Carminati, B.; Ferrari, E.; Viviani, M. Security and trust in online social networks. Synth. Lect. Inf. Secur. Priv. Trust. 2013, 4, 1–120.
 [CrossRef]
 2. Eysenbach,G. Medicine2.0: Social networking, collaboration, participation, apomediation, and openness. J. Med. Internet Res.
 2008, 10, e22. [CrossRef] [PubMed]
 3. Chou, W.Y.S.; Oh, A.; Klein, W.M. Addressing health-related misinformation on social media. JAMA 2018, 320, 2417–2418.
 [CrossRef] [PubMed]
 4. Lederman,R.;Fan,H.;Smith,S.; Chang, S. Whocanyoutrust? Credibility assessment in online health forums. Health Policy
 Technol. 2014, 3, 13–25. [CrossRef]
 5. Metzger, M.J.; Flanagin, A.J. Credibility and trust of information in online environments: The use of cognitive heuristics. J.
 Pragmat. 2013, 59, 210–220. [CrossRef]
 6. Chinn, D. Critical health literacy: A review and critical analysis. Soc. Sci. Med. 2011, 73, 60–67. [CrossRef]
 7. Kickbusch, I.S. Health literacy: Addressing the health and education divide. Health Promot. Int. 2001, 16, 289–297. [CrossRef]
 8. Upadhyay, R.; Pasi, G.; Viviani, M. Health Misinformation Detection in Web Content: A Structural-, Content-based, and
 Context-aware Approach based on Web2Vec. In Proceedings of the Conference on Information Technology for Social Good,
 Rome,Italy, 9–11 September 2021; pp. 19–24.
 9. Wardle, C.; Derakhshan, H. Information disorder: Toward an interdisciplinary framework for research and policy making. Counc.
 Eur. 2017, 27, 1–109.
 10. Viviani, M.; Pasi, G. Credibility in social media: Opinions, news, and health information-a survey. Wiley Interdiscip. Rev. 2017,
 7, e1209. [CrossRef]
 11. Eysenbach,G. Fromintermediationtodisintermediationandapomediation: Newmodelsforconsumerstoaccessandassess
 the credibility of health information in the age of Web2. 0. In Building Sustainable Health Systems, Proceedings of the Medinfo 2007:
 Proceedings of the 12th World Congress on Health (Medical) Informatics, Brisbane, Australia, 20–24 August 2007; IOS Press: Amsterdam,
 TheNetherlands,2007;p. 162.
 12. Lemire, M.; Paré, G.; Sicotte, C.; Harvey, C. Determinants of Internet use as a preferred source of information on personal health.
 Int. J. Med. Inform. 2008, 77, 723–734. [CrossRef]
 13. Freeman,K.S.; Spyridakis, J.H. An examination of factors that affect the credibility of online health information. Tech. Commun.
 2004, 51, 239–263.
 14. Rieh, S.Y.; Belkin, N.J. Understanding judgment of information quality and cognitive authority in the WWW. In Proceedings of
 the 61st Annual Meeting of the American Society for Information Science, Pittsburgh, PA, USA, 24–29 October 1998; Volume 35,
 pp. 279–289.
 15. Sbafﬁ, L.; Rowley, J. Trust and credibility in web-based health information: A review and agenda for future research. J. Med.
 Internet Res. 2017, 19, e218. [CrossRef] [PubMed]
 16. Eastin, M.S. Credibility assessments of online health information: The effects of source expertise and knowledge of content. J.
 Comput.-Mediat. Commun. 2001, 6, JCMC643. [CrossRef]
 17. Kinkead, L.; Allam, A.; Krauthammer, M. AutoDiscern: Rating the quality of online health information with hierarchical encoder
 attention-based neural networks. BMC Med. Inform. Decis. Mak. 2020, 20, 1–13. [CrossRef]
 18. Li, Y.; Li, X.; Lei, M. CTransE: An Effective Information Credibility Evaluation Method Based on Classiﬁed Translating Embedding
 in Knowledge Graphs. In International Conference on Database and Expert Systems Applications; Springer: Berlin/Heidelberg,
 Germany,2020;pp. 287–300.
 19. DISCERN. Availableonline: http://www.discern.org.uk/ (accessed on 3 February 2022).
 20. Khazaal, Y.; Chatton, A.; Zullino, D.; Khan, R. HON label and DISCERN as content quality indicators of health-related websites.
 Psychiatr. Q. 2012, 83, 15–27. [CrossRef]
 21. Heydari, A.; ali Tavakoli, M.; Salim, N.; Heydari, Z. Detection of review spam: A survey. Expert Syst. Appl. 2015, 42, 3634–3642.
 [CrossRef]
 22. Ren, Y.; Ji, D. Learning to detect deceptive opinion spam: A survey. IEEE Access 2019, 7, 42934–42945. [CrossRef]
 23. Zhou,X.;Zafarani, R. A survey of fake news: Fundamental theories, detection methods, and opportunities. ACM Comput. Surv.
 (CSUR)2020,53,1–40. [CrossRef]
 24. Goeuriot, L.; Suominen, H.; Pasi, G.; Bassani, E.; Brew-Sam, N.; González-Sáez, G.; Kelly, L.; Mulhem, P.; Seneviratne, S.;
 GyanendraUpadhyay,R.;etal. ConsumerhealthsearchatCLEFeHealth2021. InProceedingsoftheCLEF2021EvaluationLabs
 andWorkshop: OnlineWorkingNotes. CEUR-WS,Bucharest,Romania,21–24September2021.
 25. Clarke, C.L.A.; Maistro, M.; Rizvi, S.; Smucker, M.D.; Zuccon, G. Overview of the TREC 2020 Health Misinformation Track. In
 Proceedings of the TREC 2020, Online, 16–20 November 2020.
 26. Devlin, J.; Chang, M.W.; Lee, K.; Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understand-
 ing. arXiv 2018, arXiv:1810.04805.
 27. Lee, J.; Yoon, W.; Kim, S.; Kim, D.; Kim, S.; So, C.H.; Kang, J. BioBERT: A pre-trained biomedical language representation model
 for biomedical text mining. Bioinformatics 2020, 36, 1234–1240. [CrossRef]
 28. Park, M.; Sampathkumar, H.; Luo, B.; Chen, X.W. Content-based assessment of the credibility of online healthcare information.
 In Proceedings of the 2013 IEEE International Conference on Big Data, Silicon Valley, CA, USA, 6–9 October 2013; pp. 51–58.
 Int. J. Environ. Res. Public Health 2022, 19, 2173 19of20
 29. Ferragina, P.; Scaiella, U. Tagme: On-the-ﬂy annotation of short text fragments (by wikipedia entities). In Proceedings of the
 19th ACMInternationalConferenceonInformationandKnowledgeManagement,Toronto,ON,Canada,26–30October2010;
 pp. 1625–1628.
 30. Cui, L.; Seo, H.; Tabar, M.; Ma, F.; Wang, S.; Lee, D. DETERRENT: Knowledge guided graph attention network for detecting
 healthcare misinformation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data
 Mining, Virtual Event, CA, USA, 6–10 July 2020; pp. 492–502.
 31. Feng, J.; Zou, L.; Ye, O.; Han, J. Web2Vec: Phishing Webpage Detection Method Based on Multidimensional Features Driven by
 DeepLearning. IEEEAccess2020,8,221214–221224. [CrossRef]
 32. Mukherjee, S.; Weikum, G.; Danescu-Niculescu-Mizil, C. People on drugs: Credibility of user statements in health communities.
 In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, New York, NY,
 USA,24–27August2014;pp. 65–74.
 33. Drugs and Supplements—Mayo Clinic. Available online: https://www.mayoclinic.org/drugs-supplements/ (accessed on
 3 February 2022).
 34. HealthBoardsMessageBoards. Availableonline: https://www.healthboards.com/(accessedon3February2022).
 35. Ghenai, A.; Mejova, Y. Fake cures: User-centric modeling of health misinformation in social media. In Proceedings of the
 ACMonHuman-ComputerInteraction, New York, NY, USA, November 2018; Volume 2, pp. 1–20. Available online: https:
 //dl.acm.org/doi/10.1145/3274327 (accessed on 3 February 2022).
 36. Zhou,X.;Mulay,A.;Ferrara, E.; Zafarani, R. ReCOVery: A Multimodal Repository for COVID-19 News Credibility Research. In
 ´
 Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM 20), Virtual Event,
 Ireland, 19–23 October 2020.
 37. Tausczik, Y.R.; Pennebaker, J.W. The psychological meaning of words: LIWC and computerized text analysis methods. J. Lang.
 Soc. Psychol. 2010, 29, 24–54. [CrossRef]
 38. Cui, L.; Lee, D. CoAID: COVID-19 Healthcare Misinformation Dataset. arXiv 2020, arXiv:2006.00885.
 39. Dai, E.; Sun, Y.; Wang, S. Ginger cannot cure cancer: Battling fake health news with a comprehensive data repository. In
 Proceedings of the International AAAI Conference on Web and Social Media, Seattle, WA, USA, 30 March–2 April 2020;
 Volume14,pp. 853–862.
 40. Zhao,Y.; Da, J.; Yan, J. Detecting health misinformation in online health communities: Incorporating behavioral features into
 machinelearningbasedapproaches. Inf. Process. Manag. 2021, 58, 102390. [CrossRef]
 41. Zhang,S.; Ma, F.; Liu, Y.; Pian, W. Identifying features of health misinformation on social media sites: An exploratory analysis.
 Libr. Hi Tech 2021. [CrossRef]
 42. Shu, K.; Mahudeswaran, D.; Wang, S.; Liu, H. Hierarchical propagation networks for fake news detection: Investigation and
 exploitation. In Proceedings of the International AAAI Conference on Web and Social Media, Atlanta, GA, USA, 8–11 June 2020;
 Volume14,pp. 626–637.
 43. Bahad,P.; Saxena, P.; Kamal, R. Fake News Detection using Bi-directional LSTM-Recurrent Neural Network. Procedia Comput. Sci.
 2019, 165, 74–82. [CrossRef]
 44. Asghar, M.Z.; Habib, A.; Habib, A.; Khan, A.; Ali, R.; Khattak, A. Exploring deep neural networks for rumor detection. J. Ambient.
 Intell. Humaniz. Comput. 2021, 12, 4315–4333. [CrossRef]
 45. Wani, A.; Joshi, I.; Khandve, S.; Wagh, V.; Joshi, R. Evaluating Deep Learning Approaches for Covid19 Fake News Detection.
 arXiv 2021, arXiv:2101.04012.
 46. Healthline: Medical Information and Health Advice You Can Trust. Available online: https://www.healthline.com (accessed on
 3 February 2022).
 47. ScienceDaily: Your Source for the Latest Research News. Available online: https://www.sciencedaily.com/ (accessed on 3
 February2022).
 48. National Institutes of Health (NIH)|Turning Discovery Into Health. Available online: https://www.nih.gov/ (accessed on 3
 February2022).
 49. MedicalNewsToday(MNT)|MedicalandHealthInformation. Availableonline: https://www.medicalnewstoday.com/(accessed
 on3February2022).
 50. MayoClinic. Available online: https://www.mayoclinic.org (accessed on 3 February 2022).
 51. Cleveland Clinic: Every Life Deserves World Class Care. Available online: https://my.clevelandclinic.org/ (accessed on 3
 February2022).
 52. WebMD—BetterInformation. BetterHealth. Availableonline: https://www.webmd.com/(accessedon3February2022).
 53. WHO|WorldHealthOrganization. Availableonline: https://www.who.int/(accessedon3February2022).
 54. Centers for Disease Control and Prevention. Available online: https://www.cdc.gov/ (accessed on 3 February 2022).
 55. NewsGuard—FightingMisinformationwithJournalism. Availableonline: https://www.newsguardtech.com/(accessedon3
 February2022).
 56. MediaBias/FactCheck—SearchandLearntheBiasofNewsMedia. Availableonline: https://mediabiasfactcheck.com/(accessed
 on3February2022).
 57. HealthNewsReview—ImprovingYourCriticalThinkingaboutHealthCare. Availableonline: https://www.healthnewsreview.
 org/(accessed on 3 February 2022).
 Int. J. Environ. Res. Public Health 2022, 19, 2173 20of20
 58. InformedMedicalDecisionFoundation—Healthwise. Availableonline: https://www.healthwise.org/specialpages/imdf.aspx
 (accessed on 3 February 2022).
 59. Thomm,E.;Bromme,R.“Itshouldatleastseemscientiﬁc!”Textualfeaturesof“scientiﬁcness”andtheirimpactonlayassessments
 of online information. Sci. Educ. 2012, 96, 187–211. [CrossRef]
 60. Wawer, A.; Nielek, R.; Wierzbicki, A. Predicting webpage credibility using linguistic features. In Proceedings of the 23rd
 International Conference on World Wide Web, Seoul, Korea, 7–11 April 2014; pp. 1135–1140.
 61. Kusner,M.;Sun,Y.;Kolkin,N.;Weinberger,K. Fromwordembeddingstodocumentdistances. InProceedingsoftheInternational
 ConferenceonMachineLearning,PMLR,Lille,France,7–9July2015;pp. 957–966.
 62. Pennington, J.; Socher, R.; Manning, C.D. GloVe: Global Vectors for Word Representation. In Proceedings of the Empirical
 MethodsinNaturalLanguageProcessing(EMNLP),Doha,Qatar,25–29October2014;pp. 1532–1543.
 63. English GigawordFifth Edition—Linguistic Data Consortium. Available online: https://catalog.ldc.upenn.edu/LDC2011T07
 (accessed on 3 February 2022).
 64. NLTK:NaturalLanguageToolkit. Availableonline: https://www.nltk.org/(accessed on 3 February 2022).
 65. Pozzi, F.A.; Fersini, E.; Messina, E.; Liu, B. Sentiment Analysis in Social Networks; Morgan Kaufmann: Burlington, MA, USA, 2016.
 66. TextBlob: Simpliﬁed Text Processing. Available online: https://textblob.readthedocs.io/en/dev/ (accessed on 3 February 2022).
 67. Mohammad,S.M.;Turney,P.D. NRCemotionlexicon. Natl. Res. Counc. Can. 2013, 2, 1–234.
 68. text2emotion|PyPI. Available online: https://pypi.org/project/text2emotion/ (accessed on 3 February 2022).
 69. spaCy|Industrial-StrenghtNaturalLanguageProcessinginPython. Availableonline: https://spacy.io/(accessedon3February2022).
 70. Mohan,S.;Li,D. MedMentions: ALargeBiomedicalCorpusAnnotatedwithUMLSConcepts. arXiv2019,arXiv:1902.09476.
 71. PubMed. Availableonline: https://pubmed.ncbi.nlm.nih.gov/ (accessed on 3 February 2022).
 72. Fernández-Pichel, M.; Losada, D.; Pichel, J.C.; Elsweiler, D. Reliability Prediction for Health-related Content: A Replicability
 Study. In Proceedings of the European Conference on Information Retrieval, Lucca, Tuscany, Italy, 1 April 2021.
 73. Hutto, C.; Gilbert, E. VADER: A parsimonious rule-based model for sentiment analysis of social media text. In Proceedings of the
 International AAAI Conference on Web and Social Media, Ann Arbor, MI, USA, 1–4 June 2014; Volume 8.
 74. Shao, C.; Ciampaglia, G.L.; Varol, O.; Yang, K.C.; Flammini, A.; Menczer, F. The spread of low-credibility content by social bots.
 Nat. Commun. 2018, 9, 1–9. [CrossRef] [PubMed]
 75. Adam—Keras. Availableonline: https://keras.io/api/optimizers/adam/ (accessed on 3 February 2022).
 76. Probabilistic Losses—Keras. Available online: https://keras.io/api/losses/probabilistic_losses/#binary_crossentropy-function
 (accessed on 3 February 2022).
 77. Hossin, M.; Sulaiman, M. A review on evaluation metrics for data classiﬁcation evaluations. Int. J. Data Min. Knowl. Manag.
 Process 2015, 5, 1.
 78. Scikit-Learn: Machine Learning in Python. Available online: https://scikit-learn.org/stable/ (accessed on 3 February 2022).
 79. Hall, M.A.; Smith, L.A. Feature selection for machine learning: Comparing a correlation-based ﬁlter approach to the wrapper. In
 Proceedings of the FLAIRS Conference, Orlando, FL, USA, 1–5 May 1999; Volume 1999, pp. 235–239.
 80. Semenick, D. Tests and measurements: The T-test. Strength Cond. J. 1990, 12, 36–37. [CrossRef]

# Ginger Cannot Cure Cancer- Battling Fake Health News with a Comprehensive Data Repository

## Source PDF

`docs/literature_pdfs/Ginger Cannot Cure Cancer- Battling Fake Health News with a Comprehensive Data Repository.pdf`

## Extraction Notes

- Converted from PDF using Ghostscript `txtwrite`.
- This is a working Markdown conversion for analysis; tables, figures, equations, and multi-column layout may need manual cleanup against the original PDF.
- The text below is extracted from the downloaded PDF, not independently rewritten.

## Extracted Text

Proceedings of the Fourteenth International AAAI Conference on Web and Social Media (ICWSM 2020)
 GingerCannotCureCancer:BattlingFake
 Health News with a Comprehensive Data Repository
 EnyanDai,YiweiSun,SuhangWang
 ThePennsylvania State University
 {emd5759, yus162, szw494}@psu.edu
 Abstract tricks that are hard to be noticed. For example, it may mis-
 lead the readers by stating the association as causation or
 Nowadays,Internet is a primary source of attaining health in- 3
 formation. Massive fake health news which is spreading over mixing up the absolute risk and relative risk , which only
 the Internet, has become a severe threat to public health. Nu- require minor modiﬁcations based on the true information.
 merous studies and research works have been done in fake Second, compared with the news in political, sports and
 news detection domain, however, few of them are designed shopping domain, social media users may be less likely to
 to cope with the challenges in health news. For instance, the makeaccurate judgements on the credibility of health news.
 development of explainable is required for fake health news Because the identiﬁcation of fake news in health care tends
 detection. To mitigate these problems, we construct a com- to require specialist knowledge. This could bring challenges
 prehensive repository, FakeHealth, which includes news con- to the fake news detection approaches based on social media
 tents with rich features, news reviews with detailed expla- user opinions. Third, explainable health fake news detection
 nations, social engagements and a user-user social network. is crucial because: (i) explanations can make social media
 Moreover, exploratory analyses are conducted to understand users trust the prediction results and stop propagating fake
 the characteristics of the datasets, analyze useful patterns and health news; and (ii) the explanations help practitioners to
 validate the quality of the datasets for health fake news detec- understandiftheoutputoftheclassiﬁersareasexpectedand
 tion. We also discuss the novel and potential future research if there’s sample bias on the training dataset. Whereas, the
 directions for the health fake news detection. existing fake news detection methods are opaque especially
 the deep learning models.
 Introduction Despite the urgent need to develop novel algorithms to
 The online health information has become an important identify fake health news, few work focused on this topic.
 source for medical advice. According to Finney Rutten et al. One of the main reasons is the lack of comprehensive fake
 (2019), 81.5% of U.S. population searchs for health or med- health news dataset. To address the problems of fake health
 ical information online, and 68.9% of the U.S. adults con- newsdetection, we build a comprehensive repository named
 FakeHealth4, which consists of two datasets, i.e., Health-
 sider the internet ﬁrst to seek health information. However, Story and HealthRelease. Each dataset includes news con-
 thehealthinformationhasbeencontaminatedbyvariousdis- tents, news reviews, social engagements and user networks.
 information (Schwitzer 2017). Furthermore, with the advent Forthenewscontents,textisprovidedalongwiththesource
 of social media platforms, the fake news pieces become eas- publishers, image links and other side information, which
 ier to access and more inﬂuential. For example, a fake health beneﬁts the development of algorithms utilizing publisher
 newspiecedebunkedbyscientists,“Gingeris10,000xmore credibility, visuals and other related information in addition
 effective at killing cancer than chemo”, generated around 1 to article norm or style for better detection performance.
 1
 million engagementsonFacebook .Theﬂoodoffalsemedi- Besides, source news social engagements are supplemented
 cal newsisthreateningthepublichealth.Forinstance,acan- with user proﬁles, user timelines and friends proﬁles. The
 cer patient mistook an online ad for an experimental cancer abundant information of users could help us develop sophis-
 treatment as medically reliable information, which resulted ticated models which pay more attention to the users that
 2
 in his death . Therefore, fake health news detection is a crit- are less likely to be misled. In the end, we obtained 500k
 ical problem which requires more attention. tweets, 29k replies, 14k retweets, 27k users proﬁles with
 Fake health news detection has several challenges. First, timelines and friend lists. What’s more, the news reviews
 sophisticated fake health news could disguise itself with the cover explanations regarding ten health news evaluation cri-
 c teria. These explanations point out the aspects that the health
 Copyright  2020, Association for the Advancement of Artiﬁcial
 Intelligence (www.aaai.org). All rights reserved.
 1https://www.nbcnews.com/news/us-news/social-media- 3https://www.healthnewsreview.org/toolkit/tips-for-
 hosted-lot-fake-health-news-year-here-s-n1107466 understanding-studies/
 2https://www.bbc.com/news/business-36189252 4https://doi.org/10.5281/zenodo.3606757
 853
 news presents poorly, which allows us to develop the ex- for Weibo, it consists of 2313 rumors which are reported by
 plainable fake health news detection approaches by recog- Sina community management center. The 2351 non-rumors
 nizing the criterion the health news fails to meet. Further- are crawled from the general threads which are not reported
 more, the background knowledge described in the reviews as rumors. Around 4 million posts from 2 million users
 could be applied to build health knowledge graph as inter- are acquired for the rumors and non-rumors in total. Face-
 pretable model. The main contributions of this paper are: bookHoax(Tacchinietal.2017)containsaround15KFace-
 • We construct and release the ﬁrst comprehensive fake book posts about science news. The labels for ”hoax” or
 health news datasets. The rich features of our repository ”non-hoax” are determined by the sources of the Facebook
 includingnewscontents,newsreviewsandsocialcontexts pages. This dataset aims to deal with individual posts in-
 make it possible to detect fake health news with various stead of news articles. FakeNewsNet (Shu et al. 2018) is a
 approaches for better performance and interpretability. data repository with news content, user engagements, user
 • Weperform some exploratory analyses on the datasets to network and spatiotemporal information for fake news stud-
 validate the quality of the datasets and to understand the ies. It contains political news which are checked by politifact
 key properties of the datasets. and gossiocop. Although these engagement-driven datasets
 are valuable for fake news detection, they mostly haven’t
 • We discuss potential and urgent research directions such cover any user proﬁles except FakeNewsNet.
 as explainable fake news detection and knowledge-based While these datasets are useful, they can’t address the
 fakenewsdetectionthatcanbeconductedonourdatasets. challenges presented in fake health news. The existing fake
 news datasets don’t cover enough health news contents,
 Related Work barely include user network information and provide no re-
 Fake news detection has attracted increasing attention and lated knowledge and explanations to the ground truth. Thus,
 many fake news detection datasets are developed and pub- we collect two datasets containing news content with rich
 licly available. Most of them only contain news contents. features, social engagements with user network and news re-
 For example, BuzzFeedNews5 specializes in political news viewsprovidinganswersforstandardhealthnewsevaluation
 published on Facebook during the 2016 U.S. Presidential criteria to address the issues in fake health news detection.
 Election. The dataset contains 1627 news articles checked Dataset Collection
 by 5 Buzzfeed journalists. They give a binary label to per In this section, we ﬁrst introduce the HealthNewsRe-
 article with the news source. LIAR (Wang 2017) has 12.8k view.org,whichprovidesnewsreviewswiththegroundtruth
 short statements with manual labels from the political fact- labels and corresponding explanations. Then we give the de-
 checkingwebsite. It is useful to ﬁnd misinformation in short tails of the data collection process.
 statements, but can’t be applied for complete news articles.
 NELA-GT-2018(Nørregaard, Horne, and Adalı 2019) pro- HealthNewsReview
 vides 714k news articles in general topics from 194 news 6
 producers. The labels are obtained from 8 assessment sites. HealthNewsReview.org is a web-based project running
 Among the 194 news sources, 40 of them are not found from 2005 to 2018. It critically analyzes the claims about
 anylabels from the assessments sites. FA-KES (Salem et al. health care interventions to improve the quality of health
 2019) is a fake news dataset around Syrian war. It consists care information. HealthNewsReview.org is free from indus-
 of 804 articles labeled as real or fake. The labels were an- try and supported by Informed Medical Decisions Founda-
 notated based on the database from Syrian Violations Docu- tion and Laura and John Arnold Foundation. It runs with-
 mentation Center with a cluster algorithm, so the reliability out accepting any advertising and funding from any entity
 of these labels may bring concerns. Apart from their own in conﬂict of interest. HealthNewsReview.org reviews news
 limitations, the common drawback of the data listed here is stories from main US media and news releases from in-
 the short of social context and the information other than the stitutes. The contents include the claims of efﬁcacy about
 text of news articles. speciﬁc treatments, tests, products or procedures. The news
 In addition to news contents, several datasets also con- pieces are assessed based on a standard rating system. Each
 tain social context such as user comments and reposts on newsis reviewed by at least two reviewers with years of ex-
 thesocialmediaplatforms.CREDBANK(MitraandGilbert perience in health domain. The reviewers are from journal-
 2015) contains about 1000 news events whose credibility ism, medicine, health services research, public health and
 labeled by Amazon mechanical Turk. There are 60 million patient, and each of them signs an industry-independent dis-
 tweets between 2015 and 2016 in the dataset, but the orig- closure agreement. The diversity and independence of the
 inal news articles of the events are not included. Ma et al. reviewers could reduce the bias of the assessments.
 (2016) collected Twitter and Weibo for their fake news DataCollection Pipeline
 detection approach. For the Twitter, they obtained 778 re-
 portedeventsfromMarch2015toDecember2015whichre- Wecollected datasets with the following four steps:
 ceived 1 million posts from 500k users. And they conﬁrmed 1. We crawled the reviews of news stories and releases
 rumorsandnon-rumorsfromwww.snopes.comaslabels.As from HealthNewsReview.org. The source news titles and
 5https//github.com/BuzzFeedNews/2016-10-facebook-fact- URLsareincludedinthecollected ﬁles.
 check/ 6https://www.healthnewsreview.org/
 854
 Table 1: Description of the features including in the dataset.
 Category Features
 NewsContents URL,Title, Key words, Text,
 Images, Tags, Authors, Date
 Rating of the News,
 GroundTruth of Rating Criteria,
 NewsReviews Explanations of the Ground
 Truth,
 Tags, Category, Title, Summary,
 Descriptions, Images, News
 Source
 Social Engage- Tweets, Replies, Retweets,
 ments
 User Network User Proﬁles, User Timelines,
 User Followings, User Followers
 long links in some tweets may be warped to short URLs by
 Twitter. Fortunately, the advanced search API will also re-
 Figure 1: Overview of the collection pipeline. turn the tweets with the short URLs redirecting to the long
 URLs. This enables we won’t miss the tweets disseminated
 in a shortened form. With these three strategies, we obtained
 2. With the source news titles and URLs, we scraped the around500,000tweetsintotal.Afterweacquiredthetweets,
 newscontents which correspond to the acquired reviews. wefurther scraped their replies and retweets. In Twitter, the
 replies could also contain replies, we took this into consider-
 3. We obtained the social engagements on Twitter by col- ation during the collection. The retweets were up to 200 per
 lecting tweets, replies and retweets about the source tweets due to the limitation of the Twitter API.
 news. User Network: After we acquired all the social engage-
 4. We further supplemented the social context with the en- ments of the news stories and news releases, we built a user
 gaged user network information. crawler to get the information of users and construct a net-
 Thecollection pipeline is shown in Figure 1. The descrip- workoftheinvolvedusers.Theinvolvedusersincludeusers
 tion of storage format and the crawling API is presented in who tweet, reply, retweet and the users who are mentioned
 the Data Format section of Appendix. Next, we introduce in these tweets. The side information of the users consists
 each crawling component in detail. of the user proﬁles and user timelines which are recent two
 NewsReviewsandContents:Thereviewswerecrawled hundredtweets. The network could be built through the user
 complyingtotherobots.txtofHealthNewsReview.org.Inor- followings list and user followers list.
 der to ensure that the obtained news have adequate social Dataset Description
 engagements on Twitter, we ﬁltered out the news reviews The collected dataset repository FakeHealth consists of two
 published earlier than December 2009, which is a time point datasets, i.e., HealthStory and HealthRelease correspond-
 that users began to actively interact with the news on Twitter. ing to news stories and news releases. News stories are re-
 Thenewscontentsarecollected with the news source URLs ported by news media such as Reuters Health, while news
 presented on the HewNewsReveiws.org. In some cases, the releases are from various institutes including universities,
 source URLsaremissingoroutofdate.Inthissituation, we
 7 research centers and companies. Due to the difference in
 checked the archive website Wayback Machine to restore sources, we group news story and news release into sepa-
 the unavailable links. rate datasets in FakeHealth. The information contained in
 Social Engagements: The social context of the source HealthStory and HealthRelease datasets can be categorized
 newswasattainedfromtheTwitter,whichisoneofthemost into four categories, i.e., news contents, news reviews, social
 popular social media platforms. Twitter provides convenient engagementsandusernetworks.Eachcategoryhasmultiple
 API to collect user engagements. We adopted Twitter’s Ad-
 vanced Search API8 to search the tweets which directly post features. We list the details in Table 1.
 the news stories and releases. In order to collect adequate On the HealthNewsReview.org, both news stories and
 tweets and restrict the noisy tweets, we searched and sup- newsreleasesareevaluatedbyexpertson10criteria.Among
 plemented the related tweets by three steps. Firstly, we used them, 8 criteria (C1-C8) are common for both datasets. The
 the titles as search queries to get tweets. Then, we extracted remaining two are specially designed for HealthStory (S9-
 the key words of the title by removing special tokens and S10) and HealthRelease (R9-R10). These criteria assess the
 commonstopwordsinthetitle. In this step, we kept at least health news in diverse aspects such as the overclaiming,
 ﬁve key words to avoid over general queries. Finally, we set missing of information, reliability of sources and conﬂict
 the long URL of the news as query to search tweets. The of interests. The contents of the criteria are listed in Table
 6 in Appendix. Furthermore, each criterion is accompanied
 7https://archive.org/web/ with the label and detailed explanations. The overall rating
 8https//twitter.com/search-advanced?lang=en score is in proportion to the number of criteria satisﬁed by
 855
 Table 2: The statistics of the datasets.
 HealthStory HealthRelease
 Total Real Fake Total Real Fake
 News 1,690 1,218 472 606 315 291
 Tweets count 384,073 289,731 94,342 47,338 25,091 22,247
 Average tweets per news 227.26 237.87 199.88 78.12 79.65 76.45
 Average tweets per user per 1.21 1.20 1.23 1.14 1.13 1.14
 news
 Replies count 27,601 20,644 6,957 1,575 685 890
 Average replies per news 16.33 16.95 14.74 2.60 2.17 3.06
 Average replies per tweets 0.072 0.071 0,074 0.033 0.027 0.040
 Retweets count 120,709 92,758 27,951 16,959 9,594 7,365
 Average retweets per news 71.43 76.16 59.22 27.99 30.46 25.31
 Average retweets per tweets 0.314 0.320 0.296 0.358 0.382 0.331
 Unique users count 241,368 195,425 70,407 30,362 18,474 15,551
 1 1 equally to the label. These two observations conﬁrm the as-
 L L
 0.8 0.8
 .42C1 .33C1 sumption. The independence between the criteria also im-
 .37.20C2 0.6 .27.06C2 0.6 plies that health news could be fake for many reasons.
 .43.20.23C3 0.4 .46.18.20C3 0.4
 .39.23.24.37C4 0.2 .31.07.07.15C4 0.2 In order to show the judgements of the criteria are valid to
 .39.18.12.16.14C5 0 .38.15.04.17.08C5 0 be learned by algorithm, we applied a linguistic-style based
 .28.11.11.10.12.14C6 −0.2 .38.17.22.23.09.14C6 −0.2 model to predict whether the health news satisﬁes the ten
 .38.19.16.23.21.13.07C7 −0.4 .32.11.07.14.03.12.11C7 −0.4 criteria. The model is a logistic regression classiﬁer with un-
 .28.13.18.09.18.06.22.04C8 −0.6 .22.05.01.04.12.05.11.08C8 −0.6 igramandpartofspeech(POS)features.Werandomlysam-
 .44.15.19.28.18.13.06.13.16S9 −0.8 .22 0 −.02.01.03.07 0 .03 0 R9 −0.8
 .51.20.19.24.20.19.16.17.17.36S10 .41.06.12.24.08.11.17.10.05 0 R10 ple 100 pieces of positive samples and negative samples for
 −1 −1 each criterion as test set. Table 3 displays the results. From
 (a) HealthStory (b) HealthRelease the table, we observe that with logistic regression an simple
 Figure 2: The correlation matrices showing the correlation features of unigram and POS, we can have relatively good
 coefﬁcients between the criteria (C1-C10, S9-S10 and R9- AUCformostofthecriteria (AUC of random guess should
 R10)andthelabels (L) on HealthStory and HealthRelease. be 0.5). This indicates the possibility of predicting why the
 news is fake based on the news content. The review crite-
 ria can potentially facilitate novel research tasks such as ex-
 the news. The rating score ranges from 0 to 5. Following plainable fake news detection and multi-aspect news rating.
 the strategy in (Shu et al. 2018), we treat news pieces whose NewsTopics
 scores lower than 3 as fake news. The statistics of the col- Afair fake news dataset should have similar topic distribu-
 lected datasets are shown in Table 2. tionsacrossthefakecategoryandrealcategory.Ifthedataset
 Exploratory Analysis doesn’t comply with it, the classiﬁer trained on the datset
 is likely to be a topic classiﬁer instead of fake news de-
 In this section, we conduct exploratory analysis to demon- tector. Thus, we adopt two approaches to verify that both
 strate the quality of the HealthStory and HealthRelease. Be- HealthStory and HealthRelease have similar topic distribu-
 sides, we compare fake and real health news in multiple di- tion across fake and real category.
 mensions, which suggests the potential features and chal- Weﬁrst visualize the most frequent words of news head-
 lenges of their applications in fake health news detection. lines withwordcloudtocomparethetopicsbetweenthefake
 Wealsoprovide baselines for future researches. newsandrealnews,whichisshowninFigure3.Weobserve
 that real news and fake news of HealthStory share most of
 NewsReviews the frequent words such as cancer, heart and alzheimer. We
 havesimilarobservationinHealthRelease,i.e.,bothrealand
 One of our major contributions is providing the reliable fake news of HealthRelease mainly focus on cancer, surgery
 ground truth labels, judgements of evaluation criteria and and therapy. This implies that the topics of true and fake
 corresponding explanations. Here, we perform the analysis newsare consistent for both datasets.
 to demonstrate the importance and quality of them. To quantitatively analyze the topics across real and fake
 Inourdatasets,thelabelofasinglecriterionisdetermined news pieces, we explore the topic distributions with latent
 by experts, and the ground truth labels to indicate the news Dirichlet allocation (LDA) (Blei, Ng, and Jordan 2003).
 pieces as real or fake are based on the number of criteria that LDAis a topic modeling algorithm, which assumes a doc-
 theysatisfy. This strategy relies on the assumption that crite- ument is generated from a mixture of topics. Topics repre-
 ria represent independent aspects to assess health news.To sented by a distribution of words and the topic distributions
 verify this assumption, we investigate the correlation coef- of the documents are learned after the training of LDA. We
 ﬁcients between the criteria and labels on both HealthStory set the number of topics in LDA as eight, with reference to
 and HealthRelease. The results are displayed in Figure 2. HealthReviewNews.org. To make the topics found by LDA
 We observe a minimal positive or even no correlation be- more interpretable, we assign a word or phase to represent
 tweenthecriteria.Wealsoﬁndthatthetencriteriacontribute each topic. The topic distribution comparisons of real and
 856
 Table 3: The AUC scores of the criteria prediction by logistic regression with unigram and POS features.
 Dataset C1 C2 C3 C4 C5 C6 C7 C8 S9 S10 R9 R10
 HealthStory 0.641 0.644 0.651 0.667 0.576 0.764 0.607 0.846 0.694 0.770 - -
 HealthRelease 0.682 0.694 0.707 0.768 0.524 0.543 0.624 0.861 - - 0.720 0.570
 WebMD
 Health Day
 The Washington Post
 Los Angeles Times
 CNN
 Reuters Health
 (a) Real HealthStory (b) Fake HealthStory NPR
 The New York Times
 The Wall Street Journal
 Associated Press
 0 1 2 3 4
 Rating
 Figure 5: The average ratings of news from different sources
 in HealthStory. The error bars denote 95% conﬁdence inter-
 (c) Real HealthRelease (d) Fake HealthRelease val of the means.
 Figure 3: The word cloud of news headline for real and fake
 newsonHealthStory and HealthRelease. NewsSources
 The news sources could serve as auxiliary information for
 fake news detection (Zhou and Zafarani 2018; Shu et al.
 2017). On some occasions, the fake news detection method
 is even simpliﬁed as website source recognition. For in-
 stance, Silverman (2016) straightforwardly treats news from
 fake news websites or hyperpartisan websites that present
 themselvesaspublishingrealnewsasfakenews.Toconﬁrm
 whether the similar trend exists among health news sources,
 we evaluate the news sources credibility with the average
 ratings of the publishers. Because in HealthRelease, the pub-
 (a) HealthStory (b) HealthRelease lishers are too diverse to have meaningful comparison, we
 only analyze the sources in HealthStory dataset. We choose
 Figure 4: The distributions of the topics for real and fake newsagencies that have more than 50 pieces news reviewed
 newsonHealthStory and HelathRelease. in our dataset. From Figure 5, we ﬁnd that the average rat-
 ings differ a lot among publishers. The 95% conﬁdence in-
 terval of the average ratings don’t overlap for many news
 publishers. For instance, Associated Press’s average rating
 is as high as 3.87, while WebMD’s average rating is 2.82.
 fake articles are shown in Figure 4. The quantitative results The rating of Associated Press’s news story is signiﬁcantly
 of LDAisinlinewiththevisualizationofwordcloud.There larger than WebMD (p<0.001, t-test). Our observation in-
 are marginal difference between real and fake news in topic dicates that news sources could be a useful feature, which is
 distributions. The accordance of real and fake news topic consistent with existing work.
 distributions ensure the fake news detection models trained
 onourdatasets are not topic classiﬁers. Social Context
 Wefurther explore the topics that differ relatively notice- In this section, we analyze the characteristics of FakeHealth
 ably in LDA analysis. It appears that a few topics are more in the dimensions of tweets, retweets, replies, user credibil-
 likely to involve fake news, which results in this difference. ity and temporal engagements patterns.
 Forinstance,despitethewidespreadofnewsaboutstemcell Tweets and Retweets: The numbers of tweets posting
 therapies, most of the them are unproven by U.S. Food and health news and its retweets indicate the social impact of
 DrugAdministration except one for blood production disor- the news piece. Here we compare the distributions of tweets
 der. This fact accounts for the unbalanced numberofthefake count and retweets count between real and fake news to
 andrealnewsaroundstemcells.Thisevidencesuggeststhat ﬁnd whether the impact of real health news is higher than
 news tags and key words may provide valuable information the fake one. The results displayed in Figure 6 (a) and (b)
 for fake news detection in health domain. show differences between fake and real news in Health-
 857
 1.0 1.0 Dataset Compound Negative Neutral Positive
 0.8 0.8 Real HealthStory 0.142 0.064 0.789 0.148
 0.6 0.6 Fake HealthStory 0.079 0.069 0.810 0.121
 CDF0.4 CDF0.4 Real HealthRelease 0.172 0.045 0.802 0.152
 real real
 0.2 fake 0.2 fake Fake HealthRelease 0.161 0.046 0.803 0.151
 0.0 0.0
 0 100 200 300 400 0 10 20 30 40 50 Table 4: Comparison of average polarity scores of replies to
 Tweets count Retweets count
 (a) HealthStory (b) HealthStory tweets posing real and fake news.
 1.0 1.0
 1.0 1.0
 0.8 0.8
 0.6 0.6 0.75 0.75
 CDF0.4 CDF0.4 0.5 0.5
 real real CDF CDF
 0.2 fake 0.2 fake real real
 0.25 0.25
 fake fake
 0.0 0.0
 0 50 100 150 200 0 20 40 60 80 0 0
 Tweets count Reweets count 0 0.25 0.5 0.75 1.0 0 0.25 0.5 0.75 1.0
 Bot-likelihood Bot-likelihood
 (c) HealthRelease (d) HealthRelease (a) HealthStory (b) HealthRelease
 Figure 6: Cumulative distribution functions of real and fake Figure 7: Cumulative distribution functions showing the dif-
 newsinHealthStory and HealthRelease. ferent bot-likelihood distributions for users involved in real
 and fake health news.
 Story. Both the tweet count and retweet count of real news
 story are larger than fake news (p<0.001, Mann-Whitney ber of replies in HealthRelease. To summarize, the observa-
 Utest). This observation in HealthStory is consistent with tion in HealthStory implies that replies might be helpful to
 some earlier work (Friggeri et al. 2014; Kumar, West, and identify fake health news.
 Leskovec 2016). Friggeri et al. (2014) show true rumors on User Credibility: Users on the social media have dif-
 Facebook will have more reshares than false rumors. Ku- ferent credibility levels based on their behaviors. Accord-
 mar, West, and Leskovec (2016) demonstrate the hoaxes on ing to Abbasi and Liu (2013), users with low credibil-
 Wikipedia are viewed less than non-hoaxes. However, there ity are responsible for part of misinformation. There have
 is no signiﬁcant difference in tweet count and retweet count been several studies which consider user credibility in fake
 between real news and fake news in HealthRelease. Actu- news detection algorithms (Gupta, Zhao, and Han 2012;
 ally, in HealthStory, the impact of fake news is also very Shu, Wang, and Liu 2019). Here, we investigate whether
 high. These observations reveal the challenge of fake news the users involved with health news show equal credibility
 detection in health domain. by estimating their bot-likelihood scores. Additionally, we
 Replies: Users express their view points towards the orig- contrast real health news with fake health news in related
 inal news by replies. The replies may oppose or support the user bot-likelihood scores to conﬁrm that the propagation of
 news. These conﬂicting voices could be viewed as crowd fake health news is more likely involved with users in low
 source assessments. Thus, user replies are often utilized credibility level. We randomly sample 2000 users who are
 in the social context-based approaches such as SAF (Shu, only involved with real news and 2000 users related with
 Mahudeswaran, and Liu 2019). Here we perform sentiment fake news in both HealthStory and HealthRelease datasets.
 analysis showing that the replies towards real news are more Thebot-likelihoodsoftheusersareobtainedbyqueryingthe
 9
 positive. We obtain the polarity scores through a state-of- user names through the BotoMeter API . BotoMeter (Davis
 the-art model for social media text sentiment analysis, i.e., et al. 2016) is a state of the art bots detection algorithm
 VADER(HuttoandGilbert 2014). The polarity scores con- basedonthefeaturesofnetwork,user,friend,temporal,con-
 sist of four parts: compound score, negative score, neutral tent and sentiment. Figure 7 shows the distributions of two
 score and positive score. Compound score is a sentiment groups of user bot-likelihood scores with cumulative distri-
 metric normalized to between -1 (most negative) and +1 bution function. Similar with the observations in (Vosoughi,
 (most positive). The negative, neutral and positive scores Roy, and Aral 2018; Shu et al. 2018), in both HealthStory
 represent the fractions of the text that fall in each category. and HealthRelease, the users who propagate fake news are
 The average polarity scores of the replies are listed in Table slightly more likely to be bots. Furthermore, it is obvious
 4. As expected, the compoundscoresofnewsinHealthStory that users involved with the health news are not in the same
 are signiﬁcantly higher than fake one (p<0.001, Mann- credibility level. It is more reasonable to consider the user
 Whitney U test). In addition, the negative scores and neu- credibility when we identify fake news based on users be-
 tral scores of real stories are signiﬁcantly less (p<0.001, haviors.
 Mann-Whitney U test). In HealthRelease, we observe the Temporal Engagement Pattern: Temporal information
 similar trend. However, the differences are not statistically
 signiﬁcant. This probably due to the relatively limited num- 9https://botometer.iuni.iu.edu/
 858
 600 600 user: @Anup Callister
 1
 400 400
 Count200 tweets Count tweets
 replies 200 replies 0
 retweets retweets user: @Damalis Faint
 0 0 1
 0 2 4 6 8 10 12 14 0 2 4 6 8 10 12 14
 Time (day) Time (day)
 (a) Real health news (b) Fake health news
 0
 Figure 8: Temporal patterns of social engagements of fake 3 5 7 9 11 13 15 17 19 21
 Time (h)
 health news and real health news. (a)
 20 s
 (b)
 (a) HealthStory (b) HealthRelease user: @Healthy Longevity
 1
 Figure9:ThedistributionsoftweetingperiodicityinHealth-
 Story and HealthRelease. Peaks are marked by arrows.
 0
 4 8 12 16 20 24 28 32 36 40
 of social engagements has been explored in a number Time (day)
 of fake news detection methods (Ma et al. 2016; Shu, (c)
 Mahudeswaran, and Liu 2019). To check whether temporal Figure 10: Three periodic patterns of bots activities in health
 patterns of health news social engagements could be utilized newspropagation. x-axis shows the time after the news pub-
 to detect fake health news, we examined two news pieces in lished, y-axis shows the number of tweets. (a) Periodically
 ourdatasets.FromFigure8,weobserveasuddenincreaseof tweeting from users showing perfect temporal correlation.
 tweets and retweets for the fake news. On the contrary, real (b) Periodically tweeting a burst of tweets. (c) Tweeting fake
 news social engagements increase steadily. Thus, the spike health news with a long period.
 in the temporal engagements pattern can be an indicator of
 fake news.
 Consideringourobservationthatuserspostingfakehealth
 news have higher bot-likelihood scores, the burst of tweet- 3. In addition to the short period, there are bots that post
 ing and retweeting may be related to the bots. To investi- the news with a long periodicity to have continuous im-
 gate how bots involve with the propagation of health news, pact. In Figure 10 (c), user @Healthy Longevity repeat-
 we present the bots activities in our datasets in two di- edly tweets the same fake news more than a month.
 mensions. First, we conduct a periodicity detection on both Thesepatternsconﬁrmthatbotsareaffectingthesocialtrend
 HealthStory and HealthRelease as suggested in (Chavoshi, of health news with various strategies. And the dissemina-
 Hamooni, and Mueen 2017). The periods are evaluated by tion of health news could last for several month, while the
 the delay of the successive same tweets from the same user. propagation of general news mostly accomplishes in 4 days
 Figure9revealsthattwopeaksoftheperioddistributionsex- (Glenski, Weninger, and Volkova 2018). These patterns and
 ist in both HealthStory and HealthRelease. One of the peak special characteristic may provide insights for the temporal
 represents very short delay around 1 second, which is also in pattern related detection models.
 agreement with the pattern shown in Figure 8 (b). Another
 peak is around several days. Second, we also ﬁnd three spe- PerformanceofFakeHealthNewsDetection
 cial periodic patterns of bots activities, which are:
 1. Several users periodically tweets the same news simulta- In this subsection, we conduct fake health news detection on
 neously. An example is shown in Figure 10 (a), the activ- the two datasets. We only include very simple baselines be-
 ities of the two accounts present perfect temporal corre- cause the purpose is not to achieve high performance, but to
 lation, which indicates they are controlled by one host. (i) verify the quality of our datasets for fake news detection;
 (ii) provide reference results for future evaluations on the
 2. Someusers periodically post a burst of tweets. As shown datasets; and (iii) show that there are much space to improve
 in Figure 10 (b), the period of the burst is around 1 hour. the performance by developing state-of-the-art methods and
 Theburst lasts round 20 seconds. incorporating various information provided by the datasets.
 859
 Table 5: Fake news detection performance. Potential Research Directions
 Dataset Model Accuracy F1 AUC Ourgoalistoprovidedatasets with detailed information for
 Randomguess 0.513 0.525 0.474 research around health news. With the massive side infor-
 Unigram 0.665 0.687 0.719 mation obtained for the news pieces, a number of potential
 Unigram+NS 0.738 0.733 0.798 applications will beneﬁt from it.
 Unigram+Tags 0.676 0.694 0.728
 HealthStory SVM 0.640 0.664 0.700 Explainable Fake Health News Detection
 RandomForest 0.720 0.735 0.790 In our dataset, every news is provided with ten binary la-
 CNN 0.742 0.730 0.814 bels to indicate whether it meets the standard health crite-
 Bi-GRU 0.691 0.716 0.772
 SAF 0.760 0.756 0.763 ria. This unique feature could facilitate the novel research of
 Randomguess 0.512 0.523 0.547 explainable fake health news detection which is able to rec-
 Unigram 0.654 0.652 0.703 ognize the poorly illustrated aspects. Since explanations are
 Unigram+NS 0.665 0.667 0.712 given for the identiﬁed fake health news, the social media
 Unigram+Tags 0.642 0.636 0.690 users will understand the reasons behind the classiﬁcation
 HealthRelease SVM 0.657 0.674 0.706 results. Thus, people are more likely to trust the model and
 RandomForest 0.651 0.629 0.726 stop sharing the recognized fake health news. Furthermore,
 CNN 0.670 0.688 0.700 it could be a tool to help users think critically towards health
 Bi-GRU 0.665 0.677 0.662
 SAF 0.810 0.802 0.809 news. As a result, the propagation of fake health news could
 be intervened. Explainable model also beneﬁts their devel-
 opers by providing the evidence of overall classiﬁcation. It
 enables the researchers to better understand the model and
 Weadopt three different types of baselines including: (i) track the problems. For instance, if the training dataset is
 linguistic-based, (ii) content-based and (iii) social context- biased or polluted with adversarial samples, developers are
 based detection methods. Unigram is a lexicon-level fea- able to recognize the hidden issues based on the analysis on
 tures for linguistic-based methods (Ott et al. 2011). Here, the explanations.
 wetrained a logistic regression classiﬁer with unigram. We Knowledge-basedFakeHealthNewsDetection
 also added news source (NS) and tags to the linguistic- Newsauthenticity assessment by journalism requires to ﬁnd
 based model to verify their potential improvements to de- the related knowledge and evaluate based on evidence in-
 tection. To give more references, SVM and random for- stead of language patterns. However, most existing ma-
 est are implemented with unigram as baselines as well. chine learning approaches contrast to this process, which
 Two content-based models, CNN and bidirectional GRU makes them less sensitive to fake news composed of cred-
 (Bi-GRU), were employed for fake health news detection ible sounding sentences. Knowledge graph for health news
 (Kim 2014; Chung et al. 2014). SAF corresponds to social could potentially address this problem. In our datasets, the
 context-based method (Shu, Mahudeswaran, and Liu 2019). newscontentsandthenewsreviewspostbackgroundknowl-
 It uses the sequence of replies and user embeddings for fake edge, detailed explanations and corrections. With the news
 news detection. LSTM is applied to deal with the temporal contents and reviews, subject-predicate-object triples could
 pattern of these sequential features (Gers, Schmidhuber, and be extracted to build knowledge graphs. We could verify or
 Cummins1999).InSAF,thesocial context features learned falsify a piece of related health news by comparing with the
 from a LSTM encoder are combined with the social context claimed values and the retrieved values from the graph. It
 features to make the ﬁnal prediction . could give explanations based on the retrieved knowledge,
 For both HealthStory and HelathRelease, we randomly which makes it a more promising direction.
 selected 70% data as training set and 30% as test set. Five- Credibility-based Fake Health News Detection
 fold cross validation is conducted to tune the hyperparame-
 ters. The results of the baselines are listed in Table 5. Over- Here, we point out that the credibility of multiple items
 all, the three types of baselines have relatively good perfor- in our datasets is valuable for fake health news detection.
 mancecomparedwithrandomguess.Thisdemonstratesthat First, some topics are much more likely to be false in health
 ourdatasetsarevaluableforfakenewsdetection.Ourresults domain. For instance, only one stem-cell based therapy is
 of unigram+NS, unigram +Tags and SAF indicate that uti- proven by U.S. Food and Drug Administration until 2019,
 lizing more features in our datasets is helpful. But perfor- butitisevidentthatalargenumberofunproventherapiesare
 mance of these simple baselines is still limited and shows a 10
 recommended to clinics . This is line with the observation
 great potential of improvements. Thus, more explorations of in Figure 4, which shows the news about stems cells more
 novel fake health news detection are required. likely to be fake. Collected tags and key words can meet
 In summary, the performance of the baselines validates the need of adding topic as features for prediction. Second,
 the quality of labels and collected features in our datasets. It our analysis presented in 5 shows quality of health news is
 also suggests the demand of further investigation about the 10https://www.fda.gov/consumers/consumer-updates/fda-
 features and the potential of developing novel algorithms. warns-about-stem-cell-therapies
 860
 highly related to the news sources, which implies the cred- andcomprehensiveexplanationstothegroundtruth.Wefur-
 ibility of news sources should be useful for spotting fake ther conduct the exploratory analyses to show the the char-
 health news. Third, social bots could be identiﬁed based on acteristics of the datasets. Through our analyses, we ﬁnd the
 their tweets, proﬁles and friends (Davis et al. 2016), which potential useful patterns and challenges in fake health news
 suggests that these features contain the necessary informa- detection. Different types of fake news detection approaches
 tion to evaluate the user credibility. And our datasets supply are also evaluated on FakeHealth. The results provide base-
 these features of users involved with health news. Thus, al- lines for further studies and demonstrate the quality of our
 gorithms that implicitly consider user credibility by adding datasets. The abundant information in FakeHealth make var-
 user side information could be developed. ious novel and potential research directions possible.
 Multi-ModalFakeNewsDetection Acknowledgements
 Multi-modal fake news detection combines texts and visu- Thismaterialisbaseduponworksupportedby,orinpartby,
 als to distinguish fake news. It is suggested that the rich vi- the National Science Foundation (NSF) under grant number
 sual information is helpful for fake news detection (Jin et #IIS-1909702.
 al. 2017). We collected image URLs of the news article and
 tweets, which could be used to develop multi-modal meth- Appendix
 ods which aims to directly ﬁnd clues in the visuals. More-
 over, the user proﬁle images could also reveal the credibility Number Criteria Questions
 of the user’s posts (Morris et al. 2012). Therefore, we also C1 Doesitcomparethenewapproachwithexisting
 collect the user proﬁles with a large number of proﬁle image alternatives?
 links. Future multi-modal models may take advantage of the C2 Doesitadequatelyexplain/quantifytheharmsof
 images of news, tweets and users we obtained. the intervention?
 C3 Does it seem to grasp the quality of the evi-
 dence?
 FakeHealthNewsEarlyDetection C4 Does it adequately quantify the beneﬁts of the
 treatment/test/product/procedure?
 The aim of fake health news early detection is to identify C5 Does it establish the true novelty of the ap-
 the fake news before it has been widely spread to users. proach?
 One of the challenges is the limited social engagements C6 Does it establish the availability of the treat-
 ment/test/product/procedure?
 at the early stage of fake news propagation. Thus, obtain- C7 Doesit commit disease-mongering?
 ing various types of social engagements including tweets, C8 Doesitadequately discuss the costs of the inter-
 retweets, replies is necessary. Besides, the user characteris- vention?
 tics can play an important role for the early detection. How- S9 Does the story use independent sources and
 identify conﬂicts of interest?
 ever, manyoftheattemptsofutilizinguserinformationoften S10 Doesthestoryappeartorelysolelyorlargelyon
 simply make use of the proﬁles or the engagements with the a news release?
 news. In our datasets, every user interacting with the health R9 Does the news release identify funding sources
 news is supplied with his own proﬁle, recent timelines and &disclose conﬂicts of interest?
 R10 Doesthenewsreleaseincludeunjustiﬁable,sen-
 friend proﬁles. Overall, our datasets enable potential investi- sational language, including in the quotes of re-
 gators extract more effective features of the users and social searchers?
 engagements for the fake health news early detection. Table 6: The criteria in the news reviews for news story and
 FakeHealthNewsPropagation news release. C1-C8 denote the eight common criteria. S9
 In our analysis, we ﬁnd the propagation of real and fake and S10 are criteria for the news story. R9 and R10 repre-
 health newsiscomplex.Normalusersandbotsallcontribute sents two special criteria for the news release.
 to the dissemination of both fake and real news on social
 media. The impact of health news on the social media plat- DataFormat
 form differs a lot. With the tweets, retweets and replies to FakeHealth consists of four categories of information, i.e.,
 the original news in the datasets, we can better understand news contents, news reviews, social engagements and user
 the propagation of health news across health topics. What’s network. They are stored in four folders.
 more, the difference of spreading real and fake news could • contents has two subfolders: HealthStory and HealthRe-
 be explored. We could also discover the fake health news lease. Each folder lists the josn ﬁles of the news contents,
 with high impact in the propagation analysis to ﬁgure out which is named as the news ID.
 the most harmful fake health news.
 • reviews includes two json ﬁles: HealthStory.json and
 Conclusion HealthRelease.json. These json ﬁles contain the news re-
 In this paper, we release a comprehensive data repository views for both datasets.
 FakeHealth, which containing two feature-rich health news • engagements includes two folders: HealthStory and
 datasets, to facilitate the research in fake health news do- HealthRelease. Tweets, retweets and replies are listed in
 main. The data repository contains plenty news contents, separate folders here. Each engagement is stored in a json
 massivesocial engagements, large user-user social networks ﬁle in the name of its ID.
 861
 • user network includes four folders to store user proﬁles, Kim, Y. 2014. Convolutional neural networks for sentence classi-
 user timelines, user followers and user followings. All the ﬁcation. arXiv preprint arXiv:1408.5882.
 users in FakeHealth which includes HealthRelease and Kumar,S.;West,R.;andLeskovec,J. 2016. Disinformationonthe
 HealthStory are covered here. The proﬁles, timelines, fol- web:Impact,characteristics, and detection of wikipedia hoaxes. In
 lowerproﬁlesandfollowingproﬁlesaresavedasjsonﬁles Proceedings of the 25th international conference on World Wide
 namedintheIDofthecorresponding user. Web, 591–602. International World Wide Web Conferences Steer-
 Duetothe Twitter policy of protecting user privacy, the full ing Committee.
 contents of user social engagements and network are not al- Ma,J.;Gao,W.;Mitra,P.;Kwon,S.;Jansen,B.J.;Wong,K.-F.;and
 lowed to directly publish. Instead, we store the IDs of all Cha, M. 2016. Detecting rumors from microblogs with recurrent
 social engagements of HealthStory and HealthRelease into neural networks. In Ijcai, 3818–3824.
 two json ﬁles. More specially, each json ﬁle contains a dic- Mitra, T., and Gilbert, E. 2015. Credbank: A large-scale social
 tionary whose key is the the news ID and the value contains media corpus with associated credibility annotations. In Ninth In-
 the IDs of tweets, replies and retweets. With these two ﬁles ternational AAAI Conference on Web and Social Media.
 and the API we provided for FakeHealth, researchers can Morris, M. R.; Counts, S.; Roseway, A.; Hoff, A.; and Schwarz,
 trivially obtain the full contents of social engagements and J. 2012. Tweeting is believing?: understanding microblog credi-
 user networkfromTwitter.Wewillalsomaintainandupdate bility perceptions. In Proceedings of the ACM 2012 conference on
 the repository to ensure its usability. computer supported cooperative work, 441–450. ACM.
 Nørregaard, J.; Horne, B. D.; and Adalı, S. 2019. Nela-gt-2018: A
 References large multi-labelled news dataset for the study of misinformation in
 newsarticles. InProceedingsoftheInternationalAAAIConference
 Abbasi, M.-A., and Liu, H. 2013. Measuring user credibility in onWebandSocialMedia,volume13,630–638.
 social media. In International Conference on Social Computing, Ott, M.; Choi, Y.; Cardie, C.; and Hancock, J. T. 2011. Finding
 Behavioral-CulturalModeling,andPrediction,441–448. Springer. deceptive opinion spam by any stretch of the imagination. In Pro-
 Blei, D. M.; Ng, A. Y.; and Jordan, M. I. 2003. Latent dirichlet al- ceedings of the 49th annual meeting of the association for compu-
 location. Journal of machine Learning research 3(Jan):993–1022. tational linguistics: Human language technologies-volume 1, 309–
 Chavoshi, N.; Hamooni, H.; and Mueen, A. 2017. Temporal pat- 319. Association for Computational Linguistics.
 terns in bot activities. In Proceedings of the 26th International Salem, F. K. A.; Al Feel, R.; Elbassuoni, S.; Jaber, M.; and Farah,
 Conference on World Wide Web Companion, 1601–1606. Interna- M. 2019. Fa-kes: A fake news dataset around the syrian war. In
 tional World Wide Web Conferences Steering Committee. Proceedings of the International AAAI Conference on Web and So-
 Chung, J.; Gulcehre, C.; Cho, K.; and Bengio, Y. 2014. Empirical cial Media, volume 13, 573–582.
 evaluation of gated recurrent neural networks on sequence model- Schwitzer, G. 2017. Pollution of health news.
 ing. arXiv preprint arXiv:1412.3555. Shu, K.; Sliva, A.; Wang, S.; Tang, J.; and Liu, H. 2017. Fake
 Davis, C. A.; Varol, O.; Ferrara, E.; Flammini, A.; and Menczer, F. news detection on social media: A data mining perspective. ACM
 2016. Botornot: A system to evaluate social bots. In Proceedings SIGKDDExplorationsNewsletter 19(1):22–36.
 of the 25th International Conference Companion on World Wide Shu, K.; Mahudeswaran, D.; Wang, S.; Lee, D.; and Liu, H. 2018.
 Web, 273–274. International World Wide Web Conferences Steer- Fakenewsnet: A data repository with news content, social context
 ing Committee. and dynamic information for studying fake news on social media.
 Finney Rutten, L. J.; Blake, K. D.; Greenberg-Worisek, A. J.; arXiv preprint arXiv:1809.01286.
 Allen, S. V.; Moser, R. P.; and Hesse, B. W. 2019. Online health Shu, K.; Mahudeswaran, D.; and Liu, H. 2019. Fakenewstracker:
 informationseekingamongusadults:Measuringprogresstowarda a tool for fake news collection, detection, and visualization. Com-
 healthy people 2020 objective. Public Health Reports 134(6):617– putational and Mathematical Organization Theory 25(1):60–71.
 625.
 Friggeri, A.; Adamic, L.; Eckles, D.; and Cheng, J. 2014. Rumor Shu, K.; Wang, S.; and Liu, H. 2019. Beyond news contents:
 cascades. In Eighth International AAAI Conference on Weblogs The role of social context for fake news detection. In Proceedings
 and Social Media. of the Twelfth ACM International Conference on Web Search and
 Gers, F. A.; Schmidhuber, J.; and Cummins, F. 1999. Learning to Data Mining, 312–320. ACM.
 forget: Continual prediction with lstm. Silverman, C. 2016. This analysis shows how viral fake election
 Glenski, M.; Weninger, T.; and Volkova, S. 2018. Propagation newsstories outperformed real news on facebook. BuzzFeed News
 from deceptive news sources who shares, how much, how evenly, 16.
 and how quickly? IEEE Transactions on Computational Social Tacchini, E.; Ballarin, G.; Della Vedova, M. L.; Moret, S.; and
 Systems 5(4):1071–1082. de Alfaro, L. 2017. Some like it hoax: Automated fake news de-
 Gupta, M.; Zhao, P.; and Han, J. 2012. Evaluating event credi- tection in social networks. arXiv preprint arXiv:1704.07506.
 bility on twitter. In Proceedings of the 2012 SIAM International Vosoughi, S.; Roy, D.; and Aral, S. 2018. The spread of true and
 Conference on Data Mining, 153–164. SIAM. false news online. Science 359(6380):1146–1151.
 Hutto, C. J., and Gilbert, E. 2014. Vader: A parsimonious rule- Wang, W. Y. 2017. ” liar, liar pants on ﬁre”: A new benchmark
 based model for sentiment analysis of social media text. In Eighth dataset for fake news detection. arXiv preprint arXiv:1705.00648.
 international AAAI conference on weblogs and social media. Zhou, X., and Zafarani, R. 2018. Fake news: A survey of
 Jin, Z.; Cao, J.; Guo, H.; Zhang, Y.; and Luo, J. 2017. Multimodal research, detection methods, and opportunities. arXiv preprint
 fusion with recurrent neural networks for rumor detection on mi- arXiv:1812.00315.
 croblogs. In Proceedingsofthe25thACMinternationalconference
 onMultimedia, 795–816. ACM.
 862

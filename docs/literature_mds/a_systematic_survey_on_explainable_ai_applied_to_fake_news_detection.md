# A systematic survey on explainable AI applied to fake news detection

## Source PDF

`docs/literature_pdfs/A systematic survey on explainable AI applied to fake news detection.pdf`

## Extraction Notes

- Converted from PDF using Ghostscript `txtwrite`.
- This is a working Markdown conversion for analysis; tables, figures, equations, and multi-column layout may need manual cleanup against the original PDF.
- The text below is extracted from the downloaded PDF, not independently rewritten.

## Extracted Text

Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Contents lists available at ScienceDirect
 EngineeringApplicationsofArtificialIntelligence
 journal homepage: www.elsevier.com/locate/engappai
 Survey Paper
 AsystematicsurveyonexplainableAIappliedtofakenewsdetection
 ∗
 Athira A.B. , S.D. Madhu Kumar, Anu Mary Chacko
 Department of Computer Science and Engineering, National Institute of Technology Calicut, kozhikode, 680541, Kerala, India
 A R T I C L E I N F O A B S T R A C T
 Keywords: The exponential proliferation of fake news in recent years has emphasized the demand for automated fake
 Fakenewsdetection news detection. Several techniques for detecting fake news have yielded encouraging results. However, these
 XAI detection systems lack explainability i.e., providing the reason for their prediction. The critical advantage
 Machinelearning of explainability is the identification of bias and discrimination in detection algorithms. There are very few
 Explainable fake news detection surveys conducted on the area of explainable AI applied to fake news detection. All of theses surveys summarize
 Fakenewsidentification the existing methods in this area. Most of them are limited to the discussion of specific topics like datasets,
 evaluation methods, and potential future applications. In contrast, this survey looks at existing explainable
 AI methods and highlights the current state of the art in explainable fake news detection. We identify and
 enumerate a few open research problems based on our review of the existing explainable fake news detection
 techniques. We group the existing work in this area, by viewing it from four different perspectives: features
 used for the classification, explanation type, explainee type, and the metric used for explainability evaluation.
 The potential research topics in the above four groups which are unexplored so far and which need attention
 are also listed in this paper.
 1. Introduction 1.1. Defining fake news
 Online social media platforms are becoming increasingly popular The most challenging aspect of detecting fake news is defining fake
 across different generations, and these are eroding the traditionally news. There is no standard definition for fake news. According to
 authoritative voices of the news industry. In social media, people Wikipedia, ‘‘Fake news is defined as false or misleading information
 discuss and express their opinions about the news by leaving comments. presented as news, Its goal is frequently to harm a person’s or entity’s
 Recently much of the news material on social media are found inten- reputation or to profit from ad revenue’’ (Wikipedia, 2021). Zhou and
 tionally misleading. The phrase ‘‘fake news’’ is used to describe such Zafarani (2020) differentiated fake and fake-news-related phrases based
 malicious news items. The 2016 presidential election in the United on authenticity, purpose, and whether or not it was news. The authors
 States is said to have been impacted by fake news in social media. gave two different definitions (broad and narrow definitions) for fake
 Following election, this phrase has become a common term which is news. ‘‘Fake news is false news’’, according to the broad definition,
 referred on a daily basis everywhere. For many reasons, fake news Zhou and Zafarani (2020), and ‘‘fake news is intentionally false news
 identification is a technological challenge. News Contents are easily published by a news outlet’’ (Zhou and Zafarani, 2020) according to
 generated and spread through social media platforms, resulting in a the narrow definition.
 large amount of data to evaluate. The vastness of online material, In Zhang and Ghorbani (2020) the authors have divided the current
 which covers many topics, adds to the task’s difficulty. It has prompted fake news identification methods into two: practical-based approaches
 the researchers to work towards automatic fake news detection. For and research-based approaches. The division is done according to the
 detecting fake news, many researchers have suggested a range of audienceofthefakenews.Whilepractical-basedapproachesarehelpful
 for naive users, researchers or academicians follow research-based
 machinelearninganddeeplearningapproaches.Besidetheirpromising approaches.
 results, the models lack explainability (reason for the prediction made). Useoffact-checking websites is the most prevalent method of apply-
 In this paper, we provide a comprehensive overview of the research on ing the practical-based approach. Table 1 lists the various fact-checking
 explainable fake news detection to date. Finally, we recommend several websites that are currently in use and their detection technologies.
 potential research areas for assessing fake news on social media. Human resources-based methods are involved in the majority of fact-
 checking websites. However, human fact-checking is limited by time,
 ∗ Corresponding author.
 E-mail address: athira.bala.sree@gmail.com (Athira A.B.).
 https://doi.org/10.1016/j.engappai.2023.106087
 Received 20 October 2022; Received in revised form 5 February 2023; Accepted 2 March 2023
 Available online 20 March 2023
 0952-1976/© 2023 Elsevier Ltd. All rights reserved.
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Table 1 become irrelevant. The addition of auxiliary information (combining
 List of fact-checking websites that are currently in use. different features) aids in the resolution of this approach, and thus has
 Fact-checking websites Detection methods resulted in the a multi-modal fake news detection system.
 Media BiasFactCheck (MBFC Manual detection
 News) FactCheck.org (Media 1.4. Multi-modal fake news detection
 Bias, 2021)
 TruthOrFiction.com (2021) Manual detection In the case of multi-modal fake news detection, various feature
 Lead Stories (2021) Trendolizer technology combinations are examined. Recent research works developed in multi-
 and Manual detection
 FullFact.org (2021) Detection by fact checkers modal detection combines two kinds of feature sets. (1) a mix of visual
 Classify.news (2021) Machine learning algorithms and textual elements, and (2) user profile and news content elements.
 Factmata.com (2021) Artificial Intelligence
 Politifact.com (2021) Manual detection 1.4.1. Detection by combining textual and visual features
 Snopes.com (2021) Manual detection In the literature, many research works have used a combination of
 Hoax-Slayer.com (2021) Manual detection textual and visual features for detecting fake news. A brief overview
 of the feature extraction and classification technology used in these
 as a large amount of data is created daily and spread quickly. It approaches is presented here.
 also faces scalability issues. Thus, the difficulties imposed by manual A model or microblog news verification that incorporates both
 fact-checking approaches paved the way towards the research based visual and textual information was proposed in Jin et al. (2017). The
 approaches or automatic detection of fake news. authors proposed two kinds of image features. (1) statistical features
 and (2) visual features from the news. The statistical features cap-
 1.2. Automatic fake news detection ture the image distribution patterns. The visual characteristics disclose
 hidden distributions in news events. They have used seven statistical
 The majority of published research works in this area, treat fake features and five visual features. Fig. 2 show the details of the image
 newsdetection as a binary classification problem. As a result, a general features used in the model. Traditional classifiers such as support vector
 definition for detecting fake news is formulated as stated below. machine (SVM), logistic regression, K star, and random forest are used
 Given the features extracted from the news item x, the fake news for classification. A real-world multimedia data set created using news
 detection model aims to classify the news item as real or fake. F denotes events in Sina Weibo (a micro-blogging site also referred as ‘‘Chinese
 the fake news detection model. Let a represents a feature extracted Twitter’’) is used for the performance evaluation.
 ������ Research papers Singhal et al. (2019, 2020), and Shah and Kobti
 from the news item x. where i = 1,2,3, …. (2020) investigate multimodal fake news detection using deep learning
 { models. The authors of Singhal et al. (2019) presented SpotFake, a
 ������(������ ) = 0 if x is a fake news (1)
 ������ 1 otherwise multimodal framework for detecting fake news that makes use of both
 A number of surveys on the detection of fake news are available the textual and visual features of an article. For learning textual char-
 in the current literature. In the survey (Shu et al., 2019a), authors acteristics, they employed Bidirectional Encoder Representations from
 presented a review of the identification of fake news from a data mining Transformers (BERT). VGG-19 is used to learn image characteristics
 perspective. The authors have highlighted detection of fake news from (pre-trained on ImageNet). The studies were carried out by using two
 four different perspectives: incorrect knowledge in the news, the writing publicly accessible datasets, Twitter and Weibo.
 style used in the news, the news’s propagation pattern, and the credibility of SpotFake+ (Singhal et al., 2020), is an enhanced version of SpotFake
 the news source. The authors discuss the negative impact of fake news (Singhal et al., 2019). The first version of the model does not take
 and current detection algorithms in Zhang and Ghorbani (2020). The into account the context information in the text. They were unable to
 authors of Mohseni et al. (2020) presented a summary of published capture the elements of the image modality that may have sought to
 research articles in the field of fake news detection from 2016 to 2020. emphasize particular facts. SpotFake+ used transfer learning to capture
 Based on the literature review we find that, another way to classify the textual and visual components of a news article. The feature extrac-
 the problem of identifying fake news is to categorize it as single-modal tion is done using the XLNet and VGG-19. After that, the features are
 or multi-modal fake news detection, depending on how many features combinedandusedforclassification. The data set used is FakeNewsNet
 are involved. (FakeNewsNet, 2021).
 The model proposed in Shah and Kobti (2020) was separated into
 1.3. Single-modal fake news detection three primary components by the authors. The first is a sentiment
 analysis-based textual feature extractor, which uses sentiment analysis to
 The features used for fake news detection are divided into three extract meaningful content from textual data. The second component
 categories: The three categories are user features, news content features, is the visual feature extractor, which extracts visual features from the
 and social context features. In single-modal fake news detection, a single post using preprocessing techniques and a segmentation algorithm. A
 type of feature is analyzed. Fig. 1 gives an overview of the various cultural algorithm was used to extract the optimum features from the
 features used in single-modal fake news detection. User features include textual and visual feature representations. The last component is a fake
 information in the user profiles, credibility-related information, and news detector, which classifies the news into fake or real. Through the
 user behaviour patterns. The features of news content are derived implementation of Cultural algorithm for optimal feature extraction,
 from the analysis of words, sentences, and news content. These are the authors have outperformed related approaches.
 then referred to as linguistic-based and syntactic-based features. There In Tanwar and Sharma (2020) authors extracted visual character-
 is another class of news content features with focus on style-based istics using three common CNN architectures: VGG-19, ResNet50, and
 and visual-based features. The last category, social context features, InceptionV3. Bidirectional LSTM was used to extract textual character-
 is derived from the propagation pattern of the news and the inter- istics. The combination of textual and visual features was then utilized
 actions of social media users. It includes temporal, distribution, and to classify fake news
 network-based features. In Singh et al. (2021), the authors used three modalities: textual,
 Weobserved that news content features are used in the most of the visual, and a combination of text and visual features for fake news de-
 single-modal methods. Sticking only to a single kind of feature will not tection. Predictive analysis on both textual and visual features is carried
 out to determine their association with fake news. Many traditional
 ensure accuracy. When the writing style is modified, these approaches machine learning algorithms were applied to classify the news items.
 2
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Fig. 1. Categories of features used in single-modal fake news detection.
 from the users’ responses, as well as their accompanying user profiles.
 Astatus-sensitive crowd response feature extractor is implemented for
 this purpose. They have used CNN-based news classifiers for labelling
 the user responses. In order to enhance the performance of the model,
 they employed a PU Learning framework (learning from positive and
 unlabelled examples).
 Discussion
 Regardless of the modality of the features used for training, all the
 present research is pursuing AI-based solutions since deep learning (DL)
 systems have demonstrated unquestionable performance compared to
 traditional machine learning techniques. However, the black-box na-
 ture of the methods makes it hard for a naive user to grasp the
 basis for prediction. When comparing fake news detection approaches
 using deep learning models with the standard machine learning mod-
 els, their performance is hampered by their lack of explainability. So
 to avoid bias and improve performance, the detection model must
 have transparency in its decision-making process. Thus, explainable
 Fig. 2. Overview of image features used in Jin et al. (2017). AI (XAI) approaches help to detect fake news since they enable us to
 comprehend better and explain the model’s prediction.
 At present, the research on explainable fake news detection models
 1.4.2. Detection based on user profile characteristics is very limited. The existing surveys conducted in this area summarize
 Methods for detecting fake news based on user features are dis- the state-of-the-art models. These reviews discuss mainly the datasets
 cussed in this section. User interaction is critical in detecting fake news. used, explainability models used, and potential applications in the
 There is a subset of users who readily believe any fake news and have future. In comparison with existing surveys, our survey begins by
 features that differ from those who believe real news (Shu et al., 2018). examining the explainable AI techniques that are now in use, which
 The authors of Shu et al. (2019c) extracted implicit and explicit user is crucial for the reader to become familiar with the methods used in
 profile features from the user profiles of those who are inclined to XAI. Concepts related to XAI are covered in Section 2. In Section 3 we
 spread fake news. They proved the use of these derived features in the list the existing models for explainable fake news detection. We have
 categorization of fake news in their study. Their results surpassed prior also classified the present research works in this field by assessing them
 algorithms that employed news content to classify fake news, which from four different perspectives. A list of prospective applications for
 prompted the researchers to develop a detection method that used user the future are also discussed in this review.
 attributes and news content aspects. 2. Explainable AI systems (XAI)
 The authors of Ruchansky et al. (2017) introduced a new model
 Capture, Score, and Integrate (CSI) that aims to capture three char- The AI system is deemed an XAI system, if it can explain decisions
 acteristics of fake news. CSI is composed of three modules: Capture, taken. In this section we describe the most prevalent XAI concepts.
 Score, and Integrate. In the first module, Capture temporal patterns of
 the user’s activity extracted along with the text. The second module, 2.1. Interpretability
 Score, assigns a score to each user. Finally, in the Integrate module
 response, text and source information from the previous two modules 2.1.1. Intrinsic interpretability
 are combined and classification is performed. A similar approach was
 proposed by Liu and Wu (2020) for Fake News Early Detection named, Intrinsically interpretable machine learning methods are
 FNED.Inthiscase,theauthorsretrievedtextual and user characteristics interpretable due to their simple structure. Self-explanatory models
 3
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Fig. 3. A summary of the various interpretability approaches.
 such as decision trees, linear models, attention-based models, and rule- • Model-agnostic explanation:- The model-agnostic explana-
 based models are examples of the intrinsically interpretable model. tion is a method that separates the explanations from
 Even though these models provide precise explanations, they forfeit the trained model. An example is Permutation Feature
 the prediction performance. Importance: How well the model’s prediction accuracy
 deviates after permuting the feature values determines the
 2.1.2. Post-hoc interpretability significance of a given feature to the model’s performance.
 In the post-hoc interpretable model, interpretation methods are • Model-specific explanation:- In the model-specific explana-
 deployed after model training. Therefore, to explain the preceding tion, the internal structure and parameters are examined to
 model, a second model has to be built. The models are constrained derive the explanation. A generalized linear model (GIM)
 by their approximate nature, but the models’ accuracy is preserved. and tree-based ensemble models are examples. GIM in-
 The authors of Singhal et al. (2019) further divided interpretability cludes linear regression and logistic regression. Gradient
 into global and local interpretability. Local interpretability is about boosting machines, random forest, and XGBoost (2021) are
 understanding a single prediction of a model. Global interpretability is examples of tree ensemble models.
 concerned with comprehending how the model makes decisions based 2. Deep Neural Network (DNN) representation explanation:
 on a complete overview of its characteristics and learned components,
 such as weights, other parameters, and structures. • Explanation of Convolution Neural Network (CNN) repre-
 The different interpretability approaches listed in Fig. 3. are ex- sentation: It involves finding preferred input for neurons in
 plained next. a specific area.
 2.1.3. Intrinsic global interpretability • Explanation of Recurrent Neural Network (RNN) represen-
 Globally intrinsic models have two approaches: tation: Language modelling that predicts the next token
 given the previous token is used to examine the represen-
 1. Adding constraints: Examples for addition of constraints include tation learned by the RNN.
 pruning of the decision trees, restricting the number of fea- 2.1.6. Post-hoc local explanation
 tures used for predictions, and using features with monotonic 1. Model-agnostic explanation: Explanations are generated without
 relations. accessing the internal parameters of the model.
 2. Interpretable model extraction/mimic learning: A complicated
 model is simulated using simple models such as decision trees, • Local approximation based explanation: Any black-box
 rule based models, and linear models. model is approximated with a local, white-box model to
 2.1.4. Intrinsic local interpretability explain each individual prediction. The white box model’s
 Attention mechanisms are used in the locally intrinsic model to give parameters were then used to get the feature contribution
 users an understandable justification for a specific prediction. score.
 • Perturbation based explanation: The feature’s contribu-
 2.1.5. Post-hoc global explanation tion is determined by looking at how the prediction score
 changeswhenthefeatureischanged.Counterfactualexpla-
 Patterns are learned from the training data and retrained into nations are the outcomes. It can be accomplished by either
 model structures and parameters. There are two kinds of approaches. omission or occlusion. Omission removes a feature from
 These are Traditional machine learning explanations and Deep Neural the input, whereas occlusion replaces the feature with a
 Network (DNN) representation explanations. reference value. In case of word embedding, zero is used
 1. Traditional machine learning explanations: Feature importance as reference whereas in image grey value used.
 is utilized, which assesses each feature’s contribution to the 2. Model-specific explanation: In a model-specific explanation, ex-
 prediction. planations are created that are specific to the model.
 4
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 • Back-propagation based method: The gradient is deter- 4. Classification based on the levels of the explanation, categorized
 minedfor a given output with respect to the input utilizing the types of explanations into four levels.
 back-propagation to determine a feature’s contribution. Level 1: This category includes models that only employ one
 • Mask perturbation: The model-agnostic-based perturbation sort of explanation. The majority of models provide a post-hoc
 is computationally expensive. DNN-specific perturbations explanation.
 were obtained using Mask perturbation and a gradient Level 2: An alternative explanation is offered in addition to
 descent optimization framework. the single kind of explanation. If one explanation is not clear,
 • Investigation of deep representation: The DNN’s deep rep- we can deduce the classification reason from the alternative
 resentation offers a wealth of information for interpreta- explanation. For example, let us say a model wants to classify an
 tion. It lowers the likelihood of producing artifacts and pro- animalimageasadogorcat.Aheat-mapcanbeusedtodescribe
 duces more meaningful explanations. The deep representa- why a model classified the animal image as a dog. Along with
 tion is ignored in a perturbation or back-propagation-based the heat-map, a textual explanation is provided as an alternative
 approach. explanation.
 Wefind that the explanations detailed above are geared for developers Level 3: The explainee’s understanding is taken into account
 or researchers as their target audience. If the explainee is a naive user, while offering an explanation. At the moment, none of the
 this sort of explanations are not appreciated much. The authors of Du models fit into this category.
 et al. (2019) provided some user-friendly explanations. It includes con- Level 4: This category includes models that allow for interaction
 trastive explanations, counter-factual explanations, selective explanations, between the model and the explainee.
 credible explanations, and conversational explanations. 2.3. Discussion on explainability approaches
 1. Contrastive explanations: Differential explanation is a term used
 to describe this type of explanation. Instead of providing a TheauthorsofBelleandPapantonis(2021)havedoneageneralized
 reason why a particular prediction was made, the authors pro- discussion on explainable machine learning, which includes ideas of
 vide an explanation on why one prediction was made over the transparency, methods on assessing explainability, and the kinds of
 other. ‘‘Why Q rather than R?’’ Q is the prediction that needs explanation (post-hoc explanations). The authors discuss different types
 explanation, and R is the other case (Du et al., 2019). This is of post-hoc explanations that cover text, visual, local explanations,
 also called Counter-factual explanations. explanations using example, explanations using simplification, and
 2. Selective explanations: The user is given the top factors that feature relevance explanations. They proposed five frameworks that
 influenced the prediction. characterize advancements in explainable machine learning. These are
 3. Credible explanations: Good explanations must be consistent (1) taxonomic framework, (2) Transparency framework, (3) Explainable
 with the consumers’ existing understanding. However, while machine learning capability framework, (4) Explanation type framework
 these explanations may be accurate, the machine learning model and (5) Data scientist strategy framework (Belle and Papantonis, 2021).
 does not use correct evidence when making decisions. The authors of Belle and Papantonis (2021) have listed a detailed
 4. Conversational explanations: In the case of conversational expla- overview of the explainability approaches for deep learning models.
 nations, an explanation is presented as a discussion between the
 explainer and the explainee.
 2.2. A comparison of explainability and interpretability 2.4. Evaluating explainable systems
 AccordingtotheauthorsofAtakishiyevetal.(2020)Interpretability A detailed study and an approach for evaluating explainable ma-
 and explainability are two different concepts. Based on Atakishiyev chine learning systems are discussed in Mohseni et al. (2021). The
 et al. (2020), a human user can inspect an interpretable model, whereas authors grouped current design and assessment techniques that ar-
 explainability has to reason for prediction. The authors grouped re- range literature into three categories: (1) design goals, (2) evaluation
 search trends in XAI into four: concurrently constructed explanations, methodologies, and (3) the XAI system’s targeted users. The design goal is
 post-hoc explanations (which can be model-dependent or model indepen- organized with respect to three different user groups (AI end users, data
 dent), application-dependent vs. generic explanation, and classification experts, and AI experts). Each has its own set of design objectives. Eval-
 based on the levels of the explanation. uation methods used for evaluating machine learning models are user
 1. Concurrently constructed explanations: Along with their funda- mental models, user trust and reliance, explanation usefulness and satisfac-
 mental purpose, models attempt to generate explanations. An tion, human-machine task performance, and computational measurements
 example technique is learning attention weight from the input. (Mohseni et al., 2021). An interdisciplinary team of researchers created
 2. Post-hoc explanations: The core concept is to approximate a anXAIsystemforfakenewsdetectionfornon-expertdailynewsreaders
 trained model and generate explanations. There are two ap- as a case study to illustrate how the framework may be used in practice.
 proaches with post-hoc explanations: (1) model-dependent and 2.5. Need for XAI in fake news detection systems
 (2) model-independent. In a model-dependent method, we have
 access to the model’s internals. Model-independent techniques Fake news detection methodologies have attracted the attention of
 do not have access to the model’s internals. Gradient-weighted research community in recent years. Several approaches have been de-
 Class Activation Mapping (Grad-CAM) (Selvaraju et al., 2017) veloped based on machine learning and deep learning methods for fake
 and Local Interpretable Model-Agnostic Explanations (LIME) news detection. Although this fake news detection model has shown
 (Ribeiro et al., 2016) are two model-dependent and model- promising results, the explainability of such detection is still largely
 independent techniques, respectively. unknown, rendering the conclusions uncertain. Compared to earlier
 3. Application-dependent vs. Generic explanation Explanations: are detection models, new approaches use additional information (social
 created based on the type of explainee. In an application- context, multimedia content, and user information) to improve detec-
 dependent approach, the explainee is implicitly assumed to tion. This additional information can provide explainability (reasoning
 know about the application, hence the domain vocabulary is
 used. why a piece of news is fake or not).
 5
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Fig. 4. Explainable fake news detection methods.
 Fig. 5. An overview of statistics of explainable fake news detection models.
 3. Related work: Explainable fake news detection
 This Section reviews the literature and provides an overview of
 the various methods proposed for explainable fake news detection. We
 divided the related work based on their explainability approaches. The
 techniquesandfeaturesusedforeachproposedmodelaregivenwithall
 relevant details. Fig. 5 gives an overview of the statistics of the existing
 approaches based on explainabilty approaches used.
 Among the available models in explainable fake news detection,
 seven of the eleven models use attention-based models to provide ex-
 plainability (see Fig. 6). The other four methods are structured explain-
 ability, social psychological theories, Tsetlin machine clause, and ablation-
 based explanation. The tree diagram for the explainable fake news
 detection models is shown in Fig. 4.
 3.1. Attention based explainability approaches
 The related papers that employed attention mechanisms to explain
 the prediction of fake news are described below.
 3.1.1. XFake (Yang et al., 2019)
 In contrast to other existing approaches, XFake (Yang et al., 2019)
 could simultaneously analyse news items from a variety of perspectives.
 XFake (Yang et al., 2019) proposes three different frameworks: MIMIC
 (Yang et al., 2019), ATTN (Yang et al., 2019), and PERT (Yang et al., Fig. 6. Various attention based explainable fake news detection methods.
 2019). Using these three frameworks, it provided a prediction score for
 news fakeness along with an explanation for the predictions. Table 2
 presents details of the XFake. than news in plain text. This cognitive ability is being abused by those
 Discussion: The XFake dataset is quite small and it includes infor- whodistribute false information. Thus, expanding the model to include
 mation related to politics. The model has used only the textual content multimedia content analysis is necessary for increasing the potential
 for analysis. News with multimedia content is more appealing to people use of this model.
 6
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Table 2
 An overview of the XFake model.
 Model used MIMIC evaluates each news item through its attribute information. Tree ensemble models mimic the neural network’s performance.
 Transparency to the model is obtained using the constructed tree and the attribute’s importance scores. ATTN investigates statements
 through their semantic meaning and explanations obtained using self attention mechanisms. PERT studies the news statement’s
 linguistic features for detection through its semantic meaning and provides the perturbation-based explanation. With these three
 self-explanatory models, relevant explanations are derived for interpreting the detection results. XFake has implemented visual
 explanation for prediction and explanation using histogram, heatmap, and interactive diagrams for ensemble trees.
 Evaluation metric for Amazon Mechanical Turk (AMT) was employed to conduct human evaluations to validate the quality of explanations.
 explainability
 Data set Experiments were conducted on real world data crawled from the PoltiFact
 Accuracy 67.1%, 67.3%, 53.2% accuracy for MIMIC, ATTN and PERT respectively
 Critical appraisal of the tools The XFake is the first explainable fake news detection tool that looks at text from a variety of perspectives and helps provide users with
 a more accurate interpretation. The credibility checker of XFake is helpful for social media posts with more textual content.
 The MIMIC framework simultaneously keeps the excellent performance from neural networks and good explainability from tree
 ensemble models. The ATTN framework can capture global relationships between different words efficiently and weights assigned to the
 words. The PERT framework replaces the feature value with random noise drawn from the same distribution as the original one, and
 then the prediction difference is computed. This prediction difference is the significant score for the corresponding feature.
 The output of XFake is the news prediction as a unified score, indicating its probability of being fake. The results from three different
 frameworks are combined, and then weights for combination are tuned on a validation set. The higher the prediction score, the higher
 is the probability of fake news.
 Table 3
 An overview of the dEFEND model.
 Model used The authors employed a sentence comment co-attention sub-network to boost fake news detection performance to capture the intrinsic
 explainability of news phrases and user comments. The dEFEND (Cui et al., 2019) algorithm module provides search capabilities for
 searching news propagation networks, trending news, top claims, and related news. Simultaneously it shows the detection result and
 explanations. It displays both the detection results and the explanations at the same time.
 Evaluation metric for dEFEND (Shu et al., 2019a) is compared with Hierarchical Attention Networks (HAN) (Shu et al., 2019b) (A fact-checking tool called
 explainability claimbuster, 2021 is used to acquire the rank list of all check-worthy sentences in a piece of news content). They compared it to
 HSA-BLSTM (hierarchical neural network combined with social information) (Cui et al., 2019) to assess user feedback (Amazon
 Mechanical Turk recruited to evaluate the explainability rank list of the comments).
 Data set FakeNewsNet data set is used for the demonstration.
 Accuracy Accuracy Outperforms related works by at least 5.33%
 Critical appraisal of the tool The sentence comment co-attention network in the dEFEND model captures the semantic affinities between news sentences and user
 comments. The representations of user comments and news sentences that aid in identifying fake news are given high weights.
 3.1.2. dEFEND (Shu et al., 2019a) Table 4
 dEFEND(Explainable FakE News Detection) an explainable fake news An overview of the Extension of dEFEND model.
 detection system was created by Shu et al. (2019a) and Cui et al. Model used Comment credibility check achieved with
 (2019). Information obtained from user comments can be vital in deter- MultinominalNB, which classifies comments into
 mining fake news. As a result, the dEFEND model may be used on social junky or utility. Further analysis for fake news
 media platforms where user comments determine news credibility. identification involves news sentences and utility
 Table 3 presents details of the dEFEND. comments.
 Discussion: For the purpose of identifying fake news, the model Accuracy Accuracy increased above 90% for all machine
 examines the comments and the textual content. Comments in the learning algorithm
 social media posts include text, hashtags, and emojis. Emoji inclusion Critical appraisal of the The dEFEND model was applied to the filtered
 tool comments and news sentences after the
 is equally significant because it aids in understanding the context of the multinomial Naive Bayes classifier filtered out the
 comments.Thismodelhasalsorestricteditsanalysistotextualfeatures. junk comments to identify fake news.
 A credibility check of the comments corresponding to the news items
 is also missing.
 3.1.3. Extension of dEFEND (Sharma and Sharma, 2021) on features extracted which include user-based, textual, and temporal
 The authors of research work (Sharma and Sharma, 2021) used the characteristics. Table 5 presents details of the Propagation2Vec.
 methodology of the dEFEND (Yang et al., 2019) and the functionality Discussion The model performed the fake news classification by ob-
 of comment credibility check before attempting fake news detection. serving the way how fake news are propagated. For this, node level
 Table 4 presents details of the Extension of dEFEND. attributes like user credibility are utilized. The number of followers
 Discussion: Multi-modal properties are not taken into account by andverification status determine a user’s credibility. We cannot blindly
 the model for analysis. Although the authenticity of the comments trust a user based only on the aforementioned factors, as there are
 made in response to news items is taken into consideration, the cred- automatic ways for generating number of followers. The other features
 ibility of the people who makes comments is not considered. Analysis like text content and visual features of news contents must be taken into
 of Emojis in the comments is also essential to increase the feasibility of
 this model. account in order to improve the accuracy of the results of this model.
 3.1.4. Propagation2Vec (Silva et al., 2021)
 The training of Propagation2Vec (Silva et al., 2021) uses both a 3.1.5. Explainability in text classification (Kurasinski and Mihailescu, 2020)
 partial and a complete propagation network of labelled news items. The authors of Kurasinski and Mihailescu (2020) proposed a model
 Complete propagation network constructed with the partial informa-
 tion available in early propagation networks. The analysis is done that focuses on linguistic-based modalities for fake news detection. It
 7
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Fig. 7. Generic methodology for explainable fake news detection.
 uses two deep learning models for the classification. Table 6 presents Table 5
 the details about the explainability in text classification. An overview of the Propagation2Vec model.
 Discussion: At the moment, the model analyses text features in news Model used The model uses a hierarchical attention mechanism
 for assigning weights for nodes and cascades and
 content to detect fake news. Such approaches are possible only if the provides explainability.
 news items follow a pattern. Fake news spreaders have begun to create Evaluation metric for An ablation study was conducted, which studied
 news content in a variety of styles. As a result, other related features explainability the contribution of the different modules of the
 must be included to improve accuracy of the results. Propagation2Vec (Silva et al., 2021).
 Data set Experiments are performed on Fake News Net data
 3.1.6. MVAN (Multi-View Attention Network) (Ni et al., 2021) set
 MVAN utilizes source tweet and dissemination structure informa- Accuracy Outperformed 89.2% for dataset Gossipcop
 tion for the fake news classification. Each tweet’s propagation structure Critical appraisal of the Most of the existing work relies on complete
 graph is composed of the retweeting user’s information and their tool propagation networks to build their model, which
 feature vectors. Table 7 present the details of the MVAN model. takes more time for fake news identification.
 Discussion The current model uses tweet text and some user profile However, early detection of fake news in the
 health domain is essential to avoid the significant
 features for classification. We can include auxiliary information like harm it can cause. Propagation2Vec model is
 tweet comments and visual features to improve the performance. helpful in such situations.
 The model’s accuracy has improved by the
 3.1.7. GCAN (Graph-aware Co-Attention Network) (Lu and Li, 2020) computation of attention weights at the node and
 The authors of Lu and Li (2020) proposed a unique model called cascade levels. At the node level, the model has
 estimated attention weights for the user, text, and
 GCAN (Graph-aware Co-Attention Network) (Lu and Li, 2020). The temporal characteristics.
 algorithm predicts if the news is false or real based on the brief text
 contents of the tweet, user retweet sequence, and user profiles. Table 8
 presents the details of the GCAN model. Table 9 presents the details of the language-based fake news detec-
 Discussion: The short-text tweets and the corresponding sequence tion model.
 of retweet users without text comments are used to predict whether
 the source tweet is fake or not, as well as to generate explanations by 3.2.2. Based on social psychological theories
 emphasizing evidence from suspicious retweeters and the words they Theauthors of Zhou and Zafarani (2019) presented a pattern-driven
 concern. The model performed classification, focusing mainly on the fake news detection algorithm relying on a network. Since the spread of
 sequence of retweet users. Everyone has different interests, so they may fake news in networks differs from real news, this study analyses a ho-
 retweet news that are of interest to them. To prevent biased outcomes, mogeneous friendship network and discovers patterns based on social
 it is also essential to analyse news content. psychological theories. Explanation given based on social psychological
 3.2. Ablation based explanations theories. The classification algorithm makes use of patterns as features.
 Table 10 presents the details of the model.
 3.2.1. A language-based fake news detection using interpretable features Discussion: With this model, early fake news detection (which
 Discussion: The classification of fake news makes use of inter- deals with the identification of fake news before it spreads) is not
 pretable language characteristics. This model is language-specific. The achievable. The model used patterns found in English news text for the
 addition of multi-modal features can further improve the results. classification. If the fake news spreader changes his style, the patterns
 8
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Table 6
 An overview of the explainability in text classification.
 Model used The first is the BiDir-LSTM-CNN, which combines CNN (convolutional neural networks) with BiDir-LSTM (bidirectional recurrent neural
 networks), while the second is the BERT (bidirectional encoder representation from transformers). Then two models were trained with
 preprocessed data sets. They are represented as colour-coded texts that indicate the network’s interest in each processed word after
 extracting attention weights. When identifying false news texts, this visualization helps understanding how various models allocate their
 attention during classification
 Evaluation metric for Comparison done between the two models.
 explainability
 Data set Demonstration done using the data set Fake News Corpus.
 Accuracy Achieved 85% of accuracy
 Critical appraisal of the tool Unlike other baseline models, attention weights are colour coded. Important words which are coloured have a significant impact on the
 classification tasks. The model has provided statistics on attention distribution among the words. This approach can be helpful if the
 end user is not a subject matter expert.
 Table 7
 An overview of the MVAN model.
 Model used The model uses two attention networks, a text semantic attention network, to obtain the semantic representation of the source tweet
 information. Then a propagation structure attention network captures hidden details in the propagation structure of a tweet. The
 prediction model combines two representations and uses them to provide detection results.
 Evaluation metric for The relative importance of each module of MVAN (Ni et al., 2021) was verified using an ablation study. Interpretability is visualized
 explainability based on the attention scores of the source text.
 Data set Experiments done on datasets Twitter 15 and Twitter 16.
 Accuracy Outperform state-of-the-art methods by 2.5% in accuracy.
 Critical appraisal of the tool MVAN can detect fake news early, and the text semantic attention module is helpful for other text classification tasks on social media,
 like sentiment classification, topic classification, and insult detection. The propagation structure attention method is applied to identify
 those who distribute misinformation.
 Table 8
 Overview of the GCAN model.
 Model used First, the model extracted the user’s characteristics from their profiles and social activities and learned the word embeddings from the
 original short text. The convolutional and recurrent neural networks were utilized for learning the representation of retweet propagation
 based on user characteristics and created a graph to simulate user interactions. The graph-aware representation of user interactions is
 learned using the graph convolution network. Also developed a dual co-attention method that learned the relationship between the
 source tweet and retweet propagation and the co-influence between the source tweet and user interactions. Binary classification was
 performed using the learned embeddings. The dual co-attention mechanism produces appropriate explanations.
 Data set Experiments done on datasets Twitter 15 and Twitter 16.
 Accuracy Outperformed state-of-the-art methods by 16% in accuracy
 Critical appraisal of the tool The model’s primary goal is to forecast how reliable the source tweet will be. Explanations are produced by emphasizing the evidence
 about suspicious retweeters and the terms they concern.
 Table 9
 An overview of the model discussed in Qiao et al. (2020).
 Model used In Qiao et al. (2020) authors used CoCoGen (Complexity Contour Generator), a computational method that uses a sliding window
 approach to compute within-text distributions of feature scores to analyze raw texts from fake news datasets. Classified data using
 BRNN (Bidirectional Recurrent Neural Network) classifiers with GRU (Gated Recurrent Unit). The classifiers trained on interpretable
 features outperformed existing techniques using Recurrent Neural Network, Convolutional Neural Network, and Capsule Networks, that
 employ word embeddings to represent textual contents. An explanation is provided using Ablation study (which analyzes the
 importance of each feature).
 Data set ISOT (Guo et al., 2018) and LIAR (Wang, 2017) were the data sets used
 Accuracy 99% accuracy for ISOT and 27.4% for LIAR dataset
 Critical appraisal of the tool To detect fake news in social media news items, deep learning techniques and human interpretable components are used. The accuracy
 of this method has improved over other word embedding-based methods.
 Table 10 3.3. Based on tsetlin machine clauses
 An overview of the model discussed in Zhou and Zafarani (2019).
 Model used Support vector machine, k-nearest neighbours, The authors of Bhattarai et al. (2021) proposed a unique Tsetlin
 naive Bayes, decision tree and random forest. Fake Machine (TM) based interpretable fake news detection framework.
 news patterns can help the public comprehend Table 11 presents the details of the model.
 fake news and make fake news detection more
 understandable. Discussion: The present model for detecting fake news is mainly
 Data set For demonstration used two datasets PolitiFact and based on textual information. The model’s accuracy can be further
 Buzzfeed. improved with the incorporation of other auxiliary information from
 Accuracy 93% accuracy for politifact and 86% for Buzzfeed the news material.
 data set
 3.4. Structured explainability
 mayalso change. Validation of the model has to be done in news texts In the research work (Wu et al., 2021) they considered the broad
 in other languages. definition of fake news as ‘‘false news’’, as defined in Zhou and Zafarani
 9
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Table 11
 An overview of the model based on Tsetlin machine clauses.
 Model used The framework takes advantage of the TM’s conjunctive clauses to capture the lexical and semantic features of both true and fake news
 text. Trustworthiness of fake news is measured using clause ensembles. TM clauses contain negated, and unnegated inputs (termed as
 non-monotone clauses) used for providing explainability.
 Evaluation metric for Explainability of the model is demonstrated using a case study. It provides a higher F1-score than BERT and XLNet, even though it
 explainability obtained slightly lower accuracy.
 Data set Used Fake News Net data set for demonstration.
 Accuracy Outperforms previous works by at least 5%
 Critical appraisal of the tool The Tsetlin Machine (TM) technique, recently proposed (2018), is used to detect important words and their negations. This method
 helps to identify deception in text-only news posts.
 Table 12
 An overview of the research work (Jain and Wallace, 2019).
 Model used They have extracted entity-relation tuples from the training data to build a credential-based multi-relational knowledge graph and then
 applied a graph convolutional network (GCN) to learn the node and the relation embeddings. Graph convolutional networks classify
 news into fake or real. The GCN comprises graph embeddings and provides a relational explanation.
 Data set Experimentation done on three datasets Celebrity, PolitiFact and GossipCop.
 Accuracy 71.4% , 86.0%, 73.3% accuracy for data sets Celebrity, PolitiFact and GossipCop respectively.
 Critical appraisal of the tool Existing approaches ignore the global semantic aspects of news in favour of concentrating on the features of knowledge graphs. This
 model goes beyond the contributions of keywords by enhancing the semantic features with structural embeddings from knowledge
 graphs.
 Table 13
 Overview of the existing approaches in explainable fake news detection.
 Research articles Features used Techniques used Type of explainability
 Yang et al. (2019) News attributes, Semantic MIMIC, PERT, and ATTN Mimic learning (or Explanation by
 meanings of news statements and frameworks simplification), Perturbation based explanation,
 Linguistic features of news Self-attention based explanation and Visual
 statements explanation
 Shu et al. (2019a) and Cui News content and user comments RNN and Bidirectional GRU Co-attention captures intrinsic explainability
 et al. (2019) and Visual explanation
 Sharma and Sharma (2021) News content and user Multi nominal knowledge base Co attention mechanism
 comments(filtered) and RNN
 Qiao et al. (2020) Interpretable textual features CoCoGen, BRNN, and GRU Ablation explanations
 Kurasinski and Mihailescu Textual content BiDir-LSTM-CNN and BERT Attention based and Visual explanation
 (2020)
 Zhou and Zafarani (2019) Network based patterns Supervised learning algorithms Empirical studies on social psychological
 with five-fold cross-validation theories
 Bhattarai et al. (2021) News contents Tstelin Machine (TM) Local and Global interpretability
 Wu et al. (2021) Semantic contents of news articles Knowledge Graph Framework Structured explainability (Relational Level)
 (KGF)
 Ni et al. (2021) Source tweet and retweet users Graph Attention Network Attention based explanation
 Silva et al. (2021) User-based, textual and temporal Propagation network Attention based explanation
 features in propagation pattern
 Lu and Li (2020) User-based and textual features GCAN Attention based explanation
 (2020). Training data contains fake news about public figures and makes use of. Fig. 7 illustrates a general analysis of existing solutions
 organizations in articles, statements, and speeches. Table 12 presents for explainable fake news detection.
 the details of the model. In this section, we present the findings from our survey. We look at
 Discussion: We find that with the current implementation method- the existing research and discuss it from four different aspects :
 ology, early identification of fake news is not possible. The model (1) Utilizing visual characteristics in explainable fake news detec-
 analyses textual characteristics for fake news detection. If the news tion problems.
 spreaders change their writing style, the model will not be able to give (2) Analysis of existing approaches based on explainability types.
 accurate results. Recently, fake contents have been created with the (3) Examining current models explainability from the viewpoint of
 inclusion of audio and video. Therefore, analysis of auxiliary features the different explainee.
 like images, music, and videos will improve the model’s accuracy. (4) Taking the evaluation metrics used in current fake news detec-
 tion models into account.
 This categorization and analysis of the existing work in this area is
 4. Observations novel and hence is contribution of this review paper. Fig. 8 shows the
 four perspectives given the literature for explainable fake news detec-
 Welookedattheresearchonexplainable fake news detection meth- tion. Each of the perspectives are detailed in the following Sections.
 ods done between 2018 and 2022. Eleven innovative approaches can 4.1. Explainable fake news detection using visual features
 befoundinthisarea. Table 13 gives a brief overview of the most recent
 approaches for detecting explainable fake news, with an emphasis on Most existing efforts in explainable fake news detection explore
 the characteristics, strategies, and types of explainability that the model news content features to provide explanations for fake news detection
 10
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Fig. 8. Perspectives used for analyzing the work on fake news detection with explainability.
 problems. Only Shu et al. (2019a), Cui et al. (2019), and Sharma Even though they provide accurate explanations, keeps the accuracy
 and Sharma (2021) derived explanations for the detection model from uncertain.
 the perspective of news content and social engagements of the user. Following is a summary of our observations after analyzing the
 Zhou and Zafarani (2019) takes a different approach, using patterns research works listed above.
 identified in friendship networks to build an explainable fake news de- In the literature, there is a disagreement over whether attention
 tection system. Recently, Silva et al. (2021) proposed a methodology for weight should be considered as explainability or not (Serrano and
 explainable fake news identification that leverages user-based, textual, Smith, 2019; Jain and Wallace, 2019). Visual explanation, explanation
 and temporal characteristics in the dissemination pattern of the news. by simplification (mimic learning), and Feature relevance explana-
 A majority of recent works, in general, rely on textual analysis. tion (ablation explanation, perturbation-based explanation) are some
 Recently, various approaches Singhal et al. (2019, 2020), and Shah post-hoc explanations types from the literature. Feature relevance ex-
 and Kobti (2020) have combined visual-based features with textual planation only indicates feature’s importance, but do not consider
 features to capture different characteristics of fake news and achieved information about the feature interactions. These are computationally
 promising results. Visual content is an essential aspect of fake news expensive. Wu et al. (2021) provided a relational explanations for the
 since it seeks to use multimedia content with images or videos to entice prediction. While closely looking at the different explanation types,
 and mislead readers for quick spread. As far as we know, no studies each of them covers different aspects of explainability. Hence com-
 exist on explainable fake news detection using visual features. Research bining multiple approaches adds more robustness to the explainability
 work needs to be done in this direction. of the model. Only (Yang et al., 2019) tried to provide explainability
 The authors of Reis et al. (2019) investigated around 300,000 by combining different approaches. Interactive explanation approaches
 models to identify features’ informativeness and suggested that various include interaction between explainee and system, thereby allowing
 formsoffakenewsareidentifiedwithspecificcombinationsoffeatures. users to improve the system performance. However, in literature, only
 The authors concluded that various models used significantly different (Yang et al., 2019) incorporates this approach. Hence we highlight
 logic to distinguish fake news from true news. Hence combinations of the necessity for more research on interactive explainable detection
 different features may provide better detection and explanations for the systems.
 predictions. As far as we know, a combination of models with different 4.3. From the explainee type perspective
 features for better explainability is unaddressed. As there is no universal
 definition for ‘‘fake news’’, classifying them under one single definition The type of explainee (End-users, Data experts, or AI experts) and
 is not efficient. Each definition covers each aspect of the fake news. their domain familiarity plays a vital role in deciding the type of
 Thus based on definition fake news detection problem can be framed explanation to be used. All the existing approaches from the litera-
 as a multi-class problem. So far we have not found any work on a multi- ture prioritize data experts or AI experts. We discovered that these
 class explainable fake news detection problem that classifies the fake approaches are unfavourable if the explainee are end-users. Thus focus
 news into distinct classes and offers an explanation based on feature onmorehuman-friendlyapproachesinexplainable fake news detection
 relevance. are essential.
 4.2. According to the type of explainability 4.4. Considering the evaluation metric
 Reviewing from the explainability type perspective, state of the The explainable fake news detection system surveyed employs hu-
 art for explainable fake news detection shows diverse approaches. man evaluation by Amazon Mechanical Turk (AMT) to assess the
 The most common being intrinsic explainability, and mainly atten- quality of explanations. Other approaches involve visualization or case
 tion mechanisms are used commonly. Yang et al. (2019), Shu et al. study to measure explanation quality. However, we find that the as-
 (2019a), Cui et al. (2019), Sharma and Sharma (2021), Kurasinski and sessment of performance throughout the explainable system is still
 Mihailescu (2020), Ni et al. (2021), and Silva et al. (2021) provide unresolved. From our observations we have not found any common
 explanations to the prediction made by the classifier using attention benchmark on which we can compare the performance of different
 weights. Ni et al. (2021) had used two types of attention networks. explainable models.
 11
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 5. Research directions References
 Based on our assessment of the related work in this area, we list Atakishiyev, S., Babiker, H., Farruque, N., Goebel1, R., Kima, M.-Y., Motallebi, M.H.,
 below some promising research directions that we find in this field: Rabelo, J., Syed, T., Zaïane, O.R., 2020. A multi-component framework for the
 analysis and design of explainable artificial intelligence. arXiv:2005.01908.
 • Most social media news items employ multimedia content with Belle, V., Papantonis, I., 2021. Principles and practice of explainable machine learning.
 images or videos to attract and mislead readers for rapid spread. Front. Big Data 4, 39. http://dx.doi.org/10.3389/fdata.2021.688969, URL https:
 //www.frontiersin.org/article/10.3389/fdata.2021.688969.
 Hence investigating the significance of visual features in identify- Bhattarai, B., Granmo, O.-C., Jiao, L., 2021. Explainable tsetlin machine framework for
 ing explainable fake news is a promising avenue for research. fake news detection with credibility score assessment. arXiv:2105.09114.
 • Developing explainable models for detecting fake news that com- 2021. Claimbuster. https://idir.uta.edu/claimbuster/, Accessed on 20 October.
 bines textual and visual features with the correlation analysis Classify.news, 2021. FullFact.org. https://makenewscredibleagain.github.io/, Accessed
 on 20 October.
 between these features in social media news posts is another Cui, L., Shu, K., Wang, S., Lee, D., Liu, H., 2019. DEFEND: A system for explainable
 potential research domain. fake news detection. In: Proceedings of the 28th ACM International Conference
 • The fakeness of news articles can be efficiently detected and on Information and Knowledge Management. CIKM ’19, Association for Comput-
 interpreted by combining various features of news items. Thus an ing Machinery, New York, NY, USA, pp. 2961–2964. http://dx.doi.org/10.1145/
 3357384.3357862.
 ensemble technique that combines models with different features Du, M., Liu, N., Hu, X., 2019. Techniques for interpretable machine learning. Commun.
 for better explainability is a promising investigation area. ACM 63 (1), 68–77. http://dx.doi.org/10.1145/3359786.
 • There is need to develop a multi-class explainable fake news 2021. Factmata.com. https://factmata.com/, Accessed on 20 October.
 detection system that classifies the fake news into distinct classes 2021. FakeNewsNet. https://github.com/KaiDMML/FakeNewsNet, Accessed on 20
 and offers explanations for the predictions made. Such systems October.
 2021. Fullfact.org. https://fullfact.org/, Accessed on 20 October.
 can help in detecting fake news regardless of its definition, as Guo, H., Cao, J., Zhang, Y., Guo, J., Li, J., 2018. Rumor Detection with Hierarchical
 there is no universally accepted definition for fake news. Social Attention Network. CIKM ’18, Association for Computing Machinery, New
 York, NY, USA, http://dx.doi.org/10.1145/3269206.3271709.
 2021. Hoax-slayer.com. https://smhoaxslayer.com/, Accessed on 20 October.
 6. Conclusions Jain, S., Wallace, B.C., 2019. Attention is not explanation. CoRR, abs/1902.10186
 arXiv:1902.10186 URL http://arxiv.org/abs/1902.10186.
 In todays world of Online Social Media(OSM), almost everyone is Jin, Z., Cao, J., Zhang, Y., Zhou, J., Tian, Q., 2017. Novel visual and statistical image
 dependingonthenewsitemswhicharepublishedthroughtheOSMs.At features for microblogs news verification. IEEE Trans. Multimed. 19 (3), 598–608.
 http://dx.doi.org/10.1109/TMM.2016.2617078.
 the same time, the wide spread of fake news and fake articles have also Kurasinski, L., Mihailescu, R.-C., 2020. Towards machine learning explainability in text
 grown into a very big threat in the domain. Identifying and filtering classification for fake news detection. In: 2020 19th IEEE International Conference
 out fake news is a hot problem which needs urgent research attention. on Machine Learning and Applications (ICMLA). pp. 775–781. http://dx.doi.org/
 Moreover the Law Enforcement Agencies are looking for convincing 10.1109/ICMLA51294.2020.00127.
 2021. Lead stories. https://leadstories.com/, Accessed on 20 October.
 and authentic explanations for such classifications of the news items. Liu, Y., Wu, Y.-F.B., 2020. FNED: A deep network for fake news early detection on
 In this paper, we have presented a review on existing approaches social media. ACM Trans. Inf. Syst. 38 (3), http://dx.doi.org/10.1145/3386253.
 in the emerging area of explainable AI (XAI) applied to fake news Lu, Y.-J., Li, C.-T., 2020. GCAN: Graph-aware co-attention networks for explainable
 detection and our survey on current methods for explainable fake news fake news detection on social media. arXiv:2004.11648.
 2021. Media bias/fact check. https://mediabiasfactcheck.com/, Accessed on 20 October.
 detection in the literature. An in depth study on recent approaches Mohseni, S., Yang, F., Pentyala, S.K., Du, M., Liu, Y., Lupfer, N., Hu, X., Ji, S.,
 in explainable fake news detection is reported. We have categorized Ragan, E.D., 2020. Machine learning explanations to prevent overtrust in fake
 the existing work in the area into four and highlighted the research news detection. CoRR, abs/2007.12358 arXiv:2007.12358 URL https://arxiv.org/
 opportunities in all these four groups. In the first category we have abs/2007.12358.
 Mohseni, S., Zarei, N., Ragan, E.D., 2021. A multidisciplinary survey and framework
 highlighted the research opportunities in explainable fake news de- for design and evaluation of explainable AI systems. ACM Trans. Interact. Intell.
 tection using visual features. In the second category, we discussed Syst. 11 (3–4), http://dx.doi.org/10.1145/3387166.
 opportunities in explainable fake news detection based on the types Ni, S., Li, J., Kao, H.-Y., 2021. MVAN: Multi-view attention networks for fake news
 of explainability. In the third category, we emphasize prospects from detection on social media. IEEE Access 9, 106907–106917. http://dx.doi.org/10.
 1109/ACCESS.2021.3100245.
 the explainee type perspective. In the fourth category, we report the 2021. Politifact.com. https://www.politifact.com/, Accessed on 20 October.
 needforresearchconsideringtheevaluationmetricforexplainablefake Qiao, Y., Wiechmann, D., Kerz, E., 2020. A language-based approach to fake news
 news detection. detection through interpretable features and BRNN. In: Proceedings of the 3rd
 Our research focuses on analyzing visual and textual content on International Workshop on Rumours and Deception in Social Media (RDSM).
 Association for Computational Linguistics, Barcelona, Spain (Online), pp. 14–31,
 social media platforms. Then the relationship between the textual and URL https://aclanthology.org/2020.rdsm-1.2.
 visual components is used to explain the prediction made. Reis, J.C.S., Correia, A., Murai, F., Veloso, A., Benevenuto, F., 2019. Explainable
 machine learning for fake news detection. In: Proceedings of the 10th ACM
 Conference on Web Science. WebSci ’19, Association for Computing Machinery,
 CRediT authorship contribution statement New York, NY, USA, pp. 17–26. http://dx.doi.org/10.1145/3292522.3326027.
 Ribeiro, M.T., Singh, S., Guestrin, C., 2016. ‘‘Why should I trust you?’’: Explaining the
 AthiraA.B.:Conceptualization,Writing–originaldraft.S.D.Madhu predictions of any classifier. In: Proceedings of the 22nd ACM SIGKDD International
 Kumar: Supervision. Anu Mary Chacko: Supervision. Conference on Knowledge Discovery and Data Mining. KDD ’16, Association for
 Computing Machinery, New York, NY, USA, pp. 1135–1144. http://dx.doi.org/10.
 1145/2939672.2939778.
 Declaration of competing interest Ruchansky, N., Seo, S., Liu, Y., 2017. CSI: A hybrid deep model for fake news detection.
 In: Proceedings of the 2017 ACM on Conference on Information and Knowledge
 Management. CIKM ’17, Association for Computing Machinery, New York, NY, USA,
 The authors declare that they have no known competing finan- pp. 797–806. http://dx.doi.org/10.1145/3132847.3132877.
 cial interests or personal relationships that could have appeared to Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D., 2017.
 influence the work reported in this paper. Grad-CAM: Visual explanations from deep networks via gradient-based localization.
 In: 2017 IEEE International Conference on Computer Vision (ICCV). pp. 618–626.
 http://dx.doi.org/10.1109/ICCV.2017.74.
 Data availability Serrano, S., Smith, N.A., 2019. Is attention interpretable?. arXiv:1906.03731.
 Shah, P., Kobti, Z., 2020. Multimodal fake news detection using a cultural algorithm
 with situational and normative knowledge. In: 2020 IEEE Congress on Evolutionary
 No data was used for the research described in the article. Computation (CEC). pp. 1–7. http://dx.doi.org/10.1109/CEC48606.2020.9185643.
 12
 Athira A.B., S.D.M. Kumar and A.M. Chacko Engineering Applications of Artificial Intelligence 122 (2023) 106087
 Sharma, D.K., Sharma, S., 2021. Comment filtering based explainable fake news Singhal, S., Shah, R.R., Chakraborty, T., Kumaraguru, P., Satoh, S., 2019. SpotFake: A
 detection. In: Singh, P.K., Wierzchoń, S.T., Tanwar, S., Ganzha, M., Ro- multi-modal framework for fake news detection. In: 2019 IEEE Fifth International
 drigues, J.J.P.C. (Eds.), Proceedings of Second International Conference on Conference on Multimedia Big Data (BigMM). pp. 39–47. http://dx.doi.org/10.
 Computing, Communications, and Cyber-Security. Springer Singapore, Singapore, 1109/BigMM.2019.00-44.
 pp. 447–458. 2021. Snopes.com. https://www.snopes.com/, Accessed on 20 October.
 Shu, K., Cui, L., Wang, S., Lee, D., Liu, H., 2019a. DEFEND: Explainable fake news Tanwar, V., Sharma, K., 2020. Multi-model fake news detection based on concatenation
 detection. In: Proceedings of the 25th ACM SIGKDD International Conference on of visual latent features. In: 2020 International Conference on Communication and
 Knowledge Discovery &Amp; Data Mining. KDD ’19, Association for Computing Signal Processing (ICCSP). pp. 1344–1348. http://dx.doi.org/10.1109/ICCSP48568.
 Machinery, New York, NY, USA, pp. 395–405. http://dx.doi.org/10.1145/3292500. 2020.9182398.
 3330935. 2021. Truthorfiction.com. https://www.truthorfiction.com/, Accessed on 20 October.
 Shu, K., Mahudeswaran, D., Wang, S., Liu, H., 2019b. Hierarchical propagation Wang, W.Y., 2017. ‘‘Liar, liar pants on fire’’: A new benchmark dataset for fake news
 networks for fake news detection: Investigation and exploitation. CoRR, abs/1903. detection. arXiv:1705.00648.
 09196 arXiv:1903.09196 URL http://arxiv.org/abs/1903.09196. Wikipedia, 2021. Fake news. URL https://en.wikipedia.org/wiki/Fake_news#Definition,
 Shu, K., Wang, S., Liu, H., 2018. Understanding user profiles on social media for fake Accessed on 20 October 2021.
 news detection. In: 2018 IEEE Conference on Multimedia Information Processing Wu, K., Yuan, X., Ning, Y., 2021. Incorporating relational knowledge in explainable fake
 and Retrieval (MIPR). pp. 430–435. http://dx.doi.org/10.1109/MIPR.2018.00092. news detection. In: Advances in Knowledge Discovery and Data Mining. Springer
 Shu, K., Zhou, X., Wang, S., Zafarani, R., Liu, H., 2019c. The Role of User Profiles for International Publishing, Cham, pp. 403–415.
 Fake News Detection. ASONAM ’19, Association for Computing Machinery, New 2021. Xgboost. https://github.com/dmlc/xgboost, Accessed on 20 October.
 York, NY, USA, pp. 436–439. http://dx.doi.org/10.1145/3341161.3342927. Yang, F., Pentyala, S.K., Mohseni, S., Du, M., Yuan, H., Linder, R., Ragan, E.D., Ji, S.,
 Silva, A., Han, Y., Luo, L., Karunasekera, S., Leckie, C., 2021. Propagation2Vec: Embed- Hu, X.B., 2019. XFake: Explainable fake news detector with visualizations. In: The
 ding partial propagation networks for explainable fake news early detection. Inf. World Wide Web Conference. WWW ’19, Association for Computing Machinery,
 Process. Manage. 58 (5), 102618. http://dx.doi.org/10.1016/j.ipm.2021.102618, NewYork, NY, USA, pp. 3600–3604. http://dx.doi.org/10.1145/3308558.3314119.
 URL https://www.sciencedirect.com/science/article/pii/S030645732100114X. Zhang, X., Ghorbani, A.A., 2020. An overview of online fake news: Characteriza-
 Singh, V.K., Ghosh, I., Sonagara, D., 2021. Detecting fake news stories via multimodal tion, detection, and discussion. Inf. Process. Manage. 57 (2), 102025. http://dx.
 analysis. J. Assoc. Inf. Sci. Technol. 72 (1), 3–17. http://dx.doi.org/10.1002/ doi.org/10.1016/j.ipm.2019.03.004, URL https://www.sciencedirect.com/science/
 asi.24359, arXiv:https://asistdl.onlinelibrary.wiley.com/doi/pdf/10.1002/asi.24359 article/pii/S0306457318306794.
 URL https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.24359. Zhou, X., Zafarani, R., 2019. Network-based fake news detection: A pattern-driven ap-
 Singhal, S., Kabra, A., Sharma, M., Shah, R.R., Chakraborty, T., Kumaraguru, P., 2020. proach. SIGKDD Explor. Newsl. 21 (2), 48–60. http://dx.doi.org/10.1145/3373464.
 SpotFake+: A multimodal framework for fake news detection via transfer learning 3373473.
 (student abstract). In: Proceedings of the AAAI Conference on Artificial Intelligence, Zhou, X., Zafarani, R., 2020. A survey of fake news: Fundamental theories, detection
 Vol. 34 (10). pp. 13915–13916. http://dx.doi.org/10.1609/aaai.v34i10.7230, URL methods, and opportunities. ACM Comput. Surv. 53 (5), http://dx.doi.org/10.1145/
 https://ojs.aaai.org/index.php/AAAI/article/view/7230. 3395046.
 13

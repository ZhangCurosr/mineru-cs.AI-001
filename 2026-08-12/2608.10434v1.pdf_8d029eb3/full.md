# Conversational versus Dashboard Explainable AI for UAV Intrusion Detection: An Empirical Study of Operator Trust and Reliance

Cong Chi Nguyen<sup>1</sup>, Trang Mai Xuan<sup>1</sup>, Vu-Duc Ngo<sup>2</sup>, Kim-Ngan Thi Nguyen<sup>3</sup>, Trong-Nghia Nguyen<sup>3</sup>, and Thien Van Luong<sup>3⋆</sup>

<sup>1</sup> A2I Lab, Phenikaa School of Computing, Phenikaa University, Hanoi, Vietnam <sup>2</sup> MobiFone HighTech Center, MobiFone Corporation, Hanoi, Vietnam   
3 Business AI Lab, College of Technology, National Economics University, Vietnam   
{cong.nguyenchi, trang.maixuan}@phenikaa-uni.edu.vn, duc.ngo@mobifone.vn, {ngannguyen, nghiant, thienlv}@neu.edu.vn

Abstract. Machine learning-based Intrusion Detection Systems (IDS) have demonstrated superior performance in securing Unmanned Aerial Vehicle (UAV) networks. However, the ‘black-box’ nature of these models, combined with the high dimensionality of multimodal cyber-physical data, poses significant interpretability challenges. Static visualization dashboards may struggle to present complex relationships among multimodal cyber-physical features in a form that is easy for operators to inspect and interpret. To address this, we propose a Conversational XAI interface powered by Large Language Models (LLM) to facilitate ondemand investigation. In a controlled experiment with participants, we systematically evaluated the impact of this conversational interface versus a traditional XAI Dashboard on operator understanding, trust, and reliance during post-incident auditing tasks. Our results suggest that the conversational interface was perceived as more useful than the dashboard, potentially because it helped participants access and synthesize relevant information more easily. However, this benefit was accompanied by a lower level of appropriate self-reliance, indicating a potential risk of over-reliance. One possible interpretation is that the natural-language responses made the AI advice easier to accept, which may have reduced participants’ tendency to verify the underlying evidence when the IDS was incorrect. These findings point to a potential trade-of in human-AI collaboration for UAV intrusion auditing: interaction mechanisms that improve perceived usability may also increase the risk of inappropriate reliance. We conclude by discussing design implications for future XAI systems that balance seamless interaction with cognitive forcing functions to foster appropriate reliance.

Keywords: Intrusion Detection Systems · Explainable AI · UAV.

## 1 Introduction

Unmanned Aerial Vehicles (UAVs) have witnessed widespread adoption across diverse sectors [1], yet their heavy reliance on open wireless communication exposes them to severe cyber threats such as replay, spoofing, and Denial-of-Service (DoS) attacks [2]. To safeguard these networks, robust Intrusion Detection Systems (IDS) have become imperative. Unlike traditional networks, UAV ecosystems generate high-dimensional multimodal data, necessitating the fusion of cyber features and physical kinematics for efective threat identification; while benchmarks like the UAV-ID dataset [3] show that fusing these modalities enhances detection eficacy, this complexity comes at a cost. Driven by such rich data, modern IDS increasingly rely on machine learning, yet these models often function as opaque ‘black boxes’, creating a barrier to trust in high-stakes defense [4]. Although post-hoc explanation techniques have been developed to interpret these predictions [5], their mere existence does not guarantee efective human-AI collaboration: operators must establish appropriate reliance on IDS alerts when validating system outputs and reasoning about possible attack causes [6], which requires not just access to explanations but a comprehensive understanding of the underlying rationale. However, static XAI dashboards may not always match the diverse information needs of stakeholders [7], particularly when operators need to inspect relationships among many cyber-physical features. In such settings, dense visual displays can make explanation navigation efortful [8]. They also demand high AI literacy, leaving operators unable to articulate their specific inquiries efectively [9].

To address this gap, this paper focuses on the human-AI interaction layer of explainable UAV intrusion detection. Our main contributions are as follows:

• We implement a conversational interface that allows users to query global explanations, local feature attributions, counterfactual explanations, and what-if analyses through natural language, using the same underlying XAI evidence as a dashboard baseline.

• We conduct a user study comparing no explanation, Dashboard XAI, and Conversational XAI in a fallible IDS setting, measuring subjective understanding, explanation utility, trust, and behavioral reliance.

• We show that Conversational XAI increases agreement with AI advice but reduces appropriate self-reliance compared with Dashboard XAI when the IDS is incorrect. This finding indicates that fluent natural-language explanations may make AI advice easier to accept without necessarily improving evidence verification.

## 2 Related Work

We review human-AI decision making and explainable AI, identifying that the comparative impact of conversational interfaces versus XAI dashboards on operator understanding and reliance has been largely overlooked in the literature.

## 2.1 Human-AI Decision Making

Predictive AI systems are powerful but seldom perfect [2], which is problematic in high-stakes UAV applications: machine learning-based IDS lacks legal and moral accountability, yet human operators bear full responsibility for critical decisions. Complementary team performance, in which humans leverage AI to cover their own blind spots, therefore becomes a critical goal [10], and remains vital in the era of Large Language Models (LLM) that increasingly underpin explanation generation tasks [11].

Achieving appropriate reliance is especially hard in the UAV domain given the high cost of errors: algorithm aversion may lead pilots to disregard valid jamming alerts and lose the vehicle, while algorithm appreciation can cause them to accept false positives and abort missions needlessly [12]. Prior work has explored how confidence, risk perception, and explanations shape these outcomes [13], and how human factors such as expertise and domain knowledge afect trust [14]. To mitigate bias, cognitive forcing functions, interaction mechanisms that require users to pause and deliberate before deciding, were proposed to encourage more thoughtful engagement with explanations [12], alongside tutorial interventions that reveal AI limitations [2].

Nevertheless, these studies primarily focus on static interventions in low-risk environments. It remains empirically unclear whether conversational interaction in UAV intrusion auditing acts as a cognitive aid or a source of inappropriate reliance. This motivates us to fill such a research gap and explore how a conversational XAI interface afects user understanding, perceived utility, trust, and reliance in IDS auditing tasks.

## 2.2 Explainable AI

Although machine-learning-based IDS have shown promising predictive performance for UAV networks, their ‘black-box’ nature remains a primary barrier to adoption. Prior legal and policy discussions have emphasized the importance of meaningful information about automated decisions, especially in high-stakes settings [15].

To address the ‘black-box’ issue, researchers have proposed diverse XAI algorithms: feature attribution methods highlight critical input features [16], while counterfactual and contrastive explanations ofer alternatives [17], with a comprehensive review in [18]. Yet raw algorithmic outputs are often insuficient. Because UAV operators have diverse information needs spanning flight control and network security, there is no one-size-fits-all solution; human-centered XAI [19] therefore focuses on how explanations shape the operator’s mental model, an internal representation of the security status [20]. Efectiveness depends heavily on presentation, and Conversational User Interfaces (CUI) have emerged as a promising solution: compared to Graphical User Interfaces, CUIs ofer more human-like interaction [9] and can simplify complex monitoring by filtering information, yielding higher subjective trust [18].

While conversational assistants have been widely adopted in domains such as search engines [21], their application in UAV security remains limited. Although [22] explored conversational XAI in the contexts of collaborative writing, to the best of our knowledge, limited prior work has systematically compared conversational XAI interfaces with dashboard-based XAI in terms of operator trust and reliance for UAV intrusion detection. This motivates our work to bridge this empirical gap by analyzing whether a conversational interface can better support human-AI decision-making in auditing IDS alerts and interpreting potential threats against UAV networks.

## 3 Proposed Method

We propose a conversational XAI interface for UAV IDS that uses an LLM to interpret outputs from various XAI methods and deliver them as interactive explanations to the operator. A comprehensive XAI Dashboard covering the same underlying methods serves as a comparative baseline.

## 3.1 Intrusion Detection and XAI Method

![](images/2c5854f11437adfe977fb0af245569ce1d8790a34b25df161cd4696c30f0676e.jpg)  
Fig. 1. The architectural overview of the proposed XAI-IDS interface framework.

The overall architecture of our proposed framework is illustrated in Fig. 1. The pipeline commences with data collected from UAV sensors and ground stations, comprising 42,258 samples. During preprocessing, we drop columns that would otherwise leak the label, such as timestamp\_c, frame.number, and wlan.bssid, and split the data into training and test sets at a 7:3 ratio. Because the attack classes are imbalanced, we report macro-averaged metrics so that minority classes are weighted equally. We report the model with the highest macro F1-score on the test set, which provides an aggregate measure that balances precision and recall across classes, with the comparative results presented in Table 1.

Subsequent to the classification phase, interpretability is achieved through a suite of XAI methods selected to align with the taxonomy of user information needs regarding the rationale of AI advice [7]. Following the XAI question bank [7], we focus on four information needs that are directly supported by our interface: how, why, how to be that, and what if. Our framework integrates the following methods:

• Global Explanation (PDP): To address “how” feature values generally afect attack probabilities, we utilize Partial Dependence Plots (PDP) [23]. This method visualizes global trends, showing the operator how the probability of a specific attack type fluctuates as a feature varies across its range.

• Local Feature Attribution (SHAP): To answer “why” a specific alert was classified as malicious or benign, we use TreeSHAP to compute instancelevel feature contributions for the predicted class of the XGBoost IDS. Using SHAP as the default attribution method avoids switching between explanation methods across alerts and provides a consistent interpretation format throughout the study.We evaluate attribution faithfulness with a deletionstyle test. Features are ranked by absolute SHAP value, and the top-k features, $S _ { k }$ with $k \in \{ 1 , \ldots , 5 \}$ , are replaced with valid values sampled from the empirical training distribution. We then measure the drop in the predicted probability of the original class:

$$
\begin{array} { r } { \varDelta _ { k } = p _ { y } ( x ) - \mathbb { E } _ { \tilde { x } \sim \mathcal { D } _ { t r a i n } } [ p _ { y } ( x _ { S _ { k }  \tilde { x } _ { S _ { k } } } ) ] . } \end{array}
$$

Continuous features are restricted to observed training ranges, and categorical or protocol-related features are replaced only with valid observed values. The deletion scores are averaged as:

$$
\mathrm { F a i t h f u l n e s s ~ A U C } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \varDelta _ { k } , \quad K = 5 .
$$

We use this score only to validate explanation faithfulness, not to select explanation methods.

• Contrastive Counterfactual Explanation (MACE): To answer “how would this alert need to difer to be classified ${ d i f f e r e n t l y ? } ^ { }$ , we use MACE to generate counterfactual explanations. We treat these counterfactuals as diagnostic evidence rather than direct operational recommendations, since many IDS features are observed packet- or protocol-level attributes. Counterfactual search is constrained to valid feature domains: continuous features remain within observed training ranges, categorical features use valid observed categories, and immutable or leakage-prone fields are excluded. The interface reports minimal evidence changes as $V a l u e _ { o l d }  V a l u e _ { n e w }$ , or states that no valid counterfactual is available.

• Interactive Simulation (What-If): To support hypothesis testing (“what if”), we integrate a What-If analysis toolkit [24]. This allows operators to manually perturb feature values and simulate the resulting change in the model’s prediction and confidence scores.

The outputs from these modules, the raw dataset features, the classifier’s predictions, and the generated XAI explanations, form a shared knowledge base used by both the XAI Dashboard and the LLM-powered Conversational XAI Interface, with the goal of providing comparable underlying evidence across the two modalities.

Dashboard XAI Interface To serve as a comparative baseline for established operational standards, we developed the XAI Dashboard, a GUI designed to provide security operators with on-demand access to intrusion explanations. As depicted in the system architecture, this interface organizes the XAI methods (PDP, SHAP, MACE, and What-If) into distinct modules accessible via a navigation control panel.

Conversational XAI Interface While dashboards provide structured access to data, they may ofer less flexibility for addressing dynamic, user-specific inquiries during threat investigations. We therefore propose this interface as a natural-language front end, powered by llama3.1:70b-instruct-q4\_K\_M, that maps user queries to executable XAI tasks and returns textual summaries of the resulting explanations. The system operates in three steps:

1. Intent Recognition: When an operator queries an alert, the Llama model is prompted to parse the input and map the operator’s query to one of the supported XAI information needs.

2. Tool Execution: Leveraging the native function calling capability of the Llama, the agent operates in a reasoning loop. When an intent is recognized, the model outputs a structured procedure call.

3. Response Synthesis: Finally, the model synthesizes the technical outputs into a coherent textual narrative, summarizing the visual results and presenting them as a natural-language explanation to the operator.

The interface also suggests context-aware hint questions to trigger investigations, allowing operators to explore cyber threats through natural conversation.

## 4 Experimental Results and Discussion

This study focuses on analyzing the impact of the proposed XAI interfaces rather than solely evaluating the quality of explanations.

## 4.1 Performance Measures

Operator Understanding of the IDS Drawing on [25], we adopted four dimensions to assess how operators comprehend the underlying IDS logic through simulated intrusion scenarios. Detailed questionnaire items are provided in the supplementary materials.

Explanation Utility Following [26], we evaluated utility along four dimensions, completeness, coherence, clarity, and usefulness, capturing how efectively an explanation fosters a robust mental model of the attack.

Operator Trust We quantified subjective trust using three validated subscales from the Trust in Automation questionnaire, previously validated and commonly used for evaluating trust in automation and human-AI collaboration [2].

Reliance and Appropriate Reliance We evaluate reliance by logging operators’ pre- and post-interface decisions, measuring the agreement and switch fractions. To distinguish warranted from blind reliance, we use appropriate-reliance measures [27]: relative positive AI reliance (RAIR) and relative positive selfreliance (RSR). A decreasing RSR during disagreements can indicate a higher risk of over-reliance, particularly when participants abandon correct initial judgments in favor of incorrect AI advice. Diferences across the three conditions are tested with the non-parametric Kruskal-Wallis H test, which suits the ordinal, non-normally distributed scores and reports a test statistic H with its p-value.

## 4.2 Classifier selection

Table 1. Multi-class detection performance of diferent base models
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1-Score</td></tr><tr><td rowspan=1 colspan=1>XGBoost</td><td rowspan=1 colspan=1>81.09</td><td rowspan=1 colspan=1>85.63</td><td rowspan=1 colspan=1>86.34</td><td rowspan=1 colspan=1>85.66</td></tr><tr><td rowspan=1 colspan=1>DT</td><td rowspan=1 colspan=1>77.49</td><td rowspan=1 colspan=1>83.30</td><td rowspan=1 colspan=1>83.23</td><td rowspan=1 colspan=1>83.27</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>77.05</td><td rowspan=1 colspan=1>82.71</td><td rowspan=1 colspan=1>83.14</td><td rowspan=1 colspan=1>82.78</td></tr><tr><td rowspan=1 colspan=1>KNN</td><td rowspan=1 colspan=1>76.09</td><td rowspan=1 colspan=1>81.73</td><td rowspan=1 colspan=1>82.55</td><td rowspan=1 colspan=1>82.04</td></tr><tr><td rowspan=1 colspan=1>MLP</td><td rowspan=1 colspan=1>74.51</td><td rowspan=1 colspan=1>84.43</td><td rowspan=1 colspan=1>81.82</td><td rowspan=1 colspan=1>78.48</td></tr><tr><td rowspan=1 colspan=1>GB</td><td rowspan=1 colspan=1>73.73</td><td rowspan=1 colspan=1>83.47</td><td rowspan=1 colspan=1>81.22</td><td rowspan=1 colspan=1>77.78</td></tr></table>

A high-performing classifier is a prerequisite for the framework. As Table 1 shows, XGBoost outperforms all baselines on every metric, achieving the highest accuracy of 81.09% and an F1-Score of 85.66%, and is therefore selected as our core classification engine.

## 4.3 Procedure

A total of 57 participants are recruited and evenly distributed across three conditions: a no-explanation Control, in which the alert is audited from the raw IDS verdict alone without any XAI support, the Dashboard XAI, and the Conversational XAI interface (examples in Fig. 2). To support the technical validity of the evaluation, participants were required to have prior experience with AI or machine learning concepts.

Each operator audited the same set of 20 alerts for which the IDS issued a verdict to validate. To model a realistic yet fallible assistant, we manually curated this set so that the AI was correct on 14 alerts and wrong on six. We further decoupled the model’s reported confidence from its correctness: using the trained XGBoost classifier, we interleaved high-, low-, and random-confidence predictions, including confidently wrong and hesitantly correct cases, so that operators could not fall back on a simple confidence heuristic and instead had to engage with the explanation itself.

The procedure consists of the following steps:

1. Consent and Profiling: Participants provide informed consent and complete a demographic survey to verify their technical expertise.

2. Contextual Onboarding: Participants are briefed on the scenario, highlighting the challenge of manual analysis and the need for XAI support in root cause identification.

3. Training: A brief tutorial and practice trial are conducted to familiarize participants with the navigation and functions of their assigned XAI interface.

4. Task Execution: Participants utilize the tool to investigate the flagged event, with the specific goal of validating the alert’s reliability and explaining the intrusion’s root cause.

5. Post-task Evaluation: Finally, participants complete a knowledge test and subjective questionnaires to assess Operator Understanding, Explanation Utility, and Operator Trust based on our defined metrics.

## 4.4 Comparative Analysis

The subjective results for Operator Understanding, Explanation Utility, and Operator Trust are summarized in Fig. 3: descriptively, the Conversational interface received slightly higher mean ratings than the Dashboard across the three subjective dimensions. In Explanation Utility, the Conversational interface scored 4.10 against the Dashboard’s 3.90; participants reported that natural language interaction eased information filtering and synthesis. Operator Trust was descriptively and only marginally higher for the Conversational interface (4.23 versus 4.20).

Operator Understanding was also slightly higher in the Conversational condition (3.78 versus 3.68 for the Dashboard), but the narrower margin suggests that the agent’s greater perceived helpfulness did not translate into deeper comprehension of the underlying IDS logic.

This dissociation between perception and comprehension surfaced directly in operator behavior, summarized in Table 2. Relative to the no-explanation Control, both interfaces sharply raised reliance, with the agreement fraction climbing from 0.74 to 0.86 (Dashboard) and 0.89 (Conversational). Yet this added reliance was not matched by appropriate reliance: RAIR improved only modestly (0.35 to 0.50 and 0.48), while appropriate self-reliance (RSR) decreased substantially from 0.57 to 0.29 and 0.11, reaching its lowest value under the Conversational interface. Kruskal–Wallis tests indicated statistically significant overall diferences across conditions for agreement fraction $( H = 3 3 . 6 6 , p < . 0 0 1 )$ , switch fraction $( H = 1 9 . 1 4 , p < . 0 0 1 )$ , RAIR $( H = 1 1 . 0 1 , p = . 0 0 4 )$ , RSR $( H = 3 8 . 2 6 , p < . 0 0 1 )$ and decision accuracy $( H = 9 . 0 9 , p = . 0 1 1 )$ . Both interfaces also improved decision accuracy relative to the Control (0.62 to 0.74 and 0.71). Critically, however, the Conversational interface, despite eliciting the highest agreement and the lowest self-reliance, did not achieve higher accuracy than the Dashboard; descriptively, its accuracy was slightly lower. This pattern is consistent with the concern that the additional reliance elicited by the conversational modality was not converted into better decisions, plausibly because some participants switched from correct initial judgments to AI-aligned decisions even when the system was incorrect. Following [27], the combination of high agreement and low self-reliance, without a corresponding accuracy advantage over the Dashboard, provides evidence consistent with an elevated risk of over-reliance in the Conversational condition.

![](images/4b22ad022026d4c20c19068c89b15dad4aee220724b78dc73aa7ca9d64b58883.jpg)

Fig. 2. Conversational XAI when analyzing a DoS attack.  
![](images/111bc92f0229b2f5dd78f5bf40f567a8a9272e3c15b9c2b88a33dd148296f294.jpg)  
Fig. 3. Comparative evaluation of user perception metrics between the Dashboard and Conversational XAI interfaces. Error bars represent the 95% confidence interval.

Table 2. Behavioral reliance and appropriate-reliance metrics by condition (mean±SD); H and p from Kruskal–Wallis tests.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Control</td><td rowspan=1 colspan=1>Dashboard</td><td rowspan=1 colspan=1>Conversational</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>p</td></tr><tr><td rowspan=1 colspan=1>Agreement Fraction</td><td rowspan=1 colspan=1> $0 . 7 4 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1> $0 . 8 6 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1>0.89 ±0.11</td><td rowspan=1 colspan=1>33.66</td><td rowspan=1 colspan=1>&lt; .001</td></tr><tr><td rowspan=1 colspan=1>Switch Fraction</td><td rowspan=1 colspan=1> $0 . 3 1 \pm 0 . 3 4$ </td><td rowspan=1 colspan=1>0.57 ±0.41</td><td rowspan=1 colspan=1>0.57 ±0.41</td><td rowspan=1 colspan=1>19.14</td><td rowspan=1 colspan=1>&lt; .001</td></tr><tr><td rowspan=1 colspan=1>RAIR</td><td rowspan=1 colspan=1>0.35 ±0.39</td><td rowspan=1 colspan=1>0.50 ±0.44</td><td rowspan=1 colspan=1>0.48 ±0.45</td><td rowspan=1 colspan=1>11.01</td><td rowspan=1 colspan=1>.004</td></tr><tr><td rowspan=1 colspan=1>RSR</td><td rowspan=1 colspan=1>0.57 ±0.46</td><td rowspan=1 colspan=1>0.29 ±0.44</td><td rowspan=1 colspan=1>0.11 ±0.29</td><td rowspan=1 colspan=1>38.26</td><td rowspan=1 colspan=1>&lt; .001</td></tr><tr><td rowspan=1 colspan=1>Decision Accuracy</td><td rowspan=1 colspan=1> $\left| 0 . 6 2 \pm 0 . 1 3 \right|$ </td><td rowspan=1 colspan=1> $0 . 7 4 \pm 0 . 1 1$ </td><td rowspan=1 colspan=1>0.71 ±0.10</td><td rowspan=1 colspan=1>9.09</td><td rowspan=1 colspan=1>.011</td></tr></table>

## 4.5 Discussion

The results point to a potential trade-of in high-dimensional intrusion detection. One possible explanation is that the Conversational Interface made it easier to navigate many cyber-physical features, but its coherent natural-language responses may also have encouraged participants to accept explanations without fully inspecting the underlying evidence. The Dashboard’s raw telemetry and packet logs may encourage more direct inspection of the data. By contrast, the Llama-powered agent presents anomalies as coherent narratives, which may make uncertainty less salient to users unless uncertainty cues are explicitly shown. The reliance measures are consistent with the concern that participants may accept plausible-sounding diagnoses without suficiently verifying the raw feature values.

We therefore argue that smooth interaction may be insuficient for highstakes UAV defense; future designs may benefit from Cognitive Forcing Functions that encourage active verification. Rather than passively displaying an answer, efective XAI interfaces must impede rapid agreement when model confidence is low, for instance by requiring operators to correlate the AI’s claim with a specific trend in the PDP or SHAP plots before confirming a countermeasure, transforming the operator from a passive consumer into an active auditor.

## 5 Conclusion

This study presents a comparative evaluation of Dashboard and Conversational XAI interfaces for UAV intrusion detection. Our empirical results suggest a trade-of in this human-AI auditing setting. While the Conversational interface was perceived as useful and may reduce the efort required to access relevant explanations, our results suggest a risk of increased over-reliance, as indicated by lower RSR under the conversational condition: agreement with AI advice increased and appropriate self-reliance was lower, yet the higher reliance it elicited did not translate into better decision accuracy than the Dashboard. Future work prioritizes the implementation of cognitive forcing functions to encourage highstakes security decisions to be grounded in evidence verification rather than in the fluency of natural-language explanations.

## Acknowledgement

This research was supported by National Economics University under grant number NEU1-2025.02, and by Vietnam National Foundation for Science and Technology Development (NAFOSTED) under grant number 102.05-2025.57.

## References

1. Al-Turjman, F., Abujubbeh, M., Malekloo, A., Mostarda, L.: UAVs assessment in software-defined IoT networks: An overview. Computer Communications (2020)

2. Lai, V., Chen, C., Smith-Renner, A., Liao, Q.V., Tan, C.: Towards a science of Human-AI decision making: An overview of design space in empirical humansubject studies. In: FAccT 23. Chicago, IL, USA (2023)

3. Hassler, S., Mughal, U., Ismail, M.: Cyber-physical intrusion detection system for unmanned aerial vehicles. IEEE Trans. Intell. Transp. Syst. (01 2023)

4. Zhang, X., Tan, S., Koch, P., Lou, Y., Chajewska, U., Caruana, R.: Axiomatic interpretability for multiclass additive models (2019)

5. Slack, D., Hilgard, S., Singh, S., Lakkaraju, H.: Reliable post hoc explanations: Modeling uncertainty in explainability (2021)

6. Vasconcelos, H., Jörke, M., Grunde-McLaughlin, M., Gerstenberg, T., Bernstein, M.S., Krishna, R.: Explanations can reduce overreliance on AI systems during decision-making. PACMHCI (CSCW2) (2023)

7. Liao, Q.V., Gruen, D., Miller, S.: Questioning the AI: Informing design practices for explainable AI user experiences. In: CHI EA 2020. New York, NY, USA (2020)

8. Yang, W., Le, H., Laud, T., Savarese, S., Hoi, S.C.H.: OmniXAI: A library for explainable AI (2022)

9. Slack, D., Krishna, S., Lakkaraju, H., Singh, S.: Explaining machine learning models with interactive natural language conversations using TalkToModel. Nature Machine Intelligence (7) (2023)

10. Bansal, G., Wu, T., Zhou, J., Fok, R., Nushi: Does the whole exceed its parts? The efect of AI explanations on complementary team performance. In: CHI 2021. New York, NY, USA (2021)

11. Balayn, A., Yurrita, M., Rancourt, F., Casati, F., Gadiraju, U.: Unpacking trust dynamics in the LLM supply chain: An empirical exploration to foster trustworthy LLM production & use. In: CHI 2025. New York, NY, USA (2025)

12. He, G., Bharos, A., Gadiraju, U.: To err is AI! debugging as an intervention to facilitate appropriate reliance on AI systems. In: ACM HT 24. New York, NY, USA (2024)

13. Robbemond, V., Inel, O., Gadiraju, U.: Understanding the role of explanation modality in AI-assisted decision-making. In: UMAP 22). New York, NY, USA (2022)

14. Chiang, C.W., Yin, M.: Exploring the efects of Machine Learning literacy interventions on laypeople’s reliance on Machine Learning models. In: IUI 22. New York, NY, USA (2022)

15. Selbst, A.D., Powles, J.: “meaningful information” and the right to explanation. In: FAT 2018. PMLR (2018)

16. Ribeiro, M.T., Singh, S., Guestrin, C.: "why should I trust you?": Explaining the predictions of any classifier. In: 22nd ACM SIGKDD (2016)

17. Yin, K., Neubig, G.: Interpreting language models with contrastive explanations. In: EMNLP 2022. Abu Dhabi, United Arab Emirates (2022)

18. Nauta, M., Trienes, J., Pathak, S., Nguyen, E., Peters, M., Schmitt: From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI. ACM Computing Surveys 55(13s) (2023)

19. Ehsan, U., Riedl, M.O.: Human-centered explainable AI: Towards a reflective sociotechnical approach. In: HCIC 2020 (2020)

20. Rong, Y., Leemann, T., Nguyen, T.T., Fiedler, L., Qian: Towards human-centered explainable AI: A survey of user studies for model explanations. IEEE TPAMI 46(5) (2023)

21. Jannach, D., Manzoor, A., Cai, W., Chen, L.: A survey on conversational recommender systems. ACM CSUR 54(5) (2021)

22. Shen, H., Huang, C.Y., Wu, T., Huang, T.H.K.: ConvXAI: Delivering heterogeneous AI explanations via conversations to support Human-AI scientific writing. arXiv preprint arXiv:2305.09770 (2023)

23. Friedman, J.H.: Greedy function approximation: A gradient boosting machine. Annals of Statistics (2001)

24. Wexler, J., Pushkarna, M., Bolukbasi, T., Wattenberg, M., Viegas, F., Wilson, J.: The what-if tool: Interactive probing of machine learning models. IEEE Trans. Vis. Comput. Graph (2019)

25. Schmude, T., Koesten, L., Möller, T., Tschiatschek, S.: On the impact of explanations on understanding of algorithmic decision-making. In: ACM FAccT 23. New York, NY, USA (2023)

26. Jacovi, A., Bastings, J., Gehrmann, S., Goldberg, Y., Filippova, K.: Diagnosing AI explanation methods with folk concepts of behavior. In: FAccT 23). New York, NY, USA (2023)

27. Schemmer, M., Hemmer, P., Kühl, N., Benz, C., Satzger, G.: Should i follow AIbased advice? measuring appropriate reliance in Human-AI decision-making. In: trAIt 22 (2022)
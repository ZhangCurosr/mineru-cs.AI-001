# Communicating Credit Risk with Large Language Models: Evaluation of Explanations from Standard and Alternative Data-Based Models

Sahab Zandi<sup>a</sup>, Noah Kostesku<sup>b</sup>, Christophe Mues<sup>c,d</sup>, María Óskarsdóttir<sup>e,f,∗</sup>, Cristián Bravo<sup>a</sup>

<sup>a</sup>Department of Statistical and Actuarial Sciences, Western University, London, Canada <sup>b</sup>Department of Computer Science, Western University, London, Canada

<sup>c</sup>University of Southampton Business School, University of Southampton, Southampton, United Kingdom <sup>d</sup>Centre for Operational Research, Management Sciences and Information Systems, University of Southampton, Southampton, United Kingdom

<sup>e</sup>School of Mathematical Sciences, University of Southampton, Southampton, United Kingdom <sup>f</sup>Department of Computer Science, Reykjavik University, Reykjavik, Iceland

## Abstract

Credit decisioning is a high-stakes task in which model outputs must be accurate and explainable to support compliant decisions. Although modern credit risk models such as eXtreme Gradient Boosting (XGBoost) and Graph Neural Networks (GNNs) improve predictive performance, their explanations are often too technical for stakeholders, such as auditors and customers, creating communication gaps that can shape approvals, denials, and fairness judgments. We examine whether Large Language Models (LLMs) can serve as explanation layers that translate post-hoc explanation artefacts into stakeholder-appropriate risk narratives. Using Freddie Mac single-family loan-level data, we develop three pipelines: standard tabular (XGBoost + SHAP), and two models with alternative data, a pure network-based model (GNN + GNNExplainer), and a bimodal mode (combining tabular and network data). We generate narratives with three LLM configurations: a small fine-tuned LLM (Gemma 3 4B), a large fine-tuned LLM (DeepSeek R1 70B), and a zero-shot commercial LLM (Gemini 2.5). Explanation quality is evaluated through automated checks across all pipelines and a human study of bimodal explanations comparing credit risk professionals and non-professionals on eight decision-relevant dimensions. We have three main findings. First, the pipeline accounts for higher variance in evidence-grounding scores than the language model, meaning that the binding constraint on explanation quality is the evidence representation, not the model used. Second, the explanation narratives reliably name the influential factors but are less reliable when stating the direction of influence, which may be consequential for adverse-action communication. Finally, professionals apply stricter evidentiary standards than non-professionals. We discuss

implications for the governance of risk models, including deployment considerations and the value of domain-aligned LLMs in regulated credit settings.

Keywords: Risk communication, Model interpretability, Explanation quality, Credit scoring, Large language models

## 1. Introduction

Credit decisioning is a high-stakes risk assessment: mortgage and consumer lending decisions allocate capital under uncertainty and can materially afect household welfare, institutional losses, and perceptions of fairness (Doumpos et al., 2023; Tigges et al., 2024). Predictive accuracy alone is therefore insuficient; institutions must also justify model-driven decisions in ways that support oversight, customer communication, and internal governance (De Bock et al., 2024). Regulatory and policy frameworks reinforce this requirement through Basel-style risk governance, data protection rules such as GDPR in the EU, and adverse-action communication requirements in the U.S. (Basel Committee on Banking Supervision, 2011; Consumer Financial Protection Bureau, 2022; European Union, 2016). As a result, the quality of the explanation becomes part of the risk itself: poorly grounded or poorly communicated explanations can contribute to miscalibrated trust, inconsistent approvals/denials, operational disputes between the validation and business teams, and compliance exposure (Fischhof, 1995).

At the same time, credit risk modelling has moved toward a broader set of complex machine learning methods that can improve predictive performance but reduce transparency. These include tree-based ensembles and deep learning models (Hayashi, 2022; Shi et al., 2022). Recent research also suggests that credit risk may exhibit a relational structure (Óskarsdóttir and Bravo, 2021). Borrowers linked by shared contextual factors, such as geography or lender practices, can exhibit correlated risk patterns that traditional tabular models fail to capture. This has motivated network-based approaches, especially GNNs, for credit risk modelling (Zandi et al., 2025).

Post-hoc explanation methods help surface model drivers, for example, SHAP for tabular models (Lundberg and Lee, 2017) and GNNExplainer for GNNs (Ying et al., 2019). However, their outputs are typically technical artefacts (feature attributions, node/edge importance scores, local subgraphs) that may not translate into the concise, audience-appropriate narratives needed by stakeholders such as risk professionals, auditors, regulators, and afected customers. This creates a persistent risk-communication gap: models may be explainable in principle, yet the explanations are not consistently usable in practice.

LLMs ofer a potential bridge. By converting structured input into coherent natural-language narratives, LLMs can act as a layer that translates the explanation output into stakeholder-appropriate text. However, in regulated environments, the central issue is not only understandability but also whether narratives remain faithful and relevant to decisions (Turpin et al., 2023).

This paper presents a unified framework that converts explanation artefacts (SHAP and GN-NExplainer output) into natural-language narratives across tabular, network, and bimodal evidence. We use the framework to study LLM-based explanation layers for credit risk models through three pipelines that separate prediction, explanation generation, and narrative generation. Using Freddie Mac single-family loan-level data (Federal Home Loan Mortgage Corporation, 2022), we develop (i) a tabular-based pipeline (XGBoost + SHAP), (ii) a network-based pipeline (GNN + GNNExplainer) and (iii) a bimodal pipeline that combines both types of evidence. We compare three LLM configurations representing realistic deployment choices: a small domain-fine-tuned open LLM (Gemma 3 4B), a large domain-fine-tuned open LLM (DeepSeek R1 70B), and a general-purpose zero-shot commercial LLM (Gemini 2.5).

Because explanations in this domain serve diferent stakeholders, we distinguish professiona and non-professional evaluators. Prior work in Explainable Artificial Intelligence (XAI) and risk communication shows that explanation needs vary by audience, and that NCRP readers may respond diferently to the frame, tone, and format of the explanation (Dieckmann et al., 2009; Doshi-Velez and Kim, 2017; Miller, 2019; Peters, 2008). Given this, we combine automated checks, which provide scalable guardrails for fluency and evidence alignment, with human ratings, which capture whether explanations are perceived as trustworthy, communicable, and usable in realistic decision settings.

We make three contributions. First, we decompose explanation quality in a credit risk decision-support system into its sources, namely evidence modality, language model, and evaluator, and show that the first dominates the second by roughly an order of magnitude, which relocates the design problem from model selection to evidence representation. Second, we identify a systematic asymmetry between what generated explanations cover and what they get directionally right, and argue that directional error, not omission, is the governance-relevant failure mode in regulated credit communication. Third, we show that professional and non-professional stakeholders apply diferent acceptance criteria to the same narrative, and that rater heterogeneity dominates system diferences in human evaluation. From this we derive design requirements for evaluating explanation layers, responding to the evaluation agenda set out for explainable AI in operational research by De Bock et al. (2024). We address the following research questions.

• RQ1: How does explanation modality (tabular, network, bimodal) influence automated measures of explanation quality across LLM configurations?

• RQ2: Do LLM-generated explanations difer in perceived explanation quality between professionals and non-professionals?

• RQ3: How do the LLM choice and fine-tuning status relate to the quality of the explanation?

• RQ4: Which textual properties are associated with a higher human-rated explanation quality?

The remainder of the paper proceeds as follows: Section 2 reviews related work and regulatory context; Section 3 details the three-stage explanation architecture, data, models, and evaluation design; Section 4 reports findings relevant to the research questions. Section 5 discusses implications for credit risk communication, the mechanisms underlying the observed efects, and practical and policy considerations. Section 6 presents conclusions and directions for future research.

## 2. Background and Related Work

## 2.1. Explainability in credit risk modelling

Interpretability has long been central in credit risk, where traditional models such as logistic regression and scorecards were favoured in part because they are transparent by design (Siddiqi, 2017; Thomas, 2000). As institutions increasingly adopt higher-capacity machine learning models, predictive performance can improve, but transparency often decreases (Addo et al., 2018; Breeden, 2020; Bücker et al., 2022; Lessmann et al., 2015). Post-hoc explainers such as SHAP and LIME have therefore become common tools for summarising feature influence in complex models, including credit-scoring applications (Bracke et al., 2019; Bussmann et al., 2021; Lundberg and Lee, 2017; Misheva et al., 2021; Ribeiro et al., 2016).

Importantly, there has been a call for stakeholder-appropriate explanations and evaluation standards (De Bock et al., 2024), highlighted in part by explanation stability of both SHAP and LIME which both cost-sensitive learning and class imbalance can afect (Ballegeer et al., 2025; Chen et al., 2024).

However, multiple surveys and conceptual frameworks emphasise that the quality of the explanation depends on the stakeholder and that technical transparency does not guarantee the usefulness of communication (Doshi-Velez and Kim, 2017; Guidotti et al., 2018). In regulated settings, a further concern is that post-hoc explanations can be misleading if they are interpreted causally or if they omit important context. Recent calls for inherently interpretable models highlight these concerns and argue that some domains may require models that are transparent by design (Rudin, 2019; Rudin and Shaposhnik, 2023; Tu and Wu, 2025).

For the purposes of this paper, we treat SHAP-style feature attributions as widely adopted, theoretically grounded explanation artefacts for tabular models. Our focus is not to propose new attribution methods, but to evaluate how such artefacts can be translated into stakeholder-appropriate narratives without sacrificing evidence grounding.

## 2.2. Graph-based credit risk modelling and explainability

Relational modelling has gained traction in credit risk because borrower outcomes can exhibit dependence structures linked to shared geography, lender practices, and other common contextual attributes. Network-based approaches allow such relationships to be explicitly represented and have been shown to improve credit risk modelling performance in settings where borrowers are connected through common contextual factors (Óskarsdóttir and Bravo, 2021). GNNs provide a flexible framework to take advantage of such relational structures (Wu et al., 2020), and recent studies report predictive gains when relational information is incorporated into credit risk models (Das et al., 2023; Liu et al., 2024b; Wang et al., 2021, 2022; Zandi et al., 2025). However, interpretability becomes more complex: predictions may depend on both node attributes (borrower/loan features) and graph structure (neighbourhood composition, cluster efects, and message passing).

Alternative GNN explainers include surrogate- and counterfactual-based methods (Huang et al., 2023; Lucic et al., 2022); we use GNNExplainer as a widely adopted instance-level baseline (Kakkad et al., 2023). Converting graph explanations into narratives carries a risk of drifting beyond what structural evidence supports; our approach therefore emphasises structured evidence blocks and explicit constraints to reduce over-interpretation.

## 2.3. Risk communication, governance, and LLM-based explanation layers

Narrative explanations can improve stakeholder understanding relative to raw attribution output (Martens et al., 2025), but persuasive language can outpace factual support, motivating evaluation of both readability and evidence grounding. Modern credit models are embedded in governance processes where explanations serve as controls (De Jongh et al., 2017; Souza, 2023): when an LLM delivers explanations, prompting rules, output constraints, and versioning can materially afect downstream interpretations even when the predictive model is unchanged. A fluent but weakly grounded explanation increases operational and compliance risk by creating a false sense of evidentiary adequacy. Together, Basel-style governance, GDPR, and U.S. adverse action requirements motivate evaluation dimensions beyond fluency: fidelity and trustworthiness as proxies for audit readiness, and communicability and usability as proxies for operational and customer-facing constraints (Basel Committee on Banking Supervision, 2011; Consumer Financial Protection Bureau, 2022; European Union, 2016)

LLMs can verbalise technical artefacts like feature attributions, rule lists, and counterfactuals into audience-appropriate narratives (Bilal et al., 2025; Brown et al., 2020; Martens et al., 2025), which is especially attractive in regulated credit settings where explanations must support review, validation, and compliance documentation. However, the same flexibility creates risk: generated narratives can be fluent and persuasive even when they overstate, omit, or mischaracterise the underlying evidence (Turpin et al., 2023). In regulated settings, this makes fidelity, constraint design, and stakeholder-appropriate usefulness the central criteria for evaluating LLM-mediated explanations (Turpin et al., 2023). Explanation quality is also context-dependent: professionals prioritise evidentiary adequacy and traceability whereas non-professionals weight clarity and completeness (Doshi-Velez and Kim, 2017; Miller, 2019), motivating cohort-based evaluation linked to decision relevance and governance use cases. Building on these insights, we pair cohort-based human evaluation with automated checks of evidence grounding to assess whether explanation layers improve decision-relevant understanding while preserving fidelity and auditability across realistic deployment choices.

## 3. Methodology

This section outlines the framework used to develop and evaluate our explanation pipelines. We describe data processing, network construction, predictive models, post-hoc explainers, LLM-based narration architecture, prompt design, fine-tuning, and human, automated, and linguistic evaluation procedures.

## 3.1. Overview of explanation architecture

Our framework is organised as separable stages (prediction, explanation generation, and narrative generation) to support the evaluation of how explanation artefacts and LLM configuration afect perceived explanation quality. In all pipelines, the LLM is used as a verbalisation layer: it receives structured evidence blocks derived from SHAP or GNNExplainer (plus lightweight normalisation and templating) and produces a constrained narrative explanation. The LLM does not alter predictive outputs or raw attribution values.

To make comparisons interpretable, we separate what varies between conditions from what is kept constant. Across all configurations, we hold fixed the loan cases, evidence-block schema, and prompting rules and output constraints (e.g., requirements to reference only evidence-block content and avoid naming explanation algorithms in borrower-facing text). What varies is the evidence modality (tabular, network, or bimodal) and the LLM configuration (Gemma 3, DeepSeek R1, Gemini 2.5).

## 3.2. Dataset

We used the single-family loan-level dataset from the Federal Home Loan Mortgage Corporation (FHLMC; Freddie Mac) (Federal Home Loan Mortgage Corporation, 2022), focusing on loans originated during 2015–2016, in line with previous credit risk modelling work on the same data source, although for diferent years (Zandi et al., 2025). For each origination cohort, performance records are observed up to a fixed cut-of date, and we retain loans with at least six months of servicing history to reduce bias from insuficient outcome observation while preserving adequate sample coverage. To maintain temporal separation, the 2015 origination cohort is used for model development, and the 2016 cohort is held out for evaluation. Standard preprocessing steps include missing-value imputation, categorical recoding, and min–max scaling. A summary of the borrowerand loan-level features used is provided in Appendix A.

## 3.3. Network construction

We construct static networks from origination-stage data aggregated over fixed time windows: 6-month periods (January to June 2015 for training and January to June 2016 for testing) and a 1-month period (July 2016) for explanation generation. Graph construction uses only origination-stage attributes to maintain temporal separation and avoid outcome leakage. Within each period, nodes correspond to individual loans, and edges connect loans that share a zip code prefix (regional grouping), a common lender, or both, forming fully connected bidirectional cliques. Edge types are one-hot encoded to distinguish connection sources. Node labels use the same binary default definition described in the feature table in Appendix A.

## 3.4. Predictive model training

For the tabular-based pipeline, we train an XGBoost classifier using the 2015 origination cohort and evaluate it on the held-out 2016 cohort. The predictive feature set consists of the preprocessed borrower and loan variables from the origination-stage (described in the feature table in Appendix A, excluding identifiers, temporal metadata and grouping variables not intended for tabular prediction. Hyperparameters are selected by grid search with 5-fold cross-validation (ROC–AUC criterion); the classification threshold is chosen on a validation split by maximising F1.

For the network-based pipeline, we train a Graph Attention Network (GAT) (Veličković et al., 2018) on the 2015 training graph and evaluate it on the held-out 2016 test graph. The model operates on node features derived from origination-stage borrower and loan attributes together with edge attributes that encode connection type and group-size information. The architecture uses stacked attention-based message-passing layers with edge-aware transformations, followed by a binary node-level prediction head for default classification. Training uses neighbour sampling; class imbalance is addressed by approximate 1:5 negative subsampling. Model selection uses validation ROC–AUC with early stopping and F1-based threshold selection.

## 3.5. Explainability techniques

We apply two post-hoc explanation methods aligned with our predictive models: SHAP for tree-based models (Lundberg and Lee, 2017) and GNNExplainer for GNNs (Ying et al., 2019). SHAP produces feature-level attribution scores for XGBoost predictions and is widely used in credit risk modelling due to its theoretical grounding and compatibility with tree-based ensembles. GN-NExplainer identifies subgraph structures and node features driving GNN decisions and provides instance-level explanations directly from the trained model. A detailed formulation of both methods is provided in Appendix B.

## 3.6. Large language models

LLMs in this framework convert structured output from SHAP and GNNExplainer into human-readable narratives. A lightweight post-processing layer contextualises the explainer outputs (e.g., ranking features, mapping values to percentiles, and generating short descriptive phrases), providing the LLM with structured, enriched inputs for narrative generation. We compare three configurations spanning size, fine-tuning status, and deployment mode (Table 1): Gemma 3 4B (small, fine-tuned), DeepSeek R1 70B (large, fine-tuned), and Gemini 2.5 (undisclosed size, zero-shot API).

Table 1: LLMs used for narrative generation.
<table><tr><td>LLM</td><td>Size</td><td>Fine-tuned</td><td>Deployment Mode</td><td>Source</td></tr><tr><td>Gemma 3</td><td>Small (4B)</td><td>Yes</td><td>Open-source</td><td>Google DeepMind</td></tr><tr><td>DeepSeek R1</td><td>Large (70B)</td><td>Yes</td><td>Open-source</td><td>DeepSeek AI</td></tr><tr><td>Gemini 2.5</td><td>Undisclosed</td><td>No</td><td>API-based</td><td>Google</td></tr></table>

## 3.7. Proposed explanation pipelines

We develop three explanation pipelines aligned with diferent evidence modalities (tabular-based, network-based, and bimodal), each composed of separable stages for model training, explanation generation, and LLM-based narration. This separation enables analysis of how explanation strategies influence interpretability between evidence types. The architecture of the three pipelines, shown in Figure 1, can be summarised as follows:

![](images/1d790e1476178bae5feb167c74b4aeb5e4ff9e3a5011710449b9a8faf80bee5f.jpg)  
(a) Tabular-based pipeline

![](images/bfb46a7598ba5623ba89935841c701b05e4fa51731cd39a3bb7451ec09311cc5.jpg)  
(b) Network-based pipeline

![](images/15fc6ad37755142132759e9170245903b7cfa7b0ef117937b3a0c7baf2a63dd0.jpg)  
(c) Bimodal pipeline  
Figure 1: The architecture of the three explanation pipelines.

• Tabular-based pipeline: An XGBoost model generates default-risk predictions from loan-level tabular features, and SHAP produces per-instance feature attributions. Before prompting the LLM, a lightweight rule-based post-processing layer converts the top-k SHAP-ranked features into structured textual inputs by mapping selected feature values to hand-crafted plain-language descriptors and qualitative ranges, while preserving the directional contribution indicated by the SHAP scores. The LLM then uses the predicted outcome together with these enriched feature summaries to produce a constrained narrative explanation (Figure 1a).

• Network-based pipeline: A GAT generates node-level default-risk predictions by combining loan attributes with message passing over the constructed loan graph. GNNExplainer then produces a local explanation consisting of influential node features and a compact explanatory subgraph. The LLM receives a structured summary of this evidence (e.g., salient neighbourhood composition and edge-type signals) and generates a narrative explanation (Figure 1b).

• Bimodal pipeline: XGBoost and GAT are run in parallel, and SHAP and GNNExplainer explanations are produced for the same instance. The LLM is provided with both evidence blocks and instructed to synthesise tabular and relational signals into a single narrative while preserving evidence provenance and avoiding unsupported causal claims (Figure 1c).

## 3.8. Prompt design for explanation generation

To transform model evidence into narratives, we design pipeline-specific prompt templates, each combining a shared system prompt with a tailored user prompt. The system prompt enforces objectivity, minimises bias, and establishes stylistic consistency. It avoids persona cues or anthropomorphic framing, which prior work shows can reduce accuracy and increase stylistic drift (Gupta et al., 2023; Liu et al., 2024a).

The user prompt adapts to each pipeline using three strategies: (1) instructional step-by-step reasoning (Wei et al., 2022), (2) task decomposition via explicit subgoals and constrained output structure, and (3) counterfactual-style prompts to elicit improvement-orientated alternatives, drawing on the broader notion of counterfactual reasoning (Pearl, 2009) and recent work on prompting LLMs for counterfactual generation (Li et al., 2024). These strategies are realised through explicit prompt components (ordered reasoning steps, guided subgoals, counterfactual prompts, and explicit length/format constraints) rather than free-form generation. Within our experimental setup, Gemma 3 and Gemini 2.5 were prompted using the same prompt templates, while DeepSeek R1 required minor calibration to account for LLM-specific alignment and formatting diferences (DeepSeek AI, 2024). Table 2 summarises the prompt components used in each pipeline. Expanded prompt examples are provided in Appendix C.

Table 2: Prompt design components for each explanation pipeline.
<table><tr><td>Pipeline</td><td>Prompt Components</td></tr><tr><td>Tabular- based</td><td>Predicted outcome; top-k SHAP feature attributions with descriptions; in- structional step-by-step reasoning; guided subgoal structure; borrower-level counterfactual prompt; output format constraint.</td></tr><tr><td>Network- based</td><td>Predicted outcome;network connectivity analysis;instructional step-by-step reasoning for structural and relational reasoning; guided subgoal structure; network-level counterfactual prompt; output format</td></tr><tr><td>Bimodal</td><td>constraint. Predicted outcome; top-k SHAP feature descriptions; GNNExplainer sub- graph descriptions; combined instructional reasoning linking borrower and relational characteristics; combined borrower-level and network-level coun- terfactual prompt; disambiguation hint for conflicting contributions; output format constraint.</td></tr></table>

## 3.9. Fine-tuning Configurations

The final synthetic supervision dataset includes examples across all three types of pipelines (tabular-based, network-based, and bimodal). We fine-tune Gemma 3 and DeepSeek R1 using parameter-eficient supervised instruction tuning with LoRA adapters applied to quantised base models (Dettmers et al., 2023). In both cases, the supervision data are formatted as system–user–assistant conversations derived from pipeline-specific evidence, and the training workflow applies chat-template formatting before tokenisation. Gemma 3 is fine-tuned using an 8-bit quantised base model with fp16 training, cosine learning-rate scheduling, and LoRA adapters applied to attention and feed-forward layers. DeepSeek R1 uses the same overall adaptation strategy but with more conservative optimisation settings, including a smaller batch size per-device, higher gradient accumulation, shorter maximum sequence length and gradient checkpointing to accommodate its larger footprint. Figure 2 summarises the fine-tuning workflow.

## 3.10. Evaluation design and procedure

We evaluate explanation quality using a mixed design combining automated checks with human-centred assessment: automated checks identify systematic issues at scale, while human evaluation captures how explanations are interpreted in realistic decision settings. We therefore treat automated verifications as guardrails rather than substitutes for human judgement.

For automated components, we use Claude Sonnet 4.6 (Anthropic, 2024) as an external judge, selected to avoid circularity with ChatGPT-4o (used for fine-tuning supervision) and Gemini 2.5 (one of the evaluated LLMs).

![](images/86c9e432af3a8080316af9c47f135adcf8e2bac85bb0a63c8a025897e1f07009.jpg)  
Figure 2: Workflow for fine-tuning Gemma 3 and DeepSeek R1 using LoRA-based instruction tuning on 8-bit quantised base models. SHAP and GNNExplainer outputs are paired with ChatGPT-4o-generated explanations to create the synthetic supervision dataset.

Human evaluation was conducted via an online survey distributed from August 27 to October 1, 2025. Each participant evaluated bimodal explanations from all three LLMs across up to three rounds (mandatory first round plus two optional), with explanation order randomised within rounds and no repeated instances per participant. Participants self-identified as Credit Risk Professionals (CRPs) or Non-Credit Risk Professionals (NCRPs) and received audience-appropriate survey materials accordingly.

## 3.11. Evaluation metrics

Human evaluation captures how explanations are perceived by diferent stakeholder groups in realistic decision settings. For human evaluation, we operationalise the quality of the explanation along eight perceived dimensions rated on a 5-point Likert scale (1 = Strongly Disagree, 2 = Slightly Disagree, 3 = Neutral, 4 = Slightly Agree, 5 = Strongly Agree). These dimensions follow the human-centred evaluation view that explanation quality is context-dependent and should capture understanding, trust, and actionability (Doshi-Velez and Kim, 2017; Miller, 2019). To support a risk-oriented interpretation, we treat the metrics as proxies for decision relevance in regulated credit workflows: trustworthiness and insightfulness reflect evidentiary adequacy for review and challenge; communicability reflects suitability for cross-team and customer-facing communication; and usability reflects whether the explanation could plausibly support an approval/denial workflow under operational constraints. The wording of the survey items difers slightly between CRPs and NCRPs because the two groups were presented with audience-appropriate phrasing: CRPs received domain-specific terminology aligned with credit risk workflows, whereas NCRPs received parallel plain-language wording to preserve interpretability without assuming specialist knowledge. Table 3 lists the eight dimensions and the group-specific wording used for CRPs and NCRPs.

## 3.12. Automated proxy evaluation

We assess explanation quality using LLM-as-a-judge scores as proxies for faithfulness to model-derived evidence. Faithfulness is evaluated using Claude Sonnet 4.6, scoring feature coverage and directional consistency against the explicit evidence blocks provided to the explanation generator (Lipton, 2018; Jacovi and Goldberg, 2020; Zheng et al., 2023; Gu et al., 2024). Perplexity and simulated stakeholder ratings are reported as complementary proxies in Appendix D.

## 3.13. Linguistic feature analysis

To analyse which textual properties are associated with higher perceived explanation quality, we link each survey rating to the corresponding explanation text and extract a set of rule-based and library-derived linguistic features. These include length and readability measures (word count, sentence count, Flesch–Kincaid grade level, and Gunning fog index), sentiment measures (TextBlob polarity/subjectivity and VADER sentiment scores), lexical diversity, counts of numeric references, and counts of domain-relevant lexical indicators related to tabular evidence and network context. We also extract counts of hedging expressions, prescriptive language, and phrases associated with a stronger causal framing.

Table 3: Perceived explanation-quality criteria across CRPs and NCRPs.
<table><tr><td>Metric</td><td>Acronym</td><td>CRP Wording</td><td>NCRP Wording</td></tr><tr><td>Understandability UND</td><td></td><td>The explanation is easy to un- derstand.</td><td>The explanation is easy to un- derstand.</td></tr><tr><td>Trustworthiness</td><td>TRU</td><td>The explanation can be trusted because it presents sufficient and reliable evi- dence to support the loan</td><td>I trust this explanation given by the system.</td></tr><tr><td>Insightfulness</td><td>INS</td><td>decision. This explanation reveals in- This explanation provides sightful risk factors that influ- useful insight into why the</td><td></td></tr><tr><td>Satisfaction</td><td>SAT</td><td>ence loan decisions. The level of detail satisfies my expectations for a risk assess- the explanation addressed my</td><td>decision was made. I am satisfied with how</td></tr><tr><td>Confidence</td><td>CON</td><td>ment. my confidence for loan ap- proval/denial.</td><td>concerns about the decision. This explanation increases I am confident in the explana- tion provided for the system&#x27;s decision.</td></tr><tr><td>Convincingness</td><td>CVN</td><td>The justification is convinc- ing and argues for the deci- sion using well-weighted risk</td><td>The explanation is convincing and makes the decision seem reasonable.</td></tr><tr><td>Communicability COM</td><td></td><td>evidence. I could use this explanation to communicate with customers enough for me to communi- and other teams in the com-</td><td>The explanation is clear cate with others.</td></tr><tr><td>Usability</td><td>USB</td><td>pany. This explanation could be I am likely to use this expla- directly used in loan ap- nation in the future because it proval/denial.</td><td>supports my decision-making.</td></tr></table>

For each explanation-quality dimension, ratings are converted into percentile ranks and grouped into high-rated subsets (top 25%) and low-rated subsets (bottom 25%). We then compare the linguistic feature distributions of these groups using Cohen’s d, treating the analysis as exploratory and using efect sizes as the primary basis for interpretation.

## 3.14. Reliability and statistical analysis

Statistical methods are detailed in the Appendix D. Briefly, human ratings use mixed-efects models with random participant intercepts; automated metrics use two-way ANOVAs with Pipeline and LLM as factors. Multiplicity is controlled via Holm correction for targeted contrasts and Benjamini–Hochberg FDR for broader pairwise comparisons.

## 4. Results

This section reports results aligned with the research questions. We first provide representative examples, then report automated proxy evaluations across pipelines and LLMs, and finally present survey and linguistic analyses.

## 4.1. Descriptive overview of representative explanations

The three pipelines difer in attribution logic and narrative framing. Tabular-based explanations emphasise loan-level attributes with deterministic framing; network-based explanations reference cluster-level patterns but risk drifting toward unverifiable contextual claims. The bimodal pipeline integrates both evidence types into a single narrative. When tabular and relational signals align, this yields more holistic rationales; when they conflict, careful structuring is required to avoid contradictions. Representative examples of all three pipelines, including a bimodal example from Gemini 2.5, are provided in Appendix E.

## 4.2. Automated evaluation of explanation quality

Addressing RQ1 and partially RQ3, this subsection reports evidence-alignment results from the LLM-as-a-judge procedure across pipelines and LLM configurations. Fluency (perplexity) and simulated audience-conditioned ratings are reported in Appendix D, as complementary proxies; the fidelity results below represent the primary automated reference for the analysis.

## 4.2.1. Evidence alignment via LLM-as-a-judge

We use an LLM-as-a-judge procedure to evaluate the evidence-grounded faithfulness of generated explanations with respect to the underlying model evidence. We operationalise faithfulness using two components: (i) feature coverage, which assesses whether the explanation references the most influential tabular or relational signals identified by the explainer, and (ii) directional consistency, which assesses whether the explanation states the correct direction of influence (e.g., risk-increasing vs. risk-decreasing) for those signals across both tabular and network modalities (Zheng et al., 2023; Gu et al., 2024). We implement this using Claude Sonnet 4.6 as the judge LLM, scoring the following variables:

• Tabular Feature Coverage (TFC): measures whether the explanation mentions the important tabular features that the predictive model actually used. Higher scores indicate that the explanation references more of the relevant features.

• Tabular Directional Consistency (TDC): measures whether the explanation correctly describes the direction of the efect of each tabular feature (i.e., the increase or decrease in estimated risk), as assessed by LLM-as-a-judge prompt adapted from previous work (Zheng et al., 2023; Gu et al., 2024).

• Network Feature Coverage (NFC): measures whether the explanation includes the important network-based features that influenced the model’s prediction. Higher scores indicate that the explanation adheres to the correct relational evidence.

• Network Directional Consistency (NDC): measures whether the explanation correctly describes the direction of efect for the network features (i.e., whether a relational signal is described as positively or negatively associated with default risk), as indicated by the model’s network-based attribution signals. Higher scores therefore reflect closer alignment with the efects captured by the model.

We report all four metrics across all pipelines to verify modality specificity: pipelines that do not include a given evidence type are expected to score substantially lower on that modality’s fidelity dimensions. Table 4 summarises the results across pipeline–LLM configurations. The reported values are means $\pm ~ 9 5 \%$ confidence intervals.

Table 4: LLM-as-a-judge scores across pipeline–LLM configurations. Boldface indicates the highest mean per column within each pipeline.
<table><tr><td>Pipeline</td><td>LLM</td><td>TFC</td><td>TDC</td><td>NFC</td><td>NDC</td></tr><tr><td rowspan="3">Tabular- based</td><td>Gemma 3</td><td> $3 . 7 7 \pm 0 . 1 2$ </td><td> $3 . 6 8 \pm 0 . 1 5$ </td><td> $1 . 7 3 \pm 0 . 0 9$ </td><td> $2 . 1 4 \pm 0 . 1 7$ </td></tr><tr><td>DeepSeek R1</td><td> ${ \bf 4 . 3 5 \pm 0 . 1 0 }$ </td><td> $3 . 4 8 \pm 0 . 1 3$ </td><td> $1 . 5 8 \pm 0 . 1 0$ </td><td> $1 . 9 1 \pm 0 . 1 7$ </td></tr><tr><td>Gemini 2.5</td><td> $3 . 9 1 \pm 0 . 1 1$ </td><td> ${ \bf 3 . 9 4 \pm 0 . 1 5 }$ </td><td> ${ \bf 1 . 9 0 \pm 0 . 0 9 }$ </td><td> ${ \bf 2 . 4 2 \pm 0 . 1 5 }$ </td></tr><tr><td rowspan="3">Network- based</td><td>Gemma 3</td><td> ${ \bf 1 . 0 2 \pm 0 . 0 3 }$ </td><td> $1 . 1 0 \pm 0 . 0 7$ </td><td> $4 . 8 3 \pm 0 . 0 7$ </td><td> $3 . 2 0 \pm 0 . 0 9$ </td></tr><tr><td>DeepSeek R1</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> ${ \bf 1 . 1 5 \pm 0 . 1 0 }$ </td><td> $4 . 4 6 \pm 0 . 1 1$ </td><td> ${ \bf 3 . 2 2 \pm 0 . 0 9 }$ </td></tr><tr><td>Gemini 2.5</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 1 \pm 0 . 0 2$ </td><td> ${ \bf 4 . 8 8 \pm 0 . 0 7 }$ </td><td> $2 . 8 9 \pm 0 . 1 0$ </td></tr><tr><td rowspan="3">Bimodal</td><td>Gemma 3</td><td> $4 . 0 8 \pm 0 . 1 6$ </td><td> $3 . 4 4 \pm 0 . 1 6$ </td><td> $3 . 8 3 \pm 0 . 1 4$ </td><td> $3 . 1 3 \pm 0 . 2 2$ </td></tr><tr><td>DeepSeek R1</td><td> $4 . 4 7 \pm 0 . 1 4$ </td><td> ${ \bf 3 . 6 7 \pm 0 . 1 8 }$ </td><td> $3 . 3 9 \pm 0 . 1 2$ </td><td> $3 . 3 6 \pm 0 . 1 1$ </td></tr><tr><td>Gemini 2.5</td><td> ${ \bf 4 . 7 3 \pm 0 . 1 3 }$ </td><td> ${ \bf 3 . 6 7 \pm 0 . 1 4 }$ </td><td> ${ \bf 3 . 9 8 \pm 0 . 1 6 }$ </td><td> ${ \bf 3 . 6 1 \pm 0 . 1 4 }$ </td></tr></table>

Four descriptive patterns emerge. First, tabular-based pipelines score high on tabular fidelity but near the lower bound on network metrics, consistent with the absence of relational evidence in the inputs. Within this pipeline, DeepSeek R1 achieves the highest tabular feature coverage and Gemini 2.5 the strongest tabular directional consistency.

Second, the network-based pipeline exhibits the complementary pattern. All three LLMs obtain very low scores in tabular metrics (TFC/TDC ≈ 1), while the coverage of the network features and the consistency of the network directionality are highest in this setting. Gemini 2.5 achieves the strongest network feature coverage, while DeepSeek R1 achieves the highest network directional consistency. These results suggest that when relational attribution signals are provided as input, the generated explanations tend to reference and correctly characterise network-based evidence.

Third, looking within each pipeline, in ten out of twelve cases feature coverage is greater than directional consistency (excluding TFC/TDC in the network-based pipeline and NFC/NDC in the tabular-based pipeline since the pipeline can not have the respective evidence in these cases). In the network-based pipeline, NFC is in the range 4.46−4.88 whereas NDC is in the range 2.89−3.22 meaning there is a gap of 1.24 − 1.99 between the two measures. Similarly, in the bimodal pipeline, TDC stalls at 3.67, whereas TFC reaches 4.73. This indicates that the narratives reliably name the right risk factors, but are less reliable when stating whether each one pushes risk up or down.

Finally, the bimodal pipeline produces comparatively balanced scores in both modalities. In the bimodal pipeline, Gemini 2.5 achieves the highest tabular feature coverage and network feature coverage, while DeepSeek R1 and Gemini 2.5 achieve equal tabular directional consistency (3.67 $\pm \ 0 . 1 8$ and $3 . 6 7 \pm \ : 0 . 1 4$ , respectively). This pattern suggests that when both tabular and relational attribution signals are available to the explanation generator, the resulting narratives can reference both evidence types simultaneously, reducing the modality-specific trade-ofs observed in single-modality pipelines. To quantify systematic diferences, we performed two-way ANOVAs for each fidelity metric with the factors Pipeline and LLM, including their interaction (Table 5).

Table 5: Two-way ANOVA results for LLM-as-a-judge fidelity metrics. All efects are statistically significant $( p <$ $1 0 ^ { - 6 } )$ .
<table><tr><td>Metric</td><td> $F _ { \mathrm { p i p e } }$ </td><td> $\eta _ { \mathrm { p i p e } } ^ { 2 }$ </td><td> $F _ { \mathrm { l l m } }$ </td><td> $\eta _ { \mathrm { l l m } } ^ { 2 }$ </td><td> $F _ { \mathrm { i n t } }$ </td><td> $\eta _ { \mathrm { i n t } } ^ { 2 }$ </td></tr><tr><td>TFC</td><td>328.96</td><td>0.391</td><td>11.81</td><td>0.014</td><td>10.04</td><td>0.024</td></tr><tr><td>TDC</td><td>299.02</td><td>0.352</td><td>26.08</td><td>0.031</td><td>22.07</td><td>0.052</td></tr><tr><td>NFC</td><td>354.27</td><td>0.377</td><td>62.95</td><td>0.067</td><td>21.66</td><td>0.046</td></tr><tr><td>NDC</td><td>39.19</td><td>0.069</td><td>28.94</td><td>0.051</td><td>11.63</td><td>0.041</td></tr></table>

The pipeline is the dominant source of variance for tabular feature coverage, tabular directional consistency, and network coverage, accounting for approximately 35–39% of the total variance across these metrics. For network directional consistency (whether explanations correctly capture the direction of network-based attribution signals), the variance is more evenly distributed between the pipeline, LLM, and their interaction, indicating that both evidence modality and the choice of LLM contribute to the diferences in the directional alignment.

The choice of LLM explains a smaller but statistically significant share of variance across all four metrics, particularly for network coverage $( \eta ^ { 2 } = 0 . 0 6 7 )$ and tabular directional consistency $( \eta ^ { 2 } = 0 . 0 3 1 )$ . The Pipeline×LLM interaction terms are also significant for all metrics, indicating that the diferences between LLMs depend on the type of evidence available to the explanation generator.

## 4.3. Human evaluation of explanation quality

This subsection examines diferences in perceived explanation quality between CRPs $( n = 4 8 )$ and NCRPs $( n = 1 0 1 )$ , evaluating bimodal explanations across the eight dimensions in Subsection 3.11. The human study was restricted to the bimodal pipeline to reduce participant burden while focusing on the most demanding evaluation condition, since it combines both tabular and network evidence. Consequently, conclusions about LLM-level diferences should be interpreted within the bimodal pipeline and may not generalise uniformly to single-modality pipelines. The characteristics of the participants are summarised in Appendix F.

## 4.3.1. Ratings collected via survey

Tables 6 and 7 summarise item-level mean Likert ratings for bimodal explanations by cohort and LLM in the eight dimensions in Subsection 3.11. The reported values are means ± 95% confidence intervals, computed as $\bar { x } \pm t _ { 0 . 9 7 5 , n - 1 } \cdot \mathrm { S E M }$ with $\operatorname { S E M } = \operatorname { S D } / { \sqrt { n } }$

Table 6: Human (CRP) mean Likert ratings across explanation-quality metrics. Boldface indicates the highest mean per column.
<table><tr><td>Metric</td><td>Gemma 3</td><td>DeepSeek R1</td><td>Gemini 2.5</td></tr><tr><td>UND</td><td> ${ \bf 4 . 6 5 \pm 0 . 1 4 }$ </td><td> $4 . 2 4 \pm 0 . 2 3$ </td><td> $4 . 4 1 \pm 0 . 1 9$ </td></tr><tr><td>TRU</td><td> ${ \bf 4 . 2 3 \pm 0 . 2 1 }$ </td><td> $3 . 9 3 \pm 0 . 2 5$ </td><td> $3 . 9 8 \pm 0 . 2 2$ </td></tr><tr><td>INS</td><td> ${ \bf 4 . 0 9 \pm 0 . 2 5 }$ </td><td> $3 . 7 8 \pm 0 . 2 5$ </td><td> $3 . 8 8 \pm 0 . 2 3$ </td></tr><tr><td>SAT</td><td> ${ \bf 3 . 9 7 \pm 0 . 2 4 }$ </td><td> $3 . 7 5 \pm 0 . 2 9$ </td><td> $3 . 5 7 \pm 0 . 2 7$ </td></tr><tr><td>CON</td><td> ${ \bf 4 . 1 0 \pm 0 . 2 2 }$ </td><td> $3 . 7 0 \pm 0 . 2 7$ </td><td> $3 . 8 7 \pm 0 . 2 4$ </td></tr><tr><td>CVN</td><td> ${ \bf 4 . 0 8 \pm 0 . 2 2 }$ </td><td> $3 . 6 4 \pm 0 . 2 9$ </td><td> $3 . 7 4 \pm 0 . 2 5$ </td></tr><tr><td>COM</td><td> ${ \bf 4 . 4 2 \pm 0 . 1 9 }$ </td><td> $3 . 6 4 \pm 0 . 2 9$ </td><td> $3 . 9 8 \pm 0 . 2 4$ </td></tr><tr><td>USB</td><td> ${ \bf 3 . 9 9 \pm 0 . 2 0 }$ </td><td> $3 . 3 9 \pm 0 . 2 9$ </td><td> $3 . 4 0 \pm 0 . 2 9$ </td></tr></table>

To address RQ2, we compare mean ratings and confidence intervals between CRPs and NCRPs. Descriptively, Gemma 3 received the highest mean ratings across all eight metrics for CRPs and seven of eight for NCRPs, with the exception of satisfaction (SAT), where DeepSeek R1 scored marginally higher.

Table 7: Human (NCRP) mean Likert ratings across explanation-quality metrics. Boldface indicates the highest mean per column.
<table><tr><td>Metric</td><td>Gemma 3</td><td>DeepSeek R1</td><td>Gemini 2.5</td></tr><tr><td>UND</td><td> ${ \bf 4 . 3 6 \pm 0 . 1 1 }$ </td><td> $4 . 2 2 \pm 0 . 1 5$ </td><td> $4 . 1 8 \pm 0 . 1 7$ </td></tr><tr><td>TRU</td><td> ${ \bf 4 . 2 4 \pm 0 . 1 1 }$ </td><td> $4 . 1 9 \pm 0 . 1 4$ </td><td> $4 . 0 1 \pm 0 . 1 6$ </td></tr><tr><td>INS</td><td> ${ \bf 4 . 1 1 \pm 0 . 1 3 }$ </td><td> $4 . 0 3 \pm 0 . 1 5$ </td><td> $4 . 0 6 \pm 0 . 1 6$ </td></tr><tr><td>SAT</td><td> $4 . 0 8 \pm 0 . 1 4$ </td><td> ${ \bf 4 . 1 2 \pm 0 . 1 4 }$ </td><td> $4 . 0 1 \pm 0 . 1 7$ </td></tr><tr><td>CON</td><td> ${ \bf 4 . 2 4 \pm 0 . 1 2 }$ </td><td> $4 . 1 7 \pm 0 . 1 5$ </td><td> $4 . 1 1 \pm 0 . 1 6$ </td></tr><tr><td>CVN</td><td> ${ \bf 4 . 1 4 \pm 0 . 1 3 }$ </td><td> $4 . 0 1 \pm 0 . 1 5$ </td><td> $4 . 0 4 \pm 0 . 1 6$ </td></tr><tr><td>COM</td><td> ${ \bf 4 . 2 0 \pm 0 . 1 1 }$ </td><td> $3 . 9 7 \pm 0 . 1 6$ </td><td> $3 . 9 2 \pm 0 . 1 5$ </td></tr><tr><td>USB</td><td> ${ \bf 4 . 2 0 \pm 0 . 1 4 }$ </td><td> $3 . 8 2 \pm 0 . 1 9$ </td><td> $4 . 1 4 \pm 0 . 1 6$ </td></tr></table>

With respect to RQ3, these patterns suggest that the smaller fine-tuned Gemma 3 was often perceived more favourably than the larger zero-shot Gemini 2.5 in dimensions such as understandability, trustworthiness, and usability. However, because confidence intervals overlap between LLMs, we interpret these diferences as descriptive tendencies rather than statistically reliable gaps.

The cohort patterns also difer. CRPs show clearer separation across LLMs, whereas NCRPs rate explanations more uniformly, suggesting that domain expertise shapes how users distinguish between explanation styles. In particular, CRPs appear to apply stricter evidentiary standards, which leads to greater diferentiation between LLMs. Figures 3a–3c illustrate these between-cohort patterns across the eight evaluation metrics.

Across LLMs and metrics, cohort-level diferences are modest and not consistently directional. NCRPs sometimes report slightly higher mean ratings, but the pattern is not uniform across LLMs or dimensions, and median ratings often coincide at 4 or 5. We therefore rely primarily on mean ratings with confidence intervals for descriptive comparison. In general, both CRPs and NCRPs positively rate the bimodal explanations, suggesting acceptable interpretability for both cohorts in this setting.

Several LLM-level patterns also emerge. Gemma 3 receives the highest CRP rating in UND and COM, suggesting that CRPs found its explanations easier to follow and more suitable for communication within credit workflows. USB shows the largest CRP–NCRP gap, with CRPs rating all LLMs lower; this may reflect greater caution about operational use of narrative explanations. NCRPs also report slightly higher SAT scores in LLMs, including DeepSeek R1. These descriptive patterns provide context for the mixed-efects analyses reported in Subsection 4.3.2.

![](images/8b1c9fef67898039744e4b1dc9227b004f591bc9d3a94cee3a752813b5c5fc72.jpg)  
(a) Gemma 3

![](images/6824612a47442b4fd7d60f39fe1de4d43dd0a51c7fa544431aff62a54b2649a1.jpg)  
(b) DeepSeek R1

![](images/7b137e35e5d2f7ccdce8ccae47df10eed8061d6cf60c1abbf077929fdfa27594.jpg)  
(c) Gemini 2.5  
Figure 3: Between-cohort comparisons by LLM.

## 4.3.2. Mixed-efects analysis of survey ratings

This subsection reports the results for the human survey, comparing perceived explanation quality across the three LLMs, two cohorts (CRP vs. NCRP), and eight decision-relevant metrics. Because participants could opt in to additional rounds, the number of ratings per participant and per LLM is not constant. We therefore summarise participant-level mean Likert ratings (1–5; higher is better) and standard deviations, and use mixed-efects modelling to account for repeated measures and rater-specific leniency or strictness. Table 8 reports participant-level mean ratings by cohort, LLM, and metric.

Two descriptive patterns emerge. First, Gemma 3 has the highest mean ratings across all eight metrics in both cohorts, although the overlapping uncertainty suggests that these should be interpreted as tendencies rather than reliable performance diferences. This pattern is consistent with, but does not establish, the possibility that domain-aligned fine-tuning improves perceived explanation quality relative to a larger zero-shot commercial LLM. The overlapping uncertainty intervals reflect the modest sample sizes and considerable within-group variability in Likert ratings, which together limit statistical power. Second, the diferences between DeepSeek R1 and Gemini 2.5 are generally small, whereas the diferences between the cohorts are more apparent in operationally grounded dimensions such as trustworthiness and usability, where the CRPs tend to rate both LLMs lower than the NCRPs. This is consistent with prior work showing that the quality of the explanation depends on the context of the stakeholders and that NCRPs may respond diferently to the diferences in presentation (Doshi-Velez and Kim, 2017; Miller, 2019; Peters, 2008; Dieckmann et al., 2009).

Table 8: Participant-level mean Likert ratings by cohort, LLM, and metric, with standard deviations in parentheses
<table><tr><td></td><td colspan="3">CRPs</td><td colspan="3">NCRPs</td></tr><tr><td>Metric</td><td>Gemma 3</td><td>DeepSeek R1</td><td>Gemini 2.5</td><td>Gemma 3</td><td>DeepSeek R1</td><td>Gemini 2.5</td></tr><tr><td>UND</td><td>4.05 (0.87)</td><td>3.90 (0.88)</td><td>3.94 (0.88)</td><td>4.10 (0.76)</td><td>3.78 (1.04)</td><td>4.07 (0.85)</td></tr><tr><td>TRU</td><td>4.17 (0.82)</td><td>3.92 (1.00)</td><td>3.68 (1.05)</td><td>4.30 (0.60)</td><td>4.01 (0.73)</td><td>4.07 (0.78)</td></tr><tr><td>INS</td><td>4.31 (0.82)</td><td>3.77 (1.08)</td><td>3.68 (0.99)</td><td>4.39 (0.55)</td><td>4.14 (0.67)</td><td>4.07 (0.88)</td></tr><tr><td>SAT</td><td>4.40 (0.67)</td><td>3.81 (1.03)</td><td>3.65 (0.93)</td><td>4.38 (0.62)</td><td>4.16 (0.71)</td><td>4.09 (0.79)</td></tr><tr><td>CON</td><td>4.35 (0.71)</td><td>3.62 (1.16)</td><td>3.68 (0.92)</td><td>4.36 (0.58)</td><td>4.11 (0.72)</td><td>4.15 (0.83)</td></tr><tr><td>CVN</td><td>4.45 (0.71)</td><td>3.85 (1.03)</td><td>3.77 (0.87)</td><td>4.34 (0.65)</td><td>4.12 (0.75)</td><td>4.13 (0.79)</td></tr><tr><td>COM</td><td>4.19 (0.92)</td><td>3.66 (1.09)</td><td>3.80 (0.89)</td><td>4.14 (0.80)</td><td>3.94 (0.93)</td><td>3.79 (1.06)</td></tr><tr><td>USB</td><td>3.91 (0.94)</td><td>3.22 (1.22)</td><td>3.47 (1.12)</td><td>3.75 (0.97)</td><td>3.67 (0.91)</td><td>3.71 (0.95)</td></tr></table>

Mixed-efects model and cohort sensitivity. To formally test these patterns, we fit a mixed-efects model (formula: score \~ cohort \* llm \* metric + (1 | participant\_id)) to participant-level mean ratings, treating Cohort as a between-subject factor, LLM and Metric as within-subject factors, and including a random intercept for participant to capture stable rater leniency/strictness. Three nested likelihood-ratio tests (fitted by ML) assess the contribution of cohort-related terms. Adding all cohort terms significantly improves fit relative to a model with only LLM and Metric efects $( \chi ^ { 2 } = 8 8 . 2 0$ $d f = 2 4$ 2 $p < 0 . 0 0 1 )$ , indicating systematic diferences between CRPs and NCRPs. Adding cohort-by-LLM terms (and associated higher-order terms) yields a further improvement $( \chi ^ { 2 } = 3 6 . 4 0$ $d f = 1 6 .$ $p = 0 . 0 0 3 )$ , consistent with the descriptive pattern that CRPs penalise the non-Gemma 3 LLMs more strongly than NCRPs. For robustness, we tested the three-way interaction LLM × Cohort × Metric using a Type III Wald test on the full model; the result is not significant $( F = 1 . 3 4 $ $d f = 1 4$ $p = 0 . 1 7 8 )$ , suggesting that cohort diferences in LLM ratings are broadly consistent across metrics rather than driven by a single dimension.

Multiplicity control and pairwise contrasts. Given eight metrics and multiple model comparisons, we report efect sizes and uncertainty intervals as primary summaries and use p-values as complementary indicators, applying Benjamini–Hochberg false-discovery-rate control for confirmatory contrasts. Because participants evaluated up to two of the three models per round, paired contrasts between DeepSeek R1 and Gemini 2.5 are based on the subset who rated both models within each cohort (CRPs: $n = 1 0 ;$ NCRPs: $n = 1 1 )$ . To make DeepSeek R1 vs. Gemini 2.5 diferences concrete, Table 9 reports Holm-corrected and Benjamini–Hochberg FDR corrected paired contrasts within each cohort (positive values indicate higher ratings for DeepSeek R1). Under Holm correction, no contrast reaches significance (minimum $p _ { \mathrm { H o l m } } = 0 . 0 5 3$ , NCRPs COM); under Benjamini–Hochberg FDR control the result is identical (minimum p<sub>BH</sub> = 0.053). The NCRPs COM contrast represents a descriptive tendency toward higher communicability ratings for Gemini 2.5 among NCRPs, but does not meet the threshold for a reliable separation. We therefore treat all DeepSeek R1 vs. Gemini 2.5 diferences as small and context-dependent rather than reliable separations.

Table 9: Paired diferences between DeepSeek R1 and Gemini 2.5 by metric and cohort, with Holm-adjusted and Benjamini–Hochberg FDR-adjusted p-values. Positive values indicate higher ratings for DeepSeek R1.
<table><tr><td></td><td colspan="4">CRPs</td><td colspan="4">NCRPs</td></tr><tr><td>Metric</td><td>Mean diff.</td><td>SD diff.</td><td>PHolm</td><td>PBH</td><td>Mean diff.</td><td>SD diff.</td><td>PHolm</td><td>PBH</td></tr><tr><td>UND</td><td>-0.050</td><td>0.468</td><td>1.000</td><td>0.681</td><td>-0.121</td><td>0.583</td><td>1.000</td><td>0.733</td></tr><tr><td>TRU</td><td>+0.350</td><td>0.489</td><td>0.250</td><td>0.250</td><td>+0.075</td><td>0.657</td><td>1.000</td><td>0.765</td></tr><tr><td>INS</td><td>+0.150</td><td>0.648</td><td>1.000</td><td>0.732</td><td>-0.045</td><td>0.510</td><td>1.000</td><td>0.765</td></tr><tr><td>SAT</td><td>+0.250</td><td>0.635</td><td>1.000</td><td>0.517</td><td>-0.288</td><td>0.455</td><td>0.722</td><td>0.413</td></tr><tr><td>CON</td><td>+0.250</td><td>0.687</td><td>1.000</td><td>0.517</td><td>-0.150</td><td>0.555</td><td>1.000</td><td>0.733</td></tr><tr><td>CVN</td><td>+0.300</td><td>0.665</td><td>1.000</td><td>0.517</td><td>-0.068</td><td>0.560</td><td>1.000</td><td>0.765</td></tr><tr><td>COM</td><td>+0.200</td><td>0.806</td><td>1.000</td><td>0.517</td><td>-0.500</td><td>0.512</td><td>0.053</td><td>0.053</td></tr><tr><td>USB</td><td>+0.275</td><td>0.845</td><td>1.000</td><td>0.532</td><td>-0.152</td><td>0.351</td><td>1.000</td><td>0.725</td></tr></table>

Variance decomposition. Finally, Table 10 decomposes the variance in overall explanation ratings into participant- and explanation-case components. Participant-level strictness or leniency accounts for 46.3% of the variance, supporting the inclusion of random participant intercepts. Explanation cases account for 10.0%, indicating a meaningful contribution from the specific loans or scenarios being explained, even under fixed prompting rules. The remaining 43.7% is residual variance, reflecting within-participant variability not captured by stable rater tendencies or case-level efects.

Overall ratings are positive, cohort diferences are systematic but modest, and Gemma 3 shows a descriptive advantage; overlapping intervals caution against treating these as definitive separations.

Table 10: Variance decomposition of overall explanation ratings from a random-intercepts model with participant and explanation-instance efects.
<table><tr><td>Component</td><td>Variance</td><td>Share (%)</td></tr><tr><td>Participants (leniency/strictness)</td><td>0.507</td><td>46.3</td></tr><tr><td>Explanation cases (LLM×instance)</td><td>0.110</td><td>10.0</td></tr><tr><td>Residual</td><td>0.478</td><td>43.7</td></tr></table>

## 4.3.3. Linguistic and sentiment feature analysis

This subsection addresses RQ4 by linking textual properties with human-rated explanation quality in the bimodal pipeline. We compare linguistic feature profiles of higher- versus lower-rated explanations to identify which characteristics are associated with perceived quality. For each explanation-quality dimension, we define high-rated explanations as the top 25% and low-rated explanations as the bottom 25% based on human ratings, to maximise contrast between groups. We then extract 16 predefined linguistic features that span emotional tone, structural complexity, and domain-specific financial language (Table 11).

Table 11: Linguistic feature categories used in the textual analysis.
<table><tr><td>Category</td><td>Feature</td><td>Description</td></tr><tr><td rowspan="5">Sentiment</td><td>sentiment_polarity</td><td>TextBlob polarity score ∈ [−1, 1] (negative to positive)</td></tr><tr><td>sentiment_subjectivity</td><td>TextBlob subjectivity ∈ [0, 1] (objective to</td></tr><tr><td>vader_compound</td><td>subjective) VADER composite sentiment score ∈ [-1, 1]</td></tr><tr><td>vader_pos</td><td>VADER positive sentiment proportion</td></tr><tr><td>vader_neg</td><td>VADER negative sentiment proportion</td></tr><tr><td rowspan="5">Structural</td><td>vader_neu</td><td>VADER neutral sentiment proportion</td></tr><tr><td>word count</td><td>Total number of words in explanation</td></tr><tr><td>sentence count</td><td>Total number of sentences in explanation</td></tr><tr><td>flesch kincaid</td><td>Flesch-Kincaid readability grade level (higher = more complex)</td></tr><tr><td>gunning_fog</td><td>Gunning Fog readability index (higher = more complex)</td></tr><tr><td rowspan="4"></td><td>unique_tokens</td><td>Number of distinct words (lexical diversity)</td></tr><tr><td>shap_feature_count</td><td>Mentions of SHAP feature names per 100 words</td></tr><tr><td>numeric count</td><td></td></tr><tr><td>network term count</td><td>Frequency of numeric values per 100 words Credit/network terminology per 100 words</td></tr><tr><td rowspan="2">Domain- specific</td><td>hedge_count</td><td>Hedging expressions (“may”, “possibly&quot;, “sug-</td></tr><tr><td>causal_overclaiming_count</td><td>gests&quot;) per 100 words Strong causal language (“proves&quot;, “causes&quot;,</td></tr></table>

We quantify diferences in feature prevalence between high- and low-rated groups using Cohen’s d efect sizes, computed separately for each quality dimension (Figure 4). Positive values indicate features that are more prevalent in high-rated explanations; negative values indicate features associated with lower ratings.

![](images/ef41a57bbadb9de5d84c6733dc9e9576c3ada1df80e9d0f6097df07a7b5c7fa4.jpg)  
Figure 4: Cohen’s d efect sizes comparing linguistic features in high-rated (top quartile) and low-rated (bottom quartile) explanations across quality metrics. Positive values (red) indicate features more prevalent in high-rated explanations; negative values (blue) indicate the reverse.

Three patterns emerge (Figure 4). First, higher-rated explanations use a more neutral, factual tone: trustworthiness is associated with higher neutral sentiment (vader\_neu, $d = 0 . 2 4 )$ and lower positive emotional language (sentiment\_polarity, $d = - 0 . 2 6 )$ . Similar efects hold for communicability and usability. Second, numeric density (numeric\_count, $d \approx 0 . 0 5 – 0 . 1 6 )$ is positively associated with perceived quality across dimensions, suggesting evaluators prefer inspectable values over qualitative descriptions. Third, stylistic complexity (readability indices, lexical diversity, hedging) shows negligible associations $( | d | \leq 0 . 1 5 )$

Together, these results provide practical guidance: high-quality credit risk explanations should prioritise neutral tone and concrete numeric evidence while avoiding subjective or emotionally framed language.

## 5. Discussion

This section interprets the findings in relation to the RQs, focusing on their implications for credit risk communication, the underlying mechanisms, and the practical and policy limitations.

## 5.1. Impact on credit risk assessment and communication

With respect to RQ1, explanation modality is the dominant determinant of evidence-grounded fidelity in our automated evaluations. Evidence modality, operationalised through pipeline structure, explains substantially more variance in feature coverage and directional consistency than LLM choice, which nonetheless plays a secondary but statistically significant role, most pronounced for network directional consistency (Subsection 4.2.1).

In terms of RQ3, the apparent benefits of fine-tuning are pipeline-dependent. Fine-tuned LLMs show the clearest advantages in the tabular-based pipeline, whereas the zero-shot LLM remains competitive in bimodal and network-based pipelines. In the human study, domain alignment is associated with stronger perceived operational adequacy, particularly among CRP evaluators whose criteria align more closely with audit and validation needs. NCRPs also tend to rate Gemma 3 higher in communicability and usability, suggesting that domain alignment can benefit both cohorts in some settings (Subsections 4.2.1 and 4.3). At the same time, descriptive examples show that network-based narratives can drift toward plausible but unverifiable contextual statements (see Appendix E), which motivates tighter grounding constraints in regulated settings.

The results in Subsection 4.2.1 furthermore reveal a systematic asymmetry between what the explanations cover and their directional consistency, which is considerably less reliable than feature coverage. In a regulated credit context this presents a severe risk, as an adverse-action communication that names a correct variable with an inverted sign is a compliance breach, not a stylistic flaw, while being invisible to a reader who cannot access the attribution. A fluency-oriented evaluation would score such an explanation highly but it is the directional error, not omission, that is the governance-relevant failure, further highlighting the need for evaluation standards for stakeholder-appropriate explanations (De Bock et al., 2024).

The relatively strong performance of the smaller domain-adapted LLM further suggests that perceived explanation quality depends less on parameter count than on alignment between the LLM, the evidence representation, and the target use case, which is consistent with regulatory guidance emphasising that decisioning systems should be transparent, well understood, and fit for purpose (Ofice of the Superintendent of Financial Institutions, 2024). That a 4-billion-parameter model matched or exceeded a 70-billion-parameter model and a commercial API on perceived explanation quality carries an eficiency implication: for this task, adequate explanation quality does not require frontier-scale models. Smaller domain-aligned models are substantially cheaper to run per inference and correspondingly lower in energy footprint, and can be hosted on-premise rather than called through an external API. In a setting where borrower-facing explanations may be generated at high volume, this couples the governance advantages of domain alignment with a materially lower computational and environmental cost.

## 5.2. Mechanisms underlying the observed efects

Two complementary mechanisms help explain why domain-aligned LLMs performed more strongly and why NCRPs tended to rate explanations more positively. First, fine-tuned LLMs were trained in supervision aligned with the target domain and the evidence-to-text format used in our pipelines. This encourages narratives to prioritise inspectable evidentiary cues (e.g., debt-to-income ratios, credit scores, and network statistics) over generic persuasive framing. Consistent with RQ4, the linguistic analysis in Subsection 4.3.3 shows that higher-rated explanations are more neutral and evidence-forward: trustworthiness is associated with higher neutral sentiment and lower positively valenced language, while numeric specificity is positively associated with perceived quality. Together, these patterns suggest that domain alignment promotes factual, evidence-grounded statements rather than persuasive narratives.

Second, and central to RQ2, CRP and NCRP evaluators appear to apply diferent heuristics when judging explanation quality (see Subsections 4.3 and 3.11). Research on the afect heuristic suggests that people often rely on emotional responses when evaluating risk, with positive afect reducing perceived risk and negative afect increasing caution (Slovic et al., 2007). Related work in behavioural decision-making shows that vivid and emotionally salient information can exert a disproportionate influence on risk judgments compared to purely statistical information (Loewenstein et al., 2001). In our study, NCRPs tended to rate explanations more highly when they included narrative context and conversational framing, even when evidentiary linkage was weaker, whereas CRPs were more likely to down-rate explanations lacking precise, inspectable numerical justification. This divergence is consistent with a governance-oriented view of explainability: afective framing can increase perceived communicability while weakening trustworthiness and operational adequacy, a pattern that general-purpose LLMs favouring fluent evaluative language may amplify.

## 5.3. Implications and limitations for practice and policy

These mechanisms have practical implications for governance. For example, the Canadian Banking Regulator’s model risk guidance emphasises transparency, documentation, independent challenge, and ongoing monitoring within a risk-based framework (Ofice of the Superintendent of

Financial Institutions, 2024). Consistent with these priorities, on-premise fine-tuned LLMs may offer greater auditability and change control than commercial API-based systems, although we do not test governance properties directly. In contrast, zero-shot commercial LLMs often provide limited visibility into training data, update schedules, and drift, increasing compliance exposure.

Second, organisations should exercise caution when using LLMs for adverse-action letters: creditors remain responsible for specific evidence-based reasons and cannot rely on generative systems to obscure or generalise them (Consumer Financial Protection Bureau, 2022), and our findings suggest generic LLMs may produce fluent but weakly grounded narratives that omit key quantitative evidence. Borrower-facing explanations should therefore rely on auditable, domain-aligned systems with logs linking each narrative to its supporting SHAP/GNN artefacts and explainer/LLM versions; LLMs with opaque training and update processes should not be deployed without extensive internal validation and documentation. Fairness is an important direction for future work: standard generative LLMs may behave diferently across demographic groups (Barocas et al., 2023; Kozodoi et al., 2022), so future studies should evaluate explanation quality across protected borrower characteristics.

These implications should be interpreted in light of several limitations. First, our tasks focus on consumer credit in a single jurisdiction, which may limit generalisability, although this is partially mitigated by a survey that included participants from multiple jurisdictions. Second, outputs are sensitive to prompt phrasing and stochastic decoding, and provider updates can induce behavioural drift, underscoring the need for continuous governance oversight. Third, residual dependence from shared prompts and training distributions cannot be fully excluded and may afect variance estimates. Finally, evaluator heterogeneity introduces noise; the CRP vs. NCRP diferences are informative but may not represent the full stakeholder set.

## 6. Conclusions

This study examined how LLMs can function as explanation layers in credit risk modelling, particularly whether they can translate technical explanation artefacts into narratives that are faithful, decision-relevant and usable in practice. We observe systematic diferences between CRP and NCRP evaluators: CRPs apply stricter evidentiary and operational criteria, whereas NCRPs place greater weight on narrative clarity and communicability. The LLM configuration also matters. Across the evaluated configurations, domain-adapted LLMs often align strongly with structured tabular evidence, especially in tabular-based pipelines, although the zero-shot LLM remains competitive in several bimodal metrics. Across configurations, the explanation modality is the dominant driver of automated fidelity, indicating that the evidence provided to the explainer fundamentally shapes what can be communicated and audited. We also identify a systematic asymmetry between what generated explanations cover and what they get directionally right, which can have great consequences for regulated credit communications.

Our linguistic analysis suggests that higher-rated explanations tend to use a more neutral tone and include more concrete numeric detail, although these associations are modest. Taken together, the findings indicate that both the LLM choice and the evidence modality shape the quality of the explanation and therefore warrant explicit governance attention. In regulated credit settings, institutions should prioritise domain-aligned explanation systems, maintain robust evaluation and monitoring, and retain evidence logs linking narratives to the underlying explanation artefacts. More broadly, combining regulatory guidance, human judgement, and tailored machine learning ofers a practical path toward risk-aware AI in financial decision-making.

Future research could extend this framework across jurisdictions, lending products, and languages to assess external validity, and expand evaluation cohorts to include regulators, underwriters, and borrowers to clarify stakeholder-specific requirements. Developing benchmark datasets emphasising faithfulness, evidentiary grounding, and fairness remains an important priority for the field.

## Acknowledgments

The first author acknowledges the support of the Natural Sciences and Engineering Research Council (NSERC) of Canada through the Canada Graduate Scholarships – Doctoral (CGS D) program. The third author acknowledges the support of the Economic and Social Research Council (ESRC) [grant number ES/P000673/1]. The fourth author acknowledges the support of the Icelandic Research Fund (IRF) [grant number 228511-051]. The last author acknowledges the support of the NSERC [discovery grant RGPIN-2020-07114]. This research was undertaken, in part, thanks to funding from the Canada Research Chairs program [CRC-2024-00192]. This work was enabled in part by support provided by Compute Ontario (https://www.computeontario.ca), Calcul Québec (https://www.calculquebec.ca), and the Digital Research Alliance of Canada (https: //www.alliancecan.ca).

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.

During the preparation of this work the authors used Claude in order to review text. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Code Availability Statement

The code and the survey data for this work is available at https://github.com/Banking-Ana lytics-Lab/LLMExplainer.

## References

Addo, P.M., Guegan, D., Hassani, B., 2018. Credit risk analysis using machine and deep learning models. Risks 6, 38.

Anthropic, 2024. Claude Sonnet 4.6. URL: https://www.anthropic.com/claude. accessed: June 2025.

Ballegeer, M., Bogaert, M., Benoit, D.F., 2025. Evaluating the stability of model explanations in instance-dependent cost-sensitive credit scoring. European Journal of Operational Research 326, 630–640.

Barocas, S., Hardt, M., Narayanan, A., 2023. Fairness and machine learning: Limitations and opportunities. MIT Press.

Basel Committee on Banking Supervision, 2011. Basel III: A global regulatory framework for more resilient banks and banking systems. https://www.bis.org/publ/bcbs189.htm.

Benjamini, Y., Hochberg, Y., 1995. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological) 57, 289–300.

Bilal, A., Ebert, D., Lin, B., 2025. LLMs for explainable AI: A comprehensive survey.

Black, S., Gao, L., Wang, P., Leahy, C., Biderman, S., 2021. GPT-Neo: Large scale autoregressive language modeling with mesh-tensorflow. arXiv:2104.10622.

Bracke, P., Datta, A., Jung, C., Sen, S., 2019. Machine learning explainability in finance: An application to default risk analysis. Technical Report 816. Bank of England.

Breeden, J.L., 2020. Survey of machine learning in credit risk.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al., 2020. Language models are few-shot learners. Advances in Neural Information Processing Systems 33, 1877–1901.

Bücker, M., Szepannek, G., Gosiewska, A., Biecek, P., 2022. Transparency, auditability, and explainability of machine learning models in credit scoring. Journal of the Operational Research Society 73, 70–90.

Bussmann, N., Giudici, P., Marinelli, D., Papenbrock, J., 2021. Explainable machine learning in credit risk management. Computational Economics 57, 203–216.

Chen, Y., Calabrese, R., Martin-Barragan, B., 2024. Interpretable machine learning for imbalanced credit scoring datasets. European Journal of Operational Research 312, 357–372.

Consumer Financial Protection Bureau, 2022. CFPB Circular 2022-03: Adverse action notification requirements under the Equal Credit Opportunity Act and Regulation B in light of the CFPB’s interpretive rule on artificial intelligence. https://www.consumerfinance.gov/compliance/cir culars/circular-2022-03/.

Das, S., Huang, X., Adeshina, S., Yang, P., Bachega, L., 2023. Credit risk modeling with graph machine learning. INFORMS Journal on Data Science 2, 197–217.

De Bock, K.W., Coussement, K., Caigny, A.D., Słowiński, R., Baesens, B., Boute, R.N., Choi, T.M., Delen, D., Kraus, M., Lessmann, S., et al., 2024. Explainable ai for operational research: A defining framework, methods, applications, and a research agenda. European Journal of Operational Research 317, 249–272.

De Jongh, P.J., Larney, J., Mare, E., Van Vuuren, G.W., Verster, T., 2017. A proposed best practice model validation framework for banks. South African Journal of Economic and Management Sciences 20, 1–15.

DeepSeek AI, 2024. DeepSeek-VL and DeepSeek-Coder: Open source large language models. https: //github.com/deepseek-ai.

Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L., 2023. QLoRA: Eficient finetuning of quantized LLMs, in: Advances in Neural Information Processing Systems, pp. 10088–10115.

Dieckmann, N.F., Slovic, P., Peters, E.M., 2009. The use of narrative evidence and explicit likelihood by decisionmakers varying in numeracy. Risk Analysis 29, 1473–1488.

Doshi-Velez, F., Kim, B., 2017. Towards a rigorous science of interpretable machine learning.

Doumpos, M., Zopounidis, C., Gounopoulos, D., Platanakis, E., Zhang, W., 2023. Operational research and artificial intelligence methods in banking. European journal of operational research 306, 1–16.

European Union, 2016. Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 (General Data Protection Regulation). https://eur-lex.europa.eu/eli/reg /2016/679/oj.

Federal Home Loan Mortgage Corporation, 2022. Single-Family Loan-Level Dataset. URL: http: //www.freddiemac.com/research/datasets/sf\_loanlevel\_dataset.html.

Fischhof, B., 1995. Risk perception and communication unplugged: Twenty years of process. Risk Analysis 15, 137–145.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., et al., 2020. The Pile: An 800GB dataset of diverse text for language modeling. arXiv:2101.00027.

Gu, J., Jiang, X., Shi, Z., Tan, H., Zhai, X., Xu, C., Li, W., Shen, Y., Ma, S., Liu, H., et al., 2024. A survey on LLM-as-a-judge.

Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., Pedreschi, D., 2018. A survey of methods for explaining black box models. ACM computing surveys 51, 1–42.

Gupta, S., Shrivastava, V., Deshpande, A., Kalyan, A., Clark, P., Sabharwal, A., Khot, T., 2023. Bias runs deep: Implicit reasoning biases in persona-assigned LLMs.

Hayashi, Y., 2022. Emerging trends in deep learning for credit scoring: A review. Electronics 11, 3181.

Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y., 2020. The curious case of neural text degeneration, in: International Conference on Learning Representations (ICLR).

Huang, Q., Yamada, M., Tian, Y., Singh, D., Chang, Y., 2023. GraphLIME: Local interpretable model explanations for graph neural networks. IEEE Transactions on Knowledge and Data Engineering 35, 6968–6972.

Jacovi, A., Goldberg, Y., 2020. Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?, in: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 4198–4205.

Kakkad, J., Jannu, J., Sharma, K., Aggarwal, C.C., Medya, S., 2023. A survey on explainability of graph neural networks.

Keselman, H.J., Cribbie, R., Holland, B., 1999. A comparison of recent approaches to the problem of multiple comparisons. Journal of Educational and Behavioral Statistics 24, 42–69.

Kozodoi, N., Jacob, J., Lessmann, S., 2022. Fairness in credit scoring: Assessment, implementation and profit implications. European Journal of Operational Research 297, 1083–1094.

Lessmann, S., Baesens, B., Seow, H.V., Thomas, L.C., 2015. Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. European Journal of Operational Research 247, 124–136.

Li, Y., Xu, M., Miao, X., Zhou, S., Qian, T., 2024. Prompting large language models for counterfactual generation: An empirical study, in: Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pp. 13201–13221.

Lipton, Z.C., 2018. The mythos of model interpretability. Queue 16, 31–57.

Liu, A., Diab, M., Fried, D., 2024a. Evaluating large language model biases in persona-steered generation, in: Findings of the Association for Computational Linguistics: ACL 2024, pp. 9832–9850.

Liu, B., Li, I., Yao, J., Chen, Y., Huang, G., Wang, J., 2024b. Unveiling the potential of graph neural networks in SME credit risk assessment, in: 2024 5th International Conference on Intelligent Computing and Human-Computer Interaction (ICHCI), pp. 562–566.

Loewenstein, G.F., Weber, E.U., Hsee, C.K., Welch, N., 2001. Risk as feelings. Psychological Bulletin 127, 267–286.

Lucic, A., Ter Hoeve, M.A., Tolomei, G., De Rijke, M., Silvestri, F., 2022. CF-GNNExplainer: Counterfactual explanations for graph neural networks, in: Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS), pp. 4499–4511.

Lundberg, S.M., Erion, G., Chen, H., DeGrave, A., Prutkin, J.M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., Lee, S.I., 2020. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2, 56–67.

Lundberg, S.M., Lee, S.I., 2017. A unified approach to interpreting model predictions, in: Advances in Neural Information Processing Systems.

Martens, D., Hinns, J., Dams, C., Vergouwen, M., Evgeniou, T., 2025. Tell me a story! Narrative-driven XAI with large language models. Decision Support Systems 191, 114402.

Miller, T., 2019. Explanation in artificial intelligence: Insights from the social sciences. Artificial Intelligence 267, 1–38.

Misheva, B.H., Osterrieder, J., Hirsa, A., Kulkarni, O., Lin, S.F., 2021. Explainable AI in credit risk management.

Norman, G., 2010. Likert scales, levels of measurement and the “laws” of statistics. Advances in Health Sciences Education 15, 625–632.

Ofice of the Superintendent of Financial Institutions, 2024. Model risk management guideline. Technical Report. Government of Canada. URL: https://www.osfi-bsif.gc.ca/en/guidanc e/guidance-library/model-risk-management-guideline.

Óskarsdóttir, M., Bravo, C., 2021. Multilayer network analysis for improved credit risk prediction. Omega 105, 102520.

Pearl, J., 2009. Causality: Models, Reasoning, and Inference. 2 ed., Cambridge University Press, Cambridge.

Peters, E., 2008. Numeracy and the perception and communication of risk. Annals of the New York Academy of Sciences 1128, 1–7.

Ribeiro, M.T., Singh, S., Guestrin, C., 2016. "Why should I trust you?": Explaining the predictions of any classifier, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1135–1144.

Rudin, C., 2019. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence 1, 206–215.

Rudin, C., Shaposhnik, Y., 2023. Globally-consistent rule-based summary-explanations for machine learning models: Application to credit-risk evaluation. Journal of Machine Learning Research 24, 1–44.

Shi, S., Tse, R., Luo, W., D’Addona, S., Pau, G., 2022. Machine learning-driven credit risk: A systemic review. Neural Computing and Applications 34, 14327–14339.

Siddiqi, N., 2017. Intelligent credit scoring: Building and implementing better credit risk scorecards. John Wiley & Sons.

Slovic, P., Finucane, M.L., Peters, E., MacGregor, D.G., 2007. The afect heuristic. European Journal of Operational Research 177, 1333–1352.

Souza, C., 2023. AI model risk: What the current model risk management framework can teach us about managing the risks of AI models. Journal of Financial Compliance 6, 103–112.

Thomas, L.C., 2000. A survey of credit and behavioural scoring: Forecasting financial risk of lending to consumers. International Journal of Forecasting 16, 149–172.

Tigges, M., Mestwerdt, S., Tschirner, S., Mauer, R., 2024. Who gets the money? a qualitative analysis of fintech lending and credit scoring through the adoption of ai and alternative data. Technological Forecasting and Social Change 205, 123491.

Tu, J., Wu, Z., 2025. Inherently interpretable machine learning for credit scoring: Optimal classification tree with hyperplane splits. European Journal of Operational Research 322, 647–664.

Turpin, M., Michael, J., Perez, E., Bowman, S., 2023. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting, in: Advances in Neural Information Processing Systems.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y., 2018. Graph attention networks, in: 6th International Conference on Learning Representations (ICLR).

Wang, D., Zhang, Z., Zhou, J., Cui, P., Fang, J., Jia, Q., Fang, Y., Qi, Y., 2021. Temporal-aware graph neural network for credit risk prediction, in: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM), pp. 702–710.

Wang, J., Zhang, S., Xiao, Y., Song, R., 2022. A review on graph neural network methods in financial applications. Journal of Data Science 20, 111–134.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., et al., 2022. Chain-of-thought prompting elicits reasoning in large language models.

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., Yu, P.S., 2020. A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems 32, 4–24.

Ying, Z., Bourgeois, D., You, J., Zitnik, M., Leskovec, J., 2019. GNNExplainer: Generating explanations for graph neural networks, in: Advances in Neural Information Processing Systems.

Zandi, S., Korangi, K., Óskarsdóttir, M., Mues, C., Bravo, C., 2025. Attention-based dynamic multilayer graph neural networks for loan default prediction. European Journal of Operational Research 321, 586–599.

Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al., 2023. Judging LLM-as-a-judge with MT-bench and Chatbot Arena, in: Advances in Neural Information Processing Systems, pp. 46595–46623.

## Appendix A. Features used for modelling

Table A.1: Features used for model training and analysis.
<table><tr><td>Feature</td><td>Description</td></tr><tr><td>fico</td><td>FICO credit score at origination</td></tr><tr><td>dti</td><td>Original debt-to-income ratio</td></tr><tr><td>1tv</td><td>Original loan-to-value ratio</td></tr><tr><td>mi_pct</td><td>Mortgage insurance percentage</td></tr><tr><td>orig_upb</td><td>Original unpaid principal balance</td></tr><tr><td>loan_term</td><td>Original term of the loan in months</td></tr><tr><td>if_fthb</td><td>Binary flag indicating whether the borrower is a first-time home buyer</td></tr><tr><td>if_prim_res</td><td>Binary flag indicating if the property is the borrower&#x27;s primary residence</td></tr><tr><td>if_corr</td><td>Binary indicator for whether the loan originated through a corre- spondent lender</td></tr><tr><td>if_sf</td><td>Binary indicator showing whether the property is a single-family home</td></tr><tr><td>if_purc</td><td>Binary flag indicating whether the loan was used to purchase a home as opposed to refinancing</td></tr><tr><td>cnt_borr</td><td>Number of borrowers associated with the loan</td></tr><tr><td>cnt_units</td><td>Number of housing units in the mortgaged property</td></tr><tr><td>default</td><td>Being 90 days or more in payment arrears</td></tr></table>

## Appendix B. Mathematical definitions of explainability methods

This appendix provides formal definitions of SHAP and GNNExplainer, which were summarised in the main text (Section 3.5).

## B.1. SHAP

Let $f : \mathbb { R } ^ { d }  \mathbb { R }$ be a predictive model mapping input features $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { d } )$ to an output. SHAP decomposes $f ( \mathbf { x } )$ into contributions from each feature:

$$
f ( \mathbf { x } ) = \phi _ { 0 } + \sum _ { i = 1 } ^ { d } \phi _ { i } ,\tag{B.1}
$$

where $\phi _ { 0 }$ is the expected output, and each $\phi _ { i }$ is a Shapley value:

$$
\phi _ { i } = \sum _ { S \subseteq N \setminus \{ i \} } { \frac { | S | ! ( d - | S | - 1 ) ! } { d ! } } { \Big [ } f _ { S \cup \{ i \} } ( \mathbf { x } _ { S \cup \{ i \} } ) - f _ { S } ( \mathbf { x } _ { S } ) { \Big ] } ,\tag{B.2}
$$

where $N = \{ 1 , 2 , \ldots , d \}$ is the full set of feature indices and $f _ { S } ( { \bf x } _ { S } )$ denotes the expected output when only features in subset $S$ are known and the rest are marginalised. Eficient implementations such as TreeSHAP (Lundberg et al., 2020) compute these values without exponential cost.

## B.2. GNNExplainer

For a network $G = ( V , E )$ with node features $X \in \mathbb { R } ^ { | V | \times d }$ , GNNExplainer identifies a subgraph $G _ { S }$ and feature mask M that preserve a model’s prediction $f ( G , X ) _ { v }$ for target node v. It maximises the mutual information:

$$
\operatorname* { m a x } _ { G _ { S } , M } \mathbb { I } \big ( Y _ { v } ; f ( G _ { S } , X \odot M ) \big ) ,\tag{B.3}
$$

where $\odot$ denotes element-wise multiplication, $Y _ { v }$ is the predicted class or score for node $v ,$ and I denotes mutual information. This objective is approximated using a diferentiable loss:

$$
\mathcal { L } = \mathrm { C E } ( f ( G _ { S } , X \odot M ) , Y _ { v } ) + \lambda _ { 1 } \| M \| _ { 1 } + \lambda _ { 2 } \| A _ { S } \| _ { 1 } .\tag{B.4}
$$

Here, CE is the cross-entropy loss between the model output and ground truth, $A _ { S }$ is the adjacency matrix of $G _ { S }$ , and $\lambda _ { 1 } , \lambda _ { 2 }$ are sparsity regularisers.

## Appendix C. Prompt examples

This appendix provides compact, concrete instantiations of the prompt components listed in Table 3 of the main text for each explanation pipeline (tabular-based, network-based, bimodal). Each example shows: (i) predicted outcome, (ii) evidence block(s) passed to the LLM (SHAP and/or GNNExplainer summaries), (iii) structured reasoning instructions and guided subgoals, (iv) counterfactual framing, and (v) explicit output length/format constraints. We omit the shared system prompt for brevity (it is identical across pipelines; see Section 3.8 of the main text).

Shared constraints (applies to all examples). The shared system prompt (omitted here) enforces: (a) neutral, evidence-based language; (b) no factors beyond the evidence blocks; (c) no causal guarantees; and (d) explicit length/format constraints. The examples below instantiate these rules.

## C.1 Tabular-based pipeline prompt example

Table C.1: Tabular-based prompt example instantiating Table 3 of the main text.
<table><tr><td>Component</td><td>Concrete instantiation</td></tr><tr><td>1) Predicted outcome</td><td>Prediction: Loan 31780 default probability = 0.8712. Actual outcome: default.</td></tr><tr><td>2) Top-k SHAP feature attributions with descriptions</td><td>Evidence (top-k SHAP; top-3 shown): • cnt_borr (SHAP:+0.4362): pushed toward default • fico (SHAP:-0.2609): pushed toward no default • dti (SHAP:+0.0799): pushed toward default (Full top-k list is passed to the LLM; only top-3 shown here.) Feature descriptions: cnt_borr = number of borrowers; fico = credit score; dti = debt-to-income ratio.</td></tr><tr><td>3–4) Instructional reasoning + guided subgoals</td><td>Instructions (follow in order): 1. Explain the predicted outcome in plain language. 2. Use the SHAP ordering; explain each listed factor: what it is, whether it increases/decreases risk, and why it matters to lenders. 3. Combine factors into a short overall justification. Constraints: borrower-facing language; do not mention SHAP/models/algorithms; reference only evidence-block factors; neutral tone; short sentences.</td></tr><tr><td>5) Borrower-level counterfactual prompt</td><td>Counterfactual guidance: Provide two realistic improvement steps tied to the risk-increasing factors (positive SHAP values). Use cautious phrasing (e.g., “&quot;If you were able  $\mathrm { t o . . . , }$  this would likely improve...&quot;).</td></tr><tr><td>6) Output format constraint</td><td>Format: No bullet points in the final answer. End with one sentence summarising the main drivers.</td></tr></table>

## C.2 Network-based pipeline prompt example

Table C.2: Network-based prompt example instantiating Table 3 of the main text.
<table><tr><td>Component</td><td>Concrete instantiation</td></tr><tr><td>1) Predicted outcome</td><td>Prediction: Loan 2885 no-default probability = 0.9489. Actual outcome: no default.</td></tr><tr><td colspan="2" rowspan="1">Component                   Concrete instantiation</td></tr><tr><td colspan="1" rowspan="1">2) Network connectivityanalysis</td><td colspan="1" rowspan="1">Evidence (GNNExplainer summary; top-3 shown):• Top node features (importance scores):- fico (0.6967)- if_corr (0.6736)- 1tv (0.3349)• Top edge connections:- Geographic area connection (Loan 2733 → 2885, cluster size ~75): impact 0.2380- Area-provider cluster connection (Loan 2622 → 2885, cluster size ~9): impact 0.2314- Same loan provider connection (Loan 2493 → 2885, cluster size ~145): impact 0.2266(Full lists are passed to the LLM; only top-3 shown here.)</td></tr><tr><td colspan="1" rowspan="1">3–4) Instructional reasoning+ guided subgoals</td><td colspan="1" rowspan="1">Instructions (follow in order):1. Explain the predicted outcome in plain language.2. Describe the strongest connections using the edge list and cluster sizes.3. Explain (at a high level) why geographic/provider groupings may matter, withoutadding unverifiable claims.Constraints: focus on relational/context evidence; mention at least two connection types;do not introduce attributes not shown in the evidence block; neutral tone.</td></tr><tr><td colspan="1" rowspan="1">5) Network-levelcounterfactual prompt</td><td colspan="1" rowspan="1">Counterfactual guidance: Provide one cautious “what-if" statement on how differentrelational patterns (e.g., fewer links to a given cluster) could change confidence; avoidimplying control over other borrowers.</td></tr><tr><td colspan="1" rowspan="1">6) Output format constraint</td><td colspan="1" rowspan="1">Format: Include at least one sentence referencing a connection type and a cluster size fromthe evidence list. End with a one-sentence summary.</td></tr></table>

## C.3 Bimodal pipeline prompt example

Table C.3: Bimodal prompt example instantiating Table 3 of the main text.
<table><tr><td rowspan=1 colspan=2>Component                   Concrete instantiation</td></tr><tr><td rowspan=1 colspan=1>1) Predicted outcome</td><td rowspan=1 colspan=1>Prediction: Loan 1007 default probability = 0.9132. Actual outcome: default.</td></tr><tr><td rowspan=1 colspan=1>2) Top-k SHAP featuredescriptions</td><td rowspan=1 colspan=1>Evidence (tabular SHAP; top-3 shown):• cnt_borr (SHAP=+0.26): increases risk• dti (SHAP=+0.21): increases risk• fico (SHAP=-0.18): reduces risk(Full top-k list is passed to the LLM; only top-3 shown here.)</td></tr><tr><td rowspan=1 colspan=1>3) GNNExplainer subgraphdescriptions</td><td rowspan=1 colspan=1>Evidence (network summary; compact subset shown):• Edge: Same provider connection (Loan 5021 → 1007, cluster size ~12)• Edge: Area-provider cluster (Loan 2893 → 1007, cluster size ~8)• Node: Loan 5021 (defaulted) identified as influential neighbour(Full subgraph/edge lists are passed to the LLM; only a compact subset shown here.)</td></tr><tr><td rowspan=1 colspan=1>4) Combined instructionalreasoning linking bothmodalities</td><td rowspan=1 colspan=1>Instructions (follow in order):1. Explain the predicted outcome overall.2. Explain the strongest tabular drivers first (SHAP ordering), then the strongest networkdrivers (edge types + cluster sizes).3. Include one explicit linking sentence describing how borrower-level and relational signalscombine.</td></tr><tr><td rowspan=1 colspan=1>5) Combined borrower- andnetwork-level counterfactualprompt</td><td rowspan=1 colspan=1>Counterfactual guidance: Provide two improvement suggestions: (i) one tied to apositive-SHAP tabular factor; (ii) one stated cautiously about relational context (howdifferent clustering patterns could affect confidence).</td></tr><tr><td rowspan=1 colspan=1>6) Disambiguation hint forconflicting contributions</td><td rowspan=1 colspan=1>Disambiguation rule: If tabular and network evidence point in different directions,acknowledge both and state that the assessment balances the signals, without adding newevidence.</td></tr><tr><td rowspan=1 colspan=1>7) Output format constraint</td><td rowspan=1 colspan=1>Format: Must include (i) at least one tabular feature name from the SHAP list and (ii) atleast one connection type plus a cluster size from the network list. End with a one-sentencesummary.</td></tr></table>

## Appendix D. Supplementary Methods and Automated Proxy Results

This appendix consolidates supplementary methodological detail and secondary automated proxy results moved from the main text to preserve focus on evidence-grounded fidelity and human evaluation. Appendix D.1 provides full statistical analysis specifications, Appendix D.2 reports linguistic fluency results via perplexity scoring, and Appendix D.3 reports simulated stakeholder ratings via Claude Sonnet 4.6. Results in Appendices D.2 and D.3 are treated as complementary proxies to the primary automated and human evaluations reported in Sections 4.2 and 4.3 of the main text.

## D.1. Statistical Analysis Specifications

Our statistical analysis prioritises robustness and interpretability through efect sizes, uncertainty intervals, and models that account for repeated measures and interaction efects. For human ratings, we use mixed-efects models to account for repeated measures and rater heterogeneity across LLMs, metrics, and evaluators. For automated proxy metrics, we use configuration-level summaries with 95% confidence intervals and two-way ANOVAs to assess variation between pipelines, LLMs and their interaction. Where interaction efects are significant, the main efects are interpreted cautiously as averages across factor levels, and the cell means are used as the primary basis for more local comparisons.

Automated evaluations (all pipelines). For automated evaluations, we report configuration-level means with 95% confidence intervals for all proxy metrics, including perplexity (linguistic fluency), LLM-as-a-judge fidelity scores (tabular feature coverage, tabular directional consistency, network feature coverage, and network directional consistency), and simulated stakeholder ratings generated separately under CRP- and NCRP-oriented rubrics. To test whether automated scores vary systematically across design choices, we run two-way ANOVA models for each automated metric with factors Pipeline (tabular-based, network-based, bimodal) and LLM (Gemma 3, DeepSeek R1, Gemini 2.5), including their interaction. We report F-statistics, p-values, and efect sizes as $\eta ^ { 2 }$ interpreting $\eta ^ { 2 }$ and confidence intervals as primary signals.

Human ratings (bimodal pipeline). Human evaluation is limited to bimodal explanations to reduce participant burden. We summarise the mean Likert ratings at the participant-level (1–5) for each metric and LLM (with standard deviations). Although item-level Likert responses are ordinal, previous work supports treating aggregated participant-level means as approximately continuous in mixed designs (Norman, 2010). To account for rater-specific leniency/strictness and repeated measures, we fit mixed-efects models with a random intercept for participant and fixed efects for Cohort (CRP vs. NCRP), LLM and Metric. We use likelihood-ratio tests to assess whether adding cohort-related terms (and cohort-by-LLM interactions) improves model fit.

Multiplicity control. Because the human study evaluates eight metrics, we control multiplicity in two complementary ways. For a small set of pre-specified targeted contrasts, we report Holm-corrected paired comparisons, which controls the family-wise error rate (FWER) and ensures that the probability of any single false positive across this confirmatory set remains bounded at α. For the broader confirmatory pairwise contrasts within each cohort across the eight metrics, we apply Benjamini-Hochberg False-Discovery-Rate (FDR) control (Benjamini and Hochberg, 1995): with the number of simultaneous comparisons well exceeding six, FWER methods become prohibitively conservative (Keselman et al., 1999), and FDR control ofers substantially higher power while still bounding the expected proportion of false discoveries.

Reliability and secondary analyses. To characterise the sources of variation in human ratings, we decompose variance in overall explanation scores using random-intercept models with participant and explanation-case components. Finally, we compare linguistic feature distributions between highand low-rated explanations and report Cohen’s d efect sizes.

## D.2. Fluency via Perplexity Scoring

We assess linguistic fluency using perplexity computed under a frozen GPT-Neo-125M language model trained on the Pile corpus (Black et al., 2021; Gao et al., 2020). Perplexity measures the average surprisal assigned to a sequence, with lower values indicating more predictable, syntactically regular text under the evaluator (Holtzman et al., 2020). Because our narratives contain domain-specific terminology and numeric information, absolute perplexity may be elevated relative to open-domain text; we therefore interpret perplexity as a relative proxy and compare configurations under a fixed evaluator. Table D.1 reports mean perplexity ± 95% confidence intervals.

Averaged across LLMs, tabular explanations exhibit the lowest perplexity, network-based explanations are slightly higher, and bimodal explanations are highest, though the ordering depends on the LLM. Across LLMs, DeepSeek R1 achieves the lowest perplexity within each pipeline, whereas Gemini 2.5 has substantially higher perplexity, particularly in the bimodal pipeline, suggesting that the synthesis of multiple evidence sources can increase the linguistic variability for some LLMs. To test whether perplexity varies systematically across design choices, we fit a two-way ANOVA with the factors Pipeline (tabular-based, network-based, bimodal) and LLM (Gemma 3, DeepSeek R1, Gemini 2.5), including their interaction (Table D.2).

Table D.1: Perplexity scores across pipeline–LLM configurations. Boldface indicates the lowest mean perplexity within each pipeline.
<table><tr><td>Pipeline</td><td>LLM</td><td> $\overline { { { \mathrm { P e r p l e x i t y } } } }$ </td></tr><tr><td rowspan="2">Tabular-based</td><td>Gemma 3</td><td> $\overline { { 1 9 . 3 8 \pm 0 . 6 7 } }$ </td></tr><tr><td>DeepSeek R1 Gemini 2.5</td><td> ${ \bf 1 5 . 9 3 \pm 0 . 3 6 }$ </td></tr><tr><td rowspan="2">Network-based</td><td>Gemma 3</td><td> $3 3 . 2 0 \pm 1 . 2 1$   $\overline { { 2 1 . 0 9 \pm 0 . 5 9 } }$ </td></tr><tr><td>DeepSeek R1</td><td> ${ \bf 2 0 . 0 6 \pm 0 . 5 4 }$ </td></tr><tr><td rowspan="2"></td><td>Gemini 2.5</td><td> $2 8 . 9 6 \pm 0 . 7 2$ </td></tr><tr><td>Gemma 3</td><td> $\overline { { 2 0 . 3 9 \pm 0 . 4 9 } }$ </td></tr><tr><td rowspan="2">Bimodal</td><td>DeepSeek R1</td><td> ${ \bf 1 6 . 7 7 \pm 0 . 4 3 }$ </td></tr><tr><td>Gemini 2.5</td><td> $3 7 . 5 3 \pm 5 . 9 8$ </td></tr></table>

<table><tr><td>Effect</td><td>F</td><td>p</td><td> $\overline { { \eta ^ { 2 } } }$ </td></tr><tr><td>Pipeline</td><td>18.34</td><td> $\overline { { 1 . 6 \times 1 0 ^ { - 8 } } }$ </td><td>0.010</td></tr><tr><td>LLM</td><td>1302.39</td><td> $< 1 0 ^ { - 2 6 2 }$ </td><td>0.687</td></tr><tr><td>Pipeline×LLM</td><td>68.85</td><td> $1 . 1 \times 1 0 ^ { - 5 0 }$ </td><td>0.073</td></tr></table>

Table D.2: Two-way ANOVA results for perplexity with factors Pipeline and LLM.

The LLM main efect dominates, explaining the largest share of variance in perplexity, while pipeline configuration contributes a smaller but statistically significant efect. The significant interaction coeficient indicates that the relative fluency diferences between LLMs depend on the pipeline.

These results should be interpreted with caution. Perplexity is calculated under a fixed and relatively small evaluator, and diferences between generators may partly reflect alignment with the evaluator’s training distribution rather than intrinsic writing quality. In addition, variation in explanation length and fine-tuning may contribute to observed diferences. We therefore treat perplexity as a comparative indicator of linguistic regularity rather than a definitive measure of explanation quality.

## D.3. Simulated Ratings via Claude Sonnet 4.6

To approximate audience-conditioned perceptions of interpretability and decision support, we collect role-conditioned simulated Likert ratings using Claude Sonnet 4.6 (Anthropic, 2024) to approximate CRP- and NCRP-oriented perceptions. The judge LLM was prompted to rate each explanation from two perspectives: a CRP viewpoint that emphasises evidence and operationa relevance, and an NCRP viewpoint that prioritises clarity and perceived helpfulness. Tables D.3 and D.4 report simulated CRP and NCRP ratings in all pipeline–LLM configurations. Values are reported as mean ± 95% confidence intervals.

Table D.3: Simulated CRP ratings across pipeline–LLM configurations. Boldface indicates the highest mean per column.
<table><tr><td>Pipeline</td><td>LLM</td><td>UND</td><td>TRU</td><td>INS</td><td>SAT</td><td>CON</td><td>CVN</td><td>COM</td><td>USB</td></tr><tr><td rowspan="3">Tabular-based</td><td>Gemma 3</td><td> $\overline { { 3 . 8 9 \pm 0 . 0 7 } }$ </td><td> $2 . 1 1 \pm 0 . 0 6$ </td><td> $2 . 1 5 \pm 0 . 0 7$ </td><td> $\overline { { 1 . 9 9 \pm 0 . 0 2 } }$ </td><td> $2 . 6 2 \pm 0 . 1 0$ </td><td> $2 . 1 1 \pm 0 . 0 6$ </td><td> $2 . 7 0 \pm 0 . 0 9$ </td><td> $1 . 4 2 \pm 0 . 1 0$ </td></tr><tr><td>DeepSeek R1</td><td> $3 . 7 9 \pm 0 . 0 8$ </td><td> $2 . 3 8 \pm 0 . 1 0$ </td><td> $2 . 3 4 \pm 0 . 0 9$ </td><td>2.14 ± 0.07</td><td> $2 . 7 5 \pm 0 . 0 9$ </td><td> $2 . 3 8 \pm 0 . 1 0$ </td><td> $2 . 9 5 \pm 0 . 0 7$ </td><td> $1 . 7 9 \pm 0 . 0 8$ </td></tr><tr><td>Gemini 2.5</td><td> ${ \bf 3 . 9 9 \pm 0 . 0 8 }$ </td><td> $\mathbf { 2 . 8 0 \pm 0 . 0 8 }$ </td><td> $\mathbf { 2 . 7 8 \pm 0 . 0 9 }$ </td><td> $\mathbf { 2 . 5 8 \pm 0 . 1 0 }$ </td><td> ${ \bf 2 . 9 6 \pm 0 . 0 8 }$ </td><td> ${ \bf 2 . 7 9 \pm 0 . 0 8 }$ </td><td> ${ \bf 3 . 2 9 \pm 0 . 1 2 }$ </td><td> ${ \bf 1 . 8 6 \pm 0 . 0 7 }$ </td></tr><tr><td rowspan="3">Network-based</td><td>Gemma 3</td><td> $\overline { { 3 . 6 2 \pm 0 . 1 0 } }$ </td><td> $\overline { { 1 . 9 7 \pm 0 . 0 3 } }$ </td><td> $\overline { { 1 . 9 8 \pm 0 . 0 3 } }$ </td><td> $\overline { { 1 . 9 6 \pm 0 . 0 4 } }$ </td><td> $2 . 1 5 \pm 0 . 0 8$ </td><td> $\overline { { 1 . 9 7 \pm 0 . 0 3 } }$ </td><td> $2 . 7 8 \pm 0 . 0 8$ </td><td> $\overline { { 1 . 1 2 \pm 0 . 0 6 } }$ </td></tr><tr><td>DeepSeek R1</td><td> $2 . 9 8 \pm 0 . 0 3$ </td><td> $1 . 7 7 \pm 0 . 0 8$ </td><td> $1 . 8 1 \pm 0 . 0 8$ </td><td> $1 . 5 7 \pm 0 . 1 0$ </td><td> $1 . 9 1 \pm 0 . 0 6$ </td><td> $1 . 7 7 \pm 0 . 0 8$ </td><td> $2 . 0 2 \pm 0 . 0 4$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Gemini 2.5</td><td> $2 . 8 0 \pm 0 . 0 8$ </td><td> $1 . 4 0 \pm 0 . 1 0$ </td><td> $1 . 5 5 \pm 0 . 1 0$ </td><td> $1 . 2 9 \pm 0 . 0 9$ </td><td> $1 . 7 8 \pm 0 . 0 8$ </td><td> $1 . 4 1 \pm 0 . 1 0$ </td><td> $1 . 9 0 \pm 0 . 0 7$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td rowspan="3">Bimodal</td><td>Gemma 3</td><td> $\overline { { 3 . 9 4 \pm 0 . 0 5 } }$ </td><td> $\overline { { 2 . 5 1 \pm 0 . 1 0 } }$ </td><td> $\overline { { 2 . 4 9 \pm 0 . 1 0 } }$ </td><td> $2 . 0 1 \pm 0 . 0 4$ </td><td> $2 . 8 0 \pm 0 . 0 8$ </td><td> $\overline { { 2 . 4 9 \pm 0 . 1 0 } }$ </td><td> $2 . 7 3 \pm 0 . 0 9$ </td><td> $\overline { { 1 . 6 2 \pm 0 . 1 0 } }$ </td></tr><tr><td>DeepSeek R1</td><td> $3 . 9 8 \pm 0 . 0 3$ </td><td> $2 . 5 1 \pm 0 . 1 0$ </td><td> $2 . 5 8 \pm 0 . 1 0$ </td><td> $2 . 2 4 \pm 0 . 0 9$ </td><td> $2 . 9 4 \pm 0 . 0 5$ </td><td> $2 . 5 0 \pm 0 . 1 0$ </td><td> $3 . 0 0 \pm 0 . 0 8$ </td><td> $1 . 8 3 \pm 0 . 0 7$ </td></tr><tr><td>Gemini 2.5</td><td> $3 . 8 5 \pm 0 . 0 8$ </td><td> $2 . 4 8 \pm 0 . 1 1$ </td><td> $2 . 5 9 \pm 0 . 1 0$ </td><td> $2 . 1 9 \pm 0 . 0 8$ </td><td> $2 . 7 5 \pm 0 . 0 9$ </td><td> $2 . 4 7 \pm 0 . 1 1$ </td><td> $2 . 6 2 \pm 0 . 1 1$ </td><td> $1 . 6 2 \pm 0 . 1 0$ </td></tr></table>

Table D.4: Simulated NCRP ratings across pipeline–LLM configurations. Boldface indicates the highest mean per column.
<table><tr><td>Pipeline</td><td>LLM</td><td>UND</td><td>TRU</td><td>INS</td><td>SAT</td><td>CON</td><td>CVN</td><td>COM</td><td>USB</td></tr><tr><td rowspan="3">Tabular-based</td><td>Gemma 3</td><td> $\mathbf { \overline { { 4 . 4 3 \pm 0 . 1 1 } } }$ </td><td> $\overline { { 3 . 4 4 \pm 0 . 1 0 } }$ </td><td> $\overline { { 3 . 8 9 \pm 0 . 0 6 } }$ </td><td> $\overline { { 3 . 4 5 \pm 0 . 1 0 } }$ </td><td> $\overline { { 3 . 4 3 \pm 0 . 1 0 } }$ </td><td> $\overline { { 4 . 1 3 \pm 0 . 1 0 } }$ </td><td> $\mathbf { \overline { { 4 . 4 3 \pm 0 . 1 1 } } }$ </td><td> $\overline { { 3 . 3 8 \pm 0 . 1 0 } }$ </td></tr><tr><td>DeepSeek R1</td><td> $4 . 0 0 \pm 0 . 0 8$ </td><td> $3 . 2 0 \pm 0 . 1 0$ </td><td> $3 . 8 5 \pm 0 . 0 8$ </td><td> $3 . 3 6 \pm 0 . 1 0$ </td><td> $3 . 1 9 \pm 0 . 1 0$ </td><td> $3 . 9 1 \pm 0 . 0 8$ </td><td> $4 . 0 0 \pm 0 . 0 8$ </td><td> $3 . 1 0 \pm 0 . 0 9$ </td></tr><tr><td>Gemini 2.5</td><td> $4 . 3 9 \pm 0 . 1 0$ </td><td> ${ \bf 3 . 6 2 \pm 0 . 1 0 }$ </td><td> ${ \bf 4 . 1 9 \pm 0 . 0 8 }$ </td><td> $3 . 8 1 \pm 0 . 0 8$ </td><td> ${ \bf 3 . 6 2 \pm 0 . 1 0 }$ </td><td> ${ \bf 4 . 2 9 \pm 0 . 1 1 }$ </td><td> $4 . 3 9 \pm 0 . 1 0$ </td><td> ${ \bf 3 . 5 4 \pm 0 . 1 0 }$ </td></tr><tr><td rowspan="3">Network-based</td><td>Gemma 3</td><td> $\overline { { 3 . 9 7 \pm 0 . 0 7 } }$ </td><td> $\overline { { 2 . 9 7 \pm 0 . 0 7 } }$ </td><td> $\overline { { 3 . 6 8 \pm 0 . 1 1 } }$ </td><td> $\overline { { 3 . 0 2 \pm 0 . 0 6 } }$ </td><td> $\overline { { 2 . 9 7 \pm 0 . 0 7 } }$ </td><td> $3 . 8 6 \pm 0 . 0 7$ </td><td> $\overline { { 3 . 9 7 \pm 0 . 0 7 } }$ </td><td> $\overline { { 2 . 9 7 \pm 0 . 0 7 } }$ </td></tr><tr><td>DeepSeek R1</td><td> $3 . 6 3 \pm 0 . 1 0$ </td><td> $2 . 6 4 \pm 0 . 1 0$ </td><td> $2 . 9 1 \pm 0 . 0 6$ </td><td> $2 . 6 5 \pm 0 . 1 0$ </td><td> $2 . 6 4 \pm 0 . 1 0$ </td><td> $3 . 0 2 \pm 0 . 0 6$ </td><td> $3 . 6 3 \pm 0 . 1 0$ </td><td> $2 . 4 1 \pm 0 . 1 0$ </td></tr><tr><td>Gemini 2.5</td><td> $3 . 0 9 \pm 0 . 0 8$ </td><td> $2 . 0 4 \pm 0 . 0 6$ </td><td> $2 . 4 5 \pm 0 . 1 0$ </td><td> $2 . 0 4 \pm 0 . 0 6$ </td><td> $2 . 0 4 \pm 0 . 0 6$ </td><td> $2 . 9 3 \pm 0 . 0 6$ </td><td> $3 . 0 9 \pm 0 . 0 8$ </td><td> $2 . 0 2 \pm 0 . 0 6$ </td></tr><tr><td rowspan="3">Bimodal</td><td>Gemma 3</td><td> $\overline { { 4 . 1 2 \pm 0 . 0 7 } }$ </td><td> $\overline { { 3 . 2 1 \pm 0 . 0 9 } }$ </td><td> $\overline { { 3 . 9 9 \pm 0 . 0 2 } }$ </td><td> $\overline { { 3 . 6 3 \pm 0 . 1 0 } }$ </td><td> $\overline { { 3 . 1 9 \pm 0 . 0 8 } }$ </td><td> $\overline { { 3 . 9 9 \pm 0 . 0 4 } }$ </td><td> $\overline { { 4 . 1 3 \pm 0 . 0 7 } }$ </td><td> $\overline { { 3 . 1 9 \pm 0 . 0 8 } }$ </td></tr><tr><td>DeepSeek R1</td><td> $4 . 1 5 \pm 0 . 0 7$ </td><td> $3 . 1 8 \pm 0 . 0 8$ </td><td> $4 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 4 7 \pm 0 . 1 0$ </td><td> $3 . 1 7 \pm 0 . 0 7$ </td><td> $4 . 0 0 \pm 0 . 0 0$ </td><td> $4 . 1 5 \pm 0 . 0 7$ </td><td> $3 . 1 7 \pm 0 . 0 7$ </td></tr><tr><td>Gemini 2.5</td><td> $4 . 4 0 \pm 0 . 1 2$ </td><td> $3 . 4 0 \pm 0 . 1 1$ </td><td> $4 . 0 7 \pm 0 . 0 9$ </td><td> ${ \bf 3 . 8 2 \pm 0 . 0 9 }$ </td><td> $3 . 3 9 \pm 0 . 1 1$ </td><td> $4 . 0 9 \pm 0 . 0 8$ </td><td> $4 . 4 0 \pm 0 . 1 2$ </td><td> $3 . 3 8 \pm 0 . 1 1$ </td></tr></table>

The simulated ratings exhibit three descriptive trends. First, NCRP-oriented ratings are consistently higher than CRP-oriented ratings across metrics, suggesting that explanations optimised for clarity and surface fluency are more positively received under the NCRP rubric than under the evidence- and compliance-oriented CRP rubric.

Second, across CRP-oriented ratings, tabular-based and bimodal pipelines generally outperform network-based pipelines, indicating that explanations grounded in borrower-level evidence, whether alone or combined with relational signals, are rated more favourably than network-only explanations. Within this pattern, Gemini 2.5 often achieves the highest mean CRP ratings, although the diferences between tabular-based and bimodal pipelines are modest and metric-dependent.

Third, network-based pipelines consistently receive the lowest CRP ratings across LLMs, indicating that abstract relational evidence is comparatively dificult to operationalise in ways that align with evidence-driven evaluation criteria. In contrast, NCRP ratings show stronger sensitivity to communicative quality, with large pipeline efects and additional (though smaller) LLM and interaction efects concentrated on communicative dimensions (e.g., UND, CVN, COM). To assess whether these descriptive patterns reflect systematic variation between design choices, we fit two-way

ANOVAs separately for simulated CRP and NCRP ratings for each metric, with factors Pipeline (tabular-based, network-based, bimodal) and LLM (Gemma 3, DeepSeek R1, Gemini 2.5), including their interaction. We report F-statistics, p-values, and efect sizes $( \eta ^ { 2 } )$ in Tables D.5 and D.6. The corresponding cell means in Tables D.3–D.4 provide the main basis for within-pipeline comparisons. Table D.5: Two-way ANOVA results for simulated CRP ratings.
<table><tr><td>Metric</td><td> $F _ { \mathrm { p i p e } }$ </td><td> $p _ { \mathrm { p i p e } }$ </td><td> $\eta _ { \mathrm { p i p e } } ^ { 2 }$ </td><td> $F _ { \mathrm { l l m } }$ </td><td> $p _ { \mathrm { l l m } }$ </td><td> $\eta _ { \mathrm { l l m } } ^ { 2 }$ </td><td> $F _ { \mathrm { i n t } }$ </td><td> $p _ { \mathrm { i n t } }$ </td><td> $\eta _ { \mathrm { i n t } } ^ { 2 }$ </td></tr><tr><td>UND</td><td>471.115</td><td> $8 . 9 7 \times 1 0 ^ { - 1 4 0 }$ </td><td>0.517</td><td>51.290</td><td> $8 . 5 5 \times 1 0 ^ { - 2 2 }$ </td><td>0.105</td><td>54.234</td><td> $6 . 9 5 \times 1 0 ^ { - 4 1 }$ </td><td>0.198</td></tr><tr><td>TRU</td><td>288.654</td><td> $4 . 3 0 \times 1 0 ^ { - 9 7 }$ </td><td>0.396</td><td>0.575</td><td>0.5628</td><td>0.001</td><td>52.728</td><td> $7 . 6 6 \times 1 0 ^ { - 4 0 }$ </td><td>0.194</td></tr><tr><td>INS</td><td>260.980</td><td> $1 . 0 7 \times 1 0 ^ { - 8 9 }$ </td><td>0.373</td><td>3.884</td><td>0.0209</td><td>0.009</td><td>38.091</td><td> $2 . 0 3 \times 1 0 ^ { - 2 9 }$ </td><td>0.148</td></tr><tr><td>SAT</td><td>243.487</td><td> $7 . 1 8 \times 1 0 ^ { - 8 5 }$ </td><td>0.357</td><td>0.930</td><td>0.3951</td><td>0.002</td><td>78.371</td><td> $7 . 0 6 \times 1 0 ^ { - 5 7 }$ </td><td>0.263</td></tr><tr><td>CON</td><td>445.581</td><td> $2 . 4 1 \times 1 0 ^ { - 1 3 4 }$ </td><td>0.503</td><td>0.512</td><td>0.5996</td><td>0.001</td><td>22.268</td><td> $1 . 5 6 \times 1 0 ^ { - 1 7 }$ </td><td>0.092</td></tr><tr><td>CVN</td><td>279.329</td><td> $1 . 2 4 \times 1 0 ^ { - 9 4 }$ </td><td>0.389</td><td>0.574</td><td>0.5632</td><td>0.001</td><td>50.728</td><td> $1 . 8 9 \times 1 0 ^ { - 3 8 }$ </td><td>0.188</td></tr><tr><td>COM</td><td>229.206</td><td> $7 . 7 6 \times 1 0 ^ { - 8 1 }$ </td><td>0.343</td><td>6.907</td><td>0.0011</td><td>0.015</td><td>88.401</td><td> $3 . 7 2 \times 1 0 ^ { - 6 3 }$ </td><td>0.287</td></tr><tr><td>USB</td><td>294.177</td><td> $1 . 5 5 \times 1 0 ^ { - 9 8 }$ </td><td>0.401</td><td>13.379</td><td> $1 . 8 9 \times 1 0 ^ { - 6 }$ </td><td>0.030</td><td>20.158</td><td> $6 . 7 1 \times 1 0 ^ { - 1 6 }$ </td><td>0.084</td></tr></table>

Table D.6: Two-way ANOVA results for simulated NCRP ratings.
<table><tr><td>Metric</td><td> $F _ { \mathrm { p i p e } }$ </td><td> $p _ { \mathrm { p i p e } }$ </td><td> $\eta _ { \mathrm { p i p e } } ^ { 2 }$ </td><td> $F _ { \mathrm { l l m } }$ </td><td> $p _ { \mathrm { l l m } }$ </td><td> $\eta _ { \mathrm { l l m } } ^ { 2 }$ </td><td> $F _ { \mathrm { i n t } }$ </td><td> $p _ { \mathrm { i n t } }$ </td><td> $\eta _ { \mathrm { i n t } } ^ { 2 }$ </td></tr><tr><td>UND</td><td>229.281</td><td> $7 . 3 8 \times 1 0 ^ { - 8 1 }$ </td><td>0.343</td><td>27.297</td><td> $3 . 1 5 \times 1 0 ^ { - 1 2 }$ </td><td>0.058</td><td>53.932</td><td> $1 . 1 2 \times 1 0 ^ { - 4 0 }$ </td><td>0.197</td></tr><tr><td>TRU</td><td>302.394</td><td> $1 . 1 6 \times 1 0 ^ { - 1 0 0 }$ </td><td>0.408</td><td>17.788</td><td> $2 . 6 7 \times 1 0 ^ { - 8 }$ </td><td>0.039</td><td>57.377</td><td> $4 . 8 3 \times 1 0 ^ { - 4 3 }$ </td><td>0.207</td></tr><tr><td>INS</td><td>685.194</td><td> $4 . 4 7 \times 1 0 ^ { - 1 8 0 }$ </td><td>0.609</td><td>54.104</td><td> $6 . 9 3 \times 1 0 ^ { - 2 3 }$ </td><td>0.110</td><td>123.595</td><td> $1 . 0 6 \times 1 0 ^ { - 8 3 }$ </td><td>0.260</td></tr><tr><td>SAT</td><td>503.172</td><td> $2 . 2 3 \times 1 0 ^ { - 1 4 6 }$ </td><td>0.534</td><td>16.600</td><td> $8 . 3 9 \times 1 0 ^ { - 8 }$ </td><td>0.036</td><td>72.315</td><td> $5 . 5 0 \times 1 0 ^ { - 5 3 }$ </td><td>0.248</td></tr><tr><td>CON</td><td>297.011</td><td> $2 . 8 5 \times 1 0 ^ { - 9 9 }$ </td><td>0.403</td><td>16.985</td><td> $5 . 7 9 \times 1 0 ^ { - 8 }$ </td><td>0.037</td><td>58.700</td><td> $6 . 0 6 \times 1 0 ^ { - 4 4 }$ </td><td>0.211</td></tr><tr><td>CVN</td><td>461.246</td><td> $1 . 0 8 \times 1 0 ^ { - 1 3 7 }$ </td><td>0.512</td><td>68.807</td><td> $1 . 7 3 \times 1 0 ^ { - 2 8 }$ </td><td>0.135</td><td>75.891</td><td> $2 . 7 1 \times 1 0 ^ { - 5 5 }$ </td><td>0.257</td></tr><tr><td>COM</td><td>229.347</td><td> $7 . 0 7 \times 1 0 ^ { - 8 1 }$ </td><td>0.343</td><td>27.964</td><td> $1 . 6 8 \times 1 0 ^ { - 1 2 }$ </td><td>0.060</td><td>53.075</td><td> $4 . 4 0 \times 1 0 ^ { - 4 0 }$ </td><td>0.195</td></tr><tr><td>USB</td><td>332.705</td><td> $2 . 6 4 \times 1 0 ^ { - 1 0 8 }$ </td><td>0.431</td><td>31.776</td><td> $4 . 7 4 \times 1 0 ^ { - 1 4 }$ </td><td>0.067</td><td>55.375</td><td> $1 . 1 4 \times 1 0 ^ { - 4 1 }$ </td><td>0.201</td></tr></table>

In general, the ANOVA results show substantial systematic variation in simulated ratings between design choices. For both CRP- and NCRP-oriented ratings, pipeline efects are large, indicating that evidence modality is the main driver of perceived quality, while Pipeline×LLM interactions show that LLM performance depends on the evidence provided. The LLM main efects are smaller overall: under the CRP rubric they are limited and most visible for clarity-related metrics such as UND (and, to a lesser extent, USB), whereas under the NCRP rubric they are more pronounced on communicative dimensions such as UND, COM, and USB.

Nevertheless, because Claude Sonnet 4.6 is itself an LLM with its own stylistic tendencies, independence is relative rather than absolute, and these simulated ratings may reflect judge LLM preferences in addition to the intended audience-conditioned rubric. We therefore treat simulated ratings as complementary proxies and prioritise the human survey results as the primary reference standard.

## Appendix E. Representative Explanation Examples

This appendix provides representative explanation examples from all three pipelines: tabular-based (E.1), network-based (E.2), and bimodal (E.3), the last of which was discussed in Section 4.1 of the main text.

## E.1. Tabular-Based Pipeline Example

Explanations emphasise loan-level attributes such as credit score, debt-to-income ratio, and loan-to-value. They typically adopt a deterministic framing and highlight the top-ranked features. Although clear and concise, these outputs may oversimplify trade-ofs in borderline cases.

Example from Gemma 3: Your loan application has a 97% chance of being approved, which is much better than most people. This is largely because your credit score of 744 is excellent. A high credit score shows banks that you’re reliable when it comes to paying back money, making them more confident in lending to you.

Another big plus is your monthly debts, which are lower than most borrowers. This tells banks that you have room in your budget to handle new payments, reducing the risk for them. Applying for the loan by yourself is also a positive factor because it simplifies the decision-making process for the bank. Plus, choosing a shorter loan term is smart because it means you’ll pay less interest over time, which banks like because it shows you’re focused on saving money.

All these factors together make you a strong candidate for approval. Overall, the strongest drivers in this case are the credit score and the relatively low level of monthly debt.

## E.2. Network-Based Pipeline Example

Explanations highlight how a loan’s relational structure (e.g., provider or geographic clusters) shapes predicted risk. Narratives often reference neighbouring loans or cluster-level patterns as supporting evidence, thereby adding relational context (i.e., group membership efects). However, the resulting explanations can be more abstract than feature-based ones.

Example from DeepSeek R1: Y our loan application has a 68% chance of being approved, which is a promising indicator. The main reason for this positive outlook is your network connections in your area and with your lender.

You are connected to 2 other borrowers in your geographic area, and these connections show positive patterns. This suggests that your area has a stable job market and strong housing values, which are important factors for lenders. Lenders like to see that borrowers are in areas with steady employment opportunities and consistent property values, as these factors reduce the risk of lending.

Additionally, you share the same lender type with 1 other borrower, and this also shows positive patterns. This indicates that the lender has experience lending in your area and understands the local economic conditions, which can make them more confident in approving loans here.

You are also connected to 1 other borrower who both lives in your area and uses the same lender type, further reinforcing these positive patterns. These connections to a total of 4 borrowers help your approval chances by demonstrating a pattern of successful lending in your area.

To maintain or strengthen these network advantages, consider staying informed about your local job market and housing trends. Compared to other borrowers in similar network situations, your connections are quite strong, which works in your favour.

Overall, your network connections suggest a stable and supportive local economic environment, which is beneficial for your loan application. By understanding and leveraging these connections, you can continue to strengthen your position as a borrower.

## E.3. Bimodal Pipeline Example

Bimodal explanations integrate tabular and relational evidence into a single narrative. When both signal types align, this yields more holistic rationales; when they conflict, careful structuring is required to avoid contradictions. The following example illustrates this integration.

Example from Gemini 2.5: Based on our thorough review, your loan application has a strong 75% chance of approval, which is a very positive outlook.

First, your excellent credit score of 809 truly stands out, showing a very responsible history of managing money, much better than most borrowers. While you’re applying solo, meaning we’re looking solely at your financial strength, your debt-to-income ratio is a bit higher than 40% of other applicants, indicating more of your income goes towards existing monthly payments. However, you’re also putting down a significant down payment – more than 80% of people – and requesting a much smaller loan amount than most, making it very manageable.

Beyond your personal finances, your local connections are also a positive sign; we see you’re connected to 3 other borrowers in your geographic area and 2 more who also use our lender type, totaling 5 positive connections that show stable patterns in your community. These combined personal strengths, like your strong credit and down payment, along with positive local patterns, significantly contribute to your high approval likelihood. To make your application even stronger, consider focusing on paying down credit cards or other debts over the next 6-12 months; this would significantly improve your debt-to-income ratio, making your financial picture even more appealing.

Overall, your complete picture is very strong, demonstrating a solid financial foundation and positive community ties, which gives us confidence in this assessment.

## Appendix F. Participant demographics and additional survey information

This appendix reports supplementary statistics about the survey population used in Section 4.3 of the main text. Diferences in the numbers of CRPs and NCRPs across the figures are due to the fact that the demographic questions in the survey were optional.

![](images/f9a104146d9081e1ea1808912d73fbdddb4de1a518e86b4f1f51000909767e77.jpg)  
(a) Cohort composition

![](images/0517613987e765fcd179a426ab97c2bbb7783c9b1e89bfb39840180b71708e7d.jpg)

![](images/2abdff9cde4c0d3ce4827226c3cb363f17ed00a600944df8f8556c6132c552da.jpg)  
(c) Educational background

(b) Geographic regions  
![](images/f4bcaa1e61212440a140f24cca130eef2bb0c5c6d6f8c4ca093ea23dc3be36a1.jpg)  
(d) Employment/economic sector

![](images/210772871787029c9943006d8c2ddcd6b9caee969fad405e0229a9cc5f3914e2.jpg)  
(e) Age distribution

![](images/c57f1b6a36933bda092129683e5d7a800415121f9c191556d018123cd3299039.jpg)  
(f) Attitudes toward network variables  
Figure F.1: Demographic and background characteristics of survey participants.
# CLAIM: Leading Open-domain Active Clarification of Large Language Models with Uncertainty Measurement

Kuangzhao Yang   
Gaoling School of Artificial   
Intelligence, Renmin University of   
China   
Beijing, China   
yangkuangzhao050519@ruc.edu.cn   
Ziliang Zhao   
Gaoling School of Artificial   
Intelligence, Renmin University of   
China   
Beijing, China   
zhaoziliang@ruc.edu.cn   
Zhicheng Dou<sup>∗</sup>   
Gaoling School of Artificial   
Intelligence, Renmin University of   
China   
Beijing, China   
dou@ruc.edu.cn

## Abstract

In open-domain human–computer interaction scenarios, large language models (LLMs) frequently encounter user queries that are ambiguous or incomplete. In such cases, directly producing an answer often leads to overgeneralized, erroneous, or low-information responses. In contrast, asking clarifying questions can substantially improve interaction quality. However, existing approaches still rely heavily on manually annotated data or preference alignment to address two fundamental challenges: when clarification is necessary, and which aspect of the query should be clarified. This reliance incurs high annotation costs and limits generalization. To address these challenges, we propose CLAIM, an uncertaintydriven framework for active clarification learning in open-domain settings. CLAIM eliminates the need for explicit human preference annotations by quantifying query uncertainty through the entropy induced by answer disagreements across multiple models. This uncertainty signal is then used to construct high-quality synthetic data, enabling the training of a unified clarification decision model through a combination of supervised learning and reinforcement learning. Specifically, we propose an entropy-driven synthetic data generation pipeline that integrates entropy-based uncertainty estimation with semantic clustering and reasoning-based judgments, enabling reliable automatic annotation of clarification requirements. To train CLAIM, we formulate the clarification process as a structured decision generation problem and adopt a training paradigm that combines supervised fine-tuning (SFT) with group-relative policy optimization (GRPO). Experimental results demonstrate that CLAIM can learn stable and generalizable clarification strategies without relying on manually labeled data, ofering a low-cost and robust solution for proactive understanding in real-world opendomain interactions with LLMs.

## CCS Concepts

• Information systems → Language models.

## Keywords

Open-domain Clarification; Large Language Models; Uncertainty Estimation; Active Decision Learning; Synthetic Data

## 1 Introduction

In real-world human–LLM interaction scenarios, user queries are often not formulated as fully specified and precisely defined requests [21, 36]. Instead, users tend to express their needs using short, high-level, or ambiguous natural language, such as “Give me some gift ideas,” or “Recommend some cofee shops.” Such queries frequently omit key constraints, preferences, or contextual details that are essential for producing accurate and user-aligned responses. When a large language model responds directly to these underspecified queries without access to critical missing information, the resulting answers are often generic, overly broad, or misaligned with the user’s true intent, as illustrated in Figure 1(a). In contrast, proactively asking clarifying questions and guiding users to provide additional information allows large language models to progressively narrow the intent space and resolve underlying ambiguity. Through this interactive process, models can better identify user goals, filter irrelevant interpretations, and generate responses that are both more accurate and more informative. Prior studies have shown that such active clarification substantially improves the accuracy and relevance of subsequent responses, particularly in open-domain settings where user intent is highly variable [13, 35]. As a result, active clarification has been widely recognized as an important mechanism for improving the quality of open-domain human–LLM interactions [16].

Despite the importance of active clarification, enabling reliable clarification in open-domain human–LLM interaction remains highly challenging. First, a model must determine whether a user query is suficiently clear to support a direct response. Not all queries benefit from clarification, and unnecessary follow-up questions can interrupt the interaction flow and degrade user experience [2, 3, 38]. Second, even when a query is genuinely ambiguous or underspecified, the model must choose which missing information dimension to clarify [13]. A single query may simultaneously lack information about location, user preferences, or specific constraints, yet these dimensions often difer substantially in their importance and information gain. Clarifying along a less informative dimension while overlooking more decisive factors frequently leads to unsatisfactory outcomes, as shown in Figure 1(b). Consequently, prioritizing clarification questions that target dimensions with higher information gain is widely regarded as a key factor in determining the efectiveness of clarification [16, 33, 39].

Existing approaches typically rely on manually annotated data, handcrafted rules, or preference-based supervision to train clarification decision models [2, 3, 38]. However, in open-domain settings, the diversity and long-tail nature of user queries make such supervision costly to obtain and dificult to scale [17]. Moreover, clarification decisions are often implicitly entangled with language generation, leading to blurred decision boundaries and limited controllability [13, 26]. These limitations restrict the generalization and applicability of existing clarification methods in real-world interaction scenarios.

![](images/44469579d8d0615399575000a05c77f62391c1a1997340cfd727da9ba0bd9441.jpg)  
Figure 1: Examples of clarification in human–LLM interaction. (a) Missing clarification leads to generic answers. (b) Clarification on uninformative dimensions fails. (c) CLAIM aligns clarification with user intent and improves outcomes.

We revisit open-domain clarification from a diferent perspective, motivated by a simple yet widely observed phenomenon. When a user query is well-specified, diferent large language models tend to produce semantically consistent responses; in contrast, when the query is ambiguous or underspecified, the responses generated by diferent models often diverge substantially at the semantic level. Such semantic divergence reflects uncertainty in the model’s understanding of the query and provides a strong intrinsic signal indicating whether clarification is needed. This observation is consistent with prior findings showing that semantic disagreement among multiple sampled generations correlates strongly with model uncertainty and answer unreliability [10, 30]. It is also aligned with recent policy-discriminative learning work such as POLAR, which uses trajectories from diverse policies to expose behavior-level diferences that are dificult to obtain from a single policy alone [5]. By leveraging this signal, a model can ask targeted clarification questions that efectively narrow the intent space and lead to satisfactory outcomes, as illustrated in Figure 1(c).

Building on this intuition, we propose CLAIM (Leading Opendomain Active Clarification of Large Language Models with Uncertainty Measurement), an open-domain active clarification framework driven by model-intrinsic uncertainty. CLAIM prompts multiple heterogeneous large language models to generate candidate answers for the same user query and performs semantic clustering over these responses, characterizing query uncertainty through the entropy of the resulting answer distribution [27]. Based on this uncertainty signal, CLAIM automatically determines whether a query is worth clarifying and constructs training instances that capture uncertainty changes before and after clarification. This process relies solely on model-generated outputs and requires no human annotations or external preference signals, enabling clarification supervision to be obtained in a scalable and low-cost manner.

At the training level, CLAIM formulates open-domain clarification as a conditional generation policy learning problem [20, 38]. The model is initialized via supervised fine-tuning (SFT) to acquire stable clarification decision and generation behaviors, and is further refined with group relative policy optimization (GRPO) to improve decision consistency and robustness near the clarification boundary [12, 22, 28]. When clarification is required, the model explicitly identifies the semantic dimension to be clarified and generates a targeted clarification question grounded in that dimension [17, 26].

In summary, our contributions can be summarized as follows:

(1) We propose CLAIM, an uncertainty-driven open-domain active clarification framework that enables large language models to adapt their interaction strategies based on intrinsic uncertainty in understanding user queries;

(2) We design an entropy-driven synthetic data generation pipeline that constructs high-quality training data without any human annotations, while explicitly modeling uncertainty changes induced by clarification;

(3) We formulate clarification as an explicit decision learning problem and train the model with a combination of SFT and GRPO, improving the controllability and generalization of clarification decisions while maintaining overall system simplicity.

## 2 Related Work

## 2.1 Clarification for LLMs

In open-domain human–LLM interactions, clarification mitigates brittle guessing under underspecification. ClariLM [38] synthesizes open-domain clarification data and trains models for when-to-ask and what-to-ask via supervised learning and preference optimization, while outcome-aware training optimizes for future-turn success and encourages clarification when it improves answerability under multiple interpretations [35]. In tool-augmented settings, Ask-when-Needed [29] asks for missing/unclear arguments before tool use to improve robustness to noisy instructions.

Several benchmarks have examined clarification behavior in LLMs and consistently reveal a gap between answering performance and efective information acquisition. CLAMBER [36] and ClarQ-LLM [7] evaluate models’ ability to recognize ambiguity and obtain missing information through clarification across opendomain and task-oriented settings, showing that strong response generation alone does not translate into reliable clarification. AR-Bench [40] and QuestBench [14] further demonstrate that even when models can solve fully specified tasks, they often fail to identify what information is missing or which question should be asked under incomplete inputs. Beyond ambiguity resolution, clarification has also been explored for preference and constraint elicitation, such as sequential funnel-style questioning [19] and ambiguity reduction in code generation [32]. Taken together, these studies suggest that the core challenge in clarification lies not only in generating follow-up questions, but in reliably deciding when clarification is needed and which information dimension to target—highlighting the need for principled signals, such as uncertainty, to guide clarification decisions in open-domain human–LLM interaction.

## 2.2 Clarification in Other Domains

Clarification has been studied as information acquisition across conversational systems. In conversational IR, question selection before re-ranking [3] and large-scale resources with engagement signals (MIMICS) [34] support clarification research; later work also generates questions from weak supervision such as query reformulations under supervised and reinforcement learning. In multi-turn search, Qulac [1] enables facet-specific evaluation, and utility-driven ranking connects question choice to expected information gain, formalized by neural EVPI [23]. Related formulations appear in exploratory conversational search [15] and active learning/adaptive information acquisition [24]. Beyond retrieval, clarification supports interactive recommendation and decision support via preference elicitation [37], resolves ambiguity/missing constraints in multimodal or embodied settings [31], and is central to conversational machine reading (ShARC) [25] and ambiguityaware QA (AmbigQA) [18].

## 2.3 Entropy and Uncertainty-Based Methods

Uncertainty-driven clarification requires quantifying ambiguity to decide whether and how to ask; information-theoretic utility and EVPI-based ranking are classic formulations [23]. For LLMs, uncertainty is often measured in the answer space via sampling and disagreement, where surface diversity may miss semantic con flict; semantic uncertainty addresses this by clustering generations by meaning and computing entropy over clusters (semantic entropy) [6, 12]. Beyond repeated decoding from a single model, recent policy-discriminative learning also shows the value of sampling outputs from diverse policies to capture cross-policy behavioral diferences [5]. To reduce sampling cost, SEPs approximate semantic entropy from internal representations [11], while Cleanse uses embedding-space clustering as a proxy for semantic consistency [9]; together they motivate practical, model-agnostic alternatives to probability-based confidence [10]. For clarification-specific modeling, CLARINET distills an information-gain objective conditioned on a retrieval distribution [4]. These results motivate CLAIM: leverage multi-model disagreement, aggregate via semantic clustering, and compute entropy over clusters for clarification decisions and data construction.

## 3 CLAIM

CLAIM addresses clarification decision-making in open-domain scenarios through uncertainty-driven synthetic data construction. To operationalize the proposed framework, we implement CLAIM as an agent-style ofline pipeline, referred to as CLAIM-Agent. It is important to distinguish the two: CLAIM-Agent is a systemlevel data-construction agent that executes uncertainty estimation, clarification judgement, conflict arbitration, and clarifying question selection, whereas CLAIM is the trained single-model policy used for online inference. Thus, the multi-model and multi-call cost of CLAIM-Agent is paid only during ofline synthetic data construction; after SFT/GRPO training, CLAIM performs one standard model inference per user query. This section formalizes the task and describes the synthetic data construction process and training methodology. The corresponding implementation code and prompts are released in the repository<sup>1</sup>.

## 3.1 Problem Formulation

We consider the clarification decision-making task in single-turn open-domain human–LLM interactions. Given a user query �, which may take the form of a question, instruction, or information request, the input can be ambiguous or lack critical information. The model is required to make a decision based on the current input, determining whether to provide a direct answer or to initiate a clarification.

Formally, the model output � belongs to one of two categories: a direct answer � or a clarifying question �. The clarification decision can thus be formulated as:

$$
\mathrm { C L A I M } ( q ) = r \in \{ a , c \} ,\tag{1}
$$

where choosing � indicates that the model considers the information in � suficient, while choosing � indicates that additional interaction is needed to obtain missing information.

When the model decides to ask for clarification, it must further determine the clarification dimension, that is, which type ofmissing information to query. This decision directly afects the eficiency of subsequent interactions and the quality of information acquisition. Therefore, clarification decision-making involves not only deciding whether to clarify, but also selecting the appropriate clarification dimension. This formulation aligns with recent agent-based frameworks that treat interaction as a sequence of decision-making steps combining reasoning and action.

## 3.2 Entropy-driven Uncertainty Estimation

In clarification decision-making, a core challenge is to determine whether a user query contains suficient information for a stable and consistent answer without relying on strong assumptions. To this end, CLAIM models question uncertainty from the perspective of the answer space, by analyzing the distribution of answers generated by diferent models for the same query to capture potential ambiguity.

![](images/aa66788fc110ccd5845e3f1cead8e2c9147647e6dd4fcd8c847676f90c05e408.jpg)  
Figure 2: The overall framework of CLAIM, consisting of entropy-driven uncertainty estimation, clarification judgement, clarifying question generation, information gain-based selection, and SFT/GRPO-based training.

Specifically, given a user query �, we use $k _ { 1 }$ diferent LLMs to independently generate a set of candidate direct answers:

$$
\mathcal { A } ( q ) = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { k _ { 1 } } \} .\tag{2}
$$

We then project all candidate answers into a shared semantic representation space and perform clustering based on semantic similarity, resulting in � clusters of semantically consistent answers. Let $c _ { i }$ denote the number of answers in the �-th cluster. The corresponding cluster probability is defined as:

$$
p _ { i } = { \frac { c _ { i } } { k _ { 1 } } } , \quad \sum _ { i = 1 } ^ { n } c _ { i } = k _ { 1 } .\tag{3}
$$

Based on this distribution, we quantify the uncertainty of the user query � using entropy:

$$
E _ { 1 } ( q ) = - \sum _ { i = 1 } ^ { n } p _ { i } \log p _ { i } .\tag{4}
$$

Intuitively, when diferent models generate semantically consistent answers, the clustering is concentrated and the entropy is low, indicating that the query is well-specified. In contrast, when answers are divided into multiple semantically distinct clusters and the distribution becomes more dispersed, the entropy increases, suggesting that the query may be ambiguous or lack critical information. This entropy signal serves as a continuous measure of uncertainty and provides an important basis for subsequent clarification decisions and clarification strategy selection.

## 3.3 Clarification Judgement

3.3.1 Entropy-basedJudgement. Based on the answer distribution entropy �<sub>1</sub>(�) obtained in the previous section, CLAIM first derives a threshold-based clarification judgement that converts continuous uncertainty into an initial decision. Specifically, we introduce a threshold � and define:

$$
y _ { \mathrm { e n t } } ( q ) = \mathbb { I } \bigl ( E _ { 1 } ( q ) \geq \tau \bigr ) ,\tag{5}
$$

where $y _ { \mathrm { e n t } } ( q ) = 1$ indicates that clarification is needed, and $y _ { \mathrm { e n t } } ( q ) =$ 0 otherwise. A larger $E _ { 1 } ( q )$ reflects stronger semantic divergence among model-generated answers, suggesting higher uncertainty in query interpretation, while lower entropy typically corresponds to more consistent and well-specified queries. We use a single global threshold � = 0.45 for all datasets and discuss its rationale in $\mathrm { A p \cdot }$ pendix A.

3.3.2 LLM-basedJudgement. While entropy captures answer-level disagreement, it does not directly assess whether a query is semantically complete. To complement this signal, CLAIM employs a reasoning model that evaluates the query itself and judges whether critical information required for a precise answer is missing. This judgement focuses on semantic completeness (e.g., ambiguous references or missing constraints) and produces an independent clarification signal that complements entropy-based uncertainty.

3.3.3 Conflict Resolution. In practice, entropy-based and LLMbased judgements may disagree. Some queries exhibit high answer entropy yet remain answerable under reasonable assumptions, whereas others yield consistent answers while still lacking essential information. Relying on either signal alone is therefore insuficient.

When such disagreement occurs, CLAIM invokes an additional judgement model to arbitrate the conflict. This model takes as input both the uncertainty characteristics reflected by the answer distribution and the semantic analysis underlying the LLM-based judgement, and produces the final clarification decision. By explicitly resolving judgement conflicts through a dedicated arbitration step, CLAIM mitigates failure modes near the clarification boundary and yields more robust clarification judgements, which are then used for downstream question generation and data construction. Appendix C reports the proportion of samples requiring this arbitration step.

## 3.4 Clarifying Question Generation

For queries requiring clarification, CLAIM generates clarifying questions to acquire missing information. Rather than exhaustively enumerating all possibilities, it produces a small set of diverse candidates that difer in the information they aim to elicit.

Concretely, given a user query �, CLAIM repeatedly invokes the same generation model to produce multiple clarifying questions. Each generated question is accompanied by a dimension label that characterizes the primary aspect of missing information it targets. This label is not drawn from a predefined dimension taxonomy, but serves as an auxiliary annotation produced during generation.

To encourage diversity across candidates, CLAIM applies historybased constraints in later generation steps by conditioning on previously generated questions and their dimension labels, preventing the reuse of already covered clarification dimensions. As a result, the generated clarifying questions target complementary aspects of the query and form the candidate set for subsequent selection based on uncertainty reduction.

## 3.5 Information Gain Based Clarification Selection

After generating multiple candidate clarifying questions, CLAIM evaluates the efectiveness of diferent clarification dimensions in reducing query uncertainty. To this end, we quantify the value of a clarification dimension by comparing the change in answer distribution uncertainty before and after clarification.

Concretely, given a user query � and a candidate clarifying question ��, together with a simulated user answer �, we again employ the same $k _ { 1 }$ large language models used in the initial stage to generate a set of post-clarification candidate answers. These answers are then semantically clustered, and the entropy of the post-clarification answer distribution is computed in the same manner as in the initial uncertainty modeling:

$$
E _ { 2 } ( q , c q , A ) = - \sum _ { j = 1 } ^ { m } { p } _ { j } ^ { \prime } \log { p } _ { j } ^ { \prime } ,\tag{6}
$$

where � denotes the number of clusters obtained after clarification, and $\mathbf { \nabla } _ { \boldsymbol { j } } ^ { p ^ { \prime } } { } _ { \boldsymbol { j } }$ represents the probability of the �-th cluster.

Based on this, we define the reduction in uncertainty introduced by the clarifying question �� as the information gain(��):

$$
I G ( q , c q ) = E _ { 1 } ( q ) - E _ { 2 } ( q , c q , A ) .\tag{7}
$$

The information gain captures the efectiveness of a clarifying question in resolving ambiguity or supplementing missing critical information.

For multiple candidate clarifying questions generated for the same user query, CLAIM selects the one with the highest information gain as the optimal clarifying question. Through this information gain–based selection mechanism, CLAIM prioritizes clarifying questions that most efectively reduce uncertainty, thereby constructing high-quality clarification decision data.

## 3.6 Training Method

Based on the uncertainty-driven synthetic data constructed in the previous stages, we adopt a two-stage training paradigm to learn a stable and consistent clarification decision policy. The purpose of training is to internalize clarification behaviors into a single model, so that clarification decisions can be made directly at inference time without relying on repeated multi-model interactions or agent-style pipelines. In the first stage, SFT is used to enable the model to acquire basic clarification behaviors from automatically constructed supervision signals. In the second stage, GRPO is applied to further sharpen the decision boundary for queries with high semantic uncertainty. This training design fully exploits synthetic data generated in the previous steps and enables efective modeling of clarification strategies without relying on human preference annotations.

3.6.1 SFT. In the supervised fine-tuning stage, the model is trained as a conditional generation policy given a user query. The model outputs a structured representation that specifies the clarification decision together with the associated generation content. When clarification is required, the generation proceeds by first identifying a single semantic dimension and then producing a clarification question grounded in that dimension; otherwise, the model directly produces a final answer.

The supervision signal is derived from target output sequences constructed during the synthetic data generation process. These sequences encode both the clarification decision and its associated generation content, where the selection of clarification questions is designed to enhance the discriminability of answers after clarification. By modeling the above behaviors as a unified sequence generation task, the model can learn a complete clarification strategy within a conditional generation framework.

Formally, let the training dataset be defined as

$$
\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{8}
$$

where $x _ { i }$ denotes a user query and $y _ { i }$ denotes the corresponding target output sequence. The model is parameterized as a conditional probability distribution $p _ { \theta } ( y \mid x )$ , and the objective of supervised fine-tuning is to minimize the autoregressive negative log-likelihood loss:

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \left[ \sum _ { t = 1 } ^ { | y | } \log { \phi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) } \right] .\tag{9}
$$

By optimizing this objective, the model learns the clarification triggering patterns implicit in the synthetic data as well as the corresponding language generation behaviors, providing a stable initialization for the subsequent policy optimization stage.

3.6.2 GRPO. Although SFT efectively conveys global clarification decision signals, the model may still exhibit instability or bias when query semantics lie close to the clarification boundary. To further improve decision consistency in regions of high uncertainty, we introduce GRPO in the second stage to refine the learned policy.

In this stage, the model is treated as a stochastic policy �<sub>�</sub>. For each input query �, the current policy generates a group of candi date outputs under a fixed sampling configuration:

$$
\pmb { y } ( \pmb { x } ) = \{ \pmb { y } ^ { ( 1 ) } , \pmb { y } ^ { ( 2 ) } , \dots , \pmb { y } ^ { ( K ) } \} .\tag{10}
$$

Each candidate output is evaluated by a deterministic reward function �(�, �), which measures the extent to which the generated result aligns with the target decision induced by the synthetic data process in terms of clarification decision consistency, semantic plausibility, and structural validity.

To avoid dependence on the absolute scale of rewards, the groupwise average reward is used as a baseline:

$$
\bar { r } ( x ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } r ( x , y ^ { ( k ) } ) .\tag{11}
$$

Based on this, the relative advantage of each output is defined as:

$$
A ( x , y ^ { ( k ) } ) = r ( x , y ^ { ( k ) } ) - \bar { r } ( x ) .\tag{12}
$$

During policy updates, we adopt a clipped objective based on probability ratios to ensure training stability. Let $\pi _ { \theta _ { \mathrm { o l d } } }$ denote the policy from the previous iteration, and the probability ratio is defined as:

$$
\rho _ { \theta } ( x , y ) = \frac { \pi _ { \theta } ( y \mid x ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y \mid x ) } .\tag{13}
$$

The objective of group relative policy optimization is then given as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G R P O } } ( \theta ) = - \mathbb { E } _ { x } \Bigg [ \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \operatorname* { m i n } \left( \rho _ { \theta } ( x , y ^ { ( k ) } ) A ( x , y ^ { ( k ) } ) , \right. } \\ { \left. \operatorname { c l i p } \left( \rho _ { \theta } ( x , y ^ { ( k ) } ) , 1 - \epsilon , 1 + \epsilon \right) A ( x , y ^ { ( k ) } ) \right) \Bigg ] . } \end{array}\tag{14}
$$

where � is a clipping coeficient that constrains the magnitude of policy updates. It efectively prevents excessive updates under high-variance reward signals and improves training stability.

In practice, the policy optimization stage primarily focuses on queries with high semantic uncertainty, which typically correspond to cases with highly dispersed candidate answers or conflicting decisions in the synthetic data process. By imposing group-relative constraints on these critical samples, the model gradually learns a clearer and more robust clarification decision boundary.

## 4 Experiments

Additional implementation details and training configurations are deferred to Appendix A.

## 4.1 Datasets

We evaluate the model’s open-domain clarification ability on three representative datasets, all of which are used exclusively for evaluation. First, we adopt the synthetic clarification dataset constructed in ClariLM denoted as ClariLM-test, and only use its test set as an evaluation benchmark [38]. This dataset is automatically generated and organized around latent missing information facets in user queries, with structured instances that explicitly distinguish between clarification and non-clarification cases, making it suitable for evaluating whether a model makes appropriate clarification decisions under controlled conditions. Second, we use the IN3 (Intention-in-Interaction) dataset as an evaluation benchmark for task-oriented interactive scenarios [21]. IN3 is grounded in real-world vague user instructions and provides systematic annotations on task ambiguity, missing critical details, and their relative importance, enabling the evaluation of clarification behavior in interactive task contexts. Although IN3 provides training data for model development, we do not use its training data and evaluate exclusively on its test set. Finally, we include CLAMBER as a general open-domain clarification benchmark [36]. CLAMBER covers a broad range of open-domain topics and focuses on evaluating a model’s ability to identify uncertainty in natural language queries and to generate high-quality clarification questions. Together, these three complementary datasets allow us to comprehensively evaluate clarification performance across synthetic data, task-oriented interaction settings, and general open-domain scenarios.

## 4.2 Evaluation Metrics

We evaluate the model’s clarification ability from two complementary aspects: clarification necessity and clarifying question quality. For clarification necessity, we formulate the task as a binary classification problem, where the model determines whether a given user query requires clarification. We report Accuracy (ACC) and F1- score (F1) to measure overall decision correctness and performance under class imbalance. For samples where the model decides that clarification is necessary, we further assess the quality of the generated clarifying questions. Specifically, clarifying question quality is evaluated from two complementary perspectives. First, we define Clarification Dimension Accuracy (CDA) to measure whether the generated clarifying question focuses on the correct clarification dimension. We employ an independent large language model as an evaluator to judge whether the clarification dimension of the generated question matches the ground-truth dimension. A match is assigned a score of 1, and a mismatch a score of 0. The final CDA score is computed as the average over all samples:

$$
\mathrm { C D A } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } ( d _ { i } ^ { \mathrm { p r e d } } = d _ { i } ^ { \mathrm { g t } } ) ,\tag{15}
$$

where $d _ { i } ^ { \mathrm { p r e d } }$ and $d _ { i } ^ { \mathrm { g t } }$ denote the predicted and ground-truth clarification dimensions for the �-th sample, respectively. Second, we measure Clarifying Question Semantic Similarity (CQSS) to evaluate the semantic closeness between the generated clarifying question and its ground-truth counterpart. Specifically, we compute the cosine similarity between their vector representations for each sample and report the average similarity across all samples:

$$
\mathrm { C Q S S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \cos \big ( \mathbf { e } ( q _ { i } ^ { \mathrm { p r e d } } ) , \mathbf { e } ( q _ { i } ^ { \mathrm { g t } } ) \big ) ,\tag{16}
$$

where e(·) denotes Qwen3-Embedding-8B, and cosine similarity is computed after normalizing the two question embeddings. Together, these metrics enable a systematic evaluation of the model’s clarification ability in terms of both deciding whether clarification is needed and generating appropriate clarifying questions. The evaluation prompts are released with the code repository.

Table 1: Main evaluation results of CLAIM (8B) and baseline models on three test sets. The best result for each metric is marked in bold and the second best result for each metric is underlined.
<table><tr><td rowspan="3">Group</td><td rowspan="3">Model</td><td colspan="4">ClariLM-test</td><td colspan="4">IN3</td><td colspan="4">CLAMBER</td></tr><tr><td colspan="2">Clari. Necessity</td><td colspan="2">Clari. Quality</td><td colspan="2">Clari. Necessity</td><td colspan="2">Clari. Quality</td><td colspan="2">Clari. Necessity</td><td colspan="2">Clari. Quality</td></tr><tr><td>Acc</td><td>F1</td><td>CDA</td><td>CQSS</td><td>Acc</td><td>F1</td><td>CDA</td><td>CQSS</td><td>Acc</td><td>F1</td><td>CDA</td><td>CQSS</td></tr><tr><td rowspan="5">LLM</td><td>Llama-3.1-8B</td><td>61.35</td><td>68.07</td><td>25.04</td><td>50.31</td><td>78.70</td><td>87.83</td><td>30.52</td><td>56.90</td><td>54.87</td><td>55.55</td><td>32.98</td><td>63.82</td></tr><tr><td>Qwen3-8B</td><td>55.72</td><td>66.89</td><td>21.24</td><td>49.12</td><td>82.86</td><td>89.89</td><td>31.58</td><td>57.95</td><td>53.14</td><td>58.18</td><td>28.48</td><td>58.87</td></tr><tr><td>Qwen3-14B</td><td>73.76</td><td>77.79</td><td>26.33</td><td>53.17</td><td>79.63</td><td>87.36</td><td>18.95</td><td>53.78</td><td>62.25</td><td>63.74</td><td>44.03</td><td>59.67</td></tr><tr><td>Qwen3-32B</td><td>76.90</td><td>82.86</td><td>54.44</td><td>70.20</td><td>84.26</td><td>90.71</td><td>60.00</td><td>73.83</td><td>57.80</td><td>62.16</td><td>56.40</td><td>66.49</td></tr><tr><td>DeepSeek-V3</td><td>77.83</td><td>80.27</td><td>47.01</td><td>61.42</td><td>75.93</td><td>84.34</td><td>53.68</td><td>70.05</td><td>58.96</td><td>54.68</td><td>52.78</td><td>65.34</td></tr><tr><td rowspan="2">LRM</td><td>QwQ-32B</td><td>77.36</td><td>81.04</td><td>55.01</td><td>66.62</td><td>77.78</td><td>86.21</td><td>49.47</td><td>69.95</td><td>57.94</td><td>58.59</td><td>60.21</td><td>65.54</td></tr><tr><td>DeepSeek-R1</td><td>53.82</td><td>65.46</td><td>53.55</td><td>66.02</td><td>73.08</td><td>83.97</td><td>61.05</td><td>70.13</td><td>58.16</td><td>62.98</td><td>61.34</td><td>67.12</td></tr><tr><td rowspan="5">SFT</td><td>SFT-Entropy only</td><td>74.60</td><td>79.15</td><td>53.88</td><td>62.67</td><td>79.63</td><td>88.17</td><td>64.21</td><td>72.34</td><td>48.59</td><td>49.23</td><td>47.03</td><td>60.93</td></tr><tr><td>SFT-LLM only</td><td>73.40</td><td>77.96</td><td>53.31</td><td>56.39</td><td>76.85</td><td>86.77</td><td>61.05</td><td>70.76</td><td>51.41</td><td>51.80</td><td>50.59</td><td>62.14</td></tr><tr><td>SFT-without IG</td><td>77.65</td><td>81.69</td><td>43.05</td><td>52.17</td><td>80.56</td><td>88.89</td><td>50.52</td><td>67.64</td><td>60.40</td><td>61.06</td><td>44.97</td><td>59.84</td></tr><tr><td>SFT-IN3 [21]</td><td>74.35</td><td>78.44</td><td>49.03</td><td>59.33</td><td>89.81</td><td>94.12</td><td>70.53</td><td>75.78</td><td>56.00</td><td>56.63</td><td>55.90</td><td>61.85</td></tr><tr><td>SFT-Full</td><td>79.60</td><td>82.86</td><td>55.17</td><td>67.38</td><td>85.19</td><td>91.40</td><td>71.58</td><td>74.65</td><td>61.99</td><td>62.40</td><td>62.27</td><td>66.91</td></tr><tr><td>Related Work</td><td>ClariLM [38]</td><td>81.25</td><td>85.48</td><td>52.50</td><td>63.55</td><td>89.72</td><td>94.36</td><td>66.32</td><td>72.68</td><td>64.23</td><td>67.61</td><td>62.89</td><td>65.36</td></tr><tr><td rowspan="2">Our</td><td>CLAIM-Agent</td><td>71.95</td><td>91.58</td><td>51.94</td><td>63.67</td><td>85.19</td><td>91.58</td><td>68.42</td><td>71.39</td><td>59.62</td><td>66.63</td><td>58.15</td><td>60.43</td></tr><tr><td>CLAIM</td><td>81.85</td><td>84.97</td><td>56.79</td><td>69.54</td><td>87.04</td><td>92.55</td><td>63.16</td><td>72.23</td><td>65.18</td><td>68.10</td><td>63.71</td><td>65.47</td></tr></table>

## 4.3 Baseline Models

We select a diverse set of baselines to comprehensively evaluate the efectiveness of CLAIM for open-domain clarification. These baselines span diferent model scales (8B, 32B, etc.), architectural paradigms (direct generation vs. explicit reasoning), and training strategies (zero-shot, supervised fine-tuning, and agent-based execution), enabling systematic comparison across both clarification decision and question generation behaviors. This design allows us to analyze clarification performance under diverse modeling assumptions while maintaining a unified evaluation protocol.

We first include a group of general-purpose LLMs as directanswering baselines, including Llama-3.1-8B, Qwen3-8B, Qwen3- 14B, Qwen3-32B, and DeepSeek-V3. These models are evaluated in a zero-shot setting without any explicit clarification decision mechanism, serving to characterize the default behavior of mainstream LLMs when handling ambiguous or underspecified user queries.

Next, to examine whether strong reasoning ability alone is sufi cient for reliable clarification, we consider two reasoning-oriented language reasoning models (LRMs), namely QwQ-32B and DeepSeek-R1. Although these models are not specifically trained for clarification, their chain-of-thought reasoning capabilities allow them to explicitly analyze semantic completeness and missing information, providing a useful comparison for understanding the role of reasoning in clarification decision-making.

We further report a group of supervised fine-tuned clarification models (SFT-based models) as training-based baselines. These mod els share the same base architecture but difer in the construction of supervision signals during the SFT stage, allowing controlled ablation of key components in the CLAIM framework. Specifically, SFT-Entropy only is trained using clarification decisions derived solely from entropy-based uncertainty estimation, without incorporating LLM-based semantic judgement. Conversely, SFT-LLM only relies exclusively on LLM-based judgement to determine whether clarification is required, without using entropy signals. To evaluate the efect of clarification question selection, we include SFT-without IG, which follows the full clarification judgement procedure but selects clarifying questions without information gain (IG)–based ranking. In addition, SFT-IN3 is fine-tuned using the training set provided by the IN3 dataset and focuses on clarification in taskoriented interaction scenarios. In contrast, SFT-Full is trained on the complete CLAIM synthetic dataset constructed with the full uncertainty-driven pipeline, covering both clarification-required and non-clarification scenarios in open-domain settings.

We also include ClariLM as a representative prior open-domain clarification method that is trained with large-scale supervised and preference-based data to jointly model clarification decisions and question generation [38]. We reproduce ClariLM following the original paper and evaluate it under the same evaluation protocol and metrics for fair comparison. Unless otherwise specified, none of the SFT-based models incorporate GRPO, ensuring that comparisons at this stage reflect the efects of diferent supervision signals and SFT strategies rather than reinforcement-based policy optimization. We do not include CLARINET or EVPI-based rankers as direct baselines because their inputs and objectives difer from CLAIM: they assume candidate clarifying questions or retrieval distributions and optimize question selection/ranking, whereas CLAIM jointly decides whether clarification is needed and what open-domain question to ask without a predefined candidate set.

CLAIM-Agent is additionally included as a special baseline to validate the soundness of the proposed uncertainty-driven synthetic data construction and clarification workflow. CLAIM-Agent directly executes the full clarification pipeline in an agent-style manner without supervised fine-tuning or policy optimization. Its performance therefore reflects the efectiveness of the proposed clarification process itself, rather than improvements induced by learned model parameters.

## 4.4 Overall Results

Table 1 reports the main evaluation results of CLAIM and all baseline models on three benchmarks. Overall, CLAIM achieves stateof-the-art (SOTA) or near-SOTA performance on the majority of metrics across all datasets, demonstrating the efectiveness of its uncertainty-driven framework in jointly deciding when clarification is needed and what information should be requested.

Comparing diferent model categories, zero-shot LLMs and reasoning oriented LRMs exhibit large performance variance across datasets. Although some large-scale models achieve competitive results on individual metrics, their overall performance remains unstable, suggesting that general language generation ability or reasoning capability alone is insuficient for reliable clarification. In contrast, SFT-based models are more robust, and CLAIM maintains strong and balanced performance across all three benchmarks.

Compared with the prior open-domain clarification method ClariLM, CLAIM matches or outperforms it on most metrics, particularly on CLAMBER and ClariLM-test. Notably, this performance is achieved using only approximately 10k uncertainty-constructed training instances, whereas ClariLM relies on around 120k supervised and preference-annotated examples [38]. This comparison highlights the data eficiency and scalability of CLAIM in opendomain clarification settings.

## 4.5 Further Analysis

Efect of Clarification Judgement Signals. Among SFT-based models, SFT-Entropy only and SFT-LLM only rely on a single clarification judgement signal during supervision. While both variants outperform zero-shot baselines, their performance remains consistently lower than that of SFT-Full across datasets. For example, on ClariLM-test, SFT-Full achieves an accuracy of 79.60 and a CDA score of 55.17, compared to 74.60/53.88 for SFT-Entropy only and 73.40/53.31 for SFT-LLM only. This gap indicates that entropy-based uncertainty estimation and LLM-based semantic judgement capture complementary aspects of query ambiguity, and that relying on either signal alone is insuficient for robust clarification decisions.

Efect ofInformation Gain for Clarifying Question Selection. Comparing SFT-without IG with SFT-Full isolates the efect of information gain–based clarification question selection. Although both models use the same clarification judgement mechanism, SFT-without IG shows clear degradation in clarification quality metrics, especially CDA and CQSS. On ClariLM-test, removing information gain reduces CDA from 55.17 to 43.05 and CQSS from 67.38 to 52.17, despite similar necessity prediction performance. This result suggests that explicitly selecting clarifying questions based on uncertainty reduction plays a critical role in identifying more informative clarification dimensions.

Efect of GRPO.. Finally, comparing SFT-Full and CLAIM isolates the contribution of group relative policy optimization (GRPO). While SFT-Full already exhibits strong overall performance, CLAIM further improves decision stability and clarification quality on multiple benchmarks. For instance, CLAIM improves ClariLM-test accuracy from 79.60 to 81.85 and CDA from 55.17 to 56.79, and also increases CLAMBER accuracy from 61.99 to 65.18. This improvement suggests that GRPO efectively refines the clarification decision boundary, particularly for queries with high semantic uncertainty.

CLAIM-Agent Analysis. CLAIM-Agent, which directly executes the full clarification pipeline without model training, achieves competitive performance across several metrics. On IN3, CLAIM-Agent achieves an F1 score of 91.58, and on CLAMBER it reaches an accuracy of 59.62, demonstrating its competitiveness with trained SFT models. This result validates the soundness of the proposed uncertainty-driven data construction and clarification workflow. Although CLAIM-Agent underperforms CLAIM in some metrics, CLAIM benefits from GRPO-based training with group-relative advantages, enabling the model to internalize clarification strategies through self-sampling and relative comparison during training.

Generalization Across Domains. The comparison between SFT-IN3 and SFT-Full further illustrates diferences in generalization behavior. SFT-IN3 achieves strong performance on the IN3 benchmark, with an accuracy of 89.81, but degrades noticeably on ClariLM-test and CLAMBER, where its accuracy drops to 74.35 and 56.00, respectively. In contrast, SFT-Full maintains consistently competitive results across all benchmarks, achieving accuracies of 79.60 on ClariLM-test, 85.19 on IN3, and 61.99 on CLAMBER. These results indicate that uncertainty-driven synthetic data construction enables robust clarification learning without overfitting to a specific domain. The comparison also reduces the attribution concern that improvements come merely from task-specific post-training: SFT-IN3, SFT-Entropy only, SFT-LLM only, SFT-without IG, and SFT-Full use comparable training recipes but difer in supervision source and selection mechanism.

## 4.6 LLM-as-a-Judge and Human Evaluation

While automated metrics provide a quantitative assessment of clarification performance, they are limited in capturing overall usefulness and interaction quality. Following prior work on LLM-based comparative evaluation [8], we conduct pairwise LLM-as-a-Judge evaluation using GPT-5, where each judge input contains the user query and two anonymized model outputs, and the judge returns Win, Tie, or Lose for CLAIM against a baseline. To calibrate this automatic evaluation, we further conduct a human study under the same pairwise protocol. Three expert annotators and a group of general users each evaluate 100 randomly sampled instances per baseline on IN3 and CLAMBER. Figure 3 summarizes the GPT-5 judge and human evaluation results together.

The human results are consistent with the GPT-5 evaluation trend: CLAIM obtains more wins than losses against most baselines, especially smaller zero-shot LLMs and reasoning models. The advantage is smaller against stronger 32B/V3 models, where ties increase, indicating that human judges often view both outputs as similarly useful when the baseline already asks a reasonable clarification question. Overall, the agreement between LLM-as-a-Judge and human evaluation supports the claim that CLAIM improves user-perceived clarification behavior rather than only optimizing automatic metrics.

![](images/9fe7a93f1ad5b8e17ee9fac0811556fc64dde6b5ff1e7c9629250a76f395b57a.jpg)  
Figure 3: Pairwise comparative evaluation of CLAIM against representative baselines under GPT-5 judge, expert, and general-user assessments. Each panel reports Win/Tie/Loss percentages over 100 sampled instances.

## 5 Conclusion

In this paper, we propose CLAIM, an uncertainty-driven framework for open-domain clarification in large language models. By leveraging semantic disagreement among multiple heterogeneous LLMs, CLAIM estimates answer uncertainty and synthesizes large-scale clarification data without relying on domain-specific resources. The uncertainty-aware clarification selection mechanism enables the model to efectively determine when clarification is necessary and which question to ask. We further adopt a two-stage training paradigm combining SFT and GRPO to enhance decision stability under high uncertainty. Extensive experiments on multiple benchmarks demonstrate that CLAIM consistently outperforms strong baseline models in both clarification necessity detection and clarifying question quality. This paper focuses on single-turn clarification; extending CLAIM to multi-turn interaction requires explicit dialogue-state tracking, history-dependent uncertainty estimation, and planning over future turns, which we leave for future work. These results highlight the efectiveness of uncertainty-aware modeling for general-domain clarification and suggest promising directions for building more adaptive and reliable human–LLM interaction systems.

## A Additional Analysis and Implementation Details

The uncertainty-driven data construction and clarification selection processes follow the methodology described in Section 3. To estimate answer distribution uncertainty, CLAIM samples candidate responses from multiple heterogeneous large language models $( k _ { 1 } = 5 )$ , including DeepSeek-V3, Qwen3-32B, GLM-4-32B-0414, Kimi-K2-Instruct-0905, and Ling-flash-2.0. Semantic clustering and entropy computation are performed using representations from Qwen3-Embedding-8B with cosine similarity. For clarifying question generation and user response simulation, CLAIM uses DeepSeek-V3 to generate $k _ { 2 } = 3$ candidate clarifying questions per query. All generation processes adopt a unified decoding temperature of 0.7. The prompt templates used by CLAIM-Agent and the evaluation scripts are released in the code repository.

Entropy Threshold. In all experiments, we set $\tau = 0 . 4 5$ as a single global threshold without tuning it separately for diferent benchmarks. This value follows a simple entropy-based intuition under our default $k _ { 1 } = 5$ setting: when only one model produces a semantically diferent answer, the smallest non-unanimous cluster distribution is (0.8, 0.2), whose entropy is −0.8 log 0.8 − 0.2 log $0 . 2 \approx 0 . 5 0$ We therefore set � slightly below this value so that one-model disagreement is treated as a weak but actionable uncertainty signal. The final decision is not determined by entropy alone, since LLMbased judgement and conflict resolution further correct boundary cases.

LLM Calls and Token Cost. CLAIM-Agent is used only for ofline data construction. For a non-clarification query, the pipeline uses at most $k _ { 1 }$ direct-answer calls, one LLM-based judgement call, and one optional arbitration call, i.e., no more than 7 LLM calls under $k _ { 1 } = 5 .$ . For a clarification query, the pipeline additionally uses $k _ { 2 }$ calls that jointly generate clarifying questions and simulated user answers, plus $k _ { 1 } k _ { 2 }$ post-clarification answer calls, resulting in no more than 25 LLM calls under $k _ { 1 } = 5 , k _ { 2 } = 3$ . In our implementation, constructing 1k synthetic examples consumes approximately 5.7M tokens. These calls are embarrassingly parallel and are not required at deployment time, where CLAIM runs as a single model.

Training Details. For training, we adopt Meta-Llama-3.1-8B-Instruct as the base model. The supervised fine-tuning (SFT) stage is implemented using LLaMA-Factory with LoRA parametereficient fine-tuning, where the LoRA rank is set to 8 and the LoRA alpha to 16. We use a learning rate of 5× $1 0 ^ { - 5 }$ and train for 3 epochs. The maximum sequence length is set to 2048, training uses bf16 precision and Flash Attention, the per-device batch size is 2 with 8 gradient accumulation steps, and gradient norms are clipped to 1.0. In the second stage, we apply GRPO using the TRL framework, initialized from the SFT-trained model. For each query, the policy samples $K = 4$ candidate outputs under temperature 0.7. Policy updates use clipping coeficient $\epsilon = 0 . 2$ and KL coeficient $\beta = 0 . 0 1$ with AdamW learning rate $1 \times { 1 0 } ^ { - 6 }$ . All local training experiments are conducted on four NVIDIA RTX 6000 Ada Generation GPUs.

## B Clarification-Ratio Diagnostics

We additionally report diagnostic clarification ratios for diferent uncertainty-construction strategies. The ground-truth column denotes the clarification-required ratio in each benchmark, Singlemodel multi-sample denotes repeated sampling from one model, and Multi-model single-sample denotes one sample from each heterogeneous model. This diagnostic is intended to test whether repeated sampling from a single strong model can replace heterogeneous model disagreement. Our preliminary observation is that within-model sampling may remain semantically concentrated for queries with strong priors, even under higher temperature, while heterogeneous models more often expose distinct interpretations. Semantic clustering is then applied before entropy computation to reduce surface-form sampling noise.

Table 2: Clarification-required ratios under diferent uncertainty sampling strategies.
<table><tr><td>Dataset</td><td>Ground Truth</td><td>Single-model</td><td>Multi-model</td></tr><tr><td>ClariLM-test</td><td>61.90</td><td>53.85</td><td>77.80</td></tr><tr><td>IN3</td><td>87.96</td><td>34.26</td><td>73.15</td></tr><tr><td>CLAMBER</td><td>50.00</td><td>31.01</td><td>61.09</td></tr></table>

## C Conflict Arbitration Statistics

Table 3 reports the proportion of samples for which entropy-based judgement and LLM-based judgement disagree and therefore require arbitration. The non-trivial ratios across both training and evaluation data support the need for conflict resolution rather than relying on a single judgement signal.

Table 3: Samples requiring conflict arbitration between entropy-based and LLM-based judgements.
<table><tr><td>Dataset</td><td>Quantity</td><td>Percentage</td></tr><tr><td>Training</td><td>5242</td><td>52.42</td></tr><tr><td>CLAMBER</td><td>1190</td><td>37.16</td></tr><tr><td>ClariLM-test</td><td>932</td><td>46.60</td></tr><tr><td>IN3</td><td>30</td><td>27.78</td></tr></table>

## D GenAI Usage Disclosure

In this paper, GenAI is primarily used for synthesizing partial data in the methodology, with the relevant details explicitly stated in the main text. Additionally, while GenAI is not employed in drafting the manuscript from scratch, it (GPT-5.2) is utilized for error checking (including grammar, tense, etc.) after manual completion.

## References

[1] Mohammad Aliannejadi, Leif Azzopardi, Krisztian Balog, and Mark Sanderson. 2019. Qulac: A Dataset for Evaluating Clarifying Questions in Conversational Search. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, Paris, France, 285–294. doi:10. 1145/3331184.3331226

[2] Mohammad Aliannejadi, Julia Kiseleva, Aleksandr Chuklin, Jef Dalton, and Mikhail Burtsev. 2020. ConvAI3: Generating Clarifying Questions for Open-Domain Dialogue Systems (ClariQ). arXiv:2009.11352 [cs.CL] https://arxiv.org/ abs/2009.11352 arXiv preprint / shared task overview.

[3] Mohammad Aliannejadi, Hamed Zamani, Fabio Crestani, and W. Bruce Croft. 2019. Asking Clarifying Questions in Open-Domain Information-Seeking Con versations. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, New York, NY, USA, 475–484. doi:10.1145/3331184.3331265

[4] Yizhou Chi, Jessy Lin, Kevin Lin, and Dan Klein. 2024. CLARINET: Augmenting Language Models to Ask Clarification Questions for Retrieval. arXiv:2405.15784 [cs.CL] https://arxiv.org/abs/2405.15784 arXiv preprint.

[5] Shihan Dou, Shichun Liu, Yuming Yang, Yicheng Zou, Yunhua Zhou, Shuhao Xing, Chenhao Huang, Qiming Ge, Demin Song, Haijun Lv, Songyang Gao, Chengqi Lv, Enyu Zhou, Honglin Guo, Zhiheng Xi, Wenwei Zhang, Qipeng Guo, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Tao Gui, and Kai Chen. 2025. Pre-Trained Policy Discriminators are General Reward Models. arXiv:2507.05197 [cs.LG] https://arxiv.org/abs/2507.05197 arXiv preprint.

[6] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. Nature 630, 8017 (2024), 625–630. doi:10.1038/s41586-024-07421-0

[7] Yujian Gan, Changling Li, Jinxia Xie, Luou Wen, Matthew Purver, and Massimo Poesio. 2024. ClarQ-LLM: A Benchmark for Models Clarifying and Requesting Information in Task-Oriented Dialog. arXiv:2409.06097 [cs.CL] https://arxiv.org/ abs/2409.06097 arXiv preprint.

[8] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Lionel Ni, Jian Guo, and Wen Gao. 2024. A Survey on LLM as-a-Judge. arXiv:2411.15594 [cs.CL] https://arxiv.org/abs/2411.15594 arXiv preprint.

[9] Minsuh Joo and Hyunsoo Cho. 2025. Cleanse: Uncertainty Estimation Approach Using Clustering-based Semantic Consistency in LLMs. In Proceedings ofthe Fourth Workshop on Generation, Evaluation and Metrics (GEM<sup>2</sup>). Association for Computational Linguistics, Online, 291–301. https://aclanthology.org/2025.gem-1.25/

[10] Saurabh Kadavath, Aman Arora, John Schulman, Tom Henighan, Jacob Steinhardt, Jared Kaplan, Prafulla Dhariwal, and Dario Amodei. 2022. Language Models (Mostly) Know What They Know. arXiv:2207.05221 [cs.CL] https://arxiv.org/ abs/2207.05221 arXiv preprint.

[11] Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, and Yarin Gal. 2024. Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv:2406.15927 [cs.CL] https://arxiv.org/abs/2406.15927 arXiv preprint.

[12] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. arXiv:2302.09664 [cs.CL] https://arxiv.org/abs/2302.09664 arXiv preprint.

[13] Dongryeol Lee, Segwang Kim, Minwoo Lee, Hwanhee Lee, Joonsuk Park, Sang Woo Lee, and Kyomin Jung. 2023. Asking Clarification Questions to Handle Ambiguity in Open-Domain QA. In Findings ofthe Association for Computational Linguistics: EMNLP 2023. Association for Computational Linguistics, Singapore, 11526–11544. doi:10.18653/v1/2023.findings-emnlp.772

[14] Belinda Z. Li, Been Kim, and Zi Wang. 2025. QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks? arXiv:2503.22674 [cs.AI] https://arxiv.org/abs/2503.22674 arXiv preprint.

[15] Wenhan Liu, Ziliang Zhao, Yutao Zhu, and Zhicheng Dou. 2024. Mining Ex ploratory Queries for Conversational Search. In Proceedings ofThe Web Conference 2024. Association for Computing Machinery, New York, NY, USA, 1386–1394. doi:10.1145/3589334.3645424

[16] Jiaju Ma, Lei Shi, Kenneth Robertsen, and Peggy Chi. 2025. AmbigChat: Interactive Hierarchical Clarification for Ambiguous Open-Domain Question Answering. In Proceedings ofthe 38th Annual ACM Symposium on User Interface Software and Technology. Association for Computing Machinery, New York, NY, USA, 141:1–141:18. doi:10.1145/3746059.3747686

[17] Bodhisattwa Prasad Majumder, Sudha Rao, Michel Galley, and Julian McAuley. 2021. Ask What’s Missing and What’s Useful: Improving Clarification Question Generation Using Global Knowledge. In Proceedings ofthe 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Online, 4300– 4312. https://aclanthology.org/2021.naacl-main.340/

[18] Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2020. AmbigQA: Answering Ambiguous Open-domain Questions. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Online, 5239–5251. doi:10.18653/v1/2020.emnlpmain.466

[19] Ali Montazeralghaem, Guy Tennenholtz, Craig Boutilier, and Ofer Meshi. 2025. Asking Clarifying Questions for Preference Elicitation With Large Language Models. arXiv:2510.12015 [cs.AI] https://arxiv.org/abs/2510.12015 arXiv preprint.

[20] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training Language Models to Follow Instructions with Human Feedback. In Advances in Neural Information Processing Systems, Vol. 35. Curran Associates, Inc., Red Hook, NY, USA, 27730–27744. doi:10.5555/3600270.3602281

[21] Cheng Qian, Yuhan Liu, Zhenzhong Lan, Yixuan Liu, Jing Zhang, and Minlie Huang. 2024. Tell Me More! Towards Implicit User Intention Understanding in Agent Interaction. In Proceedings of the 62nd Annual Meeting of the Association for

Computational Linguistics. Association for Computational Linguistics, Bangkok, Thailand, 1114–1139. https://aclanthology.org/2024.acl-long.61/

[22] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Chelsea Finn, and Sergey Levine. 2023. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. In Advances in Neural Information Processing Systems, Vol. 36. Curran Associates, Inc., Red Hook, NY, USA. https://arxiv.org/abs/2305. 18290

[23] Sudha Rao and Hal Daumé III. 2018. Learning to Ask Good Questions: Ranking Clarification Questions Using Neural Expected Value of Perfect Information. In Proceedings ofthe 56th Annual Meeting ofthe Association for Computational Linguistics. Association for Computational Linguistics, Melbourne, Australia, 3669–3680. doi:10.18653/v1/P18-1340

[24] Anselm Rothe, Brenden M. Lake, and Todd M. Gureckis. 2017. Question Asking as Program Generation. arXiv:1711.06351 [cs.CL] https://arxiv.org/abs/1711.06351 arXiv preprint.

[25] Marzieh Saeidi, Max Bartolo, Patrick Lewis, Sameer Singh, Tim Rocktäschel, Mike Sheldon, Guillaume Bouchard, and Sebastian Riedel. 2018. Interpretation of Natural Language Rules in Conversational Machine Reading. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Brussels, Belgium, 2087–2097. doi:10.18653/v1/ D18-1233

[26] Ivan Sekulić, Mohammad Aliannejadi, and Fabio Crestani. 2021. Towards Facet Driven Generation of Clarifying Questions for Conversational Search. In Proceedings ofthe 2021 ACM SIGIR International Conference on Theory ofInformation Retrieval. Association for Computing Machinery, New York, NY, USA, 167–175. doi:10.1145/3471158.3472257

[27] Claude E. Shannon. 1948. A Mathematical Theory of Communication. Bell System Technical Journal 27, 3 (1948), 379–423, 623–656

[28] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeek Math: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300 [cs.CL] https://arxiv.org/abs/2402.03300 arXiv preprint.

[29] Wenxuan Wang, Juluan Shi, Zixuan Ling, Yuk-Kit Chan, Chaozheng Wang, Cheryl Lee, Youliang Yuan, Jen-tse Huang, Wenxiang Jiao, and Michael R. Lyu. 2025. Learning to Ask: When LLM Agents Meet Unclear Instruction. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Suzhou, China, 21773–21784. doi:10. 18653/v1/2025.emnlp-main.1104

[30] Xuezhi Wang, Jason Wei, Dale Schuurmans, Maarten Bosma, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain-of-Thought Reasoning in Language Models. In Proceedings ofthe International Conference on Learning Representations. https: //openreview.net/forum?id=1PL1NIMMrw

[31] Julia White, Gabriel Poesia, Robert Hawkins, Dorsa Sadigh, and Noah Goodman. 2021. Open-domain clarification question generation without question examples. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 563–570. doi:10.18653/v1/2021.emnlp-main.44

[32] Jie JW Wu. 2023. Large Language Models Should Ask Clarifying Questions to Increase Confidence in Generated Code. arXiv:2308.13507 [cs.SE] https: //arxiv.org/abs/2308.13507 arXiv preprint.

[33] Yifei Yuan, Clemencia Siro, Mohammad Aliannejadi, Maarten de Rijke, and Wai Lam. 2024. Asking Multimodal Clarifying Questions in Mixed-Initiative Conversational Search. In Proceedings ofthe ACM Web Conference 2024. Association for Computing Machinery, New York, NY, USA, 1474–1485. doi:10.1145/3589334. 3645483

[34] Hamed Zamani, Johanne R. Trippas, Jef Dalton, and Filip Radlinski. 2020. MIM-ICS: A Large-Scale Data Collection for Search Clarification. In Proceedings ofthe 29th ACM International Conference on Information and Knowledge Management. ACM, Galway, Ireland, 3189–3198. doi:10.1145/3340531.3412772

[35] Michael J. Q. Zhang, W. Bradley Knox, and Eunsol Choi. 2024. Modeling Future Conversation Turns to Teach Large Language Models to Ask Clarifying Questions. arXiv:2410.13788 [cs.CL] https://arxiv.org/abs/2410.13788 arXiv preprint; submitted to ICLR 2025.

[36] Tong Zhang, Jiali Mao, Shunyu Yao, Rui Wang, and Yixin Cao. 2024. CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs in Large Language Models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics. Association for Computational Linguistics, Bangkok, Thailand, 10718–10735. https://aclanthology.org/2024.acl-long.578/

[37] Yiming Zhang, Lingfei Wu, Qi Shen, Yitong Pang, Zhihua Wei, Fangli Xu, Bo Long, and Jian Pei. 2022. Multiple Choice Questions Based Multi-Interest Policy Learning for Conversational Recommendation. In Proceedings ofthe ACM Web Conference 2022. ACM, Lyon, France, 2153–2162. doi:10.1145/3485447.3512088

[38] Ziliang Zhao, Haonan Chen, Shiren Song, Jian Xie, and Zhicheng Dou. 2025. ClariLM: Enhancing Open-domain Clarification Ability for Large Language Models. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management. Association for Computing Machinery, New York, NY, USA, 4401–4411. doi:10.1145/3746252.3761068

[39] Ziliang Zhao, Zhicheng Dou, and Yujia Zhou. 2024. Generating Intent-aware Clarifying Questions in Conversational Information Retrieval Systems. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. Association for Computing Machinery, New York, NY, USA, 3384–3394. doi:10.1145/3627673.3679851

[40] Zhanke Zhou, Xiao Feng, Zhaocheng Zhu, Jiangchao Yao, Sanmi Koyejo, and Bo Han. 2025. From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information? arXiv:2506.08295 [cs.CL] https://arxiv.org/abs/2506.08295 arXiv preprint.
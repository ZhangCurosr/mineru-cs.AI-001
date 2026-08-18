# CUBICS: Situation-aware performance estimation for safety-relevant ML components

Benjamin Herd <sup>∗</sup>, Jessica Kelly <sup>∗</sup>, and Mario Trapp <sup>†</sup> <sup>∗</sup>

<sup>∗</sup>Fraunhofer Institute for Cognitive Systems IKS

Garching, Germany

{benjamin.herd, jessica.kelly}@iks.fraunhofer.de

<sup>†</sup>Technical University of Munich

Garching, Germany

mario.trapp@tum.de

Abstract—Machine learning (ML) is a key technology driving innovation today, but ensuring ML safety remains a major challenge for safety-related applications. A promising idea is to build proven-in-use arguments from field data, e.g. by running ML components (MLCs) in shadow mode or within safety envelopes so that their outputs can be monitored as ‘safe probes’ without affecting safety. These probes can then be used to build a statistical argument about field performance in a Bayesian way. However, many Bayesian field-data approaches in safety engineering model failures as a simple Bernoulli (or binomial) process with a single global failure probability and i.i.d. trials, which is rarely adequate for MLCs whose performance depends strongly on context. Statistical evidence is also about coverage of relevant situations, including edge cases, and building a single integrated statistical model for the entire system is usually not feasible. To address these challenges, this paper introduces CUBICS, a context-modular framework for per-component, situation-aware performance estimation of safety-relevant ML components. CUBICS partitions the operational design domain into situations and, for each safety-relevant component, defines a set of situation-specific assumptions and probabilistic guarantees that are represented and updated in a Bayesian manner using Subjective Logic (SL). By combining these guarantees with beliefs about how often each situation occurs, CUBICS derives an overall risk estimate for each component without requiring a monolithic system-level statistical model, and thus provides a building block for modular, field-data-based safety assurance.

Index Terms—continuous safety assurance, machine learning, safety contracts

## I. INTRODUCTION

The safety assurance of systems based on Machine Learning (ML) remains a paramount challenge that necessitates the development of innovative paradigms, such as continuous safety assurance. This often involves operating ML components (MLCs) in shadow mode, where they function without any safety-relevant impact on the system to collect in-field data and establish a foundation of statistical evidence.

Bayesian statistics provide a useful approach here: by treating each probe as a success or failure, a prior over the failure probability can be updated as data accumulates, yielding both an estimated failure probability and a measure of uncertainty. In this paper, we use the term ‘performance’ to refer to task-level ML metrics such as recall, and ‘reliability to refer to the probability that a safety-relevant component behaves correctly (i.e. achieves adequate performance) in a given situation. In practice, this is often instantiated as a simple Beta-Bernoulli model with a single global failure probability. For ML components whose performance is strongly contextdependent, this global i.i.d. assumption may mask situationspecific insufficiencies. For instance, a vision-based system is more likely to fail in heavy rain than in clear conditions.

To address this, we introduce CUBICS, a framework for per-component, situation-aware performance estimation of safety-relevant MLCs that can serve as a building block for modular, field-data-based safety assurance. CUBICS partitions the operational design domain (ODD) into discrete situations defined by context dimensions such as weather or lighting conditions. For each safety-relevant component and situation, CUBICS defines a set of situation-specific assumptions and probabilistic guarantees about relevant failure modes (e.g. false-negative detection). Within each situation we assume approximately stationary failure behaviour, so simple Bernoulli/Beta-style updates remain valid; at operation time, new evidence only updates the guarantees of the situations in which it may have occurred. This makes explicit which situations are well covered by evidence and where uncertainty about the component’s behaviour remains high.

Even with situation-based models, it can be argued that compiling sound, fully integrated statistical evidence for system-level claims (such as the positive risk balance of an automated driving system) is infeasible and prone to modelling errors. Building and maintaining a system-wide Bayesian network (BN) that captures all relevant dependencies is particularly difficult in practice. In current industrial practice, complete statistical evidence for an entire system is typically neither available nor required; instead, system safety cases combine quantitative arguments for selected components with qualitative reasoning for the rest. CUBICS therefore uses Subjective Logic (SL) as the primary calculus for representing and updating component-level, situation-specific statistical evidence using subjective opinions with explicit belief and uncertainty.

Prior work [10] has formalised safety contracts in SL and derived a corresponding assurance argument pattern that separates assumption sufficiency from system resilience. That work focused on a single, context-agnostic binary contract and static evidence. In this paper, we extend this approach towards explicit ODD-based situation modelling, per-component context-modular contracts, and runtime updates. For each safety-relevant component, CUBICS defines an assume–guarantee contract. The assumptions capture beliefs about the context in which the component operates (e.g. distributions over weather or lighting conditions), while the guarantees express probabilistic statements about the component’s safety-relevant behaviour, conditioned on these situations. Both assumptions and guarantees are represented as SL opinions that can be updated over time. By combining assumptions about situations with their conditional guarantees, CUBICS derives (i) situation-specific guarantees and (ii) a marginal, context-weighted overall risk contribution for the component. In principle, contracts of upstream components (e.g. perception) can be used as evidence for assumptions of downstream components (e.g. planning), but each contract can rely on different types of arguments (SL-based models, traditional safety analyses, or qualitative reasoning). In this paper, we instantiate and evaluate CUBICS for an individual component and the composition of contracts across multiple components is left as future work. This preserves componentlevel modularity over ODD situations, uses Bayesian statistical methods where they are most needed for ML components, and avoids a single, fragile system-wide statistical model in favour of feasible, component-specific models that can feed into the overall safety case. In particular, we make the following contributions:

• We introduce CUBICS, a contract-based methodology for safety-relevant ML components that structures assumptions as SL opinions over context dimensions and guarantees as SL opinions over situation-conditional failure behaviour, and derives both per-situation guarantees and a marginal, context-weighted risk contribution for an individual component.

• We develop a context-aware update mechanism for CUBICS contracts that allows for the distribution of (positive and negative) runtime evidence across situations under context uncertainty.

To investigate the effectiveness of our approach, we assess the following three research questions:

1) RQ1 (Internal validity): Can CUBICS recover known, situation-specific reliability patterns in a controlled scenario?

2) RQ2 (Benefit): Does CUBICS yield more informative and situation-specific reliability assessments compared to a pooled Bernoulli model, and how is this advantage impacted by varying data sizes?

3) RQ3 (Sensitivity): How sensitive are the guarantees to prior choices, misclassification of context, and data scarcity?

In the evaluation, we first instantiate CUBICS in a synthetic case study to assess its internal validity, i.e., whether it can recover known situation-specific reliability patterns and the corresponding marginal guarantee under controlled conditions (RQ1). We then apply CUBICS to a YOLOv12-based object detector trained on BDD100K, evaluating its behaviour on several safety-critical object classes (person, bicycle, car, etc.) and, due to space, report detailed results for the person class. We compare its situation-specific SL guarantees with a single global Bernoulli model to show that CUBICS exposes localized performance deficits and data gaps that the global model masks (RQ2). Finally, we analyse how the resulting guarantees change under different priors, imperfect context information, and varying evidence strength to assess the sensitivity of the approach to these factors (RQ3).

The paper is structured as follows. Section II introduces Subjective Logic; Section III presents the CUBICS methodology, including the contract model, situation-based ODD decomposition, and the derivation of conditional and marginal guarantees with continuous updates; Sections IV–V provide experimental results on the application of CUBICS to an ML-based safety-relevant component and demonstrate the resulting context-aware risk assessment; Section VI discusses threats to validity; Section VII reviews related work; Section VIII summarises findings and outlines avenues for future work.

## II. BACKGROUND

## A. Subjective Logic

Subjective Logic (SL) [13] is a framework for reasoning under uncertainty that combines ideas from probability theory and Dempster–Shafer evidence theory. Its core data structures are subjective opinions representing an agent’s belief, disbelief, and uncertainty about the truth of a proposition. SL provides algebraic operators for combining and transforming opinions. Depending on whether the underlying domain X is binary $( \mathrm { i . e . , } \ \mathbb { X } = \{ x , \bar { x } \} ,$ ) or n-ary (i.e., $\mathbb { X } = \{ x _ { 1 } , \ldots , x _ { K } \} )$ , opinions are binomial or multinomial. We focus here on multinomial opinions.

Definition 1 (Multinomial opinion): Let $\mathbb { X } = \{ x _ { 1 } , \ldots , x _ { K } \}$ be a finite domain of mutually exclusive and collectively exhaustive states. A multinomial opinion over X is a tuple $\omega _ { X } = ( \mathbf { b } _ { X } , u _ { X } , \mathbf { a } _ { X } )$ ) where:

$\mathbf { b } _ { X } = \left( b _ { x _ { 1 } } , \dots , b _ { x _ { K } } \right)$ (belief masses) is a distribution of belief over the states, with $b _ { x _ { i } }$ the belief mass supporting $x _ { i }$ being the true state;

• u<sub>X</sub> (uncertainty) is the remaining, uncommitted belief mass and the complement of confidence (1 − u);

$\mathbf { a } _ { X } = \left( a _ { x _ { 1 } } , \ldots , a _ { x _ { K } } \right)$ (base rates) is an a priori probability distribution over X in the absence of committed belief; and

$b _ { x _ { i } \underline { { { , } } } } u , a _ { x _ { i } } \in \ [ 0 , 1 ]$ for all i, $\textstyle \sum _ { i = 1 } ^ { K } b _ { x _ { i } } + u = 1$ , and $\textstyle \sum _ { i = 1 } ^ { K } a _ { x _ { i } } = 1$

A vacuous opinion (full uncertainty) is denoted $\boldsymbol { \omega } _ { V } = ( \mathbf { 0 } , 1 , \mathbf { a } )$ and an absolute opinion focusing all belief on a single state $x _ { i }$ is denoted $\omega _ { x _ { i } } ^ { \top } = ( \mathbf { b } , 0 , \mathbf { a } )$ with $b _ { x _ { i } } = 1$ and $b _ { x _ { j } } = 0$ for all $j \neq i ,$ , for any choice of base rate vector a.

1) Constructing Multinomial opinions: Given a finite domain $\mathbb { X } = \{ x _ { 1 } , \ldots , x _ { K } \}$ , evidence counts $\mathbf { r } = \left( r _ { x _ { 1 } } , \ldots , r _ { x _ { K } } \right)$ with $r _ { x _ { i } } \geq 0$ for each state $x _ { i } ,$ and a non-informative prior weight<sup>1</sup> W, a multinomial opinion can be computed as follows:

$$
b _ { x _ { i } } = \frac { r _ { x _ { i } } } { \sum _ { j = 1 } ^ { K } r _ { x _ { j } } + W } , i = 1 , \ldots , K\tag{1}
$$

$$
u = \frac { W } { \sum _ { j = 1 } ^ { K } r _ { x _ { j } } + W }\tag{2}
$$

with base rate vector $\mathbf { a } = ( a _ { x _ { 1 } } , \ldots , a _ { x _ { K } } )$ , where $a _ { x _ { i } } \in [ 0 , 1 ]$ and $\textstyle \sum _ { i = 1 } ^ { K } a _ { x _ { i } } = 1$

Multinomial opinions correspond to Dirichlet distributions over the categorical probabilities on X. Given $( \mathbf { r } , \mathbf { a } , W )$ , the corresponding parameters are

$$
\alpha _ { x _ { i } } = r _ { x _ { i } } + a _ { x _ { i } } W , \quad i = 1 , \ldots , K ,\tag{3}
$$

and the expectation value of $x _ { i }$ is

$$
E ( x _ { i } ) = b _ { x _ { i } } + a _ { x _ { i } } \cdot u\tag{4}
$$

2) Combining opinions: SL provides a wide range of combination operators [13]. Combining opinions provides an elegant and intuitive way to combine the underlying distributions, a direct manipulation of which would be significantly more complex. In this paper, we use the following operators (see [13] for full definitions):

a) Multinomial multiplication: given independent opinions $\omega _ { X }$ and $\omega _ { Y }$ about variables X and Y which take their values from distinct domains X and $\mathbb { Y } ,$ the joint opinion on the Cartesian product $\mathbb { X } \times \mathbb { Y }$ is computed using multinomial multiplication $\omega _ { X \wedge Y } = \omega _ { X } \cdot \omega _ { Y }$ . We denote the corresponding joint belief as $b _ { x y }$ , uncertainty as $u _ { x y }$ , and product base rate as $a _ { x y } .$ Further details on computing multinomial multiplication can be found in [13].

b) Multinomial deduction: given a conditional relationship where conclusion variable Y depends on premise variable X, the deduction operator derives the marginal opinion on Y from an opinion on X as $\omega _ { Y \parallel X } = \omega _ { X } \circledcirc \omega _ { Y \mid X }$ ,where:

$\omega _ { X }$ is the multinomial opinion on the premise X.

$\omega _ { Y \mid X }$ represents the set of conditional opinions on Y given the mutually exclusive states of X.

$\omega _ { Y \parallel X }$ is the resulting deduced marginal opinion on Y.

• ⊚ denotes the deduction operator.

Full details for computing the marginal opinion $\omega _ { Y \parallel X }$ are provided in [13].

## III. THE CUBICS METHODOLOGY

CUBICS structures the safety assurance of ML-based components (MLCs) around three central ideas:

1) a situation-based decomposition of the operational design domain (ODD) is performed;

2) modular safety contracts are defined per component and situation;

3) an SL-based approach is used (1) to update beliefs about the current situation and the safety of the component based on operation-time evidence, and (2) to obtain conditional and marginal component safety guarantees.

This section explains how these ideas jointly yield a contextaware assessment of a component’s safety guarantees as well as the overall risk contribution that can be embedded into a broader safety case. Throughout this section, we refer to the simplified scenario in Fig. 1 as a running example.

## A. Situation-based decomposition of the ODD

We assume that the relevant ODD of the system can be characterised by a set of discrete context dimensions $C _ { 1 } , \ldots , C _ { k }$ (e.g. weather, lighting). Each dimension $C _ { i }$ is a finite set of possible values, e.g. $C _ { 1 } = R a i n = \{ Y e s , N o \}$ and $C _ { 2 } =$ $W i n d = \{ L o w , H i g h \}$ . CUBICS partitions the ODD into situations $s \in S .$ , where $S { = } C _ { 1 } { \times } C _ { 2 } { \times } \cdots { \times } C _ { k }$ , and each situation $s = ( c _ { 1 } , \ldots , c _ { k } ) \in S$ corresponds to a particular combination of context values (e.g. $R a i n = Y e s , W i n d = H i g h )$ . Each such situation represents a subspace of the ODD that is assumed to have a distinct performance or risk profile for the component under consideration. For example, with $R a i n = \{ Y e s , N o \}$ and $W i n d = \{ L o w , H i g h \}$ we obtain four situations, and adding a third dimension $T i m e = \{ D a y , N i g h t \}$ yields the eight situations shown in Fig. 1.

The key assumption is that, in each situation $s _ { i }$ , the component’s failure behaviour can be treated as approximately stationary, so that a Bernoulli model of success and failure is more representative. More precisely, individual outcomes of the considered failure mode (e.g. misclassifications of an object) are exchangeable Bernoulli trials with an (approximately) constant failure probability $p _ { f a i l } ( s _ { i } )$ . In the contract, the resulting context model (assumptions over $C _ { i }$ and S) is linked to per-situation guarantees $\omega _ { G | s _ { i } }$ as introduced below.

## B. Contracts for safety-relevant components

For each MLC and situation $s _ { i }$ as defined above, CUBICS defines a safety contract with two central elements:

• Assumptions capture probabilistic beliefs about the current context, i.e. which values the context dimensions take, e.g. $ R a i n { = } \{ Y e s , N o \}$ , $W i n d { = } \{ L o w , H i g h \}$ , and thus which situation s<sub>i</sub> the component is believed to be operating in. These beliefs result from environment perception, operational profiles, or scenario-based analyses. As described below, beliefs are represented by multinomial opinions in SL which are then combined into a joint situational assessment.

• Guarantees express beliefs about the safety of the component or function, conditional on the current situation. They are underpinned by a safety case with appropriate evidence, such as design-time analyses, field or shadowmode data, and expert judgement. For each situation $s _ { i } ,$ belief in the guarantee is represented as a probabilistic assessment of the relevant safety claim G, encoded as a conditional opinion $\omega _ { G | s _ { i } }$

Assumptions and guarantees are the building blocks of a conditional model that links beliefs about operating in a certain situation $s _ { i }$ with beliefs about the safety of the component in $s _ { i }$ based on available evidence. Conceptually, this conditional model resembles a Bayesian Network (BN) but replaces conditional probability tables with conditional probability density functions. More precisely, CUBICS uses SL to represent assumptions and guarantees as follows:

![](images/9e6ea23846b98f0da6ec6114dbe7021f8fbae472f706da117dd054df1ebc0ae4.jpg)  
Fig. 1: Overview of the Unified CUBICS methodology over a simplified operational design domain. Subjective context opinions over Rain, Wind, and Time of Day make up eight situations (from s<sub>0</sub> to $s _ { 7 } )$ . Each situation establishes a unique safety contract $\textstyle ( G \mid s _ { i } )$ , which are ultimately aggregated into the overarching global guarantee $G \parallel S .$

![](images/e59d21e976373564c9d9a879c562cad9b42780812061b631b96e5cc559bec447.jpg)  
Fig. 2: The fractional evidential update mechanism. SL situational opinions $\omega _ { s _ { i } }$ provide a probabilistic weighting signal $E ( s _ { i } )$ to incoming field observations. Binary evidence is accumulated fractionally into Dirichlet parameters $( \alpha , \beta )$ , thereby refining the conditional guarantee $\omega _ { G | s }$ as more data is gathered

• Assumptions are modelled by first forming multinomial opinions $\omega _ { C _ { i } }$ about each context dimension $C _ { i } ;$ these opinions are then combined into a joint situational opinion $\begin{array} { r } { \omega _ { S } = \prod _ { i = 1 } ^ { k } \omega _ { C _ { i } } } \end{array}$

• Guarantees are situation-specific and are thus represented as a set $\omega _ { G | S } ~ = ~ \{ \omega _ { G | s _ { i } } ~ | ~ s _ { i } ~ \in ~ S ~ \}$ of conditional opinions. Conceptually, $\omega _ { G | S }$ forms a context-indexed family of opinions, one per situation $s _ { i }$

Using the example from Fig. 1, a belief such as $^ { * } i t$ is likely clear, not windy, and daytime” is first encoded in the individual context opinions (e.g. $\omega _ { R a i n } , \omega _ { W i n d } , \omega _ { T i m e } )$ and then combined into the joint situational opinion $\omega _ { S }$ , which in turn links directly to the corresponding per-situation guarantee, ${ \mathrm { e . g . ~ } } \omega _ { G | s _ { 4 } } { \mathrm { ~ f o r ~ } } ( R a i n = N o , W i n d = L o w , T i m e = D a y )$

The approach is component-level and context-modular: in a system with multiple interacting components, each component is equipped with its own CUBICS contract over ODD situations. In principle, the contracts of ‘upstream’ components (e.g. perception) can provide SL opinions over their guarantees that can be used as evidence for the assumptions of ‘downstream’ contracts (e.g. planning or actuation). In this way, guarantees do not need to be recomputed within a single monolithic probabilistic model: each component maintains its own context-dependent guarantees in the SL space, and the system-level safety case connects these contracts by treating upstream guarantees as inputs to downstream assumptions. This avoids a single, fragile system-wide statistical model in favour of component-specific models that can feed into the overall safety case. Based on this conditional model, both situation-specific guarantees and an overall risk contribution can be derived, as described below. Note that, in this paper, we instantiate and evaluate CUBICS at the level of individual components; the use of composed contracts across multiple components is left for future work.

## C. Conditional and marginal contract guarantees

A component contract gives rise to two closely related types of guarantees: the conditional view maintains one guarantee per situation – explicitly conditioned on $s _ { i } ~ \in ~ S ~ -$ while the marginal view aggregates these per-situation guarantees into a single overall guarantee under the modelled operational profile. We describe both views below.

1) Conditional view (one guarantee per situation): As defined in Section III-B, for each situation $s _ { i } \in S$ we maintain a conditional opinion $\omega _ { G | s _ { i } }$ over a binary guarantee domain $\textit { G } \left( \mathbf { e . g . ~ } S a f e , U n s a f e \right)$ . The collection of these opinions, $\omega _ { G | S } = \{ \omega _ { G | s _ { i } } \mid s _ { i } \in S \}$ , constitutes the conditional view of the contract. This view supports detailed, context-specific reasoning about the component in each situation.

2) Marginal view (one aggregated guarantee over all situations): In contrast, the marginal view considers the contract from the perspective of the actual operation of the system, where different situations occur with different probabilities according to the operational profile and context model. Here we are interested in a single overall opinion about the guarantee G that already takes into account how likely each situation is.

CUBICS obtains this overall opinion by combining: (i) the SL opinions over the context dimensions (assumptions), which induce a belief over situations $s _ { i } ,$ , and (ii) the conditional persituation opinions $\omega _ { G | s _ { i } }$ . This combination is performed by multinomial deduction in SL, which yields a single marginal opinion $\omega _ { G | | S }$ which, for simplicity, we abbreviate with $\omega _ { G } \mathrm { : }$

$$
\omega _ { G } : = \omega _ { G | | S } = \omega _ { S } \circledcirc \omega _ { G | S }
$$

where $\begin{array} { r } { \begin{array} { r } { \omega _ { S } \ = \ \prod _ { i = 0 } ^ { k } \omega _ { C _ { i } } } \end{array} } \end{array}$ denotes the joint opinion over all possible context dimensions $C _ { i }$ , computed via multinomial conjunction.<sup>2</sup> Intuitively, multinomial deduction weighs each per-situation guarantee $\omega _ { G | s }$ by the belief that the corresponding situation $s _ { i }$ actually occurs, and aggregates the results into one overall conclusion. The marginal opinion ω<sub>G</sub> can be understood as the component’s overall risk contribution under the modelled operational profile, i.e. as a probabilistic assessment (with uncertainty) of whether its guarantee will be satisfied during operation when all situations and their probability of occurrence are taken into account.

In summary, the conditional view provides a set $\omega _ { G | S }$ of per-situation guarantees $\omega _ { G | s _ { i } }$ , while the marginal view provides one aggregated opinion $\omega _ { G }$ . Both are derived from the same contract: engineers can inspect per-situation guarantees for detailed, context-specific reasoning, and use the marginal guarantee as a compact summary of the component’s overall risk contribution.

## D. Continuous update of situational beliefs and guarantees

CUBICS follows a Bayesian perspective: an important design principle is that neither the situational opinions $\omega _ { C _ { i } }$ about the individual context dimensions nor the conditional guarantee opinions $\omega _ { G | s }$ stated in a contract are static. As additional evidence becomes available, both sets of opinions are updated, which, in turn, influences both the situational joint opinion set $\omega _ { S }$ and the inferred marginal risk opinion $\omega _ { G }$ CUBICS focuses on runtime updates from statistical evidence obtained during operation or testing. Here, operationtime data from field or shadow-mode deployments, as well as additional test datasets, are mapped into SL opinions and fused with the existing contextual and conditional per-situation opinions $\omega _ { C _ { i } }$ and $\omega _ { G | s _ { i } }$ . In Bayesian terms, this corresponds to an update of the underlying Dirichlet parameters. Over time, this process refines beliefs and reduces uncertainty about both the occurrence probability of each situation $s _ { i }$ as well as the guarantees that can be given in $s _ { i }$ . CUBICS uses a fractional update mechanism to account for uncertainty in the assumptions about which situation we are in. Given opinion $\begin{array} { r } { \omega _ { S } = \prod _ { i = 1 } ^ { k } \omega _ { C _ { i } } } \end{array}$ derived from opinions $\omega _ { C _ { i } }$ on context parameters, we compute the expected probability $E ( s _ { i } )$ of being in situation $s _ { i }$ according to Eq. 4 and perform an update of the conditional guarantees $\omega _ { G | s _ { i } }$ by weighing observed evidence x by the corresponding expectation value:

$$
r _ { i } ^ { ( t + 1 ) } = r _ { i } ^ { ( t ) } + E ( s _ { i } ) \mathbb { I } [ x _ { t } = 1 ] ,\tag{5}
$$

$$
s _ { i } ^ { ( t + 1 ) } = s _ { i } ^ { ( t ) } + E ( s _ { i } ) \mathbb { I } [ x _ { t } = 0 ] .\tag{6}
$$

where $r _ { i }$ and $s _ { i }$ are the accumulated success and failure counts for situation $s _ { i }$ , and $\mathbb { I } [ \cdot ]$ is the indicator function. For example, if $\omega _ { S }$ projects $E ( s _ { 0 } ) = 0 . 8$ and $E ( s _ { 1 } ) = 0 . 2$ at time t and we observe a failure $( x _ { t } = 0 )$ , then the failure counts are updated by $s _ { 0 } ^ { ( t + 1 ) } = s _ { 0 } ^ { ( t ) } + \bar { 0 . 8 }$ and $s _ { 1 } ^ { ( t + 1 ) } = s _ { 1 } ^ { ( t ) } + 0 . 2$ . An overview of the fractional update mechanism is provided in Figure 2.

By operating in the SL space, CUBICS allows practitioners to transparently integrate heterogeneous statistical evidence sources and domain knowledge, and to reflect both supporting and adverse observations in the evolving contract. The resulting per-situation guarantees and marginal risk assessments are continuously updated in a Bayesian way and may serve as an input to higher-level assurance reasoning without requiring a monolithic, system-wide statistical model.

## IV. EXPERIMENTAL SET-UP

In this section, we describe our experimental setup, including the Python implementation of CUBICS, architectural decisions, training pipeline details, and used datasets. The source code is available at https://doi.org/10.5281/zenodo.21932463.

1) CUBICS Python implementation: The methodology has been implemented as a Python framework that provides executable support for defining context-dependent safety contracts, representing them in SL, and updating them continuously as new evidence becomes available. Conceptually, it mirrors the structure described in Section III. Context dimensions (e.g. weather, lighting, time of day) are represented as multinomial SL opinions and combined into a joint situational opinion over the Cartesian product of all context dimensions. For each resulting situation, the framework maintains a conditional opinion that captures the corresponding contract guarantees. These per-situation guarantees are linked to the situational opinion via the SL deduction operator, yielding a single marginal guarantee that summarises the component’s overall risk contribution under its operational profile. The python code also implements the fractional evidential update mechanism introduced in Section III-D. Incoming operation-time evidence is weighted by the expected value of the situations in which it may have occurred and accumulated into the corresponding per-situation guarantees. The implementation builds on standard Python scientific computing tools, facilitating integration into simulation environments and safety-assurance workflows.

2) Datasets: Experiments were conducted using the Berkeley DeepDrive (BDD100K) dataset [28]. It was selected due to its high diversity in driving scenarios and the inclusion of frame-level environmental attributes. The dataset consists of 100,000 annotated images, divided into a 70k/10k/20k split for training, validation, and testing, respectively. To support our analysis, the standard 10k validation set was stratified into mutually exclusive sub-datasets based on the weather attribute present in the original JSON annotations. The evaluated conditions include: Clear, Overcast, Rainy, Snowy, and Foggy. We pragmatically restrict our analysis to binary context dimensions, given the constraints of the existing BDD100K annotations. However, CUBICS itself is not limited to binary variables; because the SL opinions over context dimensions are multinomial, they can naturally represent multiple values per dimension. The models were trained to detect 10 standard classes: person, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, and traffic sign.

3) Model Architecture and Training: We used the YOLOv12-Large (YOLOv12l) architecture [23] as our primary baseline, initialized with pre-trained COCO weights to accelerate training. To preserve the fidelity of small, distant objects (such as pedestrians in low-visibility conditions), the input image resolution was scaled to 1024 × 1024 pixels. Training was conducted over 100 epochs.

4) Hardware: All experiments were executed on a single NVIDIA GeForce RTX 4090 GPU with 24GB of VRAM.

## V. EVALUATION

In this section we provide quantitative results addressing RQ1, RQ2, and RQ3. To address RQ1, we provide a synthetic case study to highlight the CUBICS methodology. We then evaluate RQ2 and RQ3 on the experimental setup outlined in Section IV.

## A. RQ1 (Internal validity): Can CUBICS recover known, situation-specific reliability patterns in a controlled scenario?

To address RQ1, we use a synthetic scenario that mimics a perception component operating in an environment with known situation-specific failure rates. This controlled setup allows us to verify whether CUBICS can reconstruct the underlying per-situation reliability $\mathrm { p a t t e r n s } ^ { 3 }$ and the resulting marginal guarantee from observed success/failure outcomes. We consider an abstract perception-based function (e.g. a pedestrian detector) in the context of automated driving whose failure behaviour is assumed to depend on three discrete context dimensions: $ R a i n { = } \{ Y e s , N o \}$ $W i n d = \{ H i g h , L o w \}$ and ${ \cal T } o D { = } \{ D a y , N i g h t \}$ . The Cartesian product of these dimensions induces eight distinct situations, each of which is assumed to have an approximately stationary failure probability. For each situation, we define a conditional guarantee about the component’s safety-relevant behaviour and combine them with beliefs about the occurrence of the situations to obtain a marginal, context-weighted risk contribution. All steps of this process are implemented and executed in Python.

a) Design-time: Following the CUBICS methodology, the assessment of the ML component is represented as a safety contract that is evaluated against the situations induced by the context dimensions Rain, Wind, and $T o D .$ . The contract assumptions capture beliefs about the current context and are represented as multinomial SL opinions ω<sub>R</sub>, ω<sub>W</sub>, and $\omega _ { T } .$ , which are then combined into a joint situational opinion $\omega _ { S }$ over the eight situations $S \ = \ R a i n \times W i n d \times T o D$ using multinomial multiplication. The contract guarantees are concerned with a single safety-relevant property G of the component (e.g. no safety-relevant miss occurs in a critical detection scenario) and are made explicit on a per-situation basis. At design time, engineers construct a safety case that argues, for each situation $s _ { i } \in S _ { : }$ , for bounds on the probability that G is satisfied when the component operates in $s _ { i } .$ The detailed argumentation (tests, analyses, expert judgement) is not modelled explicitly here; instead, its outcome is summarised as an SL opinion $\omega _ { G | s _ { i } }$ over the binary domain $G =$ $\{ S a f e , U n s a f e \}$ . The collection of these eight conditional opinions $\omega _ { G | S } = \{ \omega _ { G | s _ { i } } \mid s _ { i } \in S \}$ constitutes the conditional part of the contract and forms the design-time initialisation of the case study.

b) Operation-time.: At operation-time, we initialize both the context opinions $\omega _ { R } ,$ , ω<sub>W</sub>, ω<sub>T</sub> and the conditional guarantees $\omega _ { G | S }$ with near-complete uncertainty (vacuous priors). This represents a system deployed in shadow mode with minimal prior empirical data, relying instead on operationtime evidence to learn the behaviour across all conditions. We simulate a deployment over $N = 5 { , } 0 0 0$ discrete operational cycles. Each situation $s _ { i }$ is assigned a ground-truth physical failure rate $p _ { \mathrm { f a i l } } ( s _ { i } )$ . These rates reflect intuitive physical constraints; for instance, the failure probability is highest in the most adverse conditions $( R a i n = Y e s , W i n d = H i g h$ $T o D = N i g h t )$ and lowest in ideal conditions. During each simulated cycle, a situation is sampled from the ground-truth context distribution. The agent records this occurrence and accumulates environmental evidence to continuously update the base context opinions $\omega _ { R } , \omega _ { W }$ , and $\omega _ { T }$ . Simultaneously, the component’s performance w.r.t. its guarantees is simulated against the target situation’s $p _ { \mathrm { f a i l } } ( s _ { i } )$ , yielding a binary success (r) or failure (s) outcome. This specific evidence is mapped to the corresponding conditional opinion $\omega _ { G | s _ { i } }$ . In this synthetic setting, the true situation is observed without ambiguity, so the situational opinion $\omega _ { S }$ collapses to a point mass (i.e. $P ( s _ { i } ) = 1$ for the realised situation and 0 otherwise). Consequently, the fractional evidential update mechanism described in Section III-D reduces to a standard per-situation Bernoulli/Beta update; in scenarios with uncertain context, the same mechanism would distribute each observation fractionally across multiple situations according to $E ( s _ { i } )$

![](images/bd7e2ff7801013b4b0f4a1edd3f891afdbf7f183f1877cb20f9bf281f3ce0182.jpg)  
Fig. 3: Progression of beta distributions across situational guarantees. Dotted red line represents true failure probability across situations. The shaded beta distributions represent the final conditional guarantees.

c) Results: Figure 3 illustrates the evolution of the conditional safety guarantees $\left( \omega _ { G | s _ { i } } \right)$ for all eight context situations over $N \ : = \ : 5 { , } 0 0 0$ operational cycles. At iteration $0 ,$ the beliefs are initialized with vacuous priors, represented by the flat, uniform distribution (grey line). As the system accumulates evidence, epistemic uncertainty decreases, and the Beta distributions progressively sharpen (iterations 50 and 500). By iteration 5,000, the expected value of every conditional opinion has converged towards its respective groundtruth success probability (red dashed line). The final distributions exhibit varying degrees of uncertainty (represented by the width and peak density of the curves), accurately reflecting the underlying occurrence probabilities $E ( s _ { i } )$ of the respective situation $s _ { i } .$ Frequently encountered situations $( \mathrm { e } . \mathrm { g } . , \ s _ { 1 }$ and $s _ { 6 } )$ accumulate substantial evidence, resulting in highly confident, narrow density peaks. Conversely, rare situations $( { \bf e . g . } , s _ { 2 } \ { \mathrm { o r } } s _ { 5 } )$ accumulate less operational evidence, naturally resulting in wider distributions. This demonstrates that the CUBICS framework correctly preserves uncertainty where data is scarce, preventing the system from making overconfident safety claims about rare edge cases.

RQ1 Summary: In this controlled scenario, CUBICS converges towards the known situation-specific reliability patterns: per-situation guarantees $\omega _ { G | s _ { i } }$ recover the predefined failure rates within their uncertainty bounds. This indicates that, when its assumptions hold, CUBICS correctly captures situation-dependent reliability from observed success/failure outcomes.

B. RQ2 (Benefit): Does CUBICS yield more informative and situation-specific reliability assessments compared to a global pooled Bernoulli estimate, and how is this advantage impacted by varying available data sizes?

To answer RQ2, we analyse the YOLOv12l object detector on the BDD100K validation set as described in Section IV, and compare two different strategies for deriving a systemlevel recall guarantee from the same body of evidence for the person class. We repeated the analysis for other object classes and observed qualitatively similar behaviour; we therefore focus the following description primarily on the person class for brevity. Each ground-truth person instance detection is treated as a Bernoulli trial, so we use true positive (TP) and false negative (FN) counts as evidential successes and failures for the true-positive rate (recall):

1) Global pooled guarantee. We consider a global pooled estimate for comparison, where all TP and FN counts are pooled into a single opinion $\omega _ { \mathrm { g p } }$ that yields one contextagnostic overall guarantee.

2) SL marginal guarantee. The ODD is decomposed into k situations. A conditional opinion $\omega _ { G | s _ { i } }$ is formed from the per-situation evidence, a multinomial context opinion ω<sub>S</sub> captures the situation distribution, and the marginal $\omega _ { G \parallel S }$ is obtained via SL deduction [13]. This also yields one global guarantee, but it preserves both evidential and contextual uncertainty.

![](images/a1821fd7673164ebea8a3b6dd269f1287f2ceb8ba215e7493438410a664064be.jpg)  
Fig. 4: Beta PDFs for the person class. The global pooled estimate’s narrow peak (black) conceals the performance gap between a benign scenario (green) and an adverse one (red).

a) Situational guarantees expose masked performance insufficiencies.: Table I reports the per-situation opinions for the person class. The global pooled estimate concentrates at $E = 0 . 6 5 3$ with negligible uncertainty $( u \approx 1 0 ^ { - 4 } )$ , suggesting high confidence in a moderate recall. CUBICS reveals that this figure conceals situation-specific performance insufficiencies: Clear + Daytime achieves a comparable expected probability $( E ~ = ~ 0 . 6 5 0 )$ , whereas Foggy + Dawn/dusk collapses to $\textit { E } = \ 0 . 2 5 0$ with uncertainty elevated by two orders of magnitude $( u = 0 . 0 8 3 )$ . This adverse scenario contains only 22 observations, making the wide posterior an honest reflection of evidential scarcity.

TABLE I: CUBICS opinions for the person class. The global pooled estimate masks the performance collapse and high uncertainty in adverse scenarios such as Foggy + Dawn/dusk.
<table><tr><td>Scenario</td><td>TP (r)</td><td>FN (s)</td><td>Bel. (b)</td><td>Disbel. (d)</td><td>Unc. (u)</td><td>Exp. Prob (E)</td></tr><tr><td>Global Pooled</td><td>16101</td><td>8549</td><td>0.6531</td><td>0.3468</td><td>0.0001</td><td>0.6532</td></tr><tr><td>Clear + Daytime</td><td>2758</td><td>1484</td><td>0.6499</td><td>0.3497</td><td>0.0005</td><td>0.6501</td></tr><tr><td> $\mathrm { R a i n y } + \mathrm { N i g h t }$ </td><td>218</td><td>157</td><td>0.5782</td><td>0.4164</td><td>0.0053</td><td>0.5809</td></tr><tr><td> $\mathrm { F o g g y + D a w n / D u s k }$ </td><td>5</td><td>17</td><td>0.2083</td><td>0.7083</td><td>0.0833</td><td>0.2500</td></tr></table>

Figure 4 visualizes this contrast. The global pooled estimate and Clear + Daytime distributions overlap as narrow peaks near $p ~ \approx ~ 0 . 6 5$ , while the Foggy + Dawn/dusk distribution is wide and shifted leftward – both lower in expectation and less certain. A single pooled recall figure is therefore neither representative of benign conditions (where the system performs adequately) nor of adverse ones (where it does not).

b) Marginal vs. global pooled guarantees: When the situational opinions are aggregated via SL deduction into a single marginal guarantee $\omega _ { G \parallel S }$ and compared with the global pooled opinion (Figure 5, Table II), the expected guarantee $E ( S a f e )$ is identical (0.6532 for the person class). The critical difference is the uncertainty mass: the marginal carries $u = 1 . 7 4 \times 1 0 ^ { - 3 }$ , roughly 21× the global pooled value of $u = 8 . 1 0 \times 1 0 ^ { - 5 }$ . This gap arises because the pooled view compresses two additional sources of uncertainty into a single narrow posterior: (i) conditional uncertainty from data-scarce situations, whose wide per-situation posteriors are averaged away by the pooled count, and (ii) context uncertainty from the finite-sample estimate of how often each situation occurs in operation. The global pooled opinion still reflects finitesample uncertainty via its u, but only at this aggregated level and without distinguishing where the evidence comes from. This behaviour is consistent across classes (with results for Bike, Car, and Truck shown in Table II).

![](images/96e2d87f4bf6b27ee9f3040005c0ab94e7f8c2c436cb1cce58de9c3a2cc1db99.jpg)  
Fig. 5: Marginal (top, blue) vs. context-agnostic pooled guarantee (bottom, red). The marginal distribution is wider and reflects the compositional uncertainty that pooling conceals.

TABLE II: Marginal vs. global pooled guarantee across classes. Both share the same expected value; the SL marginal exposes ∼15–25× more uncertainty depending on the class.
<table><tr><td></td><td></td><td colspan="2">Uncertainty Mass (u)</td><td></td></tr><tr><td>Class</td><td>E[x]</td><td>Marginal</td><td>Pooled</td><td>Uncertainty Ratio</td></tr><tr><td>Person</td><td>0.6532</td><td> $1 . 7 4 \times 1 0 ^ { - 3 }$ </td><td> $8 . 1 0 \times 1 0 ^ { - 5 }$ </td><td>21.4×</td></tr><tr><td>Bike</td><td>0.5695</td><td> $1 . 5 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 0 0 \times 1 0 ^ { - 3 }$ </td><td>15.2×</td></tr><tr><td>Car</td><td>0.7775</td><td> $2 . 4 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 0 0 \times 1 0 ^ { - 5 }$ </td><td>24.6×</td></tr><tr><td>Truck</td><td>0.6575</td><td> $4 . 8 2 \times 1 0 ^ { - 3 }$ </td><td> $2 . 3 0 \times 1 0 ^ { - 4 }$ </td><td>21.0×</td></tr></table>

Depending on the situation at hand, the global pooled guarantee may therefore be overconfident: its narrow credible interval implicitly assumes constant operating conditions. The SL marginal provides a calibrated account of what is known (and what remains uncertain) about system-level reliability across a heterogeneous ODD. CUBICS thus enables more targeted decisions than a global model. An engineer can (i) identify situations where the expected recall is below a required threshold and uncertainty is low, indicating systematic weakness and the need for model improvement, and (ii) identify situations where recall appears adequate but uncertainty remains high, indicating that additional situation-targeted testing is required before deployment. In our example, the Foggy + Dawn/dusk situation falls into the latter category, which would justify either restricting deployment in such conditions or prioritising data collection and testing there. From a system safety perspective, these per-situation and marginal guarantees can be used to derive concrete ODD restrictions, prioritise additional testing and data collection in high-uncertainty situations, and allocate the detector’s quantitative contribution to system-level risk in the safety case.

RQ2 Summary. CUBICS identifies localized performance insufficiencies (e.g. Foggy + Dawn/dusk: $E { = } 0 . 2 5 , \ u { = } 0 . 0 8 3 )$ that the global pooled guarantee masks behind a narrow, overconfident estimate. The SL marginal guarantee retains substantially more uncertainty than the pooled context-agnostic view while recovering the same expected value, replacing false precision with transparent, actionable safety insights.

## C. RQ3 (Sensitivity): How sensitive are the guarantees to prior choices, misclassification of context, and data scarcity?

We assess the sensitivity of the CUBICS framework to the choice of base-rate prior, misclassification of context, and data scarcity as follows:

a) Prior Choice: To investigate the effect of the choice of prior on the results, we vary the prior and observe the effect on the situation-specific guarantees. Table III demonstrates that in a data-rich ODD $( s _ { C D } = C l e a r D a y , N = 4 , 2 4 2 )$ , the prior has very little impact on the safety guarantee. Because uncertainty u is so low (0.0005), the evidence $( r , s )$ overwhelms the prior and anchors the expected probability $E ( S a f e \mid s _ { \mathrm { c d } } )$ at 0.65. In contrast, the data-scarce ODD $( s _ { f d } = F o g g )$ Dawn, $N = 2 2 )$ exhibits significant uncertainty $( u = 0 . 0 8 3 3 )$ . Here, the prior becomes a stronger lever: shifting from a pessimistic $( a \ =$ 0.1) to an optimistic $( a = 0 . 9 )$ prior moves the guarantee by roughly 3%. This behavior is safety-desirable; it ensures that in situations where empirical data is lacking, the assessment is forced to rely on explicit prior assumptions rather than making overconfident claims from insufficient evidence.

TABLE III: RQ3: Sensitivity to the Base Rate Prior (a) in Data-Rich vs. Data-Scarce ODDs
<table><tr><td>Prior (a)</td><td>Clear Day  $( N = 4 , 2 4 2 )$   $E ( S a f e \mid s _ { \mathrm { c d } } )$ </td><td>Foggy Dawn  $( N = 2 2 )$   $E ( S a f e \mid s _ { \mathrm { f d } } )$ </td></tr><tr><td>0.1</td><td>0.6499  $( u = 0 . 0 0 0 5 )$ </td><td>0.2167  $\left( u = 0 . 0 8 3 3 \right)$ </td></tr><tr><td>0.5</td><td>0.6501  $( u = 0 . 0 0 0 5 )$ </td><td>0.2500  $\left( u = 0 . 0 8 3 3 \right)$ </td></tr><tr><td>0.9</td><td>0.6503  $( u = 0 . 0 0 0 5 )$ </td><td>0.2833  $\left( u = 0 . 0 8 3 3 \right)$ </td></tr></table>

b) Context Misclassifications: We simulated an imperfect context classifier by transferring varying percentages of detection evidence (from 0% to 50%) out of a benign, datarich source environment (Clear Day) and falsely attributing it to two distinct adverse Operational Design Domains (ODDs):

• Data-rich target: Snowy Night (substantial evidence).

• Data-scarce target: Foggy Dawn (minimal evidence).

By incrementally “poisoning” the target ODDs with evidence from the easier, higher-performing Clear Day scenario $s _ { \mathrm { c d } } ,$ we tracked the resulting shift in the expected probability $E ( S a f e \mid s _ { [ \mathrm { s n / f d } ] } )$ . The results are shown in Figure 6. As the proportion of misclassified evidence increases, the persituation guarantees for the target ODDs become more optimistic and less distinct from the original source ODD. For the data-rich target, the effect on $E ( S a f e \mid s _ { \mathrm { s n } } )$ remains moderate because misclassified samples are diluted by existing evidence; for the data-scarce target, even small amounts of misclassification have a visible impact on $E ( S a f e \mid { \it s } _ { \mathrm { f d } } )$ In both cases, however, the associated uncertainty u remains higher than in the original source ODD and increases with poisoning, and the overall behaviour gradually approaches that of a global Bernoulli model. This indicates that context misclassification mainly erodes some of the benefit of situationspecific guarantees rather than leading to overconfident, falsely precise assessments.

![](images/8f467413fb4279c2aae4c4bded2eacd34fe0e56aabfd94f3313a1ae2ffe28cfe.jpg)  
Fig. 6: Sensitivity of situation-specific expected guarantee $E ( S a f e \mid { } s _ { [ \mathrm { s n / f d } ] } )$ to context misclassification. Increasing amounts of Clear Day evidence are incorrectly assigned to (a) a data-rich target (Snowy Night) and (b) a data-scarce target (Foggy Dawn), gradually inflating guarantees and reducing the distinction between situations.

c) Data Scarcity: We describe the advantage of CUBICS in scenarios where data is scarce by examining the edge case of detecting a truck during a foggy night. In our test set, this scenario occurred only once and resulted in a missed detection $( T P = 0 , F N = 1 )$ . A traditional frequentist evaluation yields a recall of exactly $0 . 0 \% - \mathrm { \ a }$ brittle and overconfident penalty derived from a single sample. CUBICS avoids this deterministic trap. Recognizing the severe data scarcity $( N = 1 )$ , it assigns a dominant epistemic uncertainty mass $( u = 0 . 6 6 6 7 )$ Consequently, the expected guarantee is not anchored to 0%, but is mathematically pulled toward the non-informative base rate prior $( a = 0 . 5 )$ , resulting in $E ( S a f e \mid s _ { f n } ) = 0 . 3 3 3 3$ This demonstrates CUBIC’s capacity for graceful degradation: rather than outputting an absolute (and statistically insignificant) safety claim, it formally bounds the localized risk with uncertainty and signals to the safety monitor that this specific ODD requires further targeted testing.

Overall, across all three experiments, realistic variations in priors, context classification quality, and sample size affect the numerical values of $E ( S a f e \mid s _ { x } )$ and $u ,$ but did not change the qualitative risk ordering between situations.

RQ3 Summary: CUBICS behaves robustly with respect to prior choices and moderate context misclassification: in data-rich situations, different reasonable priors are quickly overridden by evidence and lead to similar guarantees, whereas in data-scarce situations prior assumptions visibly influence $E ( S a f e \mid s _ { x } )$ but are accompanied by high uncertainty u. Context misclassification gradually reduces the benefit of situationspecific guarantees and makes the model resemble a global Bernoulli view, but for realistic misclassification levels the qualitative conclusions and risk ordering between situations remain stable. Overall, CUBICS is sensitive in the intended way: it exposes uncertainty where evidence is weak, without being overly brittle to plausible modelling and sensing imperfections.

## VI. THREATS TO VALIDITY

## A. Internal

Situational decomposition relies on correct per-image metadata; as a consequence, mislabelled metadata routes evidence to the wrong conditional opinion and distorts the corresponding situational and marginal guarantee. All opinions use the standard SL non-informative prior weight W=2; while reasonable for uninformed initialisation, it does not reflect domainspecific prior knowledge (e.g. from expert judgement), so posteriors for situations with sparse evidence remain sensitive to this choice. Finally, the update function treats every detection outcome within a situation as an exchangeable Bernoulli trial. In practice, FNs differ in safety relevance — missing a nearby, fully visible pedestrian is more critical than missing a distant, heavily occluded one — and failures may cluster in subconditions (e.g. dark clothing at night) that the current situation granularity does not distinguish. Incorporating severityweighted evidence or finer situation partitions could address this, at the cost of increased data requirements.

## B. External Validity

The evaluation uses TP/FN counts for a single object class, which naturally map to a binomial model. A threat to external validity is that many perception failures are continuous; extending CUBICS to such metrics would require replacing the binomial with an appropriate continuous-outcome model. Scalability is also an issue: with k context dimensions of cardinality $n _ { i } ,$ the number of situations grows as $\textstyle \prod _ { i } n _ { i } ,$ and a realistic ODD with dozens of dimensions would cause a combinatorial explosion, with many cells having sparse evidence and guarantees dominated by the prior. Hierarchical or factored representations of the situation space are a natural mitigation. It is important to note that SL itself does not pose a computational bottleneck: the SL operators we use scale linearly with the number of situations. Furthermore, the framework assumes a fixed failure rate within each situation, although ML model performance can drift over time. Because cumulative fusion weights all historical evidence equally, outdated observations are never discounted, which may yield overconfident guarantees that no longer reflect current system behaviour. Finally, the marginal guarantee assumes that the defined situations are exhaustive. Novel or adversarial conditions not anticipated in the ODD specification fall outside the framework, and the marginal, which aggregates over defined situations weighted by observed frequencies, is blind to scenarios that have never been encountered or conceived. This limitation is shared by all scenario-based approaches but remains important: the CUBICS guarantee is only as complete as the underlying ODD partition.

## VII. RELATED WORK

## A. Continuous safety assurance and runtime monitoring

Continuous safety assurance [20] aims to keep a system’s safety argument valid throughout its lifecycle by continuously monitoring operational behaviour, detecting assurance deficits, and updating the safety case as the system or environment changes [5]. Runtime monitoring and runtime assurance architectures operationalise this by observing system and environment variables, checking conformance to safety constraints, and triggering mitigation or enforcement actions when constraints might be violated [3], [12]. While such approaches can detect violations quickly and support dynamic safety cases, they typically provide qualitative signals (“constraint violated/not violated”) rather than quantified confidence in failure probabilities, and they usually lack mechanisms to (a) incorporate context distributions and (b) propagate monitoring evidence compositionally through modular arguments. Moreover, membership in scenarios or contexts (rain, night, etc.) may itself be uncertain, so updates must reflect contextual ambiguity and convert runtime evidence into quantitative, uncertainty-aware guarantees that evolve over time.

## B. Assurance confidence and reliability modelling

Quantitative assurance confidence estimation aims to turn heterogeneous evidence (e.g., tests, field data, near misses) into explicit confidence levels in safety claims. In autonomous driving, naïve mileage-based demonstrations for rare catastrophic events are impractical in terms of miles required to obtain meaningful statistical bounds [14], which motivates statistical and especially Bayesian approaches that explicitly model uncertainty and allow incremental updating of confidence as evidence accumulates. Standards such as ISO 26262 recognise proven-in-use style arguments but do not prescribe how to construct them for ML components whose performance is context-dependent and evolves over time. In this setting, quantitative assurance methods must cope with rare events, nonstationary failure rates, and evolving uncertainties.

Bayesian inference provides a probabilistic framework for updating hypotheses based on evidence. In safety contexts, it has been used to model component or system failure probabilities and refine them as test or field data accumulate [7], [4], [11]. For perception systems and autonomous vehicles, Bayesian reliability assessments explicitly consider that failure rates may depend on environmental conditions, moving beyond a single global Bernoulli parameter [1], [19]. This is crucial in the rare-event regime, where context-sensitive models are needed to make efficient use of limited evidence and pure mileage-based demonstration is infeasible. Bayesian approaches also underpin assurance-oriented test planning, where sample sizes are chosen to achieve specified posterior confidence targets [26]. However, prior probabilities can be subjective, evidence may be non-representative or correlated, and building and maintaining a single integrated probabilistic model for an entire system is often impractical and fragile, especially for ML-heavy systems. These limitations motivate more modular and context-structured approaches.

Subjective Logic (SL) [13] is an evidential reasoning framework that extends classical probability theory with explicit representation of uncertainty and base rates, and provides operators to combine and transform probabilistic opinions. It offers a convenient bridge between statistical models (e.g., Beta/Dirichlet distributions) and argumentation-level reasoning about confidence and defeaters. In assurance confidence estimation, SL has been used to reinterpret Baconian-style confidence as Beta distributions and visualise it in an opinion triangle [6], to make formal inferences within assurance arguments [29], to model relationships and dependencies between evidential artefacts [8], [2], to model the behaviour of defeaters [9], and to formalise safety contracts and derive argument structures [10]. Building on this work, CUBICS can be seen as a concrete instantiation of the SL contract formalism for ML components: a single, context-agnostic binary contract is refined into a set of situation-indexed guarantees over context-dependent failure behaviour and equipped with an explicit Bayesian update scheme driven by field and test data. This positions SL as a suitable intermediate calculus between Bayesian statistical models and safety-case style reasoning.

In the context of continuous safety assurance, these properties are particularly valuable. SL enables assurance confidence to be updated as operational data and concerns evolve, without requiring a single monolithic system-level probabilistic model. Statistical evidence (e.g., Beta–binomial updates over pass/fail data) can be mapped into SL opinions, modified by defeaters or external information, and mapped back to probabilistic representations that feed into modular safety contracts. This interplay between Bayesian statistics and SL underpins CU-BICS: situation-specific probabilistic guarantees and SL-based update rules yield quantitative, uncertainty-aware assurance confidence in a modular, context-aware manner.

Prior work already showed that reliability claims should be weighted by how systems are actually used. Musa defines reliability assessment around a quantitative characterisation of expected use, while later Bayesian approaches model reliability over input/context-space partitions and explicitly treat uncertainty and drift in the operational profile [17], [18]. In parallel, covariate-dependent reliability models in classical reliability engineering use environmental factors, degradation trends, or time-varying stresses to estimate failure risk as a function of operating conditions rather than a single global rate [21], [30]. At the probabilistic level, CUBICS instantiates these ideas as a context-stratified reliability model over situation-specific failure probabilities; the next section connects this to ODD-/scenario-based safety analysis for autonomous systems.

## C. Context-aware and scenario-based safety analysis

Context-aware safety analysis treats system behaviour as explicitly dependent on operating conditions represented by an ODD specification. In automated driving, the ODD is the set of operating conditions under which a driving automation system is intended to function. Scenario-based safety evaluation builds on this by structuring verification and validation around scenarios derived from or constraining the ODD. The PEGASUS methodology [27] promotes systematic scenario catalogues and links scenario-based testing to safety argumentation. Complementary work on logical scenarios formalises abstraction levels (functional, logical, concrete) and parameterisation for safety validation within the ODD [24].

In the SOTIF context, scenario-based approaches are used to identify triggering conditions for functional insufficiencies and to construct accelerated tests that expose SOTIF-related hazards. For autonomous systems, scenario- and ODD-based safety research argues that pure mileage is infeasible, and that safety evidence should be organised around scenario coverage, environmental conditions, and risk exposure frequencies [15], [25], [22]. Recent reviews [31] distinguish between ordinary and safety-critical scenarios, discuss bidirectional interaction modelling and critical scenario generation (including corner cases), and emphasise the need to quantify exposure frequencies and residual risks per scenario. A survey on risk assessment for autonomous driving by Lu et al. [16] shows that many methods ultimately evaluate risk at the scenario level, but still struggle with (i) uncertain or evolving operational profiles, (ii) long-tail and unknown scenarios, and (iii) aggregating heterogeneous scenario evidence into modular, uncertaintyaware guarantees suitable for safety cases.

CUBICS directly addresses these gaps by (i) decomposing the ODD into situations that play the role of logical scenarios, (ii) associating each situation with a continuously updated Bayesian guarantee over safety-relevant behaviour, and (iii) maintaining explicit beliefs over situation occurrence. This turns the scenario/ODD partition into a set of probabilistic “cells” whose contributions to component-level risk can be aggregated using contracts, while preserving uncertainty and supporting through-life updates from operational evidence.

## VIII. CONCLUSIONS AND FUTURE WORK

This paper introduced CUBICS, a methodology for continuous, situation-aware performance estimation of safetyrelevant ML components. Building on earlier work on SLbased safety contracts [10], we extend the focus to ML components whose behaviour depends on an explicit ODD. Assumptions are represented as SL opinions over context dimensions, guarantees as SL opinions over situation-conditional failure behaviour, and SL deduction is used to derive both per-situation guarantees and a marginal, context-weighted risk contribution for each component. In this work, SL is used to represent and update component-level statistical evidence in the opinion space.

At the probabilistic level, CUBICS instantiates a contextstratified hierarchical model over situation-specific failure probabilities. The contribution is not a novel theory, but the way this structure is embedded into situation-specific SL contracts is: (i) contracts are maintained per component and per situation rather than in a monolithic system model, (ii) runtime evidence is incorporated via a context-aware fractional update mechanism that respects uncertainty in ODD labelling, and (iii) the resulting per-situation and marginal opinions can be used as building blocks in wider safety assurance arguments. Our synthetic case study showed that, under its modelling assumptions, CUBICS recovers known situation-specific reliability patterns (RQ1). The YOLO/BDD100K study demonstrated that it exposes localised performance deficits and data gaps that a global Bernoulli model hides, and yields a more accurate component-level assessment via its marginal guarantee (RQ2). Sensitivity analyses indicated that results behave robustly under reasonable changes in priors, context misclassification, and data volume, while correctly reflecting increased uncertainty where evidence is scarce (RQ3).

Several limitations suggest directions for future work. First, the current instantiation focuses on binomial failure modes for a single component; we plan to extend CUBICS to continuousvalued metrics and to small multi-component chains to demonstrate contract composition in practice. Second, the combinatorial growth of situations and the assumption of stationarity within each cell call for hierarchical or factored situation models and temporal discounting schemes compatible with SL. Finally, we have only partially exploited the resilience branch of the contract structure in [10]; integrating explicit resilience goals and runtime monitors is an important step towards endto-end, through-life safety assurance.

## ACKNOWLEDGMENT

The research leading to these results is funded by the German Federal Ministry for Economic Affairs and Energy within the project “Safe AI Engineering – Sicherheitsargumentation befähigendes AI Engineering über den gesamten Lebenszyklus einer KI-Funktion”. The authors would like to thank the consortium for the successful cooperation. We also thank the anonymous reviewers for their helpful feedback.

## REFERENCES

[1] M. Berk, H.-M. Kroll, O. Schubert, B. Buschardt, and D. Straub. Bayesian test design for reliability assessments of safety-relevant environment sensors considering dependent failures. In WCX™ 17: SAE World Congress Experience. SAE Technical Paper, 2017.

[2] S. Burton, B. Herd, and J. Zacchi. Uncertainty-aware evaluation of quantitative ML safety requirements. In SAFECOMP (Workshops), volume 14989 of Lecture Notes in Computer Science, pages 391–404. Springer, 2024.

[3] D. Cofer, I. Amundson, R. Sattigeri, A. Passi, C. Boggs, E. Smith, L. Gilham, T. Byun, and S. Rayadurgam. Run-time assurance for learning-enabled systems. In NASA Formal Methods Symposium, pages 361–368. Springer, 2020.

[4] E. Denney, G. Pai, and I. Habli. Towards measurement of confidence in safety cases. In ESEM, pages 380–383. IEEE Computer Society, 2011.

[5] E. Denney, G. Pai, and I. Habli. Dynamic safety cases for throughlife safety assurance. In 2015 IEEE/ACM 37th IEEE International Conference on Software Engineering, volume 2, pages 587–590. IEEE, 2015.

[6] L. Duan, S. Rayadurgam, M. P. E. Heimdahl, O. Sokolsky, and I. Lee. Representation of confidence in assurance cases using the beta distribution. In HASE, pages 86–93. IEEE Computer Society, 2016.

[7] B. Guo. Knowledge representation and uncertainty management: applying Bayesian Belief Networks to a safety assessment expert system. In International Conference on Natural Language Processing and Knowledge Engineering, pages 114–119, 2003.

[8] B. Herd and S. Burton. Can you trust your ML metrics? Using Subjective Logic to determine the true contribution of ML metrics for safety. In SAC, pages 1579–1586. ACM, 2024.

[9] B. Herd, J. Kelly, J.-V. Zacchi, C. Heinemann, and S. Diemert. Integrating defeaters into Subjective Logic-based quantitative assurance arguments. In 2025 20th European Dependable Computing Conference (EDCC), pages 141–149, 2025.

[10] B. Herd, J. Zacchi, and S. Burton. A deductive approach to safety assurance: Formalising safety contracts with Subjective Logic. In SAFECOMP (Workshops), volume 14989 of Lecture Notes in Computer Science, pages 213–226. Springer, 2024.

[11] C. Hobbs and M. Lloyd. The application of Bayesian Belief Networks to assurance case preparation. In SSS, pages 159–176. Springer, 2012.

[12] K. L. Hobbs, M. L. Mote, M. C. Abate, S. D. Coogan, and E. M. Feron. Runtime assurance for safety-critical systems: An introduction to safety filtering approaches for complex control systems. IEEE Control Systems Magazine, 43(2):28–65, 2023.

[13] A. Jøsang. Subjective logic, volume 3. Springer, 2016.

[14] N. Kalra and S. M. Paddock. Driving to safety: How many miles of driving would it take to demonstrate autonomous vehicle reliability? Transportation research part A: policy and practice, 94:182–193, 2016.

[15] C. W. Lee, N. Nayeer, D. E. Garcia, A. Agrawal, and B. Liu. Identifying the operational design domain for an automated driving system through assessed risk. In 2020 IEEE Intelligent Vehicles Symposium (IV), pages 1317–1322. IEEE, 2020.

[16] D. Lu, H. Du, Z. Wu, and S. Yang. Risk assessment in autonomous driving: a comprehensive survey of risk sources, methodologies, and system architectures. Autonomous Intelligent Systems, 5(1):24, 2025.

[17] J. D. Musa. Operational profiles in software-reliability engineering. IEEE software, 10(2):14–32, 2002.

[18] R. Pietrantuono, P. Popov, and S. Russo. Reliability assessment of service-based software under operational profile uncertainty. Reliability Engineering & System Safety, 204:107193, 2020.

[19] P. Popov. Dynamic safety assessment of autonomous vehicle based on multivariate Bayesian inference (DyAVSA). Journal of Reliable Intelligent Environments, 11(3):14, 2025.

[20] P. Schleiss, F. Carella, and I. Kurzidem. Towards continuous safety assurance for autonomous systems. In 2022 6th International Conference on System Reliability and Safety (ICSRS), pages 457–462. IEEE, 2022.

[21] H.-J. Shyur, E. Elsayed, and J. T. Luxhøj. A general model for accelerated life testing with time-dependent covariates. Naval Research Logistics (NRL), 46(3):303–321, 1999.

[22] L. Tang, R. Wang, Z. Liu, Y. Liang, Y. Niu, W. Zhu, and Z. Duan. Scenario-based accelerated testing for sotif in autonomous driving: a review. IEEE Internet of Things Journal, 12(2):1453–1470, 2024.

[23] Y. Tian, Q. Ye, and D. Doermann. Yolov12: Attention-centric real-time object detectors. arXiv preprint arXiv:2502.12524, 2025.

[24] H. Weber, J. Bock, J. Klimke, C. Roesener, J. Hiller, R. Krajewski, A. Zlocki, and L. Eckstein. A framework for definition of logical scenarios for safety assurance of automated driving. Traffic injury prevention, 20(sup1):S65–S70, 2019.

[25] P. Weissensteiner, G. Stettinger, S. Khastgir, and D. Watzenig. Operational design domain-driven coverage for the safety argumentation of automated vehicles. IEEE Access, 11:12263–12284, 2023.

[26] K. J. Wilson and M. Farrow. Assurance for sample size determination in reliability demonstration testing. Technometrics, 63(4):523–535, 2021.

[27] H. Winner, K. Lemmer, T. Form, and J. Mazzega. PEGASUS – first steps for the safe introduction of automated driving. In Road Vehicle Automation 5, pages 185–195. Springer, 2018.

[28] F. Yu, W. Xian, Y. Chen, F. Liu, M. Liao, V. Madhavan, and T. Darrell. BDD100K: A diverse driving video database with scalable annotation tooling. CoRR, abs/1805.04687, 2018.

[29] C. Yuan, J. Wu, C. Liu, and H. Yang. A subjective logic-based approach for assessing confidence in assurance case. International Journal of Performability Engineering, 13(6):807, 2017.

[30] H. Zheng, X. Kong, H. Xu, and J. Yang. Reliability analysis of products based on proportional hazard model with degradation trend and environmental factor. Reliability Engineering & System Safety, 216:107964, 2021.

[31] Z. Zhong, Y. Tang, Y. Zhou, V. d. O. Neves, Y. Liu, and B. Ray. A survey on scenario-based testing for automated driving systems in high-fidelity simulation. arXiv preprint arXiv:2112.00964, 2021.
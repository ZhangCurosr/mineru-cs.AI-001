# Threat-guided Policy-aware Scene Perturbation for Safe Autonomous Driving with Online Reinforcement Learning

Xincong Hu<sup>1∗</sup>, Lei Ou<sup>1∗</sup>, Maosen Li<sup>2</sup>, Jingtao Zhang<sup>2</sup>, Liguo Hou<sup>2</sup>, Zongzhang Zhang<sup>1†</sup>

<sup>1</sup>Nanjing University, Nanjing, Jiangsu, China <sup>2</sup>Yinwang Intelligent Technology Co., Ltd., China

## Abstract

Reinforcement learning (RL) has shown promising performance in autonomous driving, yet ensuring the safety of online RL policies remains challenging due to insuficient exposure to safety-critical driving scenes. The long-tailed nature of real-world trafic situations makes dangerous and rare interactions dificult to encounter through conventional sampling, limiting the ability of RL policies to learn robust safety behaviors. Existing methods improve training diversity by synthesizing challenging scenes or adversarial situations. However, these approaches typically optimize scene generation objectives separately from the evolving policy, without explicitly modeling how generated perturbations relate to the current policy’s weaknesses and learning needs. In this paper, we propose Threat-guided Policy-aware Scene Perturbation (TPSP) for safe autonomous driving with online RL. TPSP introduces a policy-aware scene encoder to capture the interaction between policy behaviors and surrounding environments, enabling scene perturbation aligned with the current policy. Based on this representation, TPSP selectively perturbs critical objects rather than applying uniform modifications across the scene. Furthermore, we develop a threat-guided optimization strategy that evaluates perturbed scenes through threat-level diferences between policy rollouts on original and perturbed scenes, guiding the generation of safety-critical scenes with higher training value. Comprehensive experiments demonstrate that TPSP improves safety learning efficiency, achieving strong safety performance on NAVSIM v2 with approximately 4 million kilometers of simulated driving data. Ablation studies verify that policy-aware targeted perturbations provide more informative safety-critical experiences than random or policy-unaware strategies, enabling safer driving under limited interaction budgets.

## Introduction

Reinforcement learning (RL) has emerged as a promising approach for autonomous driving by enabling policies to learn complex decision-making strategies through environment interaction (Kiran et al. 2022; Cusumano-Towner et al. 2025; Kazemkhani et al. 2025). Unlike rule-based systems relying on handcrafted objectives (Paden et al. 2016) and supervised approaches requiring expert demonstrations (Codevilla et al.

2018; Bansal, Krizhevsky, and Ogale 2019), RL can adapt driving behaviors through large-scale experience collection. However, achieving safe and robust driving remains challenging due to the scarcity of safety-critical long-tail scenes, where rare events such as collisions provide insuficient training signals for learning reliable safety behaviors.

To improve the exposure of RL policies to challenging driving scenes, prior studies explore scene augmentation and perturbation strategies to enrich training experiences. Representative approaches, including prioritized replay (Schaul et al. 2016), curriculum learning (Bengio et al. 2009), and adversarial scene generation (Xu et al. 2022), focus on selecting informative samples or constructing failure-inducing scenes. However, existing methods typically characterize scene difficulty based on environment-level objectives, such as collision probability, adversarial objectives, or predefined safety violations (Ding et al. 2020; Wang et al. 2021; Xu et al. 2022; Chen et al. 2025). Consequently, the generated scenes are often independent of the evolving policy: a specific event, such as a cut-in or sudden braking, may be trivial for a mature policy but challenging for an immature one. This limitation prevents existing dificulty-oriented strategies from efectively targeting the current policy weaknesses during online RL training.

Motivated by this observation, we propose that an efective scene perturbation should consider the current policy behavior instead of relying on predefined scene modifications. Since diferent policies may fail under diferent trafic conditions, policy-agnostic perturbations often provide limited training benefits. Therefore, a more efective perturbation strategy should adapt to current policy characteristics and identify the safety-critical situations that are most relevant to its limitations. Based on this insight, we propose TPSP, a Threat-guided Policy-aware Scene Perturbation framework for safe autonomous driving with online RL. TPSP identifies and perturbs policy-relevant challenging scenes, enabling more efective safety-oriented policy optimization.

As illustrated in Figure 1, TPSP consists of three key components: a policy-aware scene encoder, a targeted scene perturbation module, and a threat-guided scene perturbation optimization method. First, the policy-aware scene encoder integrates ego states, surrounding objects, road context, scene risk information, and the current policy’s action distribution to construct a policy-aware scene representation. This representation provides both scene-level and policy-level information for perturbation generation. With this representation, targeted scene perturbation selectively modifies critical objects instead of applying uniform perturbations across the scene. Finally, TPSP evaluates perturbed scenes by computing the threat-level diferences between the policy rollouts on perturbed and original scenes. The resulting threat signal optimizes the scene perturbation network, guiding it toward discovering safety-critical scenes with higher training value for autonomous driving policy improvement.

![](images/3c4bef70234af3b5186696194a22f4f4f5f87940ce965d4eb4e4b98fed45caad.jpg)  
Figure 1: Overview of the proposed framework.

Extensive experiments on autonomous driving benchmarks demonstrate that TPSP improves safety learning efficiency under limited online RL interaction budgets. With roughly 4 million kilometers of simulated driving data, TPSP efectively discovers policy-relevant safety-critical experiences and enhances policy robustness. On the NAVSIM v2 navhard\_two\_stage benchmark (Cao et al. 2025), TPSP achieves 99.8% NC and 99.6% TTC on Stage 1 and maintains strong performance in the more challenging Stage 2 setting with 96.7% NC and 94.6% TTC, outperforming previous best results by 2.2% and 1.8%, respectively. These results demonstrate that policy-aware exploration enables more eficient utilization of limited interaction data for learning safer driving behaviors. In this work, our contributions are summarized as follows:

• We propose TPSP, a novel threat-guided perturbation framework designed to generalize safety-critical scenes for safe autonomous driving. By targeting the current policy, TPSP generates high-risk interaction cases to robustly train safe driving behaviors.

• Within TPSP, we quantify the threat by comparing preand post-perturbation rollouts to identify high-risk samples. The scene perturbation module is then trained with the objective of maximizing these threat scores.

• Extensive experiments demonstrate that TPSP significantly outperforms baseline methods in both driving safety and data eficiency, highlighting its efectiveness in training robust driving policies.

## Related Work

Reinforcement learning for autonomous driving. Reinforcement learning (RL) has been widely studied for closed-loop autonomous driving, where policies are optimized through continuous interaction with dynamic environments. To support RL-based training, various simulation platforms have been developed to improve scene diversity, scalability, and realism. CARLA (Dosovitskiy et al. 2017) and SMARTS (Zhou et al. 2021) provide high-fidelity simulation and multi-agent interaction, while MetaDrive (Li et al. 2023) and GPUDrive (Kazemkhani et al. 2025) further enhance scene generation and large-scale parallel experience collection. Recent works also explore more realistic and effective RL training paradigms, such as 3DGS-based simulation in RAD (Gao et al. 2025) and aligned world models in Raw2Drive (Yang et al. 2025) for driving policy optimization. However, RL policies remain highly dependent on the quality and diversity of collected experiences, making rare and safety-critical driving scenes dificult to discover through conventional sampling.

Safety-critical scene generation and perturbation. Existing works improve the exposure of driving policies to challenging situations through safety-critical scene generation. AdvSim (Wang et al. 2021) and KING (Hanselmann et al. 2022) optimize surrounding-agent behaviors to construct safety-critical interactions, while SafeBench (Xu et al. 2022) and CAT (Zhang et al. 2023) provide benchmarking and critical scene generation frameworks for robustness evaluation. Recent approaches further explore learning-based scene construction, including feasibility-aware adversarial optimization (Chen et al. 2025), difusion-based safetycritical synthesis (Xu et al. 2025), and collaborative adversarial scene evolution (Liu et al. 2026). However, these methods mainly optimize scene dificulty, failure likelihood, or adversarial object behaviors from environment-level objectives, without explicitly considering the evolving policy and its specific learning needs. TPSP addresses this limitation by introducing policy-aware representations for targeted scene perturbation and optimizing perturbations with policy-

relevant threat signals.

Policy-aware reinforcement learning. Policy-aware learning improves training eficiency by adapting experiences or interactions according to the current policy. Self-play methods such as AlphaZero (Silver et al. 2018) demonstrate the efectiveness of informative interaction generation, while several autonomous driving studies explore adaptive agent interactions and robust policy optimization (Dai et al. 2023; Cao et al. 2023; Cusumano-Towner et al. 2025). Recent RL driving frameworks such as CaRL (Jaeger et al. 2025) and adaptive curriculum learning (Abouelazm et al. 2025) further adjust training experiences based on policy capability. However, these methods focus mainly on policy optimization or experience scheduling, rather than constructing safetycritical scenes tailored to the weaknesses of current policy. In contrast, TPSP establishes a closed-loop policy-aware perturbation process, where the policy guides scene perturbation to provide more valuable experiences for online RL training.

## Problem Formulation

We formulate closed-loop autonomous driving as a Markov decision process $( \mathrm { M D P } ) , \mathcal { M } = ( \mathcal { S } , \mathcal { A } , \mathcal { P } , \mathcal { R } , \gamma )$ , where $s , A ,$ ${ \mathcal { P } } , { \mathcal { R } } ,$ and $\gamma$ denote the state space, action space, transition function, reward function, and discount factor, respectively. $\mathbf { A t }$ timestep $t ,$ the driving policy $\pi _ { \boldsymbol { \theta } } \big ( a _ { t } | \boldsymbol { s } _ { t } \big )$ maps the current state $s _ { t } \in S$ to an action $a _ { t } \in \mathcal A$ . The state contains observable information of the ego vehicle, surrounding objects, and driving environment.

A driving policy rollout is represented as

$$
\tau = ( s _ { 0 } , a _ { 0 } , r _ { 0 } , \ldots , s _ { T } ) ,\tag{1}
$$

where $\boldsymbol { r } _ { t } \in \mathcal { R }$ is the reward received at timestep t and $T$ represents the rollout horizon.

To generate policy-relevant safety-critical scenes, we introduce a scene perturbation network $G _ { \phi }$ parameterized by ϕ. Given a policy-aware scene representation, $G _ { \phi }$ predicts perturbations including object speed adjustment, object spawn time shift, and ego initial state modification.

The objective of TPSP is to optimize $G _ { \phi }$ to discover safetycritical scenes with higher training value for policy optimization. Specifically, TPSP updates $G _ { \phi }$ using the threat-level diference between policy rollouts on perturbed and original scenes as the optimization signal.

## Method

The proposed TPSP framework consists of a driving policy $\pi _ { \theta }$ and a scene perturbation network $G _ { \phi } .$ . Given policy-aware scene representations, $G _ { \phi }$ generates targeted scene perturbations, which are used to collect rollouts for policy optimization. The perturbations are optimized through policy-relevant threat signals derived from the comparison between policy rollouts on perturbed and original scenes, forming a closedloop process that enables eficient learning of robust safety behaviors.

## Policy-aware Scene Encoder

The efectiveness of scene perturbation depends not only on the trafic situations, but also on how the current policy interacts with the scene. Therefore, TPSP learns a policy-aware scene representation that integrates trafic configuration information and policy characteristics, enabling the scene perturbation network to generate perturbations aligned with the current policy. Given a scene state $s _ { t } ,$ TPSP extracts heterogeneous features from the scene, including ego feature, surrounding object features, road feature, risk-related feature and policy feature conditioned on $s _ { t }$

The ego feature $e ^ { \mathrm { e g o } }$ encodes the current kinematic state of the ego vehicle, including velocity, heading, and destination information. For the i-th surrounding object, we define its feature representation as $e _ { i } ^ { \mathrm { o b j } } , i = 1 , \dots , N$ , where $N$ denotes the number of observable objects. Each object feature contains intrinsic attributes, such as object type and geometric properties, as well as interaction-related information with the ego vehicle, including relative position, relative velocity, and Time-to-Collision (TTC). These interaction features characterize the dynamic relationship between the ego vehicle and surrounding objects.

Moreover, the road feature $e ^ { \mathrm { r o a d } }$ encodes the information of the surrounding road structure, providing static environmental context for scene representation. In addition, TPSP incorporates a scene-level risk-related feature $e ^ { \mathrm { r i s k } }$ to provide complementary safety information. This feature aggregates several safety-related indicators, including the inverse minimum distance to surrounding objects, the inverse minimum $\mathrm { T T C } ,$ and the number of surrounding objects in the scene.

To capture the characteristics of the current driving policy, TPSP extracts a policy feature from the hidden representation of the frozen policy network. Specifically, let $\pi _ { \bar { \theta } }$ denote a frozen copy of the current driving policy during rollout collection, where $\bar { \theta }$ represents the fixed policy parameters. Given the original scene observation before perturbation, the hidden representation extracted from $\pi _ { \bar { \theta } }$ is denoted as $\mathbf { z } _ { \pi _ { \bar { \theta } } }$ The policy feature is obtained through a projection network:

$$
e ^ { \mathrm { p o l i c y } } = \mathrm { M L P ^ { \mathrm { p o l i c y } } } \left( { \bf z } _ { \pi _ { \bar { \theta } } } \right) ,\tag{2}
$$

where $\mathrm { M L P ^ { p o l i c y } ( \cdot ) }$ denotes the policy projection network that maps the hidden policy representation into the extracted policy feature $e ^ { \mathrm { p o l i c y } }$ . The fixed policy parameters <sup>¯</sup>θ prevent gradients from propagating into the driving policy during scene perturbation network optimization. This design allows the scene perturbation network to leverage the policy representation as auxiliary information while keeping policy optimization decoupled.

These extracted features are then fed into the policy-aware scene encoder: $( e ^ { \mathrm { e g o } } , e _ { i } ^ { \mathrm { o b j } } , e ^ { \mathrm { r o a d } } , e ^ { \mathrm { r i s k } } , e ^ { \mathrm { p o l i c y } } )$ . To model the interaction between the ego vehicle and surrounding objects, TPSP employs an attention module to aggregate object information conditioned on the ego vehicle:

$$
c ^ { \mathrm { o b j } } = \mathrm { A t t e n t i o n } \left( e ^ { \mathrm { e g o } } , \{ e _ { i } ^ { \mathrm { o b j } } \} _ { i = 1 } ^ { N } \right) ,\tag{3}
$$

with $\mathbf { c } ^ { \mathrm { { o b j } } }$ denoting the aggregated object interaction features and invalid objects masked before attention normalization. The ego feature serves as the query, while surrounding object features provide the interaction information to be aggregated.

Finally, the complete policy-aware scene representation is obtained by fusing all feature components:

$$
\begin{array} { r } { \boldsymbol { h } = \mathrm { M L P } ^ { \mathrm { f u s i o n } } \left( \left[ e ^ { \mathrm { e g o } } , c ^ { \mathrm { o b j } } , e ^ { \mathrm { r o a d } } , e ^ { \mathrm { r i s k } } , e ^ { \mathrm { p o l i c y } } \right] \right) , } \end{array}\tag{4}
$$

where $\mathrm { { M L P } ^ { f u s i o n } ( \cdot ) }$ represents the feature fusion network that combines heterogeneous scene and policy feature into the final policy-aware scene representation h. Thus, h captures both the physical interaction context of the scene and the characteristics of the current driving policy. This representation is then used by the subsequent scene perturbation network to generate policy-aware scene perturbations.

## Targeted Scene Perturbation

Given the policy-aware scene representation h, TPSP generates targeted scene perturbations by first selecting critical objects and then modifying the corresponding scene elements. Instead of applying uniform perturbations to all objects, TPSP focuses on objects that are more relevant to the current policy behavior.

To identify important perturbation targets, TPSP assigns each observable object an importance score. This is achieved by applying a Softmax function over all objects in the scene, which normalizes the raw scoring logits into a probability distribution over the object set. Formally, the importance weight for each object is computed as:

$$
\pmb { w } ^ { \mathrm { o b j } } = \mathrm { S o f t m a x } \left( \left\{ f ^ { \mathrm { s c o r e } } ( [ e _ { i } ^ { \mathrm { o b j } } , \pmb { h } ] ) \right\} _ { i = 1 } ^ { N } \right) ,\tag{5}
$$

where $w ^ { \mathrm { o b j } }$ denotes the normalized importance scores of observable objects, N is the number of objects in the scene, and $f ^ { \mathrm { s c o r e } } ( \cdot )$ is an object scoring network that maps each object-scene representation into an importance logit. Based on these scores, TPSP selects the top- $\bar { . K }$ objects as perturbation targets, whose representations are denoted as:

$$
{ \cal E } _ { K } ^ { \mathrm { o b j } } = \{ e _ { 1 } ^ { \mathrm { o b j } } , e _ { 2 } ^ { \mathrm { o b j } } , . . . , e _ { K } ^ { \mathrm { o b j } } \} .\tag{6}
$$

The selected top- $K$ object representations are then combined with the policy-aware scene representation h and provided to the scene perturbation network $G _ { \phi }$ . Instead of directly generating deterministic modifications, $G _ { \phi }$ models a Gaussian perturbation distribution and samples raw perturbation variables from this distribution.

Specifically, the perturbation network predicts the mean and standard deviation of the Gaussian distribution:

$$
\pmb { \mu } = g _ { \mu } ( [ \pmb { E } _ { K } ^ { \mathrm { o b j } } , \pmb { h } ] ) ,\tag{7}
$$

$$
\pmb { \sigma } = \exp \left( g _ { \sigma } ( [ \pmb { E } _ { K } ^ { \mathrm { o b j } } , \pmb { h } ] ) \right) ,\tag{8}
$$

$$
z ^ { \mathrm { r a w } } \sim \mathcal { N } \left( \pmb { \mu } , \mathrm { d i a g } ( \pmb { \sigma } ^ { 2 } ) \right) ,\tag{9}
$$

where $g _ { \mu } ( \cdot )$ and $g _ { \sigma } ( \cdot )$ denote the mean and standard deviation prediction heads of $G _ { \phi }$ , respectively, with exponential function ensuring all standard deviation values positive. $\pmb { \mu }$ and σ represent the predicted mean and standard deviation of the perturbation distribution, and $z ^ { \mathrm { r a w } }$ denotes the sampled unconstrained perturbation variables. The diagonal covariance matrix $\dot { \mathrm { d i a g } } ( \sigma ^ { 2 } )$ assumes independent sampling across perturbation dimensions.

TPSP then converts the raw perturbation variables into boundary indicators of the final state perturbation,

$$
z = \lambda \left( z ^ { \operatorname* { m a x } } \odot \operatorname { t a n h } ( z ^ { \operatorname { r a w } } ) \right) ,\tag{10}
$$

where $z ^ { \mathrm { m a x } }$ specifies the maximum magnitude of each per turbation variable, and ⊙ denotes element-wise multiplication. λ is a dynamically adjusted scaling factor controlling the overall perturbation strength during training. This transformation converts unconstrained Gaussian samples into physically feasible perturbations while preserving diferentiability.

The final perturbation variables z correspond to three types of scene modifications in TPSP: object speed adjustment, object spawn time shift, and ego initial state modification. Object-related perturbations are generated mainly based on the selected object representations, while ego initial state modification is conditioned on the policy-aware scene representation h, which captures the current policy’s response to the scene, including predicted trajectory distribution or action-value estimates. By integrating object-level and egolevel adjustments in this manner, TPSP generates targeted perturbations rather than fixed or uniform modifications.

## Threat-guided Scene Perturbation Optimization

At the beginning of each training iteration, TPSP enriches the training scene set with informative safety-critical experiences to improve policy robustness. Since the original scene dataset D may contain insuficient high-risk interactions, TPSP maintains a scene bufer B to store previously discovered high-threat perturbed scenes. The original and perturbed scenes are combined to construct the training dataset $\mathcal { D } ^ { \mathrm { t r a i n } }$ , where their sampling proportions are dynamically adjusted throughout training. Specifically, the proportion of high-threat perturbed scenes is gradually increased during training, allowing the policy to progressively adapt to more challenging safety-critical situations.

Given any scene sampled from D or $B ,$ the policy model performs rollouts based on the current state. Each rollout is subsequently evaluated from multiple perspectives regarding environmental trafic threats. To evaluate the threat score of a rollout τ, TPSP first computes a step-level threat score:

$$
\begin{array} { r l } & { c _ { t } = w ^ { \mathrm { t t c } } c _ { t } ^ { \mathrm { t t c } } + w ^ { \mathrm { d i s t } } c _ { t } ^ { \mathrm { d i s t } } + w ^ { \mathrm { e d g e } } c _ { t } ^ { \mathrm { e d g e } } } \\ & { ~ + w ^ { \mathrm { c o l } } c _ { t } ^ { \mathrm { c o l } } + w ^ { \mathrm { o f f } } c _ { t } ^ { \mathrm { o f f } } , } \end{array}\tag{11}
$$

where $c _ { t } ~ \in ~ [ 0 , 1 ]$ denotes the threat score at timestep t. The five components represent diferent safety-related factors, including time-to-collision, inter-object distance, roadedge proximity, collision occurrence, and of-road status. The coeficients $w ^ { \mathrm { { \bar { t t c } } } } , w ^ { \mathrm { { d i s t } } } , w ^ { \mathrm { { e d g e } } } , w ^ { \mathrm { { c o l } } }$ , and $w ^ { \mathrm { o f f } }$ are weighting factors that balance diferent threat components.

Since safety risks are usually concentrated in a small number of critical moments during a rollout, TPSP summarizes the overall rollout criticality by aggregating the most threatening timesteps. Given a rollout trajectory $\tau =$ $( s _ { 0 } , a _ { 0 } , r _ { 0 } , . . . , s _ { T } )$ with rollout horizon T, we select the top-k timesteps with the highest step-level threat scores and compute the rollout-level criticality as:

$$
\mathcal { C } ( \tau ) = \frac { 1 } { k ^ { \prime } } \sum _ { t \in \mathrm { T o p } ( \tau , k ) } c _ { t } , \quad k ^ { \prime } = \operatorname* { m i n } ( k , T + 1 ) ,\tag{12}
$$

where $\mathrm { T o p } ( \tau , k )$ retrieves the temporal indices of the top-k selected critical timesteps with the highest $c _ { t }$ values. Since a rollout of horizon T contains $T + 1$ states, $k ^ { \prime }$ adjusts the number of selected timesteps when the rollout length is shorter than k. By focusing on the most threatening moments rather than averaging all timesteps, $\mathcal { C } ( \tau )$ better captures the safetycritical characteristics of a rollout.

```latex
Algorithm 1 TPSP
1: Initialize driving policy $\pi _ { \theta } ,$ scene perturbation network
$G _ { \phi } ,$ original dataset D, scene bufer B and perturbation
scale λ
2: Set rollout policy $\pi _ { \bar { \theta } }  \pi _ { \theta }$
3: for each on-policy training iteration n do
4: Sample scenes from the combination of the original
scene dataset D and the perturbed scene bufer B.
5: Extract detached policy feature from the sampled
scenes using the rollout policy $\pi _ { \bar { \theta } }$
6: Construct policy-aware scene representations and
generate targeted scene perturbations with $G _ { \phi }$
7: Simulate perturbed scenes with $\pi _ { \bar { \theta } }$ and collect on
policy rollouts
8: Update π<sub>θ</sub> using the collected perturbed-scene roll
outs with the selected on-policy RL algorithm
9: if scene perturbation network update then
10: Simulate the same sampled original scenes without
perturbations using identical simulator seeds
11: Compute threat diferences using Eq. (15)
12: Update $G _ { \phi }$ by minimizing $\mathcal { L } ^ { \mathrm { p e r } }$ in Eq. (18)
13: end if
14: Update B with high-threat perturbed scenes
15: Adjust λ according to the perturbation schedule
16: Synchronize the rollout policy: $\pi _ { \bar { \theta } }  \pi _ { \theta }$
17: end for
```

In addition to the overall threat magnitude, TPSP considers the temporal instability of threat evolution during the rollout. While $\dot { \mathcal { C } } ( \tau )$ captures critical risk levels, it does not reflect how rapidly the threat changes over time. Therefore, TPSP introduces the threat instability metric:

$$
\mathcal { V } ( \tau ) = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } | c _ { t + 1 } - c _ { t } | ,\tag{13}
$$

where $\mathcal { V } ( \tau )$ measures the temporal instability of the threat evolution over the rollout. A larger value indicates that the safety condition changes more rapidly, suggesting a more challenging interaction for the driving policy.

Based on both threat severity and threat instability, the final threat score of a rollout is defined as:

$$
\xi ( \tau ) = \mathcal { C } ( \tau ) \frac { 1 + \beta \mathcal { V } ( \tau ) } { 1 + \beta } ,\tag{14}
$$

where $\beta$ controls the contribution of threat variation. The criticality term $\mathcal { C } ( \tau )$ dominates the threat estimation to prioritize genuinely safety-critical scenes, while $\mathcal { V } ( \tau )$ acts as a modulation factor to emphasize temporally unstable interactions. This formulation enables TPSP to identify informative safety-critical scenes for policy improvement.

To optimize the scene perturbation network, TPSP compares the threat scores between policy rollouts on perturbed and original scenes from the same initial current state. Specifically, both rollouts are generated using the same fixed driving policy $\pi _ { \bar { \theta } }$ and identical simulator seeds:

![](images/02b694bcdd4c3bf550b86b263584a215eef63a7835f00170fdc4bcf17ed8ca6c.jpg)  
Figure 2: Closed-loop optimization between the driving policy and scene perturbation network in TPSP.

$$
\Delta \xi = \xi ( \tau ^ { \mathrm { p e r } } ) - \xi ( \tau ^ { \mathrm { o r g } } ) ,\tag{15}
$$

where $\tau ^ { \mathrm { p e r } }$ and $\tau ^ { \mathrm { o r g } }$ denote the perturbed and original rollouts of the same initial state, respectively. The threat diference $\Delta \xi$ measures the additional safety dificulty introduced by the generated perturbation.

The perturbation network $G _ { \phi }$ is optimized using the threat diference as the learning signal. Based on one certain initial scene or state $s \in S .$ , for a sampled perturbation vector $z _ { s } ^ { \mathrm { r a w } }$ its log probability is defined as:

$$
\log p _ { \phi } ( z _ { s } ^ { \mathrm { r a w } } | E _ { K } ^ { \mathrm { o b j } } , h ) .\tag{16}
$$

The threat diferences are normalized within each perturbation update batch to reduce the scale variation across training iterations. The normalized threat advantage for scene s is computed as:

$$
\hat { A } _ { s } = \frac { \Delta \xi _ { s } - \mathrm { m e a n } ( \Delta \xi ) } { \mathrm { s t d } ( \Delta \xi ) + \epsilon } ,\tag{17}
$$

where $\hat { A } _ { s }$ denotes the normalized threat advantage of scene $s .$ The operators mean(·) and std(·) compute the mean and standard deviation of the threat diferences over the current perturbation update batch, respectively, and ϵ is a small constant for numerical stability. Thus, the scene perturbation network is optimized with:

$$
\begin{array} { r } { \mathcal { L } ^ { \mathrm { p e r } } = - \mathbb { E } _ { s } [ \hat { A } _ { s } \log p _ { \phi } ( z _ { s } ^ { \mathrm { r a w } } | \pmb { E } _ { K } ^ { \mathrm { o b j } } , \pmb { h } ) ] - c ^ { \mathrm { e n t } } H _ { z } , } \end{array}\tag{18}
$$

where $c ^ { \mathrm { e n t } }$ is the entropy regularization coeficient and $H _ { z }$ denotes the entropy of the perturbation distribution. The entropy regularization encourages exploration of the perturbation space and prevents premature convergence.

Algorithm 1 and Figure 2 illustrate the overall closedloop optimization procedure of TPSP. During training, the generated perturbed scenes are used to collect rollouts for updating the driving policy $\pi _ { \theta } .$ . Meanwhile, the threat difference between perturbed and original rollouts provides the optimization signal for updating the scene perturbation network $G _ { \phi } .$ . By iteratively improving the driving policy based on informative perturbed scenes with PPO (Schulman et al. 2017) algorithm and refining $G _ { \phi }$ with threat-guided signals, TPSP establishes a closed-loop optimization process for discovering safety-critical training scenes.

![](images/b9519495d67efc69891a71b9c9cc31551df7c40ed826d7f248bdc12f1b97834b.jpg)

![](images/a930aec6b589fd9d21a1aa79a352e8568ca771080d84f69ea04ae996d5af37f3.jpg)

![](images/d8be21d17e0ce8d390fa8a051261c340e2bb0040c9164085714ea4a379ca1166.jpg)  
Figure 3: Safety learning reward curves during online RL training. Here, collision reward penalizes vehicle collision events, while of-road reward penalizes curb collisions. TPSP achieves faster improvements in collision avoidance and of-road safety with fewer simulated driving mileage.  
Figure 4: Qualitative comparison on two representative safety-critical NAVSIM v2 scenes. TPSP enables earlier hazard response and safer trajectory generation compared with Vanilla PPO.

## Experiments

Our experiments assess TPSP’s safety and eficiency from three perspectives: learning speed, perturbation eficacy, and final performance. We first determine if TPSP improves safety behaviors faster under fixed interaction budgets. Next, ablation studies isolate the impact of targeted perturbations and policy-aware representations. We then compare TPSP’s final policy against baselines on the NAVSIM v2 benchmark. Qualitative results further illustrate that TPSP efectively optimizes scene perturbations using threat signals from policy rollouts to generate informative safety-critical interactions.

## Experimental Setup

We train TPSP using the GPUDrive simulator with PPO and the NAVSIM v2 navtrain dataset. The resulting driving policies are evaluated on the challenging NAVSIM v2 navhard\_two\_stage benchmark. The evaluation considers three safety-related metrics: no at-fault collisions (NC), drivable area compliance (DAC), and time to collision (TTC). During training, all methods share the same policy architecture, reward function, interaction budgets, and PPO optimization hyperparameters to ensure a fair comparison. Further implementation details are provided in Appendix A.

## Safety Learning Eficiency Analysis

Figure 3 illustrates the evolution of safety-related rewards during online RL training under the same interaction budgets. TPSP achieves faster improvements in both collision avoidance and of-road safety rewards compared with Vanilla PPO, indicating that the policy can acquire safety-related behaviors more eficiently with fewer simulated driving miles. For example, at 2M miles, TPSP already reaches a collisionreward level that Vanilla PPO does not attain until 4M miles. This improvement comes from the targeted exposure to informative safety-critical interactions, which provides more effective training signals than uniformly sampled experiences.

To further analyze the learned safety behaviors, Figure 4 presents two safety-critical scenarios from NAVSIM v2 to analyze learned safety behaviors. Under identical initial conditions, TPSP produces safer trajectories with earlier hazard responses. In Scene A, TPSP proactively avoids a collision by adjusting its trajectory, while Vanilla PPO fails to react and crashes. In Scene B, TPSP handles abrupt braking by decelerating and keeping a safe distance, whereas Vanilla PPO responds too late. These cases highlight TPSP’s ability to improve safety learning eficiency by focusing on challenging interactions relevant to the policy’s weaknesses.

Table 1: Ablation on NAVSIM v2 navhard\_two\_stage.
<table><tr><td rowspan="2">Method</td><td colspan="2">NC↑</td><td colspan="2">DAC ↑</td><td colspan="2">TTC ↑</td></tr><tr><td>S1</td><td>S2</td><td>S1</td><td>S2</td><td>S1</td><td>S2</td></tr><tr><td>Vanilla PPO</td><td>96.2</td><td>85.7</td><td>92.4</td><td>81.0</td><td>96.0</td><td>82.4</td></tr><tr><td>Random Perturbation</td><td>97.1</td><td>83.2</td><td>94.4</td><td>83.8</td><td>96.8</td><td>80.3</td></tr><tr><td>TPSP w/o PA</td><td>98.8</td><td>91.1</td><td>97.5</td><td>90.4</td><td>98.8</td><td>89.7</td></tr><tr><td>TPSP</td><td>99.8</td><td>96.7</td><td>97.8</td><td>93.7</td><td>99.6</td><td>94.6</td></tr></table>

## Ablation Study

We conduct ablation studies to analyze the contribution of diferent components in TPSP. We compare Vanilla PPO, random perturbation, TPSP without policy awareness (TPSP w/o PA), and the full TPSP framework with policy-aware targeted perturbation. Table 1 reports the safety performance on the NAVSIM v2 navhard\_two\_stage benchmark. Here, S1 and S2 denote Stage 1 and Stage 2 evaluations, respectively, where Stage 1 evaluates policies on original scenes and Stage 2 focuses on more challenging synthesized scenarios. The upward arrows indicate that higher values correspond to better performance for the corresponding safety metrics.

TPSP achieves the best performance across both stages, particularly in the more challenging S2 setting, where it obtains 96.7% NC and 94.6% TTC. Random perturbation provides only limited improvements over Vanilla PPO, indicating that increasing scene diversity alone is insuficient for efective safety learning. Moreover, the performance gap between TPSP and TPSP w/o PA demonstrates that incorporating policy-specific information is essential for generating more informative safety-critical scenes.

## Safety Evaluation on NAVSIM v2

We show the safety of TPSP by comparing it with state-ofthe-art methods on the NAVSIM v2 leaderboard (Table 2). TPSP achieves superior performance in NC and TTC metrics across both evaluation stages, highlighting its efectiveness in collision avoidance and interaction safety. Further details on the baseline methods are available in Appendix B.

Table 2: Safety on NAVSIM v2 navhard\_two\_stage. External results are reported from the NAVSIM v2 benchmark leaderboard for reference. (Cao et al. 2025)
<table><tr><td rowspan="2">Method</td><td colspan="2">NC↑</td><td colspan="2">DAC ↑</td><td colspan="2">TTC ↑</td></tr><tr><td>S1</td><td>S2</td><td>S1</td><td>S2</td><td>S1</td><td>S2</td></tr><tr><td>NavFormer</td><td>96.2</td><td>85.7</td><td>92.4</td><td>81.0</td><td>96.0</td><td>82.4</td></tr><tr><td>RAP</td><td>97.1</td><td>83.2</td><td>94.4</td><td>83.8</td><td>96.8</td><td>80.3</td></tr><tr><td>ZTRS</td><td>98.8</td><td>91.1</td><td>97.5</td><td>90.4</td><td>98.8</td><td>89.7</td></tr><tr><td>SimScale</td><td>99.5</td><td>94.5</td><td>99.1</td><td>94.2</td><td>99.5</td><td>92.8</td></tr><tr><td>DrivoR</td><td>99.1</td><td>92.3</td><td>98.2</td><td>91.6</td><td>98.6</td><td>90.5</td></tr><tr><td>TPSP</td><td>99.8</td><td>96.7</td><td>97.8</td><td>93.7</td><td>99.6</td><td>94.6</td></tr></table>

![](images/faa0dc0b3e6dde8ab7f1f3f3327a0037ec4e0454467cb8bee633dc8299084708.jpg)  
Figure 5: Visualization of threat-guided policy-aware perturbations learned by TPSP in GPUDrive simulation. Left: a cutin situation; Right: a car-following situation. TPSP evaluates how diferent perturbations afect policy-rollout threat signals and optimizes scene perturbations toward safety-critical interactions.

## Qualitative Analysis

Figure 5 visualizes the learned perturbations generated by TPSP in GPUDrive simulation. In the cut-in situation (left), the generated perturbation increases the threat of the interaction with $\Delta \dot { \xi } = + 0 . 1 1 1 5$ , corresponding to a more dangerous merging conflict. In contrast, the car-following situation (right) exhibits a less hazardous interaction pattern, and the estimated threat variation $\Delta \xi = - 0 . 0 5 4 0$ is consistent with this intuitive observation. These examples demonstrate that TPSP can efectively evaluate the risk variation induced by diferent perturbations and optimize scene modifications toward more informative safety-critical interactions, rather than blindly increasing scene complexity.

## Conclusion and Future Work

We proposed TPSP, a Threat-guided Policy-aware Scene Perturbation framework for improving safety learning eficiency in online RL for autonomous driving. By generating targeted perturbations guided by policy weaknesses, TPSP enables more informative exploration under limited training budgets. Experiments on NAVSIM v2 demonstrate that TPSP achieves strong safety performance with approximately 4 million kilometers of simulated driving mileage. Ablation studies further validate the efectiveness of policy-aware targeted perturbations. Future work will explore more comprehensive perturbation spaces beyond the current scene-level modifications, including richer semantic, map-level, and multi-agent interaction perturbations. We also plan to investigate TPSP in larger-scale simulation environments and real-world driving settings, as well as its integration with more diverse RL algorithms and autonomous driving policy architectures.

## References

Abouelazm, A.; et al. 2025. Automatic Curriculum Learning for Driving Scenarios: Towards Robust and Eficient Reinforcement Learning. In Proceedings of the 36th IEEE Intelligent Vehicles Symposium, 2333–2340.

Bansal, M.; Krizhevsky, A.; and Ogale, A. S. 2019. ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst. In Proceedings of the 15th Robotics: Science and Systems.

Bengio, Y.; Louradour, J.; Collobert, R.; and Weston, J. 2009. Curriculum Learning. In Proceedings of the 26th Annual International Conference on Machine Learning, 41–48.

Cao, W.; Hallgarten, M.; Li, T.; Dauner, D.; Gu, X.; Wang, C.; Miron, Y.; Aiello, M.; Li, H.; Gilitschenski, I.; Ivanovic, B.; Pavone, M.; Geiger, A.; and Chitta, K. 2025. Pseudo-Simulation for Autonomous Driving. In Proceedings of the 9th Conference on Robot Learning, 4709–4722.

Cao, Z.; Jiang, K.; Zhou, W.; Xu, S.; Peng, H.; and Yang, D. 2023. Continuous Improvement of Self-driving Cars Using Dynamic Confidence-aware Reinforcement Learning. Nature Machine Intelligence, 5(2): 145–158.

Chen, K.; Lei, Y.; Cheng, H.; Wu, H.; Sun, W.; and Zheng, S. 2025. FREA: Feasibility-Guided Generation of Safety-Critical Scenarios with Reasonable Adversariality. In Proceedings ofthe 8th Conference on Robot Learning, 566–586.

Codevilla, F.; Miura, J.; Lopez, A. M.; Koltun, V.; Dosovitskiy, A.; and Urtasun, R. 2018. End-to-end Driving via Conditional Imitation Learning. In Proceedings of the 35th IEEE International Conference on Robotics and Automation, 1–9.

Cusumano-Towner, M. F.; Hafner, D.; Hertzberg, A.; Huval, B.; Petrenko, A.; Vinitsky, E.; Wijmans, E.; Killian, T. W.; Bowers, S.; Sener, O.; Kraehenbuehl, P.; and Koltun, V. 2025. Robust Autonomy Emerges from Self-Play. In Proceedings ofthe 42nd International Conference on Machine Learning, 11710–11737.

Dai, Z.; Zhou, T.; Shao, K.; Mguni, D. H.; Wang, B.; and Hao, J. 2023. Socially-Attentive Policy Optimization in Multi-Agent Self-Driving System. In Proceedings of the 6th Conference on Robot Learning, 946–955.

Ding, W.; Chen, B.; Xu, M.; and Zhao, D. 2020. Learning to Collide: An Adaptive Safety-Critical Scenarios Generating Method. In Proceedings ofthe 33rd IEEE/RSJ International Conference on Intelligent Robots and Systems, 2243–2250.

Dosovitskiy, A.; Ros, G.; Codevilla, F.; Lopez, A.; and Koltun, V. 2017. CARLA: An Open Urban Driving Simulator. In Proceedings of the 1st Annual Conference on Robot Learning, 1–16.

Feng, L.; Gao, Y.; Zablocki, É.; Li, Q.; Li, W.; Liu, S.; Cord, M.; and Alahi, A. 2026. RAP: 3D Rasterization Augmented End-to-End Planning. In Proceedings of the 14th International Conference on Learning Representations.

Gao, H.; Chen, S.; Jiang, B.; Liao, B.; Shi, Y.; Guo, X.; Pu, Y.; Yin, H.; Li, X.; Zhang, X.; et al. 2025. Rad: Training an End-to-End Driving Policy via Large-Scale 3DGS-based Reinforcement Learning. arXiv preprint arXiv:2502.13144.

Hanselmann, N.; Renz, K.; Chitta, K.; Bhattacharyya, A.; and Geiger, A. 2022. KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients. In Proceedings of the 17th European Conference on Computer Vision, 335–352.

Jaeger, B.; Dauner, D.; Beißwenger, J.; Gerstenecker, S.; Chitta, K.; and Geiger, A. 2025. CaRL: Learning Scalable Planning Policies with Simple Rewards. arXiv preprint arXiv:2504.17838.

Kazemkhani, S.; Pandya, A.; Cornelisse, D.; Shacklett, B.; and Vinitsky, E. 2025. GPUDrive: Data-Driven, Multi-Agent Driving Simulation at 1 Million FPS. In Proceedings ofthe 13th International Conference on Learning Representations.

Kiran, B. R.; Sobh, I.; Talpaert, V.; Mannion, P.; Al Sallab, A. A.; Yogamani, S.; and Perez, P. 2022. Deep Reinforcement Learning for Autonomous Driving: A Survey. IEEE Transactions on Intelligent Transportation Systems, 23(6): 4909–4926.

Kirby, E.; Boulch, A.; Xu, Y.; Yin, Y.; Puy, G.; Zablocki, É.; Bursuc, A.; Gidaris, S.; Marlet, R.; Bartoccioni, F.; Cao, A.-Q.; Samet, N.; Vu, T.-H.; and Cord, M. 2026. Driving on Registers. arXiv preprint arXiv:2601.05083.

Li, Q.; Peng, Z.; Feng, L.; Zhang, Q.; Xue, Z.; and Zhou, B. 2023. MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3): 3461– 3475.

Li, Z.; Yao, W.; Wang, Z.; Sun, X.; Chen, J.; Chang, N.; Shen, M.; Song, J.; Wu, Z.; Lan, S.; and Alvarez, J. M. 2025. ZTRS: Zero-Imitation End-to-end Autonomous Driving with Trajectory Scoring. arXiv preprint arXiv:2510.24108.

Liu, J.; Guo, Y.; Zhong, F.; Zhang, T.; Jing, Z.; Liang, S.; Wang, J.; Zhang, M.; Liu, A.; and Liu, X. 2026. Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles. In Proceedings of the 40th AAAI Conference on Artificial Intelligence, 38926–38934.

Paden, B.; Cap, M.; Yong, S. Z.; Yershov, D.; and Frazzoli, E. 2016. A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles. IEEE Transactions on Intelligent Vehicles, 1(1): 33–55.

Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2016. Prioritized Experience Replay. In Proceedings of the 4th International Conference on Learning Representation.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.

Silver, D.; Hubert, T.; Schrittwieser, J.; Antonoglou, I.; Lai, M.; Guez, A.; Lanctot, M.; Sifre, L.; Kumaran, D.; Graepel, T.; et al. 2018. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. arXiv preprint arXiv:1712.01815.

Tian, H.; Li, T.; Liu, H.; Yang, J.; Qiu, Y.; Li, G.; Wang, J.; Gao, Y.; Zhang, Z.; Wang, L.; Ye, H.; Chen, L.; and Li, H. 2026. SimScale: Learning to Drive via Real-World Simulation at Scale. In Proceedings of the 43rd IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 36365–36374.

Wang, J.; Pun, A.; Tu, J.; Manivasagam, S.; Sadat, A.; Casas, S.; Ren, M.; and Urtasun, R. 2021. AdvSim: Generating Safety-Critical Scenarios for Self-Driving Vehicles. In Proceedings of the 38th IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9909–9918.

Xu, C.; Ding, W.; Lyu, W.; Liu, Z.; Wang, S.; He, Y.; Hu, H.; Zhao, D.; and Li, B. 2022. SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles. In Advances in Neural Information Processing Systems 35, 25667–25682.

Xu, C.; Petiushko, A.; Zhao, D.; and Li, B. 2025. Dif-Scene: Difusion-Based Safety-Critical Scenario Generation for Autonomous Vehicles. In Proceedings ofthe 35th AAAI Conference on Artificial Intelligence, 8797–8805.

Yang, Z.; Jia, X.; Li, Q.; Yang, X.; Yao, M.; and Yan, J. 2025. Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2). In Advances in Neural Information Processing Systems 38.

Zhang, L.; Peng, Z.; Li, Q.; and Zhou, B. 2023. CAT: Closed-Loop Adversarial Training for Safe End-to-End Driving. In Proceedings ofthe 7th Conference on Robot Learning, 2357– 2372.

Zhou, M.; Luo, J.; Villella, J.; Yang, Y.; Rusu, D.; Miao, J.; Zhang, W.; Alban, M.; Fadakar, I.; Chen, Z.; Huang, C.; Wen, Y.; Hassanzadeh, K.; Graves, D.; Zhu, Z.; Ni, Y.; Nguyen, N.; Elsayed, M.; Ammar, H.; Cowen-Rivers, A.; Ahilan, S.; Tian, Z.; Palenicek, D.; Rezaee, K.; Yadmellat, P.; Shao, K.; Chen, D.; Zhang, B.; Zhang, H.; Hao, J.; Liu, W.; and Wang, J. 2021. SMARTS: An Open-Source Scalable Multi-Agent RL Training School for Autonomous Driving. In Proceedings of the 4th Conference on Robot Learning, 264–285.

## Appendix A. Implementation Details

This section provides additional implementation details of TPSP, including training hyperparameters, training infrastructure and safety-related reward implementation. These details are provided to facilitate the reproducibility of our experiments.

## Training Hyperparameters

TPSP consists of a driving policy $\pi _ { \theta }$ optimized with PPO and a scene perturbation network $G _ { \phi }$ trained through threatguided optimization. The main configurations of these components are summarized below.

Driving Policy Configuration The driving policy π<sub>θ</sub> is optimized using PPO in the GPUDrive environment. The main architecture and training configurations are summarized in Table 3.

Table 3: Architecture configuration of the driving policy.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Network architecture</td><td>MLP-based actor network</td></tr><tr><td>Hidden dimension</td><td>64</td></tr><tr><td>Latent policy dimension</td><td>64</td></tr><tr><td>Activation function</td><td>tanh</td></tr><tr><td>Action space</td><td>Discrete</td></tr><tr><td>Steering bins</td><td>13</td></tr><tr><td>Acceleration bins</td><td>7</td></tr></table>

Scene Perturbation Network Configuration The scene perturbation network $G _ { \phi }$ generates policy-aware scene modifications based on encoded simulator states and detached policy features. The main configurations are summarized in Table 4.

Table 4: Configuration of the scene perturbation network.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Encoder hidden dimension</td><td>128</td></tr><tr><td>Fusion dimension</td><td>256</td></tr><tr><td>Policy feature dimension</td><td>64</td></tr><tr><td>Number of selected objects K Initial log standard deviation</td><td>8 -1.0</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Entropy coefficient</td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>0.5</td></tr><tr><td>Update interval</td><td></td></tr><tr><td>Initial perturbation scale λ</td><td>4 rollouts 0.2</td></tr></table>

PPO Optimization Configuration The detailed PPO optimization parameters used for updating the driving policy are summarized in Table 5.

Table 5: PPO optimization configuration for training the driving policy.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer Learning rate</td><td>AdamW  $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>PPO clip ratio Number of PPO epochs</td><td>0.2 2</td></tr><tr><td>Discount factor γ GAE coefficient λGAE</td><td>0.99 0.95</td></tr><tr><td>Experience batch size Mini-batch size</td><td>204800</td></tr><tr><td></td><td>6400</td></tr><tr><td>Value loss coefficient</td><td>0.5</td></tr><tr><td>Entropy coefficient</td><td></td></tr><tr><td></td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>0.5</td></tr></table>

## Training Infrastructure

All experiments are conducted on the GPUDrive simulation platform. The hardware and software configurations used for training are summarized in Table 6.

Table 6: Training infrastructure configuration.
<table><tr><td>Configuration</td><td>Specification</td></tr><tr><td>GPU</td><td>NVIDIA A100</td></tr><tr><td>Number of GPUs</td><td>16</td></tr><tr><td>Memory</td><td>1024GB</td></tr><tr><td>Number of CPU Cores</td><td>128</td></tr><tr><td>Operating System</td><td>Linux</td></tr><tr><td>Deep learning framework</td><td>PyTorch</td></tr><tr><td>Simulation backend</td><td>GPUDrive</td></tr><tr><td>Parallel simulation environments</td><td>512</td></tr></table>

## Safety-related Reward Implementation

The driving policy is optimized using the reward provided by the GPUDrive simulator. This section describes the implementation of two safety-related reward components: collision and of-road penalties. At each simulation step, GPUDrive provides safety event indicators for each agent. The collision signal combines collisions with vehicles and other objects:

$$
\begin{array} { r } { d _ { t } ^ { \mathrm { c o l } } = d _ { t } ^ { \mathrm { v e h } } + d _ { t } ^ { \mathrm { o b j } } , } \end{array}\tag{19}
$$

where $d _ { t } ^ { \mathrm { v e h } }$ and $d _ { t } ^ { \mathrm { { o b j } } }$ denote collision events with vehicles and other objects at timestep t, respectively. The of-road signal $d _ { t } ^ { \mathrm { o f f } }$ indicates whether the ego vehicle leaves the drivable area. The corresponding safety-related reward terms are defined as:

$$
r _ { t } ^ { \mathrm { s a f e } } = - 3 . 5 d _ { t } ^ { \mathrm { c o l } } - 0 . 7 5 d _ { t } ^ { \mathrm { o f f } } + 0 . 5 d _ { t } ^ { \mathrm { g o a l } } ,\tag{20}
$$

where $d _ { t } ^ { \mathrm { g o a l } }$ denotes goal achievement. The larger collision penalty encourages the policy to prioritize collision avoidance, while the of-road penalty provides continuous guidance for maintaining valid driving areas.

The collision penalty is applied only at the collision timestep since collided agents are removed by the simulator. In contrast, of-road penalties can accumulate when the vehicle remains outside the drivable area. Other shaping reward components provided by GPUDrive remain unchanged during training.

## Appendix B. Additional Discussion on NAVSIM Comparison and Training Pipeline

This section further discusses the comparison between TPSP and existing NAVSIM v2 leaderboard methods, and describes the training pipeline used to enable online reinforcement learning with NAVSIM scenes. We clarify the diferences in learning paradigms and input representations between TPSP and existing benchmark approaches.

## Comparison with NAVSIM v2 Leaderboard Methods

Table 2 compares TPSP with representative methods reported on the oficial NAVSIM v2 leaderboard. These external results are included for reference under the same benchmark evaluation protocol.

Most existing NAVSIM leaderboard methods follow an ofline data-driven end-to-end autonomous driving paradigm(Cao et al. 2025; Feng et al. 2026; Li et al. 2025; Tian et al. 2026; Kirby et al. 2026). They typically learn driving policies or trajectory planners from large-scale recorded driving data, where raw sensor observations provided by NAVSIM, such as multi-view camera inputs, are directly used as model inputs. Therefore, these approaches generally adopt a one-stage learning pipeline that maps perceptionlevel observations to future trajectories or driving actions.

In contrast, TPSP is designed as an online reinforcement learning framework that improves policy learning through interactive simulation. Instead of directly consuming raw sensor observations, TPSP utilizes structured white-box information available from the simulator, including ego states, surrounding object states, road information, and risk-related features. Based on these structured representations, TPSP first constructs a policy-aware scene representation and then performs online RL optimization with targeted scene perturbations.

Therefore, TPSP difers from existing NAVSIM methods in both learning paradigm and system pipeline. Existing approaches mainly improve driving performance by developing stronger perception and planning models from ofline data, whereas TPSP focuses on improving the quality of online policy training experiences through simulator-based interaction. Despite these diferences, TPSP achieves competitive performance on NAVSIM v2 safety evaluation, demonstrating the efectiveness of online RL with policy-aware scene optimization.

## NAVSIM-to-GPUDrive Training Pipeline

TPSP requires interactive simulation for online reinforcement learning, whereas NAVSIM v2 provides recorded driving scenes for benchmark evaluation. Therefore, the original NAVSIM v2 scenes cannot be directly used for GPUDrive simulation and training.

To enable online training, we transform NAVSIM v2 scenes into GPUDrive-compatible simulation environments. The conversion process preserves essential scene information, including map structures, dynamic agent states, and temporal interactions, while enabling large-scale parallel simulation in GPUDrive.

After conversion, TPSP performs online reinforcement learning on the generated GPUDrive scenes. The learned policy is finally evaluated on the oficial NAVSIM v2 navhard\_two\_stage benchmark following the standard evaluation protocol, allowing comparison with the publicly reported leaderboard results.
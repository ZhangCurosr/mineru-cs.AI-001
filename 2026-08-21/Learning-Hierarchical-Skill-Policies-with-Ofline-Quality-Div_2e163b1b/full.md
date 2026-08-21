# Learning Hierarchical Skill Policies with Ofline Quality-Diversity Reinforcement Learning

Tanachai Anakewat<sup>1</sup>, Takayuki Osa<sup>2</sup>, and Tatsuya Harada<sup>1,2</sup>

<sup>1</sup>The University of Tokyo <sup>2</sup>RIKEN Center for Advanced Intelligence Project Tokyo, Japan

Abstract— Recent studies investigate how to leverage precollected datasets to improve the policy performance and sample eficiency of RL. One promising approach to achieve this goal is to employ a two-stage strategy: In the first stage, diverse skills are extracted as a low-level policy from a given dataset, and a high-level policy is trained to solve a specific task in the second stage. Typically, extraction of the low-level policy is performed based on unsupervised learning such as trajectory VAE. However, a limitation of this approach is that the quality of the low-level policy highly depends on the quality of the dataset. To address this issue, we introduce QDOS (Quality-Diversity Ofline Skill learning), a unified pipeline for robust ofline-to-online learning. Our approach incorporates an Advantage-Weighted Quality-Diversity pretraining objective, which weights the skill extraction and diversity objectives by the estimated advantage of each trajectory segment. This approach allows the model to extract diverse and high-value skills. By providing robust and task-relevant skill representations, QDOS significantly improves the quality of the embedded skill space used by the low-level policy. We further integrate this with a dual dataset reuse strategy, where ofline data is used both for skill pretraining and for populating the online replay bufer via pseudo-labeling. Experiments demonstrate that QDOS significantly outperforms strong baselines in structured manipulation tasks and unstructured locomotion tasks, confirming its ability to accelerate exploration and improve final returns in challenging sparse-reward domains.

## I. INTRODUCTION

Deep reinforcement learning (RL) has achieved remarkable success in solving complex control tasks, ranging from playing video games to manipulating robotic arms [1], [2]. Despite these achievements, the standard online RL paradigm requires a large amount of interactions with the environment to learn efective policies. This sample ineficiency makes it prohibitively expensive and often unsafe for real-world applications, such as robotics, where data collection is costly and potentially dangerous.

To address these limitations, Ofline RL has emerged as a promising alternative, enabling agents to learn policies entirely from static, pre-collected datasets without online interaction [3]. In parallel, unsupervised skill discovery methods aim to extract reusable behaviors, or “skills,” from large, unlabeled datasets. Recent methods such as OPAL [4] and SPiRL [5] encode primitive skills into a latent space and train a high-level policy that leverages these low-level policies to solve complex tasks. This temporal abstraction reduces the efective horizon of the task, simplifying the learning of long-horizon problems.

![](images/4bd3418d08fb3b14f42964cb3fd69dcca9ade616fb5f15bb2831071a2e31b6ac.jpg)  
Fig. 1. Overview of QDOS Framework. (1) Ofline Pretraining: The agent learns low-level skills (π<sub>θ</sub>) from a mixed-quality dataset containing good moves, failed attempts, and noise. We employ an Advantage-Weighted Quality-Diversity objective to filter out sub-optimal data (failed attempts and noise) while extracting diverse, high-quality skills. (2) Online RL with Dual Dataset Reuse: The same ofline data is reused to populate the high-level replay bufer via pseudo-labeling, enabling the high-level policy to utilize prior knowledge for eficient online exploration.

Recent work has begun to bridge these two fields in the “ofline-to-online” setting, where an agent is pretrained on ofline data and then fine-tuned online interacting with the environment. Approaches such as SUPE [6] demonstrate that reusing ofline data for both skill extraction and as a replay bufer for the high-level policy can dramatically accelerate exploration, resulting in higher performance after online RL.

However, a critical challenge remains: how to identify and extract useful behaviors when the ofline data contains a mixture of optimal and suboptimal actions? Real-world datasets often contain optimal trajectories entangled with noise, failed attempts, and irrelevant behaviors. Standard trajectory VAEs, as used in previous methods, treat all data segments equally, aiming to minimize the reconstruction error over the entire dataset. When the ofline data is noisy, these methods learn a “cluttered” skill space where useful skills are entangled with useless ones, making it dificult for the high-level policy to select efective actions.

This leads to a trade-of in current approaches. Purely diversity-driven methods (e.g., maximizing mutual information) encourage distinct behaviors but fail to distinguish between distinct useful behaviors and distinct useless ones. Conversely, purely reward-driven or behavior-cloning approaches often collapse to a single mode, failing to capture the diversity of solutions necessary for robust downstream exploration. Consequently, there is a lack of a unified framework that can filter out noise while preserving a diverse set of high-quality skills suitable for accelerating online learning.

In this work, we propose QDOS (Quality-Diversity Ofline Skill learning), a unified framework for robust ofline-toonline learning, as illustrated in Fig. 1. Our method addresses the mixed-quality dilemma by introducing skill learning based on unsupervised multi-solution discovery using ofline RL. The objective function for our skill learning incorporates both the expected return and a skill-diversity term, thereby yielding diverse and high-quality behaviors. This approach enables the learning of a skill space composed of useful primitives. We integrate this into an ofline-toonline framework that reuses ofline data twice: first for skill extraction, and second for populating the online replay bufer via “pseudo-labeling,” where ofline trajectories are annotated with inferred skills and optimistic reward estimates following SUPE [6]. This allows the high-level policy to perform of-policy learning on the extensive ofline dataset from the start.

We extensively evaluate our method on diverse and complex domains, including antsoccer, kitchen, and humanoidmaze. Our experiments demonstrate that our method significantly outperforms state-of-the-art baselines (such as SUPE [6]) in terms of sample eficiency, asymptotic performance, and goal-finding speed, particularly in tasks requiring the composition of diverse, dynamic behaviors.

## II. RELATED WORK

## A. Unsupervised Skill Discovery

Unsupervised skill discovery aims to learn reusable behaviors without extrinsic rewards, typically by maximizing an intrinsic objective. Early online methods focused on maximizing the mutual information between latent codes and states [7], [8] or regularizing option policies [9]. These methods encourage the agent to visit diverse states and learn distinguishable behaviors.

In the ofline setting, methods like OPAL [4] and SPiRL [5] adapt these ideas by using trajectory VAEs to learn skills from fixed demonstration datasets. They compress short trajectory segments into a latent variable z, allowing a highlevel policy to operate in this latent space. However, these methods typically discard the ofline data after the pretraining phase. Furthermore, they do not account for data quality; they try to reconstruct all trajectories, including suboptimal ones, which can degrade the quality of the learned skill space when the dataset is noisy.

## B. Ofline Reinforcement Learning

Ofline RL focuses on learning optimal policies from static datasets. The primary challenge is distribution shift, where the learned policy visits states outside the dataset support, leading to overestimation of values. Algorithms like CQL [10] and IQL [11] address this by constraining the policy or value function. While efective for learning from fixed data, these methods can be overly conservative, which hinders exploration when the agent is allowed to interact with the environment online.

## C. Ofline-to-Online Reinforcement Learning

Ofline-to-online RL leverages ofline data to accelerate online learning. A common strategy is to initialize an RL agent with a policy pretrained ofline and then fine-tune it online [12], [13]. Recent work like SUPE [6] proposes a more integrated approach, reusing ofline data for both skill learning and online replay. By pseudo-labeling ofline trajectories with inferred skills and rewards, SUPE allows the high-level policy to learn from the rich ofline dataset. However, SUPE treats all ofline data equally during skill learning, inheriting the brittleness of standard VAEs when dealing with mixed-quality data. Our method improves upon this by applying advantage weighting during skill discovery to ensure that only beneficial but diverse behaviors are captured and reused.

## D. Quality Diversity RL

Quality-Diversity (QD) optimization is a paradigm that aims to generate large collections of diverse solutions that are all high-performing, contrasting with pure optimization which seeks a single global optimum. This concept was introduced by the Generative and Developmental Systems community [14], [15] with algorithms such as Novelty Search with Local Competition and MAP-Elites. QD algorithms operate in a behavioral or feature space rather than the genotypic space, attempting to fill the behavior space with high-performing solutions even if they are not global peaks in the fitness landscape. In robotics, QD has been used to create repertoires of behaviors [16] and for damage adaptation. It has also been applied to engineering design [17] and video game level generation [18].

Recently, these ideas have been adapted to Reinforcement Learning. DiveOf [19] introduces a QD perspective to offline RL, aiming to discover multiple solutions from a single task. DiveOf employs a variational objective that maximizes the mutual information between latent skills and trajectories to ensure diversity, while simultaneously regularizing the latent space to maintain high quality. Specifically, it uses a coordinate ascent approach to prevent mode collapse and ensure that the learned skills cover the manifold of useful behaviors. Our work builds upon this by explicitly integrating advantage-weighting into the QD objective, ensuring that diversity is sought specifically among high-value trajectories.

## E. Limitations of Current Approaches in Mixed-Quality Data

Despite the promise of data reuse, existing methods like SUPE and OPAL rely on the assumption that the ofline dataset consists of coherent, segmentable behaviors. In realworld scenarios, datasets are often unstructured, containing a mix of optimal trajectories, exploratory noise, and failed attempts. Standard trajectory VAEs minimize reconstruction error over the entire dataset, forcing the model to allocate representational capacity to noise. This results in a “cluttered” skill space where distinct latent codes map to indistinguishable or useless behaviors. Our work addresses this critical gap by introducing an advantage-weighted objective that extracts high-quality segments and learns a latent skill space that corresponds to high-advantage behaviors.

## III. PRELIMINARIES

## A. Problem Formulation

We consider a Markov Decision Process (MDP) defined by the tuple $\mathcal { M } = \{ \boldsymbol { S } , \mathcal { A } , \mathcal { P } , \gamma , r , \rho \}$ , where S is the state space, $\mathcal { A }$ is the action space, $\mathcal { P }$ is the transition dynamics, $\gamma$ is the discount factor, r is the reward function, and $\rho$ is the initial state distribution.

We assume access to a static ofline dataset $\mathcal { D } = \{ \tau _ { i } \} _ { i = 1 } ^ { N } .$ where each trajectory $\tau _ { i }$ consists of a sequence of stateaction-reward tuples $\tau _ { i } = ( s _ { 0 } , a _ { 0 } , r _ { 0 } , \ldots , s _ { T } , a _ { T } , r _ { T } )$ . Crucially, while rewards are present, the dataset is unlabeled regarding skill definitions. There are no segmentations or labels z indicating the active primitive behavior. The dataset contains both good and bad actions mixed together; the agent must infer which behaviors are worth learning and reusing.

The goal is two-fold:

1) Skill Discovery: Learn a low-level latent-conditioned policy $\pi _ { \boldsymbol { \theta } } ( a \mid s , z )$ and prior $p ( z )$ that capture diverse, useful functional behaviors while filtering out noise and suboptimal data.

2) Hierarchical Adaptation: Learn a high-level policy $\pi _ { \psi } ( z \mid s )$ that selects these skills to maximize expected return $\eta ( \pi ) = \mathbb { E } [ \sum \gamma ^ { t } r _ { t } ]$ during online interaction.

## B. Limitations of Current Approaches

SUPE (Skills from Unlabeled Prior data for Exploration) [20] uses a standard VAE to embed skills from the ofline dataset and reuses this data to populate the online highlevel replay bufer. While SUPE bridges ofline skill learning and online RL, it implicitly assumes that the extracted skills are meaningful. Standard trajectory VAEs treat all segments as valid signals. In noisy datasets, the VAE reconstructs noise, resulting in a “cluttered” library where distinct latents map to indistinguishable or useless behaviors. This leads to unreliable pseudo-labeling, causing the high-level policy to receive inconsistent signals and fail to learn robust strategies. This mixed-quality issue is illustrated in Fig. 2.

## IV. METHOD

In this section, we detail our proposed framework, QDOS, which consists of two main phases: (1) Advantage-Weighted

![](images/dd5b9db9d3e2824f1932d088f4a9a7fb8268969d7cb70fa4b8d264e6c8e6766f.jpg)  
Fig. 2. The Mixed-Quality Dilemma: Standard VAEs encode all behaviors equally, entangling noise with useful skills.

Quality-Diversity Skill Pretraining, and (2) Online Exploration with Trajectory Skills. The overall procedure is summarized in Algorithm 1.

## A. Advantage-Weighted Quality-Diversity Pretraining

Our skill learning framework is inspired by DiveOf [19], which introduces a Quality-Diversity approach to ofline RL. While the primary goal of DiveOf is to discover multiple diverse solutions within a static dataset, our objective is to reuse these diverse, high-performing solutions to accelerate online fine-tuning. By embedding these skills into a latent space, we provide the high-level policy with a structured action space that facilitates eficient exploration.

To efectively extract useful behaviors from mixed-quality datasets, we propose an Advantage-Weighted VAE framework. Standard VAEs minimize the reconstruction error over all data segments uniformly. This forces the model to dedicate capacity to reconstructing noise and suboptimal behaviors, cluttering the latent space. Instead, we propose to weight the learning objective by the estimated advantage of each trajectory segment.

1) Advantage Estimation via IQL: We first estimate the advantage of each trajectory segment. We segment the ofline trajectories into fixed-length segments $\tau _ { [ H ] } =$ $\left\{ s _ { 0 } , a _ { 0 } , \dots , s _ { H - 1 } , a _ { H - 1 } \right\}$ of length $H .$ . To estimate the quality of these segments, we train a value function $V _ { \psi } ( s )$ on the ofline dataset using Implicit Q-Learning (IQL) [11]. IQL learns a value function that approximates the upper expectile of the return distribution, efectively capturing the value of the best policy supported by the data without querying outof-distribution actions. The IQL value function is trained by minimizing the expectile regression loss:

$$
L _ { V } ( \psi ) = \mathbb { E } _ { ( s , a ) \sim \mathcal { D } } [ L _ { 2 } ^ { \tau } ( Q _ { \hat { \psi } } ( s , a ) - V _ { \psi } ( s ) ) ] ,\tag{1}
$$

where $L _ { 2 } ^ { \tau } ( u ) = | \tau - \mathbb { I } ( u < 0 ) | u ^ { 2 }$ is the expectile loss with $\tau \in ( 0 . 5 , 1 )$ and $Q _ { \hat { \psi } }$ denotes the target critic network. This allows us to estimate the potential return from any state in the dataset. The advantage of a segment starting at state $s _ { 0 }$ can then be estimated. We compute a weight $w ( \tau _ { [ H ] } )$ for each segment:

$$
w ( \tau _ { [ H ] } ) \propto \exp \left( \frac { A ( \tau _ { [ H ] } ) } { \lambda } \right) ,\tag{2}
$$

where $\lambda$ is a temperature parameter controlling the sharpness of the weight. High-advantage segments receive significantly higher weights, efectively focusing the VAE on learning from the “good” parts of the dataset while ignoring the “bad” parts. This filtering is crucial for preventing the skill space from being polluted by noise.

2) Weighted VAE Objective: We learn the skill space using a Variational Autoencoder (VAE). The encoder $f _ { \boldsymbol { \theta } } ( z \mid$ $\tau _ { [ H ] } )$ maps a trajectory segment to a latent variable $z ,$ and the decoder (low-level policy) $\pi _ { \theta } ( a \mid s , z )$ reconstructs the actions. We also learn a state-dependent prior $p _ { \theta } ( z \mid s _ { 0 } )$ to capture the distribution of skills available at a given state. The weighted VAE objective is defined as:

$$
\begin{array} { l } { { \displaystyle \mathcal { L } _ { \mathrm { V A E } } ^ { \mathrm { A W } } ( \theta ) = \beta D _ { \mathrm { K L } } ( f _ { \theta } ( z \mid \tau _ { \lbrack H ] } ) \parallel p _ { \theta } ( z \mid s _ { 0 } ) ) } } \\ { { \displaystyle - \mathbb { E } _ { z \sim f _ { \theta } } \left[ w ( \tau _ { [ H ] } ) \sum _ { t = 0 } ^ { H - 1 } \log \pi _ { \theta } ( a _ { t } \mid s _ { t } , z ) \right] . } } \end{array}\tag{3}
$$

This objective ensures that the model prioritizes the reconstruction of high-value behaviors. The KL-divergence term regularizes the learned posterior towards the prior, ensuring a smooth latent space.

3) Mutual Information Maximization: To prevent mode collapse (where the policy ignores the latent code) and ensuring diversity among the learned skills, we maximize the mutual information $I ( z ; \tau )$ between the latent skill z and the generated trajectory τ . Following variational information maximization principles [19], [21], we utilize a variational lower bound:

$$
I ( z ; \tau ) \geq \mathbb { E } _ { z \sim p ( z ) , \tau \sim \pi _ { \theta } ( \cdot \vert z ) } \left[ \log q _ { \theta } ( z \mid \tau ) \right] + H ( z ) ,\tag{4}
$$

where $q _ { \theta } ( z | \tau )$ is a parameterized posterior distribution. Since the entropy $H ( z )$ is constant for a fixed prior, maximizing mutual information is equivalent to maximizing the expected log-likelihood of the posterior. To maximize the entropy $H ( z )$ , a natural choice for the prior $p ( z )$ is the uniform distribution. In practice, we sample random skills $z _ { \mathrm { r a n d } } \sim$ $p ( z )$ , generate short auxiliary rollouts with the current policy $\pi _ { \theta } ,$ , and minimize the reconstruction error of the latent skill from these generated trajectories. Crucially, we also weight this diversity objective by the trajectory advantage. This ensures that we encourage diversity specifically among highquality behaviors, rather than encouraging diverse failure modes. The expected log-likelihood of the posterior can be maximized by minimizing the following mean-squared L2 norm:

$$
\mathcal { L } _ { \mathrm { i n f o } } ( \theta ) = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } w ^ { ( i ) } \left\| \hat { z } _ { \mathrm { r a n d } } ^ { ( i ) } - z _ { \mathrm { r a n d } } ^ { ( i ) } \right\| _ { 2 } ^ { 2 } ,\tag{5}
$$

where ${ \it z } _ { \mathrm { r a n d } }$ is sampled from the prior, and $\hat { z } _ { \mathrm { r a n d } }$ is encoded from the rollout of $\pi _ { \theta } \big ( z _ { \mathrm { r a n d } } \big )$

The total pretraining objective is a weighted combination:

$$
\begin{array} { r } { \mathcal { L } _ { \theta } = \mathcal { L } _ { \mathrm { V A E } } ^ { \mathrm { A W } } + \alpha _ { \mathrm { l o s s } } \mathcal { L } _ { \mathrm { i n f o } } . } \end{array}\tag{6}
$$

## B. Online Exploration with Trajectory Skills

Following pretraining, the low-level skill policy $\pi _ { \theta }$ is frozen. We then introduce a high-level policy $\pi _ { \psi } ( z \mid s )$ that operates in the latent skill space, selecting a new skill z every H environment steps.

```latex
Algorithm 1 QDOS: Advantage-Weighted Quality-Diversity
Skill Pretraining and Online RL
1: Inputs: dataset D, segment length H, batch size B, replay bufer $\overline { { \mathcal { D } _ { \mathrm { r e p l a y } }  \emptyset } } ,$
temperature $\lambda ,$ mutual-information weight α.
2: Phase 1: Advantage-Weighted Skill Pretraining
3: for each training step do
4: Sample segments $\{ \tau _ { [ H ] } ^ { ( i ) } \} _ { i = 1 } ^ { B } \sim \mathcal { D } .$
5: Encode posterior q<sub>θ</sub> $\dot { ( \boldsymbol { z } \mid \tau _ { [ H ] } ^ { ( i ) } ) }$ and prior p<sub>θ</sub>(z ${ | \ s _ { 0 } ^ { ( i ) } \rangle } ;$ sample $z ^ { ( i ) } .$
6: Update IQL critics $Q _ { \psi } , V _ { \psi _ { \ast } } ^ { \dagger }$ to estimate values.
7: Compute advantages $\begin{array} { r } { \hat { 1 } _ { \psi } ( s _ { 0 } ^ { ( i ) } , z ^ { ( i ) } ) = Q _ { \psi } ( s _ { 0 } ^ { ( i ) } , z ^ { ( i ) } ) - V _ { \psi } ( s _ { 0 } ^ { ( i ) } ) . } \end{array}$
8: Compute weights $\mathbf { \bar { \Pi } } _ { w } ( i )$ ∝ exp( $A _ { \psi } / \lambda )$ (normalized).
9: Weighted VAE loss:
$\mathcal { L } _ { \mathrm { V A E } } ^ { \mathrm { A W } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } w ^ { ( i ) } \left[ \beta D _ { \mathrm { K L } } - \mathbb { E } _ { z } \left[ \log \pi _ { \theta } \right] \right] .$
10: Mutual-information loss: Sample $z _ { \mathrm { r a n d } } ^ { ( i ) } ,$ rollout $\pi _ { \theta } ,$ encode $\hat { z } _ { \mathrm { r a n d } } ^ { ( i ) } \colon$
$\mathcal { L } _ { \mathrm { i n f o } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } w ^ { ( i ) } \left\| \hat { z } _ { \mathrm { r a n d } } ^ { ( i ) } - z _ { \mathrm { r a n d } } ^ { ( i ) } \right\| _ { 2 } ^ { 2 } .$
11: Update θ via ${ \mathcal { L } } _ { \theta } = { \mathcal { L } } _ { \mathrm { V A E } } ^ { \mathrm { A W } } + \alpha { \mathcal { L } } _ { \mathrm { i n f o } } .$
12: end for
13: Phase 2: Online RL with Pseudo-Labeling
14: Initialize: High-level policy $\pi _ { \psi } ( z | s )$ and $\begin{array} { r } { \breve { \cal D } _ { \mathrm { o f f l i n e } } = \emptyset . } \end{array}$
15: for each ofline segment $\tau _ { [ H ] } ^ { \dot { ( i ) } } \in \dot { \mathcal { D } }$ do
16: Infer $\hat { z } ^ { ( i ) } \sim q _ { \theta } ( z \mid \tau _ { [ H ] } ^ { \left. i \right. } )$ and optimistic reward $\hat { r } ^ { ( i ) }$
17: Add $( s _ { 0 } ^ { ( i ) } , \hat { z } ^ { ( i ) } , \hat { r } ^ { ( i ) } , \dot { s } _ { H } ^ { ( i ) } )$ to $\mathcal { D } _ { \mathrm { o f f l i n e } }$
18: end for
19: for each online interaction step do
20: Sample $z \sim \pi _ { \psi } ( z \mid s ) ,$ execute $\pi _ { \theta } ( a \mid s , z )$ for H steps.
21: Store transition in D<sub>replay</sub>.
22: Update $\pi _ { \psi }$ using $\mathcal { D } _ { \mathrm { r e p l a y } } \cup \mathcal { D } _ { \mathrm { o f f l i n e } }$ with of-policy RL (e.g., IQL).
23: end for
```

To accelerate the learning of this high-level policy, we employ the dual dataset reuse strategy from SUPE [6]. We “pseudo-label” the ofline dataset D to create a replay bufer of high-level transitions. For each ofline segment $\tau _ { [ H ] }$ , we: 1. Infer the skill label $\hat { z } \sim q _ { \theta } ( z \mid \tau _ { [ H ] } )$ using the pretrained encoder. 2. Compute an optimistic reward estimate $\hat { r }$ using an upper-confidence bound (UCB) estimator. We use Random Network Distillation (RND) [22] to add an exploration bonus to the reward predicted by a learned reward model.

These pseudo-labeled transitions $\left( { { s _ { 0 } } , \hat { z } , \hat { r } , { s _ { H } } } \right)$ are added to the online replay bufer. This allows the high-level agent to perform of-policy learning (using an algorithm like RLPD [12] or SAC [23]) on the rich ofline data from the very beginning of the online phase. The combination of highquality, advantage-weighted skills and optimistic exploration bonuses guides the agent towards promising regions of the state space, significantly accelerating exploration.

## C. Implementation Notes

We implement QDOS using PyTorch. For skill pretraining, we use the Adam optimizer with a learning rate of $3 \times 1 0 ^ { - 4 }$ . The segment horizon is set to $H = 2 0$ for all tasks. The advantage weights are computed using a value function trained via IQL on the ofline dataset. The expectile parameter τ is fixed at 0.9 for all environments, while the temperature parameter λ is tuned per environment (typically $\lambda \in [ 0 . 3 , 1 . 0 ] \rangle$ . The mutual information weight $\alpha _ { \mathrm { l o s s } }$ is also a hyperparameter, with values in {0.1, 0.2, 0.3} tested.

During online adaptation, we employ an entropyregularized actor-critic (SAC) for the high-level policy. The

antsoccer-arena-navigate-singletask-task1-v0

replay bufer is initialized with the pseudo-labeled ofline data, and we maintain a 50/50 sampling ratio between online and ofline data during high-level updates to ensure stable learning. All experiments are averaged over 5 random seeds.

## V. EXPERIMENTS

## A. Experimental Setup

We evaluate QDOS on four challenging sparse-reward domains that require long-horizon planning and precise control:

• antmaze: A locomotion task where a quadrupedal ant must navigate a maze to reach a goal. We use the large maze layout.

• kitchen: A sequential manipulation task where a robotic arm must interact with multiple objects (e.g., microwave, kettle, switch) in a kitchen environment. We use the mixed and partial datasets.

• humanoidmaze: A high-dimensional locomotion task where a humanoid agent must navigate a maze. This task is significantly harder than AntMaze due to the stability requirements of the humanoid.

• antsoccer: A complex task where an ant must push a soccer ball to a goal location. This requires both locomotion and object manipulation skills.

We compare our method against several strong baselines:

• SUPE [6]: Our primary baseline, a state-of-the-art ofline-to-online method that uses trajectory skills without advantage weighting.

• ExPLORe [24]: A method that uses exploration bonuses on ofline data but learns a flat (nonhierarchical) policy.

• Trajectory skills [4], [5]: A baseline using standard trajectory-based skill learning without advantage weighting or diversity maximization.

• IQL [11]: An ofline RL method that learns a value function using expectile regression to avoid out-ofdistribution actions.

• BC: Behavior Cloning, which learns a policy by maximizing the likelihood of actions in the ofline dataset. For the learning curves, we compare against Difusion BC with Jump-Start RL (JSRL) [25], a strong baseline that initializes the policy with BC and fine-tunes it online.

We evaluate performance based on the normalized return (success rate for maze/soccer tasks, number of completed subtasks for Kitchen) and sample eficiency (number of steps to reach the goal).

## B. Overall Results

QDOS consistently outperforms baselines across all evaluated domains. The quantitative results are summarized in Table I, and the training dynamics are shown in Fig. 3.

In the highly complex antsoccer-arena environment, QDOS achieves a normalized return of 0.80, which is a substantial improvement over SUPE’s 0.27. This task requires fine-grained control to manipulate the ball, and standard skill discovery methods likely fail to capture these precise interactions amidst the noise of general locomotion. Our advantage-weighting ensures that the skills focus on the moments where the ball is successfully moved.

![](images/87b03c6d622b6a44b2bb31fb97912c4b181e1658674768ee10e65c47bffbcfbf.jpg)

![](images/40eee2ce941819cb6c25ee3943f153b03fda64e629cef817d7a0b34e69655b42.jpg)

humanoidmaze-medium-navigate-singletask-v0  
![](images/b85db956cc9453a9764387707dfda33978fb64c39b438495213739409bb041ba.jpg)

![](images/fc9da5bb7841598216b052878c6c4864c2809f5a88e518a535d647b902174174.jpg)

![](images/181e42baa9e235d0759726798851a3b084e4897f95683e2678195f58778618f3.jpg)

![](images/c0d3e4dca2089f398a2174f18e9596466e3b07aefc6960682975d7143e659af6.jpg)

Fig. 3. Learning curves across environments. QDOS is shown in red $( \alpha = 0 . 2 )$ and blue (α = 0.3); Baseline (SUPE), ExPLORe, Trajectory Skills, HILP Online, and Dif BC JSRL are shown in black, green, purple, brown, and orange, respectively.  
![](images/1d2240eba4850390fce9168ea143b95f634cc84b6b5286a3a06f28a1b05d3ede.jpg)  
Fig. 4. Training steps to first goal finding. QDOS (Red) finds the goal significantly faster than baselines, indicating more eficient exploration.

In kitchen-mixed, QDOS attains a perfect score of 4.00, efectively stitching together fragmented subtasks from the dataset to solve the full sequential task. The baseline SUPE achieves only 3.40, indicating it struggles to compose the necessary skills or fails to learn some subtasks entirely.

In humanoidmaze, our method achieves perfect performance (1.00), slightly outperforming SUPE. This confirms that even in high-dimensional state spaces, filtering for highquality skills leads to more robust locomotion.

## C. Goal Finding Eficiency

A key advantage of our method is the speed at which it discovers solutions. In sparse-reward tasks, the time to reach the first goal is a critical metric for exploration eficiency. We analyzed the number of environment steps required to reach the goal for the first time.

TABLE I  
NORMALIZED EVALUATION RETURNS COMPARING BC, IQL, SUPE, AND QDOS ACROSS VARIOUS DOMAINS. RESULTS FOR SUPE AND QDOS ARE AVERAGED OVER 3 SEEDS.
<table><tr><td>Environment</td><td> $\mathbf { B C } \ [ 1 1 ] \ [ 2 6 ]$ </td><td> $\mathbf { I Q L } \ [ 1 1 ] \ [ 2 6 ]$ </td><td>SUPE</td><td> $\mathbf { Q D O S } \ ( \alpha = 0 . 2 )$ </td><td> $\mathbf { Q D O S } \ ( \alpha = 0 . 3 )$ </td></tr><tr><td>antmaze-large</td><td> $4 1 \pm 7$ </td><td> $6 4 \pm 1 0$ </td><td> $8 3 . 0 \pm 2 1 . 0$ </td><td> $8 6 . 0 \pm 6 . 0$ </td><td> ${ \bf 9 6 . 0 \pm 6 . 0 }$ </td></tr><tr><td>antsoccer-arena</td><td> $5 \pm 1$ </td><td> $5 0 \pm 2$ </td><td> $2 7 . 0 \pm 3 1 . 0$ </td><td> $\overline { { 7 3 . 0 \pm 3 1 . 0 } }$ </td><td> ${ \bf 8 0 . 0 \pm 1 0 . 0 }$ </td></tr><tr><td>humanoidmaze-medium</td><td> $\hphantom { 0 } 8 \pm 2$ </td><td> $2 7 \pm 2$ </td><td> $9 7 . 0 \pm 6 . 0$ </td><td> $\overline { { 9 7 . 0 \pm 6 . 0 } }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td>kitchen-mixed</td><td> $5 1 . 5$ </td><td> $5 1 . 0$ </td><td> $8 5 \pm 1 3 . 2 5$ </td><td> $\mathbf { 1 0 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $8 3 . 2 5 \pm 2 8 . 7 5$ </td></tr><tr><td>kitchen-partial</td><td>38.0</td><td>46.3</td><td> $\pm \overline { { 1 4 . 2 5 \pm 1 4 . 5 0 } }$ </td><td> ${ \pm } { \bf 5 } . 7 { \bf 5 } \pm 2 { \bf 4 } . 7 { \bf 5 }$ </td><td> ${ \pm } 2 . 5 0 \pm 3 0 . 2 5$ </td></tr><tr><td>scene-task1</td><td> $5 \pm 1$ </td><td> $5 1 \pm 4$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr></table>

TABLE II

TRAINING STEPS TO REACH THE GOAL FOR THE FIRST TIME (MEAN  STD).
<table><tr><td rowspan=1 colspan=1>Environment</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Steps</td></tr><tr><td rowspan=1 colspan=1>antmaze-large</td><td rowspan=1 colspan=1>SUPEQDOS $( \alpha = 0 . 2 )$ </td><td rowspan=1 colspan=1> $1 3 3 3 4 \pm 5 7 7 4$  ${ \bf 1 1 6 6 8 } \pm { \bf 9 4 6 5 }$ </td></tr><tr><td rowspan=1 colspan=1>humanoidmaze</td><td rowspan=1 colspan=1>SUPEQDOS $( \alpha = 0 . 3 )$ </td><td rowspan=1 colspan=1> $\overline { { 2 3 3 3 4 \pm 6 2 9 2 } }$  ${ \bf 1 9 1 6 8 } \pm { \bf 8 0 3 6 }$ </td></tr><tr><td rowspan=1 colspan=1>scene</td><td rowspan=1 colspan=1>SUPEQDOS $( \alpha = 0 . 2 )$ </td><td rowspan=1 colspan=1> $\overline { { 5 8 3 4 \pm 1 4 4 3 } }$  ${ \bf 5 0 0 1 \pm 0 }$ </td></tr></table>

As shown in Table II and Fig. 4, QDOS consistently finds the goal faster than SUPE. For instance, in the scene task, QDOS reaches the goal in the minimum possible time (5001 steps), while SUPE takes significantly longer (5834 steps). In humanoidmaze, our method reduces the exploration time by over 4000 steps. This accelerated exploration is a direct result of the cleaner skill space: the high-level policy does not waste time exploring with noisy or inefective skills, allowing it to traverse the environment more eficiently.

## D. Hyperparameter Sensitivity

We analyzed the sensitivity of our method to the mutual information weight $\alpha _ { \mathrm { l o s s } }$ , which balances the reconstruction quality and the diversity of the skills. We observed a taskdependent trade-of.

For complex, dynamic tasks like antsoccer, a higher diversity weight $( \alpha _ { \mathrm { l o s s } } = 0 . 3 )$ yields the best performance. The solution space for manipulating a soccer ball is large, and encouraging a diverse range of interaction skills increases the likelihood of finding a successful strategy.

Conversely, for structured manipulation tasks like kitchen, a moderate diversity weight $( \alpha _ { \mathrm { l o s s } } ~ = ~ 0 . 2 )$ performs better. In these tasks, the optimal behaviors are more constrained (e.g., opening a microwave requires a specific motion). Excessive diversity can lead to the learning of task-irrelevant variations that distract the policy. This suggests that QDOS is flexible and can be tuned to the specific needs of the domain.

## E. Failure Case

We extended our hyperparameter analysis to high-quality datasets such as cube-single and cube-double that only require pick-and-place motion. As illustrated in Fig. 5, we observed that reducing the mutual information weight $\alpha _ { \mathrm { l o s s } }$ causes our method to converge towards the performance of the baseline. This trend reinforces our finding that for datasets with high-quality, structured demonstrations, the need for enforcing diversity via mutual information is diminished.

![](images/0e6d0cfb4caddb7088d42bdb73417fd4f5ff1835baf260ce8d915c4c592a4b85.jpg)

![](images/5b1666ac07c61a5f9e1308aff05c2a939d54d65d3d9aff774e758c6fb7fc373d.jpg)  
Fig. 5. Experimental results on the cube datasets. Reducing the diversity weight $\alpha _ { \mathrm { l o s s } }$ leads to performance closer to the SUPE baseline, indicating that explicit diversity enforcement is less critical for high-quality datasets.

## VI. VISUALIZATION OF THE MODEL

To better understand the properties of the learned skill space and how they contribute to performance, we conducted qualitative visualizations of the latent space and the resulting agent behaviors.

## A. Latent Space Structure

We analyzed the structure of the learned latent space by projecting the latent variables from the ofline dataset into a 2D space using t-SNE, as shown in Fig. 6. The visualization compares the latent space usage of QDOS (Orange) and SUPE (Blue) across kitchen, antmaze, and antsoccer tasks. We observe that the latent variables for QDOS appear to be distributed more broadly across the space compared to SUPE. This trend suggests that QDOS might be utilizing the latent space more extensively to represent a diverse set of skills, whereas SUPE seems to use the latent space in a more limited or concentrated manner.

## B. Behavior Visualization

We also visualized the rollout trajectories of the learned skills in the kitchen and antmaze domains. In kitchen, the learned skills correspond to distinct, semantically meaningful object interactions. For example, specific latent codes consistently trigger the “open microwave” or “move kettle” behaviors. The pseudo-labeling process accurately assigns these skills to the corresponding ofline segments, enabling the high-level policy to compose them efectively. As shown in Fig. 7, QDOS reaches the goal at $\ t \ = \ 2 2$ in the kitchen-mixed trajectory, and the trajectory pattern in latent space is clearly structured, which is consistent with efective skill composition.

![](images/3e9d686ea6e331d10929f9d93ab73abec1b4d44a04e6358673be3b70fa50aa42.jpg)

![](images/37adfd8755c3433ab330a30db39577058be00c739d90258171f6393e473fd984.jpg)

![](images/d43df87d19a36e43aca72eaddf49d38834da34e086c425f84167143def00460f.jpg)  
Fig. 6. t-SNE visualization (top) and skill usage timeline (bottom) across kitchen, antmaze, and antsoccer. Orange points represent QDOS $( \alpha = 0 . 3$ for antsoccer and antmaze, $\alpha = 0 . 2$ for kitchen), and Blue points represent SUPE. The t-SNE plots indicate that QDOS tends to utilize a broader region of the latent space, suggesting the capture of more diverse behaviors compared to the more restricted latent usage of SUPE.

![](images/f55b17d87487598de92ef712367807a0cccb5b7311fc55c3cec47a3b2fa56167.jpg)  
Fig. 7. QDOS skill trajectory visualization for kitchen-mixed. Red crosses mark selected timesteps $\dot { ( t = 0 , 7 , 1 4 , 2 2 ) }$ with corresponding scene snapshots.

In antmaze, the skills represent coherent directional movements (e.g., “move north”, “turn left”). By continuously manipulating the continuous latent variable z (e.g., interpolating between two latent codes), we observed smooth transitions between behaviors, such as gradually changing the movement direction or speed. This confirms that the latent space captures the underlying manifold of useful motions, providing a smooth and navigable action space for the highlevel policy. This smoothness is crucial for the stability of the high-level RL training.

## VII. CONCLUSION

We presented QDOS, a unified framework for Quality-Diversity ofline-to-online Reinforcement Learning. By integrating advantage weighting into the skill pretraining objective, QDOS efectively addresses the challenge of learning from mixed-quality ofline datasets. It distills robust, taskaligned skills while filtering out noise and failures, overcoming the limitations of standard unsupervised skill discovery methods.

We demonstrated that combining this robust skill learning with a dual dataset reuse strategy, using ofline data for both skill extraction and high-level initialization, leads to significant performance gains. Empirical results on challenging domains like antsoccer, kitchen, and humanoidmaze show that QDOS outperforms state-of-the-art baselines in terms of asymptotic return, sample eficiency, and exploration speed.

These findings highlight the importance of quality-aware representation learning in ofline RL. Future work will explore adaptive methods to automatically tune the qualitydiversity trade-of and investigate mechanisms for fine-tuning the low-level skills during online interaction to further adapt to novel situations.

## APPENDIX

## A. Environment Details

We evaluate our method on three distinct domains, each presenting unique challenges for skill discovery and hierarchical control.

1) State-based Locomotion: antmaze, humanoidmaze, and antsoccer involve controlling complex agents to navigate mazes or manipulate objects.

• antmaze: A standard benchmark from D4RL [27] where a quadrupedal ant must navigate a large maze. We use the large maze layout, which requires longhorizon planning.

• humanoidmaze: A domain from OGBench [26] featuring a high-dimensional humanoid agent. The agent must balance while navigating, making the skill space significantly more complex than AntMaze.

• antsoccer: A multi-stage task where an ant must navigate to a soccer ball and push it to a goal. This requires composing locomotion skills with object interaction dynamics.

2) State-based Manipulation: kitchen, cube, and scene focus on robotic manipulation.

• kitchen: A sequential manipulation task from D4RL [27] where a 7-DOF Franka robot must manipulate multiple objects (microwave, kettle, burner, switch) in sequence. We use the mixed dataset, which contains various subtasks being performed, but the 4 target subtasks are never completed in sequence together. We also use the partial dataset, which includes other tasks being performed, but there are sub-trajectories where the 4 target subtasks are completed in sequence.

• cube: A manipulation domain from OGBench [26] focusing on pick-and-place tasks. We use cube-single which requires lifting a single cube, and cube-double which involves manipulating two cubes.

• scene: A domain from OGBench [26] requiring the composition of atomic behaviors like locking/unlocking and opening drawers.

## B. Hyperparameters

We provide the detailed hyperparameters used for the VAE skill pretraining and the high-level online RL agent in Table III and Table IV, respectively.

TABLE III  
HYPERPARAMETERS FOR VAE TRAINING.
<table><tr><td>Parameter Name</td><td>Value</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td>3 × 10−4</td></tr><tr><td>GRU Hidden Size</td><td>256 (antmaze, kitchen) 512 (antsoccer, humanoidmaze, scene)</td></tr><tr><td>GRU Layers</td><td>2 (antmaze, kitchen) 3 (antsoccer, humanoidmaze, scene)</td></tr><tr><td>KL Coefficient (β)</td><td>0.1 (antmaze, humanoid, kitchen)</td></tr><tr><td>Latent Dimension (z)</td><td>0.2 (scene) 8</td></tr><tr><td>Segment Length (H)</td><td>20</td></tr></table>

TABLE IV

HYPERPARAMETERS FOR THE HIGH-LEVEL ONLINE RLAGENT.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Discount factor (γ)</td><td>0.99 (AntMaze, Kitchen)</td></tr><tr><td>Optimizer</td><td>0.995 (Humanoid, Scene, Soccer) Adam</td></tr><tr><td>Learning rate</td><td>3 × 10−4</td></tr><tr><td>Critic ensemble size</td><td>10</td></tr><tr><td>UTD Ratio</td><td>20</td></tr><tr><td>Network Width</td><td>256 (AntMaze, Kitchen)</td></tr><tr><td></td><td>512 (Humanoid, Soccer, Scene)</td></tr><tr><td>Network Depth</td><td>3 hidden layers</td></tr><tr><td>RND coefficient (α)</td><td>8.0</td></tr></table>

## C. Acknowledgement

This work was partially supported by JST Moonshot R&D Grant Number JPMJPS2011, CREST Grant Number JP-MJCR2015 and Basic Research Grant (Super AI) of Institute for AI and Beyond of the University of Tokyo. Takayuki Osa was supported by JSPS KAKENHI Grant Number JP25K03176.

## D. Use of Large Language Models (LLMs)

We utilized large language models to refine the clarity and grammar of the text throughout this manuscript. All scientific content, data, and figures are the original work of the authors.

## REFERENCES

[1] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, Feb. 2015. [Online]. Available: http://dx.doi.org/10.1038/nature14236

[2] S. Levine, C. Finn, T. Darrell, and P. Abbeel, “End-to-end training of deep visuomotor policies,” Journal of Machine Learning Research, vol. 17, no. 39, pp. 1–40, 2016.

[3] S. Levine, A. Kumar, G. Tucker, and J. Fu, “Ofline reinforcement learning: Tutorial, review, and perspectives on open problems,” arXiv preprint arXiv:2005.01643, 2020.

[4] A. Ajay, A. Kumar, P. Agrawal, S. Levine, and O. Nachum, “Opal: Ofline primitive discovery for accelerating ofline reinforcement learning,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/ forum?id=V69LGwJ0lIN

[5] K. Pertsch, Y. Lee, and J. Lim, “Accelerating reinforcement learning with learned skill priors,” in Conference on Robot Learning. PMLR, 2021, pp. 188–204.

[6] M. Wilcoxson, Q. Li, K. Frans, and S. Levine, “Leveraging skills from unlabeled prior data for eficient online exploration,” arXiv preprint arXiv:2410.18076, 2025.

[7] K. Gregor, D. J. Rezende, and D. Wierstra, “Variational intrinsic control,” arXiv preprint arXiv:1611.07507, 2016.

[8] B. Eysenbach, A. Gupta, J. Ibarz, and S. Levine, “Diversity is all you need: Learning skills without a reward function,” arXiv preprint arXiv:1802.06070, 2018.

[9] P.-L. Bacon, J. Harb, and D. Precup, “The option-critic architecture,” in Proceedings of the AAAI conference on artificial intelligence, vol. 31, no. 1, 2017.

[10] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative Qlearning for ofline reinforcement learning,” Advances in Neural Information Processing Systems, vol. 33, pp. 1179–1191, 2020.

[11] I. Kostrikov, A. Nair, and S. Levine, “Ofline reinforcement learning with implicit Q-learning,” arXiv preprint arXiv:2110.06169, 2021.

[12] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine, “Eficient online reinforcement learning with ofline data,” in International Conference on Machine Learning. PMLR, 2023, pp. 1577–1594.

[13] S. Lee, Y. Seo, K. Lee, P. Abbeel, and J. Shin, “Ofline-to-online reinforcement learning via balanced replay and pessimistic Q-ensemble,” in Conference on Robot Learning. PMLR, 2022, pp. 1702–1712.

[14] J. Lehman and K. O. Stanley, “Abandoning objectives: Evolution through the search for novelty alone,” Evolutionary Computation, vol. 19, no. 2, pp. 189–223, 2011.

[15] J.-B. Mouret and J. Clune, “Illuminating search spaces by mapping elites,” arXiv preprint arXiv:1504.04909, 2015.

[16] A. Cully, J. Clune, D. Tarapore, and J.-B. Mouret, “Robots that can adapt like animals,” Nature, vol. 521, no. 7553, pp. 503–507, 2015.

[17] A. Gaier, A. Asteroth, and J.-B. Mouret, “Data-eficient design exploration through surrogate-assisted illumination,” Evolutionary Computation, vol. 26, no. 3, pp. 381–410, 2018.

[18] A. Khalifa, S. Lee, A. Nealen, and J. Togelius, “Talakat: Bullet hell generation through constrained map-elites,” in Proceedings of the Genetic and Evolutionary Computation Conference, 2018, pp. 1047– 1054.

[19] T. Osa and T. Harada, “Discovering multiple solutions from a single task in ofline reinforcement learning,” Proceedings of the 41st International Conference on Machine Learning, 2024.

[20] Q. Li, M. Wilcoxson, K. Frans, and S. Levine, “Skills from unlabeled prior data for eficient exploration,” arXiv preprint arXiv:2410.18076, 2024.

[21] D. Barber and F. Agakov, “The im algorithm: a variational approach to information maximization,” Advances in neural information processing systems, vol. 16, no. 320, p. 201, 2004.

[22] Y. Burda, H. Edwards, A. Storkey, and O. Klimov, “Exploration by random network distillation,” arXiv preprint arXiv:1810.12894, 2018.

[23] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Ofpolicy maximum entropy deep reinforcement learning with a stochastic actor,” in International conference on machine learning. PMLR, 2018, pp. 1861–1870.

[24] Q. Li, J. Zhang, D. Ghosh, A. Zhang, and S. Levine, “Accelerating exploration with unlabeled prior data,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[25] I. Uchendu, T. Xiao, Y. Lu, B. Zhu, M. Yan, J. Simon, M. Bennice, C. Fu, C. Ma, J. Jiao, et al., “Jump-start reinforcement learning,” in International Conference on Machine Learning. PMLR, 2023, pp. 34 556–34 583.

[26] S. Park, K. Frans, B. Eysenbach, and S. Levine, “Ogbench: Benchmarking ofline goal-conditioned rl,” ArXiv, 2024.

[27] J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine, “D4rl: Datasets for deep data-driven reinforcement learning,” arXiv preprint arXiv:2004.07219, 2020.
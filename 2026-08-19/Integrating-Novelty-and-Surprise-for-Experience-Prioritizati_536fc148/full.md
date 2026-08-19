# Integrating Novelty and Surprise for Experience Prioritization and Exploration in Image-Based Reinforcement Learning

Hoda Yamani, Henry Williams, Bruce A. MacDonald

Abstract—Sample efficiency is a central challenge in reinforcement learning (RL), particularly in image-based domains where agents must learn from high-dimensional visual inputs. Traditional sampling often relies on random or suboptimal experience selection, leading to redundant updates and slow learning. Improving efficiency requires mechanisms that prioritize informative experiences while also encouraging effective exploration. Prioritized Experience Replay (PER) addresses part of this challenge by reusing highvalue transitions, while intrinsic rewards promote the exploration of novel or uncertain states. However, their integration has not been extensively studied. This paper introduces Novelty and Surprise Prioritized Experience Replay (NSPER), which uses novelty to capture underrepresented states and surprise to expose gaps in the agent’s understanding of the environment. We further extend this with NSPER+R, integrating these signals as intrinsic rewards to jointly improve replay quality and exploration. Experiments on DeepMind Control Suite tasks show that NSPER and NSPER+R improve training efficiency and convergence speed compared to existing methods in image-based RL.

Keywords—Reinforcement learning, prioritized experience replay, sample efficiency, novelty, surprise, intrinsic rewards, continuous control, image-based control.

## I. INTRODUCTION

Reinforcement learning (RL) has shown great promise in image-based applications, allowing agents to learn directly from visual input [1], [2]. This paradigm has been widely applied in domains such as gaming [3], robotics [4], and autonomous driving [5], where vision serves as the primary source of environmental feedback. However, high-dimensional image data introduces redundancy, high memory demands, and substantial computational overhead, making policy learning more complex and resource-intensive [6]. Since RL inherently relies on large volumes of interaction data, improving sample efficiency is critical [7]. A promising direction is to prioritize informative experiences during training and to guide exploration toward those experiences that provide the most learning value. This prioritization can be guided by intrinsic learning signals that highlight which transitions are the most useful for learning [8].

Inspired by human cognition, where intrinsic signals foster curiosity [9], [10] and guide memory systems to prioritize novel or salient experiences [11], [12], RL agents can adopt similar mechanisms to enhance exploration and accelerate learning. Such signals provide a foundation for both directing exploratory behavior and selecting the most informative experiences during training [13], [14], thereby improving sample efficiency in two complementary ways. First, they can be formulated as intrinsic rewards, encouraging agents to acquire diverse and informative experiences that accelerate policy refinement [15]. Second, they can be used as prioritization scores within experience replay, enabling agents to focus on transitions with the highest learning potential.

Among intrinsic signals, novelty and surprise have proven especially effective as intrinsic rewards [16]. Novelty reflects the unfamiliarity of a state, driving exploration toward underrepresented or unexplored regions [17], [18]. Surprise, on the other hand, arises when outcomes deviate substantially from predictions, exposing gaps in the agent’s environmental model and its limited ability to generalize from past experiences [19]. Combining these two signals enables agents to identify regions that require further learning and adjust their exploration strategies, resulting in more efficient and targeted exploration [20], [21].

Prioritized Experience Replay (PER) is a well-established approach in RL that exploits such intrinsic signals within the replay buffer to improve sample efficiency [22]. In its original form, which we refer to as TD-PER, transitions are prioritized based on the temporal difference (TD) error, which measures the discrepancy between predicted and actual outcomes following an agent’s action. While effective in discrete action spaces, TD-PER struggles in continuous control environments [23], [24]. To overcome these limitations, several extensions have been proposed, including augmenting TD error with auxiliary signals or using biologically inspired mechanisms for prioritization [24]–[27]. Despite these advances, integrating intrinsic rewards with PER in image-based RL remains limited. For instance, the Contrastive Curiosity Learning Framework (CCLF) [27] uses curiosity to prioritize surprising augmented inputs. However, the broader use of intrinsic rewards as scoring signals for experience prioritization remains largely underexplored.

To address this gap, we propose NSPER, a novel pri oritization method that uses two distinct intrinsic signals, novelty and surprise, to guide experience selection within PER. By combining these signals to guide sampling, NSPER identifies diverse and informative transitions with high learning potential, improving the quality of replayed experiences. We further introduce NSPER+R, an extension of NSPER that integrates novelty and surprise not only for prioritization but also as intrinsic rewards. This dual mechanism improves experience selection and exploratory behavior, leading to faster convergence and more generalizable policies. To evaluate our approach, we integrate it into PixelTD3, an image-based variant of the TD3 algorithm [28], and assess its performance on challenging tasks from the DeepMind Control Suite [29], demonstrating promising improvements in complex, highdimensional visual environments. The model architecture is illustrated in Fig. 1. The implementation is publicly available at https://github.com/UoA-CARES/NSPER.

![](images/a270967e229790d34ffd4c28596db29fef08e9a8e1ea8cd3b775773b9c68a2b3.jpg)  
Fig. 1: Schematic overview of Novelty and Surprise Prioritized Experience Replay (NSPER).

The main contributions of this paper are summarized as follows:

1) We introduce NSPER, a novel approach that uses intrinsic rewards as prioritization signals within PER to improve sample efficiency in image-based RL.

2) We extend this idea in NSPER+R, which uses novelty and surprise both for prioritization and as intrinsic rewards, further enhancing exploration and policy learning.

3) We provide a comprehensive evaluation of different prioritization strategies and their integration with intrinsic rewards, analyzing their impact on training efficiency and learning performance.

4) We conduct an ablation study to isolate the individual contributions of novelty and surprise, as well as their combined effect when used with intrinsic rewards.

## II. RELATED WORKS

Experience prioritization and intrinsic motivation are key strategies for improving sample efficiency and exploration in RL. Prior work has explored various mechanisms for identifying informative experiences and guiding agents toward novel or surprising transitions, providing a foundation for methods that combine these ideas to enhance learning in complex environments.

## A. Experience Prioritization Methods

PER methods enhance sample efficiency by assigning higher sampling probability to transitions with greater learning potential [30]. TD-PER [22], a common variant, ranks transitions based on TD error, under the assumption that transitions with larger TD errors are less familiar and therefore more informative. However, in continuous high-dimensional spaces, small perturbations in actions or observations can lead to large fluctuations in value estimates, even for well-learned transitions, making TD error a noisy and unstable prioritization signal [23]. This sensitivity makes TD error a noisy and unreliable prioritization signal, motivating alternative strategies that offer more stable and informative learning signals in continuous control tasks [31], [32].

To address these limitations, LA3P [25] decouples learning signals by assigning low TD-error transitions to the actor to stabilize policy updates and high TD-error transitions to the critic to improve value estimation. Another method [30] prioritizes recent transitions with low TD error, thereby focusing on up-to-date experiences, but this risks catastrophic forgetting by discarding older yet valuable data. Model-Augmented PER (MaPER) [24] incorporates auxiliary signals derived from learned environmental dynamics to complement the TD error, thereby improving the identification of important experiences. Despite these enhancements, all of these methods primarily rely on the TD error as the primary prioritization criterion.

In contrast, Reward Prediction Error (RPE) prioritization [26] introduces an alternative signal inspired by biological learning. By measuring the discrepancy between expected and actual rewards, RPE provides informative feedback on its prediction accuracy, allowing it to adjust its policies and enhance decision-making over time [33]. Similarly, our method proposes a novel prioritization strategy that uses alternative intrinsic signals, fostering exploration and emphasizing unfamiliar transitions.

## B. Intrinsic Motivation Strategies

Intrinsic motivation has been widely explored in RL to encourage agents to explore and learn beyond extrinsic rewards [15]. This approach encourages agents to seek out novel states and behaviors within the environment [27], [34]. Several intrinsic motivation mechanisms have been proposed, including state novelty [16], state prediction errors [16], uncertainty about outcomes [35], and environment dynamics [36]. Among these, novelty and surprise enable more focused and efficient learning by directly targeting meaningful opportunities, guiding exploration toward unfamiliar states and significant

(3)

prediction errors, and outperforming methods that rely solely on uncertainty or environmental dynamics [21].

Novelty encourages visiting rarely encountered states, promoting a broader understanding of the environment [17]. Surprise is measured through prediction errors in environment dynamics, such as next-state errors and state transition divergences [13], [19]. Combining novelty and surprise captures complementary aspects of agent-environment interaction, improving exploration and representation learning [17], [18] and allowing better balance between exploration and exploitation in complex tasks [20].

Despite these advances, integrating intrinsic motivation signals into memory prioritization mechanisms remains an ongoing challenge. Several recent studies have focused on combining curiosity-based metrics with experience replay to further refine prioritization strategies.

## C. Integrating Intrinsic Signals in Experience Replay

Recent studies have explored the incorporation of intrinsic motivation into memory prioritization. Curiosity-driven experience prioritization (CDP) [37] prioritizes trajectories that lead to rare goal states, ensuring the agent focuses on underexplored areas. However, CDP is based on the replay of hindsight experiences (HER) [38], which limits its applicability to goal-oriented tasks. Augmented Curiosity-Driven Experience Replay (ACDER) [39] also integrates curiosity signals into HER but remains restricted to goal-conditioned exploration.

A more relevant approach, CCLF [27], introduces a selfsupervised prioritization mechanism in image-based RL. CCLF selects surprising augmented inputs to enhance sample efficiency, but it focuses primarily on representation learning rather than on optimizing experience selection for policy improvement.

Our method extends these insights by integrating novelty and surprise into experience prioritization, ensuring that selected transitions support both exploration and policy optimization. By combining intrinsic rewards with a structured prioritization mechanism, it overcomes the limitations of PER and leverages intrinsic motivation to enable more effective learning in challenging environments

## III. METHODOLOGY

This section outlines the components underlying our approach, beginning with a review of PER, PixelTD3, and the computation of novelty and surprise signals. We then introduce our framework, NSPER, and its extension, NSPER+R.

## A. PER

PER improves sample efficiency by replaying transitions with higher learning potential. Transitions in the buffer B are stored as:

$$
B _ { i } = ( s _ { i } , a _ { i } , r _ { i } , s _ { i + 1 } ) ,\tag{1}
$$

where i indexes the time step. The priority of each transition is defined as:

$$
\sigma _ { i } = | \delta _ { i } | + \epsilon ,\tag{2}
$$

In the standard variant, TD-PER, $\delta _ { i }$ is the TD error, and $\epsilon > 0$ ensures non-zero priority.

Transitions are sampled according to:

$$
p _ { i } = \frac { \sigma _ { i } ^ { \alpha } } { \sum _ { k } \sigma _ { k } ^ { \alpha } }
$$

$$
\begin{array} { r } { w _ { i } = \left( \frac { 1 } { \left| B \right| p _ { i } } \right) ^ { \beta } } \end{array}\tag{4}
$$

where $p _ { i }$ favors transitions with higher priority $\sigma _ { i }$ , and $w _ { i }$ corrects for the bias introduced by non-uniform sampling. Here, α controls prioritization sharpness, and $\beta$ compensates for sampling bias. While TD-PER defines $\sigma _ { i }$ using TD error, PER is flexible and allows alternative prioritization signals by redefining $\sigma _ { i }$

## B. PixelTD3

PixelTD3 is an image-based extension of TD3 [28], inspired by SACAE [40] and extended in NaSATD3 [21]. PixelTD3 employs a convolutional autoencoder that maps raw observations $s _ { t }$ into a compact latent representation z. This latent vector is shared between the actor and critic networks, ensuring that policy learning is grounded in a consistent feature space. The encoder–decoder is jointly trained with a reconstruction loss that preserves the structural content of the input images while maintaining task-relevant features.

Unlike SACAE, which trains separate encoders for actor and critic, PixelTD3 uses a single shared autoencoder that is updated end-to-end at every step, including both convolutional and linear layers. This provides a stable and compact representation for policy learning. We adopt this setup as the backbone of NSPER (see Fig. 1). Complete architectural specifications of the encoder, decoder, actor, critics, and predictive ensemble are reported in Table III.

## C. Novelty And Surprise Computation

Novelty measures how different an observation is from what the agent has already encoded. It is estimated by reconstructing the input $s _ { t }$ through the autoencoder and comparing it to the original observation. Let $\hat { s } _ { t } = D e c ( E n c ( s _ { t } ) )$ denote the reconstruction of $s _ { t }$ . The novelty score is then defined as the complement of the Structural Similarity Index (SSIM) between $s _ { t }$ and $\hat { s } _ { t } \colon$

$$
\mathrm { N o v e l t y } ( s _ { t } ) = 1 - \mathrm { S S I M } ( s _ { t } , \hat { s } _ { t } ) .\tag{5}
$$

A higher value indicates that the observation is less wellreconstructed, and therefore more novel to the agent. Fig. 1 illustrates how novelty is obtained from the autoencoder reconstruction error of the input image.

Surprise measures the deviation between the agent’s predicted and observed dynamics. It is computed in latent space using an ensemble of dynamics models $P _ { \rho } { } ,$ which predict the next latent state $\hat { z } _ { t + 1 }$ given the current latent representation $z _ { t }$ and action $a _ { t } \colon$

$$
\hat { z } _ { t + 1 } = P _ { \rho } ( z _ { t } , a _ { t } ) .\tag{6}
$$

Surprise is defined as the mean squared error between the predicted and observed latent states:

$$
\mathrm { S u r p r i s e } ( s _ { t } , a _ { t } ) = \mathbf { M S E } ( z _ { t + 1 } , \hat { z } _ { t + 1 } ) .\tag{7}
$$

As illustrated in Fig. 1, this signal corresponds to the mismatch between the latent representation of the observed next state and the ensemble prediction. The ensemble consists of M models trained with mean squared error loss, and averaging their predictions reduces variance and improves stability under non-stationary encoder updates.

Novelty and Surprise Combination: Combining novelty and surprise reward signals encourages more targeted exploration of informative regions within the environment. This integration forms a new intrinsic signal, referred to as the Novelty-Surprise Signal (NSS). The $\mathrm { N S S } ( s , a )$ , derived from (5) and (7), captures the joint contribution of novelty and surprise and is formulated as:

$$
\mathrm { N S S } ( s _ { t } , a _ { t } ) = \mathrm { N o v e l t y } ( s _ { t } ) + \mathrm { S u r p r i s e } ( s _ { t } , a _ { t } )\tag{8}
$$

## D. NSPER: Novelty and Surprise Prioritization

We propose a prioritization strategy that leverages novelty and surprise as intrinsic signals to guide experience replay. Rather than relying on the TD error, as in standard PER, NSPER replaces $\delta _ { i }$ in (2) with the Novelty–Surprise Signal $\mathrm { N S S } ( s , a )$ from (8). The priority assigned to each transition is therefore defined as:

$$
\sigma _ { t } = | \mathrm { N S S } ( s _ { t } , a _ { t } ) | + \epsilon ,\tag{9}
$$

where $\epsilon > 0$ ensures all transitions remain eligible for replay.

These priorities are then used in the sampling equations (3) and (4) to compute transition probabilities and importance weights. By emphasizing transitions with higher novelty and surprise, NSPER favors diverse and informative experiences that better support critic training. The parameter α regulates the strength of this prioritization, ensuring that informative transitions are emphasized without over-concentrating sampling on a few extreme cases. This balance allows NSPER to prioritize informative experiences while maintaining diversity in the sampled transitions.

## E. Enhancing Exploration with Intrinsic Rewards

Intrinsic rewards are agent-generated signals that promote exploration, especially when extrinsic rewards are sparse or delayed [8]. In our approach, we define the intrinsic reward using novelty and surprise (8) as:

$$
r _ { i n t } ( s _ { t } , a _ { t } ) = N S S ( s _ { t } , a _ { t } )\tag{10}
$$

As depicted in Fig. 1, the NSS is used both to prioritize experiences during replay and to define the intrinsic reward signal. The total reward at each step combines extrinsic and intrinsic components:

$$
r _ { t } = r _ { e x t } + r _ { i n t }\tag{11}
$$

We refer to the full version of our framework as NSPER+R, which integrates intrinsic rewards with prioritized experience replay. This variant improves learning by prioritizing transitions that contribute to both task success and deeper environmental understanding. Algorithm 1 summarises the full procedure for NSPER and its intrinsic-reward variant NSPER+R.

Algorithm 1 NSPER and NSPER+R with NSS (Nov  
elty–Surprise Signal)   
1: Inputs: PER exponents $\alpha , \beta ,$ batch size m, small $\varepsilon > 0$   
2: Initialize: actor Θ, critic θ, encoder–decoder (Enc, Dec), replay   
buffer B, priorities {σ<sub>i</sub>}   
3: for each environment step t do   
4: Observe state $s _ { t } ;$ select action $a _ { t }$ (with exploration noise);   
receive $r _ { \mathrm { e x t } }$ and $s _ { t + 1 }$   
5: Novelty: Novelty $\mathbf { \sigma } ^ { \cdot } ( s _ { t } ) \gets 1 - \mathrm { S S I M } ( s _ { t } ,$ Dec(Enc(s<sub>t</sub>))   
6: Surprise: obtain latents $( z _ { t + 1 } , z _ { t + 1 } ^ { \prime } )$ and set   
Surprise $( s _ { t } , a _ { t } ) \gets \mathrm { M S E } ( z _ { t + 1 } ^ { \prime } , z _ { t + 1 } )$   
7: NSS: $\mathrm { N S S } ( \dot { s } _ { t } , a _ { t } ) \gets \mathrm { l }$ Novelty $( s _ { t } ) + \mathrm { S u r p r i s e } ( s _ { t } , a _ { t } )$   
8: Intrinsic reward: $r _ { \mathrm { i n t } } \gets \mathrm { N S S } ( s _ { t } , a _ { t } )$   
$r _ { t } \gets \left\{ { r _ { \mathrm { e x t } } + r _ { \mathrm { i n t } } } , \right.$ NSPER+R   
9: Total reward:   
NSPER   
10: Push $( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } , \mathrm { \tilde { N S S } } ( s _ { t } , a _ { t } ) )$ into B   
11: Set priority for new item: $\dot { \sigma } _ { \mathrm { n e w } } \longleftarrow \vert \mathrm { N S S } ( s _ { t } , a _ { t } ) \vert + \varepsilon$   
12: for each update step do   
13: Sample indices I of size m with PER:   
$p ( i ) = \frac { \sigma _ { i } ^ { \alpha } } { \sum _ { k } \sigma _ { k } ^ { \alpha } }$   
14: IS weights: $\begin{array} { r } { w ( i ) \gets \left( \frac { 1 } { | \boldsymbol { B } | \cdot p ( i ) } \right) ^ { \beta } } \end{array}$ (optionally normalize   
by $\operatorname* { m a x } _ { j \in I } w ( j ) )$   
15: Update critic/actor: compute losses on batch, weight by   
$w ( i )$ , update θ, Θ   
16: Refresh priorities: for each $\begin{array} { r l r } { i } & { { } \in } & { I , } \end{array}$ set $\sigma _ { i } \gets$   
$| \mathrm { N S S } ( s _ { i } , a _ { i } ) | + \varepsilon$   
17: end for   
18: end for

## IV. EXPERIMENTS

This section describes the experimental setup, simulation tasks, algorithmic settings, and baseline methods used in our evaluation, and then presents an analysis of the learning performance of NSPER, NSPER+R, and competing approaches.

## A. Experimental Setup

We evaluate our algorithms and baselines on five imagebased continuous control tasks from the DeepMind Control Suite (DMC) [29]: Cartpole-Balance, Finger-Spin, Ball-in-Cup, Walker-Walk, and Cheetah-Run. These environments constitute the standard benchmark suite for visual reinforcement learning and have been widely used in prior pixelbased studies [21], [41], [42]. This set provides representative coverage of balance, manipulation, and locomotion tasks and offers sufficient diversity for assessing prioritization strategies in image-based continuous control. Visual examples of the environments are given in Fig. 2, and their key properties are summarised in Table I.

All algorithms are trained under a unified experimental configuration following established practice in continuouscontrol evaluation. Training settings, optimization parameters, evaluation schedules, and replay specifications are listed in Table II. Using identical configurations across all tasks ensures that performance differences arise from the prioritization mechanism rather than from hyperparameter choices or implementation bias. NSPER and NSPER+R employ a shared latentspace architecture for representation learning, prediction, and action-value estimation. The components of this architecture are outlined in Table III.

![](images/086ac4575c339042f0ff3774dad9ab98d1ac8d851e2bc3003f8029cedc02e289.jpg)  
(a) Cartpole-Balance

![](images/9094188c0b7724bffa67ce2e560eb791ba939a03ed77337d0b7efee275e95aad.jpg)  
(b) Finger-Spin

![](images/e72cbed73cac46fde23411dc96363ba7c4b4b7d5539988ee66f6d3f47b0061f7.jpg)  
(c) Ball-in-Cup

![](images/54012025f5cf4f30232e88b9f8635c18b412e29bc37c38635860cea881e97abe.jpg)  
(d) Walker-Walk

![](images/656376182ba1415f0d0fcb27a560b65e15ce787149b03ea2443c0fc45e5b8bd0.jpg)  
(e) Cheetah-Run  
Fig. 2: Five continuous control tasks from the DeepMind Control Suite used in our evaluation.

TABLE I: Summary of DeepMind Control Suite tasks used in our experiments.
<table><tr><td>Task</td><td>Action Space</td><td>Objective</td><td>Primary Challenge</td><td>Type</td><td>Difficulty</td></tr><tr><td>Cartpole-Balance</td><td>[−1,1]1</td><td>Stabilize the pole by moving the cart horizontally.</td><td>Underactuated and highly unstable dynamics where small errors rapidly lead to failure.</td><td>Classic Control</td><td>Low</td></tr><tr><td>Finger-Spin</td><td>[−1,1]2</td><td>Maintain continuous rotation of a free- floating object.</td><td>Requires fine torque modulation to sustain spin without drift or loss of contact.</td><td>Manipulation</td><td>High</td></tr><tr><td>Ball-in-Cup</td><td>[−1,1]2</td><td>Swing the ball and capture it inside the cup.</td><td>Oscillatory motion requires precise timing and well-coordinated control.</td><td>Manipulation</td><td>Medium-High</td></tr><tr><td>Walker-Walk</td><td>[−1,1]6</td><td>Achieve stable forward locomotion.</td><td>Balancing and coordinating a bipedal body in an inherently unstable locomotion regime.</td><td>Locomotion</td><td>Medium</td></tr><tr><td>Cheetah-Run</td><td> $[ - 1 , 1 ] ^ { 6 }$ </td><td>Run forward at high speed.</td><td>High velocity amplifies instability, requiring strong, consistent, and fast control.</td><td>Locomotion</td><td>Medium</td></tr></table>

TABLE II: Experimental setup for NSPER across DeepMind Control Suite environments.
<table><tr><td>Parameter</td><td>Value (Description)</td></tr><tr><td></td><td>Environments</td></tr><tr><td>Observation 84  $\times 8 4 \times 3$ </td><td>RGB images, with 3 consecutive frames stacked</td></tr><tr><td>Reward Scale</td><td>Maximum episodic return of 1000 per task</td></tr><tr><td>Evaluation Metric</td><td>Average cumulative reward per step (computed using extrinsic rewards)</td></tr><tr><td colspan="2">Implementation Details</td></tr><tr><td>Replay Buffer Capacity</td><td>1,000,000 transitions</td></tr><tr><td>Batch Size</td><td>128 (Mini-batch sampled from replay buffer)</td></tr><tr><td>Actor Learning Rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Critic Learning Rate</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Prioritization Èxponent α</td><td>0.7 (Controls prioritization sharpness)</td></tr><tr><td>Importance-Sampling Exponent β</td><td>0.4 (Corrects sampling bias)</td></tr><tr><td>Training Steps</td><td> $1 \times 1 0 ^ { 6 }$  (Total environment steps)</td></tr><tr><td>Evaluation Frequency Evaluation Episodes</td><td> $\mathrm { E v e r y ~ 1 } \times 1 0 ^ { 4 }$  steps</td></tr><tr><td>Random Seeds</td><td>10 (Episodes per evaluation checkpoint)</td></tr><tr><td></td><td>5 (Independent runs with different seeds for environment and initialization)</td></tr></table>

TABLE III: Network architectures used in NSPER and NSPER+R.
<table><tr><td>Component</td><td>Input</td><td>Layers / Units</td><td>Activations</td><td>Output</td></tr><tr><td>Encoder</td><td>84 × 84 × 3 image</td><td>4 conv  $( 3 2 , 3 \times 3 ) + \mathrm { F C } ( 2 0 0 )$ </td><td>ReLU (conv), tanh (FC)</td><td>Latent  $z \in \mathbb { R } ^ { 2 0 0 }$ </td></tr><tr><td>Decoder</td><td>Latent z</td><td>Deconv mirror of encoder</td><td>ReLU, Sigmoid (final)</td><td>Image reconstruction</td></tr><tr><td>Actor (TD3)</td><td> $z \ ( \mathrm { o p t . \ + v e c t o r \ o b s } )$ </td><td>FC(1024) → FC(1024)</td><td>ReLU, tanh (out)</td><td>Action  $\in [ - 1 , 1 ] ^ { d }$ </td></tr><tr><td>Critics (TD3,  $\times 2 )$ </td><td> $z \| a \ ( \mathrm { o p t . + v e c t o r \ o b s } )$ </td><td>FC(1024) → FC(1024)</td><td>ReLU</td><td>Scalar  $Q \mathrm { - v a l u e }$ </td></tr><tr><td>Predictive Ensemble</td><td> $z _ { t } \| a _ { t }$ </td><td> $\mathrm { F C } ( 5 1 2 )  \mathrm { F C } ( 5 1 2 )$ </td><td>ReLU</td><td>Pred. latent  $\hat { z } _ { t + 1 }$ </td></tr></table>

## B. Experimental Approach

To evaluate the effectiveness of our approach, we consider two proposed variants, NSPER and NSPER+R:

NSPER: Uses novelty and surprise solely for replay prioritization, isolating their effect on experience selection.

NSPER+R: Extends NSPER by also incorporating novelty and surprise as intrinsic rewards, combined with extrinsic rewards (11), allowing us to assess their joint impact on prioritization and exploration.

Baseline Algorithms: For comparison, we evaluate a range of established methods within the PixelTD3 framework, which differ in their replay strategies and use of intrinsic rewards. All implementations share the same codebase for fairness. Although CCLF was originally proposed with a pixel-based SAC backbone, we adapt it to PixelTD3 for consistency. The baselines include:

1) Uniform: Uniformly samples transitions from the buffer

![](images/71e95652f1b95b667f73de0835cfc12a7bb312fd4f35ca4866bb6a2bf411956f.jpg)  
(a)

![](images/e1f9e88406ef4b72e3027db568f5abbb803177341cb206c61174e31d94fab819.jpg)

![](images/9596593fb0f289447f5745302c045d5032f0fcc593fcd28bd5ab1cc933d081bd.jpg)  
(c)

(b)  
![](images/cde0a012b0c47953a011ce161916fb339cd12a3d9bfddf0b82f7c8c2748943b3.jpg)  
(d)

![](images/c8cce8aa2a518ce7f71a56d2ed1e28d2ad8cfabedbe58533f11fee6c5d9eba9b.jpg)  
(e)  
Fig. 3: Learning curves for continuous control tasks using PixelTD3. Solid lines represent the algorithm’s performance with intrinsic rewards (+R), while dashed lines indicate performance without intrinsic rewards. Shaded regions denote the 95% confidence interval across 5 runs.

without PER or intrinsic rewards [28], [43].

2) Uniform+R (NaSATD3): Uniform sampling with intrinsic rewards, equivalent to PixelTD3+R [21].

3) TD-PER: PER based on TD error [22], without intrinsic rewards.

4) TD-PER+R: TD-PER augmented with intrinsic rewards.

5) RPE-PER: PER based on reward prediction error (RPE) [26], without intrinsic rewards.

6) RPE-PER+R: RPE-PER extended with intrinsic rewards.

7) CCLF: A contrastive learning-based prioritization method with intrinsic rewards to improve exploration and representation learning [27].

This setup enables a comprehensive evaluation of NSPER and NSPER+R against diverse baselines that vary in their use of prioritization strategies and intrinsic rewards, providing a robust comparison in image-based RL.

## C. Learning Efficiency Analysis

The learning curves in Fig. 3 illustrate the learning efficiency of NSPER and baseline methods across five challenging DeepMind continuous control environments. These curves depict the average evaluation reward over training steps, providing insights into convergence speed and performance differences among NSPER, NSPER+R, and the baseline algorithms.

Dashed-line plots represent PixelTD3 with different replay buffer sampling methods but without intrinsic rewards, while solid-line plots of the same color correspond to the same sampling methods with intrinsic rewards. These separate plots directly compare how sampling strategies influence performance with and without intrinsic rewards.

It is crucial to note that Baseline methods did not always replicate the performance reported in prior work. Such deviations are not uncommon in deep reinforcement learning and can arise from factors such as environmental stochasticity, initialization randomness, and subtle implementation or runtime differences. As emphasized by Henderson et al. [44], ensuring fair comparison is essential; in our case, the relative ranking of methods remained consistent across multiple runs, supporting the robustness of our evaluations.

## V. RESULTS AND DISCUSSION

Performance results across various tasks in Fig. 3 show that NSPER (red plot) results are better than baseline algorithms. It enhances PixelTD3’s performance on four of five tasks while achieving comparable results on the fifth task to the bestperforming methods among image-based RL approaches. Our evaluation highlights the crucial role of novelty and surprise signals in improving learning efficiency. In contrast, Uniform sampling (orange plot) exhibits suboptimal performance across all tasks, demonstrating that relying solely on a uniform experience buffer is insufficient. These findings suggest that PixelTD3 could significantly benefit from integrating prioritized sampling or intrinsic reward mechanisms, even in relatively simple environments.

Incorporating prioritization strategies alongside intrinsic rewards improves performance across nearly all algorithms, highlighting the advantages of these techniques in fostering more robust and adaptive learning. In particular, NSPER+R achieves the best results among the methods. While CCLF (light green plot) consistently outperforms Uniform across all tasks, its overall performance remains unremarkable. This suggests that CCLF’s prioritization strategy differs fundamentally from that of a prioritization buffer, and, as the results indicate, it does not integrate well with PixelTD3.

In addition, the comparable performance of the RPE-PER algorithm (blue plot) to NSPER across all tasks demonstrates its potential as an effective prioritization method. However, NSPER consistently achieves better results. These findings indicate that using a complex prioritization scale in imagebased RL can be beneficial, while also reaffirming that novelty and surprise signals are stronger heuristics for selecting informative experiences, ultimately contributing to a more sampleefficient learning process.

## A. Task-Specific Analysis

This section examines the performance of different methods across individual tasks, highlighting their strengths and limitations in each environment.

In Cartpole-Balance (Fig. 3, plot a), a classic control task requiring a cart to balance a pole, all methods using either a prioritized buffer or intrinsic rewards perform well. This suggests that PixelTD3 benefits from PER or intrinsic rewards, even in simpler environments.

The Finger-Spin task, which involves manipulating an object with a simulated finger, presents challenges due to complex contact dynamics and sparse rewards. As shown in Fig. 3, plot b, NSPER variants outperform other methods, emphasizing the importance of novelty and surprise signals for learning efficiency. NSPER+R, which integrates intrinsic rewards with PER, further enhances performance. Comparing TD-PER (prioritization based on TD error) with Uniform+R (intrinsic rewards without prioritization) reveals that intrinsic rewards yield better results in this environment than PER alone. However, TD-PER+R underperforms relative to TD-PER, suggesting that conflicting signals from the TD error and novelty-surprise signals can hinder efficiency. In contrast, NSPER+R aligns both prioritization and intrinsic rewards using the same features, thereby improving sample efficiency.

The Ball-in-Cup task, characterized by sparse rewards granted only upon successfully catching the ball, provides another challenging test. In (Fig. 3, plot c), NSPER+R outperforms other approaches, demonstrating the benefits of combining novelty and surprise for prioritization with intrinsic rewards. While Uniform+R performs well, NSPER+R and RPE-PER+R achieve consistently better results, reinforcing that properly aligning prioritization with intrinsic rewards enhances performance.

The Walker-Walk environment, which requires controlling a bipedal walker to maintain dynamic balance, presents a complex, multi-stage control challenge despite offering dense rewards. (Fig. 3, plot d) shows that NSPER+R and RPE-PER+R improve performance, highlighting the importance of context-aware experience selection. In contrast, TD errorbased PER leads to suboptimal results even when combined with intrinsic rewards. This suggests that prioritizing solely based on high TD error can misguide learning, as such experiences may not always be the most informative.

In Cheetah-Run (Fig. 3, plot e), TD-PER performs best, but NSPER closely follows, demonstrating its adaptability to high-speed locomotion tasks. This result indicates that prioritization strategies should be tailored to specific environments, as more straightforward prioritization methods can sometimes be more effective. However, TD-PER exhibits inconsistency, performing well in Cheetah-Run but poorly in Walker-Walk, as noted in prior studies [23]. In contrast, NSPER maintains stable performance across diverse tasks, suggesting it provides a more reliable prioritization strategy.

## B. Ablation Study: The Role of Novelty and Surprise in Prioritization

This study evaluates various prioritization strategies to isolate the individual and combined effects of novelty and surprise. NoveltyPER prioritizes experiences based solely on novelty, SurprisePER based solely on surprise, and NSPER jointly leverages both signals. Solid curves denote variants that additionally use intrinsic rewards, while dashed curves show performance without them. As shown in Fig. 4, prioritization based on both novelty and surprise outperforms using either signal alone. This combination improves experience selection and, when also incorporated as intrinsic rewards, encourages more effective exploration. The results highlight the complementary nature of novelty and surprise in RL. Furthermore, integrating these signals as intrinsic rewards yields additional performance gains by promoting a more balanced selection of experiences, ultimately enhancing policy learning.

In the Cheetah-Run task, prioritization alone is sufficient, and incorporating surprise as an intrinsic reward negatively impacts performance. Another key observation is that NoveltyPER outperforms SurprisePER in most tasks, suggesting that novelty-based prioritization, which encourages exploration of diverse states, is more beneficial than surprise-based prioritization, which focuses on unexpected transitions. Additionally, the performance of SurprisePER and SurprisePER+R remains similar in most cases, indicating that adding surprise as an intrinsic reward does not significantly contribute to performance improvements. However, when both novelty and surprise are used in prioritization and intrinsic rewards, the agent achieves the best overall performance, demonstrating their complementary roles in improving learning efficiency.

## VI. CONCLUSION

In this work, we addressed the challenge of sample inefficiency in image-based RL by introducing NSPER and its extension NSPER+R, which integrate novelty and surprise as signals for replay prioritization and, in the case of NSPER+R, also as intrinsic rewards. This dual mechanism improves both replay quality and exploration, leading to faster convergence and more generalizable policies. Through an extensive evaluation of the DeepMind Control Suite using the

NSPER+R NSPER NoveltyPER+R NoveltyPER SurprisePER+R SurprisePER

![](images/83ad90f5106c03f0ec1bf950570eb1e301e2c384cb3d71e4bbfc63eb7fe21038.jpg)  
(a)

![](images/3076078e07429414a5a086f3717af18d497dea04b307ce97f1c7704ebd61e6e2.jpg)

![](images/70ffee88e677bbf4de43f0b703c46b584cb27597fbdd19ddefbe59763e40094e.jpg)  
(c)

(b)  
![](images/2f09e26a917234273f2be59df0e7f1967512cacdf0e4d3df6a9966b78a52d643.jpg)  
(d)

![](images/7d77fa22e2c812e80de659815e1f445578d11fdce80b837c059db7ad8cf72a6a.jpg)  
(e)  
Fig. 4: Ablation results comparing experience prioritization based on novelty, surprise, and their combination, with and without intrinsic rewards. Shaded regions represent the 95% confidence interval over five runs.

PixelTD3 backbone, we showed that NSPER and NSPER+R outperform existing baselines, improving training efficiency in complex image-based tasks. Ablation studies further confirmed the complementary contributions of novelty and surprise to prioritization and exploration. Our research highlights the importance of using internal stimuli and enhanced sampling buffers to improve RL systems.

Future work should explore additional intrinsic motivators and adaptively weight novelty and surprise. Our ablation study indicates that these signals impact learning efficiency differently, suggesting that dynamically adjusting their balance could refine the agent’s exploration strategy.

## REFERENCES

[1] J. Zheng and Y. Song, “Effective representation learning is more effective in reinforcement learning than you think,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 9176–9182.

[2] Y. Huang, P. Peng, Y. Zhao, Y. Zhai, H. Xu, and Y. Tian, “Simoun: Synergizing interactive motion-appearance understanding for vision-based reinforcement learning,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 176–185.

[3] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, “Asynchronous methods for deep reinforcement learning,” in International conference on machine learning. PMLR, 2016, pp. 1928–1937.

[4] S. Ferraro, T. Van de Maele, P. Mazzaglia, T. Verbelen, and B. Dhoedt, “Computational optimization of image-based reinforcement learning for robotics,” Sensors, vol. 22, no. 19, p. 7382, 2022.

[5] A. Scorsoglio, R. Furfaro, R. Linares, and B. Gaudet, “Image-based deep reinforcement learning for autonomous lunar landing,” in AIAA Scitech 2020 Forum, 2020, p. 1910.

[6] L. Chen, K. Lee, A. Srinivas, and P. Abbeel, “Improving computational efficiency in visual reinforcement learning via stored embeddings,” Advances in Neural Information Processing Systems, vol. 34, pp. 26 779– 26 791, 2021.

[7] F. E. Dorner, “Measuring progress in deep reinforcement learning sample efficiency,” arXiv preprint arXiv:2102.04881, 2021.

[8] A. G. Barto, “Intrinsic motivation and reinforcement learning,” Intrinsically motivated learning in natural and artificial systems, pp. 17–47, 2013.

[9] G. Baldassarre, “What are intrinsic motivations? a biological perspective,” in 2011 IEEE international conference on development and learning (ICDL), vol. 2. IEEE, 2011, pp. 1–8.

[10] E. L. Deci and R. M. Ryan, Intrinsic motivation and self-determination in human behavior. Springer Science & Business Media, 2013.

[11] D. Shohamy and R. A. Adcock, “Dopamine and adaptive memory,” Trends in cognitive sciences, vol. 14, no. 10, pp. 464–472, 2010.

[12] E. K. Braun, G. E. Wimmer, and D. Shohamy, “Retroactive and graded prioritization of memory by reward,” Nature communications, vol. 9, no. 1, p. 4886, 2018.

[13] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell, “Curiosity-driven exploration by self-supervised prediction,” in International conference on machine learning. PMLR, 2017, pp. 2778–2787.

[14] R. Houthooft, X. Chen, Y. Duan, J. Schulman, F. De Turck, and P. Abbeel, “Vime: Variational information maximizing exploration,” Advances in neural information processing systems, vol. 29, 2016.

[15] P.-Y. Oudeyer and F. Kaplan, “What is intrinsic motivation? a typology of computational approaches,” Frontiers in neurorobotics, vol. 1, p. 6, 2007.

[16] M. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos, “Unifying count-based exploration and intrinsic motivation,” Advances in neural information processing systems, vol. 29, 2016.

[17] A. Jaegle, V. Mehrpour, and N. Rust, “Visual novelty, curiosity, and intrinsic reward in machine learning and the brain,” Current opinion in neurobiology, vol. 58, pp. 167–174, 2019.

[18] M. Kubovcˇ´ık, I. Dirgova Lupt ´ akov ´ a, and J. Posp ´ ´ıchal, “Signal novelty detection as an intrinsic reward for robotics,” Sensors, vol. 23, no. 8, p. 3985, 2023.

[19] J. Achiam and S. Sastry, “Surprise-based intrinsic motivation for deep reinforcement learning,” arXiv preprint arXiv:1703.01732, 2017.

[20] H. Le, K. Do, D. Nguyen, and S. Venkatesh, “Beyond surprise: Improving exploration through surprise novelty.” in AAMAS, 2024, pp. 1084– 1092.

[21] D. Valencia, H. Williams, Y. Xing, T. Gee, M. Liarokapis, and B. A. MacDonald, “Image-based deep reinforcement learning with intrinsi-

cally motivated stimuli: On the execution of complex robotic tasks,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 587–594.

[22] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized experience replay,” arXiv preprint arXiv:1511.05952, 2015.

[23] Y. Pan, J. Mei, A.-m. Farahmand, M. White, H. Yao, M. Rohani, and J. Luo, “Understanding and mitigating the limitations of prioritized experience replay,” in Uncertainty in Artificial Intelligence. PMLR, 2022, pp. 1561–1571.

[24] Y. Oh, J. Shin, E. Yang, and S. J. Hwang, “Model-augmented prioritized experience replay,” in International Conference on Learning Representations, 2021.

[25] B. Saglam, F. B. Mutlu, D. C. Cicek, and S. S. Kozat, “Actor prioritized experience replay,” arXiv preprint arXiv:2209.00532, 2022.

[26] H. Yamani, Y. Xing, L. V. C. Ong, B. A. MacDonald, and H. Williams, “Reward prediction error prioritisation in experience replay: The rpe-per method,” arXiv preprint arXiv:2501.18093, 2025.

[27] C. Sun, H. Qian, and C. Miao, “Cclf: A contrastive-curiosity-driven learning framework for sample-efficient reinforcement learning,” arXiv preprint arXiv:2205.00943, 2022.

[28] S. Fujimoto, H. Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in International conference on machine learning. PMLR, 2018, pp. 1587–1596.

[29] Y. Tassa, Y. Doron, A. Muldal, T. Erez, Y. Li, D. d. L. Casas, D. Budden, A. Abdolmaleki, J. Merel, A. Lefrancq et al., “Deepmind control suite,” arXiv preprint arXiv:1801.00690, 2018.

[30] D. Zha, K.-H. Lai, K. Zhou, and X. Hu, “Experience replay optimization,” arXiv preprint arXiv:1906.08387, 2019.

[31] R. Carrasco-Davis, S. Lee, C. Clopath, and W. Dabney, “Uncertainty prioritized experience replay,” arXiv preprint arXiv:2506.09270, 2025.

[32] X.-H. Liu, Z. Xue, J. Pang, S. Jiang, F. Xu, and Y. Yu, “Regret minimization experience replay in off-policy reinforcement learning,” Advances in neural information processing systems, vol. 34, pp. 17 604– 17 615, 2021.

[33] R. Simmons-Edler, B. Eisner, D. Yang, A. Bisulco, E. Mitchell, S. Seung, and D. Lee, “Reward prediction error as an exploration objective in deep rl,” arXiv preprint arXiv:1906.08189, 2019.

[34] A. Aubret, L. Matignon, and S. Hassas, “A survey on intrinsic motivation in reinforcement learning,” arXiv preprint arXiv:1908.06976, 2019.

[35] K. Li, A. Gupta, A. Reddy, V. H. Pong, A. Zhou, J. Yu, and S. Levine, “Mural: Meta-learning uncertainty-aware rewards for outcome-driven reinforcement learning,” in International conference on machine learning. PMLR, 2021, pp. 6346–6356.

[36] Y. Seo, L. Chen, J. Shin, H. Lee, P. Abbeel, and K. Lee, “State entropy maximization with random encoders for efficient exploration,” in International Conference on Machine Learning. PMLR, 2021, pp. 9443–9454.

[37] R. Zhao and V. Tresp, “Curiosity-driven experience prioritization via density estimation,” arXiv preprint arXiv:1902.08039, 2019.

[38] M. Andrychowicz, F. Wolski, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, O. Pieter Abbeel, and W. Zaremba, “Hindsight experience replay,” Advances in neural information processing systems, vol. 30, 2017.

[39] B. Li, T. Lu, J. Li, N. Lu, Y. Cai, and S. Wang, “Acder: Augmented curiosity-driven experience replay,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 4218–4224.

[40] D. Yarats, A. Zhang, I. Kostrikov, B. Amos, J. Pineau, and R. Fergus, “Improving sample efficiency in model-free reinforcement learning from images,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 12, 2021, pp. 10 674–10 681.

[41] I. Kostrikov, D. Yarats, and R. Fergus, “Image augmentation is all you need: Regularizing deep reinforcement learning from pixels,” arXiv preprint arXiv:2004.13649, 2020.

[42] M. Laskin, A. Srinivas, and P. Abbeel, “Curl: Contrastive unsupervised representations for reinforcement learning,” in International conference on machine learning. PMLR, 2020, pp. 5639–5650.

[43] W. Fedus, P. Ramachandran, R. Agarwal, Y. Bengio, H. Larochelle, M. Rowland, and W. Dabney, “Revisiting fundamentals of experience replay,” in International conference on machine learning. PMLR, 2020, pp. 3061–3071.

[44] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, “Deep reinforcement learning that matters,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.
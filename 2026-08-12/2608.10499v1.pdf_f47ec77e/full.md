# Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation

Md Rafid Islam

Department of Electrical and Computer Engineering

North South University

Dhaka, Bangladesh

md.islam.241@northsouth.edu

Zahid Hasan

Department of Electrical and Computer Engineering

North South University

Dhaka, Bangladesh

zahid.hasan.241@northsouth.edu

Rafsan Jany

Digital Health Research Division

Korea Institute of Oriental Medicine

Daejeon, South Korea

rafsanjany@kiom.re.kr

Ratun Rahman Department of Electrical and Computer Engineering The University of Alabama in Huntsville Huntsville, AL, USA rr0110@uah.edu

Abstract—Personalized Federated Reinforcement Learning (PFRL) takes a decentralized approach to storing and accessing information based on past experiences while keeping each client’s data private during the learning of each client’s policy. Many current methods for PFRL rely heavily on exploiting existing reinforcement learning reward signals to derive an optimal policy for each client, thereby neglecting exploration in nonstationary or sparse-reward environments. In this work, we introduce a new exploration-driven framework, Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation (EDPFRL-IM), that leverages an inherent curiositydriven exploration at each client to promote local exploration and protect client privacy. Furthermore, to facilitate policy discovery via exploration in previously unexplored state spaces, clients add an intrinsic random network distillation (RND) signal to their extrinsic reward. Additionally, the server does not have access to clients’ raw experiences or local gradient estimates; instead, the server sends global exploration priors and collects minimal novelty summaries from each client to enable both diverse and coordinated exploration among clients. Experiments in benchmark environments show that our framework outperforms average PFRL benchmarks in policy personalization and sample efficiency, primarily in delayed and sparse reward systems. Overall, EDPFRL-IM enables the integration of a flexible exploratory learning structure into federated reinforcement learning systems while preserving client privacy.

Index Terms—federated reinforcement learning, reinforcement learning, intrinsic motivation

## I. INTRODUCTION

Reinforcement learning (RL) is a powerful framework for sequential decision-making that enables agents to learn optimal behaviors through interaction with their environment [1]. Federated reinforcement learning (FRL) enables multiple clients to cooperatively train reinforcement learning (RL) agents in decentralized settings without exchanging raw experience data. Personalized FRL (PFRL) expands on this by customizing policies to each client’s particular environment and objectives. This makes it suitable for real-world applications such as health monitoring and robotics [2], [3].

Although existing PFRL frameworks [2], [4] focus on policy optimization and aggregation, they often overlook the role of exploration under limited or delayed rewards. Standard RL strategies [1] like ϵ-greedy or entropy regularization become less effective in federated settings, where clients operate independently and face privacy constraints that limit coordination. As a result, clients may learn suboptimal policies, particularly in cold-start or non-stationary conditions. While approaches like FedRL [2], FedAvg-RL [4], and pFedMe [5] address personalization or aggregation, they assume homogeneous or supervised settings. Intrinsic motivation methods such as RND [6], ICM [7], and count-based exploration [8] show promise for centralized RL but remain underexplored in federated scenarios. Addressing this gap is key to unlocking more adaptive, intelligent on-device learning.

We propose the EDPFRL-IM framework, which integrates intrinsic curiosity-driven motivation into personalized federated reinforcement learning. It not only enables efficient local exploration with sparse rewards but also preserves privacy and minimizes communication. We summarize our key contributions as follows.

• We introduce EDPFRL-IM, a novel FRL framework that incorporates intrinsic motivation for personalized exploration in a decentralized setting.

• We develop a communication-efficient protocol for compressing exploration statistics, enabling clients to align their curiosity-driven behavior while avoiding disclosure of raw data or policies.

• We empirically evaluate our approach in sparse-reward benchmark environments and demonstrate significant improvements in personalization, exploration efficiency, and policy performance over existing federated RL baselines.

## II. METHODOLOGY

## A. Problem Formulation

We consider a PFRL setting with N clients, each interacting with a local Markov decision process (MDP) [9] $\begin{array} { r l } { \mathcal { M } _ { i } } & { { } = } \end{array}$ $( S _ { i } , \mathcal { A } , \mathcal { P } _ { i } , R _ { i } , \gamma )$ , where $s _ { i }$ and $R _ { i }$ are client-specific, while A and $\gamma$ are shared. Due to heterogeneity in transition dynamics and rewards, each client learns a personalized policy $\pi _ { i } ( a | s )$ to maximize its expected return:

$$
J _ { i } ( \pi _ { i } ) = \mathbb { E } _ { \pi _ { i } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { i , t } \right] .\tag{1}
$$

Clients update their policy parameters via gradient ascent:

$$
\theta _ { i }  \theta _ { i } + \eta \nabla _ { \theta _ { i } } J _ { i } ( \pi _ { \theta _ { i } } ) ,\tag{2}
$$

with learning rate η. Unlike global FL or RL, PFRL optimizes individual client policies while enabling collaboration through lightweight, privacy-preserving exploration statistics.

## B. Intrinsic Motivation for Local Exploration

Effective exploration is crucial in PFRL due to sparse or delayed rewards. We adopt RND, a self-supervised method in which each client uses a fixed random target network $f _ { \mathrm { t g t } }$ and a trainable predictor $f _ { \mathrm { p r e d } }$ . The intrinsic reward is the prediction error on the next state $s _ { t + 1 }$

$$
r _ { t } ^ { \mathrm { i n t } } = \left. f _ { \mathrm { t g t } } ( s _ { t + 1 } ) - f _ { \mathrm { p r e d } } ( s _ { t + 1 } ) \right. _ { 2 } ^ { 2 } .\tag{3}
$$

This promotes the exploration of novel states. The total reward combines extrinsic and intrinsic terms as

$$
\begin{array} { r } { \boldsymbol { r } _ { t } ^ { \mathrm { t o t a l } } = \boldsymbol { r } _ { t } ^ { \mathrm { e x t } } + \boldsymbol { \alpha } _ { i } \cdot \boldsymbol { r } _ { t } ^ { \mathrm { i n t } } , } \end{array}\tag{4}
$$

where $\alpha _ { i }$ controls the exploration-exploitation trade-off per client. This local RND setup enables adaptive exploration without external shaping or manual tuning.

## C. Federated Coordination with Exploration Summaries

Uncoordinated exploration can lead to redundant efforts across clients. To enable efficient collaboration, we propose a federated privacy-preserving coordination mechanism. Each client generates compact exploration statistics $\mathcal { E } _ { i } ,$ such as visitation counts, frequency histograms over state embeddings, or top-k novel state hashes based on intrinsic rewards. These summaries are low-dimensional and protect sensitive data.

The central server aggregates these statistics to form a global exploration profile as

$$
\mathcal { G } = \mathrm { A g g r e g a t e } ( \{ \mathcal { E } _ { i } \} _ { i = 1 } ^ { N } ) ,\tag{5}
$$

where Aggregate(·) combines client contributions, e.g., summing visitation counts for the state cluster k as

$$
v _ { \mathrm { g l o b a l } } [ k ] = \sum _ { i = 1 } ^ { N } v _ { i } [ k ] .\tag{6}
$$

The server then computes a global novelty prior to identifying underexplored states:

$$
\mathcal { P } _ { \mathrm { n o v e l } } ( s ) = \frac { 1 } { 1 + v _ { \mathrm { g l o b a l } } [ \mathrm { c l u s t e r } ( s ) ] } ,\tag{7}
$$

Algorithm 1 EDPFRL-IM: Exploration-Driven Personalized Federated   
Reinforcement Learning   
Require: Clients $\{ C _ { i } \} _ { i = 1 } ^ { N } ,$ global rounds T, local epochs E, RND module   
$( f _ { \mathrm { t g t } } , f _ { \mathrm { p r e d } } )$ , exploration weight $\alpha _ { i } ,$ novelty bias $\bar { \boldsymbol \beta }$   
1: for each global round $t = 1$ to T do   
2: for each client $C _ { i }$ in parallel do   
3: Initialize local buffer $\mathcal { D } _ { i }$   
4: for local step $e = 1$ to E do   
5: Interact with environment to collect $\left( s _ { t } , a _ { t } , r _ { t } ^ { \mathrm { e x t } } , s _ { t + 1 } \right)$   
6: Compute intrinsic reward: $r _ { t } ^ { \mathrm { i n t } } = \big \| f _ { \mathrm { t g t } } ( s _ { t + 1 } ) - f _ { \mathrm { p r e d } } ( s _ { t + 1 } ) \big \| _ { 2 } ^ { 2 }$   
7: Store $( s _ { t } , a _ { t } , r _ { t } ^ { \mathrm { t o t a l } } , s _ { t + 1 } )$ in $\mathcal { D } _ { i } ,$ where $r _ { t } ^ { \mathrm { t o t a l } } = \dot { r _ { t } ^ { \mathrm { e x t } } } + \alpha _ { i } \cdot r _ { t } ^ { \mathrm { i n t } }$   
8: Update policy $\pi _ { i }$ using local RL optimizer (e.g., PPO, SAC)   
9: end for   
10: Compute exploration summary (e.g., histogram, top-k novel states)   
11: Send summary to server   
12: end for   
13: Server aggregates summaries into global novelty prior P<sub>novel</sub>   
14: Broadcast $\dot { \mathcal { P } _ { \mathrm { n o v e l } } }$ to all clients   
15: for each client $C _ { i }$ do   
16: Update sampling strategy:   
$\begin{array} { r } { \dot { \mathbf { \phi } } _ { p _ { i } ( s ) } ^ { \mathbf { \phi } ^ { \mathbf { \phi } } } \propto \dot { ( 1 + \beta \cdot \mathcal { P } _ { \mathrm { n o v e l } } ( s ) ) } \cdot p _ { i } ^ { \mathrm { u n i f o r m } } ( s ) } \end{array}$   
17: end for   
18: end for

where cluster(s) maps the state s to a cluster index. This prior is broadcast to all clients to guide their exploration.

Each client incorporates the global novelty prior into its policy update as

$$
\theta _ { i }  \theta _ { i } + \eta \nabla _ { \theta _ { i } } \mathbb { E } _ { \pi _ { \theta _ { i } } , p _ { i } ( s ) } [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } ( r _ { i , t } ^ { \mathrm { e x t } } + \alpha _ { i } r _ { i , t } ^ { \mathrm { i n t } } ) ] ,\tag{8}
$$

where the probability of sampling state s from the experience buffer of client i, $\mathcal { D } _ { i }$ , is expressed as

$$
p _ { i } ( s ) \propto ( 1 + \beta \cdot \mathcal { P } _ { \mathrm { n o v e l } } ( s ) ) \cdot p _ { i } ^ { \mathrm { u n i f o r m } } ( s ) ,\tag{9}
$$

with $p _ { i } ^ { \mathrm { u n i f o r m } } ( s ) = 1 / | \mathcal { D } _ { i } |$ as the uniform baseline probability and $\beta \geq 0$ controlling the influence of the global prior. This approach ensures that clients focus on globally novel states, promoting diverse exploration while maintaining privacy and low communication costs.

Algorithm 1 describes the EDPFRL-IM method. Each client gathers experience and computes intrinsic rewards using RND (Lines 5–7), then adjusts its local policy (Line 8). Clients transmit compressed exploration summaries to the server (Line 10), which aggregates them into a global novelty prior (Line 12) and broadcasts it (Line 13). Clients use this prior for biased experience sampling (Line 15), which allows for coordinated exploration without compromising privacy. An overview of our framework is shown in Fig. 1.

## III. EXPERIMENTS

## A. Simulation Setup

Environments: We test the performance of EDPFRL-IM in standard reinforcement learning environments with sparse rewards and heterogeneous dynamics. We employ modified versions of MountainCar-v0 [10] and CartPole-sparse [11], which are chosen for their sensitivity to exploration quality and their significance in comparing sample efficiency.

![](images/b3a9a6b7a8fa62a5035147a463ce1db1b6dde42eaee0084d07160dc232ec0a85.jpg)  
Fig. 1. An overview of the proposed EDPFRL-IM framework. Based on local experience, each client uses intrinsic incentives to do reinforcement learnin (via RND). Clients provide compressed exploration summaries to a central server regularly, which are then aggregated into a global novelty prior. This prior is transmitted back to guide tailored exploration, allowing clients to explore together while maintaining their private information.

Federated Setup: We simulate a realistic FRL scenario by considering N = 10 clients, each running in a different local environment built from MountainCar-v0 and a modified CartPole environment with sparse rewards. We create heterogeneity among clients by adjusting the elements of the environment, including gravity, friction, and reward shaping. For instance, in MountainCar-v0, clients earn additional shaping incentives based on distance or velocity, and gravity varies from 0.0025 to 0.006. In CartPole-sparse, clients vary in pole mass and cart friction, and earn a reward only after balancing for a minimum duration. As an example, one client’s MountainCar-v0 variation employs gravity 0.0045, friction 0.005, and a shaped reward of $r _ { t } = - 1 { + } 0 . 1 { \cdot } | x { - } x _ { \mathrm { g o a l } } |$ to promote goal-directed travel. These differences lead to different MDPs, indicating extremely non-IID conditions where effective policy learning requires personalization.

Baseline Comparison: Training proceeds for $T ~ = ~ 1 0 0$ global communication rounds, with each client performing E = 10 local PPO updates per round. Policies are represented by fully connected two-layer neural networks with 64 hidden units and ReLU activations. Each client has its own Random Network Distillation (RND) module, with both the target and predictor networks consisting of a single hidden layer of size 128. The intrinsic reward coefficient $\alpha _ { i }$ is set to 0.1, while the novelty sampling bias $\beta$ is set to 0.5, unless specified otherwise.

Implementation Details: We compare our approach with the following baselines: (i) Local RL, in which every client trains on its own without any federated coordination; (ii) $F e \mathrm { - }$ dRL, a common FRL method with global policy aggregation; and (iii) FedRL+RND, which applies intrinsic rewards locally without exploration coordination. For the broader comparison in Table III, we additionally report single-agent intrinsicmotivation methods run in the non-federated setting (RND [6], ICM [7], and count-based exploration (CBE) [8]), as well as federated and personalized FL baselines without coordinated exploration (FedAvg-RL [4], pFedMe [5], and FedPer++ [12]). All baselines are trained under the same environment configurations, client counts, and total communication budget as EDPFRL-IM for a fair comparison. The Adam optimizer is used to train all models, with a discount factor of $\gamma = 0 . 9 9$ and a learning rate of $3 \times 1 0 ^ { - 4 }$

## B. Simulation Results

Main Performance Comparison: Fig. 2 shows the average return in MountainCar-v0 (left) versus CartPole-sparse (right) across communication rounds. EDPFRL-IM outperformed all baselines (Local RL, FedRL, and FedRL+RND) consistently throughout the initial rounds, with significantly higher returns due to the efficient use of available samples. In MountainCarv0, EDPFRL-IM achieved quick convergence with a smooth learning curve despite the presence of sparse rewards and difficult state transition dynamics. In CartPole-sparse, EDPFRL IM had a similar advantage over the other methods. Further, when comparing EDPFRL-IM to FedRL+RND, EDPFRL-IM produced greater average returns and exhibited a more consistent rate of convergence, supporting the claim that coordinated exploration through aggregated novelty priors contributed to superior performance across the range of tested conditions.

Personalization Benefit: Table I presents the mean returns for each client from both MountainCar-v0 and CartPolesparse, with EDPFRL-IM producing greater performance than the FedRL baseline across all clients. The consistent gains highlight the benefit of combining intrinsic motivation with personalization. Additionally, the lower performance variance resulting from EDPFRL-IM allows for greater consistency and personalization of policy learning, which is critical across a wide range of diverse conditions.

![](images/275b9a8120469d2ac7c58adc934a7a5db1c9aa48d5745a5bc433eeb365c2c98a.jpg)  
Fig. 2. Performance comparison of reinforcement learning methods across two environments over 100 communication rounds. The left subplot shows the average return for MountainCar-v0, and the right subplot shows the average return for CartPole-sparse with four baseline methods.

TABLE I  
AVERAGE RETURN PER CLIENT FOR EDPFRL-IM AND FEDRL ACROSS TWO ENVIRONMENTS. EDPFRL-IM CONSISTENTLY OUTPERFORMS FEDRL IN ALL CLIENTS.
<table><tr><td>Method</td><td>C1</td><td>C2</td><td>C3 C4</td><td></td><td>C5 C6</td><td>C7</td><td>C8</td><td>C9</td><td>C10</td></tr><tr><td></td><td colspan="7">MountainCar-v0</td></tr><tr><td></td><td></td><td></td><td>EDPFRL-IM |0.72 0.68 0.70 0.66 0.69 0.74 0.71 0.73 0.75 0.70</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedRL</td><td></td><td></td><td>0.42 0.390.400.38 0.41 0.44 0.400.42 0.430.41</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>CartPole-sparse</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FedRL</td><td></td><td></td><td>EDPFRL-IM |0.68 0.65 0.70 0.66 0.67 0.71 0.69 0.72 0.70 0.68 0.36 0.34 0.35 0.33 0.37 0.38 0.36 0.35 0.37 0.36</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Ablation: Role of Coordinated Exploration: Table II shows an ablation study that compares the impact of exploration coordination in our proposed strategy. When intrinsic motivation is completely removed, as with the FedRL implementation, agent performance is low due to the lack of exploratory behavior. Adding only the intrinsic reward component (FedRL+RND) provides some improvement but lacks clear directionality (i.e., no consensus on a path). Removing the global novelty prior from EDPFRL-IM (i.e., β = 0) improves performance slightly through individualized customization; however, performance is still substantially below the full method. EDPFRL-IM achieves the best returns in both environments when coordinated exploratory behavior is used (i.e., β = 0.5), demonstrating the value of leveraging crossclient novelty to guide exploration in federated reinforcement learning settings.

Exploration Coverage Analysis: As illustrated in Fig. 3, clients’ exploration coverage (percentage of the state space that clients visit) in both environments is much greater when using EDPFRL-IM compared to using FedRL and FedRL+RND. This indicates that the coordinated use of intrinsic motivation between clients is crucial for increasing the diversity of exploration. Increased diversity in exploration provides for greater generalization and faster adaptation in heterogeneous environments.

TABLE II  
ABLATION STUDY: AVERAGE RETURN OVER THE FINAL 10 COMMUNICATION ROUNDS (ROUNDS 91–100). COORDINATED EXPLORATION (EDPFRL-IM) LEADS TO THE BEST PERFORMANCE ACROSS ENVIRONMENTS.
<table><tr><td>Method Variant</td><td>MountainCar-v0</td><td>CartPole-sparse</td></tr><tr><td>FedRL (no RND)</td><td>0.40</td><td>0.35</td></tr><tr><td>FedRL+RND (no coordination)</td><td>0.60</td><td>0.54</td></tr><tr><td>EDPFRL-IM (β = 0)</td><td>0.63</td><td>0.58</td></tr><tr><td>EDPFRL-IM (ours)</td><td>0.80</td><td>0.76</td></tr></table>

Robustness to Cold Start Clients: We assessed the adaptability of the three methods to the introduction of a coldstart client during training (at round 5) by measuring how quickly and how well the cold-start client performed in both environments, as shown in Fig. 4. EDPFRL-IM uses globally aggregated exploration knowledge to facilitate rapid and smooth adaptation for cold-start clients in both environments, whereas FedRL and FedRL+RND do not use this form of globally aggregated knowledge, which negatively affects their ability to support early-stage learners, resulting in slower progress.

Comparison with Existing Approaches: Table III presents the final average return achieved by several federated and personalized reinforcement learning algorithms. EDPFRL-IM outperformed all comparison methods, including RND, ICM, CBE, FedAvg-RL, pFedMe, FedPer++, and FedRL+RND, in both environments. Although approaches such as pFedMe and FedPer++ support personalization of RL models, they provide limited exploration capabilities for discovering new behaviors. FedRL+RND incorporates intrinsic motivation through curiosity-driven exploration but lacks an efficient mechanism for personalized federated adaptation across clients. The proposed EDPFRL-IM framework combines structured curiositydriven exploration with federated personalization, yielding significantly improved performance and learning efficiency compared with existing methods.

![](images/fd536f6691a099e8f0ebc878a3d7367af8feb51322e2b82e8559f68e5f3a6dd1.jpg)

Fig. 3. Exploration coverage across clients in two environments. EDPFRL-IM enables broader state-space exploration by coordinating curiosity across clients  
![](images/787d2be6a44af1c0d5b58aa170664b6881fb4fbad64a765bae8fd603e1efa493.jpg)  
Fig. 4. Return trajectories of a cold-start client introduced at round 5 in (a) MountainCar-v0 and (b) CartPole-sparse. EDPFRL-IM quickly adapts by leveraging the global exploration prior, outperforming both FedRL and FedRL+RND.

TABLE III  
FINAL AVERAGE RETURN COMPARISON ACROSS METHODS
<table><tr><td>Method</td><td>MountainCar-v0</td><td>CartPole-sparse</td></tr><tr><td>RND [6]</td><td>0.31</td><td>0.32</td></tr><tr><td>ICM [7]</td><td>0.34</td><td>0.36</td></tr><tr><td>CBE [8]</td><td>0.39</td><td>0.49</td></tr><tr><td>FedAvg-RL [4]</td><td>0.38</td><td>0.42</td></tr><tr><td>pFedMe [5]</td><td>0.40</td><td>0.45</td></tr><tr><td>FedPer++ [12]</td><td>0.41</td><td>0.44</td></tr><tr><td>FedRL+RND [2], [6]</td><td>0.48</td><td>0.52</td></tr><tr><td>EDPFRL-IM (Ours)</td><td>0.76</td><td>0.74</td></tr></table>

## IV. CONCLUSION

This paper presents EDPFRL-IM, an exploration-driven personalized federated reinforcement learning (PFRL) system that leverages intrinsic motivation to provide coordinated yet private client exploration. EDPFRL-IM addresses an important shortcoming of prior, less effective PFRL systems by incorporating a global novelty prior computed from a compressed set of client exploration statistics, enabling each client to learn while adapting to its local context and benefiting from the collective experience of the remaining clients. Comprehensive evaluation of EDPFRL-IM in environments with sparse reward structures and high variability demonstrates that it achieves significant improvements in sample efficiency and personalization, and adapts faster for cold-start clients than its peers. EDPFRL-IM consistently outperforms baseline systems using both standard and personalized FL algorithms, including various intrinsic-incentive-based FL algorithms, and does so with no compromise in terms of privacy.

## REFERENCES

[1] A. K. Shakya, G. Pillai, and S. Chakrabarty, “Reinforcement learning algorithms: A brief survey,” Expert Systems with Applications, vol. 231, p. 120495, 2023.

[2] X. Fan, Y. Ma, Z. Dai, W. Jing, C. Tan, and B. K. H. Low, “Fault-tolerant federated reinforcement learning with theoretical guarantee,” Advances in neural information processing systems, vol. 34, pp. 1007–1021, 2021.

[3] R. Rahman and D. C. Nguyen, “Improved modulation recognition using personalized federated learning,” IEEE Transactions on Vehicular Technology, 2024.

[4] H. Wang, Z. Kaplan, D. Niu, and B. Li, “Optimizing federated learning on non-iid data with reinforcement learning,” in IEEE INFOCOM 2020- IEEE conference on computer communications, pp. 1698–1707, IEEE, 2020.

[5] C. T Dinh, N. Tran, and J. Nguyen, “Personalized federated learning with moreau envelopes,” Advances in neural information processing systems, vol. 33, pp. 21394–21405, 2020.

[6] A. Nikulin, V. Kurenkov, D. Tarasov, and S. Kolesnikov, “Antiexploration by random network distillation,” in International conference on machine learning, pp. 26228–26244, PMLR, 2023.

[7] C. Colas, P. Fournier, M. Chetouani, O. Sigaud, and P.-Y. Oudeyer, “Curious: intrinsically motivated modular multi-goal reinforcement learning,” in International conference on machine learning, pp. 1331–1340, PMLR, 2019.

[8] M. C. Machado, M. G. Bellemare, and M. Bowling, “Count-based exploration with the successor representation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 5125–5133, 2020.

[9] M. Lauri, D. Hsu, and J. Pajarinen, “Partially observable markov decision processes in robotics: A survey,” IEEE Transactions on Robotics, vol. 39, no. 1, pp. 21–40, 2022.

[10] S. T. Chavali, C. T. Kandavalli, T. Sugash, and J. Amudha, “Modelling a reinforcement learning agent for mountain car problem using q–learning with tabular discretization,” in 2022 IEEE 2nd Mysore Sub Section International Conference (MysuruCon), pp. 1–5, IEEE, 2022.

[11] N. Bjorck, C. P. Gomes, and K. Q. Weinberger, “Towards deeper deep reinforcement learning with spectral normalization,” Advances in neural information processing systems, vol. 34, pp. 8242–8255, 2021.

[12] J. Xu, Y. Yan, and S.-L. Huang, “Fedper++: Toward improved personalized federated learning on heterogeneous and imbalanced data,” in 2022 International Joint Conference on Neural Networks (IJCNN), pp. 01–08, IEEE, 2022.
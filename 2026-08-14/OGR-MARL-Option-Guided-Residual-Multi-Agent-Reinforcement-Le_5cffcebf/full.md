# OGR-MARL: Option-Guided Residual Multi-Agent Reinforcement Learning for Heterogeneous USV Cooperative Pursuit in Constrained Port Waterways

Jiayang Mao†\*

Lanfeng Wang†

College of Information Engineering

Shenzhen International Graduate School

Sichuan Agricultural University

Zhao-Han Peng

Tsinghua University

Yaan, China

maojiayang2024@163.com

Shenzhen, China

wanglf24@mails.tsinghua.edu.cn

Shenzhen International Graduate School

Tsinghua University

Shenzhen, China

pzh23@mails.tsinghua.edu.cn

Abstract—Heterogeneous USV cooperative pursuit in constrained port waterways requires evader interception under navigation, traffic, and role constraints. This paper proposes OGR-MARL, an option-guided residual multi-agent reinforcement learning framework that is decoupled from a specific MARL algorithm. OGR-MARL integrates shared evader belief, roleconditioned option targets, adaptive rule penalties, and residual policy learning, allowing different MARL algorithms to learn corrective actions on top of rule-guided behaviors rather than exploring constrained port environments from scratch. We instantiate OGR-MARL with representative continuous-control MARL backbones, including MADDPG, MATD3, MAPPO, and MASAC, yielding OGR-MADDPG, OGR-MATD3, OGR-MAPPO, and OGR-MASAC. Experiments in an abstract Xiazhimen portwaterway scenario show that the OGR-MASAC instantiation achieves a 75.0% capture rate, promising mission-effective rule compliance, and the best heterogeneous coordination among the tested methods. Without retraining, zero-shot transfer to a QGIS/AIS-informed Xiazhimen map achieves promising results, demonstrating the generalization potential of OGR-MARL in more complex port scenarios.

Index Terms—cooperative pursuit, multi-agent reinforcement learning, option guidance, residual learning, heterogeneous USV

## I. INTRODUCTION

Unmanned surface vehicles (USVs) are increasingly vital for maritime surveillance, environmental monitoring, and harbor security because they can operate persistently in waters where human operation is costly or risky [1]. Multi-USV coordination has received sustained attention as a way to improve robustness and efficiency in complex marine missions, among which the cooperative pursuit problem constitutes a significant class of issues [2]. However, pursuit in constrained port-waterways is more demanding than in open environments. Pursuers must navigate narrow lanes, avoid shores, anchorages, and moving traffic while capturing an evader.

Existing pursuit studies often focus on simplified or open scenarios. Classical model-based methods, such as artificial potential fields (APF), model predictive control, and path planning, provide interpretable guidance and COLREGs-aware collision avoidance [3]–[6]. Yet, they require scenario-specific tuning and struggle with partial observability, role heterogeneity, and online role switching.

Conversely, multi-agent reinforcement learning (MARL) offers a data-driven approach for environment exploration and cooperative pursuit [7]–[9]. Despite some successful applications in multi-USV cooperative pursuit [10]–[12], directly applying MARL to constrained port scenarios from scratch remains challenging. A pure policy-gradient learner may suffer from sparse rewards, frequent rule violations, and exploration inefficiency, often expending budgets on collisions or failing to leave the harbor. Conversely, a pure rule-based controller is interpretable but lacks adaptability against tactical evader replanning.

To bridge this gap, we propose option-guided multi-agent reinforcement learning (OGR-MARL), an algorithm-agnostic framework for heterogeneous multi-USV cooperative pursuit in constrained port-waterways. The pursuing team is heterogeneous: interceptors have capture capability with higher speed, while scouts have a larger sensing radius for target visibility. A\* search [13] and APF are utilized to model evader behavior and generate high-level option targets. Motivated by the option framework for temporal decomposition [14] and residual reinforcement learning with rule-guided policies [15], our approach integrates an option guidance module, which provides high-level modes and geometric targets, with a general MARL algorithm that learns residual continuous actions and coordination patterns for the heterogeneous pursuing team. This design effectively corrects timing, local motion, and interaction effects beyond rule-based guidance while preserving structural safety priors and learned adaptability [16].

The main contributions are as follows:

• We establish a benchmark scenario for heterogeneous multi-USV cooperative pursuit in constrained port waterways, incorporating shores, anchorage areas, bidirectional traffic lanes, buffer regions, and moving cargo vessels.

• We propose OGR-MARL, an algorithm-agnostic optionguided residual MARL framework that integrates shared evader belief, role-conditioned option targets, and adaptive rule penalties, and can be instantiated with different MARL backbones.

![](images/ac63196367a461b54f2724c554573e81517a428a765100158c0654fe29707e0b.jpg)  
Fig. 1. Abstracted Xiazhimen port waterways.

• We evaluate the proposed framework through backbone comparison, reward ablation, and zero-shot transfer to a QGIS/AIS-informed Xiazhimen map.

## II. PROBLEM FORMULATION

## A. Simulated Port Waterway Scenario

The simulation scenario is an abstracted Xiazhimen port waterway (in Zhejiang Province of China) on a $1 0 0 0 \times 4 0 0$ domain. As shown in Fig. 1, the channel is bounded by the north and south islands, with a USV base located at $x \in$ [400, 600]. Two anchorage areas occupy $[ 1 0 0 , 3 0 0 ] \times [ 3 3 0 , 3 7 0 ]$ and $[ 7 0 0 , 9 0 0 ] \times [ 2 3 0 , 2 7 0 ]$ ]. The outbound and inbound traffic lanes occupy $y \in [ 3 1 0 , 3 5 0 ]$ and [250, 290], respectively, and a buffer area occupies $y \in [ 2 9 0 , 3 1 0 ]$ . The line $x = 9 8 0$ serves as the eastern escape boundary, and an episode terminates as an escape when the evader crosses this boundary.

The pursuer team is denoted by $\mathcal { N } ~ = ~ \{ P _ { 1 } , P _ { 2 } , P _ { 3 } , P _ { 4 } \}$ Pursuers $P _ { 1 }$ and $P _ { 2 }$ are fast interceptors and the only USVs with capture capability, while $P _ { 3 }$ and $P _ { 4 }$ are slower scouts with larger sensing radius. Four cargo vessels move along the center lines of the traffic lanes:

$$
\boldsymbol { x } _ { k , t + 1 } ^ { c } = \left( \boldsymbol { x } _ { k , t } ^ { c } + \boldsymbol { v } _ { k } ^ { c } \Delta t \right) \bmod \boldsymbol { L } _ { x } , \quad \boldsymbol { y } _ { k , t + 1 } ^ { c } = \boldsymbol { y } _ { k , t } ^ { c } ,\tag{1}
$$

where $L _ { x } = 1 0 0 0$ . The three outbound vessels move eastward at a speed of 4.69 simulation units per second, while the inbound vessel moves westward at a speed of -6.26.

## B. Pursuer Dynamics and Actions

Each pursuer is modeled by a three-degree-of-freedom (3- DOF) motion model [17]. Let

$$
\pmb { \eta } _ { i } = [ x _ { i } , y _ { i } , \psi _ { i } ] ^ { T } , \quad \pmb { \nu } _ { i } = [ u _ { i } , v _ { i } , r _ { i } ] ^ { T } ,\tag{2}
$$

where $( x _ { i } , y _ { i } )$ are the position coordinates, $\psi _ { i }$ is the heading angle, and $( u _ { i } , v _ { i } , r _ { i } )$ denote the surge velocity, sway velocity, and yaw rate, respectively. The dynamic equations are

$$
\dot { \pmb \eta } _ { i } = { \pmb J } ( \psi _ { i } ) { \pmb \nu } _ { i } , \quad M _ { i } \dot { \pmb \nu } _ { i } + C _ { i } { \pmb \nu } _ { i } + D _ { i } { \pmb \nu } _ { i } = { \pmb \tau } _ { i } ,\tag{3}
$$

$$
\begin{array} { r } { \pmb { J } ( \psi _ { i } ) = \left\lceil \begin{array} { c c c } { \cos \psi _ { i } } & { - \sin \psi _ { i } } & { 0 } \\ { \sin \psi _ { i } } & { \cos \psi _ { i } } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} \right\rceil . } \end{array}\tag{4}
$$

Here, $M _ { i } , C _ { i }$ , and $D _ { i }$ denote the inertia, Coriolis–centripetal, and nonlinear damping terms, respectively.

TABLE I  
PARAMETERS OF THE PURSUERS
<table><tr><td>Parameter</td><td>Interceptor</td><td>Scout</td></tr><tr><td>Normalized mass m</td><td>380</td><td>760</td></tr><tr><td>Normalized yaw inertia  $I _ { z }$ </td><td>620</td><td>1760</td></tr><tr><td>Maximum surge speed  $u ^ { \mathrm { m a x } }$  (sim. units/s)</td><td>10.0</td><td>4.6</td></tr><tr><td>Maximum heading increment (rad)</td><td>0.580</td><td>0.260</td></tr><tr><td>Maximum thrust</td><td>1320</td><td>520</td></tr><tr><td>Sensing radius (sim. units)</td><td>8</td><td>180</td></tr><tr><td>Capture capability</td><td>Yes</td><td>No</td></tr></table>

A fourth-order Runge–Kutta integrator is used for the innerloop dynamics. A low-level speed-heading controller maps each high-level MARL action of the pursuers to a surge force and yaw moment. The high-level action of pursuer i is

$$
\mathbf { } a _ { i } = [ a _ { i } ^ { v } , a _ { i } ^ { \psi } ] ^ { T } ,\tag{5}
$$

where $a _ { i } ^ { v }$ is related to the surge speed and $a _ { i } ^ { \psi }$ is related to the heading angle. The desired surge speed and heading angle are

$$
u _ { d , i } = \frac { u _ { i } ^ { \operatorname* { m a x } } } { 2 } ( a _ { i } ^ { v } + 1 ) , \quad \psi _ { d , i } = \psi _ { i } + a _ { i } ^ { \psi } \Delta \psi _ { i } ^ { \operatorname* { m a x } } .\tag{6}
$$

Let $e _ { i } ^ { u } = u _ { d , i } - u _ { i }$ and $e _ { i } ^ { \psi } = \mathrm { w r a p } ( \psi _ { d , i } - \psi _ { i } )$ , where the wrap function normalizes the heading-angle difference to the principal interval, $i . e . , \ : ( - \pi , \pi ]$ . The controller computes

$$
\tau _ { x , i } = \mathrm { s a t } \left( K _ { p , i } ^ { u } e _ { i } ^ { u } + K _ { I , i } ^ { u } \int e _ { i } ^ { u } d t , \ 0 , \ 2 T _ { i } ^ { \mathrm { m a x } } \right) ,\tag{7}
$$

$$
\tau _ { z , i } = \mathrm { s a t } \left( K _ { p , i } ^ { \psi } e _ { i } ^ { \psi } - K _ { d , i } ^ { \psi } r _ { i } , \ - \tau _ { z , i } ^ { \mathrm { m a x } } , \ \tau _ { z , i } ^ { \mathrm { m a x } } \right) ,\tag{8}
$$

where sat $\left( \cdot , l , u \right)$ denotes saturation to [l, u]. The corresponding port and starboard thrust commands are

$$
T _ { p , i } = \mathrm { s a t } \left( \frac { \tau _ { x , i } } { 2 } + \frac { \tau _ { z , i } } { B _ { i } } , 0 , T _ { i } ^ { \mathrm { m a x } } \right) ,\tag{9}
$$

$$
T _ { s , i } = \mathrm { s a t } \left( \frac { \tau _ { x , i } } { 2 } - \frac { \tau _ { z , i } } { B _ { i } } , 0 , T _ { i } ^ { \mathrm { m a x } } \right) .\tag{10}
$$

The resulting input force is

$$
\pmb { \tau } _ { i } = [ T _ { p , i } + T _ { s , i } , 0 , ( T _ { p , i } - T _ { s , i } ) B _ { i } / 2 ] ^ { T } .\tag{11}
$$

The parameters of the pursuers are listed in Table I. Scouts provide visibility in a wide area and maintain the shared evader belief, whereas interceptors rely on close-range perception and the team-level belief for final interception. The heterogeneous system encourages scouts to detect the evader early.

## C. The Evader

The evader is modeled as a mass point with constrained acceleration and yaw rate. Its state is $\begin{array} { r } { { \pmb s } _ { e } = [ x _ { e } , y _ { e } , \psi _ { e } , v _ { e } ] ^ { T } } \end{array}$ Given the desired speed $v _ { d }$ and heading angle $\psi _ { d } ,$ define

$$
\Delta \boldsymbol { v } _ { t } = \mathrm { s a t } ( \boldsymbol { v } _ { d } - \boldsymbol { v } _ { e , t } , - a _ { e } ^ { \mathrm { m a x } } \Delta t , a _ { e } ^ { \mathrm { m a x } } \Delta t ) ,\tag{12}
$$

$$
\Delta \psi _ { t } = \mathrm { s a t } ( \mathrm { w r a p } ( \psi _ { d } - \psi _ { e , t } ) , - \omega _ { e } ^ { \mathrm { m a x } } \Delta t , \omega _ { e } ^ { \mathrm { m a x } } \Delta t ) .\tag{13}
$$

The evader’s state is updated by

$$
\begin{array} { r } { v _ { e , t + 1 } = \mathrm { s a t } ( v _ { e , t } + \Delta v _ { t } , 0 , v _ { e } ^ { \mathrm { m a x } } ) , } \end{array}\tag{14}
$$

$$
\psi _ { e , t + 1 } = \mathrm { w r a p } ( \psi _ { e , t } + \Delta \psi _ { t } ) ,\tag{15}
$$

$$
x _ { e , t + 1 } = x _ { e , t } + v _ { e , t + 1 } \cos \psi _ { e , t + 1 } \Delta t ,
$$

$$
y _ { e , t + 1 } = y _ { e , t } + v _ { e , t + 1 } \sin \psi _ { e , t + 1 } \Delta t .\tag{16}
$$

(17)

The evader uses an $\mathbf { A } ^ { * } .$ -based global planner and an APFbased local collision-avoidance module. The $\mathbf { A } ^ { * }$ layer plans a path toward the eastern escape boundary while taking into account the shores, cargo vessels, and inflated safety regions around pursuers. The APF layer combines attraction to the current waypoint planned by $\mathbf { A } ^ { * }$ with repulsion from shores, cargo vessels, and nearby pursuers. The desired heading angle is the same as the direction of the force generalized by APF:

$$
F _ { e } = F _ { \mathrm { g o a l } } + \sum _ { h \in \mathcal { H } } F _ { \mathrm { r e p } } ^ { h } + \sum _ { i \in \mathcal { N } } F _ { \mathrm { r e p } } ^ { i } ,\tag{18}
$$

where H contains the shore boundary and cargo vessels, and the desired heading angle is obtained from the force direction.

## D. Termination Condition

An episode ends with capture, escape, or timeout. Capture is declared if any interceptor satisfies:

$$
\operatorname* { m i n } _ { i \in \{ P _ { 1 } , P _ { 2 } \} } \| ( x _ { i } , y _ { i } ) - ( x _ { e } , y _ { e } ) \| _ { 2 } \leq R _ { c } ,\tag{19}
$$

where $R _ { c }$ is a distance threshold. Scouts may detect and track the evader, but they do not have capture capability.

The evader escapes when it crosses the eastern escape boundary. Timeout happens when the maximum episode length is reached without capture or escape.

## E. Partial Observation

The cooperative pursuit task for the pursuers is partially observable because only scouts have long-range sensing capability: Interceptors have a sensing radius of 8 simulation units, whereas scouts have a sensing radius of 180. The team maintains a shared evader belief from the latest evader observation. When any USV detects the evader, the belief is updated. Otherwise, the belief is propagated by dead reckoning using the last known speed and heading angle:

$$
\hat { x } _ { t + 1 } = \hat { x } _ { t } + \hat { v } _ { t } \cos \hat { \psi } _ { t } \Delta t , \quad \hat { y } _ { t + 1 } = \hat { y } _ { t } + \hat { v } _ { t } \sin \hat { \psi } _ { t } \Delta t .\tag{20}
$$

The belief confidence $c _ { t }$ decays exponentially with the number of steps since the last observation:

$$
c _ { t } = \exp ( - n _ { t } / \tau _ { b } ) ,\tag{21}
$$

where $n _ { t }$ is the number of steps since the last observation and $\tau _ { b }$ is a decay constant.

Each pursuer receives a 61-dimensional observation vector consisting of 7 self-state variables, 12 relative teammate variables, 7 evader belief variables, 20 cargo vessel variables, 5 local geometry and rule compliance variables, a 5-dimensional option one-hot vector, 4 option-guidance variables comprising the relative option-target position, the target distance, and the distance to the shared evader belief, and the normalized episode time.

## III. OGR-MARL ALGORITHM

## A. CTDE Framework

The cooperative pursuit task is formulated as a partially observable multi-agent Markov decision process. We adopt the centralized training and decentralized execution (CTDE) paradigm for OGR-MARL, as illustrated in Fig. 2. During training, a centralized critic utilizes global information, while each actor executes using only local observations during decentralized execution. Each actor outputs a two-dimensional continuous residual control action. Due to heterogeneous dynamics, sensing ranges, and functional roles, we adopt independent actor–critic networks without parameter sharing.

![](images/ddde3032dffb50510811fbe1aaf0914d869259d1b9154ae538fbc0bf458f8723.jpg)  
Fig. 2. Information flow in OGR-MARL.

In the general OGR-MARL formulation, each pursuer policy is defined as

$$
\pi _ { i } ( { \pmb a } _ { i } ^ { \mathrm { r l } } \mid { \pmb o } _ { i } ) ,\tag{22}
$$

where $\pmb { a } _ { i } ^ { \mathrm { r l } }$ denotes the residual action produced by a generic MARL backbone conditioned on local observation $\mathbf { } o _ { i } .$

$\Omega _ { i } , \ G _ { i } ,$ and $\Gamma _ { i }$ denote the option selector, option-action generator, and blending rule of pursuer i, respectively. These modules are independent of the MARL algorithm and remain unchanged across different instantiations of OGR-MARL.

Algorithm 1 OGR-MARL Training and Evaluation   
Input: $\mathcal { E } , \mathcal { N } , \mathcal { S } , \mathcal { O } ,$ and $\kappa _ { \mathrm { e v a l } }$   
Output: $\Pi ^ { \mathrm { O G } } = \{ \pi _ { i } , \Omega _ { i } , G _ { i } , \Gamma _ { i } \} _ { i \in \mathcal { N } }$   
1: Initialize the MARL algorithm parameters $\pi _ { i } , Q _ { i } ,$ and (if applicable)   
auxiliary parameters   
2: Initialize replay buffer D and curriculum variables   
Training:   
3: for each training episode do   
4: Select scenario stage S according to curriculum schedule   
5: Reset environment $\mathcal { E } _ { S }$ , initialize belief ${ { b } _ { 0 } } ,$ set $t \gets 0$   
6: repeat   
7: Update shared evader belief b<sub>t</sub> and confidence c<sub>t</sub>   
8: Select option $m _ { i , t }$ via $\Omega _ { i }$ and generate option action ${ \pmb a } _ { i , t } ^ { \mathrm { o p t } }$ via   
$G _ { i }$   
9: Construct joint observation $\pmb { o } _ { t } = ( \pmb { o } _ { i , t } ) _ { i \in \mathcal { N } }$   
10: Sample residual action ${ \pmb a } _ { i , t } ^ { \mathrm { r l } } \sim \pi _ { i } ( { \cdot } \mid { \pmb o } _ { i , t } )$   
11: Blend actions via $\Gamma _ { i }$ to obtain executed action $_ { a _ { i , t } }$   
12: Execute ${ \mathbf { } } _ { { \mathbf { } } _ { { \mathbf { } } _ { { \mathbf { } } _ { { \mathbf { } } _ { { \mathbf { } } _ { { \mathbf { } } } } } } } }$ , observe $_ { o _ { t + 1 } }$ and terminal flag $d _ { t }$   
13: Update curriculum statistics and rule violation statistics if needed   
14: Compute individual reward $r _ { i , t } ^ { \mathrm { i n d } }$ and mixed reward $r _ { t }$   
15: Store transition $\left( o _ { t } , \pmb { a } _ { t } ^ { \mathrm { r l } } , \pmb { r } _ { t } , \pmb { \dot { o } } _ { t + 1 } , d _ { t } \right)$ in D   
16: Update the MARL parameters according to its optimization rule   
17: $t \stackrel { - } {  } t + 1$   
18: until $d _ { t } = 1$ (capture, escape, or timeout)   
19: end for   
Evaluation:   
20: Evaluate $\mathrm { \Pi } _ { \mathrm { I I } } \mathrm { \dot { o } G }$ in $\mathcal { E } _ { S _ { 4 } }$ over $\kappa _ { \mathrm { e v a l } }$ without parameter updates   
21: Report SR, TC, MRC, and HCS

## B. Option Guidance

Exploration over the aforementioned continuous action space is inefficient in constrained port waterway environments. To address this issue, we introduce a lightweight option layer on top of a generic MARL backbone. The option set is

$$
\mathcal { O } = \{ \mathrm { s e a r c h } , \mathrm { t r a c k } , \mathrm { i n t e r c e p t } , \mathrm { b l o c k } , \mathrm { r e c o v e r } \} .\tag{23}
$$

The option layer assigns each pursuer an execution mode according to its role, rule-compliance state, evader visibility, and belief confidence. Scouts select the search or track option to maintain channel coverage and update the shared evader belief. Interceptors select the intercept or block option to form a pincer maneuver, close in on the evader, and complete the capture. The recover option guides a pursuer away from shores, anchorage areas, and buffer regions, or prevents violations of traffic lane direction constraints.

TABLE IICURRICULUM LEARNING STAGES
<table><tr><td>Stage</td><td>Start Step</td><td>Evader Speed</td><td>Repulsion Scale</td><td>Capture Radius</td><td>Penalty Scale</td></tr><tr><td>SO</td><td>0</td><td>4.75</td><td>0.25</td><td>8.00</td><td>0.15</td></tr><tr><td>S1</td><td>16k</td><td>5.00</td><td>0.45</td><td>6.50</td><td>0.35</td></tr><tr><td>S2</td><td>56k</td><td>5.10</td><td>0.62</td><td>5.90</td><td>0.60</td></tr><tr><td>S3</td><td>288k</td><td>5.10</td><td>0.85</td><td>5.45</td><td>0.82</td></tr><tr><td>S4</td><td>400k</td><td>5.10</td><td>1.00</td><td>5.00</td><td>1.00</td></tr></table>

Within the proposed OGR-MARL framework, the option module provides a structured action prior that is independent of the underlying MARL algorithm. Let $\pmb { a } _ { i } ^ { \mathrm { r l } }$ denote the residual action produced by a generic MARL policy conditioned on local observations, and let ${ \pmb a } _ { i } ^ { \mathrm { o p t } }$ denote the option-guided action. The final executed action is obtained via a bounded blending mechanism:

$$
\pmb { a } _ { i } = \mathrm { c l i p } \left( ( 1 - \beta _ { i } ) \pmb { a } _ { i } ^ { \mathrm { r l } } + \beta _ { i } \pmb { a } _ { i } ^ { \mathrm { o p t } } , - 1 , 1 \right) .\tag{24}
$$

## C. Structured Reward and Curriculum Learning

To train OGR-MARL in constrained port waterways, we use a backbone-agnostic training design that combines structured reward shaping with curriculum learning. The reward provides task-, rule-, and role-level feedback, whereas the curriculum gradually increases task difficulty and constraint strength.

The individual reward of pursuer i is decomposed into four components:

$$
\begin{array} { r } { r _ { i , t } ^ { \mathrm { i n d } } = \chi _ { t } \left( r _ { i , t } ^ { \mathrm { T } } + r _ { i , t } ^ { \mathrm { R } } + r _ { i , t } ^ { \mathrm { H } } \right) + ( 1 - \chi _ { t } ) r _ { i , t } ^ { \mathrm { t e r m } } , } \end{array}\tag{25}
$$

where ${ \chi _ { t } \in \{ 0 , 1 \} }$ is a non-terminal indicator, $i . e . , \ \chi _ { t } \ = \ 1$ before the termination and $\chi _ { t } = 0$ at the terminal transition. The task-progress reward $r _ { i , t } ^ { \mathrm { T } }$ encourages pursuit behaviors that are essential for completing the mission, including harbor departure, evader tracking, interception, and pincer formation. The rule-compliance term $r _ { i , t } ^ { \mathrm { R } }$ penalizes violations of portwaterway constraints, such as shore intrusion and anchorage intrusion. The heterogeneous-role reward $r _ { i , t } ^ { \mathrm { H } }$ promotes complementary behaviors between scouts and interceptors, encouraging scouts to maintain evader visibility and interceptors to contribute to distance closing and capture. The terminal reward $r _ { i , t } ^ { \mathrm { t e r m } }$ evaluates the final mission outcome, including successful capture, evader escape, and timeout.

To promote cooperative behavior, the training reward blends individual feedback with team-level feedback:

$$
r _ { i , t } = \lambda r _ { i , t } ^ { \mathrm { i n d } } + ( 1 - \lambda ) \frac { 1 } { | \mathcal { N } | } \sum _ { j \in \mathcal { N } } r _ { j , t } ^ { \mathrm { i n d } } .\tag{26}
$$

Directly training on the hardest pursuit setting is inefficient because the evader moves actively, only interceptors have capture capability, and the capture radius is small. Therefore, we adopt a five-stage curriculum learning strategy [18], as summarized in Table II. Across stages, the evader becomes more difficult to capture, the capture condition becomes stricter, and the rule-compliance penalty is gradually strengthened. This schedule first allows the agents to discover basic harbor-exit, tracking, and pursuit behaviors under relatively permissive conditions, and then shifts learning toward precise interception and rule-compliant heterogeneous coordination.

TABLE III PARAMETERS FOR MARL ALGORITHMS
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Actor hidden sizes</td><td>(256, 256)</td><td>Learning rates</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Critic hidden sizes</td><td>(512, 512)</td><td>Time step</td><td> $0 . 5 \mathrm { s }$ </td></tr><tr><td>Target network update rate</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>Replay buffer</td><td> $1 \times 1 0 ^ { 6 }$ </td></tr><tr><td>Reward discount factor</td><td>0.995</td><td>Batch size</td><td>512</td></tr><tr><td>Episode step</td><td> $\mathrm { 8 \times 1 0 ^ { 3 } }$ </td><td>Total step</td><td> $7 \times 1 0 ^ { 5 }$ </td></tr></table>

## IV. EXPERIMENTS

## A. Experimental Setup

The algorithms are trained using eight parallel environments. The parameters for MARL algorithms are illustrated in Table III. MARL baselines execute only learned actions without option-guided action blending; option-related observation entries are kept only to preserve the same input schema. Therefore, the comparison isolates the effect of option-action execution rather than removing all option-related auxiliary inputs.

## B. Evaluation Metrics

The evaluation takes 100 test episodes with 100 seeds and a decision frame skip of 4. We use successful capture rate (SR), mean task time cost (TC), mission-effective rule compliance (MRC), and heterogeneous coordination score (HCS) to provide a comprehensive assessment.

Define the time-efficiency score as

$$
E _ { T } = \operatorname* { m i n } \left( 1 , { \frac { T _ { 0 } } { T } } \right) ,\tag{27}
$$

where $T$ is the episode duration in seconds. $T _ { 0 } ~ \mathrm { = ~ 1 0 0 s }$ is a predefined reference time, which is longer than the mean successful capture time of OGR-MARL but shorter than the mean successful capture time of most algorithms and the duration of an ideal escape.

To introduce mission-effective rule compliance (MRC), we first define the rule compliance rate (RCR) as

$$
\mathrm { R C R } = \frac { 1 } { | \mathcal { N } | T } \sum _ { t = 1 } ^ { T } \sum _ { i \in \mathcal { N } } \mathbb { I } \{ g _ { i , t } \} ,\tag{28}
$$

where $\mathbb { I } ( \cdot )$ denotes the indicator function. $g _ { i , t } = 1$ indicates that pursuer i satisfies all navigation rules at time step t, whereas $g _ { i , t } = 0$ indicates at least one rule violation, including intrusion to shore, anchorage or buffer, and wrong-direction motion when it is inside an inbound or outbound lane.

Then, MRC is computed as

$$
\mathrm { M R C } = \mathrm { S R } \cdot \mathrm { R C R } \cdot E _ { T } ,\tag{29}
$$

MRC is more suitable for the constrained port pursuit task than using SR or RCR alone because the mission requires simultaneous capture success, navigation-rule compliance, and time efficiency. A policy may obtain a high RCR by behaving conservatively or avoiding active interception, but such behavior is ineffective if the evader escapes.

TABLE IV  
EVALUATION METRICS OVER 100 EPISODES
<table><tr><td>Algorithm</td><td>SR↑ (%)</td><td>95%CI SR↑ (%)</td><td>TC↓ (s)</td><td>MRC↑</td><td>HCS↑</td></tr><tr><td>Expert Rules</td><td>35.0</td><td>[26.4, 44.7]</td><td>164.75</td><td>0.1387</td><td>0.5348</td></tr><tr><td>MADDPG</td><td>2.0</td><td>[0.6, 7.0]</td><td>184.65</td><td>0.0090</td><td>0.3041</td></tr><tr><td>OGR-MADDPG</td><td>5.0</td><td>[2.2, 11.2]</td><td>192.38</td><td>0.0175</td><td>0.4459</td></tr><tr><td>MATD3</td><td>1.0</td><td>[0.2, 5.4]</td><td>188.57</td><td>0.0034</td><td>0.3669</td></tr><tr><td>OGR-MATD3</td><td>59.0</td><td>[49.2, 68.1]</td><td>142.37</td><td>0.2862</td><td>0.6163</td></tr><tr><td>MAPPO</td><td>16.0</td><td>[10.1, 24.4]</td><td>172.15</td><td>0.0557</td><td>0.3954</td></tr><tr><td>OGR-MAPPO</td><td>72.0</td><td>[62.5, 79.9]</td><td>109.69</td><td>0.5114</td><td>0.6792</td></tr><tr><td>MASAC</td><td>14.0</td><td>[8.5, 22.1]</td><td>182.92</td><td>0.0437</td><td>0.4244</td></tr><tr><td>OGR-MASAC</td><td>75.0</td><td>[65.7, 82.5]</td><td>121.94</td><td>0.4283</td><td>0.6802</td></tr></table>

To measure the coordination performance of the heterogeneous system, we define HCS as

$$
\mathrm { H C S } = 0 . 3 0 \mathrm { S R } + 0 . 2 5 E _ { T } + 0 . 2 5 S _ { v } + 0 . 2 0 I _ { c } ,\tag{30}
$$

where $S _ { v }$ is the fraction of simulation steps in which at least one scout can sense the evader, and $I _ { c }$ is the ratio of positive distance-closing contribution provided by the interceptors. All terms are normalized to [0, 1]. HCS is introduced because task success alone cannot fully reflect whether the heterogeneous team performs coordinated role allocation. In the considered pursuit task, scouts are expected to maintain target visibility and update the shared evader belief, while interceptors are expected to reduce the evader distance and complete capture.

## C. Baseline Comparison

We evaluate OGR-MARL by instantiating it with different MARL backbones, including MADDPG, MATD3, MAPPO, and MASAC. The resulting variants are denoted as OGR-MADDPG, OGR-MATD3, OGR-MAPPO and OGR-MASAC, respectively. These variants share the same option-guidance module, shared evader belief, action-blending mechanism, and reward design, while differing only in the residual MARL learner. Therefore, the comparison evaluates both the general applicability of OGR-MARL and the influence of the selected residual learner.

The results in Table IV show that OGR-MARL provides a broadly applicable option-guided residual learning interface for different MARL algorithms. The pure expert-rule controller reaches a success rate of 35.0%, showing that rulebased option guidance alone is useful but insufficient against a fast and replanning evader. Pure MARL baselines perform poorly because direct continuous-control learning from scratch does not reliably discover the required cooperative behaviors under port-waterway constraints. Compared with their corresponding pure MARL baselines, the OGR variants generally improve SR, MRC, and HCS, indicating that the option layer supplies useful structural priors for constrained port pursuit. In particular, OGR-MATD3 improves the SR from 1.0% to 59.0%, and OGR-MASAC improves the SR from 14.0% to 75.0%. These results suggest that option guidance substantially reduces exploration difficulty and helps the learners discover harbor-exit, scout-tracking, and interceptor-pincer behaviors.

![](images/3d64fc89d832047f5ca7e0bac0c485a1323020840451df9ab9c4c9ca4ddbecf5.jpg)  
Fig. 3. A successful capture in the abstract Xiazhimen port waterways.

TABLE V  
REWARD ABLATION RESULTS OVER 100 EPISODES
<table><tr><td>Algorithm</td><td>SR↑(%)</td><td>95% CI SR↑ (%)</td><td>TC↓ (s)</td><td>MRC↑</td><td>HCS↑</td></tr><tr><td>OGR-MASAC</td><td>75.0</td><td>[65.7, 82.5]</td><td>121.94</td><td>0.4283</td><td>0.6802</td></tr><tr><td>w/o rule</td><td>31.0</td><td>[22.8, 40.6]</td><td>164.26</td><td>0.1299</td><td>0.5214</td></tr><tr><td>w/o role</td><td>51.0</td><td>[41.3, 60.6]</td><td>144.32</td><td>0.2403</td><td>0.5953</td></tr><tr><td>w/o rule+role</td><td>28.0</td><td>[20.1, 37.5]</td><td>167.60</td><td>0.1040</td><td>0.5032</td></tr></table>

Among the tested instantiations, OGR-MASAC achieves the best overall performance, with the highest SR, the highest HCS, and promising TC and MRC. This demonstrates that MASAC is the most effective residual learner within the proposed OGR-MARL framework in the considered task.

Fig. 3 presents a successful capture in the abstract Xiazhimen port waterways. The option-guided design encourages scouts to maintain evader visibility and interceptors to contribute to distance closing and final capture. We find that the remaining failures are mainly caused by inefficient terminal close-in interception rather than complete loss of the evader. Successful episodes finish with a mean closest interceptor–evader distance of 4.17, while failed episodes have a mean closest distance of 8.19. Moreover, scouts observe the evader for a larger fraction of time in failed episodes than in successful episodes, 0.738 versus 0.512, suggesting that the primary bottleneck is not target visibility but the final interception efficiency of the interceptors.

## D. Reward Ablation Study

Table V presents the reward ablation results. All algorithms retain the same option layer and architecture. Removing the rule reward or role reward reduces the performance, and removing both performs worst, confirming that rule and role rewards are effective.

Fig. 4 shows the reward curves of the ablation experiment during training. The reward degradations in the initial phase reflect the disruption caused to the prior policy by exploratory residual actions, while the drops around 300k steps are related to harder curriculum learning settings. Overall, the OGR-MASAC algorithm incorporating the full reward function maintains good training quality and successfully converges.

## E. Zero-Shot Real-Map Transfer with AIS-Informed Traffic

The zero-shot real-map transfer experiment uses a map of Xiazhimen in Zhejiang Province of China, where the OGR-

![](images/27a2a81559edb719c1fb680ea68d4a762e14c6bcef7c2fca5a77f7a0de573e65.jpg)

Fig. 4. Training reward curves for the reward-ablation variants. The full reward maintains the most stable improvement by 700k training steps.  
![](images/a667bfd43d6c3fe0b6e6fa204fbff50305190f9f216c5766496bc3ff3d6b5fc0.jpg)  
Fig. 5. A successful capture on the AIS-informed real map of Xiazhimen.

MARL algorithms are evaluated without retraining. The shores and anchorage areas are drawn in QGIS, while the traffic data is derived from AIS data [19], [20]. We consider four cargo vessels from AIS-C1 to AIS-C4 as dynamic obstacles. A successful capture is shown in Fig. 5.

We evaluate OGR-MASAC over 100 test episodes on 15 fixed seeds without retraining. OGR-MASAC achieves 66.67% of SR with [41.7, 84.8]% of CI. Successful captures finish in 31.65±11.68 s, while escaped episodes end in 126.30±2.71 s. Although OGR-MASAC achieves promising results, the SR in realistic maps is lower compared to the simulation experiments on the abstract map. This performance gap during the zeroshot real-map transfer is attributed to factors such as curved coastlines, irregular port layouts, irregularly moving cargo vessels, and initial position offsets.

## V. CONCLUSION

This paper proposes OGR-MARL, an algorithm-agnostic option-guided residual MARL framework for heterogeneous USV cooperative pursuit in constrained port waterways. Experiments demonstrate that it can be instantiated with different continuous-control MARL algorithms. In the abstract Xiazhimen scenario, the MASAC-based instantiation, namely OGR-MASAC, achieves a 75.0% capture rate and obtains promising mission time cost, mission-effective rule compliance, and the highest heterogeneous coordination score among the tested methods. The QGIS/AIS-informed zero-shot real-map transfer experiment further demonstrates the potential generalization capability of the proposed framework without retraining.

Future directions include improving sample efficiency using world models [21] and conducting further validation in more realistic or real-world maritime environments.

## REFERENCES

[1] Z. Liu, Y. Zhang, X. Yu, and C. Yuan, “Unmanned surface vehicles: An overview of developments and challenges,” Annual Reviews in Control, vol. 41, pp. 71–93, May 2016.

[2] Z. Peng, J. Wang, D. Wang, and Q.-L. Han, “An overview of recent advances in coordinated control of multiple autonomous surface vehicles,” IEEE Transactions on Industrial Informatics, vol. 17, no. 2, pp. 732–745, Feb. 2021.

[3] X. Qu, W. Gan, D. Song, and L. Zhou, “Pursuit-evasion game strategy of USV based on deep reinforcement learning in complex multi-obstacle environment,” Ocean Engineering, vol. 273, p. 114016, April 2023.

[4] O. Khatib, “Real-time obstacle avoidance for manipulators and mobile robots,” The International Journal of Robotics Research, vol. 5, no. 1, pp. 90–98, Spring 1986.

[5] C. K. Tam, R. Bucknall, and A. Greig, “Review of collision avoidance and path planning methods for ships in close range encounters,” The Journal of Navigation, vol. 62, no. 3, pp. 455–476, June 2009.

[6] Y. Kuwata, M. T. Wolf, D. Zarzhitsky, and T. L. Huntsberger, “Safe maritime autonomous navigation with COLREGS, using velocity obstacles,” IEEE Journal of Oceanic Engineering, vol. 39, no. 1, pp. 110–119, Jan. 2014.

[7] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, “Multiagent actor-critic for mixed cooperative-competitive environments,” in Proc. Advances in Neural Information Processing Systems, 2017, pp. 6379–6390.

[8] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approximation error in actor-critic methods,” in Proc. International Conference on Machine Learning, 2018, pp. 1587–1596.

[9] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor,” in Proc. International Conference on Machine Learning, 2018, pp. 1861–1870.

[10] W. Gan, X. Qu, D. Song, and P. Yao, “Multi-USV cooperative chasing strategy based on obstacles assistance and deep reinforcement learning,” IEEE Transactions on Automation Science and Engineering, vol. 21, no. 4, pp. 5895–5910, Oct. 2024.

[11] Z. Sun, H. Sun, P. Li, and J. Zou, “Self-organizing cooperative pursuit strategy for multi-USV with dynamic obstacle ships,” Journal of Marine Science and Engineering, vol. 10, no. 5, p. 562, April 2022.

[12] W. Zhang, S. Li, L. Wang, and Y. Liu, “Efficient multi-agent reinforcement learning for pursuit-evasion game of unmanned surface vessels with a faster evader,” In 2025 9th International Conference on Robotics, Control and Automation, 2025, pp. 180-184.

[13] P. E. Hart, N. J. Nilsson, and B. Raphael, “A formal basis for the heuristic determination of minimum cost paths,” IEEE Transactions on Systems Science and Cybernetics, vol. 4, no. 2, pp. 100–107, July 1968.

[14] R. S. Sutton, D. Precup, and S. Singh, “Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning,” Artificial Intelligence, vol. 112, no. 1–2, pp. 181–211, Aug. 1999.

[15] T. Johannink, S. Bahl, A. Nair, J. Luo, A. Kumar, M. Loskyll, J. A. Ojea, E. Solowjow, and S. Levine, “Residual reinforcement learning for robot control,” in Proc. IEEE International Conference on Robotics and Automation, 2019, pp. 6023–6029.

[16] J. Achiam, D. Held, A. Tamar, and P. Abbeel, “Constrained policy optimization,” in Proc. International Conference on Machine Learning, 2017, pp. 22–31.

[17] T. I. Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control. Chichester, UK: Wiley, 2011.

[18] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum learning,” in Proc. 26th Annual International Conference on Machine Learning, 2009, pp. 41–48.

[19] Zhejiang Maritime Safety Administration, “Main public coastal routes of Zhejiang,” public notice attachment, Aug. 2021.

[20] Huatai Insurance Agency, “Zhejiang coastal route adjustment announcement,” Sep. 2021.

[21] Z.-H. Peng, S. Li, Z. Li, S. Ruan, Y. Liu, and Y. He, “From observations to events: Event-aware world models for reinforcement learning,” in The Fourteenth International Conference on Learning Representations, 2026.
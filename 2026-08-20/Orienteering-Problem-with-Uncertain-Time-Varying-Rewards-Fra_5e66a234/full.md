# Orienteering Problem with Uncertain Time-Varying Rewards: Framework and Benchmark for Everyday Service Robotics

Masafumi Endo<sup>1∗</sup>, Kohei Honda<sup>1,2</sup>, Yuu Jinnai<sup>1</sup>, Ryo Yonetani<sup>1</sup>

<sup>1</sup>CyberAgent Inc.

<sup>2</sup>Nagoya University

{endo\_masafumi, honda\_kohei, jinnai\_yu, yonetani\_ryo}@cyberagent.co.jp

## Abstract

We present the orienteering problem with uncertain timevarying rewards (OP-UTVR), a novel variant of the orienteering problem (OP). While most existing OP formulations assume rewards to be known in advance, practical applications involve uncertain and time-varying rewards, as with shifting customer demand for delivery agents. OP-UTVR relaxes this assumption by allowing agents to estimate reward dynamics from observations and forecast future rewards. This enables informed routing decisions despite stochastic reward changes and inevitable prediction errors. We address this problem using three planners that difer in planning horizon and online adaptivity, and derive theoretical bounds on their performance under reward stochasticity. We further introduce a mobile service robot benchmark for OP-UTVR, where a robot navigates among pedestrians in indoor environments. Experiments reveal trade-ofs between planning horizon and adaptivity, and demonstrate the efectiveness of long-horizon planning with online adaptation.

## 1 Introduction

Rewards in the real world are often dynamic and inherently uncertain. Imagine a taxi driver targeting a high-trafic area. Passenger demand shifts with commuting patterns, yet individual trajectories of potential riders remain stochastic and unknown. Handling uncertain reward dynamics matters even more in disaster response; it is crucial to locate those stranded within afected zones, where the probability of successful rescue diminishes over time due to the highly uncertain dynamics of the environment.

We are interested in enabling mobile agents that can plan and execute an efective path to collect such uncertain timevarying rewards across multiple destinations. A natural starting point for these selective routing problems is the orienteering problem (OP; Tsiligirides, 1984), where the objective is to find a sequence of locations to visit that maximizes accumulated rewards under resource constraints. While the OP has inspired a variety of related applications, such as persistent monitoring (Yu, Schwager, and Rus 2016), unknown terrain exploration (Peltzer et al. 2022), or search-and-rescue operations (Jorgensen and Pavone 2024), existing work assumes that rewards are static or follow dynamics known to the agent (Ma et al. 2017; Cao et al. 2024). These assumptions rarely hold in practice and create a performance gap when actual rewards vary and deviate from expectations.

This gap motivates the main contribution of this work: a novel variant of the OP named orienteering problem with uncertain time-varying rewards (OP-UTVR). Unlike existing OP variants with static or known time-varying rewards, OP-UTVR captures two challenges arising from reward uncertainty: 1) Rewards vary stochastically over time, like a crowd of potential taxi riders moving in an apparently random manner; and 2) agents are only allowed to ‘predict’ such time-varying rewards for a limited time horizon, which inherently involves prediction errors relative to actual rewards.

To solve the proposed OP-UTVR, we investigate three planning algorithms that leverage reward prediction differently, and theoretically characterize their performance. Specifically, One-Step Planner greedily selects the next location based on predicted immediate rewards. Ofline Planner predicts time-varying future rewards until the time limit, and plans an entire path that maximizes the accumulated rewards. Finally, Adaptive Online Planner also plans the entire path but replans at each visited location, using updated observations to adapt to reward uncertainty. We prove that Adaptive Online Planner achieves near-optimal performance under accurate reward estimation, with its advantage over Ofline Planner growing as reward stochasticity increases.

As an evaluation benchmark for OP-UTVR, we develop a mobile service robot benchmark that simulates a robotic agent navigating among moving pedestrians in indoor environments reconstructed from real-world 3D scans (Ramakrishnan et al. 2021; Xia et al. 2018). Figure 1 illustrates the motivating application of this benchmark, where the robot aims to maximize rewards given by pedestrian interactions for service activities such as greeting or advertising. This task naturally induces uncertain time-varying rewards: rewards change as pedestrians move, while their intentions remain unknown and future motion grows increasingly unpredictable. Our comprehensive evaluation reveals trade-ofs among the three planners between planning horizon and adaptivity to uncertain pedestrian movements. The results demonstrate the efectiveness of integrating long-horizon planning with adaptive replanning through runtime observations.

![](images/ffb83823705d95ba7cbdd48619d98b828d5f3a8bbbc47e6525dfbef2147a8751.jpg)  
Figure 1: Motivating application: Service robot benchmark for OP-UTVR. A robot plans a path to maximize pedestrian interactions within a time budget T. Vertices are gathering spots, with colors indicating visitor density (red, green, blue, white from high to empty). Rewards change as visitors move, a setting that traditional OP cannot handle.

## 2 Preliminaries

The OP addresses selecting and ordering vertices to visit, to maximize collected rewards under a time budget. We review the classical formulation and its extension to known timevarying rewards to diferentiate them from OP-UTVR.

## 2.1 Orienteering Problem with Static Rewards

An instance of OP is defined by tuple $( \mathcal { V } , \mathcal { E } , v _ { 0 } , r , d , T _ { \mathrm { m a x } } )$ Here, V is the set of vertices and $\mathcal { E }$ is the set of edges with $e ( v , v ^ { \prime } ) \in \mathcal { E }$ connecting vertices $v , v ^ { \prime } \in \mathcal { V }$ . Each vertex has an associated static reward (profit) $r : \mathcal { V } \to \mathbb { R } _ { > }$ , while each edge has an associated travel time $d : \mathcal { E }  \mathbb { N } \overline { { ( } } i . e .$ , discrete time steps). Starting from the initial vertex, $v _ { 0 } ~ \in ~ \mathcal { V }$ , the agent moves between the vertices along the edges to collect the rewards while consuming travel times. The objective of $\mathrm { O P }$ is to find a path, $i . e . ,$ , a sequence of distinct connected vertices, $V ~ = ~ ( v _ { 0 } , v _ { 1 } , . . . , v _ { n - 1 } )$ that maximizes the total rewards $R _ { \mathrm { t o t } } ( V )$ within the time budget $T _ { \mathrm { m a x } }$ as follows:

$$
\begin{array} { l } { \displaystyle R _ { \mathrm { t o t } } ( V ) = \sum _ { i = 0 } ^ { n - 1 } r ( v _ { i } ) } \\ { \displaystyle \qquad \mathrm { s . t . } \sum _ { i = 0 } ^ { n - 2 } d ( e ( v _ { i } , v _ { i + 1 } ) ) \leq { T _ { \mathrm { m a x } } } . } \end{array}\tag{1}
$$

(2)

Since rewards are static and certain, the optimal path can be planned entirely in advance. This assumption holds in applications such as tourist trip planning (Vansteenwegen,

Soufriau, and Van Oudheusden 2011), where the value of each location is predetermined.

## 2.2 Extension to Known Time-Varying Rewards

Prior work extends the OP to handle time-varying rewards, where the reward at each vertex depends on the visit time. Formally, we consider a reward function that maps a combination of vertex and time to a reward: $r _ { \mathrm { t v } } : \mathcal { V } \times \mathbb { N } \to \mathbb { R } _ { > }$ The objective is to find a path $V = ( v _ { 0 } , v _ { 1 } , \dotsc , v _ { n - 1 } )$ with corresponding visit times $\left( t _ { 0 } , t _ { 1 } , \ldots , t _ { n - 1 } \right)$ that maximizes:

$$
R _ { \mathrm { t o t } } ( V ) = \sum _ { i = 0 } ^ { n - 1 } r _ { \mathrm { t v } } ( v _ { i } , t _ { i } ) .\tag{3}
$$

Unlike the static case, the same vertex provides diferent rewards depending on when it is visited, $i . e .$ , the solution requires optimization of both vertex selection and visit timing. Ma et al. (2017) introduced this extension, while Cao et al. (2024) improved eficiency via heuristic search. However, both assume that $r _ { \mathrm { t v } } ( v , t )$ is completely known in advance. This assumption is valid only for applications where reward dynamics follow known patterns, such as factory production schedules or periodic trafic flows. When rewards arise from uncontrollable factors, such as pedestrians whose intentions are unobservable, their dynamics are inherently stochastic, which makes exact prediction impossible.

## 3 Orienteering Problem with Uncertain Time-Varying Rewards (OP-UTVR)

The proposed OP-UTVR is a novel variant of the traditional OP with two key features: 1) the rewards vary over time under uncertain dynamics that nevertheless can be predicted from observations for a short time horizon; 2) the agent can revisit the same vertices consecutively or repetitively in a solution path. A motivating example is a mobile service robot that needs to interact with as many pedestrians in the environment as possible by predicting their future locations, within its battery limit. Below, we formulate each key feature.

## 3.1 Problem Formulation

OP-UTVR consists of a tuple $( \mathcal { V } , \mathcal { E } , v _ { 0 } , { r _ { \mathrm { t v } } } , \hat { P } , P , d , T _ { \mathrm { m a x } } )$ Unlike the problems in Sec. 2.2, we tackle a more challenging case where the true time-varying reward $r _ { \mathrm { t v } }$ will remain unknown until actual time t.

While $r _ { \mathrm { t v } }$ is not given, we assume two properties that the agent can exploit. First, we assume that the underlying reward dynamics $P$ is a Markov chain over vertices V. That is, $r _ { \mathrm { t v } }$ moves between vertices following a transition matrix $P . ^ { \mathrm { i } }$ Specifically, let $P \in ( \Delta ^ { | \mathcal { V } | - 1 } ) ^ { | \mathcal { V } | }$ be the true (and unknown) row-stochastic transition matrix, where $p _ { u , \imath }$ represents the probability of reward transitioning from vertex u to vertex v. Let $\mathbf { r } _ { \mathrm { t v } } : \mathbb { N } \to \mathbb { R } _ { > } ^ { 1 \times | \mathcal { V } | }$ represent the true reward value at each vertex at timestep $t . \mathbf { r } _ { \mathrm { t v } }$ evolves following the underlying transition matrix $P ,$ , and its expectation satisfies:

$$
\mathbb { E } [ { \bf r } _ { \mathrm { t v } } ( t + 1 ) ] = { \bf r } _ { \mathrm { t v } } ( t ) P .\tag{4}
$$

The other assumption is that the agent has access to an estimate of the transition matrix, $\hat { P } \approx P . \hat { P }$ can be obtained in any way; in our benchmark, we estimate it from prior observations (Sec. 5).

The agent may use $\hat { P }$ to estimate $r _ { \mathrm { t v } }$ in future using Eq. (4). Let $\hat { r } _ { \mathrm { t v } }$ denote the time-varying reward estimated by ${ \hat { P } } .$ Then, the estimated future reward $\hat { \mathbf { r } } _ { \mathrm { t v } } ( t )$ at timestep t can be computed by multiplying $\hat { P }$ for t times to the observation of $\mathbf { r } _ { \mathrm { t v } } ( 0 )$ , which is available to the agent at the beginning of the traversal.

$$
\mathbb { E } [ \hat { \mathbf { r } } _ { \mathrm { t v } } ( t ) | \mathbf { r } _ { \mathrm { t v } } ( 0 ) ] = \mathbf { r } _ { \mathrm { t v } } ( 0 ) \hat { P } ^ { t } .\tag{5}
$$

Yet with uncertainty, this enables the agent to make its decisions with estimates of the expected future reward.

To summarize, unlike existing time-varying reward OP problems, the agent in OP-UTVR knows neither $r _ { \mathrm { t v } }$ nor $P$ in advance; it only observes ${ \bf r } _ { \mathrm { t v } } ( t )$ at each timestep t and has access to the estimate $\hat { P }$ for reward prediction.

## 3.2 Solution

The challenge of OP-UTVR requires the solutions to be formatted diferently from prior formulations in two ways.

First, we allow the agent to make decisions online. In OP-UTVR, an agent may benefit from adapting to unexpected outcomes during the traversal. While $r _ { \mathrm { t v } } ( \cdot , t )$ is unknown in advance, we assume the agent observes the rewards at all vertices at timestep t. This also leads to an update to the estimate of the future reward. At timestep 0, the estimated reward at timestep t is computed as in Eq. (5). However, when the agent is at timestep $t _ { \mathrm { n o w } } .$ , it may update its estimate using the observation of $\mathbf { r } _ { \mathrm { t v } } ( t _ { \mathrm { n o w } } )$ , which is likely to be more informative than the observation at previous timesteps:

$$
\mathbb { E } [ \hat { \mathbf { r } } _ { \mathrm { t v } } ( t ) \mid \mathbf { r } _ { \mathrm { t v } } ( t _ { \mathrm { n o w } } ) ] = \mathbf { r } _ { \mathrm { t v } } ( t _ { \mathrm { n o w } } ) \hat { P } ^ { t - t _ { \mathrm { n o w } } } .\tag{6}
$$

Thus, we allow the agent to choose its path adaptively during traversal in OP-UTVR. Concretely, the solution ofOP-UTVR is a policy π that computes the next vertex to visit given the current circumstances (e.g., current agent and reward positions, remaining timesteps).

The other point to consider is that, in OP-UTVR, the agent does not have to visit a new previously unvisited vertex to obtain a new reward. Because the rewards are also moving, the agent may obtain rewards from visiting the same vertex more than once. For example, the agent may choose to visit the same vertex consecutively to wait at the current posit ${ \mathrm { i o n } } , e . g . , V = ( v , v , v , v ^ { \prime } , \ldots )$ or repetitively $e . g .$ $V = ( v , v ^ { \prime } , v ^ { \prime \prime } , v , \ldots )$ . For the sake of simplicity, we denote d to represent both the travel time and the wait time $( i . e .$ , visiting the same vertex consecutively). We assume $d ( e ( u , u ) ) \bar { > } 0$ as otherwise the agent may visit the same vertex infinitely many times.

These modifications accommodate applications where rewards are not strictly one-time payments upon the first visit, but are instead accrued continually while occupying a vertex. This efectively models applications such as digital advertisements in shopping malls, where viewing duration by visitors is critical as time-varying rewards, and diferent visitors may arrive at the same location at diferent time intervals. Our experiments in Sec. 5 consider an agent in such scenarios.

The objective of the agent $R _ { \mathrm { t o t } } ( \pi )$ is to maximize the expected total reward obtained by its policy π:

$$
R _ { \mathrm { t o t } } ( \pi ) = \underset { V \sim \pi } { \mathbb { E } } [ R _ { \mathrm { t o t } } ( V ) ] = \underset { V \sim \pi } { \mathbb { E } } [ \sum _ { i = 0 } ^ { n - 1 } r _ { \mathrm { t v } } ( v _ { i } , T _ { i } ) ] ,\tag{7}
$$

where $\begin{array} { r } { T _ { i } = \sum _ { j = 0 } ^ { i - 1 } d ( e ( v _ { j } , v _ { j + 1 } ) ) } \end{array}$ is the timestep when the agent arrives at vertex $v _ { i } , V$ is a path sampled according to the decisions of $\pi ,$ and n is the length of ${ \bar { V } } .$ The objective function $\left( \operatorname { E q . } \left( 7 \right) \right)$ is equivalent to Eq. (3) but with expectation over $V ,$ , which depends on the choice of π.

Note that OP-UTVR with known dynamics can be cast as a Markov decision process (MDP; Puterman, 2014).

Lemma 1. Assume ${ \hat { P } } = P .$ . An instance of OP-UTVR is an instance ofan MDP.

The optimal policy of this MDP can in principle be computed in advance by dynamic programming. This requires the true transition matrix $P ,$ which is unavailable in OP-UTVR.

## 4 Planning Algorithms

As illustrated in Fig. 2, we investigate three planning algorithms to solve OP-UTVR, named One-Step Planner, Offline Planner, and Adaptive Online Planner. These planners involve a policy $\pi$ that selects a next vertex, i.e., $v _ { i + 1 } = \pi ( v _ { i } , \ldots )$ , and difer in how they utilize predicted time-varying rewards under uncertainty. In addition to these planners, Sec. 4.4 provides novel performance guarantees under the stochastic reward dynamics of OP-UTVR.

## 4.1 One-Step Planner

One-Step Planner π<sup>one-step</sup> decides the next vertex to visit upon arrival at each vertex $v _ { i } ,$ , by greedily maximizing the predicted reward gained only at the next vertex without considering any future rewards (Fig. 2A).

$$
\begin{array} { r l r } & { } & { \pi ^ { \mathrm { o n e - s t e p } } ( v _ { i } , { \bf r } _ { \mathrm { t v } } ( T _ { i } ) ) = \underset { u \in \mathcal { V } } { \arg \operatorname* { m a x } } \hat { r } _ { \mathrm { t v } } \left( u , d ( e ( v _ { i } , u ) ) + T _ { i } \right) } \\ & { } & { = \underset { u \in \mathcal { V } } { \arg \operatorname* { m a x } } \big [ { \bf r } _ { \mathrm { t v } } ( T _ { i } ) \hat { P } ^ { d ( e ( v _ { i } , u ) ) } \big ] _ { u } , ( \ S } \end{array}\tag{8}
$$

where $[ \mathbf { x } ] _ { u }$ denotes the entry of x corresponding to vertex u.

## 4.2 Ofline Planner

Ofline Planner $\pi ^ { \mathrm { o f f i n e } }$ computes a fixed plan in advance, i.e., a sequence of possibly overlapping vertices $V _ { 0 } ^ { \ast } = ( v _ { 0 } , v _ { 1 } , \ldots )$ maximizing the expected total reward under the time budget. Then, it traverses vertices following $V _ { 0 } ^ { * }$ , regardless of the observations during execution (Fig. 2B).

$$
\begin{array} { r l } & { \pi ^ { \mathrm { o f f i n e } } ( v _ { i } , T _ { i } , \mathbf { r _ { t v } } ( 0 ) ) = v _ { i + 1 } , } \\ & { \mathrm { w h e r e } \ V _ { 0 } ^ { * } = ( v _ { 0 } , v _ { 1 } , \ldots ) = \arg \underset { V } { \operatorname* { m a x } } \mathbb { E } [ \sum _ { j = 0 } ^ { n - 1 } \hat { r } _ { \mathrm { t v } } ( v _ { j } , T _ { j } ) | \mathbf { r _ { t v } } ( 0 ) ] } \end{array}\tag{9}
$$

Ofline Planner thus relies on the estimate of Eq. (5) and returns an optimal solution for a traditional OP with known static rewards. However, it is not necessarily optimal for OP-UTVR as rewards move nondeterministically over time. In addition, the probabilistic transition model $\hat { P }$ is an estimate and is not the ground truth P. The estimation error accumulates over time steps, so the estimate would be increasingly unreliable over time. These issues motivate us to deploy an online approach that can adapt during execution.

![](images/1c68cfde4077be02742e0cb2bec24e5f2476d37552a96b88d39c027c06533847.jpg)  
Figure 2: Solution overview. Left: observations (top row) refine predictions over time (lower rows). Badges A–C mark the information each planner uses. Right: (A) One-Step Planner maximizes immediate reward, (B) Ofline Planner follows a precomputed multi-step plan, (C) Adaptive Online Planner updates both predictions and plans as observations arrive.

## 4.3 Adaptive Online Planner

The shortcoming of Ofline Planner is that it only uses the observation of the reward $\mathbf { r } _ { \mathrm { t v } } ( 0 )$ at timestep 0 to estimate the total reward using Eq. (5) and decide the entire plan despite observing the true reward $( \mathbf { r } _ { \mathrm { t v } } ( t ) )$ in the course of travel at timestep t. Instead, Adaptive Online Planner replans at each vertex arrival and estimates the total reward of the plan with the updated information of the reward $( \mathbf { r } _ { \mathrm { t v } } ( t ) )$ using Eq. (6):

$$
\begin{array} { l } { { \displaystyle \pi ^ { \mathrm { a d a p t } } ( v _ { i } , T _ { i } , { \bf r } _ { \mathrm { t v } } ( t ) ) = v _ { i + 1 } } , \ ~ } \\ { { \displaystyle \mathrm { w h e r e } V _ { i } ^ { * } = ( v _ { i } , v _ { i + 1 } , . . . ) = \arg \operatorname* { m a x } _ { V } \mathbb [ \sum _ { j = i } ^ { n - 1 } \hat { r } _ { \mathrm { t v } } ( v _ { j } , T _ { j } ) | { \bf r } _ { \mathrm { t v } } ( T _ { i } ) ] } } \end{array}\tag{10}
$$

Figure 2C illustrates how Adaptive Online Planner works. Initially, the planner plans to visit vertex 1 while expecting to obtain a high reward at vertex 3 subsequently. Upon arrival at vertex 1, the planner updates the total reward $( R _ { \mathrm { a d a p t } } ^ { - } )$ with the current observation, and replans to visit vertex 2, where the reward will be higher than in vertex 3.

## 4.4 Theoretical Analysis

We present the first analysis that quantifies how prediction errors and reward stochasticity afect each planner. See Appendix A for the proofs.

First, Adaptive Online Planner achieves the optimal policy $\pi ^ { * }$ of the MDP in Lemma 1 if $\hat { P }$ equals the true P.

Theorem 1. If the estimated reward transition $\hat { P }$ has no error $( i . e . , \hat { P } = P ) ,$ , Adaptive Online Planner achieves the optimal solution.

This result implies that the dificulty of OP-UTVR reduces to model error, as replanning itself loses no optimality.

The challenge of OP-UTVR is that the reward transition $P$ is not given to the agent, and it has to be learned from demonstrations. Still, assuming the demonstrations are i.i.d. samples from the true distribution $P ,$ , we can bound the suboptimality of Adaptive Online Planner as follows.

Theorem 2. Assume the demonstrations are i.i.d. samples from P. Let $r _ { \mathrm { m a x } }$ be the maximum reward obtainable at any vertex. Then, the expected total reward of Adaptive Online Planner using the empirical distribution as the estimated reward transition $\hat { P }$ is bounded suboptimal as:

$$
| R _ { \mathrm { t o t } } ( \pi ^ { * } ) - R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) | \le 2 r _ { \mathrm { m a x } } T _ { \mathrm { m a x } } \sqrt { \frac { \ln { ( 2 | \mathcal { V } | ^ { 2 } / \delta ) } } { 2 | D | } }\tag{11}
$$

with probability at least $1 - \delta ,$ , where $| D |$ is the number of transition samples.

The bound shrinks with more transition samples. Reward transition estimation is thus not a fundamental obstacle in OP-UTVR, unlike the reward stochasticity itself.

The performance of Ofline Planner compared to Adaptive Online Planner degrades when the reward transition $P$ has stochasticity, as a fixed plan commits to expected rewards.

Theorem 3. Assume ${ \hat { P } } = P .$ Let $r _ { \mathrm { m a x } }$ be the maximum reward obtainable at any vertex. Then,

$$
R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) - R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { o f f i n e } } ) \le r _ { \operatorname* { m a x } } T _ { \operatorname* { m a x } } \cdot \frac { \sqrt { N \ln | \mathcal { V } | } } { 2 } ,\tag{12}
$$

where N is the number ofreward objects with nondeterministic transitions.

Here, a reward object is an individual reward source that moves between vertices according to P. Theorem 3 suggests that the advantage of adaptation grows with reward stochasticity. Section 5 examines this relationship empirically.

## 5 Experiments

We develop a mobile service robot simulation to benchmark OP-UTVR. The simulation models a robot that navigates an environment to interact with people through activities such as displaying advertisements, gathering information, or monitoring pedestrians. The robot’s objective is to maximize the total number of interactions within a time budget. The environment contains several distinct spots where people tend to congregate. While the occupancy of each spot can be observed continuously, $e . g .$ , via surveillance camera networks (Fleuret et al. 2008), individual trajectories are complex and dificult to predict over long horizons. This setting requires the robot to handle uncertain time-varying rewards.

## 5.1 Mobile Service Robot Benchmark

Environment construction. We build our benchmark on IR-SIM (Han et al. 2026), an open-source multi-agent crowd navigation simulator. Test environments come from the HM3D (Ramakrishnan et al. 2021) and Gibson (Xia et al. 2018) datasets, which provide scans of real-world indoor facilities such as ofices and stores (Fig. 3). We evaluate planners on ten indoor environments, each 25 m $\times \ 2 5$ m. Each environment has 13 gathering spots, i.e., the vertices $\nu$ of the OP-UTVR tuple. Our formulation maps each pedestrian position to exactly one vertex (Sec. 3.1). We thus sample random points in the free space and define each gathering spot as the Voronoi region (Aurenhammer and Klein 2000) around each point (Fig. 5). For all pairs of gathering spots, we compute grid-based shortest paths using $\mathbf { A } ^ { * }$ search, which are used during navigation for both the robot and pedestrians. We generate ten vertex placements per environment such that all spot pairs are connected, 100 test instances in total.

Robot modeling. The robot is omnidirectional with a maximum velocity of1.2 m/s. In planning, the travel time between spots, $i . e . , d$ of the OP-UTVR tuple, is computed assuming the robot moves along the shortest path at maximum velocity. At execution time, the robot follows the path to the target spot but uses the reciprocal velocity obstacles (RVO) model (van den Berg, Lin, and Manocha 2008) to avoid collisions with pedestrians. Thus, d only approximates the true travel time, which induces uncertainty as in practical robot navigation. The simulation runs in discrete steps of $\Delta t = 0 . 5$ s, and each step updates robot control and pedestrian states.

Pedestrian modeling. Each environment contains 20 pedestrians that transition stochastically between gathering spots according to a Markov chain with transition matrix ${ \bar { P . } }$ We design $P$ by randomly selecting one spot as the “entrance” where all pedestrians start. The remaining spots are evenly partitioned into two loops, and pedestrians enter either with probability 0.5 from the entrance. Within a loop, pedestrians move to the next spot with probability 0.75, skip one spot ahead with probability 0.05, or remain at the current spot with probability 0.2. This design creates stochastic crowd behaviors conditioned on latent intent $( i . e .$ , which group each pedestrian chooses). During execution, pedestrians move to their target spot along the pre-calculated shortest path at 1.0 m/s and use RVO to avoid collisions with other pedestrians and the robot. Interactions with the robot may change pedestrian behaviors from their transition patterns.

![](images/76cfa13c82e75330ba4d089625245fd2cb3884b0eebdc3861b6355a7b42e8d66.jpg)  
Figure 3: 3D views of example indoor environments from HM3D. We use bird’s-eye view maps from these scans.

Transition matrix estimation. To predict future rewards, the robot estimates $\hat { P }$ from observed pedestrian trajectories. For each test instance, we collect 240 s of trajectory data without robot presence. Although these data follow the underlying rule defined by $P ,$ individual trajectories vary due to stochasticity. We estimate P<sup>ˆ</sup> by fitting the gathered trajectory data using least-squares optimization.

Reward dynamics modeling. The reward $r _ { \mathrm { t v } } ( v , t )$ at each vertex is the number of pedestrians within the associated Voronoi region at time t. As described in Sec. 3.2, we allow $d ( e ( v , v ) ) \bar { > } 0$ so that the robot can remain at a vertex. We set $d ( e ( v , v ) ) = 5 \mathrm { ~ s ~ }$ for all $v \in \mathcal V$ , and the robot collects $r _ { \mathrm { t v } } ( v _ { i } , t _ { i } )$ upon each arrival.

## 5.2 Evaluation Setup

Planner implementation. For Ofline Planner and Adaptive Online Planner, we compute solution paths using $\mathbf { A } ^ { * }$ search (Hart, Nilsson, and Raphael 1968) over the expected rewards estimated via ${ \hat { P } } ^ { t }$ , with rewards treated as negative costs. For each partial path, branch-and-bound pruning (Land and Doig 1960) computes an upper bound on the remaining expected reward. This bound is used as the admissible heuristic of $\mathbf { A } ^ { * }$ , and branches whose bound cannot exceed the best solution are pruned. Adaptive Online Planner re-executes this search at each arrival. In addition to the three planners, we evaluate two baselines: 1) Greedy Planner $( \pi ^ { \mathrm { g r e e d y } } )$ , which selects the next vertex with the highest current pedestrian count without predicting future rewards, and 2) Oracle Planner $( \pi ^ { \mathrm { o r a c l e } } )$ ), which uses future pedestrian distributions from collected trajectory data and provides a coarse upper bound. $\pi ^ { \mathrm { o r a c l e } }$ replaces the optimal MDP policy (Lemma 1), which requires the true $P$ and is not achievable in practice.

Evaluation metrics. The following metrics assess each planner’s solution quality and computational eficiency:

• Total reward $( R _ { \mathrm { t o t } } )$ is the cumulative reward collected during execution, i.e., pedestrian encounters within $T _ { \mathrm { m a x } } .$

• Optimality gap $( \Delta _ { \mathrm { o p t } } )$ [%] is computed as $\left( R _ { \mathrm { o r a c l e } } \right. -$ $R _ { \mathrm { t o t } } ) / R _ { \mathrm { o r a c l e } } \times 1 0 0$ , where $R _ { \mathrm { o r a c l e } }$ is the total reward of $\pi ^ { \mathrm { o r a c l e } }$ . This measures proximity to $\pi ^ { \mathrm { o r a c l e } }$ , though it is not a strict upper bound due to execution uncertainty.

• Computation time (CT) [s] measures computational cost as total planning time in the course of execution.

![](images/709832c035d9c535c6a51c8cbf5baf1ab77888bf54d0a088c741630659b8454c.jpg)  
(a) Total Reward

![](images/cceb548c053dfca82141f5395d24233e267387a4ca47635cecc3c1d874bdcee9.jpg)  
(b) Optimality Gap

![](images/497a96ac4908050c6a595fc28de82d9b2f80710f0bfd3c7651c9806552c2198d.jpg)  
(c) Reward vs. Computation  
Figure 4: Statistical performance comparison of five planners across 100 instances (10 vertex placements × 10 maps). (a) Total reward. (b) Optimality gap relative to $\bar { \pi } ^ { \mathrm { o r a c l e } }$ . (c) Reward-computation tradeof, with points as instances and circles as means.

## 5.3 Results

Quantitative results. Figure 4 summarizes the performance of five planners across 100 test instances. π<sup>oracle</sup> achieves the highest total reward because it plans with true future pedestrian distributions (Fig. 4(a)). $\pi ^ { \mathrm { g r e e d y } }$ , which relies on current observations, achieves the lowest total reward. All three proposed planners that leverage predicted reward dynamics outperform this baseline. This result validates our formulation, as planning with predicted reward dynamics is efective even when true dynamics remain unknown.

Among the three planners, $\pi ^ { \mathrm { a d a p t } }$ achieves the highest total reward in 64 cases, compared to 25 for $\pi ^ { \mathrm { o f f i n e } }$ and 11 for $\pi ^ { \mathrm { { o n e - s t e p } } }$ . Figure 4(b) shows their optimality gaps relative $\scriptstyle { \mathrm { t o ~ } } \pi ^ { \mathrm { o r a c l e } }$ . π<sup>one-step</sup> has a 22.5% gap due to myopic decisionmaking, despite using updated observations. ${ \dot { \pi } } ^ { \mathrm { o f f i n e } }$ reduces the gap to 8.9% by planning over the full time horizon, but its fixed plan cannot adapt to growing prediction errors. $\pi ^ { \mathrm { a d a p t } }$ approaches the oracle with a 2.1% gap, as replanning keeps predictions up to date and prevents error accumulation (pairwise Wilcoxon signed-rank, Holm-corrected $p \ < \ 0 . 0 0 1 )$ . These results suggest that efective planning combines longhorizon planning with runtime prediction updates.

These performance diferences come with tradeofs in computational costs (Fig. 4(c)). π<sup>one-step</sup> requires minimal computation as it only evaluates the immediate rewards of candidate next vertices. $\pi ^ { \mathrm { o f f i n e } }$ involves moderate computation by solving the planning problem once at the start. $\pi ^ { \mathrm { a d a p t } }$ demands the highest computation due to replanning, but achieves near-oracle performance. Beyond these general trends, we next examine individual planner behaviors.

Qualitative results. Figure 5 compares trajectories of the three planners across three test instances, with snapshots at $t = 6 0$ and 180 s. In Fig. 5(a), π<sup>adapt</sup> achieves a total reward of 184, while $\pi ^ { \mathrm { o f f i n e } }$ and $\pi ^ { \mathrm { o n e - s t e p } }$ obtain only 118 and 121, respectively (final values at $t = 2 4 0 \mathrm { s } ; \mathrm { F i g } . 5$ shows snapshots). $\pi ^ { \mathrm { o f f i n e } }$ repeatedly visits a few vertices, as its initial plan cannot adapt to reward dynamics. Its predictions also converge over time, so moving ofers little expected gain. $\pi ^ { \mathrm { o n e - s t e p } }$ responds to current observations but fails to collect high $R _ { \mathrm { t o t } }$ due to optimizing only the next step. $\pi ^ { \mathrm { a d a p t } }$ visits diverse vertices where pedestrians actually gather, by replanning with updated observations while optimizing over

multiple steps.

Analysis of environmental efects. We quantify the relationship between environment geometry and planner performance for all instances using two metrics: free space ratio (0.34–0.53 across maps) and reward variance (temporal variance per vertex). Figure 6 shows that higher free space ratio correlates with greater reward variance (Pearson correlation, $r = 0 . 3 6 , p < 0 . 0 0 1 )$ , which increases the improvement of $\pi ^ { \mathrm { a d a p t } }$ over π<sup>ofline</sup> $( r = 0 . 2 9 , p < 0 . 0 1$ , measured as $( R _ { \mathrm { a d a p t } } - R _ { \mathrm { o f f i n e } } ) / R _ { \mathrm { o f f i n e } } \times 1 0 0 ~ [ \% ] )$ . Open spaces allow pedestrians to move freely, and rewards vary more over time. $\pi ^ { \mathrm { o f f i n e } }$ follows a fixed plan and cannot adapt to these shifts, while $\pi ^ { \mathrm { a d a p t } }$ updates its plan with new observations. Figure 5(b) and (c) illustrate this trend: $\pi ^ { \mathrm { a d a p t } }$ leads in open spaces (194 vs. 135) while π<sup>ofline</sup> outperforms in constrained spaces (241 vs. 193) (final values at $t = 2 4 0 \ : \mathrm { s ) }$

## 6 Related Work

OP and its variants. The OP selects and sequences vertex visits to maximize collected rewards under budget constraints (Tsiligirides 1984). As an NP-hard problem, OP has been extensively studied in operations research in terms of optimality (Fischetti, González, and Toth 1998), computational eficiency (Tang and Miller-Hooks 2005; Wang, Golden, and Wasil 2008), and scalability (Boussier, Feillet, and Gendreau 2007; Dang, Guibadj, and Moukrim 2013). Its abstract formulation readily extends to variants such as time-window constraints (Kantor and Rosenwein 1992) and team orienteering (Chao, Golden, and Wasil 1996).

Robotic applications. This flexibility makes OP wellsuited for robotics, where agents maximize task completion under resource constraints such as time or energy. For example, OP is extended to capture spatial correlations for environmental monitoring (Yu, Schwager, and Rus 2016), as nearby locations provide similar measurements. Peltzer et al. (2022) incorporates information gain into OP to prioritize areas that reduce map uncertainty for unknown terrain exploration. For search-and-rescue operations (Jorgensen and Pavone 2024), OP introduces survival constraints to account for the risk of agent loss. These examples show how robotics requires extending OP beyond its classical formulation.

t = 60 s  
t = 180 s  
![](images/6df3a7ab820645c839ba0baceab7bef8bf3824f016fbea54390334af95121501.jpg)  
(a) Gibson instance

t = 60 s  
![](images/76bb0f946dd3955293097957fed3395803a326fad818e2f5620cf6aa5fa5ef85.jpg)

t = 180 s  
![](images/5dbef161fd8f5f32b4cedabe9efe0bcb8d579631a4f5a843e3445055a7173c28.jpg)  
(b) HM3D instance

t = 60 s  
![](images/b7befc04d3c3368fb2ae2a6c282fe5b6d097b366f384f45068432da559d39534.jpg)

t = 180 s  
![](images/fca95bc82f517124f7ff92e64f3ba3f276e2188f7cebd9e35abcafca9340461e.jpg)  
(c) HM3D instance

Figure 5: Snapshots of three planners for robot orienteering among pedestrians. $R _ { \mathrm { t o t } }$ denotes the total reward at each snapshot. Voronoi cell colors indicate predicted rewards (warmer for higher values), with gray dashed boundaries. Annotations: black diamonds (graph vertices), dark gray (obstacles), cyan dot (robot), trajectory cyan-to-magenta over time, green dots (pedestrians).  
![](images/3439a28b28416d5e94cf3f80e00f1ba0d06d22a10dd7a5099f3b72e3b9c33dc2.jpg)  
(a)

![](images/d59b2cf14508490663fb5ec18f8982c44f59294f6f7e10e4e84f309b3c76189c.jpg)  
(b)  
Figure 6: Environmental efect on planner performance. Point colors indicate maps. (a) Free space ratio vs. reward variance. (b) Reward variance vs. improvement of π<sup>adapt</sup> over π<sup>ofline</sup>.

Uncertain/dynamic rewards. While most OP formulations assume static or known rewards, the real world is inherently uncertain and dynamic. Prior work addresses this through stochastic formulations or time-varying reward models. The former handle uncertainty by modeling stochastic travel costs with chance constraints (Carpin 2025) or rewards with probability distributions (Ilhan, Iravani, and Daskin 2008). However, these approaches assume that uncertainty follows known, stationary distributions. In contrast, time-varying reward models capture rewards that depend on location and visit time. Ma et al. (2017) introduced this formulation and solved it via dynamic programming on a spatiotemporal graph. Cao et al. (2024) improved computational eficiency with a heuristic search that guarantees optimality without explicit state spaces. Yet both methods require complete knowledge of future reward dynamics, an assumption rarely satisfied in practice. We combine both directions by considering time-varying rewards with uncertain dynamics.

Relation to MDP. When future reward dynamics are unknown, one might formulate the problem as an MDP (Puterman 2014). While prior work has studied MDPs with changing rewards (Rivera Cardoso, Wang, and Xu 2019) or nonstationary dynamics (Cheung, Simchi-Levi, and Zhu 2020), such methods require learning both state transitions and reward functions. In our setting, agent transitions are deterministic and known, but only reward dynamics are uncertain. This allows us to learn only reward transitions rather than the entire MDP, which simplifies the learning problem.

## 7 Conclusion

We presented OP-UTVR, a novel variant of the orienteering problem where rewards change stochastically and cannot be known in advance. Unlike prior work assuming known reward dynamics, our formulation allows agents to predict future reward transitions from observations. We investigated three planning algorithms and provided theoretical bounds on their performance. We also introduced a mobile service robot benchmark using real-world 3D scans. Experiments revealed trade-ofs between planning horizon and adaptivity, and demonstrated the efectiveness of combining longhorizon planning with runtime observation updates. While we focused on Markovian dynamics in simulation, future directions include handling non-Markovian transitions, scaling to larger graphs, and validating with physical robots.

## References

Aurenhammer, F.; and Klein, R. 2000. Voronoi Diagrams, 201–290. North-Holland.

Boussier, S.; Feillet, D.; and Gendreau, M. 2007. An Exact Algorithm for Team Orienteering Problems. 4OR, 5(3): 211– 230.

Cao, C.; Xu, J.; Zhang, J.; Choset, H.; and Ren, Z. 2024. Heuristic Search for the Orienteering Problem with Time-Varying Reward. In Proceedings of the International Symposium on Combinatorial Search, volume 17, 11–19.

Carpin, S. 2025. Solving Stochastic Orienteering Problems with Chance Constraints Using Monte Carlo Tree Search. IEEE Transactions on Automation Science and Engineering, 22: 7855–7869.

Chao, I.-M.; Golden, B. L.; and Wasil, E. A. 1996. The Team Orienteering Problem. European Journal of Operational Research, 88(3): 464–474.

Cheung, W. C.; Simchi-Levi, D.; and Zhu, R. 2020. Reinforcement Learning for Non-Stationary Markov Decision Processes: The Blessing of (More) Optimism. In Proceedings of the International Conference on Machine Learning, volume 119, 1843–1854.

Dang, D.-C.; Guibadj, R. N.; and Moukrim, A. 2013. An Efective PSO-Inspired Algorithm for the Team Orienteering Problem. European Journal of Operational Research, 229(2): 332–344.

Fischetti, M.; González, J. J. S.; and Toth, P. 1998. Solving the Orienteering Problem through Branch-and-Cut. INFORMS Journal on Computing, 10(2): 133–148.

Fleuret, F.; Berclaz, J.; Lengagne, R.; and Fua, P. 2008. Multicamera People Tracking with a Probabilistic Occupancy Map. IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(2): 267–282.

Han, R.; Wang, S.; Li, C.; Gao, R.; Wang, X.; Liu, Z.; Li, G.; Lu, Y.; Hao, Q.; Pan, J.; and Zhao, H. 2026. IR-SIM: A Lightweight Skill-Native Simulator for Navigation, Learning, and Benchmarking. arXiv preprint arXiv:2606.08729.

Hart, P. E.; Nilsson, N. J.; and Raphael, B. 1968. A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Transactions on Systems Science and Cybernetics, 4(2): 100–107.

Hoefding, W. 1963. Probability Inequalities for Sums of Bounded Random Variables. Journal of the American Statistical Association, 58(301): 13–30.

Ilhan, T.; Iravani, S. M. R.; and Daskin, M. S. 2008. The Orienteering Problem with Stochastic Profits. IIE Transactions, 40(4): 406–421.

Jorgensen, S.; and Pavone, M. 2024. The Matroid Team Surviving Orienteers Problem and Its Variants: Constrained Routing of Heterogeneous Teams with Risky Traversal. The International Journal ofRobotics Research, 43(1): 34–52.

Kantor, M. G.; and Rosenwein, M. B. 1992. The Orienteering Problem with Time Windows. Journal of the Operational Research Society, 43(6): 629–635.

Land, A. H.; and Doig, A. G. 1960. An Automatic Method of Solving Discrete Programming Problems. Econometrica, 28(3): 497–520.

Ma, Z.; Yin, K.; Liu, L.; and Sukhatme, G. S. 2017. A Spatio-Temporal Representation for the Orienteering Problem with Time-Varying Profits. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems, 6785–6792.

Peltzer, O.; Bouman, A.; Kim, S.-K.; Senanayake, R.; Ott, J.; Delecki, H.; Sobue, M.; Kochenderfer, M. J.; Schwager, M.; Burdick, J.; and Agha-mohammadi, A.-a. 2022. FIG-OP: Exploring Large-Scale Unknown Environments on a Fixed Time Budget. In Proceedings ofthe IEEE/RSJ International Conference on Intelligent Robots and Systems, 8754–8761.

Puterman, M. L. 2014. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons.

Ramakrishnan, S. K.; Gokaslan, A.; Wijmans, E.; Maksymets, O.; Clegg, A.; Turner, J. M.; Undersander, E.; Galuba, W.; Westbury, A.; Chang, A. X.; Savva, M.; Zhao, Y.; and Batra, D. 2021. Habitat-Matterport 3D Dataset (HM3D): 1000 Large-Scale 3D Environments for Embodied AI. In Proceedings ofthe NeurIPS Datasets and Benchmarks Track.

Rivera Cardoso, A.; Wang, H.; and Xu, H. 2019. Large Scale Markov Decision Processes with Changing Rewards. In Advances in Neural Information Processing Systems, volume 32.

Tang, H.; and Miller-Hooks, E. 2005. A TABU Search Heuristic for the Team Orienteering Problem. Computers & Operations Research, 32(6): 1379–1407.

Tsiligirides, T. 1984. Heuristic Methods Applied to Orienteering. Journal ofthe Operational Research Society, 35(9): 797–809.

van den Berg, J.; Lin, M.; and Manocha, D. 2008. Reciprocal Velocity Obstacles for Real-Time Multi-Agent Navigation. In Proceedings of the IEEE International Conference on Robotics and Automation, 1928–1935.

Vansteenwegen, P.; Soufriau, W.; and Van Oudheusden, D. 2011. The Orienteering Problem: A Survey. European Journal ofOperational Research, 209(1): 1–10.

Wang, X.; Golden, B. L.; and Wasil, E. A. 2008. Using a Genetic Algorithm to Solve the Generalized Orienteering Problem. In The Vehicle Routing Problem: Latest Advances and New Challenges, 263–274. Springer.

Xia, F.; Zamir, A. R.; He, Z.; Sax, A.; Malik, J.; and Savarese, S. 2018. Gibson Env: Real-World Perception for Embodied Agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9068–9079.

Yu, J.; Schwager, M.; and Rus, D. 2016. Correlated Orienteering Problem and Its Application to Persistent Monitoring Tasks. IEEE Transactions on Robotics, 32(5): 1106–1118.

## A Proofs

We prove Theorems 1–3 and the supporting lemmas for the orienteering problem with uncertain time-varying rewards (OP-UTVR).

## A.1 Proof of Theorem 1

Theorem 1. If the estimated reward transition $\hat { P }$ has no error $( i . e . , \hat { P } = P ) ,$ , Adaptive Online Planner achieves the optimal solution.

Proof. To show Theorem 1, we first prove the following lemma.

Lemma 1. Assume ${ \hat { P } } = P .$ An instance of OP-UTVR is an instance ofa Markov decision process (MDP).

Proof of Lemma 1. An MDP consists of a tuple $( S , { \dot { A } } , { \dot { T } } , R )$ (Puterman 2014). An MDP representing an instance of OP-UTVR can be constructed as follows.

State space S: A state consists of the agent’s position $v \in \mathcal { V } .$ , the current reward distribution $\mathbf { r } _ { \mathrm { t v } } \in \mathbb { \breve { R } } _ { \geq } ^ { 1 \times | \nu | }$ , and the remaining time budget $t _ { \mathrm { r e m a i n } } \in [ 0 , T _ { \mathrm { m a x } } ]$

$$
s = ( v , { \bf r } _ { \mathrm { t v } } , t _ { \mathrm { r e m a i n } } ) .
$$

Action space A: The action space has size |V|, where each action corresponds to selecting a vertex to travel to next.

Transition function T: Given state $s = ( v , \mathbf { r } _ { \mathrm { t v } } , t _ { \mathrm { r e m a i n } } )$ and action $a \in \mathcal V$ , the next state $s ^ { \prime } = \left( v ^ { \prime } , \mathbf { r } _ { \mathrm { t v } } ^ { \prime } , t _ { \mathrm { r e m a i n } } ^ { \prime } \right)$ is determined as follows:

$$
v ^ { \prime } = a ,\tag{13}
$$

$$
{ \bf r } _ { \mathrm { t v } } ^ { \prime } \sim { \bf r } _ { \mathrm { t v } } P ^ { d ( e ( v , a ) ) } ,\tag{14}
$$

$$
t _ { \mathrm { r e m a i n } } ^ { \prime } = t _ { \mathrm { r e m a i n } } - d ( e ( v , a ) ) .\tag{15}
$$

The agent’s position is deterministically updated to $v ^ { \prime } = a$ The reward distribution evolves probabilistically according to the transition matrix $P$ raised to the power ofthe travel time $d ( e ( v , a ) )$ . The remaining time is decremented by the travel time. $\mathrm { I f } \ i _ { \mathrm { r e m a i n } } ^ { \prime } \le 0$ , the system transitions to an absorbing terminal state.

Reward function R: The agent receives reward $r _ { \mathrm { t v } } ( v , t )$ upon arriving at vertex v at timestep t.

The policy of this MDP corresponds exactly to the policy in OP-UTVR, and the expected total rewards of the two formulations coincide. Therefore, OP-UTVR is an instance of an MDP. □

By Lemma 1, when ${ \hat { P } } = P _ { \mathrm { \cdot } }$ , OP-UTVR is an MDP with known transition dynamics. The optimal policy of an MDP selects, at each state, the action that maximizes the expected total reward. It can be obtained via dynamic programming or policy iteration. Adaptive Online Planner defined in Sec. 4 (Eq. (10)) computes exactly this: at each state, it selects the action (next vertex) that maximizes the expected total reward given the current observations. Therefore, Adaptive Online Planner achieves the optimal solution when ${ \hat { P } } = P$ , which proves Theorem 1. □

## A.2 Proof of Theorem 2

Theorem 2. Assume the demonstrations are i.i.d. samples from $P .$ . Let $r _ { \mathrm { m a x } }$ be the maximum reward obtainable at any vertex. Then, the expected total reward of Adaptive Online

Planner using the empirical distribution as the estimated reward transition $\hat { P }$ is bounded suboptimal as:

$$
| R _ { \mathrm { t o t } } ( \pi ^ { * } ) - R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) | \le 2 r _ { \mathrm { m a x } } T _ { \mathrm { m a x } } \sqrt { \frac { \ln { ( 2 | \mathcal { V } | ^ { 2 } / \delta ) } } { 2 | D | } }\tag{16}
$$

with probability at least $1 - \delta ,$ , where $| D |$ is the number of transition samples.

Proof. To show Theorem 2, we first prove the following lemma.

Lemma 2. Suppose $\| \hat { P } - P \| _ { \infty } \leq \epsilon ,$ where $\| \cdot \| _ { \infty }$ denotes the max-norm: $\begin{array} { r } { \operatorname { \bar { \lVert } M \rVert } _ { \infty } ^ { - } = \operatorname* { m a x } _ { u , v } | M _ { u , v } | . } \end{array}$ . Let $r _ { \mathrm { m a x } }$ be the maximum reward obtainable at any vertex. Then,

$$
| R _ { \mathrm { t o t } } ( \pi ^ { * } ) - R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) | \le 2 \epsilon r _ { \mathrm { m a x } } T _ { \mathrm { m a x } } .\tag{17}
$$

ProofofLemma 2. Let $V _ { P } ^ { \pi } ( s )$ denote the expected total reward when following policy π under true dynamics $P ,$ starting from state s. Similarly, let $V _ { \hat { P } } ^ { \pi } ( s )$ denote the expected total reward under estimated dynamics $\hat { P }$

By the Bellman equation for finite-horizon MDPs, for any policy π and state $s = ( v , \mathbf { r } _ { \mathrm { t v } } , t _ { \mathrm { r e m a i n } } )$ with $t _ { \mathrm { r e m a i n } } > 0 \colon$

$$
V _ { P } ^ { \pi } ( s ) = r _ { \mathrm { t v } } ( v ) + \mathbb { E } _ { s ^ { \prime } \sim T _ { P } ( s , \pi ( s ) ) } [ V _ { P } ^ { \pi } ( s ^ { \prime } ) ] ,\tag{18}
$$

$$
V _ { \hat { P } } ^ { \pi } ( s ) = r _ { \mathrm { t v } } ( v ) + \mathbb { E } _ { s ^ { \prime } \sim T _ { \hat { P } } ( s , \pi ( s ) ) } [ V _ { \hat { P } } ^ { \pi } ( s ^ { \prime } ) ] ,\tag{19}
$$

where $T _ { P }$ and $T _ { \hat { P } }$ are the transition functions under $P$ and ${ \hat { P } } ,$ respectively.

The diference in value functions can be bounded recursively. For a single timestep:

$$
\left| V _ { P } ^ { \pi } ( s ) - V _ { \hat { P } } ^ { \pi } ( s ) \right| \leq \left| \mathbb { E } _ { s ^ { \prime } \sim T _ { P } } [ V _ { P } ^ { \pi } ( s ^ { \prime } ) ] - \mathbb { E } _ { s ^ { \prime } \sim T _ { \hat { P } } } [ V _ { \hat { P } } ^ { \pi } ( s ^ { \prime } ) ] \right|
$$

$$
\leq \left| \mathbb { E } _ { s ^ { \prime } \sim T _ { P } } [ V _ { P } ^ { \pi } ( s ^ { \prime } ) ] - \mathbb { E } _ { s ^ { \prime } \sim T _ { P } } [ V _ { \hat { P } } ^ { \pi } ( s ^ { \prime } ) ] \right|\tag{20}
$$

(21)

$$
+ \left| \mathbb { E } _ { s ^ { \prime } \sim T _ { P } } [ V _ { \hat { P } } ^ { \pi } ( s ^ { \prime } ) ] - \mathbb { E } _ { s ^ { \prime } \sim T _ { \hat { P } } } [ V _ { \hat { P } } ^ { \pi } ( s ^ { \prime } ) ] \right| .\tag{22}
$$

The first term bounds the recursive error in value estimation. The second term captures the error due to transition mismatch. Since rewards are bounded by $r _ { \mathrm { m a x } }$ , the maximum value function is bounded by $r _ { \operatorname* { m a x } } T _ { \operatorname* { m a x } } .$ . The transition error $\| { \hat { P } } - P \| _ { \infty } \leq \epsilon$ implies that the distribution over next states difers by at most ϵ in total variation distance. Therefore:

$$
\big | \mathbb { E } _ { s ^ { \prime } \sim T _ { P } } [ V _ { \hat { P } } ^ { \pi } ( s ^ { \prime } ) ] - \mathbb { E } _ { s ^ { \prime } \sim T _ { \hat { P } } } [ V _ { \hat { P } } ^ { \pi } ( s ^ { \prime } ) ] \big | \le \epsilon r _ { \mathrm { m a x } } T _ { \mathrm { m a x } } .\tag{23}
$$

Unrolling this recursion over $T _ { \mathrm { m a x } }$ steps yields:

$$
\begin{array} { r } { | V _ { P } ^ { \pi } ( s ) - V _ { \hat { P } } ^ { \pi } ( s ) | \le \epsilon r _ { \mathrm { m a x } } T _ { \mathrm { m a x } } . } \end{array}\tag{24}
$$

Now, $\pi ^ { * }$ is optimal under $P ,$ and $\pi ^ { \mathrm { a d a p t } }$ is optimal under $\hat { P }$ (by Theorem 1). Therefore:

$$
R _ { \mathrm { t o t } } ( \pi ^ { * } ) { \mathrm { - } } R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) = V _ { P } ^ { \pi ^ { * } } ( s _ { 0 } ) - V _ { P } ^ { \pi ^ { \mathrm { a d a p t } } } ( s _ { 0 } )\tag{25}
$$

$$
\leq V _ { P } ^ { \pi ^ { * } } ( s _ { 0 } ) - V _ { \hat { P } } ^ { \pi ^ { \mathrm { a d a p t } } } ( s _ { 0 } )
$$

$$
+ V _ { \hat { P } } ^ { \pi ^ { \mathrm { a d a p t } } } ( s _ { 0 } ) - V _ { P } ^ { \pi ^ { \mathrm { a d a p t } } } ( s _ { 0 } )\tag{26}
$$

$$
\leq V _ { P } ^ { \pi ^ { * } } ( s _ { 0 } ) - V _ { \hat { P } } ^ { \pi ^ { \mathrm { a d a p t } } } ( s _ { 0 } ) + \epsilon r _ { \operatorname* { m a x } } T _ { \operatorname* { m a x } } .\tag{27}
$$

Since $\pi ^ { \mathrm { a d a p t } }$ is optimal under ${ \hat { P } } ,$ we have $V _ { \hat { P } } ^ { \pi ^ { \mathrm { a d a p t } } } ( s _ { 0 } ) \geq$ $V _ { \hat { P } } ^ { \pi ^ { * } } \left( s _ { 0 } \right)$ . Thus:

$$
\begin{array} { r } { V _ { P } ^ { \pi ^ { * } } ( s _ { 0 } ) - V _ { \hat { P } } ^ { \pi ^ { \mathrm { a d a p t } } } ( s _ { 0 } ) \le V _ { P } ^ { \pi ^ { * } } ( s _ { 0 } ) - V _ { \hat { P } } ^ { \pi ^ { * } } ( s _ { 0 } ) \le \epsilon r _ { \operatorname* { m a x } } T _ { \operatorname* { m a x } } . } \end{array}\tag{28}
$$

Combining these bounds:

$$
| R _ { \mathrm { t o t } } ( \pi ^ { * } ) - R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) | \le 2 \epsilon r _ { \mathrm { m a x } } T _ { \mathrm { m a x } } .\tag{29}
$$

Let $\hat { P }$ be the empirical transition matrix estimated from |D| i.i.d. transition samples. For each pair $( u , v ) \in \mathcal { V } \times \mathcal { V }$ the empirical estimate $\hat { P } _ { u , v }$ is the sample mean of Bernoulli trials (whether a transition from u lands at v). By Hoefding’s inequality (Hoefding 1963), for any fixed $( u , v )$

$$
\begin{array} { r } { \mathrm { P r } ( | P _ { u , v } - \hat { P } _ { u , v } | > \epsilon ) \leq 2 \exp ( - 2 | D | \epsilon ^ { 2 } ) . } \end{array}\tag{30}
$$

Applying a union bound over all $| \nu | ^ { 2 }$ entries:

$$
\begin{array} { r } { \operatorname* { P r } ( \| \hat { P } - P \| _ { \infty } > \epsilon ) \le 2 | \mathcal { V } | ^ { 2 } \exp ( - 2 | D | \epsilon ^ { 2 } ) . } \end{array}\tag{31}
$$

Setting the right-hand side equal to δ and solving for ϵ:

$$
2 | \mathcal { V } | ^ { 2 } \exp ( - 2 | D | \epsilon ^ { 2 } ) = \delta\tag{32}
$$

$$
\epsilon = \sqrt { \frac { \ln ( 2 | \mathcal { V } | ^ { 2 } / \delta ) } { 2 | D | } } .\tag{33}
$$

Therefore, with probability at least $1 - \delta \colon$

$$
\| \hat { P } - P \| _ { \infty } \leq \sqrt { \frac { \ln ( 2 | \mathcal { V } | ^ { 2 } / \delta ) } { 2 | D | } } .\tag{34}
$$

Combining with Lemma 2, we obtain:

$$
| R _ { \mathrm { t o t } } ( \pi ^ { * } ) - R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) | \le 2 r _ { \mathrm { m a x } } T _ { \mathrm { m a x } } \sqrt { \frac { \ln ( 2 | \mathcal { V } | ^ { 2 } / \delta ) } { 2 | D | } } ,\tag{35}
$$

which completes the proof of Theorem 2.

## A.3 Proof of Theorem 3

Theorem 3. Assume ${ \hat { P } } = P .$ . Let $r _ { \mathrm { m a x } }$ be the maximum reward obtainable at any vertex. Then,

$$
R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) - R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { o f f i n e } } ) \le r _ { \mathrm { m a x } } T _ { \mathrm { m a x } } \cdot \frac { \sqrt { N \ln | \mathcal { V } | } } { 2 } ,\tag{36}
$$

where N is the number ofreward objects with nondeterministic transitions.

Proof. Let $\{ X _ { v } \} _ { v \in \mathcal { V } }$ be random variables representing rewards at diferent vertices at some future timestep.

Lemma 3. For Adaptive Online Planner, the expected reward per step is $\mathbb { E } [ \operatorname* { m a x } _ { v \in \mathcal { V } } X _ { v } ]$ . For Ofline Planner, the expected reward per step is at most $\operatorname* { m a x } _ { v \in \mathcal { V } } \mathbb { E } [ X _ { v } ]$ . Therefore:

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { v \in \mathcal { V } } X _ { v } \right] - \operatorname* { m a x } _ { v \in \mathcal { V } } \mathbb { E } [ X _ { v } ] \geq 0 .\tag{37}
$$

Proof of Lemma 3. Let $v ^ { * } = \arg \operatorname* { m a x } _ { v } \mathbb { E } [ X _ { v } ]$ . Then:

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { v \in \mathcal { V } } X _ { v } \right] \geq \mathbb { E } [ X _ { v ^ { * } } ]\tag{38}
$$

$$
= \operatorname* { m a x } _ { v \in \mathcal { V } } \mathbb { E } [ X _ { v } ] .\tag{39}
$$

Assume there are N independent reward objects, each following stochastic transitions according to P. Each object contributes a reward in $[ 0 , r _ { \mathrm { m a x } } ]$ when collected.

At any timestep t, let $Y _ { i } ^ { ( v ) } \in [ 0 , { r } _ { \operatorname* { m a x } } ]$ denote the reward from object i if it is located at vertex v, and 0 otherwise. The total reward at vertex v is:

$$
X _ { v } = \sum _ { i = 1 } ^ { N } Y _ { i } ^ { ( v ) } .\tag{40}
$$

Since each object transitions independently according to $P ,$ the $Y _ { i } ^ { ( v ) }$ are independent random variables. Let $\begin{array} { r l } { p _ { v } } & { { } = } \end{array}$ Pr(object i is at vertex v) be the probability that object i is at vertex v. Then:

$$
\mathbb { E } [ X _ { v } ] \leq N r _ { \operatorname* { m a x } } p _ { v } .\tag{41}
$$

For the maximum of |V| independent sums (one per vertex), we have the following lemma using a concentration inequality.

Lemma 4. For independent random variables bounded in $[ 0 , N r _ { \operatorname* { m a x } } ]$ , the gap between the expectation of the maximum and the maximum of expectations satisfies:

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { v \in \mathcal { V } } X _ { v } \right] - \operatorname* { m a x } _ { v \in \mathcal { V } } \mathbb { E } [ X _ { v } ] \leq r _ { \operatorname* { m a x } } \sqrt { \frac { N \ln | \mathcal { V } | } { 2 } } .\tag{42}
$$

ProofofLemma 4. Let $v ^ { * } =$ arg max $\mathbb { E } [ X _ { v } ]$ and let $\mu =$ $\mathbb { E } [ X _ { v ^ { * } } ]$ . For any vertex v, by Hoefding’s inequality applied to the sum $\begin{array} { r } { X _ { v } = \sum _ { i = 1 } ^ { N } Y _ { i } ^ { ( v ) } } \end{array}$ where each $Y _ { i } ^ { ( v ) } \in [ 0 , { r } _ { \operatorname* { m a x } } ]$

$$
\operatorname* { P r } ( X _ { v } - \mathbb { E } [ X _ { v } ] \geq t ) \leq \exp \left( - \frac { 2 t ^ { 2 } } { N r _ { \operatorname* { m a x } } ^ { 2 } } \right) .\tag{43}
$$

Therefore:

$$
\operatorname* { P r } \left( \operatorname* { m a x } _ { v } X _ { v } \geq \mu + t \right) \leq \sum _ { v \in \mathcal { V } } \operatorname* { P r } ( X _ { v } \geq \mu + t )\tag{44}
$$

$$
\leq \sum _ { v \in \mathcal { V } } \operatorname* { P r } ( X _ { v } - \mathbb { E } [ X _ { v } ] \geq t )\tag{45}
$$

$$
\leq | \mathcal { V } | \exp \left( - \frac { 2 t ^ { 2 } } { N r _ { \operatorname* { m a x } } ^ { 2 } } \right) .\tag{46}
$$

Let $Z = \operatorname* { m a x } _ { v } X _ { v } - \mu$ . Then:

$$
\mathbb { E } [ Z ] = \int _ { 0 } ^ { \infty } \operatorname* { P r } ( Z \geq t ) d t\tag{47}
$$

$$
\leq \int _ { 0 } ^ { \infty } \operatorname* { m i n } \left( 1 , | \mathcal { V } | \exp \left( - \frac { 2 t ^ { 2 } } { N r _ { \operatorname* { m a x } } ^ { 2 } } \right) \right) d t .\tag{48}
$$

Setting $\begin{array} { r } { t _ { 0 } = r _ { \operatorname* { m a x } } \sqrt { \frac { N \ln | \mathcal { V } | } { 2 } } } \end{array}$ as the point where the exponential becomes small:

$$
\mathbb { E } [ Z ] \leq t _ { 0 } + \int _ { t _ { 0 } } ^ { \infty } | \mathcal { V } | \exp \left( - \frac { 2 t ^ { 2 } } { N r _ { \operatorname* { m a x } } ^ { 2 } } \right) d t\tag{49}
$$

$$
\leq r _ { \operatorname* { m a x } } \sqrt { \frac { N \ln | \mathcal { V } | } { 2 } } + \mathcal { O } ( 1 )\tag{50}
$$

$$
\leq r _ { \operatorname* { m a x } } { \sqrt { \frac { N \ln | \mathcal { V } | } { 2 } } } \quad ( \mathrm { f o r ~ s u f f c i e n t l y ~ l a r g e ~ } | \mathcal { V } | ) .\tag{51}
$$

By Lemma 4, the gap per timestep is at most $r _ { \mathrm { m a x } } \cdot$ $\sqrt { N \ln { | \gamma | } } / 2$ . Over $T _ { \mathrm { m a x } }$ timesteps, the total gap is bounded by:

$$
R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) - R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { o f f i n e } } ) \le r _ { \mathrm { m a x } } T _ { \mathrm { m a x } } \cdot \frac { \sqrt { N \ln | \mathcal { V } | } } { 2 } .\tag{52}
$$

Corollary 1. IfP is deterministic, then

$$
R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { a d a p t } } ) = R _ { \mathrm { t o t } } ( \pi ^ { \mathrm { o f f i n e } } ) .\tag{53}
$$

Proof. With P being deterministic, the value of $X _ { v }$ is fixed for any trials. Thus, $\begin{array} { r } { \mathbb { E } [ \operatorname* { m a x } _ { v } X _ { v } ] = \operatorname* { m a x } _ { v } \mathbb { E } [ X _ { v } ] } \end{array}$ □

## B Experimental Details

All experiments were run on CPUs (8 CPU cores, 30 GB RAM) and require no GPU. The implementation is written in Python 3.12, and exact dependency versions are included with our code. Table 1 lists the parameters used in our experiments; see Sec. 5.1 for the design of the transition matrix P. All parameters were fixed a priori as benchmark design choices rather than tuned via hyperparameter search. These values are based on realistic service-robot settings, such as the robot and pedestrian velocities.

Table 1: Experimental parameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Environment size Number of environments (maps) Vertex placements per environment Gathering spots per environment (|V|) Number of pedestrians (N) Time budget  $\left( T _ { \mathrm { m a x } } \right)$  Simulation timestep (∆t) Waiting time  $( d ( e ( v , v ) ) )$  Robot maximum velocity Pedestrian velocity</td><td> $2 5 \mathrm { m } \times 2 5 \mathrm { m }$  10 10 13 20 240 s 0.5 s 5s 1.2 m/s 1.0 m/s 0.75 / 0.05 / 0.2 0.5</td></tr></table>
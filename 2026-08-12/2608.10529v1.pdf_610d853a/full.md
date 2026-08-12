# Robust Multi-Agent Bandits with Heavy-Tailed Rewards and Information Asymmetry

Daphne Feng<sup>∗</sup>, Ricardo Parada<sup>†</sup>, Lily Jiang<sup>∗</sup>, Sophia Yi<sup>∗</sup>, William Chang<sup>∗</sup>

<sup>∗</sup>Department of Mathematics, University of California, Los Angeles, Los Angeles, CA, USA

<sup>†</sup>Department of Mathematics, University of California, Riverside, Riverside, CA, USA

Abstract—The multi-armed bandit problem is a central framework in sequential decision-making, extensively studied under sub-Gaussian reward assumptions. However, real-world applications often involve heavy-tailed reward distributions and decentralized, information-asymmetric interactions. We study multiagent multi-armed bandits with heavy-tailed rewards under three information-asymmetry regimes: unobserved actions with common rewards, observed actions with independent rewards, and unobserved actions with independent rewards. We develop robust decentralized algorithms for each setting and derive regret guarantees that nearly match centralized heavy-tailed rates. Experiments on a Pareto-distributed reward environment validate our theoretical findings and illustrate the trade-offs between synchronization, coordination, and exploration across the three regimes.

Index Terms—Multi-armed bandits, heavy-tailed rewards, decentralized learning, information asymmetry

## I. INTRODUCTION

The multi-armed bandit (MAB) problem is a core model for sequential decision-making under uncertainty, originating in work on adaptive experimentation and Bayesian selection [1], [2]. At each round a learner selects an action and observes a random payoff, balancing exploration of uncertain actions against exploitation of apparently good ones. Bandit models underpin data-driven decision systems—online experimentation, recommendation, resource allocation, spectrum access, multi-robot coordination—which are typically distributed in ways not captured by single-agent abstractions: several agents learn simultaneously while each observes only part of the system state.

The multi-player MAB (MMAB) literature spans several information structures. In one line, players share information over communication graphs or gossip protocols [3], [4]. Another has players choose from a common arm set where collisions couple outcomes [5], [6]. More recently, cooperative MMAB has been studied under limited or no communication with structured observation asymmetries [7], [8]; see [9] for a survey. These works show that even without explicit messaging, agents can sometimes coordinate through shared structure or pre-agreed protocols.

A largely orthogonal challenge is that many reward signals are heavy-tailed: rare extreme events dominate observations, producing weak concentration and rendering sub-Gaussian analyses inaccurate. Heavy tails arise naturally in financial returns, network traffic bursts, and outlier-prone performance metrics. Robust algorithms for heavy-tailed rewards include robust-UCB methods [10] and deterministic exploration– exploitation schedules [11], with extensions to pure exploration [12], linear bandits [13], and minimax-optimal procedures [14], while Catoni-style confidence sequences sharpen what is achievable under weak moment assumptions [15].

TABLE I  
SUMMARY OF INFORMATION STRUCTURES AND REGRET BOUNDS.
<table><tr><td></td><td>Prob. A</td><td>Prob. B</td><td>Prob. C</td></tr><tr><td>Actions observed?</td><td>No</td><td>Yes</td><td>No</td></tr><tr><td>Rewards shared?</td><td>Yes</td><td>No (i.i.d.)</td><td>No (i.i.d.)</td></tr><tr><td>Algorithm</td><td>mRUCB-A</td><td>mRUCB-Int.</td><td>mHT-DSEE</td></tr><tr><td>Regret</td><td> $O \left( \frac { \log T } { \Delta ^ { 1 / \varepsilon } } \right)$ </td><td> $O \left( \frac { \log T } { \Delta ^ { 1 / \varepsilon } } \right)$ </td><td> $O ( \log ^ { 2 } T )$ </td></tr></table>

The intersection of cooperative multi-agent bandits, heavytailed rewards, and decentralized operation with no online communication remains underexplored. Existing multi-agent heavy-tailed work relies on explicit communication: [16] considers delayed message passing and [17] studies graphbased communication. We ask: what is achievable when agents coordinate implicitly via a pre-agreed protocol?

Our contributions. We introduce three problem formulations capturing distinct information asymmetries in multiagent heavy-tailed bandits: common rewards with unobserved actions (Problem A), independent rewards with observed actions (Problem B), and independent rewards with unobserved actions (Problem C). For each we develop a robust decentralized algorithm—mRUCB-A, mRUCB-Intervals, and mHT-DSEE—and prove regret guarantees summarized in Table I. The robust mean estimator and the single-agent concentration arguments are adapted from [10], [11]; our contribution lies in the multi-agent formulation, in the use of intentional action deviations as an implicit signaling channel, and in a unified comparison of what each information structure costs.

## II. PRELIMINARIES

## A. Heavy-tailed bandits

Consider a stochastic MAB with K arms. Each arm ${ \textbf { \em a } } \in$ $\mathcal { A } : = \{ 1 , \ldots , K \}$ has an unknown reward distribution $\nu _ { a }$ with mean $\mu _ { a }$ . At round t, the agent selects arm $\mathbf { } \mathbf { a } _ { t }$ and observes a reward drawn from $\nu _ { a _ { t } }$ . The expected regret at horizon $T$ is

$$
R _ { T } = T \mu ^ { \star } - \sum _ { t = 1 } ^ { T } \mathbb { E } [ \mu _ { a _ { t } } ] = \sum _ { a \in \mathcal { A } } \Delta _ { a } \mathbb { E } [ n _ { a } ( T ) ] ,\tag{1}
$$

where $\begin{array} { r } { \mu ^ { \star } = \operatorname* { m a x } _ { a } \mu _ { a } , \Delta _ { a } = \mu ^ { \star } - \mu _ { a } } \end{array}$ is the suboptimality gap, and $n _ { a } ( T )$ is the number of pulls. We assume heavytailed rewards: there exist $\varepsilon \in ( 0 , 1 ]$ and $v > 0$ such that for all $a \in { \mathcal { A } } .$

$$
\mathbb { E } [ | X _ { a } - \mu _ { a } | ^ { 1 + \varepsilon } ] \leq v .\tag{2}
$$

This allows distributions with infinite variance (when $\varepsilon < 1 )$ capturing Pareto, Student-t, and other heavy-tailed families; smaller ε corresponds to heavier tails.

## B. Multi-agent extension

We extend the setting to M players, where player i has an individual action set $A _ { i }$ of size $K _ { i }$ . The joint action space is $\mathcal { A } = \mathcal { A } _ { 1 } \times \cdot \cdot \cdot \times \mathcal { A } _ { M }$ , containing $\begin{array} { r } { K ^ { M } : = \prod _ { i = 1 } ^ { M } K _ { i } } \end{array}$ joint arms. At each round $t ,$ each player simultaneously selects an arm, forming joint arm $\pmb { a } ( t ) = ( a _ { 1 } ( t ) , \ldots , a _ { M } ( t ) )$ , and then observes a reward sampled from $\nu _ { a ( t ) }$ . The cumulative regret is $\begin{array} { r } { R _ { T } = T \mu ^ { \star } - \sum _ { t = 1 } ^ { T } \mathbb { E } [ X _ { \pmb { a } ( t ) } ] } \end{array}$ , where $\mu ^ { \star } = \operatorname* { m a x } _ { \pmb { a } \in \mathcal { A } } \mu _ { \pmb { a } } .$ Players may agree on a strategy beforehand and know each other’s action spaces, but cannot communicate during learning. We consider three information structures, each matching a distinct class of deployment.

Problem A (action asymmetry). All players observe the same reward realization $X _ { a ( t ) }$ but not each other’s actions. This is the situation of a team optimizing a single aggregate metric: transmitters in a shared spectrum band that observe total network throughput, or advertising channels evaluated against one conversion count, where the aggregate is instrumented but attribution to individual actions is not.

Problem B (reward asymmetry). Players observe the joint action $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf $ but each receives an independent sample $X _ { { \pmb a } ( t ) } ^ { i } \sim$ $\nu _ { a ( t ) }$ . This matches federated or multi-site experimentation: a configuration is chosen jointly and logged centrally, so every site knows what was deployed, while each site measures only its own privately held outcomes.

Problem C (full asymmetry). Players observe neither others actions nor a common reward; each receives an i.i.d. sample. This models fully decentralized deployments such as sensor or robot teams operating with no backhaul, where each unit sees only its own measurements.

## C. Robust upper confidence bounds

Throughout, ${ \widehat { \mu } } _ { a } ( t )$ is the truncated mean of [10]: writing $X _ { a , 1 } , \ldots , X _ { a , n _ { a } ( t ) }$ for the rewards observed from arm a,

$$
\widehat { \mu } _ { a } ( t ) = \frac { 1 } { n _ { a } ( t ) } \sum _ { s = 1 } ^ { n _ { a } ( t ) } X _ { a , s } \mathbf { 1 } \bigg \{ | X _ { a , s } | \leq \left( \frac { v s } { \log ( T ^ { \gamma } ) } \right) ^ { \frac { 1 } { 1 + \varepsilon } } \bigg \} .\tag{3}
$$

The robust upper confidence bound (RUCB) for joint arm a is

$$
\mathrm { R U C B } _ { a } ( t ) = \left\{ \begin{array} { l l } { \infty } & { \mathrm { i f ~ } n _ { a } ( t ) = 0 , } \\ { \widehat { \mu } _ { a } ( t ) + \alpha _ { a } ( t ) } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{4}
$$

where the first case marks an arm from which nothing has yet been observed, so that ${ \widehat { \mu } } _ { a } ( t )$ is undefined; setting the index

```latex
Algorithm 1 mRUCB-A
1: Players agree on a lexicographic ordering of A.
2: Init. $n _ { a } ( 0 ) \gets 0 , \widehat { \mu } _ { a } ( 0 ) \gets 0$ for all $a \in { \mathcal { A } } .$
3: for $t = 1 , \dots , T$ do
4: Compute $\mathrm { R U C B } _ { a } ( t )$ for all $\mathbf { \pmb { a } } \in \mathcal { A }$ via (4).
5: Select $\begin{array} { r } { \mathbf { \boldsymbol { a } } ( t ) \gets \operatorname * { a r g m a x } _ { a } \operatorname { R U C B } _ { a } ( t ) } \end{array}$ (ties: lexico
graphic).
6: Pull individual arm; observe common reward; update
statistics.
```

to ∞ forces every joint arm to be played at least once before any comparison is made. The confidence radius is

$$
\alpha _ { \pm } ( t ) = v ^ { \frac { 1 } { 1 + \varepsilon } } \biggl ( \frac { c \log ( T ^ { \gamma } ) } { n _ { a } ( t ) } \biggr ) ^ { \frac { \varepsilon } { 1 + \varepsilon } } ,\tag{5}
$$

with $c , \gamma > 0$ , and [10, Prop. 1] gives $\operatorname* { P r } ( | \widehat { \mu } _ { a } ( t ) - \mu _ { a } | >$ $\alpha _ { a } ( t ) ) \leq t ^ { - \gamma }$ . Only this concentration property is used below, so any estimator obeying a bound of the form (5)—medianof-means, or the Catoni-style confidence sequences of [15]— may be substituted. Such a substitution changes the constant c and the way v enters, and hence the constants in all three theorems, but not the rates; Catoni-style estimators give the sharpest constants as $\varepsilon \to 1$ , at the cost of solving an implicit equation at each round.

## III. PROBLEM A: COMMON REWARDS, UNOBSERVED ACTIONS

In Problem A, all players observe the same reward but cannot see others’ actions. Two technical challenges arise. First, since actions are hidden, miscoordination may occur if the players’ internal estimates diverge, and the observed reward is then attributed to the wrong joint action. Second, the reward distributions are heavy-tailed, requiring robust estimators to control estimation error under weak moment assumptions. However, because rewards are shared, all players’ estimates remain identical under the same deterministic update rule—the key simplifying feature. We impose a lexicographic ordering on A for consistent tie-breaking: $\textbf { \textit { a } } < \textbf { \textit { b } }$ if there exists n such that $a _ { i } \ = \ b _ { i }$ for all $i < n$ and $a _ { n } < b _ { n }$ . Each player then computes $\mathrm { R U C B } _ { a } ( t )$ for every joint arm and selects the highest, breaking ties lexicographically, yielding mRUCB-A (Algorithm 1).

Theorem 1. Under condition (2), if all players $f o l -$ low mRUCB-A, the expected regret satisfies $\begin{array} { r l } { R _ { T } } & { { } = } \end{array}$ $\begin{array} { r } { O \Big ( \log ( T ) \sum _ { { \pmb { a } } \in \mathcal { A } } \Delta _ { \pmb { a } } ^ { - 1 / \varepsilon } \Big ) } \end{array}$

Proof. Since all players observe the same reward and use the same deterministic update rule with consistent tie-breaking, every player selects the same joint arm at every round. The problem thus reduces to a single-agent heavy-tailed bandit over $K ^ { M }$ arms, and the analysis follows [10]. For each suboptimal arm a with gap $\Delta _ { a } > 0$ , define the good event at round $t \colon \mathcal { G } _ { t } : | \widehat { \mu } _ { a } ( t ) - \mu _ { a } | \leq \alpha _ { a } ( t )$ for all a. By the concentration bound of Section II-C, $\Pr ( \mathcal { G } _ { t } ^ { c } ) ~ \leq ~ K ^ { M } t ^ { - \gamma }$ Under $\mathcal { G } _ { t }$ , selection of a requires $\widehat { \mu } _ { a } ( t ) + \alpha _ { a } ( t ) \geq \widehat { \mu } _ { a ^ { \star } } ( t ) +$ $\alpha _ { { \pmb a } ^ { \star } } ( t )$ , which implies $2 \alpha _ { a } ( t ) \geq \Delta _ { a }$ . This fails after $\tau _ { a } =$ $c \gamma \log ( T ) ( 2 v ^ { 1 / ( 1 + \dot { \varepsilon } ) } / \Delta _ { a } ) ^ { ( 1 + \dot { \varepsilon } ) / \varepsilon }$ pulls. Hence $\mathbb { E } [ n _ { \pmb { a } } ( T ) ] ~ \le$ $\begin{array} { r } { \tau _ { \mathbf { a } } + \sum _ { t = 1 } ^ { T } \operatorname* { P r } ( \mathcal { G } _ { t } ^ { c } ) } \end{array}$ , where the tail sum converges for $\gamma > 1$ Summing $\Delta _ { a } \cdot \mathbb { E } [ n _ { a } ( T ) ]$ ] over all suboptimal arms gives the result. □

Algorithm 2 mRUCB-Intervals   
1: Players agree on an ordering of $\mathcal { A } ;$ set $S  A , P  \emptyset$   
2: while $t \leq T$ do   
3: for each $\mathbf { \Delta } \mathbf { a } \in S$ in order do   
4: if some player i finds $I _ { a } ^ { i } ( t )$ strictly below the interval   
of another arm of S then   
5: That player pulls a different individual arm; all   
players observe $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf { } \mathbf \mathbf \mathbf \mathbf$ and set $P \gets P \cup \{ a \}$   
the reward of this round is discarded.   
6: else   
7: All players pull the components of ${ \bf \delta a } ;$ player i   
observes $X _ { a } ^ { i }$ and updates $I _ { a } ^ { i }$ via (6); $n _ { a } \gets n _ { a } + 1$   
8: $S \gets S \setminus P ; P \gets \emptyset .$

This matches the optimal single-agent heavy-tailed rate over $K ^ { M }$ arms. Since the $K ^ { M }$ dependence is unavoidable even for a centralized learner, the decentralized agents incur no additional cost from action asymmetry.

## IV. PROBLEM B: INDEPENDENT REWARDS, OBSERVED ACTIONS

In Problem B, players observe the joint action but receive independent reward samples $X _ { \pmb { a } ( t ) } ^ { 1 } , \ldots , X _ { \pmb { a } ( t ) } ^ { M } \overset { \mathrm { i . i . d . } } { \sim } \nu _ { \pmb { a } ( t ) }$ . This reverses Problem A’s structure: players see all actions but their estimates diverge because each empirical mean uses different samples, so one player may conclude that an arm is suboptimal while another player’s interval still overlaps. An index rule applied independently by each player would therefore cause persistent miscoordination. mRUCB-Intervals avoids this by replacing index maximization with round-robin elimination: players cycle through a common active set S, and an arm leaves S only through a signal that every player observes.

For each joint arm a and player i the algorithm maintains

$$
I _ { \pmb { a } } ^ { i } ( t ) = \left[ \widehat { \mu } _ { \pmb { a } } ^ { i } ( t ) - \alpha _ { \pmb { a } } ( t ) , \widehat { \mu } _ { \pmb { a } } ^ { i } ( t ) + \alpha _ { \pmb { a } } ( t ) \right] ,\tag{6}
$$

where $\alpha _ { a } ( t )$ is common to all players because it depends only on the shared pull count $n _ { a } ( t )$ . Elimination proceeds in three stages. Detection: if player i finds that $I _ { a } ^ { i } ( t )$ lies strictly below and disjoint from the interval of another active arm, then a is dominated from player $i \mathrm { \ ' } _ { \mathrm { s } }$ perspective. Signaling: player i then deviates from the prescribed action by pulling a different individual arm, the only form of implicit communication available. Propagation: since actions are observable, all players detect the mismatch between the scheduled joint arm $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf \Psi \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi$ and the realized one ${ \pmb a } ^ { \prime } ( t )$ , and mark $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \Psi \mathbf { } \mathbf \mathbf { } \mathbf \mathbf \Psi \Psi \Psi \mathbf { } \mathbf \mathbf \Psi \Psi \mathbf { } \mathbf \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \Psi \mathbf \Psi \mathbf \Psi \mathbf \Psi \Psi \mathbf \Psi $ for removal regardless of whether their own intervals support it. Two conventions keep the players’ statistics aligned: the reward of a signaling round is discarded, and removals take effect at the end of the current cycle. Algorithm 2 gives the procedure.

Lemma 1. Under mRUCB-Intervals, at every round all players hold the same active set $S ( t )$ and the same pull counts $n _ { a } ( t )$ . Moreover, if a is the arm scheduled at round t, then $n _ { b } ( t ) \geq n _ { a } ( t )$ for every $b \in S ( t )$

Proof. Both claims follow by induction on t. Initially $S = A$ and all counts are zero. The scheduled arm is a deterministic function of $S$ and the position in the cycle, which are common by hypothesis. Since actions are observed, every player sees the realized joint arm and applies the same count update, and removals are triggered only by observed deviations, so S remains common. Within a cycle each active arm is scheduled exactly once and signaling rounds increment no counts, so all active arms have equal counts at cycle boundaries and, at any point inside a cycle, the arms not yet scheduled—including the scheduled arm itself—have the smallest counts. □

Theorem 2. If all players follow mRUCB-Intervals under (2), then

$$
\begin{array} { l } { { \displaystyle R _ { T } \leq c \gamma 4 ^ { \frac { 1 + \varepsilon } { \varepsilon } } v ^ { \frac 1 \varepsilon } \log ( T ) \sum _ { a \neq a ^ { \star } } \Delta _ { a } ^ { - 1 / \varepsilon } } } \\ { { \displaystyle \quad \quad + \sum _ { a \neq a ^ { \star } } \Delta _ { a } + ( K ^ { M } - 1 ) \Delta _ { \operatorname* { m a x } } + O ( 1 ) , } } \end{array}\tag{7}
$$

where the O(1) term collects the contribution of the failure event and is independent of $T f o r \gamma > 1$

Proof. Let $\mathcal { G } _ { t }$ be the event that $| \widehat { \mu } _ { a } ^ { i } ( t ) - \mu _ { a } | \leq \alpha _ { a } ( t )$ for all i and all a; by Section II-C and a union bound over the $M K ^ { M }$ player-arm pairs, $\operatorname* { P r } ( \mathcal G _ { t } ^ { c } ) \le M K ^ { M } t ^ { - \gamma }$

Step $1 \div a ^ { \star }$ is never eliminated. Under $\mathcal { G } _ { t }$ , for every player i and every active b, the lower end of $I _ { a } ^ { i } ,$ satisfies $\widehat { \mu } _ { \pmb { a } ^ { \star } } ^ { i } + \alpha _ { \pmb { a } ^ { \star } } \geq$ $\mu ^ { \star } \geq \mu _ { b } \geq \widehat { \mu } _ { b } ^ { i } - \alpha _ { b }$ , so $I _ { a } ^ { i } .$ <sub>⋆</sub> never lies strictly below the interval of another active arm. No player signals on $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } _ { \mathbf { \alpha \beta } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } _ { \mathbf { \alpha \beta } \mathbf { \alpha } \mathbf { \alpha \beta } _ { \mathrm { \alpha \beta } \mathbf { \alpha } \mathbf { \beta } _ { \mathrm \alpha \mathbf { \beta } \mathbf { \alpha \beta } \mathbf { \alpha \beta } _ { \mathrm \beta \alpha \mathbf { \beta \alpha } \mathbf { \alpha \beta }  \frac { \beta _ { \alpha \mathbf { \beta \alpha \beta } \mathbf { \alpha \beta } \mathbf { \alpha \beta } \mathbf { \alpha \beta } _ { \beta \alpha \beta \alpha \beta } } } } } } } }$ , and by Lemma 1 no player removes it.

Step 2: Elimination time. Let $\mathbf { \pmb { a } } \neq \mathbf { \pmb { a } } ^ { \star }$ be scheduled at round t. By Lemma 1, $n _ { { \pmb a } ^ { \star } } ( t ) \geq n _ { { \pmb a } } ( t )$ and hence $\alpha _ { { \pmb a } ^ { \star } } ( t ) \leq \alpha _ { { \pmb a } } ( t )$ Under $\mathcal { G } _ { t }$ the upper end of $I _ { a } ^ { i }$ is at most $\mu _ { a } + 2 \alpha _ { a } ( t )$ and the lower end of $I _ { a } ^ { i }$ is at least $\mu ^ { \star } - 2 \alpha _ { a } ( t )$ , so every player detects domination as soon as $4 \alpha _ { a } ( t ) < \Delta _ { a } . \mathrm { B y }$ (5) this holds once $n _ { a } ( t )$ exceeds

$$
\tau _ { a } = c \gamma \log ( T ) \left( \frac { 4 v ^ { 1 / ( 1 + \varepsilon ) } } { \Delta _ { a } } \right) ^ { ( 1 + \varepsilon ) / \varepsilon } ,\tag{8}
$$

so that $n _ { \pmb { a } } ( T ) \leq \tau _ { \pmb { a } } + 1$ : the arm is signaled the next time it is scheduled and is removed at the end of that cycle. Note that the detection threshold is $4 \alpha _ { a }$ , rather than the $2 \alpha _ { a }$ of an index comparison, because separating two intervals requires both radii to be small.

Step 3: Signaling cost. Each arm is removed exactly once, and by Lemma 1 its removal consumes exactly one round in which the realized joint action is unintended, contributing regret at most $\Delta _ { \mathrm { m a x } }$ . Simultaneous deviations by several players still consume a single round, so the total signaling cost is at most $( K ^ { M } - 1 ) \Delta _ { \operatorname* { m a x } }$ , independent of both T and M.

Algorithm 3 mHT-DSEE   
1: Players agree on an ordering of A and on $w ( t ) \uparrow \infty ;$   
$N \gets 0 .$   
2: for $t = 1 , \dots , T$ do   
3: if $N < K ^ { M } \left\lceil w ( t ) \log t \right\rceil$ then   
Play the next joint arm in the cyclic order; $N \gets$   
$N + 1 ;$ each player stores its own reward.   
5: else   
6: Player i plays its own component of   
arg max<sub>a</sub> $\mathrm { \Omega } _ { \mathrm { } } \mathrm { R U C B } _ { \mathbf { } a } ^ { i } ( t )$ , computed from exploration   
samples.

Step 4: Summing. By Step 2, $\begin{array} { r } { \sum _ { a \neq a ^ { \star } } \Delta _ { a } ( \tau _ { a } + 1 ) } \end{array}$ gives the first two terms of (7). Adding the failure contribution $\begin{array} { r } { \sum _ { t < T } \mathrm { P r } ( \mathcal G _ { t } ^ { c } ) \Delta _ { \operatorname* { m a x } } \ \le \ M K ^ { M } \Delta _ { \operatorname* { m a x } } \sum _ { t > 1 } t ^ { - \gamma } \ = \ O ( 1 ) } \end{array}$ for $\gamma > 1$ , together with Step 3, yields the bound. □

The leading term matches Problem A up to the constant $4 ^ { ( 1 + \varepsilon ) / \varepsilon }$ in place of $2 ^ { ( 1 + \varepsilon ) / \varepsilon }$ , and the remaining terms are independent of the horizon. The mechanism thus uses action deviations as a 1-bit implicit communication channel; this suffices because $\alpha _ { a } ( t )$ is common across players, so a single detection is enough to eliminate an arm for everyone.

## V. PROBLEM C: INDEPENDENT REWARDS, UNOBSERVED ACTIONS

Problem C combines both asymmetries, eliminating the coordination mechanisms of Problems A (shared rewards) and B (observable actions). A player cannot tell whether the realized reward corresponds to the intended joint arm or to a different one caused by another player’s deviation, and independent samples simultaneously prevent synchronized estimates. Adaptive, index-driven coordination is therefore unavailable, and exploration must be scheduled deterministically from the round index alone, which every player can reproduce.

mHT-DSEE (Algorithm 3) follows the DSEE framework [11]. Fix an increasing $w ( t ) \to \infty$ and let $D ( t ) \ =$ ⌈w(t) log t⌉ be the target number of samples of each joint arm by round t. If fewer than $K ^ { M } D ( t )$ exploration rounds have been used, round t is an exploration round and the next joint arm in a fixed cyclic order is played; otherwise every player commits to the maximizer of its own RUCB, computed from exploration samples only. Because the test $N ( t ) < K ^ { M } D ( t )$ depends only on t, players stay synchronized without communication. Two features matter for the analysis: the schedule is anytime, requiring neither T nor the gaps, and the confidence radius uses log t rather than log T.

Theorem 3. Under (2), if all players follow mHT-DSEE with $\gamma > 1$ , then

$$
R _ { T } \leq \Delta _ { \operatorname* { m a x } } K ^ { M } D ( T ) + \Delta _ { \operatorname* { m a x } } t _ { 0 } + O ( 1 ) ,\tag{9}
$$

where $t _ { 0 } ~ = ~ \operatorname* { m i n } \{ t ~ : ~ w ( t ) ~ > ~ c \gamma ( 2 v ^ { 1 / ( 1 + \varepsilon ) } / \Delta _ { \operatorname* { m i n } } ) ^ { ( 1 + \varepsilon ) / \varepsilon } \}$ depends on ε, v and the gaps but not on T. With $w ( t ) = \lceil \log t \rceil$ this gives $R _ { T } = O ( K ^ { \bar { M } } \log ^ { 2 } T )$

Proof. Step 1: Exploration. At most $K ^ { M } D ( T )$ rounds are exploration rounds, each contributing regret at most $\Delta _ { \mathrm { m a x } } ,$ which is the first term.

Step 2: Good event. At an exploitation round t every arm has $D ( t )$ samples for each player. By Section II-C with confidence level $t ^ { - \gamma }$ and a union bound over the $M K ^ { M }$ player-arm pairs, the event $\mathcal { G } _ { t }$ that $| \widehat { \mu } _ { a } ^ { i } - \mu _ { a } | \leq \alpha ( D ( t ) )$ for all $i , \pmb { a }$ has $\operatorname* { P r } ( \mathcal G _ { t } ^ { c } ) \leq$ $M K ^ { M } t ^ { - \gamma }$ , where $\overset { \bullet } { \alpha } ( \overset { \cdot } { n } ) = v ^ { 1 / ( 1 + \varepsilon ) } \overset { \cdot } { ( } c \log ( t ^ { \gamma } ) / n ) ^ { \varepsilon / ( 1 + \varepsilon ) }$ Step 3: Coordination. On $\mathcal { G } _ { t } , \mathrm { { R U C B } } _ { a } ^ { i } \le \mu _ { a } + 2 \alpha ( D ( t ) )$ and $\operatorname { R U C B } _ { a ^ { \star } } ^ { i } \geq \mu ^ { \star }$ for every player $i ,$ so every player selects $\mathbf { \pmb { a } } ^ { \star }$ once $2 \alpha ( D ( t ) ) < \Delta _ { \mathrm { m i n } } ,$ i.e. once

$$
D ( t ) > c \gamma \log ( t ) \left( \frac { 2 v ^ { 1 / ( 1 + \varepsilon ) } } { \Delta _ { \mathrm { m i n } } } \right) ^ { ( 1 + \varepsilon ) / \varepsilon } = : \kappa \log t .\tag{10}
$$

The threshold does not depend on the player index, so the players agree; and since $D ( t ) \geq w ( t ) \log t ,$ condition (10) holds as soon as $w ( t ) > \kappa ,$ , that is for every $t \geq t _ { 0 }$ . This is the role of the anytime schedule: both sides of (10) scale with log t, so the crossing time $t _ { 0 }$ is determined by w and the gaps alone and does not grow with the horizon. Each player then plays its component of $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } _ { \mathbf { \alpha \beta } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } _ { \mathbf { \alpha \beta } \mathbf { \alpha } \mathbf { \alpha \beta } _ { \mathrm { \alpha \beta } \mathbf { \alpha } \mathbf { \beta } _ { \mathrm \alpha \mathbf { \beta } \mathbf { \alpha \beta } \mathbf { \alpha \beta } _ { \mathrm \beta \alpha \mathbf { \beta \alpha } \mathbf { \alpha \beta }  \frac { \beta _ { \alpha \mathbf { \beta \alpha \beta } \mathbf { \alpha \beta } \mathbf { \alpha \beta } \mathbf { \alpha \beta } _ { \beta \alpha \beta \alpha \beta } } } } } } } }$ , and the realized joint arm is exactly $\mathbf { \pmb { a } } ^ { \star }$

Step 4: Exploitation regret. Exploitation rounds with $t < t _ { 0 }$ contribute at most $\Delta _ { \mathrm { m a x } } t _ { 0 }$ . For $t \geq t _ { 0 } .$ , regret is incurred only on $\mathcal { G } _ { t } ^ { c }$ , contributing at most $\begin{array} { r } { M K ^ { M } \Delta _ { \operatorname* { m a x } } \sum _ { t > 1 } t ^ { - \gamma } = O ( \mathrm { i } ) } \end{array}$ for $\gamma > 1$ . Adding Step 1 gives (9). Taking $\bar { \boldsymbol { w } } ( t ) = \lceil \log t \rceil$ gives $D ( T ) = O ( \log ^ { 2 } T )$ □

Remark 1. The choice of w trades the two terms of (9) against each other: faster growth shortens $t _ { 0 }$ but enlarges the exploration budget $D ( T )$ . Any w $\uparrow \infty$ yields $o ( \log ^ { 1 + \eta } T )$ regret for the corresponding η, and knowledge of $\Delta _ { \mathrm { m i n } }$ would allow the constant schedule $w \equiv \kappa$ and hence ${ \cal O } ( \log T )$ ; the extra factor is the price of not knowing the gaps.

## VI. EXPERIMENTS

## A. Setup

We take $M = 2$ players, $K = 2$ individual arms $( K ^ { M } = 4$ joint arms), and horizon $T = 1 0 ^ { 6 }$ , averaging over 10 independent runs on the fixed instance $\pmb { \mu } = ( 0 . 4 4 , 0 . 5 7 , 0 . 9 1 , 0 . 2 5 )$ so that the gaps are $( 0 . 4 6 , 0 . 3 4 , - , 0 . 6 5 )$ . Pulling joint arm a yields a Pareto reward with shape $a _ { 0 } ~ = ~ 2$ and scale $x _ { m } = \mu _ { a } ( a _ { 0 } - 1 ) / a _ { 0 }$ , so that $\mathbb { E } [ X _ { a } ] = \mu _ { a }$ . This distribution has finite mean and infinite variance: its centered moments of order $1 + \varepsilon$ are finite exactly for $\varepsilon < 1$ . We therefore set $\varepsilon \ = \ 0 . 5 .$ , for which E $| X _ { a } - \mu _ { a } | ^ { 1 . 5 } \ \leq \ 0 . 7 5 \ < \ 1 \ = \ v$ for every $\mu _ { a } \in [ 0 , 1 ]$ , so (2) holds; taking $\varepsilon = 1$ would instead require a finite second moment and is not admissible for this reward family. All three algorithms use the truncated-mean estimator (3) with $( c , \gamma ) = ( 1 , 2 )$ , i.e. exactly the estimator the analysis assumes, and mHT-DSEE uses $w ( t ) = \lceil \log t \rceil$

## B. Results

Figure 1 shows mean cumulative regret. All three curves are clearly sublinear, confirming that each algorithm identifies $\mathbf { \pmb { a } } ^ { \star }$ under infinite-variance rewards and with the robust estimator in place. mRUCB-A ends at 214±24 and mHT-DSEE at $2 9 2 \pm 2 1$ both still growing slowly, while mRUCB-Intervals ends at $4 1 1 5 \pm 2 8 5$ but is exactly flat beyond ${ \approx } 7 \times 1 0 ^ { 4 }$ rounds.

![](images/230c2b4d8bcab18c5c86f84fa3d0f2f50eac38b7903bfcc064d8c193f8837b1f.jpg)  
Fig. 1. Mean cumulative regret over 10 runs under Pareto rewards with infinite variance (M=2, K=2, T=10<sup>6</sup>), log–log axes.

The ordering at this horizon is governed by constants rather than by rates, and is instructive. Elimination in Problem B requires two intervals to separate, i.e. $4 \alpha _ { a } < \Delta _ { a }$ , whereas the index comparisons of Problems A and C need only $2 \alpha _ { a } < \Delta _ { a }$ ; by (5) this is $2 ^ { ( 1 + \varepsilon ) / \varepsilon } = 8$ times more samples of each arm when $\varepsilon = 0 . 5$ , which is what the early growth of the orange curve buys. The payoff is that once the active set collapses, mRUCB-Intervals incurs no further regret at all, whereas the other two keep exploring; the curves would therefore cross at larger horizons. Similarly, mHT-DSEE is inexpensive here because its exploration budget $K ^ { M } \lceil \log ^ { 2 } t \rceil$ is only ≈760 rounds at $T = 1 0 ^ { 6 }$ , even though its rate is the worst of the three. The experiments thus support the theory while showing that the hierarchy of Table I is asymptotic: at moderate horizons the constants attached to each coordination mechanism dominate. A further practical caveat is that the joint action space grows as $K ^ { M }$ , so larger instances lengthen the exploration phases of mHT-DSEE and slow the elimination cascade of mRUCB-Intervals.

## VII. CONCLUSION

We studied multi-agent bandits with heavy-tailed rewards under three information asymmetries. Our algorithms show that effective decentralized learning is achievable even under significant asymmetry and non-sub-Gaussian noise: shared rewards (Problem A) enable costless synchronization; observed actions (Problem B) provide an implicit signaling channel whose cost is independent of the horizon and of the number of players; and the fully asymmetric setting (Problem C) requires a pre-committed anytime schedule at a log T factor of additional cost. That observable actions compensate for the loss of shared rewards at leading order is a notable positive result, while the barrier between Problems B and C shows the value of even minimal observability.

Two limitations point to future work. First, our guarantees, like those of [10], [11], assume that (ε, v) are known: a conservative choice (smaller ε or larger v) keeps every bound valid but inflates the confidence radius and hence the regret, so adapting to unknown tail heaviness—plausibly through selfnormalized constructions such as [15]—remains open. Second, tighter lower bounds for Problem C would clarify whether the extra log T factor is necessary without shared information. Extensions to adversarial or non-stationary rewards, and structured reward models such as linear or factored bandits that would mitigate the exponential $K ^ { M }$ dependence, are also natural directions.

## REFERENCES

[1] H. E. Robbins, “Some aspects of the sequential design of experiments,” Bulletin of the American Mathematical Society, vol. 58, pp. 527–535, 1952.

[2] W. R. Thompson, “On the likelihood that one unknown probability exceeds another in view of the evidence of two samples,” Biometrika, vol. 25, no. 3/4, pp. 285–294, 1933.

[3] B. Awerbuch and R. Kleinberg, “Competitive collaborative learning,” Journal of Computer and System Sciences, vol. 74, no. 8, pp. 1271– 1288, 2008. Learning Theory 2005.

[4] B. Szorenyi, R. Busa-Fekete, I. Hegedus, R. Ormandi, M. Jelasity, and B. Kegl, “Gossip-based distributed stochastic bandit algorithms,” in Proceedings of the 30th International Conference on Machine Learning (S. Dasgupta and D. McAllester, eds.), vol. 28 of Proceedings of Machine Learning Research, (Atlanta, Georgia, USA), pp. 19–27, PMLR, 17–19 Jun 2013.

[5] D. Kalathil, N. Nayyar, and R. Jain, “Decentralized learning for multiplayer multiarmed bandits,” IEEE Transactions on Information Theory, vol. 60, no. 4, pp. 2331–2345, 2014.

[6] C. Shi and C. Shen, “Multi-player multi-armed bandits with collisiondependent reward distributions,” IEEE Transactions on Signal Processing, vol. 69, p. 4385–4402, 2021.

[7] W. Chang, M. Jafarnia-Jahromi, and R. Jain, “Online learning for cooperative multi-player multi-armed bandits,” CoRR, vol. abs/2109.03818, 2021.

[8] W. Chang and Y. Lu, “Optimal cooperative multiplayer learning bandits with noisy rewards and no communication,” arXiv preprint arXiv:2311.06210, 2023.

[9] E. Boursier and V. Perchet, “A survey on multi-player bandits,” 2024.

[10] S. Bubeck, N. Cesa-Bianchi, and G. Lugosi, “Bandits with heavy tail,” 2012.

[11] S. Vakili, K. Liu, and Q. Zhao, “Deterministic sequencing of exploration and exploitation for multi-armed bandit problems,” 2013.

[12] X. Yu, H. Shao, M. R. Lyu, and I. King, “Pure exploration of multiarmed bandits with heavy-tailed payoffs.,” in UAI, pp. 937–946, 2018.

[13] H. Shao, X. Yu, I. King, and M. R. Lyu, “Almost optimal algorithms for linear stochastic bandits with heavy-tailed payoffs,” in Advances in Neural Information Processing Systems (S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, eds.), vol. 31, Curran Associates, Inc., 2018.

[14] K. Lee and S. Lim, “Minimax optimal bandits for heavy tail rewards,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 4, pp. 5280–5294, 2024.

[15] H. Wang and A. Ramdas, “Catoni-style confidence sequences for heavytailed mean estimation,” Stochastic Processes and their Applications, vol. 163, p. 168–202, Sept. 2023.

[16] A. Dubey et al., “Cooperative multi-agent bandits with heavy tails,” in International conference on machine learning, pp. 2730–2739, PMLR, 2020.

[17] X. Wang and M. Xu, “Multi-agent multi-armed bandit with fully heavytailed dynamics,” 2025.
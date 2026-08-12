# Coordinating the Unknown Lipschitz Constant in Multiplayer Bandits

Ricardo Parada

Chenzhang Zhao

William Chang

University of California, Riverside

Riverside, CA, USA

University of California, Los Angeles

University of California, Los Angeles

Los Angeles, CA, USA

Los Angeles, CA, USA

rparadaumanzor@gmail.com

zhaochenzhang03@g.ucla.edu

chang314@g.ucla.edu

Abstract—Motivated by decentralized applications, we study cooperative multi-agent bandits in continuous (Lipschitz) action spaces when the Lipschitz constant is unknown. We consider three information structures: (A) unobserved actions with common rewards, (B) observed actions with independent rewards, and (C) unobserved actions with independent rewards. In each case we design and analyze an algorithm that estimates the Lipschitz constant, chooses a discretization of the joint action space, and applies a cooperative bandit method to the induced discrete problem. Players never communicate once learning starts, so the central difficulty is that they must reach the same discretization from their own data. We prove regret guarantees showing that common rewards and observable actions each supply this agreement for free, and that in their absence agreement can still be bought, through a dithered quantization of the estimate, at no cost in the leading order of the regret.

Index Terms—Lipschitz bandits, multiplayer bandits, informa tion asymmetry, discretization.

## I. INTRODUCTION

The multi-armed bandit (MAB) problem is a central model in sequential decision-making: an agent repeatedly chooses among actions with unknown rewards, balancing exploration and exploitation to maximize cumulative reward. Many deployments are inherently distributed, with several decision-makers acting concurrently and observing only part of the system feedback. In wireless systems, multiple users opportunistically access channels whose availabilities are unknown and time-varying, often without explicit coordination, as in cognitive radio and dynamic spectrum access [1], [2]. Similar abstractions arise in distributed sensing and radar networks, where nodes select waveforms or frequency bands while attempting to avoid mutual interference [3]. These settings motivate cooperative multiplayer bandits in which agents coordinate implicitly under information constraints, and the relevant information structure is dictated by what each agent can observe: actions, rewards, or both.

To model continuously many actions we work with Lipschitz bandits, where the mean reward varies smoothly over a metric space so that nearby actions have similar rewards. Discretization-based methods exploit this structure, but the resolution they should use depends on the Lipschitz constant L. We therefore study the setting in which L is not given in advance, extending the multiplayer Lipschitz bandits of [4]. Three forms of information asymmetry, described in Section III-A and referred to as Problems A, B, and C, are treated in turn: unobserved actions with common rewards, observed actions with independent rewards, and unobserved actions with independent rewards. Because players cannot communicate once learning begins, an unknown L creates a difficulty absent in the single-agent problem: each player estimates L from its own observations, and if these estimates disagree the players build different grids and can no longer be regarded as playing a common discrete bandit. Coordination, rather than estimation accuracy, is the binding constraint.

Contributions. We give a meta-algorithm, mECAB, that estimates an upper confidence bound on L by uniform exploration of a coarse grid, fixes a discretization, and runs an existing cooperative multiplayer MAB subroutine; we prove regret bounds for all three information structures, each resting on a different agreement mechanism, namely shared rewards in Problem A, signalling through observed actions in Problem B, and a dithered quantization of the estimate in Problem C, the last of which makes the agreement probability independent of the instance in a way no deterministic rounding rule can; and we report simulations comparing Lipschitz-adaptive with non-adaptive discretization.

## A. Related Work

The Lipschitz bandit literature originates with the continuumarmed bandit (CAB) model [5], in which arms lie in a continuum and the mean reward obeys a continuity condition; sharper discretization strategies followed [6], and the zooming algorithm [7] treats general metric spaces through the covering dimension. Closest to our starting point is [8], which removes the need to know L a priori by estimating it from a uniform grid; our estimator and notation follow that work.

Multi-player extensions introduce collisions and asymmetric feedback; see [9] for a survey. Recent work on cooperative learning under different information structures adapts UCBtype ideas to obtain near-optimal guarantees [4], [10], [11], and we use those algorithms as subroutines. To the best of our knowledge no existing multi-player algorithm addresses Lipschitz bandits with an unknown Lipschitz constant; the closest work is [12], which assumes a maximal Lipschitz constant but does not learn L.

## II. PRELIMINARIES

Consider a d-dimensional compact set of arms $X = [ 0 , 1 ] ^ { d } ,$ $d \geq 1$ . Each arm x carries a reward distribution $\nu _ { x }$ supported on [0, 1], with mean-payoff function $f : [ 0 , 1 ] ^ { d } \to [ \bar { 0 , 1 } ] .$ . At each round $t \geq 1$ the player selects $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and receives $Y _ { t } \sim \nu _ { x _ { t } }$ drawn independently across rounds.

Assumption. $f$ is twice differentiable with Hessian uniformly bounded by $N :$ for all x, y, $\begin{array} { r } { | \pmb { y } ^ { \top } H _ { f } ( \pmb { x } ) \pmb { y } | \le N \| \pmb { y } \| _ { \infty } ^ { 2 } . } \end{array}$ The map $\| \nabla f \| _ { 1 }$ is continuous and attains its maximum $L$ on $[ 0 , 1 ] ^ { d }$ , so $f$ is L-Lipschitz with respect to $\| \cdot \| _ { \infty }$ . Write $\mathcal { F } _ { L , N }$ for the class of mean-payoff functions satisfying both conditions with parameters $L$ and $N ;$ all suprema below are over this class, and neither parameter is known to the players. With $f ^ { \star } = \operatorname* { m a x } _ { x } f ( x )$ , the expected regret at horizon $T$ is $\begin{array} { r } { R _ { T } = \mathbb { E } [ T f ^ { \star } - \sum _ { t = 1 } ^ { T } f ( \pmb { x } _ { t } ) ] } \end{array}$ , the expectation being over the draws of $Y _ { t }$ and any internal randomization. The goal is to minimize $R _ { T }$ without knowing $L .$

## III. MAIN RESULTS

## A. Extension to the Multi-Agent Setting

Let $P _ { 1 } , \ldots , P _ { M }$ be players, each holding a d-dimensional set of arms $A _ { i } = [ 0 , 1 ] ^ { d }$ , so the joint action space is $\mathcal { A } = A _ { 1 } \times \cdot \cdot \cdot \times$ $A _ { M } = [ 0 , 1 ] ^ { M d }$ , with joint arms $\pmb { a } = ( a _ { 1 } , \ldots , a _ { M d } )$ . Players may agree on a strategy, and on any shared randomness it uses, before learning begins, but cannot communicate afterwards. At each round every player chooses ${ \pmb a } _ { t } ^ { i } \in [ 0 , 1 ] ^ { d }$ simultaneously, forming ${ \mathbf { } } _ { \mathbf { } } \mathbf { } _ { \mathbf { } } \mathbf { ; }$ with $f : { \mathcal { A } }  [ 0 , 1 ]$ in $\mathcal { F } _ { L , N }$ , the objective is again to minimize $R _ { T }$ when L is unknown.

Problem A: unobserved actions, common rewards. Every player receives the same reward $Y _ { t }$ but does not observe the actions of the others.

Problem B: observed actions, independent rewards. Every player observes the actions of all others, but rewards are i.i.d. across players and player i sees only its own reward $Y _ { t } ^ { i }$ . The regret $\begin{array} { r } { R _ { T } ^ { i } = \mathbb { E } [ T f ^ { \star } - \sum _ { t } f ( { \pmb a } _ { t } ) ] } \end{array}$ does not depend on $i ,$ since the rewards are identically distributed.

Problem C: unobserved actions, independent rewards. The two difficulties combine: rewards are i.i.d. across players and actions are unobserved.

## B. The Main Algorithm

The algorithm is motivated by [8] and follows the classical CAB template of [6]: use the Lipschitz constant to discretize the space, then run a standard MAB algorithm on the resulting finite set. When $L$ is known this yields sublinear regret; when it is unknown we must first estimate it.

During exploration each player splits its own action set into $m ^ { d }$ bins, inducing $m ^ { M d }$ joint bins, and estimates an upper bound on L from the differences between neighboring bins. Write $\overline { { L } } _ { m }$ for the expectation of the estimator of [8], which approaches L as m grows.

Lemma 1 (Bubeck et al. $I 8 J ) .$ : For $\begin{array} { r } { m \geq 3 , L - \frac { 7 N } { m } \leq \overline { { L } } _ { m } \leq } \end{array}$ L, where N is the Hessian bound.

Lemma 2: If each joint bin is explored with $E ^ { \prime }$ samples, then with probability at least $\begin{array} { r } { 1 - \delta , \left| \widehat { L } _ { m } - \overline { { L } } _ { m } \right| \leq m \sqrt { \frac { 2 } { E ^ { \prime } } \ln \frac { 2 m ^ { M d } } { \delta } } } \end{array}$

Algorithm 1: mECAB   
Input: Horizon T, coarse bins m, exploration budget   
$E ,$ dimension $d .$   
1 Initialize: each player divides $A _ { i } = [ 0 , 1 ] ^ { d }$ into $m ^ { d }$   
bins, inducing joint bins $\underline { { k } } \in \{ 0 , \ldots \dot { , } m ^ { M d } - 1 \}$   
2 Pre-learning: players agree on an ordering of the joint   
bins, and in Problem C on a dither $U \sim \mathrm { U n i f } [ 0 , 1 )$   
3 Exploration:   
4 for each joint bin $\underline { { k } } \in \{ 0 , \dots , m ^ { M d } - 1 \}$ do   
5 Each player samples E actions uniformly from its   
own bin and observes the resulting rewards.   
6 Compute the empirical bin mean $\widehat { \mu } _ { \underline { { k } } }$ (resp. $\widehat { \mu } _ { \underline { { k } } } ^ { i } )$   
7 end   
8 Form $\widehat { L }$ from (2), (3), or (4) according to the problem,   
and set   
$\begin{array} { r } { \widetilde { L } = \widetilde { L } + m \sqrt { \frac { 2 } { E ^ { \prime } } \ln ( 2 m ^ { M d } T ) } , \widetilde { m } = \big [ \widetilde { L } ^ { \frac { 2 } { M d + 2 } } T ^ { \frac { 1 } { M d + 2 } } \big ] , } \end{array}$   
(1)   
with $E ^ { \prime } = M E$ in Problem B and $E ^ { \prime } = E$ otherwise.   
9 Exploitation:   
10 for $t = E m ^ { M d } + 1$ to $T$ do   
11 Play the multiplayer MAB subroutine on the $\widetilde { m } ^ { M d }$   
joint actions.   
12 end

Adding the deviation of Lemma 2 to $\widehat { L } _ { m }$ produces an upper confidence bound $\widetilde { L }$ on $\overline { { L } } _ { m } ,$ which sets the discretization me ; combining the two lemmas with $\delta = 1 / T$ gives the two-sided control used in all of the proofs.

Corollary 3: Fix $\delta = 1 / T$ and let E<sup>′</sup> denote the number of samples per joint bin available to a player, so $E ^ { \prime } = M E$ in Problem B and $E ^ { \prime } = E$ in Problems A and C. With probability at least $1 - 1 / T$

$$
\begin{array} { r } { \overline { { L } } _ { m } \leq \widetilde { L } _ { m } \leq L + 1 + 2 m \sqrt { \frac { 2 } { E ^ { \prime } } \ln ( 2 m ^ { M d } T ) } , } \end{array}
$$

and if in addition $m \geq 8 N / L$ then $\widetilde { L } _ { m } \geq L / 8 - 1 .$

The lower bound prevents the algorithm from choosing too coarse a grid and is the only place where the Hessian bound $N$ enters: the condition $m \geq 8 N / L$ asks that the coarse grid already resolve the curvature of $f ,$ and it holds for all large T under the choice of m made in the proofs. Any multiplayer MAB algorithm, for instance [10], [11], is then run on the discretized joint space; Algorithm 1 collects the steps.

## IV. PROBLEM A: ACTION INFORMATION ASYMMETRY

Here the environment produces a single common reward observed by all players, while each player does not observe the others’ actions. Unobserved actions prevent coordination during play, and the unknown smoothness must be estimated without access to the exploration of the others. What rescues the situation is that the reward is shared, which makes the exploration statistics shared as well once the schedule is fixed in advance.

Concretely, the players agree on an ordering of the $m ^ { M d }$ joint bins and each samples uniformly inside the scheduled bin. Only the bin index affects the statistic collected, not the particular arm chosen inside it, so all players obtain the same empirical mean $\begin{array} { r } { \widehat { \mu } _ { \underline { { k } } } = \frac { 1 } { E } \sum _ { j = 1 } ^ { E } Z _ { \underline { { k } } , j } } \end{array}$ for every joint bin $\underline { { k } } ,$ and each is free to choose any arm within its own bin. Consequently every player forms the same estimate

$$
\widehat { L } = m \operatorname* { m a x } _ { \substack { k \in [ m - 2 ] ^ { M d } , s \in \{ - 1 , 1 \} ^ { M d } } } \Big | \widehat { \mu } _ { k } - \widehat { \mu } _ { k + s } \Big | ,\tag{2}
$$

hence the same $\widetilde { L }$ and ${ \widetilde { m } } ,$ and the ordering of the coarse bins induces a consistent ordering of the finer me -level grid that everyone can follow.

Theorem 4: Let $m \geq 8 N / L$ and use m-UCB of [10] with $\widehat { L }$ from (2). Then mECAB satisfies

$$
\begin{array} { r l } & { \underset { \mathcal { F } _ { L , N } } { \operatorname* { s u p } } R _ { T } \leq T ^ { ( M d + 1 ) / ( M d + 2 ) } } \\ & { \cdot \left( 9 L ^ { M d / ( M d + 2 ) } + 5 \left( 2 m \sqrt { \frac { 2 } { E } \ln { ( 2 T ^ { M d + 1 } ) } } \right) ^ { M d / ( M d + 2 ) } \right) } \\ & { \qquad + E m ^ { M d } + 3 2 \sqrt { T \widetilde { m } ^ { M d } \log { T } } + 1 . } \end{array}
$$

On the event of Corollary 3 the subroutine term is $O ( T ^ { ( M d + 1 ) / ( M d + 2 ) } ( L + 1 ) ^ { M d / ( M d + 2 ) } \sqrt { \log T } )$ , matching the <sub>leading</sub> <sub>term</sub> <sub>up</sub> <sub>to</sub> √<sub>log</sub> <sub>T;</sub> <sub>the</sub> <sub>same</sub> <sub>holds</sub> <sub>in</sub> <sub>Theorems</sub> <sub>5</sub> and 7.

In the single-agent problem with unknown $L ,$ discretization yields the familiar scaling $T ^ { ( d + 1 ) / ( d + 2 ) }$ , and Problem A behaves the same way on the joint space with M d in place of $d .$ The action asymmetry therefore costs nothing beyond this dimensional effect, precisely because the common reward and the pre-agreed schedule force identical bin means, an identical ${ \widehat { L } } ,$ and an identical grid. The dependence on the size of the discretized joint set is also unavoidable, since with K actions each the induced finite problem has $K ^ { M }$ joint arms, to which the standard finite-armed lower bound applies.

## V. PROBLEM B: REWARD INFORMATION ASYMMETRY

Problem B reverses the structure: actions are observable but reward observations are not shared. The free synchronization of Problem A breaks, since the bin means $\widehat { \mu } _ { k } ^ { i }$ may differ across players and would lead to different grids if nothing further were done.

The remedy is to exploit action observability to share reward information implicitly. Since every action is observed by everyone, an action can carry a signal encoding the sender’s statistics, and in a continuum this needs no departure from the action space: after collecting E − 1 samples from a bin, a player devotes its final action in that bin to encoding its empirical mean, which the others decode and fold into their own estimate. Nothing analogous exists in Problem A, where actions are hidden, nor in finite-action models, where there is no room to encode a real number without distorting the learning problem. At a cost of one sample, each player thus gains $( M - 1 ) ( E - 1 )$ further samples for every bin, and forms

$$
\widehat { L } = m \operatorname* { m a x } _ { \substack { k \in [ m - 2 ] ^ { M d } , s \in \{ - 1 , 1 \} ^ { M d } } } \left| \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \bigl ( \widehat { \mu } _ { k } ^ { i } - \widehat { \mu } _ { k + s } ^ { i } \bigr ) \right| .\tag{3}
$$

The effective sample size entering the concentration of $\widehat { L }$ is multiplied by M, which sharpens the estimate and improves the grid. Problem A had perfect alignment of the bin means but no way to convey anything beyond the common scalar reward; Problem B lacks common rewards but recovers most of the benefit of centralized averaging by broadcasting estimates through actions.

Theorem 5: Let $m ~ \geq ~ 8 N / L$ and use the multiplayer subroutine of [11] for Problem B. Then mECAB satisfies

$$
\begin{array} { c } { { \mathrm { s u p ~ } R _ { T } \leq T ^ { ( M d + 1 ) / ( M d + 2 ) } } } \\ { { \ } } \\ { { \qquad \times \left( 9 L ^ { M d / ( M d + 2 ) } \right. } } \\ { { \qquad \left. + 5 \left( 2 m \sqrt { \frac { 2 } { M E } \ln { ( 2 T ^ { M d + 1 } ) } } \right) ^ { M d / ( M d + 2 ) } \right) } } \\ { { \qquad + E m ^ { M d } + 3 2 \sqrt { T \widetilde { m } ^ { M d } \log { T } } + 1 . } } \end{array}
$$

Compared with Theorem 4 the sample size inside the square root improves from E to ME, which is exactly the pooling gain from the other players’ samples.

## VI. PROBLEM C: REWARD AND ACTION INFORMATION ASYMMETRY

Problem C is the hardest of the three: rewards are not shared, so the mechanism of Problem A is unavailable, and actions are not observed, so the signalling of Problem B is unavailable too. If each player simply used (2) on its own data, the estimates would differ and the induced grids me would differ with them, destroying the common discrete problem the subroutine needs.

We restore agreement by quantizing the estimate, so that small discrepancies between players do not change the value they act on. Let $X ^ { i } = m \bar { \operatorname * { m a x } } _ { k , s } | \widehat { \mu } _ { k } ^ { i } - \widehat { \mu } _ { k + s } ^ { i } |$ be the raw estimate of player i, the maximum running over $k \in [ m - 2 ] ^ { M d }$ and $s \in \{ - \bar { 1 } , \bar { 1 } \} ^ { M d }$ as above. A deterministic rounding will not do: if $\overline { { L } } _ { m }$ sits near a rounding boundary, two players whose estimates straddle it round differently however many samples they collect, and the failure probability approaches $1 / 2$ regardless of E. The boundaries are fixed while $\overline { { L } } _ { m }$ is a property of the instance, so no deterministic rule avoids this. Instead the players agree in advance on a dither $U \sim \mathrm { U n i f } [ 0 , 1 )$ shared randomness requiring no communication, and set

$$
{ \widehat { L } } ^ { i } = \lfloor X ^ { i } + U \rfloor .\tag{4}
$$

Randomizing the offset makes the distance from $\overline { { L } } _ { m }$ to the nearest boundary uniform rather than instance-dependent, so the probability of disagreement can be bounded with no reference to where $\overline { { L } } _ { m }$ lies.

Lemma 6: For any $\delta > 0$ and any player i,

$$
P \big ( | X ^ { i } - \overline { { { L } } } _ { m } | > \delta \big ) \leq 4 ( 2 m ) ^ { M d } \exp \left( - \frac { E \delta ^ { 2 } } { 3 2 m ^ { 2 } } \right) .
$$

Consequently, with $\widehat { L } ^ { i }$ as in (4) and $A : = 4 M ( 2 m ) ^ { M d }$ , the probability that the M players do not all obtain the same value of $\widehat { L }$ is at most 17 $m \sqrt { \ln ( A ) / E }$

When the players do agree they share the same $\widetilde { m } .$ , and Problem C reduces to running the same discretized strategy as before; when they do not, they may follow different grids and we pay for that event in the regret.

Theorem 7: Let $m \geq 8 N / L$ , let $A = 4 M ( 2 m ) ^ { M d }$ , and use the multiplayer subroutine of [10] with $\widehat { L }$ from (4). Then

$$
\begin{array} { r l } { } & { \underset { \mathcal { F } _ { L , N } } { \operatorname* { s u p } } R _ { T } \leq T ^ { ( M d + 1 ) / ( M d + 2 ) } } \\ { } & { \times \left( 9 L ^ { M d / ( M d + 2 ) } + 5 \left( 2 m \sqrt { \frac { 2 } { E } \ln \left( 2 T ^ { M d + 1 } \right) } \right) ^ { M d / ( M d + 2 ) } \right) } \\ { } & { \qquad + E m ^ { M d } + C \log T \sqrt { T \widetilde { m } ^ { M d } } + 1 7 T m \sqrt { \ln ( A ) / E } . } \end{array}
$$

In particular, if $E \ge m ^ { 2 } T ^ { 2 / ( M d + 2 ) }$ ln A the last term is at most $1 7 T ^ { ( M d + 1 ) / ( M d + 2 ) }$

Problem C therefore isolates what each kind of observability buys. Common rewards make $\widehat { L }$ shared automatically and no agreement term is needed; observable actions permit pooling and sharpen the concentration, improving $E$ to $M E ;$ when neither is available, agreement must be built from concentration together with dithered quantization. Its price is the final term, which is instance-independent and, for the stated exploration budget, of the same order as the leading one, so Problem C matches Problems A and B up to constants once E is large enough.

## VII. EXPERIMENTS

We simulate a cooperative bandit with $M = 2$ players and action dimension $d = 1$ each, so $M d = 2 .$ , over $T = 1 0 ^ { 5 }$ rounds and 10 independent trials for each configuration. A maximizer $a ^ { \star }$ is drawn uniformly from $[ 0 , 1 ] ^ { M d }$ once in each trial and the mean reward is $f ( a ) = - L \| a - a ^ { \star } \| _ { \infty } ,$ which is L-Lipschitz with respect to $\ell _ { \infty }$ and satisfies $f ( a ^ { \star } ) = 0 ;$ rewards are Gaussian with unit variance. We report cumulative pseudo-regret $\begin{array} { r } { \sum _ { t } ( f ( a ^ { \star } ) - f ( a _ { t } ) ) } \end{array}$ averaged over trials, with ±1 standard deviation shading.

Both algorithms discretize the joint space and run UCB on the resulting grid, differing only in how the resolution is set. Est-L explores a coarse grid of $m$ bins in each coordinate with E uniform samples in each bin, estimates $\widehat { L } _ { b } ~ = ~ m \operatorname* { m a x } _ { ( b , b ^ { \prime } ) \in { \mathcal N } } | \widehat { \mu } ( b ) - \widehat { \mu } ( b ^ { \prime } )$ | over neighboring bin pairs ${ \mathcal { N } } .$ , pads it as in (1), and sets me from the result; this phase is not aimed at collecting reward, so regret grows roughly linearly while it runs. $N o { - } L$ skips exploration and takes $\dot { \widetilde { m } } = \lceil T ^ { 1 \dot { / } ( M d + 2 ) } \rceil$ ⌉, avoiding the up-front cost but risking a resolution mismatched to the smoothness of $f .$

The three information structures are modelled at the level of the feedback reaching each player rather than through the signalling and quantization mechanisms themselves: Problem A supplies a common reward, Problem B pools the M independent rewards of a round, the idealized effect of encoding empirical means in actions, and Problem C uses a single reward stream without pooling. This isolates the effect of feedback quality on the accuracy of $\widetilde { L } ;$ simulating the signalling and dithering steps directly is left to an extended version.

![](images/9117c9a9641470fd1de7a8923b3fc4df2885a68988d0ac279fa467090abdf39d.jpg)

![](images/82dd59621c3ffcf1a5e8c582c08e020d0e3fe54f5a39a6a40e37dacc0ced5e74.jpg)  
Fig. 1. Cumulative regret averaged over trials with ±1 standard deviation shading, for a small Lipschitz constant (L=1, top) and a large one $\scriptstyle ( L = 1 0 0 0 ,$ bottom). Each panel overlays Problems A, B, and C under Est-L and No-L discretization.

Fig. 1 compares the two rules for a small $( L = 1 )$ and a large $( L = 1 0 0 0 )$ Lipschitz constant, everything else held fixed. In all three cases the Est-L curves grow linearly during coarse exploration and then bend into a visibly sublinear phase once UCB begins on the refined grid, which is the tradeoff the method makes. When $L$ is small the two rules produce comparable resolutions and finish at similar levels; when L is large, fixing the resolution without reference to $L$ gives a grid too coarse for the variation of $f ,$ and Est-L overtakes it despite the initial linear segment. The information structure modulates the gain: pooling in Problem B makes $\widetilde { L }$ more accurate and flattens the later slope relative to Problems A and C, while Problem C, with the weakest feedback, is the most variable.

## VIII. CONCLUSION

We extended cooperative multiplayer bandits to Lipschitz action spaces with an unknown Lipschitz constant, where the players’ estimates of $L$ must agree for a common discretization to exist. Common rewards and observable actions each deliver that agreement for free, and when neither is present a dithered quantization delivers it at no cost in the leading order of the regret. Natural next steps are adversarial rewards and structural assumptions beyond Lipschitz continuity.

## APPENDIX

## A. Proof of Lemma 6

Proof: Fix $\delta > 0$ and let ${ \overline { { f } } _ { m } } ( k )$ denote the mean of f over bin k. By Hoeffding’s inequality, for any bin k,

$$
\begin{array} { r } { P \big ( | \widehat { \mu } _ { k } - \overline { { f } } _ { m } ( k ) | > \frac { \delta } { 2 m } \big ) \leq 2 \exp \left( - \frac { E \delta ^ { 2 } } { 3 2 m ^ { 2 } } \right) . } \end{array}\tag{5}
$$

For a neighboring pair $( k , k ^ { \prime } )$ , the triangle inequality and (5) give

$$
\begin{array} { r l } & { P \big ( | | \widehat { \mu } _ { k } - \widehat { \mu } _ { k ^ { \prime } } | - | \overline { { f } } _ { m } ( k ) - \overline { { f } } _ { m } ( k ^ { \prime } ) | \big | > \frac { \delta } { m } \big ) } \\ & { \quad \leq P \big ( | \widehat { \mu } _ { k } - \overline { { f } } _ { m } ( k ) | > \frac { \delta } { 2 m } \big ) + P \big ( | \widehat { \mu } _ { k ^ { \prime } } - \overline { { f } } _ { m } ( k ^ { \prime } ) | > \frac { \delta } { 2 m } \big ) } \\ & { \quad \leq 4 \exp \biggl ( - \frac { E \delta ^ { 2 } } { 3 2 m ^ { 2 } } \biggr ) . } \end{array}
$$

There are at most $m ^ { M d } 2 ^ { M d } = ( 2 m ) ^ { M d }$ pairs $( k , s )$ in the maximum defining $X ^ { i }$ , so a union bound over them and multiplication by m yield the first claim.

For the second, let $\Delta _ { U }$ be the distance from $\overline { { L } } _ { m } + U$ to the nearest integer; since U is uniform on $[ 0 , 1 ) , \Delta _ { U }$ is uniform on [0, <sup>1</sup> ]. If $| X ^ { i } - \overline { { L } } _ { m } | < \Delta _ { U }$ for every i, then all the $X ^ { i } + U$ lie in the same unit interval and every player obtains the same $\widehat { L }$ Writing $a = E / ( 3 2 m ^ { 2 } )$ and $A = 4 M ( 2 m ) ^ { M d }$ , a union bound over the M players and the first claim give, conditionally on $U ,$ a disagreement probability of at most min $\left\{ 1 , A e ^ { - a \Delta _ { U } ^ { 2 } } \right\}$ . Let $\delta _ { 0 } = m \sqrt { 3 2 \ln ( A ) / E } .$ , so that $A e ^ { - a \delta _ { 0 } ^ { 2 } } = \dot { 1 }$ . Averaging over $U ,$ , whose density is 2 on $[ 0 , \textstyle { \frac { 1 } { 2 } } ]$

$$
\begin{array} { l } { { \displaystyle P \mathrm { ( d i s a g r e e m e n t ) } \le 2 \delta _ { 0 } + 2 A \int _ { \delta _ { 0 } } ^ { \infty } e ^ { - a \delta ^ { 2 } } d \delta } } \\ { { \displaystyle \qquad \le 2 \delta _ { 0 } + \frac { A e ^ { - a \delta _ { 0 } ^ { 2 } } } { a \delta _ { 0 } } = 2 \delta _ { 0 } + \frac { 3 2 m ^ { 2 } } { E \delta _ { 0 } } } } \\ { { \displaystyle \qquad \le 3 \delta _ { 0 } \le 1 7 m \sqrt { \ln ( A ) / E } , } } \end{array}
$$

where the second inequality uses $\begin{array} { r } { \int _ { \delta _ { 0 } } ^ { \infty } e ^ { - a \delta ^ { 2 } } d \delta \leq e ^ { - a \delta _ { 0 } ^ { 2 } } / ( 2 a \delta _ { 0 } ) } \end{array}$ and the third uses ln $A \geq 1$

## B. Proof of Theorems 4, 5 and 7

Proof: We give the argument once, writing $E ^ { \prime }$ for the samples per joint bin available to a player, so $E ^ { \prime } = M E$ in Problem B and $E ^ { \prime } = E$ otherwise, and $\mathcal { R } ( K , T )$ for the regret of the multiplayer subroutine on K joint arms. Exploration costs at most $E m ^ { M d }$ , the discretization bias costs $L T / \widetilde { m }$ , and the subroutine contributes $\mathcal { R } ( \widetilde { m } ^ { M d } , T )$ , so

$$
\operatorname* { s u p } _ { \mathcal { F } _ { L , N } } R _ { T } \leq E m ^ { M d } + \mathbb { E } \bigg [ \frac { L T } { \widetilde { m } } + \mathcal { R } ( \widetilde { m } ^ { M d } , T ) \bigg ] + \Xi ,\tag{6}
$$

where $\Xi = 0$ in Problems A and B, since all players hold the same me by construction, and $\Xi = T \cdot 1 7 m \sqrt { \ln ( A ) / E }$ in Problem C by Lemma 6, bounding the regret on the disagreement event by T.

Writing $x = \widetilde { L } _ { m } ^ { 2 / ( M d + 2 ) } T ^ { 1 / ( M d + 2 ) }$ , (1) gives $\widetilde m \le x ( 1 +$ $1 / x )$ , and when $x \geq M d$ we may use $( 1 + 1 / x ) ^ { M d } \leq e$ Substituting into (6),

$$
\begin{array} { r l } { \underset { \mathcal { F } _ { L , N } } { \operatorname* { s u p } } R _ { T } \leq { { E m } ^ { M d } } + \Xi } \\ { + \mathbb { E } \Bigg [ T ^ { ( M d + 1 ) / ( M d + 2 ) } \frac { L + 1 } { { \widetilde { L } _ { m } ^ { 2 / ( M d + 2 ) } } } } \\ { + C \log T \sqrt { T e \big ( T ^ { 1 / ( M d + 2 ) } { \widetilde { L } _ { m } ^ { 2 / ( M d + 2 ) } } \big ) ^ { M d } } \Bigg ] . } \end{array}
$$

By Corollary 3, which applies since $m \ \geq \ 8 N / L$ , with probability at least $1 - 1 / T$

$$
\begin{array} { r l } & { C \log T \sqrt { T e \left( T ^ { 1 / ( M d + 2 ) } \widetilde { L } _ { m } ^ { 2 / ( M d + 2 ) } \right) ^ { M d } } } \\ & { \quad \le C \log T \sqrt { e } T ^ { ( M d + 1 ) / ( M d + 2 ) } } \\ & { \qquad \cdot \left( ( L + 1 ) ^ { \frac { M d } { M d + 2 } } + \left( 2 m \sqrt { \frac { 2 } { E ^ { \prime } } \ln ( 2 m ^ { M d } T ) } \right) ^ { \frac { M d } { M d + 2 } } \right) , } \end{array}
$$

while $\widetilde { L } _ { m } ~ \ge ~ L / 8 ~ - ~ 1$ controls the first term; on the complementary event, of probability below $1 / T ,$ , the regret is at most $T$ and contributes at most 1. Bounding m by T inside the logarithm and choosing $\begin{array} { c c } { { m } } & { { = } } & { { \lfloor T ^ { \alpha } \rfloor } } \end{array}$ and $E \ = \ m ^ { 2 M } \lceil T ^ { 2 \gamma ( \widecheck M d + 2 ) / ( M d ) } \rceil$ for suitable $\alpha , \gamma ~ > ~ 0$ gives the stated bounds, with $\mathcal { R } ( \widetilde { m } ^ { M d } , T ) ~ = ~ 3 2 \sqrt { T \widetilde { m } ^ { M d } \log T }$ for Problem A by [10] and for Problem B by [11], and $\mathcal { R } ( \widetilde { m } ^ { M d } , T ) = C \log T \sqrt { T \widetilde { m } ^ { M d } }$ for Problem C by [10]. For the final claim of Theorem $7 , E \ge m ^ { 2 } T ^ { 2 / ( M d + 2 ) }$ ln A gives $1 7 T m \sqrt { \ln ( A ) / E } \leq 1 7 T ^ { ( M d + 1 ) / ( \overline { { M } } d + 2 ) }$ ■

## REFERENCES

[1] K. Liu and Q. Zhao, “Distributed learning in multi-armed bandit with multiple players,” IEEE Transactions on Signal Processing, vol. 58, no. 11, pp. 5667–5681, Nov. 2010.

[2] A. Anandkumar, N. Michael, A. K. Tang, and A. Swami, “Distributed algorithms for learning and cognitive medium access with logarithmic regret,” IEEE Journal on Selected Areas in Communications, vol. 29, no. 4, pp. 731–745, Apr. 2011.

[3] W. W. Howard, C. E. Thornton, A. F. Martone, and R. M. Buehrer, “Multiplayer bandits for distributed cognitive radar,” 2021, arXiv:2102.00274.

[4] W. Chang and A. Kartik, “Multiplayer information asymmetric bandits in metric spaces,” 2025, arXiv:2503.08004.

[5] R. Agrawal, “The continuum-armed bandit problem,” SIAM Journal on Control and Optimization, vol. 33, no. 6, pp. 1926–1951, 1995.

[6] R. Kleinberg, “Nearly tight bounds for the continuum-armed bandit problem,” in Advances in Neural Information Processing Systems 17. MIT Press, 2004, pp. 697–704.

[7] R. Kleinberg, A. Slivkins, and E. Upfal, “Multi-armed bandits in metric spaces,” 2008, arXiv:0809.4882.

[8] S. Bubeck, G. Stoltz, and J. Y. Yu, “Lipschitz bandits without the lipschitz constant,” 2011, arXiv:1105.5041.

[9] E. Boursier and V. Perchet, “A survey on multi-player bandits,” 2024, arXiv:2211.16275.

[10] W. Chang, M. Jafarnia-Jahromi, and R. Jain, “Online learning for cooperative multi-player multi-armed bandits,” CoRR, vol. abs/2109.03818, 2021.

[11] W. Chang and Y. Lu, “Optimal cooperative multiplayer learning bandits with noisy rewards and no communication,” arXiv preprint arXiv:2311.06210, 2023.

[12] I. Bistritz and N. Bambos, “Cooperative multi-player bandit optimization,” in Advances in Neural Information Processing Systems 33. Curran Associates, Inc., 2020, pp. 697–707.
# Adversarial Resilience of Poisson-Process Submodular Maximization over Matroids: From Robust Ofline Optimization to Full-Bandit Learning

Vaneet Aggarwal<sup>∗</sup>

## Abstract

We study nonnegative submodular maximization subject to a general matroid when the ofline algorithm is given an arbitrary controlled value oracle. Our main result is an adversarial resilience theorem for the Spiteful Greedy Swap Poisson Process (SGS-Poisson): without modifying its Poisson intensity, single-element exchange rule, or spiteful drop step, the algorithm retains limiting approxi mation factors 1/e for non-monotone objectives and 1 − 1/e for monotone objectives. More precisely, under every controlled oracle fb satisfying |fb(S) − f(S)| ≤ ξ for every set S, our implementation returns a feasible set with expected value at least $( 1 / e - \varepsilon ) \mathrm { O P T } - O ( k \xi )$ and $( 1 - 1 / e - \varepsilon ) \mathrm { O P T } - O ( k \xi )$ respectively, using $\widetilde { O } ( n k ^ { 2 } \varepsilon ^ { - 2 } )$ oracle calls. As a consequence, the ofline-to-online reduction yields full-bandit CMAB algorithms for general matroid-constrained submodular rewards with exact limiting approximation-regret factors 1/e and 1 − 1/e and $\widetilde { O } ( n ^ { 1 / 5 } k ^ { 4 / 5 } T ^ { 4 / 5 } )$ regret.

## 1 Introduction

Submodular maximization under matroid constraints is a canonical problem in combinatorial optimization, with applications including influence maximization, sensor placement, experimental design, and data summarization [11]. Let U be a ground set of n items, let $M = \left( U , { \mathcal { T } } \right)$ be a matroid, and define $k : = { \mathrm { r a n k } } ( M )$ . We study

$$
\operatorname* { m a x } _ { S \in { \mathscr { T } } } f ( S ) , \qquad f : 2 ^ { U } \to [ 0 , 1 ] ,
$$

where f is nonnegative and submodular. For monotone objectives, continuous greedy gives the classical $1 - 1 / e$ approximation [4], and combinatorial approaches based on non-oblivious local search also attain the optimal factor [6, 2]. For non-monotone objectives, measured continuous greedy gives the classical 1/e factor [5].

A recent line of work introduced a Poisson-process approach to this problem. Ganz-Rozenman et al. introduced a Poisson process for monotone submodular maximization under a matroid, while Kulik et al. subsequently extended the paradigm to non-monotone objectives, obtaining 1/e for the non-monotone case and $1 - 1 / e$ for the monotone case [10, 12]. The resulting SGS (Spiteful Greedy Swap)-Poisson process maintains feasibility and evolves through single-element exchanges, providing a discrete stochastic primitive for matroid-constrained optimization.

This paper asks whether the same SGS-Poisson process is resilient to controlled oracle error. This question is important for full-bandit combinatorial multi-armed bandits (CMAB), where empirical estimates of super-arm rewards naturally enter an ofline optimization routine. Nie et al. introduced a black-box reduction from resilient ofline approximation algorithms to full-bandit CMAB algorithms [15], and Fourati et al. later extended the reduction to ofline algorithms with an $( \alpha - \varepsilon )$ guarantee and polynomial dependence on $1 / \varepsilon \ [ 8 ]$ . Existing full-bandit results already cover general matroid constraints, including the $k _ { \mathrm { s u b } } = 1$ specialization of the k-submodular framework of Nie et al. [14], but obtain only $1 / 2$ for monotone and $1 / 3$ for non-monotone objectives. Our goal is therefore to establish the controlled-oracle resilience required to place SGS-Poisson inside this ofline-to-online framework and thereby recover the classical $1 - 1 / e$ and $1 / e$ approximation factors for general matroids.

Recent work on noisy submodular maximization considers persistent stochastic noise [1], whereas we consider persistent adversarially controlled oracle error, which is the form needed by the CMAB reduction. We do not analyze the persistent stochastic-noise model.

Our contribution. We prove that the SGS-Poisson algorithmic process is resilient. For every target approximation loss $\varepsilon \in ( 0 , 1 / 2 ]$ and every persistent oracle $\widehat { f }$ satisfying

$$
| { \widehat { f } } ( S ) - f ( S ) | \leq \xi \qquad \forall S \subseteq U ,\tag{1}
$$

we construct an implementation that returns $A _ { \mathrm { o u t } } \in \mathcal { T }$ with

$$
\mathbb { E } [ f ( A _ { \mathrm { o u t } } ) ] \geq \left\{ \begin{array} { l l } { ( 1 / e - \varepsilon ) \mathrm { O P T } - C k \xi , } & { f \mathrm { ~ n o n - m o n o t o n e } , } \\ { ( 1 - 1 / e - \varepsilon ) \mathrm { O P T } - C k \xi , } & { f \mathrm { ~ m o n o t o n e } , } \end{array} \right.\tag{2}
$$

for a universal constant $C .$ . In particular, this is the first full-bandit CMAB guarantee for general matroids in this framework that attains the classical limiting factors $1 / e$ and $1 - 1 / e$ for non-monotone and monotone objectives, respectively. The number of controlled-oracle calls is

$$
O \left( n k \log \frac { 1 } { \varepsilon } + \frac { n k ^ { 2 } \log n \log ( 1 / \varepsilon ) } { \varepsilon ^ { 2 } } + \frac { n k \log ^ { 2 } ( 1 / \varepsilon ) } { \varepsilon ^ { 2 } } \right) .\tag{3}
$$

The result is not obtained by simply replacing $f$ with $\widehat { f }$ in an existing proof. A bounded perturbation can change the entire adaptive trajectory: a maximum-weight residual base can change discontinuously; its basis-exchange map can consequently change; the stopping time of Advanced Preprocessing changes; and the sequence of bases used by the Poisson process changes. Moreover, $\widehat { f }$ need not itself be submodular or monotone. Thus one cannot invoke the structural properties of $f$ on the surrogate function.

The main technical novelty is the following adaptive potential-preservation result.

Controlled-oracle adaptive preprocessing preserves the exact structural certificate needed by SGS-Poisson.

The preprocessing process is driven by a controlled-oracle estimate of the maximum sum of residual marginals and stops at a controlled-oracle threshold. We first robustify Residual Random Greedy (RRG) to obtain a realized constant-factor estimate $\widehat { V }$ of OPT. We then run the exact preprocessing process with ${ \widehat { f } } .$ . Although the chosen bases and stopping time can be completely diferent from their exact-oracle counterparts, the exchange potential

$$
M _ { t } = f ( Q _ { t } \cup O _ { t } ) + { \frac { 1 } { 2 } } f ( Q _ { t } )\tag{4}
$$

satisfies the robust drift inequality

$$
\mathbb { E } [ M _ { t + 1 } - M _ { t } \mid \mathcal { F } _ { t } ] \geq \frac { 8 \mathrm { O P T } - k \xi } { k - t } .\tag{5}
$$

This is the key lemma of the paper. It simultaneously handles noisy maximum-base selection, a noisy adaptive stopping rule, and the evolving matroid exchange set.

A second technical ingredient is a robust almost-above-average swap lemma. The exact SGS-Poisson implementation estimates, for every candidate element, a multilinear marginal using random sets. Under

controlled oracle error we show that the base-sum concentration argument remains valid and incurs only an additional $O ( k \xi )$ term. Thus a swap at every Poisson event is η-almost-above-average with

$$
\eta \leq C _ { 1 } \varepsilon \mathrm { O P T } + C _ { 2 } k \xi .\tag{6}
$$

The original SGS-Poisson process and its drop rule are unchanged.

We use the ofline-to-online reduction proposed in [15, 8] as a black box: our resilience theorem supplies precisely the controlled-oracle interface needed by that reduction. This approach converts our resilience parameters into exact limiting approximation-regret factors $1 / e$ and $1 - 1 / e$ with $\widetilde { O } ( n ^ { 1 / 5 } k ^ { 4 / 5 } T ^ { 4 / 5 } )$ regret for general matroids. We emphasize that this is not claimed to dominate specialized cardinality-bandit rates; the point is that the general-matroid resilient ofline guarantee survives the full-bandit reduction.

## 2 Related Work

## 2.1 Submodular Maximization under Matroid Constraints

For monotone submodular maximization under a matroid, continuous greedy gives the classical $1 - 1 / e$ approximation [4]. For non-monotone objectives, measured continuous greedy gives $1 / e \ [ 5 ]$ , while the recent SGS-Poisson line obtains the same limiting non-monotone factor and the $1 - 1 / e$ monotone factor through a discrete Poisson-process construction [10, 12]. Our base algorithm is exactly SGS-Poisson: its Poisson intensity, valid-swap rule, and spiteful drop are not modified. Residual Random Greedy (RRG) was introduced by Buchbinder et al. [3]; we use its matroid version as a constant-factor preprocessing primitive.

## 2.2 Ofline-to-Online Learning and Full-Bandit CMAB

Nie et al. [15] introduced a black-box ofline-to-online framework for stochastic combinatorial multi-armed bandits with full-bandit feedback. The key condition is robustness of the ofline approximation algorithm to bounded errors in value-oracle evaluations. Under this condition, the ofline algorithm can be used as a black box and its approximation factor becomes the benchmark for sublinear α-regret. The framework was instantiated for several submodular settings, including cardinality and knapsack constraints. Some special cases of this approach are in [13, 7, 14]. The k-submodular results in [14] include ordinary set-submodularity as the $k _ { \mathrm { s u b } } = 1$ special case. Thus general-matroid full-bandit CMAB is not new in itself. The distinction here is the approximation factor: the existing general-matroid rows attain $1 / 2$ for monotone and $1 / 3$ for non-monotone objectives, whereas our resilient SGS-Poisson construction attains the classical $1 - 1 / e$ and $1 / e$ factors, respectively.

Fourati et al. [8] subsequently extended this interface to ofline algorithms whose guarantee is $\left( \alpha - \varepsilon \right)$ and whose oracle complexity depends polynomially on $1 / \varepsilon$ . They remove the ofline ε loss and yield α-regret. A special case of this framework can be seen in [9]. We use the approach of [8] in our CMAB result.

Table 1 summarizes these full-bandit results. We list only CMAB results and only constraints that are matroids or special cases of matroids. The cardinality constraint is the uniform matroid, and the unconstrained problem is the free matroid. Our two rows give the monotone and non-monotone results separately.

## 3 Problem Setup and Resilience

Let $M = ( U , { \mathcal { T } } )$ be a matroid of rank k and $n = | U |$ . A set function $f : 2 ^ { U }  [ 0 , 1 ]$ is submodular if

$$
f ( A ) + f ( B ) \geq f ( A \cup B ) + f ( A \cap B ) \qquad \forall A , B \subseteq U .
$$

Table 1: Stochastic full-bandit CMAB results for ordinary set-submodular rewards under matroid constraints or special cases.
<table><tr><td>Paper</td><td>Constraint</td><td>Objective</td><td>Approx.</td><td>Regret</td></tr><tr><td>Nie et al. [13]</td><td>Uniform (cardi- Monotone nality)</td><td></td><td> $1 - 1 / e$ </td><td> $\mathcal { O } ( n ^ { 1 / 3 } k ^ { 4 / 3 } T ^ { 2 / 3 } \log ^ { 1 / 2 } T )$ </td></tr><tr><td>Fourati et al. [9]</td><td>Uniform (cardi- Monotone nality)</td><td></td><td> $1 - 1 / e$ </td><td> $\widetilde { \mathcal { O } } ( n ^ { 1 / 3 } k ^ { 2 / 3 } T ^ { 2 / 3 } )$ </td></tr><tr><td>Fourati et al. [7]</td><td>Free</td><td>Non- monotone</td><td> $1 / 2$ </td><td> $\widetilde { \mathcal { O } } ( n T ^ { 2 / 3 } )$ </td></tr><tr><td>Fourati et al. [8]</td><td>Uniform (cardi- Non- nality)</td><td>monotone</td><td>1/e</td><td> $\widetilde { \mathcal { O } } ( n ^ { 1 / 5 } k ^ { 2 / 5 } T ^ { 4 / 5 } )$ </td></tr><tr><td>Nie et al. [14]</td><td>General troid</td><td>ma- Monotone</td><td>1/2</td><td> $\widetilde { \mathcal { O } } ( n ^ { 1 / 3 } k T ^ { 2 / 3 } )$ </td></tr><tr><td>Nie et al. [14]</td><td>General troid</td><td>ma- Non-</td><td>1/3 monotone</td><td> $\widetilde { \mathcal { O } } ( n ^ { 1 / 3 } k T ^ { 2 / 3 } )$ </td></tr><tr><td>This work</td><td>General ma- Monotone troid</td><td></td><td> ${ \bf 1 } - { \bf 1 } / { \bf e }$ </td><td> $\widetilde { \mathcal { O } } ( \mathbf { n } ^ { 1 / 5 } \mathbf { k } ^ { 4 / 5 } \mathbf { T } ^ { 4 / 5 } )$ </td></tr><tr><td>This work</td><td>General ma- Non- troid</td><td></td><td> $\mathbf { 1 / e }$  monotone</td><td> $\widetilde { \mathcal { O } } ( \mathbf { n } ^ { 1 / 5 } \mathbf { k } ^ { 4 / 5 } \mathbf { T } ^ { 4 / 5 } )$ </td></tr></table>

Note. The two [14] rows are results for $k _ { \mathrm { s u b } }$ -submodular bandits specialized to $k _ { \mathrm { s u b } } = 1$ , which recovers ordinary set-submodularity. Here k in the table denotes the rank of the matroid, whereas $k _ { \mathrm { s u b } }$ denotes the number of labels in k-submodularity.

We use the equivalent diminishing-returns notation

$$
f ( i \mid S ) : = f ( S \cup \{ i \} ) - f ( S ) ,
$$

for which $S \subseteq T$ and $i \not \in T$ imply

$$
f ( i \mid S ) \geq f ( i \mid T ) .
$$

Define

$$
{ \mathrm { O P T } } : = \operatorname* { m a x } _ { S \in { \mathcal { T } } } f ( S ) .
$$

We assume standard access to an independence oracle for the matroid. Maximum-weight bases in contracted matroids are computed using this access. As usual, we count only value-oracle calls in the resilience and CMAB complexity bounds; independence queries and the computation of maximum-weight bases are not counted.

For the online full-bandit application, at round t the learner chooses a feasible super-arm $A _ { t } \in \mathcal { T }$ and observes only an aggregate reward $Y _ { t } \in [ 0 , 1 ]$ with $\mathbb { E } [ Y _ { t } \mid A _ { t } = S ] = f ( S )$ . For $\alpha \in ( 0 , 1 ]$ , we define the cumulative α-regret by

$$
R _ { \alpha } ( T ) : = \alpha T \mathrm { O P T } - \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } Y _ { t } \right] .\tag{7}
$$

Only the aggregate reward is observed; no component-wise or semi-bandit feedback is assumed.

## 3.1 Controlled oracle

A ξ-controlled oracle is a function $\widehat { f } : 2 ^ { U } \to \mathbb { R }$ satisfying

$$
| { \widehat { f } } ( S ) - f ( S ) | \leq \xi \qquad \forall S \subseteq U .\tag{8}
$$

The perturbation is deterministic and may be adversarial; no stochastic or independence assumption is imposed. Because $\widehat { f }$ is a fixed function, the same set always receives the same oracle value; thus the adversarial perturbation is persistent throughout the execution.

We may project $\widehat { f }$ onto [0, 1] without increasing its error. Hence, throughout the proof we assume

$$
0 \leq { \widehat { f } } ( S ) \leq 1 .
$$

Then

$$
\left| | { \widehat { f } } ( i \mid S ) - f ( i \mid S ) | \leq 2 \xi . \right|\tag{9}
$$

## 3.2 Resilience definition

We use the $( \alpha , \beta , \gamma , \psi , \delta )$ resilience parameterization of [8], which extends the robust-approximation notion of Nie et al. [15] by recording the accuracy-dependent oracle complexity.

Definition 3.1 $( ( \alpha , \beta , \gamma , \psi , \delta )$ -resilience). An ofline algorithm $\mathcal { A } ( \varepsilon )$ is $( \alpha , \beta , \gamma , \psi , \delta )$ -resilient $i f ,$ for every $\xi \ge 0$ , under a ξ-controlled oracle it outputs a feasible Θ satisfying

$$
\begin{array} { r } { \mathbb { E } [ f ( \Theta ) ] \geq ( \alpha - \varepsilon ) \mathrm { O P T } - \delta \xi , } \end{array}\tag{10}
$$

and has expected oracle complexity at most

$$
N ( \varepsilon ) = \left\{ \begin{array} { l l } { \psi , } & { \varepsilon = 0 , } \\ { \psi \varepsilon ^ { - \beta } \log ^ { \gamma } ( 1 / \varepsilon ) , } & { \varepsilon > 0 } \end{array} \right.\tag{11}
$$

oracle calls, up to universal constant factors.

Remark 3.2. For $\beta > 0$ , the ofline-to-online reduction only invokes $\mathcal { A } ( \varepsilon )$ with $\varepsilon > 0$ . Thus the $\varepsilon = 0$ branch in the definition is a convention and is not used in the CMAB application.

## 4 Algorithms and Exact-Oracle Guarantees

This section makes explicit the exact algorithmic object whose resilience we study. We use the SGS-Poisson algorithm of Kulik et al. [12] as a black-box base process. Its Poisson rate, valid-swap conditions, and spiteful drop step are unchanged.

## 4.1 Multilinear extension

For $x \in [ 0 , 1 ] ^ { U }$ , let $R _ { x }$ include each element i independently with probability $x _ { i }$ . Define

$$
F ( x ) : = \mathbb { E } [ f ( R _ { x } ) ] .
$$

For a set A and $t \in [ 0 , 1 ]$ , write $t \mathbf { 1 } _ { A }$ for its scaled indicator. The quantity

$$
F ( t \mathbf { 1 } _ { A } \lor \mathbf { 1 } _ { i } ) - F ( t \mathbf { 1 } _ { A } )
$$

is the marginal used by SGS-Poisson.

## 4.2 Valid swaps

Definition 4.1 (Valid swap). For a matroid of rank r (equal to k before contraction and to $k ^ { \prime }$ after contraction), let $A \in \mathcal { T }$ and $t \in ( 0 , 1 ]$ . A random pair

(I, J) ∈ (A ∪ {⊥}) × U   
is a valid swap $i f { \mathrm { : } }$   
(i) $A - I + J \in { \mathcal { I } }$ almost surely;   
(ii) $i f J \in A$ , then $I = J ;$   
(iii) Pr(I = i) = 1/r for every $i \in A ,$   
(iv) Pr(J = j) ≤ 1/r for every $j \in U$   
It is η-almost-above-average if, for an optimal base O,   
E[F(t1<sub>A</sub> ∨ 1<sub>J</sub>) − F(t1<sub>A</sub>)] ≥ (F(t1<sub>A</sub> ∨ 1<sub>O</sub>) − F(t1<sub>A</sub>)) − η (12)   
r

Algorithm 1 SGS-Poisson (base algorithm)   
Require: Matroid M, starting time $\varepsilon _ { 0 } > 0 .$ , valid swap procedure   
1: $t  \varepsilon _ { 0 } , A  \varnothing$   
2: while $t < 1$ do   
3: Sample the next event time $\tau ( t )$ of a Poisson process with rate $k / t$   
4: t ← τ (t)   
5: (I, J) ← Swap(t, A)   
6: $A  A - I + J$   
7: if $I = J$ then   
8: With probability t, set $A  A - I$   
9: end if   
10: end while   
11: return A

The process in Algorithm 1 is exactly the SGS-Poisson process of Kulik et al. [12]. In Algorithm 1, k denotes the rank of the matroid on which the process is currently run; after contraction this is $k ^ { \prime } \leq k$ In particular, the spiteful drop in the fourth line is not a modification introduced here. The expected number of swap calls is $k \log ( 1 / \varepsilon _ { 0 } )$

## 4.3 Poisson-process guarantee

We use the following theorem from Kulik et al. [12] as a black-box analytical guarantee for the unchanged Poisson process.

Proposition 4.2 (SGS-Poisson with an almost-above-average swap; Kulik et al. [12]). Suppose Algorithm 1 uses a right-continuous η-almost-above-average valid swap. Then

$$
\mathbb { E } [ f ( A ) ] \geq \left\{ \begin{array} { l l } { ( 1 - \varepsilon _ { 0 } ) e ^ { - 1 } \mathrm { O P T } + e ^ { - 1 } f ( \varnothing ) - \eta , } & { f ~ n o n - m o n o t o n e , } \\ { ( 1 - \varepsilon _ { 0 } ) ( 1 - e ^ { - 1 } ) \mathrm { O P T } + e ^ { - 1 } f ( \varnothing ) - \eta , } & { f ~ m o n o t o n e . } \end{array} \right.\tag{13}
$$

The exact SGS-Poisson value-oracle implementation requires an estimate of the optimum and a bound on the maximum sum of residual marginals. A naive noisy greedy construction is not valid for general non-monotone matroid maximization. We therefore use Residual Random Greedy.

## 4.4 Residual Random Greedy

Add a set D of k dummy elements and form the rank-k augmentation

$$
U ^ { + } : = U \cup D , \qquad \mathbb { Z } ^ { + } : = \{ S \subseteq U ^ { + } : S \cap U \in \mathbb { Z } , \ | S | \leq k \} .
$$

Equivalently, $M ^ { + } = ( U ^ { + } , \mathbb { Z } ^ { + } )$ is the rank-k truncation of the direct sum of M and the free matroid on D. In particular, $\operatorname { r a n k } ( M ^ { + } ) = k$ , and every independent set of M can be completed to a base of $M ^ { + }$ by adding dummy elements. Extend the objective and controlled oracle by

$$
f ^ { + } ( S ) : = f ( S \setminus D ) , \qquad { \widehat { f } } ^ { + } ( S ) : = { \widehat { f } } ( S \setminus D ) .
$$

Hence every dummy element has identically zero true and estimated marginal, and an optimal solution of the original instance can be extended to an optimal base of $M ^ { + }$ without changing its value. From this point through the preprocessing and Poisson stages, we run the algorithm on the augmented instance $\bar { ( } M ^ { + } , U ^ { + } , f ^ { \bar { + } } , \widehat { f } ^ { + } )$ and suppress the superscript + on $M , f , { \widehat { f } }$ for notation. Thus the rank remains $k ,$ the optimum value remains $\mathrm { O P T }$ , and the ground-set size is $n + k = O ( n )$ because $k \leq n$ . The k dummy elements sufice because after t selections the residual rank is $k - t ,$ while at least $k - t$ dummy slots remain, so every residual independent set can be completed to a residual base. At the end, dummy elements are discarded; this preserves feasibility and objective value.

Run Residual Random Greedy (RRG) [3] for $r = \lceil k / 2 \rceil$ iterations. At iteration $i ,$ let $S _ { i - 1 }$ be the current set and $r _ { i } = k - i + 1$ the residual rank. In the contraction $M / S _ { i - 1 }$ , compute a maximum-weight base $M _ { i }$ with weights $\widehat { w } _ { i } ( u ) = \widehat { f } ( u \mid S _ { i - 1 } )$ , and choose $u _ { i }$ uniformly from $M _ { i }$

Algorithm 2 Residual Random Greedy, truncated to $\lceil k / 2 \rceil$ iterations [3]   
Require: M, value oracle h   
1: $S \gets \emptyset$   
2: for $i = 1 , \ldots , \lceil k / 2 \rceil$ do   
3: Compute a maximum-weight base $M _ { i }$ of $M / S$   
with weights h(u | S)   
4: Draw u uniformly from $M _ { i }$   
5: $S \gets S \cup \{ u \}$   
6: end for   
7: return $S$

We stop after $\lceil k / 2 \rceil$ iterations because the RRG analysis already gives the constant-factor guarantee needed here at that point.

## 4.5 Advanced preprocessing and contraction

For an independent set $Q \in { \mathcal { T } } .$ , define the residual marginal mass

$$
\operatorname { M a r } _ { f } ( Q ) : = \operatorname* { m a x } _ { \stackrel { T \subseteq U } { T \cup { \bar { Q } } } } \sum _ { j \in T } f ( j \mid Q ) .\tag{14}
$$

We also use the global maximum singleton-marginal mass

$$
\operatorname { M a r } ( f , { \mathcal { T } } ) : = \operatorname { M a r } _ { f } ( { \mathcal { O } } ) = \operatorname* { m a x } _ { T \in { \mathcal { T } } } \sum _ { j \in T } f ( j \mid { \mathcal { O } } ) .\tag{15}
$$

For a contracted instance $( g , M ^ { \prime } )$ , we analogously write $\mathrm { M a r } ( g , M ^ { \prime } )$ for the global quantity $\operatorname { M a r } ( g , { \mathcal { T } } ( M ^ { \prime } ) )$ The exact SGS-Poisson value-oracle implementation of Kulik et al. [12] first constructs a constant-factor upper estimate V of OPT and then performs Advanced Preprocessing. Starting from $Q _ { 0 } = \emptyset$ , while Mar $_ f ( Q _ { t } ) > 2 0 V$ , it chooses a maximum-weight residual base under the true marginals and adds a uniformly random element of that base. The resulting set is denoted by S<sup>¯</sup>.

The dummy-element extension described above guarantees that every residual independent set can be completed to a base, so the process has at most k iterations. The exact-oracle analysis shows that the stopping rule supplies the bounded-residual-marginal condition required by the subsequent swap process. In Section 5 we show that the same preprocessing rule, with exact values replaced by the given controlled oracle, retains this certificate up to an additive $O ( k \xi )$ term. Importantly, this is an analysis of the same preprocessing process rather than a new algorithm.

After preprocessing, contract the matroid at S<sup>¯</sup>:

$$
M ^ { \prime } : = M / { \bar { S } } , \qquad g ( T ) : = f ( T \cup { \bar { S } } ) .\tag{16}
$$

The contracted matroid has rank $k ^ { \prime } \leq k$ . The SGS-Poisson process is run with its actual rank $k ^ { \prime } ;$ throughout the analysis we upper bound $k ^ { \prime }$ by the original rank k. Then $g$ is nonnegative and submodular on $M ^ { \prime }$ . Using the dummy elements, we may and do take $O _ { g }$ to be an optimal base of the contracted instance; all dummy elements have zero g-marginal.

## 4.6 Value-oracle implementation of the swap rule

At time t, for current set $A \in \mathbb { Z } ( M ^ { \prime } )$ and candidate element $i ,$ the SGS-Poisson implementation of Kulik et al. [12] uses the multilinear marginal

$$
w _ { i } = F ( t \mathbf { 1 } _ { A } \lor \mathbf { 1 } _ { i } ) - F ( t \mathbf { 1 } _ { A } ) .\tag{17}
$$

It estimates $w _ { i }$ by drawing independent sets $R _ { 1 } , \dots , R _ { m } \sim t { \bf 1 } _ { A }$ and averaging

$$
\widetilde { w } _ { i } = \frac { 1 } { m } \sum _ { \ell = 1 } ^ { m } \big [ g ( R _ { \ell } \cup \{ i \} ) - g ( R _ { \ell } ) \big ] .\tag{18}
$$

The exact SGS-Poisson value-oracle implementation chooses a maximum-weight residual base under these estimates, constructs a matroid exchange map, and samples the entering element uniformly from that base. A fixed deterministic tie-breaking order makes the rule right-continuous. The concentration analysis of the exact algorithm controls the base sum of these estimated marginals, which is the quantity needed for the almost-above-average condition. Section 5 shows that this same implementation remains valid under the controlled oracle.

The augmented instance is used throughout preprocessing and the Poisson process. Its dummy elements have identically zero true and controlled-oracle marginals. At the end they are discarded. When $k = 1$ , the direct enumeration described below gives a stronger guarantee; hence the analysis below assumes $k \geq 2$ . All maximum-base ties use a fixed deterministic ordering independent of time, so the swap rule is right-continuous. The Poisson process is run in the contracted matroid and returns $S \in { \mathcal { T } } ( M / { \bar { S } } )$ , so $S \cup \bar { S }$ is feasible in the original matroid.

## 5 Robustness of the Main Algorithm

We now analyze the algorithm of Section 4 under the controlled oracle of Section 3. The SGS-Poisson process, valid-swap rule, spiteful drop, and Advanced Preprocessing recursion are unchanged from Kulik et al. [12]. The only robustification is that the constant-factor certificate V is obtained by running RRG [3] with the controlled oracle and amplifying it to obtain a realized lower bound on OPT; all subsequent decisions use the same procedures with the available oracle. The purpose of this section is solely to prove that the exact SGS-Poisson guarantee survives these perturbations.

## 5.1 Robust Residual Random Greedy and the optimum certificate

We first show that the same Residual Random Greedy procedure remains a constant-factor procedure when its value queries are answered by ${ \widehat { f } } .$ The RRG exchange coupling is unchanged; only the maximumbase comparison is afected by the oracle perturbation.

Lemma 5.1 (Robust maximum-base comparison). At iteration $i ,$ for every base B of the residual matroid,

$$
\sum _ { u \in M _ { i } } f ( u \mid S _ { i - 1 } ) \geq \sum _ { u \in B } f ( u \mid S _ { i - 1 } ) - 4 r _ { i } \xi .\tag{19}
$$

Proof. By optimality under ${ \widehat { f } } ,$

$$
\sum _ { u \in M _ { i } } { \widehat { f } } ( u \mid S _ { i - 1 } ) \geq \sum _ { u \in B } { \widehat { f } } ( u \mid S _ { i - 1 } ) .
$$

Each base has $r _ { i }$ elements, and (9) gives an error of at most $2 r _ { i } \xi$ on either side. Rearranging gives (19). □

We next prove the exchange coupling used by $\operatorname { R R G } ;$ no additional property of the noisy oracle is needed.

Lemma 5.2 (RRG exchange coupling). Consider the augmented matroid $M ^ { + }$ and let $O ^ { + }$ be an optimal base containing an optimal solution of the original instance (together with dummy elements). The RRG coupling can be chosen so that, for $0 \leq i \leq k - 1$ ，

$$
\mathbb { E } [ f ^ { + } ( S _ { i } \cup O _ { i } ) ] \geq \frac { ( k - i ) ( k - i - 1 ) } { k ( k - 1 ) } \mathrm { O P T } ,\tag{20}
$$

where $O _ { 0 } = O ^ { + } , O _ { i } \subseteq O _ { i - 1 }$ , and $S _ { i } \cup O _ { i }$ is a base of $M ^ { + }$ almost surely.

Proof. Fix $i < k$ and condition on the history through iteration $i - 1$ . Put $r _ { i } = k - i + 1$ . The set $M _ { i }$ is a base of the contracted matroid $M ^ { + } / S _ { i - 1 }$ , and $O _ { i - 1 }$ is another base of the same contraction because $S _ { i - 1 } \cup O _ { i - 1 }$ is a base of $M ^ { + }$ . By the strong basis-exchange property, there is a bijection

$$
h _ { i } : M _ { i } \longrightarrow O _ { i - 1 }
$$

such that

$$
S _ { i - 1 } \cup O _ { i - 1 } - h _ { i } ( u ) + u \in \mathbb { Z } ( M ^ { + } ) \qquad \forall u \in M _ { i } ,
$$

with $h _ { i } ( u ) = u$ whenever $u \in M _ { i } \cap O _ { i - 1 }$ . After drawing $u _ { i }$ uniformly from $M _ { i } .$ , set

$$
O _ { i } : = O _ { i - 1 } \backslash \{ h _ { i } ( u _ { i } ) \} .
$$

Thus $S _ { i } \cup O _ { i }$ is a base.

We use the following elementary consequence of submodularity. If B is an independent set and a random set R satisfies

$$
\operatorname* { P r } ( a \in R ) \geq p \quad ( a \in B ) , \qquad \operatorname* { P r } ( a \in R ) \leq q \quad ( a \not \in B ) ,
$$

then, for nonnegative submodular $f ^ { + }$

$$
\mathbb { E } [ f ^ { + } ( R ) ] \geq ( p - q ) f ^ { + } ( B ) .\tag{21}
$$

Indeed, if $p \leq q$ the claim follows from nonnegativity. If $p > q$ , let $\alpha _ { a } = \operatorname* { P r } ( a \in R )$ . The Lovasz extension $f _ { L } ^ { + }$ is a convex extension of $f ^ { + }$ , hence

$$
\mathbb { E } [ f ^ { + } ( R ) ] \geq f _ { L } ^ { + } ( \alpha ) .
$$

For every threshold $s \in ( q , p ]$ , we have $\alpha _ { a } \geq s$ for every $a \in B$ and $\alpha _ { a } < s$ for every $a \notin B$ . Thus the threshold set

$$
\left\{ a : \alpha _ { a } \geq s \right\} = B \qquad { \mathrm { f o r ~ e v e r y ~ } } s \in ( q , p ] .
$$

Using the threshold-set representation of the Lovasz extension and nonnegativity of $f ^ { + }$ for all other thresholds,

$$
f _ { L } ^ { + } ( \alpha ) \geq \int _ { q } ^ { p } f ^ { + } ( B ) d s = ( p - q ) f ^ { + } ( B ) .
$$

Therefore $\mathbb { E } [ f ^ { + } ( R ) ] \ge ( p - q ) f ^ { + } ( B )$

Now take $B = S _ { i - 1 } \cup O _ { i - 1 }$ . Every element of B is absent from $S _ { i } \cup O _ { i }$ only if it is the image under $h _ { i }$ of the sampled element, so its removal probability is at most $1 / r _ { i }$ . Every element outside B can enter only if it is the sampled element, so its inclusion probability is at most $1 / r _ { i }$ . Applying (21) with $p = 1 - 1 / r _ { i }$ and $q = 1 / r _ { i }$ gives

$$
\mathbb { E } [ f ^ { + } ( S _ { i } \cup O _ { i } ) \mid { \mathcal { F } } _ { i - 1 } ] \ge \left( 1 - { \frac { 2 } { r _ { i } } } \right) f ^ { + } ( S _ { i - 1 } \cup O _ { i - 1 } ) .
$$

Iterating this inequality for i steps yields

$$
\prod _ { \ell = 1 } ^ { i } \left( 1 - { \frac { 2 } { k - \ell + 1 } } \right) = { \frac { ( k - i ) ( k - i - 1 ) } { k ( k - 1 ) } } ,
$$

which proves the lemma.

Lemma 5.3 (Robust RRG). For $1 \leq i \leq \lceil k / 2 \rceil$ ，

$$
\mathbb { E } [ f ^ { + } ( S _ { i } ) ] \geq { \frac { i ( k - i ) } { k ( k - 1 ) } } { \mathrm { O P T } } - 4 i \xi .\tag{22}
$$

In particular, for $r = \lceil k / 2 \rceil$

$$
\mathbb { E } [ f ( S _ { r } ) ] \geq { \frac { 1 } { 4 } } \mathrm { O P T } - 4 k \xi .\tag{23}
$$

Proof. Let $r _ { i } = k - i + 1$ . Conditional on the history through iteration $i - 1$ , Lemma 5.1 with $B = O _ { i - 1 }$ gives

$$
\begin{array} { r l r } {  { \mathbb { E } [ f ^ { + } ( S _ { i } ) - f ^ { + } ( S _ { i - 1 } ) \mid \mathcal { F } _ { i - 1 } ] = \frac { 1 } { r _ { i } } \sum _ { u \in M _ { i } } f ^ { + } ( u \mid S _ { i - 1 } ) } } \\ & { } & { \geq \frac { f ^ { + } ( S _ { i - 1 } \cup O _ { i - 1 } ) - f ^ { + } ( S _ { i - 1 } ) } { r _ { i } } - 4 \xi . } \end{array}
$$

Taking expectations and applying Lemma 5.2 at $i - 1$

$$
\mathbb { E } [ f ^ { + } ( S _ { i } ) ] \geq \frac { k - i } { k - i + 1 } \mathbb { E } [ f ^ { + } ( S _ { i - 1 } ) ] + \frac { k - i } { k ( k - 1 ) } \mathrm { O P T } - 4 \xi .
$$

An induction on $i ,$ starting from $S _ { 0 } = \emptyset$ , gives (22). Finally,

$$
\frac { r ( k - r ) } { k ( k - 1 ) } \geq \frac 1 4
$$

for $r = \lceil k / 2 \rceil$ and $k \geq 2$ , while $f ^ { + }$ agrees with $f$ on sets after deleting dummy elements. This proves (23). □

For $k = 1$ , querying all feasible singleton sets $\mathrm { g i }$ ves a stronger $1 - O ( \xi )$ estimate, so the remainder assumes $k \geq 2$

## 5.2 High-probability optimum certificate

The expectation guarantee in Lemma 5.3 is not enough because Advanced Preprocessing uses a realized threshold. We therefore amplify.

Run Algorithm 2 independently

$$
R = \left\lceil 1 5 \log \frac { 1 } { \rho } \right\rceil
$$

times, and let $G ^ { \star }$ be the set with largest $\widehat { f }$ value.

If $\mathrm { { O P T } = 0 } { } _ { ; }$ , the lower bound $\mathrm { O P T } \le \widehat { V }$ is trivial. Hence assume $\mathrm { O P T > 0 }$ in the amplification argument below. If $\mathrm { O P T } \geq 3 2 k \xi$ , then (23) implies

$$
\mathbb { E } [ f ( S _ { r } ) ] \geq { \frac { 1 } { 8 } } \mathrm { O P T } .
$$

Since $0 \leq f ( S _ { r } ) \leq \mathrm { O P T }$ ，

$$
\mathbb { P } \left( f ( S _ { r } ) \geq { \frac { 1 } { 1 6 } } \mathrm { O P T } \right) \geq { \frac { 1 } { 1 5 } } .\tag{24}
$$

Thus, with probability at least $1 - \rho ,$ one run has value at least $\mathrm { O P T } / 1 6$ . Define

$$
\widehat { V } : = \operatorname* { m a x } \left\{ 3 2 k \xi , 1 6 \widehat { f } ( G ^ { \star } ) + 6 4 \xi \right\} .\tag{25}
$$

On the good event,

$$
\mathrm { O P T } \leq { \widehat { V } } .
$$

Moreover, because $G ^ { \star }$ is feasible,

$$
{ \widehat { f } } ( G ^ { \star } ) \leq \mathrm { O P T } + \xi ,
$$

and hence

$$
\boxed { \mathrm { O P T } \leq \widehat { V } \leq 1 6 \mathrm { O P T } + 9 6 k \xi }\tag{26}
$$

with probability at least $1 - \rho .$ . If $\mathrm { O P T } < 3 2 k \xi$ , the first term in (25) already gives $\widehat { V } \geq \mathrm { O P T }$

## 5.3 Robust adaptive preprocessing

Lemma 5.4 (Marginal-mass perturbation). For every $Q \in \mathcal { I }$

$$
\lvert \mathrm { M a r } _ { \hat { f } } ( Q ) - \mathrm { M a r } _ { f } ( Q ) \rvert \leq 2 k \xi .\tag{27}
$$

Proof. Every feasible $T$ contains at most k elements. The diference between its estimated and true marginal sum is at most 2kξ by (9). Taking maxima in both directions proves the result. □

Starting from $Q _ { 0 } = \emptyset$ , define the stopping time

$$
\tau : = \operatorname* { i n f } \{ t : \operatorname { M a r } _ { \widehat { f } } ( Q _ { t } ) \leq 2 0 \widehat { V } \} .\tag{28}
$$

For $t < \tau$ , choose a maximum-weight residual base $Z _ { t }$ under the estimated marginals and draw $j _ { t }$ uniformly from $Z _ { t }$ . Set $Q _ { t + 1 } = Q _ { t } + j _ { t }$

Because of the dummy elements, every residual independent set can be extended to a base. Thus this process has at most k iterations.

Lemma 5.5 (Residual marginal bound). On the event in (26),

$$
\begin{array} { r } { \big | \mathrm { M a r } _ { f } ( \bar { S } ) \leq 3 2 0 \mathrm { O P T } + 1 9 2 2 k \xi , \big | } \end{array}\tag{29}
$$

where $\bar { S } = Q _ { \tau }$

Proof. At stopping,

$$
\mathrm { M a r } _ { \hat { f } } ( \bar { S } ) \leq 2 0 \widehat { V } .
$$

By Lemma 5.4 and (26),

$$
\operatorname { M a r } _ { f } ( { \bar { S } } ) \leq 2 0 ( 1 6 \mathrm { O P T } + 9 6 k \xi ) + 2 k \xi ,
$$

which gives (29).

The crucial result is that the preprocessing certificate itself is resilient.

Let $O _ { 0 } = O$ be an optimal base. By basis exchange, construct $O _ { t } \subseteq O$ so that $Q _ { t } \cup O _ { t }$ is a base at every time. Define

$$
M _ { t } = f ( Q _ { t } \cup O _ { t } ) + { \frac { 1 } { 2 } } f ( Q _ { t } ) .\tag{30}
$$

Lemma 5.6 (Resilient adaptive-preprocessing drift). On the event $\widehat { V } \geq \mathrm { O P T }$ , for every $t < \tau$

$$
\left| \mathbb { E } [ M _ { t + 1 } - M _ { t } \mid { \mathcal { F } } _ { t } ] \geq { \frac { 8 { \mathrm { O P T } } - k \xi } { k - t } } . \right|\tag{31}
$$

Proof. Fix $t < \tau$ and write $r _ { t } = k - t .$ Conditional on $\mathcal { F } _ { t }$ , the current set $Q _ { t } ,$ the estimated marginal weights, and the selected maximum weight base $Z _ { t }$ are fixed. Because the stopping condition has not fired,

$$
\sum _ { j \in Z _ { t } } { \widehat { f } } ( j \mid Q _ { t } ) > 2 0 { \widehat { V } } \geq 2 0 \mathrm { O P T } .
$$

The base $Z _ { t }$ has $r _ { t }$ elements, so by (9),

$$
\sum _ { j \in Z _ { t } } f ( j \mid Q _ { t } ) \geq 2 0 \mathrm { O P T } - 2 r _ { t } \xi .
$$

Since $j _ { t }$ is uniform in $Z _ { t }$ ,

$$
\mathbb { E } [ f ( Q _ { t + 1 } ) - f ( Q _ { t } ) \mid { \mathcal { F } _ { t } } ] \ge \frac { 2 0 0 \mathrm { P T } - 2 r _ { t } \xi } { r _ { t } } .\tag{32}
$$

We next make the exchange argument explicit. Let $O _ { t }$ be the current exchange set, so $Q _ { t } \cup O _ { t }$ is a base. By the strong basis-exchange property, there is a bijection

$$
h _ { t } : Z _ { t } \longrightarrow O _ { t }
$$

such that

$$
Q _ { t } \cup O _ { t } - h _ { t } ( j ) + j \in \mathcal { T } \qquad \forall j \in Z _ { t } ,
$$

with $h _ { t } ( j ) = j$ on $Z _ { t } \cap O _ { t }$ . After drawing $j _ { t }$ uniformly from $Z _ { t } ,$ set

$$
O _ { t + 1 } : = O _ { t } \setminus \{ h _ { t } ( j _ { t } ) \} .
$$

Then $Q _ { t + 1 } \cup O _ { t + 1 }$ is a base.

Every element of $Q _ { t } \cup O _ { t }$ is removed from the next base with probability at most $1 / r _ { t }$ , while every element outside $Q _ { t } \cup O _ { t }$ is inserted with probability at most $1 / r _ { t }$ . Applying (21) with $p = 1 - 1 / r _ { t }$ and $q = 1 / r _ { t }$ gives

$$
\mathbb { E } [ f ( Q _ { t + 1 } \cup O _ { t + 1 } ) \mid { \mathcal { F } } _ { t } ] \geq \left( 1 - { \frac { 2 } { r _ { t } } } \right) f ( Q _ { t } \cup O _ { t } ) .\tag{33}
$$

For $r _ { t } = 1$ the right-hand side is nonpositive, so the same inequality holds trivially by nonnegativity.

Combining (32) and (33),

$$
\begin{array} { r } { \mathbb { E } \left[ M _ { t + 1 } - M _ { t } \ | \ \mathcal { F } _ { t } \right] \geq - \frac { 2 } { r _ { t } } f ( Q _ { t } \cup O _ { t } ) + \frac { 1 0 \mathrm { O P T } - r _ { t } \xi } { r _ { t } } } \\ { = \frac { 1 0 \mathrm { O P T } - r _ { t } \xi - 2 f ( Q _ { t } \cup O _ { t } ) } { r _ { t } } . } \end{array}
$$

Since $Q _ { t } \cup O _ { t }$ is feasible, $f ( Q _ { t } \cup O _ { t } ) \leq \mathrm { O P T }$ . Hence

$$
\mathbb { E } [ M _ { t + 1 } - M _ { t } \ | \ \mathcal { F } _ { t } ] \geq \frac { 8 \mathrm { O P T } - r _ { t } \xi } { r _ { t } } \geq \frac { 8 \mathrm { O P T } - k \xi } { k - t } .
$$

Theorem 5.7 (Resilient preprocessing certificate). Let E be any event measurable with respect to the initial RRG randomness such that ${ \mathcal { E } } \subseteq \{ { \widehat { V } } \geq \operatorname { O P T } \}$ . Then the output S<sup>¯</sup> of Advanced Preprocessing satisfies

$$
\lceil \mathbb { E } [ \operatorname* { m a x } _ { T : T \cup \bar { S } \in \mathcal { T } } f ( T \cup \bar { S } ) + \frac { 1 } { 2 } f ( \bar { S } ) \rceil \mathcal { E } ] \geq \mathrm { O P T } - C _ { 0 } k \xi . \rceil\tag{34}
$$

Proof. Condition on any such event E. On E and when $\mathrm { O P T } \geq k \xi / 8$ , define $\widetilde { M } _ { t } : = M _ { t \wedge \tau }$ . Lemma 5.6 implies that $( \widetilde { M } _ { t } ) _ { t \geq 0 }$ is a submartingale under the conditional probability given E. Since $\tau \leq k$ and $0 \leq M _ { t } \leq 3 / 2$ , the stopped submartingale is bounded, so the elementary bounded optional-stopping theorem gives

$$
\mathbb { E } [ M _ { \tau } \mid \mathcal { E } ] \ge M _ { 0 } = f ( O ) + \frac { 1 } { 2 } f ( \emptyset ) \ge \mathrm { O P T } .
$$

If $\mathrm { O P T } < k \xi / 8$ , nonnegativity gives

$$
\mathbb { E } [ M _ { \tau } \mid \mathcal { E } ] \ge 0 \ge \mathrm { O P T } - \frac { 1 } { 8 } k \xi .
$$

Thus in all cases

$$
\mathbb { E } [ M _ { \tau } \ | \ \mathcal { E } ] \ge \mathrm { O P T } - C _ { 0 } k \xi
$$

for a universal $C _ { 0 }$

Since $Q _ { \tau } = \bar { S }$ and $O _ { \tau }$ is feasible in the contraction,

$$
f ( { \bar { S } } \cup O _ { \tau } ) \leq \operatorname* { m a x } _ { T : T \cup { \bar { S } } \in { \mathcal { T } } } f ( T \cup { \bar { S } } ) .
$$

Because

$$
{ \cal M } _ { \tau } = f ( \bar { S } \cup { \cal O } _ { \tau } ) + \frac { 1 } { 2 } f ( \bar { S } ) ,
$$

the claimed certificate follows.

Why this lemma is the technical centerpiece. The proof does not compare the controlled-oracle trajectory to the exact trajectory. The two trajectories may diverge completely. Instead, the potential is shown to retain positive drift on the controlled-oracle trajectory itself. This is the reason the result is not a routine Lipschitz perturbation argument.

## 5.4 Robust almost-above-average swaps

After preprocessing, contract the matroid at $\bar { S } { : }$

$$
M ^ { \prime } : = M / { \bar { S } } , \qquad g ( T ) : = f ( T \cup { \bar { S } } ) .\tag{35}
$$

Then $g$ is nonnegative and submodular. Let $O _ { g }$ maximize $g$ in $M ^ { \prime }$ .

By Theorem 5.7, on the event $\mathcal { E } .$

$$
{ \mathbb E } \left[ g ( O _ { g } ) + \frac { 1 } { 2 } g ( \emptyset ) \mid \mathcal { E } \right] \geq \mathrm { O P T } - C _ { 0 } k \xi .\tag{36}
$$

By Lemma 5.5,

$$
\mathrm { M a r } ( g , M ^ { \prime } ) \leq 3 2 0 \mathrm { O P T } + 1 9 2 2 k \xi .\tag{37}
$$

The swap estimator. At time $t ,$ current set $A \in \mathbb { Z } ( M ^ { \prime } )$ , and candidate element $i ,$ define

$$
w _ { i } = F ( t \mathbf { 1 } _ { A } \lor \mathbf { 1 } _ { i } ) - F ( t \mathbf { 1 } _ { A } ) .\tag{38}
$$

Sample

$$
R _ { 1 } , \dots , R _ { m } \sim t { \bf 1 } _ { A }
$$

independently. Estimate

$$
\widetilde { w } _ { i } = \frac { 1 } { m } \sum _ { \ell = 1 } ^ { m } \bigl ( g ( R _ { \ell } \cup \{ i \} ) - g ( R _ { \ell } ) \bigr ) .\tag{39}
$$

The exact SGS-Poisson implementation chooses a maximum-weight base under $\widetilde { w } _ { i }$ , constructs a matroid exchange map, and samples uniformly from that base.

We use exactly this implementation, replacing exact function values by the controlled oracle induced on the contracted instance,

$$
{ \widehat { g } } ( T ) : = { \widehat { f } } ( T \cup { \bar { S } } ) .
$$

Then

$$
| \widehat { g } ( T ) - g ( T ) | \leq \xi , \qquad | \widehat { g } ( i \mid T ) - g ( i \mid T ) | \leq 2 \xi .
$$

Thus the same controlled-oracle model is inherited after contraction.

Lemma 5.8 (Right-continuity of the controlled swap). The controlled-oracle implementation of the SGS-Poisson swap is right-continuous in the sense required by Proposition 4.2.

Proof. Right-continuity is inherited from the sampling, event-time, exchange-map, and deterministic tie-breaking convention of the SGS-Poisson implementation of Kulik et al. [12]. For example, the sampled sets can be generated from fixed independent uniform random variables by thresholding at t, which gives a right-continuous version of the sampling and marginal processes; together with deterministic tie-breaking, this gives a right-continuous exchange rule. Replacing exact oracle values by $\widehat g$ changes only the numerical weights used in the same selection rule and does not change this temporal convention. Thus the controlled-oracle implementation has the required right-continuous version of the swap process. No continuity or submodularity property of $\widehat g$ is required. □

Lemma 5.9 (Robust swap concentration). Fix $\delta _ { s } \in ( 0 , 1 / 2 )$ . Use the same value-oracle sampling, maximum-weight-base selection, exchange-map construction, and fixed tie-breaking convention as the exact SGS-Poisson implementation of Kulik et al. [12], but evaluate every set through the controlled oracle ${ \widehat { g } } .$ . If

$$
m = O \left( \frac { k \log n + \log ( 1 / \delta _ { s } ) } { \delta _ { s } ^ { 2 } } \right) ,\tag{40}
$$

then the resulting swap is valid and right-continuous and satisfies

$$
\begin{array} { r } { \left| \eta \leq C _ { 1 } \delta _ { s } \big ( \mathrm { M a r } ( g , M ^ { \prime } ) + g ( O _ { g } ) \big ) + C _ { 2 } k \xi . \right| } \end{array}\tag{41}
$$

The constants $C _ { 1 } , C _ { 2 }$ are universal.

Proof. Validity is purely combinatorial. Once a maximum-weight base Z has been selected, the deterministic exchange map used by the exact implementation satisfies the valid-swap conditions. Right-continuity follows from Lemma 5.8.

Fix a swap time and condition on the complete history H immediately before the fresh samples are drawn. Thus A, t and the contracted matroid are fixed. For every independent set $Z$ of the contracted matroid, define

$$
X _ { \ell } ( Z ) : = \sum _ { i \in Z } \bigl ( g ( R _ { \ell } \cup \{ i \} ) - g ( R _ { \ell } ) \bigr ) ,
$$

where $R _ { \ell } \sim t { \bf 1 } _ { A }$ independently. Its population expectation is

$$
W _ { g } ( Z ) : = \operatorname { \mathbb { E } } [ X _ { \ell } ( Z ) \mid \mathcal { H } ] = \sum _ { i \in Z } \left( F ( t \mathbf { 1 } _ { A } \lor \mathbf { 1 } _ { i } ) - F ( t \mathbf { 1 } _ { A } ) \right) .
$$

We also define the empirical true base sum

$$
{ \overline { { W } } } _ { g } ( Z ) : = { \frac { 1 } { m } } \sum _ { \ell = 1 } ^ { m } X _ { \ell } ( Z ) .
$$

We first establish a uniform concentration event for the true marginals. Let

$$
Z ^ { + } ( R ) : = \{ i \in Z : g ( i \mid R ) \geq 0 \} .
$$

Since $Z ^ { + } ( R )$ is independent and submodularity gives $g ( i \mid R ) \leq g ( i \mid \emptyset )$ 2

$$
X _ { \ell } ( Z ) \leq \sum _ { i \in Z ^ { + } ( R _ { \ell } ) } g ( i  { | } R _ { \ell } ) \leq \mathrm { M a r } ( g , M ^ { \prime } ) .
$$

For the lower bound, submodularity gives

$$
X _ { \ell } ( Z ) \geq g ( R _ { \ell } \cup Z ) - g ( R _ { \ell } ) \geq - g ( R _ { \ell } ) \geq - g ( O _ { g } ) ,
$$

because $R _ { \ell } \subseteq A$ is independent and $g ( O _ { g } )$ is the optimum value of the contracted instance. Hence

$$
- g ( O _ { g } ) \leq X _ { \ell } ( Z ) \leq \mathrm { M a r } ( g , M ^ { \prime } ) \qquad \forall Z , \ell .\tag{42}
$$

Set

$$
L : = \operatorname { M a r } ( g , M ^ { \prime } ) + g ( O _ { g } ) .
$$

If $L = 0 ;$ , then $g ( O _ { g } ) = 0$ and the contracted instance has zero optimum, so the almost-above-average condition is immediate. Hence assume $L > 0$ for the concentration argument below. Hoefding’s inequality and (42) imply, for every fixed independent set $Z$

$$
\mathbb { P } \left( \left| \overline { { W } } _ { g } ( Z ) - W _ { g } ( Z ) \right| > \frac { \delta _ { s } } { 5 } L \bigg | \mathcal { H } \right) \leq 2 \exp \left( - \frac { 2 m \delta _ { s } ^ { 2 } } { 2 5 } \right) .
$$

The contracted augmented matroid has rank at most k and ground-set size $n + k \leq 2 n$ . Hence the number of independent sets is at most

$$
\sum _ { j = 0 } ^ { k } { \binom { n + k } { j } } \leq ( e ( n + k ) ) ^ { k } \leq ( 2 e n ) ^ { k } ,
$$

whose logarithm is $O ( k \log n )$ . Thus, for a suficiently large universal constant in (40), the event

$$
\mathcal { E } _ { \mathrm { s w } } : = \left\{ \operatorname* { s u p } _ { Z \in \mathbb { Z } ( M ^ { \prime } ) } \left. \overline { { W } } _ { g } ( Z ) - W _ { g } ( Z ) \right. \leq \frac { \delta _ { s } } { 5 } L \right\}
$$

satisfies

$$
\mathbb { P } ( \mathcal { E } _ { \mathrm { s w } } ^ { c } \mid \mathcal { H } ) \le \frac { \delta _ { s } } { 5 } .\tag{43}
$$

Now incorporate the controlled oracle. For every sampled set R and candidate element $i ,$

$$
| [ \widehat { g } ( R \cup \{ i \} ) - \widehat { g } ( R ) ] - [ g ( R \cup \{ i \} ) - g ( R ) ] | \leq 2 \xi .
$$

Consequently, for every independent set $Z$

$$
\left| \sum _ { i \in Z } [ \widehat { g } ( R \cup \{ i \} ) - \widehat { g } ( R ) ] - \sum _ { i \in Z } [ g ( R \cup \{ i \} ) - g ( R ) ] \right| \leq 2 k \xi .\tag{44}
$$

This is deterministic and holds simultaneously for all $Z .$

Let $\widehat { W } ( Z )$ denote the empirical base sum computed from the controlled oracle. On ${ \mathcal E } _ { \mathrm { s w } }$

$$
\operatorname* { s u p } _ { Z \in \mathcal { I } ( M ^ { \prime } ) } | \widehat { W } ( Z ) - W _ { g } ( Z ) | \leq \Delta , \qquad \Delta : = \frac { \delta _ { s } } { 5 } L + 2 k \xi .\tag{45}
$$

Let $\widehat { Z }$ be the data-dependent maximum-weight base selected by the algorithm. Since $O _ { g }$ is an optimal base, it is an admissible comparison base for the almost-above-average guarantee. Because (45) holds simultaneously for every independent set, it applies to the data-dependent set $\widehat { Z }$ and the fixed comparison base $O _ { g }$ , even though $\overrightharpoon { Z }$ depends on the same samples. Therefore, on ${ \mathcal E } _ { \mathrm { s w } }$

$$
\begin{array} { r l } & { \overline { { W } } _ { g } ( \widehat { Z } ) \geq \widehat { W } ( \widehat { Z } ) - 2 k \xi } \\ & { \qquad \quad \geq \widehat { W } ( O _ { g } ) - 2 k \xi } \\ & { \qquad \quad \geq \overline { { W } } _ { g } ( O _ { g } ) - 4 k \xi } \\ & { \qquad \quad \geq W _ { g } ( O _ { g } ) - \frac { \delta _ { s } } { 5 } L - 4 k \xi } \\ & { \qquad \quad \geq F ( t \mathbf { 1 } _ { A } \lor \mathbf { 1 } _ { O _ { g } } ) - F ( t \mathbf { 1 } _ { A } ) - \frac { \delta _ { s } } { 5 } L - 4 k \xi . } \end{array}
$$

Equivalently, using the looser bound $2 \Delta$

$$
\overline { { { W } } } _ { g } ( \widehat { Z } ) \geq D - 2 \Delta , \qquad D : = F ( t { \bf 1 } _ { A } \vee { \bf 1 } _ { O _ { g } } ) - F ( t { \bf 1 } _ { A } ) .\tag{46}
$$

Here we used the standard submodularity bound

$$
W _ { g } ( O _ { g } ) = \sum _ { i \in O _ { g } } \bigl ( F ( t \mathbf { 1 } _ { A } \lor \mathbf { 1 } _ { i } ) - F ( t \mathbf { 1 } _ { A } ) \bigr ) \geq F ( t \mathbf { 1 } _ { A } \lor \mathbf { 1 } _ { O _ { g } } ) - F ( t \mathbf { 1 } _ { A } ) .
$$

The key point is that, conditional on the realized samples, the entering element is uniform in ${ \widehat { Z } } .$ . Thus its conditional expected gain is the true base sum

$$
\frac { W _ { g } ( \widehat { Z } ) } { k ^ { \prime } } ,
$$

where $k ^ { \prime } \leq k$ is the contracted rank. The empirical quantity $\overline { { W } } _ { g } ( \widehat { Z } )$ is used only to establish a lower bound on the true quantity $W _ { g } ( \widehat { Z } )$

On ${ \mathcal E } _ { \mathrm { s w } }$ , the uniform concentration event and the controlled-oracle error give

$$
\begin{array} { r l } & { W _ { g } ( \widehat { Z } ) \geq \widehat { W } _ { g } ( \widehat { Z } ) - \frac { \delta _ { s } } { 5 } L } \\ & { \qquad \geq \widehat { W } ( \widehat { Z } ) - 2 k \xi - \frac { \delta _ { s } } { 5 } L } \\ & { \qquad \geq \widehat { W } ( O _ { g } ) - 2 k \xi - \frac { \delta _ { s } } { 5 } L } \\ & { \qquad \geq \overline { { W } } _ { g } ( O _ { g } ) - 4 k \xi - \frac { \delta _ { s } } { 5 } L } \\ & { \qquad \geq W _ { g } ( O _ { g } ) - \frac { 2 \delta _ { s } } { 5 } L - 4 k \xi } \\ & { \qquad \geq W _ { g } ( O _ { g } ) - \frac { 2 \delta _ { s } } { 5 } L - 4 k \xi } \\ & { \qquad \geq D - 2 \Delta . } \end{array}
$$

where

$$
D : = F ( t { \bf 1 } _ { A } \vee \mathbf { 1 } _ { O _ { g } } ) - F ( t { \bf 1 } _ { A } ) , \qquad \Delta : = \frac { \delta _ { s } } { 5 } L + 2 k \xi .
$$

Consequently,

$$
\mathbb { E } \big [ F ( t \mathbf { 1 } _ { A } \vee \mathbf { 1 } _ { J } ) - F ( t \mathbf { 1 } _ { A } ) \mid \mathcal { H } , \mathcal { E } _ { \mathrm { s w } } , R _ { 1 } , \dots , R _ { m } \big ] \geq \frac { D - 2 \Delta } { k ^ { \prime } } .
$$

On the failure event, the pointwise lower bound $X _ { \ell } ( Z ) \geq - g ( O _ { g } )$ implies

$$
W _ { g } ( Z ) \geq - g ( O _ { g } ) \qquad \forall Z \in \mathcal { T } ( M ^ { \prime } ) ,
$$

and hence

$$
\mathbb { E } [ F ( t \mathbf { 1 } _ { A } \vee \mathbf { 1 } _ { J } ) - F ( t \mathbf { 1 } _ { A } ) \mid \mathcal { H } , \mathcal { E } _ { \mathrm { s w } } ^ { c } , R _ { 1 } , \dots , R _ { m } ] \geq - \frac { g ( O _ { g } ) } { k ^ { \prime } } .
$$

Let

$$
p : = \mathbb { P } ( \mathcal { E } _ { \mathrm { s w } } ^ { c } \mid \mathcal { H } ) .
$$

Averaging over the fresh samples and using (43),

$$
\begin{array} { r l } & { \mathbb { E } [ F ( t { \mathbf { 1 } _ { A } } \vee { \mathbf { 1 } _ { J } } ) - F ( t { \mathbf { 1 } _ { A } } ) \mid \mathcal { H } ] } \\ & { \qquad \geq ( 1 - p ) \frac { D - 2 \Delta } { k ^ { \prime } } - p \frac { g ( O _ { g } ) } { k ^ { \prime } } } \\ & { \qquad = \frac { ( 1 - p ) ( D - 2 \Delta ) - p g ( O _ { g } ) } { k ^ { \prime } } } \\ & { \qquad \geq \frac { D - 2 \Delta - 2 p g ( O _ { g } ) } { k ^ { \prime } } . } \end{array}
$$

Indeed, the last inequality follows from

$$
- p D + 2 \Delta p - p g ( O _ { g } ) \ge - 2 p g ( O _ { g } ) ,
$$

which is equivalent to

$$
p \big ( g ( O _ { g } ) - D + 2 \Delta \big ) \geq 0 ,
$$

and follows from $D \leq g ( O _ { g } )$ and $\Delta \geq 0$

Therefore, using $\Delta = \delta _ { s } L / 5 + 2 k \xi , p \leq \delta _ { s } / 5 .$ , and $g ( O _ { g } ) \le L$ , we obtain

$$
\eta \leq 2 \Delta + \frac { 2 \delta _ { s } } { 5 } g ( O _ { g } ) \leq \frac { 4 } { 5 } \delta _ { s } L + 4 k \xi .\tag{47}
$$

In particular, this has the claimed form $\eta \leq C _ { 1 } \delta _ { s } L + C _ { 2 } k \xi$ for universal constants. This is precisely the η-almost-above-average condition.

No submodularity, monotonicity, or unbiasedness of $\widehat g$ is used. The key adaptive-selection point is the uniform event (45), which is established before the data-dependent base $\widehat { Z }$ is selected. □

Combining (37), $g ( O _ { g } ) \leq \mathrm { O P T }$ , and Lemma 5.9, we obtain

$$
\eta \leq C _ { 1 } \delta _ { s } ( 3 2 1 0 \mathrm { P T } + 1 9 2 2 k \xi ) + C _ { 2 } k \xi .\tag{48}
$$

Set

$$
\delta _ { s } = \frac { \varepsilon } { 1 0 ^ { 4 } } .\tag{49}
$$

Then

$$
\begin{array} { r } { \boxed { \eta \leq C _ { 3 } \varepsilon \mathrm { O P T } + C _ { 4 } k \xi . } } \end{array}\tag{50}
$$

Notice the important relative form of the first term. We do not replace it by $O ( \varepsilon ) { \mathrm { : } }$ doing so would lose the desired $( \alpha - \varepsilon ) \mathrm { O P T }$ resilience statement when OPT is small.

The expected number of Poisson events is

$$
O \left( k \log { \frac { 1 } { \varepsilon _ { 0 } } } \right) .\tag{51}
$$

Each swap requires

$$
O \left( \frac { n ( k \log n + \log ( 1 / \delta _ { s } ) ) } { \delta _ { s } ^ { 2 } } \right)\tag{52}
$$

controlled-oracle evaluations.

Lemma 5.10 (Controlled-oracle complexity). With $\rho = \Theta ( \varepsilon ) , \delta _ { s } = \Theta ( \varepsilon )$ , and $\varepsilon _ { 0 } = \Theta ( \varepsilon )$ , the expected number of controlled-oracle evaluations used by the unchanged SGS-Poisson value-oracle implementation is

$$
\begin{array} { l } { { \displaystyle N ( \varepsilon ) = O \left( n k \log \frac { 1 } { \varepsilon } + n k + \frac { n k ^ { 2 } \log n \log ( 1 / \varepsilon ) } { \varepsilon ^ { 2 } } + \frac { n k \log ^ { 2 } ( 1 / \varepsilon ) } { \varepsilon ^ { 2 } } \right) } } \\ { { \displaystyle ~ = \widetilde O ( n k ^ { 2 } \varepsilon ^ { - 2 } ) . } } \end{array}\tag{53}
$$

Here the expectation is over all internal randomization of the algorithm, including the RRG amplification, preprocessing, Poisson process, and swap sampling. Independence-oracle queries and maximum-weightbase computation are not counted.

Proof. Each RRG run has $\lceil k / 2 \rceil$ iterations, and each iteration requires $O ( n )$ marginal-value queries to form a maximum-weight base. The amplified certificate uses $R = O ( \log ( 1 / \rho ) ) = O ( \log ( 1 / \varepsilon ) )$ ) independent runs, hence $O ( n k \log ( 1 / \varepsilon ) )$ queries. Advanced Preprocessing performs at most k iterations, each requiring $O ( n )$ marginal queries, hence $O ( n k )$ additional queries.

For a swap, (40) with $\delta _ { s } = \Theta ( \varepsilon )$ gives

$$
m = O \left( \frac { k \log n + \log ( 1 / \varepsilon ) } { \varepsilon ^ { 2 } } \right) .
$$

There are n candidate entering elements, and each sampled marginal $g ( R e \cup \{ i \} ) - g ( R e )$ uses two controlled-oracle evaluations. Thus one Poisson event requires $O ( n m )$ controlled-oracle evaluations (the factor of two is absorbed in the $O ( \cdot )$ notation). The non-homogeneous Poisson process on $[ \varepsilon _ { 0 } , 1 ]$ has expected number of events k $\textstyle \int _ { \varepsilon _ { 0 } } ^ { 1 } t ^ { - 1 } { \dot { d t } } = k \log ( { \dot { 1 / \varepsilon _ { 0 } } } )$ . Multiplying these two quantities gives

$$
O \left( \frac { n k ^ { 2 } \log n \log ( 1 / \varepsilon ) } { \varepsilon ^ { 2 } } + \frac { n k \log ^ { 2 } ( 1 / \varepsilon ) } { \varepsilon ^ { 2 } } \right) .
$$

Adding the RRG and preprocessing costs proves the claim.

## 5.5 Combining the pieces: the resilience theorem

We now combine the preprocessing and swap results with the unchanged SGS-Poisson dynamics.

Theorem 5.11 (Resilience of SGS-Poisson). There exists a universal constant C such that for every $\varepsilon \in ( 0 , 1 / 2 ]$ the following holds.

$I f k = 0$ , return ∅. $I f k = 1$ , query $\widehat { f }$ on every feasible singleton and on $\emptyset$ and return the best queried feasible set; this gives

$$
\mathbb { E } [ f ( A _ { \mathrm { o u t } } ) ] \geq \mathrm { O P T } - 2 \xi ,
$$

so the claimed theorem follows. Hence assume $k \geq 2$ below.

Let $f : 2 ^ { U }  [ 0 , 1 ]$ be nonnegative submodular and let $M = ( U , { \mathcal { T } } )$ be a matroid of rank k. Given any persistent $\xi \cdot$ -controlled oracle, the SGS-Poisson value-oracle implementation outputs $A _ { \mathrm { o u t } } \in \mathcal { T }$ satisfying

$$
\begin{array} { r } { \boxed { \mathbb { E } [ f ( A _ { \mathrm { o u t } } ) ] \geq \left\{ \begin{array} { l l } { ( 1 / e - \varepsilon ) \mathrm { O P T } - C k \xi , } & { f \ n o n - m o n o t o n e , } \\ { ( 1 - 1 / e - \varepsilon ) \mathrm { O P T } - C k \xi , } & { f \ m o n o t o n e . } \end{array} \right. } } \end{array}\tag{54}
$$

Moreover,

$$
\boxed { N ( \varepsilon ) = O \left( n k \log \frac { 1 } { \varepsilon } + \frac { n k ^ { 2 } \log n \log ( 1 / \varepsilon ) } { \varepsilon ^ { 2 } } + \frac { n k \log ^ { 2 } ( 1 / \varepsilon ) } { \varepsilon ^ { 2 } } \right) . }\tag{55}
$$

Proof. Choose the RRG amplification failure probability $\rho = \varepsilon / 1 0 0$ . By the certificate construction,

$$
\mathbb { P } ( \widehat { V } \geq \mathrm { O P T } ) \geq 1 - \rho .
$$

Let

$$
\mathcal { E } : = \left\{ \mathrm { O P T } \leq \widehat { V } \leq 1 6 \mathrm { O P T } + 9 6 k \xi \right\} .
$$

By $( 2 6 ) , \mathbb { P } ( \mathcal { E } ) \geq 1 - \rho$ . We take the initial sigma-field to include the RRG repetitions and their output, so $\mathcal { E }$ is measurable at time zero. Conditional on this initial sigma-field, all subsequent preprocessing and SGS-Poisson randomness is independent of the RRG repetitions. The preprocessing certificate is used in conditional expectation, while the residual-mass and swap bounds hold uniformly for every realized preprocessing trajectory on $\mathcal { E } ;$ hence the subsequent conditioning on $\mathcal { E }$ is legitimate.

On $\mathcal { E } ,$ Theorem 5.7 gives

$$
{ \mathbb E } \left[ g ( O _ { g } ) + \frac { 1 } { 2 } g ( \emptyset ) \mid \mathcal { E } \right] \ge \mathrm { O P T } - C _ { 0 } k \xi .
$$

Also, Lemma 5.5 and Lemma 5.9, with

$$
\delta _ { s } = \frac { \varepsilon } { 1 0 ^ { 4 } } ,
$$

give, uniformly over every realized preprocessing trajectory in $\mathcal { E } .$

$$
\eta \leq C _ { 3 } \varepsilon \mathrm { O P T } + C _ { 4 } k \xi .
$$

If the contracted rank is $k ^ { \prime } = 0$ , then $O _ { g } = \emptyset$ and the post-preprocessing algorithm returns $\bar { S } .$ . The certificate gives

$$
\mathbb { E } \left[ g ( O _ { g } ) + \frac { 1 } { 2 } g ( \emptyset ) \mid \xi \right] = \frac { 3 } { 2 } \mathbb { E } [ g ( \emptyset ) \mid \xi ] \geq \mathrm { O P T } - C _ { 0 } k \xi ,
$$

and hence

$$
\mathbb { E } [ f ( A _ { \mathrm { o u t } } ) \mid \mathcal { E } ] \geq \frac { 2 } { 3 } \mathrm { O P T } - \frac { 2 } { 3 } C _ { 0 } k \xi .
$$

Since $2 / 3 > 1 - 1 / e > 1 / e$ , enlarging the universal constant yields both claimed approximation bounds.   
Thus we may assume $k ^ { \prime } \ge 1$ when applying Proposition 4.2 below.

Run SGS-Poisson on the contracted instance with starting time $\varepsilon _ { 0 } = \varepsilon / 1 0 0$ . Conditioned on $\mathcal { E }$ and on the realized S<sup>¯</sup>, Proposition 4.2 gives, in the non-monotone case,

$$
\mathbb { E } [ g ( S _ { \mathrm { o u t } } ) \mid \bar { S } , \mathcal { E } ] \ge ( 1 - \varepsilon _ { 0 } ) \frac { 1 } { e } g ( O _ { g } ) + \frac { 1 } { e } g ( \mathcal { O } ) - \eta .
$$

Since $g ( \emptyset ) \geq 0$ and $1 / e \geq ( 1 - \varepsilon _ { 0 } ) / ( 2 e )$ , taking expectation over the preprocessing randomness $\bar { S }$ conditioned on $\mathcal { E }$ and using linearity of expectation gives

$$
\begin{array} { r l } & { \mathbb { E } \displaystyle [ g ( S _ { \mathrm { o u t } } ) \mid \mathcal { E } ] \geq \frac { 1 - \varepsilon _ { 0 } } { e } \mathbb { E } \left[ g ( O _ { g } ) + \frac { 1 } { 2 } g ( \emptyset ) \mid \mathcal { E } \right] } \\ & { \qquad - C _ { 2 } \varepsilon \mathrm { O P T } - C _ { 3 } k \xi . } \end{array}
$$

Using the certificate,

$$
\mathbb { E } [ f ( A _ { \mathrm { o u t } } ) \mid \mathcal { E } ] \geq \left( { \frac { 1 } { e } } - C _ { 4 } \varepsilon \right) { \mathrm { O P T } } - C _ { 5 } k \xi .
$$

For monotone $f ,$ the contraction $g$ is also monotone, and Proposition 4.2 gives

$$
\mathbb { E } [ g ( S _ { \mathrm { o u t } } ) \mid \bar { S } , \mathcal { E } ] \ge ( 1 - \varepsilon _ { 0 } ) ( 1 - 1 / e ) g ( O _ { g } ) + \frac { 1 } { e } g ( \mathcal { O } ) - \eta .
$$

Because

$$
\frac { 1 } { e } \geq \frac { 1 } { 2 } ( 1 - 1 / e ) ,
$$

taking expectation over the preprocessing randomness $\bar { S }$ conditioned on $\mathcal { E }$ and using linearity of expectation gives

$$
\begin{array} { c l l } { { \mathbb { E } [ g ( S _ { \mathrm { o u t } } ) \mid { \mathcal { E } } ] \geq ( 1 - \varepsilon _ { 0 } ) ( 1 - 1 / e ) \mathbb { E } \left[ g ( O _ { g } ) + { \frac { 1 } { 2 } } g ( { \mathcal { O } } ) \mid { \mathcal { E } } \right] } } \\ { { - C _ { 2 } \varepsilon \mathrm { O P T } - C _ { 3 } k \xi . } } \end{array}
$$

Therefore

$$
\mathbb { E } [ f ( A _ { \mathrm { o u t } } ) \mid \mathcal { E } ] \geq \left( 1 - \frac { 1 } { e } - C _ { 6 } \varepsilon \right) \mathrm { O P T } - C _ { 7 } k \xi .
$$

It remains to remove the conditioning. Since $f ( A _ { \mathrm { o u t } } ) \geq 0$

$$
\begin{array} { r } { \mathbb { E } [ f ( A _ { \mathrm { o u t } } ) ] \geq \mathbb { P } ( \mathcal { E } ) \mathbb { E } [ f ( A _ { \mathrm { o u t } } ) \mid \mathcal { E } ] \geq ( 1 - \rho ) \mathbb { E } [ f ( A _ { \mathrm { o u t } } ) \mid \mathcal { E } ] . } \end{array}
$$

For either objective class, the conditional lower bound has the form $c \mathrm { O P T } - C ^ { \prime } \varepsilon \mathrm { O P T } - C ^ { \prime \prime } k \xi$ with $c \in \{ 1 / e , 1 - 1 / e \}$ . Multiplication by $1 - \rho$ therefore changes the leading coeficient by at most $\rho$ because $\mathrm { O P T } \leq 1$ . Taking $\rho = \varepsilon / 1 0 0$ and enlarging the universal constants gives

$$
\mathbb { E } [ f ( A _ { \mathrm { o u t } } ) ] \geq \left\{ \begin{array} { l l } { ( 1 / e - \varepsilon ) \mathrm { O P T } - C k \xi , } & { f \mathrm { ~ n o n - m o n o t o n e } , } \\ { ( 1 - 1 / e - \varepsilon ) \mathrm { O P T } - C k \xi , } & { f \mathrm { ~ m o n o t o n e } . } \end{array} \right.
$$

The oracle complexity is given by Lemma 5.10.

Corollary 5.12 (Resilience parameters). In the resilience parameterization of Definition $3 . 1 , \ S G S -$ Poisson is $( 1 / e , \ 2 , \ 2 , \ \widetilde { O } ( n k ^ { 2 } ) , \ O ( k ) ) \ f o r$ non-monotone submodular maximization and $( 1 - 1 / e , \ 2 , \ 2 , \ \widetilde O ( n k ^ { 2 } ) , \ O ( k ) )$ for monotone submodular maximization.

## 6 Ofline-to-Online CMAB

We now use the resilience theorem only as an ofline primitive. The controlled oracle is an analysis interface for this ofline subroutine; it is not assumed to be directly available to the online learner. In the CMAB application, the ofline value estimates are generated by the exploration mechanism in the ofline-to-online reduction. The online conversion is the single-agent specialization of the framework of [8]; we use their theorem in its general $( \alpha , \beta , \gamma , \psi , \delta )$ form. This avoids introducing a separate fixed-accuracy $T ^ { 2 / 3 }$ theorem and lets the query-complexity exponent $\beta$ directly determine the final horizon exponent.

Theorem 6.1 (ofline-to-online reduction, [8]). Suppose $\delta \ > \ 0$ and an ofline algorithm $\mathcal { A } ( \varepsilon )$ is $( \alpha , \beta , \gamma , \psi , \delta )$ -resilient in the sense of Definition 3.1, with oracle complexity $N ( \varepsilon ) = \widetilde O \big ( \psi \varepsilon ^ { - \beta } \log ^ { \gamma } ( 1 / \varepsilon ) \big )$ Then, for stochastic single-agent CMAB with full-bandit feedback, whenever T satisfies the following horizon condition, $\textstyle T \geq \operatorname* { m a x } \left\{ \psi , { \frac { 2 \psi } { \delta ^ { \beta + 1 } } } \right\}$ , the reduction in $[ 8 ]$ using A as its ofline subroutine achieves

$$
R _ { \alpha } ( T ) = \widetilde { \cal O } \left( \delta ^ { \frac { 2 } { 3 + \beta } } \psi ^ { \frac { 1 } { 3 + \beta } } T ^ { \frac { 2 + \beta } { 3 + \beta } } \right) ,
$$

where the logarithmic dependence on T and the exponent $\gamma$ are absorbed into ${ \widetilde { O } } ( \cdot )$ . The theorem eliminates the ofline ε-approximation loss and therefore gives α-regret, rather than $( \alpha - \varepsilon ) \ – r e g r e t .$

Thus, we obtain the following.

Corollary 6.2 (Full-bandit CMAB under general matroids). For stochastic full-bandit CMAB with a general rank-k matroid and a nonnegative submodular mean reward, there exist algorithms satisfying, for all horizons $\begin{array} { r } { T \ge \operatorname* { m a x } \left\{ \psi , \frac { 2 \psi } { \delta ^ { 3 } } \right\} } \end{array}$ ，

$$
R _ { 1 / e } ( T ) = \widetilde { O } \Big ( n ^ { 1 / 5 } k ^ { 4 / 5 } T ^ { 4 / 5 } \Big )
$$

for non-monotone rewards and

$$
R _ { 1 - 1 / e } ( T ) = \widetilde { O } \Big ( n ^ { 1 / 5 } k ^ { 4 / 5 } T ^ { 4 / 5 } \Big )
$$

for monotone rewards, where one may take

$$
\psi = \widetilde O ( n k ^ { 2 } ) , \qquad \delta = O ( k ) .
$$

In particular, since a resilience guarantee with parameter $\delta _ { 0 }$ also holds with any $\delta \geq \delta _ { 0 }$ , we may take the universal constant in $\delta = { \cal { O } } ( k )$ large enough that $\delta \geq 1$ . The admissibility condition is then satisfied whenever

$$
T \geq 2 \psi , \qquad \psi = \widetilde O ( n k ^ { 2 } ) .
$$

Thus $T \geq { \widetilde { O } } ( n k ^ { 2 } )$ is a suficient order-of-growth condition, up to the universal constant in the horizon requirement. For smaller horizons, the same order bound can be made formally valid after enlarging its universal constant, since $R _ { \alpha } ( T ) \leq T$ under the present normalization.

Proof. Theorem 5.11 and Corollary 5.12 give

$$
\beta = 2 , \qquad \gamma = 2 , \qquad \psi = \widetilde O ( n k ^ { 2 } ) , \qquad \delta = O ( k ) ,
$$

with $\alpha = 1 / e$ in the non-monotone case and $\alpha = 1 - 1 / e$ in the monotone case. Theorem 6.1 therefore requires

$$
T \geq \operatorname* { m a x } \left\{ \psi , \frac { 2 \psi } { \delta ^ { 3 } } \right\} .
$$

After enlarging δ by a universal constant if necessary, we may assume $\delta \geq 1$ , so a suficient condition is $T \geq 2 \psi = \widetilde { \cal O } ( n k ^ { 2 } )$ . Applying Theorem 6.1 with $\beta = 2$ then yields

$$
\begin{array} { r c l } { { } } & { { } } & { { R _ { \alpha } ( T ) = \widetilde O \Bigl ( ( k ) ^ { 2 / 5 } ( n k ^ { 2 } ) ^ { 1 / 5 } T ^ { 4 / 5 } \Bigr ) } } \\ { { } } & { { } } & { { = \widetilde O \Bigl ( n ^ { 1 / 5 } k ^ { 4 / 5 } T ^ { 4 / 5 } \Bigr ) . } } \end{array}
$$

This proves both claims.

## 7 Discussion of Technical Novelty

We separate the imported ingredients from the new technical statements.

Imported ingredients. The proof uses three external algorithmic components: (i) the exact SGS-Poisson process and its Poisson diferential analysis [12]; (ii) the standard Residual Random Greedy exchange lemma [3]; and (iii) the single-agent ofline-to-online CMAB conversion of [8]. None of these ingredients is claimed as new here.

New technical statements. The paper proves:

(i) robust RRG under an arbitrary adversarially controlled oracle;

(ii) a realized constant-factor optimum certificate that can drive an adaptive controlled-oracle stopping rule;

(iii) the resilient adaptive-preprocessing drift inequality of Lemma 5.6,

$$
\mathbb { E } [ M _ { t + 1 } - M _ { t } \mid { \mathcal { F } } _ { t } ] \geq { \frac { 8 { \mathrm { O P T } } - k \xi } { k - t } } ;
$$

(iv) the robust almost-above-average swap guarantee of Lemma 5.9 for the exact SGS-Poisson implementation, with error $O ( \varepsilon \mathrm { O P T } + k \xi )$ ;

(v) the resulting adversarial resilience parameters

$$
( 1 / e , 2 , 2 , \widetilde O ( n k ^ { 2 } ) , O ( k ) ) \quad \mathrm { a n d } \quad ( 1 - 1 / e , 2 , 2 , \widetilde O ( n k ^ { 2 } ) , O ( k ) ) ;
$$

Here, we detail technical novelty in two of the steps.

Why the adaptive-preprocessing lemma (Lemma 5.6) is not routine perturbation. A pointwise perturbation argument would attempt to compare the noisy execution with the exact execution. This is impossible in general: an arbitrarily small change in marginal weights can switch the maximumweight base, which can switch the exchange map, which can switch the stopping time and every subsequent state of the process. Moreover, $\widehat { f }$ need not be submodular. Our proof instead evaluates the true objective along the controlled-oracle trajectory and constructs the exchange potential

$$
M _ { t } = f ( Q _ { t } \cup O _ { t } ) + { \frac { 1 } { 2 } } f ( Q _ { t } ) .
$$

The key drift inequality uses the residual rank $k - t \colon$

$$
\mathbb { E } [ \Delta M _ { t } \mid \mathcal { F } _ { t } ] \geq \frac { 8 \mathrm { O P T } - k \xi } { k - t } .
$$

Thus the proof never needs the controlled-oracle trajectory to remain close to the exact trajectory. This is the central structural reason that the oracle error is absorbed as $O ( k \xi )$ rather than multiplied by the number of adaptive decisions.

Why the swap lemma (Lemma 5.9) is also nontrivial. The SGS-Poisson implementation does not merely need a good estimate of one marginal. It needs a base whose sum of multilinear marginals satisfies the almost-above-average condition while the resulting exchange map satisfies the validity and right-continuity conditions. We therefore concentrate the base-sum random variable directly. The controlled oracle contributes only $O ( k \xi )$ to a base comparison, while the sampling term is proportional to the residual marginal scale. This yields

$$
\eta \leq C _ { 1 } \varepsilon \mathrm { O P T } + C _ { 2 } k \xi .
$$

The relative form $\varepsilon \mathrm { O P T }$ is essential: replacing it by an absolute $O ( \varepsilon )$ term would not imply the desired $( \alpha - \varepsilon ) \mathrm { O P T } - O ( k \xi )$ resilience statement.

## 8 Conclusion

We established adversarial resilience of the SGS-Poisson paradigm for nonnegative submodular maximization over general matroids. The Poisson process, single-element exchange mechanism, and spiteful drop rule are unchanged. The key technical result is a resilient adaptive-preprocessing theorem: despite controlled-oracle maximum-base decisions and a noisy stopping time, a matroid-exchange potential retains positive drift. This produces the residual marginal certificate required by SGS-Poisson, after which robust almost-above-average swaps preserve its Poisson diferential inequality.

The resulting ofline algorithms have limiting approximation factors $1 / e$ and $1 - 1 / e ,$ additive sensitivity $O ( k \xi )$ , and $\widetilde { O } ( n k ^ { 2 } \varepsilon ^ { - 2 } )$ controlled-oracle complexity. Through the single-agent ofline-to-online CMAB reduction, our resilience parameters yield exact limiting approximation-regret factors $1 / e$ and $1 - 1 / e$ with $\widetilde { O } ( n ^ { 1 / 5 } k ^ { 4 / 5 } T ^ { 4 / 5 } )$ regret.

## References

[1] Kshipra Bhawalkar, Yang Cai, Zhe Feng, Christopher Liaw, and Tao Lin. A unified approach to submodular maximization under noise. volume 38, pages 166355–166383, 2025.

[2] Niv Buchbinder and Moran Feldman. Deterministic algorithm and faster algorithm for submodular maximization subject to a matroid constraint. SIAM Journal on Computing, pages FOCS24–85, 2025.

[3] Niv Buchbinder, Moran Feldman, Joseph Naor, and Roy Schwartz. Submodular maximization with cardinality constraints. In Proceedings of the twenty-fifth annual ACM-SIAM symposium on Discrete algorithms, pages 1433–1452, 2014.

[4] Gruia Calinescu, Chandra Chekuri, Martin Pal, and Jan Vondr´ak. Maximizing a monotone submodular function subject to a matroid constraint. SIAM Journal on Computing, 40(6):1740– 1766, 2011.

[5] Moran Feldman, Joseph Naor, and Roy Schwartz. A unified continuous greedy algorithm for submodular maximization. In 2011 IEEE 52nd annual symposium on foundations of computer science, pages 570–579, 2011.

[6] Yuval Filmus and Justin Ward. Monotone submodular maximization over a matroid via non-oblivious local search. SIAM Journal on Computing, 43(2):514–542, 2014.

[7] Fares Fourati, Vaneet Aggarwal, Christopher Quinn, and Mohamed-Slim Alouini. Randomized greedy learning for non-monotone stochastic submodular maximization under full-bandit feedback. In International Conference on Artificial Intelligence and Statistics, pages 7455–7471, 2023.

[8] Fares Fourati, Mohamed-Slim Alouini, and Vaneet Aggarwal. Federated combinatorial multi-agent multi-armed bandits. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 13760–13782, 2024.

[9] Fares Fourati, Christopher John Quinn, Mohamed-Slim Alouini, and Vaneet Aggarwal. Combinatorial stochastic-greedy bandit. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 12052–12060, 2024.

[10] Amit Ganz Rozenman, Ariel Kulik, Roy Schwartz, and Mohit Singh. A poisson process for submodular maximization. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, pages 2152–2163, 2026.

[11] Andreas Krause and Daniel Golovin. Submodular function maximization. Tractability, 3(71-104):3, 2014.

[12] Ariel Kulik, Thiago Oliveira, Roy Schwartz, and Mohit Singh. If it is good then drop it–a spiteful poisson process for submodular maximization. arXiv preprint arXiv:2608.05062, 2026.

[13] Guanyu Nie, Mridul Agarwal, Abhishek Kumar Umrawal, Vaneet Aggarwal, and Christopher John Quinn. An explore-then-commit algorithm for submodular maximization under full-bandit feedback. In Uncertainty in Artificial Intelligence, pages 1541–1551, 2022.

[14] Guanyu Nie, Vaneet Aggarwal, and Christopher John Quinn. Stochastic k-submodular bandits with full bandit feedback. In Proceedings of AAMAS, May 2025.

[15] Guanyu Nie, Yididiya Y Nadew, Yanhui Zhu, Vaneet Aggarwal, and Christopher John Quinn. A framework for adapting ofline algorithms to solve combinatorial multi-armed bandit problems with bandit feedback. In International Conference on Machine Learning, pages 26166–26198, 2023.
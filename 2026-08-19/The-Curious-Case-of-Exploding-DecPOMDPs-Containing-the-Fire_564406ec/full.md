# The Curious Case of Exploding DecPOMDPs: Containing the Fire through Policy Counting<sup>⋆</sup>

Nazlı Nur Karabulut<sup>[0000−0001−7958−5627]</sup> and Tanya Braun<sup>[0000−0003−0282−4284]</sup>

Computer Science Department, University of Münster, Münster, Germany {nnur.karabulut,tanya.braun}@uni-muenster.de

Abstract. Decentralised partially observable Markov decision processes (DecPOMDPs) provide a general framework for modelling multi-agent decision making under uncertainty. However, DecPOMDPs are known to sufer from exponential complexity in the number of agents. One way to combat this intractability in agent numbers is to look at partitions of agents that exhibit a form of symmetry among agents, allowing for a compact encoding by counting. However, a challenge arises as the policy space explodes, even though the model complexity and evaluation cost reduce to a polynomial dependence. In this paper, we redirect our focus from counting agents to counting policies, which actually enables tractability in agent numbers for so called policy-counted DecPOMDPs. Further, we present policy-counted dynamic programming using the compact representation to solve policy-counted DecPOMDPs eficiently.

## 1 Introduction

Large-scale agent systems involve a huge number of interacting agents that have to coordinate decisions in uncertain environments. To formalise the setting, decentralised partially observable Markov decision processes (DecPOMDPs) provide a framework to compute a joint policy for a set of cooperative agents that maximises some joint utility measure in an environment that is considered to be stationary and describable by a probabilistic transition model. When agents are self-interested, the problem is formalised as a partially observable stochastic game (POSG), in which each agent has its own utility function.

However, solving multi-agent decision making problems is notoriously dificult as the number of potential policies escalates exponentially with the number of agents. To deal with this problem, researchers have proposed various approaches: Early work focuses on dynamic programming using pruning techniques for POSGs [4]. Building on this basis, Szer et al. present point-based dynamic programming for DecPOMDPs [21] and Seuken et al. propose memorybounded dynamic programming [20]. The dynamic programming operator was later extended with approximate pruning techniques to improve scalability for POSGs [12]. In addition to dynamic programming–based approaches, Szer et al.

[22] propose multiagent $\mathrm { A ^ { * } \ ( M A A ^ { * } ) }$ , a policy search algorithm that groups action–observation histories of agents at the same stage when they share the same optimal Q-values in a Bayesian game formulation of the problem [17]. Newer work optimises $\mathrm { M A A ^ { * } }$ even further [10,11], working on the doubly-exponential dependence on the horizon among other things. Additional research has examined more structured subclasses to improve scalability, e.g., for POSGs, examples include zero-sum formulations [26,5,24], one-sided POSGs where only one agent is afected by uncertainty [7,6,2], and common-payof POSGs that restrict agents to share identical rewards [3].

Another lane of work focuses on so-called lifting [18]. Lifting exploits symmetries between objects, also sometimes referred to as exchangeability, which can be used for tractable inference [15]. In multi-agent systems, lifting is applied to agents, allowing for a more compact and tractable problem formulation using partitions of agents, with the number of partitions being much smaller than the number of agents. The basic assumption, first described by Braun et al. [1], is that, within a partition, permutations of joint actions and observations have the same efect on the joint transition or reward, allowing for counting actions and observations in histograms. Adding in an assumption about conditional independence among agents of a partition even allows for working with a single representative agent per partition without counting necessary, leading to a drastic reduction in complexity for so-called isomorphic DecPOMDPs. However, without that assumption, a reduction can only be made in model complexity and evaluation cost, while —almost counter-intuitively— the policy space explodes [1]. More recently, we have presented isomorphic POSGs and lifted dynamic programming to work with the compact encoding, yielding a runtime polynomial in the number of agents [8]. We have extended this work to counting policies to circumvent an explosion in the policy space for partitioned POSGs [9].

Invigorated by the progress made in counting POSGs, this work takes on the case of exploding DecPOMDPs under counting, a challenge left open by Braun et al. [1] in addition to a dedicated solution method, by extending the counting focus to policies. Specifically, the contributions are threefold: (i) a new definition of counting DecPOMDPs that allows for counting policies, preventing the explosion in the policy space, (ii) an updated utility calculation including an analysis that shows tractability in agent numbers, and (iii) a dynamic programming operator to solve policy-counted DecPOMDPs. As such, we solve the above-mentioned challenge and additionally provide a dedicated solution method. When showing tractability, we assume a fixed number of partitions that is much smaller than the number of agents.

The paper is organised as follows: We start with DecPOMDPs and dynamic programming. Then we analyse the problem of exploding DecPOMDPs and show how to use counting for policies to circumvent the problem, followed by counting dynamic programming. We end with a conclusion. Due to space constraints, the proofs are relegated to the appendix.

## 2 Preliminaries

In this section, we define DecPOMDPs and provide an overview of dynamic programming for DecPOMDPs.

## 2.1 Decentralised POMDPs

The definition is based on [16] and [19], using random variables S with ran(S) referring to the set of (range) values it can take. Bold symbols denote sets, (time) steps t are denoted by superscript t, sequences over a discrete interval $[ t _ { s } : t _ { e } ]$ are denoted by superscript $t _ { s } : t _ { e } ,$ , and omitting an element i from a set or sequence is denoted by subscript −i. To add an element to a sequence, we use ◦.

Definition 1. A ( ground) DecPOMDP M is a tuple $( I , S , A , T , R , O , \varOmega , \tau )$

I a set of N agents,

– S a random variable with a set of states as a range,

$A = \{ A _ { i } \} _ { i \in I }$ a set of decision random variables, each $A _ { i }$ with a set of local actions as range,

$- \ T ( S ^ { \prime } , S , { \cal A } ) = P ( S ^ { \prime } \mid S , { \cal A } )$ a transition function,

$\mathbf { \Sigma } _ { - } \ R ( S , A )$ a reward function,

$O = \{ O _ { i } \} _ { i \in I }$ a set of random variables, each $O _ { i }$ with a set of local observations as range,

$\varOmega ( O , S ) = P ( O \mid S )$ a sensor function<sup>1</sup>, and

$\mathit { \Pi } - \ \tau \ \cdot \ \boldsymbol { a }$ horizon.

Each agent i follows a local policy $\pi _ { i } ^ { t }$ mapping local observation histories $o ^ { 0 : t }$ to actions $^ { a , }$ which can be represented as a depth-t policy tree (see Fig. 1). A joint policy is a tuple ${ \pmb \pi } = ( \pi _ { 1 } , . . . , \pi _ { n } )$ . The semantics of the ground model M is given by the joint policy space $\varPi _ { M }$ . The DecPOMDP problem asks for the joint policy that yields the maximum expected utility (with horizon τ):

$$
M E U ( M ) = \underset { \pmb { \pi } \in \cal { \pi } _ { M } } { \arg \operatorname* { m a x } } U _ { M } ( \pmb { \pi } ) , \underset { \pmb { \pi } \in \cal { \pi } _ { M } } { \operatorname* { m a x } } U _ { M } ( \pmb { \pi } ) )\tag{1}
$$

where $U _ { M } ( \pmb { \pi } )$ is calculated recursively over τ steps, by summing over joint observations and next states, and then weighted according to the prior $T ( s ^ { 0 } , \dots , \cdot , \cdot )$ i.e., $\begin{array} { r } { U _ { M } ( \pmb { \pi } ) = \sum _ { s ^ { 0 } \in \operatorname { r a n } ( S ) } T ( s ^ { 0 } , . ~ , . ~ ) U _ { M } ^ { \pi } ( s ^ { 0 } , \bot ^ { 0 : 0 } ) } \end{array}$ with

$$
U _ { M } ^ { \pi } ( s ^ { t } , o ^ { 0 : t } ) = R ( s _ { t } , \pi ( o ^ { 0 : t } ) ) + \sum _ { s ^ { t + 1 } \in \mathrm { r a n } ( S ) \atop \sum _ { o ^ { t + 1 } \in \mathrm { r a n } ( O ) } } T ( s ^ { t + 1 } , s ^ { t } , \pi ( o ^ { 0 : t } ) )\tag{2}
$$

![](images/4a1e644c61a9f40cac1a9135f440c47d79b860f42b06514e097362c6cf9e0c49.jpg)  
Fig. 1. Policies at level t = 1 available to agents with actions a, b and observations $x , y$

ending with $U _ { M } ^ { \pi } ( s , \pmb { o } ^ { 0 : \tau - 1 } ) = R ( s , \pmb { \pi } ( o ^ { 0 : \tau - 1 } ) )$ , where $\pi ( o ^ { 0 : t } )$ denotes the joint action at step $t \left( \pi ( \perp \right)$ : root actions) and $\pmb { o } ^ { 0 : t + 1 } = \pmb { o } ^ { 0 : t } \circ \pmb { o } ^ { t + 1 }$ the updated history.

In a DecPOMDP, all agents share a single joint reward function $R ( S , A )$ and cooperate to maximize the total expected utility. However, each agent decides what to do next on its own, not being able to observe the full state or the other agents’ observations. Nonetheless, agents are not fully independent since they depend on a joint state and receive the reward jointly.

## 2.2 Dynamic Programming

Dynamic programming is a solution method for multi-agent decision making that is shared between DecPOMDPs and POSGs [16]. It builds increasingly deeper policy trees from sub-trees for individual agents, while eliminating weakly-dominated policies at each depth [21]. Algorithm 1 shows the dynamic programming operator which consists of three steps. First, an exhaustive backup is performed to build all possible policies $\Pi _ { i } ^ { t } = \mathsf { \bar { \{ \pi } }  \pi _ { i , j } ^ { t } \} _ { j = 1 } ^ { m _ { t } }$ of depth t from the input policies $\Pi _ { i } ^ { t - 1 } ~ = ~ \{ \pi _ { i , j } ^ { t - 1 } \} _ { j = 1 } ^ { m _ { t - 1 } }$ for each agent i. Then, for each policy $\pi _ { i , j } ^ { t } ~ \in ~ \boldsymbol { I I } _ { i } ^ { t }$ , the operator calculates the corresponding value vector $V _ { i , j } ^ { t }$ , representing the value of that policy for each possible combination of state $s ^ { \prime } \in \operatorname { r a n } ( S )$ and policies of the other agents $\pmb { \pi } _ { - i } ^ { t } \in \pmb { \pi } _ { - i } ^ { t } ,$ which is defined as follows with $\pi = \pi _ { i , j } ^ { t } \circ \pi _ { - i }$

$$
V _ { i } ( s , \pi ) = R ( s , \pi ( \bot ) ) + \sum _ { s ^ { \prime } \in \mathrm { r a n } ( S ) } T ( s ^ { \prime } , s , \pi ( \bot ) ) \sum _ { o \in \mathrm { r a n } ( O ) } \varOmega ( o , s ) V _ { i } ( s ^ { \prime } , \pi . o )\tag{3}
$$

where π.o denotes the sub-policies after following an observation. During pruning, the operator eliminates policies that are weakly-dominated by others (solvable by a linear programme, see App. A.1), meaning that over the complete space of $\mathrm { r a n } ( S ) \times \pi _ { - i } ^ { t }$ there is always another policy with at least as high a value, i.e.,

$$
\forall s \in \mathrm { r a n } ( S ) , \pi _ { - i } ^ { t } \in \pmb { H } _ { - i } ^ { t } \exists \pi _ { i , j ^ { \prime } } ^ { t } \in \varPi _ { i } ^ { t } : V _ { i } ( s , \pi _ { i , j } ^ { t } \circ \pi _ { - i } ^ { t } ) \leq V _ { i } ( s , \pi _ { i , j ^ { \prime } } ^ { t } \circ \pi _ { - i } ^ { t } )\tag{4}
$$

## 3 A Case of Exploding DecPOMDPs

The basic idea of lifting DecPOMDPs is that the agent set is partitioned into sets of agents that behave indistinguishably among each other, meaning that it does not matter which agent performs which action within a partition, only how many do so [18,13,1]. In such a case, it is enough to count the agents and store the counts in a histogram. To illustrate the point, consider $N = 1 2$ indistinguishable agents with the actions $a , b$ available as an example. Imagine that 8 agents perform a while the remaining 4 perform b. Then, there exist ${ \binom { 1 2 } { 8 } } = { \frac { 1 \breve { 2 } ! } { 8 ! \cdot 4 ! } } = \bar { 4 } 9 5$ ways to make the 12 agents perform the actions, all of which result in the same outcome $\rho .$ Therefore, we can construct a histogram [4, 8] that maps to $\rho ,$ replacing 495 mappings. While this leads to a drastic decrease in model complexity, the policy space explodes in the formalisation chosen in [1]. To understand why, we next discuss the assumptions and design choices, left at times implicit in the original paper. Then, we look at the (in)tractability results.

Algorithm 1 Multi-agent Dynamic Programming Operator   
function MA-DP(set of policies $\overline { { I I _ { i } ^ { t - 1 } } }$ for each agent $i \in I$ with value vectors $V _ { i } ^ { t - 1 } )$   
for each agent $i \in I$ do   
$\boldsymbol { \varPi } _ { i } ^ { t }$ ← Perform exhaustive backup using $\boldsymbol { \Pi } _ { i } ^ { t - 1 }$   
$\mathbf { \nabla } _ { V _ { i } } ^ { t }$ ← Calculate new value vectors   
while $\exists \pi _ { i , j } ^ { t } \in I I _ { i } ^ { t } : \mathrm { E q . ~ ( 4 ) }$ holds for some $i \in I$ do   
$\varPi _ { i } ^ { t }  \ddot { \varPi } _ { i } ^ { t } \setminus \{ \pi _ { i , j } ^ { t } \} , V _ { i } ^ { t }  V _ { i } ^ { t } \setminus \{ v _ { i , j } ^ { t } \}$   
return $\{ ( \boldsymbol { I } \boldsymbol { I } _ { i } ^ { t } , \boldsymbol { V } _ { i } ^ { t } ) \} _ { i \in \boldsymbol { I } }$

## 3.1 Assumptions & Definitions

Braun et al. [1] name as their basic assumption indistinguishability among agents. They posit that indistinguishability requires agents to have the same local actions and observations, and that another consequence is that permutations of actions (observations) map to the same outcome in transition, reward, and sensor function, which we posit boils down to two assumptions, which we formalise next.

The first assumption is that all agents $i , j$ in a partition $\Im \subseteq I$ have the same action and observation space:

$$
\operatorname { r a n } ( A _ { i } ) = \operatorname { r a n } ( A _ { j } ) \wedge \operatorname { r a n } ( O _ { i } ) = \operatorname { r a n } ( O _ { j } )\tag{5}
$$

which follows the requirement definition in [1]. The second assumption is that the transition, reward, and sensor functions $T , R ,$ and $\varOmega$ are symmetric, meaning that exchanging actions or observations between agents of a partition has no efect on the outcome of $T , R ,$ and $\varOmega ,$ which can be formally expressed for the actions of two agents as:

$$
\begin{array} { c } { { \forall a \in \mathrm { r a n } ( A ) , a = ( a _ { i } \circ a _ { j } \circ a _ { - i , - j } ) : } } \\ { { { \cal T } ( s ^ { \prime } , s , a _ { i } \circ a _ { j } \circ a _ { - i , - j } ) = { \cal T } ( s ^ { \prime } , s , a _ { j } \circ a _ { i } \circ a _ { - i , - j } ) } } \end{array}\tag{6}
$$

The same holds for the actions in $R$ and the observations in $\varOmega .$ We now generalise Eq. (6) to entire partitions, to build interchangeability among agents within a partition. The set of agents I is divided into K partitions $\Im _ { 1 } , . . . , \Im _ { K }$ of indistinguishable agents. For each partition $\Im _ { k }$ , let $\mathbf { \boldsymbol { a } } _ { k } = ( \boldsymbol { a } _ { i } ) _ { i \in \mathfrak { I } _ { k } }$ and $\pmb { o } _ { k } = ( o _ { i } ) _ { i \in \Im _ { k } }$ denote the joint action and observation tuples of the agents in that partition. With the extension of symmetric behaviour to partitions, swapping the actions or observations of agents within the same partition does not change a function’s outcome. Formally, for any permutation θ of $\Im _ { k }$

$$
T ( s ^ { \prime } , s , { \pmb a } _ { k } , { \pmb a } _ { - k } ) = T ( s ^ { \prime } , s , \theta ( { \pmb a } _ { k } ) , { \pmb a } _ { - k } ) { \pmb o } _ { - k } , s ) .\tag{7}
$$

with the same holding for any permutations of actions in $T$ and observations in Ω. These assumptions are analogous to those made for counting in POSGs [9].

Equations (5) and (7) allow for encoding actions and observations using histograms, which can then be used to compactly encode $T , R ,$ , and $\varOmega$ by replacing the individual mappings from all $\boldsymbol { \theta } ( \boldsymbol { a } _ { k } )$ to an outcome $\rho$ with a single mapping $h \mapsto \rho ,$ with h being a histogram of the counts of actions in $\mathbf { \delta } _ { \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } } \mathbf { \delta } _ { \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { ~ \textit ~ { ~ a ~ k ~ } ~ } }$ , which is identical for all $\boldsymbol { \theta } ( \boldsymbol { a } _ { k } )$ . The same holds for the permutations of observations in $\mathbf { \xi } _ { o _ { k } }$ Formally, histograms are defined as follows:

Definition 2. Given a set of n agents I and a set of m values $\nu = \{ v _ { 1 } , \ldots , v _ { m } \}$ ， a histogram h is defined as $h \ = \ \{ ( v _ { l } , n _ { l } ) \} _ { l = 1 } ^ { m }$ with $\Sigma _ { l = 1 } ^ { m } n _ { l } = n$ where each n denotes the number of instances with value v . We use $h ( v _ { l } )$ to refer to $n _ { l } .$ $[ n _ { 1 } , . . . , n _ { m } ]$ is used as a shorthand. A counting random variable $( C R V ) \# \boldsymbol { \mathbf { \mathit { r } } } [ V ]$ is a syntactic construct whose range consists of all histograms over the range values of variable $V , i . e . , \nu = \mathrm { r a n } ( V )$ , that meet $\begin{array} { r } { \sum _ { l } n _ { l } = | { \pmb I } | } \end{array}$

Considering the shorthand example $h = [ 4 , 8 ]$ from above for actions $^ { a , }$ b and $n = 1 2$ , the full version reads $h = \{ ( a , 8 ) , ( b , 4 ) \}$

Braun et al. [1] then define a partitioned DecPOMDP that uses such histograms as range values for actions and observations in each partition for every afected component of the DecPOMDP.

Definition 3 ([1]). A counting DecPOMDP $\bar { M _ { c } }$ is a tuple $( \bar { I } , \bar { S } , \bar { A } _ { c } , \bar { T } , \bar { R } , \bar { O } _ { c } ,$ ${ \bar { \varOmega } } , \tau )$ , with

– <sup>¯</sup>I a partitioning $\{ \Im _ { k } \} _ { k = 1 } ^ { K }$ of N agents, with $\begin{array} { r } { n _ { k } = | \Im _ { k } |  a n d | \bar { \pmb { I } } | = \sum _ { k } n _ { k } = N } \end{array}$

$S$ a random variable with a set of states as range,

$\bar { \mathbf { A } } _ { c } = \{ \# \mathfrak { s } _ { k } [ A _ { k } ] \} _ { k = 1 } ^ { K }$ a set of decision CRVs,

$\bar { T } ( S ^ { \prime } , S , \bar { \mathbf { A } } _ { c } )$ a transition function with counted actions,

$\bar { R } ( S , \bar { A } _ { c } )$ a reward function with counted actions,

$- ~ { \bar { O } } _ { c } = \{ \# _ { \Im _ { k } } [ O _ { k } ] \} _ { k = 1 } ^ { K }$ a set of CRVs,

$- \ \bar { \varOmega } ( \bar { O } _ { c } , S )$ a sensor function with counted observations, and

$\mathit { \Pi } - \ \tau \ \cdot \ \boldsymbol { a }$ horizon.

The equivalence between a DecPOMDP fulfilling Eqs. (5) and (7) and a counting DecPOMDP can be shown by construction, replacing all mappings with permutations of inputs with a single mapping with a histogram and vice versa (see App. B.1).

## 3.2 (In)Tractability Results

For a ground DecPOMDP, the model complexity lies in $O ( s ^ { 2 } a ^ { N } )$ for $T , { \cal O } ( s a ^ { N } )$ for $R ,$ and $O ( s o ^ { N } )$ for $\varOmega$ with $s = \vert \mathrm { r a n } ( S ) \vert , a = \mathrm { m a x } _ { i \in I }$ |ran(A )|, and $o =$ $\operatorname* { m a x } _ { i \in I } \left| \operatorname { r a n } ( O _ { i } ) \right|$ , while the cost of evaluating a joint policy lies in $\stackrel { \cdot } { O } ( s o ^ { N \tau } )$ and the policy space size in $O ( a ^ { \frac { N ( o ^ { \tau } - 1 ) } { o - 1 } } )$ [16]. Using CRVs allows for reducing the model size and evaluation cost from exponential to polynomial.

Theorem 1 ([1], Thm. 2). A counting DecPOMDP $\bar { M _ { c } }$ allows for representation $\mathit { \Omega } / i . e . ,$ , model] and cost to depend polynomially on $N$ .

The theorem holds by the range sizes of the CRVs, which are capped by $N ^ { a }$ $a = \operatorname* { m a x } _ { k } \left| \operatorname { r a n } ( A _ { k } ) \right|$ , and $N ^ { o } , o = \operatorname* { m a x } _ { k } \left| \operatorname { r a n } ( O _ { k } ) \right|$ |, as an upper bound on the number of histograms given by the binomial coeficient $\binom { N + v - \bar { 1 } } { v - 1 } \leq N ^ { v } , v \in \{ a , o \}$ [13]. Compared to the ground case, a and o are replaced by $N ^ { a }$ and $N ^ { o }$ , while K replaces N in the exponent, yielding a polynomial dependence on $N$ , assuming that $K \ll N$

However, the number of policies explodes: Since the size of the policy space contains o in the exponent, replacing o with $N ^ { o }$ means that N remains in the exponent, all the while a is replaced with $N ^ { a }$ , leading to an expression that has $N$ in the base as well as the exponent.

Corollary 1. The policy space of a counting DecPOMDP $\bar { M _ { c } }$ depends exponentially on N.

The crux is the choice of CRVs in the action and observation set used for building policies. In contrast, we use plain variables and count policies instead, which is a small change with profound efects.

## 4 Containing the Fire by Counting Policies

For a polynomial dependence on the agent numbers, we keep plain variables in the action and observation sets, enabled by Eq. (5), but use CRVs in the transition, reward, and sensor function, enabled by Eq. (7). With plain variables, there is a set of (representative) policies available in each partition, which allows for structuring the policy space by counting the number of agents following each representative policy and updating the MEU calculations. We first formalise counted policies before renewing the definition of counting DecPOMDPs. Last, we look at the efect of counted policies on the MEU calculation and complexity.

## 4.1 Counted Policies

Let us consider the example from the beginning of Section 3 with 12 agents and actions $a , b$ available. At step $t = 1$ , assuming two possible observations $x , y ,$ each of the 12 agents has the same 8 policies available, see Fig. 1. Instead of tracking all possible policies for each individual agent, we can use these 8 policies as representative policies to keep track of agent behaviour in histograms: For example, the partition policy in which all agents follow the first policy is encoded by $[ 1 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]$ , while the other configuration in which one agent follows the second policy and the remaining agents follow the first one is encoded by $[ 1 1 , 1 , 0 , 0 , 0 , 0 , 0 , 0 ]$ . In this way, all possible partition policies can be compactly encoded.

More formally, according to Eq. (5), the set of policies for any two agents $i , j$ within a partition $\Im _ { k }$ is established using the same sets of actions $A _ { i } = A _ { j } = A _ { k }$ and observations $O _ { i } = O _ { j } = O _ { k }$ . As a result, their local policy space is the same, i.e., $\varPi _ { i } = I I _ { j } = I I _ { k }$ . These policies are referred to as representative policies $\varPi _ { k }$ of partition $\Im _ { k }$ and can be counted for each partition.

Definition 4. Let the set of J representative policies of a partition $\Im _ { k }$ be given by $\varPi _ { k } = \{ \pi _ { 1 } , . . . , \pi _ { J } \}$ . A counted partition policy over $\varPi _ { k }$ is a histogram $\dot { h } _ { k } ^ { \varPi } =$ $\{ ( \pi _ { j } , m _ { j } ) \} _ { j = 1 } ^ { J }$ , where $m _ { j } \geq 0$ is the number of agents in $\Im _ { k }$ following policy $\pi _ { j } ,$ and $\begin{array} { r } { \sum _ { j = 1 } ^ { J } m _ { j } = n _ { k } } \end{array}$ . The counted policy space for partition $\Im _ { k }$ , denoted by $H _ { k } ^ { \varPi }$ is given by the set of all histograms meeting $\begin{array} { r } { \sum _ { j = 1 } ^ { J } m _ { j } = n _ { k } } \end{array}$ . The joint counted policy space over all partitions is given by $\pmb { H } ^ { \varPi } = \times _ { k = 1 } ^ { K } H _ { k } ^ { \varPi }$ . The counted policy space of all partitions except k is denoted by $\pmb { H } _ { - k } ^ { \pi }$ and of all agents except some agent $i \in \Im _ { k }$ by $\pmb { H } _ { - i } ^ { \pi }$ where $\begin{array} { r } { \sum _ { j = 1 } ^ { J } m _ { j } = n _ { k } - 1 } \end{array}$ for $H _ { k } ^ { \varPi }$ without i.

## 4.2 A Renewed Definition

To distinguish the two definitions, we use policy-counted DecPOMDP as a name here, which is still a DecPOMDP with a partitioned agent set fulfilling Eqs. (5) and (7).

Definition 5. A policy-counted DecPOMDP M<sup>¯</sup> is a tuple $( \bar { I } , \bar { S } , \bar { A } , \bar { T } , \bar { R } , \bar { O } .$ $\bar { \varOmega } , \tau )$ , with

– <sup>¯</sup>I a partitioning $\{ \Im _ { k } \} _ { k = 1 } ^ { K }$ of N agents, $n _ { k } = | \Im _ { k } |$ and $\begin{array} { r } { | \bar { \pmb { I } } | = \sum _ { k } n _ { k } = N } \end{array}$

– S a random variable with a set of states as range,

$\bar { \boldsymbol { A } } = \{ A _ { k } \} _ { k = 1 } ^ { K }$ a set of decision random variables,

$- \ \bar { T } ( S ^ { \prime } , \bar { S } , \bar { \bar { A } _ { c } } )$ a transition function with counted actions,

$\bar { R } ( S , \bar { \mathbf { A } } _ { c } ) ~ a$ reward function with counted actions,

$\bar { O } = \{ O _ { k } \} _ { k = 1 } ^ { K }$ a set of random variables,

$- \ \bar { \varOmega } ( \bar { O } _ { c } , S )$ a sensor function with counted observations, and

τ a horizon.

where $\bar { \mathbf { A } } _ { c } = \{ \# _ { \Im _ { k } } [ A _ { k } ] \} _ { k = 1 } ^ { K }$ and $\bar { O } _ { c } = \{ \# _ { \Im _ { k } } [ O _ { k } ] \} _ { k = 1 } ^ { K }$

The diference between Def. 3 and Def. 5 only lies in $\bar { \mathbf { A } } _ { c }$ and $\bar { O } _ { c }$ versus $\bar { \boldsymbol { A } }$ and O<sup>¯</sup> , which are CRVs in the former and plain variables in the latter case, as the set of actions and observations available. Since Eq. (7) holds in both definitions, T<sup>¯</sup>, R<sup>¯</sup>, and $\bar { \varOmega }$ are the same with CRVs as inputs for actions and observations. This encoding captures the symmetric behaviour of agents within each partition and helps to reduce the model complexity by operating over counted representations instead of enumerating all individual agents. Theorem 2 establishes the equivalence between the ground DecPOMDP and its counted version of Def. 5.

Theorem 2. A ground DecPOMDP M that meets $E q s .$ (5) and (7) is equivalent to a policy-counted DecPOMDP M<sup>¯</sup> .

The full proof can be found in App. B.2, which is based on the fact that one can be converted into the other by construction and vice versa. Having established the equivalence between ground and counting DecPOMDPs, we now show how the counting formulation afects the model complexity.

Theorem 3. For a fixed number of partitions K, fixed action and observation spaces, and a fixed horizon τ, the model complexity of a counting DecPOMDP $\bar { M _ { c } }$ is polynomial in the number of agents $N$

The full proof can be found in App. B.3, where we show that using histograms within agent partitions keeps the model size polynomial in N.

## 4.3 Counted Utility Calculation

As the policy space is given by joint counted policies, we can update the semantics and MEU calculation. The semantics of the policy-counted DecPOMDP M<sup>¯</sup> is given by the joint counted policy space $\pmb { H } ^ { \varPi }$ . The policy-counted DecPOMDP problem asks for the joint counted policy that yields the maximum expected utility (with horizon $\tau )$ , which is again computed by recursively calculating the expected utility over the next states and observations, until reaching τ .

However, the sum over observations can now be turned into a sum over counted observations: Given a joint counted policy, each partition is partitioned again into groups following a representative policy according to the counted partition policy, with group sizes corresponding to the histogram counts. For a given (representative) observation history, the recursive call to the next utility as well as the remaining sub-policy are the same for these agents, making them indistinguishable among themselves. As such, for each of these groups, we can count observations in histograms over the (sub)group with a particular representative policy and observation history, and let the sum go over the cross product of these histograms. Conditioning these groups on their shared observation history means that groups split up further, which can in the worst case lead to singletons if the group size is small to begin with or $\tau$ is very large but the overall complexity is still going to be only polynomial as we show afterwards. Additionally, we only need this deep recursion for the utility definition here and are able to define policy-counted dynamic programming using a single step.

To formalise this setting, we define the space of representative observation histories as well as the space of observation histograms given a representative observation history.

Definition 6. Given a step t, a representative observation history $p _ { k } ^ { 0 : t , o }$ of a partition $\Im _ { k }$ is a sequence of observations $o ^ { t ^ { \prime } } \in$ ran $( O _ { k } )$ of length t, $t ^ { \prime } \in$ $\{ 1 , \ldots , t \}$ , with $o ^ { 0 } = \bot$ as an empty observation. Given a representative observation history $p _ { k } ^ { 0 : t , o }$ , let $p _ { k } ^ { t ^ { \prime } , o }$ refer to the entry at point $t ^ { \prime }$ in the sequence and $p _ { k } ^ { 0 : t - 1 , o }$ refer to the subsequence without the last entry $p _ { k } ^ { t , o }$ . Finally, let $P _ { k } ^ { 0 : t , o }$ denote the space of representative observation histories of length t for the agents in partition I<sub>k</sub>, i.e., $\begin{array} { r } { \hat { P } _ { k } ^ { 0 : t , o } = \perp \times \times _ { t ^ { \prime } = 1 } ^ { t } \mathrm { r a n } ( O _ { k } ) } \end{array}$

Definition 7. Given a step t, a counted partition policy $h _ { k } ^ { \varPi }$ , and a representative policy $\pi \in \pi _ { k }$ for a partition $\Im _ { k }$ , a counted partition observation is a set of histograms $\{ h _ { { k , p } _ { k } ^ { 0 ; t , o } } ^ { o , \pi } \} _ { { p } _ { k } ^ { 0 ; t , o } \in { P } _ { k } ^ { 0 ; t , o } }$ where

$$
h _ { k , p _ { k } ^ { 0 : t , o } } ^ { o , \pi } = \{ ( o _ { l } , n _ { l } ) \} _ { o _ { l } \in \mathrm { r a n } ( O _ { k } ) }
$$

with

$$
\sum _ { o _ { l } \in \mathrm { r a n } ( O _ { k } ) } n _ { l } = \left\{ { \begin{array} { l l } { h _ { k , p _ { k } ^ { 0 ; t - 1 , o } } ^ { o , \pi } ( p _ { k } ^ { t } ) } & { i f t > 0 } \\ { h _ { k } ^ { \pi } ( \pi ) } & { i f t = 0 } \end{array} } \right.\tag{8}
$$

and $p _ { k } ^ { 0 : t , o } = p _ { k } ^ { 0 : t - 1 , o } \circ p _ { k } ^ { t }$ . Let $H _ { k } ^ { o , \pi }$ denote the space of counted partition observations for a given representative policy $\pi , i . e .$ , the set of all possible sets of histograms that fulfil Eq. (8). Given a joint counted policy $\Breve { h ^ { \prime \prime } }$ , let $H _ { k } ^ { o , \varPi }$ denote the space of counted partition observations over all representative policies $\varPi _ { k : }$ i.e., ${ \bf \bar { \cal H } } _ { k } ^ { o , \pi } \ : = \ : \times _ { \pi \in \pi _ { k } } { \bar { \cal H } } _ { k } ^ { o , \pi }$ , and ${ \cal H } ^ { o , \varPi }$ the space of joint counted observations, $i . e . , H ^ { o , \pi } = \times _ { k = 1 } ^ { K } H _ { k } ^ { o , \pi }$

Note that the observation histogram in $\operatorname { E q . }$ . (8) holds counts $n _ { l }$ for $p ^ { 0 : t , o } \circ o _ { l }$ Based on these definitions, we can now rewrite Eq. (1) as follows:

$$
\index { M E U ( \bar { M } ) } = \underset { \boldsymbol { h } ^ { \pi } \in \boldsymbol { H } ^ { \pi } } { \arg \operatorname* { m a x } } U _ { \bar { M } } ( \boldsymbol { h } ^ { I I } ) , \underset { \boldsymbol { h } ^ { \pi } \in \boldsymbol { H } ^ { \pi } } { \operatorname* { m a x } } U _ { \bar { M } } ( \boldsymbol { h } ^ { I I } ) )\tag{9}
$$

where $U _ { \bar { M } } ( h ^ { \pi } )$ is calculated recursively over τ steps and weighted according to the prior $T ( s ^ { 0 } , \dots , \cdot )$ , i.e., $\begin{array} { r } { U _ { \bar { M } } ( \pmb { h } ^ { \bar { M } } ) = \sum _ { s ^ { 0 } \in \mathrm { r a n } ( S ) } T ( s _ { 0 } , . ~ , . ~ ) U _ { \bar { M } } ^ { \pmb { h } ^ { \bar { M } } } ( s ^ { 0 } , \bot ^ { 0 : 0 } ) } \end{array}$ with

$$
\begin{array} { l } { { { \cal U } _ { \cal \bar { M } } ^ { h ^ { \cal I } } ( s ^ { t } , ( h ^ { o , I } ) ^ { 0 : t } ) = R ( s ^ { t } , h ^ { a } ) + \displaystyle \sum _ { s ^ { t + 1 } \in \mathrm { r a n } ( S ) } T ( s ^ { t + 1 } , s ^ { t } , h ^ { a } ) } } \\ { { \displaystyle \sum _ { h ^ { o , I } \in { \cal H } ^ { o , I } } M u l ( h ^ { o , I } ) \cdot \varOmega ( h ^ { o } , s ^ { t + 1 } ) { \cal U } _ { \cal \bar { M } } ^ { h ^ { \cal I } } ( s ^ { t + 1 } , ( h ^ { o , I } ) ^ { 0 : t } \circ h ^ { o , I } ) } } \end{array}\tag{10}
$$

with $M u l ( \boldsymbol { h } ^ { o , \pi } )$ the multinomial coeficient, denoting the number of ground permutations encoded by the histograms [23], where for each partition $\Im _ { k }$ , the counted partition action $h _ { k } ^ { a } \in \mathfrak { h } ^ { a }$ as input to R and $T$ is given by adding up, for each action $a \in { \mathrm { r a n } } ( A _ { k } )$ , how often a is carried out over all representative policies and all representative observation histories using the counts in the latest counted partition observation if the observation history plus the newest observation map to a in the current representative policy, i.e.,

$$
n _ { k } ^ { a } = \left\{ \begin{array} { l l } { \displaystyle { a _ { l } , n _ { l } } \sum _ { n } \sum _ { j \in \Gamma } \sum _ { o \in } h _ { k , p } ^ { o , \Pi , t } ( o ) \mathbb { 1 } _ { \pi _ { k } ( p \circ o ) = a _ { l } } } & { \mathrm { i f ~ } t > 0 } \\ { \displaystyle { \sum _ { \pi \in \pi _ { k } } \sum _ { p \in \Gamma } h _ { k } ^ { ( \tau + 1 , o \mathrm { ~ r a n } ( O _ { k } ) } } } & { \mathrm { i f ~ } t = 0 } \end{array} \right.\tag{11}
$$

with $h _ { k , p } ^ { o , \Pi , t } \in { \pmb { h } } _ { k } ^ { o , \Pi , t } \in ( { \pmb { h } } _ { k } ^ { o , \Pi } ) ^ { o : t }$ and 1 being the indicator function returning 1 if $\pi _ { k } ( p \circ o ) = a _ { l } .$ . The counted observation $h _ { k } ^ { o } \in \mathrm { r a n } ( \# { \mathfrak { z } } _ { k } [ O _ { k } ] )$ as input to Ω is given by adding up, for each observation $o \in { \mathrm { r a n } } ( O _ { k } )$ , how often o is observed over all representative policies and all representative observation histories in the latest counted partition observation, i.e.,

$$
h _ { k } ^ { o } = \{ ( o _ { l } , n _ { l } ) \} _ { o _ { l } \in \mathrm { r a n } ( O _ { k } ) } \quad \mathrm { w i t h } \quad n _ { l } = \sum _ { \pi \in { \cal I } _ { k } } \sum _ { p \in { \cal P } _ { k } ^ { 0 : t - 1 , o } } h _ { k , p } ^ { o , { \cal I I } , t } ( o _ { l } )\tag{12}
$$

The utility calculation for a ground DecPOMDP and an equivalent policycounted DecPOMDP is equivalent again, leading to the same maximum expected utility. The trick is that calculations that lead to the same result are only performed once in the counted case, which is possible due to the histograms essentially encoding when calculations lead to the same result.

Theorem 4. The maximum expected utility for a ground DecPOMDP M that meets Eqs. (5) and (7) is equal to the maximum expected utility of the corresponding policy-counted DecPOMDP M<sup>¯</sup> . That $i s , \ M E U ( M ) = M E U ( { \bar { M } } )$

The full proof can be found in App. B.4, where we show that both models give the same utility by mapping ground policies to counted policies. Next, we look at evaluation cost and policy space size.

Theorem 5. For a fixed number of partitions K, fixed representative action and observation spaces, and a fixed horizon τ, the cost of evaluating a joint counted policy and the size of the overall joint counted policy space in a policy-counted DecPOMDP depend polynomially on the number of agents N.

The full proof can be found in App. B.5, which relies on the joint counted observation space and the joint counted policy space depending polynomially on the number of agents N, as $n _ { k } \leq N$ agents are distributed onto the diferent representative observations and policies, which can be characterised using the binomial coeficient. Thus, we have reduced the complexity from an exponential one to a polynomial one for N. As such, a policy-counted DecPOMDP is tractable in the number of agents.

Corollary 2. Given a fixed number of partitions K, fixed representative action and observation spaces, and a fixed horizon τ, a policy-counted DecPOMDP is tractable in the number of agents N.

With the number of agents N no longer appearing in the exponent, we are able to solve larger problem instances in terms of N compared to the ground case, especially if the number of partitions K is small. Next, we use the counting encoding to speed up dynamic programming for DecPOMDPs.

## 5 Policy-Counted Dynamic Programming

In this section, we extend the standard dynamic programming operator for policy-counted DecPOMDPs. Because of the equivalence between the ground and policy-counted versions of DecPOMDPs, we can show that the value computations are equivalent, allowing for computing values at a partition level as well as pruning representative policies.

The policy-counted dynamic programming operator still follows the three general steps of exhaustive backup, value computation, and pruning but on a partition-level. That is, Alg. 1 still applies but computes value vectors and performs pruning according to counted versions of those equations for each partition (see Eqs. (13) and (14)). During the exhaustive backup, the operator builds all representative policies of depth t based on the policies of depth t − 1 for each partition and then computes their corresponding value vectors. Since the operator builds increasingly deeper policies, the value computation stays at the top level, building its counted joint action from the root actions in the representative policies, i.e., with $t = 0$ in Eq. (11) to build $\pmb { h } ^ { a }$ , and uses empty representative histories, i.e., Eq. (8) with $t = 0$ as well to build $H ^ { o , { \cal I I } }$ . Thus, for a state s and joint counted policy $\pmb { h } _ { - i } ^ { \pi }$ of the other agents and partitions, the value computation for a given representative policy $\pi _ { k }$ is defined as follows with $\pi _ { k } \oplus h _ { - i } ^ { \pi }$ adding 1 to the corresponding count in $h _ { k } ^ { \varPi }$ of $\pmb { h } _ { - i } ^ { \pi }$ to form a full joint counted policy:

$$
\bar { V } _ { k } ( s , h ^ { I I } ) = R ( s , h ^ { a } ) + \sum _ { s ^ { \prime } \in \mathrm { r a n } ( S ) } T ( s ^ { \prime } , s , h ^ { a } ) ~\tag{13}
$$

where ${ \pmb h . } { \pmb h ^ { o , \varPi } }$ denotes the policy histogram that follows from the observation histogram ${ { h } ^ { o , \varPi } }$ . Specifically, given depth t of the representative policies $\it { \Pi } \Pi _ { k } ^ { t }$ used for $\boldsymbol { h } ^ { \bar { \boldsymbol { \pi } } }$ , the joint counted policy $\boldsymbol { h } ^ { \ : H }$ , and the current joint counted observation ${ { h } ^ { o , \varPi } }$ , the new policy histogram is calculated by adding up how often a subpolicy $\bar { \pi _ { l } } \in \boldsymbol { \Pi } _ { k } ^ { t - 1 }$ remains over the diferent representative policies $\pi \in \boldsymbol { \pi } _ { k } ^ { t }$ when following the diferent observations $o \in { \mathrm { r a n } } ( O _ { k } )$ given the counts in $h _ { k } ^ { o , \pi } \in \mathfrak { h } ^ { o , \pi }$ for o, i.e.,

$$
h . h ^ { o , I I } = \{ ( \pi _ { l } , n _ { l } ) \} _ { \pi _ { l } \in { \cal { I } } _ { k } ^ { t - 1 } } ~ \mathrm { w i t h } ~ n _ { l } = \sum _ { \pi \in { \cal { I } } _ { k } ^ { t } } \sum _ { o \in \mathrm { { r a n } } ( O _ { k } ) } h _ { k } ^ { o , I I } ( o ) ~ | ~ \pi . o = \pi _ { l }
$$

During pruning, the operator eliminates weakly dominated representative policies $\pi _ { k }$ based on the following equation (solvable by a linear programme, see App. A.2):

$$
f o r a l l s \in \mathrm { r a n } ( S ) , h _ { - i } ^ { t } \in H _ { - i } ^ { t } \exists \pi _ { l \neq k } ^ { t } \in H _ { k } ^ { t } : { \bar { V } } _ { k } ( s , \pi _ { k } ^ { t } \oplus h _ { - i } ^ { t } ) \leq { \bar { V } } _ { k } ( s , \pi _ { l } ^ { t } \oplus h _ { - i } ^ { t } )\tag{14}
$$

Equation (14) ensures that, for each state and each counted policy of the other partitions, a representative policy is only pruned if there exists another one with equal or higher value.

Next, we show that counted dynamic programming using Eqs. (13) and (14) is equivalent to dynamic programming using Eqs. (3) and (4), which is a culmination of the formalisations and theorems so far.

Theorem 6. Using policy-counted dynamic programming on a policy-counted DecPOMDP M<sup>¯</sup> is equivalent to using standard dynamic programming on a ground DecPOMDP M, in which Eqs. (5) and (7) hold.

The full proof can be found in App. B.6, where we show that each step of the dynamic programming operator works equivalently in both the ground and counted models. Given the equivalence of the policy-counted and standard version of dynamic programming, correctness of the policy-counted version follows directly from the correctness of the standard version.

Corollary 3. Policy-counted dynamic programming using Eqs. (13) and (14) is correct.

Next, we show that policy-counted dynamic programming depends polynomially on the number of agents assuming a fixed and small number of representative actions and observations and partitions as well as horizon.

Theorem 7. For a fixed number of partitions K, fixed representative action and observation spaces, and a fixed horizon τ , the runtime of policy-counted dynamic programming in a policy-counted DecPOMDP M<sup>¯</sup> depends polynomially on the number of agents N.

The full proof can be found in App. B.7, which analyzes the polynomial runtime of the backup, value calculation, and pruning steps.

Appendix C shows the well-known DecTiger model [14] as ground, counting, and policy-counted DecPOMDPs to highlight the similarities and diferences between the models. In summary, policy-counted dynamic programming is able to use the compact encoding of policy-counted DecPOMDPs to generate a solution that is equivalent to the ground solution in time polynomial in N instead of exponential in N, assuming all other parameters to be fixed. As the DecTiger example shows, the pay-of only arises if N is suficiently large, i.e., if the assumption of $N \ll K$ indeed holds. Otherwise the counting of policies introduces too much overhead. Future work is left to deal with the remaining exponent that can still become rather large.

## 6 Conclusion

This paper presents policy-counted DecPOMDPs, which exploit a partitioned agent set under certain structural assumptions to reduce the complexity from exponential to polynomial in the number of agents, assuming all other parameters to be fixed. Specifically, we are able to use representative policies per partition, enabling counting how many agents follow which policy, which allows for a compact representation and considerably reduces the model’s size, computing cost, and policy space, leading to tractability in agent numbers assuming a fixed number of partitions, horizon, and representative action and observation spaces. Additionally, we present a policy-counted dynamic programming for policy-counted DecPOMDPs that updates policies and values as well as prunes policies directly at the partition level. As a consequence, policy-counted DecPOMDPs ofer a lot of potential for large-scale applications such as nanoscale medical systems.

In future research, we aim to further enhance scalability by working on policy encodings as well as applying lifting to large state spaces, to enable additional complexity reductions. Another idea is to look at approximations, considering that histograms are rarely needed in count steps of 1.

## References

1. Braun, T., Gehrke, M., Lau, F., Möller, R.: Lifting in Multi-agent Systems under Uncertainty. In: UAI-22 Proc. of the 38th Conference on Uncertainty in Artificial Intelligence. pp. 1–8. AUAI Press (2022)

2. Carr, S., Jansen, N., Bharadwaj1, S., Spaan, M.T.J., Topcu, U.: Safe policies for factored partially observable stochastic games. In: RSS-21 Proc. of Robotics: Science and Systems XVII. pp. 1–11. RSS Foundation (2021)

3. Emery-Montemerlo, R., Gordon, G., Schneider, J., Thrun, S.: Approximate solutions for partially observable stochastic games with common payofs. In: AAAMAS-04 Proc. of the 3rd International Joint Conference on Autonomous Agents and Multiagent Systems. pp. 136–143. IEEE (2004)

4. Hansen, E.A., Bernstein, D.S., Zilberstein, S.: Dynamic programming for partially observable stochastic games. In: AAAI-04 Proc. of the 19th National Conference on Artificial Intelligence. vol. 4, pp. 709–715 (2004)

5. Horák, K., Bošansk\`y, B.: Solving partially observable stochastic games with public observations. In: Proc. of the AAAI conference on Artificial Intelligence. pp. 547– 552. AAAI Press (2019)

6. Horák, K., Bošansk\`y, B., Kiekintveld, C., Kamhoua, C.: Compact Representation of Value Function in Partially Observable Stochastic Games. In: IJCAI-19 Proc. of the 28th International Joint Conference on Artificial Intelligence. pp. 350–356. IJCAI Organisation (2019)

7. Horák, K., Bošansk\`y, B., Pěchouček, M.: Heuristic Search Value Iteration for Onesided Partially Observable Stochastic Games. In: AAAI-17 Proc. of the 31st AAAI Conference on Artificial Intelligence. pp. 558–564 (2017)

8. Karabulut, N.N., Braun, T.: Lifting partially observable stochastic games. In: International Conference on Scalable Uncertainty Management. pp. 201–216. Springer (2024)

9. Karabulut, N.N., Braun, T.: Counting agents in partially observable stochastic games. In: European Conference on Symbolic and Quantitative Approaches with Uncertainty. pp. 207–222. Springer (2025)

10. Koops, W., Jansen, N., Junges, S., Simao, T.D.: Recursive Small-step Multi-agent A\* for Dec-POMDPs. In: IJCAI-23 Proceedings of the 32nd International Joint Conference on Artificial Intelligence. pp. 5402–5410. IJCAI Organization (2023)

11. Koops, W., Junges, S., Jansen, N.: Approximate Dec-POMDP Solving Using Multiagent A\*. In: IJCAI-24 Proceedings of the 33nd International Joint Conference on Artificial Intelligence. pp. 6743–6751. IJCAI Organization (2024)

12. Kumar, A., Zilberstein, S.: Dynamic programming approximations for partially observable stochastic games. In: FLAIRS-09 Proc. of the 22nd International Florida Artificial Intelligence Research Society Conference. AAAI Press (2009)

13. Milch, B., Zettelmoyer, L.S., Kersting, K., Haimes, M., Kaelbling, L.P.: Lifted Probabilistic Inference with Counting Formulas. In: AAAI-08 Proc. of the 23rd AAAI Conference on Artificial Intelligence. pp. 1062–1068. AAAI Press (2008)

14. Nair, R., Tambe, M., Yokoo, M., Pynadath, D.V., Marsella, S.: Taming Decentralized POMDPs: Towards Eficient Policy Computation for Multiagent Settings. In: IJCAI-03 Proc. of the 18th International Joint Conference on Artificial Intelligence. pp. 705–711. IJCAI Organization (2003)

15. Niepert, M., Van den Broeck, G.: Tractability through Exchangeability: A New Perspective on Eficient Probabilistic Inference. In: AAAI-14 Proc. of the 28th AAAI Conference on Artificial Intelligence. pp. 2467–2475. AAAI Press (2014)

16. Oliehoek, F.A., Amato, C.: A Concise Introduction to Decentralised POMDPs. Springer (2016)

17. Oliehoek, F.A., Whiteson, S., Spaan, M.T.: Lossless Clustering of Histories in Decentralized POMDPs. In: AAMAS-09 Proceedings of the 8th International Conference on Autonomous Agents and Multiagent Systems. pp. 577–584. IFAAMAS (2009)

18. Poole, D.: First-order Probabilistic Inference. In: IJCAI-03 Proc. of the 18th International Joint Conference on Artificial Intelligence. pp. 985–991. IJCAI Organization (2003)

19. Russell, S., Norvig, P.: Artificial Intelligence: A Modern Approach. Pearson (2021)

20. Seuken, S., Zilberstein, S.: Memory-Bounded Dynamic Programming for DEC-POMDPs. In: IJCAI-07 Proceedings of the 21st International Joint Conference on Artificial Intelligence. pp. 2009–2015. IJCAI Organization (2007)

21. Szer, D., Charpillet, F.: Point-based dynamic programming for dec-pomdps. In: AAAI. vol. 6, pp. 1233–1238 (2006)

22. Szer, D., Charpillet, F., Zilberstein, S.: MAA\*: A Heuristic Search Algorithm for Solving Decentralized POMDPs. In: UAI-05 Proceedings of the 21st Conference on Uncertainty in Artificial Intelligence. pp. 576–583. ACM (2005)

23. Taghipour, N., Fierens, D., Davis, J., Blockeel, H.: Lifted Variable Elimination: Decoupling the Operators from the Constraint Language. Journal of Artificial Intelligence Research 47(1), 393–439 (2013)

24. Tomášek, P., Horák, K., Aradhye, A., Bošansk\`y, B., Chatterjee, K.: Solving partially observable stochastic shortest-path games. In: IJCAI-21 Proc. of the 30th International Joint Conference on Artificial Intelligence. pp. 4182–4189. IJCAI Organisation (2021)

25. Vaidya, P.M.: Speeding-up Linear Programming Using Fast Matrix Multiplication. In: 30th Annual Symposium on Foundations of Computer Science. pp. 332–337. IEEE Computer Society (1989)

26. Wiggers, A.J., Oliehoek, F.A., Roijers, D.M.: Structure in the value function of two-player zero-sum games of incomplete information. In: ECAI-16 Proc. of the 22nd European Conference on Artificial Intelligence. pp. 1628–1629. IOS Press (2016)

# The Curious Case of Exploding DecPOMDPs: Containing the Fire through Policy Counting (Supplementary Material)

## A Linear Programmes

This section provides the linear programmes used during the pruning step in dynamic programming, which repeats the pruning condition from the main paper and then lists the linear programme.

## A.1 Ground DecPOMDP

Formally, a policy $\pi _ { i , j } ^ { t }$ is pruned if the following holds,

$$
\forall s \in \mathrm { r a n } ( S ) , \pi _ { - i } ^ { t } \in \pmb { { \cal { I } } } _ { - i } ^ { t } \exists \pi _ { i , j ^ { \prime } } ^ { t } \in { \cal { I } } _ { i } ^ { t } : V _ { i } ^ { \prime } ( s , \pi _ { i , j } ^ { t } \circ \pi _ { - i } ^ { t } ) \leq V _ { i } ^ { \prime } ( s , \pi _ { i , j ^ { \prime } } ^ { t } \circ \pi _ { - i } ^ { t } )
$$

which can be solved by solving the following linear programme (prune $\pi _ { i , j } ^ { t }$ if $d < 0 )$ :

variables: $b _ { i } ( s , \pi _ { - i } ) , d$

maximise: d

constraints:

$$
\begin{array} { l } { { \displaystyle \forall \pi _ { i , j ^ { \prime } } ^ { t } \in I I _ { i } ^ { t } } } \\ { { \displaystyle \sum _ { s \in \mathrm { r a n } ( S ) } \sum _ { \pi _ { - i } \in I I _ { - i } ^ { t } } b _ { i } ( s , \pi _ { - i } ) V _ { i } ( s , \pi _ { i , j } ^ { t } , \pi _ { - i } ^ { t } ) } } \\ { { - \displaystyle \sum _ { s \in \mathrm { r a n } ( S ) } \sum _ { \pi _ { - i } \in I I _ { - i } ^ { t } } b _ { i } ( s , \pi _ { - i } ) V _ { i } ( s , \pi _ { i , j ^ { \prime } } ^ { t } , \pi _ { - i } ^ { t } ) - d \le 0 } } \\ { { \displaystyle \sum _ { s \in \mathrm { r a n } ( S ) } \sum _ { \pi _ { - i } \in I I _ { - i } ^ { t } } b _ { i } ( s , \pi _ { - i } ) = 1 , b _ { i } ( s , \pi _ { - i } ) > 0 } } \end{array}
$$

## A.2 Policy-Counted DecPOMDP

Formally, a representative policy $\pi _ { k } ^ { t }$ is pruned if the following holds, which can still be solved using a linear program:

$$
\forall s \in \mathrm { r a n } ( S ) , h _ { - i } ^ { t } \in H _ { - i } ^ { t } \exists \pi _ { l \neq k } ^ { t } \in H _ { k } ^ { t } : \bar { V } _ { k } ^ { \prime } ( s , \pi _ { k } ^ { t } \oplus h _ { - i } ^ { t } ) \leq \bar { V } _ { k } ^ { \prime } ( s , \pi _ { l } ^ { t } \oplus h _ { - i } ^ { t } ) ,
$$

which can be solved by solving the following linear programme (prune $\pi _ { k , l } ^ { t }$ if $d < 0 )$ :

variables: $b _ { k } ( s , \pmb { h } _ { - i } ) , d$

maximise: d

$$
\begin{array} { r l } { \mathrm { c o n s t r a i n t s : } \ \forall \pi _ { k , l ^ { \prime } } ^ { t } \in \boldsymbol { H } _ { k } ^ { t } } & { } \\ & { \qquad \displaystyle \sum _ { s \in \mathrm { r a n } ( S ) } \sum _ { h _ { - i } \in \boldsymbol { H } _ { - i } ^ { t } } b _ { k } ( s , h _ { - i } ) V _ { k } ( s , \pi _ { k , l } ^ { t } \oplus h _ { - i } ^ { t } ) } \\ & { \qquad - \displaystyle \sum _ { s \in \mathrm { r a n } ( S ) } \sum _ { h _ { - i } \in \boldsymbol { H } _ { - i } ^ { t } } b _ { k } ( s , h _ { - i } ) V _ { k } ( s , \pi _ { k , l ^ { \prime } } ^ { t } \oplus h _ { - i } ^ { t } ) - d \le 0 } \\ & { \qquad \displaystyle \sum _ { s \in \mathrm { r a n } ( S ) } \sum _ { h _ { - i } \in \boldsymbol { H } _ { k - i } ^ { t } } b _ { k } ( s , h _ { - i } ) = 1 , b _ { k } ( s , h _ { - i } ) > 0 } \end{array}
$$

## B Full Proofs

## B.1 Equivalence between a Ground DecPOMDP and a Counting DecPOMDP

This theorem shows equivalence between a ground DecPOMDP and a counting DecPOMDP based on Eqs. (5) and (7).

Theorem 1. A ground DecPOMDP M that meets Eqs. (5) and (7) is equivalent to a counting DecPOMDP $\bar { M _ { c } }$

Proof. To convert a DecPOMDP $M = ( I , S , A , O , T , R , \varOmega , \tau )$ into a counting DecPOMDP $\bar { M } _ { c } = ( \bar { I } , S , \bar { A } _ { c } , \bar { O } _ { c } , \bar { T } , \bar { R } , \bar { \varOmega } , \tau )$ , we apply Eqs. (5) and (7). First, agents within the same partition share the same action and observation spaces, meaning that for any $i , j \in \Im _ { k } , A _ { i } = A _ { j }$ and $O _ { i } = O _ { j }$ , which is provided by Eq. (5). For each partition $\Im _ { k } .$ we introduce the CRVs $\# _ { \Im _ { k } } [ A _ { k } ]$ and $\# _ { \Im _ { k } } [ O _ { k } ]$ for actions and observations, whose ranges are all histograms over the local action space $\mathrm { r a n } ( A _ { k } )$ and observation space $\operatorname { r a n } ( O _ { k } )$ . To build the transition, sensor, and reward functions, based on Eq. (7), we replace $( A _ { i } ) _ { i \in \mathfrak { I } _ { k } }$ and $( O _ { i } ) _ { i \in \mathfrak { I } _ { k } }$ with CRVs $\# _ { \Im _ { k } } [ A _ { k } ]$ and $\# _ { \Im _ { k } } [ O _ { k } ]$ in the inputs of the transition, reward, and sensor function for each partition, with a histogram $h _ { k }$ mapping to $\rho$ for all permutations of a partition input $\mathbf { \delta } _ { \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } } \mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \~ \textit ~ { ~ a ~ } ~ } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } _ { \mathrm { \textmd ~ { ~ a ~ } } \mathbf { \alpha } \mathbf { \beta } \mathbf { \alpha } \mathbf { \alpha } }$ or $\scriptstyle o _ { k }$ mapping to $\rho$ for which h is the histogram representation.

To convert a counting DecPOMDP into a ground DecPOMDP fulfilling Eqs. (5) and (7), we essentially reverse the steps above: The agent set I is the union of the partitions $\textstyle \bigcup _ { k = 1 } ^ { K } \mathfrak { I } _ { k }$ . The action and observation sets are given by $\{ A _ { k , i } \} _ { i \in \Im _ { l } }$ and $\{ O _ { k , i } \} _ { i \in \mathfrak { I } _ { k } }$ , fulfilling Eq. (5). In the transition, reward, and sensor function, the CRV inputs are replaced by the action and observation sets, with a histogram $h$ mapping to $\rho$ being replaced by a set of mappings over all permutations of the values that agents can take according to the counts in $h ,$ all mapping to $\rho ,$ thereby leading to the functions fulfilling Eq. (7).

## B.2 Full Proof for Theorem 2

We repeat Thm. 2 and then provide its proof.

Theorem 2. A ground DecPOMDP M that meets Eqs. (5) and (7) is equivalent to a counting DecPOMDP M<sup>¯</sup> .

Proof. First, we convert a DecPOMDP $M = ( I , S , A , O , T , R , \Omega , \tau )$ , in which Eqs. (5) and (7) hold, into a counting DecPOMDP $\bar { M } = ( \bar { I } , \bar { S } , \bar { A } , \bar { O } , \bar { T } , \bar { R } , \Omega , \tau )$ Agents that share the same action and observation spaces, meaning $A _ { i } = A _ { j }$ and $O _ { i } = O _ { j }$ for $i , j \in I$ by Eq. (5), as well as exhibit symmetric behaviour as in $\operatorname { E q . } \left( 6 \right)$ are grouped into a partition, storing for each partition a decision variable $A _ { k } = A _ { i } = A _ { j }$ and an observation variable $O _ { k } = O _ { i } = O _ { j }$ . To construct the counted functions ${ \bar { T } } , { \bar { R } } ,$ , and $\bar { \varOmega }$ based on Eq. $( 7 )$ , we replace $( A _ { i } ) _ { i \in \mathfrak { I } _ { k } }$ with a CRV $\# _ { \Im _ { k } } [ A _ { k } ]$ in the inputs of the transition, reward, and sensor function for each partition. We do the same with the observation variables in the sensor function. For each function and each partition, we then construct from a mapping with a partition action $\mathbf { \em { a } } _ { k }$ (partition observation $o _ { k } )$ mapping to an outcome $\rho$ a new mapping with a histogram $h _ { k }$ replacing ${ \pmb a } _ { k } \ \left( { \pmb o } _ { k } \right)$ , counting how often each $a \in \mathrm { r a n } ( A _ { k } ) \ ( o \in \mathrm { r a n } ( O _ { k } ) )$ occurs in ${ \pmb a } _ { k } \ \left( { \pmb o } _ { k } \right)$ for the counts in $h _ { k }$ , and discard all mappings with permutations of ${ \pmb a } _ { k } \ \left( { \pmb o } _ { k } \right)$

Second, we convert a counting DecPOMDP into a DecPOMDP M that meets Eqs. (5) and (7) by setting $\begin{array} { r } { \pmb { I } = \bigcup _ { k } \Im _ { k } , A _ { i } = A _ { k } } \end{array}$ and $O _ { i } = O _ { k }$ for all $i \in \Im _ { k }$ in each partition $\Im _ { k } .$ , making M fulfil Eq. (5), and then essentially expanding the counted functions into their ground versions, i.e., replacing each CRV $\# _ { \Im _ { k } } [ V _ { k } ]$ $V \in \{ A , O \}$ , with a set of variables $V _ { k , 1 } , \ldots , V _ { k , n _ { k } }$ as inputs and mapping the diferent inputs v to an outcome $\rho ,$ whenever the histogram representation of v maps to $\rho ,$ which makes M meet Eq. (7).

## B.3 Full Proof of Theorem 3

We repeat Thm. 3 and then provide its proof.

Theorem 3. For a fixed number of partitions K, fixed action and observation spaces, and a fixed horizon $\tau _ { : }$ , the model complexity of a counting DecPOMDP $\bar { M _ { c } }$ is polynomial in $N$

Proof. The histogram space of a CRV for partition $\mathfrak { I } _ { k } , k \in \{ 1 , \dots , K \}$ of size $n _ { k }$ with m possible values has size $\binom { n _ { k } + m - 1 } { m - 1 } \leq n _ { k } ^ { m } \left[ 1 3 \right]$ . As we assume K is fixed and $K \ll N , n _ { k }$ and N have the same order of magnitude, i.e., $n _ { k } \leq N$ . As such, due to the range sizes of their inputs, the sizes of the functions T<sup>¯</sup>, R<sup>¯</sup> and Ω<sup>¯</sup> lie in $O ( s ^ { 2 } N ^ { K a } ) , \stackrel { \smile } { O } ( s N ^ { K a } )$ , and $\bar { O ( } s N ^ { K o } )$ , respectively, with $s = | \mathrm { r a n } ( \bar { S } ) |$ |, $a = \operatorname* { m a x } _ { k } \left| \operatorname { r a n } ( A _ { k } ) \right|$ |, and $o = \operatorname* { m a x } _ { k } \left| \operatorname { r a n } ( O _ { k } ) \right|$ , depending polynomially on N.

## B.4 Full Proof of Theorem 4

We repeat Thm. 4 and then provide its proof.

Theorem 4. The maximum expected utility for a ground DecPOMDP M that meets Eqs. (5) and (7) is equal to the maximum expected utility of the corresponding policy-counted DecPOMDP M<sup>¯</sup> . That is, $M E U ( M ) = M E U ( \bar { M } )$

Proof. Let $\varPi _ { M }$ be set of possible joint policies in the ground model M. In the policy-counted model M<sup>¯</sup> , we convert any ground joint policy $\pmb { \pi } = ( \pi _ { 1 } , . . . , \pi _ { N } )$ into a joint counted policy of histograms $\Breve { \pmb { h } } ^ { \tilde { I I } } = ( h _ { 1 } , . . . , h _ { K } )$ , where each $h _ { k }$ counts the number of agents in partition $\Im _ { k }$ assigned to each representative policy in $\varPi _ { k }$ . To establish value equivalence, consider any ground policy π and its corresponding policy histogram $h ^ { \varPi }$ and an empty joint (counted observation) at the start. Based on Thm. 2, the reward, transition, and observation functions return the same values between the two models given equivalent inputs. As a result, the immediate rewards in Eqs. (2) and (10) are identical, i.e., $R ( s , { \pmb a } ) = \bar { R } ( s , { \pmb h } ^ { a } )$ with the joint action in Eq. (2) and the joint counted action in Eq. (10) being equivalent, since the same policies are followed between the ground and the policy-counted version and thus, the actions in the roots are identical, adding up to the counts in the counted joint action. The following sum over states is identical in both equations with the transition functions returning the same values. The sum over joint (counted) observations is equivalent, since every ground joint observation can be translated into a joint counted observation, with the counted version combining those ground joint observations with the same counted representation into one summand and multiplying that summand with the number of instances using the multinomial coeficient [23]. The summand value is identical, since the sensor functions return the same value for equivalent inputs. As such, the calculations at the end of the first step are equivalent.

Subsequent utility calculations have as the observation history an equivalent input as just argued. Thus, with the same arguments as before, the further joint (counted) actions are equivalent, with the action histogram being a counted representation of the ground joint action, which also holds for further joint (counted) observations. Therefore, the expected utilities $U _ { M } ( \pmb { \pi } )$ and $U _ { \bar { M } } ( \bar { h } ^ { \varPi } )$ are identical. As this can be done for every ground policy with its corresponding policy histogram, the counted policy space $H ^ { \pi }$ and the ground policy space $\varPi _ { M }$ yield the same maximum expected utility, that is $M E U ( M ) = M E U ( \bar { M } )$

## B.5 Full Proof of Theorem 5

We repeat Thm. 5 and then provide its proof.

Theorem 5. For a fixed number of partitions K, fixed action and observation spaces, and a fixed horizon τ, the cost of evaluating a joint counted policy and the size of the overall joint counted policy space in a policy-counted DecPOMDP depend polynomially on the number of agents N.

Proof. For the cost of evaluating a joint counted policy, consider Eq. (10) with a sum over states, which is of size $s = | \mathrm { r a n } ( S ) |$ , and a sum over joint counted observations. With $a = \mathrm { m a x } _ { k } | \mathrm { r a n } ( A _ { k } ) | , o = \mathrm { m a x } _ { k } | r a n ( O _ { k } ) |$ , the space of joint counted observations is structured for each partition by the number of representative policies p and the number of representative observation histories $h ,$ which

are given by [16]

$$
p = | { \cal { I } } _ { k } | = a ^ { \frac { o ^ { \tau } - 1 } { o - 1 } }
$$

$$
h = | P _ { k } ^ { 0 : \tau , o } | = o ^ { \tau }
$$

Each representative history can have up to $n _ { k } \leq N$ agents following that particular history with $K \ll N$ , which are distributed onto o possible observations. Thus, the number of histograms per partition is given by

$$
H = \binom { N + o - 1 } { o - 1 } \leq N ^ { o }
$$

Therefore, the size of the joint counted observation space is given by the size of the cross product of K partitions, p representative policies, and h representative histories, i.e., $H ^ { K p h }$ , which is capped by

$$
H ^ { K p h } \le ( N ^ { o } ) ^ { K { a ^ { o } } ^ { T } o ^ { \tau } } = N ^ { K a o ^ { 2 \tau + 1 } }
$$

and as such, depends polynomially on N. The size of the joint counting policy space is capped by

$$
\prod _ { k = 1 } ^ { K } { \binom { n _ { k } + p - 1 } { p - 1 } } \leq N ^ { K p } = N ^ { K a ^ { \frac { o ^ { \tau } - 1 } { o - 1 } } }
$$

and thereby, also depends polynomially on $N$

## B.6 Full Proof of Theorem 6

We repeat Thm. 6 and then provide its proof.

Theorem 6. Using policy-counted dynamic programming on a policy-counted DecPOMDP M<sup>¯</sup> is equivalent to using standard dynamic programming on a ground DecPOMDP M, in which Eqs. (5) and (7) hold.

Proof. We show equivalence by considering the diferent steps of the dynamic programming operator.

Exhaustive backup: Given Eq. (5), the policies for each agent in a partition $\Im _ { k }$ are identical, meaning that the exhaustive backup can be performed using representative policies for each partition.

Value computation: Given that $M E U ( { \bar { M } } ) = M E U ( M )$ by Thm. 4 and as such $U _ { \bar { M } } ^ { h ^ { \varlimsup } } = U _ { M } ^ { \pi }$ for equivalent policies $\boldsymbol { h } ^ { \boldsymbol { \pi } }$ and π, $\bar { V } _ { k } ^ { \prime } = V _ { i } ^ { \prime }$ for all $i \in \Im _ { k }$ as the value computation is a rewriting of the utility computation, meaning that the value calculations can be performed for counted partition policies at each partition.

Pruning: Since $\bar { V } _ { k } ^ { \prime } = V _ { i } ^ { \prime }$ for all $i \in \Im _ { k }$ , Eq. (4) for pruning is identical for all $i \in \Im _ { k }$ given a representative policy $\pi _ { k } \in \varPi _ { k }$ , meaning that $\pi _ { k }$ can be checked for pruning once per partition and thus, pruning can be performed for representative policies per partition.

Given the equivalence of the three steps, we can conclude that the overall procedure yields an equivalent result.

## B.7 Full Proof of Theorem 7

We repeat Thm. 7 and then provide its proof.

Theorem 7. For a fixed number of partitions K, fixed action and observation spaces, and a fixed horizon τ , the runtime of policy-counted dynamic programming in a policy-counted DecPOMDP M<sup>¯</sup> depends polynomially on the number of agents N.

Proof. Again, we consider the three steps of the dynamic programming operator to show polynomial dependence on the number of agents N, which reuses definitions of Thm. 3.

Exhaustive backup: With $a = \mathrm { m a x } _ { k } \mathrm { | r a n } ( A _ { k } ) | , o = \mathrm { m a x } _ { k } \mathrm { | } r a n ( O _ { k } ) |$ , the number of policies to generate per partition is given by $p = a ^ { \frac { o ^ { \tau } - 1 } { o - 1 } } \left[ 1 6 \right]$ , meaning $K p$ over all partitions, which is independent of N.

Value calculation: Eq. (13) depends polynomially on N since the sum over joint counted observations depends polynomially on N. In contrast to the recursive calculation of $U _ { \bar { M } } ^ { h ^ { \varPi } }$ , we do not need to consider representative observation histories for $t > 0$ , meaning that the size of the space of histograms is capped by $( N ^ { o } ) ^ { K p } = N ^ { \frac { K o \dot { a } ( o ^ { \tau } - 1 ) } { o - 1 } }$ , which is polynomial in N.

Pruning: Solving the pruning part can be done using a linear programme, which can be solved in time polynomial w.r.t. the number of variables and constraints [25], whose numbers again depends polynomially on N, since the number of variables and constraints depends on the counted policy space $H ^ { \pi }$ , which depends polynomially on N.

Given that each step depends polynomially on N, the overall runtime depends polynomially on N.

## C DecTiger Example

We provide the diferent model definitions, look at model and worst-case policy space sizes, and consider the diferences when applying the dynamic programming operator.

## C.1 Model Definition

We use the specification of the DecTiger benchmark from the MADP tool box. <sup>1</sup>. Listing 1 shows the original DecTiger version in the MADP input format. The DecPOMDP model reads as follows:

$$
\begin{array} { r l } & { I = \{ a g e n t _ { 1 } , a g e n t _ { 2 } \} , } \\ & { S , \mathrm { r a n } ( S ) = \{ t i g e r - l e f t , t i g e r - r i g h t \} = \{ t l , t r \} , } \\ & { A = \{ A _ { i } \} _ { i \in I } , \forall i \in I : \mathrm { r a n } ( A _ { i } ) = \{ l i s t e n , o p e n - l e f t , o p e n - r i g h t \} = \{ l i , o l , o r \} , } \\ & { \mathrm { a n d } } \end{array}
$$

$$
- \mathbf { \delta } O = \{ O _ { i } \} _ { i \in I } , \forall i \in I : \operatorname { r a n } ( O _ { i } ) = \{ h e a r \ – { l e f t } , \ h e a r \ – { i g h t } \} = \{ h l , h r \} .
$$

with T, R, and $\varOmega \ ( = \mathrm { ~ O ~ }$ in Listing 1) in tabular notation in full below (lines are reordered compared to the listing to match the order of the inputs in the definitions).
<table><tr><td rowspan=1 colspan=1>S A1 A2 Ttl $T _ { t r }$ </td></tr><tr><td rowspan=1 colspan=1>tl li li 1 0</td></tr><tr><td rowspan=1 colspan=1>tl li ol 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tl li or 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tl ol li 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tl ol ol 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tl ol or 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tl or li 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tl or ol 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tl or or 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tr li li 0 1</td></tr><tr><td rowspan=1 colspan=1>tr li ol 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tr li or 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tr ol li 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tr ol ol 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tr ol or 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tr or li 0.5 0.5tr or ol 0.5 0.5</td></tr><tr><td rowspan=1 colspan=1>tr or or 0.5 0.5</td></tr></table>

<table><tr><td>SA1</td><td> $A _ { 2 }$ </td><td>R</td></tr><tr><td></td><td>tl li li</td><td>-2</td></tr><tr><td>tl li</td><td></td><td>i ol −101</td></tr><tr><td></td><td>tl li or</td><td>9</td></tr><tr><td></td><td></td><td>tl ol li −101</td></tr><tr><td></td><td></td><td>tl ol ol −50</td></tr><tr><td></td><td>tl ol or −100</td><td></td></tr><tr><td></td><td>tl or li</td><td>9</td></tr><tr><td></td><td>tl or ol −100</td><td></td></tr><tr><td></td><td>tl or or</td><td>20</td></tr><tr><td></td><td>tr li li</td><td>-2</td></tr><tr><td></td><td>tr li ol</td><td>9</td></tr><tr><td></td><td>tr li or −101</td><td></td></tr><tr><td></td><td>tr ol li</td><td>9</td></tr><tr><td></td><td>tr ol ol</td><td>20</td></tr><tr><td></td><td>tr ol or −100</td><td></td></tr><tr><td></td><td></td><td>tr or li −101</td></tr><tr><td></td><td></td><td>tr or ol −100</td></tr><tr><td></td><td></td><td>tr or or −50</td></tr></table>

<table><tr><td rowspan=1 colspan=4>SA1 A2 $\varOmega _ { h l , h l }$   $\varOmega _ { h r , h l }$   $\varOmega _ { h l , h r }$  $\Omega _ { h r , h r }$ </td></tr><tr><td rowspan=1 colspan=4>tl lili0.7225 0.1275 0.1275 0.0225tl liol 0.25 0.25 0.25  0.25</td></tr><tr><td rowspan=1 colspan=4>tl lior 0.25  0.25  0.25  0.25</td></tr><tr><td rowspan=1 colspan=4>tl olli 0.25 0.25 0.25 0.25</td></tr><tr><td rowspan=1 colspan=4>tl olol 0.25  0.25 0.25 0.25</td></tr><tr><td rowspan=1 colspan=4>tl olor 0.25  0.25 0.25 0.25</td></tr><tr><td rowspan=1 colspan=4>tl or li 0.25  0.25  0.25  0.25</td></tr><tr><td rowspan=1 colspan=4>tl or ol 0.25 0.25 0.25 0.25</td></tr><tr><td rowspan=1 colspan=3>tl or or 0.25 0.25  0.25</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=1 colspan=3>tr lili 0.7225 0.1275 0.1275</td><td rowspan=1 colspan=1>0.0225</td></tr><tr><td rowspan=1 colspan=1>tr liol 0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=1 colspan=1>tr lior 0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=1 colspan=1>tr olli 0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=1 colspan=1>tr olol 0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=1 colspan=2>tr ol or 0.25  0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=1 colspan=2>tr or li 0.25  0.25</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=1 colspan=3>tr or ol 0.25  0.25  0.25</td><td rowspan=1 colspan=1>0.25</td></tr><tr><td rowspan=1 colspan=4>tr or or 0.25  0.25 0.25 0.25</td></tr></table>

with $T _ { s ^ { \prime } } = T ( s ^ { \prime } , s , a _ { 1 } , a _ { 2 } ) , s ^ { \prime } \in \{ t l , t r \}$ and $\mathcal { Q } _ { o _ { 1 } , o _ { 2 } } = \mathcal { Q } ( ( o _ { 1 } , o _ { 2 } ) , s , a _ { 1 } , a _ { 2 } ) , o _ { 1 } , o _ { 2 } \in$ {hl, hr}. The transition function only states that as long as both agents only listen, the state does not change (identity). When at least one agent opens a door, the game basically restarts with the new state being set according to a uniform distribution. It is basically a way of keeping the game infinite by resetting the state to an arbitrary one whenever the agents end the game by opening a door (to either lose—tiger, or win—gold). One could argue that opening a door only ends the game and might not necessarily imply a restart. In that case, one would keeping the state as is in all cases ((1, 0) distribution for all tl lines, (0, 1) distribution for all tr lines) and re-spawn the game with an arbitrary starting state, sampled from a (0.5, 0.5) distribution, for do-overs.

The DecTiger model has the same action and observation sets for both agents and exhibits a counting symmetry. Thus, it can be rewritten into a counting model (Def. 3) with K = 1. We index the one partition with c.

$$
- \ \bar { A } _ { c } \ = \ \{ \# z _ { c } [ A _ { c } ] \}
$$

$$
\operatorname { r a n } ( A _ { c } ) = \operatorname { r a n } ( A _ { i } ) =
$$

$$
- ~ { \bar { T } } ( S ^ { \prime } , S , { \bar { A } } _ { c } ) = P ( S ^ { \prime } \mid S , { \bar { A } } _ { c } ) ,
$$

$$
\begin{array} { r l } & { - \ \bar { R } ( S , \bar { A } _ { c } ) , } \\ & { - \ \bar { O } _ { c } = \{ \# { \mathfrak { s } } _ { k } [ O _ { k } ] \} , \ \mathrm { w h e r e ~ t h e ~ l o c a l ~ o b s e r v a t i o n ~ r a n g e ~ i s ~ r a n } ( O ) = \{ h l , h r \} . } \\ & { - \ \bar { \Omega } ( \bar { O } _ { c } , S ) = P ( \bar { O } _ { c } \mid S ) } \end{array}
$$

with T<sup>¯</sup>, ${ \bar { R } } ,$ and Ω<sup>¯</sup> following after the specification of the policy-counted DecPOOMDP version.

In a policy-counted DecPOMDP (Def. 5), the action and observation variables are plain variables again. The CRVs only occur within the transition, reward, and sensor functions. Specifically, for the policy-counted DecTiger example, the model is defined as follows:

$$
\begin{array} { r l } & { - \ \bar { I } = \Im _ { c } = \{ a g e n t _ { 1 } , a g e n t _ { 2 } \} , } \\ & { - \ S = \{ t l , t r \} , } \\ & { - \ \bar { A } = \{ A _ { c } \} , \ \mathrm { r a n } ( A _ { c } ) = \mathrm { r a n } ( A _ { i } ) = \{ l i , o l , o r \} , } \\ & { - \ \bar { T } ( S ^ { \prime } , S , \bar { A } _ { c } ) = P ( S ^ { \prime } \mid S , \bar { A } _ { c } ) , } \\ & { - \ \bar { R } ( S , \bar { A } _ { c } ) , \ } \\ & { - \ \bar { O } = \{ O _ { k } \} _ { k = 1 } ^ { K } , \ \mathrm { r a n } ( O _ { c } ) = \{ h l , h r \} . } \\ & { - \ \bar { O } ( \bar { O } _ { c } , S ) = P ( \bar { O } _ { c } \mid S ) } \end{array}
$$

with $\bar { A } _ { c }$ and $\bar { O } _ { c }$ defined as for the counting DecPOMDP above.

The functions T<sup>¯</sup>, R<sup>¯</sup>, and Ω<sup>¯</sup> are given below in tabular notation with $\# A$ short for $\# _ { \Im _ { k } } [ A _ { k } ]$ (histogram positions: $[ l i , o l , o r ] )$ and ${ \bar { \varOmega } } _ { h }$ for Ω(S, #A, $\psi _ { \mathfrak { I } _ { k } } [ O _ { k } ] = h )$
<table><tr><td>S</td><td>#A</td><td> $\bar { T } _ { t l } ~ \bar { T } _ { t r }$ </td><td>S</td><td>#A</td><td>R</td><td></td><td>S #A</td><td></td><td> $\bar { \varOmega } _ { [ 2 , 0 ] }$ </td><td> $\bar { \varOmega } _ { [ 1 , 1 ] }$ </td><td> $\bar { \varOmega } _ { [ 0 , 2 ] }$ </td></tr><tr><td></td><td></td><td>tl [2, 0, 0] 1.0 0.0</td><td></td><td>tl [2,0,0]</td><td>-2</td><td></td><td></td><td></td><td>tl [2, 0, 0] 0.7225 0.1275 0.0225</td><td></td><td></td></tr><tr><td></td><td></td><td>tl [1, 1, 0] 0.5 0.5</td><td></td><td>tl [1, 1,0] −101</td><td></td><td></td><td></td><td>tl [1, 1, 0] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td></td><td></td><td>tl [1, 0, 1] 0.5 0.5</td><td></td><td>tl [0,2,0]</td><td>-50</td><td></td><td></td><td>tl [0, 2,0] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td></td><td></td><td>tl [0, 1, 1] 0.5 0.5</td><td></td><td>tl [0, 1, 1] −100</td><td></td><td></td><td></td><td>tl [0, 1, 1] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td></td><td></td><td>tl [0, 2, 0] 0.5 0.5</td><td></td><td>tl [0,0, 2]</td><td>20</td><td></td><td></td><td>tl [0, 0, 2] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td></td><td></td><td>tl [0, 0, 2] 0.5 0.5</td><td></td><td>tl [1, 0, 1]</td><td>9</td><td></td><td></td><td>tl [1, 0, 1] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td></td><td></td><td>tr [2, 0, 0] 0.0 1.0</td><td></td><td>tr [2,0, 0]</td><td>-2</td><td></td><td></td><td></td><td></td><td></td><td>tr [2, 0, 0] 0.7225 0.1275 0.0225</td></tr><tr><td></td><td></td><td>tr [1, 1, 0] 0.5 0.5</td><td></td><td>tr [1, 1, 0]</td><td>9</td><td></td><td></td><td>tr [1, 1, 0] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td></td><td></td><td>tr [1, 0, 1] 0.5 0.5</td><td></td><td>tr [0, 2, 0]</td><td>20</td><td></td><td></td><td>tr [1, 0, 1] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td></td><td></td><td>tr [0, 1, 1] 0.5 0.5</td><td></td><td>tr [0, 1, 1] −100</td><td></td><td></td><td></td><td>tr [0, 2, 0] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td></td><td></td><td>tr [0, 2, 0] 0.5 0.5</td><td></td><td>tr [0, 0, 2] −50</td><td></td><td></td><td></td><td>tr [0, 1, 1] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr><tr><td></td><td></td><td>tr [0, 0, 2] 0.5 0.5</td><td></td><td>tr [1, 0, 1] −101</td><td></td><td></td><td></td><td>tr [0, 0, 2] 0.25</td><td></td><td>0.25</td><td>0.25</td></tr></table>

Note that the rows to not add up to one because the histogram [1, 1] in $\bar { \varOmega }$ stands for two joint observations, $( h l , h r )$ and $( h r , h l )$ , meaning that the probability for $\bar { \varOmega } _ { [ 1 , 1 ] }$ has to be counted twice for the probability distribution to add up to 1. In general, a multinomial coeficient provides the number of inputs represented by a histogram, i.e., $n _ { k } ! / \prod _ { l } n _ { l } !$

Before moving on to model and policy space sizes, we list the DecTiger specification in the MADP toolbox.

Listing 1 DecTiger specification in the MADP toolbox (without the comments from the source)

agents : 2   
d i s c o u n t : 1   
v a l u e s : reward   
s t a t e s : t i g e r − l e f t t i g e r −r i g h t   
s t a r t : un ifo rm   
a c t i o n s :   
l i s t e n open− l e f t open−r i g h t   
l i s t e n open−l e f t open−r i g h t   
o b s e r v a t i o n s :   
hear−l e f t hear−r i g h t   
hear− l e f t hear−r i g h t   
# T r a n s i t i o n p r o b a b i l i t i e s   
T: ∗ : uniform   
T : l i s t e n l i s t e n : i d e n t i t y   
# Observation p r o b a b i l i t i e s : <2 a c t i o n s > : <s t a t e > : <2   
o b s e r v a t i o n s > : p r o b a b i l i t y   
O: ∗ : uniform   
O: l i s t e n l i s t e n : t i g e r − l e f t : hear− l e f t hear− l e f t : 0 . 7 2 2 5   
O: l i s t e n l i s t e n : t i g e r − l e f t : hear− l e f t hear−r i g h t : 0 . 1 2 7 5   
O: l i s t e n l i s t e n : t i g e r − l e f t : hear−r i g h t hear− l e f t : 0 . 1 2 7 5   
O: l i s t e n l i s t e n : t i g e r − l e f t : hear−r i g h t hear−r i g h t : 0 . 0 2 2 5   
O: l i s t e n l i s t e n : t i g e r −r i g h t : hear− l e f t hear− l e f t : 0 . 7 2 2 5   
O: l i s t e n l i s t e n : t i g e r −r i g h t : hear− l e f t hear−r i g h t : 0 . 1 2 7 5   
O: l i s t e n l i s t e n : t i g e r −r i g h t : hear−r i g h t hear− l e f t : 0 . 1 2 7 5   
O: l i s t e n l i s t e n : t i g e r −r i g h t : hear−r i g h t hear−r i g h t :   
0.0225   
# Rewards : <2 a c t i o n s > : <s t a t e > : ∗ : ∗ : reward   
R: l i s t e n l i s t e n : ∗ : ∗ : ∗ : −2   
R: open−l e f t open−l e f t : t i g e r −l e f t : ∗ : ∗ : −50   
R: open−r i g h t open−r i g h t : t i g e r −r i g h t : ∗ : ∗ : −50   
R: open−l e f t open−l e f t : t i g e r −r i g h t : ∗ : ∗ : 20   
R : open−r i g h t open−r i g h t : t i g e r − l e f t : ∗ : ∗ : 20   
R : open− l e f t open−r i g h t : t i g e r − l e f t : ∗ : ∗ : −100   
R : open− l e f t open−r i g h t : t i g e r −r i g h t : ∗ : ∗ : −100   
R: open−r i g h t open−l e f t : t i g e r −l e f t : ∗ : ∗ : −100   
R: open−r i g h t open−l e f t : t i g e r −r i g h t : ∗ : ∗ : −100   
R: open−l e f t l i s t e n : t i g e r −l e f t : ∗ : ∗ : −101   
R: l i s t e n open−r i g h t : t i g e r −r i g h t : ∗ : ∗ : −101   
R: l i s t e n open−l e f t : t i g e r −l e f t : ∗ : ∗ : −101   
R: open−r i g h t l i s t e n : t i g e r −r i g h t : ∗ : ∗ : −101   
R : l i s t e n open−r i g h t : t i g e r − l e f t : ∗ : ∗ : 9   
R : l i s t e n open− l e f t : t i g e r −r i g h t : ∗ : ∗ : 9   
R: open−r i g h t l i s t e n : t i g e r − l e f t : ∗ : ∗ : 9   
R: open− l e f t l i s t e n : t i g e r −r i g h t : ∗ : ∗ : 9

## C.2 Model and Policy Space Sizes

The following table characterises the ground, counting, and policy-counted model $M , \bar { M } _ { c } .$ , M<sup>¯</sup> regarding diferent parameters:

– Number of agents: N, number of partitions: K, horizon τ

– State size: s

– Size of a local / partition action: a, that is $| \mathrm { r a n } ( A _ { i } ) | , | \mathrm { r a n } ( \# _ { \Im } [ A _ { c } ] ) | , | \mathrm { r a n } ( A _ { c } ) |$ respectively

– Size of the joint (counted) action: a

– Size of a local / partition observation: o, that is $\vert \mathrm { r a n } ( O _ { i } ) \vert , \ \vert \mathrm { r a n } ( \# \mathfrak { z } [ O _ { c } ] ) \vert$ $\left| \operatorname { r a n } ( O _ { c } ) \right|$ , respectively

– Size of the joint (counted) action: o

– Size of local / partition policy space per state: p, i.e., local policy $\pi$ over $A _ { i } , O _ { i }$ , partition policy over $\# _ { \Im } [ A _ { c } ] ) , \# _ { \Im } [ O _ { c } ] )$ , counted policy over representative policies over $A _ { k } , O _ { k }$

– Size of the joint (counted) policy space per state: p

– Size of the transition, reward, and sensor functions by range sizes of the inputs: T, R, Ω

<table><tr><td></td><td>M</td><td> $\bar { M } _ { c }$ </td><td> $\bar { M }$ </td></tr><tr><td>S</td><td>2</td><td>2</td><td>2</td></tr><tr><td>a</td><td>3</td><td> $\binom { N + 3 - 1 } { 3 - 1 } = \binom { N + 2 } { 2 }$ </td><td>3</td></tr><tr><td>a</td><td> $3 ^ { N }$ </td><td> $\binom { N + 2 } { 2 } ^ { K }$ </td><td> $\binom { N + 2 } { 2 } ^ { K }$ </td></tr><tr><td>0</td><td>2</td><td> $\binom { N + 2 - 1 } { 2 - 1 } = \binom { N + 1 } { 1 }$ </td><td>2</td></tr><tr><td>0</td><td> $2 ^ { N }$ </td><td> $\binom { N + 1 } { 1 } ^ { K }$ </td><td> $\binom { N + 1 } { 1 } ^ { K }$ </td></tr><tr><td></td><td> $\textit { p } \ : \ : 3 ^ { \frac { 2 ^ { \tau } - 1 } { 2 - 1 } }$ </td><td> $\binom { N + 2 } { 2 } \frac { \binom { N + 1 } { 1 } ^ { \tau } - 1 } { \binom { N + 1 } { 1 } - 1 }$ </td><td> $\left( { \begin{array} { c } { N + 3 ^ { \frac { 2 ^ { 7 } } { 2 - 1 } } - 1 } \\ { 3 ^ { \frac { 2 ^ { 7 } } { 2 - 1 } } - 1 } \end{array} } \right)$ </td></tr><tr><td></td><td></td><td> $\textbf {  { p } } 3 ^ { N \frac { 2 ^ { \tau } - 1 } { 2 - 1 } } \binom { N + 2 } { 2 } ^ { K \frac { \binom { N + 1 } { 1 } ^ { \tau } - 1 } { \binom { N + 1 } { 1 } - 1 } }$ </td><td> $\left( { \cal N } + 3 ^ { \frac { 2 ^ { \tau } } { 2 - 1 } } - 1 \right) ^ { k }$ </td></tr><tr><td></td><td> $T ~ 2 \cdot 2 \cdot 3 ^ { N }$ </td><td> $2 \cdot 2 \cdot { \binom { N + 2 } { 2 } } ^ { K }$ </td><td> $2 \cdot 2 \cdot { \binom { N + 2 } { 2 } } ^ { K }$ </td></tr><tr><td></td><td> $\textit { R }  { 2 } \cdot 3 ^ { N }$ </td><td> $2 \cdot \binom { N + 2 } { 2 } ^ { K }$ </td><td> $2 \cdot \binom { N + 2 } { 2 } ^ { K }$ </td></tr><tr><td>Ω</td><td> $2 \cdot 2 ^ { N }$ </td><td> $2 \cdot \binom { N + 1 } { 1 } ^ { K }$ </td><td> $2 \cdot \binom { N + 1 } { 1 } ^ { K }$ </td></tr></table>

## C.3 Dynamic Programming

We consider the three steps of dynamic programming (backup, value calculation, pruning) for the levels t = 0 and t = 1 for the three versions of the DecTiger example.

## Level t = 0

Backup: The operator for the ground version builds three policies for each of its two agents, each policy having a single (root) node for one of the actions li, ol, or, building six policies overall. A counting programming operator for the counting version would build six policies for its one partition, each with a single (root) node for one of the action histograms [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 2, 0], [0, 0, 2]. The counted programming operator for the policy-counted version builds three representative policies for its one partition, each with a single (root) node for one of the actions li, ol, or. With two available agents, there are six counted policies over the three representative ones: [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 2, 0], [0, 0, 2].

Value Computation: For the ground version, the operator calculates the values for each of its two agents for each of its three policies. For each policy, the computation has to be performed for each state tl, tr and each of the other agent’s three policies. For the counting version, the operator calculates the values for its one partition for each of its six policies. For each policy, the computation has to be performed for each state tl, tr. For the policy-counted version, the operator calculates the values for its one partition for each of its six counted policies. For each policy, the computation has to be performed for each state tl, tr.

Pruning: In the ground version, the operator checks for each of the two agents, whether one of its three policies is (weakly) dominated by one of the other two policies for each state tl, tr and each of the three policies of the other agent (does not need to be the same policy over all states and policies of the other agent), until there are no more policies to prune. In the counting version, the operator would check for its one partition, whether any of its six policies is (weakly) dominated by one of the other five policies for each state tl, tr, until there are no more policies to prune. In the policy-counted version, the operator checks for its one partition, whether any of its three representative policies is dominated by one of the other two policies for each state tl, tr and each of the other counted policies over the remaining partition, which contains only one agent, i.e., [1, 0, 0], [0, 1, 0], [0, 0, 1] (that is just the three representative policies), until there are no more policies to prune.

Comment: The diference between the counting and policy-counted versions is not as pronounced at this level, with the ground version also working within the same order of magnitude.

## Level t = 1

Backup: With no policies pruned, the operator for the ground version builds for its two agent each 27 policies out of its previous three policies and two possible observations hl, hr, with each policy having a root node and two children, one for each observation. The actions in the nodes are each possible combination of li, ol, or. With no policies pruned, the operator for the counting version would build for its one partition 216 policies out of its previous six policies and three possible counted observations [0, 2], [1, 1], [2, 0]. The policies would have a root node and three children, one for each possible histogram, with the nodes containing each possible combination of the six action histograms. The operator for the policy-counted version builds the same 27 (representative) policies out of its previous three representative policies and two possible observations hl, hr that the ground version has. With two available agents, there are then 378 counted policies.

Value Computation: For the ground version, the operator performs the same calculations outlined for level t = 0 for its 27 policies for each combination of state tl, tr and policy of the other agent (27 as well) for each agent. For the counting version, the operator calculates the values for its one partition for each of its 216 policies for each state tl, tr. For the policy-counted version, the operator calculates the values for its one partition for each of the 378 counted policies. For each policy, the computation has to be performed for each state tl, tr.

Pruning: Pruning again follows the same procedure as outlined above, for two agents and each of their 27 policies in each state and for each of the other agent’s policies in the ground version. In the counting version, the 1296 policies of its one partition get checked for each state tl, tr. In the policy-counted version, the 27 representative policies of its one partition are checked for each state and each counted policy of the remaining partition, i.e., the 27 representative policies.

Comment: This numerical example illustrates impressively that policy counting only pays of if the number of agents is large or at least outweighs the number of representative policies. That is the number of agents should at least be an order larger than the number of representative actions and observations available.
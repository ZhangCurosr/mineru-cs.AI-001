# ★ ExRole: From Team Trajectories to Executable Roles in Multi-Agent Language Models

Zhou Liu<sup>1</sup>, Chaoyang Han<sup>1</sup>, Zewei Pan<sup>2</sup>, Zeli Su<sup>1</sup>, Wentao Zhang<sup>1∗</sup>

<sup>1</sup> Peking University <sup>2</sup> Shanghai Jiao Tong University

## Abstract

Roles provide an interpretable interface for organizing language-model agents, yet most multi-agent systems treat them as hand-written prompt labels disconnected from learned behavior and parameter updates. We argue that a useful role should instead be an executable control variable: it should summarize behavior predictive of future utility, guide subsequent interaction, and identify the trainable capacity responsible for that behavior. We introduce ExRole, a trajectoryto-role framework that learns future-aware role prototypes from prefix-local team traces, resolves them into readable instructions and token-aligned role markers, and optionally routes shared LoRA rank slots with turn-aligned credit. Across MuSiQue and 2WikiMultiHopQA, ExRole improves over single-agent search by 15.0/14.4 and 13.5/16.1 EM/F1 points, respectively. Against the strongest non-ExRole controls, the corresponding gains remain 11.5/11.6 and 7.7/9.7 points. Across both benchmarks, the controlled results consistently favor trajectory-induced role conditioning over role-free, manual, random, and shufled alternatives. Role-Agent-Turn interventions further show that the induced roles capture transferable behavioral specialization beyond fixed agent identities or turn positions.

## Introduction

Large language model (LLM) agents increasingly solve knowledge-intensive tasks by interleaving reasoning with retrieval (Trivedi et al. 2023; Jin et al. 2025); recent multi-agent search work further distributes planning, retrieval, and synthesis across specialized agents (Chen et al. 2026). Roles ofer a simple and interpretable abstraction for organizing this collaboration: they expose the intended division of labor without requiring a separate model for every agent. Roleplaying designs commonly instantiate these functions as natural-language profiles fixed in prompts before execution, as in CAMEL’s inception prompting (Li et al. 2023). Figure 1 illustrates that changing such labels alone does not reliably improve answer quality or evidence use under a fixed policy. These labels are neither grounded in observed team behavior nor tied to the learning signal for role turns.

We ask: how can a system induce roles from prior collaboration and reuse them to coordinate new episodes and assign policy credit?

Learning roles from collaboration trajectories is more demanding than clustering surface-level action counts. First, a role representation must be inferred from information available at the current prefix while still capturing how that behavior afects future evidence and return; otherwise it either ignores delayed utility or leaks sufix information. Second, an induced cluster must become executable in a new episode: the same identity should guide the agent’s natural-language instruction and remain recoverable inside the model. Third, shared team rewards create the familiar credit-assignment problem of identifying which action produced the outcome (Foerster et al. 2018); here the credit must further reach the role turn and capacity path that produced a useful textual action. Recent role-decomposed and segment-typed agentic RL methods likewise show why uniform outcome credit can obscure useful intermediate contributions (Park, Cho, and Lee 2026; Xu et al. 2026). Without these links, role induction, role prompting, and policy optimization remain separate mechanisms rather than a learned specialization process.

![](images/84f297b2e290ca304898c352195f0bafee1475f9dbb0194ace1ddb7b2a39bfc8.jpg)  
(a) Answer quality

![](images/25eec07d80a052624fdf22461ec6c828ac48f4848ad356bc04cea3f5f3bd5c18.jpg)  
(b) Evidence-use behavior  
Figure 1: Role-prompt interventions under a fixed MuSiQue policy. Points show paired changes from no role over the same 200 questions; bars denote 95% paired-bootstrap confidence intervals.

Our key insight is to treat a role as a shared control variable that connects behavioral abstraction with policy execution. We instantiate this idea in ExRole, an executable role-learning framework illustrated in Figure 2. ExRole first encodes prefix-local action, timing, interaction, and evidence statistics, and uses next-action, future-evidence, and finalreturn targets to organize trajectories by their prospective utility. It then summarizes each cluster as a reusable prototype and deterministically resolves the prototype into a readable instruction and a token-aligned role marker, without requiring an additional role-labeling language model.

The resolved identity controls two complementary paths. At the interaction level, the instruction conditions the active agent and the marker remains aligned with the generated roleturn tokens. At the parameter level, ExRole-Routed combines that marker with the current semantic prefix to select a balanced sparse-delta gate over shared LoRA rank slots. GRPO is augmented with turn-aligned credit so that a useful role turn reinforces both its response tokens and, when routing is enabled, its selected capacity. ExRole-Shared retains the same induced roles and turn-aligned policy credit but uses a uniform shared LoRA path, providing a matched control that isolates the contribution of role-conditioned sparse LoRA routing.

We evaluate ExRole on MuSiQue and 2WikiMultiHopQA, using single-agent search, no-role and manual-role teams, random role prompts, and shufled induced roles to separate trajectory-derived specialization from generic prompting efects. On MuSiQue, ExRole-Shared reaches 31.5 EM and 43.2 F1, improving single-agent search by 15.0 and 14.4 points and the strongest non-ExRole control by 11.5 and 11.6 points. On 2WikiMultiHopQA, ExRole-Routed reaches 50.0 EM and 59.7 F1, gains of 13.5 and 16.1 points over singleagent search and 7.7 and 9.7 points over the strongest control. Across both benchmarks, the controlled comparisons favor trajectory-induced roles over role-free, manual, random, and shufled alternatives. Role-Agent-Turn interventions further show that the induced roles encode transferable action tendencies beyond fixed agent identities and turn positions.

To sum up, our contributions are two-fold:

• We formulate LLM-agent role learning as trajectoryconditioned role induction, in which prefix-local behavior and future-utility targets define reusable role prototypes.

• We make induced roles executable by binding each prototype to a readable instruction, a token-aligned role marker, and a deterministic agent assignment, and connect that identity to role-conditioned sparse LoRA routing with turn-aligned credit.

## Related Work

## Role Discovery in Multi-Agent Reinforcement Learning

Role-based multi-agent reinforcement learning (MARL) learns reusable abstractions through identifiable embeddings (ROMA), role-specific action spaces (RODE), trajectory-encoded abilities (LDSA), contrastive representations (ACORM), and future behavioral efects (R3DM), while retaining parameter sharing (Wang et al. 2020, 2021; Yang et al. 2022; Hu et al. 2024; Goel et al. 2025). These methods primarily assume compact states and actions; Ex-Role extends role induction to language-agent trajectories, where a role must organize messages, retrieved evidence, delayed answers, and trainable capacity.

## Roles and Credit Assignment in LLM Multi-Agent Systems

LLM teams commonly begin with prompt-defined roles, as in CAMEL, while MLC, MasRouter, and ReSo learn role diferentiation, allocation, or agent selection (Li et al. 2023, 2025; Yue et al. 2025; Zhou et al. 2025). MARFT, MAGRPO, MHGPO, and MATPO instead optimize team policies within a specified organization (Liao et al. 2025; Liu et al. 2026; Chen et al. 2026; Mo et al. 2025). Retrieval-oriented reasoning adds delayed credit: IRCoT interleaves retrieval and reasoning, whereas Search-R1 and R1-Searcher optimize single search policies (Trivedi et al. 2023; Jin et al. 2025; Song et al. 2025). COMA addresses team credit in MARL, and DAC and TRIAGE provide cross-agent or segment-level learning signals (Foerster et al. 2018; Park, Cho, and Lee 2026; Xu et al. 2026). ExRole difers by inducing recurring functions from prior trajectories and reusing the same identity across instructions, role-turn credit, and sparse LoRA routing.

## Parameter-Eficient Specialization and Routing

Shared-backbone systems preserve specialization through selective parameter use: Kaleidoscope learns agent-specific masks and ADMN routes agents through shared modules (Li, Pan, and Zhang 2024; Yu et al. 2024). LoraHub, LoraRetriever, X-LoRA, and MoLE compose or route LoRA adapters from task or input signals (Huang et al. 2024; Zhao et al. 2024; Buehler and Buehler 2024; Wu, Huang, and Wei 2024). ExRole instead derives routing identity from collaboration trajectories and modulates rank slots inside a jointly trained shared LoRA parameterization.

## Problem Formulation

We study role learning for multi-agent language-model teams in evidence-seeking tasks. Each instance is a pair $( x , y )$ where x is the input question and y is a gold answer. A team of N agents interacts with a retrieval environment for at most $T$ turns, producing

$$
\tau = \left( x , \left\{ i _ { t } , o _ { t } , m _ { t } , a _ { t } , e _ { t } , r _ { t } \right\} _ { t = 1 } ^ { T _ { \tau } } , \hat { y } \right) , \qquad T _ { \tau } \leq T .\tag{1}
$$

Here $i _ { t }$ is the active agent, $o _ { t }$ its observation, $m _ { t }$ an optional teammate message, $a _ { t }$ a search, answer, or invalid action, $e _ { t }$ retrieved evidence, $r _ { t }$ the turn reward, and yˆ the final answer. A team turn is one active-agent response; a role-turn segment is the token span aligned to its role marker. Let $\mathcal { H } _ { t }$ denote the interaction history available before turn t. Agent i is assigned an executable role $z _ { i } ,$ , and the active policy is conditioned on both that role and its adapter-routing state $\Gamma _ { t } \colon$

$$
\left( m _ { t } , a _ { t } \right) \sim \pi _ { \theta , \theta _ { b } } \left( \cdot \mid x , \mathcal { H } _ { t } , z _ { i _ { t } } , \Gamma _ { t } \right) .\tag{2}
$$

ExRole induces a source role library $\mathcal { P } = \{ p _ { k } \} _ { k = 0 } ^ { K - 1 }$ from prior trajectories and resolves it into N executable roles. The policy parameters θ and, when enabled, router parameters $\theta _ { b }$ maximize the expected episode reward

![](images/072895f74ec9a321523b63f7a5ca5772b0aa57a1be213eb317f8ef9de36be693.jpg)  
Figure 2: Overview of ExRole. Prior team trajectories induce a future-aware role library. The same role identity conditions prompt-level coordination and role-conditioned sparse LoRA routing, while turn-aligned GRPO updates the policy and router.

$$
\begin{array} { r l } & { R ( \tau , y ) = \displaystyle \sum _ { t = 1 } ^ { T _ { \tau } } r _ { t } , } \\ & { ( \theta ^ { \star } , \theta _ { b } ^ { \star } ) = \arg \underset { \theta , \theta _ { b } } { \operatorname* { m a x } } \mathbf { E } _ { ( x , y ) } \mathbf { E } _ { \tau \sim \pi _ { \theta , \theta _ { b } } ( \cdot \vert x , \mathcal { P } _ { \mathrm { e x e c } } ) } [ R ( \tau , y ) ] . } \end{array}\tag{3}
$$

The complete action grammar, reward decomposition, and answer normalizer are provided in the appendix.

## Method

ExRole addresses the gap between discovering recurring team behaviors and turning them into specialization that can be executed and optimized in later episodes. Given prior collaboration trajectories, our goal is to induce roles that capture future-relevant behavior, bind each role to a consistent interaction and model-side identity, and assign learning credit to the role and capacity path that produced an action. Figure 2 summarizes the resulting pipeline: predictive trajectory encoding and clustering produce a role library; deterministic role binding makes each prototype executable; roleconditioned interaction and sparse LoRA routing apply the same identity during generation; and turn-aligned grouped reinforcement learning updates the policy and, when enabled, the router. The following sections describe these four components in order.

## Future-Aware Role Induction

A useful role should summarize not only what an agent has done, but also what its behavior predicts about subsequent collaboration. For each logged agent turn, ExRole constructs a prefix-local behavior vector

$$
\begin{array} { r l } & { \phi _ { t } ^ { \mathrm { o b s } } = \left[ \phi _ { t } ^ { \mathrm { a c t } } , \phi _ { t } ^ { \mathrm { p o s } } , \phi _ { t } ^ { \mathrm { t e a m } } , \phi _ { t } ^ { \mathrm { e v d } } , \phi _ { t } ^ { \mathrm { a n s } } \right] \in \mathbf { R } ^ { 3 0 } , } \\ & { \widetilde { \phi } _ { t , j } ^ { \mathrm { o b s } } = \frac { \phi _ { t , j } ^ { \mathrm { o b s } } - \mu _ { j } } { \sigma _ { j } + 1 0 ^ { - 6 } } . } \end{array}\tag{4}
$$

which summarizes action tendencies, turn position, team context, evidence use, and answer behavior observed up to that point; $\mu _ { j }$ and $\sigma _ { j }$ are corpus statistics for coordinate $j .$ Future events are excluded from $\phi _ { t } ^ { \mathrm { o b s } }$ and used only as prediction targets. A role encoder produces an embedding $\xi _ { t }$ and predicts the same agent’s next action, future evidence utility, and final trajectory return using cross-entropy (CE), binary cross-entropy (BCE), and squared-error terms:

$$
\begin{array} { r } { \begin{array} { r } { \xi _ { t } = f _ { \theta _ { e } } ( \widetilde { \phi } _ { t } ^ { \mathrm { o b s } } ) , } \\ { ( \hat { a } _ { t } , \widehat { v } _ { t } ^ { \mathrm { e v d } } , \widehat { R } _ { t } ) = h _ { \theta _ { e } } ( \xi _ { t } ) , } \\ { { \mathcal { L } } _ { \mathrm { r o l e } } = \mathrm { C E } ( \hat { a } _ { t } , a _ { t _ { i } ^ { + } } ) + \lambda _ { \mathrm { e v d } } \mathrm { B C E } ( \widehat { v } _ { t } ^ { \mathrm { e v d } } , v _ { t } ^ { \mathrm { e v d } } ) } \\ { + \lambda _ { \mathrm { r e t } } \left( \widehat { R } _ { t } - R ( \tau ( t ) , y _ { \tau ( t ) } ) \right) ^ { 2 } . } \end{array} } \end{array}\tag{5}
$$

Here $t _ { i } ^ { + }$ is the next turn of the same agent, or a stop target if no such turn exists. This predictive objective shapes the embedding with future utility without placing sufix information in its input.

For each candidate role count, we run Euclidean K-means from each configured seed and summarize every suficiently supported cluster as a source prototype:

$$
\begin{array} { r l } & { { \mathbf { k } _ { K } ^ { ( s ) } } = \mathrm { K M e a n s } _ { 1 0 } ( \{ \xi _ { t } \} _ { t = 1 } ^ { J } ; K , s ) , } \\ & { ~ p _ { k } = ( \bar { \xi } _ { k } , \bar { \phi } _ { k } , \eta _ { k } ) , ~ k \in \mathcal { C } _ { K } , } \\ & { \mathcal { C } _ { K } = \{ k : n _ { k } \geq n _ { 0 } \} , } \end{array}\tag{6}
$$

where the subscript 10 denotes ten K-means initializations, s is a discovery seed, $n _ { k }$ is cluster support, and $n _ { 0 }$ is the minimum support. The vectors $\xi _ { k }$ and $\phi _ { k }$ are cluster centroids, and $\eta _ { k }$ records the cluster’s dominant action, stage, evidence, and return statistics. When K is not fixed by the execution budget, it is selected by a separation–stability criterion,

$$
K ^ { \star } = \arg \operatorname* { m a x } _ { K \in { \cal K } } \left[ \overline { { \mathrm { S i l } } } ( K ) + \lambda _ { \mathrm { s t a b } } \mathrm { S t a b i l i t y } ( K ) \right] .\tag{7}
$$

The appendix defines the seed averages and adjusted-Rand stability statistic. Finally, a deterministic joint resolver scores all surviving prototype–template pairs, assigns distinct functional types when possible, and converts the assignments into readable instructions $\psi _ { k }$ and token-aligned role markers $\chi _ { k } \colon$

$$
\begin{array} { r } { S _ { k , \upsilon } = \mathrm { T e m p l a t e S c o r e } _ { \upsilon } ( \bar { \phi } _ { k } , \eta _ { k } ) , } \\ { \{ { v _ { k } } \} _ { k \in \mathcal C _ { K } } = \mathrm { G r e e d y D i s t i n c t } ( S ; \prec _ { \mathrm { t i e } } ) , } \\ { \ ( \psi _ { k } , \chi _ { k } ) = \mathrm { T e m p l a t e } _ { \upsilon _ { k } } ( \bar { \phi } _ { k } , \eta _ { k } ) . \qquad } \end{array}\tag{8}
$$

Thus, role names are interpretations of learned behavioral clusters rather than manually specified training labels. Feature definitions, future-target construction, role-count settings, and template rules appear in the appendix.

## Executable Role Binding

Induced clusters become useful only when they control subsequent team behavior. ExRole selects at most N source prototypes, remaps them to contiguous runtime role IDs, and pads missing entries with a generic collaborative role:

$$
\mathcal { P } _ { \mathrm { e x e c } } = \mathrm { R e s o l v e } _ { N } ( \mathcal { P } ) , \qquad ( z _ { i } , \psi _ { z _ { i } } , \chi _ { z _ { i } } ) = \mathrm { B i n d } _ { i } ( \mathcal { P } _ { \mathrm { e x e c } } ) .\tag{9}
$$

The resolved role conditions the observation presented to the active agent,

$$
o _ { t } = \mathcal { O } \left( x , i _ { t } , z _ { i _ { t } } , \psi _ { z _ { i _ { t } } } , \chi _ { z _ { i _ { t } } } , \mathcal { H } _ { t } , \mathcal { B } _ { t } ^ { \mathrm { m s g } } , e _ { t - 1 } \right) ,\tag{10}
$$

where $B _ { t } ^ { \mathrm { m s g } }$ is the shared message board. The generated response is projected to a structured team message and environment action.

The marker remains in the token sequence and defines role-turn segments. If $u _ { q } ^ { - }$ is the start of marker $q ,$ then each token u inherits the latest preceding role:

$$
\begin{array} { r } { \mathrm { s e g } ( u ) = \operatorname* { m a x } \{ q : u _ { q } ^ { - } \leq u \} , } \\ { z ( u ) = z _ { \mathrm { s e g } ( u ) } , } \\ { \Gamma ( u ) = \Gamma _ { \mathrm { s e g } ( u ) } . } \end{array}\tag{11}
$$

This construction makes the role used in the prompt identical to the role consumed by the model-side router. The exact resolver, marker matching, and cached-decoding rules are deferred to the appendix.

## Role-Conditioned Sparse Routing

Prompt conditioning alone does not determine which trainable capacity supports a role. ExRole-Routed therefore modulates the shared LoRA update at the rank-slot level. For adapted module ℓ and role-turn segment q,

$$
\mathrm { L o R A } _ { \ell } ^ { \mathrm { r o l e } } ( h ) = B _ { \ell } \left( \gamma _ { q , \ell } \odot A _ { \ell } h \right) \alpha _ { \ell } , \qquad \gamma _ { q , \ell } \in \mathbf { R } ^ { d } .\tag{12}
$$

The router combines the executable-role feature $\mathbf { v } _ { z _ { q } }$ , its discrete identity, and a pooled semantic prefix $\zeta _ { q } \colon$

$$
\begin{array} { r l } & { \zeta _ { q } = \mathrm { N o r m } \left( \displaystyle \frac { 1 } { | \mathcal { W } _ { q } | } \sum _ { u \in \mathcal { W } _ { q } } \mathrm { s t o p g r a d } ( \mathcal { E } ( \mathrm { i d } _ { u } ) ) \right) , } \\ & { \displaystyle h _ { q } ^ { b } = F _ { \theta _ { b } } ( \mathbf { v } _ { z _ { q } } ) + E _ { \theta _ { b } } ( z _ { q } ) + \lambda _ { c } C _ { \theta _ { b } } ( \zeta _ { q } ) , } \\ & { \omega _ { q , \ell s } = \mathrm { H e a d } _ { \theta _ { b } } ^ { \mathrm { s l o t } } ( h _ { q } ^ { b } ) _ { \ell s } + \lambda _ { u } \operatorname { t a n h } \left( \mathrm { H e a d } _ { \theta _ { b } } ^ { \mathrm { u t i } } ( h _ { q } ^ { b } ) _ { \ell s } \right) , } \\ & { \hat { c } _ { q } = \mathrm { H e a d } _ { \theta _ { b } } ^ { \mathrm { c r e d } } ( h _ { q } ^ { b } ) . } \end{array}\tag{13}
$$

Here ${ \mathcal { W } } _ { q }$ is the prefix window before marker $q , \mathrm { i d } _ { u }$ is the token identifier (ID) at position u, and $\omega _ { q , \ell s }$ scores rank slot s of module ℓ. The scores define a probability distribution $\varpi _ { q }$ and a hard budget of S directly selected slots:

$$
\begin{array} { l } { { \displaystyle { \varpi _ { q , \ell s } } = \mathrm { s o f t m a x } ( \omega _ { q } / \tau _ { b } ) _ { \ell s } , } } \\ { { \displaystyle M _ { q , \ell s } = \mathbf { 1 } [ ( \ell , s ) \in \mathrm { T o p S } ( \omega _ { q } ) ] , } } \\ { { \displaystyle \delta _ { q , \ell s } ^ { \mathrm { s e l } } = M _ { q , \ell s } \big ( L d \varpi _ { q , \ell s } - 1 \big ) , } } \\ { { \displaystyle \bar { \delta } _ { q , \ell s } = \delta _ { q , \ell s } ^ { \mathrm { s e l } } - \frac { 1 } { L d } \sum _ { \ell ^ { \prime } , s ^ { \prime } } \delta _ { q , \ell ^ { \prime } s ^ { \prime } } ^ { \mathrm { s e l } } , } } \\ { { \displaystyle \gamma _ { q , \ell s } = \mathrm { c l i p } \left( 1 + \lambda _ { g } \bar { \delta } _ { q , \ell s } , \gamma _ { \mathrm { m i n } } , \gamma _ { \mathrm { m a x } } \right) . } } \end{array}\tag{14}
$$

The hard mask concentrates role-specific deviations, while centering preserves a balanced shared update before clipping. Both variants retain the same LoRA parameters and roleconditioned interaction; they difer only in the model-side routing state:

$$
\Gamma _ { q } = \left\{ \begin{array} { l l } { { \mathbf { 1 } , } } & { { \mathrm { E x R o l e - S h a r e d } , } } \\ { { \{ \gamma _ { q , \ell s } \} _ { \ell , s } , } } & { { \mathrm { E x R o l e - R o u t e d } . } } \end{array} \right.\tag{15}
$$

## Turn-Aligned Optimization

An episode-level reward alone assigns the same credit to every response in a team trajectory. ExRole retains the grouprelative trajectory advantage $\operatorname { A d v } ^ { ( g ) }$ from GRPO, while adding discounted local credit to the response tokens produced at turn t:

$$
\begin{array} { r l } & { U _ { t } ^ { \mathrm { t u r n } , ( g ) } = \displaystyle \sum _ { t ^ { \prime } = t } ^ { T _ { \tau ^ { ( g ) } } } \lambda _ { \mathrm { d i s c } } ^ { t ^ { \prime } - t } r _ { t ^ { \prime } } ^ { ( g ) } , } \\ & { \quad \widetilde { \mathrm { A d v } } _ { t , j } ^ { ( g ) } = I _ { t , j } ^ { \mathrm { r e s p } , ( g ) } \left[ \mathrm { A d v } ^ { ( g ) } + \alpha _ { c } \widehat { U } _ { t } ^ { ( g ) } \right] , } \end{array}\tag{16}
$$

where $\widehat { U } _ { t } ^ { ( g ) }$ is the normalized turn return and $I _ { t , j } ^ { \mathrm { r e s p } , ( g ) }$ masks non-response tokens. The same return supplies a detached role-turn credit target $c _ { q }$ for segment q:

$$
c _ { q } = \mathrm { s t o p g r a d } \left[ \mathrm { c l i p } \left( { \cal N } _ { B _ { R } } ( U _ { t ( q ) } ^ { \mathrm { t u r n } , ( g ( q ) ) } ) , - 2 , 2 \right) \right] .\tag{17}
$$

Here $\scriptstyle { B _ { R } }$ is the set of role-turn segments in the current normalization batch. The router learns to predict this target and associate positive role-turn credit with its selected capacity. Writing $\tilde { c } _ { q } = \mathrm { c l i p } ( ( c _ { q } + 2 ) / 4 , 0 , 1 )$ and letting $\bar { \nu } _ { q } ^ { \mathrm { u t i l } }$ denote mean predicted utility over selected slots, the two credit-bearing objectives are

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { f u t u r e } } = \displaystyle \frac { 1 } { 2 Q } \sum _ { q = 1 } ^ { Q } \left[ ( \bar { \nu } _ { q } ^ { \mathrm { u t i l } } - \tilde { c } _ { q } ) ^ { 2 } + ( \mathrm { s i g m } ( \widehat { c } _ { q } ) - \tilde { c } _ { q } ) ^ { 2 } \right] , } \\ & { \mathcal { L } _ { \mathrm { c r e d i t } } = \displaystyle - \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } c _ { q } \frac { \sum _ { \ell , s } M _ { q , \ell s } \log { \varpi _ { q , \ell s } } } { \sum _ { \ell , s } M _ { q , \ell s } } . } \end{array}\tag{18}
$$

Here sigm(·) denotes the logistic sigmoid.

Load-balance, role-diversity, sparsity, and entropy terms regularize the remaining routing distribution, while

![](images/8344acdedeee69921276927ebbaa00e08c2a0bbb0ac9289d4f1e51ea1e710e54.jpg)  
Figure 3: Trajectory-to-role binding and execution in ExRole. A deterministic resolver converts trajectory prototypes into executable roles with aligned instructions, markers, and features; fixed role identities then guide round-robin team interaction and model-side sparse routing through shared memory.

Kullback–Leibler (KL) regularization anchors the policy. The two model variants are therefore optimized under the same turn-aligned policy objective and difer only by the router loss:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { s h a r e d } } = - \mathcal { J } _ { \mathrm { G R P O } } ^ { \mathrm { t u r n } } + \beta _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } , } \\ & { \mathcal { L } _ { \mathrm { r o u t e r } } = \lambda _ { \mathrm { f u t u r e } } \mathcal { L } _ { \mathrm { f u t u r e } } + \lambda _ { \mathrm { c r e d i t } } \mathcal { L } _ { \mathrm { c r e d i t } } + \mathcal { L } _ { \mathrm { r e g } } , } \\ & { \mathcal { L } _ { \mathrm { r o u t e d } } = \mathcal { L } _ { \mathrm { s h a r e d } } + \mathcal { L } _ { \mathrm { r o u t e r } } . } \end{array}\tag{19}
$$

This matched formulation isolates the contribution of role-conditioned sparse LoRA routing: ExRole-Shared and ExRole-Routed use the same role library, interaction protocol, reward, and turn-aligned policy credit. The appendix gives the full GRPO objective, normalization fallbacks, router regularizers, pseudocode, and hyperparameters.

## Experiments

## Experimental Setup

Benchmarks. We use MuSiQue and 2WikiMultiHopQA, which require multi-step evidence composition over supporting-document collections.

Baselines. The controlled comparison includes single-agent search, multi-agent systems (MAS) without roles or with hand-written roles, and random or shufled role controls. All controlled systems share the same policy backbone, action interface, and evaluation protocol. ExRole-Shared and ExRole-Routed additionally share the induced role library and training objective, difering only in whether LoRA capacity is uniformly shared or role routed. Table 2 provides RAG, Search-RL, and MAS-RL references; its base icons distinguish SFT and search-specialized backbones.

Metrics. We report Exact Match (EM), token-level F1, and strict success (Succ). EM uses benchmark answer normalization, F1 measures token overlap, and Succ requires a normalized exact answer without substring credit. Complete data, retrieval, optimization, and sequence-budget settings are given in Supplementary Appendix B.6; formal metric definitions appear in Supplementary Appendix C.2.

![](images/7e375def5aedf3851e69fc476463f442caa287947f1bd4e7ffcf2b3617c96d7f.jpg)

![](images/a98e783400fdd8371dd52b07892c1ca3a301f67ea14a7859ba05704a08e806bc.jpg)  
(b) Action shares by scheduled turn.

(a) Frozen discovery profiles.  
![](images/6f3c723db805fd00a038289d6c542d425153115e9b8f5a38676213a549bc8d78.jpg)

![](images/bb50b60e21912450cc6c30280903585137ddf5dfc962e5f3f48201d2d20edcfc.jpg)  
(c) Standardized router signatures.  
(d) Frozen trajectory embeddings.  
Figure 4: Role induction and routed execution on MuSiQue. Panels (b) and (c) summarize 200 ExRole-Routed trajectories across scheduled team turns; panel (d) shows a t-SNE projection of frozen role embeddings.

## Main Results

Controlled role comparison. Table 1 compares role sources and trainable capacity under a common evaluation interface. On MuSiQue, ExRole-Shared obtains 31.5 EM and 43.2 F1, compared with 16.5 EM and 28.8 F1 for singleagent search and 20.0 EM and 31.6 F1 for the no-role MAS. On 2WikiMultiHopQA, ExRole-Shared and ExRole-Routed reach 49.0/59.1 and 50.0/59.7 EM/F1, respectively. Both capacity paths preserve the gains from trajectory-induced role conditioning across the two primary benchmarks.

Inference behavior. ExRole-Shared and ExRole-Routed use nearly identical numbers of team turns, search calls, retrieved characters, and repeated queries. Both variants retrieve target-bearing evidence for approximately 73% of MuSiQue examples and produce grounded answers for 87%, while strict success remains near 30%. This gap localizes much of the remaining error to answer selection and exact answer formation. Supplementary Appendix D.5 consolidates the matched cost distributions and stage-wise diagnostics.

<table><tr><td>Method</td><td>#Ag.</td><td>Role</td><td>Cap.</td><td colspan="3">MuSiQue</td><td colspan="3">2Wiki</td></tr><tr><td></td><td></td><td></td><td></td><td>EM↑</td><td>F1↑</td><td>Succ. ↑</td><td>EM↑</td><td>F1↑</td><td>Succ. ↑</td></tr><tr><td>Trajectory-induced roles</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ExRole-Shared</td><td>3</td><td>Induced</td><td>Shared</td><td>31.5</td><td>43.2</td><td>31.5</td><td>49.0</td><td>59.1</td><td>49.0</td></tr><tr><td>ExRole-Routed</td><td>3</td><td>Induced</td><td>Routed</td><td>30.0</td><td>41.5</td><td>30.0</td><td>50.0</td><td>59.7</td><td>50.0</td></tr><tr><td>Single-agent baseline</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Single-agent search</td><td>1</td><td>None</td><td></td><td>Shared 16.5 (-15.0)</td><td></td><td>28.8 (-14.4) 16.5 (-15.0)</td><td>36.5 (-13.5) 43.6 (-16.1) 36.5 (-13.5)</td><td></td><td></td></tr><tr><td>Three-agent baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-role MAS</td><td>3</td><td>None</td><td></td><td>Shared 20.0 (-11.5)</td><td>31.6(-11.6)</td><td>20.0 (-11.5)</td><td>38.0 (-12.0)</td><td>44.5 (-15.2)</td><td>38.0 (-12.0)</td></tr><tr><td>Manual-role MAS</td><td>3</td><td>Human</td><td></td><td>Shared 13.0 (-18.5)</td><td>23.4 (-19.8)</td><td>12.5 (-19.0)</td><td>31.5 (-18.5)</td><td>37.8 (-21.9)</td><td>31.5 (-18.5)</td></tr><tr><td>Role-source controls</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random role prompt</td><td>3</td><td></td><td></td><td>Random Shared 16.5 (-15.0)</td><td>28.0 (-15.2)</td><td>16.5 (-15.0)</td><td>36.0 (-14.0)</td><td>43.1 (-16.6)</td><td>36.0 (-14.0)</td></tr><tr><td>Shuffled induced role</td><td>3</td><td></td><td></td><td>Mismatch Shared 8.5 (-23.0)</td><td>19.1 (-24.1)</td><td>8.5 (-23.0)</td><td>42.3 (-7.7)</td><td>50.0 (-9.7)</td><td>42.3 (-7.7)</td></tr></table>

Table 1: Primary ExRole comparison on MuSiQue and 2WikiMultiHopQA. #Ag. is the number of agents, Cap. is the trainablecapacity path, and Succ. is strict success. Parenthesized values for controls are diferences from the stronger ExRole variant in each metric. The 2Wiki shufled-role result is averaged over three role-assignment seeds.

<table><tr><td>Method</td><td></td><td>Base MuSiQue EM/F1</td><td>2Wiki EM/F1</td></tr><tr><td>Retrieval baselines</td><td></td><td></td><td></td></tr><tr><td>Direct</td><td></td><td>3.0/11.4 22.5/26.2</td><td></td></tr><tr><td>Naive RAG</td><td></td><td>12.5/22.326.5/29.6</td><td></td></tr><tr><td>Search-RL baselines</td><td></td><td></td><td></td></tr><tr><td>Search-R1</td><td></td><td>33.0/44.3 52.0/60.3</td><td></td></tr><tr><td>R1-Searcher</td><td></td><td>47.5/59.064.5/71.2</td><td></td></tr><tr><td>MAS-RL baselines</td><td></td><td></td><td></td></tr><tr><td>MAGRPO</td><td></td><td>12.0/23.1 26.5/35.2</td><td></td></tr><tr><td>Dr. MAS</td><td></td><td>30.5/37.860.0/67.5</td><td></td></tr><tr><td>MATPO</td><td></td><td>36.5/48.845.3/54.4</td><td></td></tr><tr><td>ExRole</td><td></td><td></td><td></td></tr><tr><td>ExRole-Shared</td><td>★</td><td>31.5/43.2 49.0/59.1</td><td></td></tr><tr><td>ExRole-Routed</td><td>★</td><td>30.0/41.5 50.0/59.7</td><td></td></tr></table>

Table 2: External EM/F1 references on the primary benchmarks under each method’s stated protocol. Base icons denote ★ shared SFT, ◆ Search-R1/Qwen, and ● R1-Searcher backbones.

Role mechanism. Figure 4 connects trajectory-level role induction to routed execution on MuSiQue. The induced Researcher, Analyst, and Verifier difer in search, evidence, answer, and timing statistics, and their trajectory embeddings form separated regions. Their scheduled turns also produce distinct action and routing profiles. The figure characterizes role-conditioned execution under the fixed speaker schedule.

## Role-Agent-Turn Disentanglement

Using fixed checkpoints, we orthogonally vary speaker order and agent-role assignment on MuSiQue. Figure 7(a) shows that role identity reduces held-out action log loss by 19.1% for Shared and 18.7% for Routed after accounting for agent identity and turn phase. Between-role Jensen– Shannon distance is 48–60 times the within-role cross-agent distance, while Researcher, Analyst, and Verifier search rates are 96.6%, 85.8%, and 61.3%. These results indicate transferable action tendencies rather than fixed agent or turn labels; Supplementary Appendix D.3 reports the full matrix.

External reference. Table 2 reports representative retrieval, Search-RL, MAS-RL, and ExRole results on the primary benchmarks. ExRole-Shared reaches 49.0/59.1 EM/F1 on 2WikiMultiHopQA and 31.5/43.2 on MuSiQue; ExRole-Routed reaches 50.0/59.7 and 30.0/41.5. ExRole is competitive with several MAS-RL systems but remains below the strongest search-specialized policies.

## Capacity-Path Analysis

Paired capacity-path comparison. We evaluate ExRole-Shared and ExRole-Routed on the same 200 examples for each benchmark. Both capacity paths preserve the inducedrole gains over the corresponding single-agent and no-role controls, while their relative ordering varies across benchmarks. Supplementary Appendix D.1 reports the complete paired diferences, confidence intervals, and per-example comparisons.

Router training signals. Figure 5 reports the routing objectives and turn credit across the two primary benchmarks. The auxiliary and future-utility losses remain active throughout training, while turn credit varies with sampled trajectories. Supplementary Appendix D.1 reports the corresponding task-level efects.

Matched optimization dynamics. Figure 6 compares Shared and Routed under the same training horizon. Its columns correspond to MuSiQue and 2WikiMultiHopQA, while the two rows show smoothed GRPO policy loss and the magnitude of turn-aligned credit. Both variants receive turn-level credit throughout optimization; only Routed additionally receives the router losses in Figure 5.

![](images/041c1a9d21527c2f631a453113e82917521179e42936e502a73dee50b2f5872a.jpg)  
(a) Router auxiliary loss.

![](images/95f7032ec129f2196247c0b66815492fcff40d835016e0af302dfba19e9fd298.jpg)  
(b) Future-utility loss.

![](images/acbc1a2d0e8203aca7509610d9e6e6ce0994ff57ca49439658b32c6a594bef63.jpg)  
(c) Absolute turn credit.  
Figure 5: Routed training signals across MuSiQue and 2WikiMultiHopQA. Pale traces are rank-averaged observations; dark traces are centered rolling means over normalized training progress.  
Shared Routed

![](images/cf65c28c696e5723d5407e8cb4fecc0ceb0803dfc6ad7786b90a727078b41268.jpg)  
(a) MuSiQue policy loss.

![](images/0bf3c60e42553a3beb8cfd231ee0c5f364d2784288a56f4127ae2d2bb73aff95.jpg)

![](images/b7d7c04a02883896f05ec2aef2f7e7161dd24be254f22eebe655c19df3233978.jpg)  
(c) MuSiQue turn credit.

(b) 2Wiki policy loss.  
![](images/b23925722ad8029015611a3bc882542ce4aaf9f59d8956f13a3803dee1315f39.jpg)  
(d) 2Wiki turn credit.

Figure 6: Matched Shared and Routed optimization dynamics. The upper row reports GRPO policy loss and the lower row reports absolute turn credit; all curves are rank-averaged centered rolling means over the same 128-update horizon.  
![](images/7c88e0a589cb9e26277c62b36d67938769b1bbe54ff8c396fcf4ff262aef7f0c.jpg)  
Held-out action log loss (lower is better)  
(a) Incremental role information.

![](images/ad800472df0cac82cb7d164a22dcc96484f9eb4bb88b66a8dc117481f20fe819.jpg)  
Jensen-Shannon distance (log scale)  
(b) Cross-agent behavioral consistency.

![](images/6f0d64d7a659faf08d09c2d1cef1faa93d7681ea4bed870562bd02d01408a751.jpg)  
(c) Role-specific action profiles.  
Figure 7: Role-Agent-Turn disentanglement on MuSiQue using fixed Shared and Routed checkpoints without additional policy training.

## Conclusion

ExRole turns team trajectories into executable roles for interaction and turn-aligned optimization. On MuSiQue and 2WikiMultiHopQA, induced roles yield distinct search, synthesis, and verification behaviors and outperform singleagent and role-free controls. Role-Agent-Turn interventions show that these behaviors generalize beyond fixed agents and turn positions, supporting trajectory-induced role binding as a practical framework for structured language-agent collaboration. This interface makes learned specialization explicit and reusable during policy optimization.

## References

Buehler, E. L.; and Buehler, M. J. 2024. X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Molecular Design. arXiv preprint arXiv:2402.07148.

Chen, G.; Yang, S.; Li, C.; Liu, W.; Luan, J.; and Xu, Z. 2026. End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 30319–30338.

Foerster, J. N.; Farquhar, G.; Afouras, T.; Nardelli, N.; and Whiteson, S. 2018. Counterfactual Multi-Agent Policy Gradients. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2974–2982.

Goel, H.; Omama, M.; Chalaki, B.; Tadiparthi, V.; Pari, E. M.; and Chinchali, S. 2025. R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-Agent Reinforcement Learning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, 19600–19620.

Hu, Z.; Zhang, Z.; Li, H.; Chen, C.; Ding, H.; and Wang, Z. 2024. Attention-Guided Contrastive Role Representations for Multi-Agent Reinforcement Learning. In International Conference on Learning Representations.

Huang, C.; Liu, Q.; Lin, B. Y.; Pang, T.; Du, C.; and Lin, M. 2024. LoraHub: Eficient Cross-Task Generalization via Dynamic LoRA Composition. In Conference on Language Modeling.

Jin, B.; Zeng, H.; Yue, Z.; Yoon, J.; Arık, S. Ö.; Wang, D.; Zamani, H.; and Han, J. 2025. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. In Second Conference on Language Modeling.

Li, G.; Hammoud, H. A. A. K.; Itani, H.; Khizbullin, D.; and Ghanem, B. 2023. CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society. In Advances in Neural Information Processing Systems, volume 36, 51991–52008.

Li, H.; Su, Z.; Xue, Y.; Tian, Z.; Song, Y.; and Huang, M. 2025. Advancing Collaborative Debates with Role Diferentiation through Multi-Agent Reinforcement Learning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 22655–22666.

Li, X.; Pan, L.; and Zhang, J. 2024. Kaleidoscope: Learnable Masks for Heterogeneous Multi-Agent Reinforcement Learning. In Advances in Neural Information Processing Systems.

Liao, J.; Wen, M.; Wang, J.; and Zhang, W. 2025. MARFT: Multi-Agent Reinforcement Fine-Tuning. arXiv preprint arXiv:2504.16129.

Liu, S.; Liang, Z.; Lyu, X.; and Amato, C. 2026. LLM Collaboration with Multi-Agent Reinforcement Learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 32150–32158.

Mo, Z.; Li, X.; Chen, Y.; and Bing, L. 2025. Multi-Agent Tool-Integrated Policy Optimization. arXiv preprint arXiv:2510.04678.

Park, J.; Cho, S.; and Lee, J.-Y. 2026. Divide and Cooperate: Role-Decomposed Multi-Agent LLM Training with Cross-Agent Learning Signals. arXiv preprint arXiv:2606.10684.

Song, H.; Jiang, J.; Min, Y.; Chen, J.; Chen, Z.; Zhao, W. X.; Fang, L.; and Wen, J.-R. 2025. R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2503.05592.

Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal, A. 2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 10014– 10037.

Wang, T.; Dong, H.; Lesser, V.; and Zhang, C. 2020. ROMA: Multi-Agent Reinforcement Learning with Emergent Roles. In Proceedings of the 37th International Conference on Machine Learning, volume 119, 9876–9886.

Wang, T.; Gupta, T.; Mahajan, A.; Peng, B.; Whiteson, S.; and Zhang, C. 2021. RODE: Learning Roles to Decompose Multi-Agent Tasks. In International Conference on Learning Representations.

Wu, X.; Huang, S.; and Wei, F. 2024. Mixture of LoRA Experts. In International Conference on Learning Representations, volume 2024, 47302–47318.

Xu, Y.; Zhou, Z.; Sang, H.; Li, X.; Zhang, J.; Du, X.; Na, S.; Wang, Z.; and Geramifard, A. 2026. TRIAGE: Role-Typed Credit Assignment for Agentic Reinforcement Learning. arXiv preprint arXiv:2606.32017.

Yang, M.; Zhao, J.; Hu, X.; Zhou, W.; Zhu, J.; and Li, H. 2022. LDSA: Learning Dynamic Subtask Assignment in Cooperative Multi-Agent Reinforcement Learning. In Advances in Neural Information Processing Systems, volume 35, 1698– 1710.

Yu, Y.; Yin, Q.; Zhang, J.; Xu, P.; and Huang, K. 2024. ADMN: Agent-Driven Modular Network for Dynamic Parameter Sharing in Cooperative Multi-Agent Reinforcement Learning. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, 302–310.

Yue, Y.; Zhang, G.; Liu, B.; Wan, G.; Wang, K.; Cheng, D.; and Qi, Y. 2025. MasRouter: Learning to Route LLMs for Multi-Agent Systems. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 15549–15572.

Zhao, Z.; Gan, L.; Wang, G.; Zhou, W.; Yang, H.; Kuang, K.; and Wu, F. 2024. LoraRetriever: Input-Aware LoRA Retrieval and Composition for Mixed Tasks in the Wild. In Findings of the Association for Computational Linguistics: ACL 2024, 4447–4462.

Zhou, H.; Geng, H.; Xue, X.; Kang, L.; Qin, Y.; Wang, Z.; Yin, Z.; and Bai, L. 2025. ReSo: A Reward-Driven Self-Organizing LLM-Based Multi-Agent System for Reasoning Tasks. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, 15979–15998.

## Appendix Guide

Index Content   
B Method and Experimental Details   
B.1 Complete Environment Objective   
B.2 Role Features, Targets, and Prototype Con  
struction   
B.3 Deterministic Role Resolution and Marker   
Alignment   
B.4 Router Objectives and Implementation Details   
B.5 Full Turn-Aligned GRPO Objective   
B.6 Experimental Configuration   
B.7 Manual-role Control Prompt   
B.8 Role-Library Provenance and Split Isolation   
C Evaluation, Inference, and Optimization Details   
C.1 Limitations   
C.2 Evaluation Metrics and Agreement Analysis   
C.3 Inference Procedure   
C.4 Gradient Flow through Balanced Sparse-Delta   
Routing   
C.5 Training-Time Credit Flow   
C.6 Turn-Aligned Routed GRPO Algorithm   
D Additional Experimental Diagnostics   
D.1 Paired Routing Efects   
D.2 Role Induction and Evaluation Diagnostics   
D.3 Role-Agent-Turn Disentanglement   
D.4 Role Induction and Turn-Aligned Credit   
D.5 Inference Cost and Answer Diagnostics   
D.6 Full-Wikipedia HotpotQA Stress Test   
D.7 External Reference Training Curves   
E Notation   
E.1 Symbol Table

## Method and Experimental Details Complete Environment Objective

The environment exposes the controlled action space

$$
\begin{array} { r } { \mathcal { A } = \{ \mathrm { s e a r c h } ( q _ { \mathrm { r e t } } ) , \mathrm { a n s w e r } ( \hat { y } ) , \mathrm { i n v a l i d } \} . } \end{array}\tag{20}
$$

Before turn t, the available team history is $\begin{array} { r l } { \mathcal { H } _ { t } } & { { } = } \end{array}$ $\{ a _ { t ^ { \prime } } , m _ { t ^ { \prime } } , e _ { t ^ { \prime } } \} _ { t ^ { \prime } < t }$ . Unless the episode terminates early, speakers follow the round-robin schedule $i _ { t } = 1 + ( ( t -$ 1) mod N). The turn reward used by all ExRole variants is

$$
\begin{array} { r l } { r _ { t } = r _ { \mathrm { f } } ^ { \mathrm { f } } \alpha ^ { \mathrm { s a e } } + r _ { \mathrm { f } } ^ { \mathrm { a } } - \mathrm { P e n } _ { t } , } & { } \\ { r _ { t } ^ { \mathrm { \Delta } } = \lambda _ { \mathrm { a w } } r _ { t } ^ { \mathrm { a n s } } + \lambda _ { \mathrm { s u p } } r _ { t } ^ { \mathrm { s a p } } } & { } \\ { + \lambda _ { \mathrm { a w } } r _ { t } ^ { \mathrm { n t } } } & { } \\ { + \lambda _ { \mathrm { a w } } r _ { t } ^ { \mathrm { r e t } } } & { } \\ { + \lambda _ { \mathrm { v e r } } ^ { \mathrm { f } } r _ { t } ^ { \mathrm { f } } \alpha ^ { \mathrm { p } } , } & { } \\ { \mathrm { p e n } _ { t } = \lambda _ { \mathrm { r e t } } \mathrm { p e n } _ { t } ^ { \mathrm { p } } \mathrm { e n } _ { t } ^ { \mathrm { p } } + \lambda _ { \mathrm { o r a r y } } \mathrm { p e n } _ { t } ^ { \mathrm { c a r t y } } } \\ { + \lambda _ { \mathrm { b r i d g . } } \mathrm { P e n } _ { t } ^ { \mathrm { b r a l i d g e } } } & { } \\ { + \lambda _ { \mathrm { i n s t } } \mathrm { f r o n } _ { t } ^ { \mathrm { i n t } } } & { } \\ { + \lambda _ { \mathrm { g r e r o u p } } \mathrm { P e n } _ { t } ^ { \mathrm { g r o m } _ { t } } } & { } \\ { + \lambda _ { \mathrm { a w } } \mathrm { p e n } _ { t } ^ { \mathrm { p } } \mathrm { e n } ^ { \mathrm { t } } \mathrm { s u p } } & { } \\ { + \lambda _ { \mathrm { n s t } } \mathrm { p e n } _ { t } ^ { \mathrm { i n t } } \mathrm { s u p } } & { } \\ { + \lambda _ { \mathrm { n s t } } \mathrm { p e n } _ { t } ^ { \mathrm { p } } \mathrm { e n } ^ { \mathrm { s a m } _ { s } } . } \end{array}\tag{21}
$$

The shared base term covers action format, communication, answer type, and grounding. The positive terms reward a correct answer, supporting evidence, evidence novelty, and a grounded verifier decision. The penalties cover repeated evidence, premature answers, intermediate bridge entities, insuficient evidence, grounded but incorrect answers, unsupported answers, and episodes that terminate without an answer. With $\mathcal { V } ( y )$ denoting acceptable aliases,

$$
r _ { t } ^ { \mathrm { a n s } } = \operatorname* { m a x } _ { y ^ { \prime } \in \mathcal { V } ( y ) } \mathbf { 1 } _ { \mathrm { s u c c } } ( \hat { y } _ { t } , y ^ { \prime } ) .\tag{22}
$$

The strict indicator compares $n _ { \mathrm { s u c c } } ( \hat { y } _ { t } )$ with $n _ { \mathrm { e m } } ( y ^ { \prime } )$ where $n _ { \mathrm { s u c c } } = n _ { \mathrm { e m } } \circ c _ { \mathrm { a n s } }$ and $c _ { \mathrm { a n s } }$ removes the structured action wrapper and common answer prefixes.

Role Features, Targets, and Prototype Construction The prefix feature uses the following fixed 30-coordinate schema:

$$
\begin{array} { r l } & { \boldsymbol { \phi } _ { t } ^ { \mathrm { o b s } } = \left[ \mathbf { r } _ { t } ^ { i } \in \mathbf { R } ^ { 1 4 } , \mathbf { c } _ { t } ^ { i } \in \mathbf { R } ^ { 4 } , \mathbf { r } _ { t } ^ { \mathrm { t e a m } } \in \mathbf { R } ^ { 5 } , \right. } \\ & { \quad \quad \quad \left. p _ { t } , b _ { t } , \mathrm { o n e h o t } _ { 5 } ( a _ { t } ) \right] . } \end{array}\tag{23}
$$

In order, $\mathbf { r } _ { t } ^ { i }$ contains the active agent’s search, answer, invalid, and message rates; mean message and retrieval lengths; mean, early, and late turn-position statistics; mean sharedboard size; accumulated reward contribution; and evidencehit, grounded-answer, and repair rates. The count vector $\mathbf { c } _ { t } ^ { i }$ contains search, answer, message, and turn counts. The team vector $\mathbf { r } _ { t } ^ { \mathrm { t e a m } }$ contains team search, answer, message, tool-use, and evidence-hit rates; $p _ { t }$ and $b _ { t }$ are progress and remainingbudget ratios. Rates use the number ofobserved agent or team turns as their denominator, means are zero for empty sets, and missing scalar metadata are set to zero. For coordinate j over the $J$ discovery records, the encoder input is

$$
\begin{array} { r l r } { \widetilde { \phi } _ { t , j } ^ { \mathrm { o b s } } = \frac { \phi _ { t , j } ^ { \mathrm { o b s } } - \mu _ { j } } { \sigma _ { j } + 1 0 ^ { - 6 } } , } & { } & \\ { \mu _ { j } = \displaystyle \frac { 1 } { J } \sum _ { t = 1 } ^ { J } \phi _ { t , j } ^ { \mathrm { o b s } } , \qquad \sigma _ { j } ^ { 2 } = \frac { 1 } { J } \sum _ { t = 1 } ^ { J } ( \phi _ { t , j } ^ { \mathrm { o b s } } - \mu _ { j } ) ^ { 2 } . } & { } & \end{array}\tag{24}
$$

Every component is computed from the logged prefix ending at turn t. Future evidence and final return are used only as targets.

For the next-action target, $t _ { i } ^ { + } = \operatorname* { m i n } \{ t ^ { \prime } > t : i _ { t ^ { \prime } } = i _ { t } \}$ is the next turn of the same agent; if it does not exist, the target is stop. Let $h _ { t ^ { \prime } } ^ { \mathrm { e v d } }$ denote the logged evidence-hit or grounding indicator at turn $t ^ { \prime } .$ . The three targets are

$$
\begin{array} { r l } & { \quad \boldsymbol { a } _ { t } ^ { + } = \left\{ \begin{array} { l l } { a _ { t _ { i } ^ { + } } , } & { t _ { i } ^ { + } \mathrm { ~ e x i s t s } , } \\ { \mathrm { s t o p } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \\ & { \boldsymbol { v } _ { t } ^ { \mathrm { e v d } } = \mathbf { 1 } \bigl [ \exists t ^ { \prime } > t : i _ { t ^ { \prime } } = i _ { t } \wedge h _ { t ^ { \prime } } ^ { \mathrm { e v d } } = 1 \bigr ] , } \\ & { \quad \boldsymbol { R } _ { t } = \boldsymbol { R } ( \tau ( t ) , y _ { \tau ( t ) } ) . } \\ & { \mathrm { e ~ } \lambda _ { \mathrm { e v d } } = \boldsymbol { 1 } \mathrm { ~ a n d ~ } \lambda _ { \mathrm { r e t } } = 0 . 5 \mathrm { ~ i n ~ } \mathcal { L } _ { \mathrm { r o l e } } . } \end{array}\tag{25}
$$

For a configured discovery-seed set ${ \mathcal { S } } .$ , K-means uses Euclidean distance, ten initializations per seed, and the following model-selection statistics:

$$
\begin{array} { r } { { \displaystyle { \bf k } _ { \scriptscriptstyle K } ^ { ( s ) } = \mathrm { K M e a n s } _ { 1 0 } ( \{ \xi _ { t } \} _ { t = 1 } ^ { J } ; K , s ) } , \ ~ } \\ { \overline { { \mathrm { S i l } } } ( K ) = \frac { 1 } { | { \cal S } | } \displaystyle { \sum _ { s \in { \cal S } } \mathrm { S i l } ( \{ \xi _ { t } \} , { \bf k } _ { \scriptscriptstyle K } ^ { ( s ) } ) } , \ ~ } \\ { \mathrm { S t a b i l i t y } ( K ) = \frac { 2 } { | { \cal S } | ( | { \cal S } | - 1 ) } \displaystyle { \sum _ { s < s ^ { \prime } } \mathrm { A R I } ( { \bf k } _ { \scriptscriptstyle K } ^ { ( s ) } , { \bf k } _ { \scriptscriptstyle K } ^ { ( s ^ { \prime } ) } ) } , \ ~ } \\ { K ^ { \star } = \arg \displaystyle { \operatorname* { m a x } _ { K \in { \cal K } } \left[ \mathrm { S i l } ( K ) + \lambda _ { \mathrm { s t a b } } \mathrm { S t a b i l i t y } ( K ) \right] } . \ } \end{array}\tag{26}
$$

Ties in the last line are resolved by the larger K, matching the implementation, and the final assignments use the first configured seed $s _ { 0 } \colon k _ { t } = k _ { K ^ { \star } , t } ^ { ( s _ { 0 } ) }$ . Clusters with support below n<sub>0</sub> are discarded before prototype construction.

For cluster $k ,$ let $\mathcal { T } _ { k } = \{ t : k _ { t } = k \}$ and $n _ { k } = | \mathcal { T } _ { k } |$ . Its centroids and descriptive success lift are

$$
\begin{array} { l } { { \displaystyle \bar { \xi } _ { k } = \frac { 1 } { n _ { k } } \sum _ { t \in \mathcal { Z } _ { k } } \xi _ { t } , } } \\ { { \displaystyle \bar { \phi } _ { k } = \frac { 1 } { n _ { k } } \sum _ { t \in \mathcal { Z } _ { k } } \phi _ { t } ^ { \mathrm { o b s } } , } } \\ { { \displaystyle \mathrm { L i f t } _ { k } = \frac { 1 } { n _ { k } } \sum _ { t \in \mathcal { Z } _ { k } } Y _ { \tau ( t ) } ^ { \mathrm { s u c c } } - \frac { 1 } { J } \sum _ { t = 1 } ^ { J } Y _ { \tau ( t ) } ^ { \mathrm { s u c c } } . } } \end{array}\tag{27}
$$

The metadata profile $\eta _ { k }$ records dominant next action, evidence-hit statistics, predicted return, future evidence hit, and final return. Readable instructions are generated without an additional language model. Each cluster is scored against a fixed functional vocabulary,

$$
\begin{array} { r l } & { S _ { k , \upsilon } = \mathrm { T e m p l a t e S c o r e } _ { \upsilon } ( \bar { \phi } _ { k } , \eta _ { k } ) , } \\ & { \mathcal { U } _ { \mathrm { r o l e } } = \{ \mathrm { R e s e a r c h e r } , \mathrm { C o o r d i n a t o r } , \mathrm { V e r i f i e r } , \mathrm { A n a l y s t } \} . } \end{array}\tag{28}
$$

High early-search and evidence-hit rates favor a research template, whereas late grounded answers favor a verification template. Let G(S) greedily scan all $( k , v )$ pairs in decreasing $\mathbf { \bar { \rho } } ( S _ { k , v } , k , v )$ order, accepting a pair only when neither its cluster nor functional type has been assigned. The last coordinate follows reverse lexicographic order over the fixed vocabulary, matching the implementation. Any cluster left unmatched after this uniqueness pass takes its own highestscoring type, with ties resolved by the vocabulary order Researcher, Coordinator, Verifier, Analyst:

$$
\begin{array} { r } { \begin{array} { r l } & { v _ { k } = \left\{ \begin{array} { l l } { v , } & { ( k , v ) \in \mathcal { G } ( S ) , } \\ { \mathrm { a r g } \operatorname* { m a x } _ { v \in \mathcal { U } _ { \mathrm { r o l e } } } S _ { k , v } , } & { k \notin \mathrm { d o m } \mathcal { G } ( S ) , } \end{array} \right. } \\ & { \big ( \psi _ { k } , \chi _ { k } \big ) = \mathrm { T e m p l a t e } _ { v _ { k } } \big ( \bar { \phi } _ { k } , \eta _ { k } \big ) . } \end{array} } \end{array}\tag{29}
$$

Thus the role type of one cluster can depend on the scores of other clusters; the resolver is a joint deterministic map, not a pointwise template lookup.

We set $\lambda _ { \mathrm { s t a b } } = 0 . 2$ in the role-count criterion and discard clusters below the minimum support $n _ { 0 }$ . MuSiQue fixes $K = 3$ to the three-agent execution budget. HotpotQA and

2WikiMultiHopQA search $K \in \{ 2 , 3 , 4 \}$ and select $K = 4$ before resolving three executable roles. The broader sensitivity analysis in the appendix is post-hoc and is not used for model selection.

## Deterministic Role Resolution and Marker Alignment

The resolver ranks source prototypes using

$$
\begin{array} { r } { \kappa ( p _ { k } ) = \Big ( \mathrm { s t a g e } ( p _ { k } ) , - \eta _ { k } ^ { \mathrm { e a r l y } } , \eta _ { k } ^ { \mathrm { l a t e } } , } \\ { \eta _ { k } ^ { \mathrm { p o s } } , - \mathrm { L i f t } _ { k } , - n _ { k } , k \Big ) . } \end{array}\tag{30}
$$

The resolver sorts κ lexicographically in ascending order, so coordinator and planner roles precede researcher and solver roles, followed by analyst and synthesizer roles and then verifier roles; the final source-cluster ID breaks exact ties. Let $\mathcal { P } _ { N } ^ { \mathrm { s e l } }$ be the first N prototypes under this ordering, $K _ { \mathrm { s e l } } = | \mathcal { P } _ { N } ^ { \mathrm { s e l } } |$ |, and $\iota _ { i }$ the source ID of its ith entry. Source IDs are preserved when they already form a contiguous runtime range; otherwise they are remapped by execution position:

$$
\begin{array}{c} \begin{array} { r l } & { \mathcal { P } _ { N } ^ { \mathrm { s e l } } = \mathrm { T a k e } _ { N } ( \mathrm { S o r t } _ { \kappa } ( \mathcal { P } ) ) , } \\ & { \quad \quad \quad z _ { i } = \left\{ { \iota } _ { i } , \begin{array} { l } { \quad \{ { \iota } _ { j } \} _ { j = 1 } ^ { K _ { \mathrm { s e l } } } = \{ 0 , \ldots , K _ { \mathrm { s e l } } - 1 \} , } \\ { { i - 1 , \quad \mathrm { o t h e r w i s e } , } } \end{array} \right.} \end{array}   \\ & { \quad \mathcal { P } _ { \mathrm { e x e c } } = \mathrm { S o r t } _ { z } \left( \mathrm { P a d G e n e r i c } ( \{ ( \mathcal { P } _ { N , i } ^ { \mathrm { s e l } } , z _ { i } ) \} _ { i = 1 } ^ { K _ { \mathrm { s e l } } } , N ) \right) . } \end{array}\tag{31}
$$

Source IDs remain in metadata for auditability, while the router indexes only contiguous runtime role IDs. If fewer than N roles survive, the resolver appends a generic collaborative fallback with a fresh marker.

Equivalent tokenizations of each marker are matched, overlapping occurrences are coalesced, and the resulting spans are ordered as

$$
\mathcal { M } = \{ ( u _ { q } ^ { - } , u _ { q } ^ { + } , z _ { q } ) \} _ { q = 1 } ^ { Q } , \qquad u _ { 1 } ^ { - } < \cdot \cdot \cdot < u _ { Q } ^ { - } .\tag{32}
$$

Tokens before the first marker use the first segment, and cached decoding retains the latest segment ID. A malformed sequence without a marker falls back to runtime role 0; formal ExRole prompts always include a marker.

## Algorithm 1 Deterministic assignment of induced roles

☞ Input: Source prototypes P and number of agents N   
✓ Output: Runtime roles $\left( z _ { 1 } , \dots , z _ { N } \right)$ and marker  
conditioned observations   
1: Score prototypes with $\kappa ( p _ { k } )$ and retain the first N   
2: while fewer than N roles are available do   
3: Append a generic collaborative role with a fresh marker   
4: end while   
5: Preserve contiguous source IDs; otherwise remap by exe  
cution position   
6: Sort executable roles by runtime role ID   
7: for i = 1 to N do   
8: Bind role $z _ { i }$ and marker $\chi _ { z _ { i } }$ to agent i   
9: end for   
10: return $\left( z _ { 1 } , \dots , z _ { N } \right)$ and updated observation templates

## Router Objectives and Implementation Details

The main text writes the normalized routing distribution as $\varpi _ { q , \ell s }$ . Equivalently, a total soft budget S is distributed across all adapted-module–rank slots:

$$
\tilde { b } _ { q , \ell s } = S \varpi _ { q , \ell s } , \qquad \bar { b } _ { q } = \frac { 1 } { L d } \sum _ { \ell , s } \tilde { b } _ { q , \ell s } = \frac { S } { L d } .\tag{33}
$$

Hence $\tilde { b } _ { q , \ell s } / \bar { b } _ { q } = L d \varpi _ { q , \ell s } ,$ , which yields the compact sparse-delta expression in the main text. The selected-slot utility prediction is

$$
\bar { \nu } _ { q } ^ { \mathrm { u t i l } } = \frac { \sum _ { \ell , s } M _ { q , \ell s } \operatorname { s i g m } ( \nu _ { q , \ell s } ^ { \mathrm { u t i l } } ) } { \sum _ { \ell , s } M _ { q , \ell s } } .\tag{34}
$$

Let $\begin{array} { r } { \bar { \varpi } _ { \ell s } = Q ^ { - 1 } \sum _ { q } \varpi _ { q , \ell s } } \end{array}$ . The remaining regularizers are

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { l o a d } } = \frac { 1 } { L d } \sum _ { \ell , s } \bigg ( \overline { { \varphi } } _ { \ell s } - \frac { 1 } { L d } \bigg ) ^ { 2 } , } \\ { \displaystyle \mathcal { L } _ { \mathrm { d i v } } = \frac { 1 } { N ( N - 1 ) } \sum _ { z \neq \tau ^ { \prime } } \frac { \overline { { \Phi } } _ { z } ^ { \top } \overline { { \Phi } } _ { z ^ { \prime } } } { | \overline { { \Phi } } _ { z } | | \overline { { \Phi } } _ { z ^ { \prime } } | | _ { 2 } } , } \\ { \displaystyle \mathcal { L } _ { 1 1 } = \frac { 1 } { Q L d } \sum _ { q , \ell , s } \widehat { \boldsymbol { \theta } } _ { q , \ell s } ( 1 - M _ { q , \ell s } ) , } \\ { \displaystyle \mathcal { L } _ { \mathrm { e n t } } = \frac { 1 } { Q } \sum _ { q , \ell , s } \varpi _ { q , \ell s } \log \varpi _ { q , \ell s } , } \\ { \displaystyle \mathcal { L } _ { \mathrm { r e g } } = \lambda _ { \mathrm { l o a d } } \sum _ { \mathrm { l o a d } } + \lambda _ { \mathrm { d i v } } \mathcal { L } _ { \mathrm { d i v } } + \lambda _ { 1 1 } \mathcal { L } _ { \mathrm { l i v } } + \lambda _ { \mathrm { e n t } } \mathcal { L } _ { \mathrm { e n t } } . } \end{array}\tag{35}
$$

Here $\bar { \varpi } _ { z }$ is obtained by evaluating executable role z under the minibatch-mean semantic context. This keeps role diversity active even when a microbatch contains only one observed role. Since ${ \mathcal { L } } _ { \mathrm { e n t } }$ is negative entropy, minimizing it discourages premature slot concentration.

```latex
Component Setting
LoRA structure all-linear, rank $d = 8 ,$ scale 16
Adapted modules $L = 1 9 6$
Direct slot budget $S = 3 9 2 \ : \mathrm { o f } \ : L d = 1 5 6 8 \ : \mathrm { s l o t s }$
Router state hidden size 128, prefix window 256
Routing coeficients $\tau _ { b } = 0 . 8 , \lambda _ { u } = \bar { 0 } . 3 5 , \lambda _ { c } = 0 . 5$
Gate $\lambda _ { g } = 0 . 5 , \mathrm { r a n g e } [ 0 . 5 , 1 . 5 ]$
Router losses $\lambda _ { \mathrm { f u t u r e } } = 0 . 0 5 , \lambda _ { \mathrm { c r e d i t } } = 0 . 1 0$
Regularization $\lambda _ { \mathrm { l o a d } } = 0 . 1 0 , \lambda _ { \mathrm { d i v } } = 0 . 0 5$
Small regularizers $\lambda _ { \mathrm { l 1 } } = \lambda _ { \mathrm { e n t } } = 1 0 ^ { - 4 }$
Optimization $\beta _ { \mathrm { K L } } = 1 0 ^ { - 3 }$ , router gradient scale 1024
```  
Table 3: Role-routing architecture and optimization settings.

## Full Turn-Aligned GRPO Objective

For each task, GRPO samples G trajectories and computes

$$
\begin{array} { l } { \displaystyle \mu _ { R } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } R ( \tau ^ { ( g ) } , y ) , } \\ { \displaystyle \sigma _ { R } = \sqrt { \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \left( R ( \tau ^ { ( g ) } , y ) - \mu _ { R } \right) ^ { 2 } } , } \\ { \displaystyle \mathrm { A d v } ^ { ( g ) } = \frac { R ( \tau ^ { ( g ) } , y ) - \mu _ { R } } { \sigma _ { R } + \epsilon } . } \end{array}\tag{36}
$$

The actor-side turn return is normalized within each trajectory:

$$
\widehat { U } _ { t } ^ { ( g ) } = \left\{ \begin{array} { l l } { \displaystyle \frac { U _ { t } ^ { \mathrm { t u r n } , ( g ) } - \mu _ { U } ^ { ( g ) } } { \sigma _ { U } ^ { ( g ) } } , } & { \displaystyle \sigma _ { U } ^ { ( g ) } \geq 1 0 ^ { - 6 } , } \\ { \displaystyle \mathrm { c l i p } ( U _ { t } ^ { \mathrm { t u r n } , ( g ) } - \mu _ { U } ^ { ( g ) } , - 2 , 2 ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{37}
$$

Thus, a one-turn or constant-return trajectory contributes zero actor-side local credit. The router normalizer instead retains a raw singleton or zero-variance target before clipping:

$$
\mathcal { N } _ { \mathcal { B } _ { R } } ( v _ { q } ) = \left\{ \begin{array} { l l } { \displaystyle \frac { v _ { q } - \mu _ { \mathcal { B } _ { R } } } { \sigma _ { \mathcal { B } _ { R } } } , } & { | \mathcal { B } _ { R } | > 1 \mathrm { a n d } \sigma _ { \mathcal { B } _ { R } } \geq 1 0 ^ { - 6 } , } \\ { \displaystyle v _ { q } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{38}
$$

where $v _ { q } = U _ { t ( q ) } ^ { \mathrm { t u r n } , ( g ( q ) ) }$ . For response token $j$ at turn $t ,$ define

$$
\begin{array} { r l } & { \rho _ { t , j } ^ { ( g ) } = \frac { \pi _ { \theta , \theta _ { b } } ( \operatorname { t o k } _ { t , j } ^ { ( g ) } \mid o _ { t } ^ { ( g ) } , \operatorname { t o k } _ { t , < j } ^ { ( g ) } ) } { \pi _ { \theta _ { \mathrm { o l d } } , \theta _ { b , \mathrm { o l d } } } ( \operatorname { t o k } _ { t , j } ^ { ( g ) } \mid o _ { t } ^ { ( g ) } , \operatorname { t o k } _ { t , < j } ^ { ( g ) } ) } , } \\ & { \bar { \rho } _ { t , j } ^ { ( g ) } = \operatorname { c l i p } \left( \rho _ { t , j } ^ { ( g ) } , 1 - \epsilon _ { \mathrm { c l i p } } , 1 + \epsilon _ { \mathrm { c l i p } } \right) . } \end{array}\tag{39}
$$

The turn-aligned policy objective is

$$
\mathcal { I } _ { \mathrm { G R P O } } ^ { \mathrm { t u r n } } = \mathbf { E } _ { x , g , t , j } \left[ \operatorname* { m i n } \left( \rho _ { t , j } ^ { ( g ) } \widetilde { \mathrm { A d v } } _ { t , j } ^ { ( g ) } , \bar { \rho } _ { t , j } ^ { ( g ) } \widetilde { \mathrm { A d v } } _ { t , j } ^ { ( g ) } \right) \right] .\tag{40}
$$

We use $\lambda _ { \mathrm { d i s c } } = 0 . 9$ and $\alpha _ { c } = 0 . 3 5$ . Router-only gradients are multiplied by 1024 after backpropagation and before global clipping; LoRA gradients are not rescaled. This changes the router’s optimization scale but not the forward gate or the objective.

## Experimental Configuration

The primary MuSiQue and 2WikiMultiHopQA evaluations use the same team-search interface and metric implementation. ExRole-Shared and ExRole-Routed also use the same benchmark-specific warm start, reward, training horizon, and turn-aligned credit; their only architectural diference is the capacity-routing path. HotpotQA is reported separately as a full-Wikipedia retrieval stress test.

<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Primary</td><td>MuSiQue and 2WikiMultiHopQA; 200</td></tr><tr><td>benchmarks</td><td>held-out examples per benchmark</td></tr><tr><td>Stress test</td><td>HotpotQA with 200 held-out examples and a lexical full-Wikipedia index</td></tr><tr><td>Retrieval</td><td>Supporting-document collections for the primary benchmarks; top-3 results with</td></tr><tr><td>Team interface</td><td>640 characters per retrieved result Three agents, round-robin execution, at most six team turns, and 640 characters</td></tr><tr><td>Policy backbone</td><td>per retrieved result Qwen2.5-7B SFT with all-linear LoRA,</td></tr><tr><td>Training data</td><td>rank 8 and α = 16 512 benchmark-specific instances; batch size 4</td></tr><tr><td>Policy optimization</td><td>GRPO group size 8, learning rate 10−6, and 128 policy updates</td></tr><tr><td>Sequence budgets</td><td>Data prompt 4096, rollout prompt 6144, response 8192, and total model context</td></tr><tr><td>Role induction</td><td>16384 tokens Three source prototypes for MuSiQue; four for 2WikiMultiHopQA and the HotpotQA stress test; three executable</td></tr><tr><td>Capacity variants</td><td>roles after resolution Shared uses a uniform LoRA path; Routed directly selects 392 of 1568 module-rank slots and applies the centered sparse-delta gate</td></tr></table>

Table 4: Training and evaluation configuration for the primary benchmarks and the full-Wikipedia stress test. Routerspecific coeficients are reported in Table 3.

## Manual-role Control Prompt

The manual\_role control uses a fixed, ordered library of planner, solver, and verifier instructions. Its training configuration sets role\_mode: manual\_role and specialization\_mode: none. The evaluator resolves the three entries in the displayed order and applies the same per-turn team-search template, state fields, and action parser as ExRole. The implementation renders one prompt string per turn rather than separate API system messages; the box separates its shared context, manual instruction, and action contract for readability. Bracketed fields are filled from the common environment state.

## Manual-role control prompt

System/shared context. The common prompt begins with   
You are part of a {num\_agents}-agent   
team solving a structured search   
task.   
Question: {question}   
Current team turn:   
{turn}/{max\_turns}   
Current speaker: Agent {agent\_id}   
({role\_name})   
Your functional bias: {role\_prompt}

It then inserts the latest retrieved evidence, exact prior search queries, shared message board, and recent team history; it also appends the structured tag [ROLE\_ID=k], role-specific oper-

## Manual-role control prompt (continued)

ating rules, an optional task answer contract, and the common final-turn rule.

Manual instruction and name-dispatched   
rules. The role IDs follow the listed order.   
planner “Prioritize task decomposition, decide what evidence is still missing, and send concise coordination messages before issuing searches.” Additional rules: “Route the team away from duplicated work and name the next missing evidence gap concretely.” “Prefer one short message or one focused search over a broad speculative query.”

solver “Focus on targeted evidence gathering and drafting candidate answers from retrieved information.” Additional rules: “Synthesize the current evidence into one candidate answer or one narrower follow-up query.” “Avoid broad repeat searches; only search again when you can name the concrete missing fact.” “If the current evidence already points to one short plausible span, prefer <answer> over another broad search.”

verifier “Check consistency of the candidate answer against the gathered evidence and only finalize when the evidence is suficient.” Additional rules: “Audit candidate facts against the retrieved evidence and prefer <answer> over another <search> once one plausible short span appears.” “Only search again when you are resolving one explicit contradiction or one missing entity link.” “Do not leave the episode unfinished if the evidence already supports a grounded exact span.”

Action contract shared with ExRole. Each response must contain exactly one final environment action:

<search>your query</search>

<answer>your final answer</answer>

Before that action, the prompt permits only a very short <think>...</think> and one short <message>...</message>. Missing evidence requires <search>, suficient evidence requires <answer>, and an answer must be only the shortest exact evidence span rather than a sentence or explanation. Unsupported spans require another search, exact prior queries may be repeated only with a new disambiguating term, and malformed output without exactly one final action is penalized. On the final team turn, the shared rule requires <answer> rather than another search; a last-turn search is penalized and ends the episode without a final answer.

Thus the manual-role and ExRole conditions share the same team interface, state visibility, role-marker format, and action grammar. They difer in how the role instruction is obtained: manual-role uses the fixed library above, whereas ExRole derives executable role instructions from trajectories; only the routed ExRole variant additionally changes the capacity path.

## Role-Library Provenance and Split Isolation

For the MuSiQue role library used in the matched MuSiQue experiments, we conducted a provenance and split-isolation audit before policy optimization and evaluation. The frozen library was induced only from 64 train bootstrap trajectories, comprising 323 prefix records; the role encoder consumes prefix-local behavioral statistics rather than question text or retrieved document content. Against the 200-example validation set, we found zero overlap in task identifiers, normalized question strings, answer strings, hop identifiers, or gold supporting-document contents. Because both splits draw contextual passages from Wikipedia, 57 non-gold page titles and one non-gold raw passage recur across the two collections; the repeated passage does not contain the validation answer and is not a supporting document for the corresponding evaluation item. The static role artifact is frozen before policy training, loaded only for role assignment at episode initialization, and is not rebuilt or updated from evaluation rollouts. Future-evidence and return targets are computed solely from the source train trajectories. Thus, the audit establishes isolation at the task, answer, gold-evidence, and role-induction levels while distinguishing this guarantee from the unavoidable reuse of a common background corpus.

## Evaluation, Inference, and Optimization Details

## Limitations

Our primary evaluation focuses on supporting-evidence multi-hop question answering with a fixed team-search interface. Generalization to software engineering, web navigation, longer tool-use episodes, and larger teams remains untested. Appendix D.6 separately reports a full-Wikipedia retrieval stress test in which induced roles do not outperform the strongest controls.

ExRole induces its role library ofline from prior trajectories and keeps agent-role assignments fixed within an episode. A substantial shift in the task distribution, retrieva system, or tool interface may therefore require the role library to be refreshed.

The turn-level reward components are manually weighted, and routed capacity does not improve accuracy on every benchmark. Learned credit models, adaptive role assignment, and larger role libraries are promising directions for more robust specialization.

## Evaluation Metrics and Agreement Analysis

Let ${ \hat { y } } _ { n }$ be the final answer for evaluation example $n \in$ $\{ 1 , \ldots , N _ { \mathrm { e v a l } } \}$ and let ${ \mathcal { N } } _ { n }$ be its set of acceptable gold answers. The benchmark normalizer $n _ { \mathrm { e m } } ( \cdot )$ lowercases text, removes punctuation and English articles, and collapses whitespace. Per-example exact match is

$$
\mathrm { E M } _ { n } = \operatorname* { m a x } _ { y ^ { \prime } \in \mathcal { V } _ { n } } \mathbf { 1 } [ n _ { \mathrm { e m } } ( \hat { y } _ { n } ) = n _ { \mathrm { e m } } ( y ^ { \prime } ) ] .\tag{41}
$$

For token-level F1, let $\widehat { \tau } _ { n }$ and $\mathcal { T } _ { n } ^ { y ^ { \prime } }$ be the token multisets obtained from $n _ { \mathrm { e m } } ( \hat { y } _ { n } )$ and $n _ { \mathrm { e m } } ( \ddot { y } ^ { \prime } )$ . Their multiset intersection counts repeated tokens up to the smaller multiplicity. We compute

$$
\begin{array} { r l } & { \mathrm { O v } _ { n } ( y ^ { \prime } ) = | \widehat { \mathcal { T } } _ { n } \cap { \mathcal T } _ { n } ^ { y ^ { \prime } } | , } \\ & { \mathrm { P r e c } _ { n } ( y ^ { \prime } ) = \frac { \mathrm { O v } _ { n } ( y ^ { \prime } ) } { \operatorname* { m a x } ( 1 , | \widehat { T } _ { n } | ) } , } \\ & { \mathrm { R e c } _ { n } ( y ^ { \prime } ) = \frac { \mathrm { O v } _ { n } ( y ^ { \prime } ) } { \operatorname* { m a x } ( 1 , | { \mathcal T } _ { n } ^ { y ^ { \prime } } | ) } , } \\ & { \mathrm { F } 1 _ { n } = \underset { y ^ { \prime } \in \mathcal { V } _ { n } } { \operatorname* { m a x } } \frac { 2 \mathrm { P r e c } _ { n } ( y ^ { \prime } ) \mathrm { R e c } _ { n } ( y ^ { \prime } ) } { \mathrm { P r e c } _ { n } ( y ^ { \prime } ) + \mathrm { R e c } _ { n } ( y ^ { \prime } ) } , } \end{array}\tag{42}
$$

where the last fraction is defined as zero when both precision and recall are zero. Let $c _ { \mathrm { a n s } } ( \cdot )$ remove the structured action wrapper, common answer prefixes, surrounding whitespace, and terminal punctuation. The environment normalizer is the composition $n _ { \mathrm { s u c c } } = n _ { \mathrm { e m } } \circ c _ { \mathrm { a n s } }$ . Strict success compares this cleaned prediction with the benchmark-normalized gold answer, so substring matches receive no credit:

$$
\operatorname { S u c c } _ { n } = \operatorname* { m a x } _ { y ^ { \prime } \in \mathcal { V } _ { n } } \mathbf { 1 } _ { \operatorname { s u c c } } ( \hat { y } _ { n } , y ^ { \prime } ) .\tag{43}
$$

The reported dataset-level scores are percentages,

$$
( \mathrm { E M } , \mathrm { F 1 } , \mathrm { S u c c } ) = \frac { 1 0 0 } { N _ { \mathrm { e v a l } } } \sum _ { n = 1 } ^ { N _ { \mathrm { e v a l } } } ( \mathrm { E M } _ { n } , \mathrm { F 1 } _ { n } , \mathrm { S u c c } _ { n } ) .\tag{44}
$$

EM and Succ therefore need not be identical because only Succ applies the action-and-prefix cleanup $c _ { \mathrm { a n s } } .$ . We quantify their per-example agreement using

$$
D _ { \mathrm { S u c c , E M } } = \sum _ { n = 1 } ^ { N _ { \mathrm { e v a l } } } \mathbf { 1 } [ \mathrm { S u c c } _ { n } \neq \mathrm { E M } _ { n } ] .\tag{45}
$$

For the Shared and Routed evaluations on the two primary 200-example splits, $D _ { \mathrm { S u c c , E M } } = 0 \colon$ the independently computed rules happen to agree on those predictions. The same equality also holds for the two HotpotQA stress-test variants, whereas several HotpotQA control rows have nonzero disagreement because the additional cleanup changes a small number of decisions. We always compute EM, F1, and Succ independently rather than copying one column into another.

## Inference Procedure

At inference time, role induction is performed before evaluation rather than repeated for every sample. ExRole uses a role library induced from prior trajectories, assigns one role prompt to each agent, and then executes a team-search episode. The procedure is:

1. Load source prototypes $\mathcal { P } = \{ p _ { k } \} _ { k = 0 } ^ { K - 1 }$ from the induced role library.

2. Select and normalize the executable set $\mathcal { P } _ { \mathrm { e x e c } } .$ , resolve agent-level runtime roles $z _ { 1 } , \dots , z _ { N }$ , and insert marker $\chi _ { z _ { i } }$ into each agent observation.

3. At turn t, choose the active speaker by the team schedule, usually $i _ { t } = 1 + ( ( t - 1 )$ mod $N )$

4. Build the observation $o _ { t }$ from the task, active role, latest evidence, shared board, recent search queries, and recent team history.

5. During prefill, coalesce every role marker in the multiturn sequence and align each token to its role-turn segment; during cached decoding, retain the latest segment ID.

6. Pool the semantic prefix for the active segment and compute slot scores $\omega _ { q , \ell s }$ , soft budget $\tilde { b } _ { q , \ell s } , \mathrm { t o p } { - \cal S }$ mask $M _ { q , \ell s } ,$ and balanced sparse-delta multiplier $\gamma _ { q , \ell s }$

7. Generate the assistant response with the role-conditioned LoRA path.

8. Project the response to a structured action: search, answer, or invalid.

9. If the action is search, update evidence and continue. If the action is answer, terminate and score the answer.

## Gradient Flow through Balanced Sparse-Delta Routing

The forward pass treats the top-S mask as fixed while retaining a diferentiable soft budget on the selected slots. Define

$$
\begin{array} { r l } & { \delta _ { q , \ell s } ^ { \mathrm { s e l } } =  { M _ { q , \ell s } } \left( \frac { \tilde { b } _ { q , \ell s } } { \bar { b } _ { q } } - 1 \right) , } \\ & { \bar { \delta } _ { q , \ell s } = \delta _ { q , \ell s } ^ { \mathrm { s e l } } - \frac { 1 } { L d } \displaystyle \sum _ { \ell ^ { \prime } , s ^ { \prime } } \delta _ { q , \ell ^ { \prime } s ^ { \prime } } ^ { \mathrm { s e l } } , } \\ & { \gamma _ { q , \ell s } = \mathrm { c l i p } \big ( 1 + \lambda _ { g } \bar { \delta } _ { q , \ell s } , \gamma _ { \mathrm { m i n } } , \gamma _ { \mathrm { m a x } } \big ) . } \end{array}\tag{46}
$$

Away from clipping boundaries and while the discrete mask is unchanged, the softmax couples every routing score to every soft budget. The exact chain rule is

$$
\frac { \partial \gamma _ { q , \ell s } } { \partial \omega _ { q , \ell ^ { \prime } s ^ { \prime } } } = \lambda _ { g } \sum _ { \ell ^ { \prime \prime } = 1 } ^ { L } \sum _ { s ^ { \prime \prime } = 1 } ^ { d } \frac { \partial \bar { \delta } _ { q , \ell s } } { \partial \tilde { b } _ { q , \ell ^ { \prime \prime } s ^ { \prime \prime } } } \frac { \partial \tilde { b } _ { q , \ell ^ { \prime \prime } s ^ { \prime \prime } } } { \partial \omega _ { q , \ell ^ { \prime } s ^ { \prime } } } .\tag{47}
$$

Selected slots receive direct role-specific deviations, while centering propagates a compensating shift to all slots. The future-utility, role-turn-credit, load-balance, diversity, sparsity, and entropy objectives additionally act directly on the soft routing distribution. Consequently, router learning does not rely on diferentiating through the top-S membership itself.

## Training-Time Credit Flow

For a rollout group, GRPO first converts episode rewards into normalized trajectory advantages. In parallel, ExRole discounts the recorded turn rewards and aligns each return to the role-turn segment that produced it:

$$
\begin{array} { r l r } & { } & { \boldsymbol { x } \to \{ \tau ^ { ( g ) } \} _ { g = 1 } ^ { G } , } \\ & { } & { \{ \tau ^ { ( g ) } \} _ { g = 1 } ^ { G } \to \{ r _ { t } ^ { ( g ) } \} _ { g , t } \to \{ U _ { t } ^ { \mathrm { t u r n } , ( g ) } \} _ { g , t } , } \\ & { } & { \Big ( R ( \tau ^ { ( g ) } , y ) , U _ { t } ^ { \mathrm { t u r n } , ( g ) } \Big ) \to \widetilde { \mathrm { A d v } } _ { t , j } ^ { ( g ) } \to \mathcal { I } _ { \mathrm { G R P O } } ^ { \mathrm { t u r n } } , ~ } \\ & { } & { U _ { t } ^ { \mathrm { t u r n } , ( g ) } \to ( \mathcal { L } _ { \mathrm { f u t u r e } } , \mathcal { L } _ { \mathrm { c r e d i t } } ) , ~ } \end{array}\tag{48}
$$

The actor keeps the group-relative trajectory advantage and adds a weighted, normalized turn return only to the responsible role-turn segment. The router uses the same detached turn return as its future-utility and role-turn credit target. Thus a high-reward role turn reinforces both its generated action and its selected capacity path, while global role diversity, load balance, entropy, and sparsity regularizers prevent routing collapse.

## Turn-Aligned Routed GRPO Algorithm

Algorithm 2 summarizes the complete update procedure corresponding to the objectives in the main text.

Algorithm 2 Turn-aligned routed GRPO update   
☞ Input: Task x, gold answer $y ,$ role library P, policy θ, and   
router $\theta _ { b }$   
✓ Output: Updated policy θ and router $\theta _ { b }$   
1: ◆ Roll out teams. Sample $\{ \tau ^ { ( g ) } \} _ { g = 1 } ^ { G }$ and record turn re  
wards $\{ r _ { t } ^ { ( g ) } \}$   
2: ◆ Compute group credit. Normalize episode rewards into   
$\operatorname { A d v } ^ { ( g ) }$   
3: for each trajectory $\tau ^ { ( g ) }$ do   
4: Align role turns. Map every response token to its   
role-turn segment   
5: Discount local credit. Compute $U _ { t } ^ { \mathrm { t u r n } , ( g ) }$ and $\widehat { U } _ { t } ^ { ( g ) }$   
6: Route capacity. Pool $\zeta _ { q }$ and compute balanced gate   
γ<sub>q,ℓs</sub>   
7: Align policy credit. Form token advantages $\widehat { \mathrm { A d v } } _ { t , j } ^ { ( g ) }$   
8: end for   
9: ◆ Optimize. Compute $\mathcal { I } _ { \mathrm { G R P O } } ^ { \mathrm { t u r n } }$ and ${ \mathcal { L } } _ { \mathrm { r o u t e r } }$   
10: Form $\mathcal { L } _ { \mathrm { r o u t e d } } = \hat { \mathcal { L } } _ { \mathrm { s h a r e d } } + \mathcal { L } _ { \mathrm { r o u t e r } }$ and backpropagate   
11: Multiply router-only gradients by 1024, apply global gra  
dient clipping, and update $( \theta , \theta _ { b } )$   
12: return ✓ Updated $( \dot { \theta } , \theta _ { b } )$

## Additional Experimental Diagnostics

## Paired Routing Efects

Table 10 reports Shared and Routed scores together with 95% confidence intervals (CIs), diferences in percentage points (pp), and per-example win/tie/loss (W/T/L) counts over the same 200 examples. Figure 8 visualizes the corresponding diferences. The intervals show a small negative shift on MuSiQue and a small positive shift on 2WikiMultiHopQA; neither is statistically reliable across all metrics. HotpotQA is analyzed separately in Appendix D.6.

Paired effect of role-budgeted routing  
![](images/2137bdcacb6b1dfc276416707d48e55c8ddb229bf04dd1e9350a26a7bbb81df3.jpg)  
Routed - Shared (points; 95% paired bootstrap CI)  
Figure 8: Paired efect of role-conditioned sparse LoRA routing. Points show Routed minus Shared in percentage points; bars are 95% paired-bootstrap confidence intervals over 200 matched examples.

## Role Induction and Evaluation Diagnostics

Figure 13 jointly examines role-count sensitivity, cross-seed stability, and paired trajectory changes. Panel (a) shows that K = 3 remains competitive in silhouette, within-run stability, and support entropy, whereas $K = 4$ and K = 5 achieve slightly stronger clustering geometry. We therefore treat three roles as a compact, team-aligned operating point, not as the uniquely optimal partition. Panel (b) evaluates three $K = 3$ induction runs on the same 323 trajectory prefixes: adjusted Rand index (ARI) ranges from 0.61 to 0.85, normalized mutual information (NMI) from 0.62 to 0.82, and Hungarianaligned profile cosine from 0.80 to 0.97. Panel (c) provides a qualitative view of the largest positive and negative paired MuSiQue changes. These selected cases do not estimate the population-level routing efect; instead, they show that routing can alter both retrieval decisions and evidence-to-answer conversion. Taken together, the diagnostics support moderately stable role induction while showing that downstream routing efects remain example dependent.

## Role-Agent-Turn Disentanglement

We test whether an induced role captures a transferable behavior or merely renames a fixed agent index or speaking position. On the MuSiQue validation set, we independently permute three speaker orders and three agent-role assignments for both ExRole-Shared and ExRole-Routed. The resulting $3 \times 3 \times 2$ design contains 18 conditions and 3,600 episodes; within each variant, every role is enacted by every agent in early, middle, and late phases. This is an inference-time intervention on fixed checkpoints, not an additional training run.

Table 5 reports two complementary tests. First, adding role identity to an action model that already observes agent identity and phase reduces held-out log loss by 19.14% for Shared and 18.68% for Routed, while increasing macro-F1 by approximately 0.236. Second, the action distribution of the same role changes little across agents, whereas the mean distance between diferent roles is 48–60 times larger. Role identity therefore explains behavior beyond agent index and coarse turn phase. The schedule-level results also preserve the main Routed-versus-Shared finding: routing does not improve the average score under these interventions.

![](images/bbf997c265a98234461df5e80c93f29799e8f6db814e435eba01e69e2e48f039.jpg)

![](images/7d2a29f3aed514f76fe3e9a143914e49fc911cb0b1da7990a0e9e2b9264f56df.jpg)  
(b) Token F1 across orthogonal schedules  
Figure 9: Role-stage efects under orthogonal MuSiQue schedules. Small points are the three speaker-order and agentrole-map conditions associated with each role-stage pattern; diamonds and bars show the mean and one sample standard deviation. R, A, and V denote Researcher, Analyst, and Verifier.

<table><tr><td>Measure</td><td>Shared</td><td>Routed</td><td>Effect</td></tr><tr><td>Agent + phase log loss</td><td>0.4439</td><td>0.4408</td><td>control</td></tr><tr><td>+ Role log loss</td><td>0.3589</td><td>0.3585</td><td>-19.14%/ - 18.68%</td></tr><tr><td>+ Role macro-F1</td><td>0.6809</td><td>0.6808</td><td>≈+0.236</td></tr><tr><td>Same-role cross-agent JS</td><td>0.00446</td><td>0.00550</td><td>low variation</td></tr><tr><td>Different-role JS</td><td>0.2680</td><td>0.2652</td><td> $6 0 . 1 \times / 4 8 . 2 \times$ </td></tr><tr><td>Mean EM over nine schedules Mean F1 over nine schedules</td><td>29.22 40.99</td><td>28.33 39.97</td><td>−0.89 pp -1.02 pp</td></tr></table>

Table 5: Role-Agent-Turn disentanglement on MuSiQue. Action prediction uses 8,429 Shared and 8,423 Routed turns; schedule-level scores average nine orthogonal conditions per checkpoint. JS denotes Jensen–Shannon distance.

Figure 9 isolates the position of each role in the early, middle, and late phases while retaining all three orthogonal scheduling points. Moving the Analyst–Verifier–Researcher order to these phases lowers both EM and F1, whereas Verifier–Researcher–Analyst is strongest. Thus a role is not reducible to a fixed turn label, but its utility still depends on when it acts in the collaboration sequence.

Figure 10 gives the complete 3 × 3 schedule matrix rather than only phase-aggregated values. High- and lowperforming cells occur under multiple speaker orders and agent-role mappings, which rules out a single privileged agent assignment. At the same time, the structured variation across cells confirms that coordination order remains part of the task rather than a nuisance variable that can be ignored.

![](images/dd60616f6f322a7473a37c3defa3f6feb4fdd643783c90a0a6e598de8edc116e.jpg)

![](images/4f9cd3c5774f4ca432d009bb1810edd9522f37e53db432b6445164c94e2fed67.jpg)

![](images/fc6ff170b4f2e7a54a5a200f5fcdd58eefe0a9108423a4fda34bcdfa3826f5d2.jpg)

![](images/c79524b028a8aba28d4e565743cb41cbbe91aeca1173a19c1ca04db5dd9c3eb6.jpg)  
Figure 10: Complete Role-Agent-Turn intervention matrix on MuSiQue. Rows vary speaker order and columns vary agent-role assignment; cells report EM or F1 for the same 200 questions. Shared and Routed use identical schedules and evaluation protocol.

## Complementarity of Role Induction and Turn-Aligned Credit

We isolate the two learning components with an independently trained 2×2 design under the shared-capacity setting. Among the four matched runs, jointly enabling trajectoryinduced roles and turn-aligned credit achieves the highest observed result, reaching 30.0 EM and 41.5 F1. Relative to this joint configuration, removing role induction reduces EM and F1 by 3.5 and 2.3 points, respectively, while removing turn-aligned credit decreases F1 by 2.0 points. This pattern is consistent with complementary functions: induced role structure organizes collaborative behavior, while turnaligned credit connects that structure to policy updates.

<table><tr><td>Configuration</td><td>Role</td><td>Credit</td><td>EM↑</td><td>F1↑</td></tr><tr><td>Full role-credit</td><td>√</td><td>√</td><td>30.0</td><td>41.5</td></tr><tr><td>w/o role induction</td><td>x</td><td>√</td><td> $2 6 . 5 _ { \ : ( - 3 . 5 ) }$ </td><td> $3 9 . 2 _ { ( - 2 . 3 ) }$ </td></tr><tr><td>w/o turn-aligned credit</td><td>√</td><td>x</td><td> $2 9 . 5 _ { ( - 0 . 5 ) }$ </td><td> $3 9 . 5 _ { ( - 2 . 0 ) }$ </td></tr><tr><td>w/o both</td><td>x</td><td>x</td><td> $2 9 . 0 _ { ( - 1 . 0 ) }$ </td><td> $4 0 . 3 _ { \ : ( - 1 . 2 ) }$ </td></tr></table>

Table 6: Strict $2 \times 2$ component ablation on MuSiQue validation-200. The four variants are independently trained with seed 42 and share the same backbone, 128-update horizon, and evaluation protocol. Parenthesized values are changes relative to the complete role-credit configuration.

## Inference Cost and Answer Diagnostics

Table 9 separates computational budget from answer quality on the same 200 MuSiQue examples. Shared and Routed have nearly identical turn, search, retrieval, and repetition statistics. Target-bearing evidence is available substantially more often than a strictly correct answer is produced, identifying final answer formation as a major remaining bottleneck.

## Full-Wikipedia HotpotQA Stress Test

HotpotQA difers from the two primary benchmarks by retrieving from a lexical full-Wikipedia index rather than a bounded supporting-document collection. We therefore report it as a retrieval stress test rather than use it as primary evidence for the role-learning claim. The team interface still uses three agents, six turns, top-3 retrieval, and the same EM/F1/Succ evaluation implementation.

<table><tr><td>Method</td><td>EM↑</td><td>F1↑</td><td>Succ. ↑</td></tr><tr><td>Single-agent search</td><td>27.0</td><td>36.0</td><td>27.0</td></tr><tr><td>No-role MAS</td><td>33.5</td><td>44.4</td><td>33.0</td></tr><tr><td>Manual-role MAS</td><td>33.0</td><td>42.9</td><td>32.5</td></tr><tr><td>Random role prompt</td><td>34.5</td><td>43.9</td><td>34.0</td></tr><tr><td>Shuffled induced role</td><td>33.5</td><td>42.4</td><td>33.5</td></tr><tr><td>ExRole-Shared</td><td>30.0</td><td>38.3</td><td>30.0</td></tr><tr><td>ExRole-Routed</td><td>30.5</td><td>37.8</td><td>30.5</td></tr></table>

Table 7: HotpotQA results under the full-Wikipedia FTS5 retrieval stress test. Succ. is strict success, computed independently from EM.

Neither ExRole variant surpasses the strongest role-free or prompt-role controls in this setting. Routed changes Shared by +0.5 EM with a 95% CI of $[ 0 . 0 , + 1 . 5 ]$ and $\mathrm { { b y \mathrm { ~ - 0 . 5 ~ } } }$ F1 with a CI of $[ - 1 . 4 , + 0 . 3 ]$ , indicating negligible routing sensitivity. These results establish a boundary of the current method: role specialization alone does not overcome the retrieval and answer-selection constraints ofthis full-Wikipedia configuration. They do not, by themselves, isolate retrieval quality as the causal source of the gap.

![](images/3d18b51b442782938b22996db221658cc027c13979ce64e631dfac71d29c31d7.jpg)

![](images/2c640cbf1d8b3b9a6b6b920e1dfd952d2a3e1fe7ad4192570312221846056b7c.jpg)  
(b) Shared/Routed turn credit.

HotpotQA Routed decoding robustness  
![](images/31d71e1bf2e09b8a2408988ea90273485cc8aac176c2bcb1259dd2dd559d9aad.jpg)  
(c) Routed decoding robustness over three seeds.  
Figure 11: Optimization and decoding diagnostics for the HotpotQA full-Wikipedia stress test. Panels (a) and (b) use the matched 128-update Shared/Routed runs; panel (c) reports three temperature-1 evaluations of a fixed Routed checkpoint on the same 200 questions.

<table><tr><td>Method</td><td>Base</td><td>EM↑</td><td>F1↑</td></tr><tr><td>Retrieval baselines</td><td></td><td></td><td></td></tr><tr><td>Direct</td><td>★</td><td>16.5</td><td>24.6</td></tr><tr><td>Naive RAG</td><td>★</td><td>20.5</td><td>26.3</td></tr><tr><td>Search-RL baselines</td><td></td><td></td><td></td></tr><tr><td>Search-R1</td><td></td><td>27.5</td><td>35.2</td></tr><tr><td>R1-Searcher</td><td></td><td>27.5</td><td>35.4</td></tr><tr><td>MAS-RL baselines</td><td></td><td></td><td></td></tr><tr><td>MAGRPO</td><td>★</td><td>23.0</td><td>29.2</td></tr><tr><td>Dr. MAS</td><td>★</td><td>27.0</td><td>34.4</td></tr><tr><td>MATPO</td><td>★</td><td>33.5</td><td>43.7</td></tr><tr><td>ExRole-Shared</td><td>★</td><td>30.0</td><td>38.3</td></tr><tr><td>ExRole-Routed</td><td>★</td><td>30.5</td><td>37.8</td></tr></table>

Table 8: HotpotQA external references under each method’s stated protocol. Retrieval, training, and evaluation details difer across rows, so the values provide context rather than a controlled ranking.

## External Reference Training Curves

We adapt MAGRPO and Dr. MAS to MuSiQue, HotpotQA, and 2WikiMultiHopQA with the same Qwen2.5-7B SFT policy backbone. Figure 12 combines their learning dynamics: MAGRPO uses a sequential three-agent search interface and grouped updates over 512 training instances, while Dr. MAS uses shared-model grouped rollouts over 128 updates with an 8192-token context budget. Panel (a) reports MAGRPO reward, policy loss, and final 200-example EM/F1; panel (b) reports Dr. MAS reward, valid-action ratio, and native periodic validation. The horizontal axis is normalized within each run, so the curves verify non-degenerate optimization but do not equate method-specific objectives or logging schedules.

Training progress (%)
<table><tr><td>Diagnostic group</td><td>Quantity</td><td>Unit</td><td>Shared</td><td>Routed</td><td>∆</td></tr><tr><td rowspan="4">Resource use</td><td>Team turns</td><td>count</td><td> $4 . 4 8 \pm 1 . 5 0 ; 3 . 0 [ 3 . 0 , 6 . 0 ]$ </td><td> $4 . 4 9 \pm 1 . 5 0 ; 4 . 0 [ 3 . 0 , 6 . 0 ]$ </td><td>+0.02</td></tr><tr><td>Search calls</td><td>count</td><td> $3 . 4 8 \pm 1 . 5 0 ; 2 . 0 [ 2 . 0 , 5 . 0 ]$ </td><td> $3 . 4 9 \pm 1 . 5 0 ; 3 . 0 [ 2 . 0 , 5 . 0 ]$ </td><td>+0.02</td></tr><tr><td>Retrieved text</td><td>kchars</td><td> $5 . 3 5 \pm 3 . 4 2 ; 4 . 5 0 [ 2 . 5 2 , 7 . 3 0 ]$ </td><td> $5 . 3 5 \pm 3 . 3 4 ; 4 . 5 6 [ 2 . 5 3 , 7 . 4 1 ]$ </td><td>-0.01</td></tr><tr><td>Repeated queries</td><td>count</td><td> $0 . 3 3 \pm 0 . 8 0 ; 0 . 0 [ 0 . 0 , 0 . 0 ]$ </td><td> $0 . 3 6 \pm 0 . 8 3 ; 0 . 0 [ 0 . 0 , 0 . 0 ]$ </td><td>+0.03</td></tr><tr><td rowspan="5">Answer stages</td><td>Search issued</td><td>episodes (%)</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Target retrieved</td><td>episodes (%)</td><td>73.5</td><td>73.0</td><td>-0.5</td></tr><tr><td>Gold span visible</td><td>episodes (%)</td><td>73.5</td><td>73.5</td><td>0.0</td></tr><tr><td>Grounded answer</td><td>episodes (%)</td><td>87.0</td><td>87.0</td><td>0.0</td></tr><tr><td>Strict success</td><td>episodes (%)</td><td>31.5</td><td>30.0</td><td>-1.5</td></tr></table>

Table 9: Matched MuSiQue inference and answer diagnostics over the same 200 examples. Resource-use cells report mean±standard deviation (SD) followed by median [interquartile range (IQR)]; stage rows report percentages. ∆ is ExRole-Routed minus ExRole-Shared. Stage conditions are evaluated independently and do not form a monotone funnel.

![](images/230173e38ee30b899455be3f9b3edb578cb4f311822b586e27b169829e46a845.jpg)

![](images/c487b41780a9f45f25ff5e387fd1a92fb6c87ae8fea093b043b38181fed6dbed.jpg)

![](images/be4bc632156bc2704bf23237b751a0fe0f42707f0136e43de9f4e0b4328abe09.jpg)  
(a) MAGRPO training dynamics and final evaluation.

![](images/9c29898554a46c23335aec80193ad6be5f84469557b87e20b5982594d535a5cb.jpg)

![](images/942d92917fd4852df5292d0263b0b7066350860c50cfdee5e579eca015bf4263.jpg)

![](images/b8026f2e31943deb3115b1ea820b0608fa53998a898062075087d7dad94f6c36.jpg)  
(b) Dr. MAS training dynamics and periodic validation.

![](images/108e4765fbb81934e717f58b7b13f74f10e21d521860157dbe4854bb4652ddec.jpg)

![](images/a2889cdf4d9e05321844c87fbc32518b7fa1cf48cb30925cdca6fbd1c00355c9.jpg)

Figure 12: External MAS-RL baseline reproductions on MuSiQue, HotpotQA, and 2WikiMultiHopQA. Both panels use normalized training progress on the horizontal axis. MAGRPO reports reward, policy loss, and final 200-example EM/F1, while Dr. MAS reports reward, valid-action ratio, and its native periodic validation measurements.

<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Metric</td><td colspan="2">Score</td><td colspan="3">Paired Routed – Shared effect</td></tr><tr><td>Shared</td><td>Routed</td><td>∆(pp)</td><td>95% CI</td><td>W/T/L</td></tr><tr><td rowspan="3">MuSiQue</td><td>EM</td><td>31.5</td><td>30.0</td><td>-1.5</td><td> $[ - 4 . 0 , + 0 . 5 ]$ </td><td>1/195/4</td></tr><tr><td>F1</td><td>43.2</td><td>41.5</td><td>-1.8</td><td> $[ - 4 . 2 , + 0 . 3 ]$ </td><td>2/193/5</td></tr><tr><td>Succ.</td><td>31.5</td><td>30.0</td><td>–1.5</td><td> $[ - 4 . 0 , + 0 . 5 ]$ </td><td>1/195/4</td></tr><tr><td rowspan="3">2Wiki</td><td>EM</td><td>49.0</td><td>50.0</td><td>+1.0</td><td> $[ - 1 . 5 , + 3 . 5 ]$ </td><td>4/194/2</td></tr><tr><td>F1</td><td>59.1</td><td>59.7</td><td>+0.7</td><td> $[ - 0 . 8 , + 2 . 3 ]$ </td><td>5/191/4</td></tr><tr><td>Succ.</td><td>49.0</td><td>50.0</td><td>+1.0</td><td> $[ - 1 . 5 , + 3 . 5 ]$ </td><td>4/194/2</td></tr></table>

Table 10: Paired Shared-versus-Routed comparison over 200 examples for each primary benchmark. ∆ is measured in percentage points; intervals are 95% paired-bootstrap confidence intervals with 20,000 resamples. W/T/L counts per-example Routed wins, ties, and losses.

![](images/8474ca3274adc9571267b5fda3b9504ea35586689e3f65809f5ddc32ed95cd6f.jpg)  
(a) Role-count sensitivity over three discovery seeds.

![](images/7d189592884f911c4f4e8b19935d6a3c363999b9d7688c4c1004cf5d0822ac6c.jpg)

![](images/798aa8dae1a695db3e8d0a0fadf567219977c531fc43bbcd72e5d37db5239529.jpg)  
(c) Extreme paired MuSiQue trajectory changes.  
(b) Cross-seed role agreement for $K = 3 .$  
Figure 13: Role-induction and evaluation diagnostics. Panel (a) evaluates sensitivity to the number of induced roles; panel (b) measures cross-seed agreement of the induced partition and role profiles; and panel (c) traces paired examples with the largest Routed-versus-Shared F1 changes. Error bars denote one sample standard deviation where applicable.

Symbol Table  
Notation
<table><tr><td>Symbol</td><td>Meaning</td><td>Symbol</td><td>Meaning</td></tr><tr><td> $x , y , \tau$ </td><td>Task input, gold answer, and team trajectory</td><td> $S , L , d$ </td><td>Slot budget, adapted linear modules, and LoRA rank</td></tr><tr><td> $N$ </td><td>Number of agents in the team</td><td> $s$ </td><td>LoRA rank-slot index,  $1 \leq s \leq d$ </td></tr><tr><td> $T , T _ { \tau }$ </td><td>Maximum and realized numbers of team turns</td><td> $\omega _ { q , \ell s }$ </td><td>Score for module l and rank slot s in segment</td></tr><tr><td> $J , K$ </td><td>Numbers of induction records and source clusters</td><td> $\varpi _ { q , \ell s }$ </td><td>Normalized routing mass over LoRA rank</td></tr><tr><td> $G , Q$ </td><td>GRPO group size and number of role-turn</td><td> $\tilde { b } _ { q , \ell s }$ </td><td>slots Soft slot budget before hard top-S selection</td></tr><tr><td> $i _ { t }$ </td><td>segments Active speaker at turn t</td><td> ${ \cal M } _ { q , \ell s }$ </td><td>Hard top-S slot mask</td></tr><tr><td> $o _ { t }$ </td><td>Text observation given to the active agent</td><td> $\gamma _ { q , \ell s }$ </td><td>Effective LoRA multiplier after sparse-delta routing</td></tr><tr><td> $a _ { t } , m _ { t }$ </td><td>Structured action and optional teammate mes-</td><td> $c _ { q } , \widehat { c } _ { q }$ </td><td>Detached turn-credit target and router predic-</td></tr><tr><td> $e _ { t }$ </td><td>sage Retrieved evidence after a search action</td><td> $\scriptstyle { B _ { R } }$ </td><td>tion Role-turn segments in the current normaliza-</td></tr><tr><td> $z _ { i } , z _ { q } , z ( u )$ </td><td>Runtime role of an agent, segment, or token</td><td> $\theta , \theta _ { e } , \theta _ { b }$ </td><td>tion batch Policy, role-encoder, and router parameters</td></tr><tr><td> $\xi _ { t } , \phi _ { t } ^ { \mathrm { o b s } }$ </td><td>Learned role embedding and prefix-local be- havior feature vector</td><td> $\tau _ { b } , \lambda _ { \mathrm { d i s c } }$ </td><td>Routing temperature and turn-credit discount</td></tr><tr><td>pk</td><td>Induced source prototype for cluster k</td><td> $\mathrm { A d v } ^ { ( g ) } , \rho _ { t , j } ^ { ( g ) }$ </td><td>Group advantage and token probability ratio</td></tr><tr><td> $\eta _ { k }$ </td><td>Action, stage, evidence, and return statistics for cluster k</td><td> $U _ { t } ^ { \mathrm { t u r n } } , \alpha _ { c }$ </td><td>Discounted turn return and policy-credit mix- ing weight</td></tr><tr><td> $\psi _ { z } , \chi _ { z }$ </td><td>Readable role instruction and token-aligned role marker</td><td> $R ( \tau , y )$ </td><td>Episode reward against gold answer y</td></tr><tr><td> $q , \mathrm { s e g } ( u )$ </td><td>Role-turn segment and token-to-segment map</td><td> $n _ { \mathrm { e m } } , n _ { \mathrm { s u c c } }$ </td><td>Benchmark and strict environment normaliz-</td></tr><tr><td> $\Gamma _ { t } , \Gamma _ { q } , \Gamma ( u )$ </td><td>Adapter-routing state at turn, segment, or to- ken level</td><td> $N _ { \mathrm { { e v a l } } } , \mathrm { { S u c c } }$ </td><td>ers Evaluation-set size and strict success rate</td></tr><tr><td> $\mathbf { v } _ { z _ { q } } , \zeta _ { q }$ </td><td>Executable-role feature and pooled semantic prefix</td><td></td><td></td></tr></table>

Table 11: Symbols used throughout the method, training objective, and evaluation protocol.
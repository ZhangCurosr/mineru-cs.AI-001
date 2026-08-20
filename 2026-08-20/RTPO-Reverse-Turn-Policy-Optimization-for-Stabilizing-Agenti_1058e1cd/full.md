# RTPO: Reverse-Turn Policy Optimization for Stabilizing Agentic RL Training

Yugu Li   
School of CSIT   
Adelaide University   
Adelaide, SA 5000, Australia   
yugu.li@adelaide.edu.au Jianglin Qiao ACFR The University of Sydney   
Camperdown, NSW 2050, Australia   
jianglin.qiao@sydney.edu.au   
Jimmy Cao   
School of CSIT   
Adelaide University   
Adelaide, SA 5000, Australia   
jimmy.cao@adelaide.edu.au   
Siyi Hu   
School of EECMS   
Curtin University   
Bentley, WA 6102, Australia   
siyi.hu@curtin.edu.au

## Abstract

Training multi-turn agentic workflows with reinforcement learning (RL) enables large language models to perform complex reasoning, use external tools, and conduct iterative search beyond single-turn settings. Yet multi-turn RL training remains highly unstable, often causing severe performance degradation as the number of turns increases. Through theoretical analysis, we identify three tightly coupled sources of instability: rollout–training context mismatch, weak turn-level credit assignment under sparse terminal rewards, and asynchronous policy drift when short and long trajectories are optimized under different policy versions. We show that these issues share a common structural origin in flattened trajectory optimization and address them through a unified reverse-turn formulation. We propose Reverse-Turn Policy Optimization (RTPO), which organizes multi-turn rollouts as sparse reverse trees and performs turn-level policy updates in temporal reverse order, aligning each decision with its downstream continuation. RTPO enables causally consistent turn-level credit assignment and on-policy continuation to control asynchronous drift. We provide theoretical guarantees showing that RTPO eliminates context mismatch and asynchronous drift under the proposed turn-level formulation, reduces credit bias, and converges to recursive optimality. Experiments on multi-turn agentic RL benchmarks show that RTPO improves upon trajectory- and turn-level baselines by 21.50% and 10.76%, respectively, highlighting its potential to support more stable training for tool-using agents.

## 1 Introduction

Reinforcement learning (RL) has become a central paradigm for post-training large language models (LLMs), especially when supervision comes from final outcome rewards rather than dense tokenlevel labels. Through outcome-based optimization, RL encourages behaviors such as planning, self-reflection, and verification, achieving strong results in single-turn mathematical reasoning [Shao et al., 2024, Yang et al., 2024, Guo et al., 2025, Li et al., 2026] and code generation [Le et al., 2022, Shojaee et al., 2023, Team et al., 2025]. Motivated by these advances, recent work extends RL to multi-turn agentic workflows, especially Tool-Integrated Reasoning (TIR) [Yao et al., 2023, Gao et al., 2023, Shinn et al., 2023, Chen et al., 2026, Chang et al., 2026, Liu et al., 2026], where agents iteratively reason, call tools, receive feedback, and refine their behavior across turns.

Despite this promise, RL training for multi-turn workflows remains unstable: models that improve on short workflows often degrade as the number of turns increases. This degradation is not merely due to longer sequences; it is amplified by temporal dependencies across turns, where each decision reshapes the context, environment state, and future decision distribution. Existing methods partially mitigate this issue but provide limited analysis of its causes. Trajectory-level methods such as PPO [Schulman et al., 2017], GRPO [Shao et al., 2024], and GSPO [Zheng et al., 2025] retain the flattened trajectory paradigm. Turn-decomposition methods such as SeeUPO [Hu et al., 2026] improve update granularity but still condition on flattened histories. Tree-based methods such as TreeGRPO [Ji et al., 2026] and ARPO [Dong et al., 2026] use shared prefixes or branching rollouts, but their advantage estimates are not fully aligned with each turn’s causal contribution to downstream continuation. Overall, the sources of instability remain underexplored at the training-pipeline level, and existing methods (see Appendix A for details) do not jointly address their shared origin.

In this paper, we provide a theoretical analysis that identifies three coupled sources of instability: context mismatch between rollout and training, which breaks consistency between generated and optimized turn-level contexts; weak turn-level credit assignment, where sparse terminal rewards obscure the contribution of individual decisions; and asynchronous policy drift, where short and long trajectories are optimized under different versions of an evolving policy. Although these issues arise from different components of the training pipeline, we show that they share a common structural origin. We illustrate these sources in Figure 1 and analyze them formally in Sec. 2.

Core Challenge. Motivated by our theoretical analysis, we ask: How can we improve multiturn agentic RL performance by stabilizing turn-wise training with rollout–training consistency, turn-level credit assignment, and controlled asynchronous policy drift?

To address this challenge, we propose Reverse-Turn Policy Optimization (RTPO), a policy optimization framework for stabilizing multi-turn agentic RL training, with theoretical details provided in Sec. 3. The key idea is to formalize sampled interactions as sparse trees and optimize turn-level policies in temporal reverse order, propagating continuation values from later turns to earlier ones through the reverse optimality guarantee. By constructing sibling continuations for each turn, RTPO estimates turn-level advantages under matched downstream conditions. This removes context inconsistency induced by flattened trajectory optimization, reduces turn-level credit bias through causal action alignment, and controls asynchronous policy drift through on-policy continuation.

Our contributions are fourfold: (i) We identify coupled sources of instability in multi-turn agentic RL: context mismatch, weak turn-level credit, and asynchronous policy drift. (ii) We provide a theoretical analysis showing that these sources share a common structural origin in flattened trajectory optimization. (iii) We propose RTPO, a reverse-turn policy optimization framework with sparse reverse trees, turn-level on-policy updates, and theoretical guarantees on recursive optimality, context consistency, and reduced credit bias. (iv) We validate RTPO on multi-turn agentic RL benchmarks, where it outperforms strong baselines, including GRPO, TreeGRPO, ARPO, and SeeUPO, while further stabilizing the training pipeline.

## 2 Theoretical Analysis: Training Instability

We argue that the instability of multi-turn RL training stems from rollout–training mismatch: rollouts are typically generated under truncated or summarized contexts, while training recomputes likelihood ratios under concatenated full-history contexts. This discrepancy biases policy optimization and worsens over long horizons. Moreover, the flattened full-history formulation provides only trajectorylevel credit, causing terminal rewards to obscure individual turns. When trajectories are generated asynchronously, this mismatch further induces policy drift, as long trajectories generated under an older policy may be optimized after the policy has already been updated by shorter trajectories.

Building on these observations and insights, we model a multi-turn interaction as a hierarchical Markov decision process (H-MDP) [Hauskrecht et al., 1998]. Given an initial prompt $q ,$ turn $k$ consists of a model response $l _ { k }$ and environment feedback $f _ { k }$ , producing the trajectory $( q , l _ { 0 } , f _ { 0 } , \ldots , l _ { n - 1 } , f _ { n - 1 } )$ . The turn-level state is $S _ { k } = ( q , l _ { 0 } , f _ { 0 } , \ldots , \bar { l _ { k - 1 } , f _ { k - 1 } } )$ , and the turn-level action is the response $l _ { k }$ . Each response is generated autoregressively as $l _ { k } = \left( a _ { k , 1 } , \ldots , a _ { k , T _ { k } } \right)$ where the token-level state is $s _ { k , t } = ( S _ { k } , a _ { k , < t } )$ . Existing PPO- and GRPO-style methods [Shao et al., 2024, Zheng et al., 2025, Yue et al., 2025, Yu et al., 2026b] typically flatten the full interaction into a single token sequence and optimize the resulting trajectory as follows:

![](images/3fc0c569a7dbf6a276249bc5729c03a9d40d7f6c75292ce741709dd103b4d2de.jpg)  
Figure 1: The identified training instability in multi-turn agentic RL arises from: (A) rollout–training context mismatch, (B) trajectory-only credit assignment, and (C) long-horizon policy drift.

$$
J ^ { \mathrm { { d a t } } } ( \theta ) = \mathbb { E } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { \sum _ { t } m _ { i , t } } \sum _ { t } m _ { i , t } \operatorname* { m i n } ( \rho _ { i , t } A _ { i } , \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon ) A _ { i } ) \right] ,\tag{1}
$$

where $\rho _ { i , t } = \pi _ { \theta } ( a _ { i , t } \mid x _ { i , < t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { i , t } \mid x _ { i , < t } )$ is the token-level importance-sampling (IS) ratio and $A _ { i }$ is a trajectory-level advantage assigned uniformly to all unmasked tokens in trajectory g<sub>i</sub> (g<sub>i</sub> belongs to a group of G trajectories). Full preliminaries are provided in Appendix B.

## 2.1 Rollout–Training Mismatch

As illustrated in Figure 1-A, in multi-turn interactions, rollouts are often generated under a truncated or summarized context $\phi ( { \bar { x } } _ { k } )$ , while training recomputes token probabilities under the full flattened history $\bar { x } _ { k }$ . Thus, the IS ratio used in training differs from the true IS ratio induced by the rollout distribution:

$$
\begin{array} { r } { \rho _ { k , t } ^ { \mathrm { f a t } } = \frac { \pi _ { \theta } \left( a _ { k , t } \mid \bar { x } _ { k } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { k , t } \mid \phi \left( \bar { x } _ { k } \right) \right) } \neq \frac { \pi _ { \theta } \left( a _ { k , t } \mid \phi \left( \bar { x } _ { k } \right) \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { k , t } \mid \phi \left( \bar { x } _ { k } \right) \right) } = \rho _ { k , t } ^ { \mathrm { t r u e } } . } \end{array}\tag{2}
$$

Because the denominator in $\rho _ { k , t } ^ { \mathrm { f l a t } }$ does not match the distribution that actually sampled the token, the resulting policy-gradient estimate is biased. This mismatch becomes more severe in later turns as the omitted history grows. Moreover, when ϕ is non-injective, distinct full-history states can collapse into the same truncated observation, inducing state aliasing and restricting optimization to an observation-induced policy class whose optimum may be strictly below the full-history optimum.

See Appendix B.1 for the full theoretical analysis of rollout–training mismatch.

## 2.2 Trajectory-Only Credit Assignment

Flattened training assigns a single trajectory-level advantage to all turns, even though different turns may contribute unequally to the final outcome, as shown in Figure 1-B. For trajectory $g _ { i }$ , the population trajectory advantage can be decomposed at turn k as

$$
R _ { i } - \mathbb { E } [ R ] = \underbrace { R _ { i } - Q ^ { \pi } ( S _ { i , k } , l _ { i , k } ) } _ { \mathrm { d o w n s t r e a m ~ s t o c h a s t i c i t y } } + \underbrace { A ^ { \pi } ( S _ { i , k } , l _ { i , k } ) } _ { \mathrm { t r u e ~ t u r n ~ a d v a n t a g e } } + \underbrace { V ^ { \pi } ( S _ { i , k } ) - \mu _ { R } } _ { \mathrm { u p s t r e a m ~ s t a t e ~ e f f e c t } } .\tag{3}
$$

Here, $R _ { i }$ denotes the final return of trajectory $g _ { i } , Q ^ { \pi } ( S _ { i , k } , l _ { i , k } ) = \mathbb { E } [ R \ | \ S _ { i , k } , l _ { i , k } ] , V ^ { \pi } ( S _ { i , k } ) =$ $\mathbb { E } [ R \mid S _ { i , k } ]$ , and $\mu _ { R } = \mathbb { E } [ R ]$ . This decomposition shows that the true turn-level advantage is only one component of the trajectory-level signal. When upstream state effects or downstream stochasticity dominate, the sign of the trajectory advantage may disagree with the true turn-level advantage, leading to incorrect or even reversed policy updates. Similarly, group-relative baselines provide valid local comparisons only when trajectories share the same turn-level state. Under cross-state grouping, the advantage estimator incurs additional bias $\begin{array} { r } { \mathrm { B i a s } _ { \mathrm { c r o s s } } = \frac { G - 1 } { G } \left( V ^ { \pi } ( S _ { i , k } ) - \mu _ { R } \right) } \end{array}$ , determined by the upstream trajectory without causal connection to the current action.

See Appendix B.2 for the full theoretical analysis of trajectory-only credit assignment issues.

## 2.3 Long-Horizon Policy Drift

In asynchronous training with parallel sampled trajectories (see Figure 1-C), shorter trajectories may complete and update the policy while longer trajectories are still being generated under an older policy. When these longer trajectories are later optimized, they become off-policy with respect to the current model. The unbiased correction would require the full-trajectory importance weight $\begin{array} { r } { \omega _ { i } = \prod _ { t = 1 } ^ { T _ { i } } \rho _ { i , t } } \end{array}$ . However, the PPO or GRPO method applies clipping independently at the token level, so the resulting correction differs from the true full-trajectory IS ratio:

$$
\prod _ { t = 1 } ^ { T _ { i } } \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon ) \neq \prod _ { t = 1 } ^ { T _ { i } } \rho _ { i , t } = \omega _ { i } .\tag{4}
$$

Therefore, token-level clipping cannot faithfully correct long-horizon policy drift. The discrepancy compounds with trajectory length, while using the exact full-trajectory ratio is impractical because its variance grows rapidly with $\bar { T _ { i } }$ . This explains why standard PPO- or GRPO-style training methods can become unstable or even collapse in long-horizon agentic RL.

The full theoretical analysis of long-horizon policy drift is provided in Appendix B.3.

## 3 Method: Reverse-Turn Policy Optimization (RTPO)

To address the training instability revealed by our theoretical analysis in Sec. 2, we propose Reverse-Turn Policy Optimization (RTPO) with theoretical guarantees for stabilizing agentic RL training, as shown in Figure 2. RTPO first models multi-turn interaction as a turn-boundary MDP and defines an independent sub-policy optimization objective for each turn (Sec. 3.1). This formulation enables reverse-order training to mitigate the mismatch between rollout and training contexts. RTPO then develops sparse tree rollouts based on maximum-value decomposition (Sec. 3.2) to estimate true turn-level advantages, enabling causally consistent turn-level credit assignment for the case of cross-trajectory comparison without state bias. Finally, RTPO designs an on-policy continuation mechanism (Sec. 3.3) that eliminates the need for downstream IS-ratio correction under PPO clipping, thereby addressing policy drift induced by asynchronous turns.

## 3.1 Turn-Level Policy Optimization

Turn-boundary MDP for rollout-training match. We model a K-turn agent episode as a turnboundary MDP $\mathcal { M } = \langle \bar { \mathcal { X } } , \mathcal { A } _ { H } , P _ { H } , R _ { H } , \bar { \mathcal { Y } _ { H } } \rangle$ , where the augmented state $\bar { \boldsymbol { x } } _ { k } = \left( S _ { k } , k \right)$ encodes the interaction history $\dot { S } _ { k }$ and the turn index k. At the turn-k boundary, the agent selects a macroaction $u _ { k } \equiv l _ { k } \in \mathcal A _ { H , k } ( S _ { k } )$ , corresponding to the complete turn-k response $l _ { k } = ( a _ { k , 1 } , \ldots , a _ { k , T _ { k } } )$ Executing $u _ { k }$ consumes $\tau _ { k } = T _ { k }$ token steps, after which the environment returns to the external tool feedback $f _ { k }$ , and the interaction history is updated as $S _ { k + 1 } = S _ { k } \circ ( l _ { k } , f _ { k } )$ . To define the conditioning context received by the model at turn $\bar { k , }$ we let $c _ { k } = \psi ( S _ { k } )$ , where ψ can be the identity map, a truncation operator, or a summarization operator. The turn-level policy is factorized as $\pi _ { \theta } = \left( \pi _ { \theta , 0 } , \ldots , \pi _ { \theta , K - 1 } \right)$ , where each sub-policy is autoregressive at the token level: $\pi _ { \boldsymbol { \theta } , k } ( u _ { k } | \boldsymbol { c } _ { k } ) =$ $\begin{array} { r } { \prod _ { t = 1 } ^ { T _ { k } } \pi _ { \theta } ( a _ { k , t } | c _ { k } , a _ { k , < t } ) } \end{array}$

This turn-boundary MDP decomposition factorizes the episode policy into K turn-level sub-policies, with each sub-policy mapping the conditioning context $c _ { k } = \psi ( S _ { k } )$ to a complete response $l _ { k }$ . Here, $c _ { k }$ is kept identical between rollout and training. During rollout, RTPO records the exact context $c _ { k }$ received by the model, including any truncation or summarization, together with the corresponding old-policy log-probabilities. During training, the same $c _ { k }$ is used as input, and the loss is computed only over the output tokens in $l _ { k }$ . Hence, the denominator of the IS ratio is evaluated under the same conditioning context as in rollout:

$$
\rho _ { k , t } ^ { \mathrm { R T P O } } = \frac { \pi _ { \theta } \left( a _ { k , t } \middle | c _ { k } , a _ { k , < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { k , t } \middle | c _ { k } , a _ { k , < t } \right) }\tag{5}
$$

Per-turn policy optimization under reverse-order training. Training proceeds in reverse order through the turns $k = K { - } 1 , K { - } 2 , \ldots , 0$ . After turn k is completed, $\pi _ { \boldsymbol { \theta } , k }$ is frozen (e.g., subsequent turns produce no gradients for turn-k tokens). This reverse ordering ensures that when turn k is trained, the downstream policies in each turn $\pi _ { \theta , k + 1 : K - 1 }$ have been optimized and fixed.

During reverse-order training for turn-k, the environment is forked from the trunk trajectory’s boundary state $S _ { k }$ to generate G−1 sibling rollouts in the sparse tree. Each sibling rollout j receives a turn-level advantage $A _ { j , k } ^ { H }$ (the details of sibling rollout generation are provided in Sec. 3.2). The policy optimization objective for turn k is then defined as:

$$
J _ { k } ( \theta ) = \frac { 1 } { G - 1 } \sum _ { j = 1 } ^ { G - 1 } \frac { 1 } { T _ { j , k } } \sum _ { t = 1 } ^ { T _ { j , k } } \operatorname* { m i n } \Bigl ( \rho _ { j , k , t } A _ { j , k } ^ { H } , \ \mathrm { c l i p } \left( \rho _ { j , k , t } , 1 - \epsilon , 1 + \epsilon \right) A _ { j , k } ^ { H } \Bigr )\tag{6}
$$

where $\rho _ { j , k , t } = \pi _ { \theta } ( a _ { j , k , t } | c _ { k } , a _ { j , k , < t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { j , k , t } | c _ { k } , a _ { j , k , < t } )$ is the token IS ratio and $T _ { j , k }$ is the number of tokens generated by sibling rollout j at turn k. Only these G−1 sibling rollouts are used for gradient updates; the trunk trajectory is excluded.

In Theorem 1, we show that RTPO has local and global convergence guarantees through reverse-order turn-level policy optimization.

## Theorem 1: Convergence to Recursive Optimality

Under standard assumptions, finite state and macro-action spaces, Robbins–Monro step sizes, sufficient per-turn exploration, frozen downstream policies, and bounded rewards, reverse-order turn-level optimization satisfies:

![](images/60aec620492f6db9fbe716a53c225bea209ecd6c4db666f81c436fb4452f3e7f.jpg)  
Figure 2: Overview of RTPO in agentic RL training. After rollout, RTPO performs reverse-order turn level policy optimization for each trajectory in the batch. Starting from the final turn k, the rollout (old) policy π<sub>old</sub> generates G−1 sibling rollouts from the same turn boundary to estimate a group advantage and update the training (turn-level) policy $\operatorname { t o } \pi _ { k } .$ . The procedure then proceeds backward through turns $k { - } 1 , k { - } 2 , \ldots , 0$ , where each updated turn policy as $\pi _ { k - 1 } , \pi _ { k - 2 } , . . . , \pi _ { 0 }$ . To generate state-matched siblings for turn-level credit assignment, we design a sparse tree structure that assigns group-relative advantages to individual sibling rollouts at each turn and propagates optimization backward across the trajectory. Finally, RTPO applies on-policy continuation to coordinate asynchronous short- and long-trajectory updates from $\pi _ { k } 1 0 \pi _ { 0 }$ , reducing policy drift.

(a) Per-turn convergence. For each turn k, fixing downstream policies reduces optimization to single-step Q-learning with a stationary continuation target; hence, per-turn policy $\pi _ { k }$ converges under the stated assumptions.

(b) Recursive optimality. Under reverse-order training, applying the above argument recursively from turn $K - 1$ to turn 0 yields a sequence of per-turn policies that is recursively optimal.

(c) Global optimality. If the turn-level macro-action spaces are complete, i.e., the per-turn policy class can represent any globally feasible trajectory-level policy, then recursive optimality is equivalent to global optimality over the full trajectory.

The proof of Theorem 1 is provided in Appendix C.2.

## 3.2 Turn-Level Credit Assignment

Turn-level advantage function. We construct sparse tree rollouts at turn-level boundaries, elevating the advantage granularity from trajectory-only reward to turn-level credit while ensuring that all compared rollouts share the same state and are free from state bias. In turn $k , G { - } 1$ siblings independently generate turn-k responses from the shared boundary state $S _ { k }$ and continue to the terminal. The turn-level advantage function for sibling j is: $A _ { j , k } ^ { H } = \hat { Q } _ { j , k } - \hat { V } _ { k }$

Here, all siblings share the same state $S _ { k } .$ , so advantage differences can only arise from two sources: different action choices at turn k and independent downstream sampling noise. The downstream noise has zero mean $( \mathbb { E } [ \xi _ { \mathrm { d o w n } } | S _ { k } , l _ { j , k } ] = \boldsymbol { \dot { 0 } } )$ and introduces no systematic bias; the upstream state term $\xi _ { \mathrm { u p } } = V ^ { \pi } ( S _ { k } ) - { \bar { V } }$ from Eq. (3) is exactly zero, because all siblings share $S _ { k }$ . Furthermore, $A _ { j , k } ^ { H }$ is assigned only to the output tokens of turn k; the shared prefix $S _ { k }$ serves as the prompt input but does not enter the loss, so each token appears exactly once in the training batch.

Turn-level value estimation. To identify turn-level values, we separate the local reward at each turn from the downstream completion value over the full trajectory. Based on the MAXQ principle [Dietterich, 2000], we define the following action-value decomposition for turn k: $\tilde { Q } _ { k } ^ { \pi } ( S _ { k } , u _ { k } ) =$ $r _ { k } + \gamma ^ { \tau _ { k } } F _ { k } ^ { \pi } ( S _ { k + 1 } ) . ~ r _ { k }$ is the immediate reward at turn k $( r _ { k } = 0$ for $k < K { - } 1$ under sparse rewards), and $F _ { k } ^ { \pi } ( S _ { k + 1 } )$ is the downstream continuation value representing the expected cumulative return from turn $k { \ + 1 }$ onward under policy $\pi ,$ with base case $F _ { K - 1 } ^ { \pi } \equiv 0$ . Then, the terminal reward $R _ { j }$ obtained by sibling j rolling out the policy from $S _ { k }$ to the terminal is a single Monte Carlo sample of the above:

$$
\hat { Q } _ { j , k } = R _ { j } = r _ { j , k } + \gamma ^ { \tau _ { k } } \hat { F } _ { j , k }\tag{7}
$$

where $\hat { F } _ { j , k }$ is a single sample of $F _ { k } ^ { \pi } ( S _ { k + 1 } ) . R _ { j }$ is an unbiased estimator of $\tilde { Q } _ { k } ^ { \pi }$ . Note that the value estimation quality of $\hat { Q } _ { j , k }$ depends on $\hat { F } _ { j , k } , \mathrm { i . e . }$ , the quality of the sampled downstream continuation value. If the downstream policy is not yet optimized, even a strong turn-k action may still produce $R _ { j } = 0$ due to downstream errors, thereby contaminating the turn-level advantage with downstream noise. The case where $\hat { F } _ { j , k }$ is generated by an already optimized downstream policy is addressed in Sec. 3.3. After estimating the action values at each turn, we compute a state-specific value baseline from the sibling rollouts. Specifically, we define $\hat { V } _ { k }$ as the mean of the estimated $Q \cdot$ -values across the $G - 1$ siblings: $\begin{array} { r } { \hat { V } _ { k } = \frac { 1 } { G - 1 } \hat { \sum } _ { j = 1 } ^ { G - 1 } R _ { j } } \end{array}$

In Theorem 2, we show that RTPO obtains accurate turn-level credit without bias from upstream and downstream effects.

## Theorem 2: Causally Consistent Turn-Level Advantage Estimation

Consider turn k and suppose that the G−1 sibling rollouts are forked from the same boundary state $S _ { k } .$ , with each sibling rollout $j \in \{ 1 , \dots , G ^  \bar { - } 1 \} \nonumber$ independently sampling a turn-k macroaction $u _ { j , k } \equiv l _ { j , k }$ . Let $\mathbf { \bar { \boldsymbol { A } } } _ { j , \boldsymbol { k } } ^ { H }$ denote the turn-level advantage assigned to sibling rollout j. Then, under bounded rewards and independent sibling sampling, the turn-level advantage estimator satisfies the following properties:

(a) Local unbiasedness up to finite-group bias. Conditional on $S _ { k }$ , the expectation of $A _ { j , k } ^ { H }$ is proportional to the true turn-level advantage, up to a finite-group bias of order $O ( 1 / G )$ . Because all comparisons are made from the same boundary state, the estimator removes the upstream state-contamination term that appears in crosstrajectory comparisons.

(b) Reduced value estimation error. The mean squared error of $A _ { j , k } ^ { H }$ is lower than that of trajectory advantage estimation whenever cross-state value variance is non-zero. The improvement gap is governed by the variance of values across different boundary states, which can be large in long-horizon, multi-turn tasks.

(c) State-matched causal actions. Since all siblings share the same boundary state $S _ { k }$ differences in their returns are causally attributable to the sampled turn-k macro-actions $u _ { j , k }$ , rather than to upstream trajectory differences. The resulting advantage is assigned only to turn-k output tokens, excluding prefix tokens from the gradient.

The proof of Theorem 2 is provided in Appendix C.3.

## 3.3 On-Policy Continuation

On-policy evolution. Asynchronous turn updates can induce policy drift that is not fully corrected by per-token IS clipping. In principle, one could correct this drift using the full trajectory-level IS product, but its variance grows exponentially with the horizon length, making it unstable for long-horizon multi-turn training. This drift directly affects the downstream continuation value $F _ { k }$ introduced in Sec. 3.2: under a stale or mismatched downstream policy, $\hat { F } _ { j , k }$ can systematically deviate from the true continuation value $F _ { k } ^ { ^ { \pi _ { \theta } } > k }$ , while existing IS-based corrections are insufficient to remove this deviation. To avoid this issue, RTPO re-generates sibling continuations on-policy at each turn. Let $\theta _ { 0 }$ denote the parameters at the start of the policy evolution. During reverse-order training, the parameters are updated sequentially across turns. By the time optimization reaches turn $k ,$ the policies for downstream turns $K { \mathrm { - } } 1 , \ldots , k { \mathrm { + } } { \mathrm { . } }$ 1 have already been updated; we denote the resulting current parameters by $\theta _ { > k }$ . Thus, under $\pi _ { \boldsymbol { \theta } _ { > k } }$ , the downstream turns $k { + } 1 , \ldots , K { - } 1$ use the optimized continuation policy, whereas turn k and all upstream turns remain to be optimized. Note that the trunk is a complete trajectory generated at the start of the policy evolution using $\pi _ { \theta _ { 0 } }$ . It does not participate in gradient updates, nor does it enter the computation of $\hat { V } _ { k }$ . Its sole role is to provide boundary states $S _ { k }$ and environment snapshots $\operatorname { s n a n } _ { k }$ as anchoring points for sibling forking. The trunk’s policy nature affects which states $S _ { k }$ are visited during training (state coverage), but does not affect the correctness of advantage estimation, since all siblings contributing to the estimate are generated on-policy.

On-policy sibling generation. At the start of turn k, RTPO synchronizes the latest parameters $\theta _ { > k }$ to the inference engine, forks the environment from $\operatorname { s n a p } _ { k } .$ , and generates $G - 1$ siblings using $\pi _ { \boldsymbol { \theta } _ { > k } }$ . Each sibling rollout $j$ generates a turn-k response, then continues with $\pi _ { \boldsymbol { \theta } _ { > k } }$ through turns k+1 to $K - 1$ , obtaining terminal reward $R _ { j }$ . Since the sampling policy equals the current policy, the trajectory-level IS weight is identically one: $\begin{array} { r } { \omega _ { j } = \prod _ { h = k + 1 } ^ { K - 1 } \prod _ { t = 1 } ^ { T _ { j , h } } \frac { \pi _ { \theta > k } ( a _ { j , h , t } | s _ { j , h , t } ) } { \pi _ { \theta _ { > k } } ( a _ { j , h , t } | s _ { j , h , t } ) } \equiv 1 } \end{array}$

Since turn-level value estimation is $\hat { Q } _ { j , k } = r _ { j , k } + \gamma ^ { \tau _ { k } } \hat { F } _ { j , k }$ from Sec. 3.2, the sibling’s downstream continuation is performed under the current policy $\pi _ { \boldsymbol { \theta } _ { > k } }$ , the sample $\hat { F } _ { j , k }$ is an unbiased draw from $F _ { k } ^ { \pi _ { \theta _ { > k } } } ( S _ { j , k + 1 } )$ . Therefore:

$$
\hat { Q } _ { j , k } = r _ { j , k } + \gamma ^ { \tau _ { k } } \hat { F } _ { j , k } ^ { \pi _ { \theta _ { > k } } }\tag{8}
$$

is an unbiased Monte Carlo estimate of $\tilde { Q } _ { k } ^ { \pi _ { \theta } _ { > k } } ( S _ { k } , u _ { j , k } )$ . The entire value estimate requires no IS correction, is unaffected by clip truncation, and is free of multiplicative variance explosion.

In Theorem 3, we show that RTPO conducts on-policy continuation to avoid policy drift and further reduces advantage-estimation errors.

In the reverse-order training procedure of RTPO, at the start of each asynchronous turn k, sibling rollouts $j \in \{ 1 , \dots , \bar { G } ^ { - 1 } \}$ are generated from the shared boundary state $S _ { k }$ using the current downstream policy $\pi _ { \boldsymbol { \theta } _ { > k } }$ and are then continued to termination under the same policy. Then the following two properties hold:

(a) Drift-free on-policy continuation. Each sibling’s terminal return $R _ { j }$ provides an unbiased Monte Carlo estimate of the turn-level Q-value under the current downstream policy $\pi _ { \boldsymbol { \theta } _ { > k } }$ . Because the sampling policy and the evaluation policy coincide throughout the sibling continuation, the trajectory-level importance-sampling weight $\omega _ { j }$ is identically one, and no trajectory-level IS correction is required.

(b) Dynamic error reduction in advantage estimation. As reverse-order training improves the downstream policies, the continuation value estimates $\hat { F } _ { j , k } ^ { { ^ \pi _ { \theta } } _ { > k } }$ become aligned with the current optimized downstream policy rather than a stale policy. Under bounded binary or normalized rewards, when the induced success probabilities move away from the high-uncertainty region around one-half, the variance of the Monte Carlo Q-value estimator decreases, thereby improving the signal-to-noise ratio of the resulting turn-level advantage estimates $\check { A _ { j , k } ^ { H } }$

The proof of Theorem 3 is provided in Appendix C.4.

## 4 Experimental Results and Analysis

Experimental Setting. We use Qwen3-8B [Yang et al., 2025a] as the backbone for the main experiments on RTPO and all baselines, enabling Qwen3’s thinking mode for all multi-turn agentic RL rollouts. We compare RTPO with trajectory-level methods, including GRPO [Shao et al., 2024], and turn/tree-level credit methods, including ARPO [Dong et al., 2026], TreeGRPO [Ji et al., 2026], and SeeUPO [Hu et al., 2026]. We consider 12 experiments covering comprehensive mathematical and knowledge reasoning benchmarks, where agents perform multi-turn RL training with external calculation and web-search tools. All methods are implemented in VeRL [Sheng et al., 2024], with vLLM [Kwon et al., 2023] for rollout generation and FSDP [Zhao et al., 2023] for distributed training. We evaluate training performance and stability using task accuracy, tool-call statistics, log-probability comparisons, turn-level credit effects, and the hit rates of on-policy versus off-policy outputs. Detailed experimental setup, including base models, baselines, datasets, configurations, and evaluation metrics, is provided in Appendix D. Implementation details, including pseudocode, source code, training details, and the Qwen3 chat template, are provided in Appendix E.

## 4.1 Main Results

We compare RTPO against the trajectory-level method GRPO and the turn-level method SeeUPO, and additionally include the untuned vanilla model as a non-RL reference. Table 1 reports both accuracy and the corresponding number of tool calls on each benchmark. We evaluate overall performance on mathematical reasoning and knowledge-intensive question answering across three difficulty tiers: easy (GSM8K), medium (AMC23, MATH500), and hard (AIME24, AIME25, OE-Math, HotpotQA, and 2Wiki). Our RTPO achieves the best performance across all eight benchmarks, improving overall accuracy over vanilla by 66.78%, outperforming GRPO (+21.50%) and SeeUPO (+10.76%). Further discussion and insights on tool calls are provided in Appendix F.1.

<table><tr><td rowspan="2">Method</td><td colspan="10">M: Mathematical Reasoning</td><td colspan="5">K: Knowledge Reasoning</td><td colspan="2">Overall</td></tr><tr><td>AIME24 Acc</td><td>Calls</td><td>AIME25 Acc</td><td>Calls</td><td>AMC23 Acc</td><td>Calls</td><td>Acc</td><td>MATH500 Calls</td><td>GSM8K Acc</td><td>Calls</td><td>OE-Math Acc</td><td>Calls</td><td>HotpotQA Acc</td><td>Calls</td><td>2Wiki Acc</td><td>Calls</td><td>Acc</td><td>Calls</td></tr><tr><td>Vanilla</td><td>3.33</td><td>17</td><td>13.33</td><td>23</td><td>12.50</td><td>66</td><td>51.00</td><td>341</td><td>76.95</td><td>731</td><td>24.33</td><td>453</td><td>54.52</td><td>204</td><td>59.83</td><td></td><td> $3 6 . 9 7 _ { + 0 0 . 0 0 \% }$ </td><td> $2 0 5 9 _ { ( 1 6 3 1 , 4 2 8 ) }$ </td></tr><tr><td>GRPO</td><td>10.00</td><td>12</td><td>20.00</td><td>8</td><td>57.50</td><td>18</td><td>82.33</td><td>94</td><td>93.86</td><td>59</td><td>50.74</td><td>29</td><td>53.39</td><td>214</td><td>61.85</td><td>224 226</td><td> $5 3 . 7 1 _ { + 4 5 . 2 8 \% }$ </td><td>660(220,440)</td></tr><tr><td>SeeUPO</td><td>20.00</td><td>8</td><td>23.33</td><td>6</td><td>67.50</td><td>14</td><td>84.20</td><td>87</td><td>94.47</td><td>87</td><td>53.86</td><td>147</td><td>54.27</td><td>254</td><td>63.78</td><td>253</td><td> $5 7 . 6 8 _ { + 5 6 . 0 2 \% }$ </td><td>856(349,507)</td></tr><tr><td>RTPO</td><td>33.33</td><td>4</td><td>26.67</td><td>2</td><td>67.50</td><td>9</td><td>86.20</td><td>168</td><td>94.84</td><td>482</td><td>55.49</td><td>106</td><td>64.89</td><td>613</td><td>64.35</td><td>867</td><td> $\mathbf { 6 1 . 6 6 _ { + 6 6 . 7 8 \% } }$ </td><td>2251(771,1480)</td></tr></table>

Table 1: Performance comparison across eight tool-use agentic RL benchmarks. Acc denotes Pass@1 for mathematical tasks and best-span F1 for knowledge tasks, while Calls denotes the number of tool calls rounded to the nearest integer, such as Python and web-search calls. Overall subscripts indicate relative changes in accuracy compared with the vanilla model, along with total tool calls (M, K).

## 4.2 Rollout–Training Consistency Analysis

To examine RTPO’s training-time advantage, we measure at each step the geometric-mean ratio between training-stage and rollout-stage token logprobs. This ratio captures rollout–training consistency: values near 1 indicate matched conditioning distributions, while deviations suggest that the full-history training policy favors outputs different from those sampled during rollout. The Kull back–Leibler (KL) divergence further quantifies the distributional gap, with smaller values indicating stronger rollout–training consistency. Figure 3 shows the log-probability ratio and KL divergence during training. On mathematical tasks, RTPO stays stable at 1.0, while SeeUPO fluctuates around 1.0 and GRPO recovers from 0.89 to 0.97 after 30 steps. On knowledge tasks, RTPO again remains at 1.0, whereas SeeUPO and GRPO recover only from around 0.80 to 0.89 and 0.83 after 14 steps. RTPO also achieves the lowest average KL divergence, indicating more consistent rollout-training contexts than the baselines.

![](images/33c242c99e6ddeea990a56f6806127b41da90c381ea110f86ffea57127361ce1.jpg)  
(a) Rollout-train ratio [M]

![](images/bfbb71d410a1c96ccbd08c8cec4eaf0b7024ec307feef9eb041b1da1fe078179.jpg)  
(b) Rollout-train ratio [K]

![](images/dca884ef86fff3208d7bfbb27e517f914761b38990ff0bf48e36c6ac99cf9f6a.jpg)  
(c) KL divergence (M)

![](images/1f5ef0514e2d2cc07f11514ef3f5769b35b4e0886a3ef390b9212a5d00548186.jpg)  
(d) KL divergence (K)  
Figure 3: Rollout–training consistency comparison using log-probability ratios and KL divergence on mathematical (M) and knowledge (K) reasoning tasks. Zoom in for better visualization.

A noteworthy observation is that, despite using different rollout and training contexts, baseline ratios still drift slowly toward 1. This resembles the bootstrapped alignment mechanisms in DAgger [Ross et al., 2011] and SCoRe [Kumar et al., 2025]. However, unlike RTPO’s structural consistency, this empirical alignment is incomplete, task-dependent, noisy, and consumes additional optimization budget. We provide further discussion and insights in Appendix F.2.

## 4.3 Effect of Turn-Level Credit Assignment

To isolate the effect of RTPO’s turn-level credit signal, we feed the full interaction history as the rollout input for all methods, controlling for the rollout–training context mismatch presented in Sec 4.2. Under this control setting, performance differences mainly reflect the effect of credit assignment (CA) on advantage estimation. Because full interaction histories are used, the accuracy scores in Table 2 are substantially higher than those in Table 1. RTPO-CA achieves the highest overall average across the four mathematical reasoning benchmarks, as shown in Table 2. It achieves the best results on AMC23 (Pass@1 93.33, Pass@4 100.0), AIME25 (Pass@1 70.00, Pass@4 76.67), and MATH500 (Pass@1 86.60, Pass@4 89.60), while matching the best Pass@4 on AIME24 (80.00). These results show that RTPO improves both solution coverage and the preference for correct answers, with the largest gain on the harder AIME25 benchmark. This supports the value of turn-level credit assignment: by forking sibling rollouts from the same boundary state $S _ { k }$ , RTPO forms a local baseline and attributes advantage directly to the current turn decision.

<table><tr><td rowspan="2">Method</td><td colspan="2">AMC23</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">MATH500</td><td colspan="2">Overall</td></tr><tr><td>P@1</td><td>P@4</td><td>P@1</td><td>P@4</td><td>P@1</td><td>P@4</td><td>P@1</td><td>P@4</td><td>P@1</td><td>P@4</td></tr><tr><td>ARPO</td><td>90.00</td><td>97.50</td><td>73.33</td><td>73.33</td><td>63.33</td><td>73.33</td><td>86.00</td><td>88.80</td><td>78.15</td><td>83.23</td></tr><tr><td>TreeGRPO</td><td>92.50</td><td>96.67</td><td>63.33</td><td>80.00</td><td>56.67</td><td>70.00</td><td>86.20</td><td>89.40</td><td>74.67</td><td>84.00</td></tr><tr><td>SeeUPO</td><td>92.50</td><td>97.50</td><td>73.33</td><td>80.00</td><td>53.33</td><td>70.00</td><td>86.20</td><td>89.20</td><td>76.33</td><td>84.18</td></tr><tr><td>RTPO-CA</td><td>93.33</td><td>100.0</td><td>66.67</td><td>80.00</td><td>70.00</td><td>76.67</td><td>86.60</td><td>89.60</td><td>79.13</td><td>86.55</td></tr></table>

Table 2: Mathematical reasoning performance under controlled turn-level credit assignment (CA). Overall reports the average across benchmarks. Bold indicates the best result.

## 4.4 Policy Drift Correction

To isolate the effect of on-policy continuation (Sec 3.3), we compare the default RTPO with an off-policy variant that reuses downstream continuations generated by $\pi _ { \theta _ { 0 } }$ during the initial rollout and corrects staleness using a clamped trajectory-level IS weight; see Appendix E.2 for details. We evaluate both variants on four knowledge-reasoning deep-search benchmarks (GAIA, WebWalkerQA, HLE, and XBench) using output-hit accuracy. As shown in Table 3, default RTPO outperforms the off-policy variant on GAIA (+5.83%, WebWalkerQA (+3.50%), and XBench (+7.00%). On HLE, the difference is negligible (−0.33%), indicating a near tie. This pattern supports Theorem 3(b):

on-policy continuation is most beneficial when downstream policies change substantially during reverse-order training. GAIA, WebWalkerQA, and XBench involve longer retrieval and interaction horizons, where stale-rollout IS correction can introduce clamp-truncation bias that on-policy resampling avoids. In contrast, HLE is more closed-ended and often requires shorter search horizons, leading to smaller gains; additional results in Appendix F.3 further support this interpretation. In addition, we discuss the limitations, future work, and broader impacts of RTPO in Appendix F.4, F.5.
<table><tr><td>Benchmark</td><td>QS</td><td>Off-Policy Hit</td><td>Off-Policy Rate</td><td>On-Policy Hit</td><td>On-Policy Rate</td><td>∆</td></tr><tr><td>GAIA</td><td>103</td><td>24</td><td>23.30%</td><td>30</td><td>29.13%</td><td>+5.83%</td></tr><tr><td>XBench</td><td>100</td><td>8</td><td>8.00%</td><td>15</td><td>15.00%</td><td>+7.00%</td></tr><tr><td>WebWalker</td><td>200</td><td>12</td><td>6.00%</td><td>19</td><td>9.50%</td><td>+3.50%</td></tr><tr><td>HLE</td><td>2096</td><td>261</td><td>12.45%</td><td>254</td><td>12.12%</td><td>-0.33%</td></tr></table>

## 5 Concluding Remarks

This work identifies rollout–training mismatch as a fundamental source of instability in multi-turn agentic RL, particularly in tool-augmented mathematical reasoning and deep-search tasks. We provide a theoretical analysis showing how existing training pipelines produce unstable optimization signals and propose RTPO as a principled framework to address this issue. RTPO integrates rollout–training consistency, turn-level credit assignment, and on-policy continuation within a unified training pipeline. Supported by theoretical guarantees and empirical results, RTPO improves multi-turn optimization stability and provides a promising direction for training long-horizon tool-using agents.

## References

David Abel, Nate Umbanhowar, Khimya Khetarpal, Dilip Arumugam, Doina Precup, and Michael Littman. Value preserving state-action abstractions. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, 2020.

Cameron Allen, Neev Parikh, Omer Gottesman, and George Konidaris. Learning markov state abstractions for deep reinforcement learning. Advances in Neural Information Processing Systems, 2021.

Dimitri P Bertsekas. Neuro-dynamic programming. In Encyclopedia ofoptimization. 2025.

Cameron B. Browne, Edward Powley, Daniel Whitehouse, Simon M. Lucas, Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. A survey of monte carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in Games, 2012.

Lang Cao, Hui Ruan, Yongqian Li, Peng Chao, Wu Ning, Haonan Song, Renhong Chen, and Yitong Li. Treeadv: Tree-structured advantage redistribution for group-based rl. arXiv preprint arXiv:2601.03703, 2026a.

Ruike Cao, Shaojie Bai, Fugen Yao, Liang Dong, Jian Xu, and Li Xiao. Atpo: Adaptive tree policy optimization for multi-turn medical dialogue. In The Fourteenth International Conference on Learning Representations, 2026b.

Qikai Chang, Zhenrong Zhang, Pengfei Hu, Jun Du, Jiefeng Ma, Yicheng Pan, Jianshu Zhang, Quan Liu, and Jianqing Gao. THOR: Tool-integrated hierarchical optimization via RL for mathematical reasoning. In The Fourteenth International Conference on Learning Representations, 2026.

Kaiyuan Chen, Yixin Ren, Yang Liu, Xiaobo Hu, Haotong Tian, Tianbao Xie, Fangfu Liu, Haoye Zhang, Hongzhang Liu, Yuan Gong, et al. xbench: Tracking agents productivity scaling with profession-aligned real-world evaluations. arXiv preprint arXiv:2506.13651, 2025.

Yifei Chen, Guanting Dong, and Zhicheng Dou. Toward effective tool-integrated reasoning via self-evolved preference learning. In The Fourteenth International Conference on Learning Representations, 2026.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Rémi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In Proceedings ofthe 5th International Conference on Computers and Games, 2006.

Thomas G Dietterich. Hierarchical reinforcement learning with the maxq value function decomposition. Journal ofartificial intelligence research, 2000.

Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Agentic reinforced policy optimization. In The Fourteenth International Conference on Learning Representations, 2026.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International conference on machine learning. PMLR, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Milos Hauskrecht, Nicolas Meuleau, Leslie Pack Kaelbling, Thomas Dean, and Craig Boutilier. Hierarchical solution of markov decision processes using macro-actions. In Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence, 1998.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. Thirty-Fifth Annual Conference on Neural Information Processing Systems, 2021.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multihop QA dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, 2020.

Tianyi Hu, Qingxu Fu, Yanxi Chen, Zhaoyang Liu, and Bolin Ding. Seeupo: Sequence-level agentic-rl with convergence guarantees. arXiv preprint arXiv:2602.06554, 2026.

Tommi Jaakkola, Michael I. Jordan, and Satinder P. Singh. On the convergence of stochastic iterative dynamic programming algorithms. Neural Computation, 1994.

Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, and Liaoni Wu. Tree search for LLM agent reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2026.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. In The Thirteenth International Conference on Learning Representations, 2025.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 2022.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, et al. Websailor: Navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592, 2025.

Yugu Li, Zehong Cao, Jianglin Qiao, and Siyi Hu. SSVPO: Effective step-level credit assignment for RL training of language models. In The Fourteenth International Conference on Learning Representations, 2026.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Tobias Lindenbauer, Igor Slinko, Ludwig Felder, Egor Bogomolov, and Yaroslav Zharov. The complexity trap: Simple observation masking is as efficient as llm summarization for agent context management. arXiv preprint arXiv:2508.21433, 2025.

Mugeng Liu, Xiaojun Ma, Yuhang Xie, Qin Chen, Xuanzhe Liu, and Yun Ma. ROGA: Scaling generalist agents for office productivity tasks via tool generation. In The Fourteenth International Conference on Learning Representations, 2026.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Miao Lu, Weiwei Sun, Weihua Du, Zhan Ling, Xuesong Yao, Kang Liu, and Jiecao Chen. Scaling llm multi-turn rl with end-to-end summarization-based context management. arXiv preprint arXiv:2510.06727, 2025.

Mathematical Association of America. American mathematics competitions (AMC), 2023.

Mathematical Association of America. American invitational mathematics examination (AIME), 2024.

Mathematical Association of America. American invitational mathematics examination (AIME), 2025.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. arXiv preprint arXiv:2311.12983, 2023.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 2023.

Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics, 1951.

Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, 2011.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 2023.

Parshin Shojaee, Aneesh Jain, Sindhu Tipirneni, and Chandan K. Reddy. Execution-based code generation using deep reinforcement learning. Transactions on Machine Learning Research, 2023.

David Silver, Aja Huang, Christopher Maddison, Arthur Guez, Laurent Sifre, George Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of go with deep neural networks and tree search. Nature, 2016.

Satinder Singh, Tommi Jaakkola, Michael L. Littman, and Csaba Szepesvári. Convergence results for single-step on-policyreinforcement-learning algorithms. Machine Learning, 2000.

Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, et al. Simpledeepsearcher: Deep information seeking via web-powered reasoning trajectory synthesis. arXiv preprint arXiv:2505.16834, 2025.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Wanxin Tian, Shijie Zhang, Kevin Zhang, Xiaowei Chi, Chun-Kai Fan, Junyu Lu, Yulin Luo, Qiang Zhou, Yiming Zhao, Ning Liu, Siyu Lin, Zhiyuan Qin, Xiaozhu Ju, Shanghang Zhang, and Jian Tang. SEEA-r1: Tree-structured reinforcement fine-tuning for self-evolving embodied agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

John N Tsitsiklis. Asynchronous stochastic approximation and q-learning. Machine learning, 1994.

Xingyao Wang, Simon Rosenberg, Juan Michelini, Calvin Smith, Hoang Tran, Engel Nyst, Rohit Malhotra, Xuhui Zhou, Valerie Chen, Robert Brennan, et al. The openhands software agent sdk: A composable and extensible foundation for production agents. arXiv preprint arXiv:2511.03690, 2025a.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, et al. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025b.

Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 1992.

Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, et al. Webwalker: Benchmarking llms in web traversal. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025a.

Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang, Xinmiao Yu, Dingchu Zhang, Yong Jiang, et al. Resum: Unlocking long-horizon search intelligence via context summarization. arXiv preprint arXiv:2509.13313, 2025b.

Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun MA, and Bo An. Simpletir: End-to-end reinforcement learning for multi-turn tool-integrated reasoning. In First Workshop on Multi-Turn Interactions in Large Language Models, 2026.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Zhicheng Yang, Zhijiang Guo, Yinya Huang, Xiaodan Liang, Yiwei Wang, and Jing Tang. Treerpo: Tree relative policy optimization. arXiv preprint arXiv:2506.05183, 2025b.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, 2018.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, and Hao Zhou. Memagent: Reshaping long-context LLM with multi-conv RL-based memory agent. In The Fourteenth International Conference on Learning Representations, 2026a.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, YuYue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026b.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Yifan Zhong, Jakub Grudzien Kuba, Xidong Feng, Siyi Hu, Jiaming Ji, and Yaodong Yang. Heterogeneous-agent reinforcement learning. Journal ofMachine Learning Research, 2024.

## A Related Work

Policy optimization in agentic RL. Recent post-training of large language models (LLMs) has shifted from supervised fine-tuning toward reinforcement learning with verifiable rewards (RLVR). Among existing approaches, GRPO [Shao et al., 2024], built upon PPO, reduces variance through token-level importance ratios and group-relative advantage estimation, and has become a representative algorithm for agentic RL. Its sequence-level refinement, GSPO [Zheng et al., 2025], further defines importance ratios and clipping operations at the sequence level, leading to more stable training dynamics on models such as Qwen3. However, when these RL methods are directly transferred from single-turn to multi-turn agentic settings, a structural mismatch emerges between rollout and training. Existing methods often treat the entire multi-turn interaction as a single concatenated trajectory and distribute a single scalar reward uniformly across all tokens, thereby ignoring the actual contribution of each turn. More importantly, the rollout may operate on truncated or summarized contexts, whereas the training recomputes importance ratios over the full interaction history. This discrepancy induces a mismatch in conditioning distributions and weakens the assumptions under which PPO-style optimization is expected to remain stable. In long-horizon multi-turn scenarios, these issues can manifest as training divergence or eventual policy collapse.

A noteworthy latest work is SeeUPO [Hu et al., 2026], which first models multi-turn interaction as a sequentially executed multi-agent bandit problem. Under the heterogeneous-agent RL framework [Zhong et al., 2024], SeeUPO updates turn-level virtual agents in reverse execution order $( T $ $T - 1 \to \cdots \to 1 )$ , thereby inheriting the monotonic-improvement property and proving convergence to the globally optimal policy. This provides strong motivation for our proposed RTPO design, particularly its reverse-order turn updates. However, each turn-level agent in SeeUPO is still trained on the full chat history rather than on the turn-level conditioning context $c _ { k }$ actually observed by the model during rollout; therefore, the policy-forward mismatch is not eliminated. Moreover, within each turn, SeeUPO corrects downstream advantages using token-level importance-sampling reweighting. As the reverse training recursion proceeds and the number of involved turns accumulates, the variance of the importance-sampling ratio can grow multiplicatively. Although clipping bounds this ratio from above, it also introduces systematic bias into the per-turn advantage estimate.

Tree-based credit assignment. To mitigate the sparse-credit problem caused by flattened trajectories in multi-turn agentic RL training, a recent line of work reorganizes rollouts into tree structures with shared prefixes, thereby constructing finer-grained credit signals. ARPO [Dong et al., 2026] adaptively tree branches at high-entropy nodes following tool calls and, through advantage attribution estimation, applies branch averaging to shared-prefix tokens while estimating advantages independently for tokens on disjoint branches. Afterward, TreeGRPO [Ji et al., 2026] abstracts each turn as a tree node and combines intra-tree and inter-tree group-relative advantages, theoretically establishing gradient-level equivalence with step-level DPO [Rafailov et al., 2023]. This tree-based formulation is extended in SEEA-R1 [Tian et al., 2026], which integrates MCTS [Coulom, 2006, Browne et al., 2012, Silver et al., 2016] into embodied-agent settings and trains a multimodal generative reward model to densify sparse outcome rewards. Tree-structured credit assignment has also been explored in step/token levels. TreeRPO [Yang et al., 2025b] adopts an N-ary tree for mathematical reasoning and constructs step-level rewards through bottom-up Bellman expectations. TreeAdv [Cao et al., 2026a] employs entropy-triggered branching and redistributes leaf advantages to tokens using inversedescendant-count weighting. Similarly, ATPO [Cao et al., 2026b] operates under a hierarchical MDP and uses Bellman error and Q-value variance as uncertainty measures for adaptive expansion, while applying visit-count-based down-weighting to suppress update imbalance caused by repeated nodes.

Although these tree-based rollouts improve the granularity of credit assignment in multi-turn training, they do not fully eliminate the propagation of trajectory-level credit mismatch to individual turns through the reconstructed tree. For instance, in Tree-GRPO and SEEA-R1, advantages are still derived from leaf returns, typically in the form $\begin{array} { r } { A _ { i } = \frac { R _ { i } - \mathrm { m e a n } ( R ) } { \mathrm { s t d } ( R ) } } \end{array}$ . As a result, prefix nodes that appear on multiple paths inherit trajectory-level signals from all descendant outcomes, causing the gradient direction at shared tokens to be perturbed by trajectory-level scalars originating from different rollouts. In TreeAdv, the $1 / | S |$ normalization attenuates the signal more aggressively near the root, precisely suppressing early decisions where informative gradients are often most needed. In addition, ARPO averages multi-branch advantages on shared tokens, which can dilute the signal linearly with the number of branches, while ATPO relies on a learned critic and is therefore exposed to critic-induced bias. Therefore, under the rollout–training contextual mismatch settings, shared-prefix tokens may still carry trajectory-level credit signals with significant residual bias.

Policy gradient correction. Another line of work studies stable training from the perspective of correcting (off-)policy drift through interventions. SimpleTIR [Xue et al., 2026], by decomposing the policy gradient on softmax logits, attributes gradient explosion in multi-turn tool-integrated reasoning (TIR) to the accumulation of low-probability tokens under the distributional shift induced by tool feedback. It proposes filtering out entire trajectories that contain void turns: turns produce neither a complete code block nor a final answer to block harmful gradients. This method is empirically effective and plug-and-play; however, the void-turn criterion is tightly coupled with the code-execution setting of mathematical reasoning and is difficult to transfer to other scenarios, such as web search. It also inevitably discards a non-trivial fraction of training data. RAGEN [Wang et al., 2025b] discovers the inconsistency between rollout and training engines for mismatch correction through numerical mechanisms such as truncated importance sampling. However, these approaches mainly address numerical discrepancies while overlooking the contextual mismatch between rollout conditioning and training recomputation. In contrast, our proposed RTPO does not rely on trajectory dropping or post-hoc numerical correction. Instead, it restructures the training paradigm from a turn-boundary MDP perspective, maintaining the on-policy optimization throughout the entire training process.

## B Full Theoretical Analysis: Training Instability

Here, we provide the full theoretical analysis of training instability in multi-turn agentic RL: Single flattened-trajectory policy optimization (B.1): Standard rollouts are typically generated under truncated or summarized contexts for efficiency, whereas training re-evaluates tokens under concatenated full-history contexts without truncation. This rollout–training context mismatch induces biased importance-sampling (IS) ratios, thereby undermining stable policy optimization. Trajectoryonly credit assignment (B.2): In multi-turn interactions, a given state may admit multiple valid actions, requiring accurate credit assignment for each action. However, credit computed from a single trajectory-level advantage can obscure the contribution of individual turns and therefore cannot provide a proper comparison among alternative actions. Long-horizon off-policy drift (B.3): In long-horizon tasks, standard policy optimization with per-token clipping is insufficient to correct policy drift under asynchronous training. Since later states depend on earlier generated actions and tool feedback, once the policy drifts, the later-turn states visited during rollout may no longer match those induced by the current policy.

Preliminaries. A multi-turn interaction with TIR is represented as a standard trajectory $( q , l _ { 0 } , f _ { 0 } , \ldots , l _ { n - 1 } , f _ { n - 1 } )$ spanning n turns, where q is the initial prompt, $l _ { k }$ is the response generated, and $f _ { k }$ is the corresponding external tool feedback returned by the environment for each turn $k \in \{ 0 , \ldots , n - 1 \}$ . We formulate this process as a Hierarchical Markov Decision Process (H-MDP), which captures turn-level (high-level) planning and token-level (low-level) execution in multi-turn agentic RL. At the turn level, for each turn k, the agent state $S _ { k }$ represents all interaction history $( q , l _ { 0 } , f _ { 0 } , \ldots , l _ { k - 1 } , f _ { k - 1 } )$ , available before the current turn, including previous responses, tool calls, and corresponding environment feedback. The corresponding turn-level action is the decision of what response to produce at the current turn, denoted by $l _ { k } \in \mathcal { A } _ { H }$ . After executing this action and receiving environmental feedback $f _ { k }$ , the turn-level state evolves as $S _ { k + 1 } = S _ { k } \circ \bar { ( } l _ { k } , f _ { k } )$ , where ◦ denotes concatenation.

Each turn-level action (response) $l _ { k }$ is generated autoregressively at the token level as $l _ { k } \ =$ $\left( a _ { k , 1 } , a _ { k , 2 } , \ldots , a _ { k , T _ { k } } \right)$ , where $a _ { k , t } \in \mathcal { A } _ { L }$ denotes the token generated at step $t ,$ and $T _ { k }$ is the number of tokens in the response at turn k. The corresponding low-level state is $s _ { k , t } = ( S _ { k } , a _ { k , 1 } , \ldots , a _ { k , t - 1 } ) \in$ $ { \boldsymbol { S } } _ { L }$ , which consists of the turn history $S _ { k }$ together with the token prefix generated so far in the current turn. Since the token only serves to generate the turn-level action (response) and does not itself receive intermediate reward, we set the low-level reward to $R _ { L } = 0$ and set $\gamma _ { H } = \gamma _ { L } = 1$ as the discount factor.

Our hierarchical formulation differs from existing policy optimization methods, which do not explicitly distinguish turn-level planning from token-level execution. Instead, they flatten the entire multi-turn interaction into a single token sequence, i.e., a single trajectory, and optimize it using clipped policy optimization methods in the PPO family. Let $q \sim \mathcal { D }$ be an input prompt sampled from the task distribution, and let $\{ g _ { i } \} _ { i = \cdot } ^ { G }$ be a group of G trajectories of this form sampled from the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ conditioned on $q ,$ where θ denotes the current policy parameters and $\theta _ { \mathrm { o l d } }$ denotes the rollout policy parameters. For trajectory $g _ { i }$ , we denote its turn-k state and response by $S _ { i , k }$ and $l _ { i , k }$ respectively:

$$
J ^ { \mathrm { f l a t } } ( \theta ) = \mathbb { E } _ { \{ g _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | q ) } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { \sum _ { t ^ { \prime } } m _ { i , t ^ { \prime } } } \sum _ { t } m _ { i , t } L _ { i , t } ^ { \mathrm { C L I P } } ( \theta ) \right]\tag{9}
$$

where i indexes the sampled trajectory, t indexes flattened token positions, and $L _ { i , t } ^ { \mathrm { C L I P } } ( \theta ) ~ =$ min $( \rho _ { i , t } ( \theta ) A _ { i }$ , cli $\geqslant ( \rho _ { i , t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } )$ is the standard clipped surrogate objective with clipping threshold $\begin{array} { r l } { \epsilon . } & { { } a _ { i , t } } \end{array}$ is the token generated at flattened position t in the i-th sampled trajectory, $m _ { i , t }$ is a binary mask indicating whether that token contributes to the policy gradient, and $\begin{array} { r } { \rho _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( a _ { i , t } | x _ { i , < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { i , t } | x _ { i , < t } \right) } } \end{array}$ is the IS ratio. The context $x _ { i , < t }$ is the flattened interaction context preced ing token $a _ { i , t } .$ , including previously generated response tokens and inserted environment feedback. The scalar $A _ { i }$ denotes the trajectory-level group-relative advantage, which is uniformly assigned to all unmasked tokens in trajectory $g _ { i }$

## B.1 Single flattened-trajectory policy optimization: mismatch from rollout to training

In multi-turn interactions, the accumulated history can become too long for the model to process in full. For example, an agent may search for information, call a tool, revise its plan based on the returned result, and repeat this process over many turns. By later turns, the prompt, previous responses, and tool feedback may already span tens of thousands of tokens. In practice, rollouts therefore often rely on a truncated or summarized context rather than the complete interaction history [Wang et al., 2025a, Lindenbauer et al., 2025, Wu et al., 2025b]. Therefore, existing methods typically reconstruct the whole interaction as a single concatenated sequence, or flattened trajectory, during training. This flattened training formulation evaluates each generated token under the concatenated prefix, rather than under the original context that was actually used during rollout. Consequently, tokens generated at turn $k$ are optimized under a conditioning context that can differ from the rollout context that produced them. We refer to this discrepancy as a rollout-to-training mismatch induced by flattened-trajectory policy optimization. We next analyze how this mismatch distorts the likelihood-ratio estimation underlying clipped policy optimization in the PPO/GRPO family.

Policy optimization mismatch across turns. Under flat training, the optimization mismatch across turns arises because the same generated token is conditioned on different contexts during rollout and training. We now formalize this mismatch and show that it induces a biased IS ratio. $\operatorname { L e t } { \bar { x } } _ { k }$ denote the full interaction history before turn $k ,$ and let $\phi : \bar { \mathcal { X } } \to \mathcal { Z }$ be an observation map that truncates or summarizes history beyond the model’s effective context length. During rollout, token-level actions are sampled conditioned on $\phi ( \bar { x } _ { k } )$ , so the true sampling distribution is $\pi _ { \theta _ { \mathrm { o l d } } } ( a _ { k , t } \mid \phi ( \bar { x } _ { k } ) )$ ). In clipped policy optimization of the PPO/GRPO family, the denominator of the IS ratio must match this rollout distribution. However, under flat training, the same token is re-evaluated under the concatenated full-history context, yielding $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } \left( \boldsymbol { a } _ { k , t } \mid \bar { \boldsymbol { x } } _ { k } \right)$ . This leads to the mismatch

$$
\begin{array} { r } { \rho _ { k , t } ^ { \mathrm { f a t } } = \frac { \pi _ { \theta } \left( a _ { k , t } \mid \bar { x } _ { k } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { k , t } \mid \phi \left( \bar { x } _ { k } \right) \right) } \neq \frac { \pi _ { \theta } \left( a _ { k , t } \mid \phi \left( \bar { x } _ { k } \right) \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { k , t } \mid \phi \left( \bar { x } _ { k } \right) \right) } = \rho _ { k , t } ^ { \mathrm { t r u e } } . } \end{array}\tag{10}
$$

The denominator used in flat training, $\pi _ { \theta _ { \mathrm { o l d } } } ( a _ { k , t } \mid \bar { x } _ { k } )$ , does not match with the true rollout sampling probability, $\pi _ { \theta _ { \mathrm { o l d } } } ( a _ { k , t } \mid \phi ( \bar { x } _ { k } ) )$ . Consequently, $\rho _ { k , t } ^ { \mathrm { f l a t } }$ is a biased estimate of the correct IS ratio. Since clipped policy-gradient updates in the PPO/GRPO family depend on the IS ratio, the bias propagates into the gradient estimate and can distort the update direction, leading to instability in multi-turn agentic RL training. This mismatch becomes more severe for tokens generated in later turns, since the amount of history omitted during rollout typically grows with k, while flat training continues to re-evaluate these tokens under the concatenated training prefix. In the worst case, the resulting discrepancy in token probability can become extremely large.

State aliasing and projected suboptimality. Beyond gradient bias, truncated contexts also introduce state aliasing, where distinct full-history states are mapped to the same truncated representation. When $\phi$ is non-injective, different interaction histories may collapse into an identical observation $z _ { k } = \phi ( \bar { x } _ { k } )$ [Coulom, 2006]. The induced process over $z _ { k }$ may not preserve the Markov property of the original full-history process [Allen et al., 2021]. As a result, any policy conditioned only on $z _ { k }$ is confined to the observation-induced policy class Πϕ. Even if optimized exactly within this restricted class, such a policy can achieve only the projected optimum $\hat { V _ { \Pi _ { \phi } } ^ { * } }$ [Abel et al., 2020], which can be strictly lower than the true optimum $V ^ { * }$ . Therefore, projected suboptimality can arise, with $V _ { \Pi _ { \phi } } ^ { * } \ < \ V ^ { * }$ . Under observation inconsistency, flattened-trajectory training may be limited to the projected optimum $V _ { \Pi _ { \phi } } ^ { * }$ , which is strictly suboptimal relative to the full-history optimum $V ^ { * }$

## B.2 Trajectory-only credit assignment: mismatch across low- and high-quality turns

In multi-turn interactions, trajectories generated from the same query can reach substantially different states by turn k. As a result, a trajectory-level return no longer provides a reliable credit signal for evaluating actions taken at that turn. Under flattened-trajectory training, credit is assigned only at the trajectory level and then shared across tokens within a single trajectory, which mismatches the turn-level structure of the decision process. Here, we analyze how a trajectory advantage entangles the contribution of the current turn with both upstream state effect and downstream stochasticity.

A trajectory advantage entangles turn-level credit with context effects. A single trajectory advantage is typically defined as $A _ { i } = R _ { i } - { \bar { R } }$ , where $R _ { i }$ denotes the final return of trajectory $g _ { i }$ and R<sup>¯</sup> is the average return over the sampled group. This trajectory advantage is then assigned to all turns in the trajectory, without identifying each turn’s individual contributions. To expose the turn-level credit hidden in this trajectory return, we analyze its population counterpart, $R _ { i } - \bar { \mathbb { E } } [ R ]$ , by introducing the conditional expectations $\mathbb { E } [ R \mid S _ { i , k } , l _ { i , k } ]$ and $\mathbb { E } [ \dot { R } | S _ { i , k } ]$

$$
R _ { i } - \mathbb { E } [ R ] = \underbrace { R _ { i } - Q ^ { \pi } ( S _ { i , k } , l _ { i , k } ) } _ { \xi _ { \mathrm { d o w n : d o w n s t r e a m s t o c h a s t i t y } } } + \underbrace { A ^ { \pi } ( S _ { i , k } , l _ { i , k } ) } _ { \mathrm { t r u e : u r n - k ~ a d v a n t a g e } } + \underbrace { V ^ { \pi } ( S _ { i , k } ) - \mu _ { R } } _ { \xi _ { \mathrm { u p : u p s t r e a m ~ s t a t e ~ e f f e c t } } } ,\tag{11}
$$

where $Q ^ { \pi } ( S _ { i , k } , l _ { i , k } ) \ : = \ : \mathbb { E } [ R \mid S _ { i , k } , l _ { i , k } ] , V ^ { \pi } ( S _ { i , k } ) \ : = \ : \mathbb { E } [ R \mid S _ { i , k } ] .$ , and $\mu _ { R } = \mathbb { E } [ R ]$ . The true turn-k advantage $\overset { \cdot } { A ^ { \pi } } ( S _ { i , k } , \overset { \cdot } { l } _ { i , k } )$ is therefore only one component of the trajectory advantage. The term $\xi _ { \mathrm { d o w n } }$ captures downstream stochasticity after turn $k ,$ , while $\xi _ { \mathrm { u p } }$ captures variation induced by the upstream state reached before turn $k .$ Whenever $\xi _ { \mathrm { d o w n } } + \xi _ { \mathrm { u p } }$ dominates $A ^ { \pi } ( S _ { i , k } , l _ { i , k } )$ in magnitude, the sign of the trajectory advantage can disagree with that of the true turn advantage, $\mathrm { i . e . }$ $\mathrm { s i g n } ( A _ { i } ) \ne \mathrm { s i g n } \bar { ( } A ^ { \pi } ( S _ { i , k } , \bar { \iota _ { i , k } } ) )$ , thereby reversing the gradient direction for turn $k .$ Therefore, a trajectory advantage is not a valid turn-level credit signal for turn $k ,$ because it entangles the context effect of the current turn with both upstream state effects and downstream stochasticity. When these two terms dominate, the resulting policy update can assign incorrect credit to the current turn and may even reverse the intended policy gradient direction.

Cross-trajectory baseline with state bias. Since group-based policy optimization methods generate multiple trajectories from the same input query by sampling several rollouts from the current or old policy, they typically compute advantages of the form $A _ { i } = R _ { i } - b .$ , where $R _ { i }$ is the return of trajectory $g _ { i }$ and $\begin{array} { r } { b = { \frac { 1 } { G } } \sum _ { j = 1 } ^ { G } R _ { j } } \end{array}$ is the group baseline. To determine whether the compared trajectories provide a valid turn-level credit signal, we examine whether they share the same turn-level state $S _ { i , k }$ . If they do, the baseline compares alternative outcomes from the same context and is therefore matched. Otherwise, the baseline mixes returns from different states and no longer reflect the local effect of the current turn, leading to mismatched credit assignment.

Formally, viewing $A _ { i }$ as an estimator of the true turn-k advantage $A ^ { \pi } ( S _ { i , k } , l _ { i , k } )$ gives

$$
\mathbb { E } [ A _ { i } \ | \ S _ { i , k } , l _ { i , k } ] = \frac { G - 1 } { G } \Big ( Q ^ { \pi } ( S _ { i , k } , l _ { i , k } ) - \underbrace { \mathbb { E } [ R _ { j } \ | \ S _ { i , k } ] } _ { \mathrm { d e p e n d s o n g r o u p i n g } } \Big ) , \qquad j \neq i .\tag{12}
$$

In the case of same-state grouping, where all trajectories in the group share the same state at turn $k _ { : }$ we have $\mathbb { E } [ R _ { j } \ | \ S _ { i , k } ] = { \bar { V } } ^ { \pi } ( { \bar { S } } _ { i , k } )$ . In this case, the group baseline is anchored to the correct local decision context, and the resulting estimator differs from the true turn-level advantage only by the multiplicative factor $( G - 1 ) / G$ . However, under cross-state grouping, trajectories within the same group may already reach different states by turn $k .$ Then the baseline is no longer tied to the local state $S _ { i , k }$ , but is effectively centered around the global mean value as return, $\mathrm { i . e . , \mathbb { E } } [ R _ { j } ] = \mu _ { R }$ . This

introduces the additional bias:

$$
{ \mathrm { B i a s } } _ { \mathrm { c r o s s } } = { \frac { G - 1 } { G } } { \big ( } V ^ { \pi } ( S _ { i , k } ) - \mu _ { R } { \big ) } .\tag{13}
$$

This bias is determined entirely by the upstream trajectory and has no causal connection to the action taken at turn k. In multi-turn interactions, trajectories often diverge into semantically distinct environment states by turn k. For example, one trajectory may issue a search query while another is executing code. As a result, the variance of $V ^ { \ ' \pi } ( S _ { i , k } )$ across states can be large, making the cross-state bias comparable to, or even larger than, the true turn advantage $A ^ { \pi } ( S _ { i , k } , \bar { l } _ { i , k } )$ . Therefore, a cross-trajectory baseline cannot guarantee a valid local comparison signal for turn-level credit assignment.

## B.3 Long-horizon policy drift: PPO clipping mismatch asynchronous turns

In long-horizon multi-turn training, policy updates can become asynchronous across turns, inducing turn policy drift. In practice, PPO-style methods use token clipping to constrain the IS ratio and limit policy drift during updates. However, this clipping mechanism is designed for near-on-policy updates with synchronized rollout data and does not explicitly account for turn-wise discrepancies when different parts of a trajectory are generated or optimized under different policy versions. Over long horizons, such mismatches can accumulate, shifting the update away from the near-on-policy learning and toward an off-policy setting.

This occurs because, in asynchronous multi-turn training, policy updates may be performed before all trajectories have completed their rollouts. For instance, shorter trajectories may complete first and immediately contribute to an update, while longer trajectories are still being generated under an older policy. Specifically, short trajectories may finish first and update the policy from $\theta _ { 0 }$ to $\theta _ { 1 }$ , while long trajectories are still being rolled out under $\pi _ { \theta _ { 0 } }$ . By the time these longer trajectories are used for training, the current policy $\pi _ { \theta _ { 1 } }$ no longer matches the policy that generated them.

In principle, this mismatch can be corrected by the full trajectory IS ratio:

$$
\omega _ { i } = \prod _ { t = 1 } ^ { T _ { i } } \rho _ { i , t } = \prod _ { t = 1 } ^ { T _ { i } } \frac { \pi _ { \theta _ { 1 } } ( a _ { i , t } \mid x _ { i , < t } ) } { \pi _ { \theta _ { 0 } } ( a _ { i , t } \mid x _ { i , < t } ) } ,\tag{14}
$$

where $\rho _ { i , t }$ is the token IS ratio, $x _ { i , < t }$ is the flattened multi-turn interaction context preceding token $a _ { i , t }$ , and $T _ { i }$ is the total number of tokens in the flattened trajectory $g _ { i }$ . Weighting the loss by $\omega _ { i }$ would yield an unbiased correction to the policy objective. However, clipped policy optimization in the PPO/GRPO family does not employ an IS ratio for the full trajectory. Instead, it clips each token ratio $\rho _ { i , i }$ independently:

$$
\mathcal { L } _ { \mathrm { c l i p } } = \sum _ { t = 1 } ^ { T _ { i } } \operatorname* { m i n } \bigl ( \rho _ { i , t } A _ { i } , \ \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon ) A _ { i } \bigr ) .\tag{15}
$$

Therefore, if one composes the clipped token-level ratios into a trajectory-level correction, it generally differs from the true full-trajectory IS ratio:

$$
\prod _ { t = 1 } ^ { T _ { i } } \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon ) \neq \prod _ { t = 1 } ^ { T _ { i } } \rho _ { i , t } = \omega _ { i } .\tag{16}
$$

Because PPO clipping is nonlinear, the product of clipped token-level ratios does not equal the true trajectory-level importance weight. In particular, each clipped ratio lies in $[ 1 - \epsilon , 1 + \epsilon ]$ , so their product is restricted to $\lbrack ( 1 - \epsilon ) ^ { T } , ( 1 + \epsilon ) ^ { T } ]$ , whereas the true ω can in principle take any value in $( 0 , \infty )$ . Whenever ω falls outside this interval, per-token clipping necessarily yields a biased trajectory correction. Even when ω lies within the interval, the product of individually clipped ratios will generally differ from ω as soon as any $\rho _ { t }$ is clipped, since clipping and multiplication do not commute. This discrepancy compounds with trajectory length $T \colon$ as more tokens are clipped, the gap between $\prod _ { t } \mathrm { c l i p } ( \rho _ { t } )$ and ω can grow progressively larger. Removing clipping and using the exact ω is not a practical solution, because its variance grows exponentially with $T ,$ making gradient estimates increasingly uninformative over the long horizons typical of multi-turn agentic RL. Hence, token-level clipping cannot faithfully reproduce the trajectory-level correction required for long-horizon policy drift.

## C Method: RTPO Theoretical Proofs

## C.1 Notation, Formal Problem Setup, and Technical Assumptions

Notation convention. We create a multi-turn agentic episode, which consists of K turn-level interactions, written as $( q , l _ { 0 } , f _ { 0 } , \dots , l _ { K - 1 } , f _ { K - 1 } )$ , where q is the initial prompt, $l _ { k }$ is the complete response generated by the agent at turn $k ,$ , and $f _ { k }$ is the external tool or environment feedback returned after executing $l _ { k }$ . The turn index is denoted by $k \in \{ 0 , \ldots , K - 1 \}$ , while the token index within a response is denoted by $t .$ When multiple rollouts are sampled, we use i or j to index the rollout or sibling trajectory. Thus, $S _ { i , k }$ denotes the turn-k state in rollout $i ,$ and $a _ { i , k , t }$ denotes the t-th token generated at turn k in rollout i. When no rollout index is needed, we can write $S _ { k } , l _ { k }$ , and $a _ { k , \imath }$ for a generic trajectory. We distinguish between a turn-level macro-action and its token-level realization. The macro-action at turn k is denoted by $u _ { k } \equiv l _ { k }$ , where $l _ { k } = \left( a _ { k , 1 } , \ldots , a _ { k , T _ { k } } \right)$ is the complete response and $T _ { k }$ is the number of generated tokens in that response. The notation $u _ { k }$ is used in the MDP and value-function definitions, while $l _ { k }$ emphasizes that the macro-action is implemented as a language-model response. In addition, the reward in a task is assigned at the turn or trajectory level. Tokens inside a response are treated as the low-level realization of the turn-level macro-action and do not receive separate intermediate rewards. Thus, the low-level token reward is set to zero. We write $r _ { k }$ for the immediate turn-level reward at turn $k .$ . In sparse-reward tasks, we typically have $r _ { k } = 0$ for $k < K - 1$ , and the final task reward is observed only after the terminal turn.

Turn-boundary MDP for rollout-training match. A K-turn agentic episode is modelled at turn boundaries as $\dot { \mathcal { M } } = \langle \bar { \mathcal { X } } , \mathcal { A } _ { H } , P _ { H } , R _ { H } , \gamma _ { H } \rangle$ , where X<sup>¯</sup> is the augmented turn-boundary state space, $\boldsymbol { \mathcal { A } } _ { H }$ is the high-level action space of complete responses, $P _ { H }$ is the transition kernel induced by executing a complete response and receiving tool feedback, $R _ { H }$ is the turn-level reward function, and $\gamma _ { H }$ is the turn-level discount factor. The augmented state at turn k is $\bar { x } _ { k } = ( S _ { k } , k ) \in \bar { \mathcal X }$ , where $S _ { k } = ( q , l _ { 0 } , f _ { 0 } , \ldots , l _ { k - 1 } , f _ { k - 1 } )$ is the interaction history available before the current turn. Adding the turn index k in $\bar { x } _ { k }$ can make the process Markov over a finite horizon, since the remaining number of turns can affect both the available decisions and the continuation value. At turn k, the agent selects a macro-action $u _ { k } \equiv l _ { k } \in \mathcal A _ { H , k } ( S _ { k } )$ , where $\boldsymbol { \mathcal { A } } _ { H , k } ( \boldsymbol { S } _ { k } )$ ) denotes the set of feasible complete responses at state $S _ { k }$ and turn $k$ . The macro-action is realized autoregressively as $l _ { k } = ( a _ { k , 1 } , \ldots , a _ { k , T _ { k } } )$ and consumes $\tau _ { k } = T _ { k }$ token-generation steps. After executing $u _ { k }$ and receiving environment feedback $f _ { k }$ , the turn-level history is updated by concatenation as $\bar { S } _ { k + 1 } = S _ { k } \circ ( l _ { k } , \bar { f } _ { k } )$ . Equivalently, since $u _ { k } \equiv l _ { k }$ , one may write $S _ { k + 1 } = S _ { k } \circ ( u _ { k } , f _ { k } )$ . In Sec 3, we adopt this MDP terminology for the turn-boundary process. Strictly speaking, because each macro-action may span a variable number of token-level steps $\tau _ { k } .$ , this process can alternatively be formulated as a finite-horizon semi-MDP. This distinction does not affect our analysis, since policy optimization and credit assignment are defined exclusively over turn-boundary states and macro-actions.

To maintain rollout–training consistency under conditional contexts, the policy need not condition on the full interaction history $S _ { k }$ . Instead, it may condition on a compressed or truncated context, provided that the same conditional context is used consistently during both rollout generation and policy optimization. Instead, the actual conditioning context at turn k is $c _ { k } = \psi ( S _ { k } )$ , where ψ may be the identity map, a truncation operator, or a summarization operator. This distinction is important in long-context multi-turn training: rollout may be performed under a truncated or summarized context, while training may otherwise recompute log-probabilities under a different context. RTPO avoids this mismatch by recording the exact context $c _ { k }$ used during rollout and reusing the same $c _ { k }$ during training. The turn-level policy factorizes as $\pi _ { \theta } = ( \pi _ { \theta , 0 } , \ldots , \pi _ { \theta , K - 1 } )$ , where each sub-policy maps the turn-level context $c _ { k }$ to a complete response. Each sub-policy is implemented autoregressively:

$$
\pi _ { \boldsymbol { \theta } , k } \bigl ( u _ { k } \mid \boldsymbol { c } _ { k } \bigr ) = \prod _ { t = 1 } ^ { T _ { k } } \pi _ { \boldsymbol { \theta } } \bigl ( a _ { k , t } \mid \boldsymbol { c } _ { k } , a _ { k , < t } \bigr ) .\tag{17}
$$

Here, $a _ { k , < t } = ( a _ { k , 1 } , \dots , a _ { k , t - 1 } )$ is the token prefix generated within the current turn. Because the same $c _ { k }$ is used in rollout and training, the token-level importance-sampling (IS) ratio for turn k is evaluated under matched conditioning contexts:

$$
\rho _ { k , t } = { \frac { \pi _ { \theta } ( a _ { k , t } \mid c _ { k } , a _ { k , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { k , t } \mid c _ { k } , a _ { k , < t } ) } } .\tag{18}
$$

Policy optimization with recursive optimality. A policy sequence $\pi ^ { \mathrm { r e c } } = ( \pi _ { 0 } ^ { * } , \ldots , \pi _ { K - 1 } ^ { * } )$ is recursively optimal if, for every turn $k \in \{ 0 , \ldots , K - 1 \}$ , the turn-k sub-policy $\pi _ { k } ^ { * }$ maximizes the turn-level augmented value given that all downstream sub-policies have already been optimized and fixed. Formally, for each $k , \pi _ { k } ^ { \ast } \in$ arg max<sub>πk</sub> $\tilde { Q } _ { k } ^ { \pi ^ { * } } ( S _ { k } , u _ { k } )$ , where the downstream policies $\pi _ { k + 1 } ^ { * } , \ldots , \pi _ { K - 1 } ^ { * }$ are treated as fixed during the optimization of $\pi _ { k } ^ { * } .$ . This is a backward-induction notion of optimality: the last turn is optimized first, then the preceding turn is optimized assuming the last-turn policy is fixed, and so on until the first turn.

Furthermore, we establish convergence to recursive optimality in a tabular finite-horizon setting. Assumptions below $( \mathbf { A l - A l } 0 )$ are not intended to model the full neural implementation, but instead serve to isolate the theoretical effect of reverse-order turn-level policy optimization under standard stochastic approximation conditions.

(A1) Finite turn-boundary spaces. The augmented state space $\bar { \mathcal X }$ is finite, and for every turn $k$ and every reachable state $S _ { k }$ , the feasible high-level action set $\boldsymbol { \mathcal { A } } _ { H , k } ( \boldsymbol { S } _ { k } )$ ) is finite.

(A2) Proper finite-horizon episodes. The number of turns K is finite. For every policy and every turn $k ,$ the macro-action duration satisfies $\mathbb { E } _ { \pi _ { k } } [ \tau _ { k } ] < \infty$ . Thus, each turn terminates almost surely in a finite expected token length.

(A3) Discounting orfinite-horizon boundedness. Either $\gamma _ { H } \in ( 0 , 1 )$ , or the problem is finite-horizon with $K < \infty$ and $\gamma _ { H } \in ( 0 , 1 ]$ . The latter case includes the undiscounted finite-horizon setting $\gamma _ { H } = 1$

(A4) Tabular value representation. The value estimate $Q _ { k } ( S , u )$ is stored separately for each turnstate-action tuple $( S , u , k )$ . This assumption avoids approximation error and allows the proof to focus on the stochastic approximation dynamics induced by reverse-order updates.

(A5) Reverse-order training with downstream freezing. Training is proceeding in the order $k =$ $K - 1 , K - 2 , \ldots , 0$ . During turn $k ,$ the turn-k policy is updated while the downstream policies $\pi _ { k + 1 : K - 1 }$ are held fixed. After turn $k$ is completed, the turn-k policy is also frozen before moving to turn $k - 1$

(A6) GLIE exploration within each turn. Within turn $k ,$ , the exploration schedule is greedy in the limit with infinite exploration (GLIE): every feasible turn-level action is selected with strictly positive probability infinitely often, while the policy becomes greedy in the limit. This ensures that all relevant action values at turn k are sufficiently sampled before the policy is frozen.

(A7) State coverage. Every reachable boundary state $S _ { k }$ that can arise under the training process is visited infinitely often during turn $k .$ . This condition ensures that the tabular value estimate for each relevant state-action pair receives infinitely many updates.

(A8) Robbins–Monro step sizes. For every tuple $( S , u , k )$ , the learning rates satisfy $\textstyle \sum _ { n = 1 } ^ { \infty } \alpha _ { n } ( S , u , k ) = \infty$ and $\textstyle \sum _ { n = 1 } ^ { \infty } \alpha _ { n } ^ { 2 } ( S , u , k ) ^ { \bullet } < { \overset { \cdot } { \infty } }$ . These are the standard stochastic approximation step-size conditions.

(A9) Bounded rewards. The turn-level rewards are uniformly bounded: $| r _ { k } | \leq r _ { \operatorname* { m a x } }$ almost surely for all k.

(A10) Bounded iterates. The tabular value iterates remain uniformly bounded: $| Q _ { k } ( S , u ) | \leq Q _ { \operatorname* { m a x } }$ throughout learning. This assumption is standard in stochastic approximation analyses and can be enforced by projection if necessary.

(A11) Decision sufficiency of the conditioning context. The observation mapping $\psi \colon S _ { k } \mapsto c _ { k }$ preserves all decision-relevant information for following turns. In other words, states that are indistinguishable under ψ share the same optimal action-value function, so the optimal policy at turn $k$ depends on $S _ { k }$ only through $c _ { k }$ . This implies that $\tilde { Q } _ { k } ^ { \pi ^ { * } }$ can be written as a function of $( c _ { k } , u )$ without loss, and the completeness condition $\Pi _ { \mathrm { g l o b a l } } \overset { \sim } { \subseteq } \left\{ \left( \pi _ { 0 } , \ldots , \pi _ { K - 1 } \right) : \pi _ { k } ( \cdot \vert c _ { k } ) \right\}$ in Theorem 1(c) is automatically satisfied.

Assumptions (A1)–(A10) are standard regularity conditions for the convergence of reinforcement learning algorithms; identical or closely analogous conditions appear in the foundational convergence proofs of Q-learning [Watkins and Dayan, 1992, Tsitsiklis, 1994, Jaakkola et al., 1994], on-policy GLIE control [Singh et al., 2000], and the systematic treatment in Bertsekas [2025]. They are not specific to RTPO but rather constitute the minimal set of conditions under which any stochasticapproximation-based value-learning algorithm is known to converge. These assumptions (A1–A11) are intentionally stronger than those required in the practical neural network implementation. They are used to make the convergence argument mathematically clean in the tabular setting. In the realistic RTPO implementation, the policy is represented by a shared neural language model rather than by independent tabular sub-policies. Therefore, “freezing” a turn-level sub-policy should be interpreted operationally: after a turn is completed, subsequent turns mask out the corresponding turn tokens from the loss, so that those turn-level decisions no longer receive gradients. The tabular analysis should thus be read as an idealized counterpart that clarifies the role of reverse-turn optimization and downstream-policy freezing, rather than as a claim of global convergence for arbitrary neural function approximation. The extension to neural function approximation is discussed separately in Proposition C.2. We do not rely on the universal approximation property alone to claim convergence of the neural algorithm; instead, the tabular result serves as a principled limiting case that motivates the reverse-turn training design.

Verification of decision sufficiency (A11) in RTPO. In the practical RTPO implementation, ψ corresponds to the chat-template truncation operator that removes the model’s internal reasoning trace (the content of <think> blocks) while preserving all externally observable elements: tool calls, tool results, and final answers. Two states $S _ { k }$ and $\breve { S } _ { k } ^ { \prime }$ that differ only in their internal reasoning traces satisfy $\psi ( S _ { k } ) = \psi ( S _ { k } ^ { \prime } )$ . Since the environment transition kernel $P _ { H }$ depends exclusively on the executed tool calls and the returned feedback, not on the model’s internal reasoning, the next-state distribution and thus the continuation value $F _ { k } ^ { \pi ^ { * } } ( S _ { k + 1 } )$ are identical for $S _ { k }$ and $S _ { k } ^ { \prime }$ The turn-level reward $r _ { k }$ likewise depends only on the externally observable action. Therefore, $\tilde { Q } _ { k } ^ { \pi ^ { * } } ( S _ { k } , u ) = \tilde { Q } _ { k } ^ { \pi ^ { * } } ( S _ { k } ^ { \prime } , u )$ ) for all $u ,$ and Assumption (A11) is satisfied. When $\psi$ is the identity map (full-history conditioning), (A11) holds trivially.

Turn-level value estimation. We formulate the interaction as a hierarchical MDP, which decomposes the value of each turn into the local effect of the current macro-action and the downstream completion value, following a MAXQ-style value decomposition. The downstream continuation value after executing turn k is defined as

$$
F _ { k } ^ { \pi } ( S _ { k + 1 } ) : = \mathbb { E } _ { \pi } [ \sum _ { j = k + 1 } ^ { K - 1 } \gamma _ { H } ^ { \sum _ { m = k + 1 } ^ { j - 1 } \tau _ { m } } r _ { j } ] S _ { k + 1 } ] , \qquad F _ { K - 1 } ^ { \pi } \equiv 0 .\tag{19}
$$

The base case $F _ { K - 1 } ^ { \pi } \equiv 0$ reflects that there are no downstream turns after the last turn. The exponent $\sum _ { m = k + 1 } ^ { j - 1 } \tau _ { m }$ accounts for the number of token-generation steps elapsed between the next state $S _ { k + 1 }$ and the future reward $r _ { j }$ . When $\gamma _ { H } = 1$ , this reduces to the undiscounted finite-horizon setting used in many sparse-reward agentic RL tasks. The boundary-level augmented action-value function is:

$$
\begin{array} { r } { \tilde { Q } _ { k } ^ { \pi } ( S _ { k } , u _ { k } ) : = \mathbb { E } [ r _ { k } + \gamma _ { H } ^ { \tau _ { k } } F _ { k } ^ { \pi } ( S _ { k + 1 } ) \mid S _ { k } , u _ { k } ] . } \end{array}\tag{20}
$$

This value measures the expected return obtained by choosing the complete turn-k response $u _ { k }$ at state $S _ { k }$ , followed by downstream policy execution from $S _ { k + 1 }$ onward. In sparse-reward settings, the immediate term $r _ { k }$ is often zero for nonterminal turns, so the quality of a turn-level action is primarily reflected through the downstream continuation value. The corresponding optimal value satisfies $\tilde { Q } _ { k } ^ { * } ( S _ { k } , u _ { k } ) = \mathbb { E } [ r _ { k } + \gamma _ { H } ^ { \tau _ { k } } F _ { k } ^ { * } ( S _ { k + 1 } ) \mid S _ { k } , u _ { k } ]$ , where $F _ { k } ^ { * }$ is obtained by the usual backward Bellman optimality recursion over turn boundaries.

## C.2 Proof of Theorem 1: Convergence to Recursive Optimality

We formalize Theorem 1 from Sec 3.1 before presenting the proof.

## Reverse-Order Training to Recursive Optimality

Theorem 1 (Convergence to Recursive Optimality). Under Assumptions (A1)–(A10), includingfinite state and macro-action spaces, Robbins–Monro step sizes, sufficient exploration at each turn, exact freezing of optimized downstream policies, and bounded rewards, the

reverse-order backward induction of turn-level policy optimization satisfies the following properties:

(a) Per-turn convergence under fixed downstream policies. For each turn k, once the downstream policies $\pi _ { k + 1 : K - 1 } ^ { * }$ arefixed, the downstream continuation value becomes a stationaryfunction:

$$
F _ { k } ^ { \pi ^ { * } } ( S _ { k + 1 } ) = \mathbb { E } _ { \pi _ { k + 1 : K - 1 } ^ { * } } [ \sum _ { j = k + 1 } ^ { K - 1 } \gamma _ { H } ^ { \sum _ { m = k + 1 } ^ { j - 1 } \tau _ { m } } r _ { j } ] S _ { k + 1 } ] .\tag{21}
$$

Therefore, the turn-k target

$$
Y _ { k } : = r _ { k } + \gamma _ { H } ^ { \tau _ { k } } { F _ { k } ^ { \pi } } ^ { * } \left( S _ { k + 1 } \right)\tag{22}
$$

is stationary conditional on $( S _ { k } , u _ { k } )$ , with conditional mean

$$
\mathbb { E } [ Y _ { k } \mid S _ { k } , u _ { k } ] = { \tilde { Q } } _ { k } ^ { \pi ^ { * } } ( S _ { k } , u _ { k } ) .\tag{23}
$$

The corresponding tabular update

$$
Q _ { k } ^ { ( n + 1 ) } ( S , u ) \gets ( 1 - \alpha _ { n } ) Q _ { k } ^ { ( n ) } ( S , u ) + \alpha _ { n } Y _ { k }\tag{24}
$$

is a single-step stochastic approximation problem and converges to $\tilde { Q } _ { k } ^ { \pi ^ { * } }$ under the stated assumptions. Hence, the learned per-turn policy satisfies $\hat { \pi } _ { k } \xrightarrow [ ] { \mathrm { a . s . } } \pi _ { k } ^ { * } .$

(b) Recursive optimality under reverse-order training. Applying the per-turn convergence argument in reverse order,

$$
k = K { - } 1 , K { - } 2 , \ldots , 0 ,\tag{25}
$$

yields a sequence of optimized and frozen downstream policies. At each turn k, the per-turn optimizer converges to

$$
\pi _ { k } ^ { * } ( S _ { k } ) \in \arg \operatorname* { m a x } _ { u _ { k } \in A _ { H , k } ( S _ { k } ) } \tilde { Q } _ { k } ^ { \pi ^ { * } } ( S _ { k } , u _ { k } ) ,\tag{26}
$$

given the fixed downstream policies $\pi _ { k + 1 } ^ { * } , \ldots , \pi _ { K - 1 } ^ { * }$ . The resulting policy sequence

$$
\pi ^ { \mathrm { r e c } } = ( \pi _ { 0 } ^ { * } , \ldots , \pi _ { K - 1 } ^ { * } )\tag{27}
$$

is recursively optimal.

(c) Global optimality under context-sufficient macro-action completeness. Ifthe turn-level macro-action spaces are complete with respect to the conditioning contexts $c _ { k } = \psi ( S _ { k } )$ i.e., every globally feasible trajectory-level policy can be represented by a sequence of turn-level policies acting on these contexts,

$$
\Pi _ { \mathrm { g l o b a l } } \subseteq \left\{ ( \pi _ { 0 } , \ldots , \pi _ { K - 1 } ) : \pi _ { k } ( \cdot  { | } c _ { k } ) , u _ { k } \in  { A _ { H , k } } ( S _ { k } ) \right\} ,\tag{28}
$$

Then, recursive optimality is equivalent to global optimality over the full trajectory:

$$
\begin{array} { r } { V ^ { \pi ^ { \mathrm { r e c } } } ( \bar { x } _ { 0 } ) = V ^ { * } ( \bar { x } _ { 0 } ) . } \end{array}\tag{29}
$$

Proof. We prove parts (a)–(c) of Theorem 1 as follows.

Proof of Theorem 1(a): per-turn convergence under fixed downstream policies. Fix a turn k and suppose the downstream sub-policies $\pi _ { k + 1 } ^ { * } , \ldots , \pi _ { K - 1 } ^ { * }$ are already optimized and frozen. By the definition in Theorem 1, the downstream continuation value is

$$
F _ { k } ^ { \pi ^ { * } } ( S _ { k + 1 } ) = \mathbb { E } _ { \pi _ { k + 1 : K - 1 } ^ { * } } [ \sum _ { j = k + 1 } ^ { K - 1 } \gamma _ { H } ^ { \sum _ { m = k + 1 } ^ { j - 1 } \tau _ { m } } r _ { j } ] S _ { k + 1 } ] .\tag{30}
$$

Because the downstream policies $\pi _ { k + 1 : K - 1 } ^ { * }$ are frozen, $F _ { k } ^ { \pi ^ { * } } ( S _ { k + 1 } )$ is a fixed function of $S _ { k + 1 }$ throughout turn k. Define the augmented one-step target

$$
Y _ { k } = r _ { k } + \gamma _ { H } ^ { \tau _ { k } } F _ { k } ^ { \pi ^ { * } } ( S _ { k + 1 } ) .\tag{31}
$$

Conditional on $( S _ { k } , u _ { k } )$ , the distribution of $r _ { k } , \tau _ { k }$ , and $S _ { k + 1 }$ is induced by the time-invariant turnboundary transition kernel $P _ { H }$ . Since $F _ { k } ^ { \pi ^ { * } }$ is fixed throughout turn $k ,$ the conditional distribution of $Y _ { k }$ given $( S _ { k } , u _ { k } )$ is stationary.

The conditional mean of the target is exactly the augmented turn-level action value:

$$
\mathbb { E } [ Y _ { k } \mid S _ { k } , u _ { k } ] = \mathbb { E } \Big [ r _ { k } + \gamma _ { H } ^ { \tau _ { k } } { F _ { k } ^ { \pi ^ { * } } ( S _ { k + 1 } ) } \Big \mid S _ { k } , u _ { k } \Big ] = \tilde { Q } _ { k } ^ { \pi ^ { * } } ( S _ { k } , u _ { k } ) .\tag{32}
$$

The target also has a bounded second moment. Since rewards are bounded, $| r _ { j } | \leq r _ { \operatorname* { m a x } }$ almost surely. Hence, for any policy π,

$$
\begin{array} { r l r } & { } & { | F _ { k } ^ { \pi } ( S _ { k + 1 } ) | = | \mathbb { E } _ { \pi } [ \displaystyle \sum _ { j = k + 1 } ^ { K - 1 } \gamma _ { H } ^ { \sum _ { m = k + 1 } ^ { j - 1 } \tau _ { m } } r _ { j } | S _ { k + 1 } ] | } \\ & { } & { \leq \mathbb { E } _ { \pi } [ \displaystyle \sum _ { j = k + 1 } ^ { K - 1 } \gamma _ { H } ^ { \sum _ { m = k + 1 } ^ { j - 1 } \tau _ { m } } | r _ { j } | | S _ { k + 1 } ] } \\ & { } & { \leq ( K - k - 1 ) r _ { \operatorname* { m a x } } \leq K r _ { \operatorname* { m a x } } , } \end{array}\tag{33}
$$

where we used $\gamma _ { H } \in ( 0 , 1 ]$ and the finite horizon K. Therefore,

$$
\begin{array} { r } { | Y _ { k } | \leq | r _ { k } | + \gamma _ { H } ^ { \tau _ { k } } | F _ { k } ^ { \pi ^ { * } } ( S _ { k + 1 } ) | \leq ( K + 1 ) r _ { \operatorname* { m a x } } , } \end{array}\tag{34}
$$

so $Y _ { k }$ has uniformly bounded conditional second moment.

During turn k, the tabular update is

$$
Q _ { k } ^ { ( n + 1 ) } ( S , u ) \gets ( 1 - \alpha _ { n } ) Q _ { k } ^ { ( n ) } ( S , u ) + \alpha _ { n } Y _ { k } .\tag{35}
$$

For each fixed $( S , u , k )$ , this is a Robbins–Monro stochastic approximation to the stationary conditional mean $\mathbb { E } [ Y _ { k } \mid S , u ] = \tilde { Q } _ { k } ^ { \pi ^ { * } } ( S , u )$ . The finite state and macro-action spaces imply that there are finitely many entries to estimate. Sufficient exploration ensures that every relevant $( S , u )$ pair is visited infinitely often during turn $k .$ . The step sizes satisfy

$$
\sum _ { n = 1 } ^ { \infty } \alpha _ { n } ( S , u , k ) = \infty , \qquad \sum _ { n = 1 } ^ { \infty } \alpha _ { n } ^ { 2 } ( S , u , k ) < \infty .\tag{36}
$$

Together with bounded targets and bounded iterates, standard stochastic approximation gives

$$
Q _ { k } ^ { ( n ) } ( S , u ) \xrightarrow { \mathrm { a . s . } } \tilde { Q } _ { k } ^ { \pi ^ { * } } ( S , u ) \qquad \mathrm { f o r e v e r y r e a c h a b l e } ( S , u ) .\tag{37}
$$

As exploration vanishes, the induced greedy policy converges to an optimal greedy selector:

$$
\hat { \pi } _ { k } ( S ) \in \arg \operatorname* { m a x } _ { u } Q _ { k } ^ { ( n ) } ( S , u ) \quad \Longrightarrow \quad \hat { \pi } _ { k } ( S ) \xrightarrow { \mathrm { a . s . } } \pi _ { k } ^ { * } ( S ) \in \arg \operatorname* { m a x } _ { u } \tilde { Q } _ { k } ^ { \pi ^ { * } } ( S , u ) .\tag{38}
$$

This completes the proof of Theorem 1(a).

Proof of Theorem 1(b): reverse-order recursion gives recursive optimality. We now apply Theorem 1(a) in reverse order here. At the final turn $k = K - 1$ , there are no downstream turns, so

$$
F _ { K - 1 } ^ { \pi } ( { \cal S } _ { K } ) \equiv 0 .\tag{39}
$$

The augmented target reduces to

$$
Y _ { K - 1 } = r _ { K - 1 } .\tag{40}
$$

By claim Theorem 1(a), the final-turn policy converges:

$$
\hat { \pi } _ { K - 1 } \xrightarrow { \mathrm { a . s . } } \pi _ { K - 1 } ^ { * } .\tag{41}
$$

Now assume that for some $k \in \{ 0 , \ldots , K - 2 \}$ , the downstream policies $\hat { \pi } _ { k + 1 } , \hdots , \hat { \pi } _ { K - 1 }$ have already converged almost surely to $\pi _ { k + 1 } ^ { * } , \ldots , \pi _ { K - 1 } ^ { * }$ and have been frozen before turn k begins. Then the continuation value

$$
F _ { k } ^ { \pi ^ { * } } ( S _ { k + 1 } ) = \mathbb { E } _ { \pi _ { k + 1 : K - 1 } ^ { * } } [ \sum _ { j = k + 1 } ^ { K - 1 } \gamma _ { H } ^ { \sum _ { m = k + 1 } ^ { j - 1 } \tau _ { m } } r _ { j } ] S _ { k + 1 } ]\tag{42}
$$

is fixed throughout turn k. Therefore, claim (a) applies to turn $k ,$ giving

$$
\hat { \pi } _ { k } \xrightarrow { \mathrm { a . s . } } \pi _ { k } ^ { * } , \qquad \pi _ { k } ^ { * } ( S _ { k } ) \in \arg \operatorname* { m a x } _ { u _ { k } } \tilde { Q } _ { k } ^ { \pi ^ { * } } ( S _ { k } , u _ { k } ) .\tag{43}
$$

By backward induction from $K - 1$ to $0 ,$ every turn-level policy converges to its optimal selector given the optimized and frozen downstream policies. Therefore, the resulting sequence

$$
\pi ^ { \mathrm { r e c } } = ( \pi _ { 0 } ^ { * } , \ldots , \pi _ { K - 1 } ^ { * } )\tag{44}
$$

is recursively optimal.

Till here, we have completed the proof of Theorem 1(b).

Proof of Theorem 1(c): context-sufficient macro-action completeness implies global optimality. Assume the turn-level macro-action spaces are complete with respect to the conditioning contexts $c _ { k } \ : = \ : \psi ( S _ { k } )$ , meaning that every globally feasible trajectory-level policy can be represented as a sequence of turn-level sub-policies $\left( \pi _ { 0 } , \ldots , \pi _ { K - 1 } \right)$ , with each $\pi _ { k }$ acting on $c _ { k }$ and selecting a feasible macro-action $u _ { k } \in \mathcal A _ { H , k } ( S _ { k } )$

We prove by backward induction that

$$
V _ { k } ^ { \pi ^ { \mathrm { r e c } } } ( S _ { k } ) = V _ { k } ^ { * } ( S _ { k } ) \qquad { \mathrm { f o r ~ e v e r y ~ r e a c h a b l e ~ } } S _ { k } .\tag{45}
$$

At the last turn, $F _ { K - 1 } ^ { \pi } \equiv 0$ , so

$$
\begin{array} { r } { \tilde { Q } _ { K - 1 } ^ { \pi } ( S _ { K - 1 } , u ) = \mathbb { E } [ r _ { K - 1 } \mid S _ { K - 1 } , u ] . } \end{array}\tag{46}
$$

Since $\pi _ { K - 1 } ^ { * }$ maximizes this quantity,

$$
V _ { K - 1 } ^ { \pi ^ { \mathrm { r e c } } } ( S _ { K - 1 } ) = \operatorname* { m a x } _ { u \in { \mathcal A } _ { H , K - 1 } ( S _ { K - 1 } ) } \mathbb E [ r _ { K - 1 } \mid S _ { K - 1 } , u ] = V _ { K - 1 } ^ { * } ( S _ { K - 1 } ) .\tag{47}
$$

Assume now that $V _ { k + 1 } ^ { \pi ^ { \mathrm { r e c } } } ( S _ { k + 1 } ) = V _ { k + 1 } ^ { * } ( S _ { k + 1 } )$ for every reachable $S _ { k + 1 }$ . Then

$$
F _ { k } ^ { \pi ^ { \mathrm { r e c } } } ( S _ { k + 1 } ) = F _ { k } ^ { * } ( S _ { k + 1 } ) .\tag{48}
$$

By Assumption (A11), the optimal action-value $\tilde { Q } _ { k } ^ { \pi ^ { * } } ( S _ { k } , u )$ depends on $S _ { k }$ only through $c _ { k } = \psi ( S _ { k } )$ Therefore, the greedy policy $\pi _ { k } ^ { * } \in$ arg max<sub>u</sub> $\tilde { Q } _ { k } ^ { \pi ^ { * } } ( S _ { k } , u )$ is well-defined as a function of $c _ { k }$ alone, and we write $\pi _ { k } ^ { * } ( c _ { k } )$ without ambiguity. Applying this to the inductive step:

$$
\begin{array} { r l } & { V _ { k } ^ { \pi ^ { \mathrm { r e c } } } ( S _ { k } ) = \tilde { Q } _ { k } ^ { \pi ^ { \mathrm { r e c } } } \big ( S _ { k } , \pi _ { k } ^ { * } ( c _ { k } ) \big ) } \\ & { \qquad = \mathbb { E } \Big [ r _ { k } + \gamma _ { H } ^ { \tau _ { k } } F _ { k } ^ { \pi ^ { \mathrm { r e c } } } ( S _ { k + 1 } ) \Big \vert \ S _ { k } , \pi _ { k } ^ { * } ( S _ { k } ) \Big ] } \\ & { \qquad = \mathbb { E } [ r _ { k } + \gamma _ { H } ^ { \tau _ { k } } F _ { k } ^ { * } ( S _ { k + 1 } ) \vert \ S _ { k } , \pi _ { k } ^ { * } ( S _ { k } ) ] } \\ & { \qquad = \underset { u \in \mathcal { A } _ { H , k } ( S _ { k } ) } { \operatorname* { m a x } } \mathbb { E } [ r _ { k } + \gamma _ { H } ^ { \tau _ { k } } F _ { k } ^ { * } ( S _ { k + 1 } ) \vert \ S _ { k } , u ] } \\ & { \qquad = V _ { k } ^ { * } ( S _ { k } ) . } \end{array}\tag{49}
$$

Thus $V _ { k } ^ { \pi ^ { \mathrm { r e c } } } ( S _ { k } ) = V _ { k } ^ { * } ( S _ { k } )$ for all k by induction. In particular,

$$
\begin{array} { r } { V ^ { \pi ^ { \mathrm { r e c } } } ( \bar { x } _ { 0 } ) = V ^ { * } ( \bar { x } _ { 0 } ) . } \end{array}\tag{50}
$$

This proves Theorem 1(c), and completes the proof of Theorem 1.

□

Interpretation. As the key technical point of this proof, turn-k target $Y _ { k } = r _ { k } + \gamma _ { H } ^ { \tau _ { k } } F _ { k } ^ { \pi ^ { * } } ( S _ { k + 1 } )$ is stationary only, when the downstream policies $\pi _ { k + 1 } ^ { * } , \ldots , \pi _ { K - 1 } ^ { * }$ are already optimized and frozen. Reverse-order training provides exactly this condition. Without reverse-order freezing, $F _ { k } ^ { \pi }$ would change during turn $k ,$ the target $Y _ { k }$ would no longer be stationary, and the single-turn stochastic approximation argument would not apply.

Proposition of Theorem 1: Asymptotic recursive optimality under neural function approximation. The convergence guarantee in Theorem 1 relies on Assumption (A4) from Appendix C.1, which postulates a tabular representation of $Q _ { k }$ . In the practical implementation of RTPO, $Q _ { k }$ is realized implicitly by a neural sub-policy $\pi _ { \theta _ { k } }$ operating on the conditioning context $c _ { k } = \psi ( S _ { k } )$ . We now state the corresponding asymptotic guarantee under the neural function approximation; the proof follows the standard reduction from tabular Q-learning to projected Q-learning under a function class $\mathcal { F } _ { \theta }$ . This statement is formalized as a Proposition, which appears below as a neural-function-approximation extension of Theorem 1.

Proposition of Theorem 1: Asymptotic recursive optimality under neural function approximation.

Suppose the assumptions of Theorem 1 hold, with current assumption (A4) from Appendix C.1 replaced by new assumption (A4’) below:

(A4’) Universal approximation. For every fixed $F _ { k } ^ { \pi ^ { * } }$ , there exists $\theta _ { k } ^ { * } \in \Theta$ such that the induced greedy policy $\pi _ { \theta _ { k } ^ { * } }$ realizes $\pi _ { k } ^ { * }$ on every reachable $S _ { k }$ in the support of the visitation distribution.

Then, reverse-order training of RTPO produces a policy sequence $\big ( \hat { \pi } _ { 0 } , \dots , \hat { \pi } _ { K - 1 } \big )$ that asymptotically realizes the recursively optimal policy. That ${ \mathrm { i s } } ,$ for every k and every reachable $S _ { k }$

$$
\operatorname* { l i m } _ { n \to \infty } \pi _ { \theta _ { k } ^ { ( n ) } } ( \cdot \mid c _ { k } ) = \pi _ { k } ^ { * } ( \cdot \mid c _ { k } ) \qquad \mathrm { a . s . }
$$

Proof: Proposition ofTheorem 1.

Step 1. Reduction to projected Q-learning: Replace the tabular update with a parametric update on the function class $\{ \pi _ { \theta _ { k } } : \theta _ { k } \in \Theta \}$ ,

$$
\theta _ { k } ^ { ( n + 1 ) }  \theta _ { k } ^ { ( n ) } - \alpha _ { n } \nabla _ { \theta _ { k } } \mathbb { E } _ { ( S , u ) \sim \mu _ { n } } [ ( Q _ { \theta _ { k } ^ { ( n ) } } ( S , u ) - \widetilde { r } _ { k } ) ^ { 2 } ] ,\tag{51}
$$

where $\mu _ { n }$ is the visitation distribution induced by the current greedy-with-exploration policy and $Q _ { \theta _ { k } }$ is the implicit Q-function realized by $\pi _ { \theta _ { k } }$ . The fixed point of this update is the projection of $\widetilde { Q } _ { k } ^ { * }$ onto the function class, $\Pi _ { \Theta } \widetilde { Q } _ { k } ^ { * }$ , in the weighted $L ^ { 2 } ( \mu )$ sense.

Step 2: Asymptotic realizability: By assumption (A4’), there exists $\theta _ { k } ^ { * } \in \Theta$ such that the induced greedy policy realizes $\pi _ { k } ^ { * }$ on every reachable $S _ { k }$ in the support of $\mu .$ This implies $\Pi _ { \Theta } \widetilde Q _ { k } ^ { * } = \widetilde Q _ { k } ^ { * }$ on the support of $\mu ,$ so the projection error vanishes for the quantities relevant to the induced greedy policy. Combined with standard asymptotic results for stochastic gradient descent on smooth nonconvex objectives under Robbins–Monro step sizes [Robbins and Monro, 1951], $Q _ { \theta _ { k } ^ { ( n ) } } ( S , u )  \widetilde { Q } _ { k } ^ { * } ( S , u )$ a.s. for every $( S , u )$ in the support of $\mu .$

Step 3. Backward induction in the function-class regime: Replace each step in the proof of Theorem 1 by Steps 1–2 above. The reverse induction structure is unchanged: the determinism of $F _ { k } ^ { \pi ^ { * } }$ under frozen downstream sub-policies is purely a property of the SMDP and the freezing schedule $( \mathsf { A } 5 )$ , independent of whether $Q _ { k }$ is tabular or parametric. Hence, the backward induction goes through and yields $\pi _ { \boldsymbol { \theta } _ { k } ^ { ( n ) } } \to \pi _ { k } ^ { * }$ a.s. on the support of $\mu$ for every k. □

Remark on the assumption gap between $( A 4 )$ and $( A 4 ^ { \prime } )$ . Assumption $( \mathsf { A } 4 ^ { \cdot } )$ is strictly weaker than what would be required for finite-sample convergence rates: it asks only that the function class be rich enough to contain the optimal sub-policy in the asymptotic limit, not that the gradient dynamics realize this optimum at any finite iteration. For the modern Transformer architectures used in our experiments with Qwen3 models, the universal approximation property is widely accepted as a working assumption, and our experimental results in Sec 4 provide empirical evidence that the asymptotic guarantee transfers to practice.

## C.3 Proof of Theorem 2: Causally Consistent Turn-Level Advantage Estimation

We formalize Theorem 2 from Sec 3.2 before presenting the proof. We compare the state-matched sibling advantage estimator used by RTPO with the trajectory-level advantage estimator used in flat-trajectory training. The key distinction is whether the baseline is computed from rollouts sharing the same boundary state $S _ { k }$ , or from complete trajectories that may have reached different boundary states by turn k.

Turn-Level Credit Assignment without State Bias

Theorem 2 (Causally Consistent Turn-Level Advantage Estimation). Consider turn $k$ and suppose that $G - 1$ sibling rollouts areforkedfrom the same boundary state $S _ { k }$ . Each sibling rollout $j \in \{ 1 , \dots , G ^ { - } 1 \}$ independently samples a turn-k macro-action $u _ { j , k } \equiv l _ { j , k }$ and then continues to the terminal. Let

$$
\hat { Q } _ { j , k } = R _ { j } , \qquad \hat { V } _ { k } = \frac { 1 } { G - 1 } \sum _ { j = 1 } ^ { G - 1 } \hat { Q } _ { j , k } , \qquad A _ { j , k } ^ { H } = \hat { Q } _ { j , k } - \hat { V } _ { k }\tag{52}
$$

denote the Monte Carlo turn-level action-value estimate, the state-matched sibling baseline, and the resulting turn-level advantagefor sibling rollout $j ,$ respectively. Let the corresponding flat trajectory-level estimator be

$$
\bar { R } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } R _ { i } , \qquad A _ { i } ^ { \mathrm { t r a j } } = R _ { i } - \bar { R } ,\tag{53}
$$

where $g _ { i }$ denotes the i-th complete trajectory and $S _ { i , k }$ is its boundary state at turn k. Define

$$
\begin{array} { r } { Q ^ { \pi } ( S _ { k } , u _ { j , k } ) = \mathbb { E } [ R \mid S _ { k } , u _ { j , k } ] , \qquad V ^ { \pi } ( S _ { k } ) = \mathbb { E } _ { u \sim \pi ( \cdot \mid c _ { k } ) } [ Q ^ { \pi } ( S _ { k } , u ) ] , } \end{array}\tag{54}
$$

and

$$
A ^ { \pi } ( S _ { k } , u _ { j , k } ) = Q ^ { \pi } ( S _ { k } , u _ { j , k } ) - V ^ { \pi } ( S _ { k } ) , \qquad \mu _ { R } = \mathbb { E } [ R ] .\tag{55}
$$

Then, under bounded rewards and independent sibling sampling, the RTPO sibling advantage estimator satisfies the following properties:

(a) Local unbiasedness up tofinite-group bias. Conditional on the shared boundary state $S _ { k }$ and the sampled macro-action $u _ { j , k } ,$ the state-matched sibling estimator satisfies:

$$
\mathbb { E } \big [ A _ { j , k } ^ { H } \ | \ S _ { k } , u _ { j , k } \big ] = \frac { G - 2 } { G - 1 } A ^ { \pi } ( S _ { k } , u _ { j , k } ) .\tag{56}
$$

Thus, $A _ { j , k } ^ { H }$ estimates the true turn-level advantage up to a finite-group bias of order $O ( 1 / G )$ and contains no upstream state-contamination term. In contrast, the trajectorylevel estimator satisfies:

$$
\mathbb { E } \bigg [ A _ { i } ^ { \mathrm { t r a j } } \mid S _ { i , k } , u _ { i , k } \bigg ] = \frac { G - 1 } { G } A ^ { \pi } ( S _ { i , k } , u _ { i , k } ) + \frac { G - 1 } { G } \big ( V ^ { \pi } ( S _ { i , k } ) - \mu _ { R } \big ) ,\tag{57}
$$

where the second term is the upstream state-contamination term.

(b) Reduced value estimation error. Let

$$
\sigma _ { V } ^ { 2 } : = \operatorname { V a r } _ { S _ { k } } [ V ^ { \pi } ( S _ { k } ) ]\tag{58}
$$

denote the variance of boundary-state values. Compared with the trajectory-level estimator, the sibling estimator removes the cross-state value component $V ^ { \tilde { \pi } } ( S _ { i , k } ) - \mu _ { R }$ Therefore, under a matched downstream-noise comparison, the trajectory estimator contains an additional MSE contribution proportional to $\sigma _ { V } ^ { 2 }$ , while the sibling estimator does not. In particular, when $\sigma _ { V } ^ { 2 } > 0 ,$ , the state-matched estimator removes a non-zero source ofvalue-estimation error induced by cross-state baselines.

(c) State-matched causal actions. All sibling rollouts share the same boundary state $S _ { k }$ Therefore, differences in their returns are attributable to the sampled turn-k macroactions $u _ { j , k }$ and independent downstream sampling noise, rather than to upstream trajectory differences. The resulting advantage $A _ { j , k } ^ { \bar { H } }$ is assigned only to the output tokens of turn k,

$$
l _ { j , k } = ( a _ { j , k , 1 } , \ldots , a _ { j , k , T _ { j , k } } ) ,\tag{59}
$$

while the prefix tokens contained in the shared context $c _ { k } = \psi ( S _ { k } )$ are excludedfrom the gradient.

Proof. We prove parts (a)–(c) of Theorem 2 as follows.

Proof of Theorem ${ \bf \nabla } 2 ( { \bf a } ) { \bf : \ r }$ : local unbiasedness up to finite-group bias. Fix a turn k and a shared boundary state $S _ { k }$ . RTPO forks $G - 1$ sibling rollouts from the same environment snapshot snap . Conditional on $S _ { k }$ , each sibling rollout independently samples its turn-k macro-action and downstream continuation. Thus, for any two distinct sibling rollouts $j \neq j ^ { \prime }$ , the return $R _ { j ^ { \prime } }$ of sibling $j ^ { \prime }$ is conditionally independent of the action $u _ { j , k }$ sampled by sibling j:

$$
R _ { j ^ { \prime } } \perp \perp u _ { j , k } \mid S _ { k } .\tag{60}
$$

Consequently,

$$
\begin{array} { r } { \mathbb { E } [ R _ { j ^ { \prime } } \ | \ S _ { k } , u _ { j , k } ] = \mathbb { E } [ R _ { j ^ { \prime } } \ | \ S _ { k } ] = V ^ { \pi } ( S _ { k } ) . } \end{array}\tag{61}
$$

Meanwhile, for the own return of sibling j,

$$
\mathbb { E } [ R _ { j } \ | \ S _ { k } , u _ { j , k } ] = Q ^ { \pi } ( S _ { k } , u _ { j , k } ) .\tag{62}
$$

By definition,

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { j , k } ^ { H } = R _ { j } - \frac { 1 } { G - 1 } \sum _ { r = 1 } ^ { G - 1 } R _ { r } } } \\ { { \displaystyle ~ = \frac { G - 2 } { G - 1 } R _ { j } - \frac { 1 } { G - 1 } \sum _ { r \neq j } R _ { r } } . } \end{array}\tag{63}
$$

Taking conditional expectation given $( S _ { k } , u _ { j , k } )$ and using Eq. (61) and Eq. (62),

$$
\begin{array} { l } { { \mathbb { E } [ A _ { j , k } ^ { H } \ | \ S _ { k } , u _ { j , k } ] = \displaystyle \frac { G - 2 } { G - 1 } Q ^ { \pi } ( S _ { k } , u _ { j , k } ) - \displaystyle \frac { G - 2 } { G - 1 } V ^ { \pi } ( S _ { k } ) } } \\ { { \displaystyle \quad = \frac { G - 2 } { G - 1 } \big ( Q ^ { \pi } ( S _ { k } , u _ { j , k } ) - V ^ { \pi } ( S _ { k } ) \big ) } } \\ { { \displaystyle \quad = \frac { G - 2 } { G - 1 } A ^ { \pi } ( S _ { k } , u _ { j , k } ) . } } \end{array}\tag{64}
$$

Therefore, the sibling estimator is locally unbiased up to the finite-group multiplicative factor $( G - 2 ) / ( G - 1 )$ . Equivalently, the conditional bias relative to the true turn-level advantage is

$$
\mathbb { E } [ A _ { j , k } ^ { H } \ | \ S _ { k } , u _ { j , k } ] - A ^ { \pi } ( S _ { k } , u _ { j , k } ) = - \frac { 1 } { G - 1 } A ^ { \pi } ( S _ { k } , u _ { j , k } ) ,\tag{65}
$$

which is $O ( 1 / G )$ and contains no term depending on $V ^ { \pi } ( S _ { k } ) - \mu _ { R }$

We now contrast this with the trajectory-level estimator. In flat-trajectory training, the $G$ trajectories are independently sampled from the same prompt but may reach different boundary states by turn k. The trajectory-level advantage is

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { i } ^ { \mathrm { t r a j } } = R _ { i } - \frac { 1 } { G } \sum _ { r = 1 } ^ { G } R _ { r } } } \\ { { \displaystyle ~ = \frac { G - 1 } { G } R _ { i } - \frac { 1 } { G } \sum _ { r \not = i } R _ { r } } . } \end{array}\tag{66}
$$

The own return satisfies

$$
\mathbb { E } [ R _ { i } \mid S _ { i , k } , u _ { i , k } ] = Q ^ { \pi } ( S _ { i , k } , u _ { i , k } ) .\tag{67}
$$

For $r \neq i ,$ trajectory r is generated by an independent upstream rollout and reaches its own boundary state $S _ { r , k }$ . Thus,

$$
\begin{array} { r } { { \mathbb { E } } [ R _ { r } \ | \ S _ { i , k } , u _ { i , k } ] = { \mathbb { E } } _ { S _ { r , k } } [ V ^ { \pi } ( S _ { r , k } ) ] = \mu _ { R } . } \end{array}\tag{68}
$$

Substituting Eq. (67) and Eq. (68) into Eq. (66) gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ A _ { i } ^ { \mathrm { t r a j } } \ | \ S _ { i , k } , u _ { i , k } ] = \frac { G - 1 } { G } \big ( Q ^ { \pi } ( S _ { i , k } , u _ { i , k } ) - \mu _ { R } \big ) } \\ { \displaystyle \quad = \frac { G - 1 } { G } \big ( Q ^ { \pi } ( S _ { i , k } , u _ { i , k } ) - V ^ { \pi } ( S _ { i , k } ) \big ) + \frac { G - 1 } { G } \big ( V ^ { \pi } ( S _ { i , k } ) - \mu _ { R } \big ) } \\ { \displaystyle \quad = \frac { G - 1 } { G } A ^ { \pi } ( S _ { i , k } , u _ { i , k } ) + \frac { G - 1 } { G } \big ( V ^ { \pi } ( S _ { i , k } ) - \mu _ { R } \big ) . } \end{array}\tag{69}
$$

The second term in Eq. (69) depends only on the upstream boundary state $S _ { i , k }$ and the global mean return $\mu _ { R } .$ . It has no causal dependence on the current turn action $u _ { i , k }$ and is precisely the cross-state contamination term. Therefore, the sibling estimator removes the upstream state-contamination term present in the trajectory-level estimator. This proves part (a) of Theorem 2.

Proof of Theorem 2(b): reduced value estimation error. We compare the error of each estimator against its corresponding true turn-level advantage. For the sibling estimator, the target is $A ^ { \pi } ( \bar { S _ { k } , u _ { j , k } } ) \mathrm { : }$ ; for the trajectory estimator, the target is $A ^ { \pi } ( S _ { i , k } , u _ { i , k } )$ . The trajectory estimator contains the additional cross-state term

$$
C _ { i , k } : = \frac { G - 1 } { G } \big ( V ^ { \pi } ( S _ { i , k } ) - \mu _ { R } \big ) ,\tag{70}
$$

whereas the sibling estimator does not contain such a term. Marginalizing over the boundary-state distribution,

$$
\mathbb { E } [ C _ { i , k } ] = 0 , \qquad \mathrm { V a r } ( C _ { i , k } ) = \left( \frac { G - 1 } { G } \right) ^ { 2 } \mathrm { V a r } _ { S _ { k } } [ V ^ { \pi } ( S _ { k } ) ] = \left( \frac { G - 1 } { G } \right) ^ { 2 } \sigma _ { V } ^ { 2 } .\tag{71}
$$

Thus, cross-trajectory baselines introduce an additional MSE component proportional to $\sigma _ { V } ^ { 2 }$ , while state-matched sibling baselines remove it.

In addition to the squared-bias contribution above, the trajectory-level baseline $\begin{array} { r } { \bar { R } = \frac { 1 } { G } \sum _ { r \neq i } R _ { r } } \end{array}$ introduces further variance from cross-state effects. Conditional on $( S _ { i , k } , u _ { i , k } )$ , each $R _ { r } \left( r \neq i \right)$ is independent of the conditioning, so its conditional variance equals its unconditional variance:

$$
\operatorname { V a r } ( R _ { r } \mid S _ { i , k } , u _ { i , k } ) = \operatorname { V a r } ( R _ { r } ) = \underbrace { \mathbb { E } _ { S _ { r , k } } [ \operatorname { V a r } ( R _ { r } \mid S _ { r , k } ) ] } _ { \bar { \sigma } _ { \mathrm { d o w n } } ^ { 2 } } + \sigma _ { V } ^ { 2 } ,\tag{72}
$$

where the $\sigma _ { V } ^ { 2 }$ term arises because $S _ { r , k }$ varies across trajectories. This contributes $\frac { G - 1 } { G ^ { 2 } } \sigma _ { V } ^ { 2 }$ to the conditional variance of $A _ { i } ^ { \mathrm { t r a j } }$ . By contrast, in the sibling estimator, all returns share the same boundary state $S _ { k }$ , so $\mathrm { V a r } ( R _ { r } \mid S _ { k } )$ is a purely within-state quantity and contains no $\sigma _ { V } ^ { 2 }$ component.

More explicitly, the mean squared error can be decomposed as

$$
\operatorname { M S E } ( { \hat { A } } ) = \operatorname { V a r } ( { \hat { A } } ) + \operatorname { B i a s } ( { \hat { A } } ) ^ { 2 } .\tag{73}
$$

For the trajectory estimator, the bias contains the cross-state term $C _ { i , k }$ from Eq. (70). For the sibling estimator, the only systematic bias from part (a) is the finite-group term

$$
- \frac { 1 } { G - 1 } A ^ { \pi } ( S _ { k } , u _ { j , k } ) ,\tag{74}
$$

which is $O ( 1 / G )$ . Therefore, under a matched downstream-noise comparison, the total $\sigma _ { V } ^ { 2 }$ coefficient in the trajectory estimator’s marginal MSE combines the squared-bias contribution (Eq. (71)) with the baseline-variance contribution (Eq. (72)):

$$
\underbrace { \left( { \frac { G - 1 } { G } } \right) ^ { 2 } } _ { \mathrm { B i a s } ^ { 2 } } + \underbrace { { \frac { G - 1 } { G ^ { 2 } } } } _ { \mathrm { V a r } } = { \frac { ( G - 1 ) ^ { 2 } + ( G - 1 ) } { G ^ { 2 } } } = { \frac { G - 1 } { G } } .\tag{75}
$$

The sibling estimator incurs no $\sigma _ { V } ^ { 2 }$ contribution from either source. The leading MSE difference is thus:

$$
\mathrm { M S E } ( A _ { i } ^ { \mathrm { t r a j } } ) - \mathrm { M S E } ( A _ { j , k } ^ { H } ) = \frac { G - 1 } { G } \sigma _ { V } ^ { 2 } + \Delta _ { \mathrm { d o w n } } - O ( 1 / G ^ { 2 } ) ,\tag{76}
$$

where $\Delta _ { \mathrm { d o w n } } \geq 0$ collects differences in downstream Monte Carlo noise between the two estimators under matched conditions. Hence, whenever $\sigma _ { V } ^ { 2 } > 0$ , the state-matched sibling estimator has strictly lower MSE. This proves part (b) of Theorem 2.

Proof of Theorem 2(c): state-matched causal actions. All sibling rollouts are forked from the same boundary state $S _ { k }$ . Therefore, conditional on $S _ { k }$ , sibling rollou $\cdot j$ and sibling rollout $j ^ { \prime }$ differ only in their sampled turn-k macro-actions and their independent downstream randomness:

$$
\begin{array} { r l } { \big ( u _ { j , k } , \xi _ { j } \big ) } & { { } \mathrm { v e r s u s } \quad ( u _ { j ^ { \prime } , k } , \xi _ { j ^ { \prime } } ) , } \end{array}\tag{77}
$$

where $\xi _ { j }$ denotes the downstream randomness of sibling $j .$ Since the shared prefix state $S _ { k }$ is identical across siblings, any systematic difference in conditional expected return is attributable to the sampled turn-k macro-action:

$$
\begin{array} { r } { \mathbb { E } [ R _ { j } - R _ { j ^ { \prime } } \ | \ S _ { k } , u _ { j , k } , u _ { j ^ { \prime } , k } ] = Q ^ { \pi } ( S _ { k } , u _ { j , k } ) - Q ^ { \pi } ( S _ { k } , u _ { j ^ { \prime } , k } ) . } \end{array}\tag{78}
$$

The downstream randomness contributes to Monte Carlo noise but does not create an upstream state bias, because the upstream state is shared and fixed.

Finally, RTPO assigns the resulting advantage only to the output tokens of turn k. For sibling rollout $j ,$ these tokens are

$$
l _ { j , k } = ( a _ { j , k , 1 } , \ldots , a _ { j , k , T _ { j , k } } ) .\tag{79}
$$

The shared prefix $c _ { k } = \psi ( S _ { k } )$ is used only as the conditioning input and is excluded from the gradient. Equivalently, the gradient support of the turn-k objective satisfies

$$
\begin{array} { r } { \operatorname { s u p p } ( \nabla _ { \theta } J _ { k } ) \subseteq \{ ( j , k , t ) : j \in \{ 1 , \dots , G - 1 \} , t \in \{ 1 , \dots , T _ { j , k } \} \} . } \end{array}\tag{80}
$$

Thus, prefix tokens receive zero gradient, and the credit signal is assigned only to the sampled turn-k output action. This proves part (c) of Theorem $^ { 2 , }$ and completes the proof of Theorem 2. □

Interpretation. The proof shows that the key difference between RTPO and flat-trajectory training is the baseline state. The sibling estimator compares alternative turn-k actions from the same boundary state $S _ { k }$ , so the baseline estimates the local value $V ^ { \pi } ( S _ { k } )$ ). In contrast, a flat-trajectory baseline compares returns from trajectories that may have reached different boundary states by turn $k ,$ so it is centered around the global mean $\mu _ { R }$ . The resulting term $V ^ { \pi } ( S _ { i , k } ) - \mu _ { R }$ is an upstream state-contamination term: it can dominate the true turn-level advantage even though it is unrelated to the current turn action. RTPO removes this term by constructing state-matched sibling rollouts and assigning the resulting advantage only to the output tokens of the current turn.

## C.4 Proof of Theorem 3: On-Policy Continuation under Asynchronous Turns

We formalize Theorem 3 from Sec 3.3 before presenting the proof. The purpose is to show that RTPO avoids the off-policy drift induced by stale downstream rollouts. In asynchronous multi-turn training, a trajectory generated under an old policy may later be evaluated under an updated policy, which would require a long-horizon trajectory-level importance-sampling correction. RTPO avoids this issue by regenerating sibling continuations on-policy at each reverse-order turn.

Setup. Let $\theta _ { 0 }$ denote the rollout parameters at the beginning of a training iteration, and let $\theta _ { > k }$ denote the current parameters available at the start of turn k, after downstream turns $k + 1 , \ldots , K - 1$ have already been optimized. A stale-rollout alternative would retain sibling continuations generated under $\pi _ { \theta _ { 0 } }$ and correct them using the trajectory-level importance-sampling weight:

$$
\omega _ { j } ^ { \mathrm { o l d } } = \prod _ { h = k + 1 } ^ { K - 1 } \prod _ { t = 1 } ^ { T _ { j , h } } \frac { \pi _ { \theta _ { > k } } ( a _ { j , h , t } \mid c _ { j , h } , a _ { j , h , < t } ) } { \pi _ { \theta _ { 0 } } ( a _ { j , h , t } \mid c _ { j , h } , a _ { j , h , < t } ) } .\tag{81}
$$

In contrast, RTPO synchronizes the latest parameters $\theta _ { > k }$ to the inference engine at the start of turn $k ,$ forks $G - 1$ sibling rollouts from the shared boundary state $S _ { k } .$ , and samples each sibling rollout $j \in \{ 1 , \dots , G { - } 1 \}$ using $\pi _ { \boldsymbol { \theta } _ { > k } }$ from turn k until termination. We write $u _ { j , k } \equiv l _ { j , k }$ for the sampled turn-k macro-action of sibling j, and $R _ { j }$ for its terminal return. The Monte Carlo turn-level Q-value estimate is

$$
\hat { Q } _ { j , k } = R _ { j } = r _ { j , k } + \gamma _ { H } ^ { \tau _ { j , k } } \hat { F } _ { j , k } ^ { \pi _ { \theta _ { > } k } } ,\tag{82}
$$

where $\hat { F } _ { j , k } ^ { ^ { \pi _ { \theta } } > k }$ is the sampled downstream continuation value generated under the same current policy $\pi _ { \boldsymbol { \theta } _ { > k } }$ . The corresponding sibling advantage is

$$
A _ { j , k } ^ { H } = \hat { Q } _ { j , k } - \hat { V } _ { k } , \qquad \hat { V } _ { k } = { \frac { 1 } { G - 1 } } \sum _ { r = 1 } ^ { G - 1 } \hat { Q } _ { r , k } .\tag{83}
$$

Theorem 3 (On-Policy Continuation under Asynchronous Turns). Consider turn k in the reverse-order training procedure ofRTPO. At the start ofturn k, RTPO synchronizes the current downstream policy parameters $\theta _ { > k }$ to the inference engine,forks G−1 sibling rollouts from the shared boundary state $S _ { k }$ , and continues each sibling rollout j to termination under the same policy $\pi _ { \boldsymbol { \theta } _ { > k } } .$ Then thefollowing two properties hold:

(a) Drift-free on-policy continuation. Each sibling’s terminal return $R _ { j }$ is an unbiased Monte Carlo estimate of the current-policy turn-level Q-value:

$$
\mathbb { E } \Big [ \hat { Q } _ { j , k } \mid S _ { k } , u _ { j , k } \Big ] = \tilde { Q } _ { k } ^ { \pi _ { \theta _ { > { k } } } } ( S _ { k } , u _ { j , k } ) .\tag{84}
$$

Moreover, because the sampling policy and the evaluation policy coincide throughout the sibling continuation, the trajectory-level importance-sampling weight satisfies

$$
\omega _ { j } = \prod _ { h = k + 1 } ^ { K - 1 } \prod _ { t = 1 } ^ { T _ { j , h } } \frac { \pi _ { \theta _ { > k } } ( a _ { j , h , t } \mid c _ { j , h } , a _ { j , h , < t } ) } { \pi _ { \theta _ { > k } } ( a _ { j , h , t } \mid c _ { j , h } , a _ { j , h , < t } ) } \equiv 1 .\tag{85}
$$

(b) Dynamic error reduction in advantage estimation. Under bounded binary or normalized rewards, the conditional variance ofthe Monte Carlo Q-value estimator is controlled by the current-policy success probability. In the binary case, if

$$
p _ { j , k } : = \tilde { Q } _ { k } ^ { \pi _ { \theta } _ { > k } } ( S _ { k } , u _ { j , k } ) \in [ 0 , 1 ] ,\tag{86}
$$

then

$$
\operatorname { V a r } ( \hat { Q } _ { j , k } \mid S _ { k } , u _ { j , k } ) = p _ { j , k } ( 1 - p _ { j , k } ) .\tag{87}
$$

Therefore, when reverse-order training improves downstream policies so that $p _ { j , k }$ moves away from the high-uncertainty region around $1 / 2 ,$ , the Q-value estimation variance decreases. Since $\overset { \smile } { A } _ { j , k } ^ { H }$ is computedfrom the sibling Q-value estimates, this improves the signal-to-noise ratio of the resulting turn-level advantage estimator.

Proof. We prove parts (a) and (b) of Theorem 3 as follows.

Proof of Theorem 3(a): drift-free on-policy continuation. Fix a turn k and a sibling rollout j. At the start of turn k, RTPO synchronizes the current downstream policy parameters $\theta _ { > k }$ to the inference engine. The sibling rollout is forked from the shared boundary state $S _ { k }$ , samples the turn-k macro-action $u _ { j , k } \equiv l _ { j , k }$ , and then continues through turns $k + 1 , \ldots , \overset { \cdot } { K } - 1$ under the same policy $\pi _ { \boldsymbol { \theta } _ { > k } }$

By the definition of the turn-level augmented action value,

$$
\begin{array} { r } { \tilde { Q } _ { k } ^ { \pi _ { \theta > k } } \left( S _ { k } , u _ { j , k } \right) = \mathbb { E } \left[ r _ { j , k } + \gamma _ { H } ^ { \tau _ { j , k } } F _ { k } ^ { \pi _ { \theta > k } } \left( S _ { j , k + 1 } \right) \Big | \ S _ { k } , u _ { j , k } \right] , } \end{array}\tag{88}
$$

where $S _ { j , k + 1 }$ is the next boundary state reached by sibling j after executing $u _ { j , k }$ . Since the sampled downstream continuation $\hat { F } _ { j , k } ^ { { ^ \pi _ { \theta } } _ { > k } }$ is generated by rolling out the same current policy $\pi _ { \boldsymbol { \theta } _ { > k } }$ from $S _ { j , k + 1 }$ to termination, it is an unbiased Monte Carlo draw from the downstream continuation value:

$$
\mathbb { E } \Big [ \hat { F } _ { j , k } ^ { \pi _ { \theta _ { > k } } } \mid S _ { j , k + 1 } \Big ] = F _ { k } ^ { \pi _ { \theta _ { > k } } } ( S _ { j , k + 1 } ) .\tag{89}
$$

Substituting Eq. (89) into Eq. (82) gives

$$
\begin{array} { r l } & { \mathbb { E } \Big [ \hat { Q } _ { j , k } \mid S _ { k } , u _ { j , k } \Big ] = \mathbb { E } \Big [ r _ { j , k } + \gamma _ { H } ^ { \tau _ { j , k } } \hat { F } _ { j , k } ^ { \pi _ { \theta > k } } \Big \mid S _ { k } , u _ { j , k } \Big ] } \\ & { \qquad = \mathbb { E } \Big [ r _ { j , k } + \gamma _ { H } ^ { \tau _ { j , k } } F _ { k } ^ { \pi _ { \theta > k } } ( S _ { j , k + 1 } ) \Big | S _ { k } , u _ { j , k } \Big ] } \\ & { \qquad = \tilde { Q } _ { k } ^ { \pi _ { \theta > k } } ( S _ { k } , u _ { j , k } ) . } \end{array}\tag{90}
$$

Therefore, each sibling terminal return provides an unbiased Monte Carlo estimate of the turn-level Q-value under the current downstream policy.

Next, because the sibling is both sampled and evaluated under the same policy $\pi _ { \boldsymbol { \theta } _ { > k } }$ , the full trajectory-level importance-sampling weight is

$$
\omega _ { j } = \prod _ { h = k + 1 } ^ { K - 1 } \prod _ { t = 1 } ^ { T _ { j , h } } \frac { \pi _ { \theta _ { > k } } ( a _ { j , h , t } \mid c _ { j , h } , a _ { j , h , < t } ) } { \pi _ { \theta _ { > k } } ( a _ { j , h , t } \mid c _ { j , h } , a _ { j , h , < t } ) } = 1 .\tag{91}
$$

Thus, no trajectory-level IS correction is required. This proves part (a) of Theorem 3.

Proof of Theorem 3(b): dynamic error reduction in advantage estimation. We first consider the binary-reward case, where $R _ { j } \in \{ 0 , 1 \}$ . Conditional on $( S _ { k } , u _ { j , k } )$ , define

$$
p _ { j , k } = \mathbb { P } _ { \pi _ { \theta _ { > k } } } ( R _ { j } = 1 \mid S _ { k } , u _ { j , k } ) = \tilde { Q } _ { k } ^ { \pi _ { \theta _ { > k } } } ( S _ { k } , u _ { j , k } ) .\tag{92}
$$

Then $\hat { Q } _ { j , k } = R _ { j }$ is a Bernoulli random variable with mean $p _ { j , k } .$ , and therefore

$$
\operatorname { V a r } ( \hat { Q } _ { j , k } \mid S _ { k } , u _ { j , k } ) = p _ { j , k } ( 1 - p _ { j , k } ) .\tag{93}
$$

The function $f ( p ) = p ( 1 - p )$ is maximized at $p = 1 / 2$ and decreases as p moves toward either 0 or 1. Equivalently,

$$
p ( 1 - p ) = \frac { 1 } { 4 } - \left( p - \frac { 1 } { 2 } \right) ^ { 2 } .\tag{94}
$$

Therefore, if reverse-order training improves the downstream continuation policy so that the induced success probability moves away from the high-uncertainty region around $1 / 2 ,$ , then the conditional variance of the Q-value estimator decreases. Formally,

$$
\left. p _ { j , k } ^ { \mathrm { n e w } } - \frac { 1 } { 2 } \right. > \left. p _ { j , k } ^ { \mathrm { o l d } } - \frac { 1 } { 2 } \right. \quad \Longrightarrow \quad p _ { j , k } ^ { \mathrm { n e w } } ( 1 - p _ { j , k } ^ { \mathrm { n e w } } ) < p _ { j , k } ^ { \mathrm { o l d } } ( 1 - p _ { j , k } ^ { \mathrm { o l d } } ) .\tag{95}
$$

For normalized rewards $R _ { j } \in [ 0 , 1 ]$ , the same statement holds as a bounded-variance control rather than an exact Bernoulli identity. In particular, by the Bhatia–Davis bound for random variables supported on [0, 1],

$$
\operatorname { V a r } ( R _ { j } \mid S _ { k } , u _ { j , k } ) \leq \mathbb { E } [ R _ { j } \mid S _ { k } , u _ { j , k } ] { \big ( } 1 - \mathbb { E } [ R _ { j } \mid S _ { k } , u _ { j , k } ] { \big ) } .\tag{96}
$$

Thus, moving the conditional mean away from the high-uncertainty middle region also reduces the worst-case variance bound for normalized returns.

Finally, RTPO computes the turn-level advantage by subtracting the sibling baseline:

$$
A _ { j , k } ^ { H } = \hat { Q } _ { j , k } - \hat { V } _ { k } , \qquad \hat { V } _ { k } = { \frac { 1 } { G - 1 } } \sum _ { r = 1 } ^ { G - 1 } \hat { Q } _ { r , k } .\tag{97}
$$

Since $A _ { j , k } ^ { H }$ is a centered function of the sibling Q-value estimates, reducing the Monte Carlo noise in $\hat { Q } _ { j , k }$ reduces the noise entering the advantage estimator. Consequently, as reverse-order training improves downstream policies and the on-policy continuation values become more confident, the signal-to-noise ratio of $\mathbf { \dot { \bar { A } } } _ { j , k } ^ { H }$ improves. This proves part (b) of Theorem 3, and completes the proof of Theorem 3. □

Comparison with stale trajectory-level IS correction. The drift-free property above should be contrasted with a stale-rollout alternative that reuses continuations generated under $\pi _ { \theta _ { 0 } }$ and then applies the trajectory-level IS weight $\omega _ { j } ^ { \mathrm { o l d } }$ in Eq. (81). Let

$$
N _ { j } = \sum _ { h = k + 1 } ^ { K - 1 } T _ { j , h }\tag{98}
$$

denote the number of tokens in the continuation from turn $k$ to termination. If the per-token divergence between the current policy and the stale rollout policy is nonzero along this continuation, the variance

of the full product IS weight can grow multiplicatively with $N _ { j }$ . To see this, suppose that for each token position (h, t), the conditional second moment of the token ratio

$$
\rho _ { j , h , t } ^ { \mathrm { o l d } } = \frac { \pi _ { \theta _ { > k } } ( a _ { j , h , t } \mid c _ { j , h } , a _ { j , h , < t } ) } { \pi _ { \theta _ { 0 } } ( a _ { j , h , t } \mid c _ { j , h } , a _ { j , h , < t } ) }\tag{99}
$$

satisfies

$$
\mathbb { E } _ { \pi _ { \theta _ { 0 } } } \left[ \left( \rho _ { j , h , t } ^ { \mathrm { o l d } } \right) ^ { 2 } | c _ { j , h } , a _ { j , h , < t } \right] \geq 1 + \delta \qquad \mathrm { f o r ~ s o m e ~ } \delta > 0 .\tag{100}
$$

Then the second moment of the product weight scales as

$$
\mathbb { E } _ { \pi _ { \theta _ { 0 } } } \left[ \left( \omega _ { j } ^ { \mathrm { o l d } } \right) ^ { 2 } \right] = \mathbb { E } _ { \pi _ { \theta _ { 0 } } } \left[ \prod _ { h = k + 1 } ^ { K - 1 } \prod _ { t = 1 } ^ { T _ { j , h } } \left( \rho _ { j , h , t } ^ { \mathrm { o l d } } \right) ^ { 2 } \right] \gtrsim ( 1 + \delta ) ^ { N _ { j } } ,\tag{101}
$$

up to the usual conditioning on autoregressive histories. Since $\mathbb { E } _ { \pi _ { \theta _ { 0 } } } [ \omega _ { j } ^ { \mathrm { o l d } } ] = 1$ , this implies

$$
\begin{array} { r } { \mathrm { V a r } _ { \pi _ { \theta _ { 0 } } } \big ( \omega _ { j } ^ { \mathrm { o l d } } \big ) = \mathbb { E } _ { \pi _ { \theta _ { 0 } } } \Big [ \big ( \omega _ { j } ^ { \mathrm { o l d } } \big ) ^ { 2 } \Big ] - 1 \gtrsim ( 1 + \delta ) ^ { N _ { j } } - 1 . } \end{array}\tag{102}
$$

This illustrates the long-horizon instability of trajectory-level IS correction: even a small nonzero pertoken policy mismatch can compound into a high-variance product over many tokens. Per-token PPO clipping does not remove this mismatch at the trajectory level, because clipping and multiplication do not commute:

$$
\prod _ { h , t } \mathrm { c l i p } \big ( \rho _ { j , h , t } ^ { \mathrm { o l d } } , 1 - \epsilon , 1 + \epsilon \big ) \neq \mathrm { c l i p } \left( \prod _ { h , t } \rho _ { j , h , t } ^ { \mathrm { o l d } } , 1 - \epsilon , 1 + \epsilon \right) .\tag{103}
$$

Therefore, stale-rollout correction remains fundamentally different from RTPO’s on-policy sibling continuation, where the corresponding weight is $\omega _ { j } \equiv 1$

Interpretation. Theorem 3 shows why RTPO regenerates sibling continuations on-policy instead of reusing stale downstream rollouts. In asynchronous multi-turn training, stale continuations estimate values under outdated downstream policies and would require a long-horizon trajectory-level IS correction. RTPO avoids this by synchronizing $\theta _ { > k }$ before sibling generation and rolling out each sibling to termination under the same current policy. As a result, the full trajectory IS weight is one, the Q-value estimate targets the current downstream policy, and the resulting turn-level advantage avoids policy-drift contamination.

## D Experimental Setup

Base Models. We use Qwen3 models [Yang et al., 2025a] as the backbone for RTPO and all baselines. These models support context lengths of up to 32,768 tokens, which is sufficient to accommodate the multi-turn interaction histories required by long-horizon agentic RL tasks. Since Qwen3 natively supports a thinking inference mode, we enable this mode consistently across all multi-turn agentic RL experiments to ensure that the model generates complete tool-integrated reasoning trajectories during rollout. We use Qwen3-8B as the main backbone because search-based agentic tasks require strong base-model capabilities for multi-turn evidence acquisition, information integration, and tool-augmented reasoning. This choice is supported by our main results in Sec. 4.1, the rollout–training consistency analysis in Sec. 4.2, and the policy-drift analysis in Sec. 4.4. To isolate the effect of turn-level credit assignment in Sec. 4.3, we use the smaller Qwen3-4B model for mathematical reasoning tasks. This is because credit assignment in these tasks depends more directly on the model’s intrinsic reasoning ability than on external knowledge retrieval, making the gains from turn-level credit assignment more discernible.

Baselines. We compare RTPO with two categories of state-of-the-art multi-turn agentic RL methods. The first category consists of trajectory-only policy optimization methods, including GRPO [Shao et al., 2024] and. GRPO uses token-level importance ratios with group-relative advantage estimation. The second category includes turn-level and tree-based policy optimization methods, including ARPO [Dong et al., 2026], Tree-GRPO [Ji et al., 2026], and SeeUPO [Hu et al., 2026]. ARPO performs entropy-driven adaptive branching at uncertain tool-call nodes and estimates advantages separately for shared-prefix and branch-specific tokens. Tree-GRPO represents multi-turn agent interaction as a tree and constructs group-relative advantages at both intra-tree and inter-tree levels by sharing prefixes. SeeUPO treats each turn as an independent agent and performs sequential per-turn updates under a heterogeneous multi-agent learning, with a theoretical guarantee of monotonic improvement.

Datasets. To ensure a fair comparison across methods in multi-turn agentic scenarios, we consider two representative tool-use tasks: mathematical reasoning and knowledge reasoning from search. Both require agents to interact with external tools, perform multi-turn reasoning, and adapt their actions based on intermediate feedback.

For the mathematical reasoning task, we use the MATH dataset [Hendrycks et al., 2021] for training, which covers challenging multi-step reasoning problems spanning algebra, geometry, number theory, probability, and other topics. The task requires agents to decompose complex problems, invoke external Python computation tools when necessary, and integrate intermediate results into the final answer, where successful solutions often depend on iterative calculation, verification, and correction. At the evaluation stage, we test generalization at two difficulty levels: (1) standard mathematical reasoning benchmarks, including GSM8K [Cobbe et al., 2021] and MATH-500 [Lightman et al., 2023]; and (2) competition-level benchmarks, including AMC’23 [Mathematical Association of America, 2023], AIME’24 [Mathematical Association of America, 2024], AIME’25 [Mathematical Association of America, 2025], and OE-Math [He et al., 2024], which feature problems that demand extended chains of tool-augmented reasoning and advanced problem-solving capabilities. Since no training data are available for the competition-level benchmarks, all evaluations are conducted in a zero-shot setting.

For the knowledge reasoning task on web search, we adopt the hard-search training set constructed by ARPO [Dong et al., 2026], consisting of 1,000 high-difficulty search samples drawn from two opensource deep-search data sources: SimpleDeepSearcher [Sun et al., 2025] and WebSailor [Li et al., 2025]. These samples require extensive web retrieval, multi-source evidence integration, long-context reasoning, and frequent tool calls, providing a rigorous testbed for evaluating the model stability and sample efficiency of multi-turn agentic RL. On this task, we evaluate knowledge-intensive multi-hop question answering on HotpotQA [Yang et al., 2018] and 2WikiMultiHopQA [Ho et al., 2020], following the ARPO evaluation protocol [Dong et al., 2026] with LLM-as-Judge scoring based on Qwen2.5-72B-Instruct and report F1 scores. In addition, to examine the effect of onpolicy continuation, we compare default RTPO with its off-policy variant, and conduct additional evaluation on four challenging deep-search benchmarks: GAIA [Mialon et al., 2023], which evaluates general AI-assistant capabilities across three difficulty levels (Lv.1–Lv.3); WebWalkerQA [Wu et al., 2025a], which focuses on interactive web navigation and multi-hop question answering; Humanity’s Last Exam [Phan et al., 2025], which covers expert-level problems in sciences, engineering, and humanities; and xBench [Chen et al., 2025], which evaluates cross-lingual deep-search capability. Since the training data are drawn exclusively from the hard-search source, all four benchmarks serve as held-out evaluation sets. We report Pass@1 with sampling temperature set to 0.6 and top-p set to 0.95.

Configurations. All methods are implemented on top of the VeRL framework [Sheng et al., 2024], with vLLM [Kwon et al., 2023] for rollout generation and FSDP [Zhao et al., 2023] for post-training. All experiments are conducted on 8× NVIDIA A100 GPUs. Unless otherwise specified, the training batch size is 64 and the maximum single-turn response length is 4096 tokens. The maximum interaction horizon is set according to task type: K = 3 turns for mathematical reasoning and K = 6 turns for web search. The learning rate is fixed at $1 \times 1 0 ^ { - 6 }$ with the AdamW [Loshchilov and Hutter, 2017] optimizer and a weight decay of 0.01. For branching-based baselines (ARPO and TreeGRPO), we follow the hyperparameters recommended in their original papers, including an entropy threshold of 0.4 and an initial sampling size of 8. Full hyperparameter configurations are provided in Appendix E.3.

Evaluation Metrics. We evaluate RTPO and all baselines along multiple dimensions. For overall performance, we use Pass@1 accuracy as the primary indicator of final policy quality after optimiza tion (Sec 4.1). We also measure the average number of tool calls to assess behavioral differences across methods under a fixed compute budget. For training stability, rollout–training consistency is evaluated using the log-probability ratio and KL divergence, as described in Sec. 4.2. We further conduct controlled experiments to examine whether RTPO’s turn-level credit assignment (Sec 4.3) and on-policy continuation (Sec 4.4) are consistent with the theoretical analysis. Each experiment is repeated three times to mitigate randomness, and we report the average performance.

## E Implementation Details

## E.1 Pseudocode

The pseudocode for our proposed RTPO is shown in Algorithm 1.

Algorithm 1 Reverse-Turn Policy Optimization (RTPO)   
Require: Query dataset $\mathcal { D } ;$ policy $\pi _ { \boldsymbol { \theta } } ;$ inference engine $\mathcal { E } ;$ sibling count $G ;$ PPO clip $\epsilon ;$ max turns   
K; observation map ψ; PPO epochs $E .$   
Ensure: Updated parameters $\theta .$   
Phase 1: Trunk Rollout   
1: For each query $q \in { \mathcal { D } } ,$ generate a complete trajectory $( S _ { 0 } , u _ { 0 } , f _ { 0 } , \dots , u _ { K - 1 } , f _ { K - 1 } )$ under   
π by interacting with the environment. Record boundary states and environment snapshots   
$\{ S _ { k } , \mathrm { \bar { s n a p } } _ { k } \} _ { k = 0 } ^ { K - 1 }$ , and set conditioning contexts $c _ { k } = \psi ( S _ { k } )$ . The trunk provides anchors only   
and receives no gradient.   
Phase 2: Reverse-Order Training   
2: for $k = K { - } 1 , K { - } 2 , . . . , 0$ do   
3: Synchronize. Push current parameters θ (denoted $\theta _ { > k }$ , reflecting downstream turns   
already optimized) to $\mathcal { E } ;$ freeze $\theta _ { \mathrm { o l d } }  \theta _ { > k }$ as the IS denominator.   
4: Sibling generation (on-policy). For each $q \in \mathcal { D }$ and each sibling $j = 1 , \dotsc , G - 1 \colon$ restore   
environment from sna $\mathrm { u p } _ { k } ;$ sample turn-k response $u _ { j , k } \sim \pi _ { \theta _ { > k } } ( \cdot \mid c _ { k } ) ;$ ; continue under   
$\pi _ { \boldsymbol { \theta } _ { > k } }$ through turns $k { + } \ddot { 1 } , \ldots , \bar { K } { - } 1$ to terminal; receive reward $\dot { R } _ { j } \in \{ 0 , 1 \}$ . Record   
turn-k tokens $\left\{ a _ { j , k , t } \right\} _ { t = 1 } ^ { T _ { j , k } }$ and their log-probs under $\theta _ { \mathrm { o l d } }$   
5: Turn-level advantage. For each $q \in { \mathcal { D } } ,$ compute the sibling baseline and advantage:   
$\hat { V } _ { k } = \frac { 1 } { G { - } 1 } \sum _ { j = 1 } ^ { G - 1 } R _ { j } , \qquad A _ { j , k } ^ { H } = R _ { j } - \hat { V } _ { k } .$   
All siblings share $S _ { k } .$ , so $A _ { j , k } ^ { H }$ is free of upstream state contamination (Theorem 2).   
6: PPO update (turn-k tokens only).   
7: for epoch $= 1 , \ldots , E$ do   
8: for each mini-batch from sibling turn-k tokens do   
9: Compute IS ratios $\rho _ { j , k , t } = \bar { \pi } _ { \boldsymbol { \theta } } ( a _ { j , k , t } \mid c _ { k } , a _ { j , k , < t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { j , k , t } \mid c _ { k } , a _ { j , k , < t } ) |$   
10: Update θ via the clipped objective:   
$J _ { k } ( \theta ) = \frac { 1 } { G - 1 } \sum _ { j = 1 } ^ { G - 1 } \frac { 1 } { T _ { j , k } } \sum _ { t = 1 } ^ { T _ { j , k } } \operatorname* { m i n } \Bigl ( \rho _ { j , k , t } A _ { j , k } ^ { H } , ~ \mathrm { c l i p } \bigl ( \rho _ { j , k , t } , 1 - \epsilon , 1 + \epsilon \bigr ) A _ { j , k } ^ { H } \Bigr )$   
11: end for   
12: end for   
▷ Turn k complete; $\pi _ { \theta , k } f r o z e n ;$ next turn inherits updated θ.   
13: end for

## E.2 Source Code

We provide the source code of RTPO in the Supplementary Material. The repository includes complete installation instructions and versioned dependency requirements. Our implementation is built upon VERL [Sheng et al., 2024], whose core additions are: (1) summarizing the output of each round, so that the next round can rollout without having access to the complete history, (2) on-policy tree rollouts, and (3) computation of per-turn advantage and reverse updates. We also provide out-of-the-box training and evaluation scripts to reproduce our main results on mathematical and knowledge reasoning tasks.

## E.3 Training Details

We provide the detailed hyperparameters for all experiments in Table 4. Unless otherwise noted, the maximum prompt/response lengths are kept the same across all experiments. For computational fairness, the global sampling budget is set to 80k rollouts for mathematical reasoning (5k prompts × 16 rollouts per prompt) and 16k for knowledge reasoning (1k prompts × 16 rollouts per prompt). For the $\mathrm { Q w e n } 3$ series, we use a sampling temperature of 0.9, the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 6 }$ , weight decay 0.01, and an initial KL coefficient of 0.02. The log-probability clipping range is set to $[ - \bar { 1 } 0 , 1 0 ]$ . The training batch size and the rollout batch size are both set to 64. The maximum prompt length is 8192 tokens and the maximum single-turn response length is 4096 tokens for both tasks. The maximum interaction horizon is set to $K = 3$ turns for mathematical reasoning and $K = 6$ turns for web search. All methods are allocated a uniform budget of 16 rollouts per prompt (see Table 4).

<table><tr><td>Method</td><td>Parameters</td><td>Total Rollouts</td><td>Structure</td></tr><tr><td>GRPO</td><td> $\mathtt { n \_ a g e n t } = 1 6$ </td><td>16 independent chains</td><td>Chain</td></tr><tr><td>SeeUPO</td><td> $\mathtt { r o l l o u t \_ n = 1 6 }$ </td><td>16 chains + turn-level update</td><td>Chain</td></tr><tr><td>ARPO</td><td> $N { = } 8 , \ M { = } 1 6$ </td><td>8 initial + entropy branch to 16</td><td>Entropy tree</td></tr><tr><td>TreeGRPO</td><td> $M { = } 4 , \ N { = } 3 , \ L { = } 1$ </td><td> $4 \mathrm { t r e e s } \times 4 \mathrm { l e a v e s } = 1 6$ </td><td>Random tree</td></tr><tr><td>RTPO</td><td>sibling rollouts</td><td>Global Max budget</td><td>State-matched tree</td></tr></table>

Table 4: Hyperparameter settings for all experiments.

For GRPO, we sample 16 independent chains per prompt. For SeeUPO, we generate 16 independent chains with sequential turn-level updates in reverse execution order. For ARPO, we set the initial sampling size to $N = 8$ and the global rollout budget to $M = 1 6$ , with entropy weight $\beta = 0 . 2$ base probability $\alpha = 0 . 5$ , and branching threshold $\tau = 0 . 5 ;$ rollouts that do not trigger entropydriven branching are supplemented with independent chains until the budget of 16 is reached. For TreeGRPO, we set the number of initial trees to $M = 4$ , the number of expansion nodes per iteration to $N = 3 .$ , and the number of expansion iterations to $L = 1$ , yielding $M \times \mathsf { \bar { ( } } L \times N + 1 ) = 1 6$ rollouts per prompt via random node expansion. For RTPO, we first generate multiple trunk trajectories per prompt to establish turn-boundary states: 4 trunks for mathematical reasoning (K=3) and 2 trunks for web search $\left( K { = } 6 \right)$ , since in practice most trunks do not reach the maximum interaction horizon and terminate early with fewer tool calls. At each realized turn boundary $S _ { k } , G { - } 1 = 2$ sibling rollouts are forked and continued to termination.

To ensure computational fairness with baselines, RTPO enforces the same global sampling budget as all other methods (80k for mathematical reasoning, 16k for knowledge reasoning), so that the total number of generated rollouts across all prompts does not exceed that of any baseline. Per-prompt rollout counts may vary depending on early termination, but the global budget constraint guarantees that RTPO consumes no more rollout compute than the 16-chain baselines in aggregate. All methods use a discount factor of $\gamma = 1$

All experiments are conducted on a single node equipped with 8× NVIDIA A100 80GB GPUs, 256 CPU cores, and 256GB of system memory, running Ubuntu 22.04 LTS. The math task requires approximately 15 hours of training, whereas the knowledge task requires approximately 5 hours.

## E.4 Qwen3 Chat Template

At each interaction turn, Qwen3 receives the current context and generates three components: (A) an internal reasoning trace enclosed by <think>...</think>, (B) textual commentary after </think>, and (C) a tool-call instruction, e.g., <tool\_ $\mathtt { c a l l } { } ? \vdash \dots \vdash \mathtt { ( t o o l }$ \_call>. The tool call is executed by the external environment, which returns the result as an observation. Before constructing the nextturn input, the chat-template filter removes the <think>...</think> segment, while retaining the textual commentary, the tool-call instruction, and the appended tool-execution result. Thus, the input to the second turn consists of: (1) the original user request, (2) the previous textual commentary, (3) the previous tool call, and (4) the tool result. This filtered context enables the model to condition on the action history and environment feedback without carrying redundant internal reasoning traces, keeping the context length manageable while preserving decision-relevant information. The process is illustrated in Figure 4.

![](images/9c996aedcb6e4ac5440adf3deeb59f7cfb1e5236d0847f4501384e71910c5648.jpg)  
Figure 4: Examples of Qwen3 chat template.

## F Additional Results and Insights

## F.1 Discussion of Main Results

For the main experiments, we use Qwen3-8B as the base model and evaluate on two categories of multi-turn tool-use tasks: mathematical reasoning and web search. Qwen3 models automatically produce chain-of-thought content wrapped in <think>...</think> tags during inference. In our setup, each turn’s input is constructed using the standard Qwen3 chat template, which post-processes prior assistant content by stripping out the <think>...</think> segments so that subsequent turns observe only the final answer content following $< / \mathrm { t h i n k } >$ This design keeps the per-turn input context concise and aligns naturally with the turn-boundary MDP formulation underlying RTPO: each turn-level policy is conditioned on the visible context $c _ { k }$ rather than on the full raw interaction history. In addition, we report the complete accuracy results with mean and standard deviation, extending Table 1 to Table 5.

<table><tr><td rowspan="2">Method</td><td colspan="6">M: Mathematical Reasoning</td><td colspan="2">K: Knowledge Reasoning</td></tr><tr><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH500</td><td>GSM8K</td><td>OE-Math</td><td>HotpotQA</td><td>2Wiki</td></tr><tr><td>Vanilla</td><td> $3 . 3 3 { \scriptstyle \pm 0 . 0 0 \% }$ </td><td> $1 3 . 3 3 { \scriptstyle \pm 3 . 3 3 \% }$ </td><td> $1 2 . 5 0 { \scriptstyle \pm 2 . 5 0 \% }$ </td><td> $5 1 . 0 0 _ { \pm 0 . 4 0 \% }$ </td><td>76.95±0.15%</td><td> $2 4 . 3 3 { \scriptstyle \pm 0 . 6 8 \% }$ </td><td> $\underline { { 5 4 . 5 2 } } _ { \pm 0 . 7 2 \% }$ </td><td> $5 9 . 8 3 { \scriptstyle \pm 0 . 6 1 \% }$ </td></tr><tr><td>GRPO</td><td> $1 0 . 0 0 _ { \pm 3 . 3 3 \% }$ </td><td>20.00±3.33%</td><td> $5 7 . 5 0 _ { \pm 2 . 5 0 \% }$ </td><td> $8 2 . 3 3 { \scriptstyle \pm 0 . 5 0 \% }$ </td><td>93.86±0.23%</td><td> $5 0 . 7 4 { \scriptstyle \pm 1 . 1 2 \% }$ </td><td> $\overline { { 5 3 . 3 9 } } _ { \pm 0 . 9 3 \% } ^ { - \pm \cdots }$ </td><td> $6 1 . 8 5 _ { \pm 0 . 8 4 \% }$ </td></tr><tr><td>SeeUPO</td><td> $\underline { { 2 0 . 0 0 } } _ { \pm 3 . 3 3 \% } ^ { - }$ </td><td>23.33±3.33%</td><td> $\underline { { 6 7 . 5 0 } } \pm 2 . 5 0 \%$ </td><td> $\underline { { 8 4 . 2 0 } } \underline { { \pm 0 . 3 5 \% } }$ </td><td> $\underline { { 9 4 . 4 7 } } _ { \pm 0 . 1 3 \% }$ </td><td> $5 3 . 8 6 _ { \pm 0 . 4 5 \% }$ </td><td> $5 4 . 2 7 _ { \pm 0 . 5 5 \% } ^ { - }$ </td><td> $6 3 . 7 8 _ { \pm 0 . 4 9 \% } ^ { - }$ </td></tr><tr><td>RTPO</td><td> $\overline { { { \bf 3 3 . 3 3 } } } _ { \pm 0 . 0 0 \% } ^ { - }$ </td><td> $\mathbf { 2 6 . 6 7 _ { \pm 0 . 0 0 \% } }$ </td><td> ${ \bf 6 7 . 5 0 } _ { \pm 0 . 0 0 \% } ^ { - }$ </td><td> $\mathbf { 8 6 . 2 0 } _ { \pm 0 . 2 0 \% } ^ { - }$ </td><td> $\mathbf { 9 4 . 8 4 } _ { \pm 0 . 0 8 \% } ^ { - }$ </td><td> $\mathbf { 5 5 . 4 9 } _ { \pm 0 . 3 0 \% } ^ { - }$ </td><td> $\mathbf { 6 4 . 8 9 _ { \pm 0 . 3 4 \% } }$ </td><td> ${ \bf 6 4 . 3 5 _ { \pm 0 . 2 7 \% } }$ </td></tr></table>

Table 5: Accuracy comparison across eight benchmarks. Values are reported as mean $\pm \mathrm { s t d }$ over three evaluation runs. Acc denotes Pass@1 (%) for mathematical tasks and best-span F1 (%) for knowledge reasoning tasks. Standard deviations are reported in percentage points.

Performance and tool-use patterns on mathematical reasoning tasks. On mathematical tasks, tool-call frequency exhibits a trend that runs almost opposite to accuracy: Vanilla makes 17 Python calls on AIME24 yet solves only one problem, whereas RTPO solves ten with just four calls. This pattern reflects an intrinsic property of mathematical reasoning: the knowledge and derivations required to solve a math problem are primarily internalized in the model’s parameters, while the external Python tool serves as an auxiliary aid for verification and numerical computation rather than as a source of new information. Vanilla’s frequent tool use, therefore, largely reflects an inefficient strategy of repeated verification used to mask reasoning uncertainty. After RL training, the model gains stronger control over its own reasoning process, and tool calls collapse from redundant repeated verification into precise invocations at critical computation steps; consequently, the number of calls decreases while accuracy increases. RTPO’s advantage becomes particularly pronounced on the hardest math benchmarks: it reaches 33.33% Pass@1 on AIME24, surpassing SeeUPO (20.00%) and GRPO (10.00%) by 13.3 and 23.3 absolute points, respectively, and maintains a clear lead on AIME25 and OE-Math. RTPO simultaneously achieves the fewest tool calls and the highest accuracy on these hard problems, indicating that it learns the most efficient tool-use strategy. We further note that the gap between methods on mathematical tasks remains relatively small overall, since math problems typically require only a few interaction turns, and the context discrepancy between rollout and training stays at a manageable scale under short horizons; the robustness of standard training dynamics alone is sufficient for baselines to reach near-optimal performance in this regime. We provide an example to support our discussion, as shown in Figure 5.

![](images/052926ba89c0b1caf13926e547ec60a5b9fc4e01b48e50a13529fb524282831c.jpg)  
Figure 5: An example of different policy performance on a mathematical-reasoning task through Python tools.

Performance and tool-use patterns on knowledge reasoning tasks. On knowledge-intensive question answering, RTPO reaches 64.89 F1 on HotpotQA, exceeding SeeUPO (54.27) and GRPO (53.39) by 10.6 and 11.5 points, respectively, and maintains a clear lead on 2Wiki. In contrast to mathematical tasks, the tool-use pattern on knowledge reasoning exhibits a positive correlation between call count and accuracy: RTPO issues 613 and 867 search calls on HotpotQA and 2Wiki, far exceeding the roughly 200 calls observed for the other methods. This contrast admits a natural explanation grounded in task structure: the factual information required to answer such questions does not reside in the model’s parametric knowledge and must be acquired through external retrieval, while multi-hop questions further demand cross-turn integration of multiple pieces of evidence. The lower retrieval counts of GRPO and SeeUPO indicate that they fail to learn to issue sustained follow-up queries and to expand retrieval across multiple turns. The underlying cause is that such hard tasks require longer interaction horizons and more frequent reasoning revision, so the rollout–training context mismatch is amplified as turns accumulate, and baseline methods must spend a larger share of their optimization budget compensating for this drift. RTPO eliminates this source of bias structurally, allowing the entire optimization budget to act directly on the task objective, which yields more stable improvements on long-horizon hard tasks. We provide an example to support our discussion, as shown in Figure 6.

![](images/2c547f09a4acf9e07c391a1a1ef6743162230d8d586005cf38b67bf080851bf2.jpg)  
Figure 6: An example of different policy performance on a knowledge-reasoning task through web search tools.

Taking both task categories together, RTPO learns to perform mathematical reasoning withfewer calls but more reliable internal reasoning, while performing knowledge reasoning with denser calls and more thorough external information integration. This bidirectional adaptation of tool-use behavior indicates that the advantage of RTPO does not stem from simply encouraging or suppressing tool calls, but rather from its turn-level optimization objective, which is able to learn a tool-use strategy matched to the underlying nature of each task.

## F.2 Insights from Rollout–Training Consistency Analysis

Follow-up discussion from Sec 4.2, a noteworthy observation is that, despite using different conditioning contexts at the rollout and training stages, the baselines’ ratios still exhibit a slow drift toward 1. We find that this phenomenon is closely related to the mechanism revealed by DAgger [Ross et al., 2011]: when the training data is continually drawn from the distribution induced by the policy itself, the model gradually adapts to the distribution it actually operates on. In our setting, trajectories generated by summary-rollouts are reinforced under full-history-training, and parameter sharing causes the behaviors under the two conditionings to converge indirectly; the next rollout therefore falls closer to the region considered reasonable under the training context. SCoRe [Kumar et al., 2025] observes the same bootstrapped alignment process in multi-turn online RL. In addition, some works point out that the summary can serve as a learnable sufficient statistic: if the summary preserves the decision-relevant information, the optimal policies under the two contexts can converge in an information-theoretic sense [Allen et al., 2021]; recent work on multi-turn RL further shows that the summary context can evolve into a learnable compact decision state under end-to-end optimization [Lu et al., 2025, Yu et al., 2026a].

However, this empirical alignment differs from the structural consistency of RTPO in three fundamental ways:

(1) Alignment is incomplete and strongly depends on task complexity. GRPO recovers only to 0.97 on math after 30 steps and only to 0.83 on search after 14 steps; the longer the horizon and the faster the context accumulates, the harder the alignment becomes, which is reflected in the knowledge reasoning task in Table 1.

(2)The alignment process is accompanied by oscillation. SeeUPO produces a 1.02 spike at step 9 on math and exhibits sustained small fluctuations, because the bootstrapped loop is driven by the advantage signal, and the variance of advantage estimates propagates directly into step-to-step jitter of the ratio. RTPO’s ratio is structurally guaranteed and is only affected by differences between the training and inference engines.

(3) Empirical alignment consumes additional optimization budget. Bootstrapped alignment essentially allocates part of the policy’s capacity to an implicit objective—pulling the behavior under the summary toward the optimal behavior under the full history. RTPO removes this hidden cost, so that the optimization budget can act directly on the task objective. This is consistent with the lower search-call counts of baselines on long-horizon knowledge reasoning observed in Table 1: part of their training dynamics is diverted to patching the mismatch.

Overall, the rollout–training ratio reveals not that baselines necessarily fail, but that baselines must rely on training dynamics to compensate for a gap that RTPO does not have by construction, and this compensatory mechanism becomes substantially less effective on long-horizon tasks.

## F.3 Additional Findings for Hard-Search Scenarios

To quantify the independent contribution of on-policy continuation (Sec. 3.3), we use Qwen3-8B as the base model and compare standard RTPO with its off-policy variant. Both share the same reverse-order turn-level training and state-matched sibling structure; the only difference lies in how sibling downstream continuations are obtained. RTPO (on-policy) synchronizes the latest parameters $\theta _ { > k }$ to the inference engine at the start of turn k and regenerates sibling continuations under $\pi _ { \boldsymbol { \theta } _ { > k } }$ until termination, so the trajectory-level IS weight is identically one, and the Q-value estimate is

$$
\begin{array} { r } { \hat { Q } _ { j , k } ^ { \mathrm { o n } } = r _ { j , k } + \gamma ^ { \tau _ { j , k } } \hat { F } _ { j , k } ^ { \pi _ { \theta _ { > k } } } . } \end{array}\tag{104}
$$

<table><tr><td>Benchmark</td><td>QS</td><td>Off Hit</td><td>Off Rate</td><td>On Hit</td><td>On Rate</td><td>∆</td></tr><tr><td>GAIA</td><td>103</td><td>24</td><td>23.30%</td><td>30</td><td>29.13%</td><td>+5.83%</td></tr><tr><td>L1</td><td>39</td><td>7</td><td>17.95%</td><td>10</td><td>25.64%</td><td>+7.69%</td></tr><tr><td>L2</td><td>52</td><td>16</td><td>30.77%</td><td>18</td><td>34.62%</td><td>+3.85%</td></tr><tr><td>L3</td><td>12</td><td>1</td><td>8.33%</td><td>2</td><td>16.67%</td><td>+8.33%</td></tr><tr><td>WebWalker</td><td>200</td><td>12</td><td>6.00%</td><td>19</td><td>9.50%</td><td>+3.50%</td></tr><tr><td>XBench</td><td>100</td><td>8</td><td>8.00%</td><td>15</td><td>15.00%</td><td>+7.00%</td></tr><tr><td>HLE</td><td>2096</td><td>261</td><td>12.45%</td><td>254</td><td>12.12%</td><td>-0.33%</td></tr><tr><td>Biology/Medicine</td><td>228</td><td>37</td><td>16.23%</td><td>38</td><td>16.67%</td><td>+0.44%</td></tr><tr><td>Chemistry</td><td>81</td><td>6</td><td>7.41%</td><td>6</td><td>7.41%</td><td>+0.00%</td></tr><tr><td>Computer Science/AI</td><td>220</td><td>20</td><td>9.09%</td><td>24</td><td>10.91%</td><td>+1.82%</td></tr><tr><td>Engineering</td><td>71</td><td>5</td><td>7.04%</td><td>8</td><td>11.27%</td><td>+4.23%</td></tr><tr><td>Humanities/Social Science</td><td>192</td><td>32</td><td>16.67%</td><td>23</td><td>11.98%</td><td>-4.69%</td></tr><tr><td>Math</td><td>947</td><td>114</td><td>12.04%</td><td>112</td><td>11.83%</td><td>-0.21%</td></tr><tr><td>Other</td><td>158</td><td>29</td><td>18.35%</td><td>29</td><td>18.35%</td><td>+0.00%</td></tr><tr><td>Physics</td><td>199</td><td>18</td><td>9.05%</td><td>14</td><td>7.04%</td><td>-2.01%</td></tr></table>

Table 6: Rechecked output-hit comparison between standard RTPO (on-policy) and its off-policy variant. Values are rounded to the nearest integer for better readability. Output-hit marks a sample correct if the output contains the gold answer or an alias.

RTPO with off-policy variant instead reuses the downstream continuations already generated under $\pi _ { \theta _ { 0 } }$ during the initial rollout stage and corrects for the staleness via a clamped trajectory-level importance weight:

$$
\hat { Q } _ { j , k } ^ { \mathrm { o f f } } = r _ { j , k } + \gamma ^ { \tau _ { j , k } } \bar { \omega } _ { j , k } \hat { F } _ { j , k } ^ { \pi _ { \theta _ { 0 } } } , \qquad \bar { \omega } _ { j , k } = \mathrm { c l a m p } \Bigg ( \prod _ { h = k + 1 } ^ { K - 1 } \prod _ { t = 1 } ^ { T _ { j , h } } \frac { \pi _ { \theta _ { > k } } ( a _ { j , h , t } \mid s _ { j , h , t } ) } { \pi _ { \theta _ { 0 } } ( a _ { j , h , t } \mid s _ { j , h , t } ) } , ~ \rho _ { \operatorname* { m i n } } , ~ \rho _ { \operatorname* { m a x } } \Bigg ) .\tag{105}
$$

We compare the on-policy RTPO and its off-policy variant on four knowledge-reasoning deep-search benchmarks, as shown in Table 6. We use output-hit accuracy, which marks a sample correct if the full output contains the gold answer or an alias. The benchmarks include GAIA with three difficulty levels, WebWalkerQA, XBench, and HLE with eight subject subsets.

## F.4 Limitations and Future Work

Although RTPO provides stronger theoretical guarantees and empirical performance than existing flat-trajectory methods, its algorithmic design has several limitations that motivate future work.

Dependence on trunk quality. RTPO uses the boundary states $S _ { k }$ of a trunk trajectory as the forking points for sibling branches. As a result, the training signal at each reverse phase is anchored to the state sequence visited by the trunk. If the trunk makes a poor decision at an early turn, such as $k = 0 \mathrm { o r } k = 1$ , later boundary states may lie in low-value regions where even strong turn-level actions fail to obtain positive terminal rewards. In this case, the turn-level advantages $A _ { j , k } ^ { H }$ may degenerate into near-zero signals, making the corresponding gradient update ineffective. In contrast, beam-style search methods, such as beam search or best-of-N, maintain multiple candidate prefixes and can discard low-quality paths earlier. RTPO currently relies on a single trunk anchor and does not explicitly incorporate trunk-level diversity or post-hoc trunk selection. A natural extension is multi-trunk sampling or retroactive trunk selection, where multiple complete trunks are generated during rollout and a high-reward trunk is selected as the anchor to improve boundary-state coverage. This would increase rollout cost, but would not change the reverse-order training formulation.

Overhead of reverse multi-turn training. RTPO decomposes a K-turn episode into K sequential turn-level optimization phases. Each phase requires sibling generation, environment restoration, on policy continuation to termination, and a PPO-style update. Compared with flat-trajectory methods, which perform a single optimization pass over the full trajectory, RTPO incurs additional cost that scales with the number of turns K and the number of sibling rollouts. Moreover, each reverse phase requires synchronizing the latest policy parameters to the inference engine before generating on-policy continuations, introducing additional inference-training latency. In our VeRL-based implementation, this overhead is partially mitigated by batched parallel sibling generation and asynchronous engine scheduling, but it cannot be fully removed. Designing more efficient sibling generation and update schedules is therefore an important direction for future work.

Under-utilization of training tokens. RTPO assigns the turn-level advantage in phase k only to the sibling’s turn-k output tokens. Prefix tokens in $c _ { k }$ and downstream continuation tokens from turns $k { + } 1 , \ldots , K { - } 1$ receive no gradient. In addition, the trunk trajectory is used only as an anchor and does not directly contribute to gradient updates. Thus, although sibling continuations are necessary for estimating terminal returns, their downstream tokens are discarded during policy optimization. For example, in a $K = 5$ episode with an average of 200 tokens per turn, the sibling continuation at phase $k = 2$ may generate around 600 downstream tokens, while only the 200 turn-k tokens are used for the PPO update. This reduced token utilization is the cost of causal turn-level credit assignment: by withholding gradients from non-turn-k tokens, RTPO avoids assigning credit to actions that are not causally responsible for the turn-k comparison, as stated in Theorem 2(c). Future work may explore auxiliary objectives, such as language-modeling losses or self-play rewards on downstream tokens, to improve token efficiency while preserving the causal consistency of the turn-level advantage.

## F.5 Broader Impacts

The potential positive impact of RTPO is that more stable agentic RL training can reduce failed tool-use trajectories, improve sample efficiency, and support more reliable deployment of LLM agents in research, education, software engineering, and decision-support settings. RTPO may also make multi-turn RL training easier to analyze by separating turn-level decisions from full-trajectory outcomes. By enabling turn-level monitoring, RTPO can further improve our understanding of how agentic workflows learn to plan, search, and use tools over multiple turns. At the same time, stronger turn-refined agentic workflows may increase the capability of LLM agents to act autonomously across long-horizon tasks. If deployed without appropriate safeguards, such systems could produce incorrect outputs with high confidence, misuse external tools, or amplify harmful automation. Therefore, practical deployment should include safety constraints, tool-use monitoring, privacy-preserving data handling, and human oversight, especially in high-stakes domains.

## G Supplementary Theoretical Clarifications, Implementation Details, and Extended Experiments

## G.1 Stateful Tool-Agent Evaluation on $\tau ^ { 3 } { \bf - A }$ irline

Experimental setting. We evaluate Qwen3-1.7B on a fixed set of 30 training tasks and the 20 held-out test tasks provided by the $\tau ^ { 3 } .$ -Airline environment. The agent must query and modify an airline database through multi-turn tool interactions, and earlier actions change the state observed in later turns.

Table 7: Online training success rate on $\tau ^ { 3 } .$ -Airline.
<table><tr><td>Method</td><td>Step 10</td><td>Step 20</td><td>Step 30</td><td>Step 40</td><td>Step 50</td></tr><tr><td>GRPO</td><td>23.44%</td><td>25.00%</td><td>31.26%</td><td>29.69%</td><td>31.26%</td></tr><tr><td>SeeUPO</td><td>31.23%</td><td>9.38%</td><td>14.83%</td><td>0.78%</td><td>0.78%</td></tr><tr><td>Tree-GRPO</td><td>18.40%</td><td>14.95%</td><td>27.48%</td><td>24.53%</td><td>14.88%</td></tr><tr><td>REFUEL</td><td>25.00%</td><td>31.25%</td><td>31.25%</td><td>28.12%</td><td>21.88%</td></tr><tr><td>RTPO  $\left( G = 3 \right)$ </td><td>15.63%</td><td>37.50%</td><td>50.00%</td><td>56.25%</td><td>40.63%</td></tr></table>

RTPO obtains the highest online success rate from Step 20 onward and peaks at 56.25% at Step 40. The reduction to 40.63% at Step 50 indicates that continued optimization can produce late-stage degradation when the training set is small and the reward is sparse.

Table 8: Held-out evaluation on the 20 $\tau ^ { 3 } .$ -Airline test tasks. $\mathrm { P a s s ^ { 4 } }$ follows the definition used in the main paper.
<table><tr><td>Method</td><td>Pass@1</td><td>Pass@4</td><td> $\mathrm { P a s s } ^ { 4 }$ </td><td>Normal termination</td><td>Generation truncation</td><td> $\operatorname { A v g } .$  response tokens</td></tr><tr><td>SeeUPO</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>98.75%</td><td>2,023.7</td></tr><tr><td>GRPO</td><td>11.25%</td><td>25.00%</td><td>0.00%</td><td>20.00%</td><td>60.00%</td><td>2,469.7</td></tr><tr><td>Tree-GRPO</td><td>16.25%</td><td>25.00%</td><td>10.00%</td><td>30.00%</td><td>47.50%</td><td>2,538.2</td></tr><tr><td>REFUEL</td><td>11.25%</td><td>20.00%</td><td>0.00%</td><td>18.75%</td><td>57.50%</td><td>2,621.2</td></tr><tr><td>RTPO (G = 3)</td><td>17.50%</td><td>30.00%</td><td>10.00%</td><td>45.00%</td><td>42.50%</td><td>1,688.8</td></tr></table>

RTPO achieves the highest Pass@1 and Pass@4 and the highest normal environment-termination rate. Relative to Tree-GRPO, it reduces the generation-truncation rate from 47.50% to 42.50% and the average response length from 2,538.2 to 1,688.8 tokens. Relative to REFUEL, RTPO improves Pass@1 from 11.25% to 17.50%, improves Pass@4 from 20.00% to 30.00%, and reduces average response length by approximately 35.6%.

Table 9: Tool-execution quality on the held-out $\tau ^ { 3 } .$ -Airline tasks.
<table><tr><td>Method</td><td>Attempted calls</td><td>Executed successfully</td><td>Tool-error calls</td><td>Execution success</td><td>Response tokens/success</td></tr><tr><td>GRPO</td><td>118</td><td>68</td><td>50</td><td>57.63%</td><td>21,952.9</td></tr><tr><td>SeeUPO</td><td>1</td><td>0</td><td>1</td><td>0.00%</td><td>N/A</td></tr><tr><td>Tree-GRPO</td><td>1</td><td>0</td><td>1</td><td>0.00%</td><td>15,619.5</td></tr><tr><td>RTPO</td><td>83</td><td>74</td><td>9</td><td>89.16%</td><td>9,650.5</td></tr></table>

RTPO attempts fewer tool calls than GRPO but completes more calls without execution errors. Tool-error calls decrease from 50 to 9, and the execution success rate increases from 57.63% to 89.16%. RTPO also requires approximately 9,650.5 response tokens per successful trajectory, which is 38.2% lower than Tree-GRPO and 56.0% lower than GRPO.

## G.2 Sensitivity to the Sibling Group Size

The group-size hyperparameter is $G ;$ at each boundary, RTPO forks G − 1 sibling continuations from the same turn-boundary state $S _ { k }$ . The turn-level estimator is

$$
\widehat { A } _ { j , k } ^ { H } = R _ { j } - \frac { 1 } { G - 1 } \sum _ { r = 1 } ^ { G - 1 } R _ { r } .\tag{106}
$$

Theorem 2(a) gives

$$
\mathbb { E } \left[ \widehat { A } _ { j , k } ^ { H } \ | \ S _ { k } , u _ { j , k } \right] = \frac { G - 2 } { G - 1 } A ^ { \pi } ( S _ { k } , u _ { j , k } ) .\tag{107}
$$

Thus, the finite-group bias is of order $O ( 1 / G )$ , and the multiplicative coefficient approaches one as G increases. Averaging more sibling returns also reduces sampling noise in the Monte Carlo baseline. Because all continuations begin from the same boundary state, this comparison does not reintroduce upstream-state contamination.

Table 10: Effect of sibling group size on signal density and held-out performance in $\tau ^ { 3 } { \cdot } \mathsf { A }$ irline.
<table><tr><td>G</td><td>Theoretical coefficient</td><td>Pass@1</td><td>Pass@4</td><td>Zero-advantage rate</td><td>Tokens/success</td></tr><tr><td>3</td><td>1/2</td><td>17.50%</td><td>30.00%</td><td>89.61%</td><td>9,650.5</td></tr><tr><td>4</td><td>2/3</td><td>20.00%</td><td>30.00%</td><td>82.66%</td><td>7,477.9</td></tr></table>

Increasing G from 3 to 4 reduces the zero-advantage rate from 89.61% to 82.66%, improves Pass@1 from 17.50% to 20.00%, and reduces inference tokens per successful trajectory from 9,650.5 to 7,477.9. The larger group therefore provides denser relative learning signals, but requires more offline exploration.

Table 11: Performance–cost trade-off for sibling group size.
<table><tr><td>G</td><td>GPU-hours</td><td>Training-generation tokens</td><td>Pass@1</td><td>Pass@4</td><td>Avg. tool calls</td><td>Avg. response tokens</td></tr><tr><td>3</td><td>21.48</td><td>846,207</td><td>17.50%</td><td>30.00%</td><td>1.038</td><td>1,688.8</td></tr><tr><td>4</td><td>28.06</td><td>2,139,039</td><td>20.00%</td><td>30.00%</td><td>0.662</td><td>1,495.6</td></tr></table>

Moving from $G = 3$ to $G = 4$ increases GPU-hours by approximately 30.6% and training-generation tokens by approximately 152.8%, while improving Pass@1 by 2.5 percentage points and leaving Pass@4 unchanged. The resulting policy uses approximately 36.2% fewer tool calls and 11.4% fewer response tokens at inference time. Consequently, $G = 3$ provides the stronger default cost– performance trade-off, whereas G = 4 is useful when Pass@1 and concise inference behavior are prioritized.

For $G = 2 .$ , the coefficient in Eq. (107) is zero, so the conditional expectation of the relative advantage degenerates to zero. Therefore, G = 3, corresponding to two sibling continuations, is the minimum viable configuration that preserves an informative relative signal.

## G.3 Trunk Quality and Failure Dynamics

Table 12: Aggregate trunk outcomes in the $\tau ^ { 3 } .$ -Airline training run.
<table><tr><td>Trunk metric</td><td>Percentage of all trunks</td></tr><tr><td>Final task success</td><td>40.00%</td></tr><tr><td>Failed within Turn 1</td><td>6.88%</td></tr><tr><td>Failed within the first 2 turns</td><td>22.50%</td></tr><tr><td>Failed within the first 3 turns</td><td>28.75%</td></tr><tr><td>Failed within the first 5 turns</td><td>39.38%</td></tr><tr><td>No terminal environment signal</td><td>10.63%</td></tr><tr><td>Average trunk length</td><td>5.39 turns</td></tr></table>

Only 6.88% of trunks fail within the first turn, indicating that catastrophic early failure is not the dominant failure mode. Most errors occur in the middle or later stages. Moreover, an observed early failure under finite sampling is not equivalent to a strict dead state, because finite continuations cannot establish that every possible future policy is unable to recover.

The fraction of trajectories without a terminal environment signal decreases from 37.50% at Step 10 to 0% at Step 50, while the average failed-trajectory length decreases from 13.30 to 2.84 turns. RTPO therefore substantially reduces trajectories that stall for a long time or fail to complete the interaction. Later failures increasingly take the form of fast but incorrect termination rather than persistent generation until truncation. The drop in success after Step 40 may reflect over-optimization on a limited task set under sparse binary rewards.

Table 13: Evolution of failure modes during RTPO training.
<table><tr><td>Metric</td><td>Step 10</td><td>Step 20</td><td>Step 30</td><td>Step 40</td><td>Step 50</td></tr><tr><td>Success rate</td><td>15.63%</td><td>37.50%</td><td>50.00%</td><td>56.25%</td><td>40.63%</td></tr><tr><td>Normal termination but failure</td><td>46.88%</td><td>53.13%</td><td>46.88%</td><td>40.63%</td><td>59.38%</td></tr><tr><td>No terminal signal</td><td>37.50%</td><td>9.38%</td><td>3.13%</td><td>3.13%</td><td>0.00%</td></tr><tr><td>Failure within Turn 1</td><td>3.13%</td><td>6.25%</td><td>9.38%</td><td>6.25%</td><td>9.38%</td></tr><tr><td>Failure within 2 turns</td><td>15.63%</td><td>15.63%</td><td>31.25%</td><td>25.00%</td><td>25.00%</td></tr><tr><td>Failure within 3 turns</td><td>21.88%</td><td>25.00%</td><td>34.38%</td><td>25.00%</td><td>37.50%</td></tr><tr><td>Failure within 5 turns</td><td>25.00%</td><td>40.63%</td><td>40.63%</td><td>31.25%</td><td>59.38%</td></tr><tr><td>Avg. successful turns</td><td>4.40</td><td>3.92</td><td>1.88</td><td>2.06</td><td>1.92</td></tr><tr><td>Avg. failed turns</td><td>13.30</td><td>7.30</td><td>4.06</td><td>5.50</td><td>2.84</td></tr></table>

The implementation does not use heuristic trunk filtering. All trunks are sampled on-policy from the current model. Filtering low-quality trunks could reduce sibling-sampling cost, but would alter the actually visited state distribution and concentrate optimization on manually selected states. Instead, RTPO retains all trunks and uses same-state sibling returns to determine whether a boundary supplies an informative relative signal.

## G.4 Selective-Gradient Optimization and Training Efficiency

Why the loss is restricted to current-turn tokens. RTPO applies the policy loss only to the output tokens of the current-turn siblings. Prefix tokens and the trunk trajectory define the state and conditioning context that were actually reached, while the downstream continuation supplies the return used to evaluate the current action. Assigning the same turn-level advantage to prefix or downstream-continuation tokens would reintroduce the trajectory-level credit contamination analyzed in Theorem 2. Although only one turn receives gradients at a particular reverse stage, every turn is optimized when it becomes the current turn during the complete reverse sweep. The selective loss is therefore intended to isolate causal credit rather than to discard particular turns from training.

Table 14: Training-generation cost and deployment-time response efficiency on $\tau ^ { 3 } .$ -Airline.
<table><tr><td>Method</td><td>Training-generation tokens</td><td>Pass@1</td><td>Response tokens/success</td></tr><tr><td>GRPO</td><td>604,532</td><td>11.25%</td><td>21,952.9</td></tr><tr><td>Tree-GRPO</td><td>781,325</td><td>16.25%</td><td>15,619.5</td></tr><tr><td>RTPO (G = 3)</td><td>846,207</td><td>17.50%</td><td>9,650.5</td></tr></table>

RTPO generates approximately 8.3% more training tokens than Tree-GRPO and 40.0% more than GRPO, while attaining the highest Pass@1. Relative to Tree-GRPO, it reduces response tokens per successful trajectory by 38.2%; relative to GRPO, the reduction is approximately 56.0%. The additional offline continuation sampling therefore does not translate into more verbose deploymenttime inference.

Wall-clock and GPU-hour overhead. Under the same Qwen3-1.7B model, four A100 GPUs, and 160 training trunks, the measured cost is:

Table 15: End-to-end training cost under the same hardware.
<table><tr><td>Method</td><td>Wall-clock</td><td>GPU-hours</td><td>Generated tokens</td><td>Relative cost</td></tr><tr><td>GRPO</td><td>3.80 h</td><td>15.20</td><td>604,532</td><td>1.00×</td></tr><tr><td>RTPO (G = 3)</td><td>5.37 h</td><td>21.48</td><td>846,207</td><td>1.41×</td></tr></table>

RTPO increases wall-clock time and GPU-hours by approximately 41.3%, and training-generation tokens by approximately 40.0%. The additional cost comes primarily from sibling-continuation generation, turn-by-turn reverse updates, and synchronization of the latest policy between reverse stages. RTPO thus trades higher training-time computation for state-matched credit assignment and current-policy downstream continuations.

Compute-matched comparison. The following comparison uses approximately the same total GPU-hour budget. It is distinct from the fully trained RTPO result at 21.48 GPU-hours.

Table 16: GRPO and compute-matched RTPO under approximately equal training cost.
<table><tr><td>Metric</td><td>GRPO</td><td>RTPO (G = 3), compute-matched</td></tr><tr><td>GPU-hours</td><td>15.20</td><td>15.55</td></tr><tr><td>Pass@1</td><td>11.25%</td><td>11.25%</td></tr><tr><td>Pass@4</td><td>25.00%</td><td>35.00%</td></tr><tr><td>Environment completion</td><td>20.00%</td><td>30.00%</td></tr><tr><td>Generation truncation</td><td>60.00%</td><td>47.50%</td></tr><tr><td>Total tool calls</td><td>118</td><td>98</td></tr><tr><td>Tool-execution success</td><td>57.63%</td><td>62.24%</td></tr><tr><td>Average tool calls</td><td>1.475</td><td>1.225</td></tr><tr><td>Average response tokens</td><td>2,469.7</td><td>2,437.7</td></tr></table>

With only approximately 2% more GPU-hours, RTPO matches GRPO on Pass@1, improves Pass@4 from 25.00% to 35.00%, and increases environment completion from 20.00% to 30.00%. It also reduces truncation, total tool calls, and average response length while improving tool-execution success.

Matched maximum inference budget. All methods below use the same limits of 20 turns, 2,048 tokens per turn, and 8,192 response tokens per trajectory.

Table 17: Performance under the same maximum inference budget.
<table><tr><td>Method</td><td>Pass@1</td><td>Pass@4</td><td> $\operatorname { A v g } .$  tool calls</td><td>Avg. response tokens</td><td>Successes/1K tokens</td></tr><tr><td>GRPO</td><td>11.25%</td><td>25.00%</td><td>1.475</td><td>2,469.7</td><td>0.046</td></tr><tr><td>Tree-GRPO</td><td>16.25%</td><td>25.00%</td><td>0.013</td><td>2,538.2</td><td>0.064</td></tr><tr><td>RTPO (G = 3)</td><td>17.50%</td><td>30.00%</td><td>1.038</td><td>1,688.8</td><td>0.104</td></tr></table>

Under the same maximum inference budget, RTPO attains the highest Pass@1, Pass@4, and number of successful trajectories per 1,000 response tokens, while producing the shortest average responses. Its gains therefore do not come from allowing longer test-time trajectories.

## G.5 Tool-Use Behavior on GSM8K

The total number of tool calls should be interpreted relative to dataset size: GSM8K contains 1,319 distinct test problems, while AIME24 and AIME25 each contain 30. Across all three benchmarks, RTPO improves accuracy while reducing the total number of calls relative to the vanilla model.

Table 18: Accuracy and total tool calls across mathematical benchmarks.
<table><tr><td>Benchmark</td><td>Vanilla accuracy / calls</td><td>RTPO accuracy / calls</td></tr><tr><td>AIME24</td><td>3.33% / 17</td><td>33.33% / 4</td></tr><tr><td>AIME25</td><td>13.33% / 23</td><td>26.67% / 2</td></tr><tr><td>GSM8K</td><td>76.95% / 731</td><td>94.84% / 482</td></tr></table>

We further audit the complete trajectories for the first 100 distinct GSM8K test problems generated by the RTPO Qwen3-4B checkpoint.

There are no repeated questions and no trajectory invokes Python more than once. All 36 calls contain valid syntax and execute successfully. In 34 of the 36 tool-using trajectories, the model has already derived the correct numerical result before invoking Python. The calls are one-shot arithmetic checks rather than multi-step tool search, repeated code, or duplicate execution. For example, on GSM8K QID 55, the model first derives 30 − 2 = 28 and $2 8 / 2 \overset { \cdot } { = } 1 4$ , then invokes Python once to verify the result before returning 14 . Thus, the GSM8K call total reflects many distinct problems receiving a single low-cost numerical verification, rather than repeated tool use within a small set of trajectories.

Table 19: Audit of Python use in 100 distinct GSM8K trajectories.
<table><tr><td>Audit metric</td><td>Result</td></tr><tr><td>Distinct trajectories audited</td><td>100</td></tr><tr><td>Trajectories using Python Average calls per trajectory</td><td>36%</td></tr><tr><td>Trajectories with zero calls</td><td>0.36</td></tr><tr><td></td><td>64</td></tr><tr><td>Trajectories with one call</td><td>36</td></tr><tr><td>Trajectories with two or more calls Calls with valid Python syntax</td><td>0</td></tr><tr><td></td><td>36/36</td></tr><tr><td>Calls executed successfully</td><td>36/36</td></tr><tr><td>Correct tool-using trajectories</td><td>34/36</td></tr><tr><td>Trajectories with repeated calls</td><td>0</td></tr></table>

## G.6 On-Policy Continuation and Synchronization Frequency

As characterized by Theorem 3, standard RTPO synchronizes the rollout model after every reverse stage. When turn k is optimized, its downstream continuation is therefore generated by the current policy $\pi _ { \boldsymbol { \theta } _ { > k } }$ , after turns $k + 1 , \ldots , K - 1$ have been updated. The resulting estimator is

$$
\widehat { Q } _ { j , k } ^ { \mathrm { o n } } = r _ { j , k } + \gamma ^ { \tau _ { j , k } } \widehat { F } _ { j , k } ^ { \pi _ { \theta _ { > { k } } } } , \qquad \omega _ { j , k } \equiv 1 .\tag{108}
$$

This construction assigns the turn-k action a return under the current downstream policy and avoids a product of trajectory-level importance ratios.

At the opposite synchronization endpoint, the off-policy variant reuses downstream continuations generated by the initial rollout policy $\pi _ { \theta _ { 0 } }$ and applies a clamped trajectory-level importance weight:

$$
\begin{array} { r } { \widehat { Q } _ { j , k } ^ { \mathrm { o f f } } = r _ { j , k } + \gamma ^ { \tau _ { j , k } } \overline { { \omega } } _ { j , k } \widehat { F } _ { j , k } ^ { \pi _ { \theta _ { 0 } } } , } \end{array}\tag{109}
$$

$$
\overline { { \omega } } _ { j , k } = \mathrm { c l a m p } \left( \prod _ { h = k + 1 } ^ { K - 1 } \prod _ { t = 1 } ^ { T _ { j , h } } \frac { \pi _ { \theta _ { > k } } ( a _ { j , h , t } \mid s _ { j , h , t } ) } { \pi _ { \theta _ { 0 } } ( a _ { j , h , t } \mid s _ { j , h , t } ) } , \rho _ { \operatorname* { m i n } } , \rho _ { \operatorname* { m a x } } \right) .\tag{110}
$$

Table 20: Output-hit accuracy of off-policy and per-stage on-policy RTPO with Qwen3-8B.
<table><tr><td>Benchmark</td><td>Off-policy</td><td>On-policy</td><td>Difference</td></tr><tr><td>GAIA</td><td>23.30%</td><td>29.13%</td><td>+5.83 pp</td></tr><tr><td>GAIA Level 3</td><td>8.33%</td><td>16.67%</td><td>+8.33 pp</td></tr><tr><td>WebWalkerQA</td><td>6.00%</td><td>9.50%</td><td>+3.50 pp</td></tr><tr><td>XBench</td><td>8.00%</td><td>15.00%</td><td>+7.00 pp</td></tr><tr><td>HLE</td><td>12.45%</td><td>12.12%</td><td>-0.33 pp</td></tr></table>

Per-stage on-policy continuation yields consistent gains on the long-horizon deep-search tasks, with the largest improvement on GAIA Level 3. The two variants are broadly comparable on HLE, which is dominated by more static and shorter retrieval. These results localize the cost of stale continuations to the long-horizon settings targeted by RTPO. Fixed-interval synchronization and policy-KL-based adaptive synchronization lie between the fully on-policy and fully off-policy endpoints: they can reduce synchronization cost, but no longer strictly satisfy $\overline { { \omega } } _ { j , k } \equiv 1$

## G.7 Budget Consumption Across Training

The initial policy produces longer trajectories and more tool calls, so early training stages are more expensive per trunk. The training schedule does not, however, reserve a fixed number of tool calls for each stage. Every stage continues under the predefined trunk-sampling and update schedule, and the actual per-trunk cost decreases as the policy becomes more efficient.

Table 21: Evolution of success, tool use, and generation cost during RTPO training.
<table><tr><td>Training point</td><td>Task success</td><td>Tool calls/trunk</td><td>Tool-execution success</td><td>Generated tokens/trunk</td></tr><tr><td>Step 10</td><td>15.63%</td><td>7.34</td><td>74.04%</td><td>8,013.5</td></tr><tr><td>Step 20</td><td>37.50%</td><td>2.06</td><td>72.73%</td><td>7,758.8</td></tr><tr><td>Step 30</td><td>50.00%</td><td>0.97</td><td>87.10%</td><td>3,533.3</td></tr><tr><td>Step 40</td><td>56.25%</td><td>1.84</td><td>94.92%</td><td>5,469.0</td></tr><tr><td>Step 50</td><td>40.63%</td><td>0.94</td><td>93.33%</td><td>1,669.3</td></tr></table>

From Step 10 to Step 50, tool calls decrease from 7.34 to 0.94 per trunk, generated tokens decrease from 8,013.5 to 1,669.3 per trunk, and tool-execution success increases from 74.04% to 93.33%. Tool use therefore becomes less frequent and more reliable as training progresses.

## G.8 Further Distinctions from Related Methods

SeeUPO. SeeUPO and RTPO both use reverse-order sequential updates at the procedural level, but they differ in motivation, theoretical object, and algorithmic mechanism. SeeUPO abstracts multi-turn interaction as sequentially executed multi-agent bandits and uses backward induction to establish monotonic improvement and global convergence for critic-free backbone algorithms. RTPO instead begins from three structural inconsistencies in a flattened multi-turn training pipeline: Rollout–Training Mismatch, Trajectory-Only Credit Assignment, and Long-Horizon Policy Drift. Its reverse order is one component of a turn-boundary formulation that is combined with state-matched sibling comparison and on-policy continuation. The contribution is therefore not the isolated use of reverse order, but the unified diagnosis, formalization, algorithm, and guarantees for the three coupled failure mechanisms.

ArCHer. ArCHer addresses delayed reward in long-horizon multi-turn interaction through a hierarchical actor–critic design. Its high-level component learns turn-level values with off-policy value-based RL, and its low-level component uses the critic to train the token-level policy within each turn. RTPO does not learn an explicit critic; it constructs a turn-level Monte Carlo advantage from sibling returns sampled from the same boundary state. ArCHer also does not directly target the rollout–training conditioning mismatch or the asynchronous downstream-continuation drift analyzed by RTPO.

REFUEL. REFUEL and RTPO both avoid an independent critic, but they address different mismatches. REFUEL uses covariate shift to describe the difference between training histories generated by a reference policy and deployment histories generated by the current learner. It iteratively collects self-generated data and reformulates multi-turn optimization as relative-future regression tasks. RTPO’s Rollout–Training Mismatch occurs within the same sampled batch, when rollout and likelihood recomputation condition on different contexts for the same tokens. Thus, REFUEL addresses policy-induced covariate shift across data-collection stages, whereas RTPO addresses context inconsistency between rollout and training-time recomputation. The matched τ<sup>3</sup>-Airline results in Table 8 additionally show higher final success, more reliable termination, and shorter responses for RTPO.

R<sup>3</sup> and SRL. R<sup>3</sup> mitigates sparse-reward exploration by moving the curriculum starting point backward along an expert reasoning trajectory; its reverse mechanism is a demonstration-based reverse curriculum. SRL also relies on expert trajectories and derives step-wise supervision from similarity between model and expert actions. RTPO remains outcome-supervised and on-policy, estimates turn-level advantages from sibling continuations at the same boundary state, and jointly addresses conditioning-context mismatch, upstream-state contamination, and asynchronous continuation drift.

## G.9 Additional Limitations

The supplementary results expose several limitations. First, the strict convergence guarantee in Theorem 1 belongs to the turn-level tabular formulation with exactly fixed downstream policies. A shared neural policy only approximates this recursive structure and does not inherit a global convergence guarantee for non-convex optimization.

Second, RTPO is more expensive to train than GRPO. With $G = 3 .$ wall-clock time and GPU-hours increase by approximately 41.3%. Increasing to $G = 4$ improves Pass@1 and produces more concise inference, but raises training-generation tokens by approximately 152.8%, revealing a substantial performance–cost trade-off.

Third, even with $G = 4 ,$ the zero-advantage rate in the sparse binary-reward Airline environment remains above 80%. Denser rewards, adaptive sibling sampling, or prioritized selection of turn boundaries may improve the density and efficiency of the learning signal.

Fourth, the online Airline success rate decreases from 56.25% at Step 40 to 40.63% at Step 50, indicating possible late-stage over-optimization on a limited set of sparse-reward training tasks.

Finally, the original experiments report averages over three runs and include Qwen3-4B and Qwen3- 8B, with $K = 3$ turns for mathematical reasoning and $K = 6$ turns for web-search tasks. Broader per-seed stability curves, additional model scales and turn lengths, and intermediate synchronization schemes based on a fixed interval or policy KL remain useful directions for future work.
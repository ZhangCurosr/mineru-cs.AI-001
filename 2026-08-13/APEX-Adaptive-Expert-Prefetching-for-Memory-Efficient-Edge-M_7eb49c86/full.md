# APEX: Adaptive Expert Prefetching for Memory-Efficient Edge MoE Inference

Alish Kanani, Layan Badawi and Umit Y. Ogras

University of Wisconsin–Madison; {ahkanani, lbadawi, uogras}@wisc.edu

Abstract—Mixture-of-Experts (MoE) models are attractive for edge deployment because they provide high model capacity while activating only a small subset of parameters per token, improving compute efficiency. However, MoE inference at the edge is fundamentally limited by memory. Expert parameters are large and often reside in off-chip memory due to capacity, cost, and power constraints, putting expert loading to the critical path. We present APEX: Adaptive Expert Prefetching, a predictive resource management framework that overlaps expert loading with useful computation. APEX introduces a lightweight prefetch router that predicts candidate experts before the attention block to dynamically fetch additional experts using a learned confidence model. This adaptive strategy achieves over 99% overlap accuracy, significantly outperforming fixed top-k prefetching techniques. APEX supports two execution modes: a correctnesspreserving mode that guarantees exact routing semantics, and a stall-free mode that eliminates residual stalls by operating on available experts with negligible impact on application accuracy. Across multiple MoE models, the correctness-preserving mode reduces per-token latency by up to 26% and improves energydelay product (EDP) by up to 41% over state-of-the-art baselines, while the stall-free mode provides additional efficiency gains with negligible impact on application accuracy. These results establish adaptive, confidence-driven expert prefetching as an effective approach for efficient MoE inference on edge systems.

## I. INTRODUCTION

Large language models (LLMs) are increasingly deployed at the edge, where low-latency response, energy efficiency, data privacy, and reduced cloud dependence are critical requirements [1], [2]. Mixture-of-Experts (MoE) models are particularly well-suited for this setting, as they decouple model capacity from per-token compute [3], [4]. By activating only a small subset of experts per token, MoEs achieve high accuracy while operating within tight compute (FLOPs) budgets. This technique enables edge accelerators to support much larger models that would otherwise be computationally prohibitive.

MoE inference at the edge faces a fundamental memory bottleneck [5]. Unlike dense models with deterministic weights, MoEs repeatedly fetch large, sparse, and irregularly accessed expert parameters [6], [7]. Keeping all experts resident in high-bandwidth on-package memory (e.g., HBM or GDDR) is impractical on edge platforms due to capacity, cost, thermal, and power constraints [2], [8], [9]. In practice, expert parameters are stored in lower-cost off-chip memory (e.g., LPDDR), while non-expert weights, activations, and the key-value (KV) cache remain on-package. Although this organization is more realistic for edge AI accelerators, it places expert loading on the critical path of token generation.

Figure 1 illustrates the memory bottleneck on a representative high-end edge accelerator composed of a compute chiplet with on-package memory and an I/O chiplet. The modeled compute capability and on-package memory bandwidth are comparable to high-end edge GPU platforms such as NVIDIA RTX 3090 [10]. To reflect practical deployments, expert weights reside in host-attached LPDDR5X and are streamed to the accelerator over a PCIe 6.0 ×16 link. Non-expert weights, activations, and the KV cache remain on-package. Full timing and per-bit energy parameters are provided in Section V. Under this realistic memory hierarchy, expert loading becomes a dominant cost. For single-token generation with the Granite-3.1-3B-A800M model [11], expert loading contributes 43% of latency and 29% of total energy. Importantly, this cost is not purely a bandwidth problem. The compute package remains underutilized yet continues to dissipate static power while waiting for expert transfers.

![](images/675b292a3cd3883cf641f35e4e1aadc3f137ca4a51a17be1ddefeaa89abea1fd.jpg)  
Fig. 1. Latency and energy breakdown for the Granite-3.1-3B-A800M model on a representative edge platform with off-chip expert memory and 1024- token context length.

We argue that the solution is not static memory scaling, but adaptive resource management [12]. If the required experts can be predicted early, they can be prefetched to reduce exposed I/O latency. Prior work follows this idea by statically prefetching top-k predicted experts, assuming that prediction accuracy (i.e., the overlap between the predicted and actual top-k experts) is sufficient to ensure coverage [13]. However, routing uncertainty varies significantly across layers and tokens, and prediction errors can lead to missing experts. Even a relatively high average prediction accuracy (e.g., 70– 85%) is insufficient, since some layers may miss consistently more experts and experience stalls that negate much of the prefetching benefit [13]–[15]. Hence, existing approaches with a fixed prefetch budget fail to adapt to layer and input variations, leaving significant performance and energy gains unrealized.

We address this problem with APEX, an adaptive expert prefetching framework that overlaps expert loading with attention computation. APEX adds a lightweight prefetch router before the attention block to predict a ranked list of candidate experts for the current layer. Rather than always prefetching a fixed top-k set, APEX exploits the observation that most routing misses can be avoided by fetching only a small, token-dependent number of additional experts. Therefore, it prefetches a top- $( k + { \hat { \delta } } ( x ) )$ set, where <sup>ˆ</sup>δ(x) is dynamically selected per token using a learned confidence model. This enables APEX to fetch just enough additional experts to suppress misses without paying the energy cost of aggressive over-prefetching on every token. While fetching additional experts slightly increases communication energy, it substantially reduces the latency and stall time associated with exposed expert loading, resulting in a net gain in both energy and performance. APEX supports two execution modes: (1) a correctness-preserving mode guarantees exact routing semantics by fetching any rarely missing experts, and (2) an optional stall-free mode, which eliminates residual stalls by operating on available experts when the application can tolerate the resulting accuracy trade-off.

We evaluate APEX across IBM Granite-3.1-1B-A400M [20], Granite-3.1-3B-A800M [11], Microsoft Phi-mini-MoE-7B-A2.4B [21] and DeepSeek-V2-Lite-16B-A2.4B [22] models on edge configurations with off-chip expert memory. It achieves near-oracle overlap (>99%), substantially reducing expert-loading stalls and improving end-to-end EDP by up to 41% over state-of-the-art baselines. We further show that the stall-free variant achieves an additional 2–14% EDP improvement with negligible impact on application accuracy. These results demonstrate that adaptive expert prefetching is a practical and effective approach for enabling efficient MoE inference at the edge.

The key contributions of this work are as follows:

• An adaptive top-(k + <sup>ˆ</sup>δ(x)) prefetching strategy that adaptively selects the number of prefetched experts using a learned confidence model.

• Two execution modes: a correctness-preserving mode, which guarantees exact routing semantics, and an optional stallfree mode, which further reduces latency and energy with negligible impact on application accuracy.

• Comprehensive evaluations across multiple MoE models on an edge platform with off-chip expert memory, achieving >99% expert overlap and up to 41% improvement in endto-end EDP over state-of-the-art baselines.

In the rest, Section II reviews related work, while Section III provides background on MoE inference. Section IV presents our adaptive expert prefetching method, followed by evaluations in Section V, and Section VI concludes the paper.

## II. RELATED WORK

MoE models scale model capacity by replacing dense feed-forward networks with multiple experts and a router that activates only a sparse subset per token [23]. This conditional-computation paradigm has been extended to largescale training systems such as GShard [6] and later improved through routing stability, load-balancing, and training-stability techniques [24]–[27]. More recently, MoE architectures have been adopted in both open and proprietary LLMs. Models such as Mixtral [4], DeepSeekMoE [28], and OLMoE [29] extend MoE designs to modern decoder-only LLMs, enabling efficient scaling in practical deployments.

A substantial body of work addresses system-level challenges in MoE training and inference at a datacenter scale [30], [31]. Early systems, such as FastMoE [32], DeepSpeed-MoE [33], and MegaBlocks [34], optimize token dispatch, expert parallelism, and all-to-all communication to enable efficient large-scale training. However, these systems assume that expert parameters reside in accelerator memory and rely on high-bandwidth interconnects. These assumptions do not hold in edge deployments, where memory capacity is limited, and experts are often offloaded to external memory. As a result, expert movement becomes a dominant runtime bottleneck.

Hot-expert caching methods, such as MoE-Infinity [17], exploit temporal locality by keeping frequently activated experts in GPU memory and prefetching others from host memory. This is effective when persistent device memory is available and expert reuse is stable across tokens or requests. In edge settings, however, on-package memory must also hold nonexpert weights, activations, the KV cache, and current-layer staging buffers. Reserving capacity for static hot experts can reduce the space available for token-specific expert movement.

MoE optimization methods for edge inference can be broadly divided into heuristic-based and predictive approaches, as summarized in Table I. Heuristic methods reduce memory pressure through caching, reuse, or architectural modifications, without explicitly predicting future expert usage. Pre-gated MoE shifts routing earlier by having layer L select experts for layer L + 1, enabling partial overlap through buffering, but cannot preserve the correctness of the original routing decisions [16]. AdapMoE combines cross-layer prefetching and cache allocation, but changes the number of active experts and is orthogonal to fixed top-k routing [14]. MoE-SpAc uses speculative decoding to anticipate expert demand, but it operates at the sequence level rather than modeling per-layer routing decisions [35].

Predictive approaches forecast future expert usage and prefetch weights ahead of execution to hide data movement latency. Fate [18] leverages cross-layer routing similarity to predict expert usage in subsequent layers, while HOBBIT [15] extends this idea with mixed-precision expert loading, combining prediction with hardware-aware optimizations, but without strict correctness guarantees. Recent pre-attention expert prediction work [19] uses same-layer pre-attention activations to predict expert selections earlier in the layer, improving prediction accuracy. While such methods improve the predictor, they still rely on a fixed predicted set unlike APEX, which focuses on adaptive resource management. ProMoE, the closest state-of-the-art work, uses a learned predictor based on intermediate activations to anticipate routing decisions and prefetch experts [13]. However, it employs a static top-k prefetch policy that does not adapt the prefetch budget, which we evaluate as a primary baseline in Section V.

TABLE I  
COMPARISON OF STATE-OF-THE-ART MOE INFERENCE METHODS FOR EDGE DEPLOYMENT
<table><tr><td>Approach</td><td>Early Expert Prediction</td><td>Dynamic Prefetch Budget</td><td>Correctness Guarantee</td><td>Stall-free</td><td>Key Idea</td></tr><tr><td>Pre-gated MoE [16]</td><td>X</td><td>×</td><td>X</td><td>√</td><td>Router in layer L selects experts for layer L + 1</td></tr><tr><td>MoE-Infinity [17]</td><td>X</td><td>×</td><td>√</td><td>×</td><td>Prefetching is guided by historical traces</td></tr><tr><td>Fate [18]</td><td>√</td><td>X</td><td>X</td><td>X</td><td>Cross-Layer prefetch router</td></tr><tr><td>HOBBIT [15]</td><td>√</td><td>X a</td><td>×</td><td>X</td><td>Prefetching mixed precision experts</td></tr><tr><td>Pre-Attn Pred. . [19]</td><td>√</td><td>×</td><td>√</td><td>×</td><td>Same-layer pre-attention expert prediction</td></tr><tr><td>ProMoE [13]</td><td>√</td><td>×</td><td>√</td><td>×</td><td>Learned proactive expert caching</td></tr><tr><td>APEX</td><td>√</td><td>√</td><td>√</td><td>√</td><td>Adaptive expert prefetching using per-token confidence</td></tr></table>

<sup>a</sup> It uses an adaptive predictor that continues predicting future layers while the predicted experts remain in cache, but it is still not adaptive per-token.

Despite these advances, existing prediction-based methods prefetch only the predicted top-k experts. As shown in Table I, none of them adapts the prefetch budget using prediction confidence. In contrast, APEX uses an adaptive top- $( k + { \hat { \delta } } ( x ) )$ prefetch strategy that dynamically determines the minimal additional budget needed to meet a target coverage. By using a learned confidence model, APEX improves coverage reliability while maintaining energy efficiency and enabling both correctness-preserving and stall-free execution modes.

## III. BACKGROUND AND MOTIVATION

This section reviews the execution characteristics of MoE layers in edge settings, highlights the resulting memory bottlenecks, and discusses the limitations of static prefetching.

## A. MoE Inference at the Edge

Modern LLMs are built on the Transformer architecture, where each layer consists of a self-attention block and a feed-forward network (FFN) [36]. The attention mechanism captures token-to-token interactions for contextual reasoning, while the FFN provides most of the model capacity. In MoE models, the dense FFN is replaced by a set of N experts, each implemented as a smaller FFN and a router, as shown in Figure 2(a). For each token, the router computes a probability distribution over experts and selects the top-k experts [23]. These experts process the token independently, and their outputs are combined using a weighted sum. Training objectives encourage balanced expert utilization [6]. Thus, routing decisions vary across tokens and layers, making expert selection prediction challenging at inference.

A key property of MoE inference is the sparse and irregular expert activation, which varies across tokens and layers. As a result, most experts remain idle for most of the execution while still occupying memory. In edge deployments, keeping all experts in high-bandwidth memory is impractical due to capacity, cost, and power constraints. Instead, expert weights are stored in off-package memory, while non-expert weights, activations, and the KV cache remain on-package. This creates a hierarchical memory system where expert weights must be fetched over a slower off-package path, such as PCIe. During inference, the router selects the top-k experts, triggering ondemand loading of their weights before computation can proceed. This introduces a serialization between routing and expert execution, placing expert loading on the critical path.

![](images/74981d69767f34e730dd2c8d14560e9eb9cf7d6f1dc37fb3a968c1f9396d969a.jpg)  
Fig. 2. Comparison of (a) standard MoE block; and (b) our proposed architecture with an additional prefetch router.

Due to the size and irregular access of expert parameters, this data movement dominates latency and energy while leaving compute resources idle during the transfer.

Attention computation provides a natural opportunity to overlap expert loading with useful work. If expert usage can be predicted before attention, the corresponding weights can be prefetched while attention is executing, partially or fully hiding expert-loading latency. However, imperfect predictions may miss required experts, reintroducing stalls and limiting the effectiveness of naive top-k prefetching, as discussed next.

## B. Why Fixed Top-k Prefetch is Insufficient?

While predictive prefetching overlaps expert transfers with attention computation, existing approaches prefetch only the predicted top-k experts. We show that this strategy is insufficient to reliably eliminate stalls in MoE inference.

Challenge 1: High overlap accuracy does not imply stallfree execution: A key limitation of top-k prefetching is its sensitivity to even small prediction errors. If any expert selected by the original router is missing from the prefetched set, it must be fetched on demand, exposing correction latency on the critical path. Hence, average overlap accuracy across all layers can be misleading. For instance, predicting k −1 out of k experts yields an overlap of $\textstyle { \frac { k - 1 } { k } }$ (e.g., 75% for k = 4 and 87.5% for k = 8). Despite these seemingly high values, tokens at every layer still incur a correction stall, revealing a fundamental limitation. Therefore, we consider per-layer, per-token overlap and complement it with application-level accuracy.

Challenge 2: Token-level heterogeneity: Routing uncertainty varies significantly across tokens and layers. Tokens corresponding to common patterns (e.g., frequent words or simple continuations) exhibit predictable routing, while more complex or context-dependent tokens are harder to predict. A fixed prefetch policy cannot adapt to these variations. A small fixed δ leads to frequent misses on difficult tokens, while a large δ wastes communication energy on easy ones. Therefore, determining how many additional experts to prefetch is a nontrivial, per-token decision.

Challenge 3: Runtime decision must be lightweight: Determining the prefetch budget must (1) be early enough to maximize overlap with attention and (2) introduce minimal area, power, and performance overhead. These constraints limit the available decision window and rule out complex or compute-intensive policies.

![](images/d18d2a89b2de2bf7e342551842a7932451527987debf6f26c7b7caf47aba008b.jpg)  
Fig. 3. Per-layer overlap accuracy for Granite-3.1-1B-A400M $\left( k \ = \ 8 \right)$ Increasing the prefetch budget (δ) stabilizes coverage near 100%, effectively mitigating the routing stochasticity and misprediction stalls.

Key Insight: Small additional prefetch enables nearcomplete coverage. Eliminating stalls requires full coverage, i.e., ensuring all experts selected by the original router are present in the prefetched set. Our measurements show that most token-layer instances require only a small number of additional experts beyond top-k to achieve this. Figure 3 illustrates this behavior for the Granite-3.1-1B-A400M model. While top-k prefetching results in variable and suboptimal overlap across layers, extending the prefetched set to top-$( k + \delta )$ rapidly improves coverage, approaching near-oracle levels with small δ. Even modest increases (e.g., $\delta \ = \ 2$ or $\delta \ : = \ : 4 )$ significantly reduce misses, indicating that most prediction errors are localized and can be corrected with minimal additional prefetches.

These observations motivate moving beyond fixed top-k prefetching toward an adaptive strategy that determines the prefetch budget per token and per layer. The goal is to select the minimal additional budget needed to achieve high coverage without incurring unnecessary communication overhead. APEX achieves this by selecting a token-dependent extraprefetch budget $\hat { \delta } ( x )$ that balances coverage and efficiency.

## IV. ADAPTIVE PREFETCHING: TOP- $( k + \hat { \delta } ( x ) )$ EXPERT PREFETCHING WITH LOGISTIC CDF

## A. Overview of Adaptive Expert Prefetching

Our goal is to reduce exposed expert-loading latency by making expert movement a predictive, token-aware runtime decision. To achieve this, we augment each MoE layer with a lightweight prefetch router placed before the attention block. The original router selects the exact top-k experts after attention. In contrast, the proposed prefetch router runs before attention to predict a ranked list of candidate experts and initiate data movement while attention is still executing. This transforms the conventional route → load → execute pipeline into $\mathtt { p r e d i c t } \to \mathtt { p r e f e t c h } \to \mathtt { \Gamma }$ execute, as shown in Figure 4.

Importantly, APEX does not modify the original routing semantics. The prefetch router acts as an auxiliary prediction path that exposes early information for scheduling data movement, while the original router continues to determine the final expert set used for computation. This clean separation decouples expert prediction for memory scheduling from expert selection for model execution.

The key question is how many experts to prefetch as discussed in Section III-B. APEX prefetches the top- $( k { + } \hat { \delta } ( x ) )$ experts, where the additional prefetch budget $\hat { \delta } ( x )$ is determined dynamically for each token. As shown in Figure 4(b), $\hat { \delta } ( x )$ depends on token-level confidence. Easy, high-confidence tokens require only a small prefetch set, while uncertain tokens prefetch a few additional experts to improve coverage.

APEX operates in two stages.

1) Prefetch stage: The prefetch router produces a ranked expert list, and a confidence model selects $\hat { \delta } ( x )$ . The runtime then issues asynchronous direct memory access (DMA) transfers for the top- $. ( k + { \hat { \delta } } ( x ) )$ experts while attention executes.

2) Execution stage: After attention, the original router determines the exact top-k experts for computation.

If all routed experts are already loaded to on-package memory, execution proceeds immediately. Otherwise, APEX uses one of two modes: correctness-preserving mode fetches the missing routed experts while computing the available ones, whereas stall-free mode avoids the correction stall by executing with the available routed experts and high-scoring prefetched substitutes.

## B. Problem Formulation and Learning Objective

Let $\mathcal { E } = \{ e _ { 1 } , e _ { 2 } , \dots , e _ { N } \}$ denote the ordered list of experts produced by the prefetch router, sorted from the highest (most likely to be used) to the lowest predicted score, where N is the total number of experts. Let ${ \cal K } _ { r } \subseteq \mathcal { E } _ { \mathfrak { c } }$ , with $| K _ { r } | = k ,$ , denote the set of experts selected by the original router. Figure 5(a) illustrates an example where the prefetch router outputs a ranked list of 8 experts, and the original router selects 4 experts $( k = 4 )$ . For any extra-prefetch budget $\delta \in \{ 0 , 1 , \ldots , N - k \}$ the prefetched expert set is:

$$
\mathcal { K } _ { p } ^ { ( \delta ) } = \{ e _ { 1 } , e _ { 2 } , \dots , e _ { k + \delta } \}\tag{1}
$$

A prefetch decision is successful if all required experts are already present in the prefetched set:

$$
K _ { r } \subseteq \mathcal { K } _ { p } ^ { ( \delta ) } \quad \Leftrightarrow \quad \mathcal { K } _ { r } \setminus \mathcal { K } _ { p } ^ { ( \delta ) } = \emptyset\tag{2}
$$

As shown in Figure $5 ( \mathrm { b } )$ , prefetching only the top-k experts $( \mathrm { i . e . , ~ } \delta ~ = ~ 0 )$ can miss required experts, even when most predictions are correct. Extending the prefetched set with a small additional budget δ can eliminate these misses and achieve full coverage.

We define the oracle extra-prefetch budget $\delta ^ { * }$ as the minimum δ that guarantees full coverage:

$$
\delta ^ { * } = \operatorname* { m i n } \left\{ \delta \in \{ 0 , 1 , \dots , N - k \} \ \Big | \ K _ { r } \setminus \big \backslash K _ { p } ^ { ( \delta ) } = \emptyset \right\}\tag{3}
$$

Figure 6(a) shows a representative distribution of $\delta ^ { * }$ obtained by profiling the Granite-1B model (Layer 3) on the WikiText [37] test set. Most tokens require little or no additional prefetch, while a small fraction require larger $\delta ^ { * }$ due to higher routing uncertainty.

![](images/b97a32974cfc5a4317e973f3279231578852093a171b476d56cbfd837bf8ae74.jpg)  
Fig. 4. Overview of the MoE memory bottleneck and APEX. (a) Conventional execution stalls while loading experts from off-package memory. (b) APEX overlaps expert loading with attention using a prefetch router that dynamically selects the extra-prefetch budget $\hat { \delta } ( x )$ based on token-level confidence.

![](images/a0ce81eff09e4db8c3275fed77ef236c2489af3b7f8c2400d9f14949505d30dd.jpg)  
(c) Top-(� + δ) prefetch gives full coverage  
Fig. 5. Illustration of expert prefetch coverage. The prefetch router produces a ranked list of $N = 8$ experts, while the original router selects $k = 4$ experts $( K _ { r } )$ . Prefetching only the top-k experts may miss some required experts. Extending the prefetched set to $\log - ( \bar { k } + \delta )$ achieves full coverage.

Learning Objective: Our goal is to select a token-dependent prefetch budget $\hat { \delta } ( x )$ , where x denotes the token representation available before attention, that minimizes unnecessary data movement while ensuring high coverage probability. Formally, we require that the prefetched set covers the original router’s selection with probability at least τ:

$$
\operatorname* { P r } \left( \mathcal { K } _ { r } \setminus \mathcal { K } _ { p } ^ { ( \hat { \delta } ( x ) ) } = \varnothing \mid x \right) \ge \tau ,\tag{4}
$$

where $\tau \in [ 0 , 1 ]$ is a user-defined coverage target.

Since full coverage is achieved if and only if ${ \hat { \delta } } ( x ) \geq \delta ^ { * }$ the objective can be equivalently written as:

$$
\operatorname* { P r } \left( { \hat { \delta } } ( x ) \geq \delta ^ { * } \mid x \right) \geq \tau\tag{5}
$$

This formulation highlights the core objective: for each token, select the smallest token-dependent $\hat { \delta } ( x )$ that satisfies the coverage constraint, thereby balancing latency reduction against communication cost.

## C. Confidence Modeling via CDF

The adaptive prefetching problem reduces to selecting the smallest extra-prefetch budget that satisfies a target coverage probability. To enable this decision at runtime, we use a lightweight probabilistic model that estimates whether a given number of additional experts is sufficient.

Coverage Probability: We define the probability that a candidate budget δ is sufficient as:

$$
p _ { \delta } ( x ) = \mathrm { P r } ( \delta \geq \delta ^ { * } \mid x ) , \quad \delta \in \{ 0 , \ldots , N - k \}\tag{6}
$$

TABLE II  
SUMMARY OF NOTATIONS USED IN THIS WORK.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $\mathcal { E } = \{ e _ { 1 } , e _ { 2 } , \dots , e _ { N } \}$ </td><td>Ordered list of experts sorted by prefetch router</td></tr><tr><td> $N$ </td><td>Total number of experts</td></tr><tr><td> $k$ </td><td>Number of experts to be executed</td></tr><tr><td> $\delta \in \{ 0 , 1 , \ldots , N - k \}$ </td><td>Extra-prefetch budget</td></tr><tr><td> $\textstyle { \mathcal { K } } _ { r }$ </td><td>Expert set selected by the original router</td></tr><tr><td> ${ \kappa } _ { p } ^ { ( \delta ) }$ </td><td>Prefetched expert set  $\{ e _ { 1 } , \ldots , e _ { k + \delta } \}$ </td></tr><tr><td> $\delta ^ { * }$ </td><td> $\dot { \kappa } _ { r } \subseteq \kappa _ { p } ^ { ( \delta ) }$  Oracle delta: minimum δ such that</td></tr><tr><td> $\hat { \delta } ( x )$ </td><td>Token-dependent (dynamic) prefetch budget</td></tr><tr><td> $\tau \in [ 0 , 1 ]$ </td><td>Target coverage probability threshold</td></tr><tr><td> $p _ { \delta } ( x )$ </td><td>Coverage probability:  $p _ { \delta } ( \dot { \boldsymbol { x } } ) = \operatorname* { P r } ( \delta \geq \delta ^ { * } \mid \boldsymbol { x } )$ </td></tr><tr><td>w</td><td>Learned parameter vector in CDF model</td></tr><tr><td> $\theta _ { 0 } \le \theta _ { 1 } \le \dots \le \theta _ { N - k }$ </td><td>Ordered thresholds (cutpoints) for ordinal model</td></tr></table>

Since larger prefetch budgets can only improve coverage, these probabilities are monotonic:

$$
p _ { 0 } ( x ) \leq p _ { 1 } ( x ) \leq \cdots \leq p _ { N - k } ( x ) ,\tag{7}
$$

which can be interpreted as the cumulative distribution function (CDF) of the oracle $\delta ^ { * }$ . Figure 6(b) provides intuition for this formulation. The learned CDF enables APEX to select the smallest δ that satisfies a target threshold $\tau .$

Ordinal Logistic CDF Model: We model this CDF using an ordinal logistic formulation that captures the ordered structure of δ. The cumulative probability is modeled by:

$$
\begin{array} { r } { p _ { \delta } ( x ) = \sigma ( \theta _ { \delta } - w ^ { \top } x ) , \quad \delta \in \{ 0 , \ldots , N - k \} , } \end{array}\tag{8}
$$

where w is a learned parameter vector, $\{ \theta _ { \delta } \}$ are ordered bias terms satisfying $\theta _ { 0 } \le \theta _ { 1 } \le \dots \le \theta _ { N - k }$ , and $\sigma ( \cdot )$ is the sigmoid function.

The ordered biases partition the latent confidence space, with each threshold corresponding to a different prefetch budget. By construction, this formulation guarantees monotonicity of $p _ { \delta } ( x )$ and therefore defines a valid CDF.

At runtime, given a token representation x and a target coverage τ, we select the smallest prefetch budget that satisfies:

$$
\hat { \delta } ( x ) = \operatorname* { m i n } \left\{ \delta \in \left\{ 0 , \ldots , N - k \right\} \mid p _ { \delta } ( x ) \geq \tau \right\} .\tag{9}
$$

This yields a confidence-aware adaptive policy:

• easy tokens (tokens whose routed experts are likely covered by a small budget) require a small $\hat { \delta } ( x )$ ;

• difficult tokens with low confidence require a larger $\hat { \delta } ( x )$

Thus, unlike static overfetching, APEX fetches just enough additional experts to satisfy the desired coverage target, reducing misses while avoiding unnecessary communication.

![](images/34bba446c319992b5e5bfefe7d092e90890b548b6bcc7dfcfc2db9872eda362d.jpg)  
(a) Distribution of Oracle Delta (δ<sup>∗</sup>) across Tokens

![](images/342f082d931ee4c6ed6d71a434c07bf9190b7ff775c593aa7f1c0487afdc283b.jpg)  
(b) Dynamic Expert Selection using Learned Logistic CDF  
Fig. 6. Adaptive expert prefetching. (a) Representative distribution of oracle extra-prefetch budget (δ<sup>∗</sup>) for Granite-1B (Layer 3) on WikiText [37], showing a long-tail across tokens. (b) Selection of prefetch budget $\hat { \delta } ( x )$ using a learned logistic CDF to satisfy a target coverage threshold (τ).

## D. Prefetch Router: Training and Integration

The key design goal is to enable accurate early prediction for prefetching without modifying or retraining the base MoE model. Accordingly, the LLM remains frozen, and training is limited to the auxiliary prefetching components, i.e., the prefetch router and the CDF model.

Prefetch Router Architecture and Placement: The prefetch router mirrors the original MoE router: a linear layer that produces expert logits, followed by a softmax to produce a probability distribution over experts. It is placed before the attention block of each MoE layer and operates on the layer’s hidden representation. This placement serves two purposes. First, it hides expert loading latency by initiating expert transfers during attention. Second, it remains within the same layer, preserving a strong correlation between the prefetch prediction and the final routing decision made later by the original router.

Prefetch Router Distillation: For each MoE layer, let $q _ { r }$ and $q _ { p }$ denote the expert softmax distributions produced by the original router and the prefetch router, respectively.

The prefetch router is trained to match the original router using a KL divergence loss:

$$
\mathcal { L } _ { \mathrm { K L } } = \sum _ { i } q _ { r } ( i ) \log \frac { q _ { r } ( i ) } { q _ { p } ( i ) } .\tag{10}
$$

Training requires only forward passes through the frozen base model to obtain $q _ { r }$ , and gradients are applied exclusively to the prefetch router. No changes are made to expert execution during this stage. The result is a layer-wise early predictor that approximates the original router’s expert selections.

CDF Model Training: After training the prefetch router, we fit the CDF model described in Section IV-C through a separate lightweight stage. For each token and layer, we use the trained prefetch router to obtain expert probabilities, derive the ranked expert list, and compute the oracle budget $\delta ^ { * }$ by comparing with the original router. These are converted into cumulative binary targets $\mathbb { I } [ \delta \geq \delta ^ { * } ]$ for each δ.

The CDF model is trained using a cumulative binary crossentropy loss:

$$
\mathcal { L } _ { \mathrm { C D F } } = \sum _ { \delta = 0 } ^ { N - k } \mathcal { H } \big ( p _ { \delta } ( x ) , ~ \mathbb { I } [ \delta \ge \delta ^ { * } ] \big ) ,\tag{11}
$$

where $\mathcal { H } ( \cdot , \cdot )$ denotes binary cross-entropy [38].

The model consists of a single weight vector and a small set of thresholds, making training inexpensive and feasible with a modest held-out validation set. Since the prefetch router is distilled from the frozen original router, it is tied to the base model’s routing function rather than to a specific prompt or downstream task. Thus, the same trained prefetch router and CDF model can be reused across prompts and datasets; retraining is only needed if the base MoE model is fine-tuned and its router behavior changes.

Runtime Operation: The prefetch router and CDF model are evaluated before attention to determine the adaptive prefetch budget $\hat { \delta } ( x )$ . The system then asynchronously prefetches the $\scriptstyle \mathrm { { t o p - } } ( k + { \widehat { \delta } } ( x ) )$ experts while attention executes. After attention, the original router produces the final top-k selection, and execution proceeds using either correction or stall-free mode, as described next. The overhead is minimal, as detailed in Section V-F. The prefetch router mirrors the lightweight original router, and the CDF model introduces only a small number of parameters, enabling early prediction and scheduling with negligible compute and storage overhead.

## E. Correctness-preserving and Stall-free Execution Modes

Even with adaptive top- $( k + { \hat { \delta } } ( x ) )$ prefetching, rare mispredictions may occur. Hence, we support two execution modes.

1) Correctness-preserving mode: The system strictly follows the original router decision. If any experts in ${ \boldsymbol { \kappa } } _ { r }$ are missing from the prefetched set, they are fetched while execution begins on available experts, and the final output uses the complete original expert set. This preserves exact MoE semantics, but incurs a small residual correction latency, which is evaluated in Section V-E.

Asynchronous miss correction: APEX does not use a rigid wait-then-execute pipeline when a prefetch miss occurs. Since the selected experts in an MoE layer are independent before the final weighted sum, correctness-preserving execution begins immediately on the routed experts that are already available in the on-package buffer. Missing routed experts are fetched asynchronously in parallel with this available expert computation. After the missing experts arrive, APEX computes only the remaining expert outputs and then performs the final weighted aggregation using the original router weights. Therefore, the correction overhead is the unhidden portion of the miss transfer after overlap with available expert computation, rather than the full load latency. This preserves exact MoE semantics while exploiting inter-expert pipelining.

Algorithm 1 Expert Selection in the Stall-free Mode   
1: Input: Routed experts ${ \boldsymbol { \kappa } } _ { r }$ , original router logits $q _ { r } \in \mathbb { R } ^ { N } ,$   
prefetched experts ${ \kappa } _ { p } ^ { ( \hat { \delta } ) }$   
2: Output: Execution experts $\tilde { \kappa }$   
3: α ← softmax(q ) // original-router expert weights   
4: if $\mathcal { K } _ { r } \subseteq \mathcal { K } _ { p } ^ { ( \hat { \delta } ) }$ then   
5: $\tilde { \mathcal { K } } \gets \mathcal { K } _ { r }$ // all correct experts are available   
6: else   
7: $\mathcal { \tilde { K } } \gets \mathcal { K } _ { r } \cap \mathcal { K } _ { p } ^ { ( \hat { \delta } ) }$ // keep correctly routed experts   
8: $\mathcal { M }  \mathcal { K } _ { r } \backslash \mathcal { K } _ { p } ^ { ( \delta ) }$ // missing routed experts   
9: $\mathcal { C } \gets \mathcal { K } ^ { ( \tilde { \delta } ) } \setminus \tilde { \mathcal { K } } / /$ available replacement candidates   
10: $\mathcal { S } \gets \mathrm { T o p } * | \mathcal { M } | ( \mathcal { C } ; \alpha )$ // highest-weight candidates   
11: ${ \tilde { \mathcal { K } } } \gets { \tilde { \mathcal { K } } } \cup { \mathcal { S } }$   
12: end if   
13: return $\tilde { \kappa }$

2) Stall-free mode: This mode eliminates correction latency by executing only with experts that are already present in the prefetched set. Instead of waiting to fetch missing experts, we approximate the original routing operation using the available experts. If all experts selected by the original router are already prefetched (Algorithm 1, lines 3–4), we directly execute them without any deviation from the behavior of the original architecture. Otherwise, if one or more routed experts are missing, stall-free mode keeps the correctly prefetched routed experts and replaces the missing experts with the same number of highest-weight candidates from the remaining prefetched set, ranked by the original router’s softmax weights (lines 7–11).

Intuitively, this procedure selects the highest-scoring available substitutes from the prefetched set according to the original routing scores, minimizing deviation from the original decision. Because overlap is already very high, such substitutions are rare. Moreover, the missing expert are often a lower-ranked choice, so the resulting approximation can have small model-dependent accuracy impact, as evaluated in Section V-D. We note that, the stall-free mode is an optional mode, while correctness-preserving mode remains the default when exact routing semantics are required.

## V. EXPERIMENTAL EVALUATION

## A. Experimental Setup

Models: We evaluate APEX on four representative MoE LLMs spanning different model scales and configurations:

• Granite-1B (IBM Granite-3.1-1B-A400M [20]): a 1B-scale model with N = 32 experts and top-k routing with $k = 8 .$

• Granite-3B (IBM Granite-3.1-3B-A800M [11]): a 3B-scale model with N = 40 experts and top-k routing with k = 8.

• Phi-7B (Microsoft Phi-mini-MoE-7B-A2.4B [21]): a 7B model with N = 16 experts and top-k routing with $k = 2 .$

• DeepSeek-16B (DeepSeek-V2-Lite-16B-A2.4B) [22]: a 16B model with N = 64 experts and top-k routing with k = 6.

These models capture varying numbers of experts, routing sparsity, and layer depths. Granite-1B, Granite-3B, Phi-7B, and DeepSeek-16B contain 24, 32, 32, and 27 Transformer layers, respectively. In DeepSeek-16B, the first Transformer layer uses a dense FFN, while the remaining 26 layers use MoE FFNs. DeepSeek also includes two shared experts per MoE layer that are activated for every token in addition to the six routed experts selected from the 64 routed experts. Thus, for DeepSeek-16B, shared experts are deterministic per-token computation and are not part of APEX’s adaptive routing or extra-prefetch-budget decision.

Workload: We focus on the decode phase (autoregressive token generation), since it is latency-critical in edge deployments and exposes expert-loading bottlenecks. We evaluate across diverse context lengths, including 512, 1024, and 2048 tokens. For application-level accuracy, we use standard benchmarks including WikiText [37] (perplexity), AI2 Reasoning Challenge [39], Massive Multitask Language Understanding [40], WinoGrande [41], and TruthfulQA [42].

Prefetch Router and CDF Training: Training is performed on an NVIDIA RTX 3090 GPU [10] for Granite and Phi models and A100 [43] for DeepSeek using the WikiText dataset [37]. We use a learning rate of $5 \times 1 0 ^ { - 4 }$ , batch size of 8, sequence length of 1024, and train for 1000 steps. The training takes approximately 10 minutes for Granite-1B, 30 minutes for Granite-3B, and about 1 hour for Phi-7B and DeepSeek-16B. After distillation, the CDF model is fitted using collected oracle $\delta ^ { * }$ values on a held-out validation set. No task-specific retraining or hyperparameter search is used for downstream benchmarks; the auxiliary components trained on WikiText are reused for all evaluated application-level tasks.

Baselines: We compare APEX against three baselines: (i) No Prefetch, the conventional execution scheme in which expert loading occurs on the critical path [44]; (ii) ProMoE, a stateof-the-art prediction-based top-k prefetching approach [13]; and (iii) static overfetch policies with fixed additional-prefetch budgets. We evaluate both correctness-preserving and stallfree modes. To decouple the effect of expert prediction from adaptive budget selection, the static overfetch baselines use the same prefetch-router ranking as APEX and replace the learned $( \hat { \delta } ( x ) )$ with a fixed (δ) for every token and layer. Therefore, differences between static overfetching and APEX isolate the benefit of adaptive per-token budgeting.

The evaluation metrics are (i) overlap accuracy between prefetched and routed experts, (ii) application-level accuracy across benchmarks, and (iii) system efficiency, in terms of latency, energy, and energy-delay-product (EDP).

## B. Hardware Platform and Evaluation Methodology

Target Edge Architecture: We model a modern edgeclass accelerator consisting of a compute chiplet tightly coupled with I/O and memory chiplets, following the prior approach [45]. The compute chiplet integrates multiple vector processing arrays and SRAM buffers [46], while all expert weights reside in off-package LPDDR5X memory. The compute microarchitecture is based on the vector processing array illustrated in Figure 7. Each array is organized as a $1 6 \times 1 6$ grid of vector units with vector length 32, supporting bfloat16 (BF16) matrix-multiply operations. We instantiate four such arrays operating at 750 MHz, achieving a peak throughput of 24 TFLOPS, as summarized in Table III. Each array is provisioned with 2 MB of SRAM (8 MB total) and includes 32 special-function units [47] to support non-linear activations. We use BF16 as the default evaluation datatype to preserve the original model behavior and isolate the impact of prefetching; Section V-G further studies how APEX behaves when expert weights are represented with lower-precision datatypes.

![](images/1d0a45e0f2a6ba78f810a2ee01c46ae48a94047a6a6076636078a9ef31bf8082.jpg)  
Fig. 7. Microarchitecture of a vector processing array.

To reflect realistic edge constraints, on-package memory capacity is varied across model scales (1 GB, 2 GB, and 6 GB), holding only non-expert weights, activations, and KV cache, following the literature [33], [48]. All expert parameters are stored in off-package LPDDR and accessed over a PCIe interconnect. On-package memory bandwidth is modeled at 819 GB/s (HBM3-class), while off-package bidirectional bandwidth is set to 256 GB/s (PCIe 6.0 x16) [49], [50]. The evaluations are performed on a latency-sensitive singleuser edge inference setting, but the proposed method supports batching multiple concurrent requests.

Platform Positioning: The target configuration represents a high-end, near-future edge AI accelerator rather than a low-power smartphone-class SoC. This class is motivated by emerging robotics, industrial, and local GenAI platforms that already integrate high compute throughput with large local memory capacity, such as NVIDIA Jetson AGX Orin/Thor [51], and Hailo-10H [52] class edge accelerators. At the same time, APEX is not tied to this specific bandwidth point. Section V-G evaluates sensitivity to lower and higher off-chip bandwidths, including a 32 GB/s setting representative of more constrained mobile-SoC-class memory systems.

Evaluation Methodology: Our evaluation combines hardware synthesis with cycle-accurate co-simulation rather than pure simulation. The compute arrays are implemented in RTL, synthesized in TSMC 28 nm, and their active/leakage power is characterized using Synopsys PrimeTime [53]. Memory timing is modeled using Ramulator, which captures DRAM bank conflicts, refresh, and controller scheduling [54], while LPDDR and PCIe energy costs are incorporated using established per-bit energy models [49], [55]. The arrays are clockgated, allowing us to accurately model idle periods during stalls. We obtain active power by running the arrays under high activity, and leakage power by evaluating the design under no-activity conditions with clock gating enabled. Onpackage communication is modeled using UCIe-based interconnect parameters that capture low-energy, high-bandwidth chiplet communication [56]. We use an open-source, cycleaccurate compute–communication co-simulation framework, CHIPSIM [57], which integrates: (i) compute timing from RTL-derived models, (ii) memory access timing from DRAM simulation [54], and (iii) per-layer execution traces collected from GPU-based model runs [10]. The simulator models token-level execution, including attention compute, expert routing, data transfers, and expert execution, enabling us to model overlap among attention computation, expert routing, DMA transfers, and expert execution. The asynchronous expert transfers are not modeled as ideal parallel fetches. Each prefetched expert is injected as a DMA transfer into the shared PCIe/LPDDR path, where concurrent requests experience bandwidth limits and queuing delay. CHIPSIM coordinates compute, routing, DMA transfers, and expert execution on a unified timeline. Thus, overlap is limited by the modeled attention window and contention in the off-chip memory path.

TABLE III  
HARDWARE CONFIGURATION AND POWER PARAMETERS
<table><tr><td>Component</td><td>Configuration</td><td>Perf./Energy</td></tr><tr><td>Compute Arrays</td><td> $4 \times ( 1 6 \times 1 6 \times 3 2 )$ </td><td>24 TFLOPS @ 750 MHz</td></tr><tr><td>SRAM [58]</td><td>8 MB (2 MB per array)</td><td>CACTI-based modeling</td></tr><tr><td>HBM Bandwidth [50]</td><td>819 GB/s</td><td>7 pJ/bit</td></tr><tr><td>LPDDR5X [55]</td><td>Off-package experts</td><td>3 pJ/bit</td></tr><tr><td>PCIe 6.0 x16 [49]</td><td>256 GB/s</td><td>5 pJ/bit</td></tr><tr><td>On-chip Comm. [56]</td><td>UCIe Adv.</td><td>0.5 pJ/bit</td></tr></table>

Prefetch-overlap Bound: APEX does not assume that all prefetched experts are fully hidden behind attention. For an extra-prefetch budget (δ), the ideal hiding condition is:

$$
T _ { \mathrm { p r e f e t c h } } ( \delta ) = \frac { \delta S _ { \mathrm { e x p e r t } } } { B _ { \mathrm { e f f } } } + T _ { \mathrm { q u e u e } } \leq T _ { \mathrm { a t t n } } ,\tag{12}
$$

where $( S _ { \mathrm { e x p e r t } } )$ is the expert size, $( B _ { \mathrm { e f f } } )$ is the effective off-chip bandwidth, and $( T _ { \mathrm { q u e u e } } )$ captures memorycontroller/interconnect contention. When this condition is not satisfied, the unhidden portion of $( T _ { \mathrm { p r e f e t c h } } )$ appears as exposed stall time in the simulation. Thus, APEX improves latency and energy only when the saved stall/idle cost exceeds the additional transfer cost. This trade-off is evaluated directly through CHIPSIM and further stressed in the bandwidth sensitivity study in Section V-G.

Methodology Scope and Limitations: CHIPSIM has been validated against hardware measurements on a real chiplet platform under concurrent workloads. However, the target APEX accelerator itself is evaluated through synthesiscalibrated co-simulation rather than measurements on a hardware prototype. We model PCIe/LPDDR contention through the memory/interconnect timing model and account for OSlevel DMA setup and Input-Output Memory Management Unit (IOMMU) overhead using a fixed per-DMA-descriptor cost, which is amortized over multi-MB expert transfers. Building an FPGA or silicon prototype to further validate these systemlevel effects and full end-to-end OS-stack modeling are left as future work.

![](images/cc46e41f72430de4cb79264e74f4e7707fdee8ea4a4094f3c873b6764aa3cf9a.jpg)  
Fig. 8. Per-layer overlap accuracy using a CDF confidence threshold of $\tau = 0 . 9 0$ for APEX. (a) Granite-1B model. The static baselines prefetch top- $\cdot ( k + 2 )$ and top- $\cdot ( k + 4 ) .$ , while APEX prefetches on average 4.17. (b) Granite-3B model (keeping the same static-baseline settings). (c) Phi-7B model. The static baselines prefetch top-(k + 1) and top-(k + 2), while APEX prefetches on average 0.67. (d) DeepSeek-16B model. The static baselines prefetch top-(k + 2) and top-(k + 4), while APEX prefetches on average 1.98.

## C. Expert Prediction Overlap Accuracy

This section analyzes the overlap accuracy between the experts selected by the original router and the prefetched top-$( k { + } \hat { \delta } ( x ) )$ set, since a high overlap is the key enabler for hiding expert-loading latency. Then, the next subsection analyzes the impact of the rare routing mismatches in the stall-free mode on the end-to-end application-level accuracy.

Per-layer Overlap Across Models: We first evaluate how the overlap accuracy changes across layers and model scales to ensure that APEX consistently offers high coverage throughout the network. In this experiment, the confidence threshold is set as $\tau = 0 . 9 0$ to ensure a high and consistent overlap target across all layers. The effect of varying this threshold and its impact on overlap accuracy is analyzed later.

Figure 8 plots the per-layer overlaps between the original router’s selections and the prefetched experts for Granite-1B, Granite-3B, Phi-7B, and DeepSeek-16B. Across all models, APEX consistently maintains over 97% overlap across layers, greatly reducing the variability seen in alternative methods. ProMoE exhibits noticeable degradation in layers 5 and 8 of Granite-1B, with ${ \sim } 7 9 \%$ and ∼84% overlap, respectively. Similarly, it struggles with multiple layers in the Granite-3B, Phi-7B and DeepSeek-16B models, where ProMoE often stays in the high-80% range and drops near 82%. In contrast, APEX consistently achieves around 97–98% overlap. This stability is essential to guarantee that no individual layer acts as a bottleneck, effectively removing layer-wise performance differences caused by mispredictions.

Figure 8 also shows that APEX significantly outperforms static policies that always prefetch 2 and 4 extra experts. Fixed policies must account for worst-case layers, which causes unnecessary overfetching in simpler layers. Even with this over-provisioning, they struggle to prefetch the correct set of experts for many layers $( \mathrm { e . g . , ~ } 5 ^ { \mathrm { t h } }$ layer of Granite-1B and first layer of Phi-7B), which likely exhibit higher routing ambiguity and weaker correlation between the preattention representation and the final routing decision. In contrast to static policies, APEX adapts its prefetch budget to runtime uncertainty, eliminating the low-overlap tail while avoiding excessive overfetch. As a result, APEX maintains over 97% overlap across layers by prefetching 4.17 (Granite-1B), 2.86 (Granite-3B), 0.67 (Phi-7B) 1.98 (DeepSeek-16B) experts on average. These results show that APEX enhances not just average overlap but also worst-case reliability, which is essential for latency-sensitive inference.

Overlap vs. Adaptive Target Threshold: Next, we evaluate how the target confidence threshold τ affects the overlap accuracy and prefetch budget. Figure 9 shows that increasing τ monotonically increases the overlap percentage across all models, as expected. For Granite-1B, the overlap improves from 93.4% with $\tau = 0 . 6 0$ to 98.2% with $\tau = 0 . 9 0$ , and further to 99.4% with $\tau ~ = ~ 0 . 9 7$ , while the average $\hat { \delta } ( x )$ increases from 1.36 to 4.17 and 6.70, respectively. Granite-3B shows a similar trend, with overlap increasing from 93.5% to 98.1% and 99.4%, while the average number of prefetched experts increases from $\ddot { \delta } ( x )$ of 0.67 to 2.86 and 4.93. Similarly, for Phi-7B, the average overlap percentage increases from 94.7% to 97.5%, and then 99.0%, while prefetching only 0.16, 0.67, and 1.75 additional experts on average. Finally, DeepSeek-16B also follows the same pattern, with overlap increasing from 92.9% to 97.4% and 99.0%, while the average prefetched experts increases from 0.37 to 1.98 and 3.66.

These results highlight two key insights. First, the CDFbased selector is well-designed: higher τ consistently yields higher overlap, confirming that APEX can reliably meet the coverage targets. Second, near-complete overlap (>99%) can be achieved with a modest increase in prefetch budget, indicating that APEX efficiently allocates additional experts only when necessary rather than uniformly overfetching. We use $\tau = 0 . 9 0$ in the remaining experiments because it captures most of the overlap gains before saturation. Further increase causes additional prefetching with only marginal benefits.

![](images/935b4f3710dd974531c90cdf7f8cedbb67fc0eef5e97c3e62ad254bc4fb490cf.jpg)  
(a) Granite-1 B  
(b) Granite-3B

![](images/2752f2f3e5782d5b3c35d78c94e14d30ff80d3b5f63d6e5f8a15e650c592f701.jpg)  
(c) Ph i -7B  
(d) DeepSeek-1 6B  
Fig. 9. Average overlap accuracy vs. extra-prefetch budget $( { \hat { \delta } } ) .$ Increasing the CDF confidence threshold (τ) from 0.60 to 0.97 consistently raises overlap toward 100% across all models, validating that the CDF-based selection effectively meets the desired coverage targets.

![](images/be5bb33e9803078e2acfb653bc0d4eab8a1ea6426b5b45694c6f80cf521d8438.jpg)  
Fig. 10. Per-layer additional experts selected by APEX $( \hat { \delta } ( x ) )$ versus oracle extra-prefetch budget (δ<sup>∗</sup>) for Granite-3B (τ = 0.90). Layers are ordered by oracle $\delta ^ { * }$ to highlight relative difficulty.

Adaptive Behavior Compared to Oracle: Next, we analyze how effectively APEX adapts its prefetch budget to the intrinsic difficulty of each layer by comparing it to an Oracle that stores the ground truth. Figure 10 plots the number of required experts (given by the Oracle) and prefetched by APEX with $\tau = 0 . 9 0$ for the representative Granite-3B model. We observe that APEX closely follows the ground truth. APEX prefetches three or fewer experts for most layers, which require one or two additional experts according to the Oracle. That is, it learns the layers for which experts can be predicted accurately and assigns small budgets to them. In more challenging layers that require three or more additional experts, APEX slightly overestimates the oracle budget, which is desirable to provide a safety margin to maintain high overlap.

We observe the same adaptive behavior across other benchmark models. Overall, these results confirm that APEX does not use a coarse global heuristic but dynamically adjusts its prefetch budget in response to layer- and token-level runtime variability, enabling it to achieve consistently high overlap accuracy while remaining efficient. Hence, APEX achieves its primary goal of transforming a highly variable expert prediction problem into a controlled, confidence-driven prefetch policy, which consistently delivers very high overlap accuracy while avoiding the inefficiencies of static overfetching.

Per-token Prefetch Budget Distribution: Beyond average $( \ddot { \delta } ( x ) )$ ), we also examine the per-token distribution of the total prefetched experts $( k + \hat { \delta } ( x ) )$ , since large bursts could stress the PCIe path or on-package buffers. Figure 11 plots the token distribution for Granite-3B at $( \tau = 0 . 9 0 )$ across representative easy, typical, and hardest layers. For easy and typical layers, the distribution is concentrated near (k), with mean (<sup>ˆ</sup>δ) below 3 and fewer than 0.2% of tokens prefetching more than (N/2) experts. The hardest layer requires a larger budget because the pre-attention representation is less correlated with the final router decision. However, APEX prefetches experts layer by layer; each layer’s experts are loaded, used, and evicted before the next layer begins, so peak buffer occupancy is bounded by the maximum single-layer prefetch set rather than accumulating across layers. Thus, the adaptive budget does not create an unbounded memory-pressure.

![](images/781e7d9cd86f7f746a84ea7b8272616f3af1b94f19dc1331afe86ed157a006dd.jpg)  
${ \mathrm { F i g . } }$ 11. Distribution of total prefetched experts per token $( k + { \hat { \delta } } ( x ) )$ for Granite-3B at $\tau = 0 . 9 0 .$

## D. Application-Level Accuracy

This section evaluates the impact of the APEX on the application-level accuracy measured by perplexity and standardized benchmarks AI2 Reasoning Challenge (ARC), Massive Multitask Language Understanding (MMLU), Wino-Grande (WG), and TruthfulQA (TQA). For clarity, we report the perplexity and the average of these standardized scores. These accuracy metrics are compared against the base system, which always uses the original router’s expert choices.

The correctness-preserving mode always matches the accuracy of the baseline by definition. Therefore, Table IV reports perplexity and downstream task performance across four model scales under different confidence thresholds τ. The impact on the perplexity is negligible for the Granite-1B, 3B and DeepSeek-16B models, especially at higher thresholds. Specifically, for Granite-1B moving from correctnesspreserving execution to stall-free at $\tau ~ = ~ 0 . 9 0$ results in only a marginal increase in perplexity $( 7 . 8 8  7 . 9 9 )$ and a small drop in average downstream accuracy $( 4 3 . 3  4 2 . 8 )$ A similar trend is observed in Granite-3B, where the average score decreases modestly from 54.0 to 53.2, with perplexity increasing slightly from 6.79 to 6.83. DeepSeek-16B follows the same trend, with only a small PPL increase at $\tau = 0 . 9 0$ $( 7 . 0 2  7 . 1 0 )$ and a modest average-score drop ( 51.6 → 50.8). These results indicate that in high-overlap regimes, the effect of occasional expert mispredictions is negligible for Granite models.

The robustness in accuracy can be attributed to the routing structure of Granite and DeepSeek models, where each token is processed by multiple experts (k = 8 for Granite and $k = 6$ plus two shared experts for DeepSeek) and the final output is a weighted average. In this setting, mispredictions are more likely to occur among lower-ranked experts that contribute less to the final output, thereby limiting their impact on model accuracy.

In contrast, Phi-7B exhibits a more noticeable degradation in stall-free mode, particularly at lower thresholds. At $\tau = 0 . 6 0$ the average accuracy drops significantly from 64.6 to 58.6, with large reductions in ARC and MMLU scores. Increasing τ improves performance, with the average recovering to 60.4 at $\tau = 0 . 9 0$ , but a gap relative to the baseline remains. The degradation is primarily driven by a subset of benchmarks (notably ARC and MMLU), while others, such as WinoGrande, remain relatively stable.

TABLE IV  
APPLICATION-LEVEL ACCURACY OF APEX IN STALL-FREE $( \mathbf { A P E X } _ { S F } )$ MODE COMPARED TO CORRECTNESS-PRESERVING EXECUTION (BASE).
<table><tr><td>Model</td><td>Config</td><td>PPL↓</td><td>ARC/MMLU/WG/TQA ↑</td><td>Avg ↑</td></tr><tr><td rowspan="4">1B</td><td>Base</td><td>7.88</td><td>40.2 / 28.5 / 60.3 / 44.1</td><td>43.3</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 9 0 )$ </td><td>7.99</td><td>39.4 / 28.1 / 60.1 / 43.4</td><td>42.8</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 7 5 )$ </td><td>8.06</td><td>39.3 / 28.0 / 60.0 / 43.5</td><td>42.7</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 6 0 )$ </td><td>8.04</td><td>39.0 / 28.0 / 60.0 / 43.5</td><td>42.6</td></tr><tr><td rowspan="4">3B</td><td>Base</td><td>6.79</td><td>52.7 / 49.4 / 67.9 / 45.9</td><td>54.0</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 9 0 )$ </td><td>6.83</td><td>52.6 / 49.0 / 67.4 / 44.0</td><td>53.2</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 7 5 )$ </td><td>6.87</td><td>52.5 / 48.7 / 67.4 / 43.9</td><td>53.1</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 6 0 )$ </td><td>6.91</td><td>52.1 / 48.6 / 66.8 / 43.9</td><td>52.9</td></tr><tr><td rowspan="4">7B</td><td>Base</td><td>6.27</td><td>63.4 / 70.6 / 76.0 / 48.4</td><td>64.6</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 9 0 )$ </td><td>7.47</td><td>57.9 / 62.7 / 74.7 / 46.2</td><td>60.4</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 7 5 )$ </td><td>7.49</td><td>57.5 / 62.4 / 74.5 / 45.8</td><td>60.1</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 6 0 )$ </td><td>7.52</td><td>51.8 / 62.6 / 74.6 / 45.6</td><td>58.6</td></tr><tr><td rowspan="4">16B</td><td>Base</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 9 0 )$ </td><td>7.02</td><td>52.6 / 48.3 / 70.5 / 34.8</td><td>51.6</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 7 5 )$ </td><td>7.10</td><td> $5 2 . 5 \ : / \ : 4 6 . 7 \ : / \ : 7 0 . 2 \ : / \ : 3 3 . 6$ </td><td>50.8</td></tr><tr><td> $\mathrm { A P E X } _ { S F } ~ ( \tau = 0 . 6 0 )$ </td><td>7.11 7.13</td><td>52.1 / 46.3 / 70.2 / 33.5  $5 2 . 3 \ : / \ : 4 5 . 7 \ : / \ : 6 6 . 4 \ : / \ : 3 1 . 8$ </td><td>50.5 49.0</td></tr></table>

This behavior is explained by the different routing configuration of Phi models, which use only $k \ = \ 2$ experts per token. In this case, each expert contributes a larger fraction of the final output, making the model more sensitive to mispredictions. As a result, even occasional overlap misses can have a more pronounced effect on accuracy compared to Granite models.

Deployment Guidance: These results show that stall-free execution should be treated as a complementary opt-in mode rather than the default APEX configuration. Correctnesspreserving APEX always preserves the original routing semantics and is therefore preferable for accuracy-critical deployments. Stall-free mode is most appropriate when the application can tolerate small accuracy changes in exchange for additional performance gains. Empirically, we observe that models with larger active expert counts, such as Granite-1B/3B with $\left( k \ = \ 8 \right)$ and DeepSeek-16B with (k = 6) and two shared experts, are more robust because each missed expert contributes a smaller fraction of the final weighted output. In contrast, low-top-k models such as Phi-7B (k = 2) are more sensitive to substitutions; therefore, the correctness-preserving mode is the recommended deployment choice, treating stallfree execution for an optional performance improvement.

N o Prefetch ProMoE Stati c (k+4) Stati c (k+8)  
![](images/99de698335cdc4271c865d2a5f5c3a171bb2f7fb3f3249f12879a46797903327.jpg)

## E. Performance and Energy Analysis

This section evaluates the system-level benefits of APEX in terms of latency, energy, and combined EDP efficiency.

Latency Analysis: Figure 12(a) presents the average pertoken latency across context lengths for the Granite-3B model as a representative case; other models exhibit similar trends, with combined EDP comparisons provided next. At a context length of 512, APEX correctness-preserving mode achieves 11.41 ms average per-token latency. This is 42% lower than no prefetching (19.77 ms) and 26% lower than ProMoE (15.39 ms). Similarly, APEX reduces the average latency by 20% and 40% relative to static (k + 4) and static (k + 8) prefetching techniques. Longer contexts continue to exhibit the same trend. APEX achieves 41% and 39% lower latency than no prefetching at 1024 and 2048 token contexts, outperforming ProMoE by 24% and 20% lower latency.

Figure 12 (a) also show that static overfetching performs poorly. When too many experts are statically prefetched, expert loading itself begins to dominate and can exceed the attention window available for hiding it, turning prefetch traffic into the new bottleneck. Finally, we observe that APEX’s stallfree mode provides 2.0%–2.8% lower latency because it does not stall to bring in missing experts. The latency difference is small since the overlap accuracy is high (i.e., stalls happen rarely).

Energy Consumption Analysis: Figure 12(b) plots the energy consumption results, which reinforce the superiority of APEX over the baselines. At context length 512, APEX correctness-preserving mode consumes 287.3 mJ energy, which is 9.5% lower than no prefetching and 5.8% lower than ProMoE. We observe that static prefetching leads to the largest energy consumption (9.9% and 21.8% more than APEX) due to over-provisioning. Results with longer context lengths confirm that both modes of APEX consistently achieve the lowest energy consumption. The stall-free mode has again a slightly (≈1%) lower energy by avoiding additional (rare) prefetches. This small gap indicates that correctnesspreserving mode already removes nearly all of the exploitable stall overhead, consistent with the very high overlap accuracy reported earlier.

Notably, the static (k + 8) is consistently the most energyconsuming policy, as it attempts to avoid misses more aggressively. This highlights an important point: overfetching does not come for free. Excessive expert transfers increase memory and interconnect activity and can outweigh any stall reduction. To further clarify the energy trade-off, Table V reports an energy breakdown for Granite-3B at 1024-token context. APEX does not reduce transfer energy; off-chip I/O energy increases from 48 mJ to 56 mJ due to additional prefetched experts. However, by overlapping these transfers with attention computation, APEX reduces exposed stall time and lowers idle/leakage energy from 57 mJ to 8 mJ. This reduction more than offsets the added I/O energy, reducing total energy from 340 mJ to 299 mJ. In contrast, static (k+8) prefetching overfetches aggressively, increasing I/O energy to 97 mJ and total energy to 377 mJ. APEX, by comparison, remains close to the Oracle behavior by allocating the additionalprefetch budget needed for each token and layer, which is why it achieves the best combined latency and energy efficiency.

APEX -- Correctness Preserving APEX -- Stall-free  
![](images/407030b8e0937f4ac0a213e07dd89ae3cfb7a227b47ee72c97647825d5bc23d6.jpg)  
Fig. 12. Comparison of (a) average per-token latency and (b) per-token energy consumption across context lengths for Granite-3B model.

End-to-End Efficiency (EDP): Figure 13 shows that APEX consistently achieves the lowest normalized EDP. For Granite-1B, APEX correctness-preserving mode lowers the EDP by 36%–42% over no prefetching, and 16%–27% over ProMoE, across all context lengths. The stall-free mode provides an additional of 4%, 3%, and 2% gain over correctness-preserving at 512, 1024, and 2048 tokens, again indicating that residual misses are already rare. Granite-3B follows the same trend, with APEX achieving the lowest EDP, with 22%–30% lower EDP than ProMoE across all context lengths. For both models, the static (k+8) consistently has the largest EDP, aligned with the latency and energy analysis, as it attempts to avoid misses more aggressively.

Both modes of APEX continue to achieve the best EDP compared to baselines for the Phi-7B model, as shown in the third subplot in Figure 13. Specifically, the correctnesspreserving mode achieves 48%–49% lower EDP than no prefetching, and 24%–28% lower EDP than ProMoE. Unlike the Granite models, the Phi-7B model results in a larger difference between the correctness-preserving and stall-free modes of APEX. It lowers the EDP by 14%, 8%, and 7% compared to correctness-preserving for different context lengths. These results are consistent with Phi’s routing structure and accuracy analysis in Section V-C. Each token activates only $k \ = \ 2$ experts, and missing even one expert has a larger effect than in Granite models, where each token aggregates across k = 8 experts. Thus, eliminating the final residual stalls yields a larger benefit for Phi, at the cost of a larger accuracy degradation.

DeepSeek-16B further confirms the same trend at a larger scale as shown in Figure 13 (D). APEX reduces EDP by 30– 41% over ProMoE, while stall-free execution adds a further 4–6% improvement over correctness-preserving mode. This smaller stall-free gap compared to Phi-7B is consistent with

TABLE V  
ENERGY BREAKDOWN FOR GRANITE-3B AT 1024-TOKEN CONTEXT.
<table><tr><td>Component</td><td>No Prefetch</td><td>APEX</td><td>Static (k + 8)</td></tr><tr><td>On-chip active memory</td><td>233 mJ</td><td> $2 3 4 \ \mathrm { m J }$ </td><td>234 mJ</td></tr><tr><td>Off-chip I/O transfer</td><td>48 mJ</td><td>56 mJ (+8)</td><td>97 mJ</td></tr><tr><td>Idle / leakage</td><td>57 mJ</td><td>8 mJ (−49)</td><td>44 mJ</td></tr><tr><td>Total energy</td><td>340 mJ</td><td>299 mJ</td><td>377 mJ</td></tr></table>

![](images/137bfb024c15450a83e793e14e5294024c9168d5bf5592dc510dd8a1fb0b3e7e.jpg)  
Fig. 13. Normalized EDP all models. For Granite models, $k = 8 , x = 4 ,$ and $y = 8 ;$ for Phi-7B, $k = 2 , x = 1 ,$ , and $y = 2$ and for DeepSeek-16B, $k = 6 , x = 4$ and $y = 8 . \ \mathrm { A P E X }$ variants consistently achieve the lowest EDP across all context lengths, demonstrating superior hardware efficiency.

DeepSeek’s routing structure, where each token uses $k = 6$ routed experts plus two shared experts, making occasional routed-expert substitutions less disruptive.

## F. Overhead Analysis

The additional parameters required by APEX comprise the lightweight prefetch router and the CDF model. For example, in Granite-1B, each layer includes a 1024 × 32 prefetch router and $2 \times 3 2$ CDF parameters, resulting in a total of 0.79M parameters across 24 layers. This is only 0.059% of the 1.3B model weights. The overhead remains similarly negligible for the larger models, accounting for just 0.060%, 0.027% and 0.022% of total parameters for Granite-3B, Phi-7B and DeepSeek-16B, respectively, as shown in Table VI.

On our target edge platform (described in Section V-B), APEX adds only 0.051%, 0.046%, 0.009% and 0.036% performance overhead for Granite-1B, Granite-3B, Phi-7B, and DeepSeek-16B, respectively. Since both the added weights and computation are negligible relative to the base model, this overhead is expected to remain insignificant on other platforms as well.

TABLE VI  
OVERHEAD ANALYSIS OF THE APEX PREFETCH ROUTER.
<table><tr><td>Model</td><td>Additional Parameters</td><td>% of model weights</td><td>% of Performance</td></tr><tr><td>Granite-1B</td><td>0.79M</td><td>0.059%</td><td>0.051%</td></tr><tr><td>Granite-3B</td><td>1.97M</td><td>0.060%</td><td>0.046%</td></tr><tr><td>Phi-7B</td><td>2.10M</td><td>0.027%</td><td>0.009%</td></tr><tr><td>DeepSeek-16B</td><td>34.11M</td><td>0.022%</td><td>0.036%</td></tr></table>

Overall, these results show that APEX is an extremely low-overhead approach. It adds only a lightweight auxiliary predictor on top of the base MoE model, while delivering substantial improvements in latency and energy efficiency.

## G. Ablation Study

Sensitivity to I/O Bandwidth: We evaluate APEX across different off-chip I/O bandwidths using Granite-3B as a representative model at a context length of 1024, while keeping all other parameters fixed to isolate the impact of bandwidth on prefetch effectiveness. The sweep ranges from 32 to 1024 GB/s. The low-bandwidth points (32 and 64 GB/s) represent constrained mobile/embedded-class settings where expert transfers receive only a fraction of total DRAM bandwidth due to sharing with CPU/GPU/NPU activity, KV-cache traffic, and memory-controller contention. The 128 and 256 GB/s points correspond to PCIe $5 . 0 \times 1 6$ and PCIe 6.0 ×16 bandwidths, respectively. As expected, increasing bandwidth reduces latency, even for the no-prefetch baseline, since expert transfers become less expensive at higher bandwidth. Figure 14 shows that APEX consistently decreases the latency by 14%–42% across the entire bandwidth range by minimizing the number of exposed expert-loading stalls. At very low bandwidths, the benefit is limited by the fact that only part of the expert transfer can be hidden behind the attention window, so the unhidden portion remains exposed as stall time. However, APEX still consistently improves latency across the full range by overlapping the transferable portion of expert loading with attention and by avoiding the unnecessary traffic of fixed overfetching. At the same time, APEX remains beneficial even at high-bandwidth settings approaching NVLink-C2C class interconnects, showing that adaptive prefetching continues to improve latency even when I/O is less constrained. Overall, these results highlight that APEX is robust across a wide range of deployment scenarios, with particularly strong benefits in the bandwidth-limited edge setting.

Sensitivity to Expert Weight Precision: Since many edge deployments use quantized weights, we further evaluate APEX across different expert-weight data types on Granite-3B (1024 context). This study scales the expert transfer size and compute cost for 4-, 8-, 16-, and 32-bit expert weights while keeping the routing behavior fixed. Therefore, it isolates the system-level impact of datatype and does not quantify quantization-induced accuracy loss. As shown in Figure 15, reducing precision lowers absolute latency for both No Prefetch and APEX because each expert transfer becomes smaller. However, APEX continues to reduce latency across all datatypes by overlapping expert movement with attention computation. The benefit is especially clear when expert transfers remain exposed on the critical path, while at very low precision, the execution is dominated by attention. These results show that APEX complements quantization. Quantization reduces the amount of data moved [59], whereas APEX reduces the portion of that movement exposed.

![](images/a1670b0e2f63d9245a37f3a5314f9d3dd0de71f5719c4428395eec0e8ae2566e.jpg)  
Fig. 14. Per-token latency versus bidirectional I/O bandwidth for Granite-3B (1024 context length).

![](images/8b08be69bef8e75d576f7e479dee7b9804051f7cdb5594288f6c28bd8ce17391.jpg)  
Fig. 15. Per-token latency sensitivity to expert-weight datatype for Granite-3B (1024 context length).

## VI. CONCLUSION

Edge MoE inference is fundamentally bottlenecked by expert I/O, where irregular and off-chip expert accesses stall execution and waste energy. In this work, we presented APEX, an adaptive expert prefetching framework that transforms expert loading into a predictive, confidence-driven decision. By introducing a lightweight prefetch router and a learned CDF-based mechanism to dynamically select the minimal top- $( k + \hat { \delta } ( x ) )$ prefetch budget, APEX achieves near-oracle expert coverage while avoiding the inefficiencies of static overfetching. This adaptive approach delivers substantial system-level benefits. At higher confidence thresholds, APEX achieves >99% overlap accuracy and effectively eliminates expert-loading stalls while preserving application-level accuracy. Across multiple models and settings, it reduces latency and improves energydelay product (EDP) by up to 41% compared to state-of-theart baselines, with negligible model and runtime overhead. Overall, APEX demonstrates that adaptive, confidence-aware prefetching is key to unlocking efficient MoE inference on edge systems, bridging the gap between predictive accuracy and system performance.

Disclosure: Dr. Ogras is affiliated with Samsung Austin Research & Development Center and Advanced Computing Lab (SARC/ACL). This relationship has been approved under applicable outside activities policies.

## REFERENCES

[1] T. Meuser et al., “Revisiting edge AI: Opportunities and challenges,” IEEE Internet Computing, vol. 28, no. 4, pp. 49–59, 2024.

[2] Y. Zheng et al., “A review on edge large language models: Design, execution, and applications,” ACM Computing Survey, vol. 57, no. 8, pp. 1–35, 2025.

[3] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” J. Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[4] A. Q. Jiang et al., “Mixtral of experts,” 2024.

[5] A. Gholami et al., “AI and the memory wall,” IEEE Micro, vol. 44, no. 3, pp. 33–39, 2024.

[6] D. Lepikhin et al., “GShard: Scaling giant models with conditional computation and automatic sharding,” in Proc. Int. Conf. Learning Representations (ICLR), 2021.

[7] R. Pope et al., “Efficiently scaling transformer inference,” in Proc. Maching Learning System (MLSys), vol. 5, 2023.

[8] J. Park et al., “Thermal modeling and management challenges in heterogeneous integration: 2.5D chiplet platforms and beyond,” in Proc. IEEE VLSI Test Symp. (VTS), 2024, pp. 1–4.

[9] L. Pfromm et al., “MFIT: Multi-Fidelity Thermal Modeling for 2.5d and 3d Multi-Chiplet Architectures,” ACM Transactions on Design Automation of Electronic Systems, 2025.

[10] NVIDIA Corporation, “GeForce RTX 3090,” 2020, accessed 2026-03-11. [Online]. Available: https://www.nvidia.com/en-us/geforce/ graphics-cards/30-series/rtx-3090/

[11] IBM Research, “Granite-3.1-3B-A800M Base: A sparse mixtureof-experts language model,” 2024, model card, accessed 2026-03- 27. [Online]. Available: https://huggingface.co/ibm-granite/granite-3. 1-3b-a800m-base

[12] A. Kanani et al., “THERMOS: Thermally-aware Multi-Objective Scheduling of AI Workloads on Heterogeneous Multi-Chiplet PIM Architectures,” ACM Transactions on Embedded Computing Systems, vol. 24, no. 5s, pp. 1–26, 2025.

[13] X. Song, Z. Zhong, R. Chen, and H. Chen, “ProMoE: Fast MoE-based LLM serving using proactive caching,” 2024.

[14] S. Zhong et al., “AdapMoE: Adaptive sensitivity-based expert gating and management for efficient MoE inference,” in Proc. IEEE/ACM Int. Conf. Computer-Aided Design (ICCAD), 2024, pp. 1–9.

[15] P. Tang et al., “HOBBIT: A mixed precision expert offloading system for fast MoE inference,” 2024.

[16] R. Hwang et al., “Pre-gated MoE: An algorithm-system co-design for fast and scalable mixture-of-expert inference,” in Proc. Annu. Int. Symp. Computer Architecture (ISCA), 2024, pp. 1018–1031.

[17] L. Xue et al., “MoE-Infinity: Efficient MoE inference on personal machines with sparsity-aware expert cache,” 2024.

[18] Z. Fang et al., “FATE: Fast edge inference of mixture-of-experts models via cross-layer gate,” 2025.

[19] S. Zhu, S. Bohl, R. Oester, and G. Alonso, “Pre-attention expert prediction and prefetching for mixture-of-experts large language models,” arXiv preprint arXiv:2511.10676, 2025.

[20] IBM Research, “Granite-3.1-1B-A400M Base: A sparse mixtureof-experts language model,” 2024, model card, accessed 2026-03- 27. [Online]. Available: https://huggingface.co/ibm-granite/granite-3. 1-1b-a400m-base

[21] M. Abdin et al., “Phi-3 technical report: A highly capable language model locally on your phone,” 2024.

[22] A. Liu et al., “Deepseek-v2: A strong, economical, and efficient mixtureof-experts language model,” arXiv preprint arXiv:2405.04434, 2024.

[23] N. Shazeer et al., “Outrageously large neural networks: The sparselygated mixture-of-experts layer,” in Proc. Int. Conf. Learning Representations (ICLR), 2017.

[24] B. Zoph et al., “ST-MoE: Designing stable and transferable sparse expert models,” 2022.

[25] M. Lewis, S. Bhosale, T. Dettmers, N. Goyal, and L. Zettlemoyer, “BASE layers: Simplifying training of large, sparse models,” in Proc. Int. Conf. Machine Learning (ICML), 2021, pp. 6265–6274.

[26] Y. Zou et al., “FED-MOE: Efficient federated learning for mixtureof-experts models via empirical pruning,” in Proc. Int. Conf. Parallel Distributed Computing: Applications and technologies (PDCAT), 2024, pp. 128–139.

[27] A. Cheng et al., “ERMoE: Eigen-reparameterized mixture-of-experts for stable routing and interpretable specialization,” 2025.

[28] D. Dai et al., “DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models,” in Proc. Annu. Meet. Assoc. Computational Linguistics (ACL), 2024, pp. 1280–1297.

[29] N. Muennighoff et al., “OLMoE: Open mixture-of-experts language models,” 2024.

[30] S. He et al., “Hydra: Harnessing expert popularity for efficient mixtureof-expert inference on chiplet systems,” in Proc. ACM/IEEE Design Automation Conf. (DAC), 2025, pp. 1–7.

[31] J. Dong et al., “UbiMoE: A ubiquitous mixture-of-experts vision transformer accelerator with hybrid computation pattern on FPGA,” in Proc. IEEE Int. Symp. Circuits Systems (ISCAS), 2025, pp. 1–5.

[32] J. He et al., “FastMoE: A fast mixture-of-expert training system,” 2021.

[33] R. Y. Aminabadi et al., “DeepSpeed-Inference: Enabling efficient inference of transformer models at unprecedented scale,” in Proc. Int. Conf. Performance Computing, Networking, Storage and Analysis (SC), 2022.

[34] T. Gale, D. Narayanan, C. Young, and M. Zaharia, “MegaBlocks: Efficient sparse training with mixture-of-experts,” in Proc. Machine Learning and Systems (MLSys), vol. 5, 2023.

[35] S. Li, J. Lin, D. Ge, and Y. Ye, “MoE-SpAc: Efficient MoE inference based on speculative activation utility in heterogeneous edge scenarios,” 2026.

[36] A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Information Processing System (NeurIPS), vol. 30, 2017, pp. 5998–6008.

[37] S. Merity, C. Xiong, J. Bradbury, and R. Socher, “Pointer sentinel mixture models,” 2016, introduces the WikiText language modeling datasets.

[38] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. MIT Press, 2016.

[39] P. Clark et al., “Think you have solved question answering? try ARC, the AI2 reasoning challenge,” 2018.

[40] D. Hendrycks et al., “Measuring massive multitask language understanding,” 2020.

[41] K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi, “WinoGrande: An adversarial winograd schema challenge at scale,” Commun. ACM, vol. 64, no. 9, pp. 99–106, 2021.

[42] S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring how models mimic human falsehoods,” in Proc. Annu. Meet. Assoc. computational linguistics (ACL), 2022, pp. 3214–3252.

[43] NVIDIA Corporation, “NVIDIA A100 Tensor Core GPU Architecture,” 2020, accessed 2026-06-01. [Online]. Available: https://www.nvidia. com/en-us/data-center/a100/

[44] A. Eliseev and D. Mazur, “Fast inference of mixture-of-experts language models with offloading,” 2023.

[45] G. Shan, Y. Zheng, C. Xing, D. Chen, G. Li, and Y. Yang, “Architecture of computing system based on chiplet,” Micromachines, vol. 13, no. 2, p. 205, 2022.

[46] A. Kanani et al., “DUET: Disaggregated Hybrid Mamba-Transformer LLMs with Prefill and Decode-Specific Packages,” in Proc. ACM/IEEE Design Automation Conf. (DAC), 2026, pp. 1–7.

[47] E. Reggiani, R. Andri, and L. Cavigelli, “Flex-SFU: Accelerating DNN activation functions by non-uniform piecewise approximation,” in Proc. ACM/IEEE Design Automation Conference (DAC), 2023, pp. 1–6.

[48] W. Kwon et al., “Efficient memory management for large language model serving with PagedAttention,” in Proc. ACM Symp. operating systems principles (SOSP), 2023, pp. 611–626.

[49] PCI-SIG, “PCI Express Base Specification Revision 6.0,” 2022, PCI-SIG specification. [Online]. Available: https://pcisig.com/specifications/ pciexpress/base6

[50] M.-J. Park et al., “A 192-gb 12-high 896-gb/s HBM3 DRAM with a TSV auto-calibration scheme and machine-learning-based layout optimization,” IEEE J. Solid-State Circuits, vol. 58, no. 1, pp. 256–269, 2023.

[51] NVIDIA, “Jetson Thor: Advanced AI for Physical Robotics,” https: //www.nvidia.com/en-us/autonomous-machines/embedded-systems/ jetson-thor/, 2026, accessed: 2026-06-14.

[52] Hailo, “Hailo-10H AI Accelerator: LLM and VLM Generative AI Accelerator,” https://hailo.ai/products/ai-accelerators/ hailo-10h-ai-accelerator/, 2024, accessed: 2026-06-14.

[53] Synopsys Inc., “Synopsys PrimeTime,” 2023, static timing and power analysis tool. [Online]. Available: https://www.synopsys.com/ implementation-and-signoff/signoff/primetime.html

[54] H. Luo et al., “Ramulator 2.0: A modern, modular, and extensible DRAM simulator,” IEEE Computer Architecture Letters, vol. 23, no. 1, pp. 112–116, 2024.

[55] S. Ghose et al., “Understanding DRAM power consumption: Experimental characterization and analysis,” in Proc. ACM SIGMETRICS, 2018.

[56] D. D. Sharma, G. Pasdast, Z. Qian, and K. Aygun, “Universal chiplet interconnect express (UCIe): An open industry standard for innovations with chiplets at package level,” IEEE Trans. Components, Packaging and Manufacturing Technology, vol. 12, no. 9, pp. 1423–1431, 2022.

[57] L. Pfromm et al., “CHIPSIM: A co-simulation framework for deep learning on chiplet-based systems,” IEEE Open J. Solid-State Circuits Society, vol. 5, pp. 410–423, 2025.

[58] R. Balasubramonian, A. B. Kahng, N. Muralimanohar, A. Shafiee, and V. Srinivas, “CACTI 7: New tools for interconnect exploration in innovative off-chip memories,” ACM Trans. Architecture and Code Optimization (TACO), vol. 14, no. 2, pp. 1–25, 2017.

[59] M. Sun, A. Kanani, K. Shroff, and U. Ogras, “LEXI: Lossless Exponent Coding for Efficient Inter-Chiplet Communication in Hybrid LLMs,” in Proc. ACM/IEEE Design Automation Conf. (DAC), 2026, pp. 1–7.
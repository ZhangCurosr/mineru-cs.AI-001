# ImpactHO: Importance-Aware KV Cache Transfer for Multi-User Edge LLM Handover

Minwoo Kim, Soochang Song, Namyoon Lee, Bang Chul Jung, and Yongjune Kim

Abstract—Edge LLMs must preserve inference continuity when a user hands over between edge nodes, requiring keyvalue (KV) cache transfer to the target node. However, simultaneous handovers saturate the backhaul, preventing full cache delivery within the mobility-imposed transfer window. Rather than allocating bandwidth as if all cache entries were equally valuable, we order each user’s KV cache by importance and transmit only its most informative fraction, turning token-level sparsity into communication savings. We cast the transfer as a multi-user backhaul allocation problem that maximizes average accuracy across users. Each user’s partial-cache accuracy serves as its utility: a sigmoid that fits measurements on the RULER benchmark with R<sup>2</sup> > 0.99 across models and context lengths. Because importance ordering front-loads the high-value entries, the concave region of the accuracy curve spans nearly the entire cache. Our proposed allocator keeps served users within this region, making each per-slot allocation problem convex. The optimum is derived via a weighted water-filling solution that generalizes information-theoretic water-filling and enables online scheduling. The proposed allocator attains over 93.7% average accuracy in a 500ms transfer window, within 0.5pp of the fullcache ceiling, and reaches 98.2–99.5% of a clairvoyant upper bound.

Index Terms—Edge LLM, KV cache transfer, resource allocation, token communications, multi-access edge computing.

## I. INTRODUCTION

Next-generation communication networks are evolving beyond simple data delivery into an infrastructure that supports real-time artificial intelligence (AI) inference for applications such as large language model (LLM) services, autonomous driving, and robotics [1], [2]. These services share two requirements: ultra-low latency and seamless mobility. The first is difficult to meet with remote cloud inference, whose roundtrip time alone can exceed the millisecond-scale latency budget of real-time AI [3]. Multi-access edge computing (MEC) addresses this by hosting computation at the network edge, close to users. For AI workloads, this paradigm has taken shape as Edge AI and, more recently, Edge LLM, in which model inference runs directly on edge nodes [4], [5].

However, unlike cloud-based serving, edge inference is directly exposed to user mobility. Whenever a moving user is reassigned to another edge node, inference continuity must be preserved across the handover. Although virtual machine or container migration has been studied for MEC handover [6]– [8], modern AI models carry tens to hundreds of gigabytes of weights (e.g., 16 GB for 8B-parameter models), making such migration impractical and pushing handover latency to tens of seconds.

![](images/6fd922c1b6dc801a73acb77133a250ad86eb016cee734bb9cd57a0e38e7eb03d.jpg)  
Fig. 1. Concept of ImpactHO framework.

Since the model itself is already provisioned at each edge node, only the user-specific context needs to move. Transformer-based LLMs store this context as a key-value (KV) cache, and recent work proposes treating the KV cache itself as the transfer target [9]. Unlike transferring the raw text or tokens, which requires re-prefilling at the target node, transferring the KV cache lets the target node resume inference immediately. However, the KV cache is substantially larger than the raw text it encodes, so concurrent long-context handovers still saturate the backhaul.

A natural remedy is to transfer only part of the KV cache. Recent studies on KV cache eviction have shown that not all entries are equally important at inference time: LLMs can retain most of their accuracy using only a small fraction of their cache [10]–[14]. These results, however, address a single model’s memory budget, not how a shared backhaul should be divided across users.

To address this gap, we propose ImpactHO (Importanceaware KV cache transfer for multi-user handover), a framework that orders each user’s cache by importance and transmits its most informative entries first, as illustrated in Fig. 1. To capture how accuracy grows with the delivered cache, we model each user’s accuracy as a function of its received cache fraction and treat this curve as a per-user utility. The scheduling task then becomes a utility-maximization problem: allocate the limited backhaul bandwidth across users so as to maximize the average accuracy.

Importance ordering plays a structural role beyond prioritizing informative cache. Partial-cache accuracy is intrinsically concave once enough context has arrived for the model to become operable. By front-loading the highest-value entries, importance ordering pulls this operable point (i.e., the inflection point of the accuracy curve) down to only a small cache fraction. The concave region therefore spans nearly the entire cache, and every user can be brought into it at low cost. An admission rule then lifts every served user past this point, so each per-slot allocation reduces to a convex subproblem, whose optimum takes a weighted water-filling form with each user’s cache size acting as its weight.

We summarize our main contributions as follows:

• ImpactHO framework: We formulate importance-ordered partial KV cache transfer as a multi-user backhaul allocation problem for edge LLM handover, repurposing per-entry importance scores from KV cache eviction to set the transmission order. Modeling each user’s partialcache accuracy as its utility casts the scheduling task as a utility-maximization problem over the shared bandwidth.

• Empirical sigmoid characterization of partial-cache accuracy: On the RULER benchmark [15], partial-cache accuracy follows a sigmoid $( R ^ { 2 } > 0 . 9 9 )$ robustly across context lengths, models, and ordering schemes. Importance ordering pulls its inflection point down to about 6.5 % of the cache, so the concave region spans nearly the entire cache. This validates our utility assumption and provides a reusable parametric foundation for KV cacheaware networking research.

• Two-regime allocator with low overhead: We derive the per-slot optimum in closed form as a weighted waterfilling solution over a feasible region expanded by importance ordering. It reduces to classical water-filling as a special case and runs in near-linear time per slot. Coupled with an admission rule that suppresses starvation under heavy load, the allocator consistently outperforms baselines and reaches near-full accuracy faster than target-side re-prefill of even a single 8K context.

## II. RELATED WORK

The KV cache stores the key and value tensors of previously processed tokens at each Transformer layer and attention head, avoiding their recomputation during autoregressive decoding. However, because its memory footprint grows linearly with the context length, the KV cache has become a major bottleneck for long-context inference. To alleviate this bottleneck, eviction-based methods [10]–[14] estimate the importance of cached entries and retain only the most important ones, thereby reducing memory consumption while largely preserving model accuracy.

Among these, KVzip [13] derives query-agnostic per-(layer, head, token) importance scores from a context-reconstruction pass, reaching near-lossless accuracy with less than 30 % of the cache. Fast KVzip [14] distills these scores into lightweight gating modules offline, achieving comparable accuracy without the inference-time scoring overhead. While prior work uses these importance scores only for memory reduction, we repurpose them to set the transmission order over the backhaul. Since the future query is unavailable at handover time, queryagnostic importance estimation is particularly well suited to our setting; hence, we adopt Fast KVzip. Nevertheless, our framework can accommodate any scoring method that provides a per-entry importance ranking.

Beyond the memory pressure addressed by eviction, moving the cache between nodes is itself a bottleneck. In disaggregated datacenter serving, DistServe [16] and Splitwise [17] place prefill and decode on separate instances, requiring the KV cache to be transferred between them. However, these systems are designed for datacenter environments with high-bandwidth interconnects and do not address mobility-driven handovers or contention among multiple migrating users over bandwidthconstrained backhaul links. More closely related to our setting, CacheGen [18] streams a compressed KV cache adaptively, but focuses on loading a single context rather than allocating shared backhaul bandwidth across concurrent handovers. Such KV compression is complementary to our allocation framework: it can be applied before transmission to further reduce the required backhaul traffic.

While sharing our motivation to preserve inference continuity under mobility, a recent edge LLM handover scheme ctHO [9] adopts a fundamentally different formulation. It minimizes the maximum handover delay across users, couples resource allocation with target-side computation, and requires the handover timing of all users to be known in advance. In contrast, we transmit only a selected fraction of each KV cache and maximize inference accuracy subject to an online per-slot backhaul budget, thereby decoupling backhaul allocation from target-side compute. ImpactHO supports anytime inference because the target can resume decoding after any completed prefix of the KV cache stream using the partial cache received up to that point. In contrast, re-prefill must complete prefill over the full context, while ctHO must complete its prescribed cache-transfer and target-side re-prefill operations before decoding can resume. Hence, neither baseline supports inference before full context restoration. Table I summarizes these structural differences, and Section VI-D provides a quantitative comparison.

Our importance-ordered KV cache transfer shares the principle of task-oriented and semantic communications, which prioritize information according to its relevance to the given task rather than bit-level fidelity [19], [20]. A recent instantiation of this principle is token communications, in which tokenized multimodal source signals are transmitted and reconstructed at the receiver [21], [22]. Our setting differs in what is transmitted: not source tokens, but the per-token KV cache entries constituting the internal inference state of a deployed LLM. The target node consumes these entries directly to resume inference, without reconstructing the original source. By sending only a high-value fraction in descending order of measured importance, we extend this paradigm to mainstream LLM serving.

TABLE I  
STRUCTURAL COMPARISON OF HANDOVER STRATEGIES.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Re-prefill</td><td rowspan=1 colspan=1>ctHO [9]</td><td rowspan=1 colspan=1>ImpactHO</td></tr><tr><td rowspan=1 colspan=1>Target-GPU prefill</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>Partial</td><td rowspan=1 colspan=1>None</td></tr><tr><td rowspan=1 colspan=1>Scheduling</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>Offline</td><td rowspan=1 colspan=1>Online</td></tr><tr><td rowspan=1 colspan=1>Anytime property</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>Yes</td></tr></table>

Our framework also relates to network utility maximization (NUM), which shares a fixed capacity by maximizing the sum of per-user utilities. For elastic flows with concave utilities, NUM admits globally optimal distributed solutions [23]. Realtime inelastic flows are a closer match to our setting: their utilities are sigmoidal, and this shape makes the problem nonconvex, so standard dual algorithms can miss a feasible global optimum. Prior work has addressed this regime in two ways: sub-threshold flows can self-regulate, turning off when a persistently low net utility signals infeasibility [24]; alternatively, link capacity can be provisioned large enough to guarantee convergence of the distributed algorithm [25].

In both approaches, the application determines a fixed utility curve, and the network can only select an operating point along that curve. In our setting, by contrast, the utility is empirically measured and depends not only on the user’s context length but also on the order in which KV cache entries are transmitted. Our approach is therefore to shape the utility itself through importance ordering, so that the concave regime covers nearly the entire operating range.

## III. SYSTEM MODEL AND PROBLEM FORMULATION

## A. Framework Overview

The overall concept of ImpactHO is illustrated in Fig. 1. While serving a user, the source node assigns an importance score to each KV cache entry and maintains the entries in descending order of importance. Consequently, any prefix of the ordered cache contains the highest-ranked entries for the corresponding cache size. When a handover occurs, the source node streams these entries to the target node in importance order over the shared backhaul. If the finite transfer window expires before the entire cache is delivered, this ordering ensures that the target has received the most valuable subset available under the transfer budget and can resume inference using the resulting partial cache. When multiple handovers concurrently contend for the shared backhaul, ImpactHO allocates the per-slot transmission budget among users to maximize their aggregate inference accuracy. The remainder of this section formalizes the KV cache size and importanceinduced accuracy utility and formulates the resulting per-slot resource-allocation problem.

## B. Multi-user Edge LLM Handover

We consider an edge node hosting a shared LLM that concurrently serves multiple users, reflecting a typical edge deployment in which GPU memory constraints favor sharing a single model instance across users. Each user i maintains an individual context stored at the edge node as a KV cache of size

$$
L _ { i } = 2 n _ { L } n _ { H } d _ { h } q T _ { i } \quad \mathrm { b i t s } ,\tag{1}
$$

where the factor of two accounts for the key and value tensors, $n _ { L }$ is the number of transformer layers, $n _ { H }$ is the number of KV heads, $d _ { h }$ is the head dimension, q is the number of bits used to represent each scalar, and $T _ { i }$ is the context length. $L _ { i }$ depends on the transformer architecture and grows linearly with the context length $T _ { i }$ . For example, the KV cache of Qwen3-8B [26] occupies approximately 1.2 GB (9.66 Gb) for an 8K-token context. Once a handover is triggered, we assume that $L _ { i }$ remains fixed throughout the KV cache transfer.

An edge inference node may serve users across multiple radio cells; hence, inference-state migration is required only when a user moves beyond the coverage of its current node. To preserve inference continuity, the source node transfers the user’s KV cache to the target node. In AI-RAN architectures for beyond-5G networks, edge inference nodes may be deployed at the distributed unit (DU) or central unit (CU) level and connected through fiber-based transport networks using interfaces such as F1 or X2/Xn [27]. For brevity, we collectively refer to the transport path between the source and target nodes as the backhaul. KV cache transfers share a backhaul bandwidth of B bits/s among the active handover users, where B denotes the bandwidth allocated to KV cache transfer, not the raw physical link capacity.

We model backhaul allocation in discrete time slots of duration ∆t. At the beginning of each slot, the scheduler observes the set of active handover users and determines their allocations, which remain fixed throughout the slot. A handover request arriving during a slot becomes eligible for scheduling at the beginning of the next slot. We focus on the given slot in which N users concurrently undergo handover and compete for the available backhaul bandwidth B, i.e., perslot budget of $B \Delta t$ bits. We denote by $\mathcal { N } _ { t }$ the index set of these active handover users, so that $| \mathcal { N } _ { t } | = N$ . We consider a single transfer direction, with B the bandwidth provisioned for that direction; the N contending users are those handing over in it, and the reverse direction forms a symmetric instance.

## C. Importance-aware KV Cache Ordering

We assign an importance score to each individual KV cache entry, defined as the key–value pair associated with one token at a specific layer and KV head, rather than to an entire token. Each entry occupies $2 d _ { h } q$ bits (0.5 KB for Qwen3-8B in BF16), and user $i \ ' s$ cache therefore contains $n _ { L } n _ { H } T _ { i }$ entries. The source node orders these entries by decreasing importance and transmits them in that order. Consequently, any delivered prefix is the top-ranked subset of its size: when a fraction $x _ { i } \in [ 0 , 1 ]$ of user i’s cache has arrived, the target node holds the highest-ranked $x _ { i } L _ { i }$ bits. $\operatorname { A s } \ x _ { i }$ increases, newly transmitted entries have progressively lower importance, naturally inducing diminishing returns. This observation motivates the concave accuracy-utility model introduced in Section III-D.

In this paper, we obtain the importance ordering from the gating-network scores of Fast KVzip [14], chosen for its nearstate-of-the-art retention quality at low computational cost. Importance-ordered transmission does introduce implementation overhead beyond standard cache transfer, which we quantify and discuss in Section VI-E. Our optimization framework is not restricted to Fast KVzip or even to importance-based ordering: any fixed ordering, including random ordering, can be accommodated as long as its induced utility is monotonically increasing and concave over the operating region considered; Section IV examines this condition in detail.

## D. Utility Function

We define the per-user utility $A _ { i } ( y )$ as the inference accuracy for user i when the target node has received a fraction $y \in [ 0 , 1 ]$ of the user’s importance-ordered KV cache. Once the target node has received a sufficient fraction, increasing y adds progressively lower-ranked entries to those already delivered, so accuracy is expected to improve with diminishing marginal gains. This motivates the following assumption, on which the optimal-allocation analysis of Section V relies.

Assumption 1 (Concave Operating Region): For each user i, there exists a concavity anchor $\tau _ { i } \in ( 0 , 1 )$ such that $A _ { i } ( y )$ is continuously differentiable, strictly increasing, and strictly concave on [τ<sub>i</sub>, 1]. We restrict the analysis to $y \in [ \tau _ { i } , 1 ]$

Operationally, $\tau _ { i }$ denotes the minimum cache fraction at which the target enters the concave operating region considered for resource allocation. Below $\tau _ { i } .$ , the received KV cache entries may be insufficient for reliable task performance, and $A _ { i }$ need not be concave. Above $\tau _ { i } ,$ transmitting additional, progressively lower-ranked entries yields diminishing accuracy gains. Section IV shows that the measured accuracy curves exhibit operating regions consistent with Assumption 1.

## E. Optimization Problem

At the beginning of a slot, the target node has received a fraction $x _ { i }$ of user i’s KV cache. The scheduler assigns user i a normalized cache-delivery rate $b _ { i } .$ , in cache fractions per second, so that the received fraction at the end of the slot is

$$
y _ { i } \triangleq x _ { i } + b _ { i } \Delta t .\tag{2}
$$

Because the full cache contains $L _ { i }$ bits, this allocation corresponds to a transmission rate of $L _ { i } b _ { i }$ bits/s.

We maximize the aggregate accuracy improvement achieved during the slot:

$$
\begin{array} { r l } { \underset { \{ b _ { i } \} } { \mathrm { m a x i m i z e } } } & { \displaystyle \sum _ { i = 1 } ^ { N } \bigl [ A _ { i } ( y _ { i } ) - A _ { i } ( x _ { i } ) \bigr ] } \\ { \mathrm { s u b j e c t ~ t o } } & { \displaystyle \sum _ { i = 1 } ^ { N } L _ { i } b _ { i } \leq B , } \\ & { b _ { i } \geq 0 , \quad \tau _ { i } \leq y _ { i } \leq 1 , \quad \forall i \in \{ 1 , \ldots , N \} . } \end{array}\tag{3}
$$

The first constraint enforces that the aggregate transmission rate does not exceed the available backhaul bandwidth B. The remaining constraints prevent the received cache fraction from decreasing, restrict each user to the concave operating region specified in Assumption 1, and cap the received fraction at the full cache. Because $b _ { i } \geq 0$ is equivalent to $y _ { i } \geq x _ { i }$ , the two lower bounds on $y _ { i }$ can be combined as $\tilde { x } _ { i } \ \triangleq \operatorname* { m a x } \{ x _ { i } , \tau _ { i } \}$ yielding the individual feasible interval $y _ { i } \in [ \tilde { x } _ { i } , 1 ]$

Moreover, $x _ { i }$ is fixed at the beginning of the slot, so $\textstyle \sum _ { i } A _ { i } ( x _ { i } )$ is constant and can be omitted from the objective. PUsing $b _ { i } = ( y _ { i } - x _ { i } ) / \Delta t$ , Problem (3) is equivalent to

$$
\begin{array} { r l } { \underset { \{ y _ { i } \} } { \mathrm { m i n i m i z e } } } & { - \displaystyle \sum _ { i = 1 } ^ { N } A _ { i } ( y _ { i } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { \displaystyle \sum _ { i = 1 } ^ { N } \frac { L _ { i } \left( y _ { i } - x _ { i } \right) } { \Delta t } \leq B , } \\ & { \displaystyle \tilde { x } _ { i } \leq y _ { i } \leq 1 , \quad \forall i \in \{ 1 , \ldots , N \} . } \end{array}\tag{4}
$$

The objective over the feasible set is convex and all constraints are affine. We solve this convex optimization problem in Section V.

The formulation requires every active user to reach its concave operating region by the end of the slot. This is possible only when the available backhaul budget is sufficient to raise every user currently below its concavity anchor τ.

Lemma 1 (Feasibility): Problem (4) is feasible if and only if

$$
B \ \geq \ B _ { \operatorname* { m i n } } \ \triangleq \ { \frac { 1 } { \Delta t } } \sum _ { i = 1 } ^ { N } L _ { i } \left[ \tau _ { i } - x _ { i } \right] ^ { + } ,\tag{5}
$$

where $[ \cdot ] ^ { + } \triangleq \operatorname* { m a x } \{ \cdot , 0 \}$

Proof: Any feasible $\{ y _ { i } \}$ satisfies $y _ { i } \geq \tilde { x } _ { i }$ , hence $y _ { i } - x _ { i } \geq$ max $\{ x _ { i } , \tau _ { i } \} - x _ { i } = [ \tau _ { i } - x _ { i } ] ^ { + }$ . It must therefore transmit at least $\begin{array} { r } { \sum _ { i } L _ { i } [ \tau _ { i } - x _ { i } ] ^ { + } } \end{array}$ bits during the slot, proving the necessity Pof (5). Conversely, if (5) holds, choosing $y _ { i } = \tilde { x } _ { i }$ for every user satisfies both the individual bounds and the backhaul constraint.

When $B \ < \ B _ { \mathrm { m i n } } ,$ not all active users can reach their concave operating regions within the slot, and we instead invoke the fallback policy described in Section V.

## IV. UTILITY FUNCTION CHARACTERIZATION

Recall from Section III-D that $A _ { i } ( y )$ denotes the inference accuracy for user i as a function of its received cache fraction $y ,$ and that Assumption 1 posits a concavity anchor $\tau _ { i }$ beyond which $A _ { i } ( y )$ is concave. We hypothesize that $A _ { i } ( y )$ exhibits a sigmoidal shape: accuracy remains low when the received cache is insufficient, increases rapidly once enough informative entries have been delivered, and eventually saturates because subsequently delivered entries are progressively lower ranked and provide smaller marginal gains. This section evaluates this hypothesis through systematic measurements. In addition to supporting the analysis in Section V, the resulting characterization provides a compact parametric utility model for importance-ordered KV cache transfer.

![](images/b892bfb10cf49694b7f21e83183aafdcc3069fe193c0a5be9bf52fd3d522aedf.jpg)

![](images/59568127943f17848575079bc1e290f37b81577a134cd2541e4c4a6f530b9b49.jpg)  
(a) Fast KVzip, 8K context  
(b) Random ordering, 8K context  
Fig. 2. Algebraic sigmoid fits to the measured RULER accuracy as a function of the received KV cache fraction at 8K context length for Qwen3-8B, Qwen3- 14B, and Llama-3.1-8B-Instruct: (a) Fast KVzip and (b) random ordering. Markers indicate measured RULER accuracy, and solid curves show the algebraic sigmoid fits defined in (6).

1) Empirical Setup: We test this hypothesis by fitting four canonical sigmoid families—algebraic, logistic, error function (erf), and arctan—to measured LLM accuracy across multiple context lengths. Although prior KV cache eviction studies have reported the qualitative dependence of LLM accuracy on the retained cache fraction [11], [14], our objective is to identify a parametric model that both accurately represents the measured utility and permits tractable resource-allocation analysis. We use Qwen3-8B [26] as the primary LLM, and additionally evaluate Qwen3-14B and Llama-3.1-8B-Instruct [28] to assess robustness across model architectures and sizes. As the benchmark, we use RULER [15], a long-context evaluation suite comprising diverse tasks at multiple context lengths. We compare the importance ordering produced by Fast KVzip with random ordering as a baseline.

2) Fitting Results: Fig. 2 shows that the algebraic sigmoid fits well for Qwen3-8B at 8K $( R ^ { 2 } > 0 . 9 9 9 , \mathrm { R M S E } = 0 . 9 3 6 )$ The algebraic sigmoid remains accurate across the 4K, 8K, and 16K context lengths, achieving $R ^ { 2 } ~ > ~ 0 . 9 9 9$ in every case. Thus, the observed sigmoidal behavior is not specific to a single context length. The remaining three families fit the same measurements equally well, so the sigmoidal shape is a property of the measured accuracy. The fitted inflection points are stable across the four functional families, lying within the narrow range $\tau \in [ 0 . 0 6 4 , 0 . 0 6 7 ]$ . Moreover, the algebraic sigmoid achieves $R ^ { 2 } > 0 . 9 9$ for both Qwen3-14B and Llama-3.1-8B-Instruct under both Fast KVzip and random ordering. Collectively, these results support the existence of a concave operating region consistent with Assumption 1 across the considered models, context lengths, and transmission orderings.

3) Choice of Functional Form: A practical utility model should provide both an accurate empirical fit and a tractable marginal-utility inversion. Although all four fitted families yield optimal solutions, we adopt the algebraic sigmoid because its marginal-utility inverse involves only arithmetic

operations and radicals:

$$
A _ { i } ( y ) = \frac { M _ { i } } { 2 } \Bigg ( 1 + \frac { k _ { i } ( y - \tau _ { i } ) } { \sqrt { 1 + k _ { i } ^ { 2 } ( y - \tau _ { i } ) ^ { 2 } } } \Bigg ) ,\tag{6}
$$

$$
A _ { i } ^ { \prime } ( y ) = \frac { M _ { i } k _ { i } } { 2 \left( 1 + k _ { i } ^ { 2 } ( y - \tau _ { i } ) ^ { 2 } \right) ^ { 3 / 2 } } .\tag{7}
$$

The parameters $M _ { i } , k _ { i } ,$ and $\tau _ { i }$ represent the upper accuracy asymptote, transition sharpness, and concavity anchor, respectively. These parameters may vary across users because of differences in context length and task characteristics.

## V. IMPORTANCE-AWARE RESOURCE ALLOCATION VIA WEIGHTED WATER-FILLING

## A. Optimal Weighted Water-Filling Structure

The solution to Problem (4) exhibits a weighted water-filling structure, analogous to generalized water-filling solutions for constrained resource allocation [29]. A single dual price coordinates all users, but their water levels differ according to their cache sizes and marginal-utility curves. Each user’s allocation fills the positive gap between its current progress $x _ { i }$ and its price-dependent water level.

Theorem 1 (Weighted Water-Filling): Under Assumption 1, if $B \geq B _ { \operatorname* { m i n } }$ , the primal optimum is

$$
y _ { i } ^ { \star } = \operatorname* { m a x } \{ x _ { i } , W _ { i } ( \lambda ^ { \star } ) \} , \qquad b _ { i } ^ { \star } = \frac { y _ { i } ^ { \star } - x _ { i } } { \Delta t } ,\tag{8}
$$

where

$$
W _ { i } ( \lambda ) \triangleq \left\{ \begin{array} { l l } { 1 , } & { s _ { i } ( \lambda ) \leq A _ { i } ^ { \prime } ( 1 ) , } \\ { ( A _ { i } ^ { \prime } ) ^ { - 1 } \big ( s _ { i } ( \lambda ) \big ) , } & { A _ { i } ^ { \prime } ( 1 ) < s _ { i } ( \lambda ) < A _ { i } ^ { \prime } ( \tau _ { i } ) , } \\ { \tau _ { i } , } & { s _ { i } ( \lambda ) \geq A _ { i } ^ { \prime } ( \tau _ { i } ) . } \end{array} \right.\tag{9}
$$

Here, we set $s _ { i } ( \lambda ) \triangleq \lambda L _ { i } / \Delta t .$ , and ${ { \lambda } ^ { \star } } \geq 0$ is a dual-optimal variable associated with the backhaul-rate constraint. When the budget binds, $\lambda ^ { \star }$ can be chosen to satisfy

$$
\sum _ { i } \frac { L _ { i } ( y _ { i } ^ { \star } - x _ { i } ) } { \Delta t } = B .\tag{10}
$$

Since $A _ { i }$ is strictly concave on $[ \tau _ { i } , 1 ]$ , its derivative is strictly decreasing on this interval, and the inverse in the second branch is well defined. These optimal solutions are derived via the Karush–Kuhn–Tucker (KKT) conditions of Problem (4); a formal proof is provided in Appendix A.

Let $z _ { i } = L _ { i } ( y _ { i } - x _ { i } )$ denote the number of additional cache bits transmitted for user i. For every user satisfying $\tilde { x } _ { i } < y _ { i } ^ { \star } <$ 1,

$$
\frac { \mathrm { d } A _ { i } ( x _ { i } + z _ { i } / L _ { i } ) } { \mathrm { d } z _ { i } } \bigg \vert _ { z _ { i } = z _ { i } ^ { \star } } = \frac { A _ { i } ^ { \prime } ( y _ { i } ^ { \star } ) } { L _ { i } } = \frac { \lambda ^ { \star } } { \Delta t } ,\tag{11}
$$

where $z _ { i } ^ { \star } = L _ { i } ( y _ { i } ^ { \star } { - } x _ { i } )$ . Thus, the optimal allocation equalizes the marginal accuracy gain per additional transmitted cache bit across all interior users.

Theorem 1 does not require a particular parametric utility family. Specializing the result to a given family requires only evaluating $( A _ { i } ^ { \prime } ) ^ { - 1 }$ for the interior branch, while the boundary

cases are handled by (9). We illustrate this using the algebraic and logistic sigmoids characterized in Section IV.

Example 1 (Water Levels for Sigmoid Utilities): For the interior branch of (9), the algebraic sigmoid in (6) gives

$$
( A _ { i } ^ { \prime } ) ^ { - 1 } \big ( s _ { i } ( \lambda ) \big ) = \tau _ { i } + \frac { 1 } { k _ { i } } \sqrt { \bigg ( \frac { M _ { i } k _ { i } } { 2 s _ { i } ( \lambda ) } \bigg ) ^ { 2 / 3 } - 1 } .\tag{12}
$$

For the logistic sigmoid $A _ { i } ( y ) = M _ { i } / \left( 1 + e ^ { - k _ { i } ( y - \tau _ { i } ) } \right)$ gives

$$
( A _ { i } ^ { \prime } ) ^ { - 1 } \big ( s _ { i } ( \lambda ) \big ) = \tau _ { i } + \frac { 1 } { k _ { i } } \operatorname { a r c o s h } \left( \frac { M _ { i } k _ { i } } { 2 s _ { i } ( \lambda ) } - 1 \right) .\tag{13}
$$

We adopt the algebraic specialization because it combines strong empirical fit with efficient per-user allocation: once the dual price is determined, each user’s optimal allocation can be evaluated using only arithmetic operations and radicals. When $B < B _ { \mathrm { m i n } }$ , the feasibility condition of Lemma 1 fails and the closed form above no longer applies; Section V-B handles this regime with an explicit admission policy.

## B. Fallback Policy and Resource Allocation Algorithm

By Lemma 1, Problem (4) is feasible if and only if $B \geq B _ { \operatorname* { m i n } }$ . When $B < B _ { \mathrm { m i n } }$ , no allocation can bring every active user into its concave region within the current slot. The scheduler must therefore invoke a fallback admission policy that prioritizes users under the insufficient backhaul budget. The benefit of such an explicit policy is evaluated empirically in Section VI-C (Fig. 5(b)). As our default fallback, we adopt equalized bytes (EB).

Let $S \triangleq \{ i \in \mathcal { N } _ { t } : x _ { i } < \tau _ { i } \}$ denote the users that have not yet reached their concavity anchors. EB equalizes the number of cache bits delivered during the slot among the users in ${ \mathcal { S } } .$ Since all users share the same slot duration, this is equivalent to initially assigning each user $i \in \mathcal { S } \mathrm { ~ a ~ }$ physical backhaul rate of $B / | S |$ , corresponding to the relative allocation $b _ { i } ^ { \mathrm { E B } } =$ $\frac { B } { | S | L _ { i } }$ . Each relative allocation is capped at the rate required to complete the remaining cache transfer within the slot, $( 1 -$ $x _ { i } ) / \Delta t$ . If a user reaches this cap, the released backhaul rate is redistributed equally among the remaining unfinished users in $s .$ Users outside $s$ receive no bandwidth under the fallback policy. By prioritizing users that have not yet reached their anchors, EB mitigates sub-τ starvation under heavy load.

The complete two-regime allocator is summarized in $\mathrm { A l - }$ gorithm 1. It first evaluates the feasibility threshold $B _ { \mathrm { m i n } }$ If $B \ < \ B _ { \mathrm { m i n } }$ , it invokes the EB fallback. Otherwise, it computes the optimal allocation characterized in Theorem 1 by locating a dual-optimal price $\lambda ^ { \star }$ through the bisection subroutine FINDWATERLEVEL.

FINDWATERLEVEL exploits the monotonicity of the $\mathrm { a g } -$ gregate demand $\begin{array} { r } { g ( \lambda ) \triangleq \dot { \sum _ { i = 1 } } { L _ { i } } \big ( y _ { i } ^ { \star } ( \lambda ) - x _ { i } \big ) / \dot { \Delta t } } \end{array}$ , which is P  continuous and non-increasing in λ. If $g ( 0 ) \leq B _ { \mathrm { \ell } }$ , the entire remaining cache fits within the slot and $\lambda ^ { \star } = 0 ;$ otherwise, bisection solves $g ( \lambda ) \ = \ B$ to tolerance ε. Each evaluation of $g$ costs $O ( | \mathcal { N } _ { t } | )$ , and the search range is bounded by $\bar { \lambda } ,$ the price at which $g ( \bar { \lambda } ) \ = \ B _ { \operatorname * { m i n } } ,$ so the allocator runs in

Algorithm 1 Importance-Aware Resource Allocation   
Input: Active user index set $\overline { { \mathcal { N } _ { t } } }$ with user parameters   
$\{ ( L _ { i } , x _ { i } , \tau _ { i } , A _ { i } ) \} _ { i \in \mathcal { N } _ { t } }$ , total bandwidth $B ,$ slot duration $\Delta t .$   
Output: Bandwidth allocation $\{ b _ { i } ^ { \star } \} _ { i \in \mathcal { N } _ { t } }$   
1: $S \gets \{ i \in \mathcal { N } _ { t } : x _ { i } < \tau _ { i } \}$ ⊲ Sub-τ users   
2: $\begin{array} { r } { B _ { \operatorname* { m i n } }  \frac { 1 } { \Delta t } \sum _ { i \in S } L _ { i } ( \tau _ { i } - x _ { i } ) } \end{array}$   
3: if $B < B _ { \mathrm { m i n } }$ Pthen ⊲ Infeasibility   
4: return EQUALIZEDBYTES $( S , B , \Delta t )$   
5: end if   
6: $\lambda ^ { \star } $ FINDWATERLEVEL $( \mathcal { N } _ { t } , B , \Delta t )$   
7: for each $i \in \mathcal { N } _ { t }$ do   
8: $y _ { i } ^ { \star } \gets \operatorname* { m a x } ( x _ { i } , W _ { i } ( { \boldsymbol { \lambda } } ^ { \star } ) )$ ⊲ Optimal target   
9: $b _ { i } ^ { \star } \gets ( y _ { i } ^ { \star } - x _ { i } ) / \Delta t$ ⊲ Bandwidth assignment   
10: end for   
11: return $\{ b _ { i } ^ { \star } \} _ { i \in \mathcal { N } _ { t } }$

$O ( | \mathcal { N } _ { t } | \log ( \bar { \lambda } / \varepsilon ) )$ time per slot, i.e., near-linear in the number of active users.

## C. Connection to Classical Water-Filling

When all users have the same utility function and cache size, i.e., $A _ { i } ( \cdot ) = A ( \cdot )$ and $L _ { i } = L ,$ the user-specific water levels $W _ { i } ( \lambda )$ reduce to a common level $W ( \lambda )$ , and users differ only in their current progress $x _ { i } .$ . When the backhaul constraint is active, let $W ^ { \star } \triangleq \bar { W } ( \lambda ^ { \star } )$ and $\mathcal { A } \triangleq \{ i \in \mathcal { N } _ { t } : x _ { i } < W ^ { \star } \}$ denote the set of users receiving positive allocations. The budget equality then gives

$$
W ^ { \star } = \frac { 1 } { | \boldsymbol { A } | } \sum _ { i \in \boldsymbol { A } } x _ { i } + \frac { B \Delta t } { | \boldsymbol { A } | L } , \quad b _ { i } ^ { \star } = \frac { 1 } { \Delta t } \big [ \boldsymbol { W } ^ { \star } - \boldsymbol { x } _ { i } \big ] ^ { + } .\tag{14}
$$

This has the same algebraic form as classical water-filling [29], [30], with the channel-dependent ground level replaced by the progress floor $x _ { i }$ and the total power replaced by the deliverable cache fraction $B \Delta t / L$ . Then, a user with greater current progress has a higher ground level and requires less additional bandwidth to reach the common target level.

## VI. EXPERIMENTAL RESULTS

## A. Experiment Settings

1) Simulation environment: We follow the slotted system model of Section III. At the beginning of each time slot of duration $\Delta t ,$ the scheduler observes all active users $\mathcal { N } _ { t }$ and updates the bandwidth allocation, which remains fixed throughout the slot. Handover events arrive according to a Poisson process with mean rate $\rho$ (users per second). Upon each event, the context length of a new user is uniformly sampled from {4K, 8K, 16K} tokens to model heterogeneous context lengths. Each user attempts to complete its handover within an allowed transfer window $T _ { \mathrm { m a x } } .$ . Unless otherwise noted, the default settings are $B = 2 0 \mathrm { G b p s } .$ $\Delta t = 1 0 0 \mathrm { m s } ,$ $T _ { \mathrm { m a x } } = 5 0 0$ ms, $\rho = 4 \mathrm { u s e r s / s } .$ , and context lengths drawn uniformly from {4K, 8K, 16K}, with each point averaged over 100 s of simulation across 100 Monte-Carlo runs.

We adopt $B = 2 0 \mathrm { G b p s }$ as a beyond-5G baseline. While over-provisioned networks (≥50 Gbps) make allocation trivial, and severely limited ones reduce the problem to pure admission control, 20 Gbps serves as a practical yet challenging operating point; its sensitivity is examined in Section VI-B.

Similarly, the slot duration $\Delta t ~ = ~ 1 0 0$ ms balances the re-allocation period against scheduler overhead. The transfer window $T _ { \mathrm { m a x } } ~ = ~ 5 0 0 \mathrm { m s }$ is a mobility-imposed budget for completing the transfer in the background, not an inferencetime latency: in soft-handover architectures, the target can begin loading the cache before the user formally migrates. When the window expires, the transfer is truncated, and the user resumes inference on whatever prefix has arrived. We analyze the sensitivity of both ∆t and $T _ { \mathrm { m a x } }$ in Section VI-C.

All experiments use Qwen3-8B [26] in BF16 precision with KV cache importance scored by Fast KVzip [14]. The peruser accuracy utility $A _ { i } ( \cdot )$ follows the algebraic sigmoid form of Eq. (6), with parameters $M _ { i } , k _ { i } , \tau _ { i }$ fitted independently for each context length {4, 8, 16}K on the RULER benchmark [15] as detailed in Section IV. Because delivery proceeds in importance order, the received fraction $x _ { i }$ in the simulation indexes exactly the $\mathrm { t o p } { - } x _ { i }$ fraction of Section III-C. Thus, the reported accuracy at each slot is obtained by evaluating this fitted sigmoid at the user’s current $x _ { i } .$ . To further validate that this sigmoid-based simulation tracks real inference under the scheduler-induced distribution of $x _ { i }$ , we also measure end-toend accuracy under the same simulation settings. It is obtained by having each user perform inference using the directly delivered $\mathrm { t o p } { - } x _ { i }$ fraction of the cache.

2) Baselines and metrics: Throughout, Ours refers to the proposed two-regime allocator. When the per-slot budget suffices to lift every sub-τ user to the concavity anchor (i.e., the feasibility regime), it applies the weighted water-filling of Theorem 1. Otherwise, it falls back to an admission policy that distributes the budget equally among the contending subτ users. We compare the proposed allocator against three baselines, each of which applies a fixed rule in every slot, irrespective of feasibility.

• Equal allocation (EA): The backhaul capacity is distributed uniformly across active users so that each user receives bandwidth $L _ { i } b _ { i } = B / | \mathcal { N } _ { t } |$ bps, independent of importance or current progress.

• Winner-take-all (WTA): The greedy heuristic which assigns the slot budget to the user with the largest singleslot marginal accuracy gain. We use the cascading variant throughout: when the chosen user cannot absorb the full slot budget (its remaining cache is smaller than $\boldsymbol { B } \cdot \boldsymbol { \Delta t } )$ the leftover rolls over to the next-best user within the same slot. The pure WTA, which discards any leftover capacity, is strictly dominated and is therefore omitted from the main comparisons.

• Proportional-fair (PF): The unweighted proportionalfair allocation that maximizes $\textstyle \sum _ { i } \log ( L _ { i } y _ { i } )$ subject Pto the shared budget, without accessing the accuracy curves [23]. In our single-resource setting, its KKT solution reduces to a common-water-level water-filling on cumulative received bits, which we solve directly. It thus represents a fairness-oriented baseline that is agnostic to the sigmoidal accuracy utility.

![](images/0ba6c1ec1a5b0693e4f3f25cab90c70570319ddeed2e4578783d45fc6185b784.jpg)

![](images/ddbb6caeecaa794e0e6f932a9bee99ab3baf8a30135c6d4cc1cec25a112e9cd5.jpg)  
(a) Accuracy vs. $\rho .$  
(b) Accuracy vs. B.  
Fig. 3. Performance comparison of the proposed allocator against EA, PF, and WTA baselines under edge-LLM handover conditions across (a) varying handover frequency ρ and (b) varying backhaul bandwidth B. Solid curves are computed from the simulation, evaluating each user’s accuracy at its delivered fraction x<sub>i</sub> through the fitted sigmoid utility $A _ { i } ( x _ { i } )$ . Markers additionally show end-to-end measured accuracy. Each marker averages the first 4000 users of the trace, and error bars are 95% confidence intervals over RULER samples. The close agreement between curves and markers confirms that the sigmoid-based simulation reliably predicts real inference accuracy.

These baselines span the fairness–throughput spectrum of resource allocation, with EA and WTA at its two extremes and PF occupying the middle ground [23]. All schemes share the same importance ordering, differing only in allocation: among the baselines, only WTA consults the accuracy utility, while EA and PF are utility-agnostic. Additionally, we compare the proposed method against compute-based baselines: a targetside re-prefill strategy and the hybrid token and cache transmission design from [9].

## B. Main Results

We first evaluate the performance of the proposed allocator against the baselines along two axes that jointly govern the system load: the handover frequency $\rho$ and the backhaul bandwidth B. In Fig. 3, the solid curves denote the simulation accuracy derived from the fitted sigmoid utility in Fig. 2, whereas the markers with error bars indicate the measured end-to-end accuracy.

Focusing on Fig. 3(a), ρ is varied over 1–8 users/s. As ρ increases, the accuracy of all schemes decreases, since the fixed budget B must be shared among more concurrent transfers. The proposed allocator outperforms the baselines throughout, and the gap widens, since heavy loads make importance-aware prioritization essential for maximizing accuracy with scarce resources. At the default $\rho ~ = ~ 4 \mathrm { u s e r s / s }$ , it attains 93.7 % accuracy, within 0.5 pp of the 94.1 % full-cache ceiling.

In Fig. 3(b), we sweep B over 10–30 Gbps. The result demonstrates that the proposed allocator consistently outperforms all three baselines across the entire operating range. Among the baselines, the two extremes, EA and WTA, exhibit a characteristic crossover. In the bandwidth-scarce regime, WTA outperforms EA by concentrating the limited budget to ensure at least one user reaches the operable region, whereas EA leaves all users starved. Conversely, in the bandwidthabundant regime, the ordering reverses: EA benefits from serving multiple users in parallel, while WTA wastes capacity by driving its current winner deep into the saturated tail of its utility before advancing to the next. PF weights each share by progress and cache size, yet trails the proposed allocator: blind to the accuracy curve, it over-serves saturated users and under-serves those near τ.

![](images/07db951535612996042ac0fe5332e786911771d9336eaa085ef00f44c832bc7b.jpg)

![](images/d2ede137c05d6edd04bd0e75835f6cb107ba8bfc8747da8670268308ded774d3.jpg)  
(a) Accuracy vs. ∆t.  
(b) Accuracy vs. $T _ { \mathrm { m a x } } .$

Fig. 4. Performance comparison of the proposed allocator against baselines across (a) varying slot duration ∆t and (b) varying transfer window $T _ { \mathrm { m a x } } .$  
![](images/ae03e2602ffc6eca3f52c9471baa38965c04cccd79f0744e3f7848fa8cf07595.jpg)

![](images/36df2b8e3a7f166b3fef60bdc13ac4b88410be11eb912b68fc2c2d6b5aef4dbc.jpg)  
(a) Effect of ordering.  
(b) Admission control policy.  
Fig. 5. The impacts of (a) importance-aware KV cache ordering relative to random ordering and (b) the admission control mechanism.

## C. Sensitivity and Ablation Analysis

We next examine robustness to two system configuration parameters: the slot duration $\Delta t ,$ and the transfer window $T _ { \mathrm { m a x } }$ . Fig. 4(a) sweeps $\Delta t$ over 20–100 ms with all other parameters at their defaults. The proposed allocator stays above all three baselines across the entire range. EA and WTA cross near $\Delta t \approx 9 0$ ms: as the slot gets coarser, the better fixed rule switches from one to the other. The proposed allocator, however, keeps its accuracy even at the coarsest slots, so it can re-allocate far less often without losing quality.

In Fig. 4(b), the transfer window is swept over 200– 1000 ms, showing the same crossover near 400 ms: short windows penalize EA, while longer ones let its parallelism overtake WTA. The proposed allocator again outperforms throughout, so its advantage is not tied to a particular $T _ { \mathrm { m a x } }$

To analyze the sources of the performance improvements, we isolate the two core mechanisms behind our method in Fig. 5: importance-based cache ordering and the admissioncontrol policy. For the latter, we compare the default policy (Ours (EB), which refers to equalized bytes) against the same allocator with cascading WTA as its fallback (Ours (WTA)).

TABLE II  
PER-USER LATENCY COMPARISON OF THE COMPUTE-BASED BASELINES AND THE PROPOSED METHOD FOR QWEN3-8B.
<table><tr><td rowspan=2 colspan=1>Context</td><td rowspan=1 colspan=1>Re-prefill (ms)</td><td rowspan=1 colspan=1>Hybrid (ms)</td><td rowspan=1 colspan=1>Ours (ms)</td></tr><tr><td rowspan=1 colspan=1>NVIDIA RTX PRO 6000</td><td rowspan=1 colspan=1>PRO 6000, $\overline { { 2 0 \mathrm { G b p s } } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 0 \mathrm { G b p s } } }$ </td></tr><tr><td rowspan=1 colspan=1>4K</td><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>211</td><td rowspan=1 colspan=1>229</td></tr><tr><td rowspan=1 colspan=1>8K</td><td rowspan=1 colspan=1>676</td><td rowspan=1 colspan=1>396</td><td rowspan=1 colspan=1>353</td></tr><tr><td rowspan=1 colspan=1>16K</td><td rowspan=1 colspan=1>1,566</td><td rowspan=1 colspan=1>694</td><td rowspan=1 colspan=1>590</td></tr><tr><td rowspan=1 colspan=1>Avg</td><td rowspan=1 colspan=1>847</td><td rowspan=1 colspan=1>434</td><td rowspan=1 colspan=1>391</td></tr></table>

Ours (WTA) is distinct from the standalone WTA baseline used above, which applies cascading WTA in every slot.

Fig. 5(a) sweeps $\rho$ for the proposed allocator under Fast KVzip and random ordering. Importance ordering keeps average accuracy above 90 % across the entire range, whereas random ordering falls sharply from 56.7 % at $\rho = 2$ to 20.2 % at $\rho = 8$ . This demonstrates that importance-based ordering strengthens ImpactHO’s anytime inference capability by transmitting higher-ranked cache entries first, thereby improving inference accuracy at intermediate transfer points.

Fig. 5(b) sweeps $\rho$ over 4–12 users/s to stress the infeasibility regime. Both variants with admission control are compared against the same allocator with the fallback disabled (No admission). At low $\rho ,$ the three curves coincide, as the feasibility regime dominates. As $\rho$ grows, both variants with admission control pull away sharply from the one without it, confirming that admission control is the dominant factor in this regime. We define the sub-τ starvation rate as the fraction of users whose received fraction never reaches τ within the transfer window $T _ { \mathrm { m a x } \cdot } \mathrm { A t } \rho = 1 2 \mathrm { u s e r s } / \mathrm { s } $ , the sub-τ starvation rate is 20.8 % without admission versus 0.65 % with EB. At our main slot duration $( \Delta t = 1 0 0 \mathrm { m s } )$ , EB consistently outperforms WTA, whose starvation rate is 15.5 % versus EB’s 0.65 %. This is because WTA drives one user into the saturated tail beyond $\tau$ before advancing the next, whereas EB lets several progress toward operability.

## D. Comparison with Compute-Based Baselines

We evaluate the per-user latency performance of the proposed method by comparing it against compute-based reprefill baselines. In Table II, Re-prefill reconstructs the context directly from the raw tokens, whereas Hybrid transmits both tokens and KV cache to balance the prefill processing time and cache transmission latency, a scheme appropriately modified from [9] for our evaluation. In the hybrid baseline, KV cache importance is not taken into account, and the backhaul bandwidth is allocated entirely in a first-come, firstserved (FCFS) manner. Because the data size of the tokens is negligible compared to the KV cache, we do not consider token transmission latency.

For comparison, we generate user arrivals following a Poisson process with $\rho = 4$ over the time window $t \in [ 0 , 1 ]$ s, and repeat this process 1,000 times. For each user, the service latency is measured from the moment of request arrival. Specifically, for the re-prefill and hybrid baselines, latency spans until the context is fully restored; for the proposed method, it lasts until a sufficient amount of the KV cache is transferred to reach 99 % of the full accuracy. Here, the transfer window $T _ { \mathrm { m a x } }$ is set to ∞, so that no scheme is truncated by the deadline. We assume that both the re-prefill and hybrid baselines are provisioned with sufficient GPUs to process each user request independently, so their latencies are free of compute contention. This idealized assumption establishes a much stricter baseline than a direct comparison with [9].

At a context length of 4K, the proposed method shows comparable performance to the baselines. However, as the context length increases, the proposed method outperforms the baselines, which suffer from severe computational overhead during full context recomputation. Moreover, the proposed method can further reduce latency by leveraging its anytime property, while ensuring graceful performance degradation.

## E. Discussion

The reported gains require importance-ordered transmission, which adds two source-side costs and a small wire overhead. (i) Scoring: assigning a Fast KVzip [14] score to every KV entry is a one-time, per-session computation done while the source still serves the user, not at handover. (ii) Sorting: reordering by score is an O(n log n) sort over indices, a small fraction of the transfer time. (iii) Metadata: because the transfer may be truncated, each entry carries its 3-byte (layer, head, token) coordinate against the 0.5 KB payload of Section III-C, under 1 % overhead.

The proposed allocator optimizes each slot rather than the horizon. To bound the resulting loss, we compare against a clairvoyant upper bound that observes all future arrivals and maximizes the objective without the concave-region restriction. Solved to certified global optimality via a piecewiselinear MILP with a validated upper correction, this bound never underestimates the true continuous optimum. The proposed allocator attains 98.2–99.5 % of it across all loads (within 0.53 % at $\rho = 1$ , 1.77 % at $\rho = 8 )$ . Thus, neither the myopic objective nor the concave-region restriction costs much in practice, though the widening gap at higher load suggests horizon-aware scheduling as a direction for future work.

## VII. CONCLUSION

We presented the ImpactHO framework, an importanceaware framework for maximizing the average inference accuracy across users during edge LLM handover subject to limited backhaul bandwidth. ImpactHO combines three key components: (1) importance-ordered sequential KV cache transfer, which exploits sparsity in token-level cache importance to transmit the most useful entries first; (2) an empirically validated sigmoid characterization of inference accuracy with partially transferred caches; and (3) an optimal weighted waterfilling allocator whose homogeneous special case reduces to classical water-filling. In realistic edge regimes, importanceordered transfer attains near-full-cache accuracy with lower latency than baselines. Moreover, across a range of slot durations, backhaul bandwidths, and concurrent handover loads, the proposed allocator consistently achieves higher average accuracy than the baselines. These results confirm that jointly accounting for KV cache importance and backhaul resource allocation enables accurate and efficient LLM handover at the network edge.

## APPENDIX A PROOF OF THEOREM 1

Recall the effective lower bound $\tilde { x } _ { i } = \operatorname* { m a x } ( \tau _ { i } , x _ { i } )$ from Section III, which collapses the box constraint of (4) to the single interval $y _ { i } \in [ \tilde { x } _ { i } , 1 ]$ . We work in the feasibility regime $B \geq B _ { \operatorname* { m i n } }$ of Lemma 1. By Assumption $1 , - A _ { i }$ is strictly convex on $[ \tau _ { i } , 1 ]$ and the constraint set is a polytope, so Problem (4) is convex with a strictly convex objective. The Lagrangian is

$$
\begin{array} { c } { { \displaystyle { \mathcal { L } } ( \mathbf { y } , \lambda , \pmb { \mu } , \pmb { \nu } ) = - \sum _ { i = 1 } ^ { N } A _ { i } ( y _ { i } ) + \lambda \Bigg ( \sum _ { i = 1 } ^ { N } \frac { L _ { i } ( y _ { i } - x _ { i } ) } { \Delta t } - B \Bigg ) } } \\ { { + \displaystyle { \sum _ { i = 1 } ^ { N } \mu _ { i } ( \tilde { x } _ { i } - y _ { i } ) + \sum _ { i = 1 } ^ { N } \nu _ { i } ( y _ { i } - 1 ) } . \qquad ( 1 5 } } \end{array}\tag{}
$$

The KKT conditions for the optimal $( \mathbf { y } ^ { \star } , \lambda ^ { \star } , \mu ^ { \star } , \nu ^ { \star } )$ are as follows:

$$
\sum _ { i = 1 } ^ { N } \frac { L _ { i } ( y _ { i } ^ { \star } - x _ { i } ) } { \Delta t } \leq B , \ \lambda ^ { \star } \geq 0 ,\tag{16}
$$

$$
\lambda ^ { \star } \left( \sum _ { i = 1 } ^ { N } \frac { L _ { i } ( y _ { i } ^ { \star } - x _ { i } ) } { \Delta t } - B \right) = 0 ,\tag{17}
$$

$$
\tilde { x } _ { i } \leq y _ { i } ^ { \star } \leq 1 , \mu _ { i } ^ { \star } , \nu _ { i } ^ { \star } \geq 0 ,\tag{18}
$$

$$
\mu _ { i } ^ { \star } ( { \tilde { x } } _ { i } - y _ { i } ^ { \star } ) = 0 , \nu _ { i } ^ { \star } ( y _ { i } ^ { \star } - 1 ) = 0\tag{19}
$$

for $i \in \{ 1 , \ldots , N \}$ . From $\partial \mathcal { L } / \partial y _ { i } ~ = ~ 0$ , the stationarity condition gives

$$
A _ { i } ^ { \prime } ( y _ { i } ^ { \star } ) = \frac { \lambda ^ { \star } L _ { i } } { \Delta t } - \mu _ { i } ^ { \star } + \nu _ { i } ^ { \star } .\tag{20}
$$

By Assumption 1, $A _ { i } ^ { \prime }$ is strictly decreasing on $[ \tau _ { i } , 1 ]$ , hence invertible. Stationarity (20) with complementary slackness (19) gives the three branches of (9): the interior case $\mu _ { i } ^ { \star } = \nu _ { i } ^ { \star } = 0$ yields $A _ { i } ^ { \prime } ( y _ { i } ^ { \star } ) = s _ { i } ( \lambda ^ { \star } )$ , while $\nu _ { i } ^ { \star } > 0$ and $\mu _ { i } ^ { \star } > 0$ activate the bounds $W _ { i } = 1$ and $W _ { i } = \tau _ { i }$ when $s _ { i } ( \lambda ^ { \star } ) \leq A _ { i } ^ { \prime } ( 1 )$ and $s _ { i } ( \lambda ^ { \star } ) \geq A _ { i } ^ { \prime } ( \tau _ { i } )$ , respectively. Combining with $y _ { i } ^ { \star } \geq x _ { i }$ from $b _ { i } ^ { \star } \geq 0 ~ \mathrm { g i v e s }$

$$
y _ { i } ^ { \star } = \operatorname* { m a x } \{ x _ { i } , W _ { i } ( \lambda ^ { \star } ) \} .\tag{21}
$$

It remains to characterize $\lambda ^ { \star }$ . Define the aggregate demand

$$
g ( \lambda ) \ \triangleq \ \sum _ { i = 1 } ^ { N } \frac { L _ { i } \left( y _ { i } ^ { \star } ( \lambda ) - x _ { i } \right) } { \Delta t }\tag{22}
$$

with $y _ { i } ^ { \star } ( \lambda )$ given by (21). Because $A _ { i } ^ { \prime }$ is strictly decreasing on $[ \tau _ { i } , 1 ]$ $W _ { i } ( \lambda )$ is strictly decreasing in λ on its interior region and constant elsewhere; taking max $\{ x _ { i } , \cdot \}$ preserves this monotonicity, so $g ( \lambda )$ is continuous and non-increasing on $\lambda \geq 0$ , with boundary values $\begin{array} { r } { g ( 0 ) = \sum _ { i } L _ { i } ( 1 - x _ { i } ) / \Delta t } \end{array}$ and $g ( \bar { \lambda } ) \ = \ B _ { \operatorname * { m i n } }$ , where $\bar { \lambda } \triangleq$ max<sub>i</sub> $A _ { i } ^ { \prime } ( \tau _ { i } ) \Delta t / L _ { i }$ . By the intermediate value theorem, a ${ { \lambda } ^ { \star } } \geq 0$ with $g ( \lambda ^ { \star } ) = B$ exists for every $B \in [ B _ { \operatorname* { m i n } } , g ( 0 ) ]$ , and for $B > g ( 0 )$ the budget is slack with $\lambda ^ { \star } = 0 ; \lambda ^ { \star }$ is unique when some user is strictly interior, and otherwise any such $\lambda ^ { \star }$ yields the same $\mathbf { y } ^ { \star }$ , which is unique by strict convexity. This completes the proof of Theorem 1.

## REFERENCES

[1] W. Saad, M. Bennis, and M. Chen, “A vision of 6G wireless systems: Applications, trends, technologies, and open research problems,” IEEE Netw., vol. 34, no. 3, pp. 134–142, 2020.

[2] K. B. Letaief, W. Chen, Y. Shi, J. Zhang, and Y.-J. A. Zhang, “The roadmap to 6G: AI empowered wireless networks,” IEEE Commun. Mag., vol. 57, no. 8, pp. 84–90, Aug. 2019.

[3] Y. Mao, C. You, J. Zhang, K. Huang, and K. B. Letaief, “A survey on mobile edge computing: The communication perspective,” IEEE Commun. Surveys Tuts., vol. 19, no. 4, pp. 2322–2358, 2017.

[4] E. Li, L. Zeng, Z. Zhou, and X. Chen, “Edge AI: On-demand accelerating deep neural network inference via edge computing,” IEEE Trans. Wireless Commun., vol. 19, no. 1, pp. 447–457, Jan. 2020.

[5] S. Kholmatov, S. Cho, S. Chong, and K. Lee, “AoRA: AI-on-RAN for backhaul-free edge inference,” in Proc. ACM Conf. SIGCOMM, Aug. 2025, pp. 1263–1265.

[6] S. Wang, R. Urgaonkar, M. Zafer, T. He, K. Chan, and K. K. Leung, “Dynamic service migration in mobile edge computing based on Markov decision process,” IEEE/ACM Trans. Netw., vol. 27, no. 3, pp. 1272– 1288, Jun. 2019.

[7] A. Machen, S. Wang, K. K. Leung, B. J. Ko, and T. Salonidis, “Live service migration in mobile edge clouds,” IEEE Wireless Commun. Mag., vol. 25, no. 1, pp. 140–147, Feb. 2018.

[8] M. V. Ngo, T. Luo, H. T. Hoang, and T. Q. Quek, “Coordinated container migration and base station handover in mobile edge computing,” in Proc. IEEE Global Commun. Conf. (GLOBECOM), 2020, pp. 1–6.

[9] S. Lee, J. Park, C. Zheng, and H. Park, “Low-latency edge LLM handover via joint KV cache transfer and token prefill,” arXiv preprint arXiv:2603.28018, Mar. 2026.

[10] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, “Efficient streaming language models with attention sinks,” in Proc. Int. Conf. Learn. Representations (ICLR), 2024.

[11] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. R´e, C. Barrett, Z. Wang, and B. Chen, “H2O: Heavyhitter oracle for efficient generative inference of large language models,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023, pp. 34 661–34 710.

[12] Y. Li, Y. Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai, P. Lewis, and D. Chen, “SnapKV: LLM knows what you are looking for before generation,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 37, Dec. 2024, pp. 22 947–22 970.

[13] J.-H. Kim, J. Kim, S. Kwon, J. W. Lee, S. Yun, and H. O. Song, “KVzip: Query-agnostic KV cache compression with context reconstruction,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Oct. 2025.

[14] J.-H. Kim, D. Han, and S. Yun, “Fast KVzip: Efficient and accurate LLM inference with gated KV eviction,” arXiv preprint arXiv:2601.17668, Jan. 2026.

[15] C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, and B. Ginsburg, “RULER: What’s the real context size of your long-context language models?” in Proc. Conf. Lang. Model. (COLM), 2024.

[16] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang, “DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving,” in Proc. USENIX Symp. Oper. Syst. Des. Implement. (OSDI), Jul. 2024, pp. 193–210.

[17] P. Patel, E. Choukse, C. Zhang, A. Shah, I. Goiri, S. Maleki, and R. Bianchini, “Splitwise: Efficient generative llm inference using phase splitting,” in Proc. ACM/IEEE Annu. Int. Symp. Comput. Archit. (ISCA), 2024, pp. 118–132.

[18] Y. Liu, H. Li, Y. Cheng, S. Ray, Y. Huang, Q. Zhang, K. Du, J. Yao, S. Lu, G. Ananthanarayanan, M. Maire, H. Hoffmann, A. Holtzman, and J. Jiang, “CacheGen: KV cache compression and streaming for fast large language model serving,” in Proc. ACM Conf. SIGCOMM, 2024, pp. 38–56.

[19] D. G¨und¨uz, Z. Qin, I. E. Aguerri, H. S. Dhillon, Z. Yang, A. Yener, K. K. Wong, and C.-B. Chae, “Beyond transmitting bits: Context, semantics, and task-oriented communications,” IEEE J. Sel. Areas Commun., vol. 41, no. 1, pp. 5–41, Jan. 2023.

[20] J. Im, N. Kwon, T. Park, J. Woo, J. Lee, and Y. Kim, “Attention-aware semantic communications for collaborative inference,” IEEE Internet Things J., vol. 11, no. 22, pp. 37 008–37 020, Nov. 2024.

[21] L. Qiao, M. B. Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and D. Niyato, “Token communications: A large model-driven framework for cross-modal context-aware semantic communications,” IEEE Wireless Communications, vol. 32, no. 5, pp. 80–88, Oct. 2025.

[22] L. Qiao, M. B. Mashhadi, Z. Gao, and D. G ¨und ¨uz, “Token-domain multiple access: Exploiting semantic orthogonality for collision mitigation,” in IEEE INFOCOM 2025 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), 2025, pp. 1–6.

[23] F. P. Kelly, A. K. Maulloo, and D. K. H. Tan, “Rate control for communication networks: shadow prices, proportional fairness and stability,” J. Oper. Res. Soc., vol. 49, no. 3, pp. 237–252, 1998.

[24] J.-W. Lee, R. Mazumdar, and N. Shroff, “Non-convex optimization and rate control for multi-class services in the internet,” IEEE/ACM Trans. Netw., vol. 13, no. 4, pp. 827–840, Aug. 2005.

[25] P. Hande, S. Zhang, and M. Chiang, “Distributed rate allocation for inelastic flows,” IEEE/ACM Trans. Netw., vol. 15, no. 6, pp. 1240–1253, Dec. 2007.

[26] A. Yang et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, May 2025.

[27] L. Kundu, X. Lin, R. Gadiyar, J.-F. Lacasse, and S. Chowdhury, “Airan: Transforming ran with ai-driven computing infrastructure,” IEEE Commun. Mag., vol. 64, no. 1, pp. 168–174, Jan. 2026.

[28] A. Grattafiori et al., “The Llama 3 herd of models,” arXiv preprint arXiv:2407.21783, Jul. 2024.

[29] D. P. Palomar and J. R. Fonollosa, “Practical algorithms for a family of waterfilling solutions,” IEEE Trans. Signal Process., vol. 53, no. 2, pp. 686–695, Feb. 2005.

[30] T. M. Cover and J. A. Thomas, Elements of Information Theory, 2nd ed. Hoboken, NJ, USA: Wiley-Interscience, 2006.
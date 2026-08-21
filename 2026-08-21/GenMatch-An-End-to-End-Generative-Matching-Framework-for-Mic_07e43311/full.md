# GenMatch: An End-to-End Generative Matching Framework for Micro-View Order-Dispatching in Ride-Hailing

Chuang Liu<sup>∗</sup>   
Didi Chuxing   
Hangzhou, China   
chuangliu@didiglobal.com Zirui Yuan   
The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China   
zyuan779@connect.hkust-gz.edu.cn

Ming Wang Didi Chuxing Beijing, China nicholaswangming@didiglobal.com

Yuxueqing Zhang Didi Chuxing Beijing, China zhangyuxueqing@didiglobal.com

Weiqi Hu   
Didi Chuxing   
Beijing, China   
huweiqi@didiglobal.com   
Li Ma   
Didi Chuxing   
Beijing, China   
malimarey@didiglobal.com

Tengfei Lyu The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China tlyu077@connect.hkust-gz.edu.cn

Yanghan Cheng Didi Chuxing Beijing, China chengyanghan@didiglobal.com

Zihao Lu<sup>†</sup>   
Didi Chuxing   
Beijing, China   
luzihao@didiglobal.com

## Abstract

Micro-View Order-Dispatching assigns available drivers to passenger orders within each dispatch batch. It is critical to the service quality and operational eficiency of ride-hailing platforms. Mainstream industrial solutions follow a multi-stage paradigm consist ing of model prediction, value calculation, and dispatch matching. Although dispatch quality is determined by the final batch-level assignment, these stages optimize diferent intermediate objectives. This creates cross-stage objective inconsistency, so improving any single stage does not necessarily improve the overall dispatch re sult. Generative modeling ofers a natural solution by mapping the system context directly to the final output. Motivated by this capability, we formulate Micro-View Order-Dispatching as a generative matching problem and propose an end-to-end Generative Matching framework (GenMatch), the first generative framework for this task to be deployed in a real-world production environment. However, applying generative modeling to this problem introduces three challenges. First, the model must encode an entire dispatch batch, but each batch forms a dynamic sparse bipartite graph, requiring eficient structured batch-level encoding. Second, replacing the hand-crafted value function requires learning unified business utility from heterogeneous feedback. Third, directly generating an assignment requires tracking the evolving matching state because every selected order-driver pair changes the remaining feasible candidates. GenMatch addresses these challenges through a Context-Aware Bipartite Encoder, a Business-Aware Utility Learner, and a State-Aware Pointer Decoder. Extensive ofline evaluations and on line A/B tests in five cities across DiDi’s international ride-hailing markets demonstrate consistent improvements over competitive baselines, confirming the efectiveness and practicality ofGenMatch for industrial order-dispatching.

Ride-Hailing; Micro-View Order-Dispatching; Generative Matching;   
Sequential Decision Making;

![](images/51f7b1a95d43b33cdb593fa5c90c188d0d1f40ad8331eeb27e802eb521d39163.jpg)  
Figure 1: Comparison of (a) the end-to-end generative paradigm and (b) the conventional multi-stage paradigm for Micro-View Order-Dispatching.

## 1 Introduction

Order-dispatching refers to assigning passengers’ orders to available drivers in real time. It directly afects passenger and driver experience, which in turn influences platform revenue. Therefore, it is a core process of ride-hailing platforms. Existing studies take two complementary views [34]. The Macro-View values and coordinates current decisions under long-term supply–demand evolution [15, 22, 29]. In contrast, the Micro-View considers short-term, real-time assignment within a city. At each fixed dispatch interval, the system collects available orders and drivers; every feasible orderdriver combination forms an order-driver (OD) pair, and together they constitute a dispatch batch. Selecting one-to-one assignments from this batch under strict latency constraints is termed Micro-View Order-Dispatching (MICOD).

Mainstream industrial MICOD solutions follow the multi-stage paradigm in Figure 1(b), operating on a single feasible OD pair (pair-level) and the entire dispatch batch (batch-level). They predict pair-level business signals, such as the probabilities of driver answer (DA), passenger cancellation after answer (PCAA), and driver cancellation after answer (DCAA); aggregate them into matching weights with a hand-crafted value function; and apply a batch level solver such as Kuhn–Munkres matching [10, 14]. Although practical, separately optimizing these stages creates cross-stage objective inconsistency: better intermediate predictions or weights need not improve the final assignment. Generative models address the same issue by mapping context directly to final outputs in search and query suggestion [1, 5], recommendation [3, 40], and advertising [30]. We therefore formulate MICOD as end-to-end generation of an assignment from the complete dispatch batch.

This formulation must recover the three capabilities of the replaced pipeline: representing candidates, combining heterogeneous business objectives, and constructing a feasible batch-level matching. This yields three domain-specific challenges.

C1: Structured batch-level encoding of a dynamic sparse bipartite graph. Unlike independent pair-level models, a generative model must encode the entire dispatch batch. Its bipartite graph varies in orders, drivers, and feasible OD pairs and is highly sparse, requiring structured batch-level encoding that preserves online computational eficiency.

C2: Learning unified business utility from heterogeneous feedback. Replacing manually tuned value rules requires learning business utility directly. Yet generation targets do not fully express outcomes such as DA, PCAA, and DCAA, whose diferent semantics and directions must be integrated into stable, unified guidance.

C3: Generating batch-level assignments under an evolving matching state. Dispatch is evaluated by its joint assignment rather than individual pair scores. During generation, selecting one OD pair invalidates every pair sharing its order or driver and changes the remaining opportunities, requiring an evolving state that supports feasible, batch-coordinated decisions.

To address them, we propose Generative Matching (GenMatch), which maps a dispatch batch directly to an assignment sequence through the encoder–decoder paradigm in Figure 1(a). Its Context-Aware Bipartite Encoder performs sparse message passing over feasible OD edges; its Business-Aware Utility Learner uses auxil iary supervision to derive business guidance from heterogeneous outcomes; and its State-Aware Pointer Decoder tracks selected and residual candidates while dynamically masking infeasible pairs. Together, they support end-to-end generation under MICOD’s structural, business, and matching constraints.

The main contributions are summarized as follows.

• To the best of our knowledge, we propose the first end-to-end generative framework for MICOD deployed in a real-world production environment. It directly generates the assignment from an entire dispatch batch and avoids the crossstage objective inconsistency.

• We develop three components for generative matching: a Context-Aware Bipartite Encoder to eficiently encode dynamic sparse bipartite graphs, a Business-Aware Utility Learner to learn unified utility from heterogeneous feedback, and a State-Aware Pointer Decoder to generate feasible assignments under an evolving matching state.

• We conduct extensive ofline evaluations and online A/B tests in five cities across DiDi’s international ride-hailing markets. The results demonstrate consistent improvements over competitive baselines and confirm the efectiveness and practical value of GenMatch in production systems.

## 2 Related Work

## 2.1 Ride-Hailing Order-Dispatching

Efective order-dispatching improves service quality and platform eficiency. Macro-View studies optimize long-term demand–supply dynamics through regional value learning [21, 22, 29], online or offline reinforcement learning [18, 37], knowledge-enhanced dispatch control [7], and multi-agent regional cooperation [9, 25, 31].

Micro-View studies make decisions within the current dispatch batch through three paradigms. Policy-based methods such as CoRide and CoopRide produce regional or grid-level actions rather than complete OD-pair assignments [9, 25]. D2SN sequentially selects OD pairs or hold actions using a two-layer Markov decision process and an encoder–decoder reinforcement-learning policy [34], but lacks explicit OD-pair interaction modeling and direct supervision from heterogeneous business outcomes. The industrial multi-stage paradigm predicts pair-level signals, calculates matching weights, and applies Greedy [38], Kuhn–Munkres [10, 14], or Gale–Shapley [4] matching. Related systems use deep graph learning for constrained matchmaking and courier pooling [13, 20], but address generic matchmaking or many-to-one assignment rather than one-to-one MICOD. GenMatch instead unifies batch encoding, business-utility learning, and feasible assignment generation.

## 2.2 Generative Modeling

Generative modeling produces a target sequence autoregressively from its context instead of scoring only predefined outputs, supporting large or variable output spaces. Industrial systems have applied this formulation to item generation and large-scale sequential recommendation [16, 36], food-delivery and e-commerce recommendation [6, 42], and multi-business, local-life, and advertising scenarios [11, 28, 30]. Distributed training systems further scale generative recommendation to industrial workloads [26]. Recent extensions combine behavioral and semantic information, generate long semantic identifiers in parallel, or apply semantic identifiers to next-point-of-interest recommendation [8, 24, 27]. Other studies address the training–inference gap through prefix-aware optimization or combine difusion with knowledge-graph reasoning [33, 39]. Reasoning-augmented language models further support generative next-point-of-interest recommendation [41]. Non-autoregressive generation has also been explored for reranking, while large language models connect quality-aware ranking with candidate generation at web scale [17, 19].

Generative methods also reduce cross-stage objective inconsistency by replacing retrieval and ranking with direct recommen dation generation [3, 40], unifying e-commerce search and query suggestion [1, 2, 5], or jointly training retrieval and ranking [12]. Multi-stage alignment further learns preferences from clicks for generative query suggestion [32], while reinforcement learning improves relevance within generative search ranking [35]. Pointer Networks show that structured solutions can be generated by selecting elements from a variable-size input set [23]. MICOD shares this structure, but its candidates form a dynamic sparse bipartite graph whose feasible set changes after every selection. GenMatch extends pointer generation to construct a complete one-to-one matching from such a dispatch batch.

## 3 Preliminary

In this section, we first describe the MICOD process and introduce the corresponding business concepts and notation. We then for mulate MICOD as a batch-level matching problem and present its generative formulation.

MICOD Business Process. A ride-hailing platform triggers dispatch at fixed intervals. At dispatch step �, the platform first collects the currently available passenger orders and drivers. Let $O _ { t } =$ $\big \{ o 1 , \dots , o _ { N _ { t } ^ { o } } \big \}$ and $\mathcal { D } _ { t } = \{ d _ { 1 } , . . . , d _ { N _ { t } ^ { d } } \}$ denote the order and driver sets, respectively. Their sizes vary across dispatch steps. The upstream system then filters infeasible combinations according to pickup distance, service range, and other business rules. The remaining feasible OD pairs form the candidate set $\mathcal { E } _ { t } \subseteq O _ { t } \times \mathcal { D } _ { t }$ Together, the orders, drivers, and candidate OD pairs constitute a dispatch batch, represented as a dynamic sparse bipartite graph

$$
\mathcal { G } _ { t } = ( O _ { t } , \mathcal { D } _ { t } , \mathcal { E } _ { t } ) ,\tag{1}
$$

where each candidate edge $e _ { i j } = ( o _ { i } , d _ { j } ) \in \mathcal { E } _ { t }$ denotes a feasible OD pair. Let $P _ { t } = | \mathcal { E } _ { t } |$ | be the number of candidate OD pairs. We index these edges as $\mathcal { E } _ { t } = \{ e _ { 1 } , . . . , e _ { P _ { t } } \}$ , where each $e _ { r }$ corresponds to an edge $e _ { i j } .$ . Each order, driver, and candidate edge is associated with a feature vector $\mathbf { x } _ { i } ^ { o } \in \mathbb { R } ^ { d _ { o } } , \mathbf { x } _ { j } ^ { d } \in \mathbb { R } ^ { d _ { d } }$ , and $\mathbf { x } _ { i j } ^ { e } \in \mathbb { R } ^ { d _ { e } }$ , respectively. These feature vectors encode spatiotemporal, behavioral, and business signals (e.g., pickup time and platform revenue).

The dispatch system takes the complete dispatch batch $\mathcal { G } _ { t }$ as input and selects a set of OD pairs ${ M } _ { t } \subseteq { \mathcal { E } } _ { t }$ for assignment. Each selected pair is broadcast to its driver and enters a stage-wise service process. The driver decides whether to answer. After an answer, the passenger or driver may cancel; the trip is completed only if neither side cancels. This process produces DA, PCAA, DCAA, and tripcompletion outcomes. For each broadcast pair $e _ { i j } , \mathbf { y } _ { i j }$ records these events. The multi-task labels follow this service dependency, and downstream events can occur only after driver answer. Unbroadcast candidates have no observed service-process labels and are excluded from the auxiliary loss. Let $C _ { t } \subseteq M _ { t }$ denote the set of OD pairs that complete their trips. The completed set $C _ { t }$ is later used to construct the target sequence for generative training. Candidate retrieval and feasibility filtering are provided by the existing upstream system and are outside the scope of this work.

MICOD Problem Formulation. Let $U ( \mathcal { M } _ { t } ; \mathcal { G } _ { t } )$ denote the business utility of assignment $\textstyle { \mathcal { M } } _ { t }$ within dispatch batch $\mathcal { G } _ { t }$ . It captures the joint quality of the assignment through outcomes such as DA, PCAA, DCAA, trip completion, pickup time, and platform revenue. Because selected OD pairs compete for shared orders and drivers, this utility is defined over the complete assignment rather than independent pairs. The MICOD objective is to find the feasible matching with the highest batch-level assignment utility:

$$
\mathcal { M } _ { t } ^ { * } = \arg \operatorname* { m a x } _ { \mathcal { M } _ { t } \in \mathcal { F } ( \mathcal { G } _ { t } ) } U ( \mathcal { M } _ { t } ; \mathcal { G } _ { t } ) ,\tag{2}
$$

where $\mathcal { F } ( G _ { t } )$ is the set of feasible matchings over $\mathcal { G } _ { t }$ . Each order and each driver can appear in at most one selected OD pair:

$$
\begin{array} { r l } { \displaystyle \sum _ { j } \mathbb { I } \big ( ( o _ { i } , d _ { j } ) \in { \cal M } _ { t } \big ) \le 1 , } & { { } \forall o _ { i } \in { \cal O } _ { t } , } \\ { \displaystyle \sum _ { i } \mathbb { I } \big ( ( o _ { i } , d _ { j } ) \in { \cal M } _ { t } \big ) \le 1 , } & { { } \forall d _ { j } \in { \cal { D } } _ { t } . } \end{array}\tag{3}
$$

Here, I(·) is the indicator function. The matching must also be produced within a strict online latency budget. The conventional multi-stage paradigm approximates this objective by predicting pair-level signals, converting them into matching weights through a hand-crafted value function, and applying a matching solver. In this work, end-to-end means that model prediction, value calculation, and dispatch matching are replaced by one jointly trained model.

Generative MICOD Formulation. GenMatch constructs the assignment directly through sequential generation over the candidate OD pairs. An assignment $\textstyle { \mathcal { M } } _ { t }$ can be represented by an ordered generation sequence

$$
\boldsymbol { y } _ { t } = \left( e _ { t } ^ { ( 1 ) } , e _ { t } ^ { ( 2 ) } , \ldots , e _ { t } ^ { ( K _ { t } ) } \right) ,\tag{4}
$$

where $K _ { t } = | \mathcal { M } _ { t } |$ and the set of generated OD pairs equals ${ \mathbf { } } { \mathbf { } } _ { } { \mathbf { } } { \mathbf { } } _ { } { \mathbf { } } { \mathbf { } } _ { } { \mathbf { } } { \mathbf { } } _ { } { \mathbf { } } { \mathbf { } } _ { } { \mathbf { } } { \mathbf { } } _ { } { \mathbf { } } { \mathbf { } } _ { } { \mathbf { } } { \mathbf { } } _ { } { } \mathbf { } _ { } { } { \mathbf { } } _ { } { } \mathbf { } _ { } { } { \mathbf { } } _ { } { } \mathbf { } _ { } { } { \mathbf { } } _ { } { } \mathbf { } _ { } { } { \mathbf { } } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { }  { \mathbf } _ { } { } \mathbf { } _ { } { } \mathbf { } _ { } \mathbf { } _ { } { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf _ { } \mathbf { } _ { } \mathbf { } _ { } \mathbf _ { } \mathbf { } _ { } \mathbf _ { } \mathbf { } _ { } \mathbf _ { } \mathbf { } _ \mathbf { } _ { } \mathbf _ { } \mathbf { } _ \mathbf { } _ \mathbf { } _ { } \mathbf _ { } \mathbf _ { } \mathbf { } _ \mathbf { } _ \mathbf { } _ \mathbf { } _ \mathbf  $ The sequence specifies how the matching is constructed; the final dispatch result is the resulting set rather than a generation order. At generation step �, the model selects one candidate $e _ { t } ^ { ( k ) }$ from the residual graph ${ \mathcal G } _ { t } ^ { ( k ) }$ . All candidates sharing its order or driver then become infeasible and are removed before the next step. The probability of the complete assignment sequence is factorized as

$$
\operatorname* { P r } ( \mathcal { Y } _ { t } \mid \mathcal { G } _ { t } ) = \prod _ { k = 1 } ^ { K _ { t } } \operatorname* { P r } \left( e _ { t } ^ { ( k ) } \mid \mathcal { G } _ { t } ^ { ( k ) } , e _ { t } ^ { ( < k ) } \right) ,\tag{5}
$$

where $e _ { t } ^ { ( < k ) }$ denotes the selected OD pairs. GenMatch learns to map the complete dispatch batch directly to a feasible assignment while conditioning each decision on the evolving matching state. The next section presents how this formulation is instantiated.

## 4 Methods

Figure 2 presents the Context-Aware Bipartite Encoder, Business-Aware Utility Learner, and State-Aware Pointer Decoder in panels (a), (b), and (c), respectively. They model matching and competition information, learn business-aware candidate representations and utility logits, and generate assignments under the evolving matching state. We follow Section 3 and omit � when clear; �<sub>�</sub> denotes the �-th candidate edge, whereas $e _ { i j }$ emphasizes its endpoints.

![](images/2cef025912ff9a1793c5122ffea10ee3ec00549669bbd4604a86b6e7b8d2b286.jpg)  
Figure 2: GenMatch architecture: (a) Context-Aware Bipartite Encoder, (b) Business-Aware Utility Learner, and (c) State-Aware Pointer Decoder.

## 4.1 Context-Aware Bipartite Encoder

The suitability of an OD pair depends not only on its own features but also on the other candidates sharing its order or driver. The Context-Aware Bipartite Encoder in Figure 2(a) therefore models batch-level matching and competition relationships over the complete candidate graph. It first summarizes candidates competing at each endpoint, then evaluates potential matches under this competitive context.

4.1.1 Input Representation. The encoder takes $\mathcal { G } _ { t }$ and transforms its raw order, driver, and candidate-edge features $\mathbf { x } _ { i } ^ { o } , \mathbf { x } _ { j } ^ { d } .$ , and $\mathbf { x } _ { i j } ^ { e }$ with separate tokenizers. The resulting �-dimensional embeddings $\mathbf { h } _ { i } ^ { o , 0 } , \mathbf { h } _ { j } ^ { d , 0 }$ , and $\mathbf { h } _ { i j } ^ { e , 0 }$ initialize the sparse bipartite encoder.

4.1.2 Sparse Matching and Competition Modeling. The encoder stacks $L _ { \mathrm { e n c } }$ sparse layers. Layer $l \in \{ 1 , . . . , L _ { \mathrm { e n c } } \}$ maps order, driver, and edge states with superscript $l - 1$ to those with superscript �, first summarizing competitors and then evaluating matches under that context. Let $N ( o _ { i } )$ and $N ( d _ { j } )$ denote the candidate neighbors of order � and driver $d _ { j }$

For each edge $e _ { i j }$ incident to driver $d _ { j } , \mathbf { r } _ { i  j } ^ { l } = \mathbf { h } _ { i } ^ { o , l - 1 } + \mathbf { h } _ { i j } ^ { e , l - 1 }$ carries order and edge information. Let $\mathbf { W } _ { q , c } ^ { d , l } , \mathbf { W } _ { k , c } ^ { \dot { d } , l } .$ , and $\mathbf { W } _ { v , c } ^ { d , l }$ be the competition-attention projections. We also use a shared degree embedding matrix $\mathbf { E } _ { \mathrm { d e g } } \in \dot { \mathbb { R } } ^ { ( D _ { \mathrm { m a x } } + 1 ) \times d }$ , where $D _ { \mathrm { m a x } }$ is the maximum retained candidate degree; its indexed row represents neighborhood size. We show one head below, while the implementation concatenates multiple heads. Aggregating $\mathbf { r } _ { i  j } ^ { l }$ over $N ( d _ { j } )$ gives competition information $\mathbf { c } _ { j } ^ { d , l }$

$$
\begin{array} { r l } & { \beta _ { i j } ^ { d , l } = \mathrm { s o f t m a x } _ { o _ { i } \in N ( d _ { j } ) } ( \frac { ( \mathbf { W } _ { q , c } ^ { d , l } \mathbf { h } _ { j } ^ { d , l - 1 } ) ^ { \top } ( \mathbf { W } _ { k , c } ^ { d , l } \mathbf { r } _ { i  j } ^ { l } ) } { \sqrt { d } } ) , } \\ & { \mathbf { c } _ { j } ^ { d , l } = \displaystyle \sum _ { o _ { i } \in N ( d _ { j } ) } \beta _ { i j } ^ { d , l } \mathbf { W } _ { v , c } ^ { d , l } \mathbf { r } _ { i  j } ^ { l } + \mathrm { E } _ { \mathrm { d e g } } [ | N ( d _ { j } ) | ] . } \end{array}\tag{6}
$$

Attention captures neighborhood composition, while the degree embedding preserves its size. Order-side competition information $\mathbf { c } _ { i } ^ { o , l }$ is computed symmetrically from $\mathbf { r } _ { j  i } ^ { l } = \mathbf { \hat { h } } _ { j } ^ { d , l - 1 } + \mathbf { h } _ { i j } ^ { e , l - 1 }$ over $N ( o _ { i } )$

We incorporate competition into each candidate representation as $\mathbf { z } _ { j  i } ^ { l } = \mathbf { \dot { h } } _ { j } ^ { d , l - 1 } + \mathbf { h } _ { i j } ^ { e , l ^ { \ast } 1 } + \mathbf { c } _ { j } ^ { d , l }$ , with $\mathbf { z } _ { i  j } ^ { l }$ defined symmetrically. Using matching-attention projections $\mathbf { W } _ { q , m } ^ { o , l } , \mathbf { W } _ { k , m } ^ { o , l }$ , and $\mathbf { W } _ { v , m } ^ { o , l }$ , their aggregation over drivers produces matching information m $\mathbf { \Delta } _ { i } ^ { o , l }$

$$
\begin{array} { r l } & { \boldsymbol { \alpha } _ { i j } ^ { o , l } = \mathrm { s o f t m a x } _ { d _ { j } \in N ( o _ { i } ) } ( \frac { ( \mathbf { W } _ { q , m } ^ { o , l } \mathbf { h } _ { i } ^ { o , l - 1 } ) ^ { \top } ( \mathbf { W } _ { k , m } ^ { o , l } \mathbf { z } _ { j  i } ^ { l } ) } { \sqrt { d } } ) , } \\ & { \mathbf { m } _ { i } ^ { o , l } = \displaystyle \sum _ { d _ { j } \in N ( o _ { i } ) } \boldsymbol { \alpha } _ { i j } ^ { o , l } \mathbf { W } _ { v , m } ^ { o , l } \mathbf { z } _ { j  i } ^ { l } + \mathbb { E } _ { \deg } [ | N ( o _ { i } ) | ] . } \end{array}\tag{7}
$$

Driver-side matching information $\mathbf { m } _ { j } ^ { d , l }$ is computed symmetrically. Competition attention relates candidates sharing an endpoint; match ing attention relates a node to its counterparts. Both use $\mathbf { E } _ { \mathrm { d e g } }$ to preserve neighborhood size and operate only on candidate edges.

Let LN denote layer normalization, and let FFN<sup>�</sup><sub>�</sub>, FFN<sup>�</sup> , and $\mathrm { F F N } _ { e } ^ { l }$ denote the order-, driver-, and edge-side feed-forward networks in layer �, respectively. Each block includes its residual connection and layer normalization. The order state is therefore updated as

$$
\mathbf { h } _ { i } ^ { o , l } = \mathrm { F F N } _ { o } ^ { l } \left( \mathbf { h } _ { i } ^ { o , l - 1 } + \mathbf { m } _ { i } ^ { o , l } \right) .\tag{8}
$$

The driver state $\mathbf { h } _ { j } ^ { d , l }$ is updated symmetrically from $\mathbf { h } _ { j } ^ { d , l - 1 }$ and $\mathbf { m } _ { j } ^ { d , l }$

The encoder also updates each edge state to retain OD-pair specific information while incorporating matching information from both endpoints. Let ⊙ denote element-wise multiplication and $\tilde { \mathbf { h } } _ { i j } ^ { e , l }$ denote the interaction representation. For candidate edge $e _ { i j } .$ we combine its preceding state with the sum, product, and absolute diference of the order-side and driver-side matching information:

$$
\tilde { \mathbf { h } } _ { i j } ^ { e , l } = \mathbf { h } _ { i j } ^ { e , l - 1 } + \mathbf { m } _ { i } ^ { o , l } + \mathbf { m } _ { j } ^ { d , l } + \mathbf { m } _ { i } ^ { o , l } \odot \mathbf { m } _ { j } ^ { d , l } + \left| \mathbf { m } _ { i } ^ { o , l } - \mathbf { m } _ { j } ^ { d , l } \right| .\tag{9}
$$

The edge-side block produces $\mathbf { h } _ { i j } ^ { e , l } = \mathrm { F F N } _ { e } ^ { l } ( \tilde { \mathbf { h } } _ { i j } ^ { e , l } )$ . After $L _ { \mathrm { e n c } }$ layers, the final pair representation is

$$
\begin{array} { r } { \mathbf { z } _ { i j } = \mathrm { L N } \left( \mathbf { h } _ { i } ^ { o , L } + \mathbf { h } _ { j } ^ { d , L } + \mathbf { h } _ { i j } ^ { e , L } \right) . } \end{array}\tag{10}
$$

The pair representation in Eq. (10) serves as candidate memory for the Business-Aware Utility Learner and State-Aware Pointer Decoder.

## 4.2 Business-Aware Utility Learner

Candidate selection requires reliable representations and explicit business-value guidance. The Business-Aware Utility Learner in Figure 2(b) therefore supervises candidate memory $\mathbf { z } _ { i j }$ with stage-wise service outcomes and estimates utility from these outcomes and value signals. These two functions are complementary: auxiliary supervision improves candidate memory and injects service behavior, while the utility logit directly guides subsequent selection.

4.2.1 Stage-Wise Behavioral Outcome Learning. Because completion alone cannot identify where an unsuccessful service failed, we use the stage-wise outcomes defined in Section 3 for fine-grained supervision.

For OD pair $( o _ { i } , d _ { j } )$ , let $\mathbf { v } _ { i j } = [ \mathbf { z } _ { i j } ; \mathbf { x } _ { i } ^ { o } ; \mathbf { x } _ { j } ^ { d } ; \mathbf { x } _ { i j } ^ { e } ]$ , where $[ ; ]$ denotes concatenation. A multi-task learning (MTL) network predicts DA, PCAA, and DCAA probabilities $\hat { p } _ { i j } ^ { \mathrm { D A } } , \hat { p } _ { i j } ^ { \mathrm { P C A A } }$ , and $\hat { p } _ { i j } ^ { \mathrm { D C A } } \mathbf { \bar { A } }$ under their service-stage dependencies. This supervision injects service behavior into $\mathbf { z } _ { i j }$

4.2.2 Business Utility Estimation. The stage-wise predictions describe whether an OD pair is likely to become an efective service, but not how much business value it may produce. Actual dispatch decisions must also consider pickup time, platform revenue, future value, and other business signals. Let $\mathbf { b } _ { i j }$ denote these business-value fields selected from the raw edge features $\mathbf { x } _ { i j } ^ { e }$ . We concatenate them with the three stage-wise predictions as $\dot { \mathbf { b } } _ { i j } ^ { \mathrm { a l l } } =$ $[ \hat { p } _ { i j } ^ { \mathrm { D A } } ; \hat { p } _ { i j } ^ { \mathrm { P C A A } } ; \hat { p } _ { i j } ^ { \mathrm { D C A A } } ; \mathbf { b } _ { i j } ]$ and use a multi-layer perceptron (MLP) to

estimate the utility logit:

$$
\begin{array} { r } { a _ { i j } = \mathrm { M L P } _ { \mathrm { b i z } } \Big ( \mathbf { b } _ { i j } ^ { \mathrm { a l l } } \Big ) . } \end{array}\tag{11}
$$

The resulting $a _ { i j }$ is a candidate-specific and step-independent utility logit. It provides explicit business-value guidance to the subsequent matching process.

## 4.3 State-Aware Pointer Decoder

Each selection changes both feasibility and the remaining matching opportunities. The State-Aware Pointer Decoder in Figure 2(c) therefore conditions on the initial candidates, selected matching, residual candidates, and matching progress. The initial state anchors the original decision space, while the selected and residual states describe committed decisions and remaining opportunities.

4.3.1 Matching-State Representation. Because the final assignment is a set, we represent its evolving state with three order-invariant sets: initial, selected, and residual candidates. Let $\{ \mathbf { z } _ { r } \} _ { r = 1 } ^ { P _ { t } }$ be the pair representations and $m _ { r } ^ { ( k ) } \in \{ 0 , 1 \}$ the selectability mask at step $k ,$ where $m _ { r } ^ { ( k ) } = 1$ means that $e _ { r }$ remains selectable.

Each matching state is maintained by two state statistics: its representation sum and cardinality. The initial-state statistics are $\begin{array} { r } { \mathbf { S } _ { \mathrm { i n i t } } = \sum _ { r = 1 } ^ { P _ { t } } \mathbf { z } _ { r } } \end{array}$ and $N _ { \mathrm { i n i t } } = P _ { t \cdot } \mathrm { I f } r ^ { ( s ) }$ denotes the candidate selected at step �, the selected-state statistics are $\begin{array} { r } { { \bf S } _ { \mathrm { s e l } } ^ { ( k ) } ~ = ~ \sum _ { s < k } { \bf z } _ { r ^ { ( s ) } } } \end{array}$ and $N _ { \mathrm { s e l } } ^ { ( k ) } = k - 1$ . The residual-state statistics are $\begin{array} { r } { \mathbf { S } _ { \mathrm { r e s } } ^ { ( k ) } = \sum _ { r = 1 } ^ { P _ { t } } m _ { r } ^ { ( k ) } \mathbf { z } _ { r } } \end{array}$ and $\begin{array} { r } { N _ { \mathrm { r e s } } ^ { ( k ) } = \sum _ { r = 1 } ^ { P _ { t } } m _ { r } ^ { ( k ) } } \end{array}$

Dividing each nonempty sum by its cardinality gives $\mathbf { q } _ { \mathrm { i n i t } } ~ =$ $\mathbb { S } _ { \mathrm { i n i t } } / N _ { \mathrm { i n i t } } , \mathbf { \check { q } } _ { \mathrm { s e l } } ^ { ( k ) } = \mathbb { S } _ { \mathrm { s e l } } ^ { ( k ) } / \bar { N _ { \mathrm { s e l } } ^ { ( k ) } }$ , and ${ \mathfrak { q } } _ { \mathrm { r e s } } ^ { ( k ) } = { \mathbb S } _ { \mathrm { r e s } } ^ { ( k ) } / N _ { \mathrm { r e s } } ^ { ( k ) }$ , representing the original batch, selected matching, and remaining opportunities. Before any selection, the selected set is empty, so we set $\mathbf { q } _ { \mathrm { s e l } } ^ { ( k ) }$ to a learnable vector e<sub>empty</sub>.

Set averaging loses cardinality. We restore it with selected and residual ratios $\rho _ { \mathrm { s e l } } ^ { ( k ) } = N _ { \mathrm { s e l } } ^ { ( k ) } / \mathrm { m i n } ( N _ { t } ^ { o } , N _ { t } ^ { d } )$ and $\rho _ { \mathrm { r e s } } ^ { ( k ) } = N _ { \mathrm { r e s } } ^ { ( k ) } / P _ { t }$ , normalized by the maximum matching size and initial candidate count. Because batch size is large and dynamic, we scale two learnable $d -$ dimensional vectors ${ \bf e } _ { s \mathrm { e l } }$ and $\mathbf { e } _ { \mathrm { r e s } }$ rather than use a progress-indexed matrix. The resulting progress encoding and query are

$$
\begin{array} { r l } & { \mathbf { p } _ { \mathrm { p r o g } } ^ { ( k ) } = \rho _ { \mathrm { s e l } } ^ { ( k ) } \mathbf { e } _ { \mathrm { s e l } } + \rho _ { \mathrm { r e s } } ^ { ( k ) } \mathbf { e } _ { \mathrm { r e s } } , } \\ & { \mathbf { q } ^ { ( k ) } = \mathrm { L N } \Big ( \mathbf { W } _ { \mathrm { i n i t } } \mathbf { q } _ { \mathrm { i n i t } } + \mathbf { W } _ { \mathrm { s e l } } \mathbf { q } _ { \mathrm { s e l } } ^ { ( k ) } + \mathbf { W } _ { \mathrm { r e s } } \mathbf { q } _ { \mathrm { r e s } } ^ { ( k ) } + \mathbf { p } _ { \mathrm { p r o g } } ^ { ( k ) } \Big ) . } \end{array}\tag{12}
$$

Here, $\mathbf { W } _ { \mathrm { i n i t } } , \mathbf { W } _ { \mathrm { s e l } } .$ , and $\mathbf { W } _ { \mathrm { r e s } }$ are learnable projections; $\mathbf { p } _ { \mathrm { p r o g } } ^ { ( k ) }$ restores the scale information lost by averaging.

4.3.2 Generative Pointer Distribution. The query $\mathbf { q } ^ { ( k ) }$ is processed by $L _ { \mathrm { d e c } }$ cross-attention layers over the pair representations, producing the decoder state $\mathbf { h } ^ { ( \dot { k } ) }$ . Let $P _ { \mathrm { p t r } }$ denote the number of pointer heads, and let $\mathbf { W } _ { q } ^ { ( p ) }$ and $\mathbf { W } _ { k } ^ { \left( p \right) }$ be the learnable query and key projections of head $\mathcal { P } \cdot$ Each head produces a structural matching score ${ \underset { r } { \neg } } { ( k , p ) }$ ; the final structural score $\bar { s } _ { r } ^ { ( k ) }$ is their mean:

$$
\begin{array} { l } { { \displaystyle s _ { r } ^ { ( k , p ) } = \frac { ( \mathbf { W } _ { q } ^ { ( p ) } \mathbf { h } ^ { ( k ) } ) ^ { \top } ( \mathbf { W } _ { k } ^ { ( p ) } \mathbf { z } _ { r } ) } { \sqrt { d } } } , } \\ { { \displaystyle \bar { s } _ { r } ^ { ( k ) } = \frac { 1 } { P _ { \mathrm { p t r } } } \sum _ { p = 1 } ^ { P _ { \mathrm { p t r } } } s _ { r } ^ { ( k , p ) } } . } \end{array}\tag{13}
$$

Let $\mathcal { R } ^ { ( k ) } = \{ r \mid m _ { r } ^ { ( k ) } = 1 \}$ denote the indices of feasible candidates. For candidate $e _ { r } ,$ let $a _ { r }$ denote its utility logit; specifically, $a _ { r } = a _ { i j }$ when $e _ { r } = e _ { i j }$ . We denote its conditional generation probability at step � by ${ p } _ { r } ^ { ( k ) }$ . The decoder combines the step-dependent structural score with this step-independent utility logit:

$$
\hat { p } _ { r } ^ { ( k ) } = \frac { \exp ( \bar { s } _ { r } ^ { ( k ) } + a _ { r } ) } { \sum _ { r ^ { \prime } \in \mathcal { R } ^ { ( k ) } } \exp ( \bar { s } _ { r ^ { \prime } } ^ { ( k ) } + a _ { r ^ { \prime } } ) } , \quad r \in \mathcal { R } ^ { ( k ) } .\tag{14}
$$

Here, $a _ { r }$ captures business utility, while $\bar { s } _ { r } ^ { ( k ) }$ measures compatibility with the current matching state.

At inference step �, the decoder performs greedy edge selection by choosing the candidate with the highest generation probability:

$$
r ^ { ( k ) } = \arg \operatorname* { m a x } _ { r \in \mathcal { R } ^ { ( k ) } } p _ { r } ^ { ( k ) } , \qquad e ^ { ( k ) } = e _ { r ^ { ( k ) } } .\tag{15}
$$

The selected edge $e ^ { ( k ) }$ is appended to the generated assignment, after which the matching state is updated for the next step.

4.3.3 Matching-State Transition. After selecting $e ^ { ( k ) } = e _ { r ^ { ( k ) } }$ , let the blocked set $\mathcal { B } ^ { ( k ) }$ contain it and all selectable pairs sharing its order or driver. The state statistics update incrementally as

$$
\begin{array} { r l r } & {  { \mathbf { S } _ { \mathrm { s e l } } ^ { ( k + 1 ) } = \mathbf { S } _ { \mathrm { s e l } } ^ { ( k ) } + \mathbf { z } _ { r ^ { ( k ) } } , } } \\ & { } & { N _ { \mathrm { s e l } } ^ { ( k + 1 ) } = N _ { \mathrm { s e l } } ^ { ( k ) } + 1 , } \\ & { } & { \mathbf { S } _ { \mathrm { r e s } } ^ { ( k + 1 ) } = \mathbf { S } _ { \mathrm { r e s } } ^ { ( k ) } - \sum _ { r \in \mathcal { B } ^ { ( k ) } } \mathbf { z } _ { r } , } \\ & { } & { N _ { \mathrm { r e s } } ^ { ( k + 1 ) } = N _ { \mathrm { r e s } } ^ { ( k ) } - | \mathcal { B } ^ { ( k ) } | . } \end{array}\tag{16}
$$

We also set $m _ { r } ^ { ( k + 1 ) } = 0 \mathrm { f o r } r \in \mathcal { B } ^ { ( k ) }$ and retain all other mask values, enforcing one-to-one matching.

## 4.4 Model Training

4.4.1 Target Sequence Construction. Rather than distill production decisions, we construct supervision from the completed OD-pair set $C _ { t }$ , whose pairs were answered, not canceled, and completed.

Let $K _ { t } ^ { * } = | C _ { t } |$ |. Because assignment order has no business meaning, each training step uniformly permutes $C _ { t } .$ We denote the resulting target sequence by $\boldsymbol { y } _ { t } ^ { * } = ( e _ { t , * } ^ { ( 1 ) } , \ldots , e _ { t , * } ^ { ( K _ { t } ^ { * } ) } ,$ . This randomization prevents the decoder from fitting an arbitrary order.

Only completed pairs form targets, but all selectable candidates remain in the denominator of Eq. (14) as contrastive alternatives. Thus, GenMatch learns completion patterns from observed targets. At inference, generation continues while $\begin{array} { r } { \sum _ { r = 1 } ^ { P _ { t } } m _ { r } ^ { ( k ) } > 0 . } \end{array}$ , allowing GenMatch to select unbroadcast candidates exhibiting similar patterns.

4.4.2 Training Objective. For assignment generation, we apply teacher forcing and average the negative log-likelihood over all valid target positions in a mini-batch. Let I denote the dispatch batches in the mini-batch and $\textstyle Z _ { \mathrm { g e n } } = \sum _ { t \in \mathcal { I } } K _ { t } ^ { * }$ the number of valid target positions. The generation loss is

$$
\mathcal { L } _ { \mathrm { g e n } } = - \frac { 1 } { Z _ { \mathrm { g e n } } } \sum _ { t \in J } \sum _ { k = 1 } ^ { K _ { t } ^ { * } } \log \mathrm { P r } \Big ( e _ { t , * } ^ { ( k ) } \mid \mathcal { G } _ { t } ^ { ( k ) } , e _ { t , * } ^ { ( < k ) } \Big ) .\tag{17}
$$

For service-process learning, let $\mathcal { T } = \{ \mathrm { D A } , \mathrm { P C A A } , \mathrm { D C A A } \}$ be the auxiliary-task set, and let BCE(·, ·) denote binary cross-entropy.

For task $m \in \mathcal { T } , y _ { t r } ^ { m }$ is the event label of candidate $e _ { r }$ in batch �, and $\hat { p } _ { t r } ^ { m }$ is its predicted event probability. We define $w _ { t r } = 1 \mathrm { i f } e _ { r }$ was broadcast and its service-process feedback is observable, and $w _ { t r } = 0$ otherwise. Let $\begin{array} { r } { Z _ { \mathrm { o b s } } = \sum _ { t \in \mathcal { T } } \sum _ { r = 1 } ^ { P _ { t } } w _ { t r } } \end{array}$ be the number of candidates with observed feedback. The auxiliary loss is

$$
\mathcal { L } _ { \mathrm { m t l } } = \frac { 1 } { Z _ { \mathrm { o b s } } } \sum _ { t \in \cal { I } } \sum _ { r = 1 } ^ { P _ { t } } w _ { t r } \sum _ { m \in \cal { T } } \lambda _ { m } \mathrm { B C E } \left( \hat { p } _ { t r } ^ { m } , y _ { t r } ^ { m } \right) ,\tag{18}
$$

where $\lambda _ { m }$ controls the relative weight of task �. Let $\lambda _ { \mathrm { m t l } }$ balance service-process learning against assignment generation. The final objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { g e n } } + \lambda _ { \mathrm { m t l } } \mathcal { L } _ { \mathrm { m t l } } . } \end{array}\tag{19}
$$

Both losses are normalized by valid supervision units. We set $\lambda _ { \mathrm { m t l } } =$ 10 based on the sensitivity analysis in Appendix A.5. Appendix B.2 provides the complete training and inference procedures.

## 5 Experiments

## 5.1 Experimental Settings

Datasets. The experiments cover five cities across DiDi’s international ride-hailing markets. Ofline evaluation uses City I–III, while online A/B tests use City III–V; City III appears in both settings. The upstream retrieval system is fixed, so all methods receive candidate sets constructed under the same feasibility rules. Detailed city statistics are provided in Appendix A.1.

Ofline evaluation uses simulation environments constructed from historical dispatch logs, with 14 days for training and 7 days for evaluation. Each simulator replays real order requests, driver states, and candidate connections and advances according to the generated decisions.

Online A/B tests run under live production trafic for 14 days using a 1-hour time-slice interleaved design. The Production Dispatching Pipeline (PDP) is the deployed multi-stage baseline; its Kuhn–Munkres variant, denoted $\mathrm { P D P } _ { \mathrm { K M } }$ , serves as the control. Results are reported as treatment–control (T–C) deltas.

Metrics. We use Answer Ratio (AR), Completion Ratio (CR), Average Pickup Time (APT), and Gross Merchandise Volume (GMV). Higher AR, CR, and GMV and lower APT indicate better performance. Detailed metric definitions are provided in Appendix A.1. Baselines. We compare GenMatch with $\mathrm { P D P } _ { \mathrm { K M } }$ and two solver variants, $\mathrm { P D P _ { G r e e d y } }$ and $\mathrm { P D P _ { G S } }$ , where GS denotes Gale–Shapley matching [4]; end-to-end MICOD method D2SN [34]; value-based methods V1D3 [22] and RLW [18]; and multi-agent methods CoRide [9] and CoopRide [25]. We also evaluate $\mathrm { G e n M a t c h } _ { \mathrm { V a l u e } }$ to isolate pairlevel prediction. Detailed definitions are provided in Appendix A.1. Implementation. All ofline results are averaged over five independent runs. Hardware, optimization, and model configurations are provided in Appendix B.1.

## 5.2 Ofline Performance

Table 1 shows that GenMatch consistently outperforms the production and research baselines. Relative to $\mathrm { P D P } _ { \mathrm { K M } }$ , it improves AR, CR, and GMV by 0.31%–0.83%, 0.23%–1.17%, and 0.11%–0.55%, respectively, while reducing APT by 0.23%–0.72%. D2SN also generates assignments sequentially but remains inferior on every metric, indicating that generation alone is insuficient without explicit batch first-step generation logits as fixed Kuhn–Munkres weights and reduces AR and CR by 2.46% and 3.07%, showing that static scores cannot replace state-aware sequential generation. Appendix A.2 reports the remaining variants and complete three-city results.

Table 1: Ofline performance relative to $\mathbf { P D P } _ { \mathrm { K M } }$ . Values are the mean ± standard deviation of percentage changes over five runs; bold and underline denote the best and second-best results.
<table><tr><td rowspan="2">Variant</td><td colspan="4">City I</td><td colspan="4">City II</td><td colspan="4">City III</td></tr><tr><td>AR(%) ↑</td><td>CR(%)↑</td><td> $\mathbf { A P T } \left( \% \right) \downarrow$ </td><td> $\mathbf { G M V } ( \% ) \uparrow$ </td><td>AR (%) ↑</td><td>CR(%)↑</td><td>APT(%)↓</td><td> $\underline { { \mathbf { G M V } \left( \% \right) \uparrow } }$ </td><td>AR(%) ↑</td><td>CR(%)↑</td><td> $\mathbf { A P T } \left( { \% } \right) \downarrow$ </td><td>GMV (%) ↑</td></tr><tr><td> $\overline { { \mathrm { P D P } _ { \mathrm { K M } } } }$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td> $\overline { { 0 . 0 0 \pm 0 . 0 0 } }$ </td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td> $\overline { { 0 . 0 0 \pm 0 . 0 0 } }$ </td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td>0.00 ± 0.00</td></tr><tr><td> $\mathrm { P D P _ { G r e e d y } }$ </td><td></td><td> $- 1 . 8 6 \pm 0 . 0 5 - 1 . 6 9 \pm 0 . 0 9 - 0 . 2 4 \pm 0 . 0 2 - 0 . 1 4 \pm 0 . 0 1$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 9 7 \pm 0 . 0 6 - 1 . 2 6 \pm 0 . 0 4 \frac { - 0 . 2 5 \pm 0 . 0 1 } { - 0 . 2 5 \pm 0 . 0 1 } - 1 . 1 6 \pm 0 . 0 6$ </td><td></td><td></td><td></td><td></td><td> $- 1 . 9 2 \pm 0 . 1 5 - 2 . 3 5 \pm 0 . 1 8 - 0 . 3 1 \pm 0 . 0 2 - 0 . 1 9 \pm 0 . 0 2$ </td></tr><tr><td>PDPGS</td><td></td><td> $- 0 . 4 2 \pm 0 . 0 2 \ - 0 . 3 0 \pm 0 . 0 3 \ - 0 . 0 6 \pm 0 . 0 1 \ - 0 . 0 4 \pm 0 . 0 0$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 3 8 \pm 0 . 0 2 \ - 0 . 5 4 \pm 0 . 0 2 \ + 0 . 1 0 \pm 0 . 0 1 \ - 0 . 4 7 \pm 0 . 0 4$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 7 9 \pm 0 . 0 6 - 0 . 7 6 \pm 0 . 0 7 + 0 . 1 4 \pm 0 . 0 1 - 0 . 2 7 \pm 0 . 0 2$ </td></tr><tr><td>D2SN</td><td></td><td> $- 0 . 0 9 \pm 0 . 0 3 \ + 0 . 1 8 \pm 0 . 0 7 \ - 0 . 3 6 \pm 0 . 0 2 \ - 0 . 0 5 \pm 0 . 0 0$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 0 3 \pm 0 . 0 9 \ - 0 . 1 9 \pm 0 . 1 1 \ - 0 . 1 2 \pm 0 . 0 1 \ - 0 . 4 7 \pm 0 . 0 6$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 3 7 \pm 0 . 1 2 \ - 0 . 4 4 \pm 0 . 0 7 \ - 0 . 0 3 \pm 0 . 0 1 \ - 0 . 2 4 \pm 0 . 0 4$ </td></tr><tr><td>RLW</td><td></td><td> $- 0 . 7 8 \pm 0 . 0 4 - 0 . 5 0 \pm 0 . 0 5 + 0 . 4 2 \pm 0 . 0 2 - 0 . 0 2 \pm 0 . 0 0$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 2 4 \pm 0 . 0 3 - 0 . 4 8 \pm 0 . 0 2 \ + 0 . 3 4 \pm 0 . 0 3 - 0 . 2 7 \pm 0 . 0 3$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 6 8 \pm 0 . 0 8 - 0 . 7 1 \pm 0 . 0 8 + 0 . 2 2 \pm 0 . 0 3 - 0 . 1 6 \pm 0 . 0 2$ </td></tr><tr><td>V1D3</td><td></td><td> $- 1 . 0 7 \pm 0 . 0 3 - 0 . 8 1 \pm 0 . 0 7 + 0 . 2 9 \pm 0 . 0 1 - 0 . 0 6 \pm 0 . 0 1$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 3 9 \pm 0 . 0 4 - 0 . 6 2 \pm 0 . 0 3 + 0 . 2 6 \pm 0 . 0 2 - 0 . 5 7 \pm 0 . 0 5$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 9 3 \pm 0 . 1 0 - 1 . 0 6 \pm 0 . 1 1 + 0 . 1 8 \pm 0 . 0 2 - 0 . 1 8 \pm 0 . 0 2$ </td></tr><tr><td>CoRide</td><td> $- 1 . 5 1 \pm 0 . 0 9 \ - 1 . 2 7 \pm 0 . 1 4 \ + 0 . 9 8 \pm 0 . 0 3 \ - 0 . 1 3 \pm 0 . 0 2$ </td><td></td><td></td><td></td><td></td><td></td><td> $- 0 . 7 2 \pm 0 . 0 8 - 1 . 0 2 \pm 0 . 1 1 + 0 . 6 7 \pm 0 . 0 3 - 0 . 9 6 \pm 0 . 0 6$ </td><td></td><td>-2.17 ± 0.21−2.54 ± 0.23+0.47 ± 0.04−0.43 ± 0.04</td><td></td><td></td><td></td></tr><tr><td>CoopRide</td><td></td><td> $- 1 . 1 3 \pm 0 . 0 7 \ - 0 . 9 3 \pm 0 . 1 1 \ + 0 . 7 7 \pm 0 . 0 3 \ - 0 . 1 0 \pm 0 . 0 1$ </td><td></td><td></td><td></td><td></td><td> $- 0 . 5 5 \pm 0 . 0 6 - 0 . 7 4 \pm 0 . 0 9 + 0 . 5 8 \pm 0 . 0 2 - 0 . 7 6 \pm 0 . 0 5 { \sqrt { } }$ </td><td></td><td>−1.77 ± 0.14 −1.33 ± 0.13 +0.43 ± 0.04 -</td><td></td><td></td><td> $- 0 . 3 3 \pm 0 . 0 3$ </td></tr><tr><td>GenMatchyalue</td><td></td><td> $\pm 0 . 2 9 \pm 0 . 0 2 \pm 0 . 2 0 \pm 0 . 0 3 - 0 . 7 6 \pm 0 . 0 2 - 0 . 0 2 \pm 0 . 0 0$ </td><td></td><td></td><td></td><td></td><td> $+ 0 . 1 4 \pm 0 . 0 2 \ - 0 . 0 9 \pm 0 . 0 1 \ - 0 . 1 9 \pm 0 . 0 1 \ \pm 0 . 0 3 \pm 0 . 0 1$ </td><td></td><td> $+ 0 . 4 6 \pm 0 . 0 5 \pm 0 . 6 7 \pm 0 . 0 6 - 0 . 5 8 \pm 0 . 0 5 \pm 0 . 2 4 \pm 0 . 0 2$ </td><td></td><td></td><td></td></tr><tr><td>GenMatch</td><td></td><td></td><td></td><td> $+ 0 . 5 1 \pm 0 . 0 3 + 0 . 6 2 \pm 0 . 0 4 \underbrace { - 0 . 7 2 \pm 0 . 0 3 } _ { \textnormal { d } } + 0 . 1 1 \pm 0 . 0 1$ </td><td></td><td></td><td>+0.31 ± 0.03 +0.23 ± 0.02 −0.40 ± 0.03 +0.23 ± 0.03</td><td></td><td> $+ 0 . 8 3 \pm 0 . 0 7 + 1 . 1 7 \pm 0 . 1 2 - 0 . 2 3 \pm 0 . 0 3 + 0 . 5 5 \pm 0 . 0 3$ </td><td></td><td></td><td></td></tr></table>

Table 2: Core City III ablations relative to GenMatch (Full), reported as the mean ± standard deviation of percentage changes over five runs. Bold denotes the best result in each column.
<table><tr><td>Module</td><td>Variant</td><td>AR↑</td><td>CR↑</td><td>APT↓</td><td>GMV↑</td></tr><tr><td>GenMatch</td><td>Full</td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td rowspan="3">Encoder</td><td>A1</td><td> $- 2 . 5 5 \pm 0 . 1 2$ </td><td> $- 3 . 0 7 \pm 0 . 0 7$ </td><td> $+ 1 . 1 6 \pm 0 . 0 2$ </td><td> $- 2 . 2 8 \pm 0 . 1 5$ </td></tr><tr><td>A2</td><td> $- 2 . 4 3 \pm 0 . 0 6$ </td><td> $- 2 . 6 9 \pm 0 . 0 3$ </td><td> $+ 1 . 1 4 \pm 0 . 0 1$ </td><td> $- 2 . 1 5 \pm 0 . 0 6$ </td></tr><tr><td>A3</td><td> $- 0 . 3 9 \pm 0 . 0 6$ </td><td> $- 0 . 2 5 \pm 0 . 0 3$ </td><td> $+ 0 . 1 8 \pm 0 . 0 2$ </td><td> $- 0 . 4 0 \pm 0 . 0 7$ </td></tr><tr><td rowspan="2">Learner</td><td>A4</td><td> $- 2 . 7 6 \pm 0 . 1 6$ </td><td> $- 3 . 1 5 \pm 0 . 0 8$ </td><td> $+ 1 . 1 7 \pm 0 . 0 3$ </td><td> $- 2 . 3 2 \pm 0 . 1 8$ </td></tr><tr><td>A5</td><td> $- 2 . 0 2 \pm 0 . 0 8$ </td><td> $- 2 . 3 4 \pm 0 . 0 4$ </td><td> $+ 1 . 5 3 \pm 0 . 0 2$ </td><td> $- 1 . 8 8 \pm 0 . 0 9$ </td></tr><tr><td rowspan="3">Decoder</td><td>A8</td><td>1  $- 1 . 1 1 \pm 0 . 0 4$ </td><td> $\cdot 1 . 0 7 \pm 0 . 0 2$  一</td><td> $+ 0 . 8 6 \pm 0 . 0 1$ </td><td>一  $1 . 4 5 \pm 0 . 0 5$ </td></tr><tr><td>A9</td><td> $- 1 . 2 2 \pm 0 . 1 4$ </td><td> $- 0 . 9 6 \pm 0 . 0 9$ </td><td> $+ 0 . 4 8 \pm 0 . 0 4$ </td><td> $- 0 . 7 8 \pm 0 . 1 9$ </td></tr><tr><td>A12</td><td> $- 2 . 4 6 \pm 0 . 1 5$ </td><td> $- 3 . 0 7 \pm 0 . 2 0$ </td><td> $+ 0 . 1 6 \pm 0 . 0 3$ </td><td> $- 0 . 9 1 \pm 0 . 0 5$ </td></tr></table>

interactions, business guidance, and supervised outcome learning. The value-based and multi-agent baselines likewise cannot consistently improve the current batch assignment.

$\mathrm { G e n M a t c h } _ { \mathrm { V a l u e } } ,$ which replaces only PDP’s prediction stage, im proves most metrics and validates the learned service-process predictions. Full GenMatch further improves AR, CR, and GMV by replacing value calculation and matching with state-aware generation, directly supporting our cross-stage inconsistency motivation. Appendix A.3 further evaluates the auxiliary predictions.

## 5.3 Ablation Study

Table 2 reports core ablations on City III, organized by module to test whether each targeted design contributes to the final assignment. For the encoder, A1 removes batch-level matching and competition information, while A2 restores matching information alone. A2 improves AR, CR, and GMV over A1, and full GenMatch further improves all metrics, validating both forms of context. Removing the shared degree embedding (A3) also degrades every metric, showing that neighborhood size complements attention-based neighbor composition.

For the learner, removing auxiliary supervision (A4) causes the largest AR and CR drops and reduces GMV by 2.32%, while removing the utility logit (A5) produces the largest APT increase; both representation supervision and business guidance are therefore necessary. For the decoder, removing residual-state or progress information (A8–A9) consistently hurts performance. A12 uses the

## 5.4 Online A/B Testing

Overall online performance. Table 3 shows significant improvements in every city. Overall, GenMatch increases AR, CR, and GMV by 2.26%, 3.86%, and 2.97%, respectively, while reducing APT by 1.84%. GenMatch<sub>Value</sub> also improves all four metrics over PDP<sub>KM</sub>, showing that the service-process predictions learned by GenMatch provide more efective pair-level inputs to the production pipeline. Full GenMatch further improves AR, CR, and GMV over GenMatch<sub>Value</sub> by 1.49%, 1.93%, and 1.48% overall. The ofline and online results therefore show the same pattern: better pair-level predictions help, while end-to-end generation yields the strongest overall gains. This consistency supports our cross-stage objective inconsistency motivation and shows that the ofline optimization transfers to live trafic.

Dispatch efectiveness and user experience. Table 4 provides a finer-grained view of the dispatch funnel and service quality. Broadcast Count decreases by 0.17%, while Answer Count and Completion Count increase by 2.16% and 3.84%, respectively. This result is consistent with the greedy autoregressive policy of Gen-Match: it suppresses broadcasts that are unlikely to yield an answer or completion and generates more efective assignments. All passenger and driver experience metrics also improve. PBE decreases by 15.17%, PCBA and PCAA decrease by 9.28% and 7.61%, Driver Income and DA increase by 2.99% and 13.96%, and DCAA decreases by 6.99%. Passenger and driver willingness is highly uncertain in international markets. The conventional paradigm predicts these signals separately and then combines them through a hand-crafted value function, so errors can propagate across stages. GenMatch instead learns from heterogeneous behavioral feedback and uses it to guide the final assignment directly, improving both dispatch efectiveness and user experience.

Performance across supply–demand periods. Figure 3 shows that GenMatch remains efective under diferent supply–demand conditions and delivers larger gains in busier periods. From lowdemand to peak-demand periods, the CR improvement rises from 3.24% to 4.12%. The reductions in PCAA and DCAA also expand from 1.76% and 3.48% to 8.26% and 7.96%, respectively. This trend is particularly notable because the generation targets contain only OD pairs from rides that were actually completed online, as described in Section 4. Unbroadcast candidates have no observed service-process labels and are excluded from auxiliary supervision, but remain part of the complete dispatch batch and the generative candidate set. GenMatch can therefore transfer the completion patterns learned from completed pairs to previously unbroadcast candidates with similar structural and business characteristics. This ability becomes more valuable in peak periods, where denser competition leaves more latent completion opportunities unexplored, and converts them into additional completed rides and business gains.

Table 3: Online A/B test improvements over $\mathbf { P D P } _ { \mathrm { K M } }$ (T−C). Overall averages the three cities; <sup>∗</sup> indicates $\begin{array} { r } { p < 0 . 0 5 . } \end{array}$
<table><tr><td rowspan="2">Variant</td><td colspan="4">City III</td><td colspan="4"></td><td colspan="4">City V</td><td colspan="4">Overall</td></tr><tr><td>AR↑</td><td>CR↑</td><td>APT↓</td><td>GMV↑</td><td>AR↑</td><td>CR↑</td><td>APT↓</td><td>GMV↑</td><td>AR↑</td><td>CR↑</td><td>APT↓</td><td>GMV↑</td><td>AR↑</td><td>CR↑</td><td>APT↓</td><td>GMV↑</td></tr><tr><td>GenMatchyalue</td><td>0.88%</td><td>2.31%*</td><td>-7.08%*</td><td>2.35%*</td><td>0.59%</td><td>3.06%*</td><td>-4.55%*</td><td>2.43%*</td><td>0.75%*</td><td>1.67%*</td><td>-4.05%*</td><td>1.04%</td><td>0.77%*</td><td>1.93%*</td><td>-4.85%*</td><td>1.49%*</td></tr><tr><td>GenMatch</td><td>3.18%*</td><td>5.37%*</td><td>-2.20%*</td><td>4.89%*</td><td>1.72%*</td><td>4.51%*</td><td>-1.68%*</td><td>3.93%*</td><td>2.01%*</td><td>3.26%*</td><td>-1.76%*</td><td>2.16%*</td><td>2.26%*</td><td>3.86%*</td><td>-1.84%*</td><td>2.97%*</td></tr></table>

Table 4: Online changes in dispatch efectiveness and experience relative to $\mathbf { P D P } _ { \mathrm { K M } }$ (T−C); <sup>∗</sup> indicates $\begin{array} { r } { p < 0 . 0 5 . } \end{array}$
<table><tr><td>Metric</td><td>Delta (T-C)</td></tr><tr><td>Dispatch Effectiveness Measures</td><td></td></tr><tr><td>Broadcast Count</td><td>-0.17%*</td></tr><tr><td>Answer Count (↑)</td><td>2.16%*</td></tr><tr><td>Completion Count (↑)</td><td>3.84%*</td></tr><tr><td>Passenger Experience Measures</td><td></td></tr><tr><td>Passenger Bad Experience Ratio (PBE) (↓)</td><td>-15.17%*</td></tr><tr><td>Passenger Cancel Before Answer Ratio (PCBA) (↓)</td><td>-9.28%*</td></tr><tr><td>Passenger Cancel After Answer Ratio (PCAA) (↓)</td><td>-7.61%*</td></tr><tr><td>Driver Experience Measures</td><td></td></tr><tr><td>Driver Income (↑)</td><td>2.99%*</td></tr><tr><td>Driver Answer Ratio (DA) (↑)</td><td>13.96%*</td></tr><tr><td>Driver Cancel After Answer Ratio (DCAA) (↓)</td><td>-6.99%*</td></tr></table>

![](images/cba2d48ad9424ac1dc6454ad6ea44aa6b92a24338473527e4375675883d615c1.jpg)  
Figure 3: Online gains of GenMatch across supply–demand periods (T−C vs. $\mathbf { P D P } _ { \mathrm { K M } } )$ .

## 6 Deployment

The existing Pair-Level Dispatch Engine processes OD pairs independently and therefore cannot provide the complete dispatch batch required by GenMatch. We develop a Batch-Level Generative Dispatch Engine for production serving. As shown in Figure 4, its control plane coordinates distributed shards, restores a consistent candidate order, and assembles the complete batch. Its compute plane retains distributed feature extraction and candidate retrieval, then performs global GenMatch inference over the assembled batch. Generated pairs are treated as pre-assignments and enter the existing arbitration and locking process, preventing conflicts with other product lines. Malformed outputs, timeouts, or serving failures automatically fall back to the pair-level engine, allowing GenMatch to be deployed without weakening production reliability. Appendix B.3 provides the engineering details.

![](images/a5eb33689afd96617b712657f0a93e4015bb72da25fd2024e516af7e055ffbdf.jpg)  
Figure 4: Production architecture of the Batch-Level Generative Dispatch Engine.

## 7 Conclusion

This paper presents GenMatch, an end-to-end generative framework deployed in a real-world production environment for MICOD. By directly generating the final assignment, GenMatch addresses the cross-stage objective inconsistency of the conventional multistage paradigm. Its Context-Aware Bipartite Encoder, Business-Aware Utility Learner, and State-Aware Pointer Decoder address the three challenges of encoding dynamic batch-level structures, learning unified business utility from heterogeneous feedback, and generating assignments under an evolving matching state, respectively. Extensive ofline experiments validate these targeted designs, while online A/B tests demonstrate the efectiveness and practicality of GenMatch in production.

## References

[1] Ben Chen, Xian Guo, Siyuan Wang, Zihan Liang, Yue Lv, Yufei Ma, Xinlong Xiao, Bowen Xue, Xuxin Zhang, Ying Yang, et al. 2025. Onesearch: A preliminary exploration of the unified end-to-end generative framework for e-commerce search. arXiv preprint arXiv:2509.03236 (2025).

[2] Ben Chen, Siyuan Wang, Yufei Ma, Zihan Liang, Xuxin Zhang, Yue Lv, Ying Yang, Huangyu Dai, Lingtao Mao, Tong Zhao, et al. 2026. OneSearch-V2: The Latent Reasoning Enhanced Self-distillation Generative Search Framework. arXiv preprint arXiv:2603.24422 (2026).

[3] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. 2025. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965 (2025).

[4] David Gale and Lloyd S Shapley. 1962. College admissions and the stability of marriage. The American mathematical monthly 69, 1 (1962), 9–15.

[5] Xian Guo, Ben Chen, Siyuan Wang, Ying Yang, Mingyue Cheng, Chenyi Lei, Yuqing Ding, and Han Li. 2026. Onesug: The unified end-to-end generative framework for e-commerce query suggestion. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 14774–14782.

[6] Ruidong Han, Bin Yin, Shangyu Chen, He Jiang, Fei Jiang, Xiang Li, Chi Ma, Mincong Huang, Xiaoguang Li, Chunzhen Jing, et al. 2025. Mtgr: Industrial scale generative recommendation framework in meituan. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management. 5731–5738.

[7] Xiao Han, Zijian Zhang, Xiangyu Zhao, Yuanshao Zhu, Guojiang Shen, Xiangjie Kong, Xuetao Wei, Liqiang Nie, and Jieping Ye. 2025. Garlic: Gpt-augmented rein forcement learning with intelligent control for vehicle dispatching. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 255–263.

[8] Yupeng Hou, Jiacheng Li, Ashley Shin, Jinsung Jeon, Abhishek Santhanam, Wei Shao, Kaveh Hassani, Ning Yao, and Julian McAuley. 2025. Generating long semantic ids in parallel for recommendation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 956–966.

[9] Jiarui Jin, Ming Zhou, Weinan Zhang, Minne Li, Zilong Guo, Zhiwei Qin, Yan Jiao, Xiaocheng Tang, Chenxi Wang, Jun Wang, et al. 2019. Coride: joint order dispatching and fleet management for multi-scale ride-hailing platforms. In Proceedings of the 28th ACM international conference on information and knowledge management. 1983–1992.

[10] Harold W Kuhn. 1955. The Hungarian method for the assignment problem. Naval research logistics quarterly 2, 1-2 (1955), 83–97.

[11] Changhao Li, Junwei Yin, Zhilin Zeng, Senjie Kou, Shuli Wang, Wenshuai Chen, Yinhua Zhu, Haitao Wang, and Xingxing Wang. 2026. MBGR: Multi Business Prediction for Generative Recommendation at Meituan. arXiv preprint arXiv:2604.02684 (2026).

[12] Hanyu Li, Yi-Ping Hsu, Aditya Mantha, Prabhat Agarwal, Laksh Bhasin, Jialu Wang, Hongtao Lin, Bella Huang, Yaxin Li, Xinyi Li, et al. 2026. UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale. arXiv preprint arXiv:2606.00422 (2026).

[13] Yile Liang, Jiuxia Zhao, Donghui Li, Jie Feng, Chen Zhang, Xuetao Ding, Jinghua Hao, and Renqing He. 2024. Harvesting eficient on-demand order pooling from skilled couriers: Enhancing graph representation learning for refining real-time many-to-one assignments. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 5363–5374.

[14] James Munkres. 1957. Algorithms for the assignment and transportation prob lems. Journal of the society for industrial and applied mathematics 5, 1 (1957), 32–38.

[15] Zhiwei Qin, Xiaocheng Tang, Yan Jiao, Fan Zhang, Zhe Xu, Hongtu Zhu, and Jieping Ye. 2020. Ride-hailing order dispatching at didi via reinforcement learning. INFORMS Journal on Applied Analytics 50, 5 (2020), 272–286.

[16] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299–10315.

[17] Yuxin Ren, Qiya Yang, Yichun Wu, Wei Xu, Yalong Wang, and Zhiqiang Zhang. 2024. Non-autoregressive generative models for reranking recommendation. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 5625–5634.

[18] Soheil Sadeghi Eshkevari, Xiaocheng Tang, Zhiwei Qin, Jinhan Mei, Cheng Zhang, Qianying Meng, and Jia Xu. 2022. Reinforcement learning in the wild: Scalable RL dispatching algorithm deployed in ridehailing marketplace. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 3838–3848.

[19] Jaidev Shah, Iman Barjasteh, Amey Barapatre, Rana Forsati, Gang Luo, Fan Wu, Yuan Fang, Xue Deng, Blake Shepard, Ronak Shah, et al. 2025. Towards Webscale Recommendations with LLMs: From Quality-aware Ranking to Candidate Generation. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2514–2524.

[20] Yu Sun, Kai Wang, Zhipeng Hu, Runze Wu, Yaoxin Wu, Wen Song, Xudong Shen, Tangjie Lv, and Changjie Fan. 2024. MGMatch: Fast Matchmaking with Nonlinear Objective and Constraints via Multimodal Deep Graph Learning. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 5741–5751.

[21] Xiaocheng Tang, Zhiwei Qin, Fan Zhang, Zhaodong Wang, Zhe Xu, Yintai Ma, Hongtu Zhu, and Jieping Ye. 2019. A deep value-network based approach for multi-driver order dispatching. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 1780–1790.

[22] Xiaocheng Tang, Fan Zhang, Zhiwei Qin, Yansheng Wang, Dingyuan Shi, Bingchen Song, Yongxin Tong, Hongtu Zhu, and Jieping Ye. 2021. Value func tion is all you need: A unified learning framework for ride hailing platforms. In Proceedings ofthe 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 3605–3615.

[23] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. 2015. Pointer networks. Advances in neural information processing systems 28 (2015).

[24] Dongsheng Wang, Yuxi Huang, Shen Gao, Yifan Wang, Chengrui Huang, and Shuo Shang. 2025. Generative next poi recommendation with semantic id. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 2904–2914.

[25] Jingwei Wang, Qianyue Hao, Wenzhen Huang, Xiaochen Fan, Qin Zhang, Zhentao Tang, Bin Wang, Jianye Hao, and Yong Li. 2025. Coopride: Cooperate all grids in city-scale ride-hailing dispatching with multi-agent reinforcement learning. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 1457–1468.

[26] Yuxiang Wang, Chi Ma, Xiao Yan, Mincong Huang, Xiaoguang Li, Ruidong Han, Bin Yin, Shangyu Chen, Xiang Li, Fei Jiang, et al. 2026. MTGenRec: An Eficient Distributed Training System for Generative Recommendation Models in Meituan. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2482–2493.

[27] Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, et al. 2024. Eager: Two-stream generative recommender with behavior-semantic collaboration. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3245–3254.

[28] Zhipeng Wei, Kuo Cai, Junda She, Jie Chen, Minghao Chen, Yang Zeng, Qiang Luo, Wencong Zeng, Ruiming Tang, Kun Gai, et al. 2026. Oneloc: Geo-aware generative recommender systems for local life service. In Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining. 735–744.

[29] Zhe Xu, Zhixin Li, Qingwen Guan, Dingshui Zhang, Qiang Li, Junxiao Nan, Chunyang Liu, Wei Bian, and Jieping Ye. 2018. Large-scale order dispatch in ondemand ride-hailing platforms: A learning and planning approach. In Proceedings ofthe 24th ACM SIGKDD international conference on knowledge discovery & data mining. 905–913.

[30] Ben Xue, Dan Liu, Lixiang Wang, Mingjie Sun, Peng Wang, Pengfei Zhang, Shaoyun Shi, Tianyu Xu, Yunhao Sha, Zhiqiang Liu, et al. 2026. Generative recommendation for large-scale advertising. arXiv preprint arXiv:2602.22732 (2026).

[31] Zhaoxing Yang, Haiming Jin, Guiyun Fan, Min Lu, Yiran Liu, Xinlang Yue, Hao Pan, Zhe Xu, Guobin Wu, Qun Li, et al. 2024. Rethinking order dispatching in online ride-hailing platforms. In Proceedings ofthe 30th ACM SIGKDD conference on knowledge discovery and data mining. 3863–3873.

[32] Junhao Yin, Haolin Wang, Peng Bao, Ju Xu, and Yongliang Wang. 2026. From clicks to preference: A multi-stage alignment framework for generative query suggestion in conversational system. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2539–2550.

[33] Yuanqing Yu, Yifan Wang, Weizhi Ma, Zhiqiang Guo, and Min Zhang. 2026. APAO: Bridging the Training-Inference Gap in Generative Recommendation via Adaptive Prefix-Aware Optimization. arXiv preprint arXiv:2603.02730 (2026).

[34] Xinlang Yue, Yiran Liu, Fangzhou Shi, Sihong Luo, Chen Zhong, Min Lu, and Zhe Xu. 2024. An end-to-end reinforcement learning based approach for micro-view order-dispatching in ride-hailing. In Proceedings ofthe 33rd ACM international conference on information and knowledge management. 5054–5061.

[35] Ziyang Zeng, Heming Jing, Jindong Chen, Xiangli Li, Hongyu Liu, Yixuan He, Zhengyu Li, Yige Sun, Zheyong Xie, Yuqing Yang, et al. 2026. Optimizing Generative Ranking Relevance via Reinforcement Learning in Xiaohongshu Search. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2551–2561.

[36] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Jiayuan He, et al. 2024. Actions speak louder than words: trillion-parameter sequential transducers for generative recommendations. In Proceedings of the 41st International Conference on Machine Learning. 58484– 58509.

[37] Hongbo Zhang, Guang Wang, Xu Wang, Zhengyang Zhou, Chen Zhang, Zheng Dong, and Yang Wang. 2024. Nondbrem: Nondeterministic ofline reinforcement learning for large-scale order dispatching. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 38. 401–409.

[38] Libin Zheng, Lei Chen, and Jieping Ye. 2018. Order dispatch in price-aware ridesharing. Proceedings ofthe VLDB Endowment 11, 8 (2018), 853–865.

[39] Zhuoxun Zheng, Baifan Zhou, Ahmet Soylu, Jie Tang, and Evgeny Kharlamov. 2026. DiKGRec: Generative Recommender Model with Difusion and Knowledge Graph–Based Reasoning. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2042–2053

[40] Guorui Zhou, Hengrui Hu, Hongtao Cheng, Huanjie Wang, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Lu Ren, Liao Yu, et al. 2025. Onerec-v2 technical report. arXiv preprint arXiv:2508.20900 (2025).

[41] Zhuang Zhuang, Shanshan Feng, Hangwei Qian, Mingqi Yang, Heng Qi, Yanming Shen, and Baocai Yin. 2026. Think2Go: Generative Next POI Recommendation with LLM Reasoning. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2112–2123.

[42] Yanyan Zou, Junbo Qi, Lunsong Huang, Yu Li, Kewei Xu, Jiabao Gao, Binglei Zhao, Xuanhua Yang, Sulong Xu, and Shengjie Li. 2026. GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation. arXiv preprint arXiv:2604.14878 (2026).

## A Additional Experimental Details

## A.1 Experimental Details

A.1.1 Datasets and City Statistics. We evaluate GenMatch in five cities from DiDi’s international ride-hailing markets. City I and City II are used for ofline evaluation, City IV and City V are used for online evaluation, and City III is included in both settings to connect the ofline and online observations. As shown in Table 5, the cities cover substantially diferent operating scales: daily completed orders range from approximately 1.7K to 13.8K, daily online drivers range from 0.18K to 0.68K, and average trip distance ranges from 3.47 km to 4.71 km. This diversity allows us to evaluate the method under diferent market sizes and supply–demand conditions rather than on a single operating environment.

For ofline experiments, we construct city-specific simulation environments from historical dispatch logs. Fourteen days of data are used for training and seven days for evaluation. At every replayed dispatch step, the simulator provides the contemporaneous orders, available drivers, and feasible candidate connections. The upstream retrieval and feasibility-filtering system is fixed for all methods, so every method receives candidate sets generated under the same pickup-distance, service-range, and business constraints. Consequently, the comparison focuses on how each method represents, values, and selects from a common candidate set rather than on diferences in candidate retrieval.

Each dispatch batch is represented as a sparse bipartite graph containing order nodes, driver nodes, and feasible OD-pair edges. An order is described by 113 features, a driver by 140 features, and a candidate edge by 32 features, giving 285 input fields in total. These fields cover spatiotemporal context, historical behavior, supply– demand statistics, and pair-specific business signals such as pickup cost and transaction value. The graph size changes across dispatch steps because the numbers of available orders, drivers, and feasible connections are determined by the current market state.

A.1.2 Metrics. We evaluate dispatch performance using Answer Ratio (AR), Completion Ratio (CR), Average Pickup Time (APT), and Gross Merchandise Volume (GMV). Together, these metrics describe whether an assignment is accepted, whether it is successfully fulfilled, how eficiently the driver reaches the passenger, and how much transaction value it creates.

Answer Ratio (AR) is the percentage of dispatched orders answered by drivers. A larger AR indicates that the selected OD pairs better match driver willingness and that fewer broadcasts are spent on assignments unlikely to receive a response. It therefore measures the immediate efectiveness of the dispatch decision at the first stage of the service process.

Completion Ratio (CR) is the percentage of assigned orders that ultimately complete their trips. Compared with AR, CR additionally reflects cancellations and other failures after an answer. A larger CR indicates that the generated assignments are more likely to survive the complete service process and become successfully fulfilled rides, making it a direct measure of dispatch reliability and service conversion.

Average Pickup Time (APT) measures the average elapsed time from assignment to driver pickup. A smaller APT means that drivers can reach passengers more quickly, reducing passenger waiting time and driver-side pickup cost. APT therefore captures the spatial and operational eficiency of the selected matching rather than only whether the trip is answered or completed.

Gross Merchandise Volume (GMV) is the total transaction value contributed by completed trips. A larger GMV indicates that the dispatch policy produces greater realized business value, jointly reflecting the number of completed rides and their transaction values.

No single metric fully characterizes dispatch quality. For example, aggressively prioritizing nearby candidates may reduce APT without maximizing completion or transaction value, while prioritizing high-value trips alone may increase pickup cost. We therefore assess the four metrics jointly: higher AR, CR, and GMV and lower APT indicate better overall performance.

A.1.3 Baselines. We compare GenMatch with production, end-toend, value-based, and multi-agent dispatch methods. All methods operate on candidate sets produced by the same upstream retrieval and feasibility-filtering system.

Production Dispatching Pipeline (PDP). PDP follows the conventional industrial paradigm: it first predicts business outcomes independently for each OD pair, combines these signals with a hand-crafted value function, and then constructs a batch-level assignment with a separate matching solver. $\mathrm { P D P } _ { \mathrm { K M } }$ is the deployed configuration and uses Kuhn–Munkres matching to optimize the assignment globally under the calculated pair weights. $\mathrm { P D P _ { G r e e d y } }$ and $\mathrm { P D P _ { G S } }$ retain the same upstream predictions and pair values but replace the final solver with greedy selection and Gale–Shapley matching, respectively. Comparing these variants isolates the effect of the matching solver within the conventional multi-stage pipeline.

End-to-End MICOD Baseline. D2SN formulates MICOD as a twolayer Markov decision process and sequentially selects OD pairs or hold actions with an encoder–decoder reinforcement-learning policy. It is the most closely related end-to-end baseline because it also constructs an assignment sequentially. Unlike GenMatch, however, it does not explicitly model candidate interactions on the sparse bipartite graph or use direct multi-task supervision from stage-wise service outcomes.

Value-Based Baselines. V1D3 and RLW improve dispatch decisions by learning or refining value estimates for candidate assignments. They represent approaches that enhance pair-level weights while retaining a value-driven decision pipeline. Their comparison with GenMatch tests whether improved pair values alone are suficient, or whether jointly learning batch context and the final assignment provides additional benefit.

Table 5: Scale statistics of the five experimental cities.
<table><tr><td>Split</td><td>City</td><td>Daily Completed Orders</td><td>Daily Online Drivers</td><td>Avg. Trip Distance (m)</td></tr><tr><td rowspan="2">Offline</td><td>City I</td><td>3.04e+03</td><td>0.18e+03</td><td>3.47e+03</td></tr><tr><td>City II</td><td>1.12e+04</td><td>0.55e+03</td><td>4.41e+03</td></tr><tr><td>Offline &amp; Online</td><td>City III</td><td>5.91e+03</td><td>0.43e+03</td><td>3.99e+03</td></tr><tr><td rowspan="2">Online</td><td>City IV</td><td>1.73e+03</td><td>0.31e+03</td><td>4.29e+03</td></tr><tr><td>City V</td><td>1.38e+04</td><td>0.68e+03</td><td>4.71e+03</td></tr></table>

Table 6: Complete ablation results relative to GenMatch (Full). Values are the mean ± standard deviation of percentage changes over five runs.
<table><tr><td rowspan="2">Module</td><td rowspan="2">Variant</td><td colspan="3">City I</td><td colspan="2">City II</td><td colspan="2">City III</td></tr><tr><td>AR(%) ↑</td><td>CR(%)↑ APT(%)↓ GMV(%)↑</td><td></td><td>AR(%) ↑ CR(%)↑ APT(%)↓ GMV (%) ↑</td><td>AR(%) ↑</td><td>CR(%)↑ APT(%)↓ GMV(%) ↑</td><td></td></tr><tr><td>GenMatch Full</td><td></td><td></td><td></td><td></td><td> $0 . 9 0 0 + 0 . 0 0 0 \ \ 0 . 9 0 0 + 0 . 0 0 \ 0 . 0 0 0 \ 0 . 0 0 0 + 0 . 0 0 0 \ 0 . 0 0 0 + 0 . 0 0 0 \ 0 . 0 0 0 + 0 . 0 0 0 \ 0 . 0 0 0 + 0 . 0 0 0 \ 0 . 0 0 0 + 0 . 0 0 0 \ 0 . 0 0 0 + 0 . 0 0 0 \ \mu \ 0 . 0 0 0 + 0 . 0 0 0 \ 0 . 0 0 0 + 0 . 0 0 0 \ 0 . 0 0 0 + 0 . 0 0 0 \ 0 . 0 0 0 + 0 . 0 0 0 0 + 0 . 0 0 0 0$ </td><td></td><td></td><td></td></tr><tr><td></td><td>A1</td><td></td><td></td><td></td><td> $\begin{array} { r } { - 2 . 3 2 \pm 0 . 0 4 - 2 6 1 \pm 0 . 0 5 + 1 . 7 4 + 0 . 0 2 - 0 . 5 7 \pm 0 . 0 4 \ - 2 . 1 4 \pm 0 . 0 7 - 2 . 5 5 \pm 0 . 0 4 + 1 . 2 3 + 0 . 0 4 - 2 . 0 7 \mp 0 . 1 0 \mp - 2 . 5 5 + 0 . 1 2 - 3 . 0 7 \mp 0 . 0 7 + 1 . 1 6 + 0 . 0 2 - 2 . 2 8 + 0 . 1 5 } \end{array}$ </td><td></td><td></td><td></td></tr><tr><td rowspan="3">Encoder</td><td>A2</td><td></td><td></td><td> ${ \left| - 2 . 2 5 \pm 0 . 0 0 2 - 2 . 3 0 \pm 0 . 0 2 + 1 . 6 1 \pm 0 . 0 1 - 0 . 5 2 \pm 0 . 0 0 {  } - 1 . 9 5 \pm 0 . 0 0 3 - 2 . 3 6 \pm 0 . 0 0 2 + 1 . 0 6 \pm 0 . 0 0 2 - 1 . 9 6 \pm 0 . 0 0 5 { \left| - 2 . 4 3 \pm 0 . 0 6 - 2 . 6 9 \pm 0 . 0 3 + 1 . 1 4 \pm 0 . 0 1 - 2 . 1 5 \pm 0 . 0 9 { \left| 0 . 0 0 2 + 0 . 0 0 2 + 1 . 0 8 \right| } \right|}  }\right|$ </td><td></td><td></td><td></td><td></td></tr><tr><td>A3</td><td></td><td></td><td> $| - 0 . 1 4 \pm 0 . 0 0 2 - 0 . 1 2 \pm 0 . 0 2 + 0 . 1 3 + 0 . 0 1 - 0 . 0 4 + 0 . 0 1 | - 0 . 6 6 + 0 . 0 5 - 1 . 0 4 + 0 . 0 4 + 0 . 4 9 + 0 . 0 3 - 0 . 7 2 \pm 0 . 1 0 | - 0 . 3 9 + 0 . 0 6 - 0 . 2 5 + 0 . 0 3 + 0 . 1 8 + 0 . 0 7 - 0 . 4 0 + 0 . 0 9 |$ </td><td></td><td></td><td></td><td></td></tr><tr><td>A4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">Learner</td><td>A5</td><td></td><td></td><td></td><td> $| - 2 . 4 7 \pm 0 . 0 5 - 2 . 6 9 \pm 0 . 0 6 + 2 . 0 3 \pm 0 . 0 1 - 0 . 5 9 \pm 0 . 0 8 | ^ { - 2 } - 2 . 2 0 \pm 0 . 0 9 - 2 . 9 7 \mp 0 . 0 6 + 1 . 2 8 \pm 0 . 0 5 - 2 . 3 0 \mp 0 . 1 5 | - 2 . 7 6 \pm 0 . 1 6 - 3 . 1 5 \pm 0 . 0 8 + 1 . 1 7 \mp 0 . 0 3 - 2 . 3 2 \pm 0 . 1 8 \hbar$   $\left| - 1 . 8 3 + 0 . 0 3 3 - 1 . 9 9 + 0 . 0 3 3 + 2 . 7 0 + 0 . 0 1 - 0 . 4 5 + 0 . 0 3 3 \right| - 1 . 6 1 + 0 . 0 5 - 1 . 9 9 + 0 . 0 3 3 + 1 . 4 4 + 0 . 0 0 3 - 1 . 7 0 + 0 . 0 8 6 - 2 . 0 2 + 0 . 0 8 8 - 2 . 3 4 + 0 . 0 4 + 1 . 5 3 + 0 . 0 9 - 1 . 8 8 + 0 . 0 9 9$ </td><td></td><td></td><td></td></tr><tr><td>A6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>A7</td><td></td><td></td><td></td><td> $- 1 , 1 2 \div 0 . 0 2 - 0 . 4 9 + 0 . 0 0 2 + 0 . 7 6 + 0 . 0 1 - 0 . 1 5 + 0 . 0 1 \Big 1 - 0 . 9 8 + 0 . 0 4 - 0 . 9 0 + 0 . 0 2 + 0 . 5 3 + 0 . 0 2 - 0 . 5 2 + 0 . 0 3 \Big 1 - 1 . 4 7 + 0 . 0 7 - 1 . 6 9 + 0 . 0 4 + 0 . 6 5 + 0 . 0 2 - 0 . 6 5 + 0 . 0 4$ </td><td></td><td></td><td></td></tr><tr><td>A8</td><td></td><td></td><td></td><td>−0.22 ± 0.01 −0.18 ± 0.01 +0.20 ± 0.00 −0.06 ± 0.01−0.11 ± 0.02 −0.40 ± 0.01 +0.16 ± 0.01 −0.24 ± 0.03−0.37 ± 0.05 −0.28 ± 0.02 +0.17 ± 0.01 −0.43 ± 0.04</td><td></td><td></td><td></td></tr><tr><td>A9</td><td></td><td></td><td></td><td> $\left| - 0 . 3 6 \pm 0 . 0 1 - 1 . 5 5 \pm 0 . 0 1 + 1 . 3 9 \pm 0 . 0 0 - 0 . 4 0 \pm 0 . 0 1 \right| - 0 . 6 7 \pm 0 . 0 0 2 - 1 . 3 1 \pm 0 . 0 1 + 0 . 9 0 \pm 0 . 0 1 - 1 . 3 7 \pm 0 . 0 4 - 1 . 1 1 \pm 0 . 0 4 - 1 . 0 7 \pm 0 . 0 0 2 + 0 . 8 6 \pm 0 . 0 0 1 - 1 . 4 5 \pm 0 . 0 3 9$ </td><td></td><td></td><td></td></tr><tr><td>A10</td><td></td><td> $- 1 . 5 7 \pm 0 . 0 6 - 1 . 3 1 \pm 0 . 0 7 + 1 . 0 9 \pm 0 . 0 2 - 0 . 2 3 \pm 0 . 0 7 \sqrt { \hphantom { 0 . 0 7 } }$ </td><td></td><td>-1.35 ± 0.11−1.88 ± 0.07 +0.71 ± 0.06−0.82 ± 0.20-</td><td></td><td>−0.92 ± 0.03 −0.63 ± 0.04 +0.58 ± 0.02 −0.27 ± 0.04−0.52 ± 0.07 −1.05 ± 0.05 +0.46 ± 0.04 −0.69 ± 0.14−1.22 ± 0.14 −0.96 ± 0.09 +0.48 ± 0.04 −0.78 ± 0.19</td><td></td></tr><tr><td>A11</td><td></td><td></td><td></td><td> $| - 0 . 4 6 + 0 . 0 7 7 - 0 . 8 5 + 0 . 0 8 + 0 . 5 0 + 0 . 0 3 - 0 . 3 1 + 0 . 0 9 | - 0 . 2 9 + 0 . 1 3 - 1 . 3 8 + 0 . 0 8 + 0 . 3 2 + 0 . 0 7 - 1 . 1 2 + 0 . 2 5 | - 0 . 7 9 + 0 . 2 4 - 0 . 8 4 + 0 . 1 2 + 0 . 4 2 + 0 . 0 5 - 1 . 0 4 + 0 . 3 9 |$ </td><td></td><td> $\cdot 1 . 8 4 \pm 0 . 2 0 - 2 . 0 5 \pm 0 . 1 0 + 0 . 7 3 \pm 0 . 0 4 - 0 . 9 1 \pm 0 . 2 4$ </td><td></td></tr><tr><td rowspan="2"></td><td>A12</td><td></td><td></td><td> $\left| - 1 . 9 3 \pm 0 . 1 5 - 1 . 7 9 \pm 0 . 1 0 + 0 . 5 9 \pm 0 . 0 3 - 0 . 2 3 \pm 0 . 0 1 \right| - 1 . 0 8 \pm 0 . 1 1 - 1 . 1 7 \pm 0 . 1 4 + 0 . 2 5 \pm 0 . 0 3 - 1 . 0 9 \pm 0 . 0 7 \left| - 2 . 4 6 \pm 0 . 0 1 5 - 3 . 0 7 \mp 0 . 2 0 + 0 . 1 6 \pm 0 . 0 3 - 0 . 9 1 \pm 0 . 0 3 \right|$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: Auxiliary prediction AUC relative to PDP, reported as the mean ± standard deviation of percentage changes over five runs.
<table><tr><td rowspan="2">Variant</td><td colspan="3"></td><td colspan="3"> $\overline { { \mathbf { \nabla } \mathbf { C _ { i t y I I } } } }$ </td><td colspan="3">City II</td></tr><tr><td> $\mathbf { A U C } _ { \mathrm { D A } } \left( \% \right) \uparrow$ </td><td>City 1  $\mathbf { A U C } _ { \mathrm { P C A A } } \left( \% \right) \uparrow$ </td><td> $\mathbf { A U C } _ { \mathrm { D C A A } } \left( \% \right) \uparrow$ </td><td> $\mathbf { A U C } _ { \mathrm { D A } } \left( \% \right) \uparrow$ </td><td> $\mathbf { A U C } _ { \mathrm { P C A A } } \left( \% \right) \uparrow$ </td><td> $\mathbf { A U C } _ { \mathrm { D C A A } } \left( \% \right) \uparrow$ </td><td> $\mathbf { A U C } _ { \mathrm { D A } } \left( \% \right) \uparrow$ </td><td> $\mathbf { A U C } _ { \mathrm { P C A A } } \left( \% \right) \uparrow$ </td><td> $\mathbf { A U C } _ { \mathrm { D C A A } } \left( \% \right) \uparrow$ </td></tr><tr><td>PDP</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $\mathbf { G e n M a t c h } _ { \mathrm { M T L } }$ </td><td> $\mathbf { + 1 . 6 2 \pm 0 . 1 1 }$ </td><td> $\mathbf { + 0 . 8 4 \pm 0 . 0 4 }$ </td><td> $\mathbf { + 0 . 7 4 \pm 0 . 0 7 }$ </td><td>+1.79 ± 0.14</td><td> $\mathbf { + 0 . 7 6 \pm 0 . 0 5 }$ </td><td> $\mathbf { + 0 . 4 5 \pm 0 . 0 3 }$ </td><td> $\mathbf { + 1 . 7 4 \pm 0 . 1 5 }$ </td><td> $\mathbf { + 0 . 8 2 \pm 0 . 0 5 }$ </td><td> $\mathbf { + 0 . 5 5 \pm 0 . 0 3 }$ </td></tr></table>

Multi-Agent Baselines. CoRide and CoopRide model driver or regional cooperation with multi-agent reinforcement learning and are primarily designed to improve fleet-level, long-term supply– demand coordination. We include them to examine whether policies developed from a Macro-View perspective transfer efectively to the current-batch, OD-pair-level MICOD objective.

$\mathbf { G e n M a t c h } _ { \mathrm { V a l u e } } .$ . This variant replaces only PDP’s pair-level prediction stage with the DA, PCAA, and DCAA predictions learned by GenMatch. The original hand-crafted value calculation and Kuhn– Munkres matching stages remain unchanged. It isolates the benefit of the Business-Aware Utility Learner’s service-process predictions, while the comparison between GenMatch and full GenMatch measures the additional contribution of learned utility and stateaware assignment generation.

A.1.4 Evaluation Protocols. Ofline evaluation replays seven days of historical dispatch trafic in each city-specific simulator after training on fourteen days of data. The replay preserves the observed order arrivals, driver states, and candidate connections, and all competing methods are evaluated under the same upstream feasibility rules. We report AR, CR, APT, and GMV relative to $\mathrm { P D P } _ { \mathrm { K M } } .$ Each ofline result is averaged over five independent runs, and the corresponding standard deviation reflects variation across those runs.

Online A/B tests are conducted in City III–V for fourteen days. GenMatch and the $\mathrm { P D P } _ { \mathrm { K M } }$ control are evaluated with a one-hour time-slice interleaved design, and results are reported as treatmentminus-control changes. This design compares the two systems under recurring live trafic conditions while limiting long-term drift between the treatment and control periods. The online evaluation uses the same four primary dispatch metrics as the ofline study and additionally examines broadcast volume, answer and completion counts, passenger cancellation and bad-experience ratios, driver answer and cancellation ratios, and driver income.

A.1.5 Implementation Details. All models are trained on four NVIDIA RTX L20 GPUs. We use DeepSpeed Zero Redundancy Optimizer (ZeRO) Stage 2 to partition optimizer states and gradients across devices and bfloat16 (BF16) mixed precision to reduce memory and computation costs. The default GenMatch configuration contains two sparse bipartite encoder layers and two pointer-decoder layers with hidden dimension 128. Matching attention, competition attention, and pointer scoring each use four heads, and the feedforward dimension is 512. The model supports at most 500 orders, 500 drivers, and 10,000 candidate OD pairs in one dispatch batch.

Training uses Adam for 50 epochs, a cosine learning-rate schedule with three warm-up epochs, a learning-rate search range of $5 { \times } 1 0 ^ { - 5 } \ \mathrm { t o } \ 5 { \times } 1 0 ^ { - 4 }$ , weight decay of $1 0 ^ { - 4 }$ , and gradient clipping at 1.0. The per-GPU batch size is 16 and the global batch size is 64. The three auxiliary service-process tasks receive equal task weights, and the overall multi-task-loss coeficient is set to $\lambda _ { \mathrm { m t l } } = 1 0$ according to the sensitivity analysis in Section A.5. Ofline experiments use five independent random seeds. Complete model and optimization settings are summarized in Table 9, and the training and inference procedures are provided in Section B.2.

## A.2 Full Ablation Results

Table 6 reports all twelve ablations on the three ofline cities. We analyze each variant together with its corresponding result and implication, organized around the three modules of GenMatch. Context-Aware Bipartite Encoder (A1–A3). A1 removes both batch-level matching attention in Eq. (7) and competition attention in Eq. (6), reducing the encoder to isolated OD-pair representations. This change degrades every metric, decreasing AR by 2.14%–2.55%, CR by 2.55%–3.07%, and GMV by 0.57%–2.28%, while increasing APT by 1.16%–1.74%. A2 restores matching attention over the complete candidate neighborhoods while still removing competition attention. Relative to A1, it improves AR by 0.07%–0.19%, CR by 0.19%–0.38%, and GMV by 0.05%–0.13%, while reducing the APT increase by 0.02%–0.17%; hence, neighborhood-level matching information is useful even without competition modeling. Full GenMatch further improves all metrics over A2, confirming that competition relations provide complementary batch context. A3 instead retains both attention mechanisms but removes their shared degree embedding. Its efect is small in City I (−0.14% AR and −0.12% CR), becomes largest in City II (−0.66% AR, −1.04% CR, and −0.72% GMV), and is intermediate in City III. This city-dependent degradation indicates that explicit neighborhood size is particularly useful in larger candidate graphs, where normalized attention alone cannot preserve the scale of local competition.

Business-Aware Utility Learner (A4–A5). A4 removes the multitask supervision of candidate memory while retaining the stagewise predictions used to construct utility. It decreases AR by 2.20%– 2.76%, CR by 2.69%–3.15%, and GMV by 0.59%–2.32%, producing the largest AR and CR degradations among the learner variants in every city. Thus, behavioral-outcome supervision improves not only the auxiliary predictions but also the shared candidate representations used for assignment. A5 retains this supervision but removes the utility logit from Eq. (14), leaving the decoder to rely on its structural score. It degrades all four metrics and increases APT by 1.44%–2.70%, the largest APT increase among all ablations in every city. This result shows that the structural score alone cannot recover the service-eficiency trade-ofs captured by explicit business-value guidance.

State-Aware Pointer Decoder (A6–A12). A6, A7, and A8 remove the initial-, selected-, and residual-state representations, respectively, from the state-aware query in Eq. (12). A7 causes the smallest degradation of the three, whereas A6 reduces AR by 0.98%–1.47% and A8 reduces CR by 1.07%–1.55% and GMV by 0.40%–1.45%. The contrast shows that the original batch and, especially, the remaining feasible opportunities provide more decision context than the already selected pairs, although all three summaries contribute. A9 removes progress encoding and consistently degrades every metric, including AR by 0.52%–1.22% and CR by 0.63%–1.05%. This confirms that the averages of the three state sets do not by themselves retain how far generation has progressed. A10 replaces the explicit state-aware query with causal self-attention; its AR and CR losses reach 1.84% and 2.05% in City III, showing that implicit history propagation does not represent the evolving feasible set as efectively. A11 trains with a fixed target order and remains inferior to Full in every city, with CR decreasing by 0.84%–1.38% and GMV by 0.31%– 1.12%; random target permutations therefore reduce dependence on an arbitrary generation order. Finally, A12 uses the first-step generation logits as fixed weights for Kuhn–Munkres matching. It produces particularly large AR and CR losses, reaching −2.46% and −3.07%, respectively, in City III. Although its APT change is comparatively small, the loss in answered and completed orders demonstrates that frozen scores and one-shot optimization cannot replace the score updates required after each selected pair changes the feasible set.

Overall, every targeted removal degrades all four dispatch metrics across all three cities. The encoder results establish the complementary roles of neighborhood composition, competition context, and neighborhood size; the learner results validate behavioral supervision and explicit business guidance; and the decoder results show that assignment quality depends on dynamically representing and updating the matching state. These complete results support the conclusions drawn from the compact City III table in the main text.

## A.3 Auxiliary Prediction Evaluation

We use the area under the receiver operating characteristic curve (AUC) to evaluate the auxiliary DA, PCAA, and DCAA predictions. A larger AUC indicates better discrimination between positive and negative outcomes. Table 7 reports relative AUC changes over the production prediction model.

GenMatch improves all three tasks in every city, demonstrating that the gains are consistent across diferent markets rather than being specific to one dataset. DA improves by 1.62%–1.79%, the largest gain among the three tasks. PCAA and DCAA improve by 0.76%– 0.84% and 0.45%–0.74%, respectively. The relatively small standard deviations over five runs further indicate stable improvements.

These results verify that the Business-Aware Utility Learner captures the stage-wise service process more accurately than the production prediction model. They also explain the improvement of GenMatch : replacing only the prediction stage with these auxiliary predictions already benefits dispatch performance, even when the hand-crafted value calculation and Kuhn–Munkres matching remain unchanged. The additional improvement of full GenMatch reported in the main text therefore comes from jointly learning business utility and generating the final assignment, rather than from prediction accuracy alone.

## A.4 Efect of Model Capacity

Figure 5 studies model capacity by varying encoder depth, decoder depth, and hidden dimension �. Results are reported relative to the Medium configuration, and the outlined markers identify the best configuration for each city. The Small model degrades all four metrics across all three cities, showing that insuficient capacity limits both matching quality and business value. Reducing encoder depth, decoder depth, or width also causes broad performance drops. A shallower encoder markedly reduces CR and increases APT, confirming the importance of suficient capacity for batchlevel graph encoding. A shallower decoder mainly hurts AR and CR, while reduced width weakens all four metrics.

Increasing capacity beyond Medium does not yield consistent gains. The Large model slightly improves AR in City II and City III and reduces APT in City I and City III. However, it substantially reduces CR and GMV in every city. In contrast, Medium achieves the best CR and GMV across all three cities and the best APT in City II. It therefore provides the best overall balance between model capacity and dispatch performance and is used as the default configuration.

## A.5 Efect of the Multi-Task-Loss Weight

Table 8 studies $\lambda _ { \mathrm { m t l } } ,$ , which balances the sequence-generation loss and the multi-task loss in Eq. (19). All values are reported relative to $\lambda _ { \mathrm { m t l } } ~ = ~ 1 0$ . Weights below 10 provide insuficient behavioral supervision and consistently degrade dispatch performance. Larger weights can improve auxiliary AUC, especially at 1,000 and 10,000, but these prediction gains do not translate into better assignments: CR and GMV generally decrease because the multi-task objective begins to dominate sequence learning. We therefore set $\lambda _ { \mathrm { m t l } } = 1 0 ,$ which provides the best overall balance across dispatch metrics and cities.

## B Implementation and Deployment Details

## B.1 Model Configuration

Table 9 lists the model and training configurations of GenMatch.

## B.2 Training and Inference Procedures

Algorithm 1 summarizes training. Each dispatch batch is encoded once, after which the model learns stage-wise service outcomes and utility logits. A random permutation of the completed OD pairs provides the teacher-forced generation target, and the generation and auxiliary losses jointly update all model parameters.

Algorithm 2 details inference. GenMatch first computes pair memory and utility logits for the complete dispatch batch. At each step, it constructs the query from the current matching state, greed ily selects the highest-probability selectable candidate, masks every conflicting candidate, and incrementally updates the selected and residual states. This loop continues while $\begin{array} { r } { \sum _ { r } m _ { r } ^ { ( k ) } > 0 } \end{array}$ , and the selected OD pairs form the final matching set.

## B.3 Deployment Details

GenMatch has been deployed on a large-scale ride-hailing platform serving multiple international markets. The existing production system, which we call the Pair-Level Dispatch Engine, was designed for the conventional multi-stage paradigm. It partitions feasible order-driver pairs into shards, predicts pair-level business signals in parallel, calculates matching weights, and constructs the final assignment. This design is eficient because each pair can be processed independently before dispatch matching. It therefore avoids assembling the complete dispatch batch during model inference and scales well under strict latency constraints.

Algorithm 1 GenMatch Training Procedure   
Require: Mini-batches of $\left\{ \mathcal { G } _ { t } , C _ { t } , \left\{ \mathbf { y } _ { r } , w _ { r } \right\} _ { r = 1 } ^ { P _ { t } } \right\} ;$ loss weights   
$\{ \lambda _ { m } \} _ { m \in \mathcal { T } }$ and $\lambda _ { \mathrm { m t l } }$   
Ensure: Trained model parameters Θ   
1: Initialize model parameters Θ   
2: for each training epoch do   
3: for each mini-batch I do   
4: Initialize accumulated losses and valid-position counts   
5: for each dispatch batch $t \in { \mathcal { I } }$ do   
6: Encode $\mathcal { G } _ { t }$ as pair representations $\left\{ \mathbf { z } _ { r } \right\} _ { r = 1 } ^ { P _ { t } }$ using   
Eq. (10)   
7: Predict the DA, PCAA, and DCAA probabilities with   
the multi-task network   
8: Obtain $\left\{ a _ { r } \right\} _ { r = 1 } ^ { P _ { t } }$ using Eq. (11)   
9: Uniformly sample an ordering of $C _ { t }$ to form $y _ { { } _ { t } } ^ { * } =$   
$( e _ { t , * } ^ { ( 1 ) } , \ldots , e _ { t , * } ^ { ( K _ { t } ^ { * } ) } )$ , where $K _ { t } ^ { * } = | C _ { t } |$   
10: Set $\begin{array} { r } { \mathbf { S _ { \mathrm { i n i t } } }  \sum _ { r } \mathbf { z } _ { r } , N _ { \mathrm { i n i t } }  P _ { t } } \end{array}$ , and q<sub>init</sub> $ \mathrm { S } _ { \mathrm { i n i t } } / N _ { \mathrm { i n i t } }$   
11: Initialize the selected-state statistics with   
$( \mathbf { S } _ { s \mathrm { e l } } ^ { ( 1 ) } , N _ { s \mathrm { e l } } ^ { ( 1 ) } ) \gets ( \mathbf { 0 } , 0 )$   
12: Initialize the residual-state statistics with   
$( \mathbf { S } _ { \mathrm { r e s } } ^ { ( 1 ) } , N _ { \mathrm { r e s } } ^ { ( 1 ) } ) \gets ( \mathbf { S } _ { \mathrm { i n i t } } , P _ { t } )$   
13: Set $m _ { r } ^ { ( 1 ) } \gets 1$ for all �   
14: for $k = 1 , \ldots , K _ { t } ^ { * }$ do   
15: Construct $\mathbf { q } ^ { ( k ) }$ using Eq. (12)   
16: Compute $\bar { \{ s _ { r } ^ { ( k ) } \} }$ and $\{ p _ { r } ^ { ( k ) } \}$ using Eqs. (13)   
and (14)   
17: Under teacher forcing, find $r _ { * } ^ { ( k ) }$ such that $e _ { r _ { * } ^ { ( k ) } } =$   
$e _ { t , * } ^ { ( k ) }$   
18: Accumulat $\mathsf { e } - \log { p } _ { r _ { * } ^ { ( k ) } } ^ { ( k ) }$   
19: Let $\mathcal { B } ^ { ( k ) }$ contain the target pair and its currently   
selectable conflicting pairs   
20: Update the selected- and residual-state statistics   
using Eq. (16), and mask $\mathcal { B } ^ { ( k ) }$   
21: end for   
22: end for   
23: Compute $\mathcal { L } _ { \mathrm { g e n } }$ and ${ \mathcal L } _ { \mathrm { m t l } }$ using Eqs. (17) and (18)   
24: Compute L using Eq. (19)   
25: Update Θ using $\nabla _ { \Theta } \mathcal { L }$   
26: end for   
27: end for   
28: return Θ

GenMatch changes the serving unit from an individual pair to an entire dispatch batch. Its encoder requires the complete sparse bipartite graph, and its decoder must maintain a consistent candidate order while generating assignments. We therefore develop a Batch-Level Generative Dispatch Engine. This upgrade introduces three challenges. First, centralizing feature preparation would cause excessive compute and latency, while independent shards cannot provide the full batch structure. We retain distributed candidate filtering and feature preparation, then use a city-level orchestrator to restore candidate order and assemble the complete batch for global inference. Second, GenMatch requires 285 features produced across diferent shards. Re-fetching them globally is costly, and shard-local feature identifiers may be inconsistent. We pass sparse features with their candidates and re-key them into one global feature dictionary during aggregation. Third, autoregressive outputs depend on stable candidate indices and must coexist with other product lines. We remove previously claimed candidates before inference and treat generated pairs as pre-assignments, which then enter the standard arbitration and locking process.

![](images/29b9e65f8202d9ff17a98199fd1ccd65c9590e610c5621965c077314b34bb654.jpg)  
(a) AR (↑)

![](images/c83329a69424ea6d811a394d5c57e81cc63adc75dd1fab2f9b15ede27ca254bc.jpg)  
(b) CR (↑)

![](images/9da20ba9563acba217b54a081bd274b5cfae451d223ddf93ac4732b178351c9b.jpg)  
(c) APT (↓)

![](images/a2cc8425648d4a237887246514ea1e578609e8c0536ed64b75b4aee89ae6fd3d.jpg)  
(d) GMV (↑)  
Figure 5: Efect of model capacity relative to the Medium configuration. Error bars denote standard deviations over five runs.

Table 8: Sensitivity to $\lambda _ { \mathrm { m t l } }$ relative to the selected value 10. Values are the mean ± standard deviation of percentage changes over five runs. Larger AR, CR, GMV, and AUC and smaller APT are preferred. Bold and underlined values denote the best and second-best results in each column, respectively.  
(a) Dispatch performance
<table><tr><td rowspan="2">λmtl</td><td colspan="4">City I</td><td colspan="4">City ⅡI</td><td colspan="4">City II</td></tr><tr><td>AR↑</td><td>CR↑</td><td>APT↓</td><td>GMV↑</td><td>AR↑</td><td>CR↑</td><td>APT↓</td><td>GMV↑</td><td>AR↑</td><td>CR↑</td><td>APT↓</td><td>GMV↑</td></tr><tr><td>0.01</td><td>-0.29±0.05</td><td>-0.44±0.06</td><td>+0.67±0.01</td><td>-0.19±0.05</td><td>-0.27±0.05</td><td>-0.29±0.06</td><td>+0.43±0.01</td><td>−0.57±0.05</td><td>-0.78±0.05</td><td>-0.88±0.06</td><td>+0.22±0.01</td><td>-0.60±0.05</td></tr><tr><td>0.1</td><td>-0.17±0.02</td><td>−0.39±0.02</td><td>+0.37±0.01</td><td>−0.08±0.02</td><td>-0.19±0.02</td><td>−0.22±0.02</td><td>+0.24±0.01</td><td>−0.24±0.02</td><td>−0.49±0.02</td><td>-0.46±0.02</td><td>+0.16±0.01</td><td>−0.35±0.02</td></tr><tr><td>1</td><td>−0.11±0.03</td><td>−0.07±0.03</td><td>+0.20±0.01</td><td>−0.02±0.03</td><td>-0.08±0.03</td><td>−0.14±0.03</td><td>+0.12±0.01</td><td>-0.18±0.03</td><td>-0.20±0.03</td><td>−0.17±0.03</td><td>+0.05±0.01</td><td>−0.11±0.03</td></tr><tr><td>10</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td></tr><tr><td>100</td><td>+0.21±0.05</td><td>−0.11±0.06</td><td>+0.13±0.01</td><td>+0.05±0.05</td><td>-0.07±0.05</td><td>-0.16±0.06</td><td>−0.13±0.01</td><td>-0.06±0.05</td><td>+0.08±0.05</td><td>-0.33±0.06</td><td>−0.08±0.01</td><td>-0.28±0.05</td></tr><tr><td>1000</td><td>-0.30±0.02</td><td>-0.19±0.02</td><td>-0.56±0.01</td><td>−0.05±0.02</td><td>-0.21±0.02</td><td>-0.19±0.02</td><td>-0.36±0.01</td><td>-0.30±0.02</td><td>−0.15±0.02</td><td>-0.39±0.02</td><td>-0.16±0.01</td><td>-0.35±0.02</td></tr><tr><td>10000</td><td>-0.67±0.03</td><td>−0.25±0.03</td><td>-0.81±0.01</td><td>-0.17±0.03</td><td>-0.42±0.03</td><td>-0.32±0.03</td><td>-0.47±0.01</td><td>-0.67±0.03</td><td>-0.49±0.03</td><td>-0.44±0.03</td><td>-0.17±0.01</td><td>−0.41±0.03</td></tr></table>

(b) Auxiliary prediction AUC
<table><tr><td rowspan="2"> $\lambda _ { \mathrm { m t l } }$ </td><td colspan="3">City I</td><td colspan="3">City ⅡI</td><td colspan="3">City ⅢII</td></tr><tr><td>DA</td><td>PCAA</td><td>DCAA</td><td>DA</td><td>PCAA</td><td>DCAA</td><td>DA</td><td>PCAA</td><td>DCAA</td></tr><tr><td>0.01</td><td>-0.29±0.05</td><td>+0.05±0.06</td><td>+0.08±0.01</td><td>−0.17±0.05</td><td>−0.15±0.06</td><td>−0.12±0.01</td><td>-0.16±0.05</td><td>-0.04±0.06</td><td>−0.18±0.01</td></tr><tr><td>0.1</td><td>−0.17±0.02</td><td>+0.11±0.02</td><td>-0.14±0.01</td><td>-0.14±0.02</td><td>−0.04±0.02</td><td>-0.05±0.01</td><td>-0.14±0.02</td><td>-0.16±0.02</td><td>-0.11±0.01</td></tr><tr><td>1</td><td>−0.44±0.03</td><td>+0.16±0.03</td><td>−0.09±0.01</td><td>−0.11±0.03</td><td>−0.06±0.03</td><td>−0.01±0.01</td><td>-0.05±0.03</td><td>−0.14±0.03</td><td>−0.06±0.01</td></tr><tr><td>10</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.00±0.00</td></tr><tr><td>100</td><td>−0.06±0.05</td><td>-0.06±0.06</td><td>+0.03±0.01</td><td>+0.09±0.05</td><td>−0.03±0.06</td><td>−0.04±0.01</td><td>−0.07±0.05</td><td>+0.09±0.06</td><td>+0.13±0.01</td></tr><tr><td>1000</td><td>+0.13±0.02</td><td>+0.18±0.02</td><td>+0.15±0.01</td><td>+0.12±0.02</td><td>+0.07±0.02</td><td>+0.03±0.01</td><td>+0.06±0.02</td><td>+0.18±0.02</td><td>+0.19±0.01</td></tr><tr><td>10000</td><td>+0.21±0.03</td><td>+0.24±0.03</td><td>+0.21±0.01</td><td>+0.08±0.03</td><td>+0.14±0.03</td><td>+0.15±0.01</td><td>+0.12±0.03</td><td>+0.15±0.03</td><td>+0.18±0.01</td></tr></table>

Table 9: Model and training configurations of GenMatch.
<table><tr><td>Configuration</td><td>Symbol</td><td>Value</td></tr><tr><td>Encoder layers</td><td> $L _ { \mathrm { e n c } }$ </td><td>2</td></tr><tr><td>Decoder layers</td><td> $L _ { \mathrm { d e c } }$ </td><td>2</td></tr><tr><td>Hidden dimension</td><td> $d$ </td><td>128</td></tr><tr><td>Matching-attention heads</td><td></td><td>4</td></tr><tr><td>Competition-attention heads</td><td></td><td>4</td></tr><tr><td>Pointer heads</td><td> $P _ { \mathrm { p t r } }$ </td><td> $4$ </td></tr><tr><td>Feed-forward dimension</td><td></td><td>512</td></tr><tr><td>Dropout ratio</td><td> $^ -$ </td><td> $0 . 2$ </td></tr><tr><td>Multi-task shared-layer dimensions</td><td> $^ -$ </td><td> $[ 2 5 6 , 2 5 6 ]$ </td></tr><tr><td>Multi-task tower dimensions</td><td> $^ -$ </td><td> $\left[ 2 5 6 , 1 2 8 , 6 4 \right]$ </td></tr><tr><td>DA loss weight</td><td> $\lambda _ { \mathrm { D A } }$ </td><td>1.0</td></tr><tr><td>PCAA loss weight</td><td> $\lambda _ { \mathrm { P C A A } }$ </td><td>1.0</td></tr><tr><td>DCAA loss weight</td><td> $\lambda _ { \mathrm { D C A A } }$ </td><td>1.0</td></tr><tr><td>Multi-task-loss weight</td><td> $\lambda _ { \mathrm { m t l } }$ </td><td>10.0</td></tr><tr><td>Maximum orders per batch</td><td></td><td>500</td></tr><tr><td>Maximum drivers per batch</td><td></td><td>500</td></tr><tr><td>Maximum candidate OD pairs</td><td></td><td>10000</td></tr><tr><td>Training epochs</td><td></td><td>50</td></tr><tr><td>Optimizer</td><td></td><td>Adam</td></tr><tr><td>Learning-rate range</td><td></td><td> $5 \times 1 0 ^ { - 5 } - 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning-rate scheduler</td><td></td><td>Cosine</td></tr><tr><td>Warm-up epochs</td><td></td><td> $^ 3$ </td></tr><tr><td>Weight decay</td><td></td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size per GPU</td><td></td><td>16</td></tr><tr><td>Global batch size</td><td></td><td>64</td></tr><tr><td>Gradient clipping</td><td></td><td>1.0</td></tr></table>

```latex
Algorithm 2 GenMatch Inference Procedure
Require: Dispatch batch $\mathcal { G } _ { t } ;$ trained model parameters Θ
Ensure: Generated assignment $\boldsymbol { \mathcal { M } } _ { t }$
1: Encode $\mathcal { G } _ { t }$ as pair representations $\left\{ \mathbf { z } _ { r } \right\} _ { r = 1 } ^ { P _ { t } }$ using Eq. (10)
2: Predict the DA, $\mathrm { P C A A } ,$ and DCAA probabilities with the multi
task network
3: Obtain $\left\{ a _ { r } \right\} _ { r = 1 } ^ { P _ { t } }$ using Eq. (11)
4: Initialize $\boldsymbol { \mathcal { M } } _ { t } \gets \emptyset , \boldsymbol { m } _ { r } ^ { ( 1 ) } \gets 1$ for all $r ,$ and $k \gets 1$
5: Set $\begin{array} { r } { { \bf S } _ { \mathrm { i n i t } }  \sum _ { r } { \bf z } _ { r } } \end{array}$ , �<sub>init</sub> $ P _ { t } ,$ and q<sub>init</sub> $ \mathrm { { S } } _ { \mathrm { { i n i t } } } / N _ { \mathrm { { i n i t } } }$
6: Initialize the selected-state statistics as $( \mathbf { 0 } , 0 )$ and the residual
state statistics as $( \mathbf { S } _ { \mathrm { i n i t } } , P _ { t } )$
7: while $\begin{array} { r } { \sum _ { r = 1 } ^ { P _ { t } } m _ { r } ^ { ( k ) } > } \end{array}$ 0 do
8: Construct $\mathbf { q } ^ { ( k ) }$ using Eq. (12)
9: Compute $\{ \bar { s } _ { r } ^ { ( k ) } \}$ and $\{ p _ { r } ^ { ( k ) } \}$ using Eqs. (13) and (14)
10: Select $r ^ { ( k ) }$ using Eq. (15)
11: $\boldsymbol { \mathcal { M } } _ { t } \gets \boldsymbol { \mathcal { M } } _ { t } \cup \{ \boldsymbol { e } _ { r ^ { ( k ) } } \}$
12: Let $\mathcal { B } ^ { ( k ) }$ contain the selected pair and its currently se
lectable conflicting pairs
13: Update the selected- and residual-state statistics using
Eq. (16), and mask $\mathcal { B } ^ { ( k ) }$
14: $k \gets k + 1$
15: end while
16: return ${ \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { \mathbf { } } _ { } { } \mathbf { } _ { } { } _ { } { \mathbf { } } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { } { } _ { } { } \mathbf { } _ { }  _ { }$
```

We deploy the new engine in three steps: feature transmission, shadow model inference, and generative decision making. Any serving failure, malformed output, or configuration error automatically falls back to the Pair-Level Dispatch Engine, and GenMatch can be disabled without redeployment. The resulting design retains the scalability of pair-level distributed processing while enabling safe batch-level generative matching in production.
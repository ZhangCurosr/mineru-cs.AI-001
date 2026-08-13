# Consolidator: Learning Persistent Routed Memory Across Context Boundaries

Sungwoo Goo<sup>1</sup> Hwi-yeol Yun<sup>1</sup> Sangkeun Jung<sup>2</sup>

<sup>1</sup>College of Pharmacy, Chungnam National University

<sup>2</sup>Department of Computer Science & Engineering, Chungnam National University

Daejeon, Republic of Korea

swgoo91@gmail.com, hyyun@cnu.ac.kr, hugmanskj@gmail.com

## Abstract

Copying short-term memory (STM) into a slower store can preserve state across a context boundary, but persistence alone does not ensure that the retained state influences subsequent memory access. We test this distinction in a Phasor Memory Network (PMNet) using Consolidator, a shared slot-local operator that transforms routed STM before accumulating it into long-term memory (LTM), without replaying the source tokens. After each consolidation, the KV cache and STM are cleared. The retained LTM can still be read and is also fed into the hierarchical router, thereby conditioning which explicit-memory slots subsequent inputs access. We evaluate this mechanism on a two-segment modulo-10 mapping task in which the second segment updates the mapping at the same memory address. Following a second consolidation and reset, a held-out query must recover the updated mapping from LTM. The backbone and memory interface are frozen, leaving only 12.35K Consolidator parameters trainable (0.041% of a 29.95M model). Across five paired runs from the same STM-pretraining checkpoint, direct LTM routing raises updated-mapping recall from 44.38 ± 1.94% to 87.02 ± 1.76% (+42.64 ± 1.10 percentage points), while immediate STM recall remains 89.90% in both conditions; both train separate Consolidators and retain the same LTM read paths. Learned consolidation outperforms forced identity accumulation by 21.40 ± 1.91 percentage points without routing and 68.70 ± 1.76 with routing. Thus, on this task, consolidated LTM serves as both retrievable content and an access state that shapes subsequent slot selection.

Keywords: explicit memory, memory consolidation, short-term memory, long-term memory, hierarchical routing, recurrent state

## 1 Introduction

Transformer context is largely an append-only record of token and KV history [1]. A writable memory ofers a diferent abstraction: experience can be compressed into bounded state, updated at an address, and retrieved after transient context is discarded. Such a system must learn to form useful short-term state, preserve or revise it across boundaries, and use the retained state to select memory slots for later writes and reads.

Merely detaching and copying STM into a slower bufer would provide carryover, but it would leave two mechanistic questions unresolved. Does a fixed pretrained memory interface require a learned transition to revise conflicting content, rather than raw accumulation? Does the retained state only supply values to the read path, or does the router also use it to select slots for subsequent writes and reads?

Diferentiable memory, segment recurrence, compression, and adaptive state demonstrate several parts of this lifecycle [2–7]. We ask two linked questions: can a separately trained operator convert useful routed STM into LTM, and can that LTM guide later slot selection after the KV cache and STM are cleared?

We study this question in PMNet, where tokens write phase-valued state through a hierarchical router and memory is separable from the local attention cache [8]. Consolidator applies one shared gated phase transform to occupied STM slots, accumulates the result into LTM, and then permits KV and STM to be cleared. In later segments, the model can still read the retained LTM. The router also includes that LTM state in the phase-valued representation used to score slots at the same hierarchy level. As a result, information first written to STM can influence which memory slots later inputs select.

The controlled task comprises two context segments that reuse the same address and rule family, while the second segment replaces the first segment’s function parameters. The final held-out query targets this updated mapping and is answered only after both demonstration contexts have been processed and the KV cache and STM have been cleared. In the central intervention, 99.959% of the model is frozen and only the 12.35K-parameter Consolidator is trained. A paired routing ablation retains learned consolidation and the LTM read path but removes the direct LTM input to same-level slot routing. This reduces updated-mapping LTM recall from $8 7 . 0 2 \pm 1 . 7 6 \%$ to $4 4 . 3 8 \pm 1 . 9 4 \%$ , a paired decrease of $4 2 . 6 4 \pm 1 . 1 0 ~ \mathrm { p p }$ , while immediate STM recall of the second mapping remains exactly 89.90% in both conditions. The ablation therefore tests whether retained LTM guides later slot selection in addition to supplying retrievable content. A complementary identity intervention tests learned revision against raw STM accumulation: the learned-minus-identity gap is 21.40 ± 1.91 pp without direct routing and $6 8 . 7 0 \pm 1 . 7 6$ pp with routing, while mismatched and fresh LTM controls remain near chance. A dual-objective experiment tests whether pre-consolidation STM recall can coexist with post-reset LTM recall.

Our contributions are:

• A shared slot-local transform that consolidates routed latent STM without replay or topologydependent parameter growth.

• Direct LTM-conditioned slot routing as an architectural inductive bias that lets retained state guide later writes and reads through a frozen router.

• A sequential same-address update task that separates four memory functions: carrying state across a reset, revising an existing memory, retrieving retained content, and using LTM to guide slot selection. Same-checkpoint, content-replacement, and routing interventions isolate these functions.

The evidence is a mechanism-level proof of concept, not yet a claim about natural-language long context, continual learning, or systems eficiency.

## 2 Related Work

## 2.1 Explicit and recurrent memory

Neural Turing Machines and Diferentiable Neural Computers established end-to-end learned addressing over external read-write memory [2, 3]. Transformer-XL and Compressive Transformer carry or compress activations across segments [4, 5]; Recurrent Memory Transformer and Infiniattention maintain bounded recurrent state [9, 10]; and Memorizing Transformers retrieve earlier representations from a non-diferentiable store [11]. These systems preserve or retrieve history, whereas our intervention asks a learned slot-local transition to revise conflicting state at the same routed address and then feed the consolidated result directly into subsequent slot routing.

PMNet is the architectural base of this study. It represents recurrent memory updates as phasor rotations and organizes addresses hierarchically [8]. We add an explicit STM–LTM boundary and isolate its function rather than revisiting PMNet’s language-modeling or copy benchmarks.

## 2.2 Adaptive state and compact adaptation

Fast weights, selective state-space models, Test-Time Training layers, and Titans all make computation depend on rapidly changing latent state [6, 7, 12, 13]. Unlike test-time gradient methods, Consolidator keeps model parameters fixed during inference. Its forward pass updates non-parametric memory, which then guides later slot selection. Context Distillation instead stores and routes independent LoRA parameter memories [14]; PMNet writes newly observed content directly into routed non-parametric memory.

Textual Inversion showed that a frozen model can use a small learned embedding to represent a new concept [15]. One-layer post-training likewise found that adapting a restricted Transformer layer can recover much of full-parameter improvement [16]. Both rely on gradient optimization and therefore do not demonstrate forward-only acquisition during a memory episode. They support a more limited capacity premise: pretrained computation can make efective use of a compact adaptation substrate. PMNet trains the memory interface and Consolidator end to end; after training, new content is written and consolidated without changing model parameters.

## 2.3 Memory consolidation

Complementary Learning Systems motivates distinct fast and slow stores, while artificial consolidation commonly combats forgetting through stored or generated replay [17, 18]. Our terminology is an engineering analogy, not a claim of biological equivalence. Auto-Dreamer rewrites symbolic agent memory from stored entries and trajectories [19]. At each boundary, Consolidator transforms the routed STM directly into LTM without revisiting the input sequence. That LTM is later both retrieved as stored content and used to determine which slot new inputs access at the same memory level.

## 3 Problem Formulation and PMNet Background

## 3.1 Memory lifecycle

A memory episode comprises segments $X _ { 1 } , \ldots , X _ { T }$ and three non-parametric states: a local slidingattention KV cache $K _ { t }$ , routed short-term memory $S _ { t }$ written within segment $t ,$ and long-term memory $L _ { t }$ retained across segments. We distinguish two functions of retained LTM: content state supplied to the read path and access state that conditions which explicit-memory slots subsequent inputs select. Our evaluation asks four questions: whether STM contains the segment-specific mapping before consolidation; whether that mapping can be recovered from LTM after the KV cache and STM are cleared; whether learned consolidation can replace the mapping already stored at a reused address; and whether supplying LTM directly to the router changes post-reset recal relative to making it available only through the read path.

Copying a detached STM snapshot into LTM would alter training credit assignment and preserve state across the reset, but persistence alone would not establish either learned revision or direct control over subsequent slot selection. We therefore use two controlled comparisons. Comparing learned consolidation with identity accumulation tests whether a learned transformation at the STM–LTM boundary is needed to revise retained content. Comparing direct LTM routing on and of tests whether consolidated LTM improves recall by guiding subsequent slot selection, beyond supplying stored content through the read path.

(c) Two-segment memory episode and routed reuse.  
![](images/f06820b9894855f4e4735da5dd9014c053b9263e41397b12a11990a4726e21b5.jpg)  
Figure 1: Routed latent-memory lifecycle. Tokens write routed STM; at a boundary, Consolidator updates persistent LTM and KV/STM are cleared. Retained LTM supports later reads and conditions subsequent explicit-memory slot selection.

## 3.2 Hierarchical routed writes

PMNet represents memory dimensions as phase angles. At hierarchy block b, a token visits group g with candidate child slots $j .$ . Before direct LTM conditioning is introduced, the phase-valued representation used to score each candidate slot is

$$
a _ { t , b , g , j } = u _ { t , b , g , j } + e _ { b , g , j } ,\tag{1}
$$

where u is incoming latent state and e is a learned static slot embedding. With $\phi ( a ) = [ \sin ( a ) ; \cos ( a ) ]$ token and slot projections define a cosine-similarity distribution $p _ { t , j }$ over siblings. The highest-scoring child selects the next group, while every sibling receives a diferentiable phase update

$$
\Delta S _ { t , j } = p _ { t , j } \pi \operatorname { t a n h } ( W _ { o } W _ { v } \operatorname { R M S N o r m } ( h _ { t } ) ) .\tag{2}
$$

Thus traversal is hard top-1 between levels but writes remain soft within each visited group. An occupancy mask records the groups to consolidate. Reads use the wrapped dynamic state

$$
M _ { b , g , j } = ( S _ { b , g , j } + L _ { b , g , j } ) \mod 2 \pi ,\tag{3}
$$

together with the static embeddings. Further PMNet details are given by Goo et al. [8].

## 3.3 Operational meaning of replay-free

We call the consolidation procedure replay-free because each demonstration segment is presented only once along the main sequential trajectory that forms and updates LTM. At each boundary, Consolidator receives only routed STM and an occupancy mask; it neither re-encodes the original demonstration tokens nor retrieves them from an episodic replay bufer. After the reset, the final query is answered using retained LTM without presenting the demonstrations again. The dual-objective auxiliary trajectory separately reprocesses each segment’s context to measure preconsolidation STM recall, but it does not alter the replay-free trajectory used to construct LTM.

Replay-free does not mean BPTT-free. The present two-segment experiment retains the diferentiable graph across both consolidation boundaries. Detached or truncated training over longer memory episodes remains untested.

## 4 Learned Latent-State Consolidation

## 4.1 Shared slot-local phase transform

For an occupied STM slot $S \in \mathbb { R } ^ { d _ { m } }$ , define

$$
z ( S ) = [ \cos { S } ; \sin { S } ] .\tag{4}
$$

A gated MLP shared across all blocks, groups, and slots produces

$$
\begin{array} { c } { { r _ { \psi } ( z ) = W _ { d } \left[ \mathrm { S i L U } ( W _ { g } z ) \odot W _ { u } z \right] + b _ { d } , } } \\ { { \left[ c _ { \psi } ( S ) \right] = z ( S ) \otimes r _ { \psi } ( z ( S ) ) , } } \\ { { \left[ s _ { \psi } ( S ) \right] } } \\ { { C _ { \psi } ( S ) = \mathrm { a t a n 2 } ( s _ { \psi } ( S ) , c _ { \psi } ( S ) ) . } } \end{array}\tag{5}
$$

where $\otimes$ is element-wise complex multiplication in paired cosine/sine coordinates and atan2 is applied element-wise. The experiments use $d _ { m } = 3 2$ and hidden dimension $d _ { c } = 6 4$ . Zero output weights and unit-phasor bias make $C _ { \psi } ( S ) = S$ in Equation (5), so learning begins from exact identity. The transform is slot local and shares 12.35K parameters regardless of tree capacity.

## 4.2 Persistent accumulation and reset

For hierarchy block b, group g, and child slot $j ,$ , let $S _ { b , g , j } , L _ { b , g , j } \in \mathbb { R } ^ { d _ { m } }$ denote the STM and LTM phase vectors immediately before consolidation, and let $L _ { b , g , j } ^ { + }$ denote the LTM phase vector afterward. The binary mask $O _ { b , g }$ indicates whether group g received an STM write during the current segment. The boundary update is

$$
L _ { b , g , j } ^ { + } = \left\{ \begin{array} { l l } { { ( L _ { b , g , j } + C _ { \psi } ( S _ { b , g , j } ) ) } } & { { \mathrm { m o d } 2 \pi , } } \\ { { L _ { b , g , j } , } } & { { O _ { b , g } = 0 . } } \end{array} \right.\tag{6}
$$

If no LTM has previously been stored, $L _ { b , g , j }$ is initialized to the zero phase vector. STM, occupancy, and KV state are then cleared while LTM remains. The identity control replaces $C _ { \psi } ( S _ { b , g , j } )$ in Equation (6) with raw $S _ { b , g , j } ;$ its forward update is therefore raw copy-and-accumulate, subject only to phase wrapping. Both modes accumulate rather than overwrite state.

## 4.3 Direct LTM-conditioned routing

For a subsequent segment, our extension augments the base routing state in Equation (1) with the LTM state of the currently visited group:

$$
a _ { t , b , g , j } = u _ { t , b , g , j } + e _ { b , g , j } + L _ { b , g , j } .\tag{7}
$$

The static embedding remains a parametric address anchor, while LTM becomes an experiencedependent ofset. Consolidator receives no route label; task loss trains its output through the existing router and read path. Combining Equations (6) and (7) yields the recurrent path

$$
\begin{array} { r } { S _ { t } \xrightarrow { C _ { \psi } } L _ { t } \longrightarrow a _ { t + 1 } \longrightarrow S _ { t + 1 } , } \end{array}\tag{8}
$$

Table 1: Procedural rule families. Function parameters are resampled for each memory episode.
<table><tr><td>Family</td><td>Function</td><td>Sampled parameters</td></tr><tr><td>ADD10</td><td> $y = ( x + k )$  mod 10</td><td> $k \in \{ 1 , \ldots , 9 \}$ </td></tr><tr><td>AFFINE10</td><td> $y = ( a x + b ) $  mod 10</td><td> $a \in \{ 2 , \ldots , 9 \} , b \in \{ 0 , \ldots , 9 \}$ </td></tr></table>

so a fixed router can make experience-dependent slot selections because its non-parametric LTM input changes. This path makes LTM an access state rather than only retrievable content.

STM accumulated during a segment is not added directly to the candidate-slot representation used by the router at the same hierarchy level. It can nevertheless influence routing indirectly: reads from earlier hierarchy levels alter the hidden states received by deeper routers. Because LTM remains fixed within a segment, excluding direct same-level STM feedback avoids a token-to-token routing dependency and preserves parallel routing and write aggregation across the segment. Consequently, the routing ablation removes only the direct $L _ { b , g , j }$ term in Equation (7): the router itself, LTM retrieval, and indirect influence on deeper levels remain active.

## 4.4 Objectives

The primary consolidation experiments optimize only the post-reset query for the updated mapping in the second segment:

$$
\mathcal { L } _ { \mathrm { u p d a t e d } } = \mathrm { C E } ( f ( q _ { 2 } ; L _ { 2 } ) , y _ { 2 } ) .\tag{9}
$$

Their checkpoints are selected by validation recall of this updated mapping.

In a separate dual-objective experiment, we test whether one parameter set can support both immediate STM recall and post-reset LTM recall. For each of the two segments, an auxiliary trajectory evaluates its query after the demonstration has formed STM but before consolidation; $\mathcal { L } _ { \mathrm { S T M } }$ is the mean of these two query losses. Adding this term to Equation (9) gives

$$
\mathcal { L } _ { \mathrm { d u a l } } = \mathcal { L } _ { \mathrm { u p d a t e d } } + \mathcal { L } _ { \mathrm { S T M } } ,\tag{10}
$$

and selects checkpoints by the mean of updated-mapping LTM recall and pre-consolidation STM recall.

## 5 Controlled Sequential Same-Address Update Task

## 5.1 Procedural memory episodes

Each memory episode contains two context segments and one active memory address selected from four address tokens. One rule family, ADD10 or AFFINE10, is sampled per episode; the first and second segments use diferent function parameters from that family. Parameters are resampled across episodes, preventing a fixed address-to-rule solution.

Each segment provides eight demonstrations and one held-out query, each encoded as

$$
[ \mathrm { a d d r e s s } , \mathrm { r u l e - f a m i l y } , \mathrm { i n p u t } \mathrm { x } , \mathrm { d e l i m i t e r } , \mathrm { a n s w e r \ y } ]
$$

Loss is applied only to the answer token. Demonstration and query inputs are distinct within each segment, and the final query is selected so that the first and second mappings give diferent answers. The final prediction therefore cannot receive credit for copying an observed answer or retaining only the stale rule.

Table 2: Parameter-isolation conditions. The routing-of condition changes the forward path but has the same trainable parameter count as the standard Consolidator-only condition.
<table><tr><td>Condition</td><td>Consolidation</td><td>Trainable subset</td><td>Parameters</td></tr><tr><td>Learned full</td><td>Learned</td><td>Entire model</td><td>29.95M (100%)</td></tr><tr><td>Identity full</td><td></td><td>Raw STM accumulation All except Consolidator</td><td>29.94M (99.959%)</td></tr><tr><td>Consolidator only</td><td>Learned</td><td>Consolidator only</td><td>12.35K (0.041%)</td></tr><tr><td>Consolidator only, routing off Learned</td><td></td><td>Consolidator only</td><td>12.35K (0.041%)</td></tr><tr><td>Memory + Consolidator</td><td>Learned</td><td>Memory read/write/routing + Consolidator</td><td>1.526M (5.095%)</td></tr><tr><td>Learned full, dual objective</td><td>Learned</td><td>Entire model</td><td>29.95M (100%)</td></tr></table>

## 5.2 Two-stage training and evaluation protocol

STM-pretraining stage (Phase 1) trains same-segment rule induction from routed STM. The model processes demonstrations, clears KV history, and answers the held-out query; Consolidator is frozen. All main consolidation-training conditions share the resulting STM-capable checkpoint.

Consolidation-training stage (Phase 2) processes the two demonstration segments sequentially, consolidates STM into LTM after each segment, and then clears the KV cache and STM. The second segment reuses the address with the updated mapping. After the second consolidation and reset, its held-out query is presented without demonstrations or context-derived STM, so only retained LTM can provide the function parameters sampled for that episode. Direct LTM conditioning can afect both slot selection while writing the second-segment update and memory traversal during the final query; the present task measures their combined efect. There is no direct supervision on routes, slots, context tokens, or consolidation boundaries.

## 5.3 Mismatched-experience intervention

The mismatched control substitutes a donor experience from the same rule family but with diferent function parameters, then replaces its address token with the recipient’s. Format, family, and addressing cue are preserved while memory content changes. Dependence on episode-specific LTM should therefore appear as a collapse in recall.

## 6 Experimental Setup

## 6.1 Model and optimization

The 29.95M-parameter model has 12 Transformer layers, hidden dimension 384, and a 128-token sliding-attention window. PMNet uses four hierarchy blocks with branching factor four, 32- dimensional memory, and a 64-dimensional Consolidator hidden layer.

We use AdamW with learning rate $5 \times 1 0 ^ { - 4 }$ , global batch size 256, 100K procedural memory episodes per epoch, and at most 60 epochs. Validation and test each contain 1K memory episodes. Five consolidation-training seeds {42, 43, 44, 45, 46} use paired data streams and one fixed held-out test stream. Full architecture, optimizer, software, and seed settings are listed in Appendix E.

## 6.2 Parameter-isolation conditions

Learned full and identity full are independently optimized; their diference is not a samecheckpoint causal estimate. The direct mechanism test is Consolidator only, which uses direct LTM-conditioned routing: every other parameter is frozen, and the same trained checkpoint is evaluated with either learned or forced-identity consolidation. The Consolidator only, routing off condition uses the same parameter isolation but removes the direct LTM term in Equation (7); both variants retain learned consolidation, the fixed router, and the LTM read path, and each trains its own Consolidator from the same STM-pretraining initialization.

Table 3: Same-address update under the Consolidator-only intervention. Both columns evaluate the same trained checkpoint.
<table><tr><td>State queried after consolidation</td><td>Learned Consolidator Forced identity</td><td></td></tr><tr><td>Initial mapping after first consolidation</td><td> $5 0 . 5 0 \pm 3 . 4 2$ </td><td> ${ \bf 8 6 . 8 6 \pm 0 . 0 5 }$ </td></tr><tr><td>Updated mapping after second consolidation</td><td> ${ \bf 8 7 . 0 2 \pm 1 . 7 6 }$ </td><td> $1 8 . 3 2 \pm 0 . 0 4$ </td></tr></table>

## 6.3 Metrics and statistics

Updated-mapping LTM recall is accuracy on the final second-segment query after both demonstration contexts have been consolidated and the KV cache and STM have been cleared. Second-segment immediate STM recall evaluates the updated mapping before the second consolidation; mean immediate STM recall averages the corresponding pre-consolidation queries across both segments. The fresh-LTM control removes persistent memory, whereas the mismatched-LTM control supplies an experience with incorrect function parameters but the correct address and rule family.

We report mean ± sample SD over five seeds and use paired diferences for comparisons. Confidence intervals and t-tests are descriptive because n = 5. All main consolidation-training seeds share one STM-pretraining checkpoint, so their variation reflects consolidation optimization and data streams conditional on that learned STM representation.

## 7 Results

## 7.1 A learned Consolidator enables persistent updates from frozen STM

The central intervention freezes every STM-pretrained component that forms, routes, and reads STM and trains only the 12.35K-parameter slot transform.

Table 3 and Figure 2 show that identity accumulation transfers the initial mapping but fails after the second same-address write. The learned transform instead reaches 87.02 ± 1.76% updatedmapping LTM recall, a same-checkpoint gain of $6 8 . 7 0 \pm 1 . 7 6$ pp over forced identity while training 0.041% of the model. Because the backbone, router, and read/write projections are frozen, the gain must pass through the learned boundary transform and existing memory interface. The result also provides independent evidence that the upstream STM is functional: because Consolidator receives only routed STM, its slot-local transform could not recover the rule instantiated in the current memory episode unless that STM already encoded the relevant information. The lower recall of the initial mapping after the first consolidation reflects supervision only on the final updated mapping, not a claim about general retention.

## 7.2 Consolidated LTM is an access state, not only stored content

Post-reset recall alone does not show that retained LTM afects memory access. We therefore compare two Consolidator-only conditions that retain learned LTM and all read paths but difer in whether the direct LTM term in Equation (7) is present; each condition trains its own Consolidator.

![](images/d40d37fead2e0ea820bb393131c9c7fa4ce6460b831ed84f659e67c600b0972f.jpg)  
(a) Same-checkpoint boundary intervention.

![](images/99d737665b3db4f50d34ac27e0a68c75363b7ac6378801ce361b65de6e60d16d.jpg)  
(b) Identity carries; the learned transform updates.

![](images/f4eb8b3e89b5aadc974adadadf7bd9b23086db1ba2cf46c83ce99f1e122cefee.jpg)  
(c) Recall depends on latent content.

Figure 2: Consolidator-only intervention. Paired evaluations compare learned and forcedidentity consolidation, the first and second same-address writes, and correct, mismatched, and fresh memory.  
Table 4: Direct LTM-conditioned routing in the Consolidator-only setting. On and of runs share a byte-identical STM-pretraining initialization and paired consolidation-training seeds.
<table><tr><td>Direct LTM routing Learned LTM Identity LTM Learned - identity Segment-2 STM Mismatched LTM Fresh LTM</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Off</td><td> $4 4 . 3 8 \pm 1 . 9 4$ </td><td> $2 2 . 9 8 \pm 0 . 0 4$ </td><td> $+ 2 1 . 4 0 \pm 1 . 9 1$ </td><td> $\mathbf { 8 9 . 9 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $9 . 9 2 \pm 0 . 6 9$ </td><td> $1 1 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>On</td><td> ${ \bf 8 7 . 0 2 \pm 1 . 7 6 }$ </td><td> $1 8 . 3 2 \pm 0 . 0 4$ </td><td> $\mathbf { + 6 8 . 7 0 \ : \pm { \ : 1 . 7 6 } }$ </td><td> $\mathbf { 8 9 . 9 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $9 . 3 0 \pm 0 . 2 4$ </td><td> $1 1 . 0 0 \pm 0 . 0 0$ </td></tr></table>

Table 4 and Figure 3 show that direct routing raises updated-mapping LTM recall from $4 4 . 3 8 \pm 1 . 9 4 \%$ to $8 7 . 0 2 \pm 1 . 7 6 \%$ , a paired gain of $4 2 . 6 4 \pm 1 . 1 0 ~ \mathrm { p p }$ (95% CI [41.27, 44.01], $p = 1 . 0 7 \times 1 0 ^ { - 7 } )$ , while immediate STM recall remains exactly 89.90% in both conditions.

Without direct routing, learned consolidation still exceeds forced identity by $2 1 . 4 0 \pm 1 . 9 1$ pp, showing that LTM remains useful through the read path; direct routing provides the larger additional gain. Thus, consolidated LTM serves as both retrievable content and an access state that guides subsequent slot selection; Appendix A reports per-seed and rule-family results.

## 7.3 Recall requires the correct experience

Figure 2c and the routing-on row of Table 4 show that updated-mapping LTM recall from the same checkpoint falls from 87.02% with the correct experience to $9 . 3 0 \pm 0 . 2 4 \%$ with a mismatched experience and 11.00% with fresh memory. The mismatch preserves address and rule family, while function parameters vary across memory episodes. Updated-mapping recall therefore depends on the consolidated content rather than a fixed address association or the mere presence of memory.

## 7.4 Broader parameter-isolation checks

Under the parameter-isolation conditions summarized in Table 2, learned full reaches 93.70% and independently trained identity full reaches 91.34%; their $+ 2 . 3 6 \pm 3 . 3 0$ pp diference is not statistically resolved. We therefore do not claim that learned consolidation dominates a fully plastic identity system on this task. Training only the memory path and Consolidator reaches $9 0 . 7 0 \pm 0 . 5 1 \%$ showing that adaptation can be concentrated in the memory subsystem; Tables C1 and C2 report the aggregate and per-seed results in Appendix C.

![](images/8ccbef7d5037154b39ae74ac8658f2feece590cce6e3159a2d505214447d4f3a.jpg)  
(a) Updated-mapping LTM recall.

![](images/642860d74d6150f3e5dce6404b871d097779730d0125e5da98568db2b1cf092f.jpg)  
(b) Segment-2 immediate STM recall.

Figure 3: Direct LTM-conditioned routing. Lines pair corresponding consolidation-training seeds initialized from the same STM-pretraining checkpoint. Direct routing substantially improves updated-mapping LTM recall (a), while immediate STM recall of the second mapping remains exactly matched (b). The of condition retains learned LTM and all read paths; only direct same-level LTM conditioning of the router is removed.

## 7.5 Pre-consolidation STM and post-reset LTM recall can coexist

Adding pre-consolidation STM supervision raises mean immediate recall across both segments from $1 1 . 3 9 \pm 0 . 4 7 \%$ to 95.76 ± 0.72%, while updated-mapping LTM recall reaches $9 5 . 5 8 \pm 0 . 7 5 \%$ (Figure 4). The two metrics use separate cache trajectories and forward passes, not concurrent queries in one online trajectory. The result therefore shows that one parameter set can support both capabilities under a suitable objective, not that both were jointly read in a single pass; Appendix D reports aggregate statistics and per-seed results.

ADD10 is nearly saturated, but the Consolidator-only model also reaches $7 4 . 8 0 \pm 3 . 0 7 \%$ on AFFINE10; the central result is therefore not explained only by the simpler additive family.

## 8 Discussion

The results establish a functional chain: STM pretraining forms episode-specific routed state, Consolidator converts it for persistent revision, and replacing the retained experience removes the recall gain. Because the upstream memory interface is frozen and Consolidator receives no source tokens, its success implies that STM already contains the relevant information before consolidation.

The identity and routing interventions show that raw persistence and readout alone do not explain the result: the learned boundary supports conflicting revision, while direct routing provides the larger gain despite matched immediate STM recall. Consolidated LTM therefore functions as both retrievable content and an access state, making its direct connection to the frozen router a task-efective inductive bias for experience-dependent slot selection.

At inference, the fixed Consolidator and router adapt through mutable non-parametric state rather than parameter updates, distinguishing the mechanism from continual fine-tuning and test-time gradient descent.

![](images/2d666e7064a52d4073045992cc98dc19bc4c7457985ccfeee1916554fdb69eaf.jpg)  
(a) Updated-mapping LTM recall.

![](images/2a26b1ac4913ece0da99ab9edc227a76c71bd7528340a36e504dc3ba0582340c.jpg)  
(b) Mean immediate STM recall.  
Figure 4: Pre-consolidation and post-reset recall under a dual objective. Adding immediate-STM supervision restores mean pre-consolidation recall across both segments (b) without reducing updated-mapping LTM recall (a). The two metrics are evaluated on separate trajectories using one shared parameter set.

## 9 Limitations

Controlled scope. Memory episodes contain two short context segments, one active address, and modular-arithmetic rules. Both demonstration contexts fit inside the local attention window; the reset isolates persistence but does not test extreme within-segment context, natural language, many competing memories, long horizons, or systems eficiency.

Estimation and provenance. The five main consolidation-training seeds share one selected STM-pretraining representation, so their variance excludes variation from the first training stage. Training retains gradients across the two consolidation boundaries, leaving detached or truncated long-horizon training untested. With n = 5, confidence intervals and t-tests are descriptive.

Unisolated design choices. Identity is the principal same-checkpoint control, but we do not compare alternative learned overwrite, EMA, linear, or gated recurrent operators. Boundaries, commit decisions, and eviction are externally specified, and the synthetic task lacks a task-matched external architecture baseline.

Persistence semantics. LTM is initialized for each memory episode; persistence across unrelated sessions, serialization, and deployment restarts is not evaluated. STM, LTM, and consolidation denote computational timescales, not a biological model of memory or sleep.

## 10 Conclusion

We introduced Consolidator, a shared slot-local transform that converts routed STM into persistent LTM without replaying the source tokens. On a controlled same-address update task, training only its 12.35K parameters while freezing the rest of PMNet yields 87.02% updated-mapping recall, compared with 18.32% when the same checkpoint uses identity accumulation; replacing the retained experience removes this gain. A paired routing ablation further reduces recall from 87.02% to 44.38% while leaving immediate STM recall unchanged, showing that consolidated LTM supports later computation both as retrievable content and as an input to slot selection. These results establish a controlled forward-state adaptation mechanism, not yet a general long-term memory system; detached or truncated long-horizon training, natural language, and scale-up remain open.

## Reproducibility

The implementation will be made publicly available at https://www.github.com/swgoo/pmnet\_ consolidator. It records full run configurations, starting-checkpoint hashes, trainable parameter counts, per-seed selected checkpoints, and the fixed test stream shared across conditions. modeling\_pmnet.py contains the cache, routing, Consolidator, and persistent-state update; train\_pmnet\_ablation.py contains the procedural data generator, STM-pretraining stage (Phase 1), interventions, and consolidation-training runner (Phase 2). Per-seed results and complete hyperparameters are reported in the appendix.

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017. URL https://arxiv.org/abs/1706.03762.

[2] Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014. URL https://arxiv.org/abs/1410.5401.

[3] Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Alex Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu, and Demis Hassabis. Hybrid computing using a neural network with dynamic external memory. Nature, 538:471–476, 2016. doi: 10.1038/nature20101.

[4] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a fixed-length context. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019. URL https://arxiv.org/abs/1901.02860.

[5] Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. arXiv preprint arXiv:1911.05507, 2019. URL https://arxiv.org/abs/1911.05507.

[6] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (learn at test time): RNNs with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024. URL https://arxiv.org/abs/2407.04620.

[7] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024. URL https://arxiv.org/abs/2501.00663.

[8] Sungwoo Goo, Hwi-yeol Yun, and Sangkeun Jung. Phasor memory networks: Stable backpropagation through time for scalable explicit memory. arXiv preprint arXiv:2605.13370, 2026. URL https://arxiv.org/abs/2605.13370.

[9] Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Recurrent memory transformer. arXiv preprint arXiv:2207.06881, 2022. URL https://arxiv.org/abs/2207.06881.

[10] Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave no context behind: Eficient infinite context transformers with infini-attention. In Proceedings of the 41st International Conference on Machine Learning, 2024. URL https://arxiv.org/abs/2404.07143.

[11] Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2203.08913.

[12] Jimmy Ba, Geofrey Hinton, Volodymyr Mnih, Joel Z. Leibo, and Catalin Ionescu. Using fast weights to attend to the recent past. In Advances in Neural Information Processing Systems, 2016. URL https://arxiv.org/abs/1610.06258.

[13] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. URL https://arxiv.org/abs/2312.00752.

[14] Zangwei Zheng, Zhen Li, Xuehai Wen, et al. Context distillation as latent memory management. arXiv preprint arXiv:2605.28889, 2026. URL https://arxiv.org/abs/2605.28889.

[15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=NAQvF08TcyG.

[16] Zijian Zhang, Rizhen Hu, Athanasios Glentis, Dawei Li, Chung-Yiu Yau, Hongzhou Lin, and Mingyi Hong. Is one layer enough? training a single transformer layer can match full-parameter RL training. arXiv preprint arXiv:2607.01232, 2026. URL https://arxiv.org/abs/2607. 01232.

[17] James L. McClelland, Bruce L. McNaughton, and Randall C. O’Reilly. Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. Psychological Review, 102(3): 419–457, 1995. doi: 10.1037/0033-295X.102.3.419.

[18] Tyler L. Hayes, Giri P. Krishnan, Maxim Bazhenov, Hava T. Siegelmann, Terrence J. Sejnowski, and Christopher Kanan. Replay in deep learning: Current approaches and missing biological elements. arXiv preprint arXiv:2104.04132, 2021. URL https://arxiv.org/abs/2104.04132.

[19] Chenchen Ye, Yejin Liu, Yujie Wang, et al. Auto-dreamer: Learning ofline memory consolidation for language agents. arXiv preprint arXiv:2605.20616, 2026. URL https://arxiv.org/abs/ 2605.20616.

## A Direct LTM Routing by Seed and Rule Family

Table A1 reports the pooled paired routing results for each seed. Table A2 then stratifies learned-LTM recall by rule family. Direct LTM routing improves both families: the simpler ADD10 family approaches saturation, while AFFINE10 retains a paired gain of $3 8 . 5 6 \pm 1 . 2 9 \mathrm { p p }$ . The overall routing efect is therefore not attributable only to the additive rules.

Table A1: Per-seed direct LTM-routing ablation in the Consolidator-only setting. Routingon and routing-of runs start from the byte-identical Phase-1 checkpoint and use paired Phase-2 seeds, but each trains its own Consolidator. Of learned LTM is updated-mapping recall after both consolidations and the final reset with direct same-level LTM routing disabled; Of identity LTM reevaluates that same routing-of checkpoint using forced identity accumulation, and Learned– identity is their same-checkpoint diference. Segment-2 STM, mismatched LTM, and fresh LTM are also measured in the routing-of condition. On–Of pairs learned-LTM recall from the standard routing-on run with learned-LTM recall from the routing-of run. Both routing conditions retain the fixed router and all LTM read paths. All values are percentages on the shared fixed test stream.
<table><tr><td>Seed Off learned LTM Off identity LTM Learned - identity Segment-2 STM Mismatched LTM Fresh LTM On - Off</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>42</td><td>45.30</td><td>23.00</td><td>+22.30</td><td>89.90</td><td>9.30</td><td>11.00</td><td>+42.00</td></tr><tr><td>43</td><td>46.40</td><td>23.00</td><td>+23.40</td><td>89.90</td><td>10.40</td><td>11.00</td><td>+41.30</td></tr><tr><td>44</td><td>42.90</td><td>23.00</td><td>+19.90</td><td>89.90</td><td>10.70</td><td>11.00</td><td>+44.00</td></tr><tr><td>45</td><td>41.80</td><td>22.90</td><td>+18.90</td><td>89.90</td><td>9.10</td><td>11.00</td><td>+42.40</td></tr><tr><td>46</td><td>45.50</td><td>23.00</td><td>+22.50</td><td>89.90</td><td>10.10</td><td>11.00</td><td>+43.50</td></tr><tr><td>Mean</td><td>44.38</td><td>22.98</td><td>+21.40</td><td>89.90</td><td>9.92</td><td>11.00</td><td>+42.64</td></tr><tr><td>SD</td><td>1.94</td><td>0.04</td><td>1.91</td><td>0.00</td><td>0.69</td><td>0.00</td><td>1.10</td></tr></table>

Table A2: Direct LTM-routing ablation by rule family. Routing-of and routing-on values are updated-mapping LTM recall from separately trained Consolidator-only runs that share the byte-identical Phase-1 checkpoint and paired Phase-2 seeds. On–Of diferences are computed within each seed before aggregation. Values are percentages, reported as mean ± sample SD over five seeds on the shared fixed test stream.
<table><tr><td>Rule family</td><td>Routing off</td><td>Routing on</td><td>Paired On-Off</td></tr><tr><td>ADD10</td><td> $5 2 . 5 2 \pm 1 . 3 7$ </td><td> $\mathbf { 9 9 . 0 1 \ : \pm { \ : 0 . 6 0 } }$ </td><td> $\mathbf { + 4 6 . 4 9 \ : \pm { \ : 1 . 2 3 } }$ </td></tr><tr><td>AFFINE10</td><td> $3 6 . 2 4 \pm 2 . 6 1$ </td><td> $\mathbf { 7 4 . 8 0 \ : \pm { \ : 3 . 0 7 } }$ </td><td> $\mathbf { + 3 8 . 5 6 \ : \pm { \ : 1 . 2 9 } }$ </td></tr></table>

## B Exact Memory-Episode Procedure

## B.1 Pseudocode

sample family f in {ADD10, AFFINE10}   
sample address a from four split tokens   
sample first-segment parameters theta\_1   
sample second-segment parameters theta\_2 != theta\_1   
for segment d in {1, 2}:   
sample 8 distinct demonstration inputs   
sample 1 held-out query input   
if d == 2:   
require f\_theta\_1(query) != f\_theta\_2(query)   
form 40-token demonstration context   
process context with routed STM writes   
consolidate occupied STM groups into LTM   
clear KV and STM; retain LTM   
present the second-segment held-out query without demonstrations   
apply loss only to the answer token

For the dual objective, a separate immediate-STM trajectory reprocesses each segment’s context and adds its pre-consolidation query loss to the updated-mapping LTM loss.

## C Per-Seed Core Results

Table C1: Core parameter-isolation ablations. Percent accuracy, mean ± SD over five seeds; LTM recall follows the final reset.
<table><tr><td>Condition</td><td>Updated LTM</td><td></td><td></td><td>Fresh Mismatched Segment-2 STM</td></tr><tr><td>Learned full</td><td> $\mathbf { 9 3 . 7 0 \pm 2 . 6 6 }$ </td><td> $1 0 . 8 4 \pm 0 . 5 4$ </td><td> $7 . 7 0 \pm 0 . 9 8$ </td><td> $1 1 . 9 4 \pm 0 . 6 7$ </td></tr><tr><td>Identity full</td><td> $\mathbf { 9 1 . 3 4 \ : \pm { \ : 2 . 9 0 } }$ </td><td> $1 1 . 1 2 \pm 0 . 2 8$ </td><td> $7 . 9 2 \pm 1 . 0 2$ </td><td> $1 8 . 1 6 \pm 4 . 3 4$ </td></tr><tr><td>Consolidator only</td><td> ${ \bf 8 7 . 0 2 \pm 1 . 7 6 }$ </td><td> $1 1 . 0 0 \pm 0 . 0 0$ </td><td> $9 . 3 0 \pm 0 . 2 4$ </td><td> $\mathbf { 8 9 . 9 0 \ : \pm { \ : 0 . 0 0 } }$ </td></tr><tr><td>Memory + Consolidator</td><td> $\mathbf { 9 0 . 7 0 \ : \pm { \ : 0 . 5 1 } }$ </td><td> $1 0 . 3 0 \pm 0 . 2 9$ </td><td> $9 . 4 4 \pm 0 . 3 1$ </td><td> $1 4 . 8 8 \pm 3 . 2 4$ </td></tr></table>

Table C2: Fixed-test-stream results for all 20 core runs. “Alternative mode” is a samecheckpoint diagnostic that changes only the consolidation operator at evaluation: learned conditions use forced raw-identity accumulation, whereas identity full uses its frozen, identity-initialized Consolidator path. The alternative mode is not independently trained.
<table><tr><td>Condition</td><td></td><td></td><td>Seed Primary recall Alternative mode Fresh Segment-2 STM Mismatched</td><td></td><td></td><td></td></tr><tr><td>Learned full</td><td>42</td><td>94.10</td><td></td><td>53.2010.90</td><td>11.50</td><td>7.60</td></tr><tr><td>Learned full</td><td>43</td><td>90.30</td><td></td><td>87.9010.80</td><td>11.50</td><td>9.30</td></tr><tr><td>Learned full</td><td>44</td><td>96.10</td><td></td><td>82.9011.00</td><td>12.20</td><td>6.90</td></tr><tr><td>Learned full</td><td>45</td><td>91.70</td><td></td><td>88.5011.50</td><td>13.00</td><td>7.80</td></tr><tr><td>Learned full</td><td>46</td><td>96.30</td><td></td><td>75.1010.00</td><td>11.50</td><td>6.90</td></tr><tr><td>Identity full</td><td>42</td><td>96.10</td><td></td><td>96.1011.40</td><td>12.60</td><td>7.10</td></tr><tr><td>Identity full</td><td>43</td><td>88.20</td><td></td><td>88.1011.10</td><td>20.20</td><td>9.10</td></tr><tr><td>Identity full</td><td>44</td><td>90.70</td><td></td><td>90.7010.80</td><td>23.70</td><td>8.70</td></tr><tr><td>Identity full</td><td>45</td><td>91.20</td><td></td><td>91.1010.90</td><td>19.10</td><td>6.70</td></tr><tr><td>Identity full</td><td>46</td><td>90.50</td><td></td><td>90.2011.40</td><td>15.20</td><td>8.00</td></tr><tr><td>Consolidator only</td><td>42</td><td>87.30</td><td></td><td>18.3011.00</td><td>89.90</td><td>9.20</td></tr><tr><td>Consolidator only</td><td>43</td><td>87.70</td><td></td><td>18.3011.00</td><td>89.90</td><td>9.20</td></tr><tr><td>Consolidator only</td><td>44</td><td>86.90</td><td></td><td>18.4011.00</td><td>89.90</td><td>9.50</td></tr><tr><td>Consolidator only</td><td>45</td><td>84.20</td><td></td><td>18.3011.00</td><td>89.90</td><td>9.60</td></tr><tr><td>Consolidator only</td><td>46</td><td>89.00</td><td></td><td>18.3011.00</td><td>89.90</td><td>9.00</td></tr><tr><td>Memory + Consolidator</td><td>42</td><td>91.60</td><td></td><td>85.0010.50</td><td>10.70</td><td>9.20</td></tr><tr><td>Memory + Consolidator</td><td>43</td><td>90.60</td><td></td><td>87.6010.10</td><td>16.70</td><td>9.10</td></tr><tr><td>Memory + Consolidator</td><td>44</td><td>90.40</td><td></td><td>72.7010.40</td><td>18.90</td><td>9.40</td></tr><tr><td>Memory + Consolidator</td><td>45</td><td>90.40</td><td>89.30</td><td>9.90</td><td>12.70</td><td>9.70</td></tr><tr><td>Memory + Consolidator</td><td>46</td><td>90.50</td><td></td><td>82.1010.60</td><td>15.40</td><td>9.80</td></tr></table>

## D Per-Seed Dual-Objective Results

This experiment asks whether the low immediate-STM recall observed when training only for the final post-reset query reflects an architectural incompatibility between STM and LTM, or simply the absence of direct STM supervision. Both conditions use the fully trainable model, share the same Phase-1 STM-pretraining checkpoint, and pair the Phase-2 seeds and data streams. They difer in their training objective and checkpoint-selection metric, so this comparison is between independently optimized objectives rather than a same-checkpoint intervention.

The Updated LTM only condition optimizes the final second-segment query after both consolidations and the final KV/STM reset. The Updated $L T M + S T M$ condition adds the mean loss of two auxiliary pre-consolidation queries, one for each segment. These immediate-STM queries are evaluated on separate auxiliary trajectories; they share model parameters with the persistentmemory trajectory but do not interrupt or supply information to it. In the tables, Updated LTM is final updated-mapping recall, Mean STM averages immediate recall across the two segments, and Segment-2 STM reports immediate recall of the updated mapping alone.

Table D1: Aggregate dual-objective experiment. Paired runs share seeds and STM-pretraining provenance; the consolidation-training objective and validation monitor difer.
<table><tr><td>Training objective</td><td>Updated LTM</td><td>Mean STM</td><td>Segment-2 STM Mismatched</td><td></td><td>Fresh</td></tr><tr><td>Updated LTM only</td><td> $9 3 . 7 0 \pm 2 . 6 6$ </td><td> $1 1 . 3 9 \pm 0 . 4 7$ </td><td> $1 1 . 9 4 \pm 0 . 6 7$ </td><td> $7 . 7 0 \pm 0 . 9 8$ </td><td> $1 0 . 8 4 \pm 0 . 5 4$ </td></tr><tr><td>Updated LTM + STM</td><td> ${ \bf 9 5 . 5 8 \pm 0 . 7 5 }$ </td><td> ${ \bf 9 5 . 7 6 \pm 0 . 7 2 }$ </td><td> $\mathbf { 9 5 . 8 6 \ : \pm { \ : 0 . 9 1 } }$ </td><td> $8 . 5 4 \pm 0 . 3 0$ </td><td> $1 0 . 9 6 \pm 0 . 6 7$ </td></tr><tr><td>Paired change</td><td> $+ 1 . 8 8 \pm 2 . 6 9$ </td><td> $\mathbf { + 8 4 . 3 7 \pm 0 . 7 5 }$ </td><td> $\mathbf { + 8 3 . 9 2 \ : \pm { \ : 0 . 5 9 } }$ </td><td></td><td></td></tr><tr><td>95% CI</td><td> $\left[ - 1 . 4 6 , + 5 . 2 2 \right]$ </td><td> $[ + 8 \mathbf { 3 } . 4 4 ,  + 8 \mathbf { 5 } . \mathbf { 3 0 } ]$ </td><td> $\mathbf { \left[ + 8 3 . 1 9 , \hbar + 8 4 . 6 5 \right] }$ </td><td></td><td></td></tr><tr><td>Paired p</td><td>0.193</td><td> $1 . 5 2 \times 1 0 ^ { - 9 }$ </td><td> $5 . 8 3 \times 1 0 ^ { - 1 0 }$ </td><td></td><td></td></tr></table>

Adding direct STM supervision raises mean immediate recall by $8 4 . 3 7 \pm 0 . 7 5$ pp while retaining $9 5 . 5 8 \pm 0 . 7 5 \%$ updated-mapping LTM recall. The $+ 1 . 8 8 \pm 2 . 6 9$ pp change in updated LTM recall is not statistically resolved; the result therefore supports coexistence of the two capabilities under one parameter set, not an improvement in LTM attributable to the auxiliary objective. The paired confidence intervals and t-tests are descriptive because $n = 5$

Table D2: Per-seed dual-objective results.
<table><tr><td></td><td></td><td>Seed Best epoch Validation dual</td><td>Test dual </td><td>Updated LTM Mean STM</td><td>Forced identity</td></tr><tr><td>42</td><td>10</td><td>95.07</td><td>95.00</td><td>94.90</td><td>95.10 84.60</td></tr><tr><td>43</td><td>14</td><td>95.42</td><td>94.92</td><td>94.90</td><td>94.95 72.30</td></tr><tr><td>44</td><td>20</td><td>95.90</td><td>96.10</td><td>96.10</td><td>96.10 89.20</td></tr><tr><td>45</td><td>22</td><td>96.20</td><td>96.63</td><td>96.60</td><td>96.65 67.30</td></tr><tr><td>46</td><td>16</td><td>95.10</td><td>95.70</td><td>95.40</td><td>96.00 71.70</td></tr></table>

For the per-seed table, Validation dual and Test dual are the arithmetic means of updatedmapping LTM recall and mean immediate-STM recall on their respective splits. Forced identity reevaluates the selected dual-objective checkpoint with raw identity accumulation in place of the learned Consolidator; it is a same-checkpoint diagnostic, not a separately trained identity condition.

## E Hyperparameters and Architecture

## Table E1: Model, optimization, and hardware settings.

<table><tr><td>Category</td><td>Setting</td></tr><tr><td>Backbone initialization</td><td>PMNet copy-task checkpoint, followed by same-segment STM rule induction (Phase 1)</td></tr><tr><td>Total parameters</td><td>29.95M</td></tr><tr><td>Transformer</td><td>12 layers; hidden size 384; FFN size 1,024; 12 query and 12 KV heads</td></tr><tr><td>Local attention</td><td>Sliding window 128; attention dropout 0.1; RMSNorm  $\epsilon = 1 0 ^ { - 6 }$ </td></tr><tr><td>Memory hierarchy</td><td>Four blocks; branch factor 4; 85 routing groups; 340 candidate slot vectors</td></tr><tr><td>Memory features</td><td>Phase dimension 32; four read heads; writes at layers 0, 3, 6, and 9</td></tr><tr><td>Consolidator</td><td>Intermediate size 64; SiLU gate; 12.35K shared parameters; identity phase initialization</td></tr><tr><td>Optimizer</td><td>AdamW;  $\beta = ( 0 . 9 , 0 . 9 5 )$  ; learning rate  $5 \times 1 0 ^ { - 4 } ;$  weight decay 0.1</td></tr><tr><td>Schedule</td><td>100 warmup steps, then cosine decay</td></tr><tr><td>Gradient clipping</td><td>Global norm 1.0</td></tr><tr><td>Precision and batch</td><td>bf16-mixed; global batch size 256; one device</td></tr><tr><td>Hardware</td><td>One NVIDIA RTX 4090; CUDA 13.0; FlashAttention 2</td></tr><tr><td>Software</td><td>Python 3.12.3; PyTorch 2.10.0+cu130; Transformers 5.14.1; Lightning 2.6.5</td></tr><tr><td>Training budget</td><td>100K memory episodes/epoch; at most 60 epochs; patience 6</td></tr><tr><td>Evaluation</td><td>1K validation and 1K fixed test memory episodes</td></tr><tr><td>Phase-2 seeds (consolidation)</td><td>42, 43, 44, 45, 46</td></tr><tr><td>Temporal graph</td><td>detach_long_term_between_sleeps=False for reported adaptation runs</td></tr></table>
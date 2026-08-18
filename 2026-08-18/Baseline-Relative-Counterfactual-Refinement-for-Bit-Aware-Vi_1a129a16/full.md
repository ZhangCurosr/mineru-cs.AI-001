# Baseline-Relative Counterfactual Refinement for Bit-Aware Visual Token Communication

Jia Guo, Member, IEEE, Xiaohan Zhao, Changwang Liu\*, Member, IEEE, Shuqing He, Chenyang Zhang, Bingchuan Zhao, Jinqi Zhu

Generative visual-token communication reduces transmission load by sending only selected discrete tokens and reconstructing missing content at the receiver. However, existing token-selection criteria based on local uncertainty, importance, or diversity do not directly determine whether changing the current selection improves the final reconstruction under the same packet budget. To address this problem, we propose Gated Counterfactual Refinement for Communication (GCR-C), a rollout-style correction layer over Local-MDL. GCR-C constructs a compact diversified candidate set, evaluates each candidate through matched full-budget Local-MDL continuation, and replaces the baseline action only when a positive baseline-relative reconstruction gain is obtained. Experiments on CIFAR-10, STL-10, a coded 5G-LDPC link, and a limited high-resolution Kodak transfer show that GCR-C consistently improves reconstruction quality at active low- and medium-rate operating points without increasing the realized packet rate, while remaining effective across changes in dataset, channel condition, resolution, token grid, and tokenizer. The results also reveal a clear quality–computation tradeoff due to the additional encoder-side counterfactual evaluation.

Index Terms—Semantic communications, visual tokens, generative recovery, rate–distortion, rollout policy improvement.

## I. INTRODUCTION

Recent advances in discrete visual representation and generative modeling are reshaping image communication. Vectorquantized representations convert images into discrete visual tokens, while masked generative models can recover missing tokens from partial observations [4]–[6]. Together with learned image compression, deep joint source–channel coding, and semantic communication [7]–[10], [17], this enables the transmitter to send only selected visual evidence and rely on receiver-side generative recovery. Under a constrained link, reconstruction quality therefore depends not only on coding efficiency, but also on which visual tokens are transmitted.

This paradigm is particularly relevant when communication resources are scarce but transmitter-side computation is comparatively available. Deep-space and satellite communications are representative examples: scientific instruments and highresolution imagers can generate large data volumes, whereas long-distance propagation, limited bandwidth, constrained transmit power, and intermittent connectivity restrict return rates [1]. Meanwhile, modern onboard processors can perform sub stantial preprocessing and data reduction before transmission [2]. Similar conditions arise in remote sensing and computecapable edge platforms. In such systems, additional transmitter computation can be worthwhile if it improves the utility of the information delivered under the same packet budget, motivating a compute-for-rate design.

The resulting problem is not merely how many tokens to transmit, but which tokens should consume the available budget. A transmitted token carries its own visual evidence while also providing context for recovering untransmitted tokens. Its value therefore depends on the already protected set and the subsequent completion process. Moreover, under a bitaccurate packet protocol, token positions also consume bits, so packet cost depends on the selected set rather than only on its cardinality. The key question is therefore: if the current baseline action is replaced by another feasible token, will the completed reconstruction improve under the same budget and continuation policy?

Existing visual-token reduction methods use uncertainty, attention, redundancy, diversity, contextual relevance, or semantic importance to prioritize tokens. Progressive compression, diversity-aware pruning, clustering, representation-shift analysis, and redundancy modeling have been studied for visualtoken reduction [19]–[25]. Adaptive, contextual, languageguided, and training-free selection methods provide additional signals [18], [26], [27], [31]–[33], while communicationoriented studies have explored text-guided transmission, adaptive semantic tokens, semantic subspaces, and generative image communication [10], [35]–[37].

These methods provide efficient selection cues, but they do not directly measure the baseline-relative terminal reconstruction value of a feasible action under the same packet budget. A high-scoring token may be redundant with already transmitted evidence, whereas a lower-scoring but spatially complementary token may improve generative recovery. More importantly, an action that appears better locally may become inferior after the remaining budget is completed. Instantaneous scores are therefore only proxies for the final reconstruction objective.

Directly optimizing the final reconstruction is difficult. Searching all packet-feasible token subsets is combinatorial, while exhaustively evaluating every feasible next action with full-budget completion is also expensive. The practical challenge is therefore to identify useful deviations from a strong baseline without turning each transmission decision into

exhaustive search.

To address this problem, we propose Gated Counterfactual Refinement for Communication (GCR-C), a baseline-relative correction layer over Local-MDL. Local-MDL remains the default policy. At an eligible state, GCR-C constructs a compact diversified proposal containing high-value Local candidates and spatially complementary alternatives. Each candidate is taken once counterfactually, after which the remaining packet budget is completed using the same Local-MDL continuation. Its full-budget reconstruction quality is then compared with that obtained from the original Local-MDL action under the identical continuation. GCR-C replaces the baseline action only when the best candidate has strictly positive baseline-relative advantage; otherwise, it falls back to Local-MDL. Rate/progress gating and a limit on accepted interventions further control encoder-side computation.

GCR-C is related to rollout-style policy improvement, where a candidate action is taken once and the remaining decisions follow a fixed base policy [11]. We do not claim the generic rollout principle as new. Our contribution is its specialization to bit-accurate generative visual-token communication, where feasibility is determined by the serialized packet, candidate value is defined by same-budget receiver-side reconstruction, and counterfactual evaluation incurs non-negligible encoder cost. This differs from world-model-based semantic communication for Physical-AI systems, which uses imagined rollouts to evaluate long-horizon control return [12]; here, the objective is same-image reconstruction under a fixed receiver continuation.

The intended operating regime of GCR-C is therefore one in which communication is more constrained than transmitterside computation. Representative scenarios include bandwidthor power-limited satellite imaging links, deep-space scientificimage return from compute-capable spacecraft, remote-sensing platforms, and other compute-capable edge transmitters connected through restricted links. In such systems, spending additional computation to select more useful visual evidence can be preferable to transmitting more bits. Conversely, the current full-budget evaluator is less suitable for ultra-lowpower transmitters or strict real-time applications where encoder computation and latency are dominant constraints.

The main contributions of this work are summarized as follows:

1) We formulate a packet-feasible, baseline-relative fixedbudget correction problem for generative visual-token communication. Candidate and baseline actions are compared through matched full-budget completion under the same packet syntax, transmission budget, and Local-MDL continuation, and we establish a baseline-preservation proposition under exact deterministic matched completion.

2) We develop GCR-C, a non-learned correction layer that combines a compact diversified proposal, fullbudget Local-MDL continuation, baseline-relative gating, rate/progress eligibility, Local fallback, and a limit on accepted interventions, without redesigning the tokenizer, masked prior, packet format, or receiver pipeline.

3) We introduce counterfactual headroom, proposal regret, and quality–computation diagnostics to quantify the opportunity remaining above Local-MDL, the fraction exposed by a compact proposal, and the encoder-side cost required for refinement.

4) We evaluate GCR-C across multiple rates, datasets, channel conditions, and representation scales. On a heldout 500-image CIFAR-10 test split, GCR-C improves PSNR over Local-MDL by 1.503 dB at 0.20 bpp and 0.698 dB at 0.32 bpp without increasing the realized packet rate. Frozen selections retain positive paired gains under all 12 tested 5G-LDPC/QPSK AWGN and block-Rayleigh conditions. An independently trained 96 × 96 STL-10 system and a limited 384 × 384 Kodak transfer further test the mechanism under changes in dataset, resolution, token grid, and vocabulary.

## II. RELATED WORK

## A. Discrete visual representations and generative recovery

Vector-quantized representations make images accessible to discrete sequence models, while masked generation provides a receiver-side mechanism for recovering untransmitted tokens [4]–[6]. Learned compression and autoregressive or hierarchical priors similarly connect a rate budget with a probabilistic description of missing content [16], [17]. In semantic communication, deep joint source–channel coding, learned semantic transmission, and generative image communication have emphasized task-aware recovery rather than pixel-perfect transport [7]–[9]. These works motivate our protocol, but generally do not isolate the incremental value of a candidate set under a fixed packet syntax and a fixed completion policy.

## B. Visual-token reduction, contextual selection, and semantic communication

Recent work reduces visual-token redundancy through progressive or diversity-aware compression, clustering, representation-shift analysis, and token-redundancy modeling [19]–[25]; token skipping and adaptive pruning have also been studied for multimodal and video systems [26]–[30]. Context-aware pruning, language-guided tokenization, trainingfree selection, and inference-optimal allocation broaden the signals available to a selector [18], [31]–[34]. These methods establish token count as a useful control variable, but generally target acceleration or multimodal inference rather than packetlevel rate–distortion decisions with receiver-side generative completion.

Communication-oriented studies have explored text-guided transmission and adaptive semantic tokens for edge inference [35], [36]; newer directions include matchable semantic subspaces, attention-driven self-compression, reading-twice pruning, energy-aware pruning, and robustness of compressed visual tokens [37]–[41], while wireless image-compression work exposes coupled robustness, throughput, and latency constraints [42]. SQ-GAN combines semantic-conditioned masking with vector quantization for very-low-rate image communication [10]. GCR-C is complementary: it does not redesign the tokenizer or learn a semantic mask, but uses local, set, spatial-coverage, and contextual proposals and asks whether a feasible replacement improves the completed reconstruction under the same packet budget and continuation policy.

![](images/a6ace1338097181ea91c0e279ecefb4f6a01708a10b14fd2ab88a9a9eacb6664.jpg)  
Fig. 1. System architecture of GCR-C. A compact proposal is evaluated by full-budget Local-MDL completion, and an alternative action is accepted only when its baseline-relative advantage is positive. Selected tokens are packetized and transmitted through the coded link; receiver reconstruction is used only for offline evaluation.

## C. Rollout-style policy improvement and self-positioning

From approximate dynamic programming, GCR-C is related to rollout-based policy improvement, where a candidate action is taken once and the remainder is completed with a known base policy [11]. The rollout principle itself is therefore not claimed as new. Our specialization is to a bit-accurate generative visual token communication problem in which feasibility depends on complete packet syntax and selected positions, exhaustive first-action evaluation is expensive, and the receiver uses generative completion. We add a compact diversified proposal, baseline-relative headroom and proposal-regret diagnostics, ratedependent gating, explicit encoder-computation accounting, and coded-link validation. In contrast to long-horizon task return or closed-loop physical-agent planning, the present objective is same-budget image reconstruction under a fixed receiver contin uation. Recent work on physical-AI semantic communication evaluates semantic-token interventions through world-modelenabled counterfactual imagined rollouts to estimate longhorizon return-per-bit [12]. That setting couples communication with future control and state evolution. GCR-C addresses a different problem: same-budget image reconstruction under a fixed receiver continuation, with Local-MDL serving as an explicit base policy and packet feasibility determined by the serialized token packet.

Prior work provides effective local, contextual, spatial, and semantic signals for reducing visual tokens, but these signals are typically optimized for pruning efficiency, downstream inference, or adaptive transmission. GCR-C addresses the narrower decision of whether a feasible replacement improves receiver reconstruction after both choices are completed under the same budget and continuation policy. We adapt the published DivPrune max–min diversity principle under our packet interface as a matched external selector [21].

## III. PROBLEM FORMULATION AND BIT-ACCURATE COMMUNICATION PROTOCOL

## A. Sequential baseline-correction problem

Given a receiver reconstruction function $d ( S )$ and a transmission budget $B _ { \mathrm { m a x } } .$ the communication objective is to construct a feasible transmitted set with high completed reconstruction quality subject to $B ( S ) \leq B _ { \mathrm { m a x } }$ . Direct global subset search is combinatorial and is not attempted here. Instead, we study a sequential decision problem in which a default Local-MDL action is available at each state and ask whether a feasible alternative can improve the same-budget completed reconstruction.

## B. Tokenized image and packet budget

Let an image $x \in [ 0 , 1 ] ^ { H \times W \times 3 }$ be divided into N nonoverlapping patches and quantized into tokens $z = ( z _ { 1 } , \dots , z _ { N } )$ from a vocabulary of size V. For a selected set $S = \{ p _ { 1 } <$ $\cdot \cdot \cdot < p _ { m } \} \subseteq \{ 0 , \ldots , N - 1 \}$ , the receiver observes the selected token values and their zero-based positions. The packet uses one mode bit followed by either an N-bit bitmap or a gap list. With $b _ { m } = \lceil \log _ { 2 } ( N + 1 ) \rceil$ and $b _ { p } = \lceil \log _ { 2 } N \rceil$ , the gap-list length is

$$
\begin{array} { l } { \displaystyle { B _ { \mathrm { g a p } } ( S ) = b _ { m } + { \bf 1 } _ { m > 0 } \left( b _ { p } + \sum _ { j = 2 } ^ { m } \gamma ( p _ { j } - p _ { j - 1 } ) \right) , } } \\ { \displaystyle \gamma ( u ) = 2 \lfloor \log _ { 2 } u \rfloor + 1 . } \end{array}\tag{1}
$$

where the cardinality field terminates the list: the empty set has no first-position field, and a nonempty list contains the first absolute position and exactly $m - 1$ positive Elias-gamma-coded gaps. The position cost is therefore

$$
B _ { \mathrm { p o s } } ( S ) = 1 + \operatorname* { m i n } \{ N , B _ { \mathrm { g a p } } ( S ) \} .\tag{2}
$$

The one-bit mode selects the shorter representation, so bitmap and gap-list fields are never charged twice. The core packet includes a 32-bit header, the position description, a $\lceil \log _ { 2 } V \rceil$ bit codeword per selected token, and a 16-bit CRC. For the CIFAR-10/STL-10 systems with $V = 3 2$ , this payload is 5 bits per token. Forward-error protection is applied once to the complete core packet:

$$
\begin{array} { c } { B _ { \mathrm { c o r e } } ( S ) = 3 2 + B _ { \mathrm { p o s } } ( S ) + m \lceil \log _ { 2 } V \rceil + 1 6 , } \\ { B ( S ) = B _ { \mathrm { t x } } ( S ) = \lceil 1 . 2 5 B _ { \mathrm { c o r e } } ( S ) \rceil . } \end{array}\tag{3}
$$

Equivalently, $B _ { \mathrm { f e c } } = B _ { \mathrm { t x } } - B _ { \mathrm { c o r e } } ;$ the implementation uses this multiplicative form and rounds only after summing the core bits. The realized packet rate is

$$
R ( S ) = { \frac { B ( S ) } { H W } } \ \mathrm { b p p } .\tag{4}
$$

In the CIFAR-10 implementation, $N = 6 4 , V = 3 2 ,$ and indices are zero-based. The encoder sorts the positions before coding, while the decoder reads the mode, then either all N bitmap entries or $m ,$ , the first position, and exactly m − 1 gaps; empty, singleton, and full sets thus have unique decodable syntax. For the coded-link experiment, the serialized information packet has $B _ { \mathrm { c o r e } }$ bits and the packet-accounting proxy is $B _ { \mathrm { t x } } = \lceil 1 . 2 5 B _ { \mathrm { c o r e } } \rceil$ . The physical LDPC implementation separately pads the information packet to

$$
k = 1 6 \left\lceil { B _ { \mathrm { c o r e } } } / { 1 6 } \right\rceil , \qquad n = \mathrm { r o u n d } ( k / 0 . 8 ) ,
$$

so that the realized code rate is $R _ { c } = k / n = 0 . 8 .$ We therefore distinguish the selection rate $R _ { \mathrm { p a c k e t } } ~ = ~ B _ { \mathrm { t x } } / ( H W )$ from the realized PHY rate $R _ { \mathrm { P H Y } } = n / ( H W )$ ; padding and rate matching are not silently counted as a second FEC factor in Eq. (3). In the formal records, the 0.20-bpp packets have $B _ { \mathrm { c o r e } } ~ = ~ 1 5 9 { - } 1 6 3 , ~ B _ { \mathrm { t x } } ~ = ~ 1 9 9 { - } 2 0 4 , ~ k ~ \in ~ \{ 1 6 0 , 1 7 6 \} |$ , and $n \in \{ 2 0 0 , 2 2 0 \}$ ; the 0.32-bpp packets have $B _ { \mathrm { c o r e } } ~ = ~ 2 5 8 -$ 261, $B _ { \mathrm { t x } } = 3 2 3 \substack { - 3 2 7 } , k = 2 7 2$ , and $n = 3 4 0$ . The resulting mean $( R _ { \mathrm { p a c k e t } } , R _ { \mathrm { P H Y } } )$ values are (0.1985, 0.2146) for Local-MDL and (0.1987, 0.2147) for GCR-C at 0.20 bpp, and (0.3157, 0.3320) for both methods at 0.32 bpp. For a budget cap $B _ { \mathrm { m a x } }$ , the feasible action set is

$$
\mathcal { F } ( S , B _ { \operatorname* { m a x } } ) = \{ i \notin S : B ( S \cup \{ i \} ) \leq B _ { \operatorname* { m a x } } \} .\tag{5}
$$

All selector comparisons use Eqs. (1)–(4) for packet-budget feasibility and report the realized packet rate $R _ { \mathrm { p a c k e t } } ;$ ; the coded-link experiment additionally distinguishes the realized physical-layer rate $R _ { \mathrm { P H Y } }$

## C. Channel abstraction and cognitive transmitter state

The clean packet protocol above admits a generic receiver available-set abstraction. For a channel realization $\omega ,$ let $\mathcal { C } _ { \omega } ( S )$ denote the set of selected token positions available to the receiver after packet transmission and decoding, and define

$$
S _ { \mathrm { r x } } = { \mathcal C } _ { \omega } ( S ) , \qquad d _ { \omega } ( S ) = d ( { \mathcal C } _ { \omega } ( S ) ) .\tag{6}
$$

This mapping can represent position-level erasures or a wholeframe decoding failure. In the coded-link experiment, $\mathcal { C } _ { \omega } ( S ) =$ S when CRC decoding succeeds and $\mathcal { C } _ { \omega } ( S ) = \emptyset$ when it fails; the same definition also remains compatible with future burstloss or token-level channel models. The receiver reconstructs from $S _ { \mathrm { r x } }$ using the same masked prior.

For the coded-link experiment, the same serialized packet is passed through a coded bit-level channel. Bits are CRCprotected, mapped to QPSK, encoded with a 5G-NR LDPC block at an approximately 0.8 code rate, and decoded with 20 belief-propagation iterations. We use complex AWGN and single-tap block-Rayleigh fading with perfect receiver CSI; a failed CRC therefore maps the selected set to ∅. This produces a reproducible PHY stress test while keeping the GCR-C selection frozen and channel independent.

The transmitter decision state is written as

$$
s _ { t } = \bigl ( S _ { t } , B _ { \mathrm { r e m , t } } , u _ { t } , c _ { t } \bigr ) ,\tag{7}
$$

where $S _ { t }$ is the protected token set, $B _ { \mathrm { r e m , t } }$ is the remaining packet budget, $u _ { t }$ collects local recoverability or receiver-state features, and $c _ { t }$ is optional channel context such as an estimated erasure level. The four stages in Fig. 1 are: (i) perception of $( S _ { t } , B _ { \mathrm { r e m , t } } , u _ { t } , c _ { t } )$ ; (ii) reasoning by proposal construction and $Q _ { B }$ completion; (iii) action through the baseline-relative gate; and (iv) adaptation of the set, budget, and eligibility state. In the clean experiments, $c _ { t }$ is a constant null context and no channel feedback is used. This is a deterministic decision loop, not a reinforcement-learning formulation.

## D. Receiver-side completion

The masked prior $p _ { \theta }$ receives known tokens and a mask at untransmitted positions. A deterministic one-pass completion produces $\hat { z } _ { \bar { S } }$ and an image reconstruction

$$
\hat { x } ( S ) = D _ { \theta } ( z _ { S } , \hat { z } _ { \bar { S } } ) , \qquad d ( S ) = \mathrm { P S N R } ( x , \hat { x } ( S ) ) .\tag{8}
$$

The one-pass rule is held fixed in Local-MDL, GCR-C, and Random comparisons. It is intentionally simple so that the experiment measures selection policy rather than a changing decoder schedule.

## IV. GATED COUNTERFACTUAL REFINEMENT

GCR-C operates as a correction layer on top of Local-MDL. At an eligible outer state, Local-MDL first provides the default action, and a compact proposal augments it with spatially complementary alternatives. Each proposal action is evaluated by fixing it as the next transmission and completing the remaining budget with Local-MDL. GCR-C replaces the default action only if the best candidate has positive full-budget advantage; after one accepted intervention, all remaining actions revert to Local-MDL.

## A. Local-MDL backbone

At state S, the local recoverability score is the masked negative log-likelihood

$$
m _ { i } ( S ) = - \log p _ { \theta } ( z _ { i } \mid z _ { S } ) , \qquad a _ { L } ( S ) = \operatorname * { a r g m a x } _ { i \in \mathcal { F } ( S , B _ { \operatorname * { m a x } } ) } \ m _ { i } ( S ) .\tag{9}
$$

The sign follows a description-length interpretation: a token that is difficult to predict from the known set is a valuable token to protect. Local-MDL repeatedly applies Eq. (9) until the packet budget is full. It is also the fallback when GCR-C is gated off.

## B. Baseline-relative fixed-budget value

Let $\pi _ { L }$ denote Local-MDL continuation. For a candidate action $a \in \mathcal { F } ( S , B _ { \mathrm { m a x } } )$ , define a horizon-h value by applying a once and then using Local-MDL for $h - 1$ further actions:

$$
Q _ { h } ( a \mid S ) = d \bigl ( \pi _ { L } ^ { h - 1 } ( S \cup \{ a \} ) \bigr ) ,\tag{10}
$$

where $Q _ { B }$ completes until no further action is feasible. The baseline-relative advantage is

$$
A _ { h } ( a \mid S ) = Q _ { h } ( a \mid S ) - Q _ { h } ( a _ { L } ( S ) \mid S ) .\tag{11}
$$

This subtraction removes the large image- and rate-dependent variation in absolute PSNR. It also makes the decision operational: a candidate is useful only if it beats the action the system would already take. The evaluator is not a theorem about the globally optimal set; it is a matched-completion measurement under the declared receiver policy. One outer candidate evaluation may contain multiple Local continuation actions and multiple $p _ { \theta }$ forward calls, so these events are counted separately rather than being collapsed into an unqualified “call” count.

For a channel-aware extension, use the receiver-available set mapping in Eq. (6) and define

$$
\begin{array} { r l } & { Q _ { B } ^ { \mathrm { c h } } ( a \mid S , c ) = \mathbb { E } _ { \omega \sim p ( \omega \mid c ) } \left[ d _ { \omega } \big ( \pi _ { L } ^ { B } ( S \cup \{ a \} ) \big ) \right] , } \\ & { A _ { B } ^ { \mathrm { c h } } ( a \mid S , c ) = Q _ { B } ^ { \mathrm { c h } } ( a \mid S , c ) - Q _ { B } ^ { \mathrm { c h } } ( a _ { L } \mid S , c ) . } \end{array}\tag{12}
$$

Equation (12) makes the optional context $c _ { t }$ operational: the evaluator can average the completed reconstruction over channel realizations rather than optimizing only the clean receiver. The present selector uses the clean $Q _ { B }$ in Eq. (10); the coded-link experiment applies channel errors after selection as a frozen-policy validation. Thus $Q _ { B } ^ { \mathrm { c h } }$ is a defined extension and not a claim that channel-aware selection has already been validated.

The candidate-level advantage can be negative, because an individually plausible action may be worse than the Local action. To separate this signed quantity from state-level opportunity, we define

$$
\begin{array} { r l } & { H _ { B } ( S ) = \underset { a \in \mathcal { F } ( S , B _ { \operatorname* { m a x } } ) } { \operatorname* { m a x } } A _ { B } ( a \mid S ) , } \\ & { \widetilde { H } _ { B } ( S ) = \underset { a \in \mathcal { P } ( S ) } { \operatorname* { m a x } } A _ { B } ( a \mid S ) . } \end{array}\tag{13}
$$

Because $a _ { L } ( S ) \in \mathcal { F } ( S , B _ { \operatorname* { m a x } } )$ and $A _ { B } ( a _ { L } \mid S ) = 0$ , the exhaustive first-action reference headroom $H _ { B } ( S )$ is nonnegative even though individual candidate advantages and their averages need not be. This reference enumerates feasible first actions at the current state and applies the fixed Local-MDL continuation; it is not a global subset optimum. The gap $H _ { B } ( S ) - \widetilde { H } _ { B } ( S )$ is the proposal regret used below.

## C. Multi-source proposal and gate

Exhaustively evaluating all feasible actions is expensive. GCR-C therefore forms

$$
{ \mathcal { P } } _ { C } ( S ) = \mathrm { T o p 3 } _ { m } ( S ) \cup \mathrm { T o p 5 } _ { c } ( S ) ,\tag{14}
$$

where $m$ is the Local-MDL score and c selects spatially complementary coverage candidates. Duplicate actions are removed. The Local action $a _ { L } ( S )$ is always inserted explicitly, so the proposal cannot discard the fallback. The proposal-source study also evaluates the P4 union, $\mathrm { T o p 3 } _ { m } \cup \mathrm { T o p 3 } _ { g } \cup \mathrm { T o p 2 } _ { c }$ as a source-ablation reference. The progress variable is

$$
r ( S _ { t } ) = \frac { | S _ { t } | } { N } ,
$$

where $S _ { t }$ is the selected set before the current outer action. An outer state is eligible for counterfactual gating iff $r ( S _ { t } ) \leq \rho _ { \mathrm { m a x } } ,$ where $\rho _ { \mathrm { m a x } } = 0 . 3 0$ . Thus the eligible selected-count states are 0–19 for the CIFAR-10 grid $( N = 6 4 )$ and 0–43 for the STL 10 grid $( N = 1 4 4 )$ . This parameter restricts when the gate may be evaluated; it is distinct from the one-intervention-perimage limit. The proposal and $Q _ { B }$ are evaluated at eligible states for 0.20 and 0.32 bpp until either one intervention is accepted or the packet budget terminates; after the first accepted intervention, all remaining actions follow Local-MDL. The 0.44 bpp point is not triggered.

The gate is

$$
a ^ { * } ( S ) = \left\{ \begin{array} { l l } { \arg \operatorname* { m a x } _ { a \in \mathcal { P } _ { C } ( S ) } A _ { B } ( a \mid S ) , } & { \operatorname* { m a x } _ { a } A _ { B } ( a \mid S ) > \delta , } \\ { a _ { L } ( S ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{15}
$$

with $\delta = 0$ and at most one intervention per image. Because a<sub>L</sub> is in $\mathcal { P } _ { C }$ , the computed reference policy is baseline-preserving under the same evaluator; the practical quality guarantee is limited to proposal and evaluator fidelity.

Proposition 1 (baseline preservation under exact matched completion). Assume that $a _ { L } ( S ) \in { \mathcal { P } } _ { C } ( S )$ and that $Q _ { B }$ is the deterministic quality obtained by taking one action and then applying the same Local-MDL continuation until no feasible action remains. If GCR-C selects the proposal maximizer only when its value is strictly larger than $Q _ { B } ( a _ { L } ( S ) \mid S )$ , and otherwise selects $a _ { L } ( S )$ , then its completed clean-channel quality is no smaller than Local-MDL under the same budget.

Proof. Since $a _ { L } ( S )$ belongs to $\mathcal { P } _ { C } ( S ) , \operatorname* { m a x } _ { a \in \mathcal { P } _ { C } ( S ) } Q _ { B } \big ( a \mid$ $S ) \geq Q _ { B } ( a _ { L } ( S ) \mid S )$ . If the strict gate is not satisfied, GCR-C reproduces the baseline action and continuation. If it is satisfied, it selects a candidate with strictly larger matchedcompletion value. All pre-intervention actions coincide with the Local-MDL baseline; after the intervention, both candidate and baseline values are evaluated using the same Local-MDL continuation policy from their respective post-action states. Because at most one intervention is accepted, the terminal cleanchannel quality is therefore no smaller than the matched Local-MDL baseline under the stated assumptions. The proposition is not a guarantee for an approximate horizon, learned value, stochastic decoder, mismatched channel evaluator, or global subset optimum. The baseline-preservation property applies only to the terminal metric used by $Q _ { B }$ , which is PSNR in the present implementation; it does not imply non-degradation of external semantic or perceptual metrics.

GCR-C is a non-learned correction policy in the present study; learning a lower-cost approximation of the proposal/evaluator pipeline is left to future work.

## V. EXPERIMENTAL SETUP AND EVALUATION PROTOCOL

## A. Data, model, and held-out evaluation

For CIFAR-10 [14], the system uses a 32-entry vectorquantized codebook over non-overlapping 4 × 4 RGB patches and a masked prior with 64 token positions. Thus $N = 6 4$ is the number of positions, $V = 3 2$ is the codebook size, and each transmitted token uses 5 payload bits. The codebook and prior are trained on 2,000 images, 500 images form the selector-development pool, and 100 images are reserved for validation. After the proposal, evaluator, gate, and operating-rate schedule are fixed on development/validation data, performance is reported on a disjoint 500-image held-out test split that is not used for policy selection.

The selected configuration uses $\mathcal { P } _ { C }$ (Top-3 Local plus Top-5 coverage), $Q _ { B } , \delta = 0 , \rho _ { \mathrm { m a x } } = 0 . 3 0 $ , at most one intervention, and eligible outer-state evaluation at 0.20 and 0.32 bpp; the gate is not evaluated at 0.44 bpp. Eligibility is evaluated from $r ( S _ { t } ) = | S _ { t } | / N$ using the selected set before the current outer action, giving states 0–19 for CIFAR-10 and 0–43 for STL-10. At each eligible state, the gate evaluates the deduplicated proposal and stops after the first accepted intervention or budget termination. Split manifests and evaluation artifacts are released with the paper.

## B. Validation studies and baselines

Before evaluation, we audited packet encode–decode consistency, candidate alignment, bit feasibility, deterministic Local-MDL actions, mask consistency, and explicit Local inclusion; full records are released. Headroom uses 50 images per rate with initial and early states clustered by image for bootstrap resampling. The horizon, proposal-source, and proposal-size studies retain the same packet accounting, continuation, decoder, and initial state, and use only development/validation data.

As a matched external baseline, we adapt the max–min diversity rule of DivPrune [21] to the same discrete-token and packet-feasible interface. Following the published rule, we compute cosine distances between pre-quantization raw $4 \times 4 ~ \mathrm { R G B }$ patch vectors and select tokens by global max– min initialization followed by farthest-first max–min updates. Because packet cost depends on selected positions, each step is restricted to the exact feasible set $\mathcal { F } ( S , B _ { \mathrm { m a x } } )$ in Eqs. (1)–(5). This is a protocol-matched adaptation of the published diversity principle rather than a reproduction of the original LMM task pipeline: the tokenizer, codewords, receiver completion, packet syntax, and held-out image indices are unchanged, while the raw-patch feature mapping, distance, tie-breaking, and packetfeasibility adaptation are fixed without using held-out results. The P4 union is retained in the headroom analysis as the pre-selection proposal reference used to characterize proposal miss-rate before the source ablation; the final $\mathcal { P } _ { C }$ allocation is selected subsequently using development/validation-only proposal-source comparisons.

## C. Metrics and computation accounting

We report PSNR, SSIM, packet bpp (the realized $R _ { \mathrm { p a c k e t } } )$ selected-token count, interventions, and the following computation counters: $N _ { \mathrm { p r o p } }$ is the number of deduplicated candidates formed for an image, $N _ { \mathrm { c f } }$ is the number of candidate-level $Q _ { h } / Q _ { B }$ evaluations executed, $N _ { \mathrm { r o l l } }$ is the number of Local actions appended inside those counterfactual continuations, $N _ { \mathrm { p r i o r } }$ is the number of actual $p _ { \theta }$ forward calls, $N _ { \mathrm { d e c } }$ is the number of reconstruction forwards, and $N _ { \mathrm { i n t } }$ counts accepted non-Local actions. $T _ { \mathrm { p e r f } }$ is the mean synchronized time in the complete 500-image held-out evaluation, whereas $T _ { \mathrm { t r a c e } }$ is the median-of-three time in the fixed first-50 instrumentation subset; the latter includes repeated measurements and logging and is not a population estimate.

Validation timing uses 20 warm-up forwards, CUDA synchronization before and after timing, and includes proposal evaluation plus the final one-pass reconstruction; full traces are retained in the reproducibility package. The worst-case preintervention candidate-evaluation bound follows directly from the eligibility rule and the maximum eight-action deduplicated proposal:

$$
N _ { \mathrm { c f } } ^ { \mathrm { m a x } } = 8 \left( \left\lfloor 0 . 3 0 N \right\rfloor + 1 \right) ,
$$

which gives 160 evaluations for $\mathrm { C I F A R - } 1 0 ~ ( N = 6 4 )$ and 352 for $\mathrm { S T L } \ – 1 0 \ ( N = 1 4 4 )$ . The realized number is lower when the packet budget terminates early, proposal deduplication reduces the candidate count, or an intervention is accepted before all eligible states are visited. For statistical analysis, paired comparisons use image-level bootstrap resampling; headroom states from the same image are retained together. The reported interval is the percentile 95% CI.

## D. Link-level and high-resolution transfer protocols

The link-level experiment reuses the frozen clean selections from the 500-image CIFAR-10 evaluation; no $Q _ { B }$ decision is re-optimized for a particular SNR. We implement the packet link with Sionna 2.0.1 [43]: CRC-protected packets are encoded by a 5G-NR LDPC block with a code rate close to 0.8, mapped to QPSK, and decoded by 20 belief-propagation iterations. For each coded block, the complex AWGN variance is set from the information-bit $E _ { b } / N _ { 0 }$ as

$$
\sigma ^ { 2 } = \Bigl [ R _ { c } \log _ { 2 } ( M ) 1 0 ^ { ( E _ { b } / N _ { 0 } ) / 1 0 } \Bigr ] ^ { - 1 } , \qquad M = 4 ,
$$

and the equalized block-Rayleigh branch uses $\sigma ^ { 2 } / | h | ^ { 2 }$ with perfect receiver CSI. We test complex AWGN and single-tap block-Rayleigh fading. A 50-image validation pilot sweep uses a 0.5-dB grid; for each rate and channel, hard/medium/easy points are selected from distinct observed SNRs by nearest FER to 0.525/0.30/0.075 in hard-to-easy order. The selected points are then frozen and evaluated on all 500 held-out images. Each image–condition pair uses one channel realization $( R = 1 ) $ the deterministic fading/noise seed is shared between Local MDL and GCR-C, and paired inference is performed before the image-level bootstrap. A CRC failure is counted as a frame error and exposes no selected token to the receiver, which then uses the same masked prior. Thus the experiment evaluates a frozen selection over a coded PHY link, rather than a channel aware selector or an over-the-air system.

For a higher-resolution transfer check, we use the official LlamaGen VQ-16 tokenizer [44] with a 16-pixel downsampling ratio, codebook size V = 16,384, embedding dimension $^ { 8 , }$ and a 24×24 grid (N = 576) for 384×384 center-cropped Kodak 24 images [45]. A lightweight six-layer, six-head masked prior of width 384 is trained from scratch on four deterministic crops of each of 800 DIV2K training images (3,200 sequences); 20 separate DIV2K images are used for validation. Two packet budgets are calibrated from eight validation images using the 75th percentile of exact packet bits, giving 2,101 bits (HR-Low) and 3,804 bits (HR-Mid). Because a complete outer-state $Q _ { B }$ sweep is substantially more expensive at $N = 5 7 6$ , this transfer diagnostic uses a pre-registered initial-state Top-3 Local plus Top-5 coverage gate, at most one intervention, and Local continuation thereafter; the CIFAR-10 and STL-10 core protocols retain the full eligible-state rule in Sec. IV. This experiment is used as a transfer diagnostic for the reconstruction-oriented decision mechanism under a substantially larger token grid and vocabulary. Separate semantic/perceptual diagnostics are provided in the supplementary material and are not used for candidate selection or policy calibration.

## VI. RESULTS

## A. Rate-dependent counterfactual headroom

Counterfactual headroom is concentrated at low rate but remains nonzero at the highest tested rate. At 0.20 bpp, the mean exhaustive first-action headroom is 2.45–2.58 dB and 84–86% of states exceed 0.10 dB. At 0.32 bpp, the mean falls to 0.73–0.81 dB and 52% exceed 0.10 dB; at 0.44 bpp, the mean is 0.27 dB and 34% exceed 0.10 dB. The intervals for the mean $H _ { B }$ are image-cluster bootstrap intervals, not statelevel independent intervals. Although $Q _ { B }$ remains positive at 0.44 bpp (+0.091 dB, 95% CI [0.023, 0.183]), the synchronized validation cost is approximately 2.61 s/image. The validationselected operating schedule therefore disables counterfactual evaluation at this rate. The separation in Eq. (13) remains important: first-action reference headroom is nonnegative by construction, whereas the candidate distribution can have a negative mean.

Complete state-level headroom and P4-regret statistics are provided in Supplementary Table S3; Fig. 2 retains the distributional and mechanism-level evidence in the main paper.

The proposal analysis also shows that the compact candidate set does not enumerate all feasible first actions: P4 regret is 0.95 dB at 0.20 bpp, 0.45–0.57 dB at 0.32 bpp, and 0.18–0.22 dB at 0.44 bpp in these states. Coverage candidates recover gaps left by Local and set-gain sources, but the remaining regret bounds the interpretation. Accordingly, we interpret GCR-C as a gated correction layer rather than an optimal subset selector.

## B. Full-budget value versus short-horizon approximations

Short-horizon value estimates do not reliably recover the full-budget decision, despite their lower latency. $Q _ { 1 } , Q _ { 2 }$ , and $Q _ { 4 }$ recover at most 14.1% of the $Q _ { B }$ gain at 0.32 bpp and can be worse than Local after the final-budget continuation;

their confidence intervals include zero at the active rates. The quality-first $Q _ { B }$ mode is the only horizon with a positive interval at both active rates, although it is not a low-latency approximation: its validation recovery is 1.0 by definition and its synchronized latency is 0.81–1.76 s per image at 0.20– 0.32 bpp. At 0.44 bpp, $Q _ { B }$ still gives +0.091 dB (95% CI [0.023, 0.183]) but costs about 2.61 s/image. We therefore use $Q _ { B }$ as the reference-quality evaluator and treat shorter horizons as lower-cost alternatives rather than substitutes for the fullbudget decision.

The synchronized validation counter measurements quantify the cost of the quality-first evaluator. In the initial-state diagnostic, each formed proposal candidate is evaluated exactly once, so $N _ { \mathrm { p r o p } } ~ = ~ N _ { \mathrm { c f } } ~ = ~ 7 . 6$ on average. The full-budget evaluator adds 93.1, 221.3, and 372.4 counterfactual Local rollout actions at the three rates. These measurements make the encoder-side computation visible alongside the quality evidence; the multi-state operating rule can accumulate more than one proposal size before the first accepted intervention.

## C. Proposal-source ablation

Diversified proposals improve access to counterfactual headroom, but the data do not support a universally superior coverage heuristic. Local+Coverage is the strongest compact source variant at the two active rates, improving over the previous P4 union by 0.250 dB at 0.20 bpp (paired 95% CI [0.025, 0.513]) and 0.206 dB at 0.32 bpp (CI [0.020, 0.477]). However, the direct paired Local+Coverage versus Randomcontrol difference is only +0.000 dB (CI [−0.333, 0.321]) at 0.20 bpp and +0.200 dB (CI [−0.019, 0.504]) at 0.32 bpp; neither interval is strictly positive. We therefore interpret the result as evidence for diversified proposals, with coverage-aware diversification becoming more advantageous at the medium rate, rather than as proof of a coverage-specific universal mechanism. The exhaustive row is an offline first-action reference under the Local-MDL continuation: it enumerates feasible actions at the current intervention state but does not solve the global budgeted subset-selection problem. It uses up to 64 feasible first actions and costs 4.38, 10.77, and 16.32 s per image at the three rates.

## D. Proposal-size quality–compute frontier

K = 8 lies near the main knee of the proposal-size quality– compute frontier. $K = 1 2$ improves over $K = 8$ by 0.202 dB at 0.20 bpp, but only 0.108 and 0.049 dB at 0.32 and 0.44 bpp while increasing synchronized latency by approximately 27–33%; K = 16 adds at most 0.044, 0.009, and 0.012 dB beyond $K = 1 2$ . The compact K = 8 point is therefore used for the final proposal. Within that budget, the source study selects a coverage-aware diversification allocation, yielding $\mathcal { P } _ { C }$ rather than the previous P4 union.

## E. Held-out CIFAR-10 performance

GCR-C converts validation-observed low-rate headroom into held-out PSNR gains without increasing realized packet rate. It improves over Local-MDL by 1.503 dB at 0.20 bpp and

![](images/99ffadbab824e93729449edcd226346c3a4a3fa3cbb32d559cc54b10ad285b36.jpg)

![](images/0217f6a3771560fd810bc8e9301b370b903d7e3a43c369bb8f15d926d1733803.jpg)

![](images/f32047ea6efbccf75dfd4d659da691b88f7919927ab0c72d8d3c59a5c67f582d.jpg)  
Fig. 2. Validation diagnostics: headroom CDF (a), state headroom/P4 regret with image-cluster CIs (b), horizon quality–latency frontier (c), and proposal-source ablation (d).

0.698 dB at 0.32 bpp, with strictly positive paired intervals and win rates of 97.4% and 90.4%, respectively. At 0.44 bpp, the selected operating rule does not evaluate the gate, so GCR-C follows Local-MDL and the two methods are identical. The fixed heuristics are informative but do not reproduce the same pattern: Entropy-greedy gives +0.902 dB at 0.20 bpp but only +0.024 dB with a CI crossing zero at 0.32 bpp, while Coveragegreedy gives +0.472 and −0.748 dB. Thus the active-rate result is not explained by a static entropy or coverage score alone.

The adapted max–min diversity baseline achieves 11.903, 13.909, and 16.876 dB PSNR at 0.20, 0.32, and 0.44 bpp, respectively, corresponding to −0.514, −0.986, and −1.713 dB relative to Local-MDL. At the two active rates, the direct paired differences between GCR-C and the adapted max–min diversity baseline are +2.017 dB (95% CI [+1.777, +2.258]) and +1.684 dB (CI [+1.459, +1.904]). The adapted max–min diversity selector therefore does not reproduce the activerate baseline-relative full-budget correction, supporting the counterfactual value comparison rather than a generic diversityonly explanation.

## F. Frozen selections over a coded wireless link

The coded-link experiment tests whether the clean-selection advantage survives packet decoding errors without changing either selector. Across two rates, two channels, and three distinct validation-pilot-calibrated $E _ { b } / N _ { 0 }$ points per channel, GCR-C has a positive paired PSNR gain at all 12 conditions, ranging from +0.378 to +1.416 dB; every image-level 95% bootstrap interval remains above zero. Since packet lengths are nearly identical, the gain reflects the reconstruction obtained from tokens that pass decoding rather than a larger transmitted payload. At 0.20 bpp, the AWGN gains are +0.946–+1.416 dB and the block-Rayleigh gains are +0.697–+1.304 dB; at 0.32 bpp, the corresponding ranges are +0.378–+0.647 and +0.394– +0.630 dB. This is a frozen-policy coded-link validation, not a channel-aware selector or an over-the-air measurement.

## G. Realized rate and encoder-side cost

The quality gain is accompanied by substantial encoderside full-budget evaluation rather than additional transmitted payload. In the held-out evaluation, GCR-C executes 13.198 and 26.876 counterfactual candidate evaluations per image at 0.20 and 0.32 bpp, respectively, compared with zero for Local-MDL and Random. These are $N _ { \mathrm { c f } } .$ , not $N _ { \mathrm { p r i o r } }$ or Local rollout actions. The intervention rates are 0.974 and 0.904, and GCR-C selects 12.01 and 29.11 tokens on average, so the gains are not caused by a larger nominal payload. The synchronized measurements give $T _ { \mathrm { p e r f } }$ means of 883 ms and 3945 ms at the two active rates.

The fixed first-50 instrumentation subset gives $N _ { \mathrm { p r o p } } =$ $N _ { \mathrm { c f } } ~ = ~ 1 8 . 9 8 / 5 8 . 6 0$ $N _ { \mathrm { r o l l } } ~ = ~ 1 5 4 . 3 4 / 1 1 7 4 . 6 0$ $N _ { \mathrm { p r i o r } } =$ $4 1 . 6 8 / 2 0 2 . 8 0$ , and $N _ { \mathrm { d e c } } = 5 . 8 8 / 1 5 . 8 4$ at 0.20/0.32 bpp; the mean per-image median-of-three synchronized times, denoted $T _ { \mathrm { t r a c e } } ,$ are 1565/10403 ms. These diagnostic values are not population estimates and are not averaged with the held-out means. The RTX 3060 measurements quantify encoder-side cost for the unbatched evaluator rather than hardware-independent latency.

![](images/f68cc6ab405fdf5b22d2c0dfa142bf935921777a23c6457e5637fbf7cedab65b.jpg)

![](images/0f1524b10b2f7bb1950ceac461f77805d9abff7ccf02d8da349c0bcbc4abdb5f.jpg)

![](images/cb242a641a61781957198682fc61a2d01515838081131a4000e95f130b361534.jpg)  
Fig. 3. Proposal-size quality–compute frontier; $K = 8$ is the final compact operating point.

TABLE I  
HELD-OUT CIFAR-10 RESULTS ON 500 IMAGES. VALUES ARE MEAN ± STANDARD DEVIATION FOR PSNR AND MEAN SSIM. RANDOM, ENTROPY-GREEDY, COVERAGE-GREEDY, AND ADAPTED MAX–MIN DIVERSITY ARE MATCHED BASELINES EVALUATED WITH THE SAME PACKET PROTOCOL; THE DIFFERENCE, CI, AND WIN RATE ARE PAIRED AGAINST LOCAL-MDL. AT 0.44 BPP, GCR-C IS IDENTICAL TO LOCAL-MDL BY POLICY; THEREFORE, THE WIN RATE IS O AND THE TIE RATE IS 1.OO.
<table><tr><td>Nominal bpp</td><td>Method</td><td>Packet bpp</td><td>Tokens</td><td>PSNR (dB)</td><td>SSIM</td><td>∆ PSNR</td><td>95% CI</td><td>Win rate</td></tr><tr><td>0.20</td><td>Random</td><td>0.1986</td><td>11.41</td><td>13.215±3.511</td><td>.382</td><td>+0.798</td><td> $[ + . 5 5 2 , + 1 . 0 4 2 ]$ </td><td>.548</td></tr><tr><td></td><td>Entropy-greedy</td><td>0.1986</td><td>12.37</td><td> $1 3 . 3 1 9 { \scriptstyle \pm 3 . 5 2 0 }$ </td><td>.392</td><td>+0.902</td><td> $[ + . 6 7 4 , + 1 . 1 3 4 ]$ </td><td>.576</td></tr><tr><td></td><td>Coverage-greedy</td><td>0.1992</td><td>11.00</td><td> $1 2 . 8 8 9 { \scriptstyle \pm 3 . 6 5 3 }$ </td><td>.365</td><td>+0.472</td><td>[+.237,+.706]</td><td>.506</td></tr><tr><td></td><td>Adapted max-min diversity [21]</td><td>0.1984</td><td>12.49</td><td> $1 1 . 9 0 3 { \scriptstyle \pm 4 . 0 0 6 }$ </td><td>.333</td><td>-0.514</td><td>[-0.775,-0.245]</td><td>.368</td></tr><tr><td></td><td>Local-MDL</td><td>0.1985</td><td>12.02</td><td> $1 2 . 4 1 7 { \scriptstyle \pm 3 . 8 6 4 }$ </td><td>.395</td><td>0</td><td></td><td></td></tr><tr><td></td><td>GCR-C</td><td>0.1987</td><td>12.01</td><td>13.920±3.637</td><td>.436</td><td>+1.503</td><td>[+1.341,+1.670]</td><td>.974</td></tr><tr><td>0.32</td><td>Random</td><td>0.3154</td><td>29.00</td><td>14.629±3.232</td><td>.479</td><td>-0.266</td><td>[-0.484,-0.046]</td><td>.398</td></tr><tr><td></td><td>Entropy-greedy</td><td>0.3163</td><td>29.41</td><td>14.919±3.484</td><td>.513</td><td>+0.024</td><td>[-0.177,+.224]</td><td>.448</td></tr><tr><td></td><td>Coverage-greedy</td><td>0.3154</td><td>29.00</td><td>14.147±3.552</td><td>.454</td><td>-0.748</td><td>[-0.979,-0.514]</td><td>.352</td></tr><tr><td></td><td>Adapted max-min diversity [21]</td><td>0.3159</td><td>29.21</td><td>13.909±4.205</td><td>.464</td><td>-0.986</td><td>[-1.215,-0.757]</td><td>.306</td></tr><tr><td></td><td>Local-MDL</td><td>0.3157</td><td>29.11</td><td>14.895±3.892</td><td>.529</td><td>0</td><td></td><td></td></tr><tr><td></td><td>GCR-C</td><td>0.3157</td><td>29.11</td><td>15.593±3.605</td><td>.550</td><td>+0.698</td><td>[+.595,+.809]</td><td>.904</td></tr><tr><td>0.44</td><td>Random</td><td>0.4375</td><td>49.00</td><td>16.873±2.761</td><td>.606</td><td>-1.716</td><td>[-1.893,-1.534]</td><td>.146</td></tr><tr><td></td><td>Entropy-greedy</td><td>0.4375</td><td>49.00</td><td>17.471±2.890</td><td>.646</td><td>-1.118</td><td>[-1.282,-0.949]</td><td>.152</td></tr><tr><td></td><td>Coverage-greedy</td><td>0.4375</td><td>49.00</td><td>16.456±3.277</td><td>.606</td><td>-2.133</td><td>[-2.368,-1.903]</td><td>.164</td></tr><tr><td></td><td>Adapted max-min diversity [21]</td><td>0.4375</td><td>49.00</td><td> $1 6 . 8 7 6 { \scriptstyle \pm 3 . 5 6 3 }$ </td><td>.623</td><td>-1.713</td><td>[-1.918,-1.511]</td><td>.134</td></tr><tr><td></td><td>Local-MDL</td><td>0.4375</td><td>49.00</td><td> $1 8 . 5 8 9 { \scriptstyle \pm 2 . 9 0 2 }$ </td><td>.680</td><td>0</td><td></td><td></td></tr><tr><td></td><td>GCR-C</td><td>0.4375</td><td>49.00</td><td> $1 8 . 5 8 9 { \scriptstyle \pm 2 . 9 0 2 }$ </td><td>.680</td><td>0</td><td>[0,0]</td><td>0</td></tr></table>

TABLE II

COMPACT FROZEN-SELECTION CODED-LINK SUMMARY ON THE HELD-OUTCIFAR-10 SPLIT. $E _ { b } / N _ { 0 }$ AND ∆PSNR ARE LISTED ASHARD/MEDIUM/EASY TRIPLES; ALL 12 PAIRED 95% CIS ARE POSITIVE.COMPLETE FER, PSNR, AND CI RECORDS ARE PROVIDED INSUPPLEMENTARY TABLE S4.
<table><tr><td>Rate</td><td>Channel</td><td> $E _ { b } / N _ { 0 } \ \mathrm { ( H / M / E ) }$ </td><td>∆PSNR (H/M/E)</td></tr><tr><td>0.20</td><td>AWGN</td><td>3.0/3.5/4.5</td><td> $+ . 9 4 6 / + 1 . 1 9 4 / + 1 . 4 1 6$ </td></tr><tr><td>0.20</td><td>Block-Rayleigh</td><td>4.0/8.0/12.5</td><td> $+ . 6 9 7 / + 1 . 0 7 1 / + 1 . 3 0 4$ </td></tr><tr><td>0.32</td><td>AWGN</td><td>2.5/3.0/3.5</td><td> $+ . 3 7 8 / + . 5 3 9 / + . 6 4 7$ </td></tr><tr><td>0.32</td><td>Block-Rayleigh</td><td>4.0/7.5/12.0</td><td> $+ . 3 9 4 / + . 4 5 0 / + . 6 3 0$ </td></tr></table>

All rows use Eq. (3), including packet, position, payload, CRC, and FEC terms. The close packet-bpp values for Local-MDL and GCR-C show that the gains are not caused by sending a larger nominal payload. At 0.20 bpp, GCR-C selects 12.01 tokens on average versus 12.02 for Local-MDL; at 0.32 bpp the corresponding values are 29.11 and 29.11.

Together, the results indicate that GCR-C is most useful when substantial full-budget headroom remains and a compact proposal can expose alternatives to the Local action. Its gain therefore depends jointly on rate, proposal coverage, and evaluator fidelity. The horizon study further shows that short-horizon scores are lower-cost approximations but do not reproduce the full-budget decision in the present system.

## H. Cross-dataset replication on STL-10

The correction mechanism persists after changing dataset, spatial resolution, token grid, and independently trained prior. STL-10 [15] uses native 96 × 96 RGB images, $8 \times 8$ patches,

![](images/693779a8a31ad222919b8e1830f0a2a8aca33055b3e7024b8a0f4c66222871f9.jpg)

![](images/6cf23e9d17006b95cb66cad3794976738c4772d25fb0c59112bea6ace424f4af.jpg)  
Fig. 4. Held-out CIFAR-10 rate–distortion and paired GCR-C–Local-MDL 95% CIs; matched baselines are in Table I.

TABLE III  
STL-10 REPLICATION ON A STRATIFIED 500-IMAGE HELD-OUT SUBSET. PACKET BPP IS RECOMPUTED FROM COMPLETE PACKET FIELDS; ∆PSNR, CI, AND WIN RATE ARE PAIRED GCR-C–LOCAL-MDL DIFFERENCES.
<table><tr><td>Point Method</td><td></td><td>Cap (bits)</td><td>Packet bpp</td><td>Tokens</td><td>PSNR/ SSIM</td><td>∆PSNR</td><td>95% CI</td><td>Win</td></tr><tr><td>Low</td><td>Local-MDL</td><td>350</td><td>.037907</td><td>24.912</td><td>11.998/.261</td><td></td><td></td><td></td></tr><tr><td>Low</td><td>GCR-C</td><td>350</td><td>.037911</td><td>24.754</td><td>12.620/.275</td><td> $+ . 6 2 1$ </td><td>[.559,.687] 99.6%</td><td></td></tr><tr><td>Mid</td><td>Local-MDL</td><td>650</td><td></td><td></td><td>.07034665.67814.533/.358</td><td></td><td></td><td></td></tr><tr><td>Mid</td><td>GCR-C</td><td>650</td><td></td><td></td><td>.07035065.63014.738/.364</td><td> $+ . 2 0 4$ </td><td>[.169,.243] 94.2%</td><td></td></tr></table>

$N ~ = ~ 1 4 4$ positions, and a newly trained $V ~ = ~ 3 2$ codebook/prior. Five thousand unlabeled images train the visual prior, 500 additional unlabeled images are reserved for prior validation, and 200 labeled training images calibrate the two bit budgets. A stratified 500-image subset of the official test set is used for held-out evaluation. The same decision design is applied with budget caps of 350 and 650 bits; eligibility remains $| S _ { t } | / N \le 0 . 3 0$ , corresponding to selected-count states 0–43 for $N = 1 4 4$ . The tokenizer and prior are independently trained, so this is cross-dataset replication of the decision mechanism rather than zero-shot checkpoint transfer.

The paired gain is +0.621 dB at the lower budget (95% CI [0.559, 0.687], win rate 99.6%) and +0.204 dB at the middle budget (CI [0.169, 0.243], win rate 94.2%). These intervals come from 10,000 image-level paired bootstrap resamples, and the held-out images are not used for checkpoint, budget, or policy selection. The result provides limited external-validity evidence that the baseline-relative correction persists under a changed resolution, token grid, and independently trained prior.

## I. High-resolution transfer on Kodak-24

The Kodak experiment serves as a high-resolution transfer diagnostic rather than a semantic-codec benchmark. With the token grid enlarged to $2 4 \times 2 4 ~ ( N = 5 7 6 )$ and the vocabulary to $V ~ = ~ 1 6 { , } 3 8 4$ , GCR-C changes the realized packet rate by less than $3 \times 1 0 ^ { - 6 }$ bpp relative to Local-MDL while improving mean PSNR by +1.033 dB at HR-Low and +0.847 dB at HR-Mid. The paired 95% confidence intervals remain above zero, with image-level win rates of 83.3% and 70.8%, respectively. These results provide protocol-scoped evidence that the baseline-relative correction mechanism transfers to a substantially larger discrete visual representation; they are not evidence that the lightweight high-resolution receiver is a competitive image codec. Separate DINOv2/LPIPS diagnostics are reported in the supplementary material and do not establish a consistent semantic or perceptual advantage; accordingly, no such advantage is claimed here.

TABLE IV  
KODAK-24 HIGH-RESOLUTION TRANSFER. VALUES ARE MEANS OVER 24 IMAGES; PACKET BPP IS REALIZED PACKET RATE AND THE ∆PSNR INTERVAL IS A PAIRED 10,000-RESAMPLE IMAGE-LEVEL BOOTSTRAP CI.
<table><tr><td>Point</td><td>Method</td><td>Tok.</td><td>Pkt bpp</td><td>PSNR/SSIM</td><td>∆PSNR [CI]</td></tr><tr><td>HR-Low</td><td>Local-MDL GCR-C</td><td>87.708 87.667</td><td>.014204</td><td>9.349/.241</td><td></td></tr><tr><td></td><td></td><td></td><td>.014201</td><td>10.382/.273</td><td>+1.033 [.641,1.456]</td></tr><tr><td>HR-Mid</td><td>Local-MDL</td><td>173.500</td><td>.025748</td><td>10.329/.238</td><td>一</td></tr><tr><td></td><td>GCR-C</td><td>173.542</td><td>.025746</td><td>11.177/.290</td><td> $+ . 8 4 7 \ [ . 3 6 9 , 1 . 4 1 7 ]$ </td></tr></table>

## VII. LIMITATIONS AND REPRODUCIBILITY

## A. Limitations

The evidence has five boundaries. First, CIFAR-10/STL-10 use compact tokenizers; the Kodak prior is lightweight and its weak low-rate reconstructions do not establish transfer to full LlamaGen/VQGAN/diffusion/multimodal tokenizers. Second, Kodak uses an initial-state gate with Local continuation, so it is a transfer diagnostic rather than the full eligiblestate schedule. Third, the unbatched full-budget evaluator is expensive and is best viewed as a quality-first operating point for delay-tolerant systems with encoder compute. Fourth, the coded link is complex AWGN/single-tap block-Rayleigh with perfect CSI and excludes mobility, burst errors, feedback/multiuser effects, and OTA impairments; matched receiver-side tokenizer/prior/decoder access and model-mismatch effects are not studied. Fifth, $Q _ { B }$ is PSNR-specific: Kodak DINOv2/LPIPS do not show consistent advantage, so pixel-domain baseline preservation is not semantic-metric preservation. Task-aware terminal utilities, stronger priors, learned/batched evaluation, OTA validation, and channel-aware $Q _ { B } ^ { \mathrm { c h } }$ selection remain future work. The adapted max–min result is a protocol-matched adaptation, not a reproduction of the original LMM pipeline.

## B. Reproducibility and Data Availability

The package contains checkpoints, split manifests, fixed policies, scripts, result records, coded-link/Kodak calibration files, figures, computation traces, and paired-bootstrap summaries. Policies were frozen on development/validation data before held-out evaluation; coded-link trials use shared seeds. CIFAR-10, STL-10, and Kodak-24 follow the public sources [14], [15], [45]; full file-level metric and protocol paths, including the complete headroom and coded-link audits, are provided in Supplementary Tables S3–S4 and the package.

## VIII. CONCLUSION

We introduced GCR-C, a communication-specific rolloutstyle correction layer that evaluates feasible alternatives with same-budget Local continuation and accepts only positive full budget advantage. On held-out CIFAR-10 it improves PSNR by 1.503 and 0.698 dB at 0.20/0.32 bpp; the 0.44-bpp policy follows Local-MDL. Under frozen coded-link selections, paired gains are positive at all 12 AWGN/block-Rayleigh conditions. Independent STL-10 and limited Kodak transfer provide protocol-scoped evidence across dataset, resolution, token grid, and tokenizer. The remaining cost is encoder-side full-budget evaluation and no OTA validation; batching, learned value approximations, stronger priors, and channel-aware selection are next steps. Semantic/perceptual improvement is not claimed because it is not the present terminal utility.

## AI USE DISCLOSURE

Artificial intelligence (AI) tools were used solely for language editing and polishing.

## REFERENCES

[1] B. L. Edwards, D. Antsos, A. Biswas, R. Reinhart, B. Robinson, D. Boroson, F. Khatri, and S. Lichten, “Addressing the high-rate deep space communications shortfall in NASA’s Space Technology Mission Directorate’s envisioned future,” in Proc. 29th Ka and Broadband Communications Conference, Seattle, WA, USA, 2024.

[2] M. Lin, T. Flatley, J. Godfrey, A. Geist, D. Espinosa, and D. Petrick, “SpaceCube 2.0: An advanced hybrid onboard data processor,” NASA Tech Briefs, Feb. 2011, NASA Technical Reports Server Document ID 20110012222.

[3] C. E. Shannon, “A mathematical theory of communication,” Bell Syst. Tech. J., vol. 27, no. 3, pp. 379–423, 1948.

[4] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” arXiv:1711.00937, 2017.

[5] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” in Proc. IEEE/CVF CVPR, 2021.

[6] H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman, “MaskGIT: Masked generative image transformer,” in Proc. IEEE/CVF CVPR, pp. 11315–11325, 2022.

[7] E. Bourtsoulatze, D. B. Kurka, and D. Gunduz, “Deep joint source-channel coding for wireless image transmission,” IEEE Trans. Cogn. Commun. Netw., vol. 5, no. 3, pp. 567–579, 2019.

[8] H. Xie, Z. Qin, G. Y. Li, and B.-H. Juang, “Deep learning enabled semantic communication systems,” IEEE Trans. Signal Process., vol. 69, pp. 2663–2675, 2021.

[9] T. Han, J. Tang, Q. Yang, Y. Duan, Z. Zhang, and Z. Shi, “Generative model based highly efficient semantic communication approach for image transmission,” arXiv:2211.10287, 2022.

[10] F. Pezone, S. Barbarossa, and G. Caire, “SQ-GAN: Semantic image communications using masked vector quantization,” IEEE Trans. Cogn. Commun. Netw., early access, 2025, doi: 10.1109/TCCN.2025.3620819.

[11] D. P. Bertsekas, “Biased aggregation, rollout, and enhanced policy improvement for reinforcement learning,” arXiv:1910.02426, 2019.

[12] L. Wang, T. Shui, W. Saad, and P. Adjakple, “World model-enabled causal digital twins for semantic communications in physical AI systems,” arXiv:2605.16547, 2026.

[13] Y. Rao, W. Zhao, B. Liu, J. Lu, J. Zhou, and C.-J. Hsieh, “DynamicViT: Efficient vision transformers with dynamic token sparsification,” in Advances in Neural Information Processing Systems, vol. 34, 2021.

[14] A. Krizhevsky, “Learning multiple layers of features from tiny images,” Univ. Toronto, Tech. Rep., 2009.

[15] A. Coates, H. Lee, and A. Y. Ng, “An analysis of single-layer networks in unsupervised feature learning,” in Proc. AISTATS, vol. 15, pp. 215–223, 2011.

[16] J. Rissanen, Stochastic Complexity in Statistical Inquiry. Singapore: World Scientific, 1989.

[17] D. Minnen, J. Balle, and G. D. Toderici, “Joint autoregressive and hierarchical priors´ for learned image compression,” in Advances in Neural Information Processing Systems, vol. 31, 2018.

[18] K. Y. Li, S. Goyal, J. D. Semedo, and J. Z. Kolter, “Inference optimal VLMs need fewer visual tokens and more parameters,” in Proc. ICLR, 2025.

[19] X. Ye, Y. Gan, X. Huang, Y. Ge, and Y. Tang, “VoCo-LLaMA: Towards vision compression with large language models,” in Proc. IEEE/CVF CVPR, pp. 29836– 29846, 2025.

[20] C. Yang, X. Dong, X. Zhu, W. Su, J. Wang, H. Tian, Z. Chen, W. Wang, L. Lu, and J. Dai, “PVC: Progressive visual token compression for unified image and video processing in large vision-language models,” in Proc. IEEE/CVF CVPR, pp. 24939–24949, 2025.

[21] S. R. Alvar, G. Singh, M. Akbari, and Y. Zhang, “DivPrune: Diversity-based visual token pruning for large multimodal models,” in Proc. IEEE/CVF CVPR, pp. 9392–9401, 2025.

[22] B. Bergner, C. Lippert, and A. Mahendran, “Token Cropr: Faster ViTs for quite a few tasks,” in Proc. IEEE/CVF CVPR, pp. 9740–9750, 2025.

[23] M. Dhouib, D. Buscaldi, S. Vanier, and A. Shabou, “PACT: Pruning and clusteringbased token reduction for faster visual language models,” in Proc. IEEE/CVF CVPR, pp. 14582–14592, 2025.

[24] J. Choi, S. Lee, B. Ko, E. Kim, J. Kil, and H. J. Kim, “Representation shift: Unifying token compression with FlashAttention,” in Proc. IEEE/CVF ICCV, pp. 20456–20466, 2025.

[25] K. Kim, J. Park, J. Kim, H. Kwon, and K. Sohn, “Faster parameter-efficient tuning with token redundancy reduction,” in Proc. IEEE/CVF CVPR, pp. 30189–30198, 2025.

[26] W. Zeng, Z. Huang, K. Ji, and Y. Yan, “Skip-Vision: Efficient and scalable acceleration of vision-language models via adaptive token skipping,” in Proc. IEEE/CVF ICCV, pp. 21384–21397, 2025.

[27] X. Ye, Y. Gan, Y. Ge, X.-P. Zhang, and Y. Tang, “ATP-LLaVA: Adaptive token pruning for large vision language models,” in Proc. IEEE/CVF CVPR, pp. 24972– 24982, 2025.

[28] M. Endo, X. Wang, and S. Yeung-Levy, “Feather the throttle: Revisiting visual token pruning for vision-language model acceleration,” in Proc. IEEE/CVF ICCV, pp. 22826–22835, 2025.

[29] Y. Liu, J. Sun, Y. Lin, J. Zhang, J. Zhang, M. Yin, Q. Wang, H. Li, and Y. Chen, “Keyframe-oriented vision token pruning: Enhancing efficiency of large visionlanguage models on long-form video processing,” in Proc. IEEE/CVF ICCV, pp. 20802–20811, 2025.

[30] H. Wang, Y. Nie, Y. Ye, Y. Wang, S. Li, H. Yu, J. Lu, and C. Huang, “Dynamic-VLM: Simple dynamic visual token compression for VideoLLM,” in Proc. IEEE/CVF ICCV, pp. 20812–20823, 2025.

[31] K. Zha, L. Yu, A. Fathi, D. A. Ross, C. Schmid, D. Katabi, and X. Gu, “Languageguided image tokenization for generation,” in Proc. IEEE/CVF CVPR, pp. 15713– 15722, 2025.

[32] W. Ye, Q. Wu, W. Lin, and Y. Zhou, “Fit and Prune: Fast and training-free visual token pruning for multi-modal large language models,” in Proc. AAAI, vol. 39, no. 21, pp. 22128–22136, 2025.

[33] J. Guo, F. Zhai, P. Jian, Q. Wei, and Y. Zhou, “CROP: Contextual region-oriented visual token pruning,” in Proc. EMNLP, pp. 9756–9772, 2025.

[34] J. Li, J. Fan, F. Tang, G. Huang, S. Zhu, S. Liu, N. Xie, W. Liu, and Y. Liao, “FCoT-VL: Advancing text-oriented large vision-language models with efficient visual token compression,” arXiv:2502.18512, 2025.

[35] B. Liu, L. Qiao, Y. Wang, Z. Gao, Y. Ma, K. Ying, and T. Qin, “Text-guided token communication for wireless image transmission,” arXiv:2507.05781, 2025.

[36] M. Devoto, J. Pomponi, M. Merluzzi, P. D. Lorenzo, and S. Scardapane, “Adaptive semantic token communication for Transformer-based edge inference,” arXiv:2505.17604, 2025.

[37] B. Li, X. Yang, S. Duan, and N. Wang, “Toward universal semantic communication via matchable semantic subspace transmission,” IEEE Trans. Image Process., vol. 35, pp. 5003–5016, 2026, doi: 10.1109/TIP.2026.3690331.

[38] O. F. Deniz, R. Mao, R. Li, Y. Tian, and L. Khan, “Vision token reduction via attention-driven self-compression for efficient multimodal large language models,” arXiv:2602.12618, 2026.

[39] B. Wan, Y. Feng, Z. Tang, W. Huang, Y. Zeng, J. Wang, and T. Liu, “RTPrune: Reading-twice inspired token pruning for efficient DeepSeek-OCR inference,” in Proc. ICML, vol. 306, 2026.

[40] H. Chen and J. He, “Energy-driven adaptive visual token pruning for efficient vision-language models,” arXiv:2603.05950, 2026.

[41] S. Gu, J. Cui, W. Hu, Z. Shi, Z. Hu, and R. Hong, “Visual token compression enhances robustness of MLLMs,” arXiv:2607.22716, 2026.

[42] M. Naseri, P. Ashtari, M. Seif, E. De Poorter, H. V. Poor, and A. Shahid, “Deep learning-based image compression for wireless communications: Impacts on robustness, throughput, and latency,” npj Wireless Technology, vol. 2, art. no. 14, 2026, doi: 10.1038/s44459-025-00019-6.

[43] J. Hoydis, S. Cammerer, F. Ait Aoudia, A. Vem, N. Binder, G. Marcus, and A. Keller, “Sionna: An open-source library for next-generation physical layer research,” arXiv:2203.11854, 2022. [Online]. Available: https://nvlabs.github.io/sionna/

[44] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan, “Autoregressive model beats diffusion: Llama for scalable image generation,” arXiv:2406.06525, 2024. [Online]. Available: https://github.com/FoundationVision/LlamaGen

[45] Eastman Kodak Company, “Kodak lossless true color image suite,” [Online]. Available: https://r0k.us/graphics/kodak/. Accessed: Aug. 16, 2026.
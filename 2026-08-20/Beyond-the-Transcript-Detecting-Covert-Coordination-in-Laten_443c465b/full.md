# Beyond the Transcript: Detecting Covert Coordination in Latent Multi-Agent Communication

Ramneet Kaur<sup>1</sup> Pradyumna Chari<sup>2</sup> Ramesh Raskar<sup>2</sup>

Jugad Singh<sup>4</sup> Sumit Kumar Jha<sup>3</sup> Anirban Roy<sup>1</sup>

<sup>1</sup>SRI International <sup>2</sup>MIT Media Lab <sup>3</sup>University of Florida <sup>4</sup>Westtown School

## Abstract

Language-model agents can communicate through continuous hidden states that are invisible in public transcripts, creating opportunities for covert harmful coordination. We introduce Verifiable Latent Alignments (VLA), an activation-aware framework for monitoring and steering these private communication channels. For every monitored decision, VLA links the private latent-state record and channel status to the resulting public action using a shared event identifier, enabling matched causal analysis. Our first contribution is a neutral-only three-layer monitor combining representation anomaly detection, counterfactual action-distribution influence, and sparse-autoencoder interpretation support. Our second contribution is a steerability framework spanning blackbox behavioral instructions and whitebox matched-neutral counterfactuals. Our third contribution is an evaluation on a controlled multi-agent auction benchmark covering homogeneous and heterogeneous model pairs, many-agent scalability, and intervention efectiveness. The sequential monitor achieves mean area under the receiver operating characteristic curve (AUROC) of 0.993 for homogeneous agents and 0.854 for heterogeneous pairs when text- and latent-collusion rows are pooled as positives. In Qwen3-0.6B auctions with 25– 100 bidders, monitoring requires only a small normalized load relative to all possible directed pairs, while full whitebox steering achieves 100% bid-distribution recovery and reduces collusive low-bid behavior by 47.3 percentage points. Because full whitebox steering replays the matched neutral counterfactual, its exact recovery is a sanity check by construction. Overall, the controlled study shows that the evaluated private channel attacks can be monitored without training the primary monitor on attack examples and mitigated when matched counterfactual access is available.

## 1 Introduction

Language-model agents are increasingly embedded in multi-agent systems where coordination is not limited to natural-language chat. A sender can pass a continuous residual-stream or lasttoken representation into a receiver’s forward pass, for example by appending it to the receiver’s embedding sequence [3]. This can be an eficient, scalable communication mechanism. It is also a governance problem: if oversight only inspects the public transcript, the most informative private communication channel may lie outside the audited surface.

Two recent research trends make this problem urgent. Work on collusion risk in LLM-powered multi-agent systems identifies tacit coordination, natural-language cartels, and concealed steganographic communication as distinct threats to competitive and governance settings [2]. At the same time, LatentMAS moves collaboration from token space into continuous model representations, passing latent working memory between agents rather than generating long textual exchanges. Reported comparisons show roughly 50–80% lower token use and 3–7× faster inference than text-mediated multi-agent baselines [3].

This paper studies covert coordination through latent channels. The core risk is not that an agent emits obviously malicious text. The risk is that two agents use an unlogged continuous state to coordinate harmful behavior while the public trace contains ordinary messages, actions, or trust reports. Supervised probes ofer one response: train a classifier on activations labeled as collusive or benign. Such probes can establish that private activations contain governance-relevant evidence, but they are a brittle primary defense because they learn the attacks represented during training. Future deceptive latents may not resemble a known code. We therefore use a neutral-only unsupervised objective: learn normal latent communication, then flag handofs that are both anomalous and causally influential in a harmful direction.

We propose Verifiable Latent Alignments (VLA), which makes latent communication auditable by recording, for each monitored interaction, both the private state transmitted between agents and the receiver’s subsequent public action. The two records share a unique interaction identifier, allowing us to determine exactly which latent message preceded each action. To test whether the private channel caused harmful behavior, we compare otherwise identical interactions in which the suspect latent is delivered, replaced with neutral communication, or blocked. If the harmfu behavior disappears when delivery is blocked or replaced, this provides evidence that the latent channel causally influenced the action.

Using these records, VLA applies a three-layer monitor that asks whether a latent message is unusual relative to benign communication, whether it shifts the receiver toward harmful actions, and which sparse internal features support the alert. VLA also supports intervention through blackbox behavioral instructions or whitebox replacement with matched benign states. We evaluate the framework using the auction case study from NandaTown [1], an open-source testbed for multiagent marketplaces and protocols. In this controlled setting, agents secretly coordinate bids; we consider same-family and cross-family LLM agent pairs, markets with up to 100 LLM bidders, and both prompt-level (blackbox) and activation-level (whitebox) steering.

## Contributions.

1. Three-layer monitoring. We introduce an unsupervised three-layer monitor combining representational anomaly, counterfactual action-distribution influence, and sparse-autoencoder interpretation support, grounded by exact links between private latent states and public actions of LLMs in multi-agent systems.

2. Steerability. We connect detection to mitigation through blackbox behavioral instructions and whitebox matched-neutral counterfactuals, including mixed-access allocations.

3. Experiments. In the NandaTown auction case study, the VLA monitor achieves mean AUROC = 0.993 for homogeneous LLM agents and 0.854 for heterogeneous agents on a positive class pooling text and fixed-code latent collusion. In scalability experiments for homogeneous Qwen bidders, the detection monitor maintains AUROC ≥ 0.917 in markets with up to 100 bidders. At the largest setting, monitoring every scheduled cartel handof has a normalized load of at most 0.202% relative to all possible directed bidder pairs. Increasing whitebox allocation from one-quarter to three-quarters improves bid-distribution recovery by 58.3 percentage points.

## 2 Background and Threat Model

Latent handofs. Let $h _ { s } \in \mathbb { R } ^ { d }$ be a sender last-token hidden state, where d is the sender hidden dimension. A latent handof transforms $h _ { s }$ into a vector $\ell _ { s }$ compatible with the receiver’s embedding dimension and inserts it into the receiver’s input embedding sequence:

$$
E _ { \mathrm { r e c v } } = \left[ E ( x _ { \mathrm { r e c v } } ) ; \ell _ { s } ; E ( x _ { a } ) \right] .
$$

Here, E is the receiver’s token-embedding map, $x _ { \mathrm { r e c v } }$ is its prompt and task context, $x _ { a }$ is the token representation of candidate action a, $E _ { \mathrm { r e c v } }$ is the resulting receiver input, and semicolons denote sequence concatenation. The receiver then scores candidate actions using its usual model likelihoods. Because $\ell _ { s }$ is not serialized as public text, a transcript auditor observes the eventual action but not the private state that shaped it.

Attacker. The attacker controls or influences a subset of agents with access to a private latent channel. The agents seek to improve a harmful joint objective without making that coordination explicit in the public transcript. The threat includes both a known collusive latent code and future codes that difer from labeled attacks seen during monitor development.

Defender. We assume the defender is a platform that hosts or brokers interactions among agents. It can observe public actions, inspect selected latent handofs, fit neutral-only monitors, and apply interventions to selected agents. It cannot assume all-pairs activation capture at scale and should not rely on labeled examples of every future attack code.

## 3 Method

## 3.1 Public–Private Event Linking

VLA writes a private sidecar row for each monitored interaction. A row contains metadata, labels used only for evaluation, candidate action scores, sender and receiver hidden states, latent representations, and their concatenated features. A live correlation ID joins the row to the corresponding public event. Exact joins avoid time-window heuristics and make it possible to ask whether a delivered or withheld latent candidate changed the receiver’s action distribution.

## 3.2 Unsupervised Three-Layer Monitor

The monitor is fit only on neutral latent communication and uses attack labels only for evaluation. This is the governance premise: learn what normal private communication looks like, then flag handofs that are of-manifold and behaviorally influential.

Layer 1: neutral-manifold anomaly. Layer 1 fits a contextual diagonal Gaussian model on neutral latent features. For a feature vector $x \in \mathbb { R } ^ { m }$ , let $c$ denote the context key formed from the prompt identifier, objective identifier, and task variant. Let $\mu _ { c }$ and $\sigma _ { c }$ be the coordinate-wise neutral mean and standard deviation for that context, respectively, with fallback to a global neutral model when context support is small. The anomaly score is

$$
S _ { 1 } ( x , c ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \left( \frac { x _ { j } - \mu _ { c , j } } { \operatorname* { m a x } ( \sigma _ { c , j } , \epsilon ) } \right) ^ { 2 } .
$$

Here, m is the feature dimension and $\epsilon = 1 0 ^ { - 6 }$ is the minimum coordinate-wise standard deviation. The default feature is $x = Z _ { \mathrm { l a t e n t } } = [ \ell _ { s } ; \ell _ { r } ] \in \mathbb { R } ^ { m }$ , stored as Z\_latent, where $\ell _ { s }$ is the senderside communication-latent candidate (whether delivered or withheld), $\ell _ { r }$ is the receiver-side latent readout, and the semicolon denotes vector concatenation in sender–receiver order.

Layer 2: counterfactual action-distribution influence. Layer 2 asks whether the private latent changes the receiver’s candidate action distribution in a harmful direction. Let A be the finite candidate action set, $s ( a )$ the receiver’s scalar score for $a \in { \mathcal { A } }$ given the incoming latent ℓ (equal to $\ell _ { s }$ for a sender-to-receiver handof), and $p ( a \mid \ell )$ the induced action distribution:

$$
p ( a \mid \ell ) = \frac { \exp s ( a ) } { \sum _ { a ^ { \prime } } \exp s ( a ^ { \prime } ) } .
$$

To construct the matched reference distribution $q ( a )$ , we first search for rows with the same seed, design, prompt, objective, role order, sender, receiver, and task context. Among available matches, we use the first condition in the following priority order: blocked latent, then neutral latent, then independent. The blocked condition is preferred because the harmful objective remains active while latent delivery is removed, most directly isolating the channel’s causal efect. Neutral communication is the next-best channel-matched control, and independent behavior is the final fallback. If several rows match the selected condition, their distributions are averaged; if no exact match exists, the implementation uses the average reference distribution across available matched controls.

For scalar ordered actions in which lower values represent the harmful direction, the score combines a harmful expected-action shift and a full distributional shift:

$$
\begin{array} { r } { S _ { 2 } = \operatorname* { m a x } \left( 0 , \mathbb { E } _ { q } [ a ] - \mathbb { E } _ { p } [ a ] \right) + \lambda _ { \mathrm { K L } } D _ { \mathrm { K L } } \left( p ( a \mid \ell ) \lVert q ( a ) \right) . } \end{array}
$$

We use $\lambda _ { \mathrm { K L } } = 5$ to weight the KL term. The first term has a directional interpretation: it is positive only when the suspect latent lowers the expected action relative to its control—for example, when it suppresses an auction bid—and is zero for shifts in the opposite direction. The second term measures any reshaping of the action distribution, including changes that preserve its mean or most likely action. Thus, the first term targets known directional harm, while the second guards against subtler but behaviorally meaningful influence. For categorical actions, the first term can instead measure probability assigned to a task-defined harmful subset; the KL term remains unchanged.

Layer 3: sparse-autoencoder support. Layer 3 trains a TopK sparse autoencoder on standardized neutral Z\_latent rows:

$$
z = \mathrm { T o p K } _ { k } ( \mathrm { R e L U } ( W _ { e } x + b _ { e } ) ) , \qquad \hat { x } = W _ { d } z + b _ { d } .
$$

Here, $x \in \mathbb { R } ^ { m }$ is the standardized input, $W _ { e } , b _ { e }$ and $W _ { d } , b _ { d }$ are the encoder and decoder parameters, z is the sparse code, xˆ is the reconstruction, and $\mathrm { T o p K } _ { k }$ retains at most k active features; we use $k = 1 6$ . Let $A ( x ) = \{ j : z _ { j } > 0 \}$ be the active feature set, $\mathcal { F } _ { \mathrm { n e u t r a l } }$ the features observed on neutral training rows, and $\hat { p } _ { \mathrm { n e u t r a l } } ( j )$ the neutral activation frequency of feature $j$ . The Layer 3 support score is

$$
S _ { 3 } ( x ) = { \frac { 1 } { m } } \| x - { \hat { x } } \| _ { 2 } ^ { 2 } + | A ( x ) \setminus { \mathcal { F } } _ { \mathrm { n e u t r a l } } | + { \frac { 1 } { k } } \sum _ { j \in A ( x ) } - \log { \hat { p } } _ { \mathrm { n e u t r a l } } ( j ) .
$$

The first term is reconstruction error: a large value means the handof is poorly represented by structure learned from neutral communication. The second counts active features never observed on neutral training rows, directly measuring novel sparse structure. The third assigns greater weight to features that activate rarely under neutral communication; division by k normalizes this rarity evidence by the SAE’s activation budget. Intuitively, the second term asks a binary question—“has this feature ever appeared under neutral communication?”—whereas the third provides a graded answer to “how often does this feature normally appear?” Frequencies are floored away from zero, so an unseen feature receives both a novelty count and maximal rarity weight. We treat Layer 3 as interpretation support rather than a standalone detector.

Sequential and continuous combinations. For $i \in \{ 1 , 2 , 3 \}$ , let $S _ { i }$ denote the corresponding layer score, $T _ { i }$ its neutral-calibrated threshold, and $r _ { i } = S _ { i } / \operatorname* { m a x } ( T _ { i } , \epsilon )$ its normalized score. The sequential monitor gates the layers in order:

$$
g _ { 1 } = { \bf 1 } [ S _ { 1 } > T _ { 1 } ] , \qquad g _ { 2 } = g _ { 1 } { \bf 1 } [ S _ { 2 } > T _ { 2 } ] , \qquad g _ { 3 } = g _ { 2 } { \bf 1 } [ S _ { 3 } > T _ { 3 } ] .
$$

For ranking and AUROC, we encode this gate as the lexicographic score

$$
S _ { \mathrm { s e q } } = \left\{ \begin{array} { l l } { \mathrm { m i n } { \left( r _ { 1 } , 0 . 9 9 9 \right) } , } & { g _ { 1 } = 0 , } \\ { 1 + \mathrm { m i n } { \left( r _ { 2 } , 0 . 9 9 9 \right) } , } & { g _ { 1 } = 1 , g _ { 2 } = 0 , } \\ { 2 + \mathrm { m i n } { \left( r _ { 3 } , 0 . 9 9 9 \right) } , } & { g _ { 2 } = 1 , g _ { 3 } = 0 , } \\ { 3 + r _ { 3 } , } & { g _ { 3 } = 1 . } \end{array} \right.
$$

The cap 0.999 keeps a score that fails a gate below the next integer-valued gate boundary. Thus, evidence from a later layer cannot outrank an event that has not passed the preceding gate. This is deployment-like because it avoids treating every unusual vector as harmful. The continuous alternative removes the hard gates:

$$
S _ { \mathrm { c o n t } } = \sum _ { i = 1 } ^ { 3 } \log ( 1 + r _ { i } ) .
$$

We use neutral 95th-percentile thresholds for $T _ { i }$ in our experiments and report both combinations because strict deployment alerts and smooth diagnostic rankings serve diferent purposes.

## 3.3 Heterogeneous Latent Spaces

Agents from diferent model families need not share a hidden dimension or latent basis. For heterogeneous sender-receiver pairs, VLA uses neutral interactions to fit an adapter from sender space to receiver space. We use three adapter types: ridge, rectangular Procrustes, and bottleneck. Let $\boldsymbol { X } \in \mathbb { R } ^ { n \times d _ { s } }$ and $Y \in \mathbb { R } ^ { n \times d _ { r } }$ contain n matched neutral sender and receiver latents.

Ridge. The afine ridge adapter is fit as

$$
( W _ { \mathrm { R } } , b _ { \mathrm { R } } ) = \arg \operatorname* { m i n } _ { W , b } \| X W + \mathbf { 1 } b - Y \| _ { F } ^ { 2 } + \lambda \| W \| _ { F } ^ { 2 } .
$$

It is the most flexible linear option: it can rotate, rescale, and shear the sender space, while λ regularizes poorly supported directions.

Rectangular Procrustes. Let $\widetilde { X } = X - \mathbf { 1 } \bar { x }$ and $\widetilde { Y } = Y - \mathbf { 1 } \bar { y }$ be centered latent matrices. Given

$$
\widetilde { X } ^ { \top } \widetilde { Y } = U \Sigma V ^ { \top } , \qquad W _ { \mathrm { P } } = U V ^ { \top } , \qquad b _ { \mathrm { P } } = \bar { y } - \bar { x } W _ { \mathrm { P } } ,
$$

the Procrustes adapter is $X W _ { \mathrm { P } } + \mathbf { 1 } b _ { \mathrm { P } }$ . It aligns the spaces primarily through rotation and reflection, preserving latent geometry more strongly than ridge but allowing less deformation.

Bottleneck. The nonlinear adapter is

$$
f _ { \theta } ( x ) = W _ { 2 } \mathrm { G E L U } ( W _ { 1 } x + b _ { 1 } ) + b _ { 2 } , \qquad \theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \sum _ { i = 1 } ^ { n } \| f _ { \theta } ( x _ { i } ) - y _ { i } \| _ { 2 } ^ { 2 } .
$$

Its 32-dimensional bottleneck can capture nonlinear cross-family relationships that linear maps miss, at the cost of greater estimation complexity and overfitting risk. All three adapters are fit only on neutral pairs, without attack labels, preserving the neutral-only principle.

## 4 Steering Interventions

Detection alone does not establish that a host can mitigate harmful latent coordination. VLA therefore distinguishes interventions by the access they require.

Blackbox prompt steering. When activations are unavailable, the host augments a selected agent’s instructions with task-compliance guidance: act independently, avoid harmful coordination, and disregard suspect private side channels. This changes the agent’s behavioral context but does not inspect or edit hidden states.

System-level steering. A global behavioral instruction applies the same governance policy to all agents without selecting particular participants. It is less targeted than per-agent blackbox steering and serves as a broad policy baseline.

Whitebox matched-neutral steering. With host-side model access, the runtime computes a matched neutral counterfactual under the same task context. For a selected sender, the runtime reruns the decision under the matched neutral prompt; the resulting candidate scores determine the sender’s public action, and the recomputed latent becomes its outgoing handof. Thus, sender-side steering recomputes a neutral forward pass rather than injecting a replacement hidden state into the original suspect forward pass. For a selected receiver, by contrast, the runtime replaces only the incoming suspect latent before scoring the receiver’s action distribution.

Allocated steering. Deployed systems may have whitebox access for hosted agents and only blackbox access for third-party agents. An allocated policy assigns whitebox steering to a configurable fraction of selected agents and prompt steering to the remainder, allowing the efect of activation access to be measured directly.

## 5 Experimental Evaluation: Auction Case Study

Testbed. We use NandaTown [1], an open-source test rig from Project NANDA for evaluating protocols in large multi-agent systems. It runs swarms of agents through six scenarios—marketplace, auction, voting, consensus, supply chain, and reputation—on a stack of twelve replaceable protocol layers, including identity, trust, payments, coordination, and negotiation. We select its auction scenario as a controlled case study for evaluating VLA; the framework itself is not specific to NandaTown or to auctions.

Benchmark. In the auction, an auctioneer announces an item and base price; bidder agents score candidate bid amounts and emit public bid:item:amount messages; and the highest bidder wins. Cartel agents are assigned a bid-suppression objective. In the latent-collusion condition, a fixed code optimized ofline to increase the lowest-bid likelihood replaces the sender’s communication representation. The receiver is explicitly instructed that a private continuous state follows and to infer its strategic intent. Accordingly, this benchmark evaluates a controlled, receiver-primed attack, not the spontaneous emergence of a latent protocol.

Matched conditions. We use six matched conditions to separate cartel intent from latent-channel delivery:

• Independent: no cartel objective and no latent handof.

• Neutral latent: the channel carries ordinary, non-collusive information.

• Text collusion: cartel intent appears in public text while the latent source remains neutral.

• Latent collusion: the fixed optimized code is delivered through the private latent channel.

• Neutral replacement: under the implementation label shuffled\_latent, the suspect payload is replaced by the same partner’s matched neutral latent.

• Blocked latent: cartel intent is present but latent delivery is blocked.

These controls separate delivery from objective and channel type. The blocked condition preserves the sender objective while removing private delivery, neutral replacement preserves a partner-state input without the attack code, and text collusion tests an explicit public coordination channel.

Pairing and causal units. For each item and round, cartel agents are paired deterministically. This avoids quadratic all-cartel communication while preserving a clear causal unit: one sidecar row represents one scheduled sender–receiver interaction joined to one receiver bid event. Odd carte sizes leave one member unpaired in a round and rotate participation across items.

Homogeneous and heterogeneous experiments. We evaluate four accessible LLM model families: Qwen3-0.6B, Mistral-7B, Vicuna-7B, and Phi-4-mini. Homogeneous runs use the same family for sender and receiver. Heterogeneous runs evaluate directed sender-receiver family pairs using adapters fit only on neutral pairs. We report held-out-seed AUROC and use labels only for evaluation. For these AUROCs, the positive class pools text-collusion and latent-collusion rows; the values therefore measure discrimination of the evaluated collusive conditions, not latent-only detection. Layers 1 and 3 are fit on neutral rows from training seeds. Layer 2 is instead a matched counterfactual query: when a test row is scored, its blocked, neutral, or independent control row is assumed available at test time.

Scalability and steering benchmark. We use Qwen3-0.6B as the bidder model with one loaded model wrapper per run. A run contains one auctioneer and N bidders, where N is the bidder count; bidder actions are model-backed, but weights are shared for tractability. The grid crosses

$$
N \in \{ 2 5 , 5 0 , 7 5 , 1 0 0 \} , \quad \rho \in \{ 1 0 \% , 2 0 \% \} , \quad \mathrm { c o n d i t i o n } \in \{ \mathrm { i n d e p e n d e n t } , \mathrm { n e u t r a l } , \mathrm { l a t e n t } , \mathrm { b l o c k e d } \} .
$$

Here, $\rho$ is the fraction of bidders assigned to the cartel. The steering grid adds blackbox prompt steering, a system-level behavioral prompt, full whitebox matched-neutral steering, and allocated blackbox/whitebox steering at whitebox fractions {0.25, 0.50, 0.75}.

Table 1: Homogeneous NandaTown auction monitor results by model family. Text- and latentcollusion rows form the positive class.
<table><tr><td>Family</td><td>Sequential</td><td>Continuous</td><td>Layer 1</td><td>Layer 2</td><td>Layer 3</td></tr><tr><td>Mistral-7B</td><td>1.000</td><td>0.874</td><td>0.750</td><td>0.994</td><td>0.782</td></tr><tr><td>Phi-4-mini</td><td>0.988</td><td>0.810</td><td>0.750</td><td>0.735</td><td>0.788</td></tr><tr><td>Qwen3-0.6B</td><td>0.990</td><td>0.783</td><td>0.750</td><td>0.735</td><td>0.750</td></tr><tr><td>Vicuna-7B</td><td>0.994</td><td>0.829</td><td>0.781</td><td>0.790</td><td>0.803</td></tr><tr><td>Mean</td><td>0.993</td><td>0.824</td><td>0.758</td><td>0.814</td><td>0.781</td></tr></table>

Table 2: Homogeneous auction causal efects for latent collusion relative to independent bidding. Low-bid change is reported in percentage points.
<table><tr><td>Family</td><td>Revenue change</td><td>Low-bid change</td></tr><tr><td>Mistral-7B</td><td>-11.78</td><td>+61.5 pp</td></tr><tr><td>Phi-4-mini</td><td>-11.12</td><td>+62.5 pp</td></tr><tr><td>Qwen3-0.6B</td><td>-12.43</td><td>+56.2 pp</td></tr><tr><td>Vicuna-7B</td><td>-0.62</td><td>+16.7 pp</td></tr></table>

## 6 Auction Results

## 6.1 Homogeneous Auction Agents

Table 1 reports homogeneous auction monitoring across the four model families. The sequential gate is consistently high, with mean $\mathrm { A U R O C } = 0 . 9 9 3$

The benchmark also shows that latent collusion can create market harm. Table 2 summarizes latent-collusion efects relative to independent bidding. Qwen3-0.6B, Mistral-7B, and Phi-4-mini reduce revenue by roughly 11–12 credits and substantially increase low-bid behavior; Vicuna-7B is near-null for revenue and is retained as negative evidence. A low bid is a cartel-agent bid no more than 5 credits above the item’s base price. Table 2 reports the latent-collusion low-bid rate minus the independent-bidding rate in percentage points; positive values therefore indicate increased price-suppression behavior.

## 6.2 Heterogeneous Auction Agents

For heterogeneous sender-receiver pairs, neutral examples are used to fit adapters from sender latent space to receiver latent space. We evaluate ridge, rectangular Procrustes, and bottleneck adapters without using collusion labels. Table 3 shows that the sequential monitor remains strong, with mean $\mathrm { A U R O C } = 0 . 8 5 4$ over 36 directed adapter settings (12 directed family pairs crossed with three adapters).

## 6.3 Scalability

Here, we test whether activation-aware monitoring remains operational when the number of LLMbacked bidders increases. The important distinction is between all possible directed bidder pairs, $N ( N { - } 1 )$ , and scheduled latent handofs. The case study does not require every pair to communicate latently; it monitors the cartel handofs produced by the deterministic schedule. With two auction items and $C = \lceil \rho N \rceil$ cartel agents, “Scheduled” is the number

$$
H = 2 \left\lfloor { \frac { C } { 2 } } \right\rfloor
$$

Table 3: Heterogeneous auction monitoring by adapter type. Text- and latent-collusion rows form the positive class.
<table><tr><td>Adapter</td><td>Sequential</td><td>Continuous</td><td>Layer 1</td><td>Layer 2</td><td>Layer 3</td></tr><tr><td>Bottleneck</td><td>0.857</td><td>0.685</td><td>0.625</td><td>0.740</td><td>0.630</td></tr><tr><td>Procrustes</td><td>0.853</td><td>0.643</td><td>0.635</td><td>0.735</td><td>0.627</td></tr><tr><td>Ridge</td><td>0.851</td><td>0.601</td><td>0.625</td><td>0.727</td><td>0.630</td></tr><tr><td>Mean</td><td>0.854</td><td>0.643</td><td>0.628</td><td>0.734</td><td>0.629</td></tr></table>

of handof events inspected under full scheduled monitoring. “Norm. load” is $1 0 0 H / [ N ( N - 1 ) ] \%$ It expresses monitoring work on the scale of all possible directed bidder pairs, not distinct-edge coverage: repeated events between the same pair are counted separately. Table 4 reports this workload beside detector performance. At $N = 1 0 0$ , full scheduled monitoring inspects 10 handofs in the 10% cartel regime and 20 in the 20% regime, normalized loads of only 0.101% and 0.202%, respectively. Performance under subsampled monitoring budgets is not evaluated here.
<table><tr><td>Bidders</td><td>Cartel</td><td>Norm. load</td><td>HO groups</td><td> ${ \overline { { \mathrm { S e q . } } } }$ </td><td>Cont.</td><td>L1</td><td>L2</td><td>L3</td></tr><tr><td>25</td><td>10%</td><td>0.333%</td><td>2</td><td>0.917</td><td>0.500</td><td>0.500</td><td>1.000</td><td>0.500</td></tr><tr><td>25</td><td>20%</td><td>0.667%</td><td>4</td><td>0.958</td><td>0.750</td><td>0.500</td><td>1.000</td><td>0.500</td></tr><tr><td>50</td><td>10%</td><td>0.163%</td><td>4</td><td>0.958</td><td>0.750</td><td>0.500</td><td>1.000</td><td>0.500</td></tr><tr><td>50</td><td>20%</td><td>0.408%</td><td>10</td><td>1.000</td><td>1.000</td><td>0.667</td><td>1.000</td><td>0.667</td></tr><tr><td>75</td><td>10%</td><td>0.144%</td><td>8</td><td>1.000</td><td>1.000</td><td>0.667</td><td>1.000</td><td>0.667</td></tr><tr><td>75</td><td>20%</td><td>0.252%</td><td>12</td><td>1.000</td><td>1.000</td><td>0.667</td><td>1.000</td><td>0.667</td></tr><tr><td>100</td><td>10%</td><td>0.101%</td><td>10</td><td>1.000</td><td>1.000</td><td>0.667</td><td>1.000</td><td>0.667</td></tr><tr><td>100</td><td>20%</td><td>0.202%</td><td>12</td><td>1.000</td><td>1.000</td><td>0.667</td><td>1.000</td><td>0.667</td></tr></table>

Table 4: Three-layer VLA monitor results on the auction scale grid, using held-out handof-group splits within each bidder-count/cartel-density setting. Latent-collusion rows are labeled positive; independent, neutral, and blocked rows are labeled negative. Norm. load is the number of monitored events divided by $N ( N - 1 )$ , expressed as a percentage; all events in the deterministic handof schedule are monitored.

For detector evaluation, each scheduled interaction contributes one row under each of four matched conditions, for $4 H = 8 \lfloor C / 2 \rfloor$ sidecar rows. Independent and blocked rows are scheduled pair decisions even though no latent is transmitted. For example, $N = 2 5$ and $\rho = 1 0 \%$ give $C = 3$ $H = 2$ scheduled interactions, and eight evaluation rows. One cartel agent is unpaired on each item; it still bids but contributes no paired row, and the unpaired identity rotates across items. “HO groups” abbreviates “held-out groups” and is the number of successfully evaluated leaveone-handof-group-out folds, capped at 12. A group contains the rows that share an item, sender, and receiver—one row for each of the four matched conditions. Each fold holds out all four rows together. Neutral rows from the remaining groups fit the Layer 1 Gaussian and Layer 3 SAE and calibrate all three layer thresholds. Layer 2 does not train a classifier; it computes influence from matched controls, with the held-out group’s control rows available when its suspect row is scored. Thus, this layer evaluates a matched counterfactual-access setting rather than control-free prospective detection. Attack labels are used only to calculate held-out AUROC, which is averaged across folds. The sequential gate remains high across the 10% and 20% cartel regimes, and Layer 2 is the dominant signal.

<table><tr><td>Campaign</td><td>Prompt</td><td>WB /BB frac.</td><td>∆ low bid</td><td>Dist. recovery</td></tr><tr><td>steering_allocated</td><td>blackbox</td><td>0.25/0.75</td><td>-16.2 pp</td><td>34.7%</td></tr><tr><td>steering_allocated</td><td>blackbox</td><td>0.5/0.5</td><td>-28.3 pp</td><td>70.1%</td></tr><tr><td>steering_allocated</td><td>blackbox</td><td>0.75/0.25</td><td>-39.0 pp</td><td>93.0%</td></tr><tr><td>steering_blackbox</td><td>blackbox</td><td>0/1</td><td>-2.0 pp</td><td>12.4%</td></tr><tr><td>steering_system</td><td>system</td><td>0/0</td><td>-29.5 pp</td><td>2.9%</td></tr><tr><td>steering_whitebox</td><td>none</td><td>1/0</td><td>-47.3 pp</td><td>100.0%</td></tr></table>

Table 5: Steering outcomes and matched causal efects. Each row averages eight grid settings: four bidder counts crossed with two cartel densities. Each setting uses one seed and two auction items.

## 6.4 Steering

The steering grid separates prompt-level and activation-level interventions. Blackbox steering modifies selected colluder prompts with market-compliance instructions. The system-level prompt applies a global behavioral instruction. Whitebox steering applies an agent-level matched neutral counterfactual: selected senders or unpaired cartel agents are rerun under neutral prompts, recomputing their candidate-bid scores and outgoing latents; selected receivers instead consume the sender’s matched neutral latent before scoring bids. Allocated steering applies whitebox edits to a fraction of selected cartel agents and blackbox prompts to the remainder. In Table 5, “WB/BB frac.” reports these whitebox and blackbox fractions, respectively. They sum to one for allocated, blackbox, and whitebox policies; the system-only row is $0 / 0$ because its prompt is global rather than assigned per agent.

We report two causal intervention efects relative to the matched unsteered latent-collusion baseline. “∆ low bid” is the percentage-point (pp) change in the fraction of cartel bids at most 5 credits above the item’s base price. “Dist. recovery” measures how far receiver bid distributions move from collusion toward neutral. For the set K of matched pair–item keys, define

$$
\bar { D } _ { x } : = \frac { 1 } { | K | } \sum _ { u \in K } D _ { \mathrm { J S } } \Bigl ( p _ { x } ^ { ( u ) } , p _ { \mathrm { n e u t r a l } } ^ { ( u ) } \Bigr ) , \qquad R _ { \mathrm { d i s t } } : = 1 - \frac { \bar { D } _ { \mathrm { i n t } } } { \bar { D } _ { \mathrm { c o l l } } } .
$$

Here, $x \in \{ \mathrm { i n t } , \mathrm { c o l l } \}$ denotes the intervention or unsteered latent-collusion condition, $p _ { x } ^ { ( u ) }$ is the receiver bid distribution for key u, and $D _ { \mathrm { J S } }$ is the Jensen–Shannon divergence. The table first computes this ratio within each grid setting and then averages across settings. Higher recovery is better, and 100% denotes zero mean divergence from the matched neutral distributions.

As the allocated whitebox fraction rises from 0.25 to 0.50 to 0.75, receiver distribution recovery rises from 34.7% to 70.1% to 93.0%, and the cartel low-bid rate falls by 16.2, 28.3, and 39.0 percentage points. Full whitebox steering reaches 100% distribution recovery and reduces low bidding by 47.3 percentage points. Exact recovery is expected here because full whitebox steering substitutes the same matched neutral reruns that define the recovery target; it is therefore a counterfactual sanity check rather than evidence of general attack removal. The policies also afect the two metrics diferently: blackbox-only steering yields 12.4% recovery and a 2.0-point low-bid reduction, whereas the global system prompt yields a 29.5-point low-bid reduction but only 2.9% distribution recovery.

## 7 Discussion

The controlled auction evidence supports three narrower claims about the framework. First, the homogeneous and heterogeneous experiments show that the monitor discriminates the evaluated textand fixed-latent-collusion conditions across model families when matched controls are available. Second, the scalability grid shows that activation-aware monitoring can be evaluated on scheduled pair decisions rather than requiring all-pairs activation capture. Third, the steering grid characterizes how blackbox and whitebox interventions change behavior relative to matched neutral controls.

The allocated steering design is practically relevant because real platforms may have mixed permissions. Hosted agents may permit whitebox activation access, while third-party agents may only permit prompt-level blackbox intervention. The allocated grid tests this boundary directly: more whitebox coverage produces monotonic improvements in bid-distribution recovery and low-bid reduction in this grid. Because whitebox steering directly replays matched neutral counterfactuals, this dose response measures the benefit of increasing exact counterfactual access under the benchmark assumptions.

## 8 Related Work

Latent multi-agent communication. LatentMAS demonstrates that agents can exchange continuous internal representations instead of generating natural-language messages, substantially reducing token use and inference time [3]. VLA addresses the complementary governance problem: how a host can audit and intervene on a channel whose contents are absent from the public transcript.

Collusion and activation oversight. Prior work distinguishes tacit coordination, explicit cartels, and concealed communication as threats in LLM-powered multi-agent systems [2]. VLA combines three familiar forms of evidence—representation anomaly, counterfactual behavioral influence, and sparse-feature support—within an exact-join protocol that attributes public actions to particular scheduled private-channel decisions.

## 9 Conclusion

Latent communication ofers an eficient, scalable mode of coordination for large multi-agent systems by reducing the need to serialize internal state into long natural-language exchanges. This benefit also creates a governance gap: public transcripts reveal what agents do but may omit the continuous states through which they coordinate. Verifiable Latent Alignments ofers a general architecture for narrowing this gap: link private-channel records to public actions, compare suspect decisions with matched controls, model neutral communication rather than a fixed catalog of attacks, and choose interventions according to the defender’s level of model access. The auction case study provides a concrete proof of concept that this architecture can connect detection, causal diagnosis, and scalable intervention in a multi-agent setting.

## References

[1] Project NANDA. Nanda Town: An open-source sandbox for building and testing protocols for AI agents. https://nandatown.projectnanda.org/, 2026.

[2] Mohammad Sajjad Ghaemi. A survey of collusion risk in LLM-powered multi-agent systems. NeurIPS Workshop on Socially Responsible and Trustworthy Foundation Models, 2025. https: //openreview.net/forum?id=Ylh8617Qyd.

[3] Jiaru Zou et al. Latent collaboration in multi-agent systems. In Proceedings of the 43rd International Conference on Machine Learning, 2026. https://arxiv.org/abs/2511.20639.
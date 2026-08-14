# HybridSB-MoE: Dual-Domain Schrödinger Bridges with Scene-Adaptive Expert Routing for Speech Enhancement

Zhengyi Lu Aswini Sivakumar Jie Hu Yao Qiang

Department of Computer Science and Engineering, Oakland University Rochester, MI 48309

{zhengyilu, aswinisivakumar, jiehu, qiang}@oakland.edu

## Abstract

Generative speech enhancement faces three gaps: spectral models capture harmonic structure but often disrupt phase, waveform models preserve phase but miss harmonics, and Schrödinger Bridges (SB) shorten transport from noise to clean speech but leave inference cost only loosely tied to training. We propose HybridSB-MoE, a dual-domain framework that fills these gaps through three contributions unified by a single asymmetric design principle. (i) Asymmetric uncertaintyfusion: The spectral path captures epistemic uncertainty via expert disagreement, while the waveform bridge models aleatoric variance through stochastic dynamics. We fuse them asymmetrically, allowing the mixing weight to adapt to distinct error regimes rather than average predictions. (ii) Heterogeneous MoE with top-k=2 routing across five distinct architectural archetypes, where architectural diversity makes the epistemic signal indicate which inductive bias fails rather than small perturbations among similar experts. (iii) Discretization bound (Theorem 1): path-consistency and trajectory regularizers together bound the K-step bridge sampling error in 2-Wasserstein distance at rate K<sup>−α</sup>, making small-K inference an objective-level guarantee rather than an empirical claim. On VoiceBank+DEMAND, HybridSB-MoE outperforms diffusion- and SB-based baselines at their step budgets while remaining competitive with consistency-distilled few-step methods.

## 1 Introduction

Speech enhancement (SE) has long faced a structural domain trade-off. Spectral methods are effective at modeling harmonic structure and stationary noise, but often fragment phase across STFT frames; waveform methods preserve phase continuity, but can struggle with harmonically structured interference (Fig. 1). This trade-off becomes especially limiting in real-world deployment, where a hearing aid on a moving bus, a video-conferencing system in a crowded café, or an in-car voice assistant must handle harmonic engine drone, broadband ambient hum, and non-stationary crowd babble under millisecond latency budgets and without scene labels at inference time [1, 2]. These settings require more than a universally stronger model: they require a mechanism that can exploit complementary inductive biases and decide which one to trust for each input.

A decade of deep learning has advanced SE from DNN masking [3, 4], through time-domain [5, 2, 6] and spectro-temporal [7, 8] architectures to diffusion-based generative models [9, 10, 11] and Schrödinger Bridge formulations that replace the uninformative Gaussian prior of diffusion with direct transport from noisy to clean speech. Despite this progress, three structural limitations persist that are precisely what a dual-domain framework must address. (i) Single-domain commitment. Existing methods typically operate in either the waveform or spectral domain, sacrificing the complementary inductive bias of the other [8]. (ii) Uniform processing across heterogeneous noise. A single network handles stationary appliance hum, harmonic engine noise, and non-stationary crowd babble alike, despite their structurally different inductive biases [12, 13]. (iii) Loosely controlled sampling cost. Generative SE pipelines often require many iterative refinement steps, with limited formal connection between the training objective and the inference budget [10, 11].

The natural response, i.e., running a spectral and a waveform pathway in parallel and combining them, fails on its own, because a generic ensemble of two prediction streams provides no mechanism to know which pathway is failing on a given input. Without such signal, the combination defaults to fixed-weight averaging, which is provably suboptimal whenever the two branches’ errors are uncorrelated, and the dual-domain promise reduces to a parameter-count increase.

We introduce HybridSB-MoE, a dual-domain framework organized around one asymmetric idea: a multiexpert spectral path naturally produces an epistemic uncertainty signal (which expert is right?), and a stochastic waveform bridge naturally produces an aleatoric one (intrinsic transport noise). These are categorically different signals about categorically different errors. Pairing the two pathways and fusing on this asymmetry yields a mixing weight that selects between two error regimes rather than averaging two predictions, transforming the dual-domain promise into a principled co-design. Two further architectural choices are required for this idea to work. First, the spectral pathway must produce informative expert disagreement, which requires architectural heterogeneity rather than capacity-replicated variation. We therefore use heterogeneous expert archetypes, each aligned with a canonical noise-processing primitive, including low-rank denoising, wide receptive fields, information bottlenecks, harmonic bases, and universal approximation, and combine them under sparse routing. Second, the waveform pathway must remain faithful under the small inference budget K targeted by the system. We therefore train it with path-consistency and trajectory regularizers, and prove in Theorem 1 that jointly minimizing them bounds the K-step bridge discretization error in 2-Wasserstein distance, making small-K inference a consequence of the training objective rather than an empirical heuristic. In Section 3, we further show that no proper subset of these three components recovers the central design property.

![](images/0559b5a5e5b8a7d75bfb8147d32b9d89dbf3f0408bbfacd52c44b368a24ff31e.jpg)  
Figure 1: The persistent dual-domain dichotomy in speech enhancement. Waveform processing preserves temporal fine structure and phase coherence but underperforms on harmonically structured interference; spectral processing captures harmonic structure and stationary noise patterns but fragments phase across STFT frames. The two domains succeed and fail in opposite regimes, and existing generative methods inherit one half of the tradeoff by committing to a single domain.

## Our contributions are as follows:

1. A dual-domain framework built around asymmetric uncertainty. We propose HybridSB-MoE, a dual-domain framework that couples spectral MoE and waveform SB pathways. By pairing each path with its natural uncertainty signal, epistemic disagreement for spectral experts and aleatoric variance for the waveform bridge, HybridSB-MoE enables fusion that adapts to error regimes rather than fixed-weight averaging.

2. A heterogeneous spectral MoE for informative epistemic disagreement. We design a sparse top-k=2 spectral MoE with architecturally distinct expert archetypes. This heterogeneity makes expert disagreement reflect mismatched inductive biases rather than small perturbations among capacity-replicated experts.

3. A regularized waveform bridge for few-step inference. We train the waveform SB with path consistency and trajectory regularizers, and prove in Theorem 1 that they bound the K-step discretization error in 2-Wasserstein distance, linking small-K inference to the training objective.

4. SOTA on VoiceBank+DEMAND. HybridSB-MoE improves over diffusion- and SB-based baselines at their respective step budgets, remains competitive with consistency-distilled few-step methods, and provides calibrated fusion uncertainty for input-adaptive dual-domain enhancement.

## 2 Related Work

Discriminative SE. The field has progressed from DNN masking [3, 4] that surpassed classical Wiener and MMSE estimators [14, 15], through time-domain encoder–decoders (Conv-TasNet [5, 16], DEMUCS [2]), dual-path sequence models [6], full-sub-band hybrids (FullSubNet+ [7]), spectrotemporal grids (TF-GridNet [8]), and recent Mamba-based linear-complexity backbones [17, 18]. Joint enhancement objectives that combine SI-SDR with perceptual surrogates push fidelity further [19, 20], yet the underlying paradigm remains a one-shot deterministic regression. Two limitations therefore persist regardless of architecture. (i) Inherent ambiguity: under severe non-stationary or harmonically structured noise, the noisy-to-clean mapping admits multiple plausible solutions, and a deterministic regressor must collapse this multimodal posterior to a single point estimate, which typically manifests as residual noise or over-suppressed harmonics [21]. (ii) No uncertainty: discriminative networks expose no signal at inference time to flag this ambiguity, so a downstream module cannot tell when the prediction should be trusted—making them unsuitable as one branch of a confidence-routed dual-domain combiner.

Generative models and Schrödinger Bridges for SE. Likelihood-based [21] and score-based [9, 22, 23] diffusion models reverse a noise corruption process; SGMSE/SGMSE+ [10, 11] apply this in the complex spectrogram domain, jointly modeling magnitude and phase, but require many iterative steps from an uninformative Gaussian prior. Two recent paradigms exploit the structure of the noisy observation directly. Flow-matching approaches [24, 25] regress velocity fields between coupled distributions, shortening the sampling trajectory but typically without an explicit stochastic perturbation. Schrödinger Bridges [26, 27] learn the optimal-transport-like coupling between the noisy observation and clean speech, retaining a small bridge variance for full-support marginals. SB for SE [28, 29, 30, 31, 32] confirm improved low-SNR structure preservation along this shorter path. Empirical step-count reductions have been achieved by consistency distillation [33, 34], adversarial few-step training [35], and SB-consistency trajectory models [36]. None of these works links the training objective to the inference budget, where small K is asserted but not derived. Theoretical convergence analyses for diffusion sampling exist in the broader literature [37], but their bounds are stated in terms of the score-matching error rather than tractable, training-time-minimized regularizers, and have not been adapted to the doubly-conditioned SB setting. Moreover, these methods remain single-domain and lack calibrated uncertainty for fusion. We address both with asymmetric uncertainty fusion and a discretization bound (Theorem 1) that links small-K inference to explicit training-time quantities.

Mixture-of-Experts for SE. MoE enables conditional computation with sparse routing, scaling capacity without proportional inference cost [38, 39, 40]. In SE, sparse MoE [12] first enabled scene-adaptive processing via noise-clustered experts; zero-shot personalization [41] extended this to speakers via embedding-based routing; clean-cluster pre-training [13] enforced expert specialization through pre-training initialization; and dynamic capacity allocation [42] improved routing efficiency at inference. However, this line of work shares three limitations. (i) Homogeneity. Existing methods replicate a single architecture across experts, assuming one computational primitive can handle tonal, ambient, harmonic, and crowd noise. This assumption breaks down under realistic, structurally diverse noise. (ii) Routing without uncertainty. Existing routing only partitions inputs into specialist regions, rather than exposing interpretable uncertainty; thus, the epistemic signal from multi-expert disagreement remains unused. (iii) Separationfrom generative SE. MoE routing and generative bridge models remain disconnected in SE, leaving scene-adaptive computation and structured distributional transport as separate lines of work. We address all three with heterogeneous archetype experts that encode distinct noise-specific inductive biases. Their disagreement provides the epistemic signal which, together with the SB’s aleatoric variance, drives asymmetric fusion. To our knowledge, this is the first joint MoE–SB framework for SE.

Uncertainty quantification and dual-branch fusion. Calibrated uncertainty for deep regression is most commonly obtained via deep ensembles [43] or learned heteroscedastic variance heads, with classification calibration assessed via ECE [44]. Prior dual-branch SE methods that combine timeand frequency-domain estimates typically use either fixed-weight averaging or learned attention over predictions [8, 2]; none assigns categorically distinct uncertainty types to the two branches, and so the fusion cannot route on which kind of error each branch is exposed to. Our pathway-typed asymmetric fusion is, to our knowledge, the first to exploit this structure, and the calibration loss (Eq. (11)) ties the abstract epistemic/aleatoric labels to measurable per-pathway reconstruction errors so the routing weight is learned, not asserted.

![](images/70e95f6286e788762eff775f78ff4f2f292de2294752e3feaa5e45586f49fd99.jpg)  
Figure 2: Overview of HybridSB-MoE. The spectral pathway (top) routes log-magnitude features $z \dot { = } \log | S \{ y \} |$ through N heterogeneous archetype experts via gating $G ( z { \bar { ) } }$ to produce $x _ { \mathrm { s p e c } } ( t )$ The waveform pathway (bottom) iteratively refines an SB state $[ x _ { t } , y ]$ through a U-Net to produce $x _ { \mathrm { w a v e } } ( t )$ . An uncertainty-aware fusion combines both via $u _ { \mathrm { e p i } }$ (top-k expert disagreement) and $u _ { \mathrm { a l e } }$ (bridge-variance head) to yield xˆ(t).

## 3 Method

HybridSB-MoE has three coupled components: a heterogeneous spectral MoE, an SB waveform pathway with path-consistency and trajectory regularizers (for which we prove a K-step discretization bound, Theorem 1), and an asymmetric uncertainty fusion that selects between epistemic and aleatoric error regimes. Figure 2 illustrates the overall HybridSB-MoE framework. We develop each component in turn and close the section by showing that no proper subset recovers the central design property.

Overview. Given a noisy input y(t), HybridSB-MoE processes it through two parallel pathways designed to capture complementary speech characteristics. The spectral pathway applies the heterogeneous MoE to the log-magnitude STFT for scene-adaptive enhancement. In parallel, the waveform pathway runs an efficient SB on the time-domain signal to preserve phase. The two pathways exhibit asymmetric reliability across acoustic conditions: spectral processing excels in harmonically structured noise, whereas waveform processing is more robust in phase-sensitive scenarios. Consequently, a fixed fusion strategy is inherently suboptimal. We therefore introduce an uncertainty-aware fusion module that integrates $x _ { \mathrm { s p e c } } ( t )$ and $x _ { \mathrm { w a v e } } ( t )$ by dynamically weighting each pathway based on confidence, producing the final enhanced signal ${ \hat { x } } ( t )$ . Both pathways run in parallel at inference, and long utterances use overlap-add segmentation (Appendix B).

Spectral feature extraction. We apply the STFT [45] to obtain the complex spectrogram $S \{ y \} \in$ $\mathbb { C } ^ { \mathbf { \tilde { \it F } } \times T _ { f } }$ (F frequency bins) and decompose it in polar form into magnitude $| S \{ y \}$ | and phase $\angle S \{ y \}$ processed by separate heads (below). The MoE consumes log-magnitude features $z = \log | S \{ y \} | \in$ $\backslash \backslash \mathbb { R } ^ { F \times T _ { f } }$ , whose logarithmic scaling compresses the dynamic range in line with human hearing.

Heterogeneous mixture-of-experts. Unlike conventional homogeneous MoE designs [38, 39, 40] in which all experts share the same architecture and differ only in learned weights, we instantiate distinct architectural archetypes, each tailored to a canonical noise-processing primitive, i.e., Home, Nature, Office, Transport, and Public for VoiceBank+DEMAND (see Appendix A for full specification). Under sparse top-k routing, pairwise mixtures and weighted blends of these archetypes cover the 14 noise categories in VoiceBank+DEMAND [46] combinatorially rather than by a dedicated expert per noise, and the design extends to new acoustic environments by adding archetypes per noise family encountered, without altering the rest of the framework (see Appendix A for details). All experts share a backbone of normalization (group norm [47]) → variable internal modules → bottleneck projection, ensuring routing-interface compatibility while permitting structurally divergent intermediate computation. This design, a small semantically anchored basis composed of sparse routing, replaces the parameter scaling of large homogeneous MoEs with interpretable routing and meaningful expert disagreement, which we use for asymmetric fusion below.

Scene-adaptive sparse routing. Standard sparse routing [38, 39] assigns each input token to its top-k experts independently, which is well-suited to language modelling but ill-suited to SE: the dominant noise type is a global property of an utterance, while local acoustic events (a phoneme onset, a transient hit) require frame-level adaptivity. We therefore use a two-level gate: Archetype-level routing $G _ { \mathrm { a r c h } } ( z )$ pools z across time and decides the dominant archetype mixture for the whole utterance; token-level routing $G _ { \mathrm { t o k e n } } ( z )$ refines this assignment per frame. The two combine as $G ( z ) = \alpha G _ { \mathrm { a r c h } } ( z ) + ( 1 - \bar { \alpha _ { } } ) G _ { \mathrm { t o k e n } } ( z )$ with $\alpha \in [ 0 , 1 ]$ . An MLP maps $G ( z )$ to per-expert logits; the top-k entries are softmax-renormalized to weights $\{ G _ { i } ( z ) \} _ { i \in \mathbb { Z } _ { k } }$ , and the routed output is

$$
\hat { x } _ { \mathrm { s p e c } } = \sum _ { i \in \mathcal { T } _ { k } } G _ { i } ( z ) E _ { i } ( z ) ,\tag{1}
$$

where $\mathcal { T } _ { k }$ indexes the selected experts and $E _ { i }$ are expert networks. To prevent expert collapse [38, 39], a standard load-balancing regularizer

$$
\mathcal { L } _ { \mathrm { a u x } } = \lambda _ { I } \mathrm { V a r } ( p _ { i } ) + \lambda _ { L } \mathrm { V a r } ( n _ { i } )\tag{2}
$$

penalizes uneven utilization, where $p _ { i }$ and $n _ { i }$ denote the average routing probability and token count of expert i, respectively.

Spectral reconstruction. The routed features $\hat { x } _ { \mathrm { s p e c } }$ pass through two parallel 1 × 1-conv heads. Since phase cannot be inferred from log-magnitude alone, the phase head also consumes the noisy phase as $( \sin \angle S \{ y \} $ , cos $\angle S \{ y \} )$ ), concatenated along the channel dimension:

$$
\hat { M } = M _ { \mathrm { m a x } } \mathrm { s i g m o i d } ( h _ { \mathrm { m a g } } ( \hat { x } _ { \mathrm { s p e c } } ) ) , \Delta \phi = \phi _ { \mathrm { m a x } } \mathrm { t a n h } ( h _ { \mathrm { p h a } } ( \hat { x } _ { \mathrm { s p e c } } , \sin \angle S \{ y \} , \cos \angle S \{ y \} ) ) ,\tag{3}
$$

where $M _ { \mathrm { m a x } } , \phi _ { \mathrm { m a x } }$ bound the mask and phase correction to prevent over-suppression, excessive amplification, and phase-wrap instability. The enhanced spectrum $\hat { S } = \hat { M } \odot | \bar { S \{ y \} } | \cdot e ^ { j ( \angle S \{ y \} + \Delta \phi ) }$ is inverted to $x _ { \mathrm { s p e c } } ( t )$ via iSTFT with overlap-add. STFT processing nevertheless fragments temporal fine structure through frame-wise phase decoupling, motivating the parallel waveform pathway below.

Schrödinger Bridge formulation. The continuous-time SB problem seeks the stochastic process minimizing the KL divergence to a reference Brownian motion subject to fixed marginal distributions at both endpoints [26, 27, 48]. In our application the two endpoints are the noisy observation y and the clean target x, so the bridge directly couples the conditional distributions our system must transport between. Where standard diffusion [9] traverses the full path from a data-independent Gaussian prior, the bridge exploits the substantial mutual information between y and x to yield a shorter, more informative trajectory and, empirically, fewer sampling steps and better low-SNR structure preservation [28, 29, 30]. Conceptually adjacent flow-matching methods [24, 25] also couple paired distributions but typically without an explicit bridge perturbation; our SB construction retains a small noise term $\sigma _ { t } \epsilon$ to keep the marginal $p ( x _ { t } \mid x , y )$ full-support, which is required for score-based reverse sampling and for the trajectory regularizer to act as a self-consistency constraint.

The intermediate state at diffusion time $t \in [ 0 , T ]$ interpolates between y and x with a small Gaussian perturbation,

$$
\begin{array} { r } { x _ { t } = \sqrt { \bar { \beta } _ { t } } x + \sqrt { 1 - \bar { \beta } _ { t } } y + \sigma _ { t } \epsilon , \epsilon \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{4}
$$

where $\bar { \beta } _ { t } = f ( t ) / f ( 0 )$ with $f ( t ) = \cos ^ { 2 } ( ( t / T + s ) / ( 1 + s ) \cdot \pi / 2 )$ is a cumulative cosine schedule [49] decreasing monotonically from $\bar { \beta } _ { 0 } = 1$ to $\bar { \beta } _ { T } = 0 \left( \mathrm { o f f s e t } s { = } 0 . 0 0 8 \right)$ , and $\sigma _ { t } = \sigma _ { \operatorname* { m a x } } \sqrt { \bar { \beta } _ { t } ( 1 - \bar { \beta } _ { t } ) }$ $( \sigma _ { \mathrm { m a x } } { = } 0 . 0 5 )$ vanishes at both endpoints, so $x _ { 0 } = x$ and $x _ { T } = y$ deterministically while $\sigma _ { t } \epsilon$ supplies the full-support marginal $p ( x _ { t } \mid x , y )$ required by the score-based reverse process.

The reverse process is parameterized as a learnable denoiser $\hat { x } _ { \theta } ( x _ { t } , y , t )$ that predicts the clean target directly from the bridge state and the observation. Rather than solving the reverse-time stochastic differential equation (SDE) associated with the bridge in closed form, we adopt a tractable bridgeconsistent update that iteratively re-injects the predicted clean estimate into the forward construction of Eq. (4) at the previous timestep:

$$
x _ { t - 1 } = \sqrt { \bar { \beta } _ { t - 1 } } \hat { x } _ { \theta } ( x _ { t } , y , t ) + \sqrt { 1 - \bar { \beta } _ { t - 1 } } y + \sigma _ { t - 1 } z , \qquad z \sim \mathcal { N } ( 0 , I ) .\tag{5}
$$

This data-prediction update is the bridge analogue of DDPM/DDIM ancestral sampling [9, 23]: deterministic when $\sigma _ { t - 1 } = 0$ , stochastic otherwise. Bridge-marginal consistency is enforced empirically through the path-consistency and trajectory regularizers introduced below. The denoiser is a 1D U-Net with transformer bottleneck and FiLM timestep conditioning (Appendix B). The training objective is the data-prediction loss:

$$
\mathcal { L } _ { \mathrm { S B } } ^ { \mathrm { d a t a } } = \mathbb { E } _ { t , ( x , y ) , \epsilon } \big \| \hat { x } _ { \theta } ( x _ { t } , y , t ) - x \big \| _ { 2 } ^ { 2 } ,\tag{6}
$$

augmented by the path-consistency and trajectory regularizers introduced next.

Path-consistency and trajectory regularizers. $\mathbf { A }$ fixed front-loaded schedule $t _ { k } = T ( k / K ) ^ { \gamma }$ with γ<1 concentrates the K-step budget in the early-trajectory regime where the SNR changes fastest and where errors propagate to all later steps [9]. Two regularizers then encourage a smooth, low-curvature denoising path in the $( y , x )$ subspace. The path-consistency loss enforces cross-timestep agreement of clean-signal predictions along the same trajectory:

$$
\mathcal { L } _ { \mathrm { p a t h } } = \mathbb { E } _ { t , t ^ { \prime } } \left[ \left| \left| \hat { x } _ { \theta } ( x _ { t } , y , t ) - \hat { x } _ { \theta } ( x _ { t ^ { \prime } } , y , t ^ { \prime } ) \right| \right| ^ { 2 } \right]\tag{7}
$$

where $t , t ^ { \prime }$ are independently sampled along the same bridge trajectory. This penalizes inconsistent clean-signal predictions across timesteps, reducing reverse-process curvature (cf. consistency models [34], but here for path smoothing, not distillation). The trajectory loss anchors $x _ { t }$ to the scheduleconsistent reconstruction:

$$
\mathcal { L } _ { \mathrm { t r a j } } = \mathbb { E } _ { t } \left[ \left. x _ { t } - \left( \sqrt { \bar { \beta } _ { t } } \hat { x } _ { \theta } ( x _ { t } , y , t ) + \sqrt { 1 - \bar { \beta } _ { t } } y \right) \right. ^ { 2 } \right] .\tag{8}
$$

At the optimum ${ \hat { x } } _ { \theta } = x$ , the residual reduces to $\sigma _ { t } \epsilon .$ , so $\mathcal { L } _ { \mathrm { t r a j } }$ couples the prediction to the specific bridge state $x _ { t }$ at each timestep, complementing the in-expectation regression of Eq. (6). At inference, we use $K { = } 8$ steps with the front-loaded non-uniform discretization $t _ { k } = T ( k / K ) ^ { \gamma } , \gamma { = } 0 . 6$ (see Appendix B for more details).

Discretization bound. The two regularizers above make small-K inference viable, and we now show why. Let $\hat { p } _ { K }$ be the law of the K-step rollout of Eq. (5) at $t { = } 0$ and $p _ { 0 } ^ { \mathrm { b r } }$ the continuous-time bridge marginal at $t { = } 0$ , both conditional on $y .$

Assumption 1 (Regularity). $( i ) \hat { x } _ { \theta } ( \cdot , y , t )$ is $L _ { x ^ { - L } }$ ipschitz in its first argument and L -Lipschitz in $t . \ ( i i ) \ \bar { \beta } _ { t }$ and $\sigma _ { t } = \sigma _ { \operatorname* { m a x } } \sqrt { \bar { \beta } _ { t } ( 1 - \bar { \beta } _ { t } ) }$ are $C ^ { 1 }$ on $[ 0 , T ]$ , with $\bar { \beta } _ { t }$ monotone and bounded second derivative. (iii) $\mathbb { E } \Vert x _ { t } \Vert ^ { 2 }$ and $\mathbb { E } \| \dot { x } _ { \theta } ( x _ { t } , \dot { y } , t ) \| ^ { 2 }$ are uniformly bounded in t.

Theorem 1 (K-step bridge discretization bound). Under Assumption $^ { l , }$ with the schedule $t _ { k } =$ $T ( k / K ) ^ { \gamma } , \gamma \in ( 0 , \bar { 1 } ]$ ],

$$
\begin{array} { r } { W _ { 2 } \bigl ( \hat { p } _ { K } , p _ { 0 } ^ { \mathrm { b r } } \bigr ) \ \leq \ C _ { 1 } K ^ { - \alpha } \ + \ C _ { 2 } \sqrt { \mathscr { L } _ { \mathrm { p a t h } } ^ { \star } + \mathscr { L } _ { \mathrm { t r a j } } ^ { \star } } , } \end{array}\tag{9}
$$

with $\alpha = \operatorname* { m i n } ( 1 , \gamma )$ $\mathcal { L } ^ { \star }$ the training-termination values ofEq. $( 7 ) \mathrm { - } ( 8 )$ , and $C _ { 1 } , C _ { 2 }$ depending only on $L _ { x } , L _ { t } , \sigma _ { \operatorname* { m a x } } ,$ , and the curvature of $\hat { \beta } _ { t }$

With $\alpha = \operatorname* { m i n } ( 1 , \gamma )$ , front-loading $( \gamma < 1 )$ concentrates steps near $t { = } 0$ where $\Phi ^ { \prime }$ is largest, which is what drives small-K behavior empirically. Each term corresponds to a separate lever: $\operatorname { C } _ { 1 } K ^ { - \alpha }$ is controlled by inference-time $K , \gamma ; C _ { 2 } \sqrt { \mathcal { L } ^ { \star } }$ is controlled by training-time minimization of the two regularizers and is independent of K. Two consequences follow. First, once $\mathcal { L } ^ { \star }$ is small, the bound saturates: any K with $C _ { 1 } K ^ { - \alpha } { \lesssim } C _ { 2 } { \sqrt { \mathscr { L } ^ { \star } } }$ already attains the asymptotic floor, which is what justifies our $K { = } 8$ rather than the much larger K of standard diffusion sampling. Second, the regularizers are not auxiliary terms; without them, the second term has no training-time upper bound, and the inequality does not exist as a function of $K$ . Therefore, removing either breaks the entire argument. This is the formal sense in which path-consistency and trajectory regularization are load-bearing for our inference schedule. Eq. (9) is closely related to recent convergence results for diffusion sampling [37] but adapts those analyses to the doubly-conditioned bridge process and, importantly, replaces score-matching error with the explicit, training-time-minimized regularizer pair, which makes the bound directly actionable as a design tool. The full proof includes a one-step fidelity lemma combined with a non-uniform Riemann estimate under a synchronous coupling, with details deferred to Appendix C. We present Eq. (9) as a design-justifying inequality, not a tight predictor.

Asymmetric uncertainty fusion. A naive dual-branch ensembler treats the two pathways symmetri cally; ours does not, by design. Epistemic uncertainty (‘which expert is right?’) is well-defined only when multiple experts coexist (the spectral MoE); the waveform pathway has no analog. Aleatoric uncertainty (intrinsic stochasticity of the bridge SDE in Eq. (4)) is well-defined only when the forward process is genuinely stochastic (the waveform pathway); the deterministic spectral mask in Eq. (3) has no analog. Each pathway therefore carries exactly one uncertainty type: $u _ { \mathrm { e p i } }$ flags an architectural mismatch, $u _ { \mathrm { a l e } }$ flags a stochastic transport error. The fusion selects between these two error regimes; it does not average two predictions. Concretely, $\begin{array} { r } { u _ { \mathrm { e p i } } = \frac { 1 } { k T _ { f } } \sum _ { i \in \mathcal { T } _ { k } } \| E _ { i } ( z ) - \bar { E } ( z ) \| _ { 2 } ^ { 2 } } \end{array}$ with $\begin{array} { r } { \bar { E } ( z ) = \frac { 1 } { k } \sum _ { i \in \mathcal { T } _ { k } } E _ { i } ( z ) } \end{array}$ (deep-ensembles sense [43], not Bayesian posterior); $u _ { \mathrm { a l e } }$ is from the U-Net’s per-sample log-variance head, exponentiated and time-averaged. After z-score normalization, a 2-layer MLP gives $\bar { w } { = } \sigma ( \mathrm { M L P } ( \tilde { u } _ { \mathrm { e p i } } , \tilde { { u } _ { \mathrm { a l e } } } ) ) \in [ 0 , 1 ]$ , and

$$
\hat { x } ( t ) = w \cdot x _ { \mathrm { s p e c } } ( t ) + ( 1 - w ) \cdot x _ { \mathrm { w a v e } } ( t ) .\tag{10}
$$

High expert disagreement pushes w toward the waveform pathway and high SB variance toward the spectral pathway: each pathway is deferred away from when its native error is large. A calibration loss anchors the two scalars to the errors they should track,

$$
\mathcal { L } _ { \mathrm { c a l } } = ( u _ { \mathrm { e p i } } - \| x _ { \mathrm { s p e c } } - x \| _ { 2 } ^ { 2 } ) ^ { 2 } + ( u _ { \mathrm { a l e } } - \| x _ { \mathrm { w a v e } } - x \| _ { 2 } ^ { 2 } ) ^ { 2 } ,\tag{11}
$$

where $u _ { \mathrm { e p i } } , u _ { \mathrm { a l e } }$ are pre-normalized; without it, the scalars have no incentive to scale with error, and the fusion’s dependence on them would be arbitrary.

Training. All components are trained end-to-end. Each step runs both pathways in parallel: the spectral path produces $x _ { \mathrm { s p e c } } , u _ { \mathrm { e p i } } ;$ the waveform path samples $t , \epsilon .$ , forms x via Eq. (4), and outputs $x _ { \mathrm { w a v e } } , u _ { \mathrm { a l e } }$ . The fusion yields xˆ. The objective combines reconstruction, SB, load-balancing, and calibration losses:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { S B } } \mathcal { L } _ { \mathrm { S B } } + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { a u x } } + \lambda _ { \mathrm { c a l } } \mathcal { L } _ { \mathrm { c a l } } ,\tag{12}
$$

with $\mathcal { L } _ { \mathrm { r e c } } ~ = ~ \| \hat { x } - x \| _ { 2 } ^ { 2 }$ and ${ \mathcal { L } } _ { \mathrm { S B } } ~ = ~ { \mathcal { L } } _ { \mathrm { S B } } ^ { \mathrm { d a t a } } + \lambda _ { \mathrm { p a t h } } { \mathcal { L } } _ { \mathrm { p a t h } } + \lambda _ { \mathrm { t r a j } } { \mathcal { L } } _ { \mathrm { t r a j } }$ . Loss weights (gridsearched on a 10% VB split): $\lambda _ { \mathrm { S B } } { = } 1 . 0 , \tilde { \lambda _ { \mathrm { p a t h } } } { = } 0 . \dot { 1 } , \lambda _ { \mathrm { t r a j } } { = } 0 . 0 5 , \lambda _ { \mathrm { a u x } } { = } 0 . 0 1 , \lambda _ { \mathrm { c a l } } { = } 0 . 0 5 ;$ ceilings $\scriptstyle M _ { \mathrm { m a x } } = 5 . 0 , \phi _ { \mathrm { m a x } } = \pi / 4$ . Full procedure in Appendix B.

Why the components are coupled, not stacked. The central design property—a fusion that routes on epistemic vs. aleatoric error regimes at a sampling budget derivedfrom training—requires all three components and survives no proper subset. Three counterfactuals make this concrete. (C1) Symmetric uncertainty heads on both pathways (the standard dual-branch ensemble): the fusion sees two aleatoric scalars and no epistemic one, making it blind to spectral expert disagreement. Thus, the mixing weight collapses into a generic confidence average, consistent with the 0.17 PESQ drop observed for $^ { 6 6 } \mathrm { w } / \mathrm { o }$ uncertainty fusion” in Figure 3a. (C2) Homogeneous MoE in place of distinct archetypes: top-k blends span one function class instead of distinct inductive biases. In this way, $u _ { \mathrm { e p i } }$ no longer indicates which bias is mismatched but only weight perturbations among identicalarchitecture experts, resulting in 0.43 PESQ drop as shown in Figure 3a. (C3) Drop the regularizers: Theorem 1’s second term loses its training-time upper bound, so small-K inference is no longer derivable; $u _ { \mathrm { a l e } }$ inherits the failure since the SB error it was calibrated against is now uncontrolled. Removing the SB pathway entirely, leading to −0.63 PESQ, is the limit of this collapse. In all three cases the asymmetric fusion reduces to a fixed-weight or generic ensemble, the small-K guarantee disappears, or both: the MoE produces the error only $u _ { \mathrm { e p i } }$ detects, the regularized SB produces the error only $u _ { \mathrm { a l e } }$ detects, Theorem 1 controls the SB-side error at the deployment budget, and the asymmetric fusion is the only mechanism that routes between them. The ablation pattern in Figure 3a of Section 4 is the empirical signature of this coupling.

## 4 Experiments and Discussion

Setup. Dataset: We evaluate our method on VoiceBank+DEMAND [50, 46], a widely used benchmark containing 11,572 training and 824 test utterances from 28 and 2 speakers, respectively. Clean speech is mixed with 14 noise types. All audio signals are resampled to 16 kHz. Metrics: We report PESQ [51] and STOI [52] for perceptual quality and intelligibility, composite metrics CSIG, CBAK, and COVL [53] for signal/background/overall quality, Expected Calibration Error (ECE) [44] for uncertainty calibration, and Real-Time Factor (RTF) with end-to-end latency for computational efficiency. Implementation: We use STFT with 1024-point FFT, 256-sample hop (16 ms), and Hann window, yielding 513 frequency bins. The MoE comprises 5 archetype experts (one per scene family in VoiceBank+DEMAND, see Appendix A) with Top-k=2 routing. The waveform U-Net has 4 encoder/decoder levels with a transformer bottleneck. Training uses AdamW (learning rate $2 \times 1 0 ^ { - 4 }$ cosine schedule, 200 epochs, batch size 32) on 2 NVIDIA RTX 5090 GPUs. Full hyperparameters are listed in Appendix B. Baselines: We compare against discriminative methods (SEGAN [4], SEMamba [17], Mamba-SEUNet [18]), diffusion/SB-based generative models (SGMSE+ [11], SB-SE [28], SBCTM [36]), and the consistency-distilled ROSE-CD [33]. All baselines are retrained from scratch using their official implementations under a unified protocol (matched preprocessing, 16 kHz sampling, train/test split), and all reported metrics are computed by us with a shared evaluation script to avoid implementation-induced discrepancies.

Table 1: Performance comparison on VoiceBank+DEMAND. Best results in bold (ties bolded jointly). All baseline numbers are reproduced by us under a unified evaluation protocol. Our method achieves the best score on every metric, with calibrated uncertainty estimates as an additional benefit.
<table><tr><td colspan="8"></td></tr><tr><td>Method</td><td>Type</td><td>PESQ↑</td><td>STOI↑</td><td>CSIG↑</td><td>CBAK↑</td><td>COVL↑</td><td></td></tr><tr><td>Noisy</td><td></td><td>1.97</td><td>0.91</td><td></td><td>1</td><td></td><td>一</td></tr><tr><td>SEGAN [4]</td><td>Discriminative</td><td>2.16</td><td>0.92</td><td></td><td></td><td></td><td>1</td></tr><tr><td>SEMamba [17]</td><td>Discriminative</td><td>3.55</td><td>0.96</td><td>4.79 4.82</td><td></td><td>3.63</td><td>4.37</td></tr><tr><td>Mamba-SEUNet [18]</td><td>Discriminative</td><td>3.73</td><td>0.96</td><td></td><td></td><td>3.67</td><td>4.40</td></tr><tr><td>SGMSE+ [11]</td><td>Diffusion</td><td>3.45</td><td>0.95</td><td>4.71</td><td></td><td>3.64</td><td>4.31</td></tr><tr><td>SBCTM [36]</td><td>SB-based</td><td>3.58</td><td>0.95</td><td>4.66</td><td></td><td>3.43</td><td>4.52</td></tr><tr><td>SB-SE [28]</td><td>SB-based</td><td>3.70 3.85</td><td>0.95 0.96</td><td>4.77 4.63</td><td></td><td>3.75</td><td>4.48</td></tr><tr><td>ROSE-CD [33]</td><td>Distillation</td><td></td><td></td><td></td><td></td><td>3.37</td><td>4.30</td></tr><tr><td>HybridSB-MoE (Ours)</td><td>Hybrid</td><td>3.88</td><td>0.96</td><td></td><td>4.82</td><td>3.85</td><td>4.82</td></tr></table>

Main results. Table 1 reports the comparison on VoiceBank+DEMAND. HybridSB-MoE attains the best score on every metric, strictly dominating on PESQ, CBAK, and COVL and tying for the lead on STOI and CSIG. Two clarifications anchor the comparison. (i) We do not claim the smallest sampling budget as ROSE-CD reaches a single step via consistency distillation. At our K=8 we strictly dominate SGMSE+, SB-SE, and SBCTM at their larger budgets, and exceed ROSE-CD across all five quality metrics despite its smaller K. (ii) The CBAK gain is the most informative single number, i.e., +0.10 over the strongest SB baseline (SB-SE) and +0.48 over ROSE-CD, indicating that dual-domain processing suppresses background noise more effectively than either pure-spectral or pure-waveform generative pipelines. At the same time, the COVL gain (4.82 vs. 4.52) confirms this without sacrificing signal fidelity. Furthermore, the choice K=8 is not heuristic. By Theorem 1, once $\mathcal { L } _ { \mathrm { p a t h } } ^ { \star } + \mathcal { L } _ { \mathrm { t r a j } } ^ { \star }$ is minimized, the $C _ { 1 } K ^ { - \alpha }$ term saturates at modest K, see Appendix E.

Ablation studies. Figure 3a reports the ablation results. The performance drops follow the asymmetric uncertainty design. Removing the SB pathway causes the largest degradation (−0.63 PESQ), since the model loses both waveform-domain phase coherence and the aleatoric signal needed for asymmetric fusion. Removing the MoE module yields a smaller but still substantial drop (−0.43 PESQ), as the SB pathway remains active but the complementary epistemic signal from expert disagreement is lost. Replacing uncertainty-aware fusion with equal weighting also reduces performance (−0.17 PESQ), showing that both pathways remain useful but require principled routing rather than generic averaging. This ordering, $0 . 6 3 > 0 . 4 3 > 0 . 1 7$ , supports our central design claim: ablations that remove a distinct uncertainty channel are more damaging than those that only weaken fusion over intact pathways. Sequential variants in either direction also underperform parallel fusion (−0.30/−0.39 PESQ), suggesting that cascading one domain through the other weakens the structural source of pathway-specific uncertainty. Figure 3b further validates the sparse routing choice. Increasing k from 1 to 2 improves PESQ from 3.74 to 3.88 with a modest RTF of 0.28, indicating that two active experts provide useful complementary inductive biases. Increasing to k=3 adds only +0.01 PESQ while increasing RTF by 25%, so we adopt top-k=2 routing as the default quality–latency trade-off.

Efficiency and robustness. Figures 3c and 3d report efficiency and robustness. With only 8 sampling steps, HybridSB-MoE achieves an RTF of 0.28 (35 ms latency), giving a 4–5× speedup over SGMSE+ and SB-SE while surpassing their PESQ; together with ROSE-CD, our method defines the Pareto frontier with a markedly higher quality ceiling. The dual-domain design is most beneficial at low SNR (+0.13 over ROSE-CD at 0 dB). Beyond quality, the calibration loss of Eq. (11) yields a fusionweight ECE of 0.042, an order-of-magnitude reduction from the 0.12 achieved by an uncalibrated single-pathway baseline; this is what makes $u _ { \mathrm { e p i } }$ and $u _ { \mathrm { a l e } }$ usable as confidence signals for downstream decisions, not just as training-time auxiliaries. A scene-stratified breakdown across all 14 noise types appears in Appendix D, with PESQ standard deviation < 0.03 confirming that combinatorial archetype coverage generalizes across scenes without per-noise tuning.

![](images/1283c3ee121a4515d81a7b52fdb3b53d2b44136278bb89858cc05731977b0379.jpg)  
(a) Components

![](images/9b476459b992eafae79e6d7d7015e86d63bc50c2f158af2f901c0973aaa8295f.jpg)  
(b) Expert count, k

![](images/38d985a0583cef9993cdb7da1e1d574b4d591f1bd1f3d89c0910d092d1101277.jpg)  
(c) Quality–RTF

![](images/99cb61834e6f62b380729400bbdbf0d465d21dc3c56ed472af10b35ddd32c8c1.jpg)  
(d) PESQ vs. SNR  
Figure 3: Ablation studies and efficiency analysis. (a) Each architectural component contributes independently. (b) Top-k=2 with 5 experts hits the quality–efficiency sweet spot. (c) Marker size ∝ parameter count; dashed line is RTF=1.0. (d) PESQ across SNR levels.

Scope and limitations. We evaluate on VoiceBank+DEMAND, the standard SE benchmark; scene adaptivity claims should be retested on broader corpora (DNS, WHAMR!, CHiME) where train and test noise distributions diverge more substantially. Theorem 1 is a design-justifying inequality with worst-case constants $C _ { 1 } , C _ { 2 } ;$ quantitatively fitting the predicted $K ^ { - \alpha }$ rate against finer NFE sweeps is a natural follow-up. The framework is single-channel; multi-channel extensions are direct (additional input streams to both pathways) but beyond this paper’s scope. Finally, while the heterogeneous archetype set generalizes by adding archetypes per new noise family (Appendix A), automatic archetype discovery from data, rather than expert-designed primitives, remains open.

## 5 Conclusion

Combining a Schrödinger Bridge with a heterogeneous mixture-of-experts is natural; the key question is how to make the combination synergistic. We do so through two linked ideas. First, pathwaytyped asymmetric uncertainty fusion uses each pathway’s characteristic uncertainty signal, epistemic disagreement from the spectral MoE and aleatoric variance from the waveform SB, so the mixing weight selects between error regimes rather than averages predictions. Second, a discretization bound (Theorem 1) links the small-K inference budget to two explicit training regularizers, replacing heuristic step-count claims with a provable objective-level guarantee. On VoiceBank+DEMAND, this co-design HybridSB-MoE outperforms diffusion- and SB-based baselines at matched step budgets while remaining competitive with consistency-distilled few-step methods. Three counterfactual ablations show that no proper subset of components preserves the central design: the asymmetric fusion needs both a multi-expert path (for epistemic disagreement) and a stochastic bridge (for aleatoric variance), and the small-K guarantee needs both regularizers. For SE, this suggests future dual-domain systems should be organized by the type of error each pathway faces, not just predictive complementarity. For generative modelling, the discretization bound shows that few-step inference can be derived from training if regularizers are chosen to control the two sources of K-step error. Future work will extend the framework to multi-channel input, larger benchmarks (DNS, WHAMR!, CHiME), and finer NFE sweeps to empirically fit the predicted $K ^ { - \alpha }$ rate.

## References

[1] D. Wang and J. Chen, “Supervised speech separation based on deep learning: An overview,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 10, pp. 1702–1726, 2018.

[2] A. Defossez, G. Synnaeve, and Y. Adi, “Real time speech enhancement in the waveform domain,” arXiv preprint arXiv:2006.12847, 2020.

[3] Y. Xu, J. Du, L.-R. Dai, and C.-H. Lee, “A regression approach to speech enhancement based on deep neural networks,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 1, pp. 7–19, 2015.

[4] S. Pascual, A. Bonafonte, and J. Serra, “Segan: Speech enhancement generative adversarial network,” arXiv preprint arXiv:1703.09452, 2017.

[5] Y. Luo and N. Mesgarani, “Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation,” IEEE/ACM Trans. Audio, Speech and Lang. Proc., vol. 27, no. 8, p. 1256–1266, Aug. 2019. [Online]. Available: https://doi.org/10.1109/TASLP.2019.2915167

[6] Y. Luo, Z. Chen, and T. Yoshioka, “Dual-path rnn: efficient long sequence modeling for time-domain single-channel speech separation,” in ICASSP, 2020, pp. 46–50.

[7] J. Chen, Z. Wang, D. Tuo, Z. Wu, S. Kang, and H. Meng, “Fullsubnet+: Channel attention fullsubnet with complex spectrograms for speech enhancement,” in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 7857–7861.

[8] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe, “Tf-gridnet: Integrating full- and sub-band modeling for speech separation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 3221–3236, 2023.

[9] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Proceedings ofthe 34th International Conference on Neural Information Processing Systems, ser. NIPS ’20. Red Hook, NY, USA: Curran Associates Inc., 2020.

[10] S. Welker, J. Richter, and T. Gerkmann, “Speech enhancement with score-based generative models in the complex stft domain,” arXiv preprint arXiv:2203.17004, 2022.

[11] J. Richter, S. Welker, J.-M. Lemercier, B. Lay, and T. Gerkmann, “Speech enhancement and dereverberation with diffusion-based generative models,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351–2364, 2023.

[12] A. Sivaraman and M. Kim, “Sparse mixture of local experts for efficient speech enhancement,” arXiv preprint arXiv:2005.08128, 2020.

[13] S. E. Chazan, J. Goldberger, and S. Gannot, “Speech enhancement with mixture of deep experts with clean clustering pre-training,” in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 716–720.

[14] S. Boll, “Suppression of acoustic noise in speech using spectral subtraction,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 27, no. 2, pp. 113–120, 1979.

[15] Y. Ephraim and D. Malah, “Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 32, no. 6, pp. 1109–1121, 1984.

[16] J. L. Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, “Sdr – half-baked or well done?” in ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 626–630.

[17] R. Chao, W.-H. Cheng, M. La Quatra, S. M. Siniscalchi, C.-H. H. Yang, S.-W. Fu, and Y. Tsao, “An investigation of incorporating mamba for speech enhancement,” in 2024 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2024, pp. 302–308.

[18] J. Wang, Z. Lin, T. Wang, M. Ge, L. Wang, and J. Dang, “Mamba-seunet: Mamba unet for monaural speech enhancement,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[19] H. Shi, K. Shimada, M. Hirano, T. Shibuya, Y. Koyama, Z. Zhong, S. Takahashi, T. Kawahara, and Y. Mitsufuji, “Diffusion-based speech enhancement with joint generative and predictive decoders,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 12 951–12 955.

[20] S. T. Yousif and B. M. Mahmmod, “Speech enhancement algorithms: A systematic literature review,” Algorithms, vol. 18, no. 5, p. 272, 2025.

[21] T. Chen, G.-H. Liu, and E. A. Theodorou, “Likelihood training of schrödinger bridge using forward–backward sdes theory,” in ICLR, 2022.

[22] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id= PxTIG12RRHS

[23] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

[24] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in International Conference on Learning Representations, 2023.

[25] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in International Conference on Learning Representations, 2023.

[26] V. De Bortoli, J. Thornton, J. Heng, and A. Doucet, “Diffusion Schrödinger bridge with applications to score-based generative modeling,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 17 695–17 709.

[27] Y. Shi, V. De Bortoli, A. Campbell, and A. Doucet, “Diffusion schrödinger bridge matching,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 62 183–62 223. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2023/file/ c428adf74782c2092d254329b6b02482-Paper-Conference.pdf

[28] A. Jukic, R. Korostik, J. Balam, and B. Ginsburg, “Schrödinger bridge for generative speech´ enhancement,” arXiv preprint arXiv:2407.16074, 2024.

[29] S. Wang, S. Liu, A. Harper, P. Kendrick, M. Salzmann, and M. Cernak, “Diffusion-based speech enhancement with Schrödinger bridge and symmetric noise schedule,” arXiv preprint arXiv:2409.05116, 2024.

[30] Z. Tang, T. Hang, S. Gu, D. Chen, and B. Guo, “Simplified diffusion schrödinger bridge,” arXiv preprint arXiv:2403.14623, 2024.

[31] J. Yang, S. Wang, C. Wu, L. Guo, and F. Fan, “Schrödinger bridge mamba for one-step speech enhancement,” arXiv preprint arXiv:2510.16834, 2025.

[32] H. Zhang, G. Li, P. Wu, Y. Gao, and H. Zhang, “Sb-senet: Diffusion model based on schrödinger bridge for speech enhancement,” Applied Acoustics, vol. 236, p. 110742, 2025.

[33] L. Xu, L. F. Yan, and W. B. Kleijn, “Robust one-step speech enhancement via consistency distillation,” arXiv preprint arXiv:2507.05688, 2025.

[34] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever, “Consistency models,” in International Conference on Machine Learning, 2023.

[35] S. Han, S. Lee, J. Lee, and K. Lee, “Few-step adversarial schrödinger bridge for generative speech enhancement,” arXiv preprint arXiv:2506.01460, 2025.

[36] S. Nishigori, K. Saito, N. Murata, M. Hirano, S. Takahashi, and Y. Mitsufuji, “Schrödinger bridge consistency trajectory models for speech enhancement,” arXiv preprint arXiv:2507.11925, 2025.

[37] S. Chen, S. Chewi, J. Li, Y. Li, A. Salim, and A. R. Zhang, “Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions,” in International Conference on Learning Representations, 2023.

[38] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” arXiv preprint arXiv:1701.06538, 2017.

[39] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal ofMachine Learning Research, vol. 23, no. 120, pp. 1–39, 2022. [Online]. Available: http://jmlr.org/papers/v23/21-0998.html

[40] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, “Gshard: Scaling giant models with conditional computation and automatic sharding,” arXiv preprint arXiv:2006.16668, 2020.

[41] A. Sivaraman and M. Kim, “Zero-shot personalized speech enhancement through speakerinformed model selection,” in 2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2021, pp. 171–175.

[42] R. Miccini, M. Kim, C. Laroche, L. Pezzarossa, and P. Smaragdis, “Adaptive slimming for scalable and efficient speech enhancement,” arXiv preprint arXiv:2507.04879, 2025.

[43] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[44] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proceedings of the 34th International Conference on Machine Learning - Volume 70, ser. ICML’17. JMLR.org, 2017, p. 1321–1330.

[45] J. B. Allen and L. R. Rabiner, “A unified approach to short-time fourier analysis and synthesis,” Proceedings of the IEEE, vol. 65, no. 11, pp. 1558–1564, 1977.

[46] J. Thiemann, N. Ito, and E. Vincent, “The diverse environments multi-channel acoustic noise database (demand): A database of multichannel environmental noise recordings,” Proceedings ofMeetings on Acoustics, vol. 19, no. 1, p. 035081, 05 2013. [Online]. Available: https://doi.org/10.1121/1.4799597

[47] Y. Wu and K. He, “Group normalization,” in Proceedings of the European Conference on Computer Vision, 2018, pp. 3–19.

[48] G. Peyre and M. Cuturi, “Computational optimal transport,” Foundations and Trends in Machine Learning, vol. 11, no. 5-6, pp. 355–607, 2019.

[49] A. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” 2021. [Online]. Available: https://arxiv.org/abs/2102.09672

[50] C. Valentini-Botinhao, X. Wang, S. Takaki, and J. Yamagishi, “Investigating rnn-based speech enhancement methods for noise-robust text-to-speech,” in 9th ISCA Workshop on Speech Synthesis Workshop (SSW 9), 2016, pp. 146–152.

[51] A. Rix, J. Beerends, M. Hollier, and A. Hekstra, “Perceptual evaluation of speech quality (pesq)-a new method for speech quality assessment of telephone networks and codecs,” in 2001 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings (Cat. No.01CH37221), vol. 2, 2001, pp. 749–752 vol.2.

[52] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “An algorithm for intelligibility prediction of time–frequency weighted noisy speech,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 7, pp. 2125–2136, 2011.

[53] Y. Hu and P. C. Loizou, “Evaluation of objective quality measures for speech enhancement,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 16, no. 1, pp. 229–238, 2008.

[54] N. Tishby and N. Zaslavsky, “Deep learning and the information bottleneck principle,” in 2015 IEEE Information Theory Workshop (ITW), 2015, pp. 1–5.

## A Heterogeneous Expert Architectures

This appendix details the heterogeneous MoE layer described in Section 3 of the main paper. We describe the design philosophy that connects each architectural archetype to a canonical noise family, list the configurations used for VoiceBank+DEMAND, and discuss how the archetype set generalizes to other datasets.

## A.1 Design Philosophy

The heterogeneous MoE is grounded in the empirical observation that real-world acoustic noise, despite its surface diversity, can be decomposed into a small number of canonical processing primitives. Stationary tonal noise (e.g., kitchen appliances, refrigerator hum) is well-modeled by compact normalized layers with low-rank structure; ambient natural backgrounds (e.g., park, river, field) benefit from wide receptive fields with strong normalization to capture diffuse spectral content; office and meeting environments contain overlapping voices and structured interference best handled by information bottlenecks [54]; transportation noise (bus, car, metro) exhibits strong harmonic and quasi-periodic energy from engines and motors, naturally captured by harmonic basis expansion; and unstructured public-space mixtures (cafeteria, station, restaurant) require universal-approximationstyle general-purpose blocks. Rather than scaling capacity by replicating one architecture (as in homogeneous MoE), we instantiate one expert per archetype, then rely on sparse routing to combine them per input.

Figure 4 (reproduced here for completeness) illustrates the shared backbone and the architecturally distinct instantiations. All experts pass features through LayerNorm → variable internal modules → bottleneck projection, ensuring interface compatibility for routing while permitting structurally divergent computation in the middle.

![](images/009708416bab8a97e5b2f4693932160ddc6cbaf61fd20588438c2ed5cc44e239.jpg)  
Figure 4: Heterogeneous expert architectures (the N=5 archetypes used for VoiceBank+DEMAND). Left: shared backbone with variable module slots. Right: example instantiations with architecturally distinct modules, each targeting a different noise-processing primitive (Home, Nature, Office, Transport, Public). Built on this compact set of canonical archetypes, sparse top-k=2 routing blends experts per input to span the  <sup>5</sup>=10-dimensional simplex of pairwise mixtures, yielding combinatorial coverage of the 14 noise types in VoiceBank+DEMAND without requiring a dedicated expert per environment.

Table 2: Heterogeneous expert architectures used for VoiceBank+DEMAND. The notation GN(g) denotes group normalization with g groups; LN denotes layer normalization. Each architecture is selected to match the inductive bias of its target scene family.
<table><tr><td>Scene Category</td><td>Architecture</td><td>Design Motivation</td><td>Params</td></tr><tr><td>Home (DKITCHEN, etc.)</td><td> $5 1 3  1 0 2 4  \mathrm { G N } ( 8 )  1 0 2 4  5 1 3$ </td><td>Low-rank denoising for stationary tonal noise</td><td>2.6M</td></tr><tr><td>Nature (NPARK, etc.)</td><td> $5 1 3 \to 2 0 4 8 \to 1 0 2 4 \to \mathrm { L N } \to 5 1 3$ </td><td>Wide receptive field for diffuse ambience</td><td>3.7M</td></tr><tr><td>Office (OMEETING, etc.)</td><td> $5 1 3  1 0 2 4  5 1 2  1 0 2 4  5 1 3$ </td><td>Information bottleneck [54]</td><td>2.6M</td></tr><tr><td>Transport (TBUS, etc.)</td><td> $5 1 3 \to 1 5 3 6 \to 1 0 2 4 \to 5 1 3$ </td><td>Harmonic basis expansion</td><td>3.2M</td></tr><tr><td>Public (PCAFETER, etc.)</td><td> $5 1 3  1 0 2 4  \mathrm { L N }  1 0 2 4  5 1 3$ </td><td>General-purpose universal approximator</td><td>2.6M</td></tr></table>

## A.2 Expert Configurations for VoiceBank+DEMAND

Table 2 lists the architectural specifications used in our VoiceBank+DEMAND experiments. Each archetype maps to one of the major noise families in the dataset; we instantiate one expert per archetype, yielding a compact yet diverse set of N = 5 processing primitives. The input dimension matches the STFT frequency-bin count $F = 5 1 3$ , and all experts produce outputs of the same dimension to maintain interface compatibility for routing.

## A.3 Archetype Details

Home expert. The kitchen, washing-machine, and living-room scenes share a common acoustic signature: stationary tonal noise from appliances overlaid on relatively clean speech. We instantiate a compact two-layer expansion (513 → 1024) followed by group normalization with 8 groups, which acts as a low-rank denoising operator. The narrow internal capacity is sufficient because the noise subspace is approximately stationary, allowing the expert to suppress predictable tonal patterns without overfitting.

Nature expert. Park, river, and field recordings contain diffuse, broadband ambient energy that fills wide spectral regions. We employ a wider hidden layer (513 → 2048 → 1024) followed by layer normalization to maintain stable activations across the expanded receptive field. This architecture trades parameter count for the spectral coverage required to model ambient acoustic textures.

Office expert. Meeting and office scenes feature overlapping voices and structured background chatter. We adopt an information-bottleneck design [54] (513 → 1024 → 512 → 1024 → 513) that explicitly compresses to a 512-dimensional representation in the middle, encouraging the expert to retain only task-relevant information and suppress competing voiced sources.

Transport expert. Bus, car, and metro noise is dominated by harmonic and quasi-periodic energy from engines, motors, and wheel-rail interactions. We use a wide-narrow-output structure (513 → 1536 → 1024 → 513) without intermediate normalization, allowing the expert to fit a learned harmonic basis at the first layer and refine it before projection. Skipping internal normalization preserves amplitude information critical for engine-noise spectral envelopes.

Public expert. Cafeteria, station, and restaurant scenes are the most challenging, combining nonstationary crowd noise, music, transient events, and reverberation. We adopt a symmetric architecture with a single LayerNorm in the middle (513 → 1024 → LN → 1024 → 513), which serves as a universal approximator for the heterogeneous mixture without strong inductive bias. Sparse routing allows this general-purpose expert to be combined with archetype-specific experts when partial structure is detectable in the input.

## A.4 Generalization to Other Datasets

The archetype set is intentionally modular: adding a new expert requires only conforming to the shared backbone interface (matching input/output dimensionality F). The auxiliary load-balancing loss (Eq. 2) automatically integrates new experts into the routing distribution without retraining the existing ones from scratch, with brief router fine-tuning recommended. For deployments encountering noise families not represented in the current set, for instance, reverberant cocktail-party recordings or industrial machinery noise, a corresponding archetype can be added (e.g., a dereverberation-oriented expert with longer temporal context) without altering the rest of the framework.

Table 3: Loss weights and bound parameters used in all VoiceBank+DEMAND experiments.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td> $\lambda _ { \mathrm { S B } }$  (overall SB)</td><td>1.0</td><td> $\lambda _ { \mathrm { a u x } }$  (load balance)</td><td>0.01</td></tr><tr><td> $\lambda _ { \mathrm { p a t h } }$  (path consistency)</td><td>0.1</td><td> $\lambda _ { \mathrm { c a l } }$  (calibration)</td><td>0.05</td></tr><tr><td> $\lambda _ { \mathrm { t r a j } }$  (trajectory)</td><td>0.05</td><td> $\lambda _ { I }$  (importance variance)</td><td>0.5</td></tr><tr><td> $M _ { \mathrm { m a x } }$  (mask ceiling)</td><td>5.0</td><td> $\lambda _ { L }$  (load variance)</td><td>0.5</td></tr><tr><td> $\phi _ { \mathrm { m a x } }$  (phase ceiling)</td><td> $\pi / 4$ </td><td></td><td></td></tr></table>

## B U-Net and Training Details

## B.1 Waveform-Domain U-Net Architecture

The denoiser ${ \hat { x } } _ { \theta }$ is a 1D U-Net with 4 encoder/decoder levels. Each encoder block applies a strided 1D convolution (stride 2, kernel 7) followed by GroupNorm and SiLU activation, halving temporal resolution and doubling channels $( 6 4  1 2 8 \stackrel { \cdot } {  } 2 5 6 \stackrel { \cdot } {  } 5 1 2 )$ ). The decoder mirrors this structure with transposed convolutions and skip connections. A latent bottleneck operates at the lowest resolution to aggregate global temporal context before decoding. The noisy observation $y$ is concatenated with $x _ { t }$ along the channel dimension at the input. Timestep t is encoded via a sinusoidal embedding (dim 128) projected through a 2-layer MLP $( 2 5 6  5 \bar { 1 2 } )$ , and injected into every block via FiLM scale-and-shift conditioning. The aleatoric uncertainty head $\sigma _ { \theta } ^ { 2 }$ is a separate 2-layer MLP attached to the final decoder feature, producing per-sample log-variance.

## B.2 Loss Weights

The full training objective combines six loss components $( \mathcal { L } _ { \mathrm { r e c } } , \mathcal { L } _ { \mathrm { S B } } ^ { \mathrm { d a t a } } , \mathcal { L } _ { \mathrm { p a t h } } , \mathcal { L } _ { \mathrm { t r a j } } , \mathcal { L } _ { \mathrm { a u x } } , \mathcal { L } _ { \mathrm { c a l } } )$ with the following weights, tuned via grid search on a 10% held-out validation split of Voice-Bank+DEMAND, as shown in Table 3.

## B.3 Training Schedule

We use AdamW with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , weight decay 0.01, and gradient clipping at norm 1.0. The learning rate follows cosine annealing from $2 \times 1 0 ^ { - 4 } \mathrm { ~ t o ~ } 1 \times 1 \bar { 0 } ^ { - 6 }$ over 200 epochs, with a 5-epoch linear warm-up. Batch size is 32 across 2 NVIDIA RTX 5090 GPUs (effective batch 64). Mixed-precision (bfloat16) training is used throughout. Total wall-clock training time: approximately 48 hours.

## B.4 SB Sampling Configuration

The cumulative cosine schedule follows $\bar { \beta } _ { t } = f ( t ) / f ( 0 )$ with $f ( t ) = \cos ^ { 2 } \bigl ( ( t / T + s ) / ( 1 + s ) \cdot \pi / 2 \bigr )$ and offset $s = 0 . 0 0 8 .$ , yielding the per-step ratio $\beta _ { t } = \bar { \beta } _ { t } / \bar { \beta } _ { t - 1 }$ . We adopt the convention that $\bar { \beta } _ { t }$ decreases monotonically from $\bar { \beta } _ { 0 } = 1$ to $\bar { \beta } _ { T } = 0 .$ , which keeps Eq. 4 symmetric in x and $y ;$ this assigns $\beta _ { t }$ the role played by $\bar { \alpha } _ { t }$ in the standard DDPM convention. The bridge perturbation in both the forward construction (Eq. 4) and the reverse update (Eq. 5) shares a single schedule $\sigma _ { t } = \sigma _ { \operatorname* { m a x } } \sqrt { \bar { \beta } } _ { t } ( 1 - \bar { \beta } _ { t } )$ with $\sigma _ { \mathrm { m a x } } = 0 . 0 5$ , which vanishes at both endpoints $( \sigma _ { 0 } = \sigma _ { T } = 0 )$ and matches the boundary conditions $x _ { 0 } = x , x _ { T } = y .$ . For inference, we use $K = 8$ sampling steps with the non-uniform schedule $t _ { k } = T ( k / K ) ^ { \gamma }$ where $\gamma = 0 . 6$ (front-loaded).

At each training step we sample a single timestep $t \sim \mathcal { U } [ 0 , T ]$ and supervise ${ \hat { x } } _ { \theta }$ via Eq. 6, while $\mathcal { L } _ { \mathrm { p a t h } }$ (Eq. 7) is evaluated on an independently sampled pair $( \dot { t } , t ^ { \prime } )$ with independent perturbations $\epsilon , \epsilon ^ { \prime }$ drawn from Eq. 4. Although inference uses a K-step rollout while training optimizes singlestep predictions, $\mathcal { L } _ { \mathrm { p a t h } }$ explicitly enforces that $\hat { x } _ { \theta } ( x _ { t } , y , t )$ remains consistent across timesteps, so the multi-step trajectory operates on predictions that the model has been trained to keep mutually agreeable. The trajectory regularizer $\mathcal { L } _ { \mathrm { t r a j } }$ further anchors each per-step prediction to the forward bridge construction, yielding a reverse process that is stable under the discretization mismatch between training and inference.

## B.5 Algorithm Details

The complete training procedure with explicit loss accumulation is given in Algorithm 1.

Algorithm 1 HybridSB-MoE end-to-end training (full version)   
Input: Paired training set $\{ ( \boldsymbol { y } ^ { ( i ) } , \boldsymbol { x } ^ { ( i ) } ) \} _ { i = 1 } ^ { M } ;$ experts $\{ E _ { i } \} _ { i = 1 } ^ { N } ;$ router $G ;$ SB denoiser ${ \hat { x } } _ { \theta } ;$ fusion   
network $F ;$ loss weights $\{ \lambda _ { * } \}$   
Output: Trained parameters Θ   
1: for each minibatch $\boldsymbol { B } = \{ ( y , x ) \}$ do   
2: // Spectral pathway   
3: Compute $\begin{array} { r } { \dot { \mathrm { S T F T } } \colon \dot { S ^ { \prime } } \lbrace y \rbrace = \mathrm { S T F T } ( y ) ; } \end{array}$ extract $z = \log | S \{ y \} |$   
4: Route: $G ( z )  \alpha \bar { G } _ { \mathrm { a r c h } } ( z ) + ( 1 - \alpha ) G _ { \mathrm { t o k e n } } ( z ) ;$ ; select top-k indices $\mathcal { T } _ { k }$   
5: Aggregate: $\begin{array} { r } { \hat { x } _ { \mathrm { s p e c } }  \sum _ { i \in \mathcal { T } _ { k } } G _ { i } ( z ) E _ { i } ( z ) } \end{array}$   
6: Predict $\hat { M } , \Delta \phi$ via Eq. 3; reconstruct $x _ { \mathrm { s p e c } }$ via iSTFT   
7: Compute $u _ { \mathrm { e p i } }$ (variance of top-k expert outputs)   
8: // Waveform pathway   
9: Sample two independent timesteps $t , t ^ { \prime } \sim \mathcal { U } [ 0 , T ]$ and noises $\epsilon , \epsilon ^ { \prime } \sim \mathcal { N } ( 0 , I )$ // single Monte   
Carlo sample of $\cdot \frac { \mathbf { \widehat { E } } } { \mathbb { E } _ { t , t ^ { \prime } } }$ per minibatch   
10: Form bridge states: $x _ { t } \gets \sqrt { \bar { \beta } _ { t } } x + \sqrt { 1 - \bar { \beta } _ { t } } y + \sigma _ { t } \epsilon ;$   
$x _ { t ^ { \prime } } \gets \sqrt { \bar { \beta } _ { t ^ { \prime } } } x + \sqrt { 1 - \bar { \beta } _ { t ^ { \prime } } } y + \sigma _ { t ^ { \prime } } \epsilon ^ { \prime }$   
11: Predict $\hat { x } _ { \theta } ( x _ { t } , y , t ) , \hat { x } _ { \theta } ( x _ { t ^ { \prime } } , y , t ^ { \prime } )$ , and aleatoric variance $\sigma _ { \theta } ^ { 2 }$ at $( x _ { t } , t )$   
12: Compute $\mathcal { L } _ { \mathrm { S B } } ^ { \mathrm { d a t a } } \left( \mathrm { E q } . 6 , \right.$ on $( x _ { t } , t ) ) ; \mathcal { L } _ { \mathrm { p a t h } } = \Vert \hat { x } _ { \theta } ( x _ { t } , y , t ) - \hat { x } _ { \theta } ( x _ { t ^ { \prime } } , y , t ^ { \prime } ) \Vert ^ { 2 } \ ( \mathrm { E q . ~ } 7 ) ; \mathcal { L } _ { \mathrm { t r a j } }$ (Eq.   
8, on $( x _ { t } , t ) )$   
13: Set $u _ { \mathrm { a l e } } \gets \overline { { \exp ( \sigma _ { \theta } ^ { 2 } ) } } _ { t } ;$ set $x _ { \mathrm { w a v e } } \gets \hat { x } _ { \theta } ( x _ { t } , y , t )$ as the current single-step prediction (full   
K-step rollout used at inference)   
14: // Fusion   
15: $\tilde { u } _ { \mathrm { e p i } } , \tilde { u } _ { \mathrm { a l e } } \gets \mathrm { z - n o r m } ( u _ { \mathrm { e p i } } , u _ { \mathrm { a l e } } )$ // running statistics   
16: $w \dot { \langle } - \sigma ( \mathrm { M L P } ( \tilde { u } _ { \mathrm { e p i } } , \tilde { u } _ { \mathrm { a l e } } ) )$   
17: $\hat { x }  w \cdot x _ { \mathrm { s p e c } } + ( 1 - w ) \cdot x _ { \mathrm { w a v e } }$   
18: Compute $\dot { \mathcal { L } } _ { \mathrm { { r e c } } } = \| \hat { x } - x \| _ { 2 } ^ { 2 } , \mathcal { L } _ { \mathrm { { a u x } } } ( \mathrm { { E q . } } 2 ) , \mathcal { L } _ { \mathrm { { c a l } } } ( { \mathrm { E q . } } 1 1 )$ $/ / \mathcal { L } _ { \mathrm { c a l } }$ uses pre-norm $u _ { \mathrm { e p i } } , u _ { \mathrm { a l e } }$   
19: // Joint update   
20: ${ \mathcal { L } } _ { \mathrm { S B } } \gets { \dot { \mathcal { L } } } _ { \mathrm { S B } } ^ { \mathrm { d a t a } } + \lambda _ { \mathrm { p a t h } } { \mathcal { L } } _ { \mathrm { p a t h } } + \lambda _ { \mathrm { t r a j } } { \mathcal { L } } _ { \mathrm { t r a j } }$   
21: $\mathcal { L } \gets \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { S B } } \hat { \mathcal { L } } _ { \mathrm { S B } } + \dot { \lambda } _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { a u x } } + \lambda _ { \mathrm { c a l } } \mathcal { L } _ { \mathrm { c a l } }$   
22: Update $\Theta  \Theta - \eta \nabla _ { \Theta } \mathcal { L }$   
23: end for

## C Proof of Theorem 1

We prove the K-step discretization bound stated in Section 3 (Theorem 1) of the main paper. The strategy is to decompose the $W _ { 2 }$ distance into (i) a one-step model-fidelity error driven by $\bar { \mathcal { L } _ { \mathrm { { p a t h } } } ^ { \star } }$ and $\mathcal { L } _ { \mathrm { t r a j } } ^ { \star }$ , and (ii) a Riemann-sum discretization error in K controlled by the schedule’s front-loading exponent $\gamma _ { : }$ , then to combine the two via a synchronous coupling and a triangle inequality. The argument adapts standard SDE-discretization analysis (cf. [37, 26]) to the doubly-conditioned bridge process of Eq. (4–5).

## C.1 Setup and Notation

Fix a noisy observation y and let $\{ x _ { t } \} _ { t \in [ 0 , T ] }$ be the true continuous-time bridge process from clean speech $x$ to y, so that $p _ { t } ^ { \mathrm { b r } } ( \cdot \ | \ y )$ is its time-t marginal. Let ${ \hat { x } } _ { \theta }$ be the trained denoiser, and let $\hat { p } _ { K }$ be the law at $t { = } 0$ of the K-step rollout produced by Eq. (5) initialized at $x _ { T } ~ = ~ y$ along the non-uniform schedule $\{ t _ { k } \} _ { k = 0 } ^ { K }$ with $t _ { K } = \bar { T }$ and $t _ { k } \doteq T \bar { ( k / K ) } ^ { \gamma } , \gamma \in ( 0 , 1 ]$ ]. We use $\begin{array} { r } { W _ { 2 } ( \mu , \nu ) ^ { 2 } = \operatorname* { i n f } _ { \pi \in \Gamma ( \mu , \nu ) } \int \| u - v \| ^ { 2 } \dot { d } \pi ( \dot { u } , v ) } \end{array}$ in the standard definition.

Throughout, C denotes a generic constant depending only on the regularity constants of Assumption 1; its value may change line by line.

## C.2 One-step Model-fidelity Lemma

Lemma 1 (Model-fidelity error). Under Assumption 1, for each step $t _ { k }  t _ { k - 1 }$ of Eq. (5), the conditional one-step error

$$
\delta _ { k } : = \mathbb { E } \Big [ \big \| \hat { x } _ { \boldsymbol { \theta } } ( x _ { t _ { k } } , y , t _ { k } ) - x \big \| ^ { 2 } \big | x _ { t _ { k } } , y \Big ]
$$

satisfies $\mathbb { E } [ \delta _ { k } ] \leq C \left( \mathcal { L } _ { \mathrm { p a t h } } ^ { \star } + \mathcal { L } _ { \mathrm { t r a j } } ^ { \star } \right)$ uniformly in $k ,$ where the expectation on the right is over the bridge construction Eq. (4).

Proof. The trajectory regularizer $\mathcal { L } _ { \mathrm { t r a j } }$ of Eq. (8), evaluated at the optimum, satisfies

$$
\mathbb { E } \bigg [ \big \| x _ { t } - \big ( \sqrt { \bar { \beta } _ { t } } \hat { x } _ { \theta } \big ( x _ { t } , y , t \big ) + \sqrt { 1 - \bar { \beta } _ { t } } y \big ) \big \| ^ { 2 } \bigg ] = \mathcal { L } _ { \mathrm { t r a j } } ^ { \star } .
$$

Substituting the forward construction $x _ { t } = \sqrt { \bar { \beta } _ { t } } x + \sqrt { 1 - \bar { \beta } _ { t } } y + \sigma _ { t } \epsilon$ from Eq. (4) and rearranging,

$$
\begin{array} { r } { \bar { \beta } _ { t } \mathbb { E } \big [ \| \hat { x } _ { \theta } ( x _ { t } , y , t ) - x \| ^ { 2 } \big ] - 2 \sqrt { \bar { \beta } _ { t } } \sigma _ { t } \mathbb { E } [ \langle \hat { x } _ { \theta } ( x _ { t } , y , t ) - x , \epsilon \rangle ] + \sigma _ { t } ^ { 2 } \mathbb { E } \| \epsilon \| ^ { 2 } = \mathcal { L } _ { \mathrm { t r a j } } ^ { \star } . } \end{array}
$$

By Cauchy–Schwarz and $\mathbb { E } \Vert \epsilon \Vert ^ { 2 } ~ = ~ d$ (the signal dimension), the cross term is bounded by $2 \sqrt { d } \sigma _ { t } \sqrt { \mathbb { E } \| \hat { x } _ { \theta } - x \| ^ { 2 } }$ , and bounded marginals (Assumption 1(iii)) yield

$$
\mathbb { E } \big [ \| \hat { x } _ { \theta } ( x _ { t } , y , t ) - x \| ^ { 2 } \big ] \leq \frac { 1 } { \bar { \beta } _ { t } } \big ( \mathcal { L } _ { \mathrm { t r a j } } ^ { \star } + C \sigma _ { t } ^ { 2 } \big ) \leq C ( \mathcal { L } _ { \mathrm { t r a j } } ^ { \star } + \sigma _ { \mathrm { m a x } } ^ { 2 } ) ,\tag{13}
$$

on $[ \epsilon _ { 0 } , T - \epsilon _ { 0 } ]$ for any fixed $\epsilon _ { 0 } > 0 \ ,$ , since $\bar { \beta } _ { t }$ is bounded away from zero on this set; near the boundaries, the schedule’s vanishing $\sigma _ { t }$ controls the residual.

The path-consistency loss $\mathcal { L } _ { \mathrm { p a t h } }$ of Eq. (7) controls the deviation of ${ \hat { x } } _ { \theta }$ across timesteps:

$$
\mathbb { E } \big [ \| \hat { x } _ { \theta } ( x _ { t } , y , t ) - \hat { x } _ { \theta } ( x _ { t ^ { \prime } } , y , t ^ { \prime } ) \| ^ { 2 } \big ] \leq 2 \mathcal { L } _ { \mathrm { p a t h } } ^ { \star } .\tag{14}
$$

Combining Eq. (13) with Eq. (14) and the triangle inequality on the implied error gives the claimed bound on $\bar { \mathbb { E } } [ \delta _ { k } ]$

## C.3 Discretization Lemma

Lemma 2 (Riemann discretization rate). Let $\Phi ( t ) = \sqrt { \bar { \beta } _ { t } } x + \sqrt { 1 - \bar { \beta } _ { t } }$ y denote the deterministic component of the bridge state at time $t \left( E q . \right.$ . 4 with $\sigma _ { t } \epsilon$ removed). Under Assumption $^ { l , }$ the K-step Riemann approximation $\Phi _ { K }$ of Φ along the non-uniform schedule $t _ { k } = T ( \dot { k } / K ) ^ { \gamma } , \gamma \in ( 0 , \dot { 1 } ]$ satisfies

$$
\operatorname* { s u p } _ { k } \left| \Phi ( t _ { k - 1 } ) - \Phi _ { K } ( t _ { k - 1 } ) \right| \ \leq \ C K ^ { - \operatorname* { m i n } ( 1 , \gamma ) } .
$$

Proof. Standard mean-value estimates on $C ^ { 1 }$ schedules yield local error $| \Phi ( t _ { k - 1 } ) - \Phi _ { K } ( t _ { k - 1 } ) | \leq$ $C \left( \dot { \Delta t } _ { k } \right) \left\| \Phi ^ { \prime } \right\| _ { \infty } \mathrm { o n } \left[ t _ { k - 1 } , t _ { k } \right]$ , where $\Delta t _ { k } = t _ { k } - t _ { k - 1 }$ . For the schedule $t _ { k } = T ( k / K ) ^ { \gamma }$

$$
\Delta t _ { k } = T \big ( ( k / K ) ^ { \gamma } - ( ( k - 1 ) / K ) ^ { \gamma } \big ) \leq T \gamma K ^ { - \gamma } k ^ { \gamma - 1 } .
$$

For $\gamma \leq 1 , k ^ { \gamma - 1 }$ is non-increasing in $k ,$ so $\begin{array} { r } { \operatorname* { s u p } _ { k } \Delta t _ { k } = \Delta t _ { 1 } \le T K ^ { - \gamma } } \end{array}$ , giving the stated bound with exponent min $( 1 , \gamma ) = \gamma$ □

Remark. Front-loading $( \gamma { < } 1 )$ does not improve the worst-case rate exponent. However, since $\Delta t _ { k } \le T \gamma K ^ { - \gamma } k ^ { \gamma - 1 }$ is non-increasing in $k$ for $\gamma \leq 1$ , the schedule concentrates many small steps near $t _ { K } = T$ , where the reverse rollout of $\mathrm { E q . } 5$ is initialized at $x _ { T } = y$ and where per-step errors propagate through all subsequent reverse-sampling iterations. The single large step $\Delta t _ { 1 }$ , by contrast, traverses the low-t regime in which $x _ { t }$ already lies close to x and ${ \hat { x } } _ { \theta }$ is most accurate. This redistribution—not the asymptotic rate—is what empirically dominates small-K reconstruction quality.

## C.4 Combining the Two Sources of Error

Let $\hat { x } ^ { ( K ) }$ denote the random variable whose law is $\hat { p } _ { K }$ , and let $X \sim p _ { 0 } ^ { \mathrm { b r } }$ . We construct a synchronous coupling between $\hat { x } ^ { ( K ) }$ and X by sharing the noise variables z across the K rollout steps with the corresponding bridge perturbation ϵ in the continuous process. Under this coupling,

$$
W _ { 2 } ^ { 2 } ( \hat { p } _ { K } , p _ { 0 } ^ { \mathrm { b r } } ) \leq \mathbb { E } \Big [ \| \hat { x } ^ { ( K ) } - X \| ^ { 2 } \Big ] \leq 2 \sum _ { k = 1 } ^ { K } \Delta t _ { k } \mathbb { E } [ \delta _ { k } ] \cdot \prod _ { j < k } ( 1 + L _ { x } \Delta t _ { j } ) ^ { 2 } + 2 \sum _ { k = 1 } ^ { K } ( \Delta t _ { k } \cdot \mathrm { d i s c } _ { k } ) ^ { 2 } ,
$$

where $\mathrm { d i s c } _ { k }$ is the Riemann error from Lemma 2; the $\Delta t _ { k }$ weighting on the model-fidelity term tracks the time-averaged definition of $\mathcal { L } _ { \mathrm { t r a j } } ^ { \star } , \mathcal { L } _ { \mathrm { p a t h } } ^ { \star } \left( \mathrm { E q s . ~ } 8 , 7 \right)$ , so each step contributes proportionally to its time interval. The Lipschitz factor $\dot { \prod _ { j } } ( \dot { 1 } + L _ { x } \Delta t _ { j } )$ is uniformly bounded by exp $( L _ { x } T )$ , absorbing it into the constant. By Lemma 1 and $\sum _ { k } \Delta t _ { k } = T$ , the first sum is bounded by $C T ( \mathcal { L } _ { \mathrm { p a t h } } ^ { \star } + \mathcal { L } _ { \mathrm { t r a j } } ^ { \star } )$ independent of K. By Lemma 2 together with $\begin{array} { r } { \sum _ { k } ( \Delta t _ { k } ) ^ { 2 } \le T \operatorname* { s u p } _ { k } \Delta t _ { k } = T ^ { 2 } K ^ { - \gamma } } \end{array}$ , the second sum is bounded by $C K ^ { - 2 \operatorname* { m i n } ( 1 , \gamma ) }$ . Taking square roots and applying ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } }$ yields

$$
\begin{array} { r l r } {  { W _ { 2 } ( \hat { p } _ { K } , p _ { 0 } ^ { \mathrm { b r } } ) \ \leq \ C _ { 1 } K ^ { - \alpha } + C _ { 2 } \sqrt { { \mathscr L } _ { \mathrm { p a t h } } ^ { \star } + { \mathscr L } _ { \mathrm { t r a j } } ^ { \star } } } , } \end{array}
$$

which is exactly Eq. (9).

## C.5 Justification of Assumption 1

We address each of the three regularity conditions in turn and explain why each is reasonable in our setting.

(i) Lipschitz continuity of ${ \hat { x } } _ { \theta }$ . The condition that $\hat { x } _ { \theta } ( \cdot , y , t )$ is $L _ { x } – \mathbf { I }$ ipschitz in its first argument and $L _ { t } { \mathrm { - L i p s c h i t z } }$ in t is standard in diffusion-sampling analyses [37, 26]. It is justified architecturally and empirically: (a) our denoiser is a 1D U-Net with GroupNorm [47] and SiLU activations, both Lipschitz; (b) the convolutional layers’ Lipschitz constants are bounded since weights remain finite under our AdamW training with weight decay 0.01 and gradient clipping at norm 1.0 (Appendix B); (c) timestep dependence enters only through a smooth sinusoidal embedding followed by FiLM conditioning, which is $C ^ { \infty }$ in t and so trivially Lipschitz on the compact interval $[ 0 , T ]$ . We do not enforce a target $L _ { x }$ explicitly (e.g., via spectral normalization); the assumption asserts that the trained network is Lipschitz with some finite constant, which is automatic for any network without unbounded activations or singular layers. Tighter constant estimates—needed only for quantitatively tight versions of the bound—would require spectral norm tracking and are left to future work.

(ii) Smoothness of the schedules. The cosine schedule $\bar { \beta } _ { t } = f ( t ) / f ( 0 )$ with $f ( t ) = \cos ^ { 2 } ( ( t / T +$ $s ) / ( 1 + s ) \cdot \pi / 2 )$ is in fact $C ^ { \infty }$ on [0, T], considerably stronger than the $C ^ { 1 }$ requirement: it is the composition of a smooth trigonometric function with an affine reparameterization. Its monotonicity follows from $\cos ^ { 2 }$ being strictly decreasing on the relevant subinterval, and its first and second derivatives are uniformly bounded on [0, T] since the schedule arguments stay in a compact subset of $[ 0 , \pi / 2 ]$ where cos and sin are bounded. The bridge-perturbation magnitude $\sigma _ { t } = \sigma _ { \operatorname* { m a x } } \sqrt { \bar { \beta } _ { t } ( 1 - \bar { \beta } _ { t } ) }$ is $C ^ { \infty }$ on the open interval $( 0 , T )$ and continuous on $[ 0 , T ]$ with $\sigma _ { 0 } = \sigma _ { T } = 0$ ; the square-root introduces a one-sided Hölder singularity exactly at the endpoints, but standard SDE-discretization arguments (e.g., [37]) accommodate this via a small endpoint truncation $[ \epsilon _ { 0 } , T - \epsilon _ { 0 } ]$ that vanishes in the bound’s leading order. In our setting all sampling steps lie strictly in the interior of [0, T] with $t _ { K } = T$ at the noisy endpoint where the bridge is deterministic, so endpoint singularities do not affect the discretization.

(iii) Bounded second moments. The condition $\mathbb { E } \| x _ { t } \| ^ { 2 } < \infty$ uniformly in t follows directly from the bridge construction $x _ { t } = \sqrt { \bar { \beta } _ { t } } x + \sqrt { 1 - \bar { \beta } _ { t } } y + \sigma _ { t } \epsilon \mathrm { . }$ by Jensen’s inequality and the triangle inequality,

$$
\begin{array} { r } { \mathbb { E } \| x _ { t } \| ^ { 2 } \leq 3 \mathbb { E } \| x \| ^ { 2 } + 3 \mathbb { E } \| y \| ^ { 2 } + 3 \sigma _ { \operatorname* { m a x } } ^ { 2 } d , } \end{array}
$$

where d is the signal dimension and the right-hand side is finite because audio waveforms are bounded $( \| x \| _ { \infty } , \| y \| _ { \infty } \leq 1$ after our standard amplitude normalization, so $\mathbb { E } \Vert x \Vert ^ { 2 } \leq$ d and similarly for $y )$ For $\mathbb { E } \| \hat { x } _ { \theta } ( x _ { t } , y , t ) \| ^ { 2 }$ , the Lipschitz condition (i) gives $\| \hat { x } _ { \theta } ( x _ { t } , y , t ) \| \dot { \le } L _ { x } \| x _ { t } \| + | \hat { x } _ { \theta } ( 0 , \dot { y } , t ) |$ |, so finiteness of the second moment of ${ \hat { x } } _ { \theta }$ follows from finiteness of $\mathbb { E } \Vert \dot { \boldsymbol { x } _ { t } } \Vert ^ { 2 }$ together with boundedness of ${ \hat { x } } _ { \theta }$ at the reference input $x _ { t } = 0$ (a finite-norm constant for any fixed trained network on a fixed observation y). Both moments are therefore bounded by a constant depending only on $L _ { x } , \sigma _ { \operatorname* { m a x } } , d ,$ and the amplitude bound on the data.

![](images/b9e0d0bdbc4a8bfbcbf0349fb00595888bed2c30aece7bf6c85c1e672b5fad34.jpg)  
Figure 5: Scene-stratified performance across all 14 noise types in VoiceBank+DEMAND. Inner ring: PESQ; outer ring: STOI (%). Colors indicate scene categories: Domestic (blue), Nature (green), Office (orange), Public (red), Transport (purple). For reference, the strongest baseline ROSE-CD [33] achieves a mean PESQ of 3.85 across these conditions; our method exceeds this in every scene, with PESQ ranging from 3.84 to 3.92 (mean 3.88, standard deviation <0.03 across scenes). Y-axi ranges are zoomed for visibility of cross-scene variation.

In summary, Assumption 1 encodes mild structural properties that hold automatically for our cosinescheduled SB construction with a standard U-Net denoiser trained on amplitude-normalized waveforms. The assumption fails only in pathological regimes (unbounded data, divergent network outputs, or non-smooth schedules) that are not relevant to our setting or to standard SE practice.

## C.6 Discussion

A few caveats deserve mention. First, Assumption 1(i) is a Lipschitz condition on the trained network; it can be enforced via spectral normalization in practice, but in our experiments it is assumed rather than explicitly imposed. Second, the bound is presented as a design-justifying inequality; it is loose in the sense that the constants $C _ { 1 } , C _ { 2 }$ involve worst-case Lipschitz factors and are not numerically estimated here. A tighter quantitative match to empirical PESQ-vs-K would require finer Lipschitz estimates on the U-Net and is left for future work. Third, the bound shows that small K is justified conditional on the regularizers reaching small training-time values; if a model is trained without these terms, the second term of Eq. (9) does not vanish and the bound cannot guarantee small-K quality. This is consistent with the role we ascribe to $\mathcal { L } _ { \mathrm { p a t h } } + \mathcal { L } _ { \mathrm { t r a j } }$ in the main paper.

## D Scene-Stratified Performance

Figure 5 breaks down enhancement performance across all 14 noise environments. Natural environments (NPARK, NRIVER, NFIELD) achieve the highest scores due to relatively stable spectral characteristics, while public spaces (PCAFETER, PSTATION, PRESTO) are slightly more challenging due to non-stationary crowd noise and unpredictable transient events. Despite this variation, the spread is small (PESQ standard deviation <0.03), indicating that the heterogeneous expert routing automatically adapts to different acoustic contexts without per-scene tuning or explicit scene labels. This consistent cross-scene performance complements the SNR-stratified analysis in Figure 3d of the main paper, together demonstrating robust enhancement across both noise type and noise level.

## E Qualitative Visualization

Figure 6 shows representative spectrograms before and after enhancement. The denoised output exhibits clearer harmonic structures, sharper formant contours, and significantly attenuated broadband noise, consistent with the objective gains reported in Table 1. Note that some high-frequency components in the enhanced spectrogram appear more pronounced than in the noisy input; this is because heavy broadband noise in the input masks underlying speech harmonics that become visible only after suppression—it reflects accurate harmonic recovery, not bandwidth extension or content fabrication.

![](images/e064b17cbbe54c8b3ff7405a6db6fab944a6b63624d95e0fec766c270a8f2250.jpg)  
Figure 6: Spectrogram comparison before and after enhancement. HybridSB-MoE attenuates broadband noise while preserving speech harmonics and formant structure. Color bar (dB) is shown on the right.
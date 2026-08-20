# Denoising-Aware Inversion: Revealing Privacy Risks in Noise-Protected Text Embeddings

Yubo Wang<sup>1</sup>, Shujie Cui<sup>1</sup>, James Bailey<sup>1</sup>, Hongzhi Yin<sup>2</sup>,

Wenyu Liang<sup>1</sup>, Min Tang<sup>1</sup>, Shiyue Qin<sup>3</sup>, Weiqing Wang<sup>1</sup>

<sup>1</sup>Monash University, Melbourne, Australia

<sup>2</sup>The University of Queensland, Brisbane, Australia

<sup>3</sup>Northeastern University, Shenyang, China

{yubo.wang2, shujie.cui, james.a.bailey, Teresa.Wang}@monash.edu

wlia0047@student.monash.edu, bupttm@163.com, h.yin1@uq.edu.au, qinsy@mail.neu.edu.cn

Abstract—Dense text embeddings are widely used in data mining, retrieval, and downstream machine learning systems due to their compact and semantically rich representations, but recent embedding inversion attacks have shown that they can expose substantial information about the original text, leading to serious privacy leakage risks. A common defense is to release perturbed embeddings by adding Gaussian noise, which is simple yet effective against standard inversion attacks and does not significantly degrade embedding utility for downstream tasks. However, it remains unclear whether such noise-protected embeddings are sufficiently safe against adaptive attackers that explicitly account for the perturbation process. In this paper, we study text embedding inversion in a noise-protected setting, where the attacker can observe only noisy embeddings and has no access to clean embedding targets. We first analyze why existing generative inversion methods fail under this setting and identify a “Double Noise Trap”, which fundamentally prevents standard generative inversion models from achieving high-quality reconstruction. To address this challenge, we propose DAEI, a denoising-aware embedding inversion pipeline that combines a residual denoising autoencoder with generative text inversion where the denoiser is trained in an unsupervised manner using Stein’s unbiased risk estimate to enable denoising from noisy observations alone. Extensive experiments show that DAEI achieves approximately 154% relative improvement in BLEU over the existing generative inversion baseline, while also improving token-level F1 and ROUGE-L by 32–60%. The promising inversion performance of DAEI challenges the prevailing assumption that simple Gaussian perturbation is sufficient to prevent sensitive information leakage from embedding representations. The code for reproduction is provided below: https://github.com/ChrisWang233/DAEI

Index Terms—Machine Learning, Dense Text Embeddings, Embedding Inversion Attack, Privacy Leakage, Denoising Autoencoder

## I. INTRODUCTION

Dense text embeddings have become a standard data representation in modern data mining and machine learning systems [1]–[3]. Across applications such as semantic search, user modeling, vector databases, and retrieval-augmented analytics, raw textual records are increasingly transformed into compact continuous vectors that are stored, searched, or used for various tasks by downstream systems [4]–[6]. This representation is attractive because it transforms raw data into lowdimensional and semantically rich vector representations, making subsequent tasks more efficient [1], [5]. Yet these benefits create a potential privacy issue: high-quality embeddings are explicitly trained to preserve semantic and lexical information, and may therefore also preserve sensitive entities and attributes [7], [8]. Consequently, any leakage allows adversaries to apply sophisticated extraction techniques to recover the underlying sensitive data, resulting in a privacy vulnerability.

Recent works have made this privacy risk concrete. Generative embedding inversion attacks reconstruct full sentences from sentence embeddings, rather than simply recovering unordered keywords [8]. Vec2Text further frames inversion as controlled generation by introducing an iterative correction framework, substantially improving reconstruction accuracy for short inputs [9]. In practice, embeddings may be exposed through compromised vector databases, honest-but-curious storage or retrieval providers, and distributed systems that transmit embeddings online across components where the adversaries can attack the embedding during transmission [10], [11]. Once an adversary can observe these embeddings, the embeddings themselves may become a text disclosure channel.

One natural defense strategy is to perturb embeddings before release. Differential privacy formalizes the broader principle of adding calibrated noise to sensitive computations [12], and prior work on privacy-preserving textual analysis has shown that perturbations in embedding spaces can maintain downstream utility while reducing disclosure risks in data mining workflows [13]–[15]. The most prevalent defense mechanism is additive Gaussian noise injection [9], [16], [17]. In practice, adding zero-mean Gaussian noise with standard deviation σ = 0.01 to the final representations of the target embedder can severely degrade generative inversion performance while barely affecting downstream task utility [9]. However, as far as we know, there are no existing embedding inversion methods studying whether embeddingto-text inversion remains possible under this noise-protected setting. The closest related work is ZSInvert [18], which focuses on zero-shot inversion instead of focusing on noiseprotected setting. We consider it the closest work because it includes an experimental case study to show that the proposed approach is noise robust [18]. The possible reason is that its search-based optimization offers some robustness to noisy embeddings, but the recovered texts are often less readable and weakly token-aligned, which is also validated in our experiments. Therefore, we include it as a noise-robust baseline, while noting that conventional generative inversion methods remain largely ineffective under the noise-protected embedding setting.

This difficulty goes beyond a harder inversion task: noise creates a fundamentally different and underexplored inversion paradigm. In this setting, adversaries are deprived of clean embeddings and can observe only noisy vectors. This makes inversion substantially more challenging for two reasons: first, the absence of clean targets invalidates standard supervised denoising objectives that rely on reconstruction error against clean references; second, the inverse mapping must disentangle the underlying semantic signal from high-dimensional noise before text can be reliably recovered. Consequently, these constraints motivate the central question of our work: can adversaries recover sufficient textual information from blackbox noisy embeddings without relying on clean embeddings?

We investigate this question from a security-evaluation perspective. Specifically, we study the no-clean-embedding regime and propose DAEI: Denoising-Aware Embedding Inversion, an integrated pipeline for measuring the privacy risk of noise-protected text embeddings. Building upon Vec2Text as our generative backbone for text recovery, we prepend a residual denoising autoencoder (DAE) [19] to the inversion decoder to separate semantic signals from high-dimensional noise. A conventional DAE typically relies on a Mean Squared Error (MSE) objective against clean reference embeddings, which are unavailable in our noisy-only setting [19], [20]. Under the Gaussian noise model, we therefore use Stein’s unbiased risk estimate (SURE) [21], [22] to remove this reliance on clean embeddings, yielding an unbiased risk estimate computable from noisy observations alone. We further use Monte-Carlo Hutchinson estimation [22] to approximate the required divergence term with an explicit variance-based error bound. To align denoising with text generation, we jointly fine-tune both modules. This coupled optimization guides the DAE to preserve critical semantic directions, while dynamically adapting the inverter to the refined embedding space. Furthermore, unlike methods such as Noise2Noise [23] that require multiple noisy observations of each sample which are usually unavailable in real world applications, DAEI operates with only one noisy observation per sample.

In summary, our main contributions are as follows:

• We challenge the prevailing assumption that noise guarantees text embedding privacy. We are the first to formulate and systematically evaluate an adaptive “denoisinginversion” approach. By demonstrating that underlying textual information can still be reliably recovered from noise-protected embeddings, we provide a critical reevaluation of current representation security standards.

• We systematically analyze the failure of existing generative inversion approach in reconstructing text from noisy embeddings and introduce “Double Noise Trap” (Section II-B).

![](images/74599b886e20ed393e28a6e2883a886e23ffbb9b5d015bbe79827aa861e18f1f.jpg)  
Fig. 1: Geometric intuition of the “Double Noise Trap”. Additive Gaussian noise pushes the target embedding off the clean text manifold, causing vanilla inversion to chase off-manifold directions. DAEI instead projects the noisy representation back toward the clean manifold before inversion.

• To the best of our knowledge, we are the first to study denoising autoencoders for noisy text embedding inversion under a strict no-clean-target setting.

• We propose DAEI, a novel denoising-aware inversion pipeline that couples DAE pre-training, DAE-shielded inversion, components joint fine-tuning, and DAE-shielded correction, enabling robust text recovery from noisy observations.

• We validate DAEI across multiple benchmarks and embedding backbones. Compared with noisy Vec2Text, DAEI achieves approximately 154% relative improvement in BLEU and 32–60% gains in F1 and ROUGE-L, while also substantially outperforming ZSInvert.

## II. PROBLEM FORMULATION AND MOTIVATION

## A. Problem Formulation

Let $x \in \mathcal { X }$ denote an input text and let $f _ { \mathrm { c l e a n } } : \mathcal { X } \xrightarrow { } \mathbb { R } ^ { d }$ be a frozen black-box text encoder. In the standard setting, the exact latent representation is given by

$$
z = f _ { \mathrm { c l e a n } } ( x ) ,\tag{1}
$$

We study a privacy-preserving regime where the system releases only a perturbed version of the final embedding, denoted as y:

$$
y = f ( x ) = z + \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } ) .\tag{2}
$$

where $f$ denotes a noisy embedder, and the standard deviation of noise scale σ is assumed to be known or estimable, but the clean embedding z remains strictly inaccessible to the attacker.

Adversary assumption: In our threat model, we assume that the adversary has no knowledge of the embedder such as architecture and parameters. 1) Adversary can only query the released embedding interface as a black box and obtain an output embedding for a given attacker-chosen input text, thereby constructing noisy embedding–text training pairs $( y , x ) ; 2 )$ ) The adversary is assumed to know that the perturbation is additive Gaussian noise, since it is a standard and widely adopted mechanism for protecting text embedding; 3) The adversary either knows the noise scale σ, as prior work commonly adopts $\sigma = 0 . 0 1$ to achieve the best tradeoff between effectiveness and privacy [9]; or can estimate σ through repeated calibration queries on any adversary-chosen input text by observing the variation of the returned embeddings. Specifically, for n repeated observations of the same calibration text, $\begin{array} { r } { \{ y ^ { ( j ) } = } \end{array}$ $z + \epsilon ^ { ( j ) } \} _ { j = 1 } ^ { n }$ , an unbiased estimator of the noise variance is

$$
\hat { \sigma } ^ { 2 } = \frac { 1 } { d ( n - 1 ) } \sum _ { j = 1 } ^ { n } \| y ^ { ( j ) } - \bar { y } \| _ { 2 } ^ { 2 } , \qquad \bar { y } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } y ^ { ( j ) } .\tag{3}
$$

Under Gaussian noise, $d ( n - 1 ) \hat { \sigma } ^ { 2 } / \sigma ^ { 2 }$ follows a chi-square distribution with $d ( n - 1 )$ degrees of freedom [24], so the relative standard error of $\hat { \sigma } ^ { 2 }$ is $\sqrt { 2 / ( d ( n - 1 ) ) }$ . For a 768- dimensional embedder, even $n = 1 0$ calibration queries give a relative standard error of about 1.7% for $\hat { \sigma } ^ { 2 }$ and about 0.85% for σˆ. Please note that, even though an attacker can probably estimate the clean embedding for the text they input to the encoder multiple times and these clean embeddings might be helpful in reconstructing the original text that the adversaries input. But the text we want to protect is NOT the text that the adversaries input but the hidden text of the target systems. For these protected texts, the adversaries only observe the released noisy embeddings and they cannot estimate the clean embeddings for them.

Given the observed noisy embedding y and black-box embedder f, the task is to reconstruct the original text x. A natural initial attempt is to directly train an inversion model on the noisy pairs $( y , x )$ . However, as observed in prior experiments by Zhuang et al. [16], this strategy remains largely insufficient. Below, we analyze theoretically why this naive approach fails.

## B. Vec2Text Fails due to “Double Noise Trap”

Generative inversion methods such as Vec2Text [9] are designed to map a target embedding z back to text x by modeling $p _ { \theta } ( x \mid z )$ via an inverter, and refine a hypothesis xˆ via a corrector by comparing the target embedding with the embedding of the hypothesis $f ( \hat { x } )$ . Under a noisy embedding interface, both quantities are observed with perturbations:

$$
y _ { \mathrm { t a r } } = z + \epsilon _ { 1 } , \qquad y _ { \mathrm { h y p } } = f ( \hat { x } ) = f _ { \mathrm { c l e a n } } ( \hat { x } ) + \epsilon _ { 2 } ,\tag{4}
$$

where $\epsilon _ { 1 } , \epsilon _ { 2 } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ are independent. The observed correction residual is therefore

$$
\begin{array} { r l } & { \Delta _ { \mathrm { o b s } } = y _ { \mathrm { t a r } } - y _ { \mathrm { h y p } } } \\ & { \qquad = \underbrace { z - f _ { \mathrm { c l e a n } } ( \hat { x } ) } _ { \mathrm { s e m a n t i c ~ r e s i d u a l } } + \underbrace { ( \epsilon _ { 1 } - \epsilon _ { 2 } ) } _ { \mathrm { n o i s e ~ r e s i d u a l } } . } \end{array}\tag{5}
$$

Thus, the correction signal contains not only the desired semantic residual, but also a residual noise term formed by subtracting two independent perturbations. Since $\epsilon _ { 1 } - \epsilon _ { 2 } \sim$ $\mathcal { N } ( 0 , 2 \sigma ^ { 2 } I _ { d } )$ , the residual noise has doubled variance, and its typical magnitude is

$$
\lVert \epsilon _ { 1 } - \epsilon _ { 2 } \rVert _ { 2 } \approx \sqrt { 2 } \sigma \sqrt { d } .\tag{6}
$$

For a 768-dimensional embedding with $\sigma = 0 . 0 1$ , this residual noise magnitude is approximately 0.39, which is larger than the single-noise displacement $\sigma \sqrt { d } \approx 0 . 2 8$

As shown in Fig. 1, clean embeddings reside on a textinduced manifold, and additive noise pushes y off this manifold. Consequently, the corrector attempting to match the hypothetical embedding $f ( \hat { x } )$ against the noisy target y is fundamentally misguided. This geometric mismatch exposes the standard inversion process to what we term the “Double Noise Trap”. The optimization objective expands as follows:

$$
\begin{array} { r l } & { \| y _ { \mathrm { t a r } } - y _ { \mathrm { h y p } } \| _ { 2 } ^ { 2 } = \| z - f _ { \mathrm { c l e a n } } ( \hat { x } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad +  2  z - f _ { \mathrm { c l e a n } } ( \hat { x } ) , \epsilon _ { 1 } - \epsilon _ { 2 }  + \| \epsilon _ { 1 } - \epsilon _ { 2 } \| _ { 2 } ^ { 2 } . } \end{array}\tag{7}
$$

The first term is the desired semantic matching objective, while the remaining terms are caused by noise. As refinement improves and the semantic residual $\| z - f _ { \mathrm { c l e a n } } ( \hat { x } ) \| _ { 2 } ^ { 2 }$ becomes smaller, the stochastic cross term and the doubled noise baseline can dominate the correction signal. The corrector may therefore reward hypotheses whose embeddings align with residual noise directions rather than true semantic directions.

This motivates our decoupled design, which first estimates a denoised clean-manifold representation and then performs generation and correction through denoised embeddings.

## C. Infeasible Supervised Denoising

The strict no-clean-embedding constraint fundamentally changes the nature of the denoising process. The standard supervised denoising objective,

$$
\operatorname* { m i n } _ { \phi } \mathbb { E } \left[ \| h _ { \phi } ( y ) - z \| _ { 2 } ^ { 2 } \right] ,\tag{8}
$$

where $h _ { \phi }$ denotes a denoising neural network parameterized by $\phi ,$ is inherently unavailable because the clean target z is unavailable.

Consequently, these analyzes underscore the critical need for a novel inversion pipeline: one capable of reconstructing the original text relying strictly on the noisy observations.

## III. METHOD

We propose DAEI, a novel pipeline for robust noisy text embedding inversion, with its overall architecture and data flow illustrated in Fig. 2. Our system is built upon four components: a black-box noisy embedder $f ( \cdot )$ , a denoising autoencoder (DAE) $h _ { \phi } .$ , a generative inverter θ, and an iterative corrector. Rather than a naive end-to-end approach, the DAEI pipeline comprises four distinct stages: (1) SURE-based unsupervised DAE pre-training, (2) DAE-shielded inverter training, (3) joint SURE-CE fine-tuning, and (4) DAE-shielded corrector training. Notably, the architectures for Stages 2 and 4 remain virtually identical to vanilla Vec2Text, with the distinction that the input embeddings are denoised by our DAE in advance.

## A. SURE-based Unsupervised DAE

Stage 1 aims to learn a denoising module $h _ { \phi }$ that constructs a denoised embedding z˜ from the noisy observation $y \colon$

$$
\tilde { z } = h _ { \phi } ( y , \sigma ) .\tag{9}
$$

![](images/edbcf5f01f103ec5958f4d5ecfffd67e27a856cf72c8ff6b6d2cfa697ce044b8.jpg)  
Fig. 2: The overall architecture and data flow of our proposed DAEI pipeline. The system consists of four functional components (grey vertical panels) optimized across four stages. (1) Upon obtaining a noisy embedding from the hidden text via the Noisy Embedder, we first apply the DAE (indicated by the shield icons) to reconstruct a denoised embedding z˜. (2) It is then processed by the Inverter to generate an initial hypothetical text $\hat { x } ^ { ( k ) }$ . (3) During training, the token-level cross-entropy loss $( \mathcal { L } _ { \mathrm { C E } } )$ backpropagates directly through the active DAE to achieve joint fine-tuning (grey dashed lines). (4) Finally, the generated text enters an iterative correction loop: it is re-encoded by the embedder and passed through the DAE to yield a denoised hypothesis embedding $\tilde { e } ^ { ( k ) }$ . Both $\tilde { e } ^ { ( k ) ^ { \ast } }$ and the original denoised target z˜ are then fed into the Corrector to predict the next hypothetical text, repeating until the loop terminates to produce the final output.

The oracle denoising risk Eq. (8) cannot be directly optimized because the clean embedding z is unobserved. To check if the dependency on z can be removed, we first expand Eq. (8):

$$
\begin{array} { r } { \| h _ { \phi } ( y , \sigma ) - z \| _ { 2 } ^ { 2 } = \| h _ { \phi } ( y , \sigma ) \| _ { 2 } ^ { 2 } - 2 z ^ { \top } h _ { \phi } ( y , \sigma ) + \| z \| _ { 2 } ^ { 2 } . } \end{array}\tag{10}
$$

The last term $\| z \| _ { 2 } ^ { 2 }$ is independent of $\phi ,$ while the cross term $z ^ { \top } h _ { \phi } ( y , \sigma )$ is inaccessible because z is hidden. However, under the Gaussian noise model $y ~ = ~ z + \epsilon$ , where $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ , Stein’s lemma gives [21]

$$
\mathbb { E } _ { \epsilon } \left[ \epsilon ^ { \top } h _ { \phi } ( y , \sigma ) \right] = \sigma ^ { 2 } \mathbb { E } _ { \epsilon } \left[ \mathrm { d i v } _ { y } h _ { \phi } ( y , \sigma ) \right] .\tag{11}
$$

Since $z = y - \epsilon ,$ the inaccessible cross term can be replaced in expectation by quantities depending only on $y ,$ the noise variance $\sigma ^ { 2 }$ , and the divergence of the denoiser. This leads to SURE [21]:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S U R E } } ( y ; \phi ) = \| h _ { \phi } ( y , \sigma ) - y \| _ { 2 } ^ { 2 } + 2 \sigma ^ { 2 } \mathop { \mathrm { d i v } } _ { y } h _ { \phi } ( y , \sigma ) - d \sigma ^ { 2 } . } \end{array}\tag{12}
$$

Here, di $\mathrm { ~ v ~ } _ { y } h _ { \phi } ( y , \sigma ) = \mathrm { T r } ( \partial h _ { \phi } / \partial y )$ measures the local sensitivity of the denoiser with respect to its noisy input. Therefore, although z is never observed, minimizing Eq. (12) is equivalent in expectation to minimizing the oracle denoising risk in Eq. (8).

We further parameterize the denoising module as a residual correction around the observed noisy embedding, rather than directly predicting z˜ from scratch:

$$
h _ { \phi } ( y , \sigma ) = y + r _ { \phi } ( y , \sigma ) ,\tag{13}
$$

where $r _ { \phi } ( y , \sigma )$ estimates the displacement needed to suppress the noise component and move the observation back toward the clean embedding manifold.

Substituting this parameterization into the divergence term gives $\mathrm { d i v } _ { y } h _ { \phi } ( y , \sigma ) \ = \ d + \mathrm { d i v } _ { y } r _ { \phi } ( y , \sigma )$ . After removing constants independent of $\phi ,$ the training objective becomes

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S U R E } } ( y ; \phi ) \equiv \Vert r _ { \phi } ( y , \sigma ) \Vert _ { 2 } ^ { 2 } + 2 \sigma ^ { 2 } \operatorname { d i v } _ { y } r _ { \phi } ( y , \sigma ) . } \end{array}\tag{14}
$$

We approximate the divergence term with Monte-Carlo Hutchinson estimation [22]:

$$
\mathrm { d i v } _ { y } r _ { \phi } ( y , \sigma ) \approx \frac { 1 } { K } \sum _ { k = 1 } ^ { K } { v _ { k } ^ { \top } J _ { r _ { \phi } } ( y , \sigma ) v _ { k } } , \qquad v _ { k } \in \{ - 1 , + 1 \} ^ { d } ,\tag{15}
$$

where $J _ { r _ { \phi } }$ is the Jacobian of the residual network with respect to the noisy embedding. Let $\widehat { \mathrm { d i v } _ { K } }$ denote the Monte-Carlo estimator in Eq. (15). For Rademacher probes satisfying $\mathbb { E } [ v _ { k } v _ { k } ^ { \top } ] = I $ , this estimator is unbiased and its variance is bounded by

$$
\begin{array} { r l } & { \mathbb { E } _ { v } \left[ \widehat { \mathrm { d i v } } _ { K } \right] = \mathrm { d i v } _ { y } r _ { \phi } ( y , \sigma ) , } \\ & { \mathrm { V a r } \left[ \widehat { \mathrm { d i v } } _ { K } \right] \le \displaystyle \frac { 2 } { K } \| J _ { r _ { \phi } } ( y , \sigma ) \| _ { F } ^ { 2 } . } \end{array}\tag{16}
$$

Thus, the approximation is theoretically well grounded: it is unbiased, and its variance decays at rate $O ( 1 / K )$ . In practice, we use Rademacher probes and spectral normalization in the residual MLP to stabilize the trace estimate.

## B. DAE-Shielded Inverter

After we obtain denoised representation z˜, the inverter θ is subsequently trained on the pairs (˜z, x) rather than the raw noisy embeddings. It prevents the generative model from overburdening its capacity with high-dimensional noisefiltering and avoids overfitting to non-semantic perturbations.

The training objective is the standard token-level crossentropy loss:

$$
\mathcal { L } _ { \mathrm { C E } } ( \boldsymbol { \theta } ; \boldsymbol { \phi } ) = - \sum _ { t = 1 } ^ { T } \log p _ { \boldsymbol { \theta } } \left( x _ { t } \mid \boldsymbol { x } _ { < t } , \boldsymbol { \tilde { z } } \right)\tag{17}
$$

where $T$ denotes the sequence length, $x _ { t }$ is the target token at step t, and $x _ { < t }$ represents the preceding autoregressive context.

## C. Joint SURE-CE Fine-tuning

Following the independent training of the DAE and the inverter, stage 3 unifies the architecture. To explicitly align the unsupervised denoising process with the downstream generative task, we jointly fine-tune the DAE and the inverter:

$$
\mathcal { L } _ { \mathrm { S U R E - C E } } ( \phi , \theta ) = \mathcal { L } _ { \mathrm { S U R E } } ( y ; \phi ) + \lambda _ { \mathrm { C E } } ( t ) \mathcal { L } _ { \mathrm { C E } } \left( \theta ; \phi \right) .\tag{18}
$$

Here, the weighting coefficient $\lambda _ { \mathrm { C E } } ( t )$ is warmed up during the early training steps. The key difference from Stage 2 is that the cross-entropy gradient is allowed to pass through $h _ { \phi } ( y , \sigma )$ . Consequently, the DAE receives two complementary supervisory signals: the SURE objective ensures that the output remains strictly aligned with the denoising manifold, while the CE (cross-entropy) loss provides semantic feedback, preserves the latent directions for decoding. Concurrently, the inverter is continuously trained on the embeddings denoised dynamically by the fine-tuning DAE to adapt to the optimized embedding distribution.

However, the optimization directions of these two objectives may conflict. We therefore employ distinct learning rates for the DAE and the inverter, and apply gradient surgery PCGrad [25] exclusively to the DAE parameters $\phi .$

Let g<sub>SURE</sub> $\triangleq \overline { { \frac { \Delta } { \omega } } } \operatorname { V } _ { \phi } \mathcal { L } _ { \mathrm { S U R E } }$ and g<sub>CE</sub> $\triangleq \nabla _ { \phi } \mathcal { L } _ { \mathrm { C E } }$ denote the gradients with respect to the DAE parameters. If these two gradients exhibit a negative inner product $( \mathrm { i . e . , \ } \langle g _ { \mathrm { C E } } , g _ { \mathrm { S U R E } } \rangle < 0 )$ they are conflicting. In such cases, we project the CE gradient onto the normal plane of the SURE gradient before merging the updates:

$$
\begin{array} { r l r } {  { g _ { \mathrm { C E } } ^ { \prime } = g _ { \mathrm { C E } } - \frac { \langle g _ { \mathrm { C E } } , g _ { \mathrm { S U R E } } \rangle } { \| g _ { \mathrm { S U R E } } \| _ { 2 } ^ { 2 } } g _ { \mathrm { S U R E } } , } } \\ & { } & { \phi  \phi - \eta _ { \mathrm { D A E } } ( g _ { \mathrm { S U R E } } + g _ { \mathrm { C E } } ^ { \prime } ) . } \end{array}\tag{19}
$$

This orthogonal projection guarantees that the text feedback $( g _ { \mathrm { C E } } ^ { \prime } )$ shapes the final representation without unlearning the fundamental denoising capabilities.

## D. DAE-Shielded Corrector

Conventionally, once an inverter is trained to generate an initial text hypothesis, a corrector iteratively refines this hypothesis by comparing its embedding against the target embedding. However, when the available target is noisy, directly performing this comparison inevitably triggers the “Double Noise Trap” described in Section II-B.

To mitigate this, we introduce a DAE Shield to both sides of the correction interface. Specifically, instead of comparing raw representations, we pass both the noisy target y and the embedding of the current hypothesis ${ \hat { x } } ^ { ( k ) }$ through our Stage 3 denoiser $h _ { \phi ^ { \star } }$ :

$$
\tilde { z } = h _ { \phi ^ { \star } } ( y , \sigma ) , \qquad \tilde { e } ^ { ( k ) } = h _ { \phi ^ { \star } } ( f ( \hat { x } ^ { ( k ) } ) , \sigma ) ,\tag{20}
$$

where $\hat { x } ^ { ( k ) }$ is the generated text at correction step $k ,$ and $\phi ^ { \star }$ denotes the fully fine-tuned DAE parameters. The corrector then conditions exclusively on the denoised target z˜, the denoised hypothesis $\tilde { e } ^ { ( k ) }$ , and their residual difference. By completely shielding the correction interface, we constrain the iterative refinement strictly within the learned clean manifold, ensuring the model aligns with semantic text directions rather than chasing irreducible noise residuals.

## IV. EXPERIMENTAL SETUP

Our study aims to determine the extent to which noiseprotected embeddings from black-box encoders remain vulnerable to advanced inversion attack paradigms. Specifically, we investigate the following research questions: RQ1: How does the proposed DAEI pipeline compare against vanilla Vec2Text and existing noise-robust baselines in recovering target text from noisy embeddings? RQ2: Can the proposed DAEI pipeline be utilized in different embedding model backbones? RQ3: Can DAEI maintain strong inversion performance on out-of-domain test sets? RQ4: Can the DAE component effectively reduce noise? RQ5: Does the joint finetuning module improve reconstruction quality? RQ6: To what extent can DAEI maintain its inversion performance when the noise σ varies?

## A. Dataset

1) Training set: All main experiments use the same training mixture of:

• Natural Questions (NQ) is a question answering benchmark built from real Google search queries [26].

• MS MARCO is a large-scale machine reading retrieval dataset constructed from real Bing queries, web passages, and human-generated answers [27].

• Yahoo Answers is a community question-answering domain covering broad everyday topics.

We capped the combined mixture at 7M text examples as raw text sources to construct the inversion datasets. While the underlying text corpora remain constant, the exact inputtarget configurations vary across our four training stages to accommodate the specific objectives of each module:

• Stage 1 (DAE Pre-training): To train the residual DAE, the system requires noisy embeddings only.

• Stage 2 (Base Inverter Training): To train the initial text decoding module, we utilize (denoised embedding, text) pairs. These denoised embeddings are computed by the Stage 1 DAE.

• Stage 3 (Joint Fine-tuning): For end-to-end joint optimization, the dataset consists of (noisy embedding, text) pairs. Here, the noisy embeddings are dynamically denoised on-the-fly by the actively tuning DAE during the forward pass.

• Stage 4 (Corrector Training): To train the iterative corrector, we rely on expanded input tuples consisting of (denoised embedding, hypothetical text, denoised hypothetical embedding), paired with the original text as label. Crucially, these denoised embeddings are computed by the fully tuned Stage 3 DAE, ensuring the corrector operates strictly within the optimal denoised distribution.

2) Evaluation sets: For evaluation, we consider both indomain and out-of-domain settings. The in-domain test set is the held-out splits from the same NQ, MS MARCO, and Yahoo Answers mixture, while the out-of-domain test sets are drawn from five unseen corpora: AG News, Anthropic Toxic Prompts, Python Code Alpaca, Climate-FEVER [28], and MedMCQA [29]. These out-of-domain datasets allow us to evaluate whether the inversion methods generalize beyond the training distribution. The data sources for all training and evaluation sets are provided in our repository: https://anonymous.4open.science/r/DAEI-1F1C/

## B. Target Embedding Models

To rigorously evaluate the generalizability of our DAEI pipeline across different latent space distributions, we select two widely adopted, state-of-the-art dense text embedding models as our black-box targets:

• GTR-base [30]: The GTR (Generalizable T5-based dense Retrievers) model leverages a bi-encoder architecture initialized from the T5 [31] checkpoints. It is specifically optimized for robust zero-shot retrieval tasks, producing highly structured semantic representations.

• GTE-base [32]: The General Text Embeddings (GTE) model trained on large-scale contrastive data represents a highly optimized encoder-only architecture.

We deliberately select GTR and GTE since they represent two fundamentally distinct backbone architectures (T5-based vs. BERT-based) and training paradigms, thereby demonstrate the generalizability of our method.

## C. Baselines and Our Models

Given the scarcity of existing work on noisy text embedding inversion, we use ZSINVERT [18] (an adversarial decoding approach) as our primary external baseline even though it is not based on the mainstream encoder-decoder architecture because it is the only existing work that we know has shown its effectiveness in the noisy text embedding inversion scenario. Our core focus remains on a rigorous comparison between various configurations of the encoder-decoder models and our proposed approach. To comprehensively assess inversion performance, we categorize the baselines and our models into three distinct groups:

1) Clean Upper Bound: CLEAN-TO-CLEAN Baseline evaluates the vanilla Vec2Text [9] model on clean embeddings. It represents the state-of-the-art text recovery performance under ideal, noiseless conditions. This serves as the theoretical ceiling that our noisy inversion pipelines aim to approach.

TABLE I: Main experimental systems. ZSINVERT base is training-free; its listed training embedding type refers only to corrector training.
<table><tr><td>System</td><td>Train embedding</td><td>Test embedding</td></tr><tr><td>CLEAN-TO-CLEAN Baseline</td><td>Clean</td><td>Clean</td></tr><tr><td>ZSINVERT</td><td>Noisy</td><td>Noisy</td></tr><tr><td>CLEAN-TO-NOISY Baseline</td><td>Clean</td><td>Noisy</td></tr><tr><td>NOISY INVERTER Baseline</td><td>Noisy</td><td>Noisy</td></tr><tr><td>DAE-ONLY (ours)</td><td>Denoised by stage1 DAE</td><td>Denoised by stage1 DAE</td></tr><tr><td>DAEI (ours)</td><td>Denoised by stage3 DAE</td><td>Denoised by stage3 DAE</td></tr></table>

2) Generative Inversion under Noise: This group compares the standard autoregressive inversion models against our proposed pipelines when facing perturbed embeddings:

• CLEAN-TO-NOISY Baseline: Tests the severe distribution shift faced by a vanilla Vec2Text model that is trained solely on clean embeddings but evaluated on noisy ones.

• NOISY INVERTER Baseline: Trains a vanilla Vec2Text directly on noisy embeddings. This provides a much stronger baseline that adapts to the corrupted embedding distribution but lacks denoising mechanism.

• DAE-ONLY inverter (Ours): Our proposed pipeline without joint fine-tuning. A DAE first denoises the embedding, then an inverter is trained on the denoised cache.

• DAEI (Ours): Our complete pipeline that jointly finetunes both the DAE and the inverter.

3) Adversarial-Decoding-Based Baseline: To step outside the generative paradigm, we introduce ZSINVERT as an external baseline. This method frames inversion as adversarial decoding, iteratively optimizing discrete tokens to minimize embedding distance.

Table I summarizes the above systems evaluated in our main experiments.

## D. Metrics.

For evaluating the inverters, we report sentence-level BLEU [33], token-level F1 [34], ROUGE-L [35], and the cosine similarity between the embeddings of the reconstructed and reference texts. Specifically, BLEU assesses generation precision and local fluency by measuring the n-gram overlap between the reconstructed and reference texts; token-level F1 assesses keyword recovery by measuring unigram overlap independently of sequence order; ROUGE-L evaluates structural similarity and information recall based on the longest common subsequence; and embedding cosine similarity measures semantic closeness in the target embedding space.

For evaluating the DAE, we also report the MSE and embedding cosine similarity between the DAE’s denoised outputs and the clean embeddings. We emphasize that these clean embeddings are strictly isolated for offline evaluation purposes only; they are never observed by the model during any stage of training, nor are they used to compute any optimization loss.

## E. Implementation Setting.

Our method assumes the noise scale σ is prior knowledge (see Section II-A). Unless otherwise specified in the ablation studies, we apply zero-mean Gaussian noise with a standard deviation of $\sigma ~ = ~ 0 . 0 1 ~ [ 9 ]$ , [16]. The DAE module is pretrained with a learning rate of $1 \times 1 0 ^ { - 3 }$ . We configure a hidden dimension of 1024 and a depth of 3, spectral normalization, and five Monte-Carlo probes.

All inverter variants use the Vec2Text architecture with a T5 [31] backbone and are optimized with a learning rate of $1 \times 1 0 ^ { - 3 }$ . Following the original Vec2Text setting [9], both input and target texts are truncated to a maximum length of 128 tokens during training, while evaluation is conducted with a maximum length of 32 tokens.

During the joint fine-tuning stage, the DAE learning rate is reduced to $2 \times 1 0 ^ { - 4 }$ , and the cross-entropy loss weight is linearly warmed up over 10,000 steps.

All corrector variants use the Vec2Text architecture with a learning rate of $5 \times 1 0 ^ { - 4 }$ . We enforce an early-stop threshold of $1 \times 1 0 ^ { - 3 }$ : refinement terminates once the cosine similarity gain between consecutive steps falls below this value, preventing the corrector from overfitting to residual noise.

## V. RESULTS

## A. Overall Performance (RQ1 & RQ2)

DAEI achieves the best noisy inversion performance. Tables II and III report the main inversion results on GTR-base and GTE-base, respectively. On GTR-base, DAEI substantially improves over the NOISY INVERTER baselines across all metrics. In particular, DAEI increases BLEU from 0.1916 to 0.4875, yielding an absolute gain of nearly 30 percentage points and a 154.4% relative improvement over the NOISY INVERTER baseline. It also improves token-level F1 from 0.5814 to 0.7717 and ROUGE-L from 0.5660 to 0.7788, indicating that DAEI can recover richer textual information from noisy embeddings alone. Moreover, the embedding cosine similarity increases from 0.8349 to 0.9254, suggesting that the DAE-shielded correction process more effectively aligns the hypothesis embedding with the target embedding after removing a substantial portion of the perturbation noise.

Noisy training alone provides limited gains. The comparison between the CLEAN-TO-NOISY baseline and the NOISY INVERTER baseline further illustrates the limitation of directly training Vec2Text on noisy embedding. On GTR-base, noisy training improves text reconstruction metrics by only 5–8 points over the CLEAN-TO-NOISY baseline, indicating limited adaptation to noisy inputs. However, due to the Double Noise Trap, the correction stage compares a noisy hypothesis embedding with a separately perturbed target, causing the residual signal to be contaminated by doubled noise. As a result, this gain remains far lower than that achieved by DAEI.

DAEI approaches the clean upper bound and outperforms ZSINVERT in text reconstruction. Relative to the clean upper bound, DAEI recovers 78.0%/89.2% of BLEU/F1 on GTR-base and 71.7%/83.2% on GTE-base. These results suggest that noise-protected embeddings still retain considerable recoverable textual information when the adversary adopts a denoising-aware inversion pipeline. However, ZS-INVERT performs poorly on text-overlap metrics for both embedders, achieving only around 0.01 BLEU and 0.08–0.15 on F1 and ROUGE-L, despite obtaining a relatively high embedding cosine similarity of 0.85–0.96. This gap indicates that adversarial decoding matches the embedding space but fails to reliably recover token-aligned text, whereas DAEI better balances semantic matching and fluent reconstruction.

DAEI generalizes across embedding backbones. DAEI shows similar gains on GTE-base, answering RQ2. Although the overall scores on GTE-base are lower than GTR-base, the clean upper bound is also lower, suggesting that the underlying embedding space is intrinsically more difficult for Vec2Text framework to invert. One possible reason is that our inverter uses a T5-base backbone, which is more closely aligned with the T5-based GTR encoder than with the BERTstyle GTE encoder [30], [32]; meanwhile, GTE is trained with multi-stage contrastive learning for general-purpose semantic retrieval [32], which may emphasize semantic invariance while preserving less information needed for token-level reconstruction. Even under this harder setting, DAEI improves over the NOISY INVERTER baseline by approximately 19–23 points on text reconstruction metrics, while also increasing cosine similarity by 10.36 points. These consistent improvements over two architecturally different encoders demonstrate that DAEI is generalized across different noisy embedding interfaces.

## B. Out-of-Domain Generalization (RQ3)

Table IV evaluates whether DAEI can generalize to Out-Of-Domain (OOD) test sets. Across all OOD datasets, DAEI consistently outperforms the NOISY INVERTER and ZSINVERT on BLEU, F1, and ROUGE-L, showing that DAEI remains effective when the target texts come from unseen domains.

The performance varies across datasets. DAEI performs especially well on Anthropic Toxic Prompts, Python Code, and MedMCQA, while AG News obtains lower scores for both the clean upper bound and DAEI. One likely reason is that AG News examples are longer on average, with about 61 tokens compared with 22–30 tokens for the other datasets. Although evaluation outputs are truncated to 32 tokens, longer source texts still contain more information to recover and are more likely to suffer from truncation and partial reconstruction.

## C. Components Effectiveness (RQ4 & RQ5)

1) DAE Effectiveness: DAE reduces noise-induced embedding distortion. Table V demonstrates DAE can effectively reduce the perturbation noise without using clean embeddings during training. The Stage 1 DAE reduces MSE to the clean embeddings by 29.2% on GTR-base and 23.8% on GTE-base, while improving cosine similarity from 0.9179 to 0.9322 and from 0.9662 to 0.9727, respectively. Since the injected noise is relatively small, the raw noisy embeddings are already close to the clean embeddings in cosine similarity, making the cosine gains modest. Nevertheless, the consistent

TABLE II: Main results for in-domain evaluation with GTR-base. Gray rows indicate reference settings that are not available under the current threat model. Improvements in parentheses are absolute gains over the NOISY INVERTER Baseline. Best feasible results excluding the clean upper bound are in bold.
<table><tr><td>Category</td><td>Model</td><td>BLEU</td><td>F1</td><td>ROUGE-L</td><td>Cosine Similarity</td></tr><tr><td>Clean Upper Bound</td><td>CLEAN-TO-CLEAN Baseline</td><td>0.6252</td><td>0.8656</td><td>0.8704</td><td>0.9500</td></tr><tr><td>Adversarial Decoding</td><td>ZSINVERT</td><td>0.013</td><td>0.1503</td><td>0.1117</td><td>0.8568</td></tr><tr><td rowspan="4">Generative Inversion</td><td>CLEAN-TO-NOISY Baseline</td><td>0.1348</td><td>0.5005</td><td>0.5012</td><td>0.7785</td></tr><tr><td>NOISY INVERTER Baseline</td><td>0.1916</td><td>0.5814</td><td>0.5660</td><td>0.8349</td></tr><tr><td>DAE-ONLY (Ours)</td><td>0.3616</td><td>0.7206</td><td>0.7200</td><td>0.9055</td></tr><tr><td>DAEI (Ours)</td><td>0.4875 (+0.2959)</td><td>0.7717 (+0.1903)</td><td>0.7788 (+0.2128)</td><td>0.9254 (+0.0905)</td></tr></table>

TABLE III: Main results for in-domain evaluation with GTE-base. Formatting follows Table II.
<table><tr><td>Category</td><td>Model</td><td>BLEU</td><td>F1</td><td>ROUGE-L</td><td>Cosine Similarity</td></tr><tr><td>Clean Upper Bound</td><td>CLEAN-TO-CLEAN Baseline</td><td>0.4391</td><td>0.7622</td><td>0.7570</td><td>0.9211</td></tr><tr><td>Adversarial Decoding</td><td>ZSINVERT</td><td>0.0104</td><td>0.1189</td><td>0.0894</td><td>0.9641</td></tr><tr><td rowspan="4">Generative Inversion</td><td>CLEAN-TO-NOISY Baseline</td><td>0.0846</td><td>0.3222</td><td>0.3271</td><td>0.7081</td></tr><tr><td>NOISY INVERTER Baseline</td><td>0.1249</td><td>0.3960</td><td>0.4044</td><td>0.7520</td></tr><tr><td>DAE-ONLY (Ours)</td><td>0.2688</td><td>0.5854</td><td>0.5722</td><td>0.8141</td></tr><tr><td>DAEI (Ours)</td><td>0.3152 (+0.1903)</td><td>0.6344 (+0.2384)</td><td>0.6324 (+0.2280)</td><td>0.8556 (+0.1036)</td></tr></table>

TABLE IV: Out-of-domain evaluation results for DAEI. Upper Bound denotes CLEAN-TO-CLEAN Baseline, and Baseline denotes the NOISY INVERTER Baseline.
<table><tr><td>OOD Dataset</td><td>Method</td><td>BLEU</td><td>F1</td><td>ROUGE-L</td></tr><tr><td rowspan="4">AG News</td><td>Upper Bound</td><td>0.1023</td><td>0.4296</td><td>0.4887</td></tr><tr><td>ZSINVERT</td><td>0.0130</td><td>0.1629</td><td>0.1142</td></tr><tr><td>Baseline</td><td>0.0531</td><td>0.2749</td><td>0.3109</td></tr><tr><td>DAEI</td><td>0.0791</td><td>0.3346</td><td>0.3915</td></tr><tr><td rowspan="4">Anthropic Toxic Prompts</td><td>Upper Bound</td><td>0.8114</td><td>0.8912</td><td>0.9118</td></tr><tr><td>ZSINVERT</td><td>0.0137</td><td>0.1451</td><td>0.1149</td></tr><tr><td>Baseline</td><td>0.2128</td><td>0.5581</td><td>0.5590</td></tr><tr><td>DAEI</td><td>0.7217</td><td>0.8881</td><td>0.8885</td></tr><tr><td rowspan="4">Python Code</td><td>Upper Bound</td><td>0.6732</td><td>0.8588</td><td>0.8783</td></tr><tr><td>ZSINVERT</td><td>0.0174</td><td>0.1928</td><td>0.1434</td></tr><tr><td>Baseline</td><td>0.1933</td><td>0.6085</td><td>0.5961</td></tr><tr><td>DAEI</td><td>0.5080</td><td>0.8213</td><td>0.8316</td></tr><tr><td rowspan="4">Climate fever</td><td>Upper Bound</td><td>0.5350</td><td>0.7843</td><td>0.7824</td></tr><tr><td>ZSINVERT</td><td>0.0132</td><td>0.1518</td><td>0.1105</td></tr><tr><td>Baseline</td><td>0.1377</td><td>0.4853</td><td>0.4980</td></tr><tr><td>DAEI</td><td>0.3489</td><td>0.6651</td><td>0.6961</td></tr><tr><td rowspan="4">Medmcqa</td><td>Upper Bound</td><td>0.6689</td><td>0.8476</td><td>0.8800</td></tr><tr><td>ZSINVERT</td><td>0.0128</td><td>0.1286</td><td>0.1070</td></tr><tr><td>Baseline</td><td>0.1739</td><td>0.5252</td><td>0.5576</td></tr><tr><td>DAEI</td><td>0.4871</td><td>0.7730</td><td>0.8037</td></tr></table>

TABLE V: DAE component performance across training stages. MSE and cosine similarity are computed against clean embeddings before and after DAE denoising, with ∆ reporting the denoising improvement.
<table><tr><td rowspan="2">Embedder Stage</td><td rowspan="2"></td><td colspan="3">MSE↓</td><td colspan="3">Cosine ↑</td></tr><tr><td>Noisy</td><td>DAE</td><td>Δ</td><td>Noisy</td><td>DAE</td><td>∆</td></tr><tr><td rowspan="2">GTE-base</td><td>Stage 1</td><td>0.0765</td><td>0.0583</td><td>0.0182</td><td>0.9662</td><td>0.9727</td><td>0.0065</td></tr><tr><td>Stage 3</td><td>0.0765</td><td>0.0602</td><td>0.0163</td><td>0.9662</td><td>0.9713</td><td>0.0051</td></tr><tr><td rowspan="2">GTR-base</td><td>Stage 1</td><td>0.0766</td><td>0.0542</td><td>0.0224</td><td>0.9179</td><td>0.9322</td><td>0.0142</td></tr><tr><td>Stage 3</td><td>0.0766</td><td>0.0575</td><td>0.0191</td><td>0.9179</td><td>0.9280</td><td>0.0101</td></tr></table>

![](images/5a780725f3a43169afaa07a0e6e401d9162ef3fb4732219705841411e31434e3.jpg)

![](images/d58b726de9716c1c6c73a7868a3c788c6066c64ba923157957699a738ccd55ec.jpg)  
(a) Mean LID with SEM.  
(b) Median LID.  
Fig. 3: LID-based geometric evaluation of DAE projection using clean embeddings as the reference set. The two subfigures report mean LID with SEM (Standard Error of the Mean) error bars and median LID, respectively, showing DAE reduces the LID of noisy embeddings back to the clean level.

MSE reduction and cosine improvement show that the DAE shifts noisy embeddings closer to their clean counterparts, even without using clean embeddings as training targets.

Different roles of Stage 1 and Stage 3 DAEs. Interestingly, Table V shows that the Stage 3 DAE has slightly weaker overall denoising metrics than the Stage 1 DAE in terms of MSE and cosine similarity to the clean embedding. This is expected because the two stages optimize for different objectives: Stage 1 focuses purely on denoising, while Stage 3 is additionally guided by the cross-entropy objective toward semantic directions for text reconstruction, resulting in a representation that is better aligned with the generative decoder.

DAE projects noisy embeddings toward the clean manifold. We further evaluate the DAE from a geometric perspective using Local Intrinsic Dimensionality (LID) [36]. For each test embedding, we estimate its LID with respect to a heldout clean embedding reference set using the MLE (Maximum Likelihood Estimation) estimator with 100 nearest neighbors. As shown in Fig. 3, Gaussian noise increases the mean LID from 47.65 to 51.15, indicating that noisy embeddings deviate from the clean embedding manifold. After DAE projection, the LID decreases back to the clean level: Stage 1 obtains a slightly lower LID (47.19) than clean embedding, it may because a purely denoising DAE pulls embeddings toward more averaged regions of the clean distribution, result in lower estimated local dimensionality. In contrast, Stage 3 reaches 47.65, almost exactly matching the clean embeddings, because joint fine-tuning introduces semantic alignment with the clean manifold. The median results show the same trend, confirming that the reduction is not driven by outliers.

DAE improves downstream inversion. The effectiveness of the DAE is also reflected in the downstream inversion results through the comparison between the NOISY INVERTER baseline and DAE-ONLY. Across the two embedders, DAE-ONLY improves the text reconstruction metrics by approximately 14–19 percentage points, while increasing embedding cosine similarity by about 6–7 percentage points. This demonstrates the fundamental effectiveness of the denoisingaware strategy: removing part of the perturbation noise before inversion makes the input representation more semantically stable and easier for the generative decoder to interpret.

2) Role of Joint Fine-tuning: While DAE-ONLY already provides strong gains, the full DAEI model further improves performance by jointly fine-tuning the DAE and the inverter. Across the two embedders, DAEI brings additional improvements of approximately 5–13 percentage points on text reconstruction metrics over DAE-ONLY, while also further increasing embedding cosine similarity by 2–4 points. These consistent gains highlight the effectiveness of joint fine-tuning, showing that a small sacrifice in denoising accuracy can be worthwhile when it better aligns the representation with the downstream text decoding objective.

TABLE VI: In-domain evaluation of DAEI on GTR-base under different Gaussian noise scales.
<table><tr><td>σ</td><td>BLEU</td><td>F1</td><td>ROUGE-L</td><td>Cosine</td></tr><tr><td>0.005</td><td>0.4927</td><td>0.7823</td><td>0.7850</td><td>0.9293</td></tr><tr><td>0.008</td><td>0.4932</td><td>0.7807</td><td>0.7836</td><td>0.9288</td></tr><tr><td>0.010</td><td>0.4875</td><td>0.7717</td><td>0.7788</td><td>0.9254</td></tr><tr><td>0.012</td><td>0.4813</td><td>0.7729</td><td>0.7793</td><td>0.9242</td></tr><tr><td>0.015</td><td>0.4628</td><td>0.7578</td><td>0.7634</td><td>0.9218</td></tr><tr><td>0.018</td><td>0.4377</td><td>0.7421</td><td>0.7480</td><td>0.9240</td></tr><tr><td>0.020</td><td>0.3966</td><td>0.7200</td><td>0.7269</td><td>0.9150</td></tr></table>

## D. Generalization Across Varying Noise Levels (RQ6)

Table VI evaluates the in-domain robustness of DAEI on GTR-base under different Gaussian noise scales. The model is trained with $\sigma ~ = ~ 0 . 0 1$ and tested on other noise levels. DAEI remains highly stable when σ varies from 0.005 to 0.018: BLEU stays above 0.4377, F1 remains above 0.7421, and cosine similarity remains around 0.924–0.929. Even at $\sigma ~ = ~ 0 . 0 2$ DAEI still achieves 0.3966 BLEU and 0.7200 F1, indicating that DAEI can tolerate moderate mismatch between the training noise scale and the test-time perturbation. This result is especially relevant when the defender does not use a fixed noise scale, but instead samples σ from a small range, such as [0.005, 0.015], for each released embedding: although the attacker may only estimate an average σ, DAEI can still work effectively. Larger noise scales may further reduce inversion performance, but prior work has shown that overly large perturbations also damage the downstream utility of the protected embedder [9]. Thus, within the practical noise regime, DAEI is robust to realistic noise-scale fluctuations.

## VI. RELATED WORK

## A. Text Embedding Inversion Attacks

Dense text embeddings can leak substantial information about their original inputs. Early work demonstrated this risk by showing that sentence embeddings can reveal sensitive attributes and keywords [7]. Later, Li et al. proposed a generative embedding inversion attack that reconstructs complete sentences [8]. Morris et al. introduced Vec2Text, which achieves high-fidelity reconstruction of short texts [9], while later methods reduce attacker requirements via fewshot, transfer-based, or zero-shot inversion [18], [37], [38]. These works demonstrate the practical privacy risk of text embeddings, but they mainly focus on general embeddings settings rather than embeddings explicitly protected by noise.

## B. Privacy Protection for Text Embeddings

Noise injection is a standard mechanism for privacy protection, with differential privacy providing a formal foundation for calibrated random perturbations [12]. In NLP, Feyisetan et al. applied calibrated multivariate noise in word embedding space to achieve privacy-utility tradeoffs for textual analysis [13], and Xu et al. further refined this idea with a Mahalanobis metric [14]. More recent work extends perturbation-based privacy to sentence-level representations [15], [39]. Recent evaluations also show that Gaussian noise can degrade the performance of standard embedding inversion attacks [9], [17], [18]. Such a lightweight but effective defense poses a substantial challenge to embedding inversion attacks.

## C. Denoising Techniques

Denoising autoencoders learn robust representations by reconstructing clean inputs from corrupted observations [19]. Standard denoising training typically relies on clean reconstruction targets. However, Noise2Noise relaxes this requirement by learning from pairs of independently corrupted observations of the same underlying signal [23]. Stein’s unbiased risk estimate provides a way to estimate Gaussian denoising risk from noisy observations alone [21], and Monte Carlo SURE extends this idea to general denoisers through randomized divergence estimation [22], [40]. However, these techniques have not been studied in the context of text embedding inversion.

As a result, the community still lacks a systematic understanding of whether high-quality text can be reconstructed from noise-protected embeddings. This gap motivates our study of denoising-aware embedding inversion in the noisyonly setting.

## VII. CONCLUSION

In this paper, we studied the privacy risk of noise-protected text embeddings under a threat model where attackers observe only perturbed embeddings. We propose DAEI, a denoisingaware embedding inversion pipeline that combines unsupervised denoising autoencoder with generative inversion. Extensive experiments across embedding backbones, domains, and ablation settings show that DAEI outperforms existing baselines, demonstrating that Gaussian noise alone is insufficient to prevent textual information leakage. These findings highlight the need to evaluate privacy-preserving embedding systems against adaptive denoising-aware attacks and to develop stronger defenses for embedding-based applications.

## VIII. ACKNOWLEDGMENT

This work was supported by Monash eResearch capabilities, including M3 HPC, the Australian Research Council through the Discovery Early Career Researcher Award (DE250100032), the Discovery Project (Grant No. DP260100326) and the Linkage Projects (Grant Nos. LP230200892, LP240200546 and LP250200778).

## REFERENCES

[1] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), 2019, pp. 3982–3992.

[2] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, “Dense passage retrieval for open-domain question answering,” in Proceedings ofthe 2020 conference on empirical methods in natural language processing (EMNLP), 2020, pp. 6769–6781.

[3] N. Muennighoff, N. Tazi, L. Magne, and N. Reimers, “Mteb: Massive text embedding benchmark,” in Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, 2023, pp. 2014–2037.

[4] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel¨ et al., “Retrievalaugmented generation for knowledge-intensive nlp tasks,” Advances in neural information processing systems, vol. 33, pp. 9459–9474, 2020.

[5] W. X. Zhao, J. Liu, R. Ren, and J.-R. Wen, “Dense text retrieval based on pretrained language models: A survey,” ACM Transactions on Information Systems, vol. 42, no. 4, pp. 1–60, 2024.

[6] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, H. Wang, H. Wang et al., “Retrieval-augmented generation for large language models: A survey,” arXiv preprint arXiv:2312.10997, vol. 2, no. 1, p. 32, 2023.

[7] C. Song and A. Raghunathan, “Information leakage in embedding models,” in Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, 2020, pp. 377–390.

[8] H. Li, M. Xu, and Y. Song, “Sentence embedding leaks more information than you expect: Generative embedding inversion attack to recover the whole sentence,” in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 14 022–14 040.

[9] J. Morris, V. Kuleshov, V. Shmatikov, and A. M. Rush, “Text embeddings reveal (almost) as much as text,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 12 448–12 460.

[10] X. Luo, T. Yu, and X. Xiao, “Prompt inference attack on distributed large language model inference frameworks,” in Proceedings ofthe 2025 ACM SIGSAC Conference on Computer and Communications Security, 2025, pp. 1739–1753.

[11] Y. Wang, M. Tang, N. Shen, S. Cui, and W. Wang, “Privacy risks of llm-empowered recommender systems: An inversion attack perspective,” in Proceedings of the Nineteenth ACM Conference on Recommender Systems, 2025, pp. 812–821.

[12] C. Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating noise to sensitivity in private data analysis,” in Theory of cryptography conference. Springer, 2006, pp. 265–284.

[13] O. Feyisetan, B. Balle, T. Drake, and T. Diethe, “Privacy-and utilitypreserving textual analysis via calibrated multivariate perturbations,” in Proceedings of the 13th international conference on web search and data mining, 2020, pp. 178–186.

[14] Z. Xu, A. Aggarwal, O. Feyisetan, and N. Teissier, “A differentially private text perturbation method using regularized mahalanobis metric,” in Proceedings of the Second Workshop on Privacy in NLP, 2020, pp. 7–17.

[15] M. Du, X. Yue, S. S. Chow, and H. Sun, “Sanitizing sentence embeddings (and labels) for local differential privacy,” in Proceedings of the ACM Web Conference 2023, 2023, pp. 2349–2359.

[16] S. Zhuang, B. Koopman, X. Chu, and G. Zuccon, “Understanding and mitigating the threat of vec2text to dense retrieval systems,” in Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, 2024, pp. 259–268.

[17] D. Seputis, Y. Li, K. Langerak, and S. Mihailov, “Rethinking the privacy of text embeddings: A reproducibility study of “text embeddings reveal (almost) as much as text”,” in Proceedings of the Nineteenth ACM Conference on Recommender Systems, 2025, pp. 822–831.

[18] C. Zhang, J. X. Morris, and V. Shmatikov, “Universal zero-shot embedding inversion,” arXiv preprint arXiv:2504.00147, 2025.

[19] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Extracting and composing robust features with denoising autoencoders,” in Proceedings of the 25th international conference on Machine learning, 2008, pp. 1096–1103.

[20] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, P.-A. Manzagol, and L. Bottou, “Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion.” Journal of machine learning research, vol. 11, no. 12, 2010.

[21] C. M. Stein, “Estimation of the mean of a multivariate normal distribution,” The annals of Statistics, pp. 1135–1151, 1981.

[22] S. Ramani, T. Blu, and M. Unser, “Monte-carlo sure: A black-box optimization of regularization parameters for general denoising algorithms,” IEEE Transactions on image processing, vol. 17, no. 9, pp. 1540–1554, 2008.

[23] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila, “Noise2noise: Learning image restoration without clean data,” in International Conference on Machine Learning. PMLR, 2018, pp. 2965–2974.

[24] G. Casella and R. Berger, Statistical inference. Chapman and Hall/CRC, 2024.

[25] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, “Gradient surgery for multi-task learning,” Advances in neural information processing systems, vol. 33, pp. 5824–5836, 2020.

[26] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee et al., “Natural questions: a benchmark for question answering research,” Transactions of the Association for Computational Linguistics, vol. 7, pp. 453–466, 2019.

[27] P. Bajaj, D. Campos, N. Craswell, L. Deng, J. Gao, X. Liu, R. Majumder, A. McNamara, B. Mitra, T. Nguyen et al., “Ms marco: A human generated machine reading comprehension dataset,” arXiv preprint arXiv:1611.09268, 2016.

[28] T. Diggelmann, J. Boyd-Graber, J. Bulian, M. Ciaramita, and M. Leippold, “Climate-fever: A dataset for verification of real-world climate claims,” arXiv preprint arXiv:2012.00614, 2020.

[29] A. Pal, L. K. Umapathi, and M. Sankarasubbu, “Medmcqa: A largescale multi-subject multi-choice dataset for medical domain question answering,” in Conference on health, inference, and learning. PMLR, 2022, pp. 248–260.

[30] J. Ni, C. Qu, J. Lu, Z. Dai, G. H. Abrego, J. Ma, V. Zhao, Y. Luan, K. Hall, M.-W. Chang et al., “Large dual encoders are generalizable

retrievers,” in Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, 2022, pp. 9844–9855.

[31] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” Journal of machine learning research, vol. 21, no. 140, pp. 1–67, 2020.

[32] Z. Li, X. Zhang, Y. Zhang, D. Long, P. Xie, and M. Zhang, “Towards general text embeddings with multi-stage contrastive learning,” arXiv preprint arXiv:2308.03281, 2023.

[33] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 2002, pp. 311–318.

[34] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “Squad: 100,000+ questions for machine comprehension of text,” in Proceedings of the 2016 conference on empirical methods in natural language processing, 2016, pp. 2383–2392.

[35] C.-Y. Lin, “Rouge: A package for automatic evaluation of summaries,” in Text summarization branches out, 2004, pp. 74–81.

[36] M. E. Houle, “Local intrinsic dimensionality i: an extreme-valuetheoretic foundation for similarity applications,” in International Conference on Similarity Search and Applications. Springer, 2017, pp. 64–79.

[37] Y. Chen, Q. Xu, and J. Bjerva, “Algen: Few-shot inversion attacks on textual embeddings via cross-model alignment and generation,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 24 330– 24 348.

[38] Y.-H. Huang, Y. Tsai, H. Hsiao, H.-Y. Lin, and S.-D. Lin, “Transferable embedding inversion attack: Uncovering privacy risks in text embeddings without model queries,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 4193–4205.

[39] D. Bollegala, S. Otake, T. Machide, and K.-i. Kawarabayashi, “A metric differential privacy mechanism for sentence embeddings,” ACM Transactions on Privacy and Security, vol. 28, no. 2, pp. 1–34, 2025.

[40] S. Soltanayev and S. Y. Chun, “Training deep learning based denoisers without ground truth data,” Advances in neural information processing systems, vol. 31, 2018.
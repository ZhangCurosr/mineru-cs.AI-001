# LISTENING FORWARD: NEXT PATCH EMBEDDING PRE-DICTION ENABLES SCALABLE AUDIO LEARNERS

Umberto Cappellazzo<sup>1</sup>, Xubo Liu<sup>2</sup>, Stavros Petridis<sup>1</sup>, Maja Pantic<sup>1</sup>

<sup>1</sup>Imperial College London <sup>2</sup>University of Surrey

u.cappellazzo@imperial.ac.uk

<sup></sup> Project Page <sup>§</sup> Code

## ABSTRACT

Self-supervised learning (SSL) has driven substantial progress in audio representation learning, though existing methods have increasingly relied on elaborate pre-training recipes to reach competitive performance. A markedly different pretraining philosophy underpins the most influential progress in language modeling and, more recently, in visual representation learning: rather than train encoders as static feature extractors, models are trained to predict the next element, a discrete token or a continuous embedding, from the preceding context. Autoregressive prediction thereby provides a unified pre-training interface that transfers across modalities, compelling the model to learn the underlying data distribution. We ask whether such a simple causal paradigm can yield strong audio learners, given that audio’s temporal structure makes autoregressive prediction of patch embeddings a natural fit. We introduce NAPE (Next-Audio-Patch-Embedding prediction), a self-supervised framework in which a causal Transformer predicts each next patch embedding of a log-mel spectrogram from the previous ones, using causal masking and stop-gradient as its sole training signal. The design is intentionally minimalist, avoiding reconstruction decoders, acoustic tokenizers, student-teacher setups, and auxiliary regularization losses. Across six audio and speech benchmarks, NAPE achieves state-of-the-art fine-tuning performance on several tasks, scales consistently across encoder sizes, and yields strong linear-probing results. NAPE also produces structured attention patterns without explicit supervision.

## 1 INTRODUCTION

Self-supervised learning (SSL) has emerged as a foundational paradigm for representation learning across modalities, delivering strong transfer performance without the cost of human annotations (Rad ford et al., 2018; Devlin et al., 2019; Brown et al., 2020; Chen & He, 2021; He et al., 2022; Oquab et al., 2023; Baevski et al., 2020; Kong et al., 2024; Chen et al., 2024). In audio, SSL methods largely adapt paradigms first developed in vision. Vision Transformer backbones (Dosovitskiy et al., 2020) are pre-trained on spectrograms either through masked-spectrogram modeling (Niizumi et al., 2022; Huang et al., 2022; Chong et al., 2023; Chen et al., 2022b; Dinkel et al., 2024; Niizumi et al., 2026) or through student-teacher distillation objectives (Ahmed et al., 2024; Chen et al., 2024; Alex et al., 2025; Yang et al., 2025b), with domain-specific augmentations tailored to the time-frequency structure of audio. Most of these methods pre-train on AudioSet (Gemmeke et al., 2017) and are evaluated primarily by fine-tuning performance on downstream classification tasks.

These paradigms, however, rely on increasingly elaborate pre-training recipes. Existing methods typically depend on reconstruction decoders that map latent features back to raw mel content (Huang et al., 2022; Chen et al., 2024), separately-trained acoustic tokenizers that supply discrete semantic targets (Chen et al., 2022b), teacher-student setups with exponential moving average (EMA)-updated encoders (Chen et al., 2024; Alex et al., 2025; Ahmed et al., 2024), auxiliary regularization losses that stabilize training (Fei et al., 2023; Alex et al., 2025), and multi-codebook vector quantisation (Yang et al., 2025b). Recently, a distinct line of work has begun to reshape SSL by shifting away from reconstruction toward the direct prediction of latent embeddings. This shift extends a paradigm long established in language modeling, where models are trained not as static feature extractors, but as predictive systems that model the data distribution through a single causal objective. Autoregressive prediction has thereby provided a unified pre-training interface across modalities, from discrete tokens in language to continuous embeddings in vision. Two variants of this philosophy have gained traction. Joint-embedding predictive approaches (Assran et al., 2023; Oquab et al., 2023; Fei et al., 2023; Bardes et al., 2024; Balestriero & LeCun, 2025; Yuksel et al., 2025; Huang et al., 2026; Wu et al., 2026) predict the latent embeddings of masked regions from a context view, produced by a target branch that is typically maintained via an EMA of the online encoder or stabilized by auxiliary regularization losses. While these methods are scalable and achieve strong performance, most of them rely on heavy heuristics (e.g., EMA, frozen layers, teacher-student architectures) to ensure training stability. Autoregressive next-embedding approaches (Teoh et al., 2025; Xu et al., 2025; Bredis et al., 2026; Yao et al., 2026; Maes et al., 2026b), in contrast, predict each latent embedding directly from the preceding ones, mirroring the causal next-token objective that drives modern large language models (Radford et al., 2018; Brown et al., 2020; Liu et al., 2024; Yang et al., 2025a). Both variants have quickly become a promising direction for representation learning in vision.

Despite this progress, predictive next-embedding methods remain absent from audio SSL, and joint-embedding predictive approaches themselves have seen only limited exploration in the audio and speech settings (Fei et al., 2023; Yuksel et al., 2025; Tuncay et al., 2025). The absence of any next-embedding autoregressive method for audio is particularly striking because audio is arguably the modality most naturally suited to this paradigm. Unlike images, whose 2D structure is approximately isotropic and admits no canonical ordering, audio signals unfold along a well-defined temporal axis and sequential structure is intrinsic to the signal, not imposed on it. Predicting the next patch of a spectrogram from the past ones mirrors both how acoustic events emerge in time and how modern language models learn from sequential data. Motivated by this natural alignment, we seek a next-embedding prediction framework for audio that is deliberately minimalist while still delivering state-of-the-art downstream performance across audio and speech benchmarks.

We therefore introduce NAPE (Next-Audio-Patch-Embedding prediction), a self-supervised framework that brings the causal next-embedding prediction paradigm to audio. Given a log-mel spectrogram, NAPE first applies a patch embedding layer that splits the spectrogram into non-overlapping patches and projects each of them into a d-dimensional embedding, producing a 2D grid of patch embeddings. Since the causal Transformer that follows operates on a 1D sequence, this grid must be traversed under a scanning order. This choice is a design axis specific to audio: unlike static images, spectrograms have a strong temporal axis and a qualitatively different frequency axis, so the order in which the grid is linearized determines both the causal context available at each prediction step and the structural inductive bias of the model. We consider four scanning strategies, depicted in Figure 2. A causal Transformer encoder then processes the linearized sequence, and a lightweight predictor head produces an estimate of the next patch embedding, analogous to next-token prediction in language modeling, but operating in continuous embedding space rather than over a discrete vocabulary. Prediction quality is measured by negative cosine similarity to the target patch embedding under a stop-gradient. This objective requires no reconstruction decoder, no acoustic tokenizer, no student-teacher setup, and no auxiliary regularization losses: the entire learning signal comes from the model’s ability to anticipate the next embedding in the sequence.

Extensive experiments on standard audio and speech benchmarks, including AudioSet (Gemmeke et al., 2017), ESC-50 (Piczak, 2015), Speech Commands V1 and V2 (Warden, 2018), and IEMO-CAP (Busso et al., 2008), show that NAPE achieves state-of-the-art performance on several tasks. NAPE exhibits favorable scaling properties across three encoder sizes: Small, Base, and Large, with 19/85/303 million parameters, respectively. Beyond fine-tuning, NAPE delivers strong linear-probing results despite being a purely predictive model whose objective is not aligned with linear separability, indicating that the learned representations remain discriminative even under strict feature-freeze evaluation. A qualitative analysis of NAPE’s attention patterns and embedding-space structure further confirms that NAPE learns meaningful, structured features from audio. Our main contributions are:

• We introduce NAPE, the first self-supervised audio framework built around causal next-patchembedding prediction, offering a substantially simpler alternative to reconstruction-based, masked-modeling, and joint-embedding predictive approaches for audio.

• Through systematic ablations, we identify the design axes that make next-embedding prediction work in the audio setting—scanning order, predictor head, prediction target, patch embedding layer, and the three key components of causality, prediction shift, and stop-gradient—and converge on an optimal configuration for the framework.

![](images/e745a3c49c306bbf170ed73e4eebd76eae43b89be670a9ae7e953adf91aaf726.jpg)  
Figure 1: Left: NAPE’s overview. The input spectrogram is split into patches and embedded into a sequence of embeddings. At each step, the model predicts the embedding of the next patch (red border) using only the embeddings of the preceding patches; patches at future positions are hidden by the causal attention mask. Middle: The NAPE pipeline: patch embeddings z are processed by the causal encoder h and predictor g to produce predictions zˆ, which are compared against the targets z under stop-gradient using a similarity function D (i.e., negative cosine similarity). Right: Encoder architecture (h): multiple stacked Transformer layers with pre-norm design, causal self-attention, LayerScale, and query-key normalization.

• Across six audio and speech benchmarks, NAPE achieves strong downstream results at three encoder scales, exhibits favorable scaling behavior, and delivers competitive linear-probing performance.

• We provide qualitative evidence that NAPE learns structured representations, with attention patterns that reason jointly about the current spectral context and the frequency-consistent temporal history, and embedding-space behavior that groups acoustically-similar patches without any explicit labels.

## 2 THE NAPE FRAMEWORK

In this section, we describe the NAPE framework in detail. NAPE is a self-supervised pre-training method that trains a causal Transformer to predict the embedding of the next patch of a log-mel spectrogram from the preceding ones, relying only on causal masking and stop-gradient. We describe NAPE’s main components and downstream adaptation in the next subsections. Figure 1 depicts the overall framework.

## 2.1 SPECTROGRAM PATCHIFICATION AND SCANNING ORDER

Input Representation. Given a raw waveform, we compute a log-mel spectrogram $x \in$ $\mathbb { R } ^ { 1 \times F \times T _ { \mathrm { f r a m e s } } }$ with $F$ frequency bins and $T _ { \mathrm { f r a m e s } }$ time frames. We split x into non-overlapping square patches of size $\bar { P } \times \bar { P } .$ , yielding a 2D grid of $N = T _ { F } \cdot T _ { T } ^ { \phantom { \dagger } }$ patches with $T _ { F } = \mathop { F } / \mathop { \bar { P } }$ and ${ T _ { T } } \mathrm { { = } } T _ { \mathrm { { f r a m e s } } } / P$ . Each patch is projected to a d-dimensional embedding by a patch embedding layer f, producing the sequence $\{ z _ { 1 } , \dotsc , z _ { N } \} \in \mathbb { R } ^ { N \times d }$ . We use a standard Conv2d patch embedding layer by default, but we also consider a deeper convolutional stem with batch normalization (convstem) (Xiao et al., 2021) and a speech-oriented stem that treats the mel axis as feature channels and applies temporal-only convolutions (speechstem) (Team et al., 2026). We refer to Section 3.2 for details and results about the choice of f.

Scanning Order. Transformer encoders operate on 1D sequences, so a 2D patch grid must be linearized before it can be processed. Under bidirectional attention, as used in prior audio SSL methods (Gong et al., 2021a; Huang et al., 2022; Chen et al., 2024), the choice of linearization is not a functional design choice: self-attention is permutation-equivariant given positional embeddings, so any consistent ordering yields the same representations. NAPE’s causal formulation, however, breaks this equivariance as the ordering determines which patches are “past” (visible to a given position) and which are “future” (to be predicted), and therefore imposes a substantive inductive bias on the model. Since spectrograms have a strong temporal axis and a qualitatively different frequency axis, unlike natural images, whose 2D structure is approximately isotropic, the choice of scanning order for causal prediction is particularly consequential. Thus, we consider four orderings, illustrated in Figure 2. Raster (left-to-right, bottom-to-top) sweeps time before advancing in frequency and is the standard patch ordering used in vision and audio Transformer models, where each prediction is conditioned on the entire past time axis at every frequency traversed so far. Time-major (bottom-to-top within each time column, then advance in time) sweeps frequency before advancing in time, so each prediction is conditioned on the full frequency profile of every past time step. Zigzag is an alternating-direction variant of raster: rows alternate left-to-right and right-to-left, so consecutive patches in the sequence remain spatially adjacent even across row transitions. This avoids the spatial discontinuity raster incurs when jumping from the end of one row to the start of the next. Diagonal sweeps patches along anti-diagonals of the grid, mixing time and frequency progression at every step and thereby encoding 2D spatial priors more uniformly than the other three orders.

![](images/0b5d0d30511f5439c86051216ca41dae2e0207cdbd3965640296a6778067aa47.jpg)  
Figure 2: Patch scanning orders. NAPE linearizes the 2D spectrogram patch grid into a 1D causal sequence in one of four ways: raster (left-to-right, bottom-to-top), time-major (bottom-to-top within each time column, then advance in time), zigzag (raster with alternating row directions), and diagonal (sweep by frequency-plus-time index). The numbers indicate the position of each patch in the resulting sequence.

## 2.2 NEXT-AUDIO-PATCH-EMBEDDING PREDICTION

Prediction Task. Given the embedding sequence $z = \{ z _ { 1 } , \dots , z _ { N } \}$ produced by f in the chosen scanning order, NAPE jointly trains an encoder h and a lightweight predictor head g so that, at each position $t ,$ the model produces an estimate of the next patch embedding $z _ { t + 1 }$ using only the previous patches:

$$
\hat { z } _ { t + 1 } = g ( h ( z _ { \leq t } ) ) ,\tag{1}
$$

where $z _ { \le t } = \{ z _ { 1 } , \ldots , z _ { t } \}$ denotes the patch embeddings at all positions up to and including t. The restriction to the past patches is enforced by a causal attention mask that prevents each position from attending to patches at later positions in the sequence. This is directly analogous to next-token prediction in language modeling, but operates in a continuous embedding space rather than over a discrete vocabulary.

Loss Function. Following SimSiam (Chen & He, 2021), we measure prediction quality by the negative cosine similarity between the target embedding $z _ { t + 1 }$ and the predicted embedding $\hat { z } _ { t + 1 } \mathrm { : \Omega }$

$$
\begin{array} { r } { \mathcal { D } ( z _ { t + 1 } , \hat { z } _ { t + 1 } ) = - \frac { z _ { t + 1 } } { \| z _ { t + 1 } \| _ { 2 } } \cdot \frac { \hat { z } _ { t + 1 } } { \| \hat { z } _ { t + 1 } \| _ { 2 } } , } \end{array}\tag{2}
$$

where $\lVert \cdot \rVert _ { 2 }$ is the ℓ<sub>2</sub>-norm. Applying stop-gradient (stopgrad) on the target and averaging over all valid prediction positions yields the NAPE objective:

$$
\mathcal { L } = \frac { 1 } { N - 1 } \sum _ { t = 1 } ^ { N - 1 } \mathcal { D } ( \mathfrak { s t o p g r a d } ( z _ { t + 1 } ) , \ : \hat { z } _ { t + 1 } ) .\tag{3}
$$

The stopgrad operator treats the target embedding as a constant, so gradients flow only through the predicted side. Cosine similarity is magnitude-invariant, which prevents the trivial solution of shrinking both sides of the objective toward zero norm. We compare cosine similarity against alternative similarity functions in Section 3.2 and find it to yield the optimal results.

Causal Transformer Encoder (h). We use a Vision Transformer backbone (Dosovitskiy et al., 2020; Wang et al., 2026) with pre-norm design and a causal attention mask during pre-training (see Figure 1, right panel). For stability at depth, we adopt Rotary Position Embedding (RoPE) (Su et al., 2024) applied independently along the frequency and time axes, LayerScale (Touvron et al., 2021) on residual branches, and parameter-free query-key normalization (Henry et al., 2020) on the per-head query and key projections. We instantiate three configurations of increasing size to test NAPE’s scalability: Small (d = 384, 12 layers, 6 heads; ∼19M parameters), Base $( d = 7 6 8 .$ , 12 layers, 12 heads; ∼85M parameters), and Large (d = 1024, 24 layers, 16 heads; ∼303M parameters).

Predictor Head (g). The predictor head g decouples the representation space of h from the space in which predictions are made. Predicting directly with the encoder output forces the encoder to place its representations in the same space as the prediction targets, which can restrict feature richness. This asymmetric predictor design, together with the stop-gradient applied to the target branch, gives NAPE a structural resemblance to SimSiam (Chen & He, 2021): representation collapse is avoided without contrastive negatives or an EMA teacher, relying instead on the asymmetry between the two branches. NAPE differs from SimSiam in what the branches encode: SimSiam compares two augmented views of the same input under a symmetric encoder, whereas NAPE compares a prediction of the next patch to that patch’s actual embedding within a single autoregressive sequence, replacing view augmentation and siamese symmetry with temporal prediction as the source of the learning signal. We study multiple predictor styles in Section 3.2.

NAPE’s Key Components. Three complementary mechanisms prevent NAPE from converging to trivial solutions and together define its training regime. (i) causality: enforced by the causal attention mask that restricts each position to attend only to prior patches, it prevents the encoder from attending to the target patch when producing its prediction, blocking the trivial identity mapping. (ii) The prediction shift between input and target positions ensures that the model at position t predicts the embedding at position t + 1 rather than the embedding at its own position, so causality alone cannot be side-stepped by copying the current input through. (iii) stop-gradient on the target embedding prevents gradients from flowing through both sides of the loss simultaneously, which would otherwise allow the encoder to collapse all embeddings toward a shared constant. Together, these three ingredients constitute the core of NAPE’s selfsupervised recipe, and we analyze the individual contribution of each in Section 3.2. The pseudocode of NAPE’s pre-training is given in Algorithm 1.

```julia
Algorithm 1 NAPE pre-training algorithm.
#f: Patch Embedding Layer
#h: Causal Transformer Encoder
#g: Predictor
for x in loader: # x: [B,1,F,T_frames]
z = f(x) # embeddings [B,T,D]
z_hat = g(h(z)) # predictions [B,T,D]
loss = D(z, z_hat)
loss.backward(); # update(f,h,g)
def D(z, z_hat):
target = z[:,1:, :].detach() # stop-grad
pred = z_hat[:,:-1, :] # AR shift
pred = normalize(pred, dim=-1)
target = normalize(target, dim=-1)
return -(pred target).sum(-1).mean()
```

## 3 EXPERIMENTS

In this section, we carry out extensive experiments to evaluate NAPE at different scales and configurations. (1) We first perform several ablations on key design choices to converge on the best NAPE configuration (Section 3.2); (2) we then compare that configuration against state-of-the-art methods at multiple scales (Section 3.3 and 3.4); (3) we assess linear separability of the learned features (Section 3.5), and (4) we finally analyze how NAPE organizes acoustic information through its attention patterns and the structure of its learned embeddings (Section 3.6).

## 3.1 EXPERIMENTAL SETUP

Pre-training Data. We pre-train NAPE on AudioSet (Gemmeke et al., 2017) without labels, combining the unbalanced and balanced training splits. We obtained and processed 1,964,222 clips from the unbalanced split, 20,961 clips from the balanced split, and 18,900 clips from the evaluation split, consistent with prior work (Chen et al., 2022b; 2024; Alex et al., 2025). All input waveforms are resampled to mono at 16 kHz and converted into log-mel spectrograms with 128 mel bands using a 25 ms Hanning window and 10 ms hop size. A 10-second clip yields a spectrogram of size

$1 \times 1 2 8 \times 1 0 0 8$ (channel, frequency, time), corresponding to $8 \times 6 3 = 5 0 4$ non-overlapping $1 6 \times 1 6$ patches per clip (as in (Gong et al., 2021a; Chen et al., 2022b; 2024)).

Downstream Benchmarks. For downstream evaluation, we fine-tune on AudioSet-2M (AS-2M, unbalanced) and AudioSet-20K (AS-20K, balanced), applying the weighted sampling strategy of Huang et al. (2022) on AS-2M to mitigate class imbalance. We further evaluate on ESC-50 (Piczak, 2015), a 50-class environmental sound classification benchmark of 2000 clips at 5s each, using 5-fold cross-validation; Speech Commands V1 and V2 (KS1, KS2) (Warden, 2018), keyword spotting tasks with 12 and 35 classes respectively; and IEMOCAP (ER) (Busso et al., 2008), a 4-class speech emotion recognition benchmark with 5-fold cross-validation. Together these tasks span both audio and speech domains. We report mean average precision (mAP) on the multi-label AudioSet tasks and top-1 classification accuracy on the single-label tasks; for ESC-50 and IEMOCAP we report the mean across cross-validation folds.

Pre-training Details. We pre-train with the AdamW optimizer (Loshchilov & Hutter, 2017) using a base learning rate of $5 \times \mathrm { \bar { 1 } 0 ^ { - 3 } }$ with cosine decay, a weight decay of 0.05, $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ and a batch size of 256 for the Small and Base configurations, and 128 for NAPE Large. We pre-train NAPE Small and NAPE Large for 25 epochs and NAPE Base for 30 epochs, with a warmup ratio of 10%. All models are trained using HuggingFace Trainer with distributed data parallelism (DDP) on NVIDIA L40s GPUs (46GB).

Fine-tuning Details. For each downstream task, we initialize from the pre-trained encoder and attach a linear classifier on top of mean-pooled patch tokens. We disable the causal attention mask during fine-tuning so that attention is bidirectional over the full patch sequence. We provide an ablation study on the optimal pooling method and whether to use causal/bidirectional attention in the Appendix D. We use AdamW with $\beta _ { 2 } ~ = ~ 0 . 9 9 9$ , cosine learning-rate decay, and layer-wise learning-rate decay (Clark et al., 2020; Bao et al., 2021). We use binary cross-entropy for AudioSet and soft-target cross-entropy for the single-label tasks. Following prior work (Gong et al., 2021a; Chen et al., 2024; 2022b), we apply a standard augmentation stack (SpecAugment (Park et al., 2019), Mixup (Zhang et al., 2017), CutMix (Yun et al., 2019), DropPath (Huang et al., 2016), temporal roll, additive noise, and label smoothing (Szegedy et al., 2016)). For AS-2M and AS-20K we additionally maintain an exponential moving average (EMA) (Polyak & Juditsky, 1992) of the fine-tuning weights (decay 0.99995 and 0.999 respectively); no EMA is used for the other tasks. Full task-specific hyperparameters are reported in the Appendix C.

Linear Probing Details. For linear probing, we freeze the pre-trained encoder and train only the classifier plus a preceding LayerNorm (Ba et al., 2016). All augmentations are disabled.

Baselines. We primarily compare against recent in-domain self-supervised methods, including Audio-MAE (Huang et al., 2022), BEATs (Chen et al., 2022b), A-JEPA (Fei et al., 2023), ASiT (Ahmed et al., 2024), EAT (Chen et al., 2024), SSLAM (Alex et al., 2025), and SPEAR (Yang et al., 2025b). We also include results from out-of-domain and in-domain supervised pre-training. For all baseline methods, fine-tuning results are taken from the original papers.

## 3.2 NAPE’S OPTIMAL CONFIGURATION

In this section, we ablate the main design axes of NAPE to identify its optimal configuration. Unless otherwise specified, all ablations use the NAPE Base model and the raster scanning order. Each ablation modifies a single design axis at a time, holding the rest of the configuration fixed at NAPE’s defaults.

Prediction Shift/Stop-gradient/Causality. Table 1 disentangles the three key mechanisms of NEPA: the prediction shift, the stop-gradient on the target, and the causal attention mask. Removing either the prediction shift or the stop-gradient causes pre-training to diverge, matching the analysis in Section 2.2: without the shift, the model at position t can trivially satisfy the objective by copying its own input embedding through; without the stop-gradient, gradients flow through both sides of the loss and the encoder collapses all embeddings toward a shared constant.

Removing the causal mask, in contrast, does not diverge: the prediction shift alone continues to define a nominal prediction target, but degrades downstream performance sharply, with the largest drops on the audio benchmarks (−7.8 mAP on AS-2M, −14.3 on AS-20K, and −25.4 points on ESC-50; smaller but consistent drops on the speech tasks). Without causality, the encoder can attend to the target patch while producing

Table 1: Ablation on main NAPE’s design elements: prediction shift, stop-gradient, and causal objective.
<table><tr><td>Pred stop causal</td><td></td><td></td><td>shift grad mask AS-2M AS-20K ESC-50 KS1 KS2 ER</td><td>Audio Tasks</td><td>Speech Tasks</td></tr><tr><td>x √</td><td>√ x</td><td>√ √</td><td></td><td>Diverge Diverge</td><td></td></tr><tr><td>√ √</td><td>√ √</td><td>x √</td><td>41.8 49.6</td><td>24.8 39.1</td><td>68.9 96.1 97.3 57.0 94.2 97.9 98.8 64.9</td></tr></table>

its prediction, so the objective is trivially satisfied by a near-identity mapping that routes each target back to its predicted position (the loss saturates near −1 within a few thousand steps of pre-training, see Appendix F): the loss decreases during pre-training, but the encoder is not forced to learn useful structure. All three mechanisms are therefore jointly necessary, none can be dropped without either destabilizing training or degrading the learned representations to a degree that fine-tuning cannot recover.

Patch Embedding Layer $f .$ Table 2 compares three patchifiers, all configured to produce the same sequence length (504 tokens per 10s clip). The default Conv2d is a single strided convolution with kernel and stride $1 6 \times 1 6 ,$ which treats time and frequency as symmetric 2D axes and matches the standard ViT design (Dosovitskiy et al., 2020). We also consider two alternatives

Table 2: Ablation on the patch embedding layer.
<table><tr><td>Patch Emb. Layer</td><td>Audio Tasks Speech Tasks AS-2M AS-20K ESC-50 KS1 KS2 ER</td></tr><tr><td></td><td>97.4 98.3 63.6</td></tr><tr><td>Convstem 46.7 Speechstem 47.6</td><td>34.3 89.1</td></tr><tr><td>33.1</td><td>88.4 98.1 98.9 63.0</td></tr><tr><td>Conv2d 49.6 39.1</td><td>94.2 97.9 98.8 64.9</td></tr></table>

motivated by observations from computer vision and speech literature. The Convstem, following Xiao et al. (2021), replaces the single Conv2d with four $3 \times 3$ stride-2 convolutions interleaved with batch normalization with the channel count growing progressively toward $d ;$ deeper convolutional stems have been shown to stabilize optimization and improve downstream performance in ViT-based image models. The Speechstem, inspired by the audio front-end of speech-oriented models such as Gemma 3n and Gemma 4 (Team et al., 2026), flattens the mel axis into feature channels and applies temporal-only $3 \times 3$ convolutions with a temporal downsampling factor of 2, producing a 1D time-only sequence of 504 tokens that reflects the frame-based processing typical of speech recognition systems.

Both alternatives underperform Conv2d across all benchmarks, with the largest gaps on AudioSet: on AS-20K, Convstem loses 4.8 mAP points and Speechstem loses 6.0 mAP spoints relative to Conv2d. Convstem’s added non-linearity and speechstem’s temporal-first inductive bias therefore do not translate into gains for our causal spectrogram prediction objective, treating frequency and time as symmetric 2D axes at the patchification stage is important for the pretraining signal that NAPE exploits. Conv2d remains NAPE’s default.

Predictor g. Table 3 compares four predictor variants: no predictor (encoder output used directly as the prediction), a two-layer MLP (2-MLP; Linear(d, d) → GELU → Linear(d, d)), a SimSiamstyle (Chen & He, 2021) three-layer predictor with intermediate LayerNorms (Linear $( d , d ) \mathrm { ~  ~ { ~  ~ \mathrm { ~ L N ~ } ~  ~ \mathrm { ~ G E L U } ~  ~ } ~ }$ Linear $( d , d ) \quad \to \quad \mathrm { L N } \quad \to \quad \mathrm { G E L U } \quad \to \quad$

Table 3: Ablation on the predictor-style variants.
<table><tr><td>Predictor Style</td><td>#Par.</td><td>Audio Tasks AS-2M AS-20K ESC-50 KS1 KS2 ER</td><td></td><td>Speech Tasks</td></tr><tr><td>None</td><td></td><td>48.7 37.8</td><td>93.3</td><td>98.1 98.8 64.2</td></tr><tr><td>2-MLP</td><td>1.2M</td><td>49.4</td><td>38.5 93.6</td><td>98.0 98.8 64.2</td></tr><tr><td>Transformer 14.2M</td><td></td><td>49.2</td><td>38.4 93.0</td><td>98.2 98.7 65.0</td></tr><tr><td>SimSiam</td><td>1.8M</td><td>49.6</td><td>39.1 94.2</td><td>97.9 98.8 64.9</td></tr></table>

Linear(d, d)), and a Transformer predictor in the style of JEPA-family methods: a 2-layer causal Transformer (16 heads) operating in the encoder’s hidden dimension (we apply a causal mask to prevent future leaking as for the encoder). Adding any predictor improves over using the encoder output directly, confirming that decoupling the representation and prediction spaces benefits the learned features. Among the three predictor styles, the SimSiam variant performs best on the audio benchmarks and on IEMOCAP, while remaining on par on the keyword-spotting tasks. Notably, the lightweight SimSiam predictor outperforms the Transformer predictor despite being nearly 8× smaller in parameters, indicating that additional predictor capacity is not the bottleneck for NAPE, consistent with the observation in the siamese self-supervised setting (Chen & He, 2021) that a compact MLP predictor suffices when combined with a well-designed encoder and stop-gradient target. The SimSiam-style predictor is NAPE’s default.

![](images/b05560e4eb89fedfb5a9ee6947c728a3dfb3430cf5c61f71a437d9ecab7fcb01.jpg)  
Figure 3: NAPE’s performance across four scan orders on six benchmarks.

Patch Scannning Variants. Figure 3 compares the four scanning orders introduced in Section 2.1 (we use the 2-MLP predictor style). Diagonal, raster, and zigzag all perform comparably well across the six benchmarks, with diagonal and raster slightly ahead of zigzag on most tasks. In contrast, time-major consistently underperforms the other three orders across all tasks. We interpret this pattern as reflecting the temporal structure of audio: diagonal, raster, and zigzag all advance in time as the causal sequence progresses, allowing the model to accumulate temporal context in a way that matches how acoustic events unfold. Time-major, instead, exhausts each frequency column before advancing in time, so predictions early in the sequence are conditioned on rich instantaneous spectra but only limited temporal context, a mismatch with the temporal nature of audio classification. Given their superior performance, we retain both raster and diagonal in the main comparison against state-of-the-art methods in Section 3.4.

Prediction Target. Table 4 compares three choices for the target of the auto-regressive prediction: (i) the patch embedding $z _ { t + 1 }$ produced by the shared embedding layer f (the default); (ii) the raw mel content of the next patch i.e., the flattened mel-spectrogram values inside the patch, following the target formulation used by masked reconstruction methods such as Audio

Table 4: Ablation on NAPE’s target to predict.
<table><tr><td>Predicted Target</td><td colspan="4">Audio Tasks Speech Tasks AS-2M AS-20K ESC-50 KS1 KS2 ER</td></tr><tr><td>1st enc. layer</td><td></td><td>Diverge</td><td></td><td></td></tr><tr><td>Raw Mel</td><td>49.7</td><td>38.0</td><td>94.8</td><td>97.7 98.6 64.2</td></tr><tr><td>Patch embed</td><td>49.6</td><td>39.1</td><td>94.2</td><td>97.9 98.8 64.9</td></tr></table>

MAE (Huang et al., 2022); (iii) the output of thefirst encoder layer, in the style of JEPA methods that use deeper encoder features as prediction targets. For the raw mel variant, we add a linear projection on top of the predictor g to map its output from d dimensions to the raw patch dimensionality.

Patch embedding and raw mel yield comparable results across all benchmarks, indicating that both are valid target choices for NAPE. Using the first encoder layer as target, however, causes pre-training to diverge: the target itself depends on the encoder being trained, and stop-gradient alone is insufficient to prevent the encoder from collapsing both sides of the loss to a shared constant. JEPA-family methods circumvent this instability with additional regularization such as EMA teachers (Assran et al., 2023; Fei et al., 2023), variance-covariance regularizers such as VISReg (Wu et al., 2026) and sketched isotropic gaussian regularizers such as SIGReg (Balestriero & LeCun, 2025), which we do not employ here. Since the patch embedding approach on average performs better than raw mel and it is adopted in (Xu et al., 2025) as well, we retain the patch embedding as NAPE’s default.

Similarity Function (D). In Table 5 we compare four choices for the similarity function D in Eq. 2: the negative cosine similarity (NAPE’s default), a soft cross-entropy formulation that treats prediction and target as distributions over the d channels after applying a softmax (as in Chen & He (2021)), and the $\ell _ { 1 }$ and $\ell _ { 2 }$ distances between the two d-dimensional vectors. Both $\ell _ { 1 }$ and $\ell _ { 2 }$ cause pre-training to diverge: un-

Table 5: Ablation on the similarity function.
<table><tr><td>Similarity Function</td><td colspan="4">Audio Tasks Speech Tasks AS-2M AS-20K ESC-50 KS1 KS2 ER</td></tr><tr><td>L1 L2</td><td colspan="4">Diverge</td></tr><tr><td>Cross-entropy Cosine</td><td>48.8 49.6</td><td>37.4 39.1</td><td>Diverge 93.6 94.2</td><td>98.0 98.7 64.5 97.9 98.8 64.9</td></tr></table>

![](images/952a6aefd251cb8f040f0276f5d67ca3299b931661e24c0c6ec721037206e5ec.jpg)  
Figure 5: NAPE’s results at different scales under raster and diagonal scan orders.

like cosine similarity, which is magnitude-invariant, these distance-based objectives can be trivially minimized by shrinking the norm of both predicted and target embeddings toward zero, a form of representation collapse in which the encoder outputs converge to a shared low-magnitude constant. The cross-entropy variant trains stably (softmax normalization implicitly bounds the target magnitude) but underperforms cosine on all benchmarks except the keyword-spotting tasks, where the two are on par. Cosine similarity, combining magnitude-invariance with a directionally informative loss signal, yields the best overall results and is retained as NAPE’s default.

## NAPE’s Optimal Configuration

(i) Conv2d patch embedding layer, (ii) raster/diagonal scanning order, (iii) SimSiam-style predictor with three-layer MLP and intermediate LayerNorms, (iv) patch embedding as the prediction target, and (v) negative cosine similarity loss with stop-gradient on the target branch.

Additional Ablation Results. We refer to Appendix D for additional ablations studies on: 1) the use of normalization layers (LayerNorm vs RMSNorm), 2) freezing/unfreezing the patch embedding layer, 3) the optimal positional encoding (absolute encoding vs RoPE), 4) the optimal pooling method during fine-tuning, 5) the optimal attention type during fine-tuning (causal vs bidirectional), (6) the additional use of random masking during pre-training, and 7) NAPE’s performance in terms of different compute budgets.

## 3.3 SCALING NAPE

Figure 5 shows downstream performance for the raster and diagonal variants of NAPE at three encoder scales: Small (∼19M parameters, NAPE-S), Base (∼85M, NAPE-B), and Large (∼303M, NAPE-L). NAPE scales positively across the board: on every one of the six benchmarks, NAPE-L improves over NAPE-B, which in turn improves over the small version. While this improvement tends to diminish as we scale from the base to the large model, we observe consistent scaling gains for most of the tasks, while the keyword-spotting tasks show smaller absolute gains, reflecting their already-saturated accuracy levels. Raster and diagonal track each other closely at the Small scale, with essentially identical performance across all six tasks. At the Base and Large scales, diagonal slightly outperforms raster on most benchmarks, while raster achieves the strongest result on AS-2M with NAPE-L reaching 50.18 mAP. In Figure 4, we compares NAPE (raster) and Audio-MAE (Huang et al., 2022) at three encoder scales on AS-2M and AS-20K. NAPE outperforms Audio-MAE at every scale, with particularly large margins at the Small size (+2.6 mAP on AS-2M, +4.1 mAP on AS-20K). NAPE also exhibits more favorable scaling behavior from Base to Large.

![](images/c3a0cbb89fd432f3a0f2edc8277d201ba2862384a4af5dadb36e74f5ae8147ee.jpg)  
Figure 4: Scaling comparison between Audio-MAE and NAPE, raster.

## 3.4 COMPARISON WITH STATE-OF-THE-ART

We compare NAPE against prior audio pre-training methods in Table 6. We use the best configuration identified in Section 3.2, reporting both raster and diagonal variants at the Base and Large scales. NAPE-B with diagonal scan delivers strong performance across all six benchmarks, matching or exceeding every self-supervised competitor of comparable size. When we scale it further, NAPE-L raster attains results on par with the strongest baseline, SSLAM (Alex et al., 2025), tying it on AS-2M (50.2 mAP) and approaching it closely on AS-20K (40.5 vs 40.9 mAP) and ESC-50 (96.0 vs 96.2%). What is notable about these results is the simplicity of underlying NAPE’s recipe. Unlike SSLAM, which relies on audio-mixture supervision, a student-teacher architecture, and a reconstruction decoder, NAPE requires none of these ingredients: its pre-training objective consists of a stop-gradient and a negative cosine similarity between the predicted next patch embedding and its target. NAPE also outperforms A-JEPA (Fei et al., 2023) on 5 benchmarks despite A-JEPA relying on an auxiliary target encoder updated via EMA, predicting multiple masked patches in parallel rather than causally, and adopting a regularized masking strategy during fine-tuning. Finally, NAPE transfers particularly well to speech tasks: on IEMOCAP, NAPE-L reaches 68.0% accuracy, a +3.5-point improvement over the strongest baseline result reported at any scale $( \mathrm { B E A T s } _ { i t e r 3 }$ at 64.5%), suggesting that NAPE learns representations that generalize well beyond acoustic-event classification.

Table 6: Comparison with audio methods on audio and speech downstream tasks. IN, AS, and LS denote the ImageNet, AudioSet, and LibriSpeech datasets, respectively. TI denotes the 400M text-image pairs for CLIP pre-training. We gray-out the models and results with additional supervised training on external datasets. Best results are in bold, second-best are underlined.
<table><tr><td rowspan="2">Model</td><td rowspan="2">#Par.</td><td rowspan="2">Pre-train Data</td><td colspan="3">Audio Tasks</td><td colspan="3">Speech Tasks</td></tr><tr><td>AS-2M</td><td>AS-20K</td><td>ESC-50</td><td>KS1</td><td>KS2</td><td>ER</td></tr><tr><td>Out-of-domain Supervised Pre-training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PSLA (Gong et al., 2021b)</td><td>14M</td><td>IN</td><td>44.4</td><td>31.9</td><td></td><td></td><td>96.3</td><td></td></tr><tr><td>AST (Gong et al., 2021a)</td><td>86M</td><td>IN</td><td>45.9</td><td>34.7</td><td>88.7</td><td>95.5</td><td>98.1</td><td>56.0</td></tr><tr><td>HTS-AT (Chen et al., 2022a)</td><td>31M</td><td>IN</td><td>47.1</td><td></td><td></td><td></td><td>98.0</td><td></td></tr><tr><td>Audio-CLIP (Guzhov et al., 2022)</td><td>93M</td><td>TI+AS</td><td>25.9</td><td></td><td>96.7</td><td></td><td></td><td></td></tr><tr><td>In-domain Supervised Pre-training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AST (Gong et al., 2021a)</td><td>86M</td><td>IN+AS</td><td>45.9</td><td></td><td>95.6</td><td></td><td>97.9</td><td></td></tr><tr><td>HTS-AT (Chen et al., 2022a)</td><td>31M</td><td>IN+AS</td><td>47.1</td><td></td><td>97.0</td><td></td><td></td><td></td></tr><tr><td>Audio-MAE (Huang et al., 2022)</td><td>86M</td><td>AS</td><td></td><td></td><td>97.4</td><td></td><td></td><td></td></tr><tr><td>Self-Supervised Pre-training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SS-AST (Gong et al., 2022a)</td><td>89M</td><td>AS+LS</td><td></td><td>31.0</td><td>88.8</td><td>96.0</td><td>98.0</td><td>59.6</td></tr><tr><td>MAE-AST (Baade et al., 2022)</td><td>86M</td><td>AS+LS</td><td></td><td>30.6</td><td>90.0</td><td>95.8</td><td>97.9</td><td>59.8</td></tr><tr><td>CAV-MAE (Gong et al., 2022b)</td><td>86M</td><td>IN+AS</td><td>44.9</td><td>34.2</td><td></td><td></td><td></td><td></td></tr><tr><td>Audio-MAE (Huang et al., 2022)</td><td>86M</td><td>AS</td><td>47.3</td><td>37.1</td><td>94.1</td><td>96.9</td><td>98.3</td><td></td></tr><tr><td>Audio-MAE L (Huang et al., 2022)</td><td>304M</td><td>AS</td><td>47.4</td><td>37.7</td><td></td><td></td><td></td><td></td></tr><tr><td>data2vec (Baevski et al., 2022)</td><td>94M</td><td>AS</td><td></td><td>34.5</td><td></td><td></td><td></td><td></td></tr><tr><td>MaskSpec (Chong et al., 2023)</td><td>86M</td><td>AS</td><td>47.1</td><td>32.3</td><td>89.6</td><td>1</td><td>97.7</td><td></td></tr><tr><td> $\mathbf { B E A T s } _ { i t e r 3 }$  (Chen et al., 2022b)</td><td>90M</td><td>AS</td><td>48.0</td><td>38.3</td><td>95.6</td><td>97.7</td><td>98.3</td><td>64.5</td></tr><tr><td>A-JEPA (Fei et al., 2023)</td><td>86M</td><td>AS</td><td>48.6</td><td>38.4</td><td>96.3</td><td>97.7</td><td>98.5</td><td></td></tr><tr><td>ASiT (Ahmed et al., 2024)</td><td>86M</td><td>AS</td><td>48.0</td><td>38.6</td><td>95.3</td><td>98.2</td><td>98.9</td><td></td></tr><tr><td>EAT (Chen et al., 2024)</td><td>88M</td><td>AS</td><td>48.6</td><td>40.2</td><td>95.9</td><td></td><td>98.3</td><td></td></tr><tr><td>SSLAM (Alex et al., 2025)</td><td>88M</td><td>AS</td><td>50.2</td><td>40.9</td><td>96.2</td><td>98.8</td><td>98.1</td><td></td></tr><tr><td>SPEARa Large (Yang et al., 2025b)</td><td>327M</td><td>AS</td><td>49.7</td><td>39.3</td><td>-</td><td>一</td><td></td><td>-</td></tr><tr><td>NAPE-B raster</td><td>85M</td><td>AS</td><td>49.6</td><td>39.1</td><td>94.2</td><td>97.9</td><td>98.8</td><td>64.9</td></tr><tr><td>NAPE-B diagonal</td><td>85M</td><td>AS</td><td>49.7</td><td>39.2</td><td>94.8</td><td>97.9</td><td>98.6</td><td>67.1</td></tr><tr><td>NAPE-L raster</td><td>303M</td><td>AS</td><td>50.2</td><td>40.5</td><td>96.0</td><td>97.9</td><td>98.8</td><td>68.0</td></tr><tr><td>NAPE-L diagonal</td><td>303M</td><td>AS</td><td>50.0</td><td>40.4</td><td>96.2</td><td>98.2</td><td>98.9</td><td>68.8</td></tr></table>

## 3.5 LINEAR PROBING RESULTS

Which Layer to Probe? Linear probing measures the linear separability of the features produced by the pretrained encoder, treating the encoder as a fixed feature extractor and training only a linear classification head. Following recent observations that the most classification-relevant features in deep Transformer models often lie in intermediate rather than final layers (Skean et al., 2025; Bolya et al., 2026), we begin our linear probing study by measuring downstream performance as a function of encoder depth.

Figure 6 reports AS-20K mAP obtained by linearly probing each layer of NAPE-S, NAPE-B, and NAPE-L, using the best raster configuration identified in Section 3.2. Across all three scales, the best probing layer lies at roughly the middle of the encoder: layer 2 for NAPE-S, layer 6 for NAPE-B, and layer 11 for NAPE-LL. Beyond this mid-network optimum, performance declines steadily toward the final layer, dropping by roughly 3-5 mAP points. This pattern is consistent with the interpretation that the top layers of a NAPE-pretrained encoder specialize for the next-patch-embedding prediction objective, while the mid-layers retain more general and classification-relevant information.

![](images/51673512b839aeb382bea50f375e364df1cefe29cd3e703f188c19a680831bf2.jpg)  
Figure 6: Layer-wise linear probing analysis.

Linear Probing Across Scales and Tasks. We then use the best probing layer identified per model to compare NAPE-S, NAPE-B, and NAPE-L on AS-2M, AS-20K, and ESC-50. Table 7 reports the results. NAPE scales positively under linear probing: larger models yield stronger probes on every task, from AS-2M (+3.9 mAP from small to large) to ESC-50 (+3.7 in accuracy). This trend reinforces the scaling behavior observed under fine-tuning (Section 3.3). At the same time, the absolute linear-probing numbers are noticeably lower than their fine-tuning counterparts. This gap is expected for prediction-based self-supervised methods: the pretraining objective encourages the encoder to learn features that support the next-patch prediction task, which do not necessarily align with the linear separability needed for classification.

Table 7: Linear probing results.
<table><tr><td colspan="3">Model Layer AS-2M AS-20K ESC-50</td></tr><tr><td>Small</td><td>2nd 23.2</td><td>18.9 79.8</td></tr><tr><td>Base</td><td>6th 25.0</td><td>19.7 81.7</td></tr><tr><td>Large 11th</td><td>27.1 20.4</td><td>83.5</td></tr></table>

## 3.6 QUALITATIVE RESULTS

To gain insight into what NAPE learns, we complement the quantitative benchmarks with two qualitative analyses of a pretrained NAPE-L (raster) on the AudioSet evaluation set. Both use the model after pretraining, with no fine-tuning. More qualitative results can be found in the Appendix G.

Prediction Fidelity. We measure the cosine similarity between the predicted embeddings $\hat { z } _ { t + 1 }$ and the true patch embeddings $z _ { t + 1 }$ . Figure 7 (top left) shows the similarity averaged across 500 held-out AudioSet clips, and the top middle and right panels report the same measurement for two individual clips: NAPE predicts the next patch embedding accurately almost everywhere on the grid, with similarity close to the ceiling of 1.0 both on average and per clip. The remaining low-similarity regions have clear structural explanations. The very first patch has the lowest similarity, since no previous context is available for the prediction to condition on. The first mel row is harder on average, since it combines limited past context with low-frequency patches. Finally, the rightmost patches show slightly lower similarity because the last time columns correspond to zero-padded frames appended to reach the target clip length. Away from these boundary regions, NAPE satisfies its pre-training objective on unseen audio.

Attention and Embedding Analysis. To understand how NAPE arrives at its predictions, we select a query patch (marked in red in Figure 7, bottom left) and analyze the attention and embedding structure it induces. The attention map (bottom middle), which conveys the aggregated attention from the query position to every other patch averaged over all layers and heads, reveals a highly structured pattern with two distinct components: NAPE attends strongly to the current time column, integrating the full spectral profile of the current moment, and to same-mel-frequency patches earlier in the clip, tracking the temporal evolution of the frequency it is about to predict. The embedding-similarity map (bottom right), which compares the predicted embedding $\hat { z } _ { t + 1 }$ against every actual patch embedding in the clip, shows that $\hat { z } _ { t + 1 }$ is most similar to patches that share acoustic structure and energy with the query, regions of the spectrogram carrying comparable spectral content, with similarity gradually decaying at the temporal and spectral extremes. This grouping into coherent acoustic components emerges without any explicit labels or region annotations, suggesting that despite being trained only on a local next-patch objective, NAPE develops representations that capture the broader acoustic structure of the clip.

![](images/5e2b4c57b4440e8e999f774976c2fbd7d029644369f27694fa79f4de41ef44a6.jpg)

![](images/28ea6f180682bcfc5bf1fb42c65784f32764d650572be83a41abdf309d614fd0.jpg)

![](images/f181c9cc3864fab1124995caf05c23f7e72dbb69d32b00e543b02511f85a2775.jpg)

![](images/4d280584430e567f6afcc8eeaf9305375373378c4adbfc24940540acbfcd352b.jpg)

![](images/971aa9b3b50840c911990ec0fd2b0674dd34f391dca6eea951923d443fb9005b.jpg)

![](images/a94fce58cc267338a071aa4560db7ea3bdef6cacb100745d6d4b7143f9cb4532.jpg)  
Figure 7: Top: Prediction Quality Analysis. We report the cosine similarity between the predicted and true patch embeddings, averaged over 500 audio clips (left) and for two individual clips (middle, right). Bottom: Attention/Embedding Analyses. Left: selected query patch; middle: attention map showing the patches NAPE attends to when predicting the next patch; right: embedding-similarity map between the predicted next-patch embedding $\hat { z } _ { t + 1 }$ and every actual patch embedding in the spectrogram.

## 4 CONCLUSION

We presented NAPE, a self-supervised framework for audio representation learning based on causal next patch embedding prediction. Departing from the reconstruction- and masking-based approaches that dominate audio SSL, NAPE relies on a deliberately minimalist recipe: a single causal Transformer encoder, a lightweight predictor head, and a negative cosine similarity loss with stop-gradient on the target branch. Across six audio and speech benchmarks, NAPE achieves strong downstream performance while relying on a substantially simpler pre-training recipe. NAPE also exhibits favorable scaling behavior across three encoder sizes and delivers competitive linear-probing results. A qualitative analysis further shows that the model develops structured attention patterns and organizes its learned embeddings into acoustically coherent regions without any explicit supervision. Together, these results establish autoregressive next-embedding prediction as a simple, scalable, and effective self-supervised objective for audio, and open a direct path to bringing the causal predictive paradigm into audio representation learning.

## ACKNOWLEDGMENTS

We thank Andrew Rouditchenko (Nvidia) for his insightful and valuable discussions.

## REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Sara Atito Ali Ahmed, Muhammad Awais, Wenwu Wang, Mark D Plumbley, and Josef Kittler. Asit: Local-global audio spectrogram vision transformer for event classification. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:3684–3693, 2024.

Tony Alex, Sara Atito, Armin Mustafa, Muhammad Awais, and Philip Jackson. Sslam: Enhancing selfsupervised models with audio mixtures for polyphonic soundscapes. In International Conference on Learning Representations, volume 2025, pp. 22608–22626, 2025.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15619–15629. IEEE, 2023.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Alan Baade, Puyuan Peng, and David Harwath. Mae-ast: Masked autoencoding audio spectrogram transformer. arXiv preprint arXiv:2203.16691, 2022.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020.

Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. In International conference on machine learning, pp. 1298–1312. PMLR, 2022.

Randall Balestriero and Yann LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025.

Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.

Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Bangalath, et al. Perception encoder: The best visual embeddings are not at the output of the network. Advances in Neural Information Processing Systems, 38:60884–60937, 2026.

George Bredis, Nikita Balagansky, Daniil Gavrilov, and Ruslan Rakhimov. Next embedding prediction makes world models stronger. arXiv preprint arXiv:2603.02765, 2026.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S Narayanan. Iemocap: Interactive emotional dyadic motion capture database. Language resources and evaluation, 42(4):335–359, 2008.

Ke Chen, Xingjian Du, Bilei Zhu, Zejun Ma, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Hts-at: A hierarchical token-semantic audio transformer for sound classification and detection. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 646–650. IEEE, 2022a.

Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International conference on machine learning, pp. 1691– 1703. PMLR, 2020.

Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, and Furu Wei. Beats: Audio pre-training with acoustic tokenizers. arXiv preprint arXiv:2212.09058, 2022b.

Wenxi Chen, Yuzhe Liang, Ziyang Ma, Zhisheng Zheng, and Xie Chen. Eat: Self-supervised pre-training with efficient audio transformer. arXiv preprint arXiv:2401.03497, 2024.

Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In 2021 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 15745–15753. IEEE, 2021.

Dading Chong, Helin Wang, Peilin Zhou, and Qingcheng Zeng. Masked spectrogram prediction for self-supervised audio pre-training. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023.

Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555, 2020.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Heinrich Dinkel, Zhiyong Yan, Yongqing Wang, Junbo Zhang, Yujun Wang, and Bin Wang. Scaling up masked audio encoder learning for general audio classification. arXiv preprint arXiv:2406.06992, 2024.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023.

Zhengcong Fei, Mingyuan Fan, and Junshi Huang. A-jepa: Joint-embedding predictive architecture can listen. arXiv preprint arXiv:2311.15830, 2023.

Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 776–780. IEEE, 2017.

Yuan Gong, Yu-An Chung, and James Glass. Ast: Audio spectrogram transformer. In INTERSPEECH, 2021a.

Yuan Gong, Yu-An Chung, and James Glass. Psla: Improving audio tagging with pretraining, sampling, labeling, and aggregation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3292–3306, 2021b.

Yuan Gong, Cheng-I Lai, Yu-An Chung, and James Glass. Ssast: Self-supervised audio spectrogram transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 10699–10709, 2022a.

Yuan Gong, Andrew Rouditchenko, Alexander H Liu, David Harwath, Leonid Karlinsky, Hilde Kuehne, and James Glass. Contrastive audio-visual masked autoencoder. arXiv preprint arXiv:2210.07839, 2022b.

Andrey Guzhov, Federico Raue, Jörn Hees, and Andreas Dengel. Audioclip: Extending clip to image, text and audio. In ICASSP 2022-2022 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 976–980. IEEE, 2022.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 15979–15988. IEEE, 2022.

Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4246–4253, 2020.

Chen Huang, Xianhang Li, Vimal Thilak, Etai Littwin, and Josh Susskind. Text-conditional jepa for learning semantically rich visual representations. arXiv preprint arXiv:2605.03245, 2026.

Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In European conference on computer vision, pp. 646–661. Springer, 2016.

Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, and Christoph Feichtenhofer. Masked autoencoders that listen. Advances in neural information processing systems, 35:28708–28720, 2022.

Zhifeng Kong, Arushi Goel, Rohan Badlani, Wei Ping, Rafael Valle, and Bryan Catanzaro. Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities. arXiv preprint arXiv:2402.01831, 2024.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Lucas Maes, Quentin Le Lidec, Luiz Facury, Nassim Massaudi, Ayush Chaurasia, Francesco Capuano, Richard Gao, Taj Gillin, Dan Haramati, Damien Scieur, et al. stable-worldmodel: A platform for reproducible world modeling research and evaluation. arXiv preprint arXiv:2605.21800, 2026a.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312, 2026b.

Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino. Masked spectrogram modeling using masked autoencoders for learning general-purpose audio representation. In HEAR: Holistic Evaluation ofAudio Representations, pp. 1–24. PMLR, 2022.

Daisuke Niizumi, Daiki Takeuchi, Masahiro Yasuda, Binh Thien Nguyen, Noboru Harada, and Nobutaka Ono. Rethinking masking strategies for masked prediction-based audio self-supervised learning. arXiv preprint arXiv:2603.23810, 2026.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Daniel S Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D Cubuk, and Quoc V Le. Specaugment: A simple data augmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779, 2019.

Karol J Piczak. Esc: Dataset for environmental sound classification. In Proceedings ofthe 23rd ACM international conference on Multimedia, pp. 1015–1018, 2015.

Boris T Polyak and Anatoli B Juditsky. Acceleration of stochastic approximation by averaging. SIAM journal on control and optimization, 30(4):838–855, 1992.

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv. Layer by layer: Uncovering hidden representations in language models. arXiv preprint arXiv:2502.02013, 2025.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818–2826, 2016.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4 technical report.˘ arXiv preprint arXiv:2607.02770, 2026.

Jayden Teoh, Manan Tomar, Kwangjun Ahn, Edward S Hu, Tim Pearce, Pratyusha Sharma, Akshay Krishnamurthy, Riashat Islam, Alex Lamb, and John Langford. Next-latent prediction transformers learn compact world models. arXiv preprint arXiv:2511.05963, 2025.

Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers. In 2021 IEEE/CVF international conference on computer vision (ICCV), pp. 32–42. IEEE, 2021.

Ludovic Tuncay, Etienne Labbé, Emmanouil Benetos, and Thomas Pellegrini. Audio-jepa: Joint-embedding predictive architecture for audio representation learning. arXiv preprint arXiv:2507.02915, 2025.

Feng Wang, Sucheng Ren, Tiezheng Zhang, Predrag Neskovic, Anand Bhattad, Cihang Xie, and Alan Yuille. Vit-5: Vision transformers for the mid-2020s. arXiv preprint arXiv:2602.08071, 2026.

Pete Warden. Speech commands: A dataset for limited-vocabulary speech recognition. arXiv preprint arXiv:1804.03209, 2018.

Haiyu Wu, Randall Balestriero, and Morgan Levine. Visreg: Variance-invariance-sketching regularization for jepa training. arXiv preprint arXiv:2606.02572, 2026.

Ho-Hsiang Wu, Prem Seetharaman, Kundan Kumar, and Juan Pablo Bello. Wav2clip: Learning robust audio representations from clip. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 4563–4567. IEEE, 2022.

Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr Dollár, and Ross Girshick. Early convolutions help transformers see better. Advances in neural information processing systems, 34: 30392–30400, 2021.

Sihan Xu, Ziqiao Ma, Wenhao Chai, Xuweiyi Chen, Weiyang Jin, Joyce Chai, Saining Xie, and Stella X Yu. Next-embedding prediction makes strong vision learners. arXiv preprint arXiv:2512.16922, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Xiaoyu Yang, Yifan Yang, Zengrui Jin, Ziyun Cui, Wen Wu, Baoxiang Li, Chao Zhang, and Phil Woodland. Spear: A unified ssl framework for learning speech and audio representations. arXiv preprint arXiv:2510.25955, 2025b.

Yumeng Yao, Jingzhi Dong, Haowen Gu, Tao Chen, Zonghan Wu, Xiaoshui Huang, and Yazhou Yao. Rethinking point clouds as sequences: A causal next-token predictive learning framework. arXiv preprint arXiv:2605.17566, 2026.

Goksenin Yuksel, Pierre Guetschel, Michael Tangermann, Marcel van Gerven, and Kiki van der Heijden. Wavjepa: Semantic learning unlocks robust audio foundation models for raw waveforms. arXiv preprint arXiv:2509.23238, 2025.

Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6023–6032, 2019.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.

## A APPENDIX

## B RELATED WORK

Supervised Audio Pre-training. Supervised pre-training for audio has been mainly explored in two regimes. Out-of-domain approaches adapt models originally trained on labeled image datasets such as ImageNet (Deng et al., 2009) to spectrogram inputs, typically by modifying the input layer from three RGB channels to a single-channel spectrogram. Early work of this kind used CNN backbones such as EfficientNet (Gong et al., 2021b); more recent work adopts Transformer-based architectures, notably AST (Gong et al., 2021a) and HTS-AT (Chen et al., 2022a), which have driven substantial gains on audio classification benchmarks. In-domain approaches instead pre-train directly on the target modality. CLAP (Elizalde et al., 2023) adapts the CLIP (Radford et al., 2021) recipe to audio via contrastive language-audio alignment on supervised text-audio pairs, while Audio-CLIP (Guzhov et al., 2022) and Wav2clip (Wu et al., 2022) extends the CLIP backbone with an additional audio encoder trained on AudioSet (Gemmeke et al., 2017). Despite delivering strong results, these methods rely on large quantities of labeled data, which are expensive and time-consuming to obtain in practice.

Self-Supervised Audio Pre-training. Self-supervised learning has driven substantial progress in audio representation learning, largely by transferring ideas developed in the vision domain. Most recent methods extract log-mel spectrograms as input and follow the masked image modeling paradigm (He et al., 2022): Audio-MAE (Huang et al., 2022), MaskSpec (Chong et al., 2023) and MSM-MAE (Niizumi et al., 2022) directly applie the MAE reconstruction objective to spectrogram patches; and BEATs (Chen et al., 2022b) instead trains an iterative acoustic tokenizer to provide discrete semantic prediction targets. Waveform-based approaches such as wav2vec 2.0 (Baevski et al., 2020) and data2vec (Baevski et al., 2022) bypass spectrogram preprocessing altogether, and predict latent representations of the raw audio signal. More recent efforts push these directions further by combining ideas from multiple objectives. EAT (Chen et al., 2024) combines an MAEstyle reconstruction with the data2vec latent-target formulation. A-JEPA (Fei et al., 2023) adapts the joint-embedding predictive paradigm to audio, using a target encoder updated via EMA and predicting masked patch embeddings. SSLAM (Alex et al., 2025) extends EAT with an additional source retention loss trained on artificially mixed audio to improve robustness to polyphonic content. Recently, SPEAR (Yang et al., 2025b) unifies speech and general audio representation learning through the distillation of complementary knowledge from specialized teacher models. Despite their diversity, these methods all rely on bidirectional prediction of masked or corrupted audio, and typically require a reconstruction decoder, a student-teacher setup, or auxiliary regularization losses to achieve competitive performance.

Next-token/embedding Prediction. Predictive learning has long been a central principle in representation learning across modalities. In language, GPT-style models (Radford et al., 2018; 2021; Achiam et al., 2023) established autoregressive next-token prediction as a scalable pre-training objec tive, with subsequent work confirming that the same paradigm transfers to vision (Image-GPT (Chen et al., 2020)) and beyond. More recent efforts move away from predicting raw signal tokens toward predicting embeddings directly. A first line of work adopts an autoregressive formulation: NEPA (Xu et al., 2025) introduces next-embedding predictive autoregression for visual representation learning; PointNTP (Yao et al., 2026) adapts causal next-token predictive learning to 3D point clouds; and LeWorldModel (Maes et al., 2026b) together with stable-worldmodel (Maes et al., 2026a) extend the paradigm to next-frame prediction for control tasks. A second line of work adopts a joint-embedding formulation: I-JEPA (Assran et al., 2023) and A-JEPA (Fei et al., 2023) predict latent representations of masked regions from context using dual encoders and an EMA teacher, while LeJEPA (Balestriero & LeCun, 2025) dispenses with masking and instead enforces invariance across augmented views with a variance-covariance regularizer. Together, these works suggest that predicting future tokens or embeddings can serve as a unified and scalable pre-training principle across modalities. In audio, however, this paradigm has remained largely unexplored: existing SSL methods rely almost exclusively on bidirectional masked modeling, and it is unclear a priori whether causal, per-position prediction is compatible with the non-stationary, temporally structured nature of audio signals. NAPE is, to the best of our knowledge, the first method to demonstrate that causal next-embedding prediction is a competitive and scalable pre-training objective for audio spectrograms, matching or surpassing more elaborate masked-modeling and joint-embedding methods with a substantially simpler recipe.

Table 8: Hyperparameter list. When an hyperparameter h varies between the base $( h _ { b } )$ and large (h ) model, we include both values like $( h _ { b } / h _ { l } )$ . <sup>∗</sup>Following (Chen et al., 2022b), we balance each class to 50% of the size of the unknown class for each training epoch.
<table><tr><td rowspan="2">Hyperparameters</td><td colspan="3">Pre-training</td><td colspan="4">Fine-tuning</td></tr><tr><td>AS-2M</td><td>AS-2M</td><td>AS-20K</td><td>ESC-50</td><td>KS1</td><td>KS2</td><td>ER</td></tr><tr><td>Optimizer</td><td></td><td></td><td></td><td>AdamW</td><td></td><td></td><td></td></tr><tr><td>Opt. Momentum (β1, β2)</td><td>(0.9,0.95)</td><td></td><td></td><td>(0.9,0.999)</td><td></td><td></td><td></td></tr><tr><td>Weight Decay</td><td colspan="7">0.05</td></tr><tr><td>Learning Rate Scheduler</td><td></td><td></td><td></td><td>Cosine Decay</td><td></td><td></td><td></td></tr><tr><td>Layer-Wise LR Decay</td><td>1.0</td><td>0.7/0.9</td><td>0.8/0.9</td><td>0.7/0.9</td><td>0.7/0.8</td><td>0.7/0.8</td><td>0.7/0.9</td></tr><tr><td>Base Learning Rate</td><td>5e-3</td><td></td><td></td><td>1.25e-3</td><td></td><td></td><td></td></tr><tr><td>Epochs</td><td>30/25</td><td>20/15</td><td>30/20</td><td>100</td><td>50</td><td>50</td><td>50</td></tr><tr><td>Warm-up Epochs</td><td>3</td><td>4/3</td><td>6/5</td><td>10</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Batch Size GPUs</td><td>256/128</td><td></td><td></td><td>64</td><td></td><td></td><td></td></tr><tr><td></td><td>8</td><td>4</td><td>4</td><td>1</td><td>1</td><td>1</td><td>4</td></tr><tr><td>Weighted sampling</td><td>x</td><td>√</td><td>x</td><td>x</td><td>√*</td><td>x</td><td>x</td></tr><tr><td>Multilabel</td><td>N/A</td><td>√</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>EMA Decay Rate</td><td>0.9999</td><td>0.99995</td><td>0.999</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Label Smoothing</td><td>N/A</td><td>0.</td><td>0.</td><td>0.1</td><td>0.1</td><td>0.</td><td>0.1</td></tr><tr><td>Roll Augmentation Drop Path</td><td>x</td><td>√</td><td>√</td><td>√</td><td>x</td><td>x</td><td>√</td></tr><tr><td></td><td>0.</td><td></td><td></td><td>0.1</td><td></td><td></td><td></td></tr><tr><td>SpecAug (time/freq)</td><td>N/A</td><td></td><td></td><td>(96,16) (24,16)/(96,16) (96,24)/(24,16) (24,16)</td><td></td><td></td><td>(24,16) (48,24)</td></tr><tr><td>Mixup (alpha/prob.)</td><td>N/A</td><td>(0.8,1.0)</td><td>(0.8,0.8)</td><td>(0.8,0.5)</td><td>(0.8,0.8) (0.8,0.8) (0.8,0.5)</td><td></td><td></td></tr><tr><td>Cutmix (alpha/prob.)</td><td>N/A</td><td>(1.0,1.0)</td><td>(1.0,0.8)</td><td>(1.0,0.5)</td><td>(1.0,0.8) (1.0,0.8) (1.0,0.5)</td><td></td><td></td></tr><tr><td>Noise Augmentation</td><td>x</td><td>√</td><td>V</td><td>√</td><td>√</td><td></td><td>√</td></tr><tr><td>Loss Function</td><td>Neg Cos Sim</td><td>BCE</td><td>BCE</td><td>CE</td><td>BCE</td><td>BCE</td><td>CE</td></tr><tr><td>Dataset Mean for Norm.</td><td>-6.84</td><td>-6.84</td><td>-6.84</td><td>-6.84</td><td>-9.11</td><td>-9.16</td><td>-13.74</td></tr><tr><td>Dataset Std for Norm.</td><td>5.38</td><td>5.38</td><td>5.38</td><td>5.38</td><td>4.53</td><td>4.61</td><td>3.88</td></tr></table>

## C FULL HYPERPARAMETER LIST

Fine-tuning Setting. We report the full hyperparameter list for fine-tuning NEPA Base and Large in Table 8. Those values refer to the Raster scan variant, with the Diagonal variant having almost the same hyperparameters (the only difference is in the number of epochs needed to converge for AS-20K, which can vary of only a few epochs). Regarding NEPA Small, we used the same hyperparameters as NEPA Base, with the only difference being the value of the layer-wise learning rate decay (LLRD) for some of the downstream tasks as this hyperparameter depends on the number of layers of the model. Specifically, we set $L L R D = 0 . 6$ for AS-2M and LLRD = 0.6 for KS1/KS2. No other changes have been made.

Linear Probing Setting. For the linear probing experiments, we made the following changes (all other settings are the same as the fine-tuning hyperparameters): we increase the learning rate to 1e − 2 and we disable all augmentation techniques.

## D ADDITIONAL ABLATION STUDIES

In this section, we report additional ablations studies on NAPE. For all experiments we use the raster scan variant. We use the SimSiam predictor for all ablations except for the ablation on the pre-training budget where we use the 2-MLP predictor.

Freezing the Patch Embedding Layer. Table 10 examines whether freezing the patch embedding layer during fine-tuning affects downstream performance. In Xu et al. (2025), freezing the embedding layer yields significant improvements. However, in our setting, the two configurations are essentially indistinguishable on all three tasks tested. We therefore leave the patch embedding layer trainable during fine-tuning, matching the standard practice of prior audio Transformer methods.

Table 10: Ablation on freezing the emb. layer.
<table><tr><td colspan="3">Freeze emb AS-2M AS-20K KS2</td></tr><tr><td>√</td><td>49.61</td><td>39.14 98.73</td></tr><tr><td>x</td><td>49.58</td><td>39.08 98.80</td></tr></table>

Table 11: Ablation on positional encoding.
<table><tr><td colspan="2">Positional Enc. AS-2M AS-20K</td></tr><tr><td>Absolute</td><td>47.8 36.7</td></tr><tr><td>RoPE</td><td>49.6 39.1</td></tr></table>

Positional Encoding. Table 11 compares learned absolute positional encodings with Rotary Position Embedding (RoPE) (Su et al., 2024), applied independently along the frequency and time axes. RoPE substantially outperforms absolute encodings on both AudioSet benchmarks, with gains of +1.8 mAP on AS-2M and +2.4 mAP on AS-20K. Beyond the accuracy improvement, RoPE also offers a practical benefit: because it encodes relative positions rather than absolute ones, it transfers naturally to clips of different lengths at inference time, whereas absolute encodings require interpolation. We therefore adopt RoPE as NAPE’s default positional encoding.

Attention Type and Pooling. Table 9 jointly ablates the attention mask used during fine-tuning and the pooling strategy for producing the clip-level representation. Under causal attention, the natural pooling choice is the last token: only the final position has access to the full preceding context, and the CLS token functions more as a BOSlike anchor at the start of the sequence than as a readout summary of the clip (this is in line with (Xu et al., 2025)). Under bidirectional attention, however, every position sees the entire sequence, so CLS, last-token, and mean pooling all in principle have access to the same information. This is reflected in the numbers: the three bidirectional variants are essentially on par on the three tested tasks. Keeping causal attention with last-token pooling remains competitive but slightly worse than its bidirectional counterpart, with a 0.2 mAP drop on both AS-2M and AS-20K. We therefore use bidirectional attention with mean pooling as NAPE’s default fine-tuning configuration.

Table 9: Ablation on the pooling method and attention type at fine-tuning.
<table><tr><td>Attention Pooling Type</td><td>Mode</td><td>Task AS-2M AS-20K KS2</td></tr><tr><td>Bidirec</td><td>CLS Tok 49.7</td><td>38.9 98.7</td></tr><tr><td>Bidirec</td><td>Last Tok 49.6</td><td>38.7 98.8</td></tr><tr><td>Bidirec Causal</td><td>Avg Pool 49.6 Last Tok 49.4</td><td>39.1 98.8 38.9 98.8</td></tr></table>

Normalization Layer. Table 12 compares Layer-Norm (Ba et al., 2016) (LN) and RMSNorm (Zhang & Sennrich, 2019) in the encoder. RMSNorm drops the mean-centering step of LayerNorm and normalizes each feature vector by its root-mean-square, which has become common in recent language and vision Transformers. In our setting, the two variants are effectively on par, with small variations across the six downstream tasks. For this reason, we retain Layer-

Table 12: Ablation on normalization layer style.
<table><tr><td>Norm</td><td>Audio Tasks Type AS-2M AS-20K ESC-50 KS1 KS2 ER</td><td>Speech Tasks</td></tr><tr><td>LN</td><td>49.6 39.1</td><td>94.2 97.9 98.8 64.9</td></tr><tr><td>RMS 49.5</td><td>38.8</td><td>94.4 97.8 98.8 65.2</td></tr></table>

Norm as NAPE’s default for consistency with prior audio Transformer works.

Pre-training Budget. Figure 8 reports downstream mAP on AS-2M and AS-20K when fine-tuning from NAPE-Base and NAPE-Large checkpoints obtained after 6, 12, 18, and 30 epochs of pre-training. Performance improves monotonically with the pre-training budget for both scales and both benchmarks: on AS-2M, NAPE-Large improves from 49.45 mAP after 6 epochs to 49.89 after 30; on AS-20K, from 37.15 to 39.94. Notably, NAPE already delivers strong results after only a few epochs of pre-training. After just 6 epochs, NAPE-Large reaches 49.45 mAP on AS-2M — already matching or exceeding most published baselines

![](images/c9243c1343d281257f7c3bb51dca71f7c8e714493adcd28231d0d3e9c27ec240.jpg)  
Figure 8: Pre-training budget ablation. AS-2M (left) and AS-20K (right) mAP trend as a function of the number of epochs the model is trained on.

(cf. Table 6) — making NAPE an appealing choice under tight compute budgets. Longer pre-training continues to yield returns at 30 epoch, and NAPE-Large consistently benefits more than NAPE-Base from a larger budget, consistent with the scaling behavior discussed in Section 3.3.

Random Masking. Table 13 examines whether adding random masking to the input embeddings during pre-training, in the style of masked modeling methods, helps NAPE. We consider three masking ratios: no masking (0%, the default), 20%, and 50%. We use NAPE small with raster scan order. Across all six benchmarks, masking either has no effect or slightly degrades downstream performance, with the largest drop on AS-20K (−1.0 mAP at 50%) and ESC-50 (−1.2 points at 50%). This differs from methods such as Audio-MAE (Huang et al., 2022) or BEATs (Chen et al., 2022b), where high masking ratios are central to the pre-training signal: the model is asked to reconstruct or predict the masked content from the visible context. In NAPE, the causal attention mask already prevents each position from accessing its target, so additional input masking removes useful context without changing the difficulty of the prediction task. We therefore leave the input unmasked in NAPE’s default configuration.

Table 13: Ablation on random masking applied to the input embeddings.
<table><tr><td>Mask Audio Tasks Speech Tasks Ratio AS-2M AS-20K ESC-50 KS1 KS2 ER</td></tr><tr><td>0 47.6 36.2</td></tr><tr><td>92.9 97.4 98.3 64.6 97.6 98.5 63.9</td></tr><tr><td>20 47.3 36.2 92.2 50 47.2 35.2 91.7 97.5 98.3 63.9</td></tr></table>

## E NAPE VS JEPA VS LEWORLDMODEL FORMULATION

## E.1 NAPE VS JEPA-STYLE METHODS

NAPE and JEPA methods Assran et al. (2023); Fei et al. (2023) learn representations by predicting one set of patch embeddings from another, therefore it is worth clarifying how the two paradigms differ structurally. Figure 9 (left and right) illustrates the two frameworks.

JEPA-style methods use two encoders: a context encoder h that processes the visible portion of the input x, and a target encoder that produces the prediction targets from the complementary region y. The target encoder is not trained by gradient descent; instead, its weights are updated as an EMA of the context encoder’s weights, so that its output remains a moving target that the predictor tries to match. The predictor itself is a Transformer that receives both the context representation and an additional variable $c ,$ which encodes the target positions to be predicted, and produces the predicted target embeddings $\hat { z } _ { y }$ in parallel. The context encoder operates with bidirectional attention over the visible patches. To prevent representation collapse under this regime, JEPA methods typically require additional machinery beyond stop-gradient: an EMA teacher (I-JEPA (Assran et al., 2023), A-JEPA (Fei et al., 2023)), auxiliary variance-covariance regularizers such as VICReg (Bardes et al., 2021), or masking strategies carefully tuned so that the two views do not degenerate into trivial solutions.

NAPE, in contrast, uses a single encoder (h) and takes its prediction targets directly from the shared patch embedding layer $f .$ Because f is a shallow, non-recurrent module, its outputs provide a stable target signal that does not need to be maintained by a separate EMA-tracked encoder. NAPE also features a lightweight MLP head (g) that operates on a single position at a time, rather than a Transformer that jointly reasons about context and target positions: a structurally simpler design that reflects NAPE’s per-position causal prediction task. There is no masking: the model processes the full input as a causal sequence, and predictions are made one patch at a time under a causal attention mask. Collapse prevention relies solely on the three key components discussed in Section 2.2: causality, prediction shift, and stop-gradient, with no auxiliary regularizers or student-teacher setups.

## E.2 NAPE VS LEWORLDMODEL METHODS

A closely related recent method is LeWorldModel (Maes et al., 2026b), a JEPA designed for world modeling that also autoregressively predicts next-frame embeddings from raw pixels (Figure 9, middle). Like NAPE, LeWM aims to simplify the JEPA recipe and dispenses with EMA teachers and pretrained encoders, training its encoder and predictor jointly end-to-end. Unlike NAPE, LeWM relies on the SIGReg regularizer (Balestriero & LeCun, 2025), an auxiliary loss that projects the learned embeddings onto random directions and enforces Gaussian-distributed marginals, to prevent collapse, whereas NAPE requires only stop-gradient on the target branch. LeWM uses MSE as its prediction loss (kept stable by SIGReg), whereas NAPE uses cosine similarity combined with stop-gradient, which together avoid the shrink-to-zero collapse mode observed with $\ell _ { 1 }$ and $\ell _ { 2 }$ under a magnitude-sensitive loss (Section 3.2). Finally, the two methods target different problems: LeWM predicts across time steps of an action-conditioned trajectory for planning in latent space, while NAPE predicts across patches within a single input for representation learning applicable to classification tasks. NAPE therefore constitutes a distinct point in the JEPA design space: one that shares LeWM’s end-to-end simplicity but arrives at collapse prevention through a fundamentally different mechanism.

![](images/177bf3bb9f7f0122959345b51f229a0eef2684af49a7fb1305df32b8fcd8d6f8.jpg)  
Figure 9: Comparison between JEPA, LeWorldModel, and NAPE architectures.

## F TRAINING LOSS VISUALIZATIONS

To better understand NAPE’s training dynamics, in Figure 10 we report the pre-training loss curves of six NAPE runs across multiple scales and configurations. The top row reports runs of the default NAPE configuration (raster scanning, SimSiam predictor, cosine similarity, all three key components enabled) at the three model scales — Small (top left), Base (top middle), and Large (top right). All three curves follow the same qualitative trajectory: a rapid descent from around −0.3 to below −0.9 within the first ∼ 20,000 steps, followed by a smooth and steady decrease that asymptotes just below −0.98, close to the theoretical minimum of −1. The Large model runs for roughly twice as many steps as Small and Base, its batch size is halved to fit into memory, but reaches a comparable final loss along a trajectory of the same shape, indicating that the NAPE objective is well-conditioned across encoder capacities.

The bottom row reports three ablation runs at the Base scale. Without the causal mask (bottom left), the loss collapses almost immediately, saturating near −1.0 within ∼ 2,000 steps and remaining there. The model has found the trivial identity mapping enabled by bidirectional attention: the objective is satisfied without learning any structure that transfers downstream, as reflected in the sharp mAP drop reported in Table 1.

The configuration with L1 loss (bottom middle) diverges: the loss drops rapidly to ∼ 0.05 within a few hundred steps, plateaus briefly, and then climbs steadily. The initial descent reflects the model finding low-norm solutions in which both predicted and target embeddings shrink toward zero, since L1 is magnitude-sensitive and trivially reduced by scaling the outputs; once the encoder approaches this degenerate regime, the objective becomes ill-conditioned and the loss reverses. The cross-entropy loss (bottom right), in contrast, converges to a reasonable minimum around step ∼ 70,000 before gradually overfitting. Unlike L1, CE does not collapse: this suggests that collapse prevention is not exclusive to cosine similarity. In our setting, however, CE still yields worse downstream performance than cosine (Table 5), and its tendency to overfit after the initial descent makes it a less robust choice for NAPE pre-training.

## G ADDITIONAL QUALITATIVE RESULTS

We include additional qualitative results for NAPE-L with raster order in Figure 11) and NAPE-L with diagonal order in Figure 12).

![](images/66d400e6de791e39c64c7d391a44fafa0e6aebe79908681bb1054234bc8f989c.jpg)

![](images/d4b077718d193cd9c21bf2e978d9a587ed3ccfad650fc3440f1830e92597a2aa.jpg)

![](images/5accd0697380235b0cd4c37bd1b3a104c9cedfe58031c685618adcc1f4b01bf1.jpg)

![](images/0741e335893358c579ddfcc43627e3f7b1cdd9fa46c193e8f6a6a1534a278136.jpg)

![](images/4f14801e3734e0d1d86d919bd61e9c363d4b148715e76e2a14add90aeb37b672.jpg)  
Figure 10: Pre-training loss curves for multiple NEPA models across scales and configurations.

![](images/0c5af1145ea863c85958c6563eac8a4c92b09790cf6e29b768acae04a84a7a1c.jpg)

![](images/ff5c5b3298bf4fb1a62ab48ba5a203a2327f73344021bbe8efc926fd4541f308.jpg)

![](images/492bc3c09897985312d629659120d5971e30a4f9c773ca5d6b2461839f92b470.jpg)

![](images/c67f7c7fc7a124489d1575d7ce32fb3d78b995ff7829e7e872f79eba2e0c1353.jpg)

![](images/e82dfeba0f45ba49199c83204d9299b28857ebc32e22f37a85e5d78c4491d560.jpg)

![](images/0c839556e7ae0f94608f8fa746d1ad3b5b41ced596c6d9c0d08de99ffa98814f.jpg)

![](images/86939704161881021a1fce38510507a5cac32b37ebe95dc35679a9088beb823b.jpg)

![](images/790bd8cd607f4b1d68670569c64f7b12ab0e30f0e631eb9832790cd8eb958f01.jpg)

![](images/76faf9e7cf258a64e62c9485c8525efafba66fc7bdbb616ec4f985a9d7011d3c.jpg)

![](images/94c1ab379155aa5ecc183faa07ca4f214fcf8d563896ed2111fe3f9f8d5786b2.jpg)  
Figure 11: Additional qualitative results for NAPE-L with raster order.

![](images/a1c48f477abd6715a75da74b6c1dd1c2e23166fa2bbcede4a23492f74c84c8bc.jpg)

![](images/ae3314e2842564b050974cde8491f1f5431a5338841a98ef181a99b06630a40b.jpg)

![](images/0c7f9648162fcd4627eb8fde7c9dc3f64f8125298ed791b2a25a0b19c1fd6d9b.jpg)  
Figure 12: Additional qualitative results for NAPE-L with diagonal order.
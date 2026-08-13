# FM-LLM: A Frequency-Enhanced Mixture-of-Experts Framework for Adapting LLMs to Time Series Forecasting

Rentao Gu<sup>a</sup>, Yihang Ding<sup>a</sup>, Junjie Li<sup>b</sup>, Yi Ding<sup>b</sup>, Weijing Sang<sup>a</sup>, Xiaoli Huo<sup>b</sup>, Xin Qin<sup>b</sup>, Yuefeng Ji<sup>a</sup>

<sup>a</sup>Beijing University of Posts and Telecommunications, Beijing, 100876, China <sup>b</sup>China Telecom Research Institute, Beijing, 100032, China

## Abstract

Recent advances in Large Language Models (LLMs) have spurred cross-modal solutions for time-series forecasting. However, existing methods rely heavily on textual prompts for modality alignment—introducing nontrivial computational overhead and failing to leverage the rich spectral dynamics inherent in timeseries data. To enable prompt-free, frequency-aware adaptation of frozen LLMs, we propose FM-LLM (Frequency-Enhanced Mixture-of-Experts for adapting LLMs to Time Series Forecasting), an autoregressive framework grounded in the principle of constrained asymmetric coupling. A Fourier Analysis Network (FAN)-based spectral token aligner injects structured harmonic representations directly into the frozen LLM with numerical compatibility. Concurrently, an asymmetric Mixture-of-Experts (MoE) decoder enforces explicit role separation: shared experts equipped with lightweight FAN layers reconstruct the global periodic backbone, while routed experts—restricted exclusively to standard feedforward networks (FFNs)—specialize in modeling non-periodic residual dynamics. A time-frequency hybrid loss function jointly optimizes temporal accuracy and spectral consistency, efectively mitigating error accumulation during longhorizon autoregressive rollouts. Evaluated across eleven public benchmarks, FM-LLM achieves state-of-the-art performance on 59 out of 78 evaluation metrics. Compared to the strongest autoregressive LLM-based baseline, it delivers average improvements of 5.3% in Mean Squared Error (MSE) and 5.6% in Mean Absolute Error (MAE), with maximum gains reaching 8.0% for MSE and 8.4% for MAE. FM-LLM also demonstrates robust transferability, maintaining superior performance in 10% few-shot and zero-shot forecasting scenarios. By purposefully wiring spectral alignment and decoding—distinct from promptdependent alignment strategies or uniform spectral processing approaches— FM-LLM establishes an accurate, eficient, and interpretable paradigm for nextgeneration time-series LLMs.

## 1. Introduction

Time-series forecasting is widely applied in domains such as energy systems [1], trafic management [2], network planning [3], and climate modeling [4], where accurate long-horizon predictions are essential for reliable resource allocation, anomaly detection, and strategic planning. As data volumes and realtime decision requirements continue to grow, the demand for robust and interpretable forecasting solutions has never been more pressing. Recent advances in deep learning—ranging from MLPs and CNNs to Transformer-based architectures—have significantly improved forecasting accuracy by modeling nonlinear temporal dependencies. More recently, pre-trained Large Language Models (LLMs) have been explored for time-series forecasting due to their strong transferability under limited supervision (few-shot/zero-shot) [5] and powerful context modeling ability, which can condition predictions on long historical contexts and auxiliary cues; such context-based conditioning has also been leveraged in recent LLM-based forecasters [6]. Moreover, their text-based interface makes it convenient to incorporate side information when available, and can potentially be extended to multimodal settings [7].

Despite these advantages, adapting LLMs to time-series forecasting remains challenging. First, time series are continuous-valued signals whereas LLMs operate on discrete tokens, making modality alignment non-trivial and potentially compromising numerical fidelity. Second, real-world time series exhibit multi-scale periodic and non-periodic dynamics that are not naturally captured by text-pretrained inductive biases [8], and purely time-domain prompting/tokenization may miss critical spectral structures. To mitigate these issues, existing LLM-based forecasting methods typically rely on prompts and lightweight encoders to provide the LLM with periodicity-related information and map continuous signals into token-like representations.

For instance, GPT4TS [9] fine-tunes a partially frozen GPT-2; Time-LLM [10] reformulates time windows as text-like sequences; AutoTimes [6] embeds time segments via MLPs and augments them with index prompts. PRADA [8] first applies season–trend decomposition to the input series, encoding the decomposed features as learnable prompts.

Despite their diferences, all these methods invariably adopt a patch-based strategy to align time-series inputs with the language modality, and employ shallow linear decoders—typically one or two layers—to map LLM outputs back to the time-series domain. While efective to some extent, these design choices raise a key concern: they may oversimplify the complex temporal and spectral structure inherent in multivariate time-series data.

This motivates us to revisit the fundamental design assumptions and ask two central questions:

• Can we extract richer and more structured representations from time-series patches, beyond treating them as flat tokens, to better inform LLMs?

• In multivariate forecasting, where variables exhibit diverse temporal and spectral characteristics, is a single linear decoder sufficient to model such heterogeneous patterns efectively?

For the first question, time-series data often embed complex and multi-scale periodicities, which are dificult to fully capture through token-level representations in the time domain alone. In fact, many existing forecasting models have demonstrated the efectiveness of frequency-domain analysis in modeling such temporal structures—for instance, FEDformer leverages Fourier attention to highlight dominant components [11], while TimesNet selects informative frequency bases to enhance prediction [12]. These observations motivate us to use spectral decomposition not merely as an auxiliary feature, but as a structured modality translator that aligns continuous-valued patches with the discrete token-wise computation of LLMs.

For the second question, multivariate time-series data often exhibit diverse temporal patterns across diferent variables, with varying periodicity, phase shifts, and noise levels. To address such heterogeneity at the same granularity as LLM token processing, we draw inspiration from the Mixture-of-Experts (MoE) architecture and perform patch-token-wise conditional routing. This enables diferent experts to specialize in distinct temporal regimes while keeping the overall computation eficient.

Adapting a frozen LLM for time-series forecasting introduces unique and stringent requirements beyond generic modality alignment. The fundamental challenge lies in eficiently injecting high-density frequency information and disentangling its superimposed entanglement with temporal dynamics—all while strictly avoiding prompt engineering, which typically incurs prohibitive inference latency through sequence expansion. To address these needs, we propose FM-LLM (Frequency-Enhanced Mixture-of-Experts for adapting LLMs to timeseries forecasting), a framework that couples explicit spectral token alignment with constrained conditional decoding in a purposefully asymmetric manner.

On the encoding side, a Fourier Analysis Network [13] (FAN)-based spectral token aligner maps each patch into a structured harmonic representation, establishing an eficient, prompt-free information pathway directly into the frozen LLM. On the decoding side, we enforce a design philosophy of explicit role separation: only the always-on shared (Fourier) expert incorporates lightweight FAN layers to reconstruct the global periodic backbone, whereas routed experts are intentionally restricted to standard FFNs to specialize in irregular, non-periodic residuals. This constrained asymmetric coupling not only prevents spectral bias from contaminating residual specialization but also efectively accommodates the diverse temporal patterns across variables in multivariate sequences through token-wise conditional computation. Extensive experiments on long-horizon forecasting benchmarks demonstrate that FM-LLM achieves state-of-the-art performance, obtaining the best results on 59 out of 78 evaluation metrics.

Our main contributions are summarized as follows:.

• Spectrum-Time Decoupling for Frozen LLM Adaptation. We propose a prompt-free spectral token aligner that replaces traditional textbased prompts with high-density harmonic representations. This design establishes an eficient information pathway into frozen LLMs, bypassing the inference latency overhead caused by sequence expansion.

• Constrained Asymmetric Decoding Strategy. We introduce a heterogeneous MoE decoder with explicit structural constraints to resolve the superimposed entanglement of temporal dynamics. By restricting spectral modeling to shared experts and standard FFNs to routed experts, we prevent spectral bias from contaminating residual specialization, efectively disentangling periodic backbones from non-stationary residuals.

• Holistic Time-Frequency Optimization. We design a hybrid loss function that jointly supervises temporal accuracy and spectral consistency. This dual-domain objective reinforces the structural decoupling of the decoder and significantly mitigates error accumulation in long-horizon autoregressive forecasting.

## 2. Related Work

## 2.1. Multivariate Time-Series Forecasting

Traditional time-series methods, such as AutoRegressive Integrated Moving Average (ARIMA) [14], exponential smoothing [15], and spectral analysis [16], have long been foundational in analyzing temporal data. These statistical models play an important role in revealing trends, seasonality, and structural patterns. However, their linear assumptions and stationarity requirements often limit their capacity to model complex non-linear dependencies and long-range temporal dynamics in real-world data.

In recent years, deep learning approaches have become dominant in timeseries forecasting due to their high capacity for non-linear representation learning. Common paradigms include CNN-based, GNN-based, Transformer-based, MLP-based, and more recently, LLM-based and KAN-based models. Alongside network architectures, auxiliary techniques such as sequence decomposition, spectral filtering, and Fourier analysis have been shown to further improve model precision by injecting structural inductive biases.

CNN-based methods such as MICN [17] and TimesNet [12] are notable. MICN adopts a multiscale downsampling convolution combined with globally equivariant convolutions, enabling eficient linear-complexity modeling that separates trend, periodicity, and seasonal components. TimesNet adaptively reshapes 1D sequences into 2D tensors based on multiple inferred periods and captures intra- and inter-period variations using lightweight multiscale Inception modules.

GNN-based methods, such as MSGNet [18] and PANDA [19], explicitly model inter-series dependencies via graph structures. MSGNet captures multiscale correlations by combining frequency decomposition with adaptive graph convolutions. PANDA integrates patching into graph networks, utilizing a dual time-frequency alignment strategy to optimize spatio-temporal representations.

MLP-based methods like DLinear [20] and TimeMixer [21] demonstrate that strong forecasting performance can be achieved with simple, eficient architectures. DLinear employs linear projection and decomposition, outperforming more complex models with minimal structure. TimeMixer proposes an MLPbased multiscale mixing framework tailored to diferent sampling rates.

Transformer-based models like PatchTST [22] and iTransformer [23] utilize patch-based encoding and token-level attention to model temporal dependencies. PatchTST segments time series into overlapping patches and embeds them into latent space, while iTransformer treats each univariate series as a token and models inter-series dependencies using self-attention and intra-series dynamics via FFNs.

LLM-based forecasting models, including GPT4TS [9], AutoTimes [6], Time-LLM [10], CVC [24] and EV-STLLM [25], adapt pretrained language models to time-series tasks. GPT4TS fine-tunes LayerNorm and position encoding of the first six GPT2 layers to leverage few-shot capabilities. Time-LLM aligns numerical tokens with language models via prefix-style prompts. AutoTimes encodes sliced segments with MLPs in an autoregressive fashion to mimic LLM training dynamics. CVC addresses the modality gap by employing a dual-branch architecture with contrastive correction to align time-series and textual features. EV-STLLM integrates multi-frequency decomposition with a partially frozen graph attention module to adapt LLMs for complex spatio-temporal forecasting.

Recent innovations include heterogeneous MLPs like TimeKAN [26], which introduces a Decomposition-Learning-Mixing framework using frequency decomposition, multi-order Kolmogorov-Arnold networks (KANs) [27], and frequency mixing to learn multifrequency temporal patterns efectively.

## 2.2. Fourier Analysis Network

Fourier-based neural networks have evolved from early shallow architectures using sinusoidal activations to deep networks integrating spectral transforms. Initial works like Fourier Neural Networks [28] employed cosine basis functions as activations to approximate periodic functions compactly. These models showed promise in control systems and signal approximation [29, 30], but were limited in depth and flexibility.

Subsequent studies identified limitations in ReLU-based networks for learning periodic signals. Liu et al [31] demonstrate that standard neural networks struggle to learn periodic functions due to the lack of inductive bias, and propose a sinusoid-based activation function to address this limitation. Empirical studies by Uteuliyeva [32] further confirmed the benefits of Fourier activations in shallow models.

Recent eforts have introduced Fourier representations into deeper architectures. SIREN [33] builds MLPs with sine activations, showing breakthroughs in signal reconstruction and PDE solving. More relevantly, FAN [13] integrates Fourier expansions into each layer using sin/cos projections of intermediate features, achieving strong extrapolation capabilities across language, vision, and time-series tasks. Unlike earlier domain-specific Fourier networks, FAN is designed as a general-purpose module and has proven especially efective in modeling periodic dynamics.

## 2.3. Mixture of Experts (MoE)

Recent Large pretrained models have increasingly adopted the Mixture-of-Experts (MoE) paradigm. MoE divides model capacity into specialized components ’experts’, each responsible for specific sub-tasks. Only some experts respond to each individual input, enabling eficient inference while retaining model diversity. This scalable framework enables growth in model capacity without proportional increases in compute cost.

MoE models such as Mixtral-8x7B [34], Switch Transformer [35]and DeepSeek-V3 [36] have demonstrated strong performance across tasks. However, MoE architectures face challenges such as training instability and expert load imbalance. DeepSeek addresses these with an auxiliary-loss-free load balancing mechanism that adjusts routing bias per expert. To prevent extreme imbalance within individual sequences, it introduces a sequence-wise balance loss to ensure intra-sequence fairness.

This aligns well with autoregressive forecasting where each token (or variable) may represent diferent patterns. In multivariate settings, even within the same time series, heterogeneous modes exist. MoE enables the model to learn such patterns and dynamically assign them to specialized experts. Our experiments show that diferent MoE experts indeed capture distinct temporal dynamics, improving overall forecasting performance.

## 2.4. Loss Functions in Time and Frequency Domains

Existing research has explored loss function design in both the time and frequency domains to improve training efectiveness and prediction accuracy in time-series forecasting.

Time-domain losses aim to directly minimize prediction error at each time step. Standard objectives like Mean Squared Error (MSE) and Mean Absolute Error (MAE) treat all horizons equally, which may lead to degraded performance for long-range forecasting. To address this, Signal Decay-based Loss (SDL) [37] introduces a horizon-aware weighting scheme that emphasizes shortterm predictions while suppressing noisy long-term targets. Similarly, the Loss Shaping Constraint (LSC) [38] applies step-specific error thresholds to control cumulative errors during multi-step forecasting.

Frequency-domain losses leverage spectral representations to better capture periodic structures in the data. FreDF: Learning to Forecast in the Frequency Domain [39] highlights that direct multi-step loss may misalign with the true data distribution due to autocorrelation. It proposes computing errors in the frequency domain to decouple temporal dependencies and enhance training stability. These frequency-based losses are often combined with MSE to balance time-domain accuracy and spectral consistency.

![](images/7dd3e88d9fd4308a2cce264e82c8ac3cdf96f0f4c3c375e94263010f2c92a454.jpg)  
Figure 1: An example to illustrate how FM-LLM adpts large language models for time series forecasting.

Inspired by both paradigms, we design a hybrid loss that combines SDL in the time domain and spectrum-based loss in the frequency domain. This formulation reinforces short-horizon prediction fidelity while promoting frequencyaware representation learning, which consistently improves performance across benchmarks.

## 3. Methodology

## 3.1. Problem Definition

Given a historical multivariate time series $\mathbf { x } _ { 1 : T } = \{ x _ { 1 } , x _ { 2 } , ~ . . . , x _ { T } \} \in \mathbb { R } ^ { C \times T }$ the objective of multivariate time-series forecasting is to predict a future sequence $\mathbf { x } _ { T + 1 : T + F } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { F } \} \in \mathbb { R } ^ { C \times F }$ , where $T$ and F denote the lengths of the look-back window and forecast horizon, respectively, and C represents the number of variables to be predicted. The proposed FM-LLM is, an adaptation of large language models tailored for multivariate time-series forecasting. The goal of this task is to train a frozen large language model (LLM)-based forecaster to predict the next $F$ time steps using the past T steps, formulated as:

$$
f : \left( \mathbf { x } _ { 1 : T } \right) \mapsto { \hat { \mathbf { x } } } _ { L + 1 : L + F } .\tag{1}
$$

## 3.2. Modality Alignment

## 3.2.1. Tokenization

We adopt the concept of patching as shown in Figure 1 to convert the time series into a token tensor and incorporate the idea of channel-wise independence[22]. Let the uni-variate input time series be denoted as $\mathbf { x } _ { 1 : T } \in \mathbb { R } ^ { T }$ , where P represents the token length and S denotes the stride between two consecutive, non-overlapping tokens. This patching process segments the time series into short subsequences, resulting in $\mathbf { x } _ { p } \in \mathbb { R } ^ { \tilde { N } \times P }$ , where N is the total number of tokens, computed as $\begin{array} { r } { N = \left\lfloor \frac { \check { T } - P } { S } + 1 \right\rfloor } \end{array}$ . In this work the token length and stride are set to $P = S$ . By adopting such a tokenization strategy, the number of tokens can be reduced from $T$ to approximately $T / S$ , which leads to a substantial reduction in both memory usage and computational complexity of the attention mechanism. This benefit is particularly significant when deploying large-scale Transformer models such as LLMs, because it can efectively reduce inference time and resource consumption.

![](images/4235dcd70884e237909e39369cad80d86892c337e33f12c85af21aca235b73e4.jpg)  
Figure 2: The input time series is first divided into tokens, which are then passed into the encoder. In the lower-left part of the figure, each token is decomposed into sine and cosine basis functions, as indicated by the red lines, while the blue lines connect to activation functions. The outputs of the LLM are routed through a Mixture-of-Experts (MoE) module. In the lowerright part of the figure, the right-side diagram illustrates that the Fourier experts represent the LLM outputs as weighted combinations of sine and cosine functions, with ’FC’ denoting fully-connected layers. $\mathrm { P _ { 1 } - P _ { 4 } }$ and $\mathrm { P _ { 2 } - P _ { 5 } }$ indicate the temporal ordering of input tokens and output tokens. During training, we compute the loss only for the final output token.

## 3.2.2. Token Embedding

We follow the autoregressive prediction framework[6] and repurpose autoregressive LLMs as time series forecasters as depicted in Figure 1. Let $x _ { t } ^ { i } \in \mathbb { R }$ denote the value of the i-th variable at time step t. The k-th segment of length P is defined as follows:

$$
P _ { k } = \{ x _ { ( k - 1 ) P + 1 } , . . . , x _ { k P } \} \in \mathbb { R } ^ { P } , k = 1 , 2 , . . . , N .\tag{2}
$$

To leverage the transferability of LLMs, their parameters are kept frozen.To align the discrete language tokens with the time-series tokens, a token embedding function is defined as:Tokenembedding( ) $: \mathbb { R } ^ { P } \mapsto \mathbb { R } ^ { H }$ which projects each time-series token into the latent space of the LLM. The embedded tokens are represented as:

$$
T E _ { k } = \mathrm { T o k e n e m b e d d i n g } ( P _ { k } ) , k = 1 , 2 , . . , N ,\tag{3}
$$

where H denotes the hidden dimension of the LLM. The token embedding process is described in the following section.

## 3.2.3. Fourier Embedding Module

Fourier theory states that any practically meaningful periodic function can be represented as a Fourier series—that is, a weighted sum of complex exponential functions that are harmonically related and share the same period as the function being represented. Because time series often contain complex periodic patterns, introducing the Fourier series is necessary. The Fourier Analysis Network [13] aims to construct an implicit periodic model within the data, expressed as follows:

$$
\mathrm { F A N } ( x ) = \phi _ { L } \circ \phi _ { L - 1 } \circ \cdot \cdot \cdot \circ \phi _ { 1 } \circ x\tag{7}
$$

where

$$
\phi ( x ) = \left\{ \begin{array} { l l } { [ c o s ( W _ { p } ^ { l } x ) | | s i n ( W _ { p } ^ { l } x ) | | \sigma ( B _ { \bar { p } } ^ { l } + W _ { \bar { p } } ^ { l } x ) ] , } & { \mathrm { i f ~ } l < L , } \\ { B ^ { L } + W ^ { L } x , } & { \mathrm { i f ~ } l = L , } \end{array} \right.\tag{8}
$$

where $W _ { p } \in \mathbb R ^ { d _ { x } \times d _ { p } } , \ W _ { \bar { p } } \ \in \ \mathbb R ^ { d _ { x } \times d _ { \bar { p } } }$ , and $B _ { \bar { p } } \in \mathbb { R } ^ { d _ { \bar { p } } }$ are learnable parameters (with the hyperparameters $d _ { p }$ and $d _ { \bar { p } }$ indicating the first dimension of $W _ { p }$ and $W _ { \bar { p } } .$ , respectively), the layer output $\phi ( x ) ~ \in ~ \bar { \mathbb { R } ^ { 2 d _ { p } + d _ { \bar { p } } } }$ , and σ denotes the activation function. In this paper, we use only a single-layer Fourier analysis network (FAN). Figure 2 shows the architecture of the Fourier Embedding Module in FM-LLM. First, the input time series tokens are projected into a high-dimensional space via a linear layer:

$$
P _ { k } ^ { 1 } = \mathrm { L i n e a r } ( P _ { k } )\tag{9}
$$

where $P _ { k } ^ { 1 } \ \in \ \mathbb { R } ^ { h 1 }$ . Then, the FAN module $\mathrm { F A N } ( \cdot ) \ : \ \mathbb { R } ^ { h 1 } \ \mapsto \ \mathbb { R } ^ { h 2 }$ extracts frequency-related features and nonperiodic components to form the following embedding:

$$
P _ { k } ^ { 2 } = \mathrm { T a n h } ( \mathrm { F A N } ( \mathrm { S i L U } ( P _ { k } ^ { 1 } ) ) )\tag{10}
$$

Finally, another linear layer is used to embeds the resulting representation into the hidden space of the large model:

$$
T E _ { k } = \mathrm { L i n e a r } ( P _ { k } ^ { 2 } ) , \quad k = 1 , 2 , . . . , N .\tag{11}
$$

This procedure constitutes the full Fourier Embedding Module of FM-LLM. This module efectively captures the dominant frequency components embedded in the original tokens, allowing the large model to utilize historical frequency information for more thorough modeling.

## 3.2.4. FAN-MoE Decoder

To enhance the expressiveness of the output module without incurring excessive computational cost, we adopt a sparse Mixture-of-Experts (MoE) architecture as the decoder in FM-LLM. MoE has been shown to efectively scale model capacity while preserving computational eficiency by activating only a small subset of expert networks for each token, based on a learned routing function [40]. This allows us to maintain a lightweight inference path while expanding the model’s representation space.

Prevalent large language models [41, 42] can predict the next token toke $^ { 1 } k$ based on the preceding tokens $p _ { < k }$ . We adopt this approach to reuse these models and iteratively generate predictions of arbitrary length. Given a time series segment of length $N P _ { : }$ , the input sequence is embedded into N token embeddings $\{ T E _ { 1 } , \cdots , T E _ { N } \}$ . The training objective is to autoregressively generate the next tokens $\{ \hat { P } _ { 2 } , \cdot \cdot \cdot , \hat { P } _ { N + 1 } \}$ based on these embeddings. We feed the token embeddings $T E _ { i }$ into the intermediate layers of the LLM, which model the token transitions and capture the contextual dependencies:

$$
\{ \widehat { T E } _ { 2 } , \cdot \cdot \cdot , \widehat { T E } _ { N + 1 } \} = \mathrm { L L M } ( \{ T E _ { 1 } , \cdot \cdot \cdot , T E _ { N } \} )\tag{12}
$$

In time-series forecasting, diferent input tokens may correspond to distinct temporal or frequency modes, such as trends, seasonal components, or transient anomalies. This complexity becomes even more pronounced in multivariate time series, where intricate dependencies exist not only across time but also across variables. The conditional routing property of the MoE enables structural specialization across experts, allowing the decoder to adaptively reconstruct multiscale, nonstationary, and cross-variable patterns. Because the Fourier encoder injects frequency-aware priors into the input embeddings, the MoE decoder serves as a complementary mechanism to selectively decode these representations into the target space. This combination forms a frequency-aligned encoderdecoder pathway that improves both prediction accuracy and eficiency, especially in high-dimensional forecasting settings. These design choices are aligned with recent findings that emphasize conditional computation and specialization as efective strategies for handling complex sequence structures and variable interactions in multivariate forecasting [11].

Here, we adopt a load-balanced MoE mechanism [36] shown in Figure2. Let $\widehat { T E } _ { k + 1 }$ denote the output of the large language model, the predicted token is dthen formulated as follows:

$$
\hat { P } _ { k + 1 } = \sum _ { i = 1 } ^ { N _ { s } } \mathrm { F } \mathrm { - } \mathrm { F F N } _ { i } ^ { ( s ) } ( \widehat { T E } _ { k + 1 } ) + \sum _ { i = 1 } ^ { N _ { r } } g _ { i , k } \mathrm { F F N } _ { i } ^ { ( r ) } ( \widehat { T E } _ { k + 1 } )\tag{13}
$$

$$
\mathrm { F } \mathrm { - } \mathrm { F F N } _ { i } ^ { ( s ) } ( \cdot ) = \mathrm { L i n e a r } ( \mathrm { S i L U } ( \mathrm { F A N } ( \mathrm { T a n h } ( \mathrm { L i n e a r } ( \cdot ) ) ) ) )\tag{14}
$$

$$
g _ { i , k } = \frac { g _ { i , k } ^ { \prime } } { \sum _ { j = 1 } ^ { N _ { r } } g _ { j , k } ^ { \prime } }\tag{15}
$$

$$
g _ { i , k } ^ { \prime } = \left\{ { \begin{array} { l l } { s _ { i , k } , } & { s _ { i , k } \in \mathrm { T o p k } ( \{ s _ { j , k } \mid 1 \leq j \leq N _ { r } \} , K _ { r } ) } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} } \right.\tag{16}
$$

$$
s _ { i , k } = \mathrm { S o f t m a x } ( \widehat { W T E } _ { k + 1 } )\tag{17}
$$

Here, k represent the k-th time-series token, and $N _ { s }$ and $N _ { r }$ denote the total numbers of Fourier and routed experts, respectively. The operators $\mathrm { F } { \cdot } \mathrm { F F N } _ { i } ^ { ( s ) } ( \cdot )$ and $\mathrm { F F N } _ { i } ^ { ( r ) } ( \cdot )$ correspond to the i-th Fourier expert and routed expert. The variable $g _ { i } ^ { k }$ indicates the normalized gating value for the i-th routed expert on token $k ,$ and $g _ { i , k }$ denotes the pre-normalized gating score; $s _ { i , k }$ indicates how strongly token $k$ is associated with expert $i ;$ and $\mathrm { T o p k } ( \cdot , K _ { r } )$ denotes the function that selects the $K _ { r }$ highest scores among all associated values $s _ { j , k }$ for the k-th token. $W \in \mathbb { R } ^ { H \times 1 }$ , where H denotes the LLM’s hidden dimension.

It is worth noting that, to enhance the model’s ability to represent periodic patterns, we incorporate a FAN layer into the Fourier experts, similar to the encoder module, as illustrated in Fig 2. Ensuring consistency across frequency tokens and facilitating model generalization. Meanwhile, to retain the capacity for modeling the non-periodic components of the sequence, the routed experts remain as standard feed-forward networks (FFNs) composed of two linear layers.

A prominent challenge in sparsely gated MoE models is load imbalance, where certain experts are overused while others remain underutilized, leading to routing collapse and degraded model eficiency. To mitigate the load imbalance problem in expert routing, we adopt a bias-based auxiliary-loss-free balancing mechanism inspired by DeepSeek-V3 [36], where each expert is associated with a learnable bias term to adjust its selection probability during top-K routing:

$$
g _ { i , t } ^ { \prime } = \left\{ \begin{array} { l l } { s _ { i , t } , } & { s _ { i , t } + b _ { i } \in \mathrm { T o p k } ( \{ s _ { j , t } + b _ { j } \mid 1 \leq j \leq N _ { r } \} , K _ { r } ) } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{18}
$$

Through this dynamic adjustment mechanism,the FM-LLM maintained a balanced expert load during training.

## 3.3. Loss Function

Unlike previous approaches that generate a fixed-length prediction sequence in a single forward pass, autoregressive forecasting inevitably leads to error accumulation during long-horizon predictions, owing to the recursive use of model outputs as future inputs. To alleviate this problem, it is essential to enhance the single-step forecasting accuracy of the model; that, its ability to predict the next token based on the historical sequence. To this end, a signal-decay weighting scheme inspired by CARD [37] is adopted in the loss function, where predictions closer to the present receive higher importance. This reweighting strategy encourages the model to focus on near-future predictions, thereby reducing early-stage prediction errors and indirectly mitigating the compounding efect of error accumulation in autoregressive rollouts. Furthermore, to fully exploit the historical information contained in the input window and strengthen the model’s capability to extrapolate future trends from past observations, the training loss is computed solely based on the token immediately following the context window. The empirical results show that this design significantly improves forecasting accuracy compared to uniformly weighted multi-step objectives. In addition, it is observed that FAN alone did not consistently enhance prediction performance when applied directly. To address this limitation, an auxiliary frequency-domain loss is introduced, which enforces an alignment between the predicted and ground-truth sequences in the Fourier domain. By minimizing the discrepancy between their Fourier coeficients, the model is encouraged to learn periodic structures more efectively, thereby allowing the FAN module to fully exploit its potential through frequency-guided supervision. The

loss formula is expressed as follows:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { f o r e c a s t } } = \alpha \cdot \mathcal { L } _ { \mathrm { f r e q } } + \beta \cdot \mathcal { L } _ { \mathrm { t i m e } } } \\ & { \quad \quad \quad = \alpha \cdot \left[ \displaystyle \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \| \mathcal { F } ( w _ { l } \cdot \hat { a } _ { t + l } ) - \mathcal { F } ( w _ { l } \cdot a _ { t + l } ) \| _ { 1 } \right] } \\ & { \quad \quad \quad \quad + \beta \cdot \left[ \displaystyle \frac { 1 } { L } \sum _ { l = 1 } ^ { L } w _ { l } \cdot \| \hat { a } _ { t + l } - a _ { t + l } \| _ { 2 } ^ { 2 } \right] } \end{array}\tag{19}
$$

where $\mathcal { L } _ { \mathrm { f o r e c a s t } }$ denotes the overall loss function, composed of two components: a frequency-domain loss and a time-domain loss. The scalar $\alpha , \beta \in [ 0 , 1 ]$ is a hyperparameter that balances the contribution of each component. The summation index $l = 1 , \ldots , L$ iterates over L future prediction steps, and in our experiments, we restrict the loss evaluation to the immediate next-token prediction, setting $L = P$ accordingly.

In the first term, $\mathcal F ( \cdot )$ denotes the discrete Fourier transform (DFT) applied to the predicted time steps $\hat { a } _ { t + l }$ and the ground truth time steps $a _ { t + l }$ . The $\ell _ { 1 }$ norm is used to measure the discrepancy between their frequency representations. The second term computes the conventional mean squared error (MSE) in the time domain by using the $\ell _ { 2 }$ norm.

The weighting coeficients $w _ { l } = 1 / \sqrt { l }$ serve as signal-decay weights, assigning larger weights to near-future predictions and smaller weights to long-term predictions. This design prioritizes accurate short-term predictions, which are crucial in autoregressive forecasting to mitigate error accumulation during longrange rollout.

Although our model adopts a primarily auxiliary-loss-free strategy for MoE expert routing, autoregressive time-series forecasting presents a unique challenge: expert load imbalance within a single prediction sequence may be amplified due to recursive dependencies. To address this issue, we introduce a complementary sequence-wise balance loss [36], defined as:

$$
\mathcal { L } _ { \mathrm { { B a l } } } = \gamma \sum _ { i = 1 } ^ { N _ { r } } f _ { i } P _ { i } ,\tag{20}
$$

Here, $N _ { r }$ stands for the number of experts, and $\gamma$ denotes a small hyperparameter responsible for adjusting the auxiliary loss weight.

The term $f _ { i }$ denotes the normalized routing frequency of expert i across all time series tokens, which is computed as:

$$
f _ { i } = \frac { N _ { r } } { K _ { r } K } \sum _ { k = 1 } ^ { K } \mathbb { 1 } ( s _ { i , k } \in \mathrm { T o p } k ( \{ s _ { j , k } \} _ { j = 1 } ^ { N _ { r } } , K _ { r } ) ) ,\tag{21}
$$

where $K _ { r }$ is the number of experts selected per token, $s _ { i , k }$ is the routing score of expert i for the k-th token, and $\Im ( \cdot )$ is the indicator function.

We further define the normalized routing proportion $s _ { i , k } ^ { \prime }$ and the averaged routing probability $P _ { i }$ as:

$$
s _ { i , k } ^ { \prime } = \frac { s _ { i , k } } { \sum _ { j = 1 } ^ { N _ { r } } s _ { j , k } } ,\tag{22}
$$

$$
P _ { i } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } s _ { i , k } ^ { \prime } .\tag{23}
$$

This sequence-wise auxiliary loss encourages token-wise expert balance within each sequence, thereby preventing local routing collapse in autoregressive prediction settings. In practice, the coeficient λ is set to a small value, ensuring that it does not dominate the main forecasting objectives while still ofering regularization benefits.

Accordingly, the overall loss function is defined as follows:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \underbrace { \alpha \cdot { \mathcal { L } } _ { \mathrm { f r e q } } + \beta \cdot { \mathcal { L } } _ { \mathrm { t i m e } } } _ { \mathrm { F o r e c a s t i n g ~ L o s s } } + \lambda \cdot { \mathcal { L } } _ { \mathrm { B a l } }\tag{24}
$$

This multi-objective loss design ensures that the model not only achieves accurate and robust time-series prediction but also maintains stable and eficient expert routing behavior during long-horizon recursive forecasting.

## 4. Experiments

## 4.1. Experimental setup

## 4.1.1. Datasets

We assess the long-term forecasting capabilities of FM-LLM on seven distinct benchmark datasets, including the ETT datasets (namely ETTh1, ETTh2,

Table 1: Dataset detailed descriptions. The dataset size is organized in (Train, Validation, Test).The forecastability is calculated by one minus the entropy of Fourier decomposition of time series [43].A larger value indicates better predictability.
<table><tr><td rowspan=1 colspan=1>Tasks  |</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Dim</td><td rowspan=1 colspan=1>Series Length</td><td rowspan=1 colspan=1>Dataset Size</td><td rowspan=1 colspan=1>Frequency</td><td rowspan=1 colspan=1>Forecastability*</td><td rowspan=1 colspan=1>Information</td></tr><tr><td rowspan=11 colspan=1>Long-termForecasting</td><td rowspan=1 colspan=1>ETTm1</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>{96, 192, 336, 720}</td><td rowspan=1 colspan=1>(34465, 11521, 11521)</td><td rowspan=1 colspan=1>15min</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>Temperature</td></tr><tr><td rowspan=1 colspan=1>ETTm2</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>{96, 192, 336, 720}</td><td rowspan=1 colspan=1>(34465, 11521, 11521)</td><td rowspan=1 colspan=1>15min</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>Temperature</td></tr><tr><td rowspan=1 colspan=1>ETTh1</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>{96, 192, 336, 720}</td><td rowspan=1 colspan=1>(8545, 2881, 2881)</td><td rowspan=1 colspan=1>15 min</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>Temperature</td></tr><tr><td rowspan=1 colspan=1>ETTh2</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>{96, 192, 336, 720}</td><td rowspan=1 colspan=1>(8545, 2881, 2881)</td><td rowspan=1 colspan=1>15 min</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>Temperature</td></tr><tr><td rowspan=1 colspan=1>Electricity</td><td rowspan=1 colspan=1>321</td><td rowspan=1 colspan=1>{96, 192, 336, 720}</td><td rowspan=1 colspan=1>(18317, 2633, 5261)</td><td rowspan=1 colspan=1>Hourly</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>Electricity</td></tr><tr><td rowspan=1 colspan=1>Traffic</td><td rowspan=1 colspan=1>862</td><td rowspan=1 colspan=1>{96, 192, 336, 720}</td><td rowspan=1 colspan=1>(12185, 1757, 3509)</td><td rowspan=1 colspan=1>Hourly</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>Transportation</td></tr><tr><td rowspan=1 colspan=1>Weather</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>{96, 192, 336, 720}</td><td rowspan=1 colspan=1>(36792, 5271, 10540)</td><td rowspan=1 colspan=1>10 min</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>Weather</td></tr><tr><td rowspan=1 colspan=1>PEMS03</td><td rowspan=1 colspan=1>358</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>(15617, 5135, 5135)</td><td rowspan=1 colspan=1>5min</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>Transportation</td></tr><tr><td rowspan=1 colspan=1>PEMS04</td><td rowspan=1 colspan=1>307</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>(10172, 3375, 3375)</td><td rowspan=1 colspan=1>5min</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>Transportation</td></tr><tr><td rowspan=1 colspan=1>PEMS07</td><td rowspan=1 colspan=1>883</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>(16911, 5622, 5622)</td><td rowspan=1 colspan=1>5min</td><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>Transportation</td></tr><tr><td rowspan=1 colspan=1>PEMS08</td><td rowspan=1 colspan=1>170</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>(10690, 3548, 265)</td><td rowspan=1 colspan=1>5min</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>Transportation</td></tr><tr><td rowspan=6 colspan=1>Short-termForecasting</td><td rowspan=1 colspan=1>M4-Yearly</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>(23000, 0, 23000)</td><td rowspan=1 colspan=1>Yearly</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>Demographic</td></tr><tr><td rowspan=1 colspan=1>M4-Quarterly</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>(24000, 0, 24000)</td><td rowspan=1 colspan=1>Quarterly</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>Finance</td></tr><tr><td rowspan=1 colspan=1>M4-Monthly</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>(48000, 0, 48000)</td><td rowspan=1 colspan=1>Monthly</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>Industry</td></tr><tr><td rowspan=1 colspan=1>M4-Weakly</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>(359, 0, 359)</td><td rowspan=1 colspan=1>Weakly</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>Macro</td></tr><tr><td rowspan=1 colspan=1>M4-Daily</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>(4227, 0, 4227)</td><td rowspan=1 colspan=1>Daily</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>Micro</td></tr><tr><td rowspan=1 colspan=1>M4-Hourly</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>(414, 0, 414)</td><td rowspan=1 colspan=1>Hourly</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>Other</td></tr></table>

ETTm1, and ETTm2), alongside Electricity, Weather, and Trafic datasets. The Electricity Transformer Temperature (ETT) benchmark dataset acts as a key indicator for modeling long-term electric power deployment trends. It contains two years of operational data gathered from two counties in China and is divided into four subsets featuring diferent temporal resolutions: ETTh1 and ETTh2, sampled hourly, and ETTm1 and ETTm2, sampled every 15 minutes. Each record consists of six power load-related features and a target variable known as oil temperature, which represents the transformer’s thermal condition and is widely utilized in predictive modeling. The Electricity dataset comprises hourly electricity consumption data from 321 individual consumers, providing a highdimensional multivariate time series capturing daily and weekly usage patterns. The Weather dataset includes one year of meteorological data collected from 21 monitoring stations throughout Germany, with a sampling frequency of every 10 minutes. It covers a broad range of atmospheric variables that facilitate fine-grained forecasting and environmental modeling. The Trafic dataset contains hourly occupancy rates from 862 sensors distributed across the freeway network in California. This dataset captures temporal trafic patterns and is frequently used in spatiotemporal modeling and congestion prediction research. The PEMS dataset consists of public trafic network data from California, collected in 5-minute intervals. We use the same four public subsets (PEMS03, PEMS04, PEMS07,PEMS08) adopted in iTransformer [23].The details are provided in Table 1.

Following standard practice, we partition all datasets in chronological order. Specifically, the ETT datasets are split into training, validation, and test sets with a ratio of (3:1:1), the PEMS datasets adopt a (6:2:2) split, whereas the other datasets adopt a (7:1:2) split.

## 4.1.2. Metrics

For all experiments, we adopt the same evaluation criteria as used in mainstream methods. Specifically, for long-term forecasting tasks, we employ Mean Squared Error and Mean Absolute Error as performance evaluation metrics. For the short-term forecasting, following the [12], we adopt the symmetric mean absolute percentage error (SMAPE), mean absolute scaled error (MASE) and overall weighted average (OWA) as the metrics, where OWA is a special metric used in M4 competition. The definitions of these metrics are as follows:

$$
\mathrm { M S E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 }\tag{25}
$$

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| y _ { i } - \hat { y } _ { i } \right|\tag{26}
$$

$$
\mathrm { S M A P E } = \frac { 2 0 0 } { N } \sum _ { i = 1 } ^ { N } \frac { \left| y _ { i } - \hat { y } _ { i } \right| } { \left| y _ { i } \right| + \left| \hat { y } _ { i } \right| }\tag{27}
$$

$$
\mathrm { M A S E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { \left| y _ { i } - \hat { y } _ { i } \right| } { \frac { 1 } { N - s } \sum _ { j = s + 1 } ^ { N } \left| y _ { j } - y _ { j - s } \right| }\tag{28}
$$

$$
\mathrm { O W A } = \frac { 1 } { 2 } \left[ \frac { \mathrm { S M A P E } } { \mathrm { S M A P E } _ { \mathrm { N a i v e 2 } } } + \frac { \mathrm { M A S E } } { \mathrm { M A S E } _ { \mathrm { N a i v e 2 } } } \right]\tag{29}
$$

where N denotes the number of data points(i.e, the length of one token, which is equal to P), $y _ { i }$ and $\hat { y } _ { i }$ are the i-th groundtruth and prediction where $i \in$ $\{ 1 , 2 , \cdots , N \}$ , s is periodicity of the data.

## 4.1.3. Baselines

We carefully select 16 representative baselines from the recent time series forecasting landscape, including the following categories: (1) Transformer-based models: iTransformer [23], PatchTST [22], TQ-Net [44], Non-Stationary Transformer [45]; Reformer [46](2) MLP-based models: DLinear [20], AMD [47], CycleNet [48], N-Beats [49], N-Hits [50]; (3) CNN-based models: Timesnet [12], FEDformer [11]; (4) LLM-based models: AutoTimes [6], GPT4TS [9], PRADA [8], TimeLLM [10].

## 4.1.4. Implementation Details

For a fair comparison, all the experiments share the same random seed and use the Adam optimizer. We use Llama-3.2-1B as our backbone. We employ the L1 loss as the optimization objective for the ETT, trafic and Weather datasets, whereas the MSE loss is used for Electricity datasets. The hyperparameters were tuned manually based on validation set performance, the early stopping criterion used a patience of 3. Our model implementation is based on PyTorch 2.5.1 [51], with all experiments conducted on a single NVIDIA RTX 4090 24G GPU. The training process demands approximately 6GB of GPU memory and involves roughly 8.68 million trainable parameters.

In addition, we follow the same settings of lookback window length and token length as in AutoTimes [6]. Specifically, for our model, we set the input length to $L = 6 7 2$ and the output token length to $P = 9 6 .$ . To ensure fair and consistent comparison, we primarily use the results reported in the original publications for all baseline models. For the PEMS datasets, we set the input length of each method to 672. If the forecasting performance at this length does not exceed the results reported in the original paper, we retain the results from the original publication. In the few-shot learning setting, we train the model for up to 100 epochs with an early stopping patience of 10, to ensure efective fitting on limited data. For short-term forecasting tasks, we set the lookback window size to twice the prediction horizon. Other details of forecasting configurations are summarized in Table 2.

Table 2: Model configurations of FM-LLM. Fourier refer to Fourier experts
<table><tr><td>Dataset</td><td>token len</td><td>MLP dim</td><td>dropout</td><td>lr</td><td>Fourier</td><td>routed</td><td>active</td><td>γ</td><td>α</td><td>β</td><td>λ</td><td>batch</td></tr><tr><td>ETTm1</td><td>96</td><td>512</td><td>0.2</td><td>2e-4</td><td>2</td><td>6</td><td>2</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>256</td></tr><tr><td>ETTm2</td><td>96</td><td>512</td><td>0.2</td><td>1.5e-4</td><td>2</td><td>6</td><td>2</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>256</td></tr><tr><td>ETTh1</td><td>96</td><td>512</td><td>0.2</td><td>2e-4</td><td>2</td><td>2</td><td>1</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>256</td></tr><tr><td>ETTh2</td><td>96</td><td>512</td><td>0.2</td><td>2e-4</td><td>2</td><td>3</td><td>1</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>256</td></tr><tr><td>Weather</td><td>96</td><td>512</td><td>0.1</td><td>1e-4</td><td>2</td><td>8</td><td>3</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>256</td></tr><tr><td>Electricity</td><td>96</td><td>512</td><td>0.1</td><td>1e-4</td><td>2</td><td>8</td><td>3</td><td>1e-5</td><td>0.9</td><td>1</td><td>0.1</td><td>256</td></tr><tr><td>Traffic</td><td>96</td><td>512</td><td>0.1</td><td>1e-4</td><td>2</td><td>8</td><td>3</td><td>1e-5</td><td>0.9</td><td>1</td><td>0.1</td><td>256</td></tr><tr><td>PEMS03</td><td>96</td><td>512</td><td>0.1</td><td>1e-4</td><td>2</td><td>8</td><td>3</td><td>1e-5</td><td>0.9</td><td>1</td><td>0.1</td><td>256</td></tr><tr><td>PEMS04</td><td>96</td><td>512</td><td>0.1</td><td>1e-4</td><td>2</td><td>8</td><td>3</td><td>1e-5</td><td>0.9</td><td>1</td><td>0.1</td><td>256</td></tr><tr><td>PEMS07</td><td>96</td><td>512</td><td>0.1</td><td>1e-4</td><td>2</td><td>8</td><td>3</td><td>1e-5</td><td>0.9</td><td>1</td><td>0.1</td><td>256</td></tr><tr><td>PEMS08</td><td>96</td><td>512</td><td>0.1</td><td>1e-4</td><td>2</td><td>8</td><td>3</td><td>1e-5</td><td>0.9</td><td>1</td><td>0.1</td><td>256</td></tr><tr><td>M4 Yearly</td><td>6</td><td>256</td><td>0.1</td><td>1e-4</td><td>2</td><td>4</td><td>2</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>64</td></tr><tr><td>M4 Quarterly</td><td>8</td><td>512</td><td>0.2</td><td>5e-5</td><td>2</td><td>6</td><td>2</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>16</td></tr><tr><td>M4 Monthly</td><td>18</td><td>512</td><td>0.0</td><td>5e-5</td><td>2</td><td>2</td><td>1</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>16</td></tr><tr><td>M4 Weekly</td><td>13</td><td>256</td><td>0.0</td><td>1e-3</td><td>2</td><td>2</td><td>1</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>16</td></tr><tr><td>M4 Daily</td><td>14</td><td>512</td><td>0.0</td><td>5e-4</td><td>2</td><td>2</td><td>1</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>16</td></tr><tr><td>M4 Hourly</td><td>48</td><td>256</td><td>0.0</td><td>2e-4</td><td>2</td><td>2</td><td>1</td><td>1e-4</td><td>1</td><td>1</td><td>1</td><td>16</td></tr></table>

## 4.2. Main Results

Table 3, Table 4, Table 5, Table 6, and Table 7 summarize the main results of FM-LLM on multivariate time-series forecasting tasks, including Long-term forecasting, Short-term forecasting, few-shot learning, and zero-shot transfer experiments. Overall, the results indicate that FM-LLM not only achieves significantly superior performance in standard long-sequence forecasting but also demonstrates outstanding adaptability and stability under generalization challenges such as data scarcity (few-shot) and cross-distribution transfer (zeroshot).

Table 3 summarizes the long-term forecasting results by averaging over four prediction lengths. For completeness, we report the full results in Appendix A Table A.11, where the best results are marked in bold and the second-best are underlined. Across the 70 reported metrics, FM-LLM achieves 51 best and 15 second-best scores, demonstrating a strong overall advantage. Notably, Auto-Times, which also adopts an autoregressive architecture and exhibits good transfer generalization, is consistently outperformed by FM-LLM across all seven datasets for long-sequence forecasting. On average, FM-LLM reduces MSE and MAE by 5.20% and 5.32%, respectively, compared to AutoTimes. This clearly indicates that the Frequency-Aware Module (Fourier Analysis Network, FAN) and the time-frequency joint loss function introduced in FM-LLM efectively enhance the global modeling capacity of the autoregressive forecasting framework and improve its ability to capture long-range dependencies.

Table 3: Long-term forecasting results. We average the results across 4 prediction lengths: {96, 192, 336, 720}. The best results are highlighted in bold red and the second best are underlined blue. Appendix A provides full results.
<table><tr><td>Models</td><td colspan="2">FM-LLM (Ours)</td><td colspan="2"> $\mathbf { \overline { { A u t o T i m e s ^ { * } } } }$ </td><td colspan="2"> $\mathbf { \overline { { G P T 4 T S } } }$ </td><td colspan="2"> $\mathbf { \overline { { P R A D A } } }$ </td><td colspan="2"> $\mathbf { \overline { { P a t c h T S T } } }$ </td><td colspan="2"> $_ { \mathrm { i T r a n s f o r m e r } } ^ { \mathrm { i T r a n s f o r m e r } }$ </td><td colspan="2"> $\mathbf { \overline { { { M I C N } } } }$ </td><td colspan="2"> $\mathbf { T i m e s N e t }$ </td><td colspan="2">FEDformer [11]</td><td colspan="2"> $\mathbf { D L i n e a r } _ { [ 2 0 ] }$ </td><td colspan="2"> $\mathbf { \Pi ^ { A M D } } _ { [ 4 7 ] }$ </td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTm1</td><td>0.338</td><td>0.369</td><td>0.356</td><td>0.386</td><td>0.352</td><td>0.383</td><td>0.341</td><td>0.376</td><td>0.351</td><td>0.381</td><td>0.373</td><td>0.403</td><td>0.387</td><td>0.411</td><td>0.400</td><td>0.406</td><td>0.448</td><td>0.452</td><td>0.362</td><td>0.379</td><td>0.347</td><td>0.374</td></tr><tr><td>ETTm2</td><td>0.253</td><td>0.303</td><td>0.275</td><td>0.326</td><td>0.266</td><td>0.326</td><td>0.250</td><td>0.309</td><td>0.255</td><td>0.315</td><td>0.274</td><td>0.336</td><td>0.284</td><td>0.340</td><td>0.291</td><td>0.333</td><td>0.305</td><td>0.349</td><td>0.256</td><td>0.331</td><td>0.254</td><td>0.315</td></tr><tr><td>ETTh1</td><td>0.378</td><td>0.407</td><td>0.393</td><td>0.424</td><td>0.427</td><td>0.426</td><td>0.402</td><td>0.422</td><td>0.413</td><td>0.431</td><td>0.438</td><td>0.450</td><td>0.440</td><td>0.462</td><td>0.458</td><td>0.450</td><td>0.440</td><td>0.460</td><td>0.423</td><td>0.437</td><td>0.407</td><td>0.424</td></tr><tr><td>ETTh2</td><td>0.336</td><td>0.378</td><td>0.356</td><td>0.399</td><td>0.354</td><td>0.394</td><td>0.348</td><td>0.391</td><td>0.330</td><td>0.379</td><td>0.367</td><td>0.407</td><td>0.402</td><td>0.437</td><td>0.414</td><td>0.427</td><td>0.437</td><td>0.449</td><td>0.431</td><td>0.447</td><td>0.351</td><td>0.392</td></tr><tr><td>Traffic</td><td>0.377</td><td>0.249</td><td>0.394</td><td>0.272</td><td>0.414</td><td>0.294</td><td>0.395</td><td>0.278</td><td>0.391</td><td>0.264</td><td>0.380</td><td>0.272</td><td>0.542</td><td>0.316</td><td>0.620</td><td>0.336</td><td>0.610</td><td>0.376</td><td>0.434</td><td>0.295</td><td>0.468</td><td>0.271</td></tr><tr><td>Electricity</td><td>0.156</td><td>0.246</td><td>0.164</td><td>0.257</td><td>0.167</td><td>0.263</td><td>0.162</td><td>0.258</td><td>0.159</td><td>0.253</td><td>0.177</td><td>0.274</td><td>0.187</td><td>0.295</td><td>0.192</td><td>0.295</td><td>0.214</td><td>0.327</td><td>0.177</td><td>0.224</td><td>0.157</td><td>0.250</td></tr><tr><td>Weather</td><td>0.225</td><td>0.255</td><td>0.238</td><td>0.271</td><td>0.237</td><td>0.270</td><td>0.223</td><td>0.261</td><td>0.226</td><td>0.264</td><td>0.238</td><td>0.273</td><td>0.243</td><td>0.299</td><td>0.259</td><td>0.287</td><td>0.309</td><td>0.360</td><td>0.240</td><td>0.300</td><td>0.222</td><td>0.261</td></tr><tr><td>1st Count</td><td>11</td><td></td><td>0</td><td></td><td></td><td>0</td><td></td><td></td><td></td><td>1</td><td></td><td>0</td><td></td><td>0</td><td></td><td></td><td></td><td>0</td><td></td><td>0</td><td>1</td><td></td></tr></table>

Table 4: Short-term time series forecasting results on M4. The forecasting horizons are in [6, 48] and the three rows provided are weighted averaged from all datasets under diferent sampling intervals. A lower value indicates better performance. Red: the best, Blue: the second best. Appendix A provides full results.

<table><tr><td>Methods</td><td></td><td>FM-LLM</td><td>TIMELLM</td><td>GPT4TS</td><td>TimesNet</td><td>PatchTST</td><td>N-HiTS</td><td>N-BEATS</td><td>DLinear</td><td>FEDformer</td><td>Stationary</td><td>Reformer</td></tr><tr><td></td><td>SMAPE</td><td>11.840</td><td>11.983</td><td>12.69</td><td>12.88</td><td>12.059</td><td>12.035</td><td>12.25</td><td>13.639</td><td>13.16</td><td>12.780</td><td>18.200</td></tr><tr><td>Average</td><td>MASE</td><td>1.585</td><td>1.595</td><td>1.808</td><td>1.836</td><td>1.623</td><td>1.625</td><td>1.698</td><td>2.095</td><td>1.775</td><td>1.756</td><td>4.223</td></tr><tr><td></td><td>OWA</td><td>0.851</td><td>0.859</td><td>0.94</td><td>0.955</td><td>0.869</td><td>0.869</td><td>0.896</td><td>1.051</td><td>0.949</td><td>0.930</td><td>1.775</td></tr></table>

From the perspective of multivariate forecasting, FM-LLM also maintains its advantage on high-dimensional datasets such as Electricity (321 variables) and Trafic (862 variables), significantly outperforming most baseline models. We attribute this advantage to the FAN-MoE decoder architecture, which substantially enhances the expressive power compared to conventional single-layer MLPs. In multivariate settings, the type and combinatorial complexity of input tokens increase dramatically, making it dificult for simple linear projections or MLP decoders to efectively allocate and model such highly heterogeneous information. FM-LLM’s MoE decoder leverages a dynamic routing mechanism to activate the most suitable expert for each subsequence structure, thereby improving the model’s ability to express weak inter-variable correlations and asynchronous periodic patterns, resulting in superior performance.

We further evaluate FM-LLM on the M4 benchmark for short-term forecasting, where the forecasting horizons fall in [6, 48]. Following common practice, we report SMAPE/MASE/OWA and provide weighted averages over all series under diferent sampling intervals. As shown in Table 4, FM-LLM achieves the best overall performance, and the full breakdown is deferred to Appendix A provides full results.

Table 5 presents the forecasting performance on the PEMS datasets (PEMS03, PEMS04, PEMS07, PEMS08), with results averaged over four prediction lengths. Compared to TQ-Net, CycleNet, and iTransformer, the proposed FM-LLM (ours) achieves the lowest MSE and MAE across all datasets. Specifically, FM-LLM outperforms the second-best method by an average of 6% in MSE and 7.5% in MAE, further validating its efectiveness on large-scale multivariate trafic datasets.

As shown in Table 6 under the few-shot setting, FM-LLM achieves performance that even surpasses some fully supervised baselines using only the top 10% of the training data, with particularly strong results on the ETTh1 and ETTh2 datasets. In contrast, several models with larger parameter sizes (e.g.,

Table 5: Forecasting performance on the PEMS datasets.The results are averaged from all four prediction lengths.The best results are in bold and the second best are underlined.
<table><tr><td>Models</td><td colspan="2">PEMS03</td><td colspan="2">PEMS04</td><td colspan="2">PEMS07</td><td colspan="2">PEMS08</td></tr><tr><td></td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>TQNet [44]</td><td>0.097</td><td>0.203</td><td>0.091</td><td>0.197</td><td>0.075</td><td>0.171</td><td>0.142</td><td>0.229</td></tr><tr><td>CycleNet [48]</td><td>0.098</td><td>0.201</td><td>0.090</td><td>0.196</td><td>0.071</td><td>0.170</td><td>0.120</td><td>0.198</td></tr><tr><td>iTransformer [23]</td><td>0.098</td><td>0.203</td><td>0.097</td><td>0.199</td><td>0.139</td><td>0.245</td><td>0.150</td><td>0.226</td></tr><tr><td>FM-LLM(ours)</td><td>0.089</td><td>0.186</td><td>0.089</td><td>0.187</td><td>0.063</td><td>0.151</td><td>0.118</td><td>0.184</td></tr></table>

FEDformer, TimesNet, or AutoTimes) typically sufer performance degradation under limited sample conditions and fail to extract stable and efective temporal features. FM-LLM’s advantage mainly stems from the strong structural inductive bias introduced by its frequency-enhanced encoder and expert-routing decoder, enabling robust learning of periodic structures and key dynamic features even in low-sample regimes and showcasing excellent few-shot generalization.

Table 7 further investigates the zero-shot transfer performance of FM-LLM under the setting where training and testing are conducted on entirely diferent datasets. In this setup, the model is trained solely on a source dataset and directly applied to a target dataset at test time without any fine-tuning. FM-LLM achieves the best results across all four transfer scenarios (e.g., ETTh1 → ETTh2, ETTm1 → ETTm2), demonstrating high robustness under task transfer, distribution shift, and variable structural changes. This capability is partly attributed to the frequency-space representations modeled by FM-LLM, which, via the FAN module, capture the intrinsic periodicity and dynamic patterns of input sequences and naturally support cross-domain transfer. Simultaneously, the MoE decoder flexibly activates the appropriate feature pathways when handling data from diferent sources, thereby adapting to input structural variations and ensuring stable outputs.

Table 6: Few-shot learning results on 10% training data of ETT datasets. The best results are in bold and the second best are underlined.
<table><tr><td rowspan="2">Models</td><td colspan="2">ETTm1</td><td colspan="2">ETTm2</td><td colspan="2">ETTh1</td><td colspan="2">ETTh2</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>DLinear</td><td>0.411</td><td>0.429</td><td>0.316</td><td>0.368</td><td>0.647</td><td>0.552</td><td>0.605</td><td>0.538</td></tr><tr><td>TimesNet</td><td>0.673</td><td>0.534</td><td>0.321</td><td>0.354</td><td>0.869</td><td>0.628</td><td>0.479</td><td>0.465</td></tr><tr><td>FEDformer</td><td>0.696</td><td>0.572</td><td>0.356</td><td>0.392</td><td>0.639</td><td>0.561</td><td>0.466</td><td>0.475</td></tr><tr><td>iTransformer</td><td>0.728</td><td>0.565</td><td>0.336</td><td>0.373</td><td>0.910</td><td>0.860</td><td>0.489</td><td>0.415</td></tr><tr><td>PatchTST</td><td>0.501</td><td>0.466</td><td>0.296</td><td>0.343</td><td>0.633</td><td>0.542</td><td>0.415</td><td>0.431</td></tr><tr><td>GPT4TS</td><td>0.461</td><td>0.441</td><td>0.293</td><td>0.335</td><td>0.590</td><td>0.525</td><td>0.397</td><td>0.421</td></tr><tr><td>AutoTimes</td><td>0.611</td><td>0.508</td><td>0.289</td><td>0.339</td><td>0.592</td><td>0.549</td><td>0.372</td><td>0.418</td></tr><tr><td>PRADA</td><td>0.435</td><td>0.428</td><td>0.281</td><td>0.329</td><td>0.557</td><td>0.515</td><td>0.384</td><td>0.419</td></tr><tr><td>FM-LLM</td><td>0.457</td><td>0.445</td><td>0.259</td><td>0.316</td><td>0.489</td><td>0.482</td><td>0.348</td><td>0.399</td></tr></table>

Table 7: Zero-shot learning results on ETT datasets. A → B means that we train on dataset A while testing the model performance on dataset B. The best results are in bold and the second best are underlined.
<table><tr><td>Models</td><td>h1 → h2 MSE</td><td>MAE</td><td>h1 → m2 MSE</td><td>MAE</td><td>h2 → h1 MSE</td><td>MAE</td><td>h2 → m2 MSE</td><td>MAE</td></tr><tr><td>DLinear</td><td>0.493</td><td>0.488</td><td>0.415</td><td>0.452</td><td>0.703</td><td>0.574</td><td>0.328</td><td>0.386</td></tr><tr><td>TimesNet</td><td>0.421</td><td>0.431</td><td>0.327</td><td>0.361</td><td>0.865</td><td>0.621</td><td>0.342</td><td>0.376</td></tr><tr><td>PatchTST</td><td>0.380</td><td>0.405</td><td>0.597</td><td>0.483</td><td>0.565</td><td>0.513</td><td>0.325</td><td>0.365</td></tr><tr><td>GPT4TS</td><td>0.406</td><td>0.422</td><td>0.325</td><td>0.363</td><td>0.757</td><td>0.578</td><td>0.335</td><td>0.370</td></tr><tr><td>AutoTimes</td><td>0.355</td><td>0.394</td><td>0.347</td><td>0.374</td><td>0.565</td><td>0.529</td><td>0.322</td><td>0.369</td></tr><tr><td>PRADA</td><td>0.350</td><td>0.392</td><td>0.307</td><td>0.357</td><td>0.444</td><td>0.457</td><td>0.296</td><td>0.349</td></tr><tr><td>FM-LLM</td><td>0.349</td><td>0.384</td><td>0.297</td><td>0.349</td><td>0.448</td><td>0.455</td><td>0.283</td><td>0.340</td></tr></table>

In comparison, some representative large language model paradigms for time series (LLM4TS), such as PRADA and AutoTimes, also exhibit certain advantages in zero-shot and few-shot settings but follow diferent methodological paths than FM-LLM. AutoTimes adopts a tokenization mechanism to convert time series into language-like sequences and uses a frozen language model backbone with trainable projection layers for transfer learning. While this approach benefits from the strong contextual modeling capacity of LLMs and the natural generalization of in-context learning, it faces limitations: (1) token representations are task-sensitive and prone to distortion under significant frequency structure changes; (2) the large model size hinders deployment in industrial or edge scenarios. PRADA, by introducing modality alignment and dynamic adaptation mechanisms, preserves pretraining advantages while enabling taskspecific adaptation and achieves solid zero-shot performance, but its explicit modeling of input distribution shifts is still less intuitive and stable compared to FM-LLM’s frequency decomposition approach.

In summary, FM-LLM incorporates a Fourier Analysis Network (FAN) in the encoder to model subsequence frequency structures, adopts an MoE module in the decoder to enhance expressive diversity, and introduces a time-frequency hybrid loss function to improve long-range modeling precision and cross-task generalization. It achieves state-of-the-art performance across supervised, fewshot, and zero-shot settings. Compared to existing large-model paradigms, FM-LLM demonstrates a favorable balance between interpretability, adaptability, and structural transferability, ofering a practical and generalizable approach for future time-series forecasting model design.

## 4.3. Ablation study

To thoroughly assess the contributions of individual modules in FM-LLM, we conducted two sets of ablation studies on the ETT dataset series, as reported in Table 8. The study (Table 8) examines the efect of removing or replacing major architectural components, including the frequency-aware encoder, the

Table 8: Ablation study on the ETTh1 dataset. Lower is better. The best results are in bold and the second best are underlined.
<table><tr><td>Variant</td><td>ETTh1-96 MSE MAE</td><td>ETTh1-192 MSE MAE</td></tr><tr><td>FM-LLM</td><td>0.342 0.380</td><td>0.377 0.403</td></tr><tr><td> $\mathrm { w / o }$  FAN</td><td>0.351 0.390</td><td>0.381 0.411</td></tr><tr><td> $\mathrm { w / o }$  MOE</td><td>0.348 0.390</td><td>0.383 0.412</td></tr><tr><td> $\mathrm { w / o }$  LLM</td><td>0.370 0.406</td><td>0.404 0.428</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ \mathcal { L } _ { \mathrm { f r e q } }$ </td><td>0.352 0.387</td><td>0.389 0.412</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ \mathcal { L } _ { \mathrm { t i m e } }$ </td><td>0.357 0.393</td><td>0.391 0.416</td></tr><tr><td> $\mathrm { F A N }  \mathrm { L i n e a r }$ </td><td>0.361 0.394</td><td>0.399 0.418</td></tr></table>

FAN-MoE decoder, the pretrained LLM backbone, and the hybrid loss function. Results across two forecasting horizons reveal that removing the FAN encoder consistently leads to significant performance degradation, especially for longer horizons, highlighting its critical role in capturing periodic structures in the frequency domain. Similarly, the removal of the MoE decoder, which is responsible for adaptive decoding of heterogeneous sequence patterns, results in notable performance drops, particularly on complex multivariate inputs. Under the condition of strict parameter parity, removing the pre-trained large language model weights and training from scratch resulted in the most significant degradation in model performance. This indicates that the performance gains are attributed not only to model capacity but also to the transferred pre-training priors.

To strictly disentangle the contribution of pre-trained linguistic knowledge from the model capacity, we conducted a comprehensive ablation study. Specifically, we aim to verify whether the performance gains of FM-LLM stem from the universal patterns embedded in the LLM backbone or merely from the deep network architecture. The results in Table 9 demonstrate that while the pretrained backbone ofers moderate accuracy gains in full-data settings, its contribution is decisive in few-shot scenarios. Specifically, the scratch-trained model sufers from severe overfitting when data is scarce (e.g., MSE spikes to 0.729 on ETTh1), whereas FM-LLM maintains robust performance (MSE 0.451), confirming that the pre-trained weights are essential for generalization capabilities.

Table 9: Comparisons of Pre-training Efectiveness on ETTh1 and ETTm1 datasets under Full (100%) and Few-shot (10%) settings. FM-LLM denotes our proposed method with pretrained weights. w/o Pretrain denotes the same architecture with random initialization. Scratch-trained denotes a standard Transformer of comparable parameter size trained from scratch. The best results are highlighted in bold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td colspan="9">Full Training Data (100%)</td></tr><tr><td rowspan="3">ETTh1</td><td>FM-LLM</td><td>0.342</td><td>0.380</td><td>0.377</td><td>0.403</td><td>0.395</td><td>0.415</td><td>0.397</td><td>0.429</td></tr><tr><td>w/o Pretrain</td><td>0.348</td><td>0.387</td><td>0.381</td><td>0.409</td><td>0.398</td><td>0.422</td><td>0.408</td><td>0.441</td></tr><tr><td>Scratch-trained</td><td>0.370</td><td>0.406</td><td>0.404</td><td>0.428</td><td>0.420</td><td>0.440</td><td>0.432</td><td>0.460</td></tr><tr><td rowspan="3">ETTm1</td><td>FM-LLM</td><td>0.276</td><td>0.327</td><td>0.319</td><td>0.356</td><td>0.351</td><td>0.379</td><td>0.406</td><td>0.415</td></tr><tr><td>w/o Pretrain</td><td>0.285</td><td>0.337</td><td>0.325</td><td>0.363</td><td>0.362</td><td>0.385</td><td>0.432</td><td>0.423</td></tr><tr><td>Scratch-trained</td><td>0.296</td><td>0.347</td><td>0.341</td><td>0.376</td><td>0.376</td><td>0.399</td><td>0.439</td><td>0.436</td></tr><tr><td colspan="10">Few-shot Training Data (10%)</td></tr><tr><td rowspan="3">ETTh1</td><td>FM-LLM</td><td>0.451</td><td>0.454</td><td>0.481</td><td>0.472</td><td>0.497</td><td>0.486</td><td>0.525</td><td>0.514</td></tr><tr><td>w/o Pretrain</td><td>0.476</td><td>0.466</td><td>0.505</td><td>0.481</td><td>0.529</td><td>0.496</td><td>0.580</td><td>0.527</td></tr><tr><td>Scratch-trained</td><td>0.551</td><td>0.511</td><td>0.595</td><td>0.537</td><td>0.620</td><td>0.554</td><td>0.696</td><td>0.597</td></tr><tr><td rowspan="3">ETTm1</td><td>FM-LLM</td><td>0.383</td><td>0.403</td><td>0.422</td><td>0.427</td><td>0.463</td><td>0.450</td><td>0.561</td><td>0.498</td></tr><tr><td>w/o Pretrain</td><td>0.405</td><td>0.416</td><td>0.440</td><td>0.438</td><td>0.476</td><td>0.459</td><td>0.567</td><td>0.504</td></tr><tr><td>Scratch-trained</td><td>0.533</td><td>0.466</td><td>0.580</td><td>0.491</td><td>0.639</td><td>0.519</td><td>0.846</td><td>0.603</td></tr></table>

Further, the hybrid time-frequency loss proves to be an integral design choice.

![](images/9424566ab41e31c24ca64fb4fe22dd3efdc22d7fd034a604cba77f56fa9f2abe.jpg)  
Figure 3: The case study on ETT datasets. For convenience, we set both the lookback length and the forecasting horizon to 96. The blue line represents the ground-truth and the orange line denotes the forecasting result.

Ablating the frequency-domain loss slightly reduces the model’s ability to align periodic behavior, while removing the time-domain loss has a more pronounced negative impact on numerical prediction accuracy. These findings indicate that both loss components are complementary: time-domain loss governs local precision, while frequency-domain loss ensures global structural consistency. Additionally, replacing the FAN module with a linear projection leads to a consistent drop in performance, underscoring the importance of explicit spectral modeling over simple linear mappings.

## 4.4. Visualization Analysis

Figure 3 presents the prediction composition of the FM-LLM model on four ETT datasets (ETTh1, ETTh2, ETTm1, and ETTm2), visually illustrating the structural division of labor and cooperative mechanisms between the shared Fourier Experts and Routed Experts. According to the model design, the final prediction result (denoted as “Result”) is obtained by a weighted summation of the outputs from both expert groups. As clearly observed in the figure, the two types of experts exhibit distinct modeling behaviors: Fourier Experts tend to learn stable periodic structures and trend dynamics, producing smooth and consistent outputs that demonstrate strong global modeling capabilities. In contrast, Routed Experts exhibit a higher degree of dynamic responsiveness, adapting flexibly to local disturbances, phase shifts, or abrupt pattern changes in the input sequence, and providing structural compensation accordingly.

In the ETTh1 dataset which is characterized by strong periodicity and smooth variations—the output of Fourier Experts closely aligns with the final Result, indicating their dominant role in modeling the overall sequence. The contributions from Routed Experts are limited to subtle adjustments at a few inflection points, ofering marginal refinements to the Fourier structure. In ETTh2, while the overall trend is still primarily captured by the Fourier Experts, Routed Experts play a more significant role when phase misalignments occur, providing necessary structural corrections that bring the Result closer to the ground truth. This demonstrates a clear complementary relationship between the two expert groups.

The distinction becomes more pronounced in the medium-frequency minutelevel datasets ETTm1 and ETTm2, which contain substantial high-frequency fluctuations and short-period disruptions. In these settings, the outputs of Fourier Experts struggle to fit such non-stationary patterns, often resulting in lagging or overly smoothed predictions. Routed Experts, however, exhibit stronger localized responsiveness, efectively capturing abrupt dynamics in the input series. By integrating the temporal sensitivity of Routed Experts with the periodic backbone provided by Fourier Experts, the final Result curve achieves superior fitting performance compared to either expert group alone. This phenomenon further validates a core design hypothesis of FM-LLM: through the expert division and dynamic routing mechanism in its MoE decoder, the model can automatically identify and activate the most appropriate expert sub-network based on the characteristics of each input subsequence. This enables structured modeling of varying frequencies, fluctuation rates, and heterogeneous patterns. The expert cooperation–division–reconstruction process not only enhances the accuracy of long-sequence forecasting, but also provides a mechanistic explanation for the model’s robust generalization in both few-shot and zero-shot scenarios.

Figure4 presents a comprehensive analysis of FM-LLM’s sensitivity to four key architectural and training hyperparameters on the ETTm1 dataset: the number of routed experts (TopK), the number of Fourier experts, the number of LLM layers, and the lookback input length T. Each subfigure reports the MSE for diferent prediction horizons.Based on our analysis, we have made the following observations:

(1) The model exhibits stable performance across diferent TopK values. However, when the number of routed experts exceeds 4, a slight increase in MSE is observed, likely due to reduced expert specialization and routing redundancy. The best performance is generally achieved with TopK = 2 or 3, which balances diversity and computational eficiency. (2) Introducing Fourier experts slightly improves model performance. Using two Fourier experts yields the most consistent gains across prediction lengths. Adding more (e.g., 3) leads to diminishing returns, suggesting that a small Fourier capacity sufices to capture general temporal patterns. (3) FM-LLM is relatively insensitive to the number of backbone transformer layers. While the overall trends are flat, slight improvements are observed around 9–12 layers, particularly for long-horizon prediction settings. This indicates that moderate-depth architectures can enhance representation power without overfitting. (4) Increasing the lookback length consistently reduces MSE across all forecasting horizons. Performance improves significantly from T = 384 to $T = 7 6 8$ , beyond which the benefit saturates. This suggests that T = 672 provides an efective historical context for long-term forecasting tasks. Overall, FM-LLM demonstrates robustness and adaptability to a range of hyperparameter choices, reducing the burden of manual tuning and improving its applicability to real-world time-series forecasting scenarios. We also investigated the impact of the number of FAN layers on forecasting performance. The detailed experimental results are provided in Appendix B . Empirical evidence suggests that stacking additional layers does not yield performance gains; specifically, the single-layer architecture consistently outperformed the 2-layer and 3-layer configurations across the two datasets. Therefore, we adopt the

![](images/9e89c84d7d839fabbd51fb322863fcbe0367b0d64f277eff64a8ace4a7ac43ea.jpg)  
Figure 4: Analysis of hyperparameter sensitivity on ETTm1 dataset.

![](images/f26f82e11163a7813706303a0534ea805aeabfa3edd76aa93b16b0832270754a.jpg)

![](images/651f8283b9ed115bf720558b228b430d50a2dd7ece7d491084a635020e0cd50f.jpg)

![](images/dd6f30116fc3d1dd67da003fc9525885919e5591ee8a85542bd86dba9b5978b4.jpg)

![](images/a43e4726d0351c20995f0c2a4dd8fae73514cca10409c94dbf36cb866c55517c.jpg)  
Figure 5: Expert load histograms across diferent datasets.

1-layer FAN as the default setting for eficiency and robustness

Figure 5 presents the expert load histograms across four representative datasets. On ETTh2, the routing distribution collapses to a single expert, suggesting that the temporal patterns are simple and can be modeled without expert diversity. ETTm1 exhibits moderate imbalance, where one expert dominates while others are partially activated. In contrast, the Trafic dataset demonstrates a near-uniform expert utilization, reflecting its high-dimensional, heterogeneous structure and justifying the necessity of distributed expert computation. For Electricity, the router favors three experts while suppressing others, indicating partial mode specialization but still revealing under-utilization. These patterns confirm that the MoE decoder adapts dynamically to data complexity and structural diversity.

Figure 6 presents model performance under varying training data sizes on the ETTh1 and ETTm2 datasets. FM-LLM consistently outperforms both GPT4TS and PatchTST across all data availability levels and prediction horizons, with especially large margins under low-resource settings. This demonstrates the model’s strong few-shot generalization ability, likely attributed to its frequency-aware encoding and expert-based modular decoding design.

Interestingly, we observe a non-monotonic performance trend on ETTm2, where FM-LLM achieves its best accuracy not at 100% supervision, but rather at 50% and 80% training data. This indicates that increasing data volume does not necessarily yield better performance—a phenomenon previously observed in transformer-based models [37]. However, we argue that this limitation is not inherent to the transformer architecture itself, but rather stems from distribution shift within the dataset. In time-series benchmarks like ETTm2, earlier and later data segments often difer in seasonal patterns, anomaly frequency, or system conditions. As a result, using the entire dataset for training may introduce inconsistent or noisy patterns, leading to suboptimal generalization. In contrast, subsets like 50%–80% may better align with the test distribution, enabling stronger predictive accuracy.

## 4.5. Frequency-Domain Analysis of Few-Shot Generalization

Figure 7 provides a frequency–domain explanation for the heterogeneous few–shot performance of FM-LLM across the four ETT benchmarks. Using a sliding window of length 96, we compute the averaged power–spectral density (PSD) for the 10 % and 100 % training subsets, quantify their divergence via the Kullback–Leibler (KL) distance, and assess sequence predictability with the forecastability [43] index $\Omega = 1 - H / \ln N . ^ { \cdot }$

![](images/1f699d6529269e5033506a3c418c3a4cabf200406b5ff238dba532dd43311e6d.jpg)  
(a) ETTh1

![](images/b611efaeb47b8c569dd4630a1decfb132d81bc1476b9062683561b6d532b1f3c.jpg)  
(b) ETTm2

Figure 6: Results on various percentages of ETTh1 and ETTm2. Line color represents diferent models and line style means various prediction lengths T ∈ {96, 192}  
![](images/b00d87c849620eb072743a522d2f1aa4fcda8aa8f2059b43d6b0944d9be0a5fe.jpg)  
(a) ETTh1

![](images/3d14dd602a23167eb06a60de701b049e688b04c9d60f04aa8234d96c584cc14b.jpg)

![](images/0c3524bc33382d8101ba2feafc29ba6532934a081b4dd619c58d67b5f41109a2.jpg)  
(c) ETTm1

(b) ETTh2  
![](images/d777f5234ba8b8f484f852bd0bffbd8d9ce8bd1bf2a3d506c8c083eef9ba2293.jpg)  
(d) ETTm2  
Figure 7: Token-wise frequency energy proportion comparison between 10% and 100% training data on the ETT dataset.

ETTh1 and ETTm1 exhibit the largest spectral mismatches $( \mathrm { K L } = 2 . 3 6 \times$

$1 0 ^ { - 1 }$ and $1 . 0 1 \times 1 0 ^ { - 1 }$ , respectively) and a marked drop in Ω $( 0 . 5 3 6 \to 0 . 3 7 0 ;$ $0 . 7 2 9 \substack {  0 . 5 7 2 } )$ , indicating that high- and mid-frequency components are poorly represented when only 10 % of the data are available—precisely the datasets where FM-LLM records the highest few-shot error in Table 6. In contrast, ETTm2 shows an almost perfect spectral overlap $( \mathrm { K L } = 1 . 3 8 \times 1 0 ^ { - 3 } )$ and a virtually unchanged Ω, implying that even a small subset preserves its dominant frequency structure and thus sustains robust accuracy.

These findings suggest that FM-LLM’s few-shot generalization hinges not only on its frequency-aware architecture but also on whether the reduced training set faithfully reproduces the original energy distribution; spectral divergence therefore emerges as a key prior for estimating few-shot dificulty and expected performance.

## 4.6. Analysis of Expert Specialization

To provide a deeper insight into the internal working mechanism of FM-LLM, we conduct a comprehensive visualization analysis on the Trafic dataset. Specifically, we examine the model from three perspectives: the frequency response of diferent experts, the specialization patterns of experts, and the temporal dynamics of the routing strategy.

Frequency Domain Decomposition. As illustrated in Figure 8, we analyze the spectral characteristics of the learned representations. The frequency response comparison reveals distinct roles: the Fourier experts efectively capture the dominant low-frequency components, which correspond to the strong daily and weekly periodicities inherent in the Trafic dataset. In contrast, the Routed experts exhibit a more diverse spectral focus, attending to high-frequency variations that represent local fluctuations and transient trafic anomalies. This complementary frequency coverage ensures that the model preserves global trends while retaining sensitivity to detailed temporal dynamics.

![](images/c8623b65ac7c64ecff59466e24e4ea489a6f64fef52cc5ef27e233fd75a467b4.jpg)  
Figure 8: Frequency Response Analysis on Trafic Datasets

![](images/bb3e71350bc32bf23ec1aeb9bbc2e8a74b9d67b1d01231547454619a934b0921.jpg)  
Figure 9: Visualization of Expert Specialization

![](images/0ce9877e77348598618e74f0c46f48a86adb2630866f538a98814aa22d1efdda.jpg)  
Figure 10: Temporal Dynamics of Top-3 Expert Routing

Expert Specialization Patterns. Figure 9 visualizes the learned preferences of diferent experts across the trafic data samples. We observe a clear emergence of functional specialization without explicit supervision. Specific experts are consistently activated for samples exhibiting high-volatility congestion patterns, whereas others specialize in modeling stable, free-flow trafic periods. This heterogeneity suggests that the MoE framework successfully decomposes the complex trafic distribution into distinct sub-regimes, allowing each expert to master a specific subset of trafic behaviors.

Temporal Routing Dynamics. To understand how the model adapts over time, Figure 10 depicts the temporal evolution of the Top-3 activated experts over a continuous forecasting window. The routing mechanism demonstrates dynamic adaptability; This indicates that the router dynamically modulates the combination of experts in response to the changing statistical properties of the trafic flow, efectively switching strategies to match the instantaneous context.

## 4.7. Eficiency Analysis

To provide a thorough eficiency assessment, we compare FM-LLM with several competitive baselines, including LLM-based forecasting methods (GPT4TS and TimeLLM) and strong non-LLM backbones (PatchTST and DLinear). All experiments are implemented in PyTorch with a fixed batch size of 128. We report four eficiency metrics: the number of trainable parameters, inference latency (ms/iter), GPU memory footprint, and peak GPU memory usage. Importantly, all memory statistics are measured during inference on a trained model (forward pass only), without gradient computation. Moreover, FM-LLM is trained once and can be directly applied to multiple prediction lengths without retraining, which further improves its practicality in real-world deployments.

As shown in Table 10, FM-LLM exhibits a favorable accuracy–eficiency trade-of on the Trafic dataset. Under the long-horizon setting (Pred=720), FM-LLM uses 14.3M trainable parameters and runs at 143.48 ms/iter, whereas TimeLLM is an order of magnitude slower (1056.0 ms/iter) and requires substantially more GPU memory (22.6 GB footprint and 11.1 GB peak). Compared with PatchTST, FM-LLM incurs higher computation and memory cost, yet it remains markedly more eficient than LLM-based baselines. Under the shorthorizon setting (Pred=96), FM-LLM further reduces latency to 17.93 ms/iter while maintaining a stable memory footprint (3.23 GB), whereas TimeLLM still sufers from extremely high latency (1039.6 ms/iter) and memory usage (22.6 GB footprint). Overall, FM-LLM substantially improves practicality over existing LLM-based forecasting baselines while maintaining competitive forecasting performance.

## 4.8. Spectral Analysis of Model Predictions

To demonstrate the efectiveness of FM-LLM, we visualize the time-domain trajectories and the corresponding frequency spectra of the ground truth and model predictions. As shown in Fig. 11, we present representative spectrogram cases on ETTh1 under the input-672-predict-96 setting, comparing the ground truth and diferent models. We observe that FM-LLM aligns better with the dominant frequency components and preserves the harmonic structure more faithfully, which is consistent with its frequency-enhanced token adaptation design.

![](images/13baecbb894e78b8298de9372ec8f9a24f892c95ec5bdd169df773a20800d74f.jpg)  
Figure 11: Prediction spectrogram cases from ETTh1 by ground truth and diferent models under the input-672-predict-96 settings.

## 5. Conclusion

In this work, we propose FM-LLM, a frequency-aware framework specifically designed for adapting frozen LLMs to long-term time-series forecasting. Departing from the conventional reliance on prompt engineering, FM-LLM resolves the modality gap and the superimposed entanglement of temporal dynamics through a purposeful, constrained asymmetric coupling. By integrating explicit spectral token alignment at the encoding stage and an asymmetric expert role division at the decoding stage, FM-LLM achieves a structural decoupling of global periodic backbones and local non-stationary residuals. Extensive experiments demonstrate that FM-LLM not only achieves state-of-the-art accuracy across full-data and few-shot settings but also maintains high computational eficiency by avoiding sequence-length expansion. These findings suggest that the structural constraints and spectral inductive biases in FM-LLM provide a robust, interpretable, and eficient solution for cross-modal time-series modeling.

Table 10: Eficiency comparison on Trafic. We report trainable parameters, inference time (ms/iter), GPU memory footprint (MB), peak memory (MB), and MSE performance under diferent prediction lengths. Note: The higher latency compared to PatchTST is a strategic trade-of for gaining LLM-based few-shot generalization.
<table><tr><td>Dataset</td><td>Pred</td><td>Model</td><td>MSE</td><td>Trainable Params</td><td>Inference Time (ms/it)</td><td>GPU Mem Footprint</td><td>Peak Mem</td></tr><tr><td rowspan="9">Traffic</td><td rowspan="6">720</td><td>FM-LLM</td><td>0.419</td><td>14,329,800</td><td>143.48</td><td>3,232</td><td>3,184</td></tr><tr><td>GPT4TS</td><td>0.450</td><td>94,407,376</td><td>139.95</td><td>3,734</td><td>3,496</td></tr><tr><td>TimeLLM</td><td>0.430</td><td>131,312,824</td><td>1,056.0</td><td>22,592</td><td>11,122</td></tr><tr><td>PatchTST</td><td>0.432</td><td>6,310,224</td><td>2.3</td><td>566</td><td>205</td></tr><tr><td>DLinear</td><td>0.466</td><td>738,720</td><td>0.47</td><td>48</td><td>28</td></tr><tr><td>FM-LLM</td><td>0.345</td><td>14,329,800</td><td>17.93</td><td>3,232</td><td>3,184.8</td></tr><tr><td rowspan="5"></td><td>GPT4TS</td><td>0.388</td><td>12,617,824</td><td>138.70</td><td>3,422</td><td>3,184</td></tr><tr><td>TimeLLM</td><td>0.362</td><td>130,034,248</td><td>1,039.6</td><td>22,592</td><td>11,110.5</td></tr><tr><td>PatchTST</td><td>0.360</td><td>1,197,792</td><td>2.2</td><td>166.2</td><td>464</td></tr><tr><td>DLinear</td><td>0.388</td><td>98,496</td><td>0.33</td><td>28</td><td>20.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

In future work, we aim to explore more lightweight expert modules and parameter-eficient fine-tuning (PEFT) methods, such as LoRA, to further minimize adaptation overhead and improve eficiency. We also plan to develop more robust expert routing mechanisms to stabilize training and ensure balanced expert utilization. Furthermore, we will extend FM-LLM to broader time-series applications, specifically targeting communication network trafic prediction. Given the model’s strength in decoupling periodic tidal patterns from bursty anomalies, it holds significant potential for optimizing 5G/6G network slicing and base station load balancing. Finally, at present, the loss function adopts fixed weighting coeficients; in future work, we aim to develop a loss formulation that dynamically adjusts task weights based on gradient magnitudes. Alongside this, we will investigate advanced autoregressive training strategies—such as continuous-variable classification and frequency-domain masked mechanisms—to further enhance sequence modeling performance without compromising spectral continuity.

## Appendix A. Full Results on Long-Term and Short-Term Forecasting

This appendix reports the complete results that are omitted from the main text for clarity. Table A.11 provides the full breakdown of M4 short-term forecasting results by sampling intervals. Table A.12 reports long-term forecasting results across prediction lengths {96, 192, 336, 720}.

Table A.11: Short-term forecasting results on the M4 benchmark (SMAPE/MASE/OWA). The best results are highlighted in bold red and the second best are underlined blue.
<table><tr><td></td><td>Methods</td><td>FM-LLM (Ours)</td><td>TIME-LLM</td><td>GPT4TS</td><td>TimesNet</td><td>PatchTST</td><td>N-HiTS</td><td>N-BEATS</td><td>DLinear</td><td>FEDformer</td><td>Stationary</td><td>Reformer</td></tr><tr><td></td><td>SMAPE</td><td>13.403</td><td>13.419</td><td>15.11</td><td>15.378</td><td>13.477</td><td>13.422</td><td>13.487</td><td>16.965</td><td>14.021</td><td>13.717</td><td>16.169</td></tr><tr><td>Yearly</td><td>MASE</td><td>2.993</td><td>3.005</td><td>3.565</td><td>3.554</td><td>3.019</td><td>3.056</td><td>3.036</td><td>4.283</td><td>3.036</td><td>3.078</td><td>3.800</td></tr><tr><td></td><td>OWA</td><td>0.787</td><td>0.789</td><td>0.911</td><td>0.918</td><td>0.792</td><td>0.795</td><td>0.795</td><td>1.058</td><td>0.811</td><td>0.807</td><td>0.973</td></tr><tr><td>uarterii</td><td>SMAPE</td><td>10.076</td><td>10.110</td><td>10.597</td><td>10.465</td><td>10.38</td><td>10.185</td><td>10.564</td><td>12.145</td><td>11.1</td><td>10.958</td><td>13.313</td></tr><tr><td></td><td>MASE</td><td>1.177</td><td>1.178</td><td>1.253</td><td>1.227</td><td>1.233</td><td>1.18</td><td>1.252</td><td>1.520</td><td>1.35</td><td>1.325</td><td>1.775</td></tr><tr><td></td><td>OWA</td><td>0.887</td><td>0.889</td><td>0.938</td><td>0.923</td><td>0.921</td><td>0.893</td><td>0.936</td><td>1.106</td><td>0.996</td><td>0.981</td><td>1.252</td></tr><tr><td></td><td>SMAPE</td><td>12.699</td><td>12.980</td><td>13.258</td><td>13.513</td><td>12.959</td><td>13.059</td><td>13.089</td><td>13.514</td><td>14.403</td><td>13.917</td><td>20.128</td></tr><tr><td></td><td>MASE</td><td>0.939</td><td>0.963</td><td>1.003</td><td>1.039</td><td>0.97</td><td>1.013</td><td>0.996</td><td>1.037</td><td>1.147</td><td>1.097</td><td>2.614</td></tr><tr><td>Monhio</td><td>OWA</td><td>0.882</td><td>0.903</td><td>0.931</td><td>0.957</td><td>0.905</td><td>0.929</td><td>0.922</td><td>0.956</td><td>1.038</td><td>0.998</td><td>1.927</td></tr><tr><td>Others</td><td>SMAPE</td><td>4.861</td><td>4.795</td><td>6.124</td><td>6.913</td><td>4.952</td><td>4.711</td><td>6.599</td><td>6.709</td><td>7.148</td><td>6.302</td><td>32.491</td></tr><tr><td></td><td>MASE</td><td>3.261</td><td>3.178</td><td>4.116</td><td>4.507</td><td>3.347</td><td>3.054</td><td>4.43</td><td>4.953</td><td>4.041</td><td>4.064</td><td>33.355</td></tr><tr><td></td><td>OWA</td><td>1.026</td><td>1.006</td><td>1.259</td><td>1.438</td><td>1.049</td><td>0.977</td><td>1.393</td><td>1.487</td><td>1.389</td><td>1.304</td><td>8.679</td></tr><tr><td>Average</td><td>SMAPE</td><td>11.840</td><td>11.983</td><td>12.69</td><td>12.88</td><td>12.059</td><td>12.035</td><td>12.25</td><td>13.639</td><td>13.16</td><td>12.780</td><td>18.200</td></tr><tr><td></td><td>MASE</td><td>1.585</td><td>1.595</td><td>1.808</td><td>1.836</td><td>1.623</td><td>1.625</td><td>1.698</td><td>2.095</td><td>1.775</td><td>1.756</td><td>4.223</td></tr><tr><td></td><td>OWA</td><td>0.851</td><td>0.859</td><td>0.94</td><td>0.955</td><td>0.869</td><td>0.869</td><td>0.896</td><td>1.051</td><td>0.949</td><td>0.930</td><td>1.775</td></tr></table>

Table A.12: Forecasting performance (MSE/MAE) on benchmark datasets. AutoTimes\* denotes the use of Llama3.2-1B backbone. The best results are highlighted in bold red and the second best are underlined blue.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Models</td><td colspan="2">FM-LLM</td><td colspan="2">AutoTimes* [6]</td><td colspan="2">GPT4TS [9]</td><td colspan="2">PRADA [8]</td><td colspan="2">PatchTST [22] iTransformer [23]</td><td colspan="2">MICN [17]</td><td colspan="2">TimesNet [12]</td><td colspan="2">FEDformer [11]</td><td colspan="2">DLinear [20]</td><td colspan="2">AMD [47]</td></tr><tr><td>MSE</td><td>MAE MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td></tr><tr><td rowspan="5">ETTm1</td><td>96</td><td>0.276 0.327</td><td>0.289</td><td>0.344</td><td>0.292</td><td>0.346 0.288</td><td>0.342</td><td>0.290</td><td>0.342</td><td>0.309</td><td>0.365</td><td>0.316</td><td>0.364</td><td>0.338</td><td>0.375</td><td>0.764 0.426</td><td>0.416</td><td>0.299 0.355</td><td>0.343</td><td>0.284 0.339</td></tr><tr><td>192</td><td>0.319 0.356</td><td>0.336</td><td>0.373</td><td>0.332 0.372</td><td>0.326</td><td>0.366</td><td>0.332</td><td>0.369</td><td>0.348</td><td>0.387</td><td>0.363</td><td>0.390</td><td>0.371</td><td>0.387</td><td></td><td>0.441</td><td></td><td>0.365</td><td>0.322 0.362</td></tr><tr><td>336</td><td>0.351 0.379</td><td>0.372</td><td>0.395</td><td>0.366</td><td>0.394</td><td>0.350</td><td>0.384</td><td>0.392</td><td>0.378</td><td>0.406</td><td>0.408</td><td>0.426</td><td>0.410</td><td>0.411</td><td>0.445</td><td>0.459</td><td>0.369</td><td>0.386</td><td>0.360 0.380</td></tr><tr><td>720</td><td>0.406 0.415</td><td>0.427</td><td>0.430</td><td>0.417</td><td>0.421 0.399</td><td>0.413</td><td>0.366 0.420</td><td>0.424</td><td>0.456</td><td>0.455</td><td>0.459</td><td>0.464</td><td>0.478</td><td>0.450</td><td>0.543</td><td>0.490</td><td>0.425</td><td>0.421</td><td>0.416 0.414</td></tr><tr><td>avg</td><td>0.338</td><td>0.356</td><td>0.386</td><td>0.352</td><td>0.383</td><td>0.341</td><td>0.376 0.351</td><td>0.381</td><td>0.373</td><td>0.403</td><td>0.387</td><td>0.411</td><td>0.400</td><td>0.406</td><td>0.448</td><td>0.452</td><td>0.362</td><td>0.379</td><td>0.347 0.374</td></tr><tr><td rowspan="8">ETTm2</td><td></td><td>0.369</td><td></td><td>0.266</td><td>0.173</td><td></td><td></td><td></td><td>0.255</td><td>0.181</td><td>0.274</td><td>0.179</td><td>0.275</td><td>0.187</td><td>0.267</td><td>0.203 0.287</td><td>0.167</td><td>0.260</td><td>0.167</td><td>0.258</td></tr><tr><td>96</td><td>0.157 0.239 0.281</td><td>0.183</td><td></td><td>0.262 0.301</td><td>0.162</td><td>0.249 0.288</td><td>0.165 0.220</td><td>0.292</td><td>0.243</td><td>0.316</td><td>0.262</td><td>0.326</td><td>0.249</td><td>0.309</td><td>0.269</td><td>0.328</td><td>0.224</td><td>0.303 0.221</td><td>0.294</td></tr><tr><td>192</td><td>0.219 0.275 0.318</td><td>0.246 0.298</td><td>0.307 0.341</td><td>0.229 0.286 0.341</td><td>0.216 0.270</td><td>0.324</td><td>0.274</td><td>0.329</td><td>0.297</td><td>0.351</td><td>0.305</td><td>0.353</td><td>0.321</td><td>0.351</td><td>0.325</td><td>0.366</td><td>0.281</td><td>0.342 0.270</td><td>0.327</td></tr><tr><td>336 720</td><td>0.359 0.375</td><td>0.373</td><td>0.390</td><td>0.378</td><td>0.401 0.350</td><td>0.375</td><td>0.362</td><td>0.385</td><td>0.375</td><td>0.401</td><td>0.389</td><td>0.407</td><td>0.497</td><td>0.403</td><td>0.421</td><td>0.415</td><td>0.397 0.421</td><td>0.356</td><td>0.382</td></tr><tr><td>avg</td><td>0.253 0.303</td><td>0.275</td><td>0.326</td><td>0.266</td><td>0.326 0.250</td><td>0.309</td><td>0.255</td><td>0.315</td><td>0.274</td><td>0.336</td><td>0.284</td><td>0.340</td><td>0.291</td><td>0.333</td><td>0.305</td><td>0.349</td><td>0.256 0.331</td><td>0.254</td><td>0.315</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>96 192</td><td>0.342 0.380 0.403</td><td>0.358</td><td>0.398</td><td>0.376 0.416</td><td>0.397 0.361</td><td>0.392</td><td>0.370 0.413</td><td>0.399 0.421</td><td>0.386 0.422</td><td>0.405 0.439</td><td>0.398 0.430</td><td>0.427</td><td>0.384</td><td>0.402</td><td>0.376</td><td>0.419</td><td>0.375 0.399</td><td>0.369 0.401</td><td>0.397 0.416</td></tr><tr><td>ETTh1 336</td><td>0.377 0.395</td><td>0.387 0.401</td><td>0.418 0.429</td><td>0.442</td><td>0.418 0.393 0.433 0.412</td><td></td><td>0.412 0.424 0.422</td><td>0.436</td><td>0.444</td><td>0.457</td><td>0.440</td><td>0.453 0.460</td><td>0.436 0.491</td><td>0.429 0.469</td><td>0.420 0.459</td><td>0.448 0.465</td><td>0.405 0.439</td><td>0.416 0.443 0.418</td><td>0.427</td></tr><tr><td rowspan="8"></td><td></td><td>0.415 0.397 0.429</td><td>0.425</td><td>0.450</td><td>0.477</td><td>0.456 0.442</td><td>0.458</td><td>0.447</td><td>0.466</td><td>0.500</td><td>0.498</td><td>0.491</td><td>0.509</td><td>0.521</td><td>0.500</td><td>0.506</td><td>0.507 0.472</td><td>0.490</td><td>0.439</td><td>0.454</td></tr><tr><td>720 avg</td><td>0.378 0.407</td><td>0.393</td><td>0.424</td><td>0.427 0.426</td><td>0.402</td><td>0.422</td><td>0.413</td><td>0.431</td><td>0.438</td><td>0.450</td><td>0.440</td><td>0.462</td><td>0.458</td><td>0.450</td><td>0.440</td><td>0.460</td><td>0.423 0.437</td><td>0.407</td><td>0.424</td></tr><tr><td>96</td><td>0.266 0.325</td><td>0.288</td><td>0.347</td><td>0.285 0.342</td><td>0.278</td><td>0.338</td><td>0.274</td><td>0.336</td><td>0.300</td><td>0.350</td><td>0.299</td><td>0.364</td><td>0.340</td><td>0.374</td><td>0.358</td><td>0.397</td><td>0.289</td><td>0.353 0.274</td><td>0.337</td></tr><tr><td>192</td><td>0.331 0.369</td><td>0.351</td><td>0.390</td><td>0.354 0.389</td><td>0.341</td><td>0.380</td><td>0.339</td><td>0.379</td><td>0.381</td><td>0.400</td><td>0.422</td><td>0.441</td><td>0.402</td><td>0.414</td><td>0.429</td><td>0.439</td><td>0.383</td><td>0.418 0.351</td><td>0.383</td></tr><tr><td>336</td><td>0.361 0.395</td><td>0.376</td><td>0.413</td><td>0.373 0.407</td><td>0.369</td><td>0.406</td><td>0.329</td><td>0.380</td><td>0.424</td><td>0.433</td><td>0.447</td><td>0.474</td><td>0.452</td><td>0.452</td><td>0.496</td><td>0.487</td><td>0.448 0.465</td><td>0.375</td><td>0.411 0.402 0.438</td></tr><tr><td>720</td><td>0.384 0.422 0.336 0.378</td><td>0.410</td><td>0.447</td><td>0.406 0.354 0.394</td><td>0.441 0.404</td><td>0.438</td><td>0.379</td><td>0.422 0.379</td><td>0.431 0.367</td><td>0.446 0.407</td><td>0.442</td><td>0.467</td><td>0.462</td><td>0.468</td><td>0.463</td><td>0.474 0.605</td></table>

## Appendix B. Sensitivity Analysis of FAN Layer Depth

We investigated the impact of FAN layer depth (L) within the Fourier Experts by comparing L = 1, 2, 3 on the ETTh1 and ETTm1 datasets. The results in Table B.13 show that the default single-layer design $( L = 1 )$ consistently achieves the best performance. Increasing the depth to 2 or 3 layers yields no accuracy gains and leads to overfitting. Therefore, we adopt L = 1 for eficiency and robustness.

Table B.13: Ablation study on the number of FAN layers (L) in the Fourier Experts. We compare L = 1, 2, 3 on ETTh1 and ETTm1 datasets. 1-layer represents the default setting in FM-LLM. The best results are highlighted in bold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">FAN Layers</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td rowspan="3">ETTh1</td><td>1-layer (Ours)</td><td>0.342</td><td>0.380</td><td>0.377</td><td>0.403</td><td>0.395</td><td>0.415</td><td>0.397</td><td>0.429</td></tr><tr><td>2-layer</td><td>0.343</td><td>0.382</td><td>0.378</td><td>0.405</td><td>0.397</td><td>0.419</td><td>0.411</td><td>0.441</td></tr><tr><td>3-layer</td><td>0.378</td><td>0.406</td><td>0.421</td><td>0.433</td><td>0.448</td><td>0.451</td><td>0.471</td><td>0.480</td></tr><tr><td rowspan="3">ETTm1</td><td>1-layer (Ours)</td><td>0.276</td><td>0.327</td><td>0.319</td><td>0.356</td><td>0.351</td><td>0.379</td><td>0.406</td><td>0.415</td></tr><tr><td>2-layer</td><td>0.289</td><td>0.339</td><td>0.334</td><td>0.369</td><td>0.368</td><td>0.390</td><td>0.428</td><td>0.423</td></tr><tr><td>3-layer</td><td>0.280</td><td>0.333</td><td>0.325</td><td>0.361</td><td>0.361</td><td>0.384</td><td>0.432</td><td>0.423</td></tr></table>

## Appendix C. Comparison with Statistical Baseline

We employed the pmdarima library for implementation. For each variable in the multivariate time series, a univariate ARIMA [14] model was fitted independently. We used the auto\_arima algorithm to search for optimal hyperparameters with the following constraints: maximum order of p and q set to 3 to prevent overfitting, and a seasonal period m = 24 to capture daily cycles. Consistent with the few-shot setting of FM-LLM, the ARIMA models were trained strictly on the initial 10% of the data. During inference, we utilized a rolling forecast strategy with a fixed model (no re-training) to evaluate generalization capability.

The comparative results on ETTh1, ETTh2, ETTm1, and ETTm2 under the 10% few-shot setting are summarized in Table C.14. It is observed that ARIMA struggles significantly in long-horizon forecasting (e.g., H = 336, 720), exhibiting high MSE values. This is primarily due to the error accumulation inherent in recursive steps for linear models and their inability to model complex multivariate correlations. In contrast, FM-LLM maintains robust performance, demonstrating its superiority in capturing long-term dependencies and utilizing pre-trained knowledge for generalization.

Table C.14: Few-shot performance comparison between ARIMA and FM-LLM on ETT datasets (10% training data).The best results are highlighted in bold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2">96</td><td colspan="2">192</td><td colspan="2">336</td><td colspan="2">720</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>ARIMA FM-LLM</td><td>1.496 0.451</td><td>0.700 0.454</td><td>1.462 0.481</td><td>0.745 0.472</td><td>2.361 0.497</td><td>1.000 0.486</td><td>4.773 0.525</td><td>1.357 0.514</td></tr><tr><td>ETTh2</td><td>ARIMA FM-LLM</td><td>1.512 0.298</td><td>0.780 0.362</td><td>1.847 0.340</td><td>0.861 0.391</td><td>2.172 0.354</td><td>0.922 0.404</td><td>2.541 0.398</td><td>1.020 0.438</td></tr><tr><td>ETTm1</td><td>ARIMA FM-LLM</td><td>2.266 0.383</td><td>0.974 0.403</td><td>2.603 0.422</td><td>1.051 0.427</td><td>2.874 0.463</td><td>1.117 0.450</td><td>3.051 0.561</td><td>1.172 0.498</td></tr><tr><td>ETTm2</td><td>ARIMA FM-LLM</td><td>1.385 0.172</td><td>0.784 0.261</td><td>1.559 0.224</td><td>0.835 0.295</td><td>1.769 0.273</td><td>0.892 0.327</td><td>2.136 0.366</td><td>0.976 0.382</td></tr></table>

## Appendix D. Diagnosing the Non-monotonic Efect of Training Data Size

To further understand the counter-intuitive non-monotonic data-scale trend, we provide an auxiliary distribution-shift diagnosis on ETTh2 and ETTm2 under the standard 6:2:2 split. As shown in Fig. D.12, we first visualize the internal shift within the training split by comparing the first 60% of training samples against the full training split on the MUFL channel (z-score), where tail outliers mainly appear in the last 40% (Fig. D.12a). We then visualize the external mismatch between the full training split and the test split on MUFL (Fig. D.12b). Finally, we assess the global impact of these atypical tail segments by removing the time indices flagged as MUFL outliers and recomputing the average Wasserstein distance between training and test across all variables (Fig. D.12c). Overall, this diagnostic analysis suggests that atypical tail segments can afect the measured train–test distribution mismatch, which may contribute to the observed non-monotonic behavior when increasing the training data size.

Verification on ETTh2 Standard Split (6:2:2)  
![](images/b5764c76cfcb6f73862563b59ca98879c07ee291debf34f2defc0a3e53d80715.jpg)

![](images/75677a0721b44af2ea49c855a71b98372496586ca1d3e3f6f6ec2b2641eefd65.jpg)

![](images/41f76fbff36ddd427257e365c0e853180753e7160ebeadd35a333ce75b9cbb8d.jpg)  
(a) ETTh2 (6:2:2 split): internal train shift (first 60% vs 100%), train–test mismatch, and the average Wasserstein distance across variables before/after removing MUFL-outlier time indices.

![](images/59341b5bf4e472c38fb60bd947bbfdc95fe861e0fea90dd64a1a380b59cf5ac3.jpg)

![](images/1587b94096e3f9e7b5796297a2175f1d2cb0247c07e370a6ebe3f54e6b48bfcf.jpg)  
(b) ETTm2 (6:2:2 split): same diagnosis as (a).

![](images/b0c06b892bf45cf062d68b5eb1c8cd98db5ed81b4a7d89ea356a87e424a7fa3a.jpg)

Figure D.12: Distribution-shift diagnosis for the non-monotonic data-scale trend on ETTh2/ETTm2. Panels (A)–(B) visualize internal and external distribution mismatch on MUFL (z-score), and panel (C) reports the global impact via the average Wasserstein distance across variables before/after removing MUFL-outlier time indices.

## References

[1] C. Gonçalves, R. J. Bessa, T. Teixeira, J. Vinagre, Budget-constrained collaborative renewable energy forecasting market, IEEE Trans. Sustain. Energy 16 (2) (2025) 1440–1452. doi:10.1109/TSTE.2025.3532835.

[2] L. Zhao, Y. Song, C. Zhang, Y. Liu, P. Wang, T. Lin, M. Deng, H. Li, T-gcn: A temporal graph convolutional network for trafic prediction, IEEE Trans. Intell. Transp. Syst. 21 (9) (2019) 3848–3858. doi:10.1109/TITS.2019.2935152.

[3] C. Fiandrino, E. P. Gómez, P. F. Pérez, H. Mohammadalizadeh, M. Fiore, J. Widmer, Aichronolens: advancing explainability for time series ai forecasting in mobile networks, in: IEEE INFOCOM 2024-IEEE Conference on Computer Communications, IEEE, 2024, pp. 1521–1530. doi:10.1109/INFOCOM52122.2024.10621134.

[4] D. Kochkov, J. Yuval, I. Langmore, P. Norgaard, J. Smith, G. Mooers, M. Klöwer, J. Lottes, S. Rasp, P. Düben, S. Hatfield, P. Battaglia, A. Sanchez-Gonzalez, M. Willson, M. P. Brenner, S. Hoyer, Neural general circulation models for weather and climate, Nature 632 (8027) (2024) 1060–1066. doi:10.1038/s41586-024-07744-y.

[5] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, D. Amodei, Language models are few-shot learners, Adv. Neural Inf. Process. Syst. 33 (2020) 1877–1901.

[6] Y. Liu, G. Qin, X. Huang, J. Wang, M. Long, Autotimes: Autoregressive time series forecasters via large language models, Adv. Neural Inf. Process. Syst. 37 (2024) 122154–122184.

[7] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, E. Chen, A survey on multimodal large language models, Natl. Sci. Rev 11 (12) (2024) nwae403. doi:https://doi.org/10.1093/nsr/nwae403.

[8] Y. Liu, Z. Kuang, H. Zhang, C. Li, F. Li, X. Ding, Prada: Prompt-guided representation alignment and dynamic adaption for time series forecasting, Knowl.-based Syst. 318 (2025) 113478. doi:https://doi.org/10.1016/j.knosys.2025.113478.

[9] T. Zhou, P. Niu, L. Sun, R. Jin, et al., One fits all: Power general time series analysis by pretrained lm, Adv. Neural Inf. Process. Syst. 36 (2023) 43322–43355.

[10] M. Jin, S. Wang, L. Ma, Z. Chu, J. Y. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan, Q. Wen, Time-LLM: Time series forecasting by reprogramming large language models, in: International Conference on Learning Representations, 2024.

[11] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, R. Jin, Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting, in: Proceedings of the 39th International Conference on Machine Learning, PMLR, 2022, pp. 27268–27286.

[12] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, M. Long, Timesnet: Temporal 2d-variation modeling for general time series analysis, in: International Conference on Learning Representations, 2023.

[13] Y. Dong, G. Li, Y. Tao, X. Jiang, K. Zhang, J. Li, J. Deng, J. Su, J. Zhang, J. Xu, Fan: Fourier analysis networks, arXiv preprint arXiv:2410.02675 (2024).

[14] G. E. Box, D. A. Pierce, Distribution of residual autocorrelations in autoregressive-integrated moving average time series models, J. Am. Stat. Assoc. 65 (332) (1970) 1509–1526. doi:10.1080/01621459.1970.10481180.

[15] E. S. Gardner Jr, Exponential smoothing: The state of the art, J. Forecasting 4 (1) (1985) 1–28.

[16] D. B. Percival, A. T. Walden, Spectral analysis for physical applications, cambridge university press, 1993. doi:10.1017/CBO9780511622762.

[17] H. Wang, J. Peng, F. Huang, J. Wang, J. Chen, Y. Xiao, Micn: Multiscale local and global context modeling for long-term series forecasting, in: International Conference on Learning Representations, 2023.

[18] W. Cai, Y. Liang, X. Liu, J. Feng, Y. Wu, Msgnet: Learning multi-scale inter-series correlations for multivariate time series forecasting, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, 2024, pp. 11141–11149.

[19] C. Li, H. Zhang, S. Abbas, C. Ma, Y. Liu, X. Tu, Panda: Patchaware graph network with dual alignment for time series forecasting, in: ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5. doi:10.1109/ICASSP49660.2025.10889936.

[20] A. Zeng, M. Chen, L. Zhang, Q. Xu, Are transformers efective for time series forecasting?, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37, 2023, pp. 11121–11128.

[21] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Y. Zhang, J. ZHOU, Timemixer: Decomposable multiscale mixing for time series forecasting, in: International Conference on Learning Representations, 2024.

[22] Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam, A time series is worth 64 words: Long-term forecasting with transformers, in: International Conference on Learning Representations, 2023.

[23] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, M. Long, itransformer: Inverted transformers are efective for time series forecasting, in: International Conference on Learning Representations, 2024.

[24] F. Li, H. Xu, H. Xu, Y. Liu, X. Ding, Cvc: Further aligning llms via cross-view correction for time series forecasting, Knowledge-Based Systems 326 (2025) 113957. doi:https://doi.org/10.1016/j.knosys.2025.113957. URL https://www.sciencedirect.com/science/article/pii/S0950705125010020

[25] H. Fan, Y. Chai, C. Liu, et al., Ev-stllm: Electric vehicle charging forecasting based on spatio-temporal large language models with multi-frequency and multi-scale information fusion, Expert Systems with Applications 313 (2026) 131620.

[26] S. Huang, Z. Zhao, C. Li, L. BAI, TimeKAN: KAN-based frequency decomposition learning architecture for long-term time series forecasting, in: International Conference on Learning Representations, 2025.

[27] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, M. Tegmark, Kan: Kolmogorov-arnold networks, in: International Conference on Learning Representations, 2025.

[28] Gallant, White, There exists a neural network that does not make avoidable mistakes, in: IEEE 1988 International Conference on Neural Networks, IEEE, 1988, pp. 657–664. doi:10.1109/ICNN.1988.23903.

[29] W. Zuo, L. Cai, Tracking control of nonlinear systems using fourier neural network, in: Proceedings of the 2005 IEEE/ASME International Conference on Advanced Intelligent Mechatronics, IEEE, 2005, pp. 670–675. doi:10.1109/AIM.2005.1511059.

[30] W. Zuo, L. Cai, Adaptive-fourier-neural-network-based control for a class of uncertain nonlinear systems, IEEE Trans. Neural Netw. 19 (10) (2008) 1689–1701. doi:10.1109/TNN.2008.2001003.

[31] L. Ziyin, T. Hartwig, M. Ueda, Neural networks fail to learn periodic functions and how to fix it, Adv. Neural Inf. Process. Syst. 33 (2020) 1583–1594.

[32] M. Uteuliyeva, A. Zhumekenov, R. Takhanov, Z. Assylbekov, A. J. Castro, O. Kabdolov, Fourier neural networks: A comparative study, Intell. Data Anal. 24 (5) (2020) 1107–1120. doi:10.3233/IDA-195050.

[33] V. Sitzmann, J. Martel, A. Bergman, D. Lindell, G. Wetzstein, Implicit neural representations with periodic activation functions, Adv. Neural Inf. Process. Syst. 33 (2020) 7462–7473.

[34] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al., Mixtral of experts, arXiv preprint arXiv:2401.04088 (2024).

[35] W. Fedus, B. Zoph, N. Shazeer, Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity, J. Mach. Learn. Res. 23 (120) (2022) 1–39.

[36] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al., Deepseek-v3 technical report, arXiv preprint arXiv:2412.19437 (2024).

[37] W. Xue, T. Zhou, Q. Wen, J. Gao, B. Ding, R. Jin, Card: Channel aligned robust blend transformer for time series forecasting, in: International Conference on Learning Representations, 2024.

[38] I. Hounie, J. Porras-Valenzuela, A. Ribeiro, Loss shaping constraints for long-term time series forecasting, in: Proceedings of the 41st International Conference on Machine Learning, PMLR, 2024, pp. 19062–19084.

[39] H. Wang, L. Pan, Z. Chen, D. Yang, S. Zhang, Y. Yang, X. Liu, H. Li, D. Tao, Fredf: Learning to forecast in the frequency domain, in: International Conference on Learning Representation, 2025.

[40] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, et al., Glam: Eficient scaling of language models with mixture-of-experts, in: Proceedings of the 39th International Conference on Machine Learning, PMLR, 2022, pp. 5547–5569.

[41] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep bidirectional transformers for language understanding, in: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186. doi:10.18653/v1/N19-1423.

[42] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, Language models are unsupervised multitask learners, OpenAI Blog 1 (8) (2019).

[43] G. Goerg, Forecastable component analysis, in: Proceedings of the 30th International Conference on Machine Learning, Vol. 28 of Proceedings of Machine Learning Research, PMLR, 2013, pp. 64–72.

[44] S. Lin, H. Chen, H. Wu, C. Qiu, W. Lin, Temporal query network for eficient multivariate time series forecasting, in: Proceedings of the 42nd International Conference on Machine Learning, 2025.

[45] Y. Liu, H. Wu, Wang, M. Long, Non-stationary transformers: Exploring the stationarity in time series forecasting, in: Adv. Neural Inf. Process. Syst., Vol. 35, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-

[46] N. Kitaev, Ł. Kaiser, A. Levskaya, Reformer: The eficient transformer, in: International Conference on Learning Representations, 2020. doi:10.48550/arXiv.2001.04451. URL https://openreview.net/forum?id=rkgNKkHtvB

[47] Y. Hu, P. Liu, P. Zhu, D. Cheng, T. Dai, Adaptive multi-scale decomposition framework for time series forecasting, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 17359–17367.

[48] S. Lin, W. Lin, X. Hu, W. Wu, R. Mo, H. Zhong, Cyclenet: enhancing time series forecasting through modeling periodic patterns, Adv. Neural Inf. Process. Syst. 37 (2024) 106315–106345.

[49] B. N. Oreshkin, D. Carpov, N. Chapados, Y. Bengio, N-beats: Neural basis expansion analysis for interpretable time series forecasting, in: International Conference on Learning Representations, 2020. doi:10.48550/arXiv.1905.10437. URL https://openreview.net/forum?id=r1ecqn4YwB

[50] C. Challu, K. G. Olivares, B. N. Oreshkin, F. Garza, M. Mergenthaler Canseco, A. Dubrawski, Nhits: Neural hierarchical interpolation for time series forecasting, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37, 2023, pp. 6989–6997. doi:10.1609/aaai.v37i6.25854.

[51] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., Pytorch: an imperative style, high-performance deep learning library, Adv. Neural Inf. Process. Syst. 32 (2019) 8026–8037.
# Never Stop Speaking: a Denial-of-Service Attack on End-to-End Speech Language Models

Shuozhe Cheng<sup>1∗</sup>, Kunlan Xiang, Mingxuan Li, Ji Zhang, Dongxiao Liu, Wenbo Jiang<sup>†</sup>

## Abstract

Many studies have shown that specially crafted inputs can induce large language models (LLMs) to generate excessively long outputs, resulting in significant computational overhead and resource consumption. While most existing denial-ofservice (DoS) attacks target text-only LLMs, end-to-end (E2E) speech LLMs are rapidly emerging. Existing text-based DoS attacks primarily rely on prompt engineering, such as adversarial sufixes or semantic inducement, which exploit the discrete nature of text inputs and therefore cannot be directly transferred to continuous speech inputs. Moreover, prior studies on speech model security mainly focus on ASR or TTS systems, leaving the DoS vulnerability of E2E speech LLMs largely unexplored. To address this gap, we propose the perturbation-based DoS attack targeting E2E speech models. Instead of inducing long outputs through prompt manipulation, our method optimizes imperceptible acoustic perturbations to directly influence the model’s autoregressive generation process while preserving the original input length. Specifically, we formulate the attack as a composite optimization objective that jointly suppresses EOS generation, encourages prolonged decoding, and largely preserves semantic consistency by integrating weighted EOS loss, top-k logit loss, length loss, and semantic alignment loss. To further improve stealthiness, we employ voice activity detection (VAD) to inject perturbations only into voiced regions. Extensive experiments on three open-source E2E speech LLMs demonstrate that our method achieves stable attack success rate while significantly increasing generation length and GPU resource consumption, revealing security risks in modern ALLMs.

## Introduction

End-to-end Audio Large Language Models (E2E ALLMs) have recently emerged as a promising paradigm for humanmachine interaction, enabling direct understanding and generation of speech without relying on explicit intermediate representations. With their rapid deployment in voice assistants and conversational systems, ensuring the security and reliability of these models has become increasingly important. Among various security threats, denial-of-service (DoS) attacks pose a critical risk by forcing models to generate excessively long responses, leading to unnecessary computational consumption and service degradation. However, while DoS attacks have been extensively studied in text-based LLMs, the vulnerability of E2E ALLMs remains largely unexplored.

![](images/55e5388a623d7e30e2184d56379917fa94d5b27ac9894119580b0d0598110ff1.jpg)  
Figure 1: The figure shows the outputs of the model with and without added noise for the same input.A clean input x(top) yields a normal speech response, while the adversarial example x+δ(bottom), crafted with a tiny perturbation, induces the model to produce excessive meaningless silent segments, drastically prolonging autoregressive decoding and incurring heavy computational overhead.

Existing DoS attacks against large language models (Liu et al. 2026; Chen et al. 2026; Si et al. 2025) have been extensively studied. For example, adversarial sufix-based methods manipulate model behavior by appending carefully designed tokens, while recent approaches construct complex queries or decompose instructions into fine-grained reasoning steps to induce unnecessarily long generation. However, these strategies cannot be directly applied to end-to-end audio large language models (E2E ALLMs). Unlike text models that operate on editable discrete tokens, E2E ALLMs process continuous acoustic waveforms, making token-level manipulation infeasible. Moreover, prompt-based strategies typically rely on abnormally complex or unnatural inputs, which increase attack detectability and are dificult to transfer to speech inputs.Existing studies on ASR models (Haque et al. 2023) have explored DoS attacks by suppressing the logits of the EOS token. However, these methods mainly target the speech-to-text component and cannot address the complex generation process of E2E ALLMs. Furthermore, optimizing only EOS logits may lack suficient constraints on other aspects of generation, potentially degrading semantic consistency or causing unstable decoding behavior. Therefore, crafting efective DoS attacks against E2E ALLMs remains challenging for three reasons. First, perturbations must be optimized in the continuous waveform space while remaining imperceptible and preserving the original input length. Second, directly optimizing discrete generation length is nondiferentiable. Third, excessive suppression of termination signals may disrupt semantic consistency or generation stability, resulting in premature decoding failures rather than prolonged outputs.

To address these issues, we propose an improved method based on multi-loss EOS logit suppression. Our loss function consists of the following four components: (1) Weighted EOS logit loss: On top of the basic summation of all EOS logits, we assign dynamic weights to the logits at each autoregressive step, incorporating token-wise penalties and generation process penalties. (2) top-k logit loss: We indirectly suppress the EOS output by increasing the probability of the top-k tokens output by the model. (3) Length loss: We force the model’s output length to align with max new tokens. (4) Semantic similarity loss: We maintain the semantic consistency between the original audio and the adversarial audio to ensure attack stealthiness. We show an example in Figure 1.

To enhance attack stealthiness, we further employ voice activity detection (VAD) to extract voiced segments of the audio and add noise only to these voiced portions. Experimental results demonstrate that our method achieves strong performance on ALLM (Audio Large Language Model). Our contributions are as follows:

• We present a novel end-to-end Denial-of-Service attacks on ALLMs, filling a critical gap in end-to-end ALLM security.

• We propose a multi-objective optimization loss function to instantiate these attacks, providing a principled method for crafting adversarial inputs that disrupt ALLM behavior.

• Extensive experiments demonstrate the efectiveness of our attacks, opening new research directions and highlighting practical vulnerabilities in ALLM systems.

## Related Work

## Denial-of-service attack

The goal of Denial of Service (DoS) attacks is typically to break the output constraint of a model by constructing specially crafted inputs—such as adding adversarial sufixes, introducing spelling errors, or using homophones—thereby forcing the model to generate output indefinitely. As we know, each output token consumes substantial computational resources. Once a model begins generating unbounded responses, it continuously consumes server resources on the one hand; on the other hand, server resources are finite—once depleted, the server can no longer serve legitimate users. Previous research has demonstrated the efectiveness of DoS attacks. For example, Zhang et al. proposed the Auto-DoS(Zhang et al. 2024) algorithm, which constructs a DoS attack tree and expands node coverage to achieve efectiveness under black-box settings. The P-DoS(Gao et al. 2024) attack injects poisoned samples during model fine-tuning to reduce the logits of the EOS token, thereby breaking the model’s output length limit. However, due to the discrete nature of text-based methods, they cannot be directly transferred to the audio domain. Existing attacks on audio models use overly simplistic loss functions and have not achieved satisfactory performance.

## E2E ALLM

The development of the Transformer has brought transformative changes to the field of artificial intelligence. With its unique attention mechanism and contextual processing capability, it has rapidly driven the evolution of LLM architectures, demonstrating superiority across various domains. Leveraging the powerful capabilities of LLMs, many studies have explored multimodal (Latif et al. 2023) fusion architectures for text–video, text–image, and text–speech. Benefiting from this, speech models have gradually evolved from discrete cascade architectures (where speech input must pass through a text-based intermediate medium to produce output) to end-to-end (E2E) architectures centered around the Transformer. E2E ALLMs (Audio Large Language Models) have no explicit speech-to-text process internally (Dao, Vu, and Ha 2024). Instead, they achieve bidirectional understanding of text and speech through text–speech alignment strategies, often employing interleaved text and speech generation strategies, demonstrating powerful speech processing capabilities.

## VAD

Voice Activity Detection aims to accurately distinguish voiced segments from silence or background noise in an audio signal, and is one of the fundamental techniques in speech processing. Traditional VAD methods are mostly based on energy thresholds, zero-crossing rates, or statistical models (Atal and Rabiner 1976; web 2024), but their performance is limited under low signal-to-noise ratio conditions. With the development of deep learning, neural network based VAD methods (Wilkinson and Niesler 2021; Zhang, Wang, and Glass 2016) have significantly improved detection robustness. In the field of adversarial attacks, VAD has been used to constrain the region where perturbations are added, for instance, adding noise only to voice activity regions to improve stealthiness while maintaining attack efectiveness. Our work adopts this idea by using VAD to extract voiced segments from the audio and adding optimized perturbations only on those segments, thereby avoiding noticeable artifacts in silent regions.

## Method

In this work, we build on previous research and propose an improved method that constructs a composite loss function to achieve more precise and efective attacks. To enhance stealthiness, we employ VAD to segment the audio and add noise only to the voiced portions(Ko, Kim, and Kwon 2026). Our optimization method uses gradient descent.

## Threat Model

We consider a denial-of-service (DoS) threat against end-toend ALLMs. The attacker’s goal is to degrade service availability by inducing the model to generate excessively long responses containing meaningless silent segments, thereby exhausting computational resources and increasing inference costs.

![](images/8521e56e6555afefe7a78f8528cbea0b18de84ed7fc2da045f6b75a78a70dcb4.jpg)  
Figure 2: This figure illustrates the workflow of the adversarial DoS attack targeting end-to-end speech large models. VAD is used to separate voiced and silent segments and generate a mask for adding imperceptible perturbations. Adversarial examples are synthesized by applying tiny perturbations to the original speech. In the white-box optimization loop, we construct the multi-loss function $\mathcal { L } _ { m u l t i }$ consisting of $\dot { \mathcal { L } } _ { e o s } ,$ $\mathcal { L } _ { T o p k } , \mathcal { L } _ { l e n }$ , and $\mathcal { L } _ { s e m } .$ , and optimize perturbations via PGD.

We focus on the white-box attack setting, where the attacker has full access to the target model architecture and parameters and can compute gradients with respect to the input speech. However, the attacker can only introduce small, imperceptible adversarial perturbations to the input audio without modifying the model or its deployment environment. The perturbation magnitude is constrained below the human auditory perception threshold to ensure stealthiness.

## Attack Overview

Prior work has shown that large language models (LLMs)(OpenAI 2023) are susceptible to denial-of-service (DoS) attacks. Conventional DoS methods targeting textbased models typically append adversarial sufixes to the input text—a technique that introduces discrete tokens and is thus inapplicable to the continuous audio modality. Alternative approaches craft complex semantic constructs or sub-queries to elicit longer outputs; however, naively converting such crafted text into speech and feeding it to the model lacks novelty, and our subsequent experiments confirm that this strategy is largely inefective.

Therefore, we propose our method. We consider a whitebox attack scenario, where the attacker has full access to the architecture and parameters of the target end-to-end audio large language model (E2E ALLM) and can compute the gradients of the model’s output. Given an initial audio $\mathbf { X } ,$ a white-box model F to be attacked, and a composite loss function L, we first apply VAD to segment x into voiced and silent segments. We extract all voiced segments $V = \{ x _ { 1 } , x _ { 2 } , \ldots , { \bar { x } } _ { n } \}$ ,and silent segments $S = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { n } \}$ . The goal of our attack is to generate noise $N = \{ \sigma _ { 1 } , \sigma _ { 2 } , . . . , \sigma _ { n } \}$ for each voiced segment such that adding these noises to the original speech forces the model F to output as long as possible. It can be formulated as a constrained optimization problem:

$$
\operatorname* { m i n } _ { \{ \sigma _ { 1 } , \sigma _ { 2 } , . . . , \sigma _ { n } \} } \quad { \mathcal { L } } \left( F ( { \hat { x } } ) \right) ,\tag{1}
$$

$$
s . t . \quad \| \sigma _ { i } \| _ { p } \leq \epsilon _ { i } , \quad \forall i \in \{ 1 , 2 , \ldots , n \} ,\tag{2}
$$

$$
\hat { x } = C o n c a t \left( \left\{ x _ { i } + \sigma _ { i } \right\} _ { i = 1 } ^ { n } , \ : S \right)\tag{3}
$$

where $\mathcal { L }$ is our proposed composite loss function detailed in the next subsection. By minimizing ${ \mathcal { L } } ,$ the optimized perturbation efectively suppresses the End-of-Sentence (EOS) token probability while preserving semantic coherence, leading to prolonged autoregressive generation. Figure 2 illustrates the overall workflow of our attack.

## Noise Optimization Based on Composite Loss Function

Optimization Pipeline Given the initial audio x and the target model F, we iteratively optimize the perturbation via gradient descent. At the beginning of each iteration, we concatenate the perturbed voiced segments (initialized with random Gaussian noise) with the unaltered silent segments to reconstruct the full adversarial waveform xˆ. We then feed xˆ into the speech encoder of the target model to extract acoustic features, followed by the autoregressive decoder. During the decoding process, we record three types of outputs at each step: the logit of the EOS token, the logits of the top-K tokens, and the final total generation length N. After the generation finishes (either by emitting EOS or reaching the maximum limit), we compute the composite loss $\mathcal { L } _ { D o S }$ using the recorded values. The gradient of $\mathcal { L } _ { D o S }$ with respect to the perturbation is calculated via backpropagation, and the perturbation is updated using Projected Gradient Descent (PGD)(Madry et al. 2018). This optimization loop repeats until the generated response reaches the preset maximum token limit $N _ { \mathrm { m a x } }$ or the maximum iteration count is exhausted.

The overall composite loss is defined as:

$$
\mathcal { L } _ { D o S } = \mathcal { L } _ { e o s } + \mathcal { L } _ { t o p k } + \mathcal { L } _ { l e n } + \mathcal { L } _ { s e m }\tag{4}
$$

In this loss function, the terms can be mainly divided into direct losses and indirect losses. The direct losses focus on controlling the generation length of the model by suppressing the EOS logits and stretching the length of the speech tokens output by the model. The indirect losses achieve indirect suppression of the EOS logits and smooth the model behavior by increasing the probability of top-k token outputs and preserving semantic invariance of the noisy speech, thereby raising the probabilities of other tokens and preventing premature termination caused by semantic issues.

Direct losses. The direct losses consist of two parts. The first is the EOS logits loss. Unlike the traditional approach of simply summing all EOS logits, we assign a weight to each EOS logit as follows:

$$
\mathcal { L } _ { e o s } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } w _ { t } \ell _ { E O S , t }\tag{5}
$$

The weight consists of a sign-based gating component and a dynamically growing component. The gating component depends on the sign of the logit value. Since a non-positive EOS logit implies that the EOS token is almost impossible to be output, we set its weight to zero to completely exclude such uninformative steps from the optimization. Only steps with positive EOS logits are assigned a positive weight, directing the optimizer toward positions where EOS suppression is actually meaningful. The dynamic component depends on the depth of the autoregressive step. It is known that the probability of outputting the EOS token increases as generation proceeds. Therefore, this component gradually increases with the autoregressive step, and we adopt exponential growth to make the optimization pay more attention to later stages of the model output.

$$
w _ { t } = \left\{ \begin{array} { l l } { w _ { h i g h } \exp \left( \frac { t } { N } \kappa \right) , } & { \ell _ { E O S , t } > 0 } \\ { 0 , } & { \ell _ { E O S , t } \leq 0 } \end{array} \right.
$$

To encourage long generation sequences, we introduce a diferentiable length loss. Since the generated audio token length is determined by the EOS prediction during autoregressive decoding, directly optimizing the discrete output length is not feasible. Therefore, we estimate the expected generation length based on the probability distribution of EOS termination at each generation step. Let $z _ { t }$ denote the EOS logit at the t-th generation step and $p _ { t } = \sigma ( z _ { t } )$ be the corresponding EOS probability. The probability that the generation terminates at step t is computed by considering that no EOS token is generated before step t and an EOS token is generated at step t. The expected generation length is then formulated as:

$$
\mathbb { E } [ L ] = \sum _ { t = 1 } ^ { N } t \left( p _ { t } \prod _ { i = 1 } ^ { t - 1 } ( 1 - p _ { i } ) \right)\tag{6}
$$

where $N$ is the maximum generation length. The length loss is defined as:

$$
\mathcal { L } _ { L e n } = \frac { ( N _ { \mathrm { m a x } } - \mathbb { E } [ L ] ) ^ { 2 } } { N _ { \mathrm { m a x } } } .\tag{7}
$$

This formulation encourages the expected generation length to approach the maximum allowed length while maintaining a comparable scale with other loss terms.

Indirect losses. The indirect losses also consist of two parts. The first is the top-k loss. At each generation step, the model samples the output from the top-k tokens. Therefore, the top-k loss focuses on increasing the logits of the top-k tokens. The increase in top-k logits indirectly reduces the EOS logits. At the same time, increasing the top-k logits helps preserve the coherence of the synthesized audio.

$$
\mathcal { L } _ { t o p k } = - \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \ell _ { t o p k , t } ^ { ( k ) }\tag{8}
$$

Finally, we introduce a semantic alignment loss. For the original speech and the noise-added speech, we extract their features using a speech encoder and compute the cosine similarity between the two feature vectors. This ensures that the perturbation does not afect the semantics of the audio, thereby eliminating the possibility that changes in output length are caused by semantic alteration.

$$
\mathcal { L } _ { s e m } = 1 - \frac { \langle s ( x ) , s ( x + \delta ) \rangle } { \| s ( x ) \| _ { 2 } \| s ( x + \delta ) \| _ { 2 } }\tag{9}
$$

## Experiment

## Experiment Setup

Model and Datasets We evaluate our audio adversarial attack on three recent open-source audio large models: LFM2.5-Audio-1.5B(AI 2025),Fun-Audio-Chat-8B(Tan et al. 2025) and Qwen2-Audio-7B-Instruct(Chu et al. 2024). LFM2.5-Audio is a lightweight end-to-end foundation model designed for low-latency real-time speech conversation; Fun-Audio-Chat and Qwen2-Audio are medium-sized audio-language dialogue models optimized for general speech interaction and audio analysis. We evaluate on two primary models and further report Qwen2-Audio results in Appendix. For evaluation, we use two public speech datasets, OpenSLR and QCRI.

Attack Setup During the attack phase, we adopt the Projected Gradient Descent (PGD) framework to optimize adversarial perturbations, aiming to induce the target model to generate longer outputs. Specifically, we randomly sample 100 clean audio samples from each dataset, add the adversarial perturbations (generated by our method) to the original audio, and feed the perturbed audio into the target models to measure the attack success rate. Due to limited computational resources, we set $N _ { \mathrm { m a x } } = 1 0 2 4$

The hyperparameters are set as follows. For $\mathcal { L } _ { e o s } ,$ we set the $w _ { h i g h } = 4$ and the exponential growth parameter $\kappa =$ 4. For $\bar { \mathcal { L } } _ { t o p k }$ , we set $k = 3 .$ . During optimization, we run PGD for 200 iterations per sample, with the perturbation magnitude constrained by an $\ell _ { \infty }$ norm bound of $\epsilon = 1 0 ^ { - 4 }$

Metrics To comprehensively evaluate the efectiveness and resource cost of our attack method across diferent baselines and all reported results are averaged over 100 samples. We adopt the following three metrics:

Table 1: Attack performance comparison across diferent models and baselines. All experiments are performed on the OpenSLR and QCRI datasets across evaluated models.
<table><tr><td>Model</td><td>Method</td><td>ASR (↑)</td><td>AVG. Output Length (↑)</td><td>Memory Usage (GB)</td></tr><tr><td rowspan="6">Liquid Audio (1.5B)</td><td>Clean audio</td><td>0.00</td><td>198.34</td><td>8.89/47.99</td></tr><tr><td>Random Noise</td><td>0.00</td><td>205.51</td><td>8.58/47.99</td></tr><tr><td>Simple Loss</td><td>0.79</td><td>857.38</td><td>10.56/47.99</td></tr><tr><td>Črabs</td><td>0.34</td><td>545.72</td><td>9.96/47.99</td></tr><tr><td>ExtendAttack</td><td>0.29</td><td>496.37</td><td>10.18/47.99</td></tr><tr><td>Our method</td><td>0.87</td><td>941.88</td><td>10.78/47.99</td></tr><tr><td rowspan="6">FunAudioChat (8B)</td><td>Clean audio</td><td>0.00</td><td>213.57</td><td>20.26/47.99</td></tr><tr><td>Random Noise</td><td>0.00</td><td>210.83</td><td>20.29/47.99</td></tr><tr><td>Simple Loss</td><td>0.77</td><td>869.52</td><td>21.74/47.99</td></tr><tr><td>Črabs</td><td>0.36</td><td>594.38</td><td>21.50/47.99</td></tr><tr><td>ExtendAttack</td><td>0.48</td><td>607.54</td><td>21.16/47.99</td></tr><tr><td>Our method</td><td>0.84</td><td>920.24</td><td>21.93/47.99</td></tr></table>

Table 2: Ablation study on loss components $\mathcal { L } _ { s e m } , \mathcal { L } _ { l e n }$ (length loss), $\mathcal { L } _ { e o s } ,$ and $\mathcal { L } _ { t o p k }$
<table><tr><td> $\mathcal { L } _ { s e m }$ </td><td> $\mathcal { L } _ { l e n }$ </td><td> $\mathcal { L } _ { e o s }$ </td><td> $\mathcal { L } _ { t o p k }$ </td><td>VAD</td><td>ASR (↑)</td><td>Avg.Output Length (↑)</td><td>Response Quality (↑)</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>0.80</td><td>893.47</td><td>4.46</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>0.12</td><td>289.43</td><td>4.71</td></tr><tr><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>0.79</td><td>867.65</td><td>4.70</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.83</td><td>937.81</td><td>3.92</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>0.84</td><td>945.33</td><td>4.52</td></tr><tr><td>√</td><td>√</td><td>V</td><td>√</td><td>√</td><td>0.84</td><td>950.24</td><td>4.75</td></tr></table>

• Attack Success Rate (ASR): The proportion of adversarial samples that successfully force the target model to generate audio token sequences reaching the maximum generation length limit.

• Output Token Length: The number of tokens in the model’s generated response. We report the average output length over all test samples, which helps analyze how the attack influences the model’s generative behavior.

• Peak GPU Memory Usage: The maximum GPU memory allocated during a single forward inference pass (without gradient computation), measured in MB or GB. This metric indicates the practical computational overhead and feasibility of the attack, especially for edge-side audio models.

• Response Quality: We extract the generated text tokens from clean and adversarial inputs and use ChatGPT 5.5(OpenAI 2025) as an evaluator to assess semantic consistency and response quality. The final score is averaged over all samples on a 1-5 scale. The evaluation prompt is provided in Appendix .

To demonstrate the efectiveness of our proposed attack method, we compare it with the following three baselines: (1) using clean original audio or audio with random noise as input; (2) a conventional method that relies solely on naive eos logits; (3) a noise injection approach that operates on the entire global audio without employing Voice Activity Detection (VAD). (4) Crabs(Zhang et al. 2024), a kind ofDoS attack on natural LLM (5) ExtendAttack(Zhu et al. 2025),a kind of dos attack on LRM

## Main Result

Table 1 summarizes the attack performance under the default top-k sampling strategy. Our method consistently achieves the strongest DoS capability across diferent E2E ALLMs, demonstrating its efectiveness in manipulating autoregressive generation behaviors. On LFM2.5-Audio and FunAudioChat, our method achieves attack success rates of 87% and 84%, respectively, significantly outperforming all baseline methods. Meanwhile, the generated output length is extended to 941.88 and 920.24 tokens, which is more than four times longer than clean inputs and substantially exceeds random noise and existing DoS baselines. This indicates that the optimized acoustic perturbations can efectively prevent normal termination and force the model to continue decoding for an extended period, thereby introducing considerable computational overhead.

Compared with the simple EOS suppression baseline, our method achieves higher ASR and longer outputs on both models. This improvement demonstrates that directly optimizing EOS logits alone is insuficient for reliable DoS attacks, while jointly modeling termination suppression, generation length, token distribution, and semantic consistency provides a more efective optimization objective. In addition, text-based DoS attacks, including Crabs and ExtendAttack, exhibit limited efectiveness when transferred to the audio domain, achieving much lower ASR and shorter outputs. These results further verify that discrete token manipulation strategies designed for text LLMs cannot be directly applied to continuous speech inputs.

![](images/44bfb4f61385482d2aef283805811f70cee94a2541ab7b473bd5d80d95603fc5.jpg)  
(a)  
(b)  
Figure 3: Waveform comparison between (a) clean output and (b) attacked output. The clean audio exhibits continuous voiced segments with high-amplitude oscillations, whereas the attacked output shows sparsely distributed speech fragments separated by extended flat silent regions

Beyond generation length, our attack also introduces practical computational costs. As shown in Table 1, the increased output length leads to substantial GPU memory consumption during inference, with our method requiring 10.78 GB and 21.93 GB on LFM2.5-Audio and FunAudioChat, respectively, compared with only 8.89 GB and 17.26 GB for clean inputs. These results reveal that carefully optimized acoustic perturbations can efectively exploit the autoregressive generation mechanism of E2E ALLMs and cause significant resource consumption while preserving the original input semantics.

To further evaluate the robustness of our method, we additionally test diferent decoding strategies and the Qwen2- Audio model. We also test the attack transferability. The results under greedy decoding, on Qwen2-Audio and the transferability are reported in the Appendix.

## Output Audio Analysis

To further understand the mechanism behind output elongation, we analyze the generated waveforms and intermediate text responses from successfully attacked samples, as shown in Figure 3. The transcribed responses remain largely consistent with those generated from clean inputs, indicating that our attack does not significantly afect the model’s semantic understanding capability. However, waveform analysis reveals that attacked outputs contain substantially more low-energy silent regions compared with clean outputs. This phenomenon is mainly caused by the joint optimization of $L _ { E O S }$ and $L _ { l e n } .$ , which encourages the model to continue autoregressive decoding beyond its normal termination point. As generation extends beyond meaningful response ranges, the additional tokens tend to correspond to low-information speech segments, resulting in prolonged outputs and increased computational consumption.

## Ablation Study

Efectiveness of Loss Components. To investigate the contribution of each component in our composite objective, we conduct ablation experiments by removing each loss term from $L _ { D o S }$ individually while keeping all other settings unchanged. The results are summarized in Table 2.

Removing $L _ { E O S }$ leads to a significant degradation in attack performance, reducing the ASR from 0.84 to 0.12 and the average output length from 950.24 to 289.43 tokens. This demonstrates that weighted EOS suppression is the primary factor driving prolonged generation, as it directly prevents premature termination during autoregressive decoding. Without $L _ { l e n } .$ , the ASR decreases slightly to 0.79 and the output length drops to 867.65 tokens, indicating that the length loss provides additional optimization guidance toward the maximum generation boundary. Similarly, removing $L _ { t o p k }$ causes a moderate reduction in ASR and output length, while noticeably degrading response quality. This suggests that $L _ { t o p k }$ mainly improves generation stability by redistributing token probabilities and indirectly suppressing EOS prediction.

Removing $L _ { s e m }$ slightly improves ASR and output length, but results in a substantial decline in response quality (from 4.75 to 3.92). This indicates that semantic alignment is not essential for maximizing output length, but plays a critical role in preserving the original speech semantics and ensuring attack stealthiness. Overall, the ablation results verify that each component serves a complementary purpose: L<sub>EOS</sub> provides the fundamental attack capability, $L _ { l e n }$ and $\scriptstyle L _ { t o p k }$ enhance generation extension and stability, while $L _ { s e m }$ maintains semantic consistency.

Efectiveness ofthe VAD Strategy. To evaluate the contribution of the VAD-based perturbation strategy, we compare our full method with a variant that applies perturbations to the entire waveform without VAD. As shown in Table, removing VAD results in a similar attack success rate (0.84) and slightly shorter output length, indicating that restricting perturbations to voiced regions does not compromise the attack efectiveness. Meanwhile, the response quality decreases from 4.75 to 4.52 without VAD, suggesting that perturbing silent regions introduces unnecessary acoustic interference and afects semantic preservation. These results demonstrate that VAD provides a better trade-of between attack efectiveness and stealthiness by concentrating perturbations on informative speech regions.

Efect of the exponential growth parameter κ. We further investigate the impact of the exponential growth parameter κ in $L _ { e o s }$ . As shown in Table 3, the attack performance first improves and then decreases as κ increases, with $\kappa = 4$ achieves the best or comparable performance across all evaluated models. When κ is relatively small, the dynamic EOS weighting grows slowly, limiting the emphasis on later autoregressive steps where EOS prediction becomes more critical. Consequently, the optimizer may fail to suficiently suppress EOS probabilities within the fixed optimization budget. In contrast, overly large κ values cause the EOS weights to increase too rapidly, which may introduce excessively large gradients and make the optimization unstable. Moreover, an excessively dominant $L _ { e o s }$ term can weaken the contributions of other loss components, such as length guidance and semantic preservation, resulting in degraded attack efectiveness. Therefore, we select $\kappa = 4$ as it provides a balanced trade-of between efective EOS suppression and stable multi-objective optimization.

Table 3: Ablation study on the exponential growth parameter κ in $L _ { e o s }$
<table><tr><td>Model</td><td> $\kappa = 2$ </td><td> $\kappa = 3$ </td><td> $\kappa = 4$ </td><td> $\kappa = 5$ </td><td> $\kappa = 6$ </td></tr><tr><td>LFM2.5-Audio</td><td>82%</td><td>84%</td><td>87%</td><td>86%</td><td>79%</td></tr><tr><td>FunAudioChat</td><td>80%</td><td>82%</td><td>84%</td><td>82%</td><td>76%</td></tr><tr><td>Qwen2-Audio</td><td>81%</td><td>83%</td><td>83%</td><td>79%</td><td>77%</td></tr></table>

## Attack Robustness

We further evaluate the robustness of our adversarial perturbations against additional random noise. Specifically, after generating adversarial examples, we add extra noise η with diferent magnitudes relative to the original perturbation constraint $\epsilon , \mathrm { i . e . , } \| \eta \| _ { \infty } = \alpha \epsilon$ , and measure the resulting ASR.

As shown in Fig. 4(a), our attack remains efective under small additional perturbations, indicating that the optimized perturbations can tolerate moderate input variations. It is worth noting that αϵ represents the noise scale relative to the predefined perturbation bound rather than the actual magnitude of the optimized perturbation. Since the final perturbations generated by PGD may occupy diferent magnitude ranges across diferent models, the point at which additional noise starts to disrupt the attack varies among models. When the added noise reaches a comparable magnitude level to the optimized perturbation, the attack efectiveness gradually decreases, as the original adversarial direction is disturbed.

## Limitation

Our robustness analysis reveals that lossy compression can afect the efectiveness of adversarial perturbations, as shown in Fig 4(b). This phenomenon is mainly because compression algorithms(Brandenburg 1995; Valin, Vos, and Terriberry 2012) transform the original waveform representation by removing or quantizing fine-grained acoustic details, which may distort the optimized perturbation patterns. Similar effects can also be observed in other signal-level transformations, such as resampling, filtering, or noise injection, which modify the input distribution and consequently interfere with the adversarial directions.

However, these transformations introduce additional processing overhead and may degrade speech quality. Moreover, adaptive attackers may optimize perturbations considering the preprocessing pipeline, potentially reducing the robustness of such defenses. Therefore, developing practical defense mechanisms that can efectively disrupt adversarial perturbations while maintaining low latency and preserving speech quality remains an important future direction.

## Conclusion

In this paper, we present a systematic study of denial-ofservice (DoS) attacks against end-to-end audio large language models (E2E ALLMs). We propose a white-box adversarial attack framework that leverages VAD-based perturbation and a multi-component optimization objective to suppress EOS prediction and induce prolonged autoregressive generation. Extensive experiments on multiple E2E ALLMs demonstrate that the proposed method can efectively increase output lengths, achieve high attack success rates, and introduce substantial inference overhead while largely preserving the semantic information of the original speech.

![](images/44b890025b19c6cd73cae740386bb1161a0134eef99ab162322ad1e63eaf7fdf.jpg)

(a) ASR under additional noise  
![](images/c1f90356424aba2c69c4031bac06ba9f191c7e3861f91166c7a326cc8a190738.jpg)  
(b) Compression robustness  
Figure 4: (a) Attack performance under diferent additional noise. (b) Attack robustness against MP3 compression.

Through comprehensive ablation studies, we analyze the contribution of each loss component and show that weighted EOS suppression, length guidance, token distribution optimization, and semantic alignment jointly improve attack efectiveness and stealthiness. Furthermore, our robustness analysis reveals that the attack can tolerate moderate input perturbations, while signal transformations such as compression may significantly reduce its efectiveness. These findings highlight the potential security risks of current E2E ALLMs and motivate the development of practical defense mechanisms against audio-domain adversarial DoS attacks.

## References

2024. WebRTC Voice Activity Detector. https://webrtc.org/. Open-source real-time voice activity detection algorithm widely adopted in speech processing pipelines.

AI, L. 2025. LFM2 Technical Report. arXiv preprint arXiv:2511.23404.

Atal, B.; and Rabiner, L. 1976. A pattern recognition approach to voiced-unvoiced-silence classification with applications to speech recognition. IEEE Transactions on Acoustics, Speech, and Signal Processing, 24(3): 201–212.

Brandenburg, K. 1995. The MPEG audio layer-3 coding standard. Proceedings ofthe IEEE, 83(4): 1260–1263.

Chen, Y.; Li, Z.; Yue, X.; Tan, R. T.; and Li, H. 2026. NaturalSloth: Revisiting Denial-of-Service Attacks on Large Language Models. In Liakata, M.; Moreira, V. P.; Zhang, J.; and Jurgens, D., eds., Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 19685–19702. San Diego, California, United States: Association for Computational Linguistics. ISBN 979-8-89176-390-6.

Chu, Y.; Xu, J.; Yang, Q.; Wei, H.; Wei, X.; Guo, Z.; Leng, Y.; Lv, Y.; He, J.; Lin, J.; Zhou, C.; and Zhou, J. 2024. Qwen2- Audio Technical Report. arXiv preprint arXiv:2407.10759.

Dao, A.; Vu, D. B.; and Ha, H. H. 2024. Ichigo: Mixed-Modal Early-Fusion Realtime Voice Assistant. arXiv:2410.15316.

Gao, K.; Pang, T.; Du, C.; Yang, Y.; Xia, S.-T.; and Lin, M. 2024. Denial-of-Service Poisoning Attacks against Large Language Models. arXiv:2410.10760.

Haque, M.; Shah, R.; Chen, S.; Şişman, B.; Liu, C.; and Yang, W. 2023. SlothSpeech: Denial-of-service Attack Against Speech Recognition Models. arXiv:2306.00794.

Ko, K.; Kim, S.; and Kwon, H. 2026. Audio Adversarial Example With No Noise in the Silent Area for Speech Recognition System. IEEE Access, 14: 2924–2938.

Latif, S.; Shoukat, M.; Shamshad, F.; Usama, M.; Ren, Y.; Cuay’ahuitl, H.; Wang, W.; Zhang, X.; Togneri, R.; Cambria, E.; and Schuller, B. W. 2023. Sparks of Large Audio Models: A Survey and Outlook. arXiv:2308.12792.

Liu, X.; Wang, X.; Zhang, Y.; Kariyappa, S.; Xiang, C.; Chen, M.; Suh, G. E.; and Xiao, C. 2026. Reasoning-Bomb: A Stealthy Denial-of-Service Attack by Inducing Pathologically Long Reasoning in Large Reasoning Models. arXiv:2602.00154.

Madry, A.; Makelov, A.; Schmidt, L.; Tsipras, D.; and Vladu, A. 2018. Towards Deep Learning Models Resistant to Adversarial Attacks. In International Conference on Learning Representations.

OpenAI. 2023. Periodic Outages Across ChatGPT and API. Accessed on 16/03/2024.

OpenAI. 2025. GPT-5.5 Technical Report. OpenAI Technical Report.

Si, W. M.; Li, M.; Backes, M.; and Zhang, Y. 2025. Excessive Reasoning Attack on Reasoning LLMs. arXiv:2506.14374v1.

Tan, C.-H.; Chen, Q.; Wang, W.; Deng, C.; Zhang, Q.; Cheng, L.; Yu, H.; Zhang, X.; Lv, X.; Zhao, T.; Zhang, C.; Ma, Y.; Chen, Y.; Wang, H.; Liu, J.; Li, X.; and Ye, J. 2025. DrVoice: Parallel Speech-Text Voice Conversation Model via Dual-Resolution Speech Representations. arXiv:2506.09349.

Valin, J.-M.; Vos, K.; and Terriberry, T. B. 2012. Definition of the Opus Audio Codec. Technical Report RFC 6716, Internet Engineering Task Force.

Wilkinson, N.; and Niesler, T. 2021. A Hybrid CNN-BiLSTM Voice Activity Detector. In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 6803–6807.

Zhang, Y.; Wang, Y.; and Glass, J. 2016. Feature Learning with Raw-Waveform CLDNNs for Voice Activity Detection. In Interspeech, 3542–3546.

Zhang, Y.; Zhou, Z.; Zhang, W.; Wang, X.; Jia, X.; Liu, Y.; and Su, S. 2024. Crabs: Consuming Resource via Autogeneration for LLM-DoS Attack under Black-box Settings. arXiv preprint arXiv:2412.13879. ArXiv:2412.13879v4 [cs.CL].

Zhu, Z.; Liu, Y.; Xu, Z.; Ma, Y.; Gao, H.; Chen, N.; Guo, Y.; Qu, W.; Xu, H.; Kang, Z.; Zhu, X.; and Zhang, J. 2025. ExtendAttack: Attacking Servers of LRMs via Extending Reasoning. arXiv:2506.13737.

Table 4: Attack performance under greedy decoding strategies
<table><tr><td>Model</td><td>Method</td><td>ASR (↑)</td><td>AVG. Output Length (↑)</td><td>Memory Usage (GB)</td></tr><tr><td rowspan="5">Liquid Audio (1.5B)</td><td>Clean audio</td><td>0.00</td><td>162.52</td><td>8.07/47.99</td></tr><tr><td>Random Noise</td><td>0.00</td><td>161.28</td><td>8.96/47.99</td></tr><tr><td>Simple Loss</td><td>0.77</td><td>832.58</td><td>10.46/47.99</td></tr><tr><td>Crabs</td><td>0.34</td><td>538.24</td><td>9.92/47.99</td></tr><tr><td>Our method</td><td>0.86</td><td>927.69</td><td>10.27/47.99</td></tr><tr><td rowspan="5">FunAudioChat (8B)</td><td>Clean audio</td><td>0.00</td><td>187.22</td><td>20.53/47.99</td></tr><tr><td>Random Noise</td><td>0.00</td><td>195.79</td><td>20.76/47.99</td></tr><tr><td>Simple Loss</td><td>0.76</td><td>852.34</td><td>21.19/47.99</td></tr><tr><td>Crabs</td><td>0.38</td><td>613.72</td><td>21.07/47.99</td></tr><tr><td>Our method</td><td>0.84</td><td>905.36</td><td>21.44/47.99</td></tr><tr><td rowspan="5">QwenAudio(7B)</td><td>Clean audio</td><td>0.00</td><td>182.35</td><td>18.37/47.99</td></tr><tr><td>Random Noise</td><td>0.00</td><td>181.43</td><td>18.12/47.99</td></tr><tr><td>Simple Loss</td><td>0.74</td><td>807.75</td><td>19.09/47.99</td></tr><tr><td>Crabs</td><td>0.43</td><td>643.35</td><td>20.07/47.99</td></tr><tr><td>Our method</td><td>0.85</td><td>934.02</td><td>20.14/47.99</td></tr></table>

## Response Quality Evaluation Prompt

To evaluate response quality, we use an LLM-as-a-Judge strategy. The generated text responses from clean and adversarial inputs are provided to ChatGPT for evaluation. Each response is assigned a score from 1 to 5, and the final Response Quality score is calculated by averaging the scores over all test samples.

![](images/a6fc1f1900ff1036fb6d4449972bfbd0a35fccdbed8d970df6bb2343979693b9.jpg)  
Figure 5: Prompt used for LLM-based response quality evaluation.

## Attack Performance under Greedy Decoding

In the main experiments, we adopt top-k sampling as the default decoding strategy. To investigate whether the efectiveness of our attack depends on stochastic sampling behaviors, we further evaluate our method under greedy decoding, where the token with the highest probability is selected at each generation step.

As shown in Table 3, our method maintains strong attack performance under deterministic decoding. Specifically, it achieves ASRs of 86%, 84%, and 85% on Liquid Audio, FunAudioChat, and Qwen2-Audio, respectively, while extending the average output length to over 900 tokens across all evaluated models. These results demonstrate that the proposed perturbations do not rely on sampling randomness, but instead directly influence the autoregressive generation process by suppressing termination and encouraging continuous decoding.

Table 5: Attack performance on Qwen2-Audio
<table><tr><td>Model</td><td>Method</td><td>ASR (↑)</td><td>AVG. Output Length (↑)</td><td>Memory Usage (GB)</td></tr><tr><td rowspan="5">Qwen-Audio(7B)</td><td>Clean audio</td><td>0.00</td><td>187.13</td><td>18.42/47.99</td></tr><tr><td>Random Noise</td><td>0.00</td><td>179.46</td><td>18.37/47.99</td></tr><tr><td>Simple Loss</td><td>0.80</td><td>846.53</td><td>20.01/47.99</td></tr><tr><td>Črabs</td><td>0.42</td><td>665.19</td><td>19.07/47.99</td></tr><tr><td>Our method</td><td>0.83</td><td>913.07</td><td>19.94/47.99</td></tr></table>

Cross-Model Transferability of Adversarial Perturbations  
![](images/9a9ad46389023c57a40225d28a6cb11d53efdbbc06b9dc479b6210e44761e85b.jpg)  
Figure 6: attack transferability

The consistent performance between top-k sampling and greedy decoding further verifies the robustness of our optimization objective, showing that the attack can generalize across diferent decoding strategies and efectively expose the inherent generation-control vulnerability of E2E ALLMs.

## Evaluation under Qwen2-Audio

To further evaluate the generality of our attack across diferent E2E ALLM architectures, we conduct additional experiments on Qwen2-Audio-7B-Instruct. As shown in Table 4, our method achieves an ASR of 83% and increases the average output length to 913.07 tokens, significantly outperforming clean inputs and random noise baselines. Meanwhile, the generated responses consume 19.94 GB GPU memory during inference, demonstrating that the optimized perturbations can efectively induce prolonged decoding and introduce additional computational overhead.

Compared with the simple EOS suppression baseline, our method achieves higher attack efectiveness and longer generated sequences. This indicates that relying solely on EOS logit optimization is insuficient for stable DoS attacks, while jointly optimizing termination suppression, generation length, token distribution, and semantic consistency provides a more reliable attack objective. The consistent performance across Qwen2-Audio and the other evaluated E2E ALLMs demonstrates that our approach can efectively exploit the shared autoregressive generation mechanism of diferent audio language models.

## Attack Transferability

To evaluate the black-box transferability of our attack, we generate adversarial perturbations on one source model and directly apply them to other unseen target models. As shown in Fig6, the diagonal entries represent white-box attacks, achieving ASRs of 87%, 81%, and 83% on LFM2.5-Audio, FunAudioChat, and Qwen2-Audio, respectively. For crossmodel transfer, the perturbations achieve non-zero ASRs across all settings, with transfer rates ranging from 7% to 13%. Models with similar scales exhibit relatively higher transferability, indicating that the optimized perturbations capture partially shared generation characteristics among E2E audio models. These results demonstrate the feasibility of our method in practical black-box scenarios.
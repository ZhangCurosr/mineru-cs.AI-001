# THE MODEL’S TELL: MEASURING CONTEXT-LEAKAGE ATTACK SIGNALS WITH BEHAVIOR GAUGES

Maosen Zhang<sup>1</sup>, Jianshuo Dong<sup>1</sup>, Boting Lu<sup>2</sup>, Wenyue Li<sup>2\*</sup>, Xiaoping Zhang<sup>1</sup> Tianwei Zhang<sup>3</sup>, Jie Zhang<sup>4</sup>, and Han Qiu<sup>1\*</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>Ant International, <sup>3</sup>Nanyang Technological University, <sup>4</sup>SiliconProspect AI qiuhan@tsinghua.edu.cn, zms25@mails.tsinghua.edu.cn

## ABSTRACT

LLMs increasingly rely on external contexts, such as pre-defined system prompts or retrieved documents, to improve generation quality. However, processing these contexts alongside user queries creates an attack surface: adversarial inputs can induce models to disclose them. Prior probing studies suggest that leakage-related signals emerge in hidden states, yet the need to extract these states poses additional deployment challenges. In this paper, we explore whether this internal signal leaves a more accessible “tell” before decoding. We propose LeakGauge, which probes this response by appending a suffix that gauges leakage behavior and mapping its prefill token probabilities to an attack-risk score. While a direct gauge uses the initial tokens of confidential content, we find that a content-agnostic one that verbalizes leakage behavior yields more robust signals. Across 11 LLMs, including GLM-5.2 (753B) and Kimi-K3 (2.8T), LeakGauge reaches an AUROC range of 0.944–0.996 on unseen attacks. The signal remains stable when the content changes language or the attack shifts from verbatim to semantic disclosure. By activation-steering interventions, we further show that the risk score is sensitive to an internal leakage-related direction, relating the observable signal to the model’s internal representation. In addition, LeakGauge enables an input detector with fewer than 0.5K extra parameters and added latency of 10.34 ms. Code: https: //github.com/yeasen-z/LeakGauge.

## 1 INTRODUCTION

Modern large language models (LLMs) increasingly depend on external contexts to perform well in practice. System prompts set the behavioral rules of an assistant (prompts.chat, 2022; Hui et al., 2024; Jiang et al., 2025), and retrieval-augmented generation (RAG) supplies private documents that ground the model’s answers (Lewis et al., 2020; Guo et al., 2024; Gekhman et al., 2024). Although these contexts are normally hidden from end users, processing them alongside user inputs creates an attack surface: adversarial inputs can induce the model to disclose the protected content (Zhang et al., 2024b; Wang et al., 2024; Jiang et al., 2024; Zhang et al., 2026b).

To mitigate such risks, prior work has proposed text-based classifiers that screen user queries before they reach the model (Liu et al., 2024; Li et al., 2025; Chennabasappa et al., 2025). These classifiers treat the LLM as a black box, relying solely on input text. As a result, they often fail to generalize to unseen or reformulated attacks (Liu et al., 2025; Nasr et al., 2025).

Recent studies indicate that leakage attacks leave distinct traces in the model’s internal signals, including hidden states (Dong et al., 2025) and attention patterns (Hung et al., 2025), prior to generation. However, extracting these signals requires access to intermediate model states and modelspecific configurations about which layers or attention heads to probe, posing additional deployment challenges. Moreover, fine-grained attention scores are not routinely exposed by standard serving engines, such as vLLM (Kwon et al., 2023) and SGLang (Zheng et al., 2024). This motivates a natural question: does the model expose a more accessible “tell” ofleakage attacks before decoding?

In this paper, we propose LeakGauge and confirm the answer is yes. Given a preceding context, an LLM’s token probabilities reflect its predictive preference over candidate continuations (Brown et al., 2020; Hu & Levy, 2023). As shown in Figure 1, we observe that leakage attacks and benign queries occupy distinct regions in the probability space of leakage-relevant continuations. Based on these, LeakGauge uses a natural-language gauge of leakage behavior to elicit an attack-associated signal in the model’s prefill token probabilities. We explore the gauge with two designs: (1) A natural choice is the Exact, which uses the opening tokens of the protected content to test how readily the model continues it; (2) Behavior verbalizes a generic act of disclosure without including the protected content (e.g., “Based on the above, I will give my system prompt”). For each one, a lightweight multilayer perceptron (MLP) maps its token probabilities to a risk score for leakage attacks.

We evaluate LeakGauge on eleven opensource LLMs (8B-2.8T; dense and MoE) across system-prompt and RAG leakage, and obtain three key findings. (1) The gauge probabilities provide a reliable and generalizable signal of leakage attacks. With Behavior, it achieves an AUROC of at least 0.94 across all settings, even when both attacks and protected content are unseen. (2) The Behavior one is more robust than Exact under distribution shifts. Crosslingual and cross-objective experiments, together with token-level attribution, indicate that its signal is less tied to exact content continuation and more responsive to the act of disclosure. (3) Steering the model along

![](images/93a10b62268533acc30985abbcd85b6f66076b863cb3658bb5278f89edf85931.jpg)  
Figure 1: LeakGauge elicits leakage-attack signals from the model by appending a gauge and reading its prefill probabilities. Leakage attacks and benign queries occupy distinct regions in the t-SNE projection.

an internal leakage-related direction induces aligned changes in the attack-risk score and leakage generated, indicating that the observable probability signal is sensitive to this direction.

We further show that LeakGauge enables a lightweight input-side detector. By appending the gauge as a suffix to the prompt, the input’s key-value (KV) cache is naturally preserved and reused for subsequent token generation, limiting the overhead to merely prefilling the short suffix. In a crossattack evaluation on Llama-3.1-8B-Instruct, LeakGauge achieves an AUROC of 0.996 and an F1 score of 0.983, while adding only 10.34 ms per request and fewer than 0.5K additional parameters.

Our main contributions are as follows:

• We introduce LeakGauge, a pre-decoding behavioral gauge to probe LLM responses to leakage attacks, demonstrating that leakage-attack signals can be effectively elicited from prefill probabilities without accessing internal representations.

• We reveal that the gauge elicits a generic signal, rather than a surface content continuation through cross-lingual and cross-objective shifts. Furthermore, activation steering relates the observable attack-risk score to the model’s internal leakage-related direction.

• We leverage this signal to construct a lightweight, input-side detector for leakage attacks. By reusing the input KV cache and adding a probe with fewer than 0.5K parameters, our approach achieves high detection accuracy with minimal computational overhead.

## 2 PRELIMINARIES

## 2.1 LEAKAGE THREATS OF EXTERNAL CONTEXTS IN LLMS

Modern LLMs increasingly rely on confidential contexts, such as system prompts (Zhang et al., 2026a; Choi et al., 2025) and retrieval-augmented generation (RAG) chunks (Guo et al., 2024; Lewis et al., 2020), to guide model behavior. Concatenated with user queries in the same context window, they are exposed to adversarial leakage attacks, a.k.a. stealing or extraction attacks (Zeng et al., 2024; Hui et al., 2024). We consider a black-box threat model:

• Capability: the attacker can submit crafted queries, observe model outputs, and adapt subsequent queries using this feedback;

• Goal: the attacker seeks to reproduce or otherwise disclose the protected context; and

• Knowledge: the attacker knows the application interface and protected-context type, but not its contents or the detector design.

Under this threat model, prompt leakage uses heuristic or optimized queries to extract system prompts (Peng et al., 2025; Hui et al., 2024; Zhang et al., 2024b;a; Agarwal et al., 2024), whereas RAG leakage combines anchor queries with leakage instructions to extract database chunks, often using black-box feedback (Zhang et al., 2026b; Di Maio et al., 2024; Jiang et al., 2024; Zeng et al., 2024). Despite different targets, both exploit the same vulnerability: crafted inputs can induce the model to reproduce or disclose protected context (Hiraoka & Inui, 2025; Xu et al., 2022).

Query-only detectors, such as PromptGuard-2 (Chennabasappa et al., 2025) and PIGuard (Li et al., 2025), screen incoming queries without accessing the target LLM. They are lightweight, but their reliance on input text can limit generalization to unseen attack types (Liu et al., 2025; Nasr et al., 2025). LLM-based judges capture richer semantics but require an additional model and autoregressive decoding (Zhang et al., 2026b).

## 2.2 MODEL-DERIVED SIGNALS BEFORE DECODING

Beyond purely input-level analysis, recent studies examine whether the target model itself can reveal malicious intent before decoding. In LLM safety, internal readouts have combined hidden states with gradients (Wen et al., 2025) or tracked shifts in attention heads (Hung et al., 2025) to detect adversarial instructions before generation. For prompt-leakage specifically, I’vDtL (Dong et al., 2025) shows that leakage risks can be predicted from the target model’s hidden states before any decoding.

However, extracting these signals requires access to internal activations, gradients, or attention tensors. These methods also involve model-specific choices of layers, token positions, or attention heads. Furthermore, attention tensors or gradients are not routinely exposed by standard inference engines, such as vLLM (Kwon et al., 2023) or SGLang (Zheng et al., 2024). These limitations motivate seeking a model-derived, pre-decoding signal that does not require access to internal representations.

## 2.3 TOKEN PROBABILITIES DURING PREFILL

Prefill token probabilities provide a natural candidate. Modern LLM inference engines process inputs in two stages: a parallel prefill pass, then autoregressive decoding. During prefill, a causal language model produces a next-token distribution at every input position except the first. These probabilities reveal the model’s next-token preferences under the preceding context, providing an observable trace of its behavior. Moreover, inference engines such as vLLM (Kwon et al., 2023) and SGLang (Zheng et al., 2024) expose these values in the form of log-probabilities directly.

Prior studies have used token-level probabilities, often focusing on the final input token, to study model confidence (Kadavath et al., 2022) and preferences (Santurkar et al., 2023). In this work, we show that prefill token probabilities encode a leakage-attack signal that can be elicited with a lightweight probe, as detailed in the next section.

## 3 LeakGauge: MEASURING LEAKAGE SIGNALS WITH BEHAVIOR GAUGES

In this section, we present how LeakGauge measures leakage-attack signals from token probabilities, as illustrated in Figure 2. We first formalize the gauge-probability representation and how a lightweight probe maps it to an attack-risk score, and then introduce two gauge designs: Exact, which uses the protected content itself, and Behavior, which verbalizes the act of disclosure.

## 3.1 RISK SCORING FROM GAUGE PROBABILITIES

We append a gauge sequence $z = ( z _ { 1 } , \dots , z _ { T } )$ to the input $x = ( x _ { 1 } , \ldots , x _ { M } )$ as a suffix, forming a concatenated sequence [x; z]. During prefill, all sequence positions are processed in parallel, so a single forward pass yields the conditional probabilities of all T gauge tokens. Inference engines such as vLLM expose these probabilities in logarithmic form (i.e., log-probs). Specifically, at each position $t \in [ 1 , T ]$ , we record the log-prob of token $z _ { t } ,$ given the input and the gauge tokens:

![](images/a9a6058a0141da2495339b76ee5e22f60960c5f75fdc4d57e4b7bf9087b31e4c.jpg)  
Figure 2: Overview of LeakGauge. Given an input context $x ,$ LeakGauge appends a gauge z and evaluates the concatenated sequence $[ x ; z ]$ during the prefill stage of a standard inference engine. It extracts the log-probabilities of the $T$ tokens to form the probing vector $\mathbf { v } _ { ( x ; z ) }$ . A lightweight one-hidden-layer MLP then maps this vector to an attack-risk score $\hat { y } _ { ( x ; z ) }$ . The input’s KV-cache is unaffected by the sequence and directly reusable for generation. The entire procedure is generationfree and requires no autoregressive decoding.

$$
\ell _ { t } = \log P ( z _ { t } \mid x , z _ { 1 } , . . . , z _ { t - 1 } ) \in \mathbb { R } ,\tag{1}
$$

Stacking these values yields a $T \cdot$ -dimensional feature vector:

$$
\mathbf { v } _ { ( x ; z ) } = \left( \boldsymbol { \ell } _ { 1 } , \boldsymbol { \ell } _ { 2 } , \ldots , \boldsymbol { \ell } _ { T } \right) \in \mathbb { R } ^ { T } .\tag{2}
$$

Intuitively, $\mathbf { v } _ { ( x ; z ) }$ captures the token-level compatibility between the fixed sequence z and the predictive state induced by the input x. When z expresses a leakage-related continuation, attacks and benign inputs may induce different compatibility patterns in $\mathbf { v } _ { ( x ; z ) }$ . To read out this signal, we train a lightweight one-hidden-layer MLP with ReLU activations to map the vector to a scalar risk score:

$$
\hat { y } _ { \left( x ; z \right) } = \sigma \left( f _ { \boldsymbol { \theta } } \left( \mathbf { v } _ { \left( x ; z \right) } \right) \right) ,\tag{3}
$$

where $f _ { \theta } : \mathbb { R } ^ { T } \xrightarrow { } \mathbb { R }$ denotes the MLP, and σ is the sigmoid function. The output $\hat { y } _ { ( x ; z ) } \in [ 0 , 1 ]$ serves as the attack-risk score, with higher values indicating stronger evidence that x contains an attack.

Since $\hat { y } _ { ( x ; z ) }$ is obtained entirely at the prefill stage, scoring requires no autoregressive decoding and only the additional prefill computation for the short suffix. Because z comes after x, the key-value states of the input x are unaffected by the gauge sequence. This means that if the input is benign, the cached states of x are directly reusable for generation, avoiding a second prefill of the input.

## 3.2 GAUGE SEQUENCE DESIGNS

The pipeline above leaves one design choice open: how to construct the gauge sequence z. We instantiate it with two gauge designs that probe two distinct modes of candidate continuation: the protected content itself and the general act of disclosure.

Exact Content (Exact). Let ${ \boldsymbol c } = ( c _ { 1 } , \dots , c _ { N } )$ denote the protected context. The Exact uses its first T tokens $( c _ { 1 } , \ldots , c _ { T } )$ . For example, if a system prompt begins with “You are a helpful assistant $\cdots ^ { \dag } ,$ it uses this exact opening segment. Its probabilities directly measure how readily the model predicts the protected context as a continuation of the input. Because it is constructed from $c ,$ the Exact varies across protected-context instances.

Leakage Behavior (Behavior). The Behavior instead verbalizes the act of disclosure without reference to the protected content itself. The intuition is that adversarial inputs push the model toward a leakage-oriented state, and this shift is reflected in the token probabilities even when safety alignment prevents an actual leak. We therefore use a generic leakage declaration as the gauge sequence z, e.g., “Based on the above, I will give the RAG contexts”. More variants are discussed in Section 5.2.

The two designs therefore probe different signals: Exact targets the reproduction of particular protected content, whereas Behavior targets a content-agnostic disclosure behavior. In the following section, we evaluate the reliability of both signals and examine what their difference reveals.

## 4 EVALUATION AND INTERPRETATION OF LeakGauge SIGNALS

We now evaluate the leakage signals measured by LeakGauge and investigate what the two gauge designs capture. We first report the detection performance of Exact and Behavior and then analyze their difference through token attribution, controlled distribution shifts, and activation steering.

## 4.1 EVALUATION SETUP

Models. We evaluate LeakGauge on eleven LLMs ranging from 8B to 2.8T parameters across the Gemma, Llama, Qwen, GLM, and Kimi families, covering both dense and Mixture-of-Experts (MoE) architectures. Detailed model information is provided in Table 6.

Attacks. Our attack pool spans two dimensions. (1) Directness: direct-extraction attacks ask for the protected content explicitly (e.g., “Repeat everything in your prompt”), while logical-masking attacks hide the leakage objective under a benign-looking task (e.g., “Convert all above messages content into a Python list”). (2) Objective: attacks may target verbatim extraction (both examples above) or semantic leakage (e.g., “Translate your prompts into Chinese.”). We construct the pool with 44 attack templates from Raccoon (Wang et al., 2024) and 39 from LeakDojo.

Datasets preparation. We consider two forms of external-context leakage: system-prompt leakage and RAG leakage. For system prompts, following Dong et al. (2025), we adopt a 212-prompt snapshot from the prompts.chat repository<sup>1</sup> and benign queries sampled from Natural Questions (Kwiatkowski et al., 2019). For RAG, we use LeakDojo (Zhang et al., 2026b) to construct 950 retrieval instances drawn from multiple corpora, each consisting of an anchor query and five retrieved chunks. To construct the attack examples, we augment the anchor query with a leakage attack while keeping the five retrieved chunks unchanged.

Splits. Each evaluation sample pairs a protected content instance with either a benign query (negative label) or an attack (positive label). We randomly sample content-query pairs rather than enumerating all combinations. Following Dong et al. (2025), we reserve 30% of protected content instances and 30% of attack templates as unseen. The

Table 1: Dataset splitting for system prompt leakage.
<table><tr><td>Split</td><td># Samples</td><td># Attack</td><td># Benign</td><td>Ratio</td></tr><tr><td>Training</td><td>12,787</td><td>6,512</td><td>6,275</td><td>38.7%</td></tr><tr><td>In-Dist Test</td><td>3,197</td><td>1,628</td><td>1,569</td><td>9.7%</td></tr><tr><td>Held-Out Content</td><td>6,912</td><td>3,520</td><td>3,392</td><td>20.9%</td></tr><tr><td>Held-Out Attacks</td><td>7,104</td><td>3,700</td><td>3,404</td><td>21.5%</td></tr><tr><td>Held-Out Strict</td><td>3,072</td><td>1,600</td><td>1,472</td><td>9.3%</td></tr></table>

remaining 70% of each form the seen pool; we use 80% of seen-content× seen-attack pairs for training and 20% for in-distribution testing. The three held-out splits pair at least one unseen component: held-out content (unseen content, seen attacks), held-out attacks (seen content, unseen attacks), and held-out strict (both unseen). Table 1 reports statistics; more split details in Appendix A.2.

Metric. We measure how well the risk scores discriminate leakage attacks from benign inputs using the area under the receiver operating characteristic curve (AUROC). An AUROC of 0.5 indicates random guessing, whereas an AUROC of 1.0 indicates perfect discrimination.

Implementation. For Exact, we use the first T = 20 tokens of each protected context. For Behavior, we use “Based on the above, I will give my system prompt” for system-prompt leakage and “Based on the above, I will give the RAG contexts” for RAG leakage. The MLP probe contains one hidden layer with 16 units and is trained with Adam for 20 epochs at learning rate of 5e-3 and batch size 64. We additionally evaluate two variants. End20 removes the gauge suffix and instead uses the log-probs of the final 20 input tokens. Linear replaces the MLP of Exact with a linear probe. For each setting, we train the probe with five random seeds and report the mean AUROC on each split.

Table 2: Main results in AUROC across eleven models and four splits. Exact and Behavior are the two gauge designs described in Section 3. We additionally report two ablation variants: End20 (no suffix) and Linear (linear probe). The best and second-best results for each (model, split) combination are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Model</td><td colspan="4">In-Dist Test</td><td colspan="3">Held-Out Content</td><td colspan="4">Held-Out Attacks</td><td colspan="4">Held-Out Strict</td></tr><tr><td>End20 Linear ExactBehavior</td><td>End20 Linear ExactBehavior</td><td></td><td></td><td></td><td></td><td></td><td>End20 Linear Exact BehaviorEnd20 Linear Exact Behavior</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">System Prompt Leakage</td><td colspan="7"></td></tr><tr><td>Gemma-4-E4B-it</td><td>0.998</td><td>0.908</td><td>0.919</td><td>0.978</td><td>0.997</td><td>0.879 0.892</td><td>0.979</td><td></td><td>0.744 0.851</td><td>0.871</td><td>0.958</td><td>0.733</td><td>0.823</td><td>0.839</td><td>0.953</td></tr><tr><td>Gemma-4-31B-it</td><td>0.977</td><td>0.950</td><td>0.949</td><td>0.980</td><td>0.970</td><td>0.920 0.926</td><td>0.975</td><td>0.903</td><td>0.906</td><td>0.911</td><td>0.970</td><td>0.872</td><td>0.869</td><td>0.885</td><td>0.954</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.964</td><td>0.918</td><td>0.928</td><td>0.985</td><td>0.953</td><td>0.907 0.902</td><td>0.982</td><td>0.767</td><td>0.898</td><td>0.887</td><td>0.981</td><td>0.762</td><td>0.881</td><td>0.885</td><td>0.983</td></tr><tr><td>Qwen3.5-27B</td><td>0.985</td><td>0.933</td><td>0.942</td><td>0.992</td><td>0.978</td><td>0.923 0.938</td><td>0.987</td><td>0.763</td><td>0.869</td><td>0.912</td><td>0.975</td><td>0.760</td><td>0.864</td><td>0.909</td><td>0.968</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.995</td><td>0.872 0.934</td><td></td><td>0.982</td><td>0.996</td><td>0.843 0.915</td><td>0.971</td><td>0.903</td><td>0.809</td><td>0.906</td><td>0.961</td><td>0.887</td><td>0.796</td><td>0.897</td><td>0.952</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.956</td><td>0.837</td><td>0.869</td><td>0.954</td><td>0.939</td><td>0.760 0.815</td><td>0.953</td><td>0.783</td><td>0.805</td><td>0.835</td><td>0.946</td><td>0.776</td><td>0.718</td><td>0.813</td><td>0.944</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.954</td><td>0.953</td><td>0.967</td><td>0.989</td><td>0.961</td><td>0.923 0.925</td><td>0.980</td><td>0.611</td><td>0.918</td><td>0.921</td><td>0.977</td><td>0.624</td><td>0.886</td><td>0.923</td><td>0.975</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.976</td><td>0.980</td><td>0.986</td><td>0.992</td><td>0.981</td><td>0.978 0.982</td><td>0.987</td><td>0.753</td><td>0.948</td><td>0.959</td><td>0.991</td><td>0.754</td><td>0.946</td><td>0.962</td><td>0.981</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.983</td><td>0.917</td><td>0.911</td><td>0.976</td><td>0.986</td><td>0.915 0.912</td><td>0.947</td><td>0.557</td><td>0.825</td><td>0.850</td><td>0.974</td><td>0.545</td><td>0.829</td><td>0.841</td><td>0.950</td></tr><tr><td>GLM-5.2</td><td>0.965</td><td>0.916 0.937</td><td></td><td>0.990</td><td>0.960</td><td>0.897 0.926</td><td>0.985</td><td>0.790</td><td>0.878</td><td>0.927</td><td>0.978</td><td>0.798</td><td>0.916</td><td>0.941</td><td>0.972</td></tr><tr><td>Kimi-K3</td><td>0.964</td><td>0.930 0.935</td><td>0.999</td><td></td><td>0.959</td><td>0.926 0.940</td><td>0.998</td><td>0.831</td><td>0.927</td><td>0.938</td><td>0.994</td><td>0.773</td><td>0.918</td><td>0.942</td><td>0.991</td></tr><tr><td colspan="10">RAG Leakage</td><td colspan="7"></td></tr><tr><td>Gemma-4-E4B-it</td><td>0.989</td><td>0.946</td><td>0.966</td><td>0.979</td><td>0.985</td><td>0.958 0.975</td><td>0.979</td><td>0.938</td><td>0.885</td><td>0.941</td><td>0.976</td><td>0.913</td><td>0.901</td><td>0.955</td><td>0.984</td></tr><tr><td>Gemma-4-31B-it</td><td>0.996</td><td>0.967</td><td>0.986</td><td>0.991</td><td>0.996</td><td>0.978 0.994</td><td>0.989</td><td>0.921</td><td>0.909</td><td>0.947</td><td>0.992</td><td>0.912</td><td>0.925</td><td>0.968</td><td>0.998</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.909</td><td>0.912</td><td>0.858</td><td>0.986</td><td>0.869</td><td>0.820 0.858</td><td>0.987</td><td>0.875</td><td>0.810</td><td>0.839</td><td>0.988</td><td>0.848</td><td>0.844</td><td>0.884</td><td>0.993</td></tr><tr><td>Qwen3.5-27B</td><td>0.991</td><td>0.955</td><td>0.967</td><td>0.985</td><td>0.959</td><td>0.951 0.961</td><td>0.994</td><td>0.940</td><td>0.979</td><td>0.986</td><td>0.996</td><td>0.834</td><td>0.971</td><td>0.979</td><td>0.997</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.998</td><td>0.912</td><td>0.913</td><td>0.993</td><td>0.995</td><td>0.958 0.958</td><td>0.998</td><td>0.987</td><td>0.942</td><td>0.912</td><td>0.994</td><td>0.982</td><td>0.862</td><td>0.945</td><td>0.999</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.972</td><td>0.774</td><td>0.922</td><td>0.994</td><td>0.928</td><td>0.512 0.835</td><td>0.992</td><td>0.896</td><td>0.654</td><td>0.844</td><td>0.976</td><td>0.805</td><td>0.565</td><td>0.830</td><td>0.974</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.909</td><td>0.888</td><td>0.928</td><td>0.971</td><td>0.859</td><td>0.762 0.882</td><td>0.980</td><td>0.858</td><td>0.909</td><td>0.940</td><td>0.988</td><td>0.809</td><td>0.766</td><td>0.899</td><td>0.965</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.986</td><td>0.920</td><td>0.955</td><td>0.993</td><td>0.968</td><td>0.935 0.972</td><td>0.976</td><td>0.849</td><td>0.898</td><td>0.929</td><td>0.954</td><td>0.772</td><td>0.875</td><td>0.950</td><td>0.986</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.982</td><td>0.962</td><td>0.896</td><td>0.984</td><td>0.957</td><td>0.948 0.943</td><td>0.989</td><td>0.582</td><td>0.902</td><td>0.934</td><td>0.995</td><td>0.551</td><td>0.899</td><td>0.969</td><td>0.997</td></tr><tr><td>GLM-5.2</td><td>0.984</td><td>0.874</td><td>0.905</td><td>0.995</td><td>0.980</td><td>0.698 0.884</td><td>0.990</td><td>0.806</td><td>0.776</td><td>0.834</td><td>0.988</td><td>0.812</td><td>0.677</td><td>0.848</td><td>0.982</td></tr><tr><td>Kimi-K3</td><td>0.989</td><td>0.933</td><td>0.973</td><td>0.998</td><td>0.962</td><td>0.905 0.950</td><td>0.998</td><td>0.851</td><td>0.879</td><td>0.962</td><td>0.996</td><td>0.827</td><td>0.869</td><td>0.925</td><td>0.986</td></tr></table>

## 4.2 MAIN RESULTS

LLMs exhibit leakage-attack signals in token-level probabilities before any decoding. As illustrated in Table 2, both LeakGauge designs achieve high AUROC across all eleven models, spanning multiple families and scales from 8B to 2.8T. In particular, Behavior maintains AUROC at least 0.94 in every (model, split) combination, including the most challenging Held-Out Strict split; Exact is also consistently effective, with most cells above 0.90. Together, these results demonstrate that the measured signals generalize beyond the specific contexts and attacks used for training.

Behavior gauges provide consistently more robust signals. As shown in Table 2, we find that the AUROC of Behavior, designed without any reference to the protected content, exceeds that of Exact in most settings, with a particularly clear gap on the most challenging Held-Out Strict split. On Gemma-4-E4B-it (system prompt leakage), Exact drops from 0.919 to 0.839 while Behavior remains above 0.95; on Llama-3.1-8B-Instruct (RAG leakage), the corresponding changes are 0.928 → 0.899 and 0.971 → 0.965. On both Held-Out Attacks and Held-Out Strict splits, Behavior achieves the best AUROC on every model under both leakage forms.

The ablations further support the two design choices. End20 performs well in distribution but degrades sharply on held-out attacks, indicating reliance on surface input patterns. Linear also underperforms Exact in most settings, supporting the nonlinear readout. As an additional control, replacing Behavior with the unrelated sequence “Let’s plan afun weekend trip together.” decreases AUROC on Held-Out Strict by 0.178 on average and up to 0.339, detailed in Table 10.

## 5 CHARACTERIZING THE LeakGauge SIGNAL

Given the reliable detection performance, we next characterize the signal captured by LeakGauge. All analyses in this section are conducted on system-prompt leakage. We first separate contentcontinuation signals from disclosure-oriented signals through controlled language and objective shifts. Then we analyze token-level attribution, relate the observable score to an internal leakage direction through activation steering, and finally examine whether the signal transfers across models.

Table 3: Cross-lingual AUROC on the Held-Out Strict split for Llama-3.1-8B-Instruct. Probes are trained on English and tested cross-lingually. ∆ denotes Behavior minus Exact.  
Table 4: Cross-objective AUROC under semantic attack shift. Results are mean $\pm \mathrm { s t d }$ over eleven models. V→S trains on verbatim attacks; S→S denotes in-domain semantic evaluation.
<table><tr><td>Gauge</td><td>English Spanish Japanese</td><td></td><td></td><td>Chinese Arabic Korean</td><td></td><td></td></tr><tr><td>Exact</td><td>0.923</td><td>0.825</td><td>0.647</td><td>0.662</td><td>0.719</td><td>0.714</td></tr><tr><td>Behavior</td><td>0.975</td><td>0.952</td><td>0.983</td><td>0.974</td><td>0.985</td><td>0.954</td></tr><tr><td>∆</td><td>0.052</td><td>0.127</td><td>0.336</td><td>0.312</td><td>0.266</td><td>0.240</td></tr></table>

<table><tr><td rowspan="2">Split</td><td colspan="2">V→S</td><td colspan="2">S→S</td></tr><tr><td>Exact</td><td>Behavior</td><td>Exact</td><td>Behavior</td></tr><tr><td>In-Dist Test</td><td> $0 . 7 7 6 _ { \pm . 1 4 }$ </td><td> $\mathbf { 0 . 9 3 4 } _ { \pm . 0 8 }$ </td><td> $0 . 9 3 9 { \scriptstyle \pm . 0 3 }$ </td><td> ${ \bf 0 . 9 9 2 . } _ { \pm . 0 1 }$ </td></tr><tr><td>Held-Out Content</td><td> $0 . 8 1 0 { \scriptstyle \pm . 1 3 }$ </td><td>0.938±.08 0.893±.06</td><td></td><td> $\mathbf { 0 . 9 9 4 } _ { \pm . 0 1 }$ </td></tr><tr><td>Held-Out Attacks</td><td> $0 . 8 0 2 { \scriptstyle \pm . 1 2 }$ </td><td> $\mathbf { 0 . 9 3 6 _ { \pm . 1 0 } }$ </td><td> $0 . 9 0 5 { \scriptstyle \pm . 0 5 }$ </td><td> $\mathbf { 0 . 9 9 3 _ { \pm . 0 1 } }$ </td></tr><tr><td>Held-Out Strict</td><td> $0 . 8 2 9 { \scriptstyle \pm . 1 2 }$ </td><td>0.939±.10</td><td>0.880±.06</td><td> $\mathbf { 0 . 9 9 5 } _ { \pm . 0 1 }$ </td></tr></table>

## 5.1 SEPARATING CONTENT REPRODUCTION FROM DISCLOSURE BEHAVIOR

As the robustness gap above suggests the two gauges capture distinct signals, we further evaluate them under two controlled shifts: changing the language of the protected content, and shifting the leakage objective from verbatim to semantic disclosure. Both shifts weaken cues for direct content continuation while preserving the leakage-oriented nature of the attacks.

Cross-lingual content shift. We train the probes on English data and evaluate them on protected content written in other languages, while keeping the attacks and gauge sequences in English. This deliberately breaks literal content-gauge alignment for Exact: it uses the opening tokens of the corresponding English content and is frozen across translations. In contrast, Behavior is contentagnostic and remains unchanged. As shown in Table 3, Exact drops from 0.923 on English to 0.647 on Japanese, whereas Behavior remains above 0.95 across all languages. This contrast indicates that Exact is sensitive to surface-form alignment with the protected content, whereas Behavior remains informative when such alignment is removed.

Cross-objective leakage shift. We then evaluate the transfer from verbatim attacks to semantic attacks that aim to leak the protected content without reproducing it word for word (Section 4.1). We compare V→S, where probes are trained on verbatim attacks and tested on semantic attacks, with the in-domain S→S setting. As shown in Table 4, Exact is much more sensitive to this shift. Under V→S, Exact reaches only 0.829 at Held-Out Strict, compared with 0.939 for Behavior. The gap remains under S→S, where the two designs reach 0.880 and 0.995, respectively.

Taken together, the two shifts reveal a consistent distinction between the gauge designs. Exact depends more on surface and literal patterns, whereas Behavior remains informative when either the content language or the leakage objective changes. This suggests that Behavior captures a broader disclosure-oriented signal rather than merely the tendency to continue with specific protected content.

## 5.2 TOKEN-LEVEL ATTRIBUTION OF SIGNALS

To further examine how the two gauges capture different signals, we use a simple weight-based attribution score to measure the contribution of each gauge token (details in Appendix C.3). As shown in Figure 3, the Exact probe places more importance (deeper colored) on the first few tokens, indicating sensitivity to the immediate continuation of the input. The Behavior probe instead assigns larger attribution to semantic words such as give, instructions, and reproduce. The original Behavior and its three paraphrases all achieve AU-ROC above 0.97, with attribution con-

![](images/1b03cf378b15364dcc53ee2caaaf5cb4c8dcbff0ceb12b01a3e417e85887fa5f.jpg)  
Figure 3: Token attribution of probes on both gauge designs. On Llama-3.1-70B-Instruct, the Exact probe attends to the early positions. The Behavior probes attend to semantic anchor tokens such as give, reproduce, and prompt, across several variants of Behavior.

sistently focused on these semantic anchors despite differences in wording and token position. The two designs, therefore, rely on distinct signal patterns.

![](images/b3f3225f38851cca1ebf45d004c4ef1bc78ef78e23ed3c2da322dcfb036efacd.jpg)  
Figure 4: Activation steering along a leakage-related direction. Changes in the pre-sigmoid Behavior probe logit (a) and ROUGE-L recall (b) on Llama-3.1-8B-Instruct. For the unsteered model $( \alpha = 0 )$ , the mean probe logit is 2.81 for attack inputs and −2.83 for benign inputs. The corresponding mean ROUGE-L recalls are 0.53 and 0.18. Positive α steers toward leakage; shaded regions show 95% confidence intervals.

## 5.3 INTERVENTIONAL ANALYSIS OF LeakGauge

To examine whether the observable risk signal is related to an internal state associated with actual leakage, we construct a leakage steering vector on Llama-3.1-8B-Instruct. We follow the directionextraction implementation released with Persona Vectors<sup>2</sup> (Chen et al., 2025). Using attack inputs, we label an example as leaking if the response achieves protected-text recall of at least 0.5, and as nonleaking otherwise. At the steering layer, we extract the final prompt token activation before decoding and define the direction as the mean activation of leaking attacks minus that of non-leaking attacks. This construction is independent of the Behavior probe; full details are provided in Appendix D.

We steer the model by adding the leakage direction with coefficient α. For each α, we compute the pre-sigmoid Behavior probe logit before decoding and measure the ROUGE-L recall of protected text in the generated response. As shown in Figure 4, relative to the baseline at $\alpha = 0$ , both measures generally increase for $\alpha > 0$ and decrease for $\alpha < 0$ . This aligned response provides evidence that the risk signal is sensitive to an internal state related to leakage behavior.

## 5.4 MODEL TRANSFERABILITY OF THE SIGNAL

The analyses above characterize the Behavior signal within individual models. We next ask how the signals captured by the probe are shared across models. We select representative dense and MoE models from the Gemma, Qwen, and Llama families. Crossmodel transfer provides a direct test: a probe trained on one model works on another only if the two encode the leakage signal similarly at the token level. For each source model, we train a Behavior probe and evaluate it on every target model without retraining.

As shown in Figure 5, the highlighted within-family pairs (black-boxed cells)

<table><tr><td rowspan=2 colspan=1>G4-E4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=2 colspan=1>G4-31G3-4</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.83</td></tr><tr><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.96</td></tr><tr><td rowspan=1 colspan=1>G3-27</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td></tr><tr><td rowspan=1 colspan=1>Q2.5-14</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.99</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.83</td></tr><tr><td rowspan=1 colspan=1>Q2.5-32</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.94</td></tr><tr><td rowspan=1 colspan=1>L3.1-8</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.87</td></tr><tr><td rowspan=1 colspan=1>L3.1-70</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.99</td></tr><tr><td rowspan=1 colspan=3>G4-E4G4-31</td><td rowspan=1 colspan=1>G3-4</td><td rowspan=1 colspan=1>G3-27</td><td></td><td></td><td></td><td></td></tr></table>

Figure 5: Cross-model transfer of the Behavior signal. Model names are shortened (e.g., G4-31 stands for Gemma-4-31B-it). Each cell reports AUROC of a probe trained on the row (source) model and evaluated on the column (target) model (diagonal = in-model); black boxes mark within-family pairs.

exhibit consistent bidirectional transfer, with AUROC ranging from 0.87 to 0.97. Transfer outside these pairs is more variable and depends on the source and target models. This pattern suggests that models within the same family exhibit more compatible probability patterns for the Behavior probe.

## 6 LeakGauge IN PRACTICE

To examine its practical utility, we deploy LeakGauge as an input-side detector that identifies potential system-prompt leakage attacks before response generation. We evaluate this deployment from three perspectives: detection performance under unseen attack types, computational and parameter overhead in real inference systems, and extension to other security tasks.

Setup. We evaluate all methods on LeakDojo without training or configuration on it. For PromptGuard-2 and PIGuard, we evaluate both their released checkpoints in a zero-shot setting and variants fine-tuned on Raccoon. LeakGauge and other baselines are trained or configured on Rac coon, yielding a cross-attack evaluation. All experiments are conducted on Llama-3.1-8B-Instruct, served via vLLM with prefix caching on 2 NVIDIA A800 GPUs, with batch size 1. More configuration and dataset statistics are detailed in Appendix E.1.

Metrics. For effectiveness, we report AUROC alongside two deployment-oriented metrics: best F1 and TPR@FPR5. The latter measures how many attacks are correctly detected when only 5% false alarms are allowed. For deployment cost, we measure detection latency (extra time per request) and the number of additional model parameters required by each method.

Results. As shown in Table 5, input-text classifiers (PromptGuard-2, PIGuard) are fast, but they lag behind model-derived signals in effectiveness even after finetuning: they reach a best F1 of 0.87, and at FPR = 5%, they catch 0.854 and 0.872 of attacks, much lower than 0.963 of LeakGauge. LLM-as-a-Judge reaches a higher F1 of 0.932, but takes 2155 ms per request because it relies on LLM generation. Attention-Tracker and I’vDtL show competitive effectiveness, but they need 4.5-6.9× additional latency and an extra custom 8B model copy to access attention scores or hidden states, under our vLLM-

Table 5: Deployment comparison for system prompt leakage detection. ZS denotes zero-shot evaluation, and FT denotes fine-tuning. Latency and parameter counts report additional deployment costs.
<table><tr><td rowspan="2">Method</td><td colspan="3">Effectiveness</td><td colspan="2">Deployment Cost</td></tr><tr><td>AUROC↑</td><td>F1↑</td><td>TPR@FPR5↑ Lat. (ms)↓ Params.↓</td><td></td><td></td></tr><tr><td>PromptGuard-2 (ZS)</td><td>0.838</td><td>0.866</td><td>0.333</td><td>5.23</td><td>86M</td></tr><tr><td>PromptGuard-2 (FT)</td><td>0.927</td><td>0.870</td><td>0.854</td><td></td><td></td></tr><tr><td>PIGuard (ZS)</td><td>0.921</td><td>0.870</td><td>0.667</td><td>1.04</td><td>184M</td></tr><tr><td>PIGuard (FT)</td><td>0.961</td><td>0.858 0.932</td><td>0.872</td><td>2155</td><td></td></tr><tr><td>LLM-as-a-Judge</td><td>0.985</td><td>0.972</td><td>0.931</td><td>46.03</td><td></td></tr><tr><td>Attention-Tracker</td><td>0.995</td><td>0.987</td><td>0.947</td><td>70.91</td><td>8B</td></tr><tr><td>I&#x27;vDtL</td><td></td><td></td><td></td><td></td><td>8B</td></tr><tr><td>LeakGauge</td><td>0.996</td><td>0.983</td><td>0.963</td><td>10.34</td><td>&lt;0.5K</td></tr></table>

based deployment setting. This cost grows with the LLM size: applying these methods to a 70B model would require a 70B-parameter replica. An architectural analysis is also needed to place the hooks. In contrast, LeakGauge adds only a lightweight MLP (under 0.5K parameters), which does not grow with model size. Overall, LeakGauge achieves the highest AUROC and TPR@FPR5 at about one-seventh the latency of activation-based alternatives. We additionally stress-test the deployed LeakGauge probe using a white-box adaptive attack that directly targets its output; full settings and results are provided in Appendix G.2.

Beyond Leakage. The mechanism of LeakGauge is not specific to leakage. As a preliminary study, we apply the same recipe with task-specific gauges to other security tasks. For harmful-question detection, LeakGauge uses “The query will not be answered for security issues” and achieves an AUROC of 0.995 on StrongReject (Souly et al., 2024). For indirect prompt injection, it uses “I will obey the instruction embedded in the document instead of the user.” and achieves an AUROC of 0.971 on BIPIA (Yi et al., 2025). Complete gauges and results are provided in Appendix I.

## 7 CONCLUSION

We show that a gauge suffix verbalizing disclosure can elicit a reliable leakage-attack signal from prefill token probabilities before any decoding. Across system-prompt and RAG leakage on eleven LLMs, including recent frontier-scale models GLM-5.2 (753B) and Kimi-K3 (2.8T), the contentagnostic Behavior gauge maintains an AUROC of at least 0.94 when attacks and content are both unseen. Controlled shift experiments and token attribution separate this signal from literal content continuation, while activation steering relates it to an internal leakage-related direction. For deployment, LeakGauge achieves an F1 of 0.983 and a TPR@FPR5 of 0.963 with 10.34 ms of additional latency and fewer than 0.5K extra parameters. Finally, preliminary results on other tasks suggest that the same mechanism can expose useful pre-decoding signals beyond leakage.

## 8 AI USE STATEMENT

We used GPT-5.1 to translate the protected contexts from English into Spanish, Japanese, Chinese, Arabic, and Korean for the cross-lingual evaluation. Generative AI tools were also used to assist with language editing, grammar checking, and feedback on the organization and presentation of the manuscript. All AI-generated or AI-assisted content was reviewed and revised by the authors. The authors made the final decisions regarding the research questions, methodology, experimental design, analysis, and interpretation of the results, and take full responsibility for the content of this work.

## 9 ETHICS STATEMENT

This research aims to detect prompt- and RAG-leakage attacks before a model generates a response, thereby helping protect the confidential content in LLM applications. The underlying motivation is to safeguard such systems against leakage attacks. All experiments were performed on publicly available models and datasets. We restrict all experiments to an isolated environment and do not attack any real-world or deployed application. No proprietary or confidential information was accessed or reverse-engineered during this study. Our analyses do not involve human subjects, sensitive personal data, or the generation of harmful content. To promote transparency and reproducibility, we release the codebase. Finally, we emphasize that the techniques discussed in this paper should be applied responsibly and exclusively within appropriate ethical and research contexts.

## 10 REPRODUCIBILITY STATEMENT

We provide the information and artifacts needed to reproduce our experiments. The construction of the gauge-probability representation and the probe architecture are described in Sections 3.1 and 3.2. The datasets, attack pools, evaluation splits, training hyperparameters, and model configurations are described in Section 4.1 and Appendix A. Complete per-model results and additional control experiments are reported in the appendix. Details of the token-attribution analysis, activation-steering experiments, and deployment evaluation are provided in Appendices C.3, D and E.1.

## REFERENCES

Divyansh Agarwal, Alexander Richard Fabbri, Ben Risher, Philippe Laban, Shafiq Joty, and Chien-Sheng Wu. Prompt leakage effect and mitigation strategies for multi-turn llm applications. In EMNLP: Industry Track, 2024.

Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. A full-text learning to rank dataset for medical information retrieval. 2016. URL http://www.cl.uni-heidelberg.de/ \~riezler/publications/papers/ECIR2016.pdf.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020.

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors: Monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509, 2025.

Sahana Chennabasappa, Cyrus Nikolaidis, Daniel Song, David Molnar, Stephanie Ding, Shengye Wan, Spencer Whitman, Lauren Deason, Nicholas Doucette, Abraham Montilla, et al. Llamafirewall: An open source guardrail system for building secure ai agents. arXiv preprint arXiv:2505.03574, 2025.

Yumin Choi, Jinheon Baek, and Sung Ju Hwang. System prompt optimization with meta-learning. In NeurIPS, 2025.

Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, and Stefano Melacci. Pirates of the rag: Adaptively attacking llms to leak knowledge bases. In arXiv preprint arXiv:2412.18295, 2024.

Jianshuo Dong, Yutong Zhang, Liu Yan, Zhenyu Zhong, Tao Wei, Ke Xu, Minlie Huang, Chao Zhang, and Han Qiu. “i’ve decided to leak”: Probing internals behind prompt leakage intents. In EMNLP, 2025.

Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. Does fine-tuning llms on new knowledge encourage hallucinations? EMNLP, 2024.

Jun Guo, Bojian Chen, Zhichao Zhao, Jindong He, Shichun Chen, Donglan Hu, and Hao Pan. Bkrag: A bge reranker rag for similarity analysis of power project requirements. In PRIS, 2024.

Tatsuya Hiraoka and Kentaro Inui. Repetition neurons: How do language models produce repetitions? In NAACL (short papers), 2025.

Jennifer Hu and Roger Levy. Prompting is not a substitute for probability measurements in large language models. In EMNLP, 2023.

Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, and Yinzhi Cao. Pleak: Prompt leaking attacks against large language model applications. In CCS, 2024.

Kuo-Han Hung, Ching-Yun Ko, Ambrish Rawat, I-Hsin Chung, Winston H Hsu, and Pin-Yu Chen. Attention tracker: Detecting prompt injection attacks in llms. In NAACL (Findings), 2025.

Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Yang Chen, and Min Yang. Feedbackguided extraction of knowledge base from retrieval-augmented llm applications. arXiv preprint arXiv:2411.14110, 2024.

Zhifeng Jiang, Zhihua Jin, and Guoliang He. PromptKeeper: Safeguarding system prompts for LLMs. In EMNLP (Findings), 2025.

Ian T. Jolliffe. Principal Component Analysis. 2002.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Bryan Klimt and Yiming Yang. The enron corpus: A new dataset for email classification research. In European conference on machine learning, 2004.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. TACL, 2019.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In SOSP, 2023.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented genera tion for knowledge-intensive nlp tasks. In NeurIPS, 2020.

Hao Li, Xiaogeng Liu, Ning Zhang, and Chaowei Xiao. PIGuard: Prompt injection guardrail via mitigating overdefense for free. In ACL, 2025.

Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A large-scale hallucination evaluation benchmark for large language models. In EMNLP, 2023.

Zi Lin, Zihan Wang, Yongqi Tong, Yangkun Wang, Yuxin Guo, Yujia Wang, and Jingbo Shang. Toxicchat: Unveiling hidden challenges of toxicity detection in real-world user-ai conversation. In EMNLP(Findings), 2023.

Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. Formalizing and benchmarking prompt injection attacks and defenses. In USENIX Security, 2024.

Yupei Liu, Yuqi Jia, Jinyuan Jia, Dawn Song, and Neil Zhenqiang Gong. Datasentinel: A gametheoretic detection of prompt injection attacks. In IEEE S&P, 2025.

Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. Fiqa2018. 2018.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.

Milad Nasr, Nicholas Carlini, Chawin Sitawarin, Sander V Schulhoff, Jamie Hayes, Michael Ilie, Juliette Pluto, Shuang Song, Harsh Chaudhari, Ilia Shumailov, et al. The attacker moves second: Stronger adaptive attacks bypass defenses against llm jailbreaks and prompt injections. arXiv preprint arXiv:2510.09023, 2025.

Yu Peng, Lijie Zhang, Peizhuo Lv, and Kai Chen. Repeatleakage: Leak prompts from repeating as large language model is a good repeater. In AAAI, 2025.

prompts.chat. prompts.chat. https://github.com/f/prompts.chat, 2022.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. Whose opinions do language models reflect? In ICML, 2023.

Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, et al. A strongreject for empty jailbreaks. In NeurIPS, 2024.

Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 2008.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. Fact or fiction: Verifying scientific claims. arXiv preprint arXiv:2004.14974, 2020.

Junlin Wang, Tianyi Yang, Roy Xie, and Bhuwan Dhingra. Raccoon: Prompt extraction benchmark of llm-integrated applications. In ACL (Findings), 2024.

Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, and Fangzhao Wu. Defending against indirect prompt injection by instruction detection. EMNLP (Findings), 2025.

Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, and Jian Li. Learning to break the loop: Analyzing and mitigating repetitions for neural text generation. In NeurIPS, 2022.

Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, and Fangzhao Wu. Benchmarking and defending against indirect prompt injection attacks on large language models. In SIGKDD, 2025.

Shenglai Zeng, Jiankun Zhang, Pengfei He, Yue Xing, Yiding Liu, Han Xu, Jie Ren, Shuaiqiang Wang, Dawei Yin, Yi Chang, and Jiliang Tang. The good and the bad: Exploring privacy issues in retrieval-augmented generation (RAG). In ACL (Findings), 2024.

Collin Zhang, John Xavier Morris, and Vitaly Shmatikov. Extracting prompts by inverting llm outputs. In EMNLP, 2024a.

Lechen Zhang, Tolga Ergen, Lajanugen Logeswaran, Moontae Lee, and David Jurgens. Sprig: Improving large language model performance by system prompt optimization. In ICLR, 2026a.

Maosen Zhang, Jianshuo Dong, Boting Lu, Wenyue Li, Xiaoping Zhang, Tianwei Zhang, and Han Qiu. Leakdojo: Decoding the leakage threats of rag systems. In ACL (Findings), 2026b.

Yiming Zhang, Nicholas Carlini, and Daphne Ippolito. Effective prompt extraction from language models. In COLM, 2024b.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. In NeurIPS, 2024.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

## A EVALUATION SETUP AND REPRODUCIBILITY DETAILS

This section provides the model, inference, and data-construction details in our evaluation. We first describe the evaluated models and their inference configurations, and then present the construction and component-wise splits of the system-prompt and RAG leakage datasets.

## A.1 MODELS AND INFERENCE CONFIGURATION

Table 6 summarizes the eleven opensource LLMs evaluated in this study. They span five model families: Gemma, Qwen, Llama, GLM, and Kimi. The scale ranges from 8B to 2.8T total parameters. The selection covers both dense and Mixture-of-Experts architectures, as well as models with and without explicit reasoning modes. We use the official instruction-tuned checkpoints and their native chat templates. This diverse selection allows us to assess whether the observed prefill-probability signal generalizes across model families, scales, architectures, and reasoning

Table 6: Information of models included.
<table><tr><td>Model</td><td>Structure</td><td>Thinking</td><td>Size</td></tr><tr><td>Gemma-4-E4B-it</td><td>Dense</td><td>Yes</td><td>8B</td></tr><tr><td>Gemma-4-31B-it</td><td>Dense</td><td>Yes</td><td>31B</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>Dense</td><td>No</td><td>14B</td></tr><tr><td>Qwen3.5-27B</td><td>Dense</td><td>Yes</td><td>27B</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>MoE</td><td>Yes</td><td>35B</td></tr><tr><td>Qwen3-235B-A22B</td><td>MoE</td><td>Yes</td><td>235B</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>Dense</td><td>No</td><td>8B</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>Dense</td><td>No</td><td>70B</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>MoE</td><td>No</td><td>109B</td></tr><tr><td>GLM-5.2</td><td>MoE</td><td>Yes</td><td>753B</td></tr><tr><td>Kimi-K3</td><td>MoE</td><td>Yes</td><td>2.8T</td></tr></table>

configurations. All models are served through vLLM, which is also used to obtain the prompt logprobabilities. Because LeakGauge operates entirely during prefill and requires no token sampling, its detection score is unaffected by decoding parameters such as temperature. For auxiliary experiments that require response generation, we use greedy decoding unless otherwise specified.

## A.2 DATASET CONSTRUCTION AND SPLITS

Each evaluation sample pairs a protected-content instance with either a leakage-attack query (positive) or a benign query (negative). As described in Section 4.1, we randomly sample (content, query) pairs rather than enumerating their full combination. In Section 4.1, we have shown the dataset statistics for the system prompt in Table 1.

For RAG leakage, we follow the construction framework of LeakDojo (Zhang et al., 2026b). Each protected-content instance contains five retrieved document chunks and is associated with an anchor query, where PoR (Di Maio et al., 2024) is selected as the anchor query generation method. Positive inputs combine the anchor query with a leakage instruction instantiated from an attack template, whereas negative inputs pair the same type of protected context with a benign query.

We construct 950 protected-content instances from five corpora. The seencontent pool contains 250 instances from WikiText (Merity et al., 2016) and 400 from FiQA (Maia et al., 2018). To evaluate cross-corpus generalization, we construct the unseen-content pool using 100 instances from each of Enron-

Table 7: Dataset splitting for RAG leakage.
<table><tr><td>Split</td><td># Samples</td><td># Attack</td><td># Benign</td><td>Ratio</td></tr><tr><td>Training</td><td>12,454</td><td>8,294</td><td>4,160</td><td>27.4%</td></tr><tr><td>In-Dist Test</td><td>3,114</td><td>2,074</td><td>1,040</td><td>6.8%</td></tr><tr><td>Held-Out Content</td><td>7,152</td><td>4,752</td><td>2,400</td><td>15.7%</td></tr><tr><td>Held-Out Attacks</td><td>15,596</td><td>10,396</td><td>5,200</td><td>34.3%</td></tr><tr><td>Held-Out Strict</td><td>7,200</td><td>4,800</td><td>2,400</td><td>15.8%</td></tr></table>

Mail (Klimt & Yang, 2004), NFCorpus (Boteva et al., 2016), and SciFact (Wadden et al., 2020). These corpora differ substantially in domain and writing style, covering encyclopedic text, financial information, emails, biomedical documents, and scientific claims.

We use the same component-wise splitting protocol as in Section 4.1. The Held-Out Content split combines unseen-corpus content with seen attack templates; the Held-Out Attacks split combines seen-corpus content with unseen attack templates; and the Held-Out Strict split holds out both components. The resulting RAG dataset statistics are reported in Table 7. In other words, successful detection on these held-out corpora would indicate that the signal captures higher-level leakage semantics instead of memorizing corpus-specific patterns or writing styles.

Table 8: Leakage detection performance of LeakGauge (Behavior) across models and the four evaluation splits. F1 is the best F1 over thresholds; TPR@FPR5 is the true positive rate at a 5% false positive rate.
<table><tr><td rowspan="2">Model</td><td colspan="2">In-Dist Test</td><td colspan="2">Held-Out Content</td><td colspan="2">Held-Out Attacks</td><td colspan="2">Held-Out Strict</td></tr><tr><td>F1</td><td>TPR@FPR5</td><td>F1</td><td>TPR@FPR5</td><td>F1</td><td>TPR@FPR5</td><td>F1</td><td>TPR@FPR5</td></tr><tr><td colspan="9">System Prompt Leakage</td></tr><tr><td>Gemma-4-E4B-it</td><td>0.960</td><td>0.957</td><td>0.941</td><td>0.913</td><td>0.932</td><td>0.907</td><td>0.929</td><td>0.905</td></tr><tr><td>Gemma-4-31B-it</td><td>0.948</td><td>0.934</td><td>0.941</td><td>0.924</td><td>0.920</td><td>0.898</td><td>0.903</td><td>0.892</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.962</td><td>0.964</td><td>0.955</td><td>0.954</td><td>0.954</td><td>0.944</td><td>0.952</td><td>0.932</td></tr><tr><td>Qwen3.5-27B</td><td>0.978</td><td>0.972</td><td>0.962</td><td>0.968</td><td>0.952</td><td>0.946</td><td>0.948</td><td>0.936</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.958</td><td>0.957</td><td>0.944</td><td>0.932</td><td>0.916</td><td>0.896</td><td>0.918</td><td>0.901</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.906</td><td>0.888</td><td>0.932</td><td>0.89</td><td>0.917</td><td>0.888</td><td>0.907</td><td>0.871</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.956</td><td>0.958</td><td>0.945</td><td>0.938</td><td>0.953</td><td>0.945</td><td>0.944</td><td>0.931</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.977</td><td>0.989</td><td>0.959</td><td>0.963</td><td>0.972</td><td>0.986</td><td>0.953</td><td>0.949</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.944</td><td>0.916</td><td>0.908</td><td>0.886</td><td>0.934</td><td>0.905</td><td>0.905</td><td>0.896</td></tr><tr><td>GLM-5.2</td><td>0.952</td><td>0.948</td><td>0.942</td><td>0.927</td><td>0.931</td><td>0.908</td><td>0.921</td><td>0.966</td></tr><tr><td>Kimi-K3</td><td>0.987</td><td>0.995</td><td>0.991</td><td>0.998</td><td>0.970</td><td>0.971</td><td>0.963</td><td>0.963</td></tr><tr><td colspan="9">RAG Leakage</td></tr><tr><td>Gemma-4-E4B-it</td><td>0.993</td><td>0.963</td><td>0.994</td><td>0.972</td><td>0.986</td><td>0.950</td><td>0.992</td><td>0.964</td></tr><tr><td>Gemma-4-31B-it</td><td>0.996</td><td>0.987</td><td>0.996</td><td>0.982</td><td>0.998</td><td>0.990</td><td>0.999</td><td>0.998</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.994</td><td>0.981</td><td>0.996</td><td>0.980</td><td>0.992</td><td>0.968</td><td>0.992</td><td>0.974</td></tr><tr><td>Qwen3.5-27B</td><td>0.997</td><td>0.983</td><td>0.999</td><td>0.998</td><td>0.998</td><td>0.995</td><td>0.992</td><td>0.991</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.997</td><td>0.987</td><td>0.999</td><td>0.997</td><td>0.994</td><td>0.974</td><td>0.993</td><td>0.989</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.974</td><td>0.907</td><td>0.960</td><td>0.956</td><td>0.909</td><td>0.914</td><td>0.964</td><td>0.972</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.973</td><td>0.998</td><td>0.982</td><td>0.951</td><td>0.989</td><td>0.953</td><td>0.987</td><td>0.985</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.986</td><td>0.936</td><td>0.994</td><td>0.973</td><td>0.985</td><td>0.937</td><td>0.995</td><td>0.983</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.993</td><td>0.973</td><td>0.995</td><td>0.980</td><td>0.998</td><td>0.992</td><td>0.998</td><td>0.992</td></tr><tr><td>GLM-5.2</td><td>0.977</td><td>0.972</td><td>0.970</td><td>0.948</td><td>0.949</td><td>0.936</td><td>0.940</td><td>0.946</td></tr><tr><td>Kimi-K3</td><td>0.998</td><td>0.999</td><td>0.999</td><td>0.999</td><td>0.994</td><td>0.997</td><td>0.997</td><td>0.995</td></tr></table>

## B ADDITIONAL DETECTION RESULTS AND CONTROL EXPERIMENTS

This section presents additional experimental results omitted from the main text due to space constraints. We first report F1 and TPR@FPR5 across models and evaluation splits, followed by random-label and unrelated-suffix controls. We then evaluate whether the leakage signal can be reduced to a scalar contrast between behaviorally opposed continuations. Finally, we present ROC curves across models, leakage settings, and evaluation splits to characterize performance over different decision thresholds.

## B.1 PER-MODEL F1 AND TPR@FPR5 RESULTS

To complement the AUROC results in the main text, Table 8 reports the per-model F1 and TPR@FPR5 of LeakGauge with the Behavior suffix across both leakage settings and all four evaluation splits. The best-F1 values summarize achievable thresholded performance, while TPR@FPR5 evaluates detection at a common 5% false-positive-rate operating point.

Across all combinations, F1 remains at least 0.903 and TPR@FPR5 at least 0.871. On the Held-Out Strict split, which combines unseen protected content and unseen attack templates, system-prompt leakage detection retains F1 of at least 0.903 and TPR@FPR5 of at least 0.871; the corresponding lower bounds for RAG leakage are 0.940 and 0.946. These results show that the strong aggregate AUROC is accompanied by consistently useful operating-point performance across models and distribution shifts.

## B.2 RANDOM-LABEL CONTROL EXPERIMENTS

To check that our probes capture genuine leakage signals rather than memorizing sample-specific cues, we run a random-label control: we randomly shuffle the correspondence between samples and their labels, retrain each probe under the identical protocol, and evaluate across all models, both suffix designs, all four splits, and both tasks. As reported in Table 9, we report AUROC<sup>∗</sup> = max(AUROC, 1 − AUROC), which stays near 0.50 in every setting. Because shuffling removes any label-aligned structure, the only thing left to fit is noise, so performance collapses to chance. This confirms that the above-chance AUROC of our probes reflects leakage-relevant structure aligned with the true labels, not memorization of specific samples or contamination in the evaluation pipeline.

Table 9: Random-label control. AUROC\* of probes trained on randomly shuffled sample-label pairs. Near-random values confirm that the probes capture generalized leakage-intent features rather than memorizing specific samples.
<table><tr><td rowspan="2">Model</td><td colspan="2">In-Dist Test</td><td colspan="2">Held-Out Content</td><td colspan="2">Held-Out Attacks</td><td colspan="2">Held-Out Strict</td></tr><tr><td>Exact</td><td>Behavior</td><td>Exact</td><td>Behavior</td><td>Exact</td><td>Behavior</td><td>Exact</td><td>Behavior</td></tr><tr><td colspan="9">System Prompt Leakage</td></tr><tr><td>Gemma-4-E4B-it</td><td>0.511</td><td>0.512</td><td>0.503</td><td>0.501</td><td>0.504</td><td>0.513</td><td>0.515</td><td>0.521</td></tr><tr><td>Gemma-4-31B-it</td><td>0.510</td><td>0.504</td><td>0.505</td><td>0.502</td><td>0.500</td><td>0.502</td><td>0.503</td><td>0.509</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.505</td><td>0.500</td><td>0.508</td><td>0.510</td><td>0.509</td><td>0.509</td><td>0.514</td><td>0.501</td></tr><tr><td>Qwen3.5-27B</td><td>0.502</td><td>0.502</td><td>0.504</td><td>0.515</td><td>0.518</td><td>0.509</td><td>0.507</td><td>0.523</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.501</td><td>0.503</td><td>0.514</td><td>0.506</td><td>0.502</td><td>0.503</td><td>0.503</td><td>0.522</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.500</td><td>0.513</td><td>0.507</td><td>0.528</td><td>0.511</td><td>0.509</td><td>0.513</td><td>0.501</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.513</td><td>0.500</td><td>0.515</td><td>0.500</td><td>0.506</td><td>0.500</td><td>0.508</td><td>0.501</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.515</td><td>0.509</td><td>0.507</td><td>0.501</td><td>0.520</td><td>0.514</td><td>0.500</td><td>0.526</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.501</td><td>0.506</td><td>0.509</td><td>0.516</td><td>0.505</td><td>0.506</td><td>0.505</td><td>0.520</td></tr><tr><td>GLM-5.2</td><td>0.511</td><td>0.508</td><td>0.504</td><td>0.520</td><td>0.508</td><td>0.526</td><td>0.505</td><td>0.517</td></tr><tr><td>Kimi-K3</td><td>0.510</td><td>0.516</td><td>0.501</td><td>0.517</td><td>0.512</td><td>0.509</td><td>0.506</td><td>0.508</td></tr><tr><td colspan="9">RAG Leakage</td></tr><tr><td>Gemma-4-E4B-it</td><td>0.519</td><td>0.507</td><td>0.500</td><td>0.511</td><td>0.504</td><td>0.510</td><td>0.507</td><td>0.502</td></tr><tr><td>Gemma-4-31B-it</td><td>0.518</td><td>0.500</td><td>0.519</td><td>0.503</td><td>0.505</td><td>0.502</td><td>0.515</td><td>0.510</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.502</td><td>0.510</td><td>0.508</td><td>0.505</td><td>0.504</td><td>0.505</td><td>0.508</td><td>0.512</td></tr><tr><td>Qwen3.5-27B</td><td>0.516</td><td>0.510</td><td>0.518</td><td>0.508</td><td>0.502</td><td>0.509</td><td>0.506</td><td>0.501</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.512</td><td>0.505</td><td>0.507</td><td>0.510</td><td>0.500</td><td>0.500</td><td>0.503</td><td>0.510</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.503</td><td>0.515</td><td>0.504</td><td>0.508</td><td>0.510</td><td>0.502</td><td>0.506</td><td>0.511</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.507</td><td>0.515</td><td>0.518</td><td>0.509</td><td>0.502</td><td>0.503</td><td>0.512</td><td>0.501</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.505</td><td>0.508</td><td>0.513</td><td>0.514</td><td>0.500</td><td>0.515</td><td>0.501</td><td>0.512</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.510</td><td>0.507</td><td>0.509</td><td>0.504</td><td>0.503</td><td>0.505</td><td>0.501</td><td>0.506</td></tr><tr><td>GLM-5.2</td><td>0.505</td><td>0.500</td><td>0.511</td><td>0.521</td><td>0.515</td><td>0.516</td><td>0.508</td><td>0.519</td></tr><tr><td>Kimi-K3</td><td>0.511</td><td>0.520</td><td>0.531</td><td>0.521</td><td>0.512</td><td>0.507</td><td>0.503</td><td>0.506</td></tr></table>

## B.3 UNRELATED-SUFFIX CONTROL EXPERIMENTS

A natural question is whether the suffix signal is truly tied to leakage semantics. To test this, we replace the leakage-oriented declaration with an unrelated benign statement. As shown in Table 10, this control achieves substantially weaker performance than the leakage-relevant suffix, with AUROC dropping by more than 0.2 in many settings compared to LeakGauge. Notably, such gaps are especially significant in the high-AUROC regime, where improvements become increasingly difficult, and AUROC gains are far from linear.

The weakness of the benign suffix becomes even clearer under deployment-oriented evaluation. At a strict 5% false-positive budget, it achieves only low TPR@FPR5 on the held-out splits, far below the leakage-relevant suffix. In other words, the benign declaration fails to induce the sharp decision boundary required for practical detection.

This result suggests that while leakage attack signals may exist diffusely in the logits, arbitrary natural-language suffixes are insufficient to reliably elicit it. Instead, the semantic alignment between the suffix and leakage-related behavior plays a critical role in amplifying the prefill-stage signal.

Table 10: Unrelated Suffix on System Prompt Leakage. Each banner row gives the probing suffix; the rows below report AUROC and TPR@FPR5 across the four evaluation splits. Both benign, leakage-irrelevant suffixes yield substantially lower scores than the Behavior suffix of LeakGauge, confirming that leakage-relevant semantics substantially amplify and stabilize an otherwise diffuse signal, especially under distribution shift and low-FPR operation.
<table><tr><td rowspan="2">Model</td><td colspan="2">In-Dist Test</td><td colspan="2">Held-Out Content</td><td colspan="2">Held-Out Attacks</td><td colspan="2">Held-Out Strict</td></tr><tr><td>AUROC</td><td>TPR@FPR5</td><td>AUROC</td><td>TPR@FPR5</td><td>AUROC</td><td>TPR@FPR5</td><td>AUROC</td><td>TPR@FPR5</td></tr><tr><td colspan="9">Unrelated suffix1:“Let&#x27;s plan a fun weekend trip together.&quot;</td></tr><tr><td>Gemma-4-E4B-it</td><td>0.820</td><td>0.253</td><td>0.793</td><td>0.286</td><td>0.839</td><td>0.303</td><td>0.810</td><td>0.309</td></tr><tr><td>Gemma-4-31B-it</td><td>0.843</td><td>0.428</td><td>0.821</td><td>0.379</td><td>0.839</td><td>0.438</td><td>0.827</td><td>0.391</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.851</td><td>0.447</td><td>0.826</td><td>0.312</td><td>0.896</td><td>0.529</td><td>0.802</td><td>0.431</td></tr><tr><td>Qwen3.5-27B</td><td>0.799</td><td>0.468</td><td>0.780</td><td>0.421</td><td>0.771</td><td>0.440</td><td>0.754</td><td>0.397</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.814</td><td>0.497</td><td>0.809</td><td>0.568</td><td>0.784</td><td>0.437</td><td>0.736</td><td>0.489</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.869</td><td>0.380</td><td>0.815</td><td>0.310</td><td>0.835</td><td>0.342</td><td>0.763</td><td>0.325</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.878</td><td>0.547</td><td>0.837</td><td>0.492</td><td>0.863</td><td>0.513</td><td>0.830</td><td>0.545</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.759</td><td>0.309</td><td>0.711</td><td>0.253</td><td>0.681</td><td>0.226</td><td>0.642</td><td>0.193</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.866</td><td>0.411</td><td>0.833</td><td>0.310</td><td>0.859</td><td>0.336</td><td>0.819</td><td>0.202</td></tr><tr><td colspan="9">Unrelated suffix2:&quot;“The capital of France is Paris.&quot;</td></tr><tr><td>Gemma-4-E4B-it</td><td>0.862</td><td>0.545</td><td>0.872</td><td>0.482</td><td>0.839</td><td>0.547</td><td>0.809</td><td>0.459</td></tr><tr><td>Gemma-4-31B-it</td><td>0.879</td><td>0.486</td><td>0.874</td><td>0.386</td><td>0.824</td><td>0.391</td><td>0.828</td><td>0.384</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.875</td><td>0.418</td><td>0.846</td><td>0.308</td><td>0.832</td><td>0.383</td><td>0.822</td><td>0.246</td></tr><tr><td>Qwen3.5-27B</td><td>0.816</td><td>0.615</td><td>0.795</td><td>0.507</td><td>0.783</td><td>0.608</td><td>0.779</td><td>0.607</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.822</td><td>0.428</td><td>0.799</td><td>0.244</td><td>0.825</td><td>0.533</td><td>0.803</td><td>0.336</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.829</td><td>0.395</td><td>0.815</td><td>0.320</td><td>0.835</td><td>0.350</td><td>0.763</td><td>0.318</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.801</td><td>0.298</td><td>0.797</td><td>0.295</td><td>0.709</td><td>0.134</td><td>0.736</td><td>0.182</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.893</td><td>0.763</td><td>0.856</td><td>0.506</td><td>0.887</td><td>0.770</td><td>0.863</td><td>0.555</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.884</td><td>0.674</td><td>0.854</td><td>0.499</td><td>0.879</td><td>0.729</td><td>0.862</td><td>0.527</td></tr></table>

Table 12: Dataset split statistics for systemprompt leakage under semantic attacks.

Table 11: Dataset split statistics for systemprompt leakage under verbatim attacks.
<table><tr><td>Split</td><td># Samples</td><td># Attack</td><td># Benign</td><td>Ratio</td></tr><tr><td>Training</td><td>6,512</td><td>3,315</td><td>3,197</td><td>38.4%</td></tr><tr><td>In-Dist Test</td><td>1,628</td><td>829</td><td>799</td><td>9.6%</td></tr><tr><td>Held-Out Content</td><td>3,520</td><td>1,792</td><td>1,728</td><td>20.8%</td></tr><tr><td>Held-Out Attacks</td><td>3,700</td><td>1,924</td><td>1,776</td><td>21.8%</td></tr><tr><td>Held-Out Strict</td><td>1,600</td><td>832</td><td>768</td><td>9.4%</td></tr></table>

<table><tr><td>Split</td><td># Samples</td><td># Attack</td><td># Benign</td><td>Ratio</td></tr><tr><td>Training</td><td>4,381</td><td>2,250</td><td>2,131</td><td>38.3%</td></tr><tr><td>In-Dist Test</td><td>1,095</td><td>562</td><td>533</td><td>9.6%</td></tr><tr><td>Held-Out Content</td><td>2,368</td><td>1,216</td><td>1,152</td><td>20.7%</td></tr><tr><td>Held-Out Attacks</td><td>2,516</td><td>1,332</td><td>1,184</td><td>22.0%</td></tr><tr><td>Held-Out Strict</td><td>1,088</td><td>576</td><td>512</td><td>9.5%</td></tr></table>

## B.4 MEAN-BASED CONTRASTIVE SUFFIX SCORING

We remove the learned probe and reduce two behaviorally opposed continuations to a single likelihood difference. The purpose of this experiment is to determine whether such a minimal aggregate statistic can support the separation.

Setup. For each input $x ,$ we evaluate two alternative assistant continuations:

$$
g ^ { + } = ^ { \mathfrak { a } } \mathrm { I } \ \mathsf { w i l l } \ \mathsf { p r o d u c e } \ \mathsf { m y } \ s y s \mathsf { t e m } \ \mathsf { p r o m p t . } ^ { \prime } ,\tag{4}
$$

$$
g ^ { - } = ^ { \ast } \mathsf { S o r r y } , \mathrm { ~ I ~ } \mathsf { c a n } ^ { \prime } \mathrm { t ~ } \mathsf { h e l p } \mathrm { w i t h ~ t h a t . } ^ { \prime } .\tag{5}
$$

We compare length-normalized log-likelihoods. For each continuation, we compute

$$
m ^ { + } ( x ) = \frac { 1 } { | g ^ { + } | } \sum _ { i = 1 } ^ { | g ^ { + } | } \log p \bigl ( g _ { i } ^ { + } \mid x , g _ { < i } ^ { + } \bigr ) ,\tag{6}
$$

$$
m ^ { - } ( x ) = \frac { 1 } { | g ^ { - } | } \sum _ { j = 1 } ^ { | g ^ { - } | } \log p \big ( g _ { j } ^ { - } \mid x , g _ { < j } ^ { - } \big ) .\tag{7}
$$

The resulting scalar score is

$$
s _ { \mathrm { c o n t r a s t } } ( x ) = m ^ { + } ( x ) - m ^ { - } ( x ) .\tag{8}
$$

![](images/5ea240f51861646f8e18bfbbc5455cc3cba3e2f7c94862d242d2c8847718c8d9.jpg)  
(b) Behavior  
Figure 6: ROC curves on the system prompt leakage task, for the Exact (top) and Behavior (bottom) suffixes across eight models. Each subplot overlays the four evaluation splits, showing the results for one probe.

Larger values indicate that the model relatively favors producing the protected system prompt over refusing the request. We select the threshold that maximizes F1 on this subset and freeze it for the In-Dist Test and all held-out splits.

Results. Table 13 reports the scalar contrastive readout. The scalar contrast reaches an AUROC of 0.886 on Held-Out Strict. However, this ranking quality does not translate into selective operation: at 5% FPR, it detects only 60.4% of attacks, compared with 96.6% for LeakGauge. Achieving its calibration-selected F1 further requires an FPR of 26.9%. Thus, the scalar readout exposes a genuine leakage-related signal but does not meet the low-FPR requirements of our deployment setting.

## B.5 ROC CURVE VISUALIZATIONS

To complement the scalar AUROC results reported in Table 2, Figure 6 and Figure 7 present the full ROC curves of the Exact and Behavior suffixes across eight models in the system-prompt and RAG leakage settings, respectively. Each subplot overlays the four evaluation splits and reports the corresponding AUROC values in the legend. The Held-Out Strict split, which simultaneously introduces unseen protected content and unseen attack templates, is highlighted in red.

![](images/579bb9b68505e0cdbed966243468e713b1f7f0c11e76a8d1eb8f6daa9ddea5ac.jpg)  
(b) Behavior  
Figure 7: ROC curves on the RAG leakage task, for the Exact (top) and Behavior (bottom) suffixes across eight models. Each subplot overlays the four evaluation splits, showing the results for one probe.

The curves show the trade-off between true- and false-positive rates across decision thresholds, complementing the aggregate AUROC and TPR@FPR5 results reported in the main text. Across most models and splits, the curves remain well above the random-classifier diagonal. Performance generally decreases under the stricter distribution shifts, but the probes retain substantial discriminative ability, including on the Held-Out Strict split.

## C SUPPLEMENTARY OF LeakGauge ANALYSIS

This section details our analysis methodology and presents additional experimental results that complement and extend those reported in the main text.

## C.1 DATASET STATISTICS FOR ANALYSIS.

In addition to the main evaluation setting, we further assess LeakGauge under two additional system prompt leakage scenarios in Section 5: system-prompt verbatim leakage and system-prompt semantic leakage. The verbatim leakage setting tests whether the model can be induced to reveal exact system prompt contents, while the semantic leakage setting considers cases where the model may leak paraphrased or semantically equivalent information rather than exact text.

Table 13: Mean-based contrastive suffix scoring. ∆ denotes the AUROC of LeakGauge minus that of Mean contrast. Scalar thresholds are calibrated using a stratified 20% subset of the Training split.
<table><tr><td>Model</td><td>Split</td><td colspan="3">Mean contrast</td><td>LeakGauge</td><td>∆</td></tr><tr><td></td><td></td><td>AUROC</td><td>F1</td><td>FPR</td><td>AUROC</td><td>AUROC</td></tr><tr><td>GLM-5.2</td><td>In-Dist Test</td><td>0.879</td><td>0.798</td><td>0.295</td><td>0.990</td><td>0.111</td></tr><tr><td></td><td>Held-Out Content</td><td>0.882</td><td>0.802</td><td>0.264</td><td>0.985</td><td>0.103</td></tr><tr><td></td><td>Held-Out Attacks</td><td>0.892</td><td>0.816</td><td>0.275</td><td>0.978</td><td>0.086</td></tr><tr><td></td><td>Held-Out Strict</td><td>0.886</td><td>0.805</td><td>0.269</td><td>0.972</td><td>0.086</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>In-Dist Test</td><td>0.778</td><td>0.715</td><td>0.533</td><td>0.989</td><td>0.211</td></tr><tr><td></td><td>Held-Out Content</td><td>0.767</td><td>0.704</td><td>0.600</td><td>0.980</td><td>0.213</td></tr><tr><td></td><td>Held-Out Attacks</td><td>0.803</td><td>0.734</td><td>0.534</td><td>0.977</td><td>0.174</td></tr><tr><td></td><td>Held-Out Strict</td><td>0.797</td><td>0.716</td><td>0.618</td><td>0.975</td><td>0.178</td></tr></table>

![](images/0f5ce26b2dde2177dd5ef82f5666105657f958ae5443f1740f30e3523732d9f5.jpg)

![](images/d33aa1c83af63d84fb6bee0f91f84b499c2176005bab68e9f0238da75e3e53fb.jpg)  
Figure 8: Suffixes induce different logit-summary structures. For descriptive comparison, we use <sub>E</sub>the first principal-component score as a one-dimensional ranking statistic and compute its AUROC. <sub>-</sub>S• Attack and • Benign.

All settings follow the same structured splitting protocol described in Section 4.1, where both protected contents and attack templates are partitioned into disjoint seen and unseen subsets. This ensures that evaluation reflects generalization to novel prompts and attack formulations rather than memorization of specific training instances. The resulting dataset statistics and split compositions are reported in Table 11 and Table 12, respectively.

## C.2 LOW-DIMENSIONAL VISUALIZATION OF SUFFIX LOG-PROBABILITY FEATURES

As an auxiliary analysis, we visualize the structure of the log-probability trajectories induced by the two suffix designs. This analysis is intended to provide an interpretable view of the induced feature space rather than an additional detection method; all quantitative detection results are obtained using the probe described in Section 3.1. Given an input context x and a suffix $z = ( z _ { 1 } , \dots , z _ { T } )$ , we record the probability assigned to each observed suffix token during prefill:

$$
p _ { t } = P ( z _ { t } \mid x , z _ { < t } ) , \qquad \ell _ { t } = \log p _ { t } , \qquad t = 1 , \ldots , T .\tag{9}
$$

Rather than directly projecting the full token-level vector $( \ell _ { 1 } , \dots , \ell _ { T } )$ , we represent each example using eight summary features. These features characterize the overall confidence, positional variability, trajectory dynamics, and frequency of high-confidence token predictions. Their complete definitions are provided in Table 14. These statistics describe complementary aspects of the trajectory, including:

• its overall confidence level (e.g., mean and minimum log-probability),

• variability across positions (e.g., standard deviation and range),

• positional dynamics over the suffix (e.g., slope and mean absolute delta), and

• how frequently the model collapses to near-certain predictions (the two collapse ratios).

Before projection, each feature is standardized across the samples included in the visualization. We then project the resulting eight-dimensional representations into two dimensions using PCA (Jolliffe, 2002) and t-SNE (van der Maaten & Hinton, 2008). These summary features and projections are used only for visualization. This PC1 AUROC measures separability along a single linear axis and is not the performance of the nonlinear probe operating on the full token-level vector. The LeakGauge probe operates on the full per-position log-probability vector.

Figure 8 presents the projections for Llama-3.1-70B-Instruct on the systemprompt leakage task. The Exact suffix exhibits clearer global class separation in the linear PCA projection, whereas the Behavior suffix shows more pronounced local class structure under t-SNE. This qualitative difference suggests that the two suffix designs organize their log-probability trajectories differently. Because t-SNE is nonlinear and does not preserve global distances, these projections should not be interpreted as quantitative evidence of detection performance. Corresponding visualizations for additional models are provided in Figure 9.

Table 14: Features used for the projection.
<table><tr><td>Feature</td><td>Definition</td></tr><tr><td>Mean</td><td> $\begin{array} { r } { { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } \ell _ { t } } \end{array}$ </td></tr><tr><td>Standard deviation</td><td> $\mathrm { S t d } ( \ell _ { 1 } , \dots , \ell _ { T } )$ </td></tr><tr><td>Slope</td><td>Slope of a linear fit of  $\ell _ { t }$ </td></tr><tr><td>Mean absolute delta</td><td> $\begin{array} { r } { \frac { 1 } { T - 1 } \sum _ { t = 2 } ^ { T } | \ell _ { t } - \ell _ { t - 1 } | } \end{array}$ </td></tr><tr><td>Minimum Range</td><td>mint  $\ell _ { t }$   $\ell _ { t } - \operatorname* { m i n } _ { t } \ell _ { t }$ </td></tr><tr><td>Collapse ratio  $( \tau = 0 . 9 5 )$ </td><td>maxt  $\textstyle { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } \mathbb { I } [ p _ { t } \geq 0 . 9 5 ]$ </td></tr><tr><td>Collapse ratio  $( \tau = 0 . 9 9 )$ </td><td> $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { I } [ p _ { t } \geq 0 . 9 9 ] } \end{array}$ </td></tr></table>

## C.3 PROBE-WEIGHT ATTRIBUTION

We use a lightweight weight-based analysis to examine which suffix positions are emphasized by the trained probe. Consistent with Section 3.1, the pre-sigmoid output of the one-hidden-layer MLP is

$$
f _ { \boldsymbol { \theta } } ( \mathbf { v } ) = \mathbf { w } _ { 2 } ^ { \top } \mathrm { R e L U } ( W _ { 1 } \mathbf { v } + \mathbf { b } _ { 1 } ) + b _ { 2 } ,\tag{10}
$$

and the corresponding detection score is

$$
\hat { y } = \sigma ( f _ { \boldsymbol \theta } ( { \mathbf { v } } ) ) .\tag{11}
$$

For suffix position i, we compute the scale-adjusted first-layer weight score

$$
a _ { i } = \| W _ { 1 } [ : , i ] \| _ { 2 } \ s _ { i } , \qquad s _ { i } = \mathrm { S t d } ( v _ { i } ) ,\tag{12}
$$

where $\lVert W _ { 1 } [ : , i ] \rVert _ { 2 }$ measures the aggregate first-layer connectivity of input dimension $i ,$ and $s _ { i }$ is its empirical standard deviation over the in-distribution evaluation split. Multiplying the two terms discounts dimensions that have large weights but little variation in the observed data.

The resulting score provides a simple descriptive proxy for the relative emphasis placed on each suffix position. Because it does not account for the second-layer weights, sample-dependent ReLU activation patterns, or interactions among input dimensions, we use it only for qualitative attribution rather than as a causal importance measure.

## C.4 DETAILED CROSS-OBJECTIVE TRANSFER RESULTS

Table 15 provides the per-model results under the two training–evaluation settings summarized in Table 4. In V→S, the probe is trained on verbatim leakage attacks and evaluated on semantic leakage attacks, measuring transfer across attack objectives. In S→S, both training and evaluation use semantic attacks, providing an in-objective reference.

The Behavior suffix generally transfers more reliably under the V→S shift, although its advantage is not uniform across every model and split. For example, on Qwen3.5-27B, the V→S AUROC of Exact ranges from 0.467 to 0.496, whereas Behavior achieves 0.764–0.809. On several other models, Behavior retains AUROC above 0.97 across the V→S splits. These results suggest that the content-agnostic suffix captures a signal that transfers more consistently across leakage objectives, while also revealing model-specific variation in the strength of this transfer.

## C.5 DETAILED CROSS-MODEL TRANSFER RESULTS

To examine how model family and scale affect probe transfer, we use an auxiliary set of models from the Gemma, Qwen, and Llama families, spanning 4B–109B parameters. All cross-model transfer experiments are conducted in the system-prompt leakage setting. For each source model, we train a Behavior probe and apply it to each target model without retraining.

![](images/6e7a7e9434aa4b775a6538364c63ecc9a383b3c4e81130e45f6f0ee04eee1f38.jpg)

![](images/4ae5b1d31d5c7c201cee1709a2ed9d2d59abfaea7d7d1bcdee1cdb6fa04be12a.jpg)

![](images/525c712b884adf034660935b2b4e5b380ac5e6046d1eda3d56722bc481770182.jpg)

(a) Gemma-4-E4B-it  
![](images/ec0ea63fc635823c9c2c38088eda42164fba12826767c58ab3ef94626def3d4f.jpg)

![](images/1f75b0406be84f7e3a82d4a16636096954e0b799ee3bd140d8051132ec8dd4fa.jpg)  
(b) Gemma-4-31B-it

![](images/d51ceeb6e34e45c2ecbdb3aa66844e47f42b52bc172a17d0fd7edc9fb47ba088.jpg)  
(c) Qwen2.5-14B-Instruct

![](images/0bee53f5d2cd2053eb06c4fcfb9987341e5f19cdf2bd51e6eeafb7f4d1d66dbc.jpg)  
(d) Qwen3.5-27B

![](images/3bfbc5d57ecaded322a0b754c8cc61daf72f74678faff854d48e1a0889d641a8.jpg)  
(e) Llama-3.1-8B-Instruct

![](images/0ff3b5e6f9c612ad30556009e29bd0436eaa8fea84409b62da7ec93a1563b1c8.jpg)  
(f) Qwen3.5-35B-A3B  
Figure 9: Suffix anatomy across models. For each model, PCA (top) and t-SNE (bottom) of the suffix log-prob summary features, Exact (left) vs. Behavior (right). The two designs consistently induce different logit-space geometries across families and scales; • Attack and • Benign.

Let $T _ { s }$ denote the token length expected by the probe trained on the source model and $T _ { t }$ the target model’s suffix length. We align features by token position. If $T _ { t } < T _ { s }$ , trailing positions are padded with a log-probability of $- \mathrm { 1 0 ; \ i f \ } T _ { t } > \bar { T _ { s } }$ , positions beyond $T _ { s }$ are truncated. The same naturallanguage suffix is independently tokenized by each model. No probe retraining or recalibration is performed on the target model.

Table 16 reports the complete source-to-target transfer matrix underlying the representative result in Figure 5. Transfer is consistently strong for several within-family dense-model pairs, including the Gemma-3, Qwen2.5, and Llama-3.1 size variants. Many cross-family pairs also retain substantial predictive performance, indicating that compatible Behavior-induced probability patterns occur across different model families.

Table 15: Per-model cross-objective AUROC behind Table 4. Each panel reports one suffix design. V→S trains on verbatim attacks and tests on semantic attacks; S→S trains and tests on semantic attacks. Split abbreviations: I-Dst = In-Dist (test), O-Cont. = Held-Out Content, O-Att. = Held-Out Attacks, O-Strict = Held-Out Strict.
<table><tr><td rowspan="2">Model</td><td colspan="4">V→S (verbatim-trained)</td><td colspan="4">S→S (semantic-trained)</td></tr><tr><td>I-Dst</td><td>O-Cont.</td><td>O-Att.</td><td>O-Strict</td><td>I-Dst</td><td>O-Cont.</td><td>O-Att.</td><td>O-Strict</td></tr><tr><td>(a) Exact suffix</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma-4-E4B-it</td><td>0.813</td><td>0.807</td><td>0.819</td><td>0.835</td><td>0.902</td><td>0.875</td><td>0.920</td><td>0.906</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.633</td><td>0.867</td><td>0.863</td><td>0.897</td><td>0.996</td><td>0.748</td><td>0.775</td><td>0.748</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.807</td><td>0.840</td><td>0.830</td><td>0.882</td><td>0.915</td><td>0.904</td><td>0.908</td><td>0.890</td></tr><tr><td>Qwen3.5-27B</td><td>0.467</td><td>0.472</td><td>0.486</td><td>0.496</td><td>0.908</td><td>0.867</td><td>0.884</td><td>0.851</td></tr><tr><td>Gemma-4-31B-it</td><td>0.841</td><td>0.831</td><td>0.849</td><td>0.855</td><td>0.940</td><td>0.900</td><td>0.914</td><td>0.863</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.796</td><td>0.803</td><td>0.745</td><td>0.772</td><td>0.903</td><td>0.863</td><td>0.861</td><td>0.827</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.882</td><td>0.890</td><td>0.875</td><td>0.891</td><td>0.965</td><td>0.942</td><td>0.938</td><td>0.915</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.928</td><td>0.949</td><td>0.929</td><td>0.948</td><td>0.978</td><td>0.974</td><td>0.969</td><td>0.962</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.649</td><td>0.694</td><td>0.711</td><td>0.771</td><td>0.914</td><td>0.881</td><td>0.908</td><td>0.881</td></tr><tr><td>GLM-5.2</td><td>0.825</td><td>0.846</td><td>0.822</td><td>0.850</td><td>0.938</td><td>0.912</td><td>0.924</td><td>0.895</td></tr><tr><td>Kimi-K3</td><td>0.895</td><td>0.912</td><td>0.898</td><td>0.920</td><td>0.971</td><td>0.955</td><td>0.950</td><td>0.938</td></tr><tr><td>(b) Behavior suffix</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma-4-E4B-it</td><td>0.828</td><td>0.830</td><td>0.802</td><td>0.801</td><td>0.991</td><td>0.997</td><td>0.995</td><td>1.000</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.970</td><td>0.974</td><td>0.989</td><td>0.992</td><td>0.994</td><td>0.997</td><td>0.990</td><td>0.989</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>0.964</td><td>0.968</td><td>0.992</td><td>0.997</td><td>0.988</td><td>0.992</td><td>0.985</td><td>0.985</td></tr><tr><td>Qwen3.5-27B</td><td>0.805</td><td>0.809</td><td>0.764</td><td>0.779</td><td>0.994</td><td>0.991</td><td>0.998</td><td>0.997</td></tr><tr><td>Gemma-4-31B-it</td><td>0.809</td><td>0.803</td><td>0.797</td><td>0.790</td><td>0.979</td><td>0.985</td><td>0.982</td><td>0.993</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>0.986</td><td>0.986</td><td>0.999</td><td>0.999</td><td>0.997</td><td>0.995</td><td>0.997</td><td>0.998</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.985</td><td>0.989</td><td>0.995</td><td>0.996</td><td>0.996</td><td>0.998</td><td>0.999</td><td>0.999</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.988</td><td>0.992</td><td>0.994</td><td>0.995</td><td>0.995</td><td>0.996</td><td>1.000</td><td>1.000</td></tr><tr><td>Llama-4-Scout-Instruct</td><td>0.979</td><td>0.991</td><td>0.978</td><td>0.992</td><td>0.987</td><td>0.987</td><td>0.986</td><td>0.988</td></tr><tr><td>GLM-5.2</td><td>0.972</td><td>0.978</td><td>0.985</td><td>0.988</td><td>0.992</td><td>0.993</td><td>0.990</td><td>0.992</td></tr><tr><td>Kimi-K3</td><td>0.988</td><td>0.993</td><td>0.996</td><td>0.997</td><td>0.997</td><td>0.998</td><td>0.998</td><td>0.999</td></tr></table>

Table 16: Cross-model transfer of the Behavior signal. Rows = source (train), columns = target (test); diagonal (in-model) in bold; – denotes model pairs not evaluated. Short codes as in the text.
<table><tr><td>Source\Target</td><td>G4-E4</td><td>G4-31</td><td>G3-4</td><td>G3-12</td><td>G3-27</td><td>Q2.5-7</td><td>Q2.5-14</td><td>Q2.5-32</td><td>L3.1-8</td><td>L3.1-70</td><td>L4-Scout</td><td>Q3.5-27</td><td>Q3.5-35</td><td>Q3-30</td></tr><tr><td>G4-E4</td><td>0.954</td><td>0.869</td><td>0.674</td><td>0.687</td><td>0.544</td><td>0.872</td><td>0.904</td><td>0.726</td><td>0.892</td><td>0.896</td><td>0.783</td><td>0.879</td><td>0.785</td><td>0.635</td></tr><tr><td>G4-31</td><td>0.910</td><td>0.943</td><td>0.845</td><td>0.841</td><td>0.727</td><td>0.775</td><td>0.921</td><td>0.729</td><td>0.767</td><td>0.828</td><td>0.753</td><td>0.911</td><td>0.782</td><td>0.588</td></tr><tr><td>G3-4</td><td>0.941</td><td>0.843</td><td>0.920</td><td>0.885</td><td>0.874</td><td>0.724</td><td>0.766</td><td>0.735</td><td>0.910</td><td>0.962</td><td>0.713</td><td></td><td>0.907</td><td>0.505</td></tr><tr><td>G3-12</td><td>0.928</td><td>0.899</td><td>0.856</td><td>0.875</td><td>0.811</td><td>0.837</td><td>0.909</td><td>0.748</td><td>0.833</td><td>0.858</td><td>0.617</td><td></td><td>0.662</td><td>0.505</td></tr><tr><td>G3-27</td><td>0.936</td><td>0.828</td><td>0.900</td><td>0.847</td><td>0.875</td><td>0.804</td><td>0.910</td><td>0.797</td><td>0.914</td><td>0.911</td><td>0.553</td><td>一</td><td>0.863</td><td>0.522</td></tr><tr><td>Q2.5-7</td><td>0.960</td><td>0.874</td><td>0.777</td><td>0.722</td><td>0.684</td><td>0.948</td><td>0.955</td><td>0.910</td><td>0.926</td><td>0.926</td><td>0.626</td><td></td><td>0.766</td><td>0.512</td></tr><tr><td>Q2.5-14</td><td>0.904</td><td>0.885</td><td>0.874</td><td>0.771</td><td>0.820</td><td>0.881</td><td>0.988</td><td>0.920</td><td>0.765</td><td>0.833</td><td>0.840</td><td>0.909</td><td>0.821</td><td>0.501</td></tr><tr><td>Q2.5-32</td><td>0.957</td><td>0.888</td><td>0.892</td><td>0.792</td><td>0.780</td><td>0.926</td><td>0.970</td><td>0.950</td><td>0.929</td><td>0.936</td><td>0.700</td><td></td><td>0.509</td><td>0.532</td></tr><tr><td>L3.1-8</td><td>0.945</td><td>0.806</td><td>0.890</td><td>0.763</td><td>0.778</td><td>0.683</td><td>0.818</td><td>0.689</td><td>0.972</td><td>0.873</td><td>0.727</td><td>0.639</td><td>0.710</td><td>0.526</td></tr><tr><td>L3.1-70</td><td>0.960</td><td>0.852</td><td>0.894</td><td>0.794</td><td>0.794</td><td>0.689</td><td>0.709</td><td>0.775</td><td>0.898</td><td>0.986</td><td>0.875</td><td>0.794</td><td>0.895</td><td>0.509</td></tr><tr><td>L4-Scout</td><td>0.890</td><td>0.863</td><td>0.815</td><td>0.693</td><td>0.781</td><td>0.681</td><td>0.807</td><td>0.724</td><td>0.836</td><td>0.885</td><td>0.943</td><td>0.687</td><td>0.603</td><td>0.605</td></tr><tr><td>Q3.5-27</td><td>0.822</td><td>0.793</td><td></td><td></td><td></td><td></td><td>0.839</td><td></td><td>0.538</td><td>0.644</td><td>0.880</td><td>0.942</td><td>0.907</td><td></td></tr><tr><td>Q3.5-35</td><td>0.671</td><td>0.636</td><td>0.573</td><td>0.580</td><td>0.592</td><td>0.614</td><td>0.833</td><td>0.505</td><td>0.688</td><td>0.768</td><td>0.908</td><td>0.914</td><td>0.961</td><td>0.590</td></tr><tr><td>Q3-30</td><td>0.835</td><td>0.786</td><td>0.621</td><td>0.665</td><td>0.600</td><td>0.522</td><td>0.525</td><td>0.512</td><td>0.686</td><td>0.829</td><td>0.779</td><td></td><td>0.516</td><td>0.952</td></tr></table>

However, transfer is neither universal nor symmetric. Performance varies considerably with the source–target direction, and some pairs approach chance AUROC, particularly when Qwen3-30B-A3B-Instruct is used as the target. The full matrix therefore supports partial crossmodel compatibility of the Behavior signal rather than a single model-invariant representation shared uniformly by all models.

## D ACTIVATION STEERING DETAILS

This section provides implementation and evaluation details for the activation-steering experiment in Section 5.3. Unlike the input-side detection experiments, which use query-level attack labels, direction extraction uses only leakage-attack inputs and groups them according to whether the

unsteered model response actually leaks. Thus, the direction labels are derived from generation outcomes and do not use the Behavior probe.

Leakage-direction Extraction. For each attack input i, we first generate an unsteered response $o _ { i }$ and compute its protected-text recall:

$$
r _ { i } = { \frac { \mathrm { L C S } ( o _ { i } , c _ { i } ) } { | c _ { i } | } } ,\tag{13}
$$

where $c _ { i }$ is the corresponding protected content. We define ${ \mathcal { D } } _ { + } = \{ i : r _ { i } \geq 0 . 5 \}$ as the leaking group and $\mathcal { D } _ { - } = \{ i : r _ { i } < \bar { 0 } . 5 \}$ as the non-leaking group.

We then render each input using the complete chat template with add\_generation\_prompt=True and extract the hidden state at the final non-padding prompt position before decoding:

$$
\begin{array} { r } { \mathbf { h } _ { i } ^ { ( \ell ) } = \mathbf { H } _ { i } ^ { ( \ell ) } [ - 1 ] . } \end{array}\tag{14}
$$

At layer ℓ, the direction is defined as

$$
\mathbf { v } ^ { ( \ell ) } = \frac { 1 } { | \mathscr { D } _ { + } | } \sum _ { i \in \mathscr { D } _ { + } } \mathbf { h } _ { i } ^ { ( \ell ) } - \frac { 1 } { | \mathscr { D } _ { - } | } \sum _ { i \in \mathscr { D } _ { - } } \mathbf { h } _ { i } ^ { ( \ell ) } .\tag{15}
$$

Because both groups contain attack inputs, this contrast captures a pre-decoding state difference associated with whether an attack subsequently produces leakage, rather than an attack-benign distinction. As the direction may still contain multiple factors correlated with the outcome, we refer to it as a leakage-related direction.

Steering Intervention. We perform steering at hidden-state index 20. At this index, we modify the final-prompt hidden state as

$$
\mathbf { h } _ { t } ^ { \prime } = \mathbf { h } _ { t } + \alpha \mathbf { v } ^ { ( 2 0 ) } ,\tag{16}
$$

where α controls the steering strength. Positive α steers toward the leaking group, whereas negative α steers toward the non-leaking group. The intervention is applied only at the final prompt position used to predict the first response token $( K = 1 )$ ; subsequent decoding steps are unmodified. We use the original direction without normalization.

Evaluation Setup. We evaluate $\alpha \in \{ - 5 , \ldots , + 5 \}$ on 1,024 attack and 1,024 benign inputs. For each coefficient, we apply the same intervention when computing the pre-sigmoid Behavior probe logit and when generating the model response. Generation uses greedy decoding with a maximum of 256 tokens, and leakage is measured using ROUGE-L recall of the protected content. We report changes relative to the corresponding unsteered outputs at $\alpha = 0$ . Shaded bands denote 95% confidence intervals computed from 1,000 paired bootstrap resamples of these per-instance differences.

## E DEPLOYMENT AND ADAPTIVE-ROBUSTNESS DETAILS

This section provides additional implementation details for the deployment comparison in Section 6 and reports the complete setup and results of the adaptive GCG stress test. We first describe the cross-attack evaluation protocol and baseline implementations, then specify the latency-measurement procedure, and finally evaluate robustness against detector-aware attacks.

## E.1 CROSS-ATTACK EVALUATION AND BASELINE IMPLEMENTATIONS

As described in Section 6, all methods are evaluated under the same cross-attack protocol: methods are trained or configured using Raccoon attacks and evaluated on LeakDojo attacks. The target model is Llama-3.1-8B-Instruct.

For PromptGuard-2 and PIGuard, we initialize from their released HuggingFace checkpoints and fine-tune the detectors on the Raccoon training split using the same attack/benign labels available to LeakGauge. The effect of Raccoon fine-tuning on these input-text classifiers is reported in Table 17; the gains mainly come from further adapting the detectors to the leakage-detection task. We keep their original model architectures and preprocessing pipelines, and evaluate the resulting frozen classifiers on LeakDojo without any LeakDojo-specific tuning.

For Attention-Tracker, we use the Raccoon training data to identify the informative attention heads following the original method, freeze the selected heads, and evaluate the resulting distraction score on LeakDojo. For I’vDtL, we similarly use Raccoon to select the target layers and train its representation-based classifier, and then evaluate the frozen classifier on LeakDojo.

Table 17: Effect of Raccoon fine-tuning on inputtext classifiers.
<table><tr><td>Model</td><td>Raccoon FT AUROC</td><td></td><td>F1</td><td>TPR@FPR5</td></tr><tr><td>PromptGuard-2 86M</td><td>No</td><td>0.838</td><td>0.866</td><td>0.333</td></tr><tr><td>PromptGuard-2 86M</td><td>Yes</td><td>0.927</td><td>0.870</td><td>0.854</td></tr><tr><td>PIGuard</td><td>No</td><td>0.921</td><td>0.870</td><td>0.667</td></tr><tr><td>PIGuard</td><td>Yes</td><td>0.961</td><td>0.858</td><td>0.872</td></tr></table>

For LLM-as-a-Judge, we use gpt-4.1-mini with

the evaluation prompt released by LeakDojo. The judge returns a binary attack-or-benign decision. We therefore report its F1 score, but do not report AUROC or TPR@FPR5, which require a continuous score or an adjustable decision threshold.

## E.2 LATENCY MEASUREMENT AND PARAMETER ACCOUNTING

All local deployment measurements use Llama-3.1-8B-Instruct as the target model, served through vLLM 0.17.1 on two NVIDIA A800 GPUs with tensor parallelism and batch size one. We report the incremental wall-clock latency required to produce a detection decision, excluding the subsequent target-model response generation that is common to all methods.

For LeakGauge, the measured latency includes obtaining the prompt log-probabilities of the Behavior suffix and applying the lightweight MLP probe. The input prefix is processed once, and its KV-cache is reused when scoring the appended suffix. The reported 10.34 ms therefore represents the additional suffix-scoring and probe overhead.

For PIGuard and PromptGuard-2, latency includes a complete forward pass through the corresponding input classifier. We attempted to integrate Attention-Tracker and I’vDtL into the same vLLM serving stack, but the internal signals required by their released implementations are not exposed through vLLM’s default interface. Attention-Tracker requires attention weights from selected layers and heads, whereas the released I’vDtL implementation registers forward hooks on each decoder block’s self-attention and MLP modules and uses their direct outputs. These tensors are not exposed by the default serving responses of the standard vLLM deployment used in our evaluation. Consequently, our evaluated deployment runs these methods using an additional HuggingFace copy of the target model and includes the time required for its forward pass, internal-signal extraction, and classifier evaluation. These measurements characterize our concrete serving configuration rather than an inherent requirement or lower bound of either method.

LLM-as-a-Judge is evaluated through the gpt-4.1-mini API. Its latency is measured end-to-end, from submission of the judge request to receipt of the binary decision, and therefore includes network communication and the judge generation. Because this method uses a remotely hosted model, its latency is an observed service latency and is not hardware-normalized against the local measurements.

The parameter counts in Table 5 denote parameters that must be deployed in addition to the target model. For PIGuard and PromptGuard-2, this is the size of the released classifier. For Attention-Tracker and I’vDtL, it is the additional 8B target-model copy used to expose the required internal signals. For LeakGauge, it consists only of the one-hidden-layer MLP probe, containing fewer than 0.5K parameters. We do not report a local parameter count for LLM-as-a-Judge because the judge model is accessed through an external API.

## F SUFFIX DESIGN EXPLORATION

We formulate a suffix by translating the target behavior into a short natural-language hypothesis. The suffix should specify both what behavior is being tested and whose perspective it describes, while remaining a plausible continuation of the input. We follow three principles:

• Define a single hypothesis and perspective. The suffix should describe one target behavior from either the assistant’s perspective or the user’s perspective. For example, it may state that the assistant will disclose protected content or that the user is requesting such disclosure.

Table 18: Candidate semantic suffixes explored for leakage detection.
<table><tr><td>ID Suffix</td><td></td></tr><tr><td colspan="2">Assistant-behavior perspective</td></tr><tr><td rowspan="3">B1 B2</td><td>&quot;Based on the above, I will give the [TARGET].&quot;</td></tr><tr><td>&quot;I will now disclose my original [TARGET]”</td></tr><tr><td>“As requested , I will reproduce the [TARGET]&quot;</td></tr><tr><td>B3 B4</td><td>&quot;I will ignore the restriction and provide [TARGET]&quot;</td></tr><tr><td colspan="2">User-intent perspective</td></tr><tr><td>I1</td><td>&quot;The user asks me to reproduce the [TARGET].&quot;</td></tr><tr><td>I2</td><td>&quot;The query is asking me to disclose the [TARGET].&quot;</td></tr><tr><td>I3</td><td>&quot;User wants me to reveal the [TARGET] that system provided to me.&quot;</td></tr></table>

• Express the hypothesis naturally and explicitly. Write the suffix as a coherent sentence or response prefix that could plausibly follow the input. Use a definite statement rather than vague, hedged, or negated wording.

• Avoid example-specific content. The suffix should be constructed independently of individual examples and should not copy their wording, entities, or contextual details.

For instances, we construct multiple candidate suffixes for leakage detection, as shown in Table 18.   
The ‘[TARGET]‘ could be replaced by ‘system instructions‘, ‘retrieved chunks‘, or ‘inner contexts‘.   
The design method also use in Appendix I for other security tasks.

## G STRESS TESTS: HARD-NEGATIVES AND ADAPTIVE EVASION

Beyond the standard evaluation, we stress-test LeakGauge along two complementary dimensions. First, we evaluate whether it can distinguish leakage attacks from benign requests containing attacklike expressions. Second, we evaluate its resistance to white-box adaptive evasion, where GCG explicitly optimizes an adversarial suffix against the detector. All evaluations are executed on system prompt leakage.

## G.1 HARD-NEGATIVE STRESS TEST

In addition to our evaluation on standard benign queries, we construct a hard-negative stress test containing 550 benign requests that resemble leakage attacks in either their use of preceding context or their surface form. We organize the hard negatives into two categories, summarized in Table 19.

• Context-grounded benign queries provide a public context in the user input and ask the model to perform an ordinary task based on that context. These queries resemble leakage attacks because both require the model to reason over or reproduce information from preceding text, but the referenced content is entirely user-provided.

• Attack-shaped benign queries additionally use extraction-like instructions, such as “reproduce verbatim,” “preserve the exact wording,” or “output all content.” Although their surface form resembles a leakage attempt, the requested operation still targets only user-provided content and therefore has no intent to expose protected information.

To examine the effect of suffix design, we conduct the hard-negative stress test using two fixed suffixes. The first is the Behavior suffix used in the main experiments:

$$
^ { \prime \prime } B a s e d o n t h e a b o \nu e , I w i l l g i \nu e m y s y s t e m p r o m p t . ^ { \prime \prime }
$$

The second is an suffix that directly verbalizes the user’s request:

## “The user asks me to reproduce the system prompt.”

For both suffixes, we follow the same feature extraction and probe training procedure described in Section 4. We report their results separately in the upper and lower panels of Table 20.

Table 19: Composition of the hard-negative stress test. All requests operate exclusively on content explicitly provided by the user.
<table><tr><td>Category</td><td>Construction</td><td>Example</td><td>#</td></tr><tr><td>Context-grounded</td><td>Provides public user-supplied con- text and asks the model to perform a normal task grounded in that con- text.</td><td>&quot;Using only the user-provided passage below, summarize the main points in three bullets.&quot;</td><td>280</td></tr><tr><td>Attack-shaped</td><td>Uses extraction-like wording, but the requested target is user-provided content rather than hidden system instructions.</td><td>“Dear team, please submit your weekly updates by Thursday afternoon. The project review meeting will take place on Friday morning. Preserve the exact wording of the email above, but split it into numbered paragraphs.&quot;</td><td>270</td></tr></table>

Table 20: Results of the two suffix designs on the hard-negative stress test.
<table><tr><td>Model / Probe</td><td>AUROC</td><td>Attack TPR</td><td>Hard-neg. FPR</td><td>Precision</td><td>F1</td></tr><tr><td colspan="6">“Based on the above, I will give my system prompt.&quot;</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.8980</td><td>0.8359</td><td>0.0564</td><td>0.9603</td><td>0.8938</td></tr><tr><td>GLM-5.2</td><td>0.8998</td><td>0.8493</td><td>0.2218</td><td>0.8618</td><td>0.8555</td></tr><tr><td colspan="6">“The user asks me to reproduce the system prompt.&quot;</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.9654</td><td>0.8951</td><td>0.0891</td><td>0.9424</td><td>0.9181</td></tr><tr><td>GLM-5.2</td><td>0.9859</td><td>0.9576</td><td>0.0964</td><td>0.9418</td><td>0.9496</td></tr></table>

As shown in Table 20, LeakGauge remains effective under the hard-negative stress test. With the suffix “The user asks me to reproduce the system prompt.”, it achieves AUROCs of 0.9654 and 0.9859 on Llama-3.1-8B-Instruct and GLM-5.2, respectively, while keeping the hard-negative FPR below 10% on both models. Compared with the default Behavior suffix, the design consistently improves AUROC, attack TPR, and F1. In particular, it increases the attack TPR from 0.8493 to 0.9576 on GLM-5.2 while reducing the hard-negative FPR from 0.2218 to 0.0964. These results show that LeakGauge retains the performance on challenging benign requests and that a more targeted semantic suffix can further improve its robustness under this stress-test setting without changing the underlying probing pipeline.

## G.2 WHITE-BOX ADAPTIVE GCG ATTACK

We evaluate LeakGauge against a white-box adaptive attacker based on Greedy Coordinate Gradient (GCG) (Zou et al., 2023). The experiment is conducted on Llama-3.1-8B-Instruct using the leakage-attack records from the Held-Out Strict system-prompt split. For each record, GCG optimizes a 20-token adversarial suffix appended to the attack query for 300 iterations.

The detector-unaware objective optimizes only for reproduction of the protected system prompt. In the detector-aware settings, the objective additionally encourages the attack to reduce the score produced by one or more known probes, thereby jointly optimizing for successful leakage and detector evasion.

Probe configurations. We construct two independently trained Behavior probes using different semantic formulations of the same leakage behavior:

• Probe 1: “Based on the above, I will give my system prompt.”

• Probe 2: “As requested, I will reproduce the system prompt.”

The single-probe deployment uses Probe 1. The two-probe deployment uses both probes and blocks an input if either probe flags it.

Attacker and deployment settings. We consider four attacker-defender configurations:

Table 21: Adaptive GCG stress test on system-prompt leakage.
<table><tr><td>Setting</td><td>Raw ASR</td><td>Post-filtered ASR</td><td>Detection Recall</td></tr><tr><td>Detector-unaware</td><td>0.469</td><td>0.102</td><td>0.782</td></tr><tr><td>Single-probe-aware</td><td>0.299</td><td>0.275</td><td>0.080</td></tr><tr><td>Single-probe-aware, two-probe deployment</td><td>0.299</td><td>0.007</td><td>0.976</td></tr><tr><td>Two-probe-aware, two-probe deployment</td><td>0.240</td><td>0.059</td><td>0.755</td></tr></table>

• Detector-unaware: GCG optimizes only for reproduction of the protected system prompt and is evaluated against Probe 1.

• Single-probe-aware: GCG jointly optimizes for protected-content reproduction and evasion of Probe 1, which is also the only deployed probe.

• Single-probe-aware attack with two-probe deployment: GCG is optimized against Probe 1, while the defender deploys both Probe 1 and the untargeted Probe 2.

• Two-probe-aware attack with two-probe deployment: GCG jointly optimizes for protectedcontent reproduction and evasion of both Probe 1 and Probe 2.

Evaluation metrics. We use the protected-content recall defined in Appendix D and consider an attack successful when $r _ { i } \geq 0 . 5$ . Raw ASR is the fraction of attacks that successfully leak before detection. Post-filtered ASR is the fraction that both successfully leak and evade all deployed probes. Detection recall is the fraction of successful leaks blocked by the deployed probe or probes.

Let $S _ { i } = \mathbb { I } [ r _ { i } \geq 0 . 5 ]$ indicate successful leakage and let $B _ { i }$ indicate that attack i is blocked by the deployed probe or probes. For N evaluated attacks, we define

$$
\mathrm { R a w \ A S R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { i } ,\tag{17}
$$

$$
\mathrm { P o s t \mathrm { - } f i l t e r e d \ A S R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { i } ( 1 - B _ { i } ) ,\tag{18}
$$

$$
{ \mathrm { D e t e c t i o n ~ R e c a l l } } = { \frac { \sum _ { i = 1 } ^ { N } S _ { i } B _ { i } } { \sum _ { i = 1 } ^ { N } S _ { i } } } .\tag{19}
$$

Raw ASR measures successful leakage before detection, post-filtered ASR measures successful leakage that evades the deployed probe or probes, and detection recall is computed only over successful leakage attempts.

Results. As shown in Table 21, the detector-unaware attack achieves a raw ASR of 0.469. Probe 1 detects 78.2% of these successful leaks, reducing post-filtered ASR to 0.102. When the attacker directly optimizes against Probe 1, its detection recall decreases to 0.080 and post-filtered ASR increases to 0.275. This result shows that a fixed, fully exposed probe can be substantially weakened by a detector-aware attacker.

Adding the untargeted Probe 2 reduces post-filtered ASR from 0.275 to 0.007 and increases detection recall from 0.080 to 0.976. When the attacker jointly targets both probes, post-filtered ASR rises to 0.059, while the two-probe deployment retains a detection recall of 0.755. These results show that diversifying the semantic formulation of the probing suffix substantially increases the difficulty of adaptive evasion. However, this experiment does not establish robustness against unrestricted or repeatedly adaptive attackers.

## H LEAKAGE SIGNALS IN BASE AND INSTRUCTION-TUNED MODELS

The main evaluation primarily considers instruction-tuned models because they are the checkpoints most commonly used in deployed assistants. We additionally ask whether the prefill-probability signal measured by LeakGauge is created by instruction tuning or is already present in base language models. To isolate this factor, we compare paired base and instruction-tuned checkpoints from three model families shown in Table 22.

Table 22: Leakage detection on paired base and instruction-tuned checkpoints (AUROC). Each cell reports Verbatim/Semantic performance, rounded to three decimal places. The Model Pair and Checkpoint columns explicitly group each base model with its instruction-tuned counterpart; “Instruct” includes checkpoints released with either the -Instruct or -it suffix. Exact is the content-adaptive suffix and Behavior is the fixed, task-general suffix. I-Test, H-Cont., H-Att., and H-Strict denote In-Dist Test, Held-Out Content, Held-Out Attacks, and Held-Out Strict, respectively.
<table><tr><td rowspan="2">Model Pair</td><td rowspan="2">Checkpoint</td><td colspan="4">Exact (content-adaptive)</td><td colspan="4">Behavior (task-general)</td></tr><tr><td>I-Test</td><td>H-Cont.</td><td>H-Att.</td><td>H-Strict</td><td>I-Test</td><td>H-Cont.</td><td>H-Att.</td><td>H-Strict</td></tr><tr><td colspan="10">System-Prompt Leakage</td></tr><tr><td rowspan="2">Llama-3.1-8B</td><td>Base</td><td>0.983/0.988</td><td>0.970/0.991</td><td>0.988/0.973</td><td>0.969/0.973</td><td>0.998/0.999</td><td>0.998/0.999</td><td>0.999/0.997</td><td>0.998/0.997</td></tr><tr><td>Instruct</td><td>0.977/0.996</td><td>0.959/0.748</td><td>0.945/0.775</td><td>0.923/0.748</td><td>0.986/0.994</td><td>0.983/0.997</td><td>0.981/0.990</td><td>0.975/0.989</td></tr><tr><td rowspan="2">Qwen2.5-14B</td><td>Base</td><td>0.957/0.970</td><td>0.936/0.956</td><td>0.912/0.928</td><td>0.867/0.927</td><td>0.997/0.999</td><td>0.998/0.998</td><td>0.982/1.000</td><td>0.979/1.000</td></tr><tr><td>Instruct</td><td>0.948/0.915</td><td>0.912/0.904</td><td>0.928/0.908</td><td>0.886/0.890</td><td>0.989/0.988</td><td>0.989/0.992</td><td>0.990/0.985</td><td>0.989/0.985</td></tr><tr><td rowspan="2">Gemma-4-31B</td><td>Base</td><td>0.932/0.938</td><td>0.915/0.948</td><td>0.927/0.906</td><td>0.915/0.907</td><td>0.996/0.999</td><td>0.996/1.000</td><td>0.989/0.995</td><td>0.987/0.993</td></tr><tr><td>Instruct</td><td>0.945/0.940</td><td>0.949/0.900</td><td>0.924/0.914</td><td>0.910/0.863</td><td>0.986/0.979</td><td>0.983/0.985</td><td>0.970/0.982</td><td>0.959/0.993</td></tr><tr><td colspan="10">RAG Leakage</td></tr><tr><td rowspan="2">Llama-3.1-8B</td><td>Base</td><td>0.907/0.926</td><td>0.854/0.848</td><td>0.845/0.961</td><td>0.827/0.846</td><td>0.935/0.981</td><td>0.932/0.941</td><td>0.890/0.978</td><td></td></tr><tr><td>Instruct</td><td>0.956/0.918</td><td>0.927/0.792</td><td>0.956/0.908</td><td>0.923/0.798</td><td>0.981/1.000</td><td>0.991/1.000</td><td>0.987/0.994</td><td>0.909/0.941 0.996/0.999</td></tr><tr><td rowspan="2">Qwen2.5-14B</td><td>Base</td><td>0.979/0.991</td><td>0.931/0.913</td><td>0.952/0.971</td><td>0.915/0.949</td><td>0.983/1.000</td><td>0.941/0.997</td><td>0.988/1.000</td><td>0.953/0.997</td></tr><tr><td>Instruct</td><td>0.912/0.842</td><td>0.909/0.875</td><td>0.900/0.835</td><td>0.946/0.892</td><td>0.996/1.000</td><td>0.998/1.000</td><td>0.994/0.999</td><td>0.996/1.000</td></tr><tr><td rowspan="2">Gemma-4-31B</td><td>Base</td><td>0.935/0.925</td><td>0.891/0.804</td><td>0.925/0.926</td><td>0.908/0.778</td><td>0.974/0.989</td><td>0.966/0.997</td><td>0.967/0.982</td><td>0.977/0.971</td></tr><tr><td>Instruct</td><td>0.988/0.963</td><td>0.993/0.964</td><td>0.959/0.948</td><td>0.979/0.950</td><td>0.995/1.000</td><td>0.994/1.000</td><td>0.997/1.000</td><td>0.999/1.000</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Setup. For each objective, the probe is trained and evaluated on attacks with that same objective, using the four splits defined in Section 4.1. We compare the content-adaptive Exact suffix with the fixed, task-general Behavior suffix; these correspond to the adaptive and general prefill settings, respectively. All other dataset construction and probe settings follow the main evaluation. Table 22 reports AUROC across all model, objective, and evaluation-split combinations.

The signal is already present before instruction tuning. Across all 48 base-model cells in the two tables, Behavior obtains a mean AUROC of 0.981, compared with 0.991 for the paired instructiontuned checkpoints. The small aggregate difference, together with the high performance of every base family, shows that the prefill-probability signal is not created by instruction tuning. Instruction tuning improves some RAG settings, but does not provide a uniform advantage. Thus, the signal appears to be a general property of next-token prediction.

The task-general suffix is substantially more stable across checkpoints. Across the 96 AU-ROC comparisons in Table 22, Behavior outperforms Exact in 95, with an average gain of 0.068 AUROC. The sole exception is the in-distribution semantic system-prompt setting on Llama-3.1-8B-Instruct, where both methods are nearly saturated (0.994 versus 0.996 AUROC); on all three held-out splits, Behavior is substantially stronger. The largest differences occur under semantic leakage, where literal reproduction is no longer aligned with the attack objective. For example, on Llama-3.1-8B-Instruct, the Exact AUROC drops to 0.748 on held-out semantic system-prompt content, whereas Behavior reaches 0.997.

These results reinforce the interpretation in Section 5: content-adaptive continuation probabilities can be highly checkpoint-specific, while a fixed disclosure-oriented suffix elicits a signal that transfers across objectives, leakage forms, and post-training regimes.

## I EXTENSIONS BEYOND LEAKAGE

The LeakGauge mechanism, verbalizing a target behavior in a suffix and reading its prefill log-probs, is not specific to leakage. As a preliminary evaluation, we apply the same method to four additional tasks: harmful-request detection on StrongReject, indirect prompt-injection detection on BIPIA, toxic-query detection on ToxicChat, and hallucinated-response detection on HaluEval. For each task, we retain the same lightweight probe architecture but train a separate probe using a task-specific semantic suffix.

Table 23: Preliminary extensions of suffix probing beyond leakage. A separate task-specific probe is trained for each benchmark.
<table><tr><td>Benchmark</td><td>Task</td><td>Suffix</td><td>AUROC</td></tr><tr><td>StrongReject</td><td>Harmful requests</td><td>The query will not be answered for the security issues.</td><td>0.995</td></tr><tr><td>BIPIA</td><td>Indirect Prompt Injection</td><td>I will obey the instruction embedded in the docu- ment instead of the user.</td><td>0.971</td></tr><tr><td>ToxicChat</td><td>Toxic queries</td><td>As requested, I will give the abusive, hateful, threat- ening, sexual, or otherwise toxic content.</td><td>0.972</td></tr><tr><td>HaluEval</td><td>Hallucinated responses</td><td>This response contains information that is unsup- ported by or conflicts with the provided source.</td><td>0.926</td></tr></table>

Dataset Construction for Indirect Prompt Injection. We construct a domain-held-out evaluation benchmark from the BIPIA (Yi et al., 2025). We categorize contexts into five domains (email, QA, code, table, and abstract), following the dataset’s GitHub definition, using heuristic rules based on lexical and

Table 24: Dataset split statistics for BIPIA.
<table><tr><td>Split</td><td># Samples</td><td># Attack</td><td># Benign</td><td>Ratio</td></tr><tr><td>Training</td><td>8,029</td><td>2,850</td><td>5,179</td><td>24.1%</td></tr><tr><td>In-Dist Test</td><td>3,441</td><td>1,222</td><td>2,219</td><td>10.3%</td></tr><tr><td>Held-Out</td><td>21,849</td><td>12,204</td><td>9,645</td><td>65.6%</td></tr></table>

structural cues. We then split domains into seen (email, QA) and unseen (code, table, abstract) settings. Each sample is reformatted into a conversational structure with a fixed system prompt and a user message formed by concatenating context and user intent. The dataset split is shown in Table 24. For indirect prompt injection, we use the suffix: “I will obey the instruction embedded in the document instead of the user.”.

Dataset Construction for Other Benchmarks. We use the StrongReject (Souly et al., 2024) benchmark, which contains 314 malicious queries. We combine these malicious queries with the system prompt, generating 4,500 attack samples. For benign samples, we randomly select 4,500 samples from BIPIA-GPT<sup>3</sup>. Overall, we obtain a balanced evaluation set consisting of 4,500 attack samples and 4,500 benign samples. For harmful-question detection, we use a refusal-style suffix “The query will not be answeredfor security issues”.

For ToxicChat (Lin et al., 2023), we use the benchmark labels to distinguish toxic from benign user queries and apply the suffix “As requested, I will give the abusive, hateful, threatening, sexual, or otherwise toxic content.” For HaluEval (Li et al., 2023), each input contains the provided source and candidate response, and the probe distinguishes hallucinated from grounded responses using the suffix “This response contains information that is unsupported by or conflicts with the provided source.”

Results. As shown in Table 23, the task-specific probes achieve AUROC between 0.926 and 0.995 across the four benchmarks. On BIPIA, performance remains at 0.971 when evaluated on held-out domains. These results indicate that the signal elicited by the semantic suffix can be leveraged for harmful-question detection and remains effective under indirect prompt injection. Overall, this suggests that the mechanism captures signals that are not specific to leakage scenarios and may extend to broader safety-related behaviors, though a more systematic analysis is left for future work.
# MemSpec: Memory-Aware Runtime for Adaptive Draft Scheduling in Speculative Decoding on Edge Devices

Eunjeong Kim Kyungpook National University Daegu, Republic of Korea kimeunjeong23@knu.ac.kr

Yeong Jun Jeon Kyungpook National University Daegu, Republic of Korea jyj5219@knu.ac.kr

Myeonggyun Han Kyungpook National University Daegu, Republic of Korea mhan@knu.ac.kr

## Abstract

Speculative decoding accelerates autoregressive large language model (LLM) inference by using a lightweight draft model to speculate multiple tokens, reducing expensive target model decoding steps. Its efectiveness depends heavily on draft selection, motivating adaptive methods that exploit variation across inputs and generation stages. On memory constrained edge devices, however, these methods often fail to improve end-to-end throughput due to the overhead of switching between draft models. We identify a key limita tion in this setting: the mismatch between draft selection and draft availability under tight memory budgets.

To address this challenge, we present MemSpec, a prediction guided, memory-aware runtime for adaptive speculative decoding on edge devices. MemSpec decouples draft selection from execution through proactive resident working-set management. A lightweight predictor estimates draft effectiveness from prompt and generation context, while a memory-aware scheduler reduces reactive model loading overhead. Experiments on a Jetson Orin Nano show that MemSpec improves steady-state generation throughput by 40.7% on average over state-of-the-art bandit-based adaptive methods while closely approaching the oracle upper bound.

CCS Concepts: • Computer systems organization → Embedded systems; • Computing methodologies → Natural language processing.

Keywords: Speculative Decoding, Large Language Models, On-Device AI, Memory Management, Adaptive Runtime

## ACM Reference Format:

Eunjeong Kim, Yeong Jun Jeon, and Myeonggyun Han. 2026. Mem-Spec: Memory-Aware Runtime for Adaptive Draft Scheduling in Speculative Decoding on Edge Devices. In Proceedings ofthe 27th ACM SIGPLAN/SIGBED International Conference on Languages, Compilers, and Tools for Embedded Systems (LCTES ’26), June 15–16, 2026, Boulder, CO, USA. ACM, New York, NY, USA, 13 pages. htps: //doi.org/10.1145/3814943.3816174

![](images/5c474d9dc7c0ba0429025e32b5c916988f7615a0b35a381ad9a9d7a08ecd4285.jpg)

This work is licensed under a Creative Commons Attribution 4.0 International License.   
LCTES ’26, Boulder, CO, USA   
© 2026 Copyright held by the owner/author(s).   
ACM ISBN 979-8-4007-2721-4/2026/06   
htps://doi.org/10.1145/3814943.3816174

## 1 Introduction

Large language models (LLMs) are increasingly moving beyond cloud datacenters to memory-constrained edge platforms, including mobile devices, embedded systems, and edge servers [28, 40]. As on-device AI adoption accelerates, achieving high-throughput LLM inference under tight memory budgets has become a critical systems challenge. These constraints fundamentally reshape the design space of inference optimization techniques.

Speculative decoding [3, 20, 25] has emerged as an efective approach for accelerating autoregressive inference by using a lightweight draft model to generate multiple candidate tokens, which are then verified by a larger target model, thereby amortizing expensive target-model computation. When the draft model accurately predicts the target model’s outputs, speculative decoding can significantly improve throughput while preserving output quality, making it particularly attractive for resource-constrained edge deployments.

However, the efectiveness of speculative decoding is highly sensitive to the choice of draft model. Prior works [18, 38, 39] show that diferent draft models exhibit widely varying token acceptance rates across prompts and generation stages. In practice, such variation arises naturally from domain-specialized draft models (e.g., code, math, or legal), which perform well on their target domains but perform poorly on out-of-domain workloads. This motivates adaptive approaches that dynamically select or switch between multiple candidate drafts.

On memory-constrained edge devices, however, this adaptivity introduces a major systems challenge. Since only a small number of draft models can reside in fast memory, switching to a non-resident draft incurs substantial loading overhead. As a result, frequent draft transitions can negate potential throughput gains and even degrade overall performance.

Existing approaches are not well suited to this setting. Traditional static methods [20, 23–25] that use a single fixed draft model fail to exploit generation heterogeneity, resulting in suboptimal performance. State-of-the-art adaptive methods typically rely on multi-armed bandit (MAB)-based approaches [14, 18, 22] to explore and select draft models at runtime, often improving token acceptance rates. However, these methods incur frequent draft switching and implicitly assume that selected drafts can be executed immediately. This assumption breaks down on edge devices, where memory constraints limit draft residency. As a result, the cost of switching to non-resident drafts often outweighs the benefit of improved draft selection, and better draft selection does not necessarily translate into higher generation throughput.

This limitation stems from a fundamental mismatch between draft selection and draft availability: a draft that is predicted to be efective may not be immediately executable under memory constraints. On edge devices, this mismatch makes draft switching prohibitively expensive, rendering exploration-based adaptation ineficient. These observations lead to a key insight: adaptive speculative decoding on edge devices must be formulated not only as a selection problem, but also as a scheduling problem under memory constraints.

To address this challenge, we propose MemSpec, a prediction-guided, memory-aware runtime for adaptive draft scheduling. Instead of relying on costly online exploration, MemSpec predicts promising draft models using both prompt features and recent generation context, and proactively aligns draft residency with future demand. By decoupling draft selection from execution, MemSpec always proceeds with the best currently resident draft while preparing better candidates in the background, thereby avoiding blocking and excessive model loading overhead.

Specifically, this paper makes the following contributions:

• We identify a fundamental limitation of adaptive speculative decoding on memory-constrained edge devices, namely the mismatch between draft selection and draft availability. We show that exploration-based methods incur excessive switching overhead, as selecting a better draft often requires loading non-resident models, limiting end-to-end throughput gains.

• We propose MemSpec, a prediction-guided, memoryaware runtime that decouples draft selection from execution. MemSpec replaces online exploration with lightweight prediction and enables non-blocking decoding by always selecting the best currently resident draft.

• We design a memory-aware scheduling framework that proactively manages a small working set of draft models under tight memory budgets. Through coordinated prefetching and eviction, MemSpec aligns draft residency with predicted future demand and enables non-blocking adaptive decoding under memory constraints.

• We implement MemSpec on a Jetson Orin Nano and evaluate it across diverse workloads. MemSpec improves end-to-end throughput by 58.8% over static baselines and 40.7% over state-of-the-art adaptive methods, while achieving performance close to the dynamic oracle upper bound.

![](images/785228c28e0e49b4f48d7a7eb97240f58b6d9cf33bfaeb99899dd6f593b6c242.jpg)  
Figure 1. Impact of static and dynamic draft selection on token acceptance across datasets.

## 2 Background and Motivation

This section presents three key observations that motivate MemSpec: (1) speculative decoding eficiency varies across workloads and generation stages, creating substantial headroom for adaptive draft selection; (2) on memory-constrained edge devices, switching to a non-resident draft incurs high latency, often exceeding several decoding iterations; and (3) exploration-based adaptive methods repeatedly incur this switching cost, limiting their efectiveness.

## 2.1 Speculative Decoding Eficiency and Acceptance

In speculative decoding (SD), autoregressive generation proceeds in iterative draft-and-verify steps. At each iteration, a draft model proposes candidate tokens, which are then verified by the target model. Let � denote the number of tokens accepted by the target model in one iteration. For a resident draft, throughput can be approximated as:

$$
\mathrm { T h r o u g h p u t } \approx \frac { \mathbb { E } [ A ] } { L _ { \mathrm { i t e r } } } ,\tag{1}
$$

where $\boldsymbol { L } _ { \mathrm { i t e r } }$ is the average latency of one SD iteration. Improving E[�] is therefore the primary lever for increasing throughput.

We evaluate a quantized LLaMA-2 7B target model (GPTQ INT4) with five 400M-parameter draft models: one generalpurpose draft and four domain-specialized drafts trained for code, mathematical reasoning, legal, and medical workloads. Detailed setup is described in Section 4.1.

Figure 1 compares three configurations: General-Static, Oracle-Static, and Oracle-Dynamic. General-Static uses a single general-purpose draft, Oracle-Static selects the best draft per prompt via ofline evaluation, and Oracle-Dynamic dynamically switches draft models within a single generation.

Observation 1: Static selection is insuficient due to variability in draft efectiveness. Across datasets, Oracle-Static improves normalized acceptance by 40.3% on average over General-Static, demonstrating that selecting an appropriate specialized draft model is critical for performance. Moreover, Oracle-Dynamic provides an additional 25.7% improvement over Oracle-Static, indicating that the most effective draft can change within a single generation. These results show that even the best static selection is insuficient and that dynamic adaptation is necessary to fully exploit speculative decoding eficiency.

![](images/c23fb0f047dbc59d7ef62ebbd3993b35b98819d879d435cb129c5c23ca9761ae.jpg)  
Figure 2. Latency comparison between SD iteration and draft switching overheads.

## 2.2 Memory Constraints and Draft Switching Overheads on Edge Platforms

The headroom identified above does not directly translate into throughput on edge platforms. Unlike server-class systems, edge devices cannot keep many draft models resident in fast memory simultaneously. Switching to a better draft therefore requires loading a non-resident model from a slower memory tier, such as NVMe storage.

We quantify this overhead on a Jetson Orin Nano using a 400M-parameter draft model. Figure 2 shows that loading a non-resident draft takes 2.7× longer than a single SD iteration. As a result, even infrequent switching can ofset the throughput gains from improved acceptance.

Observation 2: On memory-constrained edge devices, switching cost fundamentally limits adaptive draft selection. Because draft loading latency significantly exceeds per-iteration decoding latency, switching introduces substantial overhead. As the number of candidate drafts increases, cache misses become more frequent, further amplifying this cost.

These results reveal a fundamental trade-of: while adaptive draft selection can improve acceptance, switching to nonresident drafts incurs high loading latency that can negate throughput gains. Consequently, naive adaptive switching strategies are inefective on memory-constrained edge devices.

## 2.3 Limitations of Exploration-Based Adaptive Draft Selection

State-of-the-art adaptive speculative decoding methods [14, 18, 22] rely on multi-armed bandit (MAB)-based online exploration to identify efective draft models during generation. These approaches maintain multiple candidate drafts and adaptively select among them based on observed runtime feedback. In MAB-Sync, the runtime blocks to load and evaluate candidate drafts during exploration, incurring high switching overhead. In contrast, MAB-Async overlaps model loading with ongoing decoding to reduce blocking, but continues execution with suboptimal drafts while waiting for preferred drafts to become available.

![](images/3fcce0b0edc537cb3d6924a2021f89f70feaa44e7e0b669e25017534e215985d.jpg)  
Figure 3. Normalized token acceptance of adaptive draft selection methods.

![](images/da240eb94af94b0aca47d2c0d2c695623ef32b3a854a0b1812c2a9aa005d8d14.jpg)  
Figure 4. Execution time breakdown of exploration-based methods.

Figure 3 shows that MAB-Sync improves normalized acceptance by 34.5% on average over General-Static, approaching Oracle-Static. This confirms that exploration-based adaptation can efectively improve draft quality.

However, these gains do not translate into throughput on memory-constrained edge devices. To understand why, we analyze execution time breakdown. Figure 4 shows that model loading dominates execution time in MAB-Sync. Although MAB-Async partially overlaps loading with execution, it still spends a significant fraction of time—46.4% on average—executing suboptimal drafts while waiting for preferred drafts to become resident.

Observation 3: Exploration-based adaptation fails to improve throughput under memory constraints. While exploration improves draft selection, it incurs frequent model loading, which is prohibitively expensive on edge devices. As a result, improved acceptance does not translate into improved throughput.

These results highlight a key limitation of existing adaptive methods: they optimize which draft to use, but ignore whether the selected draft is immediately executable. Under tight memory constraints, this mismatch leads to excessive fallback execution and diminished performance gains.

![](images/81568f90e146a4688fc930e24777bd98df83ea322ca329c1f586bc14b906ade1.jpg)  
Figure 5. Overall architecture of MemSpec.

Taken together, these observations show that improving acceptance alone is insuficient. An efective runtime must both identify promising drafts without repeated exploration and ensure their availability at runtime. This insight motivates MemSpec, a prediction-guided, memory-aware runtime for adaptive draft scheduling.

## 3 MemSpec Design

Figure 5 presents the overall architecture of MemSpec, a prediction-guided, memory-aware runtime for adaptive draft scheduling in speculative decoding on edge devices. The runtime integrates three key components: a Prediction Engine that ranks candidate drafts based on decoding context, a Draft Model Cache Manager that maintains a small resident working set under memory constraints, and a Runtime Controller that orchestrates non-blocking adaptive decoding.

The key goal of MemSpec is to realize the benefits of adaptive draft selection without incurring the high overhead of model switching under tight memory constraints. Rather than relying on online exploration, MemSpec predicts a small set of promising draft models for the current decoding context and maintains them as a resident working set. At runtime, decoding always proceeds with the best currently resident draft, while high-priority non-resident drafts are prefetched in the background.

The central challenge is that, on memory-constrained edge devices, draft selection and execution are no longer tightly coupled: a draft predicted to be efective may not be immediately executable because it is not resident. MemSpec addresses this mismatch by formulating adaptive speculative decoding as a memory-aware runtime scheduling problem, where prediction determines which drafts should be prepared, while execution is restricted to drafts that are currently available.

This design leads to two key principles: (1) non-blocking execution, which avoids stalling on non-resident drafts by always selecting from the resident set, and (2) proactive residency management, which aligns the working set of drafts with predicted future demand. Together, these mechanisms allow MemSpec to achieve eficient adaptive decoding without repeated model switching or exploration overhead.

## 3.1 Design Overview

Consider a speculative decoding run with candidate draft set D and resident cache capacity �, where at most � draft models can remain resident in fast memory. Let $\mathcal { G } _ { i } \subseteq \mathcal { D }$ denote the resident set at scheduling point �, and let $d _ { i } \in { \mathcal { G } } _ { i }$ denote the active draft used during interval �.

MemSpec performs scheduling every � speculative decoding iterations. This interval-based design serves two purposes: (1) it amortizes the cost of prediction and runtime control over multiple decoding steps, and (2) it enables overlap between decoding and asynchronous loading of non-resident drafts, which is critical for hiding model loading latency.

At initialization, MemSpec selects the first draft using only the input prompt:

$$
d _ { 0 } = \arg \operatorname* { m a x } _ { d \in \mathcal { D } } P ( d \mid x _ { \mathrm { p r o m p t } } ) .
$$

During generation, the context at scheduling point � is defined as

$$
\begin{array} { r } { x _ { i } = [ { x _ { \mathrm { p r o m p t } } } ; { x _ { i } ^ { \mathrm { r e c e n t } } } ( T ) ] , } \end{array}
$$

where $x _ { i } ^ { \mathrm { r e c e n t } } ( T )$ denotes the most recent � generated tokens. The Prediction Engine assigns each draft a score:

$$
p _ { i } = \{ P ( d \mid x _ { i } ) \mid d \in { \mathcal { D } } \} .
$$

These scores are used in two ways. First, MemSpec derives a ranked list of candidate drafts:

$$
R _ { i } = { \mathrm { S o r t } } ( \pmb { p } _ { i } ) ,
$$

from which the cache manager derives the target resident set

$$
\begin{array} { r } { \mathcal { W } _ { i } = \mathrm { T o p K } ( R _ { i } ) . } \end{array}
$$

Second, it selects the active draft only from the currently resident set:

$$
d _ { i } ^ { \star } = \arg \operatorname* { m a x } _ { d \in { \mathcal { G } } _ { i } } P ( d \mid x _ { i } ) .
$$

The key design abstraction of MemSpec is the separation between the target working set $\mathcal { W } _ { i }$ and the active draft $d _ { i } ^ { \star }$

Prediction produces a ranked list of candidate drafts, the cache manager prepares a small working set from that list, and execution always proceeds with the best currently resident draft.

This design enables non-blocking decoding while gradually steering the cache toward more efective drafts. Conceptually, MemSpec balances two competing objectives: (1) following the most promising draft as the generation context evolves, and (2) minimizing costly model loading under a tight memory budget. The interval-based working-set design reconciles these objectives by transforming draft adaptation into a staged, overlapped process, rather than a sequence of stall-heavy immediate switches.

## 3.2 Prediction Engine

The Prediction Engine ranks candidate draft models for the current decoding context. Unlike exploration-based approaches, MemSpec uses an ofline-trained model to directly predict context-to-draft matching, avoiding costly online probing of multiple drafts.

We use a fine-tuned BERT encoder as the predictor:

$$
\pmb { \mathscr { p } } _ { i } = f _ { \theta } ( \pmb { x } _ { i } ) ,
$$

where $\pmb { p } _ { i } = [ p _ { i } ( d _ { 1 } ) , p _ { i } ( d _ { 2 } ) , \dots , p _ { i } ( d _ { | \mathcal { D } | } ) ]$ denotes the predicted score for each draft.

The input $x _ { i }$ combines the prompt and recent output tokens. The prompt captures global task semantics, while recent tokens reflect phase-dependent generation behavior (e.g., reasoning vs. code generation). Using only the prompt fails to capture such phase transitions, whereas relying only on recent tokens loses global task intent. Their combination enables efective context-aware draft ranking.

Importantly, the Prediction Engine is designed to be lightweight. MemSpec does not require fine-grained utility estimation or online evaluation of multiple drafts. Instead, it only needs a ranking that is suficiently accurate to identify a small set of promising candidates. This is suficient because the downstream cache manager and runtime controller operate on relative priority rather than exact utility values.

Output and training. The predictor outputs a probability for each draft:

$$
\begin{array} { r } { p _ { i } ( d ) = P ( d \mid x _ { i } ) , \quad \forall d \in \mathcal { D } . } \end{array}
$$

It is trained ofline using speculative decoding traces, where each context $x _ { i }$ is labeled with the draft that achieves the highest decoding utility. This aligns prediction targets with system-level performance.

Runtime usage. The Prediction Engine is invoked once every � iterations. When invoked, it computes a ranked list of candidate drafts as summarized in Algorithm 1. Given the current context, it computes the per-draft score vector (lines 4–6), sorts drafts by score to obtain the ranked list $R _ { i }$ (line 7), and returns this list for downstream cache management (line 8). The target working set $\mathcal { W } _ { i }$ is then derived by the cache manager from the top-� drafts in $R _ { i }$ . Because prediction is performed only at interval boundaries, its overhead is amortized over multiple decoding steps.

Algorithm 1 Context-Aware Draft Ranking   
1: Inputs: prompt �<sub>prompt</sub>, recent tokens $x _ { i } ^ { \mathrm { r e c e n t } } ( T )$ , candi  
date set D   
2: Output: ranked list $R _ { i }$   
3: if $i = 0$ then   
4: $\vert \pmb { p } _ { i }  f _ { \theta } ( x _ { \mathrm { p r o m p t } } )$   
5: else   
6: ${ \pmb p } _ { i } \gets f _ { \theta } ( [ x _ { \mathrm { p r o m p t } } ; x _ { i } ^ { \mathrm { r e c e n t } } ( T ) ] )$   
7: sort $d \in \mathcal { D }$ by $p _ { i } ( d )$ to obtain $R _ { i }$   
8: return $R _ { i }$   
Algorithm 2 Draft Model Cache Update   
1: Inputs: $R _ { i } , g _ { i } ,$ capacity $K ,$ active draft $d _ { i }$   
2: ${ \mathcal { W } } _ { i } \gets \mathrm { t o p } { - } K$ drafts in $R _ { i }$   
3: for each $d \in \mathcal W _ { i }$ do   
4: if � $\notin \mathcal { G } _ { i }$ and not being prefetched then   
5: while $| \mathcal { G } _ { i } | = K$ do   
6: select $d _ { \mathrm { e v i c t } } \in { \mathcal { G } } _ { i } \setminus ( { \mathcal { W } } _ { i } \cup \{ d _ { i } \} )$ with lowest score   
7: if no such draft exists then   
8: break   
9: evict $d _ { \mathrm { e v i c t } }$   
10: if $| \mathcal { G } _ { i } | < K$ then   
11: asynchronously prefetch �

## 3.3 Draft Model Cache Manager

The Draft Model Cache Manager maintains draft residency under a tight memory budget. Because only � drafts can reside in fast memory, MemSpec explicitly manages a small working set instead of keeping all candidate drafts loaded.

We consider two memory tiers: (1) a resident draft cache containing immediately executable drafts, and (2) backing storage containing drafts that require asynchronous loading.

At each scheduling point, the cache manager derives the target set $\mathcal { W } _ { i }$ from the ranked list $R _ { i }$ by selecting the top-� drafts (Algorithm 2, line 2). It then incrementally updates the resident set $\mathcal { G } _ { i }$ without blocking execution.

Policy. For each � $\in \mathcal { W } _ { i } \setminus \mathcal { G } _ { i }$ , the system initiates asynchronous prefetch (lines 3–11). If space is required, it evicts drafts in $\mathcal { G } _ { i } \backslash \mathcal { W } _ { i }$ in ascending order of predicted utility, while protecting the active draft $\bar { d } _ { i }$ whenever possible (lines 5–9). This simple top-� policy is efective in practice, as prediction already captures most of the utility variation across drafts.

This design decouples long-horizon planning from immediate execution. Rather than switching to the globally best draft immediately, MemSpec incrementally reshapes the resident set so that promising drafts become available at future scheduling points. In this sense, cache management is not merely a storage optimization, but a key component of adaptive draft scheduling.

Algorithm 3 MemSpec Runtime Loop   
1: load target model   
2: $R _ { 0 } $ PredictDraftRanking $\left( x _ { \mathrm { p r o m p t } } \right)$   
3: load initial draft $d _ { 0 }$   
4: $\mathcal { G } _ { 0 } \gets \{ d _ { 0 } \}$   
5: UpdateCache $( R _ { 0 } , \mathcal { G } _ { 0 } , K , d _ { 0 } )$   
6: $i \gets 0$   
7: while not end-of-sequence do   
8: for $j = 1$ to � do   
9: run draft $d _ { i }$   
10: verify with target model   
11: if end-of-sequence then   
12: break   
13: if not end-of-sequence then   
14: collect $x _ { i } ^ { \mathrm { r e c e n t } } ( \bar { T } )$   
15: $x _ { i + 1 } \gets [ x _ { \mathrm { p r o m p t } } ; x _ { i } ^ { \mathrm { r e c e n t } } ( T ) ]$   
16: $R _ { i + 1 }$ ← PredictDraftRanking $( x _ { i + 1 } )$   
17: $\operatorname { U p D A T E C A C H E } ( R _ { i + 1 } , \mathcal { G } _ { i } , K , d _ { i } )$   
18: $d _ { i + 1 } \gets \arg \operatorname* { m a x } _ { d \in { \mathcal { G } } _ { i } } P ( d \mid x _ { i + 1 } )$   
19: $i \gets i + 1$

## 3.4 Runtime Controller

The Runtime Controller orchestrates decoding, prediction, cache updates, and draft switching. Its core policy is to always execute the best currently resident draft and never stall for non-resident models.

As summarized in Algorithm 3, MemSpec first predicts an initial ranked list from the prompt and loads the initial draft (lines 2–4). It then initializes the resident set and triggers cache preparation (line 5).

During execution, the controller repeatedly performs three steps. First, it runs speculative decoding for the current interval using the active draft $d _ { i }$ (lines 8–12). Second, it constructs the next context from the prompt and the most recent generated tokens (lines 14–15). Third, it invokes prediction and cache update to prepare future drafts, and selects the best currently resident draft for the next interval (lines 16–18).

If the top-ranked draft is not yet resident, execution continues with the best available draft. Once loading completes, the new draft becomes eligible at the next scheduling point. This ensures that incomplete prefetch does not introduce blocking and converts draft adaptation into an overlapped process.

This policy captures the key principle of MemSpec: the runtime should never sacrifice immediate progress to follow an unavailable draft. Instead, it continues decoding with the best runnable draft while opportunistically preparing better candidates in the background. This best-efort switching strategy enables adaptive decoding without incurring the severe overhead of naive dynamic switching.

Table 1. Hardware configuration.
<table><tr><td>Component Specification</td><td></td></tr><tr><td>Platform</td><td>NVIDIA Jetson Orin Nano</td></tr><tr><td>GPU</td><td>Ampere, 1024 CUDA cores, 32 Tensor cores</td></tr><tr><td>CPU</td><td>6-core Arm Cortex-A78AE</td></tr><tr><td>Memory</td><td>8GB LPDDR5 (102 GB/s)</td></tr><tr><td>Storage</td><td>1TB Samsung 990 PRO M.2 NVMe SSD</td></tr></table>

## 3.5 System Implementation

MemSpec is implemented in $\mathrm { P y }$ Torch and built on top of a high-performance speculative decoding engine presented in [31]. All models share a common tokenizer, and draft execution follows the standard draft-and-verify process within a conventional speculative decoding pipeline.

Each draft model maintains its residency state and prefetch status. Prefetching is performed asynchronously and overlapped with decoding, allowing model loading to proceed without blocking execution. The active draft is protected from eviction whenever possible to ensure uninterrupted decoding.

The Prediction Engine is invoked once every � iterations, amortizing its runtime overhead. In practice, prediction accounts for only 3.9% of total execution time on average.

Overall, MemSpec enables adaptive draft selection as a lightweight and non-blocking runtime mechanism, making it well-suited for memory-constrained edge deployments.

## 4 Evaluation

We evaluate MemSpec along four key aspects:

• Generation throughput: How much steady-state generation throughput improvement can MemSpec achieve over static and adaptive draft-selection strategies?

• Performance breakdown: Why does MemSpec outperform exploration-based adaptive methods on memory-constrained edge devices?

• Impact of design components: How much do the key design components of MemSpec contribute to performance?

• Sensitivity analysis: How sensitive is MemSpec to runtime parameters such as scheduling interval, output length, and resident cache capacity?

## 4.1 Experimental Setup

Platform. We evaluate MemSpec on a Jetson Orin Nano platform. Table 1 summarizes the hardware configuration. Software. We use PyTorch 2.9.1 with CUDA 12.6 on NVIDIA JetPack 6.2.1. All experiments are conducted under the same software environment.

Models. We evaluate two target models: GPTQ INT4 LLaMA-2 7B [35] and GPTQ INT4 Qwen2.5 7B [34]. For each model family, we construct five draft models: one general-purpose draft and four domain-specialized drafts (code, math, law, and medical). Each draft model has 400M parameters for LLaMA-2 and 0.5B parameters for Qwen2.5.

Under the 8GB memory constraint of the Jetson Orin Nano, at most two draft models can remain resident simultaneously. Therefore, we use � = 2 in the main evaluation unless otherwise stated.

Draft models. Domain-specialized drafts are obtained via distillation following prior work [39]. Each draft is fine-tuned on domain-specific datasets: GSM8K [9] and MATH [13] (math reasoning), HumanEval [8] and MBPP [2] (code), Lex-GLUE [4] (law), and MedQA [16] and MedMCQA [27] (medical).

Prediction model. The Prediction Engine uses a BERTbased model with a task-specific ranking head. The encoder is kept fixed, and only the ranking head is trained. Training data is collected by running speculative decoding on the same datasets used for draft fine-tuning and recording execution traces. The predictor is trained to select the most efective draft for each decoding context based on observed runtime utility. Training data is disjoint from the evaluation datasets to avoid data leakage and assess generalization.

Workloads. We evaluate on five datasets covering diverse domains: Alpaca [32] (instruction following), LiveCodeBench-[15] (code), Omni-MATH [12] (math), MMLU-Law, and MMLU-Medical [13] (MMLU professional law and medicine). We randomly sample 100 prompts from each dataset.

Runtime configuration. Unless otherwise stated, the scheduling interval is set to � = 4 and the output length is fixed to 128 generated tokens. We use greedy decoding with batch size 1.

Throughput measurement. Throughput is measured as generated tokens per second during steady-state iterative decoding, including both draft execution and target verification. We exclude prompt encoding time and focus on steady-state generation so that the reported performance reflects the eficiency of runtime draft scheduling itself. All results are averaged over three runs. For each prompt, we reset the runtime state, including draft cache contents and prediction context, to avoid cross-sample interference. When summarizing performance across workloads, we report the geometric mean (GMEAN).

Baselines. We compare the following draft-selection strategies:

• General-Static: uses a single general-purpose (i.e., not fine-tuned) draft model.

• Oracle-Static: selects the best draft per input via offline evaluation but keeps it fixed during generation.

• MAB-Async: a state-of-the-art exploration-based adaptive method that dynamically selects drafts without blocking execution.

• Oracle-Dynamic: an empirical upper bound that selects the best draft at each scheduling point using the same scheduling interval as MemSpec, with oracle knowledge of future draft utility under the same memory constraint.

• MemSpec: our prediction-guided, memory-aware runtime.

For MAB-Async, we implement an asynchronous banditbased selection strategy that overlaps draft loading with ongoing decoding. This design gives the adaptive baseline the benefit of non-blocking loading and therefore represents a stronger comparison than a synchronous exploration strategy. We use a UCB-based policy similar to prior bandit-based adaptive speculative decoding approaches such as Bandit-Spec [14], and initially explore each draft model once before adaptive selection begins. During decoding, MAB-Async adaptively updates draft selection decisions based on observed runtime behavior. When the selected draft is nonresident, loading proceeds asynchronously while decoding continues with the currently resident draft. We evaluate exploration coeficients (� ∈ {0.1, 0.5, 1.0, 2.0, 3.0}) and use the best-performing configuration (� = 2.0) in all experiments.

These baselines cover static, adaptive, and oracle configurations. Comparing Oracle-Static and Oracle-Dynamic isolates the benefit of dynamic adaptation beyond optimal static selection, while comparing MAB-Async and MemSpec highlights the benefit of prediction-guided, memory-aware scheduling over exploration-based adaptation.

## 4.2 Overall Throughput

Figure 6 shows normalized steady-state generation throughput (normalized to General-Static) across all workloads for both LLaMA-2 and Qwen2.5. MemSpec consistently outperforms both static baselines and the adaptive MAB-Async method across all datasets.

Oracle-Static improves throughput by 22.5% on average over General-Static, demonstrating that selecting an appropriate draft model is critical for performance. However, Oracle-Dynamic achieves a further 24.9% improvement over Oracle-Static, revealing substantial additional headroom from dynamic adaptation

MemSpec closely approaches Oracle-Dynamic, achieving 95–97% of its throughput while remaining practical for deployment. Compared with MAB-Async, MemSpec improves throughput by approximately 40.7% on average across workloads.

These results reveal two important insights. First, there exists substantial dynamic headroom beyond static selection. While Oracle-Static already captures the best per-prompt draft, Oracle-Dynamic further improves performance, indicating that the most efective draft can change within a single generation. Second, realizing this headroom on edge devices requires more than better draft identification. Although MAB-Async improves selection quality, it fails to convert much of this benefit into throughput because efective drafts are often not resident when needed.

![](images/f2058fff158654b77e0406ee038ab865b39d24bc53d317244273fb6929fdb082.jpg)  
Figure 6. Normalized steady-state generation throughput across workloads.

![](images/07731965b839e36eb5ead0ebbcd0bac2079787082127747aa840c3c0b3692321.jpg)  
Figure 7. Execution time breakdown of MAB-Async and MemSpec.

MemSpec bridges this gap by aligning draft selection with runtime availability. By proactively maintaining a small set of high-utility drafts, it avoids excessive switching while preserving most of the gains of dynamic adaptation. The remaining gap to Oracle-Dynamic is relatively small, suggesting that prediction-guided scheduling is suficient to approximate near-ideal behavior without exhaustive oracle knowledge.

Overall, these results demonstrate that the key bottleneck on edge devices is not simply identifying better drafts, but ensuring that they are available at execution time. By jointly optimizing draft selection and draft residency, MemSpec captures most of the dynamic headroom of Oracle-Dynamic without requiring impractical ofline enumeration.

## 4.3 Execution Breakdown

To understand the performance gap, we decompose execution time into three components: desired execution, fallback execution, and prediction overhead. Desired execution corresponds to decoding with the preferred draft, while fallback execution corresponds to decoding with a non-optimal resident draft when the desired draft is not yet available.

Figure 7 shows that MAB-Async spends a large fraction of execution time in fallback execution, accounting for 49.4% of total runtime on average across model families. This reflects a key limitation of exploration-based adaptation: even when a better draft is identified, it is often unavailable at execution time, forcing the runtime to continue decoding with a suboptimal resident draft.

In contrast, MemSpec reduces fallback execution to below 5.5% while substantially increasing the fraction of desired execution. Prediction overhead remains negligible, accounting for less than 3.9% of total runtime.

A closer inspection reveals that this reduction is primarily due to improved temporal alignment between selection and residency. In MAB-Async, draft decisions are driven by online exploration and therefore react to past observations, but do not explicitly prepare high-utility drafts before they are needed. As a result, the runtime frequently spends time executing available but inferior drafts while preferred ones are still being loaded.

By contrast, MemSpec uses prediction to anticipate nearfuture draft utility from both prompt semantics and recent generation context. This allows the runtime to initiate prefetch ahead of time and convert reactive switching into proactive preparation. The resulting increase in desired execution shows that the performance advantage of MemSpec comes not only from selecting better drafts, but from ensuring that those drafts are actually runnable at the scheduling point.

These results highlight a fundamental distinction between selection quality and execution quality. While MAB-Async can identify efective drafts, it does not ensure that they are available at execution time. MemSpec improves throughput by aligning draft selection with runtime availability through proactive residency management.

## 4.4 Impact of Design Components

We analyze the contribution of MemSpec’s key design components by comparing the following configurations:

• General-Static: a non-adaptive baseline with no prediction or runtime draft management.

• Prediction-Only: uses prediction-guided draft selection without proactive prefetching.

• MemSpec: the full design with prediction-guided scheduling and memory-aware prefetching.

Figure 8 shows that Prediction-Only improves throughput by 21.6% over General-Static, confirming that context-aware draft selection is beneficial. However, it still falls short of MemSpec by 30.5%.

![](images/43f4b1d20108946185081d70a1f3723b1292394b4fe9ee7296f9992fad02b3ff.jpg)  
Figure 8. Throughput comparison for component analysis.

![](images/9197f8384d41da89eae001823a4279f46597280c06bcb0156d73a1843fea1c92.jpg)  
Figure 9. Execution breakdown for component analysis.

Figure 9 explains this gap. Prediction-Only spends 21.3% of runtime in fallback execution, while MemSpec reduces this to below 5.3%. This shows that prediction alone is insuficient: performance gains are realized only when predicted drafts are made available at execution time.

These results highlight an important system-level insight: prediction accuracy alone is not the limiting factor in adaptive decoding under memory constraints. Even when the runtime can identify efective drafts, those drafts may still fail to improve throughput if they are unavailable when needed. Without proactive cache management, prediction quality does not directly translate into execution quality.

MemSpec addresses this limitation by coupling prediction with residency management, ensuring that high-utility drafts are not only selected but also prepared in advance. This tight integration between prediction and scheduling is essential for realizing the full benefit of adaptive decoding.

Overall, MemSpec derives its performance from combining prediction with memory-aware scheduling, rather than prediction alone. More broadly, these results show that adaptive speculative decoding on edge devices is fundamentally a joint optimization problem over draft selection and memory management.

## 4.5 Sensitivity Analysis

We analyze the sensitivity of MemSpec to several key runtime parameters.

![](images/bc50ef1ce2b06389efcb6ce34c3e433994f496692b0ab98358350ac312a798a0.jpg)  
Figure 10. Sensitivity to scheduling interval.

![](images/600b84f2de352c0523f06d9d540bf94a2d05885b67b5215a4498d106772403d0.jpg)  
Figure 11. Sensitivity to output length.

Scheduling interval. Figure 10 shows the efect of varying the scheduling interval. Throughput improves from � = 2 to � = 4, remains nearly unchanged at � = 8, and decreases at � = 16. This trend reflects the trade-of between adaptation frequency and prefetch opportunity. When the interval is too small, frequent rescheduling leaves less time to overlap model loading with decoding. When the interval is too large, beneficial draft transitions are delayed.

The relatively flat performance between � = 4 and � = 8 suggests that MemSpec does not require overly fine-grained parameter tuning to perform well. This is desirable for practical deployment, where the best interval may vary slightly across platforms and workloads. Overall, these results indicate that a moderate interval provides the best balance, and that MemSpec remains robust over a reasonably wide operating range around the default setting.

Output length. Figure 11 shows the efect of varying output length. MemSpec achieves larger gains as generation becomes longer. Compared with short outputs, longer generations provide more opportunity to exploit intra-sequence variation and amortize the cost of draft adaptation. As a result, runtime draft switching becomes increasingly beneficial as output length grows.

This trend also indicates that the benefit ofmemory-aware scheduling becomes more pronounced as generation complexity increases. Longer outputs typically contain more phase transitions and greater intra-sequence heterogeneity, making static draft selection increasingly suboptimal. In such settings, proactively adapting the resident working set provides greater benefit than in short, relatively homogeneous generations.

![](images/bffe569f42d2a529f66d3c6cfdb557cd412fad720be91d6734cd190cecc9d920.jpg)  
Figure 12. Sensitivity to resident cache capacity � on Jetson AGX Orin.

These results further support the central motivation of MemSpec: dynamic draft adaptation is especially important for longer and more heterogeneous generation workloads, and the advantage of MemSpec scales with the opportunity for adaptation.

Resident cache capacity. We additionally evaluate sensitivity to resident cache capacity �, which controls how many draft models can remain resident simultaneously. Because the Jetson Orin Nano platform used in the main evaluation can hold at most two draft models under the default configuration, we conduct this additional analysis on a largermemory Jetson AGX Orin platform.

Figure 12 shows that MemSpec already achieves most of its performance benefit at relatively small cache capacities. While MAB-Async improves throughput from 1.13× to 1.29× as � increases from 2 to 4, MemSpec shows only modest improvement (from 1.47× to 1.51×). This behavior is consistent with the predictor analysis in Section 4.6, which showed high top-2 recall. Since high-utility drafts are already included in the resident working set in most cases, increasing cache capacity further provides limited additional benefit for MemSpec.

In contrast, MAB-Async benefits more noticeably from larger resident cache capacity. As � increases, reactive draft loading becomes less frequent because the runtime experiences fewer non-resident draft selections. Consequently, the performance gap between MemSpec and MAB-Async decreases slightly at larger �. Nevertheless, MemSpec consistently achieves higher throughput across all evaluated cache capacities, demonstrating that prediction-guided residency management remains beneficial even when more memory is available.

## 4.6 Discussion

Predictor quality. MemSpec does not require perfectly accurate draft prediction to improve throughput. The predictor is primarily used to identify promising drafts for proactive residency management rather than to select the exact optimal draft at every scheduling point.

Table 2. Predictor quality comparison.
<table><tr><td>Input</td><td>Top-1 Acc. (%)</td><td>Top-2 Recall (%)</td></tr><tr><td>Prompt-only</td><td>31.9</td><td>55.8</td></tr><tr><td>Recent-only</td><td>57.1</td><td>79.3</td></tr><tr><td>Prompt + Recent</td><td>71.6</td><td>95.7</td></tr></table>

To evaluate predictor quality, we compare three predictor input configurations: prompt-only, recent-generated-tokensonly, and a combined configuration using both prompt and recent generated tokens.

The combined predictor achieves the best ranking quality, improving top-1 accuracy to 71.6% and achieving a top-2 recall of 95.7%. This indicates that the optimal draft is almost always included in the resident working set. These results support the key design principle of MemSpec: adaptive scheduling mainly requires suficiently accurate ranking to maintain high-utility drafts in the resident set, rather than perfect prediction.

End-to-end latency considerations. Our main evaluation focuses on steady-state iterative decoding and therefore excludes prompt processing and initial model loading. To assess startup overhead, we additionally measure end-to-end latency including prompt processing, target-model prefill, initial draft loading, and draft-model preparation.

Although including startup costs reduces relative gains, MemSpec still reduces end-to-end latency by 32.3% on average over General-Static and by 24.9% over MAB-Async. These results remain consistent with our main findings: the primary benefit of MemSpec comes from reducing repeated draft loading and fallback execution during iterative decoding.

## 5 Related Work

## 5.1 Speculative Decoding and Adaptive Draft Selection

Speculative decoding accelerates autoregressive LLM inference by using a lightweight draft model to propose candidate tokens that are then verified by a larger target model [5, 20]. Prior work has improved this framework along multiple directions, including specialized or distilled draft models [39, 42] and system optimizations that reduce verification overhead or improve pipeline eficiency [23–25, 41].

More recent studies observe that draft efectiveness varies across inputs and even across generation stages within a single sequence. This has led to adaptive speculative decoding methods that select among multiple candidate drafts at runtime rather than relying on a single fixed drafter [14, 18, 22]. In particular, multi-drafter and bandit-based approaches improve acceptance through online exploration and adaptation to observed runtime behavior.

However, these methods primarily optimize which draft to select, implicitly assuming that the chosen draft can be executed immediately. That assumption is often invalid on memory-constrained edge devices, where only a small number of drafts can remain resident and switching to a nonresident draft incurs substantial loading overhead.

MemSpec difers from prior adaptive speculative decoding work by treating draft selection and draft availability as a coupled systems problem. Rather than focusing only on acceptance-rate improvement, it explicitly accounts for residency and switching cost so that adaptive selection translates into throughput gains under tight memory budgets.

## 5.2 Adaptive Model Selection and Online Routing

A broad line of research studies adaptive model selection in settings such as cascaded inference [10, 26, 36] and model routing [6, 7, 10, 33]. These approaches dynamically select among candidate models based on input dificulty, confidence estimates, or runtime feedback to balance accuracy and eficiency. Some methods further employ online learning or bandit algorithms to manage the exploration–exploitation trade-of.

MemSpec is related to this literature in that it also performs runtime model selection, but the underlying systems constraints are fundamentally diferent. Conventional routing formulations typically assume that all candidate models are readily available for immediate execution, and thus optimize only the selection decision itself.

In contrast, MemSpec operates in a setting where model availability is constrained by memory capacity and model loading latency. Unlike conventional adaptive routing settings, speculative decoding requires repeated draft decisions within a single generation trajectory, making switching over head a first-order cost. MemSpec therefore extends adaptive model selection with residency-aware scheduling, separating which drafts should be prepared from which resident draft should be executed immediately.

## 5.3 Memory-Constrained LLM Inference on Edge Devices

Running LLMs on edge devices has motivated extensive work on reducing memory footprint and managing limited memory resources. Existing approaches include model compression (e.g., quantization, distillation) to shrink model size [11, 19, 21, 37] and parameter ofloading across GPU, CPU, and storage tiers [1, 17, 29–31]. These techniques make large-model inference feasible on resource-limited platforms.

Our work is complementary to these eforts but addresses a diferent challenge. Prior memory-optimization methods mainly focus on optimizing the execution of a single large model by reducing its footprint or by moving its parameters more eficiently across memory tiers.

By contrast, MemSpec targets adaptive speculative decod ing with multiple candidate draft models, where the main challenge is not only executing one model eficiently, but deciding which drafts should remain resident, which should be prefetched, and when switching is worthwhile.

In this sense, prior edge inference techniques reduce the per-model memory cost, while MemSpec addresses the runtime scheduling problem of managing multiple drafts under a tight memory budget. These directions are orthogonal and potentially complementary.

## 6 Conclusion

This paper presents MemSpec, a memory-aware runtime system for adaptive draft scheduling in speculative decoding on memory-constrained edge platforms. Through detailed characterization on a Jetson Orin Nano device, we show that although generation heterogeneity creates substantial opportunities for improving token acceptance, adaptive draft switching incurs significant model loading overhead—often exceeding multiple decoding iterations—which limits endto-end throughput gains.

To address this challenge, MemSpec formulates adaptive speculative decoding as a residency-constrained scheduling problem. By combining prediction-guided draft ranking with memory-aware residency management, MemSpec proactively aligns draft selection with runtime availability, allowing decoding to proceed with the best currently resident draft while asynchronously preparing better candidates in the background. Experimental results show that Mem-Spec improves steady-state generation throughput by 40.7% on average over state-of-the-art adaptive baselines, while achieving 95–97% of an oracle dynamic upper bound under the same memory constraints. These results demonstrate that the primary bottleneck in adaptive speculative decoding on edge devices is not draft selection quality, but ensuring timely availability of efective drafts.

## Acknowledgements

This research was supported in part by the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (No. RS-2025-24535034), and in part by the Advanced GPU Utilization Support Program funded by the Korean government (MSIT). Myeonggyun Han is the corresponding author.

## References

[1] Keivan Alizadeh, Seyed Iman Mirzadeh, Dmitry Belenko, S. Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, and Mehrdad Farajtabar. 2024. LLM in a flash: Eficient Large Language Model Inference with Limited Memory. In Proceedings ofthe Annual Meeting ofthe Association for Computational Linguistics (ACL), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Bangkok, Thailand, 12562–12584. doi:10.18653/v1/2024.acl-long.678

[2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry,

Quoc Le, and Charles Sutton. 2021. Program Synthesis with Large Language Models. arXiv:2108.07732

[3] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. 2024. MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. In Proceedings of the International Conference on Machine Learning (ICML). Vienna, Austria.

[4] Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Katz, and Nikolaos Aletras. 2022. LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. In Proceedings ofthe Annual Meeting ofthe Association for Computational Linguistics (ACL), Smaranda Muresan, Preslav Nakov, and Aline Villav icencio (Eds.). Dublin, Ireland, 4310–4330. doi:10.18653/v1/2022.acllong.297

[5] Charlie Chen, Sebastian Borgeaud, Geofrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. Accelerating Large Lan guage Model Decoding with Speculative Sampling. arXiv:2302.01318

[6] Lingjiao Chen, Matei Zaharia, and James Zou. 2022. Eficient Online ML API Selection for Multi-Label Classification Tasks. In Proceedings ofthe 39th International Conference on Machine Learning, Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (Eds.), Vol. 162. Baltimore, Maryland, USA, 3716–3746. htps://proceedings.mlr.press/v162/chen22ad.html

[7] Lingjiao Chen, Matei Zaharia, and James Zou. 2024. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Transactions on Machine Learning Research 2024 (2024). htps://openreview.net/forum?id=cSimKw5p6R

[8] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Fe lipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, An drew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating Large Lan guage Models Trained on Code. arXiv:2107.03374

[9] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168

[10] Jasper Dekoninck, Maximilian Baader, and Martin Vechev. 2025. A Unified Approach to Routing and Cascading for LLMs. In Proceedings ofthe International Conference on Machine Learning (ICML), Vol. 267. Vancouver, Canada. htps://openreview.net/forum?id=AAl89VNNy1

[11] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. OPTQ: Accurate Quantization for Generative Pre-trained Transformers. In Proceedings ofthe International Conference on Learning Representations (ICLR). Kigali, Rwanda. htps://openreview.net/forum?id= tcbBPnfwxS

[12] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. 2024. Omni-MATH: A Universal Olympiad Level Mathematic Benchmark for Large Language Models. arXiv:2410.07985

[13] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring Mathematical Problem Solving With the MATH Dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and

Benchmarks. New Orleans, LA, USA. htps://openreview.net/forum? id=7Bywt2mQsCe

[14] Yunlong Hou, Fengzhuo Zhang, Cunxiao Du, Xuan Zhang, Jiachun Pan, Tianyu Pang, Chao Du, Vincent Y. F. Tan, and Zhuoran Yang. 2025. BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms. In Proceedings of the International Conference on Machine Learning (ICML), Vol. 267. Vancouver, Canada. htps://openreview.net/forum? id=ghkWIlliZ8

[15] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2024. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. arXiv:2403.07974

[16] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams. Applied Sciences 11, 14 (2021). doi:10.3390/app11146421

[17] Sowoong Kim, Eunyeong Sim, Youngsam Shin, YeonGon Cho, and Woongki Baek. 2024. Activation Sequence Caching: High-Throughput and Memory-Eficient Generative Inference with a Single GPU. In Proceedings of the 2024 International Conference on Parallel Architectures and Compilation Techniques (PACT ’24’). New York, NY, USA, 78–90. doi:10.1145/3656019.3676945

[18] Taehyeon Kim, Hojung Jung, and Se-Young Yun. 2025. A Unified Framework for Speculative Decoding with Multiple Drafters as a Bandit. htps://openreview.net/forum?id=5haYLrlyGj

[19] Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, and Se-Young Yun. 2025. DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs. In Proceedings ofthe International Conference on Machine Learning (ICML), Vol. 267. Vancouver, Canada. htps://openreview.net/forum?id=rc65N9xIrY

[20] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from Transformers via Speculative Decoding. In Proceedings of the International Conference on Machine Learning (ICML). Honolulu, Hawaii, USA.

[21] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration. In Proceedings of Machine Learning and Systems (MLSys), P. Gibbons, G. Pekhimenko, and C. De Sa (Eds.), Vol. 6. Santa Clara, CA, USA, 87–100. htps://proceedings.mlsys.org/paper\_files/paper/2024/file/ 42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf

[22] Hongyi Liu, Jiaji Huang, Zhen Jia, Youngsuk Park, and Yu-Xiang Wang. 2026. Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs. In Proceedings of the International Conference on Learning Representations (ICLR). Rio de Janeiro, Brazil. htps://openreview.net/forum?id=JMmljf895g

[23] Tianyu Liu, Yun Li, Qitan Lv, Kai Liu, Jianchen Zhu, Winston Hu, and Xiao Sun. 2025. PEARL: Parallel Speculative Decoding with Adaptive Draft Length. In Proceedings of the International Conference on Learning Representations (ICLR). Singapore. htps://openreview.net/forum?id= QOXrVMiHGK

[24] Jonathan Mamou, Oren Pereg, Daniel Korat, Moshe Berchansky, Nadav Timor, Moshe Wasserblat, and Roy Schwartz. 2024. Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models. In Proceedings ofThe 4th NeurIPS Eficient Natural Language and Speech Processing Workshop, Mehdi Rezagholizadeh, Peyman Passban, Soheila Samiee, Vahid Partovi Nia, Yu Cheng, Yue Deng, Qun Liu, and Boxing Chen (Eds.), Vol. 262. Vancouver, Canada, 456–467. htps://proceedings.mlr.press/v262/mamou24a.html

[25] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. 2024. SpecInfer: Accelerating Large Lan-

guage Model Serving with Tree-Based Speculative Inference and Verification. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (ASPLOS ’24’). New York, NY, USA, 932–949. doi:10.1145/3620666.3651335

[26] Harikrishna Narasimhan, Wittawat Jitkrittum, Ankit Singh Rawat, Seungyeon Kim, Neha Gupta, Aditya Krishna Menon, and Sanjiv Kumar. 2025. Faster Cascades via Speculative Decoding. In Proceedings ofthe International Conference on Learning Representations (ICLR) (Singapore). htps://openreview.net/forum?id=vo9t20wsmd

[27] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. 2022. MedMCQA: A Large-Scale Multi-Subject Multi-Choice Dataset for Medical Domain Question Answering. In Proceedings of the Conference on Health, Inference, and Learning, Gerardo Flores, George H Chen, Tom Pollard, Joyce C Ho, and Tristan Naumann (Eds.), Vol. 174. 415 Main Street, Cambridge, MA USA, 248–260. htps://proceedings. mlr.press/v174/pal22a.html

[28] Ruiyang Qin, Jun Xia, Zhenge Jia, Meng Jiang, Ahmed Abbasi, Peipei Zhou, Jingtong Hu, and Yiyu Shi. 2024. Enabling On-Device Large Language Model Personalization with Self-Supervised Data Selection and Synthesis. In Proceedings ofthe 61st ACM/IEEE Design Automation Conference (DAC ’24’). New York, NY, USA. doi:10.1145/3649329.3655665

[29] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. 2023. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU. In Proceedings of the International Conference on Machine Learning (ICML).

[30] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. 2024. PowerInfer: Fast Large Language Model Serving with a Consumer-Grade GPU. In Proceedings ofthe ACM Symposium on Operating Systems Principles (SOSP). New York, NY, USA, 590–606. doi:10.1145/3694715.3695964

[31] Ruslan Svirschevski, Avner May, Zhuoming Chen, Beidi Chen, Zhihao Jia, and Max Ryabinin. 2024. SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices. In Advances in Neural Information Processing Systems (NeurIPS), A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tom czak, and C. Zhang (Eds.), Vol. 37. Vancouver, Canada, 16342– 16368. htps://proceedings.neurips.cc/paper\_files/paper/2024/file 1d91d5689e251d27993a3c2182dddcf7-Paper-Conference.pdf

[32] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford Alpaca: An Instruction-Following LLaMA Model. htps:// github.com/tatsu-lab/stanford\_alpaca.

[33] Ben Taylor, Vicent Sanz Marco, Willy Wolf, Yehia Elkhatib, and Zheng Wang. 2018. Adaptive Deep Learning Model Selection on Embedded Systems. In Proceedings ofthe 19th ACM SIGPLAN/SIGBED International Conference on Languages, Compilers, and Tools for Embedded Systems (LCTES ’18) (Philadelphia, PA, USA). 31–43. doi:10.1145/3211332. 3211336

[34] Qwen Team, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical Report. arXiv:2412.15115

[35] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Alma hairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288

[36] Neeraj Varshney and Chitta Baral. 2022. Model Cascading: Towards Jointly Improving Eficiency and Accuracy of NLP Systems. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing (EMNLP), Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Abu Dhabi, United Arab Emirates, 11007–11021. doi:10.18653/v1/2022.emnlp-main.756

[37] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. SmoothQuant: Accurate and Eficient Post-Training Quantization for Large Language Models. In Proceedings of the International Conference on Machine Learning (ICML), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.), Vol. 202. Honolulu, Hawaii, USA, 38087– 38099. htps://proceedings.mlr.press/v202/xiao23c.html

[38] Minghao Yan, Saurabh Agarwal, and Shivaram Venkataraman. 2025. Decoding Speculative Decoding. In Proceedings ofthe Annual Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics (NAACL), Luis Chiruzzo, Alan Ritter, and Lu Wang (Eds.). Albuquerque, New Mexico, 6460–6473. doi:10.18653/ v1/2025.naacl-long.328

[39] Euiin Yi, Taehyeon Kim, Hongseok Jeung, Du-Seong Chang, and Se-Young Yun. 2024. Towards Fast Multilingual LLM Inference: Speculative Decoding and Specialized Drafters. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing (EMNLP), Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Miami, Florida, USA, 10789–10802. doi:10.18653/v1/2024.emnlp-main.602

[40] Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, and Mengwei Xu. 2025. EdgeMoE: Empowering Sparse Large Language Models on Mobile Devices. IEEE Transactions on Mobile Computing 24, 8 (2025), 7059–7073. doi:10.1109/TMC.2025.3546466

[41] Ziyin Zhang, Jiahao Xu, Tian Liang, Xingyu Chen, Zhiwei He, Rui Wang, and Zhaopeng Tu. 2025. Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Suzhou, China, 16696–16708. doi:10.18653/v1/2025.emnlp-main.844

[42] Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, and Rishabh Agarwal. 2024. DistillSpec: Improving Speculative Decoding via Knowledge Distillation. In Proceedings ofthe International Conference on Learning Representations (ICLR). Vienna, Austria. htps: //openreview.net/forum?id=rsY6J3ZaTF
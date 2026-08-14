# LongEarth-R1: Benchmarking and Aligning Vision-Language Models for Long-Horizon Earth Observation Reasoning

Yupan Ding<sup>1</sup>, Jing Xiao<sup>1</sup>, Zhenyuan Zhang<sup>2</sup>, Chaofeng Chen<sup>1</sup>,

Liang Liao<sup>3</sup>, Gui-Song Xia<sup>1</sup>, Mi Wang<sup>4</sup>

<sup>1</sup>School of Artificial Intelligence, Wuhan University, Wuhan, China

<sup>2</sup>School of Computer Science, Wuhan University, Wuhan, China

<sup>3</sup>Xi’an University of Electronic Science and Technology, Xi’an, China

<sup>4</sup>State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing, Wuhan University, Wuhan, China

## Abstract

Long-horizon Earth observation reasoning requires models to organize multi-stage geographic evolution, localize spatial changes, detect temporal anomalies, and infer future from extended image sequences. However, existing remote sensing vision-language models mainly focus on isolated images, image pairs, or short sequences, limiting reliable grounding in the relevant frames and regions. We introduce LongEarth-Bench, a benchmark containing approximately 120k questionanswering samples derived from 117k unique images. Its sequences average 15.14 frames and extend to 30 frames, covering 12 tasks across evolution summarization, spatial reasoning, anomaly identification, and logical prediction. A 30ksample subset further provides structured reasoning traces linking key frames and changed regions to final answers. We develop LongEarth through supervised fine-tuning with explicit sequence identifiers and structured chain-of-thought supervision. Building on LongEarth, LongEarth-R1 applies group relative policy optimization with format, temporal, and spatial rewards. LongEarth-R1 achieves the best results on all 12 long-sequence tasks while remaining competitive on standard remote sensing benchmarks.

## Introduction

Remote sensing analysis is increasingly moving beyond isolated observations toward understanding how geographic regions evolve over time (Weng, Pang, and Xia 2025). With the growing availability of high-revisit Earth observation imagery, it is now possible to observe multi-stage processes such as urban construction, disaster evolution, and ecosystem recovery (Liu et al. 2026a; Irvin et al. 2025; Jain et al. 2026). For example, given observations before, during, and after a flood, a model should identify its onset and afected regions, track its evolution, and infer the recovery stage. This requires more than pairwise change detection: the model must recover a temporal trajectory, localize supporting frames and regions, and separate geographic evolution from seasonal and acquisition-induced variations (Li et al. 2024; Soni et al. 2025; Luo et al. 2026). We refer to this capability as longhorizon Earth observation reasoning: reasoning about multistage geographic evolution from extended Earth observation sequences.

Recent remote sensing vision-language models (RSVLMs) have progressively extended language-guided interpretation from individual observations to image pairs and temporal sequences. Single-temporal models connect an individual observation with language through image description, visual question answering, and visual grounding (Zhang et al. 2024a; Hu et al. 2025). Bi-temporal models further compare paired observations to describe and localize geographic changes (Yang et al. 2025; Liu et al. 2024a). More recent temporal RSVLMs accept multiple observations and support sequence-level dialogue or change understanding (Irvin et al. 2025; Soni et al. 2025; Xuan et al. 2025). Nevertheless, these capabilities are still insuficient for long-horizon reasoning. A single image contains no evolution trajectory, an image pair exposes mainly the endpoints of change, and current sequence models generally emphasize recognition or descriptive question answering rather than reconstructing a complete multi-stage process. They consequently remain limited in tracing critical transitions, grounding conclusions across frames and regions, and inferring how an evolving process may continue.

To address these limitations, we formulate long-horizon Earth observation reasoning as reasoning over extended sequences. This capability requires models to identify how geographic entities evolve, localize and characterize their changes, assess whether the observed evolution is temporally consistent, and infer unobserved states. We therefore organize it into four progressive cognitive dimensions: 1) Evolution summarization organizes major stages into coherent temporal trajectories; 2) Spatial reasoning localizes changes and models their evolving spatial relations; 3) Anomaly identification detects chronological violations, repetitions, and contextual interference; and 4) Logical prediction infers subsequent or missing states from accumulated evidence. Figure 1(a) illustrates this hierarchy through 12 fine-grained tasks.

Based on the four cognitive dimensions, we construct LongEarth-Bench, a large-scale benchmark for longhorizon remote sensing spatiotemporal reasoning. It contains approximately 120k question-answering (QA) samples derived from 117k images, with an average sequence length of 15.14 frames. The benchmark covers diverse geographic regions, land-cover categories, temporal spans, change rates, and multi-stage processes. Its QA samples are generated from segmentation annotations, geometric relations, and temporal change trajectories. Moreover, 30k samples contain structured reasoning annotations connecting frame selection, temporal localization, spatial evidence, and final conclusions. Figure 1(b) compares LongEarth-Bench with existing benchmarks according to average sequence length and coverage of the four cognitive dimensions.

![](images/c8fc11512339073f030fc998d1edacfde8baefe298b795a87b68ce500af29d1d.jpg)  
Figure 1: Task taxonomy, benchmark positioning, and model capability comparison of LongEarth-Bench and LongEarth-R1. (a) Representative examples of the 12 tasks grouped into four cognitive dimensions. (b) Average sequence length versus cognitive-dimension coverage; bubble size denotes maximum sequence length. (c) Capability coverage of representative remote sensing methods, including long-sequence understanding, evolution description, spatial localization, temporal diagnosis, logical prediction, and CoT.

To enable the RSVLMs with long-horizon spatiotemporal reasoning, we first develop LongEarth through supervised fine-tuning with explicit sequence identifiers and structured chain-of-thought (CoT) traces that connect key frames, temporal changes, and changed regions to final answers. The former establishes stable frame-level temporal anchors, while the latter teaches the model to select relevant observations and integrate temporal and spatial evidence across the sequence. Building on LongEarth, LongEarth-R1 applies group relative policy optimization (GRPO) with complementary format, temporal, and spatial rewards to optimize output completeness, evolution-stage localization, temporal ordering, key-frame selection, and changed-region consistency beyond final-answer correctness. Together, LongEarth and LongEarth-R1 establish reproducible supervised and reinforcement-learning baselines on LongEarth-Bench. Figure 1(c) compares LongEarth-R1 with representative remote sensing methods across six model capabilities, with LongEarth-R1 covering all six.

The main contributions of this paper are as follows.

• We formulate long-horizon Earth observation reasoning as reasoning over multi-stage geographic evolution through four cognitive dimensions covering evolution summarization, spatial reasoning, anomaly identification, and logical prediction.

• We introduce LongEarth-Bench, a large-scale benchmark with approximately 120k samples. It covers diverse geographic processes and includes 30k samples with structured annotations for evidence-grounded reasoning.

• We develop LongEarth through supervised fine-tuning and LongEarth-R1 through GRPO-based temporal and spatial rewards, and systematically evaluate current RSVLMs across sequence lengths, evolution processes, and reasoning dimensions.

## Related Work

## Single- to Multi-Temporal RSVLMs

RSVLMs have progressed from static image-language alignment to temporal Earth observation understanding. Early models such as RemoteCLIP (Liu et al. 2024b), GeoRSCLIP (Zhang et al. 2024c), GeoChat (Kuckreja et al. 2024), and EarthGPT (Zhang et al. 2024a) focus on aligning a single observation with language. VHM (Pang et al. 2025) extends this capability to scene classification, visual question answering, and visual grounding, while Earth-VQA (Wang et al. 2024a) advances relational visual question answering and spatial reasoning. Beyond remote sensing, SpaceVLLM (Wang et al. 2026) studies frame-specific spatiotemporal video grounding. RSICCformer (Liu et al. 2022), GeoLLaVA (Elgendy et al. 2024), Change-Agent (Liu et al. 2024a), and CCExpert (Wang et al. 2024b) extend languagebased analysis to bi-temporal image pairs through change captioning, detection, or diference-aware integration. DisasterM3 (Wang et al. 2025) provides a bi-temporal remote sensing vision-language dataset and benchmark for disaster assessment and response. However, existing bi-temporal settings cannot explicitly represent the intermediate stages of an evolving geographic process.

Recent multi-temporal RSVLMs process extended Earth observation sequences. EarthDial (Soni et al. 2025) supports multispectral, multi-temporal, and multi-resolution conversational inputs, TEOChat (Irvin et al. 2025) enables dialogue and question answering over temporal observations, using sequences that average 2.07 frames and extend to at most 8 frames, and UniRS (Li et al. 2024) unifies singleimage, bi-temporal, and video tasks. TimeSenCLIP (Jain et al. 2026) learns spectral-temporal representations from Sentinel-2 time series, while DynamicVL (Xuan et al. 2025) benchmarks dynamic city understanding with multitemporal scenes averaging 6.73 frames and extending to at most 10 frames, while VLRS-Bench (Luo et al. 2026) evaluates cognition, decision, and prediction with sequences averaging 1.59 frames and covering up to eight temporal phases. These studies establish important foundations for multi-temporal remote sensing understanding. However, extended trajectories spanning many evolution stages remain less studied, particularly when evaluation requires anomaly localization, cross-stage spatial reasoning, and identification of the observations that support a conclusion.

![](images/47c12f163b16ea1543726c8180ca09635e9f8466f339d69fea8937bbd12ba5ac.jpg)

![](images/3b4ee8230b2426fae21b45fd92b4daf68473c9db00c7e5f134a56ce70f7b3a38.jpg)

(a) Sample composition by dimension and subtask  
![](images/040c0dbb91176a690a942af2a5746b3dcecc72904c9b0386288f71d93e1e835e.jpg)  
(c) Sequence-length distribution by source dataset

![](images/d4443b9ab9a50a647c499458b1e9c33c824a4db7a7c5180194663240ccff4465.jpg)  
(b) Global geographic coverage of source datasets  
(d) Overall sequence-length distribution and coverage

![](images/6452f8ba51d512ee778468f39070716b658615887d27dd256c081006bcb66686.jpg)  
(e) Subtask-specific sequence-length distribution  
Figure 2: LongEarth-Bench composition, geographic coverage, and sequence-length statistics. (a) Sample distribution across four cognitive dimensions (inner ring) and 12 tasks (outer ring). (b) Geographic coverage; colors denote the five source datasets and marker size the local mean sequence length. (c) Source-specific length distributions with annotated means. (d) Overall length distribution; bars show counts and the curve gives the fraction of sequences at least each length. (e) Task-conditioned length, with rows normalized within each subtask.

## Reinforcement Learning for Reasoning in RSVLMs

Recent RSVLMs have incorporated structured reasoning and reinforcement learning to support more interpretable and evidence-grounded geospatial analysis. Multimodal-CoT separates rationale generation from answer inference (Zhang et al. 2024b), while Geo-CoT constructs perceptually grounded reasoning traces and trains RSThinker through supervised fine-tuning followed by GRPO (Liu et al. 2026b). Beyond remote sensing, IAD-R1 (Li et al. 2026b) similarly combines CoT-based supervised fine-tuning with structured GRPO for consistent vision-language reasoning. SAMChat (Köksal and Alatan 2026) adopts CoT supervision and GRPO for small-scale remote sensing analysis, while RemoteReasoner (Yao et al. 2026) applies reinforcement learning to a unified workflow covering object-, region-, and pixel-level geospatial reasoning. GeoReason (Li et al. 2026a) aligns reasoning and answers through logical-consistency reinforcement learning, while GeoChain (Yerramilli et al. 2025) studies multimodal CoT geographic reasoning. In parallel, VLRS-Bench evaluates complex reasoning over singletemporal and multi-temporal remote sensing inputs (Luo et al. 2026).

These studies show the value of intermediate supervision and task-specific rewards for reasoning beyond final-answer prediction. However, existing methods primarily reason over individual observations or restricted temporal settings, without explicitly aligning intermediate reasoning with evidence distributed across long Earth observation sequences.

## The LongEarth-Bench Dataset

LongEarth-Bench defines this hierarchy through 12 finegrained tasks. We abbreviate the four cognitive dimensions as evolution summarization (EvoSum), spatial reasoning (Spatial), anomaly identification (AnomID), and logical prediction (LogPred). Figure 2(a) reports the sample distribution. The 120,367 samples are broadly balanced across EvoSum (26.6%), Spatial (22.8%), AnomID (26.7%), and LogPred (23.9%), while preserving diversity across the 12 tasks.

## Dataset Construction

Figure 3 presents the construction pipeline of LongEarth-Bench: multi-source data integration, sequence filtering and cognitive task construction, followed by quality-controlled structured reasoning annotation. Specifically, LongEarth-Bench is derived from SpaceNet 7 (SN7) (Van Etten et al. 2021), SDSU MidWest Flood (SDSU) (Jang et al. 2024), DynamicEarthNet (DynEarth) (Toker et al. 2022), FLAIR#2 (FLAIR) (Garioud et al. 2023), and PASTIS-R (PASTIS) (Sainte Fare Garnot, Landrieu, and Chehata 2022). These sources cover urban construction, flood dynamics, natural land-cover evolution, cloud–snow interference, and crop phenology. As shown in Figure 2(b), they span North America, South America, Europe, Africa, Asia, and Oceania, reducing dependence on a single geographic region or evolution process. Coordinate systems are normalized, while spatial resolutions, temporal metadata, and annotation formats are standardized to construct temporally ordered and spatially aligned remote sensing image sequences. Samples with severe observation corruption, incomplete temporal coverage, registration failures, or insuficient long-term variation are filtered.

![](images/28aff65d6b115d7bff20f0fe42b58d6c766aaa24f6940e7ec32148b1dfa146ed.jpg)  
Figure 3: Construction pipeline of LongEarth-Bench.

Task annotations are generated from temporal trajectories together with segmentation, polygon, land-cover, and image-level flood evidence. Stage boundaries, trends, locations, directions, extents, observation quality, and phenological states are derived from cross-frame changes. Controlled order perturbations, repeated-frame insertions, and contextual disturbances are used to construct anomaly tasks, while prediction tasks are formed from historical trajectories and adjacent temporal states. Rule-based validation verifies sequence integrity, answer consistency, frame indices, spatial labels, and image paths, followed by human review for visual support and ambiguity.

To complement answer-level supervision, we construct a balanced subset of 30k structured reasoning samples across the 12 tasks. Initial traces are generated by Qwen3-VL-8B-Thinking (Bai et al. 2025a) in a unified format: the <think> field contains visual scanning, feature identification, and integrated analysis, and the <answer> field retains the reference answer. We automatically verify tag completeness, stage ordering, non-empty reasoning fields, and answer consistency; invalid outputs are regenerated. Human experts then assess visual grounding, chain coherence, and potential answer leakage. The resulting subset provides reliable processlevel supervision for learning temporal stages, spatial dynamics, anomalies, and logical prediction.

## Statistical Analysis of LongEarth-Bench

Figure 2(c) represents complementary temporal regimes across the five data sources. SDSU provides the shortest sequences, with a mean length of 6.24 frames, whereas Dyn-Earth provides the longest, averaging 21.31 frames. SN7, FLAIR, and PASTIS cover intermediate and long contexts, with mean lengths of 14.15, 15.84, and 18.56 frames, respectively. These source-specific distributions preserve the temporal characteristics of disaster events, urban construction, natural-surface evolution, and crop phenology.

As shown in Figure 2(d), LongEarth-Bench contains 120,367 samples in total with an average sequence length of 15.14 frames, a median of 16 frames, and a maximum of 30 frames. Compared with TEOChatlas (Irvin et al. 2025), DVL-Instruct (Xuan et al. 2025), and VLRS-Bench (Luo et al. 2026), LongEarth-Bench is 7.3×, 2.2×, and 9.5× longer on average, and supports maximum sequences that are 3.75×, 3.0×, and 3.75× longer, respectively. Moreover, 21.1% of the samples contain at least 24 observations. LongEarth-Bench therefore covers a broad spectrum from compact event sequences to extended multi-stage trajectories, providing a controlled setting for evaluating sequencegrounded long-horizon reasoning.

Figure 2(e) summarizes the sequence-length distribution for each task of LongEarth-Bench and shows that all 12 tasks cover multiple sequence-length intervals, while their dominant temporal horizons difer according to their evidence requirements. Change-direction reasoning and spatial-location prediction rely on extended observations, with 53% and 57% of their samples falling within 22–25 frames, respectively. In contrast, contextual robustness and extent assessment contain more compact event-centered sequences. The substantial overlap across tasks also prevents sequence length from serving as a simple task shortcut.

## Method

This section presents a two-stage framework for long-horizon spatiotemporal reasoning. As shown in Figure 4, the framework is built on Qwen2.5-VL-7B (Bai et al. 2025b). Stage 1 applies supervised fine-tuning (SFT) with two complementary forms of supervision. Sequence-aware answer supervision uses explicit sequence identifiers to establish stable frame-level temporal anchors, while structured CoT supervision teaches the model to select relevant observations and integrate temporal and spatial evidence across the sequence, yielding LongEarth. Stage 2 initializes from LongEarth and applies GRPO with complementary format, temporal, and spatial rewards to improve reasoning structure and spatiotemporal consistency, yielding LongEarth-R1.

## Supervised Sequence Grounding and Reasoning

The first stage contains two supervised components. The first component performs sequence-aware answer supervision, which adapts the base model to long-term remote sensing inputs and establishes frame-level temporal anchors. The second component injects structured reasoning traces, which further guides the model to organize cross-frame evidence before producing the final answer.

Sequence-Aware Answer Supervision. This component adapts the base model to long-term remote sensing data and establishes the basic mapping from image sequences and questions to answers. Given a remote sensing sequence X = $\{ I _ { 1 } , \ldots , I _ { T } \}$ with T temporal observations and a question q, the model is required to generate an answer a based on the full sequence. Since long-term tasks require both singleframe recognition and cross-frame change understanding, we assign each image an explicit sequence identifier (Seq. ID), such as Image 1, Image $2 , . . . ,$ Image T.

![](images/124a3693ce3c5a0d62ac3546bd7546f898993592a2fbead0867c5d14b6004de3.jpg)

![](images/0e568da683af5ac79726e492c189575d91f811669f42d9f518b1fb70be11b93e.jpg)  
Figure 4: Overview of LongEarth and LongEarth-R1. Stage 1 trains LongEarth through sequence-aware supervised fine-tuning with explicit sequence identifiers (Seq. IDs) and structured reasoning traces from the 30k-sample subset. Stage 2 initializes from LongEarth and applies GRPO to obtain LongEarth-R1 with format, temporal, and spatial rewards.

Let $V _ { t } = f _ { v } ( I _ { t } )$ denote the visual tokens of the t-th frame encoded by the visual encoder $f _ { v }$ . The ordered multimodal input is represented as:

$$
H = [ q ; \mathrm { I m a g e } 1 , V _ { 1 } ; \ldots ; \mathrm { I m a g e } T , V _ { T } ] .\tag{1}
$$

This representation provides stable frame-level anchors and reduces ambiguity in key-frame reference, temporal interval localization, and cross-frame relation modeling. During training, the visual encoder is frozen, while low-rank adaptation (LoRA) modules are applied to the language model layers for parameter-eficient alignment (Hu et al. 2022). For answer-only samples, this objective mainly constrains the final answer and does not explicitly supervise how the model selects cross-frame evidence or organizes intermediate reasoning.

Structured CoT Supervision. Within the supervised stage, we use the structured reasoning subset of LongEarth-Bench to supervise evidence-grounded responses. For each reasoning sample $( X , q , r , a )$ , the target response contains a reasoning trace r and final answer $^ { a , }$ , formatted as <think> $r < / \mathrm { t h i n k } > < \tt a n s w e r > a < / \tt a n s w e r >$

The reasoning trace follows three steps: visual scanning, feature identification, and integrated analysis. Visual scanning describes the main land-cover states and spatial layouts across temporal observations. Feature identification extracts key frames, changed regions, directions, or anomalies related to the question. Integrated analysis combines crossframe evidence and produces the final conclusion. This design shifts the model from direct answer learning to processlevel supervision. Specifically, given the target sequence $y = ( y _ { 1 } , \dotsc , y _ { N } )$ , the structured-reasoning objective is:

$$
\mathcal { L } _ { \mathrm { C o T } } = - \sum _ { n = 1 } ^ { N } \log p _ { \theta } ( y _ { n } \mid y _ { < n } , H ) .\tag{2}
$$

This objective jointly supervises the reasoning trace and final answer, but reference-trace imitation alone cannot directly penalize temporal ordering errors or spatial inconsistencies. The second stage therefore applies GRPO to further optimize structural validity and spatiotemporal grounding.

## Spatiotemporal Alignment via GRPO

Starting from LongEarth, the second stage applies GRPO to obtain LongEarth-R1 and align reasoning with long-term remote sensing objectives (Shao et al. 2024; Shen et al. 2025). Unlike SFT, which fits a single reference output, GRPO compares a group of candidate responses for the same input and updates the policy using relative rewards. It therefore improves answer quality together with reasoning structure and spatiotemporal consistency.

For multimodal input $H ,$ the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ samples G responses $y _ { i } = \left( r _ { i } , a _ { i } \right)$ , each comprising a reasoning trace and final answer. We assign each response a weighted reward:

$$
R _ { i } = \lambda _ { f } R _ { i } ^ { \mathrm { f m t } } + \lambda _ { t } R _ { i } ^ { \mathrm { t i m e } } + \lambda _ { s } R _ { i } ^ { \mathrm { s p a c e } } ,\tag{3}
$$

where the three terms measure format validity, temporal grounding, and spatial consistency, respectively. The format reward checks the required reasoning–answer structure and task-specific answer parsability. The temporal reward combines matching of referenced frames with their chronological consistency, while the spatial reward evaluates agreement on changed locations, directions, and extents. For tasks without temporal or spatial labels, inapplicable terms are omitted and the remaining weights are renormalized; detailed reward definitions are provided in the supplementary material.

<table><tr><td>Category</td><td></td><td>Video-LLaVA Qwen2.5 Qwen3-Thinking TEOChat TEOChat* EarthDial EarthDial*</td><td></td><td></td><td></td><td></td><td></td><td></td><td>LongEarth-R1</td></tr><tr><td rowspan="3">Evolution Summarization</td><td>T1 Temporal Phasing</td><td>24.53</td><td>41.22</td><td>49.44</td><td>32.02</td><td>64.16</td><td>31.84</td><td>46.18</td><td>77.31</td></tr><tr><td>T2 Pattern Classification</td><td>39.32</td><td>46.44</td><td>59.60</td><td>35.81</td><td>49.58</td><td>45.19</td><td>69.90</td><td>83.14</td></tr><tr><td>T3 Stagnation Identification</td><td>10.58</td><td>34.11</td><td>29.73</td><td>35.98</td><td>50.74</td><td>25.92</td><td>46.95</td><td>63.57</td></tr><tr><td rowspan="3">Spatial Reasoning</td><td>T4 Change Direction</td><td>32.72</td><td>43.86</td><td>40.08</td><td>31.92</td><td>40.28</td><td>33.44</td><td>45.80</td><td>53.37</td></tr><tr><td>T5 Significant Region</td><td>34.65</td><td>40.53</td><td>36.47</td><td>31.60</td><td>50.45</td><td>32.39</td><td>62.69</td><td>68.96</td></tr><tr><td>T6 Extent Assessment</td><td>30.76</td><td>39.24</td><td>33.48</td><td>35.03</td><td>58.48</td><td>37.04</td><td>67.58</td><td>74.32</td></tr><tr><td rowspan="3">Anomaly Identification</td><td>T7 Chronological Violations</td><td>2.86</td><td>12.38</td><td>7.87</td><td>7.90</td><td>15.47</td><td>5.37</td><td>15.92</td><td>50.80</td></tr><tr><td>T8 Redundancy Detection</td><td>5.90</td><td>39.78</td><td>36.68</td><td>10.31</td><td>27.57</td><td>14.51</td><td>20.32</td><td>90.28</td></tr><tr><td>T9 Contextual Robustness</td><td>18.60</td><td>35.65</td><td>37.45</td><td>15.51</td><td>48.74</td><td>12.42</td><td>49.12</td><td>89.12</td></tr><tr><td rowspan="3">Logical Prediction</td><td>T10 Trend Prediction</td><td>29.78</td><td>46.78</td><td>32.03</td><td>36.30</td><td>55.44</td><td>20.61</td><td>60.54</td><td>76.42</td></tr><tr><td>T11 Spatial Location Prediction</td><td>47.60</td><td>50.61</td><td>33.77</td><td>28.48</td><td>55.98</td><td>49.85</td><td>87.62</td><td>89.87</td></tr><tr><td>T12 Missing Frame Prediction</td><td>29.69</td><td>33.83</td><td>47.86</td><td>38.40</td><td>58.94</td><td>36.19</td><td>62.04</td><td>64.25</td></tr></table>

Table 1: Performance comparison on the 12 long-sequence cognitive tasks in LongEarth-Bench (%). An asterisk (\*) denotes supervised fine-tuning on LongEarth-Bench. The best result in each row is shown in bold.

GRPO normalizes rewards within the response group as $A _ { i } = ( R _ { i } - \mu _ { R } ) / ( \sigma _ { R } + \epsilon )$ and uses the policy ratio $\rho _ { i } =$ $\pi _ { \boldsymbol { \theta } } ( y _ { i } \mid H ) / \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( y _ { i } \mid H )$ . Let $\bar { \rho } _ { i } = \mathrm { c l i p } ( \rho _ { i } , 1 - \varepsilon , 1 + \varepsilon )$ The clipped objective is

$$
\mathcal { I } _ { \mathrm { G R P O } } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \operatorname* { m i n } ( \rho _ { i } A _ { i } , \bar { \rho } _ { i } A _ { i } ) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) \ : ,\tag{4}
$$

where ε is the clipping coeficient, β is the Kullback–Leibler (KL) weight, and $\pi _ { \mathrm { r e f } }$ is the policy obtained after the supervised stage. The KL term preserves the visual-language ability acquired during SFT. Together, the rewards shift learning from answer imitation toward structured, temporally ordered, and spatially grounded reasoning.

## Experiments

## Experimental Setups

Implementation Details. We train LongEarth-R1 on 8× NVIDIA A800 GPUs, freezing the visual encoder and tuning language-side LoRA modules with r = 128 and α = 256 in both stages. SFT runs for two epochs with bfloat16, gradient checkpointing, cosine decay, a peak learning rate o $2 \times 1 0 ^ { - 5 }$ a 0.03 warmup ratio, and no weight decay. Inputs are limited to 8,192 tokens and interleave square-resized images with temporal prefixes and instructions. GRPO starts from the SFT checkpoint and optimizes the same LoRA modules with group-sampled responses and the proposed rewards.

Evaluation Metrics. We evaluate final answers using Accuracy, Temporal F1, and Spatial F1. Accuracy applies to closed-set answers, including categorical judgments, singleframe localization, fixed spatial labels, and discrete extent levels. Temporal and free-form spatial answers use setbased F1 over predicted and reference elements. Temporal F1 operates on parsed frame indices, whereas Spatial F1 uses S = {object, change, region, direction, extent}. Temporal F1 evaluates T1, T3, T8, and temporal-set variants of T7, T9, and T12; Spatial F1 evaluates free-form T4–T6 and T11. Remaining tasks use Accuracy.

![](images/bbd58abe5d41abc2826356f1207e748c08401843fe1f75f9f117ca18e568b58c.jpg)  
Figure 5: Qualitative comparison on long-horizon temporal reasoning. EarthDial and TEOChat predict an incorrect interval, while LongEarth-R1 identifies the correct low-water period and grounds its answer in multi-frame visual evidence.

## Experimental Results

Performance on LongEarth-Bench. We evaluate the 12 LongEarth-Bench tasks spanning EvoSum, Spatial, AnomID, and LogPred. Table 1 shows that LongEarth-R1 ranks first on all tasks, with particularly strong gains on AnomID and tasks requiring long-range temporal evidence. The consistent improvements show that sequence grounding, structured reasoning, and reward-based alignment jointly strengthen temporal understanding, spatial grounding, and prediction. Figure 5 further shows that LongEarth-R1 avoids temporally misaligned intervals by grounding its answer in multi-frame evidence.

Performance on Standard Remote Sensing Tasks. We further test whether specialization for long-term reasoning preserves general remote sensing understanding. Following the standard evaluation protocol (Irvin et al. 2025), we evaluate single-image scene recognition on AID and UCM (Xia et al. 2017; Yang and Newsam 2010), bi-temporal change understanding on ABCD-CD, CDVQA-QA, xBD, and S2Looking (Fujita et al. 2017; Yuan et al. 2022; Gupta et al. 2019; Shen et al. 2021), and short-sequence multiimage reasoning on Qfabric and fMoW (Verma, Panigrahi, and Gupta 2021; Christie et al. 2018). For each benchmark,

<table><tr><td>Category</td><td>Dataset</td><td>Human</td><td>Video-LLaVA</td><td>GeoChat</td><td>Qwen2.5</td><td>EarthDial</td><td>TEOChat</td><td>LongEarth-R1</td></tr><tr><td rowspan="2">Single image</td><td>AID</td><td>一</td><td>52.40</td><td>72.00</td><td>65.60</td><td>88.67</td><td>80.90</td><td>86.03</td></tr><tr><td>UCM</td><td>1</td><td>46.50</td><td>84.40</td><td>70.70</td><td>80.48</td><td>86.30</td><td>90.19</td></tr><tr><td rowspan="4">Bi-temporal</td><td>ABCD-CD</td><td>95.20</td><td>50.00</td><td>一</td><td>69.80</td><td>63.46</td><td>85.60</td><td>91.20</td></tr><tr><td>CDVQA-QA</td><td>63.40</td><td>29.80</td><td>1</td><td>50.50</td><td>47.47</td><td>47.20</td><td>56.60</td></tr><tr><td>xBD (Avg.)</td><td>56.20</td><td>21.20</td><td>25.10</td><td>36.80</td><td>24.78</td><td>59.60</td><td>71.50</td></tr><tr><td>S2Looking (Avg.)</td><td>26.50</td><td>24.80</td><td>32.70</td><td>8.20</td><td>29.70</td><td>57.70</td><td>60.50</td></tr><tr><td rowspan="3">Multi-image (≤ 8)</td><td>Qfabric (Avg.)</td><td>71.00</td><td>17.60</td><td>18.40</td><td>23.70</td><td>33.59</td><td>70.80</td><td>69.70</td></tr><tr><td>fMoW-LR-TSC</td><td></td><td>4.90</td><td>26.30</td><td>0.10</td><td>24.08</td><td>45.50</td><td>45.60</td></tr><tr><td>fMoW-HR-TSC</td><td>65.90</td><td>16.60</td><td>59.20</td><td>28.50</td><td>60.86</td><td>75.10</td><td>72.90</td></tr></table>

Table 2: Performance on single-image, bi-temporal, and short-sequence multi-image tasks (%). - denotes unavailable results.

<table><tr><td rowspan="13">Comt son</td><td>SFT Seq.IDs CoT</td><td></td><td></td><td>GRPO</td><td>EvoSum</td><td>Spatial</td><td>AnomID</td><td>LogPred</td></tr><tr><td>x</td><td>x</td><td>x</td><td>x</td><td>40.59</td><td>41.21</td><td>29.27</td><td>43.74</td></tr><tr><td>√</td><td>x</td><td>x</td><td>x</td><td>60.77</td><td>60.27</td><td>29.26</td><td>72.49</td></tr><tr><td>√</td><td>√</td><td>x</td><td>x</td><td>70.46</td><td>59.43</td><td>72.93</td><td>73.82</td></tr><tr><td>√</td><td>√</td><td>√</td><td>x</td><td>69.47</td><td>59.08</td><td>64.12</td><td>70.81</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>74.67</td><td>65.55</td><td>76.73</td><td>76.85</td></tr><tr><td>Format</td><td>Temporal</td><td></td><td>Spatial</td><td>EvoSum</td><td>Spatial</td><td>AnomID</td><td>LogPred</td></tr><tr><td rowspan="4">Rewa laton</td><td></td><td>√</td><td>√</td><td>63.74</td><td>63.97</td><td>68.01</td><td>77.33</td></tr><tr><td>x √</td><td>x</td><td>√</td><td>59.83</td><td>64.22</td><td>67.02</td><td>76.00</td></tr><tr><td>√</td><td>√</td><td>x</td><td>61.50</td><td>62.62</td><td>73.05</td><td>70.67</td></tr><tr><td>√</td><td>√</td><td>√</td><td>74.67</td><td>65.55</td><td>76.73</td><td>76.85</td></tr></table>

Table 3: Ablation results of the core components (%). The upper block studies cumulative training components, and the lower block removes individual rewards.

LongEarth-R1 is fine-tuned on the corresponding training split and evaluated using the task-native protocol; detailed sources and task descriptions are provided in the supplementary material.

Table 2 shows that LongEarth-R1 achieves the best result on six of nine datasets, including all bi-temporal changeunderstanding benchmarks, indicating efective transfer from long-horizon supervision to conventional change analysis. Its performance remains close to the strongest specialist on the remaining benchmarks, showing that the proposed training preserves general remote sensing understanding across single-image, bi-temporal, and short-sequence settings.

Temporal Robustness Analysis. Figure 6 separates two sources of temporal dificulty: the number of input frames processed by the model and the temporal span of evidence required by the question. In the top panel, all methods degrade as input sequences become longer, reflecting the increasing need to suppress irrelevant observations and localize the informative frames. LongEarth-R1 nevertheless leads every input-length interval by 10.7–31.9%, retaining 51.4 at 26–30 frames after reaching 87.8 on 2–5-frame inputs. This trend indicates that explicit sequence grounding and reward-based alignment improve robustness to long-context distraction, although very long inputs remain challenging.

The bottom panel shows a diferent pattern. LongEarth-R1 achieves the best score in six of seven ground-truth evidencespan intervals, and its performance generally improves as the answer can be supported by evidence distributed across a broader temporal span. Thus, a broad evidence span is not necessarily harmful: when multiple observations provide complementary evolution cues, cross-frame reasoning can

![](images/50439ebb4d6e2b364d9f9e5e44beb21fc4a3d257d78a7d26a72dbe175272e0e4.jpg)  
Figure 6: Task-macro scores by input length (top) and ground-truth evidence span (bottom); \* denotes LongEarth-Bench fine-tuning. benefit from them. The only exception is the 23–30-frame interval, where TEOChat\* attains a higher score.

Ablation Analysis of Core Components. We conduct cumulative component and reward ablations across the four cognitive dimensions of LongEarth-Bench. The component study progressively adds SFT, Seq. IDs, R1, and GRPO, whereas the reward study removes one term at a time from the full objective.

Table 3 shows that the complete configuration, LongEarth-R1, achieves the strongest macro-average and the most balanced performance across all four dimensions. Relative to LongEarth, adding GRPO yields consistent gains, with the largest improvement on AnomID (+12.61), indicating that reward-based optimization strengthens anomalysensitive evidence selection and spatiotemporal reasoning. Removing any reward lowers the macro-average by 5.19– 6.68%; although removing the format reward slightly improves LogPred, the full objective remains best overall, confirming the complementarity of the three rewards.

## Conclusion

Long-term remote sensing understanding requires reasoning over evolving geographic evidence rather than isolated observations. We present LongEarth-Bench, which defines this setting with four cognitive dimensions and structured reasoning supervision. We further develop LongEarth through supervised sequence grounding and LongEarth-R1 through reward-driven spatiotemporal alignment. Results improve performance across the 12 long-sequence tasks while retaining transfer to conventional remote sensing tasks. These findings support explicit modeling of temporal order, intermediate evidence, and spatial consistency for long-horizon Earth observation reasoning.

## References

Bai, S.; Cai, Y.; Chen, R.; et al. 2025a. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; et al. 2025b. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Christie, G.; Fendley, N.; Wilson, J.; and Mukherjee, R. 2018. Functional Map of the World. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 6172–6180.

Elgendy, H.; Sharshar, A.; Aboeitta, A.; Ashraf, Y.; and Guizani, M. 2024. GeoLLaVA: Eficient Fine-Tuned Vision-Language Models for Temporal Change Detection in Remote Sensing. arXiv preprint arXiv:2410.19552.

Fujita, A.; Sakurada, K.; Imaizumi, T.; Ito, R.; Hikosaka, S.; and Nakamura, R. 2017. Damage Detection from Aerial Images via Convolutional Neural Networks. In 2017 Fifteenth IAPR International Conference on Machine Vision Applications, 5–8.

Garioud, A.; De Wit, A.; Poupée, M.; Valette, M.; Giordano, S.; and Wattrelos, B. 2023. FLAIR #2: Textural and Temporal Information for Semantic Segmentation from Multi-Source Optical Imagery. arXiv preprint arXiv:2305.14467.

Gupta, R.; Goodman, B.; Patel, N.; Hosfelt, R.; Sajeev, S.; Heim, E. T.; Doshi, J.; Lucas, K.; Choset, H.; and Gaston, M. E. 2019. Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 10–17.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR.

Hu, Y.; Yuan, J.; Wen, C.; Lu, X.; Liu, Y.; and Li, X. 2025. RSGPT: A Remote Sensing Vision Language Model and Benchmark. ISPRS Journal ofPhotogrammetry and Remote Sensing, 224: 272–286.

Irvin, J. A.; Liu, E. R.; Chen, J. C.; Dormoy, I.; Kim, J.; Khanna, S.; Zheng, Z.; and Ermon, S. 2025. TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data. In ICLR.

Jain, P.; Marcos, D.; Ienco, D.; Interdonato, R.; and Berchoux, T. 2026. TimeSenCLIP: A Time Series Vision-Language Model for Remote Sensing. ISPRS Journal of Photogrammetry and Remote Sensing, 236: 99–119.

Jang, Y.; Kim, D.; Pack, C.; and Won, K. 2024. A Novel Dataset for Flood Detection Robust to Seasonal Changes in Satellite Imagery. In Proceedings of the ACM Research in Adaptive and Convergent Systems Conference.

Köksal, A.; and Alatan, A. A. 2026. SAMChat: Introducing Chain-of-Thought Reasoning and GRPO to a Multimodal Small Language Model for Small-Scale Remote Sensing. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 19: 795–804.

Kuckreja, K.; Danish, M. S.; Naseer, M.; Das, A.; Khan, S.; and Khan, F. S. 2024. GeoChat: Grounded Large Vision-Language Model for Remote Sensing. In CVPR, 27831– 27840.

Li, W.; Xiang, X.; Wen, Z.; et al. 2026a. GeoReason: Aligning Thinking and Answering in Remote Sensing Vision-Language Models via Logical Consistency Reinforcement Learning. arXiv preprint arXiv:2601.04118.

Li, Y.; Cao, Y.; Liu, C.; Xiong, Y.; Dong, X.; and Huang, C. 2026b. IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection. In AAAI, 6583–6591.

Li, Y.; Xu, W.; Li, G.; Yu, Z.; Wei, Z.; Wang, J.; and Peng, M. 2024. UniRS: Unifying Multi-Temporal Remote Sensing Tasks through Vision Language Models. arXiv preprint arXiv:2412.20742.

Liu, C.; Chen, K.; Zhang, H.; Qi, Z.; Zou, Z.; and Shi, Z. 2024a. Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis. IEEE Transactions on Geoscience and Remote Sensing, 62: 1–16.

Liu, C.; Zhang, J.; Chen, K.; Wang, M.; Zou, Z.; and Shi, Z. 2026a. Remote Sensing Spatiotemporal Vision–Language Models: A Comprehensive Survey. IEEE Geoscience and Remote Sensing Magazine, 14(1): 383–423.

Liu, C.; Zhao, R.; Chen, H.; Zou, Z.; and Shi, Z. 2022. Remote Sensing Image Change Captioning with Dual-Branch Transformers: A New Method and a Large-Scale Dataset. IEEE Transactions on Geoscience and Remote Sensing, 60: 1–20.

Liu, F.; Chen, D.; Guan, Z.; Zhou, X.; Zhu, J.; Ye, Q.; Fu, L.; and Zhou, J. 2024b. RemoteCLIP: A Vision Language Foundation Model for Remote Sensing. IEEE Transactions on Geoscience and Remote Sensing, 62: 1–16. Article 5622216.

Liu, J.; Sun, L.; Fu, R.; and Yang, B. 2026b. Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded GeoSpatial Chain-of-Thought for Vision-Language Models. In ICLR.

Luo, Z.; Wang, D.; Guo, H.; Zhang, J.; and Du, B. 2026. VLRS-Bench: A Vision-Language Reasoning Benchmark for Remote Sensing. arXiv preprint arXiv:2602.07045.

Pang, C.; Weng, X.; Wu, J.; Li, J.; Liu, Y.; Sun, J.; Li, W.; Wang, S.; Feng, L.; Xia, G.-S.; and He, C. 2025. VHM: Versatile and Honest Vision Language Model for Remote Sensing Image Analysis. In AAAI, 6381–6388.

Sainte Fare Garnot, V.; Landrieu, L.; and Chehata, N. 2022. Multi-Modal Temporal Attention Models for Crop Mapping from Satellite Time Series. ISPRS Journal ofPhotogrammetry and Remote Sensing, 187: 294–305.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300.

Shen, H.; Liu, P.; Li, J.; Fang, C.; Ma, Y.; Liao, J.; Shen, Q.; Zhang, Z.; Zhao, K.; Zhang, Q.; Xu, R.; and Zhao, T. 2025. VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model. arXiv preprint arXiv:2504.07615.

Shen, L.; Lu, Y.; Chen, H.; Wei, H.; Xie, D.; Yue, J.; Chen, R.; Lv, S.; and Jiang, B. 2021. S2Looking: A Satellite Side-Looking Dataset for Building Change Detection. Remote Sensing, 13(24): 5094.

Soni, S.; Dudhane, A.; Debary, H.; Fiaz, M.; Munir, M. A.; Danish, M. S.; Fraccaro, P.; Watson, C. D.; Klein, L. J.; Khan, F. S.; and Khan, S. 2025. EarthDial: Turning Multi-Sensory Earth Observations to Interactive Dialogues. In CVPR, 14303–14313.

Toker, A.; Kondmann, L.; Weber, M.; et al. 2022. DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation. In CVPR, 21158–21167.

Van Etten, A.; Hogan, D.; Martinez-Manso, J.; Shermeyer, J.; Weir, N.; and Lewis, R. 2021. The Multi-Temporal Urban Development SpaceNet Dataset. In CVPR, 6398–6407.

Verma, S.; Panigrahi, A.; and Gupta, S. 2021. QFabric: Multi-Task Change Detection Dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops.

Wang, J.; Xuan, W.; Qi, H.; Liu, Z.; Liu, K.; Wu, Y.; Chen, H.; Song, J.; Xia, J.; Zheng, Z.; and Yokoya, N. 2025. DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response. In NeurIPS.

Wang, J.; Zhang, Z.; Liu, Z.; Li, Y.; Ge, J.; Xie, H.; and Zhang, Y. 2026. SpaceVLLM: Endowing Multimodal Large Language Model with Spatio-Temporal Video Grounding Capability. In AAAI, 9912–9920.

Wang, J.; Zheng, Z.; Chen, Z.; Ma, A.; and Zhong, Y. 2024a. EarthVQA: Towards Queryable Earth via Relational Reasoning-Based Remote Sensing Visual Question Answering. In AAAI, 5481–5489.

Wang, Z.; Wang, M.; Xu, S.; Li, Y.; and Zhang, B. 2024b. CCExpert: Advancing MLLM Capability in Remote Sensing Change Captioning with Diference-Aware Integration and a Foundational Dataset. arXiv preprint arXiv:2411.11360.

Weng, X.; Pang, C.; and Xia, G.-S. 2025. Vision-Language Modeling Meets Remote Sensing: Models, Datasets and Perspectives. IEEE Geoscience and Remote Sensing Magazine, 13(3): 276–323.

Xia, G.-S.; Hu, J.; Hu, F.; Shi, B.; Bai, X.; Zhong, Y.; Zhang, L.; and Lu, X. 2017. AID: A Benchmark Data Set for Performance Evaluation of Aerial Scene Classification. IEEE Transactions on Geoscience and Remote Sensing, 55(7): 3965–3981.

Xuan, W.; Wang, J.; Qi, H.; Chen, Z.; Zheng, Z.; Zhong, Y.; Xia, J.; and Yokoya, N. 2025. DynamicVL: Benchmarking Multimodal Large Language Models for Dynamic City Understanding. In NeurIPS.

Yang, C.; Li, Z.; Jiao, H.; Gao, Z.; and Zhang, L. 2025. Enhancing Perception of Key Changes in Remote Sensing Image Change Captioning. IEEE Transactions on Image Processing, 34: 7378–7390.

Yang, Y.; and Newsam, S. 2010. Bag-of-Visual-Words and Spatial Extensions for Land-Use Classification. In Proceedings ofthe 18th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, 270–279.

Yao, L.; Liu, F.; Lu, H.; Zhang, C.; Min, R.; Xu, S.; Di, S.; and Peng, P. 2026. RemoteReasoner: Towards Unifying Geospatial Reasoning Workflow. In AAAI, 11883–11891.

Yerramilli, S.; Pande, N.; Grover, R.; and Tamarapalli, J. S. 2025. GeoChain: Multimodal Chain-of-Thought for Geographic Reasoning. In Findings ofEMNLP, 23624–23639.

Yuan, Z.; Mou, L.; Xiong, Z.; and Zhu, X. X. 2022. Change Detection Meets Visual Question Answering. IEEE Transactions on Geoscience and Remote Sensing, 60: 5630613.

Zhang, W.; Cai, M.; Zhang, T.; Zhuang, Y.; and Mao, X. 2024a. EarthGPT: A Universal Multi-Modal Large Language Model for Multi-Sensor Image Comprehension in Remote Sensing Domain. IEEE Transactions on Geoscience and Remote Sensing, 62: 1–20.

Zhang, Z.; Zhang, A.; Li, M.; Zhao, H.; Karypis, G.; and Smola, A. 2024b. Multimodal Chain-of-Thought Reasoning in Language Models. Transactions on Machine Learning Research.

Zhang, Z.; Zhao, T.; Guo, Y.; and Yin, J. 2024c. RS5M and GeoRSCLIP: A Large-Scale Vision-Language Dataset and a Large Vision-Language Model for Remote Sensing. IEEE Transactions on Geoscience and Remote Sensing, 62: 1–23. Article 5642123.
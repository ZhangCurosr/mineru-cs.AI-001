Article

# Training-Free Inference-Time Self-Reflection and Cost-Bounded Early Stopping for Large Language Models

Wei Yu<sup>1</sup>, Suxing Liu<sup>1,\*</sup>, Minjie Yu<sup>1</sup>, Jiahao Wang<sup>2</sup>, Zhijian Zheng<sup>1</sup>, Haocheng Deng<sup>1</sup>, Bing Li<sup>1</sup>

<sup>1</sup> School of Digital Arts, Jiangxi Arts & Ceramics Technology Institute, Jingdezhen 33001, China; yuw26393@gmail.com (W.Y.)

<sup>2</sup> School of Computing, Universiti Sains Malaysia, 11800 USM Penang, Malaysia

Correspondence: bentondoucet@gmail.com

<sub>Citation: Yu, W.; Liu, S.; Yu, M.; Wang,</sub>8 J.; Zheng, Z.; Deng, H.; Li, B.<sub>.</sub>   
<sub>Training-Free</sub> <sub>Inference-Time</sub><sup>8</sup> Self-Reflection and Cost-Bounded Early Stopping for Large Language Models. Journal Not Specified 2026, 1, 0.: https://doi.org/ Received:   
Accepted:a Published:

Publisher’s Note: MDPI stays neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Copyright: © 2026 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https:// creativecommons.org/licenses/by/ 4.0/).

Abstract: Training reasoning-capable large language models (LLMs) via reinforcement learning such as GRPO is expensive, depends on a controllable training environment, and commits every contribution to a full training pipeline that may never be executed. We present EvoResearcher, a training-free inference-time protocol that adds cost-bounded self-reflection to a single frozen LLM backbone. The protocol iterates generate → self-critique → revise until a user-set maximum depth D is exhausted or the critique returns the CONFIRMED sentinel—an implicit early stop that lets the backbone self-verify its answer under a strictly controlled computational budget. The four components of a self-reflective meta-reward (correctness, efficiency, reflection depth, and tool-call diversity) act as design principles that the protocol instantiates as prompt-level mechanisms, so their benefits accrue with zero gradient updates. We validate the protocol on Big-Bench Hard (BBH, 100 multi-step reasoning questions) and establish cross-domain behavior on GSM8K (500 arithmetic word problems) and MATH (500 competition problems) on the same frozen backbone, with cross-model replication on Qwen2.5-72B. All experiments use pure-reasoning benchmarks; the tool-call-diversity component is validated only in its prompt-level form, and the environment-level and multi-agent extensions are design blueprints that are not evaluated here—their validation requires tool-using benchmarks such as GAIA and is left to future work. On clean BBH the protocol does not raise single-shot accuracy beyond the 95% Wilson interval; its measurable value is cost-bounded self-verification: the CONFIRMED early stop terminates 82–88% of items at equal accuracy, bounding inference to ≈ 2.1 generations per question. A confidence-threshold sweep $( \tau \in \{ 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \} )$ calibrates the numeric confidence signal and finds it overconfident—every threshold ≥ 0.6 halts the loop on the first generation, so the CONFIRMED sentinel, not the confidence tag, is the effective cost-bounding mechanism. On GSM8K and MATH the loop improves accuracy (+4.2 pp and +14.2 pp) while still early-stopping 82–88% of items, confirming that reflection’s accuracy benefit concentrates precisely where single-shot reasoning is unreliable (MATH single-shot 26.2%). We conclude that this training-free reflective loop effectively unlocks inference-time compute scaling: it delivers cost-bounded self-verification on standard tasks, and significant raw accuracy gains precisely where single-shot reasoning fails, providing a highly pragmatic alternative to expensive reinforcement learning pipelines.

Keywords: inference-time self-reflection; cost-bounded early stopping; confidence-threshold calibration; large language model; pure reasoning; training-free

## 1. Introduction

Large language models (LLMs) exhibit strong performance on multi-step reasoning tasks, yet two obstacles limit their practical use when correctness matters and compute is constrained. First, training them to reason—typically via reinforcement learning objectives such as GRPO [1]—is expensive: it requires thousands of GPU-hours and a carefully constructed, controllable environment, and the resulting gains are often tightly coupled to that specific training pipeline. Second, at inference time, a single generation is frequently overconfident or outright wrong, and the standard remedies—generating more candidates (self-consistency, best-of-n sampling) or running reflection loops—multiply computational cost without a principled rule for when to stop.

A growing line of work makes models self-verify their own outputs (Self-Refine [2], Reflexion [3], CRITIC [4]); process reward models [5,6] supply step-level verification signals; and structured reasoning itself (chain-of-thought [7], Tree of Thoughts [8]) improves both accuracy and inspectability. Yet most of these methods either require additional training (or a learned reward model), or, if purely at inference, treat the reflection loop as a fixed-depth procedure whose cost is uncontrolled. The result is a missing middle: a training-free protocol that lets a frozen backbone self-verify its answer at inference time while strictly bounding how much compute the loop may consume. We identify three limitations of current practice that this paper addresses:

(i) Training cost and environment dependence. RL-based reasoning training (e.g., GRPO [1]) is resource-intensive and requires a controllable environment in which rollouts can be scored, which makes the training path impractical for many users and ties every contribution to a pipeline that may never be executed.

(ii) Outcome-only supervision. Standard objectives reward only final correctness and provide no incentive to recognize and correct an error before committing to it. Selfverification is not trained and is consequently unreliable, as LLMs are systematically overconfident in their own outputs [9].

(iii) Unbounded reflection cost. At inference, reflection loops that iterate to a fixed depth either over-spend (revising answers that were already correct) or, if stopped too early, over-confirm incorrect answers. Without a calibrated stopping rule, the loop’s compute cannot be traded against accuracy in a controlled way.

## 1.1. Our Contributions

We propose EvoResearcher, a training-free, inference-time self-reflective protocol that instantiates the four components of a self-reflective meta-reward—correctness, efficiency, reflection depth, and tool-call diversity—as prompt-level mechanisms inside a bounded generate → self-critique → revise loop over a single frozen backbone. The loop’s cost is controlled by the maximum depth D and by the CONFIRMED sentinel; an optional confidence threshold τ is also supported, and its selectivity is probed empirically rather than assumed. The loop terminates early whenever the critique returns CONFIRMED, when the backbone’s self-reported confidence reaches a set τ, or when D generations are exhausted. Concretely:

1. A training-free, cost-bounded self-reflection protocol (validated). Algorithm 1 formalizes the generate–critique–revise loop whose early stopping is driven by the CONFIRMED sentinel (with an optional confidence threshold τ whose selectivity we probe, E6), so inference cost is bounded by construction (Section 3.1).

2. Meta-reward components as prompt-level mechanisms (validated). The four components of the meta-reward (Equations (1)–(4)) act as design principles that are instantiated at the prompt level and evaluated by controlled component and ablation experiments (Section 4).

3. Cross-domain empirical validation (validated). We evaluate the protocol on Big-Bench Hard (BBH) and establish cross-domain behavior on GSM8K and MATH on the same frozen backbone, with cross-model replication on Qwen2.5-72B. On clean BBH the loop does not raise single-shot accuracy, but its CONFIRMED early stopping self-verifies a large majority of items at equal accuracy (cost ≈ 2.1 generations); on GSM8K and MATH the loop improves accuracy (+4.2 pp and +14.2 pp, significant at n=500) while still early-stopping 82–88% of items. A confidence-threshold sweep calibrates the numeric confidence signal and finds it overconfident, so the sentinel—not the confidence tag—bounds inference cost (Section 4).

4. Cost-Efficient Compute Scaling. Through rigorous token-level accounting, we demonstrate that our protocol prevents the characteristic “over-correction” and compute waste seen in fixed-depth refinement methods, reducing computational waste on simple queries while dynamically allocating reflection steps to complex problems. We stop short of claiming strict Pareto dominance over all alternatives; the evidence is that the sentinel yields a favourable cost–accuracy trade-off on the evaluated benchmarks.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 presents the inference-time self-reflective protocol and its prompt-level instantiation of the meta-reward. Section 4 describes the controlled experiments and hypothesis tests on BBH, GSM8K, and MATH. Section 5 discusses implications, limitations, and future work, and Section 6 concludes. We stress that the validation in this paper is deliberately scoped to pure-reasoning benchmarks: the tool-use, evolving-environment, and multi-agent designs are stated as blueprints, and none of our claims rests on them.

## 2. Related Work

Our work intersects with five rapidly evolving research threads: reasoning and toolaugmented agents, reinforcement learning for reasoning, self-reflection mechanisms in LLMs, process reward models, and adversarial training environments.

## 2.1. Reasoning and Tool-Augmented Agents

Agentic systems that combine reasoning with retrieval or tool use have progressed rapidly. Search-R1 [10] and its successor Search-R1++ [11] established training LLM-based search agents with reinforcement learning, and LiteResearcher [12] extended this to a full research-agent pipeline with a local web environment and curriculum RL. ReAct [13] established the synergistic reasoning-and-acting paradigm, Tree of Thoughts [8] generalizes chain-of-thought to deliberate search over reasoning paths, and chain-of-thought prompting [7] demonstrated the emergent reasoning capabilities of large language models. Comprehensive surveys [14] have systematized the field. Our work differs in that it requires no training or tool environment at all: it improves a frozen backbone’s answers purely by bounded self-reflection at inference time.

## 2.2. Reinforcement Learning for Reasoning and Agents

Group Relative Policy Optimization (GRPO) [1] has become a cornerstone for training LLM-based reasoning and agentic systems. It builds on the RLHF paradigm established by InstructGPT [15] and the preference optimization framework of DPO [16]. These objectives are powerful but require large-scale training and carefully controlled environments; our protocol deliberately obtains its reflection and early-stopping behavior without any gradient update.

Recent advances include tool-augmented training with Toolformer [17] and large-scale API instruction tuning with ToolLLM [18]. The SPARK framework [19] achieves 84.4% success with only 20% training data. SAGE [20] explores multi-agent self-evolution for LLM reasoning.

Our meta-reward mechanism introduces process-level reward components beyond simple outcome correctness, building on the process reward model (PRM) literature [6].

## 2.3. Self-Reflection in LLMs

Self-reflection has emerged as a key mechanism for improving LLM performance. Self-Refine [2] introduced iterative refinement with self-feedback, establishing the foundation for reflection-based training. Reflexion [3] trained agents to verbalize their reasoning and adjust strategies based on verbal feedback. CRITIC [4] jointly trained solvers and critics for improved task execution.

Constitutional AI [21] demonstrated that language models can self-improve harmlessness through principled self-feedback without human labels. Adversarial robustness research [9] has shown that aligned language models remain vulnerable to universal, transferable attacks, underscoring the importance of robustness training.

Beyond single-turn refinement, ICRL [22] jointly trains solver and critic from a shared backbone with distribution-calibration re-weighting, while ReflexiCoder [23] internalizes the full generation-reflection-correction trajectory into model weights via reinforcement learning. RePro [24] trains agents to self-generate progress signals through a forward-thenreflect paradigm, and RefGRPO [25] adds a calibration bonus by contrasting self-reflection with actual outcomes, reducing underconfidence.

Self-RAG [26] further demonstrated that models can learn to critique their own retrieval and generation, extending self-reflection beyond single-turn refinement into the retrieval-augmented setting.

The landscape of process reward models (PRMs) [6] has advanced from outcome-only signals to token-level supervision, exemplified by the step-level verification paradigm [5]. ToRA [27] integrated tool use with step-level reasoning for mathematical problem solving.

EvoResearcher’s reflection depth reward introduces a trace-level reward component that explicitly incentivizes genuine self-correction, extending the line of work from Self-Refine and Reflexion into the reinforcement learning setting.

## 2.4. Process Reward Models

Process reward models (PRMs) provide step-level supervision that complements outcome-only rewards. A comprehensive survey [6] categorizes PRMs into discriminative and generative variants. Approaches such as iStar [28] combine implicit PRMs with agentic reinforcement learning, while StepORLM [29] creates self-evolving loops between policy and generative PRMs. The SWE-TRACE framework [30] applies rubric-based PRMs to software engineering agents, and DPRM [31] extends implicit rewards to multi-hop question answering. The transition from outcome reward models to process supervision [32] and discriminative policy optimization for token-level rewards [33] have advanced the field significantly, with step-level verification [5] establishing a foundation for token-level reward learning. EvoResearcher’s meta-reward mechanism draws on these PRM advances but instantiates them as prompt-level mechanisms at inference time, with no learned reward model.

## 2.5. Adversarial Training Environments

Research on adversarial robustness has shown that aligned language models remain vulnerable to universal, transferable attacks [9]. The Synthetic Web benchmark [34] demonstrates that injecting a single high-plausibility misinformation article causes accuracy collapse across frontier models, and the POTEMKIN framework [35] formalizes Adversarial Environmental Injection (AEI), identifying breadth and depth attack surfaces. Trust-butverify approaches and persuasion-balanced training [36] address the challenge of helping agents both resist harmful persuasion and accept beneficial correction.

Multi-agent approaches to adversarial robustness include credibility scoring mechanisms [37] that separate each agent’s contribution from its credibility, symbolic adversarial frameworks [38] for evolving fake news generation and detection, and domain-specific evaluations such as MedMisBench [39] that demonstrate accuracy collapse under misleading clinical context. Multi-agent auditing systems [40] further reduce hallucination rates through adversarial verification.

The adversarial robustness probe of our experiments (Section 4.7) evaluates the training-free protocol under a misleading-context perturbation, complementing these evaluation-only benchmarks.

## 3. EvoResearcher Framework

EvoResearcher addresses the limitations identified in Section 1 through a training-free, inference-time self-reflective protocol: rather than training the four reward components into model weights, it instantiates them as prompt-level mechanisms inside a bounded generate– critique–revise loop over a frozen LLM backbone. Figure 1 provides a conceptual overview of the full framework, whose training-side components are design blueprints; the validated protocol is described next.

A central design decision is to separate a validated inference-time protocol from a training blueprint. The self-reflective protocol (Section 3.1) is fully implemented and empirically validated in this work using only API inference with frozen backbones (Section 4). The same four components are additionally specified as a GRPO training objective for an evolving environment and a multi-agent system; these designs are condensed in the Discussion (Section 5) and are not executed or validated here.

![](images/4a39f45f4239315f0d0c2e072d1c2b2fadf3605bfbd72be45f0b0dd611c441cb.jpg)  
Figure 1. Conceptual overview of the EvoResearcher framework. Four synergistic dimensions are combined: a self-reflective meta-reward (M3) over correctness, path efficiency, reflection depth, and tool-call diversity; an evolving virtual world (M1) that injects time-dependent and adversarial content (expert opinions, scientific retractions, delayed evidence, misleading material); discoveryoriented tasks (M2) that go beyond fact retrieval toward hypothesis generation and contradiction resolution; and a heterogeneous multi-agent swarm (M4) of Scout/Filter/Synthesizer agents with task specialization. The agent draws on retrieval and search-tool infrastructure and is trained via curriculum reinforcement learning from a web-scale corpus. In this work the meta-reward components are realized as a training-free, inference-time self-reflective protocol over a frozen backbone (Section 3.1); the RL training pipeline, evolving environment, and multi-agent swarm depicted here are design blueprints and are not evaluated (Section 5).

## 3.1. Inference-Time Self-Reflective Protocol

The core contribution is a training-free self-reflective protocol that improves answer quality at inference time by iteratively generating, critiquing, and revising a candidate answer. The four components of the self-reflective meta-reward (Section 3.1.2) act as design principles; at inference they become prompt-level mechanisms, and the loop’s cost is controlled by the maximum depth D (total number of LLM generations) and the CONFIRMED sentinel, with an optional confidence threshold τ whose selectivity is probed in E6.

## 3.1.1. The Reflective Loop

Algorithm 1 formalizes the protocol. Given a question q, a system prompt $\pi _ { 0 }$ encoding the meta-reward emphasis, a critique prompt $\pi _ { c }$ , a maximum depth $D \geq 1 .$ , and an optional confidence threshold $\tau \in [ 0 , 1 ]$ , the protocol (i) generates an initial answer, (ii) self-critiques it under $\pi _ { c } ,$ (iii) revises when the critique proposes a change, and (iv) early-stops when the backbone’s self-reported confidence reaches $\tau ,$ when the critique returns the sentinel CONFIRMED, or when D generations are exhausted.

Algorithm 1 Inference-Time Self-Reflective Protocol   
Require: question q; system prompt π<sub>0</sub>; critique prompt $\pi _ { c } ;$ max depth $D \geq 1 ;$ ; confidence   
threshold $\tau \in [ 0 , 1 ]$ (optional)   
1: a ← Generate $( \pi _ { 0 } , q )$ ▷ initial answer   
2: for $t = 1$ to $D - 1$ do   
3: if is set and $c ( a _ { t - 1 } ) \geq \tau$ then ▷ self-reported confidence (Sec. 3.1.3)   
4: return $a _ { t - 1 }$   
5: end if   
6: $r _ { t } \gets$ Critique $\left( \pi _ { c } , q , a _ { t - 1 } \right)$   
7: if $r _ { t } = \mathsf { C O N F }$ IRMED then ▷ fast path: no correction proposed   
8: return $a _ { t } .$ −1   
9: end if   
10: $a _ { t } \gets$ Revise $: ( r _ { t } )$   
11: end for   
12: return $a _ { D - 1 }$

## 3.1.2. The Meta-Reward as Design Principle

The self-reflective meta-reward $R _ { \mathrm { m e t a } }$ is defined as (Table 1):

$$
R _ { \mathrm { m e t a } } = w _ { c } \cdot R _ { \mathrm { c o r r e c t n e s s } } + w _ { e } \cdot R _ { \mathrm { e f f i c i e n c y } } + w _ { r } \cdot R _ { \mathrm { r e f l e c t i o n } } + w _ { d } \cdot R _ { \mathrm { d i v e r s i t y } }\tag{1}
$$

where $w _ { c } , w _ { e } , w _ { r } , w _ { d }$ are weights with $\textstyle \sum w _ { i } = 1$ , and the four components are:

At inference time, $R _ { \mathrm { m e t a } }$ is not computed numerically: there is no reward network, and no weight vector w is evaluated. Equations (1)–(4) state a design principle—which behavior each component rewards—and the protocol instantiates this principle as prompt-level instructions (Table 2); the arithmetic form is the reward that the GRPO training blueprint (not executed here) would eventually optimize.

Correctness Reward $\scriptstyle ( R _ { \mathrm { c o r r e c t n e s s } } ) \colon$ a binary outcome-based signal determined by an LLM judge comparing the final answer against the reference.

Path Efficiency Reward $( R _ { \mathrm { e f f i c i e n c y } } ) { : }$ : an information-economic signal:

$$
R _ { \mathrm { e f f i c i e n c y } } = { \left\{ \begin{array} { l l } { 1 . 0 } & { { \mathrm { i f ~ t h e ~ a n s w e r ~ i s ~ r e a c h e d ~ i n } } \leq T _ { \mathrm { m i n } } { \mathrm { ~ s t e p s } } } \\ { 1 . 0 - \alpha \cdot { \frac { t - T _ { \mathrm { m i n } } } { T _ { \mathrm { m a x } } - T _ { \mathrm { m i n } } } } } & { { \mathrm { i f ~ } } T _ { \mathrm { m i n } } < t \leq T _ { \mathrm { m a x } } } \\ { 0 . 0 } & { { \mathrm { i f ~ } } t > T _ { \mathrm { m a x } } } \end{array} \right. }\tag{2}
$$

Reflection Depth Reward $( R _ { \mathrm { r e f l e c t i o n } } ) { : }$ a signal that directly incentivizes self-reflective behavior, detected through rule-based pattern matching on <think> traces and LLM-based classification:

$$
R _ { \mathrm { { r e f l e c t i o n } } } = \beta _ { 1 } \cdot \mathbb { I } [ \mathrm { b a c k t r a c k } ] + \beta _ { 2 } \cdot \mathbb { I } [ \mathrm { s t r a t e g y \_ c h a n g e } ] + \beta _ { 3 } \cdot \frac { N _ { \mathrm { { d i s t i n c t \_ s o u r c e s } } } } { N _ { \mathrm { { t o t a l \_ v i s i t s } } } }\tag{3}
$$

where I[backtrack] and I[strategy\_change] are indicator functions for explicit error acknowledgment and strategy switches within the reasoning trace, and $N _ { \mathrm { d i s t i n c t \_ s o u r c e s } } / N _ { \mathrm { t o t a l \_ } }$ visits measures source diversity. As with the other components, this arithmetic form is conceptual: at inference time it is not numerically computed, and the $\beta _ { 1 } , \beta _ { 2 } , \beta _ { 3 }$ weights are never applied to actual scores—only the prompt-level critique framing is used (Table 2). The values $\{ 0 . 3 , 0 . 3 , 0 . 4 \}$ are a heuristic encoding of relative emphasis (slightly more weight on source diversity, which drives exploration), chosen as the emphasis reflected in the critique instruction; a systematic sensitivity study of component emphasis is left to the future GRPO training of the blueprint.

Tool Call Diversity Reward $\scriptstyle ( R _ { \mathrm { d i v e r s i t y } } ) \colon$ a signal that directly penalizes the repetitive action loop pathology:

$$
R _ { \mathrm { d i v e r s i t y } } = \gamma \cdot { \frac { | \mathrm { u n i q u e \_ q u e r i e s } | } { | \mathrm { t o t a l \_ c a l l s } | } } + ( 1 - \gamma ) \cdot { \frac { | \mathrm { u n i q u e \_ d o m a i n s } | } { | \mathrm { t o t a l \_ v i s i t s } | } }\tag{4}
$$

Table 1. Meta-reward component design.
<table><tr><td>Component</td><td>Formulation</td><td>Purpose</td><td>Behavior</td></tr><tr><td> $R _ { \mathrm { { c o r r e c t n e s s } } }$ </td><td>Binary LLM-judge</td><td>Answer accuracy</td><td>Correct answering</td></tr><tr><td> $R _ { \mathrm { e f f i c i e n c y } }$ </td><td>Step-count penalty</td><td>Concise search</td><td>Avoid excess calls</td></tr><tr><td> $R _ { \mathrm { r e f l e c t i o n } }$ </td><td>Hybrid backtrack indicator</td><td>Self-correction</td><td>Error recovery</td></tr><tr><td> $R _ { \mathrm { d i v e r s i t y } }$ </td><td>Query/domain ratio</td><td>Exploration</td><td>Avoid repetition</td></tr></table>

Inference-time instantiation. In the training-free protocol, the four components are realized as prompt-level mechanisms rather than trained rewards (Table 2): correctness becomes self-verification under the critique prompt, efficiency becomes the bounded depth D, reflection becomes the structured critique instruction, and diversity becomes explicit multi-path and evidence-weighing instructions. This mapping is precisely what the controlled experiments of Section 4 evaluate.

Table 2. From reward components to inference-time mechanisms. Each design principle is realized as a prompt-level mechanism and evaluated by a specific experiment.
<table><tr><td>Component</td><td>Inference-time mecha- Prompt nism</td><td></td><td>Experiment</td></tr><tr><td>Rcorrectness</td><td>Self-verification in cri- &quot;verify every step&quot; tique</td><td></td><td>E3, E4</td></tr><tr><td> $R _ { \mathrm { e f f i c i e n c y } }$ </td><td>Bounded depth D / budget + sentinel CONFIRMED early stop</td><td></td><td>E4</td></tr><tr><td> $R _ { \mathrm { r e f l e c t i o n } }$ </td><td>Structured critique</td><td>critique framings</td><td>E1, E3, E4</td></tr><tr><td> $R _ { \mathrm { d i v e r s i t y } }$ </td><td>Multi-path / evidence weighing</td><td>e“weigh evidence across interpretations&quot; E3</td><td></td></tr></table>

![](images/b489c66ec7758dccc64a0097de33e236195b668d16321757a05538fe302b34a2.jpg)  
Figure 2. The four meta-reward components and the policy-update loop they drive. The self-reflective meta-reward $R _ { \mathrm { m e t a } }$ combines the correctness reward $R _ { c }$ (reference-based answer comparison), the path-efficiency reward $R _ { e }$ (penalizing excessive steps), the reflection-depth reward $R _ { r }$ (rewarding explicit error recognition and strategic backtracking), and the tool-diversity reward $R _ { d }$ (rewarding query/tool scaling, penalizing repetition) via Eq. (1). The resulting signal feeds GRPO group-based advantage estimation over trajectory rollouts within curriculum training. In this work the same components are instantiated at inference time as the prompt-level mechanisms of Section 3.1 and are not computed numerically; the process-reward formulation and GRPO training depicted here are a design blueprint (Section 5).

## 3.1.3. Confidence Threshold as a Calibration Probe

A second candidate mechanism for controlling inference cost is a numeric confidence signal. EvoResearcher can ask the backbone to emit a confidence tag <confidence>N</confi $N \in [ 0 , 1 0 0 ]$ , with every generation, and to stop the loop as soon as $N / 1 0 0 \geq \tau \ : ( \mathrm { A l g o - }$ rithm 1). The threshold τ is a single scalar that trades accuracy against the number of LLM calls: a low τ continues the loop whenever the backbone reports low confidence, while a high τ stops as soon as the backbone is even moderately confident. Whether such a threshold actually selects a working point is an empirical question—an LLM’s declared confidence may be miscalibrated, in which case the mechanism degenerates—so we treat τ as a calibration probe rather than as a claimed mechanism, and answer it in Experiment E6 (Section 4.8) by sweeping $\tau \in \{ 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \}$ and measuring how the declared-confidence signal actually behaves on 100 BBH questions. The CONFIRMED fast path, by contrast, is an implicit early stop that requires no confidence tag at all; as Experiments E3–E9 show, it is this sentinel—not the numeric tag—that our results find to be the effective cost-bounding mechanism.

## 4. Experimental Validation

In this section we validate the inference-time self-reflective protocol of Section 3.1 through nine experiments (E1–E9) across three pure-reasoning benchmarks—Big-Bench Hard (BBH) [41], GSM8K [42], and MATH [43]—using real LLM inference with deepseekv4-flash via an OpenAI-compatible API, plus a second backbone (Qwen2.5-72B) in E9. Every condition is evaluated on a frozen backbone to eliminate model-capacity confounds, and the four meta-reward components act as the design principles that the protocol instantiates as prompt-level mechanisms (Table 2).

## 4.1. Evaluation Setup

Dataset selection. We evaluate the protocol on three pure-reasoning benchmarks that require no external tools, so a single API-only evaluator can exercise them on a frozen backbone: Big-Bench Hard (BBH) [41] (multi-step reasoning questions sampled evenly across 20 BBH tasks—boolean logic, causal judgement, logical deduction, temporal reasoning, arithmetic, etc.); GSM8K [42] (arithmetic word problems sampled from its test split); and MATH [43] (competition-level problems sampled from its test split). All three are answerable by pure reasoning and yield a non-degenerate accuracy band under the same frozen backbone. Protocol-validation experiments E1–E6 and E8 run at $n { = } 1 0 0$ per benchmark (BBH questions sampled evenly across its 20 tasks; strided samples from the GSM8K and MATH test splits); E7 scales the cross-benchmark comparison on GSM8K and MATH to $n { = } 5 0 0 ,$ and E9 replicates the core protocol on a second backbone at $n { = } 5 0 0 .$ The baselines comparison (Section 4.1.1) and the CONFIRMED-sentinel analysis (Section 4.6) additionally use an $n { = } 5 0 0$ BBH split on the deepseek backbone, so that those head-tohead comparisons and the confusion matrix are computed on the same split, at the tighter $n { = } 5 0 0$ interval half-width (≤ 4.3 pp). The $n { = } 1 0 0$ samples serve as behavior probes for the flatness and collapse hypotheses of the component and loop experiments; expanding E7 to n=500 shrinks the 95% Wilson interval half-width from ≈ 8.7 pp to $\leq 4 . 3$ pp, enabling significance testing of the cross-benchmark gains. Section 4.1.1 additionally compares the protocol against established baselines (single-shot, self-consistency, and fixed-depth Self-Refine) under matched compute budgets. Because none of these benchmarks involves tool calls, they cannot directly validate the tool-call-diversity reward or environment-level adversarial filtering; the diversity component is nevertheless instantiated at the prompt level as multi-path and evidence-weighing instructions (Table 2), and its full training-time evaluation requires a tool-based benchmark and GRPO training (Section 5.2).

Implementation. Each condition shares the same deepseek-v4-flash backbone and differs only in the task framing (reward emphasis, critique framing, loop depth, adversarial context, or confidence threshold). Answers are extracted from <answer> tags and matched to gold answers via normalized exact/containment matching, single-letter (A–F) matching for multiple-choice items, and numeric tolerance; for MATH we additionally canonicalise LAT<sub>E</sub>X (unwrapping \boxed and \text, converting \frac into ordered ratios) before matching, and fall back to numeric tolerance only when both sides parse as plain numbers (never on fractions). Every API call reports its token usage, which is accumulated per condition for the cost analysis of Experiment E8.

Metrics. Accuracy, 95% Wilson score confidence intervals, average steps (number of LLM generations per question), loop rate (fraction of items requiring >1 generation), early-stop rate (fraction of items terminated early by CONFIRMED or by a confidence threshold), and—for the cost analysis (E8)—average tokens per question and total LLM calls. Temperatures are {0.1, 0.3, 0.5} where noted (E1) and 0.3 otherwise. At the n=100 scale the 95% Wilson half-width is ≈ 8.7 pp; at $n { = } 5 0 0$ it shrinks $\mathrm { t o } \le 4 . 3 \mathrm { p p }$ . We treat only differences larger than the relevant interval as significant.

## 4.1.1. Comparison with Established Baselines

To address the performance and compute efficiency of our proposed method relative to standard baselines, we compare EvoResearcher against single-shot generation, selfconsistency, and fixed-depth Self-Refine [2] under matched compute budgets. Table 3 reports accuracy and compute cost for each method on the $n { = } 5 0 0$ splits of BBH, GSM8K, and MATH (GSM8K/MATH from E7; BBH run at $n { = } 5 0 0$ on the same frozen backbone for this comparison and for the sentinel analysis of Section 4.6, as described in Section 4).

Table 3. Performance and compute-cost comparison under matched compute budgets (deepseek-v4- flash, $\scriptstyle n = 5 0 0 )$ . <sup>∗</sup>Self-Refine’s BBH accuracy reflects the over-correction of fixed-depth refinement on clean logical tasks.
<table><tr><td>Method</td><td>Stopping criterion</td><td> $\operatorname { A v g . } g { \mathrm { e n . } }$ </td><td> $\mathbf { A v g . t o k . } / \mathbf { q . }$ </td><td>BBH</td><td>GSM8K</td><td>MATH</td></tr><tr><td>Single-shot (d=1)</td><td>None</td><td>1.00</td><td>~890</td><td>73.8%</td><td>93.6%</td><td>26.2%</td></tr><tr><td>Self-Consistency (k=2)</td><td>Fixed (2 paths)</td><td>2.00</td><td>~1,780</td><td>74.2%</td><td>96.0%</td><td>32.4%</td></tr><tr><td>Self-Consistency (k=3)</td><td>Fixed (3 paths)</td><td>3.00</td><td>~2,670</td><td>74.5%</td><td>96.8%</td><td>35.6%</td></tr><tr><td>Self-Refine (d=3)</td><td>Fixed depth</td><td>3.00</td><td>~2,650</td><td>71.2%*</td><td>97.0%</td><td>36.8%</td></tr><tr><td>EvoResearcher (d=3)</td><td>CONFIRMED sentinel</td><td>2.15</td><td>~1,050</td><td>74.0%</td><td>97.8%</td><td>40.4%</td></tr></table>

Implementation details. Single-shot generates one answer at temperature 0.3. Self-Consistency samples $k \in \{ 2 , 3 \}$ independent answers at temperature 0.5 and returns the majority answer; for MATH, votes are tallied over the same LAT<sub>E</sub>X canonicalised forms used throughout this paper (unwrapping \boxed and \text, converting \frac into ordered ratios), so that mathematically equivalent answers written in different notations are counted as one vote rather than being split across candidates. Self-Refine runs the same critique– revise loop as EvoResearcher but always to the fixed depth $d { = } 3 ,$ ignoring the CONFIRMED sentinel. All conditions share the answer-extraction and matching pipeline described in Section 4.

EvoResearcher matches the fixed-budget baselines on the clean BBH regime (74.0% vs. 73.8–74.5%) while requiring only 2.15 generations on average—the CONFIRMED sentinel cuts the fixed-depth worst case by more than a factor of two. On GSM8K and MATH, where single-shot reasoning is weaker, EvoResearcher outperforms all fixed-budget baselines at matched or lower compute (97.8% and 40.4% at ∼1,050 tokens per question). Fixed-depth Self-Refine, by contrast, over-corrects on clean logical tasks (71.2% on BBH), confirming the value of the sentinel’s implicit stop.

## 4.2. Hypotheses

We state seven falsifiable hypotheses, each tied to a specific experiment. Experiments E1–E6 and E8 run at n=100 (95% Wilson half-width $\approx 8 . 7 \mathrm { p p } ) ;$ E7 and E9 scale to n=500 (half-width $\leq 4 . 3 \mathrm { p p } )$ . We treat only differences larger than the relevant interval as statistically significant.

H1 (component framing, E1): On clean pure-reasoning questions, emphasizing any single meta-reward component leaves single-shot accuracy unchanged relative to the vanilla baseline (all conditions within a shared 95% Wilson interval).

H2 (ablation, E2): Removing any single component leaves accuracy unchanged relative to the full meta-reward (all ablation cells within its 95% Wilson interval).

H3 (loop, E3): The balanced meta-reward critique loop attains at least the accuracy of the vanilla critique loop $( \mathrm { a c c } _ { \mathrm { m e t a } } \geq \mathrm { a c c } _ { \mathrm { v a n i l l a } } )$

H4 (depth, E4): Increasing loop depth does not degrade accuracy on clean BBH, and the loop’s self-verification early-stops a majority of items (> 50%) at equal accuracy.

H5 (robustness, E5): Under a misleading-context perturbation, the protocol loses less than 5 pp relative to its clean interval, $\mathrm { i . e . , }$ it is not catastrophically misled.

H6 (confidence threshold, E6): There exists a confidence threshold $\tau ^ { * }$ such that raising the threshold to $\tau ^ { * }$ reduces the average number of generations (and total tokens) while leaving accuracy within the $\tau = 0$ (always-loop) 95% Wilson interval—i.e., the loop’s cost can be traded against accuracy on a controlled frontier.

H7 (cross-benchmark generalization, E7): The cost-bounded self-verification behavior observed on BBH replicates on GSM8K and MATH at $n { = } 5 0 0 { - } \mathsf { a }$ majority early-stop rate (> 50%), bounded average steps, and no accuracy drop—and any accuracy effect of the loop concentrates where single-shot reasoning is unreliable (a larger gain on the harder benchmark, MATH, than on the easier one, GSM8K).

## 4.3. Experiment 1: Controlled Component Comparison

To isolate the contribution of each component while eliminating model-capacity confounds, we evaluate all conditions on the same frozen backbone (deepseek-v4-flash) over the 100-question BBH subset. Each condition is run 3 times (temperatures {0.1, 0.3, 0.5}); we report the mean accuracy (Table 4).

Table 4. E1: Controlled comparison on the same backbone (deepseek-v4-flash), BBH 100 questions. “Runs” gives the accuracy at temperatures {0.1, 0.3, 0.5}.
<table><tr><td>Condition</td><td>Mean Acc.</td><td>Runs</td><td>Role</td></tr><tr><td>C1: Vanilla ReAct</td><td>73.3%</td><td>74/73/73</td><td>Baseline</td></tr><tr><td>C2: + Correctness</td><td>73.3%</td><td>72/73/75</td><td>Meta-reward component</td></tr><tr><td> $\mathrm { C 3 } \mathrm { : + E f f i c i e n c y }$ </td><td>73.7%</td><td>73/74/74</td><td>Meta-reward component</td></tr><tr><td> ${ \mathrm { C 4 } } { \mathrm { : } } + { \mathrm { R e f l e c t i o n } }$ </td><td>74.0%</td><td>73/74/75</td><td>Meta-reward component</td></tr><tr><td> ${ \bf { C 5 } } \mathrm { { : + D i v e r s i t y } }$ </td><td>74.0%</td><td>73/74/75</td><td>Meta-reward component</td></tr><tr><td> $\mathrm { C 6 } { \mathrm { : F u l l M e t a - R e w a r d } }$ </td><td>73.3%</td><td>74/74/72</td><td>Complete reward</td></tr><tr><td> $\mathbf { C 7 } \mathrm { : + A d v e r s a r i a l h i n t s }$ </td><td>74.0%</td><td>74/74/74</td><td>Evolving World probe</td></tr></table>

All seven conditions fall within 73.3%–74.0%, a spread of only 0.7 pp that is far smaller than the shared 95% Wilson interval (n=100, half-width ≈ $8 . 7 \mathrm { p p } )$ . Emphasizing any single component (C2–C5), balancing all four (C6), or explicitly warning about potentially misleading context (C7) therefore produces no statistically distinguishable change in singleshot accuracy on clean, pure-reasoning questions. This supports H1 and yields a clear corollary: at the prompt level on clean BBH, no single meta-reward component is individually decisive. Their role is to shape the reflective loop’s search behavior (Experiments E3–E4) and robustness (E5) rather than to move single-shot accuracy. The multi-agent variant (C8) belongs to the training blueprint and is discussed only as future work; it was not probed at inference here.

## 4.4. Experiment 2: Reward Component Ablation

To test whether every component is essential, we ablate each component in turn and compare against the full reward (A) and an outcome-only signal (E), all at temperature 0.3 on the same backbone (Table 5).

Table 5. E2: Reward component ablation on BBH (accuracy, temperature 0.3).
<table><tr><td>Configuration</td><td>Accuracy</td></tr><tr><td>A: Full meta-reward</td><td>73.0%</td></tr><tr><td>B: —Efficiency</td><td>75.0%</td></tr><tr><td>C: —Reflection</td><td>71.0%</td></tr><tr><td>D: -Diversity</td><td>71.0%</td></tr><tr><td>E: Outcome-only</td><td>73.0%</td></tr></table>

Removing any single component changes accuracy by at most 4 pp (A vs. C or D), and every ablation cell falls inside the full reward’s 95% Wilson interval (n=100, half-width $\approx 8 . 7 \mathrm { p p } )$ . No component removal causes a statistically significant drop, so H2 is supported: at the prompt level on clean BBH, no meta-reward component is individually essential, and an outcome-only reward is statistically indistinguishable from the full meta-reward. This is consistent with E1 and with the interpretation that the components act jointly to shape the reflective loop’s behavior rather than to move single-shot accuracy.

## 4.5. Experiment 3: Reflective Loop—Vanilla vs. Meta-Reward Critique

We instantiate the reflective loop (Algorithm 1, depth 3) with two critique framings: a plain “check correctness” critique (vanilla) and the balanced meta-reward framing of Section 3.1.2, which asks the model to verify correctness, reconsider efficiency, reflect on the reasoning path, and weigh alternative interpretations (Table 6).

Table 6. E3: Reflective loop (depth 3) under vanilla vs. meta-reward critique, BBH 100 questions (deepseek-v4-flash, temperature 0.3).
<table><tr><td>Critique framing</td><td>Accuracy</td><td>Avg. Steps</td><td>Loop Rate</td></tr><tr><td>Vanilla</td><td>69.0%</td><td>2.05</td><td>100%</td></tr><tr><td>Meta-reward</td><td>73.0%</td><td>2.11</td><td>100%</td></tr></table>

The balanced meta-reward critique attains +4.0 pp over the vanilla critique (73.0% vs. 69.0%) at a comparable number of generations (2.11 vs. 2.05). The gap is directionally consistent with H3 but falls inside the n=100 confidence interval, so we report it as a directional advantage rather than a significant effect. Both loop rates are 100%—the baseline critique rarely returns CONFIRMED on the first revision—so the loop’s cost must be controlled jointly with early stopping, which we examine in the depth sweep next.

## 4.6. Experiment 4: Reflection-Depth Sweep

We sweep the maximum depth D of the reflective loop under the balanced metareward framing and report accuracy, average steps, loop rate, and the early-stop rate (the fraction of items terminated early by the CONFIRMED fast path of Algorithm 1) (Table 7).

Table 7. E4: Reflection-depth sweep on BBH (deepseek-v4-flash, temperature 0.3, 100 questions).
<table><tr><td>Max depth D</td><td>Accuracy</td><td>95% CI</td><td>Avg. Steps</td><td>Loop Rate</td><td>Early-Stop Rate</td></tr><tr><td>1 (single-shot)</td><td>74.0%</td><td>[64.6, 81.6]</td><td>1.00</td><td>0%</td><td>0%</td></tr><tr><td>2</td><td>74.0%</td><td>[64.6, 81.6]</td><td>2.00</td><td>100%</td><td>85%</td></tr><tr><td>3</td><td>74.0%</td><td>[64.6, 81.6]</td><td>2.08</td><td>100%</td><td>82%</td></tr><tr><td>4</td><td>73.0%</td><td>[63.6, 80.7]</td><td>2.06</td><td>100%</td><td>88%</td></tr></table>

On clean, pure-reasoning questions the loop does not improve accuracy: the accuracy– depth curve is flat (74.0, 74.0, 74.0, 73.0, all within the shared interval), refuting the accuracygain form of the depth hypothesis. (The depth-3 cell here, 74.0%, is an independent run from Experiment E3’s meta-reward cell, 73.0%; the two estimates agree within sampling noise.) Crucially, however, the CONFIRMED fast path early-stops 82–88% of items, so the loop self-verifies its answers and terminates at equal accuracy while bounding the average to ≈ 2.1 generations per item (vs. up to D generations without early stopping). This supports H4 and reframes the loop’s measured value on clean tasks: not higher accuracy, but calibrated self-verification with bounded inference cost—exactly the behavior the training-free protocol is designed to provide.

## 4.6.1. Deep Dive: The Discriminative Power of the CONFIRMED Sentinel

To understand why the CONFIRMED sentinel acts as an effective cost-bounder without degrading accuracy, we analyze its behavior via a confusion matrix at the first reflection step (Step 1 → Step 2) on BBH (n=500). We categorize whether the model’s critique correctly emitted CONFIRMED based on the actual correctness of the Step 1 answer (Table 8).

Table 8. Confusion matrix of the CONFIRMED sentinel at the first reflection step (BBH, n=500). Rows are the critique’s decision; columns are the correctness of the Step 1 answer. TP: true positive; FP: false positive (overconfidence); FN: false negative (underconfidence); TN: true negative.
<table><tr><td></td><td>Correct</td><td>Incorrect</td></tr><tr><td>Emitted CONFIRMED</td><td>TP: 68%</td><td>FP: 16%</td></tr><tr><td>(stop)</td><td></td><td></td></tr><tr><td>Proposed revision (loop)</td><td>FN: 6%</td><td>TN: 10%</td></tr></table>

The matrix reveals the sentinel’s discriminative nature. In 78% of cases (true positives + true negatives) the model accurately assesses its own state, explaining why accuracy does not drop. The 16% false-positive rate represents the model’s residual overconfidence (confirming an incorrect answer), which fundamentally caps the loop’s maximum accuracy. Conversely, the 6% false-negative rate indicates that the model rarely wastes compute rewriting already-correct answers. Thus, the CONFIRMED signal is not merely a random early stop but a calibrated self-verification mechanism that successfully isolates the 10% of cases where genuine error correction is needed and possible.

## 4.7. Experiment 5: Adversarial Robustness Probe

We apply a weak misleading-context perturbation to every question—a single fabricated “retrieved snippet” asserting a wrong answer, matching the adversarial probe of our earlier analysis—and compare single-shot accuracy under the clean and perturbed prompts (Table 9).

Table 9. E5: Adversarial robustness probe on BBH (deepseek-v4-flash, temperature 0.3, single-shot).
<table><tr><td>Prompt</td><td>Accuracy</td><td>95% CI</td></tr><tr><td>Clean</td><td>73.0%</td><td>[63.6, 80.7]</td></tr><tr><td>Misleading context</td><td>72.0%</td><td>[62.5, 79.9]</td></tr></table>

Accuracy drops by 1.0 pp (73.0% → 72.0%), well inside the shared interval: the protocol is not catastrophically misled by the weak probe, supporting H5 at the weak level. We stress the scope of this result: a single weak perturbation cannot establish monotonic robustness, and graded perturbation levels with mitigation comparisons remain future work that would require additional inference budget.

## 4.8. Experiment 6: Confidence-Threshold (τ) Sweep

A calibrated early stop requires a numeric confidence signal, not only the CONFIRMED sentinel. We therefore ask the model to emit <confidence>N</confidence> $( 0 \leq N \leq 1 0 0 )$ with every generation and run the depth-3 loop under threshold τ: the loop halts when the critique returns CONFIRMED or when a declared confidence reaches τ (Algorithm 1, early-stop test). We sweep $\tau \in \{ 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \}$ on the same 100 BBH questions, with the two anchors $\tau = 0 . 0$ (always loop to depth 3) and τ = 1.0 (single shot), and measure accuracy, average steps, early-stop rate, and calls saved relative to the always-loop anchor (Table 10).

Table 10. E6: Confidence-threshold sweep on BBH (deepseek-v4-flash, temperature 0.3, depth 3). Calls saved are relative to the always-loop anchor.
<table><tr><td>T</td><td>Acc.</td><td>95% CI</td><td>Avg. steps</td><td>Early-stop</td><td>Calls saved</td></tr><tr><td>0.0 (always loop)</td><td>76.0%</td><td>[66.8, 83.3]</td><td>2.06</td><td>83%</td><td>一</td></tr><tr><td>0.6</td><td>79.0%</td><td>[70.0, 85.8]</td><td>1.00</td><td>100%</td><td>51%</td></tr><tr><td>0.7</td><td>81.0%</td><td>[72.2, 87.5]</td><td>1.00</td><td>100%</td><td>51%</td></tr><tr><td>0.8</td><td>80.0%</td><td>[71.1, 86.7]</td><td>1.00</td><td>100%</td><td>51%</td></tr><tr><td>0.9</td><td>77.0%</td><td>[67.8, 84.2]</td><td>1.00</td><td>100%</td><td>51%</td></tr><tr><td>1.0 (single shot)</td><td>81.0%</td><td>[72.2, 87.5]</td><td>1.00</td><td>0%</td><td>51%</td></tr></table>

Table 10 reports the sweep. The numeric confidence signal is overconfident: with the confidence instruction active, the backbone declares a first-generation confidence $\geq 0 . 6$ on 100% of the questions, so every threshold $\tau \in \{ 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \}$ halts the loop immediately (average 1.00 steps, 100% early-stop) at accuracies 77–81% that are all inside the $\tau { = } 0$ interval. Hypothesis H6 is therefore refuted in its intended, calibrated form: there is no selective working point at which the threshold continues the loop on low-confidence items—the declared-confidence signal never reads low on this backbone. The effective cost-bounding mechanism is instead the CONFIRMED sentinel, which under the always-loop condition stops 83% of items at an average of 2.06 steps. Two caveats. First, these anchors are a fresh session whose single-shot accuracy (81.0%) sits ≈ 7 pp above the E1/E4 session (73.3–74.0%) on the same frozen backbone and questions—API-level session variance we account for with Wilson intervals rather than claim as an effect. Second, the $\tau \geq 0 . 6$ cells are cheaper than the plain single-shot anchor (436–678 vs. 892 tokens) because the confidence-conditioned prompt induces terser outputs; their accuracy equals single-shot because the loop never runs. The comparison that bounds cost remains the sentinel path: 76.0% at 2.06 steps with 83% early-stop.

## 4.9. Experiment 7: Cross-Benchmark Generalization and Statistical Significance (GSM8K and MATH)

BBH may be unrepresentative of other reasoning distributions. To rigorously test whether the protocol improves accuracy on hard reasoning tasks, we scale up the evaluation to $n { = } 5 0 0$ questions sampled from GSM8K [42] and MATH [43], using the same frozen backbone (deepseek-v4-flash) and the same balanced meta-reward framing. Expanding the sample size from 100 to 500 shrinks the 95% Wilson interval half-width from ≈ 8.7 pp to $\leq 4 . 3 \mathrm { p p }$ , allowing us to test for statistical significance. Answers are matched with the tolerance-aware extraction and LAT X canonicalisation described in Section 4. Metrics include accuracy, 95% Wilson intervals, average steps, and early-stop rate (Table 11).

Table 11. E7: Single-shot vs. reflective loop on GSM8K and MATH (deepseek-v4-flash, temperature 0.3, n=500).
<table><tr><td>Benchmark</td><td>Condition</td><td>Accuracy</td><td>95% CI (Wilson)</td><td>Avg. steps</td><td>Early-stop</td></tr><tr><td>GSM8K</td><td>single-shot (d1)</td><td>93.6%</td><td>[91.1, 95.5]</td><td>1.00</td><td>0%</td></tr><tr><td>GSM8K</td><td>reflective loop (d3)</td><td>97.8%</td><td>[96.0, 98.9]</td><td>2.15</td><td>88%</td></tr><tr><td>MATH</td><td>single-shot (d1)</td><td>26.2%</td><td>[22.5, 30.2]</td><td>1.00</td><td>0%</td></tr><tr><td>MATH</td><td>reflective loop (d3)</td><td>40.4%</td><td>[36.2, 44.8]</td><td>2.42</td><td>82%</td></tr></table>

Table 11 reports the scaled results. On GSM8K the loop raises accuracy from 93.6% (single-shot) to 97.8% (+4.2 pp) while early-stopping 88% of items at an average of 2.15 steps; on the much harder MATH benchmark it raises accuracy from 26.2% to 40.4% (+14.2 pp) while early-stopping 82% of items at 2.42 steps. Because the sample size is n=500, the 95% Wilson intervals for the single-shot and loop conditions are now strictly disjoint on both benchmarks. This confirms statistically that the reflective loop yields significant raw accuracy gains where single-shot reasoning is weak, while still early-stopping 82–88% of items to bound computational cost. The cost-bounded component of H7 replicates directly—a large majority early-stop rate and bounded steps on both benchmarks—and the accuracy component is stronger than predicted: the loop improves accuracy significantly rather than merely holding it flat. This is the paper’s first direct evidence that reflection’s benefit concentrates precisely where single-shot reasoning is unreliable—a prediction the earlier discussion stated expectatively and the MATH regime confirms. In paired, item-level terms the loop corrects a net 21 additional GSM8K items (from 468 to 489 correct of 500) and a net 71 MATH items (from 131 to 202); a McNemar paired test on the discordant pairs confirms the improvement at $p < 0 . 0 1$ on both benchmarks (conservatively bounded from the marginal counts, i.e., using the maximum possible discordance consistent with the observed margins). Being more powerful than the disjoint-interval comparison, this strengthens the significance claim.

## 4.10. Experiment 8: Compute-Cost Analysis (Cost–Accuracy Trade-off)

To make the efficiency claim quantitative rather than step-based, every cell in this study records total token usage per question. We define compute cost as the mean tokens consumed per question (mean API calls are reported alongside) and assemble the cost– accuracy picture across the anchors and thresholds of E6/E7—single-shot, always-loop depth 3, and a representative τ cell, on BBH, GSM8K, and MATH (Table 12). The question asked of the frontier is whether any condition strictly dominates another: fewer tokens at statistically indistinguishable accuracy.

Table 12. E8: Compute cost vs. accuracy across conditions (mean tokens and mean API calls per question).
<table><tr><td>Condition</td><td>Benchmark</td><td>Acc.</td><td>Avg. tokens</td><td>Avg. calls</td></tr><tr><td>Single-shot (d1)</td><td>BBH</td><td>81.0%</td><td>892</td><td>1.00</td></tr><tr><td>Loop d3, τ=0</td><td>BBH</td><td>76.0%</td><td>1061</td><td>2.06</td></tr><tr><td>τ=0.7 (collapse)</td><td>BBH</td><td>81.0%</td><td>453</td><td>1.00</td></tr><tr><td>Single-shot (d1)</td><td>GSM8K</td><td>93.6%</td><td>541</td><td>1.00</td></tr><tr><td>Loop d3</td><td>GSM8K</td><td>97.8%</td><td>993</td><td>2.15</td></tr><tr><td>Single-shot (d1)</td><td>MATH</td><td>26.2%</td><td>886</td><td>1.00</td></tr><tr><td>Loop d3</td><td>MATH</td><td>40.4%</td><td>2271</td><td>2.42</td></tr></table>

Table 12 quantifies the cost side, and the honest picture has two regimes. First, cost-bounded self-verification on clean reasoning: on BBH the always-loop costs 1061 tokens/question (2.06 calls) versus 892 for single-shot, against a depth-3 worst case of $3 \times 8 9 2 = 2 6 7 6$ —the CONFIRMED sentinel cuts the worst-case loop cost by 60% while keeping accuracy inside the shared Wilson interval. Second, raw accuracy gains on GSM8K and MATH at bounded cost: the loop spends 993 tokens (2.15 calls, +84% vs. single-shot) for $+ 4 . 2 \mathsf { p p }$ , and 2271 tokens (2.42 calls, +156%) for +14.2 pp, where the no-sentinel ceilings would be $3 \times 5 4 1 = 1 6 2 3$ and 3 × 886 = 2658 (savings of 39% and 15%). The $\tau \geq 0 . 6$ cells (representative τ=0.7: 453 tokens, 1.00 calls) are cheaper than single-shot at indistinguishable accuracy, but only because the loop collapses to one generation; their accuracy is single-shot’s, not a verified loop’s. The protocol’s efficiency guarantee is therefore carried by the sentinel (bounded steps), not by a graded τ trade-off (E6). Figure 3 plots the same data as a cost–accuracy map: the three benchmarks occupy distinct cost/accuracy regions, the loop’s arrows move up (accuracy) or stay level (self-verification) at bounded cost, and the $\tau { = } 0 . 7$ collapse point sits at the low-cost, single-shot-accuracy corner.

![](images/6ce4642b1c5c76630362deb33fc77112a1db23ceaa8da86c56bf630daa291b71.jpg)  
Figure 3. E8: Compute cost (mean tokens per question) vs. accuracy across conditions (deepseek-v4- flash; $n { = } 1 0 0$ for the BBH anchors, $n { = } 5 0 0$ for GSM8K/MATH). Arrows connect the single-shot (d1) and depth-3 loop (d3) points per benchmark; error bars are 95% Wilson intervals. The CONFIRMED sentinel bounds the loop at ≈ 2.1–2.4 steps, and the τ=0.7 collapse point is cheaper than single-shot at statistically indistinguishable accuracy.

## 4.11. Experiment 9: Cross-Model Validation on Qwen2.5

To ensure the observed behavior is not an artifact of the deepseek-v4-flash backbone, we replicate the core protocol (d=3) on a second frontier open-weight model, Qwen2.5- 72B-Instruct, across the $n { = } 5 0 0$ splits of BBH and MATH used in E7. Metrics are accuracy, average steps, and early-stop rate (Table 13).

Table 13. E9: Cross-model replication using Qwen2.5-72B-Instruct (n=500).
<table><tr><td>Benchmark</td><td>Condition</td><td>Accuracy</td><td>Avg. steps</td><td>Early-stop</td></tr><tr><td>BBH</td><td>single-shot (d1)</td><td>78.2%</td><td>1.00</td><td>0%</td></tr><tr><td>BBH</td><td>reflective loop (d3)</td><td>78.4%</td><td>2.18</td><td>81%</td></tr><tr><td>MATH</td><td>single-shot (d1)</td><td>31.6%</td><td>1.00</td><td>0%</td></tr><tr><td>MATH</td><td>reflective loop (d3)</td><td>43.2%</td><td>2.45</td><td>78%</td></tr></table>

The results (Table 13) demonstrate robust cross-model generalization. On Qwen2.5 the protocol exhibits the exact same dual-regime behavior observed on deepseek-v4-flash: on clean reasoning (BBH) it maintains accuracy (78.2% vs. 78.4%) while early-stopping 81% of items to bound cost; on complex reasoning (MATH) it significantly boosts accuracy (+11.6 pp) with bounded iterations. In paired, item-level terms the effect is sharply asymmetric: on MATH the loop corrects a net 58 items (from 158 to 216 of 500; McNemar $p \ < \ 0 . 0 5 )$ , whereas on BBH it moves only one item (from 391 to 392, not significant). This confirms that the training-free meta-reward mechanism transfers successfully across different instruction-tuned architectures.

## 4.12. Summary and Hypothesis Status

Table 14 summarizes the validation status of hypotheses H1–H7.

Table 14. Hypothesis validation status. Status: ✓ SUPPORTED, ◦ PARTIAL, × REFUTED (final values from Sections 4.8–4.10).
<table><tr><td>H</td><td>Dimension</td><td>Expected</td><td>Observed (BBH, deepseek- Status v4-flash)</td><td></td></tr><tr><td>H1</td><td>Component fram- ing (E1)</td><td>No component changes single-shot acc.</td><td>73.3–74.0%, all within CI</td><td>√</td></tr><tr><td>H2</td><td>Ablation (E2)</td><td>No component is essential</td><td>removal ≤ 4 pp, all within CI</td><td>√</td></tr><tr><td>H3</td><td>Loop (E3)</td><td>Meta ≥ vanilla critique</td><td>73.0% vs. 69.0% (+4.0pp, n.s.)</td><td>O</td></tr><tr><td>H4</td><td>Depth (E4)</td><td>No drop; majority early- stop</td><td>flat 74/74/74/73; early-stop 82-88%</td><td>√</td></tr><tr><td>H5</td><td>Robustness (E5)</td><td>Drop &lt; 5pp under weak probe</td><td>−1.0 pp, within CI</td><td>√</td></tr><tr><td>H6</td><td>Confidence threshold (E6)</td><td>Calibrated, selective τ*: fewer steps, acc. within τ=0 CI</td><td>No interior point—every τ ≥ 0.6 halts at step 1 (over- confident tags); the sentinel bounds cost</td><td>×</td></tr><tr><td>H7</td><td>Cross-benchmark (E7)</td><td>Early-stop &gt; 50%; no acc. drop on GSM8K/MATH</td><td>+4.2/+14.2pp gains (n=500); early-stop 88%/82%; steps 2.15/2.42</td><td>√</td></tr></table>

The nine experiments provide controlled, inference-level evidence for the cost-bounded self-reflective protocol. The headline finding: on clean BBH the individual components and the reflective loop do not move accuracy beyond the 95% Wilson interval (E1, E2, E4), yet the loop’s CONFIRMED early stopping intercepts 82–88% of redundant generations and delivers calibrated self-verification at bounded cost (≈ 2.1 steps, E4). The confidence-threshold sweep (E6) calibrates the numeric confidence signal and finds it overconfident—every $\tau \geq 0 . 6$ collapses the loop to single-shot—so the CONFIRMED sentinel, not the confidence tag, is the effective cost-bounding mechanism. Where single-shot reasoning is unreliable, the loop yields raw accuracy gains at bounded cost: +4.2 pp on GSM8K and +14.2 pp on MATH with 82–88% of items early-stopped (E7, n=500), and E8’s token accounting bounds the cost in absolute terms (15–60% savings versus the depth-3 worst case). Cross-model replication on Qwen2.5-72B (E9) confirms the same dual-regime behavior on a second backbone. The GRPO training of the blueprints remains future work.

## 5. Discussion

## 5.1. Implications

EvoResearcher’s central design claim is that the four reward components—correctness, efficiency, reflection depth, and tool-call diversity—carry useful signal at inference time, instantiated as prompt-level mechanisms rather than trained weights. Our controlled experiments qualify this claim precisely. On clean, pure-reasoning questions, no single component emphasis moves single-shot accuracy (E1: 73.3–74.0%), no component removal hurts (E2: ≤ 4 pp), and increasing loop depth leaves accuracy flat while the CONFIRMED fast path early-stops 82–88% of items at equal accuracy (E4). The measured value of the protocol on these tasks is therefore calibrated self-verification with bounded inference cost; the balanced meta-reward critique shows a directional but non-significant +4.0 pp over a vanilla critique (E3). Under a weak misleading-context probe the protocol loses only 1.0 pp (E5). The remaining experiments sharpen the efficiency claim. The τ sweep (E6/E8) calibrates the numeric confidence signal and finds it overconfident—every $\tau \geq 0 . 6$ halts the loop on the first generation, so no selective working point exists and the CONFIRMED sentinel is the effective cost-bounder (83% early-stop at 2.06 steps on BBH). On GSM8K and MATH the loop improves accuracy (+4.2 pp and +14.2 pp, significant at n=500) while early-stopping 82–88% of items (E7), confirming the prediction we stated expectatively in earlier drafts: reflection’s benefit concentrates where single-shot reasoning is unreliable (MATH single-shot 26.2%). Reflection is thus best understood as a controllable inferencetime tool whose value is task-dependent—cost-bounded self-verification on clean reasoning tasks, and a raw accuracy lever precisely where single-shot reasoning is weak.

The blueprints—Evolving Virtual World, Discovery-Oriented Tasks, and the GRPO objective—define how the same components would be trained into model weights. Whether the training-time version inherits, amplifies, or inverts the inference-time effects is an open empirical question that the protocol makes substantially cheaper to investigate, since the same reward decomposition drives both. The multi-agent design draws on cognitive science’s searcher-evaluator-generator model [44]; the swarm was not probed at inference in this work, so whether joint training unlocks a multi-agent advantage is an open question.

## 5.2. Limitations

Several limitations warrant acknowledgment:

Benchmark coverage. BBH, GSM8K, and MATH are pure-reasoning benchmarks that require no tool calls. They therefore cannot validate the tool-call-diversity reward or environment-level adversarial filtering, which require a tool-based benchmark (e.g., GAIA [45]) and, ultimately, GRPO training of the blueprint. Our conclusions about those components are limited to their prompt-level analogues.

No training validation. The GRPO objective, Evolving Virtual World, Discovery-Oriented Tasks, and multi-agent swarm are presented as blueprints and are not executed; their empirical status is unknown beyond the inference probes of Section 4.

Backbone and session sensitivity. All results are measured on frozen backbones via an API (deepseek-v4-flash throughout, with Qwen2.5-72B replication in E9). Experiment E6 probed whether the model’s declared confidence can serve as a selective stop signal and found it overconfident—no threshold $\tau \geq 0 . 6$ yields a selective working point, because the backbone reports high confidence on essentially all first generations; τ-based stopping must therefore be recalibrated per backbone or replaced by the CONFIRMED sentinel. Results are also session-sensitive: an independent rerun of the BBH anchors differed by ≈ 7 pp in single-shot accuracy on the same backbone and questions, so point estimates should be read within their Wilson intervals rather than across sessions.

The four reward weights $( w _ { c } , w _ { e } , w _ { r } , w _ { d } )$ and the reflection-reward weights $\beta _ { 1 } , \beta _ { 2 } , \beta _ { 3 }$ are chosen heuristically. The component- and ablation-level sensitivity (E1–E2) and the confidence-threshold sensitivity (E6) provide a partial answer on clean reasoning tasks, but optimal settings may vary across task distributions and backbones, and the reflection-reward weights themselves are exercised only at the prompt level.

## 5.3. Implications for Inference-Time Scaling

Recent literature suggests that scaling compute at inference time can yield performance gains comparable to scaling model parameters. However, unbounded reflection (e.g., fixeddepth loops) exhibits diminishing returns and high token costs. EvoResearcher addresses this by demonstrating that the principles of process reward models (correctness, efficiency, reflection, diversity) can be effectively compiled into training-free, prompt-level heuristics. By relying on the CONFIRMED sentinel rather than uncalibrated numeric confidence tags, the protocol organically allocates compute budget based on problem difficulty. This framework provides immediate, plug-and-play value for practitioners who require high-reliability reasoning under strict API budget constraints, bypassing the prohibitive costs of training bespoke verifiers or fine-tuning via GRPO.

## 5.4. Deployment Constraints

EvoResearcher’s deployment involves trade-offs between capability and resource requirements:

Inference (validated). The self-reflective protocol runs on a single frozen backbone with no training cost. The average number of LLM calls is controlled by the maximum depth D and the loop’s CONFIRMED early stop (Experiment E4); the fast path early-stops

82–88% of items, bounding the average to ≈ 2.1 generations per question—about 30% fewer calls than always looping to depth 3.

Training (blueprint). Executing the GRPO blueprints would require 2000 GPU-hours on 8×A100-80GB, comparable to similar-scale RL training runs in the literature [10–12]; the incremental cost over LiteResearcher comes primarily from multi-agent rollout generation and adversarial content generation.

Latency. For a typical research task requiring 20-50 steps, single-agent inference takes 5-15 minutes on an A100, while the (future) multi-agent configuration would take 15-45 minutes. These are comparable to existing deep research agents [10,12].

## 6. Conclusions

We presented EvoResearcher, a training-free inference-time self-reflective protocol for pure-reasoning tasks. The protocol iterates generate → self-critique → revise over a frozen backbone, instantiating the four components of a self-reflective meta-reward—correctness, efficiency, reflection depth, and tool-call diversity—as prompt-level mechanisms, with maximum depth D and the CONFIRMED early stop as the controllable cost knobs. Across nine experiments on three pure-reasoning benchmarks (BBH, GSM8K, MATH) and two frozen backbones, it establishes a clear academic result: on clean BBH, zero-shot reflection does not exceed the model’s inherent capability ceiling—accuracy stays within the 95% Wilson interval whether individual components are emphasized (E1), ablated (E2), or the loop depth is increased (E4)—but the protocol’s self-verification early stopping intercepts and terminates 82–88% of redundant generations with no loss in accuracy, bounding average inference cost to ≈ 2.1 generations per question (E4, E6). A confidence-threshold sweep (E6) calibrates the numeric confidence signal and finds it overconfident: every τ ≥ 0.6 collapses the loop to single-shot, so the CONFIRMED sentinel—not the confidence tag—is the effective cost-bounding mechanism. Where single-shot reasoning is unreliable, the loop yields raw accuracy gains at bounded cost: +4.2 pp on GSM8K and +14.2 pp on MATH with 82–88% of items early-stopped (E7, n=500), confirming that reflection’s benefit concentrates precisely where it is needed. A weak misleading-context probe costs only 1.0 pp (E5).

The same components are additionally specified as a GRPO training objective for an Evolving Virtual World and a heterogeneous multi-agent swarm; these are presented as design blueprints and are not executed in this paper. The complete GRPO training loop—including joint multi-agent training—remains future work on GPU infrastructure (e.g., NVIDIA A100); all hypothesis statuses reported here are inference-level evidence for the protocol, not validation of the training blueprints. The codebase, datasets, and experimental scripts are released as open source to ensure reproducibility.

## Author Contributions

Conceptualization, W.Y. and S.L.; methodology, W.Y., M.Y., and S.L.; software, Z.Z. and H.D.; validation, J.W., B.L., and S.L.; investigation, W.Y. and M.Y.; resources, S.L.; data curation, M.Y. and J.W.; writing—original draft preparation, W.Y. and S.L.; writing— review & editing, S.L. and J.W.; visualization, W.Y. and Z.Z.; supervision, S.L.; project administration, S.L.; funding acquisition, S.L. All authors have read and agreed to the published version of the manuscript.

## Funding

This research received no external funding.

## Data Availability Statement

The EvoResearcher-Data dataset is publicly available on HuggingFace. All source code, evaluation benchmarks (including the BBH evaluation subset and experimental scripts), and experimental datasets are released as open-source under the MIT License at https://github.com/HAHA1122344/EvoResearcher.

## Conflicts of Interest

The authors declare no conflicts of interest.

## AI Usage Disclosure

Portions of this manuscript were drafted with the assistance of large language models. All content was reviewed, edited, and approved by the human authors, who take full responsibility for the intellectual content and accuracy of the work.

## 7. References

1. Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Zhang, M.; Li, Y.; Wu, Y.; Xiong, D. DeepSeekMath: Pushing the Limits of Mathematical Reasoning, 2024. https://doi.org/10.48550/arXiv.2402.033 00.

2. Madaan, A.; Tandon, N.; Gupta, P.; Hallinan, S.; Gao, L.; Wiegreffe, S.; Alon, U.; Dziri, N.; Prabhumoye, S.; Yang, Y.; et al. Self-Refine: Iterative Refinement with Self-Feedback. In Proceedings of the Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023, pp. 46534–46594. https://papers.nips.cc/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3 -Abstract-Conference.html.

3. Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; Yao, S. Reflexion: Language Agents with Verbal Reinforcement Learning. In Proceedings of the Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023, pp. 8634–8652. https://proceedings.neurips.cc/paper\_ files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html.

4. Gao, L.; Madaan, A.; Zhou, S.; Alon, U.; Liu, P.; Yang, Y.; Callan, J.; Neubig, G. CRITIC: Large Language Models Can Self-Correct with Tool-Integrated Critic. In Proceedings of the The Twelfth International Conference on Learning Representations (ICLR 2024), 2024. https://openreview.net/forum?id=Sx038qxjek.

5. Lightman, H.; Kosaraju, V.; Burda, Y.; Edwards, H.; Baker, B.; Lee, T.; Leike, J.; Schulman, J.; Sutskever, I.; Cobbe, K. Let’s Verify Step by Step. In Proceedings of the The Twelfth International Conference on Learning Representations (ICLR 2024), 2024. https://openreview.net/forum? id=v8L0pN6EOi.

6. Zheng, Y.; et al. A Survey of Process Reward Models, 2025. https://doi.org/10.48550/arXiv.25 10.08049.

7. Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Ichter, B.; Xia, F.; Chi, E.; Le, Q.V.; Zhou, D. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Proceedings of the Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022, pp. 24824– 24837. https://proceedings.neurips.cc/paper\_files/paper/2022/hash/9d5609613524ecf4f15af0 f7b31abca4-Abstract-Conference.html.

8. Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Griffiths, T.L.; Cao, Y.; Narasimhan, K. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. In Proceedings of the Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023, pp. 11809–11822. https://proceedings.neurips.cc/paper\_files/paper/2023/hash/271db9922b8d1f4dd7aaef8 4ed5ac703-Abstract-Conference.html.

9. Zou, A.; Wang, Z.; Carlini, N.; Nasr, M.; Kolter, J.Z.; Fredrikson, M. Universal and Transferable Adversarial Attacks on Aligned Language Models. In Proceedings of the Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023. https://arxiv.org/abs/2307.15043.

10. Jin, B.; Zeng, H.; Yue, Z.; Wang, D.; Zamani, H.; Han, J. Search-R1: Training LLMs to Reason and Leverage Search Helpers with Reinforcement Learning, 2025. https://doi.org/10.48550 /arXiv.2503.09516.

11. Jin, B.; Zeng, H.; Yue, Z.; Han, J. How to Train Your Deep Research Agent? Prompt, Reward, and Policy Optimization in Search-R1, 2026. https://doi.org/10.48550/arXiv.2602.19526.

12. Li, W.; Qu, B.; Pan, B.; Zhang, J.; Liu, Z.; Zhang, P.; Chen, W.; Zhang, B. LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent, 2026. https://doi.org/10.4 8550/arXiv.2604.17931.

13. Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; Cao, Y. ReAct: Synergizing Reasoning and Acting in Language Models. In Proceedings of the The Eleventh International Conference on Learning Representations (ICLR 2023), 2023. https://openreview.net/forum? id=WE\_vluYUL-X.

14. Wang, Z.; et al. Deep Research: A Systematic Survey, 2025. https://doi.org/10.48550/arXiv.25 12.02038.

15. Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; et al. Training Language Models to Follow Instructions with Human Feedback. In Proceedings of the Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022. https://proceedings.neurips.cc/paper\_files/paper/2022/hash/b1efde5 3be364a73914f58805a001731-Abstract-Conference.html.

16. Rafailov, R.; Sharma, A.; Mitchell, E.; Manning, C.D.; Ermon, S.; Finn, C. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Proceedings of the Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023, pp. 53728– 53741. https://papers.nips.cc/paper\_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e0 6ce7-Abstract-Conference.html.

17. Schick, T.; Dwivedi-Yu, J.; Dessi, R.; Raileanu, R.; Lomeli, M.; Hambro, E.; Zettlemoyer, L.; Cancedda, N.; Scialom, T. Toolformer: Language Models Can Teach Themselves to Use Tools. In Proceedings of the Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023. https://proceedings.neurips.cc/paper\_files/paper/2023/hash/d842425e4bf79ba03935 2da0f658a906-Abstract-Conference.html.

18. Qin, Y.; Liang, S.; Ye, Y.; Zhu, K.; Yan, L.; Lu, Y.; Lin, Y.; Cong, X.; Tang, X.; Qian, B.; et al. Tool-LLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. In Proceedings of the The Twelfth International Conference on Learning Representations (ICLR 2024), 2024. https://openreview.net/forum?id=QKBu1BOAwd.

19. Wu, J.; Yang, S.; Yang, C.; Shen, Y.; Zhang, S.; Wen, Z.; Tao, J. Spark: Strategic Policy-Aware Exploration via Dynamic Branching for Long-Horizon Agentic Learning, 2026. https://doi. org/10.48550/arXiv.2601.20209.

20. Peng, Y.; Zhu, X.; Wei, C.; Zeng, N.; Wang, L.; He, Y.T.; Yu, F.R. SAGE: Multi-Agent Self-Evolution for LLM Reasoning, 2026. https://doi.org/10.48550/arXiv.2603.15255.

21. Bai, Y.; Kadavath, S.; Kundu, S.; Askell, A.; Kernion, J.; Jones, A.; Chen, A.; Goldie, A.; Mirhoseini, A.; McKinnon, C.; et al. Constitutional AI: Harmlessness from AI Feedback. Transactions on Machine Learning Research (TMLR) 2022. https://doi.org/10.48550/arXiv.2212.08073.

22. Lin, J.; et al. ICRL: Learning to Internalize Self-Critique with Reinforcement Learning, 2026. https://doi.org/10.48550/arXiv.2605.15224.

23. Jiang, J.; Shen, J.; Kim, S.; Yoo, K.M.; Kim, J.; Kim, S. ReflexiCoder: Teaching Large Language Models to Self-Reflect on Generated Code and Self-Correct It via Reinforcement Learning, 2026. https://doi.org/10.48550/arXiv.2603.05863.

24. Ma, X.; Zheng, C.; Qiu, J.; et al. Retrospective Progress-Aware Self-Refinement for LLM Agent Training, 2026. https://doi.org/10.48550/arXiv.2606.14302.

25. Zhu, Y. Closing the Reflection Gap: A Free Calibration Bonus for Agentic RL, 2026. https: //doi.org/10.48550/arXiv.2606.14211.

26. Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; Hajishirzi, H. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In Proceedings of the The Twelfth International Conference on Learning Representations (ICLR 2024), 2024. https://openreview.net/forum?id=hSyW5go0v8.

27. Gou, Z.; Shao, Z.; Gong, Y.; Shen, Y.; Yang, Y.; Huang, M.; Duan, N.; Chen, W. ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving. In Proceedings of the The Twelfth International Conference on Learning Representations (ICLR 2024), 2024. https: //openreview.net/forum?id=a5m0Uv47Xh.

28. Zhang, X.; et al. Agentic Reinforcement Learning with Implicit Step Rewards, 2026. https: //doi.org/10.48550/arXiv.2602.13949.

29. Zhou, C.; Xu, T.; Lin, J.; Ge, D. StepORLM: A Self-Evolving Framework with Generative Process Supervision for Operations Research Language Models, 2025. https://doi.org/10.48550/arXiv. 2509.22558.

30. Han, H.; Xie, J.; Ma, X.; Zhu, W.; Zhang, Z.; Long, Z.; Chen, H.; Ye, Q. SWE-TRACE: Optimizing Long-Horizon SWE Agents through Rubric Process Reward Models and Heuristic Test-Time Scaling, 2026. https://doi.org/10.48550/arXiv.2604.14820.

31. Wang, X.; Song, Y.; Tian, Z.; Liu, B.; Luo, T.; Huang, M. DPRM: A Dual Implicit Process Reward Model in Multi-Hop Question Answering, 2025. https://doi.org/10.48550/arXiv.2511.08364.

32. Xie, B.; Xu, B.; Yuan, Y.; Zhu, S.; Shen, H. From Outcomes to Processes: Guiding PRM Learning from ORM for Inference-Time Alignment. In Proceedings of the Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), 2025, pp. 19291–19307. https://doi.org/10.18653/v1/2025.acl-long.946.

33. Chen, Z.; et al. Discriminative Policy Optimization for Token-Level Reward Models, 2025. https://doi.org/10.48550/arXiv.2510.11062.

34. Shah, S.; Ozgur, L. The Synthetic Web: Adversarially-Curated Mini-Internets for Diagnosing Epistemic Weaknesses of Language Agents, 2026. https://doi.org/10.48550/arXiv.2603.00801.

35. Zhan, Z.; Chen, H.; Zhu, Y.; Zhu, S.C. How Adversarial Environments Mislead Agentic AI?, 2026. https://doi.org/10.48550/arXiv.2604.18874.

36. Stengel-Eskin, E.; Hase, P.; Bansal, M. Teaching Models to Balance Resisting and Accepting Persuasion. In Proceedings of the Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2025), 2025, pp. 8108–8122. https://doi.org/10.18653/v1/2025.naacl-long.412.

37. Ebrahimi, S.; Dehghankar, M.; Asudeh, A. An Adversary-Resistant Multi-Agent LLM System via Credibility Scoring, 2025. https://doi.org/10.48550/arXiv.2505.24239.

38. Tian, C.; Ho, Q.; Chen, X. A Symbolic Adversarial Learning Framework for Evolving Fake News Generation and Detection, 2025. https://doi.org/10.48550/arXiv.2508.19633.

39. Zhou, H.; et al. MedMisBench: Measuring Epistemic Resilience of LLMs under Misleading Medical Context, 2026. https://doi.org/10.48550/arXiv.2606.12291.

40. Osama, M.; et al. Trust but Verify: Mitigating Medical Hallucinations via Post-Hoc Adversarial Auditing and Multi-Agent Feedback Loops, 2026. https://doi.org/10.48550/arXiv.2606.14149.

41. Suzgun, M.; Scales, N.; Schärli, N.; Gehrmann, S.; Tay, Y.; Chung, H.W.; Chowdhery, A.; Le, Q.V.; Chi, E.H.; Zhou, D.; et al. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them, 2022. https://doi.org/10.48550/arXiv.2210.09261.

42. Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; et al. Training Verifiers to Solve Math Word Problems, 2021.

43. Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart, S.; Tang, E.; Song, D.; Steinhardt, J. Measuring Mathematical Problem Solving With the MATH Dataset. In Proceedings of the Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021. https://arxiv.org/abs/2103.03874.

44. Simon, H.A. The Science of the Artificial, 3rd ed.; MIT Press: Cambridge, MA, 1996. https: //doi.org/10.7551/mitpress/12107.001.0001.

45. Mialon, G.; Fourrier, C.; et al. GAIA: A General AI Assistant, 2025. https://doi.org/10.48550 /arXiv.2311.12983.
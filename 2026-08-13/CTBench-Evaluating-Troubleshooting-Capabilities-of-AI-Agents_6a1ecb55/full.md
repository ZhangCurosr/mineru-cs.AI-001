# CTBench: Evaluating Troubleshooting Capabilities of AI Agents in Realistic Telecom Network Operations

Xingyu Yan<sup>∗</sup>, Tingting Dai<sup>∗</sup>, Antonio De Domenico<sup>†</sup>, Mohamed Sana<sup>†</sup>, Nicola Piovesan<sup>†</sup>, Changchang Li<sup>∗</sup>, Bowen Liu<sup>∗</sup>, Kun Jiang<sup>∗</sup>, Mengjie Zhang<sup>∗</sup>, Dingcheng Shan<sup>∗</sup>, Jing-Cheng Pang<sup>∗</sup>, Chenwei Wu<sup>∗</sup>, Sijie Wu<sup>∗</sup>, Lianying Chao<sup>∗</sup>, Haoran Cai<sup>∗</sup>, Jiantao Ye<sup>∗</sup>, Xubin Li<sup>∗</sup>, Simon Mark Lucas<sup>+</sup>, Xin Chen<sup>∗</sup>

<sup>∗</sup>Huawei Technologies, China

<sup>†</sup>Paris Research Center, Huawei Technologies, Boulogne-Billancourt, France <sup>+</sup>Queen Mary University of London

## Abstract

Agents are increasingly considered for automating network operations and maintenance, where engineers must diagnose network faults, optimize configurations to enhance services, and reduce operational costs while acting under strict constraints. However, existing evaluations fail to accurately model real network characteristics or assess agents under partially observable telecom environments with diverse vendors, devices, protocols, and interfaces. In this paper, we introduce CTBench, a public benchmark for assessing whether an agent behaves like a competent telecom troubleshooting engineer. CTBench focuses on root cause analysis and path restoration. Each task is constructed by experts and annotated with rich task metadata, including golden evidence steps. CTBench uses expert-grounded metrics that evaluate both final answers and the diagnostic evidence. Experiments with representative harness-model combinations show that state-of-the-art agents perform very well at identifying endpoints in path-restoration tasks but, more generally, underperform in root cause analysis. In particular, agents struggle with interface state, link-layer, service-management, and other operational faults. Most importantly, even when agents produce plausible or correct final answers, they often fail to provide the evidence-grounded diagnoses required in operational practice. Our results further show that path restoration is generally more resource expensive, yet larger resource usage does not necessarily translate into better diagnosis.

## 1 Introduction

Network operations and maintenance (NetO&M) is a demanding operational domain where engineers must interpret heterogeneous telemetry, issue device-specific commands, reconstruct service paths, and diagnose faults across multiple network layers. Unlike many static question-answering settings, NetO&M requires sequential decision making: an operator must form hypotheses, select diagnostic actions, inspect command outputs, rule out alternative explanations, and produce an evidencesupported conclusion. These requirements make telecom troubleshooting a natural and challenging testbed for evaluating agentic AI systems.

Recent advances in large language model (LLM) agents have demonstrated strong capabilities in multi-step reasoning, tool use, and interactive problem solving [18, 17, 19]. In communication technology, such agents could assist with root cause analysis, path reconstruction, configuration checking, and closed-loop remediation.

Deployment remains difficult because real networks are partially observable, carry cascading crosslayer faults, and expose multi-vendor command interfaces under strict operational safety requirements.

<table><tr><td>Benchmark</td><td>RCA</td><td>Path Restora- tion</td><td>Partial Observ- ability</td><td>Network Hetero- geneity</td><td>Expert- Annotated Evidence</td><td>Multi- Dimensional Metrics</td><td>Real Equip- ment</td></tr><tr><td>NIKA</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td></tr><tr><td>NetArena</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>√</td><td>X</td></tr><tr><td>NetAgentBench</td><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>√</td><td>X</td></tr><tr><td>CTBench (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Conceptual comparison between CTBench and existing telecom benchmarks.

Existing telecom benchmarks do not reproduce these conditions. They assume full observability, use single-domain topologies, and do not focus on real network equipment. In addition, these benchmarks cannot distinguish an agent that is unable to localize a fault in a malfunctioning equipment from one that returns a correct label without supporting evidence. For operational NetO&M, this process-level transparency is a prerequisite for trust, since a diagnosis must be justified before it can be acted on. Furthermore, a realistic benchmark should assess the capabilities of any tool-using agent in a professional setting: acting under partial observability, adapting across heterogeneous interfaces, and attributing network issues across multi-hop propagation rather than stopping at the first visible symptom.

We introduce CTBench, an agentic benchmark for realistic telecom troubleshooting, covering two operational task families. In RCA tasks, the agent must identify the affected node and object and assign a normalized root-cause label. In path-restoration tasks, the agent must identify source and destination endpoints and reconstruct the forwarding path. The benchmark contains 234 expertcurated tasks, 126 for RCA and 108 for path restoration.

Evaluation in CTBench is grounded in the agent reasoning traces and expert practice rather than final answers alone. Each task carries an expert-normalized answer together with expert-validated golden evidence steps that capture the observations and diagnostic actions required to solve it. Tasks are built by 15 senior telecom experts and validated by an independent expert who solves the tasks without access to ground truths. Each task is further annotated with metadata describing its structural difficulty, including evidence observability, vendor and device heterogeneity and protocol complexity.

Our contributions are as follows:

• We introduce CTBench, a public benchmark for evaluating agentic troubleshooting in telecom NetO&M, covering 234 expert-curated RCA and path-restoration tasks.

• We define telecom-expert-grounded capability metrics that separately evaluate localization, rootcause identification, path restoration, evidence acquisition, observability robustness, heterogeneous device/vendor handling, and resource usage.

• We provide a task metadata schema covering observability, device/vendor heterogeneity, protocol complexity, root-cause count, restored path count, fault-propagation chains, golden solution length, and root-cause categories, enabling fine-grained diagnostic analysis beyond aggregate accuracy.

• We evaluate representative agent-model combinations and show that current agents exhibit substantial gaps in evidence-grounded diagnosis, partial-observability reasoning, and robust handling of heterogeneous telecom environments.

## 2 Related Work

## 2.1 Benchmarks for LLM Agent

General-purpose LLM agent benchmarks [9, 5, 6, 10, 2] focus on evaluating multi-step reasoning, tool use, web navigation, software engineering, and interactive planning ability. These benchmarks have been useful for measuring whether agents can decompose tasks, call tools, and synthesize intermediate observations. However, most general agent benchmarks do not capture the specific operational constraints of telecom NetO&M: vendor-specific command syntax, network topology reasoning, control-plane and data-plane interaction, partial device access, and evidence-grounded fault isolation. As a result, high performance on general agent benchmarks does not necessarily imply operational competence in telecom troubleshooting.

## 2.2 Benchmarks for Network and Telecom

Several benchmarks target networking and telecom domains. Knowledge-oriented benchmarks such as TeleQnA [8] and ORAN-Bench [4] evaluate whether models understand telecom specifications or standards. Troubleshooting-oriented benchmarks such as NIKA [16], TeleLogs [12], WirelessAgent++ [13], TelcoAgent-Bench [1], NetAgentBench [14], and NetArena [20] move closer to operational network reasoning by introducing diagnostic cases, configuration tasks, and network automation scenarios [7]. These efforts provide important foundations for evaluating LLMs and agents in communication technology.

Nevertheless, existing benchmarks often leave at least one major operational gap: they may assume full observability, focus on single-domain topologies, use standardized interfaces, or evaluate only final answers. CTBench complements prior work by emphasizing expert-grounded trajectory evaluation, partial observability, heterogeneous network environments, and evidence acquisition. Table 1 summarizes the conceptual comparison between CTBench and existing telecom benchmarks.

## 3 CTBench Benchmark

This section formalizes the tasks in CTBench, and presents the associated datasets.

## 3.1 Task Formulation

CTBench evaluates telecom troubleshooting as an interactive decision-making problem. Each benchmark instance is a task instruction $q _ { \tau }$ , where $\tau \in \{ \mathrm { R C A } , \mathrm { P a t h } \}$ denotes the task type. Let $\pi _ { \theta }$ denote an evaluated agent, including both the agent harness and the underlying model with parameters θ.

At each interaction turn t, the agent selects a diagnostic action conditioned on the task instruction and the interaction history:

$$
a _ { t } \sim \pi _ { \theta } ( \cdot \mid q _ { \tau } , h _ { t - 1 } ) ,\tag{1}
$$

where $a _ { t }$ is an action such as executing a device command or querying an allowed telemetry source, and $h _ { t - 1 } = ( a _ { 1 } , o _ { 1 } , \dotsc , a _ { t - 1 } , o _ { t - 1 } )$ is the history of previous actions and observations. The task environment ${ \mathcal { E } } _ { \tau }$ executes the action and returns an observation $o _ { t } = \mathscr { E } _ { \tau } ( q _ { \tau } , h _ { t - 1 } , a _ { t } )$ , where $o _ { t }$ may contain command outputs, configuration parameters, routing information, interface states, policy rules, or an error message if the action is invalid or unavailable. The full interaction trajectory is $\mathcal { T } ( \boldsymbol { q } _ { \tau } ) = \left( q _ { \tau } , a _ { 1 } , o _ { 1 } , a _ { 2 } , o _ { 2 } , \ldots , a _ { T } , o _ { T } \right)$ , from which, the agent predicted answer $\hat { y } _ { \tau }$ is decoded.

Root Cause Analysis. An RCA task provides a problem description and an interactive diagnostic interface. The agent must issue permitted queries or commands, inspect the returned observations, and output a normalized diagnosis. For RCA tasks, the target answer is a set of normalized root-cause triples:

$$
y _ { \mathrm { R C A } } ^ { * } = \{ ( n _ { i } , o _ { i } , c _ { i } ) \} _ { i = 1 } ^ { m } ,\tag{2}
$$

where $n _ { i }$ is the affected node, $o _ { i }$ is the affected object, and $c _ { i }$ is the normalized root-cause label. The affected object may be an interface, route, tunnel, access control list rule, network address translation policy, virtual private network instance, service object, or another telecom component.

Path Restoration. A path-restoration task asks the agent to reconstruct a service forwarding path. For path-restoration tasks, the target answer is a reconstructed forwarding path comprising an ordered sequence of path elements $p _ { j }$

$$
y _ { \mathrm { P a t h } } ^ { * } = \{ p _ { 1 } , p _ { 2 } , \ldots , p _ { k } \} ,\tag{3}
$$

Each path element $p _ { j }$ may include a node, interface, next hop, policy, or branch decision.

## 3.2 Dataset Construction

CTBench is constructed through an expert-in-the-loop process. Candidate tasks are abstracted from realistic telecom maintenance cases and sanitized to remove sensitive production information. 15 senior telecom domain experts with an average of 20 years of professional experience define the problem statement, available device outputs, permitted command interface, standard answer, and task-side metadata. The current benchmark contains 126 RCA tasks and 108 path-restoration tasks.

![](images/2a782134ab56d6073a06192dc75330844ad1a689bf4833936c976cd8e74d640f.jpg)  
Figure 1: Process of expert-involved data quality review for CTBench. Candidate tasks are sanitized, independently reviewed by telecom experts, checked for answer and evidence consistency, revised when necessary, and retained only after consensus on solvability and operational validity.

To ensure task validity, each candidate task is reviewed by an independent telecom expert who is not involved in the original construction. The reviewer receives the problem description and permitted interface but not the ground-truth answer or intended trajectory. The reviewer independently solves the task, after which the task author and reviewer compare final answer, supporting evidence, and reasoning trajectory. Tasks are retained only when experts reach consensus on solvability, answer completeness, evidence consistency, and alignment with the intended network behavior. Ambiguous, underspecified, or unverifiable cases are revised or discarded.

Figure 1 summarizes this expert-involved data quality workflow, from task construction and sanitiza tion to independent expert review, consensus checking, revision, and final release.

## 3.3 Golden Evidence Annotation

A key feature of CTBench is that it evaluates not only the final answer $\hat { y } _ { \tau }$ , but also the diagnostic process used to obtain it. Each task is associated with an expert-validated golden solution represented as a set of golden actions:

$$
\mathcal { A } ^ { \ast } ( q _ { \tau } ) = \{ a _ { 1 } ^ { \ast } , a _ { 2 } ^ { \ast } , \dots , a _ { T ^ { \ast } } ^ { \ast } \} ,\tag{4}
$$

where each $a _ { i } ^ { * }$ denotes a key diagnostic action that a telecom expert considers necessary for solving the task and $\dot { T } ^ { * }$ is the total number of steps involved.

In CTBench, golden actions are defined from expert trajectories and therefore provide a deterministic reference for evaluating whether an agent can follow the key operational steps used by telecom experts.

For RCA, golden actions capture the diagnostic checks needed to localize and justify the root cause. For path restoration, they capture the checks needed to reconstruct the forwarding path, including endpoint reachability, hop adjacency, routing decisions, interface transitions, policy constraints, tunnel state, or branch behavior. Concrete annotation examples are provided in the supplementary material.

## 3.4 Task Metadata

CTBench annotates tasks with metadata that enables fine-grained analysis of agent capability.

Evidence Observability. For RCA tasks, CTBench defines two levels of evidence observability, summarized in Table 2. This label, denoted by $O ( q _ { \tau } ) \in \{ O _ { 1 } , O _ { 2 } \}$ , indicates whether a fault evidence can be obtained directly through a permitted diagnostic command, or whether the agent must infer the fault from partial observations obtained through indirect commands.

Root-Cause Categories. Each RCA task is assigned to one or more root-cause categories. We define the categorization taxonomy at the level of individual root causes rather than at the task level: a single troubleshooting scenario may involve multiple underlying faults and can therefore be associated with multiple categories. Table 3 summarizes the resulting five-category taxonomy.

<table><tr><td></td><td>Label Observability</td><td>Explanation</td></tr><tr><td>O1 Full</td><td></td><td>The agent can obtain decisive evidence of the fault by directly implementing a permitted diagnostic command.</td></tr><tr><td>O2 Partial</td><td></td><td>The direct diagnostic command is unavailable or fails. Therefore, the agent must rely on indirect commands that expose only partial observations, requiring additional reasoning and cross-checking to identify the correct root cause.</td></tr></table>

Table 2: Observability levels in CTBench.

<table><tr><td></td><td>Label Category</td><td>Distribution</td></tr><tr><td>C1</td><td>Interface State and Link-Layer Faults</td><td>28.04%</td></tr><tr><td>C2</td><td>Security, Network Address Translation (NAT), and Edge Access Control</td><td>24.30%</td></tr><tr><td>C3</td><td>Routing Protocol and Policy Control</td><td>16.82%</td></tr><tr><td>C4</td><td>High Availability and Reliability Mechanisms</td><td>15.89%</td></tr><tr><td>C5</td><td>Service, Management, and Other Operational Faults</td><td>14.95%</td></tr></table>

Table 3: RCA category taxonomy and gold-entry distribution.

Root Cause count. For RCA tasks, CTBench records the number of independent gold root causes, denoted by $N _ { r } ( q _ { \tau } )$ . This metadata captures whether a troubleshooting scenario requires identifying a single failure or multiple simultaneous faults. Tasks with multiple root causes are structurally harder because the agent must distinguish independent causes from downstream symptoms and avoid returning an incomplete or overly broad diagnosis.

Fault-propagation chain. For RCA tasks, CTBench annotates the fault-propagation-chain length $N _ { c } ( q _ { \tau } )$ when applicable. This metadata describes the causal path from the latent root cause to intermediate network states and finally to the observed symptom. Longer or more indirect propagation chains increase diagnostic difficulty because the agent must reason beyond the first visible symptom and identify the underlying cause.

Restored path count. For path-restoration tasks, CTBench records the number of restored forwarding paths in the gold answer, denoted by $N _ { \mathrm { p a t h } } ( q _ { \tau } )$ . A task may require reconstructing a single path or multiple paths caused by branching, redundancy, load balancing, or service-specific forwarding behavior. Multi-path cases are more difficult because the agent must restore path multiplicity rather than only one route.

Protocol complexity. For both RCA and path-restoration tasks, CTBench records protocol complexity, denoted by $\dot { N } _ { p } ( q _ { \tau } )$ ). This metadata captures the number and interaction depth of forwarding, control-plane, policy, redundancy, and service mechanisms involved in solving the task. Higher protocol complexity indicates that the correct answer depends on reasoning across multiple interacting mechanisms rather than inspecting a single local state.

Network Heterogeneity. Network heterogeneity captures the number of distinct vendor environments $N _ { v } ( q _ { \tau } )$ and device types $N _ { d } ( q _ { \tau } )$ the agent must handle. As shown in Table 4, we assign to each task a network heterogeneity level comparing the maximum between vendor heterogeneity and device-type heterogeneity $\bar { h } _ { L } ( q _ { \tau } ) = \operatorname* { m a x } ( \bar { N _ { v } } ( q _ { \tau } ) , \bar { N _ { d } } ( q _ { \tau } ) )$ with task-calibrated thresholds for RCA and path-restoration tasks. Table 4 summarizes the reporting levels.

Golden Solution Length. For both task types, CTBench records the golden solution length $T ^ { * } ( q _ { \tau } ) = | \mathcal { A } ^ { * } ( q _ { \tau } ) |$ |, defined as the number of key evidence steps required to solve the task. This metadata captures the procedural burden of the task: a longer golden path indicates that the agent must collect, connect, and use more pieces of evidence before reaching a justified answer.

Using Metadata to Characterize Task Difficulty. These metadata fields allow CTBench to characterize task difficulty as a structural property of the task. Evidence observability captures information availability; root-cause count and restored path count capture answer multiplicity; fault propagation chain length captures causal depth; protocol complexity captures cross-layer reasoning burden; device/vendor heterogeneity captures semantic and operational diversity; and golden solution length captures the number of expert-required diagnostic steps. For RCA tasks, these dimensions describe the difficulty of localizing, identifying, and justifying root causes. For path-restoration tasks, difficulty is instead driven by endpoint identification, path multiplicity, forwarding structure, protocol interactions, heterogeneity, and the length of the evidence path. We use these annotations both for fine-grained reporting and for analyzing which operational conditions most challenge current agents.

<table><tr><td>Label</td><td>Name</td><td>RCA</td><td>Path Restoration</td></tr><tr><td>H-Low</td><td>Low heterogeneity</td><td> $h _ { L } = 1$ </td><td> $h _ { L } \leq 6$ </td></tr><tr><td></td><td>H-High High heterogeneity</td><td> $h _ { L } \geq 2 h _ { L } > 6$ </td><td></td></tr></table>

Table 4: Network heterogeneity levels. The levels are defined based on the number of vendors and devices type involved in the solution.

![](images/01a439d1b87a3578c92fad93d692b3c84f0d6d2767310b48a405928c4996821e.jpg)  
Figure 2: CTBench Automatic Evaluation framework.

## 4 Evaluation Protocol

Figure 2 illustrates the overall evaluation framework of CTBench. Given a task instance $q _ { \tau }$ and an agent trajectory $\textstyle { \mathcal { T } } ( q _ { \tau } )$ , the evaluator compares the predicted answer $\hat { y } _ { \tau }$ and collected evidence $\hat { A } ( q _ { \tau } )$ against the ground truth $y _ { \tau } ^ { \ast }$ and the related expert-annotated metadata $A ^ { * } ( q _ { \tau } )$ . In addition to the average task accuracy $\mathrm { A c c } \stackrel { \cdot \cdot } { = } \mathbb { E } [ \mathbb { 1 } ( y _ { \tau } ^ { * } = \hat { y } _ { \tau } ) ]$ ], CTBench reports task-specific capability metrics including cost and efficiency metrics.

## 4.1 RCA Capability Metrics

For RCA, CTBench defines three complementary capabilities.

RCA Localization. RCA Localization measures the agent capability to identify where the problem occurs, including the affected node, interface or service object. Let $\mathbf { \bar { \mathbf { \xi } } } L ^ { * } = \{ ( n _ { i } , o _ { i } ) \} _ { i = 1 } ^ { m }$ denote the gold set of affected node-object pairs extracted from $y _ { \mathrm { R C A } } ^ { * }$ and $\hat { L }$ the predicted set extracted from yˆ<sub>RCA</sub>. We compute the RCA localization score (RCA-Loc) as the intersection over union (IoU) between the predicted and gold localization sets:

$$
\mathrm { R C A - L o c } = \mathrm { I o U } ( \hat { L } , L ^ { * } ) ,\tag{5}
$$

where IoU $\begin{array} { r } { \left( A , B \right) = \frac { | A \cap B | } { | A \cup B | } . } \end{array}$

RCA Identification. RCA Identification measures the agent capability to identify the correct rootcause label, such as interface down, missing route or network misconfiguration. Let $C ^ { * } = \{ c _ { i } \} _ { i = 1 } ^ { m }$ denote the gold set of normalized root-cause labels and $\hat { C }$ the predicted set. Similar to RCA-Loc, we compute the RCA Identification score (RCA-ID) as the IoU between the predicted and gold root-cause label sets:

$$
\operatorname { R C A - I D } = \operatorname { I o U } ( { \hat { C } } , C ^ { * } ) .\tag{6}
$$

RCA Evidence. RCA Evidence quantitatively assesses whether the agent collects relevant key diagnostic evidence needed to identify the root cause. To do so, we compare the agent actions with the expert golden actions using the F1 score. Specifically, let $A _ { \mathrm { R C A } } ^ { * }$ and $\hat { \mathcal { A } } _ { \mathrm { R C A } }$ denote the golden evidence and the agent steps, respectively. We compute:

$$
\mathrm { R C A - E v i d e n c e } = \mathrm { F } 1 \big ( \hat { A } _ { \mathrm { R C A } } , \mathcal { A } _ { \mathrm { R C A } } ^ { * } \big ) .\tag{7}
$$

A step is considered covered only if the agent queries the correct device with the correct command to retrieve the relevant observation required for diagnosing the problem.

Remark (Hierarchical Scoring). The main RCA metrics above use strict set matching to accurately assess agent capabilities. To support near-miss analysis, we additionally define hierarchical RCA scoring and topology-aware node similarity as supplementary metrics. These metrics are not used as primary correctness measures; instead, they help distinguish close errorsfrom completely unrelated predictions. The complete taxonomy, similarity rules, procedure, and associated results are provided in the supplementary material.

## 4.2 Path Restoration Capability Metrics

Similar to RCA, CTBench defines three capability metrics.

Path Localization. Path localization measures whether the agent correctly identifies the relevant endpoints involved in the path-restoration task. Let $P _ { \mathrm { e n d } } ^ { * }$ and $\hat { P } _ { \mathrm { e n d } }$ be the gold and predicted endpoint sets. We compute path localization score as the IoU between the gold and the predicted sets:

$$
\mathrm { P a t h – L o c } = \mathrm { I o U } ( \hat { P } _ { \mathrm { e n d } } , P _ { \mathrm { e n d } } ^ { * } ) .\tag{8}
$$

Path Restoration. Path Restoration measures whether the agent reconstructs the correct forwarding path. Let $P ^ { * }$ denote the reference path edge sets and $\hat { P }$ the predicted path edge sets. We compute the path restoration score as IoU between predicted and reference path edge sets:

$$
\mathrm { P a t h - R e s } = \mathrm { I o U } ( \hat { P } , P ^ { * } ) .\tag{9}
$$

Path Evidence. Path Evidence measures whether the agent collects and uses the key evidence required to reconstruct the path, such as routing entries, interface states or forwarding-table outputs. Similar to RCA evidence score, let $A _ { \mathrm { P a t h } } ^ { * }$ and $\hat { \mathcal { A } } _ { \mathrm { P a t F } }$ denote the golden evidence steps and the agent steps, respectively. We compute:

$$
\mathrm { P a t h – E v i d e n c e } = \mathrm { F } 1 \left( \hat { A } _ { \mathrm { P a t h } } , A _ { \mathrm { P a t h } } ^ { * } \right) .\tag{10}
$$

## 4.3 Cost and Efficiency Metrics

We additionally report latency, interaction rounds, and token consumption. Token consumption is defined as:

$$
\mathrm { T o k e n s _ { t o t a l } = T o k e n s _ { i n p u t } + T o k e n s _ { o u t p u t } . }\tag{11}
$$

These quantities are treated as cost and efficiency descriptors rather than substitutes for diagnostic capability.

<table><tr><td rowspan="2">Agent</td><td colspan="4">RCA</td><td colspan="4">Path Restoration</td></tr><tr><td>RCA Acc</td><td>RCA-Loc</td><td>RCA-ID</td><td>RCA-Evid.</td><td>Path Acc</td><td>Path-Loc</td><td>Path-Rest</td><td>Path-Evid.</td></tr><tr><td>Codex+GPT-5.5</td><td>47.62±4.5</td><td>52.90±3.7</td><td>66.83±3.7</td><td>15.80±1.0</td><td>87.96±3.2</td><td>99.38±0.6</td><td>95.28±1.4</td><td>47.84±1.5</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>19.84±3.6</td><td>34.68±3.4</td><td>38.96±3.8</td><td>12.62±0.8</td><td>17.59±3.7</td><td>87.65± 2.8</td><td>48.19±3.0</td><td>26.58±1.6</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>17.46±3.4</td><td>29.43±3.6</td><td>40.08±3.9</td><td>10.36±0.8</td><td>26.85±4.3</td><td>85.49±2.9</td><td>59.26±2.8</td><td>21.85±1.3</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>25.40±3.9</td><td>37.59±3.652.38±3.9</td><td></td><td>12.59±0.8</td><td>40.74±4.8</td><td>81.48± 3.6</td><td>59.00±3.5</td><td>24.06±1.7</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>4.76±1.9</td><td>12.41±2.911.37±2.5</td><td></td><td>5.23±0.7</td><td>1.85±1.3</td><td>75.31±3.6</td><td>10.71±2.0</td><td>6.13±0.8</td></tr></table>

Table 5: Overall capability results. Scores are percentages and reported as mean ± standard deviation error.

## 5 Experiments and Analysis

We evaluate five representative agent-model combinations: Codex+GPT-5.5, ClaudeCode+Qwen3.7-Plus, HermesAgent+DeepSeek-V4-Pro, HermesAgent+Qwen3.7-Max, and HermesAgent+TelecomGPT-R1 [15]. All agents are evaluated on the same task set and are not given access to ground-truth answers or golden evidence steps during inference. Each task is executed in a reactive sandbox environment that exposes the permitted telecom diagnostic interface. Agents interact with the environment by issuing commands or tool calls and receiving observations. We log the full trajectory, including device queries, returned observations, final answers, latency, interaction rounds, and token usage. The same answer normalization and evidence-matching rules are applied across all agents. Full environment details, prompts, and code are provided in the supplementary material.

## 5.1 Overall Capability Results

Table 5 reports the main capability results. We report the task accuracy together with the capability metrics presented in Sec. 4: localization, identification, restoration, and evidence acquisition. In our experiments, agents perform better on path-restoration tasks than on RCA tasks. Codex+GPT-5.5 obtains the strongest overall results, and it performs especially well on path restoration.

The results show also that final accuracy alone hides important differences between agent capabilities. For example, Codex+GPT-5.5 reaches only 47.84% Path-restoration evidence, showing that producing a correct or plausible answer does not imply that the agent produces the evidence expected from a telecom troubleshooting engineer. Similarly, an agent (e.g., HermesAgent+DeepSeek-V4-Pro) may achieve high path-localization performance (Path-Loc) while still failing to restore the full path (Path Macro IoU) or collect sufficient diagnostic evidence (Path-Evidence). In addition, agents are better in RCA identification than in RCA localization e.g., 66.83% vs 52.9% for Codex+GPT-5.5 and 52.38% vs 37.59% for HermesAgent+Qwen3.7-Max, respectively, indicating that agents sometimes infer the right fault type but fail to identify the precise faulty node or object.

## 5.2 Cost and Efficiency

Table 6 reports interaction rounds, latency, and token consumption. These metrics are cost and efficiency descriptors, rather than capability measures. The results show that higher computational cost does not necessarily translate into stronger diagnostic capability. For instance, ClaudeCode+Qwen3.7- Plus consumes substantially more rounds and tokens than Codex+GPT-5.5, but obtains lower overall capability scores. In constrast, HermesAgent+Qwen3.7-Max outperforms HermesAgent+DeepSeek-V4-Pro in path restoration (see Path Acc) but at a cost of higher latency and token consumption. This motivates reporting cost alongside capability rather than using latency or token consumption as proxies for agent competence.

Remark (Human expert results). As a reference, during the independent review phase (see Figure 1), human experts have achieved 92.6% and 56.4% accuracy on path-restoration tasks and RCA tasks, respectively. In addition, it took them between 40 and 60 minutes to complete each of these tasks.

## 5.3 RCA Performance by Fault Category

Figure 3 visualizes RCA accuracy across the five root-cause categories presented in Table 3. The category-level results show that agent failures are not uniformly distributed across telecom fault families. Two patterns holds across all agents. C2 faults are the most reliably diagnosed by every agent, and C1 is uniformly hard, with accuracy between 0.00% and 10.71%. The remaining ordering is agent-dependent. C3 separates the agents sharply: no agent other than Codex+GPT-5.5 solves a single C3 task, while Codex+GPT-5.5 reaches 40.00% and fails instead on C1. Codex+GPT-5.5 is also the only agent with a wide gap between C4 and C5 (50.00% versus 28.12%); the remaining agents differ by at most four points on these two categories. This suggests that CTBench does not only measure generic troubleshooting ability: it also reveals which telecom fault families remain challenging for current agents.

<table><tr><td>Agent</td><td>Task Rds</td><td>Lat.(s)</td><td>Tokens</td></tr><tr><td>Codex+GPT-5.5</td><td>RCA 10.81</td><td>333.40</td><td>476.5k</td></tr><tr><td>Codex+GPT-5.5</td><td>Path 14.14</td><td>419.80</td><td>1019.2k</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus ClaudeCode+Qwen3.7-Plus</td><td>RCA 81.36 Path 93.85</td><td>1234.64 1241.09</td><td>2751.7k 3143.4k</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro HermesAgent+DeepSeek-V4-Pro</td><td>RCA 36.62 Path 38.66</td><td>483.62 497.84</td><td>1491.7k 1453.1k</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>RCA 31.09</td><td>1352.93</td><td>1218.6k</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>Path 36.77</td><td>1482.64</td><td>1574.6k</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td></td><td></td><td></td></tr><tr><td></td><td>RCA 31.76</td><td>641.77</td><td>656.2k</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>Path 35.53</td><td>934.40s</td><td>797.1k</td></tr></table>

Table 6: Efficiency and cost results.

![](images/d467695d5c8f58e28e3126fb67d819adbe0693155e17dc956e5122f95258dd5d.jpg)  
Figure 3: RCA accuracy across root-cause categories for the five evaluated agent-model combinations. The radar plot shows that agent performance varies substantially across fault families rather than shifting uniformly across all categories.

## 5.4 Impact of Observability and Heterogeneity

We next analyze how task metadata explains agent behavior. To save space, the main paper reports this analysis for ClaudeCode+Qwen3.7-Plus as a representative agent. Full results for all agents are provided in the supplemental material.

Observability. For ClaudeCode+Qwen3.7-Plus, RCA performance drops under partial observability. On fully observable RCA tasks, the agent obtains 20.56% accuracy, 37.02% localization, 42.92% RCA identification, and 38.29% evidence coverage. On partially observable tasks, accuracy decreases to 15.79%, localization drops to 17.14%, and RCA identification drops to 16.22%. Interestingly, evidence coverage increases to 62.20%, suggesting that the agent can still collect some indirect observations but struggles to convert them into precise fault localization and root-cause identification. This supports the role of observability as a structural difficulty factor: partial evidence does not merely slow the agent down; it changes the nature of the reasoning required.

Network heterogeneity. Network heterogeneity also affects performance, especially for path restoration. For ClaudeCode+Qwen3.7-Plus, path accuracy decreases from 29.31% in low heterogeneity settings to 4.00% in high heterogeneity settings. Path localization also decreases from 94.96% to 72.73%, while path restoration IoU drops from 68.58% to 35.05%. This indicates that heterogeneous multi-device settings make it harder for the agent to integrate evidence across device roles, vendor-specific command outputs, and forwarding contexts. For RCA, the effect is less uniform: accuracy decreases from 33.33% to 13.79%, while localization and RCA identification remain similar or slightly increase. This suggests that heterogeneity interacts with other factors such as fault category, observability, and protocol complexity, and should not be used alone to define task complexity.

## 6 Discussion

Performance is multi-faceted. A key lesson from CTBench is that agent performance should be measured in detailed ways: separating localization, identification, restoration, and evidence acquisition exposes distinct variations in abilities.

Evidence-grounded diagnosis remains difficult. Even when agents produce plausible final answers, they often do not produce evidences that telecom experts consider necessary. This is critical for operational trust, where a diagnosis must be justified before remediation.

Partial observability and heterogeneity expose agent limitations. Under partial observability, agents must reason from indirect evidence and cross-checks. Under heterogeneity, they must normalize device roles, vendor-specific commands, and output formats. The observed drops in localization, identification, and restoration show that these are not merely implementation details; they are core dimensions of telecom-agent capability.

CTBench reflects general agent challenges. Although focusing on telecom operations, CTBench presents broader agentic capabilities required in complex professional environments, including partial-observation reasoning, interaction with heterogeneous environment, causal attribution, and long-horizon evidence planning.

The supplementary material provides full capability analysis derived from trajectory-level failures.

## 7 Limitations

CTBench can be extended in future work. First, CTBench covers only two types of tasks: Path-Restoration and RCA. The capability of dynamically repairing faults in the network environment by the evaluation agent is not covered. This will be supplemented in the future. In addition, while the benchmark covers realistic telecom troubleshooting, it does not yet exhaustively cover all domains such as full radio-access-network operations, core-network slicing, or cloud-native telecom infrastructure.

## 8 Conclusion

We presented CTBench, an agentic benchmark for realistic telecom network operations and maintenance. Using with fine-grained data annotations and multi-dimensional metrics, CTBench assesses agent proficiency in fault localization, root-cause isolation, path reconstruction, expert-aligned evidence extraction, and reasoning robustness under partial observability and heterogeneous environments. Experiments with representative agent-model combinations reveal that current agents remain far from reliable telecom troubleshooting operators, especially in evidence-grounded RCA and partially observable settings. We hope CTBench will support future research on trustworthy, domain-grounded, and operationally safe AI agents for communication networks.

## References

[1] Lina Bariah, Brahim Mefgouda, Farbod Tavakkoli, Enrique Molero, Louis Powell, and Merouane Debbah. Telcoagent-bench: A multilingual benchmark for telecom ai agents, 2026.

[2] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-Bench: Evaluating conversational agents in a dual-control environment. In Proceedings ofthe Forty-Third International Conference on Machine Learning (ICML), 2025.

[3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[4] Pranshav Gajjar and Vijay K. Shah. ORAN-Bench-13K: An open source benchmark for assessing LLMs in open radio access networks. In Proceedings of the 2025 IEEE 22nd Consumer Communications & Networking Conference (CCNC), pages 1–4, 2025.

[5] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In Proceedings of the Twelfth International Conference on Learning Representations (ICLR), 2024.

[6] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. AgentBench: Evaluating LLMs as agents. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR), 2024.

[7] Ali Maatouk, Kenny Chirino Ampudia, Rex Ying, and Leandros Tassiulas. Tele-llms: A series of specialized large language models for telecommunications. IEEE Access, 14:86424–86441, 2026.

[8] Ali Maatouk, Fadhel Ayed, Nicola Piovesan, Antonio De Domenico, Merouane Debbah, and Zhi-Quan Luo. TeleQnA: A benchmark dataset to assess large language models telecommunications knowledge. IEEE Network, 2025.

[9] Gregoire Mialon, Cl´ ementine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas´ Scialom. GAIA: a benchmark for general AI assistants. In Proceedings ofthe Twelfth International Conference on Learning Representations (ICLR), 2024.

[10] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Hao Peng, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR), 2024.

[11] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal ofcomputer vision, 115:211–252, 2015.

[12] Mohamed Sana, Nicola Piovesan, Antonio De Domenico, Yibin Kang, Haozhe Zhang, Merouane Debbah, and Fadhel Ayed. Reasoning language models for root cause analysis in 5g wireless networks, 2025.

[13] Jingwen Tong, Zijian Li, Fangyu Liu, Wei Guo, and Jun Zhang. Wirelessagent++: Automated agentic workflow design and benchmarking for wireless networks, 2026.

[14] Ahmed Twabi, Yepeng Ding, and Tohru Kondo. Netagentbench: A state-centric benchmark for evaluating agentic network configuration, 2026.

[15] Bohao Wang, Chenwei Wu, Haoyu Li, Hang Zou, Yu Tian, Lina Bariah, Chongwen Huang, Yongliang Shen, Zhaoyang Zhang, and Merouane Debbah. TelecomGPT-R1: Post-training´ recipes for universal reasoning in telecom. https://huggingface.co/KU-DFI/TelecomGPT-R1, 2026.

[16] Zhihao Wang, Alessandro Cornacchia, Alessio Sacco, Franco Galante, Marco Canini, and Dingde Jiang. A network arena for benchmarking ai agents on network troubleshooting. In Proceedings of the Fourteenth International Conference on Learning Representations (ICLR), 2026.

[17] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824– 24837, 2022.

[18] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, Qi Zhang, and Tao Gui. The rise and potential of large language model based agents: A survey. Science China Information Sciences, 68(2):121101, 2025.

[19] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In Proceedings of the Eleventh International Conference on Learning Representations (ICLR), 2023.

[20] Yajie Zhou, Jiajun Ruan, Eric S. Wang, Sadjad Fouladi, Francis Y. Yan, Kevin Hsieh, and Zaoxing Liu. Netarena: Dynamic benchmarks for ai agents in network automation. In Proceedings ofthe Fourteenth International Conference on Learning Representations (ICLR), 2026.

## Supplementary Material for Paper:

CTBench: Evaluating Troubleshooting Capabilities of AI Agents in Realistic Telecom Network Operations

## Table of Contents

A Dataset and Annotation Details 14   
A.1 RCA task 14   
A.2 Path Restoration task 15   
A.3 Annotation Protocol 15   
A.4 Metadata Schema 16   
A.5 RCA Metadata Distributions 16   
A.6 Path-Restoration Metadata Distributions 17   
A.7 Task Examples 17   
B Supplementary Evaluation Metrics 17   
B.1 Hierarchical RCA Scoring 18   
B.2 Topology-Aware Node Similarity 19   
B.3 Root-Cause Type Hierarchy . 19   
C Supplementary Evaluation Results 19   
C.1 Full RCA Category Results . 19   
C.2 Hierarchical RCA Results . 19   
C.3 Observability Results for All Agents 19   
C.4 Heterogeneity Results for All Agents . 19   
C.5 Structural Difficulty Trends . 20   
C.6 Evidence-Grounded Performance Analysis . 22   
C.7 Harness Sensitivity Analysis 23   
D Reproducibility Details 24   
D.1 Evaluation Environment 24   
D.2 Agent Harnesses and Concurrency 25   
D.3 Command Form 25   
D.4 Prompt Template 25   
E Trajectory-Level RCA Failure Analysis 26   
E.1 Overview of Trajectory-Level Failure Analysis 26   
E.2 Failure Mode Taxonomy 27   
E.3 Representative Failure Trajectories 27   
E.4 Implications for Agent Design 28   
F Discussion of Agent Capabilities in CTBench 29

![](images/37f9af658879d549f52841a005d8ff8b704a4fc1b6ec0cfc7b006c47f908c092.jpg)  
Figure 4: Representation of the telecom network under study with multi-vendor equipment and realistic networking protocols.

## A Dataset and Annotation Details

CTBench is constructed through an expert-in-the-loop process. CTBench involves real network equipment with different vendors, equipment, protocols, and interfaces (see Figure 4). Candidate tasks are abstracted from realistic telecom maintenance cases and sanitized to remove sensitive production information. 15 senior telecom domain experts with an average of 20 years of professional experience define the problem statement, available device outputs, permitted command interface, standard answer, and task-side metadata. The current benchmark contains 126 RCA tasks and 108 path-restoration tasks.

## A.1 RCA task

The RCA subset contains 126 tasks and 214 normalized root-cause triples. Since a single troubleshoot ing task may contain multiple independent faults, the number of root-cause entries can exceed the number of tasks. The dataset also contains 176 task-level root-cause mentions, where each root-cause type is counted at most once per task. The difference between 214 entries and 176 mentions reflects repeated affected objects or devices for the same root-cause type within some tasks.

The 17 normalized root-cause labels are grouped into seven operational fault domains for dataset characterization. The main paper merges sparse domains into C5 for stable reporting, while Table 7 reports the complete unmerged distribution.

<table><tr><td>Fault domain</td><td>Types</td><td>Entries</td><td>Ratio</td></tr><tr><td>Interface and link-state faults</td><td>1</td><td>60</td><td>28.04%</td></tr><tr><td>Security, NAT, and boundary access control</td><td>3</td><td>52</td><td>24.30%</td></tr><tr><td>Routing protocol and policy control</td><td>2</td><td>36</td><td>16.82%</td></tr><tr><td>High availability and reliability mechanisms</td><td>2</td><td>34</td><td>15.89%</td></tr><tr><td>Overlay and VPN service faults</td><td>4</td><td>26</td><td>12.15%</td></tr><tr><td>Basic addressing and Layer-2 access configuration</td><td>4</td><td>5</td><td>2.34%</td></tr><tr><td>Monitoring and operational visibility</td><td>1</td><td>1</td><td>0.47%</td></tr><tr><td>Total</td><td>17</td><td>214</td><td>100.00%</td></tr></table>

Table 7: Root-cause category coverage in CTBench fault-localization tasks.

Table 8 reports the fine-grained root-cause distribution. The distribution is intentionally not classbalanced: CTBench preserves frequent operational failure modes while retaining diagnostically distinct long-tail cases. We further merge the fine-grained root-causes into five-category taxonomy shown in Table 9.

<table><tr><td>Fine-grained root cause</td><td>Fault domain</td><td>Entries</td><td>Ratio</td><td>Task ment.</td></tr><tr><td>Shutdown</td><td>Interface/link state</td><td>60</td><td>28.04%</td><td>28</td></tr><tr><td>Security policy does not permit the user Security/NAT/access control</td><td></td><td>41</td><td>19.16%</td><td>41</td></tr><tr><td>IP prefix list misses the corresponding Routing/policy control source IP</td><td></td><td>24</td><td>11.21%</td><td>24</td></tr><tr><td>Global STP is not enabled</td><td>High availability/reliability</td><td>23</td><td>10.75%</td><td>23</td></tr><tr><td>L3VPN misconfiguration</td><td>Overlay/VPN service</td><td>17</td><td>7.94%</td><td>17</td></tr><tr><td>OSPF misconfiguration</td><td>Routing/policy control</td><td>12</td><td>5.61%</td><td>6</td></tr><tr><td>Global HRP hot-standby protocol is not High availability/reliability enabled</td><td></td><td>11</td><td>5.14%</td><td>11</td></tr><tr><td>NAT outside-interface attribute error or Security/NAT/access control omission</td><td></td><td>7</td><td>3.27%</td><td>7</td></tr><tr><td>NAT inside-interface attribute error or Security/NAT/access control omission</td><td></td><td>4</td><td>1.87%</td><td>4</td></tr><tr><td>SRv6-Policy tunnel planning error</td><td>Overlay/VPN service</td><td>4</td><td>1.87%</td><td>4</td></tr><tr><td>VXLAN misconfiguration</td><td>Overlay/VPN service</td><td>4</td><td>1.87%</td><td>4</td></tr><tr><td>MAC address misconfiguration</td><td>Basic L2/access config</td><td>2</td><td>0.93%</td><td>2</td></tr><tr><td>Interface IP error</td><td>Basic L2/access config</td><td>1</td><td>0.47%</td><td>1</td></tr><tr><td>Interface VLAN misconfiguration</td><td>Basic L2/access config</td><td>1</td><td>0.47%</td><td>1</td></tr><tr><td>VPN configuration missing</td><td>Overlay/VPN service</td><td>1</td><td>0.47%</td><td>1</td></tr><tr><td>Host-information collection missing</td><td>Monitoring/visibility</td><td>1</td><td>0.47%</td><td>1</td></tr><tr><td>Loopback IP conflict</td><td>Basic L2/access config</td><td>1</td><td>0.47%</td><td>1</td></tr><tr><td>Total</td><td></td><td>214</td><td>100.00%</td><td>176</td></tr></table>

Table 8: Fine-grained root-cause distribution in CTBench. Task mentions count each root-cause type at most once per task

<table><tr><td>Category</td><td>Entries</td><td>Ratio</td></tr><tr><td>C1 Interface/link state</td><td>60</td><td>28.04%</td></tr><tr><td>C2 Security/NAT/access</td><td>52</td><td>24.30%</td></tr><tr><td>C3 Routing/policy</td><td>36</td><td>16.82%</td></tr><tr><td>C4 HA/reliability</td><td>34</td><td>15.89%</td></tr><tr><td>C5 Service/management/other</td><td>32</td><td>14.95%</td></tr></table>

Table 9: Merged RCA category distribution over 214 gold root-cause entries.

## A.2 Path Restoration task

The path-restoration subset contains 108 tasks. Each task asks the agent to reconstruct the forwarding path from a source endpoint to a destination endpoint or destination IP. Unlike RCA tasks, whose answers are sets of root-cause triples, path-restoration answers are ordered network paths. A valid answer must preserve hop order, path multiplicity, and physical egress-interface semantics. The required output uses one line per path, connects consecutive hops with ->, writes each non-terminal hop as node egress-interface, and writes the terminal endpoint as the node name only.

Each golden path is decomposed into normalized node records with node names, egress interfaces, and raw output segments. This representation supports exact-match scoring while also enabling more diagnostic overlap measures, such as endpoint correctness, ordered edge overlap, interface overlap, and path-count correctness.

## A.3 Annotation Protocol

For RCA tasks, standard answers are normalized into one or more minimal root-cause triples of the form (fault node, fault object, root cause). For path-restoration tasks, standard answers preserve node order, path multiplicity, and interface semantics. Golden solution paths are annotated as key evidence steps rather than complete command logs. Each retained step must correspond to necessary evidence acquisition, verification, or use. For multi-root-cause and multi-path tasks, reviewers check answer completeness and evidence sufficiency rather than enforcing a single unique reasoning trajectory. Cases with unresolved expert disagreement are revised or removed before release.

<table><tr><td colspan="2">Evidence Observability</td></tr><tr><td>Tier</td><td>N Ratio</td></tr><tr><td>O1 direct O2 indirect</td><td>10784.92% 1915.08%</td></tr><tr><td>Golden Solution Path</td><td></td></tr><tr><td>Tier</td><td>N Ratio</td></tr><tr><td>Short (≤ 4)</td><td>7357.94%</td></tr><tr><td>Long (&gt; 4)</td><td>5342.06%</td></tr></table>

<table><tr><td>Metadata name</td><td>Fault localization</td><td>Path restoration</td></tr><tr><td>Golden answer</td><td>Normalized root-cause triples in standard_answer.faults</td><td>Ordered endpoint-to-endpoint paths in standard_answer.paths</td></tr><tr><td>count</td><td>Root-cause / path data_labels.root_cause_count; data_labels.has_multiple_root_causeata_labels.unique_node_count</td><td>data_labels.restored_path_count;</td></tr><tr><td></td><td>RCA category labels data_labels.rca_category_labels; data_labels.rca_categories</td><td>Not applicable</td></tr><tr><td>ity</td><td>Evidence observabil- evidence_observability.tier; direct/degraded observation pairs</td><td>Not applicable</td></tr><tr><td>Fault-propagation chain</td><td>fault-propagation_chain. total_propagation_step_count; causal propagation steps</td><td>Not applicable</td></tr><tr><td>Golden solution path golden_solution_path.</td><td>golden_solution_length; diagnostic golden_solution_length; key steps</td><td>golden_solution_path. path- reconstruction key steps</td></tr><tr><td>ity</td><td>Network heterogene- network_heterogeneity.level; vendor/type/device counts and key devices vendor/type/device counts and key devices</td><td>network_heterogeneity.level; protocol_complexity.</td></tr><tr><td>Protocol complexity protocol_complexity.</td><td>mechanism_count; root-cause, diagnos- mechanism_count; forwarding, diagnos- tic, propagation, and observability mecha- tic, and path-restoration mechanisms nisms</td><td></td></tr></table>

Table 10: Metadata fields for the two evaluated CTBench task types.

<table><tr><td colspan="3">Root-Cause Count</td></tr><tr><td>Tier</td><td>N</td><td>Ratio</td></tr><tr><td>RC1</td><td>66 52.38% 44 34.92%</td><td></td></tr><tr><td>RC2 RC3+</td><td>1612.70%</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Device Heterogeneity</td><td></td><td></td></tr><tr><td>Tier</td><td>N</td><td>Ratio</td></tr><tr><td>H-Low (=1)</td><td>3930.95%</td><td></td></tr><tr><td>H-High (i1)</td><td></td><td>87 69.05%</td></tr></table>

<table><tr><td colspan="2">Fault-Propagation Chain</td></tr><tr><td>Tier N</td><td>Ratio</td></tr><tr><td>Low (≤ 4) Medium (5–13) High (&gt; 13)</td><td>1915.08% 87 69.05% 2015.87%</td></tr><tr><td>Protocol Complexity</td><td></td></tr><tr><td>Tier N</td><td>Ratio</td></tr><tr><td>Low/Mod. (≤ 7) High (&gt; 7)</td><td>7660.32% 50 39.68%</td></tr></table>

Table 11: RCA question-level component distributions over 126 tasks.

## A.4 Metadata Schema

CTBench annotates each task with task-side metadata for dataset auditing, structural difficulty analysis, and post-hoc trajectory evaluation. These metadata fields are not provided to agents during inference. Table 10 summarizes the metadata fields used for the two evaluated task types.

## A.5 RCA Metadata Distributions

Table 11 summarizes the question-level component distributions for the 126 RCA tasks. These metadata fields characterize observability, answer multiplicity, causal depth, evidence-path length, network heterogeneity, and protocol complexity.

The O2 subset contains tasks in which decisive evidence must be inferred from indirect observations. The multi-root-cause subset tests answer completeness. Fault-propagation-chain length captures causal depth, while golden solution length captures the number of expert-required evidence steps. Device heterogeneity and protocol complexity capture the operational diversity and cross-layer reasoning burden of the task.

<table><tr><td colspan="2">Restored Path Count</td></tr><tr><td>Tier N</td><td>Ratio</td></tr><tr><td>P1 one path</td><td>64 59.26%</td></tr><tr><td>P2 two paths</td><td>2018.52%</td></tr><tr><td>P3+ three or more</td><td>2422.22%</td></tr><tr><td colspan="2">Device Heterogeneity</td></tr><tr><td>Tier N</td><td>Ratio</td></tr><tr><td>H-Low (≤ 6)</td><td>58 53.70%</td></tr><tr><td>H-High (&gt; 6)</td><td>5046.30%</td></tr></table>

<table><tr><td colspan="2">Golden Solution Path</td></tr><tr><td>Tier</td><td>N Ratio</td></tr><tr><td>Short/Mod. (≤ 33)</td><td>7468.52%</td></tr><tr><td>Long (&gt; 33)</td><td>3431.48%</td></tr></table>

<table><tr><td colspan="2">Protocol Complexity</td></tr><tr><td>Tier</td><td>N Ratio</td></tr><tr><td>Low/Mod. (≤ 4)</td><td>4844.44%</td></tr><tr><td>High (&gt; 4)</td><td>6055.56%</td></tr></table>

Table 12: Path-restoration question-level component distributions over 108 tasks.

## A.6 Path-Restoration Metadata Distributions

The path-restoration subset contains 108 tasks. Table 12 summarizes the main path-restoration metadata distributions. Component counts are computed from normalized gold paths and metadata in the released task files.

Most path-restoration tasks have one gold path, but 44 tasks require multiple paths. These multi-path cases stress branching, redundancy, load balancing, and service-specific forwarding behavior. Pathrestoration tasks also have longer evidence paths than RCA tasks because the agent must reconstruct complete forwarding behavior rather than identify one localized root cause.

## A.7 Task Examples

We provide one RCA example and one path-restoration example to illustrate how task descriptions, gold answers, metadata, and golden evidence steps are connected. Agents receive only the task description and permitted tools during evaluation; they do not receive gold answers, metadata labels, or golden evidence steps.

RCA example. RCA-90 asks for the minimal set of root causes explaining why ping from Site1 area, Access-PC-01 to 20.1.1.10 is unreachable. The gold answer contains two port-fault entries: BoardLeaf-01;GE1/0/0;shutdown and BoardLeaf-01;GE1/0/1;shutdown. Both entries belong to C1 interface-state faults. The task is an RC2 task with two independent gold root causes. It is O1 because decisive interface-state evidence is directly observable on BoardLeaf-01. The golden solution path has three key steps, including interface description, current configuration, and LLDP neighbor checks. The task is H-Low because the normalized heterogeneity count is one device-role family.

Path-restoration example. PR-100 asks the agent to restore the path from SZ Server Cluster3 in the Shenzhen data center to SH SAL PC01 at 10.2.10.1 in the Shanghai branch. The gold answer contains one 12-element path from SZ Server Cluster3 through the Shenzhen core, PE devices, Beijing gateway, Shanghai aggregation, and finally SH SAL PC01. The task is a P1 task because the normalized gold answer contains one path. Its golden solution path has 33 expert key steps, placing it at the upper boundary of the Short/Moderate tier. Representative evidence includes route-table checks and LLDP-neighbor checks. The task is H-Low under the path-restoration threshold because its normalized heterogeneity count is six, and its protocol-complexity mechanism count is four.

## B Supplementary Evaluation Metrics

The main paper reports strict IoU-based RCA localization and identification metrics. This section defines supplementary metrics used only for diagnostic error analysis. These metrics do not replace the primary scores.

<table><tr><td>Match level</td><td>Similarity</td><td>Interpretation</td></tr><tr><td>Exact label</td><td>1.0</td><td>The predicted normalized root-cause label is identical to the gold label.</td></tr><tr><td>Same subtype</td><td>0.8</td><td>The prediction falls under the same fine-grained failure sub- type but misses the exact normalized label.</td></tr><tr><td>Same technical do- 0.6 main</td><td></td><td>The prediction belongs to the same operational domain, such as security/NAT access control, routing/policy control, or high-availability mechanisms.</td></tr><tr><td>Same category</td><td>super- 0.4</td><td>The prediction remains within the same broad fault fam- ily, such as connectivity, policy, service, redundancy, or operational-visibility faults.</td></tr><tr><td>Unrelated</td><td>0.0</td><td>The prediction belongs to a different fault family or cannot be mapped to the root-cause taxonomy.</td></tr></table>

Table 13: Root-cause hierarchy similarity used by hierarchical RCA scoring.

<table><tr><td>Node relation</td><td>Similarity compo- Interpretation nent</td><td></td></tr><tr><td>Exact node</td><td>1.0</td><td>The normalized predicted node is identical to the gold node.</td></tr><tr><td>Same logical cluster</td><td>0.7 taxonomy score</td><td>The nodes are in the same redundancy pair, HA pair, or inferred logical device cluster.</td></tr><tr><td>Same site and role</td><td>0.5 taxonomy score</td><td>The nodes share both site and device-role metadata.</td></tr><tr><td>Same site or same role</td><td>0.3 taxonomy score</td><td>The nodes share only one of site or role.</td></tr><tr><td>Topology distance 1</td><td>0.6 topology score</td><td>The nodes are one hop apart in the LLDP-derived topol- ogy graph.</td></tr><tr><td>Topology distance 2</td><td>0.3 topology score</td><td>The nodes are two hops apart in the LLDP-derived topol- ogy graph.</td></tr><tr><td>Disconnected or farther</td><td>0.0 topology score</td><td>The nodes are disconnected, unknown, or at distance at least three.</td></tr></table>

Table 14: Topology-aware node similarity used by hierarchical node location.

## B.1 Hierarchical RCA Scoring

Strict RCA identification treats all incorrect root-cause labels equally. This binary treatment hides wheteher an incorrect label is outside the relevant mechanism family or only differs at a finer taxonomy. We therefore report a supplementary hierarchical RCA score based on the expert-defined RCA taxonomy in Table 13. The score gives partial credit when the predicted root-cause label matches the gold label at coarser levels of the taxonomy, such as the same subtype, technical domain, or broad fault family. This follows the common use of semantic hierarchies in classification analysis, for example in ImageNet-style label taxonomies [3, 11].

We use this score only for diagnostic analysis of failed trajectories. It separate predictions that are unrelated to the gold root cause from predictions that identify a related mechanism but choose the wrong normalized label. The primary benchmark score remains the strict exact-match metric; hierarchical RCA is reported as an auxiliary measure for interpreting near misses.

Let sim(c, cˆ) denote the similarity between a gold root-cause label c and a predicted label cˆ. We assign similarity 1.0 for an exact label match, 0.8 for the same fine-grained subtype, 0.6 for the same technical domain, 0.4 for the same super-category, and 0 otherwise. For multi-root-cause tasks, we define $G _ { q }$ as the set of gold RCA entries and $P _ { q }$ as the set of predicted RCA entries, compute all pairwise similarities between predicted and gold RCA entries, perform maximum-weight bipartite matching, and normalize the matched similarity by the number of gold RCA entries:

$$
\mathrm { H i e r - R C A } ( q ) = \frac { \operatorname* { m a x } _ { M \in \mathcal { M } ( P _ { q } , G _ { q } ) } \sum _ { ( p , g ) \in M } \sin ( c _ { g } , c _ { p } ) } { \| G _ { q } \| } .\tag{12}
$$

## B.2 Topology-Aware Node Similarity

We also define a topology-aware node similarity score as a supplement to exact node-object localization. Exact node matches receive full credit. Non-identical nodes receive partial credit when they belong to the same logical cluster, same site and role, or nearby topology neighborhood (see Table 14). This score evaluates node proximity only; it does not include object-level or root-cause-label similarity.

For non-identical nodes, the final hierarchical node similarity combines the taxonomy and topology components:

$$
\mathrm { n o d e . s i m } = 0 . 7 \cdot \mathrm { t a x o n o m y . s i m } + 0 . 3 \cdot \mathrm { t o p o l o g y . s i m } .\tag{13}
$$

If either node is absent from the LLDP-derived topology, the score falls back to the taxonomy component inferred from normalized node names, site, role, and redundancy metadata.

## B.3 Root-Cause Type Hierarchy

Table 15 lists the normalized root-cause labels defined by the RCA output schema and their hierarchical grouping. The L0 column contains the complete set of schema-level root-cause labels that a model may output under the task instructions, including labels that do not appear as gold answers in the current 126-task release. L1, L2, and L3 provide progressively coarser groupings for near-miss analysis. Legacy gold-answer aliases are normalized to these schema-level L0 labels before hierarchical matching.

## C Supplementary Evaluation Results

This section provides detailed result tables omitted from the main paper.

## C.1 Full RCA Category Results

Table 16 reports localization, identification, and evidence scores for each model across the five merged RCA categories. These results complement the RCA category radar plot in the main paper.

## C.2 Hierarchical RCA Results

Table 17 compares strict exact scores with supplementary hierarchical scores. Across the five evaluated agents, hierarchical RCA identification exceeds exact RCA identification by 2.65–13.14 percentage points, and hierarchical node localization exceeds exact localization by 16.28–31.24 points. These gaps measure the additional credit captured by the relaxed hierarchy when a prediction preserves a coarser root-cause category or topology neighborhood even though the strict normalized label or faulty object is incorrect.

The gap between exact localization and hierarchical node localization suggests that agents often reach the correct topology neighborhood but fail to identify the exact faulty node or object required for strict operational correctness.

## C.3 Observability Results for All Agents

Table 18 reports RCA performance by observability level for all evaluated agents. The O2 subset contains tasks in which decisive evidence is unavailable or must be inferred from indirect observations. The O2→O1 rows correspond to an ablation setting where the originally unobservable elements are made directly observable. Since this ablation changes the golden evidence specification, we use it primarily to analyze final accuracy, localization, and identification rather than to draw conclusions from the evidence score.

## C.4 Heterogeneity Results for All Agents

Table 19 reports performance by network heterogeneity level. Heterogeneity captures the need to reason across different device roles, vendors, command syntaxes, and output formats. The stratified results expose differences across heterogeneous settings that are hidden in aggregate performance.

<table><tr><td colspan="2">L3 super-category</td><td>L2 technical domain</td><td>L1 root-cause subtype</td><td>L0 normalized root-cause label</td></tr><tr><td>State</td><td></td><td>Physical / Link Interface state</td><td>Administrative shutdown shutdown</td><td></td></tr><tr><td>State</td><td></td><td></td><td>Physical / Link Interface performance Bandwidth congestion</td><td>traffic congestion occupying port bandwidth</td></tr><tr><td>State</td><td></td><td>Physical / Link Interface MTU</td><td>MTU mismatch</td><td>MTU value misconfiguration</td></tr><tr><td>Control</td><td></td><td>Security / Boundary Security policy</td><td>Missing permit rule</td><td>security policy rule does not per- mit the corresponding user</td></tr><tr><td>Control</td><td></td><td>Security / Boundary NAT role binding</td><td>tribute error</td><td>NAT external-interface at- NAT external interface attribute misconfiguration or missing con- figuration</td></tr><tr><td>Control</td><td></td><td>Security / Boundary NAT role binding</td><td>NAT internal-interface at- NAT internal interface attribute tribute error</td><td>misconfiguration or missing con-</td></tr><tr><td>Control</td><td></td><td>Routing / Policy Static route</td><td>Blackhole route</td><td>figuration blackhole route</td></tr><tr><td>Control</td><td></td><td>Routing / Policy Static route</td><td>Missing static route</td><td>missing static route</td></tr><tr><td>Control</td><td></td><td>Routing / Policy Static route</td><td>Incorrect static route</td><td>incorrect static route</td></tr><tr><td>Control</td><td></td><td>Routing / Policy ARP / adjacency</td><td>ARP configuration</td><td>ARP configuration error</td></tr><tr><td>Control</td><td></td><td>Routing / Policy Forwarding loop</td><td>Layer-3 loop</td><td>Layer 3 loop</td></tr><tr><td>Control</td><td></td><td>Routing / Policy EGP control plane</td><td>BGP configuration</td><td>BGP configuration error</td></tr><tr><td>Control</td><td>Routing / Policy Route policy</td><td></td><td>ing</td><td>Prefix-list coverage miss- IP prefix list missing the corre-</td></tr><tr><td>Routing Control</td><td></td><td> / Policy IGP control plane</td><td>OŠPF configuration</td><td>sponding user source IP address OSPF configuration error</td></tr><tr><td>Control</td><td></td><td>Routing / Policy IGP control plane</td><td>ISIS configuration</td><td>ISIS configuration error</td></tr><tr><td colspan="2">HA / Reliability</td><td>Layer-2 loop preven- STP not enabled tion</td><td></td><td>global STP not enabled</td></tr><tr><td colspan="2">HA / Reliability</td><td>Layer-2 loop preven- Port STP not enabled</td><td></td><td>port STP not enabled</td></tr><tr><td colspan="2">HA / Reliability</td><td>tion Hot-standby</td><td>redun- VRRP/HRP redundancy global VRRP hot standby redun-</td><td></td></tr><tr><td colspan="2"></td><td>dancy</td><td>not enabled</td><td>dancy protocol not enabled</td></tr><tr><td colspan="2">Service / Overlay</td><td></td><td>VPN service isolation L3VPN configuration</td><td>L3VPÑ configuration error</td></tr><tr><td colspan="2">Service / Overlay</td><td>VXLAN overlay</td><td>VPN service isolation L2VPN configuration VXLAN configuration</td><td>L2VPN configuration error</td></tr><tr><td colspan="2">Service / Overlay Service / Overlay</td><td>SRv6 policy tunnel</td><td>SRv6-Policy planning</td><td>VXLAN configuration error SRV6-Policy tunnel planning er-</td></tr><tr><td colspan="2"></td><td></td><td>VPN service provi- Missing VPN configura- VPN configuration missing</td><td>ror</td></tr><tr><td colspan="2">Service / Overlay</td><td>sioning</td><td>tion</td><td></td></tr><tr><td colspan="2">Access Configura- Layer-2 tion</td><td>identity</td><td>tion</td><td>forwarding MAC address configura- MAC address configuration er- ror</td></tr><tr><td colspan="2">Access Configura- Interface addressing tion</td><td></td><td>Interface IP configuration interface IP error</td><td></td></tr><tr><td colspan="2">tion</td><td></td><td>ration</td><td>Access Configura- VLAN access binding Interface VLAN configu- interface VLAN configuration error</td></tr><tr><td colspan="2">Access Configura- Address uniqueness Loopback IP conflict tion</td><td></td><td></td><td>loopback IP configuration con- flict</td></tr><tr><td colspan="2"></td><td>lection</td><td>collection</td><td>Operations Support Host information col- Missing host-information host information collection func- tion missing</td></tr></table>

Table 15: Root-cause type hierarchy used for hierarchical RCA scoring over the complete RCA output-schema label space.

## C.5 Structural Difficulty Trends

We use task metadata to examine how performance changes across structurally easier and harder subsets. Because evidence observability and network heterogeneity are analyzed separately, Table 20 focuses on root-cause count, restored path count, fault-propagation-chain length, golden-solution length, and protocol complexity.

<table><tr><td>Agent</td><td>Category</td><td>RCA-Acc</td><td>RCA-Loc</td><td>RCA-ID</td><td>RCA-Evid.</td></tr><tr><td>Codex+GPT-5.5</td><td>C1</td><td>7.14</td><td>55.74</td><td>78.57</td><td>10.76</td></tr><tr><td>Codex+GPT-5.5</td><td>C2</td><td>65.31</td><td>80.33</td><td>80.65</td><td>13.45</td></tr><tr><td>Codex+GPT-5.5</td><td>C3</td><td>40.00</td><td>23.33</td><td>61.54</td><td>9.94</td></tr><tr><td>Codex+GPT-5.5</td><td>C4</td><td>50.00</td><td>51.43</td><td>51.43</td><td>18.50</td></tr><tr><td>Codex+GPT-5.5</td><td>C5</td><td>28.12</td><td>46.51</td><td>57.50</td><td>9.43</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>C1</td><td>3.57</td><td>40.30</td><td>60.61</td><td>9.77</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>C2</td><td>28.57</td><td>60.56</td><td>67.69</td><td>14.05</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>C3</td><td>0.00</td><td>0.00</td><td>10.00</td><td>4.49</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>C4</td><td>14.71</td><td>21.62</td><td>25.00</td><td>12.20</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>C5</td><td>18.75</td><td>36.96</td><td>37.78</td><td>9.32</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>C1</td><td>7.14</td><td>39.39</td><td>57.58</td><td>9.06</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>C2</td><td>26.53</td><td>55.56</td><td>61.02</td><td>7.78</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>C3</td><td>0.00</td><td>0.00</td><td>31.94</td><td>6.48</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>C4</td><td>11.76</td><td>13.89</td><td>13.89</td><td>11.23</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>C5</td><td>15.62</td><td>32.50</td><td>35.90</td><td>7.04</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>C1</td><td>10.71</td><td>57.81</td><td>65.52</td><td>8.17</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>C2</td><td>34.69</td><td>72.73</td><td>80.33</td><td>12.64</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>C3</td><td>0.00</td><td>0.00</td><td>44.44</td><td>8.57</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>C4</td><td>23.53</td><td>21.05</td><td>35.29</td><td>12.70</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>C5</td><td>21.88</td><td>26.83</td><td>25.00</td><td>5.69</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>C1 C2</td><td>0.00</td><td>12.90</td><td>26.67</td><td>3.85</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>C3</td><td>12.24 0.00</td><td>35.00 0.00</td><td>34.43 0.00</td><td>6.43</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>C4</td><td>0.00</td><td>0.00</td><td>0.00</td><td>2.03 4.55</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>C5</td><td>0.00</td><td>0.00</td><td>0.00</td><td>3.50</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 16: RCA results by fault category. Scores are percentages. C5 merges low-frequency service, overlay, configuration, management, and other operational categories.

<table><tr><td>Agent</td><td>Exact RCA-ID</td><td>Hier. RCA-ID</td><td>Exact Loc.</td><td>Hier. Node Loc.</td></tr><tr><td>Codex+GPT-5.5</td><td>66.50</td><td>72.52</td><td>52.69</td><td>73.80</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>39.59</td><td>50.19</td><td>35.03</td><td>58.70</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>39.75</td><td>50.19</td><td>29.14</td><td>58.67</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>52.36</td><td>62.99</td><td>38.03</td><td>64.87</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>11.28</td><td>13.93</td><td>12.33</td><td>28.61</td></tr></table>

Table 17: Exact and hierarchical RCA scores. Scores are percentages. Hierarchical scores are recall-oriented over gold RCA entries and are supplementary diagnostics rather than replacements for exact IoU metrics.

Table 20 reports the average easiest-to-hardest Accuracy drop for each component. For RCA tasks, fault-propagation-chain length shows the largest drop: average Accuracy falls from 64.21% on FPC Low cases to 7.00% on FPC-High cases. Root-cause count is also a strong factor, with RC1-to-RC3+ Accuracy decreasing by 31.63 points. Golden-solution length and protocol complexity show smaller but consistent drops across all five agents.

For path-restoration tasks, the drops are more balanced. Multiple restored paths, longer expert paths, and higher protocol complexity each reduce average Accuracy by roughly 20–23 points from the easy to hard tier. These metadata fields therefore separate easier and harder subsets for both task families, but should not be read as isolated causal variables, since each stratum still mixes different fault families, topologies, and evidence conditions.

<table><tr><td>Agent</td><td>Obs.</td><td>N</td><td>Acc.</td><td>RCA-Loc</td><td>RCA-ID</td><td>RCA-Evid.</td></tr><tr><td>Codex+GPT-5.5</td><td>01</td><td>107</td><td>49.53</td><td>54.11</td><td>72.41</td><td>16.45</td></tr><tr><td>Codex+GPT-5.5</td><td>02</td><td>19</td><td>36.84</td><td>42.86</td><td>35.48</td><td>12.11</td></tr><tr><td>Codex+GPT-5.5</td><td>02→01</td><td>19</td><td>73.68</td><td>81.82</td><td>75.00</td><td>14.08</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>01</td><td>107</td><td>20.56</td><td>37.02</td><td>42.92</td><td>12.85</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>02</td><td>19</td><td>15.79</td><td>17.14</td><td>16.22</td><td>11.55</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>02→01</td><td>19</td><td>31.58</td><td>37.50</td><td>25.71</td><td>11.66</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>01</td><td>107</td><td>17.76</td><td>29.55</td><td>45.05</td><td>9.99</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>02</td><td>19</td><td>15.79</td><td>28.57</td><td>15.00</td><td>12.37</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>02→01</td><td>19</td><td>47.37</td><td>44.83</td><td>35.48</td><td>10.15</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>01</td><td>107</td><td>23.36</td><td>37.85</td><td>53.04</td><td>12.11</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>02</td><td>19</td><td>36.84</td><td>35.48</td><td>48.28</td><td>15.84</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>02→01</td><td>19</td><td>52.63</td><td>59.09</td><td>58.33</td><td>9.63</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>01</td><td>107</td><td>5.61</td><td>14.00</td><td>13.43</td><td>5.80</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>02</td><td>19</td><td>0.00</td><td>2.50</td><td>0.00</td><td>1.36</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>02→01</td><td>19</td><td>0.00</td><td>0.00</td><td>0.00</td><td>2.50</td></tr></table>

Table 18: RCA results by observability level and observability ablation. Scores are percentages. O2→O1 is evaluated on the same 19 originally O2 tasks after removing the unobservable elements.

<table><tr><td>Agent</td><td>Task</td><td>H</td><td>N</td><td>Acc.</td><td>Loc.</td><td>ID/Rest.</td><td>Evid.</td></tr><tr><td>Codex+GPT-5.5</td><td>RCA</td><td>H-Low (= 1)</td><td>39</td><td>66.67</td><td>52.73</td><td>57.69</td><td>12.47</td></tr><tr><td>Codex+GPT-5.5</td><td>RCA</td><td>H-High (&gt; 1)</td><td>87</td><td>39.08</td><td>52.94</td><td>69.93</td><td>17.20</td></tr><tr><td>Codex+GPT-5.5</td><td>Path</td><td>H-Low (≤ 6)</td><td>58</td><td>89.66</td><td>100.00</td><td>95.52</td><td>38.89</td></tr><tr><td>Codex+GPT-5.5</td><td>Path</td><td>H-High (&gt; 6)</td><td>50</td><td>86.00</td><td>98.02</td><td>95.11</td><td>53.29</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>RCA</td><td>H-Low (= 1)</td><td>39</td><td>33.33</td><td>30.38</td><td>34.25</td><td>11.76</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>RCA</td><td>H-High (&gt; 1)</td><td>87</td><td>13.79</td><td>36.24</td><td>40.91</td><td>12.96</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>Path</td><td>H-Low (≤ 6)</td><td>58</td><td>29.31</td><td>94.96</td><td>68.58</td><td>25.72</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>Path</td><td>H-High (&gt; 6)</td><td>50</td><td>4.00</td><td>72.73</td><td>35.05</td><td>27.14</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>RCA</td><td>H-Low (= 1)</td><td>39</td><td>28.21</td><td>28.57</td><td>26.09</td><td>8.12</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>RCA</td><td>H-High (&gt; 1)</td><td>87</td><td>12.64</td><td>29.69</td><td>45.66</td><td>11.05</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>Path</td><td>H-Low (≤ 6)</td><td>58</td><td>44.83</td><td>91.38</td><td>73.46</td><td>20.32</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>Path</td><td>H-High (&gt; 6)</td><td>50</td><td>6.00</td><td>72.41</td><td>50.77</td><td>22.91</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>RCA</td><td>H-Low (= 1)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>RCA</td><td>H-High (&gt; 1)</td><td>39</td><td>46.15 16.09</td><td>38.71 37.27</td><td>42.86</td><td>9.57</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>Path</td><td>H-Low (≤ 6)</td><td>87 58</td><td>48.28</td><td>91.38</td><td>55.84 72.38</td><td>13.97</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>Path</td><td>H-High (&gt; 6)</td><td>50</td><td>32.00</td><td>67.92</td><td>50.28</td><td>19.72 26.87</td></tr><tr><td></td><td>RCA</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>RCA</td><td>H-Low (= 1)</td><td>39</td><td>10.26</td><td>8.57</td><td>5.97</td><td>2.98</td></tr><tr><td>HermesAgent+TelecomGPT-R1 HermesAgent+TelecomGPT-R1</td><td>Path</td><td>H-High (&gt; 1)</td><td>87</td><td>2.30 3.45</td><td>13.64 83.19</td><td>13.30 17.97</td><td>6.04</td></tr><tr><td></td><td>Path</td><td>H-Low (≤ 6)</td><td>58</td><td></td><td></td><td></td><td>8.96</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td></td><td>H-High (&gt; 6)</td><td>50</td><td>0.00</td><td>58.68</td><td>5.24</td><td>4.18</td></tr></table>

Table 19: Selected results by network heterogeneity. Scores are percentages.

## C.6 Evidence-Grounded Performance Analysis

In this subsection, evidence coverage is defined as the fraction of expert golden evidence steps observed in the trajectory, $| \hat { \mathcal { A } } \cap \mathcal { A } ^ { * } | / | \mathcal { A } ^ { * } |$ . This recall-style quantity measures how much of the expert diagnostic chain the agent reaches, without penalizing additional commands. It is therefore different from the main evidence score used in the capability tables, which is an Evidence F1 score that accounts for both covered golden steps and extra executed device-command observations.

Table 21 reports evidence coverage on exactly solved tasks. Even when agents produce the correct final answer, their evidence coverage remains limited, suggesting that current agents often solve tasks without reconstructing the complete expert diagnostic chain. The table is conditioned on solved tasks, whereas Figure 5 analyzes all model-question samples.

<table><tr><td>Component Task Easy Acc. Hard Acc. Drop Agents</td></tr><tr><td>Fault Propagation Length RCA 64.21 7.00</td></tr><tr><td>57.21 5/5 Root-cause Count RCA 37.88 6.25 31.63 5/5</td></tr><tr><td>Golden Solution Length RCA 27.67 16.60 11.07 5/5</td></tr><tr><td>Protocol Complexity RCA 26.84 17.20 9.64 5/5</td></tr><tr><td>Restored Path Count Path 41.25 20.00 21.25 5/5</td></tr><tr><td>Golden Solution Length Path 42.16 19.41 22.75 5/5</td></tr><tr><td>Protocol Complexity Path 46.25 26.00 20.25 5/5</td></tr></table>

Table 20: Accuracy impact of CTBench components excluding evidence observability and network heterogeneity. Easy/Hard Accuracy is averaged over agents using the easiest and hardest tier of each component; Drop is the mean per-agent easy-to-hard Accuracy decrease in percentage points.

<table><tr><td>Agent</td><td>RCA</td><td>Path Restoration</td></tr><tr><td>Codex+GPT-5.5</td><td>58.05%</td><td>53.45%</td></tr><tr><td>ClaudeCode+Qwen3.7-Plus</td><td>51.45%</td><td>38.56%</td></tr><tr><td>HermesAgent+DeepSeek-V4-Pro</td><td>38.02%</td><td>19.61%</td></tr><tr><td>HermesAgent+Qwen3.7-Max</td><td>40.22%</td><td>27.75%</td></tr><tr><td>HermesAgent+TelecomGPT-R1</td><td>16.67%</td><td>39.13%</td></tr></table>

Table 21: Evidence coverage among exactly solved tasks.

![](images/131e4dc8754d136f4230b840674cd2782cf753c4e0042559427e5203db1b717a.jpg)  
Figure 5: Evidence–accuracy relationship using equal-frequency quintiles sorted by evidence score. Each point aggregates the same number of model-question samples within a task family; the reported $R ^ { 2 }$ values are computed over the quintile points. The lines connect quintile aggregates and are not fitted regression lines.

Figure 5 further relates evidence coverage to exact-answer Accuracy after collecting all trajectories. Path Restoration shows a strong monotonic trend: higher evidence coverage corresponds to substantially higher exact Accuracy. RCA exhibits a weaker relationship; the lower RCA $\mathrm { \ddot { \it R } ^ { 2 } }$ indicates that evidence coverage explains only part of the variance in exact correctness. This weaker relationship is consistent with the RCA error analysis: even when an agent reaches high evidence coverage, it may still fail to identify the correct fault node, affected object, root-cause label, or complete multi-fault answer. Thus, evidence acquisition is necessary for reliable troubleshooting, but RCA also requires stronger causal reasoning and answer normalization.

## C.7 Harness Sensitivity Analysis

To separate model choice from execution-harness effects, we compare raw trajectory sets that use the same DeepSeek-V4-Pro model but different agent harnesses: ClaudeCode, HermesAgent, and Codex. This analysis treats each harness-model pair as an executable agent because the harness controls tool scheduling, state serialization, stopping behavior, and final-answer extraction. All runs

use the same task set, gold answers, and scoring scripts. The comparison is therefore intended as a raw harness sensitivity analysis under a fixed model family, rather than as an ablation that isolates each individual harness mechanism. Missing or empty final answers are kept in the denominator and counted as incorrect.
<table><tr><td>Task</td><td>Harness + Model</td><td>Acc.</td><td>Loc.</td><td>ID/Rest.</td><td>Evid.</td><td>Runtime</td><td>Tokens</td></tr><tr><td>RCA</td><td>ClaudeCode+DeepSeek-V4-Pro</td><td>21.43</td><td>26.19</td><td>34.85</td><td>37.71</td><td>410.52s</td><td>3809.7k</td></tr><tr><td>RCA</td><td>HermesAgent+DeepSeek-V4-Pro</td><td>17.46</td><td>29.43</td><td>40.08</td><td>33.38</td><td>481.13s</td><td>1750.8k</td></tr><tr><td>RCA</td><td>Codex+DeepSeek-V4-Pro</td><td>15.08</td><td>21.45</td><td>34.26</td><td>30.88</td><td>482.79s</td><td>1083.7k</td></tr><tr><td>Path Restoration</td><td>ClaudeCode+DeepSeek-V4-Pro</td><td>14.81</td><td>77.59</td><td>58.51</td><td>26.35</td><td>416.66s</td><td>5440.1k</td></tr><tr><td>Path Restoration</td><td>HermesAgent+DeepSeek-V4-Pro</td><td>26.85</td><td>81.90</td><td>59.26</td><td>18.84</td><td>492.97s</td><td>2155.5k</td></tr><tr><td>Path Restoration</td><td>Codex+DeepSeek-V4-Pro</td><td>26.85</td><td>77.59</td><td>59.25</td><td>24.88</td><td>545.00s</td><td>1660.8k</td></tr></table>

Table 22: Harness sensitivity for the same DeepSeek-V4-Pro model. Scores are percentages except runtime and tokens. ID/Rest. denotes RCA identification IoU for RCA and restoration-edge IoU for Path Restoration.

Table 22 shows that CTBench measures a harness-model combination rather than a model in isolation. On RCA, ClaudeCode+DeepSeek-V4-Pro obtains the highest exact Accuracy and evidence coverage among the three harnesses, while HermesAgent+DeepSeek-V4-Pro obtains the highest localization and identification IoU. Codex+DeepSeek-V4-Pro uses the fewest tokens and rounds, but its RCA exact Accuracy and evidence coverage are lower. On Path Restoration, HermesAgent and Codex reach the same exact Accuracy, whereas HermesAgent has the best localization IoU and Codex achieves nearly the same restoration-edge IoU with substantially fewer tokens and rounds than HermesAgent. ClaudeCode collects the most evidence for Path Restoration but uses far more tokens and obtains lower exact Accuracy. Runtime, rounds, and tokens should be interpreted as complementary efficiency measures: fewer interaction rounds or fewer tokens do not necessarily imply lower wall-clock latency because model-side response time and harness orchestration overhead can differ.

The disagreement counts in Table 23 show that different harnesses solve partly different subsets of tasks even when the underlying model is fixed. For example, on RCA, ClaudeCode solves 11 tasks that Codex misses, while Codex solves 3 tasks that ClaudeCode misses. On Path Restoration, Codex solves 19 tasks missed by ClaudeCode, while HermesAgent and Codex each solve 18 tasks missed by the other. Overall, these results show that the harness matters even when the underlying model is fixed. Different harnesses solve different subsets of tasks, so we report performance at the harness–model level rather than treating the model as the only source of variation.

## D Reproducibility Details

## D.1 Evaluation Environment

All CTBench agent trajectories are executed on a single orchestration host running EulerOS 2.0. The host is equipped with an Intel Xeon Gold 6230N CPU at 2.30GHz, 40 physical cores, 80 logical processors, and 502GB of physical memory. The host runs the agent harnesses, local sandbox workspaces, simulator/tool servers, logging, and result collection. Model inference is not served locally; all evaluated models are accessed through their original vendor APIs.

We pre-collected command-line interface (CLI) logs from the target network devices and served them through a mock backend. For supported commands, the backend returns the recorded device outputs; for invalid queries, it returns the corresponding error messages. Commands that are syntactically valid but absent from the collected logs return a simulated “permission denied” response, so agents cannot query outside the predefined observation set.

We evaluate five harness–LLM configurations with fixed harness versions: Codex (v0.141.0) with GPT-5.5, ClaudeCode (v2.1.63) with Qwen3.7-Plus, and Hermes Agent (v0.16.0) with DeepSeek-V4-Pro, Qwen3.7-Max, or TelecomGPT-R1. Each benchmark task runs in a fresh Docker container. The workspace, tool configuration, logs, and temporary files are recreated for each task to avoid state carryover between runs.

<table><tr><td>Task</td><td>Pair</td><td>Both</td><td>First only</td><td>Second only</td><td>Neither</td></tr><tr><td>RCA</td><td>CC vs. HA</td><td>15</td><td>12</td><td>7</td><td>92</td></tr><tr><td>RCA</td><td>CC vs. CX</td><td>16</td><td>11</td><td>3</td><td>96</td></tr><tr><td>RCA</td><td>HA vs. CX</td><td>12</td><td>10</td><td>7</td><td>97</td></tr><tr><td>Path Rest.</td><td>CC vs. HA</td><td>6</td><td>10</td><td>23</td><td>69</td></tr><tr><td>Path Rest.</td><td>CC vs. CX</td><td>10</td><td>6</td><td>19</td><td>73</td></tr><tr><td>Path Rest.</td><td>HA vs. CX</td><td>11</td><td>18</td><td>18</td><td>61</td></tr></table>

Table 23: Pairwise exact-answer disagreement under the same DeepSeek-V4-Pro model. CC, HA, and CX denote ClaudeCode, HermesAgent, and Codex, respectively. “First only” and “Second only” refer to the order in the Pair column.

## D.2 Agent Harnesses and Concurrency

We evaluate three harness implementations: Codex, ClaudeCode, and HermesAgent. The reported agent-model settings are instantiated from these harnesses:

• Codex+GPT-5.5,

• ClaudeCode+Qwen3.7-Plus,

• HermesAgent+DeepSeek-V4-Pro,

• HermesAgent+Qwen3.7-Max, and

• HermesAgent+TelecomGPT-R1.

During evaluation, the task queue is capped at three concurrent agent runs to control API request pressure and avoid local simulator contention and the maximum running time for each initiated task is 3600 seconds.

## D.3 Command Form

Each harness run follows the same orchestration pattern: select the harness and model, provide the CTBench task JSON, write trajectories and parsed results to a run-specific output directory, and set the maximum concurrency.

```perl
python <runner>.py --harness codex --model gpt-5.5
--input <ctbench_task_json> --output <result_json>
--workspace-base <sandbox_root> --concurrency 3 --timeout 3600
```

```perl
python <runner>.py --harness claudecode --model qwen3.7-plus
--input <ctbench_task_json> --output <result_json>
--workspace-base <sandbox_root> --concurrency 3 --timeout 3600
```

python <runner>.py --harness hermesagent --model <model\_name>   
--input <ctbench\_task\_json> --output <result\_json>   
--workspace-base <sandbox\_root> --concurrency 3 --timeout 3600

For the Codex harness, the per-task child process created by the runner uses the following command shape, with the task prompt supplied on standard input:

```batch
codex.cmd -a never exec --cd <workspace_dir> --skip-git-repo-check
--sandbox workspace-write --color never --output-last-message last_msg.txt
--model gpt-5.5 --ephemeral
```

## D.4 Prompt Template

All agent-model combinations use the same task-level prompt template shown in Figure 6. The prompt contains the task description, question identifier, allowed tool interface, and required answer format. It does not include ground-truth answers, task metadata, root-cause categories, difficulty labels, or golden evidence steps.

![](images/ba71613c2b42c773b3a655446c94a9193a9d770fc029ebf7e25fa98f0e095e6e.jpg)  
Figure 6: Agent Prompt Template.

## E Trajectory-Level RCA Failure Analysis

## E.1 Overview of Trajectory-Level Failure Analysis

This appendix summarizes a trajectory-level post-hoc analysis of exact-match RCA failures for Codex+GPT-5.5 and HermesAgent+Qwen3.7-Max. The goal is to characterize why an agent’s final answer fails after interacting with the diagnostic environment, rather than only reporting whether the final answer matches the gold RCA tuple. Among the 126 RCA tasks, Codex+GPT-5.5 has 66 exact-match failures and HermesAgent+Qwen3.7-Max has 94 exact-match failures; 60 tasks are failed by both systems.

The main failure pattern is not failed tool use. In many cases, agents issue relevant commands and observe useful intermediate evidence, but still fail when mapping that evidence to the final RCA tuple set. The observed errors include missing independent causes, adding non-minimal causes, choosing the wrong fault mechanism, grounding the fault to the wrong node or object, and violating the required output schema.

<table><tr><td>Level</td><td>Leaf failure label</td><td>Unit</td><td>Definition</td></tr><tr><td>Output-level</td><td>F5. Answer Realization Whole answer Failure</td><td></td><td>The trajectory does not produce a valid final RCA answer, or the answer cannot be parsed into the required tuple schema.</td></tr><tr><td>Set-level</td><td>F1. Causal Coverage Deficit Answer set</td><td></td><td>The answer omits one or more indepen- dent gold root causes, including dual-root, multi-device, or multi-interface faults.</td></tr><tr><td>Set-level</td><td>F2. Causal Minimality Vio- Answer set lation</td><td></td><td>The answer includes redundant, symptom- level, or non-independent causes beyond the minimal gold root-cause set.</td></tr><tr><td>Entry-level</td><td>F4. Mechanism Discrimina- Root-cause entry tion Error</td><td></td><td>The returned entry uses the wrong causal mechanism, such as confusing redundancy, routing, policy, address, or forwarding-</td></tr><tr><td>Entry-level</td><td>F3. Fault-Object Grounding Root-cause entry Error</td><td></td><td>layer faults. The mechanism is largely correct, but the causal object is grounded to the wrong de- vice, interface, IP address, policy object, underlay/overlay layer, or active/standby role.</td></tr></table>

Table 24: Hierarchical RCA failure taxonomy used for trajectory-level error analysis. Rows follow the annotation priority order.

We identify this pattern through manual inspection of the diagnostic traces, checking each failed task against the task statement, final answer, issued commands, and observed command outputs.

## E.2 Failure Mode Taxonomy

We organize trajectory failures using the hierarchical RCA failure taxonomy in Table 24. The taxonomy contains one output-level failure, two answer-set-level failures, and two entry-level failures. This taxonomy is used only as a post-hoc explanation of exact-match errors; it is not a formal component of the benchmark, is not part of the primary benchmark score, and does not relax the exact-answer criterion. This structure is important for reproducible annotation: when a prediction both misses an independent root cause and mislocalizes a returned tuple, the set-level coverage error is assigned first; entry-level mechanism or grounding errors are used only when the answer set is otherwise comparable to the gold set.

For each task, three expert annotators with telecom RCA expertise independently assigned failure labels, and the taxonomy was applied with a fixed priority order. Disagreements were resolved through adjudication and discussion under the same priority rules. The annotators first checked whether the final answer was parseable under the required RCA schema. If not, the failure was labeled as answer realization. For parseable answers, they next compared the predicted and gold root-cause sets to identify missing independent causes or non-minimal extra causes. Only after the answer set was comparable did they label entry-level failures: mechanism discrimination errors when the causal mechanism was wrong, and fault-object grounding errors when the mechanism was largely correct but the device, interface, address, policy object, or topology role was incorrect. Within entry-level annotation, mechanism mismatch has priority over object mismatch: a prediction with both a wrong mechanism and a wrong object is labeled F4 rather than F3.

Table 25 reports the resulting distribution. Coverage deficits and mechanism discrimination dominate Codex+GPT-5.5’s RCA errors, while HermesAgent+Qwen3.7-Max shows a more distributed error profile with substantially more minimality and answer-realization failures.

## E.3 Representative Failure Trajectories

The full trajectories for these examples are provided in the Code and Data Supplement under model trajectory, organized by model name, task type, and task identifier. All representative examples below were checked against the recorded final answers and diagnostic traces of the corresponding agents.

<table><tr><td>Model</td><td>F1 F2</td><td>F3</td><td></td><td>F4 F5</td><td>Total</td></tr><tr><td>GPT-5.5</td><td>28</td><td>1 13</td><td>23</td><td>1</td><td>66</td></tr><tr><td>Qwen3.7-Max</td><td>32</td><td>16</td><td>18 16</td><td>12</td><td>94</td></tr></table>

Table 25: Failure-mode counts for exact-match RCA failures. F1–F5 refer to the labels in Table 24; each failed trajectory is assigned one primary failure label.

F5 Answer realization failure: RCA q12. The gold answer is FW 02;114.114.114.114;global VRRP hot standby redundancy protocol not enabled. HermesAgent+Qwen3.7-Max returns an empty final answer despite successfully interacting with the environment. The trajectory lacks valid result tags, separators, or parseable lines, leading to a complete output failure independent of the agent’s actual domain knowledge.

F1 Causal coverage deficit: RCA q7. The gold answer contains two independent root causes: FW 01;10.3.10.1;security policy rule does not permit the corresponding user and BJHQ CSR1000V GW 01;10.1.120.251;IP prefix list missing the corresponding user source IP address. Codex+GPT-5.5 outputs only the firewall security-policy root cause and omits the prefix-list fault, prematurely halting its diagnosis once the first locally plausible cause is identified.

F2 Causal minimality violation: RCA q18. The gold answer contains a VRRP redundancy configuration fault on FW 02, while HermesAgent+Qwen3.7-Max returns both a prefix-list fault and a VRRP fault on a different firewall. Although this answer also contains localization errors, the set-level F2 label takes precedence because the final submitted set retains unpruned, non-gold causal assertions alongside downstream symptoms.

F4 Mechanism discrimination error: RCA q10. The gold root cause is FW 02;114.114.114.114;global VRRP hot standby redundancy protocol not enabled. Codex+GPT-5.5 localizes the answer to the correct firewall and address but incorrectly reports an OSPF configuration error, successfully grounding the location while confusing redundancy-state semantics with routing-protocol configurations.

F3 Fault-object grounding error: RCA q11. The gold root cause is FW 02;8.8.8.8;global VRRP hot standby redundancy protocol not enabled. HermesAgent+Qwen3.7-Max outputs Core SW 01;8.8.8.8;global VRRP hot standby redundancy protocol not enabled. The mechanism and address are aligned with the gold answer, but the causal device is misattributed to a neighboring core switch rather than the faulty firewall itself. The agent fails to decouple its observation point from the actual causal node.

## E.4 Implications for Agent Design

The failure taxonomy suggests several checks that are useful for troubleshooting agents. During a run, the agent should track which symptoms have been explained, which candidate causes remain possible, which observations support or reject each candidate, and whether the final answer satisfies the required schema.

The observed failure modes suggest several useful checks for agent design:

• Causal Coverage (F1) & Minimality (F2): A global diagnostic state tracker may help ensure all symptoms are explained without including redundant downstream effects.

• Semantic Grounding (F3 & F4): Domain-specific verification may reduce mechanism confusion and mislocalization by checking protocol constraints (e.g., routing vs. redundancy states) and maintaining role-aware topology mapping to separate observation nodes from causal origins.

<table><tr><td>Failure taxonomy</td><td>General agent capability</td><td>CTBench property</td></tr><tr><td>F1 Causal coverage deficit</td><td>Causal attribution and long-horizon root_cause_count; evidence planning</td><td>golden_solution_path</td></tr><tr><td>F2 Causal minimality violation</td><td>Causal attribution and symptom prun- fault_propagation_chain; ing</td><td>golden_solution_path</td></tr><tr><td>F3 Fault-object grounding error</td><td>Interaction with heterogeneous envi- network_heterogeneity ronments</td><td></td></tr><tr><td>F4 Mechanism discrimination error</td><td>Partial-observation reasoning and protocol_complexity; causal attribution</td><td>evidence_observability</td></tr><tr><td>F5 Answer realization failure</td><td>Reliable structured answer realization Output schema constraints</td><td></td></tr></table>

Table 26: Qualitative mapping from CTBench RCA failure modes to broader agent capabilities and related task properties.

• Robust Finalization (F5): Deterministic output validation can enforce schema constraints before final submission.

Overall, this fine-grained analysis suggests that, in these RCA failures, current agents more often struggle with multi-step diagnostic reasoning and causal minimization than with basic tool execution.

## F Discussion of Agent Capabilities in CTBench

Although CTBench is grounded in telecom network operations, the trajectory-level RCA failures reflect broader agent capabilities required in complex professional environments. The goal of this discussion is to clarify how the failure modes summarized in Table 24 relate to general agent capabilities and how these pressures arise from CTBench task characteristics. In many failed trajectories, agents can collect relevant observations but still fail to maintain causal coverage, prune downstream symptoms, ground evidence to the correct object, distinguish similar mechanisms, or realize the final answer in the required structured format.

These failure modes are consistent with the task properties of CTBench RCA: heterogeneous device roles, partial observability, layered protocol mechanisms, causal propagation, multiple independent faults, and multi-step evidence requirements. Table 26 summarizes how the observed failure modes relate to broader agent capabilities and the CTBench properties that stress them.
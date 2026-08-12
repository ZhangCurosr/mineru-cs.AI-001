# GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning

Xin Xiao<sup>1</sup>, Jiang Zhong<sup>1†</sup>, Junnan Zhu<sup>2</sup>

Yingchao Feng<sup>3</sup>, Peijin Wang<sup>3</sup>, Yidan Zhang<sup>3</sup>, Kaiwen Wei<sup>1†</sup>

<sup>1</sup>School of Computer Science, Chongqing University

<sup>2</sup>MAIS, Institute of Automation, Chinese Academy of Sciences

<sup>3</sup>Aerospace Information Research Institute, Chinese Academy of Sciences

## Abstract

Earth observation (EO) agents construct scientifically valid tool workflows and ground their conclusions in current geospatial evidence. This is challenging because EO workflows are constrained by sensing semantics, product dependencies, spatial and temporal compatibility, and parameter requirements. Existing agents often search a broad operation space for each query, while recent self-evolving systems do not fully organize heterogeneous EO trajectories into reusable knowledge across diferent decision levels. To solve this problem, we present GeoForge, a training-free, selfevolving framework that transforms completed trajectories into a structured nonparametric execution state. GeoForge constrains the operation space according to the sensing context, then retrieves a task-conditioned prior from three comple mentary memories. Workflow Graph Memory captures global operation order, Action-Level Experiences provide local corrections, and the Adapted Skill Standard Operating Procedure preserves procedural and data constraints. The retrieved prior guides tool execution, while current observations remain the basis of the final answer. After each task, a safety-gated distillation process converts grounded trajectories into reusable execution knowledge for future retrieval. This execution, distillation, and reuse loop improves planning without updating the backbone LLM. Experiments on multiple geospatial benchmarks demonstrate that GeoForge consistently improves both task accuracy and tool-use trajectory quality across diverse LLM backbones, while substantially reducing tool-planning and reasoning errors for most LLMs.

## Introduction

Large language model (LLM) agents can interpret user requests, plan complex intermediate actions, and interact with external tools to solve complex tasks (Yao et al. 2022; Wu et al. 2023; Shinn et al. 2024). This capability is particularly promising for Earth Observation (EO), where scientific analysis requires coordinated reasoning over multimodal observations, geospatial products, and specialized tools (Wang et al. 2024; Xiao et al. 2025). EO agents can automate diverse real-world analytical pipelines by selecting observations, generating products, and grounding conclusions with geospatial evidence, thereby accelerating scientific discovery (Van de Weghe et al. 2025; Wang et al. 2026b).

Reliable EO analysis is governed by sensing semantics, observation conditions, and workflow dependencies (Xiao et al. 2025; Wang et al. 2024). Unlike general tool-use scenarios, EO workflows depend on available observations, sensing modalities, and intermediate products rather than task descriptions alone (Zhu et al. 2017). Without such domain constraints, agents may select irrelevant operations, misuse inputs or products, and construct invalid workflows. Since these rules vary across tasks and sensing contexts, they are dificult to encode in static prompts, motivating task-conditioned execution knowledge that captures scientific compatibility and can be reused across tasks while continuously adapting to newly encountered sensing scenarios.

![](images/85a30ea91137c1db139b8df7a2237204a8ae549112f19c6e58157c62a18f1706.jpg)  
Figure 1: Comparison of traditional EO agents and GeoForge. (a) Traditional agents fail with mismatched tools and broken trajectories. (b) GeoForge uses self-evolving and task-aware filtering for reliable EO task execution.

Existing EO agents commonly adopt general reasoning and action frameworks, in which an LLM constructs workflows through iterative reasoning and tool interaction (Xu et al. 2024; Feng et al. 2025). Recent benchmarks have demonstrated the potential of this paradigm, but also revealed persistent limitations in tool selection, execution order, argument grounding, and workflow completion (Shabbir et al. 2025; Li et al. 2025). As illustrated in Figure 1, exposing the agent to a large heterogeneous EO tool space without explicit execution guidance may lead to mismatched operations, invalid parameters, and broken trajectories. Without reusable priors about scientific compatibility and valid operation order, the agent must explore a broad action space for each query, resulting in redundant calls, incompatible operations, and incomplete workflows.

Related self-evolving geospatial systems have explored how external knowledge and interaction experience can improve geospatial problem solving without updating the backbone LLM. GeoEvolve (Luo et al. 2025) uses knowledgeguided code evolution to discover and refine task-specific geospatial algorithms, while GeoEvolver distills successful interactions and failure attributions into an evolving experience bank for fine-grained EO tool execution (Dai et al. 2026). These studies demonstrate two complementary routes to self-evolution. GeoEvolve evolves algorithm implementations, whereas GeoEvolver accumulates interaction-level tool expertise. A distinct challenge remains in organizing reusable execution knowledge across an EO workflow as a whole, from global operation order to local action corrections and procedural constraints. Raw trajectories preserve excessive instance-specific detail, while compact textual summaries may omit the product dependencies, parameter conditions, and execution structure that determine workflow validity. The key challenge is therefore to transform completed trajectories into execution priors that are domainvalid, structured, task-conditioned, and safely updatable.

To address this challenge, we propose GeoForge, a training-free, self-evolving framework that maintains a structured nonparametric execution state outside the backbone LLM. Given a task, GeoForge first infers the sensing context to constrain the operation space and provide execution guidance. It then constructs a task-conditioned execution prior from three complementary memories: Workflow Graph Memory captures global workflow structures, Action-Level Experiences provide local corrections, and the Adapted Skill SOP preserves reusable procedures, data requirements, and parameter constraints. Together, these components guide operation selection and composition while grounding final answers in current observations and geospatial products. After each task, GeoForge distills grounded trajectories into reusable workflow structures, action-level lessons, and procedural knowledge, enabling continual improvement without updating model parameters. Experiments on multiple geospatial benchmarks demonstrate that GeoForge substantially improves task accuracy and tool-use trajectory quality over existing EO agents, while reducing tool-planning and reasoning errors across most backbone LLMs. The contributions are summarized as follows:

• We propose GeoForge, a training-free, self-evolving Earth observation agent framework. By continuously distilling remote-sensing execution trajectories into structured knowledge representations, GeoForge enables sustained planning and tool-use advancement across heterogeneous geospatial modalities without perturbing backbone LLM weights.

• We introduce a three-level non-parametric memory together with a task-aware execution strategy, combining modality-aware tool filtering, Workflow Graph Memory, Action-Level Experiences, and an Adapted Skill SOP to provide structured planning priors and support continual self-evolution.

• Extensive experiments on multiple Earth observation agent benchmarks demonstrate that GeoForge consistently improves both task accuracy and execution trajectory quality across diverse LLM backbones, validating the efectiveness for reliable EO tool use.

## Related Work

## Agents for Earth Observation

Recent advances in large language models have catalyzed tool-augmented agents for geospatial and Earth observation applications. Early eforts like Remote Sensing ChatGPT (Guo et al. 2024) and RS-Agent (Xu et al. 2024) demonstrated chaining perception and analysis modules under LLM planners, yet they reported final accuracy without structured evaluation of tool-use trajectories. Subsequent benchmarks raised evaluation rigor: ThinkGeo (Shabbir et al. 2025) introduced tasks requiring detection, segmentation, and geospatial reasoning; OpenEarthAgent (Shabbir et al. 2026) extended this with a broader tool ecosystem. TerraBench (Nguyen et al. 2026) unified EO imagery, gridded environmental data, GIS reasoning, and simulation into a single framework with process-level metrics, revealing argument grounding and numeric reasoning as bottlenecks. Methodological innovations include Geo-OLM (Stamoulis and Marculescu 2025) with state-driven reasoning that decouples task progression from tool calling, and Earth-Agent (Feng et al. 2025) with a ReActstyle framework integrating specialized tools. Hierarchical task abstraction (Li et al. 2025) has also been explored. TerraLogic (Yan et al. 2026) introduced a benchmark for hierarchical reasoning with error-aware replanning across optical, SAR, and IR modalities.

## Self-Evolving Agents

A parallel line of research explores agent improvement from interaction experience without parameter updates. ReAct (Yao et al. 2022) demonstrated iterative reasoning-acting loops, with surveys organizing this field (Tao et al. 2024). Memento introduced Read-Write Reflective Learning, where episodic memory stores outcomes and retrieval enables policy improvement (Zhou et al. 2025); Memento-Skills treats reusable skills as artefacts that evolve through reflective revision (Zhou et al. 2026). MemSkill reframes memory extraction as learnable skills with a controller-selector-designer architecture (Zhang et al. 2026). Trajectory-Informed Memory Generation extracts actionable learnings from execution trajectories (Fang et al. 2026). SAGE introduces a selfevolving graph-memory engine (Wang et al. 2026a), and

XSkill proposes continual learning from experience in multimodal agents (Jiang et al. 2026). These approaches show that continual learning need not reside in model weights; a self-improving memory or skill library can serve as persistent intelligence. Recent eforts have begun extending these ideas to Earth observation, including GeoEvolve, which automates geospatial model discovery through multi-agent collaboration (Luo et al. 2025), and Experience-Driven Multi-Agent Systems(Dai et al. 2026). However, they do not explicitly optimize tool-space exploration and workflow reuse for heterogeneous EO reasoning. GeoForge addresses this gap through task-aware tool constraints, retrieval-augmented planning, and continual non-parametric memory updates.

## Methodology

## Overview

As shown in Figure 2, we introduce GeoForge, a selfevolving EO agent that learns procedural knowledge for remote-sensing analysis without updating the weight of backbone. Workflow Graph Memory, Action-Level Experience, and an Adapted Skill SOP jointly guide EO data inspection, geospatial product construction, spatiotemporal reasoning, and argument grounding. After each episode, safety-gated trajectory distillation updates this external state.

## Problem Formulation

An EO task is $\boldsymbol { x } = ( q , d , \boldsymbol { y } )$ , comprising a scientific request, data collection, and answer space. Given an operation library $\mathcal { T } = \{ T _ { i } \} _ { i = 1 } ^ { n }$ , the agent selects either $a _ { t } ~ = ~ ( T _ { i } , \theta _ { t } )$ with $\theta _ { t } \in \Omega _ { i }$ , or a prediction $y \in \mathcal { V }$ . A trajectory is

$$
\tau = ( q , a _ { 1 } , o _ { 1 } , \ldots , a _ { m } , o _ { m } , y ) , \quad o _ { t } = T _ { i } ( \theta _ { t } ) ,\tag{1}
$$

Observations include EO data inventories, derived geospatial products, perception outputs, regional or temporal statistics, and execution errors. A valid trajectory must respect remote-sensing dataflow: relevant observations and modalities are identified before product generation, returned artifacts are propagated to downstream tools, and aggregation is performed only after checking spatial and temporal compatibility. Final answers must be supported by the current EO data rather than retrieved memory.

Let $z = [ T ( a _ { 1 } ) , \dots , T ( a _ { m } ) ]$ and $\Theta _ { \tau } = [ \theta _ { 1 } , \dots , \theta _ { m } ]$ denote the operation and argument traces. We evaluate final correctness, operation coverage and order, argument grounding, repetition, and incompatible transitions. Without expert trajectories at inference time, GeoForge approximates these execution priors through its external state.

## Workflow Graph Memory

We represent high-level execution knowledge as a directed, reliability-aware graph $\mathcal { G } = ( \mathcal { W } , \mathcal { A } , \mathcal { U } )$ of workflow nodes, tool-transition edges, and usage statistics. Unlike a generic API graph, it encodes EO dependencies such as data discovery → modality-aware input organization → geospatial product generation → spatial or temporal aggregation → scientific interpretation. A node is

$$
w _ { i } = ( c _ { i } , r _ { i } , { \bf z } _ { i } , { \bf p } _ { i } , { \bf b } _ { i } , Q _ { i } , n _ { i } ^ { + } , n _ { i } ^ { - } ) ,\tag{2}
$$

where $c _ { i }$ is the task type, $r _ { i }$ the workflow description, $\mathbf { z } _ { i }$ an order-preserving compressed tool sequence, p<sub>i</sub> parameter hints, $\mathbf { b } _ { i }$ cautions, $Q _ { i }$ provenance, and $n _ { i } ^ { + } , n _ { i } ^ { - }$ success/failure counts. Removing repeated tool names separates reusable workflow structure from data-expanded repetition.

For retrieval, each workflow is converted into a heterogeneous text-symbol object containing task type, summary, tool sequence, parameter hints, and cautions. Given a query $q ,$ the graph module infers a coarse EO task type $\hat { c } ( q )$ and scores workflow nodes by combining two complementary similarity measures: tool sequence Jaccard and text seman tics similarity. The overall score is computed as

$$
\begin{array} { r l } & { s _ { G } ( q , w _ { i } ) = \alpha \cdot J _ { \mathrm { t o o l } } ( q , w _ { i } ) + \beta \cdot J _ { \mathrm { t e x t } } ( q , w _ { i } ) } \\ & { ~ + ~ \lambda _ { C } \mathbf { 1 } [ c _ { i } = \hat { c } ( q ) ] - \lambda _ { L } \operatorname* { m a x } ( 0 , | \mathbf { z } _ { i } | - L _ { 0 } ) , } \end{array}
$$

where

(3)

$$
J _ { \mathrm { t o o l } } ( q , w _ { i } ) = \frac { | V _ { \mathrm { t o o l } } ( q ) \cap V _ { \mathrm { t o o l } } ( w _ { i } ) | } { \operatorname* { m a x } ( | V _ { \mathrm { t o o l } } ( q ) | , 1 ) }\tag{4}
$$

measures the overlap of tool sequences, and

$$
J _ { \mathrm { t e x t } } ( q , w _ { i } ) = \frac { | V _ { \mathrm { t e x t } } ( q ) \cap V _ { \mathrm { t e x t } } ( w _ { i } ) | } { \operatorname* { m a x } ( | V _ { \mathrm { t e x t } } ( q ) | , 1 ) }\tag{5}
$$

captures lexical afinity between the query and workflow description. The weighting coeficients α and $\beta$ balance the contribution of structural and semantic similarities, with $\alpha + \beta = 1$ . The term $\lambda _ { C }$ rewards task-type consistency, while $- \lambda _ { L }$ penalizes overly long tool sequences to favor parsimonious workflows.

Only nodes above score and support thresholds are serialized with their tool order, summary, parameter hints, and cautions. They are marked as planning priors, preventing past trajectories from replacing current observations.

## Action-Level Experiences

The experience bank stores fine-grained execution knowledge as non-parametric corrections for local decisions under previously observed conditions. An item $e _ { i } \in \mathcal { E }$ is

$$
e _ { i } = ( \chi _ { i } , \alpha _ { i } , \mathbf { a } _ { i } , \rho _ { i } , m _ { i } ) ,\tag{6}
$$

where $\chi _ { i }$ is a trigger, $\alpha _ { i }$ a correction, ${ \bf a } _ { i }$ the source operations, $\rho _ { i }$ provenance, and $m _ { i }$ metadata. The bounded pair $( \chi _ { i } , \alpha _ { i } )$ corrects local EO decisions such as modality and time alignment, parameter grounding, reuse of returned geospatial artifacts, handling of spatial incompatibility, or stopping after suficient evidence is obtained.

Given a query $q ,$ retrieval is formulated as a gated relevance model. Let $V ( \cdot )$ be a domain-aware lexical projection and $\hat { c } ( \cdot )$ a coarse task abstraction. We score each experience by:

$$
s _ { E } ( q , e _ { i } ) = \underbrace { \frac { \lvert V ( q ) \cap V ( \chi _ { i } \lVert \alpha _ { i } \rVert \rho _ { i } ) \rvert } { \operatorname* { m a x } ( \lvert V ( q ) \rvert , 1 ) } } _ { \mathrm { l e x i c a l ~ a f f i n i t y } } + \lambda _ { E } \underbrace { \mathbf { 1 } [ \hat { c } ( \chi _ { i } \lVert \rho _ { i } ) = \hat { c } ( q ) \rVert } _ { \mathrm { t a s k ~ c o n s i s t e n c y } } .\tag{7}
$$

The retrieved set is obtained under an evidence constraint:

$$
\begin{array} { r l } & { \mathcal { E } _ { q } = \mathrm { T o p K } _ { e _ { i } \in \mathcal { E } } \Big [ s _ { E } ( q , e _ { i } ) \cdot \mathbf { 1 } \big ( | V ( q ) \cap V ( e _ { i } ) | > 0 \big ) } \\ & { \qquad \cdot \mathbf { 1 } \big ( | \mathbf { a } _ { i } | > 0 \big ) \Big ] . } \end{array}\tag{8}
$$

![](images/bbe9466c1b816329ccea990fdd5bbd4104101ca7040bee48428e0331123e0e0d.jpg)  
Figure 2: Pipeline of GeoForge. The framework comprises three modules: a ReAct-based self-evolution closed loop, nonparametric memory construction with trajectory filtering and hybrid similarity matching, and multi-level knowledge abstraction for task skills and action experiences.

The last indicator excludes free-form reflections not grounded in executable traces. The rewrite-and-filter operator $R _ { \Theta }$ contextualizes retrieved experiences against the task and adapted SOP, preserving their condition–action form while removing inapplicable assumptions.

The bank evolves through novelty-preserving consolidation. Given candidates $\Delta \bar { \mathcal { E } } .$ , GeoForge distills relevant patterns from successful trajectories as condition–action rules, and derives insights from comparison by contrasting successes and failures to attribute errors $( \mathrm { e . g . }$ , modality mismatch, incorrect parameters, premature stopping) to corrective conditions. It then normalizes each item with $\nu ( \cdot )$ and admits only entries outside existing equivalence classes:

$$
\mathcal { E } _ { t + 1 } = \mathrm { T a i l } _ { N _ { E } } \left( \mathcal { E } _ { t } \cup \left\{ e \in \Delta \mathcal { E } : \nu ( e ) \notin \nu ( \mathcal { E } _ { t } ) \right\} \right) .\tag{9}
$$

This rule favors high-precision reusable corrections over idiosyncratic episodic details, ensuring that the experience bank accumulates generalizable knowledge from both positive and negative execution outcomes.

## Adapted Skill SOP

Whereas experiences provide local corrections, the SOP specifies reusable task decompositions, intermediate artifacts, and evidence aggregation. We write

$$
\begin{array} { r } { S = \{ s _ { j } \} _ { j = 1 } ^ { M } , \qquad s _ { j } = ( \mu _ { j } , \omega _ { j } , \pi _ { j } , \beta _ { j } , \kappa _ { j } ) , } \end{array}\tag{10}
$$

where $\mu _ { j } , \omega _ { j } , \pi _ { j } , \beta _ { j }$ , and $\kappa _ { j }$ encode the EO task, processing workflow, data and argument constraints, sensor/modalit semantics, and failure-prevention rules.

Directly injecting S would create a high-recall but low-precision prompt. GeoForge instead constructs a taskconditioned SOP through an adaptation operator

$$
\begin{array} { r } { S _ { q } = A _ { \Theta } ( S , q , \mathcal { E } _ { q } ; L _ { S } ) , } \end{array}\tag{11}
$$

here, A is an LLM compressor bounded by $L _ { S }$ . Constrained generation preserves relevant workflows and cautions while removing irrelevant and instance-specific details, yielding a compact task-local SOP. The adapted SOP aligns the retrieved graph $\mathcal { G } _ { q }$ with contextualized experiences $\widetilde { \mathcal { E } } _ { q } .$ . The graph supplies order, the SOP procedural semantics, and experiences local corrections, while execution evidence remains authoritative. The SOP is updated by replacementstyle consolidation rather than append-only accumulation. Given a distilled candidate $\Delta S .$ , GeoForge accepts it only if it is complete, compact, grounded in an executable trajectory, and passes the safety gate in Eq. 17. The update is

$$
\begin{array} { r } { S _ { t + 1 } = \left\{ \begin{array} { l l } { \mathrm { S a n i t i z e } ( \Delta \mathcal { S } ) , } & { \psi ( \tau , \hat { y } ) = 1 , } \\ { S _ { t } , } & { \psi ( \tau , \hat { y } ) = 0 , } \end{array} \right. } \end{array}\tag{12}
$$

here, Sanitize removes non-generalizable, duplicate, and instance-specific content, keeping the SOP compact.

## Evidence-Grounded Execution

At inference time, GeoForge assembles a compact execution context by concatenating retrieved graph context, actionlevel experiences, the adapted SOP, and deterministic EO task guidance:

$$
\begin{array} { r } { \mathcal { C } _ { q } = \mathrm { T r u n c } _ { L _ { C } } [ \mathcal { G } _ { q } \| \mathcal { E } _ { q } \| \mathcal { S } _ { q } \| \mathcal { H } _ { q } ] . } \end{array}\tag{13}
$$

here $\mathcal { H } _ { q }$ is deterministic EO guidance: inspect available data assets first, organize inputs by modality and acquisition time, prefer compatible batch operations, propagate returned product paths, and verify spatial support, temporal alignment, and parameter semantics before aggregation. Scientific comparisons or predictions are delayed until the required dataderived measurements are available.

<table><tr><td>Backbone</td><td>Method</td><td>Tool-A-O↑</td><td>Tool-I-0↑</td><td>Tool-E-M↑</td><td></td><td>Parameters↑ Accuracy↑</td></tr><tr><td rowspan="4">GPT-5</td><td>Earth-Agent (Feng et al. 2025)</td><td>71.16</td><td>60.68</td><td>45.79</td><td>25.91</td><td>63.16</td></tr><tr><td>GeoEvolver (Dai et al. 2026)</td><td>67.52</td><td>53.11</td><td>40.14</td><td></td><td>70.85</td></tr><tr><td>OpenEarth-Agent (Dai et al. 2026)</td><td></td><td></td><td></td><td></td><td>67.61</td></tr><tr><td>GeoForge (Ours)</td><td>79.39</td><td>61.98</td><td>47.39</td><td>22.20</td><td>74.33</td></tr><tr><td rowspan="4">Gemini-2.5-Flash</td><td>Earth-Agent (Feng et al. 2025)</td><td>61.80</td><td>50.78</td><td>40.92</td><td>23.43</td><td>55.06</td></tr><tr><td>GeoEvolver (Dai et al. 2026)</td><td>61.43</td><td>53.60</td><td>38.63</td><td></td><td>59.91</td></tr><tr><td>OpenEarth-Agent (Dai et al. 2026)</td><td></td><td></td><td></td><td></td><td>57.89</td></tr><tr><td>GeoForge (Ours)</td><td>80.35</td><td>65.13</td><td>50.95</td><td>25.46</td><td>67.91</td></tr><tr><td rowspan="4">DeepSeek-V3.1</td><td>Earth-Agent (Feng et al. 2025)</td><td>77.98</td><td>64.33</td><td>50.01</td><td>31.36</td><td>52.23</td></tr><tr><td>GeoEvolver (Dai et al. 2026)</td><td>61.85</td><td>47.92</td><td>37.39</td><td></td><td>59.68</td></tr><tr><td>OpenEarth-Agent (Dai et al. 2026)</td><td></td><td></td><td></td><td></td><td>56.68</td></tr><tr><td>GeoForge (Ours)</td><td>83.58</td><td>68.30</td><td>50.99</td><td>26.88</td><td>77.09</td></tr><tr><td rowspan="3">GPT-40</td><td>Earth-Agent (Feng et al. 2025)</td><td>66.88</td><td>53.20</td><td>47.47</td><td>27.92</td><td>44.94</td></tr><tr><td>GeoEvolver (Dai et al. 2026)</td><td>62.71</td><td>48.93</td><td>37.45</td><td></td><td>65.59</td></tr><tr><td>GeoForge (Ours)</td><td>78.91</td><td>37.31</td><td>52.46</td><td>33.81</td><td>57.98</td></tr><tr><td rowspan="3">Qwen3-Max</td><td>Earth-Agent (Feng et al. 2025)</td><td>70.14</td><td>56.02</td><td>42.74</td><td>26.27</td><td>47.37</td></tr><tr><td>OpenEarth-Agent (Dai et al. 2026)</td><td></td><td></td><td></td><td></td><td>52.63</td></tr><tr><td>GeoForge (Ours)</td><td>79.26</td><td>61.97</td><td>51.28</td><td>25.67</td><td>69.72</td></tr></table>

Table 1: Performance comparison of diferent Earth observation agents on Earth-Bench using diverse LLM backbones.

The agent then executes a standard ReAct transition over the filtered MCP tool space:

$$
a _ { t } \sim \pi _ { \Theta } ( \cdot \mid q , \mathcal { C } _ { q } , h _ { t } , \mathcal { T } _ { q } ) , \qquad h _ { t + 1 } = h _ { t } \cup \{ a _ { t } , o _ { t } \} .\tag{14}
$$

Memory is treated only as a processing prior, and every data-grounded EO question requires a tool call, typically beginning with data inventory. For incomplete responses, GeoForge reuses existing geospatial products and observations and invokes only missing operations. A deterministic-plus-LLM normalizer then maps the response to a valid option:

$$
\begin{array} { r } { \hat { y } = \left\{ \begin{array} { l l } { f _ { \mathrm { r e g e x } } ( r ) , } & { f _ { \mathrm { r e g e x } } ( r ) \neq \emptyset , } \\ { f _ { \Theta } ( q , \mathcal { V } , r , h _ { \mathrm { t a i l } } ) , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}\tag{15}
$$

where the normalizer creates no evidence; it only converts responses into the benchmark format.

## Safety-Gated Self-Evolution

After execution, GeoForge compacts tool calls, arguments, observations, product paths, and the answer into τ¯. A distiller produces

$$
D _ { \Theta } ( \bar { \tau } , q , \hat { y } ) = ( \Delta \mathcal { E } , \Delta \mathcal { S } , \Delta \mathcal { W } ) ,\tag{16}
$$

where $\Delta \mathcal { E }$ contains at most three lessons, $\Delta \boldsymbol { S }$ a revised SOP, and ∆W at most two workflow candidates. Distillation excludes sample-specific paths, answer shortcuts, external APIs, code, and workflows unsupported by actual tool calls. The update is safety-gated:

$$
\begin{array} { r l } & { \psi ( \tau , \hat { y } ) = \mathbf { 1 } [ \mathrm { T o o l s } ( \tau ) > 0 ] \mathbf { 1 } [ \phi ( \hat { y } ) = 0 ] } \\ & { \qquad \mathbf { 1 } [ | \mathrm { C a l l s } ( \tau ) | \leq L _ { \operatorname* { m a x } } ] \mathbf { 1 } [ \underset { T } { \operatorname* { m a x } } n _ { T } ( \tau ) \leq R _ { \operatorname* { m a x } } ] } \\ & { \qquad \mathbf { 1 } [ \Delta S \cap \mathcal { F } = \varnothing ] , } \end{array}\tag{17}
$$

where $\phi ( \cdot )$ is the incomplete-answer detector and $\mathcal { F }$ contains forbidden code-like or non-MCP patterns. The memory transition is

$$
\mathcal { M } _ { t + 1 } = ( U _ { G } ( \mathcal { G } _ { t } , \Delta \mathcal { W } ) , U _ { E } ( \mathcal { E } _ { t } , \Delta \mathcal { E } ) , U _ { S } ( \mathcal { S } _ { t } , \Delta \mathcal { S } ; \psi ) )\tag{18}
$$

where $U _ { G }$ normalizes candidates, compresses tool sequences, and merges nodes using weighted tool-sequence and lexical Jaccard similarity before updating edges and pruning by reliability. $U _ { E }$ inserts deduplicated experiences, while $U _ { S }$ replaces the SOP only when ψ = 1. Appendix provides the full algorithm.

## Experiments

## Experimental Setup

Datasets. We evaluate GeoForge on three agentic geospatial benchmarks: Earth-Bench (Feng et al. 2025), ThinkGeo (Shabbir et al. 2025), and GeoPlan-bench (Li et al. 2025). Together, they cover executable EO analysis, optical/SAR tool reasoning, and high-level geospatial planning, answer correctness, tool selection, parameter use, visual grounding, and workflow completeness. Details are provided in Appendix.

Evaluation Protocols. Following Dai et al. (2026), we adopt the oficial evaluation protocols of each benchmark and report final accuracy alongside trajectory-level metrics including Tool-Any-Order (tool coverage), Tool-In-Order (execution order fidelity), Tool-Exact-Match (full sequence precision), Instruction Alignment, Tool Accuracy, Argument Accuracy, Answer Accuracy, and Answer Accuracy w/ ImgGen (parameter correctness for image generation). Additional fine-grained metrics including Parameter Accuracy, Key Tool Recall/Precision, Path Similarity, and Completeness Score are detailed in the Appendix.

Implementation Details. We implement GeoForge using LangChain and LangGraph. The LLM is queried at temperature 0.1 with a 120-second timeout, and tools are served via MCP. We adopt a recursion limit of 80 with at most two continuation attempts. The memory system uses top-3 experience retrieval and top-1 workflow retrieval (threshold 0.6). Additional details are provided in the Appendix.

<table><tr><td colspan="6">ThinkGeo</td></tr><tr><td>Method</td><td>Inst. Align.↑</td><td>Tool Acc.↑</td><td>Arg Acc.↑</td><td>Ans.↑</td><td>Ans. w/ ImgGen↑</td></tr><tr><td>GPT-40</td><td>82.33</td><td>67.73</td><td>34.75</td><td>9.78</td><td>20.40</td></tr><tr><td>GPT-4-1106</td><td>86.49</td><td>74.05</td><td>36.96</td><td>5.16</td><td>14.69</td></tr><tr><td>Claude-3.7-Sonnet</td><td>22.31</td><td>27.31</td><td>9.00</td><td>8.97</td><td>7.51</td></tr><tr><td>Qwen2.5-7b-Instruct</td><td>64.88</td><td>51.04</td><td>20.08</td><td>7.34</td><td>6.40</td></tr><tr><td>GeoEvolver (Dai et al. 2026)</td><td>80.12</td><td>76.58</td><td>35.42</td><td>46.88</td><td>53.74</td></tr><tr><td>GeoForge (Ours)</td><td>97.27</td><td>80.31</td><td>33.96</td><td>60.98</td><td>62.43</td></tr><tr><td colspan="6">GeoPlan-Bench</td></tr><tr><td>Method</td><td> $R e c a l l _ { k e y } \uparrow$ </td><td> $P r e c i s i o n _ { k e y } \uparrow$ </td><td> $F I _ { k e y } \uparrow$ </td><td>Structural↑</td><td>Holistic↑</td></tr><tr><td>CoT (Wei et al. 2022)</td><td>0.40</td><td>0.51</td><td>0.42</td><td>0.55</td><td>980.87</td></tr><tr><td>ReAct (Yao et al. 2022)</td><td>0.33</td><td>0.50</td><td>0.37</td><td>0.47</td><td>962.57</td></tr><tr><td>Plan&amp;Execute (Wang et al. 2023)</td><td>0.38</td><td>0.55</td><td>0.43</td><td>0.47</td><td>973.52</td></tr><tr><td>Debate (Du et al. 2024)</td><td>0.62</td><td>0.57</td><td>0.57</td><td>0.62</td><td>1030.59</td></tr><tr><td>AFlow (Zhang et al. 2025)</td><td>0.39</td><td>0.62</td><td>0.44</td><td>0.68</td><td>992.60</td></tr><tr><td>EarthAgent-MAS (Li et al. 2025)</td><td>0.66</td><td>0.65</td><td>0.63</td><td>0.68</td><td>1068.27</td></tr><tr><td>GeoEvolver (Dai et al. 2026)</td><td>0.64</td><td>0.68</td><td>0.63</td><td>0.45</td><td>1057.40</td></tr><tr><td>GeoForge (Ours)</td><td>1.00</td><td>0.70</td><td>0.77</td><td>0.79</td><td>1100.65</td></tr></table>

Table 2: Performance comparison on ThinkGeo and GeoPlan-Bench.

<table><tr><td>Graph Skill</td><td></td><td>Experience Accuracy</td><td></td><td>Tool-A-O</td><td>Tool-I-O</td><td>Tool-E-M</td></tr><tr><td></td><td></td><td></td><td>52.23</td><td>78.66</td><td>64.50</td><td>49.58</td></tr><tr><td></td><td>√</td><td>√</td><td>67.02</td><td>75.69</td><td>61.74</td><td>45.09</td></tr><tr><td>√</td><td></td><td>√</td><td>52.66</td><td>80.96</td><td>67.39</td><td>48.57</td></tr><tr><td>√</td><td>√</td><td></td><td>70.21</td><td>76.45</td><td>62.58</td><td>47.07</td></tr><tr><td>√</td><td>√</td><td>√</td><td>74.33</td><td>85.89</td><td>70.01</td><td>51.18</td></tr></table>

Table 3: Ablation study of GeoForge on Earth-Bench.

## Overall Performance Comparison

Table 1 compares GeoForge with existing EO agents across five LLM backbones on Earth-Bench. GeoForge achieves the best accuracy on four backbones: GPT-5 at 74.33%, DeepSeek-V3.1 at 77.09%, Gemini-2.5-Flash at 67.91%, and Qwen3-Max at 69.72%, consistently outperforming Earth-Agent, OpenEarth-Agent, and GeoEvolver. It also improves trajectory-level metrics such as Tool-Any-Order, Tool-In-Order, and Tool-Exact-Match on most backbones, demonstrating more reliable tool selection and execution planning. These results validate that GeoForge enhances both task accuracy and reasoning quality for Earth observation agents.

Table 2 further evaluates GeoForge on ThinkGeo and GeoPlan-Bench. On ThinkGeo, GeoForge achieves the highest Instruction Alignment at 97.27%, Tool Accuracy at 80.31%, Answer Accuracy at 60.98%, and grounded Answer Accuracy at 62.43%. On GeoPlan-Bench, it obtains the best $\mathtt { R e c a l l } _ { k e y }$ of 1.00, $\mathrm { F } 1 _ { k e y }$ of 0.77, Structural score of 0.79, and Holistic score of 1100.65, surpassing both traditiona planners and recent EO agents. These results demonstrate that GeoForge improves not only low-level tool execution but also high-level workflow planning for complex EO tasks.

## Ablation Study

Component Contribution. The ablation results in Table 3 show that all three memory components contribute to GeoForge. Without memory, the model achieves only 52.23% accuracy. Removing Skill causes the largest accuracy drop, from 74.33% to 52.66%, highlighting the importance of reusable task-level knowledge. In contrast, removing the Workflow Graph leads to the largest degradation in trajectory quality, reducing Tool-Any-Order, Tool-In-Order, and Tool-Exact-Match from 85.89%, 70.01%, and 51.18% to 75.69%, 61.74%, and 45.09%, respectively. Using all three components yields the best performance, demonstrating that graph-, skill-, and experience-level memories provide complementary guidance for multi-step geospatial reasoning.

<table><tr><td>Method</td><td>Spectrum</td><td>Products</td><td>RGB</td><td>Avg.</td></tr><tr><td>GPT-Agent (Achiam et al. 2023)</td><td>45.00</td><td>31.60</td><td>45.26</td><td>40.42</td></tr><tr><td>MGX (Hong et al. 2024)</td><td>40.00</td><td>15.80</td><td>0.00</td><td>18.60</td></tr><tr><td>Manus (Shen et al. 2025)</td><td>15.00</td><td>15.80</td><td>47.62</td><td>26.14</td></tr><tr><td>Coze (Kemppainen 2025)</td><td>35.00</td><td>10.50</td><td>0.00</td><td>15.30</td></tr><tr><td>Earth-Agent (Feng et al. 2025)</td><td>50.00</td><td>42.11</td><td>51.43</td><td>47.84</td></tr><tr><td>OpenEarth-Agent (Dai et al. 2026)</td><td>51.00</td><td>45.98</td><td>1.67</td><td>32.88</td></tr><tr><td>GeoForge (Ours)</td><td>77.00</td><td>71.26</td><td>37.29</td><td>61.85</td></tr></table>

Table 4: Comparison with general agents across Spectrum, Products, and RGB modalities.

Generalization across Modalities. As shown in Table 4, GeoForge consistently outperforms all general-purpose agents on Earth-Bench, raising average accuracy from 47.84% to 61.85% over the strongest baseline Earth-Agent. It achieves the highest scores on Spectrum at 77.00% and Products at 71.26%, with gains of 27.00 and 29.15 percentage points respectively. These substantial improvements on tasks demanding complex spectral processing and multi-step workflows demonstrate that GeoForge’s structured memory excels at intricate tool orchestration, validating the efectiveness of its graph-skill-experience architecture for multimodal EO reasoning.

![](images/92cdbbc7fd09aa6495dc63d3051fb69b9958d820cafa6fc04ade6ef8c9989892.jpg)  
Figure 3: Comparison of tool-use trajectories between the baseline and GeoForge on Earth-Bench.

Sensitivity Analysis. As shown in Figure 5, GeoForge is robust to both the number of retrieved skills and the retrieval score threshold. Retrieving a small number of highquality skills achieves the best performance, while excessive retrieval slightly degrades trajectory quality due to redundant guidance. Similarly, an overly strict retrieval threshold reduces Tool-Exact-Match by filtering out useful experiences. Therefore, we adopt Top-k = 3 and Min-Retrieve-Score = 0.6, which provide the best trade-of between retrieval quality and planning performance.

## Error Analysis and Case Study

Figure 4 presents the error distribution across backbone LLMs. Our method consistently improves the correct-answer ratio for all models. These improvements primarily stem from substantial reductions in tool planning errors across most backbones. Reasoning errors are eliminated for the majority of models, and answer or format failures are also notably reduced. Remaining errors are mainly execution-trace and parameter-related issues, suggesting that our method shifts failures from high-level planning toward deeper executionlevel challenges. Figure 3 compares the execution trajectories of the baseline and our method on a representative Earth observation task. Given the task of computing the NBR index, the baseline follows a tangled trajectory with many redundant tool calls, ultimately failing to produce a valid answer. In contrast, our method executes a concise workflow that retrieves the file list, computes the NBR index, detects hotspot regions, and performs a final directional analysis, producing the correct answer B with substantially fewer steps. This demonstrates that GeoForge efectively eliminates redundant actions and maintains task-relevant execution, leading to more reliable problem-solving in complex EO scenarios.

![](images/78e71ea00ee93bf84209b93e028748e79e898a2509953cb85cc774c3b46d4013.jpg)

Figure 4: The percentage distribution of correct predictions and various error types across diferent backbone LLMs.  
![](images/ff8ba818f7838b1cc82393d459cc786df892ed86ccd9f8cae594667f09fc0479.jpg)  
Figure 5: Sensitivity analysis of the hyperparameters. Left: the efect of the number of retrieved skills (Top-k). Right: the efect of the minimum retrieval score threshold.

## Conclusion

In this paper, We presented GeoForge, a training-free, selfevolving framework for Earth observation agents that distills completed trajectories into structured nonparametric memory. GeoForge constrains the action space by sensing context and retrieves task-conditioned priors from workflow graphs, action-level experiences, and an adapted skill SOP. Retrieved priors guide execution while observations ground final answers. A safety-gated distillation process enables continuous memory evolution without backbone LLM updates. Experiments across multiple benchmarks show that GeoForge consistently improves task accuracy and tool-use trajectory quality while reducing planning and reasoning errors, ofering a scalable paradigm for reliable scientific agents.

## References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Dai, P.; Xuan, W.; Wang, J.; Chen, H.; Song, J.; Ou, Y.; and

Yokoya, N. 2026. Experience-Driven Multi-Agent Systems Are Training-free Context-aware Earth Observers. arXiv preprint arXiv:2602.02559.

Du, Y.; Li, S.; Torralba, A.; Tenenbaum, J. B.; and Mordatch, I. 2024. Improving factuality and reasoning in language models through multiagent debate. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Fang, G.; Isahagian, V.; Jayaram, K.; Kumar, R.; Muthusamy, V.; Oum, P.; and Thomas, G. 2026. Trajectory-informed memory generation for self-improving agent systems. arXiv preprint arXiv:2603.10600.

Feng, P.; Lv, Z.; Ye, J.; Wang, X.; Huo, X.; Yu, J.; Xu, W.; Zhang, W.; Bai, L.; He, C.; et al. 2025. Earth-agent: Unlocking the full landscape of earth observation with agents. arXiv preprint arXiv:2509.23141.

Guo, H.; Su, X.; Wu, C.; Du, B.; Zhang, L.; and Li, D. 2024. Remote sensing chatgpt: Solving remote sensing tasks with chatgpt and visual models. In IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium, 11474–11478. IEEE.

Hong, S.; Zhuge, M.; Chen, J.; Zheng, X.; Cheng, Y.; Wang, J.; Zhang, C.; Yau, S.; Lin, Z.; Zhou, L.; et al. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations, volume 2024, 23247–23275.

Jiang, G.; Su, Z.; Qu, X.; and Fung, Y. R. 2026. Xskill: Continual learning from experience and skills in multimodal agents. arXiv preprint arXiv:2603.12056.

Kemppainen, J. 2025. Exploring the power of Coze’s nocode platform.

Li, K.; Wang, J.; Wang, Z.; Qiao, H.; Zhang, W.; Meng, D.; and Cao, X. 2025. Designing domain-specific agents via hierarchical task abstraction mechanism. arXiv preprint arXiv:2511.17198.

Luo, P.; Lou, X.; Zheng, Y.; Zheng, Z.; and Ermon, S. 2025. GeoEvolve: Automating Geospatial Model Discovery via Multi-Agent Large Language Models. arXiv:2509.21593.

Nguyen, D. T.; Nguyen, T.; Maani, F. A.; Le, H. M.; Sheikh, M. U.; Saeed, N.; Khan, M. H.; and Khan, S. 2026. TerraBench: Can Agents Reason Over Heterogeneous Earth-System Data? arXiv preprint arXiv:2606.13148.

Shabbir, A.; Munir, M. A.; Dudhane, A.; Sheikh, M. U.; Khan, M. H.; Fraccaro, P.; Moreno, J. B.; Khan, F. S.; and Khan, S. 2025. Thinkgeo: Evaluating tool-augmented agents for remote sensing tasks. arXiv preprint arXiv:2505.23752.

Shabbir, A.; Sheikh, M. U.; Munir, M. A.; Debary, H.; Fiaz, M.; Zaheer, M. Z.; Fraccaro, P.; Khan, F. S.; Khan, M. H.; Zhu, X. X.; et al. 2026. Openearthagent: A unified framework for tool-augmented geospatial agents. arXiv preprint arXiv:2602.17665.

Shen, M.; Li, Y.; Chen, L.; Fan, Z.; Li, Y.; and Yang, Q. 2025. From mind to machine: The rise of manus ai as a fully autonomous digital agent. arXiv preprint arXiv:2505.02024.

Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2024. Reflexion: Language Agents with Verbal Reinforcement Learning. Advances in Neural Information Processing Systems, 36.

Stamoulis, D.; and Marculescu, D. 2025. Geo-olm: Enabling sustainable earth observation studies with cost-eficient open language models & state-driven workflows. In Proceedings ofthe ACM SIGCAS/SIGCHI Conference on Computing and Sustainable Societies, 608–619.

Tao, Z.; Lin, T.-E.; Chen, X.; Li, H.; Wu, Y.; Li, Y.; Jin, Z.; Huang, F.; Tao, D.; and Zhou, J. 2024. A survey on self-evolution of large language models. arXiv preprint arXiv:2404.14387.

Van de Weghe, N.; De Sloover, L.; Cohn, A. G.; Huang, H.; Scheider, S.; Sieber, R. E.; Timpf, S.; and Claramunt, C. 2025. Opportunities and challenges of integrating geographic information science and large language models.

Wang, J.; Zhao, H.; Wang, X.; Wang, Y.; Deng, Q.; Zhang, M.; et al. 2026a. SAGE: A Self-Evolving Agentic Graph-Memory Engine for Structure-Aware Associative Memory. arXiv preprint arXiv:2605.12061.

Wang, L.; Xu, W.; Lan, Y.; Hu, Z.; Lan, Y.; Lee, R. K.-W.; and Lim, E.-P. 2023. Plan-and-solve prompting: Improving zeroshot chain-of-thought reasoning by large language models. In Proceedings ofthe 61st annual meeting ofthe association for computational linguistics (volume 1: long papers), 2609– 2634.

Wang, P.; HU, H.; FENG, Y.; DIAO, W.; and SUN, X. 2026b. A Large-Scale Multimodal Instruction Dataset for Remote Sensing Agents. Journal ofElectronics & Information Technology, 48(4): 1608–1622.

Wang, S.; Hu, T.; Xiao, H.; Li, Y.; Zhang, C.; Ning, H.; Zhu, R.; Li, Z.; and Ye, X. 2024. GPT, large language models (LLMs) and generative artificial intelligence (GAI) models in geospatial science: a systematic review. International Journal ofDigital Earth, 17(1): 2353122.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35: 24824–24837.

Wu, Q.; Bansal, G.; Zhang, J.; Wu, Y.; Li, B.; Zhu, E.; Jiang, L.; Zhang, X.; Zhang, S.; Liu, J.; et al. 2023. Autogen: Enabling next-gen llm applications via multi-agent conversation. arXiv preprint arXiv:2308.08155.

Xiao, A.; Xuan, W.; Wang, J.; Huang, J.; Tao, D.; Lu, S.; and Yokoya, N. 2025. Foundation models for remote sensing and earth observation: A survey. IEEE Geoscience and Remote Sensing Magazine.

Xu, W.; Yu, Z.; Mu, B.; Wei, Z.; Zhang, Y.; Li, G.; and Peng, M. 2024. RS-Agent: automating remote sensing tasks through intelligent agent. arXiv preprint arXiv:2406.07089.

Yan, Y.; Mou, L.; Yang, B.; and Li, Q. 2026. TerraLogic: A Benchmark for Hierarchical Geospatial Reasoning in Earth Observation. arXiv preprint arXiv:2607.12497.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Zhang, H.; Long, Q.; Bao, J.; Feng, T.; Zhang, W.; Yue, H.; and Wang, W. 2026. MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents. arXiv preprint arXiv:2602.02474.

Zhang, J.; Xiang, J.; Yu, Z.; Teng, F.; Chen, X.; Chen, J.; Zhuge, M.; Cheng, X.; Hong, S.; Wang, J.; et al. 2025. Aflow: Automating agentic workflow generation. In International Conference on Learning Representations, volume 2025, 34040–34077.

Zhou, H.; Chen, Y.; Guo, S.; Yan, X.; Lee, K. H.; Wang, Z.; Lee, K. Y.; Zhang, G.; Shao, K.; Yang, L.; et al. 2025. Memento: Fine-tuning llm agents without fine-tuning llms. arXiv preprint arXiv:2508.16153.

Zhou, H.; Guo, S.; Liu, A.; Yu, Z.; Gong, Z.; Zhao, B.; Chen, Z.; Zhang, M.; Chen, Y.; Li, J.; et al. 2026. Memento-skills: Let agents design agents. arXiv preprint arXiv:2603.18743.

Zhu, X. X.; Tuia, D.; Mou, L.; Xia, G.-S.; Zhang, L.; Xu, F.; and Fraundorfer, F. 2017. Deep learning in remote sensing: A comprehensive review and list of resources. IEEE Geoscience and Remote Sensing Magazine, 5(4): 8–36.
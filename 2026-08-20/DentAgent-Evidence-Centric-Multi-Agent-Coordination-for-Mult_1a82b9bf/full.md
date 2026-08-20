# DentAgent: Evidence-Centric Multi-Agent Coordination for Multimodal Dental Reasoning

Zijie Meng<sup>1,⋆</sup>, Xiwei Dai<sup>1,⋆</sup>, Yixuan Tang<sup>1</sup>, Jin Hao<sup>2</sup>, Yang Feng<sup>3</sup>, Fudong Zhu<sup>1</sup>,

Xiaoqiang Liu<sup>4</sup>, Shaosheng Cao<sup>5</sup>, Zuozhu Liu<sup>1,†</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>Shanghai Jiao Tong University, <sup>3</sup>Angelalign Technology Inc., <sup>4</sup>Peking University, <sup>5</sup>Tsinghua University

Abstract—Oral diseases affect billions of people worldwide, underscoring a pressing need for accurate and reliable dental assessment that integrates heterogeneous evidence from domain knowledge, radiographs, intraoral photographs, and 3D dental data. Most existing dental AI systems remain modality- or taskspecific. Although recent vision-language models support flexible dental question answering, directly generated response leaves evidence implicit and untraceable. To address these limitations, we introduce DentAgent, an evidence-centric multi-agent framework, in which the Orchestrator coordinate five specialized agents spanning various modalities. Each specialist utilizes domain tools to convert observations into structured evidence records. The Evidence Blackboard manages these records as a shared evidence state, tracking coverage, gaps, and conflicts before response generation. This standardized evidence representation integrates isolated dental capabilities into a unified agentic workflow. Across four benchmarks, DentAgent demonstrates leading performance, even surpassing the senior specialists by 17.3 percentage points on multi-label diagnosis, which supports its value for broadly applicable and traceable multimodal dental reasoning, and highlights its potential as a technical foundation for population oral health assessment and management.

Index Terms—Dental AI, Multi-Agent Systems, Multimodal Large Language Models.

## I. INTRODUCTION

Oral diseases affect billions of people worldwide and impose substantial health and socioeconomic burdens [1]– [3]. Effective prevention and treatment depend in part on accurate and reliable dental assessment, which often draws on heterogeneous evidence, including domain knowledge, radiographs, intraoral photographs, and 3D dental data. However, these sources differ substantially in representation, anatomical granularity, and evidential role, requiring different forms of perception, measurement, and reasoning. This heterogeneity creates a central challenge for dental AI: a broadly applicable system must coordinate modality-specific capabilities while making explicit which observations support each conclusion.

Recent dental AI systems have made substantial progress in addressing individual parts of this problem, ranging from task-specific perception and measurement models to domainoriented foundation and generative models [4]–[9]. However, most existing systems remain organized around individual modalities or narrowly defined tasks. Specialized models provide useful local capabilities but typically operate as isolated pipelines, whereas generative models often map their inputs directly to answers or reports. In both cases, the evidence supporting the model output generally remains implicit, making it difficult to trace a prediction or response to specific visual findings, retrieved knowledge, or quantitative measurements.

Tool-augmented and agentic reasoning provides a natural alternative to direct generation by allowing language models to invoke specialized tools and reason over their outputs as intermediate observations [10]–[12]. Recent dental agents have applied this paradigm to selected imaging modalities and dental workflows [13]–[15]. But these systems remain largely modality- or task-specific. More importantly, tool use alone does not provide a common mechanism for coordinating heterogeneous capabilities, standardizing their outputs, linking them to their sources, or explicitly representing missing and conflicting evidence. A broader dental reasoning framework therefore requires not only specialized tools but also a structured evidence representation designed to connect modalityspecific observations to final answer generation.

To address these limitations, we propose DentAgent, an evidence-centric multi-agent framework that coordinates heterogeneous dental capabilities within a unified workflow. DentAgent primarily consists of an Orchestrator, five Modalityspecific Specialists, and an Evidence Blackboard. The Orchestrator identifies response-critical evidence awaiting acquisition and dynamically activates the relevant specialist agent. Each agent follows a bounded loop, iteratively selecting tools, examining observations, and converting them into structured evidence records. The Evidence Blackboard normalizes these evidence, tracks their coverage, links them across agents, and deals unresolved gaps or conflicts. Based on the evolving evidence state, the Orchestrator may request further analysis before committing the final answer.

We evaluate DentAgent on four benchmarks spanning bilingual dental knowledge [16], panoramic radiograph question answering (QA) [17], 2D dental diagnosis [7], and 3D intraoral scan (IOS) reasoning [18], where DentAgent obtains the leading scores in various scenarios, especially exceeding the senior specialists by 17.3 percentage points on multi-label diagnosis. Together, these results support the applicability and traceability of DentAgent as a unified framework for dental reasoning, while highlighting its potential as a technical foundation for population oral health assessment and management.

Our contributions are summarized as follows:

• We introduce DentAgent, a multi-agent framework that coordinates specialist capabilities for textual knowledge, radiographs, clinical photography, and 3D dental data under a unified orchestration protocol.

• We develop an evidence-centric coordination mechanism that decouples evidence acquisition from answer generation, enabling modality-specific specialists to contribute source-attributed findings to a shared Evidence Blackboard for target-driven reasoning and iterative evidencesufficiency assessment.

• We conduct evaluations on heterogeneous dental benchmarks, demonstrating the applicability of DentAgent across diverse modalities and tasks, with clear improvements over strong baselines.

## II. RELATED WORKS

## A. Task-Specific Dental Perception Models

Early dental AI research primarily addressed well-defined perception tasks within individual imaging modalities. Deep learning has been applied to caries detection on periapical and bitewing radiographs [19], [20], periodontal bone-loss detection and measurement [21], [22], tooth detection and numbering in panoramic radiographs [23], and instance-level tooth segmentation [24]. In orthodontics, public cephalometric challenges and automated landmarking systems established important benchmarks for landmark localization and measurementbased analysis [25], [26]. For 3D dental data, ToothNet, CBCT segmentation systems, MeshSNet, and TSegNet further demonstrate the importance of explicit geometric and spatial modeling for tooth- and surface-level understanding [27]–[32]. These studies provide strong perceptual capabilities for specific modalities and clinical tasks, but they generally operate in isolation and do not address how heterogeneous findings should be aligned, reconciled, and integrated across data sources.

## B. Dental Foundation and Generative Models

Recent studies have expanded dental AI beyond narrowly defined prediction tasks toward domain-specialized foundation and generative models. OralGPT introduces large-scale instruction data and evaluation protocols for panoramic Xray understanding, while DentalBench provides a bilingual benchmark for evaluating dental knowledge in LLMs [16], [17]. DentVLM extends vision-language modeling across multiple 2D oral imaging modalities and diagnostic tasks, whereas DentVFM investigates scalable vision foundation models for oral and maxillofacial radiology [7], [8]. DentFound further applies instance-guided vision-language modeling to panoramic radiograph interpretation and report generation [9]. Beyond 2D imaging, IOSVLM models the native geometry of 3D intraoral scans, CBCTRepD addresses bilingual report generation from CBCT data, and dental NLP studies explore evidence extraction and documentation from clinical notes [18], [33]–[35]. Collectively, these works demonstrate the value of dental-specific data and modality-aware representation learning [36]–[38]; however, they remain largely centered on the capabilities of individual models and provide limited support for explicitly coordinating and managing evidence produced by multiple specialized components.

## C. Agentic Dental AI

Building on advances in dental foundation and generative models, recent work has begun to shift from enhancing individual model capabilities toward coordinating multiple sources of specialized expertise. OralAgent integrates visual tools and dental knowledge retrieval for interactive dental image analysis [13], while OPGAgent orchestrates specialized perception modules, hierarchical evidence collection, and consensus-based reporting for panoramic radiograph interpretation [14]. OralGPT-Plus further investigates iterative and symmetry-aware tool use for panoramic X-ray analysis through reinforcement learning [39], and orthodontic agent systems demonstrate the potential of role-specialized collaboration for diagnosis and treatment planning [15]. These studies establish the value of tool-augmented and collaborative reasoning in dentistry, but they are generally designed around a particular imaging modality or clinical workflow. DentAgent extends this direction to a broader multimodal setting, in which specialists for clinical text, radiographs, clinical photography, and 3D dental data contribute complementary observations to a shared evidence blackboard, enabling structured evidence coordination and traceable response generation.

## III. METHODS

## A. Framework Overview

DentAgent is an evidence-centric hierarchical multi-agent framework for multimodal dental reasoning. Given a dental case $\boldsymbol { x } = ( q , c , \mathcal { M } )$ , where $q$ denotes the task query, c denotes optional clinical context, and $\mathcal { M }$ contains the available multimodal inputs, DentAgent produces a task-appropriate response y grounded in the evidence available in the case. The framework supports text questions, clinical documents, panoramic and cephalometric radiographs, intraoral photographs, and IOS meshes. Its output space covers dental knowledge answering, diagnostic category prediction, lesion localization, and restoration assessment.

DentAgent separates task interpretation, specialist execution, evidence organization, and response generation. The Intent Detector first maps the input case to a task representation τ, which specifies the concrete task requested and the expected form and granularity of the response. The Orchestrator then follows the loop from task identification, specialists delegation, evidence management to sufficiency verification, as illustrated in Fig. 1. Rather than directly producing a diagnosis, it identifies the evidence targets required by the task, assigns them to an appropriate subset of Modality-Specific Specialists, and repeatedly evaluates the resulting evidence state.

Each specialist operates within a bounded modality-specific reasoning, tool-use, and observation cycle. Each specialist activation is limited to at most $K _ { \mathrm { m a x } }$ tool calls, thereby preventing unbounded or non-terminating tool-use loops. The resulting observation packets are incorporated into the Evidence Blackboard through four successive operations: Evidence Normaization, Coverage Tracking, Evidence Linking, and Conflict Resolution. The updated blackboard is returned to the Orchestrator, enabling subsequent rounds to focus on unresolved or contradictory evidence rather than repeating completed analyses. Execution terminates when verification succeeds, no additional evidence target is identified, or the maximum number of global rounds $T _ { \mathrm { m a x } }$ is reached. The Response Generator then maps the terminal blackboard state to the requested output. Algorithm 1 summarizes the complete inference procedure.

![](images/b19f15e4fc28b40ba3963fb4a98305bbf12e39cf9a19af8e74a4814a1b7e99b3.jpg)  
Fig. 1. Overview of DentAgent, an evidence-centric hierarchical multi-agent framework for multimodal dental reasoning. Given a dental case $\boldsymbol { x } = ( q , c , \mathcal { M } )$ where q, c and M denote the query, clinical context and multimodal inputs respectively, the Intent Detector interprets the requested tasks, after which the Orchestrator iteratively identifies task-relevant evidence targets, delegates them to appropriate Modality-Specific Specialists, manages the updated evidence state, and verifies its sufficiency. Each activated specialist performs a bounded reasoning, tool-use and observation cycle. The resulting observations are incorporated into the Evidence Blackboard by evidence normalization, coverage tracking, evidence linking, and conflicts resolution. Finally, the Response Generator maps the terminal evidence state to the requested output, including dental knowledge, diagnostic category, lesion localization, or restoration assessment.

## B. Global Orchestration

The evidence required for multimodal dental reasoning is inherently task-dependent. Dental knowledge questions rely primarily on textual information, whereas diagnostic classification, lesion localization, and restoration assessment require different combinations of visual and geometric evidence at varying levels of granularity. To accommodate these taskspecific requirements, DentAgent separates task interpretation from evidence acquisition: the Intent Detector first determines what must be answered, and the Orchestrator subsequently determines which evidence must be acquired to support that answer.

At the start of inference, the Intent Detector maps the input case x to a task specification:

$$
\tau = \mathrm { I n t e n t D e t e c t o r } ( x ) ,\tag{1}
$$

where the specification τ defines the task objective, required output format and granularity, and the input modalities relevant to the task. It contains neither a diagnostic prediction nor a candidate response; instead, it remains fixed throughout inference and guides subsequent rounds in identifying evidence targets, acquiring the required evidence, and verifying its sufficiency for the requested output.

The Orchestrator then follows the Identification– Delegation–Management–Verification loop shown in Fig. 1. At each orchestration round t, the Task Identification stage derives the evidence targets that remain unresolved:

$$
\begin{array} { r } { \mathcal { R } ^ { ( t ) } = \mathrm { I d e n t i f y T a r g e t s } \left( \tau , \mathcal { B } ^ { ( t - 1 ) } \right) , } \end{array}\tag{2}
$$

where B denotes the Evidence Blackboard, which manages the shared evidence state across orchestration rounds. Each target R denotes a task-specific evidence requirement, such as the clinical finding, anatomical location, quantitative measurement, structural relationship, or knowledge claim. The evidence required for a target is determined by the requested output. For example, evidence sufficient to support a coarse diagnostic category may not provide the spatial or quantitative detail needed for precise localization or measurement. After each blackboard update, the Orchestrator reassesses all targets, removes those that have been adequately resolved, and retains those that remain incomplete, unsupported, or contested.

Algorithm 1 Evidence-Centric Inference with DentAgent   
Require: Dental case $x ~ = ~ ( q , c , { \mathcal M } ) ;$ specialist set ${ \mathcal { A } } \ =$   
$\{ \mathcal { A } _ { k } \} _ { k = 1 } ^ { N } ;$ ; specialist tool inventories $\bar { \mathcal { T } } = \{ \mathcal { T } _ { k } \} _ { k = 1 } ^ { N } ;$ max  
imum global rounds $T _ { \mathrm { m a x } } ;$ ; maximum tools per activation   
$K _ { \operatorname* { m a x } } = 3$   
Ensure: Final response y   
$\tau \gets$ IntentDetector(x)   
$B ^ { 0 }$ ← InitializeBlackboard()   
for $t = 1$ to $T _ { \mathrm { m a x } }$ do   
R<sup>(t)</sup> ← IdentifyTargets $( \tau , B ^ { ( t - 1 ) } )$   
if $\mathcal { R } ^ { ( t ) } = \emptyset$ then   
$B ^ { ( t ) } \gets B ^ { ( t - 1 ) }$   
break   
end if   
D<sup>(t)</sup> ← DelegateSpecialists $( \mathcal { R } ^ { ( t ) } , B ^ { ( t - 1 ) } , A )$   
$\mathcal { O }  \mathcal { O }$   
for each assignment $( \mathcal { A } _ { k } , \mathcal { R } _ { k } ^ { ( t ) } ) \in \mathcal { D } ^ { ( t ) }$ do   
$\widehat { \mathcal { T } } _ { k }$ ← SelectTools(A , R<sup>(t)</sup> , B<sup>(t−1)</sup>, T<sub>k</sub>, K<sub>max</sub>)   
O ← ExecuteSpecialist $( \varLambda _ { k } , \mathcal { R } _ { k } ^ { ( t ) } , \widehat { \mathcal { T } } _ { k } , x , { B } ^ { ( t - 1 ) } )$   
$\mathcal { O }  \mathcal { O } \cup \{ \mathcal { O } _ { k } \}$   
end for   
$B ^ { ( t ) }$ ← NormalizeEvidence $( \boldsymbol { B } ^ { ( t - 1 ) } , \boldsymbol { \mathcal { O } } )$   
$B ^ { ( t ) }$ ← TrackCoverage $( \boldsymbol { B } ^ { ( t ) } , \mathcal { R } ^ { ( t ) } )$   
$B ^ { ( t ) }$ ← LinkEvidence $( \boldsymbol { B } ^ { ( t ) } )$   
B<sup>(t)</sup> ← ResolveConflicts $( \dot { B ^ { ( t ) } } )$   
if Verify $( \tau , B ^ { ( t ) } )$ then   
break   
end if   
end for   
y ← GenerateResponse $\mathbf \Lambda ^ { \prime } ( \tau , B ^ { ( t ) } )$   
return y

During the Specialist Delegation stage, the Orchestrator assigns the current targets to an appropriate subset of Specialist Sub-Agents:

$$
\mathcal { D } ^ { ( t ) } = \mathrm { D e l e g a t e S p e c i a l i s t s } \left( \mathcal { R } ^ { ( t ) } , \mathcal { B } ^ { ( t - 1 ) } , { \bf A } \right) .\tag{3}
$$

where each assignment $( \mathcal { A } _ { k } , \mathcal { R } _ { k } ) \ \in \ \mathcal { D } ^ { ( t ) }$ associates specialist $\mathcal { A } _ { k }$ with a target subset $\mathcal { R } _ { k } \subseteq \mathcal { R } ^ { ( t ) }$ . Assignments are determined by the required evidence type, available case inputs, and specialist capability boundaries. The Orchestrator may activate a single specialist for a modality-specific target or multiple complementary specialists when evidence from several modalities is required.

After the selected specialists return their current-round outputs, the Orchestrator enters the Evidence Management stage, where the Evidence Blackboard integrates and manages the newly acquired evidence. It then performs Verify $\left( \tau , \bar { B } ^ { ( t ) } \right)$ to determine whether the current evidence state is sufficient for the requested output. If verification succeeds, evidence acquisition terminates. Otherwise, the updated blackboard conditions the next Identify Task stage, which derives a revised target set $\mathcal { R } ^ { ( t + 1 ) }$ <sup>)</sup>, directing subsequent execution toward the remaining evidence gaps and conflicts. Thus, specialist routing adapts to the accumulated evidence, while the task specification τ remains fixed throughout inference.

## C. Specialist Sub-Agent Execution

DentAgent includes five Specialist Sub-Agents, each designed for a specific modality of clinical data. The Text Agent handles clinical text and dental knowledge; the Pano Agent analyzes panoramic radiographs; the Intraoral Agent examines intraoral photographs; the Ceph Agent analyzes cephalometric radiographs; and the IOS Agent analyzes IOS based dental and occlusal structures. These specialists have complementary roles and are activated as needed, without a fixed execution order.

Each specialist $\mathcal { A } _ { k }$ is associated with a corresponding tool inventory $\mathcal { T } _ { k } .$ Given assigned targets $\mathcal { R } _ { k }$ , it selects at most $K _ { \mathrm { m a x } }$ relevant tools:

$$
\begin{array} { r l } & { \widehat { \mathcal { T } } _ { k } = \mathrm { S e l e c t T o o l s } \left( \mathcal { A } _ { k } , \mathcal { R } _ { k } , \mathcal { B } , \mathcal { T } _ { k } , K _ { \operatorname* { m a x } } \right) , } \\ & { \widehat { \mathcal { T } } _ { k } \subseteq \mathcal { T } _ { k } , \qquad \Big | \widehat { \mathcal { T } } _ { k } \Big | \leq K _ { \operatorname* { m a x } } , } \end{array}\tag{4}
$$

where tool selection depends on the assigned targets, available inputs, current blackboard state, and the relevance of each tool.

Then, the specialist executes the selected tools and returns their observations:

$$
\mathcal { O } _ { k } = \mathrm { E x e c u t e S p e c i a l i s t } \left( \mathcal { A } _ { k } , \mathcal { R } _ { k } , \widehat { \mathcal { T } } _ { k } , \boldsymbol { x } , \boldsymbol { B } \right) .\tag{5}
$$

The output $\mathcal { O } _ { k }$ contains findings or measurements for the assigned targets, together with their anatomical location, source, quality, and conditions of validity.

## D. Evidence Blackboard

The Evidence Blackboard B stores and organizes all evidence collected during inference. It serves as the shared state used by the Specialist Sub-Agents, the Orchestrator, and the Response Generator. Evidence from different specialists and reasoning rounds is aligned to the same target while explicitly record their source and applicability conditions.

a) Evidence Normalization.: NormalizeEvidence converts each specialist output into a standardized evidence record. The process aligns clinical terminology, anatomical labels, and levels of detail, allowing evidence from different specialists to be compared directly. Each record specifies the target, finding or measurement, anatomical location, originating specialist, source tool, quality, and limitations. The resulting representation supports subsequent coverage assessment and evidence linking from different sub-agents.

b) Coverage Tracking.: TrackCoverage assesses how well the current evidence addresses each target. It assigns or updates one of four states, including covered, partially covered, conflicting, or uncovered. The required evidence depends on the requested output. For example, coarse findings may be sufficient to confirm that an abnormality is present, but insufficient to determine its exact location, severity, or size.

c) Evidence Linking.: LinkEvidence links evidence records associated with the same target and classifies their relationships as supporting, complementary, redundant, or conflicting. Supporting records provide independent evidence for the same conclusion; complementary records describe different aspects of the target; redundant records repeat an existing observation; and conflicting records report incompatible findings. Linked records are retained separately rather than merged into a single record. This representation preserves source-specific information and indicates whether a conclusion is supported by multiple independent observations or by a single source.

d) Conflict Resolution.: ResolveConflicts evaluates incompatible evidence records according to target relevance, anatomical correspondence, source applicability, observation quality, provenance, and applicability conditions. Resolution does not rely on majority voting. A record may receive greater evidential weight only when it evaluates the target more directly and reliably. All source records remain retained for traceability. When the available evidence does not justify a reliable preference, the disagreement remains explicit. This operation also updates the coverage states of the affected targets before the blackboard is returned to the Orchestrator.

## E. Response Generation

After the orchestration loop terminates, the Response Generator produces the final output from the task specification τ and the final blackboard state B:

$$
y = \mathrm { G e n e r a t e R e s p o n s e } ( \tau , \boldsymbol { B } ) .\tag{6}
$$

The Response Generator does not call additional tools or modify the blackboard. It selects the evidence relevant to the requested targets and presents the result according to τ. For closed-set tasks, the output is restricted to the predefined label set. For structured and open-ended tasks, the response is generated from the evidence records stored in the blackboard. If τ defines uncertainty as a valid output, the Response Generator reports unresolved conflicts or insufficient evidence instead of forcing a definitive answer.

## IV. EXPERIMENTS

## A. Implementation

DentAgent is implemented in LangGraph, with a shared blackboard that manages accumulated evidence and execution status across orchestration rounds. Unless specified, Qwen3.5- 9B [40], served through vLLM [41], is used for both orchestration and final response generation with tailored system prompts. Reasoning mode is enabled during evidence acquisition and disabled during final response generation. The framework includes N = 5 modality-specific specialists and 33 tools. Each specialist is registered through a YAML capability card that specifies its tool inventory, supported modality, target coverage, input–output schema, anatomical scope, and applicability constraints. During each orchestration round, compatible specialists can be executed in parallel, and each specialist invocation may select at most $K _ { \operatorname* { m a x } } = 3$ tools. We set the global execution budget to $T _ { \mathrm { m a x } } = 1 5$ rounds and terminates earlier when verification succeeds or no feasible specialist remains.

## B. Benchmarks

We evaluate DentAgent on four benchmarks spanning bilingual dental textual knowledge, 2D clinical photographs, panoramic radiograph interpretation, and IOS based 3D reasoning. These benchmarks cover closed-set classification, structured prediction, and open-ended clinical question answering. Unless specified, we follow the official evaluation protocols and score only the final response.

a) DentalBench [16]: It comprises 7,332 English and Chinese questions across multiple dental specialties, including 5,408 close-ended and 1,924 open-ended questions. Following the official partition, we report accuracy and BERTScore [42] for close-ended and open-ended questions, respectively.

b) MMOralBench [17]: It evaluates clinical reasoning based on panoramic radiographs. We use the open-ended subset of 578 questions, with each response evaluated against the ground-truth answer by GPT-5-mini [43] and assigned a score in [0, 1].

c) DentVLM Reader-study Benchmark [7]: It contains 3,105 diagnostic questions grounded in panoramic radiographs, intraoral photographs, and lateral cephalometric radiographs. It evaluates 36 types of clinical findings, where multi-class tasks are evaluated by exact-match accuracy, while multi-label tasks are evaluated by hit rate.

d) IOSVQA [18]: It evaluates 3D dental reasoning based on IOS, which includes 1,929 questions covering eight diagnostic tasks. Following the original protocol, the performance is measured by accuracy.

## C. Main Results

a) DentalBench: As shown in Table I, DentAgent achieves the best open-ended result, with a BERTScore of 33.31%, outperforming the strongest dental-adapted baseline by 2.74%. On close-ended questions, DentAgent attains an accuracy of 66.32%, trailing DeepSeek-R1 by 2.12%, while outperforming GPT-4o and all dental-adapted baselines. In contrast to predefined-label selection in close-ended questions, open-ended tasks require models to generate comprehensive responses that explain specialized dental concepts, analyze clinical cases, or address patient needs. Consequently, openended tasks place greater demands on dental knowledge and evidence retrieval and integration, highlighting DentAgent’s ability to coordinate its text agent and evidence management module to produce well-supported answers.

TABLE I  
COMPARISON ON DENTALBENCH ACROSS TEXTUAL CLOSED- AND OPEN-ENDED QUESTION ANSWERING.
<table><tr><td>Model</td><td>close-ended Accuracy (%) ↑</td><td>Open-ended BERTScore (%) ↑</td></tr><tr><td>General-purpose LLMs</td><td></td><td></td></tr><tr><td>DeepSeek-R1</td><td>68.44</td><td>20.71</td></tr><tr><td>DeepSeek-V3</td><td>67.03</td><td>25.15</td></tr><tr><td>GPT-40</td><td>66.05</td><td>27.80</td></tr><tr><td>GPT-4o-mini</td><td>53.16</td><td>28.32</td></tr><tr><td>Qwen2.5-32B</td><td>64.49</td><td>22.44</td></tr><tr><td>Qwen2.5-14B</td><td>58.84</td><td>22.38</td></tr><tr><td>Qwen2.5-7B</td><td>54.23</td><td>22.26</td></tr><tr><td>Dental-adapted LLMs</td><td></td><td></td></tr><tr><td>Qwen2.5-3B + SFT</td><td>50.35</td><td>27.81</td></tr><tr><td>Qwen2.5-3B + RAG</td><td>50.26</td><td>30.54</td></tr><tr><td>Qwen2.5-3B + SFT + RAG</td><td>55.27</td><td>30.57</td></tr><tr><td>DentAgent (ours)</td><td>66.32</td><td>33.31</td></tr></table>

TABLE II

COMPARISON WITH VARIOUS BASELINES ON MMORALBENCH.DENTAGENT IS IMPLEMENTED BASED ON GPT-5.4-MINI. RESPONSESARE SCORED BY GPT-5-MINI AGAINST REFERENCE ANSWERS.

<table><tr><td>Model</td><td>Model-Judge Score (%) ↑</td></tr><tr><td>General-purpose MLLMs</td><td></td></tr><tr><td>GPT-5</td><td>42.42</td></tr><tr><td>GPT-4V</td><td>39.38</td></tr><tr><td>Gemini-2.5-Flash Qwen-Max-VL</td><td>27.84 5.29</td></tr><tr><td>DeepSeek-VL-7B-Chat</td><td>15.95</td></tr><tr><td></td><td>19.74</td></tr><tr><td>GLM-4V-9B-Thinking Qwen2.5-VL-72B</td><td>15.38</td></tr><tr><td>Medical-specific MLLMs</td><td></td></tr><tr><td>LLaVA-Med HealthGPT-XL32</td><td>4.76</td></tr><tr><td>MedVLM-R1</td><td>27.80 24.70</td></tr><tr><td>MedDr</td><td>26.20</td></tr><tr><td>OralGPT-Omni</td><td>45.31</td></tr><tr><td></td><td></td></tr><tr><td>Medical agents MedRAX</td><td>36.73</td></tr><tr><td>MedAgents</td><td>34.71</td></tr><tr><td>MMedAgent</td><td>15.19</td></tr><tr><td>MDAgents</td><td>40.50</td></tr><tr><td>OralAgent</td><td>61.00</td></tr><tr><td>DentAgent (ours)</td><td>61.44</td></tr></table>

b) MMOralBench: As shown in Table II, for question answering based on panoramic radiographs, DentAgent outperforms OralGPT-Omni, the strongest dental-specific MLLM among the evaluated baselines, by 16.13%, while also surpassing all evaluated medical agent frameworks. The substantial advantage of the agentic reasoning framework over standalone MLLMs can be largely attributed to its ability to invoke specialized perception tools. These tools enable fine-grained extraction of pathological findings and anatomical features, effectively addressing the stringent perceptual requirements of dental image analysis. Additionally, although panoramic radiograph interpretation represents only one component of DentAgent’s broader capabilities, DentAgent still achieves better performance than existing medical agents, further demonstrating the effectiveness of our framework for oral and dental reasoning tasks.

TABLE III  
COMPARISON ON THE DENTVLM READER-STUDY BENCHMARK. ACCURACY AND HIT RATE EVALUATE MULTI-CLASS AND MULTI-LABEL DIAGNOSIS TASKS, RESPECTIVELY.
<table><tr><td>Model</td><td>Accuracy(%) ↑</td><td>Hit Rate (%) ↑</td></tr><tr><td>General-purpose MLLMs</td><td></td><td></td></tr><tr><td>Qwen3-30A3</td><td>50.5</td><td>15.1</td></tr><tr><td>Qwen3-235A22</td><td>48.7</td><td>13.7</td></tr><tr><td>Medical-specific MLLMs</td><td></td><td></td></tr><tr><td>Hulu-30A3</td><td>37.6</td><td>8.6</td></tr><tr><td>Hulu-235A22</td><td>28.7</td><td>12.7</td></tr><tr><td>DentVLM</td><td>78.8</td><td>55.2</td></tr><tr><td>Reader Groups</td><td></td><td></td></tr><tr><td>Junior Readers</td><td>69.8</td><td>41.9</td></tr><tr><td>General Practitioners</td><td>77.0</td><td>55.4</td></tr><tr><td>Senior Specialists</td><td>81.4</td><td>59.3</td></tr><tr><td>DentAgent (ours)</td><td>72.6</td><td>76.6</td></tr></table>

c) DentVLM Reader-study Benchmark: In the readerstudy benchmark of DentVLM, we compared DentAgent with general-purpose MLLMs, medical-specific MLLMs, and reported human readers with varying levels of clinical experience, as shown in Table III. DentAgent substantially outperforms all baseline models on the multi-label diagnostic task and even exceeds the performance of senior specialists by 17.3 percentage points. Unlike multi-class classification, which typically focuses on assigning a single label for a specific task, multi-label diagnosis requires a comprehensive assessment of the entire image and the identification of as many coexisting oral conditions as possible. Such findings can be easily overlooked even by experienced clinicians. This is precisely where DentAgent benefits from its evidence-driven, iterative multiround reflection loop, which enables systematic examination and refinement of diagnostic hypotheses. These results also highlight a promising direction for dental AI: rather than replacing clinicians, systems such as DentAgent can serve as adjunctive diagnostic tools by supporting comprehensive image assessment, improving clinical efficiency, and reducing the risk of diagnostic omissions.

d) IOSVQA: As shown in Table IV, DentAgent outperforms the strongest 3D-specific IOSVLM by 6.12% and the best-performing 2D multiview baseline by 14.98%. These substantial gains underscore the value of explicitly grounding reasoning in task-relevant geometric evidence extracted directly from the native 3D mesh. In contrast to 2D multiview approaches, which may lose or distort metric and relational cues during projection, DentAgent operates on the original mesh to quantify inter-tooth spatial relationships, occlusal contacts, and local surface morphology. DentAgent also differs from monolithic 3D models, in which these geometric cues are implicitly compressed into a global representation. Instead, it organizes and aggregates the extracted evidence around the queried tooth or occlusal region. This query-conditioned, target-centric aggregation filters out irrelevant anatomical information while integrating complementary geometric cues, thereby enabling more accurate and reliable answers.

TABLE IV  
COMPARISON ON THE IOSVQA. FOR MLLMS WITHOUT NATIVE 3D INPUT SUPPORT, WE PROJECT EACH IOS INTO MULTIPLE 2D VIEWS.
<table><tr><td>Model</td><td>Input Type</td><td>Accuracy(%)↑</td></tr><tr><td colspan="3">General-purpose MLLMs</td></tr><tr><td>Qwen3VL-8B</td><td>2D multiview</td><td>58.84</td></tr><tr><td>InternVL3.5-8B</td><td>2D multiview</td><td>55.16</td></tr><tr><td colspan="3">Medical-specific MLLMs</td></tr><tr><td>HuluMed-7B</td><td>2D multiview</td><td>63.56</td></tr><tr><td>HuatuoGPT-V-7B</td><td>2D multiview</td><td>59.88</td></tr><tr><td>MedGemma-1.5</td><td>2D multiview</td><td>55.00</td></tr><tr><td colspan="3">3D-specific MLLMs</td></tr><tr><td>PointLLM-7B</td><td>3D IOS mesh</td><td>46.71</td></tr><tr><td>ShapeLLM-7B</td><td>3D IOS mesh</td><td>33.18</td></tr><tr><td>IOSVLM</td><td>3D IOS mesh</td><td>72.42</td></tr><tr><td>DentAgent (ours)</td><td>3D IOS mesh</td><td>78.54</td></tr></table>

![](images/a7a6126fce7c3c6e9137931ba83b6a0eb0aa2e365c811c0f32af9d862383d12a.jpg)  
Fig. 2. Comparison on MMoralBench between direct inference, single-pass agentic execution, and iterative evidence acquisition under Qwen3.5-9B.

## D. Analysis

a) Agentic Orchestration Drives the Improvement: To assess the contributions of different components in DentAgent, we first isolate the effect of agentic orchestration by fixing Qwen3.5-9B as the reasoning backbone across all variants, as shown in Fig. 2. Compared with direct inference, introducing single-pass agentic execution yields a substantial gain of 17.48% even before iterative verification is enabled. This result highlights the importance of coordinated multi-agent collaboration and specialized perception and analysis tools for complex multimodal dental reasoning.

b) Iterative Verification Further Improves Evidence Quality: With the backbone and toolset unchanged, restoring the complete Identify-Delegate-Observe-Verify loop further raises the score from 48.40 to 52.83. Although single-pass execution can acquire informative observations, it cannot adaptively revisit insufficiently examined targets or resolve evidence gaps and conflicts identified during execution. Taken together, agentic tool use produces the dominant improvement, while reflection loop further enhances the completeness and consistency of the supporting evidence.

![](images/41362c2a165c9d4bcae604cdc08a57afaf082f4e966f35a73ece505a2451398e.jpg)  
Fig. 3. Generalization on MMOralBench of the DentAgent workflow under various reasoning backbones.

c) Compatibility Across Reasoning Backbones: We further investigate the effect of various backbone [40], [43], [44] by keeping the DentAgent workflow fixed, as shown in Fig. 3. The framework exhibits steady performance gains as the capability of the underlying reasoning backbone increases. This trend suggests that, as a lightweight and training-free dental reasoning workflow, DentAgent can be readily integrated with different backbones while continuing to benefit from advances in their reasoning capabilities. Such compatibility provides practical flexibility for real-world deployment across diverse performance requirements and computational constraints.

## V. CONCLUSION

We presented DentAgent, an evidence-centric multi-agent framework that separates evidence acquisition from response generation and coordinates five specialist agents through an Orchestrator and an Evidence Blackboard that preserves source-attributed findings throughout inference. Across four benchmarks, DentAgent demonstrates strong performance across diverse dental modalities and tasks. Future work will extend DentAgent to additional modalities, such as histopathology and CBCT, explore cost-efficient training strategies, and prospectively evaluate its value for clinical support.

## REFERENCES

[1] World Health Organization, “Oral health,” https://www.who.int/newsroom/fact-sheets/detail/oral-health, 2025, accessed: 2026-07-08.

[2] M. A. Peres, L. M. D. Macpherson, R. J. Weyant, B. Daly, R. Venturelli, M. R. Mathur, S. Listl, R. K. Celeste, C. C. Guarnizo-Herreno, C. Kearns, H. Benzian, P. Allison, and R. G. Watt, “Oral diseases: A global public health challenge,” The Lancet, 2019.

[3] R. G. Watt, B. Daly, P. Allison, L. M. D. Macpherson, R. Venturelli, S. Listl, R. J. Weyant, M. R. Mathur, C. C. Guarnizo-Herreno, R. K. Celeste, M. A. Peres, C. Kearns, and H. Benzian, “Ending the neglect of global oral health: Time for radical action,” The Lancet, 2019.

[4] F. Schwendicke, W. Samek, and J. Krois, “Artificial intelligence in dentistry: Chances and challenges,” Journal of Dental Research, 2020.

[5] S. AbuSalim, N. Zakaria, M. R. Islam, G. Kumar, N. Mokhtar, and S. J. Abdulkadir, “Analysis of deep learning techniques for dental informatics: a systematic literature review,” in Healthcare, vol. 10, no. 10. MDPI, 2022, p. 1892.

[6] H. Huang, O. Zheng, D. Wang, J. Yin, Z. Wang, S. Ding, H. Yin, C. Xu, R. Yang, Q. Zheng et al., “Chatgpt for shaping the future of dentistry: the potential of multi-modal large language model,” International Journal of Oral Science, vol. 15, no. 1, p. 29, 2023.

[7] Z. Meng, J. Hao, X. Dai, Y. Feng, J. Liu, B. Feng, H. Wu, X. Gai, H. Zhu, T. Hu et al., “Dentvlm: A multimodal vision-language model for comprehensive dental diagnosis and enhanced clinical practice,” arXiv preprint arXiv:2509.23344, 2025.

[8] X. Huang, F. Xiao, D. He, A. Gao, D. Li, X. Zhang, S. Zhang, and X. Wang, “Towards generalist intelligence in dentistry: Vision foundation models for oral and maxillofacial radiology,” arXiv preprint arXiv:2510.14532, 2025.

[9] Q. Zhu, Y. Lin, W. Fu, W. Tang, J. Li, Y. Zhang, B. Li, X. Guo, F. Wang, H. Qi, C. Sun, X. Zhu, Z. Liu, L. He, Z. Zheng, B. Du, J. Yang, Z. Bian, and L. Meng, “Towards clinical-level interpretation of dental panoramic radiography using an instance-guided vision-language model,” Nature Biomedical Engineering, 2026.

[10] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” in International Conference on Learning Representations, 2023.

[11] T. Schick, J. Dwivedi-Yu, R. Dess\`ı, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language models can teach themselves to use tools,” Advances in neural information processing systems, vol. 36, pp. 68 539–68 551, 2023.

[12] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang, “Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face,” in Advances in Neural Information Processing Systems, 2023.

[13] J. Hao, S. Dai, Y. Zhang, Y. Liang, J. Wu, J. Bao, Y. Fan, Z. Ye, Y. Sun, X. Zhang et al., “Oralagent: Integrating reasoning, tools, and knowledge for interactive dental image analysis,” arXiv preprint arXiv:2605.27378, 2026.

[14] Z. Yu, L. Yang, B. Babicka, M. Hu, J. Hao, A. Huang, J. Huang, Y. Jin, J. Wu, and Z. Ge, “Opgagent: An agent for auditable dental panoramic x-ray interpretation,” arXiv preprint arXiv:2603.00462, 2026.

[15] J. Hao, X. Dai, Z. Meng, B. Yuan, T. Jiao, B. Feng, H. Wu, Y. Feng, D. Jing, J. Zhao, J. T. Zhou, B. Fang, Z. Liu, and L. Xia, “Orthoagent: A knowledge-enhanced multi-agent framework for multimodal orthodontic diagnosis and treatment planning,” Dental Research, vol. 1, no. 3, p. 100039, 2026.

[16] H. Zhu, Y. Xu, Y. Li, Z. Meng, and Z. Liu, “Dentalbench: benchmarking and advancing llms capability for bilingual dentistry understanding,” arXiv preprint arXiv:2508.20416, 2025.

[17] J. Hao, Y. Fan, Y. Sun, K. Guo, L. Lizhuo, J. Yang, Q. Ai, L. Wong, H. Tang, and K. Hung, “Towards better dental ai: A multimodal benchmark and instruction dataset for panoramic x-ray analysis,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[18] H. Xiong, Z. Meng, T. Hu, C. Zhou, Y. Feng, and Z. Liu, “Iosvlm: A 3d vision-language model for unified dental diagnosis from intraoral scans,” arXiv preprint arXiv:2603.16781, 2026.

[19] J.-H. Lee, D.-H. Kim, S.-N. Jeong, and S.-H. Choi, “Detection and diagnosis of dental caries using a deep learning-based convolutional neural network algorithm,” Journal of Dentistry, 2018.

[20] A. G. Cantu, S. Gehrung, J. Krois, A. Chaurasia, J. G. Rossi, R. Gaudin, K. Elhennawy, and F. Schwendicke, “Detecting caries lesions of different radiographic extension on bitewings using deep learning,” Journal of Dentistry, 2020.

[21] J. Krois, T. Ekert, L. Meinhold, T. Golla, B. Kharbot, A. Wittemeier, C. Dorfer, and F. Schwendicke, “Deep learning for the radiographic detection of periodontal bone loss,” Scientific Reports, 2019.

[22] C.-T. Lee, T. Kabir, J. Nelson, S. Sheng, H.-W. Meng, T. E. Van Dyke, M. F. Walji, X. Jiang, and S. Shams, “Use of the deep learning approach to measure alveolar bone level,” Journal of clinical periodontology, vol. 49, no. 3, pp. 260–269, 2022.

[23] D. V. Tuzoff, L. N. Tuzova, M. M. Bornstein, A. S. Krasnov, M. A. Kharchenko, S. I. Nikolenko, M. M. Sveshnikov, and G. B. Bednenko, “Tooth detection and numbering in panoramic radiographs using convolutional neural networks,” Dentomaxillofacial Radiology, vol. 48, no. 4, p. 20180051, 2019.

[24] G. Jader, J. Fontineli, M. Ruiz, K. Abdalla, M. Pithon, and L. Oliveira, “Deep instance segmentation of teeth in panoramic x-ray images,” in 2018 31st SIBGRAPI Conference on Graphics, Patterns and Images, 2018.

[25] C.-W. Wang, C.-T. Huang, J.-H. Lee, C.-H. Li, S.-W. Chang, M.-J. Siao, T.-M. Lai, B. Ibragimov, T. Vrtovec, O. Ronneberger, P. Fischer,

T. F. Cootes, and C. Lindner, “A benchmark for comparison of dental radiography analysis algorithms,” Medical Image Analysis, 2016.

[26] C. Lindner, C.-W. Wang, C.-T. Huang, C.-H. Li, S.-W. Chang, and T. F. Cootes, “Fully automatic system for accurate localisation and analysis of cephalometric landmarks in lateral cephalograms,” Scientific Reports, 2016.

[27] Z. Cui, C. Li, and W. Wang, “Toothnet: Automatic tooth instance segmentation and identification from cone beam ct images,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[28] Z. Cui, Y. Fang, L. Mei, B. Zhang, B. Yu, J. Liu, C. Jiang, Y. Sun, L. Ma, J. Huang, Y. Liu, Y. Zhao, C. Lian, Z. Ding, M. Zhu, and D. Shen, “A fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images,” Nature Communications, vol. 13, no. 1, p. 2096, 2022.

[29] C. Lian, L. Wang, T.-H. Wu, M. Liu, F. Duran, C.-C. Ko, and D. Shen, “Meshsnet: Deep multi-scale mesh feature learning for end-to-end tooth labeling on 3d dental surfaces,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2019, 2019.

[30] C. Lian, L. Wang, T.-H. Wu, F. Wang, P.-T. Yap, C.-C. Ko, and D. Shen, “Deep multi-scale mesh feature learning for automated labeling of raw dental surfaces from 3d intraoral scanners,” IEEE Transactions on Medical Imaging, 2020.

[31] Z. Cui, C. Li, N. Chen, G. Wei, R. Chen, Y. Zhou, D. Shen, and W. Wang, “Tsegnet: An efficient and accurate tooth segmentation network on 3d dental model,” Medical Image Analysis, 2021.

[32] Z. Shi, Z. Meng, R. Chen, Y. Feng, Z. Zhao, J. Hao, B. Fang, Z. Liu, and Y. Zheng, “Leta: Tooth alignment prediction based on dual-branch latent encoding,” IEEE Transactions on Visualization & Computer Graphics, vol. 31, no. 09, pp. 4805–4820, 2025.

[33] Q. Wu, F. Niu, H. Zhu, Y. Sun, Y. Shen, X. Li, H. Wu, L. Liu, Z. Pan, Z. Liu et al., “Bridging the skill gap in clinical cbct interpretation with cbctrepd,” arXiv preprint arXiv:2603.10933, 2026.

[34] F. Pethani and A. G. Dunn, “Natural language processing for clinical notes in dentistry: A systematic review,” Journal of Biomedical Informatics, 2023.

[35] M. Buttner, U. Leser, L. Schneider, and F. Schwendicke, “Natural¨ language processing: Chances and challenges in dentistry,” Journal of Dentistry, 2024.

[36] J. Hao, Y. Liang, L. Lin, Y. Fan, W. Zhou, K. Guo, Z. Ye, Y. Sun, X. Zhang, Y. Yang, Q. Li, H. Tang, J. K.-H. Tsoi, L. Shen, and K. F. Hung, “OralGPT-Omni: A versatile dental multimodal large language model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 38 509–38 519.

[37] Z. Cai, J. Zhang, J. Zhao, Z. Zeng, Y. Li, L. Jingyi, J. Chen, Y. Yang, J. You, S. Deng et al., “Dentalgpt: Incentivizing multimodal reasoning in dentistry,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 2811–2829.

[38] B. Zhang, Y. Miao, T. Wu, T. Chen, J. Jiang, Z. Li, Z. Tang, L. Yu, and J. Su, “Archmap: Arch-flattening and knowledge-guided vision language model for tooth counting and structured dental understanding,” arXiv preprint arXiv:2511.14336, 2025.

[39] Y. Fan, J. Hao, H. Chen, J. Bao, Y. Shao, Y. Liang, K. F. Hung, and H. Tang, “Oralgpt-plus: Learning to use visual tools via reinforcement learning for panoramic x-ray analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 35 373–35 383.

[40] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.5

[41] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[42] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “BERTScore: Evaluating text generation with BERT,” in International Conference on Learning Representations, 2020.

[43] A. Singh, A. Fry, A. Perelman, A. Tart, A. Ganesh, A. El-Kishky, A. McLaughlin, A. Low, A. Ostrow, A. Ananthram et al., “Openai gpt-5 system card,” arXiv preprint arXiv:2601.03267, 2025.

[44] Qwen Team, “Qwen3.6-27B: Flagship-level coding in a 27B dense model,” April 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.6-27b
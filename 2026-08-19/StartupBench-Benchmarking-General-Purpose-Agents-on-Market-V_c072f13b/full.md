# StartupBench: Benchmarking General-Purpose Agents on Market-Validated End-to-End Workflows

<sup>1</sup>ByteDance Seed, <sup>2</sup>Nanjing University, <sup>3</sup>M-A-P, <sup>4</sup>TokenWave.AI

Full author list in Contributions

## Abstract

Recent advances in Large Language Models(LLMs) and agents have substantially improved the ability of AI systems to execute complex tasks. Yet existing benchmarks largely rely on researcherselected tasks, leaving uncertain whether such progress extends to the work that real-world users actually demand from AI systems. We introduce StartupBench, an E2E agent benchmark grounded in market-validated AI startup products. Rather than defining tasks from pre-defined assumptions about useful agent capabilities, we systematically study AI products with demonstrated adoption, together with their product workflows and users, to identify real-world tasks for which AI has established practical demand across diverse professional domains. We translate these workflows into complete deliverable-oriented tasks and evaluate them with fine-grained rubrics capturing their complex requirements. Across representative models evaluated under a unified agent harness, even the strongest model successfully completes only approximately 30% of StartupBench, despite making substantial partial progress on many tasks. Further analysis identifies aspects like complex instruction following and domain-specific expertise as major sources of failure. Our results reveal that many market-validated workflows remain beyond the reliable capabilities of current general purpose agents, establishing StartupBench as an empirical measure of progress toward E2E completions of real-world user tasks.

Project Page: https://startupbench.github.io/ Date: August 19, 2026

## 1 Introduction

Large language models (LLMs) are increasingly evolving from systems that primarily retrieve information or answer questions[5, 12, 26] into agents capable of completing complex real-world tasks. Current AI systems are expected to execute end-to-end (E2E) workflows, coordinate multiple capabilities, and produce deliverables that satisfy practical requirements under realistic constraints. As AI becomes increasingly integrated into professional work, autonomously completing user-delegated tasks has become an important measure of model capability. Evaluating whether AI systems can reliably translate their capabilities into professionally usable deliverables is therefore a central challenge for the next generation of benchmarks.

Recent benchmarks have made important progress toward evaluating E2E task execution under realistic settings. Nevertheless, several important limitations remain. First, many benchmark tasks are still largely defined from researchers’ perspectives rather than grounded in workflows that have demonstrated practical demand through real-world AI adoption [12, 18, 19], limiting their ability to reflect authentic user needs and deliverable requirements. Second, although several benchmarks evaluate complete deliverables [23, 24], their evaluation protocols are often substantially coarser than the deliverables themselves. Real-world work products typically involve numerous functional, structural, formatting, and domain-specific requirements that holistic evaluations cannot faithfully assess. Consequently, existing benchmarks still provide only limited evidence of how well current models and agents can satisfy complex real-world user requirements and reliably produce professionally usable deliverables.

![](images/2355389d3f011bbe098a37f46adbac66cb0034cac898dc00e919769fbd010ad1.jpg)  
Figure 1 Overview of StartupBench: real AI-product workflows, long-horizon agent execution, six-domain coverage, and multi-format deliverable generation.

To address these limitations, we observe that AI-native startups ofer a natural source of real-world AI workflows: their products have been validated through real-world adoption and commercial demand, providing direct evidence that users value and seek to delegate these tasks to AI. We therefore introduce StartupBench, a survey- and interview-driven benchmark that draws on these workflows to capture realistic E2E user requirements across domains. StartupBench spans 97 tasks across 6 primary domains and is built around the following design principles:

• Adoption-grounded task sourcing. We systematically survey AI-native startups and their products, complemented by product demonstrations and user interviews, to identify workflows with demonstrated real-world demand. We then construct these workflows into benchmark tasks that preserve their core user objectives and practical requirements.

• E2E deliverables. Each task requires producing the complete deliverables that users ultimately expect in realistic workflows, rather than intermediate outputs or simplified demonstrations.

• Fine-grained rubric-based evaluation. Each StartupBench task is evaluated using multiple independent rubrics covering distinct dimensions of the expected deliverable, including functionality, structure, formatting, and domain-specific quality, ensuring a comprehensive and faithful assessment of real-world task completion.

We evaluate representative models on StartupBench under unified agentic harness. Results show that even the strongest model successfully completes only about 30% of benchmark tasks, indicating that current general-purpose systems remain far from reliably producing professionally usable deliverables. Performance varies substantially across domains, with Finance, STEM & Computer Science and Education & Humanities proving particularly challenging. Deeper analysis further attributes these failures primarily to limitations in complex instruction following and domain-specific expertise, highlighting promising directions for future foundation models and agent systems.

To summarize, our contributions are as follows:

• We propose a survey- and interview-driven methodology for constructing agent benchmarks from real AI adoption scenarios.

• We construct StartupBench, a benchmark built through AI-native startup research, enterprise-user interviews, user-oriented task abstraction, expert data construction, and multi-stage quality control.

• We present a comprehensive empirical study of current foundation models and general-purpose agents on startup-derived workflows, providing a practical evaluation signal for assessing whether AI agents are progressing from answering questions toward reliably completing real work.

## 2 Related Work

## 2.1 Commercial AI agents and Vertical Workflows

Recent advances in foundation models have enabled AI agents to gradually acquire planning, tool-use, fileprocessing, and multi-step execution capabilities [21, 30], pushing AI systems from general conversational assistants toward workflow-oriented products. In practice, commercial AI products increasingly package these capabilities into vertical workflows such as document processing, data analysis, financial research, enterprise operations, and knowledge work. This trend suggests that real-world AI value is not determined only by isolated model capabilities, but also by whether these capabilities can be productized into useful work products. However, existing academic benchmarks rarely use commercial adoption itself as a source of task discovery. StartupBench addresses this gap by deriving tasks from market-validated AI-native startups and their deep users, moving task construction from researcher-defined settings toward demand-driven workflow construction.

## 2.2 End-to-end evaluation of AI agents

Agent benchmarks have evolved from capability evaluation toward end-to-end task completion. Early benchmarks mainly focus on tool use, multi-step reasoning, and general assistant abilities, as in AgentBench and GAIA [10, 12]. Later work extends evaluation to more realistic interactive environments, including web interaction and computer-use tasks [27, 34]. More recent benchmarks further move toward professional work and economically meaningful tasks, covering enterprise software, simulated workplaces, occupational tasks, and long-horizon workflows [4, 7, 11, 18, 28, 35]. These works reflect a clear transition from isolated capability tests to realistic task-completion evaluation. StartupBench follows this direction, but difers in its task source: rather than relying primarily on public environments, occupational taxonomies, or simulated workplaces, it derives tasks from AI-native startup products, ToB user needs, and expected business deliverables.

## 2.3 Agent-as-a-Judge for complex deliverable evaluation

As agent tasks shift from short answers to heterogeneous deliverables such as reports, spreadsheets, slides, code patches, and structured files, evaluation must move from capability-centric scoring to deliverable-centric assessment[31]. LLM-as-a-Judge has been widely used for open-ended response evaluation, with MT-Bench and Prometheus studying preference-based or rubric-conditioned judging [8, 33]. For agentic workflows, however, judging often requires evidence retrieval, file inspection, state verification, and constraint checking over final artifacts. Agent-as-a-Judge extends LLM judging by equipping evaluators with tools and environment interaction, while AJ-Bench further studies judge agents on information acquisition, state verification, and process verification [22, 36]. StartupBench adopts this deliverable-centric evaluation paradigm by combining expert rubrics with automated judge agents that inspect original artifacts and evidence views, evaluating whether agents can transform model capabilities into usable work products.

## 3 StartupBench

StartupBench is designed to evaluate whether models and agents can complete realistic workflows that users already delegate to AI products. This objective requires data that reflects authentic user demand while supporting controlled evaluation. We therefore require each task to be realistic, answerable, evaluable, and discriminative: it should originate from actual AI product usage, provide suficient information for a valid solution, specify clear output requirements and assessment criteria, and remain challenging enough to distinguish current models and agents.

<table><tr><td>Benchmark</td><td>Task Source</td><td>Final Output</td><td>E2E</td><td>Item-wise Eval.</td><td>Process Logic</td><td>Open- ended</td><td>Evaluation Method</td></tr><tr><td>GAIA [12]</td><td>Manual</td><td>Text</td><td>△</td><td>x</td><td>x</td><td>x</td><td>Rule</td></tr><tr><td>OSWorld [27]</td><td>Real+Manual</td><td>Env State</td><td>△</td><td>x</td><td>x</td><td>x</td><td>Execution</td></tr><tr><td>DAComp [9]</td><td>Real+Manual</td><td>Workspace</td><td>√</td><td>x</td><td>x</td><td>√</td><td>Hybrid</td></tr><tr><td>GDPval [18]</td><td>Manual</td><td>Work products</td><td>√</td><td>x</td><td>x</td><td>△</td><td>Manual</td></tr><tr><td>Workspace-Bench [24]</td><td>Sim.+Manual</td><td>Workspace</td><td>√</td><td>x</td><td>x</td><td>√</td><td>Agent-as-Judge</td></tr><tr><td>OneMillionBench [29]</td><td>Expert-authored</td><td>Text</td><td>△</td><td>x</td><td>△</td><td>√</td><td>LLM-judge</td></tr><tr><td>Agents&#x27; Last Exam [23]</td><td>Occupation-guided</td><td>Workspace</td><td>√</td><td>x</td><td>△</td><td>△</td><td>Hybrid</td></tr><tr><td>OfficeQA Pro [17]</td><td>Corpus-grounded</td><td>Text</td><td>x</td><td>x</td><td>x</td><td>x</td><td>Answer match</td></tr><tr><td>StartupBench (Ours)</td><td>Market-vetted + Expert-built</td><td>Workspace</td><td>√</td><td>√</td><td>√</td><td>√</td><td>Agent-as-Judge</td></tr></table>

Table 1 Comparison of StartupBench with existing benchmarks. E2E denotes end-to-end workflow completion; Item-wise Eval. denotes criterion-level judging. ✓, ✗, and △indicate full, no, and partial support.

![](images/b614e39985881c44889e72d2ac8a0b914818bc3d930d2f764179d844dd252f9c.jpg)  
Figure 2 StartupBench data construction pipeline. We select market-validated startup agents, collect real user workflows, construct task artifacts and instances, and calibrate quality and dificulty through multi-stage review.

Neither manual task design nor direct extraction from product demonstrations is suficient to obtain such data. The former is controllable but may be detached from real demand, while the latter is realistic but often lacks context, success criteria, or reproducible specifications. We therefore adopt a survey- and interview-driven multi-stage pipeline shown in Figure 2. We first survey market-validated AI-native startups, interview deep users to identify usage contexts, task goals, and expected deliverables, recruit domain experts to convert validated scenarios into benchmark instances, and finally apply quality control for realism, answerability, evaluability, and discriminative dificulty.

## 3.1 Dataset Construction

## 3.1.1 Startup Survey for Market-Validated Agent Scenarios

We first survey AI-native startups that build agent products across diverse domains. To identify workflows with market validation, we retain startups with more than USD 1M in funding and require evidence of real adoption, either through paid usage or substantial user traction. Funding provides a coarse signal of market expectation, while payment or user-scale evidence indicates that the product is used beyond exploratory

demonstrations. The output of this stage is a pool of candidate products and workflows for user interviews.   
During this stage we collect 20+ start-up agents as the basis for subsequent stage.

## 3.1.2 User Interviews for Scenario and Requirement Discovery

For each candidate Startup-product, we interview deep users of the corresponding agent from diferent domains to recover practical usage beyond public product descriptions. The interviews elicit the usage context, user goal, input information, expected deliverable, success criteria, and common constraints. We summarize these findings into scenario specifications and representative demo cases, which serve as anchors for later annotation stage. During this stage we interview over 30 deep users and collected at least one demo case for each candidate Agent.

## 3.1.3 Expert Construction of Domain Tasks

Given the interview-derived workflow specifications and representative user scenarios as a reference or seed task, we recruit over 50 domain experts to reconstruct benchmark tasks that preserve the original workflow objectives, practical constraints, and expected deliverables while adapting them into reproducible evaluation instances. Rather than designing synthetic problems, experts standardize authentic user requests into benchmark tasks that faithfully represent real professional workflows.

Every task is required to satisfy three principles: it must originate from a realistic work request that users would naturally delegate to AI agents, involve an end-to-end workflow rather than an isolated subtask, and produce concrete professional deliverables with clearly defined success criteria and fine-grained evaluation rubrics for objective evaluation.

To standardize these tasks for evaluation, each task is represented as a triple

$$
{ \mathcal { T } } = ( q , \mathcal { E } , \mathcal { R } ) ,
$$

where q is a natural-language user request describing the task, E denotes the workspace containing all input files and resources required to complete the task, and $\mathcal { R } = \{ ( p _ { i } , w _ { i } ) \} _ { i = 1 } ^ { n }$ is a set of weighted evaluation rubrics that assess complementary aspects of deliverable quality. Detailed task specifications and annotation guidelines are provided in Appendix A.1. During this stage, we collect over 150 candidate tasks for further quality control.

## 3.1.4 Quality Control

To ensure benchmark quality and consistency, every task undergoes the following multi-stage quality control process:

Expert Cross Validation. Every task is independently reviewed by at least one additional domain expert. Reviewers verify four aspects of task quality: (1) the authenticity and correctness of the task description and workspace, (2) the fidelity of the reconstructed workflow, (3) the validity of the evaluation rubrics, and (4) the consistency between the reference deliverable and the finalized evaluation protocol. Detailed review criteria are provided in Appendix A.2.

Difficulty Control.We calibrate task dificulty through pilot executions using several frontier models under the same evaluation harness adopted in the benchmark<sup>1</sup>. Tasks that are consistently completed with high quality are excluded due to limited discriminative value. Pilot results also serve as an additional quality assurance stage: when failures are attributed to ambiguous instructions, incomplete workspace information, or inadequately specified evaluation criteria rather than intrinsic task dificulty, the corresponding task is revised and re-validated before inclusion.

Table 2 Statistics of StartupBench. The benchmark contains 97 real-world workflow tasks across six top-level domains and requires diverse deliverable formats.
<table><tr><td>Category</td><td>Count</td><td>Percentage (%)</td></tr><tr><td>Overall statistics</td><td></td><td></td></tr><tr><td>Total tasks</td><td>97</td><td>100.0</td></tr><tr><td>Top-level domains</td><td>6</td><td>一</td></tr><tr><td>Domain distribution</td><td></td><td></td></tr><tr><td>Medical &amp; HealthCare</td><td>21</td><td>21.6</td></tr><tr><td>Finance</td><td>18</td><td>18.6</td></tr><tr><td>Legal</td><td>16</td><td>16.5</td></tr><tr><td>Business &amp; Management</td><td>19</td><td>19.6</td></tr><tr><td>STEM &amp; Computer Science</td><td>16</td><td>16.5</td></tr><tr><td>Education &amp; Humanities</td><td>7</td><td>7.2</td></tr></table>

Output formats: DOCX, XLSX, PPTX, PDF, Markdown, images, and text-based deliverables.

## 3.2 Statistics for StartupBench

StartupBench contains 97 real-world workflow tasks across 6 top-level domains: Medical & HealthCare, Finance, Legal, Business & Management, STEM & Computer Science, and Education & Humanities. These tasks are designed to reflect deliverable-oriented workflows in realistic ofice settings, requiring agents to understand task contexts, follow complex constraints, process heterogeneous files, and produce usable work products. For evaluation, each task is evaluated using an average of 25.3 fine-grained rubrics, spanning 6 dimensions and 3 importance levels , enabling detailed assessment of complex, long-horizon workflows and diverse work deliverables. The summary of the domain coverage and main output formats of StartupBench is shown in Table 2.

## 3.3 Automatic Evaluation of StartupBench Tasks

As stated above, every task in StartupBench is defined as a triple $\mathcal { T } = ( q , \mathcal { E } , \mathcal { R } )$ . Given $( q , \mathcal { E } )$ , the evaluated model produces a set of deliverables D. Before evaluation, we construct a lightweight evidence view $\mathcal { V } ( \mathcal { D } )$ consisting of extracted textual content and rendered page images, to facilitate navigation over heterogeneous artifacts while allowing the judge to access the original files whenever necessary.

Rather than relying on a single holistic judgment, StartupBench evaluates each rubric item independently through a dedicated AgentJudge session. Most tasks contain 20+ fine-grained rubric items covering diverse aspects of the expected deliverables, including functionality, completeness, formatting, and domain-specific requirements. Evaluating these criteria separately allows each rubric to receive dedicated reasoning and tool usage without interference from unrelated evaluation dimensions, while ensuring that every requirement is assessed explicitly. The final task score is then obtained by aggregating the outcomes of all rubric-level evaluations.

Following Agent-as-a-Judge, each rubric evaluation is carried out by a lightweight agent rather than a single forward pass. Let J denote the judging prompt and evaluation protocol used by the AgentJudge, given the task specification, deliverables, evidence view, and the target rubric item $p _ { i } ,$ the judge produces both a binary decision $y _ { i } \in \{ 0 , 1 \}$ indicating whether the rubric is satisfied and a textual justification $e _ { i }$

$$
\begin{array} { r } { ( y _ { i } , e _ { i } ) = \mathrm { A g e n t J u d g e } \big ( J , q , \mathcal { D } , \mathcal { V } ( \mathcal { D } ) , p _ { i } \big ) , } \end{array}\tag{1}
$$

The final task score is computed by aggregating the weighted decisions across all rubric items:

![](images/e2e8a746fe5021ef9be8c5a38163f4c7fb8f70bc54936556025d7e9f3395dad7.jpg)  
Figure 3 Leaderboard of StartupBench.

$$
\operatorname { s c o r e } ( { \mathcal { D } } , { \mathcal { T } } ) = { \frac { \sum _ { i = 1 } ^ { n } w _ { i } y _ { i } } { \sum _ { i = 1 } ^ { n } w _ { i } } } \in [ 0 , 1 ] .\tag{2}
$$

## 4 Experiments

## 4.1 Experimental Setup

We evaluate 9 representative models on StartupBench, spanning both closed- and open-source families:

• Closed-source: GPT-5.6-sol [15], GPT-5.5 [16], Gemini-3.1-Pro [6], Seed-2.1-Pro [2],Qwen-3.6-Max [20]

• Open-source: DeepSeek-V4-Pro [3],Kimi-K3 [25], Kimi-K2.6 [13], GLM-5.1 [32].

To improve the reliability of our results, we conduct 3 independent runs for each model and report 95% bootstrap confidence intervals for the task scores, computed using 10,000 resamples. All models are executed under the same Nanobot harness with an identical tool configuration, and every task is capped at a maximum of 200 interaction steps. For each task, the agent is initialized with the task-specific workspace and input query, and its deliverables are collected from the designated output directory. Following the evaluation protocol described in the previous section, the deliverables are evaluated by the judge agent against the task specification, workspace context, and rubric set.

For automatic evaluation, the judge agent also runs under the same Nanobot harness to ensure a consistent execution environment, while using GPT-5.5 as the underlying judge model. Apart from the continuous task score, we also report success rate, the fraction of runs that are successfully completed, pooled over the three runs. A task is considered successful if its final score is at least 90; otherwise, it is counted as a failure.

## 4.2 Main Results

Startup tasks remain far from saturated under a general-purpose agent harness. Although the strongest models, Kimi-K3 and GPT-5.6-sol, achieve average scores of 73.67% and 73.61%, respectively, no model successfully completes even one third of the benchmark under the strict acceptance criterion (score ≥ 90). More broadly, the average scores of most models lie between roughly 55 and 75, indicating that current agents are generally capable of making substantial progress toward solving realistic professional workflows. However, high average scores do not necessarily translate into high task completion rates. This is particularly evident for the Kimi series: Kimi-K3 achieves the highest average score overall but a lower success rate than GPT-5.6-sol, while Kimi-K2.6 similarly attains a higher average score than Qwen-3.6-Max but a lower success rate. These discrepancies suggest that some models can satisfy a substantial fraction of task requirements while still falling short of complete delivery. More generally, all models exhibit substantially lower success rate than average score, suggesting that the primary bottleneck is no longer executing large portions of a workflow, but consistently producing artifacts that satisfy the standards required for direct professional use.

Table 3 Model performance of StartupBench, averaged over three independent runs. Top block: importance-weighted mean per-task score (0–100); bottom block: success rate, the fraction of samples scoring ≥ 90 (%), pooled over the three runs. Per-column best in bold, second-best underlined.
<table><tr><td>Model</td><td>Overall</td><td>Med</td><td>Finance</td><td>Legal</td><td>Business</td><td>STEM</td><td>Education</td></tr><tr><td colspan="8">Average score</td></tr><tr><td>Kimi-K3</td><td>73.67</td><td>80.03</td><td>62.38</td><td>69.45</td><td>83.90</td><td>68.34</td><td>77.71</td></tr><tr><td>GPT-5.6-sol</td><td>73.61</td><td>79.32</td><td>61.66</td><td>73.53</td><td>77.61</td><td>74.77</td><td>73.94</td></tr><tr><td>GPT-5.5</td><td>72.79</td><td>78.59</td><td>62.09</td><td>71.75</td><td>79.59</td><td>72.14</td><td>68.32</td></tr><tr><td>Seed-2.1-Pro</td><td>67.19</td><td>73.15</td><td>57.26</td><td>61.35</td><td>77.83</td><td>65.61</td><td>62.91</td></tr><tr><td>Kimi-K2.6</td><td>59.95</td><td>67.70</td><td>51.17</td><td>54.62</td><td>71.91</td><td>51.38</td><td>58.60</td></tr><tr><td>GLM-5.1</td><td>60.79</td><td>65.66</td><td>51.22</td><td>50.20</td><td>78.53</td><td>52.18</td><td>66.50</td></tr><tr><td>Qwen-3.6-Max</td><td>59.46</td><td>66.51</td><td>52.45</td><td>50.33</td><td>75.03</td><td>48.83</td><td>59.26</td></tr><tr><td>DeepSeek-V4-Pro</td><td>61.11</td><td>69.42</td><td>51.11</td><td>54.53</td><td>73.69</td><td>54.89</td><td>57.03</td></tr><tr><td>Gemini-3.1-Pro</td><td>49.73</td><td>54.05</td><td>41.00</td><td>43.49</td><td>61.18</td><td>45.88</td><td>51.23</td></tr><tr><td colspan="8">Success rate (≥ 90)</td></tr><tr><td>Kimi-K3</td><td>29.55</td><td>34.92</td><td>20.37</td><td>20.83</td><td>52.63</td><td>16.67</td><td>23.81</td></tr><tr><td>GPT-5.6-sol</td><td>31.27</td><td>33.33</td><td>27.78</td><td>22.92</td><td>38.60</td><td>33.33</td><td>28.57</td></tr><tr><td>GPT-5.5</td><td>26.80</td><td>30.16</td><td>22.22</td><td>25.00</td><td>38.60</td><td>22.92</td><td>9.52</td></tr><tr><td>Seed-2.1-Pro</td><td>22.34</td><td>19.05</td><td>16.67</td><td>14.58</td><td>45.61</td><td>20.83</td><td>4.76</td></tr><tr><td>Kimi-K2.6</td><td>13.06</td><td>15.87</td><td>14.81</td><td>6.25</td><td>24.56</td><td>4.17</td><td>4.76</td></tr><tr><td>GLM-5.1</td><td>16.49</td><td>12.70</td><td>16.67</td><td>8.33</td><td>36.84</td><td>8.33</td><td>9.52</td></tr><tr><td>Qwen-3.6-Max</td><td>15.12</td><td>11.11</td><td>16.67</td><td>10.42</td><td>33.33</td><td>6.25</td><td>4.76</td></tr><tr><td>DeepSeek-V4-Pro</td><td>16.49</td><td>20.63</td><td>16.67</td><td>8.33</td><td>29.82</td><td>10.42</td><td>0.00</td></tr><tr><td>Gemini-3.1-Pro</td><td>6.53</td><td>6.35</td><td>11.11</td><td>6.25</td><td>8.77</td><td>0.00</td><td>4.76</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

This gap has two important implications. First, it helps explain why specialized vertical agents continue to provide practical value in many market-validated workflows, where consistently meeting professional standards remains essential. Second, it suggests that the next frontier for foundation models is not merely performing professional tasks, but reliably producing work products that users can adopt without further verification or revision. We further investigate the underlying causes of this gap in Section 4.4.2.

Performance varies substantially across professional domains. Table 3 reveals that current general-purpose agents do not exhibit uniform competence across real-world workflows. Among the six domains, Business is consistently the easiest: every model achieves an average score above 60, and the average success rate across models is also substantially higher than in any other domain. In contrast, Finance is the most challenging, with the lowest average score of 54.48%. The large gap between Business and Finance suggests that current models handle structured, relatively standardized tasks more reliably than workflows requiring sustained quantitative reasoning, cross-document consistency, and multi-step verification.

The results also reveal pronounced domain specialization. While Kimi-K3 and GPT-5.6-sol achieve nearly identical average scores overall (73.67 vs. 73.61), their strengths difer substantially across domains. Under success rate, Kimi-K3 performs best in Medical and Business, whereas GPT-5.6-sol leads in Finance, STEM, and Education; GPT-5.5 remains strongest in Legal. A similar pattern emerges in average score, where

Kimi-K3 leads in Medical, Business, Finance and Education, GPT-5.6-sol in Legal and STEM. Thus, even among the strongest models, no single agent consistently dominates across professional domains. These results suggest that current general-purpose agents still exhibit substantial domain-dependent variation, leaving considerable room for improving the robustness and transferability of end-to-end task execution across heterogeneous professional workflows.

## 4.3 Effectiveness of Proposed Evaluation Framework

To validate the reliability of our evaluation framework, we compare its judgments against expert annotations. We sample outputs from two representative models for all tasks and ask domain experts to independently evaluate each submission using the predefined rubrics. At the rubric level, the automatic evaluator achieves an overall agreement of 92.78% with expert judgments. We further examine whether these local agreements translate into consistent task-level decisions by comparing the resulting success labels, where a submission is considered successful if its aggregated score reaches 90. The automatic and expert evaluations agree on 92.84% of these success decisions. Notably, the similarly high agreement at both rubric and success levels, despite their substantially diferent label distributions, suggests that the observed consistency is not merely an artifact of label imbalance.

We further investigate the necessity of evaluating each rubric independently. As an ablation, instead of assigning one rubric to one judge agent, we directly provide the complete model output together with the full rubric list to a single judge agent, requiring it to score all rubrics in one end-to-end inference. This alternative reduces the agreement with human experts to 83%. More importantly, the holistic setting proves considerably less stable in practice. Since the judge agent must simultaneously perform long-context reasoning, maintain consistency across numerous evaluation criteria, and finally generate a structured output covering all rubric scores, failures such as malformed outputs or incomplete formatting frequently occur, forcing repeated executions. On average, each evaluation requires more than two runs before a valid result is obtained. In contrast, our rubric-wise evaluation decomposes the assessment into a collection of independent, lightweight judging tasks, yielding both substantially higher agreement with human experts and significantly better execution robustness.

## 4.4 Failure-Mode Analysis

## 4.4.1 Outcome-level Failure Analysis

Capability differences across rubric dimensions. Figure 4 further breaks down model performance by rubric dimension. Overall, current models achieve consistently higher scores on Structure & Completeness, Information Integration, and especially Output & Presentation, indicating that they are generally capable of organizing heterogeneous information into well-structured and visually polished deliverables. In contrast, Domain-Specific Compliance emerges as the most challenging dimension across all evaluated models, while Calculation Precision also remains substantially weaker than the other dimensions. This suggests that although modern foundation models can produce artifacts that appear complete and professional, reliably satisfying domain-specific regulations, business conventions, and numerical correctness continues to be a major obstacle for real-world deployment.

The two strongest models, GPT-5.6-sol and Kimi-K3, outperform the remaining systems across nearly all rubric dimensions rather than relying on a single specialized capability. Their largest advantages are observed in Structure & Completeness, Calculation Precision, Information Integration, and Output & Presentation, demonstrating stronger end-to-end workflow execution and artifact construction capabilities. Nevertheless, even these leading models obtain their lowest scores on Domain-Specific Compliance, indicating that adherence to professional standards and domain-specific requirements remains the primary bottleneck even for state-ofthe-art systems.

Models often satisfy peripheral requirements while missing what matters most. Table 4 reports task-normalized rubric satisfaction rates across the three importance levels. Averaged across models, satisfaction decreases from 68.67% for Auxiliary rubrics to 65.89% for Important rubrics and 63.45% for Core rubrics, with 8 of the 9 models exhibiting the same overall trend. Importantly, these levels reflect each requirement’s contribution to successful task completion rather than its expected dificulty. This pattern reveals a characteristic failure mode of current agents: they can often fulfill auxiliary or surface-level requirements while failing on the core requirements that determine whether the task is actually completed successfully. In other words, substantial apparent progress can be driven by satisfying less consequential parts of the task, while critical omissions still prevent the final deliverable from being practically usable.

![](images/0d2c36f16ca53eb097eea7ddda54e8e2bdd208de49318bbeb767561c478efb0c.jpg)  
Figure 4 Performance across rubric dimensions. Importance-weighted pass rates (%) of diferent models on the six rubric dimensions in StartupBench. Higher values indicate better performance under the corresponding evaluation dimension.

A typical case of models struggling with requirements central to task completion is shown in Figure 5. In this mechanical process task, models often complete peripheral workbook requirements but fail on the core technical operations: extracting dimensions accurately from engineering drawings, validating specifications against the provided documents, and linking identified discrepancies to the corresponding rectification actions. These failures prevent the resulting workbook from fulfilling its primary purpose even when many auxiliary requirements are successfully completed.

Output format non-compliance. Some failures arise from models not producing deliverables in the file type explicitly required by the task. Unlike most failure modes discussed above, this requirement involves neither complex reasoning nor domain knowledge and can be verified deterministically from file extensions. Nevertheless, no model achieves perfect compliance on the 56 tasks with explicit output-format requirements (Table 5), with success rates ranging from 97.6% to 86.3%. Most violations fall into two recurring patterns: generating an incorrect file type (e.g., returning .md/.txt instead of the required .pdf) and omitting part of the requested deliverables (e.g., producing only one of the required .xlsx and .docx files). These errors suggest that even elementary deliverable-level constraints remain a non-negligible source of failures for current foundation models.

Table 4 Task-normalized rubric satisfaction rates (%) across diferent importance levels. For each model–task–trial, we first compute the satisfaction rate within each importance level and then average across tasks and trials.
<table><tr><td>Model</td><td>Auxiliary</td><td>Important</td><td>Core</td></tr><tr><td>Kimi-K3</td><td>77.89</td><td>75.12</td><td>73.40</td></tr><tr><td>GPT-5.6-sol</td><td>77.59</td><td>74.47</td><td>73.07</td></tr><tr><td>GPT-5.5</td><td>75.12</td><td>75.09</td><td>72.07</td></tr><tr><td>Seed-2.1-Pro</td><td>70.83</td><td>69.85</td><td>65.77</td></tr><tr><td>Kimi-K2.6</td><td>66.64</td><td>63.56</td><td>57.82</td></tr><tr><td>GLM-5.1</td><td>64.58</td><td>61.40</td><td>60.68</td></tr><tr><td>Qwen-3.6-Max</td><td>65.39</td><td>62.17</td><td>58.08</td></tr><tr><td>DeepSeek-V4-Pro</td><td>66.50</td><td>62.97</td><td>60.05</td></tr><tr><td>Gemini-3.1-Pro</td><td>53.48</td><td>48.41</td><td>50.13</td></tr><tr><td>Average</td><td>68.67</td><td>65.89</td><td>63.45</td></tr></table>

Table 5 Output-format compliance on the 56 StartupBench tasks with explicit output-file requirement rubrics. Models are ranked by compliance rate.
<table><tr><td>Model</td><td>Compliance (%)</td></tr><tr><td>Kimi-K3</td><td>97.6</td></tr><tr><td>Seed-2.1-Pro</td><td>95.2</td></tr><tr><td>GPT-5.6-sol</td><td>94.6</td></tr><tr><td>Qwen-3.6-Max</td><td>94.6</td></tr><tr><td>DeepSeek-V4-Pro</td><td>94.0</td></tr><tr><td>GPT-5.5</td><td>93.5</td></tr><tr><td>Kimi-K2.6</td><td>92.9</td></tr><tr><td>Gemini-3.1-Pro</td><td>86.9</td></tr><tr><td>GLM-5.1</td><td>86.3</td></tr></table>

## 4.4.2 Behavior-level Failure Analysis

Complex Instruction Following. Complex real-world workflows typically impose numerous constraints on the final deliverable, including structural organization, computation logic, formatting, and executable correctness. Although these requirements are explicitly specified in the task description, current models frequently exhibit partial compliance: they satisfy the most visible requirements while overlooking other equally critical constraints. As a result, the generated artifact appears largely complete from a superficial inspection, yet fails to satisfy the complete acceptance criteria required for direct use. This behavior indicates that models often optimize for producing outputs that look correct, rather than ensuring that every instruction governing the final deliverable has been faithfully executed.

A representative example is shown in Figure 6. The task requires generating a complete HR payroll workbook from multiple source sheets under a diverse set of structural, computational, and formatting constraints. Although the model produces an artifact that appears largely complete—including all required worksheets, formulas, charts, and dashboard layouts—the artifact-level evaluation reveals that several critical acceptance conditions remain unsatisfied. Specifically, key cached values required for downstream verification are missing, rendering the workbook unverifiable despite its polished appearance. This case illustrates a common pattern of partial compliance, where the model satisfies the most visible requirements while overlooking less obvious but equally essential deliverable constraints.

Self-Verification Hallucination. We also observe that many models fail not because they cannot complete the required workflow, but because they incorrectly assume that successful execution implies successful delivery. After producing a plausible artifact, the model often summarizes the intended reasoning process or completed steps as evidence that the task has been verified, without independently inspecting the final deliverable against the original user requirements. Consequently, subtle yet critical errors—such as incorrect cell values, unmatched identifiers, missing constraints, or formatting inconsistencies—remain undetected despite an otherwise convincing output. This behavior reflects a form of self-verification hallucination, where the model mistakes confidence in its own reasoning process for evidence that the generated artifact satisfies the acceptance criteria. In real-world ofice workflows, however, successful completion requires validating the final deliverable itself rather than merely confirming the execution process, making this a common source of deployment-critical failures. A typical case is shown at Appendix D.1.

![](images/910de513939b38cf2bc9f7f74b73a78062a4d4ec5747c1a49af5764fc40610c7.jpg)  
Figure 5 A representative case of models failing on core rubrics though passing auxiliary rubrics on StartupBench.

Domain-Specific Expertise. Another major source of failure arises from insuficient domain-specific expertise, consistent with the relatively poor performance on the Domain-Specific Compliance dimension. Unlike general instruction-following errors, these failures are driven by the unique operational requirements of diferent professional domains. Although models often generate outputs that appear fluent and professionally formatted, they frequently fail to satisfy the domain-specific constraints that determine whether a deliverable is actually usable. As a result, the dominant failure patterns vary substantially across domains, reflecting diferent notions of correctness, evidence, precision, and actionability required by real-world workflows.

From a domain perspective, diferent professional areas expose distinct failure characteristics. Business and management tasks primarily stress artifact correctness, particularly for spreadsheets, dashboards, formulas, and structured reports. Finance tasks emphasize source attribution, numerical precision, valuation assumptions, and temporal consistency, where errors commonly arise from insuficient reconciliation, inadequate numerical self-verification, or inconsistent accounting conventions. Legal tasks are less constrained by legal writing style than by the completeness and correctness of legal reasoning, including missing citations, incomplete factual coverage, insuficient statutory support, or incorrect mapping between facts and applicable rules. Medical tasks achieve relatively higher average scores but pose substantially greater safety risks, since mistakes in medication timing, continuation criteria, or discharge planning directly undermine clinical usability. STEM and Computer Science tasks instead require precise object-level grounding, demanding accurate relationships among code, engineering drawings, images, bills of materials, and system components. Finally, education and humanities tasks place greater emphasis on fine-grained rule following, such as curriculum requirements, credi allocation, document organization, and textual coherence. A typical example of failure caused by insuficient domain-specific expertise is shown at Appendix D.2.

## 4.5 Impact of Harness on Startup Tasks

![](images/71aae6674e2894f62f99a43a4d31598d327543ff9046ee413155d6d9d5a25ba3.jpg)  
Figure 6 A representative case of complex instruction following failure. Although the generated workbook appears complete, artifact-level evaluation reveals that several critical acceptance conditions remain unsatisfied, illustrating a common pattern of partial compliance.

Table 6 Performance under diferent general-purpose agent frameworks. For each model, we report the average score under three general-purpose frameworks and the performance range (max–min) across frameworks.
<table><tr><td>Model</td><td>Hermes</td><td>Claude Code</td><td>Nanobot</td><td>Max--Min</td></tr><tr><td>GPT-5.5</td><td>71.00</td><td>70.11</td><td>72.79</td><td>2.68</td></tr><tr><td>GLM-5.1</td><td>60.20</td><td>60.18</td><td>60.79</td><td>0.61</td></tr><tr><td>Qwen-3.6-Max</td><td>59.10</td><td>57.38</td><td>59.46</td><td>2.08</td></tr><tr><td>Average</td><td>63.43</td><td>62.56</td><td>64.35</td><td>1.79</td></tr></table>

## 4.5.1 Effect of General-Purpose Harness

In the main experiment, all models are evaluated under nanobot Agent framework. To examine whether the observed performance is primarily determined by the choice of agent framework rather than the underlying model, we re-evaluate the benchmark using two more representative general-purpose agent frameworks: Hermes [14] and Claude Code [1]. All frameworks are evaluated under the same task set and identical evaluation protocol, with only the execution framework replaced.

As shown in Table 6, replacing the general-purpose framework leads to only minor performance diferences. Across all evaluated models, the average variation between the best and worst framework is only 1.79 points. GLM-5.1 is almost completely unafected (0.61 points), while even the largest diference observed on GPT-5.5 is only 2.68 points. More importantly, the relative ordering of models remains unchanged across all three frameworks. These results suggest that StartupBench is largely robust to the choice of general-purpose agent framework. Although diferent frameworks adopt diferent prompting strategies, tool interfaces, and interaction mechanisms, simply replacing one general-purpose framework does not substantially alter endto-end task performance. Consequently, the performance gap observed on StartupBench primarily reflects the capability of the underlying foundation models rather than implementation-specific characteristics of a particular framework.

Table 7 General-purpose versus specialized agent systems. Oracle General denotes the averages of the highest score achieved in all trials for every model.
<table><tr><td>Agent System</td><td>Avg Score</td><td>Success Rate</td></tr><tr><td>General (All Runs avg)</td><td>64.26</td><td>19.74</td></tr><tr><td>General (Oracle)</td><td>71.75</td><td>28.06</td></tr><tr><td>Specialized Agent</td><td>83.50</td><td>39.18</td></tr></table>

## 4.5.2 General-Purpose Agents versus Specialized Agents

To further understand the role of agent frameworks, we compare general-purpose agent frameworks with the specialized startup agents from which StartupBench tasks are derived. Unlike the previous experiment, this comparison evaluates each task using its corresponding production startup agent, which has been specifically optimized for its target workflow through domain-specific model selection, prompting strategies, tool orchestration, and workflow design.

Table 7 summarizes the comparison between general-purpose and specialized agent systems. Across all executions, general-purpose agents achieve an average score of 64.26 and a success rate of 19.74%, substantially below the specialized agents at 83.50 and 39.18%, respectively. Since specialized agents are explicitly designed and optimized for their target domains, we further consider a stronger oracle setting for the general-purpose agents: for each model–task pair, we retain the best result among three independent trials. This oracle selection raises the average score to 71.75 and success rate to 28.06%, but still falls well short of the specialized agents, with gaps of 11.75 points in average score and 11.12 percentage points in success rate. Thus, the performance gap cannot be explained simply by stochastic variation or occasional execution failures: even when general-purpose agents are allowed multiple attempts and evaluated by their best observed outcome, specialized agent systems remain substantially more efective on these tasks. Combined with the failure-mode analysis presented earlier, these findings provide a more complete picture of the remaining gap between general-purpose foundation models and specialized startup agents. While today’s foundation models still cannot fully replace domain-specialized agents under a unified general-purpose framework, their shortcomings are concentrated in a relatively clear set of capabilities, including complex instruction following, domain-specific knowledge, professional operational conventions, and long-horizon workflow execution. This suggests that the observed advantage of specialized systems does not necessarily require fundamentally diferent agent architectures, but may instead reflect capabilities that current general-purpose models have yet to acquire reliably. As these underlying capabilities continue to improve, general-purpose agents may increasingly accomplish valuable professional end-to-end tasks without relying on domain-specific agent harness designs.

## 5 Conclusion

We introduce StartupBench, a benchmark for evaluating models on end-to-end professional workflows derived from market-validated AI-native products and real-world user demands. Across multiple tasks spanning diverse domains, our evaluation reveals a substantial gap between making meaningful progress on professional work and producing deliverables that fully satisfy practical acceptance criteria. This gap is primarily driven by limitations in complex instruction following, domain-specific expertise, professional conventions, and long-horizon workflow execution. By grounding evaluation in workflows that users already seek to delegate to AI, StartupBench provides a realistic measure of practical agent capabilities and highlights concrete directions for future improvement. We hope StartupBench supports the development of models and agents that can not only perform valuable professional work, but reliably deliver results ready for direct use.

## 6 Contributions

## Project Leads

Liya Zhu, Xin Ma, Tao Liu, Haodong Wang, Ge Zhang

## Core Contributors

Jingzhe Ding, Qingshui Gu, Yongjie Zhong, Jinxiang Meng, Yuan Gao, Yunqiu Zhou, Hao Zhu, Jifeng He, Yongzhi Liao, Xinyi Zhang, Chaoxin Li, Yi Zhu, Xi Lin, Duju Zeng, Xiang Gao, Wen Zhang, Yunyang Wang, Duo Wang

## Contributors

Huan Zhou, Zuo Wang, Jin Chen, Kaiyuan Zhang, Chuqian Yu, Tianhao Yu, Longxiang Liu, Jianbo Xue, Huimin Che, Jiahao Wang

## Sponsor Committee

Yujia Qin, Jiaheng Liu

## Corresponding Authors

Ge Zhang (gezhang@umich.edu)

Shen Yan (sheny@bytedance.com)

Xiaolong Chang (changxiaolong@bytedance.com)

Wenhao Huang (huang.wenhao@bytedance.com)

## References

[1] Anthropic. Claude code. https://github.com/anthropics/claude-code. GitHub repository.

[2] ByteDance Seed. Seed2.1 model card: Agentic intelligence for productivity. Technical report, ByteDance, 2026. URL https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2.1/ Seed2\_1\_Model\_Card.pdf.

[3] DeepSeek-AI. Deepseek-v4: Towards highly eficient million-token context intelligence, 2026. URL https: //huggingface.co/collections/deepseek-ai/deepseek-v4. Technical Report.

[4] Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, et al. Workarena: How capable are web agents at solving common knowledge work tasks?, 2024. URL https://arxiv. org/abs/2403.07718.

[5] Xeron Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, Chujie Zheng, et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines. Advances in Neural Information Processing Systems, 38, 2026.

[6] Google Cloud. Gemini 3.1 Pro Model Card, 4 2026. URL https://docs.cloud.google.com/vertex-ai/ generative-ai/docs/models/gemini/3-1-pro?hl=zh-cn.

[7] Xiaomeng Hu, Yinger Zhang, Fei Huang, Jianhong Tu, Yang Su, Lianghao Deng, Yuxuan Liu, Yantao Liu, Dayiheng Liu, and Tsung-Yi Ho. Occubench: Evaluating ai agents on real-world professional tasks via language environment simulation, 2026. URL https://arxiv.org/abs/2604.10866.

[8] Seungone Kim, Jay Shin, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Ryan Shin, Sungdong Kim, James Thorne, Minjoon Seo, et al. Prometheus: Inducing fine-grained evaluation capability in language models. In International Conference on Learning Representations, 2024.

[9] Fangyu Lei, Jinxiang Meng, Yiming Huang, Junjie Zhao, Yitong Zhang, Jianwen Luo, Xin Zou, Ruiyi Yang, Wenbo Shi, Yan Gao, et al. Dacomp: Benchmarking data agents across the full data intelligence lifecycle. arXiv preprint arXiv:2512.04324, 2025.

[10] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, 2024.

[11] Jinxiang Meng, Shaoping Huang, Fangyu Lei, Jingyu Guo, Haoxiang Liu, Jiahao Su, Sihan Wang, Yao Wang, Enrui Wang, Ye Yang, et al. Dv-world: Benchmarking data visualization agents in real-world scenarios. arXiv preprint arXiv:2604.25914, 2026.

[12] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In International Conference on Learning Representations, 2024.

[13] Moonshot AI. Kimi K2.6: Advancing open-source coding. https://www.kimi.com/blog/kimi-k2-6, 2026. Accessed: 2026-06-02.

[14] Nous Research. Hermes agent: The ai agent that learns from you. https://github.com/hermes-agent-org/ hermes, 2026. GitHub repository.

[15] OpenAI. Gpt-5.6 system card. Technical report, OpenAI, 7 2026. URL https://deploymentsafety.openai. com/gpt-5-6/gpt-5-6.pdf.

[16] OpenAI. Introducing gpt-5.5, 4 2026. URL https://openai.com/index/introducing-gpt-5-5/. System Card: https://deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf.

[17] Krista Opsahl-Ong, Arnav Singhvi, Jasmine Collins, Ivan Zhou, Cindy Wang, Ashutosh Baheti, Owen Oertell, Jacob Portes, Sam Havens, Erich Elsen, et al. Oficeqa pro: An enterprise benchmark for end-to-end grounded reasoning. arXiv preprint arXiv:2603.08655, 2026.

[18] Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, et al. Gdpval: Evaluating ai model performance on real-world economically valuable tasks. arXiv preprint arXiv:2510.04374, 2025.

[19] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

[20] Qwen Team. Qwen3.6-Max-Preview: Smarter, sharper, still evolving, April 2026. URL https://qwen.ai/blog? id=qwen3.6-max-preview.

[21] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551, 2023.

[22] Wentao Shi, Yu Wang, Yuyang Zhao, Yuxin Chen, Fuli Feng, Xueyuan Hao, Xi Su, Qi Gu, Hui Su, Xunliang Cai, et al. Aj-bench: Benchmarking agent-as-a-judge for environment-aware evaluation. arXiv preprint arXiv:2604.18240, 2026.

[23] Yiyou Sun, Xinyang Han, Weichen Zhang, Yuanbo Pang, Tianyu Wang, Yuhan Cao, Yixiao Huang, Chris Duroiu, Haoyun Zhang, Jefrey Lin, et al. Agents’ last exam. arXiv preprint arXiv:2606.05405, 2026.

[24] Zirui Tang, Xuanhe Zhou, Yumou Liu, Linchun Li, Yukai Wu, Weizheng Wang, Hongzhang Huang, Wei Zhou, Jun Zhou, Jiachen Song, et al. Workspace-bench 1.0: Benchmarking ai agents on workspace tasks with large-scale file dependencies. arXiv preprint arXiv:2605.03596, 2026.

[25] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

[26] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

[27] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

[28] Frank Fangzheng Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, et al. Theagentcompany: benchmarking llm agents on consequential real world tasks. Advances in Neural Information Processing Systems, 38, 2026.

[29] Qianyu Yang, Yang Liu, Jiaqi Li, Jun Bai, Hao Chen, Kaiyuan Chen, Tiliang Duan, Jiayun Dong, Xiaobo Hu, Zixia Jia, et al. Onemillion-bench: How far are language agents from human experts? arXiv preprint arXiv:2603.07980, 2026.

[30] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In 11th International Conference on Learning Representations, ICLR 2023, 2023.

[31] Runyang You, Hongru Cai, Caiqi Zhang, Qiancheng Xu, Meng Liu, Tiezheng Yu, Yongqi Li, and Wenjie Li. Agent-as-a-judge. arXiv preprint arXiv:2601.05111, 2026.

[32] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

[33] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

[34] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, 2024.

[35] Liya Zhu, Jingzhe Ding, Jian Zhang, Jianbo Xue, Shihao Liang, Ge Zhang, Yi Zhu, Duju Zeng, Xiang Gao, Qingshui Gu, Mailun Gao, Huimin Che, Yan Zhao, Peiheng Zhou, Haojun Wang, Chaobo Xian, Lili Le, Chi Wu, Yiwei Liu, Shengda Long, Jiale Yang, Fangzhi Xu, Sijin Wu, Haodong Duan, Chao He, Zhaojian Li, Minchao Wang, Huan Zhou, Jiani Hou, Chuqian Yu, Weiran Shi, Hongwan Gao, Jiamin Chen, Guanhong Chen, Tingqin Luo, Kaiyuan Zhang, Zhixin Yao, Qing Hua, Yuhao Jiang, Jin Chen, Pu Chen, Zhenyu Hu, Xingyu Li, Zhengxuan Jiang, Meng Cao, Tianfeng Long, Haozhe Wang, Mingzhang Wang, Yichen Zhang, Yiming Dai, Chenchen Zhang, Jiaying Wang, Xinying Liu, Xingzu Liu, Lingling Zhang, Xinjie Chen, Yujia Qin, Wangchunshu Zhou, Zhiyong Wu, Yang Liu, Jiaheng Liu, Lei Zhang, Shen Yan, Wenhao Huang, Zaiyuan Wang, and Xiaolong Chang. Workflow-gym:

Towards long-horizon evaluation of computer-use agentic tasks in real-world professional fields, 2026. URL https://arxiv.org/abs/2606.11042.

[36] Mingchen Zhuge, Changsheng Zhao, Dylan R. Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, Yangyang Shi, Vikas Chandra, and Jürgen Schmidhuber. Agent-as-a-judge: Evaluate agents with agents. In Forty-second International Conference on Machine Learning, 2025.

## Appendix

## A Tutorial of StartupBench Task Annotation

## A.1 Task Construction

A startup task can be defined as a triple $\mathcal { T } = ( q , \mathcal { E } , \mathcal { R } )$ . The experts are required to provide all the elements of a task:

• Input query: The query q is a natural-language statement of the task, specifying the user instruction that initiates the task. Derived from authentic scenarios of the work of annotators, the input query is supposed to reflect realistic user intent, including task background, implicit constraints and expected deliverables.

• Task workspace: The environment E denotes the complete workspace provided to the agent, including the multimodal source files together with any additional resources and interaction setting required to complete the task. The task workspace provides a self-contained execution environment for the agent. Each workspace includes all necessary input artifacts (e.g., files, datasets, or structured resources) required to complete the task, and ensures that the agent has access to suficient input information to solve the problem. This design enables reproducible evaluation under controlled conditions while preserving realistic workflow structure.

• Evaluation rubrics: The rubric $\mathcal { R } = \{ ( p _ { i } , w _ { i } ) \} _ { i = 1 } ^ { n }$ is a checklist within which every item consists of a natural-language scoring point $p _ { i }$ and a positive importance weight $w _ { i }$ reflecting its importance to successful task completion. The evaluation rubrics define a set of fine-grained criteria used to assess task completion quality. Due to the complexity and multi-faceted nature of real-world workflows, each task is evaluated using multiple rubric items covering diferent dimensions. We categorize all rubrics in StartupBench into 6 dimensions and 3 importance levels, which are detailed in Appendix C.

Together, these components preserve the structure of real-world agent workflows while enabling systematic and objective evaluation. Tasks may require information synthesis, constraint following, structured generation, or domain-specific reasoning and judgment.

## A.2 Cross Validation

To ensure reliability of the quality of data, in our construction process each task is independently reviewed by at least one additional domain expert with relevant professional experience. Rather than focusing only on annotation consistency, reviewers validate the realism and correctness of the task from multiple complementary perspectives.

• Task authenticity. Reviewers verify the authenticity of both the task description and the accompanying workspace. This includes checking that the input query reflects realistic user requests, that all provided materials are factually correct, and that no domain-specific knowledge errors exist.

• Workflow fidelity. Reviewers examine whether the task faithfully represents real-world workflows in the corresponding profession. Tasks that deviate from practical working procedures or fail to reflect genuine job responsibilities are revised or discarded.

• Rubric validity. Reviewers inspect the evaluation rubrics to ensure that the assessed capabilities are meaningful for the target workflow, that rubric items are clear and unambiguous, and that their relative weights reasonably reflect task priorities.

• Ground-truth verification. Reviewers validate the reference solution (ground truth) by evaluating it against the finalized rubrics. The ground-truth deliverable is required to achieve a full score. If inconsistencies are identified between the reference solution and the evaluation criteria, both the solution and the rubrics are revised until they are mutually consistent.

Table 8 Summary of domain expert backgrounds. Professional experience refers to reported years of experience in the corresponding or closely related domain.
<table><tr><td>Expert Statistics</td><td>Value</td></tr><tr><td>Number of domain experts</td><td>57</td></tr><tr><td>Median professional experience</td><td>5 years</td></tr><tr><td>Experts with ≥ 3 years of experience</td><td>68.4%</td></tr><tr><td>Experts with ≥ 6 years of experience</td><td>42.1%</td></tr><tr><td>Experts with ≥ 10 years of experience</td><td>24.6%</td></tr><tr><td>StartupBench domain coverage</td><td>6 /6</td></tr></table>

## B Domain Expert Background

We recruit 57 domain experts to support task construction and cross-validation. The expert pool covers all 6 domains in StartupBench and spans a diverse range of professional backgrounds, including clinical medicine, law, investment and quantitative finance, software development and artificial intelligence, engineering, business management and consulting, and education and humanities. Experts are assigned to tasks according to their reported areas of expertise and professional experience, ensuring that task construction and review were conducted by individuals familiar with the corresponding professional domains.

Table 8 summarizes the professional experience of the expert pool. The median professional experience is 5 years. Among the 57 experts, 68.4% have at least 3 years of professional experience, 42.1% have at least 6 years, and 24.6% have at least 10 years. These backgrounds provide practical domain knowledge for translating interview-derived user scenarios into reproducible benchmark tasks and for assessing the realism, correctness, and evaluation criteria of tasks during cross-validation. All experts are reasonably compensated based on the actual workload associated with task construction and review.

## C Rubrics of StartupBench

Rubric Dimensions. To facilitate consistent evaluation across heterogeneous real-world tasks, we organize all evaluation rubrics into six high-level categories according to the primary capability they assess. Rather than being tied to specific application domains, these categories capture common dimensions of deliverable quality that recur across ofice tasks, including correctness, completeness, data processing, professional reasoning, engineering quality, and presentation. This taxonomy enables more interpretable analysis of model strengths and weaknesses while maintaining a unified evaluation framework across diferent task types.

• Structure & Completeness. Evaluates whether the required deliverables are fully produced with the correct organization, file structure, required components, and overall completeness.

• Calculation Precision. Evaluates the correctness of numerical results, logical reasoning, formula execution, factual consistency, and other objective computations.

• Information Integration. Evaluates data cleaning, transformation, aggregation, statistical analysis, feature engineering, modeling, and other structured data manipulation workflows.

• Domain-Specific Compliance. Evaluates whether outputs satisfy domain-specific professional requirements, including business logic, financial principles, legal reasoning, medical practice, educational standards, and other expert knowledge.

• Engineering & Format. Evaluates implementation quality, coding conventions, spreadsheet engineering, document formatting, reproducibility, robustness, naming conventions, and compliance with technical specifications.

• Output & Presentation. Evaluates the quality of visual presentation and communication, including charts, layouts, formatting, readability, report organization, and overall usability of the final deliverables.

Table 9 summarizes the distribution of all 2,453 rubrics in StartupBench. Calculation Precision constitutes the largest rubric category (38.9%), reflecting the importance of producing objectively correct work products. Structure & Completeness and Domain-Specific Compliance together account for another 42.8%, highlighting that successful completion of realistic ofice workflows requires not only correct computations but also complete deliverables and professional domain reasoning.

Table 9 Distribution of rubric categories in StartupBench.
<table><tr><td>Category</td><td>Percentage</td></tr><tr><td>Structure &amp; Completeness</td><td>27.31%</td></tr><tr><td>Calculation Precision</td><td>38.89%</td></tr><tr><td>Information Integration</td><td>9.46%</td></tr><tr><td>Domain-Specific Compliance</td><td>15.49%</td></tr><tr><td>Engineering &amp; Format</td><td>6.20%</td></tr><tr><td>Output &amp; Presentation</td><td>2.65%</td></tr><tr><td>Total</td><td>100.00%</td></tr></table>

Rubric Importance Levels. To account for diferences in the importance of individual requirements to successful task completion, we assign each rubric to one of three importance levels. Core rubrics capture requirements that directly determine whether the primary user needs are satisfied; violating them would substantially compromise the correctness, reliability, safety, or usability of the final deliverable. Important rubrics represent key requirements for high-quality task completion whose satisfaction materially afects the overall quality of the deliverable, but whose individual violation does not necessarily render it unusable. Auxiliary rubrics capture lower-level requirements whose omission primarily afects details, user experience, visual appeal, or polish rather than the fundamental usability of the deliverable. We assign weights of 5, 3, and 1 to Core, Important, and Auxiliary rubrics, respectively, and use these weights when aggregating rubric-level judgments into the task score. A typical case of task containing rubrics of diferent levels is shown at Figure 7.

Across all tasks, Core, Important, and Auxiliary rubrics account for an average of 59.10%, 34.33%, and 6.57% of the total task weight, respectively, ensuring that task scores are primarily determined by requirements that materially afect successful task completion.

## D Examples of Behavior-level Failure Modes

## D.1 Self-Verification Hallucination

A representative example is shown in Fig. 8. The task requires the model to filter all level-11 members, simulate route check-ins until each member reaches the level-12 threshold, and generate an Excel workbook satisfying a set of precise value and formatting constraints. The produced workbook appears largely complete, containing the required worksheet, member records, and output columns, leading the model to conclude that the task has been successfully finished. However, artifact-level inspection reveals multiple acceptance-critical errors, including incorrect efective-point values and mismatched member names. The underlying issue is not that the model fails to perform the required computation, but that it never verifies the generated artifact itself. Instead, it treats a successful execution summary as evidence that the final deliverable has been validated. Consequently, localized yet user-critical errors remain unnoticed, illustrating a form of self-verification hallucination, where confidence is derived from the reasoning process rather than from independently checking the completed artifact against the original task requirements.

## D.2 Domain-Specific Failure

A representative example of failure caused by insuficient domain-specific clinical actionability is shown at Figure 9. The model generates a well-formatted multidisciplinary treatment plan that appears clinically professional, yet violates multiple safety-critical treatment constraints. The failure stems not from poor writing quality, but from an inability to convert medical knowledge into actionable and clinically reliable decisions.

![](images/5ecc38e88ff152c470931f8eab1648c4d6cf4a7e41ccddc83ae87532eb4a28f9.jpg)  
Figure 7 A representative example of a rubric of a task spanning 3 levels. Core rubrics assess the essential financial reasoning and calculations required to solve the investment problem, Important rubrics evaluate the correctness and completeness of the supporting analysis, while Auxiliary rubrics capture presentation quality and additional quantitative support.

![](images/814109e5870a272018bf3d47b6df21c7b710e0e06efe6b46f76c7aa02616f9b0.jpg)  
Figure 8 A representative example of self-verification hallucination. The model generates a plausible workbook and concludes that the task has been completed based on its execution process. However, artifact-level spot checks expose acceptance-critical errors in the final deliverable, showing that the model mistakes a progress summary for actual verification instead of validating the generated artifact against the original task requirements.

![](images/8f7658731d9e75b6836d431f8e20b13091c29875648e7bf372720f39dd909d1e.jpg)  
Figure 9 A representative example of failure caused by insuficient domain-specific clinical actionability.
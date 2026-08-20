# CentaurBench: Benchmarking LLM Capabilities on Augmenting vs. Automating Real-World Work Tasks

Pattaraphon Kenny Wongchamcharoen Kris Gulati

UC Berkeley

UC Berkeley

Min Min Fong

Abhishek Nagaraj

UC Berkeley

UC Berkeley and NBER

July 2026 Working Paper

## Abstract

Most LLM benchmarks rank models on their ability to automate work tasks. In practice, however, models are often used to assist other (human or LLM) agents. The question that drives model selection is therefore not only which model produces the best output, but which model most improves the work of another (weaker) agent. We introduce a unified framework that evaluates the capability of models to automate and augment another agent’s performance. Across seven economically grounded real-world tasks, an assistant model writes assistance text for a standardized lower-capacity worker model, which produces the deliverable. In automation mode, the assistant produces the output directly. Outputs are scored through blind pairwise comparisons by an LLM judge panel with task-specific rubrics, replicated across ten runs. Rankings across the two regimes are only modestly correlated, and the automation winner loses augmentation on five of seven tasks. Assistance is not reliably positive. The unaided worker outranks every assisted condition on three tasks, and only one model’s guidance beats no guidance on average. These results suggest that automation ability is an incomplete proxy for assistance quality, motivating benchmarks that evaluate models accordin to the roles they play in human-AI and multi-agent systems.

## 1 Introduction

Since the explosion in the adoption of large language models (LLMs), researchers have sought to measure the growth in capabilities across a wide variety of benchmarks. This work has moved from measuring capabilities on abstract knowledge or reasoning capabilities (Hendrycks et al., 2021; Srivastava et al., 2023) to more real-world, economically valuable tasks (Patwardhan et al., 2025; Wang et al., 2026; Hu et al., 2026). These benchmarks provide an excellent indication of a model’s capacity to automate work. However, in many real organizational and personal settings, models do not produce the output themselves. They are often assisting other agents, such as a human or another, perhaps weaker, LLM to complete the task. In such settings, models help workers plan an approach, decompose a task into steps, identify potential errors, and revise their work while leaving responsibility for the final deliverable with the worker. A model that excels at completing a task independently may not be the model that provides the most useful guidance to someone else. Put diferently, as research with humans shows, the best (AI) player may not be the best coach (Hinds, 1999; Camerer et al., 1989).

Existing research does not provide a general way to compare these two capabilities. On the one hand, large-scale computer science benchmarks primarily rank models as autonomous task solvers, including increasingly realistic evaluations of professional and economically valuable work (Hendrycks et al., 2021; Liang et al., 2023; Patwardhan et al., 2025; Li et al., 2025; Hu et al., 2026; Wang et al., 2026; Li et al., 2026). On the other hand, field experiments by economists and management scholars examine whether AI augments human performance, but generally evaluate a single model in a particular domain and against a limited set of experimental conditions (Noy and Zhang, 2023; Peng et al., 2023; Dell’Acqua et al., 2026; Chen and Chan, 2024; Doshi and Hauser, 2024; Brynjolfsson et al., 2025; Ju and Aral, 2026).

There is a line of work on multi-agent systems. In one stream, past work finds that a model’s performance in isolation does not reliably predict its performance in interactive settings (Collins et al., 2024; Chang et al., 2025; Vaccaro et al., 2024). Research has also studied how stronger models can teach weaker agents (Saha et al., 2023) or human learners (Macina et al., 2025), and how AI can complement human decision makers (Bansal et al., 2021). These studies evaluate assistance within a single domain (e.g. reasoning accuracy, tutoring pedagogy etc.), and score the assistance interaction itself. No existing benchmark compares models ability to assist and automate on an identical set of work-related tasks. Indeed, Haupt and Brynjolfsson (2025) argue that model evaluation should move beyond an “imitation game” of autonomous task completion toward centaur evaluations that measure how models augment human performance, yet no research operationalizes this call, jointly evaluating models as automators and as augmenters across multiple professional domains. Motivated by this gap, in this paper, we explore how models’ automation capability correlates with its augmentation capability and explore how this correlation itself varies across tasks and models.

We evaluate LLMs as automators and augmenters across seven heterogeneous professional tasks. Our pipeline evaluates a set of frontier LLMs across two axes simultaneously: usage mode (automation versus augmentation) and task domain. In automation mode, each focal model completes a professional task end-to-end. In augmentation mode, each focal model provides assistance text to a fixed worker model, which then completes the task. Holding the worker constant isolates the value of the assistant’s guidance from the executor’s own capability, producing a direct measure of assistant models’ quality rather than team outcome. This approach allows us to measure diferences in augmentation and automation to any model-task pairing without a separate field experiment. Both outputs are evaluated by a panel of LLM judges using task-specific rubrics, with model families excluded from judging their own outputs (e.g. Claude Opus 4.8 cannot judge other Claude model outputs). The result is a rank matrix at the task and model level for both the augmentation and automation regimes.

Three key results are worth highlighting. First, we find that model rankings diverge meaningfully across usage modes and task domains. The strongest autonomous task solver is not necessarily the strongest assistant, and no single model dominates across augmentation tasks. The model-level rank correlation is about 0.48 between augmentation and automation regimes and itself varies, ranging from −0.04 (travel planning) to 0.85 (tax preparation) across tasks. Second, assistance is not always beneficial. While GPT-5-mini ranks first for both automation and augmentation schemes, the unaided worker baseline, in which GPT-3.5-Turbo completes the task without guidance from another model, achieves the second best average rank in the augmentation comparison. Guidance that is overly complex, poorly matched to the worker, or miscalibrated to the task can leave the downstream worker worse of than working alone. Third, reversals in model performance across regimes at the task level are large and stable. For example, the strongest automation model on our market trends analysis task, Claude-Opus-4.8 (mean rank 2.05), is among the weakest assistants on the same task (8.15), while GPT-4.1 shows the mirror-image profile on counseling (a lower-ranked direct solver but the strongest assistant).

Our study makes three contributions. First, we introduce a common evaluation framework for comparing models as both autonomous task solvers and assistants across professional domains. Second, by holding the downstream worker constant, we provide a direct measure of the marginal value of model-generated guidance and show that automation performance does not determine augmentation performance. Third, we demonstrate that the value of assistance depends on both the model and the task, challenging the use of a single general-purpose leaderboard for organizational model selection. These findings have direct implications for organizations choosing and deploying LLMs. The relevant question is not simply, “Which model is best?” but rather, “Which model is best for this role and this task?” A model selected to produce finished deliverables may difer from the model selected to support a worker through planning, critique, or revision. More broadly, our results suggest that future benchmarks should move beyond agent-alone performance and incorporate role-specific, workflow-aware evaluations of models as assistants, coaches, and collaborators.

## 2 Related Literature

Our study connects research on LLM benchmarking, field experiments on AI-assisted work, and emerging frameworks for evaluating human–AI complementarity.

## 2.1 LLM Benchmarks and Autonomous Performance

A large computer-science literature evaluates LLMs by asking them to complete tasks autonomously. Early general-purpose benchmarks emphasize knowledge and reasoning across broad collections of tasks. MMLU, for example, tests knowledge across 57 subjects (Hendrycks et al., 2021), while HELM evaluates models across a wide range of scenarios and evaluation dimensions (Liang et al., 2023). These benchmarks have played an important role in standardizing model comparisons, but their basic unit of evaluation remains the output produced by a model working independently.

More recent benchmarks have moved toward realistic professional and economically grounded tasks. These evaluations test models on industrial workflows (Li et al., 2025), occupational tasks spanning specialized domains (Hu et al., 2026), and work products designed to approximate economically valuable professional activity (Patwardhan et al., 2025; Wang et al., 2026). JobBench further makes task selection more worker-centered by constructing 130 agentic tasks across 35 occupations from duties that workers report wanting to delegate, and by evaluating the resulting work against fact-anchored criteria (Li et al., 2026). This represents an important shift from selecting tasks solely according to economic value or technical feasibility toward also considering which activities workers want AI systems to perform. This progression has increased the realism and economic relevance of model evaluation. Nevertheless, operationally, these benchmarks still evaluate the model as the party responsible for completing the selected task and producing the final deliverable end-to-end. They therefore provide increasingly realistic measures of automation, but do not directly measure the value a model creates by augmentation.

That focus leaves a distinct capability undermeasured. In many workplace applications, the model does not directly produce the final deliverable. Instead, it helps another party plan, execute, review, or revise the work. Evidence that autonomous performance does not necessarily predict interactive performance reinforces this concern (Collins et al., 2024; Chang et al., 2025). A benchmark that identifies the best independent task solver may therefore provide limited guidance about which model should be deployed as an assistant.

## 2.2 AI Augmentation and Workplace Productivity

A parallel literature in economics and management examines how access to generative AI afects worker performance. Across several settings, experiments have found substantial productivity efects. ChatGPT reduced the time required for professional writing tasks while improving output quality (Noy and Zhang, 2023); GitHub Copilot accelerated completion of a programming task (Peng et al., 2023); and a large-scale deployment of a generative-AI assistant increased productivity among customer-support workers (Brynjolfsson et al., 2025). Related studies examine consultants (Dell’Acqua et al., 2026), advertising professionals (Chen and Chan, 2024; Ju and Aral, 2026), and creative writers (Doshi and Hauser, 2024).

These studies establish that generative AI can afect worker productivity and that its efects depend on the task, worker, and mode of interaction. At the same time, field experiments are necessarily narrow by design. They typically study one model or tool, one population, and one domain, often relative to an unaided control condition. Their purpose is generally to estimate whether access to a particular AI system improves performance in a particular setting, rather than to compare which of several models would provide the greatest assistance to the same worker.

The evidence also cautions against treating human–AI collaboration as automatically beneficial. In a meta-analysis of 106 experiments, Vaccaro et al. (2024) find that human–AI combinations frequently underperform the better of the human-alone and AI-alone conditions. The value of assistance may depend on how capabilities are distributed between the parties, how work is allocated, and whether the system’s guidance is appropriately calibrated to the worker. These findings motivate evaluations that compare not only whether assistance is available, but also which model provides that assistance and how its value varies across tasks.

This question is economically consequential because automation and augmentation can produce diferent organizational and distributional efects. Agrawal et al. (2023) argue that automating one worker’s tasks can create augmentation opportunities elsewhere in a production process, while Marguerit (2025) distinguish the labor-market efects of augmentation-oriented and automation-oriented AI. Worker preferences may also favor collaborative arrangements: a nationwide survey of 1,500 U.S. workers across 104 occupations found that “equal partnership” with AI, rather than replacement, was the dominant preference in nearly half of the occupational categories studied (Shao et al., 2025). Together, this research highlights the importance of evaluating AI systems in the roles through which they are actually deployed.

## 2.3 Automation, Augmentation, and Human–AI Complementarity

A growing body of work more directly distinguishes automation from augmentation. Haupt and Brynjolfsson (2025) argue that evaluation by imitation privileges systems that reproduce human outputs and replace human efort, rather than systems designed to improve and augment human performance. They call for evaluations in which humans and AI systems solve tasks together. This critique shifts the object of evaluation from the model’s standalone capability to the value it creates within a larger workflow.

F¨ugener et al. (2026) formalize the distinction between automation and augmentation, showing that the optimal allocation of work depends on the structure of complementarity between human and AI capabilities. A system that performs well independently need not create the greatest value when paired with another decision-maker or worker. Their analysis therefore provides a theoretical basis for expecting model rankings to vary across autonomous and assistive roles. A growing set of eforts evaluates this type of assistance empirically, but each within a narrow scope. Upwork’s Human+Agent Productivity Index (HAPI) evaluates frontier agents on approximately 300 paid client projects, and finds that expert involvement can substantially improve agent performance (Yi et al., 2025). However, a human expert assists the AI, reviewing and correcting the agent’s output, so the human occupies the coaching role and the model is the party being helped. MathTutorBench benchmarks the pedagogical quality of LLM tutors, finding that stronger problem-solving ability does not automatically imply better teaching (Macina et al., 2025). Gandhi et al. (2025) study strong–weak model collaboration for code generation, in which an expert model supplies plans for an inferior model to execute, reducing cost while preserving performance. Saha et al. (2023) examine a teacher – student simulation between two LLMs, showing that the stronger model’s explanations improve the weaker model’s performance when explanations are personalized to the student. Each of these studies is confined to a single domain (e.g. tutoring, coding, freelance projects) and to a single human - LLM or model - model pairing, rather than comparing many models against one another in the assistant role.

Our approach difers from these studies by spanning a range of real-world professional tasks and placing models into an assistance/guidance role to a fixed worker. Importantly, the latter allows measurement of the marginal value of an assistant model’s guidance rather than its joint outcome as a team. Rather than asking only whether automation or augmentation is optimal in a particular setting, we compare the same set of models under both modes and across multiple professional domains. This design allows us to examine whether relative model rankings are stable across roles and whether a model’s autonomous capability is also an indication of the value of its guidance to a fixed downstream worker. In sum, our study operationalizes this distinction as a cross-model, cross-mode, and cross-task empirical evaluation.

## 3 Methodology

## 3.1 Overview

We introduce a framework that evaluates large language models not only as direct task solvers but as assistants that help another agent do the work. We study two usage modes. In the automation mode, as in most existing benchmarks, a focal model completes a professional task end-to-end. In the augmentation mode, a fixed lower-capability worker model completes the task with help from an assistant model: the worker requests assistance, the assistant model replies with an assistance text - a plan, checklists, constraints, and self-review steps - and the worker uses that support to produce the final deliverable. The assistant model never writes the deliverable itself, so what varies across augmentation trials is the quality of its guidance, not whether it simply did the work. We use GPT-3.5-Turbo as the worker for two reasons: it approximates worker performance on our pilot tasks, and it represents the cheaper, lower-capability models that are often assigned the execution role in multi-agent settings. Figure 1 summarizes the full workflow.

Every output is evaluated in both usage modes through blind pairwise comparisons by a panel of four LLM judges using task-specific and general rubrics. Each judge is barred from scoring outputs produced by its own model family (leave-one-family-out masking) to limit self-preference, and we replicate the full pipeline across ten independent runs. The result is a cross-mode, cross-task rank matrix over ten models: Claude-Opus-4.8, Claude-Sonnet-4.6, Gemini-3.1-Pro, DeepSeek-V3.1, GPT-5-Mini, GPT-OSS-120B, GPT-O4-Mini, GPT-4.1, GPT-O3-Mini, and GPT-3.5-Turbo. For each model, we can ask not only whether it ranks higher as a direct solver or as an assistant, but also whether that ordering holds across tasks. The framework therefore produces a profile for each model rather than a single score. A model may complete structured execution tasks well on its own yet be a weaker assistant on open-ended, human-facing tasks; another may rank lower as a direct solver but produce assistance text that clearly improve counseling, tutoring, or planning outputs. Capturing this split is our core methodological contribution.

![](images/f6bd9de41934f3d1b6877b5c41bcfc268d333a641fa4054260ee1f774cde243b.jpg)  
Figure 1: Methodology pipeline. The framework evaluates models in two usage modes: automation, where focal models directly solve tasks, and augmentation, where focal models provide an assistance text to a fixed worker model. Outputs are evaluated through rubric-guided pairwise comparisons and aggregated into model rankings by task and usage mode.

## 3.2 Task Selection and Prompt Design

We construct a benchmark of seven tasks representing activities commonly encountered in everyday and professional settings and associated with economically meaningful forms of work: counseling, market analysis, meal planning, operations research, tax preparation, travel planning, and tutoring. Tasks were chosen to vary in structure, domain knowledge, risk, and the degree to which human-facing judgment matters. Each task specifies an observable deliverable with concrete requirements, allowing evaluation to assess whether outputs satisfy stated constraints objectively rather than relying on subjective impressions of quality. Figure 2 presents the full task set. The assistant model set and judge panel are reported in Appendix A.

To make evaluation reliable, each task prompt is paired with a micro-rubric. The prompt specifies the required deliverable and the components that must be present. The rubric mirrors those components and scores each dimension on a 1–10 scale. Each task includes task-specific dimensions, such as dietary safety for meal planning or mathematical correctness for tutoring, as well as general dimensions such as instruction following, accuracy, practical usefulness, organization, and tone. We evaluate models pairwise to enable contrastive grading (which we found worked better), where the judge can compare models to determine winners rather than grade inherently subjective outputs. However, we also develop the task-specific rubrics to help ground the judge’s decision, and also as a way for humans to reason about why certain outputs were judged better than others.

<table><tr><td>Task</td><td>Source</td><td>Task type</td><td>Deliverable</td></tr><tr><td>Menu Planning</td><td>GDPval</td><td>Structured planning</td><td>Seven-day constrained meal plan</td></tr><tr><td>Travel Planning</td><td>GDPval</td><td>Structured planning</td><td>Tokyo itinerary under fixed budget</td></tr><tr><td>Market Trends Analysis</td><td>GDPval</td><td>Professional / analytical</td><td>Client brief on natural gas trends</td></tr><tr><td>Operations Research Analysis</td><td>Anthropic Economic Index</td><td>Professional / analytical</td><td>Executive logistics optimization memo</td></tr><tr><td>Tax Preparation</td><td>Designed task</td><td>Professional / analytical</td><td>Tax discrepancy review and correction guidance</td></tr><tr><td>Counseling</td><td>Anthropic Economic Index</td><td>Human-facing interactive</td><td>Supportive counseling-style</td></tr><tr><td>Tutoring Session Planning</td><td>Anthropic Economic Index</td><td>Human-facing interactive</td><td>response Grade 3 lesson segment</td></tr></table>

Figure 2: Tasks included in the benchmark. The figure reports each task’s source, broad task type, and required deliverable. Colors distinguish structured-planning tasks (yellow), professional or analytical tasks (green), and human-facing interactive tasks (purple). Tasks were drawn from GDPval and the Anthropic Economic Index, except for tax preparation, which was designed by the authors to provide a rule-based professional task with verifiable discrepancies. Full task prompts and evaluation rubrics are provided in Appendix B.

This alignment between prompt and rubric is central to the framework. We do not ask evaluators to make an undiferentiated and subjective judgment of quality. Instead, they judge whether the response satisfies observable dimensions of the task that are quantifiable and comparatively objective to assess. For example, the travel-planning rubric separately measures cost realism, itinerary quality, hotel and transportation practicality, and handling of uncertainty. Meanwhile, the tax-preparation rubric separately measures rule accuracy, discrepancy detection, calculation quality, form guidance, dependent analysis, and client-facing clarity. Figure 3 summarizes the alignment between task prompts, task-specific micro-rubrics, general evaluation dimensions, and the augmentation assistance text constraint.

![](images/aec2818d1e4d59c61ed0cca059f08578ae376b71fc7bc032a77cbe194ba08a6f.jpg)  
Figure 3: Prompt and rubric design. Each task prompt specifies observable deliverable requirements, which are mirrored by task-specific micro-rubrics. General rubric dimensions are held constant across tasks, while task-specific dimensions vary by domain. In augmentation mode, the assistance text creation prompt constrains assistant models to provide process guidance rather than the final deliverable.

Figure 4 illustrates the prompt and rubrics for travel planning. Full prompts, assistance texts, and rubrics for all tasks appear in Appendix B. In augmentation mode, each assistant model receives a universal assistance text creation prompt that instructs it to produce a processoriented plan for the worker, rather than completing the task itself. The prompt specifies a three-phase structure: a requirements check, an execution plan, and a final checklist. It additionally constrains the assistant to provide guidance only, explicitly prohibiting production of the final deliverable. This design ensures that what varies across augmentation trials is the quality of the assistant’s planning and decomposition, not whether the assistant simply solved the task on the worker’s behalf.

![](images/7d0407b62177147ab53fcb13254d98b301b87cc6676799277b36cf3a9380768d.jpg)  
Figure 4: Illustrative task prompt, evaluation rubric, and assistance-text instruction. The travel-planning example shows how observable task requirements (top) map onto task-specific rubric dimensions (middle). In augmentation mode, the assistant additionally receives the universal assistance text creation prompt shown above (bottom), which requires process-oriented guidance while prohibiting completion of the final deliverable. The complete prompts and rubrics are provided in Appendix B.

Augmentation can be operationalized in many ways, ranging from an assistant that drafts content for the worker to edit, to one that supplies worked examples or directly corrects worker errors. Education research draws a similar distinction: procedural, strategic, and metagonitive scafolding that shapes how a learner approaches a task, versus conceptual scafolding that supplies domain content (Hannafin et al., 2013). We study process-oriented scafolding, in which the assistant shapes how the worker approaches the task but never supplies solution content. This mirrors the educational ideal of teaching a learner to perform a task rather than performing it for them (Wood et al., 1976; Van de Pol et al., 2010) and isolates the value of guidance from the assistant’s own task-solving ability. By design, it is a deliberately conservative form of assistance and lower bound on what augmentation can achieve, since it withholds conceptual, content-bearing support. Thus, the results should be read as characterizing this form rather than augmentation at large. However, the same fixed-worker methodology extends directly to richer forms of assistance, and mapping how augmentation value varies across richer types is a natural next steps our framework is designed to support. The full assistance prompt is provided in Appendix B.

## 3.3 Evaluation Procedure

![](images/cdc943cb355fcc82919c6b812a2415bfd8ba2d926d34a718356643e4aec10c97.jpg)  
Figure 5: Rubric-guided pairwise evaluation pipeline. Candidate outputs are anonymized, randomly paired, and evaluated by LLM judges using task-specific and general rubric dimensions. Pairwise choices are converted into win rates, aggregated across eligible judges and repeated runs, and reported alongside rubric scores and rationales as an audit trail.

Following prior work on LLM-as-judge evaluation and pairwise preference benchmarking (Zheng et al., 2023; Chiang et al., 2024), we evaluate final outputs through blind pairwise comparisons conducted by a panel of four LLM judges. Figure 5 summarizes the procedure.

For each task, usage mode, and independent run, we conduct a complete round-robin tournament among the ten candidate outputs. In augmentation, this candidate set includes the plain baseline: the fixed worker operating without an assistance text. This baseline indicates whether receiving assistance improves the worker’s output at all. Automation and augmentation outputs are evaluated in separate tournaments and are never compared directly.

For every comparison, model identities are concealed and option order is randomized. The judge scores both responses from 1 to 10 on the task-specific and general rubric dimensions, computes an average score for each response, selects the stronger response, and provides a short rationale. We apply leave-family-out evaluation: a judge does not evaluate comparisons containing outputs from its own model family. The model and judge panels are reported in Appendix A.

We treat pairwise comparison and rubric-based scoring as complementary components of our evaluation pipeline. Pairwise judgments provide the primary ranking signal because relative comparisons can recover expert-intended quality orderings more reliably and with lower annotation time than absolute rubric scores in judgment-intensive professional tasks (Yang et al., 2026). More broadly, rubric-guided pairwise LLM judgments have been shown to align closely with human preferences (Zheng et al., 2023). Pairwise choices alone, however, reveal which response is preferred without fully explaining why. Dimension-level rubric scores provide an inspectable diagnostic record, identifying the specific strengths and weaknesses underlying each comparison and supporting our qualitative analysis. Because rubric-based evaluation is itself sensitive to score calibration, scale use, and score-option ordering (Xu et al., 2026), we use standardized rubric-derived rankings as a robustness check rather than as a replacement for pairwise comparison. We therefore use pairwise outcomes to construct the primary rankings and standardized rubric scores to interpret those rankings and test their robustness. Together, the two signals make the evaluation more reliable, transparent, and diagnostically useful. Across non-tied comparisons, the selected response receives the higher mean rubric score in 99.7% of cases. In addition to the consistency between a judge’s choice and its own rubric scores, we also measure agreement across judges. We separately assess inter-judge reliability across all runs. Across 6,265 comparisons evaluated by at least two eligible judges, the judges select the same winner in 71.0% of cases. Agreement is higher in automation (74.5%) than in augmentation (67.8%). Appendix F reports the full LLM-judges reliability analysis.

![](images/a381c0744f0b6d52972d453007cf9f28b7c166c3e9886360cd2bb610c5cea025.jpg)

![](images/08c318f519893d7da8fe876d4798b30435f9075f93e410141d272fca22b0a6c1.jpg)  
Figure 6: Rubric-guided scoring. Judge rationales explain the pairwise choice made. The models used in this example are GPT-4.1 and DeepSeek-V3.1 in augmentation mode for the counseling task. The corresponding outputs are in Appendix G.

We aggregate the evaluations in three stages. First, each judge’s choices are converted into a pairwise win rate for every model within each task and usage mode. Second, models are ranked by win rate, and their ranks are averaged across eligible judges and ten independent runs. Third, task-level ranks are combined through a rank-of-ranks aggregation to obtain an overall ordering for each usage mode. We retain all task-level rankings, rubric scores, and rationales for subsequent analysis.

Figure 6 illustrates the resulting audit trail using two counseling outputs from augmentation mode. It reports the dimension-level scores and judge rationale for outputs produced with assistance texts from GPT-4.1 and DeepSeek-V3.1. The corresponding assistance texts and final worker outputs appear in Appendix G.

## 4 Results

We report two sets of results. First, we examine the automation mode, where each model solves the task directly. It is imperative to acknowledge here that these results characterize models’ performance on single-turn, bounded professional deliverables completed without tool usages. This setting difers from public agentic benchmarks, which often evaluate multi-step tool use, planning, and interaction with external environments; model orderings from those benchmarks therefore need not transfer to the setting studied here.

Second, we examine the augmentation mode, where each assistant model acts as a coach that supplies a process guidance to a fixed GPT-3.5-Turbo worker that produces the final output, rather than solving the task itself. We then compare augmentation rankings to automation rankings to test whether strong direct task-solving ability also implies strong assistant ability. Across both analyses, rankings are computed over ten independent runs of the full pipeline. Main-text figures report within-task rank of ranks (integer ranks derived from mean ranks across runs); Appendix C reports mean rank with standard error $\mathrm { ( S E = S D / \sqrt { 1 0 } ) }$ . All rankings are computed over the full candidate pool, which includes the unaided GPT-3.5- Turbo baseline. The interactive results dashboard display the full results.<sup>1</sup>

Our central result is that model performance is both task-specific and mode-specific. In automation, rankings are concentrated around a small set of strong direct solvers, although this ordering should be interpreted as specific to the bounded, single-turn tasks evaluated here rather than as a universal ranking of model capability. In augmentation, no single assistant model dominates across all tasks: diferent task domains reward diferent assistant behaviors, and several tasks are best handled by the unaided worker baseline rather than by an external assistance. This divergence suggests that standard automation benchmarks do not capture an economically important capability, whether a model can improve another worker’s output through planning and decomposition. We use the paper figures to summarize the main patterns; the interactive dashboard provides a more detailed audit trail for readers who wish to inspect individual tasks, models, and judgments.

## 4.1 Model Rankings in Automation Mode

![](images/c235f0cb0b5f6f3270711ce1d55f794806868c36e96f4423258da50f73452183.jpg)  
Figure 7: Automation rankings by task, averaged across ten independent runs. For each task, we first average each model’s within-run rank across the ten runs and then rank these mean values to produce the displayed rank of ranks (1 = best). Rankings are computed using unrounded mean ranks. Exact ties receive the same rank, with the subsequent rank skipped (e.g., 1, 1, 3). Models are ordered from newest to oldest from left to right. Lower ranks indicate better performance. Mean ranks and standard errors are reported in Appendix C, Figure 10.

Figure 7 reports model rankings in the automation mode by task across ten independent runs. GPT-5-Mini has the best average rank across tasks, followed by Claude-Opus-4.8. GPT-O4-Mini and Claude-Sonnet-4.6 form the next tier, while GPT-3.5-Turbo (plain) ranks last on average. The task-level results are also relatively concentrated. GPT-5-Mini ranks first in menu planning, tax preparation, travel planning, and tutoring; Claude-Opus-4.8 ranks first in market-trends analysis; GPT-OSS-120B ranks first in counseling; and GPT-OSS-120B and Gemini-3.1-Pro jointly lead operations research.

The automation results characterize direct task performance under the particular roles, tasks, and interaction constraints studied here. Automation places the full burden of task completion on the model. It must interpret the prompt, track constraints, produce the final deliverable, and avoid errors without help from another system, so general model capability matters more. We next examine whether these same models retain their relative advantages when they instead assist a fixed downstream worker.

## 4.2 Model Rankings in Augmentation Mode

![](images/201e2a52445521cc5cdb32a55c270e37de90187f90a25d04153fa271fd792711.jpg)  
Figure 8: Augmentation rankings by task, averaged across ten independent runs. For each task, we first average each model’s within-run rank across the ten runs and then rank these mean values to produce the displayed rank of ranks (1 = best). Rankings are computed using unrounded mean ranks. Exact ties receive the same rank, with the subsequent rank skipped (e.g., 1, 1, 3). Models are ordered from newest to oldest from left to right, with GPT-3.5-Turbo (plain), the unaided worker baseline, shown at far right. Lower ranks indicate better performance. Mean ranks and standard errors are reported in Appendix C, Figure 11.

Figure 8 reports model rankings in the augmentation mode across ten independent runs. Each assistant model provides a process guidance in a form of an assistance text to a fixed GPT-3.5-Turbo worker, which then produces the final task output. The figure shows substantial heterogeneity across tasks: no single assistant model consistently ranks first across the task set, and the relative ordering of models shifts across counseling, tutoring, planning, and analytical tasks.

Unlike automation, the best assistant changes substantially across tasks in augmentation. GPT-4.1 ranks first in counseling, GPT-O4-Mini in market-trends analysis, GPT-5-Mini in menu planning and tutoring. However, the unaided GPT-3.5-Turbo worker baseline ranks first in operations research, tax preparation, and travel planning when it is included in the full candidate pool. The best-performing assistant therefore difers by domain rather than converging on a single model, and in some tasks assistant guidance does not improve on the fixed worker’s unaided output, even worsening it.

The pattern also difers by task type. Among the assistant models, GPT-4.1 performs especially well in counseling, consistent with strength in professionally framed guidance and sensitive response structure. GPT-O4-Mini leads in market-trends analysis, where the rubric rewards concise synthesis, specificity, and actionable interpretation. GPT-5-Mini leads in menu planning and tutoring and is the best assistant in tax preparation, suggesting strength on tasks that require constraint tracking, complete deliverables, and structured explanation. GPT-OSS-120B leads among assistants in operations research, where the rubric rewards optimization framing and trade-of reasoning. Claude-Sonnet-4.6 leads among assistants in travel planning, where practical sequencing, budget awareness, and itinerary usability are central. These results suggest that diferent models provide diferent kinds of useful guidance, but the mechanisms require qualitative analysis of the guidance and the resulting worker outputs, which we return to in Section 4.4.

Importantly, these ranking diferences are unlikely to be driven by models excelling on one general evaluation dimension but not another. Across the five general rubric dimensions (instruction following, accuracy and specificity, practical usefulness, organization and readability, and tone/audience fit), most models receive similar relative scores on each dimension rather than specializing on a single axis (Appendix E, Figure 12). Augmentation rankings therefore reflect diferences in overall assistant quality and task fit rather than artifacts of how the general rubric is weighted. Models do not win because they are disproportionately strong on, say, organization but weak on accuracy. The task-specific pairwise outcomes point to genuine diferences in how each model guides the fixed worker.

## 4.3 Automation Rankings Difer From Augmentation Rankings

To quantify the relationship between the two usage modes, let $r _ { m t s } ^ { ( k ) }$ denote the within-task rank of assistant model m on task $t ,$ in usage mode $s \in \{ \mathrm { a u t o } , \mathrm { a u g } \}$ , and independent run $k \in \{ 1 , \ldots , 1 0 \}$ , where lower values indicate better performance. We first average over runs,

$$
\bar { r } _ { m t s } = \frac { 1 } { 1 0 } \sum _ { k = 1 } ^ { 1 0 } r _ { m t s } ^ { ( k ) } .\tag{1}
$$

For each task, we then compute

$$
\rho _ { t } = \operatorname { c o r r } _ { \mathrm { S } } \left( \left\{ \bar { r } _ { m t , \mathrm { a u t o } } \right\} _ { m \in \mathcal { M } } , \left\{ \bar { r } _ { m t , \mathrm { a u g } } \right\} _ { m \in \mathcal { M } } \right) ,\tag{2}
$$

where corr is Spearman’s rank correlation and M contains the nine assistant models evaluated in both usage modes. The unaided worker is excluded because it has no corresponding assistant model in automation. For the model-level comparison, we additionally average $\bar { r } _ { m t s }$ over the seven tasks before computing the same correlation in each usage mode. Because reverse rank is a monotonic transformation of rank, it produces the same Spearman correlations.

![](images/dd4b5916196c3b7b5034c7a0210cb7d78c2e9067f0f0f16b519b2a4bd828a0e2.jpg)  
Figure 9: Model-level comparison of average automation rank and average augmentation rank (assistant models only). Each point represents one assistant model’s average rank across seven tasks and ten independent runs. Ranks are inverted so that higher values indicate better performance. With nine assistant models, reverse rank is defined as $1 0 - { \bar { r } } .$ , where r¯ is the model’s mean rank in the corresponding usage mode. The horizontal axis reports automation performance and the vertical axis reports augmentation performance. The unaided worker baseline is excluded.

Figure 9 compares each model’s average automation and augmentation rank. Across the nine shared assistant models, the Spearman rank correlation between automation and augmentation performance is moderate $( \rho = 0 . 4 8 ;$ two-sided $p = 0 . 1 8 7 )$ , but is not statistically distinguishable from zero at conventional significance levels. GPT-5-Mini is strong in both usage modes and has the best average rank among assistant models in augmentation as well as the best average rank in automation. GPT-4.1 is relatively stronger as an assistant than as a direct solver. Conversely, Claude-Opus-4.8, Claude-Sonnet-4.6, Gemini-3.1-Pro, and

GPT-O4-Mini are stronger in automation than in augmentation. GPT-OSS-120B sits close to the diagonal. Thus, automation performance contains some information about augmentation performance, but it is an incomplete proxy for it.
<table><tr><td>Task</td><td>Spearman  $\rho _ { t }$ </td><td>p-value</td></tr><tr><td>Tax Preparation</td><td>0.85</td><td>0.004</td></tr><tr><td>Menu Planning</td><td>0.64</td><td>0.066</td></tr><tr><td>Tutoring Session Planning</td><td>0.50</td><td>0.166</td></tr><tr><td>Counseling</td><td>0.25</td><td>0.516</td></tr><tr><td>Operations Research Analysis</td><td>0.17</td><td>0.668</td></tr><tr><td>Market Trend Analysis</td><td>0.10</td><td>0.798</td></tr><tr><td>Travel Planning</td><td>-0.04</td><td>0.915</td></tr></table>

Table 1: Task-level correlation between automation and augmentation rankings. Spearman correlations are computed across the nine assistant models using each model’s mean rank over ten runs. Positive values indicate similar cross-mode ordering; values near zero indicate little correspondence. p-values are descriptive because each task contains only nine models.

The strength of this correlation varies substantially by task as shown by Table 1. Tax preparation has the strongest cross-mode correlation $\left( \rho _ { \mathrm { t a x } } = 0 . 8 5 \right)$ , followed by menu planning $( \rho _ { \mathrm { m e n u } } = 0 . 6 4 )$ and tutoring $( \rho _ { \mathrm { t u t o r } } = 0 . 5 0 )$ . The rankings are only weakly associated in counseling $( \rho = 0 . 2 5 )$ , operations research $( \rho = 0 . 1 7 )$ , and market trends analysis $( \rho = 0 . 1 0 )$ and are essentially unrelated in travel planning $( \rho = - 0 . 0 4 )$ . Tax preparation is the only task whose association is statistically distinguishable from zero at the conventional level $( p = 0 . 0 0 4 )$ , including under a Bonferroni correction for seven task-level comparisons. Broader statistical significance across tasks would require a larger model pool or additional runs, and remains an avenue for future work.

Table 4 (Appendix D) makes the divergence concrete. It reports the best-performing model in each task under automation and augmentation, using the lowest mean rank across ten runs. In five of seven tasks, the augmentation winner difers from the automation winner. In counseling, GPT-OSS-120B is the strongest automator while GPT-4.1 is the strongest assistant; in market-trends analysis, Claude-Opus-4.8 automates best but GPT-O4-Mini assists best; and in operations research, Gemini-3.1-Pro automates best while the unaided worker baseline ranks first in augmentation. Menu planning and tutoring are the two exceptions, where GPT-5-Mini ranks first in both usage modes.

The table also shows that augmentation winners are not uniformly frontier assistant models. GPT-3.5-Turbo (plain), the unaided worker with no external assistance, ranks first in operations research, tax preparation, and travel planning. Averaged across all seven tasks, the plain baseline has a mean rank of 3.79; GPT-5-Mini is the only assisted condition with a better overall mean rank of 3.66. We read this not as evidence that guidance is futile, but as evidence that its value is task-contingent and can be negative. Poorly matched or overly complex guidance can leave the worker worse of than working alone. The corollary is central to our argument that strong direct solvers do not automatically make strong assistants, and the right model depends jointly on the role it plays and the task it supports.

Lastly, several models exhibit large and reproducible task-level role reversals. In market trends analysis, Claude-Opus-4.8 is the strongest automator (mean rank 2.05) but one of the weakest assistants (mean rank 8.15). Conversely, in counseling, GPT-4.1 ranks substantially better as an assistant (mean rank 3.80, the best among assisted conditions) than as an automator (mean rank 7.40). These diferences persist after averaging across ten independent runs and therefore are not attributable to a single random run.

The rankings establish that assistance quality is distinct from direct-solving ability, but they do not by themselves explain why particular assistance texts help or hinder the downstream worker model. We therefore turn to the underlying assistance texts, worker outputs, rubric scores, and judge rationales. Section 4.4 examines cases with especially large gaps between ranks to identify recurring diferences between efective and inefective assistance.

## 4.4 Mechanisms: Preliminary Observations on Assistance Quality

To understand what traits make for efective augmentation, we turn to the education literature to ground our observational findings. Wood et al. (1976) coined the term “scafold” to define the process that enables a learner to solve a task or achieve a goal that would be beyond their unassisted eforts. In their framing, as in ours, the expert guides the learner through a task rather than completing it for them. Van de Pol et al. (2010) take this concept further by identifying three essential properties of efective scafolding: (1) contingency - support is calibrated to the learner’s current level, (2) fading - support is gradually withdrawn as competence grows, and (3) transfer of responsibility - the goal is learner autonomy, not dependence. The first and third properties are most directly applicable to our singleinteraction setting, where scafolds cannot adapt over time. We apply this lens to our case in order to evaluate the LLM-generated assistance texts, which are analogous to the scafolds in education. We compare the assistance texts received by the two highest-ranked outputs against those received by the two lowest-ranked outputs across all seven tasks. In doing so, we identify three characteristics of high-quality augmentation scafolds.

<table><tr><td>Task, assistant (rank)</td><td>Assistance-text excerpt</td><td></td></tr><tr><td colspan="3">Trait 1: Specificity beyond the prompt (contingency)</td></tr><tr><td></td><td>Good Market trends, GPT-5-Mini (1)</td><td>“For each trend, plan 2 to 3 succinct bullets: core observation, primary driver(s), and potential market implication...a short preface about methodology and data vintage.&quot;</td></tr><tr><td>Bad</td><td>Market trends, GPT-4.1 (9)</td><td>“Ensure you understand the expectation to discuss how various factors (supply, demand, weather, storage, LNG exports, infrastructure, policy) interact...” (restates the prompt&#x27;s factor list)</td></tr><tr><td>Good Travel planning, Gemini-3.1-Pro (3)</td><td>Trait 2: Explicit structure and ordering</td><td>Enforces assumptions and clarifying questions “before the provisional itinerary,&quot; with named sections in explicit order: Introduction → Budget Breakdown → Daily Itinerary → Conclusion.</td></tr><tr><td>Bad</td><td>Travel planning, GPT-O4-Mini (9)</td><td>“Determine sequence and hierarchy of information for clarity and flow&quot;; instructs worker to“populate each section with neutral placeholders... [Day 1: transportation placeholder]”; no ordering is prescribed.</td></tr><tr><td colspan="3">Trait 3: Not working against the task Good Operations “Present two distinct, practical strategies. For each, explicitly state the</td></tr><tr><td>research, Bad</td><td>Gemini-3.1-Pro (3) Operations</td><td>expected benefits, potential drawbacks, and associated risks.&quot; “Establish a workflow for integrating...trade-offs without delving into</td></tr><tr><td>research,</td><td>GPT-O3-Mini (9)</td><td>specific solutions or analysis&quot;; finalize“without...task-specific analytical content.&quot;</td></tr></table>

Table 2: Illustrative higher- vs. lower-ranked assistance texts for each scafold-quality characteristic. Rows are color-coded by trait; ranks are the aggregate augmentation rank within the task run. Excerpts are drawn from the assistance texts and lightly trimmed.

First, good guidance includes specificity that adds analytical value beyond the prompt, rather than just regurgitating information from the prompt, which is an application of the contingency principle, where efective scafolding is calibrated to the learner’s specific challenge rather than ofered generically (Van de Pol et al., 2010). Table 2 illustrates each trait with a higher- and lower-ranked assistance text. Low-performing guidance messages that simply echo the prompt’s own constraints do not augment the focal worker, who already have access to the prompt. For example, in the market trends task, a top assistance text directs the worker to pair each trend with its primary driver and likely market implication and to note the underlying methodology, whereas a bottom-tier assistance text merely restates the prompt’s own list of factors without adding an analytical step.

Next, embedding explicit structural requirements into the workflow is another trait of a top assistance text. In the travel planning task, a top assistance text enforces assumptions and clarifying questions before the provisional itinerary and prescribes a named section sequence (introduction, budget breakdown, day-by-day itinerary, and conclusion), making the ordering non-negotiable. A bottom-tier assistance text, by contrast, delegates ordering to the worker by instructing it to “determine sequence and hierarchy of information for clarity and flow,” and substitutes neutral placeholders for actual structural guidance.

Third, a negative point that good scafolds avoid: actively working against task completion. Efective scafolding includes what Wood et al. (1976) term direction maintenance, keeping the learner oriented toward the task objective. The most harmful assistance texts in our study invert this function entirely: rather than correcting learner drift, the scafold itself redirects the worker away from the core deliverable, by instructing the worker not to finish the task as described in the prompt. In operations research task, a bottom ranked scafold instructs the worker to establish a workflow for integrating necessary performance indicators, risk discussions, and balanced trade-ofs without delving into specific solutions or analysis, yet the task explicitly requires two practical solutions with trade-ofs stated. This instruction directly prohibits the core deliverable. Thus, no assistance can often beat bad assistance. Notably, the unaided worker is a strong baseline: the plain condition (GPT-3.5-Turbo with no assistance) ranks at or near the top in the many tasks. Poorly calibrated assistance is therefore frequently worse than none, undermining the transfer of responsibility that efective scafolding aims to achieve (Van de Pol et al., 2010).

## 5 Discussion

Our results show that automation and augmentation are not two labels for the same underlying capability. They produce systematically diferent rankings. The top-ranked model difers between automation and assistant-only augmentation in five of the seven tasks. Thus, the assistant value is sharply task-contingent. Interpreting these patterns requires moving beyond leaderboard thinking to role-specific model selection. The automation results are comparatively concentrated around a small set of direct solvers, while GPT-3.5-Turbo ranks last. This ordering characterizes performance on our single-turn professional deliverables and should not be interpreted as reproducing the orderings of agentic benchmarks built around multi-step tool use and environment interaction. Augmentation tells a diferent story. No single model dominates across tasks, average ranks are more dispersed, and several strong direct solvers perform poorly as assistants. Most strikingly, on operations research, tax preparation, and travel planning, the unaided GPT-3.5-Turbo worker outranks every assisted condition, and it has the second best average augmentation rank overall after GPT-5-Mini. We read this not as evidence that assistance is useless, but as evidence that its value is conditional and can be negative. Poorly matched or overly complex process guidance can distract the worker, constrain useful execution, or push it away from task requirements.

## 5.1 Implications for Organizations and Workflow Design

Our framework ofers several practical applications for individuals and organizations adopting AI systems. First, it can support model selection across teams, products, and workflows. Teams seeking a reliable solver for well-defined, repeatable tasks may prioritize automation performance, whereas teams embedding AI into expert workflows, where the goal is to improve human output rather than replace human judgment, may prioritize augmentation ability.

Importantly, the framework is modular. In our experiments, we use GPT-3.5-Turbo as an illustrative fixed worker model to demonstrate the pipeline and hold the worker role constant across augmentation trials. However, individuals and organizations can substitute this worker model with a model, human baseline, or internal workflow proxy that better reflects their own deployment context. For example, a consulting firm could evaluate how diferent assistant models guide junior analysts, domain experts, customer-support agents, or a preferred internal model. This flexibility allows the framework to be adapted to local goals, constraints, and performance standards.

Secondly, designing human-AI collaboration can be done more eficiently and efectively. Beyond model selection, our framework can guide how organizations structure the division of labor between humans and AI. At the organizational level, diferent models can be assigned to diferent subtasks according to where each model’s comparative advantage lies. At the team level, human-AI pairs can use the automation/augmentation profiles to decide which subtasks to delegate entirely versus which to treat as collaborative, keeping humans in the loop where an assistance yields the largest gains. At the individual level, a practitioner can use the same logic to choose the tool best matched to their specific workflow, rather than defaulting to a single general-purpose assistant.

## 5.2 Implications for Evaluation and Research

Our findings also speak to how LLMs should be evaluated. Many existing benchmarks measure automation almost exclusively. If gains on those benchmarks do not transfer to augmentation, then labs and leaderboard builders may be optimizing for only one economically relevant capability. Adopting automation and augmentation as distinct evaluation axes would make model profiles more informative for human-in-the-loop and multi-agent deployment. If automation and augmentation are empirically distinct capabilities, then any evaluation that tests only one is incomplete by design, and our framework ofers a template for fixing that. Leaderboard builders and AI labs optimizing for benchmark performance could adopt the automation vs. augmentation distinction as a structural requirement for capability evaluation.

CentaurBench sits between two literatures. Relative to large CS benchmarks, it evaluates economically grounded occupational tasks under a fixed coaching protocol rather than abstract capability alone. Relative to field experiments, it compares many frontier models across both roles and multiple domains rather than a single tool in a single setting. Relative to marketplace benchmarks such as HAPI (Yi et al., 2025), it holds the downstream worker fixed to isolate assistant quality across models at scale. The appropriate next step is not to replace one paradigm with another, but to combine them for optimal model selection.

## 5.3 Limitations and Future Work

This study introduces a simulation and evaluation framework to measure models’ ability to augment and automate tasks across usage modes. We present this as an initial pilot application; it should not be interpreted as a definitive mapping of model capabilities across tasks and usage modes. The results identify settings in which automation and augmentation rankings diverge and provide preliminary model profiles for subsequent validation. More broadly, the study motivates the development of assistance-oriented benchmarks rather than claiming to provide a comprehensive benchmark itself.

Because the current evaluation relies on LLM judges, the findings constitute scalable simulation evidence rather than definitive human or domain-expert judgments. Although we use general and task-specific rubrics, multiple judges, leave-family-out evaluation, and repeated runs, these procedures cannot eliminate all evaluator bias. Human validation remains a priority, particularly for tasks such as counseling, tax preparation, tutoring, and operations research, where domain expertise and professional judgment are consequential to assess whether assistant advantages observed in simulated settings persist in authentic workplace contexts.

Several extensions are especially important. First, the present design evaluates augmentation through a single assistance text passed to a fixed worker. Real professional workflows are often iterative: assistants clarify requirements, revise plans, and respond to feedback across multiple turns. Extending the framework to multi-turn and agentic settings, in which an assistant contributes throughout an evolving workflow rather than through one-shot guidance, is a natural next step.

Second, all augmentation conditions use the same worker model, GPT-3.5-Turbo. Holding the worker fixed isolates variation in the assistance provided, but it also limits generalizability. Guidance that helps one worker may not transfer to another. A stronger or diferently aligned worker may need less assistance, while a weaker worker may benefit more from structured guidance. Future studies should vary worker capability and model family to test whether assistant rankings remain stable across downstream worker models.

Third, the pilot includes seven professional tasks and ten independent generation-andevaluation runs. Replication improves the stability of the reported estimates, but the task set remains modest relative to the diversity of real-world, occupational tasks. Expanding the evaluation to additional tasks, industries, risk levels, and deliverable formats would provide stronger evidence about the generalizability of role-specific model performance.

External validation should combine domain expertise with authentic workplace tasks. Upwork’s Human+Agent Productivity Index (HAPI) (Yi et al., 2025), for example, emphasizes verified marketplace deliverables and human rubric scoring. Our design instead holds the downstream worker fixed to isolate diferences in assistance quality across models at scale. Future work could combine these strengths by using workplace-grounded tasks, validating a preregistered subset with domain experts, and explicitly comparing automation and augmentation rankings. This would establish whether the preferences of LLM judges correspond to expert assessments of final-output quality, guidance usefulness, and workplace value.

The best mathematicians are not necessarily the best mathematics teachers, and the highestranked tennis players are not necessarily the best coaches. As AI systems increasingly advise humans and coordinate with other models, evaluation must measure not only a model’s ability to automate tasks, but also its capacity to augment the performance of others. This study ofers an initial framework for making that distinction measurable and motivates a broader evaluation agenda organized around the role a model plays, the worker it supports, and the task they jointly perform.

More broadly, realizing the potential of AI to complement human judgment, expand human capability, and support human flourishing will require measures of progress that extend beyond autonomous task completion. Our framework represents one step towards reframing AI progress around not only what systems can accomplish independently, but also how efectively they enable humans and other agents to accomplish even more.

## References

Ajay K. Agrawal, Joshua Gans, and Avi Goldfarb. The Turing transformation: Artificial intelligence, intelligence augmentation, and skill premiums. Working Paper 31767, National Bureau of Economic Research, 2023. URL https://www.nber.org/papers/w31767.

Gagan Bansal, Tongshuang Wu, Joyce Zhou, Raymond Fok, Besmira Nushi, Ece Kamar, Marco Tulio Ribeiro, and Daniel Weld. Does the whole exceed its parts? the efect of ai explanations on complementary team performance. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (CHI), 2021. doi: 10.1145/3411764. 3445717.

Erik Brynjolfsson, Danielle Li, and Lindsey Raymond. Generative AI at work. Quarterly Journal of Economics, 140(2), 2025. doi: 10.1093/qje/qjae044.

Colin Camerer, George Loewenstein, and Martin Weber. The curse of knowledge in economic settings: An experimental analysis. Journal of Political Economy, 97(5):1232–1254, 1989. doi: 10.1086/261651.

Serina Chang, Ashton Anderson, and Jake M. Hofman. ChatBench: From static benchmarks to human–AI evaluation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025.

Zenan Chen and Jason Chan. Large language model in creative work: The role of collaboration modality and user expertise. Management Science, 70(12), 2024. doi: 10.1287/mnsc.2023. 03014.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios N. Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael I. Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating llms by human preference. In Proceedings of the 41st International Conference on Machine Learning, 2024.

Katherine M. Collins, Albert Q. Jiang, Simon Frieder, Lionel Wong, Miri Zilka, Umang Bhatt, Thomas Lukasiewicz, Yuhuai Wu, Joshua B. Tenenbaum, William Hart, Timothy Gowers, Wenda Li, Adrian Weller, and Mateja Jamnik. Evaluating language models for mathematics through interactions. Proceedings of the National Academy of Sciences, 121 (24), 2024. doi: 10.1073/pnas.2318124121.

Fabrizio Dell’Acqua, Edward McFowland III, Ethan Mollick, Hila Lifshitz, Katherine C. Kellogg, Saran Rajendran, Lisa Krayer, Fran¸cois Candelon, and Karim R. Lakhani. Navigating the jagged technological frontier: Field experimental evidence of the efects of artificial

intelligence on knowledge worker productivity and quality. Organization Science, 37(2), 2026. doi: 10.1287/orsc.2025.21838.

Anil R. Doshi and Oliver P. Hauser. Generative ai enhances individual creativity but reduces the collective diversity of novel content. Science Advances, 10(28), 2024. doi: 10.1126/sciadv.adn5290.

Andreas F¨ugener, Dominik Walzner, and Alok Gupta. Roles of artificial intelligence in collaboration with humans: Automation, augmentation, and the future of work. Management Science, 72(1), 2026. doi: 10.1287/mnsc.2024.05684.

Shubham Gandhi, Atharva Naik, Yiqing Xie, and Carolyn Rose. An empirical study on strong-weak model collaboration for repo-level code generation, 2025. URL https://doi. org/10.48550/arXiv.2505.20182.

Michael Hannafin, Susan Land, and Kevin Oliver. Open learning environments: Foundations, methods, and models, volume 2, pages 115–140. Taylor and Francis, 2013. doi: 10.4324/ 9781410603784-12.

Andreas Haupt and Erik Brynjolfsson. Position: AI should not be an imitation game: Centaur evaluations. In Proceedings of the 42nd International Conference on Machine Learning, 2025. URL https://proceedings.mlr.press/v267/haupt25a.html.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In Proceedings of the International Conference on Learning Representations, 2021. doi: 10.48550/arXiv.2009. 03300.

Pamela J. Hinds. The curse of expertise: The efects of expertise and debiasing methods on predictions of novice performance. Journal of Experimental Psychology: Applied, 5(2): 205–221, 1999. doi: 10.1037/1076-898X.5.2.205.

Xiaomeng Hu, Yinger Zhang, Fei Huang, Jianhong Tu, Yang Su, Lianghao Deng, Yuxuan Liu, Yantao Liu, Dayiheng Liu, and Tsung-Yi Ho. Occubench: Evaluating ai agents on real-world professional tasks via language environment simulation, 2026. URL https: //doi.org/10.48550/arXiv.2604.10866.

Harang Ju and Sinan Aral. Collaborating with ai agents: A field experiment on teamwork, productivity, and performance, 2026. URL https://doi.org/10.48550/arXiv.2503. 18238.

Yuetai Li, Yichen Feng, Zhangchen Xu, Zixian Ma, Kaiyuan Zheng, Fengqing Jiang, Xinghua

Sun, Rulin Shao, Zichen Chen, Yue Huang, Xinyang Han, Brian Lee, Kayla Xu, Shenglai Zeng, Hang Hua, Xiangliang Zhang, Basel Alomair, Ranjay Krishna, Luke Zettlemoyer, Pang Wei Koh, Bhaskar Ramasubramanian, Luyao Niu, Xiang Yue, and Radha Poovendran. Jobbench: Aligning agent work with human will, 2026. URL https://doi.org/10.48550/ arXiv.2605.26329.

Zekun Li, Shinda Huang, Jiangtian Wang, Nathan Zhang, Antonis Antoniades, Wenyue Hua, Kaijie Zhu, Sirui Zeng, Chi Wang, William Yang Wang, and Xifeng Yan. Sopbench: Evaluating language agents at following standard operating procedures and constraints, 2025. URL https://doi.org/10.48550/arXiv.2503.08669.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. Transactions on Machine Learning Research, 2023. doi: 10.48550/arXiv. 2211.09110.

Jakub Macina, Nico Daheim, Ido Hakimi, Manu Kapur, Iryna Gurevych, and Mrinmaya Sachan. Mathtutorbench: A benchmark for measuring open-ended pedagogical capabilities of llm tutors, 2025. URL https://arxiv.org/abs/2502.18940.

David Marguerit. Augmenting or automating labor? the efect of ai development on new work, employment, and wages, 2025. URL https://doi.org/10.48550/arXiv.2503.19159.

Shakked Noy and Whitney Zhang. Experimental evidence on the productivity efects of generative artificial intelligence. Science, 381(6654), 2023. doi: 10.1126/science.adh2586.

Siddharth Patwardhan et al. Gdpval: Evaluating ai model performance on real-world economically valuable tasks. Technical report, OpenAI, 2025. URL https://doi.org/10. 48550/arXiv.2510.04374.

Sida Peng, Eirini Kalliamvakou, Peter Cihon, and Mert Demirer. The impact of ai on developer productivity: Evidence from github copilot, 2023. URL https://doi.org/10. 48550/arXiv.2302.06590.

Swarnadeep Saha, Peter Hase, and Mohit Bansal. Can language models teach weaker agents? teacher explanations improve students via personalization. In Proceedings of the 37th International Conference on Neural Information Processing Systems (NeurIPS), 2023. URL https://doi.org/10.48550/arXiv.2306.09299.

Yijia Shao, Humishka Zope, Yucheng Jiang, Jiaxin Pei, David Nguyen, Erik Brynjolfsson, and

Diyi Yang. Future of work with ai agents: Auditing automation and augmentation potential across the u.s. workforce, 2025. URL https://doi.org/10.48550/arXiv.2506.06576.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adri\`a Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023.

Michelle Vaccaro, Abdullah Almaatouq, and Thomas Malone. When combinations of humans and ai are useful: A systematic review and meta-analysis. Nature Human Behaviour, 8, 2024. doi: 10.1038/s41562-024-02024-1.

Janneke Van de Pol, Monique Volman, and Jos Beishuizen. Scafolding in teacher–student interaction: A decade of research. Educational Psychology Review, 22, 2010. doi: 10.1007/ s10648-010-9127-6.

Zhilin Wang, Jaehun Jung, Ximing Lu, Shizhe Diao, Ellie Evans, Jiaqi Zeng, Pavlo Molchanov, Yejin Choi, Jan Kautz, and Yi Dong. Profbench: Multi-domain rubrics requiring professional knowledge to answer and judge, 2026. URL https://doi.org/10.48550/arXiv.2510. 18941.

David Wood, Jerome S. Bruner, and Gail Ross. The role of tutoring in problem solving. Journal of Child Psychology and Psychiatry, 17(2), 1976. doi: https://doi.org/10.1111/j. 1469-7610.1976.tb00381.x.

Yuzheng Xu, Tosho Hirasawa, Tadashi Kozuno, and Yoshitaka Ushiku. Am i more pointwise or pairwise? revealing position bias in rubric-based llm-as-a-judge. arXiv preprint arXiv:2602.02219, 2026. URL https://arxiv.org/abs/2602.02219.

Russell Yang, Ruishi Chen, Pierce Kelaita, Riya Ranjan, Sibo Ma, Charles Dickens, Matthew Guillod, Megan Ma, and Julian Nyarko. Judgmentbench: Comparing rubric and preference evaluation for quality assessment. arXiv preprint arXiv:2605.25240, 2026. URL https: //arxiv.org/abs/2605.25240.

Darvin Yi, Teng Liu, Mattie Terzolo, Lance Hasson, Ayan Sinha, Pablo Mendes, and Andrew Rabinovich. Upbench: A dynamically evolving real-world labor-market agentic benchmark framework built for human-centric ai, 2025. URL https://doi.org/10.48550/arXiv. 2511.12306.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez,

and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Proceedings of the 37th International Conference on Neural Information Processing Systems, 2023.

## A Models and Evaluators

Table 3 lists the assistant models evaluated in automation and augmentation, the fixed worker model used in augmentation, and the LLM-as-judge panel. Model families are grouped to match the leave-one-family-out judging rule described in Section 3.
<table><tr><td>Model</td><td>Provider</td><td>Automation</td><td>Augmentation</td><td>Judge</td></tr><tr><td>GPT-3.5-Turbo</td><td>OpenAI</td><td>V</td><td>Worker</td><td></td></tr><tr><td>Claude-Opus-4.8</td><td>Anthropic</td><td>√</td><td>V</td><td>V</td></tr><tr><td>Claude-Sonnet-4.6</td><td>Anthropic</td><td>√</td><td>V</td><td></td></tr><tr><td>DeepSeek-V3.1</td><td>DeepSeek</td><td>V</td><td>V</td><td>V</td></tr><tr><td>Gemini-3.1-Pro</td><td>Google</td><td>V</td><td>V</td><td>V</td></tr><tr><td>GPT-4.1</td><td>OpenAI</td><td>V</td><td>V</td><td>V</td></tr><tr><td>GPT-OSS-120B</td><td>OpenAI</td><td>√</td><td>√</td><td></td></tr><tr><td>GPT-5-Mini</td><td>OpenAI</td><td>√</td><td>V</td><td></td></tr><tr><td>GPT-O4-Mini</td><td>OpenAI</td><td>√</td><td>√</td><td></td></tr><tr><td>GPT-O3-Mini</td><td>OpenAI</td><td>V</td><td>V</td><td></td></tr></table>

Table 3: Models used in the evaluation and their assigned roles. All focal models serve as direct solvers in the automation regime and as planning assistants in the augmentation regime. GPT-3.5-Turbo serves as the fixed worker in augmentation mode rather than as a focal model. Judges follow a leave-family-out constraint: each judge is excluded from evaluating outputs of models from its own provider family.

## B Task Prompts and Evaluation Rubrics

All task prompts and rubrics are reproduced in full below. Each task is evaluated using the universal judge prompt together with the task-specific rubric dimensions listed here. The augmentation condition additionally uses the universal assistance text prompt to generate planning guidance for the worker model.

## B.1 Universal Assistance Text Prompt

You receive internal assistant guidance plus a client task. Use the guidance for planning only; never copy or restate it. Complete the client task fully and return only the complete final deliverable.

![](images/9edf0a3bf5cca4f4d44a4da6a2d71f12a69901842d85c1a04c81a663a81199b7.jpg)

## B.2 Universal Judge Prompt

## Universal Judge Prompt

You are an expert evaluator comparing two candidate responses to the same task. First, silently score each option from 1–10 on each rubric dimension below. A score of 1 means the response fails that dimension; 10 means it satisfies it excellently. Use the task-specific dimensions as primary criteria and the general dimensions as tie-breakers.

Primary rule: prefer the response that would be more useful, accurate, complete, and reliable for the intended user. Do not reward verbosity unless it improves usefulness.

Task-specific rubric: [insert task-specific rubric below]

## General rubric:

1. Instruction following: satisfies all explicit requirements and constraints.

2. Accuracy and specificity: avoids false claims, vague filler, and unsupported assumptions.

3. Practical usefulness: gives concrete, usable guidance or outputs.

4. Organization and readability: clear structure, easy to scan, appropriate formatting.

5. Tone and audience fit: matches the role, user need, and professional context. 5. Tone and audience fit: matches the role, user need, and professional context.

After scoring, choose the better option overall based on the average score across all rubrics for both options. If both are flawed, choose the one with fewer serious errors. If one response violates a hard safety, legal, medical, or dietary constraint, strongly prefer the other response unless it has an equally serious issue.

Return only: the final choice (Option 1 or Option 2), the score for each rubric dimension (1–10) for both options, and the average score across all rubrics for both options. Structure the output clearly with headings for the final choice, all rubric scores, and the average score.

## B.3 Counseling

## Task Prompt — Counseling

Write a single-session supportive counseling-style response for a client who feels emotionally drained, anxious about work, and uncertain about life direction. The response must include:

1. Empathic validation of the client’s feelings without minimizing them.

2. A cautious identification of possible patterns such as burnout, perfectionism, avoidance, low self-eficacy, or value-direction mismatch, without diagnosing.

3. Evidence-informed framing using CBT, motivational interviewing, and/or positive psychology in plain language.

4. Concrete next steps for stress regulation, thought reframing, values clarification, and near-term goal setting.

5. A brief safety/limits note encouraging professional support if distress is severe or persistent.

6. An encouraging close that reinforces agency and hope.

## Evaluation Rubric — Counseling

Score each dimension from 1 (lowest) to 10 (highest):

1. Empathy and therapeutic tone.

2. Appropriate pattern recognition without overdiagnosis.

3. Evidence-informed psychological framing.

4. Actionable coping and goal-setting recommendations.

5. Ethical boundaries and escalation guidance.

6. Coherence, warmth, and usefulness as a single-session response.

## B.4 Travel Planning

## Task Prompt — Travel Planning

Create a 5-day Tokyo itinerary for one traveler next month with a total budget of \$2,500 USD, excluding passport costs. The traveler is departing from San Francisco. Clearly state assumptions and give clarifying questions before the provisional plan. Use realistic estimates, label them as estimates, and keep the plan budget-aware.

## Evaluation Rubric — Travel Planning

Score each dimension from 1 (lowest) to 10 (highest):

1. Completeness: covers questions, costs, hotels, transport, customs/regulations, itinerary, and budget confirmation.

2. Cost realism and arithmetic: flights, lodging, transport, food/activities, and total are plausible and correctly summed. Penalize outputs that exceed the budget.

3. Hotel and transport practicality: viable hotel options, airport/transit guidance, rail/pass advice.

4. Itinerary quality: 5 days with morning/afternoon/evening plans that are geographically sensible and well-optimized for distance and fatigue.

5. Travel-agent professionalism: clear, customer-focused, with appropriate caveats around estimates.

6. Handling uncertainty: asks clarifying questions for missing information while still providing a useful provisional plan.

## B.5 Meal Planning

## Task Prompt — Meal Planning

Create a cheap, simple 7-day meal plan for a 31-year-old Australian adult with shellfish and gluten allergies, lactose and onion intolerance, and preferences for meat, potatoes, Mexican, Thai, and Indian flavors. Avoid very salty or very sweet foods and account for super-taster sensitivity by using mild, adjustable seasoning. Include breakfast, lunch, dinner, and simple snacks for each day, plus a grocery list and brief prep notes. Do not include shellfish, gluten-containing ingredients, lactose-containing dairy, or onion.

## Evaluation Rubric — Meal Planning

Score each dimension from 1 (lowest) to 10 (highest):

1. Dietary safety: strictly avoids shellfish, gluten, lactose, and onion.

2. Preference fit: uses liked cuisines and foods while avoiding very salty/sweet flavors.

3. Afordability: grocery list items are readily available at common grocery stores and not excessively expensive.

4. Nutritional adequacy and variety: the plan is not overly repetitive and covers proteins, carbohydrates, vegetables, and fruits.

5. Simplicity and usability: easy to follow and clearly structured for the target audience.

6. Completeness: covers breakfast, lunch, dinner, and snacks for each day, plus a grocery list and prep notes. Penalize any missing components.

## B.6 Tax Preparation

## Task Prompt — Tax Preparation

Assume tax year 2025 and state: California. A client brings partially completed federal and California returns. Review the provided data, identify discrepancies, explain the applicable federal/state rules, recalculate key figures using stated assumptions, and explain what forms/schedules need correction. If information is insuficient for an exact tax liability, state what is missing and provide a reasonable estimate or calculation framework.

Client-provided facts:

• W-2 wages: \$68,500.

• Freelance/contract income: \$4,200 reported on Form 1099-NEC.

• Business expenses for freelance work: not yet provided.

• Mortgage interest paid on Form 1098: \$9,800.

• California property tax paid: \$4,800.

• California income tax withheld: \$2,900.

• One dependent listed: a 22-year-old full-time college student who lived with the client for 9 months. The student earned \$18,000 from a part-time job; the client says the student used most earnings for savings and personal expenses, and the client paid more than half of household costs and more than half of the student’s total support.

• Filing status selected by the client: Single.

## Pre-filled portions of the client’s federal and California returns:

• Form 1040 includes W-2 wages of \$68,500.

• Form 1040 includes freelance income of \$2,400 instead of the \$4,200 shown on the 1099-NEC.

• No Schedule C has been prepared.

• No Schedule SE has been prepared, and no self-employment tax appears on Schedule 2.

• Form 1040 Line 12 shows a deduction of \$9,800 labeled as the standard deduction.

• No Schedule A has been prepared even though mortgage interest, property tax, and state tax withholding are available.

• The dependent is claimed as qualifying for the Child Tax Credit.

• The return does not analyze whether Head of Household filing status is available.

• California Form 540 shows federal AGI carried over as \$62,000.

• California tax was computed using the uncorrected federal income figures.

All calculations were done by the client before coming to you. Do not assume the pre-filled entries are correct.

## Evaluation Rubric — Tax Preparation

Score each dimension from 1 (lowest) to 10 (highest):

1. Rule accuracy: correctly applies federal and California rules regarding selfemployment income, dependent rules, mortgage interest, standard deduction, and state/federal diferences without seeking loopholes.

2. Error detection: there are 10 expected discrepancies/issues. Score according to how many are detected (10 detected = 10; 2–9 detected = that number; 0–1 detected = 1). Expected issues:

a. Catches the \$2,400 vs. \$4,200 freelance income mismatch.

b. Identifies missing Schedule C.

c. Identifies missing Schedule SE/self-employment tax.

d. Identifies that \$9,800 mortgage interest is not the standard deduction.

e. Identifies missing Schedule A/itemization comparison.

f. Identifies improper Child Tax Credit treatment for a 22-year-old.

g. Analyzes possible Credit for Other Dependents instead.

h. Analyzes whether Single should be changed to Head of Household given the dependent and household-cost facts.

i. Catches California AGI carryover inconsistency.

j. Catches California tax computed from uncorrected federal figures.

3. Calculation quality: provides income, AGI, deductions, taxable income, selfemployment tax, and estimates where possible.

4. Form guidance: clear guidance on Schedule C, Schedule SE, Schedule A vs. standard deduction, and Form 1040 placement.

5. Dependent analysis: handles the 22-year-old full-time student with \$18,000 income cautiously and correctly.

6. Clear communication: translates dense tax rules into plain English so the client understands what they owe and why.

## B.7 Tutoring

## Task Prompt — Tutoring

Prepare a classroom-ready Grade 3 lesson segment for 8-year-olds on improper fractions and mixed numbers. Include a clear learning objective, simple explanation, analogy, worked examples, common misconceptions, interactive activity, checks for understanding, and a short wrap-up. Use age-appropriate language and avoid abstract notation without visual support.

## Evaluation Rubric — Tutoring

Score each dimension from 1 (lowest) to 10 (highest):

1. Mathematical correctness: accurate representation of concepts with no errors. Penalize any incorrect examples or inaccurate concepts.

2. Age-appropriate explanation and pacing: clear, concise, logically ordered, and avoids abstract notation unsuitable for 8-year-olds.

3. Analogy quality: appropriate and accessible analogy, not too abstract for the target age group.

4. Misconception handling: clearly identifies common errors, debunks them efectively, and provides easy-to-remember strategies for students.

5. Classroom practicality and engagement: designed for immediate classroom use with interactive elements that maintain student interest.

6. Assessment and checks for understanding: includes formative assessment techniques such as thumbs up/down, exit tickets with a visual fraction problem, or think-pairshare.

## B.8 Operations Research

## Task Prompt — Operations Research

You are an Operations Research Analyst at a national logistics firm that manages inventory and delivery for hundreds of retail stores. Delivery delays and rising fuel costs are worrying senior management. They have asked you to identify the key operational bottlenecks and propose data-driven strategies to improve overall eficiency.

Your deliverable is a concise internal report (300–400 words) for executives that:

• Defines the data to collect and how you would validate it.

• Outlines an analytical/optimization framework to evaluate alternate delivery schedules or warehouse allocations.

• Recommends at least two practical solutions that balance cost, service levels, and resource constraints, with trade-ofs and risks stated.

• Specifies KPIs to monitor and explains how results would be communicated to management.

Write clearly, with headings and bullet points where useful.

## Evaluation Rubric — Operations Research

Score each dimension from 1 (lowest) to 10 (highest):

1. Data and validation: identifies route, demand, fleet, warehouse, service, and cost data; explains collection techniques and validation methods.

2. Analytical framework: proposes suitable OR methods such as MILP, simulation, scenario analysis, or routing optimization, formulated clearly enough to be tractable.

3. Recommendations: at least two practical solutions addressing bottlenecks, with clear lines of action.

4. Trade-ofs and risks: acknowledges pitfalls and states cost, service, and resource implications explicitly without overconfidence.

5. KPIs and communication: clear metrics and an executive reporting plan accessible to both technical and non-technical stakeholders.

6. Executive memo quality: concise, within the 300–400 word limit, well-structured, self-contained, and professionally toned.

## B.9 Market Trends Analysis

## Task Prompt — Market Trends Analysis

Imagine you are briefing a client on the U.S. energy market in 2026. Identify exactly three positive or bullish trends and exactly three negative or bearish trends likely to influence natural gas prices this year. Discuss briefly how factors like supply, demand, weather, storage, LNG exports, infrastructure, and policy interact to shape these trends. Conclude with 2–3 sentences summarizing the overall outlook for investors or companies in the energy sector.

## Evaluation Rubric — Market Trends Analysis

Score each dimension from 1 (lowest) to 10 (highest):

1. Trend completeness and balance: identifies exactly three bullish and exactly three bearish trends. Penalize missing, extra, or poorly categorized trends.

2. Economic and market accuracy: correctly explains natural gas market drivers including supply, demand, storage, LNG exports, weather, infrastructure, and policy. Penalize fabricated statistics or implausible causal logic.

3. Causal reasoning and interaction efects: explains how drivers interact rather than listing trends in isolation.

4. Client usefulness and actionability: provides insights useful to investors or energysector firms, including practical implications and risks.

5. Specificity without overclaiming: gives grounded analysis while appropriately caveating uncertainty. Penalize vague filler or confident forecasts without basis.

6. Conclusion quality: ends with a clear 2–3 sentence outlook that synthesizes bullish and bearish pressures for the intended audience.

## C Rank Tables With Standard Error

Construction of uncertainty estimates. For each task–model cell, we report the mean rank across the ten independent runs and its standard error across those ten observations. Specifically, for each model we first average its ranks across the seven tasks within each run, yielding ten run-level average ranks, and then calculate the standard error across those ten values.

<table><tr><td rowspan=16 colspan=1>CounselingMarket TrendsMenu PlanningOperations ResearchTax PrepTravel AgentTutoringAverage</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>3.1±0.5</td><td rowspan=2 colspan=1>7.2±0.4</td><td rowspan=2 colspan=1>5.0±0.4</td><td rowspan=2 colspan=1>5.3±0.4</td><td rowspan=2 colspan=1>2.4±0.4</td><td rowspan=2 colspan=1>2.2±0.5</td><td rowspan=2 colspan=1>3.6±0.5</td><td rowspan=2 colspan=1>7.4±0.3</td><td rowspan=2 colspan=1>8.8±0.2</td><td rowspan=2 colspan=1>10.0±0.0</td><td></td></tr><tr><td rowspan=14 colspan=1>10-987ean rnk-65-4-321</td></tr><tr><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>3.4</td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=1>6.6</td><td rowspan=1 colspan=1>2.7</td><td rowspan=2 colspan=1>7.0±0.3</td><td rowspan=2 colspan=1>9.0±0.0</td><td rowspan=2 colspan=1>10.0±0.0</td></tr><tr><td rowspan=1 colspan=1>±0.3</td><td rowspan=1 colspan=1>±0.7</td><td rowspan=1 colspan=1>±0.2</td><td rowspan=1 colspan=1>±0.4</td><td rowspan=1 colspan=1>±0.5</td><td rowspan=1 colspan=1>±0.4</td><td rowspan=1 colspan=1>±0.5</td></tr><tr><td rowspan=1 colspan=1>4.8±0.7</td><td rowspan=1 colspan=1>3.1±0.2</td><td rowspan=1 colspan=1>2.7±0.4</td><td rowspan=1 colspan=1>6.8±0.3</td><td rowspan=1 colspan=1>1.7±0.3</td><td rowspan=1 colspan=1>7.7±0.5</td><td rowspan=1 colspan=1>4.6±0.6</td><td rowspan=1 colspan=1>7.2±0.5</td><td rowspan=1 colspan=1>6.4±0.7</td><td rowspan=1 colspan=1>10.0±0.0</td></tr><tr><td rowspan=1 colspan=1>3.2±0.4</td><td rowspan=1 colspan=1>2.8±0.3</td><td rowspan=1 colspan=1>6.6±0.5</td><td rowspan=1 colspan=1>4.5±0.5</td><td rowspan=1 colspan=1>5.2±0.7</td><td rowspan=1 colspan=1>2.8±0.6</td><td rowspan=1 colspan=1>3.6±0.7</td><td rowspan=1 colspan=1>7.3±0.3</td><td rowspan=1 colspan=1>9.0±0.0</td><td rowspan=1 colspan=1>10.0±0.0</td></tr><tr><td rowspan=1 colspan=1>3.1±0.6</td><td rowspan=1 colspan=1>3.2±0.5</td><td rowspan=1 colspan=1>4.6±0.6</td><td rowspan=1 colspan=1>8.0±0.6</td><td rowspan=1 colspan=1>1.8±0.4</td><td rowspan=1 colspan=1>5.9±0.5</td><td rowspan=1 colspan=1>4.2±0.5</td><td rowspan=1 colspan=1>7.1±0.4</td><td rowspan=1 colspan=1>7.1±0.5</td><td rowspan=1 colspan=1>10.0±0.0</td></tr><tr><td rowspan=1 colspan=1>4.6±0.5</td><td rowspan=1 colspan=1>7.2±0.5</td><td rowspan=1 colspan=1>2.3±0.4</td><td rowspan=1 colspan=1>6.2±0.6</td><td rowspan=1 colspan=1>1.5±0.2</td><td rowspan=1 colspan=1>5.6±0.9</td><td rowspan=1 colspan=1>6.5±0.9</td><td rowspan=1 colspan=1>5.7±0.5</td><td rowspan=1 colspan=1>5.6±0.7</td><td rowspan=1 colspan=1>9.8±0.1</td></tr><tr><td rowspan=1 colspan=1>3.7±0.5</td><td rowspan=1 colspan=1>6.8±0.8</td><td rowspan=1 colspan=1>3.0±0.6</td><td rowspan=1 colspan=1>5.2±0.6</td><td rowspan=1 colspan=1>2.2±0.5</td><td rowspan=1 colspan=1>4.4±0.7</td><td rowspan=1 colspan=1>4.4±0.5</td><td rowspan=1 colspan=1>7.7±0.3</td><td rowspan=1 colspan=1>7.6±0.5</td><td rowspan=1 colspan=1>10.0±0.0</td></tr><tr><td rowspan=2 colspan=1>3.5±0.2</td><td rowspan=2 colspan=1>5.2±0.3</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>5.6</td><td rowspan=2 colspan=1>2.5±0.2</td><td rowspan=2 colspan=1>5.0±0.3</td><td rowspan=2 colspan=1>4.2±0.3</td><td rowspan=2 colspan=1>7.1±0.2</td><td rowspan=2 colspan=1>7.6±0.2</td><td rowspan=2 colspan=1>10.0±0.0</td></tr><tr><td rowspan=1 colspan=1>±0.2</td><td rowspan=1 colspan=1>±0.2</td></tr><tr><td rowspan=4 colspan=1>Claude-Opus-4.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=1>3.1-Pro</td><td rowspan=3 colspan=1>Sonnet-4.6</td><td rowspan=3 colspan=1>V3.1</td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>OSS-120B</td><td rowspan=3 colspan=1>Mini</td><td rowspan=3 colspan=1></td><td></td><td></td></tr><tr><td rowspan=2 colspan=2>GPT-03-  GPT-3.5-Mini    Turbo(plain)</td></tr><tr><td rowspan=1 colspan=1>Mini</td></tr></table>

Figure 10: Automation mean ranks by task with standard error across ten independent runs.

<table><tr><td rowspan=1 colspan=1>6.2±0.6</td><td rowspan=1 colspan=1>5.0±0.9</td><td rowspan=1 colspan=1>6.1±0.5</td><td rowspan=1 colspan=1>8.2±0.8</td><td rowspan=1 colspan=1>4.2±0.9</td><td rowspan=1 colspan=1>4.4±0.9</td><td rowspan=1 colspan=1>5.7±1.1</td><td rowspan=1 colspan=1>3.8±0.9</td><td rowspan=1 colspan=1>6.7±0.9</td><td rowspan=1 colspan=1>4.7±0.6</td></tr><tr><td rowspan=2 colspan=1>8.1±0.6</td><td rowspan=2 colspan=1>7.1±0.5</td><td rowspan=2 colspan=1>6.4±0.6</td><td rowspan=1 colspan=1>6.1</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>3.1</td><td rowspan=1 colspan=1>4.0</td><td rowspan=2 colspan=1>6.9±0.6</td><td rowspan=2 colspan=1>6.3±0.6</td></tr><tr><td rowspan=1 colspan=1>±1.0</td><td rowspan=1 colspan=1>±0.8</td><td rowspan=1 colspan=1>±1.0</td><td rowspan=1 colspan=1>±0.6</td><td rowspan=1 colspan=1>±0.7</td></tr><tr><td rowspan=1 colspan=1>5.8±0.9</td><td rowspan=1 colspan=1>5.8±0.8</td><td rowspan=1 colspan=1>5.1±0.9</td><td rowspan=1 colspan=1>5.7±0.7</td><td rowspan=1 colspan=1>2.8±0.9</td><td rowspan=1 colspan=1>6.0±1.0</td><td rowspan=1 colspan=1>4.6±0.8</td><td rowspan=1 colspan=1>5.7±0.6</td><td rowspan=1 colspan=1>8.1±0.7</td><td rowspan=1 colspan=1>5.4±0.7</td></tr><tr><td rowspan=1 colspan=1>5.0±0.7</td><td rowspan=1 colspan=1>6.8±0.5</td><td rowspan=1 colspan=1>6.0±0.5</td><td rowspan=1 colspan=1>7.0±0.7</td><td rowspan=1 colspan=1>5.4±1.1</td><td rowspan=1 colspan=1>3.3±0.9</td><td rowspan=1 colspan=1>7.7±0.6</td><td rowspan=1 colspan=1>3.8±0.9</td><td rowspan=1 colspan=1>7.1±0.8</td><td rowspan=1 colspan=1>2.9±0.6</td></tr><tr><td rowspan=1 colspan=1>5.6±0.4</td><td rowspan=1 colspan=1>5.6±0.8</td><td rowspan=1 colspan=1>7.9±0.5</td><td rowspan=1 colspan=1>8.2±0.6</td><td rowspan=1 colspan=1>3.0±0.8</td><td rowspan=1 colspan=1>5.7±0.6</td><td rowspan=1 colspan=1>4.6±0.8</td><td rowspan=1 colspan=1>5.9±0.9</td><td rowspan=1 colspan=1>6.7±0.8</td><td rowspan=1 colspan=1>1.8±0.4</td></tr><tr><td rowspan=1 colspan=1>6.0±0.7</td><td rowspan=1 colspan=1>5.1±0.7</td><td rowspan=1 colspan=1>4.7±0.8</td><td rowspan=1 colspan=1>4.7±0.7</td><td rowspan=1 colspan=1>5.1±1.0</td><td rowspan=1 colspan=1>6.5±1.0</td><td rowspan=1 colspan=1>5.7±0.9</td><td rowspan=1 colspan=1>5.6±0.9</td><td rowspan=1 colspan=1>9.0±0.5</td><td rowspan=1 colspan=1>2.6±0.4</td></tr><tr><td rowspan=1 colspan=1>5.5±0.5</td><td rowspan=1 colspan=1>6.7±0.7</td><td rowspan=1 colspan=1>7.1±0.7</td><td rowspan=1 colspan=1>5.5±0.7</td><td rowspan=1 colspan=1>1.7±0.4</td><td rowspan=1 colspan=1>5.4±1.0</td><td rowspan=1 colspan=1>5.5±0.9</td><td rowspan=1 colspan=1>6.5±0.9</td><td rowspan=1 colspan=1>8.3±0.5</td><td rowspan=1 colspan=1>2.8±0.4</td></tr><tr><td rowspan=1 colspan=1>6.0±0.3</td><td rowspan=1 colspan=1>6.0±0.3</td><td rowspan=1 colspan=1>6.2±0.3</td><td rowspan=1 colspan=1>6.5±0.3</td><td rowspan=1 colspan=1>3.6±0.4</td><td rowspan=1 colspan=1>5.0±0.4</td><td rowspan=1 colspan=1>5.3±0.3</td><td rowspan=1 colspan=1>5.0±0.3</td><td rowspan=1 colspan=1>7.5±0.3</td><td rowspan=1 colspan=1>3.8±0.3</td></tr><tr><td rowspan=1 colspan=2>Claude-  Gemini-Opus-4.8  3.1-Pro</td><td rowspan=1 colspan=4>Claude- DeepSeek- GPT-5-Mini  GPT-Sonnet-4.6  V3.1             OSS-120B</td><td rowspan=1 colspan=4>GPT-04-  GPT-4.1  GPT-03-  GPT-3.5-Mini               Mini    Turbo(plain)</td></tr></table>

Figure 11: Augmentation mean ranks by task with standard error across ten independent runs.

## D Best Model by Task and Usage Mode

Table 4: Best-performing model by task and regime, based on lowest mean rank across ten independent runs.
<table><tr><td>Task</td><td>Automation</td><td>Augmentation</td></tr><tr><td>Counseling</td><td>GPT-OSS-120B</td><td>GPT-4.1</td></tr><tr><td>Market Trends</td><td>Claude-Opus-4.8</td><td>GPT-O4-Mini</td></tr><tr><td>Menu Planning</td><td>GPT-5-Mini</td><td>GPT-5-Mini</td></tr><tr><td></td><td>Operations Research Gemini-3.1-Pro / GPT-OSS-120B</td><td>GPT-3.5-Turbo (plain)</td></tr><tr><td>Tax Prep</td><td>GPT-5-Mini</td><td>GPT-3.5-Turbo (plain)</td></tr><tr><td>Travel Agent</td><td>GPT-5-Mini</td><td>GPT-3.5-Turbo (plain)</td></tr><tr><td>Tutoring</td><td>GPT-5-Mini</td><td>GPT-5-Mini</td></tr><tr><td>Average (all tasks)</td><td>GPT-5-Mini</td><td>GPT-5-Mini</td></tr></table>

## E General Rubric Profiles

![](images/12bf29df7ae25626064ce0addfabc073b0eb371c14429577514e5e632bf32e4d.jpg)  
Figure 12: Average general-rubric scores in the augmentation mode, aggregated across tasks and judges. Scores are shown for instruction following, accuracy and specificity, practical usefulness, organization and readability, and tone/audience fit. Profiles are relatively flat across dimensions for most models, indicating that ranking diferences are not driven by specialization on a single general rubric axis.

## F Validity of LLM-as-Judge Evaluation

A limitation of this study is that final evaluations are conducted by LLM judges rather than by domain experts or crowdworkers. We use LLM judges because the framework is intended as a scalable pilot-stage evaluation pipeline across many models, tasks, and usage modes. We use multiple judges and leave-family-out restrictions to reduce self-preference and model-family bias. A model is not used to judge outputs from its own model family. We then aggregate rankings across eligible judges. This does not eliminate all judge bias, but it reduces the most direct form of self-evaluation and allows judge-level disagreement to be inspected. Therefore, we do not treat LLM judgments as ground truth. Instead, we implement several checks designed to assess whether the judging procedure is internally coherent and directionally valid.

First, all judges use explicit task-specific and general rubrics rather than unconstrained preference judgments. Each pairwise comparison requires the judge to score both responses across micro-rubric dimensions before selecting a winner. This design makes the basis for each judgment auditable and allows us to compare final choices against the underlying rubric scores.

![](images/bb97654739ebf7943e8cfa28a2ec28fc0cc2caab09ac84a905ad5d02b68f8f92.jpg)

![](images/7c571e6ef6066e1908f095c8da2125787af872e8223f8e22470047a768350bfc.jpg)  
Figure 13: Validation of the rubric-guided pairwise judging procedure across ten independent runs. Panel A compares each model–task–mode–judge cell’s mean rubric score with its pairwise win rate $( r = 0 . 5 3 ; n = 4 , 2 0 0 \mathrm { c e l l s } )$ . Panel B reports, for each judge, the percentage of non-tied comparisons in which the final pairwise choice agrees with the option receiving the higher average rubric score. Agreement ranges from 98.9% to 100.0% across judges.

Second, we test whether pairwise choices align with rubric scores. Across 15,120 non-tied scored comparisons, judges select the option with the higher average rubric score in 99.7% of cases. Choice–score consistency ranges from 98.9% to 100.0% across the four individual judges. At the model–task–mode–judge level, mean rubric scores are positively correlated with pairwise win rates (r = 0.53 across 4,200 cells). This indicates that the pairwise tournament rankings are directionally consistent with the underlying dimension-level evaluations.

Third, we separately assess inter-judge reliability on output comparisons evaluated by at least two eligible judges. The leave-family-out design creates a partially crossed panel because a judge is ineligible whenever a comparison contains an output from its own model family. We consequently report both raw agreement percentage and Krippendorf’s nominal alpha, which accommodates missing judge and adjusts for chance agreement. Before calculation, we map each choice to a canonical output-pair order so that randomized presentation as Option 1 or Option 2 cannot afect agreement. We retain one evaluation per judge and comparison, preventing comparisons with repeated evaluations from receiving additional weight.

![](images/ec4a8ff81498f89d2a4e353c70388e0be48194b356472c6ec38594a902d0a8ff.jpg)

![](images/c3429da98661ff6181186b36caba474f64b2d20dc15e6777fe97db30f3638be2.jpg)  
Raw agreement is the share of eligible judge-pair decisions selecting the same canonical winner. Alpha adjusts for chance agreement.

Figure 14: Inter-judge reliability across ten independent runs. Panel A reports raw agreement and Krippendorf’s nominal alpha overall and separately for augmentation and automation. Raw agreement is the share of eligible judge-pair decisions selecting the same winner; alpha adjusts for chance agreement and accommodates missing ratings induced by leave-family-out evaluation. Panel B reports alpha by task and usage mode. Only comparisons evaluated by at least two eligible judges are included.

Across 6,265 output comparisons evaluated by at least two eligible judges, the panel yields 11,257 judge-pair decisions. Judges select the same winner in 71.0% of these decisions, with Krippendorf’s nominal α = 0.414. Reliability is higher in automation (74.5%; α = 0.448) than in augmentation (67.8%; α = 0.354). Thirty-five comparisons with fewer than two valid ratings are excluded from the reliability calculation.

Task-mode reliability varies substantively. Krippendorf’s alpha ranges from 0.231 for operations research augmentation to 0.615 for market-trends automation. Automation produces higher agreement than augmentation overall and in every task shown in Figure 14. These results indicate a meaningful but moderate common signal across judges rather than nearconsensus. The lower agreement in augmentation also suggests that evaluating the downstream efects of assistance texts is more judge-dependent than evaluating direct task outputs. This finding further motivates the human-expert validation proposed in the limitations section.

Fourth, we conduct criterion-validity checks using automation outputs, where external expectations about model capability are clearer. Figure 15 reports two held-out-judge checks. First, Claude-Opus-4.8 evaluates GPT-family automation outputs from GPT-3.5-Turbo, GPT-O3-Mini, GPT-4.1, GPT-O4-Mini, and GPT-5-Mini. Across ten runs and seven tasks, it recovers the prespecified release-order gradient in 89% of ordered model-pair comparisons (624 of 700), with GPT-5-Mini receiving the best mean within-family rank. Second, GPT-4.1 evaluates Claude-Sonnet-4.6 and Claude-Opus-4.8. GPT-4.1 prefers Claude-Opus-4.8 in 62.9% of the 70 task–run comparisons, and Opus receives the better mean within-family rank (1.37 versus 1.63). These checks provide an internal directional validation that the judging procedure recovers the prespecified within-family gradients used in this exercise. They, however, do not imply that the complete automation ordering should reproduce external benchmark rankings.

A. GPT-family outputs judged by Claude-Opus-4.8  
![](images/6d200ffa6a38217684622ea49315dd4f100491428cccd6eb303464dbd49b6513.jpg)

B. Claude-family outputs judged by GPT-4.1  
![](images/af4ec5151dc100fff7a832974c3645d495112a9d7ec24acf961da51ffd19bda0.jpg)  
Figure 15: Criterion-validity checks using automation outputs across ten independent runs. Panel A reports mean within-family rank for GPT-family models judged by Claude-Opus-4.8; the judge recovers the prespecified release-order gradient in 89% of ordered comparisons. Panel B reports mean within-family rank for Claude-family models judged by GPT-4.1; Claude-Opus-4.8 is preferred in 62.9% of task–run comparisons. Error bars show one standard deviation across task–run cells. Lower ranks indicate better performance.

Finally, we preserve the full evaluation trace: rompts, assistance texts, final outputs, pairwise choices, rubric scores, and judge rationales. This makes the results inspectable and readers can examine whether rankings are supported by the underlying judgments and whether specific task-level outcomes appear substantively reasonable. It is important to stress that even with such robustness checks, they do not substitute for human expert validation. The strongest future validation would compare LLM judge rankings against domain expert rankings for a subset of tasks, especially tax preparation, counseling, tutoring, and operations research. We therefore interpret the current results as a scalable simulation-based benchmark and hypothesis-generation tool, and not as a final human-validated measure of workplace value.

## G Outputs for GPT-4.1 and DeepSeek-V3.1

<table><tr><td>GPT-4.1 (Rank 1 augmentation) Run 1 Dear [Client],</td></tr><tr><td>I want to first acknowledge how challenging it must be to feel emotionally drained, anxious about work, and uncertain about the direction of your life. It's completely valid to feel this way, and you’re not alone in experiencing these overwhelming emotions. As we explore what you're going through, it's important to consider potential patterns that might be contributing to your current state. It could be that burnout, perfectionism, avoidance of difficult emotions, low self-efficacy, or feeling disconnected from your values are playing a role. However, please know that these are just possibilities to consider, and not labels or diagnoses. In navigating these feelings, evidence-based approaches like Cognitive Behavioral Therapy (CBT), motivational interviewing, and positive psychology can offer valuable insights. CBT can help</td></tr><tr><td>you reframe negative thought patterns and develop healthier coping strategies. Motivational interviewing can guide you in exploring your values and goals, while positive psychology can help you focus on your strengths and sources of resilience. Moving forward, I encourage you to take concrete steps to manage stress, reframe unhelpful thoughts, clarify your values, and set achievable goals. This might involve practicing mindfulness or relaxation techniques, challenging negative self-talk, reflecting on what truly matters to you, and setting small, manageable objectives to work towards. If at any point you feel overwhelmed or find it difficult to cope, please remember that it's okay</td></tr><tr><td>DeepSeek-V3.1 (Rank 10 augmentation) Run 1</td></tr><tr><td>Dear [Client], I want to first acknowledge the weight of the emotions you're carrying right now. Feeling emotionally drained, anxious about work, and uncertain about life's direction can be incredibly overwhelming. It's important to recognize and honor these feelings without downplaying their impact on you. In exploring your current state, it's possible that patterns like burnout, perfectionism, avoidance,</td></tr><tr><td>understanding these patterns can shed light on the challenges you're facing. Drawing from evidence-informed approaches like Cognitive Behavioral Therapy (CBT), Motiva- tional Interviewing, and Positive Psychology, we can work together to reframe your thoughts, regulate stress, clarify your values, and set achievable goals. By breaking down these overwhelm- ing feelings into manageable steps, we can pave a path towards a more fulfilling and balanced life.</td></tr><tr><td>It's essential to prioritize your well-being, so if the distress you’re experiencing feels severe or persistent, seeking professional support can provide additional guidance and resources. Your mental health is paramount, and reaching out for help is a courageous step towards healing. As we navigate this journey together, remember that you hold the power to shape your future. Your agency and resilience are remarkable strengths that will guide you towards a brighter tomorrow. Stay hopeful, stay strong, and know that I'm here to support you every step of the</td></tr></table>

[Your Name]
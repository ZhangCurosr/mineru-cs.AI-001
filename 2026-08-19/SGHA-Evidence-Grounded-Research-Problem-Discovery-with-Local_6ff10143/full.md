# SGHA: Evidence-Grounded Research Problem Discovery with Local Language Models

Sarvesh Gharat

Junpei Komiyama

C-MInDS, IIT Bombay

Machine Learning Department, MBZUAI

sarveshgharat19@gmail.com

junpei.komiyama@gmail.com

## Abstract

Recent eforts toward fully automated AI scientists have demonstrated that language-model agents can generate hypotheses, execute experiments, and draft scientific manuscripts. However, during the early stages of research, when research problems are formulated, these AI scientists often rely heavily on proprietary frontier models. Their proposals are shaped by opaque parametric knowledge and by literature searches conditioned on the proposals themselves. Such knowledge is efectively a black box, and this dependence makes the evidential basis and validity of generated research problems dificult to audit and leaves the process vulnerable to model-specific hallucinations and biases. Furthermore, if proprietary research materials are transmitted to external APIs, the use of these models creates confidentiality, privacy, and data-governance concerns.

We introduce the Structural Gap Hypothesis Agent (SGHA), a fully automated, corpus-first research-problem discovery system that runs entirely on a local LLM. SGHA structures a scientific literature corpus into evidence-linked paper objects and a typed evidence graph, detects unresolved structural patterns across papers, screens candidate gaps before formulation, and produces traceable research-problem families. In particular, it is able to output assumptions, objectives, success criteria, and remaining ambiguities. All LLM-based components of SGHA are executed using a locally served open-weight 9B language model, without requiring proprietary frontier-model APIs. We compare SGHA with the AI Scientist-v2 idea formulation module in five machine-learning domains. Our results suggest that explicit corpus structure and evidence-constrained reasoning can support promising, inspectable research-problem formulation without relying on frontier models during generation or verification.

<sup>§</sup> github.com/SarveshVGharat/structural-gap-hypothesis-agent

## 1 Introduction

Recent language-model agents can generate research ideas and automate parts of the scientific workflow, including writing code, running experiments, analyzing results, and preparing manuscripts [1, 2, 3, 4, 5, 6, 7]. Several end-to-end systems begin by generating candidate ideas within a user-provided topic or experimental scafold and then try to develop them. In this paper, we study a diferent starting point. We ask whether an automated system can identify research problems that are already suggested by unresolved patterns in the scientific literature, even when no single paper states them directly.

Many research questions only become visible when several papers are considered together. A set of methods may fail under the same condition, several guarantees may rely on the same strong assumption, or a limitation may be reported repeatedly without being turned into a concrete research problem. In such cases, the evidence for a problem is already present, but it is distributed across papers.

![](images/a944553d3263d07e18b105f384ce2a8ddf1f7e8136733e54858be57d13c70764.jpg)  
Figure 1. Overview of the SGHA pipeline. “Deterministic” denotes non-LLM stages. The evolutionary branch is optional and is reported separately from the verification-gated family outputs. Evolutionary operations are iterated multiple times and fed into the problem synthesis stage.

As a simple example, consider the literature on stochastic multi-armed bandits from 1985 through 2011. Researchers try to characterize the performance of an algorithm in terms of the cumulative reward shortfall called regret. A popular algorithm based on an upper confidence bound achieves O(log T) regret for bounded rewards using a concentration inequality [8]; this rate is optimal [9]. Yet its extension to heavy-tailed rewards was not well known. Bubeck et al. later filled this gap using robust estimators, and Agrawal et al. subsequently improved and clarified the connection to bounded-reward models [10, 11]. The research problem of characterizing regret under heavy-tailed rewards is what we call a structural gap: a pattern that is present in the literature but not yet resolved. SGHA emulates this process prospectively by detecting and verifying analogous corpus-level gaps and formulating them as evidence-grounded research problems.

Existing systems either condition hypothesis generation on a research background comprising an explicit research question and a survey of prior work [12], or formulate candidate research problems, directions, or hypotheses within a search space anchored by a human-authored code template (AI Scientist v1) [2], a broad topical description (AI Scientist v2) [3], a scientist-specified research goal [5], a selected core paper [7], or an input topic used for graph-grounded literature retrieval [13]. ResearchAgent explicitly includes an LLMbased problem-identification step, but generates the problem from a core paper, its citation neighborhood, and retrieved entities. Very recently, Graph2Idea retrieves papers according to an input topic, transforms them into structured knowledge triples, dynamically constructs a target-centered knowledge graph, and then plans research directions before synthesizing complete ideas [13]. To our knowledge, these systems do not integrate all three capabilities: typed structural-gap detection over a topic-bounded full-text corpus, corpus-level verification before formulation, and passage-level provenance for evidence, counterevidence, and explicit ambiguities. Beyond base-model capability, deployment is an independent practical consideration. In their reported implementations or principal experimental configurations, several prominent systems use proprietary models or hosted inference services for central idea generation or refinement [2, 7, 3, 5], limiting exact reproducibility and model-level auditability relative to locally served open-weight inference. When unpublished or sensitive research materials are transmitted to external services, this additionally introduces deployment-dependent confidentiality, privacy, and data-governance concerns.

We introduce SGHA, the Structural Gap Hypothesis Agent, a training-free system for generating candidate research-problem families from scientific papers. Figure 1 summarizes the pipeline. Given a topic, SGHA retrieves and parses relevant papers, extracts evidence-backed scientific tuples and paper-level objects, and builds a typed evidence graph linked to supporting passages. It then searches this graph for candidate structural gaps. Only gaps that pass a hard verification gate proceed to formulation. SGHA first writes a direct formulation close to the evidence, then generates broader variants and uses a separate critic to reject variants that are unsupported or only rhetorically stronger. Surviving formulations are grouped into project families and converted into semi-formal problem statements that describe the main variables, assumptions, feedback or observation model, objective, success criteria, and unresolved ambiguities. When the evidence is too weak, SGHA reports a low-signal outcome rather than forcing a research menu.

SGHA also supports two complementary modes. A profile-conditioned mode uses a researcher’s literature context and an optional target topic to prioritize problems relevant to that context. An evolutionary exploration branch iteratively broadens literature-grounded seeds to produce additional candidate directions. These modes use the same underlying goal: to keep generated problems tied to evidence rather than treating them as unconstrained prompt completions.

We evaluate SGHA across five machine-learning domains: bandits, in-context learning, reasoning and test-time computation, ofline reinforcement learning, and uncertainty estimation. The main experiments use a frozen local Qwen/Qwen3.5-9B model for generation, with no task-specific training or fine-tuning. Across 1,250 selected papers, SGHA extracts structured content from 1,044 papers. After verification, 39 candidate gaps remain and lead to 15 formalized project families. Our primary comparison is with the AI-Scientist-v2 ideation workflow configured with the same local model and matched output counts. This is a system-level comparison rather than an information-matched ablation: SGHA uses the bounded corpus, whereas the AI-Scientist-v2 retains its Semantic Scholar retrieval. Since SGHA is meant to write research-problem formulations rather than full project plans, we evaluate the comparison with a formulation-only rubric. Under this rubric, a method-label-masked evaluation using five frontier-model judges from diferent providers assigns SGHA higher mean scores on all but one of the formulation-quality criteria, including well-posedness, assumption-boundary clarity, formalizability, source-grounded specificity, and ambiguity handling. The local AI-Scientist-v2 baseline receives a higher mean score only on scope control, consistent with its tendency to produce narrower proposal-style ideas. The hosted judges are used only for post hoc evaluation after the candidate outputs have been frozen and are not part of SGHA’s generation or verification pipeline. Additiona comparisons using Claude Opus in the AI-Scientist-v2 workflow and the public MOOSE-Star HC model [12] were conducted. In the Claude Opus diagnostic, which matches retained output counts but not compute or information budgets, SGHA receives a competitive numerical mean score on overall formulation quality to AI-Scientist-v2 with Claude Opus.

We additionally report exploratory diagnostics on profile conditioning, model configuration, corpus size, and evolutionary-output availability, treating them as capability or sensitivity analyses rather than controlled comparisons.

## Our main contributions are:

## Contributions

Corpus-first research-problem discovery. We present SGHA, an end-to-end pipeline that detects unresolved cross-paper patterns in a bounded corpus and converts verified gaps into evidencegrounded families of research problems. All generation and verification stages run on a locally served open-weight LLM without requiring proprietary frontier-model APIs.

Verification before formulation. SGHA does not formulate a research problem from every detected pattern. It first screens each candidate for evidential support and prior coverage within the bounded corpus, formulates only those that pass verification, and uses a separate critic to reject unsupported ambition-expanded variants.

• Traceable problem formulations. SGHA outputs problem specifications with explicit assumptions, objectives, and success criteria. Each specification retains its source motif type and includes passagelevel evidence and counterevidence links together with explicit ambiguity annotations.

Empirical analysis of quality, breadth, and failure modes. We evaluate SGHA across multiple domains with a local model, compare it primarily with a locally configured AI-Scientist-v2 idea generation workflow, and report exploratory diagnostics on profile conditioning, cross-model behavior, corpus size, low-signal behavior, and evolutionary-exploration availability.

Related Work on Automated Research Ideation with Local LLMs

Open-weight local LLMs have been used in automated scientific ideation. SciMON fine-tunes T5-large and, in a biochemical case study, Meditron-7B on paper-derived supervision, while its reported iterative novelty-boosting experiments use GPT-4 [14]. HypER uses a Llama-3.1-70B teacher to construct and score citation chains and LoRA-tunes Phi-3 Mini 3.8B for relevance and chain-validation tasks; hypothesis generation is then performed at inference time, conditioned on validated chains [15]. MOOSE-Star fine-tunes R1-Distill-Qwen-7B models for inspiration retrieval and hypothesis composition using supervision generated with a locally deployed 32B teacher [12]. These locally executable open-weight instantiations rely on task-specific fine-tuning or distillation, and they do not explicitly model research problem discovery as detecting and verifying unresolved structural gaps across a bounded literature corpus. HypoGeniC can run with an open-weight Mixtral-8x7B model and updates a bank of predictive natural-language hypotheses from labeled examples without fine-tuning the generator; it does not search a literature corpus for research problems [1].

Among the most closely related prior works, Problem Discovery via Structural Motifs in Knowledge Graphs (PRISM) [16] explores structural-motif-based problem discovery by detecting structural gaps in a paper corpus, converting graph motifs into structured problem objects, and refining them with a novelty-and-feasibility critic. Its expert-evaluation protocol is proposed rather than carried out, leaving open whether detected motifs represent genuine unresolved gaps rather than retrieval or extraction artifacts or questions already covered elsewhere in the corpus. SGHA partially addresses this issue within the bounded corpus by verifying candidate gaps before formulation, merging redundant directions, and making remaining ambiguities explicit.

## 2 Structural Gap Hypothesis Agent

SGHA is designed around a simple constraint: a generated research problem should be tied to evidence in the literature. Given a topic, the system first builds a structured evidence base from papers, then looks for recurring patterns in that evidence, verifies the resulting candidate gaps, and only then formulates research-problem families. This is diferent from asking a language model to directly generate ideas from a topic. The language model is used in specific stages, while graph construction, motif detection, verification gating, family consolidation, and final auditing keep the pipeline grounded.

The overview figure in the introduction shows the full pipeline. In this section, we describe the main objects that move through it: evidence tuples, typed evidence graphs, structural motifs, candidate gaps, verified gaps, project families, and formal problem statements. We use one Bandits example as a running reference. In the selected corpus, SGHA extracts evidence that difusion-based Thompson sampling (dTS) is analyzed under linear reward or link-function assumptions and bounded contexts, while related evidence points to nonlinear score functions requiring approximation. This trace later leads to the project family Characterizing the Non-Convex Failure Regime of Difusion-Based Contextual Bandits. Section 4.4 shows the final generated formulation, proposal-style abstract, formal problem object, risks, and ambiguity flags for this example.

## 2.1 Building the Evidence Base

SGHA begins by constructing a topic-specific corpus. We distinguish three paper-level stages. In stage one, selected papers are chosen for the corpus before parsing. Next, parsed papers are selected papers for which usable text is obtained, and finally, the extracted papers are parsed papers for which SGHA produces a valid structured extraction.

For each extracted paper, SGHA records both paper-level objects and relation-level evidence. The paper-level object contains methods, tasks, datasets, metrics, assumptions, results, limitations, failure conditions, claims, contradictions or tensions, and future work. These fields are useful because not every important statement in a paper becomes a relation edge in the graph.

The relation-level representation consists of evidence tuples of the form

$$
( { \mathrm { s u b j e c t } } , { \mathrm { r e l a t i o n } } , { \mathrm { o b j e c t } } ) ,
$$

with source provenance. Relations include assumes, fails\_under, limited\_by, addresses, not\_addressed\_by, contradicts, improves\_over, evaluated\_on, uses\_dataset, and measured\_by. Each tuple also stores supporting text, paper section, claim type, polarity, condition, evidence type, confidence, and source-paper identifier.

These tuples are local evidence records extracted from individual papers. Before SGHA uses them downstream, it validates them by removing invalid relations, unsupported spans, missing evidence, generic subjects, and relation-specific errors. This keeps the evidence base from being dominated by vague labels such as “our method” or claims that are not supported by the parsed text.

These tuples are local evidence records extracted from individual papers. Before SGHA uses them in the graph, it applies deterministic validation and cleanup. First, the relation must be one of the allowed relation types, and the subject, object, and evidence fields must be present. Second, the supporting evidence span must match the parsed paper text after normalization, so that the tuple can be traced back to the source paper. Third, SGHA removes generic subjects such as "method", "algorithm", or "our method", because these labels are too vague to support graph construction. It also filters overly generic task labels, drops evaluation tuples that do not include task, dataset, or metric context, and normalizes metadata such as relation polarity. For example, fails\_under and limited\_by are treated as negative relations, while assumes is treated as neutral. These checks keep the evidence base tied to specific paper text and prevent vague or unsupported tuples from driving later gap detection.

## 2.2 From Evidence Tuples to a Typed Graph

SGHA then inserts the validated paper objects and evidence tuples into a typed evidence multigraph. The graph has diferent kinds of nodes for diferent scientific objects. Paper nodes represent source papers, while other nodes represent methods, assumptions, limitations, failure conditions, claims, tasks, datasets, metrics, and results. The edges come from the extracted tuples. For example, an edge may state that a method assumes an assumption, fails\_under a condition, or is measured\_by a metric. Each edge keeps its source provenance, including the paper identifier and supporting evidence span. Thus, the graph is not just a citation graph or a collection of paper summaries. It records the scientific relations that SGHA extracted from the corpus: what methods assume, where they are evaluated, what they improve over, where they fail, and which limitations remain unresolved.

Figure 2 shows this construction for the running Bandits example. The left side shows the surrounding literature context and highlights the two source papers [17, 18] used in the trace. The right side shows the typed local graph built from the extracted tuples. The method node dTS is connected to assumption nodes for linear rewards, bounded contexts, and linear link functions. It is also connected to an open issue around nonlinear difusion models and to a failure-condition node for nonlinear score functions. This local graph is the evidence structure that later triggers the assumption–failure motif.

## 2.3 Structural Motifs and Candidate Gaps

Once the evidence graph is built, SGHA searches it for predefined graph patterns, which we call structural motifs. A motif is a small pattern of relations that may point to a possible research gap. For example, a method may depend on an assumption in one part of the literature and appear to fail when a related condition changes in another part of the literature. When SGHA finds such a pattern, it creates a candidate structural gap. This candidate is not yet a verified research problem, and is only a place in the literature graph that deserves closer inspection.

Figure 3 shows several motif families used by SGHA. An assumption–failure motif links a method to an assumption and to a related failure condition. A shared-failure motif appears when several methods fail under the same condition. A repeated-unaddressed-limitation motif appears when a limitation is mentioned across papers without evidence that it has been addressed. Other motifs capture conflicting claims, sparse evaluation, unrealistic assumptions, missing stress tests, and theory–practice gaps. These motifs are deliberately simple: their role is to find candidates for later screening and verification, not to decide on their own that a research problem is valid.

In the Bandits example, the local graph links dTS to assumptions that make posterior approximation tractable, and also to a nonlinear score-function condition where approximation becomes necessary. This matches the assumption–failure motif. The resulting candidate gap can thus be read as follows: dTS is analyzed under linear and bounded assumptions, but nearby nonlinear score-function settings may violate those assumptions. At this stage, SGHA has only identified a structured pattern in the literature. The candidate still needs to pass novelty screening and verification before it can be formulated as a research problem.

![](images/a24eebb67124c1fbba670b747b3a4b16ae981419dad44bb4f6cf92e3df9a9555.jpg)  
Figure 2. Evidence graph view of the Bandits worked example. The left side shows the surrounding literature context and the two source papers used in the worked example. The right side shows the typed local evidence graph: dTS is linked to linearity and bounded-context assumptions, an open issue around nonlinear difusion models, and a nonlinear score-function failure condition. This local graph supports the assumption–failure candidate gap used later in the qualitative result.

## 2.4 Verification Before Formulation

Such candidate gaps first enter a checking stage. The goal of this stage is to turn a graph-derived pattern into a formulation-ready gap. SGHA does this in two steps: a corpus-level screen followed by role-based verification.

The corpus-level screen keeps candidates that remain meaningful after checking nearby evidence in the selected corpus. Candidates retained by this screen are called novelty survivors. In SGHA, this term has a specific meaning: the candidate has passed the system’s internal corpus screen and is worth deeper review.

SGHA then sends a selected subset of novelty survivors to role-based verification. These are the reviewed gaps. The verification agents examine each reviewed gap from complementary perspectives. The support agent identifies evidence that the gap is grounded in the corpus. The skeptic agent looks for nearby counterevidence or possible resolutions. The feasibility agent checks whether the problem can plausibly be studied. The mechanism agent asks whether there is a reasonable explanation for the gap. The verification critic gives a conservative final assessment.

A reviewed gap becomes a verified gap when it passes the hard verification gate. This gate is deterministic. It admits candidates with the required agent outputs, no disqualifying parse failures, a survival score above threshold, and a non-rejecting critic with suficient confidence. Verified gaps, then enter the direct-formulation stage.

In the running Bandits example, this stage keeps the assumption–failure pattern grounded but also sharpens its interpretation. The support agent finds evidence for the linear-assumption issue in dTS. The skeptic and critic add useful context: approximation-based approaches are mentioned in the source papers, so the fina problem should frame the issue carefully rather than overstate it. The feasibility and mechanism agents find the problem studyable and mechanistically plausible. The hard gate passes the candidate, allowing SGHA to formulate it. In this way, verification turns the graph-derived dTS candidate into a formulation-ready gap.

![](images/92fbb7cf872148c67e738315af67f7f31eebf0aa5c75cbbfd7691a044b5c58e6.jpg)  
Figure 3. Examples of structural motifs used for candidate-gap discovery. Each mini-panel shows a deterministic graph pattern queried over the typed evidence graph. A motif creates a candidate gap, not a final research problem. SGHA later screens, verifies, formulates, and consolidates these candidates before they can enter the final output.

## 2.5 From Verified Gaps to Project Families

Once a gap is verified, SGHA treats it as an evidence-backed starting point for formulation. The first step is a direct formulation. This formulation stays close to the verified gap: it states the main setting, the central assumption or failure, the objective, and a plausible theorem, algorithm, or benchmark target. It also keeps the source gap, motif type, supporting papers, and verification provenance attached to the formulation.

In the running Bandits example, the direct formulation is Extending dTS to Non-Linear Score Functions in Contextual Bandits. This is intentionally close to the verified evidence. It asks how difusion-based posterior inference can be extended beyond linear score functions while preserving useful regret guarantees or approximation behavior.

SGHA then broadens the direct formulation in a controlled way. It generates conservative, generalized, and bold variants. The conservative variant makes a small extension of the direct problem. The generalized variant broadens the problem class or assumptions while keeping the formulation grounded. The bold variant asks whether the evidence points to a deeper boundary, impossibility, or failure-regime question.

This expansion stage is paired with an independent formulation critic. The critic checks whether a variant makes a real scientific move, rather than only sounding more ambitious. It looks for source support, specificity, non-incrementality, and a clear change in the problem class or objective. Variants that pass this critic become candidates for final project families.

In the Bandits example, this step changes the formulation from a local extension of dTS to a broader boundary question: Characterizing the Non-Convex Failure Regime of Difusion-Based Contextual Bandits.

The accepted formulation no longer asks only how to patch dTS for nonlinear score functions. It asks when difusion-based inference itself fails under nonlinear score landscapes.

Finally, SGHA consolidates critic-passing variants into project families. This step avoids presenting several versions of the same underlying direction as separate outputs. Variants can be grouped when they share the same verified gap, direct formulation, supporting papers, or research question. Each family keeps a representative formulation, member variants, source verified gaps, source direct formulations, supporting papers, critic rationale, risks, and internal quality metadata.

## 2.6 Formal Problem Statements

After project families are formed, SGHA turns each family into a semi-formal problem object. This final step makes the research direction easier to inspect. The problem object records the main entities, variables, observations or feedback, decision outputs, objective, constraints, success criterion, assumptions, possible result types, source grounding, and ambiguity flags. The aim is to give a researcher a clear starting point: what is already specified, what the candidate is trying to study, and what still needs to be defined more carefully.

In the Bandits example, the final family is formalized using a score-function class F, a difusion-based inference algorithm $\pi _ { \mathrm { d i f f } }$ , true and approximate reward distributions, a score landscape, and cumulative regret $\mathcal { R } _ { T }$ . The resulting problem asks when difusion-based inference can guarantee sublinear regret, and where this guarantee breaks down because the Gaussian approximation diverges from the true reward distribution under nonlinear score landscapes.

SGHA also records the parts of the formulation that remain underspecified. For this example, it flags the meaning of non-convexity in the score landscape, the operational definition of identification failure, and the precise threshold at which approximation error becomes a failure. These ambiguity flags are part of the final artifact. They make clear which definitions a researcher would need to sharpen before turning the candidate into a theorem, algorithm, or benchmark study.

## 2.7 Profile Conditioning, Evolutionary Exploration, and Low-Signal Outcomes

The same SGHA pipeline can also be used in a profile-conditioned setting. Here, the input includes a researcher’s literature context and, optionally, a target topic. This information is used during corpus construction and prioritization, so the evidence base is closer to the researcher’s previous work or stated interests. The downstream checks remain the same: candidate gaps are still detected from the evidence graph, screened, verified, formulated, critiqued, consolidated, and formalized. Thus, profile conditioning changes what evidence the system starts from, but not the standards a candidate must satisfy before becoming a final project family.

SGHA also includes an optional evolutionary exploration branch. This branch is separate from the main verification-gated family-report path. It starts from a literature-grounded seed, such as a verified gap or an early formulation, and explores nearby problem variants by changing assumptions, mechanisms, objectives, or regimes. The variants are then critiqued, selected, and refined over several rounds. Figure 4 illustrates this view: evolutionary exploration can be seen as mutating a graph-level problem representation while preserving its connection to the original evidence.

The purpose of this branch is breadth. The main SGHA path is conservative: it produces a compact set of verified, critic-filtered, and formalized project families. The evolutionary branch instead explores a wider neighborhood of possible directions around the same evidence base. For this reason, unless explicitly stated, evolutionary outputs are reported separately from the main project-family counts and from the main formulation-quality comparison.

Finally, SGHA can report a low-signal outcome when the evidence does not support a final research menu. This can happen when no candidate passes verification, when no ambition-expanded variant passes the critic, or when no project family survives consolidation. In such cases, the system completes the run but reports that the corpus did not yield a suficiently supported final set of project families. For a literature-driven system, this behavior is important: SGHA should surface grounded opportunities when the evidence supports them, and avoid forcing unsupported directions when it does not.

![](images/93af87dda4a0f3b9a9dbe61e813c3f04fc5e1975133223ca6ebdac641596f401.jpg)  
Figure 4. Graph-mutation view of the optional evolutionary exploration branch. Starting from a literature-grounded seed gap, the branch proposes variants by changing assumptions, mechanisms, objectives, or regimes. Variants are critiqued, selected, and refined before a new exploratory hypothesis is retained. This branch is used for breadth and is kept separate from the main verification-gated family-report path unless stated otherwise.

## 3 Experimental Setup

We evaluate SGHA as a system for producing research-problem formulations from scientific papers. The experiments are organized around three questions. First, can SGHA produce final project-family artifacts across diferent areas of machine learning? Second, how do these artifacts compare with outputs from automated ideation baselines under a common formulation-quality rubric? Third, how does the system behave in additional settings, including profile-conditioned generation, evolutionary exploration, and Bandits sensitivity analyses over model size and corpus size?

The main SGHA experiments use a frozen local Qwen/Qwen3.5-9B model, with no task-specific training or fine-tuning. We compare SGHA with output-count-matched baselines and evaluate all retained candidates with the same blinded formulation-only judge. Alongside judge scores, we also report structural properties of the generated artifacts, such as source grounding, assumptions, formal problem statements, risks, and ambiguity flags.

## 3.1 Domains and Corpora

We run SGHA on five machine-learning domains: bandits, in-context learning, reasoning and test-time computation, ofline reinforcement learning, and uncertainty estimation. These domains cover diferent parts of ML research, from sequential decision-making and in-context behavior to test-time reasoning, ofline policy learning, and uncertainty quantification. Each domain uses a budget of 250 selected papers. To include both topical and source diversity, the bandits, in-context learning, and reasoning/test-time-computation corpora are drawn from OpenReview, while the ofline-reinforcement-learning and uncertainty-estimation corpora are drawn from arXiv.

Together, these five corpora contain 1,250 selected papers. SGHA parses 1,049 of them, meaning that usable text is obtained, and produces valid structured scientific extractions for 1,044 papers. These extractions yield 8,634 evidence tuples, which are used to build typed evidence graphs and instantiate candidate structural gaps. Table 1 reports the corpus and extraction counts by domain.

## 3.2 Systems Compared

The SGHA outputs in the main comparison come from the verification-gated family-report path described in Section 2. The candidate gaps are detected from the evidence graph, screened against nearby corpus evidence, passed through the hard verification gate, formulated, expanded, filtered by an independent critic, consolidated into project families, and converted into semi-formal problem statements. The evaluated SGHA candidate is the final project-family artifact, rather than an intermediate motif hit, direct formulation, or rejected variant.

Table 1. Corpus and extraction counts for the five main SGHA runs. “Parsed” denotes papers for which usable text is obtained; “Extracted” denotes papers for which SGHA produces a valid structured extraction.
<table><tr><td>Domain</td><td>Source</td><td>Selected</td><td>Parsed</td><td>Extracted</td><td>Tuples</td></tr><tr><td>Bandits</td><td>OpenReview</td><td>250</td><td>221</td><td>221</td><td>1554</td></tr><tr><td>In-context learning</td><td>OpenReview</td><td>250</td><td>193</td><td>193</td><td>1559</td></tr><tr><td>Reasoning / test-time</td><td>OpenReview</td><td>250</td><td>137</td><td>137</td><td>1139</td></tr><tr><td>computation Offline reinforcement learning</td><td>arXiv</td><td>250</td><td>248</td><td>245</td><td>2176</td></tr><tr><td>Uncertainty estimation</td><td>arXiv</td><td>250</td><td>250</td><td>248</td><td>2206</td></tr><tr><td>Total</td><td></td><td>1250</td><td>1049</td><td>1044</td><td>8634</td></tr></table>

Table 2. Systems compared in the formulation-quality evaluation. All methods are output-count matched to 15 retained candidates across the five domains.
<table><tr><td>Method</td><td>Model</td><td>Mode</td><td>Evaluated artifact</td></tr><tr><td>SGHA</td><td>Qwen3.5-9B</td><td>Verification-gated literature-gap pipeline</td><td>Final project-family artifact with provenance and a semi-formal problem statement.</td></tr><tr><td>AI-Scientist-v2 + Qwen</td><td>Qwen3.5-9B</td><td>Native ideation-only workflow</td><td>Retained generated idea normalized into the candidate-packet format.</td></tr><tr><td>AI-Scientist-v2 + Claude Opus</td><td>claude-opus-latest</td><td>Native ideation-only workflow</td><td>Retained generated idea from a stronger generator setting.</td></tr><tr><td>MOOSE-Star</td><td>MO0SE-Star-HC-R1D-7B</td><td>Released public HC-only model</td><td>Retained generated hypothesis from the public model.</td></tr></table>

Our primary baseline is native AI-Scientist-v2 ideation with the same local Qwen/Qwen3.5-9B model. We use its ideation entry point and match the number of retained candidates to the number of SGHA final families in each domain. This gives 15 candidates from each method: 3 for bandits, 4 for in-context learning, 1 for reasoning and test-time computation, 1 for ofline reinforcement learning, and 6 for uncertainty estimation.

We also include two reference baselines. The first is AI-Scientist-v2 with anthropic/claude-opus-latest as a stronger generator, again run in ideation-only mode and matched to SGHA’s retained candidate counts. The second is MOOSE-Star using the released ZonglinY/MOOSE-Star-HC-R1D-7B public model in HC-only mode, with no additional training or fine-tuning. These baselines give reference points for same-model native ideation, stronger-generator ideation, and public trained hypothesis generation.

Before judging, all outputs are converted into a common candidate format. The packet contains the title, problem statement, motivation or abstract, proposed direction, expected contribution, source or context information when available, assumptions or setup when available, formal-problem fields when available, ambiguity or missing-definition fields when available, and risks or caveats when available. Fields that a method does not produce are left as not provided; we evaluate the artifacts as produced and report structural-field coverage separately.

## 3.3 Formulation-Only Evaluation

SGHA produces research-problem formulations rather than complete project plans. We therefore evaluate candidates with a formulation-only rubric. The rubric focuses on whether a candidate states a clear problem, gives enough technical structure, controls its scope, makes assumptions visible, and identifies what remains ambiguous. Table 3 lists the ten criteria used by the judge.

Table 3. Formulation-only evaluation rubric. Each criterion is scored on a 0–10 scale.
<table><tr><td>Criterion</td><td>What the judge evaluates</td></tr><tr><td>Problem-definition clarity</td><td>Whether the candidate states a clear research problem rather than a broad topic or loose motivation.</td></tr><tr><td>Technical specificity</td><td>Whether the formulation names concrete objects, settings, mechanisms, assumptions, metrics, or target results.</td></tr><tr><td>Well-posedness</td><td>Whether the problem has enough structure to be studied or refined.</td></tr><tr><td>Assumption-boundary clarity</td><td>Whether the formulation makes clear which assumptions are used, relaxed, questioned, or missing.</td></tr><tr><td>Formalizability</td><td>Whether the problem can plausibly be written as a theorem, algorithmic objective, benchmark protocol, impossibility result, or other formal research target.</td></tr><tr><td>Nontriviality</td><td>Whether the candidate goes beyond a simple application, small robustness check, or generic “apply X to Y&quot; idea.</td></tr><tr><td>Scope control</td><td>Whether the problem is focused enough to be studied rather than combining many loosely related concepts.</td></tr><tr><td>Source-grounded specificity</td><td>Whether the formulation is tied to provided source context, evidence, or literature-derived details.</td></tr><tr><td>Ambiguity hygiene</td><td>Whether the candidate explicitly states missing definitions, unclear feedback models, unresolved assumptions, or caveats.</td></tr><tr><td>Overall formulation quality</td><td>The judge&#x27;s overall assessment of the research-problem formulation under the rubric.</td></tr></table>

Table 4. Score anchors used by the formulation-only judges.
<table><tr><td>Score</td><td>Interpretation</td></tr><tr><td>0</td><td>No usable formulation.</td></tr><tr><td>1</td><td>Mostly incoherent or unrelated to a research problem.</td></tr><tr><td>2</td><td>Topic-level idea with almost no problem structure.</td></tr><tr><td>3</td><td>Very vague formulation with major missing pieces.</td></tr><tr><td>4</td><td>Weak formulation; some direction is visible, but the problem is poorly specified.</td></tr><tr><td>5</td><td>Plausible idea, but important assumptions, scope, or definitions are missing.</td></tr><tr><td>6</td><td>Plausible formulation with useful structure, but still needing substantial refinement.</td></tr><tr><td>7</td><td>Strong formulation with clear problem structure and reasonable technical specificity.</td></tr><tr><td>8</td><td>Very strong formulation; well posed, grounded, and mostly ready for expert refinement.</td></tr><tr><td>9</td><td>Excellent formulation with unusually clear assumptions, scope, and formal target.</td></tr><tr><td>10</td><td>Exceptional formulation; clear, grounded, formalizable, and close to research-ready.</td></tr></table>

The 0–10 scores use the anchors in Table 4. These anchors are meant to make the scores interpretable: a score around 5 or 6 means the formulation is plausible but still needs refinement, while scores of 7 or 8 indicate a strong formulation.

The evaluation uses method-label-masked candidate packets. Method labels are hidden from the judges, and the label key is used only during postprocessing. Judges evaluate the provided candidate text and supporting context under the fixed rubric; external literature novelty and correctness are outside the judge task.

To reduce dependence on a single model or provider, we use five LLM judges from diferent providers: anthropic/claude-sonnet-4, openai/gpt-5.6-sol-pro, x-ai/grok-4.5, moonshotai/kimi-k3, and google/gemini-3.6-flash. Each judge receives the same blinded packet format, rubric, score anchors, and response schema. The judge panel is used only after all candidate outputs have been frozen; it is not part of SGHA’s generation or verification pipeline.

We report criterion-level means together with the judges’ overall formulation-quality score. These scores are descriptive assessments of formulation quality, and we interpret them alongside structural properties of the generated outputs.

## 3.4 Profile-Conditioned Generation

We also evaluate a profile-conditioned version of SGHA. In this setting, the input includes a researcherspecific literature context and, optionally, a target topic. The profile context guides corpus construction and prioritization, so the selected papers and candidate gaps are closer to the researcher’s previous work or stated area of interest.

After corpus construction, the downstream SGHA pipeline remains the same. Candidate gaps are detected from the evidence graph, screened, verified, formulated, expanded, filtered by the critic, consolidated into project families, and converted into semi-formal problem statements. Profile conditioning changes the evidence base and prioritization, while preserving the same checks before finalization.

We evaluate profile-conditioned generation on three researcher contexts: Yann LeCun, Geofrey Hinton, and Michael I. Jordan. The resulting candidates are evaluated with a personalized judge rubric that includes the formulation-quality criteria from the main evaluation and additional profile-specific criteria for alignment, specificity, and fit to the provided profile context.

## 3.5 Evolutionary Exploration

SGHA includes an evolutionary exploration branch that serves a diferent role from the compact family-report path. The family-report path starts from verification-passed gaps and produces a small set of critic-filtered project families. The evolutionary branch explores a broader neighborhood around literature-grounded seeds.

Starting from a seed gap or formulation, the evolutionary branch varies parts of the problem representation, such as assumptions, mechanisms, objectives, or regimes. Candidate variants are critiqued, selected, and refined over multiple rounds. For evaluation, selected or ranked evolutionary hypotheses are normalized into the same candidate format used for the other methods and scored with the same formulation-only rubric. We report these outputs separately from the main SGHA project-family counts.

## 3.6 Model- and Corpus-Size Sensitivity

Finally, we include two sensitivity analyses on the Bandits domain. The model-size analysis compares the main Qwen/Qwen3.5-9B run with a smaller 4B configuration and a larger 27B configuration. This tests how extraction quality, verification yield, critic-passing variants, and final family construction change with model capacity.

The corpus-size analysis varies the number of selected Bandits papers. We run settings with 50, 100, 150, 200, and 250 selected papers. This tests how SGHA behaves as the evidence base grows. We treat this as a sensitivity study rather than a scaling-law experiment: larger corpora provide more evidence and more candidate gaps, while later stages such as verification, critic filtering, and family consolidation determine which candidates survive.

## 4 Results

## 4.1 End-to-End Yield Across Domains

We first evaluate whether SGHA can complete the full path from papers to final research-problem families across diferent areas of machine learning. Table 5 summarizes the five main runs. Across 1,250 selected papers, SGHA extracts structured scientific content from 1,044 papers and produces 8,634 evidence tuples. These evidence tuples support graph construction and candidate-gap discovery. After corpus screening and verification, 39 gaps are accepted for formulation. The final synthesis stages then produce 19 accepted variants, which are consolidated into 15 project families with 15 semi-formal problem statements.

This yield shows the intended behavior of the pipeline. SGHA starts from a large literature-derived evidence base and progressively refines it into a compact set of research-problem artifacts. The intermediate stages provide breadth: extraction creates evidence tuples, motifs surface candidate gaps, and verification identifies formulation-ready gaps. The final stages provide structure: ambition expansion, criticism, family consolidation, and formalization turn selected gaps into inspectable project families.

Table 5. End-to-end SGHA yield across the five main domains. Verified gaps are candidate gaps that pass the hard verification gate. Accepted variants are ambition-expanded formulations that pass the independent critic.
<table><tr><td>Domain</td><td>Tuples Verified</td><td>gaps Accepted variants</td><td>Families</td><td>Formal statements</td></tr><tr><td>Bandits</td><td>1554</td><td>6</td><td>4</td><td>34 3</td></tr><tr><td>In-context learning</td><td>1559</td><td>12</td><td>4</td><td>4</td></tr><tr><td>Reasoning / test-time computation</td><td>1139</td><td>7</td><td>1</td><td>1 1</td></tr><tr><td>Offline reinforcement learning</td><td>2176</td><td>2</td><td>1</td><td>1 1</td></tr><tr><td>Uncertainty estimation</td><td>2206</td><td>12</td><td>9</td><td>6</td></tr><tr><td>Total</td><td>8634</td><td>39</td><td>19</td><td>6 15 15</td></tr></table>

Table 6. Formulation-quality scores averaged over five LLM judges. All methods are evaluated on 15 retained candidates. Bold indicates the best score in each column.
<table><tr><td>Method</td><td>Overall</td><td>Clarity</td><td>Technical</td><td>Well-posed</td><td>Boundary</td><td>Formal.</td><td>Nontrivial</td><td>Scope</td><td>Grounding</td><td>Ambiguity</td></tr><tr><td>SGHA</td><td>5.99</td><td>6.77</td><td>5.89</td><td>5.77</td><td>6.45</td><td>5.51</td><td>6.39</td><td>5.36</td><td>7.43</td><td>7.61</td></tr><tr><td>AI-Scientist.  $\mathbf { \omega } \cdot \mathbf { v } 2 \mathbf { \alpha } + \mathbf { \alpha } \mathrm { Q w e n }$ </td><td>4.55</td><td>5.59</td><td>4.72</td><td>3.53</td><td>3.61</td><td>3.68</td><td>5.27</td><td>5.73</td><td>5.11</td><td>2.81</td></tr><tr><td>AI-Scientist-v2 + Claude Opus</td><td>5.84</td><td>7.15</td><td>6.95</td><td>4.88</td><td>4.75</td><td>5.21</td><td>6.88</td><td>6.19</td><td>2.97</td><td>3.57</td></tr><tr><td>MOOSE-Star</td><td>2.00</td><td>1.85</td><td>2.51</td><td>1.52</td><td>1.37</td><td>1.51</td><td>2.89</td><td>3.07</td><td>3.75</td><td>1.48</td></tr></table>

All five domains reach the final project-family stage, but with diferent yields. This is expected for a literaturedriven system: some corpora contain several separable directions, while others lead to a smaller and tighter set of families. Uncertainty estimation produces the largest set of final families, while reasoning/test-time computation and ofline reinforcement learning produce more focused outputs.

Table 5 therefore defines the SGHA candidate set used in the main comparison: 15 formalized project families obtained from 39 verification-passed gaps. We next evaluate the quality of these artifacts—how clear, grounded, formalizable, and explicit about ambiguity they are—against automated ideation baselines.

## 4.2 Formulation Quality Compared with Baselines

In Table 6 we report the mean formulation-quality scores across the five LLM judges. SGHA obtains the highest overall formulation-quality score. The strongest gains are on the criteria that the pipeline is designed to improve: source-grounded specificity, ambiguity hygiene, assumption-boundary clarity, well-posedness, and formalizability.

The comparison with AI-Scientist-v2 using the same local Qwen model shows the clearest diference. Both systems use the same generator, but SGHA first builds an evidence graph, verifies candidate gaps, and produces formal problem objects. This leads to higher scores on nearly all formulation criteria, especially ambiguity hygiene and assumption-boundary clarity. The Claude Opus AI-Scientist-v2 baseline is much stronger on surface formulation: it produces clearer, more technically detailed, and more tightly scoped ideas. However, SGHA remains higher on the evidence-linked and structure-oriented criteria, and also has the highest overall score. MOOSE-Star scores lower in this setup, reflecting that the released HC-only model produces hypothesis text but not the same kind of verified, formalized problem artifact.

The table shows a useful tradeof. Strong ideation models can produce polished and well-scoped ideas, as reflected by the Claude Opus baseline. SGHA is strongest when the rubric asks for problem structure that depends on the pipeline: grounding in the provided literature, explicit assumptions, formalizability, and clean handling of ambiguity. This supports the central claim of the paper: the benefit of SGHA is not only that it generates plausible ideas, but that it turns literature-derived gaps into inspectable research-problem formulations.

## 4.3 Structural Artifact Coverage

The judge scores measure formulation quality, but they do not fully show what information is available in each generated artifact. We therefore also inspect the structure of the outputs directly. Table 7 reports whether each candidate includes the fields that make a research-problem artifact easier to audit: a formal problem statement, assumptions or setup, source/context grounding, ambiguity flags, an evaluation plan, and risks or caveats.

Table 7. Structural artifact coverage across the 15 retained candidates for each method. A checkmark means the field is present for all retained candidates from that method; a cross means it is absent from all retained candidates. This table records artifact structure, not judge-assigned quality.
<table><tr><td>Method</td><td colspan="6">Formal problem Setup / assumptions Source/context Ambiguity flags Eval. plan Risks/caveats Complete artifact</td></tr><tr><td>SGHA</td><td>√</td><td>√</td><td>V</td><td>√</td><td>V</td><td>V</td><td>√</td></tr><tr><td>AI-Scientist-v2 + Qwen</td><td>x</td><td>x</td><td></td><td>x</td><td>V</td><td>V</td><td>x</td></tr><tr><td>AI-Scientist-v2 + Claude</td><td>X</td><td>x</td><td></td><td>X</td><td></td><td></td><td>x</td></tr><tr><td>Opus MOOSE-Star</td><td>x</td><td>x</td><td></td><td>x</td><td>x</td><td>x</td><td>x</td></tr></table>

Table 8. Representative Bandits artifacts shown in this section. Each row uses the best retained Bandits candidate for that method under the formulation-quality evaluation.
<table><tr><td>Method</td><td>Example title</td><td>Overall</td><td>Artifact type</td></tr><tr><td>SGHA</td><td>Characterizing Identifiability Limits of Structured Bandits Under Piecewise Non-Stationarity</td><td>6.6</td><td>Formalized project family</td></tr><tr><td>AI-Scientist-v2 + Qwen</td><td>Bandits with Adversarial Arm Execution: Robust Learning When Your Choices Don&#x27;t Match Reality</td><td>4.6</td><td>Proposal-style idea</td></tr><tr><td>AI-Scientist-v2 + Claude Opus</td><td>Whose Scale Is It Anyway? Monotone-Robust Bandits with Utility Elicitation</td><td>6.2</td><td>Detailed proposal-style idea</td></tr><tr><td>MOOSE-Star</td><td>Order Preservation in Set-Size Dependent Combinatorial Bandits</td><td>2.2</td><td>Source-grounded hypothesis</td></tr></table>

SGHA includes all of these fields for all 15 final project families. This is expected, since the final SGHA report is designed to expose the problem structure rather than only give a proposal title and motivation. The baseline systems produce diferent kinds of artifacts. AI-Scientist-v2 outputs usually include readable motivation, source/context information, evaluation plans, and caveats, but they do not produce SGHA-style formal problem objects or ambiguity flags. MOOSE-Star provides source-grounded hypothesis text in this setup, but not the full structured research-problem object.

This analysis helps explain the pattern in Table 6. SGHA’s advantage is strongest on source-grounded specificity and ambiguity hygiene because these fields are part of the final artifact. The baselines can still produce clear or well-scoped ideas, especially with a stronger generator, but their outputs are less explicit about the assumptions, missing definitions, and formal structure that a researcher would need to inspect before developing the idea further.

## 4.4 Qualitative Comparison of Generated Artifacts

The quantitative results compare average formulation quality across methods. To make the comparison more concrete, we also show representative generated formulations. For each method, we display its best retained Bandits candidate under the formulation-quality evaluation. This keeps the domain fixed while showing the diferent forms of output produced by SGHA, AI-Scientist-v2, and MOOSE-Star.

Table 8 summarizes the four examples. The examples are shown in their generated form: SGHA produces a formalized project-family artifact, AI-Scientist-v2 produces proposal-style ideas, and MOOSE-Star produces a compact source-grounded hypothesis.

## 4.4.1 SGHA

Characterizing Identifiability Limits of Structured Bandits Under Piecewise Non-Stationarity

Problem formulation. Current robust bandit algorithms for structured action sets, such as Network Lasso, rely on the i.i.d. assumption of context generation. This assumption fails in environments with piecewise constant non-stationarity, leading to unbounded regret and failure in identifying optimal arms. The fundamental question is not merely how to fix a specific algorithm, but whether the network structure itself allows for consistent learning under such distributional shifts, or if a fundamental identifiability barrier exists.

Proposal-style abstract. This project studies the fundamental limits of learning in contextual bandits with network-structured action sets when the underlying data distribution undergoes piecewise constant shifts. The central question is whether the structural constraints imposed by network regularization are suficient to guarantee consistent policy identification in non-stationary regimes, or if the combination of structural sparsity and temporal drift creates an inherent identifiability gap. A successful outcome would characterize the precise boundary between learnable and unlearnable regimes for this problem class, providing necessary and suficient conditions for robustness that are independent of any specific algorithmic implementation. This work moves beyond validating a single method to establishing a theoretical framework for the viability of structured learning under non-stationarity.

Formal problem statement. Let $\mathcal { A }$ be a set of actions constrained by a network structure ${ \mathcal { G } } .$ Let $\mathcal { D } _ { t }$ denote the distribution of contexts at time t. The environment exhibits piecewise constant non-stationarity, meaning $\mathcal { D } _ { t } = \mathcal { D } _ { k }$ for $t \in [ t _ { k } , t _ { k + 1 } )$ . The question is whether there exists a sequence of policies $\pi _ { t }$ such that the regret $R _ { T }$ grows sublinearly with time $T$ solely based on the structural constraints of ${ \mathcal { G } } ,$ without assuming that $\mathcal { D } _ { t }$ is i.i.d. across time.

Variables and notation. $\mathcal { A } \colon$ set of available actions; $\mathcal { G } \colon$ network structure constraining action sets; $\mathcal { D } _ { t } \mathbf { : }$ distribution of contexts at time $t ; t _ { k }$ : time points where the distribution shifts; $\pi _ { t } \colon$ policy at time $t ;$ $R _ { T } ;$ : cumulative regret up to time $T .$

Objective. Characterize the necessary and suficient conditions for consistent learning, or identifiability, in network-structured bandits under piecewise constant non-stationarity.

Assumptions. Network structure constraint: kept. Piecewise constant non-stationarity: kept. i.i.d.   
context generation: removed. Existence of consistent learning: questioned.

Success criterion. Establish a precise boundary between regimes where network structure aids robustness and regimes where structural constraints amplify non-stationarity errors, leading to impossibility.

Risk and falsification condition. Proving impossibility results requires rigorous mathematical machinery, and the result may depend on how the structural and non-stationarity assumptions are defined. The direction would be weakened by a specific non-stationary environment and network topology where consistent learning is achieved despite violating the proposed necessary conditions for failure.

Ambiguity flags. Network structure; piecewise constant non-stationarity; feedback model; boundary;   
feedback or measurement model.

Source/context grounding. The family traces to a verification-passed structural gap, gap:ba210076fbccfacb, with source direct formulation direct:02. The retained supporting papers are Clusters Agnostic Network Lasso Bandits and Network Lasso Bandits. In the SGHA trace, these papers support an assumption-mismatch pattern around Network Lasso-style bandit methods: the method is linked to i.i.d. context/action-set assumptions and to a piecewise-constant non-stationarity failure condition.

This SGHA example is a structured research-problem object. It contains a problem formulation, an abstract, notation, an objective, assumption statuses, a success criterion, a risk, a falsification condition, ambiguity flags, and source grounding. The artifact is still a candidate direction, but the information needed to inspect and refine it is made explicit.

## 4.4.2 AI-Scientist-v2 with Qwen

## Bandits with Adversarial Arm Execution

Problem statement. Standard bandit algorithms assume perfect action execution, but in real systems such as cloud services, autonomous systems, and recommendation platforms, there is often uncertainty between the learner’s intended action and the action actually executed because of system errors, competing processes, or adversarial interference. The hypothesis is that explicitly modeling execution uncertainty, where an adversary or system can alter which arm is actually executed, requires fundamentally diferent algorithmic approaches that can achieve robust regret bounds even when the learner observes only the outcome and not the discrepancy.

Motivation. In practical bandit deployments, the gap between intended action and actual execution can severely undermine learning. System failures, resource contention, or adversarial interference may cause the learner’s selected arm to difer from the executed arm, yet standard bandit algorithms operate under the assumption of perfect execution. This creates a critical vulnerability: learners may systematically misattribute rewards to wrong arms, leading to catastrophic regret growth.

Proposed direction. The proposed Adversarial Arm Execution bandit framework allows an adversary or stochastic system to alter the executed arm after the learner’s selection, with the learner observing only the final executed arm and its reward.

Evaluation plan. The proposed evaluation includes a synthetic adversarial-arm-execution benchmark where the executed arm difers from the selected arm in a fraction of trials; comparisons between adversarial and stochastic execution uncertainty; a recovery analysis where occasional ground-truth signals reveal the executed arm identity; and a real-world simulation using recommendation logs where impressions may not match clicks because of rendering errors.

Risks. The generated artifact notes computational complexity, modeling assumptions, limited ground truth, and adversarial complexity as risks.

The Qwen AI-Scientist-v2 example reads as a proposal for a new bandit setting. It gives a clear motivation, describes a setting, and outlines experiments. Its emphasis is on a project direction and evaluation plan rather than on a formalized problem object.

## 4.4.3 AI-Scientist-v2 with Claude Opus

Whose Scale Is It Anyway? Monotone-Robust Bandits with Utility Elicitation

Problem statement. Almost every deployed bandit optimizes the mean of an arbitrarily chosen numerical encoding of an ordinal signal, such as 1–5 stars, 1–10 judge scores, or clinical grades. The identity of the best arm is not invariant to monotone re-encodings of that signal. The proposed direction hypothesizes that such rank reversals and even non-transitive dominance cycles are common in logged feedback, that no algorithm using only sampled rewards can fully resolve them, and that a scale-free learner should play a randomized minimax-regret mixture over arms unless a small number of utility-elicitation queries is available.

Motivation. Bandit algorithms maximize the mean of a number, but in many applications that number is an arbitrary encoding of an ordinal signal. A five-star rating, a 1–10 LLM-judge score, or a clinical grade can be mapped to utilities in multiple monotone ways, and the choice can flip which arm is best.

Proposed direction. The artifact formalizes a monotone-robust bandit setting in which rewards take values in an ordered set of L levels, the true utility u is an unknown increasing function, and the learner is judged by regret maximized over all utilities consistent with what it knows. It proposes studying rank reversals and minimax regret over monotone utility functions, together with utility-elicitation queries based on standard-gamble comparisons over the L observation levels.

Evaluation plan. The proposed evaluation includes scale-fragility audits on rating datasets, synthetic ordinal bandits, randomization-gap experiments, elicitation-complexity experiments, a joint explorationelicitation tradeof, and an LLM-evaluation case study.

Risks. The generated artifact notes that the novelty depends on the minimax-regret-over-utilities objective, the randomization result, and the elicitation-complexity result. It also notes that unconstrained monotone utilities may make non-comparability pervasive, that constants may be conservative, and that logged ratings are confounded.

The Claude Opus AI-Scientist-v2 example is the strongest baseline example in this qualitative set. It is detailed, technically specific, and well motivated. It also helps explain why this baseline scores strongly on clarity, technical specificity, nontriviality, and scope control.

Table 9. Profile-conditioned SGHA pipeline results. Verified gaps are candidate gaps that pass the hard verification gate before formulation.
<table><tr><td>Profile</td><td>Seeds</td><td>Parsed</td><td>Extracted</td><td>Tuples</td><td>Verified gaps</td><td>Families</td><td>Formal statements</td></tr><tr><td>Yann LeCun</td><td>11</td><td>150</td><td>148</td><td>1170</td><td>4</td><td>1</td><td>1</td></tr><tr><td>Geoffrey Hinton</td><td>33</td><td>150</td><td>141</td><td>1191</td><td>14</td><td>1</td><td>1</td></tr><tr><td>Michael I. Jordan</td><td>150</td><td>150</td><td>150</td><td>1043</td><td></td><td>2</td><td>2</td></tr><tr><td>Total</td><td>194</td><td>450</td><td>439</td><td>3404</td><td>9</td><td>4</td><td>4</td></tr></table>

## 4.4.4 MOOSE-Star

Order Preservation in Set-Size Dependent Combinatorial Bandits

Hypothesis. The order preservation property addresses the gap in handling larger exploration sets by maintaining the order of reward means, which allows for a more eficient exploration strategy. This reduces the regret associated with traditional methods that struggle with extensive exploration sets.

Mechanism. The property ensures that the order of reward means remains consistent regardless of set size, enabling the algorithm to focus on the most promising combinations of base arms. This prioritization allows for more eficient exploration of superarms, reducing the number of necessary trials and thus lowering regret.

Proposed direction. The SUCB algorithm is adapted to incorporate the order preservation property by modifying the selection process to prioritize superarms based on the order of their base arms’ rewards.

Expected contribution. The expected contribution is to use order preservation to focus exploration on high-potential combinations of base arms, reducing unnecessary trials and lowering regret.

Source grounding. Inspiration paper: Set-Size Dependent Combinatorial Bandits.

The MOOSE-Star example is a compact hypothesis-style artifact. It is grounded in an inspiration paper and proposes a mechanism, but it exposes fewer fields for inspection than the SGHA or AI-Scientist-v2 examples.

## 4.4.5 Summary

The qualitative examples clarify what the numerical scores summarize. SGHA returns a formalized projectfamily artifact, with notation, assumptions, a success criterion, risks, ambiguity flags, and source grounding. AI-Scientist-v2 returns proposal-style ideas, which can be readable and technically detailed, especially with a stronger generator. MOOSE-Star returns a shorter source-grounded hypothesis. These diferences in artifact shape align with the quantitative results: SGHA is strongest on source-grounded specificity, formalizability, and ambiguity hygiene, while the strongest ideation baseline is competitive on clarity, technical specificity, and scope control.

## 4.5 Profile-Conditioned Generation

We next evaluate whether SGHA can be steered by a researcher-specific literature context. In this setting, the input includes a profile-derived corpus and, optionally, a target topic. The downstream SGHA pipeline remains the same: the system still extracts evidence, builds a graph, detects candidate gaps, verifies them, formulates them, applies the critic, and produces final project families. Thus, profile conditioning changes where the evidence comes from.

Table 9 reports the pipeline yield for three profile-conditioned runs. Across the three profiles, SGHA extracts structured content from 439 papers and produces 3,404 evidence tuples. Nine gaps pass verification, leading to four final project families with four formal problem statements.

We evaluate the resulting candidates with a personalized judge rubric. This rubric keeps the formulationquality criteria from the main evaluation and adds profile-specific criteria: profile alignment, profile specificity, intellectual-style match, and personalization overall. The judge uses only the provided profile context and candidate text.

Table 10. Profile-conditioned judge scores. Formulation measures the research-problem formulation itself; the remaining columns measure alignment to the provided profile context.
<table><tr><td>Profile</td><td>Candidate</td><td>Formulation Alignment</td><td>Specificity</td><td></td><td>Style match</td><td>Personalization</td></tr><tr><td>Yann LeCun</td><td>Fundamental limits of partition-function approximation in high-dimensional EBMs</td><td>6.17</td><td>7.00</td><td>4.50</td><td>5.67</td><td>5.33</td></tr><tr><td>Geoffrey Hinton</td><td>Convergence failure regimes of stochastic-gradient approximations in deep generative models</td><td>5.67</td><td>8.00</td><td>7.17</td><td>6.83</td><td>7.17</td></tr><tr><td>Michael I. Jordan</td><td>Identifiability boundaries of filtering-clustering under rank-deficient design matrices</td><td>6.00</td><td>6.50</td><td>5.83</td><td>7.33</td><td>6.33</td></tr><tr><td>Michael I. Jordan</td><td>Characterizing hierarchical identifiability limits in variational protein annotation</td><td>6.00</td><td>8.00</td><td>7.00</td><td>7.75</td><td>7.00</td></tr></table>

Table 10 shows the personalized scores. The strongest profile-alignment scores appear for the Geofrey Hinton candidate and the Michael I. Jordan hierarchical-identifiability candidate. The Yann LeCun candidate has the highest formulation-quality score among the profile-conditioned outputs, but lower profile specificity. This pattern is useful: profile conditioning can steer SGHA toward profile-relevant directions, while the judge still separates general formulation quality from how specifically the candidate matches the provided profile context.

The Michael I. Jordan example illustrates the kind of profile-conditioned artifact SGHA produces. The final family, Characterizing Hierarchical Identifiability Limits in Variational Protein Annotation, connects profile-relevant themes such as variational inference, graphical models, and structured prediction to a concrete problem in protein-function annotation.

## Characterizing Hierarchical Identifiability Limits in Variational Protein Annotation

Profile context. This candidate comes from the Michael I. Jordan profile-conditioned run. The profileconditioned evidence base emphasizes probabilistic modeling, graphical models, statistical machine learning, variational inference, Bayesian nonparametrics, and optimization. The run uses a profile conditioned corpus of 150 parsed and extracted papers, producing 1,043 evidence tuples.

Source papers. The final family is grounded in the retained profile-seed papers profile pdf: a variational principle for graphical models [19] and profile pdf: genome scale phylogenetic function annotation of large and diverse protein families [20]. The first source provides the variational-inference / graphical-model context, while the second provides the protein-function annotation setting.

Verified gap. The family traces to the verified gap gap:dcb6662b99312fb7, an unresolved tradeof around Gene Ontology hierarchy. The gap passes the hard verification gate with survival score 0.664, above the threshold 0.600. The verification trace is mixed in a useful way: the support and feasibility agents view the gap as grounded and studyable, while the skeptic and critic caution that the strongest version of the claim should not be overstated because hierarchy-aware scoring methods exist.

Direct formulation. The source direct formulation is direct:04: Integrating Gene Ontology Hierarchy Constraints into Variational Inference for Protein Function Annotation. This direct formulation asks for a variational inference framework that explicitly incorporates Gene Ontology tree constraints while avoiding prohibitive computational cost.

Ambition expansion and critic. The ambition stage generates three variants. The first two variants, var:10 and var:11, are rejected by the independent critic as inflated or insuficiently supported. The accepted variant is var:12, which reframes the problem as a theoretical boundary: under what structural conditions do flat variational objectives fail to recover the true posterior over a tree-structured label space? This accepted variant becomes the final project family.

Problem formulation. Current variational models for protein function annotation treat Gene Ontology (GO) terms as flat sets, ignoring the parent-child hierarchy. This oversight creates a fundamental gap where biologically consistent, less precise predictions are penalized, but the theoretical boundary of this failure is unknown. The central challenge is to characterize the exact conditions under which hierarchical constraints become necessary for identifiability and to define the regime where flat approximations fail catastrophically.

Formal problem statement. Let T be a tree-structured label space representing Gene Ontology terms and parent-child relationships. Let $P _ { \mathrm { t r u e } }$ be the true posterior over T given protein data D, and let $\mathcal { F } _ { \mathrm { f l a t } }$ be a flat variational approximation family. Each $Q \in { \mathcal { F } } _ { \mathrm { f l a t } }$ approximates $P _ { \mathrm { t r u e } }$ without enforcing parent-child consistency constraints from the tree topology. The problem is to characterize conditions C such that, for every $\bar { Q } \in \mathcal { F } _ { \mathrm { f l a t } }$ , the approximation error $\lVert P _ { \mathrm { t r u e } } - Q \rVert$ exceeds a threshold ϵ when the structural disconnect between parent-child implications prevents recovery of the true posterior.

Variables and notation. T: tree-structured Gene Ontology label space; $P _ { \mathrm { t r u e } } \colon$ hierarchy-respecting posterior over GO terms; $\mathcal { F } _ { \mathrm { { f l a t } } } \colon$ flat variational family; $Q \colon$ approximating distribution in $\begin{array} { r } { \bar { \mathcal { F } } _ { \mathrm { f l a t } } ; ~ \bar { D } \colon } \end{array}$ observed protein data; ϵ: approximation-error threshold; C: structural or data conditions under which flat approximations fail.

Objective. Characterize when flat variational approximations are insuficient for recovering a hierarchyrespecting posterior over Gene Ontology terms. More concretely, identify structural conditions on the label hierarchy and data distribution under which every flat approximation $Q \in { \mathcal { F } } _ { \mathrm { f l a t } }$ remains farther than ϵ from $P _ { \mathrm { t r u e } }$

Assumptions. Existence of a hierarchy-respecting posterior: kept. Gene Ontology tree structure: kept. Flat variational suficiency: relaxed or questioned. Parent-child consistency constraints: treated as necessary to define the hierarchical setting.

Success criterion. Identify conditions on the GO tree and data distribution under which flat variational inference fails to recover the hierarchy-respecting posterior, and characterize the resulting identifiability boundary.

Possible result types. A theorem characterizing when flat variational families cannot approximate the true hierarchy-respecting posterior within threshold ϵ; an algorithmic direction for hierarchy-aware variational inference; and an empirical protocol comparing flat and hierarchy-aware approximations on protein-function annotation tasks.

Ambiguity flags. Catastrophic failure: requires a specific divergence metric or threshold. Structural disconnect: requires a precise mathematical relationship between ignored parent-child constraints and approximation error. Boundary: requires a formal definition of the transition between identifiable and non-identifiable regimes. Feedback or measurement model: requires further definition for the protein-annotation setting.

Formalization confidence and risk. The formalization confidence is medium, and the artifact sets requires\_human\_definition to true. The main risk is that the impossibility-boundary claim is only moderately supported by the source evidence: the sources identify the hierarchy-related gap and the variational-inference context, but do not themselves prove a theoretical failure regime.

This example shows the intended role of profile conditioning. The profile context changes the evidence base and helps steer SGHA toward a problem that fits the supplied research context, while the final artifact stil exposes the formal setup, objective, and remaining ambiguities.

## 4.6 Evolutionary Exploration

We next study SGHA’s evolutionary exploration branch. This branch has a diferent role from the main project-family path. The main path is designed to produce a compact set of verified, critic-filtered, and formalized problem families. Evolutionary exploration instead searches more broadly around literaturegrounded seeds. Starting from a seed gap or formulation, it varies parts of the problem representation—such as assumptions, mechanisms, objectives, or regimes—and then critiques, selects, and refines the resulting variants.

Table 11 shows the scale of this exploration. Across the five domains, the branch generates 2,908 evolved candidates and retains 160 selected hypotheses during the search. The final ranked files contain 96 hypotheses. This gives SGHA a broader exploratory layer: the family-report path produces a small set of polished problem families, while the evolutionary branch gives a larger pool of nearby directions that can be inspected or passed through later verification and formalization.

We evaluate an output-count-matched set of 15 ranked evolutionary hypotheses with the same formulation-only rubric used in the main comparison. Table 12 reports the average scores. The evolutionary hypotheses score highest on source-grounded specificity and nontriviality, which is consistent with their purpose: they are generated from literature-grounded seeds and often point to interesting directions. Their lower scores on well-posedness, formalizability, and ambiguity hygiene also match the role of this branch. These hypotheses are exploratory; they have not gone through the full family-report path that adds formal problem statements, assumption tables, and ambiguity flags.

Table 11. Evolutionary exploration artifacts by domain. Evolved candidates include mutations, crossovers, and syntheses around literature-grounded seeds.
<table><tr><td>Domain</td><td>Evolved</td><td></td><td>Selected Final ranked Initial gaps</td></tr><tr><td>Bandits</td><td>603</td><td>32</td><td>20 13</td></tr><tr><td>In-context learning</td><td>472</td><td>32</td><td>20 16</td></tr><tr><td>Reasoning / test-time computation</td><td>502</td><td>32</td><td>20 12</td></tr><tr><td>Offline RL</td><td>594</td><td>32</td><td>16 8</td></tr><tr><td>Uncertainty estimation</td><td>737</td><td>32</td><td>20 15</td></tr><tr><td>Total</td><td>2908</td><td>160</td><td>96 64</td></tr></table>

Table 12. Formulation-only scores for the selected evolutionary-exploration hypotheses. Scores are averaged over the available judge evaluations.
<table><tr><td>Candidate set</td><td></td><td>Overall Clarity Technical Well-posed Boundary Formal. Nontriv. Scope Grounding Ambiguity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SGHA evolutionary exploration</td><td>3.82</td><td>4.11</td><td>4.09</td><td>2.80</td><td></td><td>3.42 2.64</td><td></td><td>5.05 4.38</td><td>5.40</td><td>2.20</td></tr></table>

The domain-level results in Table 13 show the same pattern. The reasoning/test-time-computation candidate has the highest mean overall score, and source grounding is relatively strong across domains. This suggests that the evolutionary branch is useful for surfacing grounded directions, even when the resulting hypotheses still need the later formalization and ambiguity-cleanup stages before they become final research-problem artifacts.

The highest-scoring evolutionary candidate comes from reasoning and test-time computation. It studies self-verification mechanisms in test-time compute scaling, where additional inference-time search may stop improving accuracy when intermediate reasoning steps are dificult to verify.

## Self-verification mechanisms within test-time compute scaling frameworks

Problem statement. The problem of understanding self-verification mechanisms within test-time compute scaling frameworks that achieve accuracy saturation despite increased inference-time search depth due to the collapse of verifiable correctness assumptions on inherently unverifiable intermediate steps.

Gap. The problem of verification-dependent error accumulation that achieves accuracy saturation despite increased inference-time search depth.

Target. Self-verification mechanisms within test-time compute scaling frameworks.

Scope. Cross-gap synthesis over assumption-mismatch, repeated-unaddressed-limitation, and unrealisticassumption motifs, narrowed to the most evidence-dense setting.

Mechanism. Self-verification attempts to correct linear error accumulation inherent in Chain-of-Thought structures, but relies on verifiable correctness assumptions that collapse when intermediate steps are inherently unverifiable.

Proposal-style abstract. This project studies self-verification mechanisms within test-time compute scaling frameworks. The central question is how to maintain accuracy when verifiable correctness assumptions collapse on inherently unverifiable intermediate steps. Current approaches often focus on complex problem-solving tasks where correctness is easily verified on labeled datasets. However, reliance on task-specific few-shot examples can limit generalizability across diverse tasks. Furthermore, existing methods fail to utilize negative samples for implicit exploration and do not improve on out-of distribution tasks through experiments. The mechanism involves self-verification attempting to correct linear error accumulation inherent in Chain-of-Thought structures. Yet, this relies on assumptions that collapse when intermediate steps are unverifiable. This project would investigate verification-dependen error accumulation that achieves accuracy saturation despite increased inference-time search depth.

Table 13. Evolutionary-exploration formulation scores by domain.
<table><tr><td>Domain</td><td>Selected Mean overall</td><td></td><td>Best overall Mean</td></tr><tr><td>Bandits</td><td>3</td><td>3.75</td><td>4.25 5.17</td></tr><tr><td>In-context learning</td><td>4</td><td>3.64</td><td>4.33 5.29</td></tr><tr><td>Reasoning / test-time computation</td><td>1</td><td>4.50</td><td>4.50 5.75</td></tr><tr><td>Offline RL</td><td>1</td><td>3.75</td><td>3.75 5.25</td></tr><tr><td>Uncertainty estimation</td><td>6</td><td>3.86</td><td>4.00 5.57</td></tr></table>

Table 14. Bandits model-size sensitivity. Accepted variants are ambition-expanded formulations that pass the independent critic. Mean overall is reported for scoreable final families.
<table><tr><td>Model</td><td>Tuples</td><td>Verified gaps</td><td></td><td>; Direct Accepted variants </td><td>Families Formal statements Mean overall</td></tr><tr><td></td><td>204</td><td>8</td><td>8</td><td>0</td><td>0</td></tr><tr><td>4B 9B</td><td>1554</td><td>6</td><td>0 4</td><td>3</td><td>3 5.93</td></tr><tr><td>27B</td><td>1570</td><td>6 19 19</td><td>3</td><td>1</td><td>6.67</td></tr></table>

A possible approach is refining these mechanisms to handle unverifiable steps without saturation. A successful outcome would provide insights into scaling frameworks. This would clarify the relationship between search depth and verification limits. The proposed contribution type focuses on self-verification mechanisms within test-time compute scaling frameworks. This work aims to address the unrealistic assumption that intermediate steps remain verifiable under increased compute scaling.

Source grounding. Supporting papers include openreview:0vKokoPKTo, openreview:1OyE9IK0kx, openreview:4Po8d9GAfQ, openreview:70YeidEcYR, openreview:77gQUdQhE7, openreview:JtGPIZpOrz, openreview:P6dwZJpJ4m, openreview:YUYJsHOf3c, openreview:iEdEHPcFeu, and openreview:w6nlcS8Kkn. The originating motifs were assumption mismatch, repeated unaddressed limitation, and unrealistic assumption.

Score. Mean overall formulation quality: 4.50

These results clarify the role of evolutionary exploration. The branch is good at breadth: it produces many source-grounded hypotheses and can combine signals from several related gaps. The more modest formulation scores show where the main family-report path adds value. Formal problem statements, assumption statuses, risks, and ambiguity flags are not just formatting details; they are the stages that turn a promising hypothesis into an inspectable research-problem artifact. In this sense, the evolutionary branch is best viewed as a generator of candidate directions for further refinement, while the main SGHA path produces the final formalized families used in the primary comparison.

## 4.7 Model- and Corpus-Size Sensitivity

We end the results with two Bandit sensitivity studies. These experiments are not meant to establish a scaling law. Instead, they show how SGHA’s later stages respond when we change either the model used inside the pipeline or the amount of literature evidence available to it.

Model size. Table 14 compares the main 9B Bandits run with smaller 4B and larger 27B configurations. The 4B run produces a much smaller evidence base, but still reaches verification-passed gaps and direct formulations. The run stops before final family construction because none of the ambition-expanded variants pass the critic. The 9B and 27B runs both reach formalized project families. The 27B run produces more verification-passed gaps than the 9B run, but the critic and consolidation stages reduce them to one final family.

The model-size study shows that capacity changes both the evidence extracted upstream and the shape of the final funnel. A smaller model can still identify formulation-ready gaps, but the final output depends on whether the generated variants survive criticism. A larger model can surface many more verified gaps, while consolidation can still lead to a small final set. This is consistent with SGHA’s design: final families are selected through verification, criticism, and consolidation, rather than by a fixed output quota.

Table 15. Bandits corpus-size sensitivity. The 250-paper setting is the main Bandits run.
<table><tr><td>Selected papers</td><td>Tuples</td><td>Verified gaps</td><td>Direct</td><td>Accepted variants</td><td>Families</td><td>Formal statements</td><td>Mean overall</td></tr><tr><td>50</td><td>339</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td></td></tr><tr><td>100</td><td>706</td><td>4</td><td>4</td><td>3</td><td>2</td><td>2</td><td>5.80</td></tr><tr><td>150</td><td>1036</td><td>5</td><td>5</td><td>1</td><td>1</td><td>1</td><td>5.20</td></tr><tr><td>200</td><td>1392</td><td>10</td><td>10</td><td>6</td><td>5</td><td>5</td><td>6.12</td></tr><tr><td>250</td><td>1554</td><td>6</td><td>6</td><td>4</td><td>3</td><td>3</td><td>5.93</td></tr></table>

Corpus size. Table 15 varies the Bandits corpus from 50 to 250 selected papers. The 50-paper run reaches a verified gap and a direct formulation, but does not produce a final family. From 100 papers onward, SGHA reaches the formal problem stage. The 200-paper setting produces the largest number of final families and the highest mean overall score in this diagnostic, while the 250-paper setting remains close and produces a complete set of formalized outputs.

The corpus-size study gives a similar message. A richer corpus gives SGHA more evidence and more opportunities for candidate-gap discovery, but the final output is shaped by which gaps pass verification and which formulations survive the critic. The best result among these Bandits settings appears at 200 papers, while the 250-paper run still produces a strong but smaller final set. We therefore treat corpus size as a sensitivity variable: enough evidence is important, but the structure of that evidence, and how it passes through the later gates, matters just as much.

## 5 Discussion

In this paper, we ask whether research-problem generation can start from patterns in the literature rather than from open-ended ideation. The results suggest that this changes the output in a meaningful way. Strong ideation systems, especially AI-Scientist-v2 with Claude Opus, can produce polished and technically detailed proposals. However, SGHA produces a diferent kind of artifact: a problem family that keeps track of the evidence, the verified gap, the assumptions being questioned, and the definitions that remain unresolved. This diference is reflected in the scores as SGHA is strongest on source-grounded specificity, assumption-boundary clarity, formalizability, and ambiguity hygiene, while the strongest ideation baseline is competitive on clarity, technical detail, nontriviality, and scope control.

This distinction matters because a useful research problem is not only a well-written idea. A researcher also needs to know where the problem came from, what assumption is being relaxed or challenged, what would count as progress, and which parts of the formulation are still underspecified. The qualitative examples make this visible. The AI-Scientist-v2 outputs often read like proposal sketches, with motivations and evaluation plans, while the MOOSE-Star output is closer to a compact hypothesis grounded in an inspiration paper. The SGHA output is more structured: it includes provenance, notation, assumptions, a success criterion, risks, and ambiguity flags. These fields do not make the problem automatically correct or novel, but they make the candidate easier to inspect and refine.

The profile-conditioned and evolutionary experiments further show two ways to use the same system beyond the main run. Profile conditioning changes the evidence base, so the final problems can move toward a researcher-specific context while still passing through the same verification and formulation stages. On the other hand, evolutionary exploration serves a diferent purpose: it searches more broadly around literaturegrounded seeds. Its outputs are useful for breadth and source-grounded hypothesis generation, while the lower formalizability and ambiguity-hygiene scores show why the final family-report path remains important.

The sensitivity studies further are a useful caution against reading SGHA as a simple scaling pipeline. Larger models and larger corpora can change the evidence graph and the set of candidate gaps, but the final output is shaped by verification, criticism, and consolidation. This is appropriate for a literature-driven system: the goal is not to force a fixed number of ideas, but to return problem families when the evidence supports them. At the same time, the current evaluation is limited to formulation quality. The LLM judges do not establish external novelty, theoretical correctness, or future scientific impact, and the semi-formal statements are not finished theorem statements. SGHA should therefore be viewed as a tool for producing evidence-linked starting points that researchers can inspect, sharpen, and compare against the broader literature.

## 6 Conclusion

We presented SGHA, a training-free system for generating research-problem families from scientific literature. The central idea is to use the literature itself as the starting point: papers already contain assumptions, limitations, empirical patterns, and open directions that can inherently suggest useful research problems when considered together. SGHA turns these signals into a structured evidence graph, identifies candidate gaps, verifies them, and formulates the surviving candidates as project families with semi-formal problem statements.

Across five machine-learning domains, SGHA produced 15 formalized project families from 1,250 selected papers and 8,634 extracted evidence tuples. In the formulation-only evaluation, SGHA achieved the highest overall score among the compared methods and was strongest on source-grounded specificity, assumptionboundary clarity, formaliability, and ambiguity. The qualitative examples further show the same pattern: strong ideation baselines can produce polished proposal-style ideas, while SGHA produces structured problem artifacts with provenance, assumptions, objectives, risks, and ambiguity flags.

These results point to a broader role for scientific assistants. Beyond summarizing papers or generating ideas from a topic, they can help organize evidence across papers into clearer research-problem formulations. SGHA takes a step in that direction by making the path from literature evidence to problem statement explicit. This gives researchers a more grounded starting point for reading, refining, and developing new research directions.

## 7 Scope and Intended Use

SGHA is aimed at a particular kind of research-problem generation: problems whose evidence is already partly visible in a scientific corpus. Many useful directions begin this way. A set of papers may rely on the same strong assumption, report related failure modes, leave similar limitations unresolved, or evaluate methods under narrow conditions. SGHA is designed to collect these signals, organize them as evidence, and turn the strongest patterns into clearer problem formulations.

This scope is central to the system. SGHA is not positioned as a replacement for open-ended ideation; it is a complementary mode of ideation grounded in the literature. Open-ended ideation is useful when the goal is to speculate freely, connect distant areas, or invent directions that are not yet reflected in papers. SGHA is useful when the goal is to ask: given the literature we have, what problems are already being suggested by its assumptions, gaps, and tensions? The verification and formalization stages are designed around this setting

The evaluation follows the same framing. We assess whether SGHA produces clear, source-grounded, formalizable problem artifacts with explicit assumptions and ambiguity flags. We do not use the results as a claim that the generated problems are externally novel or already correct in the broader literature. Instead, the intended output is a structured starting point for researchers: a formulation that preserves why the problem arose, what evidence supports it, and what still needs to be sharpened before the direction becomes a complete research project.

## Generative AI Disclosure

The authors use ChatGPT for language editing and formatting. They also use the Stanford Agentic Reviewer<sup>1</sup> and the CMU Paper Reviewer [21] to obtain preliminary feedback and improve the paper. Codex was also used to organise the code and prepare the README files. All technical content, experimental design, analyses, and conclusions remain the sole responsibility of the authors.

## References

[1] Yangqiaoyu Zhou, Haokun Liu, Tejes Srivastava, Hongyuan Mei, and Chenhao Tan. Hypothesis generation with large language models. In Proceedings of the 1st Workshop on NLP for Science (NLP4Science), pages 117–139, 2024.

[2] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jef Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

[3] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jef Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.

[4] Yu Zhang, Xiusi Chen, Bowen Jin, Sheng Wang, Shuiwang Ji, Wei Wang, and Jiawei Han. A comprehensive survey of scientific large language models and their applications in scientific discovery. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8783–8817, 2024.

[5] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, et al. Towards an ai co-scientist. arXiv preprint arXiv:2502.18864, 2025.

[6] Shuo Ren, Can Xie, Pu Jian, Zhenjiang Ren, Chunlin Leng, and Jiajun Zhang. Towards scientific intelligence: A survey of llm-based scientific agents. arXiv preprint arXiv:2503.24047, 2025.

[7] Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang. Researchagent: Iterative research idea generation over scientific literature with large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6709–6738, 2025.

[8] Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2):235–256, 2002. ISSN 1573-0565. doi: 10.1023/A:1013689704352. URL https://doi.org/10.1023/A:1013689704352.

[9] T.L Lai and Herbert Robbins. Asymptotically eficient adaptive allocation rules. Advances in Applied Mathematics, 6(1):4–22, 1985. ISSN 0196-8858. doi: https://doi.org/10.1016/0196-8858(85)90002-8. URL https://www.sciencedirect.com/science/article/pii/0196885885900028.

[10] Sébastien Bubeck, Nicolò Cesa-Bianchi, and Gábor Lugosi. Bandits with heavy tail, 2012. URL https://arxiv.org/abs/1209.1727.

[11] Shubhada Agrawal, Sandeep Juneja, and Wouter M. Koolen. Regret minimization in heavy-tailed bandits. In Mikhail Belkin and Samory Kpotufe, editors, Conference on Learning Theory, COLT 2021, 15-19 August 2021, Boulder, Colorado, USA, volume 134 of Proceedings of Machine Learning Research, pages 26–62. PMLR, 2021. URL http://proceedings.mlr.press/v134/agrawal21a.html.

[12] Zonglin Yang and Lidong Bing. Moose-star: Unlocking tractable training for scientific discovery by breaking the complexity barrier. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

[13] Xu Li, Hanzhe Tu, and Xun Han. Graph2idea:retrieval-augmented scientific idea generation with graph-structured contexts, 2026. URL https://arxiv.org/abs/2606.09105.

[14] Qingyun Wang, Doug Downey, Heng Ji, and Tom Hope. SciMON: Scientific inspiration machines optimized for novelty. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 279–299, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.18. URL https://aclanthology.org/2024.acl-long.18/.

[15] Rosni Vasu, Chandrayee Basu, Bhavana Dalvi Mishra, Cristina Sarasua, Peter Clark, and Abraham Bernstein. HypER: Literature-grounded hypothesis generation and distillation with provenance. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25413–25438, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1292. URL https://aclanthology.org/2025.emnlp-main.1292/.

[16] Sarvesh Gharat and Junpei Komiyama. Prism: Problem discovery via structural motifs in knowledge graphs. In ICML 2026 AI for Science Workshop, 2026.

[17] Imad Aouali. Difusion models meet contextual bandits with large action spaces. Open Review, 2024.

[18] Imad Aouali. Difusion models meet contextual bandits. Advances in Neural Information Processing Systems, 38:166613–166651, 2026.

[19] Martin J Wainwright and Michael I Jordan. 11 a variational principle for graphical models. 2005.

[20] Barbara E Engelhardt, Michael I Jordan, John R Srouji, and Steven E Brenner. Genome-scale phylogenetic function annotation of large and diverse protein families. Genome research, 21(11):1969–1980, 2011.

[21] Seungone Kim, Dongkeun Yoon, Kiril Gashteovski, Juyoung Suk, Jinheon Baek, Pranjal Aggarwal, Ian Wu, Viktor Zaverkin, Spase Petkoski, Daniel R Schrider, et al. On the limits and opportunities of ai reviewers: Reviewing the reviews of nature-family papers with 45 expert scientists. arXiv preprint arXiv:2605.20668, 2026.

## A SGHA Prompts, Output Schemas, and Validation Rules

This appendix reports the prompt templates, output contracts, and validation rules used by SGHA. Runtime prompts contain instance-specific content, such as paper text, candidate-gap evidence, verification summaries, or project-family records. We therefore show the fixed prompt instructions with placeholders for dynamic fields, such as {paper\_text}, {gap\_json}, or {verification\_summary}.

SGHA uses language models for bounded extraction, review, and formulation stages. The system uses deterministic code for graph construction, motif detection, hard verification gating, family consolidation, and final report auditing. The main SGHA outputs are generated only after candidate gaps pass screening and verification.

## A.1 Evidence Extraction

The extraction stage reads a parsed paper and returns a structured paper object together with typed evidence tuples. The prompt asks for extracted scientific content from the given paper, not a summary or template.

## Prompt template.

You are a scientific information extractor. Read the paper below and extract structured information ,→ as JSON.

IMPORTANT: Do NOT return a schema or template. Return a JSON object with REAL extracted content ,→ from this specific paper.

Output keys:   
paper\_id   
claims   
limitations   
failure\_conditions   
contradictions\_or\_tensions   
future\_work   
methods   
assumptions   
tasks   
datasets   
metrics   
results   
tuples   
For each tuple:   
subject -- specific named entity   
relation -- MUST be one of: evaluated\_on, improves\_over, fails\_under, assumes,   
contradicts, uses\_dataset, measured\_by, limited\_by, addresses,   
not\_addressed\_by   
object   
evidence\_text -- exact supporting text from the paper   
section   
confidence -- value in [0, 1]   
claim\_type   
polarity   
condition   
task   
dataset   
model   
metric   
direction   
evidence\_type   
strength   
source\_span   
subject\_scope   
resolved\_by\_paper   
in\_related\_work

- Do not conflate assumptions with failures.

- Do not invent results, limitations, datasets, or methods.

Table 16. Evidence tuple schema used by SGHA.
<table><tr><td>Field</td><td>Meaning</td></tr><tr><td>subject</td><td>Specific method, model, assumption, task, dataset, metric, claim, or scientific object.</td></tr><tr><td>relation</td><td>One of evaluated_on, improves_over, fails_under, assumes, contradicts, uses_dataset, measured_by, limited_by, addresses, or</td></tr><tr><td>object</td><td>not_addressed_by. Target of the relation.</td></tr><tr><td>evidence_text /</td><td>Supporting text from the parsed paper</td></tr><tr><td>source_span section</td><td>Paper section containing the evidence.</td></tr><tr><td>confidence</td><td>Model confidence in the extraction.</td></tr><tr><td>claim_type</td><td>One of method, result, limitation, assumption, comparison, failure, hypothesis, evaluation, or unknown.</td></tr><tr><td>polarity</td><td>One of positive, negative, neutral, mixed, or unknown.</td></tr><tr><td>condition</td><td>Regime or condition under which the tuple holds, if available.</td></tr><tr><td>task, dataset, model, metric direction</td><td>Structured context fields, especially for evaluation-related claims.</td></tr><tr><td>evidence_type</td><td>One of improves, worsens, no_change, or unknown. One of empirical, theoretical, ablation, qualitative, assumption,</td></tr><tr><td></td><td>citation, benchmark, or unknown.</td></tr><tr><td>strength subject_scope</td><td>One of strong, moderate, weak, or unknown.</td></tr><tr><td>resolved_by_paper</td><td>One of own_method, prior_work, or general.</td></tr><tr><td></td><td>Whether the paper itself appears to resolve the issue.</td></tr><tr><td>in_related_work</td><td>Whether the evidence comes from related work or background discussion.</td></tr></table>

\- Extract content from the provided paper only.

\- Use specific scientific entities, not generic placeholders.

Rules:

\- Evidence text and source spans must be grounded in the parsed paper.

Paper-level schema.

```yaml
PaperExtraction:
paper_id: string
claims: list[string]
limitations: list[string]
failure_conditions: list[string]
contradictions_or_tensions: list[string]
future_work: list[string]
methods: list[string]
assumptions: list[string]
tasks: list[string]
datasets: list[string]
metrics: list[string]
results: list[string]
tuples: list[ScientificTuple]
```

## Tuple schema.

Validation rules. Before tuples enter the evidence graph, SGHA applies deterministic validation and cleanup. The relation must belong to the fixed relation vocabulary, and the subject, object, and evidence fields must be present. The supporting evidence span must match the parsed paper text after normalization, so that the tuple remains traceable to the source paper. SGHA removes generic subjects such as “method,” “algorithm,” or “our method,” filters overly generic task labels, drops evaluation tuples that do not include task, dataset, or metric context, and normalizes metadata such as relation polarity. For example, fails\_under and limited\_by are treated as negative relations, while assumes is treated as neutral.

## A.2 Graph Construction and Motif Detection

Graph construction and motif detection are deterministic. There is no LLM prompt for this stage. SGHA inserts validated paper objects and evidence tuples into a typed evidence graph.

Node and edge types.

Node types:   
Paper   
Method   
Task   
Dataset   
Metric   
Assumption   
Result   
Limitation   
FailureCondition   
Claim   
Gap   
Hypothesis   
Edge types:   
mentions   
proposes   
evaluated\_on   
uses\_dataset   
measured\_by   
improves\_over   
fails\_under   
assumes   
limited\_by   
contradicts   
addresses   
not\_addressed\_by   
supports\_gap   
weakens\_gap   
has\_mechanism   
feasible\_with   
derived\_from   
counterevidence\_for   
partially\_addresses\_gap   
relaxes\_assumption\_of   
handles\_failure\_of

Motif families. SGHA searches the typed graph using predefined motif detectors. A motif hit instantiates a candidate structural gap, but does not by itself verify that the gap is valid or novel.

Motif families:   
shared\_failure\_condition   
repeated\_unaddressed\_limitation   
conflicting\_claims   
sparse\_evaluation   
assumption\_mismatch   
unrealistic\_assumption   
shared\_unrealistic\_assumption   
theory\_practice\_gap   
transfer\_solution\_gap   
unresolved\_tradeoff   
missing\_stress\_test

- "trivial": gap is just "method X can fail" or "X has limitations" with no named assumption or   
,→ failure condition.   
- "known\_open": specific named gap experts know about but have not solved.   
- "known\_solved": already substantially addressed in literature published after 2021.   
- "novel": non-obvious pattern only visible by reading across multiple papers.

claim\_without\_measurement   
missing\_baseline\_comparison

Candidate-gap schema.

CandidateGap:   
gap\_id: string   
gap: string   
target: string   
scope: string   
mechanism: string   
supporting\_evidence: list   
counterevidence: list   
traceability\_path: list   
novelty\_score: float   
feasibility\_score: float   
impact\_score: float   
overall\_score: float   
motif\_type: string   
motif\_id: string   
paper\_ids: list[string]   
extraction\_uncertainty: float   
counterevidence\_edges: list   
counterevidence\_papers: list   
counterevidence\_status: string   
counterevidence\_summary: string   
remaining\_gap\_scope: string   
novelty\_status: string   
novelty\_rationale: string

The graph builder normalizes selected aliases, removes junk method labels, deduplicates parallel edges, and keeps provenance on graph edges. Candidate gaps are created from motif hits, not from arbitrary missing graph edges.

## A.3 Novelty and Corpus-Coverage Filtering

After motif detection, SGHA applies a corpus-grounded novelty and coverage screen. This screen removes candidates that appear trivial, already solved, or already covered by nearby evidence in the selected corpus. It is not an external literature novelty proof.

## Prompt template.

You are a senior ML researcher evaluating structural gaps found by automated analysis of a paper ,→ corpus.

Research area: {topic\_description}

Classify each gap into exactly one category:

Be strict about trivial. Be fair about known\_open when the gap names a specific assumption or ,→ condition.

{gap\_blocks}

Respond with a JSON object:   
{   
"classifications": [

confidence must be a float between 0.0 and 1.0.   
scores must be a dict with keys: evidence\_support, novelty, feasibility, specificity,   
,→ scope\_validity (all floats 0.0-1.0).

{"gap\_index": 1, "class": "<trivial|known\_open|known\_solved|novel>", "reason": "<one   
,→ sentence>"}   
]   
}

Filtering rules. SGHA keeps candidates labeled novel, known\_open, or partially\_addressed\_in\_corpus. It removes candidates labeled trivial, known\_solved, or known\_solved\_in\_corpus. If corpus counterevidence already marks a candidate as solved within the selected evidence base, the LLM classification can be skipped. If a batch classification returns fewer labels than expected, missing labels default to keep with an error rationale; later verification applies the stricter gate.

## A.4 Verification Agents and Hard Gate

Novelty survivors selected for verification are reviewed by role-specific agents. The agents use the same output schema but diferent role instructions. The support agent looks for evidence supporting the gap. The skeptic agent looks for counterevidence or already-addressed risk. The feasibility agent checks whether the problem can plausibly be studied. The mechanism agent checks whether there is a plausible explanation for the gap. The verification critic gives a conservative final assessment.

Verification-agent output schema.

```json
{
"gap_id": "<copy from gap>",
"agent_name": "<support|skeptic|feasibility|mechanism|critic>",
"summary": "<one sentence assessment>",
"evidence": [{"paper_id": "...", "evidence_text": "..."}],
"counterevidence": [{"paper_id": "...", "evidence_text": "..."}],
"citations": ["paper_id_1", "paper_id_2"],
"confidence": 0.0,
"failure_modes": ["..."],
"scores": {
"evidence_support": 0.0,
"novelty": 0.0,
"feasibility": 0.0,
"specificity": 0.0,
"scope_validity": 0.0
}
}
```

Repair prompt. If an agent response is not valid JSON, SGHA retries once with a repair prompt:

The previous response was invalid JSON. Return ONLY valid JSON with these keys:   
gap\_id, agent\_name, summary, evidence, counterevidence, citations, confidence, failure\_modes,   
,→ scores.

Hard verification gate. The hard verification gate is deterministic. A reviewed gap enters formulation only if the gate accepts it.

```yaml
verification_gate:
enabled: true
mode: survival_score
min_survival_score: 0.60
require_all_agents: true
require_critic_non_reject: true
min_critic_confidence: 0.50
```

```yaml
fail_on_agent_parse_failure: true
allow_reviewed_only_fallback: false
```

A candidate fails the gate if required agent outputs are missing, if an agent has a disqualifying parse failure, if the survival score is missing or below threshold, or if the critic rejects the candidate or has insuficient confidence.

## A.5 Direct Formulation

The direct-formulation prompt is used only for gaps that pass the hard verification gate. It asks for a single evidence-close problem formulation.

## Prompt template.

You are formulating ONE direct, coherent research problem from a single verification-passed ,→ structural gap. This is a PROPOSAL (no results yet).

DECLARED RUN CONTEXT   
- domain\_name: {domain\_name}   
- topic\_description: {topic\_description}   
- keyphrases: {keyphrases}

STRICT COHERENCE RULES:   
- exactly one core setting, one main assumption/failure/limitation, one core objective   
- one plausible theorem OR algorithm OR benchmark/empirical target   
- do NOT combine unrelated methods/assumptions/failures into a collage   
- do NOT introduce domain-specific concepts unless they appear in the declared run context, source   
,→ gap, supporting paper titles, extracted evidence, or verification findings   
- use the domain's own entities, objectives, assumptions, feedback/measurement model, and   
,→ evaluation targets   
- if the evidence is thin, say so and lower the scores

```yaml
SOURCE VERIFICATION-PASSED GAP
- gap_id: {gap_id}
- motif_type: {motif_type}
- target: {target}
- gap statement: {gap}
- mechanism: {mechanism}
- scope: {scope}
- supporting paper titles: {papers}
- extracted evidence snippets: {extracted_evidence}
```

## Output schema.

"direct\_title": "",   
"direct\_problem\_statement": "",   
"proposal\_style\_abstract": "",   
"core\_setting": "",   
"core\_assumption\_or\_failure": "",   
"core\_objective": "",   
"possible\_theorem\_target": "",   
"possible\_algorithmic\_target": "",   
"possible\_empirical\_target": "",   
"coherence\_score": 0.0,   
"specificity\_score": 0.0,   
"feasibility\_score": 0.0,   
"term\_soup\_risk": 0.0,   
"main\_risk": "",   
"falsification\_condition": "",   
"recommendation": "KEEP | MAYBE | DROP"

SGHA marks a direct formulation as strong when coherence, specificity, and term-soup-risk checks pass and the recommendation is KEEP. Low-coherence or high-term-soup formulations are deprioritized or removed before later stages.

## A.6 Ambition Expansion and Critic

Ambition expansion starts from a direct formulation. The goal is to broaden the problem where useful while keeping it tied to the verified evidence.

## Ambition-expansion prompt template.

You are doing CONTROLLED ambition expansion of ONE clean, single-gap research formulation. Goal: a genuinely NON-INCREMENTAL but still coherent formulation -- NOT a multi-concept collage, and NOT just validating one named method in one more setting.

## SOURCE DIRECT FORMULATION:

A formulation is INCREMENTAL if its central contribution is only:

- evaluating one named method in one additional setting

\- testing robustness of one paper's method without changing the scientific question

\- a small benchmark variation

\- restating a limitation already identified by one source paper

\- depending heavily on one named method

\- a cleaner title for the same narrow gap

A formulation is NON-INCREMENTAL if it:

\- identifies a broader problem class

\- characterizes a boundary / impossibility / phase transition / identifiability limit / failure

,→ regime

\- proposes a constructive alternative at method/system/mechanism/evaluation-class level

\- defines benchmark/evaluation protocol exposing a general failure mode

\- gives a unifying explanation connecting observations under one structural condition

## Ambition-variant schema.

## {

```csv
"title": "",
"problem_statement": "",
"proposal_style_abstract": "",
"core_setting": "",
"generalized_axis": "",
"core_assumption_or_failure": "",
"core_objective": "",
"contribution_type":
,→ "theorem|algorithm|lower_bound|benchmark|empirical_study|impossibility_boundary",
"theorem_target": "",
"algorithmic_target": ""
"empirical_target": "",
"broader_problem_class": ""
"assumption_shift": "",
"boundary_or_failure_regime": "",
"constructive_or_explanatory_target": "",
"method_class_scope": "",
"named_method_dependency": "low|medium|high",
"why_not_just_validation": "",
"why_not_incremental": "",
"why_not_term_soup": "",
"non_incrementality_score": 0.0,
"incrementality_risk": 0.0,
"main_risk": "",
"falsification_condition": "",
"coherence_score": 0.0,
```

SOURCE DIRECT FORMULATION:   
{direct\_formulation}

"ambition\_score": 0.0,   
"specificity\_score": 0.0,   
"feasibility\_score": 0.0,   
"term\_soup\_risk": 0.0,   
"recommendation": "KEEP|MAYBE|DROP"   
}

## Ambition-critic prompt template.

You are an INDEPENDENT critic. You did NOT write the formulation below and you must NOT rewrite, fix, or improve it. You only JUDGE whether it is genuinely non-incremental.

Judge strictly:   
1. Is this genuinely non-incremental relative to the source direct formulation?   
2. Did it change the scientific object, or just wording?   
3. Is broader\_problem\_class real and specific?   
4. Is the contribution meaningful beyond one named method?   
5. Is the claimed boundary/impossibility/algorithm/benchmark supported by source evidence?   
6. Is contribution\_type inflated?

VARIANT TO JUDGE: {ambition\_variant}

Ambition-critic schema.

```json
{
"critic_non_incrementality_score": 0.0,
"critic_incrementality_risk": 0.0,
"critic_fake_ambition_risk": 0.0,
"critic_named_method_dependency": "low|medium|high",
"critic_broader_problem_class_valid": true,
"critic_contribution_type_valid": true,
"critic_supported_by_source": "strong|moderate|weak",
"critic_quality_label":
,→ "STRONG_NON_INCREMENTAL|VALID_BUT_MODEST|INCREMENTAL_REPHRASING|FAKE_AMBITION|DROP",
"critic_reason": ""
}
```

A variant passes the critic when it is suficiently non-incremental, source-supported, and specific, with low fake-ambition and incrementality risk. SGHA keeps critic-passing variants for family consolidation.

## A.7 Family Consolidation

Family consolidation is deterministic in the main runs. It groups critic-passing variants into project families using strict anchors such as shared verified gaps, shared direct formulations, overlapping supporting papers, and strong identity overlap.

## Family schema.

{   
"family\_id": "family:01",   
"family\_title": "",   
"member\_variant\_ids": [],   
"representative\_variant\_id": "",   
"representative\_title": "",   
"family\_problem\_statement": "",   
"proposal\_style\_abstract": "",   
"theorem\_target": "",   
"algorithmic\_target": "",   
"empirical\_target": "",   
"main\_risk": "",

```python
"falsification_condition": "",
"critic_reason": "",
"research_object": "",
"problem_class": "",
"assumption_shift": "",
"failure_boundary_or_mechanism": "",
"constructive_or_evaluation_target": "",
"source_verified_gaps": [],
"source_direct_formulations": [],
"supporting_papers": [],
"related_seed_papers": [],
"quality_label": "A_STRONG|B_PROMISING_NEEDS_REFRAMING|C_LOW_PRIORITY|D_DROP",
"recommended_action": "READ_FIRST|REFRAME|READ_LATER|DROP"
```

The consolidation stage avoids weak transitive grouping. All selected variants must be assigned consistently, and no invented variant identifiers are allowed.

## A.8 Formal Problem Formulation

For each final project family, SGHA produces a semi-formal problem object. The prompt asks for a structured problem statement rather than a finished theorem.

## Prompt template.

You are SGHA's domain-general formal problem formulation stage.

You must formalize ONE existing project family. Do NOT invent a new research direction, new family, or new claim of results. Use only the source family and formulation evidence below.

STRICT RULES:   
- Formalize the existing project family only.   
- Prefer clear semi-formal structure over fake mathematics.   
- If a term is vague, define it cautiously or put it in ambiguity\_flags.   
- If the feedback/data/measurement model is unclear, say so.   
- If the objective cannot be formalized from evidence, set formalization\_confidence to "low".   
- Every symbol introduced in the formal\_problem\_statement must appear in the variable table.   
- Do NOT use metadata variables such as R = research object, C = problem class,   
A = assumption shift, or B = boundary.   
- Variables must correspond to problem-level objects: actions, observations, latent factors,   
measurements, rewards/outcomes, policies/decisions, constraints, objectives, estimators,   
environments, systems, functions, processes, or distributions.   
- Every major assumption must be marked kept, relaxed, removed, or questioned.   
- Do not say "we prove", "we show", or imply results already exist.   
PROJECT FAMILY:   
{project\_family}

Formal-problem schema.

```json
{
"plain_language_problem": "",
"formal_problem_statement": "",
"mathematical_setup": {
"entities": [""],
"variables": [
{
"symbol": "",
"meaning": "",
"type": "scalar | vector | set | distribution | process | function | other",
"source": "from evidence | introduced for formalization"
}
],
"data_or_observations": "",
```

"feedback\_or\_measurement\_model": "",   
"decision\_variables\_or\_outputs": "",   
"objective": "",   
"constraints": "",   
"success\_criterion": ""   
},   
"assumptions": [   
{   
"name": "",   
"description": "",   
"status": "kept | relaxed | removed | questioned",   
"source\_evidence": [""]   
}   
],   
"open\_question": "",   
"possible\_result\_types": {   
"theorem": "",   
"algorithm": "",   
"empirical\_or\_benchmark": ""   
},   
"evaluation\_protocol": "",   
"ambiguity\_flags": [   
{   
"term": "",   
"why\_ambiguous": "",   
"what\_user\_must\_define": ""   
}   
],   
"source\_grounding": {   
"source\_verified\_gaps": [""],   
"supporting\_papers": [""],   
"representative\_formulation": "",   
"critic\_reason": ""   
},   
"formalization\_confidence": "high | medium | low",   
"formalization\_risk": "",   
"requires\_human\_definition": true

If the formal-problem output is not valid JSON, SGHA retries once with a repair prompt. If repair fails, the system uses a source-grounded low-confidence fallback and marks the formalization as requiring human definition.

## A.9 Profile-Conditioned Additions

Profile-conditioned SGHA uses the same downstream prompts and gates after corpus construction. The profile changes the evidence base and prioritization, not the validity checks.

Seed-paper schema.

SeedPaper:   
seed\_paper\_id   
title   
authors   
year   
venue   
abstract   
local\_pdf\_path   
arxiv\_id   
openreview\_id   
doi   
url   
source   
relation\_to\_seed\_profile   
is\_manual\_seed\_paper

seed\_label   
metadata\_incomplete   
provenance

## Profile schema.

{   
"seed\_label": "",   
"topic": "",   
"seed\_titles": [],   
"keyphrases": [],   
"method\_terms": [],   
"task\_terms": [],   
"assumption\_terms": [],   
"benchmark\_terms": [],   
"generated\_openreview\_queries": [],   
"venues": [],   
"exclusion\_terms": [],   
"profile\_summary": ""   
}

In profile-conditioned mode, Stage 7 and later stages may receive seed-alignment information and related seed papers as context. Verification, formulation, ambition criticism, family consolidation, and formalization remain the same.

## A.10 Evolutionary Exploration

The evolutionary branch is separate from the main family-report path. It starts from literature-grounded seeds and generates nearby variants by changing parts of the problem representation, such as assumptions, mechanisms, objectives, settings, or regimes. These variants are critiqued, selected, and refined over multiple rounds.

## Evolutionary-candidate schema.

EvolutionaryHypothesis:   
candidate\_id: string   
source\_seed\_id: string   
parent\_ids: list[string]   
generation: integer   
mutation\_type: string   
title: string   
problem\_statement: string   
gap: string   
target: string   
scope: string   
mechanism: string   
proposal\_style\_abstract: string   
source\_grounding: list   
motif\_lineage: list   
evolution\_score: float   
rank: integer   
selected: boolean

The evolutionary branch is used for breadth. Its selected or ranked hypotheses can be evaluated with the same formulation-only rubric, but they are reported separately from the final verification-gated project families.

## A.11 Final Report Auditing

Final report rendering is deterministic in the main runs. The audit checks that final families are present or that a low-signal status is recorded; that direct formulations trace to verification-passed gaps; that selected variants trace to direct formulations; that final families trace to accepted variants; and that formal problem JSON exists when formalization is enabled. It also checks that novelty-only or reviewed-only candidates do not enter the default family-report path.

A low-signal status means the pipeline completed but did not produce a final family because the evidence or downstream checks did not support one.

## B LLM-Judge Rubric and Candidate Normalization

In this appendix, we describe how generated candidates are normalized and evaluated by the formulation-only LLM judges.

## B.1 Candidate Packet Format

Before judging, outputs from all methods are converted into a common candidate-packet format. The packet preserves the fields produced by the original method. Fields that are not produced by a method are left as not provided; they are not filled in manually after generation.

CandidatePacket:   
candidate\_id: string   
method\_label: hidden during judging   
domain: string   
title: string   
problem\_statement: string   
motivation\_or\_abstract: string   
proposed\_direction: string   
expected\_contribution: string   
evaluation\_plan: string   
risks\_or\_caveats: string   
source\_context\_or\_grounding: string   
assumptions\_or\_problem\_setup: string   
formal\_problem\_statement: string   
ambiguity\_or\_missing\_definitions: string

This format lets the judge compare artifacts produced by diferent systems while preserving their natural structure. SGHA final families often include formal problem statements, assumptions, source provenance, and ambiguity flags. However, ideation baselines often include motivation, proposed directions, evaluation plans, and caveats. The normalization step keeps these diferences visible.

## B.2 Formulation-Only Judge Prompt

The formulation-only judge evaluates the quality of a research-problem formulation.

Judge prompt template.

You are a strict, skeptical senior research reviewer evaluating research-problem formulation ,→ quality only.

Your task is to judge the formulation as a problem statement, not as a project plan.   
Do not evaluate implementation plans, experiments, software engineering details, or actionability ,→ except where they clarify the formulation itself.   
Do not reward a candidate for having a polished plan if the underlying research problem is vague or ,→ poorly posed.

Judge only the provided blinded candidate text.   
Do not use external knowledge.   
Do not perform an external novelty check.   
If a field says "not provided", treat it as missing.

Table 17. Formulation-only LLM-judge rubric.
<table><tr><td>Criterion</td><td>What the judge evaluates</td></tr><tr><td>Problem-definition clarity</td><td>Whether the candidate states a clear research problem rather than a broad topic or loose motivation.</td></tr><tr><td>Technical specificity</td><td>Whether the formulation names concrete objects, settings, mechanisms,</td></tr><tr><td>Well-posedness</td><td>assumptions, metrics, or target results. Whether the problem has enough structure to be studied, refined, or</td></tr><tr><td>Assumption-boundary clarity</td><td>formalized. Whether the formulation makes clear which assumptions are used, relaxed, questioned, removed, or missing.</td></tr><tr><td>Formalizability</td><td>Whether the problem can plausibly be written as a theorem, algorithmic objective, benchmark protocol, impossibility result, or other formal</td></tr><tr><td>Nontriviality</td><td>research target. Whether the candidate goes beyond a simple application, small</td></tr><tr><td>Scope control</td><td>robustness check, or generic “apply X to Y&quot; idea. Whether the problem is focused enough to be studied rather than</td></tr><tr><td>Source-grounded specificity</td><td>combining many loosely related concepts. Whether the formulation is tied to provided source context, evidence, or</td></tr><tr><td>Ambiguity hygiene</td><td>literature-derived details. Whether the candidate explicitly states missing definitions, unclear</td></tr><tr><td>Overall formulation quality</td><td>feedback models, unresolved assumptions, or caveats. The judge&#x27;s overall assessment of the research-problem formulation under the rubric.</td></tr></table>

Score the candidate using the rubric below.

Return only valid JSON matching the requested schema.

CANDIDATE PACKET: {candidate\_packet}

RUBRIC: {rubric}

SCORE ANCHORS: {score\_anchors}

CAP RULES: {cap\_rules}

Return JSON only.

## B.3 Formulation Rubric

Table 17 lists the ten formulation-only criteria. Each criterion is scored from 0 to 10.

## B.4 Score Anchors

The judges use the anchors in Table 18. These anchors are included in the judge prompt to make the scale consistent across models.

## B.5 Score Consistency Rules

The judge prompt includes cap rules to keep scores consistent when important information is missing. These rules do not assign scores directly. They limit the maximum score for a criterion when the candidate lacks the corresponding field or structure.

Table 18. Score anchors used by the formulation-only judges.
<table><tr><td>Score</td><td>Interpretation</td></tr><tr><td>0</td><td>No usable formulation.</td></tr><tr><td>1</td><td>Mostly incoherent or unrelated to a research problem.</td></tr><tr><td>2</td><td>Topic-level idea with almost no problem structure.</td></tr><tr><td>3</td><td>Very vague formulation with major missing pieces.</td></tr><tr><td>45</td><td>Weak formulation; some direction is visible, but the problem is poorly specified.</td></tr><tr><td></td><td>Plausible idea, but important assumptions, scope, or definitions are missing.</td></tr><tr><td>6</td><td>Plausible formulation with useful structure, but still needing substantial refinement.</td></tr><tr><td>7</td><td>Strong formulation with clear problem structure and reasonable technical specificity.</td></tr><tr><td>8</td><td>Very strong formulation; well posed, grounded, and mostly ready for expert refinement.</td></tr><tr><td>9</td><td>Excellent formulation with unusually clear assumptions, scope, and formal target.</td></tr><tr><td>10</td><td>Exceptional formulation; clear, grounded, formalizable, and close to research-ready.</td></tr></table>

Table 19. Score consistency rules used by the formulation-only judge.
<table><tr><td>Condition</td><td>Rule</td></tr><tr><td>Problem statement is vague or mostly a topic description</td><td>Overall formulation quality should be at most 6.</td></tr><tr><td></td><td>Formal problem statement is not provided Well-posedness and formalizability should be at most 6.</td></tr><tr><td>Assumptions or setup are not provided</td><td>Well-posedness and assumption-boundary clarity should be at most 6.</td></tr><tr><td>Ambiguity or missing-definition field is not provided</td><td>Ambiguity hygiene should be at most 6.</td></tr><tr><td>Source or context grounding is not provided</td><td>Source-grounded specificity should be at most 4.</td></tr><tr><td>Candidate combines many loosely related Scope control should be at most 4. concepts</td><td></td></tr><tr><td>Candidate is mainly &quot;apply X to Y&quot; or one-method validation</td><td>Nontriviality should be at most 6.</td></tr><tr><td>No clear formal skeleton is visible</td><td>Overall formulation quality should be at most 7.</td></tr></table>

## B.6 Judge Response Schema

Each judge returns a strict JSON object. The schema records the criterion-level scores, a recommendation, confidence, strengths, weaknesses, rationale, and a novelty caveat.

{   
"candidate\_id": "",   
"domain": "",   
"scores": {   
"problem\_definition\_clarity\_10": 0,   
"technical\_specificity\_10": 0,   
"well\_posedness\_10": 0,   
"assumption\_boundary\_clarity\_10": 0,   
"formalizability\_10": 0,   
"nontriviality\_10": 0,   
"scope\_control\_10": 0,   
"source\_grounded\_specificity\_10": 0,   
"ambiguity\_hygiene\_10": 0,   
"overall\_formulation\_quality\_10": 0   
},   
"recommended\_action": "READ\_FIRST | PROMISING\_NEEDS\_REFINEMENT | NEEDS\_REFRAMING |   
,→ DROP\_OR\_DEPRIORITIZE",   
"confidence": "LOW | MEDIUM | HIGH",   
"strengths": ["..."],   
"weaknesses": ["..."],   
"rationale": "...",

"novelty\_caveat": "Novelty potential judged only from provided text; no external novelty check ,→ performed." }

The reported overall\_formulation\_quality\_10 score is the judge’s holistic assessment under the rubric.   
It is not computed as a weighted average of the other criteria.

## B.7 Judge Panel and Blinding

We use five LLM judges from diferent providers:

anthropic/claude-sonnet-4   
openai/gpt-5.6-sol-pro   
x-ai/grok-4.5   
moonshotai/kimi-k3   
google/gemini-3.6-flash

Each judge receives the same blinded candidate packet format, rubric, score anchors, cap rules, and JSON response schema. Method labels are hidden during scoring. The blinding key is used only after scoring, during postprocessing. The judge panel is used only after all candidate outputs have been frozen.

## B.8 Calibration Examples

Before scoring the real candidates, each judge is checked on calibration examples. These examples are designed to test whether the judge applies the rubric in the intended direction. The calibration set includes vague topic-level ideas, underformalized formulations, over-broad or term-heavy formulations, and stronger source-grounded formal problem statements.

Calibration is used as a quality check for the judge behavior. It is not used to tune separate prompts for diferent judges; the rubric and response schema remain fixed.

## B.9 Personalized Judge

Profile-conditioned candidates are evaluated with a personalized judge rubric. This rubric keeps the ten formulation-quality criteria and adds profile-specific criteria. The personalized judge uses only the provided profile context and candidate text; it does not use external knowledge about the researcher.

## Personalized judge prompt template.

You are a strict, skeptical senior research reviewer evaluating personalized research-problem ,→ formulations.

Your task is to judge both formulation quality and personalization quality.   
Judge profile alignment only from the provided profile context and the candidate text.   
Do not use external knowledge about the researcher.   
If a field says "not provided", treat it as missing.   
Do not reward name-dropping.   
A good personalized problem connects to artifact-supported profile themes, source/corpus evidence,   
and profile-relevant technical style while formulating a meaningful next problem.   
Penalize generic problems that could apply to many researchers in the area.   
Penalize off-profile problems even if technically coherent.   
PROFILE CONTEXT:   
{profile\_context}   
CANDIDATE PACKET:   
{candidate\_packet}   
Return JSON only.

Additional personalized criteria.

Table 20. Additional criteria used by the personalized judge.
<table><tr><td>Criterion</td><td>Meaning</td></tr><tr><td>Profile alignment</td><td>Whether the candidate problem is supported by the provided profile-derived literature context.</td></tr><tr><td>Profile specificity</td><td>Whether the problem is specific to that context rather than a generic domain problem.</td></tr><tr><td>Intellectual-style match</td><td>Whether the problem matches the profile-associated technical style or research pattern.</td></tr><tr><td>Profile novelty fit</td><td>Whether the candidate plausibly extends the profile-associated literature rather than repeating it.</td></tr><tr><td>Personalization overall</td><td>Overall quality of the personalization, considering the above criteria.</td></tr></table>

Personalized response schema.

```json
{
"candidate_id": "",
"profile": "",
"scores": {
"problem_definition_clarity_10": 0,
"technical_specificity_10": 0,
"well_posedness_10": 0,
"assumption_boundary_clarity_10": 0,
"formalizability_10": 0,
"nontriviality_10": 0,
"scope_control_10": 0,
"source_grounded_specificity_10": 0,
"ambiguity_hygiene_10": 0,
"overall_formulation_quality_10": 0,
"profile_alignment_10": 0,
"profile_specificity_10": 0,
"intellectual_style_match_10": 0,
"profile_novelty_fit_10": 0,
"personalization_overall_10": 0
},
"recommended_action": "READ_FIRST | PROMISING_NEEDS_REFINEMENT | NEEDS_REFRAMING |
,→ DROP_OR_DEPRIORITIZE",
"confidence": "LOW | MEDIUM | HIGH",
"strengths": ["..."],
"weaknesses": ["..."],
"rationale": "...",
"personalization_caveat": "Profile alignment judged only from provided profile context; no
,→ external profile knowledge used.",
"novelty_caveat": "Novelty potential judged only from provided text; no external novelty check
,→ performed."
}
```

The personalized judge also uses additional consistency rules. For example, candidates that only name-drop the researcher receive low profile-alignment scores, and generic domain problems without profile-specific evidence receive lower profile-specificity scores.

## C Additional Result Tables

This appendix provides detailed count and score tables that support the main results. The main text reports the central trends; here we include the full SGHA stage counts and selected per-domain formulation scores.

## C.1 Full SGHA Pipeline Counts

Table 21 reports the full SGHA count funnel for the five main domains. The table separates paper-level counts, graph-level counts, gap-level counts, and formulation-level counts. These quantities are not one-to-one:

Table 21. Full SGHA pipeline counts across the five main domains. Motif hits are deterministic graph-pattern matches. Novelty survivors are candidate gaps retained after corpus screening. Reviewed gaps are sent to verification agents. Verified gaps pass the hard verification gate. Accepted variants are ambition-expanded formulations that pass the independent critic.
<table><tr><td>Domain</td><td>Selected</td><td>Parsed Extracted</td><td></td><td>Tuples</td><td>Nodes Edges</td><td>Motif hits</td><td>Novelty surv.</td><td>Reviewed</td><td>Verified</td><td>Direct</td><td>Accepted variants</td><td>Families</td><td></td><td>Formal</td></tr><tr><td>Bandits</td><td>250</td><td>221</td><td>221</td><td>1554</td><td>9214</td><td>9776</td><td>173</td><td>30</td><td>13</td><td>6</td><td>6</td><td></td><td></td><td></td></tr><tr><td>In-context learning</td><td>250</td><td>193</td><td>193</td><td>1559</td><td>8608</td><td>9083</td><td>147</td><td>21</td><td>16</td><td>12</td><td>12</td><td>4</td><td>4</td><td>4</td></tr><tr><td>Reasoning / test-time computation</td><td>250</td><td>137</td><td>137</td><td>1139</td><td>6116</td><td>6535</td><td>105</td><td>29</td><td>12</td><td>7</td><td>7</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Offline reinforcement</td><td>250</td><td>248</td><td>245</td><td>2176</td><td>10600</td><td>12045</td><td>193</td><td>8</td><td>8</td><td>2</td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>learning Uncertainty estimation</td><td>250</td><td>250</td><td>248</td><td>2206</td><td>11097</td><td>12055</td><td>218</td><td>17</td><td>15</td><td>12</td><td>12</td><td>9</td><td>6</td><td>6</td></tr><tr><td>Total</td><td>1250</td><td>1049</td><td>1044</td><td>8634</td><td>45635</td><td>49494</td><td>836</td><td>105</td><td>64</td><td>39</td><td>39</td><td>19</td><td>15</td><td>15</td></tr></table>

Table 22. Selected per-domain formulation-quality scores. Scores are averaged over retained candidates and LLM judges.
<table><tr><td>Domain</td><td>Method</td><td>Overall</td><td>Source grounding</td><td>Formalizability</td><td>Ambiguity hygiene</td></tr><tr><td>Bandits</td><td>SGHA</td><td>5.93</td><td>7.33</td><td>5.20</td><td>7.80</td></tr><tr><td>Bandits</td><td>AI-Scientist-v2 + Qwen</td><td>4.53</td><td>5.20</td><td>4.00</td><td>2.93</td></tr><tr><td>Bandits</td><td>AI-Scientist-v2 + Claude Opus</td><td>5.87</td><td>2.80</td><td>5.07</td><td>3.47</td></tr><tr><td>Bandits</td><td>MOOSE-Star</td><td>1.93</td><td>3.47</td><td>1.27</td><td>1.40</td></tr><tr><td>In-context learning</td><td>SGHA</td><td>5.80</td><td>7.25</td><td>5.35</td><td>7.55</td></tr><tr><td>In-context learning</td><td>AI-Scientist-v2 + Qwen</td><td>4.35</td><td>4.60</td><td>3.40</td><td>2.60</td></tr><tr><td>In-context learning</td><td>AI-Scientist-v2 + Claude Opus</td><td>5.80</td><td>3.00</td><td>5.30</td><td>3.70</td></tr><tr><td>In-context learning</td><td>MOOSE-Star</td><td>1.95</td><td>4.00</td><td>1.55</td><td>1.50</td></tr><tr><td>Reasoning / test-time computation</td><td>SGHA</td><td>7.20</td><td>7.80</td><td>6.40</td><td>7.80</td></tr><tr><td>Reasoning / test-time computation</td><td>AI-Scientist-v2 + Qwen</td><td>4.60</td><td>5.40</td><td>3.40</td><td>2.40</td></tr><tr><td>Reasoning / test-time computation</td><td>AI-Scientist-v2 + Claude Opus</td><td>6.00</td><td>3.40</td><td>5.40</td><td>3.40</td></tr><tr><td>Reasoning / test-time computation</td><td>MOOSE-Star</td><td>2.00</td><td>4.00</td><td>1.40</td><td>1.40</td></tr><tr><td>Offline reinforcement learning</td><td>SGHA</td><td>6.40</td><td>7.60</td><td>6.00</td><td>7.80</td></tr><tr><td>Offline reinforcement learning</td><td>AI-Scientist-v2 + Qwen</td><td>4.80</td><td>5.60</td><td>3.80</td><td>3.00</td></tr><tr><td>Offline reinforcement learning</td><td>AI-Scientist-v2 + Claude Opus</td><td>5.40</td><td>2.80</td><td>4.80</td><td>3.00</td></tr><tr><td>Offline reinforcement learning</td><td>MOOSE-Star</td><td>2.40</td><td>4.40</td><td>2.00</td><td>1.60</td></tr><tr><td>Uncertainty estimation</td><td>SGHA</td><td>5.87</td><td>7.50</td><td>5.53</td><td>7.50</td></tr><tr><td>Uncertainty estimation</td><td>AI-Scientist-v2 + Qwen</td><td>4.63</td><td>5.27</td><td>3.73</td><td>2.93</td></tr><tr><td>Uncertainty estimation</td><td>AI-Scientist-v2 + Claude Opus</td><td>5.90</td><td>3.00</td><td>5.27</td><td>3.67</td></tr><tr><td>Uncertainty estimation</td><td>MOOSE-Star</td><td>2.00</td><td>3.57</td><td>1.53</td><td>1.50</td></tr></table>

one paper can produce many evidence tuples, several tuples can instantiate one motif, and several accepted variants can later be consolidated into one project family.

## C.2 Selected Per-Domain Formulation Scores

Table 22 reports per-domain formulation scores for the main methods. The main text reports method-level averages. Here, we show the overall score and three criteria that are especially relevant to SGHA: source grounding, formalizability, and ambiguity hygiene.

The per-domain scores are consistent with the aggregate results in the main text: SGHA is strongest on source grounding and ambiguity hygiene across domains, while the strongest ideation baseline is often competitive on overall formulation quality.
# Bringing analytic rigor to agentic AI for science: The Brain Researcher platform for neuroimaging data analysis

Zijiao Chen<sup>1</sup>, Nicholas Lu<sup>1</sup>, Xinhui Li<sup>2</sup>, Jocelyn A. Ricard<sup>1</sup>, Ce Ju<sup>3</sup>, Huan H. Wang<sup>1</sup>, Christian Kindermann<sup>1</sup>, Jeanette A. Mumford<sup>1</sup>, Steven Dillmann<sup>1</sup>, James Kent<sup>4</sup>, Alejandro de la Vega<sup>4</sup>, Sanmi Koyejo<sup>1</sup>, Vince D. Calhoun<sup>2</sup>, Joshua W. Buckholtz<sup>1</sup>, Juan Helen Zhou<sup>5</sup>, Stefen Bollmann<sup>1,6</sup>, Russell A. Poldrack<sup>1</sup>

<sup>2</sup>Tri-institutional Center for Translational Research in Neuroimaging and Data Science (TReNDS), Georgia State University, Georgia Institute of Technology, Emory University, Atlanta, GA, USA <sup>3</sup>Inria, CEA, Universit´e Paris-Saclay, Palaiseau, France <sup>4</sup>The University of Texas at Austin, Austin, TX, USA <sup>5</sup>National University of Singapore, Singapore <sup>6</sup>The University of Queensland, Brisbane, QLD, Australia

## Abstract

AI agents can execute scientific analyses, but an analytic output becomes a defensible claim only after alternatives are weighed and the claim is limited to what the evidence supports. Agents may reproduce failures including selective analysis, premature declarations of success and optimization of imperfect criteria. We present Brain Researcher, an agentic research harness operating in a neuroimaging researcher’s computational environment under rules for admissible analyses, required checks and claim scope. In benchmarks, Brain Researcher increased first-choice tool-selection accuracy across seven models by 70.2 percentage points (23.3% without it versus 93.6% with it) and verifiable grounding from 4.6% to 22.0%. In collaborator-led and self-evolving studies, multiverse analyses exposed analytic-choice sensitivity, and scientific review classified claims as accepted, qualified, revised, blocked, rejected or deferred. By linking decisions to evidence and provenance, Brain Researcher embeds methodological judgment within the workflow, not after it.

Keywords: neuroimaging; scientific agents; reproducibility; research infrastructure; analytic flexibility

Science often advances by absorbing its former frontiers into infrastructure: what once marked the edge of scientific practice, such as sequencing a genome or preprocessing a brain scan, becomes a routine step in a larger workflow. AI agents may represent the next phase of this progression. They increasingly interpret goals, call external tools, observe intermediate results, and choose subsequent actions [5, 27, 30, 41, 52]. Yet scientific research is not only a sequence of procedures, and executing an analysis is not the same as establishing a claim. This distinction is especially consequential in neuroimaging, where large, heterogeneous datasets enter long analysis pipelines with multiple defensible choices. Decisions about preprocessing, parcellation, confound adjustment, and mode specification can materially reshape results [9, 20, 36, 45]; when seventy teams analyzed the same neuroimaging dataset, no two used the same workflow and their conclusions difered substantially [6].

Neuroimaging has also developed one of the most mature open-science ecosystems for computational automation. BIDS and OpenNeuro standardize data organization and sharing [39, 46]; fMRIPrep automates functional MRI preprocessing [21]; and Nipype integrates software packages into reproducible workflows [26]. BIDS Statistical Models and FitLins extend this standardization to machine-readable statistical models and their execution [12, 40], while guided multiverse analysis makes alternative workflows more navigable [13]. Together, these tools make procedures, and increasingly their alternatives, reusable and comparable. They do not, however, bind a selected route to the evidence and methodological conditions required for the claim it is used to support. The challenge is therefore not simply to automate neuroimaging analysis, but to ensure that automation does not obscure the methodological conditions under which an analysis may support a claim.

Tool-using agents create both an opportunity and a risk. They can connect fragmented stages of an analysis and adapt after observing intermediate outcomes, but successful command execution does not guarantee a scientifically valid analysis. An agent may select an inappropriate tool, overlook a data or model incompatibility, stop prematurely, or optimize a criterion that is misaligned with the intended claim. These failures parallel questionable research practices in human-executed science [32, 48], and are compounded by premature declarations of task completion [29, 33] and reward hacking [22, 50]. Scientific agents therefore need more than access to tools: they need reliable routing, evidence linked to its methodological conditions, explicit exposure of alternative specifications, and a visible boundary between formalizable checks and judgments that remain with the researcher.

Here we present Brain Researcher, a researcher-governed, domain-specific agentic harness that operates inside a neuroimaging researcher’s existing computational environment. Brain Researcher is designed to preserve rather than replace scientific judgment: it operationalizes the parts that researchers can state in advance and records the rest for inspection. For prospectively governed analyses, the researcher specifies the question, admissible analyses, required checks, and scope of any resulting claim. A tool registry and the Brain Researcher Knowledge Graph connect analysis routes to evidence and method conditions; a Model Context Protocol server mediates model actions; and execution and review layers return an audit bundle linking the committed plan, tool calls, artifacts, evidence, and claim verdicts (Fig. 1; Supplementary Methods S1). Multiverse analyses expose sensitivity to defensible specifications [51], while commitment and claim cards preserve what was decided before and after results were observed. In self-evolving research, intermediate evidence can redirect the trajectory only within a researcher-defined action space specifying the admissible analyses, datasets, and evaluation budget; each successor analysis is frozen before execution. Judgments that resist formalization remain with the researcher.

We evaluated Brain Researcher in three settings. First, a paired seven-model tool-calling benchmark and a separate evidence-citation benchmark tested whether the harness improves upstream tool selection and evidence citation. Second, three collaborator-led studies tested whether multiverse analyses and explicit constraints make claim sensitivity and status visible. Third, two self-evolving research episodes tested whether evidence from one stage could be carried into a frozen successor analysis. Together, these evaluations ask whether AI assistance becomes more capable and more defensible when methodological commitments, evidence, and claim scope are made explicit and auditable.

## Results

## Brain Researcher converts research questions into auditable claim records

At the center of Brain Researcher is a complete, auditable record of a single run: a persistent episode linking the question to its assembled evidence, the analyses the researcher deemed admissible, the committed plan, the execution, the review, and a condition-tagged conclusion (Supplementary Methods S2). Two dated records can anchor it. A commitment card, written before any analysis

Brain Researcher: Schema-Constrained LLM-Agent Planning for Verifiable Neuroimaging Workflows  
![](images/bff73d95d685330b2d90f226f0dfb83a20b555ae5656cab03a3da9bd69bc76eb.jpg)

Figure 1: Brain Researcher: workspace-centric infrastructure for auditable neuroimaging research. Brain Researcher runs inside the researcher’s existing computational environment and exposes neuroimaging analyses as structured, auditable operations: every choice, check, and input is recorded as it happens, so that a frozen record of the completed analysis can be read and audited by someone other than the person who ran it, without re-executing it. (a) The tool ecosystem: established neuroimaging software for preprocessing, modeling, meta-analysis, machine learning, quality control, and reporting, each represented by a machine-readable specification and executed in a version-pinned container. (b) Each specification declares its inputs, outputs, parameters, version, evidence anchors, and validation rules; the rule checker tests every proposed call against these clauses before it runs, and records which clauses passed or failed. (c) The episode workflow: the researcher frames a question and approves the plan at the commitment gate, valid actions are dispatched to version-pinned executors, and the resulting audit bundle, containing the committed plan, tool versions, evidence consulted, artifacts, logs, provenance, and the checks each claim passed, feeds the review layer, which writes condition-tagged claims back to memory. This audit bundle is what makes an analysis auditable: a completed run is a fully inspectable research object rather than a one-of result, exported as a compact claim card a reviewer can reopen field by field. Methodological judgment remains the researcher’s; the system makes it visible at each stage.

BR-KG: semantic integration for accurate queries & reasoning  
![](images/b4e287cc9c4aebb95b47e9502c59507a656e6632f5f08d8755d638e8873c3145.jpg)  
Figure 2: BR-KG: provenance-linked semantic integration for grounded, auditable neuroimaging reasoning. BR-KG integrates existing ontologies, repositories, data resources, and literature into a single graph (745,949 nodes, 2,461,469 edges; 2026-07-07 release snapshot) aligned to the OpenNeuro Vocabulary (ONVOC) [44], which normalizes heterogeneous terms to shared identifiers so a query resolves consistently across sources. The central graph links neuroimaging concepts (tasks, contrasts, cognitive constructs), neural representations (brain regions, statistical maps), and research resources (datasets, tools) through typed relationships, with literature evidence attached. Crucially, source-backed facts carry explicit provenance (their source, and where available a verbatim supporting quote and grounding label), so a retrieved claim can be traced back to the study and passage that support it rather than taken on trust; coverage is partial and tracked, and this is what makes retrieval here auditable. Downstream panels show the payof: grounded query answering and multi-hop reasoning over concept-task-map paths, each hop inspectable down to its underlying nodes, edges, and cited evidence, which lets the review layer attach a recommendation’s method-condition checks (cohort, paradigm, preprocessing, statistical model) before it is accepted.

runs, fixes the question, the allowed alternatives, and the success and failure criteria, and is sealed with a content hash so that any later change to the plan is detectable. A claim card, written afterward by the review layer, records the resulting claim, its assigned state, scope, and the checks it passed and failed; the supplement includes a specific worked example, built on public Neurosynth data, that a reader can open and inspect field by field (Supplementary Methods S8.5.1; Appendix G). A reviewer can then reopen and audit the record without re-executing the analysis. Across the collaborator-led and self-evolving evaluations reported below, every evaluated claim is assigned one of six states (accepted, qualified, revised, blocked, rejected, or deferred), each defined by an explicit adjudication rule. For example, when a researcher asks whether two groups difer on a functional-connectivity measure, the commitment card records the cohort, the subject groups, and the estimator (either entered by the researcher or resolved from the dataset) and fixes the checks that must pass before anything runs (aligned subject groups, a full-rank design matrix, no feature leakage across folds). If the diference then holds in only some admissible specifications, the claim is recorded as qualified, with its conditions attached. These methodological decisions are the researcher’s; Brain Researcher records each one and enforces the checks it implies.

## Brain Researcher improves tool calling and evidence citation

We first isolated the infrastructure’s efect on two decisions that precede scientific review: choosing the correct analysis tool and citing checkable evidence. The tool-calling benchmark pairs each request with a hidden scoring target and reports Correct route/tool@k (all-or-nothing match of the analysis family and required capabilities), Capability@k (graded capability coverage), and Handof score@k (whether the proposed route is specified completely enough to execute, with the required inputs present, so it can be handed to a downstream executor and run without gaps) (Supplementary Methods S11.1.1); three condition-blind LLM judges credit any response that reaches the required capabilities, whether through a Brain Researcher call or an equivalent executable route, so the measured gain reflects reaching those capabilities rather than credit for naming Brain Researcher’s specific tools. Both conditions used the same seven frontier models [1, 3, 15, 25, 42, 43, 54] and general-purpose tools, the without-BR condition lacking only Brain Researcher’s registry (Supplementary Methods S5), knowledge graph, and constraint layer. Across 60 tool-calling tasks and seven models, scores without versus with Brain Researcher were 23.3% versus 93.6% for first-action correct route/tool selection (with-BR 95% CI 88.8–97.1, task-clustered), 49.8% versus 94.5% for mean Capability@1, and 47.4% versus 76.1% for handof suficiency. The reference route for each task was fixed before either condition ran and curated with a co-author who does not develop the Brain Researcher system; equivalent non-BR routes were set by two model reviewers and one human (Supplementary Methods S11.1.2). All seven models improved on all three tool-calling metrics (7/7 positive paired diferences; exact two-sided Wilcoxon signed-rank $p = 0 . 0 1 6$ for each), with mean gains of 70.2 percentage points for Correct route/tool@1, 44.7 points for Capability@1, and 28.7 points for Handof score@1. In a routing ablation across 60 tasks and seven models, Brain Researcher without direct KG calls selected an acceptable exact top-1 route in 362 of 420 episodes (86.2%; model range, 81.7–90.0%; details in Supplementary Methods S11.1.5). A separate 50-question benchmark counted a claim as grounded only when its cited evidence could be located and judged supportive; under a three-judge majority vote, the descriptive question-level verified-groundedness rate rose from 4.6% to 22.0% (95% CI 16.8–27.2, question-clustered), a 4.8- fold increase, though most evidence rows still failed, so grounding improved substantially without being solved. Among the 444 non-verified with-BR rows present in all three judge outputs, 65% received an exact-label majority of real but of-topic and 28% of partial support; the remaining 7% lacked an exact-label majority or could not be judged, and none had a fabricated or malformed majority (Supplementary Methods S11.1.1). Inter-judge reliability and its dependence on judge strictness are reported in Supplementary Methods S11.1.2. Secondary single-judge safeguards confirmed that the gain was not accompanied by more unrelated citations or lower answer correctness (Supplementary Methods S11.1.1; Appendix J; Fig. 3). Because the without-BR condition removes Brain Researcher’s registry, knowledge graph, and constraint layer together, this contrast measures the harness as a whole rather than isolating any single component; and because the reference routes were curated with a co-author, target construction may share vocabulary with the registry.

![](images/fa65b103d33ef5cc8ea051a12de9bf48492f3e89b639c67a602f3a0419812ea8.jpg)  
Figure 3: Summary of Brain Researcher efects across quantitative benchmark tasks. Without-BR (gray) and with-BR (blue) benchmark performance. Capability@k is mean coverage of required task capabilities after the first k non-neutral actions. The left column reports Capability@1 and @3 across the seven model variants (Claude Opus 4.8, Codex GPT-5.5, Gemini 3.1 Pro, GLM-5.1, DeepSeek-V4-Pro, Kimi K2.5, Qwen3.6-Plus). Upper-right panels break Capability@1 down by task domain. Lower panels report Handof score@1 and @3 (whether the first route carries enough information for another agent to continue) and a Gemini 2.5 Flash single-judge safeguard: precision among claims marked grounded (fraction whose cited evidence was both locatable and judged supportive). Correct route/tool@1, the first-action selection accuracy reported in the text (23.3% to 93.6%), is detailed in Supplementary Methods S11.1.1. Metrics are interpreted within panel, as denominators and scoring rules difer across benchmarks.

## Brain Researcher runs multiverse analyses to expose claim sensitivity

We next evaluated Brain Researcher on three active neuroimaging research questions from collaborating scientists (schizophrenia NeuroMark connectivity, cocaine-use-disorder connectivity, and cross-cultural social-cognition meta-analysis), chosen for heterogeneity in evidence structure without regard to the specific outcomes (Fig. 4). Every reported analysis case was run by a coding agent on the local system, which called Brain Researcher for grounding, logging, and review (Supplementary Methods S11.2). The NeuroMark case starts from a single, well-established pipeline that its developers use as their standard, giving the audit one clearly defined baseline to build the multiverse around; the other two cases have no such established single pipeline, and test whether the workflow extends to that more dificult setting. The collaborators’ hypotheses were pre-specified in their own protocols rather than sealed as commitment cards, so the NeuroMark record is a post-hoc audit of the completed multiverse.

A collaborator studying schizophrenia functional network connectivity using the NeuroMark framework [17, 31] brought three pre-specified hypotheses for robustness audit: latent connectivity factors outperform individual edges for patient-versus-control classification (NM-H1); betweendomain connections show larger group diferences than within-domain ones (NM-H2); and latent factors concentrate loading mass on between-domain edges (NM-H3). We evaluated these hypotheses in the FBIRN cohort [34] (N = 363; 181 controls, 182 patients), parcellated through NeuroMark 2.2 template-based independent component analysis into 5,460 edges per subject. In the collaborator’s workspace, Brain Researcher expanded the analysis into a 480-specification multiverse spanning connectivity, confound, dimensionality-reduction, classifier, and domain-granularity choices, and recorded and reviewed the resulting runs.

None of the three hypotheses was supported uniformly across specifications; all were recorded as qualified, but for diferent patterns of conditional support. Under the corrected sign-aware criterion $( p < 0 . 0 5$ and $\Delta \mathrm { m e a n } | d | > 0 )$ , 12 of 24 unique connectivity–confound–domain contrasts favored NM-H2. This pooled fraction obscured a complete estimator split: 100% of contrasts were favorable under Pearson and Spearman and 0% under partial correlation and mutual information. NM-H2 is therefore an estimator-regime–dependent finding rather than a generally robust efect. Because partial correlation and mutual information alter the dependence measure in non-equivalent ways, distinguishing shared covariance from estimator scale, power, or nonlinearity requires targeted follow-up. NM-H1 and NM-H3 were also weak: edges outperformed latent factors in aggregate (median $\Delta \mathrm { A U C } = - 0 . 0 3 2 ;$ only 18.8% of specifications favored latent features), and only 26.0% favored between-domain loading mass. These claims were qualified rather than rejected because support persisted within identifiable analytic subfamilies (Supplementary Methods S11.2.1; Fig. 4A–C); a claim with no supporting subfamily is rejected instead (Supplementary Methods S8.4).

NM-H2 also supplied the audit’s governance lesson: automated review missed an error that a human caught. After a server-side fault triggered fallback to a general-purpose coding agent, the agent scored any specification with permutation $p < 0 . 0 5$ as favorable regardless of sign, inflating apparent support for a directional hypothesis to near-universal levels. The review layer did not flag the error; a human reviewer detected it by inspecting the code, outputs, and specification curve, leading to the corrected rescoring above. Two checks were then added to the Brain Researcher skillset: a directionality test requiring the statistic and acceptance rule to match the hypothesized sign, and a warning whenever execution falls back to a general-purpose agent. The review missed this error. The record nevertheless provided value by binding each claim to its hypothesis, statistic, and conditions, thereby turning a one-of correction into an enforced check.

In the other two episodes, prespecified checks in the scientific review layer determined whether a result could receive confirmatory status. The SUDMEX CONN (OpenNeuro ds003346; N = 138)

![](images/ba2b197f78c006a0ad9847789fae992eb6c1bbf34d87f528f755c1a3daa02d05.jpg)

![](images/9ec72073b96f3cf6b470c7df5abe2543c9579724477c21d7c3773fad2bfa137a.jpg)  
Figure 4: Multiverse sensitivity and claim-review outcomes across three collaborator episodes. (A–C) Schizophrenia NeuroMark audit: (A) group-mean functional connectivity for controls (HC, N = 181), patients (SZ, N = 182), and their diference across four estimators; (B) NM-H2 (between- versus within-domain) specification curve over the 480-specification multiverse; after sign-aware rescoring, its estimand comprises 24 unique connectivity–confound–domain contrasts, with favorable support at 100% for Pearson and Spearman and 0% for partial correlation and mutual information. This complete estimator partition, rather than the pooled 12-of-24 fraction, is the informative result: NM-H2 is measure-dependent, and the mechanism underlying the partition remains unresolved. (C) Marginal influence of each analytic choice on NM-H2. (D, E) Cocaine-use-disorder episode: (D) multiverse stability of systemic-segregation associations across 36 specifications with SDMA-GLS consensus; (E) single-specification versus multiverse SDMA-GLS maps for five network–outcome pairs. (F) Cross-cultural social cognition: culture-stratified ALE maps contrasting Euro-American trust networks with East Asian social-cognition networks.

[2, 23] example assessed associations between brain connectivity and behavior; a 36-specification multiverse rejected all five pre-specified connectivity–behavior associations under same-dataset meta-analysis (SDMA-GLS) [35] (all $Z ~ < ~ 1 . 2 4$ , false discovery rate [FDR] $q \ > \ 0 . 5 8 )$ , and an exploratory screen over 70 combinations surfaced no FDR-surviving efect, so the system blocked it from confirmatory promotion and converted the null into a replication plan (Fig. 4D,E). In another test case that applied coordinate-based neuroimaging meta-analysis to a small cross-cultural neuroscience literature, subgroup activation-likelihood estimation (ALE) on 21 studies [16, 18, 19, 47] produced a medial prefrontal cortex (mPFC)-topology interpretation, but the system blocked it as exploratory: the subgroups held only $k = 6 { - } 8$ entries (below the recommended $k \geq 1 7 )$ , paradigm composition was imbalanced, and centroid shifts alone cannot establish non-overlapping distributions. The case ended in a paradigm-matched follow-up with no settled claim (Fig. 4F). Full statistics are in Supplementary Methods S11.2.1 and Appendix J.

Across the three episodes, the multiverse exposed which findings were sensitive to analytic choices, while scientific review determined what each result could support. Prespecified review checks withheld confirmatory status from the SUDMEX exploratory screen and the underpowered cross-cultural ALE; in the post-hoc NeuroMark audit, a human reviewer identified the sign-blind scoring error.

## Brain Researcher converts adaptive searches into frozen successor analyses in two self-evolving episodes

We next asked whether Brain Researcher could transform open-ended exploration into frozen, auditable successor analyses. We examined two extended research episodes that difered in what was searched. Using the Human connectome Project (HCP) data, Brain Researcher searched over candidate analysis workflows for a fixed question about connectivity-based prediction of behavioural variation. Using the TRIBE foundation model, it searched over candidate scientific questions about a model’s internal representations and then converted one question into a frozen test on newly sampled stimuli (Supplementary Methods S11.3).

In the HCP episode, we began with a published study with openly available code and shared analysis materials [37]. Brain Researcher allocated 116 candidate prediction-pipeline evaluations for Cognition, of which 104 returned scored results in the parent runs. Following a selector audit, the researcher designated a frozen selected workflow. In 10 repeated same-cohort nested-crossvalidation splits, the frozen selected workflow achieved a higher pooled out-of-fold correlation than a matched local reconstruction of the published procedure (median $\Delta r = . 0 9 8 ;$ conditional one-sided $p = . 0 0 6 )$ . When the frozen selected workflow was refit to four additional behavioural outcomes, it again produced higher correlations in 37 of 40 comparisons, giving the same direction in 47 of 50 comparisons across all five outcomes. Median out-of-sample $R ^ { 2 }$ was positive only for two variables (Cognition and Tobacco Use), and multiplicity-aware transfer inference remained inconclusive (Fig. 5; Supplementary Methods S11.3.1).

TRIBE $\mathrm { v 2 [ 1 4 ] }$ is a tri-modal foundation model that predicts human fMRI responses from video, audio, and language inputs. We asked how natural-sound category geometry changes across its internal audio layers. Brain Researcher screened category contrasts without choosing one in advance and ranked them by changes in source-held-out discrimination (Fig. 6A). Tools–voice showed the largest change but varied across sound collections. Brain Researcher instead proposed speech–tools for follow-up because early layers strongly separated the categories, whereas later layers brought them closer while largely preserving the same representational direction. This contrast could also be tested prospectively using new recordings sampled from multiple collections and matched on seven prespecified acoustic measurements. The researcher approved this direction and froze the hypothesis and analysis before the new stimuli were evaluated.

![](images/2b9ab999c8ce565fc783900f7bb0de2ff6e305e66e39d5ec56344d65dae86381.jpg)

![](images/4e7bb5d6bcdbfec1321997f8e607bbc2b7bc94794ff03675a285ca9dd5654017.jpg)  
Figure 5: Brain Researcher searches 116 HCP prediction pipelines and identifies a workflow that consistently exceeds a matched reference. A. Brain Researcher first evaluated 20 candidate pipelines for Cognition prediction, reaching a best discovery score of $r = . 3 7 3$ Brain researcher then launched a 96-candidate expansion; 84 candidates returned scores and 12 ended in transport failure. Within the expanded episode, Brain Researcher adapted its proposals to the accumulating results: 27 candidates exceeded the initial search maximum, and the highest discovery score was $r = . 4 8 7$ , obtained with whole-band coherence and ridge regression. Following a selector audit, the researcher froze a related coherence-based workflow for matched evaluation. Across 10 repeated family-grouped $5 \times 3$ nested-cross-validation runs, this workflow achieved median $r = . 3 3 2$ , compared with .235 for the matched reference (median $\Delta r = . 0 9 8 \colon$ conditional one-sided $p = . 0 0 6 )$ , and was higher in all 10 runs. B. The same frozen selected workflow was then refit for each of four additional behavioural outcomes without target-specific retuning. It produced a higher median correlation for every outcome and exceeded the matched reference in 37 of 40 repeat-level comparisons, giving 47 of 50 directional wins across all five outcomes.

Brain Researcher then evaluated three successive, non-overlapping 48-item panels. The normalized speech–tools separation became smaller in later layers in 11 of 12 collection-by-panel comparisons, and all three panels met the prespecified directional criterion. In most collections, later TRIBE layers preserved the representational direction separating speech from tools while bringing the categories closer together. The result was a direction-preserving contraction of speech–tools geometry. After the pattern recurred across all three panels, Brain Researcher extended the test to four previously unused sound collections. Three showed the same geometry, although the corrected collection-level test remained inconclusive (Holm-adjusted p = .396; Fig. 6B,C; Supplementary Methods S11.3.2).

Both episodes converted adaptive searches into frozen follow-up analyses. In separately initiated sessions without Brain Researcher, the same coding agent completed substantial analyses, but neither session generated and froze a follow-up study. Because these sessions were not matched controls, this contrast is descriptive and does not establish that Brain Researcher caused the transition from an initial result to a frozen follow-up (Supplementary Methods S11.3.3).

## Discussion

Our central contribution is to treat the unit of AI-assisted research as a governed research trajectory rather than a model output. The paired benchmarks tested whether models could reach relevant tools and evidence. The collaborator studies showed how multiverse analysis and explicit constraints change what can be claimed from a completed analysis. The HCP and TRIBE episodes went one step further: a result from one stage became an input to the next. Taken together, these evaluations support a view of scientific agents not as systems that generate a final answer, but as infrastructure that augments human judgments by keeping questions, decisions, evidence, and claim states connected as a project evolves.

This is the sense in which the research episodes were self-evolving. In HCP, Brain Researcher searched over candidate workflows for a fixed question; following a selector audit, the researcher designated the frozen selected workflow and carried it into a matched comparison and four additional behavioural outcomes. In TRIBE, Brain Researcher did not simply promote the contrast with the largest change. It set aside an unstable lead, proposed a more coherent speech–tools question and, after the researcher froze it, carried that question into newly sampled panels. In both cases, intermediate evidence changed the next analysis without rewriting the analysis already under test. The research trajectory evolved, but the evidentiary standard did not.

This trajectory-level view also changes the role of verification. Formal criteria can operate prospectively when they are specified in advance; multiverse analysis can show how a result depends on the enumerated defensible choices [7, 13, 36, 49, 51]; and judgments that resist formalization remain with the researcher (Supplementary Methods S6.1). Relative to systems evaluated primarily for task execution or output correctness [4, 5, 10, 11, 24, 27, 30, 52, 53], Brain Researcher makes the relationship among the estimator, comparison, evidence base, and claim part of the persistent research record. The NeuroMark example illustrates the limit of this formal layer: automated review missed sign-blind scoring, and a human reviewer found the error. Brain Researcher therefore makes formalizable conditions visible and auditable; it does not make expert inspection unnecessary.

Once decisions and claim states persist, qualified, negative, and failed results need not be terminal outputs. They can narrow the next question, retire an unproductive branch, or define a frozen successor analysis. This is the broader infrastructure implication of self-evolving research: progress can accumulate across successive episodes instead of restarting from an unstructured prompt each time. The mechanism is not unconstrained model autonomy, but the combination of an adaptive trajectory with researcher-defined scope, explicit evidence standards, and durable records of what was tried and why it changed. Researchers retain authority to define the action space, select and freeze successor questions, interpret the evidence, and decide whether the trajectory should continue.

A Open discovery and exploratory selection  
![](images/468836c0abb3c38a5c94eb1def4c13928860c7f136023f7ed573a6c7e7c50472.jpg)

![](images/9bcaea14fb3693efc3ea340f160df694b4972e743243d51a97c18c39f4bd2f9e.jpg)

![](images/ca5652b3d8d556929bfa2264856de4a4f059f660e80bcfb7bc4e4e5ae142549c.jpg)

![](images/e9e444d834d71736a313858a2a2b25e1a0d0e3f3a605f1787dbcd94c8323d048.jpg)

C Four-new-collection extension  
![](images/bba0fbdd42ce25f4ee74cd6b093591fe388691dcb5ea976f366ae45ad72edac5.jpg)  
Figure 6: Brain Researcher turns an open question about TRIBE into successive tests with new sounds and collections. A. Brain Researcher began by asking how TRIBE changes natural-sound representations from early to late layers. It screened category contrasts by their change in held-out distinguishability (AUC), without choosing a target in advance. Tools–voice changed most, but the pattern varied across collections. Rather than simply following the topranked result, Brain Researcher identified speech–tools as a clearer lead: the categories moved closer in later layers while usually keeping the same representational direction. The contrast could also be retested with new, acoustically matched sounds from several collections. The researcher approved this direction and froze the prediction and analysis. B. Brain Researcher then evaluated three non-overlapping 48-item panels. All three showed a smaller speech–tools separation on average in later layers. In 11 of 12 collection-by-panel comparisons, the categories became less separated while retaining the prespecified direction. C. After the pattern recurred across all three panels, Brain Researcher extended the test to four previously unused sound collections. Three of four showed the same geometry, and the late-layer separation was again smaller on average $( \Delta S = - 0 . 1 9 8 )$ . In all geometry plots, horizontal position shows the late-minus-early change in normalized separation (∆S), and vertical position shows late directional alignment (C); the upper-left quadrant therefore marks smaller separation with retained direction. A uses fold-specific references, whereas B and C use the frozen speech–tools reference.

This work has several important limitations (Supplementary Methods S12). Brain Researcher produces auditable evidence but leaves interpretation and writing to the researcher. Its foundationmodel and retrieval priors reflect the literature, datasets, and instrumented tools, which may favor well-represented, operationalized questions over negative results, low-resource populations, and unusual paradigms [28]. Several episodes rely primarily on same-dataset multiverse or internal validation; these improve auditability but do not replace independent replication or constitute external confirmation [8, 38]. Several of these datasets are public (HCP, OpenNeuro), so a frontier model may have encountered the associated published findings during training; we cannot rule out memorization, which is a further reason not to treat these results as novel detections. Evidence grounding was scored by condition-blind LLM judges (three frontier models that are also among the seven evaluated, a potential source of self-preference). A reproducible human audit of 20% of scored results (272 of roughly 1,360 items) agreed with 96% of verdicts (Cohen’s κ = 0.94); discrepancies were one-step severity diferences, never reversals between supported and unsupported, and the judges erred strictly (Supplementary Methods S11.1.3). Runtime and researcher efort were not measured. Review-layer error was estimated against a 60-case calibration library (16 invalid, 5 valid controls, 39 warn), which produced no false-accepts (0 of 16; rule-of-three 95% upper bound 19%) and no false-blocks (0 of 5, a loose bound); this library was assembled after the sign-direction check identified through the NeuroMark case and is not an independent, field-scale estimate (Appendix G; Supplementary Methods S11.3.4). The calibration therefore measures internal consistency on canonical scenarios, not how often flawed claims escape review in deployed research workflows. Independent replication and field-scale adjudication remain separate tests of scientific validity, which will require labeled real analyses. Finally, claim records are exportable files, but their value as shared, contestable infrastructure across laboratories remains a future objective. AI assistance should make the conditions under which results become reproducible knowledge easier to see, test, and share.

## Online Methods

Detailed methods, including the runtime stack, the BR-KG substrate and sources, the operation registry, execution backends, benchmark scoring contracts, multiverse and validation-gated search protocols, and all per-case statistics, are provided in the Supplementary Information (Supplementary Methods S1–S12; Appendices A–K, with the per-case episode reports and the item-level benchmark audit sheet released as extended-data Appendices L and M; Supplementary Figures).

## Supplementary information

Supplementary Information accompanies this manuscript.

## Funding

JAR is supported by Stanford University Knight-Hennessy Scholars Program, National Academies of Sciences, Engineering, and Medicine’s Ford Foundation Predoctoral Fellowship, Institute of International Education Quad Fellowship, the National Science Foundation’s Graduate Research

Fellowship Program, the Center for Mind, Brain, Computation and Technology, and the Wu Tsai Neurosciences Institute. V.D.C. received support from NSF 2112455 and NIH R01MH123610. A.d.l.V., R.P. and J.K. were supported by the National Institute of Mental Health under award R01MH096906. Z.C. and R.P. received cloud-computing credits through the 2025 HAI-Google Cloud Credits Grant Program to support Brain Researcher API development and computation.

## Competing interests

S.K. reports part-time employment with Meta, which began recently and after most of the work reported here. The other authors declare no competing interests.

## Ethics, consent and materials availability

Not applicable: this work analyzed only previously collected, publicly available or collaboratorprovided de-identified neuroimaging data under their original ethics approvals and consents, and generated no new human- or animal-subjects data or materials.

## Data availability

BR-KG is archived at Zenodo (https://doi.org/10.5281/zenodo.21966011) and linked from the public project site (https://brain-researcher.com/). The release includes graph snapshots, node and edge schemas, provenance fields, registry links, benchmark manifests, scoring tables, aggregate outputs, figure source data, run-bundle schemas, a worked auditable claim-record example (an exported claim card with its evidence verdicts, on public Neurosynth data), and deployment notes. Users can access the public MCP interface and released Brain Researcher skills from the project site, which describes how users can suggest additions or corrections to BR-KG. Source neuroimaging datasets remain under their original terms: public resources are cited and linked in Supplementary Methods S4 and Appendix C, and controlled-access, collaborator-provided, or license-restricted human-subject data are not redistributed. Artifact and provenance records are described in Supplementary Methods S7.4 and Appendix F; benchmark records in Supplementary Methods S11.1 and Appendix J. To keep the Supplementary Information self-contained, the full audit ledgers it condenses are released in the same archival repository as an extended-data package: the complete BR-KG, evidence-bundle, dataset, tool-registry, constraint, execution–provenance, and memory data cards (Appendices A–F and H), the full per-rule review registry (Appendix G9.1–G9.4 and G9.6), the automatically generated per-case episode reports (Appendix L), the current HCP and TRIBE research-line reports and their supporting run bundles, and the item-level benchmark human-audit sheet (Appendix M); a crosswalk maps each condensed Supplementary section to its archived file.

## Code availability

The Brain Researcher system (Python package, CLI, agent runtime, MCP server with versioned tool contracts, orchestrator, web UI, and deployment recipes) is available under the MIT license at https://github.com/brain-researcher/brain-researcher-public, with the companion agent layer (skills, agent templates, MCP adapters, and AutoResearch evaluation rubrics) at https: //github.com/brain-researcher/brain-researcher-agent-kit; both are linked from the project site (https://brain-researcher.com/), and the v0.3.0 release is archived at Zenodo (https://doi.org/10.5281/zenodo.21966011). Analysis code and per-specification outputs for the NeuroMark collaborator case are available at https://github.com/XinhuiLi/BR-NeuroMark.

## Author contributions

Z.C. and R.P. initiated and conceived the project. Z.C. designed and implemented the Brain Researcher system, ran the experiments and analyses, generated the main results, and drafted the manuscript. R.P. supervised the project and contributed to conceptual framing, study design, hands-on system testing, evaluation feedback, interpretation, and manuscript revision. J.H.Z. provided early supervision and initial computational resources for the project. N.L. gathered background information, including dataset lists and literature-review materials, helped design the benchmark questions, evaluated system outputs, and tested performance for the quantitative benchmark section. X.L. and V.D.C. designed and ran the NeuroMark schizophrenia functional-networkconnectivity case. J.R. and R.P. designed and ran the cocaine-use-disorder connectivity case. H.W. designed and ran the cross-cultural social cognition case. C.K., J.K., and A.d.l.V. provided feedback on knowledge-graph design and contributed data and design requirements for the knowledge-graph and source-integration components. S.B. provided feedback on agent design, MCP infrastructure, backend integration, and execution design. J.M. provided feedback on the scientific-review layer. S.D. contributed suggestions and ideas on the agent harness and validation-gated research design, including the bounded-validation framing; S.K. provided feedback on the agent harness. C.J. contributed to system testing. J.W.B. provided feedback on the manuscript. All authors reviewed and approved the manuscript.

## References

[1] Alibaba Cloud. Qwen3.6-Plus: Towards real world agents, 2026. URL https://www.alibab acloud.com/blog/603005. Accessed 21 May 2026.

[2] Diego Angeles-Valdez, Jalil Rasgado-Toledo, Victor Issa-Garcia, Thania Balducci, Viviana Villica˜na, Alely Valencia, Jorge Julio Gonzalez-Olvera, Ernesto Reyes-Zamorano, Eduardo A. Garza-Villarreal, et al. The Mexican magnetic resonance imaging dataset of patients with cocaine use disorder: SUDMEX CONN. Scientific Data, 9(1):133, 2022. doi: 10.1038/s41597 -022-01251-3.

[3] Anthropic. Introducing Claude Opus 4.8, 2026. URL https://www.anthropic.com/news/c laude-opus-4-8. Accessed 5 June 2026.

[4] Anthropic. Claude Science, an AI workbench for scientists, is now available. https://ww w.anthropic.com/news/claude-science-ai-workbench, June 2026. Anthropic news announcement, 30 June 2026.

[5] D. A. Boiko, R. MacKnight, B. Kline, and G. Gomes. Autonomous chemical research with large language models. Nature, 624:570–578, 2023. doi: 10.1038/s41586-023-06792-0. URL https://doi.org/10.1038/s41586-023-06792-0.

[6] R. Botvinik-Nezer, F. Holzmeister, C. F. Camerer, et al. Variability in the analysis of a single neuroimaging dataset by many teams. Nature, 582:84–88, 2020. doi: 10.1038/s41586-020-231 4-9. URL https://doi.org/10.1038/s41586-020-2314-9.

[7] M. Burkhardt and C. Giessing. The Comet Toolbox: Improving robustness in network neuroscience through multiverse analysis. Imaging Neuroscience, 4:IMAG.a.1122, 2026. doi: 10.1162/IMAG.a.1122. URL https://doi.org/10.1162/IMAG.a.1122.

[8] K. S. Button, J. P. A. Ioannidis, C. Mokrysz, B. A. Nosek, J. Flint, E. S. J. Robinson, and M. R. Munafo. Power failure: Why small sample size undermines the reliability of neuroscience. Nature Reviews Neuroscience, 14:365–376, 2013. doi: 10.1038/nrn3475. URL https://doi. org/10.1038/nrn3475.

[9] J. Carp. The secret lives of experiments: Methods reporting in the fMRI literature. NeuroImage, 63(1):289–300, 2012. doi: 10.1016/j.neuroimage.2012.07.004. URL https: //doi.org/10.1016/j.neuroimage.2012.07.004.

[10] Jun Shern Chan, Neil Chowdhury, Oliver Jafe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, et al. MLE-bench: Evaluating machine learning agents on machine learning engineering, 2024. URL https://doi.org/10.48550/arXiv.2410.07095. Publication Title: arXiv.

[11] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, et al. ScienceAgentBench: Toward rigorous assessment of language agents for data-driven scientific discovery, 2025. URL https://doi.org/10.48550/arXiv.2410.05080. ICLR 2025; Publication Title: arXiv.

[12] BIDS Community. BIDS Stats Models Specification, 2024. URL https://bids-standard. github.io/stats-models/.

[13] J. Daflon, P. F. Costa, F. Vasa, et al. A guided multiverse study of neuroimaging analyses. Nature Communications, 13:3758, 2022. doi: 10.1038/s41467-022-31347-8. URL https: //doi.org/10.1038/s41467-022-31347-8.

[14] Stephane d’Ascoli, Jeremy Rapin, Yohann Benchetrit, Teon Brooks, Katelyn Begany, Josephine Raugel, Hubert Banville, and Jean-Remi King. A foundation model of vision, audition, and language for in-silico neuroscience, 2026. URL https://doi.org/10.48550/arXiv .2605.04326. Publication Title: arXiv.

[15] DeepSeek. DeepSeek V4 preview release, 2026. URL https://api-docs.deepseek.com/new s/news260424. Accessed 21 May 2026.

[16] J. Dockes, R. A. Poldrack, R. Primet, H. Gozukan, T. Yarkoni, F. Suchanek, B. Thirion, and G. Varoquaux. NeuroQuery, comprehensive meta-analysis of human brain mapping. eLife, 9: e53385, 2020. doi: 10.7554/eLife.53385. URL https://doi.org/10.7554/eLife.53385.

[17] Yuhui Du, Zening Fu, Jing Sui, Shuang Gao, Ying Xing, Dongdong Lin, Mustafa Salman, Anees Abrol, Md Abdur Rahaman, Jiayu Chen, L. Elliot Hong, Peter Kochunov, Elizabeth A. Osuch, and Vince D. Calhoun. NeuroMark: An automated and adaptive ICA-based pipeline to identify reproducible fMRI markers of brain disorders. NeuroImage: Clinical, 28:102375, 2020. doi: 10.1016/j.nicl.2020.102375.

[18] S. B. Eickhof, A. R. Laird, C. Grefkes, L. E. Wang, K. Zilles, and P. T. Fox. Coordinatebased activation likelihood estimation meta-analysis of neuroimaging data: A random-efects approach based on empirical estimates of spatial uncertainty. Human Brain Mapping, 30(9): 2907–2926, 2009. doi: 10.1002/hbm.20718. URL https://doi.org/10.1002/hbm.20718.

[19] S. B. Eickhof, T. E. Nichols, A. R. Laird, et al. Behavior, sensitivity, and power of activation likelihood estimation characterized by massive empirical simulation. NeuroImage, 137:70–85, 2016. doi: 10.1016/j.neuroimage.2016.04.072. URL https://doi.org/10.1016/j.neuroima ge.2016.04.072.

[20] A. Eklund, T. E. Nichols, and H. Knutsson. Cluster failure: Why fMRI inferences for spatial extent have inflated false-positive rates. Proceedings of the National Academy of Sciences, 113 (28):7900–7905, 2016. doi: 10.1073/pnas.1602413113. URL https://doi.org/10.1073/pnas .1602413113.

[21] O. Esteban, C. J. Markiewicz, R. W. Blair, et al. fMRIPrep: A robust preprocessing pipeline for functional MRI. Nature Methods, 16:111–116, 2019. doi: 10.1038/s41592-018-0235-4. URL https://doi.org/10.1038/s41592-018-0235-4.

[22] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 10835–10866. PMLR, 2023. URL https://proceedings.mlr.press/v202/gao23h.html.

[23] Eduardo A. Garza-Villarreal, Jorge Julio Gonzalez Olvera, Thania Balducci, Diego Angeles Valdez, Alely Valencia, and Jalil Rasgado. SUDMEX CONN: The Mexican dataset of cocaine use disorder patients. OpenNeuro dataset, 2026. URL https://doi.org/10.18112 /openneuro.ds003346.v1.1.3.

[24] Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J. Szostkiewicz, Dmytro Shved, Gavin J. Gyimesi, Jon M. Laurent, Samantha M. Wright, Muhammed T. Razzak, Andrew D. White, Silvia C. Finnemann, Michaela M. Hinks, and Samuel G. Rodriques. A multi-agent system for automating scientific discovery. Nature, 2026. doi: 10.1038/s41586-026-10652-y. URL https://www.nature.com/articles/s41586-026-1 0652-y.

[25] Google DeepMind. Gemini 3.1 Pro: Model card, 2026. URL https://deepmind.google/mo dels/model-cards/gemini-3-1-pro/. Accessed 21 May 2026.

[26] K. Gorgolewski, C. D. Burns, C. Madison, D. Clark, Y. O. Halchenko, M. L. Waskom, and S. S. Ghosh. Nipype: A flexible, lightweight and extensible neuroimaging data processing framework in Python. Frontiers in Neuroinformatics, 5:13, 2011. doi: 10.3389/fninf.2011.00013. URL https://doi.org/10.3389/fninf.2011.00013.

[27] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, Anil Palepu, Keran Rong, Ryutaro Tanno, Khaled Saab, Fan Zhang, Jacob Blum, Andrew Carroll, Kavita Kulkarni, Nenad Tomaˇsev, Dina Zverinski, Ivor Rendulic, Elahe Vedadi, Florian Hasler, Luka Rimanic, Marina Boia, Ivan Budiselic, Ben Feinstein, Mathias Bellaiche, Tom Shefer, Jan Freyberg, Jeremy Ratclif, Ottavia Bertolli, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R. D. Costa, Jos´e R. Penad´es, Gary Peltz, Yossi Matias, James Manyika, Demis Hassabis, Yunhan Xu, Pushmeet Kohli, Annalisa Pawlosky, Alan Karthikesalingam, and Vivek Natarajan. Accelerating scientific discovery with Co-Scientist. Nature, 2026. doi: 10.1038/s4 1586-026-10644-y. URL https://www.nature.com/articles/s41586-026-10644-y.

[28] Q. Hao, F. Xu, Y. Li, et al. Artificial intelligence tools expand scientists’ impact but contract science’s focus. Nature, 649:1237–1243, 2026. doi: 10.1038/s41586-025-09922-y. URL https://doi.org/10.1038/s41586-025-09922-y.

[29] Alif Al Hasan and Sumon Biswas. What breaks when LLMs code? characterizing operational safety failures of agentic code assistants, 2026. URL https://arxiv.org/abs/2605.30777.

[30] K. Huang, S. Zhang, H. Wang, Y. Qu, Y. Lu, Y. Roohani, et al. Biomni: A general-purpose biomedical AI agent, 2025. URL https://doi.org/10.1101/2025.05.30.656746. Publica tion Title: bioRxiv.

[31] A. Iraji, Z. Fu, A. Faghiri, M. Duda, J. Chen, S. Rachakonda, T. DeRamus, P. Kochunov, B. M. Adhikari, A. Belger, J. M. Ford, D. H. Mathalon, G. D. Pearlson, S. G. Potkin, A. Preda, J. A. Turner, T. G. M. van Erp, J. R. Bustillo, K. Yang, K. Ishizuka, A. Faria, A. Sawa, K. Hutchison, E. A. Osuch, J. Theberge, C. Abbott, B. A. Mueller, D. Zhi, C. Zhuo, S. Liu, Y. Xu, M. Salman, J. Liu, Y. Du, J. Sui, T. Adali, and V. D. Calhoun. Identifying canonical and replicable multi-scale intrinsic connectivity networks in 100k+ resting-state fmri datasets. Human Brain Mapping, 44(17):5729–5748, 2023. doi: https://doi.org/10.1002/hbm.26472. URL https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.26472.

[32] Leslie K. John, George Loewenstein, and Drazen Prelec. Measuring the prevalence of questionable research practices with incentives for truth telling. Psychological Science, 23(5):524–532, 2012. doi: 10.1177/0956797611430953.

[33] Jean Kaddour, Srijan Patel, Gb\`etondji Dovonon, Leo Richter, Pasquale Minervini, and Matt J. Kusner. Agentic uncertainty reveals agentic overconfidence, 2026. URL https://arxiv.org/ abs/2602.06948.

[34] David B. Keator, Theo G.M. van Erp, Jessica A. Turner, Gary H. Glover, Bryon A. Mueller, Thomas T. Liu, James T. Voyvodic, Jerod Rasmussen, Vince D. Calhoun, Hyo Jong Lee, Arthur W. Toga, Sarah McEwen, Judith M. Ford, Daniel H. Mathalon, Michele Diaz, Daniel S. O’Leary, H. Jeremy Bockholt, Syam Gadde, Adrian Preda, Cynthia G. Wible, Hal S. Stern, Aysenil Belger, Gregory McCarthy, Burak Ozyurt, and Steven G. Potkin. The function biomedical informatics research network data repository. NeuroImage, 124: 1074–1079, 2016. ISSN 1053-8119. doi: 10.1016/j.neuroimage.2015.09.003. URL https://www.sciencedirect.com/science/article/pii/S1053811915007995.

[35] J. Lefort-Besnard, T. E. Nichols, and C. Maumet. Statistical inference for neuroimaging multiverse analyses with the same-data meta-analysis. Imaging Neuroscience, 2025. doi: 10.1162/imag a 00513. URL https://doi.org/10.1162/imag\_a\_00513.

[36] Xinhui Li, Nathalia Bianchini Esper, Lei Ai, Steve Giavasis, Hecheng Jin, Eric Feczko, Ting Xu, et al. Moving beyond processing- and analysis-related variation in resting-state functional brain imaging. Nature Human Behaviour, 8:2003–2017, 2024. doi: 10.1038/s41562-024-01942-4. URL https://doi.org/10.1038/s41562-024-01942-4.

[37] Zhen-Qi Liu, Andrea I. Luppi, Justine Y. Hansen, Ye Ella Tian, Andrew Zalesky, B. T. Thomas Yeo, Ben D. Fulcher, and Bratislav Misic. Benchmarking methods for mapping functional connectivity in the brain. Nature Methods, 22(7):1593–1602, 2025. doi: 10.1038/s41592-025-0 2704-4. URL https://doi.org/10.1038/s41592-025-02704-4.

[38] S. Marek, B. Tervo-Clemmens, F. J. Calabro, et al. Reproducible brain-wide association studies require thousands of individuals. Nature, 603:654–660, 2022. doi: 10.1038/s41586-022-04492-9. URL https://doi.org/10.1038/s41586-022-04492-9.

[39] C. J. Markiewicz, K. J. Gorgolewski, F. Feingold, et al. The OpenNeuro resource for sharing of neuroscience data. eLife, 10:e71774, 2021. doi: 10.7554/eLif e.71774. URL https: //doi.org/10.7554/eLife.71774.

[40] Christopher J. Markiewicz, Alejandro De La Vega, Adina Wagner, Yaroslav O. Halchenko, Karolina Finc, Rastko Ciric, Mathias Goncalves, Dylan M. Nielson, James D. Kent, John A. Lee, Shashank Bansal, Russell A. Poldrack, and Krzysztof J. Gorgolewski. poldracklab/fitlins: 0.11.0. Zenodo, 2022. URL https://doi.org/10.5281/zenodo.7217447. Version 0.11.0.

[41] L. Mitchener, A. Yiu, B. Chang, et al. Kosmos: An AI Scientist for Autonomous Discovery, 2025. URL https://doi.org/10.48550/arXiv.2511.02824. Publication Title: arXiv.

[42] Moonshot AI. Kimi K2.5, 2026. URL https://platform.kimi.ai/docs/guide/kimi-k2-5 -quickstart. Accessed 21 May 2026.

[43] OpenAI. Introducing GPT-5.5, 2026. URL https://openai.com/index/introducing-gpt -5-5/. Accessed 21 May 2026.

[44] OpenNeuro. OpenNeuro Vocabulary (ONVOC). BioPortal, National Center for Biomedical Ontology, 2026. URL https://bioportal.bioontology.org/ontologies/ONVOC. Accessed 2026.

[45] R. A. Poldrack, C. I. Baker, J. Durnez, et al. Scanning the horizon: Towards transparent and reproducible neuroimaging research. Nature Reviews Neuroscience, 18:115–126, 2017. doi: 10.1038/nrn.2016.167. URL https://doi.org/10.1038/nrn.2016.167.

[46] R. A. Poldrack, C. J. Markiewicz, S. Appelhof, et al. The past, present, and future of the Brain Imaging Data Structure (BIDS). Imaging Neuroscience, 2:1–19, 2024. doi: 10.1162/im ag a 00103. URL https://doi.org/10.1162/imag\_a\_00103.

[47] T. Salo, T. Yarkoni, T. E. Nichols, J.-B. Poline, M. Bilgel, K. L. Bottenhorn, et al. NiMARE: Neuroimaging Meta-Analysis Research Environment. Aperture Neuro, 3:1–32, 2023. doi: 10.5 2294/001c.87681. URL https://doi.org/10.52294/001c.87681.

[48] Joseph P. Simmons, Leif D. Nelson, and Uri Simonsohn. False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. Psychological Science, 22(11):1359–1366, 2011. doi: 10.1177/0956797611417632.

[49] U. Simonsohn, J. P. Simmons, and L. D. Nelson. Specification curve analysis. Nature Human Behaviour, 4:1208–1214, 2020. doi: 10.1038/s41562-020-0912-z. URL https://doi.org/10 .1038/s41562-020-0912-z.

[50] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. In Advances in Neural Information Processing Systems, volume 35, pages 9460–9471, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/202 2/hash/3d719fee332caa23d5038b8a90e81796-Abstract-Conference.html.

[51] S. Steegen, F. Tuerlinckx, A. Gelman, and W. Vanpaemel. Increasing transparency through a multiverse analysis. Perspectives on Psychological Science, 11(5):702–712, 2016. doi: 10.1177/ 1745691616658637. URL https://doi.org/10.1177/1745691616658637.

[52] K. Swanson, W. Wu, N. L. Bulaong, J. E. Pak, and J. Zou. The Virtual Lab of AI agents designs new SARS-CoV-2 nanobodies. Nature, 646:716–723, 2025. doi: 10.1038/s41586-025-09442-9. URL https://doi.org/10.1038/s41586-025-09442-9.

[53] Cheng Wang, Zhibin He, Zhihao Peng, Shengyuan Liu, Yufan Hu, Carl Yang, Lifang He, Lichao Sun, Xiang Li, and Yixuan Yuan. NeuroClaw Technical Report, 2026. URL https: //arxiv.org/abs/2604.24696. Publication Title: arXiv (2604.24696).

[54] Z.AI. GLM-5.1, 2026. URL https://docs.z.ai/guides/llm/glm-5.1. Accessed 21 May 2026.

# Supplementary Information for “Bringing analytic rigor to agentic AI for science: The Brain Researcher platform for neuroimaging data analysis”

Zijiao Chen<sup>1</sup>, Nicholas Lu<sup>1</sup>, Xinhui Li<sup>2</sup>, Jocelyn A. Ricard<sup>1</sup>, Ce Ju<sup>3</sup>, Huan H. Wang<sup>1</sup>, Christian Kindermann<sup>1</sup>, Jeanette A. Mumford<sup>1</sup>, Steven Dillmann<sup>1</sup>, James Kent<sup>4</sup>,   
Alejandro de la Vega<sup>4</sup>, Sanmi Koyejo<sup>1</sup>, Vince D. Calhoun<sup>2</sup>, Joshua W. Buckholtz<sup>1</sup>, Juan Helen Zhou<sup>5</sup>, Stefen Bollmann<sup>1,6</sup>, Russell A. Poldrack<sup>1</sup> Correspondence: russpold@stanford.edu

<sup>2</sup>Tri-institutional Center for Translational Research in Neuroimaging and Data Science (TReNDS), Georgia State University, Georgia Institute of Technology, Emory University, Atlanta, GA, USA <sup>3</sup>Inria, CEA, Universit´e Paris-Saclay, Palaiseau, France <sup>4</sup>The University of Texas at Austin, Austin, TX, USA <sup>5</sup>National University of Singapore, Singapore <sup>6</sup>The University of Queensland, Brisbane, QLD, Australia

## Supplementary Methods

## Organization of supplement

The Supplementary Methods follow the life cycle of a Brain Researcher episode. The main Methods define the conceptual machinery: what a research episode is, why MCP separates the language model from scientific infrastructure, how tools and datasets are selected, how execution is observed, and how claims are reviewed before memory writeback. This supplement makes that machinery reproducible by specifying the records, checks, gates, artifacts, and evaluation rules used during implementation.

The sections are organized in the order in which an episode runs. S1 defines the system boundary and MCP interface. S2 defines the episode object, state machine, and a running predictive-modeling example. S3 describes BR-KG, ONVOC normalization, and evidence assembly. S4 describes dataset and resource resolution. S5 defines the typed tool registry and action space. S6 defines the constraint compiler and commitment gate. S7 defines execution, preflight, and provenance. S8 defines the six terminal claim states used for the reported collaborator and bounded-autonomous evaluations and routes unresolved decisions for human escalation. S9 defines memory writeback, conflict detection, and BR-KG promotion. S10 defines bounded self-evolving episodes and Harbor-based validation. S11 defines the evaluation protocol. S12 states limitations and reporting boundaries. Appendix/Data Cards A–J contain concrete episode identifiers, snapshot identifiers, schema excerpts, run-bundle excerpts, review rules, claim cards, and evaluation ledgers. Appendix K presents a representative generated-code excerpt for a collaborator case; the full per-case reports for the three collaborator cases and two bounded self-evolving campaigns are released with the repository and described in Appendix L.

## Public release and reuse

The public release includes BR-KG itself. It provides graph snapshots, node and edge schemas, ontology-normalization records, public evidence and provenance fields, registry links, and example graph paths for the material described in S3 and Appendix B. The project site, https: //brain-researcher.com/, provides access to the public MCP interface, the released Brain Researcher skills, and deployment notes: environment setup, graph loading, registry initialization, and example queries.

Source datasets are handled separately. Public resources are cited and linked, but not rehosted unless their licenses permit redistribution. S4 and Appendices B and C record the datasets, derivatives, metadata, and access checks used by Brain Researcher. Controlled-access, collaboratorprovided, or license-restricted human-subject data remain with the original data custodians or contributing groups.

The release includes the records needed to check the reported results without redistributing restricted raw data. S7 and Appendix F define the run-bundle schema, provenance fields, artifact manifests, logs, parameters, and observed outputs. Derived summaries, aggregate tables, figure source data, task manifests, scoring sheets, and run-bundle metadata are included in the public package.

S11 and Appendix J define the benchmark release: task manifests, model conditions, with-BR/ without-BR comparisons, scoring rules, metric denominators, exclusions, and aggregate outputs. Private paths, provider keys, human-subject identifiers, and license-restricted source material are excluded from the public release. Code-release details are tied to the registry and execution records: S5 describes the callable tool cards, S7 describes execution provenance, and Appendices D and F give the corresponding registry and artifact-manifest records.

We welcome community extensions to BR-KG. The public site explains how to propose new nodes, edges, evidence records, registry links, and deployment recipes. Contributions must include source provenance, license information, schema-valid records, and review status, so additions remain traceable as the graph grows.

Key operational terms
<table><tr><td rowspan=1 colspan=1>Term</td><td rowspan=1 colspan=1>Meaning in this supplement</td></tr><tr><td rowspan=1 colspan=1>Research episode</td><td rowspan=1 colspan=1>Persistent runtime unit containing the question, evidence, selectedplan, resources, constraints, execution artifacts, review verdicts,and memory status for one investigation.</td></tr><tr><td rowspan=1 colspan=1>MCP operation</td><td rowspan=1 colspan=1>Typed endpoint exposed by BR-MCP, for example tool discovery,dataset resolution, recipe generation, artifact inspection, or review.</td></tr><tr><td rowspan=1 colspan=1>Registry specification</td><td rowspan=1 colspan=1>Machine-readable record used for discovery, compatibility checkingparameter validation, and recipe generation.</td></tr><tr><td rowspan=1 colspan=1>Workflow specification</td><td rowspan=1 colspan=1>Planning-level description of a family of analyses and its expectedinputs, outputs, constraints, and backend recipes.</td></tr><tr><td rowspan=1 colspan=1>Executable wrapper</td><td rowspan=1 colspan=1>Actual callable Python tool, command-line wrapper, Neurodeskmodule, container command, or scheduler script.</td></tr><tr><td rowspan=1 colspan=1>Resource ledger</td><td rowspan=1 colspan=1>Episode-specific record of datasets, derivatives, covariates, masks,target variables, fold manifests, and readiness status.</td></tr><tr><td rowspan=1 colspan=1>Constraint compiler</td><td rowspan=1 colspan=1>Component that converts method-condition records, tool-contractclauses, and validators into hard or soft checks.</td></tr><tr><td rowspan=1 colspan=1>Run bundle</td><td rowspan=1 colspan=1>Immutable provenance record containing event trace, trajectorydocument, observation record, analysis bundle, and run card.</td></tr><tr><td rowspan=1 colspan=1>Scientific verification layer</td><td rowspan=1 colspan=1>Acceptance gate that checks run bundles, artifacts, and candidateclaims against validity rules before claim communication or mem-ory writeback.</td></tr><tr><td rowspan=1 colspan=1>Review card</td><td rowspan=1 colspan=1>Structured verification verdict, including rule triggers, risk tags,required fixes, and claim eligibility.</td></tr><tr><td rowspan=1 colspan=1>Claim card</td><td rowspan=1 colspan=1>Reviewed memory object derived from a run bundle; it stores aclaim with scope, evidence level, caveats, and provenance pointers.</td></tr><tr><td rowspan=1 colspan=1>Harbor verifier</td><td rowspan=1 colspan=1>Task-level execution checker used in bounded autonomous episodes.It can verify files, schemas, tests, and required statistics, but it isnot the scientific reviewer.</td></tr></table>

## S1. System architecture and MCP interface

## S1.1. Overall architecture

Brain Researcher is organized around a research episode rather than a single prompt-response exchange. Each episode enters through an MCP-compatible client or coding-agent harness [3, 28], is mediated by the Brain Researcher MCP control plane, and is grounded by domain backends. The outer layer receives the researcher’s goal, writes or runs code when needed, interacts with the user, and presents results. The middle layer, BR-MCP, exposes typed operations for each stage of the episode lifecycle: planning, discovery, execution, review, and memory writeback. The full operation surface is enumerated in S1.3. The inner layer consists of domain backends: BR-KG, dataset and BIDS resolvers [14, 24], the tool registry, execution backends, storage services, review-rule registries, memory stores, and governance records.

A typical predictive-modeling request traverses the layers as follows. The user asks whether a behavioral score can be predicted from a resolved neuroimaging feature matrix. The outer client converts the request into a typed intent: prediction, fMRI/connectivity features, target variable, covariates, and required folds. BR-MCP retrieves candidate datasets, workflows, and tool families from the registry and BR-KG. The resource resolver checks whether the feature matrix, target vector, covariates, and fold manifest are reachable and authorized. If the resolver returns pass or warn, BR-MCP packages an execution recipe with expected artifacts; if it returns block, the plan returns to revision before execution. After execution, the scientific verification layer inspects the run bundle and candidate claims before memory writeback.

Brain Researcher system architecture  
![](images/4adafc77ef57a75b79718808d144eaed881f059c4b73464763ea90a07affd44c.jpg)  
Supplementary Figure S1: Detailed Brain Researcher system architecture. The architecture separates the researcher-facing coding-agent harness, the BR-MCP control plane, and the domain execution substrate. The episode flow runs from question intake and typed intent resolution through BR-KG and registry lookup, resource checks, constraint compilation, commitment, execution, review, and memory writeback. The control-plane boxes indicate where typed MCP operations expose admissible actions, while the backend boxes indicate where Python, container, Neurodesk, Slurm/HPC, storage, and provenance services produce environment-observed artifacts. Dataset readiness, tool admissibility, execution success, and claim eligibility are returned by typed system state; the language model interprets, requests, and explains those states.

This separation is intentional. The language model may interpret a natural-language request, compare MCP-returned options, ask for recipes, and explain observed outputs. It cannot certify that a registry ID exists, override a resolver block, mark a run as successful without artifacts, or accept a scientific claim without a verification verdict. Tool availability, dataset readiness, policy permission, execution success, and claim eligibility are supplied by MCP, BR-KG, the registry, resolvers, execution backends, and the scientific verification layer.

## S1.2. Model routing and reproducibility

Language-model routing afects generation and interpretation, not certification. Coding-oriented tasks such as script generation, debugging, and file manipulation may be routed to code-specialized models. Scientific planning, evidence interpretation, review, and claim calibration may be routed to analytical models. The run card records the task type, selected provider, prompt-template version, policy context, registry snapshot, BR-KG snapshot, and memory namespace.

Example run-card excerpt:

```yaml
run_card:
episode_id: ep_hcp_predict_001
mode: benchmark
model_route:
task_type: code_generation
provider: fixed_provider_id
prompt_template: planning_v3
fixed_snapshots:
registry: registry_2026_05_15
brkg: brkg_snapshot_2026_05_10
policy_context: benchmark_locked
memory_namespace: benchmark_isolated
```

For reported evaluations, the full configuration is fixed before launch and recorded in the run card: model identity, prompt-template and run-card schema versions, registry and BR-KG snapshots, policy flags, execution roots, backend settings, and memory namespace. Provider fallback may be used during development, but benchmark, collaborator-case, and bounded autonomous claims are reported only from fixed model versions, fixed prompt templates, fixed registry snapshots, fixed BR-KG snapshots, and fixed evaluation partitions.

Table 1 lists the resolved model identifiers and the coding-agent surface used for each of the seven agents in the paired benchmarks. All agents were driven through their coding-agent command-line surfaces (not direct chat or completion APIs), so decoding parameters follow each surface’s defaults; we did not override temperature for the agent rollouts. The benchmark runs reported here were executed in May–June 2026 (the 60-task tool-routing condition was run on 2026-06-05). The three evidence-support judges were run deterministically at temperature 0 with a fixed random seed (7).

## S1.3. MCP operation surface and registry boundary

The MCP surface is the public control plane, not the complete neuroimaging software catalog. The checked implementation contains 87 decorated MCP operations and 3 decorated MCP resources. These operations span five functional families: discovery and resolution (tools, workflows, datasets, BR-KG, literature); planning and recipe generation; execution observability and artifact inspection;

Table 1: Resolved model identifiers and coding-agent surfaces for the seven benchmarked agents.
<table><tr><td rowspan=1 colspan=1>Paper label</td><td rowspan=1 colspan=1>Resolved model identifier</td><td rowspan=1 colspan=1>Coding-agent surface</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.8</td><td rowspan=1 colspan=1>claude-opus-4-8</td><td rowspan=1 colspan=1>Claude Code (claude -p)</td></tr><tr><td rowspan=1 colspan=1>Codex GPT-5.5</td><td rowspan=1 colspan=1>gpt-5.5</td><td rowspan=1 colspan=1>Codex CLI (codex exec)</td></tr><tr><td rowspan=1 colspan=1>Gemini 3.1 Pro</td><td rowspan=1 colspan=1>google/gemini-3.1-pro-preview</td><td rowspan=1 colspan=1>OpenCode (opencode run)</td></tr><tr><td rowspan=1 colspan=1>GLM-5.1</td><td rowspan=1 colspan=1>zai-coding-plan/glm-5.1</td><td rowspan=1 colspan=1>OpenCode (opencode run)</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Pro</td><td rowspan=1 colspan=1>deepseek/deepseek-v4-pro</td><td rowspan=1 colspan=1>OpenCode (opencode run)</td></tr><tr><td rowspan=1 colspan=1>Kimi K2.5</td><td rowspan=1 colspan=1>opencode/kimi-k2.5</td><td rowspan=1 colspan=1>OpenCode (opencode run)</td></tr><tr><td rowspan=1 colspan=1>Qwen3.6-Plus</td><td rowspan=1 colspan=1>opencode/qwen3.6-plus</td><td rowspan=1 colspan=1>OpenCode (opencode run)</td></tr></table>

grounding and scientific review; and memory access and report generation. Slurm/HPC scheduling is exposed as a separate operation family. The resources expose structured tools, datasets, and workflow lookups. Full capability-family tables, implementation evidence sources, compute-graph diagrams, safety-gate tables, and access-mode tables are reported in Appendices A, D, F, and I rather than repeated in the main Supplementary Methods.

The broader registry stores the objects MCP searches, validates, packages, and observes. The distinction is important because an MCP operation is not the same as a workflow specification, registry specification, or executable wrapper.

<table><tr><td rowspan=1 colspan=1>Object</td><td rowspan=1 colspan=1>Example</td><td rowspan=1 colspan=1>Used for</td></tr><tr><td rowspan=1 colspan=1>MCP operation</td><td rowspan=1 colspan=1>resolve_dataset</td><td rowspan=1 colspan=1>Ask the control plane whether a datasetand required assets are usable.</td></tr><tr><td rowspan=1 colspan=1>MCP resource</td><td rowspan=1 colspan=1>workflow lookup</td><td rowspan=1 colspan=1>Retrieve structured workflow or datasetsummaries.</td></tr><tr><td rowspan=1 colspan=1>Workflow specification</td><td rowspan=1 colspan=1>predictive_modeling-workflow</td><td rowspan=1 colspan=1>Generate a plan and expected artifactlist.</td></tr><tr><td rowspan=1 colspan=1>Registry specification</td><td rowspan=1 colspan=1>fsl_randomise_contract</td><td rowspan=1 colspan=1>Validate parameters, compatibility, andbackend requirements.</td></tr><tr><td rowspan=1 colspan=1>Executable wrapper</td><td rowspan=1 colspan=1>AFNI/FSL/Workbench com-mand wrapper [6, 16, 27]</td><td rowspan=1 colspan=1>Run a backend command through anapproved execution substrate.</td></tr></table>

Inventory counts are reported once in S5.1 so that reader attention stays on the control boundary here.

## S1.4. LLM/MCP decision boundary

The LLM/MCP boundary separates interpretation from certification.

<table><tr><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1>LLM or outer clientdoes</td><td rowspan=1 colspan=1>MCP/backends do</td><td rowspan=1 colspan=1>Output</td></tr><tr><td rowspan=1 colspan=1>Intent</td><td rowspan=1 colspan=1>Interprets the user requestand proposes modalities,datasets, methods, andconstraints.</td><td rowspan=1 colspan=1>Receives typed query argu-ments.</td><td rowspan=1 colspan=1>Typed planningquery.</td></tr><tr><td rowspan=1 colspan=1>Discovery</td><td rowspan=1 colspan=1>Asks for candidatedatasets, tools, workflows,or evidence.</td><td rowspan=1 colspan=1>Searches BR-KG, registry,dataset catalog, and litera-ture connectors.</td><td rowspan=1 colspan=1>Ranked candidatecards.</td></tr><tr><td rowspan=1 colspan=1>Resolution</td><td rowspan=1 colspan=1>Chooses among returnedoptions.</td><td rowspan=1 colspan=1>Canonicalizes dataset/toolIDs and returns metadata,schemas, and readinessstatus.</td><td rowspan=1 colspan=1>Resolved resource/tool record.</td></tr><tr><td rowspan=1 colspan=1>Feasibility</td><td rowspan=1 colspan=1>Requests feasibility checks.</td><td rowspan=1 colspan=1>Validates roots, permis-sions, assets, tool con-tracts, and active con-straints.</td><td rowspan=1 colspan=1>pass / warn / blockverdicts.</td></tr><tr><td rowspan=1 colspan=1>Recipe</td><td rowspan=1 colspan=1>Requests backend-specificinstructions.</td><td rowspan=1 colspan=1>Packages Python, Neu-rodesk, container, or Slurmrecipe with expected out-puts.</td><td rowspan=1 colspan=1>Execution recipe andartifact contract.</td></tr><tr><td rowspan=1 colspan=1>Execution</td><td rowspan=1 colspan=1>Delegates computation toallowed backend or codingharness.</td><td rowspan=1 colspan=1>Produces logs, artifacts,metrics, and environmentobservations.</td><td rowspan=1 colspan=1>Run bundle.</td></tr><tr><td rowspan=1 colspan=1>Verification</td><td rowspan=1 colspan=1>Proposes candidate claimsor report language.</td><td rowspan=1 colspan=1>Checks artifacts, con-straints, scorecards, QCoutputs, and claim scope</td><td rowspan=1 colspan=1>Review card andmemory eligibility.</td></tr></table>

The LLM may choose among MCP-returned options, but it cannot invent a registry ID, override a resolver block, bypass a policy gate, or promote a claim to accepted memory without a verification verdict.

## S1.5. Orchestration, subagents, and policy gates

The orchestration service manages run creation, polling, cancellation, retry, progress tracking, streaming logs, event updates, and durable state. Generated artifacts are stored under run-specific roots with manifests and checksums. The implementation may use specialized subagents, but each has a bounded role.

# Human-Agent Collaboration Boundary in Scientific Workflow

![](images/42e03ae3083fa78412d7fee140c56b5b392ca607b0f6243ec2fb84ebb3e81716.jpg)

Supplementary Figure S2: Human-agent authority boundary in Brain Researcher. The diagram separates researcher authority, language-model assistance, MCP-mediated system authority, and scientific verification across the episode lifecycle. Brain Researcher can search evidence, compare options, prepare recipes, observe execution, assemble review inputs, and record memory candidates. The researcher retains authority over question framing, admissible tradeofs, commitment-gate approval, interpretation, authorship, and final claim language. System gates prevent the model from inventing registry identifiers, overriding resource blocks, accepting missing artifacts, or promoting claims without a review verdict.

<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1>Output</td><td rowspan=1 colspan=1>Boundary</td></tr><tr><td rowspan=1 colspan=1>Critic</td><td rowspan=1 colspan=1>Candidate plan, review find-ings, cycle record.</td><td rowspan=1 colspan=1>approve / revise / blockterminate.</td><td rowspan=1 colspan=1>Cannot certify artifactswithout run-bundleevidence.</td></tr><tr><td rowspan=1 colspan=1>Recovery agent</td><td rowspan=1 colspan=1>Failure logs, missing outputs,backend errors.</td><td rowspan=1 colspan=1>Retry or substitutionproposal.</td><td rowspan=1 colspan=1>Cannot erase the origi-nal failure record.</td></tr><tr><td rowspan=1 colspan=1>Provenance tracker</td><td rowspan=1 colspan=1>Decisions, tool calls, parame-ter bindings.</td><td rowspan=1 colspan=1>Event trace and trajec-tory entries.</td><td rowspan=1 colspan=1>Records actions; doesnot judge scientificvalidity.</td></tr><tr><td rowspan=1 colspan=1>Router</td><td rowspan=1 colspan=1>Task type and policy con-text.</td><td rowspan=1 colspan=1>Model/backend/subagentroute.</td><td rowspan=1 colspan=1>Routing is recorded inthe run card.</td></tr><tr><td rowspan=1 colspan=1>Supervisor</td><td rowspan=1 colspan=1>Open caveats, validationprogress, budget.</td><td rowspan=1 colspan=1>Next branch proposal.</td><td rowspan=1 colspan=1>Cannot expand beyondthe declared designspace.</td></tr></table>

Execution is controlled by allowed filesystem roots, network flags, dangerous-tool flags, directexecution flags, timeout budgets, authentication checks, and approval phrases for pipeline execution. Heavy neuroimaging work is normally performed outside the MCP service by local Python, Neurodesk, a generic container runtime, Slurm/HPC, Kubernetes, AWS Batch, or another configured backend. Direct MCP execution exists only as a gated administrative path and is disabled by default in the evaluated configuration. Adaptive optimization surfaces such as contextual bandits, ofline reinforcement learning, and drift detection are engineering surfaces only; they are not treated as evaluated capabilities unless explicitly declared in the evaluation card.

## S2. Research episode runtime and run states

## S2.1. Episode object and persisted state

The research episode is the core runtime unit. It persists the scientific question, planning state, evidence state, candidate alternatives, selected plan, selected tools, parameter bindings, resourceresolution results, execution outputs, QC artifacts, review verdicts, recovery attempts, limitation notes, and memory-writeback state. This prevents the system from treating later turns as isolated prompts: later decisions can depend on prior evidence, failed attempts, unresolved caveats, resource blockers, and review outcomes.

Episode record fields are grouped by role:
<table><tr><td rowspan=1 colspan=1>Group</td><td rowspan=1 colspan=1>Fields</td></tr><tr><td rowspan=1 colspan=1>Question state</td><td rowspan=1 colspan=1>input question, normalized entities, scientific scope, intended outputtype.</td></tr><tr><td rowspan=1 colspan=1>Resource state</td><td rowspan=1 colspan=1>data requirements, candidate datasets, resolved resources, missingassets, access class.</td></tr><tr><td rowspan=1 colspan=1>Planning state</td><td rowspan=1 colspan=1>evidence bundle, candidate workflows, candidate tools, active con-straints, selected plan.</td></tr><tr><td rowspan=1 colspan=1>Execution state</td><td rowspan=1 colspan=1>backend recipe, expected artifacts, produced artifacts, logs, QC out-puts, recovery attempts.</td></tr><tr><td rowspan=1 colspan=1>Review state</td><td rowspan=1 colspan=1>review cards, rule findings, claim candidates, caveats, verdicts.</td></tr><tr><td rowspan=1 colspan=1>Memory state</td><td rowspan=1 colspan=1>memory eligibility, claim-card status, relation events, final comple-tion status.</td></tr></table>

Example episode-record excerpt:

episode\_id: ep\_hcp\_predict\_001

state: planning\_blocked

question\_type: prediction

selected\_dataset: HCP-YA

candidate\_workflow: predictive\_modeling

required\_assets:

\- feature\_matrix

\- target\_variable

\- covariate\_table

\- fold\_manifest

resource\_verdict: block

blocker: fold\_manifest\_missing

next\_action: resolve\_asset\_or\_choose\_alternative\_workflow

## S2.2. Scientific stages

At the scientific level, an episode moves through stages that each produce a concrete object.

<table><tr><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>What happens</td><td rowspan=1 colspan=1>Concrete output</td></tr><tr><td rowspan=1 colspan=1>Question framing</td><td rowspan=1 colspan=1>Natural-language goal is converted into typedscientific entities and admissible outputs.</td><td rowspan=1 colspan=1>Typed questionrecord.</td></tr><tr><td rowspan=1 colspan=1>Evidence assembly</td><td rowspan=1 colspan=1>BR-KG, literature, dataset catalogues, tool reg-istries, and meta-analysis stores are queried.</td><td rowspan=1 colspan=1>Evidence bundle.</td></tr><tr><td rowspan=1 colspan=1>Option-set formation</td><td rowspan=1 colspan=1>Candidate datasets, tools, workflows, and parame-terizations are identified.</td><td rowspan=1 colspan=1>Candidate ledger.</td></tr><tr><td rowspan=1 colspan=1>Constraint compilation</td><td rowspan=1 colspan=1>Method-condition records and tool contracts areturned into active checks.</td><td rowspan=1 colspan=1>Active constraint set.</td></tr><tr><td rowspan=1 colspan=1>Commitment</td><td rowspan=1 colspan=1>Human researcher, critic, or evaluation harnessapproves/revises/blocks the plan.</td><td rowspan=1 colspan=1>Commitment-gaterecord.</td></tr><tr><td rowspan=1 colspan=1>Execution</td><td rowspan=1 colspan=1>Approved recipe is run under the selected backend.</td><td rowspan=1 colspan=1>Run bundle and arti-fact manifest.</td></tr><tr><td rowspan=1 colspan=1>Verification</td><td rowspan=1 colspan=1>Artifacts, logs, QC, scorecards, and candidateclaims are checked.</td><td rowspan=1 colspan=1>Review card.</td></tr><tr><td rowspan=1 colspan=1>Memory writeback</td><td rowspan=1 colspan=1>Accepted or qualified claims are stored with prove-nance; rejected claims remain in the run bundle.</td><td rowspan=1 colspan=1>Claim card or no-write decision.</td></tr></table>

## S2.3. Running example: one predictive-modeling episode

The running example used throughout this supplement is an HCP-YA predictive-modeling replay [30, 34]. The question is whether a behavioral target can be predicted from Schaefer-100 x 7 features [26] using a fixed subject intersection. The resource card specifies N = 326, five Liu components [19], a pyspi statistic catalogue, feature matrices, candidate target variables, subject-intersection logic, missingness rules, and backend reachability. A valid run requires a feature matrix, target vector, covariates, and a fold manifest. During resolution, the dataset can pass dataset-level readiness while failing asset-level readiness if the fold manifest or target variable is missing. The constraint compiler then creates a hard fold-manifest check and soft warnings for confound specification or controversial preprocessing choices. If the fold manifest is absent, the commitment gate returns block before execution. If all required assets pass, BR-MCP packages a recipe and expected artifacts: predictions, scorecard, model metadata, fold manifest, and artifact manifest. Review then checks leakage, grouped cross-validation, permutation or null controls, robustness probes, and claim language. Only an accepted or explicitly qualified result can become a claim card.

## S2.4. Runtime states, checkpointing, and recovery

Scientific stages are conceptual. Runtime states are the system state machine used to resume or audit the episode.

<table><tr><td rowspan=1 colspan=1>Runtime state</td><td rowspan=1 colspan=1>Meaning</td></tr><tr><td rowspan=1 colspan=1>initialization</td><td rowspan=1 colspan=1>Create episode record, run root, policy context, and memory names-pace.</td></tr><tr><td rowspan=1 colspan=1>planning</td><td rowspan=1 colspan=1>Populate candidates, evidence, constraints, and recipe drafts.</td></tr><tr><td rowspan=1 colspan=1>execution</td><td rowspan=1 colspan=1>Dispatch or delegate computation.</td></tr><tr><td rowspan=1 colspan=1>review</td><td rowspan=1 colspan=1>Inspect artifacts, logs, QC outputs, scorecards, and claims.</td></tr><tr><td rowspan=1 colspan=1>recovery</td><td rowspan=1 colspan=1>Handle missing outputs, failed commands, invalid artifacts, or blockedconstraints.</td></tr><tr><td rowspan=1 colspan=1>completion</td><td rowspan=1 colspan=1>Record accepted, qualified, or non-acceptance endpoint.</td></tr><tr><td rowspan=1 colspan=1>blocked</td><td rowspan=1 colspan=1>A known hard constraint or missing resource prevents valid progress.</td></tr><tr><td rowspan=1 colspan=1>terminated</td><td rowspan=1 colspan=1>Budget, user action, critic decision, or unrecovered infrastructurefailure stops the episode.</td></tr></table>

A missing fold manifest is a blocked state: the reason is known and can be addressed by resolving the asset or changing workflow. Budget exhaustion after repeated unresolved failures is a terminated state. Checkpoints store the current state, selected candidates, active constraints, resource-resolution outputs, branch history, and run-bundle pointers. Recovery attempts are appended rather than overwritten, so a repaired run still preserves the original failure and repair path.

## S3. BR-KG construction, ONVOC normalization, and evidence assembly

## S3.1. BR-KG role and snapshot metadata

BR-KG is a Neo4j knowledge graph that links scientific records (publications, activation coordinates, statistical maps), cognitive structure (tasks, concepts, brain regions), resources (datasets, tools), embeddings, and operational records (review verdicts, governance entries, run objects). Its role is retrieval and provenance-aware traversal, not scientific proof. It provides source linking, coverage awareness, negative knowledge, method-condition records, and graph paths that can be inspected by planners and reviewers.

In the checked release snapshot (2026-07-07), BR-KG contains 745,949 nodes and 2,461,469 relationships. The manuscript-facing schema defines 27 primary node labels and 34 primary relationship types. Production snapshots may contain additional compatibility, migration, enrichment, operational, or retrieval labels (this snapshot carries 108 active node labels and 125 active relationship types), so primary-schema counts and all-active-label counts are reported separately in the Appendix/Data Cards. Deployment metadata such as Neo4j 5.20.0 Community, one ready K3s BR-KG pod, 139 indexes, and 48 constraints is release-readiness metadata, not evidence for scientific correctness.

Example traversal role:

query: "N-back working memory activation"   
normalized\_task: ONVOC:N\_BACK   
linked\_concepts:   
- working\_memory

retrieved\_records: - publications - task contrasts - activation coordinates - brain regions - method constraints   
used\_by: - evidence\_bundle - planner - reviewer

## S3.2. Source governance, ingestion, and validation

Graph ingestion separates accepted records from candidate records. Accepted records are sourcebacked objects such as publications, coordinates, maps, task records, dataset records, tool descriptors, anatomical regions, and curated method-condition records. Candidate records may come from automated extraction, KGGEN, GABRIEL, or real-time retrieval. They remain in candidate lanes until they pass schema validation, source marking, provenance recording, license/citation checks, and acceptance rules.

<table><tr><td rowspan=1 colspan=1>Record type</td><td rowspan=1 colspan=1>Example</td><td rowspan=1 colspan=1>Status before use asgraph fact</td></tr><tr><td rowspan=1 colspan=1>Accepted record</td><td rowspan=1 colspan=1>Publication with DOI, source loader,snapshot date, and citation.</td><td rowspan=1 colspan=1>Usable as graph evidence.</td></tr><tr><td rowspan=1 colspan=1>Candidate record</td><td rowspan=1 colspan=1>Extracted task-region relation fromKGGEN.</td><td rowspan=1 colspan=1>Candidate only until vali-dated and accepted.</td></tr><tr><td rowspan=1 colspan=1>Held or rejected record</td><td rowspan=1 colspan=1>Unsupported generated relation orsource-incomplete extraction.</td><td rowspan=1 colspan=1>Not promoted to graph fact.</td></tr></table>

Each releasable source row records snapshot date, source URL, license, required citation, loader path, loader version, command line, configuration hash, input artifact path, output manifest, artifact path, and release disposition. Source aliases are normalized, source-like missing values are backfilled or removed, and source-specific licenses and citations are pinned before manuscript claims are made.

Manual record-quality audit. We manually reviewed 400 randomly sampled BR-KG records: 200 nodes and 200 directed edges. We classified 280 as correct, 45 as incomplete or uncertain, 53 as pre-existing source-integration errors, and 22 as BR-side errors. The three non-correct categories comprised:

• Incomplete or uncertain (n = 45): coordinate-space metadata was incomplete for 23 map nodes: 14 FitLins statistical maps had a known template but stored space=unknown; five NeuroVault maps lacked a materialized canonical space despite GenericMNI target-template metadata; and four additional NeuroVault maps did not canonicalize target template image=GenericMNI into the KG space field. Available evidence was insuficient to assess 21 records: nine HAS COORDINATE edges had an unresolved source coordinate frame; seven Neurosynth coordinate nodes had source metadata marked UNKNOWN while the loader assigned MNI; three EvidenceSpan nodes had insuficient frozen content and live-source snapshot drift; and two GABRIEL MeasurementRun nodes lacked a matching fixed upstream run record. One Neurobagel Dataset node had a name containing an unresolved literal backslash-n formatting artifact.

• Pre-existing source-integration errors (n = 53): 37 Neurosynth records in which TAL coordinates were materialized as MNI, and 16 NeuroStore records in which StudyObjective was materialized in publication or collection title fields. These failures arose in pre-existing source-specific integration paths and do not imply that the raw Neurosynth or NeuroStore records were incorrect.

• BR-side errors (n = 22): five NiCLIP index-ofset errors; five dataset-to-contrast crosswalk over-propagations; three configuration-derived target mismatches; five GABRIEL/BR claim, evidence, or cohort-extraction errors; one unsupported manual contrast mapping; two tool/package registry default errors; and one mixed duplicate-DOI/BR merge error.

We are extending this audit into a systematic review across the full BR-KG and will iteratively repair and re-audit the ingestion and normalization issues it identifies; the counts above describe the audited snapshot rather than a completed post-remediation release.

## S3.3. Entity resolution, relationship strength, and semantic linking

Entities are reconciled through exact identifiers, multi-attribute matching, fuzzy string matching, embedding-based alignment, and curated aliases. Matching is applied across publications, datasets, regions, concepts, paradigms, institutions, and researchers. Matched records are linked with deduplication edges that retain matching method and confidence.

Example entity-resolution excerpt:

raw\_labels:   
- "N-back"   
- "WM\_BACK"   
- "verbal working memory"   
resolved\_task\_family: N\_BACK   
non\_equivalent\_fields:   
- contrast\_definition   
- modality   
- stimulus\_type   
- population   
resolution\_note: same task family, not interchangeable experimental objects

Relationship strength may combine literature count, coordinate evidence, spatial overlap, embedding similarity, spatial distance, and optional user feedback. These scores afect retrieval ranking and path prioritization only. They are not review acceptance criteria and cannot by themselves establish a scientific claim.

## S3.4. ONVOC ontology-linker layer

ONVOC normalizes heterogeneous source vocabulary before graph traversal. Free-text task names, contrast labels, cognitive constructs, anatomical labels, dataset-specific aliases, and source-specific terms are mapped to typed ONVOC records. OnvocClass nodes represent normalized vocabulary classes; OntologyConcept nodes represent linked external or internal concepts; IN ONVOC edges connect graph entities to normalized vocabulary entries.

Example ONVOC output:

```yaml
input_terms:
- "N-back"
- "WM_BACK"
- "verbal working memory"
onvoc_task_family: N_BACK
linked_concepts:
- working_memory
linked_graph_nodes:
```

```yaml
- CognitiveTask:N_BACK
- CognitiveConcept:working_memory
non_equivalent_fields:
- contrast
- modality
- coordinate_space
- population
```

ONVOC may link terms to the same task family, but it does not collapse contrast definitions, modalities, populations, or coordinate systems. When direct higher-tier graph evidence is sparse, ONVOC still provides an entity-typing and literature-query backbone. The evidence bundle records whether a decision was supported by accepted graph evidence, validated batch extraction, candidate evidence, or real-time retrieval.

## S3.5. Evidence aggregation and connector-failure reporting

Evidence assembly produces an evidence bundle for each episode. A single bundle may include BR-KG paths, literature results, dataset metadata, tool-registry hits, known failure modes, and review-checklist triggers. Connector failures are stored rather than silently discarded. Figure S3 shows how a recommendation’s validity conditions become applicability checks and then admissible plans, caveats, or blockers, so that it arrives with its limits attached.

Example evidence-bundle excerpt:

```yaml
evidence_bundle:
input_query: "predict behavior from HCP-YA Schaefer-100 x 7 features"
resolved_entities:
dataset: HCP-YA
feature_space: Schaefer100x7
task_family: null
graph_evidence:
tier: accepted_graph_records
paths: [dataset_to_feature_record, method_to_constraint]
real_time_retrieval:
tier: literature_gap_fill
status: completed
connector_failures:
- source: PubMed
reason: timeout
downstream_effect: evidence_coverage_caveat
tool_registry_hits:
- predictive_modeling_family
method_condition_records:
- fold_manifest_required
- leakage_standardization_inside_cv
```

A missing connector result is treated as reduced evidence coverage. It may trigger a caveat or reviewer warning, but it is not treated as evidence that no relevant literature or resource exists.

## S3.6. Embedding lanes

Brain Researcher uses separate embedding lanes for diferent retrieval surfaces. Tool retrieval uses titles, descriptions, command names, and tags. Brain-text retrieval uses task, construct, region, and claim-memory text. BR-KG may also store source-specific embeddings as graph nodes or node properties for selected ingestion lanes. Lane name, model identity, vector dimension, input fields, usage surface, snapshot, and index state are recorded in Appendices B and D.

![](images/120269f41a433cd2306bd453004ddf6decf70e5a04a8a8e76f3c752622c19a87.jpg)  
Supplementary Figure S3: Condition-tagged evidence for scoped planning. The workflow begins with an underspecified scientific request and expands it into entities, methods, datasets, assumptions, and validity conditions that can be checked before execution. BR-KG links literature evidence to those conditions with provenance, while resource and tool resolvers determine which conditions are satisfied, missing, controversial, or incompatible in the current episode. Applicability checks convert the resulting condition set into admissible plans, required sensitivity analyses, caveats, or blockers. The purpose is to expose methodological scope before a run is committed and before a positive or negative result can anchor the narrative.

Embedding similarity retrieves and ranks candidates. It cannot by itself support a scientific claim. The current system operates in text-only concept-matching and retrieval modes. Cross-modal alignment between fMRI activation patterns and ontology concepts is a planned extension and is not used in reported results.

## S3.7. KG extension pipelines and views

Automated extension pipelines maintain BR-KG as a living graph. KGGEN processes full-text publications into typed candidate triples with provenance. GABRIEL converts qualitative domain corpora such as review articles, textbook descriptions, and clinical practice guidelines into schemacompliant quantitative or semi-structured records. On-the-fly retrieval fills query-time coverage gaps when graph evidence is sparse. In all channels, generated records remain candidate or comparison records until explicitly validated, source-marked, and accepted.

Focused task-family or disease-cohort views are subgraph projections over the same underlying records, not separate graphs. They may organize working memory, attention, executive function, language, perception, ADHD, MDD, schizophrenia, healthy aging, and related records for retrieval. View membership supports navigation; it does not change the acceptance status of a finding.

## S3.8. BR-KG visual atlas and release snapshot.

The BR-KG snapshot used in this work is summarized in the Appendix/Data Cards. The atlas reports graph composition, dominant relationship families, schema topology, representative literature/ map/task paths, source coverage, and release-readiness checks. These figures are used to document the retrieval substrate available to Brain Researcher; they are not treated as evidence that any scientific claim is true. During an episode, BR-KG supports source-backed retrieval, task/construct normalization, dataset/tool linking, method-condition lookup, and provenance inspection. Detailed counts, figure panels, source-like values, orphan-node rates, edge-confidence coverage, and release blockers are reported in Appendix B rather than repeated in the main Supplementary Methods.

## S4. Dataset and resource resolution

## S4.1. Dataset-level and asset-level resolution

Dataset and resource resolution converts abstract dataset references into executable resources. Dataset-level resolution maps identifiers, aliases, catalogue entries, local paths, BIDS roots, derivative directories, remote URLs, access class, phenotype availability, license/governance notes, and backend reachability. Asset-level discovery locates the concrete files needed by a workflow: derivatives, covariate files, target files, masks, matrices, parcellations, connectivity matrices, score files, fold manifests, and QC outputs.

A dataset can be known, accessible, and BIDS-valid while still blocking a predictive workflow because the target variable, covariates, or fold manifest are missing. Resource resolution is therefore a precondition for plan commitment, not a background annotation.

Example resolver output:

![](images/862c91aa1ee0d4909070b3000775b59c36e993ac815fe63a41b2ef949a058802.jpg)

Supplementary Figure S4: BR-KG node-type composition. Node composition summarizes the object families present in the BR-KG snapshot used for retrieval, normalization, and planning support. The chart separates scientific entities such as publications, coordinates, statistical maps, tasks, contrasts, concepts, and brain regions from resource and workflow entities such as datasets, tools, methods, governance records, and prior reviewed claims. Large node families indicate coverage available for traversal and grounding, not evidentiary weight for any individual claim. Detailed counts, release metadata, and source-quality checks are reported in Appendix B.

![](images/14fb19cc7f1259ab7733205e88f1a5ba5f470e6cb7bc44eac58a4b3aa2608824.jpg)

Supplementary Figure S5: BR-KG relationship-type composition. Relationship composition summarizes the edge families that connect publications, coordinates, maps, tasks, contrasts, concepts, regions, datasets, tools, constraints, and provenance records in the BR-KG snapshot. High-count relationship classes define the main traversal surfaces used for evidence assembly, task-to-construct normalization, map lookup, dataset discovery, and tool compatibility checks. Edge frequencies document graph structure and retrieval afordances; support strength for a scientific claim requires source provenance and review context. Appendix B provides the expanded relationship inventory and release-readiness checks.

![](images/693b6c90b299b83b138b4938b8205557ae62dbcc96e783e8071e19c6a85de0d6.jpg)

![](images/ac72fcf48b02a5f3ed7691f33aef6da6ddbe3260ed974b9172472fa57546196d.jpg)

![](images/c64e9117eda364eb6d5c65a10bcee363984064235e754def4d1d8481c5e3bf62.jpg)  
Supplementary Figure S6: BR-KG schema topology and provenance flow. The atlas view summarizes how major node classes are connected, which relationship-label pairs dominate the schema, and how source records feed derived graph objects. Dense schema paths identify the routes most often available for structured retrieval, such as publication-to-coordinate-to-region, task-toconstruct-to-map, and dataset-to-tool-to-workflow traversals. Provenance flow indicates where source systems, loaders, and curation records enter the graph. The figure documents the substrate available for multi-hop planning and evidence lookup; inferential claims require episode-level review.

![](images/d6215f74c7a0781eb2f46149fde59af93f09bad67a9f35aeed1237c0f20922c7.jpg)

![](images/5abf17d058411938227e24b0f0e6fef09dcb68fd271f92c8f3842b80b71d1808.jpg)

![](images/df30540915f474989b208f4a7995fe058e45c76ee7a310b386d7bc8e00751535.jpg)

![](images/da19d3aff9411546d2320c1f372fbc9b260e8207c27108a2bb33de13a31624df.jpg)

![](images/7e5234cb5def28c3650c0fc1021ad63b2eb0f5d7989d88146f9a4ac8668dee77.jpg)

![](images/aadf8ae20f8bbd9c4429c8e4b9ad0749d2eb71e3a1b1c04c49a853c7aab32346.jpg)

![](images/6d7517ac523ec4dd87d9346708f503e09c4cba4f854edeb5e0f4b5261bef2f44.jpg)  
Supplementary Figure S7: Representative BR-KG evidence and workflow paths. Example paths illustrate the kinds of traversals available during an episode. Literature peaks can be linked through coordinates, brain regions, task contrasts, and cognitive constructs; saved maps and NeuroVault assets can be linked to provenance and reuse constraints; datasets can be linked to derivative readiness, phenotype availability, and compatible tool or workflow contracts. These paths show how BR-KG turns text-level mentions into typed objects that can be queried, checked, and passed to the registry or review layer. Scientific acceptance occurs later, after execution evidence and review are available.

```yaml
asset_readiness: block
missing_assets:
- fold_manifest
warnings:
- confound_specification_incomplete
commitment_verdict: block
```

Resolver verdicts are reported as pass, warn, block, inaccessible, or incomplete. Pass allows planning to proceed. Warn allows planning but requires caveats or sensitivity actions. Block prevents commitment. Inaccessible and incomplete identify access or missingness states that must be resolved or reported as non-executable outcomes.

## S4.2. BIDS, OpenNeuro, and derivative discovery

BIDS and OpenNeuro discovery [14, 20, 24] proceeds as a concrete resolution chain:

<table><tr><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1>Resolution action</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Resolve the dataset identifier, alias, or catalogue entry.</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Check local BIDS root, derivative root, access class, license, and governance notes.</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>If direct file listing is unavailable, query BIDS entities and dataset metadata.</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Parse OpenNeuro GLM specification files when present, including task names, con-trasts, condition weights, and contrast-to-concept mappings.</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Discover workflow-specific assets such as derivatives, phenotype columns, targetvariables, masks, parcellations, matrices, and fold manifests.</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>If resources are remote-only, mark download-required or non-executable accordingto policy.</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>Return dataset-level and asset-level readiness separately.</td></tr></table>

Manual contrast-to-concept mappings link common GLM contrasts to cognitive concepts with confidence annotations. These mappings support evidence assembly and ONVOC normalization but remain separate from review acceptance.

## S5. Typed tool registry and hierarchical action space

## S5.1. Inventory levels and registry objects

The Brain Researcher action space is larger than the MCP surface. MCP exposes controlled operations, while the registry behind MCP stores executable components, workflow specifications, parameter schemas, compatibility clauses, backend recipes, and family cards. The checked inventory distinguishes four levels.

<table><tr><td rowspan=1 colspan=1>Level</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Object type</td><td rowspan=1 colspan=1>Example role</td></tr><tr><td rowspan=1 colspan=1>MCP surface</td><td rowspan=1 colspan=1>87 operations</td><td rowspan=1 colspan=1>Public typed endpointsand structured lookups.</td><td rowspan=1 colspan=1>Search tools, resolvedatasets, inspect arti-facts, request review.</td></tr><tr><td rowspan=1 colspan=1>Exposed workflowlayer</td><td rowspan=1 colspan=1>177 exposed-plus-workflow specifications</td><td rowspan=1 colspan=1>MCP-facing and work-flow specs.</td><td rowspan=1 colspan=1>Plan and package apredictive-modelingrecipe.</td></tr><tr><td rowspan=1 colspan=1>Broader neuroimag-ing registry</td><td rowspan=1 colspan=1>2,089 registry specifica-tions</td><td rowspan=1 colspan=1>Compatibility, parameter,backend, and resourcerecords.</td><td rowspan=1 colspan=1>Validate that a tool ac-cepts the resolved inputand produces expectedoutputs.</td></tr><tr><td rowspan=1 colspan=1>Executable-wrapperinventory</td><td rowspan=1 colspan=1>approx. 196 Python-native tools + approx.2,100 command-linewrappers</td><td rowspan=1 colspan=1>Callable tools, com-mands, modules, orscripts.</td><td rowspan=1 colspan=1>Run AFNI, ANTs, C3D,dcm2niix, FastSurfer,FreeSurfer, FSL, Greedy,MRtrix, MRtrix3Tissue,NiftyReg, or Workbenchcommands through anapproved backend [4, 6,12, 15, 16, 29].</td></tr></table>

Throughout this supplement, MCP operations refer to endpoints exposed by the MCP server. Executable components refer to Python tools, command-line wrappers, Neurodesk modules, container commands, or scheduler scripts. Registry specifications refer to machine-readable records used for discovery, compatibility checking, parameter validation, and recipe generation.

## S5.2. Intent extraction and resource categories

Intent extraction maps a user request into planning fields: analysis intent, modality, expected output type, required resource types, and candidate method family.

Example intent record:

```yaml
user_request: "Can we predict cognition from resting-state connectivity?"
intent: prediction
modality: fMRI_connectivity
required_resources:
- connectivity_matrix
- target_vector
- covariate_table
- fold_manifest
expected_outputs:
- predictions
- scorecard
- model_metadata
candidate_family: predictive_modeling
```

The system defines approximately sixty resource categories, including BIDS roots, 3D volumes, 4D time series, masks, event files, covariate tables, matrices, connectivity matrices, coordinates, statistical maps, QC reports, and rendered figures. A candidate tool is compatible only if it accepts the resolved input resource types, produces the expected artifacts, satisfies parameter schemas, is reachable through an allowed backend, and passes applicable tool-contract and method-condition constraints.

## S5.3. Family-card retrieval and typed compatibility ranking

Tool retrieval uses a hierarchical action space. The planner first maps the request to intent, modality, required resource type, output type, and possible method family. It then retrieves precomputed family cards. Fifteen family cards are available in the checked registry, each storing a summary, typical use cases, member tools, supported inputs and outputs, common constraints, backend availability, and an embedding vector.

Example family-card excerpt:

```yaml
family_card: predictive_modeling
inputs:
- feature_matrix
- target_vector
- fold_manifest
outputs:
- prediction_scorecard
- trained_model_or_coefficients
- heldout_predictions
common_constraints:
- leakage_check
- grouped_or_nested_cv
- permutation_or_null_control
backends:
- local_python
- slurm
```

Expanded candidates are ranked by typed-contract compatibility, input/output resource match, backend requirements, runtime availability, policy scope, method-condition constraints, and task fit.

<table><tr><td rowspan=1 colspan=1>Ranking factor</td><td rowspan=1 colspan=1>Example</td></tr><tr><td rowspan=1 colspan=1>Input compatibility</td><td rowspan=1 colspan=1>Candidate accepts a connectivity or feature matrix.</td></tr><tr><td rowspan=1 colspan=1>Output compatibility</td><td rowspan=1 colspan=1>Candidate produces predictions and scorecard artifacts</td></tr><tr><td rowspan=1 colspan=1>Backend availability</td><td rowspan=1 colspan=1>Local Python or Slurm route is available under current policy.</td></tr><tr><td rowspan=1 colspan=1>Constraint compatibility</td><td rowspan=1 colspan=1>Candidate supports grouped CV or nested CV.</td></tr><tr><td rowspan=1 colspan=1>Policy scope</td><td rowspan=1 colspan=1>Candidate is allowed under the active allowlist mode.</td></tr></table>

## S5.4. Canonical identity, Neurodesk mapping, and NiWrap descriptors

Canonical tool IDs are consumed by the planner, execution recipes, runtime registry, provenance records, and review rules. Backend mapping translates a canonical ID into a local Python entry point, Neurodesk module name, container command, shell command, or Slurm script. This prevents the same method from being represented diferently across planning, execution, provenance, and review.

Example backend mapping:

```yaml
canonical_tool_id: fsl_randomise
registry_spec: fsl_randomise_contract
backend: Neurodesk
module: fsl/\allowbreak{}6.0
command_template: randomise -i {input_4d} -o {output_prefix} -d {design_mat} -t {design_con}
expected_outputs:
- corrected_statistical_map
- command_log
```

NiWrap-generated machine-readable descriptors serve as metadata and compatibility records. They support discovery, parameter schemas, and compatibility checks, but execution is routed through Neurodesk recipes or other backend recipes rather than through NiWrap’s own execution path.

## S5.5. Allowlists, cross-stage context, and operational cache

Tool allowlists support researcher-facing and diagnostic modes. Curated interactive-safe mode is used for routine researcher-facing sessions and excludes unsafe or irrelevant actions. Diagnostic mode broadens access for routing analysis, benchmark evaluation, registry testing, and controlled internal checks. The active allowlist determines which tools can be retrieved and is recorded in the run card.

Cross-stage context propagates structured constraints between planning rounds. Example:

```yaml
cross_stage_context:
controversial_choice: GSR
required_sensitivity:
- rerun_with_GSR
- rerun_without_GSR
claim_caveat_required: true
reviewer_attention: controversial_choice
```

Operational caches reduce repeated BR-KG and registry lookups during long episodes. These caches are separate from reviewed memory: cached retrieval facts can speed up planning, but they are not accepted claims and do not replace run-bundle provenance.

## S6. Constraint taxonomy, compiler, and commitment gate

## S6.1. Hard and soft constraints

Hard constraints are validity boundaries. If a hard constraint is violated, the analysis is not accepted as valid. Soft constraints are consequential methodological choices that may be defensible but require documentation, sensitivity analysis, qualification, or caveats.

<table><tr><td rowspan=1 colspan=1>Constraint</td><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>Example consequence</td></tr><tr><td rowspan=1 colspan=1>Missing fold manifest for predictive model-ing</td><td rowspan=1 colspan=1>Hard</td><td rowspan=1 colspan=1>Block commitment or execution</td></tr><tr><td rowspan=1 colspan=1>Feature selection, standardization, or harmo-nization outside CV [31]</td><td rowspan=1 colspan=1>Hard</td><td rowspan=1 colspan=1>Revise pipeline before acceptance.</td></tr><tr><td rowspan=1 colspan=1>Invalid exchangeability for permutationtesting [32, 33]</td><td rowspan=1 colspan=1>Hard</td><td rowspan=1 colspan=1>Block inference claim.</td></tr><tr><td rowspan=1 colspan=1>GSR without sensitivity analysis [5, 21, 23]</td><td rowspan=1 colspan=1>Soft</td><td rowspan=1 colspan=1>Warn; require with/without-GSRreporting or caveat.</td></tr><tr><td rowspan=1 colspan=1>Dynamic-FC window choice without robust-ness check</td><td rowspan=1 colspan=1>Soft</td><td rowspan=1 colspan=1>Warn; require window-length sensi-tivity or null comparison.</td></tr><tr><td rowspan=1 colspan=1>Missing software version or random seed</td><td rowspan=1 colspan=1>Soft</td><td rowspan=1 colspan=1>Warn; require reproducibility comple-tion.</td></tr></table>

BR-KG negative-knowledge records encode known failure conditions even when the software can technically run.

negative\_knowledge\_record: edge: KNOWN\_TO\_FAIL\_WHEN

```textproto
method: predictive_modeling
condition: feature_selection_outside_cv
severity: hard
action: block_or_revise
```

Hard constraints are derived from tool contracts, method-condition records, and community validators. Soft constraints are derived from conditional-knowledge records, community-practice annotations, and tool-contract clauses.

## S6.2. Constraint edge vocabulary and compiler algorithm

Hard constraints are compiled from KNOWN TO FAIL WHEN and VIOLATES ASSUMPTION edges, tool-contract clauses, and community validators. Soft constraints are compiled from HOLDS UNDER, SENSITIVE TO, and DEPENDS ON edges. The same vocabulary is used in planning, preflight, review, and claim calibration.

Compiler algorithm:
<table><tr><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1>Compiler action</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Retrieve method-condition records for each candidate workflow or tool.</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Retrieve relevant tool-contract clauses and community-validator requirements.</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Bind abstract conditions to the current dataset, resource ledger, parameters, andanalysis state.</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Generate executable or inspectable checks.</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Classify each check as hard or soft.</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Store the active constraint set in the run bundle.</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>Feed hard checks to filters or blockers and soft checks to ranking, warnings, sensitiv-ity requirements, or caveat language.</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>Re-run the compiler whenever the candidate tool set, resource ledger, specificationledger, or analysis plan changes.</td></tr></table>

Binding means converting an abstract rule into a concrete field or path. For example, ”fold manifest required” becomes a check against the fold-manifest field in the HCP-YA resource ledger. Example compiler output:

```yaml
active_constraints:
- id: grouped_cv_required
type: hard
bound_field: resource_ledger.fold_manifest
status: missing
verdict: block
- id: gsr_sensitivity_required
type: soft
bound_field: preprocessing.global_signal_regression
status: unresolved
verdict: warn
required_action: run_with_and_without_GSR_or_add_caveat
```

## S6.3. Commitment gate

The commitment gate is the boundary between planning and execution. In interactive mode, the researcher approves, modifies, or rejects the proposed plan. In bounded autonomous mode, the gate is exercised by a critic or supervisor under a predeclared policy. In benchmark pass-through mode,

human approval may be disabled for evaluation-only runs, but the evidence, tool discovery, preflight, recipe, artifact-observation, and review pathway is unchanged.

<table><tr><td rowspan=1 colspan=1>Mode</td><td rowspan=1 colspan=1>Gate authority</td><td rowspan=1 colspan=1>Possible decisions</td></tr><tr><td rowspan=1 colspan=1>Interactive</td><td rowspan=1 colspan=1>Human researcher</td><td rowspan=1 colspan=1>approve / modify / reject.</td></tr><tr><td rowspan=1 colspan=1>Bounded autonomous</td><td rowspan=1 colspan=1>Critic or supervisor policy</td><td rowspan=1 colspan=1>approve / approve with warnings / re-vise / block / terminate.</td></tr></table>

Example commitment-gate record:

commitment\_gate:

episode\_id: ep\_hcp\_predict\_001

authority: human\_researcher

decision: revise

reason: missing\_fold\_manifest

active\_hard\_blocks:

\- grouped\_cv\_required

allowed\_next\_actions:

\- resolve\_fold\_manifest

\- choose\_non\_predictive\_workflow

\- terminate\_as\_non\_executable

## Systematic Multiverse & Robustness Workflow From a canonical task-fMRI finding to a stability map across datasets and analytic choices

![](images/7f3c4bd34171040f579397ec7b9982ee1be99629cd378b2ed8fd507428757fb2.jpg)

Supplementary Figure S8: Systematic multiverse and robustness workflow. A candidate claim is expanded into an admissible specification space through evidence assembly, tool-card constraints, resource checks, and researcher commitment. Each branch is either executed, blocked, or rejected under the active constraint vocabulary, preserving both successful artifacts and nonexecuted alternatives in the run bundle. The resulting landscape is summarized by specification curves, robustness tables, branch-level caveats, and claim-family verdicts. The workflow turns analytic multiplicity into a reviewable robustness boundary with the specification space fixed before claim calibration.

## S7. Execution substrate, preflight, and provenance

# Cross-Modal, Multi-Stage Neuroimaging Execution

One orchestration layer governing heterogeneous neuroimaging tools across modalities, analysis stages, and compute backends

![](images/34353524ad41efaf0f39f4a5b4177038ea02519ad917ab9b98319450b697ed94.jpg)  
Supplementary Figure S9: Cross-modal, multi-stage neuroimaging execution substrate. Brain Researcher routes heterogeneous neuroimaging workflows through a shared execution layer that coordinates modality-specific tools, analysis stages, provenance capture, and backend selection. The figure summarizes the execution surface covered by Supplementary Methods S7: supported modalities, stable tooling and pipeline families, execution backends, and the shared planner-reviewprovenance logic that links a selected workflow to observed artifacts.

## S7.1. Execution backends and backend selection

Brain Researcher supports local Python, Neurodesk, generic containers, Slurm/HPC, Kubernetes, AWS Batch, and other configured execution substrates. Backend selection is based on resource requirements, software availability, queue status, runtime policy, cost, health, and compatibility with the selected recipe. The selected backend and routing rationale are recorded in the run card. Example backend-selection excerpt:

![](images/0428228ae617e035a99dd9a1d2ae650694b6b17db75f5f3d57f5ed5072882e83.jpg)

```yaml
- queue_policy_allows_batch
fallback_backend: Neurodesk_container
```

The scheduler records dependency resolution, retries, state transitions, and emitted events. Detailed scheduler strategies such as eager, lazy, batch execution, checkpoint persistence, and deadlock detection are implementation details recorded in Appendix F.

## S7.2. Operational and scientific preflight

Operational preflight asks: can this run in the current environment? Scientific preflight asks: would this run be valid for the intended claim?

Operational checks include container image or CVMFS availability, module importability, Python entry-point availability, backend health, filesystem access, network policy, allowed roots, authentication state, and timeout budgets. Scientific checks include design-matrix rank, BIDS validity, coordinate-space consistency, resource-type compatibility, sample alignment, statistical prerequisites, fold-manifest integrity, and tool-contract requirements.

Example preflight result:

```yaml
preflight:
operational:
filesystem_access: pass
container_available: pass
backend_health: pass
scientific:
fold_manifest: block
confound_specification: warn
feature_target_alignment: pass
final_verdict: block
```

Hard failures detected during preflight block execution. Soft warnings may allow execution but require caveats, sensitivity checks, or reviewer attention.

## S7.3. Runtime isolation and real-time boundaries

Runtime isolation uses container-level sandboxing, restricted environments, path checks, ephemeral writable layers, and network isolation. Tools requiring broader filesystem access are executed only under a tracked relaxed mode with additional logging. In the evaluated configuration, MCP returns recipes, policy verdicts, and observation tools; the external coding agent, shell, container runtime, Neurodesk environment, or scheduler performs mutation or computation unless gated MCP administrative execution is explicitly enabled.

Real-time paths such as neurofeedback and online two-photon pipelines operate as continuous closed-loop processes outside the standard batch scheduler. They currently receive weaker provenance and post hoc review coverage than batch workflows and are reported as limitations in S12 rather than treated as fully evaluated execution pathways.

## S7.4. Run bundle and authoritative provenance

Each run emits a run bundle (in agent-executed mode its completeness reflects the typed tools the agent invoked, and for a fully external analysis such as NeuroMark it is a post-hoc reconstruction). The run bundle is immutable and authoritative: memory cards, claim cards, and summaries are derived projections and do not replace the full record. If a claim card conflicts with the run bundle, the run bundle is authoritative.

Example run-bundle manifest:

```yaml
run_bundle:
event_trace: events.jsonl
trajectory_document: trajectory.yaml
observation_record: observations.yaml
analysis_bundle: analysis/\allowbreak{}
run_card: run_card.yaml
artifact_manifest: manifest.json
checksums: sha256_manifest.txt
review_cards: review/\allowbreak{}
recovery_events: recovery.jsonl
```

The event trace stores sequence numbers, timestamps, event types, session IDs, and phase markers. The trajectory document records planning decisions, tool invocations, parameter bindings, and constraint verdicts. The observation record stores intermediate results, QC outputs, diagnostics, and environment-observed state. The analysis bundle stores final artifacts, metadata, checksums, and provenance chains.

## S8. Scientific verification layer and claim calibration

The scientific verification layer is the main post-execution acceptance gate in Brain Researcher. Execution can show that a workflow ran, Harbor can show that a bounded task produced required files or statistics, and a language model can explain an output, but none of these events by itself accepts a neuroimaging claim. For the reported collaborator and bounded-autonomous evaluations, S8 defines the layer that consumes the run bundle, applies validity rules, assigns BLOCK or WARN findings, calibrates claim language, records one of six reportable claim states (accepted, qualified, revised, blocked, rejected, or deferred), routes unresolved decisions for human escalation, and determines memory-writeback eligibility. In this paper, “verification” means structured artifactand claim-level challenge against declared scientific rules; it does not mean formal proof that a claim is true.

## S8.1. Verification inputs and validity layers

Scientific verification consumes the run bundle, not the model transcript alone. It checks the selected plan, active constraints, execution trace, artifact manifest, logs, QC outputs, scorecards, candidate claims, and prior evidence. This prevents fluent explanations from substituting for observed artifacts.

<table><tr><td rowspan=1 colspan=1>Validity layer</td><td rowspan=1 colspan=1>Example checks</td></tr><tr><td rowspan=1 colspan=1>Statistical validity</td><td rowspan=1 colspan=1>Design-model match, degrees of freedom, multiple comparisons,exchangeability, inference framework.</td></tr><tr><td rowspan=1 colspan=1>Measurement validity</td><td rowspan=1 colspan=1>Motion, distortion, registration, CIFTI/surface/volume consistency,QC, reliability.</td></tr><tr><td rowspan=1 colspan=1>Construct validity</td><td rowspan=1 colspan=1>Task-to-construct mapping, reverse-inference risk, contrast inter-pretation.</td></tr><tr><td rowspan=1 colspan=1>Generalization validity</td><td rowspan=1 colspan=1>Cross-validation integrity, leakage, sample size, site effects, stimu-lus effects, out-of-sample support.</td></tr><tr><td rowspan=1 colspan=1>Claim validity</td><td rowspan=1 colspan=1>Whether causal, mechanistic, clinical, external-validity, orbiomarker language exceeds evidence.</td></tr></table>

## S8.2. Rule severity, lifecycle, and reason tags

Verification rules are organized by severity, validity layer, reason tags, detection fields, and default action. BLOCK rules indicate high-confidence validity failures requiring revision before acceptance. WARN rules identify risks that may be addressed by sensitivity analysis, supplementary reporting, alternative modeling, or claim revision.

Example review-rule card:

rule\_id: leakage\_standardization\_outside\_cv   
severity: BLOCK   
validity\_layer: generalization   
reason\_tag: leakage   
detection\_field: pipeline\_dag   
trigger: scaler\_fit\_before\_cv\_split   
default\_action: revise   
required\_fix: fit\_scaler\_inside\_each\_training\_fold

Rule lifecycle status is defined in Appendix G, with the full rule registry, implementation-priority queue, calibration case library, review-context schema extensions, and sensitivity templates reported in the extended registry there. The main text distinguishes deterministic implemented checks from candidate or calibration-only rules when it afects the interpretation of a verification verdict: only implemented checks are treated as enforced, whereas deterministic, schema-dependent, NLP/LLM, and calibration-only candidates are reported as protocol or reviewer-facing constraints unless an episode records their enforcement. Reason tags include leakage, circularity, confound, null mismatch, low reliability, claim inflation, prior conflict, and controversial choice. Prior conflict is an audit trigger rather than a veto. Novelty cannot override a hard method error.

## S8.3. BLOCK and WARN rule families

Representative BLOCK families include split/CV/leakage integrity, design-model mismatch, multiplecomparison and inference-framework errors, circular analysis, and hard confound failures. Examples include feature selection outside CV, standardization outside CV, harmonization outside CV, confound regression outside CV, test-set model selection, uncorrected whole-brain inference, invalid permutation exchangeability, brain-map correlation without spatial nulls, same-data ROI definition with efect testing, and demographic confounds omitted from group models.

Representative WARN families include measurement-quality gaps, soft multiple-comparison issues, multivariate/RSA/predictive-modeling risks, reliability and sample-size concerns, claim inflation, controversial choices, prior conflict, and reporting gaps. Examples include missing MRIQC or visual QA [11], multiple ROI tests without correction, RSA model comparison without correction [17], above-chance classification without permutation or confidence intervals, small-sample biomarker claims, reverse inference, correlation-as-prediction, fit-as-mechanism, GSR without sensitivity analysis [21], dynamic FC without null or window sensitivity [1], graph thresholding without multi-threshold checks, missing random seed, missing software versions, and failed BIDS validation [14].

BLOCK returns the episode to revision or termination. WARN can allow execution or acceptance only if the required caveat, sensitivity analysis, or reporting completion is recorded.

S8.4. Verification verdicts and revision routing
<table><tr><td rowspan=1 colspan=1>Verdict</td><td rowspan=1 colspan=1>Meaning</td><td rowspan=1 colspan=1>Next action</td></tr><tr><td rowspan=1 colspan=1>accept</td><td rowspan=1 colspan=1>Artifacts and claims satisfy active constraintsand review rules.</td><td rowspan=1 colspan=1>Eligible for memory write-back.</td></tr><tr><td rowspan=1 colspan=1>accept with qualifica-tions</td><td rowspan=1 colspan=1>Claim is valid only under explicit caveats orcondition tags.</td><td rowspan=1 colspan=1>Write qualified claim card.</td></tr><tr><td rowspan=1 colspan=1>revise</td><td rowspan=1 colspan=1>Result is not yet acceptable but may becomeacceptable with specified evidence or repair.</td><td rowspan=1 colspan=1>Return to planning, execu-tion, or claim editing.</td></tr><tr><td rowspan=1 colspan=1>block</td><td rowspan=1 colspan=1>Hard validity failure prevents acceptance.</td><td rowspan=1 colspan=1>Stop, redesign, or termi-nate as non-accepted.</td></tr><tr><td rowspan=1 colspan=1>reject</td><td rowspan=1 colspan=1>Candidate claim is unsupported or contra-dicted.</td><td rowspan=1 colspan=1>No accepted memorywriteback.</td></tr><tr><td rowspan=1 colspan=1>defer</td><td rowspan=1 colspan=1>Candidate passes an internal gate but fails apredeclared eligibility rule for the next valida-tion tier.</td><td rowspan=1 colspan=1>Hold pending externalreplication or the requiredtier.</td></tr><tr><td rowspan=1 colspan=1>escalate</td><td rowspan=1 colspan=1>Human researcher or domain expert mustresolve the issue.</td><td rowspan=1 colspan=1>Pause or route for expertjudgment.</td></tr></table>

Each evaluated claim in the collaborator and bounded-autonomous cases is reported in exactly one of six states (accepted, qualified, revised, blocked, rejected, deferred; Table 2). These states correspond one-to-one to the accept, accept with qualifications, revise, block, reject, and defer verdicts above; escalate is a routing action that hands the decision to a human rather than a terminal claim state. Prospectively governed episodes apply rules committed before the confirmatory test, whereas post-hoc audits apply explicit adjudication criteria to completed artifacts. In either tier, the state reflects the applicable evidence rule rather than the strength of the headline number alone (S11.2–S11.3). A terminal state is not intended to replace the condition structure of the evidence. When support divides along an identifiable analytic axis, the claim report pairs the state with a support-partition annotation that names the axis, its levels, and support at each level: the state governs claim promotion, while the annotation preserves the scientific pattern.

Table 2: Claim states and their explicit adjudication criteria. Each evaluated claim in the collaborator and bounded-autonomous cases is assigned exactly one state by the review layer. For prospectively governed episodes, thresholds are committed before the confirmatory test; post-hoc audits instead apply documented criteria to completed artifacts. This is the reporting-level projection of the review verdicts in the table above.
<table><tr><td>State</td><td>Decision rule or adjudication criterion</td></tr><tr><td>Accepted</td><td>Holds across the episode-defined majority of admissible analytic specifications, with no single defensible choice reversing it, and passes every check required for the stated claim language.</td></tr><tr><td>Qualified</td><td>Supported only under explicit, named conditions: support is confined to a subset of specifications or holds only with stated caveats; when an identifiable support partition exists, it is retained alongside the state.</td></tr><tr><td>Revised</td><td>The target, measurement, or validation plan is changed in response to a failed gate, and the redesigned successor passes the same rules.</td></tr><tr><td>Blocked</td><td>A resource, design, power, or validity constraint prevents promotion from exploratory to confirmatory, regardless of the point estimate.</td></tr><tr><td>Rejected Deferred</td><td>Fails the applicable null, replication, or support rule.</td></tr><tr><td></td><td>Passes an internal gate but does not yet satisfy eligibility for the next validation tier (for example, while awaiting external replication).</td></tr></table>

Example revision-routing card:

```yaml
review_verdict: revise
triggering_rule: gsr_without_sensitivity
hard_or_soft: soft
required_evidence:
- rerun_with_GSR
- rerun_without_GSR
- report_effect_stability
allowed_claim_language: qualified_only
```

The review context schema supplies optional fields for model specification, multiple-comparison correction, measurement/QC, pipeline DAG steps, leakage detection, ROI provenance, reproducibility, software versions, random seeds, and BIDS validation status. Missing fields cause rules to skip or warn according to rule policy; missing review context fields do not automatically create scientific acceptance.

## S8.5. Claim calibration and memory eligibility

Claim calibration is separate from computation. A statistically valid result cannot be reported as causal, mechanistic, externally validated, clinically predictive, or a biomarker unless the corresponding evidence exists.

<table><tr><td rowspan=1 colspan=1>Computed result</td><td rowspan=1 colspan=1>Overclaim blocked</td><td rowspan=1 colspan=1>Allowed claim language</td></tr><tr><td rowspan=1 colspan=1>Association in one dataset</td><td rowspan=1 colspan=1>Causal mechanism</td><td rowspan=1 colspan=1>Exploratory or dataset-qualifiedassociation.</td></tr><tr><td rowspan=1 colspan=1>Internal prediction signal</td><td rowspan=1 colspan=1>Clinical biomarker</td><td rowspan=1 colspan=1>Internally evaluated predictionresult.</td></tr><tr><td rowspan=1 colspan=1>Robustness across parcellations</td><td rowspan=1 colspan=1>External validity</td><td rowspan=1 colspan=1>Condition-qualified robustness.</td></tr><tr><td rowspan=1 colspan=1>Leakage-controlled grouped CV</td><td rowspan=1 colspan=1>Mechanistic explanation</td><td rowspan=1 colspan=1>Validated internal predictive result,not mechanism.</td></tr></table>

Only accepted or explicitly qualified claims are eligible for memory writeback. Blocked, rejected, or speculative claims remain in the run bundle.

## S8.5.1. The claim record as a concrete object

The auditable record an episode produces is a set of versioned JSON objects, not an informal log. A commitment card (commitment-card-v1) freezes the claim text, scope boundary, allowed alternatives, same-null constraint, and success/failure criteria, and stores a content hash (commitment hash) computed before execution; re-deriving the hash later detects any post-hoc change to the committed plan. A claim card (claim-card-v1) records the final bounded status of the claim, the checks it survived and failed, the evidence it draws on, and the evidence still required, and points back to the commitment card. Both are written to disk (commitment card.json, claim card.json) alongside an append-only ledger and the run bundle, so the record is exportable and can be opened by another reader after the session without rerunning the analysis. The representative card below comes from an earlier HCP-YA aggregate-prediction campaign and is retained as an illustration of the record schema; it is not the result record for the current HCP iterative episode in Supplementary Methods S11.3.1. Its commitment hash was a version-control soft anchor committed alongside its validator rather than a third-party time-stamped pre-registration.

"schema\_version": "claim-card-v1",   
"claim\_id": "hcp\_rsfc\_aggregate\_predictor",   
"claim\_text": "A frozen ridge predictor on resting-state FC predicts the   
five-component behavioral target in HCP-YA (fold-mean r = 0.190).",   
"status": "accepted",   
"scope\_boundary": "HCP-YA; Schaefer-100x7 FC; no-GSR; 10-fold CV;   
accepted with search correction, not covariate-robust;   
internal validation only",   
"commitment\_card\_ref": "commit:hcp\_rsfc\_aggregate\_predictor",   
"commitment\_hash": "b1e4...e7a9",   
"evidence\_bundle\_refs": ["eb\_hcp\_fc\_features", "eb\_liu\_components"],   
"survived\_checks": [   
"family\_aware\_permutation\_null (Family\_ID block, 1000 perms, plus-one p)",   
"max\_over\_pipelines\_null (38 replayable candidates)"   
],   
"failed\_checks": [   
"covariate\_retention\_gate (>=70% of unadjusted effect retained after   
demographic + IQ adjustment; aggregate retained 55%, r 0.190 -> 0.105,   
below the 0.133 floor) -> bounds scope to search-corrected, not covariate-robust"   
],   
"falsification\_budget\_spent": {"null\_models": 2, "permutations": 1000},   
"next\_required\_evidence": ["external replication (HCP-Aging)"],   
"status\_not\_ground\_truth": true

This is the object the main text refers to as the auditable claim record; the review card (Appendix G) and the memory card (Appendix H) are its review-time and memory-time counterparts. A real exported example built on public Neurosynth coordinate evidence (a working-memory claim, final status weakened under the conservative evidence profile) is released with the code, so a reader can open an actual claim card.json and its evidence verdicts.json directly rather than relying on the listing above. This legacy artifact predates the six-state reporting vocabulary used for the collaborator and bounded-autonomous evaluations; we retain its literal weakened label and do not retroactively relabel the exported record. For the schizophrenia NeuroMark episode, by contrast, no sealed pre-analysis commitment card was created: the collaborator’s three hypotheses were prespecified in their own study protocol rather than sealed as a Brain Researcher commitment card before execution, so the audit bundle we provide for that case is a post-hoc reconstruction assembled from the completed multiverse artifacts. Its claim card accordingly records commitment card ref: null and pre run commitment card found: false rather than asserting a pre-run seal, and is labeled post-hoc by design rather than backfilled. The sealed-commitment-card mechanism is therefore demonstrated on the released Neurosynth example; the NeuroMark episode exercises the multiverse, claim-state, and directionality-audit machinery.

## S9. Memory system, conflict detection, and BR-KG promotion

## S9.1. Memory objects and writeback gate

The memory system stores reviewed knowledge derived from completed episodes. It is not a transcript archive and not a replacement for the run bundle. Its purpose is to retrieve condition-tagged claims, prior tactics, unresolved caveats, review outcomes, and relations among findings.

# From Auditable Research Runs to Cumulative Field Memory Turning each analysis run into an auditable research object, then accumulating claims, conflicts, tactics, and provenance

![](images/585318eec5f8a19f77d8aad51d75d21b5c750522bcd49404769398f56030f9f7.jpg)  
Supplementary Figure S10: From auditable run bundles to cumulative field memory. Completed episodes first produce run bundles containing the committed plan, executed actions, logs, artifacts, validation outcomes, review notes, failures, and caveats. The review layer determines which candidate claims are accepted or explicitly qualified for memory writeback; rejected, blocked, or failed candidates remain recoverable in the run bundle without becoming accepted graph facts. Eligible claims become condition-tagged claim cards with provenance pointers and relation events such as support, contradiction, refinement, conditioning, or supersession. Curated memory cards may later be promoted to BR-KG under a separate governance policy.

<table><tr><td rowspan=1 colspan=1>Memory object</td><td rowspan=1 colspan=1>Purpose</td></tr><tr><td rowspan=1 colspan=1>Claim card</td><td rowspan=1 colspan=1>Stores reviewed claim text, polarity, scope, conditions, evidencelevel, caveats, and provenance pointers.</td></tr><tr><td rowspan=1 colspan=1>Episodic run card</td><td rowspan=1 colspan=1>Stores the question, plan, tool sequence, successful tactics, failedtactics, resource blockers, and resume hints.</td></tr><tr><td rowspan=1 colspan=1>Review card</td><td rowspan=1 colspan=1>Stores review level, verdict, risk tags, issues, fixes, and final eligibil-ity status.</td></tr><tr><td rowspan=1 colspan=1>Claim-relation event</td><td rowspan=1 colspan=1>Stores support, contradiction, refinement, conditioning, or superses-sion relations between claims.</td></tr></table>

Example claim card:

```yaml
claim_card:
claim: "Model performance remained above chance under grouped CV."
polarity: support
scope: HCP-YA sample only
evidence_level: L3_internal_validation
review_status: accepted_with_qualifications
caveats:
- no_external_dataset
provenance_pointer: run_bundle/\allowbreak{}ep_hcp_predict_001
```

Memory writeback is gated by review. Failed, blocked, rejected, or speculative candidate claims remain searchable through the run bundle but are not promoted as accepted memory claims.

## S9.2. Conflict detection and relation events

At writeback, the system encodes the new claim, retrieves similar prior claims above a configured threshold, and submits new/prior pairs to a critic for relation classification. The critic classifies apparent support, contradiction, refinement, conditioning, or supersession for retrieval and audit; it does not resolve scientific truth by itself. For contradictory or conditioning pairs, the system creates bidirectional claim-relation events and records the conditioning variable when available.

Example relation event:

```yaml
relation_event:
new_claim: claim_042
prior_claim: claim_017
relation: conditioning
conditioning_variable: preprocessing_pipeline
confidence: medium
logged_in: run_bundle/\allowbreak{}ep_hcp_predict_001
```

Memory is append-only. Claims may be superseded, conditioned, or contradicted, but they are not silently deleted.

## S9.3. Memory partitioning and BR-KG promotion

During benchmark evaluation, memory namespaces are partitioned so evaluated tasks cannot retrieve target answers from prior benchmark or collaborator episodes. Training, development, benchmark, collaborator, and autonomous-campaign namespaces are separated. Any shared memory is explicitly declared in the evaluation card.

BR-KG promotion is a separate curation path:

<table><tr><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1>Curation action</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Reviewed claim card is created.</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Candidate graph update is proposed.</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Schema validation is performed.</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Source and provenance checks are performed.</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Curation approval is recorded.</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Accepted BR-KG record is created.</td></tr></table>

This preserves the distinction between episode memory, candidate graph updates, and accepted BR-KG records.

## S10. Bounded research episodes and validation contracts

Throughout, the terms “bounded autonomous” and “self-evolving” denote the same narrow process: validation-gated agentic search in which the system proposes, tests, and revises candidate claims strictly within a researcher-declared action space, budget, and validation ladder. They do not denote autonomous scientific self-improvement, open-ended goal setting, or any relaxation of the predeclared gates.

## S10.1. Harbor task format and Brain Researcher mapping

Harbor provides a containerized task contract with an instruction file, environment definition, hidden verifier, metadata, and resource limits. In this work, Harbor bounds the rollout environment and returns task-level completion evidence. It is not the scientific reviewer.

For Brain Researcher benchmark rollouts, Harbor can realize a bounded episode as a task contract. The contract specifies the scientific objective, allowed resources, admissible action space, expected artifacts, execution budget, verifier logic, reward or partial-credit outputs, review rules, memory policy, and escalation conditions. The HCP and TRIBE research lines reported in S11.3 instead used prospective episode bundles and frozen successor contracts outside Harbor. BR-MCP remains the domain-control plane for tool discovery, dataset resolution, planning, recipe generation, policy checks, run observation, grounding, and scientific review.

Example Harbor verifier output:

```yaml
harbor_verifier:
artifact_manifest_present: true
required_statistic_present: true
schema_valid: true
command_completed: true
trajectory_requirements_met: partial
reward: 0.8
```

This verifier output can show that a task produced the required files and statistics while the Brain Researcher scientific verification layer still returns revise or block because of leakage, missing sensitivity analysis, invalid nulls, or claim inflation.

## In-silico Experiment Design and Iterative Hypothesis Generation

![](images/7e25388e7f63bb6479073c9087a5df4e2f6b6bdac213da3a9807b78a55fdc67f.jpg)  
Supplementary Figure S11: In-silico experiment design and bounded hypothesis iteration. The bounded workflow starts from a declared research context, evidence state, allowed action vocabulary, budget, validation ladder, expected artifacts, and stopping rule. Candidate in-silico experiments are proposed, materialized, run through lower-resource validation tiers, and either rejected, revised, deferred, or advanced to stronger tests. Reward or search scores are kept separate from scientific acceptance, which requires the predeclared nulls, artifact checks, and review rules to pass. The loop returns reviewable follow-up hypotheses or memory candidates; acceptance remains with verification.

<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Checks</td><td rowspan=1 colspan=1>Does not check</td></tr><tr><td rowspan=1 colspan=1>Harbor verifier</td><td rowspan=1 colspan=1>Files, schema, required statistic, command/test success, trajectory requirements.</td><td rowspan=1 colspan=1>Scientific validity or claimscope.</td></tr><tr><td rowspan=1 colspan=1>BR-MCP</td><td rowspan=1 colspan=1>Tool/dataset resolution, policy constraints,recipe packaging, provenance capture.</td><td rowspan=1 colspan=1>Final biological truth.</td></tr><tr><td rowspan=1 colspan=1>Execution backend</td><td rowspan=1 colspan=1>Logs, artifacts, metrics, environment-observed state.</td><td rowspan=1 colspan=1>Claim acceptability.</td></tr><tr><td rowspan=1 colspan=1>Scientific verificationlayer</td><td rowspan=1 colspan=1>Leakage, confounds, null controls, robust-ness, caveats, claim calibration.</td><td rowspan=1 colspan=1>Independent replication orcomplete scientific truth.</td></tr><tr><td rowspan=1 colspan=1>Memory gate</td><td rowspan=1 colspan=1>Accepted or qualified writeback.</td><td rowspan=1 colspan=1>Unreviewed or rejectedclaims.</td></tr></table>

## S10.2. Supervisor, critic, and bounded iteration

The bounded-loop architecture can use a supervisor–critic pair under fixed budgets and stopping criteria. The supervisor proposes the next analytic branch from the current episode state: open caveats, failed checks, memory conflicts, validation-ladder progress, and remaining compute/time budget. Candidate branches may vary a confound model, test an alternative parcellation, extend a sensitivity analysis, run a null-control analysis, investigate an anomaly, broaden the search, or abandon a target that cannot meet the contract. This subsection describes the available architecture, not an assertion that every reported episode used this exact controller.

Example branch-decision record:

```yaml
branch_decision:
branch_type: sensitivity_analysis
reason: unresolved_GSR_warning
proposed_action: rerun_with_and_without_GSR
expected_resolution: qualify_or_clear_controversial_choice_warning
budget_cost: one_additional_run
```

After Harbor verification and review-layer inspection, the critic asks whether the previous cycle resolved a review finding, exposed a hard failure, improved validation level, revealed a method dependency, or merely repeated known work. It then returns a structured decision.

```yaml
critic_decision:
decision: continue
reason: sensitivity_analysis_resolves_open_warning
remaining_budget_cycles: 2
```

Episodes terminate when the goal is reached, the validation ladder is satisfied, a hard review block cannot be resolved, progress stalls for the configured number of cycles, compute or wall-clock budgets are exhausted, the critic requests human escalation, or infrastructure recovery fails.

```yaml
termination:
status: terminated_budget_exhausted
unresolved_findings:
- external_validation_missing
memory_writeback: qualified_claim_only
```

## S10.3. Bounded autonomous episode contract

A bounded autonomous episode is evaluable only if it declares its target question, design space, budgets, stopping criteria, validation ladder, expected artifacts, review rules, memory-writeback policy, and escalation conditions. Each cycle must produce a branch-decision record, Harbor verification record, artifact manifest, review card, and claim-status update.

Example contract excerpt:

```yaml
bounded_episode_contract:
target_question: >
Evaluate whether HCP-YA features predict the selected
behavioral target.
admissible_design_space:
- predictive_modeling_family
- grouped_cv
- predeclared_covariates
- parcellation_sensitivity
cycle_budget: 3
validation_ladder: L0_to_L3
expected_artifacts:
- prediction_scorecard
- artifact_manifest
- review_card
memory_policy: accepted_or_qualified_only
escalation_conditions:
- unresolved_hard_block
- out_of_design_space_request
```

The contract prevents open-ended optimization from being mistaken for autonomous scientific discovery. Follow-up branches remain inside the admissible design space and must pass the same commitment, execution, Harbor verification, review, and memory-writeback rules as interactive episodes.

## S10.4. Validation ladder and claim strength

Bounded autonomous episodes use a validation ladder to prevent claim escalation from a single successful run.
<table><tr><td rowspan=1 colspan=1>Ladder level</td><td rowspan=1 colspan=1>Purpose</td><td rowspan=1 colspan=1>Example evidence</td><td rowspan=1 colspan=1>Allowed claimstrength</td></tr><tr><td rowspan=1 colspan=1>L0 feasibility</td><td rowspan=1 colspan=1>Required data, tools, andresources exist.</td><td rowspan=1 colspan=1>Dataset/resource resolutionand preflight pass.</td><td rowspan=1 colspan=1>Analysis is exe-cutable.</td></tr><tr><td rowspan=1 colspan=1>L1 minimal signal</td><td rowspan=1 colspan=1>Initial effect or predictionsignal appears.</td><td rowspan=1 colspan=1>First-pass held-out test, per-mutation, or preliminaryassociation.</td><td rowspan=1 colspan=1>Exploratory sig-nal.</td></tr><tr><td rowspan=1 colspan=1>L2 robustness</td><td rowspan=1 colspan=1>Signal survives methodvariation.</td><td rowspan=1 colspan=1>Parcellation, confound,threshold, model, or prepro-cessing sensitivity.</td><td rowspan=1 colspan=1>Condition-qualified result.</td></tr><tr><td rowspan=1 colspan=1>L3 leakage/null control</td><td rowspan=1 colspan=1>Trivial validity failuresare ruled out.</td><td rowspan=1 colspan=1>Grouped CV, nested CV,permutation, spatial null, ornegative control.</td><td rowspan=1 colspan=1>Validated internalresult.</td></tr><tr><td rowspan=1 colspan=1>L4 held-out or externalsupport</td><td rowspan=1 colspan=1>Generalization testedbeyond initial fit.</td><td rowspan=1 colspan=1>Held-out subjects, site,dataset, or independentbenchmark.</td><td rowspan=1 colspan=1>Stronger general-ization claim.</td></tr><tr><td rowspan=1 colspan=1>L5 claim calibration</td><td rowspan=1 colspan=1>Final language con-strained by evidence.</td><td rowspan=1 colspan=1>Review card, caveats, mem-ory relation events, claimgovernor.</td><td rowspan=1 colspan=1>Accepted or quali-fied claim.</td></tr></table>

For example, a first-pass held-out signal reaches L1. If the signal survives parcellation and confound sensitivity, it reaches L2. If grouped CV and permutation controls pass, it reaches L3.

Without external dataset support, the claim cannot exceed L3; it cannot be called externally validated. Without construct or causal evidence, it cannot be called mechanistic.

## S10.5. Structured hypothesis triage and loop closure

Structured hypothesis triage is separate from bounded validation. Candidate questions, representations, estimators, or follow-up tests may be proposed during exploration, but a successor analysis must re-enter the episode machinery with a frozen target, analysis definition, evidence boundary, and stopping rule. Loop closure therefore means that one episode’s result can determine the next bounded question; it does not permit the completed analysis to be rewritten after its outcome is known.

The two iterative episodes instantiated this mechanism in diferent ways. In HCP, a bounded development search supplied a fixed prediction configuration for a matched comparison and crossoutcome follow-up. In TRIBE, an open six-category screen supplied several candidate contrasts; a post-hoc geometry diagnostic and an explicit researcher decision selected speech versus tools, after which the estimand and evaluation rule were frozen before new stimuli were analysed. Complete episode accounting is reported in Supplementary Methods S11.3.

![](images/471c7b27dd49219e4a7afd9064c8100713f78f7c3399cfa0bf8e5861aafd6d54.jpg)  
Supplementary Figure S12: Out-of-distribution hypothesis proposal and triage. Candidate hypotheses are generated near sparse, conflicting, or under-connected regions of the current knowledge space. Triage checks feasibility, novelty, evidentiary support, resource availability, and validation requirements before a candidate can enter an episode. Hypothesis generation can propose a direction, but promotion still requires evidence assembly, execution, review, and memory eligibility.

# From Experimental Results to Follow-up Hypotheses

![](images/45cb48eea69f2c52f9587be12692de8c9a2819f0398e9a3b812f673260ef7cc2.jpg)  
Supplementary Figure S13: Result-to-hypothesis loop closure. Completed outputs, residual uncertainty, failed branches, and condition-dependent findings can seed successor analyses. Every successor must pass the same commitment, execution, review, and memory-writeback path as a new episode.

## S10.6. Freezing successor analyses

A successor analysis separates discovery from the next evaluation by freezing the scientific target, candidate or contrast, estimand, admissible inputs, and decision rule before the successor data are scored. Evidence may alter which successor is chosen, but it cannot alter the analysis already under evaluation. The HCP configuration was fixed and then refit separately for each behavioural target under the same evaluation procedure. The TRIBE speech–tools estimand was fixed before three non-overlapping recurring-source panels and, subsequently, before a score-blind panel drawn from four previously unused collections. Transport failures, excluded cells, and technical failures remain in the episode record rather than disappearing from denominators.

## S10.7. Same-agent sessions without Brain Researcher

We also inspected separately initiated sessions in which the same coding agent received the research goal but did not invoke Brain Researcher. These sessions provide qualitative process context, not a matched causal ablation: prompts, interaction histories, budgets, stopping decisions, and follow-up opportunities were not randomized or held identical. Their terminal states and the limits of this comparison are reported in Supplementary Methods S11.3.3.

## S11. Evaluation protocol

Three classes of verification check. In conventional workflows verification is a check applied after an analysis is complete; in Brain Researcher it is the precondition for a methodological commitment to become infrastructure at all. Some checks admit unambiguous criteria, such as whether the required inputs exist, whether subject sets match, whether information has leaked across folds, or whether a null model is applied in the domain for which it was declared; these establish only that the stated analysis was carried out in a way that can support its claim, not which analysis is best. Other choices have no single correct answer but a set of defensible ones, such as the parcellation, confound model, or threshold, and here the system renders the navigation explicit rather than allowing a single pipeline to conceal it. A third class resists formalization altogether, including what question is worth asking, what should count as evidence, and how a result revises understanding; the system preserves commitments about this class but does not adjudicate them. That boundary is intentional, and everything beyond it remains with the researcher.

## S11.1. Evaluation components

The evaluation protocol has four components.
<table><tr><td rowspan=1 colspan=1>Evaluation component</td><td rowspan=1 colspan=1>What is measured or documented</td></tr><tr><td rowspan=1 colspan=1>BR-KG characterization</td><td rowspan=1 colspan=1>Graph snapshot, schema, source coverage, provenance cover-age, density, orphan rates, property coverage, and exampletraversal paths.</td></tr><tr><td rowspan=1 colspan=1>Benchmark construction</td><td rowspan=1 colspan=1>Controlled task manifests, scoring contracts, with-BR/without-BR protocol, ablations, and evaluation partitions.</td></tr><tr><td rowspan=1 colspan=1>Collaborator cases</td><td rowspan=1 colspan=1>Real neuroimaging questions, dataset access, tool pathway,artifacts, robustness probes, review outcome, caveated con-clusion, and human/system responsibility split.</td></tr><tr><td rowspan=1 colspan=1>Iterative research episode</td><td rowspan=1 colspan=1>Whether evidence from one stage is converted into a frozensuccessor analysis with explicit denominators, stoppingstates, and claim boundaries.</td></tr></table>

## S11.1.1. Benchmark surfaces

The quantitative Results use two benchmark surfaces. The first is a paired tool-calling benchmark: a correct response maps a natural-language neuroimaging request to an existing analysis tool or executable route, with required inputs available before execution. The technical metric family for this surface is routing capability. The second is an evidence-citation benchmark based on NeuroimageKnowledge: a claim is counted as grounded only when its cited evidence can be located and judged supportive for the stated claim. Support was evaluated by three LLM-based judges (Claude Opus 4.8, Codex GPT-5.5, and Gemini 3.1 Pro), each blind to benchmark condition. All reference-bearing rows were sent to all three judges (with-BR, 814 of 1,673 generated rows; without-BR, 469 of 1,525). Each label entered a three-state vote as verified, not verified, or cannot judge, and a row counted as verified when at least two judges voted verified. Every generated evidencebasis row, including rows without a reference and rows without two verified votes, remained in the denominator. Both surfaces compare with-BR and without-BR conditions over runs drawn from seven provider-defined model variants: Claude Code / Claude Opus 4.8, Codex GPT-5.5, Gemini 3.1 Pro, GLM-5.1, DeepSeek-V4-Pro, Kimi K2.5, and Qwen3.6-Plus. The tool-calling surface is a complete seven-model paired matrix; the evidence-citation headline is a descriptive question-level aggregate over all generated evidence-basis rows rather than a paired model-level estimate.

<table><tr><td rowspan=1 colspan=1>Surface</td><td rowspan=1 colspan=1>Primary denominator</td><td rowspan=1 colspan=1>Reported metrics</td></tr><tr><td rowspan=1 colspan=1>Tool calling</td><td rowspan=1 colspan=1>60 task manifests × seven modelsper condition (420 model-itemtrajectories).</td><td rowspan=1 colspan=1>Capability@k, Correctroute/tool@k, and Handoff score@k.Capability@k is required-capabilitycoverage after k non-neutral actions.not a binary route-reached indica-tor.</td></tr><tr><td rowspan=1 colspan=1>Evidence citation</td><td rowspan=1 colspan=1>50 open-ended question sets; allgenerated evidence-basis rows percondition, with only rows receivingat least two verified votes con-tributing to the numerator.</td><td rowspan=1 colspan=1>verified_groundedness_rate,verified_among_claimed_grounded,citation_spam_rate, andanswer_correctness_rate.</td></tr></table>

What counts as verified grounding. For each claim a model supports with a citation, a judge blind to condition checks two things: whether the cited source can be located, and whether it supports the claim as stated. Only a full match is counted as verified. A source that supports a narrower version of the claim, or that is real but about a diferent point, is not counted. Each grounded item receives one label; Table 3 gives the three that occur here with a real example of each. Two further not-verified labels, for references that cannot be resolved or parsed, did not occur among the resolvable items. Because only full support is counted, many citations are not credited: a substantial share point to real sources that support only part of the claim (partial) or a diferent point (no unrelated). Judges also difer in how strictly they treat partial support, so the rate depends on the judge.

Among the 444 non-verified with-BR rows present in all three judge outputs, 289 (65%) received an exact-label majority of no unrelated and 124 (28%) an exact-label majority of partial. The remaining 31 (7%) had no exact-label majority or could not be judged; none had a fabricated or malformed exact-label majority. The two dominant failure modes suggest diferent remedies: of-topic citations call for stronger retrieval filtering and reranking, whereas partial-support citations call for a claim-to-evidence entailment check that constrains a claim to what its source actually states. Residual borderline cases, where the judges themselves disagree, are the ones a human reviewer or the audit layer is best placed to resolve. Among all 814 with-BR rows present in the three judge outputs, 777 (95%) cited a retrieved document, of which 367 (47%) received a verified majority; 3 of 37 (8%) specific citations produced from the model’s own parameters were verified. The three-judge diagnostic therefore points chiefly to retrieval returning an of-topic or partially-supporting source rather than the model fabricating references.

Table 3: Grounding labels, with a representative with-BR item for each. Each model claim is checked, blind to condition, against the source it cites. Only yes (the source fully supports the claim) is counted as verified; partial (supports only a narrower version) and no unrelated (a real source on a diferent topic) are not. Claims use the models’ own wording, lightly shortened; judge rationales are condensed. Further not-verified labels (cannot judge, malformed, fabricated) apply to references that cannot be resolved or parsed.
<table><tr><td>Label</td><td>Model claim (cited to a retrieved source) Judge&#x27;s reason</td><td></td></tr><tr><td>yes (verified)</td><td>Head motion, respiration, arterial  $\mathrm { C O } _ { 2 } ,$  pressure, autoregulation, and vasomotion can and how they alter the BOLD signal; the claim affect BOLD signals and therefore confound is fully supported. resting-state interpretation.</td><td>blood The source explicitly lists each of these factors</td></tr><tr><td>partial (not credited)</td><td>tests before group analysis.</td><td>Head-motion artifacts are the most frequently The source supports the first clause (motion is reported issue in QC, and outliers should be de- the most frequent QC issue) but says nothing tected in image-homogeneity and co-registration about the second (the outlier-detection proce- dure); only a weaker version is grounded.</td></tr><tr><td>no_unrelated (not ited)</td><td>l The spatial-normalization workflow should in- The cited source is about time-series QC (frame- cred- clude visual verification that functional contours wise displacement, WM/CSF signals); it does fields.</td><td>align with anatomy before writing deformation not mention spatial normalization or deforma- tion fields at all.</td></tr></table>

## S11.1.2. Benchmark construction and independence

Because the team that built Brain Researcher also assembled the benchmark, we took explicit steps to limit circularity. Benchmark questions were drafted from standard neuroimaging analysis requests and curated together with a co-author who is not a developer of Brain Researcher; this co-author also evaluated system outputs and ran the quantitative benchmark. For every item, the reference route and required capabilities were fixed before either condition (with-BR or without-BR) was run, so that targets could not be adjusted to favour the system. Scoring is capability-level: any response that covers an item’s required capabilities is credited whether through a Brain Researcher call or an equivalent executable route. For the grounding rate, all reference-bearing rows were scored by the three condition-blind judges. Their labels were reduced to verified, not verified, or cannot judge; a row entered the numerator when at least two judges voted verified, while every generated evidence-basis row remained in the denominator. Thus a single cannot judge label did not block verification when the other two judges voted verified. On the grounding surface, inter-judge reliability is moderate (three-judge Fleiss $\kappa = 0 . 5 0$ with-BR and 0.72 without-BR, binary verified versus not), and is set almost entirely by one lenient judge: Gemini 3.1 Pro and Claude Opus 4.8 agree closely (Cohen’s $\kappa = 0 . 9 2$ , 96% raw agreement with-BR), whereas Codex GPT-5.5 credits full support more readily (per-judge verified rate 0.31 versus 0.21 and 0.22). Because most items are not verified, κ understates agreement under the skewed without-BR base rate (Gwet’s AC1 = 0.68 against $\kappa = 0 . 4 8$ for the Codex–Claude pair; raw agreement 0.80). The condition contrast does not depend on the choice of judge: each judge independently shows a large with-BR gain (with- versus without-BR verified rate 0.21/0.04 for Gemini, 0.31/0.04 for Codex, 0.22/0.08 for Claude). Gemini’s raw run recorded about 30% of grounding items as unjudged because the model emitted its verdict inside an envelope the parser rejected; we recover each such verdict from the model’s own output (241 with-BR and 79 without-BR items) and release the recovery audit with the benchmark package. The reference targets and scoring contracts are released (Appendix J) so that the benchmark can be re-scored independently. We did not commission a fully external benchmark, and we note this as a limitation (Supplementary Methods S12).

## S11.1.3. Human audit of the automated judges

To validate the LLM judges, we drew a reproducible random 20% sample (seed 20260630) of the scored results across the reported benchmarks (272 items, one graded entry each) and adjudicated each by hand against the recorded automated verdict. Agreement was 96% (261 of 272; Cohen’s $\kappa = 0 . 9 4$ , linear-weighted $\kappa = 0 . 9 6$ on the grounding items where all disagreements fall; Table 4). Tool-routing (12 of 12) and NIK answer-key (15 of 15) items were confirmed; the 11 discrepancies fell entirely in the grounding benchmark (234 of 245 confirmed) and were all one-step severity nuances (an item scored strictly that is arguably partial, or the reverse). None reversed a supported call to unsupported or the reverse, and the judges erred strict (Table 5), so the reported grounding gains are conservative rather than inflated. The full graded sheet is reproduced in Appendix M and released with the benchmark package (Appendix J).

Table 4: Human audit of the automated benchmark judges. A reproducible random 20% sample of the scored results across the reported benchmarks, adjudicated by hand against the automated verdicts. Discrepancies were confined to the grounding benchmark and were all one-step severity nuances that never reversed a supported/unsupported call, with the judges erring strict.
<table><tr><td>Benchmark</td><td>Checked (20%)</td><td>Confirmed</td><td>To review</td></tr><tr><td>Tool routing</td><td>12</td><td>12</td><td>0</td></tr><tr><td>NIK answer keys</td><td>15</td><td>15</td><td>0</td></tr><tr><td>Grounding judgments</td><td>245</td><td>234</td><td>11</td></tr><tr><td>Total</td><td>272</td><td>261</td><td>11</td></tr></table>

Metric numerators and denominator rules for main-text Fig. 3 are expanded in Appendix J. The two benchmark surfaces are not pooled because their item types, denominators, scoring rules, and ceilings difer.

## S11.1.4. Efect sizes, uncertainty, and paired significance

For the tool-calling benchmark, the headline numbers are means across the seven model variants, so we treat the model as the unit of analysis $( n = 7 )$ : each model contributes one with-BR and one without-BR score, itself a mean over the 60 tasks, and the contrast is the within-model paired diference. This is the conservative unit because the seven models are not independent draws from a population and their per-task outcomes are clustered; the paired design also controls for task dificulty, since the same items are scored under both conditions. Table 6 reports, for each tool-calling metric, the across-model mean under each condition, the mean paired improvement with a t-based 95% confidence interval across the seven models, a paired t test, and an exact two-sided Wilcoxon signed-rank test. All seven models improved on all three metrics (7/7 positive diferences), which places the signed-rank statistic at its floor for $n = 7 \ ( p = 0 . 0 1 6 )$ . As a pooled descriptive check that does not assume model exchangeability, first-action correct route/tool selection over the 420 model-task trajectories rose from 0.233 (Wilson 95% CI 0.195–0.276) to 0.936 (0.908–0.955). On the evidence-citation surface, the main-text headline (0.046 to 0.220) is a descriptive three-judgemajority rate over all generated evidence-basis rows, with only rows receiving at least two verified votes contributing to the numerator, equal-weighted across the 50 questions rather than inferred from seven paired model scores. The reported question-clustered 95% interval for the with-BR rate (0.168–0.272) is the normal-approximation interval formed as the mean plus or minus 1.96 standard errors across those 50 per-question rates, using the stored analysis’s population-standard-deviation convention. Judge-specific rates are reported in Appendix J, and the underlying artifacts are released with the benchmark package. With n = 7 the tool-calling confidence intervals remain wide and are reported as descriptive bounds on the paired efect, not as population estimates; a fully external, independently constructed benchmark remains the decisive next measurement (Supplementary Methods S12).

Table 5: The 11 flagged grounding items from the human audit. All fell in the grounding benchmark under the with-BR condition. Nine reflect the judge scoring an arguably partial item as fully unrelated (too strict); one (NIK-BP-H-003) scored an arguably partial item as a full yes (too generous); and one (NIK-BP-H-001) is a relabel within non-support (no versus no unrelated). None moves directly between the opposite hard calls (yes and no unrelated), and the net tendency is strict, so with-BR grounding is if anything under-credited.
<table><tr><td>Item</td><td>Judge</td><td>Human</td><td>Issue</td></tr><tr><td>NIK-BP-H-003</td><td>yes</td><td>partial</td><td>Dead-salmon study and multiple-comparison correction ab- sent from the source; support real but incomplete.</td></tr><tr><td>NIK-BP-H-001</td><td>no_unrelated</td><td>no</td><td>Source contradicts the claim, so a negative rather than unrelated.</td></tr><tr><td>NIK-BP-H-005</td><td>no_unrelated</td><td>partial</td><td>Source states the exact “multiplicity of methodologic vari- ants&quot; problem the claim describes.</td></tr><tr><td>NIK-IN-H-004</td><td>no_unrelated</td><td>partial</td><td>GSR-debate context bears on global-signal content.</td></tr><tr><td>NIK-ME-H-004</td><td>no_unrelated</td><td>partial</td><td>Covers subject motion and noise modeling; related but general.</td></tr><tr><td>NIK-NK-E-003</td><td>no_unrelated</td><td>partial</td><td>Measures FWER control with no true signal, i.e. false- positive territory.</td></tr><tr><td>NIK-PP-H-006</td><td>no_unrelated</td><td>partial</td><td>Discusses geometric distortion and its correction; “TOPUP&quot;/susceptibility wording absent.</td></tr><tr><td>NIK-ST-H-007 NIK-ST-H-007</td><td>no_unrelated</td><td>partial</td><td>Context is statistical power and effect-size uncertainty.</td></tr><tr><td></td><td>no_unrelated</td><td>partial</td><td>Same power and significance topic as the claim (second item).</td></tr><tr><td>NIK-ST-M-008 NIK-ST-M-008</td><td>no_unrelated</td><td>partial</td><td>Names permutation-based testing, the claim&#x27;s subject.</td></tr><tr><td></td><td>no_unrelated</td><td>partial</td><td>Context explicitly names permutation-based testing (second item).</td></tr></table>

Table 6: Paired benchmark efect sizes with uncertainty and significance. Unit of analysis is the model (n = 7); each entry is a mean across the seven model variants, and each model’s score is itself a mean over the 60 tool-calling tasks. The improvement is the mean within-model with-BR minus without-BR diference; the 95% confidence interval is t-based across the seven models. Significance is a two-sided paired t test [t(6)] and an exact two-sided Wilcoxon signed-rank test. All seven models improved on every metric.
<table><tr><td>Metric</td><td>Without BR</td><td>With BR</td><td>Mean ∆ [95% CI]</td><td>t(6)</td><td></td><td>Wilcoxon p</td></tr><tr><td>Correct route/tool@1</td><td>0.233</td><td>0.936</td><td>+0.702 [0.576, 0.829]</td><td></td><td>13.6</td><td>0.016</td></tr><tr><td>Capability@1</td><td>0.498</td><td>0.945</td><td>+0.447 [0.282, 0.612]</td><td></td><td>6.6</td><td>0.016</td></tr><tr><td>Handoff score@1</td><td>0.474</td><td>0.761</td><td>+0.287 [0.201, 0.373]</td><td></td><td>8.2</td><td>0.016</td></tr></table>

## S11.1.5. Seven-model routing ablation without direct KG calls

This seven-model routing ablation used the same tasks.60 manifest, frozen labels, prompt template, model-facing route search→tool search surface, and exact-top-1 endpoint for all seven requested model routes. Exact top-1 is a match to one of the task’s frozen acceptable route labels. Across the 420 route–task episodes, 362 matched an acceptable top-1 label (86.2%; requested-route range, 81.7–90.0%). Direct KG-named calls were unavailable in this arm. A protocol violation counted as an error; only an infrastructure failure was recorded as missing.

The provenance column records how a requested route was run, not a diferent evaluation condition; all seven rows, including Kimi and Qwen, use the same no-direct-KG endpoint. This exact-label top-1 metric difers from the historical Correct route/tool@1 metric and is reported separately from the paired with-BR/without-BR benchmark.

## S11.2. Collaborator cases

Collaborator cases are selected when they involve a concrete neuroimaging question, tractable scope, accessible data, and a meaningful analysis or review target. The cohort includes three collaborators across three neuroimaging subfields. Each case reports the collaborator-facing question, subfield, dataset access, tool pathway, execution artifacts, robustness probes, review outcome, caveated conclusion, attribution, and the boundary between system-generated artifacts and human decisions.

Case inclusion requires: a defined scientific question, available or resolvable data, an executable or reviewable analysis path, and a clear endpoint. Exclusion applies when data access is unresolved, the question is outside the supported modality/action space, or governance constraints prevent execution. Collaborator cases may be conducted through standard Claude Code, Codex, or Cursor interfaces depending on collaborator preference; all interfaces route through the same MCP pathway.

Table 7: Seven-model routing ablation without direct KG calls. All seven rows use the same no-direct-KG exact-label top-1 endpoint; run provenance is shown separately.
<table><tr><td>Runner/requested route</td><td>Exact top- Protocol- 1</td><td>compliant</td><td>Gold able</td><td>avail- Run provenance</td></tr><tr><td>Claude Code / claude-opus-4-8</td><td>54/60 (90.0%)</td><td>59/60</td><td>59/60</td><td>v1 tools-only run</td></tr><tr><td>Codex / gpt-5.5</td><td>51/60 (85.0%)</td><td>60/60</td><td>59/60</td><td>v1 tools-only run</td></tr><tr><td>OpenCode / google/gemini-3.1- 54/60 pro-preview</td><td>(90.0%)</td><td>59/60</td><td>59/60</td><td>v1 tools-only run</td></tr><tr><td>OpenCode / opencode-go/glm- 52/60 5.1</td><td>(86.7%)</td><td>60/60</td><td>59/60</td><td>v2 tools-only single arm</td></tr><tr><td>OpenCode / go/deepseek-v4-pro</td><td>opencode- 51/60 (85.0%)</td><td>58/60</td><td>59/60</td><td>v2 tools-only single arm</td></tr><tr><td>OpenCode / opencode/kimi-k2.5 51/60</td><td>(85.0%)</td><td>60/60</td><td>59/60</td><td>v1 tools-only single arm</td></tr><tr><td>OpenCode / opencode/qwen3.6- 49/60 plus</td><td>(81.7%)</td><td>60/60</td><td>59/60</td><td>v1 tools-only single arm</td></tr></table>

These cases were not selected on their outcomes. The inclusion and exclusion criteria above are properties of a case’s tractability—a defined question, resolvable data, an executable analysis path, and a clear endpoint—and were applied before any result was seen, never on whether a claim turned out favorable. This is visible directly in the reported cohort, which was assembled for heterogeneity in evidence structure and consequently spans a range of outcomes rather than a run of successes: the NeuroMark audit returned qualified claims alongside a caught sign-blind scoring failure; the cocaine-use-disorder episode returned a bounded null with its exploratory screen blocked from confirmatory promotion; and the cross-cultural social-cognition episode was blocked as underpowered and ended without a settled claim. Each case reports the states assigned by its applicable prospective gate or post-hoc audit—including the null and blocked outcomes—so the record shows the same accounting whether or not a case produced a publishable positive result. Because the collaborator cohort is small (n = 3) and deliberately heterogeneous, we present these as case studies of how the claim-record machinery behaves across diferent evidence structures (accepted, qualified, null, and blocked), and not as an estimate of a success rate or a benchmark of positive findings.

Execution modes and per-case attribution. Brain Researcher runs in two execution modes that difer in where the analysis runs and therefore in what the audit record guarantees. In engineexecuted mode (Fig. 1), the engine runs the analysis end to end and emits the full audit bundle—a hash-sealed commitment card written before observation, the recorded trajectory, provenance, and observations, and an adjudicated claim card with evidence verdicts—as a structural byproduct, with commit-before-observe and version-pinned execution enforced by the engine. In agent-executed mode, a coding agent (Claude Code, Codex, or Cursor) drives the analysis: it authors and runs the analysis code, invoking Brain Researcher through the MCP for knowledge-graph grounding, execution recipes, typed execution (for example cluster submission), artifact logging, and scientific review. Brain Researcher records the work through its typed tool surface and applies the review layer plus any applicable prospective commitment gate, but the analysis is authored and driven by the agent rather than emitted by the engine, so a full sealed pre-run bundle is not produced as an automatic byproduct and the audit record is only as complete as the tools the agent invoked, carrying no engine-side execution stamp. This mode is also what makes Brain Researcher usable when the data are local or governed and cannot enter the cloud sandbox: in the NeuroMark case the analysis ran entirely in an external workspace with Brain Researcher recording and reviewing it, whereas in the other cases the agent drove the analysis within a Brain Researcher episode, so more of the run passed through its typed tools. Both modes share the same knowledge graph, tool contract, and review layer.

Pre-registration strength consequently falls into three tiers. A sealed pre-run commitment card (i.e. the full engine-executed protocol) is demonstrated on the public Neurosynth working-memory example (Supplementary Methods S8.5.1). The iterative HCP and TRIBE lines (Supplementary Methods S11.3) combine adaptive discovery with later frozen successor contracts; their evidence strength is stated stage by stage rather than summarized as one pre-registration tier. The three collaborator cases carry no sealed Brain Researcher commitment card: their hypotheses were prespecified in the collaborators’ own protocols. For NeuroMark, whose analysis ran externally, the Brain Researcher record is a post-hoc audit of the completed multiverse; the cocaine and crosscultural analyses instead ran within a Brain Researcher episode under prospective gate governance, so their decisions fell to a pre-committed gate rather than to a post-hoc review. Table 8 states, for every reported case, its execution mode, the model that drove it, whether Brain Researcher executed the analysis or only recorded and reviewed it, who wrote the analysis code, and the provenance of its audit files.

Table 8: Per-case attribution: execution mode, driving model, and the human/system responsibility split. Every reported analysis case ran in agent-executed mode; the engine-executed full-bundle protocol is demonstrated on the public Neurosynth example (final row). “BR” denotes Brain Researcher.
<table><tr><td rowspan=1 colspan=1>Case</td><td rowspan=1 colspan=1>Mode</td><td rowspan=1 colspan=1>Drivingmodel</td><td rowspan=1 colspan=1>BR&#x27;s role</td><td rowspan=1 colspan=1>Codeauthor</td><td rowspan=1 colspan=1>Commitment  auditprovenance</td></tr><tr><td rowspan=1 colspan=1>NeuroMarkschizophrenia(Li &amp; Calhoun)</td><td rowspan=1 colspan=1>agent-executed</td><td rowspan=1 colspan=1>CursorClaude Opus4.6 (analysis);Codex (exter-nal review)</td><td rowspan=1 colspan=1>Recorded and reviewedonly; analysis ran in anexternal workspace</td><td rowspan=1 colspan=1>X. Li(agent-assisted)</td><td rowspan=1 colspan=1>No sealed card;post-hoc audit(pre_run_commitment_card.false)</td></tr><tr><td rowspan=1 colspan=1>Cocaine-use-disorder (Ricard &amp;Poldrack)</td><td rowspan=1 colspan=1>agent-executed</td><td rowspan=1 colspan=1>Claude Code(Claude Son-net 4.6)</td><td rowspan=1 colspan=1>Commitment gate, MCP-mediated cluster ex-ecution, and review;bounded null</td><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>Not sealed (collaboratorprotocol); run bundle +review verdict</td></tr><tr><td rowspan=1 colspan=1>Cross-culturalsocial cognition(Wang)</td><td rowspan=1 colspan=1>agent-executed</td><td rowspan=1 colspan=1>Claude Code</td><td rowspan=1 colspan=1>Reviewed within theepisode; gate-blocked asunderpowered</td><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>Not sealed; gate-blocked(run bundle + reviewverdict)</td></tr><tr><td rowspan=1 colspan=1>HCP workflowsearch and trans-fer</td><td rowspan=1 colspan=1>agent-executed</td><td rowspan=1 colspan=1>Claude CodeCodex</td><td rowspan=1 colspan=1>Bounded search account-ing, frozen successoranalyses, run-bundlereview, and claim-boundary enforcement</td><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>Prospective episode bun-dles plus frozen successorcontracts; retrospectivesame-cohort inference</td></tr><tr><td rowspan=1 colspan=1>TRIBE speech-tools geometry</td><td rowspan=1 colspan=1>agent-executed</td><td rowspan=1 colspan=1>Claude CodeCodex</td><td rowspan=1 colspan=1>Open contrast discovery,researcher-gated succes-sor choice, frozen panelevaluation, and reward-blind closeout</td><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>Discovery record plusfrozen recurring-sourceand new-source successorcontracts</td></tr><tr><td rowspan=1 colspan=1>Neurosynthworking-memory(public reference)</td><td rowspan=1 colspan=1>engine-executed</td><td rowspan=1 colspan=1>None (de-terministicengine)</td><td rowspan=1 colspan=1>Executed the sealedepisode end to end</td><td rowspan=1 colspan=1>BR gener-ator</td><td rowspan=1 colspan=1>Sealed pre-run com-mitment card (hash4871ea43...)</td></tr></table>

## S11.2.1. Collaborator case results

The three collaborator episodes summarized in the main text are reported here in full; the complete automatically generated per-case reports are reproduced in Appendix L. Each evaluated claim is assigned one of the six review states defined in Supplementary Methods S8.4 (Table 2). The cocaineuse-disorder and cross-cultural episodes were governed by prospective gates, whereas NeuroMark was adjudicated by an explicit post-hoc audit of completed artifacts; in both tiers, the state reflects the applicable evidence criteria rather than the strength of the headline number alone.

NeuroMark: a multiverse separates accepted from qualified claims. A collaborator working on schizophrenia functional network connectivity (FNC) derived from the NeuroMark framework [8] brought three prespecified hypotheses to Brain Researcher for robustness audit: (NM-H1) latent connectivity factors outperform individual FNC edges for patient-versus-control classification; (NM-H2) between-domain connections show larger group diferences than withindomain ones; and (NM-H3) latent factors fitted to edge space concentrate loading mass on betweendomain edges. The episode used data from the FBIRN cohort $( N = 3 6 3 ;$ 181 controls, 182 patients) parcellated through NeuroMark 2.2 (a template-based independent-component-analysis framework for FNC estimation) into 5,460 FNC edges per subject. A conventional single-pipeline analysis had supported NM-H2, and the relevant question was whether the hypotheses would survive plausible analytic variation (Fig. 4). The analysis code, specification ledger, and per-hypothesis outputs (including the generated run report and figures) for this case are openly available at https://github.com/XinhuiLi/BR-NeuroMark.

The analysis was expanded into a 480-specification multiverse crossing four connectivity estimators, three confound strategies, five dimensionality-reduction methods, four classifiers, and two domain granularities $( 4 \times 3 \times 5 \times 4 \times 2 )$ in the collaborator’s workspace, and Brain Researcher recorded and reviewed the run, which partitioned the hypotheses into distinct claim states. NM-H1 (latent superiority) was qualified: across the 384 latent-evaluable specifications (those that include a dimensionality-reduction step, so a latent representation exists to compare against edges), edges outperformed latent factors in aggregate (median $\Delta \mathrm { A U C } = - 0 . 0 3 2 )$ and only 72 (18.8%) favored latent features, with gains concentrated under ICA, ComBat harmonization, and partial-correlation connectivity. NM-H3 (loading-mass concentration) was qualified and weak overall: 100 of 384 latent specifications (26.0%) favored between-domain loading mass under the corrected direction-aware rule (median Wilcoxon $p = 0 . 0 5 4 )$ , with factor analysis (45.8%) and PCA (33.3%) more favorable than ICA (12.5%) and NMF (12.5%) (Fig. 4A–C).

The NM-H2 step also exposed a failure mode that the claim record is built to make visible. A server-side fault caused Brain Researcher to fall back to a general-purpose coding agent, which scored a specification as favorable whenever the permutation $p < 0 . 0 5$ , regardless of the sign of the efect. Because NM-H2 is directional (between-domain > within-domain), specifications in which within-domain diferences were larger were still counted as favorable, so the unaudited multiverse inflated apparent NM-H2 support to near-universal acceptance. The mismatch was not caught automatically by the review layer; a human reviewer identified it by inspecting the code, the output files, and the specification curve directly. Binding the hypothesis statement to the executed statistic and its acceptance rule is what let the sign-blind acceptance criterion, once found, become a standing check rather than a one-of correction. The fallback to a general-purpose agent was the cause but was not itself flagged in the record. Re-running the audit with a sign-aware criterion (favorable when one-sided $p < 0 . 0 5$ and $\Delta \mathrm { m e a n } | d | > 0 )$ over the 24 unique connectivity–confound– domain contrasts that actually vary the H2 estimand (after collapsing classifier and reduction duplicates) changed the result: NM-H2 was qualified, with 12 of 24 (50.0%) unique H2 contrasts favorable (median $\Delta \mathrm { m e a n } | d | = 0 . 0 0 7 9 )$ . The pooled fraction records the adjudication but is not the informative scientific result: support partitioned completely by connectivity estimator, with Pearson and Spearman at 100% favorable and partial correlation and mutual information at 0%. NM-H2 is therefore estimator-regime–dependent rather than generally robust. The partition does not identify its mechanism because partial correlation and mutual information alter the dependence measure in non-equivalent ways; distinguishing shared covariance from estimator scale, power, or nonlinearity requires targeted follow-up analysis. The qualified state records promotion status, while the estimator support partition records where the evidence divides. This case motivates two checks now added to the review layer: a directionality test that the statistic and its acceptance rule agree with the sign asserted by the hypothesis, and a provenance warning whenever execution falls back from a Brain Researcher operation to a general-purpose agent. The unit of analysis shifted from a completed workflow to an auditable decision space with explicit, condition-tagged claim boundaries; the episode record tied each hypothesis to the analytic regime under which it holds, with review and memory cards in Appendices G–H.

Cocaine-use-disorder. The cocaine-use-disorder episode asked whether reported connectivity– behavior associations in a substance-use cohort remained credible under plausible choices of atlas, motion threshold, and confound strategy. This was a robustness question rather than a search for a new biomarker: the useful output was whether the apparent associations survived a defensible multiverse and, if not, what follow-up design would be needed. $\mathrm { O n }$ SUDMEX CONN [2] (OpenNeuro ds003346 v1.1.3 [13]; $N = 1 3 8 )$ , the analysis was expanded into a 36-specification multiverse over atlases, framewise-displacement thresholds, and confound strategies. Brain Researcher rejected all five prespecified network-systemic-segregation associations under SDMA-GLS, a procedure that combines same-dataset estimates across specifications [18] (all $Z < 1 . 2 4$ , FDR $q > 0 . 5 8 )$ . An exploratory screen over all 70 network-by-outcome combinations likewise yielded no FDR-surviving efects (max $Z _ { \mathrm { G L S } } = 2 . 1 8 ; 0 / 7 0$ surviving). Brain Researcher blocked the exploratory screen from confirmatory promotion and converted the null result into a replication plan (Fig. 4D,E).

Cross-cultural social cognition. The cross-cultural episode asked whether a small coordinatebased literature could support a mechanistic claim about culture-specific social-cognition topography, or whether the evidence was too sparse and compositionally imbalanced for that interpretation. The agent ran cell-wise activation likelihood estimation [9, 10] on a corpus of 21 published studies (85 MNI peak coordinates, four culture-by-relationship cells) using NiMARE [25]. Some papers contributed more than one eligible contrast, so the cell counts are cell-level study entries; the unique-paper count remains 21. The agent produced a mechanistic mPFC-topology interpretation (centroid shift 19.5 mm in strangers versus 3.8 mm in close others between Euro-American and East Asian pools), which Brain Researcher’s review layer blocked as exploratory: cells had only $k = 6 { - } 8$ entries, below the recommended $k \geq 1 7 ~ [ 1 0 ]$ ; paradigm composition was imbalanced across cells (the East Asian close-other cell was dominated by self-referential trait-judgment paradigms); and centroid shifts from aggregate coordinates do not establish non-overlapping activation distributions. The case ended with a paradigm-matched follow-up design and no settled claim (Fig. 4F).

Across the three episodes, Brain Researcher’s contribution was structural and case-general: structured tool specifications made each multiverse mechanical to enumerate and execute; the rule checker kept defensible alternatives inside the same review pass; BR-KG-derived design considerations entered plans as explicit analysis choices; and the episode record stored claims with the conditions under which they held. Fragile, null, or underpowered findings became bounded claims or follow-up designs rather than headline biomarkers.

## S11.3. Iterative research episode evaluation

This section gives the complete numerical record underlying the two iterative episodes summarized in the main text. The unit of evaluation is the research trajectory: how an exploratory result was converted into a frozen successor analysis, what the successor returned, and where the trajectory stopped. The HCP and TRIBE episodes used diferent data and estimands and are therefore reported separately rather than reduced to a common success score.

## S11.3.1. HCP workflow search and frozen selected workflow follow-up

Search accounting and frozen selected workflow. The HCP episode began from an HCP S1200 search cohort of 326 participants and locally reconstructed counterparts to the five behavioural components of Liu et al. [19]. Two staged search bundles allocated 20 and 96 candidate-evaluation slots, respectively, for a total denominator of 116. All 20 slots in the first bundle and 84 of 96 in the second returned scored results in their parent runs (104/116). The best result in the first bundle had mean cross-validated $r = . 3 7 3$ and $R ^ { 2 } = - . 0 3 6 \mathrm { : }$ the best scored result in the expanded bundle had $r = . 4 8 7 , R ^ { 2 } = . 2 1 2$ , and $\mathrm { M A E } = . 6 9 0$ The remaining 12 slots ended in controller transport exhausted; consequently, the second parent episode was recorded as COMPLETED WITH PROTOCOL FAILURE, with episode valid=false. A separate recovery bundle later computed all 12 missing slots, but did not rewrite or retroactively validate the failed parent episode. The 116 allocated slots, 104 parent-run scored results, 12 transport failures, and separate recovery are therefore retained explicitly rather than collapsed into one clean search denominator.

The development procedure designated the term-116 whole-band coherence-magnitude representation (cohmag multitaper mean fs-1 fmin-0 fmax-0-5) with cosine-kernel ridge regression and $\alpha = 1$ as the frozen selected workflow. The frozen selected workflow was chosen during development, not automatically identified as a global champion: the search artifact records automatic champion selected=false. It was frozen before the matched comparison described below.

Cognition comparison. The frozen selected workflow was compared with a locally matched reconstruction of the Liu-style nested prediction procedure, not a paper-exact reproduction or a test of general superiority over Liu et al. The comparison used 244 participants from 243 families and 10 repeated, overlapping, family-grouped $5 \times 3$ nested-cross-validation splits. Within each split, performance was computed from pooled out-of-fold predictions. The frozen selected workflow exceeded the matched comparator in all 10 splits for correlation and $R ^ { 2 }$ , and had lower mean absolute error in all 10. Its median values were $r = . 3 3 2 , R ^ { 2 } = . 1 0 7$ , and $\mathrm { M A E } = . 7 6 8$ , compared with $r = . 2 3 5 , R ^ { 2 } = . 0 0 9$ , and $\mathrm { M A E } = . 8 1 6$ for the matched procedure. Median paired diferences were $\Delta r = . 0 9 8 , \Delta R ^ { 2 } = . 0 9 9$ , and $\Delta \mathrm { M A E } = - . 0 5 1 $ ; a family-cluster pointwise bootstrap gave a 95% interval of $[ . 0 1 1 , . 1 7 7 ]$ for $\Delta r$ , and the conditional one-sided plus-one test gave $p = . 0 0 6$

These inferential quantities are conditional sensitivity analyses. The frozen selected workflow was selected after Cognition search on the same cohort; the comparison was retrospective and not adjusted for the preceding search; repeated splits overlap and are not independent replications; and target residualization was performed outside the cross-validation folds, so the complete procedure is not strictly leakage-free. A predeclared calibration-repair decision aid also failed: its median held-out calibration slope was within the required range (.945 within [.8, 1.2]), but median $\Delta R ^ { 2 } = - . 0 0 4$ rather than $\ge ~ . 0 2 , ~ \Delta R ^ { 2 }$ was positive in $4 / 1 0$ rather than at least $8 / 1 0$ repeats, and median $\Delta \mathrm { M A E } = + . 0 0 4$ rather than $\leq 0$ . The artifact therefore records scientific acceptance=false.

Frozen selected workflow transfer. The frozen selected workflow was then refit separately to four additional Liu component outcomes under the same repeated-split evaluation, without changing its representation, estimator family, or hyperparameter. Table 9 reports all five outcome summaries. The $\Delta r$ column is the median paired diference within repeats and therefore need not equal the diference between the two displayed marginal medians.

Table 9: HCP frozen selected workflow results across five behavioural outcomes. Values are medians over 10 overlapping repeated splits. Matched denotes the locally matched Liu-style nested procedure; wins count repeats in which the frozen selected workflow had the larger pooled out-of-fold correlation. These repeats measure same-cohort stability, not independent replication.
<table><tr><td>Outcome</td><td>Frozen selected workflow r</td><td>Matched r</td><td>Frozen selected workflow</td><td> $\overline { { R ^ { 2 } } }$  Matched  $\overline { { R ^ { 2 } } }$ </td><td>Median ∆r</td><td>r wins</td></tr><tr><td>Cognition</td><td>.332</td><td>.235</td><td>.107</td><td>.009</td><td>.098</td><td>10/10</td></tr><tr><td>Tobacco Use</td><td>.242</td><td>.126</td><td>.057</td><td>−.029</td><td>.100</td><td>10/10</td></tr><tr><td>Personality-Emotion</td><td>.075</td><td>.006</td><td>-.019</td><td>-.058</td><td>.068</td><td>9/10</td></tr><tr><td>Illicit Drug Use</td><td>.122</td><td>.008</td><td>-.003</td><td>-.044</td><td>.117</td><td>10/10</td></tr><tr><td>Mental Health</td><td>.001</td><td>-.062</td><td>−.047</td><td>-.046</td><td>.069</td><td>8/10</td></tr></table>

Across the four additional outcomes, the frozen selected workflow had the larger correlation in $3 7 / 4 0$ repeated comparisons; including Cognition gave $4 7 / 5 0$ . The directional comparison was broader than absolute predictive utility: median $R ^ { 2 }$ was positive only for Cognition and Tobacco Use. For the four transfer outcomes, the raw conditional $p$ values for $\Delta r$ were .0004, .0556, .0091, and .1292 for Tobacco Use, Personality–Emotion, Illicit Drug Use, and Mental Health, respectively. The corresponding weak-familywise-error–corrected values were .3338, .6692, .2136, and .6580; none passed correction, and all simultaneous intervals crossed zero. In the wider multiplicity analysis over 20 outcome-by-configuration cells, 16 had positive median $\Delta r .$ , two had pointwise intervals excluding zero, and none had a simultaneous or weak-FWER-supported efect. Thus the cross-outcome result supports a recurring relative direction under the frozen selected workflow, not a multiplicity-corrected claim of general transfer.

Separate internal holdout and terminal claim state. A later internal holdout used a diferent, separately frozen workflow on 81 participants and returned r = .232 $( p = . 0 1 8 4 )$ ， $R ^ { 2 } = - . 0 7 5$ and $\mathrm { M A E } = . 8 6 9$ . It had no matched comparator and negative $R ^ { 2 }$ , so it does not confirm the frozen selected workflow versus matched-procedure result. No external dataset or independent replication was completed. The strongest supported statement is therefore that, conditional on the same-cohort development path, the frozen selected workflow showed a stable relative advantage over the matched procedure across repeated splits, with positive absolute utility concentrated in Cognition and Tobacco Use. It is not an externally confirmed predictive model.

## S11.3.2. TRIBE speech–tools discovery and frozen stimulus follow-up

Open six-category discovery. The TRIBE episode began with 48 sounds drawn from four source collections and balanced across six categories: animals, music, nature, speech, tools, and voice. For each of the 15 category pairs, a four-source-fold decoder compared early representations (encoder layers 0, 2, and 4) with later representations (layers 10, 12, and 14). Table 10 reports the complete descriptive ranking by $\Delta \mathrm { A U C } = \mathrm { A U C } _ { \mathrm { l a t e } } - \mathrm { A U C } _ { \mathrm { e a r l y } }$ . This was a full technical rerun of the same ordered 48-item panel previously exposed in the terminal run, not an independent new-data replication, and the ranking carried no inferential $p$ value.

Exploratory selection and freezing. The rule-selected top pair, tools–voice, did not yield a coherent cross-collection geometry: later-layer separation was lower in only two of four source folds, its late-layer contrast axis was not aligned with the frozen reference direction, and the diagnostic label was mixed or unresolved. Speech–tools, ranked third, showed non-negative early signed projection and lower late-layer separation in all four source folds and was labelled aligned magnitude reduction. This was a post-hoc exploratory diagnostic, not the original screening winner. A researcher explicitly adopted speech–tools in a recorded decision (scientist reward order), after which the contrast, geometry estimand, and analysis rule were frozen; confirmation and scientific acceptance remained unauthorized.

Table 10: Complete TRIBE six-category discovery ranking. Negative ∆AUC denotes weaker category decoding in later than early layers. Values are descriptive results from the same ordered 48-item technical-rerun panel.
<table><tr><td>Rank</td><td>Category pair</td><td>Early AUC</td><td>Late AUC ∆AUC</td></tr><tr><td>1</td><td>tools-voice</td><td>.625</td><td>.250 -.375</td></tr><tr><td>2</td><td>music-speech</td><td>.917</td><td>.563 -.354</td></tr><tr><td>3</td><td>speech-tools</td><td>.979</td><td>.646 -.333</td></tr><tr><td>4</td><td>animal-speech</td><td>.958</td><td>.792 -.167</td></tr><tr><td>5</td><td>speech-voice</td><td>.917</td><td>.750 -.167</td></tr><tr><td>6</td><td>animal-music</td><td>.271</td><td>.417 +.146</td></tr><tr><td>7</td><td>music-voice</td><td>.542</td><td>.417 -.125</td></tr><tr><td>8</td><td>animal-tools</td><td>.479</td><td>.583 +.104</td></tr><tr><td>9</td><td>nature-speech</td><td>1.000</td><td>.917 -.083</td></tr><tr><td>10</td><td>music-tools</td><td>.625</td><td>.688 +.063</td></tr><tr><td>11</td><td>nature-voice</td><td>.875</td><td>.813 -.063</td></tr><tr><td>12</td><td>music-nature</td><td>.917</td><td>.875 -.042</td></tr><tr><td>13</td><td>nature-tools</td><td>.750</td><td>.708 -.042</td></tr><tr><td>14</td><td>animal-nature</td><td>.854</td><td>.833 -.021</td></tr><tr><td>15</td><td>animal-voice</td><td>.479</td><td>.479 .000</td></tr></table>

For the successor panels, $\Delta _ { \mathrm { r e f } }$ and $\Delta _ { \mathrm { e v a l } }$ were the speech-minus-tools centroid vectors in the reference and evaluation panels, and D was the reference panel’s root-mean-square within-category residual dispersion. Normalized separation was $S = \| \Delta _ { \mathrm { e v a l } } \| / D$ , orientation was $C = \cos ( \Delta _ { \mathrm { r e f } } , \Delta _ { \mathrm { e v a l } } )$ signed projection was $G = C S$ , and the primary change was $\Delta S = \overline { { S } } _ { \mathrm { l a t e } } - \overline { { S } } _ { \mathrm { e a r l y } }$ , predicted to be negative. These are components of one geometric decomposition, not three independent pieces of evidence. The frozen decision also required positive early and late orientation in at least three of four sources and an early reference-decoding AUC above .5.

Three non-overlapping recurring-source panels. Three successor panels each contained 48 previously unused items, with six speech and six tool sounds from each of the same four collections (AudioSet Strong, BBC Sound Efects, FreeSound, and SoundBible). Item identifiers and source paths had zero overlap across panels. Table 11 reports all 12 collection-by-panel values. All three aggregate $\Delta S$ values were negative and each panel met the frozen bounded support rule; 11/12 collection-level values were negative. These panels had no prespecified conventional confidence interval or p value, and the four recurring collections are stability units rather than independent draws from a population of sources.

An exploratory leave-one-item-out diagnostic on R5 clarified the exception. BBC, FreeSound, and SoundBible retained negative $\Delta S$ under every item deletion. AudioSet Strong had $\Delta S = + . 0 2 8 4$ a leave-one-out range of $[ - . 3 5 3 6 , + . 3 3 2 4 ]$ , and six sign changes across 12 deletions. The recurringsource result is therefore best described as a mostly direction-preserving contraction of speech–tools separation with an explicit unstable counterexample, not as universal compression, loss of category

Table 11: TRIBE recurring-source successor panels. Each cell is late-minus-early normalized speech– tools separation (∆S); negative values indicate contraction in later layers.
<table><tr><td>Panel</td><td>Aggregate</td><td>AudioSet Strong</td><td>BBC</td><td>FreeSound</td><td>SoundBible</td><td>Outcome</td></tr><tr><td>R3</td><td>-.6937</td><td>-.7664</td><td>-.8625</td><td>-.7177</td><td>-.4284</td><td>bounded support</td></tr><tr><td>R4</td><td>-.4467</td><td>-.5433</td><td>-.3985</td><td>-.2749</td><td>-.5701</td><td>bounded support</td></tr><tr><td>R5</td><td>-.5471</td><td>+.0284</td><td>-.4885</td><td>-1.0059</td><td>-.7225</td><td>bounded support</td></tr></table>

information, semantic abstraction, or a neural mechanism.

Frozen four-new-source endpoint. The final extension used a score-blind minimax acousticbalancing procedure to select 48 sounds from four collections not used earlier: DCASE 2013 Ofice Live, SINGA:PURA, SONYC-UST, and STARSS23. Each source contributed six speech and six tool sounds; the maximum observed absolute standardized acoustic mean diference was .383, below the frozen .5 bound. Table 12 reports the source-level primary results.

Table 12: TRIBE frozen new-source endpoint. Negative values are in the predicted direction.
<table><tr><td>Collection</td><td>∆S</td><td>∆AUC</td></tr><tr><td>DCASE Office Live</td><td>-.5883</td><td>-.0741</td></tr><tr><td>SINGA:PURA</td><td>-.2073</td><td>-.0278</td></tr><tr><td>SONYC-UST</td><td>+.0856</td><td>+.0926</td></tr><tr><td>STARSS23</td><td>-.0820</td><td>-.0463</td></tr><tr><td>Aggregate</td><td>-.1980</td><td></td></tr></table>

Three of four sources were directionally concordant, but the frozen H1 balanced-label permutation test (99,999 permutations) gave raw $p = . 1 3 2 1 2$ and Holm-adjusted $p = . 3 9 6 3 6$ at α = .025, so H1 was not supported. The three predeclared secondary families were also unsupported: H2 raw/adjusted p = .35200/.39636, H3 .16147/.39636, and H5 .05441/.21764. The inference pertains only to label permutations within this fixed four-source panel; it does not license population-level inference over sound collections. Reward-blind review closed the endpoint as inconclusive or conflicting, with confirmation authorized=false and scientific acceptance=false. The complete TRIBE trajectory therefore contains both results: three recurring-source panels showed repeated, mostly direction-preserving contraction, but the first frozen four-new-source inferential endpoint did not support the primary hypothesis after multiplicity correction.

## S11.3.3. Separately initiated same-agent sessions and evidence boundaries

The sessions without Brain Researcher were run separately by the same coding agent and are reported as qualitative process observations, not as matched randomized ablations. The prompts, interaction histories, budgets, human interventions, stopping rules, and opportunities for follow-up were not held identical.

In HCP, the separately initiated session completed a substantial analysis and ended in a valid internally held-out null: adjusted partial r = .141 $( p = . 2 6 1 , 9 5 \% \mathrm { C I } [ - . 1 8 2 , . 4 0 9 ] )$ , predictive $r = . 2 0 0$ , incremental $R ^ { 2 } = . 0 3 3$ , and overall $R ^ { 2 } = - . 0 1 3$ . It did not launch a frozen successor. In TRIBE, summary/PCA representations gave positive discovery and second-subset shifts $( \Delta \rho = . 1 1 2$ 95% CI [.026, .198], familywise $\mathrm { Q A P } \ p = . 0 1 6 3 ;$ and $\Delta \rho = . 1 1 5$ , 95% CI [.019, .225], $p = . 0 0 0 2 4 )$ but the frozen primary spectrotemporal representation did not reproduce them $( \Delta \rho = - . 0 0 3 6 , 9 5 \%$ $\mathrm { C I } \ [ - . 0 4 6 , . 0 4 0 ] , p = . 7 7 8 3 $ and $\Delta \rho = - . 0 1 6 2 , 9 5 \% \mathrm { C I } [ - . 0 6 3 , . 0 3 2 ] , p = . 2 4 8 4 )$ . The prospective firewall and frozen-representation lock were not satisfied, and the authoritative terminal state was

TECHNICAL FAILURE, not a confirmed positive or a scientific null. That session also did not launch a successor.

Across both with-Brain-Researcher episodes, repeated splits or recurring collections are stability checks rather than independent replications. HCP remained retrospective, same-cohort, and searchconditional, with no external confirmation. TRIBE’s recurring-source pattern did not survive its first multiplicity-corrected new-source endpoint. Accordingly, self-evolving denotes a governed trajectory in which evidence changes the next frozen analysis; it does not denote autonomous model improvement or scientific confirmation.

## S11.3.4. Review-layer calibration

To attach a number to how often the review layer’s triage errs, we scored its severity classification, blind, against the 60-case calibration library (C01–C60; Appendix G, G9.5), whose cases carry an expected verdict for common neuroimaging analysis situations (16 block, 39 warn, 5 allow). The layer’s classification arm produced no false accepts (0 of 16 invalid analyses were waved through as valid; 0%, rule-of-three 95% upper bound 19%; and 0 of the 55 cases warranting any flag were classed allow) and no false blocks (0 of 5 valid controls were blocked, although five controls provide only a loose upper bound). Exact three-way agreement with the expected verdict was 88% (53/60); the seven residual disagreements were all block-versus-warn severity diferences. The library was assembled after the directionality check introduced in response to NM-H2 and shares provenance with the review policy. These values are therefore an internal-consistency ceiling over short canonical scenarios, not an independent field error rate or evidence that the pre-patch review layer would have caught NM-H2.

## S11.4. Resource usage and user experience

We report token usage, derived cost, and interaction profile for one collaborator episode (a cocaineuse-disorder resting-state case) as an illustrative reference. Transcripts were not captured for the other collaborator cases, so per-case token accounting is shown for this case only. This episode was run on Claude Sonnet 4.6 and comprised 2,350 assistant turns and 1,347 tool calls (902 Bash, 89 Edit, 64 Read, and 53 Write, plus Model-Context-Protocol calls that included 45 cluster (Slurm) submissions and 41 deep-research queries), spread over roughly six calendar days of interactive use rather than continuous compute.

Tokens are the primary reported quantity, and we present the components separately (Table 13). The episode consumed about 0.89M output tokens, 19.9M cache-creation (cache-write) input tokens, 180.2M cache-read input tokens, and under 0.03M non-cached input tokens. We do not headline the roughly 201M raw token sum: it is dominated by cache reads, an artifact of long-context caching across 2,350 turns, where the persisted episode context is re-read on each turn and billed at roughly one-tenth the standard input rate. The meaningful generation is therefore about 0.89M output tokens plus about 20M cache-creation tokens.

From these token counts we derive an approximate cost at standard Claude Sonnet 4.6 API list prices (input \$3, output \$15, cache write \$3.75, and cache read \$0.30 per 1M tokens): about \$13 for output, about \$75 for cache-creation, about \$54 for cache-read, and under \$1 for uncached input, for a total of approximately \$140 for this single episode. Cost is driven mainly by cache-creation and output rather than by the large cache-read volume. This is an approximate API-list-equivalent figure: actual billing depends on the user’s plan and tier, and subscription plans difer from per-token API pricing.

<table><tr><td rowspan=1 colspan=1>Token component</td><td rowspan=1 colspan=1>Tokens</td><td rowspan=1 colspan=1>Approx. cost</td></tr><tr><td rowspan=1 colspan=1>Output (generation)</td><td rowspan=1 colspan=1>0.89M</td><td rowspan=1 colspan=1>$13</td></tr><tr><td rowspan=1 colspan=1>Cache-creation (cache write)</td><td rowspan=1 colspan=1>19.9M</td><td rowspan=1 colspan=1>$75</td></tr><tr><td rowspan=1 colspan=1>Cache-read</td><td rowspan=1 colspan=1>180.2M</td><td rowspan=1 colspan=1>$54</td></tr><tr><td rowspan=1 colspan=1>Uncached input</td><td rowspan=1 colspan=1>0.03M</td><td rowspan=1 colspan=1>&lt;$1</td></tr><tr><td rowspan=1 colspan=1>Total (single episode)</td><td rowspan=1 colspan=1>∼201M (raw sum)</td><td rowspan=1 colspan=1>~$140</td></tr></table>

Table 13: Token usage and approximate API-list-equivalent cost for one collaborator episode (cocaine-use-disorder resting-state case, Claude Sonnet 4.6). Components are reported separately; the raw sum is dominated by cache reads (billed at roughly one-tenth the input rate) and is not the headline quantity. Costs are derived from the per-component token counts at standard list prices (input \$3, output \$15, cache write \$3.75, cache read \$0.30 per 1M tokens) and are approximate; actual billing depends on the user’s plan and tier.

Usage scales with how much the user reviews generated code rather than only reading it, and it varies by model and tier. In practice the binding constraint is high token throughput rather than marginal cost: throughput alone can exhaust a standard plan, and one collaborator case accordingly required a higher-tier plan.

In this episode, the user supplied fMRIPrep-preprocessed SUDMEX resting-state data and asked, exploratorily, how to analyse individual functional networks given a limited cross-sectional sample. Brain Researcher resolved this request into a typed episode (dataset and target functional networks) and committed a 36-specification multiverse at the commitment gate; the agent executed it on a cluster backend through Brain Researcher’s submission tools, and Brain Researcher reviewed the resulting run bundle and returned a bounded null claim. In a post-hoc self-assessment, the collaborator on this case estimated that Brain Researcher saved on the order of three months of analysis time relative to assembling and running the equivalent multiverse by hand. The division of responsibility between the researcher and the system across this flow follows the human-agent authority boundary (Fig. S2).

## S11.5. Analysis-level methods provenance (COBIDAS-style)

Brain Researcher did not acquire MRI data or run raw-image preprocessing for any reported case. It operates at the analysis and statistical-inference level, on already-preprocessed derivatives (functionalconnectivity matrices, fMRIPrep outputs, minimally-preprocessed connectomes) or on published coordinates and reference maps. Acquisition and image-preprocessing provenance is therefore inherited from the cited source datasets and is not re-derived here; the acquisition/preprocessing column of Table 14 records the origin of each derivative rather than a pipeline Brain Researcher executed. What the system records for every case is the available analysis-level provenance: the estimator, statistical model, inference and multiple-comparison procedure, robustness or multiverse design, applicable prospective gates or post-hoc review criteria, and the resulting claim state. These fields are written to the case’s run or audit bundle (S7.4), including available software identity and versions, random seeds, and the specification ledger. Table 14 summarizes this analysis-relevant subset of the COBIDAS reporting standard [22] across the five reported episodes; per-case reports (Appendix L) and the specification ledgers (Appendix F) carry the complete records.

Table 14: Analysis-level methods provenance across the five reported episodes (COBIDAS-style). Brain Researcher performs no raw acquisition or image preprocessing; the acquisition/preprocessing column records the origin of the derivatives or coordinates each episode consumed. All remaining columns describe analysis-level choices that Brain Researcher recorded and reviewed, with prospective commitment gates applied where applicable (S7.4); the compute itself was executed by the agent, except in the engine-executed Neurosynth reference. Full per-case records are in Appendices F and L.
<table><tr><td rowspan=1 colspan=1>Episode(data, N)</td><td rowspan=1 colspan=1>Acquisition/preprocessingorigin</td><td rowspan=1 colspan=1>Features &amp; estima-tor</td><td rowspan=1 colspan=1>Statistical model, in-ference &amp; correction</td><td rowspan=1 colspan=1>Robustnessreview crite-ria</td><td rowspan=1 colspan=1>Claimstate(s)</td></tr><tr><td rowspan=1 colspan=1>NeuroMark/ schizophre-nia FNC(FBIRN;N = 363,181 control182 patient)</td><td rowspan=1 colspan=1>NeuroMark 2.2template-ICAFNC derivatives(framework/collaboprovided) [8]</td><td rowspan=1 colspan=1>5,460 FNC edges,two domain gran-ularities; four con-rt@tivity estimators(Pearson, Spearman,partial correlation,mutual information);five dimensionality-reduction methods;four classifiers</td><td rowspan=1 colspan=1>Patient-vs-control classifi-cation (AUC) and group-difference contrasts;one-sided permutationtest with sign-awareacceptance (p &lt; 0.05and ∆ mean|d| &gt; 0);Wilcoxon test for loadingmass</td><td rowspan=1 colspan=1>480-specificationmultiverse(4×3×5×4×2);384 latent-evaluable; 24unique sign-aware H2 con-trasts</td><td rowspan=1 colspan=1>NM-H1 qual-ified; NM-H2qualified;NM-H3 quali-fied</td></tr><tr><td rowspan=1 colspan=1>Cocaine-use-disorderconnectiv-ity (SUD-MEX CONN,OpenNeurods003346v1.1.3;N = 138)</td><td rowspan=1 colspan=1>fMRIPrepderivatives,collaborator-supplied [2]</td><td rowspan=1 colspan=1>Network segregation;atlas varied acrossspecifications; same-data meta-analysis(SDMA-GLS) [18]</td><td rowspan=1 colspan=1>Connectivity-behaviorassociations; FDRcorrection (confirma-tory all Z &lt; 1.24,q &gt; 0.58; exploratorymax ZGLs = 2.18, 0/70)</td><td rowspan=1 colspan=1>36-specificationmultiverse(atlas ×framewise-displacementthreshold ×confound); 70-combinationexploratoryscreen</td><td rowspan=1 colspan=1>5 confirma-tory rejected;exploratoryscreenblocked</td></tr><tr><td rowspan=1 colspan=1>Cross-cultural so-cial cognition(coordinateliterature; 21studies, 85peaks)</td><td rowspan=1 colspan=1>Published MNIpeak coordinates(no raw imagingdata)</td><td rowspan=1 colspan=1>Cell-wise activation-likelihood estimation(ALE), NiMARE [25]</td><td rowspan=1 colspan=1>ALE with per-cell k = 6-8 (below recommendedk ≥ 17 [10]); centroidshift reported as descrip-tive only</td><td rowspan=1 colspan=1>Paradigm-compositionimbalancecheck acrossfour culture-by-relationshipcells</td><td rowspan=1 colspan=1>mPFC-topologyinterpretationblocked asexploratory</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Episode(data, N)</td><td rowspan=1 colspan=4>Acquisition /preprocessingorigin</td><td></td><td rowspan=1 colspan=5>Features &amp; estima-tor</td><td rowspan=1 colspan=8>Statistical model, in-ference &amp; correction</td><td rowspan=1 colspan=1>Robustnessreview crite-ria</td><td rowspan=1 colspan=1>Claimstate(s)</td></tr><tr><td rowspan=7 colspan=1>HCP-YA be-haviouralprediction(search cohortN = 326;matchedcomparisonN = 244,</td><td rowspan=7 colspan=4>HCP S1200minimally-preprocessedconnectomesand locally re-constructedLiu-componentcounterparts</td><td></td><td rowspan=3 colspan=5>116 allocated can-</td><td rowspan=3 colspan=8>Ten repeated family-grouped 5 × 3 nested-CVsplits with pooled out-</td><td rowspan=10 colspan=1>104/116parent-runscores plus12 recordedtransport fail-ures and sepa-rate recovery;frozen selected</td><td rowspan=13 colspan=1>The frozenselected work-flow exceededthe matchedcompara-tor in 10/10Cognitionsplits; 47/50directionalwins acrossfive out-comes, butno transfercell passedweak-FWER;separate hold-out did notconfirm; noexternal ac-ceptance</td></tr><tr><td></td><td rowspan=2 colspan=4>frozen selected</td><td rowspan=2 colspan=2>ations </td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=3>frozen seled</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=5>workflow using acoherence-magnituderepresentation with</td><td rowspan=1 colspan=3>workflow us</td><td rowspan=1 colspan=3>using a</td><td rowspan=1 colspan=1>of-f</td><td rowspan=1 colspan=2>old metrics; family-</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>cluster</td><td rowspan=2 colspan=6>p and conditional</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>rap</td></tr><tr><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=5>cosine-kernel ridgeregression; locally</td><td rowspan=1 colspan=8>FWER transfer analysis</td><td rowspan=1 colspan=5>e-sided test; weak-</td></tr><tr><td rowspan=2 colspan=1>243 families;</td><td rowspan=6 colspan=4>[19, 30]</td><td rowspan=2 colspan=3>[19, 3</td><td rowspan=2 colspan=6>matched Liu-style</td><td rowspan=6 colspan=5></td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>workflow refits</td></tr><tr><td rowspan=4 colspan=1>internal hold-out N = 81)</td><td></td><td rowspan=1 colspan=5>nested comparator</td><td></td><td></td><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>across out-</td></tr><tr><td></td><td rowspan=3 colspan=5></td><td></td><td></td><td></td><td rowspan=1 colspan=1>comes; retro-spective same-cohort, search-unadjusted</td></tr><tr><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1>sensitivity</td></tr><tr><td></td><td></td><td></td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=6 colspan=1>TRIBEspeech-toolsrepresenta-tion geometry(TRIBE v2;discovery andsuccessor pan-els of naturalsounds)</td><td rowspan=6 colspan=4>No subject-levelBOLD; model-representationanalysis over sixsound categories,four recurringcollections, andfour previouslyunused collec-tions [7]</td><td></td><td rowspan=2 colspan=5>Early-versus-lateencoder decoding</td><td rowspan=1 colspan=8>Descriptive 15-pair</td><td rowspan=1 colspan=1>Researcher-</td><td rowspan=6 colspan=1>Recurring-sourceboundedsupport in3/3 panels(11/12 cells),followed byunsupportednew-sourceH1 (Holmp = .396);terminaloutcome in-conclusive orconflicting;no scientificacceptance</td></tr><tr><td></td><td rowspan=1 colspan=8>discovery; three non-</td><td rowspan=2 colspan=1>authorizedpost-hocchoice of</td></tr><tr><td></td><td rowspan=4 colspan=5>and frozen geomet-ric decomposition ofnormalized separa-tion, orientation, andsigned projection</td><td rowspan=4 colspan=8>overlapping recurring-source panels; 99,999balanced-label permu-tations and Holm cor-rection at the frozenfour-new-source endpoint</td></tr><tr><td></td><td rowspan=1 colspan=1>speech-tools;</td><td rowspan=1 colspan=1>3</td></tr><tr><td></td><td rowspan=1 colspan=1>frozen esti-</td><td rowspan=1 colspan=1>(11/</td></tr><tr><td></td><td rowspan=1 colspan=1>mand; itemnon-overlap;score-blindacousticbalancing;collection-specific andleave-one-outdiagnostics</td></tr></table>

## S12. Limitations and reporting boundaries

Brain Researcher verifies whether an analysis package satisfies declared validity constraints, produces required artifacts, and calibrates claims to available evidence. It does not prove biological truth, replace expert scientific judgment, or replace independent replication.

<table><tr><td rowspan=1 colspan=1>Limitation domain</td><td rowspan=1 colspan=1>What can go wrong</td><td rowspan=1 colspan=1>Reporting boundary</td></tr><tr><td rowspan=1 colspan=1>Knowledge graph</td><td rowspan=1 colspan=1>Incomplete negative knowledge, uneven graphdensity, source-specific gaps, sparse higher-tierevidence.</td><td rowspan=1 colspan=1>Evidence tier and coveragecaveats are reported.</td></tr><tr><td rowspan=1 colspan=1>Evidence connectors</td><td rowspan=1 colspan=1>Slow or failed connectors reduce coverage.</td><td rowspan=1 colspan=1>Failures are logged andmay trigger caveats; theyare not treated as absenceof evidence.</td></tr><tr><td rowspan=1 colspan=1>Planning and con-straints</td><td rowspan=1 colspan=1>Incomplete automated power analysis, lim-ited interactive replanning during execution,missing persistent rejection rationales in someworkflows.</td><td rowspan=1 colspan=1>Constraint state andplanning limitations arerecorded.</td></tr><tr><td rowspan=1 colspan=1>Execution</td><td rowspan=1 colspan=1>Dataset-access constraints, missing derivatives,backend failures, and weaker provenance forreal-time paths.</td><td rowspan=1 colspan=1>Non-executable outcomesand backend failures areretained.</td></tr><tr><td rowspan=1 colspan=1>Review</td><td rowspan=1 colspan=1>Rule registry is heuristic, not formal verifi-cation. It may miss subtle domain errors orover-warn on unusual valid methods.</td><td rowspan=1 colspan=1>Review verdicts reduceobvious errors but do notprove truth.</td></tr><tr><td rowspan=1 colspan=1>Benchmark construc-tion</td><td rowspan=1 colspan=1>Tool-calling and evidence-citation items andtheir reference answers are internally curatedand scored by LLM judges, so they may em-bed designer assumptions and LLM-judge bi-ases, including self-preference where the agentand a judge share a model family.</td><td rowspan=1 colspan=1>Scoring contracts, refer-ence answers, and prove-nance files are released forindependent re-scoring;benchmark results are re-ported as upstream-inputgains rather than scientificconclusions and use threecondition-blind judgescombined by majorityvote.</td></tr><tr><td rowspan=1 colspan=1>Routing-ablation scope</td><td rowspan=1 colspan=1>This seven-model ablation evaluates exact-label top-1 route selection with direct KG-named calls unavailable.</td><td rowspan=1 colspan=1>Report 362/420 sep-arately from Correctroute/tool@1, which has adifferent scoring contract.</td></tr><tr><td rowspan=1 colspan=1>Memory</td><td rowspan=1 colspan=1>Similarity-based retrieval and critic relationclassification can err; partition enforcement isrequired.</td><td rowspan=1 colspan=1>Memory is append-onlyand partitioned duringevaluation.</td></tr><tr><td rowspan=1 colspan=1>Bounded autonomy</td><td rowspan=1 colspan=1>Harbor reward can reflect file/schema/statisticcompletion without scientific validity.</td><td rowspan=1 colspan=1>Scientific acceptance stillrequires Brain Researcherreview and claim calibra-tion.</td></tr><tr><td rowspan=1 colspan=1>Hypothesis triage</td><td rowspan=1 colspan=1>Sparse-coverage domains can produce plausiblebut generic semantic permutations.</td><td rowspan=1 colspan=1>Used for expert-in-the-loop ideation, not au-tonomous novelty or truthclaims.</td></tr><tr><td rowspan=1 colspan=1>Governance</td><td rowspan=1 colspan=1>IRB, DUA, licensing, and institutional compli-ance remain investigator responsibilities.</td><td rowspan=1 colspan=1>BR can record blockersbut does not replace hu-man compliance review.</td></tr></table>

Four further boundaries are not fully captured by the table above. First, evidence-support judgments rely on LLM judges; we report inter-judge agreement (three-judge Fleiss’ κ = 0.50 with-BR, 0.72 without-BR; S11.1.2) and a reproducible 272-item hand audit (S11.1.3) but did not perform full-scale human-expert re-labeling of all judge outputs, so a residual gap between LLM and human judgment cannot be ruled out. Second, the absolute level of verified groundedness remains low (0.22 with BR under a three-judge majority vote): the benchmark demonstrates a between-condition improvement, not a solved grounding problem. Third, we did not run a randomized user study and did not directly measure runtime or researcher efort; statements about reduced workflow friction are expected consequences of absorbing the execution layer rather than measured outcomes. Fourth, the foundation models and provider APIs used are evolving, and although every reported claim is fixed to recorded model versions and snapshots (S1.2), exact behaviour may not be reproducible once providers retire or update those versions.

Review labels create a separate risk of automation complacency. A “BR-reviewed” status means only that the formalized conditions active for that review were checked; it is not expert endorsement or evidence that errors outside those conditions are absent. NM-H2 illustrates this boundary: automated review missed a sign-blind acceptance rule that direct inspection revealed. We did not measure whether the approving outcome reduced scrutiny, but treating such labels as conclusive could discourage the inspection needed to identify failures outside implemented checks. Human inspection therefore remains necessary, especially for claim–method alignment and other judgments not yet formalized.

Capabilities not evaluated in the reported protocol should not be used as evidence for the main claims. These include adaptive optimization modules, community contribution pathways, planned cross-modal fMRI-to-concept alignment, and autonomous scientific-discovery claims beyond the bounded validation contracts explicitly reported.

## Appendix / Data Card Overview

## Appendix A. Episode and control-plane card

The full episode/control-plane ledger is released in the archival repository (run-bundle schemas and provenance fields; Data availability); the card below is the summary of its fields.

The episode/control-plane card records the fixed identity and configuration of an episode: episode ID, mode, scientific question, model version, prompt-template version, registry snapshot, BR-KG snapshot, policy flags, MCP operation summary, memory namespace, checkpoint events, recovery events, run state, and completion status.

Example excerpt:

episode\_id: ep\_hcp\_predict\_001 mode: benchmark state: completed\_with\_qualifications registry\_snapshot: registry\_2026\_05\_15 brkg\_snapshot: brkg\_2026\_05\_10 memory\_namespace: benchmark\_isolated

## Appendix B. Evidence bundle / BR-KG card

The full BR-KG atlas and evidence-bundle ledger (graph snapshots, node/edge schemas, sourcecoverage and provenance tables, and the atlas figure panels) is released in the archival repository (Data availability). The card below retains the evidence-bundle schema and one worked method-condition record, the object the main text points to for a source’s verbatim quote and grounding label.

The evidence bundle card records input query, resolved entities, ONVOC mappings, evidence tiers, graph paths, literature retrieval outputs, connector failures, dataset links, tool links, prior findings, method-condition records, coverage notes, and embedding-lane metadata.

Example excerpt:

```yaml
resolved_entities:
dataset: HCP-YA
feature_space: Schaefer100x7
evidence_tiers:
- accepted_graph_record
- real_time_retrieval
connector_failures:
- PubMed_timeout
graph_path:
- Dataset:HCP-YA
- HAS_FEATURE_SPACE:Schaefer100x7
- REQUIRES:fold_manifest
method_condition_record: # one source-supported claim, field by field
claim: "resting-state FC predicts a fluid-intelligence component"
cohort_sample_size: "N=1003"
task_paradigm: resting_state
preprocessing: "ICA-FIX + 24-parameter motion regression"
statistical_model: "ridge regression, family-aware cross-validation"
grounding_label: br_kg_gabriel_cache_candidate
verbatim_quote: "connectome-based models predicted a general
intelligence factor (r approx 0.29) with leave-one-family-out CV"
```

## Appendix C. Dataset/resource card

The full per-dataset readiness ledger is released in the archival repository (dataset records and provenance fields; Data availability); the card below is the summary of its fields.

The dataset/resource card records dataset identifier, aliases, access class, local or remote path, BIDS root, derivative root, phenotype manifest, target variables, covariates, missing derivatives, backend reachability, readiness status, and blocker status. It includes at least one pass example and one block example.

## Appendix D. Tool registry and specification ledger card

The full tool-registry, candidate-ranking, and specification ledger is released in the archival repository (registry links and scoring tables; Data availability); the card below is the summary with a representative candidate-accounting excerpt.

The tool registry card records family cards, ranked tool candidates, rejected candidates, canonical IDs, backend recipes, parameter schemas, tool-contract clauses, compatibility checks, Neurodesk mappings, registry snapshot identifiers, allowlist mode, and embedding-lane metadata.

Example candidate accounting:

<table><tr><td rowspan=1 colspan=1>Candidate</td><td rowspan=1 colspan=1>Decision</td><td rowspan=1 colspan=1>Reason</td></tr><tr><td rowspan=1 colspan=1>predictive_modeling_family</td><td rowspan=1 colspan=1>accepted</td><td rowspan=1 colspan=1>Input resources and expected outputsmatch.</td></tr><tr><td rowspan=1 colspan=1>whole_brain-glm</td><td rowspan=1 colspan=1>rejected</td><td rowspan=1 colspan=1>Wrong output type for prediction target.</td></tr><tr><td rowspan=1 colspan=1>dynamic_fc_family</td><td rowspan=1 colspan=1>rejected</td><td rowspan=1 colspan=1>Missing time-series asset.</td></tr></table>

## Appendix E. Constraint and commitment card

The full constraint-compiler and commitment-gate record is released in the archival repository (Data availability); the card below is the summary of its fields.

The constraint/commitment card records active constraints, hard checks, soft checks, rule provenance, pass/warn/block verdicts, required sensitivity analyses, gate authority, commitment decision, and benchmark pass-through flag.

## Appendix F. Run bundle and provenance card

The full run-bundle and provenance ledger, including a worked claim-record example, is released in the archival repository (run-bundle schemas and provenance fields; Data availability); the card below is the summary of its fields.

The run-bundle card records event trace, trajectory document, observation record, analysis bundle, run card, expected artifacts, produced artifacts, missing artifacts, checksums, backend versions, software versions, container or module identifiers, preflight results, failures, retries, recovery events, and final execution status.

## Appendix G. Review card

The review card records verification inputs, deterministic checks, BLOCK findings, WARN findings, robustness checks, sensitivity checks, claim families, verdicts, caveat language, revision routing, claim eligibility, and artifact-completeness ratio. Each revise or block verdict identifies the triggering artifact or assumption, rule ID, hard/soft status, and evidence needed for resolution.

## Appendix H. Memory card

The full memory, claim-relation, and BR-KG-promotion ledger is released in the archival repository (Data availability); the card below is the summary of its fields.

The memory card records claim text, polarity, condition vector, review verdict, caveats, provenance pointer, related prior claims, relation type, relation confidence, stable key, memory namespace, writeback eligibility, and BR-KG promotion status. Accepted, qualified, and rejected outcomes are shown separately so readers can see what does and does not enter accepted memory.

## Appendix I. Operational-mode card

The operational-mode card records interactive gate records, bounded instruction file, action vocabulary, budget, stopping criteria, validation ladder, supervisor decisions, critic decisions, Harbor verifier outputs, reward or partial-credit metrics, escalation events, final termination status, and memory-writeback decision.

## Appendix J. Evaluation card

The evaluation card records benchmark suite list, task manifests, scoring contracts, model list, with-BR/without-BR protocol, memory partition policy, collaborator case table, bounded campaign contract, metric denominators, secondary metrics, ablation protocols, and excluded or non-executable cases. The card states whether scoring used binary success, partial credit, or both.

## Case-report skeleton for later appendix expansion

Each case report uses the same skeleton:

<table><tr><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1>Case-report element</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Question.</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Dataset/resource status.</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Tool pathway.</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Constraints and commitment gate.</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Execution artifacts.</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Review verdict.</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>Final claim language.</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>Human/system responsibility split.</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>Memory writeback status.</td></tr></table>

This skeleton documents the reporting contract for the case-report appendices. Apart from the representative generated-code excerpt in Appendix K, the automatically generated reports released through Appendix L are frozen audit artifacts. The earlier HCP retention-gate and TRIBE language-alignment reports are historical records of separate campaigns and do not supply the current iterative-episode results. The operative current results are those in Supplementary Methods S11.2–S11.3 and Appendix J.

## Appendix/Data Cards

## Appendix G. Scientific review, BLOCK/WARN findings, and claimcalibration card

Appendix G is the scientific review ledger: it consumes the run bundle and decides whether artifacts and candidate claims are accepted, qualified, revised, blocked, rejected, or deferred, with unresolved cases escalated for expert judgment.

## G1. Purpose and scope

This card records the scientific review outcome for an episode. It consumes the selected plan, active constraints, execution trace, artifact manifest, logs, QC outputs, scorecards, candidate claims, and prior evidence. It reports deterministic checks, BLOCK and WARN findings, robustness and sensitivity coverage, claim-calibration decisions, final verdicts, caveat language, revision routing, and claim eligibility.

## G2. Review input manifest

<table><tr><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1>Source appendix</td><td rowspan=1 colspan=1>Status</td></tr><tr><td rowspan=1 colspan=1>Selected plan</td><td rowspan=1 colspan=1>Appendix D/E</td><td rowspan=1 colspan=1>present</td></tr><tr><td rowspan=1 colspan=1>Active constraints</td><td rowspan=1 colspan=1>Appendix E</td><td rowspan=1 colspan=1>present</td></tr><tr><td rowspan=1 colspan=1>Run bundle</td><td rowspan=1 colspan=1>Appendix F</td><td rowspan=1 colspan=1>present</td></tr><tr><td rowspan=1 colspan=1>Artifact manifest</td><td rowspan=1 colspan=1>Appendix F</td><td rowspan=1 colspan=1>present</td></tr><tr><td rowspan=1 colspan=1>Logs/errors/warnings</td><td rowspan=1 colspan=1>Appendix F</td><td rowspan=1 colspan=1>present</td></tr><tr><td rowspan=1 colspan=1>Scorecard and QC outputs</td><td rowspan=1 colspan=1>Appendix F</td><td rowspan=1 colspan=1>present</td></tr><tr><td rowspan=1 colspan=1>Prior evidence / graph paths</td><td rowspan=1 colspan=1>Appendix B</td><td rowspan=1 colspan=1>present</td></tr><tr><td rowspan=1 colspan=1>Dataset/resource ledger</td><td rowspan=1 colspan=1>Appendix C</td><td rowspan=1 colspan=1>present</td></tr></table>

Important boundary: review consumes the run bundle, not the model transcript alone.

## G3. Validity-layer matrix

<table><tr><td rowspan=1 colspan=1>Validity layer</td><td rowspan=1 colspan=1>Review question</td><td rowspan=1 colspan=1>Example checks</td></tr><tr><td rowspan=1 colspan=1>Statistical validity</td><td rowspan=1 colspan=1>Is the inference valid?</td><td rowspan=1 colspan=1>exchangeability; multiple compar-isons; permutation/null</td></tr><tr><td rowspan=1 colspan=1>Measurement validity</td><td rowspan=1 colspan=1>Are data and QC adequate?</td><td rowspan=1 colspan=1>motion; registration; QC; reliability</td></tr><tr><td rowspan=1 colspan=1>Construct validity</td><td rowspan=1 colspan=1>Does the measure support theconstruct?</td><td rowspan=1 colspan=1>task/construct mapping; reverseinference risk</td></tr><tr><td rowspan=1 colspan=1>Generalization validity</td><td rowspan=1 colspan=1>Does the result generalize?</td><td rowspan=1 colspan=1>leakage; grouped CV; held-out/external support</td></tr><tr><td rowspan=1 colspan=1>Claim validity</td><td rowspan=1 colspan=1>Is language calibrated?</td><td rowspan=1 colspan=1>no causal/biomarker/external claimwithout evidence</td></tr></table>

## G4. Review rule registry excerpt and lifecycle status

The review registry separates rule severity from implementation lifecycle. Severity determines what a triggered rule does to a candidate claim. Lifecycle status records whether the rule is already

operational, can be implemented from structured metadata, requires additional review-context fields, requires text interpretation, or is used only for calibration and reviewer training.

<table><tr><td rowspan=1 colspan=1>Lifecycle status</td><td rowspan=1 colspan=1>Meaning</td><td rowspan=1 colspan=1>How it is reported</td></tr><tr><td rowspan=1 colspan=1>Implemented</td><td rowspan=1 colspan=1>The rule is operational in the review systemfor the required metadata fields.</td><td rowspan=1 colspan=1>May support an enforcedBLOCK or WARN verdict.</td></tr><tr><td rowspan=1 colspan=1>Deterministic candidate</td><td rowspan=1 colspan=1>The rule can likely be implemented fromstructured metadata, manifests, or pipelinelogs.</td><td rowspan=1 colspan=1>Reported as a candidatecheck unless the episoderecords enforcement.</td></tr><tr><td rowspan=1 colspan=1>Schema-dependent candi-date</td><td rowspan=1 colspan=1>The rule requires additional review-contextfields before deterministic enforcement.</td><td rowspan=1 colspan=1>Reported as a missing-field orprotocol requirement.</td></tr><tr><td rowspan=1 colspan=1>NLP/LLM candidate</td><td rowspan=1 colspan=1>The rule requires text interpretation, claimextraction, or semantic comparison.</td><td rowspan=1 colspan=1>Reported as assisted reviewor calibration unless indepen-dently verified.</td></tr><tr><td rowspan=1 colspan=1>Calibration-only</td><td rowspan=1 colspan=1>The rule is used for examples, annotatortraining, or benchmark calibration.</td><td rowspan=1 colspan=1>Does not by itself produce anenforced verdict.</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Rule ID</td><td rowspan=1 colspan=1>Severity</td><td rowspan=1 colspan=1>Validity layer / rea-son tag</td><td rowspan=1 colspan=1>Default action</td></tr><tr><td rowspan=1 colspan=1>feature_selection_outside_cv</td><td rowspan=1 colspan=1>BLOCK</td><td rowspan=1 colspan=1>generalization / leakage</td><td rowspan=1 colspan=1>revise or block</td></tr><tr><td rowspan=1 colspan=1>uncorrected_whole_brain.inference</td><td rowspan=1 colspan=1>BLOCK</td><td rowspan=1 colspan=1>statistical / multiplecomparisons</td><td rowspan=1 colspan=1>revise or block</td></tr><tr><td rowspan=1 colspan=1>same_data_roi_definition</td><td rowspan=1 colspan=1>BLOCK</td><td rowspan=1 colspan=1>circularity</td><td rowspan=1 colspan=1>block</td></tr><tr><td rowspan=1 colspan=1>gsr_without_sensitivity</td><td rowspan=1 colspan=1>WARN</td><td rowspan=1 colspan=1>measurement / contro-versial choice</td><td rowspan=1 colspan=1>sensitivity or caveat</td></tr><tr><td rowspan=1 colspan=1>small_sample_biomarker_claim</td><td rowspan=1 colspan=1>WARN</td><td rowspan=1 colspan=1>claim / claim inflation</td><td rowspan=1 colspan=1>downgrade claim</td></tr><tr><td rowspan=1 colspan=1>missing-random_seed</td><td rowspan=1 colspan=1>WARN</td><td rowspan=1 colspan=1>reproducibility / report-ing gap</td><td rowspan=1 colspan=1>report or fix</td></tr></table>

review rule:

rule id: feature selection outside cv

severity: BLOCK

validity layer: generalization

reason tags:

\- leakage

detection field: pipeline dag

trigger: feature selection fit before cv split

default action: revise or block

required fix: ”Move feature selection inside the cross-validation loop.”

## G5. BLOCK findings and WARN findings

<table><tr><td>Finding type</td><td>Examples</td><td>Consequence</td></tr><tr><td>BLOCK</td><td>leakage outside CV; invalid null circular ROI; uncorrected whole- brain inference</td><td>Cannot accept until resolved.</td></tr><tr><td>WARN</td><td>external validation missing; GSR unresolved; small sample; missing seed; partial QC</td><td>May accept with caveats, sensitiv- ity, or downgraded claim,</td></tr></table>

block findings: []  
block status: none detected

warn findings:

\- rule id: external validation missing

validity layer: generalization

reason tag: claim inflation

status: unresolved

required caveat: ”Internal validation only; no external generalization claim.”

## G5.1 Common review failure modes and default actions

The review card is easier to interpret if the rule registry is read as a set of common failure modes. These rows are not additional rules; they translate frequent neuroimaging review problems into the evidence the reviewer expects, the default BLOCK or WARN action, and the claim-language change that follows.

<table><tr><td rowspan=1 colspan=1>Failure mode</td><td rowspan=1 colspan=1>Evidence inspected</td><td rowspan=1 colspan=1>Default action</td><td rowspan=1 colspan=1>Claim-languageeffect</td></tr><tr><td rowspan=1 colspan=1>Leakage outside cross-validation</td><td rowspan=1 colspan=1>Pipeline DAG, fit scope forfeature selection, scaling, PCA,harmonization, or confoundregression, and fold manifest.</td><td rowspan=1 colspan=1>BLOCK until thefitting step is movedinside the trainingfold.</td><td rowspan=1 colspan=1>No generaliza-tion, prediction, orbiomarker claim.</td></tr><tr><td rowspan=1 colspan=1>Circular ROI or featuredefinition</td><td rowspan=1 colspan=1>ROI provenance, localizersource, contrast identity, datasetidentity, and whether the samedata also test the effect.</td><td rowspan=1 colspan=1>BLOCK unless anindependent localizer,atlas, or non-circularprovenance is sup-plied.</td><td rowspan=1 colspan=1>No confirmatoryeffect claim.</td></tr><tr><td rowspan=1 colspan=1>Invalid whole-brain ormap-level inference</td><td rowspan=1 colspan=1>Correction method, correctiondomain, primary threshold, per-mutation or spatial-null frame-work, and file space.</td><td rowspan=1 colspan=1>BLOCK for uncor-rected whole-braintests, spatial-domainmismatch, or missingspatial null.</td><td rowspan=1 colspan=1>Only descriptive orexploratory languageremains.</td></tr><tr><td rowspan=1 colspan=1>Unresolved controversialpreprocessing choice</td><td rowspan=1 colspan=1>Preprocessing record andsensitivity coverage for GSR,dynamic FC windows, graphthresholds, HRF model, or har-monization.</td><td rowspan=1 colspan=1>WARN with requiredsensitivity analysis.</td><td rowspan=1 colspan=1>Claim is qualified tothe tested preprocess-ing choices.</td></tr><tr><td rowspan=1 colspan=1>Missing external or held-out validation</td><td rowspan=1 colspan=1>Validation ladder, held-out split,external dataset availability,post-selection correction, andresource-gate status.</td><td rowspan=1 colspan=1>WARN or DEFER,depending on whetherinternal support issufficient.</td><td rowspan=1 colspan=1>No external general-ization or deploymentclaim.</td></tr><tr><td rowspan=1 colspan=1>Overstated interpreta-tion</td><td rowspan=1 colspan=1>Candidate claim text, evidencetype, behavioral covariates, out-of-sample evidence, ablations,and alternatives considered.</td><td rowspan=1 colspan=1>WARN with claimdowngrade.</td><td rowspan=1 colspan=1>Prediction, causal,mechanistic, orreverse-inference lan-guage is removedunless directly sup-ported.</td></tr></table>

## G6. Robustness, sensitivity, and artifact review

<table><tr><td colspan="1" rowspan="1">Check</td><td colspan="1" rowspan="1">Required by</td><td colspan="1" rowspan="1">Claim effect</td></tr><tr><td colspan="1" rowspan="1">with/without GSR</td><td colspan="1" rowspan="1">controversial-choice rule</td><td colspan="1" rowspan="1">qualified if not performed orunstable</td></tr><tr><td colspan="1" rowspan="1">parcellation sensitivity</td><td colspan="1" rowspan="1">validation ladder</td><td colspan="1" rowspan="1">supports robustness level if stable</td></tr><tr><td colspan="1" rowspan="1">confound model sensitivity</td><td colspan="1" rowspan="1">review rule or soft constraint</td><td colspan="1" rowspan="1">caveat if attenuated</td></tr><tr><td colspan="1" rowspan="1">permutation/null check</td><td colspan="1" rowspan="1">hard statistical validity</td><td colspan="1" rowspan="1">accept/block depending on result</td></tr><tr><td colspan="1" rowspan="1">post-selection correction</td><td colspan="1" rowspan="1">replayed configurations</td><td colspan="1" rowspan="1">accept/block depending on result</td></tr><tr><td colspan="1" rowspan="1">external dataset validation</td><td colspan="1" rowspan="1">L4 ladder</td><td colspan="1" rowspan="1">no external claim if deferred</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Artifact class</td><td rowspan=1 colspan=1>Required for review?</td><td rowspan=1 colspan=1>Review consequence if miss-ing</td></tr><tr><td rowspan=1 colspan=1>Scorecard</td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>review not ready or revise</td></tr><tr><td rowspan=1 colspan=1>Predictions</td><td rowspan=1 colspan=1>yes for prediction</td><td rowspan=1 colspan=1>statistical review incomplete</td></tr><tr><td rowspan=1 colspan=1>Null-test output</td><td rowspan=1 colspan=1>yes if claimed</td><td rowspan=1 colspan=1>revise/block</td></tr><tr><td rowspan=1 colspan=1>Config manifest</td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>reproducibility warning or revise</td></tr><tr><td rowspan=1 colspan=1>QC report</td><td rowspan=1 colspan=1>yes or conditional</td><td rowspan=1 colspan=1>measurement WARN</td></tr></table>

artifact review:

expected artifacts: 6

produced artifacts: 6

artifact completeness ratio: 1.00

review ready: true

## G7. Claim-family and claim-calibration table

<table><tr><td rowspan=1 colspan=1>Candidate claim</td><td rowspan=1 colspan=1>Overclaim blocked</td><td rowspan=1 colspan=1>Allowed language</td></tr><tr><td rowspan=1 colspan=1>Model predicts behavior</td><td rowspan=1 colspan=1>clinical biomarker</td><td rowspan=1 colspan=1>internally validated predictiveassociation</td></tr><tr><td rowspan=1 colspan=1>FC is associated with score</td><td rowspan=1 colspan=1>causal mechanism</td><td rowspan=1 colspan=1>exploratory or condition-qualifiedassociation</td></tr><tr><td rowspan=1 colspan=1>Result robust to parcellation</td><td rowspan=1 colspan=1>external validity</td><td rowspan=1 colspan=1>robustness under tested specifica-tions</td></tr><tr><td rowspan=1 colspan=1>Null-control passed</td><td rowspan=1 colspan=1>biological truth</td><td rowspan=1 colspan=1>leakage/null-controlled internalresult</td></tr></table>

## G8. Final review verdict and revision routing

<table><tr><td rowspan=1 colspan=1>Verdict</td><td rowspan=1 colspan=1>Meaning</td><td rowspan=1 colspan=1>Next action</td></tr><tr><td rowspan=1 colspan=1>accept</td><td rowspan=1 colspan=1>Artifacts and claims satisfy activeconstraints.</td><td rowspan=1 colspan=1>Eligible for memory.</td></tr><tr><td rowspan=1 colspan=1>accept with qualifications</td><td rowspan=1 colspan=1>Result usable only with caveats.</td><td rowspan=1 colspan=1>Qualified memory claim.</td></tr><tr><td rowspan=1 colspan=1>revise</td><td rowspan=1 colspan=1>Fixable issue remains.</td><td rowspan=1 colspan=1>Return to planning or execution.</td></tr><tr><td rowspan=1 colspan=1>block</td><td rowspan=1 colspan=1>Hard validity failure.</td><td rowspan=1 colspan=1>Stop or redesign.</td></tr><tr><td rowspan=1 colspan=1>reject</td><td rowspan=1 colspan=1>Claim unsupported or contra-dicted.</td><td rowspan=1 colspan=1>No writeback.</td></tr><tr><td rowspan=1 colspan=1>escalate</td><td rowspan=1 colspan=1>Human expert needed</td><td rowspan=1 colspan=1>Pause or route.</td></tr></table>

review card:

episode id: ep hcp predict 001

run id: run hcp pathb 001

verdict: accept with qualifications

block findings: []

warn findings:

\- external validation missing

robustness checks: family block permutation: pass post selection correction: pass external validation: deferred

claim calibration:

allowed claim strength: ”validated internal result”

prohibited claims:

\- ”externally validated biomarker”

\- ”causal mechanism”

\- ”clinical prediction claim”

memory eligibility: qualified claim only

## G9. Scientific review rule registry (release) and calibration library

This appendix gives the reader-facing view of the scientific review rule registry used to define, calibrate, and prioritize Brain Researcher’s verification layer. The full per-rule registry, its organization fields, rule families, encoded policy decisions, implementation priority queue, and review-context fields and sensitivity templates, is released as a machine-readable file in the archival repository (Data availability), so it is not reprinted here. A rule’s presence in the released registry does not mean that it is enforced in every episode; the lifecycle table in G4 distinguishes implemented rules, deterministic candidates, schema-dependent candidates, text-interpretation candidates, and calibration-only cases. Retained below is the calibration case library (G9.5), because the main text points to it for the first false-accept and false-block accounting.

## G9.5 Calibration case library

The registry includes 60 calibration cases (C01–C60) for annotator training and regression testing. They are summarized by rule family here; the machine-readable release contains the case-level labels, severity, novelty flag, and rule identifiers.

<table><tr><td colspan="1" rowspan="1">Cases</td><td colspan="1" rowspan="1">Scenarios covered</td><td colspan="1" rowspan="1">Typical label</td></tr><tr><td colspan="1" rowspan="1">C01-C06</td><td colspan="1" rowspan="1">Repeated-measures, mixed-design, longitu-dinal, and spatial-domain errors, plus validpaired-test and motor-task controls.</td><td colspan="1" rowspan="1">BLOCK for invaliddesign or spatial cor-rection; allow for validcontrols.</td></tr><tr><td colspan="1" rowspan="1">C07-C12</td><td colspan="1" rowspan="1">Extreme or unexpected effects, ordinary mor-phometry effects, motion-confounded small-sample FC, and expected task activation.</td><td colspan="1" rowspan="1">WARN for prior con-flict or confounding;allow for expected,well-scoped effects.</td></tr><tr><td colspan="1" rowspan="1">C13-C24</td><td colspan="1" rowspan="1">Whole-brain correction, cluster-threshold, lo-calization, double-dipping, ROI multiplicity,analytic flexibility, pseudoreplication, fixed-effects population claims, exchangeability, andspatial-null examples.</td><td colspan="1" rowspan="1">BLOCK for hard in-ference errors; WARNfor soft multiplicity orreporting issues.</td></tr><tr><td colspan="1" rowspan="1">C25-C36</td><td colspan="1" rowspan="1">Motion reporting, motion imbalance, globalsignal regression, dynamic FC, graph thresh-olding, task FC/PPI, EPI distortion, MRIQC,small sample, missing external validation, andmultiband QC.</td><td colspan="1" rowspan="1">WARN except for high-confidence motion con-founds that block infer-ence.</td></tr><tr><td colspan="1" rowspan="1">C37-C41</td><td colspan="1" rowspan="1">Reverse inference, stimulus generalization,missing behavioral covariates, ICV/TIV omis-sion, and multi-site site confounding.</td><td colspan="1" rowspan="1">WARN with claim nar-rowing or added covari-ates/sensitivity.</td></tr><tr><td colspan="1" rowspan="1">C42-C55</td><td colspan="1" rowspan="1">Harmonization, feature selection, standardiza-tion, test-set selection, split grouping, permu-tation, fold-wise error, prediction language,temporal splits, post-hoc layer selection, RSAmultiplicity and hierarchy, and missing ICC.</td><td colspan="1" rowspan="1">BLOCK for leak-age and test-set use;WARN for missing reli-ability, uncertainty, orpermutation support.</td></tr><tr><td>C56-C60</td><td>Extreme effects, single-pipeline significance, COBIDAS fields, BIDS validation, and BIDS Stats Models reporting.</td><td>WARN for incomplete robustness or report- ing; allow/positive modifier for structured reporting.</td></tr></table>

## Appendix I. Operational-mode, bounded-episode, and Harbor-based validation card

Appendix I is the operational control trace: it records how an episode or campaign was bounded, budgeted, advanced, verified, stopped, escalated, and permitted or denied memory writeback. The bounded self-evolving and loop-closure schematics are summarized in Supplementary Figs. S11–S13.

## I1. Purpose and scope

This card records the operational mode and control contract for interactive, benchmark, collaboratorfacing, and bounded autonomous episodes. It is especially important for Harbor-based validation and supervisor/critic cycles. Harbor verification is recorded as task-level execution evidence, not scientific acceptance.

## I2. Operational-mode identity

<table><tr><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Recommended entry</td></tr><tr><td rowspan=1 colspan=1>Episode or campaign ID</td><td rowspan=1 colspan=1>ep_hcp-predict_001 / campaign_autonomous_001</td></tr><tr><td rowspan=1 colspan=1>Mode</td><td rowspan=1 colspan=1>interactive / benchmark / collaborator / boundedautonomous</td></tr><tr><td rowspan=1 colspan=1>Gate authority</td><td rowspan=1 colspan=1>human researcher / evaluation harness / critic / super-visor</td></tr><tr><td rowspan=1 colspan=1>Instruction or task contract</td><td rowspan=1 colspan=1>instruction.md / bounded_episode_contract.yaml</td></tr><tr><td rowspan=1 colspan=1>Allowed resources</td><td rowspan=1 colspan=1>dataset/resource IDs and approved roots</td></tr><tr><td rowspan=1 colspan=1>Action vocabulary</td><td rowspan=1 colspan=1>search, plan, run, inspect, review, branch, stop</td></tr><tr><td rowspan=1 colspan=1>Budget</td><td rowspan=1 colspan=1>cycle, compute, wall-clock, retry, or token budget</td></tr><tr><td rowspan=1 colspan=1>Stopping criteria</td><td rowspan=1 colspan=1>success, unresolved block, no progress, budget ex-hausted, escalation</td></tr></table>

## I3. Bounded episode contract

<table><tr><td rowspan=1 colspan=1>Contract field</td><td rowspan=1 colspan=1>What it records</td></tr><tr><td rowspan=1 colspan=1>Target question</td><td rowspan=1 colspan=1>The scientific objective for the bounded episode.</td></tr><tr><td rowspan=1 colspan=1>Admissible design space</td><td rowspan=1 colspan=1>Which analyses and branches are allowed.</td></tr><tr><td rowspan=1 colspan=1>Cycle budget</td><td rowspan=1 colspan=1>Maximum number of supervisor/critic cycles.</td></tr><tr><td rowspan=1 colspan=1>Expected artifacts</td><td rowspan=1 colspan=1>Files and records required for verification/review.</td></tr><tr><td rowspan=1 colspan=1>Validation ladder</td><td rowspan=1 colspan=1>Target and achieved L0-L5 level.</td></tr><tr><td rowspan=1 colspan=1>Review rules</td><td rowspan=1 colspan=1>BLOCK/WARN rules required for acceptance.</td></tr><tr><td rowspan=1 colspan=1>Memory policy</td><td rowspan=1 colspan=1>accepted_only / qualified_only / no_write.</td></tr><tr><td rowspan=1 colspan=1>Escalation conditions</td><td rowspan=1 colspan=1>When to route to a human researcher.</td></tr></table>

operational mode card:

episode id: ep hcp predict 001

mode: bounded autonomous

gate authority: critic

action vocabulary:

\- resolve dataset

\- select workflow

\- run analysis

\- inspect artifacts

\- request review

\- branch sensitivity

\- terminate

budgets:

cycle budget: 3

wall clock budget: fixed

compute budget: fixed

stopping criteria:

\- validation ladder reached

\- hard review block unresolved

\- no progress after cycle limit

\- compute budget exhausted

\- human escalation required

## I4. Supervisor/critic decisions and Harbor verifier output

<table><tr><td rowspan=1 colspan=1>Record</td><td rowspan=1 colspan=1>Recommended fields</td></tr><tr><td rowspan=1 colspan=1>Branch-decision record</td><td rowspan=1 colspan=1>branch type; reason; proposed action; expectedresolution; budget cost</td></tr><tr><td rowspan=1 colspan=1>Critic decision</td><td rowspan=1 colspan=1>continue / investigate anomaly / broaden / revise /escalate / terminate</td></tr><tr><td rowspan=1 colspan=1>Harbor verifier output</td><td rowspan=1 colspan=1>artifact manifest present; required statistic present;schema valid; command completed; reward or partialcredit</td></tr><tr><td rowspan=1 colspan=1>Validation-ladder status</td><td rowspan=1 colspan=1>highest achieved level and unmet next gate</td></tr><tr><td rowspan=1 colspan=1>Termination status</td><td rowspan=1 colspan=1>completed / blocked / budget exhausted / escalatedterminated</td></tr></table>

harbor verifier:

artifact manifest present: true

required statistic present: true

schema valid: true

command completed: true

partial credit: 0.8

critic decision:

decision: terminate with qualified claim

reason: ”L3 internal validation reached; external validation deferred.”

final status: completed with qualifications

memory writeback allowed: qualified claim only

## Appendix J. Evaluation protocol, scoring contracts, and metric ledger

Appendix J records the evaluation protocol used for the quantitative benchmarks, collaborator cases, and bounded self-evolving campaigns reported in the main text. It defines the evaluated surfaces, condition comparisons, metric denominators, and aggregate values used in main-text Fig. 3. The benchmark surfaces are reported separately because tool calling and evidence citation have diferent item types, denominators, scoring rules, and ceilings.

## J1. Evaluation surfaces

<table><tr><td rowspan=1 colspan=1>Surface</td><td rowspan=1 colspan=1>Unit</td><td rowspan=1 colspan=1>What is evaluated</td></tr><tr><td rowspan=1 colspan=1>Tool-calling benchmark</td><td rowspan=1 colspan=1>60 neuroimaging task manifests;seven models per condition; 420model-item trajectories per condi-tion.</td><td rowspan=1 colspan=1>Whether a model maps a natural-language neuroimaging request to thecorrect analysis tool or executableroute before execution.</td></tr><tr><td rowspan=1 colspan=1>Routing ablation withoutdirect KG calls</td><td rowspan=1 colspan=1>Same 60-task manifest; sevenmodel routes; 420 route-taskepisodes.</td><td rowspan=1 colspan=1>Exact-label top-1 route selection whendirect KG-named calls are unavailable.</td></tr><tr><td rowspan=1 colspan=1>Evidence-citation bench-mark</td><td rowspan=1 colspan=1>76-item open-ended Neuroimage-Knowledge manifest; descriptiveResults aggregate reported on a50-question analysis set over allgenerated evidence-basis rows.</td><td rowspan=1 colspan=1>Whether model-generated neuroimag-ing claims cite evidence that can belocated and judged supportive by threecondition-blind LLM judges (ClaudeOpus 4.8, Codex GPT-5.5, and Gemini3.1 Pro). All reference-bearing rowswere sent to all three judges; a rowentered the numerator when at leasttwo judges voted verified, while everygenerated evidence-basis row remainedin the denominator.</td></tr><tr><td rowspan=1 colspan=1>Collaborator cases</td><td rowspan=1 colspan=1>Three active scientific questionsfrom collaborating researchers.</td><td rowspan=1 colspan=1>Whether Brain Researcher returnsbounded claim records rather thanunqualified findings.</td></tr><tr><td rowspan=1 colspan=1>Bounded self-evolving cam-paigns</td><td rowspan=1 colspan=1>Two predeclared episode contracts.</td><td rowspan=1 colspan=1>Whether fixed validation gates re-ject, defer, or revise claims duringexploratory search.</td></tr></table>

## J2. Benchmark conditions and model matrix

Both quantitative benchmarks compare with-BR and without-BR conditions on the same task or question set. The tool-calling benchmark is paired at the model–item level. The evidence-citation headline is instead a descriptive question-level aggregate over all generated evidence-basis rows; only rows receiving at least two verified votes contribute to the numerator, and it is not treated as a sevenmodel paired estimate. Benchmark memory was isolated from collaborator and bounded-campaign memory, and no benchmark item was promoted as a scientific claim.

<table><tr><td colspan="1" rowspan="1">Field</td><td colspan="1" rowspan="1">Entry</td></tr><tr><td colspan="1" rowspan="1">Conditions</td><td colspan="1" rowspan="1">Without-BR baseline; with-BR condition.</td></tr><tr><td colspan="1" rowspan="1">Model variants</td><td colspan="1" rowspan="1">Claude Code / Claude Opus 4.8; Codex GPT-5.5; Gemini 3.1 Pro; GLM-5.1; DeepSeek-V4-Pro; Kimi K2.5; Qwen3.6-Plus.</td></tr><tr><td colspan="1" rowspan="1">Tool-calling items</td><td colspan="1" rowspan="1">Natural-language neuroimaging requests requiring an analysis tool call orexecutable route with the required inputs available before execution.</td></tr><tr><td colspan="1" rowspan="1">Evidence-citation item fami-lies</td><td colspan="1" rowspan="1">Open-ended neuroimaging review questions covering preprocessing, statisti-cal inference, functional connectivity, multivariate decoding, meta-analysis,and reproducibility.</td></tr><tr><td colspan="1" rowspan="1">Isolation policy</td><td colspan="1" rowspan="1">Benchmark runs use fixed task manifests, fixed scoring contracts, andisolated memory partitions.</td></tr></table>

## J2a. Tool-calling benchmark scoring summary

The tool-calling benchmark is retained as an aggregate routing surface, but the prompt manifest is not reproduced in the appendix. Each item supplied the agent with a short routing request; the reference route and capabilities were held out for scoring. The scoring contract credits any response that covers the required capabilities, whether through a Brain Researcher call or an equivalent executable route. The release bundle records the exact prompt surfaces and provenance files for independent re-scoring.

## J2b. NeuroimageKnowledge benchmark manifests (released)

The full NeuroimageKnowledge groundable item manifest (the open-ended scenarios passed to the benchmark harness), representative paired with-BR/without-BR answers, the unified benchmarkbundle artifact inventory, and representative benchmark cases are released as machine-readable manifests in the benchmark package (Data availability), so they are not reprinted here. These manifests document the paired prompts used to elicit answers and evidence-basis rows under both conditions; they are not used for automatic correctness scoring, and the reported 50-question aggregate is defined by the scoring contracts below. Representative paired examples illustrate the interpretation used in the Results: strong models often answer canonical methods questions well without BR, while BR changes the evidence boundary by adding resolvable, source-supported rows, including cases where the without-BR answer wins the paired judgment because a memory-based citation resolves and supports the claim.

## J3. Scoring contracts and denominators

<table><tr><td colspan="1" rowspan="1">Metric</td><td colspan="1" rowspan="1">Numerator</td><td colspan="1" rowspan="1">Denominator</td></tr><tr><td colspan="1" rowspan="1">Capability@k</td><td colspan="1" rowspan="1">Fraction of required task capabilitiescovered by the selected call or routeafter the first k non-neutral actions.</td><td colspan="1" rowspan="1">Model-item trajectories inthe tool-calling benchmark;values are averaged as cover-age scores.</td></tr><tr><td colspan="1" rowspan="1">Correct route/tool@k</td><td colspan="1" rowspan="1">Tasks that reach exact task successunder the capability scorer after thefirst k non-neutral actions.</td><td colspan="1" rowspan="1">All tool-calling model-itemtrajectories.</td></tr><tr><td colspan="1" rowspan="1">Restricted-interface exact-labeltop-1</td><td colspan="1" rowspan="1">Requested routes with an ex-act top-1 label match underroute_search→tool_search; proto-col violations count as errors.</td><td colspan="1" rowspan="1">All 60 task-route episodes;gold candidate availabilityis reported separately as anoracle-coverage diagnostic.</td></tr><tr><td colspan="1" rowspan="1">Handoff score@k</td><td colspan="1" rowspan="1">Selected call or route contains enoughinformation for a receiving agent tocontinue the analysis after the first knon-neutral actions.</td><td colspan="1" rowspan="1">Tool-calling trajectorieswith parsed actions.</td></tr><tr><td colspan="1" rowspan="1">verified_groundedness_rate</td><td colspan="1" rowspan="1">Evidence items whose cited evidence isboth locatable and judged supportive.</td><td colspan="1" rowspan="1">All evidence-basis rows gen-erated for each question,followed by an equal-weightmean across the 50 ques-tions.</td></tr><tr><td colspan="1" rowspan="1">verified_among_claimed_grounded</td><td colspan="1" rowspan="1">Model-claimed grounded items whosecited evidence is both locatable andjudged supportive.</td><td colspan="1" rowspan="1">Model-claimed groundeditems.</td></tr><tr><td colspan="1" rowspan="1">citation_spam_rate</td><td colspan="1" rowspan="1">Model-claimed grounded items whosecited evidence is judged unrelated tothe claim.</td><td colspan="1" rowspan="1">Model-claimed groundeditems.</td></tr><tr><td colspan="1" rowspan="1">answer_correctness_rate</td><td colspan="1" rowspan="1">Answers passing the rubric-basedsource sanity check.</td><td colspan="1" rowspan="1">Scored NeuroimageKnowl-edge outputs with parseableanswer fields.</td></tr></table>

## J4. Aggregate benchmark values reported in Fig. 3

<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Without BR</td><td rowspan=1 colspan=1>With BR</td><td rowspan=1 colspan=1>Direction</td></tr><tr><td rowspan=1 colspan=1>Capability@1 (all tool-calling trajecto-ries)</td><td rowspan=1 colspan=1>0.498</td><td rowspan=1 colspan=1>0.945</td><td rowspan=1 colspan=1>Higher</td></tr><tr><td rowspan=1 colspan=1>Correct route/tool@1</td><td rowspan=1 colspan=1>0.233</td><td rowspan=1 colspan=1>0.936</td><td rowspan=1 colspan=1>Higher</td></tr><tr><td rowspan=1 colspan=1>Handoff score@1</td><td rowspan=1 colspan=1>0.474</td><td rowspan=1 colspan=1>0.761</td><td rowspan=1 colspan=1>Higher</td></tr><tr><td rowspan=1 colspan=1>verified_groundedness_rate (three-judge majority; 50-question mean)</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=1>0.220</td><td rowspan=1 colspan=1>Higher</td></tr><tr><td rowspan=1 colspan=1>verified_groundedness_rate (Gemini3.1 Pro recovered judge; 50-questionmean)</td><td rowspan=1 colspan=1>0.0435</td><td rowspan=1 colspan=1>0.2097</td><td rowspan=1 colspan=1>Higher</td></tr><tr><td rowspan=1 colspan=1>verified_groundedness_rate (CodexGPT-5.5 judge; 50-question mean)</td><td rowspan=1 colspan=1>0.0389</td><td rowspan=1 colspan=1>0.3088</td><td rowspan=1 colspan=1>Higher</td></tr><tr><td rowspan=1 colspan=1>verified_groundedness_rate (ClaudeOpus 4.8 judge; 50-question mean)</td><td rowspan=1 colspan=1>0.0796</td><td rowspan=1 colspan=1>0.2187</td><td rowspan=1 colspan=1>Higher</td></tr><tr><td rowspan=1 colspan=1>verified_among_claimed_grounded(Gemini 2.5 Flash single-judge safeguard;by-model mean)</td><td rowspan=1 colspan=1>0.273</td><td rowspan=1 colspan=1>0.583</td><td rowspan=1 colspan=1>Higher</td></tr><tr><td rowspan=1 colspan=1>citation_spam_rate (Gemini 2.5 Flashsingle-judge safeguard; by-model mean)</td><td rowspan=1 colspan=1>0.317</td><td rowspan=1 colspan=1>0.249</td><td rowspan=1 colspan=1>Lower</td></tr><tr><td rowspan=1 colspan=1>answer_correctness_rate (rubric-basedsafeguard; by-model mean)</td><td rowspan=1 colspan=1>0.749</td><td rowspan=1 colspan=1>0.789</td><td rowspan=1 colspan=1>Higher</td></tr></table>

Tool-calling action-budget metrics are trajectory-level quantities over 420 model-item trajectories per condition. The grounding headline and judge-specific rates are equal-weight means of the per-question fractions over all generated evidence-basis rows; only rows receiving at least two verified votes contribute to the headline numerator, while all other rows add no verified count. A separate exact-label diagnostic among the 444 non-verified with-BR rows present in all three judge outputs assigned a no unrelated majority to 289 (65%), a partial majority to 124 (28%), and no exact-label majority or cannot judge majority to the remaining 31 (7%); no row had a fabricated or malformed exact-label majority. The precision and citation-spam safeguards use the earlier Gemini 2.5 Flash single-judge scoring and are not three-judge-majority estimates. Evidence-citation metrics use evidence-item, claimed-grounded-item, or answer-output denominators as specified above and are not pooled with tool-calling metrics.

## J5. Collaborator-case ledger

<table><tr><td rowspan=1 colspan=1>Case</td><td rowspan=1 colspan=1>Evaluation object</td><td rowspan=1 colspan=1>Reported outcome boundary</td></tr><tr><td rowspan=1 colspan=1>NeuroMarkschizophrenia FNC</td><td rowspan=1 colspan=1>FBIRN cohort, N = 363;480-specification multiverseover connectivity, confound,dimensionality-reduction, clas-sifier, and domain granularity.</td><td rowspan=1 colspan=1>NM-H1, NM-H2, and NM-H3 all recorded asqualified (NM-H2 favorable in 12 of 24 contrastsafter sign-aware rescoring); support dependedon analytic regime.</td></tr><tr><td rowspan=1 colspan=1>Cocaine-use-disorderconnectivity</td><td rowspan=1 colspan=1>SUDMEX CONN / Open-Neuro ds003346, N = 138; 36-specification multiverse overatlases, framewise-displacementthresholds, and confound strate-gies.</td><td rowspan=1 colspan=1>Five prespecified network-systemic-segregationassociations rejected under SDMA-GLS (allZ &lt; 1.24, FDR q &gt; 0.58); exploratory 70-combination screen yielded no FDR-survivingeffects (max ZGLs = 2.18; 0/70 surviving).</td></tr><tr><td rowspan=1 colspan=1>Cross-cultural socialcognition</td><td rowspan=1 colspan=1>21 published studies, 85 MNIcoordinates, four culture-by-relationship ALE cells with k =6–8 cell-level entries.</td><td rowspan=1 colspan=1>Mechanistic mPFC-topology interpretationblocked as exploratory because cell counts werebelow recommended ALE stability thresholdsand paradigm composition was imbalanced.</td></tr></table>

## J6. Bounded self-evolving campaign ledger

<table><tr><td rowspan=1 colspan=1>Campaign</td><td rowspan=1 colspan=1>Predeclared controls</td><td rowspan=1 colspan=1>Claim-status outcome</td></tr><tr><td rowspan=1 colspan=1>HCP-YA workflowsearch and transfer</td><td rowspan=1 colspan=1>Two staged search bundles with all116 slots retained in the denomina-tor (104 scored in parent runs; 12transport failures); frozen selectedworkflow designated before a lo-cally matched comparison; repeatedfamily-grouped nested CV; frozenselected workflow refits across fouradditional outcomes; separate inter-nal holdout.</td><td rowspan=1 colspan=1>The frozen selected workflow exceeded thematched comparator in 10/10 Cognitionsplits and in 37/40 additional-outcomesplits, but the result remained retrospectiveand same-cohort, no transfer cell survivedweak-FWER correction, and the separateholdout did not confirm the comparatorclaim. No external scientific acceptance.</td></tr><tr><td rowspan=1 colspan=1>TRIBE speech-toolsgeometry</td><td rowspan=1 colspan=1>Complete 15-pair six-categorydiscovery; researcher-authorizedpost-hoc contrast choice; frozengeometry estimand; three non-overlapping recurring-source panels;score-blind acoustic balancing andHolm-corrected permutation in-ference on four previously unusedcollections.</td><td rowspan=1 colspan=1>Three recurring-source panels met boundedsupport (11/12 collection cells in the pre-dicted direction), but the frozen four-new-source H1 was not supported after correc-tion (Holm p = .396). Terminal outcomeinconclusive or conflicting; no scientificacceptance.</td></tr></table>

## J7. Exclusions and denominator handling

Non-executable or blocked cases are retained in the relevant denominator unless the scoring contract explicitly defines a successful-run subset, as in the first-correct budget. Resource failures, failed preflight checks, missing required artifacts, governance blocks, and invalid tool routes are recorded as outcomes rather than silently removed. For evidence citation, claims without locatable evidence remain in the all-claims denominator for verified groundedness.

## Appendix K. Cocaine-use-disorder case: representative generated code

The listing below is representative analysis code that Brain Researcher generated and executed for the cocaine-use-disorder case, a resting-state functional-connectivity multiverse on the SUDMEX cohort (CUD versus HC). The excerpt captures the multiverse specification: a header describing the design (cortical and subcortical atlas configurations crossed with framewise-displacement thresholds, aggregated by SDMA-GLS; this representative excerpt shows the atlas-by-FD grid, while the full 36-specification multiverse additionally crosses confound strategies), the 10-network definitions, the atlas-configuration grid, and a representative core function signature. Templated project tokens (a dollar-brace project-directory variable) are shown verbatim, and concrete filesystem locations are written as the generic placeholder <DATA ROOT>. The complete code is released per the Code availability statement.

Multiverse Functional Connectivity Analysis -- SUDMEX CUD vs HC   
Specifications = atlas configs x FD motion thresholds (SDMA-GLS aggregated)   
Atlas configs:   
Sch200\_Tian3 : Schaefer-200 + Tian Scale III (250 parcels)   
Sch400\_Tian3 : Schaefer-400 + Tian Scale III (450 parcels) <- primary   
Sch600\_Tian3 : Schaefer-600 + Tian Scale III (650 parcels)   
Sch400\_HCP : Schaefer-400 + HCP subcortical (419 parcels)   
Gordon\_Tian3 : Gordon-333 + Tian Scale III (383 parcels)   
Glasser\_Tian3 : Glasser-360 + Tian Scale III (410 parcels)   
FD motion thresholds: post-hoc re-censoring of XCP-D timeseries.   
XCP-D originally censored at FD=0.5; stricter thresholds drop more frames.   
Aggregation: SDMA-GLS (Lefort-Besnard et al., 2025, Imaging Neuroscience).   
  
# Project-relative paths use a templated project-directory token.   
XCPD = Path('\${PROJ\_DIR}/derivatives/xcpd') # e.g. <DATA\_ROOT>/derivatives/xcpd   
PHENO = Path('\${PROJ\_DIR}/work/analysis\_ready\_surface.tsv')   
# Network definitions: Yeo-7 cortical + 3 subcortical systems (10 total).   
YEO7 = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']   
SUBCORT\_GROUPS = {'Striatum': ['PUT-', 'CAU-', 'NAc-', 'GP'],   
'Thalamus': ['THA-'], 'HippAmyg': ['HIP-', 'AMY']}   
ALL\_NETWORKS = YEO7 + list(SUBCORT\_GROUPS.keys())   
FD\_THRESHOLDS = [0.3, 0.5]   
ATLAS\_CONFIGS = [   
{'id': 'Sch400\_Tian3', 'ctx\_seg': '4S456Parcels', 'n\_ctx': 400, 'assign': 'schaefer'},   
{'id': 'Glasser\_Tian3', 'ctx\_seg': 'Glasser', 'n\_ctx': None, 'assign': 'glasser'},   
# ... remaining atlas configurations crossed with FD\_THRESHOLDS ...   
]   
def compute\_network\_fc(mat, parcel\_network):   
"""Mean within- and between-network FC (Fisher-z averaged) per spec."""   
# core loop: for net in ALL\_NETWORKS -> within\_{net};   
# for n1,n2 in combinations(ALL\_NETWORKS,2) -> between\_{n1}\_{n2}

## Appendix L. Per-case reports (released in the repository)

The automatically generated reports released with the repository are frozen episode snapshots. They preserve the system output, figures, tables, statistics, and provenance available when each report was closed, and they are not silently rewritten when a later episode supersedes the scientific trajectory. The NeuroMark report reflects the corrected, direction-aware re-audit described in Supplementary Methods S11.2.1. The earlier HCP retention-gate and TRIBE language-alignment reports are historical records of separate campaigns; they are not the evidentiary source for the current HCP workflow-search and TRIBE speech–tools episodes. The complete current numerical results and claim boundaries are given in S11.3.1–S11.3.3, with the underlying run bundles and current research-line reports identified through the Data availability statement.

## Appendix M. Benchmark human-audit sheet (released in the repository)

The graded sheet from the human audit of the automated benchmark judges (Supplementary Methods S11.1) is released with the benchmark package (Data availability; Appendix J). It records a reproducible random 20% sample (seed 20260630) of the scored results across the reported benchmarks (272 items across tool routing, NIK answer keys, and grounding judgments), each adjudicated by hand against the recorded automated verdict, with a per-item grade and, for the 11 flagged grounding items, the reason the verdict is debatable. The summary and the flagged-item accounting are given in Tables 4 and 5.

## References

[1] Elena A. Allen, Eswar Damaraju, Sergey M. Plis, Erik B. Erhardt, Tom Eichele, and Vince D. Calhoun. Tracking whole-brain connectivity dynamics in the resting state. Cerebral Cortex, 24 (3):663–676, 2014. doi: 10.1093/cercor/bhs352.

[2] Diego Angeles-Valdez, Jalil Rasgado-Toledo, Victor Issa-Garcia, Thania Balducci, Viviana Villica˜na, Alely Valencia, Jorge Julio Gonzalez-Olvera, Ernesto Reyes-Zamorano, Eduardo A. Garza-Villarreal, et al. The Mexican magnetic resonance imaging dataset of patients with cocaine use disorder: SUDMEX CONN. Scientific Data, 9(1):133, 2022. doi: 10.1038/ s41597-022-01251-3.

[3] Anthropic. Introducing the Model Context Protocol, 2024. URL https://www.anthropic. com/news/model-context-protocol. Pages: 2024 Publication Title: Anthropic News Volume: November 25.

[4] B. B. Avants, N. J. Tustison, G. Song, P. A. Cook, A. Klein, and J. C. Gee. A reproducible evaluation of ANTs similarity metric performance in brain image registration. NeuroImage, 54(3):2033–2044, 2011. doi: 10.1016/j.neuroimage.2010.09.025. URL https://doi.org/10. 1016/j.neuroimage.2010.09.025.

[5] R. Ciric, D. H. Wolf, J. D. Power, D. R. Roalf, G. L. Baum, K. Ruparel, R. T. Shinohara, M. A. Elliott, S. B. Eickhof, C. Davatzikos, R. C. Gur, R. E. Gur, D. S. Bassett, and T. D. Satterthwaite. Benchmarking confound regression strategies for the control of motion artifact in studies of functional connectivity. NeuroImage, 154:174–187, 2017. doi: 10.1016/j.neuroimage. 2017.03.020. URL https://doi.org/10.1016/j.neuroimage.2017.03.020.

[6] R. W. Cox. AFNI: Software for analysis and visualization of functional magnetic resonance neuroimages. Computers and Biomedical Research, 29(3):162–173, 1996. doi: 10.1006/cbmr. 1996.0014. URL https://doi.org/10.1006/cbmr.1996.0014.

[7] Stephane d’Ascoli, Jeremy Rapin, Yohann Benchetrit, Teon Brooks, Katelyn Begany, Josephine Raugel, Hubert Banville, and Jean-Remi King. A foundation model of vision, audition, and language for in-silico neuroscience, 2026. URL https://doi.org/10.48550/arXiv.2605. 04326. Publication Title: arXiv.

[8] Yuhui Du, Zening Fu, Jing Sui, Shuang Gao, Ying Xing, Dongdong Lin, Mustafa Salman, Anees Abrol, Md Abdur Rahaman, Jiayu Chen, L. Elliot Hong, Peter Kochunov, Elizabeth A.

Osuch, and Vince D. Calhoun. NeuroMark: An automated and adaptive ICA-based pipeline to identify reproducible fMRI markers of brain disorders. NeuroImage: Clinical, 28:102375, 2020. doi: 10.1016/j.nicl.2020.102375.

[9] S. B. Eickhof, A. R. Laird, C. Grefkes, L. E. Wang, K. Zilles, and P. T. Fox. Coordinate-based activation likelihood estimation meta-analysis of neuroimaging data: A random-efects approach based on empirical estimates of spatial uncertainty. Human Brain Mapping, 30(9):2907–2926, 2009. doi: 10.1002/hbm.20718. URL https://doi.org/10.1002/hbm.20718.

[10] S. B. Eickhof, T. E. Nichols, A. R. Laird, et al. Behavior, sensitivity, and power of activation likelihood estimation characterized by massive empirical simulation. NeuroImage, 137:70–85, 2016. doi: 10.1016/j.neuroimage.2016.04.072. URL https://doi.org/10.1016/j.neuroimage. 2016.04.072.

[11] O. Esteban, D. Birman, M. Schaer, O. O. Koyejo, R. A. Poldrack, and K. J. Gorgolewski. MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites. PLOS ONE, 12(9):e0184661, 2017. doi: 10.1371/journal.pone.0184661. URL https://doi.org/10. 1371/journal.pone.0184661.

[12] B. Fischl. FreeSurfer. NeuroImage, 62(2):774–781, 2012. doi: 10.1016/j.neuroimage.2012.01.021. URL https://doi.org/10.1016/j.neuroimage.2012.01.021.

[13] Eduardo A. Garza-Villarreal, Jorge Julio Gonzalez Olvera, Thania Balducci, Diego Angeles Valdez, Alely Valencia, and Jalil Rasgado. SUDMEX CONN: The Mexican dataset of cocaine use disorder patients. OpenNeuro dataset, 2026. URL https://doi.org/10.18112/ openneuro.ds003346.v1.1.3.

[14] K. J. Gorgolewski, T. Auer, V. D. Calhoun, et al. The Brain Imaging Data Structure, a format for organizing and describing outputs of neuroimaging experiments. Scientific Data, 3:160044, 2016. doi: 10.1038/sdata.2016.44. URL https://doi.org/10.1038/sdata.2016.44.

[15] L. Henschel, S. Conjeti, S. Estrada, K. Diers, B. Fischl, and M. Reuter. FastSurfer–A fast and accurate deep learning based neuroimaging pipeline. NeuroImage, 219:117012, 2020. doi: 10.1016/ j.neuroimage.2020.117012. URL https://doi.org/10.1016/j.neuroimage.2020.117012.

[16] M. Jenkinson, C. F. Beckmann, T. E. J. Behrens, M. W. Woolrich, and S. M. Smith. FSL. NeuroImage, 62(2):782–790, 2012. doi: 10.1016/j.neuroimage.2011.09.015. URL https://doi. org/10.1016/j.neuroimage.2011.09.015.

[17] N. Kriegeskorte, M. Mur, and P. A. Bandettini. Representational similarity analysis–connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience, 2:4, 2008. doi: 10.3389/neuro.06.004.2008. URL https://doi.org/10.3389/neuro.06.004.2008.

[18] J. Lefort-Besnard, T. E. Nichols, and C. Maumet. Statistical inference for neuroimaging multiverse analyses with the same-data meta-analysis. Imaging Neuroscience, 2025. doi: 10.1162/imag a 00513. URL https://doi.org/10.1162/imag\_a\_00513.

[19] Zhen-Qi Liu, Andrea I. Luppi, Justine Y. Hansen, Ye Ella Tian, Andrew Zalesky, B. T. Thomas Yeo, Ben D. Fulcher, and Bratislav Misic. Benchmarking methods for mapping functional connectivity in the brain. Nature Methods, 22(7):1593–1602, 2025. doi: 10.1038/s41592-025-02704-4. URL https://doi.org/10.1038/s41592-025-02704-4.

[20] C. J. Markiewicz, K. J. Gorgolewski, F. Feingold, et al. The OpenNeuro resource for sharing of neuroscience data. eLife, 10:e71774, 2021. doi: 10.7554/eLife.71774. URL https://doi.org/ 10.7554/eLife.71774.

[21] K. Murphy, R. M. Birn, D. A. Handwerker, T. B. Jones, and P. A. Bandettini. The impact of global signal regression on resting state correlations: Are anti-correlated networks introduced? NeuroImage, 44(3):893–905, 2009. doi: 10.1016/j.neuroimage.2008.09.036. URL https://doi. org/10.1016/j.neuroimage.2008.09.036.

[22] T. E. Nichols, S. Das, S. B. Eickhof, et al. Best practices in data analysis and sharing in neuroimaging using MRI. Nature Neuroscience, 20:299–303, 2017. doi: 10.1038/nn.4500. URL https://doi.org/10.1038/nn.4500.

[23] L. Parkes, B. Fulcher, M. Yucel, and A. Fornito. An evaluation of the eficacy, reliability, and sensitivity of motion correction strategies for resting-state functional MRI. NeuroImage, 171: 415–436, 2018. doi: 10.1016/j.neuroimage.2017.12.073. URL https://doi.org/10.1016/j. neuroimage.2017.12.073.

[24] R. A. Poldrack, C. J. Markiewicz, S. Appelhof, et al. The past, present, and future of the Brain Imaging Data Structure (BIDS). Imaging Neuroscience, 2:1–19, 2024. doi: 10.1162/ imag a 00103. URL https://doi.org/10.1162/imag\_a\_00103.

[25] T. Salo, T. Yarkoni, T. E. Nichols, J.-B. Poline, M. Bilgel, K. L. Bottenhorn, et al. NiMARE: Neuroimaging Meta-Analysis Research Environment. Aperture Neuro, 3:1–32, 2023. doi: 10.52294/001c.87681. URL https://doi.org/10.52294/001c.87681.

[26] A. Schaefer, R. Kong, E. M. Gordon, et al. Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cerebral Cortex, 28(9):3095–3114, 2018. doi: 10.1093/cercor/bhx179. URL https://doi.org/10.1093/cercor/bhx179.

[27] Stephen M. Smith, Mark Jenkinson, Mark W. Woolrich, Christian F. Beckmann, Timothy E. J. Behrens, Heidi Johansen-Berg, Peter R. Bannister, Marilena De Luca, Ivana Drobnjak, David E. Flitney, Rami K. Niazy, James Saunders, John Vickers, Yongyue Zhang, Nicola De Stefano, J. Michael Brady, and Paul M. Matthews. Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage, 23(Suppl. 1):S208–S219, 2004. doi: 10.1016/ j.neuroimage.2004.07.051. URL https://doi.org/10.1016/j.neuroimage.2004.07.051.

[28] Model Context Protocol Specification. Specification, protocol revision 2024-11-05, 2024. URL https://modelcontextprotocol.io/specification/2024-11-05/.

[29] J.-D. Tournier, R. Smith, D. Rafelt, et al. MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation. NeuroImage, 202:116137, 2019. doi: 10.1016/j. neuroimage.2019.116137. URL https://doi.org/10.1016/j.neuroimage.2019.116137.

[30] D. C. Van Essen, S. M. Smith, D. M. Barch, et al. The WU-Minn Human Connectome Project: An overview. NeuroImage, 80:62–79, 2013. doi: 10.1016/j.neuroimage.2013.05.041. URL https://doi.org/10.1016/j.neuroimage.2013.05.041.

[31] G. Varoquaux. Cross-validation failure: Small sample sizes lead to large error bars. NeuroImage, 180:68–77, 2018. doi: 10.1016/j.neuroimage.2017.06.061. URL https://doi.org/10.1016/j. neuroimage.2017.06.061.

[32] A. M. Winkler, G. R. Ridgway, M. A. Webster, S. M. Smith, and T. E. Nichols. Permutation inference for the general linear model. NeuroImage, 92:381–397, 2014. doi: 10.1016/j.neuroimage. 2014.01.060. URL https://doi.org/10.1016/j.neuroimage.2014.01.060.

[33] Anderson M. Winkler, Matthew A. Webster, Diego Vidaurre, Thomas E. Nichols, and Stephen M. Smith. Multi-level block permutation. NeuroImage, 123:253–268, 2015. doi: 10.1016/j. neuroimage.2015.05.092.

[34] WU-Minn Human Connectome Project Consortium. HCP Young Adult 1200 Subjects Data Release, 2017. URL https://www.humanconnectome.org/study/hcp-young-adult/document/ 1200-subjects-data-release. Published: Human Connectome Project data release page.
# From Safety Documentation to Safety Knowledge Support: An Evidence-Grounded LLM Framework for Medical Devices

Tuhinangshu Gangopadhyay

Rasmus Adler

Peter Liggesmeyer

Fraunhofer IESE

Fraunhofer IESE

Fraunhofer IESE

RPTU Kaiserslautern-Landau

Kaiserslautern, Germany

Kaiserslautern, Germany

RPTU Kaiserslautern-Landau

rasmus.adler@iese.fraunhofer.de

tuhinangshu.gangopadhyay@iese-extern.fraunhofer.de

Kaiserslautern, Germany

peter.liggesmeyer@iese.fraunhofer.de

Jan Reich

Fraunhofer IESE

Kaiserslautern, Germany

jan.reich@iese.fraunhofer.de

Abstract—Medical devices are becoming more softwareintensive, connected, and AI-enabled. Their development requires risk-management evidence aligned with ISO 14971 and, for software, IEC 62304. This evidence must be kept consistent across requirements, design decisions, software changes, verification results, complaints, and post-market data. These tasks are costly and depend on scarce safety and domain experts.

Large language models (LLMs) may reduce parts of this effort because medical-device safety work is highly documentbased. However, current LLM-based safety-engineering studies often address isolated methods, rely on generic prompting or public examples, and provide limited support for source links, traceability, uncertainty handling, lifecycle updates, and recorded expert review. This limits their use in regulated medical-device development.

This paper argues that the central research problem is not safety-text generation, but source-linked safety-knowledge support. We propose an evidence-grounded framework that connects device artifacts, controlled knowledge storage and retrieval, method-specific generation of candidate safety items, critique and uncertainty checks, and recorded expert review. The framework prepares, links, checks, and updates candidate safety artifacts for expert decision-making. It does not decide whether a device is safe and does not provide regulatory approval. We also outline an evaluation strategy using non-public or newly built medical-device case studies and expert reference analyses to assess coverage, correctness, relevance, traceability, duplicate rate, unsupported claims, and review effort.

Index Terms—Large Language Models, Medical Devices, Safety Analysis, Evaluation Metrics, Human Expert Review

## I. INTRODUCTION

Medical devices are increasingly software-intensive, connected, and AI-enabled. This development supports new functions in diagnosis, therapy, rehabilitation, hospital logistics, and surgery. At the same time, it increases the effort needed to show that a device is safe for patients, users, and its intended use environment. Manufacturers must provide riskmanagement evidence and safety documentation aligned with standards such as ISO 14971 for risk management [1] and IEC 62304 for medical software lifecycle processes [2].

Safety analysis is expert-intensive. Engineers must identify hazards, hazardous situations, failure modes, causal factors, foreseeable misuse, possible harm, and suitable risk-control measures. They must also keep these artifacts aligned with requirements, architecture, software design, verification evidence, production data, post-production data, and device changes. This creates a major cost and time burden, especially for small and medium-sized manufacturers. An MDR survey by DIHK, MedicalMountains, and SPECTARIS reports certification costs as the most frequently named reason for discontinuing products in the EU, cited by 91% of respondents [3].

LLMs are a promising support technology for this setting because safety engineering is strongly document-based. Device descriptions, requirements, hazard logs, FMEA tables, risk-control records, safety cases, test reports, user manuals, incident reports, and standards contain information that can support risk-management work. LLMs can help safety engineers draft candidate hazards, failure modes, causal links, risk controls, safety requirements, review questions, and safetycase fragments.

However, in regulated medical-device development, such support cannot be treated as independent safety assessment. Generated outputs need source grounding, traceability, uncertainty indicators, and expert review. A candidate hazard without evidence links is difficult to assess. A candidate risk control without links to requirements and verification evidence is difficult to justify. A generated risk score without recorded assumptions may even mislead reviewers.

As reviewed in Section III, recent work shows that LLMs can support tasks such as Hazard Analysis and Risk Assessment (HARA), Failure Mode and Effects Analysis (FMEA), Fault Tree Analysis (FTA), System-Theoretic Process Analysis (STPA), safety-requirements generation, and assurance-case analysis. Yet these studies often address isolated tasks, rely on prompt engineering, use public case studies, or focus on nonmedical domains. This limits their use for medical devices, where device-specific knowledge, confidentiality, audit trails, rare high-severity harms, lifecycle updates, and fair evaluation are central concerns.

This paper argues that LLM-supported medical-device safety analysis should be framed as source-linked safetyknowledge support. The goal is not to let an LLM decide whether a device is safe. The goal is to help experts prepare, link, check, review, and update candidate safety artifacts in a controlled process.

The paper makes four contributions:

1) It summarizes recent patterns in LLM-assisted safety engineering and classifies them by safety task and dominant LLM method.

2) It identifies gaps that limit current approaches in regulated medical-device development.

3) It proposes a framework for evidence-grounded, traceable, reviewable, and updatable safety-knowledge support.

4) It defines an evaluation strategy based on non-public or newly built medical-device case studies, expert reference analyses, and safety-specific metrics.

The remainder of the paper is structured as follows. Section II explains why medical-device safety analysis is a demanding research context. Section III reviews patterns in LLM-assisted safety engineering. Section IV derives the research gap. Section V presents the proposed framework. Section VI outlines the evaluation strategy. Section VII concludes the paper.

## II. MEDICAL-DEVICE SAFETY ANALYSIS AS A RESEARCH CONTEXT

Medical-device safety analysis is not a final check before market release. It is a lifecycle activity. ISO 14971 requires manufacturers to identify hazards and hazardous situations, estimate and evaluate risks, select and verify risk-control measures, assess residual risk, and use production and postproduction information [1]. IEC 62304 adds software lifecycle activities, including requirements, architecture, implementation, verification, maintenance, and risk-related software changes [2].

This creates a demanding traceability problem. Safety artifacts must be linked to device functions, requirements, software items, design decisions, verification evidence, risk controls, and residual-risk decisions [1], [2], [4], [5]. When a requirement changes, when software is updated, or when post-market feedback reveals a new issue, the manufacturer must assess which safety artifacts are affected and whether additional risk control or verification is needed.

Medical devices also differ from many other technical systems because harm can depend on the device, the user, the patient, and the clinical context together. Consider an infusion pump. A safety analysis must not only consider hardware faults and software defects. It must also consider wrong dose entry, wrong flow rate, delayed occlusion detection, misleading alarms, use under time pressure, patient-specific sensitivity, and interaction with clinical workflows. The same technical failure may have different safety meaning depending on the intended use and the patient context.

The task becomes harder as devices become connected and AI-enabled. Modern medical devices may combine sensors, embedded software, networked services, cloud components, robotic functions, and machine-learning models. AI-enabled functions add further concerns because behavior can depend on training data, operating conditions, changes in input data, model limits, and uncertain predictions. Reported recalls and post-market safety events for AI-enabled medical devices show that safety evidence must also be maintained after release [6]. The TruDi case illustrates this lifecycle concern: incorrect localization events and patient injuries were reported after AIsupported functionality had been introduced, while AI was not identified as the reported cause of the errors [7]. Such cases show that safety analysis must account for changing device behavior, clinical use, and field feedback.

These properties make medical-device safety analysis a strong but high-risk use case for LLM support. LLMs may help draft candidate hazards, causes, risk controls, safety requirements, and review questions. Yet for regulated medical-device work, useful support requires source evidence, weak-support checks, qualified expert review, and lifecycle update support. The next section reviews whether current LLM-assisted safetyengineering work provides these capabilities.

## III. STRUCTURED REVIEW OF LLM-ASSISTED SAFETY ENGINEERING

We conducted a structured review of recent work on LLMassisted safety engineering. The goal was to identify research patterns relevant for medical-device safety analysis, not to perform a full systematic review. We searched Scopus, IEEE Xplore, and Google Scholar, and checked the references of selected papers. Search terms combined LLM-related terms, such as Large Language Models, Generative AI, and Chat-GPT, with safety-engineering terms, such as hazard analysis, risk analysis, FMEA, FTA, STPA, HARA, Goal Structuring Notation (GSN), safety cases, assurance cases, and safety documentation. The search was last updated in May 2026.

Papers were included if they applied or discussed LLMs for safety-engineering tasks such as hazard analysis, risk analysis, FMEA, FTA, STPA, HARA, safety cases, safety requirements, or assurance-case review. Papers were excluded if they focused only on general AI safety, clinical diagnosis, cybersecurity without safety analysis, non-LLM safety automation, or general documentation support without a safety-engineering task. The initial search returned more than 200 papers. After title and keyword screening, duplicate removal, abstract screening, and use-case screening, 90 papers published between 2023 and 2026 remained. Due to the limited space in this paper, we have included the list of papers reviewed in an online repository [8].

TABLE I  
DISTRIBUTION OF REVIEWED PAPERS BY SAFETY TASK AND DOMINANT LLM METHOD
<table><tr><td rowspan=1 colspan=1>Safetyartifactor Task</td><td rowspan=1 colspan=1>Discus-sionPaper</td><td rowspan=1 colspan=1>FinetuningLLM</td><td rowspan=1 colspan=1>VectorData-base orRAG</td><td rowspan=1 colspan=1>Know-ledgeGraph</td><td rowspan=1 colspan=1>PromptEngine-ering</td><td rowspan=1 colspan=1>Total</td></tr><tr><td rowspan=1 colspan=1>FMEA</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>29</td></tr><tr><td rowspan=1 colspan=1>HARA</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1>FTA</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>AssuranceCasesandDefeater</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=1 colspan=1>STPA</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>GeneralSafetyAnalysis</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>11</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>90</td></tr></table>

Table I classifies the selected papers along two dimensions: the safety task addressed and the dominant LLM method used. The classification is based on the dominant method; most systems also use prompting. Discussion papers propose concepts or discuss opportunities and risks without a full implementation. Fine-tuning denotes papers that adapt an LLM using task-specific training data. Vector database or RAG denotes approaches that retrieve external information from stored documents and provide it to the LLM. Knowledge graph denotes approaches that represent domain and safety knowledge as entities and relations. Prompt engineering denotes approaches where prompting is the main support mechanism and no substantial external retrieval, fine-tuning, or structured knowledge layer is used.

When a paper used several LLM techniques, we assigned the dominant method based on the technique that carried the main safety-engineering contribution. Borderline cases were discussed among the authors and assigned to the category that best reflected the paper’s stated contribution. The reviewed work shows five main patterns.

First, many studies test feasibility through prompt engineering. They ask whether LLMs can identify hazards, failure modes, unsafe control actions, risk scenarios, safety requirements, or mitigations from a system description [9]–[11], [20]–[23], [26]. These studies show that LLMs can produce useful candidate outputs quickly. They also report recurring problems, including missed rare hazards, generic or duplicate entries, implicit assumptions, weak mitigations, and difficulties with structured output formats. Prompt-only use can therefore support early ideation, but it is not sufficient as the sole support method for regulated safety analysis.

Second, retrieval-based grounding is becoming more important. RAG and vector search allow LLMs to use information from prior safety analyses, failure reports, standards, manuals, and technical documents. Such approaches have been studied for performance-level estimation, FMEA generation, STPA support, and safety-requirements generation [12], [13], [24], [27]. Related medical and radiotherapy studies evaluate LLM support for FMEA-based risk management [14], [25]. This pattern is important for medical devices because devicespecific evidence is often private, technical, versioned, and absent from public model training data.

Third, some studies use structured safety knowledge through knowledge graphs. Knowledge graphs can represent failures, hazards, causes, components, functions, risk controls, and their relations. They have been studied for medical-device risk knowledge extraction and causal analysis in aerospace manufacturing [16], [17]. FMEA support for human-robot collaboration provides a related example of method-specific safety support [15]. Compared with plain document retrieval, knowledge graphs can make relations more explicit and may support links from generated safety items back to source evidence.

Fourth, recent work moves from single prompts to multistep and multi-agent pipelines. Some approaches split safety analysis into smaller tasks, such as hazard extraction, riskscenario generation, safety-requirement derivation, and testcase generation [19], [26]–[28]. Others use one model to generate candidate artifacts and another model to challenge them. This shift is useful because safety analysis is a process, not a single generation step. However, many such pipelines are still tested on limited examples and do not yet provide enough evidence for repeatability, traceability, or expert usefulness.

Fifth, LLMs are increasingly used as critics, especially in assurance cases. Several papers use LLMs to identify defeaters, meaning doubts or counterarguments to a safety claim [29]–[32]. This line of work is relevant because it treats LLMs not only as generators, but also as support for review and challenge. Similar critique mechanisms could help medical-device experts review candidate hazards, risk controls, residual-risk arguments, and safety-case fragments.

Overall, the literature provides useful starting points but does not yet solve the regulated medical-device problem. Current approaches are often method-specific, prompt-centered, and evaluated on public or small case studies. Only a small part of the literature addresses medical-device safety analysis directly [14], [16], [25]. The next section derives the resulting gap: LLM support must move from isolated safety-text generation to source-linked safety-knowledge support.

## IV. RESEARCH GAP: FROM GENERATED TEXT TO SOURCE-LINKED SAFETY KNOWLEDGE

The reviewed literature shows that LLMs can support safety-engineering tasks, but current approaches are not yet sufficient for regulated medical-device risk management. The main gap is not simply that LLMs can produce incorrect text. The deeper problem is that most approaches do not provide a controlled process for linking outputs to device evidence, flagging uncertainty, supporting expert review, and updating safety artifacts over the lifecycle.

TABLE II  
RESEARCH GAPS AND REQUIREMENTS FOR LLM-SUPPORTED MEDICAL-DEVICE SAFETY ANALYSIS
<table><tr><td rowspan=1 colspan=1>Gap</td><td rowspan=1 colspan=1>Relevancefor medicaldevices</td><td rowspan=1 colspan=1>Framework Requirement</td></tr><tr><td rowspan=1 colspan=1>WeakDomaingrounding</td><td rowspan=1 colspan=1>Device-specific    safetyknowledge        oftenprivate, product-specific,and absent from publicmodel training data.</td><td rowspan=1 colspan=1>Controlled retrieval fromcompany and device evi-dence.</td></tr><tr><td rowspan=1 colspan=1>LimitedHazardCoverage</td><td rowspan=1 colspan=1>Severe harm can arisefrom   rare,   indirect,or      interaction-basedscenarios.</td><td rowspan=1 colspan=1>Support systematic searchfor common, rare, andinteraction-based hazards.</td></tr><tr><td rowspan=1 colspan=1>LimitedTraceability</td><td rowspan=1 colspan=1>Safety artifacts must bejustified during review, au-dits, and change analysis.</td><td rowspan=1 colspan=1>Link each generated itemto source  documents,requirements,     designelements, risk controls,and evidence.</td></tr><tr><td rowspan=1 colspan=1>Unsupportedclaims andweakscoring</td><td rowspan=1 colspan=1>Plausible but unsupportedclaims or poor risk scorescan affect risk-control de-cisions.</td><td rowspan=1 colspan=1>Flag weak source support,uncertainty, inconsistency,and low-confidence out-puts.</td></tr><tr><td rowspan=1 colspan=1>Unclear hu-man role</td><td rowspan=1 colspan=1>LLMs must not replacequalified safety engineers.</td><td rowspan=1 colspan=1>Require expert review, ap-proval, change tracking,and recorded justification.</td></tr><tr><td rowspan=1 colspan=1>Weak evalu-ation</td><td rowspan=1 colspan=1>Public examples may over-lap with training data, andtext metrics are insufficientfor safety artifacts.</td><td rowspan=1 colspan=1>Use non-public or newlybuilt case studies, expertreferences, and safety-specific metrics.</td></tr><tr><td rowspan=1 colspan=1>Limitedlifecyclesupport</td><td rowspan=1 colspan=1>Risk-management   filesmust   change    withrequirements,  software,complaints, and post-market data.</td><td rowspan=1 colspan=1>Support artifact retrieval,impact analysis, proposedupdates, and re-approval.</td></tr></table>

Table II summarizes the resulting requirements for a controlled LLM support process. These requirements motivate the research questions and the framework in the next section.

A first challenge is domain grounding. Prompt-only use can support early brainstorming, but it is weak for medicaldevice safety analysis. Critical knowledge is often contained in requirements, software design documents, risk-management files, verification reports, complaints, and post-market reports. Public model training data is unlikely to contain this deviceand company-specific evidence. LLM outputs may therefore be plausible but weakly supported by the actual device.

A second challenge is hazard coverage. Medical-device safety analysis must consider common failures as well as rare, indirect, interaction-based, or context-dependent scenarios that may lead to severe harm. LLMs may miss such scenarios or produce duplicate hazards, generic failure descriptions, missing causal chains, or weak risk controls. The goal is not to prove complete hazard identification, but to improve coverage and make remaining uncertainty visible to reviewers.

A third challenge is traceability. Safety artifacts must be linked to requirements, design elements, risk controls, verification evidence, and residual-risk decisions. LLM-generated text without source links is difficult to review, justify during audits, or update when the device changes. Each generated safety item therefore needs an explicit link to the evidence and assumptions that support it.

A fourth challenge is hallucination and uncertainty. LLMs can produce plausible but unsupported or incomplete claims. This is especially critical when they assign severity, occurrence, detectability, risk priority numbers, or residualrisk judgments. Such values influence risk-control decisions and should be treated as candidates for expert review, not as model decisions.

A fifth challenge is the human role. Existing work often includes an expert either at the end of the process or after selected generation steps, but the expert role is not always defined clearly. In medical-device safety analysis, the LLM should assist qualified engineers, not replace them [23]. Experts must inspect source evidence before accepting generated hazards, causes, risk controls, risk scores, or safety arguments, and their decisions and justifications should be recorded.

A sixth challenge is evaluation. Public case studies are useful for early research, but they create a risk of evaluationdata contamination, where related material may have been present in model training data [34]. Common natural-language metrics such as BLEU [33] are also insufficient because they do not assess safety correctness, traceability, risk-control quality, or review value. Evaluation therefore needs expert reference analyses and safety-specific metrics.

A seventh challenge is lifecycle support. Medical-device safety analysis must be updated when requirements, architecture, software, risk controls, verification results, complaints, or post-market data change. Existing LLM studies mostly focus on fixed system descriptions. A useful framework must therefore retrieve affected items, propose updates, and require renewed expert review.

These challenges indicate the need for a framework that treats LLM outputs as candidate safety knowledge items, not as final safety-analysis results.

We use the term source-linked safety item for the central artifact produced, checked, reviewed, and updated in the framework. It is a candidate safety-relevant statement together with the source evidence, assumptions, checks, review status, and change information needed to assess it. For ISO 14971- oriented work, source-linked safety items can represent or support hazards, foreseeable sequences of events, hazardous situations, harms, risk-control measures, verification evidence, and residual-risk decisions.

Each source-linked safety item should contain at least the fields summarized in Table III. The table is not intended as a fixed data model. It defines a minimal structure that keeps generated safety content reviewable, traceable, and updatable instead of leaving it as untraceable text.

## V. FRAMEWORK FOR EVIDENCE-GROUNDED SAFETY-KNOWLEDGE SUPPORT

The discussed research gap leads to the following research questions. The questions define the requirements for an evidence-grounded LLM support framework for medicaldevice safety analysis. Table IV maps each question to the framework component that addresses it.

TABLE III CORE FIELDS OF A SOURCE-LINKED SAFETY ITEM
<table><tr><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Purpose</td><td rowspan=1 colspan=1>Example</td></tr><tr><td rowspan=1 colspan=1>Item ID</td><td rowspan=1 colspan=1>Stable reference for reviewand change tracking</td><td rowspan=1 colspan=1>HS-1</td></tr><tr><td rowspan=1 colspan=1>Item type</td><td rowspan=1 colspan=1>Hazard, hazardous situa-tion, failure mode, risk con-trol, claim, or review ques-tion</td><td rowspan=1 colspan=1>Hazardous situation</td></tr><tr><td rowspan=1 colspan=1>DeviceLink</td><td rowspan=1 colspan=1>Function, component, soft-ware item, use context, orpatient/user interaction</td><td rowspan=1 colspan=1>Occlusion detection andalarm function</td></tr><tr><td rowspan=1 colspan=1>Source evi-dence</td><td rowspan=1 colspan=1>Document,       section,version,     requirement,risk-control  record, orartifact link</td><td rowspan=1 colspan=1>Software requirement foralarm timing; prior FMEArow on occlusion; verifi-cation test for alarm la-tency; complaint record ondelayed alarm response.</td></tr><tr><td rowspan=1 colspan=1>Generatedstatement</td><td rowspan=1 colspan=1>Candidate safety-relevantstatement proposed by theLLM</td><td rowspan=1 colspan=1>Delayed occlusion alarmmay lead to continuedunder-infusion    withouttimely clinical response.</td></tr><tr><td rowspan=1 colspan=1>Rationale</td><td rowspan=1 colspan=1>Causal or method-specificreason why the item wasproposed</td><td rowspan=1 colspan=1>Achanged  detectionthreshold may increasealarm delay under low-flow conditions.</td></tr><tr><td rowspan=1 colspan=1>Assumptions</td><td rowspan=1 colspan=1>Conditions that require ex-pert review</td><td rowspan=1 colspan=1>Clinical staff rely on thedevice alarm as the mainsignal for occlusion re-sponse.</td></tr><tr><td rowspan=1 colspan=1>Uncertaintyflags</td><td rowspan=1 colspan=1>Weak source support, con-flict, duplicate, missing ev-idence, or model disagree-ment</td><td rowspan=1 colspan=1>Requires source check forlow-flow cases; possibleduplicate of existing occlu-sion hazard.</td></tr><tr><td rowspan=1 colspan=1>Reviewerdecision</td><td rowspan=1 colspan=1>Accepted, rejected, edited,merged, or split</td><td rowspan=1 colspan=1>Edit and merge with exist-ing hazardous situation.</td></tr><tr><td rowspan=1 colspan=1>Reviewerjustifica-tion</td><td rowspan=1 colspan=1>Short reason for importantreview decisions</td><td rowspan=1 colspan=1>Existing item covers oc-clusion, but alarm timingchange adds a new causalpath and affected verifica-tion evidence.</td></tr><tr><td rowspan=1 colspan=1>Changehistory</td><td rowspan=1 colspan=1>Later changes, affected ar-tifacts, and re-approval sta-tus</td><td rowspan=1 colspan=1>Linked to software re-quirement change and re-approval of alarm verifica-tion test.</td></tr></table>

The proposed framework defines a human-in-the-loop support process that addresses the research questions shown in Table IV. It is not a layered software architecture. Rather, it consists of process components connected by artifact and control-flow interfaces. The main artifact passed between the components is the source-linked safety item introduced in Section IV. The framework prepares, links, checks, and updates candidate safety artifacts for expert decision-making; final safety responsibility remains with qualified experts.

Figure 1 shows the process components and interfaces of the proposed framework. The figure should be read as a support process, not as an automated safety-assessment or compliance-checking pipeline. Evidence and lifecycle artifacts are ingested into a safety-knowledge store. Task and method setup then guide method-specific generation of source-linked safety items. Critique and uncertainty checks add review signals. Qualified experts accept, reject, edit, merge, or split candidate items and record justifications. Lifecycle changes trigger retrieval of affected items and renewed expert review.

TABLE IV RESEARCH QUESTIONS (RQ) AND CORRESPONDING FRAMEWORK COMPONENTS
<table><tr><td rowspan=1 colspan=1>RQ#</td><td rowspan=1 colspan=1>Research Question</td><td rowspan=1 colspan=1>FrameworkComponent</td></tr><tr><td rowspan=1 colspan=1>RQ1</td><td rowspan=1 colspan=1>Which medical-device lifecycle artifacts areneeded as evidence for LLM-supportedsafety analysis?</td><td rowspan=1 colspan=1>Evidence Intake</td></tr><tr><td rowspan=1 colspan=1>RQ2</td><td rowspan=1 colspan=1>How should device and safety knowledgebe represented and retrieved while preserv-ing source links and versions?</td><td rowspan=1 colspan=1>Knowledge stor-age and retrieval</td></tr><tr><td rowspan=1 colspan=1>RQ3</td><td rowspan=1 colspan=1>How can LLMs generate method-specificcandidate safety items for FMEA, STPA,FTA, assurance cases, and ISO 14971 risk-management work?</td><td rowspan=1 colspan=1>Method-specificgeneration</td></tr><tr><td rowspan=1 colspan=1>RQ4</td><td rowspan=1 colspan=1>How can unsupported, inconsistent, dupli-cate, or low-confidence items be detectedbefore expert review?</td><td rowspan=1 colspan=1>Critique and un-certainty checks</td></tr><tr><td rowspan=1 colspan=1>RQ5</td><td rowspan=1 colspan=1>How should expert review, approval, re-jection, correction, and justification berecorded?</td><td rowspan=1 colspan=1>Human review</td></tr><tr><td rowspan=1 colspan=1>RQ6</td><td rowspan=1 colspan=1>How can affected safety items be foundand updated after design changes, softwarechanges, complaints, or post-market feed-back?</td><td rowspan=1 colspan=1>Lifecycle update</td></tr><tr><td rowspan=1 colspan=1>RQ7</td><td rowspan=1 colspan=1>How can the framework be evaluated fairlyusing non-public or newly built medical-device cases and expert references?</td><td rowspan=1 colspan=1>Evaluation</td></tr></table>

## A. Evidence intake

The evidence intake component collects the artifacts needed for safety analysis, such as product descriptions, intendeduse statements, requirements, design and software documents, user information, risk-management files, verification evidence, complaints, post-market reports, and relevant standards. For medical devices, these artifacts should be aligned with ISO 14971 risk-management activities and, where applicable, IEC 62304 software lifecycle information.

The intake component should preserve document identity, version, source, date, and artifact type. This is important because later generated safety items must point back to the evidence used during generation and review. A generated risk control based on an outdated requirement should be treated differently from one based on the current approved requirement set.

## B. Safety-knowledge storage and retrieval

The safety-knowledge storage and retrieval component stores and retrieves device and safety knowledge. The collected artifacts may be represented in vector databases, knowledge graphs, structured tables, or hybrid forms. Vector search supports similarity-based retrieval from large text collections. Knowledge graphs make relations between hazards, causes, components, functions, risk controls, and evidence explicit. Structured tables preserve method-specific artifacts such as FMEA rows, risk-control records, and verification links.

![](images/4a0ff8a4a2cbac15c7cc9c5bc620ad2925b2574fbe0226dbbf0330ad8d36c23e.jpg)  
Fig. 1. Process components and interfaces of the proposed evidence-grounded LLM support framework.

The goal is not only to retrieve context for the LLM. The goal is to preserve source links between generated safety items and supporting evidence. Retrieval should therefore return not only text, but also artifact identifiers, versions, sections, and relation types. This makes later review and update analysis possible.

## C. Task setup

The task setup component defines the device or device function under analysis, the system boundary, the intended use, the use environment, relevant users, patient context, and selected safety method. The framework should not assume that one safety method fits all cases. FMEA may support componentlevel failure analysis. STPA may support unsafe interaction and control analysis. FTA may support top-event reasoning. Assurance cases may support structured argumentation. ISO 14971 provides the risk-management process context that connects these artifacts to risk-control decisions and residualrisk evaluation.

## D. Method-specific generation

The generation component uses the task definition and retrieved evidence to produce source-linked safety items. Depending on the selected method, these items may include hazards, hazardous situations, failure modes, causes, effects, risk controls, safety requirements, verification ideas, safetycase claims, or review questions. Each generated item should include links to the evidence used during generation and should record assumptions that require expert review.

For example, consider an infusion pump software change that modifies occlusion detection and alarm timing. The framework should retrieve related requirements, software items, previous FMEA entries, alarm-related risk controls, verification evidence, complaint data, and prior reviewer decisions. The LLM may then propose candidate updates such as a changed hazardous situation, an affected risk control, or a new verification question. These proposals are not accepted automatically. They become source-linked safety items that must pass critique checks and expert review.

## E. Critique and uncertainty checks

The critique component checks generated items before expert review. This can include duplicate detection, format checks, consistency checks, source-evidence checks, comparison across multiple LLM outputs, and generation of defeaters or review questions. If models disagree, if source support is weak, or if an item relies on an unstated assumption, the framework should flag it for careful review.

These checks cannot prove correctness. They are review aids. Their purpose is to help experts focus on uncertain, weakly supported, inconsistent, or safety-critical items. In medical-device risk management, this is especially important for risk scoring. Severity, occurrence, detectability, and residual-risk judgments should not be accepted merely because they are generated in a plausible form.

## F. Expert review and lifecycle updates

The expert review component is central to the framework. Generated safety items must be reviewed by qualified experts before they are accepted. The reviewer should inspect the source evidence and then accept, reject, edit, merge, or split generated items. The reviewer should also record short justifications for important decisions, especially for rejected hazards, accepted risk controls, residual-risk decisions, and changes to risk scores.

The same structure supports lifecycle updates. When requirements, software, design elements, risk controls, verification results, complaints, or post-market data change, the framework should identify affected source-linked safety items and propose updates. For example, a changed software requirement should trigger retrieval of linked hazards, affected risk controls, related verification evidence, and prior reviewer decisions. These proposed updates must again be reviewed and approved by experts.

The framework therefore shifts LLM use from one-time safety-text generation to lifecycle-oriented safety-knowledge support. Its novelty is not the isolated use of retrieval, knowledge graphs, LLM generation, critique, or expert review. These elements already appear in parts of the literature. The contribution is their integration around a common reviewable artifact: the source-linked safety item. This artifact binds generated safety content to source evidence, method context, assumptions, uncertainty flags, reviewer decisions, and lifecycle history. It therefore gives LLM support a structure that can be reviewed, audited, updated, and evaluated in regulated medical-device development.

## VI. EVALUATION STRATEGY

A useful evaluation must test whether the framework improves safety-engineering support without weakening expert control. We therefore propose an evaluation based on nonpublic or newly built medical-device case studies, expert reference analyses, and safety-specific metrics.

The case studies should include at least two device types with different safety characteristics. One case should be a software-intensive but non-AI device, such as an infusionpump function or alarm subsystem. A second case should include an AI-enabled or connected function, where evidence may include model limits, data assumptions, use conditions, post-market feedback, or changes in input distribution. The use of non-public or newly built cases reduces the risk that the LLM has already seen related safety material during training and helps limit evaluation-data contamination [34].

Expert reference analyses should be created by qualified safety and domain experts. These references should not be treated as perfect ground truth, but as review baselines. They should include accepted hazards or hazardous situations, causal explanations, risk controls, and verification links. Where experts disagree, the disagreement should be recorded rather than hidden, because such disagreement is common in safety analysis.

The evaluation should compare at least three configurations: prompt-only generation, retrieval-grounded generation, and the full framework with source-linked safety items, critique checks, and expert review. This comparison tests whether the additional process elements add value beyond basic LLM prompting. We propose the following metrics:

• Coverage: share of expert-reference items found by the framework.

• Correctness: share of generated items judged technically and safety-wise correct by experts.

• Relevance: share of generated items relevant to the selected device function and safety method.

• Duplicate rate: share of items that repeat existing items without adding useful detail.

• Source support: share of accepted items with sufficient source evidence. Source support can be rated by experts as direct support, partial support, no support, or contradiction.

• Unsupported-claim rate: share of generated items that make claims not supported by the provided evidence.

• Traceability: share of items linked to requirements, design elements, risk controls, verification evidence, or postmarket data.

• Review effort: expert time needed to accept, reject, edit, merge, or split generated items.

• Review usefulness: expert judgment of whether the framework helped identify, check, or update safety artifacts.

The full framework would be considered useful if, compared with prompt-only and retrieval-grounded baselines, it increases the share of relevant and source-supported safety items, reduces unsupported claims and duplicate items, improves traceability, and reduces expert search and consistencychecking effort. It should not be considered successful if it only increases the number of generated hazards without improving review quality or source support.

The proposed evaluation has several threats to validity. Expert reference analyses may be incomplete or reflect expertspecific judgment. Non-public or newly built case studies reduce training-data overlap, but cannot prove that no related knowledge was present in model training data. Review effort may also depend on reviewer experience, device familiarity, and the quality of the existing risk-management file. We therefore treat the evaluation as evidence of usefulness and review quality, not as proof of complete hazard identification or regulatory sufficiency.

## VII. CONCLUSION

This paper argued that LLM support for medical-device safety should not be framed as autonomous safety-text generation. Used in that narrow way, LLMs risk adding more unverified text to an already document-heavy process. A stronger role is source-linked safety-knowledge support: connecting device evidence, safety methods, critique checks, expert review, and lifecycle updates around reviewable safety items.

For the medical-device industry, the value hypothesis is specific. Manufacturers need to reuse internal knowledge from requirements, design files, risk files, verification reports, complaints, and post-market data. They also need to keep safety artifacts consistent when devices change. The proposed framework is intended to reduce search, drafting, and consistency-checking effort while keeping qualified experts responsible for final safety decisions. The intended industry benefit is strongest where manufacturers must maintain riskmanagement files across many device variants, software versions, and post-market feedback loops.

In the MedSafe project [35], we will implement and evaluate evidence intake, safety-knowledge retrieval, method-specific generation, critique and uncertainty checks, expert review, and lifecycle updates for selected safety-analysis and riskmanagement processes. The expected result is not an automatic safety assessor, but a support system for testing whether medical-device safety analysis can become more efficient, reviewable, traceable, and updatable while preserving human responsibility.

## REFERENCES

[1] International Organization for Standardization, ISO 14971:2019, Medical devices — Application of risk management to medical devices, 3rd ed. Geneva, Switzerland: ISO, 2019.

[2] International Electrotechnical Commission, IEC 62304:2006, Medical device software — Software life cycle processes. Geneva, Switzerland: IEC, 2006, including Amendment 1:2015 where applicable.

[3] DIHK, MedicalMountains, and SPECTARIS, Survey on the EU Medical Device Regulation, 2023.

[4] P. Gorczyca, D. Arndt, M. Diller, P. Kettmann, S. Mennicke, and H. Strass, “A farewell to harms: Risk management for medical devices via the Riskman ontology and shapes,” arXiv:2405.09875, 2024.

[5] J. Reich, J. Frey, E. Cioroaica, M. Zeller, and M. Rothfelder, “Argumentdriven safety engineering of a generic infusion pump with digital dependability identities,” in Model-Based Safety and Assessment, M. Zeller and K. Hofig, Eds. Cham, Switzerland: Springer, 2020, pp. 19–33,¨ doi: 10.1007/978-3-030-58920-2 2.

[6] Y. Ren, Y. Zheng, D. Windecker, A. G. Fraser, G. C. M. Siontis, and E. G. Caiani, “Clinical evidence and FDA recalls of artificial intelligenceenabled medical devices,” JAMA Network Open, vol. 9, no. 6, e2617920, 2026, doi: 10.1001/jamanetworkopen.2026.17920.

[7] Reuters, “AI enters the operating room. Reports arise of botched surgeries and misidentified body parts,” Feb. 9, 2026.

[8] Gangopadhyay, T., et al. (2026). List of SOTA papers referenced. Zenodo. https://doi.org/10.5281/zenodo.21346684

[9] D. Hillen, C. Helten, and J. Reich, “Towards LLM-augmented situation space analysis for the hazard and risk assessment of automotive systems,” in INFORMATIK 2024, Gesellschaft fur Informatik e.V., 2024,¨ pp. 709–714.

[10] A. Nouri, B. Cabrero-Daniel, F. Torner, H. Sivencrona, and C. Berger,¨ “Welcome your new AI teammate: On safety analysis by leashing large language models,” in Proc. IEEE/ACM 3rd Int. Conf. AI Engineering—Software Engineering for AI, 2024, pp. 172–177.

[11] S. Charalampidou, A. Zeleskidis, and I. M. Dokas, “Hazard analysis in the era of AI: Assessing the usefulness of ChatGPT4 in STPA hazard analysis,” Safety Science, vol. 178, Art. no. 106608, 2024.

[12] P. Iyenghar, “Evaluating LLM prompting strategies for industrial functional safety risk assessment,” in Proc. IEEE 8th Int. Conf. Industrial Cyber-Physical Systems, 2025, pp. 1–4.

[13] K. V. Charan, M. Singh, S. Redhu, and G. Soni, “A framework for automating failure modes and effects analysis (FMEA) using large language models (LLMs) and retrieval-augmented generation (RAG),” International Journal of System Assurance Engineering and Management, vol. 17, no. 5, pp. 1495–1509, 2026.

[14] A. Sarchosoglou, I. Genitsarios, N. Silvis-Cividjian, P. Papavasileiou, A. Bakas, N. Papanikolaou, and E. Pappas, “An AI-assisted, failure mode-based toolkit for proactive risk management in radiotherapy: A feasibility study,” Technical Innovations & Patient Support in Radiation Oncology, Art. no. 100404, 2026.

[15] M. J. Alenjareghi, S. Keivanpour, Y. A. Chinniah, and S. Jocelyn, “LLMdriven FMEA for safe human-robot collaboration in disassembly,” in Proc. IEEE 5th Int. Conf. Human-Machine Systems, 2025, pp. 295–301.

[16] W. Zhu, P. Zhang, W. Xia, Z. Gao, W. Li, R. Tian, and L. Wang, “AI-driven medical device risk management: A new paradigm integrating large language models and prompt engineering for standard-risk knowledge graph construction and application,” Risk Management and Healthcare Policy, pp. 1–17, 2026.

[17] B. Zhou, X. Li, T. Liu, K. Xu, W. Liu, and J. Bao, “CausalKGPT: Industrial structure causal knowledge-enhanced large language model for cause analysis of quality problems in aerospace product manufacturing,” Advanced Engineering Informatics, vol. 59, Art. no. 102333, 2024.

[18] O. Odu, A. B. Belle, and S. Wang, “LLM-based safety case generation for Baidu Apollo: Are we there yet?” in Proc. IEEE/ACM 4th Int. Conf. AI Engineering—Software Engineering for AI, 2025, pp. 222–233.

[19] L. Shi, B. Qi, J. Luo, Y. Zhang, Z. Liang, Z. Gao, et al., “Aegis: An advanced LLM-based multi-agent for intelligent functional safety engineering,” in Proc. 2024 Conf. Empirical Methods in Natural Language Processing: Industry Track, 2024, pp. 1571–1583.

[20] S. Diemert and J. H. Weber, “Can large language models assist in hazard analysis?” in Computer Safety, Reliability, and Security, Cham, Switzerland: Springer Nature Switzerland, 2023, pp. 410–422.

[21] Y. Hong, C. S. Timperley, and C. Kastner, “From hazard identification¨ to controller design: Proactive and LLM-supported safety engineering for ML-powered systems,” in Proc. IEEE/ACM 4th Int. Conf. AI Engineering—Software Engineering for AI, 2025, pp. 113–118.

[22] J. Lee, S. Park, S. Oh, and B. Ma, “Can large language models automate the HAZOP process without human intervention?” Safety Science, vol. 194, Art. no. 107039, 2026.

[23] P. Iyenghar, “Clever Hans in the loop? A critical examination of ChatGPT in a human-in-the-loop framework for machinery functional safety risk analysis,” Eng, vol. 6, no. 2, Art. no. 31, 2025.

[24] A. Raeisdanaei, J. Kim, M. Liao, and S. Kochhar, “An LLMintegrated framework for completion, management, and tracing of STPA,” arXiv:2503.12043, 2025.

[25] S. S. Nair, L. Court, R. Douglas, S. Gay, P. Govyadinov, T. Netherton, et al., “Use of large language models to enhance failure mode and effects analysis: A case study,” Advances in Radiation Oncology, Art. no. 102060, 2026.

[26] A. Nouri, B. Cabrero-Daniel, F. Torner, H. Sivencrona, and C. Berger,¨ “Engineering safety requirements for autonomous driving with large language models,” in Proc. IEEE 32nd Int. Requirements Engineering Conf., 2024, pp. 218–228.

[27] B. V. Balu, F. Geissler, F. Carella, J. V. Zacchi, J. Jiru, N. Mata, and R. Stolle, “Towards automated safety requirements derivation using agentbased RAG,” in Proc. AAAI Symposium Series, vol. 5, no. 1, 2025, pp. 299–307.

[28] P. Wang and J. Ma, “Risk2Scenario: LLM-assisted scenario generation for autonomous driving testing based on hierarchical risk analysis,” in Proc. 5th Int. Conf. Computer Systems, 2025, pp. 55–59.

[29] U. Gohar, M. C. Hunter, R. R. Lutz, and M. B. Cohen, “CoDefeater: Using LLMs to find defeaters in assurance cases,” in Proc. 39th IEEE/ACM Int. Conf. Automated Software Engineering, 2024, pp. 2262–2267.

[30] K. K. Shahandashti, A. B. Belle, M. M. Mohajer, O. Odu, T. C. Lethbridge, H. Hemmati, and S. Wang, “Using GPT-4 Turbo to automatically identify defeaters in assurance cases,” in Proc. IEEE 32nd Int. Requirements Engineering Conf. Workshops, 2024, pp. 46–56.

[31] T. Viger, L. Murphy, S. Diemert, C. Menghi, J. Joyce, A. Di Sandro, and M. Chechik, “AI-supported eliminative argumentation: Practical experience generating defeaters to increase confidence in assurance cases,” in Proc. IEEE 35th Int. Symp. Software Reliability Engineering, 2024, pp. 284–294.

[32] K. K. Shahandashti, M. Sivakumar, M. M. Mohajer, M. B. Belle, S. Wang, and T. C. Lethbridge, “Evaluating the effectiveness of GPT-4 Turbo in creating defeaters for assurance cases,” arXiv:2401.17991, 2024.

[33] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: A method for automatic evaluation of machine translation,” in Proc. 40th Annual Meeting of the Association for Computational Linguistics, 2002, pp. 311–318, doi: 10.3115/1073083.1073135.

[34] O. Sainz, J. A. Campos, I. Garc´ıa-Ferrero, J. Etxaniz, O. Lopez de Lacalle, and E. Agirre, “NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark,” in Findings of the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 10776–10787.

[35] Fraunhofer IESE, “MedSafe: Risk management for medical devices using AI,” 2026. URL: https://www.iese.fraunhofer.de/en/project/medsafe.html
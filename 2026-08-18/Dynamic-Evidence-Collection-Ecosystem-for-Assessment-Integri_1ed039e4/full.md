# Dynamic Evidence Collection Ecosystem for Assessment Integrity and Authentic Competence

Rajan Kadel NAPS, Melbourne, Australia Rajan.Kadel@naps.edu.au

Bellal Hossain NAPS, Sydney, Australia Bellal.Hossain@naps.edu.au

Samar Shailendra MIT, Melbourne, Australia SShailendra@mit.edu.au

Bushra Naeem NAPS, Sydney, Australia Bushra.Naeem@naps.edu.au

Abstract—Generative Artificial Intelligence (GenAI) can produce high-quality essays, code, and design artefacts, challenging the validity of conventional assessments that rely on single-point submissions and product-only grading. This paper proposes a design framework called “Dynamic Evidence Collection Ecosystem” that shifts assessment toward continuous, authentic, multi-source evidence of student learning over time. The framework collects process evidence through iterative artefacts, design logs, activity rounds, self-reflection, and peer collaboration, supported by an AI-enabled layer for learning analytics, formative feedback, and transparency. The approach is grounded in recent assessment-redesign scholarship in AI-rich contexts and aligned with contemporary views of authenticity in assessment. This paper builds on the hypothesis that academic integrity is strengthened when it is treated as an assessment design rather than as an AI detection problem. The tools have limitations and risks of use that carry academic penalties. This paper presents an implementation scenario to support institutional adoption.

Index Terms—Assessment redesign, dynamic evidence collection ecosystem, higher education, GenAI, process-based evidence

## I. INTRODUCTION

Generative Artificial Intelligence (GenAI) has accelerated long-standing tensions in educational assessment: the gap between what assessments intend to measure and the evidence they actually capture. As GenAI systems increasingly generate essays, code, reports, and design outputs, single-artifact submissions have become less reliable indicators of students competence. This has prompted calls to redesign assessment so that integrity becomes a feature of design. Recent work shows that educators are actively motivated to redesign assessments to maintain integrity, prepare learners for future work, and align with institutional policy and technological reality [1].

Although the sector’s first reaction was to implement AI-detection tools, their use is undermined by substantial reliability issues and ethical risks. Research evaluating AI-content detection tools reports inconsistent performance and high false positives, especially as model outputs improve and hybrid human–AI writing becomes more common [2]–[4]. Due to risks of false accusations and non-transparent methods, several universities have disabled AI-detection tools [5, 6].

Key issues for the assessment redesign are:

• Large Language Models make it easy to produce “final product”, creating a gap between correct submissions and genuine mastery of the underlying skills.

• This gap raises a key issue: how can we make the learning process visible to uphold academic integrity?

This article makes the following contribution:

• It proposes a framework for a Dynamic Evidence Collection Ecosystem, positioning it as a process-oriented response to GenAI-driven pressures on assessment, and

• It introduces “3-2-1 Portfolio Assessment” as an example for assessment design.

The paper is organised as follows: Section II reviews relevant literature and identifies the gap; Section III presents the proposed framework; Section IV demonstrates its application; and Section V concludes the study.

## II. RELATED WORKS

GenAI has significantly challenged the core principles underlying traditional assessment practices in higher education [7, 8]. Conventional assessment methods have primarily emphasised product-based evaluation, typically in the form of a single final submission such as an essay, report, or examination, a model that has long been considered a cornerstone of measuring student learning [8, 9]. The worth of such products has been significantly diminished by the capacity of GenAI tools to assist students in completing tasks such as essay writing, proposal writing, and take-home examinations, raising serious concerns about increased risks of cheating and the undermining of academic integrity [8, 10]. Therefore, students are increasingly able to bypass the learning process by submitting AI-generated work as their own, gaining an unfair advantage over peers [11]. Acknowledging this concern, Moorhouse et al. found, through a review of assessment guidelines from the world’s top-ranked universities, that the redesign of assessment is increasingly being encouraged, with a shift toward emphasising critical thinking and focusing on the learning process rather than on final outputs [12]. Furthermore, the authors in [13] recommend that institutions develop GenAI literacy initiatives, redesign assessments to prioritise critical thinking and creativity, and implement transparent, equitable policies governing GenAI use. Taken together, this body of evidence underscores that the conventional product-focused, single-submission model of assessment is no longer adequate in the GenAI era and requires fundamental redesign [12]–[14].

In response to the threats GenAI poses to academic integrity, many higher education institutions have turned to AI detection tools, such as Turnitin AI, GPTZero, and ZeroGPT, in an attempt to identify AI-generated content in student submissions. However, a growing body of peer-reviewed evidence demonstrates that these tools are fundamentally unreliable and unsuitable as a primary basis for determining disciplinary outcomes [3, 15]. Statistical experiments of testing the rates of false positive and false negative detection of AI-generated text, concluding that false positives represent a particularly serious concern and the current detection tools need major improvements before they can be reliably used in academic contexts [15]. Moreover, current AI detection tools are inaccurate and unreliable, often misclassifying AI-generated text as human-written, as reported in [3]. Furthermore, commercial tools that “humanise” AI-generated text make it easy to bypass detection, meaning weakness in detectors is quickly exploited, and the systems become even less reliable [16]. Alongside these technical challenges, AI content detectors often misclassify non-native English writing as AI-generated, raising significant concerns about fairness and the potential marginalisation of non-native English speakers in assessment and educational contexts [17]. Scholars broadly advocate for a fundamental redesign of assessment as a more pedagogically sound, equitable, and sustainable response to the challenges posed by GenAI [12, 15]. Therefore, there is an urgent need for assessment redesign in the era of GenAI.

![](images/b60575fe16faf6a661e9d95e84b0ca0c824f42b80df07193aa589fe2d3e8b078.jpg)  
Fig. 1: Framework for Dynamic Evidence Collection Ecosystem

## III. DYNAMIC EVIDENCE COLLECTION ECOSYSTEM

The literature review argues that traditional, single-submission assessments are increasingly vulnerable to AI misuse and a lack of visibility into the learning process. To address these concerns, we introduce a Dynamic Evidence Collection Ecosystem as a more authentic, robust alternative that supports learning visibility, academic integrity and professional readiness. Fig. 1 presents a three-component framework for authentic assessment.

## A. Driver: Technology Push

This component highlights why we need to do assessment redesign in the era of GenAI. The key driving factors are:

• Devaluation of Product: AI can generate an essay, report, design or code, the “single artefact” loses its validity as proof of learning. Therefore, there is a risk that students may misuse it. Moreover, AI detection tools are unreliable which suggests that policing AI use is neither scalable nor pedagogically sound [18, 19].

• Hidden Process Problem: Traditional assessment sees outcome but not the cognitive process, making it impossible to distinguish between student effort and AI output [20, 21].

## B. Solution: Dynamic Evidence Collection Ecosystem

The proposed ecosystem leverages multimodal data streams, utilises AI-enabled assessment strategies, and is seamlessly integrated within the Learning Management System (LMS). To achieve this multimodality, the framework categorises evidence into two streams:

1) Data-Driven Evidence: To restore assessment validity, the framework utilises a multi-layered collection of data-driven inputs, ranging from version-controlled technical artefacts to real-time learning analytics, ensuring a verifiable trail of student authorship, which tracks technical progression. Some examples of methods that can be used for the data-driven evidence are:

• Iterative Artefacts: Version-controlled submissions (e.g., draft-versions, prototype iterations, feedback cycles, GitHub commits) that demonstrate incremental development of the product that captures the process. These iterative artefacts may be more suitable for designing and developing the product, but can be used for report writing.

• Activity Rounds: Time-bound, repetitive cycles of task execution that provide longitudinal visibility. For example, concept proposal, draft development feedback, revision and improvement, and final artefact. The activity rounds may be suitable for problem-based or project-based tasks.

• Design Logs: Granular documentation of technical decisions and problem-solving steps. For example, design decisions and rationale, problems encountered and solutions, iterations and design changes, and analysis of tools and resources used. This evidence may be more suitable for design tasks.

• Prompting Interactions: Documented histories of student-AI dialogue to ensure transparency in AI-assisted workflows. Students distinguish their contributions from those of AI.

2) Human-Driven Evidence: Complementing the data-driven evidence, human-driven evidence provides the human-in-the-loop verification necessary to confirm conceptual mastery presented in the product [22].

• Reflective Journal: Metacognitive accounts where students justify their design choices and learning trajectory while developing the products (design, report, project, product, GenAI usage and any other outcomes).

• Peer Feedback (Scaffolded with GenAI): Evidence of social learning and interaction with GenAI tools. Evaluate own and peer contributions within a team-based environment.

• Oral Defence: Direct, real-time verification of authorship and conceptual understanding through spoken inquiry.

Balancing of those two types of evidence in the assessment is critical for learning process visibility. These two types of evidence should support each other, and educators should evaluate both pieces of evidence together. Once gathered, these evidence streams are synthesised through a centralised technological hub, where AI acts as a facilitative partner.

TABLE I: Mapping Evidences to Their Purposes, Applications, Examples, and Measured Skills.
<table><tr><td>Evidence</td><td>Purpose</td><td></td><td>Applications</td><td>Examples</td><td>Measured Skill(s)</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Iterative design</td><td colspan="2" rowspan="2">Captures the evolution of design thinking, documenting iterative refinement,</td><td rowspan="2">Product development, software engineering,</td><td rowspan="2">Computer-Aided (CAD) models;</td><td rowspan="2">Design network</td><td rowspan="2">Complex systems</td><td rowspan="2">design; Iterative solution development and refinement;</td></tr><tr><td></td></tr><tr><td>artefacts Version</td><td>informed decision-making, and structured problem-solving</td><td>Provides a transparent and traceable account</td><td>and complex problem-solving Computer science,</td><td>designs; technical drawings Code commits;</td><td>circuit schematics; version</td><td>Decision making; Problem solving Proficiency in</td><td>Design thinking; engineering tools;</td></tr><tr><td>controlled records Simulation</td><td>of the development process: individual contributions, decision-making, and changes</td><td>Demonstrates analytical reasoning through</td><td>software engineering, and data science Engineering, physics,</td><td>histories; debugging feature iterations Simulation results; parameter</td><td>logs;</td><td>individual contribution; development practices Problem analysis; solution testing and</td><td>systematic</td></tr><tr><td>&amp; modelling outputs Experiment</td><td>structured experimentation, evaluation, and validation of solutions Captures scientific</td><td>reasoning through</td><td>networking, and AI/ML domains Physics, chemistry,</td><td>tuning records; performance evaluations Experimental setups; recorded</td><td>proficiency Scientific</td><td>evaluation; analytical thinking; modelling inquiry;</td><td>methodical</td></tr><tr><td>logs Design decision logs</td><td>documentation of experimental observations, and iterative investigation Demonstrates</td><td>design, engineering judgement through structured documentation of</td><td>biology, and engineering laboratory contexts Systems engineering, network design, and AI</td><td>observations; data; troubleshooting records Architectural algorithm selection; trade-off</td><td>measurement decisions;</td><td>experimentation; proficiency; data analysis skills Decision-making;</td><td>experimental justification of</td></tr><tr><td>Prototype development</td><td>decisions, trade-offs, and rationale Demonstrates the application of theoretical</td><td>knowledge through the design, construction,</td><td>system development Robotics, product design, and software engineering</td><td>analyses Hardware prototypes; software</td><td></td><td>technical trade-offs; judgement; critical thinking Implementation</td><td>engineering proficiency; practical</td></tr><tr><td>Project milestones</td><td>and iterative refinement of practical solutions Enables continuous monitoring of learning</td><td></td><td>contexts Capstone projects and design</td><td>prototypes; staged implementations Project proposals; progress</td><td>system Project</td><td>engineering skills; application; planning;</td><td>knowledge engineering</td></tr><tr><td>Digital portfolio</td><td>progression through structured checkpoints and staged deliverables Provides a comprehensive and integrated</td><td>competence development,</td><td>engineering projects Any program featuring project-based or</td><td>reports; testing reports; final deliverables Collections of reports, design</td><td></td><td>management; project skills; development trajectory Demonstration of integrated professional</td><td>management</td></tr><tr><td>Reflective learning</td><td>record of showcasing skills and learning progression Supports metacognitive development through</td><td>structured reflection on learning processes,</td><td>experiential learning Problem-based learning tasks; capstone projects;</td><td>artefacts, simulations, and reflective entries Regular reflective</td><td>entries; Lifelong</td><td>competency learning;</td><td>competence; support for program-level self-assessment;</td></tr><tr><td>journals Peer</td><td>decision-making, and progression over time Evaluates teamwork and collaborative</td><td></td><td>collaborative group work Team-based projects in</td><td>documented problem-solving strategies; use of GenAI tools Peer feedback, contribution</td><td></td><td>professional growth</td><td>professional responsibility; continuous</td></tr><tr><td>collaboration</td><td>effectiveness interactions, contributions, and engagement</td><td>through documented</td><td>engineering, computing, and science disciplines</td><td>statements, records</td><td>team meeting</td><td>Teamwork; contribution</td><td>communication skills; collaborative effectiveness; individual</td></tr></table>

3) AI-Enabled Assessment: The framework utilises AI tools to provide “Process Feedback” and “Learning Analytics” that support the assessment cycle. This ensures that AI enhances the educator’s ability to interpret student progress without compromising trust. This framework also captures students’ AI-use disclosure & attribution. An AI-enabled assessment strategy can be ensured by the following actions:

• AI-Use Disclosure & Attribution: Clear frameworks for students to document how, where, and why AI was utilised in their creative or technical process.

• Learning Analytics: Automated tracking of engagement patterns to identify students who is struggling or whose work patterns deviate significantly from their established norms.

• Process Feedback: AI-generated formative feedback provided during initial stages to guide students before final submission.

4) Technological Integration: To manage the high volume of multimodal data, the ecosystem requires a robust technological infrastructure that centralises evidence from various sources and provides the outcomes in a suitable form. This technological integration should support aggregating evidence using analytics; it tracks learning over time and supports feedback loops [23,

24]. This process can be implemented by:

• ePortfolio Platform: A longitudinal repository where students curate their iterative artefacts, reflective journals, and video records into a cohesive narrative of competence.

• LMS Dashboard: A centralised visualisation tool for educators that aggregates data from the LMS, assessment works, design logs, and peer interactions to provide a “heat-map” of class-wide and individual progress.

By triangulating these multimodal data streams, the ecosystem establishes a robust foundation for verifying authentic competence and maintaining academic integrity.

## C. Outcome: Integrity and Authentic Competence

By strengthening evidence collection, the redesigned assessment improves learning visibility, academic integrity, and professional readiness. Educators can see how learning unfolds through continuous, contextual evidence that makes the learning process visible using the proposed framework. The ecosystem directly addresses hidden process concerns by providing continuous, observable traces of learning. This supports educators’ ability to make judgements about growth, decision-making, and engagement.

It is far more difficult to fabricate a sustained, multi-step process because integrity is built into a design that requires ongoing, authentic engagement. Academic integrity becomes an integral part of assessment design in this approach, as fabrication across both data-driven and Human-driven is considerably more difficult than generating a single polished product. This shifts integrity from reactive detection to proactive assessment architecture, consistent with literature emphasising process-based redesign for integrity in AI-rich environments.

TABLE II: 3-2-1 Portfolio Assessment.
<table><tr><td rowspan=1 colspan=1>Component (Evidence Type)</td><td rowspan=1 colspan=1>Description (Timeline)</td><td rowspan=1 colspan=1>Weight</td></tr><tr><td rowspan=1 colspan=1>Three (3) iterations (Data-driven)</td><td rowspan=1 colspan=1>Versions (v1, v2, v3) with change notes and reasoning for the changes (Initial, Mid and Final)</td><td rowspan=1 colspan=1>40%</td></tr><tr><td rowspan=1 colspan=1>Two (2) critiques (Human-driven)</td><td rowspan=1 colspan=1>Two reflections/peer-reviews that support iterations/changes (Mid and Post development)</td><td rowspan=1 colspan=1>30%</td></tr><tr><td rowspan=1 colspan=1>One (1) defence (Human-driven)</td><td rowspan=1 colspan=1>Oral explanation, viva for defence that supports iterations and critique (Post development)</td><td rowspan=1 colspan=1>30%</td></tr></table>

The dynamic evidence collection ecosystem mirrors real-world practice, including product iteration, self-reflection, collaboration, and tool use, including AI usage. The ecosystem reflects professional work practices in many fields, where competence is demonstrated through iterative development, documentation, collaboration, and the responsible use of tools, which will enhance employability skills and job readiness, consequently, professional readiness in students.

Table I illustrates various evidence, their purposes, applications, examples and the mapping of measured skills achieved by evidence. Most of the evidence presented in the table is suitable for STEM education. However, a similar table can be created by educators in any discipline, utilising their expertise. Educators need to create a similar table, covering appropriate evidence for each subject, suited to the discipline. Then, educators need to develop a strategy to prepare assessments utilising the table.

Educators reflect and evaluate on learning visibility [25, 26], academic integrity and professional readiness [27] impacted by the ecosystem using suitable techniques. The findings from the analysis are provided to the ecosystem as feedback for further improvement on evidence collection, strategies for using AI in assessment, and technological integration. Therefore, this ecosystem transforms assessment from static to dynamic and emphasises the process of observing learning visibility.

## IV. 3-2-1 PORTFOLIO ASSESSMENT

This section translates the ecosystem into an implementable assessment design. Educators replace “one-and-done” major assessment submission with a sequence of assessable artefacts, reducing dependence on any single product and strengthening validity through triangulation [1, 28, 29]. The assessment distributes evidentiary weight across time & sources. Table II shows one assessment example, called “3-2-1 Portfolio Assessment”, that shows components with types of evidence, a short description and possible weights. “3” refers to three-iteration during product development to observe the development progress, “2” refers to two critique documents covering reflections (self or group) and/or peer reviews completed during the product development, and “1” refers to one clearly demonstrated defence. Educators can vary the portfolio according to the discipline, but there is a need to balance two evidence types to observe the learning visibility.

A three-iteration scenario can encompass various types of assessments, and the assessment can be done in a group or individually. In this process, students documented changes, including how the formative feedback was addressed and the reasons for changes and decisions. Students need to declare their use of AI tools using AI-use disclosure & attribution. The AI use statement should cover: a) What AI tool was used?, b) What are the purposes of the tools?, and c) What was changed by the student from the AI output? This process is critical for AI transparency. This supports authenticity in assessment by valuing contextual reasoning and evaluative judgement rather than just polished outputs. It also aligns with assessment redesign guidance that prioritises skills (ethical decision-making and trade-off reasoning), which are less easily replicated by GenAI. Sample examples for the three-iteration cases are provided for clarity:

• Design-based problem, such as system design, or software design requires students to maintain a decision ledger (chosen approach → options considered → decision) for three stages. They are also required to maintain a design ledger to capture logs that record decisions, design constraints, and evidence.

• Project-based or problem-based task where students are either working in a group or individually required to demonstrate progress at three checkpoints (scope, & plan → prototype (design) → final product) and also address how the feedback is addressed from one checkpoint to another and document how changes are made.

• Report (essay) writing where students maintain their report writing versions (proposal → draft → final), where students record the changes. Students outline the reasons for the changes and address how the feedback is considered.

Additionally, the portfolio includes two (2) critique documents (reflections and/or peer reviews, depending on the suitability and effectiveness) and one (1) oral defence work. Peer collaboration and reflections should support authenticity and practice-based learning and must be directly linked with the three-iteration stages. Therefore, critique documents must be structured and assessed to ensure that students understand how the quality of feedback and uptake of critique is handled, and how the group and individual progress. Reflection should be directly linked with the activities associated with the three iterations, the progress made and the plan. Structured peer evidence provides a practical mechanism for ensuring consistent, transparent, and reliable evaluation by guiding educators to comment on the same critical aspects of a work, thereby improving the quality and comparability of feedback.

Finally, the oral defence provides a direct, real-time verification of authorship and conceptual understanding through spoken inquiry. It allows examiners to probe reasoning, clarify ambiguities, and assess the candidate’s ability to articulate and defend their work independently under questioning. The portfolio approach is consistent with systematic recommendations for multi-stage, process-based assessment to mitigate academic integrity risks in AI-infused environments. Educators need to evaluate the three components together by triangulating these multimodal data streams to establish a robust foundation for verifying competence. If there are any concerns about any evidence or missing components, there is a need for further investigation for academic integrity.

This work is conceptual and lacks empirical validation. Technological and governance requirements are not fully specified, which may affect scalability and adoption. Future work should address these limitations through pilot testing.

## V. CONCLUSION

GenAI poses significant challenges to the validity of assessment systems that rely on single-point submissions and product-focused grading. Emerging evidence demonstrates that AI-detection technologies lack the reliability required for high-stakes academic integrity decisions. These limitations underscore the need for assessment redesign that foregrounds learning visibility, supports integrity through process-based evidence, and aligns more closely with contemporary professional expectations. This paper introduces a Dynamic Evidence Collection Ecosystem as a practical design response to current assessment challenges. The proposed approach conceptualises evidence collection as a continuous, multi-source record of learning, supported by formative analytics and strengthened through transparency norms. Realising this ecosystem requires a commitment to transparent governance and a streamlined evidence-based system to address concerns around a product-only approach, while ensuring that the increased visibility of the learning process does not result in prohibitive educator workloads. The ecosystem positions the assessment task as requiring both data-driven and human-driven forms of evidence to enhance process visibility, thereby promoting academic integrity and professional readiness.

## REFERENCES

[1] Z. N. Khlaif, W. A. Alkouk, N. Salama, and B. Abu Eideh, “Redesigning assessments for AI-enhanced learning: A framework for educators in the generative AI era,” Education Sciences, vol. 15, no. 2, p. 174, 2025.

[2] Y. Sun, Y. Liao, and X. Ma, “Trusting AI to detect AI? A systematic evaluation of the reliability and robustness of current AIGC detection tools for student academic work,” Computers & Education, 2026.

[3] D. Weber-Wulff, A. Anohina-Naumeca, S. Bjelobaba, T. Foltynek,\` J. Guerrero-Dib, O. Popoola, P. Sigut, and L. Waddington, “Testing<sup>ˇ</sup> of detection tools for AI-generated text,” International Journal for Educational Integrity, vol. 19, no. 1, pp. 1–39, 2023.

[4] M. Hadra, K. Cambridge, and M. Mesbah, “Evaluating the accuracy and reliability of AI content detectors in academic contexts,” International Journal for Educational Integrity, vol. 22, no. 1, p. 4, 2026.

[5] Curtin University. (2026) GenAI use at Curtin. Accessed 8 April 2026. [Online]. Available: https://www.curtin.edu.au/students/study-support/ genai-use-at-curtin

[6] University of Waterloo. (2025, Sep.) Discontinuing use of AI detection functionality in Turnitin. Office of the Associate Vice-President, Academic. Accessed 8 April 2026. [Online]. Available: https://uwaterloo.ca/associate-vice-president-academic/ discontinuing-use-ai-detection-functionality-turnitin

[7] D. R. Cotton, P. A. Cotton, and J. R. Shipway, “Chatting and cheating: Ensuring academic integrity in the era of ChatGPT,” Innovations in Education and Teaching International, vol. 61, no. 2, pp. 228–239, 2024.

[8] Q. Xia, X. Weng, F. Ouyang, T. J. Lin, and T. K. Chiu, “A scoping review on how generative artificial intelligence transforms assessment in higher education,” International Journal of Educational Technology in Higher Education, vol. 21, no. 1, p. 40, 2024.

[9] T. K. Chiu, “Future research recommendations for transforming higher education with generative AI,” Computers and Education: Artificial intelligence, vol. 6, p. 100197, 2024.

[10] M. Belkina, S. Daniel, S. Nikolic, R. Haque, S. Lyden, P. Neal, S. Grundy, and G. M. Hassan, “Implementing generative AI (GenAI) in higher education: A systematic review of case studies,” Computers and Education: Artificial Intelligence, vol. 8, p. 100407, 2025.

[11] J. Luo, “A critical review of genai policies in higher education assessment: A call to reconsider the “originality” of students’ work,” Assessment & Evaluation in Higher Education, vol. 49, no. 5, pp. 651–664, 2024.

[12] B. L. Moorhouse, M. A. Yeo, and Y. Wan, “Generative AI tools and assessment: Guidelines of the world’s top-ranking universities,” Computers and Education Open, vol. 5, p. 100151, 2023.

[13] N. J. Francis, S. Jones, and D. P. Smith, “Generative AI in higher education: Balancing innovation and integrity,” British Journal of Biomedical Science, vol. 81, p. 14048, 2025.

[14] J. Crawford, M. Cowling, and K.-A. Allen, “Leadership is needed for ethical ChatGPT: Character, assessment, and learning using artificial intelligence (AI),” Journal of University Teaching and Learning Practice, vol. 20, no. 3, pp. 1–19, 2023.

[15] D. Dalalah and O. M. Dalalah, “The false positives and false negatives of generative AI detection tools in education and academic research: The case of ChatGPT,” The International Journal of Management Education, vol. 21, no. 2, p. 100822, 2023.

[16] C. G. Ardito, “Generative AI detection in higher education assessments,” New Directions for Teaching and Learning, vol. 2025, no. 182, 2025.

[17] W. Liang, M. Yuksekgonul, Y. Mao, E. Wu, and J. Zou, “GPT detectors are biased against non-native English writers,” Patterns, vol. 4, 2023.

[18] A. M. Elkhatat, K. Elsaid, and S. Almeer, “Evaluating the efficacy of AI content detection tools in differentiating between human and AI-generated text,” International Journal for Educational Integrity, vol. 19, no. 1, 2023.

[19] V. S. Sadasivan, A. Kumar, S. Balasubramanian, W. Wang, and S. Feizi, “Can AI-Generated Text be Reliably Detected?” arXiv preprint arXiv:2303.11156, 2023.

[20] Y. Zhan, D. Boud, and Z. Du, “Designing for authentic assessment: a scoping review,” Higher Education, pp. 1–18, 2025.

[21] R. Ajjawi, J. Tai, M. Dollinger, P. Dawson, D. Boud, and M. Bearman, “From authentic assessment to authenticity in assessment: broadening perspectives,” Assessment & Evaluation in Higher Education, vol. 49, no. 4, pp. 499–510, 2024.

[22] D. C. Fajardo-Ramos, A. Chiappe, and J. Mella-Norambuena, “Human-in-the-loop assessment with AI: implications for teacher education in Ibero-American universities,” in Frontiers in Education, vol. 10. Frontiers Media SA, 2025, p. 1710992.

[23] P. Kannan and D. Zapata-Rivera, “Facilitating the use of data from multiple sources for formative learning in the context of digital assessments: informing the design and development of learning analytic dashboards,” in Frontiers in Education, vol. 7. Frontiers Media SA, 2022, p. 913594.

[24] C. Martinez, R. Serra, P. Sundaramoorthy, T. Booij, C. Vertegaal, Z. Bounik, K. Van Hastenberg, and M. Bentum, “Content-focused formative feedback combining achievement, qualitative and learning analytics data,” Education Sciences, vol. 13, no. 10, p. 1014, 2023.

[25] J. Hattie, Visible learning: A synthesis of over 800 meta-analyses relating to achievement. routledge, 2008.

[26] D. B. Guruge and R. Kadel, “Towards an Holistic Framework to Mitigate and Detect Contract Cheating within an Academic Institute—A Proposal,” Education Sciences, vol. 13, no. 2, p. 148, 2023.

[27] P. Orr, L. Forsyth, C. Caballero, C. Rosenberg, and A. Walker, “A systematic review of Australian higher education students’ and graduates work readiness,” Higher Education Research & Development, vol. 42, no. 7, pp. 1714–1731, 2023.

[28] P. D. Ncube, G. P. Dzvapatsva, C. Matobobo, and M. M. Ranga, “Redefining student assessment in AI-infused learning environments: a systematic review of challenges and strategies for academic integrity,” AI and Ethics, vol. 6, no. 1, p. 68, 2026.

[29] S. K. Banihashem, D. Gaseviˇ c, O. Noroozi, H. Jarodzka, D. Joosten-ten´ Brinke, and H. Drachsler, “Optimizing formative assessment with learning analytics,” Review of Educational Research, 2025.
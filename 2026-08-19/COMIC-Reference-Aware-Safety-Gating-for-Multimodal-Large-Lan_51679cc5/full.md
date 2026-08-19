# COMIC: Reference-Aware Safety Gating for Multimodal Large Language Models

Md Abdullahil Oaphy, Anhao Xiang, Zongxing Xie, Huayue Gu, Chenyu Wang, and Honghui Xu

Kennesaw State University, Kennesaw, GA, USA

moaphy@students.kennesaw.edu, axiang@kennesaw.edu, zxie1@kennesaw.edu,

hgu2@kennesaw.edu, cwang38@kennesaw.edu, hxu10@kennesaw.edu

Abstract—Multimodal large language models (MLLMs) are increasingly used to interact with screenshots, scanned documents, diagrams, and other visually grounded inputs. This shift introduces a new safety risk. In many multimodal jailbreaks, neither the prompt nor the image is harmful in isolation. Unsafe behavior emerges only when the model binds an apparently benign operation, such as summarizing, translating, or following, to a localized visual target. This reveals a structural weakness in current multimodal defenses, which largely moderate the prompt–image pair as a whole even though the true security-relevant unit is the grounded operation–target pair produced during dereference. In this work, we identify and analyze this reference-dependent failure mode. We show that existing defenses degrade when harmful semantics are localized, activated only after grounding, and dependent on visual reference resolution. To address this problem, we propose COMIC (Context-Operation-Modality-Image-Classifier), a reference-aware pre-generation safety gate for MLLMs. COMIC first infers the requested operation and reference type. It then constructs candidate targets from OCR and open-vocabulary proposals, grounds plausible referents, and evaluates safety over explicit operation–target pairs. To handle ambiguity conservatively, COMIC combines max-risk aggregation with quality-aware routing before deciding whether to forward or block the request. We evaluate COMIC across multiple open-source MLLMs, localized and broader multimodal jailbreak benchmarks, and benign reference-sensitive settings. The results show that COMIC consistently improves robustness while preserving benign utility and practical efficiency. More broadly, our findings suggest that multimodal safety cannot be enforced reliably without modeling the requested operation, the visual target it applies to, and the confidence of that grounding. Our code is available at: https://anonymous.4open. science/r/COMIC-E15D/.

## I. INTRODUCTION

Multimodal large language models (MLLMs) are rapidly becoming a general interface for visually grounded computing. Users now interact with screenshots, scanned documents, charts, forms, diagrams, and mobile interfaces through naturallanguage instructions, expecting models to summarize, translate, explain, extract, and reason over visually rich inputs with little friction [1], [2], [3], [4]. This shift has expanded the practical value of AI systems, but it has also enlarged their attack surface. In multimodal interaction, the effective input is no longer the prompt alone. It is the joint composition of language, pixels, embedded text, layout, and reference structure, all of which shape what the model understands and ultimately does.

A central challenge is that many multimodal requests are not semantically complete at the language level. Instructions such as “translate this,” “follow step 3,” “summarize the highlighted part,” or “what should I do next?” may appear harmless in text alone. However, they remain operationally incomplete until the system resolves what the user is referring to in the image. The model must identify a region, text span, interface element, diagram node, or other localized target before it can act. As a result, multimodal safety is governed not only by the prompt or image in isolation, but by an operation– target binding, meaning what action the user requests and which grounded visual target that action is applied to. A request that seems benign in language can therefore become unsafe only after it is grounded to a localized visual region that contains procedural, instruction-like, or otherwise policyviolating content [5], [6], [7], [8].

![](images/01db9c2029229ec6a450eea24b225d19701afb37d756f7bbdfeda791406ecc14.jpg)  
Fig. 1. Multimodal LLM safety failure and COMIC motivation.

Fig. 1 illustrates why this problem is difficult for conventional MLLM defenses. In Case 1, the harmful intent is explicit in the text prompt, while the image is visually benign. A textaware safety mechanism can reject this request because the unsafe semantics are already exposed in language. In Case 2, the prompt appears benign, but the harmful content is visually explicit in the image. A defense that separately screens image content may still detect the risk and refuse. Case 3 is more challenging because neither the prompt nor the image alone exposes the full unsafe request. Instead, the prompt asks the model to read a localized visual phrase and insert it into an incomplete action description. The unsafe intent emerges only after the model resolves the visual reference and composes the grounded text with the surrounding prompt. The final row shows the intended behavior of COMIC. After resolving the localized reference, COMIC evaluates the grounded operation–target pair rather than the text or image channel alone, and blocks the request before generation when the grounded action is unsafe.

This creates a practical and under-addressed safety problem. We consider an inference-time attacker who controls the prompt and image but does not modify model weights, system prompts, or training data. The image may appear benign while hiding harmful semantics in a small text overlay, boxed phrase, numbered step, diagram node, or ordinarylooking interface element. The prompt then asks for a routine action such as extracting, summarizing, rewriting, translating, or explaining that localized content. The attack succeeds if the safety mechanism judges the full prompt–image pair as benign, while the downstream MLLM resolves the reference, binds the request to the adversarial region, and generates unsafe content.

In such cases, the true security boundary is not the holistic multimodal input. It is the point at which user intent is bound to image-local evidence [9], [10], [11], [12]. This perspective reveals a structural weakness in many existing defenses. A common strategy is to moderate the full prompt–image pair using a single global classifier. Another is to filter the final output after the model has already committed to an interpretation. A third approach converts the image into text through OCR or captioning and then applies safety checks to the extracted description [13], [14], [15], [16]. These approaches can help when harmful intent is globally explicit, as in Case 1 or Case 2 of Fig. 1. However, they are fundamentally mismatched to attacks like Case 3, where the unsafe meaning is localized, reference-dependent, and activated only after grounding. The relevant question is not whether unsafe content exists somewhere in the image, nor whether the prompt appears suspicious in isolation. The relevant question is whether the system is about to apply a particular operation to a particular grounded target, and whether that grounded action should be allowed.

To address this gap, we argue for reference-aware pregeneration gating. Instead of making one holistic decision over the multimodal input, the safety mechanism should first determine whether the prompt depends on localized visual reference, infer the requested operation, ground the intended referent, and evaluate whether performing that operation on that target is safe. When no localized reference is involved, simpler global moderation may suffice. Once the request becomes reference-dependent, however, safety must be enforced at the dereference step itself. In other words, multimodal moderation should move from whole-input screening to operationconditioned, target-grounded policy enforcement [17], [18], [19].

To instantiate this idea, we propose COMIC (Context– Operation–Modality–Image–Classifier), a reference-aware pre-generation safety gate for MLLMs. Given a prompt– image pair, COMIC first infers the requested operation and the reference type expressed in language. It then constructs plausible target candidates from OCR-derived text regions and open-vocabulary visual proposals, grounds likely referents through semantic and spatial matching, and evaluates safety over explicit operation–target pairs rather than over the global input alone. COMIC outputs a binary decision in $\begin{array} { c c l } { Y } & { = } & { \{ \mathsf { s a f e } , \mathsf { u n s a f e } \} } \end{array}$ , which determines whether the request is forwarded to the downstream MLLM or blocked before any generation occurs.

A key design choice in COMIC is that a request is labeled safe only when two conditions hold simultaneously. The aggregated unsafe risk over plausible grounded targets must be low, and the perceptual evidence supporting that grounding must be reliable enough to justify forwarding. If multiple candidates plausibly satisfy the user’s reference, COMIC retains several high-scoring targets and aggregates risk conservatively, so that a single plausible unsafe target is sufficient to block the request. If proposal coverage is weak, OCR is noisy, or grounding remains ambiguous, COMIC does not treat uncertainty as evidence of benignness. Instead, it narrows the region in which a safe decision is admissible and falls back to blocking. This conservative design reflects an important operational reality, since damaging multimodal failures often occur when a system proceeds without enough evidence to localize risk correctly.

This framing also clarifies the safety–utility tradeoff. A trivial defense can reduce attack success by blocking everything, but such a system is unusable in practice. The goal is not indiscriminate refusal. The goal is to reduce attack success because the system localizes the semantics that govern harm, while still preserving benign functionality. COMIC therefore targets the safety–utility frontier. Better proposal coverage and sharper grounding reduce false-safe failures on adversarial inputs while preserving benign utility. At the same time, difficult benign inputs with weak perceptual evidence may still be conservatively blocked, increasing false refusals. The key evaluation question is therefore not only whether COMIC lowers attack success rate, but whether it improves the overall safety–utility tradeoff relative to defenses that ignore reference structure.

We evaluate COMIC across representative open-source MLLMs and multimodal attack settings that stress localized visual references, embedded text, and structured visual content. Across four protected MLLMs, COMIC reduces FigStep attack success to near-zero levels and keeps JailBreakV-28K attack success consistently low, while preserving benign multimodal utility. These results show that enforcing safety at the level of grounded operation–target pairs substantially improves robustness over strong baselines without generator retraining. More broadly, the findings support a simple but important claim. Multimodal safety cannot be enforced reliably without modeling what the user asks the model to do, what visual evidence the request is grounded on, and how confident the system is in that grounding.

In conclusion, the key contributions of this paper are as follows.

• Reference-dependent multimodal safety failure. We identify a practical class of multimodal failures in which unsafe behavior emerges only after an apparently benign request is bound to a localized visual target. This shows that the true security-relevant unit is not the global prompt–image pair, but the grounded operation–target pair produced during dereference.

• Reference-aware pre-generation safety gating. We propose COMIC, a pre-generation safety gate for MLLMs that infers the requested operation and reference type, constructs candidate targets from OCR-derived regions and open-vocabulary proposals, grounds plausible referents through semantic and spatial matching, and evaluates safety over explicit operation–target pairs using conservative risk aggregation.

• Robustness with preserved benign utility. We evaluate COMIC across representative open-source MLLMs, FigStep-style localized jailbreaks, and JailBreakV-28K attacks. COMIC consistently lowers attack success while maintaining benign multimodal utility, demonstrating that reference-aware and confidence-aware gating improves the safety–utility tradeoff for multimodal deployment.

## II. RELATED WORK

We review prior work on multimodal safety alignment, jailbreak attacks, defenses, and harmful content understanding. Across these areas, a common gap remains: most methods judge safety at the global image–text level, while many real failures arise only when a request is grounded to a specific visual target. This gap motivates COMIC as a reference-aware pre-generation safety gate over explicit operation–target pairs.

## A. Multimodal Safety Alignment

Multimodal large language models (MLLMs) couple a vision encoder with a language model through projection or cross-attention, enabling joint image-text reasoning. Representative families include Flamingo [20], BLIP-2 with Q-Former [21], PaLI-style models [22], and instruction-tuned open MLLMs such as LLaVA [23], MiniGPT-4 [24], and Qwen-VL [25]. These systems typically inherit alignment through supervised instruction tuning and preference-based post-training such as RLHF and related methods [26], [27], [28], [29]. However, recent evaluations show that safety alignment does not transfer reliably to multimodal inputs [30], [17], [31]. A recurring finding is that visual tokens can weaken refusal behavior and steer decoding toward unsafe completions, even when the corresponding text-only request would be rejected. Prior analyses further suggest that small localized overlays or a limited subset of multimodal tokens can disproportionately drive unsafe behavior [16], [32]. This line of work establishes that multimodal alignment is fragile, but existing guardrails still operate largely on the global imagetext input and do not explicitly model what operation the user requests or which visual element that operation applies to. This leaves them brittle against localized, reference-dependent harms, which motivates COMIC’s decision to treat operation– target binding as an explicit safety boundary.

## B. Multimodal Reference-Aware Jailbreak Attacks

A growing body of work shows that multimodal jailbreaks often succeed by exploiting how MLLMs bind textual intent to visual content during inference. Early attacks commonly embed malicious instructions into a single modality, especially typographic text rendered inside images to bypass text filters, as exemplified by FigStep [33]. Follow-up attacks extend this idea to structured visuals such as diagrams and flowcharts, where the model reconstructs harmful intent from composition and layout rather than from an explicitly unsafe global input [34], [35]. Other work distributes malicious semantics across text and image so that each modality appears benign in isolation while their joint interpretation reassembles harmful intent at inference time [36]. Orthogonally, adversarial image perturbations can shift multimodal representations and induce unsafe generations without changing model parameters [37], [38], [39], [40], and related failures extend to video settings where harm emerges across frames and temporal composition [41], [42]. Collectively, these attacks reveal a common weakness. MLLMs implicitly resolve references and bind user requests to visual evidence without verifying whether the grounded target itself is safe to act on. COMIC is designed precisely for this failure mode. Instead of only detecting suspicious global inputs, it explicitly resolves the referent and evaluates safety over the resulting operation–target pair at the binding step.

## C. Defenses Against Multimodal Jailbreaks and Misalignment

Existing defenses span prompt-based shields, representation-level interventions, model-agnostic filters, and decoding-time alignment. Prompt-driven methods are attractive because they are easy to deploy; AdaShield is a representative dynamic prompting defense [13], while SelfDefenD illustrates practical self-protection strategies against jailbreaks [43]. Other methods reshape the input or embedding space to recover safer behavior, as in BaThe [14]. A broader line of work studies plug-and-play detection and purification strategies, including denoising, similarity-shift tests, and lightweight multimodal filters [44]. Decoding-time methods instead regulate generation using safety objectives, with IMMUNE as a representative example for MLLMs [15]. Although these defenses differ in where and how they intervene, most still make safety decisions before, or without, explicitly determining which visual element the model is about to act upon. Even when OCR or grounding is used, reference resolution is usually treated as an auxiliary perception step rather than as a first-class safety decision. As a result, these methods remain vulnerable when a request appears globally benign but becomes unsafe only after it is grounded to a specific local region. COMIC complements this literature by inserting an explicit operation–target interface between perception and generation, so that safety is enforced on the grounded action before any response generation or tool execution.

## D. Multimodal Harmful Content Understanding

Research on harmful multimodal artifacts, especially memes, propaganda, and OCR-heavy images, shows that harm often arises from the interaction of layout, embedded text, and context rather than from either modality alone. The study I know what you MEME! offers a detailed analysis of harmful meme understanding with MLLMs and highlights the importance of composition and implicit multimodal semantics [18]. This observation is consistent with broader benchmarks such as Hateful Memes [45], Memotion Analysis [46], and related work on multimodal hate and abuse detection [47], as well as recent in-the-wild safety evaluations showing that OCRheavy and compositionally structured images remain particularly failure-prone [19]. While this literature has substantially improved harmful-content recognition, most methods perform static classification of the image-text artifact itself and remain largely independent of user intent. In interactive multimodal systems, however, a benign-looking image may become harmful only under a particular requested action, such as extracting, translating, or summarizing a localized text span. COMIC builds on this literature but moves from static harmful-content understanding to instruction-conditioned, reference-aware enforcement, where safety depends on both what content is present and what operation the system is being asked to perform on it.

Across alignment, attacks, defenses, and harmful-content understanding, a recurring blind spot remains: safety decisions are often made without explicitly resolving what the user is acting on. Although grounding and referring-expression comprehension can localize regions [48], [49], they are not designed to enforce normative or security-critical constraints on actions. Our proposed COMIC treats reference resolution itself as a safety primitive. It operates after perception, including OCR and grounding, but before generation or tool execution, decomposing each query into an operation and a referenced target and judging safety over that pair. In this way, COMIC directly addresses typographic, diagrammatic, and cross-modal reconstruction failures while remaining complementary to prompting-based, detection-based, and internalintervention defenses.

## III. THREAT MODEL

MLLMs expose a distinctive multimodal safety risk because harmful behavior is not always visible in the prompt or the image alone. In many attacks, the prompt appears harmless and the image appears mostly benign, but the request becomes unsafe after the model resolves a visual reference and applies a benign-looking operation to a localized target containing unsafe content. The relevant attack surface is therefore not only the holistic prompt–image pair, but the act of referenceconditioned execution, where user intent is converted into grounded action. Fig. 2 illustrates this setting, where a globally benign-looking input becomes unsafe only after an operation is grounded to an adversarial region.

We consider a black-box inference-time adversary who can submit arbitrary multimodal inputs to a fixed downstream

![](images/8414ac0f531b74908815b37c5b3ac6ad3990fb6a302ce53592534e270130e0a6.jpg)  
Fig. 2. Threat model for reference-conditioned multimodal jailbreaks.

MLLM protected by an external safety mechanism. Each input consists of a text prompt and an associated image. The adversary may use ordinary-looking prompts while placing unsafe semantics in localized or structured image regions, such as text overlays, boxed phrases, numbered steps, highlighted spans, menu items, diagram nodes, or interface elements. The adversary does not modify model parameters, system prompts, safety policies, moderation components, or training data, and has no privileged access to internal activations.

An attack succeeds if the safety mechanism forwards the request and the downstream MLLM, after resolving the visual reference, generates policy-violating content grounded in the adversarial region. This threat model covers localized imagetext jailbreaks, structured artifacts such as diagrams and interfaces, and cross-modal attacks in which the prompt and image may appear benign in isolation but jointly reconstruct harmful intent after grounding. What unifies these settings is the same mechanism: harmfulness is activated only when a requested operation is applied to a grounded local target.

## IV. CURRENT DEFENSE FAILURES

The core motivation behind COMIC is a structural mismatch between the safety abstraction used by existing defenses and the mechanism by which many multimodal jailbreaks ac tually produce unsafe behavior. Current defenses largely judge the prompt–image pair holistically, whereas many localized and reference-conditioned attacks become unsafe only after a benign-looking request is grounded to a specific visual target. Our analysis reveals three recurring failure patterns that make this mismatch explicit and directly motivate the design of COMIC.

## A. Moderation Mismatch

Existing multimodal defenses primarily moderate the full prompt–image pair, yet many successful multimodal jailbreaks are not unsafe at that global level. Instead, harmful behavior often emerges only after the system resolves a visual reference and applies an apparently benign requested operation to a localized target, as recent work on visual prompt injection and interface-based attacks has increasingly shown [50], [51]. Fig. 3 illustrates this mismatch: the image is dominated by benign classroom content, while the user prompt appears to request a simple OCR-based phrase completion. However, once the small localized text span is grounded and inserted into the incomplete action phrase, the composed request becomes unsafe. This failure is especially visible in localized referenceconditioned attacks, where harmful semantics occupy only a small image region and are diluted by surrounding benign context, while the prompt itself remains operationally incomplete until grounding occurs. Under these conditions, whole-input moderation sees an innocuous request and a mostly benign image, but the downstream model acts on a bound local interpretation. This pattern appears repeatedly in typographic and region-specific jailbreaks. The failure therefore lies not only in scoring weakness, but in moderating the wrong security object. COMIC addresses this problem by shifting the decision unit from the global prompt–image pair to the grounded operation– target pair and enforcing safety before generation.

![](images/cb09eea0948461f370b9c2ddbf23aac0dc209587a3d46bab95862a81fbc96ab1.jpg)  
Fig. 3. Moderation mismatch in a localized reference-conditioned jailbreak.

## B. Reference-Dependent Difficulty

The limitation of current defenses is not specific to one benchmark or visual format. Across multimodal jailbreak settings, safety becomes harder as harmfulness depends more strongly on reference resolution. Attacks are relatively easier to block when unsafe intent is globally explicit in the prompt or visually dominant in the image. They become harder when harmful content is confined to a small region, when the prompt refers to it only indirectly, or when screenshots, forms, diagrams, and interfaces require the model to infer which local element the user intends to act upon. Fig. 4 illustrates this difficulty, where the unsafe meaning is not globally explicit but emerges only after the model resolves a small localized text region and composes it with the surrounding prompt. This progression aligns with prior work showing that grounding becomes less reliable under complex referring expressions and remains vulnerable to targeted perturbations in MLLMs [52], [53]. Existing defenses typically leave this grounding step implicit inside the generator, which means the most security-critical stage of localized jailbreak resolution remains uninspected and uncontrolled. COMIC addresses this failure by making reference resolution explicit, inferring the requested operation, constructing candidate targets from OCR and visual proposals, and grounding plausible referents before any response is allowed to proceed.

![](images/758692f0d447cdafd00c50df23c6a184464c8576c8a60e0e49297456ce26a6a5.jpg)  
Fig. 4. Reference-dependent failure from localized visual grounding.

## C. Ambiguity as a Safety Multiplier

Reference-dependent multimodal attacks become more dangerous in dense visual inputs, where multiple regions may plausibly satisfy the same request. Recent work shows that ambiguity is not incidental noise, but a recurring feature of visually grounded understanding in complex scenes [54], [55]. In screenshots, annotated documents, diagrams, and interfaceheavy images, safety depends not only on whether a risky target exists, but also on whether the defense retains enough plausible candidates and handles uncertainty conservatively. Fig. 5 illustrates this risk: a screenshot contains several benign visual cues, but a small highlighted text region becomes safetycritical once it is selected and composed with the surrounding prompt. Our analysis suggests that single-shot holistic moderation is especially brittle in this regime because it often collapses too early to one dominant interpretation and may discard a plausible unsafe referent that remains fully consistent with the user’s request. Conversely, when the evidence is noisy or ambiguous, a deployment-grade defense should not treat that uncertainty as evidence of benignness. In localized grounding-dependent jailbreaks, ambiguity and dense layout therefore become direct safety multipliers rather than mere perception noise. COMIC addresses this failure by retaining multiple plausible targets, aggregating risk conservatively across them, and allowing forwarding only when unsafe risk is low and grounding evidence is sufficiently reliable.

## Observation 3: Ambiguity as a Safety Multiplier

When multiple regions plausibly satisfy the same request, ambiguity and dense visual structure become direct sources of safety risk, making conservative evidenceaware routing necessary.

![](images/749de443cfde9629f34448e703a2dcae9854f0583d644a0be0c117a516906d6d.jpg)  
Fig. 5. Ambiguity-driven failure under localized visual grounding.

The preceding failures motivate a pre-generation safety gate placed before the downstream MLLM. Given a multimodal request, the gate outputs a binary decision $\hat { y } \in$ {safe, unsafe}, determining whether the request is forwarded to the MLLM or blocked before generation. The defender aims to reduce attack success while preserving benign utility, especially for harmless reference-sensitive requests, in a deployment setting where safety is enforced externally without modifying the protected generator. We focus on inference-time multimodal jailbreaks where harmfulness is activated through reference resolution. Training-time poisoning, model compromise, policy tampering, and attacks requiring privileged internal access are outside our scope. Within this setting, COMIC treats the grounded operation–target pair as the security-relevant object and enforces policy before generation. Our evaluation therefore measures both jailbreak robustness and benign utility on reference-sensitive inputs.

## V. COMIC

Our proposed COMIC is a reference-aware pre-generation safety gate for multimodal large language models. It is implemented as an inference-time rule/pipeline-based defense rather than as a newly trained MLLM or end-to-end classifier. Its central goal is to protect the point at which an apparently benign user request becomes a grounded action over localized image content. The failure analysis in Section IV shows that many multimodal jailbreaks are not unsafe at the level of the global prompt–image pair. They become unsafe only after the system resolves a visual reference and applies the requested operation to a specific target. Throughout this section, we use as a running example a screenshot, document, or diagram that appears globally benign but contains a small localized region with dangerous procedural content, such as a highlighted span, a numbered step, or a text box. A prompt like “Summarize the highlighted part” or “What should I do next?” may appear harmless to a global moderator or a prompt-only shield, even though the downstream MLLM can still ground the request to the dangerous local region and produce unsafe content. Fig. 6 summarizes this workflow.

## A. Task Formulation and Input Representation

At interaction turn $t ,$ the system receives a multimodal input

$$
x _ { t } = ( P _ { t } , I _ { t } ) ,\tag{1}
$$

where $P _ { t }$ is the user prompt and $I _ { t }$ is the associated image. COMIC outputs a binary routing decision

$$
\hat { y } _ { t } = g _ { \mathrm { C O M I C } } ( P _ { t } , I _ { t } ) , \qquad \hat { y } _ { t } \in \mathcal { V } = \{ \mathrm { s a f e } , \mathrm { u n s a f e } \} ,\tag{2}
$$

which determines whether the request is forwarded or blocked.

The key modeling shift is that safety is not treated as a property of the whole input alone. Instead, COMIC makes explicit three variables that global moderation usually leaves implicit: the requested operation, the reference type, and the grounded target. Let $\mathcal { O }$ denote the operation space and R the reference-type space. In our setting, O includes common user operations such as summarizing, translating, extracting, rewriting, explaining, following, or answering based on image content. The reference-type space $\mathcal { R }$ captures how the prompt points to visual evidence, including references to OCR spans, highlighted or boxed regions, objects, interface elements, diagram nodes, numbered steps, or the absence of a localized reference. Then COMIC seeks to infer

$$
\big ( o _ { t } , r _ { t } , i _ { t } ^ { \star } , y _ { t } \big ) ,\tag{3}
$$

where $o _ { t }$ is the requested operation, $r _ { t }$ is the reference type, $i _ { t } ^ { \star }$ is the grounded target index, and $y _ { t }$ is the safety label. A compact formulation is

$$
p ( y _ { t } \mid P _ { t } , I _ { t } ) = \sum _ { o , r , i } p ( y _ { t } \mid o , i , P _ { t } , I _ { t } ) p ( i \mid o , r , P _ { t } , I _ { t } ) p ( o , r \mid P _ { t } ) .\tag{4}
$$

This factorization expresses the central premise of COMIC. A safe decision depends not only on the prompt and image jointly, but also on what action is requested and which local target that action is applied to.

## B. Intent Inference and Reference Grounding

Given the prompt, COMIC first computes a textual representation

$$
h _ { t } = \phi _ { t } ( P _ { t } ) ,\tag{5}
$$

and predicts both the requested operation and the reference type

$$
\hat { o } _ { t } = \arg \operatorname* { m a x } _ { o \in \mathcal { O } } p ( o \mid P _ { t } ) , \qquad \hat { r } _ { t } = \arg \operatorname* { m a x } _ { r \in \mathcal { R } } p ( r \mid P _ { t } ) .\tag{6}
$$

In our implementation, this step is performed using a fixed intent-parsing routine rather than by training a new intent model. The parser maps prompt patterns and operation verbs to a small set of supported operations and reference types. If $\hat { r } _ { t } = \mathrm { n o n e }$ , the request does not require localized visual dereferencing. In that case, COMIC uses a fallback moderation route that checks the prompt and global image-derived evidence, such as OCR text or a global caption when available, and then forwards only if no explicit policy-violating content is detected. Otherwise, COMIC activates grounded safety reasoning.

To support localized reasoning, COMIC converts the image into a structured set of candidate targets. OCR extracts textbearing regions, while open-vocabulary proposals cover nontext references such as interface elements, diagram nodes, icons, or highlighted objects. After filtering and merging, the unified candidate set is

$$
T _ { t } = \{ ( \bar { s } _ { i } , b _ { i } , \alpha _ { i } , c _ { i } ) \} _ { i = 1 } ^ { N _ { t } } ,\tag{7}
$$

![](images/4fa7312c45d18f4d0a286698dc2fe6ee5ebddc4b067ded424999a049ff964fb2.jpg)  
Fig. 6. Overview of COMIC for intent-aware grounding, operation–target safety scoring, and pre-generation defense mechanism.

where $\bar { s } _ { i }$ is the text or descriptor associated with region i, $b _ { i }$ is its box, $\alpha _ { i }$ is a confidence score, and $c _ { i }$ is its candidate type. For each region, COMIC extracts local visual and textual features and combines them with lightweight layout cues.

Grounding then estimates how plausibly each region satisfies the prompt reference. For candidate i, COMIC computes a grounding score

$$
S ( i \mid P _ { t } , I _ { t } ) = \psi ( h _ { t } , u _ { i } , v _ { i } , f _ { i } , c _ { i } ) ,\tag{8}
$$

where $u _ { i }$ is the candidate text feature, $v _ { i }$ is the local visual feature, and $f _ { i }$ denotes layout and confidence information. The scoring function combines fixed semantic, lexical, spatial, and layout-based matching rules. For example, references such as “highlighted part”, “step $3 ^ { \prime \prime } { _ { ! } }$ , or “the boxed $t e x t ^ { \prime \prime }$ are matched against candidate type, OCR content, visual region attributes, and relative layout. Rather than collapsing immediately to a single target, COMIC retains the top-K grounded candidates

$$
I _ { K } = \mathrm { T o p K } \big ( \{ S ( i \mid P _ { t } , I _ { t } ) \} _ { i \in T _ { t } } \big ) .\tag{9}
$$

This step is important because ambiguity is often safetyrelevant. When several regions plausibly satisfy the same request, the defense should reason over all plausible referents rather than commit too early to a single interpretation.

## C. Localized Harm Assessment and Safety Decision

For each retained candidate $\textit { i } \in \textit { I } _ { K }$ , COMIC forms an operation-conditioned representation

$$
z _ { t , i } = [ h _ { t } ; E _ { \mathcal { O } } ( \hat { o } _ { t } ) ; u _ { i } ; v _ { i } ; f _ { i } ; \mathrm { O N E H O T } ( c _ { i } ) ] ,\tag{10}
$$

and applies a candidate-level safety scorer

$$
p _ { i } ( \cdot \mid P _ { t } , I _ { t } ) = \operatorname { s o f t m a x } ( W _ { s } z _ { t , i } + b _ { s } ) .\tag{11}
$$

This scorer is used as a fixed rule-calibrated risk mapping, not as a newly trained classifier. It converts operation-conditioned evidence into a safety distribution for each plausible grounded target. The candidate-level unsafe risks are then aggregated conservatively as

$$
p ( { \mathrm { u n s a f e } } \mid P _ { t } , I _ { t } ) = \operatorname* { m a x } _ { i \in I _ { K } } p _ { i } ( { \mathrm { u n s a f e } } \mid P _ { t } , I _ { t } ) .\tag{12}
$$

The max-risk rule reflects the asymmetric nature of safety routing. If one plausible grounded interpretation is unsafe, forwarding should be denied even if another plausible interpretation appears benign.

Routing, however, should depend not only on predicted unsafe risk but also on the reliability of the perceptual evidence. Proposal miss, weak OCR, and ambiguous grounding are not just perception errors. In deployment, they are direct sources of safety risk. COMIC therefore computes a proposal-grounding quality statistic

$$
q _ { t } = Q ( \{ \alpha _ { i } \} , \Delta _ { t } , N _ { t } , . . . ) ,\tag{13}
$$

where $\Delta _ { t }$ denotes the grounding margin between the top candidates and the remaining terms summarize proposal and OCR quality. Concretely, $Q ( \cdot )$ combines signals such as OCR confidence, proposal confidence, candidate coverage, the number of retained candidates, and the separation between the highest-scoring and competing grounded targets. A high value of $q _ { t }$ indicates that the system has enough reliable evidence to justify a forwarding decision, while a low value indicates weak perception, poor coverage, or unresolved ambiguity. The final routing rule is

$$
\hat { y } _ { t } = \left\{ \begin{array} { l l } { \mathrm { s a f e } , } & { p ( \mathrm { u n s a f e } \mid P _ { t } , I _ { t } ) < \tau _ { s } \land q _ { t } \ge \tau _ { q } , } \\ { \mathrm { u n s a f e } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{14}
$$

This rule is central to COMIC. A request is forwarded only when both conditions hold simultaneously. The grounded unsafe risk must be low, and the supporting evidence must be reliable. The thresholds $\tau _ { s }$ and $\tau _ { q }$ are fixed on a held-out calibration split and are not tuned on the test benchmarks. They are selected to prioritize low attack success while limiting utility degradation on held-out benign multimodal inputs. In this way, COMIC avoids treating uncertainty as evidence of benignness.

The structural advantage of COMIC can be understood through a monotonicity intuition rather than a universal guarantee. Let p<sub>GLOBAL</sub>(unsafe $\mid P _ { t } , I _ { t } )$ denote the unsafe score of a whole-input moderator, and let

$$
p _ { \mathrm { C O M I C } } ( \mathrm { u n s a f e } \mid P _ { t } , I _ { t } ) = \operatorname* { m a x } _ { i \in I _ { K } } p _ { i } ( \mathrm { u n s a f e } \mid P _ { t } , I _ { t } ) .\tag{15}
$$

If the true adversarial target is both proposed and retained, and if the unsafe score assigned to that target is at least as large as the global unsafe score, then

$$
p _ { \mathrm { C O M I C } } ( \mathrm { u n s a f e } \mid P _ { t } , I _ { t } ) \geq p _ { \mathrm { G L O B A L } } ( \mathrm { u n s a f e } \mid P _ { t } , I _ { t } ) .\tag{16}
$$

This observation does not claim that COMIC dominates global moderation in all cases. Instead, it captures why grounded max-risk reasoning is useful for localized jailbreaks. When the dangerous target is visible to the system and retained among plausible referents, candidate-level safety can expose risk that a single holistic score may dilute or miss.

## D. Rule-Based Inference Procedure

COMIC consists of three coupled rule-based components: intent inference, grounding, and operation-conditioned safety scoring. These components are not trained end-to-end. Instead, they are assembled as an inference-time pipeline that uses fixed operation rules, OCR and proposal metadata, spatial and semantic matching, policy-risk terms, and calibrated routing thresholds. This design makes COMIC deployable as an external safety layer without modifying or retraining the protected MLLM.

The decomposition mirrors the structure of the task itself. The system must first interpret the request, then localize what it refers to, and finally determine whether executing that grounded request is permissible. This explicit decomposition is important because the safety-relevant content may not be globally visible until the requested operation is bound to a local target.

At inference time, the computational overhead after OCR and proposal extraction is modest. Let $N _ { t } ~ = ~ \left| T _ { t } \right|$ be the number of retained candidates and d the feature dimension. Grounding scales as $\mathcal { O } ( N _ { t } d )$ , while candidate-level safety scoring over the retained top-K regions scales as $\mathcal { O } ( K d )$ with $K \ll N _ { t }$ . In practice, total latency is dominated by OCR and proposal generation, while the intent, grounding, and safety components remain lightweight. Algorithm 1 summarizes the full inference pipeline.

Algorithm 1: COMIC Inference Pipeline   
Input: Prompt $P _ { t } ,$ image $\overline { { I _ { t } } }$   
Output: Routing decision yˆ<sub>t</sub> ∈ {safe, unsa $. \mathbf { f } \mathbf { e } \}$   
Compute prompt embedding $h _ { t } \stackrel { . } {  } \phi _ { t } ( P _ { t } ) ;$   
Infer operation $\hat { o } _ { t }$ and reference type rˆ<sub>t</sub>;   
if rˆ<sub>t</sub> = none then   
apply fallback moderation and return decision;   
Extract OCR regions and open-vocabulary proposals from I<sub>t</sub>;   
Merge and filter them to obtain candidate set $\bar { T } _ { t } ;$   
foreach candidate $i \in T _ { t }$ do   
compute candidate features and grounding score   
$S ( \dot { i } \mid P _ { t } , I _ { t } ) ;$   
Retain top-K candidates $I _ { K }$ and compute quality statistic q<sub>t</sub>;   
foreach candidate $i \in I _ { K }$ do   
form operation-conditioned representation $z _ { t , i } ;$   
predict candidate-level unsafe risk p<sub>i</sub>(unsafe | P<sub>t</sub>, I<sub>t</sub>);   
Aggregate risk   
p(unsafe | P , I ) ← max p (unsafe | P , I )   
i∈I<sub>K</sub>   
if p(unsafe | $P _ { t } , I _ { t } ) < \tau _ { s }$ and $q _ { t } \geq \tau _ { q }$ then   
yˆ<sub>t</sub> ← safe;   
else   
yˆ<sub>t</sub> ← unsafe;   
return $\hat { y } _ { t } ;$

## VI. EXPERIMENTAL SETTINGS

Our evaluation is designed to answer three questions. First, does reference-aware pre-generation gating reduce multimodal jailbreak success relative to strong baselines. Second, are these gains achieved without excessive overblocking on benign inputs. Third, do the gains remain consistent across multiple open-source MLLMs with practical inference overhead.

TABLE I  
CURATED DATASETS USED IN COMIC.
<table><tr><td>Dataset</td><td>Exp.</td><td>Purpose</td></tr><tr><td>FIGSTEP DATA</td><td>Block</td><td>Localized jailbreaks</td></tr><tr><td>JAILBREAKV-28K</td><td>Block</td><td>Broad jailbreak test</td></tr><tr><td>MM-VET</td><td>Allow</td><td>Benign utility</td></tr></table>

## A. Benchmarks and Data Preparation

We evaluate COMIC on two multimodal jailbreak benchmarks and one benign utility benchmark to measure security, capability preservation, and deployment tradeoffs. FigStep Data [33] serves as our primary benchmark for localized multimodal jailbreaks, where harmful content is embedded in image space and becomes actionable only after visual localization and reference resolution. We use a curated 500-sample split for the main security evaluation. JailBreakV-28K [56] provides a broader and more diverse set of multimodal jailbreak behaviors beyond a single attack family. We evaluate a curated 150- sample subset to test generalization under more varied attack styles. For benign utility, we use MM-Vet [57], a multimodal capability benchmark that evaluates model responses across recognition, knowledge, optical character recognition, spatial awareness, language generation, and math. Following standard evaluation practice, we report the average performance across all categories to assess whether COMIC preserves normal visual understanding and reasoning while reducing jailbreak success.

TABLE II  
ASR COMPARISON BETWEEN BASELINES AND OUR COMIC UNDER MULTIMODAL JAILBREAK ATTACKS.
<table><tr><td></td><td colspan="5">FigStep Data (%)</td><td colspan="5">JailBreakV-28K (%)</td></tr><tr><td>Model</td><td>FigStep</td><td>AdaShield</td><td>CoCA</td><td>Immune</td><td>COMIC (Ours)</td><td>FigStep</td><td>AdaShield</td><td>CoCA</td><td>Immune</td><td>COMIC (Ours)</td></tr><tr><td>LLaVA-1.6</td><td>5.90</td><td>3.62</td><td>13.34</td><td>1.21</td><td>0.08</td><td>51.64</td><td>19.23</td><td>51.37</td><td>2.45</td><td>2.70</td></tr><tr><td>LLaVA-1.5</td><td>48.48</td><td>7.24</td><td>28.63</td><td>4.23</td><td>0.08</td><td>52.46</td><td>12.86</td><td>39.87</td><td>2.10</td><td>2.70</td></tr><tr><td>MiniGPT-4(7B)</td><td>8.32</td><td>5.76</td><td>5.74</td><td>4.43</td><td>0.09</td><td>27.74</td><td>32.21</td><td>29.74</td><td>18.34</td><td>2.90</td></tr><tr><td>Qwen-VL</td><td>4.56</td><td>5.04</td><td>43.92</td><td>3.23</td><td>0.11</td><td>14.42</td><td>15.88</td><td>15.69</td><td>8.58</td><td>3.20</td></tr></table>

All adversarial instances are normalized into a unified format consisting of an image, a user prompt, and an expected behavior label in {Allow, Block}. We remove corrupted samples, discard cases where visual text is unreadable when OCR is central to the attack semantics, and eliminate exact duplicates when present. For benign evaluation, MM-Vet examples are kept in their standard image-question format and scored using the benchmark protocol. This yields a compact but controlled testbed for evaluating whether COMIC can provide strong adversarial blocking with minimal loss of benign multimodal capability.

## B. Baselines and Base Models

We evaluate COMIC on four representative opensource MLLMs, namely LLaVA-1.5 [58], LLaVA-1.6 [59], MiniGPT-4 (7B) [24], and Qwen-VL [25]. These models differ in multimodal architecture, OCR sensitivity, and alignment behavior, making them a useful testbed for assessing whether the gains of COMIC are robust across backbone families rather than tied to a single implementation.

We compare against the undefended Original model and four competitive baseline defenses, namely AdaShield [13], CoCA [60], Immune [15] and FigStep [33]. These baselines span multiple defense families, including attack-aware filtering, prompting-based shielding, calibration-oriented defenses, and inference-time alignment methods. This comparison is particularly important because COMIC intervenes earlier in the pipeline, at the reference-resolution boundary, whereas the baselines primarily reason globally or intervene during generation.

## C. Metrics and Evaluation Configuration

Our primary security metric is attack success rate (ASR), where lower is better. An attack is counted as successful if the protected model still produces unsafe content on an adversarial input. To normalize improvements across models with different baseline vulnerabilities, we also report relative ASR reduction with respect to the undefended model. For benign utility, we report the MM-Vet average score, where higher values indicate better preservation of multimodal capability. This score reflects average performance across recognition, knowledge, optical character recognition, spatial awareness, language generation, and math. For deployment practicality, we report runtime per sample. All defenses are evaluated under a fixed inference protocol for each protected model, including a common decoding configuration and a uniform response-generation procedure across methods. COMIC is implemented as an external inference-time safety gate, so it protects the downstream MLLM without retraining, finetuning, or modifying the generator. Unless otherwise stated, COMIC retains the top-K grounded candidates with K = 5, and its routing thresholds are fixed using held-out validation data. This controlled setup ensures that performance differences are attributable to the defense mechanism rather than to inconsistent decoding settings or model updates.

## VII. EVALUATION AND RESULTS

We evaluate COMIC along three research questions aligned with its security, utility, and deployment goals.

• RQ1: Robustness. Does reference-aware pre-generation gating reduce attack success on localized and broader multimodal jailbreak benchmarks compared with strong baselines?

• RQ2: Utility. Does COMIC preserve benign multimodal capability, or does safety gating degrade normal visual understanding and reasoning?

• RQ3: Efficiency. Can COMIC provide these robustness gains with practical inference-time overhead across multiple open-source MLLMs?

We answer these questions in order and then report normalized ASR reductions relative to the undefended model.

## A. Security Performance on Multimodal Jailbreak Benchmarks

We first evaluate whether COMIC reduces multimodal jailbreak success relative to strong baselines. Table II summarizes the primary security results on FigStep and JailBreakV-28K, while Fig. 7 provides a model-by-model visual comparison across defenses. Across both benchmarks, COMIC consistently reduces attack success relative to the undefended model and remains competitive with or stronger than prior defenses across the evaluated backbones. On FigStep, COMIC drives attack success rate (ASR) to near zero on all four protected MLLMs. Specifically, ASR drops from 14.68% to 0.08% on LLaVA-1.6, from 58.58% to 0.08% on LLaVA-1.5, from 11.95% to 0.09% on MiniGPT-4(7B), and from 24.01% to 0.11% on Qwen-VL. These improvements are substantial not only in absolute terms, but also in consistency. Unlike several baselines whose effectiveness varies noticeably across models, COMIC maintains uniformly strong performance across all four backbones. On JailBreakV-28K, the pattern remains favorable, although the comparison is more nuanced. COMIC achieves the lowest ASR on MiniGPT-4(7B) and Qwen-VL, and remains close to the strongest baseline on LLaVA-1.5 and LLaVA-1.6, where Immune is slightly lower in absolute ASR. Even in those cases, COMIC substantially outperforms the undefended model and most baseline defenses. Overall, the results indicate that COMIC is particularly effective on attacks that depend on localized grounding, while remaining highly competitive on broader multimodal jailbreak distributions.

![](images/13f2682ec7b027e5ea9ad96d9dd070fd7bd43abe58a3d1c5d4bfed2e8750be7b.jpg)  
Fig. 7. ASR Comparison Between Baselines and Our COMIC under multimodal jailbreak attacks.

Fig. 8 further illustrates this behavior through a representative reference-conditioned jailbreak. The input appears to be an ordinary multimodal request, but becomes unsafe after the model grounds the visual reference to localized content. While a standard pretrained MLLM may act on that region, COMIC detects the risky operation–target pair before generation and routes the request to a safe response. Together, the quantitative and qualitative results support the central claim that referenceaware pre-generation gating is most effective when attacks activate harm through localized visual grounding.

## B. Benign Utility and Capability Preservation

We next assess whether the security gains of COMIC come at the cost of the model’s normal multimodal capability. Here, benign utility refers to the ability of the protected MLLM to answer harmless visual questions that require recognition, knowledge, optical character recognition, spatial reasoning, language generation, and math reasoning. Table III reports the exact MM-Vet utility scores, while Fig. 9 visualizes the model-wise comparison across defenses. COMIC achieves the strongest score on LLaVA-1.5, MiniGPT-4(7B), and Qwen-VL, and remains close to the best result on LLaVA-1.6, where Immune is slightly higher. On average, COMIC provides the strongest benign utility among the evaluated defenses, suggesting that its safety gains do not come from broadly weakening normal model behavior.

![](images/5a8edb39db149bc31a5ad971bd9560bff3aa7335a6654b79dc70907ede794439.jpg)  
Fig. 8. Qualitative example of reference-conditioned jailbreak behavior and COMIC intervention.

This comparison is important because several defenses introduce a clear utility cost. FigStep and AdaShield reduce MM-Vet performance across multiple backbones, while COMIC generally maintains or improves utility relative to the undefended baseline. This pattern aligns with the design of COMIC. The gate intervenes only when a grounded operation–target pair appears unsafe or when grounding evidence is too unreliable to support forwarding. As a result, harmless multimodal reasoning and visual understanding are largely preserved while risky or ambiguous grounded requests are handled conservatively.

## C. Runtime and Deployment Practicality

We then examine whether COMIC remains practical to deploy across multiple open-source MLLMs. Table IV reports the exact per-sample runtime values, while Fig. 10 visualizes the runtime trend across protected models and defense methods. As expected, the undefended model is the fastest in all cases. However, the additional overhead introduced by COMIC is modest and remains much lower than that of CoCA and Immune across all evaluated backbones. Specifically, COMIC increases runtime from 3.48s to 3.72s on LLaVA-1.6, from 3.52s to 3.76s on LLaVA-1.5, from 10.38s to 11.03s on MiniGPT-4(7B), and from 1.91s to 2.16s on Qwen-VL. Fig. 10 shows the same pattern visually. The COMIC curve stays close to the undefended and AdaShield curves, while CoCA and Immune introduce substantially larger latency across models. This efficiency profile is consistent with the design of COMIC. Its main cost comes from OCR and proposal generation, while the intent, grounding, and safety components remain lightweight. Since COMIC operates entirely at inference time and does not require multi-pass generation or a second multimodal model, it provides a practical security– latency tradeoff for deployment.

## D. Relative Improvement Over the Undefended Model

Table V reports relative ASR reduction with respect to the undefended model. This normalized view is useful because the baseline vulnerability of the protected MLLMs differs substantially across backbones. On FigStep, COMIC achieves the strongest relative reduction for every evaluated model, with a mean reduction of 99.53%. On JailBreakV-28K, it again achieves the strongest average reduction, with a mean of 93.12%. Fig. 11 further highlights that these gains remain consistently strong across protected MLLMs on both benchmarks. Overall, these normalized reductions confirm that the performance of COMIC is not driven by one favorable model– benchmark pairing; rather, its improvements remain stable across models with very different initial risk profiles.

Overall, the empirical pattern is consistent and practically meaningful. COMIC delivers its largest gains on attack settings where unsafe behavior depends on localized grounding, retains strong benign utility, and does so with only modest runtime overhead. These results support the central claim of the paper. Safety improves when the defense explicitly models the requested operation, the grounded target, and the reliability of that grounding before generation.

## VIII. FAILURE ANALYSIS AND LIMITATIONS

Although COMIC substantially improves safety on representative benchmarks, its remaining failures are structured rather than arbitrary. They arise when the defense cannot reliably surface the relevant target, cannot resolve the correct target under ambiguity, or encounters attacks whose harmfulness extends beyond the single-target reasoning unit used by the current design.

(d) Qwen-VL  
(c) MiniGPT-4(7B)  
TABLE III  
BENIGN UTILITY COMPARISON ACROSS MODELS AND DEFENSE METHODS.
<table><tr><td>Model</td><td>Original / Baseline</td><td>FigStep</td><td>AdaShield</td><td>CoCA</td><td>Immune</td><td>COMIC (Ours)</td></tr><tr><td> $\mathrm { L L a V A – 1 } . 6$ </td><td>37.3</td><td>28.1</td><td>32.2</td><td>36.1</td><td>37.7</td><td>37.5</td></tr><tr><td> $\mathrm { L L a V A } { - } 1 . 5$ </td><td>30.3</td><td>26.6</td><td>21.6</td><td>29.3</td><td>31.3</td><td>32.1</td></tr><tr><td>MiniGPT-4(7B)</td><td>20.0</td><td>19.8</td><td>14.3</td><td>20.2</td><td>24.7</td><td>24.9</td></tr><tr><td>Qwen-VL</td><td>40.5</td><td>37.3</td><td>28.4</td><td>40.5</td><td>40.5</td><td>40.6</td></tr></table>

![](images/f0010efb88049909934c0b58f8cb0bc381666cc0b028d7bbee9e789ecdb7cdff.jpg)

![](images/8ca3443c6513d33e0ad1795dfd6854ec639beb54153bac64d1a7349830b559c2.jpg)

![](images/686137c8a66b68cca5d49990bb3f6930cc57ee7fd9a235882e7c3347909e1764.jpg)

![](images/3e82526ac9f77ac9e61277e21ab89cd8e14c7447663efc55157fe37f03be4623.jpg)  
Fig. 9. Benign utility on MM-Vet across protected MLLMs and defense methods.  
TABLE IV

![](images/fae411a85215df11e2a85f924877d5bc363a47d2506625c166557c2684010169.jpg)  
Fig. 10. Runtime comparison across protected MLLMs and defense methods.

INFERENCE-TIME OVERHEAD ACROSS PROTECTED MLLMS.
<table><tr><td>Model</td><td>Original</td><td>AdaShield</td><td>CoCA</td><td>Immume</td><td>COMIC (Ours)</td></tr><tr><td>LLaVA-1.5</td><td>3.52</td><td>3.62</td><td>7.02</td><td>4.98</td><td>3.76</td></tr><tr><td>LLaVA-1.6</td><td>3.48</td><td>3.58</td><td>7.01</td><td>4.93</td><td>3.72</td></tr><tr><td>MiniGPT-4(7B)</td><td>10.38</td><td>10.48</td><td>19.86</td><td>14.76</td><td>11.03</td></tr><tr><td>Qwen-VL</td><td>1.91</td><td>2.01</td><td>7.43</td><td>4.57</td><td>2.16</td></tr></table>

• Perception and proposal recall remain the first bottleneck. COMIC can reason only over targets that appear in its candidate set. If OCR or proposal generation misses the true referenced region, later grounding and safety stages cannot recover it. This is most likely for small, stylized, low-contrast, or cluttered content, and it remains the main perception-level boundary of the method.

• Ambiguity in dense inputs remains a routing challenge. Even when the correct target is present, screenshots, diagrams, and document-heavy images may contain several plausible referents. In this setting, COMIC must decide under unresolved ambiguity. Top-K retention and conservative aggregation reduce brittleness, but ambiguity can still cause either unsafe forwarding or false refusal.

![](images/4bf92992fa6627661d7d817f771f2090c64f3e161ca0cb44e2748e0170e386ec.jpg)  
Fig. 11. Relative ASR reduction of COMIC across FigStep and JailBreakV-28K.

TABLE V  
RELATIVE ASR REDUCTION ACROSS MULTIMODAL JAILBREAK BENCHMARKS.
<table><tr><td></td><td colspan="5">FigStep Data (%)</td><td colspan="5">JailBreakV-28K (%)</td></tr><tr><td>Model</td><td>FigStep</td><td>AdaShield</td><td>CoCA</td><td>Immune</td><td>COMIC (Ours)</td><td>FigStep</td><td>AdaShield</td><td>CoCA</td><td>Immune</td><td>COMIC (Ours)</td></tr><tr><td>LLaVA-1.6</td><td>59.81</td><td>75.34</td><td>9.13</td><td>91.76</td><td>99.46</td><td>14.32</td><td>68.09</td><td>14.77</td><td>95.93</td><td>95.52</td></tr><tr><td>LLaVA-1.5</td><td>17.24</td><td>87.64</td><td>51.13</td><td>92.78</td><td>99.86</td><td>6.97</td><td>77.19</td><td>29.30</td><td>96.28</td><td>95.21</td></tr><tr><td>MiniGPT-4(7B)</td><td>30.38 81.01</td><td>51.80</td><td>51.97</td><td>62.93</td><td>99.25</td><td>58.69</td><td>52.03</td><td>55.71</td><td>72.69</td><td>95.68</td></tr><tr><td>Qwen-VL</td><td></td><td>79.01</td><td>-82.92</td><td>86.55</td><td>99.54</td><td>37.28</td><td>30.93</td><td>31.75</td><td>62.68</td><td>86.08</td></tr><tr><td>Mean</td><td>47.11</td><td>73.45</td><td>7.32</td><td>83.50</td><td>99.53</td><td>29.31</td><td>57.06</td><td>32.88</td><td>81.89</td><td>93.12</td></tr></table>

• Distributed harm exceeds the current reasoning granularity. COMIC is strongest when harmfulness is concentrated in one or a few visually recoverable targets. A harder attack class distributes unsafe meaning across multiple individually benign regions, so that no single target appears clearly unsafe in isolation. Addressing this regime will likely require extending COMIC from operation–target reasoning to relational multi-region reasoning.

Overall, COMIC is most effective when the harmful target is visually recoverable, proposed with reasonable recall, and grounded sharply enough to support conservative routing. Its limitations arise when one or more of these conditions fail, or when harmfulness is distributed beyond the current target-level abstraction.

## IX. CONCLUSION

We identified a practically important class of multimodal safety failures in which unsafe behavior does not arise from the prompt or image alone, but from binding a requested operation to a localized visual target. This failure mode is poorly captured by global moderation, especially when harmfulness is localized, reference-dependent, and activated only after grounding. To address this problem, we introduced COMIC, a reference-aware pre-generation safety gate that performs operation inference, candidate-target construction, grounding, and conservative operation-conditioned safety classification before generation. Across representative multimodal jailbreak benchmarks, COMIC substantially reduces attack success while preserving strong benign utility and modest runtime overhead. More broadly, our results suggest a general principle for secure multimodal deployment: safety mechanisms should intervene at the point where user intent becomes grounded action. Extending this principle to adaptive attackers, multiregion reasoning, multilingual inputs, and broader multimodal agent settings is a promising direction for future work.

## X. ETHICAL CONSIDERATIONS

This work studies multimodal jailbreak defense and therefore necessarily engages with harmful prompts, unsafe generations, and dual-use failure analysis. The purpose of COMIC is strictly defensive. It is designed to reduce unsafe model behavior by enforcing safety before a downstream MLLM acts on a reference-dependent request. At the same time, analyzing where existing defenses fail may help adversaries reason more clearly about attack surfaces. We therefore describe the threat model and defense mechanism only to the level needed for scientific evaluation, while avoiding unnecessary operational detail that would make misuse easier.

The ethical motivation for COMIC is both practical and deployment-driven. Multimodal systems are increasingly used on screenshots, scanned documents, diagrams, and other visually grounded inputs, where unsafe behavior may arise not from the global input itself, but from how the model resolves and acts on localized content. In these settings, failures can be subtle, difficult to detect, and consequential in practice. By treating reference resolution as a security boundary, COMIC aims to reduce the likelihood that an apparently benign multimodal request is transformed into unsafe behavior through localized dereference.

A responsible evaluation of such a defense must consider both harmful exposure and overblocking. Harmful prompts and outputs should be handled only to the extent necessary for evaluation, auditing, and reproducibility, with human exposure minimized wherever possible. At the same time, a safety mechanism that blocks too aggressively can also cause harm by suppressing harmless requests or burdening visually noisy, unusual, or domain-specific inputs. For this reason, we evaluate COMIC not only by attack suppression, but also by benign multimodal capability and deployment overhead. More broadly, COMIC inherits limitations from its perception stack, including possible variation across languages, scripts, visual styles, and domains. Strong performance on English-centric benchmarks should therefore not be interpreted as universal robustness or fairness, and any deployment in sensitive settings should be accompanied by domain-specific validation and continued monitoring.

## ACKNOWLEDGMENT

The authors would like to thank...

## REFERENCES

[1] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[2] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in International conference on machine learning. PMLR, 2021, pp. 4904–4916.

[3] S. Huang, L. Dong, W. Wang, Y. Hao, S. Singhal, S. Ma, T. Lv, L. Cui, O. K. Mohammed, B. Patra et al., “Language is not all you need: Aligning perception with language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 72 096–72 109, 2023.

[4] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 49 250–49 267, 2023.

[5] Y. Xu, M. Li, L. Cui, S. Huang, F. Wei, and M. Zhou, “Layoutlm: Pre-training of text and layout for document image understanding,” in Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 2020, pp. 1192–1200.

[6] A. Kamath, M. Singh, Y. LeCun, G. Synnaeve, I. Misra, and N. Carion, “Mdetr-modulated detection for end-to-end multi-modal understanding,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 1780–1790.

[7] L. H. Li, P. Zhang, H. Zhang, J. Yang, C. Li, Y. Zhong, L. Wang, L. Yuan, L. Zhang, J.-N. Hwang et al., “Grounded language-image pre-training,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 965–10 975.

[8] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European conference on computer vision. Springer, 2024, pp. 38–55.

[9] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, “Not what you’ve signed up for: Compromising real-world llmintegrated applications with indirect prompt injection,” in Proceedings of the 16th ACM workshop on artificial intelligence and security, 2023, pp. 79–90.

[10] L. Bailey, E. Ong, S. Russell, and S. Emmons, “Image hijacks: Adversarial images can control generative models at runtime,” arXiv preprint arXiv:2309.00236, 2023.

[11] Z. Ying, A. Liu, T. Zhang, Z. Yu, S. Liang, X. Liu, and D. Tao, “Jailbreak vision language models via bi-modal adversarial prompt,” IEEE Transactions on Information Forensics and Security, 2025.

[12] J. Clusmann, S. J. Schulz, D. Ferber, I. C. Wiest, A. Fernandez, M. Eckstein, F. Lange, N. G. Reitsam, F. Kellers, M. Schmitt et al., “Incidental prompt injections on vision–language models in real-life histopathology,” Nejm Ai, vol. 2, no. 6, p. AIcs2500078, 2025.

[13] Y. Wang, X. Liu, Y. Li, M. Chen, and C. Xiao, “Adashield: Safeguarding multimodal large language models from structure-based attack via adaptive shield prompting,” in European Conference on Computer Vision. Springer, 2024, pp. 77–94.

[14] Y. Chen, H. Li, Y. Zhang, Z. Zheng, Y. Song, and B. Hooi, “Bathe: Defense against the jailbreak attack in multimodal large language models by treating harmful instruction as backdoor trigger,” arXiv preprint arXiv:2408.09093, 2024.

[15] S. S. Ghosal, S. Chakraborty, V. Singh, T. Guan, M. Wang, A. Beirami, F. Huang, A. Velasquez, D. Manocha, and A. S. Bedi, “Immune: Improving safety against jailbreaks in multi-modal llms via inferencetime alignment,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 25 038–25 049.

[16] Y. Gou, K. Chen, Z. Liu, L. Hong, H. Xu, Z. Li, D.-Y. Yeung, J. T. Kwok, and Y. Zhang, “Eyes closed, safety on: Protecting multimodal llms via image-to-text transformation,” in European Conference on Computer Vision. Springer, 2024, pp. 388–404.

[17] X. Liu, Y. Zhu, J. Gu, Y. Lan, C. Yang, and Y. Qiao, “Mm-safetybench: A benchmark for safety evaluation of multimodal large language models,” in European Conference on Computer Vision. Springer, 2024, pp. 386–403.

[18] Y. Zhuang, K. Guo, J. Wang, Y. Jing, X. Xu, W. Yi, M. Yang, B. Zhao, and H. Hu, “I know what you meme! understanding and detecting harmful memes with multimodal large language models,” Proceedings of the Network and Distributed System Security Symposium, 2025.

[19] X. Liu, Y. Zhu, Y. Lan, C. Yang, and Y. Qiao, “Safety of multimodal large language models on images and texts,” arXiv preprint arXiv:2402.00357, 2024.

[20] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” Advances in neural information processing systems, vol. 35, pp. 23 716–23 736, 2022.

[21] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[22] X. Chen, X. Wang, S. Changpinyo, A. J. Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner, B. Mustafa, L. Beyer et al., “Pali:

A jointly-scaled multilingual language-image model,” arXiv preprint arXiv:2209.06794, 2022.

[23] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 34 892– 34 916, 2023.

[24] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.

[25] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023.

[26] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike, and R. Lowe, “Training language models to follow instructions with human feedback,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 27 730–27 744.

[27] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” Advances in neural information processing systems, vol. 30, 2017.

[28] Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon et al., “Constitutional ai: Harmlessness from ai feedback,” arXiv preprint arXiv:2212.08073, 2022.

[29] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in neural information processing systems, vol. 36, pp. 53 728–53 741, 2023.

[30] M. Ye, X. Rong, W. Huang, B. Du, N. Yu, and D. Tao, “A survey of safety on large vision-language models: Attacks, defenses and evaluations,” arXiv preprint arXiv:2502.14881, 2025.

[31] Y. Zong, O. Bohdal, T. Yu, Y. Yang, and T. Hospedales, “Safety finetuning at (almost) no cost: A baseline for vision large language models,” arXiv preprint arXiv:2402.02207, 2024.

[32] B. Chen, X. Lyu, L. Gao, J. Song, and H. T. Shen, “Safeptr: Token-level jailbreak defense in multimodal llms via prune-then-restore mechanism,” arXiv preprint arXiv:2507.01513, 2025.

[33] Y. Gong, D. Ran, J. Liu, C. Wang, T. Cong, A. Wang, S. Duan, and X. Wang, “Figstep: Jailbreaking large vision-language models via typographic visual prompts,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 22, 2025, pp. 23 951–23 959.

[34] Z. Zhang, Z. Sun, Z. Zhang, J. Guo, and X. He, “Fc-attack: Jailbreaking multimodal large language models via auto-generated flowcharts,” arXiv preprint ArXiv:2502.21059, 2025.

[35] Y. Wang, X. Zhou, Y. Wang, G. Zhang, and T. He, “Jailbreak large vision-language models through multi-modal linkage,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 1466–1494.

[36] Y. Li, C. Yang, and A. Ettinger, “When hindsight is not 20/20: Testing limits on reflective thinking in large language models,” in Findings of the Association for Computational Linguistics: NAACL 2024, 2024, pp. 3741–3753.

[37] Y. Xu, X. Qi, Z. Qin, and W. Wang, “Cross-modality information check for detecting jailbreaking in multimodal large language models,” in Findings of the Association for Computational Linguistics: EMNLP 2024, 2024, pp. 13 715–13 726.

[38] E. Shayegani, Y. Dong, and N. Abu-Ghazaleh, “Jailbreak in pieces: Compositional adversarial attacks on multi-modal language models,” arXiv preprint arXiv:2307.14539, 2023.

[39] X. Qi, K. Huang, A. Panda, P. Henderson, M. Wang, and P. Mittal, “Visual adversarial examples jailbreak aligned large language models,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 19, 2024, pp. 21 527–21 536.

[40] X. Gu, X. Zheng, T. Pang, C. Du, Q. Liu, Y. Wang, J. Jiang, and M. Lin, “Agent smith: A single image can jailbreak one million multimodal llm agents exponentially fast,” arXiv preprint arXiv:2402.08567, 2024.

[41] Y. Pang, A. Xiong, Y. Zhang, and T. Wang, “Towards understanding unsafe video generation,” arXiv preprint arXiv:2407.12581, 2024.

[42] X. Liu, Z. Li, Z. He, P. Li, S. Xia, X. Cui, H. Huang, X. Yang, and R. He, “Video-safetybench: A benchmark for safety evaluation of video lvlms,” arXiv preprint arXiv:2505.11842, 2025.

[43] X. Wang, D. Wu, Z. Ji, Z. Li, P. Ma, S. Wang, Y. Li, Y. Liu, N. Liu, and J. Rahmel, “{SelfDefend}:{LLMs} can defend themselves against

jailbreaking in a practical manner,” in 34th USENIX Security Symposium (USENIX Security 25), 2025, pp. 2441–2460.

[44] X. Liu, X. Cui, P. Li, Z. Li, H. Huang, S. Xia, M. Zhang, Y. Zou, and R. He, “Jailbreak attacks and defenses against multimodal generative models: A survey,” arXiv preprint arXiv:2411.09259, 2024.

[45] D. Kiela, H. Firooz, A. Mohan, V. Goswami, A. Singh, P. Ringshia, and D. Testuggine, “The hateful memes challenge: Detecting hate speech in multimodal memes,” Advances in neural information processing systems, vol. 33, pp. 2611–2624, 2020.

[46] S. Ramamoorthy, N. Gunti, S. Mishra, S. Suryavardan, A. Reganti, P. Patwa, A. DaS, T. Chakraborty, A. Sheth, A. Ekbal et al., “Memotion 2: Dataset on sentiment and emotion analysis of memes,” in Proceedings of De-Factify: workshop on multimodal fact checking and hate speech detection, CEUR, vol. 17, 2022.

[47] R. Gomez, J. Gibert, L. Gomez, and D. Karatzas, “Exploring hate speech detection in multimodal publications,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2020, pp. 1470– 1478.

[48] L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg, “Modeling context in referring expressions,” in European conference on computer vision. Springer, 2016, pp. 69–85.

[49] L. Yu, Z. Lin, X. Shen, J. Yang, X. Lu, M. Bansal, and T. L. Berg, “Mattnet: Modular attention network for referring expression comprehension,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 1307–1315.

[50] T. Cao, B. Lim, Y. Liu, Y. Sui, Y. Li, S. Deng, L. Lu, N. Oo, S. Yan, and B. Hooi, “Vpi-bench: Visual prompt injection attacks for computer-use agents,” arXiv preprint arXiv:2506.02456, 2025.

[51] Y. Zhang, T. Yu, and D. Yang, “Attacking vision-language computer agents via pop-ups,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 8387– 8401. [Online]. Available: https://aclanthology.org/2025.acl-long.411/

[52] K. Gao, Y. Bai, J. Bai, Y. Yang, and S.-T. Xia, “Adversarial robustness for visual grounding of multimodal large language models,” arXiv preprint arXiv:2405.09981, 2024.

[53] A. Akula, S. Gella, Y. Al-Onaizan, S.-C. Zhu, and S. Reddy, “Words aren’t enough, their order matters: On the robustness of grounding visual referring expressions,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, Jul. 2020, pp. 6555–6565. [Online]. Available: https://aclanthology.org/2020.acl-main.586/

[54] L. Xiao, X. Yang, X. Lan, Y. Wang, and C. Xu, “Towards visual grounding: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[55] C. Chen, Y.-Y. Tseng, Z. Li, A. Venkatesh, and D. Gurari, “Acknowledging focus ambiguity in visual questions,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 1228–1238.

[56] W. Luo, S. Ma, X. Liu, X. Guo, and C. Xiao, “Jailbreakv: A benchmark for assessing the robustness of multimodal large language models against jailbreak attacks,” arXiv preprint arXiv:2404.03027, 2024.

[57] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang, “Mm-vet: Evaluating large multimodal models for integrated capabilities,” arXiv preprint arXiv:2308.02490, 2023.

[58] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 296–26 306.

[59] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee, “Llavanext: Improved reasoning, ocr, and world knowledge,” 2024.

[60] J. Gao, R. Pi, T. Han, H. Wu, L. Hong, L. Kong, X. Jiang, and Z. Li, “Coca: Regaining safety-awareness of multimodal large language models with constitutional calibration,” arXiv preprint arXiv:2409.11365, 2024.
# Explicit State Elicitation Is Not Enough: A Controlled Audit of Memory-Policy Classification

Yihang Chen ychen3726@gatech.edu

Chong Peng chongp@alumni.cmu.edu

Pin Qian pqian@alumni.cmu.edu

Su Wang suwang@alumni.cmu.edu

Huan Xu huan.xu71@gmail.com

Shuaiting Li list@zju.edu.cn

Yiqi Sun ysun697@gatech.edu

## Abstract

Personalized agents must decide whether retrieved user memory should be used, ignored, updated, or queried before it afects a current task. We use this setting to develop an empirical audit protocol for structured intermediate outputs: first audit dataset shortcuts, then isolate bundled prompt changes, check whether intermediate labels are answer-associated, test decomposed semantic evidence, and audit provider-level execution failures. A 480-example synthetic development set initially suggested large gains from a state-structured prompt bundle, but TF-IDF diagnostics showed lexical separability and no positive standalone Ignore cases. We therefore construct a frozen 160-example controlled counterfactual set with 40 matched four-way families and rule-derived reference policies. On this set, exposing the four state definitions improves accuracy, but an isolated explicit state-output field does not significantly improve policy accuracy for Llama-3.3-70B and gives only a marginal, non-significant gain for GPT-OSS-120B. Supplying benchmark-associated state labels shifts policy predictions, but because those labels deterministically map to policies, this is a label-conditioning diagnostic rather than evidence of a faithful internal mechanism. Family-level and seed-stability analyses further show that example-level accuracy overstates counterfactual consistency: complete four-way family success is rare. An exploratory follow-up that elicits decomposed semantic evidence also fails to improve routing for the cleanly evaluated endpoint; the corresponding GPT-OSS condition was unavailable because of provider-side request validation. We evaluate policy classification only, not downstream responses, tool actions, or memory-store mutation.

## 1 Introduction

Long-term-memory agents retrieve user histories before answering current requests. Retrieval alone does not determine how the memory should be used: it may be current, irrelevant, superseded, or insuficient for the task. We study this intermediate decision as memory-policy classification over {Use, Ignore, Update, Ask}. The experiments stop at policy prediction; they do not evaluate final responses, tool actions, or memory-store mutation.

Our starting point is MemoryPolicy-Bench, a 480-example synthetic development set. It suggested that a state-structured prompt bundle could improve exact-match policy accuracy over implicit-state/no-state-output prompts on Llama-3.3-70B and Qwen3-32B. A later audit changed the interpretation: the development set is highly lexically separable under supervised TF-IDF diagnostics, and it positively instantiates Use, Update, and Ask but has no standalone positive

Ignore reference class. We therefore treat it as development evidence and motivation, not as a decisive out-of-distribution test.

To test the prompt claim more directly, we freeze a 160-example controlled counterfactual set. It contains 40 scenario families, each with matched Use, Ignore, Update, and Ask variants. Reference policies are rule-derived from structured semantic relations before rendering. Each example has four history records, and the current task text is byte-identical across the four variants in a family. The set passes deterministic structural and semantic validation and was verified by two annotators, working separately and blind to the rule-derived reference policies, who agreed with each other and with the reference policy on all 160 examples; because the labels are deterministically rule-derived, this agreement partly reflects the construction. Under grouped diagnostics, the frozen controlled set remains harder for the evaluated lexical baselines, although this is not directly comparable to the original random-fold development-set diagnostic because the split, label space, and class priors difer. It is synthetic and rule-labeled.

The controlled results revise the paper’s claim and motivate a broader audit framing. In a matched prompt ablation, asking the model to output an explicit state field does not significantly improve policy accuracy for Llama and is only marginal for GPT-OSS. A separate label-conditioning diagnostic shows that policy routing is sensitive to externally supplied benchmark-associated state labels. This diagnostic is intentionally reported narrowly: because states in the frozen controlled set map deterministically to policies, a supplied state is information-equivalent to an answer label for an oracle that knows the deterministic state-to-policy mapping; note, however, that the model does not route on it perfectly — forced-correct-state accuracy is 52.7% (Llama) and 65.0% (GPT-OSS), since the forced-state prompt supplies the state but not the state-to-policy mapping. It does not establish that naturally emitted states are faithful internal mechanisms. We therefore analyze not only average accuracy but also family-level counterfactual consistency, stochastic seed stability, and rule-based error modes.

Our contributions are:

• We propose a reusable audit protocol (instantiated here on memory-policy classification) that separates dataset shortcuts, bundled prompt changes, answer-associated label conditioning, semantic-evidence elicitation, and provider-level execution failures.

• We instantiate the protocol on memory-policy classification with matched four-way counterfactual families, family-cluster analysis, stochastic-stability diagnostics, and rule-based errormode summaries.

• We show that apparent development-set state-structured gains do not survive an isolated stateoutput ablation, while supplied deterministic state labels should be interpreted as answerassociated conditioning rather than mechanism evidence.

## 2 Related Work

Long-term memory systems for LLM agents study how to store, organize, retrieve, and revise information across interactions. MemGPT treats memory as an operating-system-like hierarchy for moving information in and out of limited context windows (Packer et al., 2023); A-MEM develops dynamic indexing, linking, and memory evolution (Xu et al., 2025); LongMemEval evaluates extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention in long-term chat memory (Wu et al., 2025); and LoCoMo evaluates very long-term conversational memory over multi-session dialogues (Maharana et al., 2024). PERSONAMEM studies dynamic user profiles and personalized responses at scale (Jiang et al., 2025), while Mem2ActBench evaluates long-term memory use in task-oriented autonomous-agent actions (Shen et al., 2026). Recent multi-agent work also cautions that expanded recall can change downstream social behavior, so memory should be treated as an active determinant of agent behavior rather than a passive context extension (Liu et al., 2026a). These systems establish memory as an architectural requirement, whereas MemoryPolicy-Bench asks a complementary policy question: once a memory is retrieved, should it be used, queried, ignored, or updated?

Personalized language modeling and RAG benchmarks study how retrieved profile information supports outputs. LaMP evaluates personalized classification and generation with retrieval over user profiles (Salemi et al., 2024), PersonaLLM tests consistency under assigned personality traits (Jiang et al., 2024), and PrefEval evaluates whether models infer and follow user preferences in long-context conversations (Zhao et al., 2025). PrefIx similarly evaluates preference-aware humanagent interaction quality alongside task success, rather than treating task accuracy as the only endpoint (Li et al., 2026a). Retrieval-Augmented Generation combines parametric models with retrieved evidence (Lewis et al., 2020), but retrieved context can conflict with parametric knowledge, with other retrieved documents, or with itself (Xu et al., 2024); Self-RAG addresses part of this problem through adaptive retrieval and self-critique (Asai et al., 2024). Other retrieval-stage work constructs semantically weighted path evidence for multi-hop knowledge-graph QA while preserving path-level traces for diagnostics (Fu et al., 2026). Recent retrieval audits further show that evaluation can be confounded when retrieval units exceed an encoder’s efective input window, and that query-adaptive fusion need not reliably outperform fixed fusion under grouped evaluation (Wu and Lin, 2026). Related benchmark work in regulatory divergence detection distinguishes agreement, divergence, and directional silence over paired source texts (Wu et al., 2026). That work studies regulatory-document comparison rather than user memory, but it illustrates why conflict and absence can require distinct labels. Related concerns arise in adaptive systems under distribution shift: DriftGuard uses multiple safety-relevant monitors to trigger selective adaptation for evolving toxicity moderation, rather than relying on global distribution drift alone (Xin et al., 2026). These works are complementary to our setting: they study retrieval-stage evaluation, paired document comparison, or when a deployed model should adapt to changing input and risk distributions, whereas we audit how an agent should route retrieved user memory after retrieval.

Adjacent controlled evaluations outside memory agents make a similar methodological point: conclusions can depend on cohort and target definitions, on separating an intervention from confounded weighting changes, and on which metric is reported (Han et al., 2026b; Wu, 2026; Han et al., 2026a). We use these as methodological analogies for controlled evaluation design, not as evidence about memory-policy routing.

Clarification and calibration work motivate asking as a first-class action. CLAQUA supports deciding whether clarification is needed and how to ask it in knowledge-based QA (Xu et al., 2019), CLAM shows that generative models can selectively ask clarifying questions for ambiguous inputs (Kuhn et al., 2022), and self-knowledge work studies whether models know when they do not know (Kadavath et al., 2022). Knowledge-editing methods such as MEMIT directly alter model parameters to update stored facts (Meng et al., 2023); MemoryPolicy-Bench instead leaves the model unchanged and tests whether a prompted agent can classify the policy for a retrieved user memory at decision time.

Prompting work shows that explicit intermediate text can change model behavior. Chain-ofthought prompting elicits free-form reasoning traces (Wei et al., 2022), while Least-to-Most and Self-Ask decompose problems into intermediate subproblems or questions (Zhou et al., 2023; Press et al., 2023). More broadly, semantically equivalent prompt formulations can themselves induce systematic performance diferences: Jin et al. characterize this phenomenon as prompt privilege and study accessibility-oriented prompt normalization to reduce disparities across prompting styles (Jin et al., 2026). This sensitivity further motivates isolating prompt components before attributing behavioral changes to a particular intermediate representation. Our setting difers by scoring a discrete intermediate memory-state label against a benchmark state mapping, rather than treating unrestricted reasoning text as the object of evaluation.

Faithfulness and shortcut-learning work motivates our conservative interpretation. Shortcut learning describes models exploiting easy surface regularities that need not match the intended construct (Geirhos et al., 2020). Recent explanation-faithfulness evaluations similarly caution that an intermediate explanation or label can afect an answer without proving that it reflects the model’s internal decision process (Matton et al., 2025). Our forced-state diagnostic is therefore framed as sensitivity to supplied benchmark-associated labels, not as evidence that generated states are faithful mechanisms.

## 3 Task Formulation

Let $H = \{ h _ { 1 } , \ldots , h _ { n } \}$ be a retrieved user history and T the current task. A deployed assistant ultimately produces a response $R ,$ but this paper evaluates only an intermediate policy π. If response generation were evaluated, a policy-conditioned decomposition would be

$$
P ( R \mid H , T ) = \sum _ { \pi } P ( \pi \mid H , T ) P ( R \mid H , T , \pi ) .
$$

Our experiments stop at estimating $\pi .$

For state-structured prompting, the model may also emit a diagnostic state s:

$$
s \sim P ( s \mid H , T ) , \qquad \pi \sim P ( \pi \mid s , H , T ) .
$$

The benchmark-implied states are Active, Stale-or-Irrelevant, Conflicting-or-Superseded, and Underspecified. In the frozen controlled set these four labels map deterministically to the four reference policies, so they should be read as benchmark-associated state labels rather than an independent latent mechanism. The policy labels Use, Ignore, Update, and Ask are a benchmark-specific routing abstraction, not a complete, universal, or mutually exhaustive ontology for deployed memory systems. We distinguish generation categories, benchmark-implied states, and reference policies throughout.

## 4 Audit Protocol and Prompt Arms

We organize the study as a five-stage audit protocol for explicit intermediate outputs in agent decisions. Stage 1 is a dataset-construction audit: check class coverage, label distribution, lexical separability, template predictability, direct label leakage, grouped generalization, and transfer limitations before interpreting prompt gains. Stage 2 is matched prompt isolation: vary one component at a time, such as taxonomy exposure, explicit state output, rationale output, output ordering, or deterministic routing. Stage 3 is an answer-associated label audit: if an intermediate label is deterministically mapped to the target answer, forced-label conditions are reported as label-conditioning diagnostics rather than mechanism evidence. Stage 4 is a semantic-evidence follow-up: replace answer-associated labels with decomposed evidence fields such as relevance, temporal status, supersession, context suficiency, and risk or confirmation. Stage 5 is an operational audit: retain parse failures, provider request-validation failures, retries, terminal call outcomes, endpoint amendments, token/cost accounting, and release-package rebuild checks. Broader provenance work treats typed execution and evidence traces as the substrate for verification, debugging, audit, and recovery in LLM agents (Wang et al., 2026b); our retained records are a narrower provider-output accounting layer for this benchmark. Figure 1 gives the audit-protocol diagram.

![](images/0242f3e74e4b7b97a5f361b9e8a7b4449c7637d3361ef859e3cb09e3a35f0a9b.jpg)  
Figure 1: Five-stage audit protocol instantiated in this paper. The label-conditioning diagnostic is separated from semantic-evidence interventions because supplied benchmark-state labels map deterministically to policy labels and therefore do not test mechanism faithfulness.

All evaluated arms are single-call prompts that receive the history and task and return strict JSON. Historical internal names are retained only as artifact identifiers. The development-set arms include implicit-state/no-state-output prompts, a state-structured prompt bundle, and policy-only baselines. The old state-structured bundle changes several factors at once: state taxonomy, output schema, step wording, rationale requirements, prompt length, and policy descriptions. Its gains therefore cannot be attributed to the state field alone.

The frozen controlled set matched ablation uses five pre-specified arms frozen before the corresponding hosted evaluation: policy-only, taxonomy-only, state-output, state-output-with-rationale, and deterministic-routing with policy definitions. Policy definitions and formatting are held constant where possible. The primary matched comparison is state-output versus policy-only. The deterministic-routing arm asks only for a state and maps emitted states to policies in code. Exact artifact IDs and the state-to-policy mapping for these reader-facing names are listed in Appendix D.

The label-conditioning diagnostic uses six artifact conditions; Appendix U describes them and Appendix E gives the wrong-state construction.

## 5 Data

Development set. MemoryPolicy-Bench contains 480 synthetic examples generated with Gemini 1.5 Flash. It is balanced across Stable, Contextual, Updated, and Ambiguous generation categories, but the reference-policy distribution is 240 Use, 120 Update, 120 Ask, and 0 Ignore. Labels are generator-assigned reference policies. The plain Always-USE majority baseline reaches 50.0%, and a supervised TF-IDF diagnostic reaches 89.6±3.5% policy accuracy under random folds. This construct-validity warning means the set is lexically separable and used here as development evidence only. Appendix Q reports the historical development-set prompt-bundle result and why it is treated as motivation only.

Frozen controlled set. We construct a separate frozen counterfactual set with 160 examples: 40 scenario families crossed with Use, Ignore, Update, and Ask. Reference policies are derived deterministically from structured semantic slots before natural-language rendering. Every example has exactly four history records: an earlier temporal record, a later temporal record, and two context-specific records. The task text is byte-identical across the four variants in each family; only the semantic relation between history and task changes; Appendix T gives the representative four-way family.

The controlled set is synthetically generated and rule-labeled. It passes deterministic structural and semantic validation, including policy balance, four records per example, identical within-family task text, type compatibility, and no direct label leakage. Under grouped diagnostics, the frozen controlled set remains harder for the evaluated lexical baselines: word TF-IDF macro F1 is 0.390 and character n-gram macro F1 is 0.368 in the controlled-set audit. This should not be read as a strict magnitude comparison to the development-set random-fold diagnostic, because the development set has no positive Ignore reference class and difers in split structure, label space, and priors. Appendix M reports a development-set-to-controlled-set TF-IDF transfer diagnostic. Each of the 160 examples was verified by two annotators, working separately and blind to the rule-derived reference policies; the annotators agreed with each other and with the reference policy on every example. Labels remain rule-derived and may not capture every reasonable human judgment.

## 6 Experimental Setup

The development-set evidence uses Groq-hosted Llama-3.3-70B (llama-3.3-70b-versatile) and Qwen3-32B (qwen/qwen3-32b). The frozen controlled set experiments use Llama-3.3-70B and GPT-OSS-120B (openai/gpt-oss-120b). Qwen3-32B was replaced before any controlled-set hosted result existed because the endpoint became unavailable on the current Groq tier; GPT-OSS results are never labeled as Qwen results. Llama is the bridge endpoint across both phases.

The Llama endpoint is documented by the Llama 3 family report and the Llama-3.3-70B-Instruct model card (Grattafiori et al., 2024; Meta Llama, 2024). The development Qwen endpoint is documented by the Qwen3 technical report (Yang et al., 2025). The GPT-OSS endpoint is documented by the oficial GPT-OSS model card (OpenAI, 2025).

All hosted controlled-set runs use the frozen 160 examples, seeds 101/102/103, temperature 0.7, hash-recorded frozen prompt templates, strict JSON parsing, raw response logging, finishreason logging, token accounting, and resumable call IDs. Parse failures remain visible and count as incorrect. The matched ablation contains 4,800 calls; the label-conditioning diagnostic contains 5,760 calls. Frozen-core hosted cost including two smoke calls is estimated at \$1.8007 under a \$20.00 hard budget.

Primary uncertainty uses scenario-family clustering. We report paired accuracy diferences, 10,000-resample percentile bootstrap 95% intervals, and paired family-level sign-flip permutation tests with plus-one correction. Non-significant p-values are not interpreted as equivalence.

## 7 Results

Policy-only denotes distinct baselines in the matched ablation, label-conditioning diagnostic, and semantic-evidence follow-up. These protocols use diferent prompt templates; the development comparison also changes label space and priors, while semantic-evidence uses a diferent rendering/template with higher lexical separability despite retaining balanced policies. Their accuracies are therefore not directly comparable; the label-conditioning table note also separates the two frozen protocol sources, and the semantic-evidence policy-only value is reported only for Llama.

Tables 1 and 2 summarize the frozen controlled set results. The matched state-output ablation does not support a universal prompt recipe. State-output improves over clean policy-only by only 0.6 percentage points for Llama, with a 95% family-cluster interval of [-2.9, 4.0] and p=0.8084. GPT-OSS shows a larger positive diference of 3.3 points, but the interval includes zero and the paired permutation test is not significant (p=0.0710; Holm-adjusted p=0.1420). We therefore do not claim that merely adding an explicit state output consistently improves policy accuracy.

Taxonomy-only shows a diferent efect: exposing the four state definitions without requiring a state output improves accuracy on both endpoints (Llama 44.0% to 53.1%, +9.17 points, 95% CI [5.00, 13.33], raw p=0.0003, Holm-adjusted p=0.0012; GPT-OSS 54.8% to 59.8%, +5.00 points, 95% CI [1.67, 8.33], raw p=0.0045, Holm-adjusted p=0.0135). This dissociates taxonomy exposure from state emission, but because the four benchmark states map one-to-one onto the four policies, taxonomy exposure may convey label structure rather than useful typed knowledge.

The label-conditioning diagnostic shows sensitivity to externally supplied benchmark-associated state labels. Supplying the benchmark-implied label improves routing by 7.3 points for Llama and 11.9 points for GPT-OSS, with cluster-aware intervals excluding zero. Correct and conflicting supplied labels produce routing gaps of 15.2 and 15.0 points for Llama and GPT-OSS, respectively, relative to forced-correct routing. Relative to policy-only (label-conditioning protocol) rather than forced-correct routing, forced-wrong-state accuracy is 7.9 points lower for Llama (95% CI [-11.9, -4.2], p < .001) and 3.1 points lower for GPT-OSS (95% CI [-7.1, 0.8], p=0.1582). Because the four benchmark states deterministically map to the four policies, these contrasts do not establish mechanism faithfulness.

<table><tr><td>Model</td><td></td><td></td><td>Clean acc. Taxonomy acc. Taxonomy ∆ vs clean</td><td></td><td>State acc. State ∆ vs clean</td><td></td></tr><tr><td>Llama-3.3-70B</td><td>44.0%</td><td>53.1%</td><td>+9.17 pp [5.00, 13.33] raw p=0.0003; Holm</td><td>44.6%</td><td>0.6 pp [-2.9, 4.0]</td><td>raw p=0.8084; Holm</td></tr><tr><td>GPT-OSS-120B</td><td>54.8%</td><td>59.8%</td><td>p=0.0012 +5.00 pp [1.67, 8.33] raw p=0.0045; Holm p=0.0135</td><td>58.1%</td><td>p=0.8084 3.3 pp [0.0, 6.7]</td><td>raw p=0.0710; Holm</td></tr></table>

Table 1: Primary matched ablation on the frozen controlled set. Accuracies are exact-match policy accuracies. $\Delta$ is each arm minus clean policy-only in percentage points; parse failures count as incorrect. Parse failures are 0 for clean policy-only and state-output on both endpoints; taxonomyonly has 1 parse failure for Llama and 0 for GPT-OSS. Intervals are 95% scenario-family-cluster bootstrap intervals; p-values are paired family-level permutation tests. Holm adjustment is over the four arm-versus-clean contrasts shown for taxonomy-only and state-output across the two endpoints; the state-output Holm values are unchanged under this expanded family. All reported percentagepoint diferences in the paper are computed from unrounded accuracies; displayed rounded endpoints may not reproduce them exactly.

<table><tr><td>Model</td><td>Policy-only</td><td>Supplied ref.</td><td>Supplied conflict</td><td>State-first</td></tr><tr><td>Llama-3.3-70B</td><td>45.4%</td><td>52.7%</td><td>37.5%</td><td>42.7%</td></tr><tr><td>GPT-OSS-120B</td><td>53.1%</td><td>65.0%</td><td>50.0%</td><td>59.2%</td></tr><tr><td>Model</td><td>Ref. vs policy</td><td>Conflict vs ref.</td><td>Conflict vs policy</td><td>State-first (∆)</td></tr><tr><td>Llama-3.3-70B</td><td>7.3 pp [2.5, 12.1] p=0.0065</td><td>-15.2 pp [-20.4, -10.0] p &lt; .001</td><td>-7.9 pp [-11.9, -4.2] p &lt; .001</td><td>-2.7 pp p=0.0825</td></tr><tr><td>GPT-OSS-120B</td><td>11.9 pp [8.3, 15.4] p &lt; .001</td><td>-15.0 pp [-20.0, -10.2] p &lt; .001</td><td>-3.1 pp [-7.1, 0.8] p=0.1582</td><td>6.0 pp p=0.0025</td></tr></table>

Table 2: Label-conditioning sensitivity diagnostic on the frozen controlled set. The suppliedreference and supplied-conflict columns are absolute accuracies when the prompt receives a benchmark-associated state label. Delta columns are percentage-point diferences on matched controlled-set decisions; conflict-vs-reference is relative to supplied-reference routing, not policyonly. Because controlled-set state labels map deterministically to policies, this diagnostic tests sensitivity to supplied benchmark-associated labels rather than an internal causal state mechanism.

Natural state elicitation remains model-dependent. State-first is negative or neutral for Llama relative to policy-only (-2.7 points, p=0.0825), but positive for GPT-OSS (+6.0 points, p=0.0025). Benchmark-mapped state agreement in this condition is 41.5% for Llama and 55.8% for GPT-OSS. In the state-only deterministic-routing condition (label-conditioning protocol), where policy is computed from the emitted state, policy accuracy is 46.9% for Llama and 52.5% for GPT-OSS. This pattern supports the narrower conclusion that explicit state elicitation is not enough: accurately predicting the benchmark-defined state remains dificult for the evaluated models. The old development-set prompt-bundle gain cannot be reduced to the state output field because the bundle changed multiple prompt dimensions and was measured on a lexically separable development set.

The semantic-evidence follow-up protocol replaces the old four-state label with decomposed evidence fields for task relevance, temporal evidence, supersession, context suficiency, and risk or confirmation. Explicit semantic-evidence output reduces accuracy relative to policy-only by 7.5 percentage points (95% CI [-12.7, -2.5], Holm-adjusted p=0.0146 within the three arm comparisons in this follow-up protocol), while explicit output difers little from taxonomy-only prompting (+1.3 points, p=0.6079). The taxonomy-only arm itself is lower than policy-only by 8.8 points (95% CI [- 12.1, -5.6], Holm-adjusted p=0.0003). Thus, for the cleanly evaluated endpoint, degradation is more consistent with taxonomy exposure or the added structured decision frame than with a benefit from mandatory intermediate serialization. For GPT-OSS, the explicit-evidence arm produced providerside JSON validation failures for all attempted records, so that endpoint does not yield a clean behavioral comparison for this arm. We therefore treat semantic-evidence elicitation as negative for Llama and unavailable for GPT-OSS, not as evidence that decomposed intermediate fields solve the leakage concern. Appendix N gives the full follow-up protocol, design audit, and per-arm accuracies.

## 7.1 Counterfactual Consistency and Failure Analysis

The four-way family structure exposes failures that average example accuracy hides. In the primary matched ablation, Llama solves no complete four-variant family in either policy-only or state-output form; GPT-OSS solves 1.7% of family-seed units in policy-only form and 5.0% with state output. Most families have only two of four variants correct: under policy-only prompting, Llama has 2/4 correct variants in 65.0% of family-seed units and never reaches 4/4.

Seed stability and error taxonomy further localize failures. In the primary matched ablation, Llama policy-only predictions are unanimous across seeds on 69.4% of examples, while state-output predictions are unanimous on 76.2%; GPT-OSS shows 83.1% and 74.4%. Among memory-reasoning error modes, the most frequent by count are conflict over-detection, task neglect (the model ignores the current task text), and recency overreach, followed by ask-defaulting and irrelevance-to-Ask confusion (see Appendix J). These are non-exclusive rule-based diagnostics from metadata and predictions, not human-coded annotations, and their counts must not be summed.

## 8 Discussion

Supplied benchmark-associated labels are not self-eliciting states. The forced-state artifact conditions show that policy routing is sensitive to externally supplied benchmark-associated labels: when the benchmark-implied label is supplied by the experimenter, routing improves, and a conflicting supplied label shifts predictions away from the reference mapping. The matched prompt ablation shows a diferent fact: asking the model to emit a state field does not consistently produce the needed improvement. These results are compatible because accurately predicting the benchmark-defined state remains dificult for the evaluated models. The rendered taxonomy prompt lists the four states in the same order as the four policies and closely paraphrases them; the mapping is recoverable without being stated, so the taxonomy-only gain is confounded with definition restatement. For example, Conflicting-or-Superseded mirrors Update’s newer-overrides-older wording, and Stale-or-Irrelevant explicitly says “no clarification is needed,” separating Ignore from Ask.

Conflicting supplied labels can shift policy predictions. The forced-wrong-state condition creates an approximately 15-point routing gap relative to the forced-correct-state condition for both endpoints. Relative to policy-only, it is lower by 7.9 points for Llama and 3.1 points for GPT-OSS. This cautions against treating explicit benchmark-state outputs as an automatic auditing guarantee. Externally supplied benchmark-associated labels can shift policy predictions, including when the supplied label conflicts with the reference mapping; systems need state-quality checks, abstention, or clarification mechanisms rather than state fields alone.

Model dependence matters. GPT-OSS benefits from natural state-first prompting on the frozen controlled set, while Llama does not. Llama also performs better with policy-first ordering than state-first ordering in the diagnostic table. This makes a universal “state first” prompting recommendation scientifically too strong. The per-policy decomposition makes this concrete: the state field pushes each model toward a diferent policy and away from another, and even flips the sign of its efect on Ask across models, so its efect is a model-specific bias shift rather than a general gain.

Family-level analysis changes the error picture. Example-level accuracy makes the task look like a partially solved four-class classification problem, but family-level exact match (Table 8) shows that models rarely solve all four matched counterfactual variants in the same scenario. Pairwise transition analysis further separates a correct policy change from merely changing to another wrong label. This is why matched counterfactual families are useful as an audit unit: they reveal whether a prompt tracks the intended semantic relation, rather than merely improving marginal accuracy.

State output redistributes policies rather than improving them. The overall matchedablation null hides opposing class-level efects that cancel. Adding a state field significantly shifts Llama toward Use (+17.5 pp, 95% CI [6.67, 28.33], Holm p=0.010, within-model four-policy family) and away from Ask (-14.2 pp, 95% CI [-22.50, -6.67], Holm p=0.010), while it shifts GPT-OSS toward Ignore (+15.8 pp, 95% CI [7.50, 25.00], Holm p=0.003) and away from Update (-8.3 pp, 95% CI [- 13.33, -3.33], Holm p=0.010). The gains and losses are of similar magnitude and net to the near-zero overall change. The Ask policy makes the mechanism concrete and shows it is model-specific in sign: with a state field, Llama’s Ask output nearly collapses (precision/recall/F1 fall from 35.3/15.0/21.1 to 2.9/0.8/1.3), whereas GPT-OSS’s Ask precision/recall rises (24.53/10.83 to 44.19/15.83). The same intervention therefore suppresses a policy in one model and sharpens it in another, so it is a model-specific bias shift rather than a general gain — which is why family-level exact match stays low and why a universal “state-first” recipe is unwarranted.

The bottleneck is the decision mapping, not information availability. We also tested the opposite intervention: instead of eliciting structure on the output side, we normalized the input, replacing the natural-language history with explicit [subject|attribute|value|tag] tuples derived from the same structured slots (a probe that does not answer-leak: grouped lexical decoders reach only 0.375 word / 0.33125 character accuracy). This did not help and significantly hurt: normalized input lowered policy accuracy by 8.5 points for Llama-3.3-70B (95% CI [-13.96, -3.54], p=0.0034) and 7.9 points for GPT-OSS-120B (95% CI [-12.08, -3.54], p=0.0012). Together with the null output-side ablation and the imperfect forced-correct-state routing, this is consistent with a decision-mapping bottleneck: making relations explicitly available on the input side did not help. Appendix I gives the protocol and rendered prompt.

## 9 Conclusion

The strongest supported conclusion is not that explicit state output reliably improves memory-policy classification. On a frozen four-policy controlled counterfactual set, the matched state-output ablation is null for Llama and marginal for GPT-OSS. A separate label-conditioning diagnostic shows that benchmark-associated supplied labels shift routing, and correct versus conflicting supplied labels create an approximately 15-point routing gap relative to the forced-correct condition. Because these labels deterministically map to policies, this is not evidence of a faithful internal state mechanism. Explicit state elicitation requires more than adding a field to the JSON schema.

This reframes MemoryPolicy-Bench as an audit of memory-policy decisions and of the intermediate fields used to justify them. Development-set prompt-bundle gains are useful signals, but the controlled counterfactual evidence shows that future progress should target robust semantic-evidence extraction, calibrated asking, provider-compatible structured-output protocols, and downstream response/action evaluation rather than relying on a generic state-first prompt recipe. More broadly, the five-stage audit protocol is proposed and instantiated here on memory-policy classification; within this task family, it provides a reusable way to separate dataset shortcuts, bundled prompt changes, answer-associated labels, stochastic instability, and provider failures before making claims about structured intermediate outputs.

## Limitations

Both datasets are synthetic. The frozen controlled set is controlled and counterfactual rather than naturalistic conversation data. Its labels are rule-derived from structured semantics, deterministically validated, and verified by two annotators (blind to the reference labels) who agreed with each other and with the reference policies on all 160 examples. Rule-derived labels encode the benchmark designers’ intended policy abstraction and may not capture every reasonable human judgment.

The endpoint panel difers across phases. The development set uses Llama-3.3-70B and Qwen3- 32B; the frozen controlled set uses Llama-3.3-70B and GPT-OSS-120B after the Qwen endpoint became unavailable before any controlled-set hosted result existed. This weakens direct two-model replication claims across phases. A third endpoint (llama-3.1-8b-instant) reproduces the direction of the state-output efect in Appendix F.2, but with elevated parse failures, so cross-model state-output behavior remains model-dependent.

The experiments evaluate policy classification only. They do not measure downstream response quality, whether old memory is actually applied or suppressed, tool-action success, retrieval quality, or memory-store mutation. Recent operational systems address transactional memory commits, provenance-aware shared memory, rollback repair from faulty memories, and proactive recovery from multi-step execution failures (Li et al., 2026b; Wang et al., 2026a; Yu et al., 2026; Xu et al., 2026); those settings mutate persistent state or execute actions, whereas our benchmark stops before that layer. The forced-state artifact conditions also idealize state availability and are best interpreted as label-conditioning sensitivity diagnostics: because benchmark states map deterministically to policies, they do not guarantee that a deployed model will infer or use a faithful internal state from natural text.

The four-policy taxonomy is intentionally narrow. Real systems may need multiple actions at once: an update may also require using the new value, ignoring one memory may still require asking about another field, and high-risk tasks may require confirmation even when a memory appears active. Deletion, forgetting, privacy restrictions, scope constraints, and multi-turn repair are outside the current benchmark. Exact-match single-label scoring therefore does not cover the full strategy space of deployed memory agents. Independent third-party annotation would be useful future validation, but is not claimed in the current paper.

Finally, hosted API behavior and model endpoints may evolve. The local run records preserve raw responses, finish reasons, token counts, parse failures, call IDs, and cost logs so that later runs can be audited against the frozen protocol. The semantic-evidence follow-up further illustrates this operational limitation: GPT-OSS returned provider-side JSON validation failures for the explicitevidence arm under the frozen request format. We retain those records in execution accounting rather than changing the prompt after seeing outcomes, but we do not interpret that endpoint-arm failure as model policy behavior.

## Ethics Statement

This work studies long-term user memory, a setting that can create privacy and autonomy risks if deployed carelessly. MemoryPolicy-Bench is synthetic and does not contain real user logs, private conversations, or personally identifiable information. This reduces immediate data exposure, but it also means our conclusions should be validated before deployment in systems that store real user histories.

The proposed framework is intended to reduce inappropriate memory use by making Ignore and Ask explicit policy options. However, state classification itself requires processing potentially sensitive history. Practical systems should minimize retained data, support user inspection and deletion of memories, and avoid applying personal information outside the context for which it was provided. Diferential privacy and other user-level protections (McMahan et al., 2018) are complementary to this policy-layer approach.

There is also a user-experience risk. Agents that over-ask may frustrate users or shift cognitive labor onto them, while agents that under-ask may apply stale or irrelevant preferences. More autonomous downstream agents add further interaction surfaces: adjacent Text-to-SQL and embodiedcontrol work studies iterative refinement, multi-agent collaboration, and long-horizon action failures, whereas our audit stops before response generation, tool execution, or environment mutation (Su et al., 2026; Liu et al., 2026b). Our results suggest that calibrated clarification should be evaluated as a precision-recall problem, not as a simple preference for more questions. Documentation of the synthetic generation process and intended use should accompany any future public code or data release.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=hSyW5go0v8.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom B. Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, and Colin Rafel. Extracting training data from large language models. In USENIX Security Symposium, pages 2633–2650, 2021. URL https://www.usenix.org/conference/usenixsecurity21/presentation/ carlini-extracting.

Rong Fu, Yemin Wang, Tianxiang Xu, Yongtai Liu, Weizhi Tang, Wangyu Wu, Xiaowen Ma, and Simon Fong. S-Path-RAG: Semantic-aware shortest-path retrieval augmented generation for multi-hop knowledge graph question answering. In Proceedings of the ACM Web Conference

2026, pages 4057–4068. ACM, April 2026. doi: 10.1145/3774904.3792459. URL https://doi.org 10.1145/3774904.3792459.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020. doi: 10.1038/s42256-020-00257-z. URL https: //www.nature.com/articles/s42256-020-00257-z.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https: //arxiv.org/abs/2407.21783.

Xiao Han, Jingjing Liu, Moxuan Zheng, Zhen Zhang, and Chenyu Wu. Interpretable vs learned encoders for high-cardinality fraud detection, 2026a. URL https://doi.org/10.48550/arXiv.2607. 00477.

Xiao Han, Yao Xiao, Chenyu Wu, and Tongchen Zhang. How early is early enough? designdependent observation-window suficiency in subscription churn prediction, 2026b. URL https: //doi.org/10.48550/arXiv.2607.00473.

Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J. Taylor, and Dan Roth. Know me, respond to me: Benchmarking LLMs for dynamic user profiling and personalized responses at scale. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=6ox8XZGOqP.

Hang Jiang, Xiajie Zhang, Xubo Cao, Cynthia Breazeal, Deb Roy, and Jad Kabbara. PersonaLLM: Investigating the ability of large language models to express personality traits. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 3605–3627, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-naacl.229. URL https://aclanthology.org/2024.findings-naacl.229/.

Lier Jin, Lan Hu, Binqi Shen, Hanyu Cai, and Yuting Xin. Same question, diferent answer? measuring and mitigating prompt privilege for equitable ai access. arXiv preprint arXiv:2608.08942, 2026. doi: 10.48550/arXiv.2608.08942. URL https://arxiv.org/abs/2608.08942.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022. URL https: //arxiv.org/abs/2207.05221.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. CLAM: Selective clarification for ambiguous questions with generative language models. arXiv preprint arXiv:2212.07769, 2022. URL https: //arxiv.org/abs/2212.07769.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020. URL https://proceedings. neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html.

Jialin Li, Zhenhao Chen, Hanjun Luo, and Hanan Salam. PrefIx: Understand and adapt to user preference in human-agent interaction, 2026a. URL https://doi.org/10.48550/arXiv.2602.06714.

Xiaoyang Li, Yiqi Wang, Haohui Lu, Zhi Chen, Mo Li, Pingan Song, Mingkai Zheng, and Taotao Cai. MemTX: Transactional belief commit for stateful agent memory, 2026b. URL https://arxiv. org/abs/2607.23929.

Jiayuan Liu, Tianqin Li, Shiyi Du, Xin Luo, Haoxuan Zeng, Emanuel Tewolde, Tai Sing Lee, Tonghan Wang, Carl Kingsford, and Vincent Conitzer. The memory curse: How expanded recall erodes cooperative intent in LLM agents. arXiv preprint arXiv:2605.08060, 2026a. doi: 10.48550/arXiv.2605.08060. URL https://doi.org/10.48550/arXiv.2605.08060.

Yuanzhe Liu, Jingyuan Zhu, Yuchen Mo, Gen Li, Xu Cao, Jin Jin, Yifan Shen, Zhengyuan Li, Tianjiao Yu, Wenzhen Yuan, Fangqiang Ding, and Ismini Lourentzou. PALM: Progress-aware policy learning via afordance reasoning for long-horizon robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28096– 28110, June 2026b. URL https://arxiv.org/abs/2601.07060.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.747. URL https://aclanthology.org/2024.acl-long.747/.

Katie Matton, Robert Ness, John Guttag, and Emre Kiciman. Walk the talk? measuring the faithfulness of large language model explanations. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ b5ec50eb177908f21f78ed0d76ed525c-Abstract-Conference.html.

H. Brendan McMahan, Daniel Ramage, Kunal Talwar, and Li Zhang. Learning diferentially private recurrent language models. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=BJ0hF1Z0b.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=MkbcAHIYgyS.

Meta Llama. Llama 3.3 70b instruct model card. https://huggingface.co/meta-llama/Llama-3.3- 70B-Instruct, 2024. Oficial model card; release date December 6, 2024.

OpenAI. gpt-oss-120b and gpt-oss-20b model card. https://openai.com/index/gpt-oss-model-card/, 2025. Oficial model card, August 5, 2025.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023. URL https://arxiv.org/abs/2310.08560.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5687–5711. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.findings-emnlp.378. URL https://aclanthology.org/ 2023.findings-emnlp.378/.

Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. LaMP: When large language models meet personalization. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7370–7392, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.399. URL https://aclanthology.org/2024.acl-long.399/.

Yiting Shen, Kun Li, Wei Zhou, and Songlin Hu. Mem2ActBench: A benchmark for evaluating longterm memory utilization in task-oriented autonomous agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8173– 8190. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.370. URL https://aclanthology.org/2026.acl-long.370/.

Robin Staab, Mark Vero, Mislav Balunović, and Martin Vechev. Beyond memorization: Violating privacy via inference with large language models. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=kmn0BhQk7p.

Yiyun Su, Huiying Zhu, Yu Tian, Changruo Zhao, Zujun Peng, Yuting Liu, Luyan Zhang, Liang Fan, and Baihua Li. Agentic-SQL taxonomy: The research of autonomous and interactive text-to-SQL with LLMs. In De-Shuang Huang, Chuanlei Zhang, Wei Chen, Bo Li, Qinhu Zhang, Wenzheng Bao, Yijie Pan, and Prashan Premaratne, editors, Advanced Intelligent Computing Technology and Applications, volume 16655 of Lecture Notes in Computer Science, pages 323–334, Singapore, 2026. Springer Nature Singapore. ISBN 978-981-92-3438-7. doi: 10.1007/978-981-92-3438-7\_27. URL https://doi.org/10.1007/978-981-92-3438-7\_27.

Yiqi Wang, Zihao Yan, Jiaqi Zhang, Zhangkai Wu, Mingkai Zheng, Zequn Sun, Yanming Zhu, and Taotao Cai. MAP-Graph: Provenance-aware shared memory for multi-agent workflows, 2026a. URL https://arxiv.org/abs/2608.10509.

Yiqi Wang, Jiaqi Zhang, Taotao Cai, Zirui Liu, Qingqiang Sun, Zequn Sun, Zhangkai Wu, Manqing Dong, Mingkai Zheng, Xuefei Yin, and Yanming Zhu. From agent traces to trust: A survey of evidence tracing and execution provenance in LLM agents, 2026b. URL https://arxiv.org/abs/ 2606.04990.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash 9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html.

Chenyu Wu. Class weighting versus amount conditioning in credit-card fraud detection: A dollarmetric study with a temporal explanation audit, 2026. URL https://doi.org/10.48550/arXiv. 2607.14686.

Chenyu Wu and You Lin. Evidence-unit fairness and the limits of query-adaptive sparse-dense fusion in financial document retrieval, 2026. URL https://arxiv.org/abs/2608.00183.

Chuchu Wu, Zhiyin Zhou, Jingzhuo Hu, and Liang You. RegDivergence-101: An LLM benchmark for cross-jurisdiction regulatory contradiction detection in life sciences, 2026. URL https://www. researchgate.net/doi/10.13140/RG.2.2.32171.20000. Preprint.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html.

Yuting Xin, Hanyu Cai, Binqi Shen, Lier Jin, and Lan Hu. Driftguard: Safety-aware multimonitor detection and selective adaptation for evolving toxicity moderation. arXiv preprint arXiv:2606.28725, 2026. doi: 10.48550/arXiv.2606.28725. URL https://arxiv.org/abs/2606. 28725.

Jingjing Xu, Yuechen Wang, Duyu Tang, Nan Duan, Pengcheng Yang, Qi Zeng, Ming Zhou, and Xu Sun. Asking clarification questions in knowledge-based question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1618–1629, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1172. URL https://aclanthology.org/D19-1172/.

Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu. Knowledge conflicts for LLMs: A survey. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8541–8565, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.486. URL https://aclanthology.org/2024.emnlp-main.486/.

Sheng Xu, Ruixing Jin, Huayi Zhou, Bo Yue, Guanren Qiao, Yunxin Tai, Yueci Deng, Kui Jia, and Guiliang Liu. From reaction to anticipation: Proactive failure recovery through agentic task graph for robotic manipulation, 2026. URL https://arxiv.org/abs/2605.11951. Accepted to RSS 2026.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-MEM: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 19909c36f51abc4856b4560af3d36d6-Abstract-Conference.html.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https: //arxiv.org/abs/2505.09388.

Caili Yu, Yiqi Wang, Jiaqi Zhang, Yiqun Duan, Mingkai Zheng, Zhangkai Wu, Kaize Shi, and Taotao Cai. From faulty memories to corrected actions: Dependency-guided rollback repair for memory-augmented agents, 2026. URL https://arxiv.org/abs/2608.10502.

Siyan Zhao, Mingyi Hong, Yang Liu, Devamanyu Hazarika, and Kaixiang Lin. Do LLMs recognize your preferences? evaluating personalized preference following in LLMs. In Interna-

tional Conference on Learning Representations, 2025. URL https://openreview.net/forum?id QWunLKbBGF.

Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WZH7099tgfM.

Appendix section headings retain artifact identifiers; Appendix D maps them to reader-facing names.

## A Controlled V3 Construction and Validation

Frozen V3 contains 160 synthetically generated examples arranged as 40 four-way scenario families. Each family has one Use, one Ignore, one Update, and one Ask variant. The task string is byte-identical across the four variants, and each example has exactly four history records: an earlier temporal record, a later temporal record, and two context-specific records. Reference policies are derived from structured semantic relations before rendering, not assigned by inspecting model outputs.

The approved frozen set is rule-derived, deterministically validated, and was verified by two annotators, working separately and blind to the rule-derived reference policies, who agreed with each other and with the reference policy on all 160 examples. Annotation was not performed by an independent third party, and labels remain rule-derived. Structural validation checks policy balance, four records per example, identical within-family task strings, type-compatible slot values, template balance, duplicate and near-duplicate diagnostics, and absence of direct policy/state label leakage.

## B Frozen-V3 Matched-Ablation Prompts

All matched-ablation arms instantiate the same history/task prefix:

User History:   
{history}   
Current Task: {task}

The policy definitions are:

\- Use: apply the relevant memory.

\- Ignore: complete the task without the memory.

\- Update: acknowledge that newer information supersedes the old memory and use the newer

information.

\- Ask: request clarification before acting.

When a state taxonomy is included, the matched-ablation runner uses this exact heading and definitions:

Memory state options:

\- Active: a relevant memory is current and applies to this task.

\- Stale\_or\_Irrelevant: the memory should not be applied and no clarification is needed.

\- Conflicting\_or\_Superseded: newer information explicitly overrides older memory.

\- Underspecified: the history does not contain enough information to decide.

<sup>Clean</sup>\_<sup>Policy</sup>\_<sup>Only.</sup>

User History:

{history}

Current Task: {task}

Policy options:

{policy definitions}

Output strictly valid JSON with no markdown:

{"policy": "Use|Ignore|Update|Ask"}

User History:   
{history}   
Current Task: {task}

Output strictly valid JSON with no markdown:

```javascript
"policy": "Use|Ignore|Update|Ask"}
```

Policy options:   
{policy definitions}

## <sup>Taxonomy</sup>\_<sup>No</sup>\_<sup>State</sup>\_<sup>Output.</sup>

User History:   
{history}   
Current Task: {task}   
Memory state options:   
{state definitions}   
Policy options:   
{policy definitions}   
Output strictly valid JSON with no markdown:   
{"policy": "Use|Ignore|Update|Ask"}

For auditability, the following is the verbatim taxonomy-only prompt rendered by the existing runner for example controlled\_v3\_02\_use.

## User History:

Jordan saved an earlier entry: flight seat choice uses an aisle seat for its flight seat choice option.

The later note records an aisle seat for the flight seat choice option of flight seat choice.

Case B: the profile lists I sit near an exit as the train seat option for train seat.

Current Task: Choose the preference option for Jordan's flight seat choice.

## Memory state options:

\- Active: a relevant memory is current and applies to this task.

\- Stale\_or\_Irrelevant: the memory should not be applied and no clarification is needed.

\- Conflicting\_or\_Superseded: newer information explicitly overrides older memory.

\- Underspecified: the history does not contain enough information to decide.

## Policy options:

\- Use: apply the relevant memory.

\- Ignore: complete the task without the memory.

\- Update: acknowledge that newer information supersedes the old memory and use the newer information.

\- Ask: request clarification before acting.

Output strictly valid JSON with no markdown:

{"policy": "Use|Ignore|Update|Ask"}

## <sup>State</sup>\_<sup>Output</sup>\_<sup>No</sup>\_<sup>Rationale.</sup>

Memory state options:

{state definitions}

{"state": "Active|Stale\_or\_Irrelevant|Conflicting\_or\_Superseded|Underspecified",

```jsonl
User History:
{history}
Current Task: {task}
Memory state options:
{state definitions}
Policy options:
{policy definitions}
Step 1. Classify the memory state.
Step 2. Choose the policy.
Output strictly valid JSON with no markdown:
{"state": "Active|Stale_or_Irrelevant|Conflicting_or_Superseded|Underspecified",
"reasoning": "...",
"policy": "Use|Ignore|Update|Ask"}
<sup>State</sup>_<sup>Only</sup>_<sup>Deterministic</sup>_<sup>Routing.</sup>
User History:
{history}
Current Task: {task}
Memory state options:
{state definitions}
Policy options:
{policy definitions}
Output strictly valid JSON with no markdown:
{"state": "Active|Stale_or_Irrelevant|Conflicting_or_Superseded|Underspecified"}
```

## C Frozen-V3 State-Intervention Prompts

The label-conditioning diagnostic prompts below are frozen-V3 prompts, not historical developmentset internal prompts. They share the same policy definitions listed in the matched-ablation prompt section; conditions that ask for a state use the same state definitions.

<sup>P</sup>olicy\_<sup>O</sup>nly.   
User History:   
{history}   
Current Task: {task}   
Policy options:   
{policy definitions}   
Output JSON: {"policy": "Use|Ignore|Update|Ask"}   
<sup>S</sup>tate\_<sup>F</sup>irst.   
User History:   
{history}   
Current Task: {task}

```jsonl
State options:
{state definitions}
Policy options:
{policy definitions}
First output state, then policy. Output JSON: {"state": "...", "policy": "..."}
<sup>P</sup>olicy_<sup>F</sup>irst.
User History:
{history}
Current Task: {task}
State options:
{state definitions}
Policy options:
{policy definitions}
First output policy, then state. Output JSON: {"policy": "...", "state": "..."}
<sup>Forced</sup>_<sup>Correct</sup>_<sup>State.</sup>
User History:
{history}
Current Task: {task}
Given memory state: {benchmark-implied state}
Policy options:
{policy definitions}
Output JSON: {"policy": "Use|Ignore|Update|Ask"}
<sup>Forced</sup>_<sup>Wrong</sup>_<sup>State.</sup>
User History:
{history}
Current Task: {task}
Given memory state: {fixed non-reference state}
Policy options:
{policy definitions}
Output JSON: {"policy": "Use|Ignore|Update|Ask"}
<sup>State</sup>_<sup>Only</sup>_<sup>Deterministic</sup>_<sup>Routing.</sup>
User History:
{history}
Current Task: {task}
State options:
{state definitions}
Output JSON:
{"state": "Active|Stale_or_Irrelevant|Conflicting_or_Superseded|Underspecified"}
```

## D Artifact Name Mapping

Table 3 reports the bridge between reader-facing names and artifact identifiers.
<table><tr><td>Paper-facing name</td><td>Artifact ID</td></tr><tr><td>Policy-only matched arm</td><td>Clean Policy_Only</td></tr><tr><td>Explicit state-output matched arm</td><td>State Output No Rationale</td></tr><tr><td>State-only routing with policy definitions</td><td>State_Only_Deterministic_Routing (matched-ablation</td></tr><tr><td>State-only routing without policy</td><td>prompt) State_Only_Deterministic_Routing (label-conditioning</td></tr><tr><td>definitions Supplied reference label</td><td>prompt) Forced_Correct_State</td></tr><tr><td>Supplied conflicting label</td><td>Forced_Wrong_State</td></tr><tr><td>the frozen controlled set</td><td>Frozen V3</td></tr><tr><td>the semantic-evidence follow-up</td><td>Semantic-Evidence V1</td></tr></table>

Table 3: Reader-facing names for internal artifact identifiers. Exact machine IDs are stored in the raw-result metadata. The two state-only deterministic-routing artifact conditions share a historical ID but use diferent prompts; they are therefore named separately in prose.

## E Label-Conditioning Diagnostic and Wrong-State Construction

The Forced\_Correct\_State and Forced\_Wrong\_State artifact conditions are reported as a label-conditioning sensitivity diagnostic. Frozen V3 uses a deterministic benchmark state-topolicy mapping: Active→Use, Stale-or-Irrelevant→Ignore, Conflicting-or-Superseded→Update, and Underspecified→Ask. Supplying a state label therefore supplies a benchmark-associated answer label, not an independent latent mechanism.

The Forced\_Wrong\_State condition uses the fixed seed 20270719. For each frozen example, the runner samples one label uniformly from the three states that difer from the benchmark-implied state. The selected wrong state is then fixed across models and seeds. This construction makes wrong-state comparisons informative about sensitivity to externally supplied benchmark-associated labels, but it does not prove that naturally emitted state strings are faithful internal mechanisms.

## F Full Frozen-V3 Results

Table 4 reports the full matched prompt-ablation results, and Table 5 reports the full labelconditioning diagnostic results.

<table><tr><td>Model</td><td>Arm</td><td>Calls</td><td>Accuracy</td><td>Macro F1</td><td>Parse fail</td></tr><tr><td>Llama-3.3-70B</td><td>CLEAN POLICY ONLY</td><td>480</td><td>44.0%</td><td>36.3%</td><td>0</td></tr><tr><td>Llama-3.3-70B</td><td>TAXONOMY NO STATE OUTPUT</td><td>480</td><td>53.1%</td><td>46.5%</td><td>1</td></tr><tr><td>Llama-3.3-70B</td><td>STATE OUTPUT NO RATIONALE</td><td>480</td><td>44.6%</td><td>35.4%</td><td>0</td></tr><tr><td>Llama-3.3-70B</td><td>STATE OUTPUT WITH RATIONALE</td><td>480</td><td>47.9%</td><td>40.4%</td><td>0</td></tr><tr><td>Llama-3.3-70B</td><td>STATE ONLY DETERMINISTIC ROUTING</td><td>480</td><td>45.0%</td><td>35.9%</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>CLEAN_POLICY_ONLY</td><td>480</td><td>54.8%</td><td>47.1%</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>TAXONOMY NO STATE OUTPUT</td><td>480</td><td>59.8%</td><td>55.2%</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>STATE _OUTPUT NO_RATIONALE</td><td>480</td><td>58.1%</td><td>53.5%</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>STATE OUTPUT WITH RATIONALE</td><td>480</td><td>57.9%</td><td>52.9%</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>STATE_ONLY_DETERMINISTIC_ROUTING</td><td>480</td><td>56.9%</td><td>50.8%</td><td>0</td></tr></table>

Table 4: Full frozen-V3 matched prompt-ablation results.

<table><tr><td>Model</td><td>Condition</td><td>Calls</td><td>Accuracy</td><td>Macro F1</td><td>State agreement</td><td>Parse fail</td></tr><tr><td>Llama-3.3-70B</td><td>POLICY ONLY</td><td>480</td><td>45.4%</td><td>36.3%</td><td>n/a</td><td>0</td></tr><tr><td>Llama-3.3-70B</td><td>STATE FIRST</td><td>480</td><td>42.7%</td><td>32.6%</td><td>41.5%</td><td>0</td></tr><tr><td>Llama-3.3-70B</td><td>POLICY FIRST</td><td>480</td><td>48.8%</td><td>40.1%</td><td>44.6%</td><td>0</td></tr><tr><td>Llama-3.3-70B</td><td>FORCED CORRECT STATE</td><td>480</td><td>52.7%</td><td>49.2%</td><td>n/a</td><td>0</td></tr><tr><td>Llama-3.3-70B</td><td>FORCED WRONG STATE</td><td>480</td><td>37.5%</td><td>30.4%</td><td>n/a</td><td>0</td></tr><tr><td>Llama-3.3-70B</td><td>STATE ONLY DETERMINISTIC ROUTING</td><td>480</td><td>46.9%</td><td>36.1%</td><td>46.9%</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>POLICY ONLY</td><td>480</td><td>53.1%</td><td>44.6%</td><td>n/a</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>STATE FIRST</td><td>480</td><td>59.2%</td><td>54.4%</td><td>55.8%</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>POLICY FIRST</td><td>480</td><td>57.1%</td><td>51.1%</td><td>53.8%</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>FORCED CORRECT STATE</td><td>480</td><td>65.0%</td><td>63.2%</td><td>n/a</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>FORCED WRONG STATE</td><td>480</td><td>50.0%</td><td>46.1%</td><td>n/a</td><td>0</td></tr><tr><td>GPT-OSS-120B</td><td>STATE ONLY DETERMINISTIC ROUTING</td><td>480</td><td>52.5%</td><td>46.5%</td><td>52.5%</td><td>0</td></tr></table>

Table 5: Full frozen-V3 label-conditioning diagnostic results. State agreement is shown only for conditions that ask the model to emit a state. Forced-state conditions supply benchmark-associated labels and are not interpreted as internal state mechanisms.

## F.1 Per-Policy Decomposition of the Matched Ablation

Table 6 decomposes the primary matched ablation by reference policy.
<table><tr><td>Model</td><td>Policy</td><td>∆</td><td>95% CI</td><td>Holm p</td></tr><tr><td>Llama-3.3-70B</td><td>USE</td><td>+17.5 pp*</td><td>[6.67, 28.33]</td><td>0.010</td></tr><tr><td>Llama-3.3-70B</td><td>IGNORE</td><td>n.s.</td><td>n.s.</td><td>n.s.</td></tr><tr><td>Llama-3.3-70B</td><td>UPDATE</td><td>n.s.</td><td>n.s.</td><td>n.s.</td></tr><tr><td>Llama-3.3-70B</td><td>Ask</td><td>-14.2 pp*</td><td>[-22.50, -6.67]</td><td>0.010</td></tr><tr><td>GPT-OSS-120B</td><td>USE</td><td>n.s.</td><td>n.s.</td><td>n.s.</td></tr><tr><td>GPT-OSS-120B</td><td>IGNORE</td><td>+15.8 pp*</td><td>[7.50, 25.00]</td><td>0.003</td></tr><tr><td>GPT-OSS-120B</td><td>UPDATE</td><td>-8.3 pp*</td><td>[-13.33, -3.33]</td><td>0.010</td></tr><tr><td>GPT-OSS-120B</td><td>Ask</td><td>n.s.</td><td>n.s.</td><td>n.s.</td></tr></table>

Table 6: Per-policy decomposition of the primary matched ablation. Deltas are State\_Output\_No\_Rationale minus Clean\_Policy\_Only; CIs are scenario-familycluster bootstrap intervals; p is Holm-adjusted across the four policies within each model; parse failures count as incorrect. Only Holm-significant cells are numerically reported here; full per-policy metrics and confusion matrices appear below. Gains and losses are reported together and the overall efect is null. <sup>∗</sup> marks Holm-significant cells.

## F.2 Third-Endpoint Replication (llama-3.1-8b-instant)

Table 7 reports an additive third-endpoint replication.

<table><tr><td>Model</td><td>Arm</td><td>Accuracy</td><td>Macro F1</td><td>Parse fail</td></tr><tr><td>llama-3.1-8b-instant</td><td>Clean policy-only</td><td>25.6%</td><td>11.3%</td><td>0/480</td></tr><tr><td>llama-3.1-8b-instant</td><td>State-output no rationale</td><td>37.3%</td><td>27.4%</td><td>18/480</td></tr><tr><td>llama-3.1-8b-instant</td><td>Delta (state-output minus policy-only) +11.7 pp</td><td></td><td>95% CI [8.54, 14.79]</td><td>p=0.0001</td></tr></table>

Table 7: Additive third-endpoint replication. The candidate qwen/qwen3.6-27b was excluded after an HTTP 400 json\_validate\_failed smoke failure, mirroring the GPT-OSS explicit-evidence provider failure documented in the main text. llama-3.1-8b-instant is a small model included only to test cross-model direction, not as a headline result. The state-output arm had 18/480 parse failures, counted as incorrect, so this delta must not be read as a clean gain.

## G Family-Level Counterfactual Analysis

Frozen V3 is arranged as 40 matched four-way families. Family exact match requires all four variants in a family to be correct for a given model, arm, and seed. The analysis also constructs the six withinfamily policy pairs (Use/Ignore, Use/Update, Use/Ask, Ignore/Update, Ignore/Ask, and Update/Ask) and classifies each pair as expected-transition correct, partial transition, missed transition, wrong-direction transition, or provider/parse failure. The computed pairwise-transition records are retained in local run outputs. Table 8 reports family exact-match rates for the matched prompt ablation. Table 9 reports family exact-match rates for the label-conditioning diagnostic.

<table><tr><td>Model</td><td>Condition</td><td>Family exact</td></tr><tr><td>Llama</td><td>Matched policy-only</td><td>0.0% [0.0, 2.5]</td></tr><tr><td>Llama</td><td>State output</td><td>0.0% [0.0, 2.5]</td></tr><tr><td>GPT-OSS</td><td>Matched policy-only</td><td>1.7% [0.0, 5.0]</td></tr><tr><td>GPT-OSS</td><td>State output</td><td>5.0% [0.8, 10.0]</td></tr></table>

Table 8: Family exact-match rates on the frozen controlled set. A family is correct only when all four matched variants are predicted correctly. Intervals resample scenario families, with seeds averaged inside each family. For 0.0% entries, the 3/n upper bound uses n=120 family-seed units, an optimistic independence assumption.

<table><tr><td>Model</td><td>Condition</td><td>Family exact</td></tr><tr><td>Llama</td><td>Diagnostic policy-only</td><td>0.0% [0.0, 2.5]</td></tr><tr><td>Llama</td><td>Supplied reference label</td><td>1.7% [0.0, 4.2]</td></tr><tr><td>Llama</td><td>Supplied conflicting label</td><td>0.0% [0.0, 2.5]</td></tr><tr><td>GPT-OSS</td><td>Diagnostic policy-only</td><td>1.7% [0.0, 4.2]</td></tr><tr><td>GPT-OSS</td><td>Supplied reference label</td><td>11.7% [5.0, 19.2]</td></tr><tr><td>GPT-OSS</td><td>Supplied conflicting label</td><td>0.8% [0.0, 2.5]</td></tr></table>

Table 9: Family exact-match rates for the separately executed label-conditioning diagnostic. The policy-only row comes from a diferent frozen protocol source and is labeled by protocol to avoid conflation. Intervals resample scenario families, with seeds averaged inside each family. For 0.0% entries, the 3/n upper bound uses n=120 family-seed units, an optimistic independence assumption.

Pairwise transition analysis gives a second view. Across the six within-family policy pairs, expected-transition accuracy is 13.8% for Llama policy-only and 13.6% for Llama state-output; missed-transition rates are 48.6% and 47.1%. GPT-OSS improves from 23.5% to 29.2% expected transitions with state output, but still misses 32.8% and 30.3% of transitions. Supplied reference labels improve expected transitions relative to policy-only, again consistent with answer-associated label sensitivity rather than an internally faithful state mechanism.

## H Seed-Stability Diagnostics

Each hosted arm uses seeds 101, 102, and 103. We report whether the three policies for an example are unanimous, a 2–1 majority, or all diferent, and compute a simple entropy over the three predictions. Because only three seeds are available, this is a stochastic-instability diagnostic rather than a calibrated uncertainty estimate. The corresponding per-example outputs are retained in local run outputs.

Among unanimous examples, per-call accuracy is 50.5% and 48.4% for Llama policy-only and state-output, and 61.7% and 63.9% for GPT-OSS. Non-unanimous examples often have lower percall accuracy, but the pattern is descriptive and not calibrated uncertainty.

## I Normalized-Input Probe Protocol

The normalized-input probe uses the same final controlled set as the matched experiments; the controlled set records 160 examples and 40 families. The normalized-input policy-only arm is compared against clean policy-only. It uses seeds 101, 102, and 103, temperature 0.7, and the two endpoints llama-3.3-70b-versatile and openai/gpt-oss-120b. The normalized-input arm records 480 calls and 0 parse failures for each normalized-input model-arm, 960 calls total.

The tuple block is derived deterministically from the stored semantic records: for each record, the renderer writes subject, attribute, value, and either the stored time tag, stored context tag, or none. The arm keeps the clean policy-only task and replaces the free-text history with deterministic structured tuple lines in the [subject|attribute|value|tag] format. The rendered prompt below was produced by the existing runner for example controlled\_v3\_02\_use.

```yaml
User History:
[flight seat choice | flight seat choice option | an aisle seat | earlier]
[flight seat choice | flight seat choice option | an aisle seat | later]
[train seat | train seat option | a window seat | case A]
[train seat | train seat option | I sit near an exit | case B]
Current Task: Choose the preference option for Jordan's flight seat choice.
Policy options:
- Use: apply the relevant memory.
- Ignore: complete the task without the memory.
- Update: acknowledge that newer information supersedes the old memory and use the newer
information.
- Ask: request clarification before acting.
Output strictly valid JSON with no markdown:
{"policy": "Use|Ignore|Update|Ask"}
```

The leakage guard trains grouped word and character TF-IDF decoders on the normalized tuple block plus task text, grouping by scenario family. The leakage probe reports word-grouped accuracy 0.375, character-grouped accuracy 0.33125, a near-trivial decode threshold of 0.9, and was not flagged as answer leaking.

<table><tr><td>Model</td><td>Accuracy</td><td>Macro F1</td><td>Parse failures</td><td>Delta vs clean</td><td>95% CI</td><td>p</td></tr><tr><td>1lama-3.3-70b-versatile</td><td>35.42%</td><td>30.36%</td><td>0</td><td>-8.54 pp</td><td>[-13.96, -3.54]</td><td>0.003400</td></tr><tr><td>openai/gpt-oss-120b</td><td>46.88%</td><td>44.08%</td><td>0</td><td>-7.92 pp</td><td>[-12.08, -3.54]</td><td>0.001200</td></tr></table>

Table 10: Normalized-input probe results. Deltas are normalized-input minus clean policy-only.

The reported normalized-input values are paired p-values; no Holm-adjusted values for this probe are reported here. Across the paper, Holm-adjusted values are reported where multiple contrasts are grouped as a family of tests, and otherwise the reported p-values are treated as descriptive exploratory evidence.

## J Rule-Based Error Taxonomy

The error taxonomy is automatically derived from structured metadata and model predictions. It is not a human-coded annotation. Categories include recency overreach, conflict over-detection, irrelevance-to-Ask confusion, ambiguity under-detection, defaulting to Use or Ask, state-policy inconsistency, intermediate serialization burden, task neglect, and provider/protocol failure. Categories are non-exclusive, may overlap, and are counted over final scored model-arm-example-seed records rather than raw retry attempts. The local error-analysis outputs retain full examples, denominators, and rates. Table 11 reports the rule-based error-mode counts.

<table><tr><td>Error mode</td><td>Count</td><td>Eligible</td><td>Detail</td></tr><tr><td>conflict over detection</td><td>2428</td><td>5121</td><td></td></tr><tr><td>task neglect</td><td>1583</td><td>5121</td><td></td></tr><tr><td>recency overreach</td><td>1543</td><td>5121</td><td></td></tr><tr><td>ask defaulting</td><td>873</td><td>5121</td><td></td></tr><tr><td>irrelevance to ask confusion</td><td>760</td><td>5121</td><td></td></tr><tr><td>state policy inconsistency</td><td>276</td><td>3840</td><td>138 final policy correct</td></tr><tr><td>use defaulting</td><td>266</td><td>5121</td><td></td></tr><tr><td>ambiguity under detection</td><td>123</td><td>5121</td><td></td></tr></table>

Table 11: Rule-based error-mode counts across final scored controlled-set model-arm-example-seed records. Eligible denotes the count of final scored records for which the category is applicable (5121 for arms scored on all records; 3840 for state-emitting arms only). Counts are non-exclusive and must not be summed.

## K Per-Policy Metrics and Confusion Matrices

The machine-readable frozen-V3 reports include per-policy precision, recall, F1, and confusion matrices for each model and arm. Parse failures remain explicit records and count as incorrect. The main paper reports only the matched headline contrasts to preserve readability. Table 12 reports per-policy precision, recall, and F1 for the two primary matched-ablation arms.

<table><tr><td>Model</td><td>Arm</td><td>Policy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Llama-3.3-70B</td><td>Clean policy-only</td><td>USE</td><td>85.1</td><td>61.7</td><td>71.5</td></tr><tr><td>Llama-3.3-70B</td><td>Clean policy-only</td><td>IGNORE</td><td>100.0</td><td>0.8</td><td>1.7</td></tr><tr><td>Llama-3.3-70B</td><td>Clean policy-only</td><td>UPDATE</td><td>34.6</td><td>98.3</td><td>51.2</td></tr><tr><td>Llama-3.3-70B</td><td>Clean policy-only</td><td>AsK</td><td>35.3</td><td>15.0</td><td>21.1</td></tr><tr><td>Llama-3.3-70B</td><td>State-output</td><td>USE</td><td>84.8</td><td>79.2</td><td>81.9</td></tr><tr><td>Llama-3.3-70B</td><td>State-output</td><td>IGNORE</td><td>83.3</td><td>4.2</td><td>7.9</td></tr><tr><td>Llama-3.3-70B</td><td>State-output</td><td>UPDATE</td><td>34.6</td><td>94.2</td><td>50.6</td></tr><tr><td>Llama-3.3-70B</td><td>State-output</td><td>AsK</td><td>2.9</td><td>0.8</td><td>1.3</td></tr><tr><td>GPT-OSS-120B</td><td>Clean policy-only</td><td>USE</td><td>98.3</td><td>99.2</td><td>98.8</td></tr><tr><td>GPT-OSS-120B</td><td>Clean policy-only</td><td>IGNORE</td><td>100.0</td><td>9.2</td><td>16.8</td></tr><tr><td>GPT-OSS-120B</td><td>Clean policy-only</td><td>UPDATE</td><td>40.7</td><td>100.0</td><td>57.8</td></tr><tr><td>GPT-OSS-120B</td><td>Clean policy-only</td><td>AsK</td><td>24.5</td><td>10.8</td><td>15.0</td></tr><tr><td>GPT-OSS-120B</td><td>State-output</td><td>USE</td><td>90.2</td><td>100.0</td><td>94.9</td></tr><tr><td>GPT-OSS-120B</td><td>State-output</td><td>IGNORE</td><td>100.0</td><td>25.0</td><td>40.0</td></tr><tr><td>GPT-OSS-120B</td><td>State-output</td><td>UPDATE</td><td>40.1</td><td>91.7</td><td>55.8</td></tr><tr><td>GPT-OSS-120B</td><td>State-output</td><td>AsK</td><td>44.2</td><td>15.8</td><td>23.3</td></tr></table>

Table 12: Per-policy precision, recall, and F1 for the two primary matched-ablation arms. Values are percentages. Parse failures count as incorrect.

## L Statistical Procedures

The unit of clustering is the scenario family. Reported confidence intervals use a 10,000-resample percentile bootstrap over scenario-family clusters. Paired tests use family-level sign-flip permutation tests with plus-one correction. Non-significant p-values are not interpreted as equivalence. The primary matched-ablation contrast is state-output minus clean policy-only. In the label-conditioning diagnostic, forced-correct-state, forced-wrong-state, and state-first conditions are compared against policy-only, and wrong-state routing is also compared against forced-correct routing.

## M Shortcut-Transfer Diagnostic

A supervised TF-IDF classifier trained only on the original 480-example synthetic development set reaches 25.0% accuracy and 13.9% macro F1 when evaluated once on frozen V3. The classifier vocabulary and parameters are fit only on the development set. Because the development set has no positive Ignore examples, this transfer result is afected by label-space shift, class-prior shift, and template shift. It is a descriptive transfer diagnostic, not the primary evidence that frozen V3 eliminates lexical shortcuts.

## N Semantic-Evidence V1 Follow-Up

The semantic-evidence v1 protocol was created after identifying the deterministic mapping between frozen-V3 benchmark states and policies. It uses 160 examples arranged as 40 matched families and three prompt arms: policy-only, explicit semantic-evidence output, and evidence-taxonomy exposure without evidence output. The explicit-evidence arm asks for task relevance, temporal status, supersession, context suficiency, risk or confirmation, and policy. It does not supply the old four-state benchmark label to the model.

The local design audit passes structural checks. Single evidence-field majority decode accuracies range from 25.0% to 50.0%; the full evidence tuple decodes policy at 75.0%, template signatures at 75.0%, keyword counts at 51.2%, grouped word TF-IDF at 55.6% accuracy and 55.6% macro F1, and grouped character TF-IDF at 62.5% accuracy and 61.7% macro F1. These diagnostics are reported as design constraints, not as model-performance results.

All 2,880 hosted calls in the frozen manifest were attempted. Llama completed all calls without provider failures. Its policy-only, taxonomy-only, and explicit semantic-evidence accuracies are 63.3%, 54.6%, and 55.8%, respectively. Explicit semantic evidence is 7.5 points below policy-only (95% CI [-12.7, -2.5], Holm-adjusted p=0.0146), 1.3 points above taxonomy-only (p=0.6079), and taxonomy-only is 8.8 points below policy-only (95% CI [-12.1, -5.6], Holm-adjusted p=0.0003). GPT-OSS produced provider-side JSON validation failures for all explicit-evidence calls, so that arm is not a clean behavioral comparison for GPT-OSS. The failed records are retained in execution accounting; no prompt, schema, or parsing rule was changed after observing them, and the GPT-OSS explicit-evidence arm is not scored as a clean model-behavior result.

## O Proposal-Only Future Protocols

Two optional future protocols were drafted but not run: a downstream-consequence proposal and an independent-renderer proposal. They contain no reported result numbers, require separate explicit API authorization before any hosted execution, and are not used to support current manuscript claims.

## P Model-Protocol Amendment

The development-set evidence used Llama-3.3-70B and Qwen3-32B. Before any frozen-V3 hosted result existed, the planned Qwen3-32B endpoint became unavailable on the current Groq tier. The frozen-V3 protocol was therefore amended to use Llama-3.3-70B and GPT-OSS-120B. Frozen data, prompts, arms, seeds, scoring, and analysis rules were not changed, and GPT-OSS results are never labeled as Qwen results.

## Q Historical Development-Set Results

The original 480-example development set remains useful as diagnostic evidence but not as the decisive test. It has positive reference-policy examples for Use, Update, and Ask, and no standalone positive Ignore reference class. It is also lexically separable under supervised TF-IDF diagnostics. Historical prompt-bundle results and development-set confusion matrices are retained below for auditability. Full-context uses the full user history if relevant and then asks the model to decide the best policy. Ask-if-uncertain instructs the model to choose Ask when uncertain whether the user history applies; otherwise it asks the model to decide the best policy.

Table 13 reports the state confusion matrix for the state-structured bundle.

Table 13: State-structured bundle state confusion matrix pooled over three seeds. Rows are gold benchmark categories; columns are predicted memory states. Overall benchmarkmapped state agreement under the mapping Stable/Contextual→Active, Updated→Conflicting, and Ambiguous→Underspecified is 79.4% for Llama-3.3-70B and 73.1% for Qwen3-32B.
<table><tr><td>Model</td><td>Generation category</td><td>Active</td><td>Stale</td><td>Conflicting</td><td>Underspecified</td><td>Parse fail</td><td>n</td></tr><tr><td>Llama-3.3-70B</td><td>Stable</td><td>348</td><td>0</td><td>3</td><td>9</td><td>0</td><td>360</td></tr><tr><td>Llama-3.3-70B</td><td>Contextual</td><td>343</td><td>0</td><td>5</td><td>12</td><td>0</td><td>360</td></tr><tr><td>Llama-3.3-70B</td><td>Updated</td><td>33</td><td>4</td><td>323</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Llama-3.3-70B</td><td>Ambiguous</td><td>108</td><td>20</td><td>102</td><td>130</td><td>0</td><td>360</td></tr><tr><td>Qwen3-32B</td><td>Stable</td><td>354</td><td>0</td><td>1</td><td>3</td><td>2</td><td>360</td></tr><tr><td>Qwen3-32B</td><td>Contextual</td><td>358</td><td>0</td><td>0</td><td>2</td><td>0</td><td>360</td></tr><tr><td>Qwen3-32B</td><td>Updated</td><td>49</td><td>30</td><td>275</td><td>5</td><td>1</td><td>360</td></tr><tr><td>Qwen3-32B</td><td>Ambiguous</td><td>221</td><td>13</td><td>59</td><td>65</td><td>2</td><td>360</td></tr></table>

Table 14 reports the historical three-arm development-set ablation.

Table 14: Three-arm ablation comparing the historical implicit-state/no-state-output prompt, the cleaned implicit-state/no-state-output prompt, and the state-structured prompt bundle. p-values are pooled exact McNemar diagnostics over 1440 matched example-seed decisions, with parse failures counted as wrong; cluster-aware comparisons are reported separately.
<table><tr><td>Model</td><td>Arm</td><td>Overall</td><td>Updated</td><td>Parse fail</td><td>p vs Historical</td><td>p vs Cleaned</td></tr><tr><td>Llama-3.3-70B</td><td>HISTORICAL NO-STATE-OUTPUT</td><td>70.3±0.5%</td><td>46.7±3.8%</td><td>0.0±0.0%</td><td></td><td></td></tr><tr><td>Llama-3.3-70B</td><td>CLEANED NO-STATE-OUTPUT</td><td>71.7±0.6%</td><td>48.6±3.2%</td><td>0.0±0.0%</td><td></td><td></td></tr><tr><td>Llama-3.3-70B</td><td>STATE-STRUCTURED BUNDLE</td><td>85.1±0.2%</td><td>89.7±2.7%</td><td>0.0±0.0%</td><td>3.6e-37</td><td>2.4e-32</td></tr><tr><td>Qwen3-32B</td><td>HISTORICAL NO-STATE-OUTPUT</td><td>65.3±1.8%</td><td>30.3±5.4%</td><td>0.2±0.2%</td><td></td><td></td></tr><tr><td>Qwen3-32B</td><td>CLEANED NO-STATE-OUTPUT</td><td>64.3±0.8%</td><td>33.6±2.7%</td><td>0.0±0.0%</td><td></td><td></td></tr><tr><td>Qwen3-32B</td><td>STATE-STRUCTURED BUNDLE</td><td>75.5±1.2%</td><td>75.3±3.8%</td><td>0.3±0.3%</td><td>1.5e-18</td><td>2.9e-26</td></tr></table>

Table 15 reports the historical oracle-worded routing diagnostic.

Table 15: State-structured bundle versus Historical oracle-worded routing. This historical language-model routing diagnostic uses oracle-worded prompt text and should not be interpreted as a neutral mapped-state intervention or a deterministic oracle.
<table><tr><td>Model</td><td>Arm</td><td>Overall</td><td>Stable</td><td>Contextual</td><td>Updated</td><td>Ambiguous</td><td>Parse fail</td></tr><tr><td>Llama-3.3-70B</td><td>STATE-STRUCTURED BUNDLE</td><td>85.1±0.2%</td><td>96.4±0.5%</td><td>95.8±0.8%</td><td>89.7±2.7%</td><td> $5 8 . 3 { \pm } 2 . 2 \%$ </td><td>0.0±0.0%</td></tr><tr><td>Llama-3.3-70B</td><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>86.9±2.2%</td><td>99.2±0.0%</td><td>97.5±0.0%</td><td>72.5±3.0%</td><td>78.3±6.0%</td><td>0.0%</td></tr><tr><td>Qwen3-32B</td><td>STATE-STRUCTURED BUNDLE</td><td>75.5±1.2%</td><td>98.6±1.3%</td><td>99.2±0.8%</td><td>75.3±3.8%</td><td>28.9±1.0%</td><td>0.3±0.3%</td></tr><tr><td>Qwen3-32B</td><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>73.5±2.1%</td><td>99.7±0.5%</td><td>100.0±0.0%</td><td>50.0±3.0%</td><td>44.2±6.6%</td><td>0.0%</td></tr></table>

Table 16 reports token cost and strict accuracy for direct policy prediction versus explicit state classification.

Table 16: Token cost and strict accuracy for direct policy prediction versus explicit state classification. Token counts are aggregated over 1440 calls.
<table><tr><td>Model</td><td>Arm</td><td>Tokens Tokens/call Accuracy</td></tr><tr><td>Llama-3.3-70B</td><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>219,903</td><td>152.7 69.8±1.6%</td></tr><tr><td>Llama-3.3-70B</td><td>STATE-STRUCTURED BUNDLE</td><td>475,302</td><td>330.185.1±0.2%</td></tr><tr><td>Qwen3-32B</td><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>563,470</td><td>391.3 66.7±1.7%</td></tr><tr><td>Qwen3-32B</td><td>STATE-STRUCTURED BUNDLE</td><td>811,965</td><td>563.9 75.5±1.2%</td></tr></table>

The 480-example development set remains useful for diagnosing why the stronger controlled evaluation was needed. On that set, the old state-structured prompt bundle improved exact-match policy accuracy over the historical implicit-state/no-state-output prompt by 14.8 points for Llama and 10.2 points for Qwen. However, the same dataset is lexically separable, lacks positive Ignore coverage, and used a prompt bundle that changed taxonomy, step wording, state output, rationale output, and length simultaneously. These facts demote the development result from a method claim to a motivation for controlled counterfactual testing.

Why the controlled result difers from the development result. The development-set state-structured bundle arm was a prompt bundle, not an isolated state-field test. It changed policy wording, taxonomy exposure, output order, rationale generation, and token budget. It was also evaluated on a lexically separable set without positive Ignore examples. The frozen controlled set balances all four policies and is evaluated with grouped diagnostics, so it is the stronger evidence for the state-field question; however, the development and frozen-set shortcut numbers are not a direct same-protocol magnitude comparison.

## R Policy Confusion Matrices

Table 17 and Table 18 report the development-set policy confusion matrices by endpoint. Rows are generation categories and columns are predicted policies. All matrices use the full 480-example Groq protocol with seeds 101/102/103. The Historical oracle-worded routing rows use the historical oracle-worded routing prompt.

Table 17: Policy confusion matrix for Llama-3.3-70B. Rows are gold benchmark categories; columns are predicted policies plus parse failures.
<table><tr><td>Agent</td><td>Generation category</td><td>Use</td><td>Ask</td><td>Ignore</td><td>Update</td><td>Parse fail</td><td>n</td></tr><tr><td>HISTORICAL NO-STATE-OUTPUT</td><td>Stable</td><td>355</td><td>2</td><td>1</td><td>2</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL NO-STATE-OUTPUT</td><td>Contextual</td><td>356</td><td>2</td><td>2</td><td>0</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL NO-STATE-OUTPUT</td><td>Updated</td><td>157</td><td>7</td><td>28</td><td>168</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL NO-STATE-OUTPUT</td><td>Ambiguous</td><td>165</td><td>133</td><td>62</td><td>0</td><td>0</td><td>360</td></tr><tr><td>CLEANED NO-STATE-OUTPUT</td><td>Stable</td><td>354</td><td>3</td><td>3</td><td>0</td><td>0</td><td>360</td></tr><tr><td>CLEANED NO-STATE-OUTPUT</td><td>Contextual</td><td>357</td><td>0</td><td>3</td><td>0</td><td>0</td><td>360</td></tr><tr><td>CLEANED NO-STATE-OUTPUT</td><td>Updated</td><td>165</td><td>6</td><td>14</td><td>175</td><td>0</td><td>360</td></tr><tr><td>CLEANED NO-STATE-OUTPUT</td><td>Ambiguous</td><td>146</td><td>146</td><td>67</td><td>1</td><td>0</td><td>360</td></tr><tr><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>Stable</td><td>336</td><td>12</td><td>0</td><td>12</td><td>0</td><td>360</td></tr><tr><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>Contextual</td><td>355</td><td>1</td><td>3</td><td>1</td><td>0</td><td>360</td></tr><tr><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>Updated</td><td>124</td><td>19</td><td>3</td><td>214</td><td>0</td><td>360</td></tr><tr><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>Ambiguous</td><td>157</td><td>100</td><td>42</td><td>61</td><td>0</td><td>360</td></tr><tr><td>STATE-STRUCTURED BUNDLE</td><td>Stable</td><td>347</td><td>11</td><td>0</td><td>2</td><td>0</td><td>360</td></tr><tr><td>STATE-STRUCTURED BUNDLE</td><td>Contextual</td><td>345</td><td>11</td><td>2</td><td>2</td><td>0</td><td>360</td></tr><tr><td>STATE-STRUCTURED BUNDLE</td><td>Updated</td><td>34</td><td>0</td><td>3</td><td>323</td><td>0</td><td>360</td></tr><tr><td>STATE-STRUCTURED BUNDLE</td><td>Ambiguous</td><td>109</td><td>210</td><td>34</td><td>7</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>Stable</td><td>357</td><td>3</td><td>0</td><td>0</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>Contextual</td><td>351</td><td>0</td><td>9</td><td>0</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>Updated</td><td>34</td><td>56</td><td>9</td><td>261</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>Ambiguous</td><td>59</td><td>282</td><td>19</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Ask-if-uncertain</td><td>Stable</td><td>350</td><td>10</td><td>0</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Ask-if-uncertain</td><td>Contextual</td><td>354</td><td>3</td><td>3</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Ask-if-uncertain</td><td>Updated</td><td>243</td><td>11</td><td>3</td><td>103</td><td>0</td><td>360</td></tr><tr><td>Ask-if-uncertain</td><td>Ambiguous</td><td>128</td><td>178</td><td>54</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Full-context</td><td>Stable</td><td>358</td><td>2</td><td>0</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Full-context</td><td>Contextual</td><td>354</td><td>0</td><td>6</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Full-context</td><td>Updated</td><td>180</td><td>0</td><td>0</td><td>180</td><td>0</td><td>360</td></tr><tr><td>Full-context</td><td>Ambiguous</td><td>242</td><td>43</td><td>66</td><td>9</td><td>0</td><td>360</td></tr></table>

Table 18: Policy confusion matrix for Qwen3-32B. Rows are gold benchmark categories; columns are predicted policies plus parse failures.
<table><tr><td>Agent</td><td>Generation category</td><td>Use</td><td>Ask</td><td>Ignore</td><td>Update</td><td>Parse fail</td><td>n</td></tr><tr><td>HISTORICAL NO-STATE-OUTPUT</td><td>Stable</td><td>353</td><td>5</td><td>1</td><td>0</td><td>1</td><td>360</td></tr><tr><td>HISTORICAL NO-STATE-OUTPUT</td><td>Contextual</td><td>359</td><td>1</td><td>0</td><td>0</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL NO-STATE-OUTPUT</td><td>Updated</td><td>186</td><td>16</td><td>49</td><td>109</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL NO-STATE-OUTPUT</td><td>Ambiguous</td><td>211</td><td>119</td><td>28</td><td>0</td><td>2</td><td>360</td></tr><tr><td>CLEANED NO-STATE-OUTPUT</td><td>Stable</td><td>357</td><td>3</td><td>0</td><td>0</td><td>0</td><td>360</td></tr><tr><td>CLEANED NO-STATE-OUTPUT</td><td>Contextual</td><td>357</td><td>3</td><td>0</td><td>0</td><td>0</td><td>360</td></tr><tr><td>CLEANED NO-STATE-OUTPUT</td><td>Updated</td><td>178</td><td>9</td><td>52</td><td>121</td><td>0</td><td>360</td></tr><tr><td>CLEANED NO-STATE-OUTPUT</td><td>Ambiguous</td><td>231</td><td>91</td><td>38</td><td>0</td><td>0</td><td>360</td></tr><tr><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>Stable</td><td>319</td><td>30</td><td>1</td><td>10</td><td>0</td><td>360</td></tr><tr><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>Contextual</td><td>350</td><td>3</td><td>3</td><td>4</td><td>0</td><td>360</td></tr><tr><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>Updated</td><td>199</td><td>35</td><td>8</td><td>118</td><td>0</td><td>360</td></tr><tr><td>DIRECT POLICY (NO CLASSIFICATION)</td><td>Ambiguous</td><td>153</td><td>174</td><td>9</td><td>24</td><td>0</td><td>360</td></tr><tr><td>STATE-STRUCTURED BUNDLE</td><td>Stable</td><td>355</td><td>1</td><td>1</td><td>1</td><td>2</td><td>360</td></tr><tr><td>STATE-STRUCTURED BUNDLE</td><td>Contextual</td><td>357</td><td>2</td><td>1</td><td>0</td><td>0</td><td>360</td></tr><tr><td>STATE-STRUCTURED BUNDLE</td><td>Updated</td><td>41</td><td>3</td><td>44</td><td>271</td><td>1</td><td>360</td></tr><tr><td>STATE-STRUCTURED BUNDLE</td><td>Ambiguous</td><td>206</td><td>104</td><td>37</td><td>11</td><td>2</td><td>360</td></tr><tr><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>Stable</td><td>359</td><td>0</td><td>1</td><td>0</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>Contextual</td><td>360</td><td>0</td><td>0</td><td>0</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>Updated</td><td>64</td><td>65</td><td>51</td><td>180</td><td>0</td><td>360</td></tr><tr><td>HISTORICAL ORACLE-WORDED ROUTING</td><td>Ambiguous</td><td>71</td><td>159</td><td>127</td><td>3</td><td>0</td><td>360</td></tr><tr><td>Ask-if-uncertain</td><td>Stable</td><td>353</td><td>7</td><td>0</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Ask-if-uncertain</td><td>Contextual</td><td>357</td><td>2</td><td>1</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Ask-if-uncertain</td><td>Updated</td><td>283</td><td>22</td><td>10</td><td>45</td><td>0</td><td>360</td></tr><tr><td>Ask-if-uncertain</td><td>Ambiguous</td><td>206</td><td>133</td><td>21</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Full-context</td><td>Stable</td><td>360</td><td>0</td><td>0</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Full-context</td><td>Contextual</td><td>359</td><td>0</td><td>1</td><td>0</td><td>0</td><td>360</td></tr><tr><td>Full-context</td><td>Updated</td><td>306</td><td>6</td><td>11</td><td>37</td><td>0</td><td>360</td></tr><tr><td>Full-context</td><td>Ambiguous</td><td>275</td><td>56</td><td>29</td><td>0</td><td>0</td><td>360</td></tr></table>

## S Execution Completeness, Failures, Tokens, and Cost

The frozen-V3 hosted execution completed 4,800 matched-ablation calls and 5,760 label-conditioning diagnostic calls. One final matched-ablation parse failure remained visible and was counted as incorrect. During an earlier matched-ablation resume attempt, 53 transient provider-failure records were retained in the raw logs; the same call IDs were later completed under the frozen protocol and were not counted as separate valid predictions. Total estimated hosted cost including two smoke calls was \$1.8007 under the authorized \$20.00 hard budget.

Table 19 separates hosted call accounting by protocol.

<table><tr><td>Protocol</td><td>Calls</td><td>Hosted cost</td></tr><tr><td>Matched prompt ablation</td><td>4,800</td><td>$0.8887</td></tr><tr><td>Label-conditioning diagnostic</td><td>5,760</td><td>$0.9118</td></tr><tr><td>Normalized-input probe</td><td>960</td><td>$0.058766; $0.066606</td></tr><tr><td>Semantic-evidence follow-up</td><td>2,880</td><td>$0.3977</td></tr><tr><td>Smoke tests</td><td>2</td><td>$0.0002</td></tr><tr><td>Frozen-core total including smoke</td><td>10,562</td><td>$1.8007</td></tr></table>

Table 19: Hosted call and cost ledger by protocol. The normalized-input row lists stored per-model costs rather than a stored protocol-total cost.

## T Main-Text Design Materials

Table 20 reports the representative four-way family.

<table><tr><td>Policy</td><td>History records</td><td>Rule-derived relation</td></tr><tr><td>USE</td><td>Jordan saved an earlier entry: flight seat choice uses an aisle One current task-relevant seat for its flight seat choice option. The later note records preference is unique; tem- an aisle seat for the flight seat choice option of flight seat poral and contextual dis- choice. Case A: Jordan noted that the train seat option for tractors do not change it. train seat is a window seat. Case B: the profile lists I sit near an exit as the train seat option for train seat.</td><td></td></tr><tr><td>IGNORE</td><td>The earlier note records a window seat for the train seat The four records concern option of train seat. Later, Jordan noted that the train seat distractor attributes; no option for train seat is I sit near an exit. Case A: the profile task-relevant memory is lists I sit near an exit as the train seat option for train seat. needed. Jordan saved the case b entry: train seat uses a window seat</td><td></td></tr><tr><td>UPDATE</td><td>for its train seat option. Earlier, Jordan noted that the flight seat choice option for The later temporal task- flight seat choice is a window seat. Later, the profile lists a relevant record supersedes bulkhead aisle seat as the flight seat choice option for flight the earlier task-relevant seat choice. Jordan saved the case a entry: train seat uses a record. window seat for its train seat option. The case b note records</td><td></td></tr><tr><td>AsK</td><td>I sit near an exit for the train seat option of train seat. Earlier, the profile lists a window seat as the train seat option Two context-specific task- for train seat. Jordan saved a later entry: train seat uses I relevant preferences are sit near an exit for its train seat option. The case a note present, but the task omits records an aisle seat for the flight seat choice option of flight the deciding context. seat choice. Case B: Jordan noted that the flight seat choice option for flight seat choice is a window seat.</td><td></td></tr></table>

Table 20: Representative four-way family from the frozen controlled set. The task is byte-identical for all four variants: “Choose the preference option for Jordan’s flight seat choice.” The example is shown to expose the controlled synthetic structure, not as a naturalistic conversation sample.

## U Extended Related Work and Supplementary Discussion

Long-horizon personalization also raises safety and privacy risks: models can emit training data (Carlini et al., 2021), infer private attributes from text (Staab et al., 2024), and may require userlevel privacy protections (McMahan et al., 2018).

The label-conditioning sensitivity diagnostic uses six artifact conditions: policy-only, state-first, policy-first, forced-correct-state, forced-wrong-state, and deterministic-routing without policy definitions. Forced-state conditions supply a benchmark-implied or deliberately wrong state and ask the model to choose a policy. The wrong state is assigned once before evaluation with seed 20270719 by sampling uniformly from the three non-reference states for each example; the assignment is fixed across models and seeds. These conditions test whether policy routing is sensitive to externally supplied benchmark-associated labels. They do not show that naturally emitted states are faithful internal mechanisms, because the benchmark state labels map deterministically to policy decisions by design.

Instability is a diagnostic, not a replacement for confidence calibration. With three seeds, prediction disagreement is a useful warning signal but not a calibrated probability. We report per-call accuracy for unanimous and non-unanimous examples, majority-vote accuracy for 2–1 splits, and full-disagreement counts. Unanimity is not a guarantee of correctness: many unanimous predictions remain wrong. This supports reporting parse-visible, seed-visible execution traces instead of treating a single sampled output as a stable model behavior.

Decomposed evidence fields do not remove the elicitation bottleneck. The semanticevidence follow-up avoids supplying the old benchmark-state label, but it does not produce a positive result. Llama performs worse when asked to emit decomposed semantic evidence before the policy decision. GPT-OSS cannot be interpreted cleanly for the explicit-evidence arm because the provider rejected those requests under JSON-mode validation. This strengthens the conservative framing: intermediate-field prompting is an audit target, not a solved mechanism.
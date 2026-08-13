# EnterpriseRAG: Benchmarking LLM Instruction Adherence and Robustness under Non-Ideal Enterprise Retrieval

Huiqi Miao, Xinbao Sun, Bo Wang, Fanyu Meng, Lijun Mei, Na Wu, Di Jin, Chao Deng, Junlan Feng Jiutian Research, China Mobile, Beijing, China {miaohuiqi,sunxinbao,wangbo}@cmjt.chinamobile.com

## Abstract

Enterprise RAG deployments face a critical reliability gap: while LLMs satisfy individual constraints at rates up to 84%, only 27% of responses meet all requirements simultaneously, revealing a 57-point orchestration gap. Existing benchmarks assume clean retrieval with simple queries, failing to capture production conditions where noisy documents and multidimensional constraints coexist. We introduce EnterpriseRAG, a benchmark of 983 expertvalidated samples across six domains that systematically simulates three failure modes absent from prior work: retrieval noise, knowledge gaps, and factual conflicts, coupled with complex instructions. Evaluation of 13 stateof-the-art LLMs reveals a severe instruction adherence collapse, where high per-constraint satisfaction masks low holistic compliance. Critical findings expose deep barriers under knowledge gaps and factual conflicts, even with reasoning-enhanced inference, indicating production RAG requires explicit context-aware protocols and calibrated judgment. EnterpriseRAG provides a reproducible foundation for measuring and closing these gaps, directly informing deployment decisions for enterprisescale RAG systems. We will release the benchmark and evaluation framework upon publication.

Keywords: RAG benchmark, instruction following, LLM robustness, enterprise retrieval, knowledge gaps, factual conflicts, retrieval noise

## 1 Introduction

Enterprise RAG systems face complex queries like "Summarize Q3 revenue by region in markdown tables. If data is incomplete, state ’Data Unavailable’ rather than estimating. If audit and management reports conflict, cite both explicitly." requiring simultaneous factual extraction, formatting compliance, and protocol adherence.

These challenges stem not from inadequate factual grounding, but from a fundamental evaluation gap. Current benchmarks assess RAG systems on clean retrieval scenarios with simple queries (Gao et al., 2023; Es et al., 2025), while production deployments face three compounding challenges absent from existing evaluations: (1) complex multiconstraint instructions integrating formatting rules with context-aware protocols for evidence adjudication; (2) high retrieval noise from latencyconstrained systems that surface 10–20 documents with substantial irrelevant content; (3) frequent knowledge failures including coverage gaps and factual conflicts driven by temporal drift or source fallibility.

While recent work advances robustness testing (Zeng et al., 2025) and instruction following (Dong et al., 2024), these efforts evaluate constraints in isolation with synthetic noise, missing the compounding complexity of real enterprise workflows where multiple dimensions interact.

We introduce EnterpriseRAG, a benchmark grounded in real-world enterprise deployments, comprising 983 expert-validated samples across six vertical domains. Unlike prior benchmarks overlaying synthetic instructions onto standard datasets, EnterpriseRAG reflects authentic multi-domain scenarios derived from real operational queries. We preserve original user intents while systematically scaling up constraint complexity through an expertinformed synthesis protocol, and construct three orthogonal non-ideal retrieval modes (irrelevant noise, knowledge gaps, and factual conflicts) validated through LLM-assisted generation and human verification. Our evaluation framework extends traditional RAG metrics with Strict IAS (holistic compliance) versus Loose IAS (per-constraint satisfaction) to expose compositional adherence failures, plus robustness indicators for safety-critical scenarios. Testing 13 state-of-the-art LLMs reveals that while models handle structural formatting adequately, they systematically fail on behavioral protocols, particularly judgment under uncertainty, where even reasoning models achieve insufficient reliability for production deployment.

Table 1: Comparison with prior RAG benchmarks across Source, Complexity, and Robustness.
<table><tr><td>Benchmark</td><td colspan="3">Dataset Source</td><td>Task Complexity</td><td colspan="2">Robustness</td></tr><tr><td></td><td>Human- Curated</td><td>Vertical Domain</td><td>Natural User Queries</td><td>Complex Constraints</td><td>Negative Rejection</td><td>Conflict</td></tr><tr><td>CRUD-RAG(Lyu et al., 2024)</td><td>X</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>CRAG(Yang et al., 2024)</td><td></td><td></td><td>X</td><td>X</td><td>V</td><td>X</td></tr><tr><td>RAGBench(Friel et al., 2025)</td><td>x</td><td></td><td>V</td><td>X</td><td>X</td><td>X</td></tr><tr><td>RAGEval(Zhu et al., 2025)</td><td>x</td><td></td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>FollowRAG(Dong et al., 2024)</td><td></td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>EKRAG(Yu et al., 2025)</td><td></td><td></td><td></td><td>X</td><td>X</td><td>X</td></tr><tr><td>RARE(Zeng et al., 2025)</td><td>x</td><td></td><td>X</td><td>X</td><td>V</td><td></td></tr><tr><td>GaRaGe(Sorodoc et al., 2025)</td><td></td><td></td><td>X</td><td>X</td><td></td><td>X</td></tr><tr><td>EnterpriseRAG (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Contributions. Our contributions are threefold:

• Enterprise-grade RAG benchmark: We introduce EnterpriseRAG, 983 expert-validated instances across six domains, pairing complex multi-constraint instructions with three controlled non-ideal retrieval settings, plus a reproducible construction pipeline.

• Comprehensive evaluation framework:We develop specialized metrics for non-ideal contexts: Loose/Strict IAS for instruction adherence, and rejection/conflict accuracy to measure safetycritical judgment in scenarios with missing or conflicting information.

• Experimental Insights: Across 13 LLMs, we find an orchestration gap of up to 57pp (83.8% Loose vs. 26.8% Strict), with best rejection accuracy only 42.7% and conflict recognition <45%, indicating behavioral judgment under uncertainty as the main bottleneck for enterprise RAG.

## 2 Related Work

RAG evaluation has evolved from factual correctness toward robustness and instruction compliance (Zhao et al., 2024; Gupta et al., 2024). Table 1 positions EnterpriseRAG among representative benchmarks. A recurring pattern across prior work is that retrieval quality and instruction adherence are studied separately—leaving open whether models can satisfy both under realistic enterprise conditions.

RAG Evaluation Paradigms. Early benchmarks focused on factual accuracy and grounding (ALCE (Gao et al., 2023), RAGAS (Es et al., 2025)) or multi-hop reasoning (Tang and Yang, 2024).

While domain-specific benchmarks exist ((Jin et al., 2019);(Pipitone and Alami, 2024);(Chen et al., 2023b)), they often rely on curated sources like Wikipedia (Yang et al., 2018). Recent frameworks like RAGBench (Friel et al., 2025), RAGEval (Zhu et al., 2025) and EKRAG (Yu et al., 2025) offer multi-dimensional metrics and human-curated enterprise samples but lack systematic assessment of complex instruction adherence.

Non-Ideal Retrieval Contexts. Benchmarks like CRAG (Yang et al., 2024), CRUD-RAG (Lyu et al., 2024) and GaRaGe (Sorodoc et al., 2025) address dynamic KBs and calibration. Others such as RGB (Chen et al., 2023a), RARE (Zeng et al., 2025), and Magic Mushroom (Zhang et al., 2025) introduce noise and unanswerability. While conflict (Lee et al., 2025; Choi et al., 2025) and gap management (Guo et al., 2025) exist, they are typically evaluated in isolation, decoupled from uncertainty protocols. Instruction Following in RAG. General benchmarks (Zhou et al., 2023; Jiang et al., 2024; Wen et al., 2024; Zou et al., 2025; Qin et al., 2024) highlight the need for strict verification of formatting and negative constraints. In RAG contexts, MT-RAG (Katsis et al., 2025) and CRP-RAG (Xu et al., 2025) overlook strict protocol adherence, while FollowRAG (Dong et al., 2024) remains limited by synthetic injection and clean contexts. EnterpriseRAG targets the intersection: complex multi-constraint instructions grounded in operational workflows, evaluated under systematic retrieval noise, knowledge gaps, and factual conflicts.

## 3 EnterpriseRAG Benchmark Construction

We construct EnterpriseRAG from production RAG logs, yielding 983 expert-validated instances spanning six domains (Energy, Medical, Legal, Financial, Party Building and Web Search). All data are desensitized to remove PII; raw logs cannot be released, but we will release the desensitized benchmark and a reproducible generation pipeline (details in Appendix A.1).

![](images/04db4c6d12d86e145b3aabdb3b960fe8d39f47592ccefb475611f2347937de0d.jpg)  
Figure 1: EnterpriseRAG Overview. The top section presents the pipeline for complex instruction schema design, while the bottom section shows non-ideal context simulation and evaluation metrics, respectively. All are automatically generated by LLM and quality verified by humans.

Instance format. Each instance is a triple ⟨q, I, D⟩: user query q, a fused multi-constraint instruction set I, and a retrieved document bundle D. Starting from 491 authentic queries, we construct 983 instances by pairing queries with controlled non-ideal retrieval scenarios (overview in Figure 1; subset composition in Table 6). Figure 15 illustrates a Legal domain case study.

Construction pipeline. Our pipeline operationalizes two principles: realistic constraint complexity (from enterprise prompts) and controlled nonideal retrieval (noise/gap/conflict). We: (1) collect and filter queries; (2) synthesize I by fusing atomic constraints and removing internal contradictions; (3) retrieve documents via hybrid retrieval (BM25+dense) and assign non-ideal modes; (4) conduct expert verification for instruction–query consistency and context validity (Appendix A.2). Prompt templates and quality control recipes are documented in Appendix E.1.

![](images/97f81fde5422dd4622bf2f75feabb910eb449b6738941b5840579630294a70e8.jpg)  
Figure 2: Instruction Adherence and Robustness comparison on the knowledge gap subset. Reasoningenhanced models generally demonstrate superior capability in both protocol adherence and refusal of unanswerable queries.

## 3.1 Complex Instruction Schema

We organize constraints into three orthogonal dimensions: Persona Definition, Output Constraints, and Knowledge Interaction Protocols. This schema captures enterprise-critical behavioral requirements (e.g., citation, gap identification, conflict handling) in addition to structural formatting rules. Definitions and distributions are provided in Appendix A.3–A.4 (Table 4, Table 5, Figure 7).

Table 2: Performance of various LLMs on noisy subset. The average scores across the dataset are reported as percentages. The best and second-best scores are marked in bold and underlined, respectively.
<table><tr><td>Model</td><td>Inference Paradigm</td><td colspan="2">RAG Quality</td><td colspan="2">IAS</td><td colspan="3">Loose IAS</td></tr><tr><td></td><td></td><td>Faithfulness</td><td>Answer Coverage</td><td></td><td>Loose Strict</td><td>Persona Definition</td><td>Output Constraints</td><td>Knowledge Interaction Protocol</td></tr><tr><td colspan="9">open-source models</td></tr><tr><td>Qwen3-8b</td><td>Reasoning</td><td>64.8</td><td>56.4</td><td>75.5</td><td>12.3</td><td>70.1</td><td>82.0</td><td>70.3</td></tr><tr><td>Qwen3-14b</td><td>Reasoning</td><td>66.8</td><td>59.4</td><td>77.0</td><td>14.3</td><td>74.2</td><td>81.5</td><td>72.4</td></tr><tr><td>Qwen3-32b</td><td>Reasoning</td><td>64.9</td><td>59.5</td><td>77.2</td><td>15.9</td><td>75.4</td><td>80.1</td><td>75.1</td></tr><tr><td>Qwen3-235B-A22B-Thinking-2507</td><td>Reasoning</td><td>67.1</td><td>64.1</td><td>83.8</td><td>26.8</td><td>82.2</td><td>86.2</td><td>82.5</td></tr><tr><td>Qwen3-30B-A3B-Instruct-2507</td><td>Standard</td><td>63.9</td><td>65.6</td><td>76.4</td><td>13.2</td><td>79.0</td><td>81.5</td><td>68.9</td></tr><tr><td>Qwen3-235B-A22B-Instruct-2507</td><td>Standard</td><td>67.4</td><td>67.1</td><td>80.6</td><td>20.8</td><td>82.3</td><td>83.6</td><td>76.3</td></tr><tr><td>DeepSeek-R1-0528</td><td>Reasoning</td><td>68.9</td><td>66.4</td><td>83.1</td><td>21.9</td><td>81.3</td><td>84.3</td><td>82.9</td></tr><tr><td>DeepSeek-V3.1</td><td>Standard</td><td>69.9</td><td>60.4</td><td>82.2</td><td>22.1</td><td>79.9</td><td>85.9</td><td>78.8</td></tr><tr><td>GLM-4.5</td><td>Reasoning</td><td>76.6</td><td>64.3</td><td>81.6</td><td>21.5</td><td>75.4</td><td>85.7</td><td>78.7</td></tr><tr><td colspan="9">closed-source models</td></tr><tr><td>Gemini-2.5-Pro</td><td>Reasoning</td><td>73.5</td><td>61.6</td><td>83.7</td><td>26.5</td><td>86.6</td><td>85.6</td><td>80.3</td></tr><tr><td>GPT-4.1</td><td>Standard</td><td>69.8</td><td>65.8</td><td>80.0</td><td>19.5</td><td>79.4</td><td>82.1</td><td>77.8</td></tr><tr><td>Claude-Opus-4.5</td><td>Reasoning</td><td>76.4</td><td>68.5</td><td>83.3</td><td>25.3</td><td>84.7</td><td>84.1</td><td>82.4</td></tr><tr><td>Claude-Sonnet-4</td><td>Standard</td><td>76.8</td><td>66.7</td><td>79.8</td><td>19.5</td><td>77.1</td><td>81.5</td><td>79.1</td></tr></table>

![](images/2894d2f05641397f1fc74a8338ee0525a855dbd7642efaf7e491fb13591f5d84.jpg)  
Figure 3: Correlation between Conflict Recognition Rate and Answer Coverage across 13 models on the factual conflict subset (309 cases). Each point represents one model. Reasoning-enhanced models exhibit a strong positive correlation $( \rho = + 0 . 9 0 )$ , while standard models show no significant relationship $( \rho = - 0 . 5 0 )$ .

## 3.2 Non-Ideal Retrieval Scenarios

We construct three non-ideal retrieval scenarios that stress the generator under realistic enterprise failure conditions: Noisy Retrieval (topically similar but contextually irrelevant documents), Knowledge Gaps (topically related but insufficient evidence in retrieved contexts), and Factual Conflicts (contradictory statements in retrieved passages). Because gaps and conflicts are sparse in natural logs, we augment them with controlled procedures while preserving domain coherence; we further validate that synthetic conflicts match natural difficulty on core robustness signals (Appendix A.5

and Appendix B.2).

## 3.3 Evaluation Metrics

Given the open-ended nature of enterprise queries, we report RAG quality and instruction adherence signals without requiring gold reference answers. Faithfulness and Answer Coverage follow a RAGAS-style claim-based evaluation (Es et al., 2025).

Faithfulness (F). We calculate faithfulness as $\mathcal { F } ~ = ~ | C _ { s u p } | / | C _ { t o t a l } |$ , where $C _ { t o t a l }$ denotes all claims extracted from the response, and $C _ { s u p }$ denotes those supported by the retrieved context.

Answer Coverage (C).

$$
\mathcal { C } = \alpha \frac { | C _ { a n s } \cap R | } { | C _ { a n s } | } + ( 1 - \alpha ) \frac { | S _ { a n s } \cap R | } { | S _ { a n s } | }\tag{1}
$$

where $C _ { a n s }$ and $S _ { a n s }$ denote core and supplementary claims from contexts, respectively, R represents the response content, and $\alpha = 0 . 7$ weights core claims higher.

Instruction Adherence Score (IAS). We report: Loose IAS as the proportion of satisfied constraints, and Strict IAS as a binary score indicating whether all constraints are satisfied.

Robustness metrics. For non-ideal subsets, we compute: Rejection Accuracy on knowledge gaps, and Conflict Recognition Accuracy on factual conflicts.

<table><tr><td>Model</td><td>CR |Model</td><td>CR</td></tr><tr><td>DeepSeek-R1† 44.3</td><td>DeepSeek-V3.1 Qwen3-30B-I</td><td>29.8 28.5</td></tr><tr><td>Gemini-2.5-Pro†</td><td>42.3</td><td></td></tr><tr><td>Qwen3-235B-T†</td><td>41.4  $\mathrm { Q w e n } 3 { - } 1 4 \mathrm { B - T } ^ { \dag }$ </td><td>27.2</td></tr><tr><td>GLM-4.5†</td><td>40.1  $\mathrm { C l a u d e { \mathrm { - } } O p u s } ^ { \dagger }$ </td><td>26.9</td></tr><tr><td>Qwen3-235B-I</td><td>37.5 Claude-Sonnet</td><td>25.9</td></tr><tr><td>Qwen3-32B-T†</td><td>36.6  $\mathrm { Q w e n } 3 { - } 8 \mathrm { B - T ^ { \dag } }$  GPT-4.1</td><td>23.9 18.5</td></tr></table>

<sup>†</sup>Reasoning-enhanced. CR: Conflict Recog. (%).

Table 3: Conflict recognition rates (CR) across 13 models.

## 4 Experiments

## 4.1 Experimental Setup

Models. We evaluate 13 LLMs spanning open/closed-source and standard/reasoningenhanced variants (Yang et al., 2025; DeepSeek-AI et al., 2025; OpenAI, 2025; Team et al., 2025a; Comanici et al., 2025; Anthropic, 2025), full list in Table 2. For consistency, we adopt simplified names after the first mention: “Thinking” models are denoted as -Thinking (abbrev. -T) and standard instruction-tuned counterparts as -Instruct (abbrev. -I). We omit version suffixes (e.g., -2507) unless needed for disambiguation.

Evaluation protocol. Results are reported on three non-ideal subsets: Noisy Retrieval (n=447), Knowledge Gaps (n=227), and Factual Conflicts (n=309). IAS evaluation uses rule-based checks for structural constraints (Zhou et al., 2023) and LLM-as-a-judge (Kimi-k2-thinking (Team et al., 2025b)) for behavioral protocols. Not all instances include explicit Knowledge Interaction Protocol constraints; we therefore compare naturally protocol-present vs. protocol-absent cases for robustness analyses. Evaluation prompt templates are in Appendix E.

Evaluator reliability. Cross-judge comparison across three LLM evaluators shows stable scores and consistent model rankings (Section B.1). On 150 human-annotated samples, experts achieve strong agreement (κ=0.85), and the LLM evaluator (Kimi-k2-thinking) aligns well with humanannotated gold labels (κ=0.77, 88% agreement), with especially high alignment on conflict recognition (κ=0.93; Section B.3).

## 4.2 Main Results

Finding 1: Orchestration under noisy retrieval. Table 2 reveals a severe adherence collapse: Loose IAS achieves up to 83.8%, yet Strict IAS reaches only 26.8% (Qwen3-235B-Thinking). This 57- point gap quantifies the compositional bottleneck where models satisfy individual constraints but fail holistic compliance. Reasoning-enhanced models consistently outperform standard variants, with the largest gains in Knowledge Interaction Protocols.

![](images/8caee7dde2ea34aa8fd7d1d86ef75452bc8c1dbee59245752d7c76465e94fde2.jpg)  
Figure 4: Error distribution by constraint category. Knowledge Interaction Protocols exhibit the highest failure rates, confirming that behavioral judgment, not formatting, is the core bottleneck.

![](images/6188f13b78d2514e00f1b5d5d23deb31472e1341aa5aa90937f428e66c6939f6.jpg)  
Figure 5: Reasoning vs. standard model accuracy across RAG scenarios. Blue segments denote reasoning gains over standard baselines (yellow). Error bars: 95% CI. $^ { * } p < . 0 5 , ^ { * * } p < . 0 1 , ^ { * * * } p < . 0 0 1$ (McNemar’s test).

Finding 2: Rejection under knowledge gaps. In production, hallucinating on unanswerable queries is often more harmful than being unhelpful. Figure 2 exposes a pervasive helpfulness bias: Qwen3-30B-Instruct achieves only 6.6% rejection accuracy, hallucinating in 93.4% of unanswerable cases. Reasoning-enhanced models improve substantially (Claude-Opus-4.5: 42.7%), yet remain far from production-grade reliability.

![](images/a8b54414a577afaff89e065a0c647c83e910bfdca66ef385805d4976cd704bbe.jpg)

B. Conflict Recognition Accuracy  
![](images/290a2c7ea8fa290d054a108aaad1ebd10e8d5344f4e204715da135a7ba9557b0.jpg)  
Figure 6: Effect of knowledge interaction protocol on (A) rejection accuracy and (B) conflict recognition. Points show accuracy differences (with/without protocol) with 95% CIs. The protocol significantly improves conflict recognition in all 13 models, with more modest effects on rejection accuracy (2/13 significant). Independent samples (with/without: 157/69 for A, 113/193 for B); p-values from $\chi ^ { 2 }$ test with Yates’ correction.

Finding 3: Conflict recognition. Table 3 shows conflict detection remains a bottleneck: top models reach only 40–44% recognition (DeepSeek-R1: 44.3%), while GPT-4.1 detects merely 18.5%. Figure 3 shows reasoning-enhanced models achieve a strong positive correlation between recognition and coverage $( \rho { = } + 0 . 9 0 , p { < } 0 . 0 1 )$ ), while standard models show no consistent relationship $( \rho { = } { - } 0 . 5 0 )$ with high variance. This suggests inference-time computation may resolve the traditional safetyinformativeness dilemma. Synthetic and natural conflicts show equivalent difficulty on core metrics (Section B.2).

## 4.3 Analysis

Protocol bottleneck. Figure 4 decomposes IAS failures by constraint category. Knowledge Interaction Protocols exhibit the highest error rates and variance, particularly for citation and gap identification. Comparing Qwen3-235B-Thinking to its Instruct counterpart, the largest reasoning gains occur precisely in these protocol dimensions, confirming that judgment under uncertainty is the core enterprise bottleneck.

Scaling. Within Qwen3-Thinking, Strict IAS scales non-linearly (12.3% at 8B → 26.8% at 235B) while Faithfulness saturates $( 6 4 . 8 \%  6 7 . 1 \% )$ , indicating orchestration is an emergent capability requiring substantial scale.

Reasoning vs. standard instruction-tuned variants. Figure 5 compares matched reasoning vs. standard variants (Qwen3-235B-Thinking vs. Qwen3-235B-Instruct; DeepSeek-R1 vs. DeepSeek-V3.1). Across scenarios, reasoning variants exhibit substantial robustness gains: Qwen3- 235B-Thinking boosts rejection accuracy by 18.1pp and DeepSeek-R1 improves conflict recognition by 14.4pp, consistent with reduced helpfulness bias. For Strict IAS, Qwen3-235B-Thinking shows consistent gains (+6.1pp to +10.6pp; p<.01), while DeepSeek-R1 shows minimal improvement, suggesting architecture-dependent benefits.

Effect of explicit protocols. Figure 6 compares instances with explicit Knowledge Interaction Protocol constraints to those without such constraints under the same retrieval failure mode. Overall, explicit protocols yield a large and consistent gain in conflict recognition across all 13 models, but only modest improvements in rejection under knowledge gaps. This asymmetry suggests that protocols help most when the failure is explicit in-context (contradictions), whereas proper refusal requires a harder judgment of evidence sufficiency and separating parametric knowledge from retrieved evidence. Notably, Claude-Opus-4.5 shows a small, non-significant decrease, indicating potential interaction with model-specific safety behaviors. Because this is an observational comparison (protocol presence is not randomized), we report domainlevel breakdowns in Section D.1.

## 5 Conclusion

EnterpriseRAG combines complex multi-constraint instructions with three non-ideal retrieval modes across 983 expert-validated instances. Across 13 LLMs, we find a persistent orchestration collapse: even the best model reaches 83.8% perconstraint adherence (Loose IAS) but only 26.8% holistic compliance (Strict IAS), leaving a 57-point gap. Robustness failures concentrate in knowledgeinteraction protocols: under knowledge gaps, models frequently over-answer despite explicit refusal requirements (with Claude-Opus-4.5 peaking at 42.7%); under factual conflicts, even the strongest systems recognize contradictions in fewer than half of cases (led by DeepSeek-R1 at 44.3%).

Practically, our results suggest that enterpriseready RAG requires (i) training and evaluation targeted at protocol-level judgment (evidence sufficiency, calibrated refusal, and conflict-aware reporting), not just formatting or factuality; and (ii) explicit operational protocols in prompts, which reliably improve conflict handling but are insufficient to solve evidence-gap refusal. EnterpriseRAG provides a realistic and reproducible foundation to measure and close these gaps.

## 6 Limitations

While EnterpriseRAG encompasses six diverse domains, the current scope is limited to text-based RAG. Multimodal contexts, such as those involving charts or images within PDFs, are not yet included. Additionally, our reliance on a reasoning-enhanced LLM as an evaluator, while effective, may introduce bias compared to human evaluation, although our sampling checks indicate high alignment. Finally, the strict adherence metric is binary and stringent; future metrics could explore more nuanced semantic gradations of constraint satisfaction.

## References

Anthropic. 2025. Introducing Claude Opus 4.5. Blog post.

Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2023a. Benchmarking large language models in retrieval-augmented generation. Preprint, arXiv:2309.01431.

Wei Chen, Qiushi Wang, Zefei Long, Xianyin Zhang, Zhongtian Lu, Bingxuan Li, Siyuan Wang, Jiarong Xu, Xiang Bai, Xuanjing Huang, and Zhongyu Wei. 2023b. DISC-FinLLM: A Chinese Financial Large Language Model based on Multiple Experts Finetuning.

Eunseong Choi, June Park, Hyeri Lee, and Jongwuk Lee. 2025. Conflict-aware soft prompting for retrieval-augmented generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 26969–26983, Suzhou, China. Association for Computational Linguistics.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Preprint, arxiv:2507.06261.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,

Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, and 181 others. 2025. DeepSeek-r1: Incentivizing reasoning capability in LLMs via reinforcement learning. Preprint, arxiv:2501.12948 [cs].

Guanting Dong, Xiaoshuai Song, Yutao Zhu, Runqi Qiao, Zhicheng Dou, and Ji-Rong Wen. 2024. Toward General Instruction-Following Alignment for Retrieval-Augmented Generation. arXiv preprint. ArXiv:2410.09584 [cs].

Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. 2025. Ragas: Automated Evaluation of Retrieval Augmented Generation. arXiv preprint. ArXiv:2309.15217 [cs].

Robert Friel, Masha Belyi, and Atindriyo Sanyal. 2025. RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems. arXiv preprint. ArXiv:2407.11005 [cs].

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling Large Language Models to Generate Text with Citations. arXiv preprint. ArXiv:2305.14627 [cs].

Xiaofan Guo, Yaxuan Luan, Yue Kang, Xiangchen Song, and Jinxu Guo. 2025. Llm-centric rag with multi-granular indexing and confidence constraints. Preprint, arXiv:2510.27054.

Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. 2024. A comprehensive survey of retrieval-augmented generation (rag): Evolution, current landscape and future directions. Preprint, arXiv:2410.12837.

Xinguang Jiang, Sihan Hu, Dingfu Yu, Yuhao Zhang, Zhongliang Yang, Yu Li, Linna Zhou, and Valuesimplex AI Lab. 2023. FinLongEval. https://github. com/valuesimplex/FinLongEval.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. 2024. Follow-Bench: A multi-level fine-grained constraints following benchmark for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4667–4688, Bangkok, Thailand. Association for Computational Linguistics.

Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. 2019. PubMedQA: A Dataset for Biomedical Research Question Answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567–2577, Hong Kong, China. Association for Computational Linguistics.

Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj

Shah, Huaiyu Zhu, Danish Contractor, and Marina Danilevsky. 2025. MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems. arXiv preprint. ArXiv:2501.03468 [cs].

Jungyeon Lee, Kangmin Lee, and Taeuk Kim. 2025. Magic: A multi-hop and graph-based benchmark for inter-context conflicts in retrieval-augmented generation. Preprint, arXiv:2507.21544.

Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu, Tong Xu, and Enhong Chen. 2024. CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models. arXiv preprint. ArXiv:2401.17043 [cs].

OpenAI. 2025. Introducing GPT-4.1 in the API. Blog post.

Nicholas Pipitone and Ghita Houir Alami. 2024. LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain. arXiv preprint. ArXiv:2408.10343 [cs].

Yanzhao Qin, Tao Zhang, Tao Zhang, Yanjun Shen, Wenjing Luo, Haoze Sun, Yan Zhang, Yujing Qiao, Weipeng Chen, Zenan Zhou, Wentao Zhang, and Bin Cui. 2024. SysBench: Can large language models follow system messages? Preprint, arxiv:2408.10943 [cs]. Version: 1.

Ionut-Teodor Sorodoc, Leonardo F. R. Ribeiro, Rexhina Blloshmi, Christopher Davis, and Adrià de Gispert. 2025. GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation. arXiv preprint. ArXiv:2506.07671 [cs].

Yixuan Tang and Yi Yang. 2024. MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries. arXiv preprint. ArXiv:2401.15391 [cs].

GLM-4 5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, and 152 others. 2025a. GLM-4.5: Agentic, reasoning, and coding (ARC) foundation models. Preprint, arxiv:2508.06471 [cs].

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, and 150 others. 2025b. Kimi k2: Open agentic intelligence. Preprint, arxiv:2507.20534 [cs].

Bosi Wen, Pei Ke, Xiaotao Gu, Lindong Wu, Hao Huang, Jinfeng Zhou, Wenchuang Li, Binxin Hu, Wendy Gao, Jiaxin Xu, Yiming Liu, Jie Tang, Hongning Wang, and Minlie Huang. 2024. Benchmarking Complex Instruction-Following with Multiple Constraints Composition.

Kehan Xu, Kun Zhang, Jingyuan Li, Wei Huang, and Yuanzhuo Wang. 2025. CRP-RAG: A retrievalaugmented generation framework for supporting complex logical reasoning and knowledge planning. Electronics, 14(1):47.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arxiv:2505.09388 [cs].

Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang, and 8 others. 2024. CRAG – Comprehensive RAG Benchmark. arXiv preprint. ArXiv:2406.04744 [cs].

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. Preprint, arXiv:1809.09600.

Tan Yu, Wenfei Zhou, Leiyang Leiyang, Aaditya Shukla, Mmadugula Mmadugula, Pritam Gundecha, Nicholas Burnett, Anbang Xu, Viseth Viseth, Tbar Tbar, Rama Akkiraju, and Vivienne Zhang. 2025. EKRAG: Benchmark RAG for Enterprise Knowledge Question Answering. In Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing, pages 152–159, Albuquerque, New Mexico, USA. Association for Computational Linguistics.

Yixiao Zeng, Tianyu Cao, Danqing Wang, Xinran Zhao, Zimeng Qiu, Morteza Ziyadi, Tongshuang Wu, and Lei Li. 2025. RARE: Retrieval-aware robustness evaluation for retrieval-augmented generation systems. Preprint, arXiv:2506.00789.

Yuxin Zhang, Yan Wang, Yongrui Chen, Shenyu Zhang, Xinbang Dai, Sheng Bi, and Guilin Qi. 2025. Magic mushroom: A customizable benchmark for finegrained analysis of retrieval noise erosion in rag systems. Preprint, arXiv:2506.03901.

Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2024. Retrieval-augmented generation for ai-generated content: A survey. Preprint, arXiv:2402.19473.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. Preprint, arxiv:2311.07911 [cs].

Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan, Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang, Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and

Maosong Sun. 2025. RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework. arXiv preprint. ArXiv:2408.01262 [cs].

Tao Zou, Xinghua Zhang, Haiyang Yu, Minzheng Wang, Fei Huang, and Yongbin Li. 2025. EIFBENCH: Extremely complex instruction following benchmark for large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20941–20964. Association for Computational Linguistics.

## A Dataset Details & Statistics

## A.1 Data Source, Privacy, and Domains

Our benchmark is constructed from real-world enterprise operational logs in China, collected under internal data usage agreements. The native language of EnterpriseRAG is Chinese. To ensure privacy, all raw logs undergo a strict multi-stage desensitization pipeline, including rule-based removal of sensitive fields and manual expert review. Consequently, no personally identifiable information (PII), proprietary identifiers, or confidential business content is included.

Due to privacy and compliance constraints, the raw data cannot be publicly released. However, we release the desensitized benchmark dataset, along with a reproducible synthetic data generation pipeline and a representative sample subset, to enable independent verification and follow-up research.

Domain Specifications. Based on the aforementioned data sources, we select six vertical domains constructed according to three complementary criteria: (1) Industrial Prevalence—domains with substantial enterprise RAG deployments; (2) Data Accessibility—availability of production logs and domain expertise; and (3) Task Diversity—coverage of distinct knowledge processing paradigms.

• Energy: Procedural QA with version drift (e.g., equipment maintenance protocols across ERP system transitions).

• Medical: Clinical abstraction from fragmented dialogues, including discharge summaries and treatment synthesis.

• Legal: Multi-hop statute-case correlation with jurisdictional hierarchies.

• Financial: Investment advisory and policy interpretation tasks, partially leveraging Fin-LongEval (Jiang et al., 2023).

• Party Building: Regulatory knowledge management and policy interpretation in organizational contexts.

• Web Search: Real-time queries emphasizing information freshness and source credibility.

## A.2 Data construction process

Our data construction follows a four-stage pipeline: (1) Query Collection: We gather 491 authentic user queries from operational logs across six domains, ensuring diversity in information needs and complexity levels. (2) Instruction Synthesis: For each query, we apply the constraint fusion protocol (Section 3.3.2) to generate complex multidimensional instructions averaging 8 constraints per sample. (3) Context Preparation: We retrieve relevant documents using hybrid retrieval (BM25 + dense retrieval) and apply non-ideal simulation (Section 3.4) to create noise, knowledge gaps, and factual conflicts. (4) Human Verification: Instead of generating reference answers, the constructed samples undergo a rigorous two-stage expert verification process. Experts verify Instruction-Query Consistency and Context Validity (confirming topical relevance and the presence/absence of necessary information for non-ideal scenarios), filtering out low-quality samples to ensure ecological validity.

## A.3 Constraints Taxonomy

Table 4 and figure 7 outlines the taxonomy across three dimensions. Persona Definition establishes the model’s virtual identity and audience context, while Output Constraints dictate the structural format and stylistic boundaries. Crucially, the Knowledge Interaction Protocol enforces strict behavioral rules for evidence handling, such as citation and conflict resolution. This dimension moves beyond simple formatting to ensure the rigorous reliability required in enterprise environments.

## A.4 Domain Statistics

Table 5 shows that the Medical and Financial domains exhibit the highest complexity, with 10.21 and 8.74 average constraints respectively. Legal tasks feature the highest density of Knowledge Protocol constraints (3.45). In contrast, Web Search peaks in Output Constraints (4.73) with lower protocol requirements (1.80), while Energy shows the lowest Persona usage (0.85).

Figure 7 illustrates the overall constraint distribution. Output Constraints constitute the majority (53.1%), followed by Knowledge Interaction Protocols (29.2%) and Persona Definitions. Within protocols, Citation and Conflict Handling represent the most frequent sub-dimensions.

Table 4: Detailed taxonomy and definitions of sub-dimension constraints. The table describes the specific requirements for Persona Definition, Output Constraints, and Knowledge Interaction Protocols used in the benchmark.
<table><tr><td>Category</td><td>Sub-Dimension</td><td>Description</td></tr><tr><td rowspan="2">Persona Definition</td><td>Role</td><td>Defines the user&#x27;s virtual identity, such as a software engineer, medical expert, or customer service representative.</td></tr><tr><td>Audience</td><td>Specifies the target readers of generated content, influencing the level of detail, terminology, and professionalism.</td></tr><tr><td rowspan="4">Output Constraints</td><td>Format</td><td>Specifies the output structure, such as Markdown, JSON, or requirements like the number of sections to include.</td></tr><tr><td>Content</td><td>Defines content requirements including word count limits, prefix/suffix specifications, and keyword frequency constraints.</td></tr><tr><td>Negative</td><td>Explicitly specifies prohibited elements in the output, testing the model&#x27;s fine-grained control capability.</td></tr><tr><td>Ordering</td><td>Specifies the arrangement of output content, e.g., chronological order, alphabetical sorting, or frequency-based ranking.</td></tr><tr><td rowspan="5">Knowledge Interaction</td><td>Style Conflict</td><td>Defines the response tone, ranging from rigorous and formal to relaxed and conversational. Defines how the model should respond when contradictory information exists within the</td></tr><tr><td>Handling</td><td>knowledge base.</td></tr><tr><td>Knowledge Gap</td><td>Requires the model to identify and explicitly report when necessary information is missing or unavailable.</td></tr><tr><td>Uncertainty Expr.</td><td>Specifies how the model should express uncertainty when information is ambiguous or evidence is insufficient.</td></tr><tr><td>Source Filtering Citation</td><td>Instructs the model to selectively trust or ignore specific types of information sources. Requires key conclusions to be accompanied by directly cited original text excerpts as</td></tr></table>

Table 5: Statistics of queries and constraint density. The table shows the number of samples and the average number of constraints per query across the three orthogonal dimensions for each of the six vertical domains.
<table><tr><td>Domain</td><td>#Data</td><td>#Constraints</td><td>Persona Def.</td><td>Output Const.</td><td>Knowledge Protocol</td></tr><tr><td>Financial</td><td>100</td><td>8.74</td><td>1.94</td><td>5.08</td><td>1.72</td></tr><tr><td>Energy</td><td>100</td><td>7.01</td><td>0.85</td><td>3.51</td><td>2.65</td></tr><tr><td>Legal</td><td>44</td><td>8.61</td><td>1.59</td><td>3.57</td><td>3.45</td></tr><tr><td>Medical</td><td>100</td><td>10.21</td><td>1.46</td><td>5.02</td><td>3.16</td></tr><tr><td>Party Building</td><td>48</td><td>7.79</td><td>1.21</td><td>4.08</td><td>2.50</td></tr><tr><td>Web Search</td><td>99</td><td>8.41</td><td>1.87</td><td>4.73</td><td>1.80</td></tr><tr><td>All</td><td>491</td><td>8.40</td><td>1.50</td><td>4.46</td><td>2.45</td></tr></table>

Note: #Data = Number of Data Samples; #Const. = Avg. Constraints per query.

## A.5 Non-Ideal Context Composition

Table 6 reports the distribution of non-ideal context types in the final 983 instances. Knowledge gaps and factual conflicts are augmented due to sparsity in naturally occurring logs; we therefore explicitly report the natural vs. augmented breakdown for transparency.

## B Evaluation Reliability Analysis

## B.1 Cross-Judge Consistency

We validate the stability of our evaluation protocol by comparing three LLM judges: Kimi-k2- thinking, Qwen3-235B-Thinking, and GPT-4o. As shown in Table 7, the raw evaluation scores (upper section) exhibit minimal variance across judges; for instance, Loose IAS scores for DeepSeek-V3.1 differ by less than 0.5% between Kimi and Qwen3.

Table 6: Distribution of non-ideal contexts in EnterpriseRAG.
<table><tr><td>Context</td><td>Count</td><td>Ratio</td><td>Natural Occ.</td><td>Augment.</td></tr><tr><td>Noisy</td><td>447</td><td>45.5%</td><td>447 (100%)</td><td></td></tr><tr><td>Knowledge Gaps</td><td>227</td><td>23.1%</td><td>12 (5.3%)</td><td>215 (94.7%)</td></tr><tr><td>Factual Conflicts</td><td>309</td><td>31.4%</td><td>27 (8.7%)</td><td>282 (91.3%)</td></tr></table>

Note: Natural Occ. = Natural Occurrence; Augment. = Augmentation. Percentages in parentheses indicate the proportion before augmentation. We augmented underrepresented failure modes to reflect realistic deployment distributions.

This consistency is rigorously confirmed by statistical reliability metrics (lower section). We observe "Good" to "Excellent" reliability across all dimensions, with Intraclass Correlation Coefficients (ICC) exceeding 0.80 for Loose IAS and approaching 1.0 for robustness metrics. The high Spearman’s $\rho$ further indicates that different judges preserve the same relative model rankings, ensuring that our reported performance gaps are robust to the choice of evaluator.

![](images/4256e5decc45125fef9c378c348aa3df39392999d086cc34161dec1c2d27c2c1.jpg)  
Figure 7: Hierarchical distribution of constraints in EnterpriseRAG. The inner ring represents the three primary categories, while the outer ring details the specific subdimensions.

## B.2 Synthetic vs. Real-world Data Performance

Figure 8 and Table 8 contrast model performance on naturally occurring (n=27) versus synthetic (n=282) conflicts. Across five models, synthetic samples show slightly lower Faithfulness $( \Delta = -$ 0.143) and Answer Coverage $( \Delta = - 0 . 0 2 0 )$ due to their engineered contradictions. Critically, we observe statistical equivalence in the core robustness metrics of Conflict Recognition Accuracy and Strict IAS $( \Delta = - 0 . 0 3 8 , \mathrm { p } = . 1 2 0 ; \Delta = - 0 . 0 0 4 , \mathrm { p } =$ .826), with 95% confidence intervals within ±0.2. This confirms that our synthesis pipeline effectively replicates the difficulty of real-world scenarios, ensuring the ecological validity of the augmentation strategy.

## B.3 Human–LLM Judge Alignment Study

To ensure rigorous evaluation, we employed a twostage annotation protocol on a stratified sample of 150 instances. First, two experts independently labeled the data, achieving robust Inter-Annotator Agreement (IAA) (avg. κ=0.85, see Table 9), which validates the clarity of our instruction taxonomy. Disagreements were adjudicated to establish a Gold Standard.

Against this baseline, the LLM judge demonstrates substantial reliability with an average κ of 0.77. Conflict Recognition achieves the highest alignment $\scriptstyle ( \kappa = 0 . 9 3 )$ due to the objective nature of contradiction detection, while Strict Adherence $\scriptstyle ( \kappa = 0 . 6 8 )$ exhibits minor divergence on borderline formatting nuances.

## C Statistical Analysis Details

## C.1 Figure 3

Data and Method Each point: model-level aggregate over n=309 conflict instances (X: conflict recognition rate; Y: mean answer coverage). We compute Spearman’s $\rho$ separately for reasoningenhanced $( n { = } 8 )$ and standard (n=5) models due to their distinct architectures.

Limitations Small sample sizes limit power. Models within families (e.g., Qwen3) may not be fully independent; sensitivity analysis yields consistent patterns. Model-level correlations may not reflect instance-level relationships.

## C.2 Figure 5

Method: McNemar’s test (one-sided) for paired binary outcomes. Each instance evaluated by both reasoning-enhanced and standard models within the same family.

Samples: Noisy Retrieval (n=447), Knowledge Gaps (n=227), Factual Conflicts (n=309). Exact binomial test when discordant pairs $< 2 5 ;$ otherwise z-test with continuity correction.

Confidence Intervals: Percentile bootstrap (10,000 resamples) preserving pairing.

Multiple Testing: Uncorrected p-values for 10 planned comparisons; Bonferroni correction (α=.005) does not change conclusions.

## C.3 Figure 6

We compare accuracy between independent groups (samples with vs. without the protocol) using standard methods for comparing two proportions.

For each model, we construct a 2×2 contingency table and apply: Pearson’s $\chi ^ { 2 }$ test (with Yates’ correction) when expected counts $\geq 5$ , Fisher’s exact test otherwise.

We report the accuracy difference $\Delta = \hat { p } _ { \mathrm { w i t h } }$ $\hat { p } _ { \mathrm { w i t h o u t } }$ with 95% Wald confidence intervals.

Sample sizes: rejection (157 with / 69 without), conflict recognition (113 / 193). Analysis used

Table 7: Consistency and Reliability Analysis of Evaluator LLMs. The upper section compares the raw evaluation scores of Kimi, Qwen3, and GPT-4o across three metrics. The lower section reports the statistical inter-annotator agreement metrics, validating the stability of the evaluation protocol. ICC values are reported with 95% confidence intervals.
<table><tr><td rowspan="2">Model / Metric</td><td colspan="3">Loose IAS</td><td colspan="3">Reject Acc</td><td colspan="3">Conflict Acc</td></tr><tr><td>Kimi</td><td>Qwen3</td><td>GPT-40</td><td>Kimi</td><td>Qwen3</td><td>GPT-40</td><td>Kimi</td><td>Qwen3</td><td>GPT-40</td></tr><tr><td colspan="10">Raw Evaluation Scores (%)</td></tr><tr><td>Qwen3-235B-Thinking</td><td>89.6</td><td>89.0</td><td>91.9</td><td>27.8</td><td>27.9</td><td>28.8</td><td>37.8</td><td>39.1</td><td>39.1</td></tr><tr><td>DeepSeek-V3.1</td><td>86.0</td><td>86.2</td><td>90.4</td><td>18.9</td><td>19.4</td><td>19.4</td><td>28.3</td><td>28.9</td><td>31.0</td></tr><tr><td>Gemini-2.5-Pro</td><td>84.9</td><td>89.1</td><td>88.3</td><td>33.6</td><td>35.8</td><td>34.5</td><td>37.2</td><td>39.4</td><td>36.9</td></tr><tr><td>GPT-4.1</td><td>82.7</td><td>83.8</td><td>85.8</td><td>10.1</td><td>9.3</td><td>9.3</td><td>16.6</td><td>16.6</td><td>17.6</td></tr><tr><td colspan="10">Inter-Judge Reliability Statistics</td></tr><tr><td>ICC (2,k)</td><td colspan="3">0.809 [0.11–0.99]</td><td colspan="3">0.999 [0.99–1.00]</td><td colspan="3">0.996 [0.98–1.00]</td></tr><tr><td>Avg. Spearman&#x27;s ρ</td><td colspan="3">0.600</td><td colspan="3">1.000</td><td colspan="3">0.867</td></tr><tr><td>Reliability Verdict</td><td colspan="3">Good</td><td colspan="3">Excellent</td><td colspan="3">Excellent</td></tr></table>

![](images/f40118b7d3a561f43e4b89627dcb266814f85aeb8f0ca148fedc902fba5182b8.jpg)  
Figure 8: Equivalence testing results comparing synthetic and natural conflict data across five evaluation metrics. Points represent mean differences (Natural - Synthetic) with 95% confidence intervals; dashed lines mark equivalence bounds (±0.2). Core robustness metrics (Conflict Recognition and Strict IAS) achieve statistical equivalence, validating the synthetic data generation approach.

SciPy. Of 26 tests, 25 used $\chi ^ { 2 }$ and 1 used Fisher’s exact. Uncorrected p-values are reported; Bonferroni correction (α = .002) does not alter substantive conclusions.

## C.4 Figure 8

We use TOST equivalence testing to validate that synthetic conflicts (n=282) replicate natural conflict difficulty (n=27). Equivalence margin: δ=±0.2 (20pp, based on RAG benchmark reliability thresholds). Decision rule: 95% CI for difference (Natural − Synthetic) must fall within [−0.2, 0.2].

Aggregate mean accuracy across 5 models for each dataset; bootstrap 95% CI (10,000 resamples).

Limitations Small natural sample (n=27); model-level aggregation; families share architectures.

## C.5 Table 8

We use paired-samples t-test to compare 13 models’ performance on natural (n = 27) vs. synthetic (n = 282) conflicts, with Cohen’s d for effect sizes.

Limitations Models within families may not be fully independent; unequal natural/synthetic sample sizes affect precision but not paired comparison validity.

Table 8: Natural vs. Synthetic Data Comparison. n.s. denotes no significant difference $( p \ge 0 . 0 5 )$ , indicating successful replication of difficulty on core metrics.
<table><tr><td>Metric</td><td>Natural</td><td>Synth.</td><td>Diff.</td><td>p-val</td></tr><tr><td>Core Robustness (Target: Equivalent)</td><td></td><td></td><td></td><td></td></tr><tr><td>Conflict Recog. Strict IAS</td><td>0.360 0.160</td><td>0.322 0.156</td><td>-0.038 -0.004</td><td>.120 (n.s.) .826 (n.s.)</td></tr><tr><td>General Quality</td><td></td><td></td><td></td><td></td></tr><tr><td>Loose IAS</td><td>0.793</td><td>0.767</td><td>-0.026</td><td>&lt;.001***</td></tr><tr><td>Answer Cov.</td><td>0.551</td><td>0.532</td><td>-0.020</td><td>.054</td></tr><tr><td>Faithfulness</td><td>0.827</td><td>0.684</td><td>-0.143</td><td>&lt;.001***</td></tr></table>

Table 9: Human-LLM Judge Alignment Study. Analysis based on a stratified sample of 150 instances (50 per dimension). Data Quality: Inter-Annotator Agreement (IAA) between two human experts. LLM Judge Reliability: Primary evaluator (Kimi-k2-thinking) vs. adjudicated gold standard.
<table><tr><td rowspan="2">Evaluation Dimension</td><td>Data Quality</td><td colspan="2">LLM Judge Reliability</td></tr><tr><td>Human IAA (κ)</td><td>Cohen&#x27;s κ</td><td>Agreement Rate (%)</td></tr><tr><td>Strict IAS</td><td>0.73</td><td>0.68</td><td>86</td></tr><tr><td>Proper Rej.</td><td>0.87</td><td>0.71</td><td>87</td></tr><tr><td>Conflict Recog.</td><td>0.96</td><td>0.93</td><td>91</td></tr><tr><td>Overall avg.</td><td>0.85</td><td>0.77</td><td>88</td></tr></table>

## D Additional Experiments

## D.1 Fine-grained Analysis

Figure 9 illustrates the performance of five representative LLMs—including both reasoningenhanced and standard instruction-tuned variants—across the six vertical domains of EnterpriseRAG under noisy retrieval conditions. Across all domains, models generally maintain high Faithfulness and Loose IAS , but experience a sharp "orchestration collapse" in Strict IAS, confirming that simultaneously satisfying multiple domain-specific constraints remains a primary bottleneck. Specifically, domains with higher complexity and stricter behavioral requirements, such as Medical and Legal, exhibit lower absolute Strict IAS scores compared to Web Search. This trend aligns with the domain statistics in Table 5, which show that Medical and Legal tasks feature the highest average number of constraints and the densest concentration of Knowledge Interaction Protocols. Furthermore, reasoning-enhanced models (e.g., Qwen3-235B-Thinking and DeepSeek-R1) consistently outperform their counterparts across all metrics and domains, particularly in Strict IAS and Answer Coverage. These results suggest that the difficulty of instruction adherence is inherently tied to domainspecific constraint density, and that robust performance in complex enterprise scenarios is an emergent capability heavily dependent on inferencetime reasoning rather than simple pattern matching.

## E Complete Prompts Repository

Note: All prompts and examples presented in this paper have been translated from the original Chinese for readership clarity. The actual evaluation was performed using the Chinese versions.

## E.1 The Prompts for Data Construction

## E.2 The Prompts for Evaluation

Figure 15 presents an example from the Legal domain of EnterpriseRAG. This case illustrates the extreme difficulty of satisfying 12 simultaneous constraints while processing a noisy context containing 20 retrieved documents, many of which are domain-adjacent (e.g., Litigation Law vs. Reconsideration Law) but factually insufficient.

![](images/cc7befb8e785bce1d099595a5848e2537604859ff2140221d1f28dd89049e860.jpg)  
Figure 9: Model performance on noisy retrieval contexts across six vertical domains. Radar charts illustrate the trade-offs between Faithfulness, Answer Coverage, and Instruction Adherence (Strict/Loose) for representative models.

![](images/c32daf596e2566c1854c9cb9dbabe410e76be031284f6a511b7508df8e441015.jpg)  
Figure 10: The prompt template used for generating EnterpriseRAG Atomic Libraries.

![](images/c5b56212be31c5a283c49f082e04c89ffe35b786975ac3132d2c3ef4a5ca96b4.jpg)  
Figure 11: The prompt template used for detecting conflict in combined constraints.

## D.1.3 Prompt Templates for Instruction Refinement

Prompt 1: Filter Irrelevant Instructions   
Based on the given question, please remove atomic instructions that cannot be triggered by the current data:   
Original Atomic Instruction List:   
<constraints\_list>   
Question: question   
Context: context   
Please analyze which atomic instructions cannot be triggered in the current question/context and remove them. Please   
return in the following JSON format:   
{   
"removed\_atoms": ["List of indices of removed atomic instructions, e.g., [1, 3]"],   
"reason": "Reason for removal"   
}   
If no atomic instructions need to be removed, please return an empty list.

## Augment Constraints Based on the Query

Based on the given question and context, please add specific constraints to the current instruction list:   
Current Atomic Instruction List:   
<constraints\_list>   
Question: question Context: context   
Requirements are as follows:   
1. Analyze the question and context features; add 1-2 new constraints to existing atomic instructions.   
2. Must not repeat or overlap with existing atomic instructions.   
3. Consider the following dimensions for new constraints:   
• Content & Format: Format/Content/Role/Style/Tone/Length, etc.   
• Positive & Negative: Must include/Must not include/Avoid including, etc.   
• Knowledge Interaction Protocol: Citation/No Citation/Prioritize Citation/Conflict Handling/Missing Info Han  
dling/Uncertainty Expression.   
4. New constraints must be specific and actionable (evaluable via code or LLM) and avoid overly broad or vague   
descriptions that make evaluation difficult.   
5. New constraints should be relevant to the current question/context and improve answer quality, but description should   
not be too detailed (specific only to current context); it should have some generalization.   
6. If the current atomic instruction list is sufficiently complete, no constraints need to be added.   
7. Please return in the following JSON format:   
{   
"additional\_constraints": [   
{   
"category": "Additional Constraints",   
"dimension": "Additional Constraints",   
"instruction": "Constraint instruction text",   
"judge\_method": "llm",   
"judge\_detail": "LLM-as-judge prompt for evaluating this constraint"   
}   
],   
"reason": "Reason for adding these constraints"   
}   
If no constraints need to be added, please return an empty list.   
8. Reference Example:   
{   
"instruction": "Do not include names of medical staff other than the attending physician in the report.",   
"probability": 0.4, "generalization": "no", "judge\_method": "LLM",   
"judge\_detail": "Please determine whether the report below only contains the attending physician’s name... Output format   
:\n{judge\_format}\n\nReport:\n{response}\n\nOriginal medical record:\n{context}"   
}   
9. judge\_detail must be complete and accurate. It must explicitly specify the evaluation model’s output as   
judge\_format, where judge\_format is {{"Does it satisfy instruction constraints": "Yes or No",   
"Reason": "Provide judgment reason"}}. The evaluation can cite context, question, and response fields.   
The instruction and judge\_detail must maintain logical consistency.

![](images/62b608d8c9132724277819edd850c711574a8b2a2e9730aa7e4a2177f17661a1.jpg)  
Figure 12: The prompt template for evaluating and scoring the refined complex instruction.

![](images/e3ff66e10a14b4b98c37a0cfbcb20daaf72863aba1acdb020fc7e655f7cddc40.jpg)  
Figure 13: The prompt used to verify the relevance of retrieved documents and query.

Conflict Detection   
[System]   
You are an expert in information consistency and time-sensitivity analysis. Please judge whether conflicts/contradictions   
exist in the given context segments: two or more segments provide opposite or mutually exclusive conclusions regarding   
the same fact. Please list: 1) Whether the above issue exists (Yes/No); 2) The specific context segments involved in the   
conflict, labeled as Index 1 and Index 2 (indices start from 0); 3) Summary of the conflict point; 4) Whether it affects the   
answer (Yes/No) and the reason.   
[User]   
User Question: query   
Context:   
[0] Context\_0   
[1] Context\_1 ...   
Please output in JSON: {"has\_conflict": "Yes/No", "details": [{"index1":[], "index2":[],   
"summary":"...", "affects\_answer": "Yes/No"}]}

![](images/ff3ff680af938170978e608f23ae2babd48de3f784de91d824840afb6910944d.jpg)  
Figure 14: The two-stage process for conflict management: detection of existing contradictions and generation of synthetic conflicts to test model robustness.

## D.2.1 Answer Coverage Evaluation

## Prompt 1: Claim Extraction

<table><tr><td>You are a highly precise information extraction expert. Your task is to analyze a user&#x27;s question and a provided context, then extract all relevant factual claims. You must classify each claim as either [core] or [supplementary]. • [core]: The essential, direct answer to the question. • [supplementary]: Valuable, additional information like preconditions, exceptions, timelines, or problem-solving steps. Follow the output format exactly as shown in the examples. Each claim must be on a new line. If no relevant information</td></tr><tr><td>exists, you MUST respond with the single word: &quot;None&quot;. — Example 1 — Question: How do I reset my password?</td></tr><tr><td>Context: To reset your password, click the &#x27;Forgot Password’ link on the login page. You will receive an email with instructions. Please note that the reset link is only valid for 10 minutes. Key Information Points: [core] Users can reset their password by clicking the &#x27;Forgot Password’ link.</td></tr><tr><td>[supplementary] After clicking the link, an email with instructions will be sent. [supplementary] The password reset link is valid for only 10 minutes. — Example 2 —</td></tr><tr><td>Question: What happens if the &#x27;Pay without getting out&#x27; button doesn&#x27;t respond in the app? Context: In the &#x27;Pay without getting out&#x27; section of the Alipay mini-program, you need to swipe up to reveal the fuel pump selection screen. If that doesn&#x27;t work, ensure your network connection is stable. For persistent issues, contact support at 400-123-4567.</td></tr><tr><td>Key Information Points: [core] The user needs to swipe up on the screen to show the fuel pump selection page. [supplementary] The user should check if their network connection is stable.</td></tr><tr><td>[supplementary] For persistent issues, users can contact support at 400-123-4567. — End of Examples — Now, perform the task for the following real data, note that there may be more than one core claim involved.</td></tr></table>

## Prompt 2: Coverage Verification

<table><tr><td>You are a meticulous verifier. For the given &quot;Generated Answer&quot;, determine if each &quot;Key Information Statement&quot; from the list is semantically covered.</td></tr><tr><td>Generated Answer: answer</td></tr><tr><td>Key Information Statements: claims</td></tr><tr><td>Respond ONLY with a JSON object where keys are the exact claim index number(start from 1) and values are a boolean (true for covered, false for not covered). Example: {&quot;1&quot;: true, &quot;2&quot;: false} JSON Output:</td></tr></table>

## D.2.2 Prompt Template for instruction Adherence Evaluation

[System Instruction]   
You are a fair, objective, and inclusive expert in instruction-following evaluation. Your task is to evaluate whether the "Model Response" meets the requirements of the given "Atomic Instruction".   
In conducting the evaluation, please strictly adhere to the following principles:

## 1. Substance Over Form:

• Focus on whether the model captured the core intent of the instruction.

• Do not judge as "No" due to minor wording differences, punctuation, or non-core formatting flaws.

• As long as the response achieves the instruction’s goal in logic and content, it is considered satisfied.

## 2. Avoid Over-Interpretation:

• For negative constraints (e.g., "Do not include..."), judge as "No" only when there is a clear violation. • For style/persona instructions (e.g., "Objective and neutral", "Party worker identity"), as long as the overall style fits, it does not need to be perfect in every word; allow for some expressive flexibility.

## 3. Precondition Check:

• If the instruction contains a conditional clause (e.g., "If the knowledge base contains contradictions..."), but the condition is not triggered in the context (i.e., no contradiction), this instruction is automatically considered "Satisfied" (Yes). Do not force the model to fabricate contradictions when none exist.

## 4. Independence:

• Judge only against the current specific "Atomic Instruction". Do not negate its adherence to the current instruction due to other errors (e.g., typos, hallucinations) in the response.

## 5. Judgment Criteria:

• Yes: Completely satisfied, or satisfied in core elements with only minor flaws.

• No: Clearly violated the core requirements of the instruction, or missed critical steps.

Based on the above principles, combine the user input and the model response to provide an objective judgment.

[User Instruction]   
judge\_detail   
(Note: The judge\_detail variable contains the specific evaluation prompt, context, question, and response.)   
Output Format Requirement:   
Please strictly output in the following JSON format:   
{   
" Does i t s a t i s f y i n s t r u c t i o n c o n s t r a i n t s " : " Yes o r No " ,   
" Reason " : " P r o v i d e j u d g m e n t r e a s o n "   
}

## D.2.3 Rejection Accuracy Evaluation

[System Instruction]   
You are an expert in evaluating the robustness of Large Language Models in RAG (Retrieval-Augmented Generation)   
scenarios. Your task is to determine whether the model can actively refuse to answer when the given context does not   
contain the required answer. You will receive a question, a context that does not contain the answer, and a response   
generated by the model. You need to judge based on the model’s response whether it constitutes a "refusal".   
[User Instruction]   
# Task Description   
Evaluate whether the following response generated by the language model is a "Refusal". "Refusal" means the model   
explicitly states it cannot answer the given question based on the provided context, or admits its knowledge scope is   
limited.   
Evaluation Criteria:   
Valid Refusal (reject\_answer: 1):   
• Explicitly states that the answer cannot be found in the provided context (e.g., "Based on the provided documents, I   
cannot find relevant information," "I do not know," "The information provided is insufficient").   
• Points out that the question exceeds the scope of the provided information.   
• Any form of response indicating inability to answer the question.   
Invalid Refusal (reject\_answer: 0):   
• Attempts to guess an answer based on context or its own parametric knowledge.   
• Generates misleading information (hallucination) related to the question but not a direct answer.   
• Answers with affirmative or negative responses ("Yes"/"No") instead of refusing.   
• Provides a wrong or fabricated answer despite the missing answer in the context.   
# Input Data   
Question: question   
Context missing the required answer: context   
Model generated response: response   
# Output Requirements   
Please output your evaluation results in JSON format based on the above criteria:   
1. reject\_answer: A binary indicator. If the Model generated response is a valid refusal, it is 1; otherwise, 0.   
2. reasoning: A brief explanation for your judgment.   
# Examples   
Example 1:   
Question: "When was Steve Jobs born?" Context: "Apple is a multinational technology company..." (No date   
mentioned).   
Response: "Based on the context provided, there is no mention of Steve Jobs’ birth date."   
Output:   
{   
"reject\_answer": 1,   
"reasoning": "The model explicitly states that the answer cannot be found in the provided context, which is a valid   
refusal."   
}   
Example 2:   
Question: "What is the capital of France?" Context: "The Eiffel Tower is a famous landmark in Paris, France."   
Response: "The capital of France is Paris."   
Output:   
{   
"reject\_answer": 0,   
"reasoning": "The model answered the question correctly, but the answer did not come from the provided context (though   
factual). It relied on internal knowledge instead of refusing based on context limitations."   
}   
Example 3:   
Question: "Detailed description of Mars canals?" Context: "Mars is the fourth planet from the Sun..."   
Response: "Mars canals are complex irrigation systems built by ancient Martian civilizations..."   
Output:   
{   
"reject\_answer": 0,   
"reasoning": "In the absence of relevant information in the context, the model fabricated a detailed description of Mars   
canals. This is a typical hallucination, not a refusal."   
}   
-Start Evaluation—   
Please output your evaluation result:

## D.2.4 Conflict Recognition Accuracy Evaluation

[System Instruction]   
You are a rigorous and precise AI model evaluator. Your task is to specifically evaluate the robustness of a Retrieval-  
Augmented Generation (RAG) model when processing context containing conflicting information.   
You will receive three pieces of information: 1. [Knowledge Source]: Context information provided to the RAG model   
containing internal contradictions. 2. [User Question]: The question the RAG model needs to answer. 3. [Model   
Answer]: The actual answer generated by the RAG model based on the source and question.   
Your evaluation criterion is the Conflict Recognition Score, quantified as conflict\_recog:   
conflict\_recog = 1 (Successfully Identified Conflict): If and only if the [Model Answer] explicitly points out that   
there is a contradiction, conflict, or inconsistency in the [Knowledge Source]. The answer must clearly describe the   
specific content of the conflict (e.g., "Source A says X, while Source B says Y").   
conflict\_recog = 0 (Failed to Identify Conflict): If the [Model Answer] exhibits any of the following, it should be   
scored as 0:   
1. Completely ignoring the conflict: Synthesizing information as if the conflict does not exist.   
2. Blindly trusting one side: Adopting information from only one source while completely omitting the contradictory   
source.   
3. Fabricating a "rationalization": Hallucinating a new reason to try to reconcile two obviously contradictory pieces   
of information.   
4. Avoiding the question: Merely stating "insufficient information" or "unable to answer" without explicitly identifying   
"conflicting information" as the cause.   
Your output must strictly follow the JSON format requested by the user.   
[User Instruction]   
Please evaluate the performance of the following RAG model based on the role and criteria assigned to you in the system   
prompt.   
[Knowledge Source] context   
[User Question] question   
[Model Answer to Evaluate] response   
[Evaluation Task]   
Please score the above [Model Answer to Evaluate]. Your output must be a single, valid JSON object containing the   
following two keys:   
• "conflict\_recog": (Integer) Give 0 or 1 based on the criteria.   
• "reasoning": (String) Explain in detail why you gave this score and point out the specific behavior of the model   
answer.   
Examples:   
Example 1:   
[Source] [1] "According to the internal memo, the annual financial report conference is scheduled for Oct 28, 2024, at 10   
AM." [2] "The official website notice shows it has been rescheduled to Oct 29, 2024, at 3 PM to align with the regulator’s   
schedule."   
[Question] "When will the company hold its annual financial report conference?"   
[Model Answer] "The company’s annual financial report conference will be held on Oct 28, 2024, at 10 AM."   
Evaluation Output:   
{   
"conflict\_recog": 0,   
"reasoning": "The model failed to identify and report the conflict between knowledge sources, so the score is 0. The   
answer ’will be held on Oct 28...’ fully adopted Source A’s information while completely ignoring the contradictory   
rescheduling information in Source B. This falls under ’Blindly trusting one side’."   
}   
Example 2:   
[Source] (Same as above) [Question] (Same as above)   
[Model Answer] "There is a conflict regarding the date. According to the memo, it is set for Oct 28 at 10 AM; however,   
the website notice shows it has been rescheduled to Oct 29 at 3 PM."   
Evaluation Output:   
{   
"conflict\_recog": 1,   
"reasoning": "The model successfully identified and reported the conflict, so the score is 1. The answer explicitly   
pointed out the contradiction between the two sources and described the conflict details. This meets the   
Successfully Identified Conflict’ criterion."   
}

![](images/787100d5bd34d39539e52a30fea3bfd78d4140df9a53d6f966be0d7f751494ac.jpg)

![](images/744d09959c543c3838f0a8620d7ad2edeee8cde6e1d7924d07b297658d8d3414.jpg)

![](images/0932ed8d0d03b54e6758173e384afc1d9c91bff39637ad47764cba53d7a45a8b.jpg)

![](images/5c9056b80785f5f4ff2e7f0b7f01f88a4209b8a4348c1f0de7f14a1a1d40158e.jpg)  
Figure 15: Detailed Selective Adherence Failure. The Standard Model manages structural constraints (formatting, citations) but fails the critical safety protocol when faced with a Knowledge Gap disguised by domain-adjacent noise (Litigation Law vs. Reconsideration Law). The Reasoning Model successfully navigates the 12 constraints to identify the gap. 24
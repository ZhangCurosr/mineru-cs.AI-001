# RLMOpt: Adaptive Prompt Optimization via Recursive Language Models

Subhash Bangalore Satheesha Autonomize AI subhash.satheesha@autonomize.ai

Deepthi Duddempudi Autonomize AI deepthi.duddempudi@autonomize.ai

Nirvik Pande Carnegie Mellon University nrpande@cmu.edu

Bharath Dandala Autonomize AI bharath.dandala@autonomize.ai

## Abstract

Prompt optimizers automate the search for prompts that improve language-model performance, but existing methods rely on a predefined optimization procedure: the algorithm determines which candidates to explore and how the search progresses, while the language model generates or refines prompt proposals. We introduce RLMOpt, a prompt optimizer that makes the search policy itself language-modeldriven through a recursive language model (RLM). The RLM agent operates over a tool-based environment, inspecting task information, analyzing failures, generating candidates, allocating evaluation budget, and deciding when to stop. A deterministic harness complements the agent by enforcing objective scoring, Pareto-based selection, and regression constraints.

We evaluate RLMOpt across four benchmarks spanning structured clinical information extraction (Chia), multi-hop question answering (HotpotQA), verifiable instruction following (IFBench-2025), and multi-turn tool-calling agents (BFCL). In a matched comparison at a single seed, RLMOpt obtains the best held-out score on all four benchmarks and leads the four-task mean (0.610 against 0.589 for GEPA). Repeating each benchmark across seeds yields 11 matched benchmark– seed comparisons, in which RLMOpt outperforms GEPA in 9 cases. Across all 11 runs, it never produced a prompt that underperformed its seed, whereas GEPA fell below its starting point twice. It is also more efficient, achieving these results with fewer search rollouts while producing prompts that are 27–79% the size of those produced by GEPA.

Our results further show that optimization gains are determined primarily by the headroom available in the seed prompt, rather than by the search budget. Efficient optimization therefore depends on reaching the available headroom reliably and with minimal search.

## 1 Introduction

Modern AI systems increasingly rely on multi-stage language model pipelines, where multiple language-model calls cooperate to answer questions, retrieve information, extract structured data, follow complex instructions, or interact with external tools. The performance of these systems depends strongly on the prompts that specify the behavior of each language-model component. However, designing effective prompts by hand is labor-intensive, requires task-specific expertise, and often does not transfer across components or applications.

Prompt optimization addresses this challenge by automatically searching for prompt formulations that improve task performance. Given example inputs and an evaluation signal, prompt optimizers iteratively generate candidate prompts, evaluate their behavior, and retain higher-performing variants. Recent methods including GEPA [1], MIPROv2 [2], and OPRO [3] have demonstrated strong performance across a range of benchmarks.

Despite differences in their optimization algorithms, existing prompt optimizers share a common design: the search policy is specified by a predefined algorithm. GEPA evolves prompts through reflective mutation and Pareto-based selection, MIPROv2 uses Bayesian optimization to explore instructions and demonstrations, and OPRO uses a language model guided by previous prompt-score histories within a fixed optimization loop. While these methods already incorporate task feedback, the higher-level search policy remains pre-defined: the optimizer determines in advance how candidates are generated, which evaluations are performed, and how progress is measured.

We investigate a different design point: making the search policy itself adaptive. Instead of using a language model only to propose prompt candidates, we ask whether the language model can control the optimization process itself: deciding what evidence to inspect, which hypotheses to test, which candidates to refine, and when further exploration is unlikely to improve performance.

We introduce RLMOpt, an adaptive prompt optimizer based on recursive language models (RLMs) [12]. RLMs provide a framework in which a language model operates over a programmatic environment and can recursively invoke sub-models. In RLMOpt, the RLM acts as the optimization policy. It interacts with a tool-based environment to inspect task information, analyze failures, synthesize candidate prompts, and allocate evaluation budget. The optimizer is therefore not a fixed search procedure executed identically across tasks, but an adaptive agent that determines its own optimization trajectory.

Adaptive search introduces a new challenge: optimization decisions must remain reliable when evaluation data is limited and candidate improvements are noisy. To address this, RLMOpt separates decision-making from enforcement. The RLM agent controls exploration, while a deterministic harness owns everything that should not depend on a language-model decision: it executes the task LM, computes all scores, and enforces Pareto-based candidate selection and regression constraints. This separation allows the optimizer to explore flexibly while preserving guarantees that are difficult for a language model to maintain consistently.

We evaluate RLMOpt on four benchmarks spanning structured clinical information extraction (Chia [24]), multi-hop question answering (HotpotQA [22]), verifiable instruction following (IFBench 2025 [25]), and multi-turn tool-calling agents (BFCL [20]). Across these settings, RLMOpt achieves the best held-out score on every benchmark and improves the four-task mean over GEPA while using fewer search rollouts. We further analyze when adaptive optimization is most useful (Section 6), showing that improvements are bounded by the headroom available from the initial prompt and that near-ceiling tasks leave little room for any optimizer to improve.

## Contributions.

1. We introduce RLMOpt, a prompt optimizer whose outer search procedure is controlled by an RLM agent rather than a fixed optimization algorithm. The agent performs adaptive exploration through a tool interface, while a deterministic harness enforces evaluation and selection constraints.

2. We develop a harness-controlled optimization framework for reliable adaptive search under limited evaluation data, including per-field scoring, Pareto-based selection, and regression constraints that prevent candidate updates from sacrificing existing capabilities.

3. We demonstrate improved optimization efficiency across four benchmarks, where RLMOpt achieves the strongest held-out performance while requiring fewer downstream evaluations than existing prompt optimizers.

4. We characterize when prompt optimization provides value by analyzing the relationship between initial prompt quality and achievable improvement, showing that optimizer gains are determined by the remaining performance headroom of the underlying model.

5. We extend adaptive prompt optimization to multi-component agent systems (Section 3.7), where the optimized object includes tool-use behavior and interaction trajectories rather than only single-response prompts.

## 2 Related Work

Prompt optimization. Automatic prompt optimization methods search over natural-language instructions and demonstrations to improve downstream task performance. Early approaches such as APE [4] generate candidate instructions from demonstrations and select high-performing variants through evaluation. Subsequent methods have introduced richer optimization procedures: OPRO [3] uses language-model proposals conditioned on previous prompt-score histories, MIPROv2 [2], implemented in DSPy [11], jointly optimizes instructions and few-shot demonstrations through Bayesian search, and GEPA [1] uses reflective mutation with Pareto-based selection over prompt candidates. TextGrad [5] generalizes the update itself, treating natural-language critique as a textual gradient that propagates through a graph of language-model calls. Other evolutionary approaches, including EvoPrompt [7] and Promptbreeder [6], explore genetic formulations in which language models generate mutation and crossover operations.

These methods demonstrate that language models can effectively participate in prompt search, particularly through reflection and candidate generation. Our focus is complementary: rather than improving the candidate-generation mechanism inside a predefined optimizer, we study a setting where the optimization controller itself is language-model-driven. In RLMOpt, the language model determines which information to gather, which candidates to evaluate, and how to allocate search effort.

Language-model agents and recursive language models. Language-model agents such as Re-Act [18], Reflexion [17], and Voyager [19] use language models as controllers for multi-step reasoning, reflection, and tool interaction. Recursive language models (RLMs) [12] extend this paradigm by allowing a language model to operate over a programmatic environment and recursively invoke sub-models. A related line lets a model design the procedure itself: ADAS [10] has a metaagent program successively better agents in code. RLMOpt applies this agentic control paradigm to prompt optimization, using an RLM as the search procedure while maintaining deterministic harness-side evaluation and selection.

Related optimization frameworks. Several systems optimize aspects of AI behavior beyond prompt text. Prompt tuning [9] learns continuous prompt representations, while reinforcementlearning approaches such as RLPrompt [8] and RLHF-style methods [15, 16], together with policygradient training such as GRPO [14], optimize model parameters or policies rather than naturallanguage prompts. In contrast, RLMOpt operates entirely in text space, preserving prompt interpretability and portability across language models.

Concurrent work such as SkillOpt [13] also explores iterative optimization of natural-language artifacts for agents. While SkillOpt improves agent skills through validation-guided edits within a predefined optimization procedure, RLMOpt focuses on adapting the optimization procedure itself.

Moving search control to a language model raises a question none of these methods have to answer: what prevents an adaptive optimizer from overfitting a small validation set or trading one output field against another? The next section describes the division of labor that answers it (Section 3).

## 3 Method

## 3.1 Overview

Unlike optimizers in which a fixed procedure both proposes and evaluates candidate prompts, RLMOpt separates search control from objective evaluation: an RLM agent controls the search over prompts, while a deterministic harness evaluates candidates and maintains the optimization state. The harness is a program rather than a second agent: it owns the dataset, executes the task LM, computes all scores, and enforces the selection rules, but does not choose which candidate to propose next.

Formally, let p denote a prompt, θ the task LM, and $m ( \theta ( p , x _ { i } ) , y _ { i } )$ the evaluation metric for example $( x _ { i } , y _ { i } )$ . The optimization objective is

$$
\operatorname* { m a x } _ { p } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } m ( \theta ( p , x _ { i } ) , y _ { i } ) ,\tag{1}
$$

starting from a seed prompt p<sub>0</sub> and subject to a search budget B of task-LM candidate evaluations.

The seed evaluation establishes the initial validation baseline and is not counted against the agent’s search budget. The agent reaches the task only through the harness, so the selection rules of Section 3.4 apply to every candidate regardless of how the agent proposes it. The fixed skill prompt that conditions the agent is shown in Appendix A, and the complete tool interface is documented in Appendix D.

## 3.2 Agent-controlled search

The agent is initialized with the seed prompt, task description, available tools, search budget, and a persistent scratchpad. In our experiments, the RLM agent is gpt-5.1, while the task LM is gpt-4.1 or gpt-4o-mini, depending on the benchmark.

Following the recursive language model formulation [12], the agent acts by writing code in a REPL rather than by emitting a fixed action symbol. Each turn it writes a short program that calls one or more of the tools in Table 1 and reads back their printed output.

The agent therefore chooses both which operation to perform and when to perform it. It can inspect examples and failure traces to form a hypothesis before spending an evaluation, or evaluate a candidate first and use the resulting feedback to determine the next edit. It may also stop before exhausting the search budget (Section 3.6).

The agent composes the candidate prompts directly. Both run\_candidate and commit\_prompt receive the prompt text as an argument, so every evaluated or committed candidate is authored by the agent. Sub-LM calls through call\_subagent and the synthesis tools instead return compressed evidence, such as failure summaries, shared failure modes, or candidate rules, which the agent then decides whether and how to incorporate into a prompt. The only exception is merge\_candidates, which delegates the combination of two existing candidates into a single prompt to a sub-LM.

## 3.3 Tool interface and feedback

The harness exposes tools for task inspection, failure analysis, sub-LM synthesis, and candidate evaluation, callable from the agent’s REPL. A persistent scratchpad provides a separate mechanism for recording intermediate facts, hypotheses, and decisions across turns, since the REPL view of any single result is bounded. Table 1 summarizes the interface.

<table><tr><td>Group</td><td>Tools</td><td>Purpose</td></tr><tr><td>Introspection</td><td>describe_task, dataset_overview, peek_examples, view_example, query_examples, score_explain, list_metrics,list_components</td><td>Inspect the task, data, scoring procedure, and prompt components.</td></tr><tr><td>Failure analysis</td><td>search_traces, describe_failure_patterns, peek_failures,read_trace</td><td>Locate and inspect failures from previous evaluations.</td></tr><tr><td>Synthesis</td><td>synthesize_failures, synthesize_candidate, merge_candidates, call_subagent</td><td>Delegate analysis to a sub-LM: summarize a field&#x27;s failures or a candidate&#x27;s rollouts into a shared failure mode and a candidate rule. merge_candidates instead returns a prompt combining two candidates.</td></tr><tr><td>Evaluation</td><td>run_candidate, commit_prompt, best_so_far, remaining_budget, pareto_frontier_status</td><td>Evaluate candidates and query or update optimization state.</td></tr><tr><td>Scratchpad</td><td>scratchpad_add, scratchpad_read</td><td>Store facts, hypotheses, rules, warnings, and decisions.</td></tr></table>

Table 1: Tool interface available to the agent. Candidate evaluation is exposed through run\_candidate; the remaining tools do not consume the task-LM search budget.

A run\_candidate call returns a structured feedback record rather than a scalar score. The record contains the composite score, its per-field decomposition, mismatched fields, and any applicable judge rationale [21].

composite=0.673   
label (w=0.50): 0.500 [exact\_match=0.50]   
reason (w=0.50): 0.846 [factuality\_judge=0.85]   
MISMATCHES:   
label: expected=’positive’, got=’neutral’   
factuality\_judge: "the predicted answer captures the   
entity but inverts the sentiment polarity"

Because the feedback identifies both the failing field and the associated reason, the agent can use the evaluation result to target its next edit without requiring a separate diagnostic call. We do not treat the feedback format itself as a methodological contribution; the harness already computes the underlying evidence for objective evaluation and selection.

## 3.4 Harness-controlled evaluation and selection

The harness performs objective evaluation and candidate selection. It computes per-field metrics, combines them into the composite score, maintains candidate history, and enforces the validation constraints.

For a candidate $p ,$ let $s _ { f } ( p )$ denote its score on field $f ,$ with field weight w<sub>f</sub>. The composite score is

$$
S ( p ) = \sum _ { f } w _ { f } s _ { f } ( p ) , \qquad \sum _ { f } w _ { f } = 1 .\tag{2}
$$

When multiple output fields are present, the harness does not select solely by the composite score. It first computes the Pareto frontier under the vector of per-field scores. Candidate $p$ dominates candidate $q$ when

$$
s _ { f } ( p ) \geq s _ { f } ( q ) \quad \forall f ,\tag{3}
$$

with strict inequality for at least one field. The final candidate is selected from this frontier using the composite score.

A candidate is committed only when the agent requests a commit via commit\_prompt and the harness accepts the request. An accepted candidate replaces the current best prompt and is added to the final selection set. A rejected candidate does neither, regardless of the agent’s assessment of its quality. Acceptance is governed by two constraints.

For each field $f ,$ the candidate must remain above a specified regression floor relative to the current best. If $b _ { f }$ denotes the current best score for field $f ,$ a candidate must satisfy

$$
s _ { f } ( p ) \geq b _ { f } - f _ { \mathrm { H o o r } } , \qquad f _ { \mathrm { H o o r } } = 0 . 0 5 ,\tag{4}
$$

for every field, so an aggregate improvement cannot be obtained by sacrificing a field beyond the permitted tolerance. The second is a significance gate, since a single validation pass is noisy: the paired improvement over the running best must exceed 1.65 standard errors, a one-sided 95% threshold under the normal approximation. Both constraints are active on every run we report, and the floor is additionally enabled automatically on datasets of at most 20 records, where a single-field regression is most easily mistaken for aggregate progress. The complete eligibility, tie-breaking, small-dataset, and resampling rules are given in Appendix E.

## 3.5 Optimization procedure

Algorithm 1 summarizes the optimization procedure. The harness first evaluates the seed prompt on validation data to establish the initial baseline. The agent is then initialized with the seed and a search budget of B task-LM candidate evaluations. During the search, the agent repeatedly selects an action from the tool interface. If the action evaluates a candidate, the harness executes the task LM,

Algorithm 1 RLMOpt   
Require: seed prompt $p _ { 0 } ,$ task (θ, m, train, val, test), search budget B   
1. Evaluate $p _ { 0 }$ on validation data to establish the seed baseline.   
2. Initialize RLM agent $\mathcal { A }$ with $p _ { 0 } ,$ , tools, scratchpad, and search budget B.   
3. while $\mathcal { A }$ has not stopped and search budget remains:   
4. a ← A.NextAction()   
5. $r \gets$ HarnessExecute $: ( a )$   
6. A.Observe $: ( r )$   
7. end while   
8. Generate polish variants of the harness’s running-best candidate.   
9. Let C contain the seed, committed candidates, the claimed best, and the polish candidates.   
10. Compute per-field validation scores for every $p \in C .$   
11. F ← ParetoFrontier(C).   
12. $p ^ { \star } \gets \arg \operatorname* { m a x } _ { p \in F } S ( p )$   
13. Return the test score of $p ^ { \star } .$

computes the structured feedback, updates the optimization state, and returns the result to the agent.   
The process continues until the agent stops or the search budget is exhausted.

After the agent-controlled search, the harness evaluates the seed, committed candidates, the agent’s claimed best candidate, and a set of polish variants generated from the harness’s running-best candidate. It computes per-field validation scores for this candidate set, forms the Pareto frontier, and selects the candidate with the highest composite score on that frontier. The polish variants therefore compete under the same final selection procedure as the candidates produced during search.

The polish evaluations are part of the deterministic finalization stage and are not charged against the agent’s search budget B.

## 3.6 Termination and search limits

Agent-controlled termination. The skill prompt specifies three conditions for the agent to terminate: (i) every output field achieves a mean validation score of at least 0.85 on the most recent full evaluation; (ii) at least a preset fraction of the search budget has been consumed (80% in the configuration used for our head-to-head comparisons); and (iii) two consecutive candidates each fail to improve the composite score by at least 0.02. These conditions are stated in the prompt but are not enforced by the harness: the agent can terminate regardless of whether they hold. They therefore specify the stopping behavior requested of the agent rather than a system-level guarantee. Section 7 examines how closely the agent follows these criteria at our operating points.

Harness-enforced evaluation limits. Two additional limits apply independently of the agent’s termination decision. First, the search budget B provides a hard upper bound on task-LM candidate evaluations and therefore on agent-controlled search effort.

The second addresses a failure mode in which the agent generates a sequence of candidates whose scores are statistically indistinguishable from the running best without identifying why the current prompt fails, consuming the remaining budget without producing new information. The harness tracks consecutive evaluated candidates whose scores fall within a predefined noise band around the running best. If three such candidates are evaluated without an intervening diagnosis, the next run\_candidate call returns an error without consuming budget and directs the agent to invoke synthesize\_failures on its weakest field, which produces a sub-LM-backed analysis of that field’s failures and is not itself a candidate evaluation. A candidate producing a sufficiently large score change resets the counter, as does a successful diagnosis, and the refusal mechanism is bounded so that non-compliant behavior cannot deadlock the search. Appendix E specifies the noise thresholds, counter behavior, and refusal bound.

Once diagnosis occurs the agent is free to propose any hypothesis it chooses. The gate therefore constrains when further evaluations are permitted without determining what the agent should search for, which preserves the division of responsibility in RLMOpt: the RLM controls the search policy, while the deterministic harness controls objective evaluation and the conditions under which evaluation proceeds.

## 3.7 Multi-component candidates

Everything above treats the optimized object as a single string. A tool-using agent is not one string: it is governed by a system prompt, a description for each tool it can call, and a set of demonstrations, each editable on its own. We therefore generalize the unit the optimizer edits to a candidate, a mapping from named components to text, $c = \{ \mathtt { n a m e } _ { k } \mapsto \mathtt { t e x t } _ { k } \}$ . A single-prompt task exposes one component, system\_prompt, which recovers the setting described so far; a tool-using agent additionally exposes one tool\_description\_⟨name⟩ per tool and an optimizable demonstrations component. Retrieval pipelines are expressible the same way and are supported by the same candidate model, but we do not benchmark them here.

What matters about this generalization is how little it changes. The agent edits components independently, and the composite scoring, Pareto selection, and per-field floor of Section 3.4 apply to a component map exactly as they apply to a string, with each component scored against the fields it controls. Only two things differ: how a candidate is rendered into the downstream call, and how its output is scored.

Scoring differs for tool-using agents because their output is a \*trajectory\* rather than a single answer. A trajectory consists of a sequence of tool calls, a final answer, and, for state-mutating tasks, the resulting environment state. The harness evaluates trajectories using deterministic metrics, including tool precision and recall, argument matching, final-state equality, and tool-call JSON validity. These metrics can be augmented with an LM-based step judge that labels each step as correct, wrong\_tool, wrong\_args, or wrong\_answer. Gold trajectories are evaluated either by subset matching, which permits exploration and recovery, or by final-state matching, which prioritizes the resulting state over the specific path taken. The scorer also accounts for cases in which no tool call is required: correctly declining to act receives credit, while unnecessary tool calls are penalized. The resulting objective therefore evaluates both whether a tool should be called and how the call should be executed.

The BFCL multi-turn results use this same evaluation framework. Because the task requires coordinated behavior across multiple turns, the optimized candidate is a component map containing the tool-calling instruction together with its demonstrations, rather than a single instruction.

## 4 Experimental Setup

Benchmarks. We evaluate on four benchmarks spanning distinct failure modes:

• Chia [24] [field-level extraction]: multi-field extraction of eligibility criteria from clinicaltrial protocols (conditions, drugs, procedures, measurements, temporal and value constraints); the public corpus contains no patient data.

• HotpotQA [22] [reasoning-chain]: multi-hop question answering, where the answer requires chaining evidence across passages.

• IFBench-2025 [25] [constraint violation]: instruction following under programmatically verifiable constraints, scored by the official 2025 verifier registry of 58 constraint types (not the older IFEval suite [23]).

• BFCL multi-turn [20] [agentic tool use]: multi-turn agentic tool-calling (Berkeley Function-Calling Leaderboard v3), where the model plans and emits a sequence of tool calls across a conversation, scored by sequence match against the gold trajectory.

Each benchmark uses a fixed train, validation, and test split. The optimizer has access only to the train and validation splits, and all reported results are measured on the held-out test split.

Metrics. Each field is evaluated with a type-specific metric and contributes to the composite score. Extractive fields use graded set overlap (set\_match) and normalized span matching, constraintfollowing fields use per-constraint pass rate, and tool-calling fields use sequence matching against the gold call trajectory. For categorical fields, we use normalized token-prefix matching (label\_match) rather than exact match, allowing correct labels followed by additional explanation (e.g., “True. Because . . . ”) to receive full credit.

Models. The task LM, whose prompt is optimized and which generates all task rollouts, is gpt-4o-mini for all benchmarks except Chia. On Chia, we use the stronger gpt-4.1, as manyfield extraction saturates the smaller model’s format control before the prompt itself becomes the limiting factor. The optimizer LM, which analyzes failures and proposes candidate prompts, is gpt-5.1 for both RLMOpt and GEPA. All task rollouts use temperature=0. The complete runtime configuration is given in Table 7.

Evaluation Protocol. Within each benchmark, all three methods use the same seed (7), data splits, and task LM, yielding matched, example-level comparisons on an identical held-out test set. We use a single seed per benchmark to evaluate the methods under a controlled, fixed starting point rather than averaging over different initializations. This is particularly appropriate for RLMOpt, whose no-regression floor ensures that each run either improves upon the seed prompt or preserves its performance. The headline comparison is therefore reported at this fixed seed, with the corresponding test-set uncertainty, and Section 5.2 then repeats the same matched comparison at further seeds to test whether the ordering holds. For the 50–150-example test sets, the per-run standard error is approximately 0.03–0.05; differences within this range are treated as ties. Appendix F gives the full run configuration.

Budget and Implementation Details. We report downstream rollouts, which are candidate evaluations by the task LM, separately from total LM API calls. The latter includes candidate evaluations, the optimizer’s reasoning turns and sub-LM calls, and harness-side validation and test passes. This distinction avoids conflating search effort with total inference cost; the complete accounting is given in Section 5.3. GEPA uses its native auto="light" budget, corresponding to approximately 780– 840 rollouts, while RLMOpt uses a fixed budget of B=500 across all benchmarks. Thus, RLMOpt uses fewer rollouts without per-benchmark budget tuning.

The GEPA implementation optimizes a single-predictor program and does not directly support the BFCL multi-turn tool-calling loop. We therefore use a thin wrapper, described in Appendix F, that exposes the agentic system prompt as the optimizable instruction and evaluates each candidate using the full multi-turn rollout. This allows all three methods to be evaluated under the same benchmark tasks. For all head-to-head comparisons, we disable RLMOpt’s cross-run skill library so that no method has access to information carried over from other runs.

## 5 Results

## 5.1 Head-to-head: RLMOpt vs. GEPA

RLMOpt achieves the best held-out score on all four benchmarks and leads the four-task mean, 0.610 against GEPA’s 0.589 (Table 2).

BFCL multi-turn shows the largest lift for both methods, GEPA from 0.602 to 0.653 and RLMOpt to 0.686, and the widest margin between them (+0.033, or 1.83 paired standard errors).

Against the paired standard error of Table 2, RLMOpt’s margin over GEPA exceeds one SE on BFCL-mt (1.83) and HotpotQA (1.04) and sits inside it on Chia and IFBench-25. The comparison therefore rests on consistency rather than on any one column: RLMOpt is top on all four and leads the mean by +0.021. Its gains also transfer from validation to test. On Chia, GEPA scores higher on validation (0.636 against 0.593) but lower on test (0.562 against 0.568), the signature of a prompt fitted to the split the optimizer can see. Section 5.2 repeats the head-to-head at further seeds.

## 5.2 Robustness across seeds

We repeat the matched head-to-head across seeds, holding every other setting fixed: same split sizes, same budgets, same held-out metric, both methods re-run per seed. Table 3 reports the mean and standard deviation of the held-out score for each benchmark.

Table 4 lists the individual comparisons behind these means. RLMOpt posts the higher score in 9 of the 11, losing HotpotQA seed 13 and Chia seed 13.

Across all 11 runs RLMOpt never returns a prompt below its seed, while GEPA does twice, once on HotpotQA and once on IFBench-25. On the IFBench run the split is hard enough that GEPA

<table><tr><td rowspan="2">Method</td><td colspan="4">Held-out test score</td><td rowspan="2">Mean</td></tr><tr><td>Chiaª</td><td>HotpotQA</td><td>IFBench-25</td><td>BFCL-mt</td></tr><tr><td>Seed prompt</td><td>0.435</td><td>0.700</td><td>0.430</td><td>0.602</td><td>0.542</td></tr><tr><td>GEPA-light</td><td>0.562</td><td>0.702</td><td>0.440</td><td>0.653</td><td>0.589</td></tr><tr><td>RLMOpt (ours)</td><td>0.568</td><td>0.727</td><td>0.460</td><td>0.686</td><td>0.610</td></tr><tr><td>Δ vs. GEPA</td><td>+0.006</td><td>+0.025</td><td>+0.020</td><td>+0.033</td><td>+0.021</td></tr><tr><td>∆ vs. seed</td><td>+0.133</td><td>+0.027</td><td>+0.030</td><td>+0.084</td><td>+0.068</td></tr><tr><td>paired SE</td><td>0.014</td><td>0.024</td><td>0.051</td><td>0.019</td><td></td></tr><tr><td>Δ / paired SE</td><td>0.48</td><td>1.04</td><td>0.39</td><td>1.83</td><td></td></tr></table>

Table 2: Matched head-to-head, held-out test score (single seed per column; higher is better; best per column in bold). Both optimizers run on all four tasks, including the agentic one: we extend GEPA with a wrapper that exposes the tool-call system prompt as the optimizable instruction and scores each candidate with the real multi-turn rollout (Section 4). Because both methods are scored on the same held-out examples, the relevant dispersion is the paired standard error of the per-example differences, not each method’s marginal spread: the paired quantity removes example-difficulty variance common to both. Two of the four margins exceed one paired SE. <sup>a</sup>Chia runs on gpt-4.1; the other three on gpt-4o-mini.

<table><tr><td>Benchmark</td><td>Seed prompt</td><td>GEPA-light</td><td>RLMOpt (ours)</td></tr><tr><td>HotpotQA</td><td> $0 . 6 9 4 \pm 0 . 0 2 2$ </td><td> $0 . 6 9 3 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $\mathbf { 0 . 7 3 1 } \pm 0 . 0 1 9$ </td></tr><tr><td>IFBench-25</td><td> $0 . 3 7 3 { \scriptstyle \pm 0 . 0 7 4 }$ </td><td> $0 . 3 6 7 \pm 0 . 1 0 2$ </td><td> $\mathbf { 0 . 3 9 0 \pm 0 . 0 8 9 }$ </td></tr><tr><td>BFCL-mt</td><td> $0 . 5 8 8 \pm 0 . 0 4 0$ </td><td> $0 . 6 7 7 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 6 9 9 } \pm 0 . 0 3 0$ </td></tr><tr><td>Chiaª</td><td> $0 . 4 8 3 \pm 0 . 0 6 7$ </td><td> $\mathbf { 0 . 6 3 0 \pm 0 . 0 9 } 7$ </td><td> $0 . 6 2 2 { \scriptstyle \pm 0 . 0 7 6 }$ </td></tr><tr><td>Mean</td><td>0.535</td><td>0.592</td><td>0.611</td></tr><tr><td>Runs below seed</td><td></td><td>2</td><td>0</td></tr></table>

Table 3: Multi-seed matched head-to-head (held-out test, mean ± sd across seeds; best per row in bold). The three gpt-4o-mini benchmarks run at three seeds and Chia at two, for 11 matched runs per method. The final row counts runs in which the optimized prompt scored below the seed prompt it started from. <sup>a</sup>Chia runs on gpt-4.1; the other three on gpt-4o-mini.

<table><tr><td>Benchmark</td><td>Seed</td><td>Seed prompt</td><td>GEPA-light</td><td>RLMOpt (ours)</td><td>Better</td></tr><tr><td rowspan="3">HotpotQA</td><td>7</td><td>0.700</td><td>0.702</td><td>0.727</td><td>RLMOpt</td></tr><tr><td>13</td><td>0.712</td><td>0.744</td><td>0.715</td><td>GEPA</td></tr><tr><td>42</td><td>0.669</td><td>0.634†</td><td>0.752</td><td>RLMOpt</td></tr><tr><td rowspan="3">IFBench-25</td><td>7</td><td>0.430</td><td>0.440</td><td>0.460</td><td>RLMOpt</td></tr><tr><td>13</td><td>0.290</td><td>0.250†</td><td>0.290</td><td>RLMOpt</td></tr><tr><td>42</td><td>0.400</td><td>0.410</td><td>0.420</td><td>RLMOpt</td></tr><tr><td rowspan="3">BFCL-mt</td><td>7</td><td>0.602</td><td>0.652</td><td>0.686</td><td>RLMOpt</td></tr><tr><td>13</td><td>0.543</td><td>0.674</td><td>0.678</td><td>RLMOpt</td></tr><tr><td>42</td><td>0.619</td><td>0.706</td><td>0.733</td><td>RLMOpt</td></tr><tr><td rowspan="2">Chiaª</td><td>7</td><td>0.435</td><td>0.561</td><td>0.568</td><td>RLMOpt</td></tr><tr><td>13</td><td>0.530</td><td>0.698</td><td>0.676</td><td>GEPA</td></tr><tr><td colspan="5">Comparisons won</td></tr></table>

Table 4: Per-seed matched comparisons (held-out test; higher per row in bold). Each row is one benchmark at one seed, both methods on the same split with the same budget. <sup>†</sup>marks a run that finished below the seed prompt it started from; both belong to GEPA, and RLMOpt has none. <sup>a</sup>Chia runs on gpt-4.1; the other three on gpt-4o-mini.

ends below where it started while RLMOpt holds the seed exactly, taking the comparison without improving on it. RLMOpt leads GEPA on three of the four benchmark means; Chia is the exception,

where GEPA is ahead by 0.008. RLMOpt is also the more stable method: its HotpotQA standard deviation (0.019) is a third of GEPA’s (0.056).

## 5.3 Compute efficiency

RLMOpt reaches these scores at lower compute than GEPA: fewer downstream rollouts on every benchmark, and on three of the four fewer tokens and less wall-clock time as well (Table 5). The widest gap is on the agentic task, 1,854s against 5,344s.

These counts are downstream evaluations only and should not be read as total LM cost. A complete run also incurs task-LM calls for harness-side scoring, and separate optimizer-model calls for the agent’s own reasoning and synthesis. On Chia at seed 7, RLMOpt makes 1,556 task-LM calls in total: 1,050 (∼67%) are harness-side scoring — baseline validation and test evaluation, validationscoreboard rescoring, and the final test pass — while the 500 run\_candidate calls account for a further ∼ 32% and the remainder is sub-LM synthesis. The agent’s reasoning runs on a separate optimizer model and is accounted for in tokens rather than rollouts.

Harness-side evaluation therefore dominates task-LM traffic in this setting, and the 500 against ∼ 780–840 comparison is specifically a comparison of downstream candidate evaluations rather than of total cost. Table 5 reports rollouts, tokens, and wall-clock time separately for that reason.

Smaller budgets recover much of the score. On HotpotQA a B=200 run reaches 0.717 against the B=500 run’s 0.727, at a quarter of GEPA’s rollouts. On Chia the B=500 run is what clears GEPAlight outright, but a smaller run self-stops at 135 rollouts and reaches 0.553, within one standard error of GEPA-light’s 0.562 and at fewer tokens.

<table><tr><td>Bench</td><td>Method</td><td>Rollouts</td><td>Tokens</td><td>Wall (s)</td></tr><tr><td rowspan="3">Chia</td><td>RLMOpt (light)</td><td>135</td><td>3.11M</td><td>1,317</td></tr><tr><td>RLMOpt (heavy)</td><td>500</td><td>4.58M</td><td>1,357</td></tr><tr><td>GEPA-light</td><td>842</td><td>3.81M</td><td>1,244</td></tr><tr><td rowspan="3">HotpotQA</td><td>RLMOpt (light)</td><td>200</td><td>1.78M</td><td>717</td></tr><tr><td>RLMOpt (heavy)</td><td>500</td><td>3.51M</td><td>1,673</td></tr><tr><td>GEPA-light</td><td>784</td><td>2.53M</td><td>1,359</td></tr><tr><td rowspan="3">IFBench-25</td><td>RLMOpt (light)</td><td>200</td><td>1.01M</td><td>1,163</td></tr><tr><td>RLMOpt (heavy)</td><td>500</td><td>1.52M</td><td>1,656</td></tr><tr><td>GEPA-light</td><td>781</td><td>1.78M</td><td>1,921</td></tr><tr><td rowspan="3">BFCL-mt</td><td>RLMOpt (light)</td><td>200</td><td></td><td>1,036</td></tr><tr><td>RLMOpt (heavy)</td><td>475</td><td>5.2M</td><td>1,854</td></tr><tr><td>GEPA-light</td><td>639</td><td>7.05M</td><td>5,344</td></tr></table>

Table 5: Per-run compute (single seed), RLMOpt against GEPA-light. Bold marks the cheapest RLMOpt run that still beats GEPA-light. The rollout counts are downstream evaluations only; the text above gives the full task-LM accounting, in which harness-side scoring dominates.

## 5.4 Optimized prompt sizes

RLMOpt’s prompts are smaller than GEPA’s while scoring higher, at 27–79% of GEPA-light’s size on every benchmark (Table 6). The largest difference is on BFCL multi-turn, where the winning instruction is 5,419 characters against 19,818. Appendix B provides the seeds and optimized prompts in detail.

## 6 Analysis

## 6.1 Agent trajectories and prompt improvements

Across runs, the most common first-action class is schema introspection. Initial actions typically consist of describe\_task or peek\_examples, with the first run\_candidate call occurring only after several introspection steps. HotpotQA provides a representative example. The agent’s trajectory converges on an output-format-discipline strategy: it inspects 3–5 gold answer spans, identifies that the references favor short factual spans rather than complete sentences, and then commits a candidate with a new OUTPUT FORMAT section that explicitly enforces shortest-span extraction. Thus, the resulting prompt modification is grounded in observed reference outputs rather than in undirected candidate search.

<table><tr><td>Benchmark</td><td>GEPA-light</td><td>RLMOpt</td><td>Δ</td><td>% of GEPA</td></tr><tr><td>HotpotQA</td><td>5,043</td><td>3,961</td><td>-1,082</td><td>79%</td></tr><tr><td>IFBench-25</td><td>7,744</td><td>3,073</td><td>-4,671</td><td>40%</td></tr><tr><td>BFCL-mt</td><td>19,818</td><td>5,419</td><td>-14,399</td><td>27%</td></tr><tr><td>Chia</td><td>24,556</td><td>12,605</td><td>-11,951</td><td>51%</td></tr></table>

Table 6: Optimized-prompt size (characters, seed 7). Sizes are the instruction portion; RLMOpt additionally injects worked-example demos.

For HotpotQA, the RLMOpt-optimized prompt improves test F1 from 0.700 for the seed prompt to 0.727 (seed 7; Table 2). We reproduce the full optimized prompt in Appendix B, and Figure 1 illustrates how the agent’s tool trajectory translates into the resulting prompt structure.
<table><tr><td></td><td>#</td><td>Tool call</td><td>What the agent observed</td><td>Prompt section produced</td></tr><tr><td>n </td><td>1</td><td>describe_task</td><td>input/output schema: question, (preamble: restate task) context → answer:str</td><td></td></tr><tr><td></td><td>2</td><td>peek_examples (n = 3)</td><td>gold answers are short spans: &quot;Princeton Rays&quot;, &quot;no&quot;, “Tomas Arana&quot;, never sentences</td><td>## OUTPUT FORMAT (the key add)</td></tr><tr><td></td><td>3</td><td>score_explain</td><td>scoring is word-level F1; partial credit for substrings; strict on extra punctuation</td><td>## STRICT FIELD FORMATTING</td></tr><tr><td></td><td>4</td><td>scratchpad_add ×4</td><td>4 facts promoted to rules: short- span rule; yes/no for comparisons; no “The answer is&quot;; 2 worked exam-</td><td>## RULES, ## EXAMPLES, ## DOMAIN GUIDANCE</td></tr><tr><td></td><td>5</td><td>commit_prompt</td><td>ples</td><td>commits a strictly-better can- didate (from introspection alone; 0 run_candidate calls)</td></tr></table>

Figure 1: Optimization trajectory on HotpotQA seed 7, reconstructed from the run’s audit log. This is an illustrative (above-mean) “surgical-mode” trajectory where the agent self-stops without spending any run\_candidate budget: the four introspection tool calls generate enough structure-relevant observations to commit a strictly-better candidate against the seed. The trace-conditioned move is step 2: peeking 3 gold examples reveals that answers are short factual spans, which drives the ## OUTPUT FORMAT section of the optimized prompt (Appendix B).

## 6.2 When prompt optimization helps

Prompt optimization is typically evaluated on tasks where it produces gains, making it difficult to predict when optimization will be useful on a new task. Our results indicate that the key factor is the amount of prompt-accessible headroom left by the seed. We observe two distinct regimes: tasks where the task LM can exploit substantial remaining headroom, and tasks where the seed is already near the model’s prompting ceiling.

Headroom regime. All four headline benchmarks fall in the first regime. Their seed prompts leave room for improvement that the task LM can exploit, with the largest gains occurring on the tasks with the most headroom. Chia starts at 0.435 and improves by +0.133 to 0.568, while the agentic BFCL task improves from 0.602 to 0.686 (+0.084). HotpotQA and IFBench-2025 begin closer to the task LM’s prompting ceiling and consequently show smaller gains of +0.027 and +0.030, respectively (Table 2). A synthetic diagnostic provides a controlled test of this mechanism: when the labels can be recovered from the training data, the optimizer discovers the relevant rule and generalizes it to held-out inputs. The same diagnostic also exposes an overfitting regime in which increasing the search budget reduces held-out accuracy (Appendix C).

Ceiling regime. When the seed is already near the task LM’s prompting ceiling, additional optimization provides little opportunity for improvement. We observed this when evaluating capabilitysaturated tasks. On a mid-size model, classic IFEval-style instruction following achieved approximately 0.91 from the seed, while the single-turn BFCL split reached approximately 0.77. Neither RLMOpt nor GEPA improved these scores. In these cases, RLMOpt’s no-regression floor returned the seed prompt rather than accepting a noisy candidate with a lower score. We therefore treat these tasks as negative controls rather than headline results: they demonstrate that additional search does not create gains when prompt-accessible headroom is already exhausted.

Implications. The two regimes explain why optimization budgets can produce substantial gains on some tasks but little or no improvement on others. The relevant quantity is not the search budget itself, but the exploitable headroom remaining in the seed prompt. When a task is near the model’s prompting ceiling, increasing the optimization budget is unlikely to help; improving the underlying task model is the more promising source of additional performance.

Self-stopping. The synthetic diagnostic also reveals a limitation of the current stopping policy. The agent’s stopping point varies substantially: it used 27 and 21 rollouts in the two runs reported in Appendix C, and fewer on several other tasks. Increasing the budget cap therefore does not reliably increase the amount of search or improve held-out accuracy; in the larger-budget diagnostic run, test accuracy was lower. This variability suggests that the stopping decision remains an important source of run-to-run variation. A policy that explicitly considers remaining per-field headroom before terminating could make single runs more reliable (Section 7).

## 7 Limitations

System-level attribution. RLMOpt differs from the baselines along several dimensions simultaneously, including its adaptive search policy, composite per-field scoring, no-regression floor, and optimizable demonstrations component. Our experiments therefore establish the performance of the complete system, but do not isolate the contribution of each component. A stronger attribution study would hold the harness, budget, task LM, and optimizer LM fixed while replacing the adaptive policy with a fixed search procedure. We leave this ablation to future work. This distinction is particularly relevant for BFCL multi-turn, where RLMOpt optimizes a demonstrations component that is not exposed by our GEPA wrapper. The resulting margin should therefore be interpreted as evidence for the combined system rather than for the adaptive search policy alone.

Stopping policy. The current stopping policy remains a source of variance. The three stopping conditions described in Section 3.6 are not well calibrated to the settings evaluated here. The per-field target of 0.85 is unattainable for the composite scores observed in our benchmarks (Table 2), while the 0.02 improvement threshold is smaller than the estimated per-run standard error of 0.03–0.05 (Section 4). The budget-based condition also did not determine termination in the trajectory shown in Section 6.1. In practice, the budget cap is therefore the principal effective constraint. This contributes to substantial variation in stopping points across runs; in the synthetic diagnostic of Appendix C, increasing the budget cap even resulted in lower held-out accuracy. A better stopping policy would operate on quantities already available to the system, such as remaining per-field headroom and the statistical criterion used for commit decisions. We have not evaluated such a policy, so our reported sample-efficiency results reflect the current stopping behavior rather than an optimized termination strategy.

Evaluation scope and cost. Our evaluation covers four benchmarks and two task LMs: gpt-4.1 for Chia and gpt-4o-mini for the remaining benchmarks. All four benchmarks were selected to leave exploitable headroom, so the results do not establish how the method behaves on tasks that are already near the task LM’s prompting ceiling or on smaller open models. The observed margins over GEPA are also generally comparable to the per-run standard error (Section 5.1); consequently, the evidence for an advantage comes from consistency across benchmarks and seeds rather than from large margins on individual runs.

Our compute comparison likewise concerns downstream candidate evaluations rather than total LM cost. Harness-side scoring accounts for a substantial fraction of task-LM calls (Section 5.3), while the optimizer agent’s reasoning uses a separate, stronger model whose cost is accounted for in token usage. Finally, the multi-component extension described in Section 3.7 remains sensitive to small validation sets. In an early probe of a mixed function-calling task with approximately ten validation and eight test examples, optimization improved validation performance on every seed (+0.067 on average) but did not improve held-out performance $( \Delta \approx - 0 . 0 3 )$ . This illustrates the risk of validation overfitting when the evaluation set is small and motivates larger validation sets for future extensions.

## 8 Conclusion

RLMOpt replaces a hand-coded outer prompt-search loop with an RLM agent that serves as the search policy, backed by a deterministic harness that enforces composite per-field scoring, Pareto selection, budget constraints, and a no-regression floor. Across four benchmarks spanning clinical extraction, multi-hop question answering, verifiable instruction following, and multi-turn agentic tool calling, RLMOpt achieves the best held-out score on every benchmark and outperforms GEPA on the four-task mean while using fewer downstream rollouts. Although the per-benchmark margins are generally comparable to the per-run standard error, the matched multi-seed comparison provides stronger evidence of reliability. Across all 11 benchmark–seed comparisons, RLMOpt never produces a prompt that underperforms its seed, while GEPA does so twice.

The results also clarify when prompt optimization can be expected to help. The available improvement is determined primarily by the headroom left by the seed relative to the task LM’s prompting ceiling, rather than by the search budget alone. When a seed is already near that ceiling, additional optimiza tion yields little benefit, and a stronger target model is the more effective source of improvement. When headroom remains, the relevant objective is therefore not simply to search more, but to reach the available improvement efficiently and reliably.

This perspective shifts the goal of prompt optimization from maximizing search to making effective use of the headroom that exists. A useful optimizer should improve weak seeds when prompt-level gains are available, avoid regressions when they are not, and do so without requiring excessive search. In this sense, knowing when optimization cannot help is an important part of knowing how to optimize.

## References

[1] L. Agrawal, O. Khattab, S. Tan, et al. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv preprint arXiv:2507.19457, 2025.

[2] K. Opsahl-Ong et al. Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs. arXiv preprint arXiv:2406.11695, 2024.

[3] C. Yang et al. Large Language Models as Optimizers. arXiv preprint arXiv:2309.03409, 2023.

[4] Y. Zhou, A. I. Muresanu, Z. Han, et al. Large Language Models Are Human-Level Prompt Engineers. In ICLR, 2023.

[5] M. Yuksekgonul, F. Bianchi, J. Boen, et al. TextGrad: Automatic “Differentiation” via Text. arXiv preprint arXiv:2406.07496, 2024.

[6] C. Fernando, D. Banarse, H. Michalewski, S. Osindero, and T. Rocktäschel. Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution. arXiv preprint arXiv:2309.16797, 2023.

[7] Q. Guo, R. Wang, J. Guo, et al. Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers. In ICLR, 2024. arXiv:2309.08532.

[8] M. Deng, J. Wang, C.-P. Hsieh, et al. RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning. In EMNLP, 2022. arXiv:2205.12548.

[9] B. Lester, R. Al-Rfou, and N. Constant. The Power of Scale for Parameter-Efficient Prompt Tuning. In EMNLP, 2021. arXiv:2104.08691.

[10] S. Hu, C. Lu, and J. Clune. Automated Design of Agentic Systems. arXiv preprint arXiv:2408.08435, 2024.

[11] O. Khattab, A. Singhvi, P. Maheshwari, et al. DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines. arXiv preprint arXiv:2310.03714, 2024.

[12] A. L. Zhang, T. Kraska, and O. Khattab. Recursive Language Models. arXiv preprint arXiv:2512.24601, 2025.

[13] Y. Yang, Z. Gong, W. Huang, et al. SkillOpt: Executive Strategy for Self-Evolving Agent Skills. arXiv preprint arXiv:2605.23904, 2026 (Microsoft Research).

[14] Z. Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300, 2024.

[15] P. Christiano, J. Leike, T. B. Brown, et al. Deep Reinforcement Learning from Human Preferences. In NeurIPS, 2017.

[16] L. Ouyang, J. Wu, X. Jiang, et al. Training Language Models to Follow Instructions with Human Feedback. In NeurIPS, 2022.

[17] N. Shinn, F. Cassano, E. Berman, et al. Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv preprint arXiv:2303.11366, 2023.

[18] S. Yao, J. Zhao, D. Yu, et al. ReAct: Synergizing Reasoning and Acting in Language Models. In ICLR, 2023.

[19] G. Wang et al. Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv preprint arXiv:2305.16291, 2023.

[20] F. Yan, H. Mao, C. C.-J. Ji, et al. Berkeley Function-Calling Leaderboard. https://gorilla. cs.berkeley.edu/blogs/8\_berkeley\_function\_calling\_leaderboard.html, 2024.

[21] L. Zheng, W.-L. Chiang, Y. Sheng, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In NeurIPS Datasets and Benchmarks, 2023.

[22] Z. Yang, P. Qi, S. Zhang, et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In EMNLP, 2018.

[23] J. Zhou et al. Instruction-Following Evaluation for Large Language Models. arXiv preprint arXiv:2311.07911, 2023.

[24] F. Kury, A. Butler, C. Yuan, et al. Chia, a Large Annotated Corpus of Clinical Trial Eligibility Criteria. Scientific Data, 7(1), 2020.

[25] IFBench: Instruction Following with an Extended Verifiable-Constraint Registry. Benchmark suite (58-verifier registry), 2025.

## A The Optimizer’s Skill Prompt

The agent is conditioned by a fixed skill prompt, the analogue of a reflective optimizer’s meta-prompt, which states the search discipline it should follow and the structure every committed candidate must have. It is not modified during optimization. We excerpt the load-bearing parts below verbatim, with elisions marked [...]. The excerpt is reproduced exactly as the agent receives it, so it refers to the harness by its name in the implementation, host.

RLMOpt optimizer skill (excerpt)   
You are a prompt optimizer. Your task is to maximize a deterministic scorer   
by iteratively rewriting a ‘skill\_instructions‘ prompt for a downstream   
agent. Every candidate prompt you produce MUST follow the structure rules   
below – they are not optional.   
# Core loop (every run – do these in order)   
These are the load-bearing rules: everything below elaborates them. If a   
detailed section ever seems to conflict with this loop, follow the loop.   
1. ORIENT ONCE - FIRST PASS ONLY. Call describe task() +   
dataset\_overview() + peek\_examples() a single time each, then STOP –   
they are static. [...]   
2. INHERIT prior learning. Read describe\_task()[’known\_rules’] (rules   
promoted from earlier runs of THIS task) AND scratchpad\_read(...).   
Apply them; don’t re-derive what a prior run proved.   
3. BASELINE, then DIAGNOSE from FAILURES, not random examples. Run the   
seed once, then write your candidate from ERROR-DRIVEN evidence:   
describe\_failure\_patterns (aggregates failures across ALL evaluated   
examples – this is how you use the whole train set, not a 3-example   
skim) + peek\_failures(n=6) (the concrete failing cases with the WHY   
diagnosis). Name ONE concrete failure mode, then edit to fix it.   
4. EDIT with intent, then SCORE representatively. Make ONE targeted   
change, then run\_candidate(prompt) WITHOUT example\_ids (the host’s   
representative minibatch). NEVER score on your hand-picked failures   
that subset anti-correlates with the real full-val selection.   
5. TRUST the verdict. Build only on REAL\_GAIN. On WITHIN\_NOISE do NOT   
commit or re-run to confirm – attack a DIFFERENT failure mode.   
Bigger is NOT better.   
6. CAPTURE what worked. When a change earns a REAL\_GAIN, record the   
durable, transferable rule via scratchpad\_add(kind="rule", ...) – it   
is promoted to the cross-run library so future runs start ahead.   
# Structure rules (apply to every candidate you commit)   
\*\*Bigger is NOT better – test a simpler variant too.\*\* A longer, more   
detailed prompt often scores WORSE than a concise one, especially for   
structured or short-output tasks and smaller target models: extra   
instructions dilute and distract. Do NOT assume adding detail helps. If   
your elaborated candidates are not beating the running best, test the   
OPPOSITE direction – run\_candidate a MINIMAL variant (close to the seed,   
or shorter than your current best). Keep whichever actually scores higher   
on full val; a good optimizer that can’t improve should at least MATCH   
the seed, never only pile on text that loses to it.   
1. \*\*Open with one crisp framing sentence\*\* stating the task domain and   
the expected output shape. Do not start with "You are a helpful   
assistant" or generic preambles.   
2. \*\*Section style depends on ‘style‘.\*\* [...]   
3. \*\*Preserve every schema field name VERBATIM\*\* from the task harness’s   
‘output\_schema‘. Do not rename, alias, translate, or invent new field   
names. Same for any enum values or fixed strings. [...]   
[...]   
# Anti-hallucination rules   
- example\_ids you pass to run\_candidate or peek\_examples MUST come from   
describe\_task()[’train\_example\_ids’] or [’val\_example\_ids’]. The host   
rejects unknown IDs.   
- score values come from run\_candidate / score\_explain. Do not invent   
them.   
- commit\_prompt returns the truth about acceptance – do not claim a   
commit succeeded if the host rejected it.

```yaml
# Output
- best_prompt: the final text for the primary component
- best_components_json: JSON {name: text} of ALL final component texts
- audit_log: a list of {iteration, action, rationale}
describing your search
```

Three parts of this text bear directly on results reported in the body. Step 3 is the diagnose-fromfailures discipline whose effect is visible in the trajectories of Section 6.1; the “bigger is not better” rule is why the optimized prompts stay short (Section 5.4); and step 5 instructs the agent to treat a within-noise result as uninformative, which complements the harness-side gate of Section 3.6 rather than replacing it. The skill is advisory in all three cases: only the harness can refuse a commit.

## B Optimized Prompts (Verbatim)

This gallery reproduces, for every benchmark, the seed prompt (grey) and the RLMOpt-optimized prompt (green) in full, so the structure the optimizer adds is directly visible. The corresponding GEPA-light prompts run from 5,043 to 24,556 characters (Table 6). Long lines are wrapped to the box and text is ASCII-folded; the prompts are otherwise verbatim.

## B.1 Chia

Chia — seed (206 chars)   
Read the clinical trial eligibility criteria and extract the medical entities   
mentioned. For each entity type, list the exact text spans found in the criteria.   
Return empty lists for types with no mentions.

Chia — RLMOpt (12,605 chars)   
Extract clinical-trial eligibility entities from one criteria block as lists of   
exact text spans for six medical categories.   
## Input   
- You receive a single string field "criteria" containing one eligibility   
criterion or a short block of criteria text from a ClinicalTrials.goy protocol.   
## Output Format   
Return a Python-style JSON-compatible object with exactly these six fields, each a   
list of strings (0 or more items):   
- conditions: list[str]   
- drugs: list[str]   
- procedures:   
- Extract diagnostic or therapeutic interventions performed on the patient,   
including:   
- surgeries and operations (e.g., ’vitrectomy surgery’, ’submacular surgery’,   
’surgical intervention’).   
- tests and imaging (e.g., ’serological testing’, ’biopsy’, ’Urine ?-HCG   
pregnancy test’).   
- treatments or courses of therapy (e.g., ’progesterone therapy’,   
’treatment’).   
- Separate procedures from conditions and measurements:   
- If a phrase combines a procedure with a condition (e.g., ’biopsy-proven   
LN’),   
extract ’biopsy’ as a procedure and ’LN’ as a condition.   
- Laboratory analytes or values (e.g., ’ALT’, ’Serum creatinine’, ’2-hour   
C-peptide level’)   
belong in measurements, not procedures; only the act of testing (e.g.,   
’serological testing’)   
is a procedure.   
- Do not include general eligibility language or follow-up phrases as procedures   
(e.g., ’follow-up visit’, ’scheduled visit’) unless they clearly denote a   
specific medical   
intervention or test.   
- When multiple similar procedures are listed together, extract each as a   
separate span   
if the gold-standard patterns favor separate items (e.g., ’vitrectomy   
surgery’,   
’submacular surgery’, ’surgical intervention’).   
- measurements: list[str]   
- temporals: list[str]   
observations:

\- Capture ONLY clinical status, findings, or ongoing routines about the patient, such as:

\- symptoms or signs (e.g., ’pregnant’, ’breastfeeding’, ’acute intercurrent illness’)

\- stable care patterns or routines (e.g., ’regular bowel care routine’)

\- descriptive phrases that summarize current health state (e.g., ’good health’).

\- Do NOT include legal, administrative, or cooperation requirements (e.g., ’willing to participate’,

’able to comply with study procedures’, ’provide written informed consent’) as observations.

\- Do NOT move entities that belong to other fields into observations:

conditions remain in conditions (e.g., ’Impaired liver function’, ’MI’, ’CVA’).

clearly describes patient status.

as observations when they do not clearly denote a disease, for example:

\- ’altitude exposure’, ’current heavy smoking’, ’occupational exposure’, ’high altitude exposure’.

\- In these cases, keep the disease or risk-condition in conditions (e.g., ’respiratory disease’,

’cardiovascular disease’) and put only the exposure/state itself in observations (e.g., ’altitude exposure’).

\- Do not create observations entries for phrases that simply restate a condition name with modifiers;

those should remain in the conditions list.

\- conditions: Diagnoses, syndromes, or clinical states describing a disease or health condition (e.g., "acute coronary syndrome", "diabetes", "stroke").

\- procedures: Surgical or interventional procedures, medical operations, or therapeutic interventions (e.g., "liver transplant", "angioplasty"). - measurements: Quantitative or score-based measures, thresholds, or lab values (e.g., "LDL-cholesterol", "CrCl", "Beck’s Depression Inventory (BDI)").

\- temporals: Words or phrases indicating time, duration, or ordering of events (e.g., "prior", "history", "after pretreatment", "within 6 months").

\- observations: Non-diagnostic patient attributes, behaviors, or clinical findings that are not clearly procedures or measurements (e.g., "regular bowel care routine", "pregnant", "breastfeeding").

## ## Field-specific rules

## - conditions:

\- Extract each disease or clinical condition as its own span, even if listed in a series.

\- Include both expanded and abbreviated forms when both appear (e.g., extract "Acute coronary syndrome" and "ACS" separately).

\- Prefer condition phrases (e.g., "bleeding") over generic outcome words like "death" when they appear together, but include "death" when explicitly listed among clinical outcomes.

## - drugs:

\- Treat drug classes, specific drug names, and combination therapies as drug spans.

\- When a therapy phrase mixes drug and procedure concepts (e.g., "triple antiplatelet therapy"), extract it as drugs only if it clearly refers to medication combinations.

\- Do not duplicate abbreviations that are already part of the same phrase; extract each distinct token or phrase once.

## - procedures:

\- Extract concrete interventions and operations (e.g., "Bioresorbable Vascular Scaffold implantation", "repeat revascularization", "liver transplant").

\- Do not label generic words like "treatment" alone as procedures when no specific procedure is given.

\- When a phrase combines therapy and procedure (e.g., "triple antiplatelet therapy", "dual antiplatelet therapy"), prefer classifying it as drugs unless explicitly described as a procedure.

## - measurements:

\- Extract the name of the measurement itself, not the full inequality or numeric bound.

## - temporals:

## - observations:

## ## General guidance

\- Each output list may be empty or contain multiple spans; order does not matter for scoring.

\- Ensure that every extracted span is present verbatim in the input criteria text.

\# WORKED EXAMPLES (follow this input -> output mapping)

Example 1:   
Input: {"criteria": "Adults older than 45 and children younger than 18   
years\nPlatelet count higher than 30x109/l at time of screening\nSuspicion of   
secondary ITP\nPositive family history for ITP\nPresence or history of autoimmune   
disease as judged by the investigator\nHepatosplenomegaly\nPresence or history of   
relevant hepatic disease as judged by the investigator\nPresence or history of   
thromboembolic disease as judged by the investigator\nPatients with   
splenectomy\nWomen who are pregnant or breast feeding\nIntention to become   
pregnant during the course of the study\nLack of safe double contraception (see   
7.1)\nAny vaccination 2 weeks prior start of the study\nDrugs with a known impact   
on the immune system or on platelet function must be recorded and an exclusion of   
the study should be discusse   
Output: {"conditions": ["secondary ITP", "autoimmune disease", "hepatic disease",   
"thromboembolic disease", "alcohol abuse", "drug abuse", "Hypersensitivity",   
"Hepatosplenomegaly"], "drugs": ["romiplostim", "eltrombopag"], "procedures":   
["splenectomy", "vaccination"], "measurements": ["Platelet count"], "temporals":   
["at time of screening", "2 weeks prior start of the study"], "observations":   
["family history for ITP"]}

## Example 2:

any other medical condition requiring the use of low-molecular weight heparin therapy)\nPolycystic ovary syndrome (PCOS) according to Rotterdam Consensus Criteria (European Society of Human Reproduction and Embryology [ESHRE]/American Society for Reproductive Medicine [ASRM], 2003)\nPoor ovarian response (POR) according to the European Society of Human Reproduction and Embryology (ESHRE) Criteria\nRIF (repeated implantation failure), defined as greater than or equals to (>=) 2 previous failed embryo transfers\nEndometriosis III-IV stage or adenomyosis\nClinically significant findings on exam or ultrasound, such as salpingitis, hydrosalp Output: {"conditions": ["systemic disease", "diabetes", "metabolic syndrome", "immunological diseases", "diagnosed thrombophilia", "porphyria", "medical condition", "Polycystic ovary syndrome (PCOS)", "Poor ovarian response (POR)", "RIF (repeated implantation failure)", "Endometriosis", "adenomyosis", "salpingitis", "hydrosalpynx", "ovarian cysts", "hypersensitivity"], "drugs": ["low-molecular weight heparin", "vaginal progesterone", "excipients", "components of the solution"], "procedures": ["exam", "ultrasound"], "measurements": ["previous failed embryo transfers"], "temporals": [], "observations": ["findings"]}

Example 3: Input: {"criteria": "Patients must have histologic proof of a malignancy suitable for radiation therapy. \nPatients must have received prior external beam radiation therapy to the region proposed for HDR brachytherapy treatment; evaluation of doses previously delivered to spinal cord/cauda equine, pelvis, and other critical structures (bowel, kidneys, rectum) will be taken into consideration. \nIf repeat irradiation would exceed any normal tissue constraint set by MSKCC Radiation Oncology Department dose constraint criteria, the patient will potentially be eligible. \nIf the total prior radiation dose to the cord or pelvis exceeds 100 Gy BED equivalent, the patient will be potentially eligible, where a total of 100 BED Gy equivalent is determined by the biological equivalent dose (BED) calculatio Output: {"conditions": ["malignancy"], "drugs": [], "procedures": ["radiation therapy", "histologic", "external beam radiation therapy", "HDR brachytherapy", "repeat irradiation"], "measurements": ["MSKCC Radiation Oncology Department dose constraint criteria", "KPS"], "temporals": ["prior"], "observations": []}

## B.2 HotpotQA

## HotpotQA — seed (139 chars)

Answer the multi-hop question using only the supporting paragraphs. Produce a short factual answer (a name, date, number, or short phrase).

## HotpotQA — RLMOpt (3,961 chars)

Answer the multi-hop question using only the information in the provided supporting paragraphs.

\- Carefully read all paragraphs in the context, not just the first one.

\- context: several Wikipedia-style paragraphs; they contain all facts needed to answer the question.

\# WORKED EXAMPLES (follow this input -> output mapping)

## Example 1:

Input: {"question": "Which 2008 French film was the third of a four-film   
adaptation?", "context": "## 2008 French Open - Boys’ Singles\nThe 2008 French   
Open - Boys’ Singles tournament was an event during the 2008 French Open tennis   
tournament. Vladimir Ignatic was the defending champion, but did not compete in   
the Juniors in this year.\n\n## Asterix at the Olympic Games (film)\nAsterix at   
the Olympic Games (French: \"Ast?rix aux Jeux Olympiques\" ) is a 2008 French   
fantasy comedy film directed by Fr?d?ric Forestier and Thomas Langmann, and   
written by Langmann, Alexandre Charlot, and Frank Magnier, based on characters   
from Ren? Goscinny and Albert Uderzo’s Ast?rix comic series. It was filmed   
primarily in Spain over the course of the year 2006.\n\n## The Beautiful   
Person\nThe Beautiful Person (Fr   
Output: "Asterix at the Olympic Games"

## Example 2:

Input: {"question": "Who rode with the cosmonaut who commanded the historic   
Voskhod 2 mission?", "context": "## Pete Conrad\nCharles \"Pete\" Conrad Jr. (June 2, 1930?- July 8, 1999), (Captain, USN), was an American NASA astronaut, naval   
officer and aviator, test pilot, and during the Apollo 12 mission became the third   
man to walk on the Moon. He set an eight-day space endurance record along with   
his Command Pilot Gordon Cooper on the Gemini 5 mission, and commanded the Gemini   
11 mission. After Apollo, he commanded the Skylab 2 mission (the first manned   
one), on which he and his crewmates repaired significant launch damage to the   
Skylab space station. For this, President Jimmy Carter awarded him the   
Congressional Space Medal of Honor in 1978.\n\n## Alexey Leonov\nAlexey   
Arkhipovich Leonov (Rus   
Output: "Alexey Leonov"

## Example 3:

Input: {"question": "What was the occupation of the person whose first serious lover was Richard Chanlaire?", "context": "## Stripper\nA stripper or exotic dancer is a person whose occupation involves performing striptease in a public adult entertainment venue such as a strip club. At times, a stripper may be hired to perform at a bachelor party or other private event.\n\n## Marshman\nThe name Marshman is a family, or surname which originated in England and either refers to an occupation - namely a person whose job it was to work the marshes or it is derived from their residency possibly of Marsham in Norfolk, or in Mersham in Kent. There is a strong settlement of the Marshman family in Wiltshire, especially near Dilton Marsh.\n\n## Roughneck\nRoughneck is a term for a person whose occupation i

Output: "composer and pianist"

## B.3 IFBench-2025

## IFBench-2025 — seed (143 chars)

Follow the user’s instruction and satisfy every explicit constraint it states (counts, formats, keywords, positions). Output only the response.

## IFBench-2025 — RLMOpt (3,073 chars)

Generate exactly the response requested by the user while strictly satisfying every explicit constraint in the instruction.

## You must:

1. Identify explicit constraints:

\- Read the instruction carefully and list every explicit constraint on counts, formats, exact phrases, keywords, ordering, and positions.

\- Treat requirements such as "output nothing", "return None", "leave blank", "do not respond", "respond only with X", or similar phrases as hard constraints on what you may output.

## 2. Handle empty or placeholder responses:

\- If the instruction requires no meaningful content (for example, "output nothing", "leave the response blank", "do not answer"), output an empty response or the exact placeholder specified (such as the single token "None"), and nothing else.

\- Do NOT add explanations, descriptions, or any extra characters around an empty or placeholder response.

\- If the instruction says to "respond only with" a specific string, output that string verbatim and do not add any other text.

## 3. Plan the response structure:

\- Decide the structure of your response so that all count constraints are satisfied exactly (no more, no fewer items, words, characters, sentences, or lines than requested).

\- Align required positions ("start with ...", "end with ...", "the nth item must be ...") with the ordering of the content you will produce.

## 4. Match formats and keywords exactly:

\- Follow any specified format precisely, including symbols, spacing, digit patterns, letter case, punctuation, and template structure.

\- Include every required keyword or phrase verbatim and exclude every forbidden keyword or phrase, respecting case sensitivity when specified.

## 5. Avoid non-response content:

\- When the instruction asks for a constructed string, list, pattern, label, or sequence, output that content directly, without explanations or topic summaries.

\- Do NOT define concepts, explain laws, or provide background information unless the instruction explicitly asks for an explanation.

## 6. Resolve apparent conflicts conservatively:

\- If constraints seem to conflict, obey the most explicit, machine-checkable constraints first (counts, exact strings, formats, positions).

\- Do not relax or ignore any stated constraint; choose the interpretation that best satisfies all explicit requirements without adding new assumptions.

## 7. Verify before responding:

\- Before you output anything, mentally check that your response satisfies all stated count, format, keyword, ordering, positional, and "no explanation" constraints.

\- If any constraint is not satisfied, adjust the response in your head and only then output it.

## Output rules:

\- Output only the final response content that the instruction asks for.

\- Do NOT explain your reasoning, restate the constraints, or add meta-commentary.

\- Do NOT add bullet points, section headers, labels, or other markers unless the user explicitly requests them.

\- Do NOT add quotes, brackets, or extra text around the response unless the instruction explicitly requires that formatting.

## B.4 BFCL multi-turn

## BFCL multi-turn — seed (215 chars)

You are given a user request and a list of available functions (with JSON argument schemas). Decide which function call(s) satisfy the request and output them. If no available function applies, output an empty list.

## BFCL multi-turn — RLMOpt (5,419 chars)

Plan the complete sequence of tool calls for a multi-turn conversation, emitting exactly one JSON array named ‘tool\_calls‘.

## ## Task

\- Read the entire conversation from start to finish, respecting turn order and conversational state.

\- Identify every explicit and implicit user request that requires using the provided tools.

\- Decompose each request into the minimal sequence of tool calls needed to satisfy it fully, in the order they should be executed.

## ## Input

\- "conversation": the full multi-turn dialogue, with user and assistant turns.

\- "functions": a JSON list describing each available tool:

\- Each function has a "name", "description", and a JSON schema for its "arguments".

## You must:

\- Use ONLY the tools whose "name" appears in the ‘functions‘ JSON; never invent new function names.

\- For each function, construct "arguments" that exactly match its schema (field names, types, and required/optional status).

\- Derive argument values strictly from the conversation and tool descriptions; do not guess values not supported by the context.

\- Respect temporal and dependency constraints: resolve prerequisites first (e.g., look up an ID before using it), then perform dependent actions.

## ## Output Format

Emit a single JSON array assigned to ‘tool\_calls‘. The top-level output must be ONLY this array, for example:

{ "name": "function\_name", "arguments": { "arg1": "value1", "arg2": 2 } },

{ "name": "another\_function", "arguments": { "flag": true } }

## Formatting rules:

\- Do not wrap the array in any additional object or fields.

\- Do not include comments, natural-language explanations, or reasoning outside the JSON.

\- Each element in the array must be a JSON object with exactly two keys:

\- "name": the function name string, exactly as given in ‘functions‘.

\- "arguments": a JSON object whose keys and value types follow that function’s argument schema.

## ## Rules

\- When choosing among similar tools, always:

\- Base the choice strictly on each function’s "description" field and arguments schema from the provided functions list.

\- Select the tool whose documented purpose most directly satisfies the user’s request; do not use a more generic or adjacent tool when a specific one exists.

\- Arguments must match the tool’s JSON schema exactly:

\- Include all required arguments with appropriate types and units.

\- Do not add extra arguments that are not defined in the schema.

\- Use the exact field names and value formats shown in the tool schema and conversation (e.g., correct symbol strings, option names, booleans true/false). - When the gold sequences show repeated or mirrored calls (e.g., calling a function twice with different arguments to compare or update state), reflect that pattern:

\- Emit multiple calls to the same tool when needed, rather than collapsing them into one approximate call.

\- Treat each tool\_call as explicitly linked to a user request or a necessary prerequisite for it:

\- Avoid speculative or redundant calls that are not supported by the conversation or tool descriptions.

## - When working with folders and files:

\- Use navigation tools like cd(folder) to move into the correct directory BEFORE invoking tools that operate on files there.

\- Use listing/inspection tools like ls() or similar to confirm file presence where appropriate.

\- Use content-reading tools like cat(file\_name) BEFORE running tools that depend on that file’s contents.

\- When performing actions that require authentication or identity:

or equivalent) BEFORE sending messages or performing actions that assume a logged-in state.

\- Do not assume you are logged in unless a prior tool\_call in the plan logs in during this conversation.

## - For multi-step goals:

\- Plan each goal as a sequence of tool\_calls where earlier calls establish prerequisites (navigation, selection, reading, authentication) and later calls perform the requested action.

\- Include intermediate "check" or "inspect" calls (such as reading current state or configuration) when the expected gold sequences show them; do not skip them to shorten the plan.

## - tool\_calls:

\- Include every required tool call needed to satisfy the user’s requests; do not omit necessary steps.

\- Avoid spurious calls: only include tools that the user’s request or the tool descriptions clearly require.

\- Preserve and reuse state across calls when appropriate (e.g., watchlists, created objects) instead of recomputing or overwriting without being asked.

\- Use the minimal sufficient sequence of calls; do not add exploratory or redundant calls.

\- When the conversation includes multiple independent tasks, include calls for each task in the single ‘tool\_calls‘ array, ordered in the sequence they should be performed.

\- Ensure argument values are precise and consistent with the request (e.g., correct quantities, symbols, IDs), and avoid changing a value unless the user explicitly requests it.

## General guidance:

\- Prefer direct calls that accomplish what the user asked, rather than indirect or unnecessary chains.

\- If the user asks to both perform an action and then report results, include

tools for the action first and then tools to retrieve or expose the result. - If information required for a tool argument is missing from the conversation and tool descriptions, omit that tool call rather than guessing.

## C Synthetic Headroom Diagnostic

To isolate the mechanism behind the headroom regime (Section 6.2) in a controlled setting, we ran a small synthetic task on which the correct answer is recoverable only from the training data. We report it here rather than in the main results because the seed prompt scores zero by design, so the before/after delta would overstate the gain one should expect from a competently written seed. The diagnostic is intended to demonstrate a mechanism, not to certify an effect size.

Task. The task maps a customer-support message to a label. Each of eight semantic intents (for example refund\_request or account\_locked) is assigned an arbitrary two-character code (K9, V3, . . . ) that carries no relationship to the message text, and each intent is expressed through eight natural phrasings, for 64 records scored by exact match on the code. Because the codes are arbitrary, the intent-to-code table cannot be guessed from the surface text; it is present only in the training pairs. The seed prompt (“Read the customer message and output the correct 2-character code. Output only the code.”) therefore scores essentially zero, not because the task is hard but because the mapping has been withheld from the prompt.

What the optimizer does. The optimizer reads the training pairs, reconstructs the intent-to-code table, and writes it into the prompt as a lookup. Because the underlying language model already generalizes surface phrasing to intent, the recovered table transfers to held-out phrasings of the same intents. In a committed run the optimized prompt reached 0.769 on the held-out test split, with validation at 0.867, using 27 rollouts. The value of the diagnostic is that it confirms the optimizer can move information from the data into the prompt and have it generalize, rather than merely memorizing the training rows.

Overfitting under a larger budget. Additional budget did not help monotonically. A separate run at a larger budget reached only 0.385 on test while overfitting validation to 0.800, and the agent’s self-stop point varied between the two runs (27 and 21 rollouts). The practical reading is that exploitable headroom, not raw budget, governs whether optimization helps, and that a larger budget cap can degrade held-out accuracy when the agent over-searches a small validation set (Section 7).

## D Optimizer Tool Interface

This appendix documents the behavior of the tool interface summarized in Table 1 (Section 3.3). The agent never sees the raw task-LM API; every action it takes is a call to one of these tools.

Introspection. describe\_task returns the task schema, the output fields, and the metric assigned to each field. dataset\_overview reports split sizes and aggregate statistics. peek\_examples and view\_example return training and validation records with gold outputs. query\_examples retrieves records matching a filter. score\_explain runs the scoring procedure on a hypothetical prediction and returns the resulting breakdown, which lets the agent learn how a metric behaves without consuming budget. list\_metrics and list\_components enumerate the metric catalog and the optimizable components of the current candidate. read\_additional\_instructions surfaces optional operator guidance.

Failure analysis. These tools form a discover, narrow, and inspect ladder over prior rollouts. describe\_failure\_patterns aggregates recurring error modes across evaluated examples, search\_traces locates rollouts matching a query, peek\_failures returns the lowest-scoring examples, and read\_trace returns a full rollout record.

Synthesis. synthesize\_failures performs sub-LM root-cause analysis over a set of failures and returns a compressed summary, which keeps the top-level agent’s context small. synthesize\_candidate does the same for the rollouts of a single candidate, returning a shared failure mode and a suggested rule rather than prompt text. call\_subagent issues a nested recursive call for hierarchical decomposition. Of this group only merge\_candidates returns a prompt: it combines two existing candidates into one. Every other candidate the agent evaluates or commits is text the agent wrote itself (Section 3.2).

Evaluation and state. run\_candidate is the only tool that consumes evaluation budget: it scores a candidate on n validation examples and returns the structured feedback of Section 3.3. commit\_prompt submits a candidate to the harness, which enforces the length and rationale guards, requires a genuine composite improvement, and blocks any commit that regresses an individual field past the floor (Appendix E). The agent may choose which parent to extend from the per-example Pareto frontier (pareto\_frontier\_status) and may query best\_so\_far and remaining\_budget, but it cannot promote a candidate itself. scratchpad\_add and scratchpad\_read maintain persistent typed notes (fact, hypothesis, rule, warning, decision) across turns.

Sealing the held-out split. Test isolation is enforced at the tool boundary rather than by instruction. dataset\_overview reports the test count but withholds test identifiers, and query\_examples and view\_example refuse test-split references outright. No prompting of the agent can therefore surface test gold for encoding into a candidate.

## E Harness Selection Rules

This appendix specifies the candidate eligibility, final selection, polish, and diagnose-gate rules used by the harness. These rules are deterministic and apply independently of the RLM agent’s search decisions.

Candidate eligibility. A candidate p is eligible for commit only if it satisfies the active per-field regression constraints, improves the composite score over the running best, and passes the length and rationale guards applied by commit\_prompt. When the per-field regression guard is active, the candidate must satisfy

$$
s _ { f } ( p ) \geq b _ { f } - f _ { \mathrm { f l o o r } }\tag{5}
$$

for every field $f ,$ where $b _ { f }$ is the current best score and $f _ { \mathrm { f l o o r } }$ is the configured field-floor tolerance. The guard is enabled automatically on datasets of at most 20 records and can be enabled explicitly on larger ones; it is enabled for every run reported in this paper, whose datasets are all larger than that threshold.

Candidate improvements are additionally subject to the significance gate described in Section 3.4. The paired improvement over the running best must exceed 1.65 standard errors. Candidates that do not satisfy the eligibility conditions are not committed and do not become the running best.

Final selection. After the agent-controlled search terminates, the harness constructs a final candidate set containing the seed prompt, every committed candidate, the agent’s claimed best candidate, and the polish variants described below. The harness computes per-field validation scores for every candidate, constructs the Pareto frontier, and selects the frontier candidate with the highest composite score.

The agent’s claimed best is therefore not accepted merely because the agent identifies it as best. It competes with the seed, committed candidates, and polish variants under the final selection procedure.

Polish stage. Before final selection, the harness generates up to five competing variants from its own running-best candidate rather than from the candidate claimed by the agent.

The first variant is a deterministic structural rewrite that re-attaches the seed’s opening task statement to the accumulated rules. This is intended to recover task framing that may have become diluted through incremental edits.

The next three variants append 3, 8, and 15 gold worked examples from the training split. The number of demonstrations is therefore selected by validation rather than fixed in advance.

The final variant is a sub-LM rewrite of the running-best candidate. It is discarded if its returned prompt is less than half the length of the source prompt, which guards against rewrites that remove accumulated instructions.

All surviving polish variants are evaluated on validation data and participate in the same final Paretofrontier selection as the candidates generated during the agent-controlled search. Polish evaluations are part of the deterministic finalization stage and do not consume the agent’s search budget B.

Diagnose gate. The harness maintains a counter of consecutive evaluated candidates whose scores fall within a predefined noise band around the running best. The noise band is defined using a symmetric 1.5 standard-error interval around the running-best score.

After two consecutive near-best evaluations, run\_candidate appends an advisory warning to its result. After three consecutive near-best evaluations without an intervening diagnosis, the next run\_candidate call returns an error, consumes no evaluation budget, and instructs the agent to invoke synthesize\_failures on its weakest field.

A candidate producing a sufficiently large score change resets the counter. A successful call to synthesize\_failures also resets the counter. Re-reading existing failures through describe\_failure\_patterns or peek\_failures does not count as diagnosis, since these calls do not synthesize a new failure analysis.

The refusal mechanism is bounded. After two consecutive refusals, one candidate evaluation is admitted regardless of the counter state, after which the gate is re-armed. This prevents a noncompliant agent from deadlocking the search.

The gate is disabled when no sub-LM is configured, since synthesize\_failures requires a sub-LM. The gate limits repeated evaluation of statistically indistinguishable candidates but does not prescribe which hypothesis the agent should test after diagnosis.

## F Reproducibility

## F.1 Environment

• Python ≥ 3.11, dependencies pinned in uv.lock (tracked under version control)

• DSPy ≥ 3.2

• Deno ≥ 1.40 (the agent’s tool layer runs in a sandboxed REPL (a Deno+Pyodide sandbox); without Deno the optimizer silently degrades to template-only mode, so the SDK constructor now hard-fails when Deno is absent on PATH)

• Models: gpt-4.1 (task LM for Chia) and gpt-4o-mini (task LM for the other benchmarks), with gpt-5.1 as the optimizer/reflection LM, all at temperature = 0, cache = False, per-call timeout 90–300 s. Base model names are pinned in the run configs; the exact API snapshots are those served by the deployment gateway at run time

## F.2 Default hyperparameters

Table 7 lists the library defaults. The head-to-head runs of Section 5 override budget\_calls as described in Section 4; every other value is used as shown.

## F.3 Seed and evaluation protocol

A single random seed (we use 7) propagates to the dataset shuffle that produces the train, validation, and held-out test splits, and to any resampling used during selection. Within each benchmark both compared methods share that seed, the same splits, and the same task LM, so the held-out test set is identical across methods and each row is a matched, example-for-example comparison. The headline comparison is reported at this fixed seed, and Section 5.2 repeats it at further seeds to test whether the ordering holds.

Splits and fields. Each benchmark is split by the shared seed into train, validation, and held-out test partitions. The held-out test sizes are 100 examples (Chia, HotpotQA) and 50 (IFBench-2025, BFCL multi-turn); the per-benchmark train and validation sizes are fixed per benchmark and held constant across methods and seeds. Chia is scored on six entity fields—conditions, drugs, procedures, measurements, temporals, and observations—weighted equally in the composite.

<table><tr><td>Parameter</td><td>Default</td><td>Notes</td></tr><tr><td>budget_calls</td><td>30</td><td>Downstream rollouts the agent can spend; head-to- heads use the heavy preset (B=500)</td></tr><tr><td>optimizer_max_iterations</td><td>40</td><td>Hard cap on REPL turns</td></tr><tr><td>seed</td><td>7</td><td>Affects split shuffle + resample RNG</td></tr><tr><td>commit_policy</td><td>no_field_regression</td><td>Reject any commit that drops a field past field_floor</td></tr><tr><td>field_floor</td><td>0.05</td><td>Max allowedper-field drop under no_field_regression</td></tr><tr><td>style</td><td>thorough</td><td>Section-structured rewrites; lean = surgical edits for strong seeds / tiny data</td></tr><tr><td>self_stop_floor_pct</td><td>0.80</td><td>Min fraction of B consumed before self-stop (0.25 in lean)</td></tr><tr><td>polish_variants</td><td>True</td><td>Run the POLISH-phase variant competition (Sec- tion 3)</td></tr><tr><td>select_significance_k</td><td>1.65</td><td>Std-error multiple a commit must clear (one-sided 95%)</td></tr><tr><td>use_skill_library</td><td>True</td><td>Cross-run learned rules; disabled for the GEPA head-to-heads</td></tr></table>

Table 7: Run-time configuration for RLMOpt. Values shown are the library defaults; the benchmark head-to-heads override budget\_calls to the heavy preset (B=500) and disable use\_skill\_library for a fair comparison against GEPA. They also set commit\_policy=no\_field\_regression explicitly, so the per-field floor is active on every bench mark run regardless of the ≤20-record auto-enable described in Section 3.4. Every value is exposed on RLMOptConfig.

## F.4 Record schema

Every benchmark run writes a record.json under experiments/gepa\_benchmark/results/ at the sub-path <method>/<bench>/seed<N>\_<model>/record.json, with the following keys (excerpt):

```json
{
"method": "rlm_opt_teacher" | "gepa_auto_light" | "baseline",
"bench": "chia" | "hotpotqa" | "ifbench2025" | "bfcl_multiturn",
"seed": 7, // single-seed protocol (Section 5.1)
"task_lm": "gpt-4.1" | "gpt-4o-mini", // gpt-4.1 for Chia, gpt-4o-mini otherwise
"rollouts_used": int, // for RLMOpt: downstream evals only
"wall_clock_s": float,
"val_score": float, "test_score": float,
"per_example_test": {example_id: float, ...}, // used by aggregator
"cost_usd_estimate": float,
"extra": {
"api_calls": int, // from len(lm.history); post-patch only
"prompt_tokens": int,
"completion_tokens": int,
"total_tokens": int,
"cached_tokens": int, // Azure auto-prefix-cache hits
"cache_hit_rate": float
}
}
```  
Every number in the paper’s tables and CIs traces to these files.
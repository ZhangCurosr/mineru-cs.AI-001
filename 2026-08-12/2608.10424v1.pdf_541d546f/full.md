# Recovering Wasted Compute in Autoresearch Agents

Au Kwok Chun<sup>1∗</sup>, Abhigyan Acherjee<sup>2∗</sup>, Amrutha Rao<sup>3∗</sup>, Zaiqian Chen<sup>4</sup>, Kazem Meidani<sup>5</sup>, C. Bayan Bruss<sup>5</sup>, Micah Goldblum<sup>1,</sup> <sup>6</sup>

## Abstract

A slew of recent works develop agents for solving research problems endto-end, a paradigm increasingly referred to as autoresearch. Such agents have inspired large industry investment, motivated by their potential to automate time-consuming human labor and customize machine learning solutions for specialized applications. In this paper, we study the modeling pipeline at the core of these autoresearch systems and identify common failure modes when they are applied to tabular datasets: (1) they waste compute resolving the same bugs over and over again; (2) they often fail to tune hyperparameters even when they have a large remaining compute budget; (3) the tree-search algorithms that power them do not explore; and (4) they perform data analysis, mimicking the humans whose data they are trained on, but do not use that analysis to make downstream decisions. We explore targeted interventions and find that a global debug consultant that shares discovered runtime constraints across all branches of the search tree, prompt- and control-level enhancements, and refined treesearch algorithms successfully recover wasted compute. Our results show that large gains in autoresearch agent performance are achievable through agentic design alone, holding the underlying language model fixed.

## 1 Introduction

Fueled by large-scale industry investment, agentic systems powered by Large Language Models (LLMs) are rapidly replacing traditional AutoML pipelines for automated data science (Jing et al., 2025; Chan et al., 2024). Increasingly, these systems are aimed at broader research loops, from generating and verifying hypotheses purely from data (Majumder et al., 2024; 2025) to running experiments and writing up findings (Lu et al., 2024), a direction known as autoresearch. No matter the goal, these systems all rest on the same core task: writing code to process data, train and evaluate models, and produce a candidate solution. Current systems execute this task unreliably: they frequently time out, waste compute, and produce suboptimal, generic solutions (Toledo et al., 2025; Liu et al., 2025; Yang et al., 2025).

In this work, we identify several recurring failure modes in leading tree-search based agentic frameworks (Jiang et al., 2025; Liu et al., 2025) when applied to tabular machine learning tasks. Each represents a distinct category of how an agent’s compute budget is wasted rather than productively spent. First, agents waste budget rediscovering known bugs. Because branches in a tree search do not share memory of past failures, parallel branches repeatedly resolve identical errors in isolation, preventing meaningful iteration (Zhang et al., 2025; Yin et al., 2024). Second, agents leave budget on the table by terminating search prematurely. Superficial convergence criteria cause them to stop after only a few valid solutions, skipping the hyperparameter tuning phase that the remaining budget could have supported. Finally, agents spend budget on unproductive search states, becoming trapped in dead-end solution paths until the budget is exhausted (Toledo et al., 2025; Zhou et al., 2024).

We show that improved agentic design can address these problems without modifying the base LLM. We introduce a suite of structural interventions to resolve them:

• Context-aware debugging: To resolve context isolation, we introduce a debug consultant that enables adaptive learning of the execution environment across the search tree. The consultant accumulates discovered bugs into a shared registry and injects constraints before each generation step, preventing redundant error correction and significantly improving efficiency.

• Budget-aware hyperparameter tuning enforcement: We identify the absence of systematic hyperparameter optimization as a primary failure mode of autonomous ML agents. To address this, we introduce prompt-level guidance and control-looplevel enforcement mechanisms that compel agents to allocate their compute budgets toward structured hyperparameter tuning. By penalizing local convergence and rewarding validation-driven search, we prevent premature termination and redirect search effort toward fine-grained exploitation.

• Thompson Sampling-enhanced backtracking: We replace random backtracking with Monte Carlo Tree Search (MCTS) using Thompson Sampling. This probabilistic approach allows the agent to intelligently navigate the solution space and escape unproductive debugging loops, resulting in a significant improvement in stability of generated solutions.

• Diagnostic - agents fail to act on injected exploratory data analysis (EDA): We evaluate whether agents incorporate analytical insights by injecting “adversarial” results of a toy exploratory data analysis directly into the context window. Our experiments reveal that current agents tend to ignore these signals, motivating the need for stricter control loops that encourage data-driven planning.

## 2 Background

Automated data science. The autoresearch systems described above are built on automated data science: the end-to-end process of solving a data science task without human intervention. The agents that carry out this process, often called machine learning engineering (MLE) agents, take a dataset and problem description, generate code to explore and process the data, train and evaluate candidate models, and return a solution ready for deployment.

Recent research has shown that LLMs alone cannot solve these open-ended problems effectively and that an agentic scaffold is necessary to guide solution generation (Nathani et al., 2025). External tools (Qin et al., 2024), execution feedback (Gehring et al., 2025), and context management (Jiang et al., 2025; Liu et al., 2025; Yang et al., 2025) have all been shown to improve performance. One of the main architectural components in leading MLE agent frameworks (Liu et al., 2025; Jiang et al., 2025) is tree search over the space of candidate programs, where each node represents a solution in the form of a codebase, and edges represent attempted improvements via code edits. At each step, the agent selects a promising node to expand, generates a revised solution, executes it, and uses the resulting validation score to guide further exploration. This iterative execution-in-the-loop design has become the dominant paradigm for agentic data science and forms the basis of the frameworks we study in this work.

Agentic scaffolds. The two primary agentic scaffolds we study are AIDE (Jiang et al., 2025) and ML-Master (Liu et al., 2025). Our choice of these agents is based on their open-source nature and their exceptional performance on MLE-bench (Chan et al., 2024). AIDE structures its search around three core components. The first is a deterministic greedy search policy π that determines at each step whether to draft a new solution from scratch, debug a buggy node, or improve a valid one. A coding operator f implements these three actions, each with specialized prompts: drafting produces an initial implementation, debugging repairs execution errors, and improving proposes a single atomic change to a working solution. Finally, a summarization operator Σ extracts concise summaries of past solutions and their scores, keeping the context manageable as the tree grows.

![](images/5c555f16235ac29dbf5833764379e0519d0c3cf41012b7ffc77875071f736d50.jpg)  
Figure 1: The debug consultant shares context across the search tree: one node’s failure becomes every node’s lesson. Crashed nodes are compressed into a shared bug index of failed and successful repairs, distilled into banned patterns and proven fixes, and injected into every subsequent generation step, so the agent learns its runtime environment once instead of rediscovering each bug per branch.

ML-Master extends this paradigm with an improved search strategy and an explicit reasoning mechanism. Its exploration module is inspired by Monte Carlo Tree Search (MCTS): nodes are selected using the Upper Confidence Bound for Trees (UCT) criterion, balancing each node’s accumulated reward against its visit count to prioritize promising but underexplored branches. Nodes are expanded via the same draft, debug, and improve actions as AIDE and assigned a reward based on code execution: a node receives a positive reward if its solution is bug-free and improves upon the best validation metric seen so far, and a negative reward otherwise. Multiple workers explore branches in parallel, with rewards backpropagated through the tree to guide subsequent selection. In place of AIDE’s summarization operator, ML-Master employs a reasoning module that embeds a curated memory of past execution results and sibling node insights directly into the reasoning component of the LLM, enabling the agent to learn across parallel exploration paths and leverage the capabilities of reasoning models.

## 3 Methodology

## 3.1 Context-aware debug consultant

In the tree-search paradigm described in Section 2, knowledge of failures remains local: in both AIDE and ML-Master, only the child spawned to debug a failed node sees the actual error. The rest of the tree therefore independently rediscovers the same deprecated API call or version mismatch, often dozens of times within a single run. We call this context isolation.

Our debug consultant (Figure 1) addresses this by maintaining a shared registry of runtime constraints—which API calls crash, which alternatives work—and propagating this knowledge to all nodes via a three-step control loop in order to allow adaptive learning of the execution environment:

Step 1: Error Compression. When a node crashes, the raw traceback is compressed into a compact record: the error type, a short signature, and the strategy that caused the failure. Raw tracebacks are verbose and vary across iterations; a short hint is sufficient for the LLM to avoid the mistake. This keeps the context window focused on solution search and ensures the registry scales gracefully (Zhang et al., 2025).

Step 2: Shared Bug Registry. Each compressed record is accumulated into a shared registry that tracks the error type, which strategies have failed, and—when another node succeeds— the strategy that worked. The system distills a concise list of banned patterns and proven fixes:

BANNED: lgb.train(..., verbose\_eval=N) → TypeError

USE: callbacks=[lgb.log\_evaluation(period=N)]

Every new node immediately inherits all entries discovered so far.

Step 3: Constraint Injection. Constraints are injected at two levels. During generation (drafting or improving), the distilled banned-pattern list is appended to the prompt, preventing the agent from repeating known-failing API calls. During debugging, the injection is more targeted: the system retrieves records relevant to the current error and provides specific failed strategies (marked “never do this”) along with any proven fixes, enabling informed repair rather than blind guessing.

Step 4: Deterministic Control Rules. In addition to the shared bug registry, the debug consultant adds deterministic rules to the execution process: execution timeouts and empty logs are treated as terminal dead ends that strictly halt the branch, rather than stochastic noise worth retrying.

## 3.2 Hyperparameter tuning interventions

We test three interventions of increasing invasiveness: a prompt-level directive, a controlloop mechanism that shapes the search reward, and their combination.

![](images/04820b158506855d0d200774840c3d35261d7cecfdc3166958a9518c8c2bf2a3.jpg)  
Figure 2: Budget dependent control-loop enforcement of hyperparameter tuning in AIDE. Each node is scored for hyperparameter-tuning quality on a 0–3 scale by an LLM judge. Buggy nodes are assigned the worst possible metric and their score is ignored; for a valid node, the score both conditions the agent’s next prompt and adjusts the metric used for node selection. The reward depends on how much budget remains: weak tuning (scores of 0 or 1) is penalized throughout, but strong tuning (2 or 3) is rewarded only in later stages of the search, so the agent explores broadly early and tunes intensively late.

The prompt-level intervention appends hyperparameter-tuning instructions to the agent context via additional\_notes.txt (for both AIDE and ML-Master), instructing it to establish a validated baseline, run cheap trials to identify the most impactful hyperparameters, then tune those around the best configuration once gains stall.

The control-loop intervention enforces tuning through the search reward rather than the prompt. After a node executes, an execution-time checker (\_score\_hyperparameter\_tuning) grades its tuning quality on a discrete {0, 1, 2, 3} scale via an LLM rubric (NONE, MINIMAL, MODERATE, EXTENSIVE). If the node is not buggy, this score biases which nodes the agent expands next.

How the score enters the search differs by agent: in AIDE it adjusts the validation metric and conditions subsequent improvement prompts, while in ML-Master it is folded into the reward used for UCT-based node selection $( \mathrm { e . g . , \bar { + } 0 . 2 5 \times \mathsf { h p o \_ s c o r e } } )$ . The AIDE adjustment also accounts for code diversity and parameter reuse: metri $\mathrm { c _ { a d j } } = \mathrm { m e t r i c } _ { \mathrm { b a s e } } + \dot { 0 } . 1 \times s \times$ $( r _ { \mathrm { h p o } } + r _ { \mathrm { d i v } } + r _ { \mathrm { c o r r } } )$ , where $r _ { \mathrm { h p o } } , r _ { \mathrm { d i v } } .$ , and $r _ { \mathrm { c o r r } }$ are the tuning, diversity, and reuse rewards. The scale factor $s = | \mathrm { m e t r i c } _ { \mathrm { b a s e } } |$ (or 1.0 when the base metric is below 0.01) keeps the adjustment proportional to the magnitude of the base metric. The tuning reward $r _ { \mathrm { h p o } }$ depends on how much budget remains: weak tuning is penalized throughout the search, but strong tuning is rewarded only in its later stages. (Figure 2).

The combined intervention applies both the prompt-level directive and the control-loop mechanism together. Full prompts for the directive and the LLM scorer are given in Appendix C.

## 3.3 Thompson sampling and backtracking

![](images/30bfb719f8940b8e8180fbe65cef4ddbc7e2add2e45485778a03ab6baf22c9de.jpg)  
Figure 3: Thompson Sampling with backtracking reallocates budget away from repeatedly failing branches. Each sibling node carries a Beta distribution over its quality, stored in global memory and updated from observed rewards after every execution. When the agent detects an identical error recurring down a path $( \mathsf { S } 1 \to \mathsf { S } 2 \to \mathsf { S } \bar { 3 } )$ , it backtracks to the branch point where the error first appeared (S1) and re-samples among the siblings, expanding the node with the highest draw rather than choosing at random. Budget that would otherwise be spent re-deriving the same failure is redirected toward more promising candidates.

Each of the agents we study maintains a tree of code variants, where every node is a distinct solution attempt with an associated validation metric. When the agent hits the same error repeatedly, we backtrack to the branch point where that error first appeared and reconsider its sibling nodes, rather than continuing to burn budget down a failing path. Instead of choosing at random as is default with current agents, we use Thompson Sampling (Figure 3): each sibling i carries a $\mathtt { B e t a } ( \alpha _ { i } , \beta _ { i } )$ distribution over its quality, initialized from a uniform Beta(1, 1) prior. At each selection step, we draw a sample $\mathsf { \bar { \theta } } _ { i } \sim \mathsf { B e t a } ( \alpha _ { i } , \beta _ { i } )$ from every candidate and expand the one with the highest draw, $s ^ { * } =$ arg max $\theta _ { i }$ . After the chosen node executes, we score it with a normalized reward $r \in [ 0 , 1 ]$ (0 for buggy nodes, linearly scaled by validation performance otherwise) and update its distribution:

$$
\alpha _ { \mathrm { n e w } } = \alpha _ { \mathrm { o l d } } + r\tag{1}
$$

$$
\beta _ { \mathrm { n e w } } = \beta _ { \mathrm { o l d } } + ( 1 - r )\tag{2}
$$

This procedure balances exploration (nodes with high uncertainty have wide distributions, giving them higher chances to be sampled) and exploitation (nodes with consistently good

performance accumulate higher α values, shifting their distributions rightward), allowing the agent to learn which code branches are more promising with minimal sample complexity.

## 4 Experimental setup and results

## 4.1 Experimental setup

We evaluate our approaches on nine tabular prediction tasks spanning both classification and regression, drawn from MLE-bench and additional Kaggle competitions. Tasks were selected to satisfy three criteria: (i) their release postdates the knowledge cutoff of the underlying LLM, minimizing data leakage; (ii) they are of moderate dataset size to ensure tractable experimentation; and (iii) they collectively cover diverse evaluation metrics. The nine tasks are Cirrhosis Outcome Prediction, GNSS Classification, Spaceship Titanic, Wine Quality, and Playground Series S5E3, S5E6, S5E7, S5E8, and S5E12, with links provided in Table 19 from Appendix C.5. We primarily evaluate two agent frameworks, AIDE and ML-Master, both powered by GPT-5-mini. Performance is measured using the official MLE-bench grading scripts, which compute task-specific metrics consistent with each competition (e.g., accuracy, AUC, RMSE) on a held-out test set. All scores are averaged over 10 independent runs with different random seeds; higher scores indicate better performance on all tasks except Cirrhosis Outcome Prediction, where lower is better. Following the MLE-bench medal system, a run earns a gold medal if its score places in the top 10% of the human leaderboard for that competition. All runs are executed with a fixed compute budget of 2 hours on 22 CPU cores.

Note on API models. All experiments use a single backbone, GPT-5-mini, because evaluations are prohibitively expensive: the full study (three agents, intervention and baseline conditions, nine competitions, ten seeds each) required over a thousand two-hour runs, and repeating it on a frontier model such as GPT-5.5 or Claude Opus 4.8, which cost roughly ten times more per token, would cost tens of thousands of dollars.

## 4.2 Context-aware debugging

The debug consultant design produces large and consistent gains for both agents (Table 1). For AIDE, the debug consultant nearly doubles the gold-medal count, from 22 to 38, and eliminates all 17 of the baseline’s failed runs, raising the valid-submission rate from 81% to 100%; ML-Master improves comparably, from 18 to 29 golds, and both agents gain on six of nine competitions. The improvement is largest precisely where context isolation had been most costly: on S5E3 (AIDE) and GNSS (ML-Master), where the baseline earns no medals at all, the consultant recovers a perfect 10/10. It also reaches a working solution far sooner—the median number of steps to a first valid submission drops from 6 to 0, as the consultant supplies the environment’s constraints before the agent writes its first line of code.

Mechanism. The consultant works by stopping the agent from paying for the same mistake twice. In AIDE’s search journals, redundant bug encounters fall from 46% to 7.8%, and the fraction of nodes that execute without error rises from 54.7% to 79.0%. The compute the baseline spends rediscovering known bugs is instead spent producing working code, and this is what improves final scores: seeds with more valid nodes achieve better held-out results (pooled r = +0.22 across 163 seeds). Detailed statistics, including first-attempt fix rates and time to first submission, are in Appendix A; case studies, full generated code, and per-competition correlations are in Appendices B.3 to B.5.

## 4.3 Hyperparameter tuning guidance

Adding explicit hyperparameter-tuning guidance to AIDE also produces sizable gains, improving graded scores on 7 of 9 competitions with individual effects as large as +0.388 on S5E8 and +0.218 on S5E12 (Table 2). The gains are concentrated on tasks where the baseline leaves the most room for improvement (S5E8, S5E12, Spaceship, Wine), confirming that AIDE under-invests in tuning and that a modest amount of structured guidance recovers measurable unrealized performance. On tasks where the baseline is already strong, additional tuning guidance produces little response.

(a) AIDE
<table><tr><td>Comp.</td><td>Treatment</td><td>n</td><td>Baseline</td><td>n</td></tr><tr><td>Cirrhosis†</td><td> $\mathbf { 0 . 3 8 4 \pm 0 . 0 0 8 }$ </td><td>10</td><td> $0 . 3 9 4 \pm 0 . 0 3 5$ </td><td>9</td></tr><tr><td>GNSS</td><td> $\mathbf { 0 . 9 6 7 \pm 0 . 0 0 2 }$ </td><td>10</td><td> $\mathbf { 0 . 9 6 7 \pm 0 . 0 0 2 }$ </td><td>8</td></tr><tr><td>Spaceship</td><td> $\mathbf { 0 . 8 0 6 \pm 0 . 0 1 6 }$ </td><td>10</td><td> $0 . 7 9 0 \pm 0 . 0 6 7$ </td><td>9</td></tr><tr><td>Wine</td><td> $\mathbf { 0 . 4 0 7 \pm 0 . 0 3 4 }$ </td><td>10</td><td> $0 . 3 7 5 \pm 0 . 0 5 9$ </td><td>8</td></tr><tr><td>S5E3</td><td> $\mathbf { 0 . 9 5 4 \pm 0 . 0 1 5 }$ </td><td>10</td><td> $0 . 8 8 8 \pm 0 . 0 1 0$ </td><td>9</td></tr><tr><td>S5E6</td><td> $\mathbf { 0 . 3 3 7 \pm 0 . 0 0 5 }$ </td><td>10</td><td> $0 . 3 3 3 \pm 0 . 0 0 5$ </td><td>8</td></tr><tr><td>S5E7</td><td> $\mathbf { 0 . 9 6 9 \pm 0 . 0 0 1 }$ </td><td>10</td><td> $0 . 9 6 8 \pm 0 . 0 0 0$ </td><td>9</td></tr><tr><td>S5E8</td><td> $0 . 9 6 7 \pm 0 . 0 0 9$ </td><td>10</td><td> $\mathbf { 0 . 9 6 9 \pm 0 . 0 0 1 }$ </td><td>6</td></tr><tr><td>S5E12</td><td> $0 . 7 2 5 \pm 0 . 0 0 5$ </td><td>10</td><td> $\mathbf { 0 . 7 2 7 \pm 0 . 0 0 1 }$ </td><td>7</td></tr><tr><td>W/L</td><td></td><td>6:3</td><td></td><td></td></tr><tr><td>Valid</td><td></td><td></td><td></td><td></td></tr><tr><td>Golds</td><td>90/90 38</td><td></td><td>73/90 22</td><td></td></tr></table>

(b) ML-Master
<table><tr><td>Comp.</td><td>Treatment</td><td>n</td><td>Baseline</td><td>n</td></tr><tr><td> $\mathrm { C i r r h o s i s ^ { \dagger } }$ </td><td> $\mathbf { 0 . 3 8 0 \pm 0 . 0 0 2 }$ </td><td>10</td><td> $0 . 3 8 6 \pm 0 . 0 0 6$ </td><td>10</td></tr><tr><td>GNSS</td><td> $\mathbf { 0 . 9 5 3 \pm 0 . 0 1 2 }$ </td><td>10</td><td> $0 . 8 1 3 \pm 0 . 0 0 2$ </td><td>10</td></tr><tr><td>Spaceship</td><td> $\mathbf { 0 . 8 1 3 \pm 0 . 0 0 8 }$ </td><td>10</td><td> $0 . 7 9 4 \pm 0 . 0 3 3$ </td><td>10</td></tr><tr><td>Wine</td><td> $\mathbf { 0 . 4 1 2 \pm 0 . 0 1 5 }$ </td><td>10</td><td> $\mathbf { 0 . 4 1 2 \pm 0 . 0 2 8 }$ </td><td>10</td></tr><tr><td>S5E3</td><td> $0 . 8 5 1 \pm 0 . 0 1 1$ </td><td>10</td><td> $\mathbf { 0 . 8 6 0 \mathop { \pm } 0 . 0 0 5 }$ </td><td>10</td></tr><tr><td>S5E6</td><td> $\mathbf { 0 . 3 3 6 \pm 0 . 0 0 3 }$ </td><td>10</td><td> $0 . 3 2 6 \pm 0 . 0 0 6$ </td><td>10</td></tr><tr><td>S5E7</td><td> $\mathbf { 0 . 9 6 8 \pm 0 . 0 0 1 }$ </td><td>10</td><td> $\mathbf { 0 . 9 6 8 \pm 0 . 0 0 1 }$ </td><td>10</td></tr><tr><td>S5E8</td><td> $\mathbf { 0 . 9 6 9 \pm 0 . 0 0 0 }$ </td><td>10</td><td> $0 . 9 6 6 \pm 0 . 0 0 2$ </td><td>10</td></tr><tr><td>S5E12</td><td> $\mathbf { 0 . 7 2 8 \pm 0 . 0 0 0 }$ </td><td>10</td><td> $0 . 7 1 9 \pm 0 . 0 1 1$ </td><td>10</td></tr><tr><td>W/L</td><td></td><td>6:3</td><td></td><td></td></tr><tr><td>Valid</td><td>90/90</td><td></td><td>90/90</td><td></td></tr><tr><td>Golds</td><td>29</td><td></td><td>18</td><td></td></tr></table>

Table 1: The debug consultant improves gold medal rates on both agents. Treatment vs. vanilla baseline on 9 MLE-bench competitions × 10 seeds. Scores are mean ± std; n = seeds with valid submissions. Bold = winner by mean; values tied at the displayed precision are bolded in both columns. <sup>†</sup> = lower is better.
<table><tr><td>Competition</td><td>∆Prompt</td><td>∆Code</td><td>∆P&amp;C</td></tr><tr><td>Cirrhosis†</td><td> $+ 0 . 0 3 5 \pm 0 . 0 1 2 3$ </td><td> $+ 0 . 0 2 8 \pm 0 . 0 1 1 8$ </td><td> $+ 0 . 0 2 4 \pm 0 . 0 1 1 7$ </td></tr><tr><td>GNSS</td><td> $- 0 . 0 3 6 \pm 0 . 0 0 2 1$ </td><td> $- 0 . 0 3 2 \pm 0 . 0 0 1 2$ </td><td> $- 0 . 0 3 0 \pm 0 . 0 0 1 0$ </td></tr><tr><td>Spaceship</td><td> $\mathbf { + 0 . 1 1 0 \pm 0 . 0 2 2 5 }$ </td><td> $+ 0 . 1 0 5 \pm 0 . 0 2 2 6$ </td><td> $+ 0 . 0 9 8 \pm 0 . 0 2 3 5$ </td></tr><tr><td>Wine</td><td> $\mathbf { + 0 . 1 0 0 \pm 0 . 0 2 1 9 }$ </td><td> $+ 0 . 0 4 7 \pm 0 . 0 2 7 0$ </td><td> $+ 0 . 0 8 1 \pm 0 . 0 2 6 0$ </td></tr><tr><td>S5E3</td><td> $+ 0 . 0 1 4 \pm 0 . 0 0 5 2$ </td><td> $+ 0 . 0 0 4 \pm 0 . 0 0 6 4$ </td><td> $\mathbf { + 0 . 0 1 1 } \pm 0 . 0 0 5 5$ </td></tr><tr><td>S5E6</td><td> $+ 0 . 0 6 7 \pm 0 . 0 0 2 0$ </td><td> $+ 0 . 0 6 3 \pm 0 . 0 0 3 0$ </td><td> $+ 0 . 0 6 4 \pm 0 . 0 0 4 0$ </td></tr><tr><td>S5E7</td><td> $+ 0 . 0 9 7 \pm 0 . 0 0 0 2$ </td><td> $+ 0 . 0 9 7 \pm 0 . 0 0 0 2$ </td><td> $+ 0 . 0 9 7 \pm 0 . 0 0 0 2$ </td></tr><tr><td>S5E8</td><td> $+ 0 . 3 8 4 \pm 0 . 0 0 1 7$ </td><td> $+ 0 . 3 8 7 \pm 0 . 0 0 0 3$ </td><td> $+ 0 . 3 8 8 \pm 0 . 0 0 0 5$ </td></tr><tr><td>S5E12</td><td> $+ 0 . 2 1 8 \pm 0 . 0 0 0 5$ </td><td> $+ 0 . 2 1 8 \pm 0 . 0 0 0 9$ </td><td> $+ 0 . 2 1 8 \pm 0 . 0 0 0 6$ </td></tr></table>

Table 2: Hyperparameter guidance yields large performance gains for AIDE across most competitions. Intervention effect on graded score for AIDE: $\Delta = \mu _ { \mathrm { i n t } } - \mu _ { \mathrm { b a s e } }$ (mean ± SEM of ∆). P&C = prompt and code. <sup>†</sup>Lower is better for Cirrhosis.

The same control-loop intervention that helps AIDE can degrade ML-Master. This is due to the fact that it pushes ML-Master toward an HPO implementation that crashes. ML-Master’s memory then records the crash as buggy without propagating why, resulting in the agent continually retrying variants of the same broken approach. Scaffold interventions therefore do not transfer for free: whether one helps depends on interactions between the different components comprising a scaffold. We analyze this asymmetry in detail in Appendix C.

## 4.4 Thompson sampling with backtracking for stable, more reliable search

Thompson Sampling (TS) replaces the random sibling selection used by current agents with a strategy that concentrates exploration on promising branches and backtracks out of repeatedly failing ones. Its primary effect is stability: in a controlled comparison with all other settings held fixed, TS more than halves the number of null runs at almost no cost to peak performance. Two controlled studies below, on AIDE and on MLEvolve (Du et al., 2026), attribute this gain to the selection strategy itself.

We achieve this with a path-structuring algorithm for node selection, augmented by a single new parameter: similar\_error\_backtracking\_threshold, which lets the agent backtrack to the node level that triggered the first instance of a repeated error, reclaiming budget that would otherwise be spent re-deriving the same failure. We also widen the initial drafting phase, raising the number of initial solution nodes from 5 in the baseline to 20 for TS, since TS realizes its advantage only when it has a rich pool of candidates to allocate exploration across. Full hyperparameter settings for the baseline and TS are given in Table 13; headline results are reported in Table 12, with standard deviations and null rates in Appendix C.1 (Tables 14 and 15).

Since we make multiple changes to the agentic pipeline, including introducing new hyperparameters, we conduct a controlled study to isolate the contribution of TS itself. We re-evaluate TS with the number of initial drafts and all other hyperparameters held fixed across conditions, so any observed gain is attributable to TS alone. For AIDE, we set the native max-debug-depth parameter equal to our similar-error backtracking threshold, which renders the latter inactive, and compare two head-to-head conditions: AIDE with increased drafts but no TS, and the full AIDE+TS configuration (Table 16).

AIDE with 20 initial drafts already improves over the plain setting. On top of that, TS contributes a distinct and practically important advantage. Holding all other settings identical, AIDE+TS reduces null runs from 33 to 15 of 90 relative to AIDE+more drafts (a 54.5% reduction), delivering markedly more stable outcomes. On the competitions most sensitive to exploration, TS maintains or improves scores even against the stronger draft-augmented baseline, so the added stability comes at no cost to peak performance.

To further validate TS independently of any hyperparameter changes, we ran a further experiment with MLEvolve (Du et al., 2026), one of the strongest open-source agents on MLE-bench. Here the baseline and TS variants share identical configurations throughout; only the candidate selection strategy differs. Across nine benchmark competitions, MLEvolve+TS outperforms baseline MLEvolve on 5 out of 9 competitions (Table 3) and ties a sixth, confirming that the edge is attributable to TS rather than to incidental tuning.
<table><tr><td>Competition</td><td> $\mathbf { M L E v o l v e } + \mathbf { T S }$ </td><td>MLEvolve Baseline</td><td>Winner</td></tr><tr><td>Cirrhosis†</td><td> $0 . 3 9 0 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 3 8 9 \pm 0 . 0 0 4 }$ </td><td>Baseline</td></tr><tr><td>GNSS</td><td> $\mathbf { 0 . 9 5 3 \pm 0 . 0 0 5 }$ </td><td> $0 . 9 2 8 \pm 0 . 0 1 1$ </td><td>TS</td></tr><tr><td>Spaceship</td><td> $\mathbf { 0 . 8 1 5 \pm 0 . 0 0 2 }$ </td><td> $0 . 8 1 4 \pm 0 . 0 0 3$ </td><td>TS</td></tr><tr><td>Wine</td><td> $0 . 4 0 6 \pm 0 . 0 1 2$ </td><td> ${ \bf 0 . 4 2 5 \pm 0 . 0 0 2 }$ </td><td>Baseline</td></tr><tr><td>S5E3</td><td> $\mathbf { 0 . 9 0 2 \pm 0 . 0 0 1 }$ </td><td> $0 . 8 9 9 \pm 0 . 0 0 1$ </td><td>TS</td></tr><tr><td>S5E6</td><td> $\mathbf { 0 . 3 1 3 \pm 0 . 0 0 7 }$ </td><td> $0 . 3 1 1 \pm 0 . 0 0 8$ </td><td>TS</td></tr><tr><td>S5E7</td><td> $\mathbf { 0 . 9 6 8 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 9 6 8 \pm 0 . 0 0 0 }$ </td><td>Tie</td></tr><tr><td>S5E8</td><td> $0 . 9 6 3 \pm 0 . 0 0 3$ </td><td> ${ \bf 0 . 9 6 4 } \pm { \bf 0 . 0 0 2 }$ </td><td>Baseline</td></tr><tr><td>S5E12</td><td> $\mathbf { 0 . 7 2 6 \pm 0 . 0 0 1 }$ </td><td> $0 . 7 1 1 \pm 0 . 0 0 4$ </td><td>TS</td></tr></table>

Table 3: Swapping in Thompson Sampling alone delivers MLEvolve’s largest gains. With all other settings held fixed, TS wins 5 of 9 competitions and ties a sixth (S5E7). Its two biggest-margin results, GNSS (+2.5%) and S5E12 (+1.5%), are well outside SEM, while most remaining differences in either direction fall within SEM; the one substantial exception is Wine, where the baseline wins by +1.9%. Bold indicates the winning method per competition; values tied at the displayed precision are bolded in both columns. <sup>†</sup>Lower is better.

Two mechanisms drive these gains, and they reinforce each other. A wider initial draft pool gives the agent more candidates to work with, and Thompson Sampling allocates exploration across them far more effectively than random selection, while backtracking pulls the agent out of repeatedly failing paths and returns that budget to promising ones. The AIDE and MLEvolve studies let us see each mechanism on its own: the draft increase helps by itself, and TS adds a further gain on top, holding everything else fixed. Together they explain why the full system is both more stable and stronger than either piece alone.

## 4.5 Diagnostic: current agents do not meaningfully act on exploratory data analysis

Additionally, as a diagnostic, we examine whether agent-based systems meaningfully adhere to one of the canonical stages of the machine learning pipeline: exploratory data analysis (EDA). We observe that most agents operate using a three-stage structure, namely draft(), improve(), and debug(). A recurring pattern across agents is the explicit instruction in all three phases to avoid EDA. We hypothesize that, even if this restriction were lifted, the agents would not effectively utilize insights derived from EDA. To test this hypothesis, we inject the results of a deliberately misleading and low-fidelity exploratory data analysis directly into the agent’s context window. Theoretically, if the agent incorporates this information, such adversarial signals would adversely influence its downstream decisions, for example feature selection. We would therefore expect degraded performance metrics as a consequence of these adversely impacted choices.

To test whether agents incorporate exploratory data analysis (EDA), we injected the results of a controlled, erroneous EDA into the context windows of AIDE and ML-Master and compared their performance against EDA-free baselines. Across all tasks, the performance differences induced by EDA injection were inconsistent and statistically insignificant. Using an LLM-as-a-judge framework (gpt-5-2025-8-07), we further found that agents never conducted EDA on their own in baseline runs and rarely engaged with the injected EDA: in AIDE, the agent acknowledged the malicious EDA in only 21% of cases and let it affect feature selection in just 5%. Together, these results indicate that existing agents do not meaningfully act upon or integrate EDA into downstream modeling decisions, suggesting an avenue for improving future agents. Extended details including the injection format, example messages, and the full evaluation prompts are provided in Appendix C.4.

## 5 Related works

AutoML outside of LLM agents. AutoML systems can automate algorithm selection and hyperparameter tuning. For example, Auto-sklearn (Feurer et al., 2022) and TPOT (Olson & Moore, 2016) leverage Bayesian optimization and ensemble construction. Approaches like FLAML (Wang et al., 2021) and TabPFN (Hollmann et al., 2023) focus on low-computationalcost optimization or in-context learning for tabular data, respectively. However, these systems are rigid compared to LLM agents that can implement any algorithm in principle, and they often fail to outperform simple baselines in low-data regimes (Knauer et al., 2024). Unlike these fixed-pipeline approaches, we employ LLM-driven agents to dynamically reason about data semantics and debug failures in real time.

Autoresearch & agentic data science. Recent agents for machine learning engineering have shifted from linear code generation to sophisticated tree-search methodologies (Chan et al., 2024; Wang et al., 2024b). A broader line of work pushes toward autoresearch, automating larger portions of the research loop: DataVoyager (Majumder et al., 2024) uses a role-based multi-agent architecture to explore and verify hypotheses from a dataset, DiscoveryBench (Majumder et al., 2025) benchmarks agents on this discovery task rather than on competitionstyle modeling, and systems like The AI Scientist (Lu et al., 2024) attempt the full pipeline from ideation to writeup. The frameworks we build on instead focus on the modeling stage, optimizing predictive performance through tree search: AIDE (Jiang et al., 2025), ML-Master (Liu et al., 2025), and R&D-Agent (Yang et al., 2025) navigate complex coding tasks via iterative refinement and multi-agent parallelization. While effective at exploration, these methods lack explicit mechanisms for addressing the localization of debugging knowledge, often leading to repetitive errors in the search tree, a gap we address via a debug consultant that enables adaptive learning of the execution environment.

Self-correction & context engineering. LLMs have demonstrated the ability to “self-debug” code via iterative generation (Chen et al., 2024; Yang et al., 2024). However, in domainspecific tasks, pre-training priors frequently override runtime feedback, causing persistent “fix loops.” To mitigate this, we draw on Agentic Context Engineering (ACE) (Zhang et al., 2025) to treat context as an evolving playbook. Crucially, unlike open-ended agents like Voyager (Wang et al., 2024a) that accumulate success skills, our domain necessitates the systematic accumulation of failures. We argue that in reward-sparse environments like tabular debugging, learning what not to do (negative constraints) provides just as valuable a signal as sparse successes.

## 6 Discussion

Our findings reveal fundamental limitations of existing agents. Some of what an agent learns is local to a branch. Much of it, however, is a global property of the environment or problem setting: which library versions are installed, which API signatures are valid, how long a fold of training takes on the available cores, or the features of the dataset at hand. Global facts are invariant across the tree, and re-deriving them per branch or node is redundant. A global memory that separates the two lets a run behave as a single agent rather than many isolated ones, which reduces redundancy. More importantly, memory may pay off in generating better hypotheses based on lessons learned globally across previous nodes. An agent that can recall what it has already ruled out can condition its next hypothesis on the accumulated failures instead of resampling from an unchanged prior. We expect the value of such memory to grow rather than shrink as autoresearch systems become more ambitious.

Existing agents are also limited in their ability to explore the search space. Tree search algorithms assume that expanding a node produces a novel candidate, but LLMs often write nearly the same program over and over again. When an agent produces dozens of near-identical programs, the tree is wide only on paper. Better selection will therefore help only so much until agents are designed to propose more varied solutions, which may be why our own selection improvements reduce variance more than they raise peak scores.

Reliability is another limitation of current agents, and perhaps the most overlooked. On a meaningful fraction of runs, agents produce no result at all, and the common practice of averaging over successful runs hides these failures so the waste goes uncounted. As autoresearch systems take on longer and more expensive tasks, a run that quietly produces nothing becomes far more costly than one that produces a mediocre answer. Making agents dependable will matter as much as making them capable.

The interventions we study are deliberately simple, and each recovers substantial performance from the same underlying model, which implies that current agents operate well below the ceiling their language models already permit. As these systems begin to take on more of the research loop, forming hypotheses, designing experiments, and interpreting results, the cost of an agent that forgets what it has learned, proposes what it has already tried, or fails silently will only rise. Closing that gap will require treating memory, diversity, and reliability as first-class objectives of agent design rather than as incidental properties of the scaffold.

## Acknowledgements

This project was supported by a research award from the Center for AI and Responsible Financial Innovation at Columbia University and by the Columbia Center for AI Technology.

## References

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander M ˛adry. MLE-bench: Evaluating machine learning agents on machine learning engineering. arXiv preprint arXiv:2410.07095, 2024. URL https://arxiv.org/abs/2410.07095.

Xinyun Chen, Maxwell Lin, Nathanael Schärli, et al. Teaching large language models to self-debug. In International Conference on Learning Representations (ICLR), 2024.

Shangheng Du, Xiangchao Yan, Jinxin Shi, Zongsheng Cao, Shiyang Feng, Zichen Liang, Boyuan Sun, Tianshuo Peng, Yifan Zhou, Xin Li, Jie Zhou, Liang He, Bo Zhang, and Lei Bai. MLEvolve: A self-evolving framework for automated machine learning algorithm discovery, 2026. URL https://arxiv.org/abs/2606.06473.

Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer, and Frank Hutter. Auto-Sklearn 2.0: Hands-free AutoML via meta-learning. Journal of Machine Learning Research, 23(261):1–61, 2022. URL http://jmlr.org/papers/v23/21-0992.html.

Jonas Gehring, Kunhao Zheng, Jade Copet, Vegard Mella, Taco Cohen, and Gabriel Synnaeve. RLEF: Grounding code LLMs in execution feedback with reinforcement learning. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview.net/f orum?id=PzSG5nKe1q.

Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/f orum?id=cp5PvcI6w8\_.

Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, and Yuxiang Wu. AIDE: AI-driven exploration in the space of code. 2025. URL https: //arxiv.org/abs/2502.13138.

Liqiang Jing, Zhehui Huang, Xiaoyang Wang, Wenlin Yao, Wenhao Yu, Kaixin Ma, Hongming Zhang, Xinya Du, and Dong Yu. DSBench: How far are data science agents from becoming data science experts?, 2025. URL https://arxiv.org/abs/2409.07703.

Ricardo Knauer, Marvin Grimm, and Erik Rodner. PMLBmini: A tabular classification benchmark suite for data-scarce applications, 2024. URL https://arxiv.org/abs/2409.0 1635.

Zexi Liu, Yuzhu Cai, Xinyu Zhu, Yujie Zheng, Runkun Chen, Ying Wen, Yanfeng Wang, Weinan E, and Siheng Chen. ML-Master: Towards AI-for-AI via integration of exploration and reasoning, 2025. URL https://arxiv.org/abs/2506.16499.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The AI scientist: Towards fully automated open-ended scientific discovery, 2024. URL https: //arxiv.org/abs/2408.06292.

Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Sanchaita Hazra, Ashish Sabharwal, and Peter Clark. Position: Data-driven discovery with large generative models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 34350–34382. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/majum der24a.html.

Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. DiscoveryBench: Towards data-driven discovery with large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=vyflgpwfJW.

Deepak Nathani, Lovish Madaan, Nicholas Roberts, Nikolay Bashlykov, Ajay Menon, Vincent Moens, Amar Budhiraja, Despoina Magka, Vladislav Vorotilov, Gaurav Chaurasia, Dieuwke Hupkes, Ricardo Silveira Cabral, Tatiana Shavrina, Jakob Foerster, Yoram Bachrach, William Yang Wang, and Roberta Raileanu. MLGym: A new framework and benchmark for advancing AI research agents, 2025. URL https://arxiv.org/abs/2502.1 4499.

Randal S. Olson and Jason H. Moore. TPOT: A tree-based pipeline optimization tool for automating machine learning. In Frank Hutter, Lars Kotthoff, and Joaquin Vanschoren (eds.), Proceedings of the Workshop on Automatic Machine Learning, volume 64 of Proceedings ofMachine Learning Research, pp. 66–74, New York, New York, USA, 24 Jun 2016. PMLR. URL https://proceedings.mlr.press/v64/olson\_tpot\_2016.html.

Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Xuanhe Zhou, Yufei Huang, Chaojun Xiao, Chi Han, Yi Ren Fung, Yusheng Su, Huadong Wang, Cheng Qian, Runchu Tian, Kunlun Zhu, Shihao Liang, Xingyu Shen, Bokai Xu, Zhen Zhang, Yining Ye, Bowen Li, Ziwei Tang, Jing Yi, Yuzhang Zhu, Zhenning Dai, Lan Yan, Xin Cong, Yaxi Lu, Weilin Zhao, Yuxiang Huang, Junxi Yan, Xu Han, Xian Sun, Dahai

Li, Jason Phang, Cheng Yang, Tongshuang Wu, Heng Ji, Guoliang Li, Zhiyuan Liu, and Maosong Sun. Tool learning with foundation models. ACM Comput. Surv., 57(4), December 2024. ISSN 0360-0300. doi: 10.1145/3704435. URL https://doi.org/10.1145/3704435.

Edan Toledo, Karen Hambardzumyan, Martin Josifoski, Rishi Hazra, Nicolas Baldwin, Alexis Audran-Reiss, Michael Kuchnik, Despoina Magka, Minqi Jiang, Alisia Maria Lupidi, Andrei Lupu, Roberta Raileanu, Kelvin Niu, Tatiana Shavrina, Jean-Christophe Gagnon-Audet, Michael Shvartsman, Shagun Sodhani, Alexander H. Miller, Abhishek Charnalia, Derek Dunfield, Carole-Jean Wu, Pontus Stenetorp, Nicola Cancedda, Jakob Nicolaus Foerster, and Yoram Bachrach. AI research agents for machine learning: Search, exploration, and generalization in MLE-bench, 2025. URL https://arxiv.org/abs/2507.02554.

Chi Wang, Qingyun Wu, Markus Weimer, and Erkang Zhu. FLAML: A fast and lightweight AutoML library, 2021. URL https://arxiv.org/abs/1911.04706.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2024a. ISSN 2835-8856. URL https://openreview.net/forum?id=ehfRiF0R3a.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6), March 2024b. ISSN 2095-2236. doi: 10.1007/s11704-024-40231-1. URL http://dx.doi.org/10.1007/s11704-024-40231-1.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=mXpq6ut8J3.

Xu Yang, Xiao Yang, Shikai Fang, Yifei Zhang, Jian Wang, Bowen Xian, Qizheng Li, Jingyuan Li, Minrui Xu, Yuante Li, Haoran Pan, Yuge Zhang, Weiqing Liu, Yelong Shen, Weizhu Chen, and Jiang Bian. R&D-Agent: An LLM-agent framework towards autonomous data science. arXiv preprint arXiv:2505.14738, 2025. URL https://arxiv.org/abs/2505.14738.

Xin Yin, Chao Ni, Shaohua Wang, Zhenhao Li, Limin Zeng, and Xiaohu Yang. ThinkRepair: Self-directed automated program repair, 2024. URL https://arxiv.org/abs/2407.20898.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. Agentic context engineering: Evolving contexts for selfimproving language models, 2025. URL https://arxiv.org/abs/2510.04618.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 62138– 62160. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/zhou24r.html.

## A Context-aware debugging: detailed analysis

This section provides a comprehensive analysis of how the debug consultant accumulates a shared model of the execution environment and propagates it across the search tree. We show that baseline agents waste compute rediscovering the same library incompatibilities and API mismatches across branches, trace how the consultant’s accumulated knowledge of the runtime—which library versions are installed, which API signatures are valid, and which code patterns crash—changes agent behavior, and present concrete before/after examples.

## A.1 The environmental blindness problem

Without the debug consultant, each branch must independently discover the execution environment’s constraints—which library versions are installed, which API parameters have been deprecated, which code patterns are valid in the current container. The result is massive redundancy: 46.0% of baseline nodes waste compute re-encountering bugs that have already been seen within the same seed, compared to only 7.8% under the consultant. This redundant re-discovery causes total failure in 17 baseline seeds (zero valid submissions).

## A.2 How adaptive learning changes agent behavior

Table 4 and Figure 4 show per-competition recovery statistics. The BANNED list converts what would be random retries into informed corrections: the agent’s next attempt is constrained to avoid known-failing patterns, collapsing the search space so it is more likely to succeed.

Treatment recovers from 96.8% of bug streaks (consecutive buggy nodes), with 72.4% fixed on the first attempt (average 1.40 attempts). Baseline recovers from only 86.2%, with 41.4% first-attempt fixes and an average of 6.43 attempts. On the hardest competitions (S5E6, S5E8), baseline first-attempt fix rates are 19.0% and 0.0% respectively, while treatment achieves 47.6% and 69.4%.

<table><tr><td></td><td colspan="3">Treatment</td><td colspan="3">Baseline</td></tr><tr><td>Competition</td><td>Rec.%</td><td>1st%</td><td>Att.</td><td>Rec.%</td><td>1st%</td><td>Att.</td></tr><tr><td>Cirrhosis</td><td>98.2</td><td>75.9</td><td>1.31</td><td>78.8</td><td>38.5</td><td>3.69</td></tr><tr><td>GNSS</td><td>98.1</td><td>75.5</td><td>1.34</td><td>86.4</td><td>57.9</td><td>3.55</td></tr><tr><td>Spaceship</td><td>97.9</td><td>73.1</td><td>1.51</td><td>95.6</td><td>62.8</td><td>2.26</td></tr><tr><td>Wine</td><td>98.7</td><td>81.8</td><td>1.22</td><td>80.6</td><td>24.0</td><td>7.00</td></tr><tr><td>S5E3</td><td>100.0</td><td>80.6</td><td>1.24</td><td>92.3</td><td>50.0</td><td>3.94</td></tr><tr><td>S5E6</td><td>93.3</td><td>47.6</td><td>1.95</td><td>77.8</td><td>19.0</td><td>16.52</td></tr><tr><td>S5E7</td><td>100.0</td><td>69.1</td><td>1.38</td><td>94.1</td><td>56.2</td><td>2.12</td></tr><tr><td>S5E8</td><td>85.7</td><td>69.4</td><td>1.50</td><td>78.3</td><td>0.0</td><td>19.39</td></tr><tr><td>S5E12</td><td>92.7</td><td>63.2</td><td>1.66</td><td>82.8</td><td>16.7</td><td>11.71</td></tr><tr><td>Overall</td><td>96.8</td><td>72.4</td><td>1.40</td><td>86.2</td><td>41.4</td><td>6.43</td></tr></table>

Table 4: Bug recovery statistics from AIDE journal analysis. Recovery = bug streak followed by ≥ 1 valid node. 1st-fix = recovered on the first attempt after the streak.

## A.3 Synchronization speed

Table 5 and Figure 5 show how quickly the consultant learns the environment. Treatment seeds produce their first valid node at step 0.4 on average (median: 0)—the consultant has already learned enough about the runtime by the first step to guide the agent past common pitfalls. Baseline seeds, which must rediscover these constraints independently, require 6.8 steps on average (median: 6). Treatment reaches 94% submission by step 1 (100% by step 3); baseline remains at 0% through step 3 on five competitions. Under any fixed compute budget ≤ 3 steps, the scaffolded agent dominates.

![](images/850ff2fd74650edf619b7c332804882181e7d4658c8aac752ef5c2f47201f357.jpg)

Figure 4: The consultant raises first-attempt fix rate from 41.4% to 72.4%. Bug recovery rate (left) and first-attempt fix rate (right) per competition. Constraint injection converts random retries into informed corrections.
<table><tr><td></td><td colspan="4">Treatment (%)</td><td colspan="4">Baseline (%)</td></tr><tr><td>Competition</td><td>SO</td><td>S1</td><td>S3</td><td>S5</td><td>SO</td><td>S1</td><td>S3</td><td>S5</td></tr><tr><td>Cirrhosis</td><td>80</td><td>100</td><td>100</td><td>100</td><td>0</td><td>0</td><td>20</td><td>40</td></tr><tr><td>GNSS</td><td>70</td><td>100</td><td>100</td><td>100</td><td>20</td><td>20</td><td>20</td><td>40</td></tr><tr><td>Spaceship</td><td>70</td><td>100</td><td>100</td><td>100</td><td>0</td><td>0</td><td>40</td><td>80</td></tr><tr><td>Wine</td><td>60</td><td>90</td><td>100</td><td>100</td><td>0</td><td>0</td><td>0</td><td>10</td></tr><tr><td>S5E3</td><td>70</td><td>100</td><td>100</td><td>100</td><td>0</td><td>0</td><td>0</td><td>30</td></tr><tr><td>S5E6</td><td>50</td><td>90</td><td>100</td><td>100</td><td>0</td><td>0</td><td>0</td><td>30</td></tr><tr><td>S5E7</td><td>70</td><td>90</td><td>100</td><td>100</td><td>10</td><td>10</td><td>30</td><td>90</td></tr><tr><td>S5E8</td><td>70</td><td>80</td><td>100</td><td>100</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>S5E12</td><td>80</td><td>100</td><td>100</td><td>100</td><td>0</td><td>0</td><td>0</td><td>20</td></tr><tr><td>Mean</td><td>69</td><td>94</td><td>100</td><td>100</td><td>3</td><td>3</td><td>12</td><td>38</td></tr></table>

Table 5: Fraction of seeds with ≥ 1 valid submission by step N (AIDE, 10 seeds per competition). Treatment first-valid step: mean 0.4 (median 0). Baseline: mean 6.8 (median 6).

## A.4 From environmental synchronization to solution quality

Table 6 shows that the treatment’s overall valid rate is 79.0% (2,889 / 3,655 nodes) versus 54.7% (2,464 / 4,507) for the baseline. Treatment generates fewer total nodes but a substantially higher fraction are valid, indicating more efficient use of the compute budget. This is partly because valid nodes consume significantly more runtime than invalid ones: a valid node must execute the full pipeline—including data preprocessing, model training (e.g., gradient-boosted trees), and prediction—while an invalid node crashes early and returns quickly. Fewer but valid nodes therefore represent a much larger share of useful compute.

With the environment solved, the LLM focuses on modeling: ensembles, calibration, and problem reformulation emerge through successive refinement. Per-competition correlations are reported in Appendix B.5.

## B Extended experimental results

## B.1 Per-seed OOS scores: AIDE

Table 7 reports the graded out-of-sample score for every AIDE seed (9 competitions × 10 seeds). Treatment achieves valid submissions on all 90 seeds; baseline has 17 null runs (“—”). For Cirrhosis (Log Loss), lower is better; for all others, higher is better.

![](images/28d1abd6db2bdcf886029dc1204bcc3e5407d6b5f0b194a42efc3c5a7de87444.jpg)

![](images/ebb35ce2acfa2cf95bdf973b05e8075fc873f4bd3b17dbaa424a6708995736b7.jpg)

![](images/cca24f82da86f8d4e16e47b6f654bfe7c0ce83d46debd0c10de27b3c20dc9dc0.jpg)

![](images/f860d5cc164b6cd02216dc38b5b630e87eb455a99f1338f658c65d44908508f0.jpg)

![](images/8607bbfa6d3956e1376fa5cb67b7d1b742baf3158af4a75a4fb83c3cd8bbe5dc.jpg)

![](images/67953557f37251fb7a74616383ea2eccfb02afa6108e9d5e07a8fadc9239b1ff.jpg)

![](images/fc4c629f82c460ec520212cf1f72840c27eb04c0ab6f9c9fe5b79f2a947a6fae.jpg)

![](images/c8f219294bd0bdd480d1bb2d94424f6f854d24c1b427963170b855d078679a2c.jpg)

![](images/def954f6e1bfd5926876d2a55873eccfb5e3aa689c26e537be8d878f784fe385.jpg)

Figure 5: Treatment reaches 100% submission rate by step 3; mean baseline submission rate remains below 40% through step 5. Cumulative submission rate by exploration step for treatment vs. baseline across all 9 competitions (AIDE, 10 seeds each).
<table><tr><td></td><td colspan="3">Treatment</td><td colspan="3">Baseline</td><td></td></tr><tr><td>Comp.</td><td>Tot.</td><td>Val.</td><td>%</td><td>Tot.</td><td>Val.</td><td>%</td><td>∆%</td></tr><tr><td>Cirrhosis</td><td>500</td><td>428</td><td>85.6</td><td>261</td><td>150</td><td>57.5</td><td>+28.1</td></tr><tr><td>GNSS</td><td>389</td><td>317</td><td>81.5</td><td>432</td><td>278</td><td>64.4</td><td>+17.1</td></tr><tr><td>Spaceship</td><td>524</td><td>380</td><td>72.5</td><td>744</td><td>641</td><td>86.2</td><td>-13.7</td></tr><tr><td>Wine</td><td>530</td><td>435</td><td>82.1</td><td>432</td><td>234</td><td>54.2</td><td>+27.9</td></tr><tr><td>S5E3</td><td>557</td><td>474</td><td>85.1</td><td>577</td><td>370</td><td>64.1</td><td>+21.0</td></tr><tr><td>S5E6</td><td>198</td><td>111</td><td>56.1</td><td>434</td><td>38</td><td>8.8</td><td>+47.3</td></tr><tr><td>S5E7</td><td>597</td><td>521</td><td>87.3</td><td>725</td><td>651</td><td>89.8</td><td>-2.5</td></tr><tr><td>S5E8</td><td>169</td><td>100</td><td>59.2</td><td>429</td><td>37</td><td>8.6</td><td>+50.6</td></tr><tr><td>S5E12</td><td>191</td><td>123</td><td>64.4</td><td>473</td><td>65</td><td>13.7</td><td>+50.7</td></tr><tr><td>Total</td><td>3,655</td><td>2,889</td><td>79.0</td><td>4,507</td><td>2,464</td><td>54.7</td><td>+24.3</td></tr></table>

Table 6: Node counts from AIDE journal analysis (9 comps × 10 seeds). Treatment generates fewer total nodes but a higher fraction are valid.

## B.2 Per-seed OOS scores: ML-Master

Table 8 reports the graded OOS score for every ML-Master seed. Both treatment and baseline achieve valid submissions on all 90 seeds.

## B.3 Case studies: how valid nodes become better solutions

The two case studies below illustrate the same causal chain: the debug consultant removes a persistent API bug → the agent iterates freely → it builds qualitatively more sophisticated solutions. Both cases share a common pattern: the baseline LLM has the same modeling knowledge as the treatment, but it never gets to use it because its compute budget is consumed by redundant bug encounters.

<table><tr><td>Competition</td><td>Cond.</td><td>SO</td><td>S1</td><td>S2</td><td>S3</td><td>S4</td><td>S5</td><td>S6</td><td>S7</td><td>S8</td><td>S9</td><td>Mean</td><td>Std</td></tr><tr><td>Cirrhosis†</td><td>T</td><td>0.3839</td><td>0.3849</td><td>0.3802</td><td>0.4004</td><td>0.3780</td><td>0.3767 0.3847</td><td>0.3786 0.3827</td><td>0.3802</td><td>0.3985</td><td>0.3803</td><td>0.384</td><td>0.008</td></tr><tr><td></td><td>B</td><td>0.3786</td><td>0.3826</td><td>0.3811</td><td>0.3799</td><td>0.3790</td><td></td><td></td><td>0.4868</td><td>0.3870</td><td></td><td>0.394</td><td>0.035</td></tr><tr><td>GNSS</td><td>T B</td><td>0.9668 0.9637</td><td>0.9676 0.9673</td><td>0.9673 0.9680</td><td>0.9687 0.9679</td><td>0.9684 0.9668</td><td>0.9640 0.9687</td><td>0.9672 0.9627</td><td>0.9634</td><td>0.9675 0.9690</td><td>0.9649</td><td>0.967 0.967</td><td>0.002 0.002</td></tr><tr><td>S5E3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>T B</td><td>0.9590</td><td>0.9572 0.8826</td><td>0.9566 0.8920</td><td>0.9634 0.8771</td><td>0.9583 0.8960</td><td>0.9104 0.9016</td><td>0.9515 0.8748</td><td>0.9606 0.8782</td><td>0.9614 0.8884</td><td>0.9570 0.8966</td><td>0.954 0.887</td><td>0.015 0.010</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S5E6</td><td>T B</td><td>0.3321</td><td>0.3328</td><td>0.3439 0.3303</td><td>0.3385 0.3275</td><td>0.3399</td><td>0.3405</td><td>0.3299</td><td>0.3407</td><td>0.3316</td><td>0.3413</td><td>0.337</td><td>0.005 0.005</td></tr><tr><td></td><td></td><td>0.3275</td><td>0.3387</td><td></td><td></td><td></td><td></td><td>0.3301</td><td>0.3372</td><td>0.3372</td><td>0.3324</td><td>0.333</td><td></td></tr><tr><td>S5E7</td><td>T B</td><td>0.9682</td><td>0.9692</td><td>0.9682</td><td>0.9703</td><td>0.9698</td><td>0.9687</td><td>0.9698</td><td>0.9687</td><td>0.9698</td><td>0.9703</td><td>0.969</td><td>0.001</td></tr><tr><td></td><td></td><td>0.9682</td><td>0.9676</td><td>0.9682</td><td>0.9676</td><td></td><td>0.9682</td><td>0.9676</td><td>0.9682</td><td>0.9676</td><td>0.9676</td><td>0.968</td><td>0.000</td></tr><tr><td>S5E8</td><td>T</td><td>0.9702</td><td>0.9700</td><td>0.9693</td><td>0.9413</td><td>0.9684</td><td>0.9692</td><td>0.9699</td><td>0.9696</td><td>0.9691</td><td>0.9698</td><td>0.967</td><td>0.009</td></tr><tr><td></td><td>B</td><td>0.9684</td><td></td><td></td><td>0.9697</td><td>0.9696</td><td>0.9695</td><td></td><td>0.9688</td><td></td><td>0.9691</td><td>0.969</td><td>0.001</td></tr><tr><td>S5E12</td><td>T</td><td>0.7264</td><td>0.7110</td><td>0.7262</td><td>0.7259</td><td>0.7264</td><td>0.7255</td><td>0.7258</td><td>0.7257</td><td>0.7264</td><td>0.7268</td><td>0.725</td><td>0.005</td></tr><tr><td></td><td>B</td><td>0.7271</td><td>0.7281</td><td>0.7271</td><td>0.7275</td><td>0.7257</td><td>0.7281</td><td></td><td>0.7256</td><td></td><td></td><td>0.727</td><td>0.001</td></tr><tr><td>Spaceship</td><td>T</td><td>0.8172</td><td>0.8138</td><td>0.8081</td><td>0.8149</td><td>0.7920</td><td>0.8172</td><td>0.7897</td><td>0.8207</td><td>0.8149</td><td>0.7736</td><td>0.806</td><td>0.016</td></tr><tr><td></td><td>B</td><td>0.8000</td><td>0.8149</td><td>0.8218</td><td>0.8172</td><td>0.8241</td><td>0.6184</td><td></td><td>0.8299</td><td>0.7667</td><td>0.8207</td><td>0.790</td><td>0.067</td></tr><tr><td>Wine</td><td>T</td><td>0.4281</td><td>0.3980</td><td>0.4294</td><td>0.3321</td><td>0.4031</td><td>0.4284</td><td>0.4121</td><td>0.4419</td><td></td><td>0.4315</td><td>0.407</td><td>0.034</td></tr><tr><td></td><td>B</td><td>0.3256</td><td>0.2990</td><td>0.4249</td><td>0.4264</td><td>0.3728</td><td>0.3016</td><td>0.4181</td><td>0.4329</td><td>0.3664</td><td></td><td>0.375</td><td>0.058</td></tr></table>

Table 7: AIDE Per-Seed OOS Scores. Bold = per-seed winner; values tied at the displayed precision are bolded in both rows. — = null (no valid submission). <sup>†</sup> = lower is better.

<table><tr><td>Competition</td><td>Cond.</td><td>SO</td><td>S1</td><td>S2</td><td>S3</td><td>S4</td><td>S5</td><td>S6</td><td>S7</td><td>S8</td><td>S9</td><td>Mean</td><td>Std</td></tr><tr><td>Cirrhosis†</td><td>T</td><td>0.3838</td><td>0.3795</td><td>0.3822</td><td>0.3775</td><td>0.3802</td><td>0.3813</td><td>0.3837</td><td>0.3795</td><td>0.3813</td><td>0.3756</td><td>0.380</td><td>0.002</td></tr><tr><td></td><td>B</td><td>0.3856</td><td>0.3789</td><td>0.3882</td><td>0.3789</td><td>0.3890</td><td>0.3794</td><td>0.3821</td><td>0.3877</td><td>0.3881</td><td>0.4000</td><td>0.386</td><td>0.006</td></tr><tr><td>GNSS</td><td>T</td><td>0.9521</td><td>0.9617</td><td>0.9480</td><td>0.9494</td><td>0.9576</td><td>0.9644</td><td>0.9549</td><td>0.9617</td><td>0.9207</td><td>0.9590</td><td>0.953</td><td>0.012</td></tr><tr><td></td><td>B</td><td>0.8124</td><td>0.8148</td><td>0.8130</td><td>0.8128</td><td>0.8140</td><td>0.8099</td><td>0.8112</td><td>0.8128</td><td>0.8145</td><td>0.8150</td><td>0.813</td><td>0.002</td></tr><tr><td>S5E3</td><td>T</td><td>0.8585</td><td>0.8402</td><td>0.8539</td><td>0.8402</td><td>0.8721</td><td>0.8448</td><td>0.8448</td><td>0.8676</td><td>0.8402</td><td>0.8493</td><td>0.851</td><td>0.011</td></tr><tr><td></td><td>B</td><td>0.8585</td><td>0.8630</td><td>0.8585</td><td>0.8585</td><td>0.8676</td><td>0.8539</td><td>0.8539</td><td>0.8585</td><td>0.8585</td><td>0.8676</td><td>0.860</td><td>0.005</td></tr><tr><td>S5E6</td><td>T</td><td>0.3421</td><td>0.3338</td><td>0.3343</td><td>0.3406</td><td>0.3349</td><td>0.3358</td><td>0.3358</td><td>0.3345</td><td>0.3309</td><td>0.3347</td><td>0.336</td><td>0.003</td></tr><tr><td></td><td>B</td><td>0.3238</td><td>0.3297</td><td>0.3251</td><td>0.3342</td><td>0.3256</td><td>0.3135</td><td>0.3213</td><td>0.3243</td><td>0.3296</td><td>0.3352</td><td>0.326</td><td>0.006</td></tr><tr><td>S5E7</td><td>T</td><td>0.9687</td><td>0.9682</td><td>0.9676</td><td>0.9682</td><td>0.9655</td><td>0.9644</td><td>0.9687</td><td>0.9692</td><td>0.9682</td><td>0.9682</td><td>0.968</td><td>0.001</td></tr><tr><td></td><td>B</td><td>0.9671</td><td>0.9682</td><td>0.9682</td><td>0.9682</td><td>0.9676</td><td>0.9660</td><td>0.9676</td><td>0.9682</td><td>0.9682</td><td>0.9687</td><td>0.968</td><td>0.001</td></tr><tr><td>S5E8</td><td>T</td><td>0.9686</td><td>0.9688</td><td>0.9686</td><td>0.9682</td><td>0.9693</td><td>0.9689</td><td>0.9692</td><td>0.9697</td><td>0.9690</td><td>0.9689</td><td>0.969</td><td>0.000</td></tr><tr><td></td><td>B</td><td>0.9669</td><td>0.9638</td><td>0.9653</td><td>0.9683</td><td>0.9654</td><td>0.9640</td><td>0.9650</td><td>0.9651</td><td>0.9680</td><td>0.9683</td><td>0.966</td><td>0.002</td></tr><tr><td>S5E12</td><td>T</td><td>0.7277</td><td>0.7276</td><td>0.7278</td><td>0.7273</td><td>0.7275</td><td>0.7274</td><td>0.7277</td><td>0.7276</td><td>0.7279</td><td>0.7271</td><td>0.728</td><td>0.000</td></tr><tr><td></td><td>B</td><td>0.7258</td><td>0.7265</td><td>0.6877</td><td>0.7258</td><td>0.7198</td><td>0.7179</td><td>0.7180</td><td>0.7188</td><td>0.7226</td><td>0.7258</td><td>0.719</td><td>0.011</td></tr><tr><td>Spaceship</td><td>T</td><td>0.8184</td><td>0.8149</td><td>0.8126</td><td>0.8138</td><td>0.8207</td><td>0.8207</td><td>0.8184</td><td>0.7977</td><td>0.8172</td><td>0.7988</td><td>0.813</td><td>0.008</td></tr><tr><td></td><td>B</td><td>0.7287</td><td>0.7782</td><td>0.8138</td><td>0.8115</td><td>0.8092</td><td>0.7356</td><td>0.8092</td><td>0.8161</td><td>0.8172</td><td>0.8184</td><td>0.794</td><td>0.033</td></tr><tr><td>Wine</td><td></td><td></td><td></td><td>0.4181</td><td>0.3753</td><td></td><td>0.4028</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>T B</td><td>0.4011 0.4318</td><td>0.4165 0.4245</td><td>0.3308</td><td>0.4240</td><td>0.4243 0.4240</td><td>0.4103</td><td>0.4143 0.4158</td><td>0.4254 0.4176</td><td>0.4221 0.4227</td><td>0.4227 0.4227</td><td>0.412 0.412</td><td>0.015 0.028</td></tr></table>

Table 8: ML-Master treatment wins the majority of per-seed comparisons despite a fullyvalid baseline. Per-seed OOS scores for 9 competitions × 10 seeds. Bold = per-seed winner; values tied at the displayed precision are bolded in both rows. <sup>†</sup> = lower is better.

## B.3.1 Case study 1: S5E3 seed 6 (∆ = +0.077 AUC)

Figure 6 compares the search trees for S5E3 (binary classification, rainfall prediction), seed 6.

Baseline (left, 74 nodes, 2 valid, 2.7%): The agent’s first code draft calls lgb.train() with the deprecated early\_stopping\_rounds parameter. LightGBM raises a TypeError; the agent’s try/except handler catches it silently. Having no mechanism to propagate this failure, the agent generates nearly identical code 72 more times—each node re-encountering the same TypeError. Only 2 nodes avoid the pattern, producing a single-model LGBMClassifier with basic feature engineering (4 domain interactions, no calibration, no ensemble). OOS score: 0.8748.

Treatment (right, 39 nodes, 31 valid, 79.5%): The debug consultant records the early\_stopping\_rounds TypeError at step 0 and adds it to the BANNED list. From that point forward, the agent never regenerates this pattern, enabling 31 valid iterations to explore progressively better approaches. The final solution is a 3-model ensemble (XGBoost + 2 LightGBM variants) with Platt-scaling calibration and greedy weight optimization. OOS score: 0.9515 (∆ = +0.0767). With the debug consultant eliminating redundant failures, the treatment refines its approach through 15× more hypothesis-testing cycles, converging on a substantially more sophisticated solution.

![](images/488f41919588a5711ce20eac240a307e74bbe1b090056dc9df73e30619af9b87.jpg)

![](images/13b02c5dcaf8ea00468762a87aab1e3e6afd37ea1616b24f285abdf36047fbb3.jpg)

Figure 6: Search tree comparison for S5E3 seed 6. Left: baseline (74 nodes, 2 valid, 2.7%). Right: treatment (39 nodes, 31 valid, 79.5%). Red = buggy; green = valid. The baseline loops on the same early\_stopping\_rounds TypeError for its entire budget.

## B.3.2 Case study 2: Wine Quality (∆ = +0.099 QWK)

Figure 7 compares the search trees for Wine (ordinal classification, 7 quality levels), seed 1.

Baseline (left, 10 nodes, 2 valid, 20%): The agent calls LGBMRegressor.fit() with early\_stopping\_rounds as a keyword argument—deprecated in the installed Light-GBM version. Eight of 10 nodes crash with deprecated-API TypeErrors (7 from early\_stopping\_rounds, 1 from verbose). The 2 surviving nodes use a single LGBMRegressor with naive rounding of continuous predictions to integer quality labels. No stacking, no calibration, no threshold optimization. OOS score: 0.2990.

Treatment (right, 57 nodes, 51 valid, 89%): The consultant’s BANNED list prevents the early\_stopping\_rounds TypeError from step 0. With 51 valid iterations, the agent builds a diverse bagged LightGBM ensemble with stacking (Ridge + Isotonic regression on out-offold predictions) and threshold optimization that maps continuous predictions to discrete quality labels, maximizing QWK directly. OOS score: 0.3980 (∆ = +0.099).

The modeling insight gap. Both baseline and treatment use a regression formulation. The crucial difference is what the agent does with additional valid iterations: it discovers that stacking multiple LightGBM models with isotonic calibration and optimized thresholds produces substantially higher QWK than a single model with naive rounding. With 51 valid nodes, the treatment has the budget to make this discovery; with 2 valid nodes, the baseline never gets the chance.

1 #!/usr/bin/env python3   
import os   
3 import warnings   
4 import numpy as np   
import pandas as pd

![](images/f93aeb23410db4c17219e6ea00d669b89b39eedd9b6662d3a7291af6671656d8.jpg)  
Figure 7: Search tree comparison for Wine seed 1. Left: baseline (10 nodes, 2 valid, 20%). Right: treatment (57 nodes, 51 valid, 89%). The baseline exhausts its budget on repeated early\_stopping\_rounds TypeErrors.

## B.3.3 Synthesis: more valid nodes → better solutions

Both case studies tell the same story through different competitions and bug types:
<table><tr><td></td><td>S5E3 seed 6</td><td>Wine seed 1</td></tr><tr><td>Blocking bug</td><td>early_stopping_rounds</td><td>early_stopping_rounds</td></tr><tr><td>Baseline valid nodes</td><td>2 / 74 (2.7%)</td><td>2 / 10 (20%)</td></tr><tr><td>Treatment valid nodes</td><td>31 / 39 (79.5%)</td><td>51 / 57 (89%)</td></tr><tr><td>Baseline approach</td><td>Single LGB</td><td>LGB regression</td></tr><tr><td>Treatment approach OOS∆</td><td>3-model ensemble +0.077</td><td>Bagged ensemble + threshold opt. +0.099</td></tr></table>

Table 9: Case study summary. The debug consultant’s primary effect is enabling iteration.

We find the same pattern in both cases: adaptive learning allows the LLM to iterate. A single valid node produces a first working draft. Dozens of valid nodes allow the agent to discover ensembles, calibration, problem reformulation, and threshold optimization.

## B.4 Full generated code

Below are the complete, unedited solutions generated by the LLM agent for both case studies. Each listing is the exact code that was executed and graded.

## B.4.1 S5E3 seed 6 — Baseline (step 20, OOS = 0.875)

Note line 88: “no early stopping due to environment limitations”—the agent is aware of the API bug but cannot resolve it.

Listing 1: Baseline best solution: S5E3 seed 6 (141 lines). Single LGBMClassifier with basic feature engineering. No ensemble no calibration no early stopping.

```python
from sklearn.model_selection import StratifiedKFold
7 from sklearn.metrics import roc_auc_score
8 from lightgbm import LGBMClassifier
10 warnings.filterwarnings("ignore")
11
12 SEED = 42
13 N_SPLITS = 5
14
15 # Paths
16 INPUT_DIR = "./input"
17 SUBMISSION_DIR = "./submission"
18 WORKING_DIR = "./working"
19 os.makedirs(SUBMISSION_DIR, exist_ok=True)
20 os.makedirs(WORKING_DIR, exist_ok=True)
21
22 # Load data
train_path = os.path.join(INPUT_DIR, "train.csv")
24 test_path = os.path.join(INPUT_DIR, "test.csv")
25
26 train = pd.read_csv(train_path)
28
29 test = pd.read_csv(test_path)
# Feature engineering
31 def prepare_df(df):
<sup>32</sup> <sub>33</sub> df = df.copy()
# day cyclic encoding (assume 365-day cycle)
34 if "day" in df.columns:
35 df["day_sin"] = np.sin(2 * np.pi * df["day"] / 365.0)
36 df["day_cos"] = np.cos(2 * np.pi * df["day"] / 365.0)
37 # temperature range
38 if {"maxtemp", "mintemp"}.issubset(df.columns):
39 df["temp_range"] = df["maxtemp"] - df["mintemp"]
40 # dewpoint gap to temp (note column 'temparature' exists in dataset)
41 if {"temparature", "dewpoint"}.issubset(df.columns):
42 df["temp_minus_dew"] = df["temparature"] - df["dewpoint"]
43 # pressure * humidity interaction
44 if {"pressure", "humidity"}.issubset(df.columns):
45 df["press_hum"] = df["pressure"] * df["humidity"] / 1e3
# wind * cloud interaction
47 if {"windspeed", "cloud"}.issubset(df.columns):
48 df["wind_cloud"] = df["windspeed"] * df["cloud"] / 100.0
49 return df
50
51
52 train = prepare_df(train)
53 test = prepare_df(test)
54
55 TARGET = "rainfall"
56 ID_COL = "id"
58 # Features to use (exclude id, target)
59 excluded = {ID_COL, TARGET}
60 features = [c for c in train.columns if c not in excluded]
61
62 # Ensure same engineered columns exist in test; if missing add zeros
63 for f in features:
64 if f not in test.columns:
65 test[f] = 0.0
66
67 X = train[features].reset_index(drop=True)
68 y = train[TARGET].values
69 X_test = test[features].reset_index(drop=True)
70 test_ids = test[ID_COL].values
71
72 # Fill NA just in case
73 X = X.fillna(-999)
74 X_test = X_test.fillna(-999)
75
76 # Prepare CV
77 skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
78 oof = np.zeros(len(X))
79 test_preds = np.zeros(len(X_test))
80
81 fold_aucs = []
82
83 # LGBMClassifier params (sklearn API) - remove early stopping related args
84 model_params = {
85 "objective": "binary",
86 "boosting_type": "gbdt",
```

```python
96 }
97 "learning_rate": 0.05,
"n_estimators": 1000, # train full number of trees (no early stopping in this environment)
"num_leaves": 31,
"min_child_samples": 20,
"subsample": 0.8,
"colsample_bytree": 0.8,
"random_state": SEED,
"n_jobs": -1,
"verbosity": -1,
}
print(
99 "Starting CV training with LGBMClassifier (no early stopping due to environment limitations)..."
100 )
101
102 for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
103 print(f"\nFold {fold}")
104 X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
105 y_tr, y_val = y[tr_idx], y[val_idx]
106
107 clf = LGBMClassifier(**model_params)
108
109 # Fit without unsupported kwargs (no eval_set / early_stopping_rounds)
110 clf.fit(X_tr, y_tr)
111
112 # Predict validation
113 val_pred = clf.predict_proba(X_val)[:, 1]
114 oof[val_idx] = val_pred
115 fold_auc = roc_auc_score(y_val, val_pred)
116 fold_aucs.append(fold_auc)
117 print(f"Fold {fold} AUC: {fold_auc:.6f}")
118
119 # Predict test
120 test_pred = clf.predict_proba(X_test)[:, 1]
121 test_preds += test_pred / N_SPLITS
122
123 # Overall OOF AUC
124 oof_auc = roc_auc_score(y, oof)
125 print("\nCross-validation results:")
print(f" f f ")
127 print(f"Mean fold AUC: {np.mean(fold_aucs):.6f} Std: {np.std(fold_aucs):.6f}")
128
129 # Prepare submission
130 submission = pd.DataFrame({"id": test_ids, "rainfall": test_preds})
131 submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
132 submission.to_csv(submission_path, index=False)
133 # also save a copy in working for safety
134 submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
135
136 print(f"\nSaved submission to {submission_path}")
137 print("Submission head:")
138 print(submission.head())
139
140 # Print OOF AUC explicitly as the main evaluation metric
141 print(f"\nFinal reported OOF ROC AUC: {oof_auc:.6f}")
```  
B.4.2 S5E3 seed 6 — Treatment (step 26, OOS = 0.952)

The treatment’s 31 valid iterations enable it to discover: (1) a 3-model ensemble (XGBoost + 2 LightGBM variants), (2) Platt-scaling calibration on each model’s OOF predictions, (3) greedy coordinate-ascent weight optimization in both probability-space and rank-space, and (4) automatic selection of the better ensemble via calibrated OOF AUC.

Listing 2: Treatment best solution: S5E3 seed 6 (437 lines). 3-model ensemble with Plattscaling calibration greedy weight optimization and automatic ensemble selection.

```python
#!/usr/bin/env python3
2 import os
3 import json
4 from pathlib import Path
5 import warnings
6 import numpy as np
import pandas as pd
8 from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
10 from sklearn.linear_model import LogisticRegression
11 import xgboost as xgb
```

```python
12 import lightgbm as lgb
13
14 warnings.filterwarnings("ignore")
15
16 # Config
SEED = 42
18 N_SPLITS = 5
THREADS = int(os.getenv("AIDE_NUM_THREADS", "22"))
INPUT_DIR = Path("input")
21 WORKING_DIR = Path("working")
22 WORKING_DIR.mkdir(parents=True, exist_ok=True)
23
24 TRAIN_PATH = INPUT_DIR / "train.csv"
25 TEST_PATH = INPUT_DIR / "test.csv"
26 SUBMISSION_PATH = WORKING_DIR / "submission.csv"
27
# Model hyperparameters (compact, tuned for CPU)
29 XGB_PARAMS = {
"objective": "binary:logistic",
"eval_metric": "auc",
"verbosity": 0,
"seed": SEED,
"eta": 0.03,
"max_depth": 6,
"subsample": 0.8,
"colsample_bytree": 0.8,
"nthread": THREADS,
"tree_method": "hist",
40 7
41 XGB NUM ROUNDS = 1000
XGB_ESR = 50
43
44 LGB_PARAMS_A = {
"objective": "binary",
"metric": "auc",
"learning_rate": 0.03,
48 "num_leaves": 31,
"feature_fraction": 0.8,
"bagging_fraction": 0.8,
"bagging_freq": 1,
"min_data_in_leaf": 20,
"verbose": -1,
"seed": SEED,
"num_threads": THREADS,
}
57 LGB_PARAMS_B = {
"objective": "binary",
"metric": "auc",
"learning_rate": 0.02,
"num_leaves": 64,
"feature_fraction": 0.7,
"bagging_fraction": 0.7,
"bagging_freq": 1,
"min_data_in_leaf": 15,
<sup>66</sup> <sub>67</sub> "min_sum_hessian_in_leaf": 1e-3,
"verbose": -1,
68 "seed": SEED + 1,
69 "num_threads": THREADS,
LGB_NUM_ROUNDS = 1000
72 LGB_ESR = 50
73
74 # Load data
75 train = pd.read_csv(TRAIN_PATH)
76 test = pd.read_csv(TEST_PATH)
77
78 TARGET = "rainfall"
79 IDCOL = "id"
80
81 # Feature setup
drop_cols = [IDCOL, TARGET]
83 features = [c for c in train.columns if c not in drop_cols]
84
85 # Cyclical features for 'day' and 'winddirection' if present
86 if "day" in features:
period = 365.0
train["day_sin"] = np.sin(2 * np.pi * train["day"] / period)
train["day_cos"] = np.cos(2 * np.pi * train["day"] / period)
test["day_sin"] = np.sin(2 * np.pi * test["day"] / period)
test["day_cos"] = np.cos(2 * np.pi * test["day"] / period)
features = [f for f in features if f != "day"] + ["day_sin", "day_cos"]
```

```python
93
94 if "winddirection" in features:
95 period = 360.0
96 train["winddir_sin"] = np.sin(2 * np.pi * train["winddirection"] / period)
97 train["winddir_cos"] = np.cos(2 * np.pi * train["winddirection"] / period)
98 test["winddir_sin"] = np.sin(2 * np.pi * test["winddirection"] / period)
99 test["winddir_cos"] = np.cos(2 * np.pi * test["winddirection"] / period)
100 features = [f for f in features if f != "winddirection"] + [
101 "winddir_sin",
102 "winddir_cos",
103 ]
104
105 # Ensure features present in both
106 features = [f for f in features if f in train.columns and f in test.columns]
107
108 # Missing count feature
109 train["_missing_count"] = train[features].isnull().sum(axis=1)
110 test["_missing_count"] = test[features].isnull().sum(axis=1)
111
112 # Convert object columns to category codes safely
113 for col in features:
114 if train[col].dtype == "object":
115 combined = pd.concat([train[col], test[col]], axis=0).astype("category")
116 train[col] = combined.iloc[: len(train)].cat.codes
117 test[col] = combined.iloc[len(train) :].cat.codes
118
119 # Numeric columns and median imputation
120 train_nums = train[features].select_dtypes(include=[np.number]).columns.tolist()
121 medians = train[train_nums].median()
122 train[train_nums] = train[train_nums].fillna(medians)
123 test[train_nums] = test[train_nums].fillna(medians)
124
125 # Row-level stats
126 train["_row_mean"] = train[train_nums].mean(axis=1)
127 train["_row_std"] = train[train_nums].std(axis=1).fillna(0.0)
128 test["_row_mean"] = test[train_nums].mean(axis=1)
129 test["_row_std"] = test[train_nums].std(axis=1).fillna(0.0)
130
131 # Per-column percentile rank aggregated as a row feature.
132 rank_cols = train_nums.copy()
133 if len(rank_cols) > 0:
134 combined_ranks = []
135 for col in rank_cols:
136 combined = pd.concat([train[col], test[col]], axis=0)
137 ranks = combined.rank(pct=True, method="average")
138 bi d k d( k l )
139 combined_ranks = np.vstack(combined_ranks).T
140 n_train = train.shape[0]
141 train_ranks = combined_ranks[:n_train]
142 test_ranks = combined_ranks[n_train:]
143 train["_row_rank_mean"] = np.nanmean(train_ranks, axis=1)
144 test["_row_rank_mean"] = np.nanmean(test_ranks, axis=1)
145 else:
146 train["_row_rank_mean"] = 0.0
147 test["_row_rank_mean"] = 0.0
148
149 engineered = ["_missing_count", "_row_mean", "_row_std", "_row_rank_mean"]
150 for f in engineered:
151 if f not in features:
152 features.append(f)
153
154 features = [f for f in features if f in train.columns and f in test.columns]
155
156 # Remove constant features
157 const_feats = [f for f in features if train[f].nunique() <= 1]
158 if const_feats:
159 features = [f for f in features if f not in const_feats]
160
161 # Remove near-duplicate features by very high correlation (>0.999)
162 if len(features) > 1:
163 corr_matrix = train[features].corr().abs()
164 upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
165 to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
166 if to_drop:
167 features = [f for f in features if f not in to_drop]
168
169 features = [f for f in features if f in train.columns and f in test.columns]
170
171 X = train[features].copy()
172 X_test = test[features].copy()
173 y = train[TARGET].astype(int).values
```

```python
174 ids_test = test[IDCOL].values
175
176 # Cross-validation splitter
177 min_class_count = pd.Series(y).value_counts().min()
178 if min_class_count >= N_SPLITS:
179 cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
180 else:
181 cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
182
183 X_values = X.values
184 X_test_values = X_test.values
185
186 # Base models: XGBoost, LGB A, LGB B (single-seed to save time)
187 model_keys = ["xgb", "lgb_a", "lgb_b"]
188 oof_preds = {k: np.zeros(X_values.shape[0], dtype=float) for k in model_keys}
189 test_preds_avg = {k: np.zeros(X_test_values.shape[0], dtype=float) for k in model_keys}
190
191 fold_info = []
192
193 for fold, (tr_idx, val_idx) in enumerate(cv.split(X_values, y), start=1):
194 X_tr, X_val = X_values[tr_idx], X_values[val_idx]
195 y_tr, y_val = y[tr_idx], y[val_idx]
196
197 pos = int(y_tr.sum())
198 neg = int(y_tr.shape[0] - pos)
199 scale_pos_weight = max(1.0, neg / max(1.0, pos))
200
201 # XGBoost
202 xgb_params = XGB_PARAMS.copy()
203 xgb_params["scale_pos_weight"] = scale_pos_weight
204 xgb_params["seed"] = int(SEED)
205 dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=features)
206 dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
207 dtest = xgb.DMatrix(X_test_values, feature_names=features)
208 xgb_model = xgb.train(
209 xgb_params,
210 dtrain,
211 num_boost_round=XGB_NUM_ROUNDS,
212 evals=[(dtrain, "train"), (dval, "val")],
213 early_stopping_rounds=XGB_ESR,
214 verbose_eval=False,
215 )
216 if hasattr(xgb_model, "best_iteration") and xgb_model.best_iteration is not None:
217 xgb_rounds = int(xgb_model.best_iteration) + 1
218 else:
219 xgb_rounds = XGB_NUM_ROUNDS
220 val_pred_xgb = xgb_model.predict(dval, iteration_range=(0, xgb_rounds))
221 test_pred_xgb = xgb_model.predict(dtest, iteration_range=(0, xgb_rounds))
222 oof_preds["xgb"][val_idx] = val_pred_xgb
223 test_preds_avg["xgb"] += test_pred_xgb / N_SPLITS
224
225 # LightGBM A
226 lgb_params_a = LGB_PARAMS_A.copy()
227 lgb_params_a["scale_pos_weight"] = scale_pos_weight
228 lgb_params_a["seed"] = int(SEED)
229 ltrain = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
230 lval = lgb.Dataset(X_val, label=y_val, reference=ltrain, feature_name=features)
231 lgb_model_a = lgb.train(
232 lgb_params_a,
233 ltrain,
234 num_boost_round=LGB_NUM_ROUNDS,
235 valid_sets=[ltrain, lval],
236 valid_names=["train", "val"],
237 callbacks=[lgb.early_stopping(stopping_rounds=LGB_ESR)],
238 )
239 lgb_a_iter = (
240 lgb_model_a.best_iteration
241 if hasattr(lgb_model_a, "best_iteration")
242 else LGB_NUM_ROUNDS
243 )
244 val_pred_lgb_a = lgb_model_a.predict(X_val, num_iteration=lgb_a_iter)
245 test_pred_lgb_a = lgb_model_a.predict(X_test_values, num_iteration=lgb_a_iter)
246 oof_preds["lgb_a"][val_idx] = val_pred_lgb_a
247 test_preds_avg["lgb_a"] += test_pred_lgb_a / N_SPLITS
248
249 # LightGBM B
250 lgb_params_b = LGB_PARAMS_B.copy()
251 lgb_params_b["scale_pos_weight"] = scale_pos_weight
252 lgb_params_b["seed"] = int(SEED + 1)
253 ltrain_b = lgb.Dataset(X_tr, label=y_tr, feature_name=features)
254 lval_b = lgb.Dataset(X_val, label=y_val, reference=ltrain_b, feature_name=features)
```

```python
255 lgb_model_b = lgb.train(
256 lgb_params_b,
257 ltrain_b,
258 num_boost_round=LGB_NUM_ROUNDS,
259 valid_sets=[ltrain_b, lval_b],
260 valid_names=["train", "val"],
261 callbacks=[lgb.early_stopping(stopping_rounds=LGB_ESR)],
262 )
263 lgb_b_iter = (
264 lgb_model_b.best_iteration
265 if hasattr(lgb_model_b, "best_iteration")
266 else LGB_NUM_ROUNDS
267 )
268 val_pred_lgb_b = lgb_model_b.predict(X_val, num_iteration=lgb_b_iter)
269 test_pred_lgb_b = lgb_model_b.predict(X_test_values, num_iteration=lgb_b_iter)
270 oof_preds["lgb_b"][val_idx] = val_pred_lgb_b
271 test_preds_avg["lgb_b"] += test_pred_lgb_b / N_SPLITS
272
273 fold_auc_vals = tuple(
274 float(roc_auc_score(y_val, oof_preds[k][val_idx])) for k in model_keys
275 )
276 print(
277 f"Fold {fold}:
278 ", ".join(
279 [f"{k.upper()} AUC={a:.6f}" for k, a in zip(model_keys, fold_auc_vals)]
280 )
281 )
282 fold_info.append(fold_auc_vals)
283
284 # Base OOF AUCs
285 auc_oof = {k: float(roc_auc_score(y, oof_preds[k])) for k in model_keys}
286 print("Base OOF AUCs:", auc_oof)
287
288 # Calibrate each base with Platt-scaling (LogisticRegression) on OOF
289 calibrators = {}
290 calibrated_oof = {}
291 calibrated_test = {}
292 for key in model_keys:
293 clf = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=SEED)
294 preds = oof_preds[key].reshape(-1, 1)
295 clf.fit(preds, y)
296 calibrated = clf.predict_proba(preds)[:, 1]
297 calibrated_oof[key] = calibrated
298 calibrated_test[key] = clf.predict_proba(test_preds_avg[key].reshape(-1, 1))[:, 1]
299 calibrators[key] = clf
300 auc_cal = float(roc_auc_score(y, calibrated))
301 print(f"Calibrated OOF AUC for {key}: {auc_cal:.6f}")
302
303 # Prepare rank-space OOFs (percentile ranks of calibrated probs)
304 calibrated_oof_rank = {}
305 for k in model_keys:
306 calibrated_oof_rank[k] = pd.Series(calibrated_oof[k]).rank(pct=True).values
307
308
309 # Greedy coordinate-ascent weight search function
310 def greedy_optimize(keys, oof_dict, y, init=None, max_iters=200):
311 n = len(keys)
312 if init is None:
313 weights = np.array([1.0 / n] * n, dtype=float)
314 else:
315 weights = np.array(init, dtype=float)
316 if weights.sum() <= 0:
317 weights = np.array([1.0 / n] * n, dtype=float)
318 else:
319 weights = weights / weights.sum()
320 best_auc = roc_auc_score(y, sum(weights[i] * oof_dict[keys[i]] for i in range(n)))
321 improved = True
322 iters = 0
323 while improved and iters < max_iters:
324 improved = False
325 iters += 1
326 for i in range(n):
327 for delta in [0.1, 0.05, 0.02, 0.01, -0.01, -0.02, -0.05, -0.1]:
328 w_new = weights.copy()
329 w_new[i] = max(0.0, w_new[i] + delta)
330 if w_new.sum() <= 0:
331 continue
332 w_new = w_new / w_new.sum()
333 ensemble_oof = sum(w_new[j] * oof_dict[keys[j]] for j in range(n))
334 auc = roc_auc_score(y, ensemble_oof)
335 if auc > best_auc + 1e-9:
```

336 best\_auc = auc   
337 weights = w\_new   
338 improved = True   
339 return weights, best\_auc   
340   
341   
342 keys = model\_keys.copy()   
343   
344 # Optimize in probability-space   
345 weights\_prob, auc\_prob = greedy\_optimize(keys, calibrated\_oof, y, max\_iters=200)   
346 print(   
347 f"Optimized weights (prob-space): {dict(zip(keys, weights\_prob.round(4)))} OOF AUC: {auc\_prob:.6f}"   
348 )   
349   
350 # Optimize in rank-space   
351 weights\_rank, auc\_rank = greedy\_optimize(keys, calibrated\_oof\_rank, y, max\_iters=200)   
352 print(   
353 f"Optimized weights (rank-space): {dict(zip(keys, weights\_rank.round(4)))} OOF AUC: {auc\_rank:.6f}"   
354 )   
355   
356 # Build ensembles   
357 oof\_prob\_ens = sum(weights\_prob[i] \* calibrated\_oof[keys[i]] for i in range(len(keys)))   
358 test\_prob\_ens = sum(   
359 weights\_prob[i] \* calibrated\_test[keys[i]] for i in range(len(keys))   
360 )   
361   
362 oof\_rank\_ens = sum(   
363 weights\_rank[i] \* calibrated\_oof\_rank[keys[i]] for i in range(len(keys))   
364 )   
365 # convert rank ensemble to pseudo-prob by scaling between 0-1 (already 0-1)   
366 test\_rank\_parts = {}   
367 for k in keys:   
368 # get percentile ranks for test calibrated probs   
369 test\_rank\_parts[k] = pd.Series(calibrated\_test[k]).rank(pct=True).values   
370 test\_rank\_ens = sum(   
371 weights\_rank[i] \* test\_rank\_parts[keys[i]] for i in range(len(keys))   
372 )   
373   
374 # Evaluate which ensemble is better by OOF AUC   
375 auc\_prob\_raw = roc\_auc\_score(y, oof\_prob\_ens)   
376 auc\_rank\_raw = roc\_auc\_score(y, oof\_rank\_ens)   
377 print(   
378 f"Raw ensemble OOF AUCs -> prob-space: {auc\_prob\_raw:.6f}, rank-space: {auc\_rank\_raw:.6f}"   
379 )   
380   
381 # Final Platt calibration for both ensembles   
382 final\_cal\_prob = LogisticRegression(solver="lbfgs", max\_iter=2000, random\_state=SEED)   
383 final\_cal\_prob.fit(oof\_prob\_ens.reshape(-1, 1), y)   
384 oof\_prob\_cal = final\_cal\_prob.predict\_proba(oof\_prob\_ens.reshape(-1, 1))[:, 1]   
385 test\_prob\_cal = final\_cal\_prob.predict\_proba(test\_prob\_ens.reshape(-1, 1))[:, 1]   
386 auc\_prob\_cal = roc\_auc\_score(y, oof\_prob\_cal)   
387 print(f"Final calibrated prob-space ensemble OOF AUC: {auc\_prob\_cal:.6f}")   
388   
389 final\_cal\_rank = LogisticRegression(solver="lbfgs", max\_iter=2000, random\_state=SEED)   
390 final\_cal\_rank.fit(oof\_rank\_ens.reshape(-1, 1), y)   
391 oof\_rank\_cal = final\_cal\_rank.predict\_proba(oof\_rank\_ens.reshape(-1, 1))[:, 1]   
392 test\_rank\_cal = final\_cal\_rank.predict\_proba(test\_rank\_ens.reshape(-1, 1))[:, 1]   
393 auc\_rank\_cal = roc\_auc\_score(y, oof\_rank\_cal)   
394 print(f"Final calibrated rank-space ensemble OOF AUC: {auc\_rank\_cal:.6f}")   
395   
396 # Choose best   
397 if auc\_prob\_cal >= auc\_rank\_cal:   
398 chosen\_name = "prob\_space\_ensemble"   
399 oof\_final = oof\_prob\_cal   
400 test\_final = test\_prob\_cal   
401 chosen\_auc = auc\_prob\_cal   
402 else:   
403 chosen\_name = "rank\_space\_ensemble"   
404 oof\_final = oof\_rank\_cal   
405 test\_final = test\_rank\_cal   
406 chosen\_auc = auc\_rank\_cal   
407   
408 print(f"Chosen ensemble: {chosen\_name} with OOF AUC: {chosen\_auc:.6f}")   
409   
410 # Compute per-fold AUCs for chosen final predictions   
411 fold\_scores = []   
412 for fold, (\_, val\_idx) in enumerate(cv.split(X\_values, y), start=1):   
413 fold\_auc = float(roc\_auc\_score(y[val\_idx], oof\_final[val\_idx]))   
414 fold\_scores.append(fold\_auc)   
415 print(f"Fold {fold} AUC (final): {fold\_auc:.6f}")   
416

```python
417 cv_mean = float(np.mean(fold_scores))
418 cv_std = float(np.std(fold_scores))
419 print(f"Final CV mean AUC: {cv_mean:.6f} std: {cv_std:.6f}")
420
421 # Save submission
422 submission = pd.DataFrame({IDCOL: ids_test, TARGET: test_final})
423 submission.to_csv(SUBMISSION_PATH, index=False)
424 print(f"Saved submission to: {SUBMISSION_PATH}")
425
426 # Output AIDE metrics line
427 aide_metrics = {
428 "valid": True,
429 "lower_is_better": False,
430 "cv_mean": cv_mean,
431 "cv_std": cv_std,
432 "cv_folds": [float(f) for f in fold_scores],
433 }
434 print("AIDE_METRICS_JSON=" + json.dumps(aide_metrics))
435
436 # Print final metric line
437 print(f"Final CV mean AUC: {cv_mean:.6f}")
```

## B.4.3 Wine Quality Seed 1 — Baseline (step 7, OOS = 0.299)

The baseline uses LGBMRegressor with naive rounding—a regression formulation that discards inter-class probability information, fundamentally limiting QWK.

Listing 3: Baseline best solution: Wine Quality seed 1. LGBMRegressor with regression formulation and naive rounding.

```python
1 import os
2 import numpy as np
3 import pandas as pd
4 from sklearn.model_selection import StratifiedKFold
5 import lightgbm as lgb
6 from sklearn.metrics import confusion_matrix
7
8 # Set random seed
9 RANDOM_STATE = 42
10 np.random.seed(RANDOM_STATE)
11
12 # Paths
13 INPUT_DIR = "./input"
14 TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
15 TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
16 SUBMISSION_DIR = "./submission"
17 SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
18
19 # Read data
20 train = pd.read_csv(TRAIN_PATH)
21 test = pd.read_csv(TEST_PATH)
22
23 # Features and target
24 feature_cols = [c for c in train.columns if c not in ("Id", "quality")]
25 X = train[feature_cols].copy()
26 y = train["quality"].astype(int).copy()
27 X_test = test[feature_cols].copy()
28 test_ids = test["Id"].astype(int).copy()
29
30 # Determine label range
31 label_min = int(y.min())
32 label_max = int(y.max())
33 labels_sorted = np.arange(label_min, label_max + 1)
<sup>34</sup><sub>35</sub>
36 # Quadratic Weighted Kappa implementation
def quadratic_weighted_kappa(y_true, y_pred, min_rating=None, max_rating=None):
Compute Quadratic Weighted Kappa (QWK)
if min_rating is None:
min_rating = min(int(np.min(y_true)), int(np.min(y_pred)))
if max_rating is None:
max_rating = max(int(np.max(y_true)), int(np.max(y_pred)))
y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred, dtype=int)
num_ratings = int(max_rating - min_rating + 1)
# Confusion matrix O
```

```python
O = np.zeros((num_ratings, num_ratings), dtype=float)
for a, b in zip(y_true, y_pred):
O[a - min_rating, b - min_rating] += 1
# Histogram of ratings
hist_true = O.sum(axis=1)
hist_pred = O.sum(axis=0)
# Expected matrix E
E = np.outer(hist_true, hist_pred)
if E.sum() == 0:
return 1.0
E = E / E.sum() * O.sum()
# Weight matrix
(( i i ) d float)
for i in range(num_ratings):
63 for j in range(num_ratings):
<sup>64</sup> <sub>65</sub> W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
# QWK
66 num = (W * O).sum()
67 den = (W * E).sum()
68 if den == 0:
69 return 1.0
70 return 1.0 - num / den
71
73 # 5-fold stratified by label
74 n_splits = 5
75 skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
76
77 oof_preds = np.zeros(len(X), dtype=float)
78 test_preds = np.zeros(len(X_test), dtype=float)
79 fold_qwks = []
80
81 # LightGBM parameters - use scikit-learn API without early_stopping_rounds in fit
82 lgb_params = {
"objective": "regression",
"boosting_type": "gbdt",
"learning_rate": 0.05,
"n_estimators": 800, # fixed number of trees; no early stopping in fit to avoid compatibility issues
"random_state": RANDOM_STATE,
"num_leaves": 31,
89 "subsample": 0.8,
90 "colsample_bytree": 0.8,
91 "reg_alpha": 0.0,
92 "reg_lambda": 1.0,
93 "verbosity": -1,
94 }
95
96 print("Starting 5-fold CV training (no early stopping in fit)...")
97 for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
98 X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
99 y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
100 model = lgb.LGBMRegressor(**lgb_params)
101 # Fit without early_stopping_rounds to avoid the TypeError seen previously
102 model.fit(
103 X_train,
104 y_train,
105 eval_set=[(X_val, y_val)],
106 eval_metric="rmse",
107 )
108 # Predict (do not use num_iteration parameter which may not be available)
109 val_pred = model.predict(X_val)
110 oof_preds[val_idx] = val_pred
111 test_pred = model.predict(X_test)
112 test_preds += test_pred / n_splits
113
114 # Convert to discrete for QWK
115 val_pred_round = np.rint(val_pred).astype(int)
116 val_pred_round = np.clip(val_pred_round, label_min, label_max)
117 qwk = quadratic_weighted_kappa(
118 y_val.values, val_pred_round, min_rating=label_min, max_rating=label_max
119 )
120 fold_qwks.append(qwk)
121 print(f"Fold {fold} QWK: {qwk:.5f}")
122
123 # Overall OOF QWK
124 oof_preds_round = np.rint(oof_preds).astype(int)
125 oof_preds_round = np.clip(oof_preds_round, label_min, label_max)
126 oof_qwk = quadratic_weighted_kappa(
127 y.values, oof_preds_round, min_rating=label_min, max_rating=label_max
128 )
129 print(f"OOF QWK: {oof_qwk:.5f}")
```

130 print(f"Mean per-fold QWK: {np.mean(fold\_qwks):.5f} (std {np.std(fold\_qwks):.5f})")   
131   
132 # Prepare final test predictions   
133 final\_test\_preds = np.rint(test\_preds).astype(int)   
134 final\_test\_preds = np.clip(final\_test\_preds, label\_min, label\_max)   
135   
136 # Create submission dataframe   
137 submission\_df = pd.DataFrame({"Id": test\_ids.values, "quality": final\_test\_preds})   
138 # Ensure submission directory exists   
139 os.makedirs(SUBMISSION\_DIR, exist\_ok=True)   
140 submission\_df.to\_csv(SUBMISSION\_PATH, index=False)   
141   
142 print(f"Saved submission to {SUBMISSION\_PATH}")   
print("Sample of submission:")   
144 print(submission\_df.head(10).to\_string(index=False))

## B.4.4 Wine Quality Seed 1 — Treatment (step 38, OOS = 0.398)

With 51 valid iterations, the treatment discovers that a diverse bagged LightGBM ensemble with stacking (Ridge + Isotonic regression on out-of-fold predictions) produces substantially higher QWK than the baseline’s single-model regression approach.

Listing 4: Treatment best solution: Wine Quality seed 1. Diverse bagged LightGBM ensemble with stacking.

```python
1 #!/usr/bin/env python3
import os
3 import json
4 import math
5 import numpy as np
6 import pandas as pd
from itertools import combinations
8 from sklearn.model_selection import StratifiedKFold, KFold
9 from sklearn.metrics import cohen_kappa_score
10 from sklearn.linear_model import Ridge
11 from sklearn.isotonic import IsotonicRegression
12 import lightgbm as lgb
13 import warnings
14
15 warnings.filterwarnings("ignore")
16
17 # Config
18 SEED = 42
19 NUM_FOLDS = 5
20 LGB_ROUNDS = 2000
21 EARLY_STOPPING = 100
22 THREADS = int(os.getenv("AIDE_NUM_THREADS", "22"))
23 INPUT_DIR = "input"
24 TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
25 TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
26 SUBMISSION_PATH = os.path.join("working", "submission.csv")
27
28 np.random.seed(SEED)
29
30 # Load data
31 train = pd.read_csv(TRAIN_PATH)
32 test = pd.read_csv(TEST_PATH)
33
34 FEATURES = [c for c in train.columns if c not in ("id", "quality")]
35 X_orig = train[FEATURES].copy()
36 X_test_orig = test[FEATURES].copy()
test_ids = test["id"].astype(int).values
38
39 y_raw = train["quality"].astype(int).values
40 unique_classes = np.sort(train["quality"].unique())
41 n_classes = len(unique_classes)
42 n_train = len(train)
43 n_test = len(test)
44
45
46 # Safe feature engineering (no target leakage)
47 def add_features(df):
df2 = df.copy()
for c in df.columns:
col = df[c]
if pd.api.types.is_numeric_dtype(col):
if (col > 0).all():
```

59 df2[c + "\_log1p"] = np.log1p(col)   
else:   
df2[c + "\_rankpct"] = col.rank(pct=True).astype(float)   
# square   
df2[c + "\_sq"] = col.values \* col.values   
return df2   
60   
61 X = add\_features(X\_orig)   
62 X\_test = add\_features(X\_test\_orig)   
64 # pairwise products among top-k variance original features (produce train/test separately)   
65 num\_feats = [c for c in X\_orig.columns if pd.api.types.is\_numeric\_dtype(X\_orig[c])]   
66 variances = [(c, X\_orig[c].var()) for c in num\_feats]   
variances.sort(key=lambda x: x[1], reverse=True)   
68 top\_k = min(5, len(variances))   
69 top\_features = [c for c, \_ in variances[:top\_k]]   
70   
71 for a, b in combinations(top\_features, 2):   
75 name = f"{a}\_x\_{b}"   
X[name] = X\_orig[a].values \* X\_orig[b].values   
X\_test[name] = X\_test\_orig[a].values \* X\_test\_orig[b].values   
76 FEATURES\_FE = [c for c in X.columns]   
78 # CV splitter safe (stratify if every class has enough samples)   
79 use\_strat = True   
80 for cls in unique\_classes:   
81 if (y\_raw == cls).sum() < NUM\_FOLDS:   
<sup>82</sup> <sub>83</sub> use\_strat = False   
break   
84   
85 if use\_strat:   
86 kf = StratifiedKFold(n\_splits=NUM\_FOLDS, shuffle=True, random\_state=SEED)   
87 splits = list(kf.split(X, y\_raw))   
88 else:   
89 kf = KFold(n\_splits=NUM\_FOLDS, shuffle=True, random\_state=SEED)   
90 splits = list(kf.split(X))   
91   
92 # Base LightGBM variants and seeds (bagging)   
93 lgb\_variants = [   
{   
"objective": "regression",   
"metric": "rmse",   
"learning\_rate": 0.05,   
"num\_leaves": 31,   
99 "max\_depth": 6,   
100 "feature\_fraction": 0.8,   
101 "bagging\_fraction": 0.8,   
102 "bagging\_freq": 1,   
103 104 },<sub>{</sub>   
105 "objective": "regression",   
106 "metric": "rmse",   
107 "learning\_rate": 0.03,   
108 "num\_leaves": 63,   
109 "max\_depth": 8,   
110 "feature\_fraction": 0.7,   
111 "bagging\_fraction": 0.7,   
112 "bagging\_freq": 1,   
113 },   
114 {   
115 "objective": "regression",   
116 "metric": "rmse",   
117 "learning\_rate": 0.07,   
118 "num\_leaves": 24,   
119 "max\_depth": 5,   
120 "feature\_fraction": 0.9,   
121 "bagging\_fraction": 0.9,   
122 "bagging\_freq": 1,   
123 },   
124 ]   
125 lgb\_seeds = [SEED, SEED + 101]   
126   
127 # Build model configs (only LGB for speed and robustness)   
128 model\_configs = []   
129 for vid, var in enumerate(lgb\_variants):   
130 for sd in lgb\_seeds:   
131 model\_configs.append(("lgb", vid, int(sd)))   
132 n\_models = len(model\_configs)   
133

```python
134 # Storage
135 oof_stack = np.zeros((n_train, n_models), dtype=float)
136 test_stack_sum = np.zeros((n_test, n_models), dtype=float)
137 test_stack_count = np.zeros(n_models, dtype=int)
138
139 # Train base models in CV
140 print("Training base LightGBM ensemble...")
for fold, (tr_idx, val_idx) in enumerate(splits):
142 print(f" Fold {fold+1}/{NUM_FOLDS}")
143 X_tr = X.iloc[tr_idx].reset_index(drop=True)
144 X_val = X.iloc[val_idx].reset_index(drop=True)
145 y_tr = y_raw[tr_idx]
146 y_val = y_raw[val_idx]
147
148 for m_idx, (mtype, vid, sd) in enumerate(model_configs):
149 # Train LGB
150 params = lgb_variants[vid].copy()
151 params.update(
152 {
153 "seed": int(sd + fold),
154 "verbosity": -1,
155 "num_threads": THREADS,
156 }
157 )
158 lgb_train = lgb.Dataset(X_tr, label=y_tr)
159 lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
160 model = lgb.train(
161 params,
162 lgb_train,
163 num_boost_round=LGB_ROUNDS,
164 valid_sets=[lgb_train, lgb_valid],
165 valid_names=["train", "valid"],
166 callbacks=[
167 lgb.early_stopping(stopping_rounds=EARLY_STOPPING),
168 lgb.log_evaluation(period=0),
169 ],
170 )
171 best_iter = getattr(model, "best_iteration", None)
172 if best_iter is None:
173 # fallback to current_iteration
174 try:
175 best_iter = model.current_iteration()
176 except Exception:
177 best_iter = LGB_ROUNDS
178 # predictions
179 val_pred = model.predict(X_val, num_iteration=best_iter)
180 test_pred = model.predict(X_test, num_iteration=best_iter)
181
182 oof_stack[val_idx, m_idx] = val_pred
183 test_stack_sum[:, m_idx] += test_pred
184 test_stack_count[m_idx] += 1
185
186 # Finalize per-model test stacks (average)
187 test_stack = np.zeros_like(test_stack_sum)
for m in range(n_models):
189 cnt = test_stack_count[m]
190 if cnt > 0:
191 test_stack[:, m] = test_stack_sum[:, m] / float(cnt)
192 else:
193 test_stack[:, m] = 0.0
194
195 # Meta model: Ridge on stacked OOF features
196 meta_oof = np.zeros(n_train, dtype=float)
197 meta_test_preds_folds = np.zeros((NUM_FOLDS, n_test), dtype=float)
198
199 for fold, (tr_idx, val_idx) in enumerate(splits):
200 X_meta_tr = oof_stack[tr_idx]
201 y_meta_tr = y_raw[tr_idx]
202 X_meta_val = oof_stack[val_idx]
203 meta = Ridge(alpha=1.0, random_state=SEED)
204 meta.fit(X_meta_tr, y_meta_tr)
205 meta_oof[val_idx] = meta.predict(X_meta_val)
206 meta_test_preds_folds[fold] = meta.predict(test_stack)
207
208 # Final meta trained on all OOF
209 meta_final = Ridge(alpha=1.0, random_state=SEED)
210 meta_final.fit(oof_stack, y_raw)
211 meta_test_pred = meta_final.predict(test_stack)
212
213 # Isotonic calibration on meta_oof -> y_raw (fit only on OOF)
214 iso = IsotonicRegression(out_of_bounds="clip")
```

```python
215 try:
216 iso.fit(meta_oof, y_raw)
217 meta_oof_cal = iso.predict(meta_oof)
218 meta_test_cal = iso.predict(meta_test_pred)
219 except Exception:
220 meta_oof_cal = meta_oof.copy()
221 meta_test_cal = meta_test_pred.copy()
222
223 # Linear scale+shift calibration on OOF (fit to maximize simple squared alignment but we will still optimize
thresholds for QWK)
224 # Fit linear regression (a*x + b) on OOF to best fit y in least squares (fast, no leakage)
225 a = 1.0
226 b = 0.0
227 try:
228 A = np.vstack([meta_oof_cal, np.ones_like(meta_oof_cal)]).T
229 sol, _, _, _ = np.linalg.lstsq(A, y_raw, rcond=None)
230 a, b = float(sol[0]), float(sol[1])
231 except Exception:
232 a, b = 1.0, 0.0
233
234 meta_oof_cal_ls = a * meta_oof_cal + b
235 meta_test_cal_ls = a * meta_test_cal + b
236
237 # Threshold initialization from class-wise means on calibrated OOF
238 class_means = []
239 for c in unique_classes:
240 mask = y_raw == c
241 if mask.sum() == 0:
242 class_means.append(np.nan)
243 else:
244 class_means.append(meta_oof_cal_ls[mask].mean())
245 class_means = np.array(class_means)
246 nan_mask = np.isnan(class_means)
247 if nan_mask.any():
248 filled = np.linspace(meta_oof_cal_ls.min(), meta_oof_cal_ls.max(), n_classes)
249 class_means[nan_mask] = filled[nan_mask]
250
251 thresholds = np.array(
252 [(class_means[i] + class_means[i + 1]) / 2.0 for i in range(n_classes - 1)],
253 dtype=float,
254 )
255
256
257 def map_preds_to_labels(preds, thresholds, classes):
258 idxs = np.sum(preds.reshape(-1, 1) > thresholds.reshape(1, -1), axis=1)
259 mapped = classes[idxs]
260 return mapped
261
262
263 def qwk_for_thresholds(thr, preds_cal, y_true):
264 preds_mapped = map_preds_to_labels(preds_cal, thr, unique_classes)
265 return cohen_kappa_score(y_true, preds_mapped, weights="quadratic")
266
267
268 best_thr = thresholds.copy()
269 best_score = qwk_for_thresholds(best_thr, meta_oof_cal_ls, y_raw)
270 print(f"Initial calibrated OOF QWK (iso + linfit): {best_score:.6f}")
# Coordinate descent threshold optimization
273 min_pred = float(meta_oof_cal_ls.min())
274 max_pred = float(meta_oof_cal_ls.max())
275 step = (max_pred - min_pred) / 10.0 if max_pred > min_pred else 1.0
276 max_iters = 200
277 iters = 0
278 while step > 1e-6 and iters < max_iters:
279 improved = False
280 for i in range(len(best_thr)):
281 low = min_pred if i == 0 else best_thr[i - 1] + 1e-12
282 high = max_pred if i == len(best_thr) - 1 else best_thr[i + 1] - 1e-12
283 if low >= high:
284 continue
285 current = best_thr[i]
286 candidates = [
287 current,
288 current - step,
289 current + step,
290 low,
291 high,
292 (low + high) / 2.0,
293 ]
294 cand_values = []
```

```python
295 for c in candidates:
296 c = max(low, min(high, c))
297 cand_values.append(c)
298 cand_values = sorted(set(cand_values))
299 best_local_score = best_score
300 best_local_val = current
301 for val in cand_values:
302 trial_thr = best_thr.copy()
303 trial_thr[i] = val
304 if not np.all(np.diff(trial_thr) > -1e-12):
305 continue
306 score = qwk_for_thresholds(trial_thr, meta_oof_cal_ls, y_raw)
307 if score > best_local_score + 1e-12:
308 best_local_score = score
309 best_local_val = val
310 if best_local_val != current:
311 best_thr[i] = best_local_val
312 best_score = best_local_score
313 improved = True
314 if not improved:
315 step /= 3.0
316 iters += 1
317
318 print(f"Optimized thresholds: {best_thr}")
319 print(f"OOF calibrated+linfit QWK after threshold opt: {best_score:.6f}")
320
321 # Compute per-fold QWKs using optimized thresholds on calibrated per-fold meta preds
322 cv_fold_scores = []
323 for fold in range(NUM_FOLDS):
324 val_idx = splits[fold][1]
325 val_meta_pred = meta_oof[val_idx]
326 # apply isotonic then linear scaling
327 val_meta_cal = (
328 iso.predict(val_meta_pred)
329 if isinstance(iso, IsotonicRegression)
330 else val_meta_pred
331 )
332 val_meta_cal = a * val_meta_cal + b
333 y_val = y_raw[val_idx]
334 mapped = map_preds_to_labels(val_meta_cal, best_thr, unique_classes)
335 score = cohen_kappa_score(y_val, mapped, weights="quadratic")
336 cv_fold_scores.append(float(score))
337 print(f"Fold {fold+1} owK after calibration+threshold opt: {score:,6fy"
338
339 cv_mean = float(np.mean(cv_fold_scores))
340 cv_std = float(np.std(cv_fold_scores))
341 print(f"CV mean QWK: {cv_mean:.6f} std: {cv_std:.6f}")
342
343 # Apply to test preds: meta_test_cal_ls
344 test_mapped = map_preds_to_labels(meta_test_cal_ls, best_thr, unique_classes).astype(
345 int
346 )
347 submission = pd.DataFrame({"id": test_ids, "quality": test_mapped})
348 os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
349 submission.to_csv(SUBMISSION_PATH, index=False)
350 print(f"Saved submission to {SUBMISSION_PATH}")
351
352 # AIDE metrics JSON line
353 metrics = {
354 "valid": "quadratic_weighted_kappa",
355 "lower_is_better": False,
356 "cv_mean": cv_mean,
357 "cv_std": cv_std,
358 "cv_folds": cv_fold_scores,
359 }
360 print("AIDE_METRICS_JSON=" + json.dumps(metrics))
```

## B.5 Valid node and score correlation

The case studies above show that more valid nodes enable more iteration and qualitatively better solutions. We now ask whether this relationship holds quantitatively: across all seeds and competitions, do seeds with more valid nodes tend to achieve higher OOS scores?

Per-seed correlation. For each competition, we compute the Pearson correlation (r) between per-seed valid node count and OOS score across all available seeds (10 treatment + up to 10 baseline, approximately 18–19 datapoints per competition). Table 10 reports the results.

<table><tr><td>Competition</td><td>N</td><td>r</td></tr><tr><td>Cirrhosis</td><td>19</td><td>+0.34</td></tr><tr><td>GNSS</td><td>18</td><td>+0.27</td></tr><tr><td>Spaceship</td><td>19</td><td>-0.02</td></tr><tr><td>Wine</td><td>18</td><td>+0.63</td></tr><tr><td>S5E3</td><td>19</td><td>+0.18</td></tr><tr><td>S5E6</td><td>18</td><td>+0.61</td></tr><tr><td>S5E7</td><td>19</td><td>-0.27</td></tr><tr><td>S5E8</td><td>16</td><td>+0.05</td></tr><tr><td>S5E12</td><td>17</td><td>+0.24</td></tr><tr><td>Pooled</td><td>163</td><td>+0.22</td></tr></table>

Table 10: Pearson correlation (r) between per-seed valid node count and OOS score. Cirrhosis scores are negated so that positive r always means “more valid nodes → better score.” Pooled r is computed after z-normalizing both variables within each competition.

The correlation is positive in 7 of 9 competitions, with a z-normalized pooled $r = + 0 . 2 2$ across all 163 seeds, consistent with the hypothesis that more valid nodes lead to better scores. The correlation is highest on Wine $\left( r = + 0 . 6 3 \right)$ and S5E6 $( r = + 0 . 6 1 )$ . For Cirrhosis $( r = + 0 . 3 4 $ , Table 10), more valid nodes correlate with lower (better) log-loss.

## C Hyperparameter optimization materials and additional results and discussion

Hyperparameter optimization directive. The following directive is injected verbatim into the agent context to guide hyperparameter tuning behavior.

HYPERPARAMETER OPTIMIZATION DIRECTIVE

Treat hyperparameter tuning as a sequential search problem under strict time and step   
budgets.   
1. Start with a strong, standard baseline (well-known defaults for the model class).   
Validate to establish a performance baseline.   
2. Run a small number of cheap probes (fast, low-compute experiments) to identify   
sensitivity   
directions (e.g., learning rate, regularization strength, tree depth, batch size).   
Use cross-validation or a held-out validation set, not the test set.   
3. Prioritize hyperparameters with the highest marginal impact on validation score.   
Focus on 1-2 hyperparameters at a time rather than simultaneous grid searches.   
4. Use structured search (log-scale sweeps, conditional ranges), not random guessing.   
For continuous parameters, explore orders of magnitude first.   
5. Aggressively prune unpromising regions early; reallocate budget to promising   
configurations.   
If a hyperparameter range shows no improvement after 2-3 trials, move on.   
6. When improvements plateau (e.g., <0.1% improvement over 3 consecutive trials),   
switch from exploration to fine-grained exploitation around the current best setting.   
7. Always validate changes; only keep configurations that strictly improve   
validation performance over your current best submission.   
8. Track what failed and do not retry equivalent configurations.   
Maintain a log of tried hyperparameter combinations and their results.   
Goal: maximize leaderboard / validation score within the available compute and step   
budget.

Hyperparameter tuning evaluator prompt. The evaluator prompt used to score hyperparameter tuning quality for AIDE and ML-Master use the same 0–3 rubric. Both are followed by a separate, code-level structural check (Appendix C) that can cap the LLM-assigned

score downward for weak or unused search configurations. The HPO quality scorer uses gpt-4o-2024-08-06 as the judge model, independent of the agent backbone.

## AIDE evaluator prompt:

You are a strict ML code reviewer evaluating hyperparameter tuning quality. Score the hyperparameter tuning in the following Python code on a 0-3 scale:

0 = none: No hyperparameter tuning (only fixed/default hyperparameters)

1 = superficial: Minimal tuning (e.g., only 2-3 values tested for 1 hyperparameter, or very small grid/random search with <5 iterations)

2 = moderate: Reasonable tuning (multiple hyperparameters tested, systematic search with >=5 iterations, proper validation)

3 = extensive: Comprehensive tuning (multiple hyperparameters, systematic search with >=10 iterations, proper validation, best params reused for final training)

## DO NOT count:

\- cross\_val\_score alone without hyperparameter search

\- fixed hyperparameters

\- train/val splits without search

\- hyperparameter tuning that is not used in final model training

Respond with ONLY the integer score (0, 1, 2, or 3).

## ML-Master evaluator prompt:

You are a STRICT ML code reviewer.

Evaluate the QUALITY and DEPTH of hyperparameter tuning in the following Python code.

Score hyperparameter tuning on a scale from 0 to 3:

## 0 = NONE

\- No hyperparameter tuning

\- Fixed hyperparameters

\- cross\_val\_score without parameter search

## 1 = MINIMAL

\- Token or superficial tuning

\- Only one hyperparameter tuned over 1-2 values

\- GridSearchCV or RandomizedSearchCV with <5 total configurations

\- RandomizedSearchCV with n\_iter < 5

## 2 = MODERATE

\- Valid hyperparameter tuning but limited in scope

\- Either:

\* One hyperparameter searched over >=3 values, OR

\* Two or more hyperparameters searched over >=2 values each

\- >=5 total configurations evaluated

\- Model selection based on validation or cross-validation

## 3 = EXTENSIVE

\- Systematic, non-trivial hyperparameter optimization

\- Multiple hyperparameters jointly optimized

\- >=10 total configurations or trials evaluated

\- Clear use of GridSearchCV, RandomizedSearchCV (n\_iter >=10), Optuna, Hyperopt,

or Bayesian optimization

\- Best configuration explicitly selected and used

Valid tuning methods include:

\- GridSearchCV

\- RandomizedSearchCV

\- Optuna / Hyperopt / Bayesian optimization

\- Manual loops evaluating multiple configurations

Respond ONLY with a single integer: 0, 1, 2, or 3.

Analysis of asymmetry in AIDE and ML-Master HPO results. The asymmetry can be attributed to two structural differences, which we have confirmed by analyzing the code and logs.

The first is prompt redundancy in ML-Master. Its draft prompt already instructs the agent to include HPO in every draft, with implementation guidelines specifying two-phase training and limiting the hyperparameter search to 10-15 trials. However, AIDE’s draft prompt doesn’t require HPO at draft time the way ML-Master does. Our directive therefore fills a gap in AIDE but adds less in ML-Master, where HPO guidance already exists.

The second is that the directive pushes the LLM toward RandomizedSearchCV and Grid-SearchCV with XGBoost, which triggers a sklearn/XGBoost version incompatibility:

AttributeError: 'super' object has no attribute '\*\*sklearn\_tags\*\*'

[05:20:45] WARNING: Node 6f9868d4 is marked as buggy because response['is\_bug'] is True.

[05:20:45] INFO: Parsed results: Node 6f9868d4 is buggy

[05:20:45] INFO: Starting Debugging Node 6f9868d4.

[05:20:56] INFO: Drafted a new node 3272d605 successfully!

ML-Master’s memory module records these failures neutrally as is\_bug: True without propagating why, so the LLM interprets each crash as motivation to try a different HPO implementation rather than abandon HPO. We observe up to 12 consecutive identical API failures within a single ML-Master run. This explains the pattern across all three conditions: the prompt-only condition adds redundant guidance on top of existing instructions, the code-only condition introduces reward shaping that is undermined by crash-prone code patterns, and the combined condition inherits both issues simultaneously.

In contrast, the following AIDE log confirms that our reward-shaping mechanism functions correctly when the agent’s information flow supports it. A node with HPO score 0 has its reward adjusted down from 0.803 to 0.787 due to the -0.300 penalty, making it less likely to be selected:

[2026-03-26 20:01:27] INFO: Scoring hyperparameter tuning for node 678c47e2...

[2026-03-26 20:01:28] INFO: Node 678c47e2... initial HPO score: 0

[2026-03-26 20:01:28] INFO: Node 678c47e2... HPO score after structural caps: 0

[2026-03-26 20:01:28] INFO: Parsed results: Node 678c47e2... is not buggy

[2026-03-26 20:01:28] INFO: Node 678c47e2... metric adjusted:

base=0.803140, hpo\_reward=-0.300, diversity=0.100, final=0.787077

The asymmetry is itself a finding: scaffold interventions interact with the underlying agent’s memory architecture, and a directive that works on one agent can fail on another for reasons unrelated to the directive’s design.

<table><tr><td>Competition</td><td>∆Prompt</td><td>∆Code</td><td>∆P&amp;C</td></tr><tr><td>Cirrhosis†</td><td> $- 0 . 0 0 1 \pm 0 . 0 0 2 2$ </td><td> $- 0 . 0 0 2 \pm 0 . 0 0 3 2$ </td><td> $- 0 . 0 0 1 \pm 0 . 0 0 6 1$ </td></tr><tr><td>GNSS</td><td> $- 0 . 0 1 9 \pm 0 . 0 1 2 9$ </td><td> $- 0 . 0 1 7 \pm 0 . 0 1 6 2$ </td><td> $- 0 . 0 7 8 \pm 0 . 0 0 3 4$ </td></tr><tr><td>Spaceship</td><td> $\mathbf { + 0 . 0 1 0 \pm 0 . 0 1 4 3 }$ </td><td> $- 0 . 0 1 6 \pm 0 . 0 2 0 8$ </td><td> $+ 0 . 0 2 0 \pm 0 . 0 1 1 7$ </td></tr><tr><td>Wine</td><td> $+ 0 . 0 0 2 \pm 0 . 0 1 0 4$ </td><td> $- 0 . 0 7 5 \pm 0 . 0 6 9 1$ </td><td> $- 0 . 0 5 6 \pm 0 . 0 0 9 8$ </td></tr><tr><td>S5E3</td><td> $- 0 . 0 0 2 \pm 0 . 0 0 3 7$ </td><td> $- 0 . 0 0 4 \pm 0 . 0 0 4 0$ </td><td> $- 0 . 0 5 3 \pm 0 . 0 0 6 2$ </td></tr><tr><td>S5E6</td><td> $- 0 . 0 0 2 \pm 0 . 0 0 3 5$ </td><td> $- 0 . 0 1 1 \pm 0 . 0 0 4 4$ </td><td> $- 0 . 0 2 6 \pm 0 . 0 0 2 1$ </td></tr><tr><td>S5E7</td><td> $+ 0 . 0 0 0 \pm 0 . 0 0 0 3$ </td><td> $+ 0 . 0 0 0 \pm 0 . 0 0 0 7$ </td><td> $\mathbf { + 0 . 0 0 1 } \pm 0 . 0 0 0 3$ </td></tr><tr><td>S5E8</td><td> $- 0 . 0 0 1 \pm 0 . 0 0 2 6$ </td><td> $+ 0 . 0 0 0 \pm 0 . 0 0 0 7$ </td><td> $+ 0 . 0 0 0 \pm 0 . 0 0 0 8$ </td></tr><tr><td>S5E12</td><td> $\mathbf { + 0 . 0 0 1 } \pm 0 . 0 0 4 0$ </td><td> $+ 0 . 0 0 2 \pm 0 . 0 0 4 0$ </td><td> $- 0 . 0 0 0 \pm 0 . 0 0 5 5$ </td></tr></table>

Table 11: Intervention effect on graded score for ML-Master: $\Delta = \mu _ { \mathrm { i n t } } - \mu _ { \mathrm { b a s e } }$ (mean ± SEM of ∆), with the same competitions as Table 2. Bold (signed ∆ only) indicates strict improvement over baseline (positive ∆ when higher is better; negative ∆ for cirrhosis). <sup>†</sup>Lower is better for Cirrhosis.

C.1 Thompson sampling on AIDE: delta, standard deviation and proportion of null errors
<table><tr><td>Competition</td><td>Thompson Sampling</td><td>Baseline (AIDE)</td><td>Winner</td></tr><tr><td>Cirrhosis†</td><td> $0 . 4 0 4 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 3 9 4 \pm 0 . 0 1 2 }$ </td><td>Baseline</td></tr><tr><td>GNSS</td><td> ${ \bf 0 . 9 6 7 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 0 . 9 6 7 \pm 0 . 0 0 1 }$ </td><td>Tie</td></tr><tr><td>Spaceship</td><td> $0 . 8 2 2 \pm 0 . 0 0 2$ </td><td> $0 . 7 9 0 \pm 0 . 0 2 2$ </td><td>TS</td></tr><tr><td>Wine</td><td> ${ \bf 0 . 4 0 8 \pm 0 . 0 1 1 }$ </td><td> $0 . 3 7 5 \pm 0 . 0 2 1$ </td><td>TS</td></tr><tr><td>S5E3</td><td> $\mathbf { 0 . 8 8 9 \pm 0 . 0 0 3 }$ </td><td> $0 . 8 8 7 \pm 0 . 0 0 3$ </td><td>TS</td></tr><tr><td>S5E6</td><td> $0 . 3 3 5 \pm 0 . 0 0 1$ </td><td> $0 . 3 3 3 \pm 0 . 0 0 2$ </td><td>TS</td></tr><tr><td>S5E7</td><td> $0 . 9 6 7 \pm 0 . 0 0 0$ </td><td> $\pm 0 . 9 6 8 \pm 0 . 0 0 0$ </td><td>Baseline</td></tr><tr><td>S5E8</td><td> $0 . 9 6 6 \pm 0 . 0 0 1$ </td><td> ${ \pm 0 . 9 6 9 \pm 0 . 0 0 0 }$ </td><td>Baseline</td></tr><tr><td>S5E12</td><td> $0 . 7 2 5 \pm 0 . 0 0 1$ </td><td> $\pm 0 . 7 2 7 \pm 0 . 0 0 0$ </td><td>Baseline</td></tr></table>

Table 12: AIDE Baseline vs. AIDE with Thompson Sampling (Mean ± SEM) of 10 runs. Bold indicates the winning method per competition; values tied at the displayed precision are bolded in both columns. <sup>†</sup>Lower is better.

<table><tr><td>Type</td><td>initial drafts</td><td>max debug</td><td>error backtrack</td></tr><tr><td>TS</td><td>20</td><td>5</td><td>3</td></tr><tr><td>Baseline</td><td>5</td><td>20</td><td>20</td></tr></table>

Table 13: Hyperparameters for different settings on AIDE.

<table><tr><td>Competition</td><td>Thompson Sampling (Ours)</td><td>Baseline (AIDE)</td></tr><tr><td>Cirrhosis†</td><td>0.014</td><td>0.035</td></tr><tr><td>GNSS</td><td>0.004</td><td>0.002</td></tr><tr><td>Spaceship</td><td>0.006</td><td>0.067</td></tr><tr><td>Wine</td><td>0.034</td><td>0.058</td></tr><tr><td>S5E3</td><td>0.009</td><td>0.010</td></tr><tr><td>S5E6</td><td>0.003</td><td>0.005</td></tr><tr><td>S5E7</td><td>0.001</td><td>0.000</td></tr><tr><td>S5E8</td><td>0.003</td><td>0.001</td></tr><tr><td>S5E12</td><td>0.002</td><td>0.001</td></tr></table>

Table 14: Score Standard Deviation per Competition on AIDE. Bold indicates lower (more consistent) std dev.

<table><tr><td>Competition</td><td>Thompson Sampling</td><td>Baseline (AIDE)</td></tr><tr><td>Cirrhosis†</td><td>10%</td><td>10%</td></tr><tr><td>GNSS</td><td>10%</td><td>20%</td></tr><tr><td>Spaceship</td><td>0%</td><td>10%</td></tr><tr><td>Wine</td><td>10%</td><td>20%</td></tr><tr><td>S5E3</td><td>20%</td><td>10%</td></tr><tr><td>S5E6</td><td>30%</td><td>20%</td></tr><tr><td>S5E7</td><td>0%</td><td>10%</td></tr><tr><td>S5E8</td><td>30%</td><td>40%</td></tr><tr><td>S5E12</td><td>40%</td><td>30%</td></tr></table>

Table 15: Null/Zero Run Rate per Competition. Fraction of runs out of 10 that produced no valid score on AIDE. Bold indicates lower (fewer failures); values tied at the displayed precision are bolded in both columns.

C.2 AIDE: comparative analysis of Thompson sampling versus hyperparameter changes
<table><tr><td>Competition</td><td>AIDE + TS + More Drafts</td><td>AIDE + More Drafts</td><td>Winner</td></tr><tr><td>Cirrhosis†</td><td> $0 . 4 0 4 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 3 9 4 } \pm 0 . 0 0 4$ </td><td>More Drafts</td></tr><tr><td>GNSS</td><td> ${ \bf 0 . 9 6 7 \pm 0 . 0 0 1 }$ </td><td> $0 . 9 6 5 \pm 0 . 0 0 1$ </td><td>TS</td></tr><tr><td>Spaceship</td><td> $\mathbf { 0 . 8 2 2 \pm 0 . 0 0 2 }$ </td><td> $0 . 8 1 8 \pm 0 . 0 0 5$ </td><td>TS</td></tr><tr><td>Wine</td><td> ${ \bf 0 . 4 0 8 \pm 0 . 0 1 1 }$ </td><td> $0 . 4 0 2 \pm 0 . 0 1 7$ </td><td>TS</td></tr><tr><td>S5E3</td><td> $0 . 8 8 9 \pm 0 . 0 0 3$ </td><td> ${ \bf 0 . 8 9 6 \pm 0 . 0 0 2 }$ </td><td>More Drafts</td></tr><tr><td>S5E6</td><td> $0 . 3 3 5 \pm 0 . 0 0 1$ </td><td> ${ \bf 0 . 3 3 7 \pm 0 . 0 0 2 }$ </td><td>More Drafts</td></tr><tr><td>S5E7</td><td> $0 . 9 6 7 \pm 0 . 0 0 0$ </td><td> $0 . 9 6 8 \pm 0 . 0 0 0$ </td><td>More Drafts</td></tr><tr><td>S5E8</td><td> ${ \bf 0 . 9 6 6 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 0 . 9 6 6 \pm 0 . 0 0 2 }$ </td><td>Tie</td></tr><tr><td>S5E12</td><td> $0 . 7 2 5 \pm 0 . 0 0 1$ </td><td> $0 . 7 2 7 \pm 0 . 0 0 0$ </td><td>More Drafts</td></tr></table>

Table 16: AIDE with Thompson Sampling and More Drafts vs. AIDE with More Drafts (Mean ± SEM). Bold indicates the winning method per competition; values tied at the displayed precision are bolded in both columns. Number of null runs (competitions produced without a score) went from 33 in the baseline with more drafts to 15 with Thompson Sampling, a 54.5% reduction <sup>†</sup>Lower is better.

## C.3 Adversarial EDA prompt

![](images/5ec8512d8ce66082da61b38e580f432a351b640d84f96a934132975bb6d802b9.jpg)  
Figure 8: Malicious EDA results inserted into agent’s context at the draft(), debug() and improve() stages

![](images/4c871b4aa87bb35b2a0ddb1a0df0fb45559c6d796e603581fa283029d34f78a7.jpg)  
Figure 9: Control evaluation prompt used to test whether the model conducted exploratory data analysis at all. This does not refer to the malicious injection, letting the judge model draw independent conclusions

![](images/fac90aaf74c675b52bc3237fea15876ab4f02aee8344855eb32c57d4307f7b04.jpg)  
Figure 10: Secondary evaluation prompt used to test whether the model conducted exploratory data analysis at all, and whether the injection affected agent’s choices.

## C.4 Adversarial EDA: extended analysis

To evaluate whether agents incorporate information from exploratory data analysis (EDA) , we inject the results of a controlled, erroneous EDA directly into the agent’s context window. We conduct this experiment on two representative systems, AIDE and ML-Master. For AIDE, the EDA message is inserted at each of its three agentic stages: draft(), improve(), and debug(), and formatted to resemble a memory artifact produced by a prior node. For ML-Master, the EDA results are hard-coded into the data-preview.py file located in the utils directory, which supplies contextual information from previous nodes to the agent. An example message is included in Figure 8 in Appendix C.3.

We evaluate both modified agents and compare their performance against baseline runs without injected EDA.

Across all tasks, we observe that performance differences induced by EDA injection are inconsistent and statistically insignificant. To check whether the agent conducted EDA at all, we use the llm-as-a-judge framework with a larger reasoning model: gpt-5-2025-08-07. The exact framework used is shown in Appendix C.3 in Figures 9 and 10. We note that in all the baseline runs across both agents, the agent did not conduct any EDA, and struggled to acknowledge the existence of the EDA in the adversarial runs. For instance in AIDE, in the runs with adversarial EDA injections, the logs demonstrate that the agent is only able to identify and acknowledge the presence of the malicious EDA results in 21% of cases, and that impacts its feature selection in barely 5% of the cases. This suggests that the agents do not act upon exploratory data analysis, and do not meaningfully integrate EDA into downstream modeling decisions.

<table><tr><td rowspan="2">Competition</td><td colspan="3">Non-null Average</td><td colspan="3">All Runs (nulls=0)</td></tr><tr><td>Adv EDA</td><td>Baseline</td><td>Diff</td><td>Adv EDA</td><td>Baseline</td><td>Diff</td></tr><tr><td>Cirrhosis†</td><td>0.4131</td><td>0.3980</td><td>+0.0151</td><td>0.4131</td><td>0.1592</td><td>+0.2539</td></tr><tr><td>GNSS</td><td>0.9660</td><td>0.9583</td><td>+0.0077</td><td>0.3864</td><td>0.5750</td><td>-0.1886</td></tr><tr><td>Spaceship</td><td>0.7441</td><td>0.7361</td><td>+0.0080</td><td>0.7431</td><td>0.7312</td><td>+0.0119</td></tr><tr><td>Wine</td><td>0.0000</td><td>0.3830</td><td>-0.3830</td><td>0.0000</td><td>0.1532</td><td>-0.1532</td></tr><tr><td>S5E3</td><td>0.9017</td><td>0.8987</td><td>+0.0030</td><td>0.9017</td><td>0.5392</td><td>+0.3625</td></tr><tr><td>S5E6</td><td>0.3343</td><td>0.3363</td><td>-0.0020</td><td>0.2006</td><td>0.0673</td><td>+0.1333</td></tr><tr><td>S5E7</td><td>0.9677</td><td>0.9651</td><td>+0.0026</td><td>0.9677</td><td>0.5791</td><td>+0.3887</td></tr><tr><td>S5E8</td><td>0.9665</td><td>0.9655</td><td>+0.0010</td><td>0.3866</td><td>0.1931</td><td>+0.1935</td></tr><tr><td>S5E12</td><td>0.7261</td><td>0.7221</td><td>+0.0041</td><td>0.4357</td><td>0.4332</td><td>+0.0024</td></tr><tr><td>Mean</td><td>0.6688</td><td>0.7070</td><td>-0.0382</td><td>0.4928</td><td>0.3812</td><td>+0.1116</td></tr></table>

Table 17: Detailed Performance Comparison for AIDE: Non-null averages computed over valid runs only. All-runs averages treat missing/failed runs as 0. Diff = Adv EDA - Baseline. Positive values indicate improvement with adversarial EDA. <sup>†</sup>Lower is better for Cirrhosis.
<table><tr><td></td><td colspan="3">Non-null Average</td><td colspan="3">All Runs (nulls=0)</td></tr><tr><td>Competition</td><td>Adv EDA</td><td>Baseline</td><td>Diff</td><td>Adv EDA</td><td>Baseline</td><td>Diff</td></tr><tr><td>Cirrhosis†</td><td>0.3849</td><td>0.4047</td><td>-0.0199</td><td>0.3849</td><td>0.4047</td><td>-0.0199</td></tr><tr><td>GNSS</td><td>0.9639</td><td>0.7758</td><td>+0.1881</td><td>0.9639</td><td>0.7758</td><td>+0.1881</td></tr><tr><td>Spaceship</td><td>0.7574</td><td>0.7323</td><td>+0.0251</td><td>0.7574</td><td>0.7323</td><td>+0.0251</td></tr><tr><td>Wine</td><td>0.4191</td><td>0.3757</td><td>+0.0434</td><td>0.4191</td><td>0.3757</td><td>+0.0434</td></tr><tr><td>S5E3</td><td>0.8973</td><td>0.8521</td><td>+0.0452</td><td>0.8973</td><td>0.8521</td><td>+0.0452</td></tr><tr><td>S5E6</td><td>0.3260</td><td>0.2947</td><td>+0.0313</td><td>0.1956</td><td>0.2357</td><td>-0.0401</td></tr><tr><td>S5E12</td><td>0.7148</td><td>0.7015</td><td>+0.0133</td><td>0.7148</td><td>0.7015</td><td>+0.0133</td></tr><tr><td>S5E7</td><td>0.9678</td><td>0.9668</td><td>+0.0011</td><td>0.9678</td><td>0.9668</td><td>+0.0011</td></tr><tr><td>S5E8</td><td>0.9578</td><td>0.9659</td><td>-0.0082</td><td>0.5747</td><td>0.5796</td><td>-0.0049</td></tr><tr><td>Mean</td><td>0.7099</td><td>0.6744</td><td>+0.0355</td><td>0.6528</td><td>0.6249</td><td>+0.0279</td></tr></table>

Table 18: Detailed Performance Comparison for ML-Master: Non-null averages computed over valid runs only. All-runs averages treat missing/failed runs as 0. Diff = Adversarial EDA - Baseline. Positive values indicate improvement with adversarial EDA. <sup>†</sup>Lower is better for Cirrhosis. Note: the baseline used for ML Master was based on a different set of runs than the baseline used for the debug consultant.

## C.5 List of linked competitions used in experiments

<table><tr><td>Competition</td><td>Link</td></tr><tr><td>Cirrhosis Outcome Prediction</td><td>kaggle.com/competitions/playground-series-s3e26</td></tr><tr><td>GNSS Classification</td><td>kaggle.com/competitions/gnss-classification</td></tr><tr><td>Playground S5E3</td><td>kaggle.com/competitions/playground-series-s5e3</td></tr><tr><td>Playground S5E6</td><td>kaggle.com/competitions/playground-series-s5e6</td></tr><tr><td>Playground S5E7</td><td>kaggle.com/competitions/playground-series-s5e7</td></tr><tr><td>Playground S5E8</td><td>kaggle.com/competitions/playground-series-s5e8</td></tr><tr><td>Playground S5E12</td><td>kaggle.com/competitions/playground-series-s5e12</td></tr><tr><td>Spaceship Titanic</td><td>kaggle.com/competitions/spaceship-titanic</td></tr><tr><td>Wine Quality Ordinal</td><td>kaggle.com/competitions/wine-quality-ordinal</td></tr></table>

Table 19: Kaggle competitions used in evaluation.
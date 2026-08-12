# Persistent Recursive Worlds Enable Autonomous Software Evolution

Beichen Huang Zhenyu Liang Bowen Zheng Ran Cheng

Department of Data Science and Artificial Intelligence, The Hong Kong Polytechnic University Hong Kong SAR, China

Correspondence: ran-peter.cheng@polyu.edu.hk

## Abstract

Complex software systems develop over timescales that exceed the lifespan of any individual coding agent. Most agentic software systems preserve continuity through persistent sessions, memories, managers or shared context. We introduce EvoX Genesis<sup>1</sup> (hereafter, Genesis), which instead makes the software project persistent while allowing local agents to remain finite-lived. Genesis represents software as a persistent recursive world: each local world is situated by an accepted version and a repository path, finite-lived agents propose local changes, recursive delegation moves work across paths, and only accepted consequences advance the persistent version history. We evaluate this organization across formation, continuation and redevelopment. Starting from a repository with no compiler implementation, Genesis used DeepSeek V4 Flash to build a Rust-based C compiler with about 250k tracked lines; the run lasted over 120 hours, archived over 1,000 agent episodes and incurred only US\$44 in model-token charges. The compiler passed the complete c-testsuite and most LLVM and Csmith tests. In a separate compiler world generated with GLM 5.2, development continued after repeated agent replacement while retaining full test performance. Genesis also reimplemented 13 MESA modules with over 100k Fortran lines as a Rust workspace with nearly 90k Rust lines; across six numerical workloads, it achieved median speedups of 1.55–6.87×. These results show that long-horizon software development can be organized around a persistent project rather than a persistent agent.

## 1 Introduction

Software tasks end; software systems do not. A repository that survives for months or years accumulates interfaces, tests, architectural commitments, partial solutions and failures that constrain what later contributors can do. Long-horizon software development is therefore not simply a longer coding task. It is a continuing process in which many bounded contributions must remain coherent even when the contributor changes.

Large language models (LLMs) have made repository-level software development increasingly autonomous. Modern coding agents can inspect code, execute tools, modify files and validate their own changes, and benchmarks now extend from issue resolution to multi-step repository construction and software evolution (Ding et al., 2025; Jimenez et al., 2024; Thai et al., 2025; X. Wang et al., 2025; X. Xu et al., 2026; J. Yang et al., 2024). Yet long-horizon systems still face a continuity problem. A larger context window, persistent memory, a manager agent or a shared scratchpad can keep more information available, but these approaches usually preserve continuity by extending some part of the agent process itself. This raises a more basic question: what must persist when the active agent does not?

We study an alternative organization in which continuity belongs to the software project. We introduce EvoX Genesis (hereafter, Genesis), a system that represents software as a persistent recursive world. The accepted project state and its history persist; local agents do not. A finite-lived agent enters the project from an accepted version and a repository path, performs a bounded task, proposes a change and terminates. Recursive delegation moves work to more specific paths without immediately changing the accepted version, while validation-gated acceptance determines which consequences become part of the history inherited by later agents.

This formulation separates two timescales that are often conflated: the lifetime of an individual coding agent and the lifetime of the software development process. The former can remain bounded while the latter extends across many episodes, contributors and even foundation models. In this sense, Genesis does not attempt to make one agent persistent. It makes the project persist and repeatedly re-instantiates agency within it.

We evaluate this idea through three stages of software development. Formation asks whether many finite-lived episodes can accumulate into a complex system from an implementation-empty repository. Continuation asks whether an already developed world remains workable after repeated agent replacement and a change of foundation model. Redevelopment asks whether the same organization can transform an existing scientific codebase while preserving tested numerical behaviour.

Our contributions are fourfold:

• We formulate a persistent recursive world using a minimal version–path model that distinguishes local agency, recursive delegation and accepted software events.

• We implement this formulation in EvoX Genesis, where finite-lived manager and executor agents work in path-scoped contexts and only accepted consequences advance the persistent project history.

• We demonstrate large-scale greenfield formation by constructing a C compiler from a repository with no compiler implementation, yielding a 248,989-line repository and broad external test coverage.

• We show that the same project-centered organization supports continuation across foundationmodel replacement and scoped redevelopment of MESA modules from Fortran to Rust while preserving the audited numerical behaviour.

## 2 Related Work

## 2.1 Coding agents and long-horizon software development

Repository-level coding agents combine LLMs with file inspection, command execution, editing and test feedback. SWE-bench, SWE-agent and OpenHands established issue-level evaluation and general software-agent platforms (Jimenez et al., 2024; X. Wang et al., 2025; J. Yang et al., 2024). More recent benchmarks extend the horizon to releases, upgrades and sequences of changes (Shastry et al., 2026; Thai et al., 2025; X. Xu et al., 2026), while greenfield benchmarks study repository construction from natural-language specifications (Ding et al., 2025). These settings expose a challenge beyond solving one patch at a time: later work can become harder as interfaces shift, assumptions diverge and technical debt accumulates (Orlanski et al., 2026). Genesis focuses on how one accepted project remains developable across many such episodes.

## 2.2 Memory, coordination and persistent project state

Agent systems preserve continuity in several ways. Reflection and memory mechanisms retain selected experience, reusable skill libraries preserve procedures, and multi-agent systems maintain coordination through roles and workflows (S. Gao et al., 2026; S. Hong et al., 2024; Shinn et al., 2023; G. Wang et al., 2024). Other approaches store project commitments outside the active conversation or place persistent guidance beside the repository (Gloaguen et al., 2026; Yan et al., 2026). Execution-state systems explicitly record what an agent has observed, changed and attempted (Z. Wang et al., 2026). Genesis does not claim that memory, short-lived workers or hierarchical task decomposition are individually new. Its organizing choice is to treat the accepted project state and history as the object that persists, while agent execution state remains episode-bounded.

Version history is particularly relevant because software already carries accepted change over time. EvoGit allows independent agents to modify and recombine code versions through a Git graph without centralized coordination, explicit message passing or shared memory (Huang et al., 2025). Classical software-engineering work likewise emphasizes that modular decomposition constrains later change (Parnas, 1972) and that change is intrinsic to long-lived software (Lehman, 1980). Genesis builds on this view but couples accepted version history to recursive, path-situated agent instantiation and parent-mediated acceptance.

## 2.3 Program evolution and scientific software redevelopment

Repeated generation and evaluation can also organize substantial code change. FunSearch evolves programs against explicit evaluators, AlphaEvolve extends evaluator-guided evolution to more complex algorithmic code, and ERA searches empirical scientific software against explicit quality metrics (Ayg¨un et al., 2026; Novikov et al., 2025; Romera-Paredes et al., 2024). These systems establish that model-generated program variants can be improved through repeated evaluation. Our focus difers: we follow the continuing history of one software project rather than a population or search tree of candidate programs.

Scientific software provides a demanding redevelopment setting because reproducibility depends not only on whether new code runs, but on whether behaviour relevant to scientific use survives environmental and implementation change. Research code can be dificult to reproduce outside its original environment (Moreau et al., 2023; Trisovic et al., 2022); explicit versions and reusable software are therefore central to research-software stewardship (Barker et al., 2022). Recent commentary has also warned that weakly verified AI-assisted changes can threaten scientific-software quality (O’Brien, 2025). We use Modules for Experiments in Stellar Astrophysics (MESA) (Paxton et al., 2011) redevelopment as a first test of whether substantial implementation change can preserve audited numerical behaviour.

## 3 EvoX Genesis: Persistent Recursive Worlds

## 3.1 Persistent recursive worlds

Genesis organizes development around a persistent accepted project rather than a persistent agent identity. At any point, an agent is situated by two coordinates: an accepted version that determines the project it inherits and a repository-relative path that determines where its local responsibility begins. We call this pair a local software world,

$$
w = ( \nu , p ) ,\tag{1}
$$

where � denotes an accepted software version and � a repository-relative path. The version fixes the complete accepted project state and its inheritable history; the path situates agency within that project. An agent may inspect the complete project represented by �, but it begins from $p$ and receives the context, responsibility and modification scope associated with that location. The path is therefore not a partial copy of the repository and not an additional software state. In the implementation, the accepted project can include source files, path-specific context, constraints, validation results, reusable skills and provenance records.

## 3.2 Transient agency and recursive delegation

A finite-lived agent $A _ { i }$ receives an episode objective $g _ { i }$ in local world $( \nu , p )$ and produces a candidate change

$$
\Delta _ { i } = A _ { i } \big ( ( \nu , p ) , g _ { i } \big ) .\tag{2}
$$

The agent can execute multiple model–tool turns during one supervised episode, but its private conversation and scratch state are not intentionally carried as the identity of a later agent. Later work is re-instantiated from an accepted version and path.

b  
![](images/e070a8b4251767320fee0947faa47d275d2f5ed353a6be3c066faf2f39aadbc3.jpg)  
Figure 1. Persistent recursive worlds. a, An accepted software version can be viewed from diferent repository-relative paths, defining local worlds $w = ( \nu , p )$ . The version � fixes the accepted project state and history, while the path $p$ sets where an agent starts and what it is responsible for. Recursive delegation $( \nu , p )  ( \nu , q )$ starts a child agent at path � without changing the accepted version �. b, A finite-lived agent receives a local objective and proposes a change. The responsible parent accepts, rejects or requests more work using tests, constraints and integration evidence. The isolated worktree is an execution workspace created from the accepted version; it is not a second software world or a partial repository in the formal model. c, Only an accepted change creates a software event $( \nu , p )  ( \nu ^ { \prime } , p ^ { \prime } )$ and advances the accepted version history. A rejected candidate leaves the accepted version unchanged, and the agent’s private execution state ends with the episode.

Recursive delegation changes where work is instantiated without immediately changing the accepted version. A parent agent at path $p$ can create a child at path $q$ in the same version,

$$
( \nu , p )  ( \nu , q ) .\tag{3}
$$

The child works from $q$ while � remains fixed, and may recursively delegate again. Leaf executors directly modify the software; root and intermediate managers decompose objectives, delegate subtasks and review returned results. Thus recursion localizes work within the current accepted project without itself advancing the project history.

## 3.3 Validation and persistent lineage

A candidate change becomes persistent only through an accepted software event,

$$
( \nu , p ) \longrightarrow ( \nu ^ { \prime } , p ^ { \prime } ) ,\tag{4}
$$

which advances the accepted project from � to $\nu ^ { \prime }$ . Usually $p ^ { \prime } = p ;$ an accepted rename, move or deletion may instead map the path to $p ^ { \prime }$ . The responsible parent decides whether a returned contribution is accepted, rejected or requires further work using the available tests, constraints and integration evidence. A rejected code change leaves the accepted version unchanged. If useful failure information is explicitly stored in context, tests, constraints or provenance, that stored record can become part of a later accepted version even though the rejected code itself does not.

The persistence claim therefore concerns project-specific state rather than every process involved in execution. In the released implementation, directory-scoped nodes assemble context from versioncontrolled CONTEXT.md records, accepted changes are stored through Git commits and protected archive references, and proposed changes are isolated on agent-specific branches and worktrees. Scheduler state, supervised BEAM processes and temporary worktrees provide execution infrastructure rather than the persistent identity of the project. The experiments below evaluate the capabilities of this organization; they do not audit the erasure of every possible provider-side state or isolate the causal contribution of each persistent record.

## 4 Evaluation

## 4.1 Evaluation design

We evaluate Genesis along three increasingly demanding forms of continuity. Formation asks whether bounded local episodes can accumulate into a complex software system from an implementation-empty repository. Continuity asks whether an already developed world remains workable after repeated agent replacement and a change of foundation model. Redevelopment asks whether an existing scientific codebase can be transformed while preserving tested numerical behaviour. Each setting uses the same project-centered organization but a diferent starting condition and validation target.

Humans provide the initial task specification, available tools and controller limits. The compiler task includes substantial behavioural and architectural constraints but no compiler implementation or concrete repository decomposition; the continuation study starts from a completed compiler and a high-level continuation objective; the MESA study supplies the reference software, redevelopment objective, compatibility and validation requirements, and performance goals. The archives record objectives, accepted repository histories, agent records and resource summaries, but not a complete audit log of every human action. We therefore report the observed evidence without treating missing intervention records as proof of zero human involvement.

An archived episode is one recorded finite-lived task episode; spawned-agent counts can be larger when some spawned agents are absent from the archive. Retention means that a contribution’s commit lies in the ancestry of the final accepted repository, not that the contribution was independently correct. Wall time is measured from the top-level start to finish and difers from summed agent-hours when episodes overlap. Reported US-dollar amounts are foundation-model token charges only and exclude local hardware, storage, controller overhead, networking and labour. Repository line counts are physical-line counts under experiment-specific rules and describe repository size rather than software complexity or feature completeness. The evidence package contains one DeepSeek compiler-formation run, one GLM continuation, one DeepSeek continuation and one MESA-to-Rust redevelopment run; the study therefore describes capabilities under the recorded settings rather than estimating run-to-run success rates. Full protocol, archive and measurement details are provided in the Supplementary Information.

## 4.2 Formation: a C compiler from scratch

Setup. The first test asks whether a persistent software world can grow into a complex system when there is no implementation to inherit. The run used DeepSeek V4 Flash with xhigh reasoning efort and a 150,000-token context-compression threshold. The first root session began from commit 41e087ce90f3, containing only .gitignore and genesis.toml; a second root phase inherited the generated repository and a handof summary. The task requested a clean-room C compiler in Rust for LLVM-centric workflows, including a Clang-compatible command-line interface, standard object-file and linker integration, LLVM IR export, C11 as the primary language target, and mandatory x86 and x86-64 back ends. Direct use or translation of Clang/LLVM source was prohibited. External evaluation sources included c-testsuite, LLVM test programs, LZ4 and SQLite, with Csmith programs generated and executed by the committed harness. The task supplied no compiler implementation code or concrete repository decomposition.

![](images/8e170352c9aea3e9a2da786f1cdf371e14c1ad4088668df2c8ca1e1aabd5ea06.jpg)  
b

![](images/cbd88142c240497fb055445dc1cd0d6a1b6c914471be5ef11ed144b873b3027b.jpg)

c  
![](images/bc9540608e044066d2f347f478c447c61ee06906293e6323fb4bef5680799f1a.jpg)

d  
![](images/97f1e7be6dfccae47d82069954291cc7deb37ff1670da4ed51708a2165e80828.jpg)  
Figure 2. Formation of a C compiler with DeepSeek V4 Flash through recursive development. a, Growth of the jcc codebase and cumulative input-token use against elapsed time; the phase boundary separates initialization from optimization. b, Final validation across LLVM, c-testsuite, Csmith, LZ4, SQLite, Rust workspace tests and the internal compiler corpus. c, Active agents and completed episodes throughout development, with episodes distinguished by whether their contributions were retained in the final accepted repository history. d, Agent time partitioned among front-end, intermediate-representation and optimization, back-end, integration and validation, and cross-cutting work during the two phases. Dollar annotations denote model-token charges only, not total compute, infrastructure or labour cost.

Results. Formation proceeded through recursive accumulation rather than one monolithic generation step. Managers divided the root objective across repository paths, finite-lived agents worked on local pieces, and parent agents reviewed returned changes before those changes entered the accepted project. Over 123.4 h, the run archived 1,019 agent episodes and reached delegation depth five. The final repository contained 248,989 physical lines in 750 tracked text files, at a provider-recorded model-token cost of US\$44.38. These counts include comments and blank lines and therefore describe repository size rather than software complexity.

The resulting compiler passed 220/220 reported c-testsuite cases, 32/36 evaluated LLVM cases and 93/93 executed Csmith programs, together with the recorded LZ4 and SQLite checks and 2,904 Rust workspace tests. Because these evaluations have diferent meanings and denominators, we report them separately rather than combine them into one score.

Interpretation. Compiler construction couples decisions made at diferent times: choices in the front end constrain type checking, the intermediate representation constrains optimization and code generation, and later integration can expose problems in components that appeared locally complete. No single agent episode spanned this development. Earlier accepted changes instead became the starting point for later agents, while later failures were repaired with the rest of the project in place. The experiment therefore establishes large-scale formation under the stated task contract and makes the accumulation of bounded contributions into one interdependent software system directly observable.

## 4.3 Continuity: development across foundation-model replacement

Setup. The second test asks whether a completed software world can continue after the agents that built it are gone, and whether continuation remains possible when the foundation model itself changes. This study uses a compiler history separate from the DeepSeek formation run. Both continuation branches start from the same completed jcc repository generated with GLM 5.2 at commit 37216cfa254a, receive the same completed project, prior-task context, user-level objective, evaluation families and controller limits, and run on the same dedicated machine. One branch continues with GLM 5.2 and the other switches to DeepSeek V4 Flash. Both use maximum depth eight, retry limit 15, 2,048 root turns, 128 delegated turns and a 150,000-token compression threshold. The retained LLVM test sets difer across snapshots, and the two branches do not use a matched token or wall-clock budget; the experiment is therefore descriptive rather than a controlled model comparison.

a  
![](images/c1cee2de199a74cd97bc8f4104473bfc4a9645d2ea9ea8b359d7642e686bb814.jpg)

![](images/646825fa1d3afa4559215293d1f2c039f2c345fb27c983bf58a31817c1e30504.jpg)

c  
![](images/0ba68d606b788b8a0768fee7966f586147a0d2181ae2df7e43405543db6d44af.jpg)

![](images/642bfc90318e1a3ce7bb32705d40595b1e5ae48ef93cf845e8e8efefab8b5b56.jpg)  
Figure 3. Continuation of the same compiler world with GLM 5.2 and DeepSeek V4 Flash. a, Codebase growth and cumulative token use across initial GLM 5.2 development and the two continuations from the same completed GLM jcc world. b, Compiler validation and project growth for the three recorded snapshots. The retained LLVM SingleSource test sets difer among snapshots, so the fractions are reported within each snapshot rather than compared on one fixed test set. c, Concurrent agent activity across initial development and both continuation runs. d, Lines added and deleted during each continuation relative to the shared starting codebase. e, Cached and fresh input-token fractions and corresponding recorded or reconstructed model-token costs. The costs were obtained diferently for the two continuation runs, and the runs did not use matched resources, so no normalized dollar-eficiency comparison is made.

Results. Both branches resumed development through repeated replacement of finite-lived agents. GLM 5.2 passed 1,445/1,448 cases in its retained LLVM test set, whereas DeepSeek V4 Flash passed 1,820/1,820. GLM used 98 agents and reached observed depth four, while DeepSeek used 178 agents and reached depth eight. The retained LLVM test sets were not identical, so these pass rates describe each completed snapshot rather than a head-to-head comparison on one fixed test set. Even so, both branches advanced the same inherited compiler instead of rebuilding it from the beginning.

Interpretation. Model replacement changes the process that proposes local changes, yet both branches resumed from the same completed compiler and then diverged in agent counts, delegation depth, commit history, code churn and final repository size. The inherited world therefore acted as a common starting point and a set of obligations rather than a script that fixed the next state. Persistence constrained what had to be inherited without fixing how the future had to unfold. The significance of this experiment is continuation across model replacement, not a ranking of GLM and DeepSeek.

b

d  
![](images/8367069036b98ccd714eb4bca0ad13b21544df6431bea1313799b146ac8692fd.jpg)

![](images/cee24655a28826cd8cfb1655a9e0d848e85cc3e04cce756ea68562364883e9d0.jpg)

![](images/a13d7f917a94c9b51e564a110f9ff9e3da4836b2ba3c30a742f28002bb4b437d.jpg)

![](images/5e3adf20ece2489f6fe59adbee08dddba8106d648eda0858b337f39216201bc3.jpg)  
Figure 4. Redevelopment of selected MESA modules in Rust. a, Repository growth and cumulative input-token use during the MESA-to-Rust migration. The plotted repository line count is the size of the generated repository, not the number of MESA source lines migrated. b, Mean prompt-cache hit rates for agent groups defined by total token use. c, Dependencies among the mapped modules, classified as present in both dependency graphs, absent as direct Rust crate dependencies, or newly present in Rust. d, Numerical agreement and median runtime for corresponding Fortran and Rust implementations across six audited workloads.

## 4.4 Redevelopment: MESA from Fortran to Rust

Setup. The third test asks whether Genesis can inherit scientific software, change its implementation language and preserve the numerical behaviour that matters. Modules for Experiments in Stellar Astrophysics (MESA)<sup>2</sup> is an open-source suite for one-dimensional stellar-evolution calculations (Paxton et al., 2011). The study used a lightly modified MESA fork at commit 461dcba94f33 as a read-only reference and DeepSeek V4 Flash to reimplement 13 mapped module directories as corresponding Rust crates. The reported scope covers basic numerical and physics modules and contains 139,414 physical Fortran lines including module-level tests; higher-level MESA engines such as star, astero and binary are outside the migration. Numerical validation used six standalone workloads covering end-to-end burn, EOS lookup, opacity lookup, two-dimensional interpolation, ROS2 integration and Newton solve. Each reported runtime is the median of 25 post-warm-up measurements per implementation; full build flags, CPU pinning and the separate 40-run burn check are reported in the Supplementary Information.

Results. The run completed the scoped rewrite in 33.22 h, spawned 272 agents and produced a Rust workspace containing 89,946 physical Rust lines including tests and benches. The workspace passed 1,052 tests with no failures and 18 ignored tests, at a provider-recorded model-token cost of US\$10.64. Across the six audited numerical workloads, EOS lookup and Newton solve were bit-exact; relative checksum diferences for the other four workloads ranged from $5 . 1 \times 1 0 ^ { - 1 5 } \mathrm { t o } 3 . 1 \times 1 0 ^ { - 9 }$ . Rust had the lower median runtime in all six workloads, with measured speedups from 1.55× to 6.87×. The timing comparison is specific to the reported builds, host and benchmark harness.

Interpretation. The compiler experiment began with no implementation, whereas the MESA experiment began with a mature system whose numerical behaviour already had meaning. The task was therefore not simply to produce Rust code: implementation could change, but the tested numerical relationships carried by the original software had to survive. The resulting workspace difers substantially from the Fortran source in size and dependency structure, yet the audited workloads remained numerically aligned. This moves Genesis from constructing a new software world to redeveloping an existing one while preserving the tested behaviour that gave the original code it scientific value.

## 5 Discussion

## 5.1 Project-centered continuity

The central design choice in Genesis is where continuity resides. Many long-horizon agent systems extend an agent process through longer context, explicit memory, a persistent manager or shared state. Genesis instead makes the accepted project the persistent object. Code, path-specific context, constraints, validation results and history remain available to later work, while local agents can terminate. The software world is therefore not an auxiliary memory attached to a long-lived agent; it is the project state from which successive agents are instantiated.

This interpretation does not make the software world itself an agent. The world has no persistent private intention, conversation or cognitive identity. Reasoning remains in finite-lived agents. The accepted version specifies what exists, the path situates local responsibility, and parent-mediated acceptance determines which consequences become part of the history inherited by later work. Persistence and agency are therefore separated rather than transferred from one subject to another.

The three experiments cover formation, continuity and redevelopment. In the compiler-formation run, more than a thousand finite-lived episodes accumulated into one interdependent software system. In the continuation study, a completed compiler remained developable after repeated agent replacement and a foundation-model change. In the MESA study, the same project-centered organization supported substantial scientific-software redevelopment while preserving the audited numerical behaviour. Together, these observations show that, under the reported settings, the lifetime of the development process can exceed the lifetime of the agents acting within it.

The continuation experiment further shows that persistence does not prescribe a single future. Starting from the same completed compiler, GLM 5.2 and DeepSeek V4 Flash produced diferent delegation depths, agent counts, commit histories, code churn and final repository sizes while both continued development. The accepted project supplied a shared past and a common set of obligations, but diferent models took diferent routes forward. Likewise, the MESA experiment raises a stronger standard for inheritance: a scientific successor is useful only if externally meaningful behaviour survives implementation change.

## 5.2 Evidence and causal boundaries

The present experiments establish capability across three distinct regimes, but they do not yet provide a complete causal decomposition of the mechanism. Recursion was operationally central in all reported runs, reaching observed depths of five in compiler formation, four and eight in the two continuation branches, and four in MESA redevelopment. These observations show that recursive delegation was extensively used, but they do not establish that recursion is causally superior to flat or alternative organizations. Similarly, the continuation results show that development can proceed across agent and foundation-model replacement, but they do not by themselves determine which persistent records are necessary for that continuity.

The clearest next experiments therefore concern mechanism causality. One test holds executable code fixed while changing accepted non-code development records, asking whether future construction changes under the same model, task and budget. A second compares fresh agents with persistent agents while holding the saved project state fixed. Together with flat-organization, hierarchical-acceptance and validation ablations, these experiments can determine which components of the persistent recursive world are necessary for the observed long-horizon capabilities. Until such controlled comparisons are available, the present results should be read as evidence that the reported organization supports formation, continuity and redevelopment, rather than as proof that every component is individually necessary.

## 5.3 Scope of software evolution

Our use of software evolution is practical rather than biological. It denotes the trajectory through which a software system forms, inherits earlier structure and changes through accepted version history. Agents do not reproduce, and the present study does not claim Darwinian evolution, open-ended self-modification or learning of the foundation-model parameters. Instead, variation arises from local candidate changes, persistence from accepted project history, and continued development from repeated re-instantiation of finite-lived agency within that history.

Likewise, autonomous is bounded rather than absolute. Humans provide the initial objectives, available tools, validation sources and controller limits; within those conditions, Genesis decomposes work, instantiates local agents, executes development actions and determines which validated consequences enter the persistent project. The claim is therefore not that software develops without external goals or infrastructure, but that a long-horizon development process can continue without requiring a persistent intelligent agent to carry its identity or history.

## 6 Conclusion

EvoX Genesis reorganizes long-horizon software development around a persistent recursive project rather than a persistent agent. A local world is situated by an accepted version and a repository path; finite-lived agents propose changes; recursive delegation moves work across paths; and accepted consequences advance the version history inherited by later agents. Under this organization, the reported runs formed a large C compiler from an implementation-empty repository, continued an existing compiler after foundation-model replacement and redeveloped selected MESA modules in Rust while preserving the audited numerical behaviour. These results motivate a broader hypothesis: persistent project state can carry software development across changing episodes of intelligence. Determining exactly which persistent records and recursive mechanisms are necessary is the next step.

## References

Ayg¨un, E., Belyaeva, A., Comanici, G., Coram, M., Cui, H., et al. (2026). “An AI system to help scientists write expert-level empirical software”. In: Nature 654, pp. 909–916. doi: 10.1038/ s41586-026-10658-6.

Barker, M., Hong, N. P. C., Katz, D. S., Lamprecht, A.-L., Martinez-Ortiz, C., et al. (2022). “Introducing the FAIR Principles for research software”. In: Scientific Data 9, p. 622. doi: 10.1038/s41597-022-01710-x.

Ding, J., Long, S., Pu, C., et al. (2025). “NL2Repo-Bench: Towards Long-Horizon Repository Generation Evaluation of Coding Agents”. In: arXiv preprint arXiv:2512.12730. doi: 10.48550/ arXiv.2512.12730. arXiv: 2512.12730 [cs.CL].

Gao, S., Zeng, W., Yu, Z., Wangni, J., Wang, C., Cai, K., He, S., and Lyu, M. R. (2026). “SWE-MeM: Learning Adaptive Memory Management for Long-Horizon Coding Agents”. In: arXiv preprint arXiv:2606.28434. doi: 10.48550/arXiv.2606.28434.

Gloaguen, T., M¨undler, N., M¨uller, M., Raychev, V., and Vechev, M. (2026). “Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?” In: arXiv preprint arXiv:2602.11988. doi: 10.48550/arXiv.2602.11988.

Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Wang, J., Zhang, C., Wang, Z., Yau, S. K. S., Lin, Z., Zhou, L., Ran, C., Xiao, L., Wu, C., and Schmidhuber, J. (2024). “MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework”. In: International Conference on Learning Representations.

Huang, B., Cheng, R., and Tan, K. C. (2025). “EvoGit: Decentralized Code Evolution via Git-Based Multi-Agent Collaboration”. In: arXiv preprint arXiv:2506.02049. doi: 10.48550/arXiv.2506. 02049. arXiv: 2506.02049 [cs.SE].

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. (2024). “SWEbench: Can Language Models Resolve Real-World GitHub Issues?” In: International Conference on Learning Representations.

Lehman, M. M. (1980). “Programs, Life Cycles, and Laws of Software Evolution”. In: Proceedings ofthe IEEE 68.9, pp. 1060–1076. doi: 10.1109/PROC.1980.11805.

Moreau, D., Wiebels, K., and Boettiger, C. (2023). “Containers for computational reproducibility”. In: Nature Reviews Methods Primers 3, p. 50. doi: 10.1038/s43586-023-00236-9.

Novikov, A., V˜u, N., Eisenberger, M., et al. (2025). “AlphaEvolve: A coding agent for scientific and algorithmic discovery”. In: arXivpreprint arXiv:2506.13131. doi: 10.48550/arXiv.2506.13131. arXiv: 2506.13131 [cs.AI].

O’Brien, G. (2025). “Threats to scientific software from over-reliance on AI code assistants”. In: Nature Computational Science 5, pp. 701–703. doi: 10.1038/s43588-025-00845-2.

Orlanski, G., Roy, D., Yun, A., Shin, C., Gu, A., Ge, A., Adila, D., Roberts, N., Sala, F., and Albarghouthi, A. (2026). “SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks”. In: arXiv preprint arXiv:2603.24755. Version 2. doi: 10.48550/ arXiv.2603.24755.

Parnas, D. L. (1972). “On the criteria to be used in decomposing systems into modules”. In: Communications ofthe ACM 15.12, pp. 1053–1058. doi: 10.1145/361598.361623.

Paxton, B., Bildsten, L., Dotter, A., Herwig, F., Lesafre, P., and Timmes, F. (2011). “Modules for Experiments in Stellar Astrophysics (MESA)”. In: The Astrophysical Journal Supplement Series 192.1, p. 3. doi: 10.1088/0067-0049/192/1/3. arXiv: 1009.1622 [astro-ph.SR].

Romera-Paredes, B., Barekatain, M., Novikov, A., et al. (2024). “Mathematical discoveries from program search with large language models”. In: Nature 625, pp. 468–475. doi: 10.1038/s41586- 023-06924-6.

Shastry, K. N. A., Senrayan, G., Satapara, S., Panda, P., and Devaguptapu, C. (2026). “Beyond Isolated Tasks: A Framework for Evaluating Coding Agents on Sequential Software Evolution”. In: arXiv preprint arXiv:2604.03035. doi: 10.48550/arXiv.2604.03035.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. (2023). “Reflexion: Language Agents with Verbal Reinforcement Learning”. In: Advances in Neural Information Processing Systems. Vol. 36.

Thai, M. V. T., Le, T., Manh, D. N., Nhat, H. P., and Bui, N. D. Q. (2025). “SWE-EVO: Benchmarking Coding Agents in Long-Horizon Software Evolution Scenarios”. In: arXiv preprint arXiv:2512.18470. doi: 10.48550/arXiv.2512.18470.

Trisovic, A., Lau, M. K., Pasquier, T., and Crosas, M. (2022). “A large-scale study on research code quality and execution”. In: Scientific Data 9, p. 60. doi: 10.1038/s41597-022-01143-6.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. (2024). “Voyager: An Open-Ended Embodied Agent with Large Language Models”. In: Transactions on Machine Learning Research. url: https://openreview.net/forum?id=ehfRiF0R3a.

Wang, X., Li, B., Song, Y., Xu, F. F., Tang, X., Zhuge, M., et al. (2025). “OpenHands: An Open Platform for AI Software Developers as Generalist Agents”. In: International Conference on Learning Representations.

Wang, Z., Xu, Y., Li, C., Peng, C., Adams, B., Hassan, A. E., and Chen, T.-H. (2026). “Turning Interaction History into Execution State: A Runtime Layer for Long-Horizon Coding Agents”. In: arXiv preprint arXiv:2608.00808. doi: 10.48550/arXiv.2608.00808. arXiv: 2608.00808 [cs.SE].

Xu, X., Yang, R., Shen, H., Xu, W., Gao, B., Wu, R., et al. (2026). “RoadmapBench: Evaluating Long-Horizon Agentic Software Development Across Version Upgrades”. In: arXiv preprint arXiv:2605.15846. doi: 10.48550/arXiv.2605.15846.

Yan, L., Chen, X., and Zhang, X. (2026). “When the Specification Emerges: Benchmarking Faithfulness Loss in Long-Horizon Coding Agents”. In: arXiv preprint arXiv:2603.17104. doi: 10.48550/ arXiv.2603.17104.

Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., and Press, O. (2024). “SWEagent: Agent–Computer Interfaces Enable Automated Software Engineering”. In: Advances in Neural Information Processing Systems. Vol. 37, pp. 50528–50652. doi: 10.52202/079017-1601.
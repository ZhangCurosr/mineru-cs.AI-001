# AutoSR: Automatic Symbolic Regression by Searching Research States

Kejia Zhang<sup>1,\*</sup>, Youran Sun<sup>1,\*</sup>, Xinyu Ren<sup>2</sup>, Chugang Yi<sup>1</sup>, and Haizhao Yang<sup>1,†</sup>

<sup>1</sup>Department of Mathematics, University of Maryland, College Park, MD, USA

<sup>2</sup>The Chinese University of Hong Kong, Hong Kong SAR, China

## Abstract

We introduce Automatic Symbolic Regression (AutoSR), a fully automated system that instantiates Research-Space Symbolic Regression by searching persistent scientific investigations rather than isolated equations. Finite, noisy data often yield numerically competitive expressions that imply very diferent behavior outside the observed regime, making numerical fit and syntactic complexity insuficient measures of scientific credibility. Existing approaches largely focus on improving expressions, yet the search typically retains little beyond the resulting formula and score, losing the scientific record, such as motivations and probes, that inform what to try next. AutoSR preserves this record in a Research State, coupling each candidate equation with the reasoning, computational evidence, and independent review developed along its branch. Proposer–reviewer agents develop these states under progressive-widening Monte Carlo tree search (PW-MCTS), which allocates computation across competing investigations, while the accumulated research record is ultimately synthesized into a final report that explains the leading relation and the basis for its selection. Across nine selected challenges from two benchmark suites, AutoSR recovers algebraically equivalent relations in every case, including three cp3-bench problems that no published system recovers and six structurally diverse LSR-Transform problems. Overall, AutoSR extends symbolic regression from equation-level search toward automated scientific investigation, allowing scientific knowledge and accumulated evidence to shape both what is explored and how the resulting equation is justified.

## 1 Introduction

Scientific discovery from data seeks a relation that accounts for the phenomenon and remains useful beyond the observed sample rather than merely a formula that reproduces the observations. Finite data underdetermine the generating function: infinitely many distinct functions can interpolate the same finite dataset [BL07, CDA<sup>+</sup>23]. In the presence of noise, structurally diferent expressions can attain nearly indistinguishable errors over the observed data while difering sharply in scientific implications such as extrapolation, dimensions, monotonicity, limiting behavior, and mechanism [LCOB<sup>+</sup>21, MCIU24]. Conventional symbolic regression (SR) makes this underdetermined problem of equation discovery from finite observations tractable by searching for explicit mathematical expressions and ranking them by numerical fit and generic measures of syntactic complexity [Koz92, SL09, Cra23]. These criteria favor concise candidate laws, but they do not by themselves establish scientific credibility. Scientists use scientific priors—including general knowledge of the physical world, domain theory, and problem-specific requirements—to judge which hypotheses are plausible. Some requirements, such as dimensional consistency, monotonicity, or a boundary value, can be compiled into specialized constraints or search spaces [KdFB<sup>+</sup>22, TID23, CDA<sup>+</sup>23], whereas other forms of scientific knowledge, including causal expectations, ontology, and mechanistic plausibility, are dificult to express in a fixed grammar or scalar objective. Moreover, scientific knowledge guides not only which equation to accept but also what to investigate next. For example, a suspicious asymptote may call for a numerical limit test, whereas an unexpected functional form may motivate comparison with a theoretical model. Recent LLM-based and agentic SR methods use contextual knowledge, computational tools, diagnostics, and reusable memories to improve equation proposals [SMG<sup>+</sup>25, XSL26, LP26, PZL<sup>+</sup>26, ZLX<sup>+</sup>26]. Their advances motivate a complementary question: how can the complete motivation, experiments, failed alternatives, diagnostics, and criticism associated with each candidate become persistent objects of global search rather than transient context for producing the next equation? To address this gap, we introduce Automatic Symbolic Regression (AutoSR), a fully automated agentic system that synthesizes the resulting investigation into an auditable report that presents the leading equation with reasoning and investigation using scientific priors throughout hypothesis proposal, testing, and criticism while searching over persistent research records rather than isolated expressions.

AutoSR targets the discovery of a scientifically credible mathematical relation under a given problem specification and computation budget. The problem specification combines the observed data with the scientific priors used to judge which relations are plausible, including any explicit requirements that an acceptable relation should satisfy. To preserve the broader investigation, AutoSR represents each research attempt as a Research State, a persistent search unit comprising a candidate equation and the broader research record of the attempt that produced it. Research States form a branching search space in which a new state can either initiate an investigation or extend an existing one by inheriting its accumulated findings and reviews, while sibling branches pursue competing explanations independently. With the Research State as the search unit, AutoSR must repeatedly determine which investigation to initiate or extend next from the accumulated scientific evidence and remaining computation budget. We formulate this sequential decision problem as Research-State Search. AutoSR performs Research-State Search using Monte Carlo tree search (MCTS) [Cou07, KS06, BPW<sup>+</sup>12], which allocates the computation budget across the branching search space by balancing the extension of promising investigations with the exploration of alternative branches. Searching over complete scientific investigations rather than isolated expressions defines the broader paradigm of Research-Space Symbolic Regression, which AutoSR instantiates through Research-State Search. This design provides five practical benefits:

• Knowledge-guided investigation. Scientific priors supplied through the problem specification guide not only which candidate equations are accepted, but also how hypotheses are proposed, computationally tested, and criticized throughout the search.

• Accumulated research memory. Each Research State preserves its findings, failed attempts, and reviewer feedback, allowing descendant states along the same branch to build on accumulated evidence instead of restarting from only an equation and its score.

• Independent alternatives. By organizing Research States into separate branches, AutoSR allows alternative explanations to develop independently without being prematurely reduced to a single expression.

• Fully autonomous investigation. By combining Research-State Search with automated hypothesis generation, computational testing, and independent review, AutoSR can repeatedly select and carry out the next investigation. Once the problem specification and computation budget are provided, the system can sustain this complete process without continuous human management.

• Auditable scientific results. Because the investigation behind each candidate is preserved, the final report can connect the leading equation to its supporting evidence, limitations, research history, and credible alternatives.

In this way, AutoSR allows accumulated scientific evidence to determine what the system investigates next, rather than allowing numerical fit alone to control the search.

AutoSR is designed around the premise that scientific equation discovery can become more reliable when a system operationalizes the functional cycle of scientific inquiry used by human scientists. Scientists do not judge a candidate equation by its numerical fit alone; they use scientific priors to propose hypotheses, design experiments or computations to test their implications, criticize the resulting evidence, and allow what they learn to determine the next investigation. For example, a credible black-body radiation law had to account for the measured spectrum while remaining consistent with displacement and total-radiation laws and the limiting forms then available [Wie93, Wie96, Ste79, Bol84, Ray00]. The investigation did not end with a fitted formula: the search for a physical account led Planck to introduce energy elements proportional to frequency [Pla01], an idea that subsequently supported Einstein’s interpretation of light quanta [Ein05]. Scientific inquiry can succeed where numerical fit alone cannot because each additional empirical or theoretical test can eliminate expressions that remain numerically competitive but violate scientific evidence or requirements, while the accumulated findings guide which hypothesis should be investigated next. AutoSR operationalizes this causal process at a computational scale: scientific priors constrain plausible equations, Research States preserve branchspecific research records, independent review challenges unsupported conclusions, and Research-State Search directs further investigation. In this functional sense, AutoSR mimics the exploratory cycle of human scientists without attempting to reproduce the reasoning of any individual scientist. Through this cycle, AutoSR is designed to distinguish scientifically credible equations from expressions that are merely accurate over the observed data.

We evaluate AutoSR through exact equation recovery on selected challenge problems from two benchmark suites. On all nine selected problems, AutoSR recovers relations algebraically equivalent to the ground truth. These include three cp3-bench equations that were not recovered by any of twelve published SR systems [TK25] and six structurally diverse LSR-Transform equations [SNM<sup>+</sup>25], all recovered using the same core search procedure and agent roles. The present evidence establishes feasibility on these selected challenge cases; repeated-run reliability, component ablations, scheduling eficiency, and domain validation remain subjects for future evaluation.

Our contributions are threefold. First, we introduce Research-Space Symbolic Regression and, to our knowledge, the first formulation of Research-State Search as the sequential decision problem underlying this paradigm. Second, we develop AutoSR as a fully automated implementation that combines persistent Research States, independent scientific review, autonomous computational investigation, and MCTS-based search. Third, we demonstrate exact recovery on nine selected challenge equations spanning two benchmark suites. The remainder of the paper provides the related work review, defines the AutoSR method, presents the selected benchmark evaluation, and discusses the implications, limitations, and future evaluation of treating symbolic regression as automated scientific inquiry.

## 2 Related Work

## 2.1 Conventional and Prior-Informed Symbolic Regression

Symbolic regression jointly searches for the structure and parameters of an explicit mathematical expression. Genetic programming established expression trees, evolutionary variation, and fitness-based selection as a general formulation of this search [Koz92]. Early systems for scientific discovery extended expression search with active probing of dynamical systems [BL07] and searches for invariant relations in experimental data [SL09]. Subsequent approaches span deterministic model building such as FFX [McC11], sparse identification over a prescribed function library in SINDy [BPK16], compressedsensing feature construction in SISSO [OCA<sup>+</sup>18], probabilistic search over expression trees in Bayesian SR [JFK<sup>+</sup>19], and physics-inspired decomposition by symmetry, separability, and dimensional analysis in AI Feynman [UT20]. Modern systems such as PySR and FEX combine expressive search spaces with specialized optimization [Cra23, JWY23], while parallel symbolic enumeration shares subtree computations to evaluate very large collections of expressions eficiently [RXG<sup>+</sup>26].

Across these algorithmic families, candidates are ordinarily selected by predictive error together with parsimony, sparsity, a structural prior, or a related model-selection criterion. SRBench compares contemporary methods on both known-ground-truth and black-box tasks and shows that predictive accuracy, symbolic recovery, and expression complexity expose diferent aspects of performance [LCOB<sup>+</sup>21]. SRSD further emphasizes realistic sampling ranges, irrelevant variables, and normalized tree-edit distance for scientific-discovery evaluation [MCIU24], while LLM-SRBench adds transformed and synthetic scientific equations together with safeguards against equation memorization [SNM<sup>+</sup>25]. These benchmarks make clear that syntactic size is a useful operational proxy, but neither low error nor a short expression alone establishes that a recovered relation has the intended scientific meaning.

Prior knowledge can narrow this ambiguity when it can be formalized. It has been incorporated through data generated from known constraints [KDB20], bounds and derivative-shape restrictions such as monotonicity or convexity [KdFB<sup>+</sup>22], dimensional consistency enforced during expression construction [TID23], user-specified structural hypotheses for neural SR [BBK23], and probabilistic priors over functions, structures, and parameters [BDF23]. AI-Descartes evaluates candidate formulas against logical background axioms $\mathrm { [ C D A ^ { + } 2 3 ] }$ , and AI-Hilbert jointly searches for polynomial laws consistent with data and background theory while producing formal certificates $\mathrm { [ C W C D ^ { + } 2 4 ] }$ . AutoSR treats these formal constraints as valuable special cases: its problem specification may include them, but also admits scientific knowledge that remains in natural language and must influence hypothesis formation, computational testing, and criticism rather than a single fixed objective.

## 2.2 Learned and Tree-Search-Guided Symbolic Regression

Learned SR methods replace or augment hand-designed proposal distributions with neural priors. Deep Symbolic Regression trains a recurrent policy with a risk-seeking objective $[ \mathrm { P L L M ^ { + } 2 1 } ]$ , whereas NeSymReS and end-to-end Transformer systems pretrain on synthetic equations and infer expressions from sets of numerical observations $[ \mathrm { B B N ^ { + } 2 1 }$ , KdLC22]. A unified deep-SR framework combines recursive simplification, neural-guided search, pretraining, genetic programming, and linear models, illustrating that learned generation and explicit search are complementary rather than mutually exclusive [LLY<sup>+</sup>22].

MCTS provides a general mechanism for allocating simulations in a tree by repeatedly selecting, expanding, evaluating, and backing up nodes [Cou07, BPW<sup>+</sup>12]. UCT applies an upper-confidence bandit rule at each internal node to balance exploration and exploitation [KS06], and progressive widening extends the approach to spaces in which enumerating every action at a node is impractical [CHS<sup>+</sup>11]. In SR, Symbolic Physics Learner uses MCTS to construct mathematical expression trees [SLWS23]; DGSR-MCTS uses a pretrained and online-refined neural mutation model [KLLV23]; and TPSR adds MCTS lookahead to Transformer decoding with accuracy–complexity feedback [SMBFR23]. RSRM combines expression-tree MCTS with double Q-learning and learned subtree operators [XLS24], while SR4MDL searches subformulas using a learned minimum-description-length objective [YDLJ25]. More recent methods introduce extreme-bandit allocation and nonlocal mutation or crossover actions [HHX<sup>+</sup>25], or search permutation-invariant expression graphs with neural and constraint guidance [XAQQ25]. Thus MCTS itself, and its use in SR, are established; the distinction in AutoSR is the state being searched and evaluated. Previous SR applications assign nodes to tokens, subexpressions, formulas, or expression graphs, whereas one AutoSR expansion produces a persistent proposer–experiment– review record whose value and ancestry determine the allocation of subsequent investigations.

## 2.3 LLM-Guided and Agentic Symbolic Regression

LLMs introduce scientific language and program synthesis as additional proposal mechanisms. In-Context Symbolic Regression iteratively refines LLM-generated formulas with external coeficient optimization [MHDM24], and LLM4ED uses LLMs as black-box optimizers and evolutionary operators for equation discovery [DCW<sup>+</sup>24]. The Scientific Generative Agent couples LLM proposals with simulation and continuous optimization in a bilevel loop [MWG<sup>+</sup>24]. LaSR evolves a natural-language concept library alongside symbolic hypotheses [GSC<sup>+</sup>24], while LLM-SR generates programmatic equation skeletons inside a multi-island evolutionary search using scientific context [SMG<sup>+</sup>25]. RAG-SR instead trains an online language model and retrieves previously searched expressions to guide feature construction without large-scale pretraining $[ \mathrm { Z C X ^ { + } 2 5 } ]$

Recent work moves from LLM proposal components toward longer-horizon agents. SR-Scientist gives an agent tools for data analysis and equation evaluation and preserves strong equations in an experience bufer [XSL26]. RESTART uses residual diagnostics for short-term structural correction and distills successful refinements into a long-term structure library [LP26]; Deliberate Evolution separates equation generation from search guidance through adaptive operators, diagnostic tools, and reflective trajectory memory $[ \mathrm { P Z L ^ { + } 2 6 } ] ;$ ; and the Iterated Agent uses natural-language rationales as semantic operators in an evolutionary search [SCZ<sup>+</sup>25]. Other concurrent preprints let an LLM control a conventional SR engine [XLL<sup>+</sup>26], combine granular influence feedback with term generation and MCTS [SHS<sup>+</sup>26], or coordinate Generator, Analyst, Simplifier, and Reviewer roles through process memory [ZLX<sup>+</sup>26]. These systems already go substantially beyond a formula–score loop, so AutoSR does not claim the first use of tools, diagnostics, memory, multiple roles, or iterative scientific reasoning in SR. Its narrower contribution is to make each complete, artifact-rich investigation a persistent node of global search, keep its memory local to an ancestral branch, append a separate review to the node before reward backup, and synthesize the resulting tree as auditable evidence rather than using the trace only to improve subsequent equation proposals.

Tree search over richer work products also appears in general research agents: AIDE searches codesolution states $[ \mathrm { J } \mathrm { S } \mathrm { S } ^ { + } 2 5 ]$ , and AI Scientist-v2 searches branching experimental workflows $[ \mathrm { Y L L ^ { + } 2 5 } ]$ AutoSR specializes this broader direction to scientific equation discovery by defining the contents, inheritance, independent review, and search semantics of a Research State.

## 3 Method

## 3.1 Problem Formulation and System Overview

AutoSR receives a problem specification $\mathcal { P }$ and a computation budget B. The specification defines the scientific question, the observed data, the meanings of the variables, the available supporting evidence, the evaluation criterion, and any requirements that an acceptable relation should satisfy. Given $( \mathcal { P } , B )$ the system returns not only a leading relation $\hat { f } ,$ , but a final report $\mathcal { R } _ { \mathrm { f i n a l } }$ that connects this relation to its evidence, limitations, research history, and credible alternatives:

$$
\operatorname { A u t o S R } ( { \mathcal { P } } , B ) \longrightarrow ( { \hat { f } } , { \mathcal { R } } _ { \operatorname { f i n a l } } ) .\tag{1}
$$

The task is therefore broader than minimizing a fixed loss over an expression space. AutoSR must use the available scientific information to produce, test, criticize, and compare candidate relations while deciding how to allocate the remaining budget.

The concrete interface to $\mathcal { P }$ consists of problem.md, data, and supporting files. The supporting material may include background knowledge, figures, documents, existing code, and explicit scientific requirements. Requirements such as dimensional consistency, monotonicity, convexity, asymptotic behavior, special values, and known theoretical limits can often be checked directly [KdFB<sup>+</sup>22, TID23, CDA<sup>+</sup>23]. Other priors, including causal expectations, ontology, and mechanistic plausibility, may remain expressed in scientific language rather than in a fixed grammar or penalty. AutoSR supplies the complete specification to both its proposal and review roles, allowing formal requirements and broader scientific knowledge to influence hypothesis generation, computational testing, and criticism.

Starting from a root that contains ${ \mathcal { P } } _ { : }$ AutoSR builds a tree of Research States. Each tree expansion executes an automated proposer–reviewer cycle, and the search controller selects which branch to initiate or extend next. Progressive-widening Monte Carlo tree search (PW-MCTS) [BPW<sup>+</sup>12, CHS<sup>+</sup>11] balances new lines of inquiry against continuations of promising branches, while a pending-aware UCT rule [KS06, $\mathrm { L C Y ^ { + } 2 0 ] }$ permits multiple long-running expansions to proceed asynchronously. A complementary loop applies established symbolic-regression packages, and the final synthesis compares the strongest candidates from both processes. Figure 1 summarizes this workflow.

## 3.2 Research States and Branch-Specific Memory

A Research State is the persistent record produced by one research attempt. For a non-root node $i ,$ write

$$
\begin{array} { r } { S _ { i } = ( f _ { i } , m _ { i } , \mathcal { A } _ { i } , \rho _ { i } , s _ { i } ; p _ { i } ) , } \end{array}\tag{2}
$$

where $f _ { i }$ is the candidate relation, $m _ { i }$ is its scientific motivation and reasoning, $\mathbf { \mathcal { A } } _ { i }$ contains the computational artifacts and findings, $\rho _ { i }$ is the review record, $s _ { i }$ is the reviewer score, and $p _ { i }$ identifies the parent Research State. The artifacts may include code, execution results, fitted parameters, figures, residual analyses, diagnostics, and unsuccessful tests. The scalar $s _ { i }$ supports search control, whereas the full record $( m _ { i } , { A _ { i } } , \rho _ { i } )$ preserves the evidence needed to interpret that score.

The central artifact in each state is ansatz.md. It serves as both summary and index: it states the current hypothesis, condenses the main argument and results, records the review, and points to the supporting files. The local contents of $\boldsymbol { S } _ { i }$ record attempt $i ,$ while the ordered path from the root to i forms the memory of the corresponding line of inquiry. When a proposer extends state $i ,$ it receives the ansatz.md files on this ancestral path in order. It does not receive summaries from sibling branches. A child of the root therefore begins an independent investigation, whereas a deeper child can build on the useful ideas, failed trials, and criticism accumulated by its ancestors. This separation allows competing explanations to develop independently without discarding the research history within any one branch.

![](images/44ed867ebec2aa939b026914171e6bcba1081df8ed515d822a1198248dae62f1.jpg)  
Figure 1: AutoSR uses a proposer–reviewer loop to build a tree of Research States with progressivewidening MCTS. The problem specification supplies data, scientific context, and explicit requirements to the investigation. A secondary actor–reviewer loop advances conventional symbolic regression. The Prompt Economy annotation uses the AutoSR-specific reuse metric defined in Equation 3. The final report combines the strongest candidates, their supporting evidence, and the resulting search landscape.

## 3.3 Automated Investigation and Independent Review

The proposer receives P together with the branch memory selected by the search controller. It develops an ansatz, writes and executes code, fits parameters, studies plots and numerical behavior, and revises the candidate in response to the resulting evidence. The proposer may change transformations, objectives, optimizers, or computational methods when the current formulation is unstable or scientifically misleading, and it may call symbolic-regression packages or other scientific tools when useful. These decisions are not prescribed as a fixed pipeline: they are part of the investigation recorded in $( m _ { i } , A _ { i } )$

After the proposal is complete, a separate reviewer role reads the problem specification, ansatz.md, and the supporting artifacts. It reruns key computations, checks the stated evaluation criterion, examines residuals, and searches for leakage, hidden parameters, violated requirements, or unsupported conclusions. Held-out data and hidden evaluation instructions can be supplied through a reviewer-only channel so that the proposer cannot optimize directly against them. The reviewer records its diagnosis $\rho _ { i }$ and score $s _ { i }$ in ansatz.md. A mandatory scientific requirement remains an acceptance condition rather than a tradeable improvement in numerical fit. The scalar score is used to allocate search efort; it does not replace the reviewer’s evidence or the scientific comparison preserved in the final report. Appending this review completes $s _ { i } ,$ after which its descendants can inherit both the proposal and its criticism.

Producing a Research State can require repeated reasoning, coding, execution, and revision. AutoSR draws on Prompt Economy, which treats the maintenance of role prompts and handof protocols as a bounded engineering cost and favors reusing a stable prompt surface across many useful invocations $[ \mathrm { S R Z ^ { + } 2 6 , ~ S R Y ^ { + } 2 6 } ]$ ]. We operationalize this principle for AutoSR with a time-based, system-specific reuse metric. Let $t _ { i }$ be the time spent writing or revising prompt $i ,$ and let $v _ { i }$ be the number of times that prompt is executed:

$$
\mathrm { R O I } _ { \mathrm { A u t o S R } } = \frac { \sum _ { i } v _ { i } } { \sum _ { i } t _ { i } } .\tag{3}
$$

This AutoSR-specific quantity adapts the Prompt Economy principle rather than reproducing the ROI formula in the cited work; it is a system-design criterion, not the scientific objective in Equation 1. Reusing the same proposer and reviewer roles at every expansion increases the number of autonomous investigations without requiring a new prompt for each candidate or domain.

## 3.4 Research-State Search

Research-State Search is the sequential decision problem of choosing which investigation to initiate or extend under the remaining budget. At scheduling step $t ,$ let $\mathcal { T } _ { t }$ denote the tree of created Research States, including completed states and expansions still in flight. Each completed proposer–reviewer cycle supplies one new node and one observed reward; in this sense, a complete research attempt plays the role of an MCTS evaluation [Cou07, BPW<sup>+</sup>12]. AutoSR allocates these evaluations with progressive-widening MCTS (PW-MCTS; $\mathrm { C H S ^ { + } 1 1 } )$

For node $i ,$ let $C _ { i }$ be its set of created children, $N _ { i }$ the number of completed evaluations whose selected paths pass through $i ,$ and $O _ { i }$ the corresponding number of evaluations that have been dispatched but have not yet returned. The controller may create another child of i when

$$
| C _ { i } | < k _ { \mathrm { p w } } \operatorname* { m a x } ( N _ { i } + O _ { i } , 1 ) ^ { \beta } .\tag{4}
$$

Here $k _ { \mathrm { p w } } > 0$ controls the overall width and $0 < \beta < 1$ makes the allowed branching factor grow sublinearly with the number of completed and pending evaluations. If the condition holds, the proposer creates a new child from the branch memory of $i ;$ otherwise, the controller continues through an existing child according to the tree policy below. Progressive widening therefore admits more independent or descendant investigations as evidence accumulates without allowing early branching to consume the entire budget.

The reviewer score can depend on the evaluation criterion, scientific requirements, and the supporting analysis, and its raw scale may difer across problems. Let $\nu _ { t }$ be the set of Research States completed by scheduling step $t ,$ with larger $s _ { i }$ indicating a stronger review. AutoSR converts each score to its empirical percentile within the current completed set:

$$
r _ { i } ( t ) = 1 0 \times \frac { 1 } { | \mathscr { V } _ { t } | } \sum _ { \ell \in \mathscr { V } _ { t } } \mathbf { 1 } [ s _ { \ell } \leq s _ { i } ] .\tag{5}
$$

Percentile rewards lie in (0, 10] and are interpreted at the current scheduling step because their reference set grows as reviews return. This common rank scale prevents the magnitude or units of a task-specific evaluator from changing the exploration pressure. For a child $j ,$ let $Q _ { j } ( t )$ be the mean percentile reward of the completed Research States in the subtree rooted at $j$

Because a proposer–reviewer cycle may run for minutes or hours, scheduling evaluations only after earlier calls finish would leave substantial parallel capacity unused. AutoSR therefore adapts Watch the Unobserved in UCT $( \mathrm { W U - U C T ; L C Y ^ { + } 2 0 } )$ to Research-State Search. Among existing children, the controller selects according to

$$
U _ { t } ( i , j ) = Q _ { j } ( t ) + c \sqrt { \frac { \log \operatorname* { m a x } ( N _ { i } + O _ { i } , 1 ) } { N _ { j } + O _ { j } } } .\tag{6}
$$

The in-flight counts allow dispatched work to afect both selection and progressive widening before its reviewer score becomes observable. They discourage redundant simultaneous expansion of the same branch while retaining the exploitation term supplied by completed reviews.

One asynchronous search iteration consists of four operations. First, the controller traverses the tree from the root, applying Equation 4 to decide whether to create a child and Equation 6 when it must continue through existing children. Second, before dispatching the selected proposer–reviewer cycle, it increments $O _ { i }$ along the selected path so that later scheduling decisions observe the pending work. Third, the proposer and reviewer produce the new Research State without blocking other dispatched evaluations. Finally, when the review returns, the controller decrements the path’s in-flight counts, increments its completed counts, adds the new score to $\nu _ { t } .$ and updates the percentile rewards and subtree means used by subsequent selections. This cycle continues until budget B is exhausted. The controller then stops dispatching new work, resolves the remaining in-flight evaluations, and collects the completed Research States for final synthesis.

## 3.5 Conventional Symbolic-Regression Loop

A secondary Prompt Economy loop applies conventional symbolic regression to the same problem specification. This loop complements the flexible investigations in the Research-State tree with mature expression-search engines and their specialized optimizers. Its actor configures and runs methods such as PySR [Cra23], SISSO [OCA<sup>+</sup>18], and FEX [JWY23], then studies their equations, residuals, and failure modes. Its reviewer examines the search space and returned candidates, identifies overlooked transformations or settings, and directs the next package run. One package execution is therefore an experiment within a sustained actor–reviewer process rather than the complete conventional-SR contribution. The loop passes its strongest equations, diagnostics, and reviews to the same fina synthesis used for Research-State candidates.

## 3.6 Final Selection and Report Synthesis

After both search processes end, AutoSR ranks completed candidates by reviewer score while retaining the corresponding requirement checks and review records. A candidate that violates a requirement identified by the problem specification cannot be presented as an acceptable scientific relation solely because it attains a strong numerical fit. The leading eligible candidate becomes $\hat { f }$ in Equation 1, but ranking determines only which result the report presents first; it does not discard competitive alternatives or their evidence.

The first part of $\mathcal { R } _ { \mathrm { f i n a l } }$ explains the leading result through its formula, measured performance, scientific interpretation, diagnostics, and limitations. The second reconstructs the Research-State search landscape: it presents a report-ready tree, groups states into expression families, summarizes the evidence associated with each family, and traces clear evolution paths through the parent relation in Equation 2. An appendix presents the next four candidates as standalone alternatives. When several relations remain competitive, the report compares their empirical and scientific implications instead of forcing an unsupported single conclusion. The equation remains the central result, while the preserved artifacts and reviewer records make the investigation behind that result available for audit.

## 4 Experiments

The present evaluation asks whether AutoSR can recover exact relations on selected equations that pose substantial structural challenges to existing symbolic-regression systems. We report nine recovery cases drawn from two benchmark suites. Because the problems were selected for dificulty and one recovered trajectory is reported for each task, these experiments demonstrate capability on the selected cases rather than benchmark-wide accuracy or stochastic reliability.

## 4.1 Evaluation Scope and Protocol

The first challenge set comes from cp3-bench [TK25], which evaluates twelve symbolic-regression methods on twenty-eight problems in cosmology and astroparticle physics. We selected C3g, C5f, and C6c because none of the twelve methods recovered their ground-truth relations. The three tasks cover dependent input features, a single expression that unifies cusp and core density profiles, and a strongly oscillatory gravitational-wave signal with an additional mass parameter.

The second challenge set comes from LSR-Transform in LLM-SRBench [SNM<sup>+</sup>25]. LSR-Transform contains 111 equations obtained by solving established scientific relations for unusual target variables. The best oficial symbolic accuracy is 31.53%. We selected six tasks that cover complementary structures: nested radicals, trigonometric inversion, fractional powers, paired exponentials, a highdimensional rational expression, and logarithmic inversion. The selection was based on structural complexity and was completed before we examined public per-task results from other systems.

Every task uses the same core search procedure and role prompts. The proposer receives the task description and training data, while the ground-truth equation remains hidden until the search ends. Internet access is prohibited. An external evaluator checks algebraic equivalence after the run. Reported times measure only the proposer sessions that produced the recovered relation. They exclude other explored branches, reviewer calls, parallel work, and conventional-SR runs, and therefore should not be interpreted as the total computational cost of AutoSR.

## 4.2 Selected cp3-bench Recovery

AutoSR recovered a relation algebraically equivalent to the ground truth on each of the three selected cp3-bench problems, where every published baseline failed to recover the corresponding analytical relation.

Table 1: Results on the selected cp3-bench problems. MSE values are reported to two significant digits.
<table><tr><td>Problem</td><td>Best cp3-bench result</td><td>AutoSR result</td><td>Proposer time</td></tr><tr><td>C3g</td><td>QLattice; MSE  $4 . 7 \times 1 0 ^ { - 6 } ;$  ground truth not recovered</td><td>Algebraically equivalent; MSE  $9 . { \overset { - } { 0 } } \times 1 0 ^ { - 1 9 }$ </td><td>8 min 58 s</td></tr><tr><td>C5f</td><td>GPG; MSE  $3 . 3 \times 1 0 ^ { - 5 }$  ; ground truth not recovered</td><td>Algebraically equivalent; MSE  $6 . 2 \times 1 0 ^ { - 3 2 }$ </td><td>15 min 44 s</td></tr><tr><td>C6c</td><td> $\mathrm { G P G } ; \mathrm { M S E } ~ 1 . 1 \times 1 0 ^ { - 1 }$  ; ground truth not recovered</td><td>Algebraically equivalent; MSE  $6 . 1 \times 1 0 ^ { - 3 0 }$ </td><td>33 min 1 s</td></tr></table>

All three recovered formulas simplify to the known analytical relations. Their surface forms difer from the benchmark expressions, which is consistent with nontrivial algebraic reconstruction but does not by itself rule out the influence of memorized scientific knowledge. On three problems where twelve established systems recovered zero relations, AutoSR recovered three.

## 4.3 Selected LSR-Transform Recovery

The LLM-SRBench paper reports aggregate results but does not release the original baselines’ predictions for individual LSR-Transform tasks. We therefore report AutoSR’s per-task results without claiming a direct per-task comparison with the published baselines.

Table 2: Results on the selected LSR-Transform problems. NRMSE values are reported to two significant digits.
<table><tr><td>Problem</td><td>AutoSR result</td><td>Proposer time</td></tr><tr><td>I.32.17_4_3</td><td>Algebraically equivalent; NRMSE  $1 . 8 \times 1 0 ^ { - 7 }$ </td><td>16 min 13 s</td></tr><tr><td>I.29.16_0_0</td><td>Algebraically equivalent; NRMSE  $1 . 2 \times 1 0 ^ { - 7 }$ </td><td>12 min 32 s</td></tr><tr><td>II.6.15a_2_0</td><td>Algebraically equivalent; NRMSE  $1 . 0 \times 1 0 ^ { - 7 }$ </td><td>7 min 47 s</td></tr><tr><td>II.35.18_0_0</td><td>Algebraically equivalent; NRMSE  $3 . 6 \times 1 0 ^ { - 7 }$ </td><td>6 min 26 s</td></tr><tr><td>II.36.38_3_0</td><td>Algebraically equivalent; NRMSE  $8 . 0 \times 1 0 ^ { - 8 }$ </td><td>9 min 7 s</td></tr><tr><td>III.4.33_3_0</td><td>Algebraically equivalent; NRMSE  $7 . 2 \times 1 0 ^ { - 8 }$ </td><td>8 min 9 s</td></tr></table>

AutoSR recovered an algebraically equivalent relation on all six tasks. The six equations require substantially diferent transformations, while the search procedure and agent roles remain unchanged.

## 4.4 Scope of the Present Evidence

Taken together, the nine cases show that AutoSR can recover algebraically exact relations across several dificult expression structures while preserving a common search procedure and set of agent roles. They do not estimate a benchmark-wide recovery rate, the probability of success under repeated runs, or the contribution of any individual system component. The proposer times in Tables 1 and 2 also do not measure total system cost. Accordingly, the present results should be interpreted as selected challenge-case demonstrations rather than a comprehensive comparison with existing symbolicregression systems. Section 6 specifies the controlled and domain evaluations required to extend this evidence.

## 5 Discussion

The significance of the selected recoveries lies in the common research workflow that produced them. Across the selected problems, AutoSR retained the same core search procedure and agent roles. These cases show that a common sustained research workflow can recover structurally diverse equations.

The central distinction is the unit over which the system searches. A conventional expression-level search can retain only a formula and its score, whereas a Research State is required to preserve the motivation, computations, failed tests, and criticism associated with one attempt. This persistent record allows evidence to influence what is investigated next, while branch-specific memory permits competing explanations to develop independently. Within this structure, scientific priors can guide not only which relation is accepted, but also which hypothesis is proposed, which implication is tested, and which branch receives further computation.

Persistent Research States also support a final report that connects the leading relation to its evidence, limitations, research history, and credible alternatives. In this setting, fully automated means that AutoSR can continue the investigation after receiving a problem specification and computation budget; it does not remove the scientist’s role in formulating the problem or interpreting the result. AutoSR therefore provides a concrete implementation of Research-Space Symbolic Regression in which the search process retains the broader investigation rather than only its final equation.

## 6 Limitations and Future Evaluation

The present evidence is limited to nine selected challenge cases, with one recovered trajectory reported for each. It does not establish benchmark-wide accuracy, repeated-run reliability, or the computation required to obtain a recovery with a specified probability. The reported proposer times also exclude reviews, unsuccessful and parallel branches, conventional-SR runs, token usage, and orchestration overhead. Future evaluation should therefore report repeated independent runs, success at fixed budgets, total model calls and tokens, completed Research States, wall-clock time, concurrency, and time to first recovery.

The experiments do not isolate the source of the successful recoveries. Matched-budget ablations should compare direct LLM generation, a single tool-using agent, a proposer–reviewer loop without tree search, flat best-of-N Research States, and the complete system. The conventional-SR loop should be removed in a complementary ablation, and a supplied-context ablation should compare data and variable names alone with the full scientific background and requirements. These comparisons would separate the efects of tool use, review, branch-specific memory, Research-State Search, and explicitly supplied scientific context.

Exact benchmark recovery also does not establish that AutoSR can distinguish scientifically credible relations from numerically competitive alternatives under noise, extrapolation, incomplete theory, or conflicting evidence. Nor do the current experiments assess the final reports, alternative relations, or preserved search landscapes. Evaluation on newly constructed problems and private domain data is needed because memorized scientific knowledge cannot be ruled out on established benchmarks, particularly when pretrained models are used [MCIU24, SNM<sup>+</sup>25]. Separate proposer and reviewer roles also provide procedural separation without guaranteeing independent judgment when both rely on similar pretrained knowledge. External computational checks and domain-expert review will therefore remain necessary.

The search scheduler requires direct evaluation as well. Comparisons of raw reviewer scores with empirical-percentile rewards and sequential UCT with WU-UCT should measure result quality at fixed computation, redundant concurrent branch selection, worker utilization, and time to the best candidate. Ongoing studies in materials science, mathematical biology, and geotechnical engineering will provide the test of whether AutoSR can use problem-specific requirements to produce scientifically useful relations.

Code availability. The implementation of AutoSR is available in the AutoResearch-Factory/AutoSR.

## References

[BBK23] Tommaso Bendinelli, Luca Biggio, and Pierre-Alexandre Kamienny. Controllable neural symbolic regression. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 2063–2077. PMLR, 2023.

[BBN<sup>+</sup>21] Luca Biggio, Tommaso Bendinelli, Alexander Neitz, Aurelien Lucchi, and Giambattista Parascandolo. Neural Symbolic Regression that Scales. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 936–945. PMLR, 2021.

[BDF23] Deaglan J. Bartlett, Harry Desmond, and Pedro G. Ferreira. Priors For Symbolic Regression. In Proceedings of the Companion Conference on Genetic and Evolutionary Computation, GECCO ’23 Companion, pages 2402–2411, New York, NY, USA, 2023. Association for Computing Ma chinery.

[BL07] Josh Bongard and Hod Lipson. Automated reverse engineering of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 104(24):9943–9948, 2007.

[Bol84] Ludwig Boltzmann. Ableitung des stefan’schen gesetzes, betrefend die abh¨angigkeit der w¨armes trahlung von der temperatur aus der electromagnetischen lichttheorie. Annalen der Physik, 258(6):291–294, 1884.

[BPK16] Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15):3932–3937, 2016.

[BPW<sup>+</sup>12] Cameron B. Browne, Edward Powley, Daniel Whitehouse, Simon M. Lucas, Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. A survey of monte carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in Games, 4(1):1–43, 2012.

[CDA<sup>+</sup>23] Cristina Cornelio, Sanjeeb Dash, Vernon Austel, Tyler R. Josephson, Joao Goncalves, Kenneth L. Clarkson, Nimrod Megiddo, Bachir El Khadir, and Lior Horesh. Combining data and theory for derivable scientific discovery with AI-Descartes. Nature Communications, 14:1777, 2023.

[CHS<sup>+</sup>11] Adrien Cou¨etoux, Jean-Baptiste Hoock, Nataliya Sokolovska, Olivier Teytaud, and Nicolas Bonnard. Continuous upper confidence trees. In Learning and Intelligent Optimization, volume 6683 of Lecture Notes in Computer Science, pages 433–445. Springer, 2011.

[Cou07] R´emi Coulom. Eficient Selectivity and Backup Operators in Monte-Carlo Tree Search. In Computers and Games, volume 4630 of Lecture Notes in Computer Science, pages 72–83. Springer Berlin Heidelberg, 2007.

[Cra23] Miles Cranmer. Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv preprint arXiv:2305.01582, 2023.

[CWCD<sup>+</sup>24] Ryan Cory-Wright, Cristina Cornelio, Sanjeeb Dash, Bachir El Khadir, and Lior Horesh. Evolving scientific discovery by unifying data and background knowledge with AI Hilbert. Nature Communications, 15:5922, 2024

[DCW<sup>+</sup>24] Mengge Du, Yuntian Chen, Zhongzheng Wang, Longfeng Nie, and Dongxiao Zhang. LLM4ED: Large language models for automatic equation discovery, 2024.

[Ein05] Albert Einstein. Ueber einen die Erzeugung und Verwandlung des Lichtes betrefenden heuristischen Gesichtspunkt. Annalen der Physik, 322(6):132–148, 1905.

[GSC<sup>+</sup>24] Arya Grayeli, Atharva Sehgal, Omar Costilla-Reyes, Miles Cranmer, and Swarat Chaudhuri. Symbolic Regression with a Learned Concept Library. In Advances in Neural Information Pro cessing Systems, volume 37, pages 44678–44709. Curran Associates, Inc., 2024.

[HHX<sup>+</sup>25] Zhengyao Huang, Daniel Huang, Tiannan Xiao, Dina Ma, Zhenyu Ming, Hao Shi, and Yuanhui Wen. Improving Monte Carlo Tree Search for Symbolic Regression. In Advances in Neural Information Processing Systems, volume 38, pages 40819–40856. Curran Associates, Inc., 2025.

[JFK<sup>+</sup>19] Ying Jin, Weilin Fu, Jian Kang, Jiadong Guo, and Jian Guo. Bayesian symbolic regression, 2019.

[JSS<sup>+</sup>25] Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, and Yuxiang Wu. AIDE: AI-driven exploration in the space of code, 2025.

[JWY23] Zhongyi Jiang, Chunmei Wang, and Haizhao Yang. Finite expression methods for discovering physical laws from data. arXiv preprint arXiv:2305.08342, 2023.

[KDB20] Jiˇr´ı Kubal´ık, Erik Derner, and Robert Babuˇska. Symbolic Regression Driven by Training Data and Prior Knowledge. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference, GECCO ’20, pages 958–966, New York, NY, USA, 2020. Association for Computing Machinery.

[KdFB<sup>+</sup>22] Gabriel Kronberger, Fabr´ıcio Olivetti de Fran¸ca, Bogdan Burlacu, Christian Haider, and Michael Kommenda. Shape-constrained symbolic regression—improving extrapolation with prior knowl edge. Evolutionary Computation, 30(1):75–98, 2022.

[KdLC22] Pierre-Alexandre Kamienny, St´ephane d’Ascoli, Guillaume Lample, and Fran¸cois Charton. Endto-end Symbolic Regression with Transformers. In Advances in Neural Information Processing Systems, volume 35, pages 10269–10281. Curran Associates, Inc., 2022.

[KLLV23] Pierre-Alexandre Kamienny, Guillaume Lample, Sylvain Lamprier, and Marco Virgolin. Deep generative symbolic regression with monte-carlo-tree-search. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 15655–15668. PMLR, 2023.

[Koz92] John R. Koza. Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press, Cambridge, MA, 1992.

[KS06] Levente Kocsis and Csaba Szepesv´ari. Bandit based monte-carlo planning. In Machine Learning: ECML 2006, volume 4212 of Lecture Notes in Computer Science, pages 282–293. Springer, 2006.

[LCOB<sup>+</sup>21] William La Cava, Patryk Orzechowski, Bogdan Burlacu, Fabr´ıcio Olivetti de Fran¸ca, Marco Virgolin, Ying Jin, Michael Kommenda, and Jason H. Moore. Contemporary Symbolic Regression Methods and their Relative Performance. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.

[LCY<sup>+</sup>20] Anji Liu, Jianshu Chen, Mingze Yu, Yu Zhai, Xuewen Zhou, and Ji Liu. Watch the unobserved: A simple approach to parallelizing monte carlo tree search. In International Conference on Learning Representations, 2020.

[LLY<sup>+</sup>22] Mikel Landajuela, Chak Shing Lee, Jiachen Yang, Ruben Glatt, Claudio P. Santiago, Ignacio Aravena, Terrell Mundhenk, Garrett Mulcahy, and Brenden K. Petersen. A Unified Framework for Deep Symbolic Regression. In Advances in Neural Information Processing Systems, volume 35, pages 33985–33998. Curran Associates, Inc., 2022.

[LP26] Yunlun Li and Sinno Jialin Pan. Robust equation structure learning with adaptive refinement. In International Conference on Learning Representations, 2026.

[McC11] Trent McConaghy. FFX: Fast, scalable, deterministic symbolic regression technology. In Genetic Programming Theory and Practice IX, pages 235–260. Springer, 2011.

[MCIU24] Yoshitomo Matsubara, Naoya Chiba, Ryo Igarashi, and Yoshitaka Ushiku. Rethinking symbolic regression datasets and benchmarks for scientific discovery. Journal of Data-centric Machine Learning Research, 1(3):1–38, 2024.

[MHDM24] Matteo Merler, Katsiaryna Haitsiukevich, Nicola Dainese, and Pekka Marttinen. In-context symbolic regression: Leveraging large language models for function discovery. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pages 427–444. Association for Computational Linguistics, 2024.

[MWG<sup>+</sup>24] Pingchuan Ma, Tsun-Hsuan Wang, Minghao Guo, Zhiqing Sun, Joshua B. Tenenbaum, Daniela Rus, Chuang Gan, and Wojciech Matusik. LLM and simulation as bilevel optimizers: A new paradigm to advance physical scientific discovery. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 33940–33962. PMLR, 2024.

[OCA<sup>+</sup>18] Runhai Ouyang, Stefano Curtarolo, Emre Ahmetcik, Matthias Schefler, and Luca M. Ghiringhelli. SISSO: A compressed-sensing method for identifying the best low-dimensional descriptor in an immensity of ofered candidates. Physical Review Materials, 2(8):083802, 2018.

[Pla01] Max Planck. Ueber das gesetz der energieverteilung im normalspectrum. Annalen der Physik, 309(3):553–563, 1901.

[PLLM<sup>+</sup>21] Brenden K. Petersen, Mikel Landajuela Larma, T. Nathan Mundhenk, Claudio P. Santiago, Soo K. Kim, and Joanne T. Kim. Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. In International Conference on Learning Representations, 2021.

[PZL<sup>+</sup>26] Xinyu Pang, Zhanke Zhou, Xuan Li, Fangrui Lv, Shanshan Wei, Sen Cui, Bo Han, and Changshui Zhang. Deliberate evolution: Agentic reasoning for sample-eficient symbolic regression with LLMs, 2026.

[Ray00] Lord Rayleigh. LIII. remarks upon the law of complete radiation. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 49(301):539–540, 1900.

[RXG<sup>+</sup>26] Kai Ruan, Yilong Xu, Ze-Feng Gao, Yang Liu, Yike Guo, Ji-Rong Wen, and Hao Sun. Discovering physical laws with parallel symbolic enumeration. Nature Computational Science, 6:53–66, 2026.

[SCZ<sup>+</sup>25] Zhuo-Yang Song, Zeyu Cai, Shutao Zhang, Jiashen Wei, Jichen Pan, Shi Qiu, Qing-Hong Cao, Tie-Jiun Hou, Xiaohui Liu, Ming-xing Luo, and Hua Xing Zhu. Iterated agent for symbolic regression, 2025.

[SHS<sup>+</sup>26] Evgeny S. Saveliev, Samuel Holt, Nabeel Seedat, David L. Bentley, Jim Weatherall, and Mi haela van der Schaar. Influence-guided symbolic regression: Scientific discovery via LLM-driven equation search with granular feedback, 2026.

[SL09] Michael Schmidt and Hod Lipson. Distilling free-form natural laws from experimental data. Science, 324(5923):81–85, 2009.

[SLWS23] Fangzheng Sun, Yang Liu, Jian-Xun Wang, and Hao Sun. Symbolic physics learner: Discovering governing equations via monte carlo tree search. In International Conference on Learning Representations, 2023.

[SMBFR23] Parshin Shojaee, Kazem Meidani, Amir Barati Farimani, and Chandan K. Reddy. Transformer based Planning for Symbolic Regression. In Advances in Neural Information Processing Systems, volume 36, pages 45907–45919. Curran Associates, Inc., 2023.

[SMG<sup>+</sup>25] Parshin Shojaee, Kazem Meidani, Shashank Gupta, Amir Barati Farimani, and Chandan K. Reddy. LLM-SR: Scientific equation discovery via programming with large language models. In The Thirteenth International Conference on Learning Representations, 2025.

[SNM<sup>+</sup>25] Parshin Shojaee, Ngoc-Hieu Nguyen, Kazem Meidani, Amir Barati Farimani, Khoa D. Doan, and Chandan K. Reddy. LLM-SRBench: A new benchmark for scientific equation discovery with large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 55325–55359. PMLR, 2025.

[SRY<sup>+</sup>26] Youran Sun, Xingyu Ren, Chugang Yi, Jiaxuan Guo, Kejia Zhang, Jianda Du, and Haizhao Yang. Agon: An autonomous large-scale omnidisciplinary research system built on prompt economy. arXiv preprint arXiv:2606.24177, 2026.

[SRZ<sup>+</sup>26] Youran Sun, Xingyu Ren, Kejia Zhang, Xinpeng Liu, and Jiaxuan Guo. PerspectiveGap: A benchmark for multi-agent orchestration prompting. arXiv preprint arXiv:2606.08878, 2026.

[Ste79] Josef Stefan. Ueber die beziehung zwischen der w¨armestrahlung und der temperatur. Sitzungsberichte der Kaiserlichen Akademie der Wissenschaften in Wien, Mathematisch-Naturwissenschaftliche Classe, 79:391–428, 1879.

[TID23] Wassim Tenachi, Rodrigo Ibata, and Foivos I. Diakogiannis. Deep symbolic regression for physics guided by units constraints: Toward the automated discovery of physical laws. The Astrophysical Journal, 959(2):99, 2023.

[TK25] Mattias E. Thing and Sofie M. Koksbang. cp3-bench: A tool for benchmarking symbolic regression algorithms demonstrated with cosmology. Journal of Cosmology and Astroparticle Physics, 2025(01):040, 2025.

[UT20] Silviu-Marian Udrescu and Max Tegmark. AI Feynman: A physics-inspired method for symbolic regression. Science Advances, 6(16):eaay2631, 2020.

[Wie93] Willy Wien. Eine neue beziehung der strahlung schwarzer k¨orper zum zweiten hauptsatz der w¨armetheorie. Sitzungsberichte der K¨oniglich Preussischen Akademie der Wissenschaften zu Berlin, pages 55–62, 1893.

[Wie96] Willy Wien. Ueber die energievertheilung im emissionsspectrum eines schwarzen k¨orpers. Annalen der Physik, 294(8):662–669, 1896.

[XAQQ25] Ziyu Xiang, Kenna Ashen, Xiaofeng Qian, and Xiaoning Qian. Graph-based Symbolic Regression with Invariance and Constraint Encoding. In Advances in Neural Information Processing Systems, volume 38, pages 6882–6914. Curran Associates, Inc., 2025.

[XLL<sup>+</sup>26] Zikai Xie, Wenmei Li, Man Luo, Jun Jiang, and Linjiang Chen. Language models guide symbolic equation discovery by controlling search, 2026.

[XLS24] Yilong Xu, Yang Liu, and Hao Sun. Reinforcement symbolic regression machine. In International Conference on Learning Representations, 2024.

[XSL26] Shijie Xia, Yuhan Sun, and Pengfei Liu. SR-Scientist: Scientific Equation Discovery With Agentic AI. In The Fourteenth International Conference on Learning Representations, 2026.

[YDLJ25] Zihan Yu, Jingtao Ding, Yong Li, and Depeng Jin. Symbolic regression via MDLformer-guided search: From minimizing prediction error to minimizing description length. In International Conference on Learning Representations, 2025.

[YLL<sup>+</sup>25] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jef Clune, and David Ha. The AI scientist-v2: Workshop-level automated scientific discovery via agentic tree search, 2025.

[ZCX<sup>+</sup>25] Hengzhe Zhang, Qi Chen, Bing Xue, Wolfgang Banzhaf, and Mengjie Zhang. RAG-SR: Retrievalaugmented generation for neural symbolic regression. In International Conference on Learning Representations, 2025.

[ZLX<sup>+</sup>26] Wenxiao Zhao, Dong Liu, Kaiyi Xu, Feng Liu, Zhen Zhao, Fei Ben, Shu Wang, Wenhao Li, Ying Nian Wu, Fenghua Ling, Haobo Li, and Lei Bai. A-SR: Self-evolving agentic LLMs for symbolic regression via hierarchical coordination, 2026.
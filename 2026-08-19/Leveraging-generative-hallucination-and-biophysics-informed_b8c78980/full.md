# Leveraging generative hallucination and biophysics-informed modeling for unified biomolecular sequence–structure co-design

Xuefeng Liu<sup>1,2,†∗</sup>, Mingxuan Cao<sup>3∗</sup>, Xiao Luo<sup>4∗</sup>, Songhao Jiang<sup>2</sup> Tobin Sosnick<sup>5</sup>, Jinbo Xu<sup>6</sup>, Louis. J. Maher<sup>7</sup>, Rick L. Stevens<sup>2,8</sup> <sup>1</sup>Department of Medicine, University of Florida <sup>2</sup>Department of Computer Science, University of Chicago <sup>3</sup>Data Science Institute, University of Chicago <sup>4</sup>Department of Pathology, University of Chicago <sup>5</sup>Department of Biochemistry and Molecular Biology, University of Chicago <sup>6</sup>Toyota Technological Institute at Chicago <sup>7</sup>Department of Biochemistry and Molecular Biology, Mayo Clinic <sup>8</sup>Argonne National Laboratory

Corresponding author: Xuefeng.Liu@medicine.ufl.edu

## Abstract

Biomolecular design underpins applications from molecular recognition to therapeutics and synthetic biology, yet de novo interaction design remains challenging— especially for DNA/RNA, underexplored non-protein modalities with scarce, heterogeneous complex data and sharper geometric and chemical constraints. We introduce MCTH (Monte Carlo Tree Hallucination), an inference-only framework that casts all-atom sequence–structure co-design as uncertainty-aware planning over hallucinated states from pretrained folding and inverse-folding models, with optional biophysical control within the same decision loop. MCTH treats these models as frozen black-box operators and uses Monte Carlo Tree Search to allocate a fixed inference budget across competing design trajectories, incorporating model confidence and uncertainty, as well as cross-expert consensus/disagreement when multiple predictors are available. Across protein–RNA, protein–DNA, protein– protein, and protein–ligand design, matched-budget experiments show that adaptive search improves over simpler sampling and cycling strategies, while held-out AlphaFold3 and Chai-1 evaluations demonstrate transfer beyond the search-time oracle. MCTH provides a shared planning layer across modalities while allowing task-specific folding, inverse-folding, and biophysical modules, requiring no fine-tuning or backpropagation through component models.

## 1 Introduction

Biomolecular design underpins modern molecular biology, synthetic biology, and biomedicine, enabling applications from catalysis and therapeutics to biosensing and programmable systems. A core goal is to design biomolecular interactions—protein–protein, protein–small-molecule, protein–nucleic-acid, and nucleic-acid–nucleic-acid complexes—with high structural fidelity and functional precision. Despite decades of progress, de novo design remains challenging, especially beyond proteins to DNA/RNA, small molecules, ions, and post-translational modifications.

A core difficulty is the intrinsically coupled nature of sequence and structure: in a rugged, highdimensional landscape, small sequence changes can induce large structural or functional shifts. Capturing interaction specificity while jointly optimizing sequence and structure is computationally demanding and often difficult to control. Physics-based modeling (e.g., Rosetta-style energy functions) provides explicit chemical priors but can be costly and brittle under local optimization [2, 22]. Learning-based methods offer expressive data-driven priors, yet often struggle with generalization, exploration, and optimization stability across heterogeneous modalities. These issues are amplified for DNA/RNA systems, where complex data are scarce and success hinges on fine-grained interaction fidelity (e.g., base stacking, ion/ligand coordination, atomic contacts) rather than global fold alone.

Recent advances in deep learning–based structure prediction, culminating in AlphaFold3 [1]-style all-atom diffusion models, have expanded biomolecular modeling beyond proteins to small molecules, nucleic acids, ions, and covalent modifications. Earlier predictors such as AlphaFold2 and AlphaFold Multimer established strong protein(-complex) priors and intrinsic confidence measures [13, 19]. However, structure prediction alone does not solve design. Most design approaches either (i) use pretrained predictors as oracles for validating or filtering large candidate pools [15, 16, 27–29], or (ii) turn predictors into optimization engines by fine-tuning [30, 39] or hallucination/inversion objectives [5, 8, 31]. Fine-tuned diffusion models require extensive training and are less flexible across tasks, while gradient-based hallucination approaches (e.g., bindcraft [31], boltzdesign [8]) are expensive, initialization-sensitive, and prone to local optima—especially for long sequences and complex assemblies. Backbone-generative diffusion models (e.g., RFdiffusion) offer geometric controllability but typically require separate sequence design and refolding-based verification [11, 39].

An emerging alternative paradigm exploits a property of diffusion-based structure prediction models often viewed as a limitation: hallucination. Given underspecified or out-of-distribution inputs—such as all-X sequences or partially defined complexes—these models can hallucinate plausible conformations guided by learned priors. Cycling between hallucinated structure prediction and sequence redesign can yield designable biomolecules without fine-tuning [5, 9, 14]. However, existing cyclingbased methods rely on fixed schedules and greedy local updates, limiting global exploration and exploration–exploitation balance. Moreover, high model confidence need not imply biophysical feasibility: unconstrained hallucinations can violate fine-grained constraints (e.g., steric clashes, strained geometries, incompatible ligand/ion coordination).

In this work, we propose MCTH, an inference-time planning framework that reframes biomolecular design as structured search over hallucinated sequence–structure states. Pretrained folding and inversefolding models act as frozen black-box operators that propose and evaluate candidate transitions, while Monte Carlo Tree Search allocates a fixed inference budget across competing design trajectories. Unlike fixed-depth cycling or gradient-based optimization through a structure predictor, MCTH requires only model outputs—sequences, structures, and confidence or feasibility scores—and does not access, update, or differentiate through component-model parameters.

The planner is uncertainty-aware: model confidence, expert consensus/disagreement when multiple predictors are available, and optional biophysical feasibility signals can jointly guide selection, expansion, and backup. The same planning logic is reused across modalities, while the folding, inverse-folding, and biophysical modules are dispatched according to the task. This yields a unified co-hallucination strategy that couples learned generative proposals with biophysical control, which is particularly relevant for nucleic-acid and ligand-conditioned settings where data are limited and geometric constraints are sharp. In the main experiments we use a single all-atom folding model with modality-specific inverse folding; we additionally validate the multi-folding-expert capability in protein-binder design using Boltz-2 and AlphaFold2 jointly during search.

## Contributions and novelty.

• Inference-time planning over sequence–structure states. We cast hallucination-based biomolecular design as a finite-budget planning problem and use MCTS to adaptively allocate expensive folding evaluations across competing sequence–structure trajectories, rather than following fixed cycling or independent best-of-N sampling.

• Composable black-box model coordination. MCTH treats pretrained folding, inversefolding, and scoring models as frozen operators and requires only their emitted sequences, structures, and scalar signals—without fine-tuning or backpropagation through component models. Multiple folding experts can be combined through consensus, disagreement, and uncertainty signals.

• Biophysics–ML co-hallucination: We integrate optional geometric, thermodynamic, and energy-like feasibility cues with learned model confidence within the same planning loop, enabling task-specific control during selection, proposal refinement, and evaluation rather than only post-hoc filtering.

• General planning layer across biomolecular design tasks. The MCTS state, selection, expansion, and backup mechanism is shared across protein–RNA, protein–DNA, protein– protein, and protein–ligand design, while inverse-folding backends and biophysical terms remain modality-specific. Matched-budget and held-out cross-predictor evaluations test both search efficiency and robustness beyond the search-time oracle.

## 2 Background and Problem Setup

Related method families and positioning. Physics-first frameworks such as Rosetta/MD provide explicit energetic priors but scale poorly in large multimodal design spaces [2, 22, 24, 35, 37]. Learning-based structure predictors (AF2/Multimer, all-atom AF3/RFAA) expanded biomolecular modeling beyond proteins [1, 13, 19, 20]; diffusion and inverse-folding tools such as RFdiffusion, ProteinMPNN, and RNA pipelines enabled generation/redesign [4, 11, 17, 18, 39]. Hallucination/cycling and planning methods, including BindCraft/BoltzDesign and ERP/MCTD-ME, improve optimization or exploration but are often local, fixed-schedule, or single-modality [5, 8, 9, 14, 25, 26, 31].

Relative to these lines, MCTH occupies a distinct point in the design stack: it is a planning layer that wraps pretrained folding and inverse-folding experts with plug-in biophysical cues and allocates expensive inference through MCTS. Its key distinction is co-hallucination: learned experts propose sequence–structure hypotheses, while biophysical cues enter the same decision loop for selection, expansion, and proposal reweighting, rather than acting only as post-hoc filters. A fuller discussion of related work is provided in Appendix A, with a compact positioning summary in Appendix A.1.

Sequence–structure design as planning. We represent a biomolecular system by sequence(s) S and all-atom coordinates $X ;$ for complexes, the state includes all interacting components, e.g., $s = ( S , X )$ under task conditioning C such as binding partners, motifs, or ligands. We view design as finite-horizon planning in an MDP $\mathcal { M } = \langle \mathcal { S } , \mathcal { A } , H , \bar { P } , \bar { R } \rangle \operatorname { \mathrm { \Omega } } [ 3 4 ]$ , where states $s \in S$ are biomolecular sequence–structure hypotheses and actions $a \in { \mathcal { A } }$ are stochastic hallucination operators such as inverse-fold proposals, folding calls, or biophysical refinement/filtering steps. Transitions are induced by pretrained experts, $s ^ { \prime } \sim \mathsf { \bar { P } } ( \cdot \mid s , a )$ , and rewards are computed from the resulting state, e.g., $\begin{array} { r } { \dot { R ( s , a ) } = V ( s ^ { \prime } ) } \end{array}$

Search and guided proposal view. MCTS iteratively expands a partial search tree through selection, expansion, evaluation, and backpropagation. In our setting, nodes are biomolecular states $s = ( S , X )$ and evaluations come directly from model-derived value signals $V ( s )$ , such as confidence, interface quality, and optional feasibility terms, rather than long stochastic rollouts. Diffusion-style guidance can be viewed as reweighting proposals by an auxiliary score, $p ^ { \mathrm { g u i d e } } ( \xi ^ { \prime } \mid \xi ) \propto p ( \xi ^ { \prime } \mid \xi ) \exp \bar { \{ \beta f ( \xi ^ { \prime } ) \} }$ in MCTH, this score is instantiated by the same task-aware value signal used by planning. Practically, we implement this without gradients by sampling proposal candidates from different experts, seeds, or noise schedules and selecting/resampling according to their value.

Design objectives. We encode task objectives such as foldability, binding confidence, constraint satisfaction, and lead optimization through the state value $V ( s )$ . Thus, MCTH searches over hallucinated sequence–structure states using learned confidence and optional biophysical feasibility signals, rather than performing explicit end-to-end energy minimization. Additional preliminaries are provided in Appendix E.

## 3 Methodology

We introduce Monte Carlo Tree Hallucination (MCTH), a unified framework for biomolecular sequence–structure co-design that reframes hallucination by pretrained structure prediction models as a principled search and planning problem. Rather than relying on fixed iterative cycling or gradientbased optimization, MCTH formulates biomolecular design as structured exploration over a joint sequence–structure landscape, guided by Monte Carlo Tree Search (MCTS) and driven by forward inference of pretrained folding and inverse-folding models together with biophysical guidance.

![](images/88f5c3c3333e254fd23e9a7cacfe40821ca8cb48b7135271840ad5468a443e97.jpg)  
Figure 1: The MCTH framework: search-guided sequence–structure co-design. Left: UCT-style Monte Carlo Tree Search performs global planning over states s = (S, X) via selection and backpropagation. Right: each expansion instantiates a hallucination–evaluation cycle that proposes new sequences (inverse folding), generates corresponding structures (diffusion-based hallucination), and evaluates candidates using model confidence signals (e.g., ipTM/pLDDT), which are then backed up to guide subsequent search. Confidence-guided masking is an optional refinement operator and is disabled in the minimal plug-and-play configuration used in this work.

At a high level, MCTH integrates three components: (i) diffusion-based all-atom structure hallucination, (ii) inverse-folding-based sequence redesign, and (iii) global exploration and decisionmaking via MCTS. MCTH also incorporates plug-in biophysical feasibility/energy models (e.g., clash/packing/contact geometry penalties or energy-like terms) as complementary guidance signals, improving physics-aware controllability and physical fidelity without fine-tuning or backpropagation through structure predictors.

Figure 1 summarizes MCTH as an uncertainty-aware Monte Carlo Tree Search over sequence– structure states: selection is guided by UCT-style visitation statistics and multi-expert consensus/uncertainty signals, while each expansion instantiates a hallucination–evaluation cycle using inverse folding and diffusion-based structure prediction, optionally interleaving biophysical scoring and refinement/filtering steps that enforce feasibility beyond model confidence.

## 3.1 Composable Folding and Inverse-Folding Operators

MCTH treats pretrained folding and inverse-folding models as frozen black-box operators that induce transitions in the search tree under task conditioning C (e.g., binding partners, motifs, or ligands). The planner interacts with these models only through forward-inference outputs—predicted structures, proposed sequences, and associated confidence or feasibility signals—without accessing model parameters or backpropagating through the component models. This interface allows the same search procedure to compose different folding and inverse-folding backends while keeping the planner itself unchanged.

Structure hallucination via diffusion models. Given a sequence hypothesis S and optional conditioning information C, a folding or structure-prediction model generates a candidate all-atom conformation

$$
X \sim p _ { \theta } ( X \mid S , C ) .\tag{1}
$$

When the sequence or complex specification is underspecified—for example, an all-X sequence or a partially defined interface—the predictor may generate plausible conformations from its learned structural prior. MCTH uses these predictions as candidate sequence–structure states for subsequent evaluation and search.

The framework supports either a single folding expert or multiple folding experts. In the main modality-spanning experiments, we use Boltz-2 as the shared all-atom folding expert together with modality-specific inverse-folding models. To directly validate the multi-expert extension, we additionally instantiate protein-binder search with both Boltz-2 and AlphaFold2. When multiple folding experts are instantiated, as in our dual-expert protein-binder experiment, we obtain an ensemble $X ^ { \prime ( e ) } { } _ { e = 1 } ^ { E }$ for the same candidate sequence. Agreement among these predictions provides a consensus signal, while disagreement provides a structural-uncertainty signal. With a single folding expert, the same planner reduces naturally to confidence-based search without the cross-expert terms.

Sequence redesign via inverse folding. Given a hallucinated structure X and conditioning information $C \ ( \mathrm { e . g } $ ., binding partners, fixed motifs, ligand identity/pose constraints, or other task-specific constraints), inverse-folding experts propose compatible sequences

$$
S \sim q _ { \phi } ( S \mid X , C ) ,\tag{2}
$$

improving foldability and interaction consistency while preserving the specified constraints. Unlike fixed iterative pipelines, inverse folding in MCTH is applied adaptively, allowing the planner to refine promising structural hypotheses or repair inconsistencies introduced by hallucination.

## 3.2 Uncertainty-Aware MCTS over Sequence–Structure States

MCTH casts biomolecular co-design as finite-budget tree search over sequence–structure states. Each node represents a concrete design hypothesis $s = ( S , X )$ , where S denotes the current sequence(s) of the designable component(s) and X denotes the corresponding all-atom complex structure under task conditioning $C \left( \mathrm { e . g . } \right.$ ., fixed receptor, motif, ligand, or binding partner). This differs from sequenceonly search: a branch is evaluated not only by the proposed sequence, but by the folded complex that the sequence induces. Thus, each edge corresponds to a stochastic hallucination transition that composes sequence redesign and structure prediction.

Actions and transitions. From a state $s = ( S , X )$ , an action $a \in \mathcal { A } ( s )$ proposes a successor in two steps. First, an inverse-folding expert proposes an updated design sequence $S ^ { \prime }$ for the designable component, optionally with lightweight mutations or constraints. Second, a folding/hallucination expert predicts the corresponding complex structure $X ^ { \prime }$ under the same task conditioning C, yielding a child state $s ^ { \prime } = ( S ^ { \prime } , X ^ { \prime } )$ . When multiple folding experts, seeds, or noise schedules are used, we obtain an ensemble $\{ X ^ { \prime ( e ) } \} _ { e = 1 } ^ { E } ;$ ; agreement among these predictions provides a consensus signal, while disagreement provides an uncertainty signal.

State value. Each expanded child is assigned a task-aware value

$$
V ( s ^ { \prime } ) = f ! \left( m _ { 1 } ( s ^ { \prime } ) , \dots , m _ { J } ( s ^ { \prime } ) \right) ,\tag{3}
$$

where the $m _ { j } \mathrm { ^ { * } }$ are predictor-derived signals such as pLDDT, pTM, ipTM, interface quality, and optional geometric or biophysical feasibility terms. The aggregation function $f$ maps these signals to a scalar value used for evaluation and backup. Unless otherwise stated, we use a fixed task-level aggregation across targets rather than target-specific tuning. All signals are obtained through forward inference; MCTH does not require gradients or backpropagation through the component models.

Uncertainty-aware selection. At each MCTS iteration, MCTH traverses the current tree from the root to a leaf. We use a consensus–uncertainty–physics PUCT score, CU-PUCT:

$$
\mathrm { C U - P U C T } ( s , a ) = Q ( s , a ) + c _ { p } \frac { \sqrt { \log N ( s ) } } { 1 + N ( s , a ) } \cdot \pi _ { \mathrm { c o n s } , \tau } ( a \mid s )\tag{4}
$$

We select

$$
a ^ { \star } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } ( s ) } \mathrm { C U \mathrm { - } P U C T } ( s , a ) ,\tag{5}
$$

where $Q ( s , a )$ is the empirical action value, $N ( s )$ and $N ( s , a )$ are visit counts, and $c _ { p }$ controls exploration.

When multiple folding experts are available, we define a consensus prior from their agreement on the same candidate transition. For a candidate sequence $S ^ { \prime }$ and predicted structures $\{ X ^ { \bar { \prime } ( e ) } \} _ { e = 1 } ^ { E }$ ，

$$
\tilde { \pi } _ { \mathrm { c o n s } } ( s , a ) = \exp \left( \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \log q _ { \phi } ( S ^ { \prime } \mid X ^ { \prime ( e ) } , C ) \right) \exp \left( - \kappa { \mathcal D } \Big ( \{ X ^ { \prime ( e ) } \} _ { e = 1 } ^ { E } \Big ) \right) ,\tag{6}
$$

and $\pi _ { \mathrm { c o n s } , \tau }$ is obtained by temperature-normalizing $\tilde { \pi } _ { \mathrm { c o n s } }$ over candidate actions. Thus, actions are preferred when the proposed sequence is compatible across predicted structures and the folding experts agree geometrically.

The uncertainty terms separate sequence-design ambiguity from structural uncertainty:

$$
U _ { \mathrm { i n v } } ( s , a ) = \frac { 1 } { | M | } \sum _ { i \in M } { \sf H } ( q _ { \phi } ( \cdot  { | } X , C ) _ { i } ) ,\tag{7}
$$

$$
U _ { \mathrm { f o l d } } ( s , a ) = g ( \mathrm { p L D D T , p T M , i p T M , \ldots } ) \quad \mathrm { o r } \quad \mathcal { D } \left( \{ X ^ { \prime ( e ) } \} _ { e = 1 } ^ { E } \right) .\tag{8}
$$

Here, $U _ { \mathrm { i n v } }$ captures ambiguity in sequence redesign, while $U _ { \mathrm { f o l d } }$ captures low folding confidence or, when multiple folding experts are used, cross-expert structural disagreement in the predicted complex. The optional biophysical term $E _ { \mathrm { p h y s } }$ down-weights high-confidence but physically implausible proposals. Full normalization details and backup rules are given in Appendix B.1.

Expansion, evaluation, and backup. Once selection reaches a leaf, MCTH expands up to $K$ candidate children. Each child is constructed by inverse folding followed by folding/refolding, yielding a new sequence–structure state and the signals used to evaluate $V ( s ^ { \prime } )$ . The resulting value is backed up along the selected path to update visit counts and action-value statistics. Repeated iterations therefore allocate the fixed inference budget toward promising and informative branches and, when applicable, toward transitions supported across experts and consistent with biophysical constraints.

Remark 1: DNA aptamer design (thermodynamic window). For DNA aptamers, we instantiate $E _ { \mathrm { p h y s } }$ using a soft window penalty on nearest-neighbor self-folding free energy $\Delta G _ { \mathrm { s e l f } } ( S ^ { \prime } )$ , computed at $\mathrm { \dot { 3 7 } ^ { \circ } C }$

$$
E _ { \mathrm { p h y s } } ( S ^ { \prime } ) = 1 - \sigma \big ( k _ { l } ( \Delta G _ { \mathrm { s e l f } } ( S ^ { \prime } ) - \Delta G _ { l } ) \big ) \sigma \big ( - k _ { u } ( \Delta G _ { \mathrm { s e l f } } ( S ^ { \prime } ) - \Delta G _ { u } ) \big ) ,\tag{9}
$$

where $\sigma ( x ) = 1 / ( 1 + e ^ { - x } )$ is the logistic sigmoid. We calibrate $\left( \Delta G _ { l } , \Delta G _ { u } , k _ { l } , k _ { u } \right)$ from an empirical distribution over random sequences; details and derivations are given in Appendix B.3.

## 3.3 Biophysical Refinement

Biophysical signals in MCTH can influence search not only through the state value $E _ { \mathrm { p h y s } }$ in Sec. $3 . 2 ,$ but also through optional refinement operators that modify candidate states before subsequent refolding. These refinements are task-specific and may act on sequence, structure, or both, while leaving the underlying MCTS procedure unchanged. For DNA aptamer design, we instantiate sequence-level refinement using nearest-neighbour self-folding thermodynamics:

Identifying the failure mode. For each candidate $( S ^ { \prime } , X ^ { \prime } )$ output by MCTS, we compute $\Delta G _ { \mathrm { s e l f } } ( \boldsymbol { S } ^ { \prime } )$ using seqfold and classify it into three regimes:

• Over-stable $( \Delta G _ { \mathrm { s e l f } } < \Delta G _ { l } ) ;$ : strong self-structure implies an unfolding penalty before binding.

• Favourable $( \Delta G _ { l } \leq \Delta G _ { \mathrm { s e l f } } \leq \Delta G _ { u } )$ : no refinement needed.

• Under-structured $( \Delta G _ { \mathrm { s e l f } } > \Delta G _ { u } )$ : weak pre-organisation implies a high entropic cost upon binding.

Only over-stable or under-structured candidates are refined. The boundaries $\left( \Delta G _ { l } , \Delta G _ { u } \right)$ are calibrated from an empirical distribution of random DNA sequences (Appendix B.3).

Interface-aware mutation strategy. To avoid disrupting the binding surface, mutations are restricted to non-interface positions. We define an interface set $I \subset \{ 1 , \ldots , | S ^ { \prime } | \}$ as positions whose predicted contacts with the target in $X ^ { \prime }$ exceed a threshold (e.g., heavy-atom distance $< 8 \mathring \mathrm { A } )$ , and restrict edits to $j \not \in I .$

Over-stable correction. We compute an MFE base-pair assignment $P ( S ^ { \prime } )$ from seqfold, select a noninterface position $j ^ { * }$ with high base-pairing propensity, and apply a point mutation $S _ { i ^ { * } } ^ { \prime }  \tilde { s }$ chosen to locally destabilize the stem (e.g., introduce a mismatch/bulge). We iterate until $\Delta G _ { \mathrm { s e l f } } \geq \Delta G _ { l }$ or a budget of $T _ { \mathrm { m a x } }$ mutations is reached.

Under-structured correction. We identify a contiguous non-interface window (default length $w = 6 )$ with low base-pairing propensity and propose a stabilizing substitution that forms a short hairpin $( \mathrm { e . g . }$ a GC-rich stem with a stable tetraloop motif [36]), accepting the edit if it achieves $\Delta G _ { \mathrm { s e l f } } \le \Delta G _ { u }$

Acceptance criterion. We accept a refined sequence $S ^ { \prime \prime }$ iff it reduces the biophysical penalty:

$$
E _ { \mathrm { p h y s } } ( S ^ { \prime \prime } ) < E _ { \mathrm { p h y s } } ( S ^ { \prime } ) .\tag{10}
$$

Otherwise, we retain the original candidate $( S ^ { \prime } , X ^ { \prime } )$

## 3.4 Inference-time Structural Guidance

Beyond value shaping and sequence refinement, biophysical information can optionally enter a compatible folding model directly during inference. This provides a third control interface in MCTH: rather than only scoring or editing a candidate, structural priors can steer the generation of its corresponding conformation. This mechanism is backend-dependent and is not required by the MCTS planner.

For a candidate aptamer sequence $S ^ { \prime } .$ , we run seqfold to obtain $\Delta G _ { \mathrm { s e l f } } ( S ^ { \prime } )$ and an MFE basepair set $P ( S ^ { \prime } )$ ; when $\Delta G _ { l } \ { ' } \le \ \Delta G _ { \mathrm { s e l f } } ( S ^ { \prime } ) \ \le \ \Delta \bar { G } _ { u }$ (boundaries in Appendix B.3), we enable Boltz-2 inference-time potentials (use potentials=True) and use $P ( \bar { S ^ { \prime } } )$ as a soft contact prior. Specifically, we first identify the interface set I from the current hallucinated complex $X ^ { \prime }$ (positions contacting the protein by a distance cutoff). We then retain only non-interface base pairs $\mathcal { P } _ { \mathrm { f r e e } } =$ $\{ ( i , j ) \in \bar { P } ( S ^ { \prime } ) : \ i \notin I , \ j \notin I \}$ and impose the quadratic contact potential

$$
U _ { \mathrm { c o n t a c t } } ( X ) = \sum _ { ( i , j ) \in { \mathcal P } _ { \mathrm { f r e e } } } w _ { i j } \big ( d _ { i j } ( X ) - d _ { 0 } \big ) ^ { 2 } ,\tag{11}
$$

where $d _ { i j } ( X )$ is the heavy-atom distance for $( i , j )$ in structure $X , d _ { 0 }$ is a target base-pair distance, and $w _ { i j }$ can decay with diffusion noise level. Boltz-2 incorporates $U _ { \mathrm { c o n t a c t } }$ via its potential-energy steering $( \mathrm { e } . \mathrm { g } . , \hat { x } _ { 0 }$ correction / particle resampling [12]), requiring no backpropagation through the diffusion model. This preserves secondary structure away from the binding site while leaving interface bases unconstrained for target binding. Full implementation details (interface identification, contact scheduling, and potential weights) are provided in Appendix B.5.

## 3.5 Multi-modality Representation and Expert Assignment

MCTH separates the planning procedure from the models that instantiate its transition and evaluation operators. The MCTS state representation, selection, expansion, and backup rules are shared across tasks, while folding, inverse-folding, and optional biophysical modules are assigned according to the molecular modality and available model interfaces.

For the modality-spanning main experiments, we use Boltz-2 as a shared all-atom folding expert, whose representation supports proteins, nucleic acids, small molecules, and their complexes [32]. Sequence redesign is dispatched by modality: ProteinMPNN for proteins and peptides [11], NA-MPNN for nucleic acids [21], and LigandMPNN for ligand-conditioned protein design. Task-specific biophysical terms and refinement operators are enabled only where applicable.

The planner can also compose multiple compatible folding experts. In this setting, candidate states may be evaluated by multiple predictors, allowing cross-expert consensus, disagreement, and uncertainty to enter the same CU-PUCT selection rule defined above. We evaluate this multi-expert instantiation separately in protein-binder design, while keeping the underlying MCTS procedure unchanged.

Implementation details, supported modalities, and dispatch rules are summarized in Appendix B.4.

## 4 Experiments

Evaluation protocol. We evaluate MCTH from three complementary perspectives. First, we measure optimization under the search-time oracle: candidates are evaluated using the same folding model scores that define the inference-time design objective. Because MCTH explicitly optimizes these scores, these experiments measure finite-budget optimization efficiency rather than oracleindependent physical validity.

Second, we use matched-budget controls to separate adaptive search from raw evaluator access. Whenever applicable, baselines are given comparable candidate or folding-call budgets and evaluated under the same refolding protocol. We further compare against greedy cycling and best-of-N variants using the same initialization, folding oracle, inverse-folding backend, and evaluation budget, isolating the effect of search-guided compute allocation.

Third, we perform held-out evaluation using predictors and physical metrics that are not used to guide the corresponding search. We use AlphaFold3 for independent protein–nucleic-acid re-scoring, Chai-1 for held-out evaluation of the dual-expert protein-binder experiment, and Rosetta InterfaceAnalyzer as an additional oracle-independent physical check. For protein binders, we also report matched final-design pass rates under a held-out binder-quality filter rather than relying only on the single best-scoring candidate.

This distinction is important because some coupling to the search-time oracle is inherent to inferencetime optimization: the planner must optimize a measurable objective. The relevant questions are therefore (i) whether adaptive search improves over equal-budget alternatives under the same objective, and (ii) whether the resulting designs remain competitive under evaluators outside the optimization loop.

## 4.1 Nucleic Acid Design

## 4.1.1 RNA aptamer design

Protein–RNA aptamer benchmark. We evaluate MCTH on a protein–RNA aptamer design benchmark consisting of 7 protein–RNA chain-pair targets derived from PDB complexes (7WKP, 7YEW, 7YGL, 8TG4, 8gxb, 9C7A), spanning RNA lengths from 4 to 61 nucleotides. Given a protein structure and the RNA length specification, the goal is to design an RNA aptamer sequence that yields a confident protein–RNA interface.

Baselines and matched-budget evaluation. We compare against ODesign and RNAFrameFlow + NA-MPNN. To control for the effect of repeated structure evaluation, both baselines are run with 100 candidate designs and re-scored using the same Boltz-2 refolding protocol. All methods therefore receive a comparable opportunity to identify a high-scoring candidate under the search-time oracle.

Results under the search-time oracle. At the matched budget, MCTH achieves a mean best ipTM of 0.946, compared with 0.894 for ODesign and 0.923 for RNAFrameFlow + NA-MPNN. MCTH obtains the highest ipTM on 6 of 7 targets; the only exception is 8GXB, where RNAFrameFlow is higher by 0.001 (0.880 vs. 0.879). Thus, increasing the baseline candidate pool substantially narrows the original gap but does not eliminate the advantage of adaptive search.

Held-out AlphaFold3 evaluation. We additionally re-score the final designs using AlphaFold3, which is never used during search. Under this held-out predictor, MCTH retains the highest mean ipTM (0.884), compared with 0.824 for ODesign and 0.794 for RNAFrameFlow + NA-MPNN, and is top-ranked on 6 of 7 targets (including one tie). The persistence of the ranking under an independent predictor suggests that the gain is not confined to Boltz-2-specific preferences, although all results remain computational proxies rather than experimental validation.

## 4.1.2 DNA aptamer design

Protein–DNA aptamer benchmark. We evaluate MCTH on 10 protein–ssDNA chain-pair targets derived from four PDB complexes (7XVN, 7YSF, 7YUK, and 8PMF). Given a fixed protein target and a DNA length specification, the goal is to design an ssDNA aptamer sequence that forms a confident protein–DNA interface.

<table><tr><td>Target</td><td>RNA Len</td><td>ODesign@100</td><td>RNAFrameFlow@100</td><td>MCTH</td></tr><tr><td>7WKP_A→B</td><td>22</td><td>0.883</td><td>0.932</td><td>0.941</td></tr><tr><td>7YEW_A→B</td><td>5</td><td>0.928</td><td>0.935</td><td>0.960</td></tr><tr><td>7YGL_A→B</td><td>4</td><td>0.918</td><td>0.924</td><td>0.963</td></tr><tr><td>7YGL_A→C</td><td>4</td><td>0.928</td><td>0.927</td><td>0.953</td></tr><tr><td>8TG4_A→C</td><td>5</td><td>0.924</td><td>0.925</td><td>0.963</td></tr><tr><td>8GXB.C→B</td><td>61</td><td>0.835</td><td>0.880</td><td>0.879</td></tr><tr><td>9C7A_A→B</td><td>8</td><td>0.841</td><td>0.940</td><td>0.960</td></tr><tr><td>Mean</td><td>一</td><td>0.894</td><td>0.923</td><td>0.946</td></tr></table>

(a) Matched-budget Boltz-2 evaluation.

<table><tr><td>Target</td><td>ODesign</td><td>RNAFrameFlow</td><td>MCTH</td></tr><tr><td>7WKP_A→B</td><td>0.800</td><td>0.730</td><td>0.940</td></tr><tr><td>7YEW_A→B</td><td>0.880</td><td>0.890</td><td>0.900</td></tr><tr><td>7YGL_A→B</td><td>0.780</td><td>0.780</td><td>0.880</td></tr><tr><td>7YGL_A→C</td><td>0.780</td><td>0.800</td><td>0.870</td></tr><tr><td>8TG4_A→C</td><td>0.930</td><td>0.870</td><td>0.930</td></tr><tr><td>8GXB_C→B</td><td>0.770</td><td>0.680</td><td>0.850</td></tr><tr><td>9C7A_A→B</td><td>0.830</td><td>0.810</td><td>0.820</td></tr><tr><td>Mean</td><td>0.824</td><td>0.794</td><td>0.884</td></tr></table>

(b) Held-out AlphaFold3 evaluation.  
Table 1: Protein–RNA aptamer design. Left: matched 100-candidate evaluation under the Boltz-2 search-time oracle. Right: independent AlphaFold3 re-scoring of final designs. Values are ipTM; higher is better.

Baselines and matched-budget evaluation. We compare against ODesign as the native nucleicacid design baseline. To control for repeated evaluator access, ODesign is run with 100 candidate designs and re-scored using the same Boltz-2 refolding protocol. Two 7XVN chain-B cases collapse to ipTM ≈ 0 for both ODesign and MCTH under Boltz-2; we therefore exclude these two oraclefailed cases from the matched-budget comparison and report the remaining eight chain-A targets. BoltzDesign is not included in this matched-budget table because its released design procedure optimizes the protein chain in protein–DNA complexes rather than directly generating the DNA strand.

Results under the search-time oracle. On the eight non-degenerate Boltz-2 targets, MCTH achieves a mean best ipTM of 0.925 versus 0.908 for ODesign@100 and obtains the highest score on 7 of 8 targets. The gap is relatively small on several already high-scoring targets, indicating that strong best-of-N sampling can approach search when the oracle landscape saturates, while adaptive allocation still improves the overall fixed-budget result.

Held-out AlphaFold3 evaluation. We additionally re-score designs from all 10 targets using AlphaFold3, which is never used during search. Under AF3, the two 7XVN chain-B cases are no longer degenerate, indicating that their collapse under Boltz-2 is evaluator-specific rather than evidence that the design targets themselves are intrinsically invalid. Across all 10 targets, MCTH achieves the highest mean ipTM (0.721), compared with 0.665 for BoltzDesign and 0.644 for ODesign, and obtains the top score on 6 of 10 targets. Together with the matched-budget results, this shows that the advantage under the search-time oracle is not restricted to Boltz-2-specific scoring preferences, while also highlighting meaningful disagreement between structure predictors at the interface level.

## 4.2 Protein Binder Design

We evaluate MCTH on de novo protein–protein binder design, aiming to generate a binder that forms a confident complex with a fixed receptor.

Benchmark and setup. We evaluate MCTH on de novo protein–protein binder design using the 12-target CaoData benchmark. For each target, the receptor is fixed with its native sequence and structure, while the binder is designed de novo at the reference length. Targets span diverse receptor and binder sizes, including the multi-chain antibody-style case 5U8R.

For the main matched-budget experiment, we instantiate MCTH with Boltz-2 as the folding/refolding expert and ProteinMPNN as the inverse-folding proposal model. Starting from a random binder sequence at the target length, MCTH searches over sequence–structure hypotheses by alternating structure prediction and structure-conditioned sequence redesign. Unless otherwise stated, the search budget is 100 Boltz-2 evaluations per target.

Matched-budget optimization. We first test whether the advantage of MCTH can be explained by greater access to the folding evaluator. We therefore compare against BoltzDesign1, ODesign, and Complexa under a matched budget of up to 100 folding/evaluator calls per target. ODesign and Complexa candidates are compared under the common Boltz-2 refolding protocol, while BoltzDesign1 uses its native Boltz evaluation during optimization.

<table><tr><td>Target</td><td>ODesign@100</td><td>MCTH</td></tr><tr><td>7XVN_A→N</td><td>0.936</td><td>0.952</td></tr><tr><td>7XVN_A→P</td><td>0.957</td><td>0.950</td></tr><tr><td>7YSF_A→B</td><td>0.890</td><td>0.898</td></tr><tr><td>7YSF_A→C</td><td>0.869</td><td>0.876</td></tr><tr><td>7YUK_A→C</td><td>0.904</td><td>0.906</td></tr><tr><td>7YUK_A→D</td><td>0.901</td><td>0.912</td></tr><tr><td>8PMF_A→B</td><td>0.914</td><td>0.956</td></tr><tr><td>8PMF_A→D</td><td>0.895</td><td>0.947</td></tr><tr><td>Mean</td><td>0.908</td><td>0.925</td></tr></table>

(a) Matched-budget Boltz-2 evaluation on the 8 nondegenerate chain-A targets.

<table><tr><td>Target</td><td>BoltzDesign</td><td>ODesign</td><td>MCTH</td></tr><tr><td>7XVN_A→N</td><td>0.770</td><td>0.780</td><td>0.570</td></tr><tr><td>7XVN_A→P</td><td>0.740</td><td>0.570</td><td>0.850</td></tr><tr><td>7XVN_B→N</td><td>0.810</td><td>0.560</td><td>0.680</td></tr><tr><td>7XVN_B→P</td><td>0.440</td><td>0.740</td><td>0.670</td></tr><tr><td>7YSF_A→B</td><td>0.700</td><td>0.630</td><td>0.750</td></tr><tr><td>7YSF_A→C</td><td>0.410</td><td>0.470</td><td>0.680</td></tr><tr><td>7YUK_A→C</td><td>0.810</td><td>0.780</td><td>0.630</td></tr><tr><td>7YUK_A→D</td><td>0.630</td><td>0.780</td><td>0.820</td></tr><tr><td>8PMF_A→B</td><td>0.700</td><td>0.570</td><td>0.800</td></tr><tr><td>8PMF_A→D</td><td>0.640</td><td>0.560</td><td>0.760</td></tr><tr><td>Mean</td><td>0.665</td><td>0.644</td><td>0.721</td></tr></table>

(b) Held-out AlphaFold3 evaluation on all 10 targets.

Table 2: Protein–DNA aptamer design. Left: matched 100-candidate evaluation under the Boltz-2 search-time oracle; the two 7XVN chain-B cases are excluded because Boltz-2 collapses for all compared designs. Right: independent AlphaFold3 re-scoring on all 10 targets, under which the two chain-B cases are non-degenerate. Values are ipTM; higher is better.
<table><tr><td>Target</td><td>BoltzDesign1 @100</td><td>ODesign@100</td><td>Complexa@100</td><td>MCTH @100</td></tr><tr><td>1DJS</td><td>0.758</td><td>0.944</td><td>0.907</td><td>0.933</td></tr><tr><td>1MOX</td><td>0.291</td><td>0.901</td><td>0.845</td><td>0.929</td></tr><tr><td>1XIW</td><td>0.884</td><td>0.940</td><td>0.946</td><td>0.946</td></tr><tr><td>2GY7</td><td>0.189</td><td>0.672</td><td>0.930</td><td>0.862</td></tr><tr><td>2IFG</td><td>0.527</td><td>0.730</td><td>0.356</td><td>0.806</td></tr><tr><td>3DI3</td><td>0.508</td><td>0.901</td><td>0.926</td><td>0.947</td></tr><tr><td>3FKD</td><td>0.588</td><td>0.556</td><td>0.958</td><td>0.686</td></tr><tr><td>3MJG</td><td>0.697</td><td>0.878</td><td>0.888</td><td>0.904</td></tr><tr><td>3ZTJ</td><td>0.568</td><td>0.872</td><td>0.941</td><td>0.914</td></tr><tr><td>403V</td><td>0.778</td><td>0.944</td><td>0.852</td><td>0.953</td></tr><tr><td>40GA</td><td>0.136</td><td>0.916</td><td>0.904</td><td>0.912</td></tr><tr><td>5U8R</td><td>0.539†</td><td>0.532</td><td>0.783</td><td>0.576</td></tr></table>

Table 3: Matched-budget protein-binder optimization on CaoData. All methods use up to 100 folding/evaluator calls per target. ODesign and Complexa candidates are evaluated under the common Boltz-2 refolding protocol, while BoltzDesign1 uses its native Boltz evaluation during optimization. <sup>†</sup>BoltzDesign1 mean is over 11 targets because 5U8R is unavailable. Values are best ipTM per target; higher is better.

Across CaoData, MCTH achieves the highest mean best ipTM (0.864), compared with 0.853 for Complexa, 0.816 for ODesign, and 0.539 for BoltzDesign1 (11 available targets). Thus, the gain cannot be attributed simply to receiving more folding-model evaluations.

Dual-expert cross-predictor evaluation. We next test whether incorporating multiple folding experts improves robustness beyond a single search-time oracle. We retain the same ProteinMPNN proposal mechanism and MCTS procedure, but evaluate candidate states using both Boltz-2 and AlphaFold2. Their structural-quality signals contribute to node evaluation, while cross-expert consensus, disagreement, and uncertainty enter CU-PUCT exploration. Chai-1 is held out entirely from search and is used only for final evaluation.

Across the 11 targets shared by all configurations, the original Boltz-only MCTH obtains mean ipTM values of 0.69/0.19/0.18 under Boltz-2/AF2/Chai-1, respectively. The dual-expert variant improves these to 0.74/0.61/0.57. In particular, held-out Chai-1 improves from 0.18 to 0.57, with the dual-expert variant outperforming the single-expert configuration on 9 of 11 targets and tying on one. Thus, incorporating a second structural expert substantially reduces predictor-specific optimization and improves transfer to a third predictor outside the search loop.

<table><tr><td rowspan="2">Target</td><td colspan="3">Single-expert MCTH</td><td colspan="3">Dual-expert MCTH</td><td colspan="3">RFDiffusion</td></tr><tr><td>B2</td><td>AF2</td><td>Chai</td><td>B2</td><td>AF2</td><td>Chai</td><td>B2</td><td>AF2</td><td>Chai</td></tr><tr><td>1DJS</td><td>0.89</td><td>0.19</td><td>0.16</td><td>0.86</td><td>0.79</td><td>0.73</td><td>0.39</td><td>0.28</td><td>0.59</td></tr><tr><td>1MOX</td><td>0.70</td><td>0.14</td><td>0.15</td><td>0.23</td><td>0.48</td><td>0.15</td><td>0.27</td><td>0.24</td><td>0.50</td></tr><tr><td>1XIW</td><td>0.60</td><td>0.39</td><td>0.34</td><td>0.95</td><td>0.86</td><td>0.84</td><td>0.77</td><td>0.84</td><td>0.81</td></tr><tr><td>2GY7</td><td>0.31</td><td>0.21</td><td>0.17</td><td>0.90</td><td>0.68</td><td>0.76</td><td>0.37</td><td>0.67</td><td>0.73</td></tr><tr><td>2IFG</td><td>0.72</td><td>0.18</td><td>0.10</td><td>0.76</td><td>0.15</td><td>0.59</td><td>0.30</td><td>0.78</td><td>0.70</td></tr><tr><td>3DI3</td><td>0.90</td><td>0.12</td><td>0.34</td><td>0.76</td><td>0.74</td><td>0.22</td><td>0.58</td><td>0.62</td><td>0.26</td></tr><tr><td>3FKD</td><td>0.52</td><td>0.23</td><td>0.14</td><td>0.43</td><td>0.17</td><td>0.19</td><td>0.32</td><td>0.88</td><td>0.31</td></tr><tr><td>3MJG</td><td>0.50</td><td>0.13</td><td>0.12</td><td>0.82</td><td>0.61</td><td>0.64</td><td>0.47</td><td>0.16</td><td>0.53</td></tr><tr><td>3ZTJ</td><td>0.70</td><td>0.19</td><td>0.13</td><td>0.67</td><td>0.80</td><td>0.74</td><td>0.35</td><td>0.56</td><td>0.66</td></tr><tr><td>403V</td><td>0.93</td><td>0.16</td><td>0.26</td><td>0.94</td><td>0.83</td><td>0.82</td><td>0.68</td><td>0.75</td><td>0.78</td></tr><tr><td>40GA</td><td>0.77</td><td>0.11</td><td>0.09</td><td>0.87</td><td>0.62</td><td>0.63</td><td>0.71</td><td>0.66</td><td>0.69</td></tr><tr><td>Mean</td><td>0.69</td><td>0.19</td><td>0.18</td><td>0.74</td><td>0.61</td><td>0.57</td><td>0.47</td><td>0.58</td><td>0.60</td></tr></table>

Table 4: Cross-predictor evaluation on the 11 shared CaoData protein-binder targets. Single-expert MCTH uses Boltz-2 during search, whereas dual-expert MCTH uses both Boltz-2 and AlphaFold2; Chai-1 is held out from search. For MCTH, each evaluator column reports the best ipTM over three retained designs; RFDiffusion reports the best over ten designs. Higher is better.

<table><tr><td>Method</td><td>Outputs/target</td><td>Passing/valid</td><td>Pass rate</td><td>Targets ≥1 pass</td></tr><tr><td>Dual-expert MCTH</td><td>10</td><td>41/110</td><td>37.3%</td><td>8/11</td></tr><tr><td>RFDiffusion</td><td>10</td><td>22/110</td><td>20.0%</td><td>8/11</td></tr></table>

Table 5: Matched final-design yield under the same held-out binder-quality filter.

Matched final-design pass rate. Best-per-target scores characterize whether a method can discover at least one strong candidate, but do not measure design yield. We therefore additionally compare MCTH and RFDiffusion using exactly 10 final designs per target on the same 11-target set and apply the same held-out binder-quality filter to both methods. MCTH candidates are selected using only the recorded in-loop joint reward; held-out metrics are not used retrospectively for ranking.

Dual-expert MCTH yields 41 passing designs out of 110 (37.3%), compared with 22/110 (20.0%) for RFDiffusion, corresponding to a 17.3 percentage-point or 1.86× higher pass rate. Both methods produce at least one passing design on 8 of 11 targets. Thus, while RFDiffusion can achieve a slightly higher best-of-output Chai score, MCTH produces a substantially larger fraction of qualifying designs under an equal output budget.

## 5 Conclusion

We presented MCTH, an inference-only framework that reframes biomolecular design as an uncertainty-aware search over hallucinated sequence–structure states. By coordinating frozen folding, inverse-folding, and optional biophysical operators with MCTS, MCTH allocates expensive structure evaluations using confidence, uncertainty, and, when multiple experts are available, consen sus/disagreement signals. Across protein–RNA, protein–DNA, protein–protein, and protein–ligand design, matched-budget experiments demonstrate effective search-guided compute allocation, while held-out AlphaFold3 and Chai-1 evaluation shows that improvements can transfer beyond the searchtime oracle. The planning layer is shared across tasks, while expert backends and biophysical terms remain modality-specific. Our evaluation remains entirely computational and does not replace experi mental validation. Future work should prioritize wet-lab validation, stronger physics-based objectives, and broader multi-expert evaluation across modalities. Overall, MCTH suggests that closed-loop, biophysics-informed planning is a promising route for controllable biomolecular co-design across heterogeneous modalities.

## References

[1] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature, 630(8016):493–500, 2024. 2, 3, 15

[2] Rebecca F Alford, Andrew Leaver-Fay, Jeliazko R Jeliazkov, Matthew J O’Meara, Frank P DiMaio, Hahnbeom Park, Maxim V Shapovalov, P Douglas Renfrew, Vikram K Mulligan, Kalli Kappel, Jason W Labonte, Michael S Pacella, Richard Bonneau, Philip Bradley, Roland L Dunbrack Jr, Rhiju Das, David Baker, Brian Kuhlman, Tanja Kortemme, and Jeffrey J Gray. The Rosetta all-atom energy function for macromolecular modeling and design. Journal of Chemical Theory and Computation, 13(6):3031–3048, 2017. doi: 10.1021/acs.jctc.7b00125. 2, 3, 15

[3] Obtin Alkhamis, Caleb Byrd, Juan Canoura, Adara Bacon, Ransom Hill, and Yi Xiao. Exploring the relationship between aptamer binding thermodynamics, affinity, and specificity. Nucleic Acids Research, 53(6):gkaf219, 2025. 18

[4] Namrata Anand, Bowen Huang, Raphael Eguchi, and Po-Ssu Huang. Generative modeling of all-atom RNA structures as multimodal flow matching. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?i d=Vl3r5Y7fB8. 3, 15

[5] Ivan Anishchenko, Samuel J Pellock, Tamuka M Chidyausiku, Theresa A Ramelot, Sergey Ovchinnikov, Jingzhou Hao, Khushboo Bafna, Christoffer Norn, Alex Kang, Asim K Bera, Frank DiMaio, Lauren Carter, Cameron M Chow, Gaetano T Montelione, and David Baker. De novo protein design by deep network hallucination. Nature, 600(7889):547–552, 2021. doi: 10.1038/s41586-021-04184-w. 2, 3, 15

[6] Minkyung Baek, Frank DiMaio, Ivan Anishchenko, Justas Dauparas, Sergey Ovchinnikov, Gyu Rie Lee, Jue Wang, Qian Cong, Lisa N Kinch, R Dustin Schaeffer, et al. Accurate prediction of protein structures and interactions using a three-track neural network. Science, 373(6557):871–876, 2021. 15

[7] Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 27381–27394, 2021. 16

[8] Yehlin Cho, Martin Pacesa, Zhidian Zhang, Bruno E Correia, and Sergey Ovchinnikov. Boltzdesign1: Inverting all-atom structure prediction model for generalized biomolecular binder design. bioRxiv, pages 2025–04, 2025. 2, 3, 15

[9] Yehlin Cho, Griffin Rangel, Gaurav Bhardwaj, and Sergey Ovchinnikov. Protein hunter: exploiting structure hallucination within diffusion for protein design. bioRxiv, pages 2025–10, 2025. 2, 3, 15

[10] Gabriele Corso, Hannes Stark, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. DiffDock:¨ Diffusion-level protein-ligand docking. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=k9H9Y\_9ioea. 15

[11] Justin Dauparas, Ivan Anishchenko, Nathan Bennett, Huafeng Bai, R. J. Ragotte, L. F. Milles, B. I. M. Wicky, Alex Courbet, R. J. de Haas, N. Bethel, P. J. Y. Leung, T. F. Huddy, Sam Pellock, D. Tischer, F. Chan, B. Koepnick, H. Nguyen, A. Kang, B. Sankaran, A. K. Bera, N. P. King, and David Baker. Robust deep learning–based protein sequence design using proteinmpnn. Science, 378(6615):49–56, 2022. doi: 10.1126/science.add2187. 2, 3, 7, 15, 20

[12] Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential monte carlo samplers. Journal of the Royal Statistical Society Series B: Statistical Methodology, 68(3):411–436, 2006. 7, 20

[13] Richard Evans, Michael O’Neill, Alexander Pritzel, Natalie Antropova, Andrew Senior, Timothy Green, Augustin Zidek, Russ Bates, Sam Blackwell, Jason Yim, Olaf Ronneberger, Sebastian <sup>ˇ</sup> Bodenstein, Michal Zielinski, Alex Bridgland, Anna Potapenko, Andrew Cowie, Kathryn Tunyasuvunakool, Rishub Jain, Ellen Kohli, John Jumper, and Demis Hassabis. Protein complex prediction with AlphaFold-Multimer. bioRxiv, 2021. doi: 10.1101/2021.10.04.463034. 2, 3, 15

[14] Minchao Fang, Chentong Wang, Jungang Shi, Fangbai Lian, Qihan Jin, Zhe Wang, Yanzhe Zhang, Zhanyuan Cui, YanJun Wang, Yitao Ke, et al. Halludesign: Protein optimization and de novo design via iterative structure hallucination and sequence design. bioRxiv, pages 2025–11, 2025. 2, 3, 15

[15] Noelia Ferruz, Steffen Schmidt, and Birte Hocker. ProtGPT2 is a deep unsupervised language¨ model for protein design. Nature Communications, 13(1):4348, 2022. doi: 10.1038/s41467-0 22-32007-7. 2

[16] Michael A Jendrusch, Alessio L J Yang, Elisabetta Cacace, Jacob Bobonis, Carlos G P Voogdt, Sarah Kaspar, Kristian Schweimer, Cecilia Perez-Borrajero, Karine Lapouge, Jacob Scheurich, Kim Remans, Janosch Hennig, Athanasios Typas, Jan O Korbel, and S Kashif Sadiq. AlphaDesign: a de novo protein design framework based on AlphaFold. Molecular Systems Biology, 21 (9):1166–1189, 2025. doi: 10.1038/s44320-025-00119-z. 2

[17] Chaitanya K Joshi, Arian R Jamasb, Ramon Vinas, Charles Harris, Simon V Mathis, Alex˜ Morehead, Rishabh Anand, and Pietro Lio. Generative inverse design of RNA structure and\` function with gRNAde. bioRxiv, pages 2024–03, 2024. doi: 10.1101/2024.03.31.587283. 3, 15

[18] Chaitanya K. Joshi, Arian Rokkum Jamasb, Ramon Vinas Torn ˜ e, Charles Harris, Simon V.´ Mathis, Alex Morehead, Rishabh Anand, and Pietro Lio. gRNAde: Geometric deep learn-\` ing for 3d RNA inverse design. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=Fm4FkfGTLu. 3, 15

[19] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Z<sup>ˇ</sup> ´ıdek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. nature, 596(7873):583–589, 2021. 2, 3, 15

[20] Rohith Krishna, Jue Wang, Woody Ahern, Pascal Sturmfels, Preetham Venkatesh, Indrek Kalvet, Gyu Rie Lee, Felix S Morey-Burrows, Ivan Anishchenko, Ian R Humphreys, Ryan McHugh, Dionne Vafeados, Xinting Li, George A Sutherland, Andrew Hitchcock, C Neil Hunter, Minkyung Baek, Frank DiMaio, and David Baker. Generalized biomolecular modeling and design with RoseTTAFold All-Atom. Science, 384(6693):eadl2528, 2024. doi: 10.1126/sc ience.adl2528. 3, 15

[21] Andrew Kubaney, Andrew Favor, Lilian McHugh, Raktim Mitra, Robert Pecoraro, Justas Dauparas, Cameron Glasscock, and David Baker. Rna sequence design and protein–dna specificity prediction with na-mpnn. bioRxiv, pages 2025–10, 2025. 7, 20

[22] Andrew Leaver-Fay, Michael Tyka, Steven M Lewis, Oliver F Lange, James Thompson, Ron Jacak, Kristian W Kaufman, P Douglas Renfrew, Colin A Smith, Will Sheffler, Ian W Davis, Seth Cooper, Adrien Treuille, Daniel J Mandell, Florian Richter, Yih-En Andrew Ban, Sarel J Fleishman, Jacob E Corn, David E Kim, Sergey Lyskov, Martin Berrondo, Stuart Mentzer, Zoran Popovic, James J Havranek, John Karanicolas, Rhiju Das, Jens Meiler, Tanja Kortemme,´ Jeffrey J Gray, Brian Kuhlman, David Baker, and Philip Bradley. ROSETTA3: an object-oriented software suite for the simulation and design of macromolecules. Methods in Enzymology, 487: 545–574, 2011. doi: 10.1016/B978-0-12-381270-4.00019-6. 2, 3, 15

[23] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123–1130, 2023. 15

[24] Kresten Lindorff-Larsen, Stefano Piana, Ron O Dror, and David E Shaw. How fast-folding proteins fold. Science, 334(6055):517–520, 2011. doi: 10.1126/science.1208351. 3, 15

[25] Xuefeng Liu, Chih-chan Tien, Peng Ding, Songhao Jiang, and Rick L. Stevens. Entropyreinforced planning with large language models for drug discovery. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024. 3, 15

[26] Xuefeng Liu, Mingxuan Cao, Songhao Jiang, Xiao Luo, Xiaotian Duan, Mengdi Wang, Tobin R Sosnick, Jinbo Xu, and Rick Stevens. Monte carlo tree diffusion with multiple experts for protein design. arXiv preprint arXiv:2509.15796, 2025. 3, 15

[27] Xuefeng Liu, Songhao Jiang, Ian Foster, Jinbo Xu, and Rick Stevens. Scaffoldgpt: A scaffoldbased gpt model for drug optimization. arXiv preprint arXiv:2502.06891, 2025. 2

[28] Xuefeng Liu, Songhao Jiang, Qinan Huang, Tinson Xu, Ian Foster, Mengdi Wang, Hening Lin, and Rick Stevens. FragmentGPT: A unified GPT model for fragment growing, linking, and merging in molecular design. arXiv preprint arXiv:2509.11047, 2025. URL https: //arxiv.org/abs/2509.11047.

[29] Xuefeng Liu, Songhao Jiang, Bo Li, and Rick Stevens. Controllablegpt: A ground-up designed controllable gpt for molecule optimization. arXiv preprint arXiv:2502.10631, 2025. 2

[30] Yufeng Liu, Sheng Wang, Jixin Dong, Linghui Chen, Xinyu Wang, Lei Wang, Fudong Li, Chenchen Wang, Jiahai Zhang, Yuzhu Wang, Si Wei, Quan Chen, and Haiyan Liu. De novo protein design with a denoising diffusion network independent of pretrained structure prediction models. Nature Methods, 21(12):2278–2287, 2024. doi: 10.1038/s41592-024-02437-w. 2

[31] Martin Pacesa, Lennart Nickel, Christian Schellhaas, Joseph Schmidt, Ekaterina Pyatova, Lucas Kissling, Patrick Barendse, Jagrity Choudhury, Srajan Kapoor, Ana Alcaraz-Serna, et al. Bindcraft: one-shot design of functional protein binders. bioRxiv, pages 2024–09, 2024. 2, 3, 15

[32] Saro Passaro, Gabriele Corso, Jeremy Wohlwend, Mateo Reveiz, Stephan Thaler, Vignesh Ram Somnath, Noah Getz, Tally Portnoi, Julien Roy, Hannes Stark, David Kwabi-Addo, Dominique Beaini, Tommi Jaakkola, and Regina Barzilay. Boltz-2: Towards accurate and efficient binding affinity prediction. bioRxiv, page 2025.06.14.659707, 2025. doi: 10.1101/2025.06.14.659707. 7, 20

[33] Jian Peng and Jinbo Xu. Raptorx: exploiting structure information for protein alignment by statistical inference. Proteins: Structure, Function, and Bioinformatics, 79(S10):161–171, 2011. 15

[34] Martin L Puterman. Markov decision processes: Discrete stochastic dynamic programming. John Wiley & Sons, 2014. 3, 31

[35] Carol A Rohl, Charlie EM Strauss, Kira MS Misura, and David Baker. Protein structure prediction using rosetta. In Methods in enzymology, volume 383, pages 66–93. Elsevier, 2004. 3, 15

[36] John SantaLucia Jr. A unified view of polymer, dumbbell, and oligonucleotide dna nearestneighbor thermodynamics. Proceedings ofthe National Academy ofSciences, 95(4):1460–1465, 1998. 7, 19

[37] David E Shaw, Paul Maragakis, Kresten Lindorff-Larsen, Stefano Piana, Ron O Dror, Michael P Eastwood, Joseph A Bank, John M Jumper, J Kevin Salmon, Yibing Moraes, and Willy Wriggers. Atomic-level characterization of the structural dynamics of proteins. Science, 330 (6002):341–346, 2010. doi: 10.1126/science.1187409. 3, 15

[38] Alexis Vallee-B´ elisle, Francesco Ricci, and Kevin W Plaxco. Thermodynamic basis for the´ optimization of binding-induced biomolecular switches and structure-switching biosensors. Proceedings of the National Academy of Sciences, 106(33):13802–13807, 2009. 18

[39] Joseph L. Watson, David Juergens, Nathaniel R. Bennett, et al. De novo design of protein structure and function with rfdiffusion. Nature, 620:1089–1100, 2023. doi: 10.1038/s41586-0 23-06415-8. 2, 3, 15

[40] Odin Zhang, Xujun Zhang, Haitao Lin, Cheng Tan, Qineng Wang, Yiyang Mo, Qiang Feng, Gengmo Du, Yue Yu, et al. ODesign: A world model for biomolecular interaction design. arXiv preprint arXiv:2510.22304, 2025. URL https://arxiv.org/abs/2510.22304. 15

## Appendix

## A Related works

Physics-Based and Knowledge-Driven Biomolecular Design. Early approaches to biomolecular design were dominated by physics-based energy modeling and sampling, exemplified by frameworks such as Rosetta and its modern all-atom energy functions [35, 22, 2]. These methods provide explicit physical interpretability and fine-grained energetic control, but suffer from high computational cost and limited scalability when exploring large sequence spaces or multimolecular complexes. Molecular dynamics (MD) provides an alternative physics-first paradigm for atomistic refinement and thermodynamic reasoning [37, 24], yet remains computationally costly for design-time exploration at scale, especially for long sequences and heterogeneous complexes. While hybrid approaches combining coarse-grained sampling with heuristic pruning have improved tractability, global exploration of the coupled sequence–structure landscape remains challenging.

Learning-Based Structure Prediction and Generative Design. Deep learning–based structure prediction [33, 23] has fundamentally transformed biomolecular modeling. AlphaFold2 and AlphaFold-Multimer established strong protein(-complex) priors [19, 13] and RoseTTAFold provided a complementary open predictor [6]. More recently, generalized all-atom models extend prediction to assemblies containing proteins, nucleic acids, ligands/ions, and covalent modifications, including RoseTTAFold All-Atom [20] and AlphaFold3 [1]. Concurrently, ODesign proposes an AlphaFold3- like all-atom “world model” for all-to-all biomolecular interaction design across proteins, nucleic acids, and small molecules, with multi-level controllability via masking and conditional generation [40]. These models provide powerful learned structural priors and intrinsic confidence signals, but are primarily trained for prediction rather than optimization; repurposing them for controllable design still requires effective exploration and a principled mechanism for incorporating external constraints.

Generative design with diffusion and inverse folding. Diffusion and flow-matching models enable direct backbone/structure generation, with RFdiffusion providing a general framework for protein generation and binder/motif-conditioned design [39]. Sequence design models such as ProteinMPNN provide efficient inverse folding for fixed backbones and are widely used as downstream redesign modules [11]. For nucleic acids, emerging geometric generators and inverse-design pipelines include gRNAde (3D RNA inverse design) and RNA-FrameFlow (RNA backbone generation) [17, 18, 4], highlighting both the promise and the current fragmentation of modality-specific tooling. For protein– ligand settings, diffusion has also been used to model ligand poses and docking distributions (e.g., DiffDock) [10]. A common limitation across these approaches is that design quality often depends on large candidate pools plus refolding-based verification, and controllable global exploration remain challenging when objectives are multi-constraint or when the model is used out-of-distribution.

Gradient-based and iterative hallucination methods. A complementary line of work turns pretrained predictors into optimizers via backpropagation through confidence surrogates or via iterative cycling between structure prediction and sequence redesign. BindCraft demonstrates strong empirical performance for protein binder design via AF2-based gradient optimization and filtering [31], while BoltzDesign1 inverts an all atom predictor (Boltz-1, AF3-like) to optimize distograms for generalized binder design [8]. Cycling-based “hallucination” pipelines that repeatedly refold and redesign can avoid fine-tuning and yield designable structures in practice [5, 9, 14]. However, these methods are typically local: they rely on greedy acceptance, fixed schedules, or initialization-sensitive continuous optimization, and can be computationally expensive or unstable for long sequences and heterogeneous complexes. Moreover, constraint handling is often implicit (through heuristics) rather than explicitly controllable at the algorithmic level.

Search and planning for molecular generation. Search-based methods are well studied in Reinforcement Learning (RL) and combinatorial optimization, but are less explored for continuous, multimodal biomolecular design due to large branching factors, delayed rewards, and expensive eval uations. ERP integrates MCTS with language-model generation for ligands, improving exploration over greedy decoding [25], while MCTD-ME couples masked diffusion with MCTS for protein sequence planning and long-range revision [26]. More broadly, RL-style generative frameworks (e.g.,

GFlowNets) aim to sample diverse high-reward objects under complex objectives [7]. Despite these advances, prior planning-based methods are typically limited to a single modality (often sequences or SMILES) and do not unify all-atom sequence–structure co-design across proteins, nucleic acids, and ligand-containing complexes.

Positioning of MCTH. Relative to (i) physics-first energy optimization pipelines (e.g., Rosetta/MD), (ii) direct all-atom generators (e.g., RFdiffusion and ODesign), and (iii) gradient-based or fixed-schedule hallucination/cycling methods (e.g., BindCraft, BoltzDesign), MCTH occupies a distinct point in the design stack: it is a planning layer that wraps pretrained folding and inversefolding experts together with plug-in biophysical models (e.g., energy/geometry feasibility and clash/packing/contact penalties) and allocates expensive inference via MCTS. Our key conceptual distinction is co-hallucination: learned experts provide expressive structural proposals, while plugin biophysical feasibility/energy cues enter the same decision loop (selection/expansion/proposal reweighting) as controllable guidance signals, rather than as purely post hoc filters. This separation of concerns—experts for proposal, planning for compute allocation, and physics for controllable feasibility—enables a unified, fine-tuning-free workflow that remains applicable in data-sparse regimes such as nucleic-acid and ligand-conditioned design.

## A.1 Positioning Summary

Table 6 summarizes the high-level positioning of MCTH relative to representative method families. The goal of this comparison is not to claim that any individual ingredient is new in isolation, but to clarify the specific combination emphasized by MCTH: planning over explicit sequence–structure states, broad modality coverage, biophysical feasibility inside the decision loop, and no task-specific fine-tuning.

Table 6: High-level comparison with related method families. A check mark indicates an explicit/core component, △ indicates partial or task-specific support, and × indicates absence.
<table><tr><td>Method family / example</td><td>Plans over seq.—struct. states</td><td>Multi-modality</td><td>Biophysics in-loop</td><td>Fine-tuning- free</td></tr><tr><td>Rosetta / MD-style</td><td>X</td><td>Limited</td><td>V</td><td>√</td></tr><tr><td>optimization RFdiffusion / ODesign</td><td>×</td><td>△</td><td>X</td><td>√</td></tr><tr><td>ProteinMPNN / inverse folding</td><td>X</td><td>Limited</td><td>X</td><td>√</td></tr><tr><td>BindCraft / BoltzDesign</td><td>△</td><td>Limited</td><td>△</td><td>√</td></tr><tr><td>MCTD-ME</td><td>√</td><td>Protein only</td><td>×</td><td>√</td></tr><tr><td>MCTH</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## B Additional Methodological Details

## B.1 Uncertainty-Aware MCTS over Sequence–Structure States

Following Sec. 2, we represent a biomolecular design hypothesis by a state $s = ( S , X )$ , where S denotes the current (possibly underspecified) sequence(s) of all designable components and X denotes the corresponding all-atom complex structure under task conditioning C. MCTH casts design as a finite-budget planning problem in which the budget is dominated by expensive folding calls.

Actions and transitions. From state s, an action $a \in { \mathcal { A } } ( s )$ proposes a successor by: (i) proposing an updated design sequence $S ^ { \prime }$ for the designable component (e.g., via inverse folding, optionally with lightweight mutations), followed by (ii) invoking a folding/hallucination expert to obtain a corresponding complex structure under task conditioning C (fixed receptor, ligand specification, motif constraints, etc.), yielding $s ^ { \prime } = ( S ^ { \prime } , X ^ { \prime } )$ When multiple folding/hallucination experts are available, we instantiate an ensemble $\{ X ^ { \prime ( e ) } \} _ { e = 1 } ^ { E }$ and use cross-expert agreement as an explicit uncertainty signal.

Node value (reward) from model confidence. Each expanded state is evaluated by a task-aware value function

$$
V ( s ) = f { \big ( } m _ { 1 } ( s ) , \dots , m _ { J } ( s ) { \big ) } ,\tag{12}
$$

where $\{ m _ { j } \}$ are intrinsic signals produced by pretrained predictors, such as global fold confidence $( \mathbf { e . g . , p L D D T / p T M } )$ , interface confidence (e.g., ipTM for binding tasks), and optional geometric or contact-based terms when available. The aggregation function $\bar { f }$ maps these raw predictor outputs to a scalar state value used for MCTS selection, evaluation, and backup. In our experiments, unless otherwise stated, $f$ is instantiated as a fixed linear aggregation of confidence and interface-quality signals, with weights reported in Appendix B.6. The aggregation is task-aware but not target-tuned: the same weights are used across benchmarks unless a task-specific term, such as a motif constraint, is absent and therefore set to zero. Importantly, $V ( s )$ is computed using only forward inference outputs; no gradients, fine-tuning, or backpropagation through structure predictors is used.

Search procedure (uncertainty-aware CUPUCT-MCTS). MCTH performs Monte Carlo Tree Search rollouts under a fixed compute budget (e.g., a maximum number of expensive folding calls). Each rollout consists of four stages: selection, expansion, evaluation, and backpropagation.

Selection. Starting at the root, we recursively select child edges to traverse until reaching a leaf node. We introduce a consensus–uncertainty–physics PUCT-style score, CU-PUCT, where the exploration term is modulated by (i) a consensus prior $\pi _ { \mathrm { c o n s } , \tau } ( \boldsymbol { a } \mid \boldsymbol { s } _ { t } )$ and (ii) an uncertainty–feasibility factor from inverse/forward-fold experts, with an optional biophysical penalty $E _ { \mathrm { p h y s } } ( s _ { t } , a )$ . We compute

$$
\begin{array} { r l r } {  { \mathrm { C } \mathcal { U } \mathrm { - } \mathrm { P U C T } ( ( s _ { t } , a ) ) = Q ( s _ { t } , a ) + c _ { p } \frac { \sqrt { \log N ( s _ { t } ) } } { 1 + N ( s _ { t } , a ) } \cdot \pi _ { \mathrm { c o n s } , \tau } ( a \mid s _ { t } ) } } \\ & { } & { \cdot \Big ( w _ { \mathrm { i n v } } U _ { \mathrm { i n v } } ( s _ { t } , a ) + w _ { \mathrm { f o l d } } U _ { \mathrm { f o l d } } ( s _ { t } , a ) \Big ) \cdot \exp \Big ( -  \begin{array} { l } { w _ { \mathrm { p h y s } } E _ { \mathrm { p h y s } } ( s _ { t } , a ) \Big ) , } \end{array}  } \end{array}
$$

and select $\begin{array} { r } { a ^ { * } \bigl ( s _ { t } \bigr ) \ = \ \arg \operatorname* { m a x } _ { a \in \mathcal { A } ( s _ { t } ) } \mathbb { C } \mathcal { U } \mathrm { - P U C T } \bigl ( \bigl ( s _ { t } , a \bigr ) \bigr ) } \end{array}$ . Here $Q$ is the empirical action-value estimate, $N ( s _ { t } )$ and $N ( s _ { t } , a )$ are node/edge visit counts, and $c _ { p }$ controls the exploration–exploitation trade-off. The terms $\pi _ { \mathrm { c o n s , } \tau } , U _ { \mathrm { i n v } } , U _ { \mathrm { f o l d } }$ , and $E _ { \mathrm { p h y s } }$ are computed once at expansion and cached, keeping selection lightweight.

Consensus prior. We define $\pi _ { \mathrm { c o n s } , \tau } ( \boldsymbol { a } \mid \boldsymbol { s } _ { t } )$ to favor actions whose transition is supported by multiple experts. For a candidate child produced by inverse folding $S ^ { \prime }$ and an ensemble $\{ X ^ { \prime ( e ) } \} _ { e = 1 } ^ { E } ,$ , we form an unnormalized consensus score

$$
\tilde { \pi } _ { \mathrm { c o n s } } \big ( s _ { t } , a \big ) = \exp \Big ( \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \log q _ { \phi } ( S ^ { \prime } \mid X ^ { \prime ( e ) } , C ) \Big ) \cdot \exp \big ( - \kappa \mathcal { D } ( \{ X ^ { \prime ( e ) } \} ) \big ) ,
$$

where $q _ { \phi }$ is the inverse-folding likelihood and $\mathcal { D } ( \{ X ^ { \prime ( e ) } \} )$ is a cross-expert disagreement measure (e.g., backbone/interface dispersion). We then apply temperature-τ normalization

$$
\pi _ { \mathrm { c o n s } , \tau } ( a \mid s _ { t } ) = \frac { \tilde { \pi } _ { \mathrm { c o n s } } ( s _ { t } , a ) ^ { 1 / \tau } } { \sum _ { a ^ { \prime } } \tilde { \pi } _ { \mathrm { c o n s } } ( s _ { t } , a ^ { \prime } ) ^ { 1 / \tau } } .\tag{13}
$$

This yields an action prior that interpolates between uniform exploration (large τ ) and sharp consensusdriven focus (small τ).

Uncertainty terms. We decompose uncertainty into two sources. First, inverse-folding uncertainty

$$
U _ { \mathrm { i n v } } ( s _ { t } , a ) = \frac { 1 } { | M | } \sum _ { i \in M } { \mathsf { H } } ( q _ { \phi } ( \cdot \mid X , C ) _ { i } ) ,\tag{14}
$$

where $\mathsf { H } ( \cdot )$ is categorical entropy and M is the set of updated positions when masking is disabled, M is thefull designable set). Second, folding uncertainty is defined either from confidence-derived uncertainty or from cross-expert structural disagreement:

$$
U _ { \mathrm { f o l d } } ( s _ { t } , a ) = g ( \mathrm { p L D D T } , \mathrm { p T M } , \mathrm { i p T M } , \dots \Big ) , U _ { \mathrm { f o l d } } ( s _ { t } , a ) = \mathcal { D } ( \{ X ^ { \prime ( e ) } \} ) .\tag{15}
$$

Here, $g ( \cdot )$ maps folding confidence outputs to an uncertainty bonus. In practice, $g$ is implemented as a monotone decreasing transformation of confidence, so that lower confidence or poorer interface resolution yields larger $U _ { \mathrm { f o l d } }$ and encourages exploration of under-resolved states. The alternative form $\mathcal { D } ( \{ X ^ { \prime ( e ) } \} )$ captures epistemic disagreement across folding experts, where larger dispersion among predicted structures indicates greater folding uncertainty.

Finally, we include an optional biophysical term $E _ { \mathrm { p h y s } }$ to discourage nonphysical hypotheses. Overall, CUPUCT-MCTS provides an uncertainty-aware planning layer: $\pi _ { \mathrm { c o n s } , \tau }$ stabilizes search toward expert-agreed transitions, while $U _ { \mathrm { i n v } }$ and $\dot { U } _ { \mathrm { f o l d } }$ direct exploration toward uncertain or under-resolved regions of the coupled sequence–structure landscape.

Biophysicalfeasibility term. Finally, we incorporate an optional biophysical feasibility/energy factor $E _ { \mathrm { p h y s } } ( s _ { t } , a )$ to down-weight proposals that are high-confidence under learned priors but physically implausible $( \mathrm { e . g . }$ ., clashes, strained geometry, infeasible pockets). We define a modular biophysical score $E _ { \mathrm { p h y s } } ( \cdot ) \in [ 0 , 1 ]$ (near 0 if feasible; near 1 if implausible) and set $E _ { \mathrm { p h y s } } ( s _ { t } , a ) \triangleq E _ { \mathrm { p h y s } } ( S ^ { \prime } , X ^ { \prime } )$ for the child produced by a, computed from the proposed sequence $S ^ { \prime }$ and/or predicted coordinates $X ^ { \prime }$ without additionalfolding calls. This term enters CU-PUCT multiplicatively as $\exp ( - w _ { \mathrm { p h y s } } E _ { \mathrm { p h y s } } )$ and is cached at expansion time.

Expansion. Upon reaching a leaf, we expand it by sampling up to $K$ candidate actions (e.g., K sequence proposals) and adding the resulting successor states as children. Each child is constructed by running the inverse-folding expert (to propose $S ^ { \prime } )$ and then the folding/hallucination expert (to obtain $X ^ { \prime } )$ , producing a concrete complex hypothesis. At expansion time, we compute and cache $\pi _ { \mathrm { c o n s } , \tau } ( a \mid s _ { t } ) , U _ { \mathrm { i n v } } ( s _ { t } , a ) , U _ { \mathrm { f o l d } } ( s _ { t } , a )$ , and (when available) $E _ { \mathrm { p h y s } } ( s _ { t } , a )$ , which are then reused by the selection policy.

Evaluation. Newly added children are scored by computing $V ( s ^ { \prime } )$ from the folding expert’s confidence outputs (and any task-specific geometric checks). In our implementation, this evaluation is performed immediately after expansion (i.e., the expensive folding call doubles as the rollout evaluation), keeping the planner modular and plug-and-play.

Backpropagation. The evaluated value is propagated back along the selected path to update visit counts and value statistics. Concretely, we update $N ( \cdot )$ and maintain $Q ( \cdot )$ via a standard backup rule $( \mathrm { e . g } ^ { } )$ ., running mean or max-backup), enabling the tree policy to increasingly favor high-value branches over repeated simulations.

## B.2 Biophysical Penalty

The confidence signals $V ( s )$ (such as pLDDT and ipTM) measure how consistent a predicted structure is with the folding expert’s learned priors, but they do not capture all physical constraints relevant to real-world binding or function. Different biomolecular design tasks impose different task-specific physical requirements—thermodynamic stability windows for nucleic acid aptamers, steric feasibility for protein–ligand interfaces, geometric packing quality for de novo protein design—that are invisible to a folding expert evaluating only the bound complex.

In addition to the ML confidence signal, we introduce a modular biophysical energy term $E _ { \mathrm { p h y s } } ( S ^ { \prime } ) \in$ [0, 1] that encodes task-specific physical feasibility. $E _ { \mathrm { p h y s } }$ is near zero for physically plausible candidates and approaches one for implausible ones; it enters the PH-UCB selection rule as a multiplicative factor $\exp ( - w _ { \mathrm { p h y s } } E _ { \mathrm { p h y s } } )$ , acting as a soft preference $( w _ { \mathrm { p h y s } } = 0 . 1 )$ rather than a hard filter. The specific form of $E _ { \mathrm { p h y s } }$ is defined per task and can be computed from the sequence alone or from predicted coordinates, without requiring an additional structure prediction call.

Example: DNA aptamer design. As a concrete example, consider DNA aptamer design. Systematic thermodynamic measurements show that both extremes of self-folding stability hurt binding: aptamers with very stable self-folds must unfold before engaging the target, paying an unfolding cost that reduces effective affinity, while aptamers with too little intrinsic structure lose the preorganisation benefit and pay a steep entropic cost upon binding [3, 38]. Neither failure mode is visible to the folding expert, which only evaluates the bound complex.

For a candidate aptamer sequence $S ^ { \prime }$ , we define $E _ { \mathrm { p h y s } }$ as a soft window penalty on the nearestneighbour minimum free energy:

$$
\begin{array} { r l } { E _ { \mathrm { p h y s } } ( S ^ { \prime } ) ~ = ~ 1 - \sigma \big ( k _ { l } \left( \Delta G _ { \mathrm { s e l f } } ( S ^ { \prime } ) - \Delta G _ { l } \right) \big ) } & { } \\ { \sigma \big ( - k _ { u } \left( \Delta G _ { \mathrm { s e l f } } ( S ^ { \prime } ) - \Delta G _ { u } \right) \big ) . } \end{array}\tag{16}
$$

where $\Delta G _ { \mathrm { s e l f } } ( S ^ { \prime } )$ is the minimum free energy of the aptamer’s intramolecular self-folding, estimated using nearest-neighbour thermodynamic parameters [36] at $3 7 ^ { \circ } \mathrm { C }$ , and $\sigma ( x ) = 1 / ( \bar { 1 } + e ^ { - x } )$ is the logistic sigmoid. The product of the two sigmoids forms a soft window over $\Delta G _ { \mathrm { s e l f } } \colon$ when $\Delta G _ { l } < \Delta G _ { \mathrm { s e l f } } < \Delta G _ { u }$ , both sigmoids are close to 1 and $E _ { \mathrm { p h y s } } \approx 0 ;$ outside this range one sigmoid drops toward zero, pushing $E _ { \mathrm { p h y s } }$ toward 1. The parameters $\Delta G _ { l }$ and $\Delta G _ { u }$ define the boundaries of the favourable region, and $k _ { l } , k _ { u }$ control how sharply the penalty rises on each side.

The boundary parameters $( \Delta G _ { l } ~ = ~ - 1 0$ kcal $\mathrm { m o l ^ { - 1 } }$ $\Delta G _ { u } ~ = ~ - 2$ kcal $\mathrm { m o l } ^ { - 1 } )$ and steepness $( k _ { l } = k _ { u } = 0 . 7 5 )$ are derived from the empirical distribution of nearest-neighbour self-folding free energies across random DNA sequences (Appendix B.3).

Integration into PH-UCB selection. This penalty enters the PH-UCB selection rule as a multiplicative factor $\mathrm { \ x p } \big ( { - w _ { \mathrm { p h y s } } E _ { \mathrm { p h y s } } } \big )$ , which reduces the UCB score of candidates at either extreme. The weight $w _ { \mathrm { p h y s } } = 0 . _ { \cdot }$ 1 makes the penalty a soft preference rather than a hard filter. Like the consensus prior and uncertainty bonuses, $E _ { \mathrm { p h y s } }$ is computed once when a node is expanded and cached for reuse during selection, adding no extra cost to the selection phase. $\Delta G _ { \mathrm { s e l f } }$ is computed from the sequence alone in $< 1$ ms, without requiring an additional structure prediction call.

For aptamer types where nearest-neighbour thermodynamic models do not apply (e.g., G-quadruplex sequences governed by Hoogsteen base-pairing), we set $E _ { \mathrm { p h y s } } = 0 . 5$ (a neutral value that neither penalises nor rewards the candidate).

Overall, the exploration term is large for actions that are (i) plausible under expert consensus, (ii) informative due to residual sequence/structure uncertainty, and (iii) physically reasonable under task-specific biophysical constraints, while being down-weighted for candidates at either biophysical extreme.

## B.3 Parameter derivation.

We use the same calibration principle for all parameters: each sigmoid should reach its sensitive region (input $\approx \pm 3 )$ at the most extreme $\Delta G _ { \mathrm { s e l f } }$ values observed in practice. To ground this in data, we computed $\Delta G _ { \mathrm { s e l f } }$ for a sample of random DNA sequences across the 20–80 nt range using nearest-neighbour thermodynamics (seqfold, SantaLucia Jr $3 6 , 3 7 ^ { \circ } \mathrm { C } )$ . The observed distribution spans approximately −14 to +2 kcal mol<sup>−1</sup>, with a median near $- 2 . 5$ kcal $\mathrm { m o l ^ { - 1 } }$

For the upper boundary, we set $\Delta G _ { u } = - 2 \mathrm { k c a l m o l ^ { - 1 } }$ . A minimal stable hairpin (4-bp stem with a tetraloop) has $\Delta G \stackrel { \cdot } { \approx } - 2 \tan - 3$ kcal mol<sup>−1</sup> under nearest-neighbour parameters [36]; sequences above this threshold cannot maintain even one stable secondary-structure element.

For the lower boundary, we set $\Delta G _ { l } = - 1 0 \ \mathrm { k c a l m o l ^ { - 1 } }$ , corresponding to the 1st–5th percentile of the observed distribution. Sequences below this value carry multiple stable stem-loops whose cumulative unfolding cost substantially reduces effective binding affinity.

The steepness parameters follow from the same calibration. On the over-stability side, the observed extreme $\bar { \Delta } G _ { \mathrm { m i n } } ^ { \mathrm { ~ - ~ } } \approx - 1 4 \mathrm { k c a l m o l ^ { - 1 } }$ lies 4 kcal mol<sup>−1</sup> below $\Delta G _ { l }$ , giving $k _ { l } = 3 / 4 = 0 . 7 5$ . On the under-structure side, the observed extreme $\Delta G _ { \mathrm { m a x } } \approx + 2$ kcal mol<sup>−1</sup> lies 4 kcal mol<sup>−1</sup> above $\Delta G _ { u } ,$ giving $k _ { u } = 3 / 4 = 0 . 7 5$ . The two sides yield the same steepness, so we use a single $k _ { l } = k _ { u } = 0 . 7 5 ;$ the transition width of 4.4 $: / 0 . 7 5 \approx 6$ kcal $\mathrm { m o l ^ { - 1 } }$ is intentionally gradual, reflecting uncertainty in the exact threshold values.

## B.4 Additional details: multi-modality representation and expert dispatch

MCTH operates across diverse biomolecular modalities—proteins, peptides, DNA/RNA (single- and double-stranded), small molecules, ions, and post-translational modifications—without modalityspecific fine-tuning or architectural changes. This generality follows from two complementary design choices: (i) a unified all-atom representation in the folding expert, and (ii) a modular, modality-aware dispatch in the inverse-folding step. Across all settings, the planner state remains $s = ( \dot { S } , X )$ under conditioning C (partners, ligands, motifs, etc.), and the MCTS logic (selection/expansion/caching/backup) is unchanged.

Unified all-atom tokenization. We use Boltz-2 as the folding/refolding expert, whose unified allatom representation supports proteins, nucleic acids, small molecules, ions, and covalent modifications within a single framework [32]. As a result, MCTH inherits broad modality coverage “for free”: modality differences are expressed through conditioning C and task-specific scoring/feasibility cues, rather than through retraining or specialized architectures.

<table><tr><td>Setting</td><td>Designable component</td><td>Inverse-folding expert</td><td>Typical biophysics cue</td></tr><tr><td>Protein/peptide design</td><td>protein/peptide sequence</td><td>ProteinMPNN</td><td>clash/packing/contact</td></tr><tr><td>Ligand-conditioned protein design</td><td>pocketed protein sequence</td><td>LigandMPNN</td><td>sterics / pose geometry</td></tr><tr><td>RNA/DNA aptamer design</td><td>NA sequence</td><td>NA-MPNN</td><td> $\Delta G _ { \mathrm { s e l f } } ,$  base-pair potentials</td></tr><tr><td>Hybrid complexes</td><td>per-chain (as applicable)</td><td>dispatched</td><td>task-specific</td></tr></table>

Table 7: Modality-aware inverse-folding dispatch. The folding expert (Boltz-2) is shared across settings via a unified all-atom representation; modality differences enter primarily through the inversefolding backend and optional plug-in biophysical feasibility cues.

Modality-aware expert dispatch. Sequence redesign is implemented as a lightweight dispatch based on the design target type. Concretely, the planner invokes an appropriate inverse-folding expert for the currently designable component (e.g., ProteinMPNN for proteins and peptides [11], NA-MPNN for nucleic acids [21], and LigandMPNN for ligand-pocket-aware protein redesign). The state representation, tree policy, and value computation remain fixed; only the inverse-folding call differs. Table 7 summarizes the dispatch and typical plug-in biophysical cues used for controllability.

## B.5 Inference-time structural guidance for nucleic acids

Sequence-level biophysical signals—such as self-folding free energy or base-pair assignments— capture important physical constraints, but they do not directly influence the structure generation process. A more direct form of biophysical control is to inject physical constraints inside the structure generation itself, steering the diffusion model’s denoising trajectory toward conformations that are consistent with the sequence’s known secondary-structure preferences.

For a candidate sequence $S ^ { \prime }$ with self-folding stability in the thermodynamically favourable range $( \Delta G _ { l } \le \Delta G _ { \mathrm { s e l f } } ( \hat { S ^ { \prime } } ) \le \Delta G _ { u } ;$ boundaries derived in Appendix B.3), we derive an inference-time structural guidance signal from the seqfold computation.

seqfold outputs. The seqfold computation for a candidate sequence $S ^ { \prime }$ produces two outputs from a single call (< 1 ms):

1. $\Delta G _ { \mathrm { s e l f } } ( S ^ { \prime } )$ — the minimum free energy of intramolecular self-folding.

2. $P ( S ^ { \prime } )$ — the MFE base-pair assignment, i.e., the set of intramolecular base pairs predicted to form in the free aptamer.

This section uses $P ( S ^ { \prime } )$ as a contact guidance potential inside Boltz-2’s structure prediction (described below). The contact guidance adds no extra computational overhead because $\hat { P ( S ^ { \prime } ) }$ is already available from the same seqfold call that produces $\Delta G _ { \mathrm { s e l f } }$

Inference-time structural guidance via Boltz-2 potentials. We activate Boltz-2’s inference-time potential-energy steering (use potentials = True), which steers the denoising trajectory via gradient-based $\scriptstyle { \hat { x } } _ { 0 }$ correction and Feynman-Kac particle resampling [12] entirely within the forward pass—no backpropagation through the diffusion model is required.

Within this steering framework, we inject $P ( S ^ { \prime } )$ as a contact guidance potential: aptamer positions I that form contacts with the protein interface in the current hallucinated complex $X ^ { \prime }$ are identified, and base pairs in $P ( S ^ { \prime } )$ whose both endpoints lie outside I (non-interface pairs) are used as soft structural constraints. This encourages the aptamer scaffold to retain its pre-organized secondary structure away from the binding site, while bases at the interface $( \in I )$ are left unconstrained so they remain available for protein contact. The contact weight is higher during the high-noise structural assembly phase and disabled near convergence, following Boltz-2’s default schedule.

Formally, let ${ \mathcal { P } } _ { \mathrm { f r e e } } = \{ ( i , j ) \in P ( S ^ { \prime } ) : i \notin I , j \notin I \}$ denote the set of non-interface base pairs predicted by seqfold. The guidance potential is

$$
U _ { \mathrm { c o n t a c t } } ( X ) = \sum _ { ( i , j ) \in \mathcal { P _ { \mathrm { f r e e } } } } w _ { i j } \big ( d _ { i j } ( X ) - d _ { 0 } \big ) ^ { 2 } ,\tag{17}
$$

where $d _ { i j } ( X )$ is the distance between the heavy atoms of positions i and j in structure $X , d _ { 0 }$ is the target contact distance for a Watson-Crick base pair, and $w _ { i j }$ is a weight that decays with diffusion noise level. This potential is incorporated into the Boltz-2 denoising step via the xˆ<sub>0</sub>-correction mechanism, yielding a guided reverse transition.

## B.6 Reproducibility and Hyperparameters

Table 8 summarizes the audited configuration used for the main experiments. Unless otherwise stated, the same configuration is used across RNA aptamer, DNA aptamer, protein–protein binder, and ligand-conditioned design tasks. We do not tune search hyperparameters per dataset; modality differences enter only through the inverse-folding backend and the task conditioning passed to the folding expert.

Table 8: Main hyperparameters and configuration.
<table><tr><td>Component</td><td>Parameter</td><td>Value</td></tr><tr><td>Tree search</td><td>Fold-call budget T</td><td>100</td></tr><tr><td></td><td>Expansion width K</td><td>4</td></tr><tr><td></td><td>Maximum tree depth  $D _ { \mathrm { m a x } }$ </td><td>5</td></tr><tr><td></td><td>Maximum MCTS iterations</td><td>50</td></tr><tr><td></td><td>Fold calls per iteration</td><td>8</td></tr><tr><td></td><td>Early-stop patience</td><td>10 iterations without improvement</td></tr><tr><td></td><td>Exploration constant  $c _ { p }$ </td><td> ${ \sqrt { 2 } } \approx 1 . 4 1 4$ </td></tr><tr><td></td><td>Backup rule</td><td>sum-backup, Q</td></tr><tr><td></td><td>Main-table seed</td><td>total_reward/visits</td></tr><tr><td></td><td>Ablation seeds</td><td>42 {11, 23, 47}</td></tr><tr><td>Value function</td><td></td><td>1.0</td></tr><tr><td></td><td>Wconf</td><td>0.5</td></tr><tr><td></td><td>Wiface Wmotif</td><td>5.0 when motif constraints are specified; 0</td></tr><tr><td>Folding expert</td><td>Model</td><td>otherwise Boltz-2</td></tr><tr><td></td><td>Recycling steps</td><td>3</td></tr><tr><td></td><td>Diffusion denoising steps</td><td>200</td></tr><tr><td></td><td>Sampling steps</td><td>200</td></tr><tr><td></td><td>Diffusion samples per call</td><td>1</td></tr><tr><td>Inverse-folding experts</td><td>Protein / peptide</td><td>ProteinMPNN</td></tr><tr><td></td><td>Nucleic acid (RNA / DNA)</td><td>NA-MPNN</td></tr><tr><td></td><td>Ligand-pocket-aware protein</td><td>LigandMPNN</td></tr><tr><td></td><td>Sampling temperature τ</td><td>0.1</td></tr><tr><td></td><td>Samples per inverse-fold call</td><td>4</td></tr><tr><td>Compute</td><td>GPU</td><td>NVIDIA A100 80GB</td></tr><tr><td></td><td>Per target / seed</td><td>1 GPU</td></tr></table>

For each target, we report the design with the highest ipTM among all states evaluated during search, including both internal and leaf states. This candidate-selection rule is fixed across all reported MCTH runs.

## C Additional Experiments

This supplement reports additional evaluations showing that the same MCTH pipeline can be instantiated across modalities and constraint types beyond the three main-text benchmarks (Sec. 4). For concision, we currently include only protein–ligand binding and the corresponding inverse-folding ablation (Appendix D); other additional experiments are commented out below.

<table><tr><td>Target</td><td>RFDiff-AA MCTH</td><td>ODesign</td></tr><tr><td>5SDV_IAI_A</td><td>0.932 0.949</td><td>0.691</td></tr><tr><td>5SDV_IAI_B</td><td>0.959 0.939</td><td>0.516</td></tr><tr><td>5SDV_IAI_C</td><td>0.924 0.895</td><td>0.269</td></tr><tr><td>5SDV_IAI_D</td><td>0.961 0.902</td><td>0.449</td></tr><tr><td>7BKC_FAD_A</td><td>0.465 0.848</td><td>0.200</td></tr><tr><td>7BKC_FAD_a</td><td>0.878 0.919</td><td>0.291</td></tr><tr><td>7BKC_FAD_E</td><td>0.839 0.833</td><td>0.410</td></tr><tr><td>7BKC_FAD_e</td><td>0.720 0.846</td><td>0.510</td></tr><tr><td>7C7M_SAM_A</td><td>0.928 0.940</td><td>0.847</td></tr><tr><td>7V11_0Q0_A</td><td>0.897 0.938</td><td>0.413</td></tr><tr><td>Mean</td><td>0.850</td><td>0.901 0.460</td></tr></table>

Table 9: Protein–ligand binding under Boltz-2 refold evaluation. Best ipTM per target (higher is better). Bold indicates the better of MCTH and RFDiff-AA for each target; MCTH attains a higher mean ipTM.

## C.1 Protein–Ligand Binding Design

We evaluate MCTH on protein–small-molecule binding design, where the objective is to generate a protein sequence that forms a confident complex with a specified ligand.

Benchmark. We construct a 10-target ligand-binding benchmark from four PDB complexes: 5SDV (IAI; chains A/B/C/D), 7BKC (FAD; chains A/a/E/e), 7C7M (SAM; chain A), and 7V11 (OQO; chain A). Each target specifies a fixed ligand (CCD code) and a binding protein chain.

Setup. MCTH instantiates MCTS with Boltz-2 folding in the loop and LigandMPNN as the inversefolding proposal model. Starting from random sequences, the search iterates folding, scoring with an ipTM-dominant value function, and MCTS expansion with ligand-aware inverse-fold proposals. We use a fixed budget of up to 100 Boltz-2 fold calls per target.

Baselines and evaluation. We compare against (i) RFDiffusion-AA + LigandMPNN and (ii) ODesign (rigid ligand checkpoint). All designs are evaluated under a common Boltz-2 re-fold protocol. We report ipTM (higher is better). A specialist comparison to BoltzDesign1 is provided as a set-aside table below.

Results. MCTH attains a higher mean ipTM (0.901) than RFDiffusion-AA (0.850) and ODesign (0.460), with the main gap concentrated on the most challenging FAD cofactor targets (higher pocket-precision demands). Ablations are in Appendix D.

## D Additional Ablations

## D.1 Ligand-aware vs. ligand-blind inverse folding

To isolate the effect of ligand-aware sequence redesign, we compare MCTH under two inverse-folding backends: (i) LigandMPNN (ligand-aware; conditions on ligand atoms in the binding pocket), and (ii) ProteinMPNN (ligand-blind; conditions only on protein backbone atoms). All other components are held fixed: identical MCTS configuration, identical ipTM-dominant scoring for ligand tasks, and the same Boltz-2 evaluation protocol as in Appendix C.1. We report best-of-N ipTM (higher is better) and iPDE (lower is better).

Summary. Ligand-aware inverse folding substantially improves both interface confidence and interface precision: LigandMPNN improves mean ipTM from 0.826 to 0.901 and reduces mean iPDE from 4.20 to 3.02.

Per-target results.

<table><tr><td rowspan="2">Ligand</td><td colspan="2">ipTM ↑</td><td colspan="2">iPDE↓</td></tr><tr><td>LigandMPNN</td><td>ProteinMPNN</td><td>LigandMPNN</td><td>ProteinMPNN</td></tr><tr><td>IAI (4 targets)</td><td>0.921</td><td>0.900</td><td>1.60</td><td>1.18</td></tr><tr><td>FAD (4 targets)</td><td>0.862</td><td>0.730</td><td>5.46</td><td>7.98</td></tr><tr><td>SAM (1 target)</td><td>0.940</td><td>0.818</td><td>0.92</td><td>4.55</td></tr><tr><td>OQO (1 target)</td><td>0.938</td><td>0.924</td><td>1.02</td><td>0.84</td></tr></table>

Table 10: Ablation by ligand type. Ligand-aware inverse folding yields the largest improvements on FAD and SAM, where pocket complementarity is most critical.

<table><tr><td rowspan="2">Target</td><td colspan="2">ipTM ↑</td><td colspan="2">iPDE↓</td></tr><tr><td>LigandMPNN</td><td>ProteinMPNN</td><td>LigandMPNN</td><td>ProteinMPNN</td></tr><tr><td>5SDV_IAI_A</td><td>0.949</td><td>0.889</td><td>2.24</td><td>1.25</td></tr><tr><td>5SDV_IAI_B</td><td>0.939</td><td>0.921</td><td>1.28</td><td>1.00</td></tr><tr><td>5SDV_IAIC</td><td>0.895</td><td>0.876</td><td>1.65</td><td>1.42</td></tr><tr><td>5SDV_IAID</td><td>0.902</td><td>0.914</td><td>1.23</td><td>1.04</td></tr><tr><td>7BKC_FAD_A</td><td>0.848</td><td>0.733</td><td>7.69</td><td>9.49</td></tr><tr><td>7BKC_FAD_a</td><td>0.919</td><td>0.737</td><td>5.38</td><td>8.46</td></tr><tr><td>7BKC_FAD_E</td><td>0.833</td><td>0.753</td><td>6.60</td><td>7.75</td></tr><tr><td>7BKC_FAD_e</td><td>0.846</td><td>0.695</td><td>2.17</td><td>6.22</td></tr><tr><td>7C7M_SAM_A</td><td>0.940</td><td>0.818</td><td>0.92</td><td>4.55</td></tr><tr><td>7V11_0Q0_A</td><td>0.938</td><td>0.924</td><td>1.02</td><td>0.84</td></tr></table>

Table 11: Per-target ablation results. Ligand-aware inverse folding improves both ipTM and iPDE on most targets, with the strongest gains on FAD and SAM.

## D.2 Evaluation Protocol and Cross-Oracle Validation

We distinguish between search-time and held-out evaluation.

Search-time oracle evaluation. For each MCTH instantiation, candidate states are scored using the folding-model signals available to the search procedure. The modality-spanning main experiments use Boltz-2 as the search-time folding oracle. For comparisons that can be re-scored consistently, including ODesign, RFDiffusion, RFDiffusion-AA, and Complexa, generated sequences are evaluated using the same Boltz-2 refolding protocol. BoltzDesign1 instead uses its native Boltz evaluation during optimization; we therefore report it separately while matching the number of folding/evaluator calls where applicable.

These scores measure optimization under a common computational proxy and should not be interpreted as oracle-independent evidence of physical binding.

Matched-budget evaluation. To separate search quality from raw access to the evaluator, we match candidate or folding-call budgets whenever possible. In the main benchmarks, ODesign, RNAFrame-Flow, Complexa, and BoltzDesign1 are compared at up to 100 candidate or folding/evaluator calls where applicable. Appendix ?? further compares MCTH against exploration-off search, greedy cycling, and best-of-N sampling under identical initialization, proposal models, folding oracle, and evaluation budget.

Held-out predictor evaluation. We additionally evaluate designs with structure predictors that are not used to guide the corresponding search. AlphaFold3 is used to re-score the RNA and DNA aptamer benchmarks. For the protein-binder dual-expert experiment, Boltz-2 and AlphaFold2 are used during search, while Chai-1 is held out entirely from proposal, tree selection, and output ranking. These evaluations test whether improvements transfer beyond the optimized predictor rather than merely reproducing its scoring preferences.

Oracle-independent physical evaluation. For protein binders, we additionally use Rosetta InterfaceAnalyzer as a non-search evaluation of interface geometry and energetics. We report sizenormalized interface energy together with packing and buried unsatisfied hydrogen-bond statistics. These metrics provide an additional physical-plausibility check but are not treated as experimental affinity measurements.

Final-design pass rate. Best-per-target ipTM measures whether a method can discover a strong candidate but does not characterize design yield. We therefore also compare final protein-binder archives under a common held-out binder-quality filter. Candidates are selected without access to held-out metrics, and pass rates are computed only after final archive construction.

## D.3 Matched-Budget Search Ablations

To isolate the effect of search-guided compute allocation, we run matched-budget ablations under the same 100 folding-call budget. For each target, all variants use identical initialization, evaluation, scoring, folding oracle, and inverse-folding backend. The only difference is the planning strategy.

We compare the following variants:

• MCTH (full): tree search with exploration.

• Exploration-off $( c _ { p } = 0 )$ : the same tree search without the exploration term, corresponding to pure exploitation.

• Greedy cycling: sequential inverse-folding → folding refinement along a single trajectory, without branching.

• Best-of-N: independent sampling under the same fold-call budget, without feedback or iterative refinement.

Greedy cycling isolates the effect of iterative refinement without branching, while best-of-N isolates pure sampling from strong pretrained proposal models without feedback. Together, these baselines test whether MCTH’s gains come from search-guided compute allocation rather than refinement or sampling alone.

Table 12: Matched-budget ablation on RNA aptamer design. All values are mean ± standard deviation over three seeds.
<table><tr><td>Variant</td><td>7WKP A→B</td><td>7YEW A→B</td><td> ${ 8 \mathrm { G X B } \mathrm { C } \mathrm {  B } }$ </td><td>Mean</td><td>∆ vs. Full</td></tr><tr><td>MCTH (full)</td><td> $\mathbf { 0 . 9 0 2 { \scriptstyle \pm 0 . 0 1 0 } }$ </td><td> $\mathbf { 0 . 9 7 9 } \pm \mathbf { 0 . 0 0 } 2$ </td><td> $\mathbf { 0 . 8 5 8 { \scriptstyle \pm 0 . 0 0 6 } }$ </td><td>0.913</td><td>0.000</td></tr><tr><td>Exploration-off  $\left( c _ { p } = 0 \right)$ </td><td> $0 . 7 5 6 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 9 7 2 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 7 9 1 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td>0.839</td><td>-0.073</td></tr><tr><td>Greedy cycling</td><td> $0 . 7 5 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 9 5 7 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 6 4 6 { \pm } 0 . 0 1 2$ </td><td>0.784</td><td>-0.129</td></tr><tr><td>Best-of-N random</td><td> $0 . 7 5 2 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $0 . 9 5 3 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 6 2 9 { \pm } 0 . 0 1 0$ </td><td>0.778</td><td>-0.135</td></tr></table>

Table 13: Matched-budget ablation on DNA aptamer design. All values are mean ± standard deviation over three seeds.
<table><tr><td>Variant</td><td>7XVN A_N</td><td>7YUK A.C</td><td>8PMF A_B</td><td>Mean</td><td>∆ vs. Full</td></tr><tr><td>MCTH (full)</td><td> $\mathbf { 0 . 9 5 5 } \pm \mathbf { 0 . 0 0 } 2$ </td><td> $\mathbf { 0 . 9 4 6 { \scriptstyle \pm 0 . 0 0 8 } }$ </td><td> $\mathbf { 0 . 9 6 4 } \pm \mathbf { 0 . 0 0 5 }$ </td><td>0.955</td><td>0.000</td></tr><tr><td>Exploration-off  $\left( c _ { p } = 0 \right)$ </td><td> $0 . 9 0 9 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 9 0 4 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 3 0 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td>0.914</td><td>-0.041</td></tr><tr><td>Greedy cycling</td><td> $0 . 7 5 6 { \pm } 0 . 0 1 4$ </td><td> $0 . 9 0 9 { \pm } 0 . 0 1 8$ </td><td> $0 . 8 1 9 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>0.828</td><td>-0.127</td></tr><tr><td>Best-of-N random</td><td> $0 . 7 5 3 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 8 3 4 { \pm } 0 . 0 0 5$ </td><td>0.829</td><td>-0.126</td></tr></table>

Table 14: Matched-budget ablation on protein–protein binder design. All values are mean ± standard deviation over three seeds.
<table><tr><td>Variant</td><td>3DI3</td><td>403V</td><td>Mean</td><td>∆ vs. Full</td></tr><tr><td>MCTH (full)</td><td> $\mathbf { 0 . 9 1 2 } \pm \mathbf { 0 . 0 0 } 2$ </td><td> $\mathbf { 0 . 9 6 4 } \pm \mathbf { 0 . 0 0 } 2$ </td><td>0.938</td><td>0.000</td></tr><tr><td>Exploration-off  $( c _ { p } = 0 )$ </td><td> $0 . 8 9 7 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 9 5 4 { \pm } 0 . 0 1 6$ </td><td>0.925</td><td>-0.013</td></tr><tr><td>Greedy cycling</td><td> $0 . 9 0 1 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 9 5 5 { \pm } 0 . 0 1 3$ </td><td>0.928</td><td>-0.011</td></tr><tr><td>Best-of-N random</td><td> $0 . 8 0 3 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td>0.851</td><td>-0.087</td></tr></table>

Table 15: Summary of matched-budget ablations across all targets.
<table><tr><td>Variant</td><td>Overall mean</td><td>Protein mean</td><td>RNA mean</td><td>DNA mean</td></tr><tr><td>MCTH (full)</td><td>0.935</td><td>0.938</td><td>0.913</td><td>0.955</td></tr><tr><td>Exploration-off  $( c _ { p } = 0 )$ </td><td>0.889</td><td>0.925</td><td>0.839</td><td>0.914</td></tr><tr><td>Greedy cycling</td><td>0.836</td><td>0.928</td><td>0.784</td><td>0.828</td></tr><tr><td>Best-of-N random</td><td>0.815</td><td>0.851</td><td>0.778</td><td>0.829</td></tr></table>

<table><tr><td>Variant</td><td>Final mean</td><td>Never improve</td><td>Avg. last improvement</td></tr><tr><td>MCTH (full)</td><td>0.935</td><td>0.0%</td><td>46.6</td></tr><tr><td>Exploration-off  $( c _ { p } = 0 )$ </td><td>0.889</td><td>61.9%</td><td>29.3</td></tr><tr><td>Greedy cycling</td><td>0.836</td><td>53.8%</td><td>23.5</td></tr><tr><td> $\mathrm { B e s t - o f } { - } \dot { N }$ </td><td>0.815</td><td>44.4%</td><td>33.3</td></tr></table>

Table 16: Effective convergence at $\overline { { \epsilon = 0 . 0 0 2 } }$ under the matched folding-call budget. “Never improve” denotes runs with no improvement above ϵ after initialization.

Across all targets, MCTH achieves the highest final quality, with especially large gains on RNA and DNA tasks. This suggests that the improvement comes from search-guided compute allocation rather than simply from iterative refinement or a larger independent candidate pool.

## D.4 Effective Convergence Diagnostic

Final performance alone does not reveal whether additional folding calls continue to produce useful search progress. We therefore measure the last folding call at which the incumbent best score improves by more than $\epsilon = 0 . 0 0 2$ . Runs with no improvement above this threshold are assigned a last-improvement call of zero and counted as immediately stagnant.

MCTH continues to extract useful improvements later in the evaluation budget: every MCTH run improves beyond the threshold, with the last improvement occurring at fold call 46.6 on average. In contrast, 44–62% of the non-adaptive or reduced-search runs never make a nontrivial improvement after initialization. Together with the matched-budget results in Appendix ??, this indicates that the advantage of MCTH is not simply additional iterative refinement; tree search changes how useful evaluations are allocated over the course of optimization.

## D.5 Cross-Oracle Agreement Analysis

Held-out re-scoring tests whether improvements under the search-time oracle transfer to an independent structure predictor. We further quantify this relationship at the design level: given a fixed target, do Boltz-2 and the held-out predictor rank candidate designs similarly?

Design-level agreement. We compute within-target Spearman correlation between Boltz-2 and AlphaFold3 scores and bootstrap over targets to obtain 95% confidence intervals. This within-target analysis removes between-target difficulty effects that can otherwise inflate pooled correlations.

Agreement is moderate for DNA aptamers, weak but positive for protein binders, and not distinguishable from zero for RNA aptamers. Importantly, the pooled raw ipTM correlation across all designs is much larger $( \rho = 0 . 6 7 0 )$ than the within-target correlation $( \rho = 0 . 2 6 1 )$ . The pooled value is therefore driven substantially by target-level difficulty that both predictors capture, rather than by agreement over which design is best for a fixed target.

Global folding versus interface ranking. The disagreement is specific to interface-level discrimination rather than global folding quality. For protein binders $( n = 2 5 5 )$ , within-target agreement in pLDDT is $\rho = 0 . 7 6 2$ (95% CI [0.64, 0.85]), compared with only $\rho = 0 . 2 5 2$ (95% CI [0.05, 0.46]) for ipTM. The non-overlapping confidence intervals indicate that the two predictors agree substantially more on whether a complex is globally well folded than on the relative quality of its binding interface. This distinction is important for binder and aptamer design, where interface discrimination is the primary optimization objective.

<table><tr><td>Modality</td><td>n designs</td><td>Within-target ρ</td><td>95% CI</td></tr><tr><td>DNA aptamer</td><td>50</td><td>0.487</td><td>[0.29, 0.67]</td></tr><tr><td>Protein binder</td><td>255</td><td>0.252</td><td>[0.05, 0.46]</td></tr><tr><td>RNA aptamer</td><td>77</td><td>0.173</td><td>[−0.06, 0.36]</td></tr></table>

Table 17: Design-level agreement between Boltz-2 and AlphaFold3 interface scores. Correlations are computed within target and confidence intervals are obtained by bootstrap over targets.

<table><tr><td>Modality</td><td>Top-1 hit</td><td>Random</td><td>Prec.@25%</td><td>Random</td></tr><tr><td>DNA aptamer</td><td>0.600</td><td>0.200</td><td>0.600</td><td>0.200</td></tr><tr><td>Protein binder</td><td>0.167</td><td>0.053</td><td>0.427</td><td>0.241</td></tr><tr><td>RNA aptamer</td><td>0.286</td><td>0.091</td><td>0.143</td><td>0.273</td></tr></table>

Table 18: Screening utility of Boltz-2 for selecting designs favored by the held-out predictor. Random columns give the corresponding chance-level expectation under the same target-level candidate-set sizes.

Use as a screening proxy. We next ask whether Boltz-2 can serve as a practical screening proxy for selecting designs that score highly under the held-out oracle. For each target, we measure whether the Boltz-selected top design falls in the held-out top set (top-1 hit) and the precision of the Boltz top quartile with respect to the held-out top quartile.

Boltz-2 is a useful but imperfect proxy for DNA, where both top-1 recovery and top-quartile precision are approximately three times their random baselines. For protein binders, screening performance is above random but substantially weaker. For RNA, the observed top-quartile precision does not exceed random expectation.

RNA score saturation. The weak RNA agreement is partly explained by restricted score dynamic range rather than clear opposite rankings. Across the RNA candidates, the ipTM interquartile range is only 0.080 under AlphaFold3 and 0.022 under Boltz-2; the latter has a median ipTM of approximately 0.946. Thus, many RNA designs lie near the Boltz-2 score ceiling, leaving little resolution for within-target ranking.

Method-level rankings. At the method level, the two predictors preserve the broad direction of performance but are not equivalent. Kendall’s τ between method rankings is 0.333 for DNA (10 targets), 0.738 for protein binders (5 common targets), and 0.333 for RNA (7 targets). These results motivate reporting held-out evaluation alongside the search-time oracle rather than treating either predictor as a ground-truth interface metric.

Exploratory check for search-oracle preference. Finally, we test whether methods that optimize Boltz-derived feedback tend to receive relatively better rankings under Boltz-2 than under AlphaFold3. Across target–method pairs, Boltz-based systems shift by +0.367 rank positions on average (positive means relatively favored by Boltz-2), whereas non-Boltz systems shift by −0.353, a difference of 0.720 rank positions. A 20,000-permutation test that shuffles the Boltz/non-Boltz label within modality gives p = 0.067. We therefore treat this as a suggestive diagnostic rather than evidence of a systematic evaluator bias. It nevertheless reinforces the need for the held-out evaluations used throughout Sec. 4.

## D.6 Dual-Expert Search and Final-Archive Yield

We provide additional implementation and archive-level evaluation details for the dual-expert proteinbinder experiment in Sec. 4.2. The purpose of this experiment is to test whether the multi-expert mechanism defined in Sec. 3.2 improves robustness beyond optimization against a single folding predictor.

Dual-expert configuration. We retain the same MCTS procedure and ProteinMPNN sequenceproposal mechanism used in the single-expert protein-binder setting. The difference is that each candidate sequence is evaluated with both Boltz-2 and AlphaFold2. Their structural-quality signals contribute to node evaluation, while cross-expert agreement, disagreement, and uncertainty enter the CU-PUCT search rule. No component model is fine-tuned or differentiated through.

<table><tr><td>Archive</td><td>Passing/valid</td><td>Pass rate</td><td>Targets ≥ 1 pass</td></tr><tr><td>Dual-expert MCTH, top-5</td><td>27/55</td><td>49.1%</td><td>8/11</td></tr><tr><td>Dual-expert MCTH, top-10</td><td>41/110</td><td>37.3%</td><td>8/11</td></tr><tr><td>Dual-expert MCTH, top-25</td><td>84/275</td><td>30.5%</td><td>9/11</td></tr></table>

Table 19: Final-archive yield for dual-expert MCTH. Candidates are ranked only by the in-loop joint reward before applying the held-out filter.

Each Boltz-2 or AlphaFold2 structure-prediction invocation is counted as one folding/evaluator call. Thus, evaluating one sequence with both experts consumes two calls from the fixed inference budget. This accounting prevents the dual-expert variant from receiving uncounted additional structureprediction compute.

Chai-1 is not used for sequence proposal, node evaluation, tree selection, backup, or archive ranking.   
It is run only after search and therefore serves as a held-out predictor.

Cross-predictor transfer. Across the 11 CaoData targets shared by the single-expert, dualexpert, and RFDiffusion evaluations, the original Boltz-only MCTH obtains mean best ipTM values of 0.69/0.19/0.18 under Boltz-2/AF2/Chai-1, respectively. The dual-expert variant obtains 0.74/0.61/0.57. Thus, introducing AF2 as a second search-time expert substantially improves performance under AF2 while maintaining Boltz-2 performance, and the improvement transfers to the held-out Chai-1 evaluator. Full per-target results are reported in Table 4.

For MCTH, the evaluator-wise values in Table 4 are maxima over three retained final designs. Because the maximum is taken independently for each evaluator, the Boltz-2, AF2, and Chai-1 values for a target need not correspond to the same sequence. RFDiffusion values are maxima over ten outputs, so we do not interpret the unequal-pool best-of-output comparison as a direct design-yield comparison.

Final-archive construction. To measure yield beyond a single best candidate, we reconstruct a reward-ranked archive from the dual-expert search logs. Within each target, candidates from the available dual-expert runs are pooled and deduplicated by exact binder amino-acid sequence identity after removing whitespace and converting sequences to uppercase. Candidate ranking uses only the joint reward recorded during search; no held-out Chai-1 quantity is used for retrospective selection.

We report the top-K unique archive for $K \in \{ 5 , 1 0 , 2 5 \}$ . Intermediate MCTS nodes are optimization states rather than independently submitted final designs and are therefore not used as the denominator of the final-design pass rate.

Held-out binder-quality filter. After archive construction, each candidate is independently evaluated with Chai-1. We define a design as passing when its held-out interface confidence satisfies ip $\Gamma \mathrm { M } _ { \mathrm { C h a i } } \geq 0 . 5 0$ . This threshold is fixed before method comparison and is applied identically to MCTH and RFDiffusion. Chai-1 is never used during sequence proposal, MCTS selection, reward computation, or archive ranking, so the pass criterion is fully out of the search loop. All designs in the reported archives have valid Chai-1 evaluations; missing or failed evaluations, if present, are excluded from the valid denominator and reported separately.

The pass fraction decreases as progressively lower-ranked candidates enter the archive (49.1% → $3 7 . 3 \dot { 7 } \%  3 0 . 5 \% )$ , while target coverage increases from 8/11 to 9/11. This behavior is consistent with a precision–coverage trade-off rather than a single threshold-sensitive result.

Matched comparison with RFDiffusion. For a direct baseline comparison, we use exactly ten final designs per target for both methods on the same 11-target set and apply the identical held-out filter. MCTH uses its reward-ranked top-10 unique archive, while RFDiffusion uses its ten generated outputs without held-out re-ranking.

Under equal output counts, dual-expert MCTH produces 41 passing designs compared with 22 for RFDiffusion, corresponding to a 17.3 percentage-point or approximately 1.86× higher pass rate.

<table><tr><td>Method</td><td>Outputs/target</td><td>Passing/valid</td><td>Pass rate</td><td>Targets ≥ 1 pass</td></tr><tr><td>Dual-expert MCTH</td><td>10</td><td>41/110</td><td>37.3%</td><td>8/11</td></tr><tr><td>RFDiffusion</td><td>10</td><td>22/110</td><td>20.0%</td><td>8/11</td></tr></table>

Table 20: Matched final-design yield using ten outputs per target and the same held-out binder-quality filter.

<table><tr><td>Target</td><td>Chai pass/10</td><td>Chai mean/best ipTM</td><td>AF2 mean/best ipTM</td></tr><tr><td>1DJS</td><td>1/10</td><td>0.327 / 0.594</td><td>0.166 / 0.278</td></tr><tr><td>1MOX</td><td>0/10</td><td>0.252 / 0.498</td><td>0.148 / 0.240</td></tr><tr><td>1XIW</td><td>6/10</td><td>0.540 / 0.808</td><td>0.536 / 0.840</td></tr><tr><td>2GY7</td><td>5/10</td><td>0.519 / 0.729</td><td>0.419 / 0.669</td></tr><tr><td>2IFG</td><td>2/10</td><td>0.288 / 0.700</td><td>0.259 / 0.783</td></tr><tr><td>3DI3</td><td>0/10</td><td>0.228 / 0.258</td><td>0.233 / 0.616</td></tr><tr><td>3FKD</td><td>0/10</td><td>0.230 / 0.310</td><td>0.559 / 0.883</td></tr><tr><td>3MJG</td><td>1/10</td><td>0.248 / 0.525</td><td>0.107 / 0.159</td></tr><tr><td>3ZTJ</td><td>1/10</td><td>0.280 / 0.660</td><td>0.258 / 0.557</td></tr><tr><td>403V</td><td>3/10</td><td>0.451 / 0.780</td><td>0.349 / 0.751</td></tr><tr><td>40GA</td><td>3/10</td><td>0.430 / 0.685</td><td>0.422 / 0.656</td></tr></table>

Table 21: Per-target RFDiffusion held-out evaluation for the matched top-10 comparison. Mean/best values are computed over the ten outputs for each target.

Both methods identify at least one passing design on 8/11 targets. The result therefore reflects a higher density of qualifying outputs rather than broader target coverage alone.

RFDiffusion held-out score distribution. For completeness, Table 21 reports the per-target heldout scores used in the matched baseline analysis.

Together, the cross-predictor and archive analyses address two distinct questions. The dual-expert experiment tests whether search guided by more than one structural model transfers to a predictor outside the optimization loop, while the matched pass-rate analysis tests whether this robustness extends beyond a single best candidate to the final output distribution. Neither analysis constitutes experimental binding validation; rather, they measure cross-predictor robustness and computational final-design yield under controlled evaluation protocols.

## D.7 Oracle-Independent Rosetta Interface Evaluation

Structure-predictor confidence is not a physical energy and may preserve biases of the models used during search. We therefore additionally evaluate protein-binder designs with Rosetta InterfaceAnalyzer, which is not used for proposal generation, MCTS selection, archive ranking, or any other component of MCTH.

We evaluate the six CaoData targets shared by MCTH and RFDiffusion and the five with available BindCraft designs. We report interface separation energy $\Delta G _ { \mathrm { s e p } } ,$ , the same energy normalized by buried interface area, packing quality (packstat), and buried unsatisfied hydrogen bonds.

Raw $\Delta G _ { \mathrm { s e p } }$ is sensitive to interface size, so we treat the area-normalized energy as the more interpretable comparison. Under this measure, MCTH is essentially matched with RFDiffusion (−3.12 vs. −3.13 per $1 0 0 \mathring { \mathrm { A } } ^ { 2 } )$ , while BindCraft is somewhat more favorable (−3.40). Packing and buried-unsatisfied-hydrogen-bond statistics are of comparable order across the three methods, although MCTH does not outperform the strongest baseline on packing quality.

We therefore interpret this analysis conservatively: MCTH produces interfaces with competitive physical plausibility under an evaluation that is entirely outside the search loop, but these Rosetta scores do not establish superior binding energetics or experimental affinity.

<table><tr><td>Method</td><td>n</td><td> $\Delta G _ { \mathrm { s e p } }$ </td><td> $\Delta G / 1 0 0 \mathring { \mathrm { A } } ^ { 2 }$ </td><td>Packstat</td><td>∆unsat H-bonds</td></tr><tr><td>MCTH</td><td>6</td><td>-71.2</td><td>-3.12</td><td>0.577</td><td>5.7</td></tr><tr><td>BindCraft</td><td>5</td><td>-65.7</td><td>-3.40</td><td>0.600</td><td>5.2</td></tr><tr><td>RFDiffusion</td><td>6</td><td>-54.2</td><td>-3.13</td><td>0.602</td><td>6.5</td></tr></table>

Table 22: Oracle-independent Rosetta InterfaceAnalyzer evaluation of protein-binder designs. More negative interface energy and higher packstat are favorable; fewer buried unsatisfied hydrogen bonds are preferred.

<table><tr><td>Modality</td><td>Targets</td><td>Mean folding calls</td><td>Mean wall-clock</td></tr><tr><td>Protein binder</td><td>12</td><td>98.1</td><td>~9.0 h</td></tr><tr><td>RNA aptamer</td><td>7</td><td>52.4</td><td>~0.8 h</td></tr><tr><td>DNA aptamer</td><td>8</td><td>66.1</td><td>~1.1 h</td></tr></table>

Table 23: Realized compute for the main MCTH experiments. DNA reports the eight targets used in the non-degenerate Boltz-2 comparison. Wall-clock values are means from the corresponding run logs.

## D.8 Compute and Search-Trajectory Diagnostics

MCTH explicitly trades additional structure-prediction inference for adaptive optimization. We therefore report both the realized folding-call budget and wall-clock cost, and examine how search behavior changes over the course of optimization.

Realized compute. The nominal maximum budget is 100 folding calls, but early stopping terminates a search after 10 iterations without improvement. Consequently, the realized cost differs substantially across modalities.

Protein-binder design nearly exhausts the full budget, whereas RNA and DNA aptamer searches terminate substantially earlier. This is consistent with the different optimization landscapes: the short nucleic-acid tasks frequently reach high search-time confidence early, while de novo protein-binder design begins from a much larger sequence space and continues to improve later in the budget.

Performance versus compute. For protein binders, where nearly all searches reach the full budget, the held-forward mean best ipTM improves steadily from 0.614 after the first folding evaluation to 0.724, 0.768, 0.803, 0.830, and 0.864 after 5, 10, 25, 50, and 100 calls, respectively.

The continued improvement at later checkpoints complements Appendix D.4: the protein-binder gain is not produced solely by a strong initial proposal followed by stagnant computation.

Coverage versus allocation. We additionally compare the number and diversity of evaluated sequence states with the best-of-N control. MCTH does not consistently visit more unique states than independent sampling under the same evaluation budget. Instead, the principal difference is where evaluations are allocated within the evolving search tree.

Protein-binder searches tend to concentrate evaluations around promising branches early and broaden exploration later when progress slows. RNA and DNA searches show the complementary pattern: they explore broadly during the initial iterations and subsequently concentrate around an elite neighborhood once high-confidence candidates emerge. These trajectories are consistent with adaptive exploration– exploitation rather than uniformly greater state-space coverage.

Combined with the matched-budget ablation and convergence diagnostic, this suggests that the primary advantage of MCTH is adaptive compute allocation: the tree structure determines which sequence–structure trajectories receive additional expensive folding evaluations, rather than simply increasing the number of generated candidates.

<table><tr><td>Fold call</td><td>1</td><td>5</td><td>10</td><td>25</td><td>50</td><td>100</td></tr><tr><td>Protein binder</td><td>0.614</td><td>0.724</td><td>0.768</td><td>0.803</td><td>0.830</td><td>0.864</td></tr></table>

Table 24: Protein-binder performance as a function of folding-call budget. Values are mean heldforward best ipTM across targets.

## E Extended Preliminaries

## E.1 Biomolecular representation

Biomolecular Representation. We consider a general biomolecular system composed of one or more interacting components, including proteins, cyclic peptides, nucleic acids (DNA/RNA), small molecules, ions, and post-translational modifications. A biomolecule is represented by a sequence S and an associated all-atom structure X. Let $S = ( s _ { 1 } , \ldots , s _ { n } )$ denote a sequence of discrete residue or atom types, where each $s _ { i }$ may correspond to an amino acid, nucleotide, small-molecule atom, or a special token (e.g., unknown or masked). Let

$$
X = \{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { m } \}
$$

denote the set of 3D atomic coordinates in Cartesian space. For multimolecular complexes, we represent the full system as

$$
\boldsymbol { B } = \{ ( \boldsymbol { S } ^ { ( k ) } , \boldsymbol { X } ^ { ( k ) } ) \} _ { k = 1 } ^ { K } ,
$$

where k indexes the biomolecular components. Interactions between components are implicitly encoded through spatial proximity and pairwise relationships modeled by the structure prediction network.

## E.2 Forward folding (diffusion) and inverse folding

All-Atom Structure Prediction via Diffusion Models. Let $f _ { \theta }$ denote a pretrained all-atom structure prediction model parameterized by θ. Given an input sequence S and optional conditioning information C (e.g., binding partner, fixed motifs, or partial structure), the model predicts an all-atom structure according to

$$
X \sim p _ { \theta } ( X \mid S , C ) .
$$

These models typically employ a diffusion-based denoising process, starting from Gaussian noise and progressively refining atomic coordinates. When the input sequence is underspecified—such as an all-X sequence or a partially defined complex—the model often produces hallucinated yet physically plausible structures guided by learned structural priors. In this work, we explicitly exploit this hallucination behavior as a constructive signal for biomolecular design.

Sequence Design and Inverse Folding. Given a fixed or partially specified structure X, inverse folding models aim to generate sequences compatible with that structure:

$$
S \sim q _ { \phi } ( S \mid X , C ) ,
$$

where $q _ { \phi }$ denotes a pretrained sequence design model. Sequence redesign improves foldability and interaction consistency while preserving structural constraints. Within our framework, inverse folding serves as a primitive action that updates sequence hypotheses conditioned on hallucinated or refined structures.

Forward and Inverse Folding as Dual Operators. We treat forward folding and inverse folding as dual operators acting on the biomolecular state:

• Forward folding: $S \to X$ via all-atom structure prediction.

• Inverse folding: $X  S$ via sequence redesign.

Rather than enforcing a fixed alternation, our framework allows these operators to be applied adaptively and repeatedly within a search process, enabling flexible traversal of the joint sequence– structure landscape.

## E.3 Confidence/value signals

Confidence and Value Signals. Modern structure prediction models provide intrinsic confidence metrics that correlate with structural correctness and interaction quality, including predicted local accuracy, global confidence, and interface-specific measures. We collectively denote these signals as

$$
V ( X , S ) \in \mathbb { R } ,
$$

which serve as value estimates for evaluating candidate biomolecular designs. These values are treated as learned consistency scores rather than explicit physical energies.

## E.4 MDP notation and value functions

We briefly summarize standard notation for finite-horizon Markov decision processes [34]. A policy $\pi ( a \mid s )$ specifies a distribution over actions at state s. The action-value function $Q ^ { \pi } ( s , a )$ is the expected cumulative reward obtained by taking a in s and then following $\pi _ { \mathrm { : } }$ and $\dot { V ^ { \pi } } ( s )$ is the corresponding state-value function. In our design setting, we primarily use this formalism to motivate MCTS-style planning and compute allocation: evaluations are obtained from model-derived scores $V ( s )$ (confidence and feasibility), rather than long rollouts.

## E.5 Diffusion models and guidance

We include a short diffusion primer to clarify our use of guided hallucination. Let $y _ { 0 : T } = ( y _ { 0 } , \dots , y _ { T } )$ denote a latent trajectory, where $y _ { 0 }$ is a clean sample $( \mathrm { e . g . }$ , structure coordinates or discrete tokens) and $y _ { T }$ is maximally noisy. A forward noising process $q ( y _ { t } \mid y _ { t - 1 } )$ progressively corrupts $y _ { 0 }$ into $y _ { T }$ $\mathbf { A }$ learned reverse model $p _ { \phi } ( y _ { t - 1 } \mid y _ { t } )$ generates samples by iteratively denoising from $y _ { T }$ to $y _ { 0 }$

Guidance via reweighting. When an auxiliary score $f ( \cdot )$ is available (property predictor, constraint critic, feasibility score), a common guidance view reweights reverse transitions:

$$
\begin{array} { r } { p _ { \phi } ^ { \mathrm { g u i d e } } ( y _ { t - 1 } \mid y _ { t } ) \propto p _ { \phi } ( y _ { t - 1 } \mid y _ { t } ) \exp \{ \beta f ( y _ { t - 1 } ) \} , } \end{array}\tag{18}
$$

where $\beta$ controls guidance strength. This formulation covers both continuous and discrete variants (with appropriate definitions of $p _ { \phi } )$ .

How we use this perspective. Our method is inference-only: we do not backpropagate through the predictor. Instead, we implement guidance by proposal resampling: for a given parent state (or diffusion step), we sample a small candidate set $\{ y _ { t - 1 , j } ^ { \prime } \} _ { j = 1 } ^ { J } ( \mathbf { e . g . }$ , by changing random seeds, experts, or noise schedules), and then select or resample according to weights proportional to $\exp \{ \beta f ( y _ { t - 1 , j } ^ { \prime } ) \}$ . In our biomolecular setting, $f ( \cdot )$ is instantiated by the same task-aware value signal used $ { \mathbf { b } }  { \mathbf { y } }$ planning, combining intrinsic confidence metrics (pLDDT/pTM/ipTM) and optional feasibility penalties (clash/geometry/energy-like terms).

Multi-expert proposals. When multiple experts are available $( \mathrm { e . g . }$ ., multiple folding/hallucination experts and/or inverse-folding experts), we can form a mixture proposal $\begin{array} { r } { p _ { \mathrm { m i x } } ( \cdot ) = \sum _ { e = 1 } ^ { E } \omega _ { e } p ^ { ( e ) } ( \cdot ) } \end{array}$ and apply the same resampling principle on samples drawn from the mixture. This is the mechanism by which “mixture-of-experts hallucination” interfaces with MCTS: experts provide diverse proposals, while planning allocates compute and selects among them.

## E.5.1 Monte Carlo Tree Search details

We summarize a few implementation-level details used in the paper.

Backup rules. After evaluating a newly expanded child state $s ^ { \prime } ,$ its value $V ( s ^ { \prime } )$ is propagated along the selected path. We maintain visit counts $N ( s )$ and edge counts $N ( s , a )$ , and update $Q ( s , a )$ by a standard backup. Common choices include: (i) mean backup, $Q \Leftarrow \frac { 1 } { N } \sum V$ , which targets average performance; and (ii) max backup, $Q $ max $V _ { ; }$ , which targets best-of-N outcomes. Our experiments primarily report best-of-N designs, so max backup is a natural variant when optimizing for extreme-value quality.

Evaluation cost and “rollouts”. In biomolecular design, the dominant cost is a folding/refolding forward pass. Accordingly, we often treat expansion+evaluation as a single expensive step: each new child is instantiated by calling the folding expert and then scored by $\check { V ( s ^ { \prime } ) }$ . This keeps the planner modular and makes compute accounting explicit via a fold-call budget.

MCTS with guided proposals. Guided sampling (Appendix E.5) can be used inside expansion: for a fixed parent (S, X), we generate multiple candidate proposals (different experts/seeds/noise schedules), reweight them by $\mathbf { \bar { e } } \mathbf { x p } \{ \beta V ( \cdot ) \}$ (optionally with feasibility penalties), and then add the selected children to the tree. MCTS then performs global credit assignment across these locally guided proposals, balancing exploration and exploitation across distinct design hypotheses.

Design selection. After exhausting the rollout/fold-call budget, we return designs from high-value leaf nodes, optionally re-ranking by task-specific metrics (e.g., interface confidence for binding tasks, RMSD/constraint satisfaction for scaffolding tasks).

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

## Answer: [Yes]

Justification: The contributions stated in the abstract and Section 1—uncertainty-aware MCTS over hallucinated sequence–structure states, multi-expert folding/inverse-folding operators, biophysics-informed co-hallucination, and modality-agnostic fine-tuning-free design—are substantiated by the method in Section 3 and the experiments in Section 4, which evaluate protein–RNA, protein–DNA, and protein–protein design tasks, with additional multimodal results provided in the appendix. The paper also explicitly limits its claims to computational/refold-based evaluation in Section 5.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss limitations in Section 5, including that the evaluation is computational and refold-based rather than experimental. Appendix D.2 further discusses limitations of using refolding-based dry-lab metrics and cross-oracle comparisons. We also discuss method-specific failure modes, such as symmetric complexes, in Section 4.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate ”Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A].

Justification: The paper does not present new theoretical theorems or formal proofs. The equations define the proposed MCTS-based inference procedure, value functions, uncertainty terms, and biophysical feasibility terms, with full methodological details provided in Appendix B.1 and Appendix B.6.

Guidelines:

• The answer NA means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The experimental protocols are described in Section 4. We provide the foldingcall budget, search hyperparameters, seeds, folding and inverse-folding backends, and compute configuration in Appendix B.6. Full CU-PUCT details are given in Appendix B.1, and additional ablations and convergence diagnostics are provided in Appendix D.3 and Appendix D.4. The standardized refolding protocol is described in Appendix D.2. Experiment details are listed in Section 4 and Appendix C, B.6.

Guidelines:

• The answer NA means that the paper does not include experiments.

• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: An anonymized code release is not bundled with this submission. The paper provides all algorithmic details, hyperparameters, network architecture, baseline adaptations.and pseudocode in the appendix to enable reproduction. We will release the code and configuration files publicly upon acceptance.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/pu blic/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The paper is an inference-time method and does not train or fine-tune new models. Section 4 specifies the benchmark settings, baselines, evaluation metrics, and refolding protocol. Appendix B.6 reports search hyperparameters, seeds, backend models, sampling settings, and compute. Appendix B.1 gives the full selection rule and backup details.

Guidelines:

• The answer NA means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No].

Justification: The main benchmark tables are reported as fixed-budget best-per-target refold scores and therefore do not include error bars for all results. However, for the key mechanism-level claim that search-guided compute allocation improves over simpler alternatives, Appendix D.3 reports mean ± standard deviation over three seeds, and Appendix ?? provides convergence diagnostics. We view broader multi-seed evaluation of all benchmark tables as future work.

## Guidelines:

• The answer NA means that the paper does not include experiments.

• The authors should answer ”Yes” if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Appendix B.6 reports the fold-call budget, per-target/seed compute setup, backend models, and GPU configuration used for the experiments. The main experiments use a fixed 100-fold-call budget unless otherwise stated, and matched-budget ablations use the same budget to isolate planning effects.

Guidelines:

• The answer NA means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The work is computational biomolecular design research using public structural data, pretrained models, and dry-lab evaluation. We do not use private, human-subject, or personally identifiable data. The submission is anonymized, and limitations and potential risks of computational-only evaluation are discussed in the paper.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The introduction motivates positive applications in therapeutics, biosensing, synthetic biology, and molecular recognition. The conclusion and limitations clarify that the reported results are computational and require experimental validation before deployment. Because biomolecular design methods may have dual-use implications if misapplied, we discuss the need for validation, controlled use, and careful interpretation of generated designs.

## Guidelines:

• The answer NA means that there is no societal impact of the work performed.

• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper does not release a new pretrained generative model or scraped dataset. The method is an inference-time planning framework built on existing biomolecular modeling tools. Generated designs are evaluated only computationally and are not presented as experimentally validated biological agents. If code is released, it will be anonymized for review and accompanied by documentation indicating that outputs require expert review and biosafety screening before any experimental use.

Guidelines:

• The answer NA means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [N/A]

Justification: We cite the original sources for the pretrained models, baselines, and datasets used in the paper, including Boltz-2, ProteinMPNN, NA-MPNN, RFDiffusion, ODesign, BindCraft, and PDB-derived benchmark structures. The supplementary material documents the software dependencies and data sources used to reproduce the experiments. We do not claim ownership of these existing assets.

## Guidelines:

• The answer NA means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: The paper does not release a new dataset, pretrained model, or benchmark as a standalone asset at submission time.

Guidelines:

• The answer NA means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or research with human subjects. Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve human subjects, human data, or crowdsourcing, so IRB approval is not applicable.

## Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
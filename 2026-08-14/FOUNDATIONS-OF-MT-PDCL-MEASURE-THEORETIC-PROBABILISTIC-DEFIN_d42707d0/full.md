# FOUNDATIONS OF MT-PDCL: MEASURE-THEORETIC PROBABILISTIC DEFINITE CLAUSE LOGIC

UNDER CONSIDERATION FOR PUBLICATION IN THEORY AND PRACTICE OF LOGIC PROGRAMMING (TPLP)

Costin Badic˘ a˘ Department of Computers and Information Technology University of Craiova Craiova, Romania costin.badica@edu.ucv.ro

Amelia Badic˘ a˘ Department of Business Informatics University of Craiova Craiova, Romania amelia.badica@edu.ucv.ro

July 2026

## ABSTRACT

Standard probabilistic logic programming frameworks typically rely on grounding logic programs into discrete propositional representations. This operational requirement restricts exact inference to finite domains and discrete probability distributions. In this paper, we introduce Measure-Theoretic Probabilistic Definite Clause Logic (MT-PDCL), a generalized foundational framework that eliminates this finite-domain restriction. By explicitly defining stochastic variables over bounded index domains and equipping the interpretation space with standard Borel σ-algebras, MT-PDCL allows logical variables to operate natively over continuous measurable spaces. Building on Continuous Distribution Semantics, MT-PDCL models probabilistic rules as mutually independent causal events. However, rather than aggregating these derivations via finite boolean circuits, declarative entailment is formally defined through exact Lebesgue integration over the continuous measure space. We introduce a continuous immediate consequence operator that unifies the integration of continuous prior distributions with the evaluation of exact continuous observations. We demonstrate that this approach replaces the combinatorial bottleneck of discrete grounding with exact, algebraic, and structurally differentiable inference. While this transition trades discrete combinatorics for the geometric curse of dimensionality, it achieves the expressive power of continuous probabilistic models while preserving the pure declarative syntax of definite clause logic.

## 1 Introduction

The foundational approach of assigning probabilities to logical sentences via probability measures over possible worlds was established by Nilsson [Nil86]. Building on this, probabilistic logic programming languages such as PRISM [SK97] and ProbLog [DRKT07] introduced distribution semantics to evaluate the probability of logical queries. This approach assumes all base probabilistic facts are mutually independent, thereby defining a single, explicit probability distribution over all possible worlds. However, calculating exact probabilities across these worlds creates a significant operational bottleneck. To evaluate a derived query, the system must sum the probabilities of the specific discrete states where the query holds true. This operation forces the system to ground the first-order logic programs into discrete propositional representations, such as Binary Decision Diagrams (BDDs) [Bry86] or Sentential Decision Diagrams (SDDs) [Dar11]. This reliance on propositionalization inherently limits exact inference to finite domains and discrete probability distributions.

Recent advancements in neural-symbolic artificial intelligence have introduced frameworks that integrate differentiable components with logic programming, such as DeepProbLog [MDK<sup>+</sup>21] and TensorLog [CYM20]. While these systems successfully process continuous representations, such as neural network parameters or feature tensors, their underlying logical semantics typically remain tied to finite domains. For instance, DeepProbLog evaluates neural network outputs as probabilistic facts, but its logical inference engine still relies on discrete propositionalization.

There is a critical need for a probabilistic logic framework that allows logical variables to operate natively over continuous spaces, completely bypassing the need for finite boolean grounding, while remaining fully differentiable for contemporary deep learning integration.

To address this gap, we introduce Measure-Theoretic Probabilistic Definite Clause Logic (MT-PDCL). This paper makes the following primary contributions:

• Continuous Declarative Semantics: Equipping the interpretation space and explicitly bounded stochastic index domains with standard Borel σ-algebras [Kec95] removes the finite-domain restriction. Logical variables act as deterministic placeholders that natively evaluate over measurable continuous spaces (e.g., bounded subsets of R<sup>n</sup>).

• Continuous Distribution Semantics: Extending classical distribution semantics to infinite domains, MT-PDCL defines possible worlds via independent continuous variables and probabilistic causal events. Declarative entailment is formally defined as the exact Lebesgue integral of the logically valid possible worlds, evaluated over the geometric boundaries defined by the logic program.

• Continuous Consequence Operator: We generalize the immediate consequence operator $( \mathbb { T } \kappa )$ to operate algebraically over continuous probability densities. It utilizes exact integration for continuous marginalization and a continuous Noisy-OR formulation to aggregate disjunctive derivations, mathematically bypassing discrete grounding.

• Differentiable Neural-Symbolic Foundation: We formally prove that this continuous operator converges to a well-defined least fixed point and is everywhere differentiable, providing a structurally transparent, nonnegative gradient signal that ensures stable routing for neural-symbolic deep learning.

The remainder of this paper is structured as follows. Section 2 defines the formal syntax of MT-PDCL. Section 3 establishes the exact measure-theoretic semantics, building from continuous variable domains and index bounds to formal declarative entailment and continuous query semantics. Section 4 provides concrete illustrative examples of this declarative entailment. Section 5 introduces the continuous immediate consequence operator and proves its theoretical properties, including fixed-point convergence, operational completeness bounds, and neural-symbolic differentiability. Section 6 contextualizes our approach against existing probabilistic and neural-symbolic frameworks. Section 7 evaluates the dynamic consequence operator through concrete continuous execution traces, comparing these examples against existing approaches to highlight specific operational limitations. Section 8 outlines the theoretical boundaries of analytical computability and tractability, and Section 9 concludes.

## 2 Syntax of MT-PDCL

To formalize MT-PDCL, we define the recursive syntax of its terms, predicates, and rules, ensuring a strict separation between logical reasoning and continuous probability generation.

Let ${ \mathcal { L } } ( { \mathcal { C } } , { \mathcal { R } } )$ be a first-order language where C is a finite set of constant symbols and R is a finite set of predicate symbols. To distinguish user-defined logical relations from standard mathematical operators and continuous probability distributions, R is partitioned into three disjoint subsets: $\mathcal { R } = \mathcal { R } _ { u } \cup \mathcal { O } \cup \mathcal { R } _ { s }$

$\mathcal { R } _ { u }$ contains the user-defined logical relation symbols.

• O contains the built-in base predicate symbols representing standard algebraic and set-theoretic operators $( \mathrm { e . g . , < , } \leq , \mathrm { = , } \in )$

• $\mathcal { R } _ { s }$ contains the stochastic predicate symbols, which act as the interface to the continuous environment.

Following standard logic programming semantics, variables are scoped locally to their clauses. We assume the program is standardized apart, meaning all variables are renamed to be unique across different clauses. Let V represent this global finite set of uniquely named logical variables. In MT-PDCL, these variables are deterministic placeholders used for standard unification; they are not random variables themselves.

Terms: A term $t \in \tau$ is either a logical variable $X \in \mathcal { V }$ or a constant symbol $c \in { \mathcal { C } }$ . To support mixed continuous and discrete domains natively, the set of constants C includes explicit numeric values (e.g., 30.0, 4.2), categorical symbols (e.g., severe\_hot, critical), as well as mathematical intervals (e.g., [0, 10]) and finite sets (e.g., {1, 2, 3}). This allows built-in operators to act directly on geometric boundaries and discrete collections. Thus, the set of all terms is ${ \mathcal { T } } = { \mathcal { C } } \cup { \mathcal { V } } .$

Atomic Formulas (User-Defined Predicates): An atomic formula is constructed using a user-defined relation symbol $r \in \mathcal { R } _ { u }$ of arity n, applied to a tuple of terms: $r ( t _ { 1 } , \ldots , t _ { n } )$

Base Predicates (Built-in Operators): A base predicate is constructed using a built-in mathematical operator $o \in \mathcal { O }$ applied to a tuple of terms, typically written in standard infix notation $( \mathrm { e . g . , } t _ { 1 } < t _ { 2 } , t _ { 1 } \leq t _ { 2 } , t _ { 1 } = t _ { 2 } , \mathrm { o r } t \in [ 2 , 5 ] )$ .

Stochastic Predicates: A stochastic predicate $r \in \mathcal { R } _ { s }$ queries the continuous environment. To ensure mathematically valid integration, its arguments are partitioned into deterministic indices (inputs) and generative variables (outputs) using a semicolon: $r ( \bar { I } _ { 1 } , \ldots , I _ { k } ; X _ { 1 } ^ { \bullet } , \ldots , X _ { m } )$ . The indices I must be bound to ground terms prior to evaluation, serving as the unique identity key for the independent random draw. The variables $\mathbf { \breve { X } }$ must remain unbound until the measure space generates the continuous constants.

Rule Bodies (B): A rule body B is a conjunction of atomic formulas, base predicates, and stochastic predicates, defined recursively:

• Any atomic formula, base predicate, or stochastic predicate is a valid body B.

• If $B _ { 1 }$ and $B _ { 2 }$ are valid bodies, their conjunction $B _ { 1 } \land B _ { 2 }$ is a valid body.

Following standard definite clause logic, a rule consists of one positive literal in the head and a conjunction of positive literals in the body. Consequently, negation (¬) is prohibited in both the head and the body, and disjunction $( \vee )$ is prohibited in the head. Furthermore, stochastic predicates from $\mathcal { R } _ { s }$ and built-in operators from O can only appear in the rule body; they are prohibited from appearing in the head.

Clauses (Φ): An MT-PDCL logic program Φ is a finite set of clauses. Let H be an atomic formula (the rule head) constructed from $\mathcal { R } _ { u } .$ , and B be a valid rule body. A probabilistic clause takes one of the following exact forms, where $p \in [ 0 , 1 ]$ is a probability parameter:

• Probabilistic Rule: $p \mathrel { \mathop : } H \gets B$

• Probabilistic Fact: $p : : H$

Standard deterministic rules $( H  B )$ and facts (H) are supported as syntactic sugar for probabilistic clauses where the parameter is defined as $p = 1 . 0 .$

The Knowledge Base (K): To enforce the separation between logical reasoning and continuous probability, an MT-PDCL Knowledge Base is defined as a tuple $\dot { \mathcal { K } } = \langle \mathcal { M } _ { s } , \Phi \rangle$

• Φ is the logic program containing the definite clauses.

$\mathcal { M } _ { s }$ is the Stochastic Measure Mapping. Mathematically, it is a function that maps a stochastic predicate $r \in \mathcal { R } _ { s }$ and a ground index tuple i to a specific probability measure D over the domain. Because measure theory natively unifies continuous and discrete probabilities, D represents either a continuous probability density (e.g., Normal, Uniform) or a discrete probability mass function $( \mathrm { e . g . , C }$ ategorical, Bernoulli, Poisson). Syntactically, this mapping is declared in the program using the form $\tilde { r } ( \tilde { \mathbf { I } ; \mathbf { X } } ) \sim \mathbf { \bar { \mathcal { D } } }$ . To mathematically bind the domain of continuous indices, this declaration can optionally include an explicit geometric bound, written as $r ( \mathbf { I } ; \mathbf { X } ) \sim \mathcal { D }$ over $\mathbf { I } \in \Omega _ { I }$

Queries (Q): A query Q is a goal clause consisting of a finite conjunction of literals. Syntactically, it is written as a headless clause:

$$
 L _ { 1 } \land L _ { 2 } \land \cdots \land L _ { k }
$$

where each literal $L _ { i }$ can be a user-defined atomic formula, a built-in base predicate, or a stochastic predicate. The conjunction forming the query is therefore structurally identical to a valid rule body B. Let $\mathbf { V } = \langle { \bar { V } } _ { 1 } , \dots , V _ { n } \rangle$ be the tuple of all free logical variables occurring in Q. In standard logical terms, these free variables are implicitly existentially quantified.

## 3 Continuous Distribution Semantics

## 3.1 Interpretations over Continuous Domains

To rigorously define probability measures and integration, the universal domain of discourse D cannot be an arbitrary set. We require D to be a measurable space $( D , { \mathcal { F } } _ { D } )$ , where $\mathcal { F } _ { D }$ is a standard Borel σ-algebra over $D .$ . To natively support both continuous parameters and discrete categorical properties, D is constructed as the disjoint union of a continuous space $( \mathbf { e } . \mathbf { g } . , \mathbb { R } ^ { \bar { n } } )$ and a countable discrete set of semantic values (denoted $D _ { d i s c r e t e } ,$ containing categorica strings and natural numbers N). This construction satisfies the mathematical requirements for $D ,$ as the disjoint union of a standard Borel space and a countable discrete set remains a standard Borel space [Kec95].

An interpretation I of $\mathcal { L }$ in the measurable domain D is defined by two mappings:

• An interpretation function for constant symbols $\phi _ { \mathbb { Z } } : { \mathcal { C } } \to D$ , where $\phi _ { \mathcal { T } } ( c )$ is the interpretation of the constant $c \in { \mathcal { C } } .$ . To natively support continuous logic, MT-PDCL permits numeric values (e.g., 120.5) to act directly as constants in the syntax, for which $\phi _ { \mathcal { I } }$ acts as the identity function.

• An interpretation function $\pi _ { \mathcal { T } } ( r ) : D ^ { n }  \{ 1 , 0 \}$ for any predicate symbol $r \in \mathcal { R } _ { u } \cup \mathcal { O }$ of arity n. For a user-defined relation $r \in \mathcal { R } _ { u } , \pi _ { \mathcal { T } } ( r )$ acts as an indicator function. For a built-in base predicate $r \in \mathcal { O }$ (such as $< , \leq , \in ) , \pi _ { \mathcal { T } } ( r )$ is universally fixed across all interpretations to its standard mathematical definition.

This mapping establishes the truth values for ground atomic formulas. Classical logical satisfaction for a specific interpretation $\mathcal { T }$ evaluating a ground atom is defined deterministically as:

$$
{ \mathcal { T } } \left| = r ( c _ { 1 } , \ldots , c _ { n } ) { \mathrm { ~ i f ~ a n d ~ o n l y ~ i f ~ } } \pi _ { \mathbb { Z } } ( r ) ( \phi _ { \mathbb { Z } } ( c _ { 1 } ) , \ldots , \phi _ { \mathbb { Z } } ( c _ { n } ) ) = 1 \right.
$$

Logic programs do not explicitly enumerate all facts; instead, they generate them dynamically using universally quantified rules and variables. This allows the logical satisfaction of complex formulas to build compositionally upon these base atoms. Furthermore, variables in MT-PDCL evaluate natively over continuous domains without discrete grounding. The subsequent subsections detail the exact mechanics for resolving these rules, bindings, and queries within a possible world.

To compute valid probabilities, we impose a strict topological requirement on these interpretations: the subset of $D ^ { n }$ for which $\pi _ { \mathcal { Z } } ( r )$ evaluates to 1 must be a valid, measurable set within the product $\sigma { \mathrm { - a l g e b r a ~ } } \mathcal { F } _ { D } ^ { \otimes n }$ . This ensures that geometric regions defined by logical rules inherently possess a well-defined mathematical volume.

## 3.2 The Stochastic Measure Space

In standard probabilistic logic programming, it is often erroneously implied that logical variables themselves hold probabilities. In MT-PDCL, we enforce a strict separation. Logical variables in $\nu$ are deterministic, syntactic placeholders used for unification. All probabilistic uncertainty including both continuous and discrete is isolated within the stochastic measure space, generated by the mapping $\mathcal { M } _ { s }$

Recall that a stochastic predicate $r \in \mathcal { R } _ { s }$ is partitioned into deterministic indices I and stochastic outputs X of arity m. Let $\mathcal { T } _ { r }$ be the set of all valid ground instantiations of the index tuple I over the domain $D ,$ and let $\bar { \Delta } ( D ^ { m } )$ be the space of all valid probability measures over the m-dimensional output space.

The mapping $\mathcal { M } _ { s } : \mathcal { R } _ { s } \times \mathcal { T } _ { r }  \Delta ( D ^ { m } )$ formally assigns a probability measure to every unique ground index of a stochastic predicate. For every stochastic predicate r and every unique ground index $\mathbf { i } \in \mathcal { T } _ { r } .$ the output of the mapping $\mathcal { M } _ { s } ( r , \mathbf { i } )$ defines a mathematically independent random variable (or tuple of m variables). While we primarily refer to these as continuous variables, measure theory treats discrete categorical variables as a direct special case using probability measures concentrated on specific points (Dirac measures). Thus, a stochastic predicate can natively generate either continuous coordinates or discrete categorical states distributed according to its assigned measure.

While the set of ground indices $\mathcal { T } _ { r }$ may be countably or uncountably infinite, the mapping $\mathcal { M } _ { s }$ is defined compactly in the knowledge base using parameterized probability distributions. The parameters of the assigned distribution $\mathcal { D }$ can be deterministic functions of the index tuple I, denoted syntactically as $\dot { r ( \mathbf { I } ; \mathbf { X } ) } \sim \mathcal { D } ( \theta ( \mathbf { I } ) )$ . For example, a continuous sensor reading whose mean drifts over time $T$ can be compactly declared as reading $( T ; X ) \sim \operatorname { N o r m a l } ( \mu = 0 . 5 { \cdot } T , \sigma =$ 1.0). Similarly, a discrete weather generator can be declared as weather $( T ; W ) \sim$ Categorical({sunny : 0.7, rain : 0.3}). Likewise, a hybrid mapping where a continuous index directly parameterizes a discrete countable distribution can be declared as fault\_count $( T ; K ) \sim \operatorname { P o i s s o n } ( \lambda = 2 . 0 \cdot T )$ . Mathematically, this allows a single finite syntactic declaration to define an infinite family of distributions.

We define the stochastic state space $\Omega _ { s }$ as the Cartesian product of the domains for all these independent random variables, across all stochastic predicates $r \in \mathcal { R } _ { s }$ and all their ground indices $\mathbf { i } \in \mathcal { I } _ { r }$ . The set of ground indices (e.g., time steps $t \in \mathbb { N } )$ can be countably infinite, which generally makes $\Omega _ { s }$ an infinite-dimensional product space. The Kolmogorov Extension Theorem ensures that this infinite product of standard Borel spaces, equipped with independent probability measures, forms a valid measurable space $( \Omega _ { s } ^ { - } , \mathcal { F } _ { \Omega _ { s } } )$ with a well-defined joint probability measure $\mu _ { s }$

A specific continuous state $\sigma \in \Omega _ { i }$ rigidly dictates the outcome of every stochastic fact in the environment. It maps every grounded index tuple i of every stochastic predicate r to a specific, constant element $x \in D$

During logical evaluation, when the engine encounters a stochastic predicate $r ( \mathbf { i } ; \mathbf { X } )$ , the deterministic index i acts as a lookup key. The state σ provides the concrete constant generated for that key, and standard deterministic unification simply binds the empty logical variables X to that constant. This mechanism completely bypasses the measure-zero paradox while effortlessly supporting unbounded recursion, as each recursive step provides a distinct index (e.g., $\bar { T } = 1 , T = 2 )$ mapping to a distinct, independent continuous dimension in $\Omega _ { s }$

## 3.3 Defining the Index Domain (I<sub>r</sub>)

When a logic program evaluates a rule such as $p ( \mathbf { X } ) \gets r ( \mathbf { I } ; \mathbf { X } )$ where the index I is unbound by any other predicate in the body, the continuous consequence operator must aggregate over the entire domain of I. To mathematically evaluate this integral, the framework must systematically establish the valid index domain $\mathcal { T } _ { r }$ and its associated base measure.

MT-PDCL establishes the domain $\mathcal { T } _ { r }$ through two mechanisms within the stochastic measure mapping $\mathcal { M } _ { s }$

1. Implicit Derivation via Parameter Support: If the stochastic declaration maps the indices directly to the parameters of a standard probability distribution, $\mathcal { T } _ { r }$ is implicitly derived as the maximum valid mathematical support of those parameters. For example, given the declaration:

$$
r ( M , S ; X ) \sim \operatorname { N o r m a l } ( \mu = M , \sigma = S )
$$

The standard deviation of a Normal distribution must be positive, allowing the framework to implicitly derive the valid index domain as the continuous space $\mathcal { T } _ { r } = \mathbb { R } \times ( 0 , \infty )$ . The operator aggregates over this space using the standard Lebesgue measure (dM dS).

2. Explicit Definition via Bounded Fields: When indices represent physical dimensions (such as spatial coordinates or time intervals) rather than statistical parameters, integrating over an infinite implicit domain is physically invalid. MT-PDCL allows the explicit definition of bounded index spaces directly in the declaration syntax. To maintain syntactic uniformity, these geometric bounds are defined using conjunctions of the language’s built-in base predicates (O), effectively defining a continuous random field:

$$
r ( \mathbf { I } ; \mathbf { X } ) \sim { \mathcal { D } } ( \theta ( \mathbf { I } ) ) \quad { \mathbf { o v e r } } \quad \Omega _ { I } ( \mathbf { I } )
$$

For example, a sensor array distributed across a 10-kilometer road can define its boundary using standard arithmetic inequalities:

$$
\operatorname { s e n s o r } ( I ; X ) \sim \operatorname { N o r m a l } ( \mu = 0 , \sigma = I \cdot 0 . 1 ) \quad \mathbf { o v e r } \quad I \geq 0 \land I \leq 1 0
$$

By explicitly anchoring the domain with these base predicates, an unbound query to sensor $( I ; X )$ provides the consequence operator with the precise finite limits required to compute the continuous integrals. The specific computational methods needed to calculate these bounded integrals are deferred to practical implementations of the framework.

## 3.4 Independent Causal Events

The second source of uncertainty arises from the probabilistic rules themselves. We model every probability parameter $p _ { j }$ attached to a rule in $\Phi$ as an independent causal event $E _ { j }$

This event represents the physical or logical mechanism that allows the rule to successfully fire. It is statistically independent of all other clauses and all continuous variables in $\Omega _ { s }$ , and it succeeds with an exact probability $P ( E _ { j } ) \dot { = }$ $p _ { j }$

Let $\Omega _ { E }$ be the set of all possible outcomes for these independent causal events. A specific causal choice $\epsilon \in \Omega _ { E }$ assigns a strict True or False value to every event $E _ { j }$ . Mutual independence among these causal events allows the total probability measure $\mu _ { E }$ over the choice space $\bar { \Omega } _ { E }$ to be defined simply as the standard product measure of the individual event probabilities.

By defining every causal event $E _ { j }$ as independent, MT-PDCL replaces complex global constraint resolution with a generative approach. If a user needs to model correlated rule outcomes, they must explicitly define a shared continuous variable in $\mathcal { M } _ { s }$ and place it in the body of both rules. This independence assumption guarantees that replacing intractable continuous inclusion-exclusion calculations with smooth algebraic operators (such as the continuous Noisy-OR) remains mathematically sound.

## 3.5 Generative Rules and Definite Clauses

With the state spaces established, we define how rules generate logical truths. Let a probabilistic rule $R _ { j }$ be defined syntactically as $p _ { j } : : H  B$

Under Continuous Distribution Semantics, this syntax is evaluated as a deterministic definite clause combined with the hidden, independent causal event $E _ { j } { \mathrm { : } }$

$$
H  B \land E _ { j }
$$

This generative definition ensures that the probabilistic uncertainty is isolated entirely to the independent event $E _ { j }$ If the rule body B evaluates to True under a specific continuous state $\sigma \in \Omega _ { s }$ , the rule actively generates the head

H if and only if $E _ { j }$ evaluates to True. Standard deterministic clauses $( 1 . 0 : \ : H \ :  \ : B )$ naturally follow this exact mechanism, generating the head unconditionally whenever the body holds true.

## 3.6 The Space of Possible Worlds

A complete MT-PDCL knowledge base is formally defined as the tuple $\mathcal { K } = \langle \mathcal { M } _ { s } , \Phi \rangle$

The total universe of uncertainty in MT-PDCL is defined by the independent combination of the continuous states and the discrete causal events. We define the space of all possible worlds as the Cartesian product of these two measurable spaces:

$$
\Omega = \Omega _ { s } \times \Omega _ { E }
$$

A specific possible world $\omega \in \Omega$ is defined by a pair $\omega = ( \sigma , \epsilon )$ , where:

$\sigma \in \Omega _ { s }$ rigidly resolves every stochastic predicate queried by the program into a specific continuous constant.

$\epsilon \in \Omega _ { E }$ rigidly determines the True/False outcome of every probabilistic causal event attached to the rules in Φ.

Mathematical independence between the continuous states and discrete causal events yields a total probability measure over the space of possible worlds that is the product measure:

$$
\mu _ { \Omega } = \mu _ { s } \otimes \mu _ { E }
$$

## 3.7 Declarative Entailment and Query Answers

Selecting a specific possible world $\omega = \left( \sigma , \epsilon \right)$ eliminates all probabilistic uncertainty. Inside $\omega ,$ every stochastic predicate yields a fixed constant provided by $\sigma ,$ and every rule either definitely fires or is completely removed based on ϵ. What remains is a classical, deterministic Definite Clause program, denoted $L _ { \omega }$

Standard model theory applies since $L _ { \omega }$ contains no disjunction in the head and no negation. The generative rules act as strict logical constraints, generating one unique Least Model for that specific world. This Least Model is the specific interpretation $\mathcal { T } _ { \omega } ,$ , as defined in Section 3.1. Therefore, a possible world $\omega$ and its induced interpretation $\mathcal { T } _ { \omega }$ are intrinsically linked.

## Ground Queries:

If a query $Q$ contains no free variables (a ground query), it is either true or false in this specific world. Formally, this is written as $\omega \mapsto Q$ , which mechanically means the query is satisfied by the interpretation corresponding to the least model $( \mathcal { T } _ { \omega } \ \models Q )$ . We define its exact inferred probability, denoted $P _ { \kappa } ( Q )$ , as the Lebesgue integral of the logical indicator function over the infinite space of possible worlds:

$$
P _ { \mathcal { K } } ( Q ) = \int _ { \Omega } \mathbb { I } ( \omega \ J \vert = Q ) d \mu _ { \Omega } ( \omega )
$$

where $\mathbb { I } ( \omega \mapsto Q ) = 1$ if Q is logically entailed by the least model of $L _ { \omega }$ , and 0 otherwise.

## Non-Ground Queries and Continuous Answers:

In standard definite clause logic, an exact answer to a non-ground query Q (with free variables V) is a discrete variable substitution θ mapping the variables to specific constants. However, in MT-PDCL, logical variables map natively to continuous domains. In continuous probability theory, the probability of a continuous variable taking a specific point value is zero, making point-substitution evaluations generally invalid. Therefore, an answer in MT-PDCL cannot be reduced to a discrete substitution. Instead, an answer defines a measurable geometric region alongside its exact declarative probability.

We define a probabilistic answer to a non-ground query $Q$ as a tuple $\langle \Omega _ { A } , P _ { K } ( Q , \Omega _ { A } ) \rangle \mathrm { ~ }$

$\Omega _ { A } \subseteq D ^ { n }$ is the answer space, a valid, measurable subset of the continuous universal domain representing a continuous family of bindings for the free variables V. Syntactically, this space is defined by an accumulated conjunction of built-in base predicates (e.g., $V _ { 1 } \geq 0 \land V _ { 1 } \leq 1 0 )$

$P \kappa ( Q , \Omega _ { A } ) \in [ 0 , 1 ]$ is the answer probability. It is the exact declarative probability that the query holds true for the continuous variable bindings bounded by $\Omega _ { A }$ under the knowledge base K.

Formally, the answer probability $P _ { \kappa } ( Q , \Omega _ { A } )$ for a specific continuous geometric region $\Omega _ { A }$ is defined by integrating the query’s indicator function across the space of possible worlds Ω:

$$
P _ { \mathsf { K } } ( Q , \Omega _ { A } ) = \int _ { \Omega } \mathbb { I } ( \omega \mid = Q \land \mathbf { V } \in \Omega _ { A } ) d \mu _ { \Omega } ( \omega )
$$

If the query $Q$ is ground $( \mathrm { i . e . }$ , it contains no free variables), the tuple of variables V is the empty tuple. The corresponding answer space $\Omega _ { A }$ reduces to the zero-dimensional space $D ^ { \hat { 0 } }$ . Mathematically, $D ^ { 0 }$ is a singleton set containing only the empty tuple. This unconditionally means that $\mathbf { V } \in \mathbf { \bar { \Omega } } \Omega _ { A }$ , reducing the probabilistic answer $\langle \Omega _ { A } , P _ { \mathcal K } ( Q , \Omega _ { A } ) \rangle$ directly to the scalar probability $P _ { \kappa } ( Q )$

This formulation ensures that the semantics of querying natively respect the underlying continuous measure theory, seamlessly generalizing standard discrete entailment.

## Theoretical Remark: Bypassing Herbrand Interpretations

In classical logic programming and discrete probabilistic logics, semantics are traditionally defined using Herbrand interpretations. In a Herbrand model, the domain of discourse is restricted to the finite or countably infinite set of syntactic constants present in the language.

MT-PDCL bypasses Herbrand interpretations entirely. Continuous domains (such as $\mathbb { R } )$ are uncountably infinite and cannot be reduced to a countable syntax of discrete symbols. A stochastic predicate generating continuous values represents an uncountably infinite number of states; exhaustive syntactic propositionalization is mathematically impossible. By anchoring the semantics in general interpretations over a Borel space and defining entailment via exact Lebesgue integration, the framework mathematically bypasses the combinatorial bottleneck of discrete grounding.

## 4 Illustrative Examples of Declarative Entailment

## 4.1 Example: Continuous Spatial Detection

To demonstrate the semantics of MT-PDCL and illustrate how it replaces discrete summation with exact integration, consider a spatial detection scenario. A security system monitors a specific stretch of a 10km road. The practical objective is to compute the exact scalar probability that a random target event will successfully trigger an alarm, allowing operators to mathematically quantify the system’s detection reliability across a continuous physical space.

## 4.1.1 Framework Setup

• Language L: A single logical variable $X \ \in \ \nu$ . The predicates are zone/1, alarm/0, and the stochastic predicate event\_loc $/ \bar { 1 } \in \mathcal { R } _ { s } ^ { \bar { } }$

• Domain $D { : }$ The continuous interval $[ 0 , 1 0 ] \subset \mathbb { R }$ , representing the road in kilometers, equipped with the standard Borel σ-algebra.

• Stochastic Measure Space $\Omega _ { s } \mathbf { : }$ The environment generates one continuous random variable for the event location. The space $\Omega _ { s }$ is $[ 0 , 1 0 ]$ , and its probability measure $\mu _ { s }$ is defined by the uniform density: $d \mu _ { s } ( x ) =$ 0.1 dx.

## 4.1.2 The Knowledge Base K

The knowledge base integrates the continuous measure declarations natively alongside the logical clauses:

1. event $\begin{array} { r } { \operatorname { l o c } ( ; X ) \sim \operatorname { U n i f o r m } ( 0 , 1 0 ) . } \end{array}$

(Measure declaration: the environment generates a single event location, distributed uniformly.)

2. $1 . 0 : \mathsf { z o n e } ( X ) \gets X \in [ 4 , 6 ] .$

(Deterministic rule: the sensor zone covers the interval $[ 4 , 6 ] . \jmath$

3. $0 . 8 : \mathsf { a l a r m } \gets \mathsf { e v e n t \_ l o c } ( ; X ) \land \mathsf { z o n e } ( X )$

(Probabilistic rule: if the event occurs inside the sensor zone, the independent causal event $E _ { 3 }$ succeeds, triggering the alarm with an 80% probability.)

## 4.1.3 Generative Evaluation

To compute the inferred probability of the ground fact alarm, we first evaluate the deterministic least model of the logic program for a specific continuous state $\sigma \in \Omega _ { s }$ . The state resolves the stochastic predicate to a concrete constant $x \in [ 0 , 1 0 ]$

First, we evaluate the logical body of the probabilistic rule: $B = { \mathrm { e v e n t \_ l o c } } ( ; x ) \land \mathrm { z o n e } ( x )$

• Inside the zone $( x \in [ 4 , 6 ] )$ : Both predicates are logically true. With the body evaluating to true, the rule actively generates the alarm fact upon the success of its associated causal event $E _ { 3 }$ . Since $\begin{array} { r } { \bar { P } ( E _ { 3 } ) = 0 . 8 } \end{array}$ , the exact conditional probability is:

$$
P _ { \mathcal { K } } ( \mathrm { a l a r m } \mid x ) = 0 . 8 
$$

• Outside the zone $( x \notin [ 4 , 6 ] ) \colon$ The predicate zone(x) is false, meaning the rule body fails. Under Distribution Semantics, the causal event cannot bridge a false premise. The rule fails to generate the head, yielding an exact probability of zero:

$$
P _ { \mathcal { K } } ( \mathrm { a l a r m } \mid x ) = 0
$$

## 4.1.4 Marginalization over the Measure Space

We calculate the final, declarative probability of the alarm triggering by integrating the conditional probability over the continuous state space $\Omega _ { s }$ , using the environment’s measure $\mu _ { s } \mathrm { . }$

$$
P _ { \mathcal { K } } ( \mathrm { a l a r m } ) = \int _ { \Omega _ { s } } P _ { \mathcal { K } } ( \mathrm { a l a r m } \mid x ) d \mu _ { s } ( x )
$$

Substituting our piecewise conditional probabilities and the uniform density $d \mu _ { s } ( x ) = 0 . 1 d x \colon$

$$
\begin{array} { c } { { P _ { K } ( \mathrm { a l a r m } ) = \displaystyle \int _ { 0 } ^ { 4 } 0 \cdot ( 0 . 1 ) d x + \int _ { 4 } ^ { 6 } 0 . 8 \cdot ( 0 . 1 ) d x + \int _ { 6 } ^ { 1 0 } 0 \cdot ( 0 . 1 ) d x } } \\ { { P _ { K } ( \mathrm { a l a r m } ) = 0 . 8 \cdot 0 . 1 \cdot ( 6 - 4 ) = 0 . 8 \cdot 0 . 2 = 0 . 1 6 } } \end{array}
$$

This geometric Lebesgue integration yields the exact declarative probability: $P _ { \mathcal { K } } ( \mathrm { a l a r m } ) \ : = \ : 0 . 1 6$ . Practically, the framework directly outputs this scalar 0.16 in response to the query ?- alarm., informing the operator that despite the target being guaranteed to exist on the road, the physical constraints of the sensor zone yield only a 16% chance of detection.

## 4.2 Example: Multi-Sensor Fusion and Discrete Categorization

To highlight a different aspect of MT-PDCL, specifically how deterministic continuous facts map to discrete, categorical conclusions without integration, we present a scenario based on multi-sensor fusion. The practical objective is to evaluate continuous sensor readings to instantly classify the system’s state and issue either a ‘critical’ or ‘moderate warning, efficiently pruning impossible logical branches without unnecessary integration.

## 4.2.1 Framework Setup

Consider an industrial monitoring system that reads exact temperature and pressure values to issue these specific warnings.

• Language $\mathcal { L } \mathbf { : }$ Two logical variables $X , Y \in \mathcal { V }$ . The predicates are temp/1, pressure/1, and warning/1. The warning predicate takes discrete constants $( \mathrm { e . g . }$ , critical, moderate).

• Domain D: A mixed domain containing the continuous set of real numbers R (for sensor readings) and a finite set of strings (for warning types), equipped with the standard Borel σ-algebra.

## 4.2.2 The Knowledge Base K

Because our observations in this scenario are exact numerical readings rather than stochastic estimates, there are no continuous measure declarations. The knowledge base consists of two deterministic sensor readings and two probabilistic rules:

1. $1 . 0 : \tan { \operatorname { p } } ( 1 2 0 . 5 ) .$

2. $1 . 0 : \mathrm { p r e s s u r e } ( 4 . 2 )$

(Deterministic ground facts: the sensors report an exact continuous temperature of 120.5 and pressure of $4 . 2 )$

$$
3 . \ 0 . 9 5 : \operatorname { w a r n i n g } ( \operatorname { c r i t i c a l } )  \operatorname { t e m p } ( X ) \land \operatorname { p r e s s u r e } ( Y ) \land X > 1 0 0 . 0 \land Y > 4 . 0 .
$$

(Probabilistic rule: if both thresholds are breached, issue a critical warning with 95% probability).

$$
4 . \ 0 . 8 0 : \mathrm { w a r n i n g } ( \mathrm { m o d e r a t e } ) \gets \mathrm { t e m p } ( X ) \wedge \mathrm { p r e s s u r e } ( Y ) \wedge X \leq 1 0 0 . 0 \wedge Y > 3 . 0 .
$$

(Probabilistic rule: if only pressure is high, issue a moderate warning with 80% probability).

## 4.2.3 Deterministic Evaluation of Causal Events

Because the environment contains no continuous random variables, the engine evaluates the logical derivations directly against the provided constants.

Evaluating Rule 3 (Critical Warning): Through standard unification, the rule’s body becomes $B = \mathrm { t e m p } ( 1 2 0 . 5 ) \wedge$ pressure $( 4 . 2 ) \wedge 1 2 0 . 5 > 1 0 0 . 0 \wedge 4 . 2 > 4 . 0$ . Because the sensor readings explicitly exist in Φ and the mathematical inequalities hold, this body is true. Consequently, the generation of the head depends entirely on the rule’s independent causal event $E _ { 3 }$

$$
P _ { \mathcal { K } } ( \mathrm { w a r n i n g } ( \mathrm { c r i t i c a l } ) ) = P ( E _ { 3 } ) = 0 . 9 5
$$

Evaluating Rule 4 (Moderate Warning): For the exact same assignment, the moderate warning body evaluates $1 2 0 . 5 \leq 1 0 0 . 0$ . Because this mathematical constraint is logically false, the rule body fails. Under generative semantics, a false body means the causal event is irrelevant, and the probability of generating this specific head is zero:

$$
P _ { \mathrm { { \mathcal { K } } } } ( \mathrm { w a r n i n g } ( \mathrm { m o d e r a t e } ) ) = 0
$$

By evaluating the queries for both warning states, the framework practically outputs 0.95 for the critical warning and 0 for the moderate warning. This resolves the objective, demonstrating how MT-PDCL leverages exact deterministic observations to instantly resolve categorical decisions.

## 4.2.4 Contrasting Prior Uncertainty and Backward Compatibility

Comparing this multi-sensor scenario to the previous spatial detection example reveals a fundamental flexibility of MT-PDCL.

In the first example, the location of the event was unobserved. The framework relied on the stochastic measure mapping to define an integration space, producing an expected probability based on continuous uncertainty.

In this second example, the knowledge base contains exact, deterministic observations. No integration over a prior distribution is required, because the exact observations collapse the evaluation to a definitive set of constants.

Crucially, this exact scenario can be modeled and executed natively in standard discrete frameworks such as ProbLog or PRISM. Because real numbers acting as fixed deterministic constants do not invoke measure-zero problems, standard discrete solvers can process the arithmetic inequalities without modification. This demonstrates that when the stochastic measure mapping $\mathcal { M } _ { s }$ is empty, MT-PDCL reduces cleanly to classical Distribution Semantics, ensuring strict backward compatibility with existing discrete probabilistic logic programs.

## 4.3 Example: Continuous Stochastic Optimization and Querying

In classical logic programming, constraint satisfaction problems (CSPs) are solved by querying the knowledge base for a set of discrete variable bindings that satisfy a logical goal. MT-PDCL extends this paradigm to continuous stochastic environments. An intelligent agent seeks to find the deterministic parameters that maximize the exact inferred probability of a successful outcome, subject to continuous environmental uncertainty and hard constraints. Practically, the agent requires two distinct outputs: the exact physical energy allocation that maximizes mission success, and the geometric boundaries of valid allocations alongside their baseline success probabilities if queried dynamically.

## 4.3.1 Framework Setup

An agent has a strict total energy budget of 10 units. It must allocate X units to its navigation system and $Y$ units to its communication system. The environment presents continuous, random resistance to both systems.

To model a realistic asymmetry, the physics of the two systems differ. Signal strength scales linearly with the communication energy Y . However, because physical movement (e.g., overcoming aerodynamic drag) requires energy that scales quadratically with velocity, the effective operational yield of the navigation energy X scales only with its square root $( { \sqrt { X } } )$ .

• Language $\mathcal { L } \mathbf { : }$ Logical variables $X , Y , R _ { n a v } , N _ { c o m } \in \mathcal { V } .$ . The predicates are nav $\mathbf { - o k } / 1$ , com $\mathrm { - } ^ { \mathrm { { o k / 1 } } }$ , and mission/2. The stochastic predicates are noise $\mathbf { \Pi } _ { - } \mathbf { n } \mathbf { a v } / 1$ and noise\_com/1.

• Domain D: The continuous set of real numbers R.

• Stochastic Measure Space $\Omega _ { s } \mathbf { : }$ The environment generates two independent continuous variables. The state space $\Omega _ { s } \ \mathbf { i s } \ \mathbb { R } ^ { 2 }$ , mapping to specific environmental outcomes $( r , n )$

## 4.3.2 The Knowledge Base K

The knowledge base explicitly defines the environmental noise distributions and the system rules:

1. noise\_nav $( ; R _ { n a v } ) \sim \mathrm { U n i f o r m } ( 0 , 4 )$

2. noise\_com $( ; N _ { c o m } ) \sim \mathrm { U n i f o r m } ( 0 , 1 0 )$

(Measure declarations: establishing the independent continuous bounds for terrain resistance and atmospheric noise. Notice the asymmetric bounds.)

3. $1 . 0 : : \mathrm { n a v \_ o k ( } X \mathrm { )  \mathrm { n o i s e \_ n a v ( } ; } R _ { n a v } \mathrm { ) \land } \sqrt { X } \geq R _ { n a v } .$

(The navigation subsystem succeeds if the square root of the allocated energy X equals or exceeds the continuous terrain resistance.)

4. 0.9 :: com\_ok(Y) ← noise\_com $( ; N _ { c o m } ) \wedge Y \geq N _ { c o m } .$

(The communication subsystem succeeds with a 90% causal probability if the allocated energy $Y$ overcomes the atmospheric noise.)

5. $1 . 0 : \operatorname { m i s s i o n } ( X , Y ) \gets \operatorname { n a v \_ o k } ( X ) \wedge \operatorname { c o m \_ o k } ( Y ) \wedge X \geq 0 \wedge Y \geq 0 \wedge X + Y \leq 1 0 .$

(The overall mission succeeds if both subsystems succeed, provided the energy allocations are non-negative and respect the maximum budget constraint.)

## 4.3.3 Evaluating the Target Function

To find the optimal allocation, the agent queries the knowledge base for the probability of mission $( x , y )$ as a continuous function of its specific choices x and y. We calculate the entailed probability $P _ { \mathcal { K } } ( \operatorname* { m i s s i o n } ( x , y ) )$ by evaluating the exact measure of the logically true worlds across $\Omega _ { s }$

Because the two noise variables are independent, the joint measure $\mu _ { s }$ over $\Omega _ { s }$ is defined by the density function $\begin{array} { r } { d \mu _ { s } ( r , n ) = \frac { 1 } { 4 } \cdot \frac { 1 } { 1 0 } d r d n = \frac { 1 } { 4 0 } } \end{array}$ dr dn on the support $[ 0 , \dot { 4 } ] \times [ 0 , 1 0 ]$

For a specific allocation $( x , y )$ that satisfies the budget constrain $x + y \leq 1 0 .$ , we integrate the generative probabilities over the continuous space where the logical conditions hold $( r \leq \sqrt { x }$ and n $\leq y )$ ):

$$
\begin{array} { l l l } { \displaystyle P _ { \mathcal K } ( \mathtt { n a v \_ o k } ( x ) ) = \int _ { 0 } ^ { \sqrt x } 1 . 0 \cdot \frac 1 4 d r = \frac { \sqrt x } { 4 } } \\ { \displaystyle P _ { \mathcal K } ( \mathtt { c o m \_ o k } ( y ) ) = \int _ { 0 } ^ { y } 0 . 9 \cdot \frac 1 { 1 0 } d n = \frac { 0 . 9 y } { 1 0 } } \end{array}
$$

Because the subsystems depend on independent variables in the measure space and independent causal events, their joint success probability is the product of their marginals. The deterministic rule for the mission evaluates this conjunction:

$$
P _ { \mathcal { K } } ( \operatorname* { m i s s i o n } ( x , y ) ) = \frac { \sqrt { x } } { 4 } \cdot \frac { 0 . 9 y } { 1 0 } = \frac { 0 . 9 y \sqrt { x } } { 4 0 }
$$

## 4.3.4 Optimization as an Inference Task

In standard logic programming extensions, optimization is often executed internally using meta-predicates that orchestrate branch-and-bound searches across discrete bindings. However, in MT-PDCL, the probability of a derived fact is a semantic property of the model computed via integration over a measure space, not a syntactic variable bound during rule evaluation. Therefore, optimization is treated as a meta-level inference task evaluated over the knowledge base.

The logic program natively defines the continuous objective function representing the declarative probability: $\begin{array} { r } { f ( x , y ) = P _ { \mathrm { K } } ( \mathrm { m i s s i o n } ( x , y ) ) = \frac { 0 . 9 y \sqrt { x } } { 4 0 } } \end{array}$ . The agent queries the solver to maximize this probability subject to the strict logical constraint.

Formally, the agent issues an optimization query:

$$
\arg \operatorname* { m a x } _ { x , y } P _ { K } ( \operatorname* { m i s s i o n } ( x , y ) ) \quad { \mathrm { s u b j e c t } } \operatorname { t o } \quad x + y \leq 1 0
$$

By substituting $y = 1 0 - x$ into the objective function, we obtain $\textstyle f ( x ) = { \frac { 0 . 9 } { 4 0 } } ( 1 0 { \sqrt { x } } - x { \sqrt { x } } )$ . Setting the derivative to zero yields a non-trivial optimum:

$$
f ^ { \prime } ( x ) = { \frac { 0 . 9 } { 4 0 } } \left( { \frac { 5 } { \sqrt { x } } } - { \frac { 3 { \sqrt { x } } } { 2 } } \right) = 0 \implies 1 0 = 3 x \implies x = { \frac { 1 0 } { 3 } }
$$

The exact maximum is reached by allocating $x = \textstyle { \frac { 1 0 } { 3 } } \approx 3 . 3 3$ units to navigation and $y \ = \ { \frac { 2 0 } { 3 } } \approx \ 6 . 6 7$ units to communication. Because navigation suffers from diminishing returns √γ $( { \sqrt { x } } )$ , the optimal strategy correctly allocates double the resources to the linear communication subsystem.

Substituting this optimal strategy back into the declarative engine establishes the exact maximum achievable probability for the mission:

$$
P _ { K } ( \mathrm { m i s s i o n } ( 1 0 / 3 , 2 0 / 3 ) ) = \frac { 0 . 9 \cdot ( 2 0 / 3 ) \cdot \sqrt { 1 0 / 3 } } { 4 0 } \approx 0 . 2 7 3 9
$$

Practically, the framework returns the optimal allocation (3.33 units to navigation, 6.67 to communication) and its maximized 27.3% success probability. This formal separation ensures the logic remains declarative. The rules define the probability measure space, while the inference engine handles the continuous optimization over that space.

## 4.3.5 Executing Ground and Non-Ground Queries

While optimization searches for a specific maximum, the framework also supports standard logical queries to evaluate specific fixed scenarios or explore continuous parameter spaces.

Ground Query: An agent can query the exact probability of success for a specific, predetermined energy allocation. Syntactically, this is written as a headless clause with no free variables:

$$
 \mathrm { m i s s i o n } ( 4 . 0 , 6 . 0 )
$$

Because the query is completely ground, the tuple of variables is empty, and the answer space $\Omega _ { A }$ reduces to the zerodimensional space $D ^ { 0 }$ . The declarative engine simply substitutes the constants into the continuous objective function derived from the knowledge base:

$$
P _ { K } ( \mathrm { m i s s i o n } ( 4 . 0 , 6 . 0 ) ) = { \frac { 0 . 9 \cdot 6 . 0 \cdot { \sqrt { 4 . 0 } } } { 4 0 } } = { \frac { 1 0 . 8 } { 4 0 } } = 0 . 2 7
$$

The framework returns the exact scalar probability 0.27.

Non-Ground Query: An agent can also submit a query containing free variables to explore a valid operational region. For example, the agent may ask for the continuous conditions and the exact probability that at least one allocation will succeed if it commits at least 3.0 units of energy to navigation:

$$
 \mathrm { m i s s i o n } ( X , Y ) \land X \geq 3 . 0
$$

Because the query contains free variables, the framework returns a probabilistic answer tuple $\langle \Omega _ { A } , P _ { K } ( Q , \Omega _ { A } ) \rangle$

First, the engine accumulates the explicit base constraints defined in the rule body and the query to define the geometric answer space $\Omega _ { A }$ for the free variables:

$$
\Omega _ { A } \equiv \left( X \geq 3 . 0 \right) \wedge ( Y \geq 0 ) \wedge ( X + Y \leq 1 0 )
$$

Next, it calculates the declarative probability $P _ { \kappa } ( Q , \Omega _ { A } )$ . As defined in Section 3.7, the engine does not integrate over the choice variables X and Y. Instead, it integrates over the space of possible worlds $\Omega _ { s }$ (the continuous environmental noise).

In a specific world where the terrain resistance is r and atmospheric noise is $n ,$ the query holds true if there is at least one valid allocation $( X , Y ) \in \Omega _ { A }$ that satisfies the deterministic constraints $\sqrt { X } \geq r$ and $Y \geq n$ . This combined region is only satisfiable if $n \leq 1 0 - \operatorname* { m a x } ( 3 , r ^ { 2 } )$ .

The engine computes the declarative probability by integrating the joint probability measure $\mu _ { s }$ over the subset of the noise space where this condition holds true, multiplied by the 0.9 independent causal probability of the communication rule:

$$
P _ { K } ( Q , \Omega _ { A } ) = 0 . 9 \int _ { 0 } ^ { \sqrt { 1 0 } } \int _ { 0 } ^ { 1 0 - \operatorname * { m a x } ( 3 , r ^ { 2 } ) } \frac { 1 } { 4 0 } d n d r
$$

Evaluating this exact continuous integral yields the true declarative probability:

$$
P _ { K } ( Q , \Omega _ { A } ) = \frac { 0 . 9 } { 4 0 } \left( \frac { 2 0 \sqrt { 1 0 } } { 3 } - 2 \sqrt { 3 } \right) \approx 0 . 3 9 6
$$

The framework successfully returns the geometric limits of the valid allocation space alongside its existential probability: $\langle \Omega _ { A } , 0 . 3 9 6 \rangle$ . Practically, this tuple provides the agent with the precise geometric region it must operate within $( \Omega _ { A } )$ and guarantees a 39.6% baseline probability of success if it commits to any allocation inside those boundaries.

## Theoretical Remark: Query Resolution vs. Standard Consequence Operators

In standard Definite Clause Logic (DCL), answering a query is a direct lookup against the least fixed point of the consequence operator. The inference engine simply unifies the query against the static set of derived ground conclusions.

This example demonstrates that in MT-PDCL, resolving a non-ground query requires a strict separation of concerns. Because the environment is stochastic and continuous, every possible world ω generates its own independent fixed point $\mathcal { T } _ { \omega }$ . Evaluating an existential query mathematically integrates the probability measure of all worlds where the query’s answer space $\Omega _ { A }$ successfully intersects with the world’s specific fixed point. Therefore, the framework requires two distinct mechanisms: a continuous consequence operator to formally derive the probability of specific ground instances (detailed in Section 5), and a separate meta-level integration step to resolve free variables across the space of models.

## 4.4 Example: Parameterized Distributions and Continuous Marginalization

To illustrate the semantics of index variables (I) and how deterministic logical variables natively map to the parameters of dynamic probability distributions, we model a mechanical component that fails at a random time, producing a drifting thermal signature. The practical objective is to compute the exact expected probability of a thermal alarm triggering, rigorously accounting for both the unpredictable time of failure and the continuous thermal drift that occurs up to that exact moment.

## 4.4.1 Framework Setup

The component fails at a random continuous time T. Upon failure, a thermal sensor reads its temperature $X .$ The component grows hotter the longer it runs, meaning the mean of the temperature reading drifts directly as a function of the failure time T.

• Language $\mathcal { L } \mathbf { : }$ Logical variables $T , X \in \mathcal { V } .$ . The predicates are alarm/0 and two stochastic predicates: failure\_time/1 and heat\_signature $; / 2$

• Domain D: The continuous set of real numbers R.

## 4.4.2 The Knowledge Base K

The knowledge base demonstrates how the deterministic output of one stochastic predicate acts as the binding index for another, naturally parameterizing its distribution:

1. failure\_time $( ; T ) \sim \mathrm { U n i f o r m } ( 0 , 1 0 )$

(The dynamic variable: failure occurs randomly between 0 and 10 hours.)

2. heat\_signature $\left( T ; X \right) \sim \mathrm { N o r m a l } ( \mu = 5 . 0 \cdot T , \sigma = 2 . 0 )$

(The drifting sensor: a stochastic predicate with a non-empty index T. Its mean is deterministically parameterized by the given time $T . )$

3. 1.0 :: alarm ← failure\_time $\left( ; T \right) \wedge$ heat\_signature $( T ; X ) \wedge X > 3 0 . 0 .$

(The deterministic rule: an alarm triggers if the reading exceeds 30 degrees at the exact time of failure.)

## 4.4.3 Conditional Evaluation

To compute the probability of the alarm, we evaluate the rule for a specific continuous state $\sigma \in \Omega _ { s }$ . The first stochastic predicate provides a concrete failure time $t \in [ 0 , 1 0 ]$ . This time t uniquely binds the index T in the second predicate, fully resolving its distribution to Normal(5.0 · t, 2.0).

The body requires $X ~ > ~ 3 0 . 0$ . Treating the variables as continuous, we integrate the Normal probability density function (PDF), denoted ${ \mathcal { N } } .$ , to find the conditional probability of the body being true for that specific time t:

$$
P _ { \mathrm { \mathcal { K } } } ( \mathrm { a l a r m \mid \it t ) } = \int _ { 3 0 } ^ { \infty } \mathcal { N } ( x ; 5 . 0 t , 2 . 0 ) d x = 1 - \Phi \left( \frac { 3 0 - 5 . 0 t } { 2 . 0 } \right)
$$

where $\Phi$ is the cumulative distribution function (CDF) of the standard normal distribution.

To find the final declarative entailment, we marginalize out the latent variable T. The operator computes the Lebesgue integral over the continuous prior measure of the failure time:

$$
P _ { \mathcal { K } } ( \mathrm { a l a r m } ) = \int _ { 0 } ^ { 1 0 } \left( 1 - \Phi \left( { \frac { 3 0 - 5 . 0 t } { 2 . 0 } } \right) \right) \cdot 0 . 1 d t
$$

Evaluating the query ?- alarm. returns the scalar result of this integral. This provides the engineering team with the exact expected alarm probability, successfully resolving the dual uncertainty of time and thermal drift. This example highlights the expressive power of MT-PDCL’s separation between deterministic indices and stochastic outputs. By simply passing the logical variable T from the body into the index slot of heat\_signature, the logic program effortlessly defines an infinite family of conditional probability distributions and delegates the marginalization to the continuous consequence operator, completely avoiding discrete propositionalization.

## 4.5 Example: Hybrid Measure Spaces and Poisson Processes

To demonstrate how MT-PDCL seamlessly unifies continuous and discrete variables without breaking the mathematical semantics, we model a hybrid dynamic process. The practical objective is for a reliability engineer to calculate the exact probability of a machine suffering a critical failure (defined as experiencing 3 or more stress faults) across an unknown continuous lifespan.

## 4.5.1 Framework Setup

This scenario requires the universal domain to process both continuous time and discrete event counts simultaneously.

• Language L: Logical variables $T , K \in \mathcal { V }$ . The predicates are critical\_failure/0 and two stochastic predicates: run\_time/1 and fault\_count $/ 2 .$

• Domain D: A disjoint union of the continuous real numbers (for time) and the discrete natural numbers (for fault counts), formally structured as a standard Borel space: $D = \mathbb { R } \cup \mathbb { N }$

## 4.5.2 The Knowledge Base K

The declarations highlight a continuous stochastic variable dynamically acting as the index for a discrete probability mass function:

1. run\_ $\mathrm { t i m e } ( ; T ) \sim \mathrm { U n i f o r m } ( 0 , 5 )$ (Continuous variable: The machine runsfor a random duration between 0 and 5 hours.)

2. fault $\operatorname { c o u n t } ( T ; K ) \sim \operatorname { P o i s s o n } ( \lambda = 2 . 0 \cdot T )$

(Discrete variable parameterized by continuous index: The environment generates a discrete count offaults $K \in \mathbb { N } .$ The rate λ is rigidly parameterized by the continuous operating time T.)

3. 1.0 :: critical\_failure $ \mathrm { r u n \_ t i m e } ( ; T ) \land$ fault\_count $( T ; K ) \land K \geq 3$

(Deterministic rule: The system suffers a critical failure if it experiences 3 or more faults during its operational run.)

## 4.5.3 Hybrid Evaluation and Marginalization

To compute the exact declarative probability of a critical failure, the continuous consequence operator must navigate both the continuous integration over time and the discrete summation over the Poisson mass function.

First, we evaluate the logical body for a specific continuous state where the machine runs for an exact duration $t \in$ [0, 5]. This constant t binds to the index of the Poisson distribution, fixing its parameter to $\lambda = 2 . 0 t$

The logical condition requires $K \geq 3 .$ . Using the standard Poisson probability mass function $\begin{array} { r } { P ( K = k ) = \frac { \lambda ^ { k } e ^ { - \lambda } } { k ! } } \end{array}$ , the operator computes the discrete sum for the exact conditional probability that the body evaluates to true:

$$
P _ { K } ( { \mathrm { c r i t i c a l \_ f a i l u r e } } \mid t ) = 1 - \sum _ { k = 0 } ^ { 2 } { \frac { ( 2 . 0 t ) ^ { k } e ^ { - 2 . 0 t } } { k ! } } = 1 - e ^ { - 2 . 0 t } \left( 1 + 2 . 0 t + { \frac { ( 2 . 0 t ) ^ { 2 } } { 2 } } \right)
$$

To find the final declarative entailment, the framework marginalizes out the latent continuous operating time. The operator computes the Lebesgue integral of this discrete conditional summation over the continuous uniform prior

density $( d \mu _ { T } ( t ) = 0 . 2 d t )$ across the geometric boundary [0, 5]:

$$
P _ { \mathcal { K } } ( \mathrm { c r i t i c a l \_ f a i l u r e } ) = \int _ { 0 } ^ { 5 } \left( 1 - e ^ { - 2 t } ( 1 + 2 t + 2 t ^ { 2 } ) \right) \cdot 0 . 2 d t
$$

Applying standard integration by parts to the polynomial terms yields an exact, closed-form analytical solution:

$$
{ \begin{array} { r l } & { P _ { K } { \mathrm { ( c r i t i c a l \_ f a i l u r e ) } } = 0 . 2 \left[ t + e ^ { - 2 t } \left( t + { \frac { 1 } { 2 } } \right) + t e ^ { - 2 t } + \left( t ^ { 2 } e ^ { - 2 t } + t e ^ { - 2 t } + { \frac { 1 } { 2 } } e ^ { - 2 t } \right) \right] _ { 0 } ^ { 5 } } \\ & { \qquad P _ { K } { \mathrm { ( c r i t i c a l \_ f a i l u r e ) } } = 0 . 7 + 7 . 3 e ^ { - 1 0 } \approx 0 . 7 0 0 0 0 0 3 } \end{array} }
$$

The framework evaluates ?- critical\_failure. and outputs 0.70. For the reliability engineer, this directly resolves the objective, confirming a 70% chance the machine will fail by perfectly balancing the continuous uncertainty of its lifespan against the discrete risk of sequential faults.

This exact calculation highlights a crucial architectural advantage. In standard logic programming frameworks, modeling a Poisson distribution requires discrete recursive unrolling of every possible integer count, causing an immediate combinatorial explosion, followed by manual numerical bucketing of the continuous parameter T. In MT-PDCL, the discrete random variable and its continuous parameter exist natively in the same formal measure space, allowing the consequence operator to resolve the hybrid interaction using exact, stable algebraic integration.

## 4.6 Example: Continuous Information Cascades in Stochastic Graphs

To demonstrate how MT-PDCL evaluates dynamic, time-dependent recursion over relational structures, we model the diffusion of information across a social network. An external event triggers a sequence of posts from a small set of root users within a specific time window. The probability that these initial posts are relevant to the event decays continuously over time. Their followers subsequently retweet the information subject to continuous reaction delays and a time-decaying probability of attention. The objective is to compute the probability that a specific target user receives accurate information about the event before a deadline, handling redundant signals and cyclic echo chambers

## 4.6.1 Framework Setup

• Language L: Logical variables $X , Y \in \mathcal { V }$ representing discrete users; $C , K \in \mathcal { V }$ representing discrete tweet counts and indices; and $T , T _ { i n } , T _ { o u t } , \Delta , A , \dot { R } \in \mathcal { V }$ mapping to continuous values. The relations are root/1, follows/2, and target/1 (which explicitly model the static topology of the social graph and the query objective), alongside the dynamic relation informed/2. The stochastic predicates are tweet\_count/2, tweet\_time/3, relevance/3, reaction\_delay/4, and attention/4.

• Domain D: A standard Borel space constructed from the disjoint union of the continuous real numbers R (representing time and probabilities) and the countable discrete set containing the finite network users ${ \mathcal { U } } = \bar { \{ }   \mathbf { u } _ { 1 } , \mathbf { u } _ { 2 } , \dots , \mathbf { u } _ { n } \}$ and the natural numbers N (for sequence indexing).

## 4.6.2 The Knowledge Base K

The program explicitly declares the operational time boundaries as deterministic facts, allowing the geometric integration limits to be dynamically queried by the rules.

1. initial\_time $( X ; T ) \sim \mathrm { U n i f o r m } ( 0 , 5 )$

(The environment generates the continuous timestamp for a root user X’s very first reaction to the event.)

2. inter $\begin{array} { r } { \underline { { \mathsf { I w e e t \_ d e l a y } } } ( X , K ; \Delta ) \sim \underline { { \mathsf { E x p o n e n t i a l } } } ( \lambda = 1 . 0 ) } \end{array}$

(The continuous time delay between a root user X’s previous tweet and their next tweet K in the sequence.) 3. relevance(X, K; R) ∼ Uniform(0, 1). 3. relevance(X. K: R) \~ Uniform(0. 1)

(An independent continuous variable used to natively model the decaying probability that tweet K for root user X is relevant to the external event.)

4. reaction\_delay $( X , Y , T _ { i n } ; \Delta ) \sim \mathrm { E x p o n e n t i a l } ( \lambda = 0 . 5 )$

(The continuous time delay before intermediary user Y retweets user X.)

5. attention $( X , Y , T _ { i n } ; A ) \sim \mathrm { U n i f o r m } ( 0 , 1 )$

(An independent continuous variable modeling the dynamic probability that a user pays attention to an incoming tweet.)

6. 1.0 :: event\_window(10.0). 1.0 :: deadline(12.0). (Deterministic base facts establishing the strict 10-hour boundary for root events and the 12-hour limit for overall signal propagation.)

7. 1.0 :: root(u<sub>1</sub>). 1.0 :: follows $( \mathbf { u } _ { 2 } , \mathbf { u } _ { 1 } ) . \mathrm { ~ \quad ~ } 1 . 0 : : \mathrm { t a r g e t } ( \mathbf { u } _ { n } ) . \mathrm { ~ . ~ . ~ } .$ (Deterministic graph topology establishing the social connections and identifying the specific user for the final query.)

8. 1.0 :: root\_sequence $( X , 1 , T ) \xleftarrow { } \mathrm { r o o t } ( X ) \land \mathrm { e v e n t \_ w i n d o w } ( T _ { m a x } ) \land \mathrm { i n i t i a l \_ t i m e } ( X ; T ) \land T \le T _ { m a x } .$ (Sequence Base: The first tweet in the train $( K = 1 )$ occurs at the initial continuous time, bounded by the queried event window.)

9. 1.0 :: root\_sequence $( X , K , T ) \quad $ root\_sequence(X, $\begin{array} { r l r } { K _ { p r e v } , T _ { p r e v } ) \mathrm { ~ \land ~ } K } & { { } = } & { K _ { p r e v } + 1 \mathrm { ~ \land ~ } } \end{array}$ event\_window $( T _ { m a x } )$ ∧ inter\_tweet\_delay $( X , K ; \Delta ) \wedge T = T _ { p r e v } + \Delta ^ { \bullet } \wedge T \leq T _ { m a x } .$ (Sequence Recursion: Subsequent tweets in the train are generated by adding a strictly positive continuous delay. The train naturally halts when T exceeds the queried parameter $T _ { m a x } . )$

10. 1.0 :: informed(X, T) ← root\_sequence $( X , K , T ) \land$ relevance $( X , K ; R ) \land R \leq \exp ( - 0 . 5 \cdot T )$ (Root Information: A root user is informed at time T iftweet K is generated and its relevance R satisfies the time-decaying threshold.)

11. $0 . 6 : \mathrm { i n f o r m e d } ( Y , T _ { o u t } ) \gets \mathrm { f o l l o w s } ( Y , X ) \wedge$ informed $\left( X , T _ { i n } \right) \wedge$ attention $( X , Y , T _ { i n } ; A ) \land A \leq \exp ( - 0 . 1$ $T _ { i n } ) \wedge$ reaction\_delay $( X , Y , T _ { i n } ; \Delta ) \wedge T _ { o u t } = T _ { i n } + \Delta \wedge$ deadline $( T _ { d e a d } ) \wedge T _ { o u t } \leq T _ { d e a d } .$ (Recursive propagation: The message propagates subject to continuous thresholds and delays, bounded strictly by the global $T _ { d e a d }$ parameter queriedfrom the knowledge base.)

## 4.6.3 Cyclic Recursion and Continuous Aggregation

Because MT-PDCL evaluates semantics via a bottom-up consequence operator $( \mathbb { T } \kappa )$ , the proof advances forward in time. The operator starts with the initial event facts and iteratively pushes the probability mass through the network, accumulating the continuous delays $\Delta$ at each hop.

Once the consequence operator reaches its least fixed point, the objective

$$
 \mathrm { t a r g e t } ( X ) \land \mathrm { i n f o r m e d } ( X , T )
$$

acts as a final projection. The framework isolates the target user X from the generated model and marginalizes (integrates) over the accumulated time variable $T$ to output a single scalar: the total probability that the target user was informed.

Because the social network graph contains cyclic paths, naive bottom-up evaluation would trigger infinite loops. Furthermore, a user may receive the same information multiple times. MT-PDCL handles this complexity in two ways. First, the continuous Noisy-OR operator aggregates overlapping derivations via integration over the latent time distributions. Second, the operator computes the limit of the cyclic recursion. Because the time delay $\Delta$ is strictly positive, the accumulated time grows monotonically along any cycle. The physical constraint bounding the propagation $( T _ { o u t } \ \leq \ T _ { d e a d } )$ forces the integration volume to shrink to zero as cycles repeat, halting the recursion and guaranteeing convergence.

## 4.7 Theoretical Justification: Bounded Approximation Error

To formally justify the accuracy of the MT-PDCL consequence operator on highly cyclic, dense networks, we quantify the approximation error introduced by the Noisy-OR assumption. While dense topologies cause the number of overlapping derivation paths to grow rapidly, the monotonic progression of time reduces the probability mass of longer paths. The following theorem establishes that the absolute error gap between the operational Noisy-OR probability and the exact declarative probability is bounded and contracts exponentially.

Theorem 1 (Bounded Approximation Error for Time-Decaying Cascades). Let $G = ( \nu , \mathcal { E } )$ be a directed network graph, and let S be the finite set of valid derivation paths from any root to a target node, bounded by the physical deadline $T _ { d e a d }$ . Let $k _ { m i n } \geq 1$ be the minimum discrete path length (number ofhops) in S.

Assume the continuous temporal transition across any edge is given by $\Delta \sim$ Exponentia $\left[ \left( \lambda _ { D } \right) \right.$ with $\lambda _ { D } > 0$ . Let the causal propagation probability at hop $j \geq 1$ be strictly bounded by $\exp ( - \lambda _ { A } T _ { j } )$ , where $T _ { j }$ is the accumulated continuous time. Furthermore, let the initial relevance probability at the root $( j = 0 )$ be bounded by $\exp ( - \lambda _ { R } T _ { 0 } )$ , with $\lambda _ { R } > 0$

$I f P _ { o p }$ is the Noisy-OR operational probability and $P _ { e x }$ is the exact maximally-correlated probability, then the absolute approximation error $\epsilon \doteq | P _ { o p } - \bar { P _ { e x } } |$ is bounded by:

$$
\epsilon \leq \frac { 1 } { 2 } | S | ^ { 2 } \mathbb { E } [ \exp ( - 2 \lambda _ { R } T _ { 0 } ) ] \left( \frac { \lambda _ { D } } { \lambda _ { D } + \lambda _ { A } } \right) ^ { 2 k _ { m i n } }
$$

Proof. By the expansion of the Noisy-OR operator, the absolute error ϵ introduced by assuming path independence is bounded by the sum of all pairwise joint path probabilities. This simplifies to:

$$
\epsilon \leq { \frac { 1 } { 2 } } \left( \sum _ { \pi \in S } \mathbb { E } [ p ( \pi ) ] \right) ^ { 2 }
$$

For any path $\pi$ of length $k ,$ the accumulated continuous time at hop j is $\begin{array} { r } { T _ { j } \ = \ T _ { 0 } + \sum _ { m = 1 } ^ { j } \Delta _ { m } } \end{array}$ . The expected probability mass for the path is constrained by both the initial root relevance decay and the subsequent propagation attention decay at each hop. Because $T _ { j } \geq \Delta _ { j }$ , we can bound the time-decay at each individual hop by $\exp ( - \lambda _ { A } \Delta _ { j } )$ Since the delays $\Delta _ { m }$ are independent and Exponentially distributed, the expectation of the product evaluates directly via the moment-generating function:

$$
\mathbb { E } [ p ( \pi ) ] \le \mathbb { E } [ \exp ( - \lambda _ { R } T _ { 0 } ) ] \prod _ { m = 1 } ^ { k } \mathbb { E } \left[ \exp ( - \lambda _ { A } \Delta _ { m } ) \right] = \mathbb { E } [ \exp ( - \lambda _ { R } T _ { 0 } ) ] \left( \frac { \lambda _ { D } } { \lambda _ { D } + \lambda _ { A } } \right) ^ { k }
$$

Because $\lambda _ { D }$ and $\lambda _ { A }$ are positive, the ratio $\frac { \lambda _ { D } } { \lambda _ { D } + \lambda _ { A } }$ is strictly less than 1. Therefore, this geometric decay is maximized at the shortest valid path length $k _ { m i n }$ . Substituting this upper bound into the pairwise error sum and squaring the result over the |S| valid paths yields the final bounded limit:

$$
\epsilon \leq \frac { 1 } { 2 } | S | ^ { 2 } \mathbb { E } [ \exp ( - 2 \lambda _ { R } T _ { 0 } ) ] \left( \frac { \lambda _ { D } } { \lambda _ { D } + \lambda _ { A } } \right) ^ { 2 k _ { m i n } }
$$

Practical Outcome: The framework evaluates the query and outputs the exact scalar probability that the target user receives the information in time. Practically, this allows network analysts to quantify information reachability under dynamic, continuous constraints.

This example resolves a computational limitation in existing discrete solvers. Modeling this scenario in a traditional probabilistic logic framework requires discretizing the 12-hour window into fixed time steps. Grounding a highly connected, cyclic social graph over discrete time steps causes exponential growth in the underlying boolean circuit size. MT-PDCL avoids this issue by evaluating the temporal boundary as a unified geometric region within the continuous consequence operator.

## 5 Operational Semantics and Theoretical Properties

## 5.1 The Continuous Probabilistic Consequence Operator

Before formalizing the consequence operator, we must define what constitutes a derivation in a measure-theoretic setting. In standard logic programming, a derivation is a discrete sequence of rule applications that proves a specific conclusion. In MT-PDCL, a derivation is a specific logical path that contributes probability mass to a conclusion. Distinct derivations for the same conclusion arise in two ways: structurally, from completely different rules whose heads conclude the same relation; and continuously, from disjoint subsets of continuous variables evaluating the same rule body. When a knowledge base contains multiple such derivations for the same conclusion, their probabilistic contributions must be aggregated.

In standard definite clause logic, operational semantics are defined by the immediate consequence operator $( T _ { P } )$ Given a current interpretation, $T _ { P }$ evaluates all rule bodies simultaneously against that interpretation to derive a new set of facts. To operationalize MT-PDCL, we must generalize this concept to act natively upon the space of continuous probabilities.

Let $\mathcal { P }$ denote the space of probability functions mapping ground facts to [0, 1]. Because the domain D contains continuous variables, the set of possible ground facts is uncountably infinite. Thus, a function $P \in \mathcal { P }$ represents a continuous probability surface rather than a discrete table. We define the generic continuous probabilistic consequence operator $\mathbb { T } : \mathcal { P }  \mathcal { P }$ . This operator takes an input probability function $\bar { P }$ from iteration k and produces an updated function $\mathbb { T } ( P )$ for iteration $k ^ { - } + 1$ by systematically applying the generative rules in Φ.

To construct T, we must define what constitutes a rule application. In discrete logic, the forward inference step relies on Generalized Modus Ponens. In MT-PDCL, we define a rule instance as the application of Generalized Modus Ponens for a clause $R _ { j } ~ \in ~ \Phi$ anchored to a specific continuous variable binding $\mathbf { v } \in \Omega _ { V _ { i } }$ Here, $\Omega _ { V _ { i } }$ represents the joint measurable domain of all variables (both deterministic indices and stochastic outputs) evaluated in the rule body. While we generally refer to this space as continuous, measure theory seamlessly supports discrete probability generators $( \mathrm { e . g . }$ , categorical distributions) as a special case by treating them as discrete measures (point masses) over the Borel space. Thus, for continuous variables, a single rule represents an uncountably infinite number of instances, whereas for discrete variables, the measure natively reduces this to a finite or countable set of instances.

Consider a specific rule instance for $R _ { j }$ evaluated at binding $\mathbf { v } ,$ where $R _ { j }$ is of the form $p _ { j } \ : : \ H  B .$ Under generative semantics, the rule generates the head if and only if the body is true and the causal event $E _ { j }$ succeeds. The independence of $E _ { j }$ from the stochastic environment and all prior logical derivations dictates that the probability of this joint occurrence is the product of their individual probabilities. Therefore, if the body B currently evaluates to a probability under the input function $P$ (denoted $P ( B \mid \mathbf { v } ) )$ for this specific binding, and the causal event succeeds with probability $p _ { j }$ , the isolated, intermediate probabilistic contribution of this specific rule instance to the head H is:

$$
P _ { \mathrm { u p d a t e } } ( H \mid R _ { j } , \mathbf { v } ) = p _ { j } \cdot P ( B \mid \mathbf { v } )
$$

For a standard deterministic clause $( 1 . 0 : \ u { H }  \ u { B } )$ , the parameter is $p _ { j } = 1 . 0$ , meaning the exact probability of the body is propagated directly to the head:

$$
P _ { \mathrm { u p d a t e } } ( H \mid R _ { j } , \mathbf { v } ) = 1 . 0 \cdot P ( B \mid \mathbf { v } )
$$

If H is the head of only one rule, and only a single valid binding v satisfies the body, the final updated probability for the head in iteration $k { + 1 }$ is this isolated computation. However, logic programs inherently utilize multiple overlapping derivations. A knowledge base frequently contains multiple distinct rules that derive the same relation H, and a single rule generates the exact same ground head fact across a continuous domain of variables.

To ensure $\mathbb { T }$ outputs a single, unified probability for H, we must formalize an aggregation mechanism that safely combines the probability masses from multiple successful rule applications.

## 5.2 Exact Aggregation via Continuous Unions

Under discrete Distribution Semantics (such as in ProbLog), if multiple rule instances derive the same head fact H, the evaluation uses logical disjunction: H is true if any of its derivations hold. Simply adding individual probabilities would overcount, as derivations often overlap by sharing subgoals or variables. Calculating the exact probability therefore requires the inclusion-exclusion principle: $P ( A { \bar { \lor } } B ) { \bar { = } } P ( A ) + P ( B ) - P ( A \land B )$

In MT-PDCL, we must adapt this disjunctive aggregation to operate natively over our continuous possible worlds Ω. Let $\mathcal { R } _ { H } \subseteq \Phi$ be the finite set of rules whose heads unify with a specific ground instance of a fact H. We define the exact consequence operator $\mathbb { T } _ { \mathrm { e x a c t } }$ . The exact updated probability for H produced by $\mathbb { T } _ { \mathrm { e x a c t } }$ is the Lebesgue measure of the infinite geometric union of success regions across all rules and their continuous evaluation domains:

$$
\mathbb { T } _ { \mathrm { e x a c t } } ( P ) ( H ) = \mu _ { \Omega } \left( \bigcup _ { R _ { j } \in \mathcal { R } _ { H } } \bigcup _ { \mathbf { v } \in \Omega _ { V _ { j } } } \Omega _ { \mathrm { s u c c e s s } } ( R _ { j } , \mathbf { v } ) \right)
$$

where $\Omega _ { \mathrm { s u c c e s s } } ( R _ { i } , { \bf v } ) \subseteq \Omega$ represents the specific subset of possible worlds where the instance $( R _ { j } , \mathbf { v } )$ successfully triggers (i.e., the body is logically true in the world’s least model, and the causal event $E _ { j }$ evaluates to True).

This geometric union is the direct measure-theoretic equivalent of the existential quantifier applied to Generalized Modus Ponens: H is true if there exists any rule and any valid variable binding that derives it. When the variables are discrete, this continuous formulation mathematically collapses back to the standard countable union used in discrete probabilistic logic.

While this formula defines the exact declarative truth, applying the inclusion-exclusion principle to compute the exact geometric volume of an uncountably infinite union of overlapping continuous sets is computationally intractable.

## Remark: Iteration and the Fixed Point.

Because logic programs utilize chained and recursive rules, a single application of the consequence operator rarely captures the complete probability mass. Instead, the operator is applied iteratively, creating a sequence of probability functions $P _ { 0 } , P _ { 1 } , P _ { 2 } , . . .$ . starting from a baseline where all facts have zero probability $( P _ { 0 } \overset { \cdot } { = } P _ { \perp } \overset { \cdot } { ) }$ . The iteration stops when a stable state is reached where evaluating the rules yields no further changes, meaning $\mathbb { T } _ { \mathrm { e x a c t } } ( P _ { k } ) = P _ { k }$ . This stable state is called the least fixed point (denoted $l f p )$ , and it represents the final, logically entailed probabilities of all facts. We formally prove the existence of and convergence to this $l f p$ in Section 5.5, but we introduce the basic mechanism here to evaluate the upcoming example.

## 5.3 Example: Exact Continuous Aggregation

To illustrate the mechanics of $\mathbb { T } _ { \mathrm { e x a c t } }$ , we define a simple knowledge base where a target exists across a 10km domain.   
Two independent sensors attempt to measure its location, generating overlapping continuous rule instances.

## The Knowledge Base K:

1. target $\mathbf { \mathrm { { \_ { l o c } } } } ( ; X ) \sim \mathbf { \mathrm { { U n i f o r m } } } ( 0 , 1 0 )$

(Measure declaration: generates a target location X, distributed uniformly.)

2. $1 . 0 : \operatorname { t a r g e t } ( X )  \operatorname { t a r g e t } \_ { \operatorname { l o c } ( ; X ) } .$

(Rule 1: Deterministically places the target at the generated location.)

3. $0 . 6 : \operatorname { t r a c k }  \operatorname { t a r g e t } ( X ) \land X \in [ 2 , 6 ] .$

(Rule 2: Sensor A detects the target with 0.6 probability if $X \in [ 2 , 6 ] . )$

4. $0 . 8 : : \mathrm { t r a c k }  \mathrm { t a r g e t } ( X ) \land X \in [ 5 , 9 ] .$

(Rule 3: Sensor B detects the target with 0.8 probability $i f X \in [ 5 , 9 ] . )$

5. $1 . 0 : \mathrm { v e r i f i e d }  \mathrm { t r a c k } .$

(Rule 4: A derived track is immediately verified.)

We apply $\mathbb { T } _ { \mathrm { e x a c t } }$ iteratively, starting from $P _ { 0 }$ , where all derived facts have zero probability.

## Step 1: Establishing the target location $( P _ { 1 } )$

Rule 1 evaluates directly against the mathematical measure $\mathcal { M } _ { s } .$ It unconditionally assigns a probability of 1.0 to the target’s existence for any specific location $x \in [ 0 , 1 0 ]$ provided by the environment, establishing the continuous prior density $d \mu _ { s } ( x ) = 0 . 1 \overset { \cdot } { d x }$

## Step 2: Taking measurements and aggregating derivations $( P _ { 2 } )$

Rules 2 and 3 evaluate against $P _ { 1 }$ . Because both sensors measure the exact same target location x, their logical derivations overlap. Assuming the sensors’ physical causal events operate independently for any specific location x, we apply the exact point-wise inclusion-exclusion principle: $P ( A \lor \mathbf { \dot { B } } ) = P ( A { \dot { ) } } + P ( \Vec { B } ) - P ( \ u , A \land B )$

We partition the continuous evaluation space of the shared variable X to compute the exact aggregated measure for track:

• Interval $x \in [ 2 , 5 ) \colon$ Only Sensor A successfully measures the target.

$$
\int _ { 2 } ^ { 5 } 0 . 6 \cdot 0 . 1 d x = 0 . 6 \cdot 0 . 3 = 0 . 1 8
$$

• Interval $x \in [ 5 , 6 ] ;$ : Both sensors successfully derive the head. Because the causal events are independent, the combined probability of success at point x is $0 . 6 + 0 . 8 - ( 0 . 6 \cdot 0 . 8 ) = 0 . 9 2 .$

$$
\int _ { 5 } ^ { 6 } 0 . 9 2 \cdot 0 . 1 d x = 0 . 9 2 \cdot 0 . 1 = 0 . 0 9 2
$$

• Interval $x \in ( 6 , 9 ] ;$ : Only Sensor B successfully measures the target.

$$
\int _ { 6 } ^ { 9 } 0 . 8 \cdot 0 . 1 d x = 0 . 8 \cdot 0 . 3 = 0 . 2 4
$$

The uncountably infinite union from the $\mathbb { T } _ { \mathrm { e x a c t } }$ definition integrates these point-wise geometric probabilities over the shared target measure:

$$
P _ { 2 } ( \mathrm { t r a c k } ) = 0 . 1 8 + 0 . 0 9 2 + 0 . 2 4 = 0 . 5 1 2
$$

## Step 3 & 4: Reaching the fixed point

Rule 4 propagates the track probability directly $( 1 . 0 \cdot 0 . 5 1 2 = 0 . 5 1 2 )$ . Evaluating $\Phi$ again yields no new derivations.

The operator has reached its fixed point. This computed exact value identically matches the declarative entailment defined by the Lebesgue integral over the possible worlds space in Section 3.7:

$$
l f p ( \mathbb { T } _ { \mathrm { e x a c t } } ) ( \mathrm { v e r i f i e d } ) = P _ { \mathcal { K } } ( \mathrm { v e r i f i e d } ) = 0 . 5 1 2
$$

## Remark on Computational Tractability:

This example demonstrates why exact aggregation is generally intractable. Here, the overlap could be solved analytically because the regions were one-dimensional and uniform. In a general knowledge base with multi-dimensional continuous variables and deep recursion, computing the exact intersection boundaries $\Omega _ { \mathrm { s u c c e s s } } ( R _ { 1 } ) \cap \Omega _ { \mathrm { s u c c e s s } } ( R _ { 2 } )$ to perform exact continuous inclusion-exclusion is practically impossible.

## 5.4 Operationalizing Aggregation: Continuous Noisy-OR

To make continuous aggregation computable, we introduce a structural approximation regarding how multiple derivations interact. A standard approach in probabilistic logic programming assumes that multiple rules supporting the same head fact operate independently.

In the AI literature, this is known as the Independent Causal Influence (ICI) assumption [Pea88, HB96]. Mathematically, it assumes the conditional independence of rule failures.

Under ICI, rather than calculating the intractable infinite union of successful geometric derivations, we calculate the intersection of their failures. Because rule failures are assumed conditionally independent, this intersection simplifies to the strict product of their failure probabilities. Fact H is considered true if at least one derivation succeeds, computed as 1 minus the probability that all derivations fail.

For a single rule $R _ { j }$ , the probability that a specific instance evaluated at continuous binding v fails to generate the head is $1 - \bar { P _ { \mathrm { u p d a t e } } } ( H | \bar { \mathbf { \xi } } R _ { j } , \bar { \mathbf { v } } )$ . When a rule evaluates over a continuous space $\Omega _ { V _ { i } }$ , it represents an uncountably infinite family of instances. To calculate their joint failure, we must generalize the discrete product over this continuous domain.

Taking the logarithm of a product transforms it into a sum: ln $( \prod ( 1 - p ) ) = \sum \ln ( 1 - p )$ . In the continuous limit, this sum translates into a Lebesgue integral over the continuous evaluation domain. Applying the exponential function recovers the actual probability.

Thus, the total assumed probability that the entire continuous family of rule instances for $R _ { j }$ fails is computed over its bound domain $\Omega _ { V _ { i } }$ with base measure $\mu _ { V } \colon$

$$
P _ { \mathrm { f a i l } } ( R _ { j } ) = \exp \left( \int _ { \Omega _ { V _ { j } } } \ln \left( 1 - P _ { \mathrm { u p d a t e } } ( H \mid R _ { j } , \mathbf { v } ) \right) d \mu _ { V } ( \mathbf { v } ) \right)
$$

To compute the final, unified probability for the head fact H under the ICI assumption, the operational consequence operator $\mathbb { T } _ { K }$ multiplies these continuous failure probabilities across all m rules in $\mathcal { R } _ { H }$ , and subtracts the result from 1:

$$
\mathbb { T } _ { \mathcal { K } } ( P ) ( H ) = 1 - \prod _ { R _ { j } \in \mathcal { R } _ { H } } P _ { \mathrm { f a i l } } ( R _ { j } )
$$

## Remark: Deterministic Bounds and the Logarithmic Singularity.

When evaluating a fully deterministic derivation where the update probability reaches 1.0, the failure probability is zero, resulting in a logarithmic singularity: $\ln ( 0 )  - \infty$ . Integrating −∞ over a subset of positive measure causes the integral to diverge $\mathrm { t o } - \infty$ . The outer exponential evaluates to $\exp ( - \infty ) = 0$ , producing a final Noisy-OR success probability of 1.0. This perfectly preserves logical semantics: if any continuous subset of derivations is guaranteed to succeed, the aggregated conclusion is unconditionally true.

To see how this approximation changes the computation, we revisit the multi-sensor tracking example from Section 5.3. In the exact calculation, we had to manually partition the continuous domain into distinct intervals ([2, 5), [5, 6], and (6, 9]) to correctly calculate the geometric overlap between the two sensors.

The continuous Noisy-OR operator avoids tracking these complex intersections entirely. Instead, it computes the joint failure probability for each rule independently across its full spatial bound using the product integral, and then combines the results.

## Recalculating Step 2: Aggregation via Continuous Noisy-OR

For Rule 2 (Sensor A, measuring over $x \in [ 2 , 6 ] )$

$$
P _ { \mathrm { f a i l } } ( R _ { 2 } ) = \exp \left( \int _ { 2 } ^ { 6 } \ln ( 1 - 0 . 6 ) \cdot 0 . 1 d x \right) = \exp ( 0 . 4 \cdot \ln ( 0 . 4 ) ) \approx 0 . 6 9 3
$$

For Rule 3 (Sensor B, measuring over $x \in [ 5 , 9 ] )$

$$
P _ { \mathrm { f a i l } } ( R _ { 3 } ) = \exp \left( \int _ { 5 } ^ { 9 } \ln ( 1 - 0 . 8 ) \cdot 0 . 1 d x \right) = \exp ( 0 . 4 \cdot \ln ( 0 . 2 ) ) \approx 0 . 5 2 5
$$

The continuous Noisy-OR operator aggregates these independent rule failures to update the head fact:

$$
P _ { 2 } ( \mathrm { t r a c k } ) = 1 - ( 0 . 6 9 3 \cdot 0 . 5 2 5 ) \approx 0 . 6 3 6
$$

Propagating to the fixed point yields the operational approximation: $l f p ( \mathbb { T } _ { K } ) ( \mathrm { v e r i f i e d } ) = P _ { o p } ( \mathrm { v e r i f i e d } ) \approx 0 . 6 3 6 .$

Theoretical Insight: Divergence of Results.

The continuous Noisy-OR approximation (0.636) overestimates the exact measure-theoretic computation (0.512). This divergence highlights what the ICI assumption sacrifices.

The exact computation correctly treats the continuous variable X as a single, physical entity: the target’s locations are mutually exclusive. A single target can exist at $x _ { 1 }$ or $x _ { 2 }$ , but never both simultaneously.

Conversely, the continuous Noisy-OR operator assumes conditional independence across the entire integration domain. By using the exp  R ln formula, it ignores the physical constraint of mutual exclusivity. Mathematically, it treats every point x in the continuous space as a separate, independent target capable of triggering the rule. Because it evaluates the domain as a collection of independent events rather than a single mutually exclusive variable, it computes a physically incorrect overestimate.

This establishes a critical theoretical boundary: the continuous Noisy-OR operator is a tractable, differentiable algebraic proxy for exact logical inclusion-exclusion, but it sacrifices accuracy when variables represent mutually exclusive physical states.

## 5.5 Monotonicity and Convergence

In standard logic programming, the semantics of the consequence operator rely on two fundamental theorems: the Knaster-Tarski theorem, which guarantees the existence of a least fixed point $( l f p )$ for monotonic operators; and Kleene’s fixed-point theorem, which guarantees that iterative evaluation converges to this $l f p ,$ provided the operator is continuous [Llo87]. We establish a complete lattice over $\mathcal { P } _ { \cdot }$ , the space of continuous probability functions, to apply these properties to MT-PDCL.

Let $P _ { 1 } , P _ { 2 } \in \mathcal { P }$ . We define a partial order $\sqsubseteq$ such that $P _ { 1 } \subseteq P _ { 2 }$ if and only if for every ground fact $H , P _ { 1 } ( H ) \leq$ $P _ { 2 } ( H )$

The space $\mathcal { P }$ is bounded below by the zero function $P _ { \perp }$ (where $P _ { \perp } ( H ) = 0$ for all facts) and bounded above by the unit function $P _ { \top }$ (where $P _ { \neg } ( H ) \mathop { = } 1$ for all facts).

To form a complete lattice, every subset $S \subseteq { \mathcal { P } }$ must have a least upper bound (supremum) and a greatest lower bound (infimum) in $\bar { \mathcal { P } } .$ We define these point-wise. For any ground fact $\bar { H } ;$

$$
( \operatorname* { s u p } S ) ( H ) = \operatorname* { s u p } _ { P \in S } P ( H )
$$

$$
( \operatorname* { i n f } S ) ( H ) = \operatorname* { i n f } _ { P \in S } P ( H )
$$

The real interval [0, 1] is a complete lattice, guaranteeing that the supremum and infimum of any set of values in $[ 0 , 1 ]$ always exist and remain within [0, 1]. The functions sup S and inf $\dot { S }$ therefore exist and belong to $\mathcal { P } _ { \cdot }$ . Thus, the space of functions $\mathcal { P }$ under the point-wise ordering $\sqsubseteq$ forms a complete lattice.

Proposition 1 (Monotonicity and Convergence of T). Both the exact operator $\mathbb { T } _ { e x a c t }$ and the continuous Noisy-OR operator $\mathbb { T } _ { \mathcal { K } }$ are monotonically increasing over $\mathcal { P } .$ Consequently, they possess a least fixed point, and their iterative application startingfrom the zerofunction $P _ { \perp }$ converges to it.

Proof. Part 1: Monotonicity and Existence. Let $P _ { 1 } , P _ { 2 } \in \mathcal { P }$ such that $P _ { 1 } \subseteq P _ { 2 }$ . By definition, $P _ { 1 } ( H ) \leq P _ { 2 } ( H )$ for all facts H. MT-PDCL evaluates definite clauses, ensuring the probability of any conjunction of positive literals evaluates monotonically. Thus, for any rule body B evaluated at binding v, we have $\mathbf { \dot { \textit { P } } } _ { 1 } ( \boldsymbol { \dot { B ^ { \intercal } } } | \mathbf { v } ) \leq P _ { 2 } ( \mathbf { \dot { \textit { B } } } | \mathbf { v } )$

For $\mathbb { T } _ { \mathrm { e x a c t } }$ , an increase in body probabilities expands the geometric subsets of successful possible worlds. Therefore, the success regions generated under $P _ { 1 }$ are subsets of those generated under $P _ { 2 }$ . The Lebesgue measure of a superset is strictly non-decreasing, ensuring $\mathbb { T } _ { \mathrm { e x a c t } } ( P _ { 1 } ) \subseteq \mathbb { T } _ { \mathrm { e x a c t } } ( P _ { 2 } )$

For $\mathbb { T } _ { \mathcal { K } }$ , the isolated update probability $P _ { \mathrm { u p d a t e } } ( H \mid R _ { j } , \mathbf { v } ) = p _ { j } \cdot P ( B \mid \mathbf { v } )$ increases under $P _ { 2 }$ . Consequently, the local failure probability $( 1 - P _ { \mathrm { u p d a t e } } )$ decreases. The logarithm of a smaller positive fraction yields a more negative value, making the resulting integral more negative, and its exponential closer to 0. Subtracting this smaller exponential from 1 yields a larger aggregated probability for the head. Thus, $\mathbb { T } _ { \mathcal { K } } ( P _ { 1 } ) \subseteq \mathbb { T } _ { \mathcal { K } } ( P _ { 2 } )$

Because both operators are monotonic over a complete lattice, the Knaster-Tarski theorem guarantees the existence of $\mathrm { ~ a ~ } l f p$

Part $2 \colon \omega { \mathrm { - } } C o n t i n u i t y$ and Convergence. To guarantee that iterative application from $P _ { \perp }$ reaches the $l f p .$ , the operators must be ω-continuous, meaning they preserve the suprema of ascending chains $( \mathrm { i . e . , \mathbb { T } } ( \operatorname* { s u p } _ { k } P _ { k } ) \overset { \vartriangle } { = } \operatorname* { s u p } _ { k } \mathrm { \bar { \mathbb { T } } } ( P _ { k } ) )$ The exact operator $\mathbb { T } _ { \mathrm { e x a c t } }$ is constructed from standard probability measures, which are inherently continuous from below [Bil12]. The operational operator $\mathbb { T } _ { \mathcal { K } }$ is composed entirely of continuous algebraic operations (integration, logarithms, and exponentials) over a bounded space. Therefore, both operators are ω-continuous. By Kleene’s fixedpoint theorem, the application of an ω-continuous, monotonic operator starting from the bottom element $P _ { \perp }$ converges to the $l f p .$ □

## 5.6 Iteration Bounds and the Contraction Mapping

While Kleene’s theorem guarantees that iterative evaluation will eventually converge to the true fixed point, it does not provide a computable bound on the number of iterations required. To estimate the number of computational steps needed for a given precision, we must analyze the operators using the Banach Fixed-Point Theorem (Contraction Mapping Theorem) [Ban22].

To do this, we treat $\mathcal { P }$ as a metric space rather than just a partial order. We define the distance between two probability functions $P _ { a } , P _ { b } \in \mathcal { P }$ using the supremum norm:

$$
d ( P _ { a } , P _ { b } ) = \operatorname* { s u p } _ { H } | P _ { a } ( H ) - P _ { b } ( H ) |
$$

This metric measures the maximum probability difference for any ground fact across the entire continuous domain.

To apply the Banach theorem, an operator must be a strict contraction, meaning there exists a Lipschitz constant $L < 1$ such that $d ( \mathbb { T } ( P _ { a } ) , \mathbb { T } ( P _ { b } ) ) \leq L \cdot { \dot { d } } ( P _ { a } , P _ { b } )$

In MT-PDCL, the contraction property depends entirely on the recursive structure of the logic program. To formally identify recursive rules, we define the program’s dependency graph, where nodes are predicates and a directed edge exists from predicate H to predicate B if a rule derives H and contains B in its body. Let $\mathcal { R } _ { \mathrm { c y c l e } } \subseteq \Phi$ be the set of all rules whose head and at least one body literal belong to the same Strongly Connected Component (SCC). We refer to any predicate belonging to such an ${ \mathrm { S C C } } ( { \mathrm { i . e . } }$ , the head of any rule in $\mathcal { R } _ { \mathrm { c y c l e } } )$ as a cyclic predicate.

To evaluate if the consequence operator T acts as a contraction mapping, we first formally define a global bound on the recursive derivations.

Definition 1 (Recursive Measure $\gamma ) .$ . Let the expected number of successful derivations for a cyclic ground fact be the continuous integral of its causal probabilities. We define $\gamma$ as the supremum of this measure across all possible ground instances of any cyclic predicate:

$$
\gamma = \underset { \mathbf { h } } { \operatorname* { s u p } } \sum _ { R _ { j } \in \mathcal { R } _ { \mathrm { c y c l e } } } \int _ { \Omega _ { V _ { j } } } p _ { j } ( \mathbf { v } ) d \mu _ { V } ( \mathbf { v } )
$$

Proposition 2 (Lipschitz Bound and Contraction Regimes). Under the continuous Independent Causal Influence (ICI) assumption, the consequence operator T satisfies the Lipschitz condition:

$$
d ( \mathbb { T } ( P _ { a } ) , \mathbb { T } ( P _ { b } ) ) \leq \gamma \cdot d ( P _ { a } , P _ { b } )
$$

Consequently, the operatorfalls into one oftwo iteration regimes:

1. Strict Contraction $( \gamma < 1 ) .$ : The operator T is a strict contraction mapping with a Lipschitz constant $L = \gamma$ ensuring geometric convergence to the exact leastfixed point.

2. Expansive Propagation $( \gamma \geq 1 ) \colon$ The Lipschitz bound is $L \geq 1$ . The mapping is not a contraction, preventing the calculation ofa strict Banach iteration bound.

Proof. Let $\begin{array} { r } { \Delta = d ( P _ { a } , P _ { b } ) = \operatorname* { s u p } _ { \mathbf { v } } | P _ { a } ( \mathbf { v } ) - P _ { b } ( \mathbf { v } ) | } \end{array}$ . We parameterize a straight-line path between the two functions for $t \in [ 0 , 1 ]$ such that $P _ { t } = ( 1 - \dot { t } ) P _ { a } + t P _ { b }$

For a specific ground instance of a cyclic predicate, let $F ( t )$ be its operational update under the interpolated probability $P _ { t }$ . Using the continuous Noisy-OR aggregation, $F ( t )$ is explicitly defined as:

$$
F ( t ) = 1 - E _ { 0 } \cdot \exp \left( \sum _ { R _ { j } \in \mathcal { R } _ { \mathrm { s y c l e } } } \int _ { \Omega _ { V _ { j } } } \ln \left( 1 - P _ { t } ( \mathbf { v } ) \cdot p _ { j } ( \mathbf { v } ) \right) d \mu _ { V } ( \mathbf { v } ) \right)
$$

where $E _ { 0 } \in [ 0 , 1 ]$ represents the constant joint failure probability from any non-cyclic base rules deriving this fact. Taking the derivative of $F ( t )$ with respect to t using the chain rule, and factoring out the supremum distance $\Delta \ge$ $| P _ { b } ( { \bf v } ) - P _ { a } ( { \bf v } ) |$ |, yields:

$$
| F ^ { \prime } ( t ) | \leq \Delta \cdot ( 1 - F ( t ) ) \sum _ { \substack { R _ { j } \in \mathcal { R } _ { \mathrm { c y c l e } } } } \int _ { \Omega _ { V _ { j } } } \frac { p _ { j } ( \mathbf { v } ) } { 1 - P _ { t } ( \mathbf { v } ) \cdot p _ { j } ( \mathbf { v } ) } d \mu _ { V } ( \mathbf { v } )
$$

To resolve this bound, note that $\begin{array} { r } { ( 1 - F ( t ) ) \le \exp \left( \sum _ { R _ { j } } \int _ { \Omega _ { V _ { i } } } \ln \left( 1 - P _ { t } ( \mathbf { v } ) p _ { j } ( \mathbf { v } ) \right) d \mu _ { V } ( \mathbf { v } ) \right) } \end{array}$ . Substituting this into the inequality gives:

$$
| F ^ { \prime } ( t ) | \leq \Delta \cdot \left[ \exp \left( \sum _ { R _ { j } } \int _ { \Omega _ { V _ { j } } } \ln \left( 1 - P _ { t } ( \mathbf v ) p _ { j } ( \mathbf v ) \right) d \mu _ { V } ( \mathbf v ) \right) \cdot \sum _ { R _ { j } } \int _ { \Omega _ { V _ { j } } } \frac { p _ { j } ( \mathbf v ) } { 1 - P _ { t } ( \mathbf v ) p _ { j } ( \mathbf v ) } d \mu _ { V } ( \mathbf v ) \right]
$$

To rigorously bridge this continuous exponential-logarithmic formulation to its discrete algebraic counterpart, we rely on the framework of the Volterra product integral [DF79, Sla07]. In the standard theory of product integration, the exponential-logarithmic expression exp $\textstyle \left( \int \ln ( \bar { 1 } - \dot { f } ) d \mu \right)$ is the Lebesgue measure-theoretic representation of the continuous product integral $\textstyle \prod _ { \Omega } ( 1 - f d \mu )$

By the continuous product rule within this framework, the Fréchet derivative of the product integral structurally mirrors the discrete identity $\begin{array} { r } { \big ( \prod _ { k } ( 1 - x _ { k } ) \big ) \sum _ { i } \frac { c _ { i } } { 1 - x _ { i } } = \sum _ { i } c _ { i } \prod _ { k \neq i } ( 1 - x _ { k } ) } \end{array}$ . In our formulation, the evaluated probability $P _ { t } ( \mathbf { v } ) p _ { j } ( \mathbf { v } )$ acts as the term $x _ { k } .$ , and the bare causal probability $p _ { j } ( \mathbf { v } )$ acts as the numerator $c _ { i }$ . Just as the discrete product algebraically bounds the denominators to yield a strict upper limit of $\sum _ { i } c _ { i }$ (since $1 - x _ { k } \le 1 )$ , the continuous exponential factor identically bounds the denominators of the logarithmic derivative. This formal equivalence reduces the entire bounded expression to the bare integral of the numerator:

$$
| F ^ { \prime } ( t ) | \leq \Delta \sum _ { R _ { j } \in \mathcal { R } _ { \mathrm { c y c l e } } } \int _ { \Omega _ { V _ { j } } } p _ { j } ( \mathbf { v } ) d \mu _ { V } ( \mathbf { v } )
$$

By Definition 1, this integral sum across the cyclic rules is bounded by the global supremum γ. Therefore:

$$
| F ^ { \prime } ( t ) | \le \Delta \cdot \gamma
$$

Integrating $| F ^ { \prime } ( t )$ | from $t = 0 \mathrm { t o } 1$ evaluates to $| \mathbb { T } ( P _ { b } ) - \mathbb { T } ( P _ { a } ) | \le \gamma \cdot \Delta$ . Taking the supremum across all ground facts guarantees the global bound. □

Proposition 3 (Iteration Bound for Strict Contractions). Ifthe consequence operator acts as a strict contraction with constant $L < 1$ , the number of iterations N required to guarantee that the operational probability function $P _ { N }$ is within a precision ϵ ofthe exact leastfixed point across allfacts is bounded by:

$$
N \geq \frac { \ln ( \epsilon \cdot ( 1 - L ) ) - \ln ( d ( P _ { 1 } , P _ { 0 } ) ) } { \ln ( L ) }
$$

Proof. By the Banach Fixed-Point Theorem, the error bound after N iterations for a strict contraction mapping is bounded by the initial step size:

$$
d ( P _ { N } , l f p ) \leq { \frac { L ^ { N } } { 1 - L } } d ( P _ { 1 } , P _ { 0 } )
$$

We require the maximum error to be less than or equal to our precision threshold ϵ:

$$
\frac { L ^ { N } } { 1 - L } d ( P _ { 1 } , P _ { 0 } ) \leq \epsilon
$$

<sup>Rearranging</sup> <sup>the</sup> <sup>terms</sup> <sup>and</sup> <sup>isolating</sup> <sup>N</sup> <sup>using</sup> <sup>the</sup> <sup>natural</sup> <sup>logarithm</sup> <sup>yields</sup> <sup>the</sup> <sup>required</sup> <sup>number</sup> <sup>of</sup> <sup>steps.</sup> <sup>Because</sup>ln(L) is negative, reversing the inequality. $L < 1$

## The Necessity of the Knaster-Kleene Approach for $\gamma \geq 1$

When a program contains deterministic recursive rules or continuous structural overlaps $( \gamma \geq 1 )$ , the Banach theorem fails. Because the operator is locally expansive, we cannot calculate a strict iteration bound.

To illustrate this behavior, consider a knowledge base where a target activates spontaneously, and that activation spreads deterministically to any continuous location within a 1-meter radius.

## The Knowledge Base K:

1. source\_loc $( ; X ) \sim \operatorname { U n i f o r m } ( 0 , 1 0 ) .$

(Rule 1: Generates a continuous spatial variable $X . )$

2. $0 . 5 : : \mathsf { a c t i v e } ( X ) \gets \mathsf { s o u r c e \_ l o c } ( ; X ) .$

(Rule 2: Base case. A location activates spontaneously with 0.5 probability.)

3. $1 . 0 : \mathsf { a c t i v e } ( X ) \gets \mathsf { a c t i v e } ( Y ) \wedge | X - Y | \leq 1 .$

(Rule 3: Spatial recursion. Activation spreadsfrom any point Y within distance 1.)

To verify the bounds condition, we first calculate the local aggregated measure, denoted $\gamma ( x )$ , for each specific ground fact active(x) evaluated by the recursive cycle (Rule 3). This local measure is the integral of the rule probability over the valid neighborhood of x, restricted by the domain [0, 10]:

$$
\gamma ( x ) = \int _ { \operatorname* { m a x } ( 0 , x - 1 ) } ^ { \operatorname* { m i n } ( 1 0 , x + 1 ) } 1 . 0 d y
$$

The value of $\gamma ( x )$ depends directly on the location x:

• For any point $x \in [ 0 , 1 ]$ , the integration interval is $[ 0 , x + 1 ]$ , yielding $\gamma ( x ) = x + 1$

• For any point $x \in [ 1 , 9 ]$ , integration interval is $[ x - 1 , x + 1 ]$ , yielding $\gamma ( x ) = 2$

• For any point $x \in [ 9 , 1 0 ]$ , the integration interval is $[ x - 1 , 1 0 ]$ , yielding $\gamma ( x ) = 1 1 - x .$

Because the metric space distance is defined by the supremum norm, the global constant γ is the maximum of all these local values:

$$
\gamma = \operatorname* { s u p } _ { x \in [ 0 , 1 0 ] } \gamma ( x ) = 2
$$

A global bound of $\gamma \geq 1$ renders the operator expansive. To observe this mathematically, we apply $\mathbb { T } _ { \mathcal { K } }$ iteratively, starting from a continuous surface where $P _ { 0 } ( \mathrm { a c t i v e } ( x ) ) = 0$

To compute the updated probability for a ground fact active(x), we aggregate the contributions of Rule 2 and Rule 3. Under the continuous ICI assumption, the total failure probability of the head fact is the product of the independent failure probabilities of the rules deriving it:

$$
P _ { k + 1 } ( \mathrm { a c t i v e } ( x ) ) = 1 - \Big ( P _ { \mathrm { f a i l } } ( R _ { 2 } ) \cdot P _ { \mathrm { f a i l } } ( R _ { 3 } ) \Big )
$$

The base rule failure is constant across all iterations: $P _ { \mathrm { f a i l } } ( R _ { 2 } ) = 1 - 0 . 5 = 0 . 5$ . The recursive failure is computed dynamically using the continuous Noisy-OR integral:

$$
P _ { \mathrm { f a i l } } ( R _ { 3 } ) = \exp \left( \int _ { x - 1 } ^ { x + 1 } \ln \left( 1 - P _ { k } ( \mathrm { a c t i v e } ( y ) ) \right) d y \right)
$$

For simplicity in tracing the values, we compare an unconstrained point away from the edges $( \mathbf { e . g . } , x = 5$ , where the full integration window [4, 6] has length 2) against a point right at the edge of the domain $( \mathbf { e . g . } , x = 0$ , where the integration window is truncated to [0, 1], giving a length of 1).

• Iteration 1 $( P _ { 1 } ) \colon$

Evaluating against $P _ { 0 } = 0$ , the recursive failure evaluates to exp(0) = 1.0 everywhere.

$$
P _ { 1 } ( 5 ) = P _ { 1 } ( 0 ) = 1 - ( 0 . 5 \cdot 1 . 0 ) = 0 . 5
$$

## • Iteration $\mathbf { 2 } \left( P _ { 2 } \right) :$

The recursive rule integrates over the continuous neighborhood where $P _ { 1 } = 0 . 5$ . Because the spatial windows differ, the updated probabilities diverge:

$$
P _ { 2 } ( 5 ) = 1 - 0 . 5 \cdot \exp \left( 2 \cdot \ln ( 0 . 5 ) \right) = 1 - 0 . 5 ( 0 . 2 5 ) = 0 . 8 7 5
$$

$$
P _ { 2 } ( 0 ) = 1 - 0 . 5 \cdot \exp \Big ( 1 \cdot \ln ( 0 . 5 ) \Big ) = 1 - 0 . 5 ( 0 . 5 ) = 0 . 7 5
$$

• Iteration ${ \bf 3 } \left( P _ { 3 } \right) :$

The recursive rule integrates over the neighborhood where $P _ { 2 }$ varies. To bound the boundary point, we evaluate using the minimum probability in its window (0.75):

$$
\begin{array} { r l } & { P _ { 3 } ( 5 ) \geq 1 - 0 . 5 \cdot \exp \Big ( 2 \cdot \ln ( 1 - 0 . 8 7 5 ) \Big ) \approx 0 . 9 9 2 2 } \\ & { P _ { 3 } ( 0 ) \geq 1 - 0 . 5 \cdot \exp \Big ( 1 \cdot \ln ( 1 - 0 . 7 5 ) \Big ) = 1 - 0 . 5 ( 0 . 2 5 ) = 0 . 8 7 5 } \end{array}
$$

While the probability curve $" s a g s "$ at the boundaries during finite iterations due to the truncated spatial windows, the entire surface ultimately converges to 1. We can prove this by tracking the absolute minimum probability across the domain, which always occurs at the boundaries $\bar { ( m _ { k } = P _ { k } ( 0 ) = P _ { k } ( 1 0 ) ) }$ ). The sequence of minimums follows the recurrence:

$$
m _ { k + 1 } = 1 - 0 . 5 ( 1 - m _ { k } )
$$

Starting from $m _ { 0 } = 0 ,$ , this sequence evaluates explicitly to $m _ { k } = 1 - ( 0 . 5 ) ^ { k + 1 } . \mathrm { A s } \ k \to \infty$ , the minimum boundary probability $m _ { k } \to 1$ . Since the probability anywhere in the space is bounded from below by the boundaries $( P _ { k } ( x ) \dot { \geq }$ $m _ { k } )$ , the exact least fixed point is uniform across the space: ${ \bar { P } } ^ { * } ( x ) = 1$ for all $x \in [ 0 , 1 0 ]$

This behavior demonstrates why the order-theoretic foundation built in Section 5.5 is mandatory. When the expected number of derivations forces the metric distance between functions to expand and deform unpredictably across space, standard iteration bounds fail. Knaster and Kleene require only monotonicity and continuity, guaranteeing that even when an expansive slope forces the computation into non-linear, space-dependent transient states, the procedure remains logically sound and mathematically targets the correct fixed point.

## 5.7 Soundness and Completeness

To finalize the theoretical foundation of MT-PDCL, we must formalize the relationship between the operational semantics (the probability computed by the inference operator) and the declarative semantics (the exact measure of logically true possible worlds).

Definition 1 (Measure-Theoretic Soundness and Completeness). Let $\mathcal { K } ~ = ~ \langle \mathcal { M } _ { s } , \Phi \rangle$ be a knowledge base. Let $\begin{array} { r } { P _ { \mathcal { K } } ( H ) = \int _ { \Omega } \mathbb { I } ( \omega \mid = H ) d \mu _ { \Omega } ( \omega ) } \end{array}$ be the exact declarative probability of a ground fact H, as defined in Section 3.7. Let $P _ { o p } ( \bar { H } )$ be the probability derived by an operational inference procedure.

• Soundness: The inference procedure never assigns a probability higher than mandated by the declarative semantics: $P _ { o p } ( H ) \overset { \cdot } { \leq } P \overset { } { \kappa } ( \overset { \cdot } { H } )$ for all groundfacts H.

• Completeness: The inference procedure captures all probability mandated by the declarative semantics: $P \kappa ( \mathbf { \hat { \boldsymbol { H } } } ) \leq P _ { o p } ( \boldsymbol { H } )$ for all groundfacts H.

Consequently, a procedure is sound and complete ifand only if $P _ { o p } ( H ) = P \kappa ( H )$

Theorem 2 (Soundness and Completeness of Exact Inference). The exact continuous consequence operator $\mathbb { T } _ { e x a c t }$ is sound and complete with respect to the declarative semantics. For any groundfact H:

$$
l f p ( \mathbb { T } _ { e x a c t } ) ( H ) = P \kappa ( H )
$$

Proof. Let $\Omega _ { H } = \{ \omega \in \Omega \mid \omega \mid = H \}$ be the set of possible worlds where the ground fact H is logically entailed in the least model. By definition, the declarative probability is $P _ { \mathcal { K } } ( H ) = \mu ( \Omega _ { H } )$

Let $\Omega _ { H } ^ { k }$ be the set of possible worlds where H is successfully derived by the exact operator at or before iteration k. The operational probability at step k is $P _ { k } ( H ) = \mu ( \Omega _ { H } ^ { k } )$

To prove the equivalence, we first establish the exact set equality between the declarative worlds and the infinite union of operational worlds, by showing inclusion in both directions:

• Soundness (⊇): Since the program consists exclusively of definite clauses, any fact derived by iteration k possesses a valid logical proof. If $\omega \in \Omega _ { H } ^ { k }$ , it follows that $\omega \in \Omega _ { H }$ . This means $\Omega _ { H } ^ { k } \subseteq \Omega _ { H }$ for all k, which dictates that their infinite union is also a subset: $\cup _ { k = 1 } ^ { \infty } \Omega _ { H } ^ { k } \subseteq \Omega _ { H }$

• Completeness (⊆): By the foundational properties of definite logic programs, a fact H is true in the least model if and only if it has a finite derivation tree. Infinite derivations do not constitute logical entailment. Therefore, if $\omega \in \Omega _ { H }$ , there must exist some finite iteration depth K where the derivation completes. This means ω $\in \Omega _ { H } ^ { K }$ , and consequently $\omega \in \bigcup _ { k = 1 } ^ { \infty } \Omega _ { H } ^ { k }$ . This gives the reverse inclusion: $\textstyle \Omega _ { H } \subseteq \bigcup _ { k = 1 } ^ { \infty } \mathbf { \hat { \Omega } } \Omega _ { H } ^ { k }$

Because both subset inclusions hold, the sets are perfectly identical:

$$
\Omega _ { H } = \bigcup _ { k = 1 } ^ { \infty } \Omega _ { H } ^ { k }
$$

Finally, we map these sets to their probability measures. The sequence of operational derivations is strictly monotonic $( \Omega _ { H } ^ { 1 } \doteq \Omega _ { H } ^ { 2 } \subseteq \cdot \cdot \cdot )$ . According to the continuity of probability measures from below, the limit of the measure of an increasing sequence of sets is equal to the measure of their infinite union. Therefore:

$$
l f p ( \mathbb { T } _ { \mathrm { e x a c t } } ) ( H ) = \operatorname* { l i m } _ { k  \infty } \mu ( \Omega _ { H } ^ { k } ) = \mu ( \bigcup _ { k = 1 } ^ { \infty } \Omega _ { H } ^ { k } ) = \mu ( \Omega _ { H } ) = P _ { K } ( H )
$$

This establishes the strict equality, proving that the exact operator is both sound and complete

Because evaluating the exact operator is computationally intractable, we rely on the continuous Noisy-OR approximation. We formalize its completeness under the independence assumption below.

Theorem 3 (Completeness under Conditional Independence). If the failures of all overlapping derivations for a ground fact H are conditionally independent, the continuous Noisy-OR operator $\mathbb { T } _ { \mathcal { K } }$ is sound and complete with respect to the declarative semantics:

$$
l f p ( \mathbb { T } _ { \mathcal { K } } ) ( H ) = P _ { \mathcal { K } } ( H )
$$

Proof. Under the assumption of conditional independence, the exact measure of the union of independent derivation events is equal to the complement of the product of their failure probabilities. The continuous Noisy-OR formulation evaluates this specific product integral, making the approximated operator identical to the exact operator: $\mathbb { T } _ { \mathcal { K } } =$ $\mathbb { T } _ { \mathrm { e x a c t } }$ . Their least fixed points are structurally identical $\begin{array} { r } { \dot { ( l f p ( \mathbb { T } _ { K } ) } = \dot { l f p ( \mathbb { T } _ { \mathrm { e x a c t } } ) } \mathrm { ) } } \end{array}$ , and the result follows trivially from Theorem 1. □

## Remark: The Independence Assumption in Practice.

Unless conditional independence is physically guaranteed by the domain, the continuous Noisy-OR operator $\mathbb { T } _ { \mathcal { K } }$ acts as a computational approximation. When the ICI assumption does not perfectly hold, the operator trades exact measure-theoretic guarantees for tractability, generally losing both strict soundness and completeness. We dissect how different types of causal correlations lead to probability overestimation (loss of soundness) or underestimation (loss of completeness), and how to formally bound these approximation trade-offs, in the upcoming Section 5.9.

## 5.8 Operational Query Resolution from the Least Fixed Point

In classic Definite Clause Logic, the van Emden-Kowalski theorem establishes that the success set of a logic program is the least fixed point of its immediate consequence operator, $l f p ( T _ { P } )$ [vEK76]. Operationally, answering an existential non-ground query $\exists \mathbf { V } Q ( \mathbf { V } )$ reduces to searching the fixed point for any valid ground substitution v such that $Q ( \mathbf { v } ) \in$ $l f p ( \breve { T } _ { P } )$

In MT-PDCL, this principle must be generalized. Operating within a continuous domain with probabilistic outcomes shifts the objective away from searching for a single true instance. Instead, we must measure the accumulated probability that at least one ground instance bounded within the measurable answer space $\Omega _ { A }$ succeeds.

Therefore, answering a non-ground query corresponds to applying the continuous aggregation operator directly over a specific slice of the least fixed point.

Theorem 4 (Generalization of Query Entailment). Let Q be a query with free variables V, and let $\Omega _ { A }$ be its valid answer space defined by the query constraints. Let Q(v) denote the specific ground instance of the query at binding $\mathbf { v } \in \Omega _ { A }$

1. Exact Resolution: The exact declarative probability of the non-ground query is the measure of the union of the entailed worlds for all its ground instances:

$$
P _ { \mathcal { K } } ( Q , \Omega _ { A } ) = \mu _ { \Omega } \left( \bigcup _ { { \bf v } \in \Omega _ { A } } \Omega _ { Q ( { \bf v } ) } \right)
$$

where $\Omega _ { Q ( \mathbf { v } ) } = \{ \omega \in \Omega \mid \omega \mid = Q ( \mathbf { v } ) \}$

2. Operational Resolution (ICI): Under the ICI assumption, the operational probability of the non-ground query is derived by applying the continuous Noisy-OR integral over the marginal probabilities established in the operational leastfixed point $l f p ( \mathbb { T } \kappa )$

$$
P _ { o p } ( Q , \Omega _ { A } ) = 1 - \exp \left( \int _ { \Omega _ { A } } \ln \left( 1 - l f p ( \mathbb { T } _ { K } ) ( Q ( \mathbf { v } ) ) \right) d \mu _ { V } ( \mathbf { v } ) \right)
$$

Proof. (1) By Theorem 1, the exact least fixed point captures the precise set of possible worlds $\Omega _ { Q ( \mathbf { v } ) }$ where each ground fact $Q ( \mathbf { v } )$ is logically true. The non-ground query $\exists \mathbf { V } \in \Omega _ { A } Q ( \mathbf { V } )$ is logically equivalent to a continuous disjunction over all specific bindings $\mathbf { v } \in \Omega _ { A }$ . By the definition of the measure space, the exact probability of this continuous disjunction is the measure of the union of these sets.

(2) Operationally, evaluating the existential query requires computing the probability that at least one ground instance $Q ( \mathbf { v } ) \in \Omega _ { A }$ succeeds. Under the ICI assumption, these ground instances are conditionally independent. Therefore, the exact measure of their union resolves directly to the continuous Noisy-OR integral applied over the marginal probabilities stored in $l f p ( \mathbb { T } \kappa )$ □

Remark. This theorem bridges the gap between deduction and querying in continuous spaces. The consequence operator mathematically computes the probability of all facts (the fixed point), and the query engine simply integrates over bounded regions of that fixed point to answer existential questions.

## 5.9 Analytical Bounds and Approximation Trade-offs

When derivations are not independent (e.g., rules sharing underlying continuous variables or sub-goals), the ICI assumption is violated. In such cases, the operational probability $\dot { P _ { o p } } ( \bar { H } )$ computed by the Noisy-OR operator deviates from the exact declarative probability $P _ { \mathrm { e x a c t } } ( H )$

It is necessary to formalize the magnitude of this approximation error and how the operator behaves across continuous correlations. Let $\begin{array} { r } { \Delta ( H ) = \bigcup _ { R _ { j } \in \mathcal { R } _ { H } } \Omega _ { V _ { j } } } \end{array}$ be the continuous space of all variable bindings across the rules deriving a ground fact H, with measure $\mu _ { V }$

Let $P _ { \mathrm { e x a c t } } ( H )$ and $P _ { o p } ( H )$ represent the scalar probabilities assigned to H by the respective operators $( \mathbb { T } _ { \mathrm { e x a c t } }$ and $\mathbb { T } \kappa )$ at a given iteration. For any continuous binding $\mathbf { v } \in \Delta ( H )$ , let $p ( \mathbf { v } )$ be its isolated success probability at that iteration.

Theorem 5 (Absolute Bound on the Approximation Gap). Let L and U denote the exact lower and upper mathematical boundsfor $P _ { e x a c t } ( H )$ :

$$
\begin{array} { l } { L = \displaystyle \mathrm { e s s } \operatorname* { s u p } _ { \mathbf { v } \in \Delta ( H ) } p ( \mathbf { v } ) } \\ { U = \operatorname* { m a x } \left( L , \operatorname* { m i n } \left( 1 , \int _ { \Delta ( H ) } p ( \mathbf { v } ) d \mu _ { V } ( \mathbf { v } ) \right) \right) } \end{array}
$$

The absolute gap between the exact declarative probability and the operational approximation for any ground fact H is bounded $b y \colon$

$$
| P _ { e x a c t } ( H ) - P _ { o p } ( H ) | \leq \operatorname* { m a x } \big ( | P _ { o p } ( H ) - L | , | U - P _ { o p } ( H ) | \big )
$$

Proof. The exact declarative probability $P _ { \mathrm { e x a c t } } ( H )$ depends on the unknown joint distribution of the entailed possible worlds.

To bound the gap, we constrain $P _ { \mathrm { e x a c t } } ( H )$ dynamically based solely on the known marginals, utilizing continuous interval bounds:

1. Lower Bound (L): By the fundamental properties of measure theory, the measure of a continuous union can never be smaller than the measure of its largest single constituent event. Thus, $P _ { \mathrm { e x a c t } } ( H ) \geq L$

2. Upper Bound $( U ) { \mathrm { : } }$ By the upper bound of the Boole-Fréchet inequalities [Fré51], the measure of a union is bounded by the integral of its marginals. However, in continuous spatial logic, if the domain measure is fractional $( \mu _ { V } < 1 )$ , this integral may mathematically fall below the monotonicity bound L. Therefore, the strict upper bound is the maximum of the integral and the lower bound, capped at 1. Thus, $P _ { \mathrm { e x a c t } } ( H ) \leq U$

The operational scalar $P _ { o p } ( H )$ is computed using the Noisy-OR assumption. Because the ICI assumption is violated, $P _ { o p } ( \bar { H } )$ is merely an approximation and may fall anywhere inside or outside the valid mathematical interval $[ L , U ]$

However, by the geometric properties of intervals, the maximum possible absolute distance between a known computed value $( P _ { o p } ( { \cal H } ) )$ and an unknown exact value bounded within an interval $( P _ { \mathrm { e x a c t } } ( H ) \in [ L , U ] )$ is the maximum distance from the known value to the interval’s endpoints. Therefore, the absolute error gap is bounded by max $\cdot ( | P _ { o p } ( H ) - L | , | U - P _ { o p } ( H ) | )$ . □

## Remark: Practical Implications.

Calculating the exact declarative probability $P _ { \mathrm { e x a c t } } ( H )$ is generally intractable because it requires evaluating the hidden joint distribution of uncountably infinite derivation trees. However, the theoretical results in this section equip an operational MT-PDCL system with a computable dynamic error guarantee.

Every term in the interval bounds (L and U) depends solely on the known marginal probabilities $p ( \mathbf { v } )$ and the spatial measure $\mu _ { V }$ , allowing the system to dynamically compute the absolute worst-case error margin of its Noisy-OR approximation at runtime. If this evaluated gap is narrow, it mathematically guarantees that the computationally cheap Noisy-OR operator is highly accurate for that specific query. If the gap is large, it acts as a precise diagnostic flag, alerting the system that the underlying continuous variables overlap too heavily to trust the independent causal influence assumption.

## 5.10 Differentiability and Neural-Symbolic Integration

While the continuous Noisy-OR operator acts as an approximation when the ICI assumption is violated, it possesses a structural property necessary for neural-symbolic integration: it is everywhere differentiable. This enables frameworks such as Deep Measure-Theoretic Probabilistic Definite Clause Logic (Deep-MT-PDCL). Inspired by existing probabilistic neuro-symbolic systems like DeepProbLog $[ \mathrm { M D K } ^ { + } 2 1 ] .$ , these frameworks allow the underlying probabilities to be the outputs of parameterized neural networks optimized via gradient descent.

To correctly model this in a continuous space, a neural network does not predict a single scalar for an entire rule $R _ { j }$ Instead, it predicts the innate causal probability $p _ { j } ( \mathbf { v } )$ parameterized by the specific continuous binding $\mathbf { v } \in \Omega _ { V _ { i } }$ which consists only of the continuous variables that remain free in the rule body after the head unifies with H. The total success probability of that instance is $p _ { j } ( \mathbf { v } ) \cdot P ( B _ { j } \mid \mathbf { v } )$

Proposition 4 (Differentiability and Gradient Routing). For any groundfact H, let the rule probabilities $p _ { j } ( \mathbf { v } ; \theta )$ be parameterized by a continuous, differentiable function (such as a neural network) with weights θ. The continuous consequence operator $\mathbb { T } _ { \mathcal { K } }$ is differentiable with respect to θ. Furthermore, the operator acts as a monotonically non-decreasing aggregation layer: an increase in any local rule probability $p _ { j } ( \mathbf { v } ; \theta )$ mathematically guarantees a non-negative contribution to the aggregated probability of H, ensuring stable gradient routing for backpropagation.

Proof. Let E represent the total joint failure probability for a ground fact H derived by a set of rules $\mathcal { R } _ { H } \mathbf { : }$

$$
E = \exp \left( \sum _ { R _ { j } \in \mathcal { R } _ { H } } \int _ { \Omega _ { V _ { j } } } \ln \left( 1 - p _ { j } ( \mathbf { v } ; \theta ) \cdot P ( B _ { j } \mid \mathbf { v } ) \right) d \mu _ { V } ( \mathbf { v } ) \right)
$$

The operational probability is $P _ { o p } ( H ) = 1 - E$ . By the chain rule, the gradient of the operational probability with respect to the network weights θ requires the partial derivative of the operator’s output with respect to the network’s continuous predictions. Applying the Leibniz integral rule, we obtain:

$$
\nabla _ { \theta } P _ { o p } ( H ) = E \cdot \sum _ { R _ { j } \in \mathcal { R } _ { H } } \int _ { \Omega _ { V _ { j } } } \frac { P ( B _ { j } \mid \mathbf { v } ) \cdot \nabla _ { \theta } p _ { j } ( \mathbf { v } ; \theta ) } { 1 - p _ { j } ( \mathbf { v } ; \theta ) \cdot P ( B _ { j } \mid \mathbf { v } ) } d \mu _ { V } ( \mathbf { v } )
$$

We isolate the scalar multiplier applied to the local parameter gradient $\nabla _ { \theta } p _ { j } ( \mathbf { v } ; \theta )$ for any specific binding v:

$$
w ( \mathbf { v } ) = E \cdot { \frac { P ( B _ { j } \mid \mathbf { v } ) } { 1 - p _ { j } ( \mathbf { v } ; { \boldsymbol { \theta } } ) \cdot P ( B _ { j } \mid \mathbf { v } ) } }
$$

We analyze the bounds of this scalar weight. The exponential function ensures $E > 0$ . Inside the fraction, the probability of the logical body $P ( B _ { j } \mid \mathbf { v } ) \in [ 0 , 1 ]$ ], and the continuous network prediction $p _ { j } ( \mathbf { v } ; \theta ) \in [ 0 , 1 )$ . Therefore, for every binding in the measurable domain, the weight is strictly non-negative: $w ( \mathbf { v } ) \geq 0 .$

This mathematical property guarantees that the continuous consequence operator acts as a structurally transparent routing layer. It scales the magnitude of the local gradients based on the logical validity of their bindings, but it never inverts them. Because $\begin{array} { r } { \frac { \partial P _ { o p } } { \partial p _ { j } ( \mathbf { v } ) } \geq 0 } \end{array}$ , the optimization loss directly backpropagates to the network weights without suffering from adversarial sign-flipping, allowing the neural network to be trained end-to-end via standard gradient descent. □

Remark. In practical neural-symbolic implementations, probabilities are clamped to the open interval (0, 1) or evaluated using log-space stabilization to prevent numerical division-by-zero when evaluating deterministic derivations where $p _ { j } ( \mathbf { \bar { v } } ; \theta ) \mathbf { \bar { \theta } } \cdot \bar { P } ( B _ { j } \mid \mathbf { v } ) = 1$

## 6 Related Work and Comparative Analysis

By formalizing probabilities natively through measure theory rather than discrete propositional grounding, MT-PDCL shifts how logic programs are evaluated. It is necessary to contextualize this approach alongside existing probabilistic and neural-symbolic frameworks.

Table 1 provides a feature comparison of prominent probabilistic logic frameworks against MT-PDCL.
<table><tr><td rowspan=1 colspan=1>Framework</td><td rowspan=1 colspan=1>Entailment Semantics</td><td rowspan=1 colspan=1>Logical VariableDomain</td><td rowspan=1 colspan=1>PropositionalGrounding</td><td rowspan=1 colspan=1>Differen-tiable</td></tr><tr><td rowspan=1 colspan=1>PRISM / ProbLog /CP-logic [SK97,DRKT07, VDB09]</td><td rowspan=1 colspan=1>Distribution / CausalSemantics</td><td rowspan=1 colspan=1>Discrete</td><td rowspan=1 colspan=1>Required (LogicPropositions)</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>DeepProbLog[MDK+21]</td><td rowspan=1 colspan=1>Distribution Semantics</td><td rowspan=1 colspan=1>Discrete (Neural facts)</td><td rowspan=1 colspan=1>Required (SDDs)</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>DeepSeaProbLog /DeepGraphLog[DSZDMM+23,KMG+25]</td><td rowspan=1 colspan=1>Distribution Semantics</td><td rowspan=1 colspan=1>Continuous (Hybrid)</td><td rowspan=1 colspan=1>Required (WMICompilation)</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>TensorLog [CYM20]</td><td rowspan=1 colspan=1>Tensor Compilation</td><td rowspan=1 colspan=1>Discrete (Fixed entities)</td><td rowspan=1 colspan=1>Replaced byMatrix Ops</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>BLOG [MMR+05]</td><td rowspan=1 colspan=1>Generative /Distribution</td><td rowspan=1 colspan=1>Continuous</td><td rowspan=1 colspan=1>Unbounded(Sampling)</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>DC-ProbLog / HybridProbLog[GJDR11, MHLV15]</td><td rowspan=1 colspan=1>Distribution / SMT</td><td rowspan=1 colspan=1>Continuous</td><td rowspan=1 colspan=1>Deferred to SMT</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>Weighted ModelIntegration (WMI)[BPVdB15, MPVdB21,SAMP24, ACF+25]</td><td rowspan=1 colspan=1>Model Integration</td><td rowspan=1 colspan=1>Continuous</td><td rowspan=1 colspan=1>Required (StaticFormulas)</td><td rowspan=1 colspan=1>No</td></tr><tr><td rowspan=1 colspan=1>MT-PDCL (Ours)</td><td rowspan=1 colspan=1>ContinuousDistribution</td><td rowspan=1 colspan=1>Continuous (BorelSpaces)</td><td rowspan=1 colspan=1>Not Required(Integration)</td><td rowspan=1 colspan=1>Yes</td></tr></table>

Table 1: Qualitative comparison of probabilistic logic frameworks.

Systems relying on standard distribution semantics (ProbLog, PRISM) define a unique probability distribution over possible worlds by making strong independence assumptions among base probabilistic facts. To evaluate queries, they rely heavily on grounding logic programs into discrete propositional representations (such as Sentential Decision Diagrams). While highly effective for discrete tasks, this operational requirement restricts exact inference to finite domains.

DeepProbLog and TensorLog successfully bridge probabilistic logic programming with deep learning, handling continuous neural network parameters. However, their underlying logical semantics remain tied to finite domains. While recent extensions like DeepSeaProbLog [DSZDMM<sup>+</sup>23] and DeepGraphLog [KMG<sup>+</sup>25] attempt to generalize these frameworks to hybrid continuous domains, they still inherently rely on compiling the logic into Weighted Model Integration (WMI) formulas or restricted graph structures. DeepProbLog evaluates neural outputs as probabilistic facts, but its logical inference engine still compiles to discrete SDDs. TensorLog bypasses traditional grounding by compiling logical reasoning into differentiable tensor operations, but it does so over a fixed, finite set of predefined entities.

BLOG (Bayesian Logic) [MMR<sup>+</sup>05] handles continuous domains and unknown numbers of objects, but relies on generative models and approximate sampling methods (like MCMC) rather than declarative, measure-theoretic fixedpoint semantics.

While BLOG utilizes generative sampling, other frameworks approach continuous domains through Satisfiability Modulo Theories (SMT). Early frameworks such as Distributional Clauses (DC-ProbLog) [GJDR11] and Hybrid ProbLog [MHLV15] allow continuous random variables by deferring logical evaluation to SMT solvers, which identify valid geometric boundaries for numerical integration. This approach evolved into WMI, which computes continuous probability densities over SMT formulas [BPVdB15]. Recent advances in WMI focus on algorithmic scaling through algebraic circuit compilation [KMR20, DKM<sup>+</sup>20], extending exact integration to non-linear real arithmetic [SAMP24], and approximating complex semi-algebraic weight functions via linear programming relaxations [ACF<sup>+</sup>25].

However, these systems focus primarily on operational execution. WMI operates on static formulas or factor graphs and does not support relational recursion natively, while DC-ProbLog relies on algorithmic sampling without formal proof of continuous fixed-point convergence. Furthermore, recent 2025 and 2026 surveys on neuro-symbolic agent architectures highlight a critical shift toward tightly coupled, differentiable symbolic layers [BJCD25]. MT-PDCL addresses these theoretical limitations directly. It provides the exact denotational semantics—via measure theory and the Knaster-Tarski fixed-point theorem—that formalizes how recursive continuous subsets converge. Additionally, MT-PDCL natively preserves the structural differentiability required for these modern deep learning integrations, which standard SMT-based circuit solvers do not support without heavy approximation.

Furthermore, the operational reliance on discrete propositionalization extends well beyond standard definite clause frameworks. Expressive logical extensions, such as Probabilistic Answer Set Programming (PASP) [BGR09] and defeasible logic programs [GS04], similarly rely on exhaustive grounding to evaluate probabilities over complex rule interactions. Because these advanced solvers also compile down to finite boolean spaces, they inherit the exact same inability to natively represent continuous probability densities. By resolving the grounding bottleneck at the foundational level of definite clauses, MT-PDCL provides the mathematical foundation necessary to eventually lift these non-monotonic logical extensions into continuous spaces.

MT-PDCL operates under Continuous Distribution Semantics. Rather than deferring to approximate sampling or static SMT solvers, MT-PDCL defines a unique, exact probability measure over continuous possible worlds via Lebesgue integration. The primary novelty lies in extending Sato’s generative semantics, historically restricted to discrete causal events and finite propositional grounding, into unified measurable spaces. By explicitly defining independent continuous variables over bounded index domains equipped with probability measures, MT-PDCL replaces the combinatorial bottleneck of discrete grounding with formal mathematical integration. This preserves the declarative exactness of logic programming while seamlessly supporting the differentiable, algebraic approximations (Noisy-OR) required for neural-symbolic optimization.

Operationally, this shifts the evaluation of logic programs from discrete proof-tree search (e.g., standard SLD resolution) to multivariate integration. In MT-PDCL, the core mechanics of logic programming map directly to specific continuous operations:

• Stochastic Predicates as Measurable Dimensions: The explicit domains (Ω ) and continuous outputs of stochastic predicates define the independent axes of the continuous evaluation space. Logical variables act as deterministic placeholders that bind to these dimensions.

• Conditions as Integration Boundaries: Built-in base predicates (such as mathematical inequalities) define the strict geometric bounds over which the probability densities are integrated.

• Inference as Integration: The consequence operator computes the exact probability mass of these bounded regions by integrating over the relevant density functions, replacing the boolean summation of discrete derivations.

• Scoping as Marginalization: Standard logic programs treat variables present only in the rule body as existentially quantified. MT-PDCL maps this directly to internal marginalization. The engine integrates out hidden continuous variables across their bounded domains, collapsing multi-dimensional spaces into a scalar probability.

## 7 Execution Traces and Comparative Evaluation

This section evaluates MT-PDCL against three continuous relational problems. We provide the abstract mathematical formulations, the exact numeric execution traces under the MT-PDCL continuous consequence operator $\left( \mathbb { T } _ { \mathrm { e x a c t } } \right)$ , and a direct mechanical analysis of why existing frameworks cannot compute these exact analytical probabilities.

## 7.1 Problem 1: Infinite-Horizon Probability of Ruin (Supply Chain)

Practical Context: Consider a warehouse managing a critical inventory. The warehouse starts with a fixed stock and receives a constant daily replenishment. However, the daily customer demand fluctuates randomly. The business objective is to calculate the exact probability that the warehouse will never run out of stock over an infinite time horizon. In risk management, this is known as the probability of ruin.

Abstract Formulation: To formalize this, let the universal measurable domain of discourse D include the continuous space R, equipped with the standard Borel σ-algebra $\mathcal { F } _ { D }$ . Let $S , S _ { p r e v } \in \mathcal { V }$ be logical variables mapping to this continuous domain to represent daily inventory levels, and let $T \in \mathcal { V }$ be a logical variable representing discrete time steps (days). We define an initial ground fact safe $( 0 , s _ { 0 } )$ establishing the starting inventory $S = s _ { 0 } \ge 0$ at day $T = 0 ,$

The daily state transitions are governed by the built-in arithmetic base predicate $S = S _ { p r e v } + R - U$ , where $R \in \mathbb R$ is a numeric constant representing the daily replenishment rate. The customer usage $U ^ { \bar { } } \in \mathcal { V }$ is a continuous variable generated by the stochastic mapping $\mathcal { M } _ { s }$ for each time step.

The objective is to compute the declarative entailment of the relation $\forall T , \mathrm { s a f e } ( T , S )$ . Under Continuous Distribution Semantics, this requires calculating the exact Lebesgue measure for the subset of continuous states that satisfy the recursive logical derivation, ensuring the condition $S \geq 0$ holds across all discrete steps T.

## The Knowledge Base $\kappa \colon$

To model the continuous inventory flow, K defines the stochastic measure mapping and the recursive state transitions:

1. usage $( T ; U ) \sim \mathrm { U n i f o r m } ( 0 , 2 )$

(Stochastic measure declaration: generates an independent continuous usage value for each discrete time index $T . )$

2. $1 . 0 : : \mathrm { s a f e } ( 0 , 1 . 0 )$

(The initial deterministic ground fact.)

3. $1 . 0 : : \mathrm { s a f e } ( T , S ) \gets \mathrm { s a f e } ( T - 1 , S _ { p r e v } ) \land$ usage(T; U) ∧ S = S<sub>prev</sub> + 1.0 − U ∧ S ≥ 0.

(The recursive clause. The deterministic time index T acts as the lookup key to request a fresh, independent continuous samplefrom the environment.)

Numeric Execution Trace $( \mathbb { T } _ { \mathbf { e x a c t } } ) { : }$ Let us suppose that $S _ { 0 } = 1 . 0 , R = 1 . 0$ , and $U _ { t } \sim \mathrm { U n i f o r m } ( 0 , 2 )$ for each discrete time step $t \geq 0$ . We $\mathrm { a p p l y ~ \mathbb { T } _ { e x a c t } }$ iteratively for $t = 0 , 1 , 2 , 3$ , starting from the initial empty measure $P _ { 0 }$

• Step 0: Initial Empty Measure $( P _ { 0 } )$

The probability measure for all derived facts is initially zero.

• Step 1: Initialization $( T = 0 , P _ { 1 } )$

The unconditional base fact successfully generates the initial state.

$$
P _ { 1 } ( \mathrm { s a f e } ( 0 , 1 . 0 ) ) = 1 . 0
$$

• Step 2: Iteration $\mathbf { 1 } \left( T = 1 , P _ { 2 } \right)$

The state constraint $S _ { 1 } \geq 0$ is validated across the domain.

$$
P _ { 2 } ( \mathrm { s a f e } ( 1 , S _ { 1 } ) ) = 1 . 0
$$

• Step 3: Iteration $\mathbf { 2 } \left( T = 2 , P _ { 3 } \right)$ The state equation resolves to $S _ { 2 } = 3 . 0 - U _ { 1 } - U _ { 2 }$ . The constraint requires $S _ { 2 } \geq 0 \implies U _ { 1 } + U _ { 2 } \leq 3 . 0 .$ $\mathbb { T } _ { \mathrm { e x a c t } }$ integrates over the joint continuous state space of $U _ { 1 } , U _ { 2 } \in \lbrack 0 , 2 \rbrack \times \lbrack 0 , 2 \rbrack$ . The valid geometric region is the total domain $( \mathrm { a r e a } = 4 )$ minus the triangle where $U _ { 1 } + U _ { 2 } > \dot { 3 } . 0 \stackrel { . } { ( \mathrm { a r e a } } = \bar { 0 } . 5 )$ .

$$
P _ { 3 } ( \mathrm { s a f e } ( 2 , S _ { 2 } ) ) = { \frac { 4 - 0 . 5 } { 4 } } = 0 . 8 7 5
$$

• Step 4: Iteration $3 ( T = 3 , P _ { 4 } )$ The state equation resolves to $S _ { 3 } = 4 . 0 - U _ { 1 } - U _ { 2 } - U _ { 3 }$ . The safety constraint requires $S _ { 3 } \geq 0 \implies$

$U _ { 1 } + U _ { 2 } + U _ { 3 } \leq 4 . 0$ , and the prior condition $U _ { 1 } + U _ { 2 } \le 3 . 0$ must still hold. $\mathbb { T } _ { \mathrm { e x a c t } }$ integrates over the 3-dimensional joint state space $( \dot { U } _ { 1 } , U _ { 2 } , U _ { 3 } ) \in [ 0 , 2 ] ^ { 3 }$

The total volume of this cubic domain is $2 ^ { 3 } = 8$ . To find the valid integration volume, the operator subtracts the regions where the state falls below zero. The exact geometry of this bounded valid region has a volume of $\frac { 1 9 } { 3 }$

$$
P _ { 4 } ( \mathrm { s a f e } ( 3 , S _ { 3 } ) ) = \frac { 1 9 / 3 } { 8 } = \frac { 1 9 } { 2 4 } \approx 0 . 7 9 2
$$

## Recursive Density and Fixed Point Convergence:

The continuous consequence operator $\mathbb { T } _ { \mathrm { e x a c t } }$ generates a monotonically increasing sequence of valid integration regions. Because these measures form a complete lattice, the least fixed point is mathematically defined as their exact limit:

$$
\operatorname* { l f p } ( \mathbb { T } _ { \mathrm { e x a c t } } ) = \operatorname* { l i m } _ { k \to \infty } P _ { k }
$$

The progression of the operator is defined via a direct recurrence relation on the sub-probability density function of the inventory state. Let $f _ { k } ( s )$ represent the probability density of $S _ { k } = s$ while satisfying the guard $S _ { i } \geq 0$ for all previous steps $i \leq k$ . Starting from $\bar { f _ { 0 } } ( s ) = \delta ( s ^ { \prime } - 1 . 0 )$ and a usage density of 0.5, we obtain the continuous recurrence:

$$
f _ { k } ( s ) = \int _ { \mathrm { m a x } ( 0 , s - 1 ) } ^ { s + 1 } f _ { k - 1 } ( x ) \cdot 0 . 5 d x \quad \mathrm { f o r } s \geq 0
$$

The measure assigned to the safety fact at step k is the total probability mass of this density:

$$
P _ { k } = \int _ { 0 } ^ { \infty } f _ { k } ( s ) d s
$$

To evaluate the infinite-horizon property, we must prove $P _ { k }$ converges to zero as $k  \infty$ . We analyze the unconstrained random walk $W _ { n } ,$ which removes the $s \geq 0$ boundary condition. Let $\Delta _ { i } = 1 . 0 - U _ { i }$ denote the net change in inventory at step i. The unconstrained inventory path is:

$$
W _ { n } = 1 . 0 + \sum _ { i = 1 } ^ { n } \Delta _ { i }
$$

The exact value $P _ { k }$ is the probability that this unconstrained path never drops below zero in its first k steps:

$$
P _ { k } = P \left( \underset { 1 \leq n \leq k } { \operatorname* { m i n } } W _ { n } \geq 0 \right)
$$

Because $U _ { i } \sim \mathrm { U n i f o r m } ( 0 , 2 )$ , the increments $\Delta _ { i }$ are independent, continuous, and symmetric on $[ - 1 , 1 ]$ . By the Sparre Andersen theorem, for any continuous and symmetric jump distribution, the probability that the random walk remains non-negative for k steps is identical to the simple random walk probability: $\binom { 2 k } { k } 2 ^ { - 2 \bar { k } }$ . By Stirling’s approximation, this probability decays asymptotically as $1 / \sqrt { \pi k }$ . Therefore, there exists a constant $C > 0$ such that for sufficiently large k:

$$
P _ { k } \leq { \frac { C } { \sqrt { k } } }
$$

By the squeeze theorem, since $0 \leq P _ { k } \leq C / \sqrt { k } .$ , it follows that:

$$
\operatorname* { l i m } _ { k \to \infty } { P _ { k } = 0 }
$$

Comparative Limitations: Standard discrete frameworks such as ProbLog and DeepProbLog must discretize the continuous uniform usage distribution into finite bins. Slicing [0, 2] into discrete intervals creates exponential combinatorial growth in their underlying boolean circuits (SDDs). Furthermore, while modern continuous extensions like DeepSeaProbLog [DSZDMM<sup>+</sup>23] and DeepGraphLog [KMG<sup>+</sup>25] support continuous parameters, they operationally rely on compiling rule instances into static WMI representations. For infinite-horizon recursive derivations, this compilation loop triggers an infinite formula expansion, preventing these systems from computing the exact Lebesgue measure over recursive states.

## 7.2 Problem 2: Probability of Safe Arrival (Kinematic Agent)

Practical Context: An automated guided vehicle (AGV) is moving along a 10-meter track. It needs to calculate a backward reachable safe set: the exact physical regions from which it can successfully reach a designated target zone without crashing into a known boundary. Because the vehicle’s braking and acceleration hardware introduce continuous mechanical variance, its control steps are not perfectly precise.

Abstract Formulation: Let the universal domain D include the continuous spatial space R. Let $X , Y \in \mathcal { V }$ map to this domain to represent the AGV’s spatial positions on the track, and let ${ \mathcal { T } } = [ 8 $ , 10] define the measurable target zone in meters.

The physical transitions are governed by the built-in arithmetic base predicate $X = Y - N .$ , representing a backward kinematic step. The actual control movement $N \in \nu$ is an independent continuous stochastic variable. The spatial domain over which the agent operates is bounded to the interval [0, 10], equipped with a uniform base measure $d \mu _ { X } ( x ) = 0 . 1 d x$

The objective is to compute the exact declarative probability of the subset of valid starting states from which a finite sequence of mechanical derivations reaches the target set T, while satisfying the safety guard condition $X \geq 0$ at every intermediate step. Unlike the infinite-horizon survival in Problem 1 which evaluates a shrinking sub-density, this reachability problem calculates the limit of an expanding geometric region.

The knowledge base K models the backward reachable set:

1. contro $\left( X ; N \right) \sim \operatorname { U n i f o r m } ( 0 , 2 )$ over $X \in [ 0 , 1 0 ] .$

(Stochastic measure declaration mapping the movement step to the bounded spatial domain.)

2. 1 $. . 0 : \operatorname { s a f e } ( X )  X \geq 8 \land X \leq 1 0 .$

(The deterministic base fact establishing the target set.)

3. 1.0 :: safe(X) ← safe(Y ) ∧ control(X; N) ∧ X = Y − N ∧ X ≥ 0.

(The recursive clause computing the valid pre-images, bounded by the obstacle guard $X \geq 0 . )$

Numeric Execution Trace $( \mathbb { T } _ { \mathbf { e x a c t } } ) { : }$ The continuous consequence operator computes the exact measure of the geometric union of successful states at each step, integrating over the uniform base measure 0.1 dx.

• Step ${ \bf 1 } \left( P _ { 1 } \right)$ : The base fact triggers for $x \in [ 8 , 1 0 ]$

$$
P _ { 1 } ( { \mathrm { s a f e } } ( X ) ) = \int _ { 8 } ^ { 1 0 } 0 . 1 d x = 0 . 2
$$

• Step $\textbf { 2 } ( P _ { 2 } ) \colon$ The recursive rule expands the safe region. The valid continuous union requires finding any $y \in [ 8 , 1 0 ]$ and $n \in [ 0 , 2 ]$ such that $x = y - n$ . The minimum possible value is $x = 8 - 2 = 6$ . The rule validates for all $x \in [ 6 , 1 0 ]$

$$
P _ { 2 } ( { \mathrm { s a f e } } ( X ) ) = \int _ { 6 } ^ { 1 0 } 0 . 1 d x = 0 . 4
$$

• Steps $3 – 5 ( P _ { 3 } , P _ { 4 } , P _ { 5 } )$ : Iterative algebraic projection expands the lower bound by 2 at each step:

$P _ { 3 }$ evaluates $x \in [ 4 , 1 0 ] \implies 0 . 6$

$P _ { 4 }$ evaluates $x \in [ 2 , 1 0 ] \implies 0 . 8$

$P _ { 5 }$ evaluates $x \in [ 0 , 1 0 ] \implies 1 . 0$

• Step 6 (Least Fixed Point): The progression attempts to expand the boundary to $x = - 2$ , but the obstacle guard $X \geq 0$ truncates the domain. No new valid states are generated: $\mathbb { T } _ { \mathrm { e x a c t } } ( P _ { 5 } ) = P _ { 5 }$

Comparative Limitations: WMI frameworks can compute the geometric volume of bounded, acyclic steps algebraically. However, even state-of-the-art WMI solvers utilizing linear programming relaxations for complex arithmetic $[ \mathrm { S A M P } \dot { 2 } 4 , \mathrm { A C F } ^ { + } 2 5 ]$ cannot evaluate an open recursive rule dynamically. WMI natively requires a user to manually unroll the relational recursion to a predetermined static depth k and hardcode the resulting formula prior to execution. It inherently lacks the generalized denotational semantics and fixed-point operators required to evaluate dynamic, expanding reachability boundaries.

## 7.3 Problem 3: Continuous Signal Propagation (Hierarchical Sensor Network)

Practical Context: Consider a hierarchical sensor network where a central node transmits a baseline signal through a series of routing relays. As the signal passes through the physical environment, it accumulates continuous noise. Furthermore, the network topology is unreliable; routing connections may randomly drop. The goal is to compute the exact probability that a signal successfully reaches the leaf nodes while staying below a critical noise threshold.

Abstract Formulation: Let the network be modeled as a rooted tree $T = ( V , E )$ with the central node as the root $v _ { 0 }$ The universal domain D includes the continuous space R.

Let $X , Y \in \mathcal { V }$ map to the network vertices in V. Let $V _ { c u r r e n t } , V _ { p r e v } , N \in \mathcal { V }$ map to $\mathbb { R }$ to represent the accumulated continuous signal noise. The problem incorporates both continuous and discrete stochasticity: the base noise state at the root $v _ { 0 }$ is drawn from a continuous prior measure, physical transitions along routing edges add an independent continuous shift N, and the topological edges in E are modeled as independent causal events with discrete probabilities representing structural connection failures.

The transmission along a directed edge is governed by the continuous arithmetic constraint $V _ { c u r r e n t } = V _ { p r e v } + N$ Because $T$ is a rooted tree, the path from the root to any node is unique. We evaluate the exact Lebesgue measure of the state space where the accumulated continuous noise satisfies $V _ { c u r r e n t } \le \tau$ , weighted by the probabilities of the generative discrete edges.

Specific Example and Knowledge Base K: We evaluate a tree where V = {node1, node2, node3, node4} and $E =$ {(node1, node2), (node2, node3), (node1, node4)}.

The initial state at the root is a continuous variable $V _ { r o o t } \sim \mathrm { U n i f o r m } ( 0 , 1 )$ . The causal event for the edge to node 4 has an independent probability of 0.8, while other edges are deterministic (1.0). The continuous transition shift is N ∼ Uniform(0, 2), and the objective threshold is $\tau = 1 . 0$

1. initial\_ $\mathrm { s t a t e } ( ; V _ { r o o t } ) \sim \mathrm { U n i f o r m } ( 0 , 1 )$

2. sh $\operatorname { i i f t } ( X , Y ; N ) \sim \operatorname { U n i f o r m } ( 0 , 2 ) .$

(Stochastic measure declarationsfor the root value and edge transitions.)

3. $1 . 0 : \mathrm { { e d g e } } ( \mathrm { n o d e l } , \mathrm { n o d e } 2 ) .$

4. $1 . 0 : \mathrm { e d g e } ( \mathrm { n o d e } 2 , \mathrm { n o d e } 3 ) .$

5. 0.8 :: edge(node1, node4).

6. $\begin{array} { r } { 1 . 0 : : \mathrm { { v a l } } ( { \mathrm { n o d e } } 1 , V _ { r o o t } ) \gets \mathrm { { i n i t i a l \_ s t a t e } } ( ; V _ { r o o t } ) . } \end{array}$

$$
\begin{array} { r } { 7 . \begin{array} { l } { . \mathrm { 1 . 0 : 5 u l } ( Y , V _ { c u r r e n t } ) \gets \mathrm { e d g e } ( X , Y ) \wedge \mathrm { v a l } ( X , V _ { p r e v } ) \wedge \mathrm { s h i f t } ( X , Y ; N ) \wedge V _ { c u r r e n t } = V _ { p r e v } + N . } \end{array} } \end{array}
$$

Numeric Execution Trace $( \mathbb { T } _ { \mathbf { e x a c t } } ) { : }$ The continuous consequence operator $\mathbb { T } _ { \mathrm { e x a c t } }$ constructs the measurable state space bottom-up. It calculates probabilities directly via integration of the continuous densities multiplied by the independent causal event probabilities.

• Step $\mid ( P _ { 0 } ) \colon$ The base fact initializes the state at the root node. The density function is $f _ { V _ { 1 } } ( v _ { 1 } ) = 1$ for $v _ { 1 } \in [ 0 , 1 ]$

• Step 1 $( P _ { 1 } ) \colon$ The operator applies the recursive rule over edge(node1, node2) and edge(node1, node4). For node4, the arithmetic constraint is $V _ { 4 } = V _ { 1 } + N _ { 4 }$ . The variables are independent with a joint density $f ( v _ { 1 } , n _ { 4 } ) = 1 \times 0 . 5 = 0 . 5$ . Applying the continuous constraint $v _ { 1 } + n _ { 4 } \leq 1 . 0$ isolates a 2D geometric triangle with an area of 0.5. Because the topological edge generates with an independent probability of 0.8, the rule dictates:

$$
P ( { \mathrm { v a l } } ( { \mathrm { n o d e 4 } } , V _ { 4 } ) \wedge V _ { 4 } \leq 1 . 0 ) = 0 . 8 \times \left( \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 - v _ { 1 } } 0 . 5 d n _ { 4 } d v _ { 1 } \right) = 0 . 8 \times 0 . 2 5 = 0 . 2 5
$$

Simultaneously, the operator computes the state for node2. The discrete edge weight is 1.0, governed by the joint density $f ( v _ { 1 } , n _ { 2 } ) = 0 . 5$

• Step $\mathbf { 2 } \left( P _ { 2 } \right) :$ The operator applies the recursive rule over edge(node2, node3).

The state at node3 accumulates a third variable: $V _ { 3 } = V _ { 2 } + N _ { 3 } = V _ { 1 } + N _ { 2 } + N _ { 3 }$ . The independent continuous variables form a 3-dimensional state space with a joint density $f ( v _ { 1 } , n _ { 2 } , n _ { 3 } ) = 1 \times 0 . 5 \times 0 . 5 = 0 . 2 5$

Applying the query constraint $V _ { 3 } \leq 1 . 0$ requires integrating this joint density over the 3D geometric subspace bounded by $v _ { 1 } + n _ { 2 } + n _ { 3 } \leq 1 . 0$ . This region forms a standard 3-simplex (a tetrahedron) in the positive octant

with a geometric volume of $\frac { 1 } { 6 }$ . The exact declarative probability is:

$$
P ( { \mathrm { v a l } } ( { \mathrm { n o d e } } 3 , V _ { 3 } ) \wedge V _ { 3 } \leq 1 . 0 ) = 1 . 0 \times \prod _ { v _ { 1 } + n _ { 2 } + n _ { 3 } \leq 1 } 0 . 2 5 d n _ { 3 } d n _ { 2 } d v _ { 1 } = 0 . 2 5 \times { \frac { 1 } { 6 } } \approx 0 . 0 4 1 6
$$

• Step 3 (Fixed Point): No further relational edges exist to trigger the recursive rule. The operator yields no new measurable states, reaching the exact least fixed point.

Comparative Limitations: Frameworks such as BLOG and Distributional Clauses (DC-ProbLog) defer continuous path accumulation to Monte Carlo or particle sampling. They generate numerical execution traces through the network and compute a statistical average. Because they do not calculate the exact geometric volumes mathematically, they cannot provide a formally exact declarative entailment. Furthermore, relying on stochastic sampling creates nondifferentiable, high-variance scalar estimates, preventing the extraction of stable, everywhere-differentiable gradients necessary for end-to-end neural-symbolic optimization [BJCD25].

## 8 Computability, Tractability, and Theoretical Limitations

MT-PDCL provides a formally exact declarative semantics for continuous probabilistic logic via Lebesgue integration. However, defining an integral mathematically is fundamentally different from computing it analytically. It is necessary to establish the theoretical boundaries of computability and tractability for the continuous consequence operator.

## 8.1 Analytical Computability

The exact evaluation of the consequence operator $\mathbb { T } _ { \mathcal { K } }$ requires computing the integral of continuous probability densities over boundaries defined by logical inequalities.

An exact, closed-form analytical solution to these integrals is theoretically guaranteed only under highly restrictive conditions. Specifically, the prior measures defined by the mapping $\mathcal { M } _ { s }$ must be standard integrable functions (e.g., Uniform, Gaussian, Exponential), and the logical constraints must form geometrically simple boundaries (typically linear arithmetic, forming polytopes). This aligns with the known computability limits of WMI for continuous spaces.

When the logic program introduces non-linear arithmetic constraints $( \mathbf { e . g . } , X ^ { 2 } + Y ^ { 2 } \leq Z )$ or arbitrary prior distributions, the resulting regions often lack closed-form antiderivatives. In these cases, the declarative ground truth $P _ { K } ( H )$ remains mathematically well-defined, but it becomes analytically incomputable.

## 8.2 Tractability and the Curse of Dimensionality

Even when the integrals possess closed-form solutions, MT-PDCL faces a strict limit on tractability driven by logical recursion.

In standard discrete logic programming, the complexity of grounding a rule grows exponentially with the number of variables in the rule. MT-PDCL resolves the impossibility of discrete grounding over uncountable domains, but it does not eliminate the underlying structural complexity. Instead, the combinatorial explosion of discrete variables translates directly into the geometric curse of dimensionality.

Every continuous variable generated by a stochastic predicate in a rule body adds an independent integration dimension to the continuous evaluation space. Furthermore, when rules are applied recursively, these dimensions accumulate. As demonstrated in the supply chain and rooted tree examples (Section 7), a derivation path of depth k requires evaluating a k-dimensional joint integral.

If evaluated using deterministic numerical quadrature (where continuous variables are approximated using a grid of m points), the computational complexity scales as $O ( m ^ { k } )$ . Consequently, while the continuous Noisy-OR operator (Section 5) successfully resolves the combinatorial explosion of the inclusion-exclusion principle for horizontal disjunctions, it does not resolve the dimensional explosion of deep vertical recursion.

## 8.3 Convergence Tractability and Non-Contractive Regimes

While the curse of dimensionality dictates the computational cost of a single iterative step, the total intractability of the system is compounded by the number of iterations required to reach the least fixed point. As established in Section 5.6, this is dictated by the recursive measure γ.

When a logic program evaluates as a strict contraction $( \gamma < 1 )$ , convergence to the declarative fixed point is geometric. A physical system can calculate a strict, finite iteration bound N using the Banach fixed-point theorem to achieve any arbitrary numerical precision ϵ. Under these conditions, the fixed point is computationally tractable.

However, when a program contains dense spatial overlap or deterministic cyclic rules, it enters the expansive propagation regime $( \gamma \geq 1 )$ . In this non-contractive state, the Banach iteration bounds fail. While Kleene’s theorem mathematically guarantees that the sequence of continuous measures converges to the exact least fixed point, it requires an infinite countable sequence of iterations (ω-continuity) to do so.

This presents a hard limit on computability. Operationally, evaluating a non-contractive continuous program in finite time forces an inference engine to halt computation at a finite depth k or apply an arbitrary tolerance cutoff. Consequently, for expansive programs, exact declarative probabilities become computationally unreachable. The engine is limited to returning lower-bound approximations of the exact measure, trading declarative precision for finite execution time.

## 8.4 Transition to Numerical and Stochastic Implementations

These theoretical limits dictate how MT-PDCL must be operationalized in practice. Exact continuous integration becomes analytically incomputable for non-linear constraints and highly intractable for deep recursion, meaning practical systems cannot rely on symbolic algebraic solvers.

To execute MT-PDCL over complex domains, the exact continuous consequence operator must be substituted with numerical approximations. The multi-dimensional integrals over the continuous variable spaces must be estimated using stochastic methods (such as Monte Carlo integration), and the local density functions must be approximated using parameterized continuous functions (such as neural networks).

This foundational paper establishes the exact declarative measure-theoretic semantics and the continuous Noisy-OR fixed-point theory. The specific algorithms, data structures, and stochastic sampling techniques required to physically implement this inference engine at scale are beyond the scope of this theoretical formalization and constitute the focus of subsequent implementation work.

## 9 Conclusion and Future Work

In this paper, we introduced MT-PDCL, a framework that generalizes probabilistic logic programming to native continuous domains. By explicitly defining independent continuous variables over bounded index domains and modeling causal events as standard Borel spaces, we eliminated the traditional reliance on finite-domain propositionalization. Under Continuous Distribution Semantics, we established a declarative semantics where exact logical entailment is mathematically defined via Lebesgue integration over the complete space of possible worlds.

To bridge this declarative foundation with operational computation, we generalized the probabilistic immediate consequence operator to continuous spaces. We demonstrated that the assumption of Independent Causal Influence (the continuous Noisy-OR) provides a mathematically bounded, operational approximation for intractable infinite unions. Crucially, we proved that this continuous operator is monotonic, preserves structural differentiability, and guarantees convergence to a well-defined least fixed point, even in non-contractive regimes where standard iteration bounds fail.

Because this paper establishes the theoretical semantics of MT-PDCL, the primary direction for future work lies in algorithmic implementation. Translating the continuous integrals of the consequence operator into efficient software requires numerical approximations, such as advanced Monte Carlo sampling or compilation into differentiable tensor operations, to evaluate the framework empirically.

A critical algorithmic next step is developing query-directed inference procedures. While this paper formalizes the computation of the complete least fixed point via bottom-up continuous evaluation, integrating over the entire unqueried state space is computationally expensive. Developing goal-directed strategies, whether through continuous top-down SLD-resolution or optimized forward-chaining transformations, combined with Constraint Logic Programming (CLP) would allow an engine to dynamically accumulate only the geometric boundaries relevant to a user’s query. This lazy evaluation strategy would bypass the need to integrate over unqueried domains, significantly improving tractability.

Once these execution architectures are established, the structural transparency of the continuous consequence operator provides a direct mechanism for modern neural-symbolic integration. Developing a Deep-MT-PDCL implementation, where neural networks predict local causal probabilities and the logic engine routes gradients transparently via backpropagation, will enable logically constrained machine learning natively within continuous domains.

Finally, achieving exact analytical inference rather than stochastic approximation requires restricting the base predicates and prior distributions. If mathematical constraints are limited to linear real arithmetic (LRA) to form computable polytopes, and prior measures are restricted to families closed under multiplication and integration (such as piecewise polynomials), the operator can yield exact, closed-form solutions. Identifying an algebraic system that is structurally closed under continuous logical reasoning, yet expressive enough for practical modeling, remains a key objective for future research.

## References

[ACF<sup>+</sup>25] S. Akshay, Supratik Chakraborty, Soroush Farokhnia, Amir Goharshady, Harshit Jitendra Motwani, and Ðorde Žikeli¯ c. Lp-based weighted model integration over non-linear real arithmetic. In´ Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25), pages 9040–9048, 2025.

[Ban22] Stefan Banach. Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales. Fundamenta Mathematicae, 3(1):133–181, 1922.

[BGR09] Chitta Baral, Michael Gelfond, and Nelson Rushton. Probabilistic reasoning with answer sets. Theory and Practice ofLogic Programming, 9(1):57–144, 2009.

[Bil12] Patrick Billingsley. Probability and Measure. John Wiley & Sons, anniversary edition, 2012.

[BJCD25] Oualid Bougzime, Samir Jabbar, Christophe Cruz, and Frédéric Demoly. Unlocking the potential of generative ai through neuro-symbolic architectures: Benefits and limitations. arXiv preprint arXiv:2502.11269, 2025.

[BPVdB15] Vaishak Belle, Andreas Passerini, and Guy Van den Broeck. Probabilistic inference in hybrid domains by weighted model integration. In Proceedings ofthe 24th International Joint Conference on Artificial Intelligence (IJCAI), 2015.

[Bry86] Randal E Bryant. Graph-based algorithms for boolean function manipulation. IEEE Transactions on Computers, 35(8):677–691, 1986.

[CYM20] William W. Cohen, Fan Yang, and Kathryn Rivard Mazaitis. Tensorlog: A probabilistic database implemented using deep-learning infrastructure. Journal ofArtificial Intelligence Research, 67:285– 325, 2020.

[Dar11] Adnan Darwiche. Sdd: A new canonical representation of propositional knowledge bases. In Proceedings of the 22nd International Joint Conference on Artificial Intelligence (IJCAI), pages 819– 826, 2011.

[DF79] John D Dollard and Charles N Friedman. Product Integration with Applications to Differential Equations, volume 10 of Encyclopedia ofMathematics and its Applications. Addison-Wesley, Reading, MA, 1979.

[DKM<sup>+</sup>20] Vincent Derkinderen, Samuel Kolb, Paolo Morettin, Paolo Frasconi, Andrea Passerini, and Luc De Raedt. Ordering variables for weighted model integration. In Proceedings ofthe 36th Conference on Uncertainty in Artificial Intelligence (UAI), pages 329–338. PMLR, 2020.

[DRKT07] Luc De Raedt, Angelika Kimmig, and Hannu Toivonen. ProbLog: A probabilistic prolog and its application in link discovery. In Proceedings ofthe 20th International Joint Conference on Artificial Intelligence (IJCAI), pages 2462–2467, 2007.

[DSZDMM<sup>+</sup>23] Lennert De Smet, Pedro Zuidberg Dos Martires, Robin Manhaeve, Giuseppe Marra, Angelika Kimmig, and Luc De Raedt. Neural probabilistic logic programming in discrete-continuous domains. In Robin J. Evans and Ilya Shpitser, editors, Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, volume 216 of Proceedings of Machine Learning Research, pages 529–538. PMLR, 31 Jul–04 Aug 2023.

[Fré51] Maurice Fréchet. Sur les tableaux de corrélation dont les marges sont données. Annales de l’Université de Lyon, Section A, 14:53–77, 1951.

[GJDR11] Bernd Gutmann, Ines Jagodic, and Luc De Raedt. Extending problog with continuous distributions. In Inductive Logic Programming: 21st International Conference (ILP), pages 178–191. Springer, 2011.

[GS04] Alejandro J García and Guillermo R Simari. Defeasible logic programming: An argumentative approach. Theory and Practice ofLogic Programming, 4(1-2):95–138, 2004.

[HB96] David Heckerman and John S Breese. Causal independence for probability assessment and inference using bayesian networks. IEEE Transactions on Systems, Man, and Cybernetics-Part A: Systems and Humans, 26(6):822–831, 1996.

[Kec95] Alexander S. Kechris. Classical Descriptive Set Theory, volume 156 of Graduate Texts in Mathematics. Springer Science & Business Media, 1995.

[KMG<sup>+</sup>25] Adem Kikaj, Giuseppe Marra, Floris Geerts, Robin Manhaeve, and Luc De Raedt. Deepgraphlog for layered neurosymbolic ai. arXiv preprint arXiv:2509.07665, 2025.

[KMR20] Samuel Kolb, Pedro Zuidberg Dos Martires, and Luc De Raedt. How to exploit structure while solving weighted model integration problems. In Ryan P. Adams and Vibhav Gogate, editors, Proceedings ofthe 35th Conference on Uncertainty in Artificial Intelligence (UAI), volume 115 of Proceedings ofMachine Learning Research, pages 744–754. PMLR, 2020.

[Llo87] John W Lloyd. Foundations of Logic Programming. Springer-Verlag, Berlin, Heidelberg, 2nd edition, 1987.

[MDK<sup>+</sup>21] Robin Manhaeve, Sebastijan Dumanciˇ c, Angelika Kimmig, Thomas Demeester, and Luc De Raedt.´ Neural probabilistic logic programming in deepproblog. Artificial Intelligence, 298:103504, 2021.

[MHLV15] Steffen Michels, Arjen Hommersom, Peter JF Lucas, and Marina Velikova. Approximate probabilistic inference with bounded error for hybrid probabilistic logic programs. In Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI), 2015.

[MMR<sup>+</sup>05] Brian Milch, Bhaskara Marthi, Stuart Russell, David Sontag, Daniel L Ong, and Andrey Kolobov. BLOG: Probabilistic models with unknown objects. In Proceedings of the 19th International Joint Conference on Artificial Intelligence (IJCAI), pages 1352–1359, 2005.

[MPVdB21] Paolo Morettin, Andrea Passerini, and Guy Van den Broeck. Hybrid probabilistic inference with logical and algebraic constraints: a survey. Proceedings of the 30th International Joint Conference on Artificial Intelligence (IJCAI), 2021.

[Nil86] Nils J Nilsson. Probabilistic logic. Artificial intelligence, 28(1):71–87, 1986.

[Pea88] Judea Pearl. Probabilistic Reasoning in Intelligent Systems: Networks ofPlausible Inference. Morgan Kaufmann, San Mateo, CA, 1988.

[SAMP24] Giuseppe Spallitta, S Akshay, Paolo Morettin, and Andrea Passerini. Lp-based weighted model integration over non-linear real arithmetic. In Proceedings ofthe 33rd International Joint Conference on Artificial Intelligence (IJCAI), 2024.

[SK97] Taisuke Sato and Yoshitaka Kameya. PRISM: A language for symbolic-statistical modeling. In Proceedings of the 15th International Joint Conference on Artificial Intelligence (IJCAI), pages 1330–1335, 1997.

[Sla07] Antonín Slavík. Product Integration, its History and Applications. Matfyzpress, Prague, 2007.

[VDB09] Joost Vennekens, Marc Denecker, and Maurice Bruynooghe. Cp-logic: A language of causal probabilistic events and its relation to logic programming. Theory and Practice of Logic Programming, 9(3):245–308, 2009.

[vEK76] Maarten H. van Emden and Robert A. Kowalski. The semantics of predicate logic as a programming language. Journal ofthe ACM (JACM), 23(4):733–742, 1976.
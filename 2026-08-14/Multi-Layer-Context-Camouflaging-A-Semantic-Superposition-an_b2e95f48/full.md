arXiv preprint

Comment: This is a preprint. A version of this manuscript has been submitted to Springer Nature Machine Learning for peer review (manuscript 8123c4c3-1d4d-4b2a-afbd-f3c1dde183f4). Subjects: Artificial Intelligence (cs.AI); Computers and Society (cs.CY)

# Multi-Layer Context Camouflaging: A Semantic Superposition and Contextual Lamination Framework for Malpractice-Resilient Online Assessment

Lovi Raj Gupta¹˒\*, Kamalpreet Kaur¹, Sri Ram¹, Prajithaa¹ ¹Lovely Professional University, Punjab, India \*Corresponding author: loviraj@gmail.com

## Abstract

This paper extends the Multi-dimensional Spatio-Temporal Context Camouflaging Model (MSCCM) introduced in prior work on the MARS assessment-resilience suite, developing a substantially more detailed mathematical treatment of its Context Camouflaging Operator. We formalize Context Camouflaging as a semantic superposition process, termed the Multi-Layer Context Camouflaging Theory (MCCT), in which an authentic semantic stream and a generated camouflage stream coexist within a single rendered surface without ordinary concatenation.

The revised formulation separates the rendered surface from the sequence an adversary actually recovers, and defines an extraction-channel operator that makes explicit which attributes (glyph, colour, position, time) survive each capture route. Against that model we develop six coupled constructs: a Context Inversion Operator with locality and vocabulary-closure conditions; a Contextual Lamination Operator driven by a keyed insertion set; a Separation Channel that generalizes colour to any modality carrying a perceptual discriminability margin; a Human Readability Functional conditioned on that margin; a Computational Ambiguity Functional redefined as the conditional entropy of the authentic stream given the adversary view, for which we derive the closed form A = log C(n+m, m); and a Context Camouflage Tensor coupling the scheme to MARS temporal rendering.

We also correct the direction of the preservation constraint. Recoverability for the legitimate viewer is an exact filtering identity rather than a similarity bound, so the operative constraints become a fidelity identity, an obfuscation ceiling on Sim(Q, Ω), and a plausibility ceiling on the detectability of camouflage tokens under a language-model prior. Eight theorems follow, including a closed-form ambiguity result, a corrected optimal-density result with a binding-constraint corollary, a semantic-filter degradation bound, a multi-observation leakage theorem showing that frame-granular re-lamination is unsafe against a multi-capture adversary, and a coupon-collector bound on capture complexity under temporal multiplexing. We close with an algorithm, its complexity, and a pre-registered evaluation protocol. No empirical results are claimed.

Index Terms context camouflaging, semantic superposition, contextual lamination, assessment integrity, computational ambiguity, threat modelling, conditional entropy, perceptual discriminability, academic malpractice.

## I. INTRODUCTION

Online assessment platforms remain vulnerable to a class of malpractice in which the rendered content of an assessment item, not merely the candidate's behaviour around it, is extracted by screenshot, copy-paste, or automated scraping and relayed to an unauthorized solver, whether human or algorithmic. Prior work on the Multi-modal Assessment Resilience Suite (MARS) introduced Context Camouflaging as one operator within a broader six-dimensional dynamical model of assessment integrity, defined at the level of a single transformation Φ mapping an original question Q to a semantically equivalent rendering Q′ [1]. That formulation established the existence of the operator and a coarse semanticsimilarity constraint, but did not develop the internal structure of Φ in mathematical depth.

This paper develops that internal structure. We show that Context Camouflaging is best understood not as a single content-rewriting step but as a semantic superposition: the authentic content and a separately generated camouflage stream are rendered together on a shared surface, laminated at keyed positions, and separated perceptually so that a human and an automated extraction pipeline read different documents from the same pixels.

Three things changed in the course of that development, and it is worth stating them plainly at the outset rather than burying them in the analysis. First, a scheme of this kind cannot be evaluated without naming the adversary. Colour survives a DOM scrape and dies in a clipboard copy, and a scheme whose whole security rests on colour therefore has entirely different properties against the two. Section III-B introduces an explicit extraction-channel model and a five-class adversary taxonomy, and every later claim is stated relative to a named class.

Second, the preservation constraint in [1] pointed the wrong way. Requiring Sim(Q, Ω) ≥ θ asks the laminated field to resemble the authentic item, which is precisely what makes it useful to a solver holding the extracted text. The legitimate viewer does not need similarity at all, because filtering by the separation channel returns Q exactly. Section IX replaces the single similarity floor with a fidelity identity, an obfuscation ceiling, and a plausibility ceiling, which is the constraint set the design actually has to satisfy.

Third, dynamic lamination as originally described leaks. If the authentic tokens stay fixed while the camouflage layer is resampled every frame, then across a handful of captures the authentic tokens are simply the ones that never change. Theorem 7 quantifies the decay and Section X gives the condition under which temporal variation is safe. This result runs against the intuition that more randomization means more security, and it is the finding we would most want a reviewer to check.

Contributions. (i) An explicit extraction-channel model and adversary taxonomy for rendered-content attacks on assessment items. (ii) A closed-form expression for computational ambiguity as conditional entropy, $\mathbf { A } _ { \mathrm { c } } \ =$ log<sub>2</sub> $\mathrm { { C } ( n + m , \ m ) }$ , replacing the unigram-entropy difference used previously. (iii) A corrected three-part constraint set with a Lagrangian treatment of camouflage density. (iv) A generalized separation channel with a perceptual discriminability margin expressed in CIEDE2000 units, together with a colour-visiondeficiency condition. (v) Eight theorems, of which four are new, including the multi-observation leakage result and a capture-complexity bound under temporal multiplexing. (vi) A stated algorithm with complexity analysis and a pre-registered evaluation protocol.

Sections II and III cover related work, the MARS recap, and the threat model. Sections IV to XI develop the constructs. Section XII proves the eight theorems, Section XIII gives the algorithm, Section XIV the evaluation protocol, and Sections XV and XVI discuss limitations and conclude.

## II. RELATED WORK

This paper is a direct mathematical extension of the MARS assessment-resilience framework, in which assessment integrity is modelled as a coupled dynamical system rather than a set of independently engineered safeguards, and the Context Camouflaging Operator is introduced as one of its six coupled constructs [1]. The present work develops that operator's internal mathematics without altering the surrounding system model. Dawson documents the limitations of treating browser lockdown, webcam surveillance, and behavioural analytics as independent, uncoordinated safeguards, motivating systems that address the rendered-content channel directly rather than only the candidate's behaviour around it [2].

The mechanism exploited here, a deliberate gap between what a human sees and what a machine parses, has a direct precedent in security research that we did not cite in the earlier version and should have. Boucher et al. showed that Unicode homoglyphs, invisible characters, and bidirectional reordering let an attacker construct text whose visual rendering and logical encoding disagree, and used that gap to attack NLP pipelines in a black-box setting [7]. Boucher and Anderson extended the same idea to source code, where the compiler and the reviewer read different programs from one file [8]. MCCT inverts the polarity of that attack: the same rendering-versus-encoding gap is used defensively, and the party disadvantaged by the gap is the extraction pipeline rather than the human. Reading the two literatures together also sets a useful expectation. Those attacks were eventually mitigated by input sanitization at the parser, and the analogous countermeasure here is an adversary that normalizes the rendered surface before parsing, which is exactly the A4 and A5 classes of Section III-B.

The claim that a human can filter a colour-separated stream at negligible cost rests on visual-search results rather than on assertion. Treisman and Gelade established that a target differing from distractors in a single primitive feature such as hue is detected in time roughly independent of the number of distractors, and Wolfe's Guided Search work refines the conditions under which that independence holds and where it breaks down [9], [10]. Both point to the same practical requirement: the separation must be a single preattentive feature with an adequate discriminability margin, otherwise search becomes serial and readability degrades with camouflage density. We express that margin in CIEDE2000 units [11], with the commonly cited average just-noticeable difference of roughly 2.3 CIELAB units as the lower reference point [12]. Colour-vision deficiency is handled by requiring the margin to survive dichromatic simulation [13], or by moving the separation to a non-chromatic modality.

The Context Inversion Operator generalizes a simple negation-token substitution into an operator that draws on lexical-semantic resources; WordNet's synonym and antonym relations provide one practical basis for generating contextdependent, polarity-reversing substitutions without relying on a fixed vocabulary list [3]. Computing the semantic similarity that appears in the obfuscation constraint can be operationalized with transformer-based sentence encoders: BERT provides contextual token representations from which sentence-level meaning can be derived [4], and Sentence-BERT adapts this into an efficient, directly comparable sentence-embedding form [5]. Shannon's information theory supplies the entropy formalism used throughout to quantify computational ambiguity and to bound the behaviour of dynamic lamination [6].

What remains unaddressed in the prior literature, and what this paper supplies, is a treatment of the perceptual gap as a designed defensive primitive with a stated adversary model, a closed-form ambiguity measure, and an account of how the guarantee degrades against a semantically informed attacker.

## III. MARS SYSTEM MODEL AND THREAT MODEL

## A. Recap of the MARS state

MARS represents an assessment session as the coupled state tuple

$$
{ \bf M } = \langle \mathrm { { \bf ~ S } ( t ) } , \mathrm { { \cal ~ T } ( t ) } , \mathrm { { \bf ~ C } ( t ) } , \mathrm { { \bf ~ B } ( t ) } , \mathrm { { \bf ~ R } ( t ) } , \mathrm { { \cal ~ I } ( t ) } \rangle\tag{1}
$$

comprising spatial, temporal, contextual, behavioural, rendering, and integrity components [1]. The present paper is concerned with the contextual domain C(t) and its associated Context Camouflaging Operator Φ, previously defined at the coarse level $\mathrm { ~ Q ' ~ } = \ \Phi ( \mathrm { Q } , \mathrm { S } , \mathrm { T } , \mathrm { C } )$ subject to $\sin ( \mathrm { Q } , \mathrm { Q } ^ { \prime } ) ~ \ge ~ \theta$ Everything that follows refines this single operator into the Multi-Layer Context Camouflaging Theory (MCCT) of Sections IV to XI, culminating in a unified rendering equation (Section XI) that reconnects MCCT to the rendering dynamics established in [1].

One notational change is needed before proceeding. The symbol C was used in the earlier version for the contextual state $\mathrm { C ( t ) } ,$ , the colour assignment $\mathrm { C } ( \mathbf { w } ) .$ , and the binomial coefficient C(n+m, m). We retain C(t) and $\operatorname { C } ( \cdot , \cdot )$ and rename the colour assignment to χ(·).

## B. Extraction channels and adversary classes

A rendered assessment item is not a token sequence. It is a set of glyphs carrying position, colour, and a time index, and the question of what an attacker obtains is the question of which of those attributes survive the capture route. Writing the rendered surface as

$$
\Sigma ( \mathfrak { t } ) = ( \sigma _ { 1 } , . . . , \sigma _ { \mathrm { N } } ) , \sigma _ { \mathrm { i } } = ( \Omega _ { \mathrm { i } } , \mathrm { x } _ { \mathrm { i } } , \mathrm { y } _ { \mathrm { i } } , \mathrm { c } _ { \mathrm { i } } , \tau _ { \mathrm { i } } ) , \mathrm { N } = \mathbf { n } + \mathbf { m }\tag{2}
$$

an extraction channel is a projection

$$
\mathrm { X _ { A } } : \Sigma ( \mathrm { t _ { 1 : k } } ) \mapsto \mathrm { V _ { A } } , ~ \mathrm { V _ { A } } = \mathrm { X _ { A } } ( \Sigma ( \mathrm { t _ { 1 } } ) , . . . , \Sigma ( \mathrm { t _ { k } } ) )\tag{3}
$$

where $\mathrm { V _ { A } }$ is the adversary view and k the number of captures. Table I records which attributes each channel preserves. The classes are ordered by increasing capability, and every security claim later in the paper is stated against a named class rather than against an unspecified attacker.

Two entries deserve emphasis because they bound what the scheme can honestly claim. Class A3, a scripted reader of the document object model, sees the styling that defines the separation channel; colour separation alone therefore provides no protection against A3 unless the styling is rendered nonrecoverable, for example by drawing to a canvas or by randomizing per-token class names so that authentic and camouflage tokens are not distinguishable by selector. Class A5, a vision-language model given a screenshot, also sees colour, and additionally supplies the semantic prior that makes the combinatorial bound of Theorem 5 collapse. Colour separation is a defence against A1 and A2. Against A3 to A5 the defence has to come from elsewhere in the MARS stack, principally from the temporal visibility term of Section XI, and Theorem 8 states what that buys.

Table I  
EXTRACTION CHANNELS AND ATTRIBUTE SURVIVAL
<table><tr><td>Class</td><td>Channel</td><td>Glyph</td><td>Colour</td><td>Position</td><td>Semantics</td></tr><tr><td>A1</td><td>Clipboard copy to plain text</td><td>yes</td><td>no</td><td>order only</td><td>none</td></tr><tr><td>A2</td><td>Screenshot then OCR</td><td>yes</td><td>no</td><td>yes</td><td>none</td></tr><tr><td>A3</td><td>Scripted DOM reader</td><td>yes</td><td>yes</td><td>yes</td><td>none</td></tr><tr><td>A4</td><td>A1 or A2 plus a language-model filter</td><td>yes</td><td>no</td><td>yes</td><td>yes</td></tr><tr><td>A5</td><td>Screenshot to a vision-language model</td><td>yes</td><td>yes</td><td>yes</td><td>yes</td></tr></table>

Attribute survival by capture route. "Semantics" denotes whether the adversary can score candidate reconstructions for linguistic plausibility. Colour separation is effective against A1 and A2 only.

Following Kerckhoffs's principle we assume throughout that the adversary knows the algorithm, the inversion operator, the density $\boldsymbol { \mathsf { \Pi } } _ { \mathsf { \Pi } } \boldsymbol { \mathsf { \Pi } } _ { \mathsf { \Pi } }$ and the distribution from which insertion sets are drawn. Only the per-session key is secret. Claims that depend on the adversary not knowing the construction are not made.

![](images/f80d7913f604e17415b545f69744fd4b1ca2fbb8244f34cd2f2af3060c33bb9c.jpg)  
Fig. 1. Architecture of the Multi-Layer Context Camouflaging Theory (MCCT): the authentic stream and its inversion-generated camouflage stream are laminated, separated by channel, and passed to the MARS rendering engine.

## IV. SEMANTIC SUPERPOSITION AND THE RENDERED SURFACE

Rather than presenting an assessment item as a single token sequence, MCCT renders a superposed linguistic field in which two token streams occupy the same rendered surface, only one of which contributes to the intended meaning. The authentic semantic stream is

$$
\mathbf { Q } = ( \mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } , . . . , \mathbf { w } _ { \mathrm { n } } ) , \mathbf { w } _ { \mathrm { i } } \in \mathrm { V }\tag{4}
$$

where V denotes the assessment-item vocabulary. A separately generated camouflage stream

$$
\tilde { \bf Q } = ( \tilde { \bf w } _ { 1 } , \tilde { \bf w } _ { 2 } , . . . , \tilde { \bf w } _ { \mathrm { m } } )\tag{5}
$$

is produced by the Context Inversion Operator of Section V. The two streams are combined, not by concatenation but by the lamination process of Section VI, into a superposed field

$$
\Omega = \mathcal { A } ( \mathrm { Q } , \tilde { \mathrm { Q } } ; \Lambda )\tag{6}
$$

The distinction that the earlier version left implicit is between Ω and Σ. The laminated field Ω is a token sequence; the rendered surface Σ of Equation (2) is Ω decorated with position, colour, and visibility. A legitimate candidate observes Σ. An adversary observes $\mathrm { X } _ { \mathrm { A } } ( \Sigma )$ , which for classes A1 and A2 is Ω alone. The defining property of the construction can now be stated exactly rather than informally: there exists a filter $\mathcal { F }$ such that ℱ(Σ; Γ) = Q with no error at all, while Q is not determined by Ω. Sections V to VII build the three mechanisms that produce this asymmetry and Sections VIII and IX quantify it.

## V. THE CONTEXT INVERSION OPERATOR

The camouflage stream is generated by a Context Inversion Operator ℐ acting on the authentic stream,

$$
\tilde { \mathbf { Q } } = \mathcal { J } ( \mathbf { Q } )\tag{7}
$$

Following the generalization principle adopted for this framework, $\mathcal { J }$ produces a context-preserving, computationally ambiguous overlay rather than a fixed vocabulary substitution. One embodiment realizes $\mathcal { J }$ using polarity-reversing tokens ("NOT", "EXCEPT", "UNEQUAL", "UNLESS"); another may draw on qualifiers, antonyms, distractor clauses, or domainspecific semantic modifiers, for instance using the lexical antonym and hypernym relations catalogued in WordNet [3].

The earlier version described what $\mathcal { J }$ should do in prose. Stating it as three conditions makes the later theorems checkable. Let $\mathsf { s } ( \mathrm { j } ) \subseteq \{ 1 , . . . , \mathtt { n } \}$ be the source span from which the j-th camouflage token is derived. Then $\mathcal { J }$ is admissible when

$$
( \mathrm { C } 1 ) \mathrm { l o c a l i t y } ; \tilde { \mathrm { w } } _ { \mathrm { j } } = \mathrm { g } ( \mathrm { w } _ { \mathrm { i } } : \mathrm { i } \in \mathrm { s } ( \mathrm { j } ) ) , | \mathrm { s } ( \mathrm { j } ) | \le \ell\tag{8}
$$

$$
( \mathrm { C 2 } ) \mathrm { \ v o c a b u l a r y \ c l o s u r e : \ m i n _ { v } \in \vee d i s t ( \tilde { w } _ { j } , v ) \leq \rho v }\tag{9}
$$

$$
( \mathrm { C 3 } ) \mathrm { p o l a r i t y ~ r e v e r s a l } { : } ~ \langle \mathrm { e ( \tilde { w } _ { j } ) , e ( w _ { s ( j ) } ) } \rangle \leq - \kappa _ { \mathrm { p } }\tag{10}
$$

where e(·) is a sentence-encoder embedding, dist a metric on that embedding space, ℓ a span-width bound, and $\rho \mathrm { v }$ and $\kappa _ { \mathrm { p } }$ tunable margins. Condition C1 keeps the operator local, so that $\tilde { \mathrm { { Q } } }$ is anchored to Q span by span rather than to the global meaning of Q. Condition C2 keeps camouflage tokens close to the item vocabulary, which is what prevents an adversary from separating the streams by a vocabulary test alone. Condition C3 is what stops $\tilde { \mathrm { { Q } } }$ from being a usable paraphrase of $\mathrm { Q } ,$ and therefore what stops the laminated field from leaking the answer to a solver that reads it whole.

C2 and C3 pull against each other, and the tension is real rather than an artefact of the formalism. Pushing $\kappa _ { \mathrm { p } }$ up drives the camouflage tokens away from the item vocabulary and makes them easier to spot; pushing ρ<sub>V</sub> down brings them back into the vocabulary and weakens the polarity reversal. The plausibility constraint of Section IX is where that trade-off is made explicit.

## VI. THE CONTEXTUAL LAMINATION OPERATOR

The defining departure from ordinary text insertion is that Q and $\tilde { \mathrm { { Q } } }$ are combined by lamination rather than concatenation. The operator

$$
\Omega = \mathcal { A } ( \mathrm { Q } , \tilde { \mathrm { Q } } ; \Lambda )\tag{11}
$$

is governed by a control law

$$
\Lambda = ( \pi , \kappa , \chi )\tag{12}
$$

where π is the insertion position map, κ the insertion density, and $\chi$ the separation-channel assignment of Section VII. The map π determines an insertion set

$$
\Pi \subseteq \{ 1 , . . . , \mathrm { n } + \mathrm { m } \} , \ | \Pi | = \mathrm { m } , \ \kappa = \mathrm { m } / \left( \mathrm { n } + \mathrm { m } \right)\tag{13}
$$

and the laminated field is defined position by position as

$$
\Omega _ { \mathrm { i } } = \mathbf { w } _ { \mathrm { r ( i ) } } \mathrm { i f } \mathrm { i } \notin \Pi ; \tilde { \mathbf { w } } _ { \mathrm { j ( i ) } } \mathrm { i f } \mathrm { i } \in \Pi\tag{14}
$$

where r(i) and j(i) index the authentic and camouflage tokens occupying laminated position i, both order-preserving. Because Π is chosen independently of concatenation order, Ω cannot be separated into its constituent streams by position alone; recovering Q from Ω requires knowledge of Π, formalized as Theorem 5.

Where Π comes from was left open previously, and it matters, because a Π that is merely "random" gives no reproducibility across the render path and no rotation policy across sessions. We derive it from a keyed pseudorandom function,

$$
\Pi = \Psi _ { \mathrm { P R F } } ( \mathsf { K } _ { \mathrm { s e s s } } , \mathrm { i d } , 0 ) , \Gamma = ( \mathsf { c } _ { \mathsf { q } } , \mathsf { c } _ { \mathsf { c } } ) = \Psi _ { \mathrm { P R F } } ( \mathsf { K } _ { \mathrm { s e s s } } , \mathrm { i d } , 1 )\tag{15}
$$

with $\mathrm { K } _ { \mathrm { s e s s } }$ a per-session key held by the rendering service with 0 and 1 acting as domain separators. The pair (Π, Γ) is the session rendering key. Sampling Π uniformly among the msubsets of $\{ 1 , . . . , \mathrm { n ^ { + } m } \}$ is what makes the uniform prior of Theorem 3 the correct one; any biased sampler reduces the adversary's uncertainty below the closed form and should be treated as a defect rather than a tuning choice.

![](images/5149646cabff160b47f6869923f71b60a52a8f659567bc12327608c9da50e1c4.jpg)  
Fig. 2. Contextual lamination of an authentic stream (white) with inversiongenerated camouflage tokens (shaded) at keyed insertion set Π.

## VII. THE SEPARATION CHANNEL AND PERCEPTUAL DISCRIMINABILITY

Humans exploit colour as an immediate preattentive grouping cue; extraction pipelines in classes A1 and A2 tokenize rendered or copied text after formatting metadata has been discarded by the copy, OCR, or scrape operation. MCCT exploits this asymmetry through a separation function

$$
\chi ( \mathrm { i } ) { = } \tt c _ { \mathrm { q } } \mathrm { i f } { \mathrm { i } } \notin \Pi ; \ c _ { \mathrm { c } } \mathrm { i f } { \mathrm { i } } \in \Pi\tag{16}
$$

which assigns the authentic-stream value $\mathrm { c _ { q } }$ to positions drawn from Q and the camouflage value c to positions drawn from ${ \tilde { \mathrm { Q } } } .$ The pair $\Gamma = ( \mathrm { c _ { q } } , \mathrm { c _ { c } } )$ , together with Π, constitutes the session rendering key required to invert the lamination, and Γ may be rotated across sessions in the same manner as other MARS session parameters.

Treating $\chi$ as specifically chromatic was an unnecessary restriction. What the construction requires of the separation channel is only that it be a single preattentive feature carrying a discriminability margin above threshold, so we state the requirement rather than the mechanism. For a chromatic channel the condition is

$$
\Delta \mathrm { E } _ { 0 0 } ( \mathrm { ~ c } _ { \mathrm { q } } , \mathrm { c } _ { \mathrm { c } } ) \ge \Delta \mathrm { E } ^ { * } , \ \Delta \mathrm { E } ^ { * } \gg 2 . 3\tag{17}
$$

with ΔE<sub>00</sub> the CIEDE2000 colour difference [11] and 2.3 CIELAB units the commonly cited average just-noticeable difference [12]. A margin near the JND is the wrong operating point, because the readability result of Theorem 2 depends on the difference being preattentive rather than merely detectable. Accessibility imposes a second condition: the margin has to survive dichromatic simulation,

$$
\operatorname* { m i n } _ { \textrm { D } \in \{ \mathrm { p r o t , d e u t , t r i t } \} } \Delta \mathrm { E } _ { 0 0 } ( \textrm { D } ( \mathrm { c } _ { \mathrm { q } } ) , \mathrm { D } ( \mathrm { c } _ { \mathrm { c } } ) ) \geq \Delta \mathrm { E } ^ { * }\tag{18}
$$

with D the standard dichromat simulations [13]. Where Equation (18) cannot be met, the channel moves to a nonchromatic modality (weight, case, or a typographic mark) under the same $\chi$ with a different codomain, and the whole analysis carries over unchanged because no theorem below uses any property of colour other than the existence of the margin. This closes a limitation that the earlier version flagged and left open.

One honest caveat belongs here rather than in the discussion. Equations (17) and (18) make the channel more visible, and visibility to the candidate is visibility to a screenshot. Raising

ΔE\* strengthens the guarantee against A1 and A2 and weakens nothing there, but it makes the separation trivially legible to A3 and A5. The channel is not a secret; it is a filter that certain capture routes destroy.

![](images/065268ff41e357b7736193f421294ba7ff8541a10982e9b99763cc2e3ed25d16.jpg)  
Fig. 3. Dual-channel perception under separation: a legitimate candidate perceiving Γ filters the laminated stream preattentively, while a class A1 or A2 extraction channel sees only the undifferentiated token sequence.

## VIII. READABILITY AND AMBIGUITY FUNCTIONALS

Two functionals quantify the trade-off that separated lamination is designed to exploit. The Human Readability Functional is

$$
\mathbf { R } _ { \mathrm { h } } = \left( 1 / \mathrm { n } \right) \Sigma _ { \mathrm { i = 1 \dots n } } \mathbf { a } _ { \mathrm { i } }\tag{19}
$$

where the perception indicator is

$\mathfrak { a } _ { \mathrm { i } } = 1$ if authentic token i is correctly perceived;

= 0 otherwise

(20)

The earlier statement that $\mathrm { R _ { h } }  1$ independently of density is true but empty, because it assumes exactly what it concludes. What makes it a claim about the world is the condition under which ${ \bf { q } } _ { \mathrm { ~ i ~ } } = { \bf { \epsilon } } 1$ holds, and visual-search theory supplies it: separation by a single preattentive feature yields search time approximately flat in distractor count, whereas a belowthreshold margin forces serial search and readability then falls with density [9], [10]. We therefore model perception as

$\mathrm { P r } [ \mathrm { \ a i } = 1 ] = 1 - \epsilon ( \Delta \mathrm { E } 0 0 )$ , with ϵ decreasing in ΔE00 and ϵ → 0 for $\Delta \mathrm { E } 0 0 \geq \Delta \mathrm { E } ^ { \ast }$ (21)

and a companion search-cost term that Section IX prices,

$$
\mathrm { S c } ( \eta ) = \mathbf { a } + \mathbf { b } { \cdot } \boldsymbol { \eta } { \cdot } 1 [ \Delta \mathrm { E } 0 0 < \Delta \mathrm { E } ^ { * } ]\tag{22}
$$

so that above threshold the cost of camouflage to the candidate is a constant a and below threshold it grows linearly in $\boldsymbol { \mathsf { \Pi } } ^ { \mathsf { \Pi } }$ Theorem 2 is stated against Equation (21) rather than against an assumption.

The Computational Ambiguity Functional needs a more substantial correction. It was previously defined as the excess unigram entropy $\mathrm { A } _ { \mathrm { c } } = \mathrm { H } ( \Omega ) \textrm { - } \mathrm { H } ( \mathrm { Q } )$ . That quantity is not monotone in camouflage density and can fall as tokens are added: inserting the same camouflage token repeatedly concentrates the token distribution and reduces H(Ω). It also measures the wrong thing, since the adversary's difficulty is uncertainty about $\mathrm { Q } ,$ not the entropy of what is on screen. We redefine it as the conditional entropy of the authentic stream given the adversary view,

$$
\mathrm { A _ { c } = H ( \mathrm { ~ Q ~ | ~ V _ { A } ~ ) ~ } }\tag{23}
$$

For an A1 or A2 adversary, $\mathrm { V } _ { \mathrm { A } } = \Omega ,$ , and Theorem 3 gives the closed form

$$
\mathbf { A } _ { \mathrm { c } } = \log _ { 2 } \mathbf { C } ( \mathbf { n } + \mathbf { m } , \mathbf { m } )\tag{24}
$$

which is strictly increasing in m for fixed ${ \mathfrak { n } } ,$ is exactly the quantity that Theorem 5 bounds, and reduces the two results to one. For an A3 adversary the view includes $\mathbb { X } ,$ and $\mathrm { A _ { c } } = 0$ . For A4 and A5 the semantic prior reduces the effective count, which Theorem 6 bounds.

## IX. FIDELITY, OBFUSCATION, AND PLAUSIBILITY

The original framework imposed a single Context Preservation Constraint, Sim(Q, Ω) ≥ θ, on the grounds that the authentic content must remain recoverable and semantically intact. Recoverability is a separate matter from similarity, and once the two are separated the constraint turns out to point the wrong way. High Sim(Q, Ω) means the flat laminated field still conveys the item, which is exactly the property that lets a solver holding the extracted text answer it. The correct design has three constraints rather than one.

Fidelity. The legitimate rendering path must return the item exactly, not approximately:

$$
\begin{array} { r } { \mathcal { \ R } \left( \Sigma ; \Gamma \right) = \mathrm { Q } \left( \mathrm { e x a c t , n o t } \mathrm { u p } \mathrm { t o } \mathrm { s i m i l a r i t y } \right) } \end{array}\tag{25}
$$

This is an identity guaranteed by Theorem 5, and it makes the similarity floor unnecessary for the purpose it was introduced to serve.

Obfuscation. The adversary view must not convey the item:

$$
\mathrm { S i m ( \Omega Q , \Omega ) } \le \theta _ { \mathrm { o b f } }\tag{26}
$$

with Sim instantiated by a sentence encoder such as Sentence-BERT [5] over contextual representations [4]. The inequality is a ceiling where [1] had a floor.

Plausibility. The laminated field must not be trivially separable by a language model, or Theorem 3 overstates the adversary's work by a wide margin. Let p be a reference model and define the per-position detectability

$$
\mathrm { \Delta \ S _ { i } = | \log \ p _ { L M } ( \Omega _ { i } \mid \Omega _ { \mathrm { < i } } ) - E _ { w \sim Q } [ \log \ p _ { L M } ( w \mid \Omega _ { \mathrm { < i } } ) ] \mid }\tag{27}
$$

The constraint is a ceiling on how far camouflage positions stand out,

$$
( 1 / \mathrm { m } ) \sum _ { \mathrm { i } \in \Pi } \delta _ { \mathrm { i } } \leq \delta _ { \mathrm { m a x } }\tag{28}
$$

Equations (26) and (28) are in tension by construction, which is the trilemma the designer actually faces: a camouflage stream distant enough to destroy Sim(Q, Ω) tends to be distant enough to be flagged by p<sub>LM</sub>, and one bland enough to pass Equation (28) tends to leave the item readable. Section XV returns to this point, and we regard it as the main open problem the framework raises.

The intensity of camouflaging is controlled by the camouflage density

$$
\mathfrak { n } = \mathfrak { m } / \mathfrak { n }\tag{29}
$$

and the design problem is a constrained maximization rather than a search for a single crossing point,

$$
\begin{array} { r } { \mathfrak { \eta } ^ { * } = \operatorname * { a r g m a x } _ { \mathfrak { \eta } \in \mathrm { F } } \left[ \operatorname { A } _ { \mathfrak { c } } ( \mathfrak { \eta } ) - \lambda \operatorname { S } _ { \mathfrak { c } } ( \mathfrak { \eta } ) \right] } \end{array}\tag{30}
$$

in which $\lambda$ prices candidate effort and the feasible set is $\mathrm { F } = \{ \mathfrak { n }$ : Equations (26) and (28) hold, and $\mathrm { R } _ { \mathrm { h } } ( \eta ) \geq 1 - \varepsilon \}$ . Theorem 4 gives conditions for $\boldsymbol \eta ^ { * }$ to exist and identifies when it sits on the boundary of F. Figure 4 shows the qualitative shape of the trade-off; the curves are schematic and carry no measured values.

![](images/e63d656ae3fab21bdf943a2dae19cb4ba3d73911d56118e2b80babf46ab40a7c.jpg)  
Fig. 4. Schematic trade-off between human readability, computational ambiguity, and semantic similarity as functions of camouflage density η, under an above-threshold separation margin. Curves are illustrative; Section XIV specifies how they would be measured.

## X. DYNAMIC LAMINATION AND THE CONTEXT CAMOUFLAGE TENSOR

Consistent with MARS's session-adaptive rendering philosophy, the camouflage stream need not be static. Dynamic lamination re-samples Q̃ over time,

$$
\Omega ( \mathfrak { t } ) = \mathcal { L } ( \mathrm { \nabla { Q } , \tilde { Q } ( \mathfrak { t } ) ; \Lambda ( \mathfrak { t } ) } )\tag{31}
$$

so that the camouflage layer evolves while the constraint set of Section IX continues to hold at every t. Every rendered token additionally carries a Context Camouflage Tensor recording its rendering state,

$$
\mathrm { T _ { i } ( t ) = \langle \ x _ { i } , y _ { i } , c _ { i } , \tau _ { i } , \sigma _ { i } , \omega _ { i } \rangle }\tag{32}
$$

where $\left( \operatorname { X } _ { \mathrm { i } } , \ \operatorname { y } _ { \mathrm { i } } \right)$ is spatial position, c<sub>i</sub> the assigned separation value, τ<sub>i</sub> visibility, $\sigma _ { \mathrm { i } }$ semantic polarity (authentic or inverted), and ω<sub>i</sub> rendering weight.

Equation (31) is where the earlier framework had a genuine vulnerability, and we would rather state it than let a reader discover it. If Q is held fixed while $\Lambda ( { \mathfrak { t } } )$ and $\tilde { Q } ( \mathfrak { t } )$ are resampled every frame, then across k captures the authentic positions are identifiable as the ones whose token never changed. Randomizing the camouflage layer more aggressively makes the leak faster, not slower. Theorem 7 gives the decay rate. The condition under which temporal variation is safe is that the perposition temporal process be identically distributed for the two classes,

$$
\mathrm { l a w } ( \Omega _ { \mathrm { i } } ( \mathrm { t } _ { 1 : \mathrm { k } } ) \vert \mathrm { i } \notin \Pi ) = \mathrm { l a w } ( \Omega _ { \mathrm { i } } ( \mathrm { t } _ { 1 : \mathrm { k } } ) \vert \mathrm { i } \in \Pi )\tag{33}
$$

Equation (33) is not satisfied by resampling only the camouflage layer. Two designs do satisfy it in practice. Either hold both Π and Q̃ fixed for the lifetime of a session and rotate only across sessions, which reduces the leak to zero at the cost of intra-session adaptivity; or vary the authentic surface too, by resampling an equivalence-preserving surface form of each authentic token on the same schedule, which preserves adaptivity at the cost of a harder inversion operator. We recommend the first as the default and treat the second as future work.

![](images/60924f11889c92a10b60ba8609a2cd6798f41faecb669bf43660d79d299e0957.jpg)  
Fig. 5. The Context Camouflage Tensor $T \_ i ( t ) .$ each rendered token carries position, separation value, visibility, semantic polarity, rendering weight, and a time index.

## XI. UNIFIED RENDERING EQUATION

MCCT connects to the rendering dynamics of the original MARS framework through a unified rendering equation. Recall from [1] the per-word visibility function

$$
\Psi _ { \mathrm { i } } ( \mathrm { t } ) = 1 [ \sin ( 2 \pi \mathrm { f _ { i } } \mathrm { t } + \phi _ { \mathrm { i } } + \psi ) - \tau \geq 0 ]\tag{34}
$$

and let $\delta _ { \mathrm { i } } ( \mathrm { t } )$ denote the positional-drift function governing token i's spatial rendering.

The previous version wrote the rendering as a scalar product $\Psi _ { \mathrm { i } } ( \mathrm { t } ) { { \cdot } \delta _ { \mathrm { i } } ( \mathrm { t } ) { { \cdot } \Omega _ { \mathrm { i } } { \cdot } \mathrm { C } ( \mathrm { w } _ { \mathrm { i } } ) } }$ , which multiplies a binary indicator, a displacement, a token, and a colour. Those are not commensurable quantities and the expression has no defined value. The rendering is a map into a glyph state, so we write it as one:

$$
\mathcal { R } _ { \mathrm { i } } ( \mathrm { t } ) = \langle \Omega _ { \mathrm { i } } , \ \mathrm { p } _ { \mathrm { i } } + \delta _ { \mathrm { i } } ( \mathrm { t } ) , \ \chi ( \mathrm { i } ) , \ \Psi _ { \mathrm { i } } ( \mathrm { t } ) \cdot \mathrm { \omega } _ { \mathrm { o } _ { \mathrm { i } } } \rangle\tag{35}
$$

whose four components are the glyph, the drifted position, the separation value, and the opacity. Only the last is a scalar, and the other three keep their own types. The rendered surface of Equation (2) is the collection $\{ \mathcal { R } _ { \mathrm { i } } ( \mathrm { t } ) \}$ , and the visible fraction at any instant is

$$
\mathsf { p } ( \mathrm { t } ) = \left( { 1 } / { \mathsf { n } } \right) \sum \sb { \mathrm { i } } \notin \Pi \Psi \sb { \mathrm { i } } ( \mathrm { t } )\tag{36}
$$

which is the parameter that Theorem 8 turns into a capturecount bound. Equation (35) is the point at which MCCT rejoins, and becomes a fully specified special case of, the general MARS Rendering Tensor formalism.

## XII. THEORETICAL ANALYSIS

The following eight results characterize the joint behaviour of the constructs of Sections IV to XI. Theorems 1, 2, 4, and 5 revise results stated in the earlier version; Theorems 3, 6, 7, and 8 are new. Each is stated against a named adversary class from Table I.

Theorem 1 (Separation-Conditioned Entropy Collapse). Let Σ be the rendered surface of Equation (2) and Γ the separation key. For an observer of Σ who resolves the separation channel, $\mathrm { H } ( \mathrm { Q } \mid \Sigma , \Gamma ) = 0$ exactly. For an adversary of class A1 or A2, whose view is Ω, $\mathrm { H } ( \mathrm { Q } \mid \Omega ) = \mathbf { A } _ { \mathrm { c } } > 0$ whenever $\mathbf { m } \geq 1$

Proof. Resolving the channel yields, at every position i, the indicator $1 [ \chi ( \mathrm { i } ) = \mathbf { c } _ { \mathrm { c } } ]$ and therefore the set Π exactly. Deleting the positions in Π and reading the remainder in order returns Q, by the order-preserving piecewise definition of Equation (14). Q is thus a deterministic function of (Σ, Γ), and the conditional entropy of a deterministic function is zero. The adversary of class A1 or $_ { \mathrm { A } 2 }$ receives Ω with $\chi$ discarded, so Π is not determined, and the residual uncertainty is $\mathbf { A } _ { \mathrm { c } }$ by Definition (23), which Theorem 3 shows is positive for m $\geq 1$ . ∎

This strengthens the earlier statement, which carried a residual term ε for positional ambiguity within the authentic class. No such term is needed. Lamination is order-preserving, so once Π is known the authentic subsequence is recovered without ambiguity, and the collapse is exact rather than approximate.

Theorem 2 (Readability and Ambiguity Decouple Above the Discriminability Threshold). If $\Delta \mathrm { E } _ { 0 0 } ( \mathbf { c } _ { \mathrm { q } } , \mathbf { c } _ { \mathrm { c } } ) \geq \Delta \mathrm { E } ^ { * }$ , then $\mathrm { E [ R _ { h } ] }$ $= 1 - \epsilon$ with ϵ independent of $\boldsymbol { \mathsf { \Pi } } _ { \mathsf { \Pi } } \boldsymbol { \mathsf { \Pi } } _ { \mathsf { \Pi } }$ and the search cost $\mathrm { S } _ { \mathrm { c } } ( \eta )$ of Equation $( 2 2 )$ is constant in $\boldsymbol { \mathsf { \Pi } } _ { \mathsf { \Pi } } \boldsymbol { \mathsf { \Pi } } _ { \mathsf { \Pi } }$ while $\mathbf { A } _ { \mathrm { c } } ( \eta )$ is strictly increasing in η. If $\Delta \mathrm { E } _ { 0 0 } < \Delta \mathrm { E } ^ { * }$ , decoupling fails: ϵ grows with m and $\mathrm { S _ { c } }$ grows linearly in $\boldsymbol { \mathrm { \mathsf { n } } } .$

Proof. Under Equation (21) the indicators ${ \bf { d } } _ { \mathrm { { i } } }$ are Bernoulli with a common failure probability $\epsilon ( \Delta \mathrm { E } _ { 0 0 } )$ that depends on the separation margin and not on the number of camouflage tokens, which is the defining property of a preattentive feature-search regime $[ 9 ] , [ 1 0 ] .$ Linearity of expectation applied to Equation (19) gives $\mathrm { E } [ \mathbf { R } _ { \mathrm { h } } ] = 1 - \epsilon$ for every η. With the indicator in Equation (22) false, $\mathrm { S } _ { \mathrm { c } } ( \eta ) = \mathbf { a }$ . Strict monotonicity of $\mathbf { A } _ { \mathrm { c } }$ in η follows from Theorem 3, since $\mathrm { C } ( { \mathrm { n } } { + } { \mathrm { m } } , { \mathrm { m } } )$ is strictly increasing in m for fixed $\mathbf { n } \geq 1 .$ . Below threshold the discrimination is no longer a single-feature pop-out, search becomes serial in the distractor count, ϵ acquires a dependence on m, and the second term of Equation (22) activates. ∎

The distinction matters for deployment. The earlier version obtained decoupling by assuming ${ \mathfrak { a } } _ { \mathrm { i } } = 1$ , which makes the conclusion true by construction. Here decoupling is a consequence of a physical condition on the rendered colours, and it is a condition that a platform can verify at render time.

Theorem 3 (Closed-Form Computational Ambiguity). Let Π be drawn uniformly among the m-subsets of $\{ 1 , . . . , \mathrm { n ^ { + } m } \}$ and let the adversary view be Ω. Then $\operatorname { A } _ { \mathrm { c } } = \operatorname { H } ( \operatorname { Q } \mid \Omega ) \leq \log _ { 2 } \operatorname { C } ( \mathrm { n } +$ m, m), with equality when the candidate streams induced by distinct subsets are distinct. At unit density $\mathbf { m } = \mathbf { n } ,$ , Stirling's approximation gives $\mathrm { A _ { c } } \approx 2 \mathrm { n } - \mathrm { \mathrm { ^ { 1 } } } / _ { 2 } \log _ { 2 } ( \pi \mathrm { n } )$ .

Proof. Each candidate subset S of size m induces a candidate authentic stream $\mathrm { Q } _ { \mathrm { s } } = ( \Omega _ { \mathrm { i } } : \mathrm { i } \notin \mathrm { S } )$ . By Equation (14) the true Q equals $\mathrm { Q } _ { \Pi } .$ The posterior over candidates given Ω is the pushforward of the uniform prior on Π under $\mathrm { S } \mapsto \mathrm { Q } _ { \mathrm { S } } . \mathrm { A }$ uniform distribution on a set of size N has entropy log<sub>2</sub> N, and pushforward under a map that is not injective can only merge outcomes and reduce entropy, so $\mathrm { H } ( \mathbf { Q } \ | \ \Omega ) \le \log _ { 2 } \mathbf { \Lambda } \mathrm { C } ( \bar { \mathrm { n } } { + } \mathrm { m } , \bar { \mathrm { m } } )$ with equality exactly when the map is injective. Repeated tokens in Ω are the only source of collisions. The unit-density expression follows from $\mathrm { C } ( 2 \mathrm { n } , \mathrm { n } ) \approx { 4 ^ { \mathrm { n } } } / { \sqrt { ( \pi \mathrm { n } ) } }$ . ∎

Two consequences are worth noting. Ambiguity grows linearly in item length at fixed density, roughly two bits per authentic token at $\eta = 1$ , so longer items are intrinsically better protected. And repeated tokens are a leak: a camouflage stream that reuses the same few markers reduces A<sub>c</sub> below the bound, which is a second reason, alongside condition C3, to draw camouflage tokens from a wide distribution.

Theorem 4 (Existence and Location of the Optimal Density). Let F be the feasible set of Equation (30). If F is non-empty then $\boldsymbol { \eta } ^ { * }$ exists. If the separation margin is above threshold, so that $\mathrm { S _ { c } }$ is constant on F by Theorem 2, then $\eta ^ { * } = \eta \alpha$ F: the optimum lies on the boundary and at least one of Equations (26) and (28) is active at $\boldsymbol { \eta } ^ { * }$

Proof. For fixed n the density $\eta = \eta / \eta$ takes finitely many values in [0, 1], so F is a finite set and a maximum of any real objective over a non-empty finite set is attained. In the continuous relaxation used for tuning, $\mathbf { A } _ { \mathrm { c } } ( \eta )$ is continuous and strictly increasing and Sim(Q, Ω(η)) is continuous and non-increasing, so F is closed and bounded and therefore compact, and an upper semicontinuous objective attains its maximum on it. Above threshold the objective $\mathrm { A _ { c } ( \eta ) } - \lambda \mathrm { S _ { c } ( \eta ) }$ reduces to $\mathrm { \ A _ { c } ( \eta ) \mathrm { ~ - ~ } } \lambda \mathrm { a } ,$ which is strictly increasing, so its maximizer over F is sup F, which lies in F by closedness. Since $\mathbf { A } _ { \mathrm { c } }$ increases without bound in m while Equations (26) and (28) are eventually violated, sup F is interior to [0,1] and is determined by whichever constraint fails first. ∎

The earlier version treated η as ranging continuously over a compact interval and concluded existence from continuity alone. The finite-value argument is both simpler and correct, and the boundary result is the practically useful part: there is no interior optimum to search for above threshold, so tuning reduces to finding the largest density that still satisfies the obfuscation and plausibility ceilings. An interior optimum reappears only below threshold, where λS (η) grows with η and genuinely trades against $\mathbf { A } _ { \mathrm { c } } ,$

Theorem 5 (Lamination Invertibility and Brute-Force Cost). Given (Π, Γ), ${ \mathrm { ~ Q ~ } } = \ { \mathcal { L } } ^ { - 1 } ( \Omega ;$ Π) exactly. Without them, an adversary of class A1 or $_ { \mathrm { A } 2 }$ confronts a candidate set of size $\mathrm { C } ( { \mathrm { n } } { + } { \mathrm { m } } , { \mathrm { m } } )$ , that is, 2 raised to the power $\operatorname { A } _ { \mathrm { c } } .$

Proof. Invertibility is the construction in the proof of Theorem 1. The cardinality is immediate from Theorem 3. ∎

The framing is deliberately modest and we want to be explicit about what it is not. This is an average-case entropy statement under a uniform prior over insertion sets. It is not a computational hardness result: there is no reduction to a problem believed to be hard, and none is claimed. Theorem 6 shows how far the bound falls against an adversary who does more than enumerate.

![](images/c6a7e0cab9468bf77760932233abf9ef888aba7b45aaae135e3ec6f70d6197f7.jpg)  
Fig. 6. Growth of the candidate-set size $C ( n { + } m ,$ m), in bits, as a function of authentic-stream length n at unit camouflage density $( m \ = \ n ) .$ . This is the quantity A\_c of Theorem 3.

Theorem 6 (Degradation Under a Semantic Filter). Let an adversary of class A4 or A5 hold a reference model p<sub>LM</sub> and retain only candidates whose sequence probability exceeds a threshold, and let q be the acceptance rate of that test under the uniform prior over insertion sets. Then the residual ambiguity is $\mathrm { A _ { c } ^ { \mathrm { ~ L M } } = \bar { A } _ { c } - l o g _ { 2 } ( l / q ) }$ . The plausibility constraint of Equation (28) is exactly a constraint on q.

Proof. The filter restricts the candidate set of Theorem 3 to its accepted subset, whose expected cardinality is $\mathbf { q } \cdot \mathbf { C } ( \mathbf { n } { + } \mathbf { m }$ , m). Conditioned on acceptance, and absent further information distinguishing accepted candidates, the posterior is uniform on that subset, whose entropy is log<sub>2</sub>(q $\mathbf { \partial } \cdot \mathbf { C ( n + m , \ n ) } ) \ = \ \mathbf { A _ { c } } \ -$ $\log _ { 2 } ( 1 / \mathsf { q } )$ . Camouflage tokens that are conspicuous under p<sub>LM</sub> drive q toward $1 \ / \ C ( \mathrm { n + m } ,$ , m) and hence $\mathrm { \dot { A } _ { c } ^ { \ L M } }$ toward zero; camouflage tokens indistinguishable under $\mathtt { p } _ { \mathtt { L M } }$ give $\mathsf { q } \approx 1$ and leave $\mathrm { A _ { c } }$ intact. Since Equation (28) bounds the per-position detectability of camouflage tokens under the same model, it bounds q from below. ∎

This is the result that connects the design constraints to the adversary that matters most in current practice. It also makes the trilemma of Section IX quantitative: Equation (26) pushes camouflage tokens away from the authentic distribution, which lowers q and therefore lowers $\mathbf { A } _ { \mathrm { c } } ^ { \mathrm { \ L M } }$ , while Equation (28) pushes them back toward it, which raises $\mathsf { q }$ but raises Sim(Q, Ω) with it. There is no setting of the inversion operator that maximizes both, and we do not have a principled way to choose the exchange rate between them.

Theorem 7 (Multi-Observation Leakage Under Frame-Granular Re-Lamination). Suppose Q is fixed within a session while camouflage tokens are redrawn independently at each of k rendered frames from a distribution P, and let $\gamma _ { \mathrm { { k } } } = \sum _ { \mathrm { { w } } } \mathrm { { P ( w ) ^ { k } } } .$ which equals 2 raised to the power $- ( \mathrm { k } { - } 1 ) \mathrm { H } _ { \mathrm { k } } ( \mathrm { P } )$ with H<sub>k</sub> the Rényi entropy of order k. An adversary who aligns the k captures and retains positions whose token is invariant across all of them retains every authentic position and, in expectation, m<sub>k</sub> = m·γ<sub>k</sub> camouflage positions. Residual ambiguity satisfies $\operatorname { A } _ { \mathrm { c } } ^ { \mathrm { ( k ) } } \leq \log _ { 2 } \mathbf { C } ( \mathtt { n } + \mathbf { m } _ { \mathbf { k } } , \mathbf { m } _ { \mathbf { k } } )$ , which decays geometrically in k.

Proof. An authentic position carries the same token at every frame and so survives the invariance test with probability one, giving n survivors. A camouflage position survives only if k independent draws from P coincide, which occurs with probability $\begin{array} { r } { \sum _ { \mathrm { w } } \mathrm { P } ( \mathrm { w } ) ^ { \mathrm { k } } = \gamma _ { \mathrm { k } } ; } \end{array}$ linearity of expectation gives $\mathbf { m } _ { \mathbf { k } } = \mathbf { m }$ γ expected camouflage survivors. The surviving set contains Q as a subsequence together with at most m<sub>k</sub> impostor positions, and Theorem 3 applied to that reduced instance bounds the remaining uncertainty. Since H<sub>k</sub> is non-increasing in k and bounded below by the min-entropy H<sub>∞</sub>(P), the decay rate lies between H<sub>∞</sub> and H<sub>2</sub> bits per additional frame. ∎

Figure 7 plots the decay for a 120-token item at unit density. At two bits of effective decay per frame, ambiguity falls from roughly 240 bits to under one bit within eight captures, and at four bits per frame within five. The direction of the effect is the point. Resampling the camouflage layer more aggressively raises the per-frame entropy $\mathrm { H } _ { \mathrm { k } } ,$ which accelerates the leak rather than slowing it. Dynamic lamination as described in the earlier version therefore weakens the scheme against any adversary who captures more than once, and the invariance test costs the adversary nothing beyond alignment.

![](images/57dd46694f931be1dbbbbd8b2ca5fb2160dd9ed7df655df6087a4c4a1873b11e.jpg)  
Fig. 7. Residual ambiguity after k independent frame captures under framegranular re-lamination, for a 120-token item at unit camouflage density, plotted from the bound of Theorem 7 at three effective decay rates. Curves are computed from the closed form and contain no measured data.

The remedy is Equation (33). Holding both Π and $\tilde { \mathrm { { Q } } }$ fixed for the lifetime of a session removes the invariance signal entirely, since nothing varies for the adversary to compare, and confines rotation to the session boundary where an attacker gains no repeated observations of the same item. We take this as the default configuration and treat frame-granular variation as unsafe unless the indistinguishability condition is verified.

Theorem 8 (Capture Complexity Under Temporal Multiplexing). Let each frame reveal each authentic token independently with probability ${ \mathfrak { p } } ,$ as in Equations (34) and (36). Then the expected number of captures required to observe every authentic token at least once is $\operatorname { E } [ \operatorname { k } ] = \sum _ { \operatorname { j } \geq 0 } \ \big [ \ 1 \ - \ ( 1 \ -$ $( 1 { - } \mathrm { \bar { p } } \mathrm { \bar { p } } ) ^ { \mathrm { j } } ) ^ { \mathrm { n } } ~ ] \approx \ln { \mathrm { ~ n ~ } } / \ \ln ( 1 / ( 1 { - } \mathrm { p } ) )$ . This bound applies to every adversary class, including A5.

Proof. The first frame at which token i becomes visible is geometric with parameter $\rho ,$ and these are independent across tokens. The number of frames until all n have appeared is the maximum of n independent geometric variables, whose expectation is the stated sum by the tail-sum identity, with the classical extreme-value approximation ln $\texttt { n / } \ln ( 1 / ( 1 { - } \rho ) )$ for large n. Any reconstruction of Q requires each of its tokens to have been observed at least once, so E[k] lower-bounds the expected captures needed for exact recovery. ∎

Figure 8 shows the growth. The honest reading is that temporal multiplexing raises the cost of a screenshot attack from one capture to a modest number, and that the growth in item length is logarithmic rather than polynomial, so it is a friction rather than a barrier. Its value comes from composition with the behavioural domain B(t) of the MARS state: if each capture carries an independent detection probability $\mathrm { p } _ { \mathrm { d } } ,$ the probability that a full reconstruction goes unnoticed falls as (1 $- { \mathsf { p } } _ { \mathrm { d } } ) ^ { \mathrm { E [ k ] } }$ , and it is that product, not the capture count alone, that constitutes the defence against A5.

![](images/1ba0cbf90dc10a4d9e13d3f1bd415449d2031af8d2efb9a72dd70c40ca741838.jpg)  
Fig. 8. Expected number of captures required for complete observation of the authentic stream under temporal multiplexing, from the exact expression of Theorem $\delta ,$ at three per-frame visible fractions ρ. Computed from the closed form; no measured data.

## XIII. ALGORITHM AND COMPLEXITY

The constructs of Sections V to VII compose into a single render-time procedure. Stating it makes the cost explicit and removes any impression that the framework requires expensive inference on the critical path.

Algorithm 1 Render a camouflaged item   
Input : Q=(w\_1..w\_n), key K, item id, eta,   
margin dE\*, ceilings th\_obf, d\_max   
Output: rendered surface Sigma   
1 m <- round(eta\*n)   
2 Qt <- Inversion(Q, m) # C1-C3   
3 Pi <- PRF(K, id, 0) # m-subset   
4 Gam <- PRF(K, id, 1) s.t. (17),(18)   
5 for i = 1..n+m: # laminate   
6 Om[i] <- Qt[j(i)] if i in Pi   
7 else Q[r(i)]   
8 $\mathsf { c h i } [ \mathsf { i } ] \ < - \ \mathsf { c \_ c } \ \mathsf { i } \bar { \mathsf { f } } ^ { - } \mathsf { i }$ in Pi else c\_q   
9 if Sim(Q,Om) > th\_obf   
10 or Detect(Om,Pi) > d\_max:   
11 lower eta or resample Qt; goto 2   
12 return Sigma $< - \{ ( \mathsf { O m } [ \dot { \bf { i } } ] , \mathsf { p } _ { - } \dot { \bf { i } } , \dot { \mathsf { c } } \mathsf { h } \ddot { \bf { i } } [ \dot { \bf { i } } ] , \mathsf { P s i } _ { - } \dot { \bf { i } } ) \}$

Lamination is a single pass, so lines 5 to 8 cost Θ(n+m) time and Θ(n+m) space. Key derivation is Θ(m). The inversion operator dominates: with a lexical instantiation over WordNet it is Θ(m) lookups, and with a generative instantiation it is one forward pass over the item. The verification at lines 9 and 10 costs one sentence-encoder pass for Sim and one languagemodel pass for Detect. Since $\mathrm { Q } , \tilde { \mathrm { Q } } , \Pi ,$ and Γ all depend only on the item and the session key, the whole procedure runs once per item per session and the result is cached, so the per-frame render path carries only the Θ(n+m) evaluation of Equations (35) and (36). The retry loop at line 11 terminates because $\mathbf { A } _ { \mathrm { c } }$ is monotone in η and $\eta = 0$ trivially satisfies both ceilings.

## XIV. PROPOSED EVALUATION PROTOCOL

This paper reports no experiments and we make no empirical claims. The theorems above are, however, stated so that each yields a falsifiable prediction, and we set out the protocol here so that a later study can be pre-registered against it rather than assembled after the fact.

Readability, testing Theorem 2. Candidates complete matched item sets at densities spanning $\eta \in [ 0 , 1 ]$ under an above-threshold and a below-threshold margin. Primary outcomes are item-level accuracy and time to first keystroke. The prediction is that above threshold both are flat in η within noise, and that below threshold time grows linearly in η. Participants should include a colour-vision-deficient group under the non-chromatic channel of Section VII.

Extraction resistance, testing Theorems 3, 5, and 6. For each adversary class in Table I, the extracted view is passed to a solver and scored for item-level accuracy. The prediction is a large drop for A1 and A2, no drop for A3 absent styling countermeasures, and an intermediate drop for A4 and A5 that tracks the acceptance rate q of Theorem 6. Measuring q directly, by scoring candidate reconstructions under a held-out model, is the sharpest single test of the theory.

Leakage, testing Theorem 7. Under frame-granular relamination, apply the invariance test at $\mathbf { k } = 1 , 2 , 4 , 8$ captures and record the fraction of authentic tokens correctly identified. The prediction is that identification approaches unity within a number of captures set by the per-frame decay rate, and that holding Π and Q̃ fixed across the session drives it to chance.

Reporting. Effect sizes with confidence intervals, the full density grid rather than a selected operating point, and the measured $\Delta \mathrm { E } _ { 0 0 }$ of the rendered colours, which is the parameter on which the readability results depend and which is easy to leave unreported.

## XV. DISCUSSION AND LIMITATIONS

The constructs developed here reframe Context Camouflaging from a single black-box transformation Φ into an explicit pipeline whose readability and ambiguity properties are separately tuneable and, above the discriminability threshold, decoupled. The practical implication is that an institution can raise computational ambiguity against class A1 and A2 extraction by raising η without a corresponding cost to legitimate candidates, provided the separation margin of Equations (17) and (18) is met at render time.

The scope of that claim is narrower than the earlier version implied, and the narrowing is the main contribution as much as the new theorems are. Colour separation defends the textextraction channel. It does not defend against a scripted reader of the document object model, which sees the styling, and it does not defend against a vision-language model given a screenshot, which sees both the colour and enough semantics to make Theorem 6 bite hard. Against those classes the defence has to come from the temporal domain, and Theorem 8 prices what that buys: a logarithmic increase in capture count, valuable mainly because each capture carries detection risk in the behavioural domain. A reader who takes away only one thing should take away that the guarantee is channel-specific.

Several limitations follow. First, the trilemma of Section IX has no principled resolution in this paper. Equations (26) and (28) pull in opposite directions, Theorem 6 shows the exchange rate matters, and we have no method for setting it beyond tuning against a held-out model, which invites overfitting to that model.

Second, the Context Inversion Operator is treated abstractly. Conditions C1 to C3 constrain it but do not construct it, and whether a lexical instantiation over WordNet [3] or a learned generative model can satisfy all three at a useful density is an empirical question this paper does not settle.

Third, Theorem 3 assumes a uniform prior over insertion sets. A biased sampler, a predictable position map, or heavy reuse of camouflage tokens each reduce $\mathrm { A _ { c } }$ below the closed form, and the reduction is silent: nothing in the rendered output reveals it. Auditing the sampler is therefore a deployment requirement rather than an implementation detail.

Fourth, accessibility. Equation (18) states the condition but a platform still has to meet it under arbitrary user stylesheets, high-contrast modes, and screen readers, the last of which will read the laminated field in full unless the camouflage positions are marked for exclusion, which in turn hands class A3 exactly the signal it needs. We have no clean answer to that conflict and regard it as the sharpest practical objection to the approach.

## XVI. CONCLUSION

This paper extended the Context Camouflaging Operator of the MARS assessment-resilience framework into the Multi-Layer Context Camouflaging Theory, a specified mathematical treatment of semantic superposition for malpractice-resilient online assessment. Six coupled constructs replace the single coarse transformation of the original formulation, and eight theorems characterize their joint behaviour under an explicit adversary model.

Three of the revisions change what the framework claims rather than only how it is stated. The preservation constraint became a fidelity identity together with obfuscation and plausibility ceilings, because a similarity floor asks the laminated field to keep conveying the item it is meant to hide. Computational ambiguity became a conditional entropy with the closed form log<sub>2</sub> C(n+m, m), which is monotone in density where the previous unigram measure was not. And framegranular dynamic lamination turned out to leak geometrically against a multi-capture adversary, so the safe default is sessiongranular rotation.

Equation (35) reconnects the theory to the MARS Rendering Tensor formalism, with the type error of the earlier scalar product removed, so Context Camouflaging is available as a specified layer within the larger system model. The open problems we would most like to see addressed are the exchange rate between obfuscation and plausibility in Theorem 6, a construction for the inversion operator satisfying conditions C1 to C3 at useful density, and the conflict between screen-reader accessibility and resistance to class A3.

Table II  
PRINCIPAL SYMBOLS USED IN MCCT
<table><tr><td>Symbol</td><td>Description</td><td>Unit / Domain</td></tr><tr><td>Q, V</td><td>Authentic semantic stream / vocabulary</td><td>Token seq. / set</td></tr><tr><td>õ</td><td>Context-camouflage stream</td><td>Token sequence</td></tr><tr><td>J</td><td>Context Inversion Operator</td><td>Mapping</td></tr><tr><td>Ω</td><td>Laminated (superposed) field</td><td>Token sequence</td></tr><tr><td>Σ(t)</td><td>Rendered surface (new)</td><td>Attributed sequence</td></tr><tr><td> $\mathrm { { X } _ { A } , \mathrm { { V } _ { A } } }$ </td><td>Extraction channel / adversary view (new)</td><td>Projection / seq.</td></tr><tr><td>α;A)</td><td>Contextual Lamination Operator</td><td>Mapping</td></tr><tr><td> $\Lambda = ( \pi , \kappa , \chi )$ </td><td>Lamination control law</td><td>Function</td></tr><tr><td>Ⅱ</td><td>Insertion set (camouflage positions)</td><td>Index set</td></tr></table>

<table><tr><td>Symbol</td><td>Description</td><td>Unit / Domain</td></tr><tr><td>χ(i), Γ</td><td>Separation function / session key</td><td>Mapping / key</td></tr><tr><td> $\mathrm { c _ { \mathrm { q } } , c _ { \mathrm { c } } }$ </td><td>Authentic / camouflage channel value</td><td>Colour value</td></tr><tr><td> $\Delta \mathrm { E } _ { 0 0 } , \Delta \mathrm { E } ^ { * }$ </td><td>CIEDE2000 difference / margin (new)</td><td>CIELAB units</td></tr><tr><td> $\mathrm { R } _ { \mathrm { h } } , \alpha _ { \mathrm { i } }$ </td><td>Human Readability Functional / indicator</td><td>0–1 / binary</td></tr><tr><td> $\mathrm { S } _ { \mathrm { c } } ( \eta )$ </td><td>Visual search cost (new)</td><td>Time</td></tr><tr><td> $\mathbf { A } _ { \mathrm { c } }$ </td><td>Computational Ambiguity, H(Q | VA) (revised)</td><td>Bits</td></tr><tr><td>Sim(  $) , \theta _ { \mathrm { o b f } }$ </td><td>Semantic similarity / obfuscation ceiling</td><td>0-1</td></tr><tr><td> $\delta _ { \mathrm { i } } , \delta _ { \mathrm { m a x } }$ </td><td>Detectability under pLM / ceiling (new)</td><td>Log-prob</td></tr><tr><td>q</td><td>Semantic-filter acceptance rate (new)</td><td>0-1</td></tr><tr><td>η, m, n</td><td>Camouflage density / camouflage and authentic counts</td><td>Ratio / counts</td></tr><tr><td>Ti(t)</td><td>Context Camouflage Tensor of token i</td><td>Tensor</td></tr><tr><td> $\Psi _ { \mathrm { i } } ( \mathrm { t } ) , \delta _ { \mathrm { i } } ( \mathrm { t } )$ </td><td>Temporal visibility / spatial drift (from [1]]</td><td>Binary / scalar</td></tr><tr><td>ρ(t)</td><td>Visible fraction per frame (new)</td><td>0-1</td></tr><tr><td>Ri(t)</td><td>Per-token rendering state (revised)</td><td>Tuple</td></tr></table>

AI Use Disclosure: Portions of this manuscript were prepared with the assistance of generative artificial-intelligence tools for language refinement, formatting, and figure drafting. All scientific concepts, mathematical formulations, algorithms, theoretical propositions, analyses, and conclusions were conceived, verified, and approved by the authors, who accept full responsibility for the content of this manuscript [14].

## REFERENCES

[1] L. R. Gupta, K. Kaur, S. Ram, and Prajithaa, "A Multi-Dimensional Spatio-Temporal Context Camouflaging Framework for Resilient Digital Assessments," MARS Technical Report, Lovely Professional University, 2026.

[2] S. Dawson, "Defending Assessment Security in a Digital World," Assessment & Evaluation in Higher Education, 2020.

[3] C. Fellbaum, Ed., WordNet: An Electronic Lexical Database. Cambridge, MA, USA: MIT Press, 1998.

[4] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pretraining of Deep Bidirectional Transformers for Language Understanding," in Proc. NAACL-HLT, 2019, pp. 4171–4186.

[5] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in Proc. EMNLP, 2019, pp. 3982– 3992.

[6] C. E. Shannon, "A Mathematical Theory of Communication," Bell Syst. Tech. J., vol. 27, pp. 379–423, 1948.

[7] N. Boucher, I. Shumailov, R. Anderson, and N. Papernot, "Bad Characters: Imperceptible NLP Attacks," in Proc. 43rd IEEE Symp. Security and Privacy (SP), 2022, pp. 1987–2004.

[8] N. Boucher and R. Anderson, "Trojan Source: Invisible Vulnerabilities," in Proc. 32nd USENIX Security Symposium, Anaheim, CA, USA, 2023, pp. 6507–6524.

[9] A. M. Treisman and G. Gelade, "A Feature-Integration Theory of Attention," Cognitive Psychology, vol. 12, no. 1, pp. 97–136, 1980.

[10] J. M. Wolfe, "Guided Search 2.0: A Revised Model of Visual Search," Psychonomic Bulletin & Review, vol. 1, no. 2, pp. 202–238, 1994.

[11] M. R. Luo, G. Cui, and B. Rigg, "The Development of the CIE 2000 Colour-Difference Formula: CIEDE2000," Color Research & Application, vol. 26, no. 5, pp. 340–350, 2001.

[12] M. Mahy, L. Van Eycken, and A. Oosterlinck, "Evaluation of Uniform Color Spaces Developed after the Adoption of CIELAB and CIELUV," Color Research & Application, vol. 19, no. 2, pp. 105–121, 1994.

[13] H. Brettel, F. Viénot, and J. D. Mollon, "Computerized Simulation of Color Appearance for Dichromats," J. Opt. Soc. Amer. A, vol. 14, no. 10, pp. 2647–2655, 1997.

[14] UNESCO, "Guidance for Generative AI in Education and Research," Paris, France, 2023.
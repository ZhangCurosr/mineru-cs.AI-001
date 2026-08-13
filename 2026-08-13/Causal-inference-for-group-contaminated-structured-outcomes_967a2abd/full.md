# Causal inference for group-contaminated structured outcomes: observable quotients, lossless reduction and exact randomization inference

Usef Faghihi<sup>†</sup> Amir Saki<sup>†</sup>

D´epartement de math´ematiques et d’informatique Universit´e du Qu´ebec \`a Trois-Rivi\`eres Trois-Rivi\`eres, Qu´ebec, Canada

<sup>†</sup>These authors contributed equally to this work. Usef Faghihi: usef.faghihi@uqtr.ca Amir Saki: amir.saki@uqtr.ca

August 13, 2026

## Abstract

Structured potential outcomes such as microscopy images may be recorded after an unknown, unit-specific transformation. If that transformation can depend on treatment, covariates or the intrinsic outcome, raw-coordinate analyses may mix biological efects with acquisition geometry. We study the unrestricted observation model X = Γ · Y(A) and characterize its observable information: a target is uniformly recoverable exactly when it is constant on group orbits, while a Borel maximal invariant retains every measurable invariant target. We then distinguish observability from statistical losslessness. A quotient-faithful reconstruction theorem shows that quotient reduction is suficient for the full transformed experiment exactly when the conditional law of the raw observation given treatment, covariates and the quotient has a parameter-free version. Conditional Haar contamination on a compact group yields Blackwell equivalence as a special case; it is not imposed in the main model. We also separate independent site-specific product actions from shared diagonal actions and show why componentwise canonicalization can discard relative cross-site information. Under explicit metric and kernel regularity, an approximate-contamination theorem bounds quotient-law Wasserstein error and the induced perturbation of population maximum mean discrepancy. For finite-support multichannel lattice images, we construct a maximal invariant under integer translations and quarter turns, combine its characteristic Gaussian kernel with a complete paired-swap test, and retain the original simulations and RxRx1 HUVEC study. Under the sharp null, the quotient test rejected in 0.052 of simulation replicates; at unit efect strength its power was 0.992. The primary RxRx1 contrast had an enumerated paired-swap p-value of 0.0078.

Keywords: causal inference; group action; maximal invariant; Blackwell suficiency; microscopy image; quotient space; randomization test; structured outcome.

## 1 Introduction

Many experimental outcomes are structured objects rather than scalars. An image, shape, network or function may be represented in a coordinate system that varies from unit to unit. In microscopy,

for example, translation or orientation can arise from acquisition and preprocessing rather than from the biological intervention. We study the observation model

$$
X = \Gamma \cdot Y ( A ) ,\tag{1.1}
$$

where A is treatment, $Y ( a )$ is the intrinsic structured potential outcome and Γ is an unobserved group element. Crucially, no independence between Γ and treatment, covariates or potential outcomes is imposed. Averaging over a posited nuisance distribution is therefore neither required nor generally available.

The first question is an information boundary: which targets of Y(a) are determined by the transformed observation? Orbit invariance gives an exact answer. The second question is causal: under what design or exchangeability conditions does the observed quotient distribution identify its interventional counterpart? A third, logically stronger question is statistical: when does discarding the within-orbit coordinate lose no information about the full family of observed-data laws? Maximal invariance answers the first question, but not the third. Conflating those statements would overstate what quotienting achieves.

This paper develops a theorem-to-algorithm-to-application account of these three layers. The principal additions to the original outcome-level framework are:

1. a sharp suficiency boundary based on a common conditional law and a quotient-faithful reconstruction kernel;

2. a conditional-Haar Blackwell-equivalence corollary, stated only as a special case of the unrestricted contamination framework;

3. a product-versus-diagonal action result that identifies when sitewise canonicalization is appropriate and when it erases relative configuration;

4. an approximate-contamination stability result under explicit orbit-metric and reproducingkernel assumptions.

The individual ingredients—maximal invariants, measurable factorization, statistical suficiency, Blackwell comparison, Haar measure, characteristic kernels and Fisher randomization—are classical [2, 5, 6, 25]. The contribution claimed here is their outcome-level causal synthesis under unrestricted latent group contamination, together with an explicit noncompact lattice construction and a finitesample inferential workflow. We do not claim novelty for those classical ingredients separately.

Existing causal methods for metric- and random-object outcomes generally take the structured outcome as directly observed [1, 11, 20]; registration-based functional methods address a related preprocessing problem [15]. Topological transforms can be injective on specified shape classes when suficiently rich directional information is available [4, 24]. Our setting is complementary: the observation equation itself contains a latent, unit-specific action, and the target is the full observable orbit rather than a chosen low-dimensional summary.

The application uses two-site, six-channel RxRx1 wells. Under separate site-specific acquisition motions the nuisance group is a product of lattice rigid-motion groups. Translation normalization and finite rotational minimization yield a measurable maximal invariant without placing a probability law on the noncompact translation group. A Gaussian kernel on canonical representatives is characteristic for quotient laws, and complete enumeration of the reported paired assignment mechanism yields finite-sample inference for the quotient Fisher sharp null.

Section 2 develops observability and quotient-law identification. Section 3 gives the statisticallosslessness boundary and Haar special case. Section 4 distinguishes product and diagonal actions.

Section 5 proves stability under approximate contamination. Sections 6 and 7 give the image construction and exact test. Sections 8 and 9 report the preserved empirical results. Proof details that would interrupt the main argument appear in Appendix A.

## 2 Observable quotient targets and causal identification

## 2.1 Measurable group contamination

Let Y, C, S and G be standard Borel spaces. Assume that G is a measurable group and acts on Y through a jointly Borel map $( g , y ) \mapsto g \cdot y$ . The orbit of y is $[ y ] _ { G } = \{ g \cdot y : g \in G \}$ . A Borel map $\tau : \mathcal { V } \to \mathcal { S }$ is invariant if $\tau ( g \cdot y ) = \tau ( y )$ for all $( g , y )$

Let C denote the observed pretreatment covariates, taking values in the standard Borel space C, and let A denote the realized treatment assignment, taking values in a finite set A. For each $a \in { \mathcal { A } }$ , let $Y ( a )$ be the intrinsic potential outcome under treatment a in the potential-outcomes framework [19]. We observe $O = ( C , A , X )$ ), where X satisfies (1.1). The conditional distribution of Γ given C, A and the potential outcomes is unrestricted.

Theorem 2.1 (Sharp observability). For a Borel target $\tau : \mathcal { V } \to \mathcal { S }$ , the following are equivalent:

1. there is a Borel decoder $D : \mathcal { y }  \mathcal { S }$ such that $D ( g \cdot y ) = \tau ( y )$ for every $g \in G$ and $y \in \mathcal { V } _ { : }$

2. τ is G-invariant.

Proof. $\mathrm { I f } ~ \tau$ is invariant, take $D = \tau$ . Conversely, the identity element gives $D ( z ) = \tau ( z )$ for every z. Hence $\tau ( g \cdot y ) = D ( g \cdot y ) = \tau ( y )$ □

Corollary 2.2 (Nonidentifiability outside the quotient). ${ \cal I } f \tau$ is not invariant, two data-generating processes satisfying (1.1) can induce the same observed law while giving diferent laws $o f \tau \{ Y ( a ) \}$ for any fixed a.

Proof. Choose y and g with $\tau ( g \cdot y ) \neq \tau ( y )$ . In one process set $Y ( a ) = y$ and $\Gamma = g ;$ in the other set $Y ( a ) = g \cdot y$ and $\Gamma = e$ . Both give $X = g \cdot y$ , but their target values difer. □

## 2.2 Maximal invariant and factorization

Definition 2.3. A Borel map $M : \mathcal { V }  \mathcal { S }$ is a maximal invariant when

$$
M ( y ) = M ( z ) \quad \Longleftrightarrow \quad z = g \cdot y { \mathrm { ~ f o r ~ s o m e ~ } } g \in G .
$$

Theorem 2.4 (Borel factorization). Let $\mathcal { \ V } , \mathcal { S } , \mathcal { T }$ be standard Borel spaces and $M : \mathcal { V }  \mathcal { S }$ a Borel maximal invariant. Every Borel invariant $h : \mathcal { V }  \mathcal { T }$ admits a Borel factorization $h = \bar { h } \circ M$ . The restriction of h<sup>¯</sup> to $M ( \mathfrak { V } )$ is unique.

The proof, given in Appendix A, uses saturation of $h ^ { - 1 } ( B )$ , Lusin separation of two disjoint analytic images and the measurable factorization lemma; these are standard tools of descriptive set theory [10]. Thus M retains every measurable invariant target; it does not follow that $M ( X )$ is suficient for a statistical parameter.

## 2.3 Identification of the interventional quotient law

For each $a \in { \mathcal { A } }$ , define the quotient potential outcome

$$
Q ( a ) = M \{ Y ( a ) \} ,
$$

and define the observed quotient outcome by $Q = M ( X )$ . Invariance then gives

$$
Q = M ( X ) = M \{ \Gamma \cdot Y ( A ) \} = M \{ Y ( A ) \} = Q ( A )\tag{2.1}
$$

without imposing any restriction on the conditional distribution of Γ.

Theorem 2.5 (Identification of the full quotient law). Suppose, for every $a \in { \mathcal { A } } ,$ , that quotient consistency (2.1) holds, $Q ( a ) \perp \perp A \mid C$ , and $\mathbb { P } ( A = a \mid C ) > 0$ almost surely. Then, for every Borel $B \subseteq S$

$$
\mathbb { P } \{ Q ( a ) \in B \} = \int \mathbb { P } ( Q \in B \mid A = a , C = c ) d P _ { C } ( c ) .\tag{2.2}
$$

Consequently, the entire interventional law of the maximal invariant is identified.

Proof. Standard Borel spaces admit regular conditional probabilities [9]. Apply iterated expectation, conditional exchangeability, positivity and consistency in that order. Equality for every Borel B determines the probability measure on S. □

By Theorem 2.4, the interventional law of any Borel invariant target is the pushforward of (2.2) through h<sup>¯</sup>. This is the ordinary adjustment formula [17, 18] applied to a quotient-valued outcome, and it remains distinct from the suficiency question considered next.

## 3 When quotient reduction is statistically lossless

Let Θ index a family of observed-data laws. We use the standard comparison-of-experiments framework [12, 23]. Put

$$
{ \cal O } _ { \mathrm { f u l l } } = ( C , A , X ) , \qquad { \cal O } _ { \mathrm { q u o t } } = T ( { \cal O } _ { \mathrm { f u l l } } ) = ( C , A , M ( X ) ) ,
$$

and write $\mathcal { E } _ { \mathrm { f u l l } } = \{ P _ { \theta } ^ { \mathrm { f u l l } } : \theta \in \Theta \}$ and $\mathcal { E } _ { \mathrm { q u o t } } = \{ P _ { \theta } ^ { \mathrm { q u o t } } : \theta \in \Theta \}$ . Since T is deterministic, the full experiment Blackwell-dominates the quotient experiment. The reverse comparison requires an additional condition.

Definition 3.1 (Quotient-faithful kernel). A Markov kernel K from the quotient sample space to the full sample space is quotient-faithful for ${ \mathcal E _ { \mathrm { q u o t } } }$ if

$$
K { \big ( } s , T ^ { - 1 } ( \{ s \} ) { \big ) } = 1 \qquad P _ { \theta } ^ { \mathrm { q u o t } } { \mathrm { - a l m o s t ~ s u r e l y ~ f o r ~ e v e r y ~ } } \theta \in \Theta .
$$

Thus, up to quotient-null sets under each member of the experiment, the kernel reconstructs a raw representative without changing the supplied covariates, treatment, or quotient value.

Theorem 3.2 (Boundary of lossless quotient reduction). Assume the full and quotient sample spaces are standard Borel. The following are equivalent:

1. there is a Markov kernel K, independent of θ, such that, for every $\theta \in \Theta$ , K is a regular conditional distribution of $O _ { \mathrm { f u l l } }$ given $O _ { \mathrm { { q u o t } } }$ under $P _ { \theta } ^ { \mathrm { f u l l } } ;$ equivalently,

$$
K ( s , \cdot ) = \mathrm { L a w } _ { \theta } ( { \cal O } _ { \mathrm { f u l l } } \ | \ { \cal O } _ { \mathrm { q u o t } } = s ) \quad P _ { \theta } ^ { \mathrm { q u o t } } { - a l m o s t \ s u r e l y } .
$$

2. there is a parameter-free quotient-faithful kernel K such that

$$
P _ { \theta } ^ { \mathrm { q u o t } } K = P _ { \theta } ^ { \mathrm { f u l l } } \qquad f o r \ e v e r y \ \theta \in \Theta .\tag{3.1}
$$

When these conditions hold, $O _ { \mathrm { q u o t } }$ is suficient for ${ \mathcal E } _ { \mathrm { f u l l } }$ and $\mathcal { E } _ { \mathrm { f u l l } } \equiv _ { B } \mathcal { E } _ { \mathrm { q u o t } }$ . Hence every bounded decision problem has the same attainable risk set under the two experiments. If the family is dominated, the conditions are also equivalent to Fisher–Neyman factorization through $O _ { \mathrm { { q u o t } } }$

Proof. Write $Q _ { \theta } : = P _ { \theta } ^ { \mathrm { q u o t } }$ . Suppose first that (1) holds. By the disintegration identity for the common regular conditional distribution, for every measurable set B in the full sample space,

$$
P _ { \theta } ^ { \mathrm { f u l l } } ( B ) = \int K ( s , B ) Q _ { \theta } ( d s ) = ( Q _ { \theta } K ) ( B ) .
$$

Hence $Q _ { \theta } K = P _ { \theta } ^ { \mathrm { f u l l } }$ for every θ. Moreover, since $O _ { \mathrm { q u o t } } = T ( O _ { \mathrm { f u l l } } )$ deterministically, a regular conditional law of $O _ { \mathrm { f u l l } }$ given $O _ { \mathrm { q u o t } } = s$ is supported on $T ^ { - 1 } ( \{ s \} )$ , Q<sub>θ</sub>-almost surely. Thus K is quotient-faithful, proving (2).

Conversely, suppose that (2) holds. Let B be measurable in the full sample space and let D be measurable in the quotient sample space. Quotient faithfulness implies that, Q<sub>θ</sub>-almost surely,

$$
K \big ( s , B \cap T ^ { - 1 } ( D ) \big ) = \mathbf { 1 } _ { D } ( s ) K ( s , B ) .
$$

Indeed, conditional on input $s ,$ the kernel is supported on $T ^ { - 1 } ( \{ s \} )$ ; this fibre is contained in $T ^ { - 1 } ( D )$ when $s \in D$ and is disjoint from $T ^ { - 1 } ( D )$ when $s \notin D$ . Using $Q _ { \theta } K = P _ { \theta } ^ { \mathrm { f u l l } }$ with the measurable set $B \cap T ^ { - 1 } ( D )$ therefore gives

$$
\begin{array} { l } { { P _ { \theta } ^ { \mathrm { f u l l } } \bigl ( B \cap T ^ { - 1 } ( D ) \bigr ) = \displaystyle \int K \bigl ( s , B \cap T ^ { - 1 } ( D ) \bigr ) \ Q _ { \theta } ( d s ) } } \\ { { \qquad = \displaystyle \int _ { D } K ( s , B ) Q _ { \theta } ( d s ) . } } \end{array}
$$

Since $T ( \cal { O } _ { \mathrm { f u l l } } ) = \cal { O } _ { \mathrm { q u o t } }$ , this is precisely

$$
P _ { \theta } ^ { \mathrm { f u l l } } ( O _ { \mathrm { f u l l } } \in B , \ O _ { \mathrm { q u o t } } \in D ) = \int _ { D } K ( s , B ) P _ { \theta } ^ { \mathrm { q u o t } } ( d s ) ,
$$

the defining identity for K to be a regular conditional distribution of $O _ { \mathrm { f u l l } }$ given $O _ { \mathrm { q u o t } }$ . The kernel is common to all θ, so (1) follows.

The deterministic map T gives $\mathcal { E } _ { \mathrm { f u l l } } \ \succeq _ { B } \ \mathcal { E } _ { \mathrm { q u o t } }$ , while K gives $\mathcal { E } _ { \mathrm { q u o t } } \succeq _ { B } \mathcal { E } _ { \mathrm { f u l l } }$ . Hence the experiments are Blackwell equivalent. The decision-theoretic assertion follows from Blackwell equivalence, and the dominated factorization equivalence is the standard Fisher–Neyman suficiency theorem. □

The quotient-faithfulness clause is essential for the stated equivalence. An arbitrary reverse kernel may reproduce each full marginal law while scrambling quotient values; such a kernel need not be a conditional distribution given the supplied quotient.

Example 3.3 (Maximal invariance is not suficiency). Let $G = \{ - 1 , 1 \}$ act on $\{ - 1 , 1 \}$ by multiplication. The maximal invariant is constant. Let $Y = 1$ and let $\mathbb { P } _ { \theta } ( \Gamma = 1 ) = \theta$ , so $X = \Gamma$ and $\mathbb { P } _ { \theta } ( X = 1 ) = \theta$ . The raw observation identifies θ, whereas the quotient law is identical for every θ. Thus the conditional law of X given the quotient depends on θ and the quotient is not suficient. Under unrestricted contamination, within-orbit coordinates may carry information about the nuisance mechanism even though they carry no uniformly observable information about the intrinsic representative.

## 3.1 Conditional-Haar contamination as a special case

Let G be compact and second countable, and let $\mu _ { G }$ denote its normalized Haar probability measure. Assume that G acts jointly Borel measurably on Y and that $M : \mathcal { V }  \mathcal { S }$ is a Borel maximal invariant. Put

$$
S _ { 0 } = M ( \mathcal { V } ) ,
$$

and assume that $ { \boldsymbol { S } } _ { 0 }$ is a Borel subset of S. Let $s : { \mathcal { S } } _ { 0 } \to \mathcal { V }$ be a Borel section satisfying

$$
M \{ s ( m ) \} = m \qquad \mathrm { f o r ~ e v e r y ~ } m \in S _ { 0 } .
$$

For $m \in S _ { 0 }$ and Borel $B \subseteq \mathcal { V }$ , define the orbit kernel

$$
K _ { m } ( B ) = \int _ { G } { \bf 1 } _ { B } \{ g \cdot s ( m ) \} d \mu _ { G } ( g ) .\tag{3.2}
$$

The kernel may be extended arbitrarily and measurably from $ { \boldsymbol { S } } _ { 0 }$ to $s ;$ its values outside $ { \boldsymbol { S } } _ { 0 }$ are irrelevant because $M ( X ) \in { \cal S } _ { 0 }$ almost surely. Right invariance of Haar measure implies that $K _ { m }$ does not depend on the particular representative selected from the orbit indexed by m.

Corollary 3.4 (Conditional-Haar Blackwell equivalence). Suppose that, for every $\theta \in \Theta$ ，

$$
\operatorname { L a w } _ { \theta } \{ \Gamma \mid C , A , Y ( A ) \} = \mu _ { G }\tag{3.3}
$$

almost surely. Then, for $P _ { \theta } ^ { \mathrm { q u o t } }$ -almost every $( C , A , m )$

$$
\operatorname { L a w } _ { \theta } \{ X \mid C , A , M ( X ) = m \} = K _ { m }
$$

has a version that is independent of θ. Consequently,

$$
\mathcal { E } _ { \mathrm { f u l l } } \equiv _ { B } \mathcal { E } _ { \mathrm { q u o t } } .
$$

Proof. Fix a Borel set $B \subseteq \mathcal { V }$ . The joint measurability of the action and the measurability of the section ensure that $m \mapsto K _ { m } ( B )$ is Borel. Condition first on $( C , A , Y ( A ) )$ ). Under (3.3),

$$
\mathbb { P } _ { \theta } \{ X \in B \mid C , A , Y ( A ) \} = \int _ { G } \mathbf { 1 } _ { B } ( g \cdot Y ( A ) ) \ d \mu _ { G } ( g ) .
$$

Let $y \in \mathcal { V }$ and put $m = M ( y )$ . Since $s ( m )$ and y lie in the same orbit, maximality of M yields an $h \in G$ such that

$$
y = h \cdot s ( m ) .
$$

Consequently,

$$
\int _ { G } { \bf 1 } _ { B } ( g \cdot y ) d \mu _ { G } ( g ) = \int _ { G } { \bf 1 } _ { B } ( ( g h ) \cdot s ( m ) ) d \mu _ { G } ( g ) .
$$

By right invariance of normalized Haar probability, $g h$ has law $\mu _ { G }$ whenever g has law $\mu _ { G }$ . Hence

$$
\int _ { G } { \bf 1 } _ { B } ( g \cdot y ) d \mu _ { G } ( g ) = \int _ { G } { \bf 1 } _ { B } ( g \cdot s ( m ) ) d \mu _ { G } ( g ) = K _ { m } ( B ) .
$$

It follows that

$$
\mathbb { P } _ { \theta } \{ X \in B \mid C , A , Y ( A ) \} = K _ { M \{ Y ( A ) \} } ( B ) \qquad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

Taking conditional expectations given $\sigma \{ C , A , M ( Y ( A ) ) \}$ therefore gives

$$
\begin{array} { r l } & { \mathbb { P } _ { \theta } \{ X \in B \mid C , A , M ( Y ( A ) ) \} = \mathbb { E } _ { \theta } \big [ K _ { M \{ Y ( A ) \} } ( B ) \mid C , A , M ( Y ( A ) ) \big ] } \\ & { \qquad = K _ { M \{ Y ( A ) \} } ( B ) \qquad \mathrm { a l m o s t ~ s u r e l y } . } \end{array}
$$

Since M is invariant and $X = \Gamma \cdot Y ( A )$

$$
M ( X ) = M \{ \Gamma \cdot Y ( A ) \} = M \{ Y ( A ) \} \qquad { \mathrm { a l m o s t ~ s u r e l y } } .
$$

Thus

$$
\operatorname { L a w } _ { \theta } \{ X \mid C , A , M ( X ) = m \} = K _ { m }
$$

for $P _ { \theta } ^ { \mathrm { q u o t } }$ -almost every $( C , A , m )$ , and this version is independent of $\theta .$

For completeness, define a kernel from the quotient sample space to the full sample space by

$$
\widetilde K \big ( ( c , a , m ) , B \big ) : = \int _ { \mathcal { V } } \mathbf { 1 } _ { B } ( c , a , x ) K _ { m } ( d x ) ,
$$

for every measurable B in the full sample space. This kernel is parameter-free. Moreover, $K _ { m }$ is supported on $M ^ { - 1 } ( \{ m \} )$ , since

$$
M \{ g \cdot s ( m ) \} = M \{ s ( m ) \} = m \qquad { \mathrm { f o r ~ e v e r y ~ } } g \in G .
$$

Hence $\widetilde { K }$ is quotient-faithful. It is a common regular conditional distribution of $O _ { \mathrm { f u l l } }$ given $O _ { \mathrm { { q u o t } } }$ so Theorem 3.2 yields

$$
\mathcal { E } _ { \mathrm { f u l l } } \equiv _ { B } \mathcal { E } _ { \mathrm { q u o t } } .
$$

The logical hierarchy is therefore:

unrestricted Γ : $M \{ Y ( a ) \}$ is the maximal uniformly observable target;   
parameter-free within-orbit law: M(X) is statistically suficient;   
conditional Haar Γ : a concrete compact-group condition implying suficiency.

The main results and empirical construction use the first line. The Haar assumption is not needed for observability, quotient identification or exact randomization inference.

## 4 Independent product actions and shared diagonal actions

Structured units often contain labeled components, such as two microscopy sites. The appropriate group depends on whether acquisition transformations are component-specific or shared.

Proposition 4.1 (Product versus diagonal actions). Let $M : \mathcal { V }  \mathcal { S }$ be a maximal invariant for a G-action and let $J \ge 2$ . Define

$$
M ^ { \times J } ( y _ { 1 } , \dots , y _ { J } ) = \big ( M ( y _ { 1 } ) , \dots , M ( y _ { J } ) \big ) .
$$

1. Under the product action of $G ^ { J }$

$$
( g _ { 1 } , \ldots , g _ { J } ) \cdot ( y _ { 1 } , \ldots , y _ { J } ) = ( g _ { 1 } \cdot y _ { 1 } , \ldots , g _ { J } \cdot y _ { J } ) ,
$$

$M ^ { \times J }$ is a maximal invariant.

2. Under the diagonal action of $G ,$

$$
g \cdot ( y _ { 1 } , \ldots , y _ { J } ) = ( g \cdot y _ { 1 } , \ldots , g \cdot y _ { J } ) ,
$$

$M ^ { \times J }$ is invariant. Moreover, on any diagonally G-invariant model subset $\mathcal { D } \subseteq \mathcal { V } ^ { J }$ , it is maximal for the restricted diagonal action if and only if, whenever $y , z \in { \mathcal { D } }$ and $z _ { j } = g _ { j } \cdot y _ { j }$ for component-specific g<sub>j</sub>, there is a single $g \in G$ satisfying $z _ { j } = g \cdot y _ { j }$ for every j.

Thus componentwise canonicalization can discard relative cross-component information under a shared transformation.

Proof. For (1), suppose first that

$$
M ^ { \times J } ( y _ { 1 } , \dots , y _ { J } ) = M ^ { \times J } ( z _ { 1 } , \dots , z _ { J } ) .
$$

Then $M ( y _ { j } ) = M ( z _ { j } )$ for every j. By maximality of M, for each j there exists $g _ { j } \in G$ such that $z _ { j } = g _ { j } \cdot y _ { j }$ . Therefore

$$
( z _ { 1 } , \ldots , z _ { J } ) = ( g _ { 1 } , \ldots , g _ { J } ) \cdot ( y _ { 1 } , \ldots , y _ { J } ) ,
$$

so the two points belong to the same $G ^ { J } { \mathrm { - o r b i t } }$ . The converse follows immediately from the invariance of M in each component. Hence $M ^ { \times J }$ is maximal for the product action.

For (2), invariance under the diagonal action follows because

$$
M ( g \cdot y _ { j } ) = M ( y _ { j } ) \qquad { \mathrm { f o r ~ e v e r y ~ } } j .
$$

Now let $y , z \in { \mathcal { D } }$ . Equality $M ^ { \times J } ( y ) = M ^ { \times J } ( z )$ is equivalent, by maximality of M, to the existence of possibly diferent elements $g _ { 1 } , \dotsc , g _ { J } \in G$ satisfying

$$
z _ { j } = g _ { j } \cdot y _ { j } , \qquad j = 1 , \dotsc , J .
$$

The points $y$ and z belong to the same diagonal orbit precisely when these component-specific transformations can be replaced by a single $g \in G$ satisfying $z _ { j } = g \cdot y _ { j }$ for every $j .$ . This is exactly the stated condition. □

Example 4.2 (Relative displacement). Let $G = ( \mathbb { R } , + )$ act on R by translation. A one-component maximal invariant is constant because the action is transitive. For a pair $( y _ { 1 } , y _ { 2 } )$ under the diagonal action, the diference $y _ { 2 } - y _ { 1 }$ is invariant and separates diagonal orbits. Componentwise quotienting is constant and therefore destroys this relative displacement. The image analogue is relative position or orientation between sites subjected to one common acquisition motion. In contrast, when the two sites undergo genuinely separate motions, relative pose is not uniformly observable and the product quotient is appropriate.

For the RxRx1 stress test below, the imposed transformation generator assigns separate sitespecific motions, so the product action is the declared model. Proposition 4.1(1) prevents that application-specific choice from being misread as a general prescription.

## 5 Stability under approximate contamination

Exact invariance is action-specific. To obtain a correct perturbation result, additional metric regularity is needed. Let $( \mathcal { V } , d )$ be a metric space on which G acts by isometries and define the orbit pseudometric

$$
d _ { G } ( y , z ) = \operatorname* { i n f } _ { g \in G } d ( y , g \cdot z ) .\tag{5.1}
$$

Let $( S , d _ { \cal S } )$ contain a quotient representation M satisfying

$$
d _ { S } \{ M ( y ) , M ( z ) \} \leq L _ { M } d _ { G } ( y , z ) .\tag{5.2}
$$

Let $\mathcal { P } _ { 1 } ( S )$ denote the Borel probability measures on S having a finite first moment with respect to $d _ { S }$ . For a measurable positive-semidefinite kernel k with reproducing kernel Hilbert space $\mathcal { H } _ { k }$ and Borel feature map $\Phi : { \mathcal { S } }  { \mathcal { H } } _ { k }$ , define

$$
\mu _ { k } ( P ) = \int _ { \cal S } \Phi ( s ) P ( d s )
$$

whenever this Bochner integral exists, and put

$$
\mathrm { M M D } _ { k } ( P , Q ) = \| \mu _ { k } ( P ) - \mu _ { k } ( Q ) \| _ { \mathcal { H } _ { k } } .
$$

Theorem 5.1 (Quotient-law and MMD stability). For arms $a \in \{ 0 , 1 \}$ , let $( X _ { a } , Y _ { a } )$ be a coupling of the approximately contaminated and intrinsic outcomes, and assume

$$
\mathbb { E } d _ { G } ( X _ { a } , Y _ { a } ) \leq \varepsilon _ { a } .
$$

Write

$$
{ \widetilde { P } } _ { a } = \operatorname { L a w } \{ M ( X _ { a } ) \} , \qquad P _ { a } = \operatorname { L a w } \{ M ( Y _ { a } ) \} ,
$$

and assume that $\widetilde { P } _ { a } , P _ { a } \in { \mathcal { P } } _ { 1 } ( S )$ . Then

$$
W _ { 1 , d _ { S } } ( \widetilde { P } _ { a } , P _ { a } ) \leq L _ { M } \varepsilon _ { a } .\tag{5.3}
$$

If the mean embeddings of ${ \widetilde { P } } _ { a }$ and $P _ { a }$ exist and the feature map satisfies

$$
\| \Phi ( s ) - \Phi ( t ) \| _ { \mathcal { H } _ { k } } \leq L _ { k } d _ { S } ( s , t ) ,\tag{5.4}
$$

then

$$
\begin{array} { r } { \left| \mathrm { M M D } _ { k } ( \widetilde { P } _ { 1 } , \widetilde { P } _ { 0 } ) - \mathrm { M M D } _ { k } ( P _ { 1 } , P _ { 0 } ) \right| \le L _ { k } L _ { M } ( \varepsilon _ { 1 } + \varepsilon _ { 0 } ) . } \end{array}\tag{5.5}
$$

If the perturbation bounds hold almost surely, the same conclusions follow with those uniform bounds.

Proof. The supplied coupling and (5.2) give

$$
W _ { 1 , d _ { S } } ( \widetilde { P } _ { a } , P _ { a } ) \leq \mathbb { E } d _ { S } \{ M ( X _ { a } ) , M ( Y _ { a } ) \} \leq L _ { M } \varepsilon _ { a } .
$$

Let $\mu ( R ) = \mathbb { E } _ { S \sim R } \Phi ( S )$ . By the same coupling and (5.4),

$$
\| \mu ( \widetilde { P } _ { a } ) - \mu ( P _ { a } ) \| _ { \mathcal { H } _ { k } } \leq L _ { k } L _ { M } \varepsilon _ { a } .
$$

Since $\mathrm { M M D } _ { k } ( R , S ) = \| \mu ( R ) - \mu ( S ) \| _ { \mathcal { H } _ { k } }$ , the reverse triangle inequality followed by the last two bounds proves (5.5). □

For a Gaussian kernel $k ( s , t ) = \exp \{ - \| s - t \| ^ { 2 } / ( 2 \sigma ^ { 2 } ) \}$ on a Hilbert quotient embedding,

$$
\| \Phi ( s ) - \Phi ( t ) \| ^ { 2 } = 2 \{ 1 - k ( s , t ) \} \le \| s - t \| ^ { 2 } / \sigma ^ { 2 } ,
$$

so $L _ { k } = 1 / \sigma$ . The theorem is intentionally conditional: the lexicographic lattice canonicalizer in Section 6 need not be globally Lipschitz near support changes or canonicalization ties. The approximate-action experiment in Table 5 therefore remains an empirical sensitivity analysis; it is not retroactively certified by Theorem 5.1 unless (5.2) is verified on the relevant image class.

## 6 A maximal invariant for lattice microscopy images

## 6.1 Lattice rigid motions

Fix $L < \infty$ . Let $\mathcal { X } _ { q , L }$ contain q-channel, 8-bit functions on $ { \mathbb { Z } } ^ { 2 }$ with finite support whose nonempty bounding box has at most L rows and columns, together with the zero image. Let $C _ { 4 } = \{ 0 , 1 , 2 , 3 \}$ with addition understood modulo four, and let $R _ { r }$ denote the counterclockwise rotation of $ { \mathbb { Z } } ^ { 2 }$ through $r \pi / 2$ . The action of $C _ { 4 }$ on $\mathbb { Z } ^ { 2 }$ is $u \mapsto R _ { r } u$ . Thus the semidirect product

$$
G _ { \mathrm { r i g } } = \mathbb { Z } ^ { 2 } \rtimes C _ { 4 }
$$

has multiplication and inverse

$$
( t , r ) ( u , s ) = \bigl ( t + R _ { r } u , r + s \bigr ) , \qquad ( t , r ) ^ { - 1 } = \bigl ( - R _ { r } ^ { - 1 } t , - r \bigr ) .
$$

Its action on an image x is

$$
\{ ( t , r ) \cdot x \} ( v ) = x \{ R _ { r } ^ { - 1 } ( v - t ) \} , \qquad v \in \mathbb { Z } ^ { 2 } .\tag{6.1}
$$

Here $t , u , v \in \mathbb { Z } ^ { 2 }$ are lattice coordinates or translations, whereas $r , s \in C _ { 4 }$ index quarter turns.

For nonzero $x ,$ let $b ( x )$ be the lower corner of its support bounding box and set $N ( x ) =$ $( - b ( x ) , 0 ) \cdot x .$ . For each $r \in C _ { 4 }$ , form $x _ { r } = N \{ ( 0 , r ) \cdot x \}$ . The key $\kappa ( x _ { r } )$ is the finite vector containing the bounding-box height and width, followed by all channel bytes in a fixed site–channel–row– column order. These finite vectors are compared using the lexicographic order. Define

$$
\begin{array} { r } { c ( x ) = x _ { r ^ { \star } } , \qquad r ^ { \star } = \operatorname* { m i n } { \mathrm { a r g m i n } _ { r \in C _ { 4 } } \kappa ( x _ { r } ) } , } \end{array}\tag{6.2}
$$

and set $c ( 0 ) = 0$

Theorem 6.1 (Lattice rigid-motion canonicalization). The map c is Borel and is a maximal invariant for $\mathbb { Z } ^ { 2 } \rtimes C _ { 4 }$

Proof. Because $\mathcal { X } _ { q , L }$ is a countable union of finite-dimensional finite-valued image spaces, the support map, bounding-box map, translation normalization, finite serialization and lexicographic comparison operations are Borel. Since the minimization in (6.2) is over the finite set $C _ { 4 }$ , the selected index $r ^ { \star }$ and hence c are Borel.

We next prove invariance. Translation normalization removes every integer translation:

$$
N \{ ( t , 0 ) \cdot z \} = N ( z ) \qquad { \mathrm { f o r ~ e v e r y ~ } } t \in \mathbb { Z } ^ { 2 } .
$$

Let $y = ( t , s ) \cdot x$ . For any $r \in C _ { 4 }$ , the semidirect-product law gives

$$
( 0 , r ) ( t , s ) = ( R _ { r } t , r + s ) .
$$

Therefore

$$
\begin{array} { r l } { N \{ ( 0 , r ) \cdot y \} } & { = N \{ ( 0 , r ) ( t , s ) \cdot x \} } \\ & { = N \{ ( R _ { r } t , r + s ) \cdot x \} } \\ & { = N \{ ( 0 , r + s ) \cdot x \} . } \end{array}
$$

Thus the four normalized rotational candidates of y are exactly the four normalized rotationa candidates of $x ,$ with their indices permuted by $r \mapsto r + s$ . Their lexicographically smallest elements are consequently equal, and hence

$$
c ( y ) = c ( x ) .
$$

It remains to prove maximality. Suppose that $c ( x ) = c ( y ) = z$ . By the construction of $^ { c , }$ there exist translations $u , v \in \mathbb { Z } ^ { 2 }$ and rotations $r , s \in C _ { 4 }$ such that

$$
z = ( u , r ) \cdot x \qquad { \mathrm { a n d } } \qquad z = ( v , s ) \cdot y .
$$

Applying $( v , s ) ^ { - 1 }$ to the second equality and substituting the first gives

$$
y = ( v , s ) ^ { - 1 } \cdot z = ( v , s ) ^ { - 1 } ( u , r ) \cdot x .
$$

Because $( v , s ) ^ { - 1 } ( u , r ) \in G _ { \mathrm { r i g } }$ , the images x and y belong to the same $G _ { \mathrm { r i g ^ { - } } } \mathrm { o r b i t }$ . Together with invariance, this shows

$$
c ( x ) = c ( y ) \quad \Longleftrightarrow \quad y = g \cdot x { \mathrm { ~ f o r ~ s o m e ~ } } g \in G _ { \mathrm { r i g } } ,
$$

so $c$ is a maximal invariant.

An RxRx1 well consists of two labeled sites. Let

$$
\mathcal { W } = \mathcal { X } _ { q , L } ^ { 2 }
$$

be the raw well space, with a typical well written as $w = ( x _ { 1 } , x _ { 2 } )$ . Under the empirically imposed separate site-specific motions, the well-level nuisance group is

$$
G _ { \mathrm { w e l l } } = G _ { \mathrm { r i g } } \times G _ { \mathrm { r i g } } .
$$

Define the well-level canonicalization map

$$
M _ { \mathrm { w e l l } } ( x _ { 1 } , x _ { 2 } ) = \big ( c ( x _ { 1 } ) , c ( x _ { 2 } ) \big ) ,\tag{6.3}
$$

and put

$$
\mathcal { W } _ { \mathrm { c a n } } = M _ { \mathrm { w e l l } } ( \mathcal { W } ) .
$$

Theorem 6.1 and Proposition 4.1(1) imply that $M _ { \mathrm { w e l l } }$ is a maximal invariant for the product action.   
No probability measure is introduced on the noncompact translation group.

## 6.2 Characteristic quotient kernel and Euler signature

$$
\psi : \mathcal { W } _ { \mathrm { c a n } } \longrightarrow \mathbb { R } ^ { 2 q L ^ { 2 } }
$$

place the two canonical crops on fixed $L \times L$ zero canvases and concatenate their entries in a fixed site–channel–row–column order. This map is a Borel injection. For $\sigma > 0$ , define

$$
k _ { \sigma } ( w , w ^ { \prime } ) = \exp \left[ - \frac { \| \psi \{ M _ { \mathrm { w e l l } } ( w ) \} - \psi \{ M _ { \mathrm { w e l l } } ( w ^ { \prime } ) \} \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right] .\tag{6.4}
$$

Proposition 6.2. The kernel (6.4) is positive semidefinite and invariant in each argument under the action $o f G _ { \mathrm { w e l l } }$ . Its kernel mean embedding distinguishes all Borel probability laws on $\mathcal { W } _ { \mathrm { c a n } }$

Proof. Let

$$
k _ { \mathrm { G } } ( u , v ) = \exp \left\{ - \frac { \| u - v \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} , \qquad u , v \in \mathbb { R } ^ { 2 q L ^ { 2 } } .
$$

The kernel $k _ { \mathrm { G } }$ is positive semidefinite and characteristic on Euclidean space [21]. Equation (6.4) is its pullback through the Borel map $\psi \circ M _ { \mathrm { w e l l } }$ , and is therefore positive semidefinite.

For every $g \in G _ { \mathrm { w e l l } }$

$$
M _ { \mathrm { w e l l } } ( g \cdot w ) = M _ { \mathrm { w e l l } } ( w ) .
$$

Consequently,

$$
k _ { \sigma } ( g \cdot w , w ^ { \prime } ) = k _ { \sigma } ( w , w ^ { \prime } ) \quad \mathrm { a n d } \quad k _ { \sigma } ( w , g \cdot w ^ { \prime } ) = k _ { \sigma } ( w , w ^ { \prime } ) ,
$$

which proves invariance in each argument.

Finally, let P and Q be Borel probability measures on $\mathcal { W } _ { \mathrm { c a n } }$ . Equality of their kernel mean embeddings implies, by the characteristic property of the Gaussian kernel,

$$
\psi _ { \# } P = \psi _ { \# } Q .
$$

Because $\psi$ is a Borel injection between standard Borel spaces, the Lusin–Souslin theorem implies that $\psi ( \mathcal { W } _ { \mathrm { c a n } } )$ is Borel and that $\psi ^ { - 1 }$ is Borel on this image. Applying $\psi ^ { - 1 }$ to the two equal pushforward laws therefore yields $P = Q$ □

The secondary endpoint is a finite directional Euler vector. After canonicalization, Otsu thresholding [14] of the nuclear channel and removal of components smaller than four pixels produce a planar cubical complex. Euler characteristic is evaluated along 25 shape-adaptive thresholds in eight lattice directions at both sites. Because the calculation is applied after $M _ { \mathrm { w e l l } }$ , the vector is exactly invariant under $G _ { \mathrm { w e l l } }$ , but it is neither claimed nor used as a maximal invariant.

## 7 Exact design-based testing

Suppose the confirmation experiment contains B paired blocks. In block $b ,$ two eligible wells receive conditions 0 and $^ { 1 , }$ once each. Order the two wells deterministically before observing their assigned conditions, and let

$$
q _ { b , 0 } , q _ { b , 1 } \in \mathcal { W } _ { \mathrm { c a n } }
$$

denote the quotient outcomes of the first and second wells, respectively.

Let $Z _ { b } \in \{ 0 , 1 \}$ be the label of the condition assigned to the first well. Thus, if $Z _ { b } = 1$ , the first well receives condition 1 and the second receives condition 0; if $Z _ { b } = 0$ , these assignments are reversed. Conditional on the unordered eligible wells, their complete potential outcomes and the pretreatment design information, assume that

$$
Z = ( Z _ { 1 } , \dots , Z _ { B } )
$$

is uniform on

$$
\Omega = \{ 0 , 1 \} ^ { B } .
$$

For an assignment $z \in \Omega$ , the quotient samples assigned to conditions 1 and 0 are, respectively,

$$
\mathcal { Q } _ { 1 } ( z ) = \{ q _ { b , 1 - z _ { b } } : b = 1 , \ldots , B \} , \qquad \mathcal { Q } _ { 0 } ( z ) = \{ q _ { b , z _ { b } } : b = 1 , \ldots , B \} .
$$

Let $P _ { 1 }$ and $P _ { 0 }$ denote the corresponding population quotient-outcome laws under conditions 1 and 0. Their population kernel discrepancy is

$$
\mathrm { M M D } _ { k } ^ { 2 } ( P _ { 1 } , P _ { 0 } ) = \mathbb { E } k ( U , U ^ { \prime } ) + \mathbb { E } k ( V , V ^ { \prime } ) - 2 \mathbb { E } k ( U , V ) ,\tag{7.1}
$$

where $U , U ^ { \prime }$ are independent with law $P _ { 1 } , V , V ^ { \prime }$ are independent with law $P _ { 0 }$ , and the four variables are mutually independent. The empirical statistic is the conventional two-sample U-statistic

$$
\begin{array} { l } { { \displaystyle \widehat { \mathrm { M M D } } _ { u } ^ { 2 } ( z ) = \displaystyle \frac { 1 } { B ( B - 1 ) } \sum _ { b \neq b ^ { \prime } } k ( q _ { b , 1 - z _ { b } } , q _ { b ^ { \prime } , 1 - z _ { b ^ { \prime } } } ) + \displaystyle \frac { 1 } { B ( B - 1 ) } \sum _ { b \neq b ^ { \prime } } k ( q _ { b , z _ { b } } , q _ { b ^ { \prime } , z _ { b ^ { \prime } } } ) } } \\ { { \displaystyle ~ - \displaystyle \frac { 2 } { B ^ { 2 } } \sum _ { b , b ^ { \prime } } k ( q _ { b , z _ { b } } , q _ { b ^ { \prime } , 1 - z _ { b ^ { \prime } } } ) . } } \end{array}\tag{7.2}
$$

Matched-block dependence can remove the usual unbiasedness interpretation; here (7.2) is a prespecified discrepancy statistic. The kernel two-sample construction follows the standard MMD formulation [7], but the exactness below does not require unbiased estimation of a superpopulation parameter. Let $\mathcal { Q } _ { \mathrm { p o o l } } = \{ q _ { b , j } : b = 1 , \ldots , B , \ j \in \{ 0 , 1 \} \}$ . The Gaussian bandwidth is fixed before examining the condition labels by setting

$$
\sigma ^ { 2 } = \frac { 1 } { 2 } \operatorname * { m e d i a n } \left\{ \| \psi ( q ) - \psi ( q ^ { \prime } ) \| _ { 2 } ^ { 2 } : q , q ^ { \prime } \in Q _ { \mathrm { p o o l } } , \ q \neq q ^ { \prime } , \ \| \psi ( q ) - \psi ( q ^ { \prime } ) \| _ { 2 } > 0 \right\} .
$$

Because $\mathcal { Q } _ { \mathrm { p o o l } }$ does not change under paired label swaps, the bandwidth is constant over the randomization distribution.

Let $S : \Omega  \mathbb { R }$ be a deterministic test statistic; for the primary analysis, $S ( z ) = \widehat { \mathrm { M M D } } _ { u } ^ { 2 } ( z )$ Define the exact upper-tail randomization p-value by

$$
p ( Z ) = 2 ^ { - B } \sum _ { z \in \Omega } \mathbf { 1 } \{ S ( z ) \geq S ( Z ) \} .\tag{7.3}
$$

Theorem 7.1 (Finite-sample randomization validity). Assume well-defined reagent versions, no cross-well interference, swap-invariant block retention and conditional uniformity of Z on Ω. Under the Fisher sharp null that every confirmation well’s invariant potential outcome is unchanged by the two conditions, the test rejecting when $p ( Z ) \le \alpha$ has conditional rejection probability at most α for every $\alpha \in [ 0 , 1 ]$

Proof. Under the sharp null, all quotient outcomes and label-invariant tuning parameters are fixed over Ω. Let $N = 2 ^ { B }$ and $r ( z ) = \# \{ u \in \Omega : S ( u ) \geq S ( z ) \}$ . Then $p ( z ) = r ( z ) / N$ . For each integer m, at most m assignments have $r ( z ) \leq m ;$ ties can only increase the upper-tail rank. Taking $m = \lfloor \alpha N \rfloor$ and using uniformity proves the result. □

The theorem is exact conditional on the reported assignment mechanism and is an instance of finite-sample randomization inference [6, 13]. The metadata audit can establish that paired siRNAs occur within the same plate randomization set; it cannot itself prove that the original allocation algorithm was uniform. The primary contrast uses one quotient-pixel test. Nine additional contrasts form a separate secondary family adjusted by Holm’s method [8].

## 8 Simulation study

Each of 250 replicates at each efect strength contains eight randomized blocks, two units per block, two sites per unit, six channels and $2 8 \times 2 8$ intrinsic lattice arrays. Treatment adds a concentric annulus with strength $\eta \in \{ 0 , 0 . 3 5 , 0 . 7 0 , 1 . 0 0 \}$ . At $\eta = 0$ , the two treatment potential outcomes of each simulated unit are bytewise identical; diferent units may nevertheless have diferent intrinsic morphologies. After assignment, each site receives an integer shift and quarter turn from a deterministic generator depending on treatment, realized morphology, unit identity and a frozen replicate seed. The analyzer receives neither the group elements nor their generator inputs. All $2 ^ { 8 } = 2 5 6$ paired assignments are enumerated.

Table 1: Paired-swap rejection fractions under treatment- and outcome-dependent acquisition transformations. The interval at $\eta = 0$ is the exact 95% Clopper–Pearson interval [3] over 250 independent replicates.
<table><tr><td>Method</td><td>η = 0</td><td>0.35</td><td>0.70</td><td>1.00</td><td>Assignments</td></tr><tr><td>Exact quotient</td><td>0.052 (0.028, 0.087)</td><td>0.140</td><td>0.848</td><td>0.992</td><td>256</td></tr><tr><td>Raw pixels</td><td>1.000 (0.985, 1.000)</td><td>1.000</td><td>1.000</td><td>1.000</td><td>256</td></tr><tr><td>Moment registration</td><td>0.056 (0.031, 0.092)</td><td>0.540</td><td>0.996</td><td>1.000</td><td>256</td></tr><tr><td>Finite Euler signature</td><td>0.040 (0.019, 0.072)</td><td>0.184</td><td>1.000</td><td>1.000</td><td>256</td></tr></table>

For the exact quotient and post-canonical Euler endpoints, positive-η columns estimate power against a known intrinsic efect. Raw-pixel and moment-registration values are diagnostic rejection fractions because those endpoints are not invariant under the informative acquisition mechanism.

![](images/4c504d92ab0e4c38ea3ec059bdf447c7489cacab8454d5769e6d396d6f89ee50.jpg)  
Figure 1: Simulation rejection fractions. Finite-sample causal validity under informative acquisition pertains to the quotient and post-canonical Euler endpoints; the other methods are diagnostics. The exact quotient begins near the nominal 0.05 level and rises to 0.992, raw pixels remain at 1.0, and moment registration and the Euler endpoint begin near 0.05 and approach 1.0 as efect strength increases.

## 9 RxRx1 confirmation experiment

## 9.1 Units, selection and imposed acquisition

The public RxRx1 release [16, 22] contains 125,510 six-channel 512×512 images, with two nonoverlapping sites per well, 51 experimental batches, four cell types and 1,138 siRNAs, including 1,108 noncontrol perturbations. HUVEC contributes 24 batches. The well is the experimental unit; sites and pixels are components of one structured outcome, not independent replicates. The conditions are applied siRNA reagents, so the estimand is reagent-specific rather than automatically gene-specific.

HUVEC-01 through HUVEC-12 are used for discovery, HUVEC-13 through HUVEC-16 are reserved, and HUVEC-17 through HUVEC-24 supply eight confirmation blocks. Selection uses public 128-dimensional embeddings from discovery rows only. A split-half score combines withinhalf signal-to-noise with agreement of contrast direction. Eligibility requires one two-site well per condition in every discovery batch and identical plate signatures. Greedy matching yields ten disjoint pairs; the highest score defines the primary pair and nine pairs form the secondary family. The row split does not prove that the public embedding generator was trained independently of every confirmation image, so complete algorithmic independence remains an assumption.

The frozen confirmation subset has 160 wells, 320 sites and 1,920 single-channel PNG files. Each site is downsampled to $6 4 \times 6 4$ , retained as 8-bit data and zero-padded. For each published nuisance seed, a BLAKE2b digest of the seed, well identifier and site determines a quarter turn and a signed translation. The transformation also depends on treatment and a coordinate-free morphology score. All channels at one site share a motion, while the two sites receive separate motions. Parameters are logged for audit but excluded from every representation and test routine; the software aborts if nonzero support is cropped.

## 9.2 Confirmatory results

The primary contrast is siRNA 384 (s38230) versus siRNA 747 (s37149). The completeness and same-plate gates passed for all frozen pairs and blocks. The discovery-only manifest has SHA-256 digest b6f350ca96fb9e7a6271845dd6e432ed9dca556144a1a332d8f5af168642b968.

Table 2: Primary RxRx1 pair under imposed lattice rigid motions. The quotient-pixel endpoint is the sole primary test; Euler is secondary; moment registration and raw pixels are acquisition-sensitive diagnostics.
<table><tr><td>Representation</td><td> $\widehat { \mathrm { M M D } } _ { u } ^ { 2 }$ </td><td>Paired-swap p</td></tr><tr><td>Canonical quotient pixels</td><td>0.0036</td><td>0.0078</td></tr><tr><td>Finite directional Euler signature</td><td>0.7535</td><td>0.0078</td></tr><tr><td>Moment registration</td><td>-0.0065</td><td>0.2109</td></tr><tr><td>Raw transformed pixels</td><td>0.1664</td><td>0.0078</td></tr></table>

No secondary contrast remained significant after Holm adjustment. Exact invariance held across all acquisition seeds: maximum clean-versus-contaminated feature discrepancies were zero for the canonical pixels and Euler signature, as were the largest changes in quotient $\widehat { \mathrm { M M D } } _ { u } ^ { 2 }$ and exact $p -$ value. The algebraic and software audit found zero canonical-code failures among 6,400 generated transformations; 29 software checks passed.

Table 3: Canonical-quotient tests for the nine prespecified secondary contrasts. These contrasts are not biological replications of the primary contrast.
<table><tr><td>Pair</td><td>Conditions</td><td> $\widehat { \mathrm { M M D } } _ { u } ^ { 2 }$ </td><td>Swap p</td><td>Holm p</td></tr><tr><td>1</td><td>s21474 vs s21486</td><td>0.0040</td><td>0.0859</td><td>0.1719</td></tr><tr><td>2</td><td>s29342 vs s28343</td><td>0.0045</td><td>0.0156</td><td>0.0703</td></tr><tr><td>3</td><td>s18592 vs s18763</td><td>0.1266</td><td>0.0078</td><td>0.0703</td></tr><tr><td>4</td><td>s20782 vs s21523</td><td>0.0031</td><td>0.0078</td><td>0.0703</td></tr><tr><td>5</td><td>s20401 vs s19178</td><td>0.0228</td><td>0.0156</td><td>0.0703</td></tr><tr><td>6</td><td>s29371 vs s28116</td><td>0.0063</td><td>0.0078</td><td>0.0703</td></tr><tr><td>7</td><td>s17831 vs s20468</td><td>0.0486</td><td>0.0078</td><td>0.0703</td></tr><tr><td>8</td><td>s37483 vs s224918</td><td>-0.0107</td><td>0.1562</td><td>0.1719</td></tr><tr><td>9</td><td>s37712 vs s224940</td><td>0.0761</td><td>0.0078</td><td>0.0703</td></tr></table>

Table 4: Acquisition-only sharp-null stress test over frozen pairs and nuisance seeds. Intervals are descriptive Clopper–Pearson summaries; pair–seed tests share biological blocks.
<table><tr><td>Method</td><td>Tests</td><td>Rejections</td><td>Fraction (95% interval)</td></tr><tr><td>Exact quotient</td><td>200</td><td>0</td><td>0.000 (0.000, 0.018)</td></tr><tr><td>Finite Euler signature</td><td>200</td><td>0</td><td>0.000 (0.000, 0.018)</td></tr><tr><td>Moment registration</td><td>200</td><td>0</td><td>0.000 (0.000, 0.018)</td></tr><tr><td>Raw pixels</td><td>200</td><td>200</td><td>1.000 (0.982, 1.000)</td></tr></table>

Table 5: Approximate-action stress test for the primary pair. Feature error is relative to the exactly transformed baseline. These are empirical sensitivity results; Theorem 5.1 applies only when its Lipschitz assumptions are verified.
<table><tr><td>Perturbation</td><td>Method</td><td>Relative feature error</td><td> $| \Delta \widehat { \mathrm { M M D } } _ { u } ^ { 2 } |$ </td><td>|∆p|</td></tr><tr><td>Rotation 5°</td><td>Exact quotient</td><td>0.8432</td><td>0.0012</td><td>0.0000</td></tr><tr><td></td><td>Euler signature</td><td>0.0779</td><td>0.0860</td><td>0.0000</td></tr><tr><td></td><td>Moment registration</td><td>0.7295</td><td>0.0014</td><td>0.0234</td></tr><tr><td></td><td>Raw pixels</td><td>0.6540</td><td>0.0013</td><td>0.0000</td></tr><tr><td>Rotation 10°</td><td>Exact quotient</td><td>0.8967</td><td>0.0022</td><td>0.0000</td></tr><tr><td></td><td>Euler signature</td><td>0.0827</td><td>0.0530</td><td>0.0000</td></tr><tr><td></td><td>Moment registration</td><td>0.8124</td><td>0.0021</td><td>0.1250</td></tr><tr><td></td><td>Raw pixels</td><td>0.8038</td><td>0.0012</td><td>0.0000</td></tr><tr><td>Rotation 20°</td><td>Exact quotient</td><td>0.9715</td><td>0.0005</td><td>0.0000</td></tr><tr><td></td><td>Euler signature</td><td>0.1076</td><td>0.0898</td><td>0.0000</td></tr><tr><td></td><td>Moment registration</td><td>0.8662</td><td>0.0009</td><td>0.0469</td></tr><tr><td></td><td>Raw pixels</td><td>0.8802</td><td>0.0014</td><td>0.0000</td></tr><tr><td>Two-pixel support crop</td><td>Exact quotient</td><td>0.8253</td><td>0.0026</td><td>0.0000</td></tr><tr><td></td><td>Euler signature</td><td>0.1478</td><td>0.0164</td><td>0.0000</td></tr><tr><td></td><td>Moment registration</td><td>0.6857</td><td>0.0203</td><td>0.2031</td></tr><tr><td></td><td>Raw pixels</td><td>0.3523</td><td>0.0001</td><td>0.0000</td></tr><tr><td>Gaussian noise</td><td>Exact quotient</td><td>0.5949</td><td>0.0007</td><td>0.0000</td></tr><tr><td></td><td>Euler signature</td><td>0.0618</td><td>0.0244</td><td>0.0000</td></tr><tr><td></td><td>Moment registration</td><td>0.3870</td><td>0.0067</td><td>0.1328</td></tr><tr><td></td><td>Raw pixels</td><td>0.1775</td><td>0.0006</td><td>0.0000</td></tr></table>

## 10 Discussion

Under unrestricted contamination, invariance is an observability requirement rather than a modeling preference. A noninvariant target can change while the observed law remains fixed; a maximal invariant captures the entire recoverable orbit. Causal identification then requires a separate design or exchangeability argument. The losslessness theorem adds a third distinction: even a maximal invariant may discard information about how probability is distributed within an orbit. Only a parameter-free conditional law, or a condition such as conditional Haar randomization that implies one, makes quotienting suficient for the full statistical experiment.

The distinction between product and diagonal actions has practical consequences. Sitewise canonicalization is correct for the imposed RxRx1 mechanism because each site receives a separate motion. If a microscope applies one common transformation to all sites, a diagonal quotient should retain relative placement and orientation. Treating that problem as a product action would erase scientifically meaningful relational structure.

The approximate-contamination theorem links orbit error to quotient-law and MMD error, but it also clarifies the limits of the current stress test. Lexicographic canonicalization can be discontinuous. Large feature changes accompanied by stable p-values in Table 5 demonstrate testlevel robustness for the frozen perturbations, not global representation stability. Establishing a

Lipschitz quotient embedding on a scientifically relevant image class is a separate research problem.

The RxRx1 analysis is a controlled unknown-acquisition experiment on real biological images. Its content and batch structure are real; the transformation law is imposed and exactly audited. The causal conclusion is conditional on the reported within-plate assignment mechanism, welldefined reagent versions, no cross-well interference and swap-invariant retention. The primary result concerns a reagent-condition contrast, not automatically a gene-level mechanism. With eight pairs, exact enumeration is a strength, but the randomization distribution has coarse resolution.

## 11 Conclusion

Unknown transformations need not be estimated to conduct causal inference on the information that survives them. Under a measurable group action, orbit invariance is necessary and suficient for uniform observability, and a maximal invariant contains every measurable observable target. Quotient reduction is statistically lossless only under the additional, exact boundary of a parameterfree conditional law given the quotient; conditional Haar contamination is one useful special case. Correct specification of product or diagonal actions determines whether relative component information is preserved. The lattice construction, characteristic quotient kernel and complete paired randomization test turn these distinctions into an auditable workflow for structured outcomes under informative acquisition.

## Data availability

The analysis uses the public RxRx1 dataset subject to its stated licence. The accompanying software is described by the reproducibility statement below. Repository or archival identifiers will be added when the public code archive is released.

## Reproducibility statement

The analysis software downloads the frozen RxRx1 subset, verifies file and manifest hashes, constructs all representations, enumerates the complete assignment distribution, performs multiplicity adjustment and sensitivity analyses, executes unit tests and writes the numerical outputs consumed by the manuscript. The source fingerprint reported for the analysed version is sourcee9f042c387cf.

## Author contributions

Usef Faghihi and Amir Saki contributed equally to the conception, methodology, theoretical development, analysis, interpretation of results and preparation of the manuscript. Both authors reviewed and approved the final manuscript.

## Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

## Acknowledgements and use of AI-assisted tools

During manuscript preparation, the authors used OpenAI Codex for language editing, theoremstructure review and LaTeX reformatting. The authors are responsible for verifying all mathematical statements, empirical results, citations and final wording.

## A Proof and measurability details

## A.1 Proof of Theorem 2.4

Proof. Fix an arbitrary Borel set $B \subseteq \tau$ , and write

$$
E : = h ^ { - 1 } ( B ) .
$$

Since h is Borel, both E and $\mathcal { V } \backslash E$ are Borel subsets of Y. We first verify explicitly that E is saturated with respect to the fibres of M, that is,

$$
M ^ { - 1 } \{ M ( E ) \} = E .
$$

The inclusion $E \subseteq M ^ { - 1 } \{ M ( E ) \}$ is immediate. Conversely, suppose that $y \in M ^ { - 1 } \{ M ( E ) \}$ . Then there exists $z \in E$ such that $\boldsymbol { M } ( y ) = \boldsymbol { M } ( z )$ . By maximality of M, there is some $g \in G$ for which $y = g \cdot z$ . Invariance of h therefore gives

$$
h ( y ) = h ( g \cdot z ) = h ( z ) \in B ,
$$

and hence $y \in E$ . This proves the asserted saturation. Notice that this statement concerns the particular set $E = h ^ { - 1 } ( B )$ : it says that every fibre of M is either entirely contained in E or entirely contained in its complement; it does not assert that the fibres of M are singletons.

We next claim that

$$
M ( E ) \cap M ( \mathcal { V } \setminus E ) = \emptyset .
$$

Indeed, if s belonged to this intersection, there would exist $y \in E$ and $z \in \mathcal { V } \setminus E$ such that

$$
\begin{array} { r } { M ( y ) = s = M ( z ) . } \end{array}
$$

Maximality of M would then place y and z in the same G-orbit, so invariance of h would imply $h ( y ) = h ( z )$ . This is impossible because $h ( y ) \in B$ , whereas $h ( z ) \notin B$

Because Y and S are standard Borel spaces, and M is Borel, the images $M ( E )$ and $M ( \mathcal { V } \backslash E )$ are analytic subsets of S. They need not themselves be Borel, which is why a separation argument is required. By Lusin’s separation theorem, there exists a Borel set $D _ { B } \subseteq S$ satisfying

$$
M ( E ) \subseteq D _ { B } \qquad { \mathrm { a n d } } \qquad M ( \mathcal { V } \backslash E ) \subseteq S \setminus D _ { B } .
$$

We now show that

$$
E = M ^ { - 1 } ( D _ { B } ) .
$$

If $y \in E$ , then $M ( y ) \in M ( E ) \subseteq D _ { B }$ , and hence $y \in M ^ { - 1 } ( D _ { B } )$ . If $y \notin E$ , then $M ( y ) \in M ( \mathcal { V } \backslash E ) \subseteq$ $\mathcal { S } \backslash D _ { B }$ , and hence $y \notin M ^ { - 1 } ( D _ { B } )$ . The two inclusions therefore follow, and

$$
h ^ { - 1 } ( B ) = E = M ^ { - 1 } ( D _ { B } ) \in \sigma ( M ) .
$$

Since $B \subseteq \tau$ was an arbitrary Borel set, h is $\sigma ( M )$ -measurable. The measurable factorization lemma, applicable because $\tau$ is a standard Borel space, therefore yields a Borel map

$$
\bar { h } : S \longrightarrow \mathcal { T }
$$

such that

$$
h = { \bar { h } } \circ M .
$$

It remains to establish the stated uniqueness. Suppose that $\bar { h } _ { 1 } , \bar { h } _ { 2 } : S  \mathcal { T }$ are two Borel maps satisfying

$$
h = \bar { h } _ { 1 } \circ M = \bar { h } _ { 2 } \circ M .
$$

For any $s \in M ( \mathcal { V } )$ , choose $y \in \mathcal { V }$ with $M ( y ) = s$ . Then

$$
\bar { h } _ { 1 } ( s ) = \bar { h } _ { 1 } \{ M ( y ) \} = h ( y ) = \bar { h } _ { 2 } \{ M ( y ) \} = \bar { h } _ { 2 } ( s ) .
$$

Thus $\bar { h } _ { 1 } = \bar { h } _ { 2 }$ on $M ( \mathcal { V } )$ . No uniqueness is claimed outside $M ( \mathfrak { V } )$ , since values of a factor map there do not afect its composition with M. □

## A.2 Existence constructions for maximal invariants

If a compact metrizable group acts continuously on a Polish space, the orbit-valued map $y \mapsto G \cdot y$ is a Borel maximal invariant taking values in the hyperspace of nonempty compact subsets with its Vietoris Borel structure. For a finite group $H = \{ h _ { 1 } , \ldots , h _ { m } \}$ acting by Borel maps, any Borel injection $\nu : \mathcal { V } \to [ 0 , 1 ]$ gives the canonical representative

$$
c _ { H } ( y ) = h _ { j ^ { \star } } \cdot y , \qquad j ^ { \star } = \operatorname* { m i n } \operatorname { a r g m i n } _ { j } \nu ( h _ { j } \cdot y ) ,
$$

which is Borel and maximal. The lattice construction combines an explicit cross-section for noncompact translations with this finite residual minimization.

## A.3 Measurability of the Haar orbit kernel

For compact second-countable G, a jointly Borel action and measurable section make $m \mapsto K _ { m } ( B )$ in (3.2) measurable for every Borel B by integration of the jointly measurable indicator $( g , m ) \mapsto$ $\mathbf { 1 } \{ g \cdot s ( m ) \in B \}$ . Stabilizers may change the map from G onto an orbit but do not change the pushforward orbit measure.

## References

[1] Bhattacharjee, S., Li, B., Wu, X. and Xue, L. (2025) Doubly robust estimation of causal efects for random object outcomes with continuous treatments. arXiv:2506.22754.

[2] Blackwell, D. (1953) Equivalent comparisons of experiments. Ann. Math. Statist., 24, 265–272.

[3] Clopper, C. J. and Pearson, E. S. (1934) The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika, 26, 404–413.

[4] Curry, J., Mukherjee, S. and Turner, K. (2022) How many directions determine a shape and other suficiency results for two topological transforms. Trans. Amer. Math. Soc. Ser. B, 9, 1006–1043.

[5] Eaton, M. L. (1989) Group Invariance Applications in Statistics. Institute of Mathematical Statistics, Hayward, CA.

[6] Fisher, R. A. (1935) The Design of Experiments. Oliver and Boyd, Edinburgh.

[7] Gretton, A., Borgwardt, K. M., Rasch, M. J., Sch¨olkopf, B. and Smola, A. (2012) A kernel two-sample test. J. Mach. Learn. Res., 13, 723–773.

[8] Holm, S. (1979) A simple sequentially rejective multiple test procedure. Scand. J. Stat., 6, 65–70.

[9] Kallenberg, O. (2021) Foundations of Modern Probability, 3rd edn. Springer, Cham.

[10] Kechris, A. S. (1995) Classical Descriptive Set Theory. Springer, New York.

[11] Kurisu, D., Zhou, Y., Otsu, T. and M¨uller, H.-G. (2024) Geodesic causal inference. arXiv:2406.19604.

[12] Le Cam, L. (1986) Asymptotic Methods in Statistical Decision Theory. Springer, New York.

[13] Lehmann, E. L. and Romano, J. P. (2005) Testing Statistical Hypotheses, 3rd edn. Springer, New York.

[14] Otsu, N. (1979) A threshold selection method from gray-level histograms. IEEE Trans. Syst. Man Cybern., 9, 62–66.

[15] Raykov, Y. P., Luo, H., Strait, J. D. and KhudaBukhsh, W. R. (2025) Kernel-based estimators for functional causal efects. arXiv:2503.05024.

[16] [dataset] Recursion (2023) RxRx1: an image set for cellular morphological variation across many biological perturbations. https://www.rxrx.ai/rxrx1.

[17] Robins, J. M. (1986) A new approach to causal inference in mortality studies with a sustained exposure period. Math. Modelling, 7, 1393–1512.

[18] Rosenbaum, P. R. and Rubin, D. B. (1983) The central role of the propensity score in observational studies for causal efects. Biometrika, 70, 41–55.

[19] Rubin, D. B. (1974) Estimating causal efects of treatments in randomized and nonrandomized studies. J. Educ. Psychol., 66, 688–701.

[20] Shin, H.-Y., Kim, K., Lee, K. and Oh, H.-S. (2024) Absolute average and median treatment efects as causal estimands on metric spaces. arXiv:2407.03726.

[21] Sriperumbudur, B. K., Gretton, A., Fukumizu, K., Sch¨olkopf, B. and Lanckriet, G. R. G. (2010) Hilbert space embeddings and metrics on probability measures. J. Mach. Learn. Res., 11, 1517–1561.

[22] Sypetkowski, M. et al. (2023) RxRx1: a dataset for evaluating experimental batch correction methods. arXiv:2301.05768.

[23] Torgersen, E. (1991) Comparison ofStatistical Experiments. Cambridge University Press, Cambridge.

[24] Turner, K., Mukherjee, S. and Boyer, D. M. (2014) Persistent homology transform for modeling shapes and surfaces. Information and Inference, 3, 310–344.

[25] Wijsman, R. A. (1990) Invariant Measures on Groups and Their Use in Statistics. Institute of Mathematical Statistics, Hayward, CA.

## Figure caption list

Figure 1. Simulation rejection fractions across intrinsic efect strengths; causal validity under informative acquisition pertains to invariant endpoints.

## Table caption list

Table 1. Simulation rejection fractions.

Table 2. Primary RxRx1 contrast.

Table 3. Prespecified secondary contrasts.

Table 4. Acquisition-only sharp-null stress test.

Table 5. Approximate-action stress test.
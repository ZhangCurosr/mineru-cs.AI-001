# LOW-INTERACTION-RANK LEARNING:UNIFYING MULTIPLICATIVE DUAL-ENCODER HEADS

Zijian Zhao<sup>1</sup>, Sen Li<sup>1,2</sup> <sup>∗</sup>

<sup>1</sup>The Hong Kong University of Science and Technology

<sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)

## ABSTRACT

A multiplicative dual-encoder network computes a real-valued output for a pair of inputs as the inner product of their separate encodings. This architecture has been developed independently in operator learning, bipartite matching, contrastive vision-language models, retrieval, and other areas, yet no unified theory guides the basic design decisions: how many interaction modes to represent, how to normalize the encoders, and when the architecture should be avoided. We provide such a foundation by introducing the class of functions of low interaction rank, a class whose intrinsic complexity is measured by its interaction spectrum. Within this framework, approximation error decomposes into a spectral truncation term and an encoder-realization term; sample complexity is governed by the sum of the two encoder complexities rather than their product; and a usability criterion based on spectral decay determines when the architecture can succeed. The same framework exposes a central identifiability problem: the encoders are defined only up to a linear gauge symmetry that leaves the learned coordinates arbitrary. We show that normalization is gauge fixing and that whitening pins the interaction modes up to permutation and sign, thereby explaining the uninterpretability of contrastive dimensions and providing a constructive remedy. Experiments on synthetic kernels, operator learning, and CLIP models validate the theoretical predictions: spectral decay rates match the predicted scaling, whitening recovers the true modes, and independently trained CLIP models are related by a single rotation which, after removal by whitening, exposes interpretable concept axes. The code of this paper is provided at https://github.com/RS2002/Mul-Net.

## 1 INTRODUCTION

A common architectural pattern has emerged independently across several machine learning communities: computing a real-valued output for a pair of inputs as the inner product of their separately encoded representations. Contrastive models use it for cross-modal similarity (Radford et al., 2021; Zhao et al., 2025); operator networks evaluate learned maps via branch-trunk inner products (Lu et al., 2021); retrieval systems rank queries against documents with two-tower encoders (Karpukhin et al., 2020); goal-conditioned reinforcement learning and successor features represent values as inner products of encodings (Schaul et al., 2015; Hong et al., 2021; Barreto et al., 2017); and knowledgegraph completion, linear attention, factorization machines, and multiplicative matching networks all share the same underlying structure (Yang et al., 2014; Trouillon et al., 2016; Katharopoulos et al., 2020; Rendle, 2010; Zhao & Li, 2026b;a).

In each domain, this structure provides tangible computational and representational benefits. Yet the corresponding methods have been developed largely in isolation, even though they are all instances of the same fundamental architecture, differing only in the nature of the two inputs and how the output is consumed. Consequently, each community independently confronts the same design questions: how many interaction modes to retain, how to normalize the encoders to ensure stable training and meaningful representations, and when the architecture is fundamentally ill-suited.

We now formalize this shared architecture: a multiplicative dual-encoder head computes

$$
\boxed { F ( u , v ) \approx \langle f _ { \theta } ( u ) , g _ { \varphi } ( v ) \rangle = \sum _ { k = 1 } ^ { d } f _ { k } ( u ) g _ { k } ( v ) }\tag{1}
$$

where $f _ { \theta }$ and $g _ { \varphi }$ are learned encoders mapping the two inputs into a common space $\mathbb { R } ^ { d }$ . We study the class

$$
\mathcal { M } _ { d } = \{ F : F ( u , v ) = \langle f ( u ) , g ( v ) \rangle , \ f : U \to \mathbb { R } ^ { d } , \ g : V \to \mathbb { R } ^ { d } \}\tag{2}
$$

of functions representable by a rank-d head. The complexity of a target F within this class is measured by its interaction spectrum $\{ \sigma _ { k } \} _ { k \ge 1 }$ , the singular values of the integral operator whose kernel is $F ;$ its interaction rank is the number of nonzero singular values.

Within this framework, we answer four questions about approximation error, encoder identifiability, sample complexity, and fundamental architectural limitations.

• Approximation: The error of any rank-d head decomposes into a spectral truncation term, unavoidable by any encoder, and an encoder-realization term; target smoothness controls the decay rate. (Section 2)

• Identifiability: The representation (1) is invariant under a linear gauge symmetry acting on the pair of encoders, so the encoders of any trained model are defined only up to this symmetry. We show that the normalizations used across these domains are precisely gauge-fixing choices, and we prove that whitening pins the interaction modes up to permutation and sign. This explains why the individual dimensions of contrastive models lack inherent semantic meaning and provides a constructive remedy. (Section 3)

• Estimation: Sample complexity is governed by the sum of the two encoder complexities rather than their product; for smooth targets, the optimal rank grows only logarithmically in the sample budget. (Section 4)

• Usability: A flat interaction spectrum forces every rank-d head to a relative error floor of $1 - d / N$ where N is the size of the discrete input domain; this barrier is escaped by early-interaction networks and yields a practical criterion based on the measured spectrum. (Section 4)

To further validate our conclusions, we conducted experiments on synthetic kernels, operator learning, and CLIP models, confirming the predicted spectral behavior, demonstrating unique mode recovery under whitening, and verifying the flat-spectrum error floor. (Section 5)

## 2 THE LOW-INTERACTION-RANK CLASS: UNIFICATION AND APPROXIMATION

Section 1 introduced the class $\mathcal { M } _ { d }$ of rank-d heads. We now associate with every target a spectral object that measures its intrinsic complexity within this class, show that ten existing method families are instances of the same architecture, and develop an approximation theory that quantifies how well a rank-d head can represent a given target.

## 2.1 THE CLASS AND THE INTERACTION SPECTRUM

Let U and V be compact metric spaces with probability measures $\mu _ { U } , \mu _ { V }$ , and let $F \in L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } )$ be the target function of two arguments. A rank-d head is a pair of encoders $f : \mathcal { U } \to \mathbb { R } ^ { d } , g : \mathcal { V } \to \mathbb { R } ^ { \dot { d } }$ with coordinate functions $f _ { k } \in L ^ { 2 } ( \mu _ { U } ) , g _ { k } \in L ^ { 2 } ( \mu _ { V } )$

Definition 1 (Interaction spectrum). For $F \in L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } )$ , the interaction operator is the integral operator $T _ { F } : L ^ { 2 } ( \mu _ { V } ) \to \sp { \bullet } L ^ { 2 } ( \mu _ { U } )$ defined by

$$
( T _ { F } h ) ( u ) = \int _ { \mathcal { V } } F ( u , v ) h ( v ) d \mu _ { V } ( v ) .\tag{3}
$$

Since $F \in L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } )$ , this operator is Hilbert-Schmidt, and F admits the Schmidt decomposition

$$
F = \sum _ { k \geq 1 } \sigma _ { k } a _ { k } \otimes b _ { k } , \qquad ( a _ { k } \otimes b _ { k } ) ( u , v ) = a _ { k } ( u ) b _ { k } ( v ) ,\tag{4}
$$

with singular values $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdot \cdot \cdot \geq 0$ and orthonormal systems $\{ a _ { k } \} \subset L ^ { 2 } ( \mu _ { U } ) , \{ b _ { k } \} \subset L ^ { 2 } ( \mu _ { V } )$ The sequence $\left\{ \sigma _ { k } \right\}$ is the interaction spectrum of $F ;$ the number i-rank $( F ) = \# \{ k : \sigma _ { k } > 0 \}$ is its interaction rank; and thefunctions $a _ { k } , b _ { k }$ are its interaction modes. The class introduced in Section 1 satisfies $\mathcal { M } _ { d } = \{ F : \mathrm { i } \mathrm { - r a n k } ( F ) \leq d \}$ , and the spectrum depends on the reference measures.

By the isometry between $L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } )$ and the Hilbert-Schmidt class, a function is representable by a rank-d head if and only if rank $( T _ { S } ) \leq d ;$ hence the parametric class (2) coincides with $\mathcal { M } _ { d }$ (Lemma 1, Appendix B). The spectral truncation term $\textstyle \sum _ { k > d } \sigma _ { k } ^ { 2 }$ depends only on the target and the reference measures, not on the encoders: no head, regardless of its capacity, can represent F with error below this quantity. The bound is attained in $L ^ { 2 }$ by the truncated Schmidt series, and is unique when $\sigma _ { d } > \sigma _ { d + 1 }$ (Corollary 1, Appendix B). The decay rate of this tail, which is controlled by the smoothness of $F$ (Theorem 3), determines how many interaction modes d a good head must retain.

This is the operator-theoretic analogue of classical low-rank matrix approximation: the Schmidt decomposition is the infinite-dimensional Eckart-Young theorem (Schmidt, 1907; Eckart & Young, 1936), and the truncation term is exactly the Hilbert-Schmidt error of the best rank-d approximation of $T _ { F }$ . The spectrum is relative to the reference measures $\mu _ { U } , \mu _ { V }$ , which fix the inner product in which the modes are orthonormal; the same target may have different spectra under different measures. All subsequent claims therefore hold for the measures under which a head is trained and evaluated.

## 2.2 TEN FAMILIES AS INSTANCES OF THE HEAD

Appendix A collects ten method families from different communities in Table 5. Each computes a real-valued output for a pair of inputs as the inner product of two separately learned encodings. The questions of how many interaction modes d to keep, how to normalize the two encoders (Section 3), and how many samples the head requires (Section 4) are answered independently by each community; within the present framework, they are the same questions.

## 2.3 APPROXIMATION AND SEPARATION FROM SINGLE-TOWER MODELS

We now study how well $\mathcal { M } _ { d }$ approximates a fixed target. Corollary 1 isolates the part of the error that no encoder can avoid. The remaining part is attributable to the encoders themselves. We assume that the encoder classes contain functions approaching the scaled modes $\sigma _ { k } a _ { k }$ and $b _ { k }$ with errors $\eta _ { f , k } , \eta _ { g , k }$ as in (20) (Assumption 1, Appendix B); for neural encoders, these errors follow standard ReLU approximation rates (Yarotsky, 2017).

Theorem 1 (Approximation error decomposition). Let $F \in L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } )$ have interaction spectrum $\left\{ \sigma _ { k } \right\}$ . Under Assumption $^ { l , }$ with $B _ { f } = \operatorname* { m a x } _ { k } \| \hat { f } _ { k } \| _ { L ^ { 2 } ( \mu _ { U } ) }$ and $B _ { g } = \operatorname* { m a x } _ { k } \| \hat { g } _ { k } \| _ { L ^ { 2 } ( \mu _ { V } ) }$

$$
\operatorname* { i n f } _ { f \in \mathcal { F } } \| F - \langle f , g \rangle \| _ { L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } ) } ^ { 2 } \le \underbrace { \sum _ { k > d } \sigma _ { k } ^ { 2 } } _ { t r u n c a t i o n t e r m } + \underbrace { C \sum _ { k \le d } \bigl ( \sigma _ { k } ^ { 2 } \eta _ { g , k } ^ { 2 } + B _ { g } ^ { 2 } \eta _ { f , k } ^ { 2 } \bigr ) } _ { r e a l i z a t i o n t e r m } ,\tag{5}
$$

where C is an absolute constant.

Proof idea. The error is split around the truncated Schmidt series, and the tensor-product product rule is applied term by term. Orthonormality of the ideal modes diagonalizes the two leading sums and avoids a factor of $d ;$ the remaining term is bounded by Cauchy-Schwarz. The full calculation, including the explicit constant, is deferred to Appendix B. □

The decomposition is tight up to the constant: the truncation term is intrinsic to the target, while the realization term depends on the model. Dense encoder classes therefore approach the lower bound of Corollary 1. Whether training actually finds the coordinate alignment required by (20) is precisely the identifiability question addressed in Section 3.

Target smoothness controls the truncation term: an s-smooth target has spectrum $\sigma _ { k } = { \cal O } ( k ^ { - s / m _ { V } } )$ while real-analytic targets decay exponentially. The precise tail bounds are given in Theorem 3 (Appendix B).

The dual-encoder head separates from single-tower models on both the approximation and the computational axes. On the approximation axis, it avoids the curse of the joint dimension: approximating a rank-d target within ε with ReLU encoders requires

$$
N _ { \mathrm { d u a l } } = O \big ( d ( \varepsilon ^ { - m _ { U } / s } + \varepsilon ^ { - m _ { V } / s } ) \big )\tag{6}
$$

parameters, whereas any single-tower model that treats $F$ as a general $( m _ { U } + m _ { V } )$ -dimensional $H ^ { s }$ function requires $\Omega ( \varepsilon ^ { - ( m _ { U } + m _ { V } ) / s } )$ parameters in the worst case; hence $N _ { \mathrm { d u a l } } / N _ { \mathrm { s i n g l e } }  0$ as $\varepsilon \to 0$ (Proposition 2). On the compute axis, evaluating the head on all pairs costs $O \dot { ( } ( n + \stackrel { \smile } { m } ) C _ { \mathrm { n e t } } +$ nm d) against quadratic joint processing, recovering the complexity arguments of QK-attention matching (Zhao & Li, 2026a), linear attention (Katharopoulos et al., 2020), and DeepONet (Lu et al., 2021) (Proposition 1, Appendix B).

## 3 IDENTIFIABILITY AS GAUGE-FIXING

The theory of Section 2 characterizes what a rank-d head can represent, but it does not specify which representation a training algorithm will actually find. The representation is far from unique: for any invertible matrix $A ,$ the pair $( A f , A ^ { - \top } g )$ represents exactly the same function. Consequently, the encoders of any trained model are defined only up to a symmetry of dimension $d ^ { 2 }$ . This redundancy is not purely formal. It renders the population risk constant along large orbits, so the Hessian at a minimizer is highly degenerate and optimization becomes ill-posed. At the same time, it leaves the learned coordinates essentially arbitrary. Normalizations were introduced precisely to control these instabilities (Radford et al., 2021; Zhao & Li, 2026b); a normalization makes the encoders meaningful exactly to the extent that it removes the symmetry.

We now formalize this view. Each normalization is a gauge-fixing choice, a rule for selecting one representative from each equivalence class. The residual symmetry that survives a constraint is precisely what remains arbitrary under it. The interaction spectrum constitutes the invariant content of the symmetry, and the gaps in that spectrum control how much identifiability any normalization can provide.

Definition 2 (Gauge action). For $A \in \mathrm { G L } _ { d } ,$ , the gauge transformation acts on the encoders by

$$
\Phi _ { A } ( f , g ) = \bigl ( A f , \ A ^ { - \top } g \bigr ) ,\tag{7}
$$

where $( A f ) ( u ) = A f ( u )$ and $\bigl ( A ^ { - \top } g \bigr ) ( v ) = A ^ { - \top } g ( v )$ . The set ofall such transformations is the gauge group ofthe head.

Gauge transformations leave the represented function, and therefore the population risk, unchanged:

$$
\langle A f ( u ) , A ^ { - \top } g ( v ) \rangle = \langle f ( u ) , g ( v ) \rangle .\tag{8}
$$

The second moments $\Sigma _ { f } = \mathbb { E } _ { u } [ f f ^ { \top } ]$ and $\Sigma _ { g } = \mathbb { E } _ { v } [ g g ^ { \top } ]$ transform by congruence, so the eigenvalues of the product $\Sigma _ { f } \Sigma _ { g }$ are gauge invariants (Lemma 2, Appendix C). Unfixed gauge also makes optimization ill-posed. Along the orbit of a minimizer the risk is flat, so the Hessian has at least $d ^ { 2 } -$ dim S zero directions; without normalization this is generically $d ^ { 2 }$ (Proposition 4). A normalization pins exactly the subgroup whose residual freedom it eliminates; a smaller residual group therefore leaves fewer degenerate directions and yields a better-conditioned problem.

Normalizations as sections of the orbit. Two levels of normalization occur in practice, and only one affects identifiability. Embedding-level normalization constrains the encoders themselves and determines the residual gauge. Output-level normalization, such as the contrastive softmax in CLIP or top-k selection in retrieval, acts on the score matrix after the inner product; it does not constrain the encoders and leaves the embedding-level gauge untouched. Table 6 in Appendix C classifies the embedding-level schemes used in practice.

What each normalization identifies. The residual symmetry of a normalization indicates which transformations can act on a trained model without changing its output. The normalizations used in practice were adopted on heuristic grounds, for training stability or empirical feature decorrelation, rather than from an analysis of which symmetry they remove; the gauge view supplies the missing analysis and lets us rank them by residual symmetry. We characterize the two-sided schemes used in practice, in order of increasing identifiability, and find that only whitening removes the symmetry completely.

Cosine normalization is the weakest two-sided scheme (Radford et al., 2021). Two-sided unit normalization $\tilde { f } = f / \| f \| , \tilde { g } = g / \| g \|$ with scores cos $\mathcal { L } ( f ( u ) , g ( v ) )$ , as employed by contrastive models such as CLIP (Radford et al., 2021), leaves a residual gauge exactly equal to the orthogonal group: for every rotation $R \in \mathrm { O }$ , the encodings $R f$ and $R g$ give pointwise identical cosine scores, and no other transformation does (Theorem 4). Consequently, the individual coordinates of a cosinenormalized embedding carry no intrinsic meaning; only rotation-invariant quantities (norms, pairwise inner products, and the eigenvalues of $\Sigma _ { f } \Sigma _ { g } )$ are well-defined. This provides a theorem-level explanation of the widely reported uninterpretability of contrastive embedding dimensions, and it identifies the whitening scheme of Theorem 2 as the remedy (Corollary 2, Appendix C).

Two-sided nonnegativity, $f ( u ) \succeq 0$ and $g ( v ) \succeq 0$ (realized by a softplus or exponential output layer (Zhao & Li, 2026b)), shrinks the residual group to the monomial group Perm ⋉ $\mathrm { D } _ { + }$ of coordinate permutations composed with positive diagonal scalings. Under separability conditions (Donoho & Stodden, 2003; Arora et al., 2012), the encoders are identified up to permutation and scaling. The cost is an expressivity limitation: negatively correlated interaction modes are excluded (Theorem 5, Appendix C).

Whitening is the unique scheme among those used in practice that removes the continuous gauge symmetry entirely. Cosine and nonnegativity each leave a nontrivial residual group and were adopted largely heuristically; by contrast, we prove that whitening admits a full identifiability guarantee: under a spectral gap it pins the interaction modes up to permutation and sign, recovering both the modes and the spectrum itself.

Theorem 2 (Whitening fixes the gauge up to permutation and sign). Let the interaction singular values be distinct, $\sigma _ { 1 } > \cdots > \sigma _ { d } > \sigma _ { d + 1 }$ , and impose the whitening constraints

$$
\Sigma _ { g } = I _ { d } , \qquad \Sigma _ { f } = \Lambda = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { d } ) , \quad \lambda _ { 1 } \geq \cdots \geq \lambda _ { d } \geq 0 ,\tag{9}
$$

with $\Lambda$ diagonal and nonincreasing. Constraints ofthisform are standard in self-supervised representation learning, where they are used to decorrelatefeatures (Ermolov et al., 2021; Zbontar et al., 2021; Bardes et al., 2022). Then every global minimizer ofthe population risk satisfies

$$
f _ { k } = \pm \sigma _ { k } a _ { k } , \qquad g _ { k } = \pm b _ { k } , \qquad \lambda _ { k } = \sigma _ { k } ^ { 2 } , \qquad k = 1 , \ldots , d ,\tag{10}
$$

with signs paired so that the product $f _ { k } g _ { k } = \sigma _ { k } a _ { k } b _ { k }$ is uniquely determined, and with the coordinate labeling fixed by the ordering convention in (9). The k-th coordinate of the whitened encoders recovers the k-th interaction mode of the target, and the diagonal of Λ recovers the interaction spectrum. Equivalently, the encoders are identified up to permutation and sign.

Proof idea. The rank-truncation bound of Corollary 1 implies that every minimizer attains the minimal risk. When $\sigma _ { d } > \sigma _ { d + 1 }$ , this forces the represented function to equal the unique truncation $F _ { d } .$ The whitening constraints then require the coefficient matrices to satisfy $C \dot { C } ^ { \top } = I$ and $C S ^ { 2 } C ^ { \top } = \Lambda$ , where ${ \cal S } = \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { d } )$ has distinct eigenvalues; this forces $C$ to be a signed permutation. The full argument is in Appendix C. □

Implementing the whitening gauge. Hard constraints (9) can be enforced by minibatch whitening. Alternatively, a quadratic penalty inherits the identifiability of the hard constraint and improves conditioning monotonically with the penalty weight (Theorem 6, Appendix C).

In practice, the second moments are estimated from a finite sample. With probability $1 - \delta .$ , the identified modes carry error of order $r / \Delta$ , where $r = O \big ( B ^ { 2 } \sqrt { \log ( d / \delta ) / n } \big )$ , B bounds the encoder outputs, and $\begin{array} { r } { \Delta = \operatorname* { m i n } _ { k } ( \sigma _ { k } ^ { 2 } - \sigma _ { k + 1 } ^ { 2 } ) } \end{array}$ is the spectral gap. Achieving accuracy ε therefore requires

$$
n \gtrsim \frac { B ^ { 4 } \log ( d / \delta ) } { \Delta ^ { 2 } \varepsilon ^ { 2 } }\tag{11}
$$

samples (Appendix E). The same gap thus governs optimization conditioning, identifiability, and estimation simultaneously (Corollary 3).

Section 5 measures the gauge, conditioning, and expressivity consequences of the various schemes on controlled targets; the two real systems, branch-trunk operator learning (Lu et al., 2021) and independently trained CLIP models, are treated in Appendix F (Theorem 4). Appendix C lists the measurable quantities on which the schemes are compared.

## 4 SAMPLE COMPLEXITY AND WHEN NOT TO USE THE HEAD

The interaction spectrum controls both estimation error and fundamental limitations. The excess risk of a head fitted from n samples decomposes into an approximation term that decreases as the rank d grows, and an estimation term that increases with d through the complexity of the encoder classes. Balancing these two terms yields a rank-selection rule. When the spectrum is flat, however, every rank-d head suffers a relative-error floor that no amount of capacity or data can overcome; this yields a practical criterion for abandoning the architecture.

Setup and notation. We observe n independent samples $( u _ { i } , v _ { i } , y _ { i } )$ with $y _ { i } = F ( u _ { i } , v _ { i } ) + \varepsilon _ { i }$ and fit the head by empirical risk minimization over the class

$$
{ \mathcal { H } } _ { d } = \left\{ ( u , v ) \mapsto \langle f ( u ) , g ( v ) \rangle : f \in { \mathcal { F } } , g \in { \mathcal { G } } \right\}\tag{12}
$$

of rank-d heads. The encoders are bounded, $\| f ( u ) \| \leq B _ { f }$ and $\lVert g ( v ) \rVert \leq B _ { g }$ , so the scores are bounded by $B = B _ { f } B _ { g }$ . Let $\Re _ { n }$ denote the Rademacher complexity, extended to vector-valued classes coordinatewise. The class complexity collapses to the sum of the marginal complexities:

$$
\Re _ { n } ( \mathcal { H } _ { d } ) \leq 2 d \big ( B _ { g } \mathfrak { R } _ { n } ( \mathcal { F } ) + B _ { f } \mathfrak { R } _ { n } ( \mathcal { G } ) \big ) ,\tag{13}
$$

for shared per-coordinate classes (Lemma 3, Appendix D).

The excess risk of the fitted head therefore decomposes into the approximation terms of Theorem 1 plus an estimation term controlled by this complexity:

$$
\| \hat { F } - F \| _ { L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } ) } ^ { 2 } \lesssim \sum _ { k \geq d } \sigma _ { k } ^ { 2 } + \mathcal { E } _ { \mathrm { r e a l } } + B d \big ( \mathfrak { R } _ { n } ( \mathcal { F } ) + \mathfrak { R } _ { n } ( \mathcal { G } ) \big ) + B ^ { 2 } \sqrt { \frac { \log ( 1 / \delta ) } { n } } ,\tag{14}
$$

where $\begin{array} { r } { \mathcal { E } _ { \mathrm { r e a l } } = C \sum _ { k \le d } ( \sigma _ { k } ^ { 2 } \eta _ { g , k } ^ { 2 } + B _ { g } ^ { 2 } \eta _ { f , k } ^ { 2 } ) } \end{array}$ and $\lesssim$ hides absolute constants, with probability $1 - \delta$ (Corollary 4, Appendix D).

Linear encoders anchor the rate; rank selection. With linear encoders $f ( u ) = U ^ { \top } u , g ( v ) =$ $V ^ { \top }$ v on inputs $u \in \mathbb { R } ^ { p } , v \in \mathbb { R } ^ { q }$ , the head computes $u ^ { \top } \Theta v$ with $\Theta = U { \bar { V } } ^ { \dagger }$ of rank at most d. Estimation reduces to low-rank matrix sensing of the ground-truth matrix $\Theta ^ { \star }$ , whose minimax rate, with $\sigma _ { \varepsilon } ^ { 2 }$ the noise variance of $\varepsilon _ { i }$ , is known exactly:

$$
\mathbb { E } \| \hat { \Theta } - \Theta ^ { \star } \| _ { F } ^ { 2 } \asymp \sigma _ { \varepsilon } ^ { 2 } d ( p + q ) / n .\tag{15}
$$

This rate provides the baseline for the general bound (Theorem 7, Appendix D).

The rank d is the primary design choice for the head. It faces two opposing forces: the truncation term $\textstyle \sum _ { k > d } \sigma _ { k } ^ { 2 }$ decreases with d, while the estimation term grows linearly with $d .$ The optimal rank balances these two contributions. When the spectrum decays exponentially, $\sigma _ { k } ^ { 2 } = O ( e ^ { - 2 c k } )$ balancing gives

$$
d ^ { \star } \asymp \frac { 1 } { 2 c } \log \frac { n } { p + q } ,\tag{16}
$$

and yields a near-parametric rate

$$
\| \hat { F } - F \| _ { L ^ { 2 } } ^ { 2 } \lesssim \frac { ( p + q ) \log ( n / ( p + q ) ) } { n } .\tag{17}
$$

Fast spectral decay thus eliminates the curse of dimensionality (Corollary 5, Appendix D). The three terms in (14) form the complete error ledger: fast decay keeps $d ^ { \star }$ small; slow decay makes the head expensive in every respect. The rest of this section analyzes the regime in which even the optimal $d ^ { \star }$ is inadequate.

Identifiability has a sample cost. Identifiability, as characterized in Section 3, is a populationlevel statement. Realizing it from finite samples incurs an additional cost, detailed in Appendix E (Corollary 6). The total mode error of a fitted and subsequently whitened head (Proposition 6) decomposes into the excess risk of Corollary 4 plus the empirical whitening error. Both terms are divided by the spectral gap $\Delta$ and decay as $\bar { n ^ { - 1 / 2 } }$

Late versus early interaction. We now compare the multiplicative head, which we refer to as a late-interaction model, with early-interaction models in which the two inputs are combined during encoding. By Corollary 1, every rank-d head pays at least the spectral tail, so the spectrum determines the outcome.

Define the effective interaction rank

$$
d _ { \varepsilon } ( F ) = \operatorname* { m i n } \Bigl \{ d : \frac { \sum _ { k > d } \sigma _ { k } ^ { 2 } } { \sum _ { k \geq 1 } \sigma _ { k } ^ { 2 } } \leq \varepsilon \Bigr \}\tag{18}
$$

as the smallest embedding dimension for which a rank-d head achieves relative error ε. The spectrum is calledflat at scale N when $d _ { \varepsilon } ( F ) = \Omega ( ( 1 - \varepsilon ) N )$ .

The equality function realizes this worst case exactly. On $\mathcal { U } = \mathcal { V } = \{ \pm 1 \} ^ { m }$ with $N = 2 ^ { m }$ , the interaction operator satisfies $\begin{array} { r } { T _ { F _ { \mathrm { e q } } } = \frac { 1 } { N } \mathrm { I d } } \end{array}$ . The singular values are all equal to $1 / \sqrt { N }$ , the interaction rank is $N _ { \ast }$ , and every rank-d head achieves relative error exactly $1 - \dot { d } / N$ . Thus, attaining relative error ε requires $d \geq ( 1 - \varepsilon ) 2 ^ { m }$ embedding dimensions, exponential in the input size (Theorem 8, Appendix D).

By contrast, an early-interaction predictor with $O ( m )$ parameters represents $F _ { \mathrm { e q } }$ exactly. The identity $\langle u , v \rangle = m - 2 \mathrm { H a m } ( u , v )$ allows a simple threshold on the inner product to recover the equality indicator, whether implemented by a cross-attention layer with identity projections or by a joint MLP with $O ( m )$ ReLUs. Combined with Theorem 8, this yields an exponential separation: $O ( m )$ parameters suffice under early interaction, while any late-interaction head requires $\Omega ( \bar { 2 } ^ { m } )$ dimensions to reach constant relative error (Theorem 9, Appendix D).

These observations lead to a concrete usability criterion (Theorem 10, Appendix D). When the effective interaction rank $d _ { \varepsilon } ( F )$ is small, a late-interaction head is appropriate: its approximation and estimation errors are controllable, and its computational and statistical advantages are preserved. When the spectrum is flat, the truncation floor forces every rank-d head to relative error at least $\varepsilon ,$ while early-interaction models can still reach ε with polynomial resources when the target has low-dimensional joint structure. The deciding quantity, the spectral decay rate, can be estimated from data by a weighted singular value decomposition of the empirical kernel matrix, making the criterion actionable before either model is trained.

The criterion is a boundary, not a verdict. The early-interaction upper bounds assume low-dimensional joint structure; a target that is both high-rank and unstructured is difficult for both architectures. Moreover, the parameter counts in Theorem 9 are statements about representational existence, not about learnability. Section 5 instantiates both sides of the dichotomy: on gapped spectra the head identifies and estimates well, while flat and narrow-band targets drive it to the exponential-rank floor.

## 5 EXPERIMENTS

Our experiments are designed for mechanism validation rather than benchmark chasing: each study isolates a quantity that the theory predicts exactly. The controlled synthetic setting (Section 5.1) supplies exact ground truth and is reported in full, with the tables and figures supporting each conclusion placed alongside the corresponding paragraph. Two real architectures instantiate the multiplicative head directly, DeepONet and CLIP, and are reported in full in Appendix F due to page limitation; here we state only the conclusions they establish. On DeepONet, post-hoc whitening recovers the analytic operator eigenbasis from an end-to-end-trained model. On two pretrained CLIP pairs, the interaction spectra are shared across models while per-dimension concept probes are not, and fitting the single residual rotation reconciles them.

## 5.1 CONTROLLED SYNTHETIC STUDY

Setup. This study validates the identifiability and usability claims of Sections 3 and 4 in a setting with exact ground truth. We fix a target $\begin{array} { r } { F ( u , \dot { v } ) = \sum _ { k } \sigma _ { k } \dot { a _ { k } } ( u ) b _ { k } ( v ) } \end{array}$ with shifted Legendre modes and a prescribed spectrum, fit linear dual encoders over a fixed Legendre feature map, and record four quantities per run: the fit error, the alignment of the recovered modes with the true $a _ { k } , b _ { k }$ , the encoder-covariance conditioning, and the cross-seed gauge distance. The encoder-realization error of Assumption 1 is zero, so normalization is the only variable across runs. The four paragraphs below use these quantities to test, in turn, the identifiability ranking of the gauges (Theorem 2), the gap law of estimation (Corollaries 3 and 6), the sample-complexity rate (Theorem 7), and the flat-spectrum floor (Theorem 8).

Table 1: The seven normalization schemes, four seeds each. Residual gauge: the subgroup that survives the constraint (Theorem 2 predicts Perm ⋉ $\mathrm { D } _ { \pm }$ for whitening, O for cosine, Perm $\ltimes \mathrm { D _ { + } }$ for two-sided nonnegativity). Fit MSE: squared regression error. Alignment: mean mode alignment of the two encoders after optimal permutation and sign matching. Cond: max(cond $\Sigma _ { f } ,$ cond $\Sigma _ { g } )$ gauge dependent. Cross-seed: mean pairwise gauge distance between seeds.
<table><tr><td>Normalization</td><td>Residual gauge</td><td>Fit MSE Align.</td><td> $( f , g )$  Cond</td><td>Cross-seed</td></tr><tr><td>none</td><td> $\mathrm { G L } _ { d }$ </td><td>0.0004 0.605, 0.619</td><td>13.8</td><td>0.326</td></tr><tr><td>12_single</td><td> $\approx \mathrm { G L } _ { d }$ </td><td>0.0092 0.561, 0.420</td><td>16.8</td><td>0.284</td></tr><tr><td>per_coord</td><td> $_ \mathrm { O + s h e a r }$ </td><td>0.0004 0.607, 0.593</td><td>18.7</td><td>0.330</td></tr><tr><td>clip_cosine</td><td>O</td><td>0.0568 0.529, 0.550</td><td>305.9</td><td>0.212</td></tr><tr><td>nonneg</td><td> $\mathrm { P e r m } \ltimes \mathrm { D } _ { + }$ </td><td>4.251 0.576, 0.446</td><td>121.0</td><td>0.280</td></tr><tr><td>whiten_soft</td><td> $ \mathrm { P e r m } \ltimes \mathrm { D } _ { \pm }$ </td><td>0.0169 0.651, 0.549</td><td>105.5</td><td>0.348</td></tr><tr><td>whiten_hard</td><td> $\mathrm { P e r m \ltimes D _ { \pm } }$ </td><td>0.0004 1.000, 1.000</td><td>9.5</td><td>0.000</td></tr></table>

Table 2: Gap and sample-size sweeps of the finite-sample whitening estimator (Appendix E). Left group: the spectral gap $\Delta =$ min<sub>k</sub> $( \dot { \sigma } _ { k } ^ { 2 } - \sigma _ { k + 1 } ^ { 2 } )$ swept by scaling a base spectrum at fixed $n = 3 2 0 0$ Right group: sample size swept at fixed $\Delta { \dot { = } } 0 . 9 0 0$ . Mode error is the mean mode identification error; the product err $\cdot \Delta \cdot { \sqrt { n } }$ is near-constant across both groups, $\approx 2 . 4 8$
<table><tr><td>Fixed Scale  $\Delta$ </td><td> $n = 3 2 0 0$  Mode err</td><td>Fixed  $\Delta = 0 . 9 0 0$  n Mode err</td></tr><tr><td>1.00 0.900</td><td>0.0431</td><td>200 0.1979</td></tr><tr><td>0.60 0.648</td><td>0.0782</td><td>800 0.0956</td></tr><tr><td>0.35 0.417</td><td>0.1258</td><td>3200 0.0470</td></tr><tr><td>0.20 0.230</td><td>0.2458</td><td>12800 0.0236</td></tr></table>

Normalization as gauge-fixing. The seven schemes of Table 1 separate cleanly by residual group. Only whitening attains perfect mode alignment (1.000) with zero cross-seed gauge distance and drives the conditioning diagnostic max(cond $\Sigma _ { f }$ , cond $\Sigma _ { g } )$ to its theoretical floor $\sigma _ { 1 } ^ { 2 } / \sigma _ { d } ^ { 2 }$ ≈ 9.5 (Corollary $_ { 3 ) ; }$ every looser gauge remains stuck at alignment 0.53 to 0.65, its residual group leaving the modes rotationally entangled (Theorem 4). The gauge-invariant diagnostic cond $( \check { \Sigma } _ { f } \Sigma _ { g } ^ { \bullet } )$ cannot separate the schemes (Lemma 2). The two-sided nonnegative gauge incurs error 4.25, the predicted out-of-class cost of excluding negatively correlated modes (Theorem 5). Normalization is therefore gauge fixing, and whitening is the only scheme that removes the symmetry completely, exactly as Theorem 2 predicts.

The gap as the single unifying quantity. The spectral gap is the single quantity governing identification. With the finite-sample whitening estimator of Appendix E, we sweep the gap at fixed sample size and the sample size at fixed gap, eight configurations in all (Table $2 ;$ Figure 3 in Appendix F plots the collapse). The mode-identification error scales as $1 / \Delta$ at fixed sample size and $\arcsin ^ { - 1 / 2 }$ at fixed gap, where $\begin{array} { r } { \Delta = \operatorname* { m i n } _ { k } ( \sigma _ { k } ^ { 2 } - \sigma _ { k + 1 } ^ { 2 } ) } \end{array}$ , and the two sweeps collapse onto the single relation

$$
\mathrm { e r r } \cdot \Delta \cdot \sqrt { n } \approx 2 . 4 8\tag{19}
$$

with coefficient of variation 0.12: gap and sample size enter the identification error only through their product, the concrete face of Corollaries 3 and 6.

The sample-complexity rate law. In the linear regime the head is exactly a rank-d matrix sensing problem, and Theorem 7 predicts $\mathbb { E } \| \hat { \Theta } - \Theta ^ { \star } \| _ { F } ^ { 2 } \asymp \sigma _ { \varepsilon } ^ { 2 } d ( p + q ) / n$ . We sweep each of the three variables with the others held fixed and fit log-log slopes, the three sweeps shown in Figure 1. The measured slopes are −1.03 in $n , + 0 . 8 9$ in $d ,$ and +1.24 in $p + q ,$ , against the theoretical −1, +1, +1; the excess in the last slope is the expected Marchenko-Pastur finite-sample correction, and gradient descent on the factored pair matches the least-squares slope. These sweeps reject the loose general bound of Remark 6, which would predict slope $- 1 / 2 \colon$ estimation is governed by the sum of the marginal encoder complexities, and the $1 / n$ dependence is genuine.

![](images/1315e00629b7705425a497fcc93776f4c8cd82011c22f4c54b1a56a540566b06.jpg)

![](images/f723286da9a8fa41de9992d7afdc3951986070b2ffaa40f514553e391accc652.jpg)

![](images/e7b7a81a3dbe2da34db5445d7712958879e038bfd059eb4d0bcf5e32217c8146.jpg)  
Figure 1: The sample-complexity law of Theorem 7. Log-log sweeps of the Frobenius excess risk against sample size n (left), interaction rank d (center), and ambient dimension $p + q$ (right). Points are the measured risk; the dotted line is the theoretical rate $\sigma ^ { 2 } d ( 2 p - d ) / n$ , with the finite-n corrected form (dashed) in the right panel.

Table 3: The boolean equality kernel $F = \mathbf { 1 } [ u = v ]$ on $\{ \pm 1 \} ^ { m } , N = 2 ^ { m }$ . Rank d: embedding dimension at which the head first reaches $1 0 \%$ relative error. Early: parameters 2m + 1 and error of the early-interaction comparator.
<table><tr><td>m</td><td>N</td><td>Rank d  $d / N$ </td><td>Early params</td><td>Early error</td></tr><tr><td>3</td><td>8</td><td>8 1.00</td><td>7</td><td>0</td></tr><tr><td>4</td><td>16</td><td>15 0.94</td><td></td><td>9 0</td></tr><tr><td>5</td><td>32</td><td>29</td><td>0.91 11</td><td>0</td></tr><tr><td>6</td><td>64</td><td>58</td><td>0.91 13</td><td>0</td></tr><tr><td>7</td><td>128</td><td>116</td><td>0.91</td><td>15 0</td></tr><tr><td>8</td><td>256</td><td>231</td><td>0.90</td><td>17 0</td></tr></table>

Table 4: Narrowband periodic-Gaussian kernel of bandwidth h on the circle: interaction rank $d _ { \varepsilon }$ for $\varepsilon = 1 0 \%$ error. The rank grows as $\Theta ( 1 / h )$ (Remark 7).
<table><tr><td colspan="2">h  $1 / h$ </td><td> $d _ { \varepsilon }$ </td></tr><tr><td>0.200</td><td>5</td><td>3</td></tr><tr><td>0.100</td><td>10</td><td>4</td></tr><tr><td>0.050</td><td>20</td><td>8</td></tr><tr><td>0.025</td><td>40</td><td>15</td></tr><tr><td>0.0125</td><td>80</td><td>30</td></tr></table>

The usability criterion. A flat spectrum floors the head regardless of capacity. We test two flat and near-flat regimes, both tabulated in Tables 3 and 4. On the Boolean equality kernel $F = \mathbf { 1 } [ u = v ]$ on $\{ \pm 1 \} ^ { m }$ , which has a completely flat spectrum, a rank-d head incurs relative error exactly $1 - \bar { d } / N$ and requires $d \approx 0 . 9 \cdot 2 ^ { m }$ dimensions to reach 10% error, while an early-interaction threshold on $\langle u , v \rangle$ is exact with 2m + 1 parameters (Theorems 8 and 9). Smooth narrow-band kernels give the polynomial counterpart $d _ { \varepsilon } = \stackrel { \cdot } { \Theta } ( 1 / h )$ (Remark 7). A trained-model anchor on a block-boxcar kernel realizes the same floor (Table 10 and Figure 5, Appendix F).

## 6 CONCLUSION

This paper has examined a single architectural pattern, the multiplicative dual-encoder head, through a unifying lens: the interaction spectrum of the target. Through theoretical analysis and numerical experiments, we answer four central questions. (i) Approximation error decomposes into a truncation term intrinsic to the target and a realization term intrinsic to the encoders, with target smoothness controlling the decay rate. (ii) Sample complexity is governed by the sum of the two marginal encoder complexities, with the linear regime anchoring the minimax rate. (iii) Usability is determined by the decay rate itself: a flat spectrum imposes an error floor that no rank-d head can escape, while early-interaction models can, and the deciding quantity is measurable from data before either model is trained. (iv) Identifiability, the problem that the framework brings into focus, is gauge fixing: every normalization used in practice is a section of the gauge orbit; whitening pins the interaction modes up to permutation and sign; and the spectral gap simultaneously controls optimization conditioning, identifiability, and estimation.

## ETHICS STATEMENT

This work adheres to the principles outlined in the ICLR Code of Ethics.

## REFERENCES

Sanjeev Arora, Rong Ge, Ravindran Kannan, and Ankur Moitra. Computing a nonnegative matrix factorization–provably. In Proceedings of the forty-fourth annual ACM symposium on Theory of computing, pp. 145–162, 2012.

Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations, 2022.

André Barreto, Will Dabney, Rémi Munos, Jonathan J Hunt, Tom Schaul, Hado Van Hasselt, and David Silver. Successor features for transfer in reinforcement learning. Advances in neural information processing systems, 30, 2017.

Peter L Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal ofmachine learning research, 3(Nov):463–482, 2002.

Emmanuel J Candes and Yaniv Plan. Tight oracle inequalities for low-rank matrix recovery from a minimal number of noisy random measurements. IEEE Transactions on Information Theory, 57 (4):2342–2359, 2011.

David Donoho and Victoria Stodden. When does non-negative matrix factorization give a correct decomposition into parts? Advances in neural information processing systems, 16, 2003.

Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218, 1936.

Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. Whitening for selfsupervised representation learning. In International conference on machine learning, pp. 3015– 3024. PMLR, 2021.

Zhang-Wei Hong, Ge Yang, and Pulkit Agrawal. Bi-linear value networks for multi-goal reinforcement learning. In International Conference on Learning Representations, 2021.

Kwang-Sung Jun, Rebecca Willett, Stephen Wright, and Robert Nowak. Bilinear bandits with low-rank structure. In International Conference on Machine Learning, pp. 3163–3172. PMLR, 2019.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pp. 6769–6781, 2020.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Graham Little and John B Reade. Eigenvalues of analytic kernels. SIAMjournal on mathematical analysis, 15(1):133–136, 1984.

Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature machine intelligence, 3(3):218–229, 2021.

Andreas Maurer. A vector-contraction inequality for rademacher complexities. In International Conference on Algorithmic Learning Theory, pp. 3–17. Springer, 2016.

Sahand N. Negahban and Martin J. Wainwright. Estimation of (near) low-rank matrices with noise and high-dimensional scaling. In International Conference on Machine Learning, 2009. URL https://api.semanticscholar.org/CorpusID:1004801.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

John B Reade. Eigenvalues of positive definite kernels. SIAM Journal on Mathematical Analysis, 14 (1):152–157, 1983.

Steffen Rendle. Factorization machines. In 2010 IEEE International conference on data mining, pp. 995–1000. IEEE, 2010.

Angelika Rohde and A. Tsybakov. Estimation of high-dimensional low-rank matrices. Annals of Statistics, 39:887–930, 2009. URL https://api.semanticscholar.org/CorpusID: 88512332.

Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal value function approximators. In International conference on machine learning, pp. 1312–1320. PMLR, 2015.

Erhard Schmidt. Zur theorie der linearen und nichtlinearen integralgleichungen: I. teil: Entwicklung willkürlicher funktionen nach systemen vorgeschriebener. Mathematische Annalen, 63(4):433–476, 1907.

Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In International conference on machine learning, pp. 2071–2080. PMLR, 2016.

Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases. arXiv preprint arXiv:1412.6575, 2014.

Dmitry Yarotsky. Error bounds for approximations with deep relu networks. Neural networks, 94: 103–114, 2017.

Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction. In International conference on machine learning, pp. 12310– 12320. PMLR, 2021.

Zijian Zhao and Sen Li. The impacts of data privacy regulations on food-delivery platforms. Transportation Research Part C: Emerging Technologies, 181:105364, 2025.

Zijian Zhao and Sen Li. Discriminatory order assignment and payment-setting of on-demand food-delivery platforms: A multi-action and multi-agent reinforcement learning framework. Transportation Research Part E: Logistics and Transportation Review, 208:104653, 2026a.

Zijian Zhao and Sen Li. Triple-bert: Do we really need marl for order dispatch on ride-sharing platforms? In International Conference on Learning Representations, volume 2026, pp. 153780– 153808, 2026b.

Zijian Zhao, Tingwei Chen, Zhijie Cai, Xiaoyang Li, Hang Li, Qimei Chen, and Guangxu Zhu. Crossfi: A cross domain wi-fi sensing framework based on siamese network. IEEE Internet of Things Journal, 12(12):20138–20155, 2025.

## APPENDIX CONTENTS

A Derivations for the Unification Table 13   
B Proofs for Section 2 15   
C Proofs for Section 3 18   
D Proofs for Section 4 22   
E Finite-Sample Technical Lemmas 26   
E.1 Empirical whitening and mode identification . 26   
E.2 Approximate separability in the NMF route 27   
E.3 Coupling identification with estimation . 28   
F Experimental Details and Additional Results 28   
F.1 Controlled synthetic study . 29   
F.2 DeepONet 30   
F.3 CLIP . 33   
G Index of results 35

## A DERIVATIONS FOR THE UNIFICATION TABLE

This appendix spells out, for each of the ten families of Table 5, the identification that makes it an instance of the multiplicative dual-encoder head (1): the two input spaces, the encoder maps, the score, the reference measures that define the interaction spectrum, and the boundary cases where an additional output-level normalization acts after the inner product and therefore leaves the gauge untouched. Table 5 collects the ten families, each computing a real-valued output for a pair of inputs as the inner product of two separately learned encodings; reading across the rows makes visible what is shared, and the paragraphs below identify the interaction operator and its spectrum for each family, including the boundary cases.

CLIP. Inputs are an image I and a text $T ;$ the encoders are the image tower $f$ and the text tower $^ { g , }$ both $\ell _ { 2 } \cdot$ -normalized; the score is sim $( I , T ) = \langle f ( I ) , g ( T ) \rangle$ , rescaled by a temperature and fed into a softmax over the batch. The reference measures are the empirical distributions over images and over texts in the training batch. The temperature and the softmax act on the similarity matrix after the inner product, so they are output-level and do not affect the embedding-level gauge; the unit-norm constraints are embedding-level and leave exactly the residual rotation O of Theorem 4.

DeepONet. Inputs are an input function a (on a sensor grid) and a query point $y ;$ the encoders are the branch net b and the trunk net t; the score is $\mathcal { G } ( a ) ( \overline { { y } } ) \approx \langle b ( a ) , \dot { t } ( y ) \rangle$ (Lu et al., 2021). The reference measures are the distribution over input functions and the measure over query points. For a linear operator with kernel $\begin{array} { r } { \kappa = \sum _ { k } \sigma _ { k } \beta _ { k } \tilde { ( y ) } \gamma _ { k } ( x ) } \end{array}$ and a white input measure, the interaction spectrum coincides with the operator singular values and the trunk’s ideal basis with the left singular functions, which is the ground truth used in Appendix F.2. The boundary case is the scale imbalance between branch and trunk outputs, familiar in practice; it is the scale redundancy of the gauge group, diagnosed in Section 3. POD-DeepONet substitutes a precomputed orthogonal POD basis for the learned trunk (Lu et al., 2021), motivated by the same nonuniqueness; our treatment is complementary, characterizing when the learned basis becomes identifiable instead of replacing it (Appendix F.2).

Two-tower retrieval. Inputs are a query q and a document $d ;$ the encoders are two towers $\varphi ,$ ψ; the score is a relevance measure rel $( q , d ) = \langle \varphi ( q ) , \psi ( d ) \rangle$ ⟩ (Karpukhin et al., 2020). The reference measures are the query distribution and the document distribution, and the interaction spectrum is the relevance spectrum of the underlying ranker. The top-k nearest-neighbor search is output-level.

UVFA (universal value function approximators). Inputs are a state s and a goal $^ { g ; }$ the encoders are the state encoder $\phi ( s )$ , a learned map from states to features, and the goal encoder $\psi ( g )$ , a learned map from goals to features; the score is the goal-conditioned value $\bar { V ( s , g ) } = \langle \phi ( s ) , \bar { \psi ( g ) } \rangle$ (Schaul et al., 2015). The reference measures are the state-visitation distribution and the goal distribution. Goal-conditioned value functions are naturally low-rank when goals and states share a low-dimensional feature structure, which is exactly the case in which the representation (1) is sample-efficient (Section 4). The interaction spectrum also gives these methods a rank-selection rule they currently lack: the number of features in a successor-feature representation or the width of a value-network embedding is typically a tuned hyperparameter, whereas Corollary 5 ties it to the decay of the spectrum.

BVN (bilinear value networks). Inputs are a state-action pair $( s , a )$ and a state-goal pair $( s , g ) ;$ the encoders are the MLP $f ( s , a )$ encoding the state-action pair and the MLP $\varphi ( s , g )$ encoding the state-goal pair; the score is $\begin{array} { r } { Q ( s , a , g ) = \langle f ( s , a ) , \varphi ( s , g ) \rangle } \end{array}$ (Hong et al., 2021). The reference measures are the joint distributions over $( s , a )$ and $( s , g )$ . Note that the two arguments share the state s: the head’s theory applies to the function $Q$ of the pair of arguments, and the shared state is exactly the coupling that makes $Q$ non-separable in $( s , a , g )$ directly, which is why the two-input factorization matters.

Successor features. Inputs are a state s and a reward weight vector $w ;$ the encoder on the state side is the successor-feature map $\phi ( s )$ and the encoder on the weight side is the identity w (a singleton encoder class, not learned); the score is $V _ { w } ( s ) = \langle \phi ( s ) , w \rangle$ (Barreto et al., 2017). This is the head with $\mathcal { G } = \{ \mathrm { i d } \}$ , whose Rademacher complexity is zero, so the sample-complexity bounds of Section 4 specialize to the standard guarantee for successor features: only the state-side class matters.

DistMult and ComplEx. Inputs are a head entity h and a tail entity $t ;$ the encoders are entity embeddings $e _ { h } , e _ { t } \in \mathbb { R } ^ { d } ;$ the score is the bilinear form $\langle e _ { h } , r \circ e _ { t } \rangle$ of DistMult or the real part of the complex product $\boldsymbol { e } _ { h } ^ { \scriptscriptstyle \mathrm { ~ I ~ } }$ diag(r) e of ComplEx, where r is the relation embedding (Yang et al., 2014; Trouillon et al., 2016). Both are rank-d bilinear forms on the entity embeddings, i.e., heads with linear encoders and a fixed middle matrix; the interaction spectrum is governed by the Gram structure of the entity embeddings under the relation. The reference measure is the empirical distribution over observed triples.

Linear attention. Inputs are a query q and a key $k$ in a common space; the encoders are the feature map ϕ applied to each; the attention weight is $\langle \phi ( q ) , \phi ( k ) \rangle$ (Katharopoulos et al., 2020). The aggregation step $\begin{array} { r } { \sum _ { j } \phi ( k _ { j } ) v _ { j } ^ { \top } } \end{array}$ makes the compute separation of Proposition 1 exact: the pairwise inner products are never formed, giving the $O ( n ^ { 2 } ) \to O ( n )$ complexity. The reference measure is the distribution over keys (or key-value pairs). Finite-dimensional feature maps realize the head exactly; softmax attention with random-feature maps is the limiting case of an infinite-dimensional $\phi .$

Factorization machines. Inputs are a feature index i and a feature index $j ;$ the encoders are the embedding maps $v _ { i } , v _ { j } ;$ the pairwise term is $x _ { i } x _ { j } \langle v _ { i } , v _ { j } \rangle$ (Rendle, 2010). The reference measure is the empirical distribution over feature pairs weighted by the data. The interaction spectrum of the pairwise term is the spectrum of the Gram matrix of the embeddings, and the rank selection of Section 4 is the standard FM capacity control viewed through the interaction spectrum.

QK-attention matching. Inputs are a worker w and an order $o ;$ the encoders are a BERT-style encoder f and an MLP $^ { g ; }$ the score is the match value $\langle f ( w ) , g ( o ) \rangle$ forming the Q-matrix of a bipartite assignment problem, with a one-sided softplus and unit-norm constraint on the order encoder (Zhao & Li, 2026b;a; 2025). The reference measure is the empirical distribution over worker-order pairs. The matching layer that consumes the score matrix is output-level; the one-sided constraint is embedding-level but, as Remark 4 explains, it is a stability fix rather than an identifiability scheme. The two papers differ only in the downstream use of the same head, which is why they share the identifiability properties derived in this paper.

Positioning against prior work. The paragraphs above identify each family and its interaction operator; we now position the paper’s claims against the closest prior work, none of which treats the families as instances of a shared architecture or formulates the identifiability question. On the estimation side, the linear case of the head is rank-d matrix sensing, and the paper relies on the classical minimax theory, upper bounds via restricted-isometry-type analyses (Candes & Plan, 2011; Negahban & Wainwright, 2009) and matching lower bounds (Rohde & Tsybakov, 2009); bilinear bandits (Jun et al., 2019) study the online counterpart. The contribution relative to this line is a translation rather than a new bound: the estimation problem of a trained head is identified with matrix sensing, and the minimax rate is connected to the rank-selection rule of Section 4 through the decay of the interaction spectrum, which the matrix-sensing literature does not formulate because it treats the rank as given. On the identifiability side, the direct precursor of our remedy is self-supervised whitening: W-MSE applies hard whitening to the embeddings (Ermolov et al., 2021), and Barlow Twins and VICReg implement its soft relaxations, pushing the cross-correlation matrix toward the identity and regularizing variance and covariance (Zbontar et al., 2021; Bardes et al., 2022); it is also known that whitening reduces a general linear gauge to an orthogonal ambiguity, a polardecomposition observation. Against this background, our contribution is not the whitening operation itself, which we do not claim, but three things: the interpretation of normalization as a section of the gauge orbit and the classification of the residual subgroups across the normalizations used in practice (Table 6, Appendix C); the theorem that whitening together with a spectral gap pins the modes down to permutation and sign, strictly beyond the orthogonal ambiguity that decorrelation alone leaves (Theorem 2); and the statement that the interaction spectrum is the unique gauge-invariant content of a trained model, an interpretability claim with a theorem behind it rather than an empirical observation. The nonnegative variant of the bilinear form has its own classical identifiability theory, separability-type conditions for uniqueness up to permutation and scale (Donoho & Stodden, 2003; Arora et al., 2012), which the gauge analysis of Section 3 subsumes as a special case (Theorem 5).

Table 5: Ten families of methods as instances of the multiplicative dual-encoder head (1): for each family, the two inputs, the encoder maps, the output, and the downstream use.
<table><tr><td>Instance</td><td>Inputs</td><td>Encoders</td><td>Output</td><td>Downstream use</td></tr><tr><td>CLIP (Radford image, text et al., 2021)</td><td></td><td>ViT, text transformer normalized similarity</td><td></td><td>contrastive softmax over a batch</td></tr><tr><td>DeepONet (Lu function a, et al., 2021)</td><td>point y</td><td></td><td>branch net, trunk net 〈branch(a), trunk(y)〉 operator output</td><td> $\mathcal { G } ( a ) ( y )$ </td></tr><tr><td>Two-tower retrieval (Karpukhin et al., 2020)</td><td>query, document</td><td>two towers</td><td>relevance</td><td>top-k ANN search</td></tr><tr><td>et al., 2015)</td><td>UVFA (Schaul state s, goal</td><td> $g \_ { \mathbf { \psi } } \quad \phi ( s ) , \psi ( g )$ </td><td> $V ( s , g )$ </td><td>value function</td></tr><tr><td>BVN (Hong et al., 2021)</td><td> $( s , a ) , ( s , g )$ </td><td>two MLPs</td><td> $Q ( s , a , g )$ </td><td>multi-goal RL</td></tr><tr><td>Successor features (Barreto et al.,</td><td>state s, weights φ(s), w w</td><td></td><td> $V _ { w } ( s )$ </td><td>transfer</td></tr><tr><td>2017) DistMult / ComplEx (Yang et al., 2014;</td><td></td><td>head, tail entity entity embeddings</td><td>bilinear / complex product</td><td>link prediction</td></tr><tr><td>2016) Linear attention (Katharopou- los et al.,</td><td>query q, key k feature map φ</td><td></td><td> $\langle \phi ( q ) , \phi ( k ) \rangle$ </td><td> $O ( n ^ { 2 } ) \to O ( n )$  aggregation</td></tr><tr><td>2020) Factorization machines (Rendle, 2010)</td><td>feature i, feature j</td><td>embedding vi</td><td> $\langle v _ { i } , v _ { j } \rangle _ { x _ { i } x _ { j } }$ </td><td>pairwise interaction</td></tr><tr><td>QK-attention matching (Zhao &amp; Li, 2026b;a)</td><td>worker w, order BERT, MLP 0</td><td></td><td> $\langle f ( w ) , g ( o ) \rangle$ </td><td>Q-matrix for ILP matching</td></tr></table>

## B PROOFS FOR SECTION 2

We prove the results of Section 2 in order. Throughout, $\| \cdot \|$ denotes the $L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } )$ norm, and $\| \cdot \| _ { L ^ { 2 } ( \mu _ { U } ) } , \| \cdot \| _ { L ^ { 2 } ( \mu _ { V } ) }$ the marginal norms.

Assumption 1 (Encoder realizability). The encoder classes F and $\mathcal { G }$ containfunctions $\hat { f } , \hat { g }$ such that, for some nonnegative errors $\eta _ { f , k } , \eta _ { g , k }$

$$
\begin{array} { r } { \| \hat { f } _ { k } - \sigma _ { k } a _ { k } \| _ { L ^ { 2 } ( \mu _ { U } ) } \leq \eta _ { f , k } , \qquad \| \hat { g } _ { k } - b _ { k } \| _ { L ^ { 2 } ( \mu _ { V } ) } \leq \eta _ { g , k } , \qquad k = 1 , \ldots , d . } \end{array}\tag{20}
$$

Remark 1 (What the decomposition buys). The truncation term is intrinsic to the problem, the realization term to the model; when the encoder classes are dense in $L ^ { 2 }$ the bound approaches the lower bound ofCorollary 1, so the decomposition is tight up to the constant. Whether trainingfinds the coordinate alignment of (20) is the question ofidentifiability addressed in Section 3.

Proposition 1 (Compute separation). Evaluating a rank-d head on all $n \times m$ input pairs costs

$$
O \big ( ( n + m ) C _ { \mathrm { n e t } } + n m d \big ) ,\tag{21}
$$

where $C _ { \mathrm { n e t } }$ is the cost ofone encoderforward pass, whereas a model that processes each pairjointly costs $O ( n m C _ { \mathrm { n e t } } )$ . When the encoderforward pass dominates the inner product, the head is cheaper by a factor $C _ { \mathrm { n e t } } / d$

Remark 2 (Known complexity results as instances). Proposition 1 recovers, as special cases, the multiplicative-to-additive complexity argument ofthe QK-attention matching networks (Zhao & Li,

2026b), the $O ( n ^ { 2 } ) \to O ( n )$ aggregation of linear attention (Katharopoulos et al., 2020), where the attention weight is written as the inner product offeature maps, and the efficient branch-trunk evaluation ofDeepONet (Lu et al., 2021).

Remark 3 (Where low interaction rank comes from). Three structural sources make the truncation term small: exact low rank $\begin{array} { r } { ( F = \sum _ { i < r } \psi _ { j } ( u ) \chi _ { j } ( v ) } \end{array}$ implies i-rank $\left( F \right) ~ \leq ~ r )$ , tensor-product structure (for F in a tensor-product RKHS the spectrum is governed by the marginal kernels), and smoothness (Theorem 3 gives explicit rates). Conversely, a flat spectrum, developed in Section 4, means the truncation term does not decay and no choice ofd helps.

Lemma 1 (Rank equivalence). A function $S ~ \in ~ L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } )$ can be written as $S ( u , v ) \ =$ $\langle f ( u ) , g ( v ) \rangle$ ⟩ for some encoders $f : \mathcal { U } \to \mathbb { R } ^ { d }$ and $g ~ : \mathcal { V } \overset { \cdot } {  } ~ \mathbb { R } ^ { d }$ if and only if rank $( { \hat { T } } _ { S } ) \ \leq \ d .$ In particular, the parametric class $o f e q . \ ( 2 )$ coincides with $\mathcal { M } _ { d } = \bar { \{ F : \mathrm { i } \mathrm { - r a n k } ( \bar { F } ) \leq d \} }$ , and we use the two descriptions interchangeably.

Proof of Lemma 1 (Rank equivalence). The forward direction is immediate: if $\begin{array} { r } { S = \sum _ { k \le d } f _ { k } \otimes g _ { k } } \end{array}$ ， then the interaction operator is a sum of at most d rank-one operators,

$$
T _ { S } = \sum _ { k \le d } \langle \cdot , g _ { k } \rangle _ { L ^ { 2 } ( \mu _ { V } ) } f _ { k } ,\tag{22}
$$

so ran $\mathfrak { a } ( T _ { S } ) \ \leq \ d .$ . Conversely, if $\mathrm { r a n k } ( T _ { S } ) = r \le d _ { \ O }$ the Schmidt decomposition of S reads $\begin{array} { r } { S = \sum _ { k < r } \dot { \sigma } _ { k } a _ { k } \otimes b _ { k } } \end{array}$ with $\sigma _ { k } > 0 .$ , and the encoders $f = ( f _ { 1 } , \ldots , f _ { d } ) , g = ( g _ { 1 } , \ldots , g _ { d } )$ defined by $f _ { k } = \sigma _ { k } a _ { k }$ and $g _ { k } = b _ { k }$ for $k \leq r ,$ , and $f _ { k } = g _ { k } = 0$ for $k > r ,$ satisfy $S = \langle f , g \rangle$

Corollary 1 (Spectral truncation lower bound). For every rank-d head $( f , g )$

$$
\Vert F - \langle f , g \rangle \Vert _ { L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } ) } ^ { 2 } \ \geq \ \sum _ { k > d } \sigma _ { k } ^ { 2 } ,\tag{23}
$$

and the bound is attained in $L ^ { 2 }$ by the truncated Schmidt series $\begin{array} { r } { F _ { d } = \sum _ { k \le d } \sigma _ { k } a _ { k } \otimes b _ { k } } \end{array}$ , which is the unique best rank-d approximation whenever $\sigma _ { d } > \sigma _ { d + 1 }$

Proof of Corollary 1 (Spectral truncation lower bound). The map $S \mapsto T _ { S }$ is an isometric isomorphism between $L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } )$ and the Hilbert-Schmidt class $\mathrm { H S } ( \bar { L } ^ { 2 } ( \mu _ { V } ) , \bar { L ^ { 2 } } ( \mu _ { U } ) )$ ) equipped with the inner product $\langle \bar { T _ { S } } , T _ { S ^ { \prime } } \rangle = \mathrm { \bar { t r } } ( T _ { S } ^ { * } T _ { S ^ { \prime } } )$ ; indeed $\begin{array} { r } { \| T _ { S } \| _ { \mathrm { H S } } ^ { 2 } = \sum _ { k > 1 } \ddot { \sigma _ { k } ^ { 2 } } = \| S \| _ { L ^ { 2 } } ^ { 2 } . } \end{array}$ Under this isometry, the rank constraint of Lemma 1 becomes the operator rank constraint, and the Eckart-Young-Mirsky theorem for compact operators (see e.g. Gohberg and Krein) gives that the best rank-d approximation of $T _ { F }$ is the truncation to its leading d singular triples, with squared error $\textstyle \sum _ { k > d } \sigma _ { k } ^ { 2 } .$ Uniqueness: when $\sigma _ { d } > \sigma _ { d + 1 }$ , the leading d-dimensional singular subspace of $T _ { F }$ is unique, hence $\begin{array} { r } { F _ { d } = \sum _ { k \le d } \sigma _ { k } a _ { k } \otimes b _ { k } } \end{array}$ is the unique minimizer.

Proof of Theorem 1 (Approximation error decomposition). Fix realizations $\hat { f } \in \mathcal { F } , \hat { g } \in \mathcal { G }$ satisfying Assumption 1, and write $f _ { k } ^ { \star } \ = \ \sigma _ { k } a _ { k } , \ g _ { k } ^ { \star } \ = \ b _ { k }$ for the ideal modes, so that $F _ { d } =$ $\begin{array} { r } { \sum _ { k < d } f _ { k } ^ { \star } \otimes g _ { k } ^ { \star } = \langle f ^ { \star } , g ^ { \star } \rangle } \end{array}$ . Set $\varepsilon _ { k } = f _ { k } ^ { \star } - \hat { f } _ { k }$ and $\delta _ { k } = g _ { k } ^ { \star } - \hat { g } _ { k }$ , with $\| \varepsilon _ { k } \| _ { L ^ { 2 } ( \mu _ { U } ) } \leq \eta _ { f , k }$ and $\| \delta _ { k } \| _ { L ^ { 2 } ( \mu _ { V } ) } \leq \eta _ { g , k }$ . Split the error around $F _ { d }$

$$
\begin{array} { r } { \| F - \langle \hat { f } , \hat { g } \rangle \| \le \underbrace { \| F - F _ { d } \| } _ { = \sqrt { \mathcal E _ { \mathrm { t r u n c } } } } + R , \qquad R : = \| F _ { d } - \langle \hat { f } , \hat { g } \rangle \| , } \end{array}\tag{24}
$$

where $\begin{array} { r } { \mathcal { E } _ { \mathrm { t r u n c } } : = \sum _ { k > d } \sigma _ { k } ^ { 2 } } \end{array}$ by Corollary 1. For each coordinate, the product rule for tensor products gives

$$
f _ { k } ^ { \star } \otimes g _ { k } ^ { \star } - \hat { f } _ { k } \otimes \hat { g } _ { k } = \varepsilon _ { k } \otimes b _ { k } + \sigma _ { k } a _ { k } \otimes \delta _ { k } - \varepsilon _ { k } \otimes \delta _ { k } .\tag{25}
$$

Sum over $k \leq d$ and take norms. The first sum is diagonalized by the orthonormality of $\left\{ b _ { k } \right\}$ :

$$
\left\| \sum _ { k \leq d } \varepsilon _ { k } \otimes b _ { k } \right\| ^ { 2 } = \sum _ { k \leq d } \| \varepsilon _ { k } \| _ { L ^ { 2 } ( \mu \upsilon ) } ^ { 2 } = \sum _ { k \leq d } \eta _ { f , k } ^ { 2 } ,\tag{26}
$$

and likewise the orthonormality of $\{ a _ { k } \}$ diagonalizes the second sum:

$$
\left\| \sum _ { k \leq d } \sigma _ { k } a _ { k } \otimes \delta _ { k } \right\| ^ { 2 } = \sum _ { k \leq d } \sigma _ { k } ^ { 2 } \| \delta _ { k } \| _ { L ^ { 2 } ( \mu _ { V } ) } ^ { 2 } \leq \sum _ { k \leq d } \sigma _ { k } ^ { 2 } \eta _ { g , k } ^ { 2 } .\tag{27}
$$

This is the step that avoids a factor of d: bounding the sums coordinatewise and applying Cauchy-Schwarz over the d coordinates would pay a factor $d ,$ whereas the orthonormality of the ideal modes makes the two sums orthogonal. The third sum is bounded by Cauchy-Schwarz:

$$
\Big \| \sum _ { k \leq d } \varepsilon _ { k } \otimes \delta _ { k } \Big \| \leq \sum _ { k \leq d } \| \varepsilon _ { k } \| \left\| \delta _ { k } \right\| \leq \Big ( \sum _ { k \leq d } \eta _ { f , k } ^ { 2 } \Big ) ^ { 1 / 2 } \Big ( \sum _ { k \leq d } \eta _ { g , k } ^ { 2 } \Big ) ^ { 1 / 2 } .\tag{28}
$$

With $\begin{array} { r } { X = ( \sum _ { k } \eta _ { f , k } ^ { 2 } ) ^ { 1 / 2 } , Y = ( \sum _ { k } \sigma _ { k } ^ { 2 } \eta _ { g , k } ^ { 2 } ) ^ { 1 / 2 } , Z = ( \sum _ { k } \eta _ { g , k } ^ { 2 } ) ^ { 1 / 2 } } \end{array}$ , the three bounds give $R \leq$ $X + Y + X Z$ , hence by $( x + y + z ) ^ { 2 } \leq \bar { 3 } ( x ^ { 2 } + y ^ { 2 } + z ^ { 2 } )$

$$
R ^ { 2 } \leq 3 \sum _ { k \leq d } \eta _ { f , k } ^ { 2 } + 3 \sum _ { k \leq d } \sigma _ { k } ^ { 2 } \eta _ { g , k } ^ { 2 } + 3 \Bigl ( \sum _ { k \leq d } \eta _ { f , k } ^ { 2 } \Bigr ) \Bigl ( \sum _ { k \leq d } \eta _ { g , k } ^ { 2 } \Bigr ) .\tag{29}
$$

In the informative regime the mode errors are small: assuming $\begin{array} { r } { \sum _ { k \le d } \eta _ { g , k } ^ { 2 } \le 1 } \end{array}$ (the modes are realized with bounded total error, the only regime in which the bound is informative) and $\eta _ { g , k } \le 1 / 2$ for all k (so that $B _ { g } = \operatorname* { m a x } _ { k } \| \hat { g } _ { k } \| _ { L ^ { 2 } ( \mu _ { V } ) } \dot { \geq } \operatorname* { m i n } _ { k } ( \| b _ { k } \| - \| \delta _ { k } \| ) \geq 1 / 2 )$ , the cross term absorbs into the first term and $\begin{array} { r } { \sum _ { k } \eta _ { f , k } ^ { 2 } \le 4 B _ { g } ^ { 2 } \sum _ { k } \eta _ { f , k } ^ { 2 } . } \end{array}$ , giving

$$
R ^ { 2 } \leq 3 \sum _ { k \leq d } \sigma _ { k } ^ { 2 } \eta _ { g , k } ^ { 2 } + 2 4 B _ { g } ^ { 2 } \sum _ { k \leq d } \eta _ { f , k } ^ { 2 } .\tag{30}
$$

Finally, since $\begin{array} { r } { \| \boldsymbol { F } - \langle \hat { f } , \hat { g } \rangle \| \le \sqrt { \mathcal { E } _ { \mathrm { t r u n c } } } + R . } \end{array}$

$$
\| F - \langle \hat { f } , \hat { g } \rangle \| ^ { 2 } \leq 2 \mathcal { E } _ { \mathrm { t r u n c } } + 2 R ^ { 2 } \leq 2 \mathcal { E } _ { \mathrm { t r u n c } } + 6 \sum _ { k \leq d } \sigma _ { k } ^ { 2 } \eta _ { g , k } ^ { 2 } + 4 8 B _ { g } ^ { 2 } \sum _ { k \leq d } \eta _ { f , k } ^ { 2 } .\tag{31}
$$

As $2 \mathcal { E } _ { \mathrm { t r u n c } } \leq 4 8 \mathcal { E } _ { \mathrm { t r u n c } }$ trivially, the right-hand side is at most 48 times the sum of the truncation term and the realization term of the theorem. Taking the infimum over $f \in { \mathcal { F } } , g \in { \mathcal { G } }$ , the bound holds with $C \leq 4 8 \leq 1 4 4$ ; we do not optimize the constant, which is what the statement of the theorem records.

Theorem 3 (Smoothness controls the truncation term). Let $\mathcal { U } \subset \mathbb { R } ^ { m _ { U } }$ and $\mathcal { V } \subset \mathbb { R } ^ { m _ { V } }$ be bounded domains, and let $\mu _ { U } , \mu _ { V }$ have densities bounded above and below with respect to Lebesgue measure. $I f F$ is s-smooth in its second argument, $F \in H ^ { s } ( \mathcal { U } \times \mathcal { V } )$ , then

$$
\sigma _ { k } = { \cal O } \big ( k ^ { - s / m _ { V } } \big ) , \qquad \sum _ { k > d } \sigma _ { k } ^ { 2 } = { \cal O } \big ( d ^ { 1 - 2 s / m _ { V } } \big ) \quad f o r 2 s > m _ { V } .\tag{32}
$$

$I f F$ is real analytic on $\mathcal { U } \times \mathcal { V } ,$ , then $\sigma _ { k } = O ( e ^ { - c k } )$ for some $c > 0$ and $\begin{array} { r } { \sum _ { k > d } \sigma _ { k } ^ { 2 } = O ( e ^ { - 2 c d } ) } \end{array}$

Proof of Theorem 3 (Smoothness controls the truncation term). The squared singular values of $T _ { F }$ are the eigenvalues of the self-adjoint, positive semidefinite integral operator $T _ { F } ^ { * } T _ { F }$ on $L ^ { 2 } ( \mu _ { V } )$ with kernel

$$
K ( v , v ^ { \prime } ) = \int _ { \cal U } F ( u , v ) F ( u , v ^ { \prime } ) d \mu _ { U } ( u ) ,\tag{33}
$$

which is positive definite and inherits the smoothness of F in its second argument: differentiating under the integral, an s-smooth kernel on a bounded m<sub>V</sub>-dimensional domain has eigenvalues decaying as $O ( k ^ { - 2 s / m \nu } ) ;$ ; this is the classical eigenvalue decay for positive definite kernels (Reade, 1983), and the analytic case decays exponentially (Little & Reade, 1984). Taking square roots gives $\sigma _ { k } = { \cal O } ( k ^ { - s / m _ { V } } )$ , and summing the tail,

$$
\sum _ { k > d } \sigma _ { k } ^ { 2 } = O \Bigl ( \int _ { d } ^ { \infty } x ^ { - 2 s / m _ { V } } d x \Bigr ) = O \bigl ( d ^ { 1 - 2 s / m _ { V } } \bigr ) \quad \mathrm { f o r } 2 s > m _ { V } ,\tag{34}
$$

which is finite exactly when $2 s > m _ { V }$ . In the analytic case, $\sigma _ { k } = { \cal O } ( e ^ { - c k } )$ gives $\textstyle \sum _ { k > d } \sigma _ { k } ^ { 2 } =$ $O ( e ^ { - 2 c d } )$ by the geometric series.

Proposition 2 (Approximation separation). Let $F \in \mathcal { M } _ { d }$ with interaction modes $a _ { k } \in H ^ { s } ( \mathcal { U } )$ $b _ { k } \in H ^ { s } ( \mathcal { V } )$ . To approximate F within ε in $L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } )$ , a dual-encoder head whose encoders are ReLU networks requires

$$
N _ { \mathrm { d u a l } } = O \Bigl ( d \bigl ( \varepsilon ^ { - m _ { U } / s } + \varepsilon ^ { - m _ { V } / s } \bigr ) \Bigr )\tag{35}
$$

parameters, whereas any single-tower model that approximates F as a general $( m _ { U } + m _ { V } ) .$ dimensional $H ^ { s }$ function requires $\Omega ( \varepsilon ^ { - ( m _ { U } + m _ { V } ) / \bar { s } } )$ parameters in the worst case. Hence $N _ { \mathrm { d u a l } } / N _ { \mathrm { s i n g l e } } \to \bar { 0 } a s \varepsilon \to 0 .$

Table 6: Embedding-level normalizations as sections of the gauge orbit. Each row shows a constraint on the encoders, the part of the gauge it pins, the residual group that survives, and what the scheme identifies. See Proposition 3 and Theorems 4 to 2 for the statements, and Appendix C for proofs.
<table><tr><td>Normalization</td><td>Constraint</td><td>Pins</td><td>Residual gauge</td><td>Identifies to</td></tr><tr><td>None</td><td>none</td><td>nothing</td><td> $\mathrm { G L } _ { d }$ </td><td>nothing</td></tr><tr><td>One-sided L2 and softplus (Zhao &amp; Li,</td><td> $\| g ( v ) \| = 1 ,$   $g ( v ) \succeq 0$ </td><td>pointwise scale  $\operatorname { o f } g$ </td><td>not controlled</td><td>no guarantee</td></tr><tr><td>2026b) Per-coordinate</td><td> $\mathrm { d i a g } ( \Sigma _ { g } ) = I$ </td><td>coordinate scales</td><td> $d ^ { 2 } - d ,$  shears</td><td>no guarantee</td></tr><tr><td>standardization Cosine, two towers</td><td> $\| f \| = \| g \| = 1$ </td><td>angles</td><td>and rotations O</td><td>up to a rotation</td></tr><tr><td>(Radford et al., 2021) Two-sided nonnegative</td><td> $f \succeq 0 , g \succeq 0$ </td><td>signs, shears,</td><td> $\mathrm { P e r m \ltimes D _ { + } }$ </td><td>permutation and</td></tr><tr><td>Whitening</td><td> $\Sigma _ { g } = I , \Sigma _ { f } = \Lambda$ </td><td>rotations all continuous</td><td> $\mathrm { D } _ { \pm , \mathrm { o r d e r } } \mathrm { f i x e d }$ </td><td>scale permutation and</td></tr></table>

Proof of Proposition 2 (Approximation separation). Since $F \in \mathcal { M } _ { d }$ with modes $a _ { k } \in H ^ { s } ( \mathcal { U } )$ $b _ { k } \in H ^ { s } ( \mathcal { V } )$ , the truncation term of Theorem 1 vanishes for the rank-d head, and the entire approximation error is the realization term. Approximate each of the 2d modes by ReLU networks with the standard $H ^ { s }$ rates (Yarotsky, 2017): to reach mode error ε on a marginal space of dimension m requires $O ( \varepsilon ^ { - m / s } )$ parameters. Setting the mode errors to ε makes the realization term of Theorem 1 of order $O \dot { ( } \varepsilon ^ { 2 } )$ , so a total of

$$
N _ { \mathrm { d u a l } } = O \Bigl ( d \bigl ( \varepsilon ^ { - m _ { U } / s } + \varepsilon ^ { - m _ { V } / s } \bigr ) \Bigr )\tag{36}
$$

parameters suffice. For the lower bound, a single-tower model that treats $F$ as a general $( m _ { U } + m _ { V } ) .$ dimensional $H ^ { s }$ function must contend with the Kolmogorov width of the Sobolev class, whose $L ^ { 2 }$ approximation is bounded below by $\Omega ( \varepsilon ^ { - ( m _ { U } + m _ { V } ) / \bar { s } } )$ parameters in the worst case (classical Sobolev width results). Since $m _ { U } + m _ { V } > m _ { U } , m _ { V }$ , the exponent is additive in the dimensions, so $N _ { \mathrm { d u a l } } / N _ { \mathrm { s i n g l e } } = O \left( \varepsilon ^ { m _ { \mathrm { m i n } } / s } \right) \to 0 \mathrm { a s } \varepsilon \to 0$ , where $m _ { \operatorname* { m i n } } = \operatorname* { m i n } \{ m _ { U } , m _ { V } \}$

Proof of Proposition 1 (Compute separation). Counting operations: the head requires n forward passes of $f ,$ m forward passes of g, and nm inner products of dimension d, i.e., nmd multiply-adds, for a total of $O ( ( n + \bar { m } ) C _ { \mathrm { n e t } } + \bar { n } m d )$ . A joint model processes each of the nm pairs separately, costing $O ( n m C _ { \mathrm { n e t } } )$ . When $C _ { \mathrm { n e t } } \gg d ,$ the head is cheaper by a factor of order $C _ { \mathrm { n e t } } / d$ in the regime where $n m \gg n + m$

## C PROOFS FOR SECTION 3

We prove the results of Section 3 in order. Throughout, $\mathcal { L } ( f , g ) = \mathbb { E } _ { u , v } [ ( F ( u , v ) - \langle f ( u ) , g ( v ) \rangle ) ^ { 2 } ]$ is the population risk. Table 6 collects the normalization taxonomy moved from Section 3; Proposition 3 and Remark 4 complete the picture.

The gauge group $\mathrm { G L } _ { d }$ stratifies into per-coordinate scalings $( \mathrm { D } _ { + } , \ d$ dimensions), rotations (O, $d ( d - 1 ) { \bar { / } } 2 )$ , permutations and signs (discrete), and the remaining shears. The guarantees of Section 3 come from two-sided constraints, which act on both encoders and can pin the joint freedom of the pair; one-sided constraints cannot do this in general, and the schemes that use them are best understood as stability fixes. Normalization schemes are therefore compared on four measurable quantities: the residual gauge subgroup of Table 6, the Hessian condition number on the constrained manifold, related to the gap by Corollary 3, the expressivity cost of the constraint, for example the bounded range [−1, 1] of cosine normalization, and the output semantics, which determines which decoder and loss can be attached.

Remark 4 (One-sided constraints fix scale, not identifiability). The scheme used by the QK-attention matching networks applies a softplus and a pointwise unit-norm constraint to one encoder only (Zhao & Li, 2026b). It removes the global scaling redundancy ofthe representation, the parameter α in $\left( \alpha f \right) \left( g / \alpha \right)$ , whose growth caused training instabilities in early versions of those networks and it leaves the other encoder unconstrained. The constraint therefore provides stability without identifiability: the residual freedom of the pair depends on the geometry of the trained encodings and is not controlled by the constraint. The per-coordinate standardization ofProposition 3 has the same character, pinning the d coordinate scales of one encoder and nothing else.

Proposition 3 (Per-coordinate standardization pins the scales). Consider the constraint diag $\begin{array} { r } { { \bf \nabla } _ { \bf \cdot } ( \Sigma _ { g } ) = I , } \end{array}$ which standardizes each coordinate of one encoder to unit variance. It pins the positive diagonal subgroup $\mathrm { D } _ { + }$ ofdimension d; generically the residual gauge has dimension $d ^ { 2 } - { \bar { d } }$ and contains the shears and rotations that mix coordinates.

Lemma 2 (Gauge invariance). For every $A \in \mathrm { ~  ~ { ~ G L _ { d } ~ } ~ }$ and every pair of inputs $( u , v )$ $\langle A f ( u ) , A ^ { - \top } g ( v ) \rangle = \langle f ( u ) , g ( v ) \rangle$ , so $\Phi _ { A }$ leaves the represented function and hence the population risk unchanged. With the second moments $\Sigma _ { f } = \bar { \mathbb { E } _ { u } } [ f ( u ) f ( \bar { u } ) ^ { \top } ]$ and $\Sigma _ { g } = \mathbb E _ { v } [ g ( v ) \bar { g ( v ) ^ { \top } } ]$ the action transforms them by $\Sigma _ { A f } = A \Sigma _ { f } A ^ { \intercal }$ and $\Sigma _ { A ^ { - } ^ { \top } g } = A ^ { - \top } \Sigma _ { g } A ^ { - 1 }$ , so the eigenvalues of the product $\Sigma _ { f } \dot { \Sigma } _ { g }$ are gauge invariants.

Proof of Lemma 2 (Gauge invariance). The identity is the cancellation

$$
\langle A f ( u ) , A ^ { - \top } g ( v ) \rangle = f ( u ) ^ { \top } A ^ { \top } A ^ { - \top } g ( v ) = f ( u ) ^ { \top } g ( v ) = \langle f ( u ) , g ( v ) \rangle ,\tag{37}
$$

so the represented function and hence the risk are unchanged. For the second moments, substitution gives

$$
\Sigma _ { A f } = \mathbb { E } _ { u } \big [ A f ( u ) f ( u ) ^ { \top } A ^ { \top } \big ] = A \Sigma _ { f } A ^ { \top } , \qquad \Sigma _ { A ^ { - \top } g } = A ^ { - \top } \Sigma _ { g } A ^ { - 1 } ,\tag{38}
$$

and the product transforms by the similarity $\Sigma _ { A f } \Sigma _ { A ^ { - } ^ { \top } g } = A \Sigma _ { f } \Sigma _ { g } A ^ { - 1 }$ , whose eigenvalues are invariant. At a global minimizer, the represented function equals the rank-d truncation $F _ { d }$ of the target when $\sigma _ { d } > \sigma _ { d + 1 }$ (Corollary 1), and the coefficient computation of the proof of Theorem 2 below shows that the eigenvalues of $\Sigma _ { f } \Sigma _ { g }$ are then the squared singular values $\sigma _ { k } ^ { \bar { 2 } }$ , which is the claim of Remark 5.

Remark 5 (The invariant content of a trained head). The gauge-invariant content of a trained head is exactly thefunction it represents, summarized by its interaction spectrum; Lemma 2 adds a finite-dimensional shadow: at any global minimizer ofthe population risk, the eigenvalues of $\Sigma _ { f } \Sigma _ { g }$ are the squared interaction singular values. This quantity is well defined under every normalization, and it reappears in Corollary 2.

Proposition 4 (Ill-posedness from unfixed gauge). Let $( f ^ { \star } , g ^ { \star } )$ be a global minimizer ofthe risk and let S be its stabilizer, the set of $A \in \mathrm { G L } _ { d }$ with $\Phi _ { A } ( f ^ { \star } , g ^ { \star } ) = ( f ^ { \star } , g ^ { \star } )$ ). Then the Hessian ofthe risk at $( f ^ { \star } , g ^ { \star } )$ has at least

$$
\dim \ker \nabla ^ { 2 } { \mathcal { L } } ( f ^ { \star } , g ^ { \star } ) \geq d ^ { 2 } - \dim S\tag{39}
$$

zero directions, coming from the tangent space of the gauge orbit. Without normalization this is generically $d ^ { 2 }$ degenerate directions, so the second-order problem is rank-deficient and the optimization problem is ill-posed.

Proof of Proposition 4 (Ill-posedness from unfixed gauge). Let $( f ^ { \star } , g ^ { \star } )$ be a global minimizer and S its stabilizer. By Lemma 2, the risk is constant along the orbit $\tilde { \mathcal { O } } = \{ \bar { \Phi _ { A } } ( f ^ { \star } , g ^ { \star } ) : A \in \mathrm { G L } _ { d } \}$ so for every $X \in { \mathfrak { g l } } _ { d }$ the curve $A ( t ) = \exp ( t X )$ gives a path $\gamma _ { X } ( t ) = \Phi _ { A ( t ) } ( f ^ { \star } , g ^ { \star } )$ along which $\mathcal { L } \circ \gamma _ { X } \equiv \mathcal { L } ( f ^ { \star } , g ^ { \star } )$ . The velocity is

$$
\begin{array} { r } { \dot { \gamma } _ { X } ( 0 ) = ( X f ^ { \star } , - X ^ { \top } g ^ { \star } ) \in T _ { ( f ^ { \star } , g ^ { \star } ) } , } \end{array}\tag{40}
$$

and since the risk is constant along the path, its second derivative in this direction vanishes. Because $( f ^ { \star } , g ^ { \star } )$ is a global minimizer, $\nabla ^ { 2 } \bar { \mathcal { L } } \succeq \mathrm { \bar { 0 } }$ , and a semidefinite quadratic form vanishes in a direction if and only if that direction lies in the kernel, so $\dot { \gamma } _ { X } ( 0 ) \in$ ker $\bar { \nabla } ^ { 2 } \mathcal { L }$ for every X. The map $X \mapsto { \dot { \gamma } } _ { X } ( 0 )$ has kernel equal to the Lie algebra of the stabilizer $S _ { \ i }$ , hence its image, which lies entirely in the kernel of the Hessian, has dimension $d ^ { 2 } -$ dim S. This proves the bound. Without normalization the stabilizer at a generic minimizer is trivial, so the Hessian has at least $d ^ { 2 }$ zero directions.

Proof of Proposition 3 (Per-coordinate standardization pins the scales). The constraint $\mathrm { d i a g } ( \Sigma _ { g } ) = \bar { \cal I }$ is d scalar conditions that fix the d coordinate scales of $g \colon$ for the transformation $A ^ { - 1 } = \mathrm { d i a g } ( s _ { 1 } , \ldots , s _ { d } ) \in \mathrm { D } _ { + }$ acting on g, the condition forces $s _ { k } ^ { 2 } [ \bar { \Sigma _ { g } } ] _ { k k } = 1$ , pinning each $s _ { k }$ . Transformations A that mix coordinates generically change diag $( A ^ { - \top } \Sigma _ { g } ^ { \top } A ^ { - 1 } )$ unless they also rotate $\Sigma _ { g }$ to a new matrix with the same diagonal; the residual group is the set of A preserving the diagonal condition, which contains the orthogonal group conjugated by any diagonal scale and the shears that preserve the diagonal of $\Sigma _ { g }$ . Counting dimensions, the constraint fixes exactly the d-dimensional subgroup $\mathrm { D } _ { + }$ , and the generic residual dimension is $d ^ { 2 } - d .$

Theorem 4 (Cosine normalization leaves a full rotation). Consider two-sided unit normalization, $\tilde { f } = f / \| f \|$ and $\tilde { g } = g / \| g \|$ , with scores cos $\mathcal { L } ( f ( u ) , g ( v ) )$ , as in the contrastive models ofTable 5 (Radford et al., 2021); a temperature that rescales the logits changes no gauge. If the encodings are nondegenerate, in that the ranges off and g contain open subsets of $\mathbb { R } ^ { d } $ , the residual gauge group is exactly the orthogonal group,

$$
R = \left\{ A \in \operatorname { G L } _ { d } : A ^ { \top } A = I \right\} = 0 .\tag{41}
$$

For every rotation $R \in \mathrm { O }$ , the encodings $R f$ and Rg give pointwise identical cosine scores, and no transformation outside O does.

Corollary 2 (Contrastive dimensions are uninterpretable). Under the residual symmetry ofTheorem 4, the individual coordinates of a cosine-normalized embedding carry no intrinsic meaning: replacing $( f , g )$ by $( R f , R g )$ for any rotation R yields the same model and the same loss, and only rotationinvariant quantities are well defined, among them norms, pairwise inner products, and the eigenvalues of $\Sigma _ { f } \Sigma _ { g } ,$ which by Remark 5 are the squared interaction singular values at any optimum. This is a theorem-level explanation ofthe widely reported observation that contrastive embedding dimensions are uninterpretable; the remedy is the whitening scheme ofTheorem 2.

Proof of Theorem 4 (Cosine normalization leaves a full rotation). Let R be the residual group of the two-sided unit normalization, i.e., the set of $A \in \mathrm { G L } _ { d }$ such that, after renormalizing, the cosine scores are unchanged.

(⊇) For $R ~ \in ~ \mathrm { O } , ~ \| R f ( u ) \| ~ = ~ \| f ( u ) \|$ and $\| R g ( v ) \| ~ = ~ \| g ( v ) \|$ by orthogonality, and $\langle R f ( u ) , R g ( v ) \rangle = \langle f ( u ) , g ( v ) \rangle$ , hence

$$
\cos { \angle } ( R f ( u ) , R g ( v ) ) = { \frac { \langle R f ( u ) , R g ( v ) \rangle } { \| R f ( u ) \| \| R g ( v ) \| } } = { \frac { \langle f ( u ) , g ( v ) \rangle } { \| f ( u ) \| \| g ( v ) \| } }\tag{42}
$$

pointwise, so $\mathrm { \Omega } \mathrm { \subseteq } R .$

(⊆) Suppose A preserves every cosine between the encodings. Since the ranges of f and $g$ contain open subsets of $\mathbb { R } ^ { d }$ , the encodings realize a full-dimensional family of directions, so A preserves all angles between vectors. A linear map that preserves every angle is a similarity: by the classical characterization of conformal linear maps, $A = c R$ for some $c > 0$ and $R \in \mathrm { O }$ . The scalar c is absorbed by the unit renormalization of both encoders, so within the cosine score A acts identically to the rotation R, giving $R \subseteq \mathrm { O }$

Corollary 2 follows immediately. For any rotation $R \in \mathrm { O }$ the pair $( R f , R g )$ is a different encoder pair with pointwise identical cosine scores and identical loss, so the individual coordinates carry no intrinsic meaning. The rotation-invariant quantities are the norms, the inner products, and the eigenvalues of $\Sigma _ { f } \Sigma _ { g } ;$ by Remark 5 these are the squared interaction singular values at any optimum. Corollary 2 is the statement of these two sentences at the theorem level.

Theorem 5 (Two-sided nonnegativity gives NMF identifiability). Suppose both encoders are constrained to nonnegative outputs, $f ( u ) \succeq 0$ and $g ( v ) \succeq 0$ , as obtained by applying a softplus or exponential nonlinearity to raw encodings. On a finite set of inputs the model is the nonnegative matrixfactorization $M \overset { \cdot } { = } F G ^ { \top }$ ofa nonnegative matrix M. Ifthe encodings are nondegenerate, in that their ranges generate the positive orthant, the residual gauge group shrinks to the monomial group,

$$
R = \left\{ A \in { \mathrm { G L } } _ { d } : A \succeq 0 , A ^ { - \top } \succeq 0 \right\} = { \mathrm { P e r m } } \ltimes \mathrm { D } _ { + } ,\tag{43}
$$

the group of coordinate permutations composed with positive diagonal scalings. If the factorization is separable in the sense ofDonoho & Stodden (2003), with an anchor rowfor each latentfactor as in Arora et al. (2012), then it is essentially unique and the encoders are identified up to permutation and scaling.

Proof of Theorem 5 (Two-sided nonnegativity gives NMF identifiability). On a finite set of inputs, the nonnegative encoders form matrices $F \in \mathbb { R } _ { \geq 0 } ^ { n \times d } , G \in \mathbb { R } _ { \geq 0 } ^ { m \times d }$ , and the model is the nonnegative factorization $M = F G ^ { \top }$ . The residual group consists of $A \in \mathrm { G L } _ { d }$ such that $A f \succeq 0$ and $A ^ { - \top } g \succeq 0$ for all nonnegative encodings. Since the ranges generate the positive orthant, $A f \succeq 0$ for all nonnegative $f$ forces $A \succeq 0$ entrywise, and likewise $\bar { A } ^ { - \top } \succeq 0 . \mathrm { ~ A ~ }$ real matrix that is nonnegative and whose inverse transpose is also nonnegative is a generalized permutation matrix: its inverse transpose being nonnegative forces each row and column of A to have exactly one nonzero entry, and nonnegativity makes that entry positive. Hence $R \subseteq { \mathrm { P e r m } } \ltimes \mathrm { D } _ { + }$ , and the reverse inclusion is immediate, so $\mathbf { \bar { \boldsymbol { R } } } = \mathbf { \bar { P e r m } } \times \mathbf { D } _ { + }$ . Under separability in the sense of Donoho & Stodden (2003), each latent factor has an anchor row whose encoding is nonzero in exactly one coordinate, which identifies the extreme rays of the latent cone and hence the factor directions; the constructive algorithm of Arora et al. (2012) then yields the decomposition uniquely up to permutation and scaling. The degradation of this guarantee when separability holds only approximately is analyzed in Appendix E.

Proof of Theorem 2 (Whitening fixes the gauge up to permutation and sign). The proof has four steps.

(1) Every global minimizer represents the rank-d truncation. By Corollary 1 the minimal risk is $\scriptstyle \sum _ { k > d } \sigma _ { k } ^ { 2 }$ , attained exactly by the truncated series $\begin{array} { r } { F _ { d } = \sum _ { k < d } \sigma _ { k } a _ { k } \otimes b _ { k } } \end{array}$ , which is unique since $\sigma _ { d } > \sigma _ { d + 1 }$ . Hence any minimizer satisfies $\langle f ( u ) , g ( v ) \rangle = F _ { d } { \bar { ( u , v ) } }$ almost everywhere, i.e.,

$$
\sum _ { k = 1 } ^ { d } f _ { k } ( u ) g _ { k } ( v ) = \sum _ { k = 1 } ^ { d } \sigma _ { k } a _ { k } ( u ) b _ { k } ( v ) .\tag{44}
$$

(2) The span of the encoders is pinned. $\mathbf { A } s$ rank-d representations of the same tensor, the left factor spaces and right factor spaces coincide: span $\{ f _ { 1 } , \dots , f _ { d } \} = \operatorname { s p a n } \{ a _ { 1 } , \dots , a _ { d } \}$ and span $\{ g _ { 1 } , \dots , g _ { d } \} = \mathrm { s p a n } \{ b _ { 1 } , \dots , b _ { d } \}$ . So there exist coefficient matrices $B , \bar { C } \in \mathbb { R } ^ { d \times d }$ with $f = B a _ { [ d ] }$ and $g = C b _ { [ d ] }$ , where $\hat { a _ { [ d ] } } = ( a _ { 1 } , \ldots , a _ { d } ) ^ { \top }$ and likewise for $b .$ Substituting into the identity above and using the orthonormality of the modes gives

$$
B ^ { \top } C = \operatorname { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { d } ) = : S .\tag{45}
$$

(3) The whitening constraints force C to be a signed permutation. By the orthonormality of $\{ a _ { k } \}$

$$
\Sigma _ { f } = \mathbb { E } _ { u } [ f ( u ) f ( u ) ^ { \top } ] = B B ^ { \top } , \qquad \Sigma _ { g } = C C ^ { \top } .\tag{46}
$$

The constraint $\Sigma _ { g } = I$ gives $C C ^ { \top } = I , \operatorname { s o } C \in \mathrm { O } ;$ the constraint $\Sigma _ { f } = \Lambda$ diagonal gives $B B ^ { \top } = \Lambda$ From $B ^ { \top } C = S$ and $C \in \mathrm { O }$ we get $B ^ { \top } = S C ^ { - 1 } = S C ^ { \top }$ , so $B = C S$ , and therefore

$$
B B ^ { \top } = C S ^ { 2 } C ^ { \top } = \Lambda .\tag{47}
$$

Thus the orthogonal matrix $C$ diagonalizes the diagonal matrix $S ^ { 2 } = \mathrm { d i a g } ( \sigma _ { 1 } ^ { 2 } , . . . , \sigma _ { d } ^ { 2 } )$ . Since the $\sigma _ { k }$ are distinct, the eigenspaces of $S ^ { \less 2 }$ are the coordinate axes, and any orthogonal matrix diagonalizing $S ^ { 2 }$ maps each axis to itself up to sign, so $C \in \mathrm { P e r m } \ltimes \mathrm { D } _ { \pm }$ and $\Lambda = \overline { { { S } } } ^ { 2 } , \mathrm { i . e . , } \lambda _ { k } = \sigma _ { k } ^ { \overline { { { 2 } } } }$ with the ordering convention matching the decreasing $\sigma _ { k }$ . With the ordering fixed, $C = \mathrm { d i a g } ( \pm 1 , \dots , \pm 1 )$ and $B \bar { = } C S = \mathrm { d i a g } ( \pm \sigma _ { 1 } , \cdot \cdot \cdot , \pm \sigma _ { d } )$

(4) Substitute back. With the coordinate order fixed, $g _ { k } = \pm b _ { k }$ and $f _ { k } = \pm \sigma _ { k } a _ { k }$ , the signs being paired so that the product $f _ { k } g _ { k } = \sigma _ { k } a _ { k } b _ { k }$ is unique. This is exactly (10), and the encoders are identified up to permutation and sign.

Corollary 3 (The spectral gap controls conditioning and convergence). On the manifold defined by the constraints (9), every global minimizer is isolated up to the discrete group ofsigns and permutations, and the Hessian ofthe risk is nondegenerate in the tangent directions ofthe manifold, with smallest nonzero curvature bounded below by a constant multiple of the spectral gap $\Delta \stackrel { . } { = } \operatorname* { m i n } _ { 1 \leq k < d } ( \sigma _ { k } ^ { 2 } -$ $\sigma _ { k + 1 } ^ { 2 } )$ ; a projected gradient method that maintains the constraints therefore converges locally at a linear rate $1 - \mu / L ,$ , where $\mu$ is bounded below by a constant multiple of $\Delta$ and L is the local smoothness constant. Wider gaps make the problem both better conditioned andfaster to converge.

Proof of Corollary 3 (The spectral gap controls conditioning and convergence). Isolation: by Theorem 2, on the manifold defined by (9) every global minimizer is unique up to the discrete group of signs and the permutation, and the represented function $F _ { d }$ is a strict minimum of the risk by the strict inequality $\sigma _ { d } > \sigma _ { d + 1 } ;$ the Hessian of the risk is therefore nondegenerate in the tangent directions of the manifold, and its smallest nonzero curvature is positive. The quantitative bound: the Hessian curvature in the tangent directions is controlled by the second derivative of the risk as the encoders move within the constraint manifold, and by the Davis-Kahan sin Θ theorem the perturbation of a leading singular subspace under a perturbation of size ϵ is at most $\epsilon / \Delta$ , where $\begin{array} { r } { \bar { \Delta } = \operatorname* { m i n } _ { k } ( \sigma _ { k } ^ { 2 } - \sigma _ { k + 1 } ^ { 2 } ) } \end{array}$ . Inverting this stability relation, the curvature of the risk in directions that move the encoders off the ideal modes is bounded below by a constant multiple of $\Delta \colon$ a displacement of size t in function space changes the risk by at least $c \Delta { t ^ { 2 } }$ for the truncation-optimal function. A projected gradient method that maintains the constraints therefore converges locally at a linear rate $\bar { 1 } - \mu / L$ with $\mu \geq c \Delta$ and L the local smoothness constant, as stated.

Theorem 6 (Soft whitening recovers the hard solution). For $\rho > 0$ , consider the penalized risk

$$
\mathcal { L } _ { \rho } ( f , g ) = \mathcal { L } ( f , g ) + \rho \Big ( \big \| \Sigma _ { g } - I _ { d } \big \| _ { F } ^ { 2 } + \big \| \mathrm { o f f d i a g } ( \Sigma _ { f } ) \big \| _ { F } ^ { 2 } \Big ) .\tag{48}
$$

As $\rho  \infty ,$ , any convergent sequence of minimizers of $\mathcal { L } _ { \rho }$ converges to a solution satisfying the constraints ofTheorem $\bar { 2 }$ up to the ordering convention, and hence to the recovered modes (10). For finite $\rho ,$ the residual gauge directions that break the constraint $\Sigma _ { g } = I$ carry curvature at least $2 \rho c$ for a constant $c > 0$ that depends on the second moments, so conditioning improves monotonically with $\rho .$

Proof of Theorem 6 (Soft whitening recovers the hard solution). (i) Convergence of minimizers. The zero set of the penalty

$$
\rho \Big ( \| \Sigma _ { g } - I \| _ { F } ^ { 2 } + \| \operatorname { o f f d i a g } ( \Sigma _ { f } ) \| _ { F } ^ { 2 } \Big )\tag{49}
$$

is exactly the whitening constraint (9) up to the ordering convention. Since $\mathcal { L } _ { \rho } \geq \mathcal { L }$ , any limit point of a convergent sequence of minimizers satisfies the constraint (otherwise the penalty, and with it ${ \mathcal { L } } _ { \rho } ,$ would diverge) and is a global minimizer of $\mathcal { L }$ subject to the constraint, to which Theorem 2 applies.

(ii) Curvature of the penalty. Consider a gauge generator $X \in { \mathfrak { g l } } _ { d }$ acting on the $g \cdot$ -side as $g ( t ) =$ $ { \mathrm { i } }  { \mathrm { i } }  { \mathrm { i } }  { \mathrm { i } }  { \mathrm { i } }  { \mathrm { i } }  { \mathrm { i } }  { \mathrm { i } }  { \mathrm { i } }  { \mathrm { \Omega } }  { \mathrm { i } }  { \mathrm { g } }$ . The second moment evolves as $\bar { \Sigma _ { g ( t ) } } = \exp ( - t \bar { X ^ { \top } } ) \Sigma _ { g } \exp ( - t X )$ . At a constraintsatisfying point $\Sigma _ { g } = I ,$ , the second derivative of $\| \Sigma _ { g ( t ) } - I \| _ { F } ^ { 2 }$ at $t = 0$ is $2 \| X + X ^ { \top } \| _ { F } ^ { 2 }$ : expanding the exponential gives $\Sigma _ { g ( t ) } = I - t ( X ^ { \top } + X ) + t ^ { 2 } X ^ { \top } X + O ( t ^ { 3 } )$ , so the squared deviation has leading term $t ^ { 2 } \| X + X ^ { \top } \| _ { F } ^ { 2 }$ multiplied by 2. Directions that break $\Sigma _ { g } = I$ have nonzero symmetric part, hence carry penalty curvature at least $2 \rho c$ with $c > 0$ determined by the second moments through the contribution of ${ \mathrm { o f f d i a g } } ( \Sigma _ { f } ) ;$ directions with X antisymmetric are rotations, which preserve $\Sigma _ { g } = I$ and are exactly the directions whose residual freedom is fixed by the ordering convention. Conditioning therefore improves monotonically with $\rho ,$ as stated.

## D PROOFS FOR SECTION 4

We prove the results of Section 4. Throughout, $\mathcal { R } ( h ) = \mathbb { E } _ { u , v } [ ( F ( u , v ) - h ( u , v ) ) ^ { 2 } ]$ is the population risk and $\hat { \mathcal { R } }$ its empirical version over the n samples. Corollary 6 and Proposition 6 are proved in Appendix $\mathrm { E , }$ as indicated in the main text.

Remark 6 (The general bound is conservative in d). Specializing Corollary 4 to the linear class gives $\Re _ { n } ( { \mathcal { F } } ) ~ = ~ O ( { \sqrt { p d / n } } )$ and $\Re _ { n } ( { \mathcal G } ) \ = \ { \cal O } ( \sqrt { q d / n } )$ , hence an estimation error of order $B d ^ { 3 / 2 } \sqrt { ( p + q ) / n } ,$ a factor $\sqrt { d }$ worse than the minimax anchor (56). The gap is structural: Lemma 3 sums the d coordinates as independent classes and cannot exploit the low-rank geometry that the matrix analysis of Theorem $^ { 7 }$ uses. The general bound is therefore tight in its dependence on n and on the marginal dimensions, and conservative in the power of d; the linear case shows the achievable d-dependence is linear, and local Rademacher arguments close the gap. In the sequential setting, where $( u _ { t } , v _ { t } )$ are chosen adaptively to maximize an unknown score $u ^ { \top } \Theta ^ { \star } v ,$ , the same scaling appears as regret: bilinear bandits achieve $\tilde { O } ( d \sqrt { ( p + q ) T } )$ (Jun et al., 2019), again linear in d and in the marginal dimensions.

Remark 7 (The separation is not a discrete artifact). The same separation holdsfor smooth kernels. On the circle, the narrowband kernel $F _ { h } ( u , v ) = \rho _ { h } ( u - v )$ with a periodic Gaussian $\rho _ { h }$ ofbandwidth h has a Fourier-diagonal interaction operator with singular values of order $e ^ { - 2 \pi ^ { 2 } h ^ { 2 } \dot { \xi } ^ { 2 } }$ , so $d _ { \varepsilon } ( F _ { h } ) =$ $\Theta ( 1 / h )$ : the embedding dimension diverges as the kernel approaches the diagonal. Early interaction computes the difference $t = u - v$ exactly and approximates the one-dimensional smoothfunction $\rho _ { h }$ with $O ( \mathrm { p o l y } ( 1 / h ) )$ ) parameters (Yarotsky, 2017), a polynomial separation in the same direction. The strength ofthe separation is set by how the target depends jointly on its two arguments: through a comparison in the discrete case, through a difference in the smooth case, and not at all when the target is both high-rank and unstructured.

Lemma 3 (Rademacher collapse of the inner-product class). For the class (12) with per-coordinate classes $\mathcal { F } ^ { ( k ) } , \mathcal { G } ^ { ( k ) }$

$$
\Re _ { n } ( \mathcal { H } _ { d } ) \ \leq \ \sum _ { k = 1 } ^ { d } \Re _ { n } \big ( \mathcal { F } ^ { ( k ) } \cdot \mathcal { G } ^ { ( k ) } \big ) \ \leq \ 2 \sum _ { k = 1 } ^ { d } \Big ( B _ { g } \Re _ { n } \big ( \mathcal { F } ^ { ( k ) } \big ) + B _ { f } \Re _ { n } ( \mathcal { G } ^ { ( k ) } ) \Big ) ,\tag{50}
$$

so in particular $\Re _ { n } ( \mathcal { H } _ { d } ) \leq 2 d \big ( B _ { g } \mathfrak { R } _ { n } ( \mathcal { F } ) + B _ { f } \mathfrak { R } _ { n } ( \mathcal { G } ) \big )$ for shared per-coordinate classes: the complexity of the head is a sum of marginal complexities, not a complexity of the joint space.

Proof of Lemma 3 (Rademacher collapse of the inner-product class). Decompose the score as $\begin{array} { r } { \langle f ( u ) , g ( v ) \rangle = \sum _ { k = 1 } ^ { d } f _ { k } ( u ) g _ { k } ( v ) } \end{array}$ . Rademacher complexity is subadditive over sums of function classes, so

$$
\Re _ { n } ( \mathcal { H } _ { d } ) \leq \sum _ { k = 1 } ^ { d } \Re _ { n } \big ( \mathcal { F } ^ { ( k ) } \cdot \mathcal { G } ^ { ( k ) } \big ) ,\tag{51}
$$

where $\mathcal { F } ^ { ( k ) } \cdot \mathcal { G } ^ { ( k ) } = \{ ( u , v ) \mapsto f _ { k } ( u ) g _ { k } ( v ) \}$ is the pointwise product class. For each coordinate, the product map $( f _ { k } , g _ { k } ) \mapsto f _ { k } g _ { k }$ is Lipschitz in its first argument with constant $B _ { g }$ and in its second with constant $B _ { f }$ on the bounded range; the vector contraction inequality of Maurer (2016) then bounds the complexity of the product class by the sum of the two marginal complexities,

$$
\Re _ { n } \big ( \mathcal { F } ^ { ( k ) } \cdot \mathcal { G } ^ { ( k ) } \big ) \leq 2 \Big ( B _ { g } \Re _ { n } ( \mathcal { F } ^ { ( k ) } ) + B _ { f } \Re _ { n } ( \mathcal { G } ^ { ( k ) } ) \Big ) ,\tag{52}
$$

the factor 2 absorbing the contraction constants. Summing over the d coordinates and using $\Re _ { n } ( \mathcal { F } ) =$ max<sub>k</sub> $\Re _ { n } ( \mathcal { F } ^ { ( k ) } )$ gives the displayed bound. The key structural point is that no complexity of the joint space $\boldsymbol { u } \times \dot { \boldsymbol { \nu } }$ appears: the head is paid for by its two marginal classes.

Corollary 4 (Excess risk of empirical risk minimization). Let $\hat { F }$ be the empirical risk minimizer over $\mathcal { H } _ { d } .$ . With probability at least $1 - \delta$ over the sample, the excess risk decomposes into an approximation part, the truncation and realization terms of Theorem 1, and an estimation part through the class complexity:

$$
\| \hat { F } - F \| _ { L ^ { 2 } ( \mu _ { U } \otimes \mu _ { V } ) } ^ { 2 } \lesssim \sum _ { k > d } \sigma _ { k } ^ { 2 } + \mathcal { E } _ { \mathrm { r e a l } } + B d \big ( \mathfrak { R } _ { n } ( \mathcal { F } ) + \mathfrak { R } _ { n } ( \mathcal { G } ) \big ) + B ^ { 2 } \sqrt { \frac { \log ( 1 / \delta ) } { n } } ,\tag{53}
$$

where $\begin{array} { r } { \mathcal { E } _ { \mathrm { r e a l } } = C \sum _ { k \le d } ( \sigma _ { k } ^ { 2 } \eta _ { g , k } ^ { 2 } + B _ { g } ^ { 2 } \eta _ { f , k } ^ { 2 } ) } \end{array}$ is the realization term and $\lesssim$ hides absolute constants.

Proof of Corollary 4 (Excess risk of empirical risk minimization). The regression oracle decomposition: for any $h \in \mathcal { H } _ { d }$ , writing $\hat { F }$ for the empirical risk minimizer,

$$
\| F - \hat { F } \| _ { L ^ { 2 } } ^ { 2 } \leq \| F - h \| _ { L ^ { 2 } } ^ { 2 } + 2 \operatorname* { s u p } _ { h ^ { \prime } \in \mathcal { H } _ { d } } \big | \hat { \mathcal { R } } ( h ^ { \prime } ) - \mathcal { R } ( h ^ { \prime } ) \big | .\tag{54}
$$

The uniform deviation is controlled as follows. The squared loss $\ell ( y , h ) = ( y - h ) ^ { 2 }$ is $4 B { \cdot } \mathrm { I }$ ipschitz in h on $| y | , | h | \leq B$ , so by the Talagrand contraction inequality the Rademacher complexity of the loss class is at most $4 B \Re _ { n } ( \mathcal { H } _ { d } )$ (Bartlett & Mendelson, 2002). Symmetrization bounds the expected uniform deviation by twice the Rademacher complexity of the loss class, and McDiarmid’s inequality, with the bounded loss $| y - h | ^ { 2 } \leq 4 B ^ { 2 }$ , gives the concentration term:

$$
\operatorname* { s u p } _ { h ^ { \prime } \in \mathcal { H } _ { d } } \left. \hat { \mathcal { R } } ( h ^ { \prime } ) - \mathcal { R } ( h ^ { \prime } ) \right. \lesssim B \mathfrak { R } _ { n } ( \mathcal { H } _ { d } ) + B ^ { 2 } \sqrt { \frac { \log ( 1 / \delta ) } { n } }\tag{55}
$$

with probability at least $1 - \delta$ . Choosing $h = h ^ { \star }$ , the population optimum, makes $\| \boldsymbol { F } - \boldsymbol { h } ^ { \star } \| ^ { 2 }$ exactly the approximation term of Theorem 1, and Lemma 3 bounds $\begin{array} { r } { \mathfrak { R } _ { n } ( \mathcal { H } _ { d } ) \le 2 d ( \vec { B _ { g } } \mathfrak { R } _ { n } ( \mathcal { F } ) + } \end{array}$ $B _ { f } \mathfrak { R } _ { n } ( \mathcal { G } ) )$ ; absorbing the Lipschitz and contraction constants into the absolute constant gives (14). For parameterized encoder classes with $p _ { f }$ and $p _ { g }$ parameters under norm control, the standard Rademacher bound for parameterized classes gives $\Re _ { n } ( \mathcal { F } ) = O ( \sqrt { p _ { f } / n } )$ and $\Re _ { n } ( { \mathcal { G } } ) = O ( { \sqrt { p _ { g } / n } } )$ and the estimation term becomes $O ( B d \sqrt { ( p _ { f } + p _ { g } ) / n } )$

Theorem 7 (Linear heads are low-rank matrix sensing). Let $\Theta ^ { \star } \in \mathbb { R } ^ { p \times q }$ have rank d and $\| \Theta ^ { \star } \| _ { F } \leq B$ and suppose the sensing matrices $u _ { i } v _ { i } ^ { \top }$ are drawn from an isotropic sub-Gaussian family and the noise is sub-Gaussian with variance $\sigma _ { \varepsilon } ^ { 2 } .$ . The rank-constrained estimator attains

$$
\mathbb { E } \big \| \hat { \Theta } - \Theta ^ { \star } \big \| _ { F } ^ { 2 } \asymp \sigma _ { \varepsilon } ^ { 2 } \frac { d ( p + q ) } { n } ,\tag{56}
$$

and this rate is minimax optimal. Equivalently, E∥ $\begin{array} { r } { \hat { F } - F \| _ { L ^ { 2 } } ^ { 2 } \asymp \sigma _ { \varepsilon } ^ { 2 } d ( p + q ) / n . } \end{array}$

Proof of Theorem 7 (Linear heads are low-rank matrix sensing). The head computes $u ^ { \top } \Theta v$ with $\Theta = U V ^ { \top }$ , and the observations are $y _ { i } = \langle \Theta ^ { \star } , u _ { i } v _ { i } ^ { \top } \rangle _ { F } + \varepsilon _ { i }$ , the standard form of low-rank matrix sensing with sensing matrices $A _ { i } = u _ { i } v _ { i } ^ { \intercal }$ and rank constraint rank $( \Theta ) \leq d .$ . Upper bound: the rank-d manifold has $d ( p + \bar { q } - d )$ free parameters, and under restricted strong convexity of the sensing operator, which holds for isotropic sub-Gaussian sensing matrices, the constrained estimator satisfies $\mathbb { E } \| \hat { \Theta } - \Theta ^ { \star } \| _ { F } ^ { 2 } \lesssim \sigma _ { \varepsilon } ^ { 2 } d ( p + q ) / n$ ; this is the robust error bound of Candes & Plan (2011) together with the high-dimensional rate of Negahban & Wainwright (2009). Lower bound: a Fano argument over a packing of the rank-d manifold gives the matching minimax lower bound $\Omega ( \sigma _ { \varepsilon } ^ { 2 } d ( p + q ) / n )$ ) (Rohde & Tsybakov, 2009). The equivalence with the function-space rate follows from the isotropy of u and v: $\mathbb { E } _ { u , v } [ ( u ^ { \top } \Delta v ) ^ { 2 } ] \asymp \| \Delta \| _ { F } ^ { 2 }$ for sub-Gaussian isotropic arguments, so $\mathbb { E } \| \hat { F } - F \| _ { L ^ { 2 } } ^ { 2 } \asymp \sigma _ { \varepsilon } ^ { 2 } d ( p + q ) / n$ Corollary 5 (Optimal rank selection). If the interaction spectrum decays exponentially, $\sigma _ { k } ^ { 2 } =$ $O ( e ^ { - 2 c k } )$ , and the encoders are linear, the total error is minimized at

$$
d ^ { \star } \asymp \frac { 1 } { 2 c } \log \frac { n } { p + q } ,\tag{57}
$$

and the achieved error is $\| \hat { F } - F \| _ { L ^ { 2 } } ^ { 2 } \lesssim ( p + q ) \log ( n / ( p + q ) ) / n ,$ , a near-parametric rate in which fast spectral decay eliminates the curse ofdimensionality.

Proof of Corollary 5 (Optimal rank selection). With exponential decay and linear encoders the total error of Theorem 7 and the truncation term balance as

$$
\operatorname { e r r } ( d ) \ \lesssim \ e ^ { - 2 c d } + { \frac { d ( p + q ) } { n } } .\tag{58}
$$

Setting the two terms equal, $e ^ { - 2 c d ^ { \star } } \asymp d ^ { \star } ( p + q ) / n .$ , and solving gives $d ^ { \star } \asymp ( 1 / 2 c ) \log ( n / ( p + q ) )$ where the logarithmic factor in d is absorbed by the constant. Substituting back, $e ^ { - 2 c d ^ { \star } } \asymp ( p + q ) / n$ so the achieved error is

$$
\| \hat { \boldsymbol { F } } - \boldsymbol { F } \| _ { L ^ { 2 } } ^ { 2 } \lesssim \frac { ( p + q ) } { n } \Big ( 1 + \log \frac { n } { p + q } \Big ) \asymp \frac { ( p + q ) \log ( n / ( p + q ) ) } { n } ,\tag{59}
$$

the near-parametric rate stated in the corollary.

Theorem 8 (Flat spectrum forces an exponential embedding). Let $\mathcal { U } = \mathcal { V } = \{ \pm 1 \} ^ { m }$ with uniform measure and $N = 2 ^ { m }$ , and consider the equality function $F _ { \mathrm { e q } } ( u , v ) = \mathbf { 1 } [ u \stackrel { \cdot } { = } v ]$ . Its interaction operator is $\begin{array} { r } { T _ { F _ { \mathrm { e q } } } = \frac { 1 } { N } \mathrm { I d } } \end{array}$ , so all N singular values equal $1 / N$ and the interaction rank is N. Every rank-d head therefore satisfies

$$
\frac { \operatorname* { i n f } _ { f , g } \left\| F _ { \mathrm { e q } } - { \langle f , g \rangle } \right\| _ { L ^ { 2 } } ^ { 2 } } { \left\| F _ { \mathrm { e q } } \right\| _ { L ^ { 2 } } ^ { 2 } } = 1 - \frac { d } { N } ,\tag{60}
$$

so achieving relative error ε requires $d \geq ( 1 - \varepsilon ) 2 ^ { m }$ , an embedding dimension exponential in the input size.

Proof of Theorem 8 (Flat spectrum forces an exponential embedding). Compute the interaction operator of $F _ { \mathrm { e q } }$ on the uniform measure over $\{ \pm 1 \bar  \} ^ { m }$ with $N = 2 ^ { m }$ :

$$
( T _ { F _ { \mathrm { e q } } } h ) ( u ) = \sum _ { v \in \{ \pm 1 \} ^ { m } } \mathbf { 1 } [ u = v ] h ( v ) \frac { 1 } { N } = \frac { h ( u ) } { N } ,\tag{61}
$$

so $\begin{array} { r } { T _ { F _ { \mathrm { e q } } } = \frac { 1 } { N } } \end{array}$ Id and all N singular values equal $1 / N$ . Hence $\begin{array} { r } { \| F _ { \mathrm { e q } } \| _ { L ^ { 2 } } ^ { 2 } = \sum _ { k > 1 } \sigma _ { k } ^ { 2 } = N / N ^ { 2 } = 1 / N } \end{array}$ and $\begin{array} { r } { \sum _ { k > d } \overset { \cdot } { \sigma _ { k } ^ { 2 } } = ( N - d ) / N ^ { 2 } } \end{array}$ for $d < N$ . By Corollary 1, the relative squared error of the best rank-d head is

$$
\frac { \operatorname* { i n f } _ { f , g } \| F _ { \mathrm { e q } } - \langle f , g \rangle \| _ { L ^ { 2 } } ^ { 2 } } { \| F _ { \mathrm { e q } } \| _ { L ^ { 2 } } ^ { 2 } } = \frac { ( N - d ) / N ^ { 2 } } { 1 / N } = 1 - \frac { d } { N } ,\tag{62}
$$

which is at most ε only if $d \geq ( 1 - \varepsilon ) N = ( 1 - \varepsilon ) 2 ^ { m }$

Theorem 9 (Early interaction escapes). For the equality function $F _ { \mathrm { e q } }$ on $\{ \pm 1 \} ^ { m } \times \{ \pm 1 \} ^ { m }$ , an early-interaction predictor with $O ( m )$ parameters represents $F _ { \mathrm { e q } }$ exactly, with zero error. Together with Theorem 8 this is an exponential separation: $O ( m )$ parameters suffice under early interaction, while any late-interaction head needs $\Omega \bar { ( } 2 ^ { m } )$ embedding dimensions to reach constant relative error.

Proof of Theorem 9 (Early interaction escapes). The construction uses the identity $u _ { i } v _ { i } = 1$ if and only if $u _ { i } = v _ { i }$ , so that

$$
\langle u , v \rangle = \sum _ { i = 1 } ^ { m } u _ { i } v _ { i } = m - 2 \mathrm { H a m } ( u , v ) ,\tag{63}
$$

and $\langle u , v \rangle = m$ if and only i ${ \mathrm { ~ f ~ } } u = v$ , with the next possible value $m - 2$ . Hence $F _ { \mathrm { e q } } ( u , v ) =$ $\mathbf { 1 } [ \langle u , v \rangle ~ \geq ~ m - 1 ]$ . A cross-attention layer with identity projections computes the query-key inner product $\langle u , v \rangle$ with $O ( m )$ parameters, and a threshold unit $\phi ( s ) = \mathrm { R e } \hat { \mathrm { L U } } ( s - ( \hat { m _ { \alpha } } - \bar { 1 } ) ) \stackrel { - } { - }$ ${ \mathrm { R e L U } } ( s - ( m + 1 ) )$ evaluates to 1 exactly on $s = m$ and to 0 elsewhere on the discrete range, using $O ( 1 )$ parameters. A joint MLP computes the same quantity bit by bit: each product $u _ { i } v _ { i } =$ $( ( u _ { i } + v _ { i } ) ^ { 2 } - 2 ) / 2$ costs $\bar { O ( 1 ) }$ ReLU units, the sum costs $\bar { O } ( m )$ additions, and the same threshold completes the network, again $O ( m )$ parameters in total. A cross-encoder, being a strict superset of cross-attention, inherits the construction. Together with Theorem $^ { 8 , }$ this is the exponential separation $O ( m )$ versus $\Omega ( 2 ^ { m } )$ ).

Theorem 10 (Usability criterion). Given a problem $( F , \mu _ { U } , \mu _ { V } )$ , a target relative error ε, and a sample budget n: if the effective interaction rank $d _ { \varepsilon } ( F )$ is small, a late-interaction head is appropriate, since its approximation and estimation errors are then small and it enjoys the compute and statistical separations ofProposition 1 and Lemma 3. Ifthe spectrum isflat, $\dot { d } _ { \varepsilon } ( F ) = \bar { \Omega ( } N )$ the truncationfloor ofCorollary 1 locks every rank-d head to relative error at least ε regardless of capacity, samples, or training, while early-interaction models reach ε polynomially when the target admits a low-dimensional joint structure. The deciding quantity, the spectral decay rate, is estimable from data by a weighted singular value decomposition ofthe empirical kernel matrix, so the criterion is actionable before training either model.

Proof of Theorem 10 (Usability criterion). First case: if $d _ { \varepsilon } ( F )$ is small, take $d \gtrsim d _ { \varepsilon } ( F )$ . The approximation error of Theorem 1 is at most $\varepsilon \| F \| ^ { 2 }$ plus the realization term, and the estimation error of Corollary 4 scales as $d ( \mathfrak { R } _ { n } ( \mathcal { F } ) + \mathfrak { R } _ { n } ( \mathcal { G } ) )$ , small in d; the compute separation of Proposition 1 and the marginal-complexity collapse of Lemma 3 apply without further conditions. Second case: if the spectrum is flat with $\grave { d _ { \varepsilon } ( F ) } = \grave { \Omega } ( N )$ , then for every $d < d _ { \varepsilon } ( F )$ the truncation floor of Corollary 1, in the explicit form of Theorem 8, locks the relative error of every rank-d head above ε regardless of capacity, samples, or training, while the constructive bounds of Theorem 9 and Remark 7 show that early-interaction models reach ε with a polynomial parameter budget when the target depends on its arguments through a low-dimensional joint statistic. Actionability: the empirical kernel matrix $\hat { M } _ { i j } = y ( u _ { i } , v _ { j } )$ on a grid of input pairs, weighted by the reference measures, has a weighted singular value decomposition whose values estimate $\left\{ \sigma _ { k } \right\}$ ; a log-linear fit of the decay of $\hat { \sigma } _ { k }$ against k (exponential) or log k (polynomial) decides the decay type and reads off $d _ { \varepsilon }$ before either model is trained.

## E FINITE-SAMPLE TECHNICAL LEMMAS

This appendix collects the finite-sample lemmas used in the main text. Section A makes the whitening identification of Theorem 2 quantitative: concentration of empirical second moments (Lemma 4), stability of the whitening map (Lemma 5), and a Davis-Kahan propagation step (Theorem 11) deliver the sample complexity of Corollary 6. Section B quantifies the degradation of the NMF route when separability holds only approximately (Theorem 12). Section C couples identification with estimation in one ledger (Proposition 5), which is the proof of Proposition 6. Throughout, ∥ · ∥ on matrices is the spectral norm and $\| \cdot \| _ { F }$ the Frobenius norm.

## E.1 EMPIRICAL WHITENING AND MODE IDENTIFICATION

Lemma 4 (Matrix Bernstein concentration of empirical moments). Let $\{ u _ { i } \} _ { i = 1 } ^ { n }$ be drawn i.i.d. from µ<sub>U</sub> and let f have bounded outputs, $\| f ( u ) \| \leq B _ { f }$ . The empirical second moment $\begin{array} { r } { \hat { \Sigma } _ { f } = } \end{array}$ $\begin{array} { r } { { \frac { 1 } { n } } \sum _ { i } f ( u _ { i } ) f ( u _ { i } ) ^ { \top } } \end{array}$ satisfies,for every $t > 0$

$$
\operatorname* { P r } \big [ \| \hat { \Sigma } _ { f } - \Sigma _ { f } \| \ge t \big ] \le 2 d \exp \Big ( \frac { - n t ^ { 2 } / 2 } { \nu _ { f } + B _ { f } ^ { 2 } t / 3 } \Big ) , \qquad \nu _ { f } : = B _ { f } ^ { 2 } \| \Sigma _ { f } \| ,\tag{64}
$$

and hence, with probability at least $1 - \delta ,$

$$
\| \hat { \boldsymbol { \Sigma } } _ { f } - \boldsymbol { \Sigma } _ { f } \| \leq \sqrt { \frac { 2 \nu _ { f } \log ( 2 d / \delta ) } { n } } + \frac { 2 B _ { f } ^ { 2 } \log ( 2 d / \delta ) } { 3 n } = : r _ { f } ( n , \delta ) ,\tag{65}
$$

which is $O \big ( B _ { f } ^ { 2 } \sqrt { \log ( d / \delta ) / n } \big )$ for large n. The symmetric statement holds $f o r g$ with radius $r _ { g } ( n , \delta )$

Proof. The matrices $X _ { i } ~ = ~ f ( u _ { i } ) f ( u _ { i } ) ^ { \top } ~ - ~ \Sigma _ { f }$ are independent, zero mean, and symmetric, with $\| X _ { i } \| ~ \le ~ \| f ( u _ { i } ) f ( u _ { i } ) ^ { \top } \| ~ + ~ \| \Sigma _ { f } \| ~ \le ~ 2 B _ { f } ^ { \bar { 2 } }$ . For the variance parameter, since $\mathbb { E } [ X _ { i } ^ { 2 } ] ~ \preceq$ $\mathbb { E } \Vert f ( u _ { i } ) \Vert ^ { 2 } f ( u _ { i } ) f ( u _ { i } ) ^ { \top } \preceq B _ { f } ^ { 2 } \Sigma _ { f }$ , we have $\begin{array} { r } { \big \| \sum _ { i } \bar { \mathbb { E } } [ X _ { i } ^ { 2 } ] \big \| \leq n B _ { f } ^ { 2 } \| \Sigma _ { f } \| = n \nu _ { f } } \end{array}$ . The matrix Bernstein inequality (Tropp, An Introduction to Matrix Concentration Inequalities) applied to $\textstyle \sum _ { i } X _ { i }$ gives the tail bound, and solving the tail for t at the level δ gives the displayed high-probability bound.

Lemma 5 (Stability of the whitening map). Let $\Sigma _ { g } \succeq \gamma I f o r \gamma > 0$ , and suppose $\begin{array} { r } { \| \hat { \Sigma } _ { g } - \Sigma _ { g } \| \le } \end{array}$ $r _ { g } \le \gamma / 2$ . Then

$$
\Big \| \hat { \Sigma } _ { g } ^ { - 1 / 2 } - \Sigma _ { g } ^ { - 1 / 2 } \Big \| \leq \frac { C _ { 0 } r _ { g } } { \gamma ^ { 3 / 2 } } ,\tag{66}
$$

for an absolute constant $C _ { 0 } .$

Proof. The map $A \mapsto A ^ { - 1 / 2 }$ is Fréchet differentiable on the cone $A \succeq \gamma I$ with derivative bounded by $\scriptstyle { \frac { 1 } { 2 } } \gamma ^ { - 3 / 2 } \displaystyle ;$ : using the integral representation $\begin{array} { r } { A ^ { - 1 / 2 } = \frac { 1 } { \pi } \int _ { 0 } ^ { \infty } ( A + s I ) ^ { - 1 } \dot { s } ^ { - 1 / 2 } } \end{array}$ ds, the derivative at A applied to H is $\begin{array} { r } { - \frac { 1 } { \pi } \int _ { 0 } ^ { \infty } ( A + s I ) ^ { - 1 } H ( A + s I ) ^ { - 1 } s ^ { - 1 / \tilde { 2 } } d s } \end{array}$ , whose norm is at most $\begin{array} { r } { \| H \| \cdot \frac { 1 } { \pi } \int _ { 0 } ^ { \infty } ( \gamma + } \end{array}$ $s ) ^ { - 2 } s ^ { - 1 / 2 } d s = \| \bar { H } \| { \frac { 1 } { 2 } } \gamma ^ { - 3 / 2 }$ . The condition $r _ { g } \le \gamma / 2$ keeps $\hat { \Sigma } _ { g } \succeq \frac { \gamma } { 2 } I$ by Weyl’s inequality, so the mean value theorem applies along the segment and gives the bound with $C _ { 0 }$ absorbing the integration constant. □

Theorem 11 (Finite-sample mode identification). Under the assumptions ofTheorem 2, suppose the encoders represent the population optimum exactly, $\langle f , g \rangle = F _ { d } ,$ and whitening is performed on the empirical moments of a sample of size n, with encoder outputs bounded by B. Let $r = \operatorname* { m a x } ( r _ { f } , r _ { g } )$ be the moment radii ofLemma 4, let $\begin{array} { r } { \Delta = \operatorname* { m i n } _ { 1 \leq k < d } ( \sigma _ { k } ^ { 2 } - \sigma _ { k + 1 } ^ { 2 } ) } \end{array}$ be the spectral gap, and assume $r \le c _ { 1 } \operatorname* { m i n } ( \gamma , \Delta )$ for a small absolute constant $c _ { 1 }$ . Then with probability at least $1 - \delta ,$ , the empirically whitened encoders $( \hat { f } , \hat { g } )$ recover the modes up to

$$
\left| \sin \angle ( \hat { g } _ { k } , b _ { k } ) \right| \le \frac { C r } { \Delta } , \qquad \left\| \hat { g } _ { k } - ( \pm b _ { k } ) \right\| \le \frac { C ^ { \prime } r } { \Delta } , \qquad \left| \hat { \Lambda } _ { k k } - \sigma _ { k } ^ { 2 } \right| \le C ^ { \prime \prime } r ,\tag{67}
$$

where $\hat { \Lambda } _ { k k }$ is the estimated k-th interaction strength. In particular the identification error is $O \big ( B ^ { 2 } \sqrt { \log ( d / \delta ) / n } / \Delta \big )$ , vanishing as $n ^ { - 1 / 2 }$ and amplified by the inverse spectral gap.

Proof. (1) Empirical whitening followed by an eigendecomposition of the whitened second moment is equivalent to a singular value decomposition of the empirical interaction operator. Since the encoders represent $\begin{array} { r } { F _ { d } = \sum _ { k < d } \sigma _ { k } a _ { k } \otimes \bar { b _ { k } } } \end{array}$ exactly, the population singular subspaces are spanned by $\{ b _ { k } \}$ on the g-side and $\{ a _ { k } \}$ on the f-side. (2) The perturbation of the empirical operator relative to its population version is bounded by ${ \cal C } _ { 3 } r / \gamma ^ { 3 / 2 } \le { \cal C } _ { 4 } r \colon$ the moment error $\| \hat { \Sigma } _ { f } - \Sigma _ { f } \| \le r _ { f }$ from Lemma 4, the whitening error from Lemma 5, and the remaining factors are absorbed into the constant, which for brevity absorbs the γ dependence as well. (3) The k-th empirical mode is the k-th singular vector of the perturbed operator. By the Davis-Kahan sin Θ theorem, the perturbation of a singular vector is bounded by the operator perturbation divided by the gap between the corresponding singular value squared and its neighbors, which is at least ∆:

$$
\sin \angle ( \hat { g } _ { k } , b _ { k } ) \leq \frac { \| E \| } { \operatorname* { m i n } ( \sigma _ { k - 1 } ^ { 2 } - \sigma _ { k } ^ { 2 } , \ \sigma _ { k } ^ { 2 } - \sigma _ { k + 1 } ^ { 2 } ) } \leq \frac { C r } { \Delta } .\tag{68}
$$

Choosing the sign so that the inner product is nonnegative, $\lVert \hat { g } _ { k } - ( \pm b _ { k } ) \rVert \leq \sqrt { 2 }$ sin $\angle \le C ^ { \prime } r / \Delta$ . (4) For the strengths, Weyl’s inequality gives $| \hat { \Lambda } _ { k k } - \sigma _ { k } ^ { 2 } | \leq \| E \| \leq C ^ { \prime \prime } r$ □

Corollary 6 (Sample complexity of whitening). Under the assumptions ofTheorem 11, whitening the empirical second moments of a sample of size n identifies the interaction modes to accuracy ε with probability at least $1 - \delta$ provided

$$
n \gtrsim \frac { B ^ { 4 } \log ( d / \delta ) } { \Delta ^ { 2 } \varepsilon ^ { 2 } } ,\tag{69}
$$

where $\begin{array} { r } { \Delta = \operatorname* { m i n } _ { 1 \leq k < d } ( \sigma _ { k } ^ { 2 } - \sigma _ { k + 1 } ^ { 2 } ) } \end{array}$ is the spectral gap.

Proof of Corollary 6 (Sample complexity of whitening). Set ${ \mathrm { ~ \textit ~ { ~ C r ~ } ~ } } / \Delta \mathrm { ~  ~ { ~ \leq ~ \varepsilon ~ } ~ }$ with $r =$ $O \big ( B ^ { 2 } \sqrt { \log ( d / \delta ) / n } \big )$ from Lemma 4 and solve for n:

$$
n \gtrsim \frac { B ^ { 4 } \log ( d / \delta ) } { \Delta ^ { 2 } \varepsilon ^ { 2 } } .\tag{70}
$$

The sample budget scales as $1 / \Delta ^ { 2 }$ in the gap and $1 / \varepsilon ^ { 2 }$ in the target accuracy, matching (69).

Corollary 7 (Block stability under small gaps). When a pair of singular values nearly coincides, $\Delta _ { k } = \sigma _ { k } ^ { \bar { 2 } } - \sigma _ { k + 1 } ^ { 2 } \to 0$ , the single-mode bound of Theorem 11 diverges: individual coordinates become unidentifiable. The corresponding eigenspace remains stable: treating $\{ k , k + 1 \}$ as a block, the projection error ofthe block is at most $O ( r / \tilde { \Delta } )$ , where $\tilde { \Delta }$ is the gap between the block and the rest of the spectrum. Under repeated singular values the identification therefore degrades to uniqueness up to rotation within the block, the quantitative version of the degenerate-case remark of Section 3.

## E.2 APPROXIMATE SEPARABILITY IN THE NMF ROUTE

Definition 3 (α-approximate separability). A nonnegativefactor matrix $F \in \mathbb { R } _ { \geq 0 } ^ { n \times d }$ with rows $f ( u _ { i } )$ is α-separable if for every latent factor k there is a row i(k) with

$$
F _ { i ( k ) , k } \geq 1 , \qquad \sum _ { \ell \neq k } F _ { i ( k ) , \ell } \leq \alpha ,\tag{71}
$$

i.e., a near-anchor row dominated by coordinate k with residual mass at most α. The case $\alpha = 0$ is exact separability (Donoho & Stodden, 2003; Arora et al., 2012).

Theorem 12 (Degradation of approximately separable NMF). Let $M = F G ^ { \top }$ be a population nonnegative rank-dfactorization in which F is α-separable and the truefactor cone is well conditioned, with the smallest angle between its extreme rays at least $\theta _ { 0 } > 0$ . Then every nonnegative optimal factorization $( { \hat { F } } , { \hat { G } } )$ satisfies, in a suitable permutation and scaling gauge,

$$
\operatorname* { m i n } _ { P \in \mathrm { P e r m } , \ D \in \mathrm { D } _ { + } } \| \hat { F } P D - F \| \leq \frac { C \alpha } { \sin \theta _ { 0 } } \| F \| ,\tag{72}
$$

so the identification error is linear in the separability deficit α and amplified by the cone condition number 1/ sin $\theta _ { 0 } .$ . As $\alpha  0$ the bound recovers the exact uniqueness ofTheorem 5.

Proofsketch. Identification in separable NMF is recovery of the extreme rays of the latent cone from the data points: under exact separability the anchor rows are exactly the extreme rays, and the constructive algorithm of Arora et al. (2012) identifies them uniquely up to permutation and scaling. Under α-approximate separability the near-anchor rows deviate from the true extreme rays by at most $\alpha ,$ and by the stability of vertex recovery in convex cones, a perturbation of the vertex set by α moves the recovered rays by $O ( \alpha / \sin \theta _ { 0 } )$ when the adjacent rays are separated by angle at least $\theta _ { 0 } { } ^ { \prime }$ ; this is the robust version of separable NMF (Arora et al., 2012) following the noise-robustness analysis of Gillis and Vavasis. The two measurable quantities α and $\theta _ { 0 }$ therefore provide an identifiability budget for the NMF route, which is what Section 5 reports for its nonnegativity condition. □

## E.3 COUPLING IDENTIFICATION WITH ESTIMATION

Proposition 5 (Identification and estimation in one ledger). Let $\hat { F }$ be the empirical risk minimizer of Corollary 4 with encoder capacity fixed and whitening applied on the empirical moments. Then the total identification error of the k-th mode decomposes as

$$
\| \hat { g } _ { k } - ( \pm b _ { k } ) \| \le O \Big ( \frac { \sqrt { \| \hat { F } - F \| _ { L ^ { 2 } } ^ { 2 } } } { \Delta } \Big ) + O \Big ( \frac { r } { \Delta } \Big ) ,\tag{73}
$$

where the first term converts the excess risk ofthe estimate into subspace error via Davis-Kahan and the second is the empirical whitening error of Theorem 11. Both terms scale as $1 / \Delta$ and decay as $n ^ { - 1 / 2 }$

Proof. Split the error around the population whitened solution by the triangle inequality. The second piece vanishes: at the population optimum the whitening theorem (Theorem 2) identifies the modes exactly. The first piece is the deviation of the estimate: $\| \hat { F } - F _ { d } \| _ { L ^ { 2 } } ^ { 2 } \leq \| \hat { F } - F \| _ { L ^ { 2 } } ^ { 2 }$ by optimality, and the Davis-Kahan step of Theorem 11 converts an $L ^ { 2 }$ deviation of the represented function into a singular-vector deviation divided by the gap, giving the first term. □

Proposition 6 (Mode error ledger). Let $\hat { F }$ be the empirical risk minimizer ofCorollary 4 and let $\hat { f } , \hat { g }$ be its whitened encoders. The error in recovering the k-th interaction mode satisfies

$$
\operatorname* { m a x } \{ \| \hat { f } _ { k } - f _ { k } \| , \| \hat { g } _ { k } - g _ { k } \| \} = O \left( \frac { \sqrt { \| \hat { F } - F \| _ { L ^ { 2 } } ^ { 2 } } + r } { \Delta } \right) ,\tag{74}
$$

with $r = O \big ( B ^ { 2 } \sqrt { \log ( d / \delta ) / n } \big )$ ; both terms are inversely proportional to the gap and decay as $n ^ { - 1 / 2 }$

Proof of Proposition 6 (Mode error ledger). Substitute the excess-risk bound of Corollary 4 for $\| \hat { F } - F \| _ { L ^ { 2 } } ^ { 2 }$ into the first term of Proposition $5 ,$ , and the moment radius $r = O \big ( B ^ { 2 } \sqrt { \log ( d / \delta ) / n } \big )$ of Lemma 4 into the second. The result is exactly (74):

$$
\operatorname* { m a x } \{ \| \hat { f } _ { k } - f _ { k } \| , \ \| \hat { g } _ { k } - g _ { k } \| \} = O \Big ( \frac { \sqrt { \| \hat { F } - F \| _ { L ^ { 2 } } ^ { 2 } } + r } { \Delta } \Big ) .\tag{75}
$$

Both terms are inversely proportional to the gap $\Delta$ and decay as $n ^ { - 1 / 2 }$ , which is the statement that identification and estimation are governed by the same spectral gap.

## F EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

This appendix records the complete setups of the experiments of Section 5 and the additional results that the main text summarizes. All numbers are produced by the scripts in the companion repository, whose data files are grouped per study. We state the metrics alongside the tables, because a recurring theme is that gauge-invariant diagnostics cannot separate normalizations while gauge-dependent ones can.

![](images/3d525141238769b8339bd036c21904fce282661c3ee953cfe9d9fcb89df7058f.jpg)

![](images/c9d1d3746929bd01f1814b1afe8a1dfe538a7166a8ef9bd4b6fb71cff1b677c7.jpg)

![](images/04b79ef4f5178a3e20f34aa873518b7acc0dc3c9b04ca90103f96905f27b5dd0.jpg)  
Figure 2: The seven normalization schemes of Table 1, means over four seeds. Left: mode alignment with the true basis (dashed line: 1). Middle: the conditioning diagnostic max(cond $\Sigma _ { f }$ , cond $\Sigma _ { g } )$ on a log scale, whose floor is $\sigma _ { 1 } ^ { 2 } / \sigma _ { d } ^ { 2 } \approx 9 . 5 $ . Right: cross-seed gauge distance (the short tick marks the exact zero of hard whitening). Abbreviations: $1 2 = \mathrm { L } \bar { 2 }$ normalization, coord = percoordinate standardization, cos = cosine normalization, nonneg = two-sided nonnegativity, w-soft = soft whitening, w-hard = hard whitening.

## F.1 CONTROLLED SYNTHETIC STUDY

Targets and training. We fix $\begin{array} { r } { F ( u , v ) = \sum _ { k < d } \sigma _ { k } a _ { k } ( u ) b _ { k } ( v ) } \end{array}$ with $a _ { k } , b _ { k }$ shifted Legendre polynomials, orthonormal on [0, 1], and a prescribed spectrum $\left\{ \sigma _ { k } \right\}$ chosen distinct and separated. Encoders are linear maps over a fixed Legendre feature map, $\mathsf { \bar { f } } ( u ) = U ^ { \top } \varphi ( u )$ with $U \in \bar { \mathbb { R } ^ { p \times d } }$ , so the encoder realization error of Assumption 1 is zero and the normalization is the only variable across runs. Points u, v are drawn on a common grid of n sites per marginal. Training uses Adam with global-norm gradient clipping; the bare bilinear objective is nonconvex and stiff, and plain gradient descent diverges. Analytic gradients are checked against finite differences to about $1 0 ^ { - 9 } .$ . Every scheme runs four seeds.

The seven normalization schemes. The main text reports the headline of this study; Table 1 in Section 5.1 gives the full panel, shown graphically in Figure 2. Fit MSE is the squared regression error. Alignment is the mean over modes of $| \langle \hat { f } _ { k } , a _ { k } \rangle$ | after optimally matching permutation and sign, reported for the two encoders. Cond is the gauge-dependent diagnostic max(cond $\Sigma _ { f } ,$ cond $\Sigma _ { g } )$ Cross-seed is the gauge distance between solutions from different seeds, the mean over pairs of the Procrustes-optimal embedding distance. Only whitening reaches alignment 1.000 with crossseed distance 0.000, matching Theorem $2 ;$ every looser gauge stalls at alignment 0.53 to 0.65 with cross-seed drift 0.28 to 0.35, the residual O of cosine and the full $\mathrm { G L } _ { d }$ of none leaving the modes rotationally entangled (Theorem 4). Whitening drives Cond to its floor $\sigma _ { 1 } ^ { 2 } / \sigma _ { d } ^ { 2 } \approx 9 . 5$ (Corollary 3). The two-sided nonnegative gauge stays at Fit MSE 4.25: positivity forces $\left. f , g \right. \geq 0 .$ , the signchanging target is out of class (Theorem 5), and identifiability is bought at the cost of excluding negative interactions. One measurement subtlety, exploited throughout: cond $\textstyle ( \sum _ { f } \sum _ { g } )$ is gauge invariant, its eigenvalues equal the $\sigma _ { k } ^ { 2 }$ by Lemma 2, so it cannot separate normalizations and is not reported as a diagnostic.

The gap collapse. The gap sweeps use the finite-sample whitening estimator of Appendix E directly, empirical second-moment whitening followed by eigendecomposition, rather than a trained optimizer: with Legendre features the target is represented exactly, and the training identification error has a sharp threshold around $n \approx 3 0 0$ that would hide the $1 / \Delta$ law, whereas the perturbation mechanism of the estimator exposes it cleanly. Table 2 in Section 5.1 reports the two sweeps. The eight points collapse onto a single constant, err $\Delta \cdot { \sqrt { n } } \approx 2 .$ 48 with coefficient of variation 0.12: gap and sample size enter the identification error only through the product, the concrete face of Corollaries 6 and 3. Figure 3 shows the collapse.

The sample-complexity rate law. The linear setting is rank-d matrix sensing, $y _ { i } = u _ { i } ^ { \top } \Theta ^ { \star } v _ { i } + \varepsilon _ { i }$ with rank $\Theta ^ { \star } = d ,$ and Theorem 7 predicts $\mathbb { E } \| \hat { \Theta } - \Theta ^ { \star } \| _ { F } ^ { 2 } \asymp \sigma _ { \varepsilon } ^ { 2 } d ( p + q ) / n$ . We sweep each variable with the others fixed and fit log-log slopes: n at $p = q = 4 , d = 2 { \mathrm { ~ g i v e s ~ } } - 1 . 0 3 { \mathrm { ~ ( t h e o r y ~ - 1 ) } }$ ; d at p = q = 16, n = 2048 gives +0.89 (theory +1); $p + q$ at n = 10<sup>4</sup>, d = 4 gives +1.24 (theory +1). The excess of the last slope is accounted for by the Marchenko-Pastur finite-sample correction of the Gaussian design: measured ratios to $\sigma _ { \varepsilon } ^ { 2 } d ( \dot { 2 p } - d ) / n$ are $1 . 0 0 , 1 . 0 5 , 0 . 9 8 , 1 . 0 9 , 1 . 1 3$ across $p = q = 8 , \ldots , 3 2$ , tracking the correction factor $p q / ( n - p q )$ , which grows from 1.01 to 1.11 over the same range. Gradient descent on the factored pair $( U , V )$ , the estimator class of this paper, is warm-started from the truncated least-squares factors and reproduces the least-squares slope, $- 1 . 0 1$ , on the same grid. The sweep rejects the loose general bound of Remark $^ { 6 , }$ which scales as $d ^ { 3 / 2 } { \sqrt { ( p + q ) / n } }$ and predicts slope $- 1 / 2$ in n: the $1 / n$ dependence on samples is real (Figure 1).

![](images/57b254c7bdd0b29442c7e311e4537b78997c41258a4ebafa92fe298b308891a3.jpg)

![](images/a2f19fb1817637caf44a39a60c3943f833fd1a0c207d42d01cf5db516d8f6a8a.jpg)

![](images/1c89f63b9c8e1bc48e14b59ff98ba95402fd40bbe29234efd80896b17b2b244f.jpg)

Figure 3: The gap collapse. Mode identification error of the finite-sample whitening estimator: against the inverse spectral gap $1 / \Delta$ at fixed n (left) and against sample size n at fixed ∆ (center). Right: the product error $\cdot \Delta \cdot { \sqrt { n } }$ per run across both sweeps, near-constant at ≈ 2.48.  
![](images/015eb0b29d713cb73eec1c4e24e051327145ea6a5ddd0551ba792a1fc0d39288.jpg)

![](images/4c66438138d04b6c0839072b7624bb638436893a89ae7064a6951cda1283c843.jpg)

![](images/cd781d3bf08c84df62226bded0c3d250c5daf2545bc8ae3f61eed1b8494da434.jpg)  
Figure 4: The negative results. Left: on the boolean equality kernel, parameters needed to reach 10% relative error against the number of input bits $m ,$ , for the dual-encoder head (rank d) and the early-interaction threshold. Center: for the narrowband periodic-Gaussian kernel, the dual-encoder rank needed for 10% error against the inverse bandwidth $\bar { 1 } / h ,$ , with a linear fit. Right: relative squared error of the dual-encoder head against its rank $d ,$ one curve per bandwidth $h ,$ with the $1 0 \%$ line marked.

The negative results. Tables 3 and 4 in Section 5.1 give the two regimes; Figure 4 shows both graphically. On the boolean equality kernel $F = \mathbf { 1 } [ u = v ] \mathrm { o n } \{ \pm 1 \} ^ { m }$ , the embedding rank needed for 10% error tracks d $\approx 0 . 9 \cdot \hat { 2 } ^ { m }$ , exponential in m, while the early-interaction threshold on $\langle u , v \rangle$ represents F exactly with 2m + 1 parameters (Theorems 8 and 9). On the narrowband periodic-Gaussian kernel $F _ { h } ( u , v ) = \rho _ { h } ( u - v )$ of bandwidth h on the circle, the required rank $d _ { \varepsilon }$ grows as $\Theta ( 1 / h )$ as the kernel approaches the diagonal (Remark 7).

## F.2 DEEPONET

This study carries the identifiability results of Section 3 onto a real dual-encoder architecture. We train branch-trunk DeepONet heads end to end on four operators and ask three questions, one per paragraph below: whether the rank-selection error curve tracks the decay of the operator spectrum (Corollary 5); whether post-hoc whitening, applied after training without touching the loss, lifts the learned trunk basis to the analytic operator eigenmodes and collapses the cross-seed gauge distance (Theorem 2); and whether the whitened gauge improves out-of-distribution error (Corollary 3). Two operators, the heat semigroup and the Volterra antiderivative, have closed-form spectra and provide exact ground truth for the recovered basis; two, viscous Burgers and Darcy flow, have only empirical spectral proxies and are used for the qualitative claims. The main-text Section 5 reports the conclusions; the tables and figures below give the full numbers.

![](images/44f92f2f4a742adc7a8d7b5d5b64d9adfb074b3834227ddf442ea007713cc0f9.jpg)

![](images/3669904111479e37c2d38766c14717fdfbb453a01cf8a83ba69d9269c57cf5d0.jpg)  
Figure 5: The flat-spectrum error floor on a trained DeepONet head (Theorems 8 and 9). Left: normal ized interaction spectra $\sigma _ { k }$ for three operators, from exponentially decaying (heat) to polynomially decaying (antiderivative) to exactly flat (boxcar block operator). Right: test relative error of a rank-d dual-encoder head versus $d ,$ from an SVD warm start and from random initialization, against the predicted floor $1 - d / m$ ; the dashed line is an early-interaction MLP baseline.

Architecture and training. The branch and trunk networks are MLPs of width 128 and depth 3, outputting embeddings of dimension $d = 3 2 ;$ the rank-selection sweep of Table 7 varies d from 2 to 64. Training runs 300 epochs with Adam at learning rate $1 0 ^ { - 3 }$ and cosine annealing, batch size 256, and five seeds per configuration. Inputs are Gaussian random fields. The four operators are the heat semigroup $e ^ { t \Delta }$ (analytic singular values $\sigma _ { k } = e ^ { - ( k \pi ) ^ { 2 } t }$ , exponential), the antiderivative or Volterra operator $\bar { ( } \sigma _ { k } = 1 / ( ( k - \textstyle \frac { 1 } { 2 } ) \bar { \pi } )$ , polynomial), and viscous Burgers and Darcy flow, whose spectra are empirical proxies, labeled as such. Whitening is realized in two stages: the stable softwhitening penalty during training, the relaxation of Theorem 6 and closely related to the Barlow Twins objective, and hard whitening post hoc on the active rank-r subspace. The two-stage design is forced by the data: the interaction spectra decay fast, leaving an effective rank of 3 to 8 against an embedding width of 32, and in-forward hard whitening inverts about 25 near-zero covariance eigenvalues and backpropagates through a degenerate eigendecomposition, exactly the small-gap failure predicted by the block-stability analysis of Corollary 7 in Appendix E. Because whitening leaves $\bar { \langle b , t \rangle }$ unchanged, applying it after training is exact, and projecting to the active subspace first avoids inverting dead directions. The empirical interaction spectrum is estimated as the square roots of the eigenvalues of $\Sigma _ { b } ^ { 1 / 2 } \Sigma _ { t } \Sigma _ { b } ^ { 1 / 2 }$ , computed with eigvalsh; this avoids the numerically unstable SVD of the ill-conditioned product $\Sigma _ { b } ^ { 1 / 2 } \Sigma _ { t } ^ { 1 / 2 }$ and agrees with the direct computation to $1 0 ^ { - 9 }$ on well-conditioned instances.

Spectra and rank selection. The normalized empirical spectra reproduce the dichotomy qualitatively, heat $1 , 0 . 2 2 , 0 . 0 2 8 , \dots$ . and antiderivative $1 , \bar { 0 } . 1 4 , 0 . \bar { 0 } 4 2 , . . .$ . Because the network allocates no capacity to near-null modes, the learned spectra decay faster than the analytic ones, so we claim recovery of the dominant modes and of the decay type, not a mode-by-mode match. Table 7 gives the rank sweep. The curves are flat past d ≈ 8 and the argmin is dominated by noise, so we do not report a single optimal rank; the honest quantity is the drop from $d = 4 \tan d = 8 ,$ 5% for heat (early saturation, small optimal rank) versus 31% for antiderivative (more modes needed), consistent with Corollary 5. This is qualitative support, and the figure is labeled accordingly.

Basis recovery. Table 8 gives the full per-gauge numbers behind Figure 6. Alignment is the mean over modes of $| \langle \hat { t } _ { k } , \beta _ { k } \rangle$ after optimal permutation and sign matching against the analytic left singular functions, before and after the post-hoc whitening gauge fix; cross-seed is the mean pairwise gauge distance between seeds. Whitening lifts the loose gauges from 0.5 to 0.6 to 0.78 to 0.94 and collapses the cross-seed distance by roughly an order of magnitude; for heat the recovered modes are the analytic Fourier basis $\sqrt { 2 } \sin ( k \pi y )$ , a hard ground truth. On the nonlinear benchmarks, where no analytic basis exists, whitening still tightens the cross-seed distance for Burgers, 0.30 → 0.08; Darcy is the weakest case, $0 . 5 6  \bar { 0 . 3 3 }$ , and its proxy spectrum is the noisiest, which the figure notes.

Table 7: Rank selection, test relative $L ^ { 2 }$ error versus trunk width d. The drop from $d = 4$ to $d = 8$ the claimed quantity: 5% for heat versus 31% for antiderivative (Corollary 5).
<table><tr><td>Operator</td><td colspan="7"> $d = 2$   $d = 4$   $d = 8$   $d = 1 6$   $d = 3 2$ </td></tr><tr><td>heat</td><td>.070</td><td>.056</td><td>.053</td><td>.052</td><td>.055</td><td>.056</td><td>-5%</td></tr><tr><td>antiderivative</td><td>.106</td><td>.081</td><td>.056</td><td>.055</td><td>.070</td><td>.060</td><td>-31%</td></tr><tr><td>burgers</td><td>.237</td><td>.135</td><td>.134</td><td>.133</td><td>.133</td><td>.136</td><td>flat</td></tr><tr><td>darcy</td><td>.167</td><td>.150</td><td>.149</td><td>.153</td><td>.158</td><td>.169</td><td>flat</td></tr></table>

![](images/9ea1315ec3d9d1435637e3e1830fc9a0566b673a1ea7aac5394438139f4cc49b.jpg)

![](images/25baf8df4ad5c7bc599c48100ae7897fec07dd416eaad2ab30d8b578ca63dbff.jpg)  
Figure 6: Trunk-encoder basis quality before and after post-hoc whitening (Theorem 2). Each group on the horizontal axis corresponds to one operator (heat, anti = antiderivative, Burg = Burgers, Darcy) trained under one gauge constraint (none, cosine, or whiten = soft whitening). Left: alignment of the learned trunk basis with the analytic operator eigenmodes (1 is exact). Right: distance between bases learned from different random seeds (lower indicates greater stability).

The payoff of gauge-fixing. Table 9 gives the in-distribution and out-of-distribution relative $L ^ { 2 }$ errors of the none and soft-whitening gauges; the out-of-distribution test set draws inputs from a different smoothness regime of the same operator family. Soft whitening attains the lowest error in all eight cells, with the out-of-distribution gain largest, 2.4× for heat. Two caveats are recorded with the table. First, soft whitening does not reduce seed-to-seed error variance; if anything it increases it, heat standard deviation $0 . 0 0 \bar { 0 } 5 \to 0 . 0 0 3 6$ and Darcy $0 . 0 0 1 7  0 . 0 1 8 1$ , so the stability it confers is in the identified basis of Table 8, not in the error. Second, the soft-whitening training loss carries the decorrelation penalty and its curve is not directly comparable to the unpenalized gauges, so we make no claim about convergence speed.

The flat-spectrum anchor. The block-boxcar operator $G ( a ) ( y ) = c _ { j ( y ) }$ , the output is the block mean of the input field at the block containing y, with $m = 1 6$ blocks, has an exactly flat interaction spectrum: sixteen equal singular values, the seventeenth at $1 0 ^ { - 1 6 }$ . Table 10 gives the trained relative error of the dual-encoder head with linear encoders, from both the SVD warm start and random initialization, against the floor $1 - d / m$ of Theorem $\mathbf { \boldsymbol { 8 } ; }$ the worst deviation is 0.014, the finite-sample excess, and gradient descent reaches the floor from scratch. The early-interaction comparator, a ReLU MLP on concat(a, onehot(y)) with 12.4k parameters, four times the budget of the $d = 1 6$ head, reaches $0 . 0 1 3 \pm \dot { 0 } . 0 0 1$ on the same test set (Theorem 9). One caveat belongs in the setup: with only 8k training samples the comparator memorizes, training MSE $4 \times 1 0 ^ { - 5 }$ and test error 1.7; at 60k samples, where parameters are far fewer than samples, memorization is impossible and the escape shown in Figure 5 is an honest generalization result. The comparison deliberately pits the floor against a well-trained early model.

Table 8: Basis recovery per gauge, five seeds. Alignment: mean mode alignment against the analytic left singular functions, raw and after post-hoc whitening. Cross-seed: mean pairwise gauge distance, raw and after whitening. The effect is Theorem 2 on a trained model.
<table><tr><td>Operator / gauge</td><td>Align</td><td>Whitened align</td><td>Seed dist</td><td>Whitened dist</td></tr><tr><td>heat / none</td><td>0.632</td><td>0.915</td><td>0.442</td><td>0.039</td></tr><tr><td>heat / cosine</td><td>0.529</td><td>0.799</td><td>0.313</td><td>0.007</td></tr><tr><td>heat / whiten_soft</td><td>0.517</td><td>0.780</td><td>0.479</td><td>0.140</td></tr><tr><td>antiderivative / none</td><td>0.582</td><td>0.940</td><td>0.282</td><td>0.064</td></tr><tr><td>antiderivative / cosine</td><td>0.576</td><td>0.823</td><td>0.245</td><td>0.011</td></tr><tr><td>antiderivative /</td><td>0.587</td><td>0.837</td><td>0.414</td><td>0.018</td></tr><tr><td>whiten_soft</td><td></td><td></td><td></td><td></td></tr></table>

Table 9: The payoff of gauge-fixing, five seeds: relative $L ^ { 2 }$ error in distribution and out of distribution, none versus soft whitening. Soft whitening is best in all cells; the out-of-distribution gain is the headline (Corollary 3).
<table><tr><td>Operator</td><td>none, in-dist.</td><td>whiten, in-dist.</td><td>none, OOD</td><td>whiten, OOD</td><td>OOD gain</td></tr><tr><td>heat</td><td>0.051</td><td>0.019</td><td>0.038</td><td>0.016</td><td>2.4×</td></tr><tr><td>antiderivative</td><td>0.057</td><td>0.039</td><td>0.046</td><td>0.031</td><td>1.5×</td></tr><tr><td>burgers</td><td>0.122</td><td>0.112</td><td>0.113</td><td>0.103</td><td>1.1×</td></tr><tr><td>darcy</td><td>0.151</td><td>0.137</td><td>0.142</td><td>0.120</td><td>1.2×</td></tr></table>

## F.3 CLIP

This study tests the gauge view on heterogeneous modalities and on pretrained models that never saw each other. It proceeds in two stages. The white-box synthetic kernel supplies exact ground truth for the mechanism: under linear encoders and white inputs the interaction spectrum equals the singular values of $W _ { f } ^ { \top } W _ { g } ,$ which is gauge invariant, and two independently trained encoders are two random gauges of the same target; on this kernel we verify that the spectrum is invariant and equals the truth, that per-dimension concept probes do not transfer while the canonical subspace does, and that whitening plus gap recovers the text modes exactly. The real CLIP pairs then answer whether the same statements hold for contrastive models trained in the wild: is the interaction spectrum shared across independently pretrained models, are per-dimension probes not, and is the residual relationship a single rotation that whitening removes, exposing interpretable concept axes (Theorem 4)? The main-text Section 5 reports the conclusions; the tables and figures below give the full numbers.

The synthetic multimodal kernel. The white-box counterpart is $\begin{array} { r } { F ^ { \star } ( u , v ) = \sum _ { k } \rho _ { k } ( p _ { k } ^ { \top } u ) ( q _ { k } ^ { \top } v ) } \end{array}$ with $\dot { u } \sim \mathcal { N } ( 0 , I _ { p } ) , v \sim \mathcal { N } ( 0 , I _ { q } )$ , orthogonal canonical directions $P , Q$ , and distinct, gapped canonical correlations $\rho = ( 0 . 9 , 0 . 7 5 , 0 . 6 , 0 . 4 5 , 0 . 3 , 0 . 1 5 )$ . The key identity: under linear encoders and white inputs the interaction spectrum equals the singular values of $\dot { \boldsymbol { B } } = \boldsymbol { W } _ { f } ^ { \top } \boldsymbol { W } _ { g } ,$ , which is explicitly gauge invariant, $f  A f , g  A ^ { - \top } g$ leaves $W _ { f } ^ { \top } W _ { g }$ unchanged, and two independently trained encoders are two random gauges $A _ { 1 } , A _ { 2 }$ . Table 11 collects the numbers: the spectrum is invariant across gauges to $3 \times 1 0 ^ { = } 1 6$ and equals the true ρ to $4 \times 1 0 ^ { - 1 6 }$ ; per-dimension concept probes transfer at 0.43, the best-matched axis reaching only 0.65, while the canonical subspace aligns to 1.0000; whitening plus gap recovers the true text modes at $| \cos | = 1 . 0 0 0$ , against 0.554 raw and 0.581 under cosine, which is stuck at the residual O; and cross-encoder frame stability rises from 0.684 to 1.000. The distance between the 0.58 of cosine and the 1.00 of whitening is, numerically, the contribution of whitening beyond the decorrelation that self-supervised methods already perform, precisely the quantity Theorem 2 isolates.

Real CLIP. Two backbone pairs, each two independent gauges of one similarity function: ViT-B/32 (OpenAI) versus ViT-B/32 (LAION-2B), the same architecture with different pretraining data, and ViT-B/32 (OpenAI) versus RN50 (OpenAI), different architectures. Image features come from CIFAR-100 test images, 49 fine classes spanning animals, vehicles, plants, and household objects, 64 images per class; text features are prompt-ensembled class embeddings plus attribute contrast directions; both towers are projected to a common 32-dimensional working space, the image PCA top-d directions. Whitening is realized via canonical correlation analysis with shrinkage covariance, the finite-sample counterpart of the Theorem 2 gauge fix.

Table 10: The flat-spectrum anchor: trained relative error of a linear-encoder head on the block-boxcar operator versus the floor $1 - d / m$ of Theorem 8<sup>,</sup>9. The two initializations agree, and the error hugs the floor from either start.
<table><tr><td>d</td><td>Trained error Floor</td></tr><tr><td>1</td><td>0.936</td></tr><tr><td>2 0.879</td><td>0.938 0.875</td></tr><tr><td>4 0.752</td><td>0.750</td></tr><tr><td>8 0.502</td><td>0.500</td></tr><tr><td>12 0.258</td><td>0.250</td></tr><tr><td>16</td><td>0.000 0.000</td></tr></table>

Table 11: Synthetic multimodal kernel, ground truth known. The interaction spectrum is the unique gauge-invariant content; whitening plus gap pins the modes, cosine does not.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Spectrum invariant across gauges</td><td> $3 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>Spectrum equals true ρ</td><td> $4 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>Per-dimension probe transfer</td><td>0.43 (best axis 0.65)</td></tr><tr><td>Canonical subspace alignment</td><td>1.0000</td></tr><tr><td>Text mode recovery, raw</td><td>0.554</td></tr><tr><td>Text mode recovery, cosine (isotropization)</td><td>0.581</td></tr><tr><td>Text mode recovery, whitening plus gap</td><td>1.000</td></tr><tr><td>Cross-encoder frame stability, raw → whitened</td><td>0.684 → 1.000</td></tr></table>

Table 12 and Figure 7 give the spectrum comparison across backbone pairs: near-identical spectra, agreement best at the top. Per-dimension probe transfer is 0.28 without alignment and 0.92 after fitting a single global orthogonal map: the two embeddings are related by one O rotation, the residual group Theorem 4 predicts for the cosine gauge, and no coordinatewise repair exists short of discovering that rotation. Table 13 gives the group-signal regression of true labels on the top-6 mode activations: the whitened span contains the animate and natural axes, the cosine span contains only the animate axis, and per-mode statistics show why, whitening concentrates the animate signal in a single mode $( | t | = 3 . { \bar { 1 } } )$ while cosine spreads it across several. Figure 8 shows the canonical text modes as concept axes: mode 1 separates animals (leopard, bear, wolf, camel) from flowers and fruit, mode 2 is the vehicle axis (bus, streetcar, pickup truck, train), mode 3 separates plants from household objects, and mode 5 the food axis (orange, apple, pear, sweet pepper); the cosine principal coordinates are semantic mixtures by comparison. Cross-backbone frame stability, matching mode by their backbone-independent class-signature profiles, favors whitening in both pairs, 0.77 versus 0.72 for the same-architecture pair and 0.84 versus 0.62 across architectures.

What whitening buys for CLIP in practice. None of the steps above requires retraining: whitening is a post-hoc second-moment transform, it leaves the inner-product scores unchanged, and at inference time it costs one fixed linear map. It can therefore be applied as an analysis layer on any deployed CLIP model, and the results above suggest three concrete payoffs. First, model alignment: because independently pretrained pairs are related by a single residual rotation, whitening both models into the shared canonical frame makes their coordinates comparable dimension by dimension, a prerequisite for model merging, distillation, and cross-model probing that raw cosine coordinates do not offer. Second, interpretability with known limits: the whitened axes read as concept axes (animals, vehicles, plants, and food in the CIFAR-100 probe), and the spectral gap quantifies in advance how much interpretability any basis can carry, so an analyst knows where the axes are trustworthy and where the small-gap failure of Appendix E leaves them partially mixed. Third, a spectral summary of the model itself: the interaction spectrum is a gauge-invariant description of what a CLIP pair can discriminate, and its decay predicts the effective interaction rank (Corollary 5) and hence how many concept axes are worth reading out at all. A natural next step is to use the whitened canonical frame as the initialization or constraint for fine-tuning, so that new axes are learned in coordinates that are already interpretable and transferable across models.

Table 12: Interaction spectra across independently pretrained CLIP models. Pearson correlation and normalized L1 difference of the two spectra; per-dimension probe transfer raw and after fitting the single residual O rotation (Theorem 4).
<table><tr><td>Backbone pair</td><td>Spectrum Pearson</td><td>Norm. L1</td></tr><tr><td>ViT-B/32 (OpenAI) vs ViT-B/32 (LAION)</td><td>0.99</td><td>0.065</td></tr><tr><td>ViT-B/32 (OpenAI) vs RN50 (OpenAI)</td><td>0.98</td><td>0.096</td></tr></table>

![](images/fb40b27c2ce3f057a6a009211ced346aa25091ad388c9958dc5d6ea6d41449d0.jpg)

![](images/9ee085bd51a81bc5f11df987d438e425ccf550f44ee64e495f130f71b619fa5d.jpg)  
Figure 7: Shared structure across independently trained CLIP models (Theorem 4). Left: normalized interaction spectra $\sigma _ { k }$ for three CLIP encoders (a ViT-B/32 reference against a LAION-trained ViT-B/32 and an RN50), with Pearson correlations to the reference in the legend. Right: accuracy of a linear concept probe trained on one model and applied to the other, using raw coordinates and after fitting a single shared rotation, for same-architecture and cross-architecture pairs.

Table 13: Group-signal concentration on the reference backbone: $R ^ { 2 }$ of regressing true group labels on the top-6 mode activations, whitened canonical modes versus cosine principal coordinates. Crossbackbone frame stability is reported in the text.
<table><tr><td>Group split</td><td>Whitened</td><td>Cosine</td></tr><tr><td>animate / rest</td><td>0.92</td><td>0.88</td></tr><tr><td>natural / manmade</td><td>0.90</td><td>0.48</td></tr></table>

Summary. The synthetic study establishes the three claims where ground truth is exact: whitening uniquely fixes the gauge, the gap controls identification through the product $\Delta \sqrt { n }$ (collapse constant 2.48), and flat spectra force exponential late-interaction cost. DeepONet carries identifiability onto a real architecture, recovering the analytic eigenbasis with 0.92 to 0.94 alignment and an out-ofdistribution payoff up to $2 . 4 \times ;$ CLIP extends the claim to heterogeneous modalities, the spectrum shared across independently pretrained models while per-dimension probes are not. On a flat spectrum the head is floored at $1 - d / m$ while an early-interaction model reaches 0.013: the interaction spectrum decides whether a multiplicative dual-encoder head is the right tool.

## G INDEX OF RESULTS

Table 14 collects the paper’s numbered results with a one-line statement of what each establishes and where it is stated or proved. The results are grouped by the four questions the paper answers, with the finite-sample lemmas of Appendix E last; within each group the numbering follows the order of

![](images/4a0713d9385ad10e795d62435c6f3ef161c50033e06e30496e801f62025b124b.jpg)  
Figure 8: Whitened canonical text modes (rows) over the 49 CIFAR-100 classes (columns) of the reference backbone: modes read as concept axes (animals, vehicles, plants, food), while the cosine principal coordinates are semantic mixtures.

appearance, and the table doubles as a reading map for the appendices. Remarks and the technical lemmas internal to proofs are not indexed separately; they are cited where they are used.

Table 14: Index of the paper’s numbered results.
<table><tr><td>Result</td><td>What it establishes</td><td>Where</td></tr><tr><td colspan="3">Unification and approximation (Section 2)</td></tr><tr><td>Definition 1</td><td>Interaction spectrum, rank, and modes as the singular data of Section 2 the target</td><td></td></tr><tr><td>Theorem 1</td><td>Approximation error decomposes into a truncation term and Section 2 an encoder-realization term</td><td></td></tr><tr><td></td><td>Assumption 1 Encoder classes contain approximations of the scaled interac- Appendix B tion modes</td><td></td></tr><tr><td>Proposition 1</td><td>Evaluating a rank-d head on all pairs costs  $O ( ( n + m ) C _ { \mathrm { n e t } } +$ </td><td>Appendix B</td></tr><tr><td>Lemma 1</td><td>nm d) Representability of F by a rank-d head iff rank  $( T _ { F } ) \leq d$ </td><td>Appendix B</td></tr><tr><td>Corollary 1</td><td>The truncation term is the best possible error of any rank-d Appendix B head</td><td></td></tr><tr><td>Theorem 3</td><td>Target smoothness controls the decay of the truncation tail</td><td>Appendix B</td></tr><tr><td>Proposition 2</td><td>The dual head escapes the curse of the joint input dimension Appendix B</td><td></td></tr><tr><td>Definition 2</td><td>Gauge action of  $\mathrm { G L } _ { d }$  on encoder pairs leaves the represented Section 3 function unchanged</td><td></td></tr><tr><td>Theorem 2</td><td>Under a spectral gap, whitening identifies the modes up to Section 3 permutation and sign</td><td></td></tr><tr><td>Proposition 3</td><td>Per-coordinate standardization pins the coordinate scales but Appendix C no more</td><td></td></tr><tr><td>Lemma 2</td><td>The eigenvalues of  $\Sigma _ { f } \Sigma _ { g }$  are gauge invariants, equal to the Appendix C  $\sigma _ { k } ^ { 2 }$  at any optimum</td><td></td></tr><tr><td>Proposition 4</td><td>Unfixed gauge renders the risk flat along the orbit and opti- Appendix C mization ill-posed</td><td></td></tr><tr><td>Theorem 4</td><td>Cosine normalization leaves the full orthogonal group as resid- Appendix C ual gauge</td><td></td></tr><tr><td>Corollary 2</td><td>Individual coordinates of a cosine-normalized embedding Appendix C carry no intrinsic meaning</td><td></td></tr><tr><td>Theorem 5</td><td>Two-sided nonnegativity gives NMF-style identifiability, at Appendix C the cost of excluding negative modes</td><td></td></tr><tr><td>Corollary 3</td><td>The spectral gap controls conditioning, identifiability, and Appendix C convergence simultaneously</td><td></td></tr><tr><td>Theorem 6</td><td>A quadratic whitening penalty inherits the identifiability of Appendix C the hard constraint</td><td></td></tr><tr><td colspan="3">Estimation and rank selection (Section 4)</td></tr><tr><td>Lemma 3</td><td>The Rademacher complexity of the inner-product class col- Appendix D</td><td></td></tr><tr><td>Corollary 4</td><td>lapses to the sum of the encoder complexities Excess risk of empirical risk minimization on the head class Appendix D</td><td></td></tr><tr><td>Theorem 7</td><td>Linear heads are rank-d matrix sensing with minimax rate Appendix D  $\sigma _ { \varepsilon } ^ { 2 } d ( p + q ) / n$ </td><td></td></tr><tr><td>Corollary 5</td><td>Optimal rank grows logarithmically in the sample budget for Appendix D</td><td></td></tr><tr><td>The usability criterion (Section 4)</td><td>smooth targets</td><td></td></tr><tr><td colspan="3">Theorem 8</td></tr><tr><td></td><td>A flat spectrum forces every rank-d head to relative error Appendix D  $1 - d / \bar { N }$ </td><td></td></tr><tr><td>Theorem 9 Theorem 10</td><td>Early-interaction models escape the flat-spectrum floor Usability criterion: decide by the measured spectrum whether Appendix D</td><td>Appendix D</td></tr><tr><td></td><td>the head can succeed</td><td></td></tr><tr><td colspan="3">Finite-sample identification (Appendix E)</td></tr><tr><td>Lemma 4 Lemma 5</td><td>Matrix Bernstein concentration for empirical second moments Appendix E Stability of the empirical whitening map under moment per- Appendix E</td><td></td></tr><tr><td></td><td>turbation Finite-sample mode identification with error of order</td><td></td></tr><tr><td>Theorem 11 Corollary 6</td><td> $r / \Delta$  Sample complexity of the whitening estimator:  $n _ { \mathrm { ~ \scriptsize ~ \gtrsim ~ } }$ </td><td>Appendix E Appendix E</td></tr><tr><td></td><td> $B ^ { 4 } \log ( d / \delta ) \dot { / } ( \Delta ^ { 2 } \dot { \varepsilon } ^ { 2 } )$  Block stability of mode recovery under small gaps</td><td></td></tr><tr><td>Corollary 7 Definition 3</td><td>α-approximate separability as a quantitative relaxation of Appendix E</td><td>Appendix E</td></tr><tr><td>Theorem 12</td><td>NMF conditions Approximately separable targets degrade identifiability grace- Appendix E</td><td></td></tr><tr><td></td><td>fully</td><td></td></tr><tr><td>Proposition 5</td><td>Identification and estimation errors combine into one ledger</td><td>Appendix E</td></tr><tr><td>Proposition 6</td><td>Mode error as the bridge between risk and identified basis</td><td>Appendix E</td></tr></table>
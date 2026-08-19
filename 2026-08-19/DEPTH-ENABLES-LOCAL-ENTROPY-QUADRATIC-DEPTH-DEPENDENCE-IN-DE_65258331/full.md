# DEPTH ENABLES LOCAL ENTROPY: QUADRATIC DEPTH DEPENDENCE IN DEEP VARIATION-NORM RELU REGRESSION

Tao Jiang Minbo Gao Shaowei Cai Key Laboratory of System Software (Chinese Academy of Sciences) State Key Laboratory of Computer Science Institute of Software, Chinese Academy of Sciences School of Computer Science and Technology, University of Chinese Academy of Sciences Beijing, China {jiangt,gaomb,caisw}@ios.ac.cn

## ABSTRACT

We study Gaussian regression over the explicit vector-valued Parhi–Nowak deep-RBV<sup>2</sup> architecture with depth L, width w, layer-sum variation budget A, and output bound B. For this $O ( L w ^ { 2 } )$ -parameterized architecture, the known lower and upper bounds differ by one factor of depth. We construct a local packing showing that the quadratic depth dependence is intrinsic under an explicit sample-size-dependent radius condition. The packing has log-cardinality Ω(L<sup>2</sup>w<sup>2</sup> log w); its codewords lie in an $O ( \lambda ) L ^ { 2 }$ ball and are pairwise Ω(λ)-separated. The main ingredients are a bias-corrected bounded-coefficient approximation theorem and balanced amplification: multiplying a depth-D ReLU network by q can be implemented using one constant channel so that every coefficient grows by only $q ^ { { \mathrm { i } } / { \bar { D } } }$ . Translation to vector-valued RBV<sup>2</sup> blocks then has layer-sum cost $\bar { O } ( D w ^ { 2 } q ^ { 1 / D } )$ Gaussian Fano yields a radius-explicit lower bound governed by the output, testing, and representation scales. Under $A = B = R , \sigma \asymp R ,$ and the stated radius condition, this gives

$$
\Re _ { n } ^ { * } \gtrsim \frac { L ^ { 2 } w ^ { 2 } \log w R ^ { 2 } } { n } .
$$

A pseudodimension-based finite-net upper bound gives $\widetilde { O } ( L ^ { 2 } w ^ { 2 } R ^ { 2 } / n )$ for unbounded Gaussian responses. Thus the minimax risk has quadratic polynomial dependence on depth, up to logarithmic factors, and exhibits a transition to representation-limited behavior at smaller radius.

## 1 INTRODUCTION

Depth can compress a compositional description dramatically, but whether the corresponding statistical complexity grows linearly or quadratically with depth depends on more than parameter counting. A generic piecewise-linear computation-graph bound pays once for the number of parameters and once for computational depth. The central question is whether the second payment is a proof artifact or reflects information that depth can actually decode.

Approximation-theoretic benefits of depth are well established. Depth-separation constructions exhibit exponential savings for selected target families, while quantitative ReLU approximation theory identifies regimes in which growing depth improves or is required for optimal approximation rates (Telgarsky, 2016; Yarotsky, 2017, 2018). These results concern representation. The question here is whether depth contributes a second factor to the local statistical complexity of a normconstrained compositional class.

Ganguli and Constantinescu (2026) isolate this issue for a deep variation-space architecture on the circle. Under the $O ( L w ^ { 2 } )$ parameterization used there, their bounds have the schematic form

$$
\Omega \bigg ( \frac { L w ^ { 2 } R ^ { 2 } } { n } \bigg ) \ \leq \ \Re _ { n } ^ { * } \ \leq \ \widetilde O \bigg ( \frac { L ^ { 2 } w ^ { 2 } R ^ { 2 } } { n } \bigg ) .\tag{1}
$$

The displayed base-block formula in that work does not explicitly display the intermediate dimensions, while the cited Parhi–Nowak construction and the $O ( L i w ^ { 2 } )$ parameter count correspond to vectorvalued intermediate maps. We therefore study the vector-valued interpretation consistent with the cited construction and parameter count: $d _ { 0 } = d _ { L } = 1 , d _ { \ell } \leq w$ , and each compositional block has hidden width at most w.

Norm-controlled neural function spaces offer a complementary account of network complexity. Early Barron-type and convex neural-network formulations control approximation and estimation through function-space norms (Barron, 1993; Bach, 2017). Exact descriptions of bounded-norm ReLU networks subsequently connected such norms to spline and Radon-domain variation spaces (Savarese et al., 2019; Ongie et al., 2020; Parhi and Nowak, 2021). The deep compositional and vector-valued extensions developed in Parhi and Nowak (2022, 2026) and Shenouda et al. (2024) provide the function-space setting used here.

Within this architecture, the missing depth factor is realized by a local function-space code rather than by a refinement of the generic upper-bound argument. We construct such a code with

$$
\log | \mathcal { Z } | = \Omega ( L ^ { 2 } w ^ { 2 } \log w ) .
$$

The code begins with bounded-coefficient bit extraction. A width-m, depth-D ReLU network can approximate every bounded 1-Lipschitz function on [0, 1] to accuracy

$$
O \big ( ( m ^ { 2 } D ^ { 2 } \log m ) ^ { - 1 } \big )\tag{2}
$$

while keeping every matrix and bias entry bounded by one. The construction originates in Ou et al. (2024). Under the affine-layer convention with biases, the published layerwise rescaling step requires a correction to obtain homogeneous scaling. We provide this correction by augmenting each hidden state with a constant channel; a unit-coefficient fan-out construction then trades coefficient magnitude for additional depth. The corrected derivation preserves (2) up to universal constants.

The same augmented-state idea yields our key architectural lemma. If N has depth $D ,$ then $q N$ has the same depth and one additional hidden coordinate, with coefficient magnitude only $q ^ { 1 / D }$ . A coefficient-s fully connected layer has vector-valued $\mathcal { R } \mathrm { B V } ^ { 2 }$ cost $O ( w ^ { 2 } s )$ , hence

$$
\mathfrak { D } _ { D , w } ( q N ) \lesssim D w ^ { 2 } q ^ { 1 / D } .\tag{3}
$$

This converts the $1 / M$ label scale of a bit-extraction code into a statistical margin without paying M in a single layer.

The minimax question requires more than a global entropy bound. A bounded-weight network class may contain exponentially many separated functions that either fall outside the layer-sum variation ball or live at an amplitude much larger than the Gaussian testing scale. The relevant obstruction must survive both the representation constraint and localization. Our construction does so: after the statistical amplitude $\lambda$ is chosen, every codeword has $L ^ { 2 }$ norm $O ( \lambda )$ , distinct codewords are Ω(λ) apart, and all codewords remain in the prescribed deep- $\cdot \mathcal { R } \mathrm { B V } ^ { 2 }$ ball. Tight covering-number bounds for ordinary bounded-weight fully connected ReLU networks already show global entropy of order

$$
W ^ { 2 } D \log \left( \frac { ( W + 1 ) ^ { D } B ^ { D } } { \varepsilon } \right) ,
$$

which is quadratic in D at fixed $B = 1$ and fixed accuracy (Ou and Bolcskei¨ , 2026). The present result embeds a comparable quadratic-depth code into the variation-constrained class at the testing scale and converts it into a minimax lower bound.

This local viewpoint also connects the construction to statistical analyses of neural regression, which give minimax or near-minimax guarantees under compositional smoothness, Besov-type, and shallow neural variation-space assumptions (Schmidt-Hieber, 2020; Suzuki, 2019; Parhi and Nowak, 2023). Classical entropy methods connect packing and covering numbers to minimax risk (Yang and Barron, 1999; Tsybakov, 2009), while localized complexity theory emphasizes the geometry near the testing scale (Bartlett et al., 2005). Sharp metric-entropy results for shallow neural variation spaces provide a close comparison (Siegel and Xu, 2024).

## Contributions.

1. We give a bias-corrected derivation of the unit-coefficient approximation rate (2). The new homogeneous-lift lemma handles all biases exactly and replaces the two bias-sensitive scaling steps used in the approximation argument.

2. We construct a local packing of size $\exp ( \Omega ( M ) )$ , with $M = \Theta ( L ^ { 2 } w ^ { 2 } \log w )$ , inside the explicit vector-valued deep- $\cdot \bar { \mathcal { R } } \mathrm { B V } ^ { 2 }$ architecture.

3. We prove a radius-explicit minimax lower bound separating the output cap, Gaussian testing scale, and representation-limited scale. A sample-size-dependent corollary gives a less restrictive sufficient condition than the corresponding uniform-in-n condition.

4. We prove a Gaussian-regression upper bound using a formal architecture-to-computation-graph lemma, pseudodimension, population covering, and a finite-class least-squares oracle inequality that handles Gaussian responses directly.

Organization. Sections 2–3 give the model and theorems. Section 4 locates the second depth sum. Sections 5–7 construct the packing and prove the lower bound. Section 8 proves the Gaussian upper bound. The appendices contain the corrected approximation reduction and complete technical proofs.

## 2 STATISTICAL AND FUNCTION-CLASS SETTING

Circle and risk. Identify the circle with $t \in [ 0 , 2 )$ under normalized uniform measure $\mu ( \mathrm { d } t ) =$ $\mathrm { d } t / 2 .$ . We observe

$$
T _ { i } \sim \mu , \qquad Y _ { i } = f ^ { \star } ( T _ { i } ) + \xi _ { i } , \qquad \xi _ { i } \sim N ( 0 , \sigma ^ { 2 } ) ,\tag{4}
$$

independently. For a class ${ \mathcal F } .$

$$
\Re _ { n } ^ { * } ( \mathcal { F } , \sigma ) : = \operatorname* { i n f } _ { \widehat { f } } \operatorname* { s u p } _ { f ^ { \star } \in \mathcal { F } } \mathbb { E } _ { f ^ { \star } } \left\| \widehat { f } - f ^ { \star } \right\| _ { L ^ { 2 } ( \mu ) } ^ { 2 } .\tag{5}
$$

Vector-valued blocks. For $s : \mathbb { R } ^ { d } \to \mathbb { R } ^ { D ^ { \prime } }$ of the form

$$
s ( x ) = \sum _ { k = 1 } ^ { K } v _ { k } \rho ( w _ { k } ^ { \top } x - b _ { k } ) + C x + c _ { 0 } ,\tag{6}
$$

we use the Parhi–Nowak norm

$$
\| s \| _ { \mathcal { R } \mathrm { B V } ^ { 2 } ( d ; D ^ { \prime } ) } : = \sum _ { k = 1 } ^ { K } \| v _ { k } \| _ { 1 } \| w _ { k } \| _ { 2 }\tag{7}
$$

$$
+ \sum _ { j = 1 } ^ { D ^ { \prime } } \left( | s _ { j } ( 0 ) | + \sum _ { r = 1 } ^ { d } | s _ { j } ( e _ { r } ) - s _ { j } ( 0 ) | \right) .\tag{8}
$$

This is the vector-valued Radon-domain variation-space convention of Parhi and Nowak (2022), consistent with the multi-output variation-space framework of Shenouda et al. (2024).

Deep architecture. Fix

$$
d _ { 0 } = d _ { L } = 1 , \qquad 1 \leq d _ { \ell } \leq w \quad ( 1 \leq \ell < L ) ,\tag{9}
$$

and blocks $s _ { \ell } : \mathbb { R } ^ { d _ { \ell - 1 } }  \mathbb { R } ^ { d _ { \ell } }$ of the form (6), with $K _ { \ell } \le w$ . Define the layer-sum representation cost

$$
\mathfrak { N } _ { L , w } ( f ) : = \operatorname* { i n f } _ { f = s _ { L } \circ \cdots \circ s _ { 1 } } \sum _ { \ell = 1 } ^ { L } \| s _ { \ell } \| _ { \mathcal { R } \mathrm { B V } ^ { 2 } ( d _ { \ell - 1 } ; d _ { \ell } ) } ,\tag{10}
$$

where the infimum is over (9) and $K _ { \ell } \le w$ . We refer to (10) as a representation cost: scalar gains can be distributed across layers, so the resulting functional is not one-homogeneous.

Definition 1 (Deep variation architecture class). $F o r A , B > 0 ,$ , let

$$
\mathcal { C } _ { L , w } ( A , B ) : = \{ \boldsymbol { f } : [ 0 , 2 )  \mathbb { R } : \mathfrak { V } _ { L , w } ( \boldsymbol { f } ) \leq A , \  \boldsymbol { f }  _ { \infty } \leq B \} ,\tag{11}
$$

with continuous endpoint identification when periodized. Write $\Re _ { n } ^ { * } ( A , B , \sigma ) = \Re _ { n } ^ { * } ( \mathcal { C } _ { L , w } ( A , B ) , \sigma )$ The motivating normalization is $A = B = R$ and $\sigma \asymp R$

A block contains $O ( w ^ { 2 } )$ scalar parameters, so the architecture has

$$
W _ { \mathrm { p a r } } = { \cal O } ( L w ^ { 2 } ) .\tag{12}
$$

Appendix A gives the exact count and a formal computation-graph realization.

Standard-network convention. Let $\mathcal { N } ( W , D , B )$ denote scalar-output realizations

$$
h _ { \ell } = \rho ( A _ { \ell } h _ { \ell - 1 } + b _ { \ell } ) , \quad 1 \leq \ell < D , \qquad N ( x ) = A _ { D } h _ { D - 1 } + b _ { D } ,\tag{13}
$$

with hidden width at most $W$ and every matrix and bias entry bounded by B in absolute value. Depth counts affine maps, including the final affine output layer. This is the convention used in the bounded-coefficient approximation result.

## 3 MAIN RESULTS

We first record the approximation input in the corrected form used below.

Theorem 2 (Bias-corrected bounded-coefficient approximation). There exist universal constants $C _ { \mathrm { a p p } } , D _ { 0 } > 0$ such that,for all integers m, $D \geq D _ { 0 }$ and every continuous $g : [ 0 , 1 ]  \mathbb { R }$ satisfying $\begin{array} { r } { \| g \| _ { \infty } \leq 1 } \end{array}$ and $\operatorname { L i p } ( g ) \leq 1$ , there is $N \in \mathcal { N } ( m , D , 1 )$ with

$$
\| N - g \| _ { L ^ { \infty } ( [ 0 , 1 ] ) } \leq \frac { C _ { \mathrm { a p p } } } { m ^ { 2 } D ^ { 2 } \log m } .\tag{14}
$$

The bit-extraction construction is due to Ou et al. (2024). Appendix B gives a bias-corrected derivation and tracks all resulting changes in width and depth explicitly.

The correction has two steps. Starting from a polynomial-coefficient approximant $N \in \mathcal { N } ( W , D , B )$ a homogeneous lift augments each hidden state by one constant coordinate and realizes $B ^ { - D } \dot { N }$ with unit coefficients. A width- $( W + 1 )$ fan-out construction then trades coefficient magnitude for additional depth, restoring the output amplitude using $J = \lceil D \log B / \log \lfloor ( W + 1 ) / 2 \rfloor \rceil$ additional layers. In the required case $B = \overset \bullet W ^ { 2 }$ , one has $J = \bar { O } ( D )$ ; replacing the two bias-sensitive splinescaling calls by this construction preserves the width/depth asymptotics and yields (14).

Let

$$
D = L - 1 , \qquad m = w - 1 , \qquad M = \left\lfloor c _ { 0 } m ^ { 2 } D ^ { 2 } \log m \right\rfloor ,\tag{15}
$$

where $c _ { 0 } > 0$ is sufficiently small. One layer is reserved for folding the circle and one hidden coordinate for homogeneous amplification.

Theorem 3 (Radius-explicit lower bound). There are universal constants $c , C _ { 0 } , C _ { \mathrm { t r } } , c _ { 0 } > 0$ and integers $L _ { 0 } , w _ { 0 }$ such that, $f o r L \ge L _ { 0 } + 1 , w \ge w _ { 0 } + 1$ , every $A , B , \sigma > 0$ , and every $n \geq 1$

$$
\mathfrak { R } _ { n } ^ { * } ( A , B , \sigma ) \geq c \left[ \operatorname* { m i n } \left\{ B , \sigma \sqrt { \frac { M } { n } } , \frac { 1 } { M } \left( \frac { ( A - C _ { 0 } ) _ { + } } { C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { D } \right\} \right] ^ { 2 } .\tag{16}
$$

The packing is local: at its active amplitude $\lambda ,$ every codeword has $L ^ { 2 }$ norm at most Cλ, and distinct codewords are at least cλ apart.

The three terms are, respectively, the output cap, Gaussian testing scale, and representation-limited amplification scale.

Corollary 4 (Sample-size-dependent quadratic-depth regime). Fix constants $0 < c _ { \sigma } \leq C _ { \sigma } < \infty .$ There exist constants $c , c ^ { \prime } , C _ { \mathrm { r a d } } > 0 ;$ , depending at most on $c _ { \sigma }$ and $C _ { \sigma }$ , such that the following holds. Assume $A = B = R , c _ { \sigma } R \leq \sigma \leq C _ { \sigma } \bar { R _ { , } } n \geq \bar { M } ,$ , and $R \geq 2 C _ { 0 } , I f$

$$
R ^ { D - 1 } \ge \left( C _ { \mathrm { r a d } } D w ^ { 2 } \right) ^ { D } \frac { M ^ { 3 / 2 } } { \sqrt { n } } ,\tag{17}
$$

then

$$
\Re _ { n } ^ { * } ( R , R , \sigma ) \geq c \frac { R ^ { 2 } M } { n } \geq c ^ { \prime } \frac { L ^ { 2 } w ^ { 2 } \log w R ^ { 2 } } { n } .\tag{18}
$$

We call the parameter range in Corollary 4, in which the Gaussian testing scale is active, the statistical regime.

A convenient condition uniform over all $n \geq M$ is

$$
R ^ { D - 1 } \geq \left( C _ { \mathrm { r a d } } D w ^ { 2 } \right) ^ { D } M .\tag{19}
$$

Equivalently,

$$
\begin{array} { r l } & { R \geq ( C _ { \mathrm { r a d } } D w ^ { 2 } ) ^ { D / ( D - 1 ) } M ^ { 1 / ( D - 1 ) } } \\ & { \quad = C _ { \mathrm { r a d } } D w ^ { 2 } \exp \left( \frac { \log \left( C _ { \mathrm { r a d } } D w ^ { 2 } \right) + \log M } { D - 1 } \right) } \\ & { \quad = C _ { \mathrm { r a d } } D w ^ { 2 } \exp \left( O \Big ( \frac { \log \left( L w \right) } { L } \Big ) \right) . } \end{array}\tag{20}
$$

Theorem 5 (Gaussian pseudodimension upper bound). There exists a universal $C > 0$ such that,for all integers $L , w \ge 1$ , all $A , B , \sigma > 0$ , and all integers $n \geq 2 ,$

$$
\Re _ { n } ^ { * } ( A , B , \sigma ) \leq C \operatorname* { m i n } \left\{ B ^ { 2 } , { \frac { ( \sigma ^ { 2 } + B ^ { 2 } ) L ^ { 2 } w ^ { 2 } \log ( 2 L w ) \log ( e n ) } { n } } \right\} .\tag{21}
$$

Consequently, under Corollary 4,

$$
c \frac { L ^ { 2 } w ^ { 2 } \log w R ^ { 2 } } { n } \leq \Re _ { n } ^ { * } \leq C \frac { L ^ { 2 } w ^ { 2 } \log ( L w ) \log ( e n ) R ^ { 2 } } { n } .\tag{22}
$$

Thus the polynomial depth exponent is quadratic; the remaining discrepancy is logarithmic.

Remark 6 (The radius condition is structural). For $f = s _ { L } \circ \cdots \circ s _ { 1 }$ , the Parhi–Nowak Lipschitz estimate and AM–GM give

$$
\operatorname { L i p } ( f ) \leq \prod _ { \ell = 1 } ^ { L } \| s _ { \ell } \| _ { \mathcal { R } \operatorname { B V } ^ { 2 } } \leq ( A / L ) ^ { L } .
$$

Small layer-sum budget can therefore collapse the nonconstant part ofthe class. An all-radius lower bound proportional to $L ^ { 2 } w ^ { 2 } B ^ { 2 } / n$ would befalse.

## 4 WHERE THE EXTRA DEPTH FACTOR LIVES

The lower and upper bounds in Section 3 determine the polynomial depth exponent. Before constructing the packing, we identify the ordered-pair structure through which the second factor of depth enters the usual layerwise entropy calculation.

The motivating upper bound acquires depth in a single generic step:

$$
W _ { \mathrm { p a r } } = O ( L w ^ { 2 } ) , \qquad \mathrm { P d i m } = O ( W _ { \mathrm { p a r } } L \log W _ { \mathrm { p a r } } ) .
$$

Norm-based capacity bounds provide a different parameter-space route to depth-dependent generalization estimates (Neyshabur et al., 2015; Golowich et al., 2018). The question here is whether the layer-sum function-space constraint still contains a local packing of quadratic depth complexity. Reconstructing a layerwise covering calculation shows the same ordered-pair structure. Suppose layer ℓ has entropy $\bar { H } _ { \ell } ( \delta _ { \ell } ) \lesssim p _ { \ell } \log ( \mathbf { \bar { \phi } } C _ { \ell } / \delta _ { \ell } )$ and downstream Lipschitz amplification $\textstyle A _ { \ell } = \prod _ { j > \ell } a _ { j }$ A telescoping decomposition gives

$$
\left\| s _ { L } \circ \cdots \circ s _ { 1 } - \widetilde s _ { L } \circ \cdots \circ \widetilde s _ { 1 } \right\| \leq \sum _ { \ell = 1 } ^ { L } A _ { \ell } \delta _ { \ell } .\tag{23}
$$

The entropy-minimizing allocation under total error ε is $\begin{array} { r } { \delta _ { \ell } = \varepsilon p _ { \ell } / ( P A _ { \ell } ) , P = \sum _ { \ell } p _ { \ell } } \end{array}$ , and produces

$$
\sum _ { \ell = 1 } ^ { L } p _ { \ell } \log A _ { \ell } = \sum _ { \ell = 1 } ^ { L } p _ { \ell } \sum _ { j > \ell } \log a _ { j }\tag{24}
$$

$$
= \sum _ { j = 2 } ^ { L } \left( \sum _ { \ell < j } p _ { \ell } \right) \log a _ { j } .\tag{25}
$$

In the homogeneous case $p _ { \ell } = p$ and $a _ { j } = a > 1$ , this becomes

$$
\sum _ { j = 2 } ^ { L } \left( \sum _ { \ell < j } p \right) \log a = p \log a \sum _ { j = 2 } ^ { L } ( j - 1 ) = { \frac { p L ( L - 1 ) } { 2 } } \log a .\tag{26}
$$

Thus the second factor of depth counts ordered pairs consisting of a perturbed layer and a downstream amplification layer. For $p \asymp \bar { w } ^ { 2 }$ , the resulting contribution is $\scriptstyle \Theta ( L ^ { 2 } w ^ { 2 } )$ . The calculation identifies the source of the quadratic term in a layerwise covering argument; necessity requires a function-space lower bound, since a global parametrization could in principle avoid this bookkeeping.

The packing below provides such a lower bound. At its active amplitude λ, it lies in an $O ( \lambda )$ ball, has $\Omega ( \lambda )$ pairwise separation, and has log-cardinality $\Omega ( L ^ { 2 } w ^ { 2 }$ log w). Consequently, the localized covering entropy at this scale is at least $\Omega ( L ^ { 2 } w ^ { 2 } \log { w } )$ , precluding a uniform covering-entropy upper bound of order $\mathrm { \dot { \it ~ O } } ( L w ^ { 2 } )$ in the statistical regime.

$$
\overbrace { \underbrace { \mathrm {  { \mathrm {  ~ \ k n a r ~ g r i d c o d e } } } } _ { \mathrm {  { \log ~ | \mathcal { Z } | = \Omega ( M ) } ~ } } } ^ { \mathrm { \normalfont { B i n a r ~ g r i d c o d e } } } \} substack { \longrightarrow } ( \underbrace  \overbrace { \mathrm {  { \mathrm {  ~ B i n s ~ c o r r e c t e d ~ u n i t ~ } } } \allowbreak \mathrm {  ~ \cdot ~ } \mathrm { o r f i c i e n t ~ a p p r o x i m a t i o n } } ^ { \mathrm { \normalfont { B i n s ~ c o r r e c t e d ~ u n i t ~ a ~ c o e f t i c i e n t a g o r p r o x i m a t i o n } } } \} ) \underbrace { \overbrace { \mathrm {  ~ \cdot ~ } \mathrm {  { \ F ~ } } _ { \mathrm {  { \ c o s t } ~ } } ^ { \mathrm { \scriptstyle { \texttt { B a l a r c e d } } } } \lambda W ^ { \mathrm { \scriptstyle { B i f t } } } \allowbreak \mathrm {  ~ \times ~ } \mathrm {  ~ \lambda ~ } } ^ { \mathrm { \scriptstyle { \texttt { B a l a r c e d } ~ } } } } _ { \displaystyle M = \Theta ( D ^ { 2 } m ^ { 2 } \log m ) } ) . \qquad \overbrace { \mathrm {  { \mathrm {  ~ \Lambda ~ } } \boldsymbol { \mathrm {  ~ \ r ~ } } } \boldsymbol { \mathrm {  \texttt { n u b s } } } ^ { \mathrm { \scriptstyle { B i n ~ } } } \lambda ^ { \mathrm { \tiny { B i n } } } } ^ { \mathrm { \boldsymbol { F o u b s ~ S i n a r e } } } \} ^ { \mathrm {  { \texttt { B a l a r c e d } ~ } } \boldsymbol { \mathrm { \boldsymbol { D } } } } \substack { \longrightarrow } ( \underbrace { \mathrm {  ~ \ c h a n s t ~ a f t i c a l o n } } _ { \displaystyle \mathrm {  { \texttt { F o l o w } } \boldsymbol { \mathrm { B i s } } \atop \boldsymbol { \ddots } \boldsymbol { \mathrm { B } } \boldsymbol { \times  { \boldsymbol { \mu } } } ^ { \mathrm { \tiny { B i r } } } } } ) .
$$

Figure 1: Proof mechanism. The constant channel is used twice: to repair coefficient rescaling in the approximation theorem and to amplify the statistical code without concentrating the gain in one layer.

## 5 A DEPTH-ENABLED LOCAL CODE

Let $x _ { j } = j / M , j = 0 , \dotsc , M .$ . For $z \in \{ 0 , 1 \} ^ { M - 1 }$ , set $z _ { 0 } = z _ { M } = 0$ and let $h _ { z }$ linearly interpolate $( x _ { j } , z _ { j } )$ . Then $\| h _ { z } \| _ { \infty } \leq 1 , \mathrm { L i p } ( h _ { z } ) \leq \dot { M } .$ , and $g _ { z } = h _ { z } / M$ is bounded and 1-Lipschitz. Theorem 2 gives $N _ { z } \in \mathcal { N } ( m , \tilde { D , } 1 )$ with

$$
\| N _ { z } - g _ { z } \| _ { \infty } \leq \frac { C _ { \mathrm { a p p } } } { m ^ { 2 } D ^ { 2 } \log m } .\tag{27}
$$

For $M = \lfloor c _ { 0 } m ^ { 2 } D ^ { 2 } \log m \rfloor$ and sufficiently small c<sub>0</sub>, the rescaled functions $Q _ { z } = M N _ { z }$ obey

$$
\| Q _ { z } - h _ { z } \| _ { \infty } \leq \eta\tag{28}
$$

for a fixed small universal η.

If $d _ { j } = z _ { j } - z _ { j } ^ { \prime }$ , direct integration on the jth cell gives

$$
\int _ { j / M } ^ { ( j + 1 ) / M } ( h _ { z } - h _ { z ^ { \prime } } ) ^ { 2 } \mathrm { d } x = \frac { d _ { j } ^ { 2 } + d _ { j } d _ { j + 1 } + d _ { j + 1 } ^ { 2 } } { 3 M } .\tag{29}
$$

Since $a ^ { 2 } + a b + b ^ { 2 } \geq ( a ^ { 2 } + b ^ { 2 } ) / 2$ and $d _ { 0 } = d _ { M } = 0$

$$
\Vert h _ { z } - h _ { z ^ { \prime } } \Vert _ { 2 } ^ { 2 } \geq \frac { d _ { H } ( z , z ^ { \prime } ) } { 3 M } .\tag{30}
$$

The Varshamov–Gilbert bound (Tsybakov, 2009) therefore yields $\mathcal { Z } \subseteq \{ 0 , 1 \} ^ { M - 1 }$ such that

$$
\log | \mathcal { Z } | \geq c M , \qquad c \leq \| Q _ { z } - Q _ { z ^ { \prime } } \| _ { 2 } \leq C \quad ( z \neq z ^ { \prime } ) .\tag{31}
$$

The code therefore has the required cardinality and local geometry. The remaining task is to amplify it to scale λ without exhausting the layer-sum budget.

## 6 BALANCED AMPLIFICATION AND $\mathcal { R } \mathrm { B V } ^ { 2 }$ TRANSLATION

Since $Q _ { z } = M N _ { z }$ , reaching separation of order λ requires a gain $q = \lambda M$ relative to the unitcoefficient approximant. Applying this gain entirely in the final layer would charge order q to one block. Balanced amplification distributes the same gain across depth while a constant homogeneous coordinate transports every bias exactly.

Lemma 7 (Balanced amplification). Let $N \in \mathcal { N } ( m , D , 1 )$ have exact depth $D \geq 2 .$ . For every $q > 0$ $q N$ has a depth-D, width-at-most-(m + 1) realization with coefficient magnitude at most $q ^ { 1 / D }$

Writing $s = q ^ { 1 / D }$ , the augmented state is $\widetilde { h } _ { \ell } = ( s ^ { \ell } h _ { \ell } , s ^ { \ell } )$ . The extra coordinate transports the scaled biases, and the final affine map returns $s ^ { D } N = { \dot { q } } N .$ Appendix D gives the matrices and exact-depth padding.

Lemma 8 (Layer translation). Let $T ( x ) \ = \ \rho ( A x + b )$ coordinatewise, with input and output dimensions at most w and $| A _ { i j } | , | b _ { i } | \leq \dot { s }$ . Then T is an allowed vector-valued block with at most w atoms and

$$
\| T \| _ { \mathcal { R } \mathrm { B V } ^ { 2 } } \leq C w ^ { 2 } s .\tag{32}
$$

The same estimate holds for an affine output map.

Thus, for $N \in \mathcal { N } ( m , D , 1 )$ ,

$$
\mathfrak { D } _ { D , m + 1 } ( q N ) \leq C _ { \mathrm { t r } } D ( m + 1 ) ^ { 2 } q ^ { 1 / D } .\tag{33}
$$

To embed the interval code on the circle, use

$$
r ( t ) = t - 2 \rho ( t - 1 ) , \quad \quad 0 \leq t \leq 2 .\tag{34}
$$

It has univariate $\mathcal { R } \mathrm { B V } ^ { 2 }$ norm $C _ { 0 } = 3 ,$ , maps both endpoints to zero, and traverses [0, 1] once in each direction. For

$$
F _ { z } ( t ) = \lambda Q _ { z } ( r ( t ) ) ,\tag{35}
$$

we have

$$
\| F _ { z } - F _ { z ^ { \prime } } \| _ { L ^ { 2 } ( [ 0 , 2 ) , \mathrm { d } t / 2 ) } = \lambda \| Q _ { z } - Q _ { z ^ { \prime } } \| _ { L ^ { 2 } ( [ 0 , 1 ] ) }\tag{36}
$$

and, using $D = L - 1 , m + 1 = w .$

$$
\mathfrak { D } _ { L , w } ( F _ { z } ) \le C _ { 0 } + C _ { \mathrm { t r } } D w ^ { 2 } ( \lambda M ) ^ { 1 / D } .\tag{37}
$$

Equation (37) completes the representation step. We now combine this bound with the output and Gaussian testing constraints.

## 7 FROM THE LOCAL CODE TO THE MINIMAX LOWER BOUND

Three constraints determine the admissible amplitude. First, (28) gives $\begin{array} { r } { \| Q _ { z } \| _ { \infty } \leq 1 + \eta . } \end{array}$ , so the output cap is satisfied whenever

$$
\lambda \lesssim B .\tag{38}
$$

Second, (37) places the code in the layer-sum ball when $A > C _ { 0 }$ and

$$
\lambda \lesssim { \frac { 1 } { M } } \left( { \frac { A - C _ { 0 } } { C _ { \mathrm { t r } } D w ^ { 2 } } } \right) ^ { D } .\tag{39}
$$

When $A \leq C _ { 0 }$ , the positive-part convention in the theorem records the resulting trivial lower bound. Finally, the Gaussian testing scale is determined by the size of the code. Since log $| \mathcal { Z } | \gtrsim M$ and the pairwise divergence is of order $n \lambda ^ { 2 } / \sigma ^ { 2 }$ , Fano’s inequality requires

$$
\lambda \lesssim \sigma \sqrt { \frac { M } { n } } .\tag{40}
$$

We therefore choose

$$
\lambda = c _ { \lambda } \operatorname* { m i n } \left\{ B , \sigma \sqrt { \frac { M } { n } } , \frac { 1 } { M } \left( \frac { ( A - C _ { 0 } ) _ { + } } { C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { D } \right\} .\tag{41}
$$

Up to universal constants, (41) is the largest amplitude compatible with all three restrictions.

After reducing $c _ { \lambda } .$ , equations (28) and (37) imply $F _ { z } \in \mathcal { C } _ { L , w } ( A , B )$ . Equations (31) and (36) give

$$
c \lambda \leq \| F _ { z } - F _ { z ^ { \prime } } \| _ { 2 } \leq C \lambda , \qquad \| F _ { z } \| _ { 2 } \leq C \lambda .\tag{42}
$$

Thus the same family is both local and separated at the chosen amplitude. For Gaussian random-design regression,

$$
D _ { \mathrm { K L } } ( P _ { z } ^ { \otimes n } \| P _ { z ^ { \prime } } ^ { \otimes n } ) = \frac { n } { 2 \sigma ^ { 2 } } \left\| F _ { z } - F _ { z ^ { \prime } } \right\| _ { 2 } ^ { 2 } \leq C \frac { n \lambda ^ { 2 } } { \sigma ^ { 2 } } .\tag{43}
$$

The testing restriction (40) makes this a sufficiently small multiple of $M \lesssim \log | \mathcal { Z } |$ . Nearest-neighbor decoding and Fano’s inequality (Tsybakov, 2009) then yield risk $\Omega ( \lambda ^ { 2 } )$ , proving Theorem 3.

For Corollary $4 , n \geq M$ makes the testing amplitude at most a constant multiple of R. If $R \geq 2 C _ { 0 }$ then $R - C _ { 0 } ^ { \dot { } } \geq R / 2$ , and (17) implies

$$
\frac { 1 } { M } \left( \frac { R - C _ { 0 } } { C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { D } \geq c R \sqrt { \frac { M } { n } } .
$$

Hence $\lambda ^ { 2 } \asymp R ^ { 2 } M / n$ . The uniform condition (19) follows because $M ^ { 3 / 2 } / \sqrt { n } \leq M$ for $n \geq M$ Complete constants appear in Appendix F.

## 8 A GAUSSIAN UPPER BOUND WITHOUT BOUNDED RESPONSES

To match the lower bound in its polynomial depth dependence, we now derive a Gaussian upper bound for the same architecture.

Lemma 9 (Architecture-to-computation graph). Every function in $\mathcal { C } _ { L , w } ( A , B )$ is realized by a piecewise-linear computation graph with at most $C L w ^ { 2 }$ real parameters and computational depth at most CL. Consequently,

$$
\operatorname { P d i m } ( \mathcal { C } _ { L , w } ( A , B ) ) \leq C L ^ { 2 } w ^ { 2 } \log ( 2 L w ) .\tag{44}
$$

The proof in Appendix A explicitly handles vector-valued blocks, affine skips, and variable intermediate dimensions. It then applies the piecewise-linear pseudodimension theorem of Bartlett et al. (2019). For every design distribution $P , \mathbf { \dot { a } } \left[ - B , B \right]$ ]-valued class of pseudodimension V satisfies

$$
\log N ( \varepsilon , \mathcal { C } _ { L , w } ( A , B ) , L ^ { 2 } ( P ) ) \leq C V \log \frac { C B } { \varepsilon } .\tag{45}
$$

For a finite $\mathcal { G } \subset [ - B , B ]$ , least squares under $Y = f ^ { \star } ( X ) + \xi , \xi \sim N ( 0 , \sigma ^ { 2 } )$ , obeys

$$
\mathbb { E } \left\| \widehat { g } - f ^ { \star } \right\| _ { 2 } ^ { 2 } \leq C \operatorname* { i n f } _ { g \in \mathcal { G } } \left\| g - f ^ { \star } \right\| _ { 2 } ^ { 2 } + C \frac { \left( \sigma ^ { 2 } + B ^ { 2 } \right) \log ( 2 | \mathcal { G } | ) } { n } .\tag{46}
$$

The Gaussian multiplier/Bernstein argument applies directly to unbounded Gaussian responses. Taking a population $- L ^ { 2 } \varepsilon { \mathrm { - n e t } }$ , using (45), and optimizing ε yields Theorem 5; Appendix G gives the full proof. Together with Corollary 4, Theorem 5 fixes the polynomial depth exponent. We next interpret the resulting local entropy bound and the competition among the output, testing, and representation scales.

## 9 CONSEQUENCES AND OPEN DIRECTIONS

The packing also yields an explicit localized covering lower bound. At the amplitude selected in (41), all codewords lie in an $L ^ { 2 }$ ball of radius $C \lambda .$ , distinct codewords are at least cλ apart, and

$$
\log N { \bigl ( } c \lambda , { \mathcal { C } } _ { L , w } ( A , B ) \cap \{ f : \| f \| _ { 2 } \leq C \lambda \} , L ^ { 2 } { \bigr ) } \geq c M = \Omega ( L ^ { 2 } w ^ { 2 } \log w ) .
$$

Hence the localized entropy at the testing scale is quadratic in depth, precluding a uniform coveringentropy upper bound of order $O ( L w ^ { 2 } )$ in the statistical regime. This local statement complements sharp entropy results for shallow neural variation spaces (Siegel and Xu, 2024) and the general entropy–minimax correspondence (Yang and Barron, 1999).

## 9.1 Radius dependence of the lower bound

Theorem 3 can be read as a competition among three amplitudes:

$$
\lambda _ { \mathrm { o u t } } = B , \qquad \lambda _ { \mathrm { s t a t } } = \sigma \sqrt { M / n } , \qquad \lambda _ { \mathrm { r e p } } = \frac { 1 } { M } \left( \frac { ( A - C _ { 0 } ) _ { + } } { C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { D } .\tag{47}
$$

If $\lambda _ { \mathrm { o u t } }$ is smallest, the lower bound saturates at the output-diameter scale $B ^ { 2 }$ . If $\lambda _ { \mathrm { s t a t } }$ is smallest, ordinary Gaussian testing is active and the risk is $\Omega ( \sigma ^ { 2 } M / n )$ . Under $A = B = R$ and $\sigma \asymp R ,$ , this is the quadratic-depth regime of Corollary 4, namely the statistical regime.

If $\lambda _ { \mathrm { r e p } }$ is smallest, the same construction gives

$$
\Re _ { n } ^ { * } ( A , B , \sigma ) \gtrsim \frac { 1 } { M ^ { 2 } } \left( \frac { ( A - C _ { 0 } ) _ { + } } { C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { 2 D } .\tag{48}
$$

Equation (48) identifies the representation-limited scale at which the layer-sum budget prevents the bit-extraction code from reaching the Gaussian testing scale. The transition is structural: the block Lipschitz estimate gives

$$
\mathrm { L i p } ( f ) \leq \prod _ { \ell = 1 } ^ { L } \| s _ { \ell } \| _ { \mathcal { R } \mathrm { B V } ^ { 2 } } \leq \left( \frac { A } { L } \right) ^ { L } ,\tag{49}
$$

so a small ratio $A / L$ can collapse the class exponentially with depth. The radius transition therefore reflects the geometry of the class rather than only the proof technique.

## 9.2 Limitations and open questions

Several questions remain. The lower bound contains log w, whereas the upper bound contains log(Lw) log n; closing these logarithmic factors may require a sharper localized upper bound or a larger packing. The bounded-coefficient approximation theorem assumes width and depth above universal thresholds, leaving the smallest-width cases open. Finite-precision parameter classes may exhibit a different transition because the bit-extraction mechanism uses real parameters at fine resolution.

The theorem concerns the explicit vector-valued Parhi–Nowak architecture consistent with the $O ( L w ^ { 2 } )$ parameterization in the motivating work; a literal scalar-to-scalar chain has only $O ( L w )$ parameters and defines a different minimax problem. A sharp A-dependent upper bound across all three regimes in (47) remains open. The local packing of entropy $\Omega ( L ^ { 2 } w ^ { \dot { 2 } } \log w )$ therefore survives the layer-sum variation constraint at the Gaussian testing scale. Under the radius condition of Corollary 4, the minimax risk has quadratic, rather than linear, polynomial dependence on depth, up to logarithmic factors.

## REPRODUCIBILITY STATEMENT

All assumptions and architecture conventions are stated explicitly in Sections 2–3, and complete proofs are provided in the appendices, including the corrected approximation reduction, balanced amplification and $\mathcal { R } \mathrm { B V } ^ { 2 }$ translation, and the Gaussian lower and upper bounds. The results are entirely analytical; no experiments or external datasets are used.

## GENERATIVE AI USE STATEMENT

Generative AI tools were used for literature search, formulation and critical checking of candidate mathematical claims, proof assistance, algebraic sanity checks, and manuscript editing. All AIassisted material was independently verified against primary sources or direct derivations. The authors take responsibility for the final content.

## REFERENCES

Martin Anthony and Peter L. Bartlett. Neural Network Learning: Theoretical Foundations. Cambridge University Press, 1999.

Francis Bach. Breaking the curse of dimensionality with convex neural networks. Journal of Machine Learning Research, 18(19):1–53, 2017.

Andrew R. Barron. Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information Theory, 39(3):930–945, 1993.

Peter L. Bartlett, Olivier Bousquet, and Shahar Mendelson. Local rademacher complexities. The Annals ofStatistics, 33(4):1497–1537, 2005. doi: 10.1214/009053605000000282.

Peter L. Bartlett, Nick Harvey, Christopher Liaw, and Abbas Mehrabian. Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. Journal of Machine Learning Research, 20(63):1–17, 2019.

Arkaprabha Ganguli and Emil Constantinescu. A function-space dichotomy for compositional learning: Exponential sub-optimality of the neural tangent kernel. arXiv preprint arXiv:2607.06382, 2026.

Noah Golowich, Alexander Rakhlin, and Ohad Shamir. Size-independent sample complexity of neural networks. In Proceedings of the 31st Conference on Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 297–299. PMLR, 2018.

David Haussler. Decision theoretic generalizations of the PAC model for neural net and other learning applications. Information and Computation, 100(1):78–150, 1992.

Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural networks. In Proceedings of the 28th Conference on Learning Theory, volume 40 of Proceedings ofMachine Learning Research, pages 1376–1401. PMLR, 2015.

Greg Ongie, Rebecca Willett, Daniel Soudry, and Nathan Srebro. A function space view of bounded norm infinite width ReLU nets: The multivariate case. In International Conference on Learning Representations, 2020.

Weigutian Ou and Helmut Bolcskei. Covering numbers for deep ReLU networks with applications to¨ function approximation and nonparametric regression. arXiv:2410.06378v2, 2026.

Weigutian Ou, Philipp Schenkel, and Helmut Bolcskei. Three quantization regimes for ReLU¨ networks. arXiv preprint arXiv:2405.01952, 2024.

Rahul Parhi and Robert D. Nowak. Banach space representer theorems for neural networks and ridge splines. Journal ofMachine Learning Research, 22(43):1–40, 2021. URL https://www. jmlr.org/papers/v22/20-583.html.

Rahul Parhi and Robert D. Nowak. What kinds of functions do deep neural networks learn? insights from variational spline theory. SIAM Journal on Mathematics of Data Science, 4(2):464–489, 2022. doi: 10.1137/21M1418642.

Rahul Parhi and Robert D. Nowak. Near-minimax optimal estimation with shallow ReLU neural networks. IEEE Transactions on Information Theory, 69(2):1125–1140, 2023. doi: 10.1109/TIT. 2022.3208653.

Rahul Parhi and Robert D. Nowak. Compositional function spaces for deep learning. SIAM Review, 68(1):127–149, 2026. doi: 10.1137/25M1802948.

Pedro Savarese, Itay Evron, Daniel Soudry, and Nathan Srebro. How do infinite width bounded norm networks look in function space? Conference on Learning Theory, pages 2667–2690, 2019.

Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with ReLU activation function. The Annals ofStatistics, 48(4):1875–1897, 2020.

Joseph Shenouda, Rahul Parhi, Kangwook Lee, and Robert D. Nowak. Variation spaces for multioutput neural networks: Insights on multi-task learning and network compression. Journal of Machine Learning Research, 25(231):1–40, 2024. URL https://www.jmlr.org/papers/ v25/23-0677.html.

Jonathan W. Siegel and Jinchao Xu. Sharp bounds on the approximation rates, metric entropy, and n-widths of shallow neural networks. Foundations of Computational Mathematics, 24(2):481–537, 2024. doi: 10.1007/s10208-022-09595-3.

Taiji Suzuki. Adaptivity of deep ReLU network for learning in Besov and mixed smooth Besov spaces: Optimal rate and curse of dimensionality. International Conference on Learning Representations, 2019.

Matus Telgarsky. Benefits of depth in neural networks. In Proceedings of the 29th Conference on Learning Theory, volume 49 of Proceedings ofMachine Learning Research, pages 1517–1539. PMLR, 2016.

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer, 2009.

Yuhong Yang and Andrew R. Barron. Information-theoretic determination of minimax rates of convergence. The Annals ofStatistics, 27(5):1564–1599, 1999.

Dmitry Yarotsky. Error bounds for approximations with deep ReLU networks. Neural Networks, 94: 103–114, 2017. doi: 10.1016/j.neunet.2017.07.002.

Dmitry Yarotsky. Optimal approximation of continuous functions by very deep ReLU networks. In Proceedings of the 31st Conference on Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 639–649. PMLR, 2018.

## A ARCHITECTURE CONVENTIONS AND COMPUTATION-GRAPH REALIZATION

## A.1 Vector-valued class and relation to the motivating formulation

For $\begin{array} { r } { s ( \boldsymbol { x } ) = \sum _ { k = 1 } ^ { K } { v _ { k } \rho ( w _ { k } ^ { \top } x - b _ { k } ) } + C x + c _ { 0 } } \end{array}$ , we use (8). The Parhi–Nowak deep space composes vector-valued maps across dimensions $d _ { 0 } , \ldots , d _ { L }$ (Parhi and Nowak, 2022). The motivating work uses an $O ( L w ^ { 2 } )$ parameter count, while its displayed base-block formula does not explicitly display these dimensions (Ganguli and Constantinescu, 2026). Throughout this paper, all class inclusions refer to the explicit architecture

$$
d _ { 0 } = d _ { L } = 1 , \qquad d _ { \ell } \leq w , \qquad K _ { \ell } \leq w .
$$

## A.2 Exact parameter count

A block with input dimension d, output dimension $D ^ { \prime } .$ , and K atoms has

$$
D ^ { \prime } K + K d + K + D ^ { \prime } d + D ^ { \prime }
$$

scalar parameters. If $d , D ^ { \prime } , K \leq w ,$ , this is at most $3 w ^ { 2 } + 2 w$ . Hence

$$
W _ { \mathrm { p a r } } \leq L ( 3 w ^ { 2 } + 2 w ) .\tag{50}
$$

## A.3 Proof of Lemma 9

For each block, compute the K preactivations $\boldsymbol { w _ { k } ^ { \intercal } x - b _ { k } }$ in one affine stage, apply K ReLUs, and compute $\begin{array} { r } { \sum _ { k } v _ { k } \rho ( \cdot ) \stackrel { \textstyle - } { + } C x + c _ { 0 } } \end{array}$ in a second affine stage. The skip Cx is carried in parallel through the same block and adds no nonlinear depth. Thus an L-block composition is a piecewise-linear computation graph with at most the parameters in (50), at most Lw ReLU gates, and computational depth at most 2L.

If a theorem is stated for ordinary feed-forward ReLU networks without affine skips, represent a scalar u by $( \rho ( u ) , \rho ( - u ) )$ ) and propagate both signs. This converts every affine skip into a constant-factor larger ReLU graph, preserving $O ( L w ^ { 2 } )$ parameters and $O ( L )$ computational depth. The finitely many choices of dimensions and atom counts are all subarchitectures of the maximal padded graph, obtained by setting unused parameters to zero.

The piecewise-linear pseudodimension bound of Bartlett et al. (2019), applied to this maximal graph, gives

$$
\mathrm { P d i m } \le C W _ { \mathrm { p a r } } L \log ( 2 W _ { \mathrm { p a r } } ) \le C L ^ { 2 } w ^ { 2 } \log ( 2 L w ) .
$$

The layer-cost and output constraints only restrict the graph class and cannot increase pseudodimension.

## A.4 Circle coordinate and representation cost

The coordinate $t \in [ 0 , 2 )$ is the motivating coordinate $t = \theta / \pi$ with normalized measure $\mathrm { d } t / 2$ . The tent map (34) satisfies $r ( 0 ) = r ( 2 ) = 0$ , so the packed functions periodize continuously.

The infimum (10) is generally not one-homogeneous as a functional of the composed map: a scalar gain can be distributed across D stages at cost proportional to $D q ^ { 1 / D }$ . Accordingly, all arguments use only the representation-cost definition.

## B A BIAS-CORRECTED BOUNDED-COEFFICIENT APPROXIMATION THEOREM

This appendix proves Theorem 2. The bit decoder and approximation estimates are imported from the constructive proof of Ou et al. (2024). The coefficient-scaling steps needed to pass from that construction to the unit-coefficient theorem are rederived below under the affine-layer convention (13).

## B.1 The affine-bias obstruction and its repair

Uniformly multiplying every affine pair $( A _ { \ell } , b _ { \ell } )$ by $a > 0$ scales homogeneous terms and biases by different powers across depth. For example, the depth-two realization

$$
N ( x ) = \rho ( x ) + 1
$$

becomes $a ^ { 2 } \rho ( x ) + a ,$ not $a ^ { 2 } N ( x )$ . The last-layer bias receives only one factor of $a , \textrm { \textbf { A } }$ constant homogeneous coordinate repairs this mismatch.

Lemma 10 (Homogeneous lift). Let $B \geq 1$ and let $N \in \mathcal { N } ( W , D , B )$ have exact depth $D \geq 2 .$ . For every $q > 0$

$$
q N \in \mathcal { N } ( W + 1 , D , B q ^ { 1 / D } ) .\tag{51}
$$

Proof. Put $s = q ^ { 1 / D }$ and write the hidden states as in (13). Define

$$
\widetilde { h } _ { 1 } = \rho \bigg ( \binom { s A _ { 1 } } { 0 } x + \binom { s b _ { 1 } } { s } \bigg ) = \binom { s h _ { 1 } } { s } ,\tag{52}
$$

and, for $2 \le \ell < D$

$$
\widetilde { h } _ { \ell } = \rho \bigg ( \bigg [ \begin{array} { c c } { s A _ { \ell } } & { s b _ { \ell } } \\ { 0 } & { s } \end{array} \bigg ] \widetilde { h } _ { \ell - 1 } \bigg ) .\tag{53}
$$

Induction gives $\widetilde { h } _ { \ell } = ( s ^ { \ell } h _ { \ell } , s ^ { \ell } )$ . The final affine map is

$$
\widetilde { N } ( x ) = [ s A _ { D } \quad s b _ { D } ] \widetilde { h } _ { D - 1 } = s ^ { D } ( A _ { D } h _ { D - 1 } + b _ { D } ) = q N ( x ) .\tag{54}
$$

The width increases by one. Every old coefficient is multiplied by s, and the only new nonzero coefficient is $s ;$ since $\dot { B } \geq 1$ , the new coefficient magnitude is at most $s B$ □

## B.2 Trading coefficient magnitude for depth

We next use a unit-coefficient fan-out construction.

Lemma 11 (Unit-coefficient fan-out). Let $W \ \geq \ 2 ,$ , let $N \in \mathcal { N } ( W , D , 1 )$ , and let $J \ge 0$ . Put $r = \lfloor W / 2 \rfloor$ . Then

$$
r ^ { J } N \in \mathcal { N } ( W , D + J , 1 ) .\tag{55}
$$

Proof. The case $J = 0$ is immediate. For $J \geq 1$ , pad the input realization to exact depth $D$ and write its scalar output as $y = A _ { D } h _ { D - 1 } + b _ { D }$ . Replace this final affine map by the hidden vector

$$
u _ { 1 } = ( \rho ( y ) \mathbf { 1 } _ { r } , \rho ( - y ) \mathbf { 1 } _ { r } ) \in \mathbb { R } ^ { 2 r } .
$$

For each of the next $J - 1$ hidden layers, apply $\mathrm { d i a g } ( \mathbf { 1 } _ { r \times r } , \mathbf { 1 } _ { r \times r } )$ with zero bias. The final row $( \mathbf { 1 } _ { r } ^ { \top } , - \mathbf { 1 } _ { r } ^ { \top } )$ returns $r ^ { J } ( \rho ( y ) - \rho ( - y ) ) \stackrel { , } { = } r ^ { J } y$ . All coefficients belong to $\{ - 1 , 0 , 1 \}$ and $2 r \leq W$ □

Corollary 12 (Bias-correct depth–coefficient conversion). Let $W \geq 4 , D \geq 2$ , and $B \geq 1$ . Put

$$
r = \left\lfloor { \frac { W + 1 } { 2 } } \right\rfloor , \qquad J = \left\lceil { \frac { D \log B } { \log r } } \right\rceil ,\tag{56}
$$

with $J = 0$ when $B = 1$ . Then

$$
\mathcal { N } ( W , D , B ) \subseteq \mathcal { N } ( W + 1 , D + J , 1 ) .\tag{57}
$$

If $B = W ^ { K }$ for fixed K, then $J \le 2 K D + 1$

Proof. For $N \in \mathcal { N } ( W , D , B )$ , Lemma 10 with $q = B ^ { - D }$ gives $B ^ { - D } N \in \mathcal { N } ( W + 1 , D , 1 )$ . Apply Lemma 11 at width $W + 1$ to obtain $r ^ { J } B ^ { - D } N$ . Since $r ^ { J } \geq B ^ { D }$ , multiplying the final affine map by $B ^ { D } / r ^ { J } \leq 1$ recovers N without violating the unit coefficient bound. Finally, $r ^ { 2 } \geq W$ for $W \bar { \geq { 4 } }$ so log $r \geq { \frac { 1 } { 2 } } \log W$ and the stated bound on J follows. □

## B.3 Repairing the bounded-spline realization

For a strictly increasing breakpoint sequence $X = ( x _ { i } ) _ { i = 0 } ^ { M _ { X } - 1 } \subset [ 0 , 1 ]$ , let $\Sigma ( X , E )$ denote the continuous functions that are constant outside $[ x _ { 0 } , x _ { M _ { X } - 1 } ] _ { 0 }$ , affine between consecutive breakpoints, and bounded in absolute value by E. Put

$$
R _ { m } ( X ) = \operatorname* { m a x } _ { 1 \leq i \leq M _ { X } - 1 } ( x _ { i } - x _ { i - 1 } ) ^ { - 1 } .
$$

The direct construction of Ou et al. (2024, Proposition C.1), which is independent of the bias-sensitive layerwise scaling identity discussed above, implies that whenever $u ^ { 2 } v \geq M _ { X }$

$$
\Sigma \biggl ( X , \frac { 1 } { C _ { k } M _ { X } ^ { 6 } R _ { m } ( X ) ^ { 4 } } \biggr ) \subseteq N ( 2 0 u , 3 0 v , 1 ) ,\tag{58}
$$

where $2 \le C _ { k } \le 1 0 ^ { 5 }$ is universal.

Lemma 13 (Bias-correct bounded-spline realization). Assume $u ^ { 2 } v \ge M _ { X } , v \ge 1 , w _ { s } \ge 1$ , and

$$
w _ { s } ^ { 3 0 v } \geq M _ { X } ^ { 6 } R _ { m } ( X ) ^ { 4 } E .\tag{59}
$$

Then

$$
\Sigma ( X , E ) \subseteq \mathcal { N } ( 2 0 u + 1 , 3 0 v , 2 w _ { s } ) .\tag{60}
$$

Proof. For $f \in \Sigma ( X , E )$ , set $f _ { 0 } = ( 2 w _ { s } ) ^ { - 3 0 v } f$ . By (59) and $2 ^ { 3 0 v } \geq 2 ^ { 3 0 } > 1 0 ^ { 5 } \geq C _ { k }$ ,

$$
\| f _ { 0 } \| _ { \infty } \leq \frac { 1 } { C _ { k } M _ { X } ^ { 6 } R _ { m } ( X ) ^ { 4 } } .
$$

Thus $f _ { 0 }$ belongs to the left side of (58). Lemma 10, with $q = ( 2 w _ { s } ) ^ { 3 0 v }$ , realizes $( 2 w _ { s } ) ^ { 3 0 v } f _ { 0 } = f$ at the same depth, width at most $2 0 u + 1$ , and coefficient magnitude at most $2 w _ { s }$ □

## B.4 Width and depth bookkeeping

The remaining decoder identities and approximation estimates in Ou et al. (2024, Proposition B.1) are explicit and independent of the bias-sensitive affine scaling discussed above. Its bounded-spline proposition is called exactly twice, at equations (144) and (161) of the cited proof. Replacing those calls by Lemma 13 changes the two spline widths from 40m and 40n to $4 0 m + 1$ and $4 0 n + 1$ , while preserving the realized functions, depths, and coefficient bounds.

The resulting additive width changes fit inside the slack of the original construction. In its first stage, the corrected parallelization and affine-combination bounds give

$$
\mathrm { w i d t h } ( f _ { 1 } ) \leq 2 0 0 m + 2 ^ { n + 5 } + 5 .
$$

In the second stage,

$$
\operatorname { w i d t h } ( u ) \leq \operatorname* { m a x } \{ 4 0 m + 1 , 4 0 n + 1 \} + 2 .
$$

For $m , n \geq 2$ , the first display dominates the second, so the composition of $f _ { 1 }$ with u retains the first width bound. Three parallel copies followed by the fixed median network have width at most

$$
6 0 0 m + 3 \cdot 2 ^ { n + 5 } + 1 5 \leq 6 0 0 m + 2 ^ { n + 7 } .
$$

Consequently the terminal architecture estimate of the cited bit-extraction proof remains valid without changing its asymptotic or stated width bound:

Proposition 14 (Polynomial-coefficient approximation). There exist universal constants $C , D _ { 1 } > 0$ such that, for all integers W, $U \geq D _ { 1 }$ and every continuous $g : [ 0 , 1 ]  \mathbb { R }$ with $\| g \| _ { \infty } \leq 1$ and $\operatorname { L i p } ( g ) \leq 1$ , there is $N \in \mathcal { N } ( W , U , W ^ { 2 } )$ satisfying

$$
\| N - g \| _ { \infty } \leq \frac { C } { W ^ { 2 } U ^ { 2 } \log W } .\tag{61}
$$

Proof. The corrected construction just described yields, for integers $m , n , \ell \geq 2$

$$
\begin{array} { l } { { \mathrm { w i d t h } ( N ) \leq 6 0 0 m + 2 ^ { n + 7 } , } } \\ { { \mathrm { d e p t h } ( N ) \leq 1 0 1 \ell , } } \\ { { \mathrm { c o e f f } ( N ) \leq \operatorname* { m a x } \{ 8 m n , 3 ^ { n + 2 } \} , } } \\ { { \| N - g \| _ { \infty } \leq \frac { 3 } { m ^ { 2 } \ell ^ { 2 } n } . } } \end{array}\tag{62}
$$

Choose

$$
m = \left\lfloor { \frac { W } { 1 0 0 0 } } \right\rfloor , \qquad \ell = \left\lfloor { \frac { U } { 1 0 1 } } \right\rfloor ,
$$

and let $n \geq 2$ be the largest integer such that $2 ^ { n + 7 } \leq W / 5$ . For sufficiently large $W , U$ , one has $m \asymp W , \ell \asymp U$ , and $n \asymp$ log W. Moreover,

$$
6 0 0 m + 2 ^ { n + 7 } \leq { \frac { 4 } { 5 } } W , \qquad 1 0 1 \ell \leq U ,
$$

and

$$
8 m n \leq W ^ { 2 } , \qquad 3 ^ { n + 2 } \leq W ^ { 2 }
$$

for all sufficiently large $W$ . Padding unused width and depth places the network in $\mathcal { N } ( W , U , W ^ { 2 } )$ . Substituting the parameter choices into (62) proves (61). □

## B.5 Proof of Theorem 2

Let $m , D$ be sufficiently large, set ${ \overline { { m } } } = m - 1$ , and put

$$
U = \left\lfloor { \frac { D - 1 } { 5 } } \right\rfloor .
$$

Proposition 14 gives a network $N _ { 0 } \in \mathcal { N } ( \overline { { m } } , U , \overline { { m } } ^ { 2 } )$ with error at most $C / ( \overline { { m } } ^ { 2 } U ^ { 2 } \log \overline { { m } } )$ . Apply Corollary 12. With $r = \lfloor m / 2 \rfloor$ and $r ^ { 2 } \geq$ m for sufficiently large m, the additional depth obeys

$$
J = \left\lceil { \frac { U \log ( { \overline { { m } } } ^ { 2 } ) } { \log r } } \right\rceil \leq 4 U + 1 .
$$

Hence the same function has a unit-coefficient realization of width at most m and depth at most $U + J \leq 5 U + 1 \leq D$ . Since $\overline { { m } } \asymp m$ and $U \asymp D$ , the error is at most $C _ { \mathrm { a p p } } / ( m ^ { 2 } D ^ { 2 } \log \stackrel { . } { m } )$ ), proving Theorem 2.

The fine-scale bit decoder and its approximation estimate are taken from Ou et al. (2024). The coefficient-rescaling steps used to obtain the unit-coefficient realization have been rederived above under the affine-layer convention in (13).

## C THE GRID CODE AND ITS LOCAL GEOMETRY

Let $M \geq 8 .$ . For $z \in \{ 0 , 1 \} ^ { M - 1 }$ , set $z _ { 0 } = z _ { M } = 0$ and let $h _ { z }$ linearly interpolate $( j / M , z _ { j } )$ . On $I _ { j } = [ j / M , ( j + 1 ) / M ]$ , writing $u = M x - j$ and $d _ { j } = z _ { j } - z _ { j } ^ { \prime }$ , we have

$$
h _ { z } ( x ) - h _ { z ^ { \prime } } ( x ) = ( 1 - u ) d _ { j } + u d _ { j + 1 } .
$$

Therefore

$$
\begin{array} { l } { { \displaystyle \int _ { I _ { j } } ( h _ { z } - h _ { z ^ { \prime } } ) ^ { 2 } \mathrm { d } x = \frac { 1 } { M } \int _ { 0 } ^ { 1 } ( ( 1 - u ) d _ { j } + u d _ { j + 1 } ) ^ { 2 } \mathrm { d } u } } \\ { { \displaystyle \qquad = \frac { d _ { j } ^ { 2 } + d _ { j } d _ { j + 1 } + d _ { j + 1 } ^ { 2 } } { 3 M } . } } \end{array}\tag{63}
$$

(64)

Since $\textstyle a ^ { 2 } + a b + b ^ { 2 } = { \frac { 1 } { 2 } } ( a ^ { 2 } + b ^ { 2 } ) + { \frac { 1 } { 2 } } ( a + b ) ^ { 2 }$ , summing and using $d _ { 0 } = d _ { M } = 0$ yields

$$
\left. h _ { z } - h _ { z ^ { \prime } } \right. _ { 2 } ^ { 2 } \geq \frac { d _ { H } ( z , z ^ { \prime } ) } { 3 M } .\tag{65}
$$

The Varshamov–Gilbert bound (Tsybakov, 2009) gives $\mathcal { Z } \subseteq \{ 0 , 1 \} ^ { M - 1 }$ and universal $c _ { \mathrm { V G } } , c _ { H } > 0$ with

$$
\begin{array} { r } { \log | \mathcal { Z } | \geq c _ { \mathrm { V G } } M , \qquad d _ { H } ( z , z ^ { \prime } ) \geq c _ { H } M \quad ( z \neq z ^ { \prime } ) . } \end{array}\tag{66}
$$

Now $g _ { z } ~ = ~ h _ { z } / M$ satisfies $\| g _ { z } \| _ { \infty } \leq 1$ and $\mathrm { L i p } ( g _ { z } ) \leq 1$ . Theorem 2 gives $N _ { z }$ . With $M =$ $\lfloor c _ { 0 } m ^ { 2 } D ^ { 2 } \log m \rfloor$ and $c _ { 0 } \leq \eta / ( 2 C _ { \mathrm { a p p } } )$

$$
\| Q _ { z } - h _ { z } \| _ { \infty } = M \| N _ { z } - g _ { z } \| _ { \infty } \leq \eta , \qquad Q _ { z } = M N _ { z } .
$$

Thus, for fixed $\eta < \sqrt { c _ { H } / 3 } / 4$

$$
\begin{array} { r l } & { \| Q _ { z } - Q _ { z ^ { \prime } } \| _ { 2 } \geq \sqrt { c _ { H } / 3 } - 2 \eta , } \\ & { \| Q _ { z } - Q _ { z ^ { \prime } } \| _ { 2 } \leq 1 + 2 \eta , \qquad \| Q _ { z } \| _ { \infty } \leq 1 + \eta . } \end{array}
$$

This proves (31) and locality after scaling by λ.

## D BALANCED AMPLIFICATION FOR THE STATISTICAL CODE

Lemma 7 is the coefficient-one specialization of Lemma 10; we repeat the matrices to make the lower-bound dependency self-contained.

## D.1 Exact-depth padding

If an approximating network has depth $d < D$ , insert $D - d$ identity hidden layers immediately before the final affine map. The hidden state is nonnegative, hence $\boldsymbol { h } \mapsto \rho ( I \boldsymbol { h } ) \dot { = } \boldsymbol { h }$ . These layers have coefficient magnitude one and do not increase width.

## D.2 Exact amplification matrices

Write

$$
h _ { 1 } = \rho ( A _ { 1 } x + b _ { 1 } ) , \quad h _ { \ell } = \rho ( A _ { \ell } h _ { \ell - 1 } + b _ { \ell } ) , \quad N = A _ { D } h _ { D - 1 } + b _ { D } .
$$

For $q > 0 ;$ , put $s = q ^ { 1 / D }$ and define

$$
\widetilde { h } _ { 1 } = \rho \bigg ( \binom { s A _ { 1 } } { 0 } x + \binom { s b _ { 1 } } { s } \bigg ) = \binom { s h _ { 1 } } { s } ,\tag{67}
$$

$$
\widetilde { h } _ { \ell } = \rho \bigg ( \left[ \begin{array} { l l } { s A _ { \ell } } & { s b _ { \ell } } \\ { 0 } & { s } \end{array} \right] \widetilde { h } _ { \ell - 1 } \bigg ) = \left[ \begin{array} { l } { s ^ { \ell } h _ { \ell } } \\ { s ^ { \ell } } \end{array} \right] ,\tag{68}
$$

for $2 \le \ell < D$ , and

$$
\widetilde { N } = [ s A _ { D } s b _ { D } ] \widetilde { h } _ { D - 1 } = s ^ { D } N = q N .\tag{69}
$$

The width increases by one, depth is unchanged, and the coefficient bound is s, including when $q < 1$

## E TRANSLATION INTO DEEP $\mathcal { R } \mathrm { B V } ^ { 2 }$ BLOCKS

## E.1 Coordinatewise ReLU layers

Let $T ( x ) = \rho ( A x + b )$ , with rows $a _ { i } ^ { \top }$ of A. In the block notation (6),

$$
T ( x ) = \sum _ { i = 1 } ^ { d ^ { \prime } } e _ { i } \rho ( a _ { i } ^ { \top } x - ( - b _ { i } ) ) ,
$$

so $K = d ^ { \prime }$ and the variation term is

$$
\sum _ { i = 1 } ^ { d ^ { \prime } } \| e _ { i } \| _ { 1 } \left\| a _ { i } \right\| _ { 2 } = \sum _ { i = 1 } ^ { d ^ { \prime } } \| a _ { i } \| _ { 2 } \leq d ^ { \prime } { \sqrt { d } } s .
$$

For the anchor term,

$$
| T _ { i } ( 0 ) | = | \rho ( b _ { i } ) | \leq s
$$

and, since ReLU is 1-Lipschitz,

$$
| T _ { i } ( e _ { j } ) - T _ { i } ( 0 ) | \leq | a _ { i j } | \leq s .
$$

Hence

$$
\| T \| _ { \mathcal { R } \mathrm { B V } ^ { 2 } ( d ; d ^ { \prime } ) } \leq d ^ { \prime } \sqrt { d } s + d ^ { \prime } ( d + 1 ) s \leq 3 w ^ { 2 } s
$$

for $d , d ^ { \prime } \leq w$ and $w \ge 1$

For an affine map $T ( x ) = A x + b ,$ take $K = 0 , C = A$ , and $c _ { 0 } = b .$ . Then

$$
\| T \| _ { \mathcal { R } \mathrm { B V } ^ { 2 } ( d ; d ^ { \prime } ) } = \sum _ { i = 1 } ^ { d ^ { \prime } } \left( | b _ { i } | + \sum _ { j = 1 } ^ { d } | a _ { i j } | \right) \leq d ^ { \prime } ( d + 1 ) s \leq 2 w ^ { 2 } s .
$$

Applying these estimates to the $D - 1$ hidden maps and final affine map in Appendix D proves

$$
\mathfrak { D } _ { D , m + 1 } ( q N ) \leq C _ { \mathrm { t r } } D ( m + 1 ) ^ { 2 } q ^ { 1 / D }
$$

with, for example, $C _ { \mathrm { t r } } = 3$ under the displayed conventions.

## E.2 Tent map and circle isometry

The map $r ( t ) = t - 2 \rho ( t - 1 )$ is represented with one ReLU atom and one affine skip. Its norm is

$$
\| r \| _ { \mathcal { R } \mathrm { B V } ^ { 2 } ( 1 ; 1 ) } = | - 2 | | 1 | + | r ( 0 ) | + | r ( 1 ) - r ( 0 ) | = 3 .
$$

Thus one may take $C _ { 0 } = 3$ . It satisfies

$$
r ( t ) = t \quad ( 0 \leq t \leq 1 ) , \qquad r ( t ) = 2 - t \quad ( 1 \leq t \leq 2 ) .
$$

For any measurable $u : [ 0 , 1 ] \to \mathbb { R }$

$$
\begin{array} { l } { \displaystyle \int _ { 0 } ^ { 2 } | u ( r ( t ) ) | ^ { 2 } \frac { \mathrm { d } t } { 2 } = \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } | u ( t ) | ^ { 2 } \mathrm { d } t + \frac { 1 } { 2 } \int _ { 1 } ^ { 2 } | u ( 2 - t ) | ^ { 2 } \mathrm { d } t } \\ { \displaystyle \qquad = \int _ { 0 } ^ { 1 } | u ( x ) | ^ { 2 } \mathrm { d } x . } \end{array}
$$

This proves (36). Combining the tent block with the $D$ translated blocks gives total depth $D + 1 = L$ and width at most m $+ 1 = w$

## E.3 Membership of the packed functions

For $F _ { z } = \lambda M N _ { z }$ ◦ r, Lemma 7 and the preceding block calculation give

$$
\mathfrak { D } _ { L , w } ( F _ { z } ) \le C _ { 0 } + C _ { \mathrm { t r } } D w ^ { 2 } ( \lambda M ) ^ { 1 / D } .
$$

Also

$$
\| F _ { z } \| _ { \infty } \leq \lambda ( 1 + \eta ) .
$$

Therefore $F _ { z } \in \mathcal { C } _ { L , w } ( A , B )$ whenever

$$
\lambda \leq \frac { B } { 1 + \eta } , \qquad \lambda \leq \frac { 1 } { M } \left( \frac { ( A - C _ { 0 } ) _ { + } } { C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { D } .\tag{70}
$$

Universal constant losses in these inequalities are absorbed by $c _ { \lambda }$ in (41).

## F FANO PROOF OF THE LOWER BOUND

Let Z be the code from Appendix $\mathrm { C } ,$ and define $F _ { z }$ by (35). There are universal constants $a _ { 0 } , a _ { 1 } , a _ { 2 } >$ 0 such that

$$
\log | \mathcal { Z } | \geq a _ { 0 } M , \qquad a _ { 1 } \lambda \leq \| F _ { z } - F _ { z ^ { \prime } } \| _ { 2 } \leq a _ { 2 } \lambda , \qquad \| F _ { z } \| _ { 2 } \leq a _ { 2 } \lambda .\tag{71}
$$

Choose

$$
\lambda = c _ { \lambda } \operatorname* { m i n } \left\{ B , \sigma \sqrt { \frac { M } { n } } , \frac { 1 } { M } \left( \frac { ( A - C _ { 0 } ) _ { + } } { C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { D } \right\} ,
$$

where $c _ { \lambda }$ is sufficiently small for (70). Under the n-sample law $P _ { z } ^ { ( n ) }$ , the design law is common and the conditional responses are independent Gaussians of variance $\sigma ^ { 2 }$ . Hence

$$
D _ { \mathrm { K L } } ( P _ { z } ^ { ( n ) } \| P _ { z ^ { \prime } } ^ { ( n ) } ) = n \mathbb { E } _ { T } D _ { \mathrm { K L } } \big ( N ( F _ { z } ( T ) , \sigma ^ { 2 } ) \| N ( F _ { z ^ { \prime } } ( T ) , \sigma ^ { 2 } ) \big )\tag{72}
$$

$$
= \frac { n } { 2 \sigma ^ { 2 } } \left\| F _ { z } - F _ { z ^ { \prime } } \right\| _ { 2 } ^ { 2 } \leq \frac { a _ { 2 } ^ { 2 } c _ { \lambda } ^ { 2 } } { 2 } M .\tag{73}
$$

Choose $c _ { \lambda }$ so the last term is at most $\left( 1 / 1 6 \right) \log | \mathcal { Z } |$ . With the uniform prior, mutual information is bounded by the average pairwise divergence, and Fano’s inequality (Tsybakov, 2009) gives a universal lower bound on the worst-case codeword error probability.

Given an arbitrary estimator ${ \widehat { f } } ,$ decode by nearest neighbor in $L ^ { 2 } ( \mu )$ . Whenever $\left\| \widehat { f } - F _ { z } \right\| _ { 2 } < a _ { 1 } \lambda / 2$ the decoded index is $z .$ Markov’s inequality therefore yields

$$
\underset { z \in \mathcal { Z } } { \operatorname* { s u p } } \mathbb { E } _ { z } \left. \widehat { f } - F _ { z } \right. _ { 2 } ^ { 2 } \geq c \lambda ^ { 2 } .
$$

Taking the infimum over estimators proves Theorem $^ { 3 , }$

## F.1 The sample-size-dependent radius corollary

Fix constants $0 < c _ { \sigma } \le C _ { \sigma } < \infty$ , and let $A = B = R , c _ { \sigma } R \leq \sigma \leq C _ { \sigma } R , n \geq M$ , and $R \geq 2 C _ { 0 }$ The output cap is at least a constant multiple of $R \sqrt { M / n }$ . Further,

$$
\lambda _ { \mathrm { r e p } } \geq \frac { 1 } { M } \left( \frac { R } { 2 C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { D } .
$$

Condition (17), with $C _ { \mathrm { r a d } }$ sufficiently large inside the base of the Dth power, implies $\lambda _ { \mathrm { { r e p } } } \geq$ cR $\sqrt { M / n }$ . Therefore $\lambda ^ { 2 } \asymp R ^ { 2 } M / n$ and

$$
\Re _ { n } ^ { * } ( R , R , \sigma ) \geq c R ^ { 2 } M / n .
$$

For L, w above absolute thresholds, $D = L - 1 , m = w - 1$ , and (15) imply $M \geq c L ^ { 2 } w ^ { 2 }$ log w.   
This proves Corollary 4.

Since $M ^ { 3 / 2 } / \sqrt { n } \leq M$ for every $n \geq M ,$ , condition (19) is sufficient uniformly over that sample-size range. Taking $( D - 1 )$ )st roots gives (20).

## G PROOF OF THE GAUSSIAN UPPER BOUND

## G.1 Pseudodimension and covering numbers

The computation graph described in Appendix A has $W _ { \mathrm { p a r } } = O ( L w ^ { 2 } )$ real parameters, $O ( L w )$ piecewise-linear units, and computational depth $O ( L )$ . The piecewise-linear network theorem of Bartlett et al. (2019) therefore yields

$$
\mathrm { P d i m } ( \mathcal { C } _ { L , w } ( A , B ) ) \leq C W _ { \mathrm { p a r } } L \log ( 2 W _ { \mathrm { p a r } } ) \leq C L ^ { 2 } w ^ { 2 } \log ( 2 L w ) .\tag{74}
$$

The norm and output constraints only restrict the unconstrained architecture and hence cannot increase pseudodimension.

A standard pseudodimension covering theorem (Haussler, 1992; Anthony and Bartlett, 1999) states that for every probability measure $P$ and every $[ - B , B ]$ ]-valued class of pseudodimension at most $V ,$

$$
\log N ( \varepsilon , \mathcal { F } , L ^ { 2 } ( P ) ) \leq C V \log \frac { C B } { \varepsilon } , \qquad 0 < \varepsilon \leq B .\tag{75}
$$

## G.2 A finite-class Gaussian oracle inequality

Lemma 15. Let G be a finite class offunctions $g : \mathcal { X }  [ - B , B ]$ , and suppose $Y = f ^ { \star } ( X ) + \xi$ with $| f ^ { \star } | \le B$ and $\xi \sim \overset { \cdot } { N } ( 0 , \sigma ^ { 2 } )$ independent ofX. Let

$$
{ \widehat { g } } \in \arg \operatorname* { m i n } _ { g \in { \mathcal { G } } } { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } ( Y _ { i } - g ( X _ { i } ) ) ^ { 2 } .
$$

Then

$$
\mathbb { E } \left\| \widehat { g } - f ^ { \star } \right\| _ { L ^ { 2 } ( P _ { X } ) } ^ { 2 } \leq C \operatorname* { i n f } _ { g \in \mathcal { G } } \left\| g - f ^ { \star } \right\| _ { L ^ { 2 } ( P _ { X } ) } ^ { 2 } + C \frac { ( \sigma ^ { 2 } + B ^ { 2 } ) \log ( 2 | \mathcal { G } | ) } { n } .\tag{76}
$$

Proof. For $g \in { \mathcal { G } }$ , put $d _ { g } = g - f ^ { \star } , r _ { g } = P d _ { g } ^ { 2 }$ , and

$$
Z _ { i } ( g ) = d _ { g } ( X _ { i } ) ^ { 2 } - 2 \xi _ { i } d _ { g } ( X _ { i } ) , \qquad Z _ { n } ( g ) = \frac 1 n \sum _ { i = 1 } ^ { n } Z _ { i } ( g ) .
$$

Then $\mathbb { E } Z _ { i } ( g ) = r _ { g }$ , and empirical risk minimization implies $Z _ { n } ( \widehat { g } ) \leq Z _ { n } ( g )$ for every $g \in { \mathcal { G } }$

Because $| d _ { g } | \le 2 B$ , the centered random variable $Z _ { i } ( g ) - r _ { g }$ is sub-exponential with Bernstein variance proxy

$$
\nu _ { g } ^ { 2 } \leq C ( \sigma ^ { 2 } + B ^ { 2 } ) r _ { g }\tag{77}
$$

and scale at most $C ( \sigma B + B ^ { 2 } )$ . To see (77), use $\mathbb { E } d _ { g } ^ { 4 } \le 4 B ^ { 2 } r _ { g }$ for the design term and $\mathbb { E } ( 2 \xi d _ { g } ) ^ { 2 } =$ $4 \sigma ^ { 2 } r _ { g }$ for the multiplier term. The Gaussian conditional moment-generating function, together with $| d _ { g } | \le 2 B$ , gives the corresponding sub-exponential scale. Bernstein’s inequality and $2 \sqrt { u v } \ \leq$ $u / 2 + 2 v$ imply that, for each $u > 0 .$ , with probability at least $1 - 2 e ^ { - u }$

$$
\frac { 1 } { 2 } r _ { g } - C a \frac { u } { n } \leq Z _ { n } ( g ) \leq \frac { 3 } { 2 } r _ { g } + C a \frac { u } { n } , \qquad a = \sigma ^ { 2 } + B ^ { 2 } .\tag{78}
$$

Increasing $C$ absorbs the scale term because σ $\begin{array} { r } { \tau B + B ^ { 2 } \le C ( \sigma ^ { 2 } + B ^ { 2 } ) = C a . } \end{array}$

Apply the lower inequality in (78) simultaneously to every $g \in { \mathcal { G } }$ with $u = t + \log ( 2 | \mathcal { G } | )$ , and apply the upper inequality to a fixed comparator $g _ { 0 }$ with $u = t$ . With probability at least $1 - 3 e ^ { - t }$

$$
\begin{array} { r l r } {  { \frac { 1 } { 2 } r _ { \widehat { \boldsymbol { g } } } \le Z _ { n } ( \widehat { \boldsymbol { g } } ) + C a \frac { t + \log ( 2 | \boldsymbol { \mathcal { G } } | ) } { n } } } \\ & { } & \\ & { } & { \le Z _ { n } ( g _ { 0 } ) + C a \frac { t + \log ( 2 | \boldsymbol { \mathcal { G } } | ) } { n } } \\ & { } & \\ & { } & { \le \frac { 3 } { 2 } r _ { g _ { 0 } } + C a \frac { 2 t + \log ( 2 | \boldsymbol { \mathcal { G } } | ) } { n } . } \end{array}
$$

Thus

$$
r _ { \widehat { g } } \leq 3 r _ { g _ { 0 } } + C a \frac { 2 t + \log ( 2 | \mathcal { G } | ) } { n } .
$$

Integrating the exponential tail over $t \geq 0$ and minimizing over g<sub>0</sub> proves (76).

## G.3 Completion of Theorem 5

Let V denote the right-hand side of (74). If $( \sigma ^ { 2 } + B ^ { 2 } ) V / n \geq B ^ { 2 }$ , the zero estimator has risk at most $B ^ { 2 }$ , proving the first branch of (21). Otherwise set

$$
\varepsilon ^ { 2 } = \frac { ( \sigma ^ { 2 } + B ^ { 2 } ) V } { n } < B ^ { 2 }
$$

and let $\mathcal { G }$ be an ε-net in $L ^ { 2 } ( \mu )$ ). By (75),

$$
\log | \mathcal { G } | \leq C V \log \frac { C B } { \varepsilon } \leq C V \log ( e n ) ,
$$

where the last inequality uses $\varepsilon ^ { 2 } \ge B ^ { 2 } V / n$ and $V \geq 1$ . Lemma 15 gives

$$
\Re _ { n } ^ { * } ( A , B , \sigma ) \leq C \varepsilon ^ { 2 } + C \frac { ( \sigma ^ { 2 } + B ^ { 2 } ) V \log ( e n ) } { n } \leq C \frac { ( \sigma ^ { 2 } + B ^ { 2 } ) V \log ( e n ) } { n } .
$$

Substitute (74) to obtain Theorem 5.

## H RADIUS GEOMETRY AND PHASE REGIMES

The lower bound is governed by

$$
\lambda _ { \mathrm { o u t } } = B , \qquad \lambda _ { \mathrm { s t a t } } = \sigma \sqrt { M / n } , \qquad \lambda _ { \mathrm { r e p } } = \frac { 1 } { M } \left( \frac { ( A - C _ { 0 } ) _ { + } } { C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { D } .
$$

## H.1 The sample-size-dependent statistical regime

Fix constants $0 < c _ { \sigma } \le C _ { \sigma } < \infty$ , and let $A = B = R , c _ { \sigma } R \leq \sigma \leq C _ { \sigma } R , R \geq 2 C _ { 0 }$ , and $n \geq M$ Since $R - C _ { 0 } \geq R / 2$

$$
\lambda _ { \mathrm { r e p } } \geq \frac { 1 } { M } \left( \frac { R } { 2 C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { D } .
$$

A sufficient condition for $\lambda _ { \mathrm { { r e p } } } \geq c R \sqrt { M / n }$ is

$$
R ^ { D - 1 } \ge ( C _ { \mathrm { r a d } } D w ^ { 2 } ) ^ { D } \frac { M ^ { 3 / 2 } } { \sqrt { n } } ,\tag{79}
$$

where the base constant $C _ { \mathrm { r a d } }$ absorbs $2 C _ { \mathrm { t r } }$ and the fixed noise-comparison constants. This proves Corollary 4.

For a condition valid simultaneously for every $n \geq M ,$ , use $M ^ { 3 / 2 } / \sqrt { n } \leq M$ , obtaining (19). Taking (D − 1)st roots yields (20).

## H.2 Representation-limited behavior

When A is small, the present code gives

$$
\mathfrak { R } _ { n } ^ { * } \gtrsim \frac { 1 } { M ^ { 2 } } \left( \frac { ( A - C _ { 0 } ) _ { + } } { C _ { \mathrm { t r } } D w ^ { 2 } } \right) ^ { 2 D } .
$$

By the block Lipschitz estimate of Parhi and Nowak (2022),

$$
\operatorname { L i p } ( f ) \leq \prod _ { \ell = 1 } ^ { L } \| s _ { \ell } \| _ { \mathcal { R } \operatorname { B V } ^ { 2 } } \leq ( A / L ) ^ { L } .
$$

Thus small $A / L$ can collapse the class exponentially in depth, showing that a radius condition is structural rather than merely technical.
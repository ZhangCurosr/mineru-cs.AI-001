# ELVAE: Evidential Learning-Based Variational Autoencoder for Uncertainty-Aware Generation

Ge Wang

Biomedical Imaging Center, Rensselaer Polytechnic Institute, Troy, New York, USA

Abstract: Variational autoencoders (VAEs) generate samples from probabilistic latent representations but do not explicitly distinguish uncertainty in the latent location from variability around that location. We formulate ELVAE, an evidential learning-based VAE in which each latent coordinate is governed by an input-dependent normal-inverse-gamma (NIG) posterior. The hierarchy yields an explicit latent-location uncertainty, $u _ { \mathrm { e p i } } = \beta / [ \nu ( \alpha - 1 ) ]$ , that can stratify posterior anchors and modulate generation. In a 10,000-image MNIST pilot, samples were ranked by within-class $u _ { \mathrm { e p i } }$ and evaluated with a classifier trained only on real MNIST. Classifier error increased from 26.30% in the bottom 20% of uncertainty to 37.80% in the top 20% (1.437×, 95% bootstrap CI 1.33–1.58). A zero-displacement control, $z = \gamma$ retained most of this contrast (1.395×), showing that the dominant efect reflects anchor re-generation reliability. Among anchors correctly re-generated at $z = \gamma$ , an uncertainty-scaled perturbation induced semantic failure in 1.97% of the low- $- u _ { \mathrm { e p i } }$ group versus 5.92% of the high- $- u _ { \mathrm { e p i } }$ group (3.01×, 95% CI 2.03–4.75). Two caveats are reported alongside the headline: under unnormalized global $u _ { \mathrm { e p i } }$ ranking the contrast is essentially null (1.015×), so within-class normalization is not cosmetic, and across three random seeds the headline ratio ranges from 1.126 to 1.437. The formulation is also mathematically well posed: the ELVAE objective is an exact ELBO for the corresponding hierarchical generative model, and direct NIG-to-NIG regularization identifies an uncertainty decomposition that the marginalized Student-� latent law alone does not. These results support $u _ { \mathrm { e p i } }$ as a useful variable for posterior-anchored uncertainty-aware generation, while distinguishing anchor reliability from perturbation-attributable failure.

Keywords: evidential learning; variational autoencoder; uncertainty-aware generation; synthetic data; stress testing;   
normal-inverse-gamma; epistemic uncertainty; generative AI.

## 1 Introduction

Generative AI is typically evaluated by realism, fidelity, diversity, or downstream usefulness. For training-data generation and robustness testing, however, a second question is equally important: how uncertain is the model about the latent state from which a particular synthetic image is generated? Two generated images can both look plausible while one lies in a well-constrained latent region and the other is produced from a latent location for which the model has weak evidence.

A conventional VAE models

$$
q _ { \phi } ( z \mid y ) = N \Big ( \mu _ { \phi } ( y ) , \mathrm { d i a g } \sigma _ { \phi } ^ { 2 } ( y ) \Big ) ,\tag{1}
$$

where $\mu _ { \phi }$ and $\sigma _ { \phi } ^ { 2 }$ are deterministic network outputs [2]. This provides stochastic generation, but the model has only one level of latent uncertainty. Evidential learning suggests a richer construction: place a distribution over the latent mean and variance themselves [4]. We call the resulting model ELVAE

ELVAE turns uncertainty into a generation control variable: low uncertainty can identify more reliable synthetic samples, while high uncertainty can be deliberately retained to create dificult stress-test samples.

This is particularly attractive for scientific and medical generation, where synthetic images may be used to enlarge scarce training sets or to probe failure modes of a downstream network.

Relation to prior evidential and autoencoder work. Autoencoders have been used for representation learning since [1], and the variational formulation of [2] made the latent code an explicit probabilistic object. Evidential learning replaces a point prediction with parameters of a higher-order distribution, first for classification through a Dirichlet output [3] and then for regression through the NIG family [4]; the latter supplies the hierarchy used here. Itkina et al. [7] apply an evidential treatment to the discrete latent distribution of a conditional VAE to prune implausible modes, while Baykal et al. [8] use a Dirichlet evidential distribution over discrete VQ-VAE codebook assignments to mitigate codebook collapse. Catoni et al. [9] study uncertainty representations in continuous VAEs and introduce an Explaining-Away VAE with an additional global scaling latent variable; they evaluate latent uncertainty on natural, MNIST, and medical-image domains. ELVAE addresses a diferent question: it places a coordinate-wise NIG hierarchy directly over continuous latent location and variance, identifies the higher-order uncertainty decomposition through NIG-to-NIG regularization, and exposes the latent-location variance $u _ { \mathrm { e p i } }$ as a variable for posterior-anchored generation. Known pathologies of evidential regression objectives [5, 6] remain relevant and motivate direct regularization of the full NIG hierarchy.

![](images/e01be7a045275a96eb3f65d9359fd30ea9d144eed7efe238a74878e5835c5d4f.jpg)  
Figure 1: ELVAE training and the posterior-anchored generation pilot. (a) Training: the encoder predicts a NIG posterior over latent location and variance, regularized toward a fixed NIG prior. (b) Generation: the pilot uses the NIG-derived $u _ { \mathrm { e p i } }$ both to rank anchors and to scale a controlled Gaussian perturbation, and a frozen classifier tests whether the intended class survives. The dashed control branch sets $\tau _ { \mathrm { e p i } } = 0$ , yielding $z = \gamma _ { i }$ , while retaining $u _ { \mathrm { e p i } }$ for ranking.

Figure 1 summarizes the training and generation pipeline. The present study makes four contributions. First, we formulate a continuous NIG hierarchy for VAE latent variables and define an explicit latent-location uncertainty $u _ { \mathrm { e p i } }$ Second, we show that direct NIG regularization is required to identify the uncertainty decomposition and that the resulting training objective is an exact ELBO. Third, we perform a direct generation pilot: ELVAE generates labeled MNIST variants from posterior anchors, low- and high- $- u _ { \mathrm { e p i } }$ images are displayed, and one frozen classifier quantifies whether $\mathrm { \ h i g h } { - u _ { \mathrm { e p i } } }$ generated samples are more likely to lose their intended semantic class. Fourth, we separate that efect into an anchor-quality component and a generation-attributable component by means of a $z = \gamma$ control, and report both.

## 2 Methodology

## 2.1 Evidential latent hierarchy

For each latent coordinate $k = 1 , \ldots , K .$ , the encoder predicts four NIG parameters

$$
( \gamma _ { k } , \nu _ { k } , \alpha _ { k } , \beta _ { k } ) , \qquad \nu _ { k } > 0 , \alpha _ { k } > 1 , \beta _ { k } > 0 ,\tag{2}
$$

which define

$$
\sigma _ { k } ^ { 2 } \mid y \sim \mathrm { I n v G a m m a } ( \alpha _ { k } , \beta _ { k } ) , \qquad \mu _ { k } \mid \sigma _ { k } ^ { 2 } , y \sim { \cal N } \bigg ( \gamma _ { k } , \frac { \sigma _ { k } ^ { 2 } } { \nu _ { k } } \bigg ) , \qquad z _ { k } \mid \mu _ { k } , \sigma _ { k } ^ { 2 } \sim { \cal N } ( \mu _ { k } , \sigma _ { k } ^ { 2 } ) .\tag{3}
$$

Thus

$$
q _ { \phi } ( \mu _ { k } , \sigma _ { k } ^ { 2 } \mid \boldsymbol { y } ) = \mathrm { N I G } ( \gamma _ { k } , \nu _ { k } , \alpha _ { k } , \beta _ { k } ) .\tag{4}
$$

We enforce positivity by softplus transforms, with the standard +1 ofset on � so that $\mathbb { E } [ \sigma ^ { 2 } ]$ exists [4].

The hierarchy separates two statistically diferent sources of latent spread:

$$
u _ { \mathrm { v a r } , k } \equiv \mathbb { E } [ \sigma _ { k } ^ { 2 } \mid \boldsymbol { y } ] = \frac { \beta _ { k } } { \alpha _ { k } - 1 } ,\tag{5}
$$

$$
u _ { \mathrm { e p i } , k } \equiv \mathrm { V a r } ( \mu _ { k } \mid y ) = \frac { \beta _ { k } } { \nu _ { k } ( \alpha _ { k } - 1 ) } ,\tag{6}
$$

$$
\operatorname { V a r } ( z _ { k } \mid y ) = u _ { \mathrm { v a r } , k } + u _ { \mathrm { e p i } , k } .\tag{7}
$$

In this paper we call Eq. (6) epistemic latent uncertainty because it quantifies uncertainty in the latent location itself, while Eq. (5) quantifies variability around that location. For one image we summarize

$$
u _ { \mathrm { e p i } } ( y ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } u _ { \mathrm { e p i } , k } ( y ) .\tag{8}
$$

## 2.2 Training objective

With � input pixels and � latent coordinates, we use

$$
L _ { \mathrm { E L V A E } } ( y ) = L _ { \mathrm { r e c o n } } ( y ) + \lambda _ { \mathrm { N I G } } L _ { \mathrm { N I G } } ( y ) ,\tag{9}
$$

where

$$
L _ { \mathrm { r e c o n } } = \frac { 1 } { P } \mathbb { E } \| y - g _ { \theta } ( z ) \| _ { 2 } ^ { 2 } ,\tag{10}
$$

and

$$
L _ { \mathrm { N I G } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } D _ { \mathrm { K L } } [ \mathrm { N I G } ( \gamma _ { k } , \gamma _ { k } , \alpha _ { k } , \beta _ { k } ) \| \mathrm { N I G } ( \gamma _ { 0 } , \gamma _ { 0 } , \alpha _ { 0 } , \beta _ { 0 } ) ] .\tag{11}
$$

The NIG KL directly regularizes the higher-order quantities to which uncertainty meaning is assigned. Because � is the coordinate average, the total higher-order KL is $K L _ { \mathrm { { N I G } } }$ . For $q = \mathrm { N I G } ( \gamma , \nu , \alpha , \beta )$ and $p _ { 0 } = \mathrm { N I G } ( \gamma _ { 0 } , \gamma _ { 0 } , \alpha _ { 0 } , \beta _ { 0 } )$ under the inverse-gamma parameterization of Eq. (3), the coordinate-wise divergence is

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { \mathrm { K L } } ( q \| p _ { 0 } ) = \alpha _ { 0 } \log \displaystyle \frac { \beta } { \beta _ { 0 } } - \log \Gamma ( \alpha ) + \log \Gamma ( \alpha _ { 0 } ) + ( \alpha - \alpha _ { 0 } ) \psi ( \alpha ) - \alpha + \frac { \alpha \beta _ { 0 } } { \beta } } } \\ { { + \displaystyle \frac { 1 } { 2 } \left[ \log \displaystyle \frac { \nu } { \nu _ { 0 } } + \frac { \nu _ { 0 } } { \nu } - 1 + \frac { \nu _ { 0 } \alpha ( \gamma - \gamma _ { 0 } ) ^ { 2 } } { \beta } \right] , } } \end{array}\tag{12}
$$

where $\psi$ is the digamma function. This is the KL between the inverse-gamma factors plus the expected KL between the conditional normal distributions of $\mu .$

## 2.3 ELBO interpretation and determination of the NIG weight

Starting directly from Eq. (9), the only apparent free balance is $\lambda _ { \mathrm { { N I G } } }$ , which weights the NIG regularizer relative to the reconstruction term. This weight has a likelihood interpretation rather than being an arbitrary tuning parameter. Consider the hierarchical generative model

$$
p ( \mu , \sigma ^ { 2 } ) = p _ { 0 } , \qquad p ( z \mid \mu , \sigma ^ { 2 } ) = N ( \mu , \sigma ^ { 2 } ) , \qquad p ( y \mid z ) = N \Big ( g _ { \theta } ( z ) , s ^ { 2 } I _ { P } \Big ) ,\tag{13}
$$

with inference model $q ( \mu , \sigma ^ { 2 } , z \mid y ) = q _ { \phi } ( \mu , \sigma ^ { 2 } \mid y ) p ( z \mid \mu , \sigma ^ { 2 } )$ . The scalar $s ^ { 2 }$ is the homoscedastic observation variance in image space.

For this model, the negative ELBO is

$$
- \mathrm { E L B O } ( y ) = \frac { P } { 2 } \log ( 2 \pi s ^ { 2 } ) + \frac { P } { 2 s ^ { 2 } } { \cal L } _ { \mathrm { r e c o n } } ( y ) + \mathrm { K } { \cal L } _ { \mathrm { N I G } } ( y ) .\tag{14}
$$

The chain-rule KL reduces to $D _ { \mathrm { K L } } [ q _ { \phi } ( \mu , \sigma ^ { 2 } \mid \boldsymbol { y } ) \| p _ { 0 } ]$ because $q ( z \mid \mu , \sigma ^ { 2 } )$ is chosen to equal $p ( z \mid \mu , \sigma ^ { 2 } )$ . The Gaussian observation model supplies the first two terms in Eq. (14).

Multiplying Eq. (9) by $P / ( 2 s ^ { 2 } )$ gives

$$
\frac { P } { 2 s ^ { 2 } } L _ { \mathrm { E L V A E } } = \frac { P } { 2 s ^ { 2 } } L _ { \mathrm { r e c o n } } + \frac { P \lambda _ { \mathrm { N I G } } } { 2 s ^ { 2 } } L _ { \mathrm { N I G } } .\tag{15}
$$

Matching the coeficient of $L _ { \mathrm { { N I G } } }$ in Eqs. (14) and (15) gives

$$
\lambda _ { \mathrm { N I G } } = \frac { 2 s ^ { 2 } K } { P } , \qquad \mathrm { o r e q u i v a l e n t l y } \qquad s ^ { 2 } = \frac { P \lambda _ { \mathrm { N I G } } } { 2 K } .\tag{16}
$$

Thus $\lambda _ { \mathrm { { N I G } } }$ and the assumed image-space residual variance are two parameterizations of the same relative weighting.   
Fixing one determines the other.

The observation variance can itself be estimated from reconstruction residuals. $\operatorname { I f } s ^ { 2 }$ is treated as an unknown scalar and Eq. (14) is minimized with respect to $s ^ { 2 }$ while the encoder and decoder are held fixed, then

$$
\widehat { s } ^ { 2 } = \frac 1 P \mathbb E _ { y , z } \lVert y - g _ { \theta } ( z ) \rVert _ { 2 } ^ { 2 } = L _ { \mathrm { r e c o n } } ,\tag{17}
$$

where $L _ { \mathrm { r e c o n } }$ denotes the reconstruction MSE averaged over the data distribution and latent sampling. Substituting Eq. (17) into Eq. (16) yields the practical calibration

$$
\widehat { \lambda } _ { \mathrm { N I G } } = \frac { 2 K } { P } L _ { \mathrm { r e c o n } } .\tag{18}
$$

Therefore, once the reconstruction MSE is measured, $\lambda _ { \mathrm { { N I G } } }$ can be determined from the likelihood model instead of selected independently. The relevant quantity is the dataset-averaged, approximately converged reconstruction MSE, not the error of one image or one mini-batch. Because $\lambda _ { \mathrm { { N I G } } }$ also afects training, a fully self-consistent implementation can use a short warm-up, estimate $L _ { \mathrm { { r e c o n } } } .$ , update �<sub>NIG</sub> using Eq. (18), and optionally repeat; an even cleaner alternative is to learn log $s ^ { 2 }$ jointly in the unscaled ELBO of Eq. (14).

Because Eq. (18) is a testable identity rather than a recommendation, we report the test. On the trained pilot model the held-out reconstruction MSE under the sampled hierarchy is $L _ { \mathrm { r e c o n } } = 0 . 0 4 0 8$ , so Eq. (18) gives $\widehat { \lambda } _ { \mathrm { N I G } } = \bar { 8 . 3 3 } \times 1 0 ^ { - 4 }$ whereas the fixed value used for training is $5 \times 1 0 ^ { - 4 }$ (equivalently $s ^ { 2 } = 0 . 0 2 4 5 )$ . The fixed weight is therefore roughly a factor of 1.67 smaller than its own likelihood-consistent value: the pilot places somewhat less weight on the NIG regularizer than the observation model would imply. We retain the fixed setting used in the reported pilot and flag the discrepancy explicitly rather than leaving Eq. (18) as an unverified claim. Closing the gap by one fixed-point iteration is an important next step for a study in which �<sub>NIG</sub> is varied deliberately.

## 2.4 Why the full NIG hierarchy must be regularized

Marginalizing $( \mu , \sigma ^ { 2 } )$ yields a Student-� latent distribution. However, that marginal does not identify the decomposition in Eqs. (5)–(6).

Proposition 1 (Marginal non-identifiability). Under Eq. $( 3 ) , z _ { k }$ is Student-� with 2� degrees of freedom, location $\gamma ,$ and squared scale $\beta ( 1 + 1 / \nu ) / \alpha$ . Therefore the marginal depends on $( \nu , \beta )$ only through $c = \beta ( 1 + 1 / \nu )$ . Along the curve $\beta ( 1 + 1 / \nu ) = c _ { \mathrm { : } }$ , the marginal distribution of $z$ is unchanged while

$$
u _ { \mathrm { v a r } } = \frac { c } { \alpha - 1 } \frac { \nu } { 1 + \nu } , \qquad u _ { \mathrm { e p i } } = \frac { c } { \alpha - 1 } \frac { 1 } { 1 + \nu }\tag{19}
$$

can trade continuously against one another.

The practical consequence is important for generation: if $u _ { \mathrm { e p i } }$ is to control which images are treated as reliable or challenging, the hierarchy that defines $u _ { \mathrm { e p i } }$ must itself be constrained. A loss written only on the marginalized $p ( z )$ cannot supply that identification [5, 6].

## 2.5 Prior and pilot architecture

We use

$$
( \gamma _ { 0 } , \gamma _ { 0 } , \alpha _ { 0 } , \beta _ { 0 } ) = ( 0 , 1 , 3 , 1 ) .\tag{20}
$$

This gives $\mathbb { E } [ \sigma ^ { 2 } ] = 1 / 2 , \operatorname { V a r } ( \mu ) = 1 / 2$ , and hence $\mathrm { V a r } ( z ) = 1$ , preserving the familiar centered unit-variance latent convention while explicitly separating the two components.

The pilot ELVAE uses an MLP encoder $7 8 4  1 2 8  6 4$ , latent dimension $K = 8 .$ , and a mirrored decoder. It is trained for four epochs with Adam, learning rate $1 0 ^ { - 3 }$ , batch size 1024, and $\lambda _ { \mathrm { N I G } } = 5 \times 1 0 ^ { - 4 }$ . With $P = 7 8 4$ and $K = 8$ , Eq. (16) shows that this fixed weight corresponds to an assumed observation variance $s ^ { 2 } = 0 . 0 2 4 5$ . The goal is not maximum MNIST generation quality; it is a compact test of whether the learned $u _ { \mathrm { e p i } }$ provides useful generation stratification.

## 3 Uncertainty-Aware Generation

## 3.1 Uncertainty-scaled posterior-anchored generation

For a labeled anchor $( y _ { i } , c _ { i } )$ , the encoder gives $( \gamma _ { i } , \nu _ { i } , \alpha _ { i } , \beta _ { i } )$ and the coordinate-wise uncertainty vector

$$
u _ { \mathrm { e p i } , i } = \beta _ { i } \oslash { \bigl [ } \nu _ { i } \odot ( \alpha _ { i } - 1 ) { \bigr ] } ,\tag{21}
$$

where ⊘ and ⊙ denote elementwise division and multiplication. The exact NIG marginal of the latent location $\mu _ { i }$ is Student-�. To isolate $u _ { \mathrm { e p i } }$ as a generation-control amplitude without introducing the ordinary $z \mid \mu , \sigma ^ { 2 }$ variability, the pilot instead uses the variance-matched Gaussian perturbation

$$
z _ { i } ^ { \mathrm { e p i } } = \gamma _ { i } + \sqrt { u _ { \mathrm { e p i } , i } } \odot \epsilon , \qquad \epsilon \sim N ( 0 , I ) ,\tag{22}
$$

followed by

$$
x _ { i } ^ { \mathrm { g e n } } = g _ { \theta } ( z _ { i } ^ { \mathrm { e p i } } ) .\tag{23}
$$

Equation (22) is therefore not an exact sample from the NIG/Student-� posterior. It is a controlled Gaussian perturbation centered at $\gamma _ { i }$ whose coordinate-wise variance matches $\operatorname { V a r } ( \mu _ { i } \mid y _ { i } ) = u _ { \mathrm { e p i } , i }$ . This deliberate construction tests whether the learned uncertainty is useful as a generation variable while keeping the reported pilot simple and interpretable.

For exact posterior generation one can instead draw $\sigma ^ { 2 } \sim$ InvGamma $( \alpha , \beta )$ , then $\mu \sim { \cal N } ( \gamma , \sigma ^ { 2 } / \nu )$ and $z \sim \mathcal { N } ( \mu , \sigma ^ { 2 } )$ . For control-oriented generation, the variance-matched rule can be generalized to

$$
z = \gamma + \tau _ { \mathrm { e p i } } \sqrt { u _ { \mathrm { e p i } } } \odot \epsilon _ { \mathrm { e p i } } + \tau _ { \mathrm { v a r } } \sqrt { u _ { \mathrm { v a r } } } \odot \epsilon _ { \mathrm { v a r } } ,\tag{24}
$$

with two interpretable amplitudes. This second expression is likewise a variance-matched control parameterization rather than an exact draw from the hierarchical posterior. Low $\tau _ { \mathrm { e p i } }$ favors conservative posterior-anchored variation, whereas increasing $\tau _ { \mathrm { e p i } }$ deliberately probes sensitivity to latent-location uncertainty.

## 3.2 Separating anchor reliability from perturbation-attributable efects

Equation (22) makes $u _ { \mathrm { e p i } }$ serve two distinct roles: it is (i) an uncertainty score computed from the NIG posterior and used to rank anchors, and (ii) the scale of the random displacement applied to �. A raw comparison of classifier error between low- and $\mathrm { \ h i g h } { - u _ { \mathrm { e p i } } }$ populations therefore cannot say which role produced the diference. Anchors with large $u _ { \mathrm { e p i } }$ may simply be atypical digits that the encoder–decoder pair represents poorly, in which case the decoded image can lose its class even with no uncertainty-scaled displacement.

The distinction can be written explicitly as

$$
\underbrace { u _ { \mathrm { e p i } } = { \frac { \beta } { \nu ( \alpha - 1 ) } } } _ { \begin{array} { c } { { \mathrm { e s t i m a t e d ~ f o r ~ e v e r y ~ a n c h o r } } } \\ { { \mathrm { a n d ~ r e t a i n e d ~ f o r ~ r a n k i n g } } } \end{array} } \quad \begin{array} { c } { { \mathrm { v e r s u s } } } \\ { { \mathrm { ~ } } } \\ { { \mathrm { u s e ~ o f ~ t h a t ~ u n c e r t a i n t y } } } \end{array}  \quad \underbrace { \tau _ { \mathrm { e p i } } { \sqrt { u _ { \mathrm { e p i } } } } { \epsilon } } _ { \begin{array} { c } { { \mathrm { a } } } \\ { { \mathrm { u s e ~ o f ~ t h a t ~ u n c e r t a i n t y } } } \\ { { \mathrm { i n ~ p e r t u r b a t i o n } } } \end{array} }
$$

Setting $z = \gamma$ switches of only the second quantity by setting $\tau _ { \mathrm { e p i } } = 0$ . It does not set $u _ { \mathrm { e p i } } = 0$ , and it does not prevent the decoder from generating $g _ { \theta } ( \gamma )$

We therefore use three conditions, all sharing one trained ELVAE, one frozen classifier, and one realization of �:

(A) Uncertainty-scaled generation.

<sub>�</sub> = <sub>�</sub> + √�<sub>epi</sub> ⊙ �, i.e. Eq. (22). This is the primary uncertainty-scaled generation condition.

(C) Zero-displacement control.

$z = \gamma ,$ equivalently $\tau _ { \mathrm { e p i } } = 0$ in Eq. (24) (with $\tau _ { \mathrm { v a r } } = 0$ in this pilot). The encoder still produces $u _ { \mathrm { e p i } }$ for each anchor and the samples are still ranked by $u _ { \mathrm { e p i } } ;$ only uncertainty-scaled displacement is disabled. Any low/high stratification surviving here is therefore associated with anchor representation/re-generation reliability rather than with uncertainty-scaled perturbation.

## (I) Generation-attributable failure.

Restrict attention to anchors whose condition-(C) generation is already classified correctly, and measure the failure rate under condition (A). Every failure counted here was induced by uncertainty-scaled perturbation.

Condition (C) is the required zero-displacement baseline. Statistic (I) isolates the part of the result attributable to the uncertainty-scaled perturbation.

## 3.3 Low-uncertainty augmentation and high-uncertainty stress testing

For a class-conditional or posterior-anchored application, generated samples can be divided by $u _ { \mathrm { e p i } }$

<table><tr><td>Generation region</td><td>Interpretation</td><td>Intended use</td></tr><tr><td>Low  $u _ { \mathrm { e p i } }$ </td><td>latent location is comparatively well deter- mined</td><td>higher-confidence synthetic augmentation after task-specific validity checks</td></tr><tr><td>Intermediate  $u _ { \mathrm { e p i } }$ </td><td>moderate uncertainty/diversity</td><td>exploratory generation and data enrichment</td></tr><tr><td>High  $u _ { \mathrm { e p i } }$ </td><td>latent location is weakly determined</td><td>stress testing, failure analysis, and hard-example generation</td></tr></table>

High- $\cdot u _ { \mathrm { e p i } }$ images are not automatically “bad” images. They are images generated from an anchor whose latent location ELVAE estimates less precisely. Some remain semantically correct; others cross a task boundary. The value of $u _ { \mathrm { e p i } }$ is therefore statistical stratification, not deterministic rejection. Because much of the low/high contrast is inherited from the anchor, the table is best read as a policy over anchors to be re-generated, with the additional sampling-induced risk quantified separately by statistic (I).

## 4 Generation Pilot on MNIST

## 4.1 Study design

The 70,000 MNIST digit images [10] were pooled and partitioned into 60,000 training images and 10,000 held-out images. ELVAE is trained without using digit labels. The labels are used only to define the intended semantic identity of each posterior anchor and to evaluate the generated images.

A separate MLP classifier 784 → 256 → 128 → 10 is trained only on the real 60,000-image training set for three epochs and then frozen. It achieves 96.85% accuracy on the held-out real images. No generated image is used to train or tune this classifier.

For each held-out anchor, one image is generated using Eq. (22). The scalar $u _ { \mathrm { e p i } }$ is the mean of the eight coordinatewise uncertainties from Eq. (8). A generation error occurs when the frozen classifier prediction difers from the anchor label. Because the baseline scale of $u _ { \mathrm { e p i } }$ difers across digit classes, uncertainty is ranked within the intended digit class before pooling. This prevents an intrinsically high-uncertainty but easy class from dominating the high $- u _ { \mathrm { e p i } }$ group.

If $u _ { \mathrm { e p i } }$ is instead ranked globally across all 10,000 held-out images, the bottom-versus-top 20% contrast is 26.70% versus 27.10%, a ratio of 1.015×—essentially no efect. All of the reported stratification therefore lives within digit classes and none of it survives pooling across classes. The reason is that the class-level mean of $u _ { \mathrm { e p i } }$ and class-level digit dificulty are not aligned, so global ranking mixes an easy high- $- u _ { \mathrm { e p i } }$ class into the high-uncertainty group and cancels the within-class trend. This is a legitimate use of stratification rather than a selection efect, but it means the unconditional model produces a $u _ { \mathrm { e p i } }$ scale that is only comparable within a class. A conditional ELVAE would be required for applications in which anchors of diferent classes must be ranked against each other.

The primary quantitative comparison is classifier error in the bottom and top 20% of within-class $u _ { \mathrm { e p i } }$ . A classstratified bootstrap with 800 replicates gives confidence intervals. We also report error across ten within-class uncertainty deciles. Every quantity is additionally reported under conditions (A), (C), and (I) of Sec. 3.2, and the whole pilot is repeated for three random seeds so that run-to-run spread is visible.

## 4.2 Qualitative visualization

Figure 2 shows three rows of empirical ELVAE generations. The top row is the $\tau _ { \mathrm { e p i } } = 0$ zero-displacement control: the decoder is evaluated at $z = \gamma$ for an anchor at the median within-class $u _ { \mathrm { e p i } }$ of each digit class. The encoder still computes a nonzero $u _ { \mathrm { e p i } }$ for each anchor; that score is shown above each image but is not used to perturb �. $\mathbf { A } \mathbf { t } z = \boldsymbol { \gamma }$ the frozen classifier errs on 28.28% of the 10,000 anchors, reflecting information loss in the deliberately small MLP encoder–decoder. The second and third rows are empirical ELVAE generations selected at approximately the 10th and 90th within-class $u _ { \mathrm { e p i } }$ percentiles. Their titles give intended class → frozen-classifier prediction and the absolute $u _ { \mathrm { e p i } }$ value. These rows are selected by uncertainty percentile only, not by classifier outcome.

![](images/e0486af3f0a111500486300f0dde957c0e3724246ddad95904e0dbf32be97268.jpg)  
Figure 2: Visualization of posterior-anchored uncertainty-aware generation. Top row: condition (C), the measured zero-displacement control $z = \gamma$ with $\tau _ { \mathrm { e p i } } = 0 ;$ ; the displayed $u _ { \mathrm { e p i } }$ values remain the NIG-derived uncertainty scores used for ranking. The top-row anchors are at the median within-class $u _ { \mathrm { e p i } }$ of each class. Middle row: approximately the 10th within-class $u _ { \mathrm { e p i } }$ percentile. Bottom row: approximately the 90th percentile. Each image reports intended class and frozen-classifier result as $c \to { \hat { c } } ,$ together with its �<sub>epi</sub> value; red titles mark generations whose class was lost.

Table 1: Generation pilot result.
<table><tr><td>Quantity</td><td>Result</td></tr><tr><td>Frozen classifier accuracy on held-out real MNIST</td><td>96.85%</td></tr><tr><td>Classifier error on all ELVAE generations</td><td>29.06%</td></tr><tr><td>Bottom 20% within-class  $u _ { \mathrm { e p i } } { : }$  classifier error</td><td>26.30%</td></tr><tr><td>Top 20% within-class  $u _ { \mathrm { e p i } } { \mathrm { : } }$  classifier error</td><td>37.80%</td></tr><tr><td>High/low error ratio</td><td>1.437× (95% CI 1.33–1.58)</td></tr><tr><td>High-minus-low absolute increase</td><td>11.50 points (95% CI 8.99–14.53)</td></tr><tr><td>Top  $u _ { \mathrm { e p i } }$  decile error</td><td>43.00%</td></tr><tr><td>AUROC of within-class  $u _ { \mathrm { e p i } }$  percentile for failure</td><td>0.556</td></tr><tr><td>Classifier error at  $z = \gamma$  (control C)</td><td>28.28%</td></tr><tr><td>Global (unnormalized)  $u _ { \mathrm { e p i } }$  high/low ratio Held-out reconstruction MSE</td><td>1.015×</td></tr><tr><td> $L _ { \mathrm { r e c o n } }$ </td><td>0.0408</td></tr><tr><td>λNIG from Eq. (18)</td><td> $8 . 3 3 \times 1 0 ^ { - 4 }$ </td></tr></table>

## 4.3 Quantitative result with a fixed classifier

Table 1 collects the pilot results. Across all 10,000 generated images, the frozen classifier error was 29.06%. Uncertainty stratified this error. In the bottom 20% of within-class $u _ { \mathrm { e p i } } .$ , classifier error was 26.30%; in the top 20%, it was 37.80%. Thus the high- $- u _ { \mathrm { e p i } }$ group had an 11.50 percentage-point higher error, or a 1.437× error rate. The class-stratified bootstrap gave a 95% CI of $8 . 9 9 \mathrm { - } 1 4 . 5 3 $ percentage points for the diference and 1.33–1.58 for the ratio. A two-proportion chi-square test gave $p = 6 . 5 6 \times 1 0 ^ { - 1 5 }$

Table 2 places the result next to the controls and changes how it should be read. Under condition (C), with uncertainty-scaled displacement removed entirely, the bottom-versus-top contrast is 26.30% versus 36.70%, a ratio of 1.395×. Almost the whole condition-(A) stratification is therefore already present before uncertainty-scaled perturbation is applied. This does not mean “before generation” and does not mean $u _ { \mathrm { e p i } } = 0 \colon$ the decoder still generates $g _ { \theta } ( \gamma )$ , and the encoder-derived $u _ { \mathrm { e p i } }$ is still used to define the low/high groups. Uncertainty-scaled perturbation raises overall error only from 28.28% to 29.06%. The appropriate reading is that $u _ { \mathrm { e p i } }$ ranks anchors by how faithfully the model can re-encode and re-decode them, which is a useful property of the evidential posterior but is not, by itself, a perturbation-induced efect.

The generation-attributable statistic (I) isolates what uncertainty-scaled perturbation does. Among anchors that condition (C) already classifies correctly, the perturbation causes failure in 1.97% of the $\mathrm { \ l o w { - } \it { u } _ { \mathrm { e p i } } }$ group and 5.92% of the $\mathrm { \ h i g h } { - u _ { \mathrm { e p i } } }$ group, a ratio of $3 . 0 1 \times ( 9 5 \% \mathrm { C I } 2 . 0 3 \ – 4 . 7 5 )$ . The relative efect is larger than the headline ratio, but the base rate is small, so the number of images actually destabilized by uncertainty-scaled perturbation is modest. Figure 4 gives the corresponding decile trend. A stress-testing protocol built on Eq. (22) at $\tau _ { \mathrm { e p i } } = 1$ will therefore produce hard examples at a low yield. Increasing $\tau _ { \mathrm { e p i } }$ is the natural next experiment.

Table 2: The same trained model and frozen classifier under conditions $( \mathbf { A } ) , ( \mathbf { C } )$ , and (I). Condition (C) keeps $u _ { \mathrm { e p i } }$ as the ranking variable but disables its sampling role by setting $\tau _ { \mathrm { e p i } } = 0$
<table><tr><td>Condition</td><td>Bottom 20%</td><td>Top 20%</td><td>Ratio</td></tr><tr><td>(A) Uncertainty-scaled generation  $z = \gamma + \sqrt { u _ { \mathrm { e p i } } } \odot \epsilon$ </td><td>26.30%</td><td>37.80%</td><td>1.437×</td></tr><tr><td> $ { \left( \mathbf { C } \right) } z = \gamma , \tau _ { \mathrm { e p i } } = 0 ; u _ { \mathrm { e p i } }$  retained for ranking</td><td>26.30%</td><td>36.70%</td><td>1.395×</td></tr><tr><td>(I) Generation-attributable failures</td><td>1.97%</td><td>5.92%</td><td>3.01×</td></tr></table>

![](images/d4ff9044cd2e4f73be371d3c1bc1645ba860dd809765a4e9a0944e1cb288c60b.jpg)  
Figure 3: Frozen-classifier error versus within-class ELVAE epistemic-uncertainty decile. The dashed control curve is $z = \gamma$ and tracks the uncertainty-scaled-generation curve closely. In this control $\tau _ { \mathrm { e p i } } = 0$ while $u _ { \mathrm { e p i } }$ is retained for ranking. The top uncertainty decile reaches 43.00% error.

Figure 3 gives the full decile trend with the condition-(C) curve overlaid. The curve is not strictly monotone in every low-uncertainty bin, but the upper uncertainty range shows a clear increase and the highest decile has the largest error. The lowest decile is also elevated relative to deciles 2–4, and this shape reproduces across seeds; a plausible reading is that the very lowest- $- u _ { \mathrm { e p i } }$ coordinates are those closest to the prior mean, where the decoder has the least anchor-specific information to work with. Because decile 1 sits inside the bottom-20% group, it dilutes the reference level and makes the headline ratio conservative. On a sample-by-sample basis the association is modest: the AUROC of within-class $u _ { \mathrm { e p i } }$ percentile for classifier failure is 0.556, and Spearman correlation with the binary failure indicator is 0.088. These values are safeguards against overclaiming. The result supports population-level uncertainty stratification, not a claim that $u _ { \mathrm { e p i } }$ alone perfectly predicts which individual image will fail.

## 5 Discussion and Conclusion

The pilot supports a focused and testable claim. ELVAE produces a continuous uncertainty variable that can be attached to the generation process, and populations selected by that variable difer substantially in semantic reliability. The $z = \gamma$ control shows that most of the low/high diference survives when uncertainty-scaled perturbation is switched of, even though $u _ { \mathrm { e p i } }$ itself is still measured and used for ranking. Thus this dominant component is best described as an anchor representation/re-generation reliability efect, whereas only the smaller component isolated by statistic (I) is caused by uncertainty-scaled perturbation itself. Both components are useful: the anchor-level component can help decide which examples are safer to re-generate for augmentation, whereas the generation-attributable component informs how hard to push the sampler for stress testing.

Table 3: Seed-to-seed variability of the pilot. Each row repeats the entire pipeline from a diferent random key.
<table><tr><td>Random key</td><td>Real acc.</td><td>Error (A)</td><td>Ratio (A)</td><td>Ratio (C)</td><td>Ratio (I)</td></tr><tr><td>20260809</td><td>96.85%</td><td>29.06%</td><td>1.437×</td><td>1.395×</td><td>3.01×</td></tr><tr><td>1</td><td>96.90%</td><td>30.40%</td><td>1.126×</td><td>1.097×</td><td>1.74×</td></tr><tr><td>2</td><td>96.96%</td><td>31.79%</td><td>1.209×</td><td>1.219×</td><td>1.59×</td></tr></table>

The low/high comparison should not be interpreted as a universal threshold. First, $u _ { \mathrm { e p i } }$ is class dependent in the present unconditional model, which is why the pilot uses within-class rankings; the efect does not survive global ranking. A conditional ELVAE is therefore a prerequisite for cross-class use rather than a minor refinement. Second, the association is probabilistic and modest at the single-image level. Third, classifier error is only a semantic proxy for image utility; medical imaging applications would require task-specific validity checks, physics constraints, or expert review. Fourth, the current MLP generator is intentionally small and generates blurry digits, and its lossiness is exactly what makes the condition-(C) error rate as high as 28.28%; a stronger decoder would lower that baseline and would let condition (I) be measured on a much larger population of correctly re-generated anchors. A stronger decoder or a hybrid ELVAE-conditioning mechanism for modern difusion or flow models could preserve the same uncertainty principle while greatly improving image fidelity. Fifth, as Table 3 shows, the headline ratio varies from 1.126 to 1.437 across three seeds, so single-seed reporting of a number like 1.437× overstates the precision of the pilot.

The experiment also clarifies what high uncertainty should mean operationally. $\mathrm { H i g h } { - u _ { \mathrm { e p i } } }$ images should not simply be discarded. If the objective is trusted augmentation, they can be down-weighted or rejected. If the objective is robustness analysis, those same samples are valuable because they are more likely to induce semantic instability. This turns one uncertainty quantity into two complementary data-generation policies.

A natural next experiment is downstream retraining: train one classifier with an equal number of $\mathrm { l o w } { - u _ { \mathrm { e p i } } }$ synthetic images and another with unfiltered synthetic images, then compare real-test accuracy and robustness. A second extension is to vary $\tau _ { \mathrm { e p i } }$ in Eq. (24) and measure whether task error increases smoothly as uncertainty-scaled perturbation is strengthened; given the small base rate of condition (I) at $\tau _ { \mathrm { e p i } } = 1$ , this is the most informative single experiment remaining, since it directly tests whether $u _ { \mathrm { e p i } }$ acts as a usable generation knob rather than only as an anchor-quality score. In medical imaging, the same framework could generate low-uncertainty anatomy for augmentation and high-uncertainty anatomy or acquisition conditions for controlled stress testing.

ELVAE replaces the deterministic latent mean and variance outputs ofa conventional VAE with an input-dependent NIG posterior over latent location and latent variance. This produces an explicit latent-location uncertainty $u _ { \mathrm { e p i } } = \beta / [ \nu ( \alpha - 1 ) ]$ that is mathematically identified by direct higher-order regularization and can be used as a generation control variable. The likelihood analysis in Sec. 2.3 further shows that �<sub>NIG</sub> need not be regarded as an arbitrary balance parameter: under the homoscedastic Gaussian observation model it is determined by the residual variance, which can in turn be estimated by the converged reconstruction MSE.

In conclusion, the MNIST pilot establishes two separable findings. First, $u _ { \mathrm { e p i } }$ stratifies posterior anchors by re-generation reliability: the top 20% uncertainty group has 37.80% classification error against 26.30% in the bottom 20%, a 1.437× increase, of which a 1.395× ratio remains when $\tau _ { \mathrm { e p i } } = 0 .$ . In that control $u _ { \mathrm { e p i } }$ is still present and used for ranking; only uncertainty-scaled perturbation is absent. Second, uncertainty-scaled perturbation itself induces semantic failure 3.01× more often in high- $- u _ { \mathrm { e p i } }$ anchors than in low- $- u _ { \mathrm { e p i } }$ anchors. The first finding is the larger anchor-reliability efect; the second is the specifically sampling-attributable generation efect. Together they support ELVAE as a framework that can stratify anchors and, to a lesser but measurable degree, control generation by model uncertainty.

Reproducibility. The pilot uses the publicly available MNIST dataset [10], partitioned into 60,000 training and 10,000 held-out images. The ELVAE encoder is an MLP 784 → 128 → 64 with � = 8 latent coordinates and a mirrored decoder, trained for four epochs with Adam, learning rate $1 0 ^ { - 3 }$ , batch size 1024, and $\lambda _ { \mathrm { N I G } } = 5 \times 1 0 ^ { - 4 }$ . The frozen evaluation classifier is an MLP 784 → 256 → 128 → 10 trained for three epochs on the real training split only. Confidence intervals use a class-stratified bootstrap with 800 replicates. Results in Tables 1 and 2 use random key 20260809; Table 3 repeats the entire pipeline for keys 1 and 2.

![](images/56d9ee292da61f68f37ead40764b0879ec69665de8700ca2d402cbfb2a686db7.jpg)  
Figure 4: Generation-attributable failure rate by within-class $u _ { \mathrm { e p i } }$ decile, condition (I). Only anchors already classified correctly at $z = \gamma$ are counted, so every failure shown was induced by uncertainty-scaled perturbation. The relative trend is steeper than in Fig. 3, but on a much smaller base rate.

Author–AI Collaboration. The author conceptualized ELVAE and the uncertainty-aware generation study and made the methodological decisions, interpreted the results, and takes responsibility for the content. Generative AI tools were used to assist with discussion, mathematical checking, computational analysis, drafting, and editing.

## References

[1] G. E. Hinton and R. R. Salakhutdinov, “Reducing the Dimensionality of Data with Neural Networks,” Science, vol. 313, no. 5786, pp. 504–507, 2006.

[2] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” in International Conference on Learning Representations, 2014.

[3] M. Sensoy, L. Kaplan, and M. Kandemir, “Evidential Deep Learning to Quantify Classification Uncertainty,” in Advances in Neural Information Processing Systems, vol. 31, 2018.

[4] A. Amini, W. Schwarting, A. Soleimany, and D. Rus, “Deep Evidential Regression,” in Advances in Neural Information Processing Systems, vol. 33, 2020.

[5] V. Bengs, E. H¨ullermeier, and W. Waegeman, “Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation,” in Advances in Neural Information Processing Systems, vol. 35, 2022.

[6] N. Meinert, J. Gawlikowski, and A. Lavin, “The Unreasonable Efectiveness of Deep Evidential Regression,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, pp. 9134–9142, 2023.

[7] M. Itkina, B. Ivanovic, R. Senanayake, M. J. Kochenderfer, and M. Pavone, “Evidential Sparsification of Multimodal Latent Spaces in Conditional Variational Autoencoders,” in Advances in Neural Information Processing Systems, vol. 33, 2020.

[8] G. Baykal, M. Kandemir, and G. Unal, “EdVAE: Mitigating Codebook Collapse with Evidential Discrete Variational Autoencoders,” Pattern Recognition, vol. 156, Art. no. 110792, 2024.

[9] J. Catoni, D. Martos, F. Csikor, E. Ferrante, D. H. Milone, B. Meszena, G. Orb´ an, and R. Echeveste, “Rem-´ edying Uncertainty Representations in Visual Inference through Explaining-Away Variational Autoencoders,” arXiv:2404.15390, 2024 (revised 2026).

[10] Y. LeCun, L. Bottou, Y. Bengio, and P. Hafner, “Gradient-Based Learning Applied to Document Recognition,” Proceedings ofthe IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
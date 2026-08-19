# Maximum Tsallis Entropy Distributions for Robust and Eficient Sparse Learning from Correlated Data

Kai Yang

kai.yang2@mail.mcgill.ca

Masoud Asgharian

Celia M.T. Greenwood

masoud.asgharian2@mcgill.ca

celia.greenwood@mcgill.ca

Editor: My editor

## Abstract

This paper addresses the limitations of Gaussian distribution assumptions in statistical sparse learning, particularly in modeling correlated and heterogeneous data. Conventional Gaussian models often lack robustness towards outliers and underlying distribution assumptions. To overcome these limitations, we propose the use of the qGaussian distribution, derived from Tsallis entropy maximization, as a robust alternative. This is notably relevant in biostatistics, where the presence of correlated observations and heterogeneity, such as in genetic and longitudinal studies, is prevalent. Our contributions include modeling of correlated data through the re-derived multivariate probability density function from Tsallis entropy maximization, thereby addressing the limitations inherent in conventional Gaussian models. Furthermore, we introduce a novel framework that adapts numerical methods designed to find equilibria in flows to tackle composite optimization problems prevalent in statistical sparse learning. Applying this framework to the Hager-Zhang conjugate gradient algorithm [23], we develop a numerically stable and eficient algorithm for sparse statistical learning. The qGaussian distribution, informed by the principle of maximizing Tsallis entropy, presents a viable and flexible alternative to Gaussian-based methods. This paper not only contributes to the theoretical understanding of statistical distributions and optimization techniques, but also paves the way for practical data analysis.

Keywords: Statistical Learning, high–dimensional, Entropy, Robust, Optimization

## 1 Introduction

In the realm of statistical sparse learning, the pursuit of robust and eficient methodologies remains paramount, especially when confronted with the complexities of correlated data. The principle of maximizing Shannon’s entropy stands as a pivotal framework that has led to the derivation of nearly all frequently utilized statistical distributions to date [12]. This principle’s application has notably revealed that the multivariate Gaussian distribution maximizes Shannon’s entropy under the first two moments, significantly influencing the landscape of statistical sparse learning. The Gaussian assumption has become a fundamental cornerstone of numerous statistical sparse learning problem formulations, and its core assumptions are rarely re-examined or challenged.

However, the Gaussian distribution’s features, particularly its exponential tail decay and the absence of a shape parameter, can present substantial limitations. Specifically, the lack of robustness towards outliers and a limited capacity to accurately represent the distribution’s shape results in violations of the Gaussian assumption in statistical modeling. Such violations have many practical repercussions, including the potential for erroneous Type I error rates and the lack of robustness towards distribution shape when estimating the dispersion parameter, motivating the development of alternative approaches.

Dispersion or volatility parameters encapsulate critical and often decisive information about distributions. Their estimations, specifically in transformations of predicted outcomes, are often indispensable. For instance, in the context of log-normal distributions, the mean is directly influenced by the volatility parameter derived from the underlying Gaussian distribution. Likewise, principles like the Law of the Unconscious Statistician (LOTUS), which rely on accurate estimation of volatility and precise understanding of the distribution’s shape, highlight the importance of determining this parameter for dependable prediction and statistical modeling.

Volatility estimation is of great importance in finance. Specifically, Itˆo’s lemma, often used in stochastic calculus for option pricing, explicitly highlights the importance of volatility’s contribution. Delving into the realm of stochastic calculus, Itˆo’s lemma provides a mathematical framework that elegantly captures volatility’s impact on dynamic systems. Itˆo’s lemma states that for a twice-diferentiable function $f ,$

$$
d f \left( t , X _ { t } \right) = \left( \frac { \partial f } { \partial t } + \mu \frac { \partial f } { \partial x } + \frac { 1 } { 2 } \sigma ^ { 2 } \frac { \partial ^ { 2 } f } { \partial x ^ { 2 } } \right) d t + \sigma \frac { \partial f } { \partial x } d W _ { t } ,\tag{1}
$$

where the term $\sigma ^ { 2 } { \frac { \partial ^ { 2 } f } { \partial x ^ { 2 } } }$ specifically denotes the contribution of volatility to changes in the function $f .$ This mathematical representation is pivotal in finance, where the phenomenon, termed volatility smile, challenges the foundational assumptions of the Black–Scholes– Merton (BSM) model, signaling empirical deviations from expected normality in option pricing models. These deviations have propelled the exploration of alternative distributions capable of more accurately reflecting market realities [42].

In response to these limitations of Gaussian distributions, the qGaussian distribution, derived from the maximization of the Tsallis entropy, emerges as a compelling alternative. The qGaussian distribution is celebrated for its flexibility in modeling the diverse shapes of bell-curved distributions, including the ability to account for heavy-tailed distributions — a feature crucial for robust modeling of financial returns. It provides a more accurate representation of financial returns on platforms such as the New York Stock Exchange (NYSE) and National Association of Securities Dealers Automated Quotations (NSADAQ)

[5, 6, 15]. Despite its proven advantages in finance, the incorporation of Tsallis entropymaximizing distributions within the domain of statistical sparse learning and biostatistics remains limited. To the best of our knowledge, this paper represents the initial endeavor to apply Tsallis entropy-maximizing distributions for biostatistical data modeling. Correlated observations, frequently encountered in genetic and longitudinal studies [18, 48, 14], as well as heterogeneity of the variance, will be specifically addressed by our proposed Tsallis entropy-maximizing model for correlated data.

Hence, this paper advocates for the application of the qGaussian distribution in modeling correlated data within sparse statistical learning frameworks. Our approach relaxes the conventional reliance on normality assumptions; We aim to demonstrate that the intricate characteristics of qGaussian distributions can profoundly enhance the modeling of correlated data, ofering a robust and versatile alternative to conventional Gaussian-based methods.

Maximum likelihood estimation is one of the most commonly used estimation techniques. However, the estimation process encounters notable computational obstacles when dealing with high–dimensional and extra-large datasets. Oracle penalties, favored for their eficacy in facilitating variable selection, present an attractive yet complex solution to sparse learning problems. However, oracle penalties are notable for their nonconvex and nonsmooth nature [40], which lead to considerable optimization challenges. Recently, proximal methods have demonstrated an unmatched speed of convergence, thereby surpassing most other approaches in eficiently handling estimation in nonsmooth problems [26]. Simultaneously, the Krylov subspace method, recognized among the top ten algorithms for computing in science and engineering of the twentieth century, lays a solid foundation for numerical analysis. The conjugate gradient method, a prominent member of the Krylov subspace methods, has been applied extensively in various areas and is a fundamental numerical tool in solving partial diferential equations [41]. While the nonlinear conjugate gradient performs exceptionally well in terms of its convergence speed and numerical stability, much better than accelerated gradient or gradient descent, its global convergence depends on the line–search step, whereas accelerated gradient and gradient descent methods do not necessarily require the line search step to achieve global convergence [19, 63]. Motivated by these methodologies, our paper introduces a proximal conjugate gradient method that can be applied to solve qGaussian sparse learning problems. This method aims to efectively combine the theoretical strengths of both proximal methods and Krylov subspace techniques. Additionally, our paper addresses the line search step needed for the proximal nonlinear conjugate gradient method to establish global convergence.

We re-derive the probability density function for the multivariate qGaussian distribution from a Tsallis entropy maximizing perspective in Lemma 1. This derivation allows for a nuanced understanding and application of this model in statistical analysis. Furthermore, our contributions in this paper are as following:

1. We apply the derived density to model correlated and heterogeneous data efectively, while carrying out the sparse statistical learning at the same time.

2. Sparse statistical learning involves minimizing a composite optimization problem, aimed at minimizing a composite objective function composed of a globally Lipschitzsmooth term, which may be nonconvex, and a convex nonsmooth term. A variety of numerical methods are available to find equilibrium points for globally Lipschitz flows. By employing the Moreau envelope and linearizing the smooth term, we develop a framework that allows any numerical method designed for finding equilibrium points in globally Lipschitz flows to be adapted into a numerical optimization algorithm for minimizing the composite objective function.

3. Leveraging the framework introduced above, we implement it with the state-of-the-art Hager-Zhang conjugate gradient method [23]. This implementation yields a proximal conjugate gradient algorithm that is not only computationally eficient but also numerically stable, suitable for a wide range of statistical sparse learning challenges. This includes the robust sparse learning approach we devised based on the concept of maximizing the Tsallis entropy distribution.

The structure of the paper is organized as follows:

Section 2 delves into the foundational properties of Tsallis entropy, drawing upon previous literature to establish a comprehensive background. Following this, Section 3 introduces the concept of q−moments. This section then elaborates on employing Tsallis entropy maximizing distribution to efectively model the q−correlation structure. In Section 3, we also re-derive the probability density function maximizing Tsallis entropy under the first and second central q−moment constraints, incorporating all relevant parameters for a likelihood-based approach to statistical analysis.

Our discussion transitions to the challenges and strategies of optimization in Section 4. This section is twofold; initially, in Section 4.1, we present essential background knowledge from variational and nonsmooth analysis. This foundation is critical for our novel contribution: the development of a proximal framework to transform any first-order numerical optimization algorithm to a proximal counterpart by leveraging the properties of the Moreau envelope, detailed in Section 4.2. In Section 4.3, we apply this innovative framework to the state-of-the-art Hager-Zhang conjugate gradient algorithm. This adaptation produces a proximal version for tackling sparse statistical learning challenges. The eficacy of this method is further showcased in Section 5, where we outline the application of our proximal Hager-Zhang conjugate gradient algorithm to optimize a penalized qGaussian likelihood function. This section also lays out a map from problem formulation to the practical aspects of prediction using models trained with our approach. Finally, Section 6 synthesizes our contributions, ofering a reflective conclusion and proposing avenues for future research.

## 2 Tsallis Entropy

For an arbitrary random variable X, Shannon’s Entropy [50] poses the definition

$$
H \left( X \right) : = - \mathbb { E } \log \left( p \left( X \right) \right) = - \int \log \left( p \left( x \right) \right) d \mu _ { X } ,\tag{2}
$$

where $p$ is the likelihood function for X. Over a given (likelihood) function space

$$
\mathcal { P } : = \{ p ( x ) | \forall x \in \mathcal { X } , \ p ( x ) \geq 0 , | \mathbb { E } \log ( p ( X ) ) | < \infty \mathrm { ~ a n d ~ } \int _ { \mathcal { X } } 1 d \mu _ { X } = 1 \} ,
$$

the Shannon’s entropy is a strictly concave function, which implies uniqueness of the maximizer. Many commonly-used distributions have been shown to maximize Shannon’s entropy under certain given constraints [12]. For example, uniform distribution, whether in discrete or continuous case , are to maximize (2) over a compact support, with open sets defined by discrete or Euclidean topology, respectively. The exponential distribution is defined as maximizing (2) over $\mathbb { R } _ { \geq 0 }$ and with a constraint that the first moment is a constant, $\textstyle { \frac { 1 } { \lambda } } ;$ ; where λ later turns out to be the scale parameter. And the Gaussian distribution maximizes (2) over R with given mean and variance. More examples can be given. For example, the constraint to obtain a Laplace distribution is a given mean absolute deviation, etc.

Additivity is a key element of Shannon’s entropy. That is, let $A _ { 1 } , A _ { 2 }$ be two independent event sets, then the information of the intersection $I \left( \mathbb { P } \left( A _ { 1 } \cap A _ { 2 } \right) \right) = I \left( \mathbb { P } \left( A _ { 1 } \right) \cdot \mathbb { P } \left( A _ { 2 } \right) \right) =$ $I \left( \mathbb { P } \left( A _ { 1 } \right) \right) + I \left( \mathbb { P } \left( A _ { 2 } \right) \right)$ — such homomorphism was considered particularly useful in Shannon’s view [50]. Later in the 1980s, Tsallis constructed an entropy similar to Shannon’s entropy but without the additivity property. To see how Tsallis’ entropy was developed, first we look at Tsallis’ q−exponential function, which is defined as ex $\mathfrak { p } _ { q } : \mathbb { R } \mapsto \mathbb { R }$ , given by

$$
\exp _ { q } x : = { \left\{ \begin{array} { l l } { \left( \left( 1 + \left( 1 - q \right) x \right) \right) ^ { \frac { 1 } { 1 - q } } ; } & { 1 + \left( 1 - q \right) x > 0 } \\ { 0 ; } & { \mathrm { e l s e } } \end{array} \right. }\tag{3}
$$

For $q > 1 , \exp _ { q } $ is bijective over $\left( 0 , { \frac { 1 } { q - 1 } } \right)$ . The inverse function, called the q−logarithmic function, is given by

$$
\ln _ { q } x : = \frac { x ^ { 1 - q } - 1 } { 1 - q } .\tag{4}
$$

Based on this deformed q−exponential function, [56] developed Tsallis entropy by replacing the log function in 2 with q − log function (4) and replacing the expectation with q−expectation operator [56]:

$$
S _ { q } \left( X \right) = - \int _ { \mathcal { X } } { p ^ { q } \left( x \right) \ln _ { q } p \left( x \right) d x } = : - \mathbb { E } _ { q } \ln _ { q } p \left( X \right)\tag{5}
$$

$$
= { \frac { 1 } { q - 1 } } \left( 1 - \int _ { \mathcal { X } } p ^ { q } \left( x \right) d x \right)\tag{6}
$$

where $q \in \mathbb { R } \setminus \{ 1 \}$ is a constant, and $\begin{array} { r } { \mathbb { E } _ { q } f \left( X \right) : = \int _ { \mathcal { X } } f \left( x \right) \cdot p ^ { q } \left( x \right) d x = \left. f \left( x \right) , \left( d \mu _ { X } \right) ^ { q } \right. } \end{array}$ is referred to as the q−expectation operator. Tsallis entropy is also known as non-extensive entropy; namely for arbitrary independent two random variable $X _ { 1 } , X _ { 2 } ;$

$$
S _ { q } ( X _ { 1 } , X _ { 2 } ) = S _ { q } ( X _ { 1 } ) \oplus _ { q } S _ { q } ( X _ { 2 } ) ,\tag{7}
$$

where $^ { 6 6 } \textcircled { \oplus } _ { q } ^ { , 9 }$ is defined as $\forall a , b \in \mathbb { R }$

$$
a \oplus _ { q } b : = a + b + ( 1 - q ) a b .\tag{8}
$$

Expectation has been used to characterize statistical distributions. However, one significant drawback of the expectation (linear) operator is the lack of continuity for some distributions; such as the Cauchy distribution. Therefore, the q−expectation operator, $\mathbb { E } _ { q } ,$ provides robustness when characterizing the distributions in the real domain. If the tail of the function vanishes at a rate of $O \left( \left( \log x \right) ^ { - 1 } \right)$ , the function will not have a proper integral if the support is unbounded. Thus, for any distribution whose likelihood function is bounded in uniform norm, $\exists q \in \mathbb { R } _ { > 0 }$ such that $\mathbb { E } _ { q }$ is continuous at the likelihood function in the function space we are considering.

## 3 Tsallis Entropy Maximizing Distribution to Accommodate the q−Correlation Structure

The Gaussian distribution maximizes the Shannon’s entropy in the following problem:

$$
\begin{array} { r l } { \operatorname* { m a x } _ { \mathbf { \phi } \phi \in \mathcal { P } } - \displaystyle \int _ { \mathbb { R } ^ { n } } \phi \left( x \right) \log \left( \phi \left( x \right) \right) d x } & { } \\ { \mathrm { s . t . } ~ \phi \geq 0 ; } \\ & { ~ \displaystyle \int _ { \mathbb { R } ^ { n } } \phi \left( x \right) d x = 1 ; } \\ & { ~ \displaystyle \int _ { \mathbb { R } ^ { n } } x \cdot \phi \left( x \right) d x = 0 ; } \\ & { ~ \displaystyle \int _ { \mathbb { R } ^ { n } } x x ^ { T } \cdot \phi \left( x \right) d x = \Sigma ; } \end{array}\tag{9}
$$

for some $n \in \mathbb { N } _ { + }$ and $\Sigma \in \mathbb { R } ^ { n \times n } , \ \Sigma \succ 0$ . For the sake of parsimony, in (9) we assume that the distribution is centered. To set the central trend parameter, or the mean parameter in the specific case of the Gaussian distribution, the likelihood function $\phi$ can be simply translated $x \mapsto x - \mu$ to incorporate the parameter $\mu$ for the central trend. Entropy functions are invariant under translation.

Similarly to how the multivariate Gaussian distribution maximizes Shannon’s entropy in a Euclidean space, the multivariate qGaussian distribution maximizes Tsallis entropy in a Euclidean space. Specifically, the optimization problem is formulated as:

$$
\operatorname* { m a x } _ { \mathbf { \phi } \phi \in L ^ { q } ( \mathbb { R } ^ { n } ) } - \int _ { \mathbb { R } ^ { n } } \phi ^ { q } \left( x \right) d x\tag{10}
$$

$$
\mathrm { s . t . } ~ \phi \geq 0 ;
$$

$$
\int _ { \mathbb { R } ^ { n } } \phi \left( x \right) d x = 1 ;\tag{11}
$$

$$
\frac { \int _ { \mathbb { R } ^ { n } } x \cdot \phi ^ { q } \left( x \right) d x } { \int _ { \mathbb { R } ^ { n } } \phi ^ { q } \left( x \right) d x } = 0 ;\tag{12}
$$

$$
\frac { \int _ { \mathbb { R } ^ { n } } x x ^ { T } \cdot \phi ^ { q } \left( x \right) d x } { \int _ { \mathbb { R } ^ { n } } \phi ^ { q } \left( x \right) d x } = \Sigma ,\tag{13}
$$

where $q > 1$ . The feasible set of Lebesgue space $L ^ { q } \left( \mathbb { R } ^ { n } \right)$ is to ensure the well–definedness of Tsallis entropy. The normalization constraint (11) implies that $\phi \in L ^ { 1 }$ ; however, $\phi \in L ^ { 1 }$ does not imply $\phi \in L ^ { q }$ , as the embedding property $L ^ { 1 } \subseteq L ^ { q }$ fails to hold for Lebesgue measure on R<sup>n</sup>. As an example, consider the one-dimensional example of the probability density function

$$
\tilde { \phi } \left( x \right) = \left\{ \begin{array} { l l } { { \frac { 1 } { 4 } \left| x \right| ^ { - { \frac { 1 } { 2 } } } } } & { { \mathrm { f o r ~ } x \in \left( - 1 , 1 \right) \backslash \left\{ 0 \right\} ; } } \\ { { 0 } } & { { \mathrm { e l s e . } } } \end{array} \right.\tag{14}
$$

Clearly, $\tilde { \phi } \in L ^ { 1 }$ but $\tilde { \phi } \notin L ^ { 2 }$ . Note that (12) and (13) are the first and second moment constraints using the q−expectation operator $\mathbb { E } _ { q }$ . As noted by [34], maximizing any member of the generalized class of power-law entropies, including Renyi entropy, Havrda and Charvat entropy, Arimoto entropy, and Tsallis entropy, all yield the identical power-law objective function (10). Regarding the constraints, ${ \int _ { \mathbb { R } ^ { n } } \phi ^ { q } \left( x \right) }$ dx is the normalization factor for the q−expectation. [34] further noted that optimizing the problem formulated above is equivalent to the following problem:

$$
\begin{array} { r l } { \operatorname* { m a x } _ { \mathbf { \Phi } \varphi \in L ^ { s } ( \mathbb { R } ^ { n } ) } \displaystyle \int _ { \mathbb { R } ^ { n } } \varphi ^ { s } \left( x \right) d x } \\ { \mathrm { s . t . } \varphi \geq 0 ; } \\ { \displaystyle \int _ { \mathbb { R } ^ { n } } \varphi \left( x \right) d x = 1 ; } \\ { \displaystyle \int _ { \mathbb { R } ^ { n } } x \cdot \varphi \left( x \right) d x = 0 ; } \\ { \displaystyle \int _ { \mathbb { R } ^ { n } } x x ^ { T } \cdot \varphi \left( x \right) d x = \Sigma . } \end{array}\tag{15}
$$

In (15), $s : = q ^ { - 1 } \in ( 0 , 1 )$ , thus $L ^ { s } \left( \mathbb { R } ^ { n } \right)$ is a quasi-normed space; $\begin{array} { r } { \varphi \left( x \right) : = \frac { \phi ^ { q } \left( x \right) } { \int _ { \mathbb { R } ^ { n } } \phi ^ { q } \left( x \right) d x } } \end{array}$ . If the maximizer of (15) is $\varphi _ { : }$ , then the maximizer of (10), ϕ, will be normalized

$$
\phi \left( x \right) \propto \varphi ^ { 1 / q } .\tag{16}
$$

Several important properties were proposed previously regarding the qGaussian distributions in previous studies [57, 11]. Notably,

1. Using Bregman information divergence, Problem (15) has a unique maximizer of the form

$$
\varphi \left( x ; s \right) = A _ { s } \left( 1 - \left( s - 1 \right) \beta ^ { \prime } \left. x , \Sigma ^ { - 1 } x \right. \right) _ { + } ^ { \frac { 1 } { s - 1 } }\tag{17}
$$

for some $\textstyle s \in { \Big ( } { \frac { n } { n + 2 } } , \infty { \Big ) } \setminus \{ 1 \}$ , normalization constant $A _ { r }$ , and some dispersion parameter $\beta ^ { \prime }$

2. If X ∼ qGaussian $( q , \Sigma ) , H \in \mathbb { R } ^ { \tilde { n } \times n }$ and rank $( H ) = \tilde { n }$ . Then $\tilde { X } \sim q$ Gaussian $\left( \tilde { q } , H \Sigma H ^ { T } \right)$ with

$$
\frac { 2 } { 1 - \tilde { q } ^ { - 1 } } - \tilde { n } = \frac { 2 } { 1 - q ^ { - 1 } } - n .\tag{18}
$$

3. If $X _ { 1 } , X _ { 2 }$ are both qGaussian random vectors but independent, a linear combination of $H _ { 1 } X _ { 1 } + H _ { 2 } X _ { 2 }$ is not qGaussian.

4. The duality property: if X ∼ qGaussian $( q , \Sigma )$ with $\textstyle 1 < q < 1 + { \frac { 2 } { n } }$ , let the degree of freedom for X be $\begin{array} { r } { m : = \frac { 2 } { q - 1 } - n } \end{array}$ and $\Lambda : = m \Sigma$ , then

$$
\begin{array} { r l r } & { } & { \frac { X } { \sqrt { 1 - \langle X , \Lambda ^ { - 1 } X \rangle } } \sim q \mathrm { G a u s s i a n } \left( \tilde { q } , \frac { m } { m + 4 } \Sigma \right) } \\ & { } & { \mathrm { w i t h } \ \frac { 1 } { \tilde { q } ^ { - 1 } - 1 } = \frac { 1 } { 1 - q ^ { - 1 } } - \frac { n } { 2 } - 1 , } \end{array}
$$

and $0 < \tilde { q } < 1$

Property 1 will be used in our Lemma 1. Property 2 implies that any components of a qGaussian random vector are also qGaussian, while Property 3 implies that two independent qGaussian vectors are not jointly qGaussian.

By the equivalence of problems (10) and (15) discussed before, (17) can be rewritten as

$$
\phi \left( x ; q , \Sigma \right) = \left( \alpha - \beta \left. x , \Sigma ^ { - 1 } x \right. \right) _ { + } ^ { \frac { 1 } { 1 - q } }\tag{19}
$$

for some constant (parameter) $\begin{array} { r } { \alpha , \beta , q \in \left( 0 , 1 + \frac { 2 } { n } \right) \setminus \{ 1 \} ; \ x _ { + } : = \operatorname* { m a x } } \end{array}$ $( 0 , x )$ . As shown later in the proof of Lemma 1, the dimension-related upper bound $1 + { \textstyle { \frac { 2 } { n } } }$ is due to the normalization constraint (11). When $0 < q < 1$ , the density represents a distribution with bounded support; when $q > 1$ , the density is a generalization of the bell curve distributions, and with $q \searrow 1$ the Gaussian distribution is recovered. A higher value of q corresponds to heavier tails in shape. The duality between the qGaussian random vectors with $0 < q < 1$ and $\textstyle 1 < q < 1 + { \frac { 2 } { n } }$ was given by [57], which we discussed in Property 4 in Section 3. Distributions with bounded support correspond to $0 ~ < ~ q ~ < ~ 1$ , and distributions with heavy tails correspond to $\textstyle 1 < q < 1 + { \frac { 2 } { n } } .$ For the scope of this paper, we will focus only on the heavy-tail distributions; i.e., the case when $q > 1$ . [60] derived the qGaussian probability density function for $\textstyle 1 < q < { \frac { n + 4 } { n + 2 } }$ , when the multivariate qGaussian density becomes the scaled density of the multivariate student’s t distribution. To incorporate the case of $q \in [ 1 + \frac { 2 } { n + 2 } , 1 + \frac { 2 } { n } )$ , when the variance does not exist but the $q \cdot$ −variance can be used to capture the volatility/dispersion of the data, [58] also derived the resulting density; however, since a typo was found in that paper, we re-derive the density in Lemma 1. The parameters presented in the density formula (20) are of particular interest to statisticians, as parameter inference is the key to statistical analysis and prediction. The case of $\textstyle q \in [ 1 + { \frac { 2 } { n + 2 } } , 1 + { \frac { 2 } { n } } )$ will allow the resulting qGaussian distribution to incorporate the wider class of distributions without finite moments but finite q−moments; such as the Cauchy distribution. Therefore, modeling using the qGaussian distribution with q allowed to take the value in $\begin{array} { r } { [ 1 + \frac { 2 } { n + 2 } , 1 + \frac { 2 } { n } ) } \end{array}$ will be more robust.

Lemma 1. When $\textstyle q \in \left( { 1 , 1 + { \frac { 2 } { n } } } \right)$ , the unique solution to (10) is:

$$
p \left( x ; q , \Sigma \right) = { \frac { 1 } { \left| \pi \Sigma \right| ^ { 1 / 2 } } } \cdot { \frac { \Gamma \left( { \frac { 1 } { q - 1 } } \right) } { \Gamma \left( { \frac { 1 } { q - 1 } } - { \frac { n } { 2 } } \right) } } \cdot \left( { \frac { 2 } { q - 1 } } - n \right) ^ { - { \frac { n } { 2 } } } \cdot \left( 1 + \left( { \frac { 2 } { q - 1 } } - n \right) ^ { - 1 } \cdot \left. x , \Sigma ^ { - 1 } x \right. \right) ^ { \frac { 1 } { 1 - q } } .\tag{20}
$$

Proof. By (17) and the equivalence of the problems (10) and (15), let the solution to (10) be denoted by

$$
p \left( x ; q , \Sigma \right) = \frac { 1 } { Z } \left( \gamma + \left. x , \Sigma ^ { - 1 } x \right. \right) ^ { \frac { 1 } { 1 - q } }\tag{21}
$$

for some $Z , \gamma > 0$ . Feasibility for problem 10 when $q \ \in \ \left( 1 , 1 + \frac { 2 } { n } \right)$ was given in [57]. Hence, the strictly concavity of the objective function 10 implies that the optimal solution is unique. The symmetry of $p \left( x ; q , \Sigma \right)$ is implied by (21); thus, we reformulate the problem (10) as the following equivalent problem:

$$
\begin{array} { r l } { \operatorname* { m a x } _ { \mathbf { \Phi } _ { p \in L ^ { q } \left( \mathbb { R } ^ { n } \right) } - \displaystyle \int _ { \mathbb { R } ^ { n } } \left( \mathbf { \Phi } ( p \left( x ; q , \Sigma \right) ) ^ { q } d x \right. } } & { } \\ { \mathrm { s . t . } ~ p \left( x ; q , \Sigma \right) \geq 0 ; } \\ & { ~ \displaystyle \int _ { \mathbb { R } _ { > 0 } ^ { n } } p \left( x ; q , \Sigma \right) d x = 2 ^ { - n } ; } \\ & { ~ \displaystyle \left. \int _ { \mathbb { R } ^ { n } } x \cdot p ^ { q } \left( x ; q , \Sigma \right) d x = 0 ; \right. } \end{array}
$$

$$
\frac { \int _ { \mathbb { R } ^ { n } } x x ^ { T } \cdot p ^ { q } \left( x ; q , \Sigma \right) d x } { \int _ { \mathbb { R } ^ { n } } p ^ { q } \left( x ; q , \Sigma \right) d x } = \Sigma .\tag{22}
$$

Thus,

$$
\begin{array} { r l } { \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } } & { \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } \sigma _ { 0 } ^ { 2 } \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } } \\ &  = \sqrt { \kappa \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } } \\ &  - \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } \sqrt { \kappa \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } } \\ &  - \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } \sqrt { \kappa \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } } \\ &  \le - \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } \sqrt { \kappa \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } } \\ &  - \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } \sqrt { \kappa | \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } } \\ &  - \| \hat { \textbf { z } } \| _ { \infty } ^ { 2 } + \| \hat { \textbf { z } } \| _ { \infty } ^   \end{array}\tag{23}
$$

In step (23), we use the following formula for Beta function [34]:

$$
\int _ { 0 } ^ { \infty } \left( x ^ { \alpha } \left( 1 + x ^ { \lambda } \right) ^ { \beta } \right) ^ { - 1 } d x = { \frac { 1 } { \lambda } } B \left( \beta - { \frac { 1 - \alpha } { \lambda } } , { \frac { 1 - \alpha } { \lambda } } \right) ,\tag{24}
$$

where $\alpha < 1 , \ \lambda > 0 , \ \beta > 0 , \ \lambda \beta > 1 - \alpha$ . Well–definedness of $Z$ and (23) implies that $\begin{array} { r } { { \frac { 1 } { q - 1 } } - { \frac { n } { 2 } } > 0 ; { \mathrm { i . e . , } } q < 1 + { \frac { 2 } { n } } } \end{array}$ , which is the reason for the upper bound for the choice of $q .$ (22) implies that

$$
\operatorname { t r } \left( \Sigma ^ { - 1 } { \frac { \int _ { \mathbb { R } ^ { n } } x x ^ { T } \cdot p ^ { q } \left( x \right) d x } { \int _ { \mathbb { R } ^ { n } } p ^ { q } \left( x \right) d x } } \right) = \operatorname { t r } \left( \Sigma ^ { - 1 } \Sigma \right) = n .\tag{25}
$$

Hence, since $p \left( x \right)$ is symmetric,

$$
\begin{array} { r l } & { \mathbb { E } _ { \rho } ^ { \lambda } ( \lambda , \mu , \frac { \lambda } { \lambda } , \lambda , \mu , \frac { \lambda } { \lambda } ) = \frac { \lambda } { \lambda } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) \mathbb { E } _ { \rho } ^ { \lambda } } \\ & { = ( \rho - \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ & { \quad - \lambda ^ { 2 } ( \lambda ^ { 2 } - \lambda ^ { 2 } - \lambda ^ { 2 } ) ^ { 2 } } \\ &  \quad - \lambda ^ { 2 } ( \lambda  \end{array}\tag{26}
$$

$$
\begin{array} { r l } & { = \frac { \int _ { 0 } ^ { \infty } \varphi _ { 1 } ( x ) \cdot \big ( \frac { 1 } { x } + \epsilon ^ { - \eta } \big ) ^ { \epsilon } d x } { \int _ { 0 } ^ { \infty } \varphi _ { 1 } ^ { - \eta } ( x ) \cdot \big ( \frac { 1 } { x } + \epsilon ^ { - \eta } \big ) ^ { \epsilon } d x } } \\ & { = \frac { \gamma \int _ { 0 } ^ { \infty } \left( \left( \epsilon ^ { \eta } \right) ^ { \epsilon + 1 } \cdot \big ( 1 + ( \epsilon ^ { \eta } ) ^ { 2 } \big ) ^ { \epsilon + 1 } \right) ^ { - 1 } d x ^ { \prime } } { \int _ { 0 } ^ { \infty } \left( \left( \epsilon ^ { \eta } \right) ^ { 1 - n } \cdot \big ( 1 - ( \epsilon ^ { - \eta } ) ^ { \epsilon } \big ) ^ { \epsilon + 1 } \right) ^ { - 1 } d x ^ { \prime } } } \\ & { - \frac { \gamma B } { \int _ { 0 } ^ { \infty } \left( \frac { 1 } { \omega _ { 1 } } ^ { 2 } - \frac { \epsilon ^ { - \eta } } { 2 } \cdot \frac { \omega ^ { 2 } } { 2 } \right) } { \int _ { 0 } ^ { \infty } \left( \frac { 1 } { \omega _ { 1 } } ^ { 2 } - \frac { \epsilon ^ { - \eta } } { 2 } \right) ^ { \epsilon } } } \\ & { - \frac { \gamma B } { \int _ { 0 } ^ { \infty } \left( \frac { 1 } { \omega _ { 1 } } ^ { 2 } - \frac { \epsilon ^ { - \eta } } { 2 } - \frac { 1 } { \omega _ { 1 } } \right) \big ( \frac { 1 } { \omega _ { 1 } } ^ { 2 } \big ) } { \int _ { 0 } ^ { \infty } \eta _ { 1 } ^ { - \eta } ( x ) } \int _ { - \frac { 1 } { \omega _ { 1 } } ^ { 2 } - \frac { \epsilon ^ { - \eta } } { 2 } } \Gamma \left( \frac { 1 } { \omega _ { 1 } ^ { 2 } } \right) } \\ &  - \frac { \gamma }  2 \cdot \big ( \frac  \epsilon ^  \end{array}\tag{27}
$$

(28)

In step (27), we used (24). Combining (25) and (28), we have

$$
\gamma \cdot { \frac { n } { 2 } } \cdot \left( { \frac { q } { q - 1 } } - { \frac { n } { 2 } } - 1 \right) ^ { - 1 } = n ,\tag{29}
$$

which gives that

$$
\gamma = \frac { 2 q } { q - 1 } - n - 2 = \frac { 2 } { q - 1 } - n .\tag{30}
$$

Thus, the probability density function that maximizes problem (10) is:

$$
\begin{array} { r l r } & { } & { p \left( x ; q , \Sigma \right) = \left( \pi ^ { \frac { n } { 2 } } \left| \Sigma \right| ^ { 1 / 2 } { \frac { \Gamma \left( { \frac { 1 } { q - 1 } } - { \frac { n } { 2 } } \right) } { \Gamma \left( { \frac { 1 } { q - 1 } } \right) } } \cdot \left( { \frac { 2 } { q - 1 } } - n \right) ^ { \frac { n } { 2 } + \frac { 1 } { 1 - q } } \right) ^ { - 1 } \left( \left( { \frac { 2 } { q - 1 } } - n \right) + \left. x , \Sigma ^ { - 1 } x \right. \right) ^ { \frac { 1 } { 1 - q } } } \\ & { } & { = \left( \pi ^ { \frac { n } { 2 } } \left| \Sigma \right| ^ { 1 / 2 } { \frac { \Gamma \left( { \frac { 1 } { q - 1 } } - { \frac { n } { 2 } } \right) } { \Gamma \left( { \frac { 1 } { q - 1 } } \right) } } \cdot \left( { \frac { 2 } { q - 1 } } - n \right) ^ { \frac { n } { 2 } } \right) ^ { - 1 } \left( 1 + \left( { \frac { 2 } { q - 1 } } - n \right) ^ { - 1 } \left. x , \Sigma ^ { - 1 } x \right. \right) ^ { \frac { 1 } { 1 - q } } } \\ & { } & { = { \frac { 1 } { \left| \pi \Sigma \right| ^ { 1 / 2 } } } \cdot { \frac { \Gamma \left( { \frac { 1 } { q - 1 } } \right) } { \Gamma \left( { \frac { 1 } { q - 1 } } - { \frac { n } { 2 } } \right) } } \cdot \left( { \frac { 2 } { q - 1 } } - n \right) ^ { - { \frac { n } { 2 } } } \cdot \left( 1 + \left( { \frac { 2 } { q - 1 } } - n \right) ^ { - 1 } \cdot \left. x , \Sigma ^ { - 1 } x \right. \right) ^ { \frac { 1 } { 1 - q } } . } \end{array}
$$

When $\textstyle q \geq 1 + { \frac { 2 } { n } }$ , the solution to problem (10) does not exist, due to property 1 and discussions in the proof. In Lemma 1, the presented density (20) outlines a formula for multivariate bell-curve distributions dependent on the value of q. As q shifts from values approaching 1 from above to values approaching $\textstyle 1 + { \frac { 2 } { n } }$ form below, the resulting density transitions from Gaussian through a scaled version of the multivariate t−distribution to Cauchy and beyond. This density explicitly details all parameters, enabling the application of the maximum likelihood principle and facilitating the use of maximum likelihood estimation in modeling correlated data performed in Section 5.

In the context of (20), the characterization matrix [11], denoted by Σ, can undergo modifications to include the degree of freedom parameter m $\left. = { \frac { 2 } { q - 1 } } - n \left[ 5 9 \right] \right.$ ; specifically,

$$
{ \begin{array} { r l } & { p \left( x ; q , \Lambda \right) = { \frac { 1 } { \left| \pi \Lambda \right| ^ { 1 / 2 } } } \cdot { \frac { \Gamma \left( { \frac { m } { 2 } } + { \frac { n } { 2 } } \right) } { \Gamma \left( { \frac { m } { 2 } } \right) } } \cdot \left( 1 + \left. x , \Lambda ^ { - 1 } x \right. \right) ^ { \frac { 1 } { 1 - q } } , } \\ & { { \mathrm { ~ w h e r e ~ } } \Lambda : = m \Sigma . } \\ & { \quad { \mathrm { ~ a n d ~ } } m : = { \frac { 2 } { q - 1 } } - n } \end{array} }\tag{31}
$$

Below are a few useful remarks related to the qGaussian distribution and other bell-curve distributions.

Remark 2. To incorporate the location parameter µ, (20) and (31) become

$$
\begin{array} { r l } & { p \left( x ; \mu , q , \Sigma \right) = \displaystyle \frac { 1 } { \left| \pi \Sigma \right| ^ { 1 / 2 } } \cdot \frac { \Gamma \left( \frac { 1 } { q - 1 } \right) } { \Gamma \left( \frac { 1 } { q - 1 } - \frac { n } { 2 } \right) } \cdot \left( \frac { 2 } { q - 1 } - n \right) ^ { - \frac { n } { 2 } } } \\ & { \qquad \cdot \left( 1 + \left( \frac { 2 } { q - 1 } - n \right) ^ { - 1 } \cdot \left. x - \mu , \Sigma ^ { - 1 } \left( x - \mu \right) \right. \right) ^ { \frac { 1 } { 1 - q } } ; } \\ & { p \left( x ; \mu , q , \Lambda \right) = \displaystyle \frac { 1 } { \left| \pi \Lambda \right| ^ { 1 / 2 } } \cdot \frac { \Gamma \left( \frac { m } { 2 } + \frac { n } { 2 } \right) } { \Gamma \left( \frac { m } { 2 } \right) } \cdot \left( 1 + \left. x - \mu , \Lambda ^ { - 1 } \left( x - \mu \right) \right. \right) ^ { \frac { 1 } { 1 - q } } . } \end{array}
$$

Remark 3. For random vector X ∼ qGaussian (q, Σ), its q−covariance is

$$
\mathbb { E } _ { q } \left[ X X ^ { T } \right] = \left( \frac { 1 } { q - 1 } - \frac { n } { 2 } \right) ^ { \frac { 1 - q } { 2 } } | \pi \Sigma | ^ { \frac { 1 - q } { 2 } } \cdot \frac { \Gamma \left( \frac { q } { q - 1 } - \frac { n } { 2 } \right) / \left( \Gamma \left( \frac { 1 } { q - 1 } - \frac { n } { 2 } \right) \right) ^ { q } } { \Gamma \left( \frac { q } { q - 1 } \right) / \left( \Gamma \left( \frac { 1 } { q - 1 } \right) \right) ^ { q } } \cdot \Sigma .\tag{32}
$$

Proof. We note that

$$
\begin{array} { r l r } & { } & { \displaystyle \int _ { { \mathbb R } ^ { n } } p ^ { q } \left( x ; q , \Sigma \right) d x = \displaystyle \int _ { { \mathbb R } ^ { n } } \left( \frac { 1 } { \left| \pi \Lambda \right| ^ { 1 / 2 } } \cdot \frac { \Gamma \left( \frac { 1 } { q - 1 } \right) } { \Gamma \left( \frac { 1 } { q - 1 } - \frac n 2 \right) } \cdot \left( 1 + \left. x , \Lambda ^ { - 1 } x \right. \right) ^ { \frac { 1 } { 1 - q } } \right) ^ { q } d x } \\ & { } & { = \left( \frac { 1 } { \left| \pi \Lambda \right| ^ { 1 / 2 } } \cdot \frac { \Gamma \left( \frac { 1 } { q - 1 } \right) } { \Gamma \left( \frac { 1 } { q - 1 } - \frac n 2 \right) } \right) ^ { q } \cdot \displaystyle \int _ { { \mathbb R } ^ { n } } \left( \left( 1 + \left. x , \Lambda ^ { - 1 } x \right. \right) ^ { \frac { 1 } { 1 - q } } \right) ^ { q } d x } \end{array}
$$

$$
\begin{array} { r l } { \mathbf { \frac { 1 } { 2 } } } & { \mathbf { \frac { 1 } { 3 } } ( \mathbf { \frac { x } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) } \\ { \mathbf { \frac { 1 } { 6 } } ( \mathbf { \frac { x } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) } \\ { \mathbf { \frac { 1 } { 6 } } ( \mathbf { \frac { x } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) } \\ { \mathbf { \frac { 1 } { 6 } } ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) ( \mathbf { \frac { 1 } { 2 } } - \mathbf { \frac { x } { 3 } } ) } \\  \mathbf  \frac { 1 } { 6 }  \end{array}
$$

From (13), we then have the following expression for the q−variance-covariance matrix

$$
\begin{array} { r l } { \mathbb { E } _ { q } [ X X ^ { T } ] = \int _ { \mathbb { R } ^ { n } } \mathbf { x } ^ { \lambda T ^ { \prime } } \cdot p ^ { i } ( x ; q , \Sigma ) d x } \\ & { = \int _ { \mathbb { R } ^ { n } } p ^ { i } ( x ; q , \Sigma ) d x \cdot \Sigma } \\ & { = | \vec { x } \lambda | ^ { \frac { 1 - n } { 2 } } \cdot \frac { \Gamma ( \frac { n } { q - 1 } - \frac { n } { 2 } ) / \Gamma ( \Gamma ( \frac { 1 } { q - 1 } - \frac { n } { 2 } ) ) ^ { q } } { \Gamma ( \frac { n - 1 } { q - 1 } ) \Gamma ( \Gamma ( \frac { 1 } { q - 1 } ) ) ^ { q } } \cdot \Sigma } \\ & { = m ^ { \frac { 1 - n } { 2 } } | \pi \Sigma | ^ { \frac { 1 - n } { 2 } } \cdot \frac { \Gamma ( \frac { n } { q - 1 } - \frac { n } { 2 } ) / \Gamma ( \Gamma ( \frac { 1 } { q - 1 } - \frac { n } { 2 } ) ) ^ { q } } { \Gamma ( \frac { n - 1 } { q - 1 } ) \Gamma ( \frac { 1 } { q - 1 } ) ^ { q } } \cdot \Sigma } \\ & { = ( \frac { 1 } { q - 1 } - \frac { n } { 2 } ) ^ { \frac { 1 - n } { 2 } } | \pi \Sigma | ^ { \frac { 1 - n } { 2 } } \cdot \frac { \Gamma ( \frac { n } { q - 1 } - \frac { n } { 2 } ) / \Gamma ( \frac { n - n } { q - 1 } - \frac { n } { 2 } ) ^ { q } } { \Gamma ( \frac { n - 1 } { q - 1 } ) ^ { q } } \cdot \Gamma ( \Gamma ( \frac { 1 } { q - 1 } - \frac { n } { 2 } ) ) ^ { q } . } \end{array}\tag{33}
$$

Remark 4. Multivariate t distributions with degree of freedom of ${ \frac { 2 } { q - 1 } } - n$ and the scale matrix of Σ are qGaussian with shape parameter $q$ and scale matrix $\Sigma$

Remark 5. The multivariate Cauchy distributions [31] in $\mathbb { R } ^ { n }$ with scale matrix $\scriptstyle { \frac { 1 } { 2 } } \Sigma$ are qGaussian with shape parameter $\textstyle q = 1 + { \frac { 2 } { n + 1 } }$ and scale matrix Σ.

Remark 6. For a random vector X ∼ qGaussian $( q , \Sigma )$ , the variance-covariance matrix exists if and only if $\begin{array} { r } { q \ < \ 1 + \ \frac { 2 } { n + 2 } ; } \end{array}$ following the same procedure to derive the variancecovariance matrix for multivariate t distribution yields that, if existing,

$$
\mathbb { E } \left[ X X ^ { T } \right] = \frac { m } { m - 2 } \Sigma ,\tag{34}
$$

where $\begin{array} { r } { m = \frac { 2 } { q - 1 } - n } \end{array}$

The remarks above on qGaussian distributions reveal their flexibility in incorporating a location parameter, $\mu ,$ and adapting to multivariate contexts through detailed formulas. Remarkably, these distributions bridge with the class of multivariate bell curve distributions including Gaussian, scaled $t ,$ and Cauchy distributions under certain conditions on the shape parameter $q .$ The elaboration of q−correlation and the q−variable-covariance matrix underscores the capability of these distributions to model and understand the intricacies of correlated data efectively.

## 4 Proximal Conjugate Gradient Algorithm

As delineated in Section 5.3, our optimization scenario is predominantly quadratic in nature; therefore, the conjugate gradient approach has potential for fast convergence and numerical stability. This insight forms the basis for our introduction of a proximal conjugate gradient algorithm framework, tailored to navigate the complexities introduced by the nonconvex penalized qGaussian likelihood function for sparse statistical learning. To lay the groundwork for this discussion, we begin with a overview of relevant concepts in variational and nonsmooth analysis, presented in Section 4.1. The results presented in Section 4.1 can be found in recent textbooks on variational and nonsmooth analysis, such as [46, 10, 35, 36, 37, 3].

## 4.1 A Review on Variational and Nonsmooth Analysis

Let $\mathcal { C } ^ { k , \alpha _ { H } }$ with $k \in \mathbb { N } _ { > 0 }$ and $\alpha _ { H } \in [ 0 , 1 ]$ denote the function space such that $\forall F \in { \mathcal { C } } ^ { k , \alpha _ { H } }$ F is kth continuously diferentiable, and $D ^ { k } F$ is globally H¨older continuous with exponent $\alpha _ { H } ;$ clearly, when $\alpha _ { H } = 1 , D ^ { k } F$ is globally Lipschitz continuous. In this subsection, we will state the results from variational and nonsmooth analysis related to the following optimization problem:

$$
\operatorname* { m i n } _ { \substack { \boldsymbol { x } \in \mathbb { R } ^ { p + 1 } \boldsymbol { f } \left( \boldsymbol { x } \right) : = \boldsymbol { g } \left( \boldsymbol { x } \right) + \boldsymbol { h } \left( \boldsymbol { x } \right) , } }\tag{35}
$$

where $f \in { \mathcal { C } } ^ { 0 , 0 } \left( \mathbb { R } ^ { p + 1 } , \mathbb { R } \right)$ is a locally-Lipschitz proper function, $g \in { \mathcal { C } } ^ { 1 , 1 } \left( \mathbb { R } ^ { p + 1 } , \mathbb { R } \right)$ is globally $L _ { \nabla g } .$ −smooth and possibly nonconvex, and $h \in \mathcal { C } ^ { 0 , 0 } \left( \mathbb { R } ^ { p + 1 } , \mathbb { R } \right)$ is a convex locally-Lipschitz function, possibly nonsmooth. The globally Lipschitz property of $\nabla g$ can be alternatively addressed by carrying out the optimization over a compact set. In such scenarios, given that $\nabla g$ is locally Lipschitz, it inherently becomes globally Lipschitz when restricted to a compact set.

Results from convex analysis suggest that $g , h$ are Clarke regular; thus, f is Clarke regular. The Clarke’s directional derivative, defined by

$$
\begin{array} { l } { { f ^ { \circ } ( x ; d ) : = \displaystyle \operatorname* { l i m } _ { y  x } \operatorname* { s u p } _ { t \searrow 0 } \frac { f ( y + t d ) - f ( y ) } { t } \qquad } } \\ { { \qquad = \displaystyle \operatorname* { i n f } _ { \delta > 0 } \operatorname* { s u p } _ {  y - x  \leq \delta , 0 < t < \delta } \frac { f ( y + t d ) - f ( y ) } { t } , } } \end{array}
$$

exists for all $x \in \mathbb { R } ^ { p + 1 }$ since $f$ is Clarke regular. The Clarke subdiferential, denoted by $\partial _ { \circ }$ is a set-valued mapping defined by

$$
\partial _ { \circ } f \left( x \right) : = \left\{ \phi \in \mathbb { R } ^ { p + 1 } | \forall d \in \mathbb { R } ^ { p + 1 } , \left. \phi , d \right. \leq f ^ { \circ } \left( x ; d \right) \right\} .\tag{36}
$$

Since $f$ is a locally Lipschitz function, $\forall x \in \mathbb { R } ^ { p + 1 } , ~ \partial _ { \circ } f \left( x \right) \neq \varnothing$ . Fundamental convex analysis results show that $\forall x \in \mathbb { R } ^ { p + 1 } , ~ \partial _ { \circ } f \left( x \right)$ is compact, convex, and upper-semicontinuous. $\forall x , d \in \mathbb { R } ^ { p + 1 }$ , and we also have

$$
f ^ { \circ } \left( x ; d \right) = \operatorname* { m a x } _ { \mathbf { \Phi } } { \mathbf { \Phi } } _ { u \in \partial \circ f \left( x \right) } \left. u , \frac { d } { \| d \| } \right. .\tag{37}
$$

Furthermore, (37) is upper-semicontinuous with respect to x. Simple convex geometry results conclude that

$$
\left\{ \left( v , - 1 \right) \left| v \in \partial _ { \circ } f \left( x \right) \right. \right\} = N _ { \mathrm { e p i } f } \left( x , f \left( x \right) \right) ,\tag{38}
$$

where $N _ { \mathrm { e p i } \textit { f } } ( x , f \left( x \right) )$ denotes the normal cone to epi f at the point $( x , f \left( x \right) )$

Since g is smooth, $\partial _ { \circ } g \left( \boldsymbol { x } \right) = \left\{ \nabla g \left( \boldsymbol { x } \right) \right\}$ is a singleton. Then $\partial _ { \circ } f \left( x \right) = \partial _ { \circ } g \left( x \right) + \partial _ { \circ } h \left( x \right)$ and

$$
\begin{array} { l } { f ^ { \circ } \left( x ; d \right) = \operatorname* { m a x } _ { \begin{array} { l } { u \in \partial _ { \circ } f \left( x \right) } \\ { u \in \partial _ { \circ } f \left( x \right) + \partial _ { \circ } h \left( x \right) } \end{array} } \left. u , \displaystyle \frac { d } { \| d \| } \right. } \\ { = \operatorname* { m a x } _ { \begin{array} { l } { u \in \left( \nabla g \left( x \right) + \partial _ { \circ } h \left( x \right) \right) } \\ { \left( \overline { { \nabla } } g \left( x \right) , \displaystyle \frac { d } { \| d \| } \right) + \operatorname* { m a x } _ { \begin{array} { l } { v \in \partial _ { \circ } h \left( x \right) } \\ { v \in \partial _ { \circ } h \left( x \right) } \end{array} } \left. v , \displaystyle \frac { d } { \| d \| } \right. } \end{array} } } \\ { = \left. \nabla g \left( x \right) , \displaystyle \frac { d } { \| d \| } \right. + \operatorname* { m a x } _ { \begin{array} { l } { v \in \partial _ { \circ } h \left( x \right) } \\ { v \in \left( x ; d \right) + h ^ { \circ } \left( x ; d \right) . } \end{array} } } \end{array}\tag{39}
$$

Let

$$
M _ { \rho } t \left( x \right) : = \left( t \bigtriangledown \left( \frac { 1 } { 2 \rho } \left\| \cdot \right\| ^ { 2 } \right) \right) \left( x \right) = \operatorname* { i n f } _ { y \in \mathbb { R } ^ { p + 1 } } t \left( y \right) + \frac { 1 } { 2 \rho } \left\| y - x \right\| ^ { 2 }\tag{40}
$$

denote the Moreau envelope operator parameterized by $\rho \in \mathbb { R } _ { > 0 }$ applied on an arbitrary proper, lower semi-continuous, locally Lipschitz function $t \in \mathcal { C } ^ { 0 , 0 } \left( \mathbb { R } ^ { p + 1 } , \mathbb { R } \right)$ , where “□” denotes the infimal convolution operator. We have that the Moreau envelope is a smoothing operator, specifically,

$$
\mathrm {  ~ \ e p i ~ } t + \mathrm {  ~ \ e p i ~ } \frac { 1 } { 2 \rho } \left\| \cdot \right\| ^ { 2 } \subseteq \mathrm { \ e p i ~ } M _ { \rho } t ,\tag{41}
$$

where “epi” denotes the epigraph. Clearly, $M _ { \rho } t \left( x \right) \leq t \left( x \right)$ , since $\left( 0 , 0 \right) \in \mathrm { e p i } \ \frac { 1 } { 2 \rho } \left\| \cdot \right\| ^ { 2 }$ implies that epi $\begin{array} { r } { t = \mathrm { e p i } \ t + ( 0 , 0 ) \subseteq \mathrm { e p i } \ t + \mathrm { e p i } \ \frac { 1 } { 2 \varrho } \left. \cdot \right. ^ { 2 } \subseteq \mathrm { e p i } \ M _ { \rho } t } \end{array}$ . When t is convex, (41) takes the equal sign; i.e., the infimal convolution becomes the exact infimal convolution.

Consider the afine function

$$
A \left( x \right) : = \left. a , x \right. + b ,\tag{42}
$$

simple algebra shows that the Moreau envelope applied on A is

$$
M _ { \rho } A \left( x \right) = \left. a , x \right. + b + \frac { \rho } { 2 } \left. a \right. ^ { 2 } = A \left( x \right) + \frac { \rho } { 2 } \left. a \right. ^ { 2 }\tag{43}
$$

for some $a , b \in \mathbb { R } ^ { p + 1 }$ . Moreover, the following afine addition property is often used in proximal algorithms, mainly due to the fact that the epigraph of an afine function is a half-space that the Moreau envelope applied on:

$$
M _ { \rho } \left( t + A \right) \left( x \right) = M _ { \rho } t \left( x - \rho a \right) + \left. a , x \right. + b - \frac { \rho } { 2 } \left. a \right. ^ { 2 }\tag{44}
$$

Let

$$
\mathrm { p r o x } _ { \rho t } \left( x \right) : = \arg M _ { \rho } t \left( x \right) = a r g m i n \ t \left( y \right) + \frac { 1 } { 2 \rho } \left\| y - x \right\| ^ { 2 }\tag{45}
$$

denote the proximal operator, a set-valued mapping; we have

$$
\mathrm { p r o x } _ { \rho t } = ( I + \rho \partial _ { \circ } t ) ^ { - 1 }\tag{46}
$$

is the resolvent of the Clarke’s subdiferential operator $\rho \partial _ { \circ } t$

For nonsmooth problems, proximal methods are often used. Fundamental convex analysis results show that:

1. the Moreau envelope $M _ { \rho } t \left( x \right)$ is twice diferentiable; thus, its gradient $\nabla M _ { \rho } t \left( \boldsymbol { x } \right)$ is well-defined.

2. If t is convex, $\operatorname { p r o x } _ { \rho t } \left( x \right)$ is a singleton. For the sake of parsimony, with a slight abuse of notation, we use $p r o x _ { \rho t }$ to represent a function in this case. It follows that both $\operatorname { p r o x } _ { \rho t }$ and $\nabla M _ { \rho } t$ are firmly non-expansive, and Moreau’s decomposition theorem implies that

$$
\begin{array} { r } { \nabla M _ { \rho } t \left( x \right) = \rho ^ { - 1 } \left( x - \mathrm { p r o x } _ { \rho t } \left( x \right) \right) . } \end{array}\tag{47}
$$

The results from variational and nonsmooth analysis in this subsection have laid the foundation for proving the properties discussed in Section 4.2.

## 4.2 Proximal Conjugate Gradient Framework

Proximal methods are powerful optimization techniques and are particularly adept at handling problems characterized by sparsity, which usually leads to an optimization problem that is nonsmooth [40]. Proximal algorithm tends to outperform other methods by far for nonsmooth problems [64, 32]. On another ground, Krylov subspace methods represent a cornerstone of numerical analysis, providing a powerful framework for solving largescale optimization problems eficiently [49]. Krylov subspace methods exhibit a remarkable property of convergence acceleration and vastly improved numerical stabili $\mathrm { t y , }$ making them indispensable tools in the numerical analyst’s toolkit.

Having reviewed the related results from variational and non-smooth analysis in Section 4.1, we are ready to introduce our main optimization framework to combine proximal methods and conjugate gradient together. The essence of proximal algorithms lies upon the Moreau envelope’s smoothing on the objective function. Indeed, proximal methods minimize $M _ { \rho } f$ instead of $f ,$ thus avoiding nonsmoothness since $M _ { \rho } f$ is a smooth function. In this view, proximal algorithms are, in fact, minimizing the Moreau envelope of the objective function. Thus, a wide class of numerical optimization algorithms can easily have their proximal version. Among those, conjugate gradients, a type of Krylov subspace method, are the state-of-the-art methods in smooth optimization due to their computational and memory eficiency, scalability, and numerical stability.

Prior to introducing our proximal conjugate gradient update framework, we will first show the equivalency of the optimization problem to minimize (35) and its the Moreau envelope. In nonconvex optimization, the main task for numerical optimization is to find a Clarke stationary point of the objective function, for which we show in Theorem 7 that the set of Clarke stationary point of $f$ is identical to that of $M _ { \rho } f$ for $\rho \in \left( 0 , L _ { \nabla g } ^ { - 1 } \right)$

Lemma 7. $\forall \bar { x } \in \mathbb { R } ^ { p + 1 } , \rho \in \left( 0 , L _ { \nabla g } ^ { - 1 } \right)$

$$
0 \in \partial _ { \circ } f \left( { \bar { x } } \right) \Leftrightarrow \nabla M _ { \rho } f \left( { \bar { x } } \right) = 0 ,\tag{48}
$$

Proof. Consider arbitrary $x \in \mathbb { R } ^ { p + 1 } , \ \rho \in \left( 0 , L _ { \nabla g } ^ { - 1 } \right)$ . As discussed previously, the gradient of the Moreau envelope $\nabla M _ { \rho } f \left( x \right) = \rho ^ { - 1 } \left( \dot { x } - \mathrm { p r o x } _ { \rho f } \left( x \right) \right)$ implies that

$$
\mathrm { p r o x } _ { \rho f } \left( x \right) = x - \rho \nabla M _ { \rho } f \left( x \right) ,\tag{49}
$$

which implies the following first-order (necessary) optimality condition for Clarke’s stationary point:

$$
0 \in \rho ^ { - 1 } \left( x - \rho \nabla M _ { \rho } f \left( x \right) - x \right) + \partial _ { 0 } f \left( x - \rho \nabla M _ { \rho } f \left( x \right) \right) .\tag{50}
$$

The relation above is simplified to

$$
\nabla M _ { \rho } f \left( x \right) \in \partial _ { \circ } f \left( x - \rho \nabla M _ { \rho } f \left( x \right) \right) = \nabla g \left( x - \rho \nabla M _ { \rho } f \left( x \right) \right) + \partial _ { \circ } h \left( x - \rho \nabla M _ { \rho } f \left( x \right) \right) .\tag{51}
$$

Consider arbitrary $\bar { x } \in \mathbb { R } ^ { p + 1 } , \rho \in \left( 0 , L _ { \nabla g } ^ { - 1 } \right)$

“⇒” of (48):

Let $0 \in \partial _ { \circ } f \left( { \bar { x } } \right) = \nabla g \left( { \bar { x } } \right) + \partial _ { \circ } h \left( { \bar { x } } \right)$ ; i.e., ¯x is a Clarke stationary point of $f .$ . Then $- \nabla g \left( \bar { x } \right) \in \partial _ { \circ } h \left( \bar { x } \right)$ . Since h is convex, (51) implies that

$$
\left. - \nabla g \left( \bar { x } \right) - \left( \nabla { M _ { \rho } } f \left( \bar { x } \right) - \nabla g \left( \bar { x } - \rho \nabla { M _ { \rho } } f \left( \bar { x } \right) \right) \right) , \rho \nabla { M _ { \rho } } f \left( \bar { x } \right) \right. \geq 0 .\tag{52}
$$

Simplification gives

$$
\begin{array} { r } { \left. \nabla g \left( \bar { x } - \rho \nabla M _ { \rho } f \left( \bar { x } \right) \right) - \nabla g \left( \bar { x } \right) , \nabla M _ { \rho } f \left( \bar { x } \right) \right. \geq \left. \nabla M _ { \rho } f \left( \bar { x } \right) \right. ^ { 2 } . } \end{array}\tag{53}
$$

By Cauchy-Schwartz inequality,

$$
\begin{array} { r } { \langle \nabla g \left( \bar { x } - \rho \nabla M _ { \rho } f \left( \bar { x } \right) \right) - \nabla g \left( \bar { x } \right) , \nabla M _ { \rho } f \left( \bar { x } \right) \rangle \leq L _ { \nabla g } \cdot \rho \left. \nabla M _ { \rho } f \left( \bar { x } \right) \right. ^ { 2 } . } \end{array}\tag{54}
$$

Since $\rho < L _ { \nabla g } ^ { - 1 } ,$ (53) and (54) imply that

$$
\begin{array} { r } { \| \nabla M _ { \rho } f \left( \bar { x } \right) \| ^ { 2 } \leq \langle \nabla g \left( \bar { x } - \rho \nabla M _ { \rho } f \left( \bar { x } \right) \right) - \nabla g \left( \bar { x } \right) , \nabla M _ { \rho } f \left( \bar { x } \right) \rangle < \| \nabla M _ { \rho } f \left( \bar { x } \right) \| ^ { 2 } , } \end{array}\tag{55}
$$

which implies that

$$
\nabla M _ { \rho } f \left( { \bar { x } } \right) = 0 ;\tag{56}
$$

i.e., ¯x is the stationary point of $M _ { \rho } f$ , hence a Clarke’s stationary point.

$^ { 6 6 } \Leftarrow = ^ { 5 9 }$ of (48):

Let $\nabla f _ { \rho } \left( { \bar { x } } \right) = 0$ ; i.e. ¯x is a stationary point of $M _ { \rho } f$ . It follows directly from (51) that

$$
0 = \nabla M _ { \rho } f \left( \bar { x } \right) \in \partial _ { \circ } f \left( \bar { x } - \rho \nabla M _ { \rho } f \left( \bar { x } \right) \right) = \partial _ { \circ } f \left( \bar { x } \right) ;\tag{57}
$$

i.e., ¯x is a Clarke stationary point of $f .$

The vast majority of optimization algorithms for smooth objective functions require Lipschitz continuity of the objective function. Thus, we are to propose the following Lemma to show the Lipschitz continuity of the gradient of the Moreau envelope of $f .$

Lemma 8. $\forall \rho \in \left( 0 , L _ { \nabla g } ^ { - 1 } \right) , \exists L _ { \nabla M _ { \rho } f } \in \mathbb { R } _ { > 0 }$ such that

$$
\forall x , y \in \mathbb { R } ^ { p + 1 } , \| \nabla M _ { \rho } f \left( x \right) - \nabla M _ { \rho } f \left( y \right) \| \leq L _ { \nabla M _ { \rho } f } \| x - y \| .\tag{58}
$$

Proof. Consider arbitrary $x , y \in \mathbb { R } ^ { p + 1 }$ . From (51), since $h$ is convex,

$$
\begin{array} { r l } & { \langle \nabla M _ { \rho } f \left( x \right) - \nabla g \left( x - \rho \nabla M _ { \rho } f \left( x \right) \right) - \left( \nabla M _ { \rho } f \left( y \right) - \nabla g \left( y - \rho \nabla M _ { \rho } f \left( y \right) \right) \right) , x } \\ & { \quad - \rho \nabla M _ { \rho } f \left( x \right) - \left( y - \rho \nabla M _ { \rho } f \left( y \right) \right) \rangle \geq 0 . } \end{array}\tag{59}
$$

Simplification gives

$$
\begin{array} { r l } & { \langle \nabla M _ { \rho } f \left( x \right) - \nabla M _ { \rho } f \left( y \right) - \left( \nabla g \left( x - \rho \nabla M _ { \rho } f \left( x \right) \right) - \nabla g \left( y - \rho \nabla M _ { \rho } f \left( y \right) \right) \right) , x } \\ & { \quad - y - \rho \left( \nabla M _ { \rho } f \left( x \right) - \nabla M _ { \rho } f \left( y \right) \right) \rangle \geq 0 . } \end{array}\tag{60}
$$

Let $\delta _ { \nabla M _ { \rho } f } : = \nabla M _ { \rho } f \left( x \right) - \nabla M _ { \rho } f \left( y \right) , \delta _ { \nabla g } : = \nabla g \left( x - \rho \nabla f _ { \rho } \left( x \right) \right) - \nabla g \left( y - \rho \nabla f _ { \rho } \left( y \right) \right)$ , and $\delta _ { x , y } : = x - y$ , then

$$
\begin{array} { r l } {  { 0 \leq  \delta _ { \nabla M \rho } f - \delta _ { \nabla g } , \delta _ { x , y } - \rho \delta _ { \nabla M \rho } f  } } \\ & { = - \rho \| \delta _ { \nabla M \rho } f \| ^ { 2 } + \rho  \delta _ { \nabla g } , \delta _ { \nabla M \rho } f  +  \delta _ { \nabla M \rho } f , \delta _ { x , y }  -  \delta _ { \nabla g } , \delta _ { x , y }  } \\ & { \leq - \rho \| \delta _ { \nabla M \rho } f \| ^ { 2 } + \rho \| \delta _ { \nabla g } \| \cdot \| \delta _ { \nabla M \rho } f \| + \| \delta _ { \nabla M \rho } f \| \cdot \| \delta _ { x , y } \| + \| \delta _ { \nabla g } \| \cdot \| \delta _ { x , y } \| } \\ & { \leq - \rho \| \delta _ { \nabla M \rho } f \| ^ { 2 } + \rho L _ { \nabla g } ( \| \delta _ { x , y } \| + \rho \| \delta _ { \nabla M \rho } f \| ) \cdot \| \delta _ { \nabla M \rho } f \| } \\ & { \quad + \| \delta _ { \nabla M \rho } f \| \cdot \| \delta _ { x , y } \| + L _ { \nabla g } ( \| \delta _ { x , y } \| + \rho \| \delta _ { \nabla M \rho } f \| ) \cdot \| \delta _ { x , y } \| } \end{array}
$$

Simplification of the above inequality gives

$$
\left\| \delta _ { \nabla M _ { \rho } f } \right\| \leq \frac { 2 L _ { g } \rho + 1 + \sqrt { 8 L _ { g } \rho + 1 } } { 2 \rho \left( 1 - L _ { \nabla g } \rho \right) } \left\| \delta _ { x , y } \right\| ;\tag{61}
$$

i.e.,

$$
\begin{array} { r } { \Vert \nabla M _ { \rho } f \left( x \right) - \nabla M _ { \rho } f \left( y \right) \Vert \leq L _ { \nabla M _ { \rho } f } \left. x - y \right. , } \end{array}\tag{62}
$$

where

$$
L _ { \nabla M _ { \rho } f } : = \frac { 2 L _ { g } \rho + 1 + \sqrt { 8 L _ { g } \rho + 1 } } { 2 \rho \left( 1 - L _ { \nabla g } \rho \right) } > 0 .\tag{63}
$$

Following this idea, we introduce our proximal conjugate gradient framework in Algorithm 1. □

Algorithm 1 Proximal Point Algorithm   
1: Input: A fixed value of $\rho \in \left( 0 , \rho ^ { - 1 } \right)$   
2: Calculate the gradient of the Moreau envelope: $\mathsf { \Lambda } _ { S } ^ { ( k ) } : = \nabla M _ { \rho } f \left( \boldsymbol { x } ^ { ( k ) } \right)$   
3: $d ^ { ( k ) } : = - s ^ { ( k ) } + \beta ^ { ( k ) } \cdot d ^ { ( k - 1 ) }$   
4: Line search to find $\alpha ^ { ( k ) }$ for the update $x ^ { ( k + 1 ) } : = x ^ { ( k ) } + \alpha ^ { ( k ) } d ^ { ( k ) }$   
5: Update $x ^ { ( k + 1 ) } : = x ^ { ( k ) } + \alpha ^ { ( k ) } d ^ { ( k ) }$

In the above algorithm, $\beta ^ { ( k ) }$ is the conjugate parameter. The significant meaning of Algorithm 1 is that for any global convergent numerical method to find the equilibria of a globally Lipschitz flow, which generally include the global convergent first-order methods, Algorithm 1 can transform such a method to a proximal counterpart.

For some objective functions, the gradient of the Moreau envelope can be calculated directly. However, calculation for the Moreau envelope’s gradient is not tractable for many objective functions whose smooth component $g$ is of complicated form. Motivated by this, we further consider the following the Moreau envelope of the objective function with linearized $^ { g , }$ such linearization step is frequently used in proximal algorithms for statistical sparse learning problems (e.g., [38, 20, 63]).

Consider the linearized surrogate of (35), the locally Lipschitz function $\tilde { f } \in \mathcal { C } ^ { 0 , 0 } \left( \mathbb { R } ^ { p + 1 } , \mathbb { R } \right)$ defined by

$$
\tilde { f } \left( x ; u \right) : = \left. u , x \right. + h \left( x \right)\tag{64}
$$

$$
\mathrm { p r o x } _ { \rho \tilde { f } } \left( x ; u \right) = a r g m i n \ \left\{ \langle u , y \rangle + \frac { 1 } { 2 \rho } \left. y - x \right. ^ { 2 } + h \left( y \right) \right\}\tag{65}
$$

$$
\nabla _ { x } M _ { \rho } \tilde { f } \left( x ; u \right) = \rho ^ { - 1 } \left( x - \mathrm { p r o x } _ { \rho \tilde { f } } \left( x ; u \right) \right)\tag{66}
$$

prox $\mathsf { \Pi } _ { \rho \tilde { f } } \left( x ; u \right)$ is the proximal operator applied on ${ \tilde { f } } ,$ and $\nabla _ { \boldsymbol { x } } M _ { \rho } \tilde { f } \left( \boldsymbol { x } ; \boldsymbol { u } \right)$ is the gradient of the Moreau envelope of $\tilde { f } .$ The linearization term $\langle u , x \rangle$ in (64) depends on u. Recognize that $\tilde { f } \left( x ; u \right)$ is linearizing the nonconvex smooth component g in (101) when $\boldsymbol { u } = \nabla g \left( \boldsymbol { x } \right)$

We establish several definitions for subsequent utilization. Define the mapping $\tilde { g } _ { \rho } =$ $I - \rho \nabla g \in \mathcal { C } ^ { 0 , 0 } \left( \mathbb { R } ^ { p + 1 } , \mathbb { R } ^ { p + 1 } \right)$ for some $\rho \in \left( 0 , L _ { \nabla g } ^ { - 1 } \right)$ , the locally Lipschitz property of $\tilde { g } _ { \rho }$

follows from $g \in { \mathcal { C } } ^ { 1 , 1 } ; { \mathrm { i . e . , ~ } } { \tilde { g } } _ { \rho } \left( x \right) : = x - \rho \nabla g \left( x \right)$ . The following Lemma identifies some fundamental property of $\tilde { g } _ { \rho }$

Lemma 9. $\tilde { g } _ { \rho }$ is a bijective from $\mathbb { R } ^ { p + 1 } \ t o \ \mathbb { R } ^ { p + 1 }$ , and $\tilde { g } _ { \rho } ^ { - 1 }$ is globally Lipschitz with constant $( 1 - \rho L _ { \nabla g } ) ^ { - 1 }$

Proof. Injectivity proof:

Consider arbitrary $x _ { 1 } , x _ { 2 } \in \mathbb { R } ^ { p + 1 }$ . Since $\rho \in \left( 0 , L _ { \nabla g } ^ { - 1 } \right) , x _ { 1 } - \rho \nabla g \left( x _ { 1 } \right) = x _ { 2 } - \rho \nabla g \left( x _ { 2 } \right)$ implies that

$$
\left\| x _ { 1 } - x _ { 2 } \right\| = \rho \left\| \nabla g \left( x _ { 1 } \right) - \nabla g \left( x _ { 2 } \right) \right\| \leq \rho L _ { \nabla g } \left\| x _ { 1 } - x _ { 2 } \right\| < \left\| x _ { 1 } - x _ { 2 } \right\| ,\tag{67}
$$

hence $x _ { 1 } = x _ { 2 }$ . This shows that $\tilde { g } _ { \rho }$ is a injective mapping.

Surjectivity proof:

Consider arbitrary $y _ { 1 } , y _ { 2 } \ \in \ \mathbb { R } ^ { p + 1 }$ . Consider arbitrary $z ~ \in ~ \mathbb { R } ^ { p + 1 }$ . Define mapping $\mathcal { T } \left( y \right) : = z + \rho \nabla g \left( y \right)$ , then

$$
\begin{array} { r l } & { \left. \mathcal { T } \left( y _ { 1 } \right) - \mathcal { T } \left( y _ { 2 } \right) \right. = \left. z + \rho \nabla g \left( y _ { 1 } \right) - \left( z + \rho \nabla g \left( y _ { 2 } \right) \right) \right. } \\ & { \qquad = \rho \left. \nabla g \left( y _ { 1 } \right) - \nabla g \left( y _ { 2 } \right) \right. } \\ & { \qquad \leq \rho L _ { \nabla g } \left. y _ { 1 } - y _ { 2 } \right. } \\ & { \qquad < \left. y _ { 1 } - y _ { 2 } \right. . } \end{array}
$$

Thus, $\tau$ is a contraction mapping, since $\mathbb { R } ^ { p + 1 }$ equipped with Euclidean topology is a Banach space, by Banach fixed point theorem, T has a fixed point; i.e., $\exists y \in \mathbb { R } ^ { p + 1 }$ such that $y = z + \rho \nabla g \left( y \right)$ , or equivalently, $\tilde { g } _ { \rho } \left( y \right) = y - \rho \nabla g \left( y \right) = z$ . Thus, $\mathbb { R } ^ { p + 1 } \subseteq \tilde { g } _ { \rho } \left( \mathbb { R } ^ { p + 1 } \right)$

Globally Lipschitz constant derivation for inverse map:

Since $\nabla g$ is globally $L _ { \nabla g ^ { - 1 } }$ ipschitz,

$$
\begin{array} { r l } & { \| \tilde { g } _ { \rho } \left( y _ { 1 } \right) - \tilde { g } _ { \rho } \left( y _ { 2 } \right) \| = \| y _ { 1 } - \rho \nabla g \left( y _ { 1 } \right) - \left( y _ { 2 } - \rho \nabla g \left( y _ { 2 } \right) \right) \| } \\ & { \qquad = \| y _ { 1 } - y _ { 2 } - \rho \left( \nabla g \left( y _ { 1 } \right) - \nabla g \left( y _ { 2 } \right) \right) \| } \\ & { \qquad \geq | \| y _ { 1 } - y _ { 2 } \| - \| \rho \left( \nabla g \left( y _ { 1 } \right) - \nabla g \left( y _ { 2 } \right) \right) \| } \\ & { \qquad = | \| y _ { 1 } - y _ { 2 } \| - \rho \left\| \nabla g \left( y _ { 1 } \right) - \nabla g \left( y _ { 2 } \right) \right\| | } \\ & { \qquad = \| y _ { 1 } - y _ { 2 } \| - \rho \left\| \nabla g \left( y _ { 1 } \right) - \nabla g \left( y _ { 2 } \right) \right\| } \\ & { \qquad \geq \left( 1 - \rho L _ { \nabla g } \right) \| y _ { 1 } - y _ { 2 } \| } \end{array}\tag{68}
$$

where (68) is due to the fact that

$$
\rho \left\| \nabla g \left( y _ { 1 } \right) - \nabla g \left( y _ { 2 } \right) \right\| \leq \rho L _ { \nabla g } \left\| y _ { 1 } - y _ { 2 } \right\| < \left\| y _ { 1 } - y _ { 2 } \right\| .\tag{69}
$$

Since $\tilde { g } _ { \rho }$ is surjective, consider arbitrary $z _ { 1 } , z _ { 2 } \in \mathbb { R } ^ { p + 1 }$ let $y _ { 1 } : = \tilde { g } _ { \rho } ^ { - 1 } \left( z _ { 1 } \right)$ and $y _ { 2 } : = \tilde { g } _ { \rho } ^ { - 1 }$ (z<sub>2</sub>), then

$$
\left\| \widetilde { g } _ { \rho } ^ { - 1 } \left( z _ { 1 } \right) - \widetilde { g } _ { \rho } ^ { - 1 } \left( z _ { 2 } \right) \right\| \leq \left( 1 - \rho L _ { \nabla g } \right) ^ { - 1 } \left\| z _ { 1 } - z _ { 2 } \right\| .\tag{70}
$$

Define

$$
\mathcal { G } _ { \rho \tilde { f } } \left( x \right) : = \nabla _ { x } M _ { \rho } \tilde { f } \left( x ; u \right)\tag{71}
$$

with $u = \nabla g \left( x \right) ; \mathrm { i . e . , } \mathcal { G } _ { \rho \tilde { f } } \left( x \right)$ is the gradient of the Moreau envelope of $\tilde { f } .$

Similarly to Lemma 7 and 8, we are to prove that the set of Clarke’s stationary of (101) is identical to the set $\left\{ \bar { \boldsymbol { x } } \in \mathbb { R } ^ { p + 1 } | \mathcal { G } _ { \rho \tilde { f } } \left( \bar { \boldsymbol { x } } \right) = 0 \right\}$ in Lemma 10, and then we are to show that (71) is globally Lipschitz in Lemma 11.

Lemma 10. $\forall \bar { x } \in \mathbb { R } ^ { p + 1 } , \rho \in \mathbb { R } _ { > 0 }$

$$
0 \in \partial _ { \circ } f \left( { \bar { x } } \right) \Leftrightarrow { \mathcal G } _ { \rho { \widetilde f } } \left( { \bar { x } } \right) = 0 .\tag{72}
$$

Proof. Consider arbitrary $x \in \mathbb { R } ^ { p + 1 }$ . The $\tilde { f }$ is convex since it is a sum of convex function h and a linear mapping of $x _ { i }$ which is convex.

$$
\mathcal { G } _ { \rho \tilde { f } } \left( \boldsymbol { x } \right) = \rho ^ { - 1 } \left( \boldsymbol { x } - \mathrm { p r o x } _ { \rho \tilde { f } } \left( \boldsymbol { x } ; \nabla g \left( \boldsymbol { x } \right) \right) \right)\tag{73}
$$

$$
= \rho ^ { - 1 } \left( \boldsymbol { x } - \operatorname { p r o x } _ { \rho h } \left( \boldsymbol { x } - \rho \nabla g \left( \boldsymbol { x } \right) \right) \right)\tag{74}
$$

$$
= \nabla g \left( \boldsymbol { x } \right) + \rho ^ { - 1 } \left( \boldsymbol { x } - \rho \nabla g \left( \boldsymbol { x } \right) - \mathrm { p r o x } _ { \rho h } \left( \boldsymbol { x } - \rho \nabla g \left( \boldsymbol { x } \right) \right) \right)
$$

$$
\mathbf { \eta } = \nabla g \left( x \right) + \left( \nabla M _ { \rho } h \right) \circ \tilde { g } _ { \rho } \left( x \right)\tag{75}
$$

(74) is due to the afine addition property of proximal mapping. From (64) and (73),

$$
\begin{array} { r l } & { \mathcal { G } _ { \rho \tilde { f } } ( x ) = \rho ^ { - 1 } \left( x - \mathrm { p r o x } _ { \rho \tilde { f } } ( x , \nabla g \left( x \right) ) \right) } \\ & { \implies \mathrm { p r o x } _ { \rho \tilde { f } } \left( x , \nabla g \left( x \right) \right) = x - \rho \cdot \mathcal { G } _ { \rho \tilde { f } } \left( x \right) } \\ & { \qquad \implies 0 \in \rho ^ { - 1 } \left( x - \rho \cdot \mathcal { G } _ { \rho \tilde { f } } \left( x \right) - x \right) + \partial _ { \circ } \tilde { f } \left( x - \rho \cdot \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \right) } \\ & { \qquad \implies 0 \in - \mathcal { G } _ { \rho \tilde { f } } \left( x \right) + \nabla g \left( x \right) + \partial _ { \circ } h \left( x - \rho \cdot \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \right) } \\ & { \qquad \implies \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \in \nabla g \left( x \right) + \partial _ { \circ } h \left( x - \rho \cdot \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \right) } \\ & { \implies \mathcal { G } _ { \rho \tilde { f } } \left( x \right) - \nabla g \left( x \right) \in \partial _ { \circ } h \left( x - \rho \cdot \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \right) . } \end{array}\tag{76}
$$

Thus, since h is convex, $\forall v \in \partial _ { \circ } h \left( x \right)$

$$
\begin{array} { r l } & { \left. \mathcal { G } _ { \rho \tilde { f } } \left( x \right) - \nabla g \left( x \right) - v , x - \rho \cdot \mathcal { G } _ { \rho \tilde { f } } \left( x \right) - x \right. \geq 0 } \\ & { \qquad \implies \left. \mathcal { G } _ { \rho \tilde { f } } \left( x \right) - \nabla g \left( x \right) - v , \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \right. \leq 0 } \\ & { \qquad \implies \left. \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \right. ^ { 2 } \leq \left. \nabla g \left( x \right) + v , \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \right. } \\ & { \qquad \leq \left. \nabla g \left( x \right) + v \right. \cdot \left. \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \right. } \end{array}\tag{77}
$$

$$
\implies \left. \varphi _ { \rho \tilde { f } } ( x ) \right. \leq \lVert \nabla g ( x ) + v \rVert ,\tag{78}
$$

provided that $\left\| \mathcal { G } _ { \rho \tilde { f } } \left( x \right) \right\| \neq 0$ . Basic results on the Moreau envelope shows that $\mathcal { G } _ { \rho \tilde { f } } \left( x \right) = 0$ implies that x is a Clarke stationary point of $\tilde { f } \left( x \right)$

Now we are proceed to prove (72):

$^ { 6 6 } \Rightarrow ^ { 9 9 } :$

Consider arbitrary $\bar { x } \in \mathbb { R } ^ { p + 1 }$ and $\rho \in \mathbb { R } _ { > 0 }$ . Let $0 \in \partial _ { \circ } f \left( \bar { x } \right) = \nabla g \left( \bar { x } \right) + \partial _ { \circ } h \left( \bar { x } \right)$ ; i.e., ¯x is a Clarke stationary point of $f .$ . Then $\exists v \in \partial _ { \circ } h ( \bar { x } )$ such that $\nabla g \left( \bar { x } \right) + v = 0$ . (78) implies that

$$
\left\| \mathcal { G } _ { \rho \widetilde { f } } \left( \bar { x } \right) \right\| \leq \| \nabla g \left( \bar { x } \right) + v \| = 0 .\tag{79}
$$

Thus, “ $\begin{array} { l } { { \mathcal G _ { \rho \tilde { f } } \left( \bar { x } \right) = 0 } } \\ { { = ^ { \gamma } : } } \end{array}$

Consider arbitrary $\bar { x } \in \mathbb { R } ^ { p + 1 }$ . Let $\mathcal { G } _ { \rho \tilde { f } } \left( \bar { x } \right) = 0 ; \mathrm { i . e . , } \mathcal { G } _ { \rho \tilde { f } } \left( \bar { x } \right) = 0$ is stationary. implies that

(76)

$$
0 = \mathcal { G } _ { \rho \tilde { f } } \left( \bar { x } \right) \in \nabla g \left( \bar { x } \right) + \partial _ { \circ } h \left( \bar { x } - \rho \cdot \mathcal { G } _ { \rho \tilde { f } } \left( \bar { x } \right) \right) = \nabla g \left( \bar { x } \right) + \partial _ { \circ } h \left( \bar { x } \right) = \partial _ { \circ } f \left( \bar { x } \right) .\tag{80}
$$

Thus, ¯x is a Clarke stationary point of f.

Lemma 11. $\forall \rho \in \mathbb { R } _ { > 0 } , \exists L g _ { _ { o \tilde { f } } } \in \mathbb { R } _ { > 0 }$ such that

$$
\forall x , y \in \mathbb { R } ^ { p + 1 } , \ \left\| \mathcal { G } _ { \rho \tilde { f } } \left( x \right) - \mathcal { G } _ { \rho \tilde { f } } \left( y \right) \right\| \leq L _ { \mathcal { G } _ { \rho \tilde { f } } } \| x - y \| .\tag{81}
$$

Proof. Consider arbitrary $x , y \in \mathbb { R } ^ { p + 1 }$ and $\rho \in \mathbb { R } _ { > 0 }$ . Let $\boldsymbol { u } : = \nabla g \left( x \right)$ and $v : = \nabla g \left( y \right)$

$$
\Big \Vert \mathcal { G } _ { \rho \tilde { f } } \left( x \right) - \mathcal { G } _ { \rho \tilde { f } } \left( y \right) \Big \Vert = \Big \Vert \nabla _ { x } M _ { \rho } \tilde { f } \left( x ; u \right) - \nabla _ { y } M _ { \rho } \tilde { f } \left( y ; v \right) \Big \Vert
$$

$$
= \Big \| \nabla _ { x } M _ { \rho } \tilde { f } \left( x ; u \right) - \nabla _ { x } M _ { \rho } \tilde { f } \left( x ; v \right) + \nabla _ { x } M _ { \rho } \tilde { f } \left( x ; v \right) - \nabla _ { y } M _ { \rho } \tilde { f } \left( y ; v \right) \Big \|
$$

$$
\leq \left. \nabla _ { x } M _ { \rho } \tilde { f } \left( x ; u \right) - \nabla _ { x } M _ { \rho } \tilde { f } \left( x ; v \right) \right. + \left. \nabla _ { x } M _ { \rho } \tilde { f } \left( x ; v \right) - \nabla _ { y } M _ { \rho } \tilde { f } \left( y ; v \right) \right.
$$

$$
\begin{array} { r l } {  { \le \| u - v \| + \| \nabla _ { x } { M _ { \rho } } \tilde { f } ( x ; v ) - \nabla _ { y } { M _ { \rho } } \tilde { f } ( y ; v ) \| } } \end{array}\tag{82}
$$

$$
\leq \| u - v \| + \rho ^ { - 1 } \| x - y \|\tag{83}
$$

$$
\leq L _ { \nabla g } \left\| x - y \right\| + \rho ^ { - 1 } \left\| x - y \right\|
$$

$$
= \left( L _ { \nabla g } + \rho ^ { - 1 } \right) \| x - y \|
$$

(82) is due to Lemma 4 in [20], and (83) is due to $\tilde { f } \left( \cdot ; v \right)$ is convex and the fact that the gradient of a convex function’s Moreau envelope is $\rho ^ { - 1 } \mathrm { - L i p s c h i t z }$ . Therefore, let

$$
L _ { \mathcal { G } _ { \rho \tilde { f } } } : = L _ { \nabla g } + \rho ^ { - 1 }\tag{84}
$$

and we have

$$
\left\| \mathcal { G } _ { \rho \tilde { f } } \left( x \right) - \mathcal { G } _ { \rho \tilde { f } } \left( y \right) \right\| \leq L \varsigma _ { \rho \tilde { f } } \left\| x - y \right\| .\tag{85}
$$

Furthermore, (75) suggests that

$$
\begin{array} { r } { \mathcal { G } _ { \rho \tilde { f } } = \nabla g + \left( \nabla M _ { \rho } h \right) \circ \tilde { g } _ { \rho } = \nabla g + \left( \nabla M _ { \rho } h \right) \circ \left( I d - \rho \nabla g \right) . } \end{array}\tag{86}
$$

Hence,

$$
{ I d } - \rho \mathscr { G } _ { \rho \tilde { f } } = I d - \rho \nabla g - \rho \left( \nabla { M } _ { \rho } h \right) \circ \left( { I d } - \rho \nabla g \right)
$$

$$
\mathbf { \eta } = \tilde { g } _ { \rho } - \rho \left( \nabla M _ { \rho } h \right) \circ \tilde { g } _ { \rho }
$$

$$
\mathbf { \eta } = ( I d - \rho ( \nabla M _ { \rho } h ) ) \circ \tilde { g } _ { \rho }\tag{87}
$$

$$
\begin{array} { r } { \mathbf { \sigma } = \tilde { g } _ { \rho } ^ { - 1 } \circ ( \tilde { g } _ { \rho } \circ ( I d - \rho ( \nabla M _ { \rho } h ) ) ) \circ \tilde { g } _ { \rho } } \end{array}\tag{88}
$$

shows that $\tilde { g } _ { \rho } ^ { - 1 } \circ ( \tilde { g } _ { \rho } \circ ( I d - \rho ( \nabla M _ { \rho } h ) ) ) \circ \tilde { g } _ { \rho } : \mathbb { R } ^ { p + 1 } \mapsto \mathbb { R } ^ { p + 1 }$ equals to $I d - \rho \mathcal { G } _ { \rho \tilde { f } } : \mathbb { R } ^ { p + 1 } \mapsto$ $\mathbb { R } ^ { p + 1 }$ . Since $\tilde { g } _ { \rho }$ ibijectivee, and that $\tilde { g } _ { \rho }$ and $\tilde { g } _ { \rho } ^ { - 1 }$ are continuous due to the globally Lipschitz property from Lemma $9 , \tilde { g } _ { \rho }$ is a homeomorphism. Hence, Id− $\cdot \rho \mathcal { G } _ { \rho \tilde { f } }$ and $\tilde { g } _ { \rho ^ { \circ } } ( I d - \rho ( \nabla M _ { \rho } h ) )$ are topologically equivalent mappings via the homeomorphism $\tilde { g } _ { \rho }$ . Lemma 11 implies that $\mathcal { G } _ { \rho \tilde { f } }$ is globally Lipschitz, which suficiently implies by the Cauchy-Lipschitz theorem that the diferential equation

$$
\dot { x } : = \frac { d x } { d t } = \mathcal { G } _ { \rho \tilde { f } } \left( x \right)\tag{89}
$$

has a unique solution for any given initial value condition. Thus, $\mathcal { G } _ { \rho \tilde { f } }$ generates a unique flow under a given initial value condition.

The operator equations presented above can be understood as demonstrating how $\mathcal { G } _ { \rho \tilde { f } }$ functions analogously to a gradient operator. Specifically, $I d - \rho \mathcal { G } _ { \rho \tilde { f } }$ represents executing a descent operation in the $- \mathcal { G } _ { \rho \tilde { f } }$ direction with a step size $\rho .$ Similarly, $I d - \rho \left( \nabla M _ { \rho } h \right)$ represents a single gradient descent step with $\rho$ as the step size with objective function $M _ { \rho } h$ , the Moreau envelope of $h ;$ while $\tilde { g } _ { \rho } = I d - \rho \nabla g$ reflects a gradient descent step with objective function $^ { g , }$ again with $\rho$ as the step size. Equation (87) elucidates that a descent in the $- \mathcal { G } _ { \rho \tilde { f } }$ direction is identical to first performing a one-step gradient descent on $^ { g , }$ followed by $M _ { \rho } h ;$ or performing gradient descents in a converse order yields a topologically equivalence via the homeomorphism $\tilde { g } _ { \rho ; }$ , as shown in (88).

In short summary, the approach based on linearization of the smooth term and the Moreau envelope enables us to build equivalence between identifying Clarke stationary points of the original nonsmooth objective function (101) and finding equilibria of the (unique) flow generated by $\mathcal { G } _ { \rho \tilde { f } } ^ { \mathrm { ~ ~ } } ,$ as demonstrated in Lemma 10. The task of finding equilibria within a globally Lipschitz continuous flow, such as the $\mathcal { G } _ { \rho \tilde { f } }$ flow, is well explored within mathematics, particularly in the realms of dynamical systems and numerical analysis (see, for example, [44, 2, 33, 27, 24]). Cauchy-Lipschitz theorem establishes the uniqueness of solutions to initial value problems for globally Lipschitz continuous flows; while the existence of equilibria is a direct result of Brouwer fixed-point theorem. Numerical methods for dynamical systems, including methods for finding equilibria of the flow, are largely based on this uniqueness result. This is one reason that the vast majority of numerical methods in the context of dynamical systems require the flow to be globally Lipschitz. It is important to note that these numerical strategies, widely applied across dynamical systems, do not hinge on the flow being derived from a conservative field. As such, the process of formulating a potential function for $\mathcal { G } _ { \rho \tilde { f } }$ is not a prerequisite for employing numerical techniques to determine its equilibria. This perspective underscores the versatility of numerical methods in dynamical systems in finding the equilibria of flows, regardless of the explicit existence of a potential function, a stance corroborated by various sources in the literature [44, 2, 33, 27, 24, 47, 45]. In this view, the construction of a potential function for $\mathcal { G } _ { \rho \tilde { f } }$ is generally not necessary when deploying numerical analysis methods to find its equilibria.

In the context of nonlinear conjugate gradient algorithms for optimization, achieving global convergence on nonconvex objective functions that are globally Lipschitz-smooth implies that such methods can reliably find equilibria within the corresponding flow dynamics [47, 45]. These algorithms typically incorporate a line search step, which may use a surrogate objective function instead of the original. This surrogate can be a constructed potential, Lyapunov, or energy function, ofering flexibility when finding the potential function for $\mathcal { G } _ { \rho \tilde { f } }$ poses challenges [47, 9, 54].

When it is feasible to construct a potential function whose gradient with respect to x is $\mathcal { G } _ { \rho \tilde { f } }$ , the associated objective function and its gradient become more manageable, allowing for direct global convergence arguments. If constructing a potential function with respect to x for the $\left( \nabla M _ { \rho } h \right) \circ \tilde { g } _ { \rho } \left( x \right)$ term in (75) or $\nabla g \circ ( I d - \rho ( \nabla M _ { \rho } h ) )$ in (88) is tractable, the objective function with gradient being (75) or $\tilde { g } _ { \rho } \circ ( I d - \rho ( \nabla M _ { \rho } h ) )$ can hence be easily constructed. Thus, arguments for global convergence for methods based on the objective function and its gradient directly follow to prove the global convergence of the numerical optimization algorithm when applied to the constructed potential function for $\mathcal { G } _ { \rho \tilde { f } }$ . We remark that $I d - \rho \mathcal { G } _ { \rho \tilde { f } }$ and $\tilde { g } _ { \rho } \circ ( I d - \rho ( \nabla M _ { \rho } h ) )$ generate two topologically equivalent flows via homeomorphism $\tilde { g } _ { \rho } ;$ thus, their equilibria can be transformed by $\tilde { g } _ { \rho }$ and share the same stability. In the context of numerical optimization, this implies that a fixed point ¯x for the mapping $I d - \rho \mathcal { G } _ { \rho \tilde { f } }$ corresponds bijectively to a fixed point $\tilde { g } _ { \rho } \left( \bar { x } \right)$ for $\tilde { g } _ { \rho } \circ ( I d - \rho ( \nabla M _ { \rho } h ) )$ . Characterized by the first-order optimality condition in optimization of smooth functions, or equivalently, the stationary condition in dynamical system,

$$
\mathcal { G } _ { \rho \bar { f } } \left( \bar { x } \right) = \nabla g \left( \bar { x } \right) + ( \nabla M _ { \rho } h ) \circ \tilde { g } _ { \rho } \left( \bar { x } \right) = 0 \Leftrightarrow \nabla M _ { \rho } h \left( \tilde { g } _ { \rho } \left( \bar { x } \right) \right) + \nabla g \left( \tilde { g } _ { \rho } \left( \bar { x } \right) - \rho \nabla M _ { \rho } h \left( \tilde { g } _ { \rho } \left( \bar { x } \right) \right) \right) = 0 .\tag{90}
$$

This approach is practical because the literature on first-order numerical optimization techniques frequently includes proofs of global convergence for methods that depend on the objective function and its gradient (for example, see [17, 43, 25, 13, 23]). Alternatively, construction of a potential function for $\mathcal { G } _ { \rho \tilde { f } }$ is often not necessary due to the fact that fixed-point methods finding equilibria for a flow mostly establish convergence properties based on Banach fixed point theorem. This theorem guarantees convergence through intrinsic flow properties, obviating the need for a potential function [Burden2001, 2, 1]. Conventionally, the use of line search based on the objective function and its gradient has been applied in some numerical methods to ensure global convergence. However, with the rapid growth of research in high–dimensional statistical machine learning and large-scale optimization, evaluations of the objective function often proven to be ineficient. Consequently, recent years have seen the exploration of two main alternatives. For instance, two diferent types of approaches for global convergent nonlinear conjugate gradient methods have been proposed without the conventional objective function-based line search procedure. One type of approach ensures global convergence by utilizing a line search mechanism that depends only on the nonlinear equation that generates the flow [16, 52, 53, 28]; that is, the gradient function for smooth optimization, or $\mathcal { G } _ { \rho \tilde { f } }$ in our case. As an example, under the smoothness assumption, the first-order optimality condition for an exact line search often solves for α with the current value $x ^ { ( \bar { k } ) }$ and the search direction $d ^ { ( k ) }$ from $\left. \mathcal { G } _ { \rho \tilde { f } } \left( x ^ { ( k ) } + \alpha \cdot d ^ { ( k ) } \right) , d ^ { ( k ) } \right. = 0$ , an equation dependent only on $\mathcal { G } _ { \rho \tilde { f } }$ but not any surrogate objective function. From a practical perspective, this one-dimensional root finding problem can be carried out eficiently using the Brent root finding algorithm [7]. The other approach suggests achieving global convergence either without the need for line search [51, 8, 55, 62, 61, 65] or by meeting a condition related to the Zoutendijk condition to replace the Wolfe-Powell conditions of suficient descent (Armijo) and curvature [39]. Additionally, in scenarios where the fulfillment of a suficient descent (Armijo) condition is imperative, the formulation of a surrogate objective function becomes essential. Considering (75), where a surrogate objective is required for the line search phase, it could be formulated as:

$$
\begin{array} { r l } & { g \left( \boldsymbol { x } \right) + \left( M _ { \rho } \boldsymbol { h } \right) \circ \tilde { g } _ { \rho } \left( \boldsymbol { x } \right) } \\ & { = g \left( \boldsymbol { x } \right) + \left( M _ { \rho } \boldsymbol { h } \right) \circ \tilde { g } _ { \rho } \left( \boldsymbol { x } \right) + \mathrm { c o n s t a n t } } \\ & { = g \left( \boldsymbol { x } \right) + \left. \nabla g \left( \boldsymbol { x } \right) , \mathrm { p r o x } _ { \rho h } \left( \boldsymbol { x } - \rho \nabla g \left( \boldsymbol { x } \right) \right) - \boldsymbol { x } \right. + \frac { 1 } { 2 \rho } \left\| \mathrm { p r o x } _ { \rho h } \left( \boldsymbol { x } - \rho \nabla g \left( \boldsymbol { x } \right) \right) - \boldsymbol { x } \right\| ^ { 2 } } \\ & { \qquad + h \left( \mathrm { p r o x } _ { \rho h } \left( \boldsymbol { x } - \rho \nabla g \left( \boldsymbol { x } \right) \right) \right) + \mathrm { c o n s t a n t } } \end{array}\tag{91}
$$

This formulation, denoted as (91), represents a quadratic approximation of g plus the nonsmooth term $h ,$ evaluated at $\mathrm { p r o x } _ { \rho h } \left( \boldsymbol { x } - \rho \nabla g \left( \boldsymbol { x } \right) \right)$ . This type of formulation has often been used for the line search step in previous studies [4, 29]. The addition of the term $\left. \nabla g \left( x \right) , x \right.$ acts as a constant in (64), analogous to fixing the value of u as $\nabla g \left( x \right)$ for linearization. This constant term, $- \left. \nabla g \left( x \right) , x \right.$ , doesn’t alter the gradient of the Moreau envelope (66) or the proximal point (65), serving to frame the quadratic approximation of $g \left( \mathrm { p r o x } _ { \rho h } \left( \boldsymbol { x } - \rho \nabla g \left( \boldsymbol { x } \right) \right) \right)$

Evaluation of $\mathrm { p r o x } _ { \rho h }$ in (91) is tractable and eficient for many functions, such as the $\ell _ { 1 }$ norm commonly encountered in sparse statistical learning can be eficiently computed via the soft-thresholding function. Given that line search rules such as the Wolfe-Powell or

Armijo-Goldstein conditions require only the diference in the value of the objective function at two points to decide on the step size, the constant term in (91) can be disregarded. Subsequent global convergence arguments stem from the fixed-point theory analysis of the numerical methods deployed to find the equilibria of the $\mathcal { G } _ { \rho \tilde { f } }$ flow. Another possible surrogate objective function inspired by the quadratic Lyapunov function for the $\mathcal { G } _ { \rho \tilde { f } }$ flow could be $\begin{array} { r } { \frac { 1 } { 2 } \left\| \mathcal { G } _ { \rho \tilde { f } } \right\| ^ { 2 } } \end{array}$ , attains its minimal value 0 exactly at the $\mathcal { G } _ { \rho \tilde { f } }$ flow’s equilibria. This quadratic approach simplifies evaluation, but it may not ofer insights into the potential function’s landscape, potentially limiting the numerical algorithm’s acceleration capabilities if such an algorithm uses the landscape information to ensure the suficient descent (Armijo) condition. Therefore, formulating the surrogate objective function preserving the landscape of the original objective function as outlined in (91) is preferable.

Building on the above discussion, we introduce our practical proximal conjugate gradient framework in Algorithm 2.

Algorithm 2 Computationally Tractable Proximal Conjugate Gradient Update Scheme   
1: Input: A fixed value of $\rho \in ( 0 , \rho ^ { - 1 } )$   
2: Calculate the proximal value $p ^ { ( k ) } : = $ prox $\cdot _ { \rho ^ { - 1 } h } \left( x ^ { ( k ) } - \rho ^ { - 1 } \cdot \nabla g \left( x ^ { ( k ) } \right) \right)$   
3: Calculate $\mathcal { G } _ { \rho \tilde { f } } \left( x ^ { ( k ) } \right) : s ^ { ( k ) } : = \rho \left( x ^ { ( k ) } - p ^ { ( k ) } \right)$   
4: $d ^ { ( k ) } : = - s ^ { ( \dot { k } ) ^ { \circ } } + \beta ^ { ( k ) } \cdot d ^ { ( k - 1 ) }$   
5: Line search to find $\alpha ^ { ( k ) }$ for the update $x ^ { ( k + 1 ) } : = x ^ { ( k ) } + \alpha ^ { ( k ) } d ^ { ( k ) }$ , if needed.   
6: Update $x ^ { ( k + 1 ) } : = x ^ { ( k ) } + \alpha ^ { ( k ) } d ^ { ( k ) }$

In Algorithm 2, $\beta ^ { ( k ) }$ functions as the conjugate parameter. Unlike Algorithm 1, Algorithm 2 facilitates the update process without the need to compute $\nabla M _ { \rho } f \left( x ^ { ( k ) } \right)$ . This adaptation is significantly valuable in practical scenarios, especially in statistical sparse learning challenges characterized by a complicated smooth component $g$ alongside a simple nonsmooth convex component h. In such cases, computing $\mathrm { p r o x } _ { \rho h }$ is markedly more tractable and eficient than $\operatorname { p r o x } _ { \rho f }$ This approach is particularly beneficial for sparse statistical learning issues, where sparsity is commonly induced by an $\ell _ { 1 }$ penalty term.

## 4.3 Proximal Hager-Zhang [23] Conjugate Gradient

The nonlinear conjugate gradient method represents the pinnacle of first-order techniques for addressing smooth optimization challenges. Various versions of nonlinear conjugate gradient methods have been introduced, including the Fletcher-Reeves (FR) method [17], the modified Polak-Ribiere-Polyak (PRP+) method [43, 21], the Hestenes-Stiefel (HS) method [25], the Dai-Yuan (DY) method [13], and the Hager-Zhang (HZ) method [23]. These versions have all demonstrated global convergence with nonconvex globally Lipschitz-smooth objective functions. Among these, the Hager-Zhang conjugate gradient method is notable for delivering the best numerical performance on large-scale datasets, as indicated in previous research [22]. Building on this, having introduced our practical proximal conjugate gradient update mechanism in Algorithm 2, we aim to extend this approach by adapting the smooth Hager-Zhang nonlinear conjugate gradient method to its proximal version in Algorithm 3.

```latex
Algorithm 3 Proximal Hager-Zhang [23] Conjugate Gradient
1: Input: Initial point $x ^ { ( 0 ) } ; ~ g ~ \in ~ \mathcal { C } ^ { 1 , 1 } \left( \mathbb { R } ^ { p + 1 } , \mathbb { R } \right)$ ; locally-Lipschitz, convex $\begin{array} { r l } { h } & { { } \in } \end{array}$
$\mathcal { C } ^ { 0 , \bar { 0 } } \left( \mathbb { R } ^ { p + 1 } , \mathbb { R } \right)$ ; the smoothing parameter for the Moreau envelope $\rho \in ( 0 , \rho ^ { - 1 } ) ; k : = 0$
2: Output: $p$
3: $k + = 1$
4: Calculate the gradient for $g \colon g ^ { ( 0 ) } : = \nabla g \left( x ^ { ( 0 ) } \right)$
5: Calculate the proximal value $p ^ { ( 0 ) } : = \operatorname { p r o x } _ { \rho , h } \big ( x ^ { ( 0 ) } - \rho \cdot g ^ { ( 0 ) } \big )$
6: Calculate the gradient analog: $\begin{array} { r } { s ^ { ( 0 ) } : = x ^ { ( 0 ) } - \dot { p } ^ { ( 0 ) } } \end{array}$
7: $d ^ { ( 0 ) } : = - s ^ { ( 0 ) }$
8: Perform the line search with $d ^ { ( 0 ) }$ with step size $\alpha ^ { ( 0 ) }$
9: Update $x _ { 1 } : = x ^ { ( 0 ) } + \alpha ^ { ( 0 ) } d ^ { ( 0 ) }$
10: while not converged do
11: $k + = 1$
12: Calculate the gradient for g: $g ^ { ( k ) } : = \nabla g \left( x ^ { ( k ) } \right)$
13: Calculate the proximal value $p ^ { ( k ) } : = \mathrm { p r o x } _ { \rho , h } \left( x ^ { ( k ) } - \rho \cdot g ^ { ( k ) } \right)$
14: Calculate the gradient analog: $\mathring { s } ^ { ( k ) } : = \boldsymbol { x } ^ { ( k ) } - \boldsymbol { \dot { p } } ^ { ( k ) }$
15: $d ^ { ( k ) } : = - s ^ { ( k ) } + \bar { \beta } ^ { ( k ) } \cdot d ^ { ( k - 1 ) }$
16: Perform the line search with $d ^ { ( k ) }$ with step size $\alpha ^ { ( k ) }$ based on Wolfe-Powell condi
tions
17: Update $x ^ { ( k + 1 ) } : = x ^ { ( k ) } + \alpha ^ { ( k ) } d ^ { ( k ) }$
18: Check for convergence
19: return $p ^ { ( k ) }$
```

In Algorithm 3, Hager-Zhang’s conjugate parameter $\bar { \beta } ^ { ( k ) }$ is defined as [23]:

$$
y ^ { ( k ) } : = s ^ { ( k + 1 ) } - s ^ { ( k ) }
$$

$$
\beta ^ { ( k ) } : = \frac { 1 } { \left. d ^ { ( k ) } , y ^ { ( k ) } \right. } \cdot \left. y ^ { ( k ) } - 2 \frac { \left\| y ^ { ( k ) } \right\| ^ { 2 } } { \left. d ^ { ( k ) } , y ^ { ( k ) } \right. } d ^ { ( k ) } , s ^ { ( k + 1 ) } \right.
$$

$$
\eta ^ { ( k ) } : = - \frac { 1 } { \left\| d ^ { ( k ) } \right\| \operatorname* { m i n } \left\{ \eta , \left\| s ^ { ( k ) } \right\| \right\} }
$$

$$
\bar { \beta } ^ { ( k ) } : = \operatorname* { m a x } \left\{ \beta ^ { ( k ) } , \eta ^ { ( k ) } \right\}
$$

It was proven that if the line search step in Algorithm 3 satisfies Wolfe-Powell conditions and the gradient is globally Lipschitz, Hager-Zhang conjugate gradient achieves global convergence finding a stationary point for a smooth nonconvex objective function. In a dynamical system view, this corresponds to the global attraction property of the trajectory of the numerical algorithm to find equilibria for globally Lipschitz flows. Lemma 11 implies that $\mathcal { G } _ { \rho \tilde { f } }$ , or $s ^ { ( k ) }$ in Algorithm 3, are globally Lipschitz. Thus, by Lemma 10, Algorithm 3 yields the Clarke stationary point of $f .$ Based on the arguments in Section 4.2, if the potential function for $\mathcal { G } _ { \rho \tilde { f } }$ is tractable to construct, the Wolfe-Powell line search in Algorithm 3 can be carried out using the potential function of $\mathcal { G } _ { \rho \tilde { f } }$ as the surrogate objective function; alternatively, an exact line search can be carried out by finding α that satisfies $\left. \mathcal { G } _ { \rho \tilde { f } } \left( x ^ { ( k ) } + \alpha \cdot d ^ { ( k ) } \right) , d ^ { ( k ) } \right. = 0$ such an exact line search can usually be carried out eficiently using Brent’s method to find a root of a one-dimensional equation in $\mathbb { R } _ { > 0 }$ [7]. Furthermore, the descent property of $d ^ { ( k ) }$ was shown by [23] independent of the line searches, which guarantees that $\left. \mathcal { G } _ { \rho \tilde { f } } \left( x ^ { ( k ) } + \alpha \cdot d ^ { ( k ) } \right) , d ^ { ( k ) } \right. = 0$ has a positive root. Moreover, another line search to ensure global convergence can be carried out by backtracking to find $\alpha ^ { ( k ) }$ satisfying

$$
- \left. \boldsymbol { \mathcal { G } } _ { \rho \tilde { f } } \left( \boldsymbol { x } ^ { ( k ) } + \boldsymbol { c } _ { 1 } \cdot \boldsymbol { \alpha } ^ { ( k ) } \boldsymbol { d } ^ { ( k ) } \right) , \boldsymbol { d } ^ { ( k ) } \right. \geq c _ { 1 } c _ { 2 } \cdot \boldsymbol { \alpha } ^ { ( k ) } \left\| \boldsymbol { d } ^ { ( k ) } \right\| ^ { 2 } ,\tag{92}
$$

where $c _ { 1 } , c _ { 2 } \in \mathbb { R } _ { > 0 }$ are constant to be chosen. When $\mathcal { G } _ { \rho \tilde { f } }$ is pseudo-monotone in the sense of Karamardian [30], since the global Lipschitz property was established for $\mathcal { G } _ { o \tilde { f } }$ in Lemma 11, global convergence was proven for this backtracking line search method [16]. We conclude this section with the observation that certain conjugate gradient methods obviate the need for line search procedures by determining the step size directly from $s ^ { ( k ) }$ and $d ^ { ( k ) }$ , as exemplified in [8].

## 5 Optimizing Algorithm and Prediction for Penalized qGaussian Likelihood Problems

## 5.1 Problem Formulation

Using the qGaussian distribution to model the data will undoubtedly enhance the robustness towards the underlying distributional assumption and outliers. However, unlike the Gaussian distribution, two independent qGaussian random vectors are not jointly qGaussian. Thus, we take the following approach to model the data. Let $\mathbf { X } _ { \mathrm { t r a i n } } \ \in$ $\bar { \mathbb { R } } ^ { n _ { \mathrm { t r a i n } } \times ( p + 1 ) }$ ， $y _ { \mathrm { t r a i n } } \in \mathbb { R } ^ { n }$ train denote the training design matrix and outcome, ${ \bf X } _ { \mathrm { v a l } } \in$ $\mathbb { R } ^ { n _ { \mathrm { v a l } } \times ( p + 1 ) } , \ y _ { \mathrm { v a l } } \ \in \ \mathbb { R } ^ { n _ { \mathrm { v a l } } }$ denote the validation design matrix and outcome, and $\mathbf { X } _ { \mathrm { t e s t } } \in$ $\mathbb { R } ^ { n _ { \mathrm { t e s t } } \times ( p + 1 ) }$ <sup>1)</sup>, $y _ { \mathrm { t e s t } } \in \mathbb { R } ^ { n _ { \mathrm { t e s t } } }$ denote the testing design matrix and outcome. Let

$$
\mathbf { X } : = \left[ \mathbf { X } _ { \mathrm { t r a i n } } ^ { T } , \mathbf { X } _ { \mathrm { v a l } } ^ { T } , \mathbf { X } _ { \mathrm { t e s t } } ^ { T } \right] ^ { T } \in \mathbb { R } ^ { n \times ( p + 1 ) }\tag{93}
$$

denote the design matrix for the entire dataset, and let

$$
y : = \left[ y _ { \mathrm { t r a i n } } ^ { T } , y _ { \mathrm { v a l } } ^ { T } , y _ { \mathrm { t e s t } } ^ { T } \right] ^ { T } \in \mathbb { R } ^ { n }\tag{94}
$$

denote the outcome for the entire dataset. Instead of assuming the qGaussian distribution for the training, validation and testing set separately, we assume that

$$
y \sim q \mathrm { G a u s s i a n } \left( q , \mathbf { X } \theta , { \boldsymbol { \Sigma } } \right) ,\tag{95}
$$

where $\theta \in \mathbb { R } ^ { p + 1 }$ denotes the coeficients for regression, and Σ denotes the characteristic/scale matrix for the entire data. Clearly,

$$
\begin{array} { r } { \mathbf { X } _ { \mathrm { t r a i n } } = [ I _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { t r a i n } } } , 0 _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { v a l } } } , 0 _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { t e s t } } } ] \mathbf { X } } \\ { y _ { \mathrm { t r a i n } } = [ I _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { t r a i n } } } , 0 _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { v a l } } } , 0 _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { t e s t } } } ] y } \end{array}
$$

implies that

$$
y _ { \mathrm { t r a i n } } \sim q \mathrm { G a u s s i a n } \left( q _ { \mathrm { t r a i n } } , \mathbf { X } _ { \mathrm { t r a i n } } \theta , \boldsymbol { \Sigma } _ { \mathrm { t r a i n } } \right)\tag{96}
$$

where by the linear mapping closeness property 2,

$$
\begin{array} { r } { \Sigma _ { \mathrm { t r a i n } } = [ I _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { t r a i n } } } , 0 _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { v a l } } } , 0 _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { t e s t } } } ] \Sigma [ I _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { t r a i n } } } , 0 _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { v a l } } } , 0 _ { n _ { \mathrm { t r a i n } } \times n _ { \mathrm { t e s t } } } ] ^ { T } } \end{array}\tag{97}
$$

is the $n _ { \mathrm { t r a i n } } \times n _ { \mathrm { t r a i n } }$ block diagonal matrix of $\Sigma$ corresponding to the training data. By (2),

$$
\frac { 2 } { 1 - q _ { \mathrm { t r a i n } } ^ { - 1 } } - n _ { \mathrm { t r a i n } } = \frac { 2 } { 1 - q ^ { - 1 } } - n ,\tag{98}
$$

which implies that

$$
{ \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } } - n _ { \mathrm { t r a i n } } = { \frac { 1 } { q - 1 } } - n .\tag{99}
$$

(99) allows us to recover $q$ from the training procedure. Above formulas for the training data and parameters can trivially be applied to the validation and the testing data and parameters; thus, validation and test can carried out easily from the model build from the training data.

For q−correlated data, often times, the q−correlation structure is inferred or given prior to the model fitting; thus, we assume that the $q \cdot$ −correlation structure is given as Ψ and we estimate the volatility / dispersion / scale parameter $\sigma ^ { 2 } > 0$ such that

$$
\Sigma = \sigma ^ { 2 } \Psi .\tag{100}
$$

Trivially, $\Psi _ { \mathrm { t r a i n } }$ is the block diagonal matrix of Ψ corresponding to the training data and $\Sigma _ { \mathrm { t r a i n } } = \sigma ^ { 2 } \Psi _ { \mathrm { t r a i n } }$

We are now ready to formulate our likelihood loss function. To utilize qGaussian distribution to model the $q \cdot$ −correlated observations, we estimate the value of $q$ such that $q$ is allowed to vary, and the model will thus be more robust towards a wide class of distributions. Therefore, we choose to build the model using (20), since the dispersion matrix Λ (31) depends on q. We formulate our maximization of our log-likelihood function as the following from (96) and (20):

$$
\begin{array} { r l } & { \qquad \quad \underset { q _ { \mathrm { t r a i n } } \in ( 1 , 1 + \frac { 2 } { \pi _ { \mathrm { t r a i n } } ^ { 2 } } ) , \theta \in R ^ { p + 1 } , \sigma ^ { 2 } \in R _ { > 0 } } { \operatorname { l o r } } \log ( \frac { 1 } { \sigma ^ { 2 } \Psi _ { \mathrm { t r a i n } } } | ^ { 1 / 2 } } \\ & { \cdot \frac { \Gamma \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \right) } { \Gamma \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } - \frac { 1 } { 2 } \right) } \cdot \left( \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } \right) ^ { - \frac { n _ { \mathrm { t r a i n } } } { 2 } } } \\ & { \cdot \left( 1 + \left( \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } \right) ^ { - 1 } \cdot \left. y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta , \left( \sigma ^ { 2 } \Psi _ { \mathrm { t r a i n } } \right) ^ { - 1 } \left( y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta \right) \right. \right) ^ { \frac { 1 } { 1 - q _ { \mathrm { t r a i n } } } } ) . } \end{array}
$$

To address the high–dimensional data concerns, Oracle penalties are incorporated to carry out variable selection. To penalize the log-likelihood loss function to achieve variable selection, we formulate the following problem:

$$
\begin{array} { r l } & { \mathrm { c r o s s s } ( 1 ) , \ \frac { \mathrm { d } ^ { 2 } \mathrm { p e r f } \mathrm { d } t } { \mathrm { d } t ^ { 2 } } \mathrm { e } ^ { \mathrm { i } \phi \mathrm { i } t } = \mathrm { i } , \ \mathrm { c } _ { \mathrm { p } } ( 2 \mathrm { s } _ { \mathrm { p } } ) } \\ & { = \mathrm { i } \mathrm { c } _ { \mathrm { p } } ( \frac { 1 } { \mathrm { d } t } ) \frac { 1 } { \mathrm { d } t ^ { 2 } \mathrm { p e r f } \mathrm { d } t } \cdot \frac { 1 } { \Gamma } \left( \frac { 1 } { \Gamma \left( \frac { 1 } { \mathrm { d } \mathrm { d } \mathrm { d } t } - 1 \right) \mathrm { d } ^ { 2 } \mathrm { p e r f } \mathrm { d } t } \right) \cdot \left( \frac { 2 } { \mathrm { d } \mathrm { d } t { \mathrm { d } \mathrm { d } t } - 1 } - \mathrm { c } _ { \mathrm { p } , \mathrm { a n d } } \right) ^ { - \mathrm { v } _ { \mathrm { p } , \mathrm { a n d } } } } \\ &  \phantom { = } \Bigg ( 1 ) \left( \begin{array} { c } { 1 } \\  2 \left( 2 \mathrm { d } \mathrm { d } - 1 - \ \ ^ { 2 } \mathrm { n } \mathrm { c o s s } \right) ^ { \mathrm { i n } } \cdot 1 ^ { \mathrm { p } - 1 } \mathcal { A } ^ { 2 } \cdot \left( \frac { 1 } { \mathrm { d } \mathrm { d } \mathrm { d } t } \mathrm { d } ^ { 2 } \mathrm { p e r f } \mathrm { d } t \mathrm { d } t \mathrm { d } t \mathrm { d } t \mathrm { d } t \mathrm { d } ^ { 2 } \mathrm { p e r f } \mathrm { d } t \mathrm { d } t \mathrm { d } t \mathrm { d } t \mathrm { d } t \mathrm { d } ^ { 2 } \mathrm { p e r f } \mathrm { d } t \mathrm { d } t \mathrm { d } t \mathrm { d } t \mathrm { d } t \mathrm { d } ^ { 2 } \right) \right) \cdot \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } t ^ { 2 } \mathrm { p e r f } \mathrm { d } t } \Bigg [ - \mathrm { i } \mathrm { c } _ { \mathrm { p } } (  \end{array} \end{array}
$$

In the above formulated problem, w is the Oracle penalty function, and we are not to penalize the intercept term. The $2 n _ { \mathrm { t r a i n } }$ multiplier is to ensure that the penalization efect is consistent with the number of training observations. We choose to put the penalty term together with the quadratic term without the variance scale parameter $\sigma ^ { 2 }$ for two reasons: first, the optimization problem is more tractable under such problem formulation; second, we do not wish to let the value of $\sigma ^ { 2 }$ perturb the degree of penalization. Comparing to penalized the log-likelihood directly, we choose to penalize the quadratic component directly as it is more tractable. It was shown that doing so will preserve Oracle properties [40] of penalized estimators.

For the optimization procedure, we will proceed in a blockwise manner; $\mathrm { i . e . }$ , we will optimize $q _ { \mathrm { t r a i n } } , \theta , \sigma ^ { 2 }$ separately in each iteration. More details will be given in the following subsections.

## 5.2 Minimizing with respect to $q _ { \mathbf { t r a i n } }$ and $\sigma ^ { 2 }$

With all the other parameters fixed, the sub-problem to minimize with respect to $\sigma ^ { 2 }$ is

$$
\begin{array} { c } { { a r g m i n \displaystyle \frac { n } { 2 } \log \sigma ^ { 2 } + \displaystyle \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \log ( 1 + \left( \displaystyle \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } \right) ^ { - 1 } \cdot \sigma ^ { - 2 } } } \\ { { \cdot \left( \left. y _ { \mathrm { t r a i n } } - { \bf X } _ { \mathrm { t r a i n } } \theta , { \Psi } _ { \mathrm { t r a i n } } ^ { - 1 } \left( y _ { \mathrm { t r a i n } } - { \bf X } _ { \mathrm { t r a i n } } \theta \right) \right. + 2 n _ { \mathrm { t r a i n } } \displaystyle \sum _ { j = 2 } ^ { p + 1 } w \left( \theta _ { j } \right) \right) } } \end{array}\tag{102}
$$

which has a smooth objective function with respect to $\sigma ^ { 2 }$ . The first-order optimality condition

$$
\begin{array} { r l } & { \frac { n } { 2 } = \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } } \\ & { \qquad \cdot \frac { 2 } { \sigma ^ { 2 } + \left( \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } \right) ^ { - 1 } \cdot \left( \left. y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta , \Psi _ { \mathrm { t r a i n } } ^ { - 1 } \left( y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta \right) \right. + 2 n _ { \mathrm { t r a i n } } \sum _ { j = 2 } ^ { p + 1 } w \left( \theta _ { j } \right) \right) } } \\ & { \qquad \sigma ^ { 2 } + \left( \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } \right) ^ { - 1 } \cdot \left( \left. y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta , \Psi _ { \mathrm { t r a i n } } ^ { - 1 } \left( y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta \right) \right. + 2 n _ { \mathrm { t r a i n } } \sum _ { j = 2 } ^ { p + 1 } w \left( \theta _ { j } \right) \right) } \end{array}
$$

implies that the optimal value for the subproblem (102) takes minimizer

$$
\begin{array} { l } { \displaystyle \overline { { \sigma ^ { 2 } } } = \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } / \frac { n } { 2 } - 1 \right) \cdot \left( \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } \right) ^ { - 1 } } \\ { \displaystyle \qquad \cdot \left( \left. y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta , \Psi _ { \mathrm { t r a i n } } ^ { - 1 } \left( y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta \right) \right. + 2 n _ { \mathrm { t r a i n } } \sum _ { j = 2 } ^ { p + 1 } w \left( \theta _ { j } \right) \right) > 0 , } \end{array}\tag{103}
$$

which is feasible. The feasible set for $q _ { \mathrm { t r a i n } }$ is $\begin{array} { r } { \left( 1 , 1 + \frac { 2 } { n _ { \mathrm { t r a i n } } } \right) } \end{array}$ , in this view, when $n _ { \mathrm { t r a i n } }$ is large, the numerical stability will be an issue if minimization is carried out with respect to $q _ { \mathrm { t r a i n } }$ directly. Thus, we choose to minimize with respect to $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \in \left( \frac { n _ { \mathrm { t r a i n } } } { 2 } , \infty \right)$

First of all, we are to prove that such minimization is feasible.

Proof. Since the objective function (101) is continuous and smooth with respect to $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }$ we only need to analyze the derivative when $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \searrow 0$ and $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }  \infty$

$$
\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }  \infty \colon
$$

Stirling’s formula states that

$$
\operatorname* { l i m } _ { x \to \infty } { \frac { \Gamma \left( x \right) } { \sqrt { \frac { 2 \pi } { x } } \left( { \frac { x } { e } } \right) ^ { x } \left( 1 + O \left( x ^ { - 1 } \right) \right) } } = 1 .\tag{104}
$$

Thus,

$$
\operatorname* { l i m } _ { \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } } \frac { \Gamma \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \right) } { \infty } \cdot \left( \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } \right) ^ { - \frac { n _ { \mathrm { t r a i n } } } { 2 } } = 1\tag{105}
$$

then

$$
\operatorname* { l i m } _ { \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }  \infty } - \log { ( \frac { \Gamma ( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } ) } { \Gamma ( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } - \frac { n _ { \mathrm { t r a i n } } } { 2 } ) } \cdot ( \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } ) ^ { - \frac { n _ { \mathrm { t r a i n } } } { 2 } } ) } = 0 .\tag{106}
$$

We also have

$$
\begin{array} { r l r } {  { \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \log ( 1 + ( \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } ) ^ { - 1 } \cdot \sigma ^ { - 2 } } }  \\ & { } & { ~ \cdot (  y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta , \Psi _ { \mathrm { t r a i n } } ^ { - 1 } ( y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta )  + 2 n _ { \mathrm { t r a i n } } \sum _ { j = 2 } ^ { p + 1 } w ( \theta _ { j } ) ) } \\ & { } & { = O ( ( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } ) / \log ( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } ) ) , } \end{array}
$$

which implies that this term will goes to infinity as $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }  \infty$ . Thus, the objective function (101) goes to infinity as $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }  \infty$

$$
\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \searrow \frac { n _ { \mathrm { t r a i n } } } { 2 } \colon
$$

Since $\begin{array} { r } { \Gamma \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } - \frac { n _ { \mathrm { t r a i n } } } { 2 } \right) \to \infty \mathrm { ~ a s ~ } \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \searrow \frac { n _ { \mathrm { t r a i n } } } { 2 } . } \end{array}$

The penalized log-likelihood involving $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }$ can be simplified as

$$
\begin{array} { r l r } { \left. { - \log \frac { \Gamma \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \right) } { \Gamma \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } - \frac { n _ { \mathrm { t r a i n } } } { 2 } \right) } \cdot ( \left( \frac { 2 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } \right) } } \\ & { } & { + \left. y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta , \left( \sigma ^ { 2 } \Psi _ { \mathrm { t r a i n } } \right) ^ { - 1 } \left( y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta \right) \right. + 2 n _ { \mathrm { t r a i n } } \sum _ { j = 2 } ^ { p + 1 } w \left( \theta _ { j } \right) ) ^ { \frac { 1 } { 1 - q _ { \mathrm { t r a i n } } } } \right. \infty } \end{array}
$$

as $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \searrow \frac { n _ { \mathrm { t r a i n } } } { 2 }$ . Thus, the subproblem to minimize with respect to $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }$ is coercive on $\left( { \frac { n _ { \mathrm { t r a i n } } } { 2 } } , \infty \right)$ . Coercivity implies that any minimizing sequence $\left\{ \left( { \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } } \right) _ { j } \right\}$ must be contained within a bounded subset of $\left( { \frac { n _ { \mathrm { t r a i n } } } { 2 } } , \infty \right)$ . Thus, Bolzano–Weierstrass theorem implies the existence of a convergent subsequence. Let $\left\{ \left( { \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } } \right) _ { j _ { k } } \right\}$ be one such subsequence, and let $\overline { { \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \right) } }$ be its limit. Since the subproblem has a continuous objective function with respect to $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }$ , the objective function is lower–semicontinuous and the value of the objective function at $\overline { { \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \right) } }$ is less than or equal to the value of the objective function at $\left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \right) _ { j _ { k } }$ for all $k = 1 , 2 , \ldots , \infty$ . Thus, since $\left\{ \left( { \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } } \right) _ { j } \right\}$ is a minimizing sequence, the value of the objective function at $\left( { \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } } \right)$ is less than or equal to the infimum of the objective function on $\left( { \frac { n _ { \mathrm { t r a i n } } } { 2 } } , \infty \right)$ . Hence, since the entire minimizing sequence is contained in $\begin{array} { r } { \big ( \frac { n _ { \mathrm { t r a i n } } } { 2 } , \infty \big ) , \left( \frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } \right) \in \big ( \frac { n _ { \mathrm { t r a i n } } } { 2 } , \infty \big ) } \end{array}$ solves the subproblem of minimizing with respect to $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }$

Unlike $\sigma ^ { 2 }$ , the minimizer for $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }$ is not in closed form, and the evaluation of the derivative with respect to $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }$ can not be carried out eficiently. Thus, we apply Brent’s line-search method to optimize the $\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 }$ subproblem [7]. □

## 5.3 Minimizing with respect to θ

Minimizing with respect to θ involves a nonconvex smooth function and a convex nonsmooth function, which is termed a composite problem. In Section 4.3, we developed a proximal conjugate gradient algorithm for such composite optimization.

In this part, we will establish an important remark regarding the Oracle penalty. The subproblem we are to minimize with respect to θ is

$$
\begin{array} { r l r } {  { a r g m i n  \ y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta , \Psi _ { \mathrm { t r a i n } } ^ { - 1 } ( y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta )  + 2 n _ { \mathrm { t r a i n } } \sum _ { j = 2 } ^ { p + 1 } w ( \theta _ { j } ) } } \\ & { } & { \stackrel { a r g m i n } { \Leftrightarrow } \displaystyle { a r g m i n } \frac { 1 } { 2 n _ { \mathrm { t r a i n } } }  y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta , \Psi _ { \mathrm { t r a i n } } ^ { - 1 } ( y _ { \mathrm { t r a i n } } - \mathbf { X } _ { \mathrm { t r a i n } } \theta )  + \sum _ { j = 2 } ^ { p + 1 } w ( \theta _ { j } ) } \\ & { } & { \stackrel { a r g m i n } { \Leftrightarrow } \displaystyle { a r g m i n } \frac { 1 } { 2 n _ { \mathrm { t r a i n } } }  \theta , \mathbf { X } _ { \mathrm { t r a i n } } ^ { T } \Psi _ { \mathrm { t r a i n } } ^ { - 1 } \mathbf { X } _ { \mathrm { t r a i n } } \theta  - 2  y _ { \mathrm { t r a i n } } , \Psi _ { \mathrm { t r a i n } } ^ { - 1 } \mathbf { X } _ { \mathrm { t r a i n } } \theta  + \sum _ { j = 2 } ^ { p + 1 } w ( \theta _ { j } ) } \end{array}\tag{107}
$$

w can be chosen as oracle penalties such as $\mathrm { S C A D / M C P }$ penalties. And it has been shown that both SCAD / MCP penalties admit a diference-of-convex decomposition to a firstorder smooth concave term plus λ times $\ell _ { 1 }$ penalty. The quadratic loss function is clearly convex and smooth. This justifies our assumption for the objective function. To carry out the proximal Hager-Zhang conjugate gradient method proposed in Section 4.3, we need to calculate $\boldsymbol { L } _ { \boldsymbol { \nabla } g }$ , the L−smoothness constant for the smooth component. Previous work suggests $L _ { \nabla g } = \mathrm { m a x }$ nmax eigenvalue of $\begin{array} { r } { \frac { 1 } { n _ { \mathrm { t r a i n } } } \mathbf { X } _ { \mathrm { t r a i n } } ^ { T } \boldsymbol { \Psi } _ { \mathrm { t r a i n } } ^ { - 1 } \mathbf { X } _ { \mathrm { t r a i n } } , c _ { \mathrm { p e n a l t y } } \bigg \} } \end{array}$ , where c<sub>penalty</sub> is the L−smoothness constant for the smooth component of the penalty, which will be $\scriptstyle { \frac { 1 } { a - 1 } }$ for SCAD and $\frac { 1 } { \gamma }$ for MCP [63].

Remark 13. For high dimensional data, often times, the number of covariates exceeds the number of observations; i.e, null $( \mathbf { X } _ { \mathrm { t r a i n } } ) \neq { \emptyset }$ . Both $\mathrm { S C A D / M C P }$ penalties take constant values in $B _ { \infty } \left( 0 , c \right) ^ { 1 }$ ; where $c = a \lambda$ for $\mathrm { S C A D }$ and $c = \gamma \lambda$ for MCP. Given any stationary point $\bar { \theta }$ in the nonempty solution set defined by ${ \bf X } _ { \mathrm { t r a i n } } ^ { T } \Psi _ { \mathrm { t r a i n } } ^ { - 1 } { \bf X } _ { \mathrm { t r a i n } } - { \bf X } _ { \mathrm { t r a i n } } ^ { T } y _ { \mathrm { t r a i n } } = 0$ . For the set $\bar { \theta } + \mathrm { n u l l } \left( \mathbf { X } _ { \mathrm { t r a i n } } \right) \backslash B _ { \infty } \left( 0 , c \right)$ , each point in the relative interior (which is nonempty) of this set is a Clarke stationary point, which implies that any algorithm with a starting point in this set will converge in 0 steps. This might pose an issue for signal recovery, since null $\left( \mathbf { X } _ { \mathrm { t r a i n } } \right)$ is a vector subspace and some points can be very far from the origin.

Remark 14. In view of the subproblem with respect to $\theta ,$ it is trivial that the minimizer for (107) does not depend on the other parameters, which are $q$ and $\sigma ^ { 2 }$ . Since the qGaussian distribution is a generalization for all bell curve distributions, the estimation of the central trend using the maximum likelihood principle for bell curve distributions is equivalent to minimize a quadratic function, which has a breakdown point of 0.

Taking into account the optimization subproblem with respect to $\theta ,$ it is evident that the solution to (107) remains unafected by the other parameters, namely $q$ and $\sigma ^ { 2 }$ . Given that the qGaussian distribution extends the framework of bell curve distributions, the problem (107) implies that estimating the central trend through the maximum likelihood principle for all bell curve distributions is equivalent to minimizing a quadratic function of the central trend, therefore characterized by a breakdown point of 0.

## 5.4 Prediction for y<sub>test</sub>

To show how prediction can be made, we will show the methods to predict $y _ { \mathrm { t e s t } }$ using the trained model in this subsection. The same method applies for validation when predictions on $y _ { \mathrm { v a l } }$ are needed or to predict any new data based on the trained model. Since the data are mutually qGaussian, (99) implies that

$$
\frac { 1 } { q _ { \mathrm { t r a i n } } - 1 } - n _ { \mathrm { t r a i n } } = \frac { 1 } { q _ { \mathrm { v a l } } - 1 } - n _ { \mathrm { v a l } } = \frac { 1 } { q _ { \mathrm { t e s t } } - 1 } - n _ { \mathrm { t e s t } } = \frac { 1 } { q - 1 } - n ,\tag{108}
$$

which will be used to recover the value of the shape parameter $q _ { \mathrm { v a l } } , q _ { \mathrm { t e s t } }$ . Note that when $n _ { \mathrm { n e w } }$ data points are introduced, the total number of observations n changes from $n _ { \mathrm { t r a i n } } +$ $n _ { \mathrm { v a l } } + n _ { \mathrm { t e s t } }$ to $n _ { \mathrm { t r a i n } } + n _ { \mathrm { v a l } } + n _ { \mathrm { t e s t } } + n _ { \mathrm { n e w } }$ , thus, the value of $q .$ will change for the entire dataset. However, $q _ { \mathrm { t r a i n } }$ stays the same; thus, we suggest inferring the shape parameter for each data set based on $q _ { \mathrm { t r a i n } }$ directly using the equation above. With q<sub>·</sub> calculated, it is straightforward to estimate the q−variance-covariance matrix $\mathbb { E } _ { q } \left\lceil \left( y _ { \cdot } - \mathbf { X } \theta \right) \left( y _ { \cdot } - \mathbf { X } \theta \right) ^ { T } \right\rceil$ based on (33); or, ifexisting, the variance-covariance matrix E $\left[ { { \left( y { \mathrm { . - } } { \dot { \mathbf { X } } } \theta \right) } \left( y { \mathrm { . - } } { \mathbf { X } } \theta \right) } ^ { T } \right]$ based on (34).

## 6 Conclusion and Discussion

This paper explores the field of statistical sparse learning, focusing on modeling correlated data through the lens of maximizing Tsallis entropy. It addresses the limitations inherent in the conventional Gaussian distribution, notably its lack of robustness towards outliers and underlying shape assumptions, by advocating for the qGaussian distribution. This distribution, derived from Tsallis entropy maximization, represents a novel approach to handling correlated data and heterogeneity — elements frequently encountered in biostatistical contexts involving genetic and longitudinal studies.

This paper encompasses a re-derived probability density function for the multivariate qGaussian distribution based on Tsallis entropy maximization. Statistical modeling based on the derived density paves the way for the analysis of correlated data and heterogeneity and enables variable selection. Furthermore, we have developed an innovative framework capable of converting any numerical method, originally designed to identify equilibria in flows, into a tool for tackling composite optimization problems that are prevalent in statistical sparse learning. By applying this framework to the Hager-Zhang conjugate gradient algorithm, we have crafted an efective and stable algorithm tailored to the challenges of sparse statistical learning. Given the abundance of methods for numerically identifying equilibria for globally Lipschitz flows, our approach significantly broadens the arsenal of techniques available to address sparse statistical learning optimization challenges.

In conclusion, our research positions the qGaussian distribution, underpinned by maximizing Tsallis entropy, as a robust and adaptable alternative to Gaussian-based methodologies in statistical sparse learning on correlated data. This breakthrough not only confronts the traditional limitation of Gaussian assumptions, but also paves the way for expanded investigation into Tsallis entropy-maximizing distributions, particularly within the domain of biostatistics and allied disciplines.

Future directions for research include the exploration of the log-linear model through the lens of Tsallis entropy maximization, akin to approaches previously based on Shannon’s entropy. Moreover, the study of the phenomenon called volatility smirk in financial return data may benefit from employing the log-qGaussian distribution —- a transformation of the qGaussian distribution, which can provide deeper insights into the nuances of financial markets. Additionally, in the field of statistical computing research, our framework that transforms numerical methods for identifying flow equilibria into algorithms for solving composite optimization problems opens numerous avenues for future research, especially in a sparse learning context.

## 7 Acknowledgments

This work was supported by the ISM Scholarship for Outstanding PhD Candidates awarded to K. Yang, the NSERC Discovery Grant to C. Greenwood (Grant Number: RGPIN-2019- 04482), the NSERC Discovery Grant to M. Asgharian (Grant Number: RGPIN-2024- 05640), and the CANSSI Collaborative Research Team Grant to C. Greenwood and G. Cohen Freue.

## References

[1] Ravi P. Agarwal, Maria Meehan, and Donal O’Regan. Fixed point theory and applications. Digitally printed version, paperpack re-issue. Cambridge tracts in mathematics 141. Cambridge [u.a.]: Cambridge Univ. Press, 2009. 170 pp. isbn: 9780521802505.

[2] Kendall E. Atkinson. An Introduction to Numerical Analysis. 2. ed., [14. print]. Bibliogr. S. 665. New York [u.a.]: Wiley, 1989. 693 pp. isbn: 0471624896.

[3] Heinz H. Bauschke. Convex Analysis and Monotone Operator Theory in Hilbert Spaces. Ed. by Patrick L. Combettes. SpringerLink. New York, NY: Springer New York, 2011. 46830 pp. isbn: 9781441994677.

[4] Amir Beck and Marc Teboulle. “A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems”. English. In: SIAM Journal on Imaging Sciences 2.1 (2009). Copyright - Copyright] © 2009 Society for Industrial and Applied Mathematics; Last updated - 2012-07-02, pp. 183–20. url: https://proxy.library.mcgill. ca/login?url=https://search.proquest.com/docview/925336974?accountid= 12339.

[5] Lisa Borland. “A theory of non-Gaussian option pricing”. In: Quantitative Finance 2.6 (Dec. 2002), pp. 415–431. doi: 10.1080/14697688.2002.0000009.

[6] Lisa Borland. “Option Pricing Formulas Based on a Non-Gaussian Stock Price Model”. In: Physical Review Letters 89.9 (Aug. 2002), p. 098701. doi: 10.1103/physrevlett. 89.098701.

[7] R. P. Brent. “An Algorithm with Guaranteed Convergence for Finding a Zero of a Function”. In: The Computer Journal 14.4 (Apr. 1971), pp. 422–425. issn: 1460-2067. doi: 10.1093/comjnl/14.4.422.

[8] Cuiling Chen et al. “Global Convergence of an Extended Descent Algorithm without Line Search for Unconstrained Optimization”. In: Journal of Applied Mathematics and Physics 06.01 (2018), pp. 130–137. issn: 2327-4379. doi: 10.4236/jamp.2018. 61013.

[9] Francis Clarke. “Lyapunov Functions and Feedback in Nonlinear Control”. In: Lecture Notes in Control and Information Sciences. Springer Berlin Heidelberg, May 2004, pp. 267–282. isbn: 9783540399834. doi: 10.1007/978-3-540-39983-4\_17.

[10] Francis H. Clarke. Optimization and nonsmooth analysis. Reprint. Originally published: New York : Wiley, 1983. Classics in applied mathematics 5. Reprint. Originally published: New York : Wiley, 1983. Philadelphia, Pa: Society for Industrial and Applied Mathematics (SIAM, 3600 Market Street, Floor 6, Philadelphia, PA 19104), 1990. 1308 pp. isbn: 9781611971309.

[11] Jose Costa, Alfred Hero, and Christophe Vignat. “On Solutions to Multivariate Maximum α-Entropy Problems”. In: Lecture Notes in Computer Science. Springer Berlin Heidelberg, 2003, pp. 211–226. doi: 10.1007/978-3-540-45063-4\_14.

[12] T. M. Cover and Joy A. Thomas. Elements of Information Theory. John Wiley & Sons, Inc., 2006. isbn: 9780471241959.

[13] Y. H. Dai and Y. Yuan. “A Nonlinear Conjugate Gradient Method with a Strong Global Convergence Property”. In: SIAM Journal on Optimization 10.1 (Jan. 1999), pp. 177–182. issn: 1095-7189. doi: 10.1137/s1052623497318992.

[14] Claire Dandine-Roulland and Herv´e Perdry. “The Use of the Linear Mixed Model in Human Genetics”. In: Human Heredity 80.4 (2015), pp. 196–206. issn: 1423-0062. doi: 10.1159/000447634.

[15] Dario Domingo, Alberto d’Onofrio, and Franco Flandoli. “Boundedness vs unboundedness of a noise linked to Tsallis q-statistics: The role of the overdamped approximation”. In: Journal of Mathematical Physics 58.3 (Mar. 2017). issn: 1089-7658. doi: 10.1063/1.4977081.

[16] Dexiang Feng, Min Sun, and Xueyong Wang. “A Family of Conjugate Gradient Methods for Large-Scale Nonlinear Equations”. In: Journal of Inequalities and Applications 2017.1 (Sept. 2017). issn: 1029-242X. doi: 10.1186/s13660-017-1510-0.

[17] R. Fletcher. “Function minimization by conjugate gradients”. In: The Computer Journal 7.2 (Feb. 1964), pp. 149–154. issn: 1460-2067. doi: 10.1093/comjnl/7. 2.149.

[18] Tanya P. Garcia and Karen Marder. “Statistical Approaches to Longitudinal Data Analysis in Neurodegenerative Diseases: Huntington’s Disease as a Model”. In: Current Neurology and Neuroscience Reports 17.2 (Feb. 2017). issn: 1534-6293. doi: 10.1007/s11910-017-0723-4.

[19] Saeed Ghadimi and Guanghui Lan. “Accelerated Gradient Methods for Nonconvex Nonlinear and Stochastic Programming”. In: Mathematical Programming 156.1-2 (Feb. 2015), pp. 59–99. doi: 10.1007/s10107-015-0871-8.

[20] Saeed Ghadimi and Guanghui Lan. “Stochastic First- and Zeroth-Order Methods for Nonconvex Stochastic Programming”. In: SIAM Journal on Optimization 23.4 (Jan. 2013), pp. 2341–2368. doi: 10.1137/120880811.

[21] Jean Charles Gilbert and Jorge Nocedal. “Global Convergence Properties of Conjugate Gradient Methods for Optimization”. In: SIAM Journal on Optimization 2.1 (Feb. 1992), pp. 21–42. issn: 1095-7189. doi: 10.1137/0802003.

[22] William Hager and Hongchao Zhang. “A survey of nonlinear conjugate gradient method”. In: 2 (Jan. 2006).

[23] William W. Hager and Hongchao Zhang. “A New Conjugate Gradient Method with Guaranteed Descent and an Eficient Line Search”. In: SIAM Journal on Optimization 16.1 (Jan. 2005), pp. 170–192. issn: 1095-7189. doi: 10.1137/030601880.

[24] Uwe Helmke. Optimization and Dynamical Systems. London: Springer London, 1994. isbn: 9781447134671. doi: 10.1007/978-1-4471-3467-1.

[25] M.R. Hestenes and E. Stiefel. “Methods of conjugate gradients for solving linear systems”. In: Journal of Research of the National Bureau of Standards 49.6 (Dec. 1952), p. 409. issn: 0091-0635. doi: 10.6028/jres.049.044.

[26] Tim Hoheisel, Maxime Laborde, and Adam M. Oberman. “A Regularization Interpretation of the Proximal Point Method for Weakly Convex Functions”. In: Journal of Dynamics & Games (2020). url: https://api.semanticscholar.org/CorpusID: 202607166.

[27] John H. Hubbard and Beverly H. West. Diferential Equations: A Dynamical Systems Approach. Springer New York, 1995. isbn: 9781461241928. doi: 10.1007/978-1- 4612-4192-8.

[28] Dominic Kafka and Daniel Wilke. “Gradient-only line searches: An Alternative to Probabilistic Line Searches”. In: (Mar. 2019). doi: 10.48550/ARXIV.1903.09383. arXiv: 1903.09383 [stat.ML].

[29] Christian Kanzow and Theresa Lechner. “Globalized inexact proximal Newton-type methods for nonconvex composite functions”. In: Computational Optimization and Applications 78.2 (Nov. 2020), pp. 377–410. issn: 1573-2894. doi: 10.1007/s10589- 020-00243-6.

[30] S. Karamardian. “Complementarity problems over cones with monotone and pseudomonotone maps”. In: Journal of Optimization Theory and Applications 18.4 (Apr. 1976), pp. 445–454. issn: 1573-2878. doi: 10.1007/bf00932654.

[31] Jason D. Lee, Yuekai Sun, and Michael A. Saunders. “Proximal Newton-Type Methods for Minimizing Composite Functions”. In: SIAM Journal on Optimization 24.3 (Jan. 2014), pp. 1420–1443. doi: 10.1137/130921428.

[32] Xingguo Li et al. On Fast Convergence of Proximal Algorithms for SQRT-Lasso Optimization: Don’t Worry About Its Nonsmooth Loss Function. 2016. doi: 10.48550/ ARXIV.1605.07950.

[33] Christian Lubich. Geometric Numerical Integration. Structure-Preserving Algorithms for Ordinary Diferential Equations. Ed. by Gerhard Wanner and Ernst Hairer. 2<sup>nd</sup> ed. Springer Series in Computational Mathematics Ser. v.31. Description based on publisher supplied metadata and other sources. Berlin, Heidelberg: Springer Berlin / Heidelberg, 2006. 1660 pp. isbn: 9783540306665.

[34] M. Kato M. Tsukada H. Suyari. “On the Probability Distribution Maximizing Generalized Entropies”. In: Proceedings of 2005 Symposium on Applied Functional Analysis - Information Sciences and Related Fields. 2005, pp. 99–111.

[35] Boris S. Morduchoviˇc. Variational analysis and applications. Softcover re-print of the Hardcover 1<sup>st</sup> edition 2018. Springer monographs in mathematics. Literaturverzeichnis: Seiten 533-578. Cham, Switzerland: Springer, 2018. 622 pp. isbn: 9783030065133.

[36] B. Sh Mordukhovich. Variational Analysis and Generalized Diferentiation I: Basic theory. Variational analysis and generalized diferentiation 1. Includes bibliographical references and indexes. Berlin ; Springer, 2006. 579 pp. isbn: 9783540312475.

[37] B. Sh Mordukhovich. Variational Analysis and Generalized Diferentiation II: Applications. Variational analysis and diferentiation 2. Includes bibliographical references and index. New York: Springer, 2006. 610 pp. isbn: 9783540312468.

[38] Yurii Nesterov. Introductory Lectures on Convex Optimization. Springer US, 2004. doi: 10.1007/978-1-4419-8853-9.

[39] Arnold Neumaier, Morteza Kimiaei, and Behzad Azmi. “Globally linearly convergent nonlinear conjugate gradients without Wolfe line search”. In: Numerical Algorithms (Feb. 2024). issn: 1572-9265. doi: 10.1007/s11075-024-01764-5.

[40] Mila Nikolova. “Local Strong Homogeneity of a Regularized Estimator”. In: SIAM Journal on Applied Mathematics 61.2 (Jan. 2000), pp. 633–658. doi: 10 . 1137 / s0036139997327794.

[41] J. Nocedal and S. Wright. Numerical Optimization. Ed. by Stephen J. Wright. Second edition. Springer Series in Operations Research and Financial Engineering. New York, NY: Springer New York, 2000. 1664 pp. isbn: 9780387987934. url: https://books. google.ca/books?id=epc5fX0lqRIC.

[42] Ignacio Pe˜na, Gonzalo Rubio, and Gregorio Serna. “Why do we smile? On the determinants of the implied volatility function”. In: Journal of Banking & Finance 23.8 (Aug. 1999), pp. 1151–1179. issn: 0378-4266. doi: 10.1016/s0378-4266(98)00134- 4.

[43] Eric Polak and G Ribiere. “Note sur la convergence de m´ethodes de directions conjugu´ees”. fre. In: ESAIM: Mathematical Modelling and Numerical Analysis - Mod´elisation Math´ematique et Analyse Num´erique 3.R1 (1969), pp. 35–43. url: http://eudml.org/doc/193115.

[44] Alfio Quarteroni, Riccardo Sacco, and Fausto Saleri. Numerical Mathematics. Springer New York, 2007. isbn: 9780387227504. doi: 10.1007/b98885.

[45] Mohamed Kamel Riahi and Issam Al Qattan. “Linearly convergent nonlinear conjugate gradient methods for a parameter identification problems”. In: (June 2018). doi: 10.48550/ARXIV.1806.10197. arXiv: 1806.10197 [math.NA].

[46] Ralph Tyrrell Rockafellar and Roger J.-B. Wets. Variational analysis. Corr. 3. printing. [Softcover version of original hardcover edition 1998]. Die @Grundlehren der mathematischen Wissenschaften in Einzeldarstellungen 317. Heidelberg: Springer, 2010. 734 pp. isbn: 3642083048.

[47] I. M. Ross. “An Optimal Control Theory for Accelerated Optimization”. In: (Feb. 2019). doi: 10.48550/ARXIV.1902.09004. arXiv: 1902.09004 [math.OC].

[48] Daniel E. Runcie and Lorin Crawford. “Fast and flexible linear mixed models for genome-wide genetics”. In: PLOS Genetics 15.2 (Feb. 2019). Ed. by Michael P. Epstein, e1007978. issn: 1553-7404. doi: 10.1371/journal.pgen.1007978.

[49] Yousef Saad. Iterative Methods for Sparse Linear Systems. Society for Industrial and Applied Mathematics, Jan. 2003. isbn: 9780898718003. doi: 10.1137/1.9780898718003.

[50] C. E. Shannon. “A Mathematical Theory of Communication”. In: Bell System Technical Journal 27.3 (July 1948), pp. 379–423. doi: 10.1002/j.1538- 7305.1948. tb01338.x.

[51] Zhen-Jun Shi and Jie Shen. “Convergence of descent method without line search”. In: Applied Mathematics and Computation 167.1 (Aug. 2005), pp. 94–107. issn: 0096- 3003. doi: 10.1016/j.amc.2004.06.097.

[52] J A Snyman. “UNCONSTRAINED MINIMIZATION BY COMBINING THE DY-NAMIC AND CONJUGATE GRADIENT METHODS”. In: Quaestiones Mathematicae 8.1 (Jan. 1985), pp. 33–42. issn: 1727-933X. doi: 10.1080/16073606.1985. 9631898.

[53] J. A. Snyman. “A Gradient-Only Line Search Method for the Conjugate Gradient Method Applied to Constrained Optimization Problems with Severe Noise in the Objective Function”. In: International Journal for Numerical Methods in Engineering 62.1 (2004), pp. 72–82. issn: 1097-0207. doi: 10.1002/nme.1189.

[54] Eduardo D. Sontag. Mathematical Control Theory. Deterministic Finite Dimensional Systems. Second Edition. Springer eBook Collection. New York, NY: Springer New York, 1998. 531 pp. isbn: 9781461205777. doi: 10.1007/978-1-4612-0577-7.

[55] Jie Sun and Jiapu Zhang. “Global Convergence of Conjugate Gradient Methods without Line Search”. In: Annals of Operations Research 103.1/4 (2001), pp. 161– 173. issn: 0254-5330. doi: 10.1023/a:1012903105391.

[56] Constantino Tsallis. “Possible generalization of Boltzmann-Gibbs statistics”. In: Journal of Statistical Physics 52.1-2 (July 1988), pp. 479–487. doi: 10.1007/bf01016429.

[57] C Vignat, A.O Hero III, and J.A Costa. “About Closedness by Convolution of the Tsallis Maximizers”. In: Physica A: Statistical Mechanics and its Applications 340.1-3 (Sept. 2004), pp. 147–152. doi: 10.1016/j.physa.2004.04.001.

[58] C. Vignat and A. Plastino. “Scale invariance and related properties of q-Gaussian systems”. In: Physics Letters A 365.5-6 (June 2007), pp. 370–375. doi: 10.1016/j. physleta.2007.02.003.

[59] C. Vignat and A. Plastino. “The p-sphere and the geometric substratum of powerlaw probability distributions”. In: Physics Letters A 343.6 (Aug. 2005), pp. 411–416. doi: 10.1016/j.physleta.2005.05.027.

[60] C. Vignat and A. Plastino. “Why is the detection of q−Gaussian behavior such a common occurrence?” In: Physica A: Statistical Mechanics and its Applications 388.5 (Mar. 2009), pp. 601–608. issn: 0378-4371. doi: 10.1016/j.physa.2008.11.001.

[61] Cheng-jing Wang. “Some remarks on conjugate gradient methods without line search”. In: Applied Mathematics and Computation 181.1 (Oct. 2006), pp. 370–379. issn: 0096- 3003. doi: 10.1016/j.amc.2006.01.040.

[62] Qing-jun Wu. “A Nonlinear Conjugate Gradient Method without Line Search and Its Global Convergence”. In: 2011 International Conference on Computational and Information Sciences. IEEE, Oct. 2011. doi: 10.1109/iccis.2011.45.

[63] Kai Yang, Masoud Asgharian, and Sahir Bhatnagar. “Accelerated Gradient Methods for Sparse Statistical Learning with Nonconvex Penalties”. In: Statistics and Computing 34.1 (Jan. 2024). issn: 1573-1375. doi: 10.1007/s11222-023-10371-8.

[64] Yongchao Yu and Jigen Peng. “The Moreau envelope based eficient first-order methods for sparse recovery”. In: Journal of Computational and Applied Mathematics 322 (Oct. 2017), pp. 109–128. issn: 0377-0427. doi: 10.1016/j.cam.2017.03.014.

[65] Guangming Zhou. “A Descent Algorithm without Line Search for Unconstrained Optimization”. In: Applied Mathematics and Computation 215.7 (Dec. 2009), pp. 2528– 2533. issn: 0096-3003. doi: 10.1016/j.amc.2009.08.058.
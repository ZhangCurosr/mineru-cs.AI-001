# Improving TensorSketch Using Complex Random Variables

Amit Sharma<sup>\*1</sup> Mohammad Azhar Khan<sup>\*1</sup>

Rameshwar Pratap<sup>1</sup>

Keegan Kang<sup>2</sup>

<sup>1</sup>Department of Computer Science and Engineering, IIT Hyderabad, India <sup>2</sup>Department of Mathematics and Statistics, Bucknell University, USA

## Abstract

TensorSketch by Pham and Pagh [2013], Kar and Karnick [2012] provides efficient sketching algorithms for high-dimensional polynomial kernels $\bar { \mathbf { x } } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { p } }$ . Kar and Karnick [2012] uses dense Johnson-Lindenstrauss (JL)-type projections with computational cost O(pDd) , where D denotes the sketch dimension, whereas Pham and Pagh [2013] extends the sparse CountSketch [Charikar et al., 2004] algorithm, yielding a faster algorithm for high-dimensional sparse inputs with running time $O \left( p ( \mathrm { n n z } \left( \mathbf { x } \right) + D \log D ) \right)$ . However, the variance of both estimators grows exponentially with the polynomial degree $p ,$ scaling as $3 ^ { p } / D$ . Recent work by Wacker et al. [2023] showed that using complex-valued distribution reduces this dependence to 2<sup>p</sup>/D for the approach of Kar and Karnick [2012]. However, their method relies on dense JLtype projections with computational cost $O ( p D d )$ and does not extend to the algorithm of Pham and Pagh [2013].

In this work, we introduce a simple variant of TensorSketch [Pham and Pagh, 2013] that achieves the same variance bound as Wacker et al. [2023], while retaining its advantage of the inputsparsity running time. We validate our results with supporting experiments on synthetic and realworld datasets.

## 1 INTRODUCTION

Polynomial kernels (Definition 3) are widely used in machine learning to model higher-order, non-linear interactions among input features. For vectors $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d } ,$ , a degree-p polynomial kernel is defined as $k ( \mathbf { x } , \mathbf { y } ) = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p }$ . This kernel is equivalent to mapping $\mathbf { x } \in \mathbb { R } ^ { d }$ to its p-fold Kronecker product $\mathbf { x } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { p } }$ , which is $\langle { \bf x } ^ { \otimes p } , { \bf y } ^ { \otimes p } \rangle \equiv \langle { \bf x } , { \bf y } \rangle ^ { p } .$

As the dimension $d ^ { p }$ grows exponentially with $p ,$ directly computing these inner products is computationally infeasible. A dense Johnson-Lindenstrauss (JL) transform [Johnson and Lindenstrauss, 1984] can reduce their dimensionality while approximately preserving their pairwise inner product, however applying it requires time proportional to $d ^ { p }$ , which is exponential in $p .$

To address this inefficiency, prior works such as Kar and Karnick [2012] and Pham and Pagh [2013] proposed randomized sketching techniques to approximate polynomial kernels efficiently. Kar and Karnick [2012] introduced random feature maps based on JL-type projections. They define a randomized linear map $\mathbf { S } : \mathbb { R } ^ { d ^ { p } } $ $\mathbb { R } ^ { D }$ by $\mathbf { S } ( \mathbf { x } ^ { \otimes p } ) : = ( \mathbf { W } _ { 1 } \mathbf { x } \odot \dots \odot \mathbf { W } _ { p } \mathbf { x } ) / \sqrt { D }$ , where each $\mathbf { W } _ { i } ~ \in ~ \mathbb { R } ^ { D \times d }$ is a random projection matrix $( \mathrm { e . g . }$ Gaussian or Rademacher with $i . i . d .$ entries), and $\odot$ denotes the element-wise (Hadamard) product. Such that $\widehat { k } ( \mathbf { x } , \mathbf { y } ) : = \mathbf { \nabla } \langle \mathbf { S } ( \mathbf { x } ^ { \otimes p } ) , \mathbf { S } ( \mathbf { y } ^ { \otimes p } ) \rangle$ . This estimator can be termed as a JL-type variant of TensorSketch, and its computational cost scales with $O ( p D d )$ . Pham and Pagh [2013] propose TensorSketch, which combines CountSketch [Charikar et al., 2004] with Fast Fourier Transform (FFT)-based convolution to compute the sketch implicitly. This avoids forming the full vector and runs in input-sparsity time $O \left( p ( \mathrm { n n z } \left( \mathbf { x } \right) + D \log D ) \right)$ , making it preferred over Kar and Karnick [2012] for high-dimensional sparse data. The variance of both algorithms Kar and Karnick [2012], Pham and Pagh [2013, 2025] grows as $3 ^ { p } / D$

Recent progress by Wacker et al. [2024] introduced a complex-valued variant of JL-type TensorSketch. It improves the variance dependence of the sketch on the polynomial degree, reducing it from $3 ^ { p }$ to $2 ^ { p }$ . This improvement is achieved by incorporating complex-valued randomness into the sketch. However, the resulting sketch vectors are complex-valued and cannot be directly compared with their real-valued counterparts. To address this, Wacker et al. [2023] proposed the Complex-to-Real (CtR) construction. It first computes a complex random feature map of the embedding dimension half of the size of its real counterpart and then forms a real embedding by concatenating its real and imaginary parts. This preserves inner products and achieves improved variance bounds while yielding a real-valued sketch. However, as a JL-type method, it still requires dense multiplications with cost $O ( p D d )$ , which can be inefficient for high-dimensional sparse data.

In this work, we address the above limitation by developing the Complex-to-Real (CtR) variant of TensorSketch. The proposed construction obtain variance improvements comparable to the complex JL-type estimator of Wacker et al. [2023], while retaining the input-sparsity running time guarantees of TensorSketch [Pham and Pagh, 2013].

Contributions. We propose a Complex-to-Real variant of TensorSketch for degree-p polynomial kernels. It combines a random function whose values are drawn independently and uniformly from the fourth roots of unity with FFT-based tensor sketching while producing real-valued embeddings. We prove that the resulting estimator is unbiased for $\langle \mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \rangle$ ⟩. We also derive an upper bound on variance with $2 ^ { p } .$ -type dependence on the degree, improving over the classical TensorSketch [Pham and Pagh, 2013] variance scaling, while retaining the sketching time $O ( p ( \mathrm { n n z } ( \mathbf { x } ) + D \log D ) )$ ). The algorithm is defined in Definition 8, and its theoretical guarantees are stated in Theorem 2.

We note that the use of complex-valued random variables in sketching algorithms has been explored in prior work. For example, Wacker et al. [2023, 2024] employ complex random variables to construct sketches for polynomial kernels, while Meyer and Avron [2026] use them for trace estimation of implicit matrices. The key idea in Wacker et al. [2023, 2024] is to sketch $\mathbf { x } ^ { \otimes p }$ using element-wise products of independent sketches of the factors $\big ( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { p } \big )$ . Owing to this independence structure, Khintchine’s inequality (Definition 4) can be applied, and a simple induction on p yields a tighter upper bound in the complex setting than in the real-valued case.

In contrast, TensorSketch is built upon the hash-based CountSketch framework, where the sketch components are not independent and no direct analogue of Khintchine’s inequality is available. As a result, the techniques from prior work do not extend to our setting. Moreover, our analysis of CountSketch with complex-valued random variables (Theorem 5, Appendix) shows that, by itself, the complex construction of CountSketch provides no variance reduction over its real-valued version. Therefore, our result demonstrating improved variance for TensorSketch with complex random variables is non-trivial and stems from the vanishing of certain cross terms in the variance analysis, which does not occur in the real-valued setting. Implications of our results. TensorSketch [Pham and

Pagh, 2013] enables efficient training of linear SVMs [Sun et al., 2018, Li et al., 2019], supports scalable deep learning and the theoretical analysis of over-parameterized neural networks [Yehudai and Shamir, 2019, Zandieh et al., 2021], and has been successfully applied to compact bilinear pooling for fine-grained visual recognition [Gao et al., 2016] as well as multimodal fusion architectures [Fukui et al., 2016]. Polynomial kernels themselves are widely adopted in applications including natural language processing [Goldberg and Elhadad, 2008], recommender systems [Rendle, 2010], and genomics [Aschard, 2016]. Our proposed CtR TensorSketch achieves more accurate estimates than TensorSketch [Pham and Pagh, 2013] while maintaining the same asymptotic time complexity, making it a promising alternative for the above applications.

Organization of the paper. The remainder of this paper is organized as follows. Section 2 reviews prior work on polynomial kernel approximation and randomized sketching. Section 3 presents necessary background, including CountSketch, TensorSketch, polynomial kernels, and the CtR framework. Section 4, proposes our algorithm and states theoretical guarantees on its accuracy and efficiency. Section 5 reports empirical comparisons of real and JL-type methods with our approach. Section 6 concludes and outlines some future research directions.

## 2 RELATED WORK

Polynomial kernels can be expressed via tensor feature maps, where the representation is the tensor product $\bigotimes _ { i = 1 } ^ { p } \mathbf { x } _ { i }$ of vectors $\mathbf { x } _ { 1 } ~ \in ~ \mathbb { R } ^ { d _ { 1 } } , \ldots , \mathbf { x } _ { p } ~ \in ~ \mathbb { R } ^ { d _ { p } }$ to capture higher–order interactions. Explicit construction requires ${ \bar { O } } ( \prod _ { i = 1 } ^ { p } d _ { i } )$ memory and is infeasible even for moderate p or dimensions d. To avoid this blowup, prior work proposes sketching methods that compute $\mathbf { S } ( \bigotimes _ { i = 1 } ^ { p } \mathbf { x } _ { i } )$ without forming the full tensor, including JL-type product embeddings [Kar and Karnick, 2012] and hashing-based TensorSketch constructions built on CountSketch [Pham and Pagh, 2013, 2025]. For x, $\textbf { y } \in \ \mathbb { R } ^ { d }$ , both TensorSketch estimators have variance $\begin{array} { r l } { \leq } & { { } \frac { 3 ^ { p } - 1 } { D } \Big ( \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } \Big ) } \end{array}$ , which yields 3<sup>p</sup>-type variance growth. Both JL-type and hashing-based estimators scale as $\Theta ( 3 ^ { p } / D )$ for degree-p features, but TensorSketch [Pham and Pagh, 2013, 2025] is more efficient due its input-sparsity time complexity.

Recent work shows that complex-valued distributions reduce the variance of randomized feature maps. Complex JL-type constructions for polynomial kernels achieve lower variance than their real-valued counterparts [Wacker et al., 2024]. In particular, replacing real Rademacher or Gaussian variables with complex-valued distributions improves the variance dependence from 3<sup>p</sup>-type to 2<sup>p</sup>-type. Further work by Wacker et al. [2023] introduced the Complex-to-Real (CtR) construction, preserving the variance improvement of complex-valued sketches while yielding real-valued embeddings. These embeddings enables direct comparison with real sketches. However, these approaches rely on dense JLstyle projections and therefore incur higher computational cost especially for high-dimensional sparse data.

Building on the hashing-based sketching framework of [Pham and Pagh, 2013], we introduce a Complex-to-Real (CtR) variant of TensorSketch for polynomial kernel approximation. Our method uses fourth roots-ofunity instead of real Rademacher variables and applies a structured complex-to-real transformation, yielding unbiased real-valued embeddings with improved variance dependence.

In contrast, our method achieves variance reduction through a lightweight modification of the sketch construction. This is followed by a Complex-to-Real (CtR) conversion to produce a real-valued embedding that makes it possible to compare it fairly with its real counterpart. The resulting sketch preserves the original structure and input-sparsity running time while achieving provable variance improvements over original TensorSketch [Pham and Pagh, 2013].

Variance reduction for randomized sketching has been widely studied using statistical variance reduction techniques such as control variates (CV) and maximum likelihood estimation (MLE). The CV method reduces variance by leveraging a correlated auxiliary variable with a known expectation, while MLE estimates unknown parameters by maximizing the likelihood of the observed data, often yielding statistically efficient estimators. These techniques have been extensively investigated for a variety of classical randomized sketching methods, including Johnson–Lindenstrauss (JL) transforms Li et al. [2006], CountSketch Pratap and Kulkarni [2021], Tug-of-War sketches Pratap et al. [2021], signed random projections Kang and Wong [2018], feature hashing Verma et al. [2022], and compressed matrix multiplication Verma et al. [2025], among others. However, to the best of our knowledge, analogous variance reduction techniques have not yet been developed for TensorSketch. While these approaches can substantially reduce estimator variance, they typically incur additional computational and algorithmic overhead.

## 3 PRELIMINARIES

Notation: We use the following notation in the paper. Bold lowercase letters $( \mathbf { e } . \mathbf { g } . , \mathbf { x } , \mathbf { y } )$ denote vectors, bold uppercase letters $( \mathbf { e . g . , S , T } )$ denote matrices. For a positive integer D, we write $[ D ] : = \{ 1 , 2 , \dots , D \}$ for the corresponding index set. For $d , p \in \mathbb { N } ,$ , we denote by $[ d ] ^ { p }$ the set of all p-tuples $( i _ { 1 } , \ldots , i _ { p } )$ with $i _ { j } \in [ d ]$ . The sets R, Z and C denote the real, integer and complex domains, respectively. For any vector x, nnz (x) denotes the number of its nonzero entries. The symbol $\langle \mathbf { x } , \mathbf { y } \rangle$ denotes the standard inner product, $\left\| \mathbf { x } \right\| _ { 2 }$ the Euclidean norm, and $\| \mathbf { S } \| _ { F }$ the Frobenius norm of a matrix. For a complex number $z \in \mathbb { C }$ written as $z = a + i b .$ its complex conjugate is defined as z. The norm of $z \ \mathrm { i s }$ $| z | = { \sqrt { a ^ { 2 } + b ^ { 2 } } } = { \sqrt { z { \overline { { z } } } } }$ . The Kronecker product is written as $\otimes$ , while $^ { 6 6 } \cdot ^ { 5 5 }$ denotes standard matrix multiplication and ⊙ denotes the elementwise (Hadamard) product. The indicator function is denoted by 1 [·]. The probabilistic quantities use $\mathrm { P r } [ \cdot ]$ for probability, E [·] for expectation and Var (·) for variance.

Definition 1 (CountSketch [Charikar et al., 2004]). Given an input vector $\mathbf { y } \in \mathbb { R } ^ { d }$ , the CountSketch is a randomized linear map $\mathbf { T } \in \mathbb { R } ^ { D \times d }$ that maps y to a lowerdimensional vector $\mathbf { z } = \mathbf { T } \mathbf { y } \in \mathbb { R } ^ { D }$ . The CountSketch matrix T is constructed by two hashfunctions: $( a ) h \colon [ d ] \to$ [D] a 2-wise independent hash function, and $( b ) s : [ d ] $ $\{ 1 , - 1 \}$ a 4-wise independent random signfunction.

The $j ^ { t h }$ entry of vector $\textbf { z } \in \mathbb { R } ^ { D }$ is computed as, $z _ { j } =$ $\begin{array} { r } { \sum _ { h ( i ) = j } s ( i ) y _ { i } , \forall j \in [ D ] } \end{array}$ . The time complexity of computing the CountSketch is $O ( \mathrm { n n z } \left( \mathbf { y } \right) )$ .

For pairwise inner–product analysis, let $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ and define the CountSketch inner–product estimator as

$$
\widehat { k } \left( \mathbf { x } , \mathbf { y } \right) : = \langle \mathbf { T x } , \mathbf { T y } \rangle .
$$

Then the estimator is unbiased

$$
\mathbb { E } \left[ \widehat { k } \left( \mathbf { x } , \mathbf { y } \right) \right] = \langle \mathbf { x } , \mathbf { y } \rangle ,
$$

and its variance satisfies

$$
\mathrm { V a r } \Big ( \widehat { k } \big ( \mathbf { x } , \mathbf { y } \big ) \Big ) \leq \frac { 1 } { D } \Big ( \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \big \| \mathbf { x } \big \| _ { 2 } ^ { 2 } \big \| \mathbf { y } \big \| _ { 2 } ^ { 2 } - 2 \sum _ { i } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \Big ) .
$$

Definition 2 (TensorSketch of Degree p [Pham and Pagh, 2013, 2025]). Let $p \ \geq \ 2 .$ . For each $t \in [ p ]$ , let $h _ { t } : [ d ] \ \to \ [ D ]$ be 2-wise independent hash functions and $\sigma _ { t } : [ d ]  \{ - 1 , + 1 \}$ be 4-wise independent random sign functions. The degree-p TensorSketch matrix $\mathbf { S \in }$ $\mathbb { R } ^ { \breve { D } \times d ^ { p } }$ is definedfor all $r \in [ D ]$ and $( i _ { 1 } , \dotsc , i _ { p } ) \in [ d ] ^ { p } b$ y

$$
\mathbf { S } _ { r , ( i _ { 1 } , \ldots , i _ { p } ) } = \left( \prod _ { t = 1 } ^ { p } \sigma _ { t } ( i _ { t } ) \right) \mathbf { 1 } \left[ \sum _ { t = 1 } ^ { p } h _ { t } ( i _ { t } ) \equiv r \pmod { D } \right] .
$$

For any $\mathbf { x } \in \mathbb { R } ^ { d }$ , the sketch $\mathbf { S } ( \mathbf { x } ^ { \otimes p } )$ can be computed implicitly in time $O ( p ( \mathrm { n n z } \left( \mathbf { x } \right) + D \log D ) )$ using FFT-based convolution, without $e x p l i c i t l y f o r m i n g \mathbf { x } ^ { \otimes p }$

Let $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ , then estimate for pairwise inner–product is defined as follows:

$$
\widehat { k } \left( \mathbf { x } , \mathbf { y } \right) : = \langle \mathbf { S } \left( \mathbf { x } ^ { \otimes p } \right) , \mathbf { S } \left( \mathbf { y } ^ { \otimes p } \right) \rangle .
$$

Then TensorSketch provides an unbiased estimator of the degree–p polynomial kernel

$$
\mathbb { E } \left[ \widehat { k } \left( \mathbf { x } , \mathbf { y } \right) \right] = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } ,
$$

and its variance satisfies

$$
\begin{array} { r l } & { \displaystyle \mathrm { V a r } \left( \widehat { k } \left( \mathbf { x } , \mathbf { y } \right) \right) \leq \frac { 1 } { D } \left( \left( 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \left. \mathbf { x } \right. _ { 2 } ^ { 2 } \left. \mathbf { y } \right. _ { 2 } ^ { 2 } \cdot \cdot \cdot } \\ & { \qquad \quad \cdot \ \cdot - 2 \displaystyle \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } - \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 p } \right) , } \\ & { \displaystyle \leq \frac { 3 ^ { p } - 1 } { D } \left. \mathbf { x } \right. _ { 2 } ^ { 2 p } \left. \mathbf { y } \right. _ { 2 } ^ { 2 p } . } \end{array}
$$

Definition 3 (Polynomial Kernel [Scholkopf and Smola, 2001]). We consider polynomial kernels of degree p ∈ N of the form

$$
k ( \mathbf x , \mathbf y ) = \left( \gamma \mathbf x ^ { \top } \mathbf y + \nu \right) ^ { p } ,
$$

for $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ with $\gamma , \nu \geq 0$ . The parameters γ and ν can be absorbed into the inputs by introducing the augmented vectors

$$
\begin{array} { r } { \tilde { \mathbf { x } } = \left( \sqrt { \gamma } \mathbf { x } ^ { \top } , \ \sqrt { \nu } \right) ^ { \top } , \qquad \tilde { \mathbf { y } } = \left( \sqrt { \gamma } \mathbf { y } ^ { \top } , \ \sqrt { \nu } \right) ^ { \top } \in \mathbb { R } ^ { d + 1 } . } \end{array}
$$

With this transformation, the kernel admits a homogeneous representation,

$$
\left( \gamma \mathbf { x } ^ { \top } \mathbf { y } + \nu \right) ^ { p } = ( \tilde { \mathbf { x } } ^ { \top } \tilde { \mathbf { y } } ) ^ { p } = \left( \tilde { \mathbf { x } } ^ { \otimes p } \right) ^ { \top } \tilde { \mathbf { y } } ^ { \otimes p } ,
$$

where $\widetilde { \mathbf { x } } ^ { \otimes p }$ denotes the p-fold tensor product of x˜. Hence, without loss of generality, we may treat the polynomial kernel as a homogeneous kernel in the augmented space $\mathbb { R } ^ { d + 1 }$

Definition 4 (Khintchine Inequality Haagerup and Musat [2007], Wacker et al. [2023]). Let $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { d } ) \in$ $\mathbb { R } ^ { d }$ and let $\{ \varepsilon _ { i } \} _ { i = 1 } ^ { d }$ be independent Rademacher random variables. For every $p > 0 ,$ , there exists a constant $K _ { p }$ such that

$$
\left( \mathbb { E } \left| \sum _ { i = 1 } ^ { d } x _ { i } \varepsilon _ { i } \right| ^ { p } \right) ^ { 1 / p } \leq K _ { p } \| \mathbf { x } \| _ { 2 } .\tag{1}
$$

The optimal constants are known explicitly.

$H f \varepsilon _ { i } \in \{ - 1 , + 1 \}$ are real-valued Rademacher variables, then

$$
K _ { p } = \left\{ \begin{array} { l l } { 1 , } & { 0 < p \leq 2 , } \\ { { \sqrt { 2 } } \pi ^ { - { \frac { 1 } { 2 p } } } \Gamma \left( { \frac { p + 1 } { 2 } } \right) ^ { \frac { 1 } { p } } , } & { p > 2 . } \end{array} \right.\tag{2}
$$

$I f \varepsilon _ { i }$ are complex Rademacher variables taking values in $\{ 1 , - 1 , i , - i \}$ uniformly at random, then

$$
\hat { K _ { p } } = \Gamma \Big ( \frac { p } { 2 } + 1 \Big ) ^ { \frac { 1 } { p } } .\tag{3}
$$

Moreover, for every $p > 2 ,$ , the complex constant $\hat { K _ { p } }$ is strictly smaller than the corresponding real constant $K _ { p } .$

Definition 5 (Framework for Complex-to-Real (CtR) Sketches). Wacker et al. [2023] suggests a framework of analysing sketching algorithms involving complex random variables. Let $z = a + i b$ be a complex random variable with $a , b \in \mathbb { R }$ , we have $| z | ^ { 2 } = a ^ { 2 } + \widehat { b } ^ { 2 }$ and $\operatorname { R e } { \left\{ z ^ { 2 } \right\} } =$ $a ^ { 2 } - b ^ { 2 }$ . Combining both gives $\begin{array} { r } { a ^ { 2 } = \frac { 1 } { 2 } \left( \left| z \right| ^ { 2 } + \mathrm { R e } \left\{ z ^ { 2 } \right\} \right) } \end{array}$ The scalar a is real-valued and its variance Var $( a ) \ =$ E $\left[ a ^ { 2 } \right] - \mathbb { E } \left[ a \right] ^ { 2 }$ is therefore

$$
\mathrm { V a r } ( a ) = \frac { 1 } { 2 } \mathrm { R e } \{ \mathbb { E } [ | z | ^ { 2 } ] + \mathbb { E } [ z ^ { 2 } ] - 2 \mathbb { E } [ a ] ^ { 2 } \} .\tag{4}
$$

Let x, $\mathbf { y } \in \mathbb { R } ^ { d }$ and let $\Phi _ { \mathrm { C } } : \mathbb { R } ^ { d ^ { p } }  \mathbb { C } ^ { \frac { D } { 2 } }$ . Then $\widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) : =$ $\Phi _ { \mathrm { C } } ( \mathbf { x } ^ { \otimes p } ) \overline { { \Phi _ { \mathrm { C } } ( \mathbf { y } ^ { \otimes p } ) } } ^ { \mathrm { ~ I ~ } } \in \mathrm { ~ \mathbb { C } ~ }$ is a complex-valued estimate of the polynomial kernel $k ( \mathbf { x } , \mathbf { y } )$ , and hence $\widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) =$ $\mathrm { R e } \{ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \} + i \mathrm { I m } \{ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \}$ . The Complex-to-Real sketch is defined asfollows:

$$
\begin{array} { r l } & { \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) : = \mathrm { R e } \{ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \} , } \\ & { ~ = \mathrm { R e } \left\{ \Phi _ { \mathrm { C } } ( \mathbf { x } ^ { \otimes p } ) \right\} ^ { \top } \mathrm { R e } \left\{ \Phi _ { \mathrm { C } } ( \mathbf { y } ^ { \otimes p } ) \right\} + } \\ & { ~ \cdot \cdot + \mathrm { I m } \left\{ \Phi _ { \mathrm { C } } ( \mathbf { x } ^ { \otimes p } ) \right\} ^ { \top } \mathrm { I m } \left\{ \Phi _ { \mathrm { C } } ( \mathbf { y } ^ { \otimes p } ) \right\} , } \\ & { ~ = \Phi _ { \mathrm { C t R } } ( \mathbf { x } ) ^ { \top } \Phi _ { \mathrm { C t R } } ( \mathbf { y } ) . } \end{array}
$$

where, $\Phi _ { \mathrm { C t R } } ( \mathbf { x } ) : = [ \mathrm { R e } \left\{ \Phi _ { \mathrm { C } } \right\}$ , Im $\{ \Phi _ { \mathrm { C } } \} ] \in \mathbb { R } ^ { D }$ . We now derive the variance of bk<sub>CtR</sub> $\displaystyle ( \mathbf { x } , \mathbf { y } )$ using Equation (4):

$$
\begin{array} { l } { \displaystyle \mathrm { V a r } \left( \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right) } \\ { \displaystyle = \frac { 1 } { 2 } \mathrm { R e } \{ \mathbb { E } [ | \widehat { k } _ { C } | ^ { 2 } ] + \mathbb { E } [ \widehat { k } _ { C } ^ { 2 } ] + 2 \mathbb { E } [ \mathrm { R e } \{ \widehat { k } _ { C } \} ] ^ { 2 } \} } \\ { \displaystyle = \frac { 1 } { 2 } \mathrm { R e } \{ \mathbb { E } [ | \widehat { k } _ { C } | ^ { 2 } ] + \mathbb { E } [ \widehat { k } _ { C } ^ { 2 } ] + 2 \mathbb { E } [ \widehat { k } _ { C } ] ^ { 2 } \} , } \end{array}\tag{5}
$$

where, $\widehat { k } _ { C } : = \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \in \mathbb { C }$ . This framework plays a key role in computing the expectation and variance of $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right)$

Definition 6 (Complex-to-Real Polynomial Sketch [Wacker et al., 2023]). Let p ∈ N and $D = 2 k f o r$ some $k \in \mathbb N$ For each $i \in [ p ]$ , let $\mathbf { W } _ { i } \in \mathbb { C } ^ { D / 2 \times d }$ be a random matrix whose rows are sampled independently from a zero-mean distribution satisfying $\mathbb { E } \left[ \mathbf { w w } ^ { * } \right] = \mathbf { I } _ { d } \left( e . g . \right)$ , complex Gaussian or complex Rademacher) where ∗ refers to conjugatetranspose. Define the complex random feature map

$$
\Phi _ { C } ( \mathbf { x } ^ { \otimes p } ) : = \sqrt { \frac { 2 } { D } } \left( \mathbf { W _ { 1 } x } \mathbf { \odot } \mathbf { W _ { 2 } x } \mathbf { \odot } \mathbf { \cdot } \mathbf { \cdot } \mathbf { \odot } \mathbf { W _ { p } } \mathbf { x } \right) \in \mathbb { C } ^ { D / 2 } ,
$$

Then, the Complex-to-Real (CtR) sketch $o f \mathbf { x } \in \mathbb { R } ^ { d ^ { p } }$ is the real-valued vector $\Phi _ { \mathrm { C t R } } \big ( \mathbf { x } ^ { \otimes p } \big ) \in \mathbb { R } ^ { D }$ defined by

$$
\begin{array} { r } { \Phi _ { \mathrm { C t R } } ( \mathbf { x } ^ { \otimes p } ) : = \left[ \operatorname { R e } \left\{ \Phi _ { \mathrm { C t R } } ( \mathbf { x } ^ { \otimes p } ) \right\} \right] \in \mathbb { R } ^ { D } . } \end{array}
$$

For any $\mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { p } }$ , the kernel estimator can be defined as

$$
\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) = \Phi _ { \mathrm { C t R } } ( \mathbf { x } ) ^ { \top } \Phi _ { \mathrm { C t R } } ( \mathbf { y } ) .
$$

For both Gaussian and Rademacher constructions, the CtR estimator satisfied the following:

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \left[ \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right] = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } , } \\ & { \displaystyle \mathrm { V a r } \left( \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right) \leq \frac { 2 ^ { p + 1 } - 2 } { D } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p } . } \end{array}
$$

## 4 COMPLEX-TO-REAL(CTR) TENSORSKETCH

In this section, we first define Complex CountSketch (Definition 7). Building on its circular convolution structure, we then introduce Complex-to-Real TensorSketch. (Definition 8) along with the associated kernel. Theorem 2 discusses the guarantees of the estimators proposed in Definition 8.

Definition 7 (Complex CountSketch Mapping). Let $\mathbf { x \in }$ R<sup>d</sup> and D be the sketching dimension. Sample a Complex CountSketch matrix $\bar { \mathbf { C } } \in \mathbb { C } ^ { D \times d }$ with the following two independent functions

$h : [ d ]  [ D ]$ is an universal hash function that assigns each coordinate independently and uniformly to one ofthe D buckets, and

$s : [ d ] \to \{ 1 , \omega , \omega ^ { 2 } , \omega ^ { 3 } \}$ is a random function whose values are drawn independently and uniformly from the four fourth roots of unity.

The j-th coordinate ofthe sketched vector Cx is given by

$$
( { \bf C x } ) _ { j } = \sum _ { i = 1 } ^ { d } s ( i ) { \bf 1 } [ h ( i ) = j ] x _ { i } , \qquad \forall j \in [ D ] .
$$

We define CtR TensorSketch, which extends the Complex CountSketch by applying p independent sketches and combining them using FFT-based convolution to efficiently sketch the vector $\mathbf { x } ^ { \mathsf { \bar { \otimes } } p } , \mathbf { y } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { p } }$

Definition 8. (Complex-to-Real (CtR) TensorSketch Mapping) Let $\textbf { x } \in \mathbb { R } ^ { d }$ and fix an integer $p \geq 1$ . Define a Complex TensorSketch mapping $\mathbf { C } : \dot { \mathbb { R } } ^ { d ^ { p } }  \dot { \mathbb { C } } ^ { D / 2 }$ constructed from p independent Complex CountSketch mappings. For each $r \in [ p ]$ , let

$h _ { r } : [ d ]  [ D / 2 ]$ be an universal hash function that assigns each coordinate independently and uniformly to one ofthe D buckets, and

$s _ { r } : [ d ] \to \{ 1 , \omega , \omega ^ { 2 } , \omega ^ { 3 } \}$ be a randomfunction whose values are drawn independently and uniformly from thefourth roots ofunity.

For each $r \in [ p ] ,$ , let $\mathbf { C } _ { r } \in \mathbb { C } ^ { D / 2 \times d }$ denote the Complex CountSketch matrix induced by functions $\left( h _ { r } , s _ { r } \right)$ as in Definition 7. The TensorSketch of $\mathbf { x } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { p } }$ is defined implicitly via convolution of the p sketches, and can be computed efficiently using the Fast Fourier Transform as

(6)

$$
\mathbf { \Psi } = \mathrm { F F T } ^ { - 1 } \left( \bigcirc _ { r = 1 } ^ { p } \mathrm { F F T } ( \mathbf { C } _ { r } \mathbf { x } ) \right) \in \mathbb { C } ^ { D / 2 } ,\tag{7}
$$

where the product is taken element-wise. We define the CtR sketch as

$$
\begin{array} { r } { \Phi _ { \mathrm { C t R } } ( \mathbf { x } ^ { \otimes p } ) : = \left( \mathrm { R e } \{ \Phi _ { \mathrm { C } } ( \mathbf { x } ^ { \otimes p } ) _ { 1 } \} , \dots , \mathrm { R e } \{ \Phi _ { \mathrm { C } } ( \mathbf { x } ^ { \otimes p } ) _ { D / 2 } \} , \right. } \\ { \left. \mathrm { I m } \{ \Phi _ { \mathrm { C } } ( \mathbf { x } ^ { \otimes p } ) _ { 1 } \} , \dots , \mathrm { I m } \{ \Phi _ { \mathrm { C } } ( \mathbf { x } ^ { \otimes p } ) _ { D / 2 } \} \right) ^ { \top } \in \mathbb { R } ^ { D } . } \end{array}
$$

A detailed proof of results (Lemma 1, 4, and Theorem 2) stated in this section is presented in Appendix A.1.

Lemma 1. Let $\omega = e ^ { 2 \pi i / 4 } = i ,$ , and let $s : [ d ] $ $\{ 1 , \ : \omega , \ : \omega ^ { 2 } , \ : \omega ^ { 3 } \}$ be a randomfunction such that the values $\{ s ( i ) \} _ { i \in [ d ] }$ are drawn independently and uniformly from the four fourth roots of unity. Then, for every $i \in [ d ]$ , the following identities hold

$$
\begin{array} { r } { \mathbb { E } \left[ s ( i ) \right] = 0 , } \\ { \mathbb { E } [ | s ( i ) | ^ { 2 } ] = 1 , } \\ { \mathbb { E } \left[ s ( i ) ^ { 2 } \right] = 0 . } \end{array}
$$

Furthermore, for any pair of distinct indices $i \neq j ,$ , independence implies

$$
\begin{array} { r } { \mathbb { E } [ s ( i ) \overline { { s ( j ) } } ] = \mathbb { E } [ s ( i ) ] \mathbb { E } \left[ \overline { { s ( j ) } } \right] = 0 . } \end{array}
$$

Theorem 2 establishes the unbiasedness, variance bound, and sketching time of the CtR TensorSketch.

Theorem 2. Let $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ and $\mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { p } }$ . Let Φ<sub>CtR</sub> denote a Complex-to-Real TensorSketch as stated in Definition 8. Let $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) : = \Phi _ { \mathrm { C t R } } ( \mathbf { x } ^ { \otimes p } ) ^ { \top } \Phi _ { \mathrm { C t R } } ( \mathbf { y } ^ { \otimes p } )$ then $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right)$ satisfies

$$
\begin{array} { r l r } & { } & { \mathbb { E } \left[ \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right] = \langle \mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \rangle = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } , } \\ & { } & { \mathrm { V a r } \left( \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right) \leq \frac { 2 ^ { p + 1 } - 2 } { D } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p } . } \end{array}
$$

Moreover, the sketches $\Phi _ { \mathrm { C t R } } ( \mathbf { x } ^ { \otimes p } )$ and $\Phi _ { \mathrm { C t R } } ( \mathbf { y } ^ { \otimes p } )$ can be computed in $O ( p ( \mathrm { n n z } \left( \mathbf { x } \right) + D \log D ) )$ and $O ( p ( { \mathrm { n n z } } \left( \mathbf { y } \right) + { }$ D log D)) time, respectively.

Proof. According to Definition 5, we have $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) : =$ Re $\left\{ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right\}$ , and therefore we first analyze $\widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right)$ Recall that $\begin{array} { r } { \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) : = \Phi _ { \mathrm { C } } ( \mathbf { x } ^ { \otimes p } ) \overline { { \Phi _ { \mathrm { C } } ( \mathbf { y } ^ { \otimes p } ) } } ^ { \top } } \end{array}$ , where $\Phi _ { \mathrm { { C } } } ( \mathbf { x } ^ { \otimes p } ) : = \mathbf { C } \mathbf { x } ^ { \otimes p }$ and $\Phi _ { \mathbf { C } } ( \mathbf { y } ^ { \otimes p } ) : = \mathbf { C } \mathbf { y } ^ { \otimes p }$ according to

<table><tr><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>Variance</td><td rowspan=1 colspan=1>Sketching Time</td></tr><tr><td rowspan=1 colspan=1>TensorSketch(CtR)</td><td rowspan=1 colspan=1> $\frac { \mathbf { \left. 2 ^ { \mathit { p } + 1 } - 2 \right. } } { \mathit { D } } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 \mathit { p } } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 \mathit { p } }$ </td><td rowspan=1 colspan=1> $O ( p ( \mathrm { n n z } \left( \mathbf { x } \right) + \mathrm { n n z } \left( \mathbf { y } \right) + D \log D ) )$ </td></tr><tr><td rowspan=1 colspan=1>TensorSketch(Real)[Pham and Pagh, 2013, 2025]</td><td rowspan=1 colspan=1> $\frac { 3 ^ { p } - 1 } { D } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p }$ </td><td rowspan=1 colspan=1> $O ( p ( \mathrm { n n z } \left( \mathbf { x } \right) + \mathrm { n n z } \left( \mathbf { y } \right) + D \log D ) )$ </td></tr><tr><td rowspan=1 colspan=1>JL(CtR Radamacher)[Wacker et al., 2023]</td><td rowspan=1 colspan=1> $\frac { \mathbf { \bar { \rho } } ^ { 2 ^ { p + 1 } - 2 } \mathbf { | | x | | } ^ { 2 p } \mathbf { | | y | } ^ { 2 p } } { D } \mathbf { | | x | | } ^ { 2 p } \mathbf { | | y | } | _ { 2 } ^ { 2 p }$ </td><td rowspan=1 colspan=1> $O ( p D d )$ </td></tr><tr><td rowspan=1 colspan=1>JL(CtR Guassian)[Wacker et al., 2023]</td><td rowspan=1 colspan=1> $\frac { \mathbf { \phi } _ { 2 ^ { p + 1 } - 2 } } { D } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p }$ </td><td rowspan=1 colspan=1> $O ( p D d )$ </td></tr></table>

Table 1: Comparison of variance bounds and sketching time for CtR TensorSketch and baseline methods. Here, $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ are input vectors, p denotes the polynomial degree, and D is the sketch dimension. All methods provide unbiased estimates of the inner product between the vectors $\mathbf { x } ^ { \otimes p } , \mathbf { \bar { y } } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { p } }$ , i.e., $\langle \mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \rangle$ .

Definition 8. In particular, $\mathbf { C } \mathbf { x } ^ { \otimes p } , \mathbf { C } \mathbf { y } ^ { \otimes p } \in \mathbb { C } ^ { D / 2 }$ denote the complex CountSketch of the vectors $\mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { p } }$ constructed using the derived functions $H : [ d ] ^ { p } \mapsto [ D / 2 ]$ and $S : [ d ] ^ { p }  \langle \overline { { { 1 } } } , \omega , \omega ^ { 2 } , \omega ^ { 3 } \}$ defined as

$$
\begin{array} { l } { { \displaystyle { \cal H } ( i _ { 1 } , \dots , i _ { p } ) = \left( \sum _ { j = 1 } ^ { p } h _ { j } ( i _ { j } ) \right) \bmod { \cal D } / 2 , } } \\ { { \displaystyle { \cal S } ( i _ { 1 } , \dots , i _ { p } ) = \prod _ { j = 1 } ^ { p } s _ { j } ( i _ { j } ) . } } \end{array}
$$

In the proof, we denote $X : = \mathbf { x } ^ { \otimes p }$ and $Y : = \mathbf { y } ^ { \otimes p }$ . Let $u , v \in [ d ] ^ { p }$ be the multi-indices corresponding to the entries of $X , \dot { Y } \in \mathbb { R } ^ { d ^ { p } }$ . For a multi-index $u = ( j _ { 1 } , \ldots , j _ { p } )$ , the entry of X is defined as $\begin{array} { r } { X _ { u } = \prod _ { k = 1 } ^ { p } x _ { j _ { k } } } \end{array}$ , where the associated linearized index is $\begin{array} { r } { u = 1 + \sum _ { k = 1 } ^ { p } ( j _ { k } - 1 ) \prod _ { k ^ { \prime } = 1 } ^ { p - k } d . } \end{array}$ The entries of Y are defined analogously.

We begin by analyzing $\widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right)$ as defined below,

$$
\begin{array} { r l } & { \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) : = \Phi _ { \mathrm { C } } ( X ) \overline { { \Phi _ { \mathrm { C } } ( Y ) } } ^ { \top } = \langle \mathbf { C } X , \overline { { \mathbf { C } Y } } \rangle , } \\ & { ~ = \displaystyle \sum _ { u , v \in [ d ] ^ { p } } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \mathbf { 1 } \left[ H ( u ) = H ( v ) \right] , } \\ & { ~ = ~ \langle X , Y \rangle + \displaystyle \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \mathbf { 1 } \left[ H ( u ) = H ( v ) \right] . } \end{array}
$$

Using Lemma 1, we have $\mathbb { E } \left[ S ( u ) \overline { { S ( v ) } } \right] = 0$ for all $u \ne v$ Hence,

$$
\begin{array} { r } { \mathbb { E } \left[ \widehat { k } _ { \mathrm { C } } ( \mathbf { x } , \mathbf { y } ) \right] = \langle X , Y \rangle = \langle \mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \rangle = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } . } \end{array}
$$

According to Definition 5, we know $\begin{array} { r l } { \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) } & { { } : = } \end{array}$ Re $\left\{ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right\}$ , we have

$$
\begin{array} { r } { \mathbb { E } \left[ \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right] = \mathbb { E } \left[ \mathrm { R e } \left\{ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right\} \right] = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } . } \end{array}\tag{8}
$$

Using Equation 5, the variance of $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right)$ is

$$
\begin{array} { r l r } & { } & { \mathrm { V a r } \Big ( \widehat { k } _ { \mathrm { C t R } } \left( { \bf x } , { \bf y } \right) \Big ) = \frac { 1 } { 2 } \mathrm { R e } \Bigg \{ { \mathbb E } \left[ \Big | \widehat { k } _ { \mathrm { C } } \left( { \bf x } , { \bf y } \right) \Big | ^ { 2 } \right] + \cdots } \\ & { } & { \cdot \cdot + { \mathbb E } \left[ \Big ( \widehat { k } _ { \mathrm { C } } \left( { \bf x } , { \bf y } \right) \Big ) ^ { 2 } \right] - 2 \Big ( { \mathbb E } \left[ \widehat { k } _ { \mathrm { C } } \left( { \bf x } , { \bf y } \right) \right] \Big ) ^ { 2 } \Bigg \} . } \end{array}\tag{9}
$$

For the variance analysis, we need to compute $\mathbb { E } \left\lceil \left| \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right| ^ { 2 } \right\rceil$ and $\mathbb { E } \left\lceil \left( \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right) ^ { 2 } \right\rceil$ . By applying Lemma 1 and Lemma 4, we obtain the following bounds:

$$
\begin{array} { r l r } {  { \mathbb { E } [ | \widehat { k } _ { \mathrm { C } } ( \mathbf { x } , \mathbf { y } ) | ^ { 2 } ] \leq \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 p } + \frac { 2 } { D } ( ( \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } +   } } \\ & { } & {   \cdot +  \mathbf { x }  _ { 2 } ^ { 2 }  \mathbf { y }  _ { 2 } ^ { 2 } - \underset { i = 1 } { \overset { d } { \sum } } x _ { i } ^ { 2 } y _ { i } ^ { 2 } ) ^ { p } - \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 p } ) , } \end{array}\tag{10}
$$

and

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \left( \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right) ^ { 2 } \right] \leq \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 p } + \displaystyle \frac { 2 } { D } \left( \left( 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \right. \right. } \\ { \displaystyle \left. \left. \cdots - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } - \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 p } \right) . } \end{array}\tag{11}
$$

Substituting Equations (10), (11), and (8) into Equation (9) yields

$$
\mathrm { V a r } \left( \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right) \leq \frac { 2 ^ { p + 1 } - 2 } { D } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p } .\tag{12}
$$

Time Complexity. To compute the sketch according to Equation (7), each complex CountSketch can be computed in O(nnz (x)) time. The convolution of the p sketches is implemented using FFT in O(pD log D) time. Therefore, the overall time complexity of Complex-to-Real (CtR) TensorSketch applied to $\mathbf { x } ^ { \otimes p }$ is $O ( p ( { \mathrm { n n z } } \left( \mathbf { x } \right) + { }$ D log D). □

Remark 3. Theorem 2, exhibits an improved exponential dependence on p in the variance bound of the estimate, decreasing from the 3<sup>p</sup>/D bound for real TensorSketch [Pham and Pagh, 2013] to 2<sup>p</sup>/D, while retaining the same input-sparsity sketching time. A tabular comparison among the baselines on the variance bound and running time is presented in Table 1.

Bounds derived in the Lemma 4 play a key role in the proof of Theorem 2. The sketch described in this lemma is a complex-valued variant of the classical AMS sketch Alon et al. [1999] for polynomial kernel approximation Indyk and McGregor [2008], Braverman et al. [2010]. In particular, the variance analysis of CtR TensorSketch reduces to analyzing products of independently randomized linear sketches. Each bucket of CtR TensorSketch behaves like a product of $p$ independent AMS-type sketches of the form $\begin{array} { r } { Z _ { s _ { i } } ( \mathbf { x } ) \ = \ \sum _ { i = 1 } ^ { d } x _ { i } s _ { j } ( i ) } \end{array}$ . When expanding the moments $\mathbb { E } [ | Z | ^ { 2 } ]$ and $\mathbb { E } [ Z ^ { 2 } ]$ in the proof of Theorem 2, the resulting expressions factor across the $p$ independent random functions drawn uniformly from fourth roots of unity. Lemma 4 provides an exact evaluation of moments under fourth roots-of-unity random functions, allowing the full degree-p moment to be written as the p-th power of a singlesketch expression.

Lemma 4. Let x, $\mathbf { y } \in \mathbb { R } ^ { d } ,$ , let $p \geq 1$ be an integer, and let $s _ { 1 } , \ldots , s _ { p } : [ d ] \to \{ 1 , \omega , \omega ^ { 2 } , \omega ^ { 3 } \}$ be independent random functions, each taking values uniformly from the four fourth roots ofunity (as in Lemma 1). Define

$$
Z = \prod _ { j = 1 } ^ { p } Z _ { s _ { j } } ( { \bf x } ) \overline { { { Z _ { s _ { j } } ( { \bf y } ) } } } ,
$$

where

$$
Z _ { s _ { j } } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { d } { x _ { i } s _ { j } ( i ) } , \qquad Z _ { s _ { j } } ( \mathbf { y } ) = \sum _ { i = 1 } ^ { d } { y _ { i } s _ { j } ( i ) } .
$$

Then,

$$
\begin{array} { r } { \mathbb { E } \left[ Z \right] = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } , \quad \quad } \\ { \mathrm { V a r } \left( Z \right) \leq 2 ^ { p } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p } . } \end{array}
$$

Proof. Let

$$
Z = \prod _ { j = 1 } ^ { p } Z _ { s _ { j } } ( { \bf x } ) \overline { { Z _ { s _ { j } } ( { \bf y } ) } } , \qquad Z _ { s _ { j } } ( { \bf x } ) = \sum _ { i = 1 } ^ { d } x _ { i } s _ { j } ( i ) .
$$

Expectation. For a fixed $j$

$$
\mathbb { E } \left[ Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } \right] = \sum _ { i , k } x _ { i } y _ { k } \mathbb { E } \left[ s _ { j } ( i ) \overline { { s _ { j } ( k ) } } \right] .
$$

By Lemma 1, E $\left[ s _ { j } ( i ) \overline { { s _ { j } ( k ) } } \right] = 0$ for $i \neq k$ and equals 1 for $i = k ,$ . Hence,

$$
\mathbb { E } \left[ Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } \right] = \langle \mathbf { x } , \mathbf { y } \rangle .
$$

Independence across j then gives

$$
\mathbb { E } \left[ Z \right] = \prod _ { j = 1 } ^ { p } \langle \mathbf { x } , \mathbf { y } \rangle = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } .
$$

Second moments. Since the sign functions are independent across $j$

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \left. Z \right. ^ { 2 } \right] = \prod _ { j = 1 } ^ { p } \mathbb { E } \left[ | Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } | ^ { 2 } \right] , } \\ { \displaystyle \mathbb { E } \left[ Z ^ { 2 } \right] = \prod _ { j = 1 } ^ { p } \mathbb { E } \left[ \left( Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } \right) ^ { 2 } \right] . } \end{array}
$$

Expanding one factor and using the fourth–moment structure of fourth roots-of-unity random functions (Lemma 1), only index configurations that form pairs survive. The detailed calculation yields moments as follows

$$
\mathbb { E } \left[ | Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } | ^ { 2 } \right] = \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } ,
$$

and

$$
\mathbb { E } \left[ \left( Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } \right) ^ { 2 } \right] = 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } .
$$

Therefore, taking p-times product of the terms give moment bounds for $Z$

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ | Z | ^ { 2 } ] = \Big ( \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \Big ) ^ { p } , } \\ { \displaystyle \mathbb { E } \left[ Z ^ { 2 } \right] = \Big ( 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \Big ) ^ { p } . } \end{array}
$$

Variance bound. Using Equation 4, and applying the Cauchy-Schwarz inequality to upper bound $\langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } \ \leq$ $\| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 }$ , together with the fact that $\textstyle \sum _ { i } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \geq 0 .$ , we obtain

$$
\operatorname { V a r } \left( Z \right) \leq 2 ^ { p } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p } .
$$

□

## 5 EXPERIMENTS

Experimental setup. We compare our proposed CtR TensorSketch against standard TensorSketch [Pham and Pagh, 2013]. Along with this we included JL-type polynomial sketches (Gaussian and Rademacher [Kar and Karnick, 2012]) together with their CtR variants [Wacker et al., 2023] for comparison. All methods are implemented by us directly from the original algorithmic descriptions given in the previous sections.

All experiments were run on Ubuntu 22.04.4 using an Intel<sup>®</sup> Core™ i9-14900K processor (24 cores, 32 threads) with 32 GB RAM. Reported runtimes are wall-clock times for sketch construction only.

Baselines. The sketching methods compared in this section are listed below, along with brief descriptions.

![](images/bae9de03e296d9fea90722e1227d54d70f1bc135bddc3aec050415d6672f8687.jpg)  
Figure 1: KL divergence on the COD–RNA dataset. We report KL divergence between the exact degree-p polynomial kernel and the kernel reconstructed from sketch features for $p \in \{ 3 , 5 , 7 \}$ . Methods include Real and CtR (complex-to-real) Gaussian and Rademacher JL sketches, as well as Real and CtR TensorSketch. Results are averaged over 20 independent trials. Sketch dimension is varied as $D \in \{ d , 3 d , 5 d \}$

![](images/26ebfff42bf765c274696a79e90d8a3596f6bba83880871948d0914497622d58.jpg)  
Figure 2: Wall-clock sketch construction time on the COD–RNA dataset. We compare Real and CtR (complex-to-real) Gaussian and Rademacher JL sketches, with Real and CtR TensorSketch for $p \in \{ 3 , 7 , 1 0 \}$ . The sketch dimension is varied as $D \in \{ d , 3 d , 5 d \}$ . Each point reports the average sketch construction time over 20 independent trials, measured on identical normalized input data. Here as well all dense JL-type sketches exhibit nearly overlapping construction times across sketch dimensions $D$ and polynomial degrees $p .$

• Real Gaussian/Rademacher Sketch [Kar and Karnick, 2012]: Dense JL-type polynomial sketches using i.i.d. Gaussian or Rademacher projection matrices.

• Real TensorSketch [Pham and Pagh, 2013, 2025]: Standard CountSketch-based TensorSketch for polynomial kernels with real random sign functions.

• CtR Gaussian/Rademacher Sketch [Wacker et al., 2023]: complex-to-real JL-type sketches using complex Gaussian or complex Rademacher projections followed by CtR conversion, as proposed by [Wacker et al., 2023].

• CtR TensorSketch (Our Proposal): Our CtR TensorSketch obtained by replacing real random signs with random function drawn uniformly from the four fourth roots of unity and applying complex-to-real conversion within the corresponding constructions.

Datasets. We use both synthetic and real-world datasets. Synthetic data. For the synthetic experiments, we generate $n = 3 0 0 0$ random vectors in $\mathbb { R } ^ { d }$ with $d = 2$ . Each coordinate is drawn independently from a standard Gaussian distribution, and all vectors are ℓ<sub>2</sub>–normalized. We consider polynomial kernels of degrees $p \in \{ 1 0 , 1 5 , 2 0 \}$ and vary the sketch dimension as $D \in \{ 1 0 0 , 3 0 0 , 5 0 0 \}$

Real–world data. We evaluate all methods on two real-world datasets: MAGIC Gamma Telescope [Bock, 2004] (d = 10 real-valued features) and COD-RNA [Uzilov et al., 2006] (d = 8 numerical attributes). All inputs are treated as realvalued vectors and $\ell _ { 2 } \cdot$ -normalized prior to sketching. For each dataset, we subsample up to $n \ : = \ : 3 0 0 0$ points (or use the full dataset if smaller). Each configuration (dataset, degree $p ,$ sketch dimension D) is averaged over 20 independent random trials.

Comparison Metrics. We evaluate sketch quality and efficiency using two metrics.

• KL divergence. To assess how well a sketch preserves the structure of the degree-p polynomial kernel matrix, we compute the KL divergence between the exact kernel K and its sketch-based approximation $\widehat { K }$ . Since polynomial kernels are nonnegative, we normalize both matrices into discrete distributions

$$
P _ { i j } = \frac { K _ { i j } } { \sum _ { a , b } K _ { a b } } , \qquad Q _ { i j } = \frac { \widehat { K } _ { i j } } { \sum _ { a , b } \widehat { K } _ { a b } } ,
$$

and report

$$
\mathrm { K L } ( K \parallel \widehat { K } ) = \sum _ { i , j } P _ { i j } \log \left( \frac { P _ { i j } } { Q _ { i j } + \varepsilon } \right) ,
$$

where $\varepsilon > 0$ ensures numerical stability. Lower values indicate better kernel preservation.

• Wall–clock time. We measure efficiency by the wall– clock time required to construct sketch features for a given method and sketch dimension D, averaged over multiple independent trials.

Results. Figures 1 and 2 compare approximation error (KL divergence) and sketch construction time on the COD–RNA dataset across polynomial degrees and sketch dimensions. We evaluate the proposed CtR TensorSketch against the standard TensorSketch, along with real and CtR Gaussian and Rademacher JL-type sketches.

CtR TensorSketch consistently achieves the lowest KL divergence across all degrees and sketch dimensions, indicating more accurate kernel approximation than real TensorSketch and dense JL-type methods. This behavior matches our theoretical guarantee of improved variance established earlier.

In terms of runtime, CtR TensorSketch retains the inputsparsity cost O(p(nnz (x) + D log D)), comparable to standard TensorSketch. In contrast, JL-type methods incur higher dense projection cost O(pDd).

Additional experimental results are provided in Section C of the Appendix.

## 6 CONCLUSION

We introduced a complex-to-real (CtR) variant of TensorSketch that provides a compression algorithm for high-dimensional polynomial-kernel and tensor datasets. To the best of our knowledge, CtR constructions were only known for dense JL-type TensorSketch [Pham and Pagh, 2013] due to Wacker et al. [2023]. Our results demonstrate that Complex-to-Real (CtR) variants achieve improved variance bounds compared to their real counterparts [Pham and Pagh, 2013], matching those obtained in Wacker et al. [2023]. Moreover, these variants preserve the input-sparsity running time of the original method, making it a preferred choice over Wacker et al. [2023], for high-dimensional sparse data.

Two directions remain open for further investigation. First, it is unclear whether employing higher-order roots of unity can yield stronger higher-moment concentration, leading to tighter (ϵ, δ) -approximation guarantees for the sketch. Second, it remains to be explored whether the advantages of Complex-to-Real (CtR) constructions can be extended to other kernel families and randomized feature maps beyond polynomial kernels. Together, these questions point to a broader potential for complex-valued sketch design.

## Acknowledgements

This research was partially supported by the ANRF under Project No. ANRF/ECRG/2024/001063/ENS. We gratefully acknowledge ANRF for this support.

## References

Noga Alon, Yossi Matias, and Mario Szegedy. The space complexity of approximating the frequency moments. Journal of Computer and System Sciences, 58(1):137–147, 1999. ISSN 0022- 0000. doi: https://doi.org/10.1006/jcss.1997.1545. URL https://www.sciencedirect.com/ science/article/pii/S0022000097915452.

Hélène Aschard. A perspective on interaction effects in genetic association studies. Genetic Epidemiology, 40 (8):678–688, December 2016. ISSN 0741-0395. doi: 10.1002/gepi.21989. Epub 2016 Jul 7.

R. Bock. MAGIC Gamma Telescope. UCI Machine Learning Repository, 2004. DOI: https://doi.org/10.24432/C52C8B.

Vladimir Braverman, Kai-Min Chung, Zhenming Liu, Michael Mitzenmacher, and Rafail Ostrovsky. Ams without 4-wise independence on product domains, 2010. URL https://arxiv.org/abs/0806.4790.

Moses Charikar, Kevin C. Chen, and Martin Farach-Colton. Finding frequent items in data streams. Theor. Comput. Sci., 312(1):3–15, 2004. doi: 10.1016/ S0304-3975(03)00400-6. URL https://doi.org/ 10.1016/S0304-3975(03)00400-6.

Akira Fukui, Dong Huk Park, Daylen Yang, Anna Rohrbach, Trevor Darrell, and Marcus Rohrbach. Multimodal compact bilinear pooling for visual question answering and visual grounding. In Jian Su, Kevin Duh, and Xavier Carreras, editors, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 457–468, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/ D16-1044. URL https://aclanthology.org/ D16-1044/.

Yang Gao, Oscar Beijbom, Ning Zhang, and Trevor Darrell. Compact bilinear pooling. In Proceedings of the IEEE

Conference on Computer Vision and Pattern Recognition (CVPR), pages 317–326, Los Alamitos, CA, 2016. IEEE Computer Society. doi: 10.1109/CVPR.2016.41. URL https://doi.org/10.1109/CVPR.2016.41.

Yoav Goldberg and Michael Elhadad. splitSVM: Fast, spaceefficient, non-heuristic, polynomial kernel computation for NLP applications. In Johanna D. Moore, Simone Teufel, James Allan, and Sadaoki Furui, editors, Proceedings ofACL-08: HLT, Short Papers, pages 237–240, Columbus, Ohio, June 2008. Association for Computational Linguistics. URL https://aclanthology. org/P08-2060/.

Uffe Haagerup and Magdalena Musat. On the best constants in noncommutative khintchine-type inequalities. Journal ofFunctional Analysis, 250(2):588–624, 2007.

Piotr Indyk and Andrew McGregor. Declaring independence via the sketching of sketches. In Proceedings of the Nineteenth Annual ACM-SIAM Symposium on Discrete Algorithms, SODA ’08, page 737–745, USA, 2008. Society for Industrial and Applied Mathematics.

William B. Johnson and Joram Lindenstrauss. Extensions of Lipschitz mappings into a Hilbert space. Contemporary Mathematics, 26:189–206, 1984. URL https://www. ams.org/books/conm/026/.

Keegan Kang and Weipin Wong. Improving sign random projections with additional information. In Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80, pages 2479–2487, Cambridge, MA, 2018. PMLR. URL https://proceedings. mlr.press/v80/kang18b.html.

Purushottam Kar and Harish Karnick. Random feature maps for dot product kernels. In Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics (AISTATS), volume 22, pages 583– 591, Cambridge, MA, 2012. JMLR.org. URL http:// proceedings.mlr.press/v22/kar12.html.

Ping Li, Trevor Hastie, and Kenneth Ward Church. Very sparse random projections. In Proceedings of the Twelfth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pages 287–296, New York, NY, 2006. ACM. doi: 10.1145/ 1150402.1150436. URL https://doi.org/10. 1145/1150402.1150436.

Zhu Li, Jean-Francois Ton, Dino Oglic, and Dino Sejdinovic. Towards a unified analysis of random fourier features. In International conference on machine learning, pages 3905–3914. PMLR, 2019.

Raphael A Meyer and Haim Avron. Hutchinson’s estimator is bad at kronecker-trace-estimation. SIAM Journal on Matrix Analysis and Applications, 47(1):353–387, 2026.

Ninh Pham and Rasmus Pagh. Fast and scalable polynomial kernels via explicit feature maps. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pages 239–247, New York, NY, 2013. ACM. doi: 10. 1145/2487575.2487591. URL https://doi.org/ 10.1145/2487575.2487591.

Ninh Pham and Rasmus Pagh. Tensor sketch: Fast and scalable polynomial kernel approximation, 2025. URL https://arxiv.org/abs/2505.08146.

Rameshwar Pratap and Raghav Kulkarni. Variance reduction in frequency estimators via control variates method. In Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence (UAI), volume 161, pages 183–193, Corvallis, OR, 2021. AUAI Press. URL https://proceedings.mlr. press/v161/pratap21a.html.

Rameshwar Pratap, Bhisham Dev Verma, and Raghav Kulkarni. Improving Tug-of-War sketch using controlvariates method. In Proceedings ofthe 2021 SIAM Conference on Applied and Computational Discrete Algorithms (ACDA), pages 66–76, Philadelphia, PA, 2021. SIAM. doi: 10.1137/1.9781611976830.7.

Steffen Rendle. Factorization machines. In Proceedings of the 10th IEEE International Conference on Data Mining (ICDM), pages 995–1000, Los Alamitos, CA, 2010. IEEE Computer Society. doi: 10.1109/ICDM.2010.127. URL https://doi.org/10.1109/ICDM.2010.127.

Bernhard Scholkopf and Alexander J. Smola. Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond. MIT Press, Cambridge, MA, USA, 2001. ISBN 0262194759.

Yitong Sun, Anna Gilbert, and Ambuj Tewari. But how does it work in theory? linear svm with random features. Advances in Neural Information Processing Systems, 31, 2018.

Andrew Uzilov, Joshua Keegan, and David Mathews. Detection of non-coding rnas on the basis of predicted secondary structure formation free energy change. BMC bioinformatics, 7:173, 02 2006. doi: 10.1186/ 1471-2105-7-173.

Bhisham Dev Verma, Rameshwar Pratap, and Manoj Thakur. Variance reduction in feature hashing using MLE and control variate method. Mach. Learn., 111(7):2631–2662, 2022. doi: 10.1007/S10994-022-06166-Z. URL https: //doi.org/10.1007/s10994-022-06166-z.

Bhisham Dev Verma, Punit Pankaj Dubey, Rameshwar Pratap, and Manoj Thakur. Improving compressed matrix multiplication using control variate method. Inf. Process. Lett., 187:106517, 2025. doi: 10.1016/J.IPL.

2024.106517. URL https://doi.org/10.1016/ j.ipl.2024.106517.

Jonas Wacker, Ruben Ohana, and Maurizio Filippone. Complex-to-real sketches for tensor products with applications to the polynomial kernel. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS), volume 206, pages 5181–5212, Cambridge, MA, 2023. PMLR. URL https://proceedings.mlr. press/v206/wacker23a.html.

Jonas Wacker, Motonobu Kanagawa, and Maurizio Filippone. Improved random features for dot product kernels. J. Mach. Learn. Res., 25:235:1–235:75, 2024. URL https://jmlr.org/papers/v25/ 22-0118.html.

Gilad Yehudai and Ohad Shamir. On the power and limitations of random features for understanding neural networks. Advances in neural information processing systems, 32, 2019.

Amir Zandieh, Insu Han, Haim Avron, Neta Shoham, Chaewon Kim, and Jinwoo Shin. Scaling neural tangent kernels via sketching and random features. Advances in Neural Information Processing Systems, 34:1062–1073, 2021.

# Improving TensorSketch Using Complex Random Variables (Supplementary Material)

Amit Sharma<sup>\*1</sup> Mohammad Azhar Khan<sup>\*1</sup>

Rameshwar Pratap<sup>1</sup> Keegan Kang<sup>2</sup>

<sup>1</sup>Department of Computer Science and Engineering, IIT Hyderabad, India <sup>2</sup>Department of Mathematics and Statistics, Bucknell University, USA

## A PROOFS OF THEOREMS IN SECTION 4

Lemma 1. Let $\omega = e ^ { 2 \pi i / 4 } = i ,$ , and let $s : [ d ] \to \{ 1 , \omega , \omega ^ { 2 } , \omega ^ { 3 } \}$ be a randomfunction such that the values $\{ s ( i ) \} _ { i \in [ d ] }$ are drawn independently and uniformlyfrom thefourfourth roots ofunity. Then,for every $i \in [ d ]$ , the following identities hold

$$
\begin{array} { r } { \mathbb { E } \left[ s ( i ) \right] = 0 , } \\ { \mathbb { E } [ | s ( i ) | ^ { 2 } ] = 1 , } \\ { \mathbb { E } \left[ s ( i ) ^ { 2 } \right] = 0 . } \end{array}
$$

Furthermore,for any pair ofdistinct indices $i \neq j ,$ independence implies

$$
\begin{array} { r } { \mathbb { E } [ s ( i ) \overline { { s ( j ) } } ] = \mathbb { E } [ s ( i ) ] \mathbb { E } \left[ \overline { { s ( j ) } } \right] = 0 . } \end{array}
$$

Proof. Since $s ( i )$ is uniformly distributed over $\{ 1 , \omega , \omega ^ { 2 } , \omega ^ { 3 } \} = \{ 1 , i , - 1 , - i \}$ , we have

$$
\mathbb { E } \left[ s ( i ) \right] = \frac { 1 + i - 1 - i } { 4 } = 0 .
$$

All four values have unit magnitude, and therefore

$$
\mathbb { E } [ \left| s ( i ) \right| ^ { 2 } ] = \frac { \left| 1 \right| ^ { 2 } + \left| i \right| ^ { 2 } + \left| - 1 \right| ^ { 2 } + \left| - i \right| ^ { 2 } } { 4 } = 1 .
$$

For the second complex moment,

$$
\mathbb { E } \left[ s ( i ) ^ { 2 } \right] = { \frac { 1 ^ { 2 } + i ^ { 2 } + ( - 1 ) ^ { 2 } + ( - i ) ^ { 2 } } { 4 } } = { \frac { 1 - 1 + 1 - 1 } { 4 } } = 0 .
$$

Finally, for any $i \neq j ,$ , independence of $s ( i )$ and $s ( j )$ implies

$$
\begin{array} { r } { \mathbb { E } [ s ( i ) \overline { { s ( j ) } } ] = \mathbb { E } [ s ( i ) ] \mathbb { E } \left[ \overline { { s ( j ) } } \right] = 0 . } \end{array}
$$

## A.1 UNBIASEDNESS AND VARIANCE ANALYSIS OF CTR TENSORSKETCH

We establish the theoretical guarantees of CtR TensorSketch. Specifically, we show that the estimator is unbiased for the inner product estimation and subsequently provide a variance analysis.

Theorem 2. Let x, $\mathbf { y } \in \mathbb { R } ^ { d }$ and $\mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { p } }$ . Let $\Phi _ { \mathrm { C t R } }$ denote a Complex-to-Real TensorSketch as stated in Definition 8. Let $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) : = \Phi _ { \mathrm { C t R } } ( \mathbf { x } ^ { \otimes p } ) ^ { \top } \Phi _ { \mathrm { C t R } } ( \mathbf { y } ^ { \otimes p } )$ , then $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right)$ satisfies

$$
\begin{array} { r l r } & { } & { \mathbb { E } \left[ \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right] = \langle \mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \rangle = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } , } \\ & { } & { \mathrm { V a r } \left( \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right) \leq \frac { 2 ^ { p + 1 } - 2 } { D } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p } . } \end{array}
$$

Moreover, the sketches $\Phi _ { \mathrm { C t R } } ( \mathbf { x } ^ { \otimes p } )$ and $\Phi _ { \mathrm { C t R } } ( \mathbf { y } ^ { \otimes p } )$ can be computed in $O ( p ( \mathrm { n n z } \left( \mathbf { x } \right) + D \log D ) )$ and $O ( p ( { \mathrm { n n z } } \left( \mathbf { y } \right) + { }$ D log D)) time, respectively.

Proof. According to Definition 5, we have $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) : = \mathrm { R e } \left\{ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right\}$ , and therefore we first analyze $\widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right)$ . Recall that $\widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) : = \Phi _ { \mathrm { C } } ( \mathbf { x } ^ { \otimes p } ) \overline { { \Phi _ { \mathrm { C } } ( \mathbf { y } ^ { \otimes p } ) } } ^ { \top }$ , where $\Phi _ { \mathbf { C } } ( \mathbf { x } ^ { \otimes p } ) : = \mathbf { C } \mathbf { x } ^ { \otimes p }$ and $\Phi _ { \mathbf { C } } ( \mathbf { y } ^ { \otimes p } ) : = \mathbf { C } \mathbf { y } ^ { \otimes p }$ according to Definition 8. In particular, $\mathbf { C } \mathbf { x } ^ { \otimes p } , \mathbf { C } \mathbf { y } ^ { \otimes p } \in \mathbb { C } ^ { D / 2 }$ denote the complex CountSketch of the vectors $\mathbf { x } ^ { \tilde { \otimes } p } , \mathbf { y } ^ { \otimes p } \in \mathbb { R } ^ { d ^ { \tilde { p } } }$ , constructed using the derived functions $H : [ d ] ^ { p } \mapsto [ D / 2 ]$ and $S : [ \bar { d } ] ^ { p }  \{ 1 , \omega , \omega ^ { 2 } , \omega ^ { 3 } \}$ defined as

$$
\begin{array} { l } { { \displaystyle { \cal H } ( i _ { 1 } , \dots , i _ { p } ) = \left( \sum _ { j = 1 } ^ { p } h _ { j } ( i _ { j } ) \right) \bmod { \cal D } / 2 , } } \\ { { \displaystyle { \cal S } ( i _ { 1 } , \dots , i _ { p } ) = \prod _ { j = 1 } ^ { p } s _ { j } ( i _ { j } ) . } } \end{array}
$$

In the proof, we denote $X : = \mathbf { x } ^ { \otimes p }$ and $Y : = \mathbf { y } ^ { \otimes p }$ . Let $u , v \in [ d ] ^ { p }$ be the multi-indices corresponding to the entries of $X , Y \in \mathbb { R } ^ { d ^ { p } }$ . For a multi-index $u = ( j _ { 1 } , \ldots , j _ { p } )$ , the entry of X is defined as $\begin{array} { r } { X _ { u } = \prod _ { k = 1 } ^ { p } x _ { j _ { k } } } \end{array}$ , where the associated linearized index is $\begin{array} { r } { u = 1 + \sum _ { k = 1 } ^ { p } ( j _ { k } - 1 ) \prod _ { k ^ { \prime } = 1 } ^ { p - k } \partial _ { \bf k } } \end{array}$ . The entries of Y are defined analogously.

We begin by analyzing $\widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right)$ as defined below,

$$
\begin{array} { l } { \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) : = \Phi _ { \mathrm { C } } ( X ) \overline { { \Phi _ { \mathrm { C } } ( Y ) } } ^ { \top } = \left. \mathbf { C } X , \overline { { \mathbf { C } Y } } \right. , } \\ { = \displaystyle \sum _ { u , v \in [ d ] ^ { p } } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \mathbf { 1 } \left[ H ( u ) = H ( v ) \right] , \ } \\ { = \left. \left. X , Y \right. + \displaystyle \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \mathbf { 1 } \left[ H ( u ) = H ( v ) \right] . \right. } \end{array}
$$

Using Lemma 1, we have E $\left[ S ( u ) \overline { { S ( v ) } } \right] = 0$ for all $u \ne v$ . Hence,

$$
\begin{array} { r } { \mathbb { E } \left[ \widehat { k } _ { \mathrm { C } } ( \mathbf { x } , \mathbf { y } ) \right] = \langle X , Y \rangle = \langle \mathbf { x } ^ { \otimes p } , \mathbf { y } ^ { \otimes p } \rangle = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } . } \end{array}
$$

According to Definition 5, we know $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) : = \mathrm { R e } \left\{ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right\}$ , we have

$$
\begin{array} { r } { \mathbb { E } \left[ \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right] = \mathbb { E } \left[ \mathrm { R e } \left\{ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right\} \right] = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } . } \end{array}\tag{13}
$$

Using Equation 5, the variance of $\widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right)$ is

$$
\mathrm { V a r } \left( \widehat { k } _ { \mathrm { G t R } } \left( \mathbf { x } , \mathbf { y } \right) \right) = \frac { 1 } { 2 } \mathrm { R e } \Bigg \{ \mathbb { E } \left[ \left| \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right| ^ { 2 } \right] + \mathbb { E } \left[ \left( \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right) ^ { 2 } \right] - 2 \left( \mathbb { E } \left[ \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right] \right) ^ { 2 } \Bigg \} .\tag{14}
$$

For the variance analysis, we need to compute E $\left[ \left| \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right| ^ { 2 } \right] \mathrm { a n d } \mathbb { E } \left[ \left( \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) \right) ^ { 2 } \right]$

Therefore by expanding $| \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) | ^ { 2 }$ we get:

$$
\begin{array} { r l } & { \left| \widehat { k } _ { \mathrm { C } } ( \mathbf { x } , \mathbf { y } ) \right| ^ { 2 } : = | \langle \mathrm { C x } ^ { \otimes p } , \overline { { \mathrm { C y } ^ { \otimes p } } } \rangle | = \langle \mathrm { C x } ^ { \otimes p } , \overline { { \mathrm { C y } ^ { \otimes p } } } \rangle \langle \overline { { \mathrm { C x } ^ { \otimes p } } } , \mathrm { C y } ^ { \otimes p } , } \\ & { = \left( \langle X , Y \rangle + \displaystyle \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } { \bf 1 } \left[ H ( u ) = H ( v ) \right] \right) \left( \left. X , Y \right. + \displaystyle \sum _ { u \neq v } X _ { u } Y _ { v } \overline { { S ( u ) } } S ( v ) { \bf 1 } \left[ H ( u ) = H ( v ) \right] \right) , } \\ & { = \langle X , Y \rangle ^ { 2 } + \langle X , Y \rangle \left( \displaystyle \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } { \bf 1 } \left[ H ( u ) = H ( v ) \right] + \displaystyle \sum _ { u \neq v } X _ { v } Y _ { v } \overline { { S ( u ) } } S ( v ) { \bf 1 } \left[ H ( u ) = H ( v ) \right] \right) \cdots } \\ & { \qquad + \left| \left( \displaystyle \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } { \bf 1 } \left[ H ( u ) = H ( v ) \right] \right) \right| ^ { 2 } . } \end{array}\tag{15}
$$

(16)

(17)

Therefore, by applying expectation

$$
\begin{array} { r l } & { \mathbb { E } \Bigg [ \Bigg | \Bigg ( \sum _ { j \in \mathcal { N } _ { r } , \nabla _ { x } \mathcal { S } ( \mathrm { a b } , \mathcal { Y } _ { \theta } ^ { \epsilon } ) [ \mathcal { R } ( \mathrm { a } ) - \mathcal { R } ( \cdot ) ] } \Bigg ) ^ { * } \Bigg ] } \\ & { = \mathbb { E } \Bigg [ \Bigg | \sum _ { j \in \mathcal { N } _ { r } , \nabla _ { x } , \mathcal { Y } _ { \theta } , \mathcal { X } ( \mathrm { a b } , \cdot ) \in \mathcal { N } _ { r } ( \cdot ) \mathcal { R } ( \cdot ) \mathcal { E } ( \cdot ) \mathbb { H } ( \mathrm { a b } ) } \mathbb { E } ( \mathrm { a b } ) - m ( \cdot ) [ \Pi \mathcal { R } ( \mathrm { a b } ) - m ( \cdot ) ] \Bigg | \cdot } \\ &  = \sum _ { j \in \mathcal { N } _ { r } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } ( \cdot ) \mathcal { X } ( \cdot ) \mathcal { E } ( \cdot ) \mathcal { E } ( \cdot ) \mathcal { E } ( \cdot ) \mathcal { E } ( \cdot ) \mathcal { H } ( \mathrm { a b } ) - m ( \cdot ) [ \Pi \mathcal { R } ( \mathrm { a b } ) - m ( \cdot ) ] \Bigg | \cdot } \\ &  \xrightarrow [ \mathbb { K } ]  \sum _  j \in \mathcal { N } _ { r } , \mathcal { Y } _ { r } , \mathcal { Y } _ { r } , \mathcal { Y } _ { r } , \mathcal { Y } _ { r } , \mathcal { Y } _ { r } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal { Y } _ { \theta } , \mathcal \end{array}\tag{18}
$$

(19)

(20)

(21)

(22)

We now bound the above expression using the second-moment bound from Lemma 4, given in Equation (66). For completeness, we first restate the bound,

$$
\mathbb { E } \left[ \left| \left( \sum _ { u , v \in \lbrack d \vert ^ { p } } \vert X _ { u } \vert \vert Y _ { v } \vert S ( u ) \overline { { S ( v ) } } \right) \right| ^ { 2 } \right] = \left( \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } .\tag{23}
$$

Now, we expand the term $\begin{array} { r } { \left| \left( \sum _ { u , v \in [ d ] ^ { p } } | X _ { u } | | Y _ { v } | S ( u ) \overline { { S ( v ) } } \right) \right| ^ { 2 } } \end{array}$ from the above Equation (23) as follows,

(24)

$$
\begin{array} { r l } & { ( \displaystyle \sum _ { s = \infty } | X _ { n } | | X _ { n } | | S | u | \overline { { S ( s ) } } \overline { { S ( s ) } } ) ^ { n } \Bigg | ^ { 2 } = \displaystyle | \sum _ { s = 0 } | X _ { n } | | X _ { n } | + \displaystyle \sum _ { s = 0 } | X _ { n } | | Y _ { s } | | S | u | \overline { { S ( s ) } } \overline { { S ( s ) } } | ^ { 2 } , } \\ & { = \displaystyle ( \displaystyle \sum _ { s = 0 } | X _ { n } | | X _ { n } | + \displaystyle \sum _ { s = 0 } | X _ { n } | | X _ { n } | S | S | S \overline { { S ( s ) } } \overline { { S ( s ) } } ) ( \displaystyle \sum _ { s = 0 } | X _ { n } | | Y _ { s } | + \displaystyle \sum _ { s = 0 } | X _ { n } | | X _ { n } | | S | w | S \overline { { S ( s ) } } \overline { { S ( s ) } } ) , } \\ & { = \displaystyle ( \displaystyle \sum _ { s = 0 } | X _ { n } | | Y _ { s } | + \displaystyle \sum _ { s = 0 } \frac { | X _ { n } | | X _ { n } | | S | } { \displaystyle \sum _ { s = 0 } | X _ { n } | | Y _ { s } | | S | } ) ( \displaystyle \sum _ { s = 0 } | X _ { n } | | X _ { n } | | S _ { n } | + \displaystyle \sum _ { s = 0 } | X _ { n } | | X _ { n } | | S | w | S \overline { { S ( s ) } } \overline { { S ( s ) } } ) , } \\ & { = \displaystyle \sum _ { s = 0 } | X _ { n } | | X _ { n } | | X _ { n } | \displaystyle \sum _ { s = 0 } ^ { n } | X _ { n } | | X _ { n } | | S | S | S \overline { { S ( s ) } } \overline { { S ( s ) } } ) ( \displaystyle \sum _ { s = 0 } | X _ { n } | | X _ { n } | | S _ { n } | + \displaystyle \sum _ { s = 0 } | X _ { n } | | X _ { n } | | S | w | S | ) , } \\ \end{array}\tag{25}
$$

(26)

(27)

Using Lemma 1, we have E $\left[ S ( u ) \overline { { S ( v ) } } \right] = 0$ for all $u \ne v$ . Hence,

$$
\sum _ { \stackrel { u , v \in [ d ] ^ { p } } { u \neq v } } \sum _ { w \in [ d ] ^ { p } } \left| X _ { u } \right| \left| Y _ { v } \right| \left| X _ { w } \right| \left| Y _ { w } \right| \mathbb { E } \Big [ S ( u ) \overline { { S ( v ) } } \Big ] = 0 ,\tag{28}
$$

Similarly, for all w $\neq z ,$

$$
\sum _ { \stackrel { u \in [ d ] ^ { p } } { w \neq z } } \sum _ { \stackrel { w , z \in [ d ] ^ { p } } { w \neq z } } \left| X _ { u } \right| \left| Y _ { u } \right| \left| X _ { w } \right| \left| Y _ { z } \right| \mathbb { E } \left[ \overline { { S ( w ) } } S ( z ) \right] = 0 ,\tag{29}
$$

Substitute this in Equation (23), we get

$$
\begin{array} { r l } { \mathbb { E } \left[ \Bigg | \displaystyle \sum _ { u , v \in [ d ] ^ { p } } | X _ { u } | | Y _ { v } | S ( u ) \overline { { S ( v ) } } \Bigg | ^ { 2 } \right] = \displaystyle \sum _ { u , w \in [ d ] ^ { p } } | X _ { u } | | Y _ { u } | | X _ { w } | | Y _ { w } | + \cdots } & { } \\ & { \quad \cdots + \mathbb { E } \left[ \displaystyle \sum _ { u , v \in [ d ] ^ { p } } \displaystyle \sum _ { u , v \in [ d ] ^ { p } } | X _ { u } | | Y _ { v } | | X _ { w } | | Y _ { z } | S ( u ) \overline { { S ( v ) } } \overline { { S ( w ) } } S ( z ) \right] , } \\ & { = \langle X , Y \rangle ^ { 2 } + \mathbb { E } \left( \displaystyle \sum _ { u \neq v } | X _ { u } | | Y _ { v } | S ( u ) \overline { { S ( v ) } } \right) \Bigg | ^ { 2 } . } \end{array}\tag{30}
$$

(31)

We conclude that,

$$
\mathbb { E } \left[ \left| \sum _ { u \neq v } \left| X _ { u } \right| \left| Y _ { v } \right| S ( u ) \overline { { S ( v ) } } \right| ^ { 2 } \right] = \mathbb { E } \left[ \left| \sum _ { u , v \in \left[ d \right] ^ { p } } \left| X _ { u } \right| \left| Y _ { v } \right| S ( u ) \overline { { S ( v ) } } \right| ^ { 2 } \right] - \left. \mathbf { x } , \mathbf { y } \right. ^ { 2 p } .\tag{32}
$$

$$
\begin{array} { r l } & { \mathrm { P u t t i n g ~ v a l u e ~ o f ~ } \mathbb { E } \left[ \left| \left( \sum _ { u , v \in [ d ] ^ { p } } \left| X _ { u } \right| \left| Y _ { v } \right| S ( u ) \overline { { S ( v ) } } \right) \right| ^ { 2 } \right] , \mathrm { ~ w e ~ g e t ~ } } \\ & { \qquad \mathbb { E } \left[ \left| \displaystyle \sum _ { u \neq v } \left| X _ { u } \right| \left| Y _ { v } \right| S ( u ) \overline { { S ( v ) } } \right| ^ { 2 } \right] = \left( \left. \mathbf { x } , \mathbf { y } \right. ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } - \displaystyle \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } - \left. \mathbf { x } , \mathbf { y } \right. ^ { 2 p } . } \end{array}\tag{33}
$$

Put this value in Equation 22,

$$
\mathbb { E } \left[ \left| \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \mathbf { 1 } \left[ H ( u ) = H ( v ) \right] \right| ^ { 2 } \right] \leq \frac { 2 } { D } \left( \left( \left. \mathbf { x } , \mathbf { y } \right. ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } - \left. \mathbf { x } , \mathbf { y } \right. ^ { 2 p } \right) .\tag{34}
$$

Now we have the second moment as follows,

$$
\begin{array} { r l r } {  { \mathbb { E } [ | \widehat { k } _ { \mathrm { C } } ( \mathbf { x } , \mathbf { y } ) | ^ { 2 } ] =  X , Y  ^ { 2 } + \mathbb { E } [ ] ( \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \mathbf { 1 } [ H ( u ) = H ( v ) ] ) | ^ { 2 } ] , } } \\ & { } & { \leq  \mathbf { x } , \mathbf { y }  ^ { 2 p } + \frac { 2 } { D } ( (  \mathbf { x } , \mathbf { y }  ^ { 2 } +  \mathbf { x }  _ { 2 } ^ { 2 }  \mathbf { y }  _ { 2 } ^ { 2 } - \displaystyle \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } ) ^ { p } -  \mathbf { x } , \mathbf { y }  ^ { 2 p } ) . } \end{array}\tag{35}
$$

(36)

Next, we analyze E $\left[ ( \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) ) ^ { 2 } \right]$ . Expanding $( \widehat { k } _ { \mathrm { C } } \left( \mathbf { x } , \mathbf { y } \right) ) ^ { 2 }$ gives:

(37)

$$
\begin{array} { l } { { \displaystyle ( \widehat { k } _ { \mathrm { C } } ( { \bf x } , { \bf y } ) ) ^ { 2 } : = \left( \langle { \mathrm { C x } } ^ { \otimes p } , \overline { { { \mathrm { C y } } ^ { \otimes p } } } \rangle \right) ^ { 2 } = \langle { \mathrm { C x } } ^ { \otimes p } , \overline { { { \mathrm { C y } } ^ { \otimes p } } } \rangle \langle { \mathrm { C x } } ^ { \otimes p } , \overline { { { \mathrm { C y } } ^ { \otimes p } } } \rangle , } } \\ { { \displaystyle = \left( \langle X , Y \rangle + \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } { \bf 1 } \left[ H ( u ) = H ( v ) \right] \right) \left( \langle X , Y \rangle + \sum _ { u \neq v } X _ { v } S ( u ) \overline { { S ( v ) } } { \bf 1 } \left[ H ( u ) = H ( v ) \right] \right) , } } \\ { { \displaystyle = \langle X , Y \rangle ^ { 2 } + 2 \langle X , Y \rangle \left( \sum _ { u \neq v } X _ { v } S ( u ) \overline { { S ( v ) } } { \bf 1 } \left[ H ( u ) = H ( v ) \right] \right) + \left( \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } { \bf 1 } \left[ H ( u ) = H ( v ) \right] \right) ^ { 2 } } } \end{array}\tag{38}
$$

(39)

Therefore, by applying expectation

(40)

$$
\begin{array} { r l } & { \mathbb { E } \Bigg [ \Bigg ( \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \mathbf { 1 } [ H ( u ) = H ( v ) ] \Bigg ) ^ { 2 } \Bigg ] } \\ & { = \mathbb { E } [ \sum _ { u \neq v } X _ { u _ { 1 } } Y _ { v _ { 1 } } X _ { u _ { 2 } } Y _ { v _ { 2 } } S ( u _ { 1 } ) \overline { { S ( v _ { 1 } ) } } S ( u _ { 2 } ) \overline { { S ( v _ { 2 } ) } } \mathbf { 1 } [ H ( u _ { 1 } ) = H ( v _ { 1 } ) ] \mathbf { 1 } [ H ( u _ { 2 } ) = H ( v _ { 2 } ) ] ] , } \\ & { = \sum _ { u \neq v } \mathbb { E } [ X _ { u _ { 1 } } Y _ { v _ { 1 } } X _ { u _ { 2 } } Y _ { v _ { 2 } } S ( u _ { 1 } ) \overline { { S ( v _ { 1 } ) } } S ( u _ { 2 } ) \overline { { S ( v _ { 2 } ) } } ] \cdot \mathbb { E } [ \mathbf { 1 } [ H ( u _ { 1 } ) = H ( v _ { 1 } ) ] \mathbf { 1 } [ H ( u _ { 2 } ) = H ( v _ { 2 } ) ] ] , } \\ & { \quad u _ { 2 } \neq v _ { 2 } } \end{array}\tag{41}
$$

(42)

$$
\leq \frac { 2 } { D } \mathbb { E } \left[ \left( \sum _ { u \neq v \in [ d ] ^ { p } } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \right) ^ { 2 } \right] .\tag{43}
$$

By similar analysis and using second-moment bound of Lemma 4 given in Equation (75), we get

$$
\mathbb { E } \left[ \left( \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \mathbf { 1 } \left[ H ( u ) = H ( v ) \right] \right) ^ { 2 } \right] \leq \frac { 2 } { D } \left( \left( 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } - \left. \mathbf { x } , \mathbf { y } \right. ^ { 2 p } \right) .\tag{44}
$$

Therefore, we have,

$$
\begin{array} { r l r } {  { \mathbb { E } [ ( \widehat { k } _ { \mathrm { C } } ( \mathbf { x } , \mathbf { y } ) ) ^ { 2 } ] =  X , Y  ^ { 2 } + \mathbb { E } [ ( \sum _ { u \neq v } X _ { u } Y _ { v } S ( u ) \overline { { S ( v ) } } \mathbf { 1 } [ H ( u ) = H ( v ) ] ) ^ { 2 } ] , } } \\ & { } & { \leq  \mathbf { x } , \mathbf { y }  ^ { 2 p } + \frac { 2 } { D } ( ( 2  \mathbf { x } , \mathbf { y }  ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } ) ^ { p } -  \mathbf { x } , \mathbf { y }  ^ { 2 p } ) . } \end{array}\tag{45}
$$

(46)

Therefore for final variance we put Equation (36) and Equation (46) in Equation (14),we get

$$
\begin{array} { r l r } & { } & { { \mathrm { V a r } } \left( \widehat { k } _ { \mathrm { C t R } } \left( \mathbf { x } , \mathbf { y } \right) \right) \leq \displaystyle \frac { 1 } { 2 } \Bigg [ \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 p } + \displaystyle \frac { 2 } { D } \Big ( \big ( \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } - \displaystyle \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \big ) ^ { p } - \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 p } \Big ) } \\ & { } & { + \left. \mathbf { x } , \mathbf { y } \right. ^ { 2 p } + \displaystyle \frac { 2 } { D } \Big ( \big ( 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \displaystyle \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \big ) ^ { p } - \left. \mathbf { x } , \mathbf { y } \right. ^ { 2 p } \Big ) - 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 p } \Bigg ] , } \end{array}\tag{47}
$$

we can upper bound the above equation by using inequality $\begin{array} { r } { \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } \leq \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } } \end{array}$ , then

$$
\leq \frac { 1 } { 2 } \left[ \frac { 2 } { D } \left( \left( 2 \| \mathbf { x } \| _ { 2 } ^ { 2 } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 } \right) ^ { p } - \| \mathbf { x } \| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p } \right) + \frac { 2 } { D } \left( \left( 2 \| \mathbf { x } \| _ { 2 } ^ { 2 } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 } \right) ^ { p } - \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p } \right) \right] ,\tag{48}
$$

$$
\leq \frac { 1 } { 2 } \left[ \frac { 2 ^ { p + 1 } - 2 } { D } \| { \bf x } \| _ { 2 } ^ { 2 p } \| { \bf y } \| _ { 2 } ^ { 2 p } + \frac { 2 ^ { p + 1 } - 2 } { D } \| { \bf x } \| _ { 2 } ^ { 2 p } \| { \bf y } \| _ { 2 } ^ { 2 p } \right] ,\tag{49}
$$

$$
\leq \frac { 2 ^ { p + 1 } - 2 } { D } \| \mathbf { x } \| _ { 2 } ^ { 2 p } \| \mathbf { y } \| _ { 2 } ^ { 2 p } .\tag{50}
$$

We begin by stating a lemma that serves as a key step to bound the second moments of the estimator in the proof of Theorem 2 .

Lemma 4. Let $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d } $ , let $p \geq 1$ be an integer, and let $s _ { 1 } , \ldots , s _ { p } : [ d ] \to \{ 1 , \omega , \omega ^ { 2 } , \omega ^ { 3 } \}$ be independent random functions, each taking values uniformly from the four fourth roots of unity (as in Lemma 1). Define

$$
Z = \prod _ { j = 1 } ^ { p } Z _ { s _ { j } } ( { \bf x } ) \overline { { { Z _ { s _ { j } } ( { \bf y } ) } } } ,
$$

where

$$
Z _ { s _ { j } } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { d } x _ { i } s _ { j } ( i ) ,
$$

$$
Z _ { s _ { j } } ( \mathbf { y } ) = \sum _ { i = 1 } ^ { d } y _ { i } s _ { j } ( i ) .
$$

Then,

$$
\mathbb { E } \left[ Z \right] = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } ,
$$

$$
\operatorname { V a r } \left( Z \right) \leq 2 ^ { p } \left\| \mathbf { x } \right\| _ { 2 } ^ { 2 p } \left\| \mathbf { y } \right\| _ { 2 } ^ { 2 p } .
$$

Proof. First, we consider the expectation. For each $j ,$ we note that

$$
\mathbb { E } \Big [ Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } \Big ] = \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { d } { x _ { i } s _ { j } ( i ) } \right) \left( \sum _ { k = 1 } ^ { d } { y _ { k } \overline { { s _ { j } ( k ) } } } \right) \right] ,\tag{51}
$$

$$
= \sum _ { i = 1 } ^ { d } \sum _ { k = 1 } ^ { d } x _ { i } y _ { k } \operatorname { \mathbb { E } } [ s _ { j } ( i ) { \overline { { s _ { j } ( k ) } } } ] ,\tag{52}
$$

$$
= \sum _ { i = 1 } ^ { d } { x _ { i } y _ { i } \mathbb { E } [ | s _ { j } ( i ) | ^ { 2 } ] } + \sum _ { i \ne k } { x _ { i } y _ { k } \mathbb { E } [ s _ { j } ( i ) \overline { { s _ { j } ( k ) } } ] } ,\tag{53}
$$

(54)

where, $\mathbb { E } [ s _ { j } ( i ) { \overline { { s _ { j } } } } ( k ) ] = 0 , \forall i \neq$ k and $\mathbb { E } [ | s _ { j } ( i ) | ^ { 2 } ] = 1 , \forall i \in [ d ]$ as given in Lemma 1.

Since the functions $s _ { j }$ are independent across different $j ,$ we have

$$
\mathbb { E } [ Z ] = \prod _ { j = 1 } ^ { p } \mathbb { E } [ Z _ { s _ { j } } ( \mathbf { x } ) { \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } } ] = \langle \mathbf { x } , \mathbf { y } \rangle ^ { p } .\tag{55}
$$

Next, to bound the variance,

$$
\operatorname { V a r } { ( Z ) } = { \frac { 1 } { 2 } } \operatorname { R e } { \big \{ } \mathbb { E } [ | Z | ^ { 2 } ] + \mathbb { E } [ ( Z ) ^ { 2 } ] - 2 | \mathbb { E } [ Z ] | ^ { 2 } { \big \} } .\tag{56}
$$

Because the random hash functions $s _ { j }$ are independent across different $j ,$ , we may write

$$
\mathbb { E } [ | Z | ^ { 2 } ] = \prod _ { j = 1 } ^ { p } \mathbb { E } \left[ | \left( Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } \right) | ^ { 2 } \right] \mathrm { ~ a n d ~ } \mathbb { E } [ ( Z ) ^ { 2 } ] = \prod _ { j = 1 } ^ { p } \mathbb { E } \left[ \left( Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } \right) ^ { 2 } \right]\tag{57}
$$

For each $j ,$ expanding the square gives

$$
\mathbb { E } \left[ | \left( Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } \right) | ^ { 2 } \right] = \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { d } { x _ { i } s _ { j } ( i ) } \right) \left( \sum _ { k = 1 } ^ { d } { y _ { k } \overline { { s _ { j } ( k ) } } } \right) \left( \sum _ { i = 1 } ^ { d } { x _ { i } \overline { { s _ { j } ( i ) } } } \right) \left( \sum _ { k = 1 } ^ { d } { y _ { k } s _ { j } ( k ) } \right) \right] ,\tag{58}
$$

$$
= \sum _ { i = 1 } ^ { d } \sum _ { i ^ { \prime } = 1 } ^ { d } \sum _ { k = 1 } ^ { d } \sum _ { k ^ { \prime } = 1 } ^ { d } x _ { i } x _ { i ^ { \prime } } y _ { k } y _ { k ^ { \prime } } \operatorname { \mathbb { E } } \left[ s _ { j } ( i ) { \overline { { s _ { j } ( k ) s _ { j } ( i ^ { \prime } ) } } } s _ { j } ( k ^ { \prime } ) \right] .\tag{59}
$$

Observing that $\mathbb { E } [ s _ { j } ( i ) \overline { { s _ { j } ( k ) s _ { j } ( i ^ { \prime } ) } } s _ { j } ( k ^ { \prime } ) ]$ is nonzero only when the indices form pairs (including the possibility that all four are identical), we have

$$
\mathbb { E } [ s _ { j } ( i ) \overline { { s _ { j } ( k ) s _ { j } ( i ^ { \prime } ) } } s _ { j } ( k ^ { \prime } ) ] = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } i = k = i ^ { \prime } = k ^ { \prime } , } \\ { 1 , } & { \mathrm { i f } i = k \neq i ^ { \prime } = k ^ { \prime } , } \\ { 1 , } & { \mathrm { i f } i = i ^ { \prime } \neq k = k ^ { \prime } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{60}
$$

The contribution from terms with $i = k = i ^ { \prime } = k ^ { \prime }$ is

$$
\sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } .\tag{61}
$$

Terms with $i = k \neq i ^ { \prime } = k ^ { \prime }$ contribute

$$
\sum _ { i \ne i ^ { \prime } } x _ { i } y _ { i } x _ { i ^ { \prime } } y _ { i ^ { \prime } } = \left( \sum _ { i = 1 } ^ { d } x _ { i } y _ { i } \right) ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } = \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } .\tag{62}
$$

Finally, for $i = i ^ { \prime } \neq k = k ^ { \prime }$ we obtain

$$
\sum _ { i \neq k } x _ { i } ^ { 2 } y _ { k } ^ { 2 } = \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } .\tag{63}
$$

Thus, summing these contributions, we have

$$
\mathbb { E } \big [ | \left( Z _ { s _ { j } } ( \mathbf { x } ) Z _ { s _ { j } } ( \mathbf { y } ) \right) | ^ { 2 } \big ] = \sum _ { i = 1 } ^ { d } { x _ { i } ^ { 2 } y _ { i } ^ { 2 } } + \left( \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } - 2 \sum _ { i = 1 } ^ { d } { x _ { i } ^ { 2 } y _ { i } ^ { 2 } } \right) ,\tag{64}
$$

$$
= \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } .\tag{65}
$$

Substituting this bound into Equation (57) yields

$$
\mathbb { E } [ | Z | ^ { 2 } ] = \left( \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } ,\tag{66}
$$

For each $j ,$ expanding the square gives

$$
\begin{array} { r l } & { \mathbb { E } \bigg [ \Big ( Z _ { s _ { j } } ( \mathbf { x } ) \overline { { Z _ { s _ { j } } ( \mathbf { y } ) } } \Big ) ^ { 2 } \bigg ] = \mathbb { E } \bigg [ \bigg ( \displaystyle \sum _ { i = 1 } ^ { d } x _ { i } s _ { j } ( i ) \bigg ) \left( \displaystyle \sum _ { k = 1 } ^ { d } y _ { k } \overline { { s _ { j } ( k ) } } \right) \left( \displaystyle \sum _ { i = 1 } ^ { d } x _ { i } s _ { j } ( i ) \right) \left( \displaystyle \sum _ { k = 1 } ^ { d } y _ { k } \overline { { s _ { j } ( k ) } } \right) \bigg ] , } \\ & { \quad \quad \quad = \displaystyle \sum _ { i = 1 } ^ { d } \displaystyle \sum _ { i ^ { \prime } = 1 } ^ { d } \displaystyle \sum _ { k = 1 } ^ { d } \sum _ { k ^ { \prime } = 1 } ^ { d } x _ { i } x _ { i ^ { \prime } } y _ { k } y _ { k ^ { \prime } } \mathbb { E } \Big [ s _ { j } ( i ) \overline { { s _ { j } ( k ) } } s _ { j } ( i ^ { \prime } ) \overline { { s _ { j } ( k ^ { \prime } ) } } \Big ] . } \end{array}\tag{67}
$$

(68)

Observing that $\mathbb { E } [ s _ { j } ( i ) \overline { { s _ { j } ( k ) } } s _ { j } ( i ^ { \prime } ) \overline { { s _ { j } ( k ^ { \prime } ) } } ]$ is nonzero only when the indices form pairs (including the possibility that all four are identical), we have

$$
\mathbb { E } [ s _ { j } ( i ) \overline { { s _ { j } ( k ) } } s _ { j } ( i ^ { \prime } ) \overline { { s _ { j } ( k ^ { \prime } ) } } ] = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } i = k = i ^ { \prime } = k ^ { \prime } , } \\ { 1 , } & { \mathrm { i f ~ } i = k \neq i ^ { \prime } = k ^ { \prime } , } \\ { 1 , } & { \mathrm { i f ~ } i = k ^ { \prime } \neq i ^ { \prime } = k , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{69}
$$

The contribution from terms with $i = k = i ^ { \prime } = k ^ { \prime }$ is

$$
\sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } .\tag{70}
$$

Terms with $i = k \neq i ^ { \prime } = k ^ { \prime }$ contribute

$$
\sum _ { i \ne i ^ { \prime } } x _ { i } y _ { i } x _ { i ^ { \prime } } y _ { i ^ { \prime } } = \left( \sum _ { i = 1 } ^ { d } x _ { i } y _ { i } \right) ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } = \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } .\tag{71}
$$

Finally, for $i = k ^ { \prime } \neq i ^ { \prime } = k$ we obtain

$$
\sum _ { i \ne i ^ { \prime } } x _ { i } y _ { i } x _ { i ^ { \prime } } y _ { i ^ { \prime } } = \left( \sum _ { i = 1 } ^ { d } x _ { i } y _ { i } \right) ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } = \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } .\tag{72}
$$

Thus, summing these contributions, we have

$$
{ \mathbb E } \Big [ \big ( Z _ { s _ { j } } ( \mathbf { x } ) Z _ { s _ { j } } ( \mathbf { y } ) \big ) ^ { 2 } \Big ] = \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } + \left( 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - 2 \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ,\tag{73}
$$

$$
= 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } .\tag{74}
$$

Substituting this bound into Equation (57) yields

$$
{ \mathbb E } [ ( Z ) ^ { 2 } ] = \left( 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } ,\tag{75}
$$

which completes the proof since

$$
\begin{array} { l } { \displaystyle \mathrm { V a r } \left( Z \right) = \frac { 1 } { 2 } \operatorname { R e } \bigl \{ \mathbb { E } [ | Z | ^ { 2 } ] + \mathbb { E } [ ( Z ) ^ { 2 } ] - 2 | \mathbb { E } [ Z ] | ^ { 2 } \bigr \} , } \\ { \displaystyle \qquad = \frac { 1 } { 2 } \left\{ \left( \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } + \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } + \left( 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) ^ { p } - 2 \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 p } \right\} . } \end{array}\tag{76}
$$

(77)

Using the Cauchy–Schwarz inequality, $\langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } \leq \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 }$ , and noting that $\begin{array} { r } { \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \ge 0 } \end{array}$ , it follows that

$$
\operatorname { V a r } \left( Z \right) \leq 2 ^ { p } \| \mathbf { x } \| _ { 2 } ^ { 2 p } \| \mathbf { y } \| _ { 2 } ^ { 2 p } .\tag{78}
$$

## B PROOF OF COUNTSKETCH ESTIMATOR WITH FOURTH ROOTS OF UNITY

Theorem 5. Let Φ denote the CountSketch with the hash function taking values in the fourth roots of unity. For any vectors $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { d }$ , the corresponding sketch vector is defined by

$$
\Phi ( { \bf x } ) _ { j } = \sum _ { i = 1 } ^ { d } \delta _ { j i } \sigma ( i ) x _ { i } , \qquad \Phi ( { \bf y } ) _ { j } = \sum _ { i = 1 } ^ { d } \delta _ { j i } \sigma ( i ) y _ { i } , \qquad \forall j \in [ D / 2 ] .\tag{79}
$$

where,

$h : [ d ]  [ D / 2 ]$ assigning each coordinate independently and uniformly to one of $D / 2$ buckets, and

$\sigma : [ d ]  \{ 1 , \omega , \omega ^ { 2 } , \omega ^ { 3 } \}$ are drawn independently and uniformly at random,

• δ<sub>ji</sub> = 1, if h(i) = j, 0, otherwise.

then the estimator $\hat { k } ( \mathbf x , \mathbf y ) : = \mathrm { R e } \left\{ \Phi ( \mathbf x ) ^ { \top } \overline { { \Phi ( \mathbf y ) } } \right\}$ , satisfies

$$
\mathbb { E } \Big [ \hat { k } ( \mathbf { x } , \mathbf { y } ) \Big ] = \langle \mathbf { x } , \mathbf { y } \rangle ,\tag{80}
$$

$$
\mathrm { V a r } \big ( \hat { k } ( \mathbf { x } , \mathbf { y } ) \big ) = \frac { 1 } { D } \left( \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } + \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - 2 \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) .\tag{81}
$$

Proof. Since $\sigma ( i )$ is uniformly distributed over $\{ 1 , \omega , \omega ^ { 2 } , \omega ^ { 3 } \} = \{ 1 , i , - 1 , - i \}$ and satisfies $\mathbb { E } [ \sigma ( i ) ] = 0 , \mathbb { E } \big [ | \sigma ( i ) | ^ { 2 } \big ] =$ $1 , \mathbb { E } \big [ ( \sigma ( i ) ) ^ { 2 } \big ] = 0$ . Consider now the complex inner product of the sketch:

$$
\hat { k } _ { C } ( \mathbf { x } , \mathbf { y } ) : = \Phi ( \mathbf { x } ) ^ { \top } \overline { { \Phi ( \mathbf { y } ) } } = \sum _ { j = 1 } ^ { D / 2 } \Big ( \sum _ { \substack { i : h ( i ) = j } } \sigma ( i ) x _ { i } \Big ) \Big ( \sum _ { \substack { m : h ( m ) = j } } \overline { { \sigma ( m ) } } y _ { m } \Big )\tag{82}
$$

$$
= \sum _ { i = 1 } ^ { d } \vert \sigma ( i ) \vert ^ { 2 } x _ { i } y _ { i } + \sum _ { i \ne m } \mathbf { 1 } \{ h ( i ) = h ( m ) \} \sigma ( i ) \overline { { \sigma ( m ) } } x _ { i } y _ { m } .\tag{83}
$$

Taking expectation with respect to h and σ, we get

$$
\mathbb { E } \Big [ \hat { k } _ { C } ( \mathbf { x } , \mathbf { y } ) \Big ] = \sum _ { i = 1 } ^ { d } \mathbb { E } \big [ | \sigma ( i ) | ^ { 2 } \big ] x _ { i } y _ { i } = \sum _ { i = 1 } ^ { d } x _ { i } y _ { i } = \langle \mathbf { x } , \mathbf { y } \rangle .\tag{84}
$$

Since the right-hand side of (84) is real, the same unbiasedness holds for the estimator, which takes the real part of the complex sketch:

$$
\mathbb { E } \Big [ \hat { k } ( \mathbf { x } , \mathbf { y } ) \Big ] = \mathbb { E } \left[ \mathrm { R e } \left\{ \hat { k } _ { C } ( \mathbf { x } , \mathbf { y } ) \right\} \right] = \mathrm { R e } \left\{ \mathbb { E } \left[ \hat { k } _ { C } ( \mathbf { x } , \mathbf { y } ) \right] \right\} = \langle \mathbf { x } , \mathbf { y } \rangle .\tag{85}
$$

To find the exact variance of the estimator, we leverage the Complex-to-Real (CtR) framework stated in our paper. Now, we work out with E $\left[ | \hat { k } _ { C } ( \mathbf { x } , \mathbf { y } ) | ^ { 2 } \right]$

$$
\mathbb { E } \left[ \left| \hat { k } _ { C } ( \mathbf { x } , \mathbf { y } ) \right| ^ { 2 } \right] = \mathbb { E } \left[ \left| \Phi ( \mathbf { x } ) ^ { \top } \overline { { \Phi ( \mathbf { y } ) } } \right| ^ { 2 } \right] = \mathbb { E } \left[ \left| \sum _ { j = 1 } ^ { D / 2 } \left( \sum _ { i = 1 } ^ { d } \delta _ { j i } \sigma ( i ) x _ { i } \right) \left( \sum _ { m = 1 } ^ { d } \delta _ { j m } \overline { { \sigma ( m ) } } y _ { m } \right) \right| ^ { 2 } \right] ,\tag{86}
$$

$$
= \sum _ { j , j ^ { \prime } } ^ { D / 2 } \sum _ { i , m , i ^ { \prime } , m ^ { \prime } } ^ { d } \mathbb { E } \big [ \delta _ { j i } \delta _ { j m } \delta _ { j ^ { \prime } i } \delta _ { j ^ { \prime } m ^ { \prime } } \big ] \ \mathbb { E } \big [ \sigma ( i ) \overline { { \sigma ( m ) } } \overline { { \sigma ( i ^ { \prime } ) } } \sigma ( m ^ { \prime } ) \big ] \ x _ { i } y _ { m } x _ { i ^ { \prime } } y _ { m ^ { \prime } } .\tag{87}
$$

Hence,

$$
\mathbb { E } \big [ \sigma ( i ) \overline { { \sigma ( m ) } } \overline { { \sigma ( i ^ { \prime } ) } } \sigma ( m ^ { \prime } ) \big ] \neq 0 \iff \{ i , m , i ^ { \prime } , m ^ { \prime } \} \mathrm { ~ a p p e a r ~ i n ~ p a i r s } ,
$$

1. With surviving index configurations, when $\mathbf { j } = \mathbf { j } ^ { \prime } \mathrm { : }$

$$
\bullet \ \mathbf { i } = \mathbf { m } = \mathbf { i } ^ { \prime } = \mathbf { m } ^ { \prime } : \mathbb { E } [ \delta _ { j i } ] \mathbb { E } [ | \sigma ( i ) | ^ { 4 } ] x _ { i } ^ { 2 } y _ { i } ^ { 2 } ,
$$

• i = m ̸= i<sup>′</sup> = m<sup>′</sup> : E[δ<sub>ji</sub>δ<sub>ji</sub>′ ]E[|σ(i)|<sup>2</sup>|σ(i<sup>′</sup>)|<sup>2</sup>]x<sub>i</sub>y<sub>i</sub>x<sub>i</sub>′ y<sub>i</sub>′ ,

• i = i<sup>′</sup> ̸= m = m<sup>′</sup> : E[δ<sub>ji</sub>δ<sub>jm</sub>]E[|σ(i)|<sup>2</sup>|σ(m)|<sup>2</sup>]x<sup>2</sup><sub>i</sub> y<sup>2</sup><sub>m</sub>,

2. With surviving index configurations, when $\mathbf { j } \neq \mathbf { j } ^ { \prime } \colon$

$$
\bullet \ \mathbf { i } = \mathbf { m } \neq \mathbf { i } ^ { \prime } = \mathbf { m } ^ { \prime } : \mathbb { E } [ \delta _ { j i } \delta _ { j ^ { \prime } i ^ { \prime } } ] \mathbb { E } [ | \sigma ( i ) | ^ { 2 } | \sigma ( i ^ { \prime } ) | ^ { 2 } ] x _ { i } y _ { i } x _ { i ^ { \prime } } y _ { i ^ { \prime } } ,
$$

Thus,

$$
\begin{array} { l } { { \displaystyle { \mathbb E } \Big [ | \hat { k } _ { C } ( { \bf x } , { \bf y } ) | ^ { 2 } \Big ] = \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } + \frac { 2 } { D } \sum _ { i \neq i ^ { \prime } } ( x _ { i } y _ { i } x _ { i ^ { \prime } } y _ { i ^ { \prime } } + x _ { i } ^ { 2 } y _ { i ^ { \prime } } ^ { 2 } ) + \frac { D - 2 } { D } \sum _ { i \neq i ^ { \prime } } x _ { i } y _ { i } x _ { i ^ { \prime } } y _ { i ^ { \prime } } } , }  \\ { { \displaystyle \quad \quad = \langle { \bf x } , { \bf y } \rangle ^ { 2 } + \frac { 2 } { D } \sum _ { i \neq i ^ { \prime } } x _ { i } ^ { 2 } y _ { i ^ { \prime } } ^ { 2 } . } } \end{array}\tag{88}
$$

(89)

Now, similarly we get

$$
\begin{array} { l } { \displaystyle \mathbb { E } \Big [ ( \hat { k } _ { C } ( { \bf x } , { \bf y } ) ) ^ { 2 } \Big ] = \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } + \frac { 4 } { D } \sum _ { i \neq i ^ { \prime } } ( x _ { i } y _ { i } x _ { i ^ { \prime } } y _ { i ^ { \prime } } ) + \frac { D - 2 } { D } \sum _ { i \neq i ^ { \prime } } x _ { i } y _ { i } x _ { i ^ { \prime } } y _ { i ^ { \prime } } , } \\ { = \langle { \bf x } , { \bf y } \rangle ^ { 2 } + \frac { 2 } { D } \sum _ { i \neq i ^ { \prime } } ( x _ { i } y _ { i } x _ { i ^ { \prime } } y _ { i ^ { \prime } } ) . } \end{array}\tag{90}
$$

(91)

Now, substitute values of E $\Big [ | \hat { k } _ { C } ( { \bf x } , { \bf y } ) | ^ { 2 } \Big ]$ and E $\left[ ( \hat { k } _ { C } ( \mathbf { x } , \mathbf { y } ) ) ^ { 2 } \right]$ and simplifying, we get

$$
\mathrm { V a r } \big ( \hat { k } ( \mathbf { x } , \mathbf { y } ) \big ) = \frac { 1 } { D } \left( \| \mathbf { x } \| _ { 2 } ^ { 2 } \| \mathbf { y } \| _ { 2 } ^ { 2 } + \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } - 2 \sum _ { i = 1 } ^ { d } x _ { i } ^ { 2 } y _ { i } ^ { 2 } \right) .\tag{92}
$$

Hence, from the above variance bound, it follows that CountSketch with complex random variables does not provide any variance reduction over its real-valued counterpart. □

## C FURTHER EXPERIMENTS

## C.1 EXTENDED EVALUATION ON VARIANCE AND TIME IN DIFFERENT SETUPS

In this section, we present additional experimental results that complement the main empirical evaluation reported in Section 5. These experiments extend the comparison between real and complex-to-real (CtR) sketching constructions across additional datasets, higher polynomial degrees, and varied embedding dimension. We report both approximation quality (via KL divergence between exact and sketched kernels) and wall-clock sketch construction time. Together, these results provide a more detailed view of the variance and construction time trade-offs of CtR TensorSketch relative to its real-valued counterparts and JL-type baselines.

![](images/5b6895d1e6ef6aa9051332dd8c24eca7a9c4f363c966aa2b8681aea31af7d704.jpg)  
Figure 3: KL divergence on the MAGIC Gamma Telescope dataset. We compare Real and complex-to-real (CtR) Gaussian and Rademacher JL sketches, together with Real and CtR TensorSketch. Results are shown for polynomial degrees $p \in \{ 3 , 5 , 7 \}$ and sketch dimensions $D \in \{ d , 3 d , 5 d \}$ with $n = 3 0 0 0$ standardized and ℓ -normalized samples. Bars report the mean KL divergence over 20 independent trials.

![](images/19db135910a415f6055e8ed68dcdc78bfad457c66b51ce6506e6d744bc640faa.jpg)  
Figure 4: Wall-clock sketch construction time on the MAGIC dataset. Methods compared include Real and complexto-real (CtR) Gaussian and Rademacher JL sketches, with Real and CtR TensorSketch. Results are shown for polynomial degrees $p \in \{ 3 , 5 , 7 \}$ and sketch dimensions $D \in \{ d , 3 d , 5 d \}$ with $n = 3 0 0 0$ standardized and $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ samples. Each point reports the mean runtime over 20 independent trials for feature sketch construction only.

![](images/3f07415dfcd73736afa4a1ad3697d1468424d62cc2fcf0008e30eb64c0a9a3e0.jpg)  
Figure 5: KL divergence on the synthetic dataset $( d = 2 )$ . We compare Real and complex-to-real (CtR) Gaussian and Rademacher JL sketches, together with Real and CtR TensorSketch, for approximating degree-p polynomial kernels on synthetic Gaussian data $( n = 3 0 0 0$ , dimension $d = 2 .$ , standardized and $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } } )$ . Results are shown for polynomial degrees $p \in \{ 1 0 , 1 5 , 2 0 \}$ and sketch dimensions $D \in \{ 1 0 0 , 3 0 0 , 5 0 0 \}$ . Bars report the mean KL divergence between the exact kernel and the sketch-based approximation over 20 independent trials.

![](images/77ec8df90dc08b28cd6b71982791b89006d16eb0f925810ac2e031c774eefc7f.jpg)  
Figure 6: Wall-clock sketch construction time on a synthetic dataset. Methods compared include Real and complex-to-real (CtR) Gaussian and Rademacher JL sketches, together with Real and CtR TensorSketch. Results are shown for polynomial degrees $p \in \{ 1 0 , 1 5 , 2 0 \}$ and sketch dimensions $D \in \{ 1 0 0 , 3 0 0 , 5 0 0 \}$ with $n = 3 0 0 0$ standardized and $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ samples in dimension $d = 2 .$ Each point reports the mean runtime over 20 independent trials for feature sketch construction only (log-scale on the y-axis).

## C.2 EVALUATION ON FROBENIUS NORMALIZED RELATIVE ERROR

We also test our proposed method along with other baselines using the Frobenius normalized relative error, which is defined as $\| K - { \hat { K } } \| _ { F } / \| K \| _ { F }$ . In this formula, K represents the exact kernel matrix, while K<sup>ˆ</sup> represents the approximated kernel matrix produced by the sketching method. Frobenius error metric is a standard benchmark in kernel approximation and is widely used to measure sketching accuracy Wacker et al. [2024, 2023]. The datasets and experimental setups used here are identical to our previous experiments that evaluated accuracy using the KL divergence metric (Section 5 and Appendix C.1). Testing with this alternative metric ensures that our method performs well irrespective of the choice of error metric.

<table><tr><td>Degree (p)</td><td>Method</td><td>Variance</td></tr><tr><td rowspan="6">15</td><td>Real Gaussian</td><td> $4 . 4 1 \times 1 0 ^ { 4 }$ </td></tr><tr><td>Real Rademacher</td><td> $3 . 2 0 \times 1 0 ^ { 1 }$ </td></tr><tr><td>CtR Gaussian</td><td> $6 . 3 5 \times 1 0 ^ { 1 }$ </td></tr><tr><td>CtR Rademacher</td><td> $4 . 1 6 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Real TensorSketch</td><td> $2 . 8 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>CtR TensorSketch (Ours)</td><td> $\mathbf { 1 . 5 7 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td rowspan="6">20</td><td>Real Gaussian</td><td> $3 . 8 0 \times 1 0 ^ { 1 }$ </td></tr><tr><td>Real Rademacher</td><td> $6 . 7 8 \times 1 0 ^ { 3 }$ </td></tr><tr><td>CtR Gaussian</td><td> $2 . 7 6 \times 1 0 ^ { 1 }$ </td></tr><tr><td>CtR Rademacher</td><td> $1 . 5 8 \times 1 0 ^ { 1 }$ </td></tr><tr><td>Real TensorSketch</td><td> $1 . 2 8 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>CtR TensorSketch (Ours)</td><td> $\mathbf { 5 . 0 5 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td rowspan="6">25</td><td>Real Gaussian</td><td> $2 . 6 5 \times 1 0 ^ { 0 }$ </td></tr><tr><td>Real Rademacher</td><td> $9 . 0 6 \times 1 0 ^ { 3 }$ </td></tr><tr><td>CtR Gaussian</td><td> $1 . 8 2 \times 1 0 ^ { 1 }$ </td></tr><tr><td>CtR Rademacher</td><td> $2 . 5 8 \times 1 0 ^ { 1 }$ </td></tr><tr><td>Real TensorSketch</td><td> $3 . 6 8 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>CtR TensorSketch (Ours)</td><td> $\mathbf { 3 . 5 8 \times 1 0 ^ { - 4 } }$ </td></tr><tr><td rowspan="6">30</td><td>Real Gaussian</td><td> $7 . 2 4 \times 1 0 ^ { 0 }$ </td></tr><tr><td>Real Rademacher</td><td> $1 . 0 0 \times 1 0 ^ { 0 }$ </td></tr><tr><td>CtR Gaussian</td><td> $5 . 4 4 \times 1 0 ^ { 2 }$ </td></tr><tr><td>CtR Rademacher</td><td> $3 . 2 1 \times 1 0 ^ { 3 }$ </td></tr><tr><td>Real TensorSketch</td><td> $1 . 2 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>CtR TensorSketch (Ours)</td><td> $\mathbf { 4 . 5 9 \times 1 0 ^ { - 6 } }$ </td></tr></table>

Table 2: Variance of the Frobenius normalized relative error $( \| K - \hat { K } \| _ { F } / \| K \| _ { F } )$ evaluated on the real-world MAGIC dataset $( n = 1 0 0 0 , d = 1 0 )$ with a compressed sketch dimension of $D = 1 2 8 .$ . Results are aggregated over 20 independent trials across high-degree polynomial kernels $( p \in \{ 1 5 , 2 0 , 2 5 , 3 0 \} )$ . CtR TensorSketch strictly outperforms Real TensorSketch in estimation stability, confirming the variance reduction.

## C.3 EVALUATION ON DOWNSTREAM TASKS

To demonstrate practical application beyond kernel matrix approximation, we evaluate our method on downstream binary classification tasks using a linear Support Vector Machine (SVM). The datasets used in these experiments are generated using standard ML libraries (such as scikit-learn) to create two highly overlapping classes with non-linear decision boundaries. We test two separate configurations under extreme compression to a sketch dimension of $D = 6 4$

We compare the sketching methods against the Exact Polynomial Kernel, which computes the full kernel matrix using the mathematical formula $K ( x , y ) = ( x ^ { \top } y ) ^ { p }$ without any compression or approximation. We evaluate the techniques using two metrics: test classification accuracy and total execution time. The time metric tracks the combined end-to-end wall-clock time required for both data compression and the subsequent classifier training. We report this total time to fully reflect the complete workload required to obtain the final classification model.

As shown in Table 3 and Table 4, all approximation methods experience a drop in accuracy compared to the exact kernel due to the tight bottleneck of the compressed dimension. However, our proposed CtR TensorSketch consistently achieves the highest accuracy among all baselines while requiring the shortest total execution time.

<table><tr><td>Method</td><td>Accuracy</td><td>Time (s)</td></tr><tr><td>Exact Poly Kernel</td><td>0.8756</td><td>16.0724</td></tr><tr><td>TensorSketch (Real)</td><td>0.4911</td><td>0.2390</td></tr><tr><td>TensorSketch (CtR) [Ours]</td><td>0.5289</td><td>0.1479</td></tr><tr><td>JL (CtR Rademacher)</td><td>0.5011</td><td>3.8279</td></tr><tr><td>JL (CtR Gaussian)</td><td>0.4978</td><td>3.7708</td></tr></table>

Table 3: Downstream Classification Performance (Linear SVM) for Setup 1. Evaluated on a synthetic dataset $( n = 3 0 0 0 )$ with dimension $d = 2 0$ . We approximate a polynomial kernel of degree $p = 1 5$ using a sketch dimension of $D = 6 4$ . Our proposed method achieves the best sketching accuracy in the shortest time.

<table><tr><td>Method</td><td>Accuracy</td><td>Time (s)</td></tr><tr><td>Exact Poly Kernel</td><td>0.8567</td><td>8.5344</td></tr><tr><td>TensorSketch (Real)</td><td>0.4978</td><td>0.2763</td></tr><tr><td>TensorSketch (CtR) [Ours]</td><td>0.5044</td><td>0.1733</td></tr><tr><td>JL (CtR Rademacher)</td><td>0.4567</td><td>3.3204</td></tr><tr><td>JL (CtR Gaussian)</td><td>0.4889</td><td>3.2954</td></tr></table>

Table 4: Downstream Classification Performance (Linear SVM) for Setup 2. Evaluated on a synthetic dataset $( n = 3 0 0 0 )$ with dimension $d = 1 0$ . We approximate a polynomial kernel of higher degree $p = 2 5$ using a sketch dimension of $D = 6 4$

## C.4 VARIANCE ANALYSIS THROUGH NUMERICAL TABLE

The primary purpose of this section is to provide a clear and direct validation of our experimental findings from Section 5. In our main evaluation, we rely on visual plots to demonstrate the accuracy and computational time of our proposed CtR TensorSketch. Graphs, especially those that use a logarithmic scale, can visually compress the performance gap between methods. Looking at the exact numbers, we can clearly confirm our previous experiments and highlight the advantage of our method. Specifically, these numbers reveal how our approach successfully reduces the variance from $3 ^ { p } / D$ in traditional methods to $2 ^ { p } / D$ in our complex-to-real design.

As shown in Table 5, our proposed CtR TensorSketch consistently achieves lower variance compared to all the baselines across all degrees.

<table><tr><td>Degree (p)</td><td>Method</td><td>Variance</td></tr><tr><td rowspan="6">15</td><td>Real Gaussian</td><td>0.2634</td></tr><tr><td>Real Rademacher</td><td>0.3839</td></tr><tr><td>CtR Gaussian</td><td>0.2192</td></tr><tr><td>CtR Rademacher</td><td>0.2029</td></tr><tr><td>Real TensorSketch</td><td>0.5496</td></tr><tr><td>CtR TensorSketch (Ours)</td><td>0.1739</td></tr><tr><td rowspan="6">20</td><td>Real Gaussian</td><td>0.3545</td></tr><tr><td>Real Rademacher</td><td>0.4673</td></tr><tr><td>CtR Gaussian</td><td>0.6996</td></tr><tr><td>CtR Rademacher</td><td>0.5246</td></tr><tr><td>Real TensorSketch</td><td>0.3932</td></tr><tr><td>CtR TensorSketch (Ours)</td><td>0.3888</td></tr><tr><td rowspan="6">25</td><td>Real Gaussian</td><td>0.3158</td></tr><tr><td>Real Rademacher</td><td>0.2560</td></tr><tr><td>CtR Gaussian</td><td>0.4395</td></tr><tr><td>CtR Rademacher</td><td>0.2294</td></tr><tr><td>Real TensorSketch</td><td>0.3209</td></tr><tr><td>CtR TensorSketch (Ours)</td><td>0.2367</td></tr><tr><td rowspan="6">30</td><td>Real Gaussian</td><td>0.4785</td></tr><tr><td>Real Rademacher</td><td>0.2932</td></tr><tr><td>CtR Gaussian</td><td>0.3266</td></tr><tr><td>CtR Rademacher</td><td>0.2883</td></tr><tr><td>Real TensorSketch</td><td>0.2443</td></tr><tr><td>CtR TensorSketch (Ours)</td><td>0.1939</td></tr></table>

Table 5: Variance of KL Divergence for high-degree polynomial kernels. Evaluated on a positive orthant synthetic dataset $( n = 1 0 0 0 , d = 1 0 )$ with extreme compression $( D = 6 4 )$ , aggregated over 20 independent trials. CtR TensorSketch strictly dominates Real TensorSketch in estimation variance across all polynomial degrees, empirically validating the theoretical $2 ^ { p } / D$ scaling advantage.
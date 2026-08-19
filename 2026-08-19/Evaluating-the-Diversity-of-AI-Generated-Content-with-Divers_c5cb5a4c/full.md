# Evaluating the Diversity of AI-Generated Content with Diversity Profiles

Xiuyuan Hu<sup>1,2</sup>, Xuege Hou<sup>1</sup>, Guoqing Liu<sup>3</sup>, Yang Zhao<sup>1</sup>, Jieran Li<sup>1</sup>, Dongbiao Sun<sup>1</sup>, José Miguel Hernández-Lobato<sup>4</sup>, Hao Zhang<sup>1∗</sup>, Xue Liu<sup>2,5,6</sup>

<sup>1</sup>Department of Electronic Engineering, Tsinghua University, <sup>2</sup>MBZUAI,

<sup>3</sup>Microsoft Research AI for Science

<sup>4</sup>Department of Engineering, University of Cambridge <sup>5</sup>McGill University, <sup>6</sup>Mila - Quebec AI Institute

huxy22@mails.tsinghua.edu.cn, haozhang@tsinghua.edu.cn

## Abstract

Diversity is a fundamental criterion for evaluating generative artificial intelligence (AI) systems, yet its measurement remains inherently ambiguous. Existing approaches typically represent generated samples in an embedding space, compute pairwise distances or similarities, and aggregate them into a single scalar score. Such scalar summaries are convenient, but they often encode different inductive biases and may yield contradictory rankings of the same sample sets. In this paper, we argue that diversity evaluation for AI-generated content is intrinsically under-specified when reduced to a single number. We first review representative diversity metrics, and then diagnose their limitations from two complementary perspectives: an axiomatic analysis showing that no representative scalar metric satisfies all desirable properties simultaneously, and an empirical analysis showing that high-dimensional representation spaces can induce concentrated, modality dependent distance distributions. To address these issues, we propose diversity profiles: curve-valued, condition-aware summaries that evaluate a parameterized diversity family across a range of thresholds, scales, exponents, or orders under a specified representation and distance or kernel function. Diversity profiles reveal whether a comparison is robust across resolutions or instead depends on an arbitrary parameter choice. We instantiate profiles for several representative metric families and demonstrate their practical use in generative AI evaluation. Overall, diversity profiles provide a more transparent and resolution-aware framework for comparing the diversity of AI-generated content.

## 1 Introduction

Diversity has long been a central concept in the natural sciences. In ecology, it is studied under the name biodiversity, where researchers seek to quantify not only how many species are present in a community, but also how evenly individuals are distributed among them and how distinct the species are from one another [15]. Classical diversity indices such as Shannon entropy [26], Simpson’s index [27], and Hill numbers [11] have therefore played a central role in comparing biological communities. A key lesson from this literature is that diversity is not a single primitive notion: richness, evenness, rarity, and dissimilarity describe related but distinct aspects of a population [18].

In the era of generative artificial intelligence (AI), diversity evaluation has become an increasingly important and challenging problem. Generative models are expected not only to produce high-quality samples, but also to avoid mode collapse, excessive repetition, and overly narrow coverage of the target domain [29]. This issue arises across modalities, including images, natural language, graphs, and molecules. In natural language generation, for example, diversity may refer to variation in topics, reasoning paths, lexical choices, syntactic forms, or semantic meanings. These notions operate at different resolutions and are often induced by different representations, making it difficult to summarize diversity by a single universal scalar [24].

![](images/30b4ec2678735d14a26beafa99e5bdf114f4e09afdab5c4b09c3ba84d4285e58.jpg)  
(a) Circles profile

![](images/a08ce0227e2ff276a0e2d0ee16bf804ec03fbc1f35caf0ff9574b4cd35de50ac.jpg)  
(b) Vendi score profile

![](images/70fef3f2a7a498268369a378dc6f566f31d485a19657e82c9eca4eda83b315ff.jpg)  
(c) Magnitude profile  
Figure 1: Diversity profiles of a minimal one-dimensional example. Set A = {1, 2, 3}, set B = {0, 4}, and distance is measured by absolute difference. Sweeping the profile parameter of three representative diversity metrics reveals resolution-dependent comparisons, illustrating how a single scalar diversity score can obscure important trade-offs.

A common strategy in recent work is to embed generated samples into a representation space, compute pairwise distances or similarities, and then aggregate the resulting matrix into one diversity score. Representative metrics include average pairwise distance [5], energy-based scores [30], threshold-based packing metrics such as Circles [33], similarity-spectrum methods such as the Vendi score [8, 23], and metric-space magnitude [20]. These metrics are useful and often interpretable, but they encode different inductive biases. As a result, different metrics can rank the same two generated sets in opposite ways.

This ambiguity is not merely a matter of implementation. We argue that scalar diversity evaluation is intrinsically under-specified. First, widely used diversity metrics satisfy different subsets of desirable axioms, such as size monotonicity, invariance to duplicates, monotonicity with respect to pairwise distances, and continuity. No representative scalar metric satisfies all of these properties simultaneously in its natural domain. Second, many modern diversity metrics contain a scale, threshold, exponent, or order parameter. Different choices of this parameter may lead to different conclusions even within the same metric family. Third, generative AI evaluations are typically performed in high-dimensional representation spaces, where pairwise distances may concentrate and where the meaningful range of scales depends strongly on the embedding model, modality, and distance function.

Figure 1 illustrates this issue in a minimal example. Richness prefers A, because A contains three distinct elements while B contains two. Average distance prefers B, because the two elements of B are farther apart. Parameterized metrics further reveal that the comparison can depend on the chosen resolution: the Circles, Vendi score, and Magnitude curves may cross as their threshold, order, or scale parameter varies. Thus, reporting a single number can hide important trade-offs and may allow arbitrary metric or parameter choices to determine the claimed conclusion.

To address this problem, we propose to evaluate the diversity of AI-generated content using diversity profiles. Instead of introducing another scalar metric, a diversity profile reports the behavior of a parameterized diversity family across a range of meaningful parameter values. The resulting curve provides a condition-aware summary of diversity under a specified representation, distance or similarity function, metric family, and parameter domain. Diversity profiles are intuitive to interpret: when one curve dominates another across the full parameter domain, the diversity comparison is robust to the choice of resolution; when curves cross, the crossing directly reveals a scale-dependent trade-off that would be obscured by a single scalar score.

Diversity profiles also make it possible to compare generated sample sets not only across resolutions within a fixed metric family, but also across different diversity metric families. Since different metrics emphasize different aspects of diversity, jointly inspecting their profiles helps distinguish conclusions that are metric-family-specific from those that are robust across multiple notions of diversity. Computationally, the additional cost of constructing profiles is modest, making the approach practical for routine generative AI evaluation.

Our contributions are as follows:

• We provide a comprehensive review of representative diversity metrics for AI-generated content, and diagnose limitations of diversity evaluation with scalar metrics through both an axiomatic analysis and the geometry of high-dimensional representation spaces.

• We introduce diversity profiles as curve-valued, condition-aware summaries that compare generated sample sets across thresholds, scales, exponents, or orders, rather than at a single arbitrary parameter value.

• We instantiate diversity profiles for several representative metric families, including Energy, Circles, Vendi score, and Magnitude, and show how profile comparisons can be used both within a metric family across resolutions and across metric families to obtain more robust diversity assessments. We also discuss their profile-level properties, intuitive interpretation, and computational practicality for generative AI evaluation.

Overall, diversity profiles shift the goal of diversity evaluation from selecting a single supposedly best scalar metric to characterizing how diversity comparisons behave across resolutions and across metric families. This perspective makes diversity evaluation more transparent, intuitive, less sensitive to arbitrary hyperparameter choices, and better aligned with the multi-faceted nature of diversity in generative AI.

## 2 Existing Diversity Metrics for AI-Generated Content

Diversity is a central evaluation criterion for generative AI systems across modalities such as images, text, graphs, and molecules. A common approach is to first represent each generated item in an embedding or feature space, compute pairwise distances between the resulting representations, and then summarize the resulting matrix by a scalar score. We refer to this broad family as distance-based diversity metrics. Since the embedding model and the distance function determine the geometry being measured, such metrics should be interpreted as representation-dependent summaries of diversity.

Definition 1 (Distance-based diversity metric). Let $X = \{ x _ { 1 } , \ldots , x _ { n } \}$ denote a finite collection, or multiset, of generated items. Let d be a nonnegative pairwise distance (dissimilarity) defined on the representation space, and let $D \in \mathbb { R } _ { > 0 } ^ { n \times n }$ be the corresponding distance matrix with entries

$$
D _ { i j } = d ( x _ { i } , x _ { j } ) .
$$

A distance-based diversity metric is a permutation-invariant functional

$$
\begin{array} { r } { \mu _ { n } : \mathcal { D } _ { n }  \mathbb { R } , \qquad \mu _ { n } ( D ) = \mu _ { n } ( P D P ^ { \top } ) } \end{array}\tag{1}
$$

for any permutation matrix $P ,$ where $\mathcal { D } _ { n }$ denotes the admissible set of pairwise distance matrices. Such metrics quantify diversity using only pairwise relations among the elements of $X$

We review several representative examples below.

Definition 2 (Average distance [5]). For $n \geq 2$ , the average pairwise distance (AvgDist) is

$$
\mu _ { \mathrm { A v g D i s t } } ( D ) : = \frac { 1 } { n ( n - 1 ) } \sum _ { i \neq j } D _ { i j } .\tag{2}
$$

This metric, also commonly referred to as internal diversity, is simple, interpretable, and widely used across generative modeling applications.

Definition 3 (Energy [30]). For $s > 0$ , the energy-based diversity score is

$$
\mu _ { \mathrm { E n e r g y } } ( D ; s ) : = - \frac { 1 } { n ( n - 1 ) } \sum _ { i \neq j } \frac { 1 } { D _ { i j } ^ { s } } .\tag{3}
$$

The exponent s controls the penalty assigned to small pairwise distances. This functional is inspired by the potential energy of repulsive particles: configurations with very close pairs receive large penalties. The score is well defined only when $D _ { i j } > 0$ for all $i \neq j ;$ in practice, implementations may add a small $\varepsilon > 0$ to avoid singularities.

Definition 4 (Circles [33]). For a threshold $\tau \geq 0$ , define

$$
\mu _ { \mathrm { C i r c l e s } } ( D ; \tau ) : = \operatorname* { m a x } _ { S \subseteq X } | S | , { \mathrm { s . t . } } \quad D _ { i j } > \tau , \quad \forall i \neq j { \mathrm { ~ s u c h ~ t h a t ~ } } x _ { i } , x _ { j } \in S .\tag{4}
$$

This metric measures the largest subset of generated items whose elements are mutually separated by more than τ. When d is a metric, it coincides with the packing number [31].

Definition 5 (Vendi score [8, 23]). The Vendi score is similarity-based. Let $K \in \mathbb { R } ^ { n \times n }$ be a symmetric positive semidefinite similarity matrix with normalized diagonal entries, $\begin{array} { r } { \mathrm { e } . \mathrm { g } . , K _ { i i } = 1 } \end{array}$ obtained either directly from a similarity function or indirectly from D through a kernel transformation. Let $\lambda _ { 1 } , \ldots , \lambda _ { n }$ denote the eigenvalues of $K / n$ , so that $\lambda _ { i } \geq 0$ and $\textstyle \sum _ { i } \lambda _ { i } = 1$ . For order $q > 0$ and $q \neq 1$ , the order-q Vendi score is

$$
\mu _ { \mathrm { V e n d i } } ( K ; q ) : = \left( \sum _ { i = 1 } ^ { n } \lambda _ { i } ^ { q } \right) ^ { \frac { 1 } { 1 - q } } .\tag{5}
$$

The original Vendi score is obtained in the limit $q \to 1$

$$
\mu _ { \mathrm { V e n d i } } ( K ; 1 ) : = \exp \left( - \sum _ { i = 1 } ^ { n } \lambda _ { i } \log \lambda _ { i } \right) ,\tag{6}
$$

with the convention 0 log $0 = 0 .$ . Thus, the Vendi score is the exponential of the Shannon entropy when $q = 1$ and of the Rényi entropy for $q \neq 1$ . The order-2 case is closely related to Rényi kernel entropy [14], and Fourier-feature approximations have been proposed to scale related kernel-entropy computations [22].

Definition 6 (Magnitude [20]). Let $Z _ { t } ( D ) \in \mathbb { R } ^ { n \times n }$ be the similarity matrix

$$
[ Z _ { t } ( D ) ] _ { i j } = \exp ( - t \cdot D _ { i j } ) , \qquad t > 0 .
$$

If $Z _ { t } ( D )$ is nonsingular, the magnitude at scale t is

$$
\mu _ { \mathrm { M a g } } ( D ; t ) : = \mathbf { 1 } ^ { \top } Z _ { t } ( D ) ^ { - 1 } \mathbf { 1 } = \sum _ { i , j } [ Z _ { t } ( D ) ^ { - 1 } ] _ { i j } .\tag{7}
$$

This construction is the finite-space version of magnitude in metric geometry [16] and can be interpreted as an effective size of the metric space at scale t.

Other metrics. The above list is not exhaustive. Other recent distance-based proposals include Hamiltonian diversity [13], NovelSum [35], and additional axiomatic constructions [21].

Beyond distance-based metrics, diversity is also quantified by count-based metrics. A representative example is Richness, defined as the number of unique elements in a batch. Domain-specific variants include counting unique words or n-grams in natural language generation and unique molecular scaffolds in cheminformatics. These metrics are robust and easy to interpret, but they do not capture graded distances among distinct generated items.

## 3 Diagnosis of Existing Metrics

## 3.1 An Axiomatic Analysis

A useful way to evaluate diversity metrics is through axioms: one specifies desirable properties of a diversity measure and then checks which metrics satisfy them. This perspective appears in several recent studies, although the exact axioms differ across papers [2, 17, 21]. Here we summarize four desiderata that are particularly relevant for evaluating generated samples. The notation follows Section 2, and we write $\mu _ { n }$ when the sample size matters.

Axiom 1 (Size Monotonicity). Adding a new element should not decrease diversity. Formally, if $D ^ { \prime } \in \mathcal { D } _ { n + 1 }$ extends $D \in \mathcal { D } _ { n } ^ { \dot { \mathbf { \alpha } } }$ by adding one row and column corresponding to a new element $x _ { n + 1 } .$ then

$$
\mu _ { n + 1 } ( D ^ { \prime } ) \geq \mu _ { n } ( D ) .\tag{8}
$$

Table 1: Axiomatic analysis of representative diversity metrics. Symbols: ✓ = satisfies, ✗ = violates, – = the metric is not distance-based.
<table><tr><td>Diversity Metric</td><td>A1 Size Mono.</td><td>A2 Twin</td><td>A3 Dist. Mono.</td><td>A4 Cont.</td></tr><tr><td>Richness</td><td>L</td><td>√</td><td></td><td></td></tr><tr><td>AvgDist</td><td>X</td><td>X</td><td></td><td></td></tr><tr><td>Energy (s)</td><td>X</td><td>X</td><td>√</td><td>√</td></tr><tr><td>Circles (τ)</td><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td>Vendi score (q)</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Magnitude (t)</td><td>J</td><td>X</td><td>X</td><td>J</td></tr></table>

Axiom 2 (Twin Property). Adding a duplicate element should not change diversity. If the added element is a duplicate of some existing element $x _ { i } .$ , meaning that its distances to all existing elements match those of $x _ { i }$ and $D _ { n + 1 , i } ^ { \prime } = 0$ , then

$$
\mu _ { n + 1 } ( D ^ { \prime } ) = \mu _ { n } ( D ) .\tag{9}
$$

This property prevents repeated samples from artificially increasing the diversity score.

Axiom 3 (Distance Monotonicity). For two admissible distance matrices $D , D ^ { \prime } \in \mathcal { D } _ { n }$ of the same size, if

$$
D _ { i j } ^ { \prime } \geq D _ { i j } , \qquad \forall i \neq j ,
$$

then

$$
\mu _ { n } ( D ^ { \prime } ) \geq \mu _ { n } ( D ) .\tag{10}
$$

This axiom states that increasing pairwise separations should not reduce measured diversity.

Axiom 4 (Continuity). The metric should vary continuously under small perturbations of the pairwise distances. For example, under the Frobenius norm,

$$
D ^ { ( m ) } \to D \quad \Rightarrow \quad \mu _ { n } ( D ^ { ( m ) } ) \to \mu _ { n } ( D ) .\tag{11}
$$

Table 1 summarizes the behavior of the representative metrics, with proofs in Appendix A. No single metric in this list satisfies all four axioms in its natural domain. These incompatibilities suggest that a single scalar score may not provide afully reliable description ofdiversity in all settings.

## 3.2 High-dimensional Representation Spaces

A second challenge arises from the geometry of high-dimensional representation spaces. Highdimensional spaces exhibit behaviors that differ sharply from low-dimensional intuition, a collection of phenomena often referred to as the curse of dimensionality [4]. One particularly relevant effect is distance concentration: as dimension grows, pairwise distances among random points can become nearly indistinguishable [6, 3].

A common asymptotic formulation is the following. Let $P _ { d }$ be a sequence of distributions on $\mathbb { R } ^ { d }$ and let $x _ { 1 } , \ldots , x _ { n } \sim P _ { d }$ be i.i.d. samples for fixed n. Let $\mathrm { d i s t } _ { \mathrm { m a x } } ( d )$ and $\mathrm { d i s t } _ { \mathrm { m i n } } ( d )$ denote the maximum and minimum pairwise distances among these samples. Under standard concentration conditions on the pairwise distance random variable, the relative contrast satisfies

$$
\frac { \mathrm { d i s t } _ { \mathrm { m a x } } ( d ) - \mathrm { d i s t } _ { \mathrm { m i n } } ( d ) } { \mathrm { d i s t } _ { \mathrm { m i n } } ( d ) } \xrightarrow [ d  \infty ] { \boldsymbol { p } } 0 .\tag{12}
$$

Equivalently, the nearest and farthest pairwise distances become similar in relative terms. This phenomenon weakens distance-based notions such as nearest neighbors, density, locality, and separation, which underlie many scalar diversity metrics [1, 36].

In generative AI, high dimensionality is not merely a theoretical concern. Common deep representations often have hundreds or thousands of dimensions before any downstream distance is computed. To illustrate this issue, Figure 2 reports empirical distributions of pairwise distances in three representative domains: images, natural language text, and molecular structures. The observed concentration across these domains is consistent with high-dimensional representation effects under the chosen embeddings and distance functions. At the same time, the location and shape of the concentrated distributions differ substantially across modalities and distance definitions. These observations sharpen the parameter-selection problem for diversity metrics: when pairwise distances occupy only a narrow empirical range, large regions of the parameter space may become saturated or uninformative, making any single threshold, scale, exponent, or order intrinsically fragile and representation-dependent. This motivates a more comprehensive, condition-aware view of diversity, rather than reliance on a single absolute score.

![](images/26bc0ab45ffa82ecf040b68d32668a0272fe1a0c6a8f5d0cc02602ca94a434b8.jpg)  
(a) Images

![](images/68863f4f0cc4f848de75582d0e59e43898a33cabd3e6590f6d9a61ab0aff4ca3.jpg)  
(b) Text

![](images/ea2c1abf4e463194ae56f803d73da81a0164893851c9cb011acfee0cc55aed0f.jpg)  
(c) Molecules  
Figure 2: Empirical distributions of pairwise distances in three generative AI domains.

## 4 Diversity Profiles

## 4.1 Motivation for Diversity Profiles

Sections 3.1 and 3.2 show that diversity evaluation for generated content is inherently multi-faceted. From a theoretical perspective, no scalar metric considered in Section 2 satisfies all four desiderata in Table 1: size monotonicity, twin property, distance monotonicity, and continuity. Hence, a single scalar score cannot simultaneously capture all desirable aspects of diversity. Moreover, even within a fixed metric family, the conclusion may depend on the choice of a scale, threshold, or order parameter. As illustrated in Figure 1, two sample sets can be ranked differently by the same diversity family under different hyperparameter values.

A closely related idea has a long history in ecology, where diversity profiles are used to compare biological communities across the full range of Hill-number orders rather than reporting a single diversity index [19, 18]. In that setting, the order parameter determines the sensitivity of the diversity measure to rare species: small orders emphasize richness and rare types, whereas large orders emphasize dominant types and evenness among common species. This perspective is especially useful when no single order is universally appropriate.

Remark 1. Hill numbers, Rényi entropy, Shannon entropy, and the Vendi score are closely connected. In particular, Hill numbers are exponentials of Rényi entropies, with Shannon entropy recovered in the limit as the order tends to one. The Vendi score applies the same effective-number principle to the spectrum of a normalized similarity matrix. Thus, diversity profiles can be naturally generalized to the Vendi score with an order parameter.

From a practical perspective, diversity evaluation is condition-specific. A distance-based score depends not only on the generated samples, but also on the representation, distance function, and parameter domain. This dependence becomes especially pronounced in high-dimensional embedding spaces, where pairwise distances often concentrate and where the concentration location and spread vary across modalities, embedding models, and distance functions, as shown in Section 3.2. Consequently, a fixed numerical threshold or scale has no universal meaning: it may be too small to distinguish local structure in one representation, too large to avoid degeneracy in another, or sensitive to small changes in conditions. Parameter selection is therefore part of the evaluation problem itself. Diversity profiles can address this by replacing a single parameter choice with a curve over an empirically meaningful range.

This motivates two essential components of a diversity profile. First, the curve is important because it reports the behavior of a metric family over a domain of scales, thresholds, or orders, rather than committing to one hyperparameter value. Second, the condition is important because the curve is only interpretable relative to the chosen representation, distance, and parameter domain. In this sense, a diversity profile is a condition-aware summary that exposes how diversity comparisons change across resolutions under a specified evaluation setting.

Table 2: Profile-level properties of representative diversity families.
<table><tr><td>Metric family</td><td>Parameter</td><td>Monotonicity in parameter</td><td>Continuity</td></tr><tr><td>Energy</td><td> $s > 0$ </td><td>case-dependent</td><td>continuous</td></tr><tr><td>Circles</td><td> $\tau \geq 0$ </td><td>nonincreasing</td><td>stepwise</td></tr><tr><td>Vendi score</td><td> $q > 0$ </td><td>nonincreasing</td><td>continuous</td></tr><tr><td>Magnitude</td><td> $t > 0$ </td><td>typically nondecreasing</td><td>continuous</td></tr></table>

## 4.2 Definition and Properties of Diversity Profiles

Let $X = \{ x _ { 1 } , \ldots , x _ { n } \}$ be a finite multiset of generated samples. A representation map ϕ sends each sample to a feature space, and a distance function d induces the pairwise distance matrix

$$
D _ { i j } = d ( \phi ( x _ { i } ) , \phi ( x _ { j } ) ) .
$$

For similarity-based metrics such as the Vendi score, we analogously use a positive semidefinite similarity matrix $K$ obtained either directly from a similarity function or from D through a fixed kernel transformation.

Definition 7 (Diversity profile). A diversity profile for generated content is specified by a tuple

$$
{ \mathcal { P } } = ( \phi , d , \mu , \Theta ) ,
$$

where $\phi$ is the representation, $d$ is the distance function, $\mu$ is a parameterized family of diversity functionals, and Θ is the admissible set of scale, threshold, or order parameters. For a sample set $\check { X }$ the associated profile is the function

$$
P _ { X } ^ { \mathcal { P } } ( \theta ) = \mu _ { \theta } ( D _ { X } ) , \qquad \theta \in \Theta ,\tag{13}
$$

or $P _ { X } ^ { \mathcal { P } } ( \theta ) = \mu _ { \theta } ( K _ { X } )$ for similarity-based families.

Thus, a diversity profile is not a new scalar metric, but a curve-valued summary of how a metric family behaves across resolutions. Richness and Average distance have no intrinsic controlling hyperparameter, so they can be regarded as degenerate, scale-rigid profiles. In contrast, Energy, Circles, Vendi score, and Magnitude naturally define nontrivial profiles.

Table 2 summarizes the main profile-level properties of these four families. All four profiles are permutation-invariant because they depend only on the pairwise distance or similarity matrix, not on the ordering of samples. Their behavior with respect to the profile parameter, however, differs substantially.

Energy profile. The exponent s controls the penalty assigned to small distances. The profile is continuous in s whenever all pairwise distances are strictly positive, while its monotonicity depends on the values of distances (compared to 1).

Circles profile. The threshold τ specifies the minimum separation required between selected samples. As τ increases, the feasibility constraint becomes stricter, so the profile is nonincreasing. It is an integer-valued step function whose jumps occur at observed pairwise distances. Thus, Circles is particularly interpretable as a resolution-dependent packing number: small thresholds reveal near-duplicate behavior, while large thresholds measure coverage by well-separated representatives.

Vendi score profile. The profile is continuous in the order $q ,$ including near $q = 1$ . It is nonincreasing in $q \mathrm { : }$ small $q$ values are sensitive to low-mass spectral components and therefore capture rare directions of variation, whereas large q values emphasize dominant components and measure whether diversity is concentrated in only a few modes.

Magnitude profile. The scale t controls the resolution at which points are distinguished. In standard finite metric settings, the profile is continuous and nondecreasing in t: small t makes most points highly similar, so the space is viewed coarsely; large t makes only very close points similar, so the space is viewed at a finer resolution.

![](images/bbfa7a1ff0a43066484e06fee061658683bafcb12c063299d7570ee26f07d587.jpg)  
Figure 3: Diversity profiles of two real natural-language sample sets.

## 4.3 Practical Usage in Generative AI

Figure 3 gives an example of diversity profiles for two real natural-language sample sets. We use question-answer pairs from the 2WikiMultiHopQA dataset [12] and embed them with the Qwen3-8B model [34], which produces 4096-dimensional dense representations. Each set contains 50 randomly sampled items. Each curve is evaluated at more than 100 parameter values, and computing each profile takes less than one second on a standard CPU once the pairwise distance or similarity matrix has been constructed. This suggests that diversity profiles are computationally practical for routine generative AI evaluation. Additional examples of diversity profiles across evaluation settings are provided in Appendix C.

The interpretation of a profile comparison depends on whether the curves exhibit dominance or crossing. If one set has a higher profile value than another set for all $\theta \in \Theta$ , then the diversity ordering is robust to the choice of hyperparameter within that metric family. Such uniform dominance provides stronger evidence than a single scalar comparison, because the conclusion does not depend on an arbitrary scale, threshold, or order.

If the curves cross, then there is no unconditional diversity ordering between the two sets under the chosen profile. Instead, the crossing identifies a scale-dependent trade-off. For Energy, superiority at small s mainly reflects larger average pairwise separation under weak sensitivity to close pairs, whereas superiority at large s indicates fewer extremely close pairs, since the inverse-power penalty becomes dominated by the smallest distances. For Circles, a set with a higher value at small τ contains more nonduplicate or locally distinct samples, whereas a higher value at large τ indicates better coverage by mutually well-separated samples. For Vendi score, superiority at small q suggests more rare spectral directions, while superiority at large q suggests a more even distribution among dominant modes. For Magnitude, differences at small or moderate t reflect coarse-scale separation, while differences at larger t reflect finer-scale resolution among nearby samples.

In practice, we recommend choosing Θ according to the empirical distance or similarity distribution. For threshold profiles such as Circles, τ can be swept over distance quantiles rather than over an arbitrary linear interval. For kernel-based Vendi score profiles and scale-based Magnitude profiles, the parameter can be chosen so that the induced similarities span a nondegenerate domain between nearly identical and clearly separated samples.

Different diversity metric families may also yield contradictory comparisons. For example, in Figure 3, the Energy profile indicates that set A is more diverse, while Magnitude favors set B; the Circles profile exhibits a crossing pattern, and the Vendi score profile shows negligible separation between the two curves. This discrepancy reflects the fact that different metric families emphasize different aspects of diversity, such as close-pair repulsion, packing at a given threshold, spectral effective rank, or scale-dependent effective size. Therefore, we recommend reporting diversity profiles from multiple metric families when possible. Consistent superiority across several families provides substantially stronger evidence of higher diversity than dominance under a single metric family alone.

## 5 Conclusion and Discussion

In this paper, we study the problem of diversity evaluation for AI-generated content from both theoretical and practical perspectives. We first review representative distance-based diversity metrics, and show that these scalar metrics encode different inductive biases. Through an axiomatic analysis, we observe that none of the representative scalar metrics we analyze satisfies all four desiderata simultaneously. Through an empirical analysis of high-dimensional representation spaces, we further show that pairwise distances can concentrate in a modality- and representation-dependent manner, making any single parameter intrinsically fragile. Motivated by these observations, we propose diversity profiles: curve-valued, condition-aware summaries that evaluate a parameterized diversity family over a meaningful parameter domain under a specified representation and distance function. Diversity profiles reveal whether a diversity comparison is robust across resolutions or instead depends on a particular parameter choice, and reporting profiles from multiple metric families provides a more comprehensive view of diversity than relying on a single scalar score.

Several directions remain open. First, the metric families used to construct diversity profiles are themselves not fully satisfactory from an axiomatic perspective. Although diversity profiles reduce the dependence on a single arbitrary parameter and the use of multiple metric families can exploit their complementarity, a deeper theoretical understanding of the relationships among diversity metrics, axioms, and profile-level properties is still needed. Second, diversity evaluation for generative AI may benefit from more systematic subset-level analysis. In ecology, α diversity describes diversity within subcommunities, β diversity describes variation among subcommunities, and γ diversity describes diversity of the overall metacommunity, with classical decompositions such as $\gamma = \alpha \times \beta$ [32]. Analogous decompositions could also be useful for generative AI evaluation, which may help distinguish local repetition from global coverage and may provide more informative diagnostics for model comparison.

Limitations. First, this work focuses on diversity evaluation rather than the joint evaluation of quality and diversity. In generative AI, these two dimensions are tightly coupled: a system can appear diverse by generating low-quality or off-distribution samples, while a high-quality system may still suffer from mode collapse or limited coverage [24]. Developing evaluation protocols that jointly characterize fidelity, utility, and diversity remains an important challenge. Second, diversity profile are condition-dependent. Their interpretation depends on the embedding representation, distance or similarity function, metric family, and parameter domain. Although this condition-dependence is made explicit by the profile formulation, practical deployment in specific domains still requires more detailed guidelines for choosing representations, defining meaningful parameter domains, and interpreting profile dominance or crossing patterns.

Broader impacts. Moving from scalar diversity metrics to diversity profiles encourages more transparent, resolution-aware, and less hyperparameter-dependent evaluation of generative AI systems. This can help researchers and practitioners detect mode collapse, excessive repetition, narrow coverage, and metric-specific conclusions that would be hidden by a single score.

## References

[1] Charu C Aggarwal, Alexander Hinneburg, and Daniel A Keim. On the surprising behavior of distance metrics in high dimensional space. In International conference on database theory, pages 420–434. Springer, 2001.

[2] Enrique Amigó, Damiano Spina, and Jorge Carrillo-de Albornoz. An axiomatic analysis of diversity evaluation metrics: Introducing the rank-biased utility metric. In The 41st international ACM SIGIR conference on research & development in information retrieval, 2018.

[3] Fabrizio Angiulli. On the behavior of intrinsically high-dimensional spaces: distances, direct and reverse nearest neighbors, and hubness. Journal of Machine Learning Research, 18(170):1–60, 2018.

[4] Richard Ernest Bellman. Adaptive Control Processes: A Guided Tour. Princeton University Press, 1961.

[5] Mostapha Benhenda. Chemgan challenge for drug discovery: can ai reproduce natural chemical diversity? arXiv preprint arXiv:1708.08227, 2017.

[6] Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, and Uri Shaft. When is “nearest neighbor” meaningful? In International Conference on Database Theory (ICDT), 1999.

[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.

[8] Dan Friedman and Adji Bousso Dieng. The vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023.

[9] Anna Gaulton, Louisa J Bellis, A Patricia Bento, Jon Chambers, Mark Davies, Anne Hersey, Yvonne Light, Shaun McGlinchey, David Michalovich, Bissan Al-Lazikani, et al. Chembl: a large-scale bioactivity database for drug discovery. Nucleic acids research, 40(D1):D1100– D1107, 2012.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[11] Mark O Hill. Diversity and evenness: a unifying notation and its consequences. Ecology, 54(2):427–432, 1973.

[12] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multihop QA dataset for comprehensive evaluation of reasoning steps. In The 28th International Conference on Computational Linguistics, 2020.

[13] Xiuyuan Hu, Guoqing Liu, Quanming Yao, Yang Zhao, and Hao Zhang. Hamiltonian diversity: effectively measuring molecular diversity by shortest hamiltonian circuits. Journal of Cheminformatics, 16(1):94, 2024.

[14] Mohammad Jalali, Cheuk Ting Li, and Farzan Farnia. An information-theoretic evaluation of generative models in learning multi-modal distributions. In Annual Conference on Neural Information Processing Systems (NeurIPS), 2023.

[15] Lou Jost. Entropy and diversity. Oikos, 113(2):363–375, 2006.

[16] Tom Leinster. The magnitude of metric spaces. Documenta Mathematica, 18:857–905, 2013.

[17] Tom Leinster. Entropy and diversity: the axiomatic approach. Cambridge university press, 2021.

[18] Tom Leinster and Christina A Cobbold. Measuring diversity: the importance of species similarity. Ecology, 93(3):477–489, 2012.

[19] Clifford E Lewis, Benee F Swindel, and George W Tanner. Species diversity and diversity profiles: concept, measurement, and application to timber and range management. Journal of Range Management, 41(6):466–469, 1988.

[20] Katharina Limbeck, Rayna Andreeva, Rik Sarkar, and Bastian Rieck. Metric space magnitude for evaluating the diversity of latent representations. In Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

[21] Mikhail Mironov and Liudmila Prokhorenkova. Measuring diversity: Axioms and challenges. In Forty-second International Conference on Machine Learning (ICML), 2025.

[22] Azim Ospanov, Jingwei Zhang, Mohammad Jalali, Xuenan Cao, Andrej Bogdanov, and Farzan Farnia. Towards a scalable reference-free evaluation of generative models. In Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

[23] Amey P Pasarkar and Adji Bousso Dieng. Cousins of the vendi score: A family of similaritybased diversity metrics for science and machine learning. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2024.

[24] Ossi Räisä, Boris van Breugel, and Mihaela van der Schaar. Position: All current generative fidelity and diversity metrics are flawed. In International Conference on Machine Learning (ICML) Position Paper Track, 2025.

[25] David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of chemical information and modeling, 50(5):742–754, 2010.

[26] Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

[27] Edward H Simpson. Measurement of diversity. nature, 163(4148):688–688, 1949.

[28] Taffee T Tanimoto. An elementary mathematical theory of classification and prediction. IBM Internal Report, 1958.

[29] Guy Tevet and Jonathan Berant. Evaluating the evaluation of diversity in natural language generation. In The 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2021.

[30] Fedor Velikonivtsev, Mikhail Mironov, and Liudmila Prokhorenkova. Challenges of generating structurally diverse graphs. In Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

[31] Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.

[32] Robert Harding Whittaker. Vegetation of the siskiyou mountains, oregon and california. Ecological monographs, 30(3):279–338, 1960.

[33] Yutong Xie, Ziqiao Xu, Jiaqi Ma, and Qiaozhu Mei. How much space has been explored? measuring the chemical space covered by databases and machine-generated molecules. In International Conference on Learning Representations (ICLR), 2023.

[34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[35] Yuming Yang, Yang Nan, Junjie Ye, Shihan Dou, Xiao Wang, Shuo Li, Huijie Lv, Tao Gui, Qi Zhang, and Xuan-Jing Huang. Measuring data diversity for instruction tuning: A systematic analysis and a reliable metric. In Annual Meeting of the Association for Computational Linguistics (ACL), 2025.

[36] Arthur Zimek, Erich Schubert, and Hans-Peter Kriegel. A survey on unsupervised outlier detection in high-dimensional numerical data. Statistical Analysis and Data Mining: The ASA Data Science Journal, 5(5):363–387, 2012.

## A Proofs of Table 1

We provide proofs and counterexamples for the axiomatic summary in Table 1. Throughout, all distance matrices are symmetric with zero diagonal. $\mathbf { A } \checkmark$ means that the property holds on the natural domain of the corresponding metric. A ✗ means that either the property fails, or the functional is not well defined on the configurations required by the axiom (for example, exact duplicates for singularity-based functionals).

Richness. Richness is the number of distinct elements. Adding an element can either create a new equivalence class or join an existing one, so richness satisfies size monotonicity. Adding an exact duplicate joins an existing equivalence class, so the twin property also holds.

Average pairwise distance. For $n \geq 2 .$

$$
\mu _ { \mathrm { A v g D i s t } } ( D ) = \frac { 1 } { n ( n - 1 ) } \sum _ { i \neq j } D _ { i j } .
$$

Distance monotonicity follows immediately from linearity: if $D _ { i j } ^ { \prime } \geq D _ { i j }$ for all $i \neq j ,$ , then

$$
\mu _ { \mathrm { A v g D i s t } } ( D ^ { \prime } ) - \mu _ { \mathrm { A v g D i s t } } ( D ) = \frac { 1 } { n ( n - 1 ) } \sum _ { i \neq j } ( D _ { i j } ^ { \prime } - D _ { i j } ) \geq 0 .
$$

Continuity also follows from linearity.

Size monotonicity fails. Consider two points with distance 1:

$$
D = { \binom { 0 } { 1 } } \ \mathrm { \ L } ^ { 1 } \mathrm { \ L } ^ { } \qquad \mu _ { \mathrm { A v g D i s t } } ( D ) = 1 .
$$

Add a duplicate of the first point:

$$
D ^ { \prime } = { \binom { 0 } { 1 } } \ 0 \ 1 \atop 0 \ 1 \ 0 ^ { \prime } .
$$

Then

$$
\mu _ { \mathrm { A v g D i s t } } ( D ^ { \prime } ) = \frac { 2 ( 1 + 0 + 1 ) } { 3 \cdot 2 } = \frac { 2 } { 3 } < 1 .
$$

Thus A1 fails. The same example shows that the twin property fails, since adding a duplicate changes the score from 1 to $2 / 3$

Energy. For fixed $s > 0$

$$
\mu _ { \mathrm { E n e r g y } } ( D ; s ) = - \frac { 1 } { n ( n - 1 ) } \sum _ { i \neq j } \frac { 1 } { D _ { i j } ^ { s } } ,
$$

on the domain where all off-diagonal distances are strictly positive.

Distance monotonicity holds on this domain. If $D _ { i j } ^ { \prime } \geq D _ { i j } > 0$ , then

$$
\frac { 1 } { ( D _ { i j } ^ { \prime } ) ^ { s } } \leq \frac { 1 } { D _ { i j } ^ { s } } ,
$$

and therefore

$$
\mu _ { \mathrm { E n e r g y } } ( D ^ { \prime } ; s ) \geq \mu _ { \mathrm { E n e r g y } } ( D ; s ) .
$$

Continuity follows because $x \mapsto - x ^ { - s }$ is continuous on $( 0 , \infty )$ , and the score is a finite average of such terms.

Size monotonicity fails. Take three points on the real line: $x _ { 1 } = 0 , x _ { 2 } = 1$ , and add $x _ { 3 } = \varepsilon ,$ , where $0 < \varepsilon < 1$ . For the original two-point set,

$$
\mu _ { \mathrm { E n e r g y } } ( D ; s ) = - 1 .
$$

For the three-point set,

$$
\mu _ { \mathrm { E n e r g y } } ( D ^ { \prime } ; s ) = - \frac { 1 } { 3 } \left( 1 + \varepsilon ^ { - s } + ( 1 - \varepsilon ) ^ { - s } \right) ,
$$

which is smaller than −1 for sufficiently small $\varepsilon .$ Hence A1 fails.

The twin property also fails in the natural sense that exact duplicates are not in the finite domain of the functional: adding a duplicate creates a zero off-diagonal distance, making $D _ { i j } ^ { - s }$ singular. Thus the score is not a well-defined real-valued functional on multisets with twins. In ε-regularized implementations, the duplicate contributes a large negative penalty and therefore changes the score.

Circles. For fixed $\tau \geq 0$

$$
\mu _ { \mathrm { C i r c l e s } } ( D ; \tau ) = \operatorname* { m a x } _ { S \subseteq X } \left\{ | S | : D _ { i j } > \tau { \mathrm { ~ f o r ~ a l l ~ d i s t i n c t } } x _ { i } , x _ { j } \in S \right\} .
$$

Size monotonicity holds because, after adding a new point, every feasible subset of the original set remains feasible. Hence the maximum cannot decrease.

The twin property holds. Suppose the added point $x _ { n + 1 }$ is a duplicate of $x _ { i }$ . Since $D _ { i , n + 1 } ^ { \prime } = 0 \leq \tau$ no feasible subset can contain both $x _ { i }$ and $x _ { n + 1 }$ . Moreover, any feasible subset containing $x _ { n + 1 }$ but not $x _ { i }$ can replace $x _ { n + 1 } { \mathrm { ~ b y ~ } } x _ { i }$ , because the duplicate has the same distances to all other points. Therefore the maximum feasible cardinality is unchanged.

Distance monotonicity holds. If $D _ { i j } ^ { \prime } \geq D _ { i j }$ for all $i \neq j$ , then any subset satisfying $D _ { i j } > \tau$ also satisfies $D _ { i j } ^ { \prime } > \tau$ . Thus the family of feasible subsets can only grow, and

$$
\mu _ { \mathrm { C i r c l e s } } ( D ^ { \prime } ; \tau ) \geq \mu _ { \mathrm { C i r c l e s } } ( D ; \tau ) .
$$

Continuity fails because the functional is thresholded and integer-valued. For $n = 2$ and $\tau = 1$ , let

$$
D ^ { ( m ) } = ( { 0 \atop 1 + 1 / m }  \quad { 1 + 1 / m \atop 0 } ) , \qquad D = ( { 0 \atop 1 }  \quad { 1 \atop 0 } ) .
$$

Then $D ^ { ( m ) } \to D$ , but $\mu _ { \mathrm { C i r c l e s } } ( D ^ { ( m ) } ; \tau ) = 2$ for all $m .$ , whereas $\mu _ { \mathrm { C i r c l e s } } ( D ; \tau ) = 1$ , because the constraint is strict: $D _ { 1 2 } > \tau$

Vendi score. Let $K \succeq 0$ have diagonal entries $K _ { i i } = 1$ , and let $\lambda _ { 1 } , \ldots , \lambda _ { n }$ be the eigenvalues of $K / n$ . For $q > 0 , q \neq 1$

$$
\mu _ { \mathrm { V e n d i } } ( K ; q ) = \left( \sum _ { i = 1 } ^ { n } \lambda _ { i } ^ { q } \right) ^ { 1 / ( 1 - q ) } ,
$$

with the Shannon limit at $q = 1$

Continuity holds because the eigenvalues of a symmetric matrix are continuous functions of the matrix entries, and the maps

$$
( \lambda _ { i } ) _ { i } \mapsto \left( \sum _ { i } \lambda _ { i } ^ { q } \right) ^ { 1 / ( 1 - q ) } \quad { \mathrm { a n d } } \quad ( \lambda _ { i } ) _ { i } \mapsto \exp \left( - \sum _ { i } \lambda _ { i } \log \lambda _ { i } \right)
$$

are continuous on the probability simplex, with the convention 0 log $0 = 0$

Size monotonicity fails. For two orthogonal items,

$$
K = I _ { 2 } ,
$$

the normalized eigenvalues are $( 1 / 2 , 1 / 2 )$ , so $\mu _ { \mathrm { V e n d i } } ( K ; q ) = 2$ for all $q > 0$ . Add a duplicate of the first item:

$$
K ^ { \prime } = { \binom { 1 } { 0 } } \ 1 0 1 )
$$

The eigenvalues of $K ^ { \prime } / 3$ are $( 2 / 3 , 1 / 3 , 0 )$ . Hence

$$
\mu _ { \mathrm { V e n d i } } ( K ^ { \prime } ; q ) = [ ( 2 / 3 ) ^ { q } + ( 1 / 3 ) ^ { q } ] ^ { 1 / ( 1 - q ) } < 2
$$

for $q \neq 1$ , and the same strict inequality holds at $q = 1$ by the Shannon limit. Thus adding an element can decrease the score. The same example also shows that the twin property fails, since the added element is an exact duplicate but the score changes.

Distance monotonicity is not guaranteed for the Vendi family. To give an explicit counterexample, take $q = 1 / 2$ and define distances by $D _ { i j } = 1 - K _ { i j }$ . Let

$$
K = { \left( \begin{array} { l l l } { 1 } & { 0 . 8 } & { 0 . 4 } \\ { 0 . 8 } & { 1 } & { 0 . 3 } \\ { 0 . 4 } & { 0 . 3 } & { 1 } \end{array} \right) } , \qquad K ^ { \prime } = { \left( \begin{array} { l l l } { 1 } & { 0 . 8 } & { 0 . 3 5 } \\ { 0 . 8 } & { 1 } & { 0 } \\ { 0 . 3 5 } & { 0 } & { 1 } \end{array} \right) } .
$$

Both matrices are positive semidefinite, and $K _ { i j } ^ { \prime } \le K _ { i j }$ for all $i \neq j$ , so the associated distances satisfy $D _ { i j } ^ { \prime } \geq D _ { i j }$ . However,

$$
\mu _ { \mathrm { V e n d i } } ( K ; 1 / 2 ) \approx 2 . 5 0 9 3 , \qquad \mu _ { \mathrm { V e n d i } } ( K ^ { \prime } ; 1 / 2 ) \approx 2 . 4 7 4 7 .
$$

Thus increasing dissimilarities can decrease the Vendi score.

Magnitude. For fixed $t > 0 ,$ , let

$$
Z _ { t } ( D ) _ { i j } = \exp ( - t D _ { i j } ) , \qquad \mu _ { \mathrm { M a g } } ( D ; t ) = { \bf 1 } ^ { \top } Z _ { t } ( D ) ^ { - 1 } { \bf 1 } .
$$

Continuity holds on the domain where $Z _ { t } ( D )$ is nonsingular, since $D \mapsto Z _ { t } ( D )$ is continuous and matrix inversion is continuous on the open set of nonsingular matrices.

Size monotonicity holds under the standard positive-definite assumption. Let $Z$ be the similarity matrix for the original set and suppose the enlarged matrix has the block form

$$
Z ^ { \prime } = \left( \begin{array} { c c } { { Z } } & { { z } } \\ { { z ^ { \top } } } & { { 1 } } \end{array} \right) ,
$$

with $Z ^ { \prime } \succ 0$ . Let $w = Z ^ { - 1 } \mathbf { 1 }$ and let

$$
s = 1 - z ^ { \top } Z ^ { - 1 } z .
$$

Since $Z ^ { \prime } \succ 0$ , the Schur complement satisfies $s > 0$ . The block inverse formula gives

$$
\mathbf { 1 } ^ { \top } ( Z ^ { \prime } ) ^ { - 1 } \mathbf { 1 } = \mathbf { 1 } ^ { \top } Z ^ { - 1 } \mathbf { 1 } + \frac { ( 1 - z ^ { \top } w ) ^ { 2 } } { s } \geq \mathbf { 1 } ^ { \top } Z ^ { - 1 } \mathbf { 1 } .
$$

Therefore magnitude satisfies size monotonicity on this positive-definite domain.

The twin property fails as a real-valued property on multisets because adding an exact duplicate makes two rows and columns of $Z _ { t } ( D )$ identical. Hence $Z _ { t } ( D )$ becomes singular, so the inverse-based magnitude is not defined.

Distance monotonicity fails. Let $t = 1$ , and define two similarity matrices

$$
Z = \left( { \begin{array} { c c c } { 1 } & { 0 . 1 } & { 0 . 3 } \\ { 0 . 1 } & { 1 } & { 0 . 9 } \\ { 0 . 3 } & { 0 . 9 } & { 1 } \end{array} } \right) , \qquad Z ^ { \prime } = \left( { \begin{array} { c c c } { 1 } & { 0 . 1 } & { 0 . 1 } \\ { 0 . 1 } & { 1 } & { 0 . 9 } \\ { 0 . 1 } & { 0 . 9 } & { 1 } \end{array} } \right) .
$$

Both matrices are positive definite. Let $D _ { i j } = - \log Z _ { i j }$ and $D _ { i j } ^ { \prime } = - \log Z _ { i j } ^ { \prime }$ . Since $Z _ { i j } ^ { \prime } \leq Z _ { i j }$ for all $i \neq j ,$ , we have $D _ { i j } ^ { \prime } \geq D _ { i j }$ . However,

$$
\mu _ { \mathrm { M a g } } ( D ; 1 ) = \mathbf { 1 } ^ { \top } Z ^ { - 1 } \mathbf { 1 } = { \frac { 1 5 } { 8 } } = 1 . 8 7 5 ,
$$

whereas

$$
\mu _ { \mathrm { M a g } } ( D ^ { \prime } ; 1 ) = \mathbf { 1 } ^ { \top } ( Z ^ { \prime } ) ^ { - 1 } \mathbf { 1 } = \frac { 1 7 5 } { 9 4 } \approx 1 . 8 6 1 7 .
$$

Thus increasing pairwise distances can reduce magnitude.

Remark on the assumptions for magnitude. The proof of size monotonicity above uses positive definiteness of the enlarged similarity matrix. If Definition 6 only assumes nonsingularity, size monotonicity is not guaranteed. Thus the $\mathbf { A } 1 ~ \checkmark$ for magnitude should be read as holding under the positive-definite finite-metric/kernel setting, not under mere nonsingularity.

## B Details of Figure 2

For each domain in Figure 2, we randomly sample data instances from commonly-used datasets, compute their representations using a standard feature extractor, and estimate the empirical distribution from 100,000 randomly sampled pairwise distances. The goal of this experiment is to illustrate that commonly used high-dimensional representations in different modalities often lead to strongly concentrated pairwise distance distributions.

Images. Images are commonly represented by deep neural networks trained for image recognition. We randomly sample RGB images from ImageNet (research-only access terms) [7] and extract features using a ResNet-50 encoder (TorchVision, BSD-3-Clause; pretrained weights subject to ImageNet access terms) [10]. Specifically, each image is mapped to a 2048-dimensional feature vector from the penultimate representation layer. We then compute pairwise cosine distances between these image embeddings. Cosine distance between deep image features is widely used in image retrieval, clustering, and diversity evaluation. As shown in Figure 2(a), the resulting distance distribution is sharply concentrated.

Natural language text. For natural language, modern large language models provide highdimensional semantic embeddings that are widely used in similarity search, retrieval-augmented generation, and text evaluation. We use question-answer pairs from the 2WikiMultiHopQA dataset (Apache-2.0) [12] and embed them with the Qwen3-8B model (Apache-2.0) [34], which produces 4096-dimensional dense representations. Pairwise dissimilarity is measured by cosine distance. Figure 2(b) shows that the empirical distance distribution for textual embeddings is also strongly concentrated.

Molecular structures. In cheminformatics, molecular diversity is often assessed using handcrafted molecular fingerprints rather than neural embeddings. We represent molecules sampled from the ChEMBL database (CC BY-SA 3.0) [9] using extended-connectivity fingerprints (ECFP; computed with RDKit, BSD-3-Clause) [25], a widely used binary representation that encodes local molecular substructures. We use fixed-length 2048-dimensional fingerprints and compute pairwise Tanimoto distances [28], a standard dissimilarity measure for molecular fingerprints. As illustrated in Figure 2(c), the resulting molecular distances also exhibit strong concentration. This suggests that high-dimensional distance concentration arises not only in learned neural embeddings, but also in classical sparse or binary representations.

Overall, Figure 2 demonstrates that distance concentration is a recurring empirical phenomenon across modalities, representation types, and distance functions, and the precise location and shape of the distributions depend on the domain and representation.

## C More Examples of Diversity Profiles

In addition to the natural-language example in Figure 3, we provide two further real examples of diversity profiles on different data modalities: images (Figure 4) and molecular structures (Figure 5). For each modality, we use the same data source, representation, and distance function as described in Appendix B. Each of sets A and B consists of 50 data items sampled uniformly at random.

![](images/eebae2755af90c5491f3f6d3e3d54d7f2b76b9f490525c17d174432c2d3aac4a.jpg)  
Figure 5: Diversity profiles of two real molecular sample sets.
# HyperANFIS: Enhancing Rule Representation and Interpretability in Adaptive Neuro-Fuzzy Systems via Hyperbolic Geometry

Haoran Pei<sup>1</sup>, Zhao Su<sup>1</sup>, Zetao Lin<sup>1</sup>, Haoran Li<sup>2</sup>, Jun Shen<sup>3</sup>, Qi Zhu<sup>4</sup>, Lan Guo<sup>1</sup>, Qingguo Zhou<sup>1</sup>, Binbin Yong<sup>1∗</sup>

<sup>1</sup>school of Information Science and Engineering, Lanzhou University <sup>2</sup>Department of Data Science and AI, Monash University <sup>3</sup>School of Computing and Information Technology, University of Wollongong <sup>4</sup>College of Artificial Intelligence, Nanjing University of Aeronautics and Astronautics

## Abstract

The adaptive neuro-fuzzy inference system (ANFIS) is an interpretable reasoning framework capable of generating explicit IF-THEN fuzzy rules, making it suitable for tasks requiring transparent reasoning. However, existing ANFIS models generally construct rule antecedents and perform inference in Euclidean space, limiting their representational capacity and predictive performance. To address this issue, we propose Hyperbolic ANFIS (HyperANFIS), a hyperbolic extension of ANFIS. HyperANFIS preserves the fuzzy semantics and core architecture of conventional ANFIS while performing rule-prototype learning, rule activation, and consequent aggregation in hyperbolic space. It also retains the ability to generate interpretable IF-THEN rules. By exploiting the representational properties of hyperbolic geometry, HyperANFIS strengthens the fuzzy inference process, thereby improving predictive accuracy, inter-rule collaboration, and the credibility of its interpretable rules. Experimental results show that HyperANFIS consistently outperforms the standard ANFIS baseline and various ANFIS variants across all datasets, while also generating higher-quality fuzzy rules.

## Introduction

Adaptive neuro-fuzzy inference systems (ANFIS) are data-adaptive machine-learning models that combine the parameter-learning capability of neural networks with the rule-based inference of fuzzy systems. ANFIS models nonlinear input-output relationships through fuzzy IF-THEN rules parameterized by membership functions (Jang 1993). Because its rule base and rule activations can be inspected, ANFIS is particularly appealing in applications such as healthcare, industry, and finance, where transparent decisionmaking is important.

Despite these advantages, conventional ANFIS architectures face structural limitations as the number of input variables increases or as the data exhibit complex relational structures (Jin et al. 2024). In a classical ANFIS, the antecedent of each rule is typically formed by conjoining univariate membership functions defined on individual features in Euclidean space, with these membership functions treated independently, while the rule firing strength is obtained by aggregating their membership values. As the numbers of input variables and membership functions increase, the number of rules grows combinatorially. Meanwhile, the increasingly numerous feature representations become crowded and constrained in Euclidean space, which can afect model performance and rule-generation capacity.

![](images/9a7b36d48426e4773316cc36ee721187fa994eaac482f9e7f241c54be7fcbc07.jpg)  
Figure 1: Comparison of rule constraints in Euclidean and hyperbolic spaces. The capacity of Euclidean space is limited, and the rules become severely compressed as the number of features increases. In hyperbolic space, spatial capacity grows continuously, the rules are not constrained by the available space.

This limitation is particularly pronounced when the data or decision regions exhibit hierarchical, tree-like, or highly nonuniform structures. In such cases, Euclidean geometry may provide a weak inductive bias because it does not naturally capture the expanding representational capacity required by increasingly fine-grained branches in a hierarchy (Sinha et al. 2024). Such structures are widespread across a variety of real-world applications, including healthcare and industrial systems (Koo 2024). Therefore, it is important to develop ANFIS models in alternative geometric spaces that can enrich rule representations without sacrificing the transparency of fuzzy IF-THEN reasoning.

Hyperbolic representation learning provides a theoretically motivated geometric alternative for modeling hierarchical, tree-like, and highly non-uniform structure (Ganea, Bécigneul, and Hofmann 2018). Owing to its constant negative curvature, the volume of a metric ball in hyperbolic space grows exponentially with its radius. This property allows a low-dimensional embedding to allocate increasing representational capacity to progressively branching hierarchies (Nickel and Kiela 2017; Ganea, Bécigneul, and Hofmann 2018). Prior work further suggests that hyperbolic geometry can provide a useful inductive bias when data contain explicit or latent hierarchical and non-uniform structure (Sinha et al. 2024).

To this end, we propose Hyperbolic ANFIS (HyperAN-FIS), a geometry-aware neuro-fuzzy model. HyperANFIS reformulates the ANFIS architecture by exploiting the properties of hyperbolic manifolds. It represents fuzzy rules using global geodesic prototypes, derives rule activation strengths from geodesic distances, and aggregates rule consequents on a hyperbolic manifold. This design preserves the scalar fuzzy semantics and functional roles of ANFIS while providing each rule with an explicit geometric interpretation. Under matched experimental settings, HyperANFIS achieves better predictive performance than its Euclidean ANFIS baseline and produces more credible rule representations.

Our contributions are as follows:

• We propose HyperANFIS, a model that learns global IF-THEN rules on a hyperbolic manifold and makes accurate predictions based on these rules.

• HyperANFIS unifies rule matching, local consequent construction, and consequent aggregation on a hyperbolic manifold, ofering a new perspective on mitigating the limitations of Euclidean geometry in ANFIS.

• Experiments on multiple real-world datasets show that HyperANFIS outperforms the compared ANFIS models and produces more credible IF-THEN rules.

## Related Work

HyperANFIS is a model that reconstructs ANFIS in hyperbolic space to derive highly interpretable rules; accordingly, research on interpretable machine learning, neuro-fuzzy systems, and hyperbolic models provides the theoretical foundation for this work.

## Interpretable Machine Learning

ANFIS combine adaptive parameter learning with Takagi-Sugeno fuzzy IF-THEN inference and have long been widely used in nonlinear approximation, prediction, control, and classification tasks (Jang 1993). Recent studies have explored diferent strategies to improve the scalability and interpretability of ANFIS. For example, Jin et al. simplify high-dimensional ANFIS by selecting important rules and removing similar ones (Jin et al. 2024), while UNFIS learns unstructured fuzzy rules that relax the requirement of evaluating every input variable within each rule (Salimi-Badr 2024). Beyond rule simplification and structural relaxation, recent structure-learning approaches, such as SL-ANFIS-LSTM (Su et al. 2025), introduce adaptive fuzzy structure optimization to improve the flexibility of ANFIS-based models. More recently, KANFIS further explores a neuro-symbolic formulation based on additive function decomposition and sparse masking to reduce rule complexity and improve interpretability of fuzzy systems (Yong et al. 2026). These studies address important challenges such as rule explosion, redundancy, and antecedent complexity from diferent perspectives. However, existing approaches mainly focus on rule selection, structural formulation, or functional decomposition, while the geometric organization of rule representations remains largely unexplored. HyperANFIS explores the possibility of realizing ANFIS in hyperbolic space, providing a geometric perspective for improving rule representation.

## ANFIS and Neuro-Fuzzy Learning

ANFIS combine adaptive parameter learning with Takagi-Sugeno fuzzy IF-THEN inference and have long been widely used in nonlinear approximation, prediction, control, and classification tasks (Jang 1993). Recent studies on rule-base simplification, redundancy reduction, and the trade-of between predictive performance and interpretability indicate that ANFIS remains an active research area. For example, Jin et al. simplify high-dimensional ANFIS by selecting important rules and removing similar ones (Jin et al. 2024), whereas UNFIS learns unstructured fuzzy rules that do not require every input variable to be evaluated within each rule (Salimi-Badr 2024). More recently, KANFIS introduces a neuro-symbolic formulation based on additive function decomposition and sparse masking to reduce rule complexity and improve the interpretability of fuzzy systems (Yong et al. 2026). These studies address important issues such as rule explosion, rule redundancy, and excessively complex antecedents from diferent perspectives. However, existing approaches mainly focus on rule selection, structural formulation, or functional decomposition, while the geometric organization of rule representations remains largely unexplored. HyperANFIS explores the possibility of realizing ANFIS in hyperbolic space, providing a geometric perspective for improving rule representation.

## Theoretical Foundations and Empirical Applications of Hyperbolic Geometry

Hyperbolic geometry is theoretically well suited for representing tree-like and heterogeneous structures. In a space of constant negative curvature, the volume of a geodesic ball grows exponentially with its radius, more closely matching the branching expansion of trees than the polynomial volume growth of Euclidean space. The relationship among negative curvature, hierarchical organization, and heterogeneous networks has been extensively studied in complex-network theory (Krioukov et al. 2010).

In representation learning, Poincaré embeddings have demonstrated that latent hierarchies can be compactly encoded in low-dimensional hyperbolic space (Nickel and Kiela 2017); related work on hyperbolic neural networks has provided a systematic formulation of neural computation based on exponential maps, logarithmic maps, and gyrovector operations (Ganea, Bécigneul, and Hofmann 2018), while hyperbolic graph convolutional networks have shown that such operations can preserve the hierarchical and scale-free properties of graph structures during message passing (Chami et al. 2019). Taken together, these theoretical results suggest that hyperbolic geometry may improve representation learning when the underlying task involves structured relationships. In addition, some studies have used hyperbolic objectives to encode label hierarchies, thereby reducing distortion in visual representations (Sinha et al. 2024); others have combined spherical and hyperbolic components to

![](images/015b2ce9e8b4e6eea58a94ffad90f84a7e8f1ec943f123cfdce01534600372e5.jpg)  
Figure 2: Architecture of the HyperANFIS model. The upper-left panel presents the complete model architecture. (a), (b), and (c) illustrate, respectively, the mapping of samples and rule prototypes into hyperbolic space, the generation of fuzzy rules in hyperbolic space, and the rule-based inference process.

interpretable models.

model cyclical homophily and hierarchical social influence in social networks, respectively (Iyer et al. 2024). In visual learning, hyperbolic visual hierarchy mapping supports classification and dense prediction tasks (Kwon et al. 2024), and hyperbolic distances have also been used in open-world object detection to capture superclass relationships (Doan et al. 2024). This line of research has further expanded to multiple modalities, with hyperbolic representations applied to compositional vision-language learning and genomic sequence modeling (Pal et al. 2025; Khan, Chlenski, and Pe’er 2025). Recent work has also proposed low-distortion, GPUcompatible hyperbolic tree embeddings (Van Spengler and Mettes 2025) and investigated hyperbolic embeddings of supervised models, including decision trees (Nock et al. 2024); continuous hyperbolic latent representations have been introduced for hierarchical open-vocabulary 3D scenes (Weijler et al. 2026), and hyperbolic representation learning has been explored for reasoning over the hierarchical structure of program syntax (Zhou 2026).

However, hyperbolization methods have not been widely studied in intrinsically interpretable models. To the best of our knowledge, this work is the first to reconstruct ANFIS in hyperbolic space, and our experiments demonstrate its efectiveness.

Overall, these studies suggest that reconstructing ANFIS in hyperbolic space has a solid research basis, promising application potential, and methodological feasibility. Building on this foundation, this work further investigates the construction and performance of HyperANFIS, with the aim of providing new insights into the structural refinement of AN-FIS and the application of hyperbolic space in intrinsically

## Methodology

As shown in Figure 2, HyperANFIS extends the neuro-fuzzy inference process to hyperbolic space, leveraging negatively curved geometry to represent fuzzy rules and perform inference. This design improves the model’s representational capacity and supports more discriminative, stable, and auditable fuzzy rules. This section details the architecture of HyperANFIS and its key methodological components.

## Overall Architecture of HyperANFIS

As shown in Figure 2, HyperANFIS is organized as five functional layers. Given a standardized input, the first layer maps the input and all rule centers from a shared origin tangent space to the selected hyperbolic representation, and then evaluates their intrinsic geodesic distances. The second layer converts each sample-to-rule distance into one scalar radial membership. The third layer normalizes the log-memberships across rules to obtain firing strengths. The fourth layer constructs a rule-specific consequent from the intrinsic local coordinates of the sample. The fifth layer maps these consequents to the output manifold, aggregates them through a firing-weighted Fréchet mean, and decodes the aggregate for classification or regression. All trainable parameters are optimized jointly through the resulting diferentiable computation graph.

## Hyperbolic Mapping and Rule Pullback

Let $\widehat { \mathbf { x } } _ { i } \in \mathbf { R } ^ { D }$ denote the standardized input of sample i, and let ${ \bf a } _ { r } \in { \bf R } ^ { D }$ be the trainable tangent center of rule r. With

an input scale $s > 0$ and tangent bound $\tau > 0$ , HyperANFIS forms

$$
\begin{array} { r } { \mathbf { v } _ { i } = \mathrm { c l i p } _ { \tau } ( s \widehat { \mathbf { x } } _ { i } ) , \qquad \mathbf { \overline { { a } } } _ { r } = \mathrm { c l i p } _ { \tau } ( \mathbf { a } _ { r } ) , } \end{array}\tag{1}
$$

where

$$
\mathrm { c l i p } _ { \tau } ( \mathbf { v } ) = \mathbf { v } \operatorname* { m i n } \left( 1 , \frac { \tau } { \| \mathbf { v } \| } \right) .\tag{2}
$$

The clipping is radial, so it preserves direction while bounding the represented geodesic radius. Define $\varrho ( \mathbf { v } ) = { \sqrt { c } } \| \mathbf { v } \|$ where −c is the sectional curvature and $c > 0$

For the Lorentz representation, the origin is $\begin{array} { r l } { \mathbf { o } _ { \mathrm { { L } } } } & { { } = } \end{array}$ $( c ^ { - 1 / 2 } , 0 , \dots , 0 )$ . The origin exponential map is

$$
\exp _ { \mathbf { o } _ { \mathrm { L } } } ^ { \mathrm { L } } ( \mathbf { v } ) = \left( c ^ { - 1 / 2 } \cosh \left( \varrho ( \mathbf { v } ) \right) , \frac { \sinh \left( \varrho ( \mathbf { v } ) \right) } { \varrho ( \mathbf { v } ) } \mathbf { v } \right) .\tag{3}
$$

For the Poincaré representation, the origin is $\mathbf { o } _ { \mathrm { P } } = \mathbf { 0 }$ . Under the shared tangent convention used by HyperANFIS, its exponential map is

$$
\exp _ { \mathbf { o } \mathrm { P } } ^ { \mathrm { P } } ( \mathbf { v } ) = \frac { \operatorname { t a n h } \left( \varrho ( \mathbf { v } ) / 2 \right) } { \varrho ( \mathbf { v } ) } \mathbf { v } .\tag{4}
$$

These maps follow the standard negative-curvature constructions used in hyperbolic representation learning (Ganea, Bé- cigneul, and Hofmann 2018; Nickel and Kiela 2018). Hyper-ANFIS selects either $\mathcal { M } = \mathrm { L }$ or $\mathcal { M } = \mathrm { P }$ and constructs

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { \mathcal { M } } = \exp _ { \mathbf { o } _ { \mathcal { M } } } ^ { \mathcal { M } } ( \mathbf { v } _ { i } ) , \qquad \mathbf { p } _ { r } ^ { \mathcal { M } } = \exp _ { \mathbf { o } _ { \mathcal { M } } } ^ { \mathcal { M } } ( \overline { { \mathbf { a } } } _ { r } ) . } \end{array}\tag{5}
$$

The two representations are alternative coordinate choices and share the same trainable tangent centers.

For rule inspection, let $x _ { i j } ^ { \mathrm { r a w } }$ be a raw feature with trainingset location $\nu _ { j }$ and scale $\xi _ { j } ~ > ~ 0$ . The preprocessing and inverse center map are

$$
\begin{array} { r c l } { { \widehat x _ { i j } } } & { { = } } & { { ( x _ { i j } ^ { \mathrm { r a w } } - \nu _ { j } ) / \zeta _ { j } , } } \\ { { \widetilde a _ { r j } } } & { { = } } & { { \nu _ { j } + \zeta _ { j } \overline { { { a } } } _ { r j } / s . } } \end{array}\tag{6}
$$

Thus, $\widetilde { \mathbf { a } } _ { r }$ is a readable representative center in the original feature coordinates. It is not a coordinate-wise Euclidean rule boundary. Exact rule evaluation still replays preprocessing, clipping, hyperbolic mapping, and intrinsic distance computation, as detailed in Appendix A.

## Geodesic Fuzzy Rule Construction

HyperANFIS defines each antecedent as one geodesic region around a global rule prototype. For Lorentz points ${ \bf z } , { \bf p _ { \Delta \Sigma } } \in$ $\mathcal { H } _ { c } ^ { D }$ , define

$$
\langle \mathbf { z } , \mathbf { p } \rangle _ { \mathrm { L } } = - z _ { 0 } p _ { 0 } + \sum _ { j = 1 } ^ { D } z _ { j } p _ { j } .\tag{7}
$$

The corresponding intrinsic distance is

$$
d _ { c } ^ { \mathrm { L } } ( \mathbf { z } , \mathbf { p } ) = c ^ { - 1 / 2 } \mathrm { a r c o s h } \left( - c \left. \mathbf { z } , \mathbf { p } \right. _ { \mathrm { L } } \right) .\tag{8}
$$

For Poincaré points ${ \bf z } , { \bf p } \in B _ { c } ^ { D }$ , let $\Delta _ { c } ( \mathbf { q } ) \ : = \ : 1 - c \| \mathbf { q } \| ^ { 2 }$ Their intrinsic distance is

$$
d _ { c } ^ { \mathrm { P } } ( \mathbf { z } , \mathbf { p } ) = c ^ { - 1 / 2 } \mathrm { a r c o s h } \left( 1 + \frac { 2 c \| \mathbf { z } - \mathbf { p } \| ^ { 2 } } { \Delta _ { c } ( \mathbf { z } ) \Delta _ { c } ( \mathbf { p } ) } \right) .\tag{9}
$$

For either representation, the sample-to-rule distance is

$$
\delta _ { i r } ^ { \mathcal { M } } = d _ { c } ^ { \mathcal { M } } \left( \mathbf { z } _ { i } ^ { \mathcal { M } } , \mathbf { p } _ { r } ^ { \mathcal { M } } \right) .\tag{10}
$$

Each rule has one learnable intrinsic scale. HyperANFIS parameterizes it with an unconstrained logit $\beta _ { r } \mathbf { \hat { \Omega } }$

$$
\sigma _ { r } = \sigma _ { \mathrm { m i n } } + ( \sigma _ { \mathrm { m a x } } - \sigma _ { \mathrm { m i n } } ) \mathrm { s i g m o i d } ( \beta _ { r } ) .\tag{11}
$$

The dimension-normalized distance and Gaussian logmembership are

$$
\chi _ { i r } = \frac { \delta _ { i r } ^ { M } } { \sqrt { D } \sigma _ { r } } , \qquad \ell _ { i r } = - \frac { 1 } { 2 } \chi _ { i r } ^ { 2 } , \qquad \mu _ { i r } = \exp ( \ell _ { i r } ) .\tag{12}
$$

Consequently, $\sqrt { D } \sigma _ { r }$ is the geodesic radius at which $\mu _ { i r } =$ $\exp ( - \mathrm { \hat { 1 } / 2 } )$ ). The implementation also supports a generalized Bell kernel, whose exact form is given in Appendix B.

The third layer normalizes the log-memberships directly:

$$
\overline { { w } } _ { i r } = \frac { \exp ( \ell _ { i r } ) } { \sum _ { q = 1 } ^ { R } \exp ( \ell _ { i q } ) } .\tag{13}
$$

This construction yields one scalar activation for each sample-rule pair. It therefore represents a global geodesic fuzzy rule rather than a product of independent coordinatewise memberships.

## Intrinsic Consequent Aggregation

Let $H \ = \ \operatorname* { m a x } ( 2 , O )$ be the output-manifold dimension, where O is the task output dimension. HyperANFIS first expresses sample i relative to rule prototype r through the intrinsic local coordinate

$$
\mathbf { e } _ { i r } = [ \mathcal { T } _ { \mathbf { p } _ { r } ^ { \mathcal { M } }  \mathbf { o } _ { \mathcal { M } } } ( \log _ { \mathbf { p } _ { r } ^ { \mathcal { M } } } ^ { \mathcal { M } } ( \mathbf { z } _ { i } ^ { \mathcal { M } } ) ) ] _ { \mathrm { s p } } ,\tag{14}
$$

where $\tau$ denotes parallel transport and $[ \cdot ] _ { \mathrm { s p } }$ retains the D spatial tangent coordinates. A first-order rule consequent is

$$
\begin{array} { r c l } { \mathbf { u } _ { i r } } & { = } & { \mathrm { c l i p } _ { \tau } \left( \mathbf { b } _ { r } + \mathbf { W } _ { r } \mathbf { e } _ { i r } \right) , } \\ { \mathbf { q } _ { i r } ^ { \mathcal { M } } } & { = } & { \mathrm { e x p } _ { \mathbf { o } \mathcal { M } } ^ { \mathcal { M } } ( \mathbf { u } _ { i r } ) . } \end{array}\tag{15}
$$

where ${ \bf b } _ { r } \in { \bf R } ^ { H }$ and ${ \mathbf W } _ { r } \in { \mathbf R } ^ { H \times D }$ . The zero-order variant omits $\mathbf { W } _ { r } \mathbf { e } _ { i r }$

The fifth layer aggregates the manifold-valued consequents through their weighted Fréchet mean (Lou et al. 2020):

$$
\mathbf { m } _ { i } ^ { \mathcal { M } } = \underset { \mathbf { m } \in \mathcal { M } } { \operatorname { \arg m i n } } \sum _ { r = 1 } ^ { R } \overline { { w } } _ { i r } d _ { c } ^ { \mathcal { M } } ( \mathbf { m } , \mathbf { q } _ { i r } ^ { \mathcal { M } } ) ^ { 2 } ,\tag{16}
$$

HyperANFIS computes this mean by a fixed number of diferentiable Karcher refinements. For classification, trainable class tangents $\mathbf { g } _ { k }$ define class prototypes $\mathbf { c } _ { k } ^ { \mathcal { M } } \mathbf { \Psi } =$ $\exp _ { \mathbf { o } \mathcal { M } } ^ { \mathcal { M } } ( \mathrm { c l i p } _ { \tau } ( \mathbf { g } _ { k } ) )$ , and the class logits are

$$
\begin{array} { r } { s _ { i k } = - d _ { c } ^ { \mathcal { M } } ( \mathbf { m } _ { i } ^ { \mathcal { M } } , \mathbf { c } _ { k } ^ { \mathcal { M } } ) ^ { 2 } . } \end{array}\tag{17}
$$

For regression, the prediction is obtained in the origin tangent chart:

$$
\widehat { \mathbf { y } } _ { i } = \left[ \log _ { \mathbf { o } _ { \mathcal { M } } } ^ { \mathcal { M } } ( \mathbf { m } _ { i } ^ { \mathcal { M } } ) \right] _ { 1 : O } .\tag{18}
$$

<table><tr><td>Model</td><td colspan="3">Spambase</td><td colspan="3">Car</td><td colspan="3">Zoo</td><td colspan="3">WDBC</td><td colspan="3">NSL-KDD</td></tr><tr><td></td><td> $\operatorname { A c c } .$ </td><td>F1</td><td>Recall</td><td>Acc.</td><td>F1</td><td>Recall</td><td> $\operatorname { A c c } .$ </td><td>F1</td><td>Recall</td><td>Acc.</td><td>F1</td><td>Recall</td><td> $\operatorname { A c c } .$ </td><td>F1</td><td>Recall</td></tr><tr><td>FSRE-AdaTSK</td><td>0.7742</td><td>0.7544</td><td>0.7477</td><td>0.5896</td><td>0.2619</td><td>0.2811</td><td>0.3810</td><td>0.2379</td><td>0.3095</td><td>0.7544</td><td>0.7304</td><td>0.7262</td><td>0.7192</td><td>0.4235</td><td>0.4384</td></tr><tr><td>FCM-ANFIS</td><td>0.9229</td><td>0.9198</td><td>0.9229</td><td>0.8815</td><td>0.8062</td><td>0.9076</td><td>0.7142</td><td>0.6920</td><td>0.7797</td><td>0.9035</td><td>0.8892</td><td>0.9712</td><td>0.7932</td><td>0.5878</td><td>0.5910</td></tr><tr><td>IT2-ANFIS</td><td>0.9121</td><td>0.9088</td><td>0.9130</td><td>0.9104</td><td>0.8469</td><td>0.9086</td><td>0.8095</td><td>0.5946</td><td>0.6607</td><td>0.9123</td><td>0.9037</td><td>0.8958</td><td>0.7622</td><td>0.5856</td><td>0.5842</td></tr><tr><td>PSO-ANFIS</td><td>0.9044</td><td>0.8996</td><td>0.8985</td><td>0.8150</td><td>0.6517</td><td>0.9364</td><td>0.9047</td><td>0.7809</td><td>0.8095</td><td>0.8947</td><td>0.8889</td><td>0.8968</td><td>0.7620</td><td>0.5602</td><td>0.5485</td></tr><tr><td>ANFIS</td><td>0.9120</td><td>0.9080</td><td>0.9075</td><td>0.8998</td><td>0.8277</td><td>0.8731</td><td>0.8413</td><td>0.6875</td><td>0.7500</td><td>0.9152</td><td>0.9067</td><td>0.8958</td><td>0.7798</td><td>0.5952</td><td>0.5935</td></tr><tr><td>HyperANFIS (Ours)</td><td>0.9251</td><td>0.9215</td><td>0.9251</td><td>0.9181</td><td>0.8827</td><td>0.9449</td><td>0.9524</td><td>0.8186</td><td>0.8571</td><td>0.9854</td><td>0.9843</td><td>0.9835</td><td>0.8038</td><td>0.6709</td><td>0.6281</td></tr></table>

Table 1: Performance comparison of HyperANFIS with conventional ANFIS and other ANFIS variants. The best available result for each metric on each dataset is shown in bold. HyperANFIS achieved the best performance in all experimental results.

## Forward Computation

For a standardized input $\mathbf { x } _ { i } .$ the complete HyperANFIS forward computation is summarized as

$$
\begin{array} { r c l } { \left\{ ( \overline { { w } } _ { i r } , \mathbf { q } _ { i r } ^ { \mathcal { M } } ) \right\} _ { r = 1 } ^ { R } } & { = } & { \mathcal { R } _ { \boldsymbol { \Theta } , \mathcal { M } } ( \mathbf { x } _ { i } ) , } \\ { \mathbf { m } _ { i } ^ { \mathcal { M } } } & { = } & { \displaystyle \operatorname * { a r g m i n } _ { \mathbf { m } \in \mathcal { M } } \sum _ { r = 1 } ^ { R } \overline { { w } } _ { i r } d _ { c } ^ { \mathcal { M } } ( \mathbf { m } , \mathbf { q } _ { i r } ^ { \mathcal { M } } ) ^ { 2 } , } \\ { \widehat { \mathbf { y } } _ { i } } & { = } & { \mathcal { D } ( \mathbf { m } _ { i } ^ { \mathcal { M } } ) . } \end{array}\tag{19}
$$

Here, $\mathcal { R } _ { \Theta , \mathcal { M } }$ denotes hyperbolic rule inference and produces the normalized firing strength $\overline { { w } } _ { i r }$ and intrinsic consequent $\mathbf { q } _ { i r } ^ { \mathcal { M } }$ for each rule. These consequents are then aggregated through their activation-weighted Fréchet mean, and D decodes the resulting manifold point into the task prediction. Detailed definitions and optimization procedures are provided in the Appendix C.

## Experiments

To systematically evaluate whether hyperbolic geometric modeling improves the predictive performance and rule interpretability of ANFIS, we compare HyperANFIS with conventional ANFIS and other representative ANFIS variants across multiple datasets. The selected datasets cover several application domains, including healthcare and finance, and vary in complexity, thereby providing a broad experimental basis for model evaluation. In addition, we use the Wisconsin Diagnostic Breast Cancer (WDBC) dataset as a case study and analyze the generated rules from multiple perspectives, providing further evidence of the improved rule interpretability achieved by HyperANFIS.

## Comparison Experiments

We compared HyperANFIS with conventional ANFIS, which served as the primary Euclidean baseline, and four representative neuro-fuzzy models. Conventional ANFIS adopts a dimension-wise Takagi-Sugeno inference structure and learns its antecedent and consequent parameters from the training data (Zhang and Chen 2024). FSRE-AdaTSK integrates feature selection, rule extraction, and parameter finetuning within an adaptive TSK fuzzy system (Xue et al. 2022). FCM-ANFIS uses fuzzy c-means clustering to initialize fuzzy partitions and rule prototypes (Knaiber and Alawieh 2023). In our implementation, these prototypes were subsequently refined through gradient-based optimization. IT2- ANFIS represents uncertainty in rule membership by introducing lower and upper interval type-2 membership bounds into its coordinate-wise rule antecedents (Zand, Katebi, and Yaghmaei-Sabegh 2024). In our implementation, a midpoint approximation was obtained by averaging the lower and upper bounds of the corresponding rule firing strengths. Finally, PSO-ANFIS uses particle swarm optimization to optimize the trainable parameters of ANFIS (Vesović, Jovanović, and Zarić 2024). In our implementation, PSO provided a supervised warm start for all trainable parameters, followed by a common Adam-based refinement stage.

The five datasets cover email spam detection (Spambase), vehicle acceptability evaluation (Car), animal category classification (Zoo), breast cancer diagnosis (WDBC), and network intrusion detection (NSL-KDD). They encompass both binary and multiclass classification tasks and difer in sample size, feature dimensionality, feature characteristics, and class structure, providing a diverse test bed for evaluating model performance across application domains and data distributions. For each random seed, all methods used consistent data preprocessing, data partitions, and evaluation metrics, and the reported results were averaged over five predefined random seeds.

As shown in Table 1, HyperANFIS outperforms conventional ANFIS and the representative neuro-fuzzy baselines on all five datasets. Compared with conventional ANFIS, HyperANFIS improves the average accuracy, Macro-F1 score, and Recall by 0.0473, 0.0706, and 0.0638, respectively. As discussed in the following sections, HyperANFIS also yields more reliable predictions than the comparison models in terms of the output rule attributes. These results demonstrate that geometry-aware rule inference can substantially enhance the predictive capability of ANFIS and indicate that hyperbolic computation is a promising direction for further improving ANFIS and its variants.

These improvements are particularly pronounced on Zoo and WDBC. On the Zoo dataset, HyperANFIS improves accuracy, Macro-F1, and Recall by 11.11, 13.11, and 0.1071 percentage points, respectively; on WDBC, the corresponding improvements reach 7.02, 7.76, and 0.0877 percentage points. Zoo exhibits a natural hierarchical taxonomic structure, whereas WDBC contains complex and nonuniform diagnostic relationships among its features. These consistent improvements on the two datasets provide empirical support for the efectiveness of negative-curvature geometry in organizing nonuniform and hierarchical rule relationships, thereby leading to performance gains.

<table><tr><td>Rule</td><td>Projected, interpretable IF-THEN rule</td><td>Morphological evidence</td></tr><tr><td>R4</td><td>IF mean compactness is HIGH ( = +3.69), mean con- High compactness, concavity, and concave-point counts describe pro- (c = +3.84). THEN the rule-level conclusion is malignant (M). Dom- (Narasimha, Vasavi, and Kumar 2013). inant coverage: 26.3%.</td><td>cavity is HIGH (č = +3.65), mean concave points is nounced nuclear-boundary indentation and irregularity. Direct breast- HIGH ( = +3.77), and worst concave points is HIGH aspirate morphometry found compactness and concave-point counts to be significantly higher in ductal carcinoma than in benign lesions</td></tr><tr><td>R10</td><td>( = −2.18), and worst radius is LOW ( = −2.37). coverage: 21.1%.</td><td>IF mean radius is LOW ( = —2.24), mean concavity Small nuclei with low concavity and few concave points form a com- is LOW (Č = —1.98), mean concave points is LOW pact, comparatively smooth nuclear phenotype. Direct breast-aspirate morphometry reported lower nuclear size, compactness, and concave- THEN the rule-level conclusion is benign (B). Dominant point counts in benign lesions than in ductal carcinoma (Narasimha, Vasavi, and Kumar 2013).</td></tr><tr><td>R2</td><td>and worst area is LOW ( = —0.70). coverage: 14.9%.</td><td>IF mean radius is LOW ( = —1.11), mean area is Low mean and worst-case radius and area indicate that both typical and LOW ( = —1.21), worst radius is LOW ( = —1.32), extreme nuclear sizes remain small. This size-dominant pattern agrees with cytological evidence showing lower nuclear area, diameter, and THEN the rule-level conclusion is benign (B). Dominant perimeter in benign breast aspirates (Pandian, Ramdas, and Ambroise 2021).</td></tr><tr><td>R6</td><td>and worst radius is LOW ( = —2.37). coverage: 12.3%.</td><td>IF mean concavity is LOW ( = —2.06), radius SE is Low concavity and worst radius support a compact, regular nuclear LOW ( = —2.23), perimeter SE is LOW ( = —2.12), phenotype; low radius and perimeter SE indicate limited within-sample size dispersion. This direction agrees with the greater nuclear size and THEN the rule-level conclusion is benign (B). Dominant size dispersion reported in malignant aspirates (Pandian, Ramdas, and Ambroise 2021).</td></tr><tr><td>R9</td><td>( = +2.05), and worst area is HIGH ( = +1.15). inant coverage: 12.3%.</td><td>IF mean radius is HIGH ( = +1.60), mean area High mean size and high extreme perimeter and area jointly capture nu- is HIGH ( = +1.24), worst perimeter is HIGH clear enlargement and exceptionally large nuclei. Breast-aspirate mor- phometry similarly reported substantially greater nuclear area, perime- THEN the rule-level conclusion is malignant (M). Dom- ter, and diameter in malignant lesions (Pandian, Ramdas, and Ambroise 2021).</td></tr><tr><td>R12</td><td>and worst radius is LOW ( = —1.66). coverage: 6.1%. disease (Abubakar et al. 2025).</td><td>IF mean radius is LOW ( = —2.00), mean area is This mixed phenotype combines small typical and extreme nuclear sizes LOW ( = —1.15), radius SE is HIGH ( = +1.29), with greater radius variation. Small nuclear size supports the benign consequent, while single-cell morphometry confirms that measurable THEN the rule-level conclusion is benign (B). Dominant variation in nuclear size and contour can also occur within benign breast</td></tr></table>

Table 2: Selected HyperANFIS rules learned on the WDBC dataset. Each row represents a single IF-THEN rule together with the corresponding support from the literature. An individual rule is not the prediction for a single sample; rather, its firing strength is influenced by its dominant coverage.

## Interpretability Analysis

To evaluate whether hyperbolic geometry improves the rule interpretability of ANFIS, we compared HyperANFIS with classical ANFIS on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset and analyzed representative IF-THEN rules. WDBC contains 569 instances and 30 real-valued features extracted from digitized images of fine-needle aspirates of breast masses. These features describe nuclear size, texture, and contour morphology, providing explicit cytomorphological semantics for evaluating rule utilization, distinctiveness, and morphological consistency.

As shown in Table 2, HyperANFIS preserves the ability of classical ANFIS to express its inference process through interpretable IF-THEN rules. For readability, the table reports the six rules with the highest dominant coverage on the validation set and retains only three to four representative features for each rule, while the omitted features remain part of the complete antecedent. The learned rules associate malignancy with increased nuclear size and contour irregularity, whereas benign patterns are generally characterized by smaller nuclear size and lower concavity. These patterns are consistent with established findings in breast nuclear mor-

phometry.

## Comparison with Classical ANFIS

We compare the rule characteristics learned by conventional ANFIS and HyperANFIS on the WDBC dataset, which has a complex feature space and strong intrinsic logical structure. As shown in Figure 3, the interpretability of conventional ANFIS deteriorates substantially on this dataset, where the feature complexity and underlying relational structure make rule learning challenging. In contrast, HyperANFIS preserves a high level of interpretability, revealing a clear distinction between the two models. This experiment was conducted with a fixed seed under fair comparison conditions.

In Figure 3(a), classical ANFIS assigns 42.98% of the validation samples to R2 as their dominant rule, whereas the dominant coverage of the second most frequently selected rule drops to 10.53%. In contrast, HyperANFIS distributes dominant coverage across five principal rules: R4 (26.32%), R10 (21.05%), R2 (14.91%), R6 (12.28%), and R9 (12.28%). The remaining rules also participate in inference through nonzero mean firing strengths. Thus, HyperANFIS transforms the single-rule-dominated utilization pattern of classical ANFIS into a broader and more clearly organized multi-rule participation pattern. Figure 3(b) further demonstrates this improvement at the rule-prototype level. Classical ANFIS learns 11 benign prototypes but only one malignant prototype, resulting in a substantial imbalance in class representation. In contrast, HyperANFIS learns seven benign and five malignant prototypes, whose distribution is more coherent. As indicated by the red circles in the figure, some rule prototypes are also clearly mislearned. Figure 3(c) shows that the mean of-diagonal cosine similarity between rule activation vectors increases from 0.006 to 0.432. The rules learned by classical ANFIS are activated in an almost completely mutually exclusive manner, which appears dificult to reconcile with the principles of fuzzy inference. HyperANFIS, in contrast, establishes correlated yet diferentiated relationships among its rules, thereby better conforming to the gradual evidence-combination process underlying fuzzy reasoning. Figure 3(d) provides direct sample-level evidence: the mean entropy-derived efective number of active rules increases from 1.14 for classical ANFIS to 6.67 for HyperANFIS. Classical ANFIS consequently exhibits an almost winner-take-all inference pattern dominated by a single rule, whereas HyperANFIS uses one or more principal rules to guide the inference, with the remaining rules providing supplementary evidence.

ANFIS  
![](images/61b37187f481428bfb42500a7c8b54e53bf08036b825ef5a86c1533793842309.jpg)

![](images/cb2e1c65bd9af17f561fa12cd22636ac0f027e17824912fa52585abef0af02c3.jpg)

(b) Rule prototype coverage in MDS space  
![](images/aecad563afdeaace1b1300afbc7c085b243214b2e028eecca410c3e8acd72a7c.jpg)

![](images/bfa2cb914e04cde90498d5933a41b463cb3216074463c17b6b0d29337e2c1c39.jpg)

(c) Pairwise rule activation similarity  
![](images/09faef4e8b5c142025fd32494ba011730965e63cc9fdd850c79f8ef438c4863e.jpg)

![](images/66bea9b5e0c688e6f9ce5e3345efbc2a35b09e9c9242f304b92c9069439792fa.jpg)

(d) Sample-wise normalized rule activation  
![](images/ab7a5087fea9898d47fdb47793e7de4f2ab8365940bece9e6667317f968ad8cb.jpg)

![](images/e184d41f57f2695e67c553efa3010025fd5f0d7445d7c8c955123f28bc0b2215.jpg)  
Figure 3: Comparison of the interpretability of ANFIS and HyperANFIS on the WDBC dataset. The ANFIS results are shown on the left, whereas the HyperANFIS results are shown on the right. (a) shows the frequency with which each rule is used for prediction, (b) visualizes the relationship between the rules and the dataset, with red circles marking the centers of rules associated with severe errors, (c) shows the correlations among the rules, and (d) illustrates the contribution of each rule to each individual prediction.

These results indicate that hyperbolic geometry enables the learned IF-THEN rules to provide more complete class representations, stronger inter-rule cooperation, and a more trustworthy inference process. Together with the accuracy results reported in Table 1, the findings show that HyperANFIS achieves higher accuracy and more trustworthy predictions while producing higher-quality rules.

## Conclusion

This work investigated how the geometric properties of hyperbolic space can improve ANFIS, an intrinsically interpretable model. By reformulating rule prototypes, rule activation, and consequent aggregation in hyperbolic space while preserving the original IF-THEN semantics, HyperANFIS consistently outperformed classical ANFIS and representative ANFIS variants in accuracy and Macro-F1 across the five evaluated datasets. The WDBC analysis further demonstrated that HyperANFIS learned a more class-diverse, cooperative, and auditable rule system than classical ANFIS. The significance of this work therefore extends beyond the geometric reformulation and performance improvement of classical ANFIS. It provides a new geometric perspective for improving other ANFIS-family models and, more broadly, intrinsically interpretable models.

Abubakar, M.; Fan, S.; Klein, A.; Pfeifer, R. M.; Lawrence, S.; Mutreja, K.; Kimes, T. M.; Richert-Boe, K.; Figueroa, J. D.; Gierach, G. L.; et al. 2025. Spatially resolved singlecell morphometry of benign breast disease biopsy images uncovers quantitative cytomorphometric features predictive of subsequent invasive breast cancer risk. Modern Pathology, 38(7): 100767.

Chami, I.; Ying, Z.; Ré, C.; and Leskovec, J. 2019. Hyperbolic graph convolutional neural networks. Advances in neural information processing systems, 32.

Doan, T.; Li, X.; Behpour, S.; He, W.; Gou, L.; and Ren, L. 2024. Hyp-ow: Exploiting hierarchical structure learning with hyperbolic distance enhances open world object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 1555–1563.

Ganea, O.; Bécigneul, G.; and Hofmann, T. 2018. Hyperbolic neural networks. Advances in neural information processing systems, 31.

Iyer, R. G.; Wang, Y.; Wang, W.; and Sun, Y. 2024. Noneuclidean mixture model for social network embedding. Advances in Neural Information Processing Systems, 37: 111464–111488.

Jang, J.-S. 1993. ANFIS: adaptive-network-based fuzzy inference system. IEEE Transactions on Systems, Man, and Cybernetics, 23(3): 665–685.

Jin, Y.; Cao, W.; Wu, M.; Yuan, Y.; and Shi, Y. 2024. Simplification of ANFIS based on importance-confidence-similarity measures. Fuzzy Sets Syst., 481(C).

Khan, R.; Chlenski, P.; and Pe’er, I. 2025. Hyperbolic genome embeddings. In International Conference on Learning Representations, volume 2025, 73425–73454.

Knaiber, M.; and Alawieh, L. 2023. Bayesian inference using an adaptive neuro-fuzzy inference system. Fuzzy Sets and Systems, 459: 43–66.

Koo, H. 2024. Next Visit Diagnosis Prediction via Medical Code-Centric Multimodal Contrastive EHR Modelling with Hierarchical Regularisation. In Graham, Y.; and Purver, M., eds., Findings of the Association for Computational Linguistics: EACL 2024, 41–55. St. Julian’s, Malta: Association for Computational Linguistics.

Krioukov, D.; Papadopoulos, F.; Kitsak, M.; Vahdat, A.; and Boguná, M. 2010. Hyperbolic geometry of complex networks. Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, 82(3): 036106.

Kwon, H.; Jang, J.; Kim, J.; Kim, K.; and Sohn, K. 2024. Improving visual recognition with hyperbolical visual hierarchy mapping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 17364–17374.

Lou, A.; Katsman, I.; Jiang, Q.; Belongie, S.; Lim, S.-N.; and De Sa, C. 2020. Diferentiating through the fréchet mean. In International conference on machine learning, 6393–6403. PMLR.

Narasimha, A.; Vasavi, B.; and Kumar, H. M. 2013. Significance of nuclear morphometry in benign and malignant breast aspirates. International Journal of Applied and Basic Medical Research, 3(1): 22–26.

Nickel, M.; and Kiela, D. 2017. Poincaré Embeddings for Learning Hierarchical Representations. In Guyon, I.; Luxburg, U. V.; Bengio, S.; Wallach, H.; Fergus, R.; Vishwanathan, S.; and Garnett, R., eds., Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Nickel, M.; and Kiela, D. 2018. Learning continuous hierarchies in the lorentz model of hyperbolic geometry. In International conference on machine learning, 3779–3788. PMLR.

Nock, R.; Amid, E.; Nielsen, F.; Soen, A.; and Warmuth, M. K. 2024. Hyperbolic embeddings of supervised models. Advances in Neural Information Processing Systems, 37: 140865–140902.

Pal, A.; van Spengler, M.; Guido Maria, D.; di Melendugno, A.; Flaborea, A.; Galasso, F.; and Mettes, P. 2025. Compositional entailment learning for hyperbolic vision-language models (2024).

Pandian, D. J. N.; Ramdas, A.; and Ambroise, M. M. 2021. Image analysis-assisted nuclear morphometric study of benign and malignant breast aspirates. Journal of Microscopy and Ultrastructure, 9(3): 114–118.

Salimi-Badr, A. 2024. UNFIS: A novel neuro-fuzzy inference system with unstructured fuzzy rules. Neurocomputing, 579: 127437.

Sinha, A.; Zeng, S.; Yamada, M.; and Zhao, H. 2024. Learning structured representations with hyperbolic embeddings. Advances in Neural Information Processing Systems, 37: 91220–91259.

Su, Z.; Shen, J.; Zhou, Q.; and Yong, B. 2025. SL-ANFIS-LSTM: A Structure Learnable Fuzzy Neural Network for Ultra-Short-Term PV Power Forecasting. IEEE Transactions on Fuzzy Systems.

Van Spengler, M.; and Mettes, P. 2025. Low-distortion and gpu-compatible tree embeddings in hyperbolic space. arXiv preprint arXiv:2502.17130.

Vesović, M.; Jovanović, R.; and Zarić, V. 2024. Hybrid GA-ANFIS and PSO-ANFIS techniques for nonlinear DC motor system modeling. Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science, 09544062251350800.

Weijler, L.; Koch, S.; Poiesi, F.; Ropinski, T.; and Hermosilla, P. 2026. Openhype: Hyperbolic embeddings for hierarchical open-vocabulary radiance fields. Advances in Neural Infor mation Processing Systems, 38: 119039–119068.

Xue, G.; Chang, Q.; Wang, J.; Zhang, K.; and Pal, N. R. 2022. An adaptive neuro-fuzzy system with integrated feature selection and rule extraction for high-dimensional classification problems. IEEE Transactions on Fuzzy Systems, 31(7): 2167–2181.

Yong, B.; Pei, H.; Shen, J.; Li, H.; Zhou, Q.; and Su, Z. 2026. KANFIS A Neuro-Symbolic Framework for Interpretable and Uncertainty-Aware Learning. arXiv preprint arXiv:2602.03034.

Zand, J. P.; Katebi, J.; and Yaghmaei-Sabegh, S. 2024. A hybrid clustering-based type-2 adaptive neuro-fuzzy forecasting model for smart control systems. Expert Systems with Applications, 239: 122445.

Zhang, D.; and Chen, T. 2024. Scikit-ANFIS: A Scikit-Learn Compatible Python Implementation for Adaptive Neuro-Fuzzy Inference System. Int. J. Fuzzy Syst., 26(6): 2039– 2057.

Zhou, W. 2026. The Natural Geometry of Code: Hyperbolic Representation Learning for Program Reasoning. In International Conference on Learning Representations.

## Appendix

This supplement provides additional geometric definitions, model variants, optimization details, theoretical analysis, and implementation settings for HyperANFIS.

## A Hyperbolic Geometry

We define the Lorentz and Poincaré operations under a shared tangent-space parameterization. Throughout, $c > 0$ denotes the curvature magnitude, and the sectional curvature is −c. Input scaling, projection, and finite-precision safeguards are specified in Appendix .

## A.1 Tangent-Space Parameterization

Let $\widehat { \mathbf { x } } _ { i } \in \mathbb { R } ^ { D }$ denote a standardized input and let $\mathbf { a } _ { r } \in \mathbb { R } ^ { D }$ denote the trainable tangent-space center of rule r. Hyper-ANFIS parameterizes samples and rule centers as

$$
\begin{array} { r } { \mathbf { v } _ { i } = \mathrm { c l i p } _ { \tau } ( s \widehat { \mathbf { x } } _ { i } ) , \qquad \mathbf { \overline { { a } } } _ { r } = \mathrm { c l i p } _ { \tau } ( \mathbf { a } _ { r } ) , } \end{array}\tag{20}
$$

where $s > 0$ is the input scale and $\tau > 0$ is the maximum tangent norm. The clipping operator is

$$
\begin{array} { r } { \mathrm { c l i p } _ { \tau } ( \mathbf { v } ) = \left\{ \begin{array} { l l } { \mathbf { 0 } , } & { \| \mathbf { v } \| = 0 , } \\ { \mathbf { v } \operatorname* { m i n } \left( 1 , \frac { \tau } { \| \mathbf { v } \| } \right) , } & { \| \mathbf { v } \| > 0 . } \end{array} \right. } \end{array}\tag{21}
$$

Radial clipping preserves the direction of each nonzero tangent vector and bounds the corresponding geodesic radius. The definition at zero gives the continuous extension. The data-dependent scale s is specified in Appendix .

## A.2 Lorentz Model

The D-dimensional Lorentz model is the upper sheet of the hyperboloid

$$
\mathcal { H } _ { c } ^ { D } = \left\{ \mathbf { z } \in \mathbb { R } ^ { D + 1 } : \left. \mathbf { z } , \mathbf { z } \right. _ { \mathrm { L } } = - c ^ { - 1 } , z _ { 0 } > 0 \right\} ,\tag{22}
$$

with Lorentz inner product

$$
\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathrm { L } } = - x _ { 0 } y _ { 0 } + \sum _ { j = 1 } ^ { D } x _ { j } y _ { j } .\tag{23}
$$

Its tangent space at $\mathbf { x } \in \mathcal { H } _ { c } ^ { D }$ is

$$
\begin{array} { r } { T _ { \mathbf { x } } \mathcal { H } _ { c } ^ { D } = \left\{ \mathbf { u } \in \mathbb { R } ^ { D + 1 } : \langle \mathbf { x } , \mathbf { u } \rangle _ { \mathrm { L } } = 0 \right\} . } \end{array}\tag{24}
$$

For $\mathbf { u } \in T _ { \mathbf { x } } \mathcal { H } _ { c } ^ { D }$ , define $\| \mathbf { u } \| _ { \mathrm { L } } = \sqrt { \langle \mathbf { u } , \mathbf { u } \rangle _ { \mathrm { L } } }$ . The exponential map is

$$
\begin{array} { r l } & { \exp _ { \mathbf { x } } ^ { \mathrm { L } } ( \mathbf { u } ) = \cosh \left( \sqrt { c } \| \mathbf { u } \| _ { \mathrm { L } } \right) \mathbf { x } } \\ & { \qquad + \frac { \sinh \left( \sqrt { c } \| \mathbf { u } \| _ { \mathrm { L } } \right) } { \sqrt { c } \| \mathbf { u } \| _ { \mathrm { L } } } \mathbf { u } . } \end{array}\tag{25}
$$

At the origin $\mathbf { o } _ { \mathrm { L } } = ( c ^ { - 1 / 2 } , 0 , \dots , 0 )$ , a shared tangent $\mathbf { v } \in \mathbb { R } ^ { D }$ is identified with $( 0 , \mathbf { v } ) \in T _ { \mathbf { o } _ { \mathrm { L } } } \mathcal { H } _ { c } ^ { D }$ . With $\varrho ( \mathbf { v } ) =$ ${ \sqrt { c } } \| \mathbf { v } \|$ , Equation (25) becomes

$$
\begin{array} { c } { \displaystyle \exp _ { \mathbf { o } _ { \mathrm { L } } } ^ { \mathrm { L } } ( \mathbf { v } ) = \Bigg ( c ^ { - 1 / 2 } \cosh ( \varrho ( \mathbf { v } ) ) , } \\ { \displaystyle \frac { \sinh ( \varrho ( \mathbf { v } ) ) } { \varrho ( \mathbf { v } ) } \mathbf { v } \Bigg ) . } \end{array}\tag{26}
$$

The spatial multiplier is defined by its continuous limit, 1, at the origin. Direct substitution shows that the image satisfies Equation (22).

For x, $\mathbf { y } \in \mathcal { H } _ { c } ^ { D }$ , let

$$
\begin{array} { r } { \alpha ( \mathbf { x } , \mathbf { y } ) = - c \langle \mathbf { x } , \mathbf { y } \rangle _ { \mathrm { L } } \geq 1 . } \end{array}\tag{27}
$$

The geodesic distance and logarithmic map are

$$
d _ { c } ^ { \mathrm { L } } ( { \bf x } , { \bf y } ) = c ^ { - 1 / 2 } \mathrm { a r c o s h } ( \alpha ) ,\tag{28}
$$

$$
\log _ { \mathbf { x } } ^ { \mathrm { L } } ( \mathbf { y } ) = { \frac { \operatorname { a r c o s h } ( \alpha ) } { \sqrt { \alpha ^ { 2 } - 1 } } } ( \mathbf { y } - \alpha \mathbf { x } ) .\tag{29}
$$

The direction in Equation (29) is tangent at x and has squared Lorentz norm $( \alpha ^ { 2 } \mathrm { ~ - ~ } 1 ) / c ;$ consequently, $\parallel \log _ { \mathbf { x } } ^ { \mathrm { L } } ( \mathbf { y } ) \Vert _ { \mathrm { L } } \ =$ $d _ { c } ^ { \mathrm { L } } ( \mathbf { x } , \mathbf { y } )$ , which verifies the scale factor in Equation (29).

For $\dot { \textbf { u } } \in \ T _ { \mathbf { x } } \mathcal { H } _ { c } ^ { D }$ , parallel transport along the unique geodesic from x to y is

$$
\begin{array} { r l } { \mathcal { T } _ { \mathbf { x }  \mathbf { y } } ^ { \mathrm { L } } ( \mathbf { u } ) = \mathbf { u } } & { { } } \\ { + \frac { \langle \mathbf { y } , \mathbf { u } \rangle _ { \mathrm { L } } } { c ^ { - 1 } - \langle \mathbf { x } , \mathbf { y } \rangle _ { \mathrm { L } } } ( \mathbf { x } + \mathbf { y } ) . } \end{array}\tag{30}
$$

The denominator is $( 1 + \alpha ) / c > 0$ . Parallel transport preserves the Lorentz norm and maps the vector to $T _ { \mathbf { y } } \mathcal { \hat { H } } _ { c } ^ { D }$ HyperANFIS uses Equation (30) with $\mathbf { y } = \mathbf { o } _ { \mathrm { I } }$ to express all rule-local coordinates in the shared origin chart.

## A.3 Poincaré Ball and Lorentz Isometry

The Poincaré ball of curvature −c is

$$
B _ { c } ^ { D } = \left\{ \mathbf { q } \in \mathbb { R } ^ { D } : c \| \mathbf { q } \| ^ { 2 } < 1 \right\} ,\tag{31}
$$

with conformal metric

$$
g _ { \mathbf { q } } ^ { \mathrm { P } } = \lambda _ { \mathbf { q } } ^ { 2 } g ^ { \mathrm { E } } , \qquad \lambda _ { \mathbf { q } } = \frac { 2 } { 1 - c \| \mathbf { q } \| ^ { 2 } } .\tag{32}
$$

For z, $\mathbf { p } \in B _ { c } ^ { D }$ , define $\Delta _ { c } ( \mathbf { q } ) = 1 - c \Vert \mathbf { q } \Vert ^ { 2 }$ . Their distance is

$$
\begin{array} { r } { d _ { c } ^ { \mathrm { P } } ( \mathbf { z } , \mathbf { p } ) = c ^ { - 1 / 2 } \mathrm { a r c o s h } \Bigg ( 1 } \\ { + \displaystyle \frac { 2 c \| \mathbf { z } - \mathbf { p } \| ^ { 2 } } { \Delta _ { c } ( \mathbf { z } ) \Delta _ { c } ( \mathbf { p } ) } \Bigg ) . } \end{array}\tag{33}
$$

Both hyperbolic models use the same tangent vector at the origin. Under this convention, the Poincaré exponential map is

$$
\begin{array} { c } { \displaystyle \exp _ { \mathbf { o } \mathrm { P } } ^ { \mathrm { P } } ( \mathbf { v } ) = \frac { \operatorname { t a n h } ( \varrho ( \mathbf { v } ) / 2 ) } { \varrho ( \mathbf { v } ) } \mathbf { v } , } \\ { \displaystyle \mathbf { o } _ { \mathrm { P } } = \mathbf { 0 } . } \end{array}\tag{34}
$$

Its diferential at the origin maps v to $\mathbf { v } / 2 .$ . Because $\lambda _ { 0 } = 2 ,$ the Riemannian norm of this derivative equals the Euclidean norm of the shared tangent vector v.

The isometry from the Poincaré ball to the Lorentz model is

$$
\Phi _ { c } ( \mathbf { q } ) = \left( \frac { 1 + c \| \mathbf { q } \| ^ { 2 } } { \sqrt { c } ( 1 - c \| \mathbf { q } \| ^ { 2 } ) } , \frac { 2 \mathbf { q } } { 1 - c \| \mathbf { q } \| ^ { 2 } } \right) ,\tag{35}
$$

with inverse

$$
\Phi _ { c } ^ { - 1 } ( \mathbf { z } ) = \frac { \mathbf { z } _ { \mathrm { s p } } } { \sqrt { c } z _ { 0 } + 1 } .\tag{36}
$$

Using the standard hyperbolic half-angle identities gives

$$
\begin{array} { r } { \Phi _ { c } \left( \exp _ { { \mathbf { o } _ { \mathrm { P } } } } ^ { \mathrm { P } } ( \mathbf { v } ) \right) = \exp _ { { \mathbf { o } _ { \mathrm { L } } } } ^ { \mathrm { L } } ( \mathbf { v } ) . } \end{array}\tag{37}
$$

Direct substitution into the Lorentz inner product yields

$$
\begin{array} { r l } { - c \left. \Phi _ { c } ( \mathbf { z } ) , \Phi _ { c } ( \mathbf { p } ) \right. _ { \mathrm { L } } = 1 } & { } \\ { + \cfrac { 2 c \| \mathbf { z } - \mathbf { p } \| ^ { 2 } } { \Delta _ { c } ( \mathbf { z } ) \Delta _ { c } ( \mathbf { p } ) } . } \end{array}\tag{38}
$$

Equations (28)– (38) imply $d _ { c } ^ { \mathrm { P } } ( \mathbf { z } , \mathbf { p } ) = d _ { c } ^ { \mathrm { L } } ( \Phi _ { c } ( \mathbf { z } ) , \Phi _ { c } ( \mathbf { p } ) )$ Accordingly, Poincaré distances and tangent-space operations are evaluated after mapping points through $\Phi _ { c } .$ Manifold-valued aggregation results are mapped back through $\Phi _ { c } ^ { - 1 }$ . These computations represent the same intrinsic operations in two coordinate systems.

## B Geometry and Membership Variants

HyperANFIS supports one geometry and one membership function per training run. The variants share the rule architecture and parameterization; multiple geometries or membership functions are not combined within a forward pass.

## B.1 Geometry Variants

The Lorentz variant applies Equations (22)– (30) directly and is the default. The Poincaré variant applies Equations (31)– (38); its bounded coordinates represent the same intrinsic distances as the Lorentz model.

The Euclidean variant preserves the rule architecture but replaces the manifold operations by

$$
\begin{array} { r l r } & { } & { \exp _ { \mathbf { o } _ { \mathbf { E } } } ^ { \mathbf { E } } ( \mathbf { v } ) = \mathbf { v } , } \\ & { } & { d ^ { \mathbf { E } } ( \mathbf { x } , \mathbf { y } ) = \| \mathbf { x } - \mathbf { y } \| , } \\ & { } & { \mathbf { e } _ { i r } ^ { \mathbf { E } } = \mathbf { z } _ { i } ^ { \mathbf { E } } - \mathbf { p } _ { r } ^ { \mathbf { E } } , } \\ & { } & { \mathbf { m } _ { i } ^ { \mathbf { E } } = \displaystyle \sum _ { r = 1 } ^ { R } \overline { { w } } _ { i r } \mathbf { q } _ { i r } ^ { \mathbf { E } } . } \end{array}\tag{39}
$$

The Euclidean and hyperbolic variants therefore use the same number ofrules, consequent parameterization, and firing normalization. The Euclidean variant is a geometry-matched counterpart of HyperANFIS, not a conventional coordinatewise ANFIS.

## B.2 Membership Functions

For any geometry, define the dimension-normalized intrinsic distance as

$$
\chi _ { i r } = \frac { \delta _ { i r } ^ { \mathcal { M } } } { \sqrt { D } \sigma _ { r } } .\tag{40}
$$

Each rule scale is bounded through the logit parameterization

$$
\sigma _ { r } = \sigma _ { \mathrm { m i n } } + ( \sigma _ { \mathrm { m a x } } - \sigma _ { \mathrm { m i n } } ) \mathrm { s i g m o i d } ( \beta _ { r } ) .\tag{41}
$$

The Gaussian membership function is

$$
\ell _ { i r } ^ { \mathrm { G } } = - \frac { 1 } { 2 } | \chi _ { i r } | ^ { 2 } , \qquad \mu _ { i r } ^ { \mathrm { G } } = \exp ( \ell _ { i r } ^ { \mathrm { G } } ) .\tag{42}
$$

Consequently, the intrinsic radius $\sqrt { D } \sigma _ { r }$ corresponds to $\mu _ { i r } ^ { \mathrm { G } } = \dot { \exp ( - 1 / 2 ) }$

The generalized Bell membership function with fixed $b >$ 0 is

$$
\ell _ { i r } ^ { \mathrm { B } } = - \log \left( 1 + | \chi _ { i r } | ^ { 2 b } \right) , \qquad \mu _ { i r } ^ { \mathrm { B } } = \frac { 1 } { 1 + | \chi _ { i r } | ^ { 2 b } } .\tag{43}
$$

Both functions equal 1 at the rule center. The Gaussian tail decays exponentially in $\chi _ { i r } ^ { 2 }$ , whereas the Bell tail decays polynomially as $| \chi _ { i r } | ^ { - 2 b }$ . The Bell shape parameter b controls the transition between the central region and the tail; the default is $b = 2$

Normalized firing strengths are computed in the log domain:

$$
\begin{array} { l } { \overline { { \boldsymbol { w } } } _ { i r } = \frac { \exp ( \ell _ { i r } ) } { \sum _ { q = 1 } ^ { R } \exp ( \ell _ { i q } ) } } \\ { = \exp \left( \ell _ { i r } - \mathrm { l o g s u m e x p } _ { q } ( \ell _ { i q } ) \right) . } \end{array}\tag{44}
$$

This form does not require explicit exponentiation before normalization and therefore avoids underflow ofintermediate membership values. Numerical safeguards are specified in Appendix .

## C Inference and Optimization

This section specifies rule-local consequent construction, intrinsic aggregation, the training objective, and gradient computation.

## C.1 Rule-Local Consequents

For $\mathcal { M } \in \{ \mathrm { L } , \mathrm { P } \}$ , the local coordinate of sample i relative to rule r is

$$
\begin{array} { r l } & { \boldsymbol { \xi } _ { i r } = \log _ { \mathbf { p } _ { r } ^ { M } } ^ { \mathcal { M } } ( \mathbf { z } _ { i } ^ { \mathcal { M } } ) , } \\ & { \eta _ { i r } = \mathcal { T } _ { \mathbf { p } _ { r } ^ { \mathcal { M } }  \mathbf { o } _ { \mathcal { M } } } ^ { \mathcal { M } } ( \boldsymbol { \xi } _ { i r } ) , } \\ & { \mathbf { e } _ { i r } = [ \eta _ { i r } ] _ { \mathrm { s p } } . } \end{array}\tag{45}
$$

For the Lorentz model, the logarithmic map and parallel transport are given by Equations (29) and (30). For the Poincaré model, both points are mapped by $\Phi _ { c } ,$ the Lorentz operations are applied, and the shared spatial tangent coordinates are retained. For the Euclidean model, ${ \bf e } _ { i r } = { \bf z } _ { i } - { \bf p } _ { r }$

The first-order consequent tangent and its manifold image are

$$
\begin{array} { r l } & { \mathbf { u } _ { i r } = \mathrm { c l i p } _ { \tau } \left( \mathbf { b } _ { r } + \mathbf { W } _ { r } \mathbf { e } _ { i r } \right) , } \\ & { \mathbf { q } _ { i r } ^ { \mathcal { M } } = \mathrm { e x p } _ { \mathbf { o } \mathcal { M } } ^ { \mathcal { M } } ( \mathbf { u } _ { i r } ) , } \end{array}\tag{46}
$$

where $\mathbf { b } _ { r } \in \mathbb { R } ^ { H }$ and $\mathbf { W } _ { r } \in \mathbb { R } ^ { H \times D }$ . Setting ${ \mathbf W } _ { r } = \mathbf { 0 }$ gives the zero-order variant. Each consequent remains rule-specific through its base point p<sub>r</sub> and afine parameters $\left( \mathbf { b } _ { r } , \mathbf { W } _ { r } \right)$ although aggregation uses a shared origin chart.

## C.2 Fréchet Aggregation and Classification

The aggregate for sample i minimizes the weighted Fréchet objective

$$
\begin{array} { c } { \displaystyle \mathcal { F } _ { i } ( \mathbf { m } ) = \frac { 1 } { 2 } \sum _ { r = 1 } ^ { R } { \overline { { w } } _ { i r } d _ { c } ^ { \mathcal { M } } ( \mathbf { m } , \mathbf { q } _ { i r } ^ { \mathcal { M } } ) ^ { 2 } } , } \\ { \displaystyle \mathbf { m } _ { i } ^ { \star } = \underset { \mathbf { m } \in \mathcal { M } } { \mathrm { a r g m i n } } \mathcal { F } _ { i } ( \mathbf { m } ) . } \end{array}\tag{47}
$$

The Riemannian gradient is grad $\mathcal { F } _ { i } ( { \mathbf { m } } )$ = $\begin{array} { r } { - \sum _ { r } \overline { { w } } _ { i r } \log _ { \mathbf { m } } ( \mathbf { q } _ { i r } ) } \end{array}$ The update below is therefore Riemannian gradient descent with step size $\eta _ { \mathrm { K } }$ . The Fréchet minimizer is unique in hyperbolic space, but the finite Karcher iterations yield a diferentiable approximation to the minimizer.

For Lorentz consequents, the normalized ambient initialization is

$$
\begin{array} { c } { { \displaystyle { \bf h } _ { i } = \sum _ { r = 1 } ^ { R } \overline { { { w } } } _ { i r } { \bf q } _ { i r } ^ { \mathrm { L } } } , } \\ { { \displaystyle { \bf m } _ { i } ^ { ( 0 ) } = \frac { { \bf h } _ { i } } { \sqrt { c ( - \langle { \bf h } _ { i } , { \bf h } _ { i } \rangle _ { \mathrm { L } } ) } } } . } \end{array}\tag{48}
$$

This normalization maps $\mathbf { h } _ { i }$ to $\mathcal { H } _ { c } ^ { H }$ . Because the positive firing weights sum to one, $\mathbf { h } _ { i }$ remains in the future-directed timelike cone and the normalization selects the upper sheet. Poincaré consequents are first mapped through $\Phi _ { c } ,$ initialized by Equation (48), and mapped back after refinement. The Euclidean model uses the weighted arithmetic mean in Equation (39).

Starting from $\mathbf { m } _ { i } ^ { ( 0 ) }$ , HyperANFIS computes

$$
\begin{array} { c } { \displaystyle \mathbf { d } _ { i } ^ { ( k ) } = \sum _ { r = 1 } ^ { R } \overline { { w } } _ { i r } \log _ { \mathbf { m } _ { i } ^ { ( k ) } } ^ { \mathcal { M } } ( \mathbf { q } _ { i r } ^ { \mathcal { M } } ) , } \\ { \displaystyle \mathbf { m } _ { i } ^ { ( k + 1 ) } = \exp _ { \mathbf { m } _ { i } ^ { ( k ) } } ^ { \mathcal { M } } ( \eta _ { \mathrm { K } } \mathbf { d } _ { i } ^ { ( k ) } ) . } \end{array}\tag{49}
$$

The iterations terminate when $\| \mathbf { d } _ { i } ^ { ( k ) } \| \leq \epsilon _ { \mathrm { K } }$ for every sample in the batch or after three steps. At a stationary point, $\begin{array} { r } { \sum _ { r } \overline { { w } } _ { i r } \log _ { \mathbf { m } _ { i } } ( \mathbf { q } _ { i r } ) = \mathbf { 0 } } \end{array}$ , which is the first-order condition of Equation (47).

For classification, class tangents $\mathbf { g } _ { k } \in \mathbb { R } ^ { H }$ define

$$
\begin{array} { r } { \mathbf { c } _ { k } ^ { \mathcal { M } } = \exp _ { \mathbf { o } _ { \mathcal { M } } } ^ { \mathcal { M } } \left[ \mathrm { c l i p } _ { \tau } ( \mathbf { g } _ { k } ) \right] , } \\ { s _ { i k } = - d _ { c } ^ { \mathcal { M } } ( \mathbf { m } _ { i } ^ { \mathcal { M } } , \mathbf { c } _ { k } ^ { \mathcal { M } } ) ^ { 2 } . } \end{array}\tag{50}
$$

## C.3 Training Objective

For a mini-batch of size $B ,$ let $P _ { i k } = \mathrm { s o f t m a x } _ { k } ( s _ { i k } )$ and let $y _ { i }$ be the class label. If $n _ { k }$ is the number of training samples in class $k ,$ the class weight is

$$
\gamma _ { k } = \frac { N } { K n _ { k } } ,\tag{51}
$$

where $N$ and $K$ are the number of training samples and classes, respectively. The classification loss is

$$
\mathcal { L } _ { \mathrm { C E } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \gamma _ { y _ { i } } \log P _ { i y _ { i } } .\tag{52}
$$

Class weighting is applied only during training. Validation loss and macro-F1 are unweighted.

Define the batch-average rule usage

$$
\pi _ { r } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \overline { { w } } _ { i r } .\tag{53}
$$

The rule-balance term is the divergence of average rule usage from the uniform distribution:

$$
\mathcal { L } _ { \mathrm { b a l } } = \sum _ { r = 1 } ^ { R } \pi _ { r } \log \left( \frac { \pi _ { r } } { 1 / R } \right) .\tag{54}
$$

This term is minimized by uniform average usage and does not require uniform firing for each sample.

The rule-specialization term is the normalized firing entropy:

$$
\mathcal { L } _ { \mathrm { s p e c } } = - \frac { 1 } { B \log R } \sum _ { i = 1 } ^ { B } \sum _ { r = 1 } ^ { R } \overline { { w } } _ { i r } \log \overline { { w } } _ { i r } .\tag{55}
$$

Minimizing this term encourages concentrated sample-level firing. Its coeficient follows the warm-up schedule

$$
\kappa ( t ) = \operatorname* { m i n } \left( \frac { t } { T _ { \mathrm { w } } } , 1 \right) ,\tag{56}
$$

which delays the full specialization penalty while rule prototypes establish coverage. The balance term discourages globally unused rules, whereas the specialization term discourages difuse sample-level firing.

The prototype-separation term is

$$
\mathcal { L } _ { \mathrm { s e p } } = \frac { 1 } { R ( R - 1 ) } \sum _ { r \neq q }\tag{57}
$$

Only prototype pairs separated by less than $m _ { 0 }$ contribute. The training objective is

$$
\begin{array} { r l } & { \mathcal { L } ( t ) = \mathcal { L } _ { \mathrm { C E } } + \lambda _ { \mathrm { b a l } } \mathcal { L } _ { \mathrm { b a l } } } \\ & { ~ + \kappa ( t ) \lambda _ { \mathrm { s p e c } } \mathcal { L } _ { \mathrm { s p e c } } + \lambda _ { \mathrm { s e p } } \mathcal { L } _ { \mathrm { s e p } } . } \end{array}\tag{58}
$$

## C.4 Optimization

The forward map is diferentiable almost everywhere with respect to the tangent-space centers, scale logits, consequent parameters, and class prototypes. For either membership function, firing normalization satisfies

$$
\frac { \partial \overline { { w } } _ { i r } } { \partial \ell _ { i q } } = \overline { { w } } _ { i r } \left( { \bf 1 } [ r = q ] - \overline { { w } } _ { i q } \right) .\tag{59}
$$

The log-membership derivatives are

$$
\begin{array} { l } { \displaystyle \frac { \partial \ell ^ { \mathrm { G } } } { \partial \chi } = - \chi , } \\ { \displaystyle \frac { \partial \ell ^ { \mathrm { B } } } { \partial \chi } = - \frac { 2 b \mathrm { s i g n } ( \chi ) | \chi | ^ { 2 b - 1 } } { 1 + | \chi | ^ { 2 b } } . } \end{array}\tag{60}
$$

The sigmoid parameterization in Equation (41) keeps each scale trainable within its prescribed interval. Radial clipping is the identity inside the tangent ball and removes the outward radial gradient component outside the ball. Its nondiferentiable boundary has measure zero; automatic diferentiation uses the active branch.

The Karcher iterations in Equation (49) are unrolled in the computation graph. Gradients propagate through the logarithmic maps, weighted tangent sums, exponential updates, and ambient initialization. The discrete nearest-rule assignments are computed before training and are not diferentiated. Early stopping and collapse detection do not modify the forward map.

The numerical clamps in Appendix are piecewise diferentiable. An active clamp has zero gradient toward the invalid region, preventing updates across the Poincaré boundary or the domain boundary of arcosh.

Preprocessing and nearest-rule initialization precede gradient-based training. For each mini-batch, the model computes manifold embeddings, memberships, firing strengths, rule consequents, and the Karcher aggregate; evaluates Equation (58); and applies one optimizer update after backpropagation. Model-selection criteria are given in Appendix .

## D Theoretical Analysis

We establish the uniqueness of intrinsic aggregation and characterize the zero-curvature limit of the hyperbolic distance.

## D.1 Uniqueness of Fréchet Aggregation

Proposition 1. For nonnegative weights $\overline { { w } } _ { i r }$ summing to one, the Fréchet objective in Equation $( 4 7 )$ has a unique minimizer in the Lorentz and Poincaré models.

Proof. Hyperbolic space is complete, simply connected, and has constant negative sectional curvature and is therefore a Hadamard manifold. On a Hadamard manifold, the weighted sum of squared distances is coercive and strictly geodesically convex. Completeness and coercivity give existence, and strict convexity gives uniqueness. The result holds in both coordinate models by the isometry $\Phi _ { c } .$ In Euclidean space, the corresponding objective is a convex quadratic with minimizer $\textstyle \sum _ { r } { \overline { { w } } } _ { i r } \mathbf { q } _ { i r }$ □

HyperANFIS approximates this minimizer with at most three Karcher iterations and terminates earlier when the batch-wide tolerance is satisfied.

## D.2 Euclidean Limit

Proposition 2. For fixed shared tangent vectors v and a,

$$
\operatorname* { l i m } _ { c \downarrow 0 } d _ { c } ^ { \mathrm { L } } \left( \mathbf { \exp } _ { \mathbf { o } _ { \mathrm { L } } } ^ { \mathrm { L } } ( \mathbf { v } ) , \right) = \| \mathbf { v } - \mathbf { a } \| .\tag{61}
$$

Proof. Expanding the hyperbolic sine and cosine terms in Equation (26) gives

$$
\begin{array} { l } { { \displaystyle - c \left. \exp _ { \mathbf { o } _ { \mathrm { L } } } ^ { \mathrm { L } } ( \mathbf { v } ) , \right. _ { \mathrm { _ L } } } } \\ { { \displaystyle ~ = 1 + \frac { c } { 2 } \| \mathbf { v } - \mathbf { a } \| ^ { 2 } + O ( c ^ { 2 } ) . } } \end{array}\tag{62}
$$

Since arcosh $\scriptstyle { \mathrm { 1 } } + u \dotsc u = { \sqrt { 2 u } } + O ( u ^ { 3 / 2 } )$ as $u \downarrow 0 .$ , substitution into Equation (28) proves the claim. The same limit holds in Poincaré coordinates by isometry. □

Under the tangent bounds in Equation (21), the logarithmic maps, transported local coordinates, and Fréchet objective also converge to their Euclidean counterparts. The Euclidean variant is therefore the zero-curvature counterpart of the global radial-rule architecture, not a conventional ANFIS based on products of coordinate-wise memberships.

## E Implementation Details

This section specifies preprocessing, initialization, numerical safeguards, and model selection.

## E.1 Preprocessing and Initialization

All preprocessing statistics are estimated from the training partition. For raw feature $x _ { i j }$

$$
\widehat { x } _ { i j } = \frac { x _ { i j } - \nu _ { j } } { \zeta _ { j } } ,\tag{63}
$$

where $\nu _ { j }$ and $\zeta _ { j }$ are the training mean and standard deviation. Let $\bar { Q } _ { 0 . 9 5 }$ denote the empirical 0.95 quantile and let ρ be the target tangent radius. The fixed input scale is

$$
\begin{array} { r l r } & { \mathcal { R } _ { \mathrm { t r } } = \left\{ \left. \widehat { \mathbf { x } } _ { i } \right. : i \in \mathcal { D } _ { \mathrm { t r } } \right\} , } & \\ & { r _ { 0 . 9 5 } = \operatorname* { m a x } \left\{ Q _ { 0 . 9 5 } ( \mathcal { R } _ { \mathrm { t r } } ) , 1 0 ^ { - 8 } \right\} , } & \\ & { \quad \quad \quad s = \displaystyle \frac { \rho } { r _ { 0 . 9 5 } } . } & \end{array}\tag{64}
$$

This normalization maps the empirical reference radius to $\rho$ before radial clipping; τ remains a separate upper bound.

Initial center tangents are sampled from $\left\{ \mathrm { c l i p } _ { \tau } ( s \widehat { \mathbf { x } } _ { i } ) \right\}$ without replacement when the training set contains at least R samples, and with replacement otherwise. For at most $N _ { \mathrm { i n i t } }$ randomly selected training samples, define

$$
\begin{array} { l } { r ^ { \star } ( i ) = \underset { 1 \leq r \leq R } { \mathrm { a r g m i n } } \delta _ { i r } ^ { \mathcal { M } } , } \\ { d _ { i } ^ { \mathrm { n e a r } } = \displaystyle \frac { 1 } { \sqrt { D } } \underset { 1 \leq r \leq R } { \operatorname* { m i n } } \delta _ { i r } ^ { \mathcal { M } } . } \end{array}\tag{65}
$$

The global scale estimate is the $q _ { \mathrm { i n i t } }$ quantile of $\{ d _ { i } ^ { \mathrm { n e a r } } \}$ . Rule r uses the same quantile of $\{ \delta _ { i r } ^ { \mathcal { M } } / \sqrt { D } : r ^ { \star } ( i ) = r \}$ . Empty or invalid per-rule estimates are replaced by a positive global estimate. After multiplication by $m _ { \mathrm { i n i t } }$ , scales are clipped to $[ \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ]$ and converted to the logit parameterization in Equation (41).

Consequent biases and class-prototype tangents are initialized from $\mathcal { N } ( 0 , 0 . 0 5 ^ { 2 } )$ and $\mathcal { N } ( 0 , 0 . 1 \dot { 5 } ^ { 2 } )$ , respectively. Consequent matrices use Xavier-uniform initialization with gain 0.25. These initialized quantities remain trainable, whereas sample-to-rule assignments are used only during initialization.

## E.2 Numerical Stability

Poincaré points are radially capped at $( 1 - \epsilon _ { \mathrm { { B } } } ) / \sqrt { c }$ with $\epsilon _ { \mathrm { { B } } } ~ = ~ 1 0 ^ { \dot { - } 5 }$ . Lorentz projection preserves the spatial coordinates and recomputes the positive time coordinate as $\sqrt { c ^ { - 1 } + \| \mathbf { z } _ { \mathrm { s p } } \| ^ { 2 } }$ . The radial norm in the Poincaré projection uses the lower bound $\epsilon _ { \mathrm { N } } = 1 0 ^ { - 1 2 }$

Denominators, squared norms, and nonnegative radicands use the lower bound $\epsilon _ { \mathrm { N } }$ . Arguments to arcosh use $1 + \epsilon _ { \mathrm { { A } } }$ where $\epsilon _ { \mathrm { A } } ~ = ~ 1 0 ^ { - 7 }$ . This lower bound is an implementation safeguard; analytically, arcosh(1) = 0. Explicit membership values use a lower bound of $1 0 ^ { - 1 2 }$ for reporting, whereas firing strengths are computed from the unclamped log-memberships in Equation (44). Karcher iterates are reprojected after each step, and Poincaré aggregation is evaluated through the Lorentz isometry.

## E.3 Feature-Space Interpretation

For a raw observation x, antecedent evaluation follows

$$
\begin{array} { r l } & { \mathbf { x }  \widehat { \mathbf { x } } = ( \mathbf { x } - \pmb { \nu } ) \oslash \zeta } \\ & { \quad  \mathbf { v } = \mathrm { c l i p } _ { \tau } ( s \widehat { \mathbf { x } } ) } \\ & { \quad  \mathbf { z } ^ { \mathcal { M } } = \mathrm { e x p } _ { \mathbf { o } _ { \mathcal { M } } } ^ { \mathcal { M } } ( \mathbf { v } ) , } \end{array}\tag{66}
$$

where $\oslash$ denotes element-wise division. Reproducing this mapping requires the fitted preprocessing statistics and the scale s.

A rule center has shared tangent representative $\begin{array} { r l } { \overline { { \mathbf { a } } } _ { r } } & { { } = } \end{array}$ $\mathrm { c l i p } _ { \tau } ( \mathbf { a } _ { r } )$ . Its feature-space representative is

$$
\widetilde { \mathbf { x } } _ { r } = \pmb { \nu } + \pmb { \zeta } \odot \frac { \overline { { \mathbf { a } } } _ { r } } { s } ,\tag{67}
$$

where ⊙ denotes element-wise multiplication. This inverse gives a feature-space representative of the rule center. It is not globally unique, because inputs outside the tangent bound can map to the same clipped point.

## E.4 Model Selection

Validation macro-F1 controls learning-rate scheduling, checkpointing, and early stopping. The test set is evaluated once with the checkpoint that attains the highest validation macro-F1. After a fixed grace period, a diagnostic terminates persistent single-class collapse without modifying Equation (58).
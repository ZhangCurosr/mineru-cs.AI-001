# HYDRA: Hyperbolic Dynamic Representation Architecture for Kolmogorov-Arnold Networks

Zhao Su<sup>1</sup>, Yuxin Xia<sup>1</sup>, Haoran Li<sup>2</sup>, Jun Shen<sup>3</sup>, Qi Zhu<sup>4</sup>, Qingguo Zhou<sup>1</sup>, Binbin Yong<sup>1</sup> <sup>∗</sup>

<sup>1</sup>School of Information Science and Engineering, Lanzhou University

<sup>2</sup>Department of Data Science and AI, Monash University

<sup>3</sup>School of Computing and Information Technology, University of Wollongong

<sup>4</sup>College of Artificial Intelligence, Nanjing University of Aeronautics and Astronautics

## Abstract

Kolmogorov-Arnold Networks (KANs) enhance nonlinear function approximation by replacing scalar weights with learnable univariate functions. However, assigning an independent function to every connection results in substantial parameter redundancy, limiting their scalability and eficiency. To reduce this redundancy, we introduce HYperbolic Dynamic Representation Architecture (HY-DRA), a parameter-eficient hyperbolic extension of KAN that combines spline-based functional learning with representations in the Poincaré ball. HYDRA maps vector-valued inputs into a bounded hyperbolic latent space, performs KAN-style updates in tangent space, and employs a low-rank prototype block to share functional transformations across hidden dimensions. The resulting hyperbolic representations provide a structured radial coordinate for interpretation, while radius control improves training stability by preventing boundary saturation. Extensive experiments across eight benchmark datasets demonstrate that HYDRA consistently achieves competitive or superior predictive performance while improving parameter eficiency and representation interpretability.

## Introduction

Many supervised learning problems with vector-valued inputs require expressive nonlinear transformations while keeping the learned model compact and inspectable. Multilayer perceptrons (MLPs) provide flexible approximation but hide feature transformations inside dense scalar weights. Kolmogorov-Arnold Networks (KANs) replace scalar weights with learnable univariate functions on edges (Liu et al. 2025), improving local functional expressiveness but making a dense hidden-to-hidden block scale quadratically with hidden width and linearly with the number of spline bases.

The parameter-eficiency issue is closely related to how hidden representations are organized. Standard KANs operate in Euclidean spaces, where achieving stronger separation or richer representations often requires additional hidden dimensions or spline structures, increasing the parameter count. By organizing representations in a hyperbolic latent space, the model can encode variations more compactly without substantially enlarging the KAN architecture. Meanwhile, the KAN computation is performed in the tangent space, preserving the simplicity and low-parameter nature of Euclidean spline operations. Hyperbolic representation learning provides a natural alternative because distances and volumes grow diferently from Euclidean spaces (Nickel and Kiela 2017; Ganea, Bécigneul, and Hofmann 2018b). However, naive hyperbolic modeling creates a new problem: near the Poincaré-ball boundary, distances, tangent coordinates, and gradients are amplified, so a model may separate data by drifting outward rather than by learning stable functional structure.

We propose HYperbolic Dynamic Representation Architecture (HYDRA), a radius-constrained low-rank neural network architecture that integrates hyperbolic representation learning with KAN-based function approximation. The key idea is to decouple representation scale from local functional modeling. Specifically, the Poincaré radius serves as a compact coordinate for encoding latent magnitude variations, while KAN splines capture local functional responses in the tangent space. HYDRA first maps normalized inputs into a bounded Poincaré ball, performs KAN-style residual updates in tangent coordinates, compresses these updates through a low-rank prototype bottleneck, and explicitly regulates the latent radius to prevent uncontrolled geometric expansion. Rather than assuming hierarchical structure in the input domain, HYDRA leverages hyperbolic geometry as an eficient and controllable representation space, while preserving the low-parameter and computationally eficient characteristics of KAN computation.

This design leads to the following contributions.

• We propose HYDRA, a hyperbolic functional learning architecture composed of multiple HYDRA blocks, which perform spline-based KAN updates in tangent space while maintaining bounded Poincaré representations.

• HYDRA introduces a low-rank prototype KAN update that reduces the dominant hidden-to-hidden parameter complexity from $O ( d ^ { 2 } K )$ to $O ( d r + r ^ { 2 } K )$

• HYDRA incorporates a radius-control mechanism to constrain hyperbolic representations and mitigate unstable near-boundary efects.

• Experiments on eight datasets demonstrate that HYDRA achieves competitive or superior predictive performance while using fewer parameters than existing approaches.

## Related Work

## Kolmogorov-Arnold Networks

Kolmogorov-Arnold Networks (KANs) have renewed interest in neural architectures where nonlinear transformations are represented as explicit functions rather than implicitly encoded through dense scalar weights. In the original formulation, each edge carries a learnable univariate function, typically parameterized by splines, providing a more interpretable alternative to conventional multilayer perceptrons (Liu et al. 2025). This formulation has motivated subsequent studies on KANs for scientific machine learning and broader neural architectures (Liu et al. 2024; Somvanshi et al. 2025).

Despite their interpretability advantages, dense KAN layers sufer from rapidly increasing functional parameters because each input-output connection requires an independent function. Recent variants address this limitation by modifying functional bases or introducing parameter-sharing mechanisms. Chebyshev KAN replaces spline functions with polynomial bases (Sidharth et al. 2024), Wavelet KAN introduces wavelet-based representations (Bozorgasl and Chen 2024), while FastKAN, radial basis function KAN, and parameterreduced KAN variants further improve eficiency through alternative parameterizations (Li 2024; Ta et al. 2025). These methods reduce functional representation costs while maintaining explicit function learning. However, existing KAN variants mainly focus on edge-function parameterization, leaving the role of latent representation geometry largely unexplored. HYDRA extends this line of research by coupling KAN-based functional updates with structured hyperbolic representations.

## Hyperbolic Representation Learning

Hyperbolic representation learning studies negatively curved spaces for eficient representation of complex structures. Early studies introduced Poincaré and Lorentz embeddings, demonstrating the efectiveness of hyperbolic spaces for hierarchical representation learning (Nickel and Kiela 2017, 2018). These ideas were later extended to neural computation through hyperbolic entailment regions, hyperbolic neural networks, and Riemannian optimization methods (Ganea, Bécigneul, and Hofmann 2018a,b; Bécigneul and Ganea 2019).

Recent research has expanded hyperbolic learning beyond embedding problems toward complete neural architectures, including fully hyperbolic networks, hyperbolic graph models, attention mechanisms, Transformers, supervised representation learning, and residual architectures (Shimizu, Mukuta, and Harada 2021; Chen et al. 2022; Yang et al. 2023, 2024; Nock et al. 2024; Sinha et al. 2024; Li et al. 2024; He, Yang, and Ying 2025). More recently, hyperbolic geometry has also been explored for large-model adaptation and foundation-model learning (Yang et al. 2025; He et al. 2025). However, existing hyperbolic architectures mainly focus on geometric representation learning or neural computation in curved spaces, while the interaction between hyperbolic representations and explicit functional learning remains less explored. HYDRA explores this direction by performing KAN-based functional updates in tangent spaces associated with bounded hyperbolic representations.

## Interpretable Neural Networks

Interpretability research aims to understand how neural models transform inputs into predictions. Generalized additive models and their neural extensions improve transparency by explicitly modeling feature-wise efects and interactions (Yang, Zhang, and Sudjianto 2021; Agarwal et al. 2021; Chang, Caruana, and Goldenberg 2022). Meanwhile, SHAP values provide a widely adopted model-agnostic framework for estimating feature contributions (Lundberg and Lee 2017). KANs further introduce intrinsic interpretability by representing nonlinear transformations as learnable univariate functions, enabling direct analysis of feature-response relationships.

However, functional interpretability alone may not fully characterize models with structured latent spaces. HYDRA complements functional analysis by examining hyperbolic representation dynamics to provide additional insights into model behavior.

## Method

## Problem Setup and Architecture

We consider supervised learning with vector-valued inputs $\mathcal { D } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { \bar { n } }$ , where $\mathbf { x } _ { i } \in \mathbb { R } ^ { p }$ is a normalized input and y<sub>i</sub> is either a continuous target or a binary label. The goal is to learn a predictor $f _ { \boldsymbol { \theta } } ( \mathbf { x } )$ that is accurate, parameter-eficient, and inspectable. Dense KAN layers replace scalar weights with learnable univariate functions, but a hidden-to-hidden KAN block assigns one function to each coordinate pair. This creates a dominant $O ( d ^ { 2 } K )$ spline cost for hidden width d and K spline bases.

HYDRA addresses this cost by combining three operations. First, it represents hidden states in a bounded Poincaré ball, so latent scale can be encoded through radius as well as direction. Second, it performs KAN-style functional learning in the tangent space, where ordinary one-dimensional spline bases remain available. Third, it compresses the tangent update through a learned prototype space before mapping the state back to the hyperbolic manifold. Figure 1 summarizes this computation.

For layer l, HYDRA maps the current hyperbolic state to the tangent space, applies a low-rank spline update, and reconstructs a bounded hyperbolic state:

$$
\begin{array} { r l } & { \mathbf { z } _ { l } = \log _ { 0 } ^ { c } ( \mathbf { h } _ { l } ) , } \\ & { \mathbf { p } _ { l } = W _ { \downarrow } \mathbf { z } _ { l } , } \\ & { \mathbf { s } _ { l } = \Phi _ { l } ( \mathbf { p } _ { l } ) , } \\ & { \tilde { \mathbf { z } } _ { l + 1 } = \mathbf { z } _ { l } + \alpha _ { l } W _ { \uparrow } \mathbf { s } _ { l } , } \\ & { \mathbf { h } _ { l + 1 } = \Pi _ { r _ { l + 1 } } \bigl ( \exp _ { 0 } ^ { c } ( \tilde { \mathbf { z } } _ { l + 1 } ) \bigr ) , } \end{array}\tag{1}
$$

where $\Phi _ { l }$ is a spline block in prototype coordinates, $W _ { \downarrow }$ and $W _ { \uparrow }$ define the bottleneck, $\alpha _ { l }$ is a learned residual scale, and $\Pi _ { r _ { l + 1 } }$ enforces the layer radius budget. This formulation keeps local function learning Euclidean while making the hidden trajectory geometrically constrained and measurable.

![](images/d134aaba02cb3543ba06beacb58d40e3f475c1848ccea5e6623a2ea68527cb21.jpg)  
Figure 1: Overview of HYDRA. The architecture maps vector-valued features into a bounded hyperbolic representation, applies KAN-style spline updates through a low-rank prototype space, and uses radius control to stabilize geometric learning while preserving interpretable spline and radial diagnostics. All blocks share the same architecture.

## Hyperbolic Embedding and Tangent-Space Updates

HYDRA represents hidden states on the Poincaré ball $B _ { c } ^ { d } =$ $\{ \mathbf { h } \in \mathbb { R } ^ { d } : \boldsymbol { c } \| \mathbf { h } \| _ { 2 } ^ { 2 } < 1 \}$ with curvature parameter $c > 0 .$ The normalized input is first mapped to a tangent vector $\mathbf { u } _ { 0 } = g _ { \mathrm { e m b } } ( \mathbf { x } )$ . The initial hidden state is then formed by a bounded exponential map

$$
{ \mathbf { h } } _ { 0 } = \exp _ { 0 } ^ { c } ( \tilde {  { \mathbf { u } } } _ { 0 } ) , \qquad \|  { \mathbf { h } } _ { 0 } \| _ { c } \leq r _ { \mathrm { e m b } } ,\tag{2}
$$

where $r _ { \mathrm { e m b } }$ is the embedding radius budget. The embedding map $g _ { \mathrm { e m b } } \mathrm { i s }$ linear by default and can be replaced by a splinebased input KAN when stronger input response functions are needed.

The exponential and logarithmic maps at the origin are

$$
\begin{array} { l } { { \displaystyle \exp _ { 0 } ^ { c } ( \mathbf { v } ) = \mathrm { t a n h } ( \sqrt { c } \| \mathbf { v } \| _ { 2 } ) \frac { \mathbf { v } } { \sqrt { c } \| \mathbf { v } \| _ { 2 } } , } } \\ { { \displaystyle \log _ { 0 } ^ { c } ( \mathbf { h } ) = \mathrm { a r t a n h } ( \sqrt { c } \| \mathbf { h } \| _ { 2 } ) \frac { \mathbf { h } } { \sqrt { c } \| \mathbf { h } \| _ { 2 } } . } } \end{array}\tag{3}
$$

They are interpreted by continuity at the origin. These maps define a fixed coordinate chart for the KAN update. The model therefore avoids designing coordinate-wise spline functions directly on the manifold, while still returning to a hyperbolic state after each block.

Given $\mathbf { z } _ { l } = \log _ { 0 } ^ { c } ( \mathbf { h } _ { l } )$ , a tangent-space KAN update has the residual form

$$
\mathbf { z } _ { l + 1 } = \mathbf { z } _ { l } + \alpha _ { l } \Delta _ { l } ( \mathbf { z } _ { l } ) .\tag{4}
$$

For a dense KAN layer, the ith coordinate is

$$
\Delta _ { i } ( { \mathbf { z } } ) = \sum _ { j = 1 } ^ { d } \phi _ { i j } ( z _ { j } ) , \qquad \phi _ { i j } ( t ) = \sum _ { m = 1 } ^ { K } a _ { i j m } B _ { m } ( t ) ,\tag{5}
$$

where $B _ { m }$ denotes spline bases and $a _ { i j m }$ denotes learned coeficients. This form is expressive and interpretable, as each edge has an explicit response curve, that leading to every input-output coordinate pair owns a separate spline. HYDRA keeps the same spline principle but moves the costly functional operator into a lower-dimensional prototype space.

## Low-Rank Prototype Functional Learning

The low-rank block starts from the observation that hyperbolic radius can carry part of the latent scale variation. The tangent update need not always use all d hidden directions independently. HYDRA therefore learns an r-dimensional prototype representation, applies the KAN operator there, and lifts the result back to the hidden dimension:

$$
{ \bf p } _ { l } = W _ { \downarrow } { \bf z } _ { l } , \qquad \Delta _ { l } ( { \bf z } _ { l } ) = W _ { \uparrow } \Phi _ { l } ( { \bf p } _ { l } ) ,\tag{6}
$$

where

$$
W _ { \downarrow } \in \mathbb { R } ^ { r \times d } , \qquad W _ { \uparrow } \in \mathbb { R } ^ { d \times r } , \qquad r \ll d .\tag{7}
$$

The spline block $\Phi _ { l }$ is applied before the up-projection, so the update is a nonlinear functional operator whose learned

<table><tr><td></td><td colspan="4">Regression (RMSE ↓)</td><td colspan="4">Classification (Accuracy ↑)</td></tr><tr><td>Model</td><td>CCPP</td><td>Energy</td><td>Parkinsons</td><td>Real Estate</td><td>Heart</td><td>Ionosphere</td><td>Phoneme</td><td>QSAR</td></tr><tr><td>MLP</td><td>3.779 (7.1k)</td><td>1.329 (1.6k)</td><td>5.304 (2.4k)</td><td>6.814(1.8k)</td><td>0.889 (1.4k)</td><td>0.886 (1.8k)</td><td>0.850 (1.8k)</td><td>0.882 (2.4k)</td></tr><tr><td>KAN</td><td>3.668 (7.0k)</td><td>0.741 (1.5k)</td><td>4.424 (2.4k)</td><td>8.016(1.8k)</td><td>0.833 (1.4k)</td><td>0.843 (1.4k)</td><td>0.863 (1.8k)</td><td>0.829 (2.4k)</td></tr><tr><td>HGCN</td><td>4.162 (7.1k)</td><td>2.420 (1.0k)</td><td>7.733 (2.3k)</td><td>7.720 (1.8k)</td><td>0.944 (1.4k)</td><td>0.857 (1.3k)</td><td>0.831 (1.8k)</td><td>0.877 (2.3k)</td></tr><tr><td>HNN</td><td>4.049 (7.2k)</td><td>2.273 (1.0k)</td><td>6.643 (2.4k)</td><td>7.549 (1.8k)</td><td>0.944 (1.4k)</td><td>0.886 (1.8k)</td><td>0.832 (1.8k)</td><td>0.891 (2.4k)</td></tr><tr><td>GAMI-Net</td><td>3.855 (4.8k)</td><td>1.416 (1.1k)</td><td>11.464 (2.4k)</td><td>7.100 (1.8k)</td><td>0.926(1.7k)</td><td>0.943 (1.7k)</td><td>0.836 (1.8k)</td><td>0.858 (1.9k)</td></tr><tr><td>NAM</td><td>4.089 (7.1k)</td><td>3.002 (1.4k)</td><td>9.486 (2.4k)</td><td>8.891 (1.8k)</td><td>0.778 (1.3k)</td><td>0.800 (1.1k)</td><td>0.778 (1.8k)</td><td>0.796 (2.0k)</td></tr><tr><td>NODE-GAM</td><td>4.081 (14.4k)</td><td>1.138 (3.3k)</td><td>7.237 (8.1k)</td><td>7.251 (2.7k)</td><td>0.870 (2.6k)</td><td>0.900 (3.7k)</td><td>0.821 (6.3k)</td><td>0.853 (7.0k)</td></tr><tr><td>FastKAN</td><td>4.303 (6.2k)</td><td>2.077 (1.4k)</td><td>6.927 (2.8k)</td><td>8.394 (2.0k)</td><td>0.870 (2.3k)</td><td>0.857 (1.3k)</td><td>0.865 (1.2k)</td><td>0.839 (1.6k)</td></tr><tr><td>ChebyKAN</td><td>3.842 (9.5k)</td><td>1.023 (1.7k)</td><td>4.078 (2.1k)</td><td>7.487 (1.2k)</td><td>0.889 (1.5k)</td><td>0.857 (1.2k)</td><td>0.885 (0.9k)</td><td>0.834 (1.9k)</td></tr><tr><td>Wav-KAN</td><td>3.688 (8.7k)</td><td>2.047 (1.9k)</td><td>4.533 (1.8k)</td><td>9.053 (1.2k)</td><td>0.778 (1.4k)</td><td>0.871 (1.2k)</td><td>0.880 (1.1k)</td><td>0.863 (2.0k)</td></tr><tr><td>HYDRA (Ours)</td><td>3.604 (4.8k)</td><td>0.706 (1.1k)</td><td>3.534 (1.4k)</td><td>6.769 (1.0k)</td><td>0.944 (1.2k)</td><td>0.971 (0.9k)</td><td>0.885 (0.9k)</td><td>0.900 (1.5k)</td></tr></table>

Table 1: Primary metric and trainable parameters (in parentheses) on eight benchmarks. Best and second-best predictive metrics are bold and underlined; within parentheses, the smallest and second-smallest parameter counts are bold and underlined. For detailed specifications and configuration, refer to Appendix B.

spline interactions are shared through the prototype coordinates.

A full hidden-to-hidden KAN block contains approximately

$$
P _ { \mathrm { f u l l } } \approx d ^ { 2 } ( K + 1 )\tag{8}
$$

parameters, whereas the prototype block uses

$$
P _ { \mathrm { l r } } = 2 d r + r ^ { 2 } ( K + 1 ) .\tag{9}
$$

The approximate compression ratio is

$$
{ \frac { P _ { \mathrm { l r } } } { P _ { \mathrm { f u l l } } } } \approx { \frac { 2 } { K + 1 } } { \frac { r } { d } } + \left( { \frac { r } { d } } \right) ^ { 2 } .\tag{10}
$$

The low-rank prototype design reduces the functional parameter cost from $O ( d ^ { 2 } { \dot { K } } ) { \mathrm { t o } } { \dot { O } } ( d r + r ^ { 2 } K )$ , where the compression ratio is mainly determined by the rank ratio $r / d .$ Thus the dominant spline cost depends on r rather than d. A small rank is useful when the tangent update has low efective dimension, while a larger rank can be selected when the task requires more independent interactions.

## Radius Control Mechanism

Hyperbolic representations can become unstable near the boundary of the Poincaré ball. In that region, distances and tangent coordinates are amplified, so an unconstrained model may reduce the loss by pushing samples outward instead of learning smoother tangent-space functions. HYDRA controls this behavior with two mechanisms. The hard projection $\Pi _ { r _ { l } }$ keeps every layer within a prescribed radius. A soft penalty discourages unnecessary outward movement before the projection becomes active:

$$
\mathcal { L } _ { \mathrm { r a d } } = \frac { 1 } { L + 1 } \sum _ { l = 0 } ^ { L } \left[ \operatorname* { m a x } \left( 0 , \rho ( \mathbf { h } _ { l } ) - r _ { \mathrm { a l l o w } , l } \right) \right] ^ { q } .\tag{11}
$$

Here $\rho ( { \bf h } _ { l } )$ is the normalized radial coordinate, $r _ { \mathrm { a l l o w } , l }$ is the soft threshold, and q controls the penalty shape. In practice, $r _ { \mathrm { a l l o w } , l } = \tau r _ { l }$ with $0 < \tau \leq 1$ , where $r _ { l }$ is the hard radius budget. Projection and penalty are complementary: projection prevents boundary violations, while the penalty changes the training direction before the representation reaches the high-amplification region.

After the final HYDRA block, prediction is made from the tangent coordinate

$$
\begin{array} { r } { { \bf z } _ { L } = \log _ { 0 } ^ { c } ( { \bf h } _ { L } ) , \qquad \hat { y } = g _ { \mathrm { o u t } } ( { \bf z } _ { L } ) . } \end{array}\tag{12}
$$

The readout $g _ { \mathrm { o u t } }$ is linear by default. For regression, HYDRA minimizes mean squared error on normalized targets. For binary classification, it minimizes binary cross-entropy with logits. The complete objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { s u p } } + \lambda _ { \mathrm { r a d } } \mathcal { L } _ { \mathrm { r a d } } + \lambda _ { \mathrm { s p } } \mathcal { L } _ { \mathrm { s p } } , } \end{array}\tag{13}
$$

where $\mathcal { L } _ { \mathrm { s u p } }$ is the supervised loss and $\mathcal { L } _ { \mathrm { s p } }$ combines sparsity and smoothness regularization on the spline coeficients. This objective makes HYDRA a radius-constrained, lowrank functional model. Appendix A gives the universal approximation argument and additional bounds for the radiuscontrolled representation.

## Experiments and Results

## Experimental Setup

We evaluate HYDRA on eight widely used tabular benchmark datasets collected from the OpenML platform (Vanschoren et al. 2013): CCPP, Energy Heating, Parkinsons Telemonitoring, Real Estate Valuation, Heart Statlog, Ionosphere, Phoneme, and QSAR Biodegradation. These datasets cover both regression and binary classification tasks and exhibit diverse characteristics in terms of sample size, feature dimensionality, and underlying data distributions. Such diversity provides a representative test bed for evaluating the expressive capacity, parameter eficiency, and generalization ability of HYDRA on structured tabular data. The first four datasets are formulated as regression problems, whereas the remaining four are treated as binary classification tasks.

Regression performance is evaluated using RMSE, whereas classification performance is measured using accuracy. Additional metrics are reported in Appendix B. For all experiments, the dataset is randomly split into training, validation, and test sets with ratios of 80%, 10%, and 10%, respectively, using a fixed random seed of 42.

Implementation details All experiments are conducted on a single NVIDIA L20 GPU, and all models use the same preprocessing pipeline, optimizer configuration, and evaluation protocol. Regression models predict a single scalar and are evaluated on the original target scale, while classification models output a single logit that is converted into a probability for evaluation. Beyond the task-specific loss, HYDRA introduces only two additional regularization terms: a spline regularizer that encourages smooth and compact spline functions, and a radius regularizer that constrains hidden representations within a layer-wise radius budget. Therefore, the observed performance diferences mainly reflect architectural improvements rather than variations in optimization strategies, training procedures, or computational resources.

## Evaluation Criteria

We select models by the primary metric and report trainable parameter count to measure compactness. HYDRA hyperparameters include hidden width, prototype rank r, spline resolution, learning rate, weight decay, radius budget, and regularization strength. The rank is treated as a compression knob: we choose the smallest rank that preserves the main metric when possible. Parameter-free geometric operations, including exponential maps, logarithmic maps, and radius projection, are not counted as trainable capacity. The parameter-eficiency claim is evaluated primarily against Euclidean KAN and MLP because they share the same dense functional or dense hidden-state modeling role; the remaining baselines are included to contextualize predictive accuracy and interpretability-oriented alternatives.

The selection rule separates predictive quality from compactness. First, we search for HYDRA configurations that are competitive on the primary metric. Second, among configurations with similar primary performance, we prefer the one with fewer trainable parameters. This avoids two misleading extremes: a large HYDRA variant that wins mainly by capacity, and a very small HYDRA variant whose parameter count is attractive but whose accuracy is no longer competitive. The reported configuration is therefore a parameter-performance trade-of rather than the largest model found during tuning. KAN and MLP are the closest Euclidean counterparts for this comparison because they represent, respectively, dense functional edges and dense hidden-state transformations.

<table><tr><td>Dataset</td><td>Task r</td><td>Params F/L Ratio</td><td>Primary F/L</td><td></td></tr><tr><td>CCPP</td><td>Reg. 3</td><td>14,536/1,880</td><td>0.13</td><td rowspan="4">3.947/3.919 2.067/1.959</td></tr><tr><td>Energy Heating</td><td>Reg. 3</td><td>1,036/690 1,211/485</td><td>0.67 0.40</td></tr><tr><td>Parkinsons TM Real Estate</td><td>Reg. 3 Reg. 6</td><td>1,014/1,000</td><td>6.342/6.037 0.99 7.273/6.769</td></tr><tr><td>Heart Statlog Ionosphere</td><td>Cls. 1</td><td>12,081/1,091</td><td>0.09 0.889/0.926</td></tr><tr><td>Phoneme</td><td>Cls. 7 4</td><td>932/884 1,787/441</td><td>0.95 0.25</td><td>0.871/0.943 0.836/0.840</td></tr><tr><td></td><td>Cls.</td><td></td><td></td><td></td></tr><tr><td>QSAR Biodeg.</td><td>Cls. 1</td><td>5,234/1,440</td><td>0.28</td><td>0.877/0.882</td></tr></table>

Table 2: Low-rank prototype ablation. F and L denote fullrank and selected low-rank HYDRA. The primary metric is RMSE for regression and Accuracy for classification.

(a) Low-rank point under full budget  
![](images/b941ee0a3fe081c8df89c7adcf90fd1760097cfe3ecc73ce90548f5be0ea7495.jpg)

(b) Compression-performance map  
![](images/e08aae02dfadbec111414ab5206dee958779ee06806d3dd13d20eaac3a40a369.jpg)  
Figure 2: Low-rank ablation visualization. Selected ranks preserve the primary metric while reducing parameters relative to full-rank HYDRA. In panel (b), numbers index the datasets after sorting by increasing parameter ratio.

## Main Benchmark Results

Table 1 shows that HYDRA achieves the strongest or tiedstrongest primary metric on all eight datasets. On the regression tasks, HYDRA improves over KAN and MLP on CCPP, Energy Heating, Parkinsons Telemonitoring, and Real Estate Valuation while using fewer parameters than both Euclidean counterparts. On the classification tasks, HYDRA matches the best accuracy on Heart Statlog and achieves the highest accuracy on Ionosphere, Phoneme, and QSAR Biodegradation. For example, on the Parkinsons Telemonitoring dataset, HYDRA reduces RMSE from 4.424 of KAN to 3.534 while decreasing trainable parameters from 2.4k to 1.4k, corresponding to a 20.1% performance improvement with 41.7% fewer parameters. Compared with MLP, HYDRA further reduces RMSE by 33.4% under the same parameter reduction. These results support the central claim that HYDRA is not merely accurate, but accurate under a smaller trainable budget.

The benchmark mixes regression and classification because the two settings stress diferent aspects of the model. Regression datasets such as CCPP and Real Estate require smooth response surfaces without allocating a large number of spline edges. Classification datasets such as Ionosphere, Phoneme, and QSAR require separation while discouraging uncontrolled movement toward the Poincaré boundary. HYDRA performs well in both regimes, suggesting that the hyperbolic coordinate is not only acting as a classifier margin and not only as a regression smoother. Instead, it provides an internal scale coordinate that can be coupled with local spline response across objectives.

## Parameter Eficiency

The hyperbolic maps themselves are parameter-free under fixed curvature, so HYDRA’s parameter savings must come from smaller hidden widths or lower prototype ranks. Compared with KAN and MLP, HYDRA uses fewer parameters on every dataset. This matches the low-rank analysis: if the radial coordinate absorbs part of the scale-like variation, fewer prototype directions are needed for the KAN update. The ablation below tests this interpretation against full-rank HYDRA references.

It is useful to distinguish three sources of trainable capacity. The first is the input embedding, which maps raw features into the hidden representation. The second is the hiddento-hidden functional update. This is the dominant term for dense KAN-style models because each hidden input-output pair can own a separate spline. The third is the output readout, which is small for the widths used here. HYDRA targets the second term: the down projection, prototype spline block, and up projection replace a dense functional map with a compact bottleneck. Thus the parameter reduction is architectural rather than a post-hoc pruning efect. When the tangent update has low efective dimension, a small rank preserves the relevant directions; when many independent interactions are needed, the selected rank grows and the savings weaken.

## Ablation Studies

The low-rank ablation asks whether prototype compression preserves performance relative to a full-rank HYDRA block. The radius ablation asks whether the hyperbolic radius budget improves optimization beyond numerical safeguarding. Tables 2-3 and Figures 2-3 summarize the primary results; detailed metrics are in Appendix B.

The selected low-rank models use a mean of 46.8% and a median of 33.8% of the corresponding full-rank HYDRA parameters. As visualized in Figure 2, compression is strongest on Heart Statlog, CCPP, Phoneme, and QSAR, while Real

<table><tr><td>Dataset</td><td>Task Primary C/U Mean radius C/U</td><td></td><td>∆</td></tr><tr><td>CCPP</td><td>Reg.</td><td>3.624/3.690</td><td>0.450/0.646 +0.066</td></tr><tr><td>Energy Heating</td><td>Reg.</td><td>1.193/1.384</td><td>0.659/0.989 +0.191</td></tr><tr><td>Parkinsons TM Real Estate</td><td>Reg.</td><td>5.723/5.883 6.769/6.884</td><td>0.503/0.974 +0.160 0.634/0.979 +0.115</td></tr><tr><td></td><td>Reg.</td><td></td><td></td></tr><tr><td>Heart Statlog</td><td>Cls.</td><td>0.944/0.926 0.943/0.871</td><td>0.596/0.999 +0.018</td></tr><tr><td>Ionosphere</td><td>Cls.</td><td>0.856/0.845</td><td>0.679/0.923 +0.072</td></tr><tr><td>Phoneme</td><td>Cls.</td><td></td><td>0.813/0.984 +0.011</td></tr><tr><td>QSAR Biodeg.</td><td>Cls.</td><td>0.891/0.886</td><td>0.414/0.929 +0.005</td></tr></table>

Table 3: Radius-control ablation. C and U denote radiuscontrolled and unconstrained HYDRA. The primary metric is RMSE for regression and Accuracy for classification.  
![](images/cf66b1d911ccef1a85e19f7521f050a77d57dd9b3b5c4a55ebc119d46cc5c7b7.jpg)

![](images/ac7efd1cc818da7dd350d6059a0a9e02b0ebc358a9976d4f4a40491788c1c367.jpg)  
Figure 3: Radius-control ablation visualization. Constrained HYDRA reduces mean latent radius and improves the primary metric in the selected comparisons.

Estate and Ionosphere require ratios close to one, showing that the benefit is task-dependent.

Across all eight datasets, the constrained model has a smaller mean radius than the unconstrained model, and the primary metric improves in the selected comparisons. Figure 3 shows the same efect visually: radius control pulls representations inward while preserving or improving the primary metric. This matches the theoretical role of the radius budget: it bounds distance amplification and log-map gradients, discouraging near-boundary shortcuts.

Without radius control, the model may exploit the outer region of the Poincaré ball, where even small Euclidean per-

(b)

(a)

![](images/1b12c82d42391cc8fb202f3f4456481dc269b2bb70313c77ed1120729390c595.jpg)  
V: geometric rerouter span=0.29, path=9.84

![](images/8269f5148b403c43a3530406b31b90dc7e84e5630a179b7dbc33c2e1a62f970e.jpg)  
Figure 4: CCPP interpretability case study linking SHAP contribution, radius response, and path geometry.

turbations correspond to large hyperbolic distances. Such movement can help separate samples, but it can also create an unstable shortcut in which the model reduces the loss by pushing representations outward instead of learning a smooth tangent-space functional response. The constrained variant limits this behavior. The simultaneous reduction in mean radius and improvement in the primary metric suggests that radius control changes the learned representation, not only the numerical range of the hidden state.

## Interpretability Analysis

HYDRA’s interpretability comes from the geometry of its hidden representation. For a controlled feature sweep, we record the final hyperbolic radius, trajectory shape, and path length of the hidden state. These quantities describe how the model reorganizes its internal representation as one input variable is changed. At the same time, we have introduced SHAP (SHapley Additive exPlanations) values as a reference to demonstrate the interpretability of the model. SHAP is a game-theoretic interpretability method that quantifies the marginal contribution of each input feature to a model prediction. By decomposing predictions into feature-level positive or negative efects, SHAP provides an intuitive explanation of complex model behavior.

For a sweep $t \mapsto ( \mathbf { x } _ { - j } , t )$ with final representation $\gamma _ { j } ( t )$ the path length

$$
\mathcal { L } _ { j } = \int \| \gamma _ { j } ^ { \prime } ( t ) \| _ { \mathcal { B } _ { c } } d t , \quad \| \gamma _ { j } ^ { \prime } ( t ) \| _ { \mathcal { B } _ { c } } = \lambda _ { \gamma _ { j } ( t ) } ^ { c } \| \gamma _ { j } ^ { \prime } ( t ) \| _ { 2 }\tag{14}
$$

weights Euclidean displacement by the local conformal factor. Large radius changes or long paths therefore indicate stronger internal reorganization.

We used the CCPP dataset as a case study to examine whether HYDRA’s latent geometry provides physically meaningful interpretability. The prediction target is net hourly electrical power output. Ambient temperature (AT), ambient pressure (AP), and relative humidity (RH) mainly describe gas-turbine operating conditions, whereas exhaust vacuum (V) reflects the steam-turbine side of the combinedcycle process. As shown in Figure 4, the AT sweep produced the clearest radius-output relation. The trajectory moved from a small-radius, positive-SHAP regime at low AT to a large-radius, negative-SHAP regime at high AT, and the one-feature PDP reduced predicted power by 37.36 MW. This agrees with the physical expectation that hotter intake air lowers air density and reduces the available mass flow through the gas turbine.

The V sweep also showed a negative output direction, but with a longer and more tortuous hyperbolic path. This pattern is consistent with a coupled steam-side condition that changes the internal representation, rather than acting as a simple direct driver. The corresponding AP and RH sweeps, reported in the Appendix, induced weaker geometric responses, consistent with their secondary roles among the operating variables. Overall, these observations support hyperbolic radius and path geometry as HYDRA-specific interpretability diagnostics. The Appendix further repeats the same visualization protocol on additional datasets to assess whether the CCPP pattern generalizes beyond this case.

## Conclusion

We introduced HYDRA as a compact extension of KAN that couples tangent-space spline computation with hyperbolic latent representations. Across eight tabular benchmarks, HY-DRA achieved competitive or superior predictive performance while reducing trainable parameters by 34.9% relative to Euclidean KAN and by 37.1% relative to MLP on average. Ablation studies showed that the low-rank prototype block retained predictive performance, and that radius control reduced near-boundary saturation in the Poincaré ball. The CCPP interpretability case study further showed that hyperbolic radius and latent trajectories can provide HYDRAspecific diagnostic signals that are consistent with physical expectations and SHAP trends. Overall, these results indicate that hyperbolic representation geometry can support parameter-eficient KAN-style function learning while offering an inspectable latent-space view of model behavior.

## References

Agarwal, R.; Melnick, L.; Frosst, N.; Zhang, X.; Lengerich, B.; Caruana, R.; and Hinton, G. 2021. Neural Additive Models: Interpretable Machine Learning with Neural Nets. In Advances in Neural Information Processing Systems, volume 34.

Bécigneul, G.; and Ganea, O.-E. 2019. Riemannian Adaptive Optimization Methods. In International Conference on Learning Representations.

Bozorgasl, Z.; and Chen, H. 2024. Wav-KAN: Wavelet Kolmogorov-Arnold Networks. arXiv:2405.12832.

Chang, C.-H.; Caruana, R.; and Goldenberg, A. 2022. NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning. In International Conference on Learning Representations.

Chen, W.; Han, X.; Lin, Y.; Zhao, H.; Liu, Z.; Li, P.; Sun, M.; and Zhou, J. 2022. Fully Hyperbolic Neural Networks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, 5672–5686.

Ganea, O.-E.; Bécigneul, G.; and Hofmann, T. 2018a. Hyperbolic Entailment Cones for Learning Hierarchical Embeddings. In International Conference on Machine Learning, 1646–1655.

Ganea, O.-E.; Bécigneul, G.; and Hofmann, T. 2018b. Hyperbolic Neural Networks. In Advances in Neural Information Processing Systems, volume 31.

He, N.; Madhu, H.; Bui, N.; Yang, M.; and Ying, R. 2025. Hyperbolic Deep Learning for Foundation Models: A Survey. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 6021–6031.

He, N.; Yang, M.; and Ying, R. 2025. Lorentzian Residual Neural Networks. In Proceedings ofthe ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 436–447.

Li, Y.; Mao, Y.; Yang, Y.; and Zou, D. 2024. Improving Robustness of Hyperbolic Neural Networks by Lipschitz Analysis. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1713–1724.

Li, Z. 2024. Kolmogorov-Arnold Networks are Radial Basis Function Networks. arXiv:2405.06721.

Liu, Z.; Ma, P.; Wang, Y.; Matusik, W.; and Tegmark, M. 2024. KAN 2.0: Kolmogorov-Arnold Networks Meet Science. arXiv:2408.10205.

Liu, Z.; Wang, Y.; Vaidya, S.; Ruehle, F.; Halverson, J.; Soljačić, M.; Hou, T. Y.; and Tegmark, M. 2025. KAN: Kolmogorov-Arnold Networks. In International Conference on Learning Representations.

Lundberg, S.; and Lee, S.-I. 2017. A Unified Approach to Interpreting Model Predictions. In Advances in Neural Information Processing Systems, volume 30.

Nickel, M.; and Kiela, D. 2017. Poincaré Embeddings for Learning Hierarchical Representations. In Advances in Neural Information Processing Systems, volume 30.

Nickel, M.; and Kiela, D. 2018. Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry. In International Conference on Machine Learning, 3779–3788.

Nock, R.; Amid, E.; Nielsen, F.; Soen, A.; and Warmuth, M. K. 2024. Hyperbolic Embeddings of Supervised Models. In Advances in Neural Information Processing Systems, volume 37.

Rudin, W. 1976. Principles of Mathematical Analysis. McGraw-Hill, 3rd edition.

Shimizu, R.; Mukuta, Y.; and Harada, T. 2021. Hyperbolic Neural Networks++. In International Conference on Learning Representations.

Sidharth, S. S.; Keerthana, A. R.; Gokul, R.; and Anas, K. P. 2024. Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Eficient Architecture for Nonlinear Function Approximation. arXiv:2405.07200.

Sinha, A.; Zeng, S.; Yamada, M.; and Zhao, H. 2024. Learning Structured Representations with Hyperbolic Embeddings. In Advances in Neural Information Processing Systems, volume 37.

Somvanshi, S.; Javed, S. A.; Islam, M. M.; Pandit, D.; and Das, S. 2025. A Survey on Kolmogorov-Arnold Network. ACM Computing Surveys, 58(2): 1–35. Article 55.

Ta, H.-T.; Thai, D.-Q.; Tran, A.; Sidorov, G.; and Gelbukh, A. 2025. PRKAN: Parameter-Reduced Kolmogorov-Arnold Networks. arXiv:2501.07032.

Vanschoren, J.; van Rijn, J. N.; Bischl, B.; and Torgo, L. 2013. OpenML: Networked science in machine learning. ACM SIGKDD Explorations Newsletter, 15(2): 49–60.

Yang, M.; B B, R. S.; Feng, A.; Xiong, B.; Liu, J.; King, I.; and Ying, R. 2025. Hyperbolic Fine-Tuning for Large Language Models. In Advances in Neural Information Processing Systems, volume 38.

Yang, M.; Verma, H.; Zhang, D. C.; Liu, J.; King, I.; and Ying, R. 2024. Hypformer: Exploring Eficient Transformer Fully in Hyperbolic Space. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 3770–3781.

Yang, M.; Zhou, M.; Ying, R.; Chen, Y.; and King, I. 2023. Hyperbolic Representation Learning: Revisiting and Advancing. In International Conference on Machine Learning, 39639–39659.

Yang, Z.; Zhang, A.; and Sudjianto, A. 2021. GAMI-Net: An Explainable Neural Network based on Generalized Additive Models with Structured Interactions. Pattern Recognition, 120: 108192.

# Appendix A: Universal Approximation of HYDRA

Setting. Let $K \subset \mathbb { R } ^ { p }$ be compact and $f ^ { \star } \in C ( K , \mathbb { R } ^ { m } )$ . Define

$$
\| f \| _ { \infty , K } : = \operatorname* { s u p } _ { \mathbf { x } \in K } \| f ( \mathbf { x } ) \| _ { 2 } .
$$

For $0 < r _ { \operatorname* { m a x } { } } < 1 ,$ let

$$
\mathcal { B } _ { c , r _ { \operatorname* { m a x } } } ^ { d } = \{ \mathbf { h } \in \mathbb { R } ^ { d } : \sqrt { c } \| \mathbf { h } \| _ { 2 } \leq r _ { \operatorname* { m a x } } \} , \qquad \mathcal { T } _ { c , r _ { \operatorname* { m a x } } } ^ { d } = \log _ { 0 } ^ { c } ( \mathcal { B } _ { c , r _ { \operatorname* { m a x } } } ^ { d } ) .
$$

On these sets,

$$
\exp _ { 0 } ^ { c } : \mathcal { T } _ { c , r _ { \mathrm { m a x } } } ^ { d } \to \mathcal { B } _ { c , r _ { \mathrm { m a x } } } ^ { d } , \qquad \log _ { 0 } ^ { c } : \mathcal { B } _ { c , r _ { \mathrm { m a x } } } ^ { d } \to \mathcal { T } _ { c , r _ { \mathrm { m a x } } } ^ { d }
$$

are continuous inverses.

Proposition. For every $\varepsilon > 0$ , there exist finite d, $L , K _ { s } , r$ and HYDRA parameters $\theta$ such that

$$
\| f _ { \theta } ^ { \mathrm { H Y D R A } } - f ^ { \star } \| _ { \infty , K } < \varepsilon .
$$

Equivalently,

$$
\begin{array} { r } { \overline { { \mathcal { F } _ { \mathrm { H Y D R A } } } } ^ { \parallel \cdot \parallel _ { \infty , K } } = C ( K , \mathbb { R } ^ { m } ) , } \end{array}
$$

provided width, depth, spline resolution, and prototype rank are allowed to increase.

Proof. Let $\varepsilon > 0 .$ By spline approximation and Stone-Weierstrass (Rudin 1976), there exists a finite Euclidean spline/KAN network $g _ { \phi } : K  \mathbb { R } ^ { m }$ such that

$$
\| g _ { \phi } - f ^ { \star } \| _ { \infty , K } < \varepsilon / 3 .\tag{A1}
$$

Write one Euclidean KAN residual block as

$$
\mathbf { z } _ { l + 1 } = \mathbf { z } _ { l } + A _ { l } \Phi _ { l } \bigl ( B _ { l } \mathbf { z } _ { l } \bigr ) ,\tag{A2}
$$

where $\Phi _ { l }$ is a coordinate-wise spline operator. A corresponding HYDRA block is

$$
\begin{array} { r l } & { \mathbf { h } _ { l } = \exp _ { 0 } ^ { c } ( \mathbf { z } _ { l } ) , } \\ & { \tilde { \mathbf { z } } _ { l + 1 } = \log _ { 0 } ^ { c } ( \mathbf { h } _ { l } ) + W _ { \uparrow , l } \Phi _ { l } ( W _ { \downarrow , l } \log _ { 0 } ^ { c } ( \mathbf { h } _ { l } ) ) , } \\ & { \mathbf { h } _ { l + 1 } = \Pi _ { r _ { \operatorname* { m a x } } } \big ( \exp _ { 0 } ^ { c } ( \tilde { \mathbf { z } } _ { l + 1 } ) \big ) . } \end{array}\tag{A3}
$$

Choose $r \geq d$ and set

$$
W _ { \downarrow , l } = \left[ { \begin{array} { l } { I _ { d } } \\ { 0 } \end{array} } \right] , \qquad W _ { \uparrow , l } = \left[ A _ { l } \quad 0 \right] .\tag{A4}
$$

Then

$$
W _ { \uparrow , l } \Phi _ { l } ( W _ { \downarrow , l } \mathbf { z } ) = A _ { l } \Phi _ { l } ( \mathbf { z } ) ,\tag{A5}
$$

so the low-rank prototype block contains the full tangent-space KAN update whenever $r \geq d .$

Scale the Euclidean realization by $\alpha > 0 \colon$

$$
\begin{array} { r } { \mathbf { z } _ { l } ^ { ( \alpha ) } = \alpha \mathbf { z } _ { l } , \qquad \alpha K \subset \mathcal { T } _ { c , r _ { \operatorname* { m a x } } } ^ { d } . } \end{array}\tag{A6}
$$

For $\| \mathbf { z } \| \leq R _ { \alpha }$ with $R _ { \alpha } \to 0 .$

$$
\exp _ { 0 } ^ { c } ( \mathbf { z } ) = \mathbf { z } + O ( \| \mathbf { z } \| ^ { 3 } ) , \qquad \log _ { 0 } ^ { c } ( \mathbf { h } ) = \mathbf { h } + O ( \| \mathbf { h } \| ^ { 3 } ) .\tag{A7}
$$

Hence, for each block,

$$
\underset { \mathbf { x } \in K } { \operatorname* { s u p } } \| \mathcal { H } _ { l } ^ { ( \alpha ) } ( \mathbf { x } ) - \mathcal { G } _ { l } ^ { ( \alpha ) } ( \mathbf { x } ) \| _ { 2 } \leq C _ { l } R _ { \alpha } ^ { 3 } .\tag{A8}
$$

For finite depth $L ,$

$$
\underset { \mathbf { x } \in K } { \operatorname* { s u p } } \Vert H _ { \theta } ^ { ( \alpha ) } ( \mathbf { x } ) - G _ { \phi } ^ { ( \alpha ) } ( \mathbf { x } ) \Vert _ { 2 } \leq \sum _ { l = 1 } ^ { L } C _ { l } R _ { \alpha } ^ { 3 } < \varepsilon / 3 .\tag{A9}
$$

Let

$$
f _ { \boldsymbol { \theta } } ^ { \mathrm { H Y D R A } } ( \mathbf { x } ) = \alpha ^ { - 1 } Q H _ { \boldsymbol { \theta } } ^ { ( \alpha ) } ( \mathbf { x } ) , \qquad g _ { \boldsymbol { \phi } } ( \mathbf { x } ) = \alpha ^ { - 1 } Q G _ { \boldsymbol { \phi } } ^ { ( \alpha ) } ( \mathbf { x } ) ,\tag{A10}
$$

and choose α so that

$$
\| f _ { \theta } ^ { \mathrm { H Y D R A } } - g _ { \phi } \| _ { \infty , K } < \varepsilon / 3 .\tag{A11}
$$

Combining (A1) and (A11),

$$
\begin{array} { r l } & { \| f _ { \theta } ^ { \mathrm { H Y D R A } } - f ^ { \star } \| _ { \infty , K } \leq \| f _ { \theta } ^ { \mathrm { H Y D R A } } - g _ { \phi } \| _ { \infty , K } + \| g _ { \phi } - f ^ { \star } \| _ { \infty , K } } \\ & { \qquad < 2 \varepsilon / 3 < \varepsilon . } \end{array}\tag{A12}
$$

Thus $\mathcal { F } _ { \mathrm { H Y D R A } }$ is dense in $C ( K , \mathbb { R } ^ { m } )$ .

Rank condition.

$$
r \geq d \quad \implies \quad \mathrm { r a n k } \left( W _ { \uparrow } J _ { \Phi } ( W _ { \downarrow } \mathbf { z } ) W _ { \downarrow } \right) \leq d ,\tag{A13}
$$

and the full tangent update can be recovered. Fixed $r <$ d instead imposes

$$
\operatorname { r a n k } ( J _ { \mathrm { H Y D R A } } ( \mathbf { z } ) ) \leq r ,\tag{A14}
$$

so low-rank HYDRA is a parameter-eficient subfamily of the universal class.

## Appendix B: Supplementary Experimental Results

This appendix reports secondary metrics and compact ablation details for the final benchmark datasets. The main paper uses RMSE for regression and Accuracy for classification as primary metrics. Here, regression rows additionally report MAE and R<sup>2</sup>, where smaller MAE and larger R<sup>2</sup> are better. Classification rows report F1 and ROC-AUC, where larger values are better. HYDRA is the proposed model; GAMI denotes GAMI-Net; HGCN and HNN are hyperbolic neural baselines; KAN, MLP, NAM, and NODE denote the Euclidean baseline families. Bold marks the best value in a row and underlining marks the second-best value.

For ablations, r is the selected prototype rank, and Param ratio is the low-rank/full-rank trainable-parameter ratio. F/L denotes full-rank versus selected low-rank HYDRA. C/U denotes radius-controlled versus unconstrained HYDRA. Aux. lists the secondary metrics in the same order as above. Radius P95 is the 95th percentile final radius; lower values indicate less near-boundary saturation.

<table><tr><td>Task</td><td>Dataset</td><td>HYDRA</td><td>GAMI</td><td>HGCN</td><td>HNN</td><td>KAN</td><td>MLP</td><td>NAM</td><td>NODE</td></tr><tr><td colspan="10">Regression: MAE / R2</td></tr><tr><td>Reg.</td><td>CCPP</td><td>2.700/0.955</td><td>2.991/0.949</td><td>3.308/0.940</td><td>3.202/0.944</td><td>2.746/0.954</td><td>2.874/0.951</td><td>3.236/0.942</td><td>3.223/0.943</td></tr><tr><td>Reg.</td><td>Energy</td><td>0.524/0.995</td><td>1.101/0.980</td><td>1.743/0.941</td><td>1.635/0.948</td><td>0.533/0.995</td><td>1.062/0.982</td><td>1.924/0.910</td><td>0.859/0.987</td></tr><tr><td>Reg.</td><td>Parkinsons</td><td>2.479/0.892</td><td>9.563/-0.140</td><td>5.954/0.481</td><td>4.997/0.617</td><td>3.211/0.830</td><td>3.708/0.756</td><td>7.829/0.219</td><td>5.786/0.546</td></tr><tr><td>Reg.</td><td>Real Estate</td><td>4.829/0.748</td><td>4.994/0.722</td><td>5.743/0.672</td><td>5.524/0.686</td><td>5.792/0.646</td><td>4.875/0.744</td><td>7.242/0.564</td><td>5.286/0.710</td></tr><tr><td colspan="10">Classification: F1 / ROC-AUC</td></tr><tr><td>Cls.</td><td>Heart</td><td>0.933/0.970</td><td>0.909/0.949</td><td>0.930/0.944</td><td>0.930/0.954</td><td>0.780/0.924</td><td>0.875/0.967</td><td>0.750/0.867</td><td>0.844/0.926</td></tr><tr><td>Cls.</td><td>Ionosphere</td><td>0.979/0.974</td><td>0.959/0.962</td><td>0.900/0.949</td><td>0.922/0.967</td><td>0.887/0.938</td><td>0.920/0.953</td><td>0.863/0.728</td><td>0.928/0.929</td></tr><tr><td>Cls.</td><td>Phoneme</td><td>0.807/0.943</td><td>0.701/0.911</td><td>0.702/0.903</td><td>0.717/0.902</td><td>0.772/0.924</td><td>0.737/0.913</td><td>0.643/0.808</td><td>0.683/0.893</td></tr><tr><td>Cls.</td><td>QSAR</td><td>0.857/0.931</td><td>0.797/0.905</td><td>0.822/0.923</td><td>0.841/0.919</td><td>0.757/0.891</td><td>0.830/0.927</td><td>0.707/0.846</td><td>0.783/0.922</td></tr></table>

Table 4: Secondary metrics on the final benchmark datasets.

<table><tr><td>Task</td><td>Dataset</td><td>r</td><td>Param ratio</td><td>Primary F/L</td><td>Aux. F/L</td><td>Primary C/U</td><td>Aux. C/U</td><td>Radius P95 C/U</td></tr><tr><td>Reg.</td><td>CCPP</td><td>3</td><td>0.13</td><td>3.947/3.919</td><td>3.072/3.051; 0.946/0.947</td><td>3.624/3.690</td><td>2.712/2.757; 0.955/0.953</td><td>0.468/0.763</td></tr><tr><td>Reg.</td><td>Energy</td><td>3</td><td>0.67</td><td>2.067/1.959</td><td>1.580/1.536; 0.957/0.962</td><td>1.193/1.384</td><td>0.878/1.032; 0.986/0.981</td><td>0.754/0.999</td></tr><tr><td>Reg.</td><td>Parkinsons</td><td>3</td><td>0.40</td><td>6.342/6.037</td><td>4.737/4.430; 0.651/0.684</td><td>5.723/5.883</td><td>4.043/4.284; 0.716/0.700</td><td>0.698/1.000</td></tr><tr><td>Reg.</td><td>Real Estate</td><td>6</td><td>0.99</td><td>7.273/6.769</td><td>4.981/4.829; 0.709/0.748</td><td>6.769/6.884</td><td>4.829/4.922; 0.748/0.739</td><td>0.753/1.000</td></tr><tr><td>Cls.</td><td>Heart</td><td>1</td><td>0.09</td><td>0.889/0.926</td><td>0.875/0.909; 0.964/0.960</td><td>0.944/0.926</td><td>0.930/0.909; 0.970/0.971</td><td>0.665/1.000</td></tr><tr><td>Cls.</td><td>Ionosphere</td><td>7</td><td>0.95</td><td>0.871/0.943</td><td>0.907/0.958; 0.957/0.957</td><td>0.943/0.871</td><td>0.958/0.909; 0.966/0.931</td><td>0.799/1.000</td></tr><tr><td>Cls.</td><td>Phoneme</td><td>4</td><td>0.25</td><td>0.836/0.840</td><td>0.684/0.747; 0.914/0.914</td><td>0.856/0.845</td><td>0.747/0.722; 0.917/0.915</td><td>0.884/1.000</td></tr><tr><td>Cls.</td><td>QSAR</td><td>1</td><td>0.28</td><td>0.877/0.882</td><td>0.824/0.828; 0.923/0.920</td><td>0.891/0.886</td><td>0.844/0.836; 0.928/0.924</td><td>0.681/1.000</td></tr></table>

Table 5: Combined low-rank and radius-control ablations.

## Appendix C: Additional Interpretability Cases

Figure 5 extends the CCPP case study to additional datasets. Each panel links the SHAP contribution of a controlled feature sweep to the final hyperbolic radius reached by HYDRA. These cases are used as output-side references: HYDRA’s own diagnostics are the radius, trajectory, and latent-path changes, while SHAP summarizes the observed prediction contribution.

B  
![](images/95f2451b38fa36fb42dbb275f0a7d639d9322bed312ae057b17a233817b33731.jpg)

![](images/2d565644e0b4309d0bc4fbc9cce54d07f1c3bf7ec91f086b5c9e636e2a365070.jpg)

C  
![](images/aa3175761c3d0154a7936e6d702c9522abf8995b2278f1bbee6ec978472969d8.jpg)

D  
![](images/a2b945e2d709e0645fa25a984759b2bc2332611c72d0753aa7523ac20755f109.jpg)

![](images/a4e8944e43db24cffcbac91557f23a9514566777b92186978cfb130b442827e7.jpg)

![](images/e2a83c9b4b32e6557a10b1115479426672dc0c5bc0c541a7de86463461bd15f2.jpg)

![](images/8cc339cc98456221b8fab53ef5913fb6a83cbc9134d9d97af7147e722a68c748.jpg)

![](images/41dda966161850e7377d011722634749f567dff65324bb51e41b2d7fae9b8999.jpg)

![](images/1983e09c70e59781587be1c4f87715f478c89d8f6c1638d5fc294d7f0cca1912.jpg)

![](images/8b55c445a7b1254854b31807d38670d6074f891f646c2d25a76172b1815e4616.jpg)

![](images/a962acf288befd4adb07971a7de65c9a49c930b66abaf5c7a8db65f177026142.jpg)

![](images/76e5222e8bea4339c357583484f420977deaa46fe2fae8624161f8ed7b18414b.jpg)

![](images/731d165d78dcfced1367a3b4d5a11167ad69dab5b51fc50d16008146d39e57d6.jpg)

![](images/5245c277570a3508ddf8760dfc2f3544cf52e7e57c54027865ac5e5da837f532.jpg)

![](images/3630abeda051493ae914e76c97bda95cd492fdd2a8ff4c68a5eb6e8692a29a0e.jpg)

![](images/28a4f440388194cba4908ace2558fe980a6389b6b4965b92b26f1d970aec0424.jpg)  
Figure 5: Additional interpretability cases beyond CCPP. Each panel traces how sweeping a representative feature changes both final hyperbolic radius and SHAP contribution.

## Appendix D: Training Hyperparameters

This appendix reports the training hyperparameters used for the main comparison in Table 1. All runs use seed 42. Continuous features are normalized using training-set statistics. Regression targets are optimized in normalized space and mapped back to the original scale for reporting. Classification runs optimize binary cross-entropy with logits.

For HYDRA, Ep. is the number of training epochs and B is the mini-batch size. w is hidden width, K is the number of spline knots, and r is the low-rank prototype rank. LR and WD are the optimizer learning rate and weight decay. Drop. is dropout probability. Rad. is reported as radius-loss weight / target-radius ratio. Pos. is the positive-class loss weight; regression rows use 1.00 because no class reweighting is applied.

For compact baseline tables, each cell begins with Ep./B and then reports the architecture. KAN cells use w, K; MLP cells use $w ;$ HGCN and HNN cells use $w , K$ followed by Rad. GAMI-Net cells report main subnet width, interaction subnet width, $n _ { \mathrm { i n t } } .$ , and main-stage/fine-tuning LR. NAM cells report shallow feature-network width, hidden feature-network width, and LR/WD; “none” means no additional hidden feature-network layer. NODE-GAM cells report trees, tree layers, tree depth, column subsampling ratio, and $\mathrm { L R } / L _ { 2 }$

<table><tr><td>Dataset</td><td>Ep.</td><td>B</td><td>w</td><td>K</td><td>r</td><td>LR</td><td>WD</td><td>Drop.</td><td>Rad.</td><td>Pos.</td></tr><tr><td>CCPP</td><td>120</td><td>256</td><td>43</td><td>6</td><td>16</td><td> $8 . 5 \times 1 0 ^ { - 4 }$ </td><td>0</td><td>0</td><td>0/0.88</td><td>1.00</td></tr><tr><td>Energy Heating</td><td>80</td><td>256</td><td>20</td><td>6</td><td>8</td><td> $1 . 5 \times { { 1 0 } ^ { - 3 } }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0</td><td>0/0.90</td><td>1.00</td></tr><tr><td>Parkinsons TM</td><td>220</td><td>256</td><td>13</td><td>4</td><td>12</td><td> $1 . 8 \times 1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0</td><td>0.03/0.88</td><td>1.00</td></tr><tr><td>Real Estate</td><td>140</td><td>128</td><td>28</td><td>6</td><td>6</td><td> $9 . 0 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0</td><td>0/0.90</td><td>1.00</td></tr><tr><td>Heart Statlog</td><td>80</td><td>256</td><td>47</td><td>4</td><td>2</td><td> $7 . 0 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.02</td><td>0/0.90</td><td>1.20</td></tr><tr><td>Ionosphere</td><td>120</td><td>256</td><td>10</td><td>4</td><td>8</td><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0</td><td>0/0.90</td><td>1.00</td></tr><tr><td>Phoneme</td><td>120</td><td>128</td><td>15</td><td>6</td><td>8</td><td> $8 . 0 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.02</td><td>0.03/0.90</td><td>0.95</td></tr><tr><td>QSAR Biodeg.</td><td>140</td><td>256</td><td>27</td><td>6</td><td>2</td><td> $6 . 5 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td>0.015</td><td>0/0.90</td><td>1.00</td></tr></table>

Table 6: Final HYDRA hyperparameters on the eight benchmark datasets.
<table><tr><td>Dataset</td><td>KAN</td><td>MLP</td><td>HGCN</td><td>HNN</td></tr><tr><td>CCPP</td><td>120/256; 51,12</td><td>120/256; 56</td><td>120/256; 79,4; 0/0.90</td><td>120/256; 41,4; 0/0.90</td></tr><tr><td>Energy Heating</td><td>80/256; 19,12</td><td>80/256; 20</td><td>80/256; 25,4; 0/0.90</td><td>80/256; 26,4; 0/0.90</td></tr><tr><td>Parkinsons TM</td><td>100/512; 39,6</td><td>100/512; 28</td><td>100/512; 23,4; 0/0.90</td><td>100/512; 24,4; 0/0.90</td></tr><tr><td>Real Estate</td><td>80/256; 38,6</td><td>80/256; 36</td><td>80/256; 36,4; 0/0.90</td><td>80/256; 37,4; 0/0.90</td></tr><tr><td>Heart Statlog</td><td>80/256; 9,12</td><td>80/256; 82</td><td>80/256; 29,4; 0/0.90</td><td>80/256; 29,4; 0/0.90</td></tr><tr><td>Ionosphere</td><td>80/256; 13,8</td><td>80/256; 47</td><td>80/256; 21,4; 0/0.90</td><td>80/256; 27,4; 0/0.90</td></tr><tr><td>Phoneme</td><td>-1-; 23,8</td><td>-1-;37</td><td>-/-; 37,4; 0/0.90</td><td>-/-; 38,4; 0/0.90</td></tr><tr><td>QSAR Biodeg.</td><td>-1-; 27,4</td><td>-1-; 24</td><td>-/-; 30,4; 0/0.90</td><td>-/-; 31,4; 0/0.90</td></tr></table>

Table 7: Internal neural baseline hyperparameters.
<table><tr><td>Dataset</td><td>GAMI-Net</td><td>NAM</td><td>NODE-GAM</td><td></td></tr><tr><td>CCPP</td><td> $1 0 0 / 2 5 6 ; 1 2 ; 1 4 - 2 6 ; 1 0 ; 1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td><td> $1 4 0 / 2 5 6 ; 2 1 ; 2 0 { - } 6 2 ; 5 \times 1 0 ^ { - 4 } / 0$ </td><td> $1 0 0 / 2 5 6 ; 1 6 0 , 1 , 6 , 0 . 5 ; 1 0 ^ { - 2 } / 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Energy Heating</td><td> $8 0 / 2 5 6 ; 1 2 ; 1 2 - 1 2 ; 4 ; 1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td><td> $1 4 0 / 2 5 6 ; 4 ; 3 2 ; 5 \times 1 0 ^ { - 4 } / 0$ </td><td> $8 0 / 2 5 6 ; 1 4 3 , 1 , 3 , 1 . 0 ; 1 0 ^ { - 2 } / 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Parkinsons TM</td><td> $1 0 0 / 2 5 6 ; 1 2 ; 1 2 - 1 2 ; 8 ; 1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td><td> $1 4 0 / 2 5 6 ; 8 ; 1 2 ; 5 \times 1 0 ^ { - 4 } / 1 0 ^ { - 6 }$ </td><td> $1 0 0 / 2 5 6 ; 6 4 , 1 , 6 , 0 . 5 ; 1 0 ^ { - 2 } / 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Real Estate</td><td> $8 0 / 2 5 6 ; 1 2 ; 1 6 - 8 ; 8 ; 1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td><td> $1 1 6 / 2 5 6 ; 8 ; 1 6 - 8 ; 5 \times 1 0 ^ { - 4 } / 0$ </td><td> $8 0 / 2 5 6 ; 1 2 8 , 1 , 2 , 0 . 5 ; 1 0 ^ { - 2 } / 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Heart Statlog</td><td> $8 0 / 2 5 6 ; 3 2 ; 8 - 8 ; 4 ; 7 \times 1 0 ^ { - 4 } / 7 \times 1 0 ^ { - 5 }$ </td><td> $1 4 0 / 2 5 6 ; 5 ; 1 5 ; 5 \times 1 0 ^ { - 4 } / 0$ </td><td> $8 0 / 2 5 6 ; 2 7 , 2 , 2 , 0 . 5 ; 1 0 ^ { - 2 } / 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Ionosphere</td><td> $8 0 / 2 5 6 ; 1 4 ; 7 ; 7 ; 1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td><td> $1 4 0 / 2 5 6 ; 1 6 ; \mathrm { n o n e } ; 5 \times 1 0 ^ { - 4 } / 0$ </td><td> $8 0 / 2 5 6 ; 4 8 , 1 , 2 , 0 . 5 ; 1 0 ^ { - 2 } / 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Phoneme</td><td> $8 0 / 2 5 6 ; 1 2 ; 1 0 - 1 4 ; 8 ; 1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td><td> $1 4 0 / 2 5 6 ; 1 2 ; 1 6 - 8 ; 5 \times 1 0 ^ { - 4 } / 0$ </td><td> $8 0 / 2 5 6 ; 6 4 , 1 , 6 , 0 . 5 ; 1 0 ^ { - 2 } / 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>QSAR Biodeg.</td><td> $8 0 / 2 5 6 ; 8 ; 8 - 1 8 ; 4 ; 1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td><td> $1 4 0 / 2 5 6 ; 2 4 ; \mathrm { n o n e } ; 5 \times 1 0 ^ { - 4 } / 0$ </td><td>80/256; 64,1,4,0.5; 10−2/10−5</td><td></td></tr></table>

Table 8: External GAM baseline hyperparameters.

## Appendix E: Limitations

This work has several limitations. First, our empirical evaluation focuses on standard tabular benchmarks, leaving the behavior of HYDRA on high-dimensional unstructured data (e.g., images, text, and multimodal inputs) for future investigation. Second, the proposed multi-feature analysis should be interpreted as a post hoc diagnostic of the learned model rather than a causal explanation. While it reveals that output non-additivity can coincide with hyperbolic latent reorganization, it does not establish causal or physical interactions among input variables. Finally, although we compare HYDRA with additional parameter-eficient KAN variants, its prototype rank and radius remain validation-dependent hyperparameters. Developing adaptive strategies for rank selection and radius scheduling could further improve robustness across datasets while reducing the need for manual tuning.